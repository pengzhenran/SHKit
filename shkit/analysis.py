# -*- coding: utf-8 -*-
"""
shkit.analysis
==============

Spherical harmonic **analysis**: arbitrary scattered points or arbitrary
(lat, lon) grids -> spherical harmonic coefficients, in the 4-pi normalised
convention of :mod:`shkit.coeffs`.

Three estimators, all sharing the same integration elements
(:mod:`shkit.weights`):

``method='quadrature'``
    Plain weighted projection

    .. math::  C_{nm} = \\frac{1}{4\\pi}\\sum_i w_i f_i \\bar P_{nm}\\cos m\\lambda

    Exact and cheap when the sampling + weights form a good quadrature rule
    (global, roughly uniform).  Aliased otherwise.

``method='iterative'``
    Quadrature followed by ``niter`` Richardson corrections

    .. math::  x \\leftarrow x + Q\\,(f - A x), \\qquad Q = A^T W/(4\\pi)

    Costs one synthesis + one quadrature per iteration, and converges to the
    weighted least-squares solution.  Measured on 3000 quasi-uniform points,
    L=20: one iteration cut the coefficient error from 7.8e-4 to 1.2e-4 at
    essentially no cost.  With noise present it is neutral.

``method='wlsq' / 'cg'``
    Weighted least squares, ``min ||W^{1/2}(A x - f)||^2``, optionally with
    Tikhonov/Kaula regularisation.  **The summation runs over the sample points
    only**, so this objective is a proxy for the field error only while the
    sampling is a valid quadrature rule for the whole domain (``K ~ I``).  On
    regional data it is not, and an unregularised solve then returns the
    minimum-norm member of a huge near-null space -- a purely algebraic choice
    with no physical content.  Use it for globally-covering but irregularly
    sampled data, always with ``reg='kaula'`` or an explicit ``alpha``.
    ``'wlsq'`` builds the normal equations; ``'cg'`` applies the operator on the
    fly so the memory footprint stays O(n_points) instead of
    O(n_points * ncoef).

``method='projection'``
    The zero-filled global L2 projection (target A of a regional dataset)::

        min over deg<=L of  integral_Omega (f - A x)^2 dOmega,  f = 0 outside

    whose minimiser is ``x = (1/4pi) A^T W f`` with **geometric** weights
    (``sum(w)`` = the covered area, *not* ``4pi``).  Unlike ``'wlsq'`` its
    objective is defined on the whole sphere, so its sample residual *is* the
    truncation leakage and stays at a non-zero floor set by ``L``.  Optional
    ``tau`` applies a truncated spectral correction of ``K`` (the Slepian
    concentration matrix) that repairs the sampling bias only in the directions
    the data can actually see.  See :func:`analysis_projection`.

``method='auto'`` inspects the diagnostics and picks.

Speed
-----
Every estimator's core is the weighted projection

.. math::  \\frac{1}{4\\pi}\\sum_i w_i f_i \\,\\bar P_{nm}(\\sin\\varphi_i)\\cos(m\\lambda_i)

which streams over the order ``m``.  When the samples happen to be a **complete
rectangular grid** on a whole-circle uniform longitude axis with
``nlon > 2*nmax``, the inner longitude sum is replaced by a single ``rfft`` per
latitude row (:mod:`shkit.lonfft`), turning ``O(nSHCS*nlat*nlon*ntime)`` into
``O(nSHCS*nlat*ntime) + O(nlat*ntime*nlon log nlon)``.  This is the
``longitude_fft`` switch on :func:`analysis` and friends: ``'auto'`` (default)
selects it, ``'direct'`` forces the universal sweep and ``'fft'`` demands it.
The decision and its Chinese reason are recorded in ``report.meta`` -- the two
paths agree to floating-point round-off (measured ≤1e-12 relative), so the only
honest reason to care which one ran is speed.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np

from .basis import legendre_columns, design_matrix_full, synthesize
from .coeffs import SHCoeffs, triangle_order
from .diagnostics import (AnalysisReport, gram_deviation, gram_matrix,
                          recommend_lmax)
from .lonfft import fft_path_applicable, plan_longitude
from .weights import FOUR_PI, WeightSet, compute_weights

__all__ = [
    "analysis",
    "analysis_quadrature",
    "analysis_projection",
    "analysis_iterative",
    "analysis_wlsq",
    "SHOperator",
    "kaula_penalty",
    "AnalysisCancelled",
    "lonfft_applicable",
    "LONGITUDE_FFT_MODES",
]

_GRAM_FLOP_BUDGET = 2.0e9        # skip the full Gram check above this

#: Accepted values of every ``longitude_fft`` argument in this module.  The
#: criterion itself lives in :mod:`shkit.lonfft` (shared with the synthesis
#: direction); this is the one place that spells the option names out.
LONGITUDE_FFT_MODES = ("auto", "fft", "direct")

#: Target size of the complex FFT buffer; larger series are processed in time
#: chunks of ``_FFT_BYTES / (nlat * (nlon//2 + 1) * 16)`` epochs.
_FFT_BYTES = 32 * 1024 ** 2


#: Observable call counters.  They exist so the *optimisation* can be asserted
#: rather than believed: batch analysis is only fast because the weights and the
#: Gram diagnostic are computed **once** for the whole series, and that is a
#: property of the code, not of the timing on one machine.
STATS: "Counter" = Counter()


def stats_snapshot() -> dict:
    """Copy of :data:`STATS` (call it before and after, then diff)."""
    return dict(STATS)


def stats_reset() -> None:
    STATS.clear()


def stats_delta(before: dict) -> dict:
    """Per-key difference between now and an earlier :func:`stats_snapshot`."""
    return {k: v - before.get(k, 0) for k, v in STATS.items()
            if v - before.get(k, 0) != 0}


class AnalysisCancelled(RuntimeError):
    """Raised when a ``cancel`` callback asks to abort an analysis.

    The library never cancels on its own; cancellation is cooperative and only
    happens inside the loops that honour ``progress``/``cancel`` (the order-``m``
    quadrature sweep, the normal-equation assembly and the CG iterations).
    """


ProgressFn = Callable[[str, float], None]
CancelFn = Callable[[], bool]


def _tick(progress: Optional[ProgressFn], cancel: Optional[CancelFn],
          message: str, fraction: float) -> None:
    """Report progress and honour a cooperative cancellation request."""
    if cancel is not None and cancel():
        raise AnalysisCancelled(f"cancelled: {message}")
    if progress is not None:
        progress(message, float(min(max(fraction, 0.0), 1.0)))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_2d(f: np.ndarray, n: int):
    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        return f.reshape(n, 1), True
    if f.ndim == 2 and f.shape[0] == n:
        return f, False
    raise ValueError(f"data shape {f.shape} does not match {n} points")


def _to_coeffs(C: np.ndarray, S: np.ndarray, single: bool,
               meta: Optional[dict] = None) -> SHCoeffs:
    if single:
        return SHCoeffs(C[:, :, 0], S[:, :, 0], dict(meta or {}))
    return SHCoeffs(C, S, dict(meta or {}))


class SHOperator:
    """Matrix-free ``A`` / ``A^T`` for scattered points.

    ``A`` maps the flat coefficient vector ``x = [C (all n,m); S (m>=1)]`` of
    length ``(nmax+1)**2`` to the field at the sample points.  Only the current
    Legendre column and the ``m-2`` column are kept alive, so the memory cost is
    ``O(n_points * nmax)`` instead of ``O(n_points * nmax**2)``.
    """

    def __init__(self, lat_deg, lon_deg, nmax: int):
        self.lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
        self.lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
        self.nmax = int(nmax)
        self.npts = self.lat.size
        self.ncoef = (self.nmax + 1) ** 2
        self.m_vec, self.n_vec = triangle_order(self.nmax)
        self.sel_s = self.m_vec >= 1
        # per-order slices into the flat [C(all n,m); S(m>=1)] vector
        self.c_slice = {}
        self.s_slice = {}
        c0 = s0 = 0
        for m in range(self.nmax + 1):
            k = self.nmax - m + 1
            self.c_slice[m] = slice(c0, c0 + k)
            c0 += k
            if m >= 1:
                self.s_slice[m] = slice(s0, s0 + k)
                s0 += k

    # ------------------------------------------------------------ index maps
    @property
    def _off_c(self):
        return 0

    @property
    def _off_s(self):
        return len(self.m_vec)

    def matvec(self, x: np.ndarray, ntime: int = 1) -> np.ndarray:
        """``A @ x`` -> ``(npoints, ntime)``."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        nc = len(self.m_vec)
        xc = x[:nc]
        xs = x[nc:]
        lam = np.deg2rad(self.lon)
        out = np.zeros((self.npts, x.shape[1]))
        for m, block in legendre_columns(self.lat, self.nmax):
            acc = block.T @ xc[self.c_slice[m]]
            if m >= 1:
                acc = acc * np.cos(m * lam)[:, None] + \
                    (block.T @ xs[self.s_slice[m]]) * np.sin(m * lam)[:, None]
            out += acc
        return out

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """``A^T @ y`` -> flat coefficient vector of length ``(nmax+1)**2``."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        nc = len(self.m_vec)
        out = np.zeros((self.ncoef, y.shape[1]))
        lam = np.deg2rad(self.lon)
        for m, block in legendre_columns(self.lat, self.nmax):
            if m == 0:
                out[self.c_slice[0]] = block @ y
            else:
                cm = np.cos(m * lam)[:, None]
                sm = np.sin(m * lam)[:, None]
                out[self.c_slice[m]] = block @ (y * cm)
                sl = self.s_slice[m]
                out[nc + sl.start:nc + sl.stop] = block @ (y * sm)
        return out


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------
def _plan_longitude(lat, lon, nmax: int, mode: str):
    """Kept as the module-local name; the logic lives in :mod:`shkit.lonfft` so
    the analysis (A3) and synthesis (A4) directions share one criterion."""
    return plan_longitude(lat, lon, nmax, mode)


def _quadrature_C(plan, lat, lon, fw, nmax,
                  progress: Optional[ProgressFn] = None,
                  cancel: Optional[CancelFn] = None,
                  label: str = "求积",
                  frac0: float = 0.0, frac1: float = 1.0):
    """Weighted projection ``(1/4pi) sum_i w_i f_i {cos,sin}(m lambda_i) Pbar_nm``.

    ``fw`` is already weighted.  ``plan`` comes from :func:`_plan_longitude` and
    selects the kernels; both produce the same coefficients, so the choice is a
    performance decision only.
    """
    if plan.path == "fft":
        STATS["quadrature_fft"] += 1
        g = plan.grid
        return _quadrature_C_fft(g, g.lat_vec, g.to_grid(fw), nmax,
                                 progress=progress, cancel=cancel, label=label,
                                 frac0=frac0, frac1=frac1)
    STATS["quadrature_direct"] += 1
    return _quadrature_C_direct(lat, lon, fw, nmax,
                                progress=progress, cancel=cancel, label=label,
                                frac0=frac0, frac1=frac1)


def _quadrature_C_direct(lat, lon, fw, nmax,
                         progress: Optional[ProgressFn] = None,
                         cancel: Optional[CancelFn] = None,
                         label: str = "求积",
                         frac0: float = 0.0, frac1: float = 1.0):
    """Weighted projection, streaming over the order ``m`` (universal path)."""
    nlat_rows = nmax + 1
    ntime = fw.shape[1]
    C = np.zeros((nlat_rows, nlat_rows, ntime))
    S = np.zeros((nlat_rows, nlat_rows, ntime))
    lam = np.deg2rad(lon)
    for m, block in legendre_columns(lat, nmax):
        n_idx = np.arange(m, nmax + 1)
        cm = np.cos(m * lam)[:, None]
        C[n_idx, m, :] = block @ (fw * cm) / FOUR_PI
        if m >= 1:
            sm = np.sin(m * lam)[:, None]
            S[n_idx, m, :] = block @ (fw * sm) / FOUR_PI
        if progress is not None or cancel is not None:
            _tick(progress, cancel, f"{label}: 阶 m={m}/{nmax}",
                  frac0 + (frac1 - frac0) * (m + 1) / (nmax + 1))
    return C, S


def _quadrature_C_fft(grid, lat_vec, fw_grid, nmax,
                      progress: Optional[ProgressFn] = None,
                      cancel: Optional[CancelFn] = None,
                      label: str = "求积(FFT)",
                      frac0: float = 0.0, frac1: float = 1.0):
    """Weighted projection with the longitude sum done by one ``rfft`` per row.

    ``fw_grid`` is ``(nlat, nlon, ntime)`` and already weighted.  For
    ``G = rfft(fw_grid, axis=lon)`` and ``phi_m = m * lam0``::

        sum_k g cos(m lam_k) = cos(phi_m) Re G[m] + sin(phi_m) Im G[m]
        sum_k g sin(m lam_k) = sin(phi_m) Re G[m] - cos(phi_m) Im G[m]

    The phase term matters: with ``lam0 == 0`` (the usual ``lon = 0, 1, ...``
    grid) it is the identity, so a missing phase passes every test that only ever
    uses that grid and silently ruins any axis that starts at ``0.5`` or
    ``-179.5``.  See ``tests/validate_lonfft.py`` for the offset-grid regression.

    Cost: ``O(nSHCS * nlat * ntime)`` for the Legendre contraction plus
    ``O(nlat * ntime * nlon log nlon)`` for the transforms, against
    ``O(nSHCS * nlat * nlon * ntime)`` for the direct sweep.
    """
    nlat, nlon, ntime = fw_grid.shape
    nf = nlon // 2 + 1
    C = np.zeros((nmax + 1, nmax + 1, ntime))
    S = np.zeros((nmax + 1, nmax + 1, ntime))
    lam0 = np.deg2rad(float(grid.lam0_deg))
    # The Legendre recurrence is re-run for every time chunk (it costs
    # O(nmax^2 * nlat), i.e. nothing next to the contractions) so that the complex
    # FFT buffer stays bounded no matter how many epochs are being processed.
    chunk = max(1, int(_FFT_BYTES // max(nlat * nf * 16, 1)))
    nchunk = (ntime + chunk - 1) // chunk
    for ci, t0 in enumerate(range(0, ntime, chunk)):
        t1 = min(t0 + chunk, ntime)
        Gh = np.fft.rfft(fw_grid[:, :, t0:t1], axis=1)      # (nlat, nf, ntc)
        for m, block in legendre_columns(lat_vec, nmax):
            n_idx = np.arange(m, nmax + 1)
            re = Gh[:, m, :].real
            if m == 0:
                # S_n0 never exists: sin(0*lambda) == 0.
                C[n_idx, 0, t0:t1] = block @ re / FOUR_PI
            else:
                im = Gh[:, m, :].imag
                phi = m * lam0
                c, s = float(np.cos(phi)), float(np.sin(phi))
                C[n_idx, m, t0:t1] = block @ (c * re + s * im) / FOUR_PI
                S[n_idx, m, t0:t1] = block @ (s * re - c * im) / FOUR_PI
            if progress is not None or cancel is not None:
                done = ci * (nmax + 1) + m + 1
                _tick(progress, cancel, f"{label}: 阶 m={m}/{nmax}",
                      frac0 + (frac1 - frac0) * done / (nchunk * (nmax + 1)))
    return C, S


def lonfft_applicable(lat, lon, nmax: int):
    """Would the longitude FFT path be used for this point set?  ``(ok, reason)``.

    Exposed so a caller (CLI ``--longitude-fft``, GUI, tests) can report the
    decision *before* paying for an analysis, without duplicating the criterion.
    """
    plan = _plan_longitude(lat, lon, nmax, "auto")
    return plan.path == "fft", plan.reason



def _absorb_weight_notes(rep, weights) -> None:
    """把 ``WeightSet.notes`` 搬进报告 —— **不搬就等于没写**。

    ``compute_weights`` 会在规则不适用时留下说明（最典型的是
    ``dh`` 但 ``nlon != 2*nlat``：此时 Driscoll–Healy 不再精确）。这些说明以前
    只存在 ``WeightSet.notes`` 里，没有任何下游读过它；而权重又按 ``global``
    归一化到 :math:`\\sum w = 4\\pi`，于是**覆盖率永远显示 1.0**，
    从数字上完全看不出规则其实不适用。
    """
    for n in list(getattr(weights, "notes", ()) or ()):
        msg = f"积分元（{getattr(weights, 'rule', '?')}）：{n}"
        if msg not in rep.warnings:
            rep.add_warning(msg)


def analysis_quadrature(lat, lon, f, nmax: int,
                        weights: Optional[WeightSet] = None,
                        niter: int = 0,
                        normalise: str = "auto",
                        weights_kw: Optional[dict] = None,
                        progress: Optional[ProgressFn] = None,
                        cancel: Optional[CancelFn] = None,
                        field_unit: str = "unknown",
                        report_fit: bool = True,
                        gram_cache: Optional[dict] = None,
                        longitude_fft: str = "auto",
                        ) -> tuple:
    """Weighted projection (optionally Richardson-corrected).

    Returns ``(SHCoeffs, AnalysisReport)``.

    ``progress(message, fraction)`` is called as the order-``m`` sweep advances
    and ``cancel()`` is polled at the same points; returning True from it raises
    :class:`AnalysisCancelled`.

    ``longitude_fft`` selects the longitude kernel -- ``'auto'`` (default) uses
    the FFT whenever the point set is a complete rectangular grid on a
    whole-circle uniform longitude axis with ``nlon > 2*nmax``, ``'direct'``
    forces the universal sweep and ``'fft'`` requires the fast path (raising if
    it does not apply).  The choice and its reason land in ``report.meta``, so a
    fallback is never silent.
    """
    f2, single = _as_2d(f, len(np.atleast_1d(lat)))
    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    if weights is None:
        STATS["compute_weights"] += 1
        weights = compute_weights(lat, lon, normalise=normalise,
                                  **(weights_kw or {}))
    w = weights.w
    if w.size != lat.size:
        raise ValueError("weight vector length does not match the samples")

    plan = _plan_longitude(lat, lon, nmax, longitude_fft)
    niter = max(int(niter), 0)
    span = 1.0 / (niter + 1)
    fw = w[:, None] * f2
    C, S = _quadrature_C(plan, lat, lon, fw, nmax,
                         progress=progress, cancel=cancel, label="求积",
                         frac0=0.0, frac1=span)

    method = "quadrature"
    n_done = 0
    if niter > 0:
        for it in range(niter):
            fit = synthesize(lat, lon, C, S, nmax)
            resid = f2 - np.asarray(fit).reshape(f2.shape)
            dC, dS = _quadrature_C(plan, lat, lon, w[:, None] * resid, nmax,
                                   progress=progress, cancel=cancel,
                                   label=f"迭代校正 {it+1}/{niter}",
                                   frac0=span * (it + 1),
                                   frac1=span * (it + 2))
            C = C + dC
            S = S + dS
            n_done += 1
        method = "iterative"

    coeffs = _to_coeffs(C, S, single,
                        {"method": method, "weight_rule": weights.rule,
                         "weight_sum": weights.total,
                         "coverage": weights.total / FOUR_PI,
                         "n_points": int(lat.size),
                         "niter": n_done,
                         "field_unit": field_unit})
    # synthesize() collapses a single time slice to (N,); reshape before
    # subtracting, otherwise (N,) - (N,1) silently broadcasts to (N,N).
    if report_fit:
        STATS["synthesize_for_fit"] += 1
        fit = np.asarray(synthesize(lat, lon, C, S, nmax)).reshape(f2.shape)
        residual_rms = float(np.sqrt(np.mean((fit - f2) ** 2)))
    else:
        fit = None
        residual_rms = float("nan")
    rep = AnalysisReport(
        method=method, n_points=lat.size, nmax=int(nmax), ntime=f2.shape[1],
        weight_rule=weights.rule, weight_sum=weights.total,
        coverage=weights.total / FOUR_PI, n_iterations=n_done,
        residual_rms=residual_rms,
        data_rms=float(np.sqrt(np.mean(f2 ** 2))),
    )
    _absorb_weight_notes(rep, weights)
    rep.meta["field_unit"] = field_unit
    rep.meta.update(plan.meta())
    if not report_fit:
        rep.meta["fit_skipped"] = (
            "report_fit=False：跳过了重建（占一次调用约 45%），"
            "residual_rms/fit_rmse_rel 为 nan。需要逐历元残差表就传 report_fit=True。")
    _fill_gram(rep, lat, lon, w, nmax, cache=gram_cache)
    _fill_dc(rep, coeffs, w, f2, fit)
    return coeffs, rep.check()


def _fill_dc(rep: AnalysisReport, coeffs, w, f2, fit) -> None:
    """Attach the area-weighted residual and the DC identity ``C00`` must meet.

    ``C00`` is the mean of the field over the sphere.  For a zero-filled regional
    dataset that mean is ``sum(w * f) / (4 pi)`` **exactly**, and truncating at any
    degree preserves it.  So a solver whose ``C00`` disagrees with the weighted
    mean of the data has not represented the field -- no matter how small its
    sample residual is.  (``wlsq``/``cg`` on the Yangtze mask: ``C00`` was 1.1e4
    times the weighted mean while the sample residual was 7e-06.)

    ``fit=None`` means the caller skipped the reconstruction (see ``report_fit``
    in :func:`analysis`); the DC identity needs no fit, so it is still reported
    and only the weighted residual becomes ``nan``.
    """
    w = np.asarray(w, dtype=float).ravel()
    f2 = np.asarray(f2, dtype=float)
    if fit is None:
        rep.residual_rms_weighted = float("nan")
    else:
        resid = np.asarray(fit, dtype=float).reshape(f2.shape) - f2
        sw = float(np.sum(w))
        ntime = resid.shape[1] if resid.ndim > 1 else 1
        rep.residual_rms_weighted = (float(np.sqrt(np.sum(w[:, None] * resid ** 2)
                                                   / (sw * ntime)))
                                     if sw > 0 else float("nan"))
    dc = (w[:, None] * f2).sum(axis=0) / FOUR_PI
    rep.meta["dc_mean_expected_all"] = np.asarray(dc, dtype=float).ravel().tolist()
    rep.dc_mean_expected = float(np.asarray(dc).ravel()[0])
    got = np.asarray(coeffs.C)
    rep.dc_mean_got = float(got[0, 0] if got.ndim == 2 else got[0, 0, 0])


def analysis_projection(lat, lon, f, nmax: int,
                        weights: Optional[WeightSet] = None,
                        tau: Optional[float] = None,
                        normalise: str = "none",
                        weights_kw: Optional[dict] = None,
                        progress: Optional[ProgressFn] = None,
                        cancel: Optional[CancelFn] = None,
                        field_unit: str = "unknown",
                        report_fit: bool = True,
                        gram_cache: Optional[dict] = None,
                        longitude_fft: str = "auto",
                        ) -> tuple:
    """Zero-filled global L2 projection, plus optional sampling-bias correction.

    **The objective** (target A of a regional dataset -- the honest reading of an
    outline saved as scattered points)::

        min over deg<=L of   integral_Omega (f - A x)^2 dOmega,   f = 0 outside

    ``Omega`` is the **whole sphere**, not the sample set.  Because ``f`` vanishes
    outside the region, the normal equations collapse to the projection

    .. math::  x = \\frac{1}{4\\pi} A^{\\mathsf T} W f

    with ``W`` the **geometric** integration elements (``sum(w)`` = the covered
    area, *not* ``4*pi``; pass ``normalise='none'`` or ``'region'``).  Dividing by
    ``4\\pi`` here is correct: it is the global normalisation of the projection,
    while the zeros outside contribute nothing to the numerator.

    Consequences, and why this differs from ``'wlsq'``:

    * the sample residual of this estimator **is** the truncation leakage, and it
      stays at a non-zero floor set by ``L`` -- it can never be driven to zero;
    * ``C00`` comes out exactly equal to ``covered area / (4 pi R^2)``, i.e. the
      area fraction, which is a geometric fact for an indicator field;
    * no near-null-space amplification: nothing is inverted.

    Parameters
    ----------
    tau : float, optional
        Enable the **sampling-bias correction**.  With ``tau=None`` (default) the
        plain projection above is returned -- the leakage view.  With ``tau`` in
        ``(0, 1]`` the completeness matrix ``K = A^T W A / (4 pi)`` is
        eigendecomposed (its eigenvectors are the spherical Slepian tapers and its
        eigenvalues their concentration factors) and the projection is corrected
        by ``1/lambda`` **only where** ``lambda >= tau``; directions below ``tau``
        are left alone.  The sampling bias the correction removes is a
        *discretisation* error and is genuinely reducible, unlike the truncation
        leakage.  ``tau -> 0`` degenerates to unregularised weighted least squares,
        which is known to fail on regional data (measured: coefficient error 74x
        on a 30-degree cap) -- so keep ``tau`` well above the noise floor.

    Returns
    -------
    (SHCoeffs, AnalysisReport)
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    f2, single = _as_2d(f, lat.size)
    if weights is None:
        weights = compute_weights(lat, lon, normalise=normalise,
                                  **(weights_kw or {}))
    w = np.asarray(weights.w, dtype=float).ravel()
    if w.size != lat.size:
        raise ValueError("weight vector length does not match the samples")

    _tick(progress, cancel, "零填充全球投影", 0.0)
    plan = _plan_longitude(lat, lon, nmax, longitude_fft)
    fw = w[:, None] * f2
    C, S = _quadrature_C(plan, lat, lon, fw, nmax,
                         progress=progress, cancel=cancel, label="投影",
                         frac0=0.0, frac1=0.5 if tau is not None else 1.0)

    lam = None
    n_corrected = 0
    if tau is not None:
        tau = float(tau)
        if not (0.0 < tau <= 1.0):
            raise ValueError("tau must lie in (0, 1]; use tau=None to disable "
                             "the sampling-bias correction")
        _tick(progress, cancel, "组装完备性矩阵 K", 0.55)
        A = design_matrix_full(lat, lon, nmax)
        K = gram_matrix(A, w)                       # = A^T W A / (4 pi)
        lam, T = np.linalg.eigh(K)
        g = np.where(lam >= tau, 1.0 / np.maximum(lam, 1e-300), 1.0)
        n_corrected = int(np.count_nonzero(lam >= tau))
        _tick(progress, cancel,
              f"采样偏差校正（{n_corrected}/{lam.size} 个方向）", 0.75)
        x0 = _pack_flat(C, S, nmax)
        x0 = T @ (g[:, None] * (T.T @ x0))
        C, S = _unpack(x0, nmax)          # _to_coeffs below collapses if single

    coeffs = _to_coeffs(C, S, single,
                        {"method": "projection", "weight_rule": weights.rule,
                         "weight_sum": weights.total,
                         "target": "zero-filled global",
                         "tau": tau, "n_corrected": n_corrected,
                         "field_unit": field_unit})
    if report_fit:
        STATS["synthesize_for_fit"] += 1
        fit = np.asarray(synthesize(lat, lon, coeffs.C, coeffs.S,
                                    nmax)).reshape(f2.shape)
        residual_rms = float(np.sqrt(np.mean((fit - f2) ** 2)))
    else:
        fit = None
        residual_rms = float("nan")
    rep = AnalysisReport(
        method="projection", n_points=lat.size, nmax=int(nmax), ntime=f2.shape[1],
        weight_rule=weights.rule, weight_sum=weights.total,
        coverage=weights.total / FOUR_PI, n_iterations=0,
        residual_rms=residual_rms,
        data_rms=float(np.sqrt(np.mean(f2 ** 2))),
    )
    _absorb_weight_notes(rep, weights)
    rep.meta["field_unit"] = field_unit
    rep.meta.update(plan.meta())
    if not report_fit:
        rep.meta["fit_skipped"] = "report_fit=False：跳过重建，residual_rms 为 nan"
    rep.meta["target"] = "zero-filled global L2 projection"
    rep.meta["objective"] = ("min over deg<=L of integral_Omega (f - A x)^2 dOmega"
                             "  (Omega = whole sphere, f = 0 outside the data)")
    if lam is not None:
        rep.meta["completeness_eigen"] = {
            "tau": tau, "n_above_tau": n_corrected,
            "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
            "sum_lambda": float(lam.sum()),
            "shannon_number": (nmax + 1) ** 2 * rep.coverage}
        if n_corrected == 0:
            rep.add_warning(
                f"tau={tau:g} corrected nothing: the largest Slepian "
                f"concentration factor reachable in this region at nmax={nmax} "
                f"is lambda_max = {lam.max():.3e}, far below tau.  The "
                "eigenvalues sum to the Shannon number "
                f"{(nmax + 1) ** 2 * rep.coverage:.3e}, so the covered area "
                "supports essentially no band-limited pattern at this degree -- "
                "no direction is well enough determined for a bias correction to "
                "be trusted, and the sampling bias is not what limits you.  The "
                "truncation leakage is, and the pure projection was returned "
                "unchanged.  Raise nmax until lambda_max approaches 1 before "
                "enabling this correction.")
    _tick(progress, cancel, "检查求积完备性", 0.9)
    _fill_gram(rep, lat, lon, w, nmax, cache=gram_cache)
    _fill_dc(rep, coeffs, w, f2, fit)
    return coeffs, rep.check()


# diag
def _fill_gram(rep: AnalysisReport, lat, lon, w, nmax: int,
               budget: float = _GRAM_FLOP_BUDGET,
               cache: Optional[dict] = None) -> None:
    """Attach the quadrature-completeness diagnostics when affordable.

    ``cache`` lets one diagnostic serve a whole series: ``gram_deviation``
    depends only on ``(lat, lon, w, nmax)``, never on time, so recomputing it for
    every epoch is pure waste (measured: 0.56 s per batch call on a 1 deg global
    grid, i.e. ~13% of the call, and 100% wasted from the second epoch on).  When
    a cached value is reused it is flagged in ``rep.meta`` so no reader mistakes
    it for a per-epoch measurement.
    """
    if cache is not None and "gram_deviation" in cache:
        rep.gram_deviation = cache["gram_deviation"]
        rep.meta["gram_checked"] = cache.get("gram_checked")
        rep.meta["gram_reused"] = True
        if cache.get("gram_note"):
            rep.meta["gram_note"] = cache["gram_note"]
        STATS["fill_gram_reused"] += 1
        return
    STATS["fill_gram"] += 1
    ncoef = (nmax + 1) ** 2
    flops = float(lat.size) * ncoef ** 2
    if flops <= budget:
        A = design_matrix_full(lat, lon, nmax)
        K = gram_matrix(A, w)
        rep.gram_deviation = gram_deviation(K)
        rep.meta["gram_checked"] = True
    else:
        # cheap proxy: degree-wise completeness along m = 0
        dev = _diagonal_completeness(lat, w, nmax)
        rep.gram_deviation = dev
        rep.meta["gram_checked"] = "diagonal-m0 only"
        rep.meta["gram_note"] = (
            f"full Gram check skipped (would need {flops:.2e} flops); "
            "reported value is max|sum_i w_i Pbar_n0^2/(4pi) - 1|")
    if cache is not None:
        cache["gram_deviation"] = rep.gram_deviation
        cache["gram_checked"] = rep.meta.get("gram_checked")
        cache["gram_note"] = rep.meta.get("gram_note")


def _diagonal_completeness(lat, w, nmax: int) -> float:
    """``max_n | (1/4pi) sum_i w_i Pbar_n0(sin lat_i)^2 - 1 |``.

    Only the ``m = 0`` column is used -- but ``legendre_columns`` is a *streaming*
    generator, so asking it for the next order after ``m = 0`` is enough to make
    it compute every remaining order.  Measured on a 1 deg global grid with
    ``nmax = 60``: skipping the rest takes this diagnostic from 1.0 s to 0.02 s
    per call, i.e. the whole "cheap proxy" was costing more than the quadrature it
    was meant to be cheaper than.  Hence the early ``return``.
    """
    for m, block in legendre_columns(lat, nmax):
        # m == 0 is yielded first; anything beyond it is never looked at.
        vals = (block * (w[None, :] * block)).sum(axis=1) / FOUR_PI
        return float(np.abs(vals - 1.0).max())
    return 0.0


# ---------------------------------------------------------------------------
# least squares
# ---------------------------------------------------------------------------
def kaula_penalty(nmax: int, spectrum: Optional[np.ndarray] = None,
                  power: float = 2.0, scale: Optional[float] = None,
                  relative_floor: float = 1e-8) -> np.ndarray:
    """Per-coefficient Tikhonov weights ``r_k`` (penalty is ``alpha * r_k^2``).

    With ``spectrum`` given (per-degree RMS amplitudes from a first pass) the
    penalty is ``1/(s_n * (n+1)**power)``, i.e. a *relative* Kaula law that has
    the right units for any field (EWH, geoid, gravity anomaly, ...).  Without a
    spectrum the classical absolute law ``(n+1)**power / scale`` is used.
    """
    m_vec, n_vec = triangle_order(nmax)
    if spectrum is not None:
        s = np.asarray(spectrum, dtype=float)[:nmax + 1]
        s = np.maximum(s, max(float(np.max(s)) * relative_floor, 1e-300))
        r = 1.0 / (s[n_vec] * (n_vec + 1.0) ** power)
    else:
        a = 1e-5 if scale is None else float(scale)
        r = (n_vec + 1.0) ** power / a
    sel_s = m_vec >= 1
    return np.concatenate([r, r[sel_s]])


def _solve_tikhonov(Aw, bw, r, alpha, cache=None):
    """Solve ``min ||Aw x - bw||^2 + alpha^2 ||r*x||^2``.

    The per-coefficient penalty ``r`` enters through the change of variables
    ``y = r*x``, after which the penalty is white and a single eigen-
    decomposition of the transformed normal matrix serves every ``alpha``.
    """
    Ar = Aw / r[None, :]
    if cache is not None and "G" in cache:
        G, V, lam = cache["G"], cache["V"], cache["lam"]
        rhs = cache["rhs"]
    else:
        G = Ar.T @ Ar
        rhs = Ar.T @ bw
        if cache is not None:
            lam, V = np.linalg.eigh(G)
            cache.update(G=G, V=V, lam=lam, rhs=rhs, Ar=Ar)
    if "lam" not in (cache or {}):
        lam, V = np.linalg.eigh(G)
        if cache is not None:
            cache.update(V=V, lam=lam)
    coef = V.T @ rhs
    inv = 1.0 / (lam[:, None] + alpha ** 2)
    y = V @ (coef * inv)
    return y / r[:, None]


def _lcurve_alpha(Aw, bw, r, alphas=(None,), ntime: int = 1):
    """Pick alpha at the maximum-curvature point of the L-curve."""
    cache: dict = {}
    sols, res = [], []
    for a in alphas:
        x = _solve_tikhonov(Aw, bw, r, a, cache)
        sols.append(np.linalg.norm(x))
        res.append(np.linalg.norm(Aw @ x - bw))
    ln_r = np.log(np.maximum(np.asarray(res), 1e-300))
    ln_s = np.log(np.maximum(np.asarray(sols), 1e-300))
    if len(alphas) < 3:
        return float(alphas[0]), np.asarray(sols), np.asarray(res)
    best, best_curv = float(alphas[len(alphas) // 2]), -np.inf
    for i in range(1, len(alphas) - 1):
        dx1, dy1 = ln_r[i] - ln_r[i - 1], ln_s[i] - ln_s[i - 1]
        dx2, dy2 = ln_r[i + 1] - ln_r[i], ln_s[i + 1] - ln_s[i]
        num = abs(dx1 * dy2 - dx2 * dy1)
        den = (dx1 ** 2 + dy1 ** 2) ** 1.5 + 1e-300
        curv = num / den
        if curv > best_curv:
            best_curv, best = curv, float(alphas[i])
    return best, np.asarray(sols), np.asarray(res)


def analysis_wlsq(lat, lon, f, nmax: int,
                  weights: Optional[WeightSet] = None,
                  sigma: Optional[np.ndarray] = None,
                  reg: Optional[str] = None,
                  alpha: Optional[float] = None,
                  reg_power: float = 2.0,
                  reg_scale: Optional[float] = None,
                  method: str = "wlsq",
                  normalise: str = "auto",
                  weights_kw: Optional[dict] = None,
                  cg_tol: float = 1e-10,
                  cg_maxiter: Optional[int] = None,
                  progress: Optional[ProgressFn] = None,
                  cancel: Optional[CancelFn] = None,
                  field_unit: str = "unknown",
                  report_fit: bool = True,
                  gram_cache: Optional[dict] = None,
                  longitude_fft: str = "auto",
                  ) -> tuple:
    """Weighted least squares (direct normal equations or matrix-free CG).

    Row weights are ``sqrt(w_i / sigma_i**2)``.  When ``sigma`` is omitted the
    area elements alone are used as row weights, which is the correct choice
    when every sample represents an equal-area block (mascon, grid-cell average)
    and the natural "Gauss-Markov with density compensation" choice otherwise.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    f2, single = _as_2d(f, lat.size)
    if weights is None:
        weights = compute_weights(lat, lon, normalise=normalise,
                                  **(weights_kw or {}))
    wn = np.asarray(weights.w, dtype=float)
    if sigma is not None:
        sig = np.atleast_1d(np.asarray(sigma, dtype=float)).ravel()
        if sig.size == 1:
            sig = np.full(lat.size, float(sig))
        wn = wn / np.maximum(sig, 1e-300) ** 2
    sw = np.sqrt(wn)

    ncoef = (nmax + 1) ** 2
    fits_memory = float(lat.size) * ncoef * 8.0 <= _memory_budget()
    if method == "cg" or (method == "auto" and not fits_memory):
        return _wlsq_cg(lat, lon, f2, single, nmax, wn, reg, alpha,
                        reg_power, reg_scale, weights, cg_tol, cg_maxiter,
                        progress=progress, cancel=cancel,
                        field_unit=field_unit, longitude_fft=longitude_fft)

    _tick(progress, cancel, "组装设计矩阵", 0.05)
    A = design_matrix_full(lat, lon, nmax)
    Aw = A * sw[:, None]
    bw = f2 * sw[:, None]

    r = None
    if reg:
        if reg == "kaula":
            spec = None
            if reg_scale is None:
                _tick(progress, cancel, "估计 Kaula 先验谱", 0.25)
                spec = _first_pass_spectrum(lat, lon, f2, nmax, wn,
                                            progress=progress, cancel=cancel,
                                            longitude_fft=longitude_fft)
            r = kaula_penalty(nmax, spec, power=reg_power, scale=reg_scale)
        elif reg == "tikhonov":
            r = np.ones(ncoef)
        else:
            raise ValueError("reg must be None, 'tikhonov' or 'kaula'")

    alpha_used = 0.0
    alpha_note = None
    if r is not None:
        if alpha is None:
            if ncoef <= 2500:
                _tick(progress, cancel, "扫描 L 曲线选正则参数", 0.55)
                alphas = np.logspace(-12, 2, 15)
                alpha_used, _s, _r = _lcurve_alpha(Aw, bw, r, alphas)
                alpha_note = "auto (L-curve)"
            else:
                alpha_used = 1e-6
                alpha_note = ("default 1e-6 (L-curve skipped: ncoef > 2500); "
                              "pass alpha= explicitly to control the damping")
        else:
            alpha_used = float(alpha)
            alpha_note = "user"
        _tick(progress, cancel, f"求解正则化法方程 (alpha={alpha_used:.3g})", 0.75)
        x = _solve_tikhonov(Aw, bw, r, alpha_used)
    else:
        _tick(progress, cancel, "组装法方程并求解", 0.6)
        G = Aw.T @ Aw
        rhs = Aw.T @ bw
        try:
            x = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(G, rhs, rcond=None)[0]

    fit = A @ x
    C, S = _unpack(x, nmax)
    if single:
        C = C[:, :, 0]
        S = S[:, :, 0]
    coeffs = SHCoeffs(C, S, {"method": "wlsq", "weight_rule": weights.rule,
                             "weight_sum": weights.total,
                             "coverage": weights.total / FOUR_PI,
                             "n_points": int(lat.size),
                             "reg": reg,
                             "alpha": alpha_used, "field_unit": field_unit})
    rep = AnalysisReport(
        method="wlsq", n_points=lat.size, nmax=int(nmax), ntime=f2.shape[1],
        weight_rule=weights.rule, weight_sum=weights.total,
        coverage=weights.total / FOUR_PI,
        residual_rms=float(np.sqrt(np.mean((fit - f2) ** 2))),
        data_rms=float(np.sqrt(np.mean(f2 ** 2))),
        regularization=reg, alpha=alpha_used,
    )
    _absorb_weight_notes(rep, weights)
    if r is not None:
        rep.meta["alpha_selection"] = alpha_note
    rep.meta["field_unit"] = field_unit
    if not report_fit:
        rep.meta["fit_note"] = (
            "wlsq 的拟合值是解方程的副产品（A 已经成形），跳过它省不下时间，"
            "所以本分支照常报告残差")
    _tick(progress, cancel, "检查求积完备性", 0.9)
    _fill_gram(rep, lat, lon, np.asarray(weights.w), nmax, cache=gram_cache)
    if ncoef <= 1500:
        rep.condition_number = float(np.linalg.cond(Aw))
    else:
        rep.meta["cond_note"] = "condition number skipped (ncoef > 1500)"
    _fill_dc(rep, coeffs, np.asarray(weights.w), f2, fit)
    return coeffs, rep.check()


def _wlsq_cg(lat, lon, f2, single, nmax, wn, reg, alpha, reg_power,
             reg_scale, weights, tol, maxiter,
             progress: Optional[ProgressFn] = None,
             cancel: Optional[CancelFn] = None,
             field_unit: str = "unknown",
             longitude_fft: str = "auto"):
    """Matrix-free CG on the normal equations (memory O(npoints)).

    Each time slice is an independent system, so they are solved one after the
    other; the design matrix is never materialised.
    """
    from scipy.sparse.linalg import LinearOperator, cg

    _tick(progress, cancel, "建立矩阵无关算子", 0.02)
    op = SHOperator(lat, lon, nmax)
    ncoef = op.ncoef
    sw = np.sqrt(wn)
    ntime = f2.shape[1]

    r = None
    alpha_used = 0.0
    if reg:
        if reg == "kaula":
            spec = _first_pass_spectrum(lat, lon, f2, nmax, wn,
                                        progress=progress, cancel=cancel,
                                        longitude_fft=longitude_fft)
            r = kaula_penalty(nmax, spec, power=reg_power, scale=reg_scale)
        elif reg == "tikhonov":
            r = np.ones(ncoef)
        else:
            raise ValueError("reg must be None, 'tikhonov' or 'kaula'")
        alpha_used = 1e-6 if alpha is None else float(alpha)

    def Aop(v):
        return op.matvec(v.reshape(ncoef, 1))[:, 0]

    def Atop(y):
        return op.rmatvec(np.asarray(y).reshape(-1, 1))[:, 0]

    def normal(v):
        return Atop(sw * Aop(v))

    if r is not None:
        rr2 = (alpha_used * r) ** 2

        def normal_reg(v):
            return normal(v) + rr2 * v
        opm = normal_reg
    else:
        opm = normal

    mit = maxiter or min(2000, 20 * ncoef)
    X = np.zeros((ncoef, ntime))
    infos = []
    for t in range(ntime):
        state = {"it": 0}

        def _cb(_xk, _state=state, _t=t):
            _state["it"] += 1
            if cancel is not None and cancel():
                raise AnalysisCancelled(
                    f"cancelled during CG (slice {_t+1}/{ntime})")
            if progress is not None and _state["it"] % 5 == 0:
                frac = (_t + min(_state["it"] / max(mit, 1), 1.0)) / ntime
                progress(f"共轭梯度 时次 {_t+1}/{ntime}，迭代 {_state['it']}",
                         0.05 + 0.85 * frac)

        b = Atop(sw * f2[:, t])
        L = LinearOperator((ncoef, ncoef), matvec=opm, dtype=float)
        xt, info = cg(L, b, rtol=tol, atol=0.0, maxiter=mit, callback=_cb)
        X[:, t] = xt
        infos.append(int(info))

    _tick(progress, cancel, "汇总结果与诊断", 0.95)
    fit = op.matvec(X)
    C, S = _unpack(X, nmax)
    if single:
        C, S = C[:, :, 0], S[:, :, 0]
    coeffs = SHCoeffs(C, S, {"method": "cg", "weight_rule": weights.rule,
                             "weight_sum": weights.total,
                             "coverage": weights.total / FOUR_PI,
                             "n_points": int(lat.size),
                             "reg": reg,
                             "alpha": alpha_used, "cg_info": infos,
                             "field_unit": field_unit})
    rep = AnalysisReport(
        method="cg", n_points=lat.size, nmax=int(nmax), ntime=ntime,
        weight_rule=weights.rule, weight_sum=weights.total,
        coverage=weights.total / FOUR_PI,
        residual_rms=float(np.sqrt(np.mean((fit - f2) ** 2))),
        data_rms=float(np.sqrt(np.mean(f2 ** 2))),
        regularization=reg, alpha=alpha_used,
    )
    _absorb_weight_notes(rep, weights)
    rep.meta["cg_info"] = infos
    rep.meta["field_unit"] = field_unit
    if any(i != 0 for i in infos):
        rep.add_warning(f"CG did not converge in {mit} iterations "
                        "(info != 0); solution may be inaccurate. "
                        "Consider lowering nmax or adding regularization.")
    _fill_gram(rep, lat, lon, np.asarray(weights.w), nmax, budget=5e8)
    _fill_dc(rep, coeffs, np.asarray(weights.w), f2, fit)
    return coeffs, rep.check()


def _unpack(x: np.ndarray, nmax: int):
    """Flat ``[C(all n,m); S(m>=1)]`` -> dense ``(L+1, L+1, ntime)`` arrays."""
    nc = (nmax + 1) * (nmax + 2) // 2
    m_vec, n_vec = triangle_order(nmax)
    sel = m_vec >= 1
    x = np.atleast_2d(x)
    if x.shape[0] != (nmax + 1) ** 2:
        x = x.T
    C = np.zeros((nmax + 1, nmax + 1, x.shape[1]))
    S = np.zeros((nmax + 1, nmax + 1, x.shape[1]))
    C[n_vec, m_vec, :] = x[:nc]
    S[n_vec[sel], m_vec[sel], :] = x[nc:]
    return C, S


def _pack_flat(C: np.ndarray, S: np.ndarray, nmax: int) -> np.ndarray:
    """Inverse of :func:`_unpack`: dense arrays -> flat ``[C; S]`` vector."""
    nc = (nmax + 1) * (nmax + 2) // 2
    m_vec, n_vec = triangle_order(nmax)
    sel = m_vec >= 1
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    if C.ndim == 2:
        C = C[:, :, None]
        S = S[:, :, None]
    out = np.zeros(((nmax + 1) ** 2, C.shape[2]))
    out[:nc] = C[n_vec, m_vec, :]
    out[nc:] = S[n_vec[sel], m_vec[sel], :]
    return out


def _first_pass_spectrum(lat, lon, f2, nmax, wn,
                         progress: Optional[ProgressFn] = None,
                         cancel: Optional[CancelFn] = None,
                         longitude_fft: str = "auto"):
    """Per-degree RMS from a fast quadrature pass, used for Kaula scaling."""
    plan = _plan_longitude(lat, lon, nmax, longitude_fft)
    C, S = _quadrature_C(plan, lat, lon, wn[:, None] * f2, nmax,
                         progress=progress, cancel=cancel,
                         label="先验谱求积", frac0=0.05, frac1=0.25)
    spec = np.zeros(nmax + 1)
    for n in range(nmax + 1):
        vals = np.concatenate([C[n, :n + 1, :].ravel(), S[n, :n + 1, :].ravel()])
        spec[n] = np.sqrt(np.mean(vals ** 2))
    return spec


def _memory_budget() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().available) * 0.4
    except Exception:
        return 2.0 * 1024 ** 3


# ---------------------------------------------------------------------------
# unified entry point
# ---------------------------------------------------------------------------
def analysis(lat, lon, f, nmax: int,
             method: str = "auto",
             rule: str = "auto",
             weights: Optional[WeightSet] = None,
             weights_kw: Optional[dict] = None,
             subset=None,
             user_w=None,
             sigma=None,
             niter: int = 0,
             tau: Optional[float] = None,
             normalise: str = "auto",
             reg: Optional[str] = None,
             alpha: Optional[float] = None,
             reg_power: float = 2.0,
             reg_scale: Optional[float] = None,
             cg_tol: float = 1e-10,
             progress: Optional[ProgressFn] = None,
             cancel: Optional[CancelFn] = None,
             field_unit: str = "unknown",
             target_unit: Optional[str] = None,
             output_unit: Optional[str] = None,
             report_fit: bool = True,
             gram_cache: Optional[dict] = None,
             longitude_fft: str = "auto",
             ) -> tuple:
    """Analyse arbitrary scattered points or a grid into SH coefficients.

    Parameters
    ----------
    lat, lon : array_like
        Sample coordinates in degrees, same length.
    f : array_like
        Values, ``(npoints,)`` or ``(npoints, ntime)``.
    nmax : int
        Maximum degree to solve for (``(nmax+1)**2`` unknowns).
    method : {'auto', 'quadrature', 'iterative', 'projection', 'wlsq', 'cg'}
        ``'auto'`` runs the quadrature pass first, inspects
        ``gram_deviation``/``coverage`` and picks: quadrature when the sampling is
        a valid quadrature rule, **zero-filled global projection** when the data
        cover only a region, and a regularised least-squares solve when they cover
        the sphere but are irregularly sampled.
    rule : str
        Integration-element rule, see :mod:`shkit.weights`.
    subset : array_like of bool, optional
        Which samples to use (e.g. a region mask); weights are computed on the
        full point set when rule is ``'voronoi'``.
    sigma : array_like, optional
        Per-sample standard deviation; switches the least-squares row weights to
        ``w_i / sigma_i**2``.
    niter : int
        Richardson corrections for the quadrature branch.
    tau : float, optional
        Only used by ``method='projection'``.  ``None`` (default) returns the pure
        zero-filled global projection -- the leakage view, where the sample
        residual is the truncation error.  A value in ``(0, 1]`` additionally
        applies a truncated spectral correction of the completeness matrix ``K``
        (whose eigenvectors are the Slepian tapers), repairing the *sampling bias*
        in the directions with concentration factor ``>= tau`` and leaving the
        rest alone.
    normalise : {'auto', 'global', 'region', 'none'}
        Passed to :func:`shkit.weights.compute_weights`.
    weights_kw : dict, optional
        **额外传给** :func:`shkit.weights.compute_weights` **的关键字**，目前唯一
        的用途是 ``{'nlon': nlon}`` —— ``rule='dh'``（Driscoll–Healy）必须知道
        经线条数才能取到精确的高斯-勒让德节点，而点集本身可能是被展平过的网格。
        其它三个求解器（``analysis_quadrature``/``analysis_wlsq``/``analysis_cg``）
        一直都有这个参数，``analysis`` 之前漏了它，于是「全球等经纬网格 + 规则 auto
        选中 dh」这条最常见的路径会在 GUI 里直接抛
        ``TypeError: analysis() got an unexpected keyword argument 'weights_kw'``。
    longitude_fft : {'auto', 'fft', 'direct'}
        **经度方向的加速开关。** ``'auto'``（默认）在点集构成「完整矩形网格
        + 整圈均匀经度 + ``nlon > 2*nmax``」时用一次 ``rfft`` 代替逐点乘
        ``cos/sin``；``'direct'`` 强制走通用直接法；``'fft'`` 要求快路径，
        条件不满足时**报错**而不是悄悄回退。两条路径的结果一致到浮点舍入
        （实测相对 ≤1e-12），所以这纯粹是速度选项；选了哪条、为什么，
        会写进 ``report.meta['longitude_path']`` / ``['longitude_reason']``。
        网格上的分析实测加速见 ``tests/validate_lonfft.py``。
    progress : callable, optional
        ``progress(message: str, fraction: float)`` - called as the solver
        advances.  Safe to pass a UI callback; it is called from the worker
        thread that runs this function.
    cancel : callable, optional
        ``cancel() -> bool``; returning True aborts with
        :class:`AnalysisCancelled`.  Polled at the same points as ``progress``.
    field_unit : str
        **「输入是」—— 正变换公式的选择器。** 它声明这串数字物理上是什么
        （见 :mod:`shkit.units`），并据此把该网格**自身**的球谐系数 ``a_nm``
        换成经典无量纲位系数::

            C_nm = a_nm / f_u

        所以**改这一项会改变导出的系数**：同一块网格声明成 ``geoid`` 与声明
        成 ``ewh`` 是两个不同的物理场，除以的因子不同（第 2 阶差
        ``A₂/R ≈ 13.2`` 倍，逐阶不同）。``'scalar'``/``'unknown'`` 因子为 1，
        导出的就是该场自身的系数（区域平均核这类无物理量纲的场走这一档）。
    target_unit : str, optional
        **「输出为」—— 反变换公式的选择器**（默认与 ``field_unit`` 相同）。
        它**不改变导出的系数**（那永远是 ``C_nm``），而是决定后续
        ``synthesis(..., target_unit=…)`` 把 ``C_nm`` 换成哪个物理量的场；
        取默认值时正反变换互为逆运算，重建精确回到输入网格。这里只做可行性
        校验（例如「普通标量」不能换算成 EWH）并记进报告。

    Returns
    -------
    coeffs : SHCoeffs
        经典无量纲位系数 ``C_nm``（``field_unit='geopotential'``）；输入声明为
        ``scalar``/``unknown`` 时是该场自身的系数。
    report : AnalysisReport
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    if target_unit is None:
        target_unit = output_unit          # 旧名兼容
    f = np.asarray(f, dtype=float)
    if f.shape[0] != lat.size:
        raise ValueError(f"data has {f.shape[0]} samples but {lat.size} coordinates")

    sub = None if subset is None else np.asarray(subset, dtype=bool).ravel()
    if sub is not None:
        if sub.size != lat.size:
            raise ValueError("subset mask size mismatch")
        lat_u, lon_u, f_u = lat[sub], lon[sub], f[sub]
        if sigma is not None:
            sigma = np.asarray(sigma).ravel()[sub]
    else:
        lat_u, lon_u, f_u = lat, lon, f

    if weights is None:
        _tick(progress, cancel, "计算积分元", 0.0)
        weights = compute_weights(lat, lon, rule=rule, subset=sub,
                                  user_w=user_w, normalise=normalise,
                                  **(weights_kw or {}))

    if method == "auto":
        probe, rep = analysis_quadrature(lat_u, lon_u, f_u, nmax,
                                         weights=weights, niter=niter,
                                         progress=progress, cancel=cancel,
                                         field_unit=field_unit,
                                         report_fit=report_fit, gram_cache=gram_cache,
                                         longitude_fft=longitude_fft)
        gd = rep.gram_deviation
        cov = rep.coverage
        healthy = (np.isfinite(gd) and gd < 1e-2) and abs(cov - 1.0) < 1e-3
        if healthy and rep.overdetermination >= 2:
            rep.meta["auto_choice"] = "quadrature"
            return _apply_forward_formula((probe, rep), target_unit, field_unit)
        if cov < 0.999:
            # Regional coverage.  The objective that is actually defined on the
            # field is the zero-filled global projection; escalating to an
            # unregularised least-squares solve here would minimise the misfit at
            # the samples only, and its normal matrix is numerically singular at
            # this coverage -- measured on the Yangtze mask: C00 came out 1.1e4
            # times the weighted mean while the sample residual fell to 7e-06.
            #
            # Efficiency: with niter == 0 and no tau the probe ABOVE is already
            # exactly that projection -- `analysis_quadrature(niter=0)` and
            # `analysis_projection(tau=None)` both evaluate (1/4pi) A^T W f -- so
            # relabel it instead of sweeping the data a second time.
            if niter == 0 and tau is None:
                probe.meta["method"] = "projection"
                probe.meta["target"] = "zero-filled global L2 projection"
                rep.method = "projection"
                rep.meta["target"] = "zero-filled global L2 projection"
                rep.meta["objective"] = (
                    "min over deg<=L of integral_Omega (f - A x)^2 dOmega"
                    "  (Omega = whole sphere, f = 0 outside the data)")
                rep.meta["auto_choice"] = "projection"
                rep.meta["reused_quadrature_probe"] = True
                rep.meta["quadrature_diagnostics"] = {
                    "gram_deviation": gd, "coverage": cov,
                    "residual_rel": rep.fit_rmse_rel}
                return _apply_forward_formula((probe, rep), target_unit,
                                              field_unit)
            _tick(progress, cancel, "区域覆盖：转零填充全球投影", 0.0)
            out, rep2 = analysis_projection(lat_u, lon_u, f_u, nmax,
                                            weights=weights, tau=tau,
                                            progress=progress, cancel=cancel,
                                            field_unit=field_unit,
                                            report_fit=report_fit,
                                            gram_cache=gram_cache,
                                            longitude_fft=longitude_fft)
            rep2.meta["auto_choice"] = "projection"
            rep2.meta["quadrature_diagnostics"] = {
                "gram_deviation": gd, "coverage": cov,
                "residual_rel": rep.fit_rmse_rel}
            return _apply_forward_formula((out, rep2), target_unit, field_unit)
        _tick(progress, cancel, "求积不健康，转加权最小二乘", 0.0)
        big = float(lat_u.size) * (nmax + 1) ** 4 * 8.0 > _memory_budget()
        m2 = "cg" if big else "wlsq"
        out, rep2 = analysis_wlsq(lat_u, lon_u, f_u, nmax, weights=weights,
                                  sigma=sigma, reg=reg, alpha=alpha,
                                  reg_power=reg_power, reg_scale=reg_scale,
                                  method=m2, progress=progress, cancel=cancel,
                                  field_unit=field_unit,
                                  report_fit=report_fit,
                                  gram_cache=gram_cache,
                                  longitude_fft=longitude_fft)
        rep2.meta["auto_choice"] = m2
        rep2.meta["quadrature_diagnostics"] = {
            "gram_deviation": gd, "coverage": cov,
            "residual_rel": rep.fit_rmse_rel}
        return _apply_forward_formula((out, rep2), target_unit, field_unit)

    if method == "quadrature":
        out = analysis_quadrature(lat_u, lon_u, f_u, nmax, weights=weights,
                                  niter=0, progress=progress, cancel=cancel,
                                  field_unit=field_unit,
                                  report_fit=report_fit, gram_cache=gram_cache,
                                  longitude_fft=longitude_fft)
    elif method == "iterative":
        out = analysis_quadrature(lat_u, lon_u, f_u, nmax, weights=weights,
                                  niter=max(niter, 1), progress=progress,
                                  cancel=cancel, field_unit=field_unit,
                                  report_fit=report_fit, gram_cache=gram_cache,
                                  longitude_fft=longitude_fft)
    elif method == "projection":
        out = analysis_projection(lat_u, lon_u, f_u, nmax, weights=weights,
                                  tau=tau, progress=progress, cancel=cancel,
                                  field_unit=field_unit,
                                  report_fit=report_fit, gram_cache=gram_cache,
                                  longitude_fft=longitude_fft)
    elif method in ("wlsq", "cg"):
        out = analysis_wlsq(lat_u, lon_u, f_u, nmax, weights=weights,
                            sigma=sigma, reg=reg, alpha=alpha,
                            reg_power=reg_power, reg_scale=reg_scale,
                            method=method, progress=progress, cancel=cancel,
                            field_unit=field_unit,
                            report_fit=report_fit, gram_cache=gram_cache,
                            longitude_fft=longitude_fft)
    else:
        raise ValueError(f"unknown method {method!r}")

    return _apply_forward_formula(out, target_unit, field_unit)


def _apply_forward_formula(result, target_unit, field_unit_in):
    """**正变换**：把刚解出的网格系数换成经典无量纲位系数 ``C_nm``。

    「输入是」选的就是这条公式（``C_nm = a_nm / f_u``），所以**改「输入是」
    确实会改变导出的系数**：同一块网格声明成 geoid 和声明成 EWH 是两个不同
    的物理场，除的因子不同（第 2 阶差 ``A₂/R ≈ 13.2`` 倍）。

    ``target_unit``（「输出为」）选的是**反变换公式**，它作用在
    ``synthesis(..., target_unit=…)`` 上、决定重建出来是什么物理量；
    导出的系数永远是 ``C_nm``，所以这里只校验可行性并记进报告。
    """
    coeffs, rep = result
    from . import units
    try:
        canon = units.to_canonical(coeffs, field_unit_in)
    except ValueError as exc:
        raise ValueError(
            f"analysis(field_unit={field_unit_in!r}) 无法完成正变换：\n"
            f"{exc}") from exc

    src_for_check = field_unit_in
    if src_for_check in ("scalar", "unknown"):
        src_for_check = "scalar"
    tgt = field_unit_in if target_unit is None else target_unit
    if target_unit is not None and target_unit != field_unit_in:
        problem = units.require_convertible(src_for_check, target_unit)
        if problem is not None:
            raise ValueError(
                f"analysis(field_unit={field_unit_in!r}, target_unit={target_unit!r}) "
                f"这一对公式不成立：\n{problem}")

    rep.meta["forward_from"] = field_unit_in
    rep.meta["field_unit"] = units.field_unit(canon)
    rep.meta["inverse_target"] = tgt
    # 也放到系数自身的 meta 上，这样会随文件头一起落盘，便于追溯
    canon.meta["forward_from"] = field_unit_in
    canon.meta["inverse_target"] = tgt
    rep.meta["forward_formula"] = (
        "C_nm = a_nm / f_u，f_u = "
        + units.formula_text(field_unit_in))
    rep.meta["inverse_formula"] = (
        f"场 = C_nm × f_t，f_t = {units.formula_text(tgt)}")
    rep.add_warning(_formula_warning(field_unit_in, tgt))
    return canon, rep


def _formula_warning(field_unit_in, target) -> str:
    from . import units
    a = units.FIELD_UNIT_LABELS.get(field_unit_in, field_unit_in)
    b = units.FIELD_UNIT_LABELS.get(target, target)
    same = (target == field_unit_in)
    return (f"正变换公式（输入=「{a}」）：\n"
            f"    C_nm = a_nm / f_u,  f_u = {units.formula_text(field_unit_in)}\n"
            f"导出的是经典无量纲位系数 C_nm。\n"
            f"反变换公式（输出=「{b}」）：\n"
            f"    重建场 = C_nm × f_u,  f_u = {units.formula_text(field_unit_in)}\n"
            f"        —— 与输入同物理量，可直接与原数据逐点比对；\n"
            f"    输出场 = C_nm × f_t,  f_t = {units.formula_text(target)}\n"
            + ("        —— 与「输入是」同档，所以输出场就是重建场本身。"
               if same else ""))


def analysis_iterative(lat, lon, f, nmax, niter: int = 1, **kw):
    """Convenience wrapper: quadrature + ``niter`` Richardson corrections."""
    return analysis_quadrature(lat, lon, f, nmax, niter=niter, **kw)
