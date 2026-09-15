# -*- coding: utf-8 -*-
"""
shkit.basis
===========

4-pi normalised associated Legendre functions and spherical harmonic basis
matrices, in the same convention as ``m2py`` / ``gridSHconvert`` and SHTOOLS
``norm=1, csphase=1``.

The recurrence is the numerically stable **cross-order** form (``method 3`` of
the reference ``pz_legendre.m``):

.. math::

    \\bar P_{n,m} = \\alpha_{n,m}\\bar P_{n-2,m}
                  + \\beta_{n,m}\\bar P_{n-2,m-2}
                  - \\gamma_{n,m}\\bar P_{n,m-2} \\qquad (m \\ge 2)

Memory strategy
---------------
The naive implementation materialises an ``(nmax+1, nmax+1, npoints)`` cube,
which for ``nmax=100`` and 100k points is ~8 GB.  Everything here is therefore
built on a **column generator** (:func:`legendre_columns`) that only ever keeps
the current column plus the ``m-2`` column alive, so the peak footprint is
``2*(nmax+1)*npoints`` doubles.  Both the quadrature analysis and the full
design matrix are built from that single primitive.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from .coeffs import triangle_order

__all__ = [
    "legendre_columns",
    "legendre_columns_vec",
    "legendre_pbar",
    "q_at_pole_m1",
    "design_matrix",
    "design_matrix_full",
    "synthesize",
    "synthesize_grid_fft",
    "ncoef",
    "ncoef_sine",
]

try:  # pragma: no cover - optional acceleration
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

from .lonfft import plan_longitude          # noqa: E402  (numpy-only module)


def ncoef(nmax: int) -> int:
    """Number of real coefficients up to degree ``nmax``: ``(nmax+1)**2``."""
    return (nmax + 1) ** 2


def ncoef_sine(nmax: int) -> int:
    """Number of sine coefficients with ``m >= 1``: ``nmax*(nmax+1)/2``."""
    return nmax * (nmax + 1) // 2


# ---------------------------------------------------------------------------
# Legendre recurrence
# ---------------------------------------------------------------------------
def _legendre_column_forward(col: np.ndarray, m: int, nmax: int,
                             t: np.ndarray, start: int) -> None:
    """Fill ``col[start..nmax, :]`` for ``m = 0`` (start=1) or ``m = 1`` (start=2)."""
    for n in range(start, nmax + 1):
        c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - m) * (n + m)))
        num = (2 * n + 1) * (n + m - 1) * (n - m - 1)
        c2 = np.sqrt(num / ((2 * n - 3) * (n + m) * (n - m))) if num > 0 else 0.0
        col[n] = c1 * t * col[n - 1] - c2 * col[n - 2]


def legendre_columns(lat_deg, nmax: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(m, block)`` where ``block[n - m, :] == Pbar_{n,m}(lat)``.

    ``n`` runs from ``m`` to ``nmax``, so ``block`` has shape
    ``(nmax - m + 1, npoints)``.  Only columns ``m``, ``m-1`` and ``m-2`` are
    alive at any time.

    Parameters
    ----------
    lat_deg : array_like
        Geocentric latitude in degrees.
    nmax : int
        Maximum degree.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    npts = lat.size
    t = np.sin(np.deg2rad(lat))          # sin(lat)
    u = np.cos(np.deg2rad(lat))          # cos(lat)

    if nmax < 0:
        raise ValueError("nmax must be >= 0")

    keep: dict[int, np.ndarray] = {}

    # ---- m = 0 -----------------------------------------------------------
    col = np.zeros((nmax + 1, npts))
    col[0] = 1.0
    if nmax >= 1:
        _legendre_column_forward(col, 0, nmax, t, start=1)
    keep[0] = col
    yield 0, col[0:nmax + 1]

    if nmax >= 1:
        # ---- m = 1 -------------------------------------------------------
        col = np.zeros((nmax + 1, npts))
        col[1] = np.sqrt(3.0) * u
        if nmax >= 2:
            _legendre_column_forward(col, 1, nmax, t, start=2)
        keep[1] = col
        yield 1, col[1:nmax + 1]

    # ---- m >= 2 ----------------------------------------------------------
    for m in range(2, nmax + 1):
        cm2 = keep.pop(m - 2)
        col = np.zeros((nmax + 1, npts))
        for n in range(m, nmax + 1):
            a1 = np.sqrt((2 * n + 1) * (n - m) * (n - m - 1) /
                         ((2 * n - 3) * (n + m) * (n + m - 1)))
            if m == 2:
                g1 = np.sqrt(2.0) * np.sqrt((n - m + 1) * (n - m + 2) /
                                            ((n + m) * (n + m - 1)))
                b1 = np.sqrt(2.0) * np.sqrt((2 * n + 1) * (n + m - 2) * (n + m - 3) /
                                            ((2 * n - 3) * (n + m) * (n + m - 1)))
            else:
                g1 = np.sqrt((n - m + 1) * (n - m + 2) / ((n + m) * (n + m - 1)))
                b1 = np.sqrt((2 * n + 1) * (n + m - 2) * (n + m - 3) /
                             ((2 * n - 3) * (n + m) * (n + m - 1)))
            col[n] = a1 * col[n - 2] + b1 * cm2[n - 2] - g1 * cm2[n]
        keep[m] = col
        yield m, col[m:nmax + 1]


# ---------------------------------------------------------------------------
# Horizontal gradient primitives (co-latitude derivative and Pbar/sin(theta))
# ---------------------------------------------------------------------------
def q_at_pole_m1(nmax: int, t_pole: float) -> np.ndarray:
    """``Q_n1 = Pbar_n1/sin(theta)`` **exactly at a pole** (``t_pole = +-1``).

    ``Pbar_n1 = sin(theta) * Q_n1(cos theta)`` with the ``sin(theta)`` factor
    independent of ``n``, so dividing the ``m = 1`` recurrence by it gives the
    same recurrence for ``Q`` with the seed ``Q_11 = sqrt(3)``.  ``Q_n1`` is a
    polynomial of degree ``n-1`` in ``t = cos(theta)``, so evaluating it at
    ``t = +-1`` *is* the analytic pole limit -- no ``eps`` offset, no ``0/0``.

    Returns an array of length ``nmax + 1`` (index 0 is unused/zero).
    """
    Q = np.zeros(nmax + 1)
    if nmax >= 1:
        Q[1] = np.sqrt(3.0)
    for n in range(2, nmax + 1):
        c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - 1) * (n + 1)))
        num = (2 * n + 1) * n * (n - 2)
        c2 = np.sqrt(num / ((2 * n - 3) * (n + 1) * (n - 1))) if num > 0 else 0.0
        Q[n] = c1 * t_pole * Q[n - 1] - c2 * Q[n - 2]
    return Q


def _q_from_p(P: np.ndarray, u: np.ndarray, pole: np.ndarray, nmax: int, m: int,
              t_pole: np.ndarray) -> np.ndarray:
    """``P/sin(theta)`` for one order ``m``, with the pole rows replaced exactly.

    * ``m = 1``: the pole value is :func:`q_at_pole_m1` at ``t = +-1``;
    * ``m >= 2``: exactly ``0`` (``Pbar_nm ~ sin^m(theta)``), not a limit.
    """
    Q = np.zeros_like(P)
    if not pole.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            Q[m:] = P[m:] / u[None, :]
        return Q
    safe = ~pole
    if safe.any():
        Q[m:, safe] = P[m:, safe] / u[safe][None, :]
    if m == 1:
        for j in np.nonzero(pole)[0]:
            Q[1:, j] = q_at_pole_m1(nmax, float(t_pole[j]))[1:]
    return Q


def legendre_columns_vec(lat_deg, nmax: int):
    """Yield ``(m, P, D, Q)`` for ``n = m..nmax`` (streaming, 3 columns alive).

    ``P[n-m, :] = Pbar_{n,m}``, ``D[n-m, :] = dPbar_{n,m}/dtheta`` and
    ``Q[n-m, :] = Pbar_{n,m}/sin(theta)``, each of shape
    ``(nmax - m + 1, npoints)``.

    * ``D`` is obtained by differentiating **each recurrence relation** with
      respect to ``theta`` (``dt/dtheta = -sin(theta)``, ``du/dtheta = cos(theta)``).
      No ``1/sin(theta)`` ever appears, so it is pole-safe with no special case.
    * ``Q`` cannot avoid the division, so only the pole rows are overridden with
      the exact values (see :func:`_q_from_p`).  ``Q`` is meaningless for
      ``m = 0`` (and never needed: the east component carries a factor ``m``),
      so ``None`` is yielded there.

    Memory: same strategy as :func:`legendre_columns` -- only orders ``m``,
    ``m-1`` and ``m-2`` are alive, so the footprint stays a constant multiple of
    ``nmax * npoints`` instead of a full ``(nmax+1, nmax+1, npoints)`` cube.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    npts = lat.size
    t = np.sin(np.deg2rad(lat))          # = cos(theta)
    u = np.cos(np.deg2rad(lat))          # = sin(theta)
    dt = -u                              # d(cos theta)/dtheta
    du = t                               # d(sin theta)/dtheta
    pole = np.abs(u) < 1e-12
    t_pole = np.where(pole, np.sign(t), 0.0)

    if nmax < 0:
        raise ValueError("nmax must be >= 0")

    keep: dict = {}

    # ---- m = 0 -----------------------------------------------------------
    col = np.zeros((nmax + 1, npts))
    dcol = np.zeros((nmax + 1, npts))
    col[0] = 1.0
    if nmax >= 1:
        col[1] = np.sqrt(3.0) * t
        dcol[1] = np.sqrt(3.0) * dt
        for n in range(2, nmax + 1):
            c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / (n * n))
            num = (2 * n + 1) * (n - 1) * (n - 1)
            c2 = np.sqrt(num / ((2 * n - 3) * n * n)) if num > 0 else 0.0
            col[n] = c1 * t * col[n - 1] - c2 * col[n - 2]
            dcol[n] = c1 * (dt * col[n - 1] + t * dcol[n - 1]) - c2 * dcol[n - 2]
    keep[0] = (col, dcol)
    yield 0, col[0:nmax + 1], dcol[0:nmax + 1], None

    # ---- m = 1 -----------------------------------------------------------
    if nmax >= 1:
        col = np.zeros((nmax + 1, npts))
        dcol = np.zeros((nmax + 1, npts))
        col[1] = np.sqrt(3.0) * u
        dcol[1] = np.sqrt(3.0) * du
        for n in range(2, nmax + 1):
            c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - 1) * (n + 1)))
            num = (2 * n + 1) * n * (n - 2)
            c2 = np.sqrt(num / ((2 * n - 3) * (n + 1) * (n - 1))) if num > 0 else 0.0
            col[n] = c1 * t * col[n - 1] - c2 * col[n - 2]
            dcol[n] = c1 * (dt * col[n - 1] + t * dcol[n - 1]) - c2 * dcol[n - 2]
        keep[1] = (col, dcol)
        Q = _q_from_p(col, u, pole, nmax, 1, t_pole)
        yield 1, col[1:nmax + 1], dcol[1:nmax + 1], Q[1:nmax + 1]

    # ---- m >= 2 ----------------------------------------------------------
    for m in range(2, nmax + 1):
        cm2, dcm2 = keep.pop(m - 2)
        col = np.zeros((nmax + 1, npts))
        dcol = np.zeros((nmax + 1, npts))
        for n in range(m, nmax + 1):
            a1 = np.sqrt((2 * n + 1) * (n - m) * (n - m - 1) /
                         ((2 * n - 3) * (n + m) * (n + m - 1)))
            if m == 2:
                g1 = np.sqrt(2.0) * np.sqrt((n - m + 1) * (n - m + 2) /
                                            ((n + m) * (n + m - 1)))
                b1 = np.sqrt(2.0) * np.sqrt((2 * n + 1) * (n + m - 2) * (n + m - 3) /
                                            ((2 * n - 3) * (n + m) * (n + m - 1)))
            else:
                g1 = np.sqrt((n - m + 1) * (n - m + 2) / ((n + m) * (n + m - 1)))
                b1 = np.sqrt((2 * n + 1) * (n + m - 2) * (n + m - 3) /
                             ((2 * n - 3) * (n + m) * (n + m - 1)))
            col[n] = a1 * col[n - 2] + b1 * cm2[n - 2] - g1 * cm2[n]
            dcol[n] = a1 * dcol[n - 2] + b1 * dcm2[n - 2] - g1 * dcm2[n]
        keep[m] = (col, dcol)
        Q = _q_from_p(col, u, pole, nmax, m, t_pole)
        yield m, col[m:nmax + 1], dcol[m:nmax + 1], Q[m:nmax + 1]


def legendre_pbar(lat_deg, nmax: int, chunk: Optional[int] = None) -> np.ndarray:
    """4-pi normalised ``Pbar_{n,m}(sin lat)`` for all ``m <= n <= nmax``.

    Returns an array of shape ``(npoints, (nmax+1)(nmax+2)/2)`` in the m2py
    triangle ordering (``m`` outer, ``n`` inner).

    Parameters
    ----------
    chunk : int, optional
        Process the points in chunks of this many rows to cap peak memory.
        The result is still fully materialised, so this only limits the
        intermediate columns.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    npts = lat.size
    NC = (nmax + 1) * (nmax + 2) // 2
    out = np.empty((npts, NC))

    if chunk is None or chunk >= npts:
        idx = 0
        for _m, block in legendre_columns(lat, nmax):
            k = block.shape[0]
            out[:, idx:idx + k] = block.T
            idx += k
        return out

    for s in range(0, npts, chunk):
        e = min(s + chunk, npts)
        idx = 0
        for _m, block in legendre_columns(lat[s:e], nmax):
            k = block.shape[0]
            out[s:e, idx:idx + k] = block.T
            idx += k
    return out


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------
def design_matrix(lat_deg, lon_deg, nmax: int,
                  coef_layout: str = "m2py"):
    """Build the spherical harmonic design matrix for arbitrary points.

    Returns
    -------
    Yc : ndarray, shape ``(npoints, NC)``
        ``Pbar_{n,m}(sin lat) * cos(m lon)`` in m2py triangle ordering.
    Ys : ndarray, shape ``(npoints, NS)``
        ``Pbar_{n,m}(sin lat) * sin(m lon)`` for ``m >= 1`` only, same ordering
        restricted to ``m >= 1`` (``NS = nmax(nmax+1)/2``).

    Notes
    -----
    ``sin(m*lon)`` vanishes identically for ``m = 0``, so those columns are
    never built.  Forgetting this makes the collocation system rank deficient -
    it is the single most common bug in scattered-point SH analysis.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
    if lat.shape != lon.shape:
        raise ValueError(f"lat{lat.shape} and lon{lon.shape} must match")
    P = legendre_pbar(lat, nmax)
    m_vec, _n_vec = triangle_order(nmax)
    lam = np.deg2rad(lon)
    Yc = P * np.cos(np.outer(lam, m_vec))
    sel = m_vec >= 1
    Ys = P[:, sel] * np.sin(np.outer(lam, m_vec[sel]))
    return Yc, Ys


def design_matrix_full(lat_deg, lon_deg, nmax: int) -> np.ndarray:
    """``[Yc | Ys]``, shape ``(npoints, (nmax+1)**2)``.

    This is the matrix used by every least-squares / Slepian routine.
    """
    Yc, Ys = design_matrix(lat_deg, lon_deg, nmax)
    return np.hstack([Yc, Ys])


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def synthesize(lat_deg, lon_deg, C, S, nmax: Optional[int] = None,
               weights_scale: float = 1.0,
               longitude_fft: str = "auto",
               report: Optional[dict] = None) -> np.ndarray:
    """Evaluate the SH series at arbitrary points.

    ``weights_scale`` multiplies every degree (used to fold in Gaussian filter
    coefficients or EWH conversion factors without touching the coefficients).

    ``longitude_fft`` (``'auto'`` default / ``'fft'`` / ``'direct'``) selects the
    longitude kernel.  With ``'auto'`` a complete rectangular grid on a
    whole-circle uniform longitude axis with ``nlon > 2*nmax`` is evaluated with
    one ``irfft`` per component instead of one ``cos``/``sin`` pair per point
    (:func:`synthesize_grid_fft`); anything else falls back to the universal
    sweep.  The two agree to floating-point round-off, so this is a pure speed
    switch.  ``report``, if given, receives ``longitude_path`` /
    ``longitude_reason`` so a fallback is never silent.

    Returns ``(npoints,)`` or ``(npoints, ntime)`` matching the coefficient
    array's time axis.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)

    if C.ndim == 2:
        C3, S3 = C[:, :, None], S[:, :, None]
    else:
        C3, S3 = C, S
    ntime = C3.shape[2]

    if nmax is None:
        nmax = C3.shape[0] - 1
    nmax = min(nmax, C3.shape[0] - 1)

    if np.ndim(weights_scale) == 0:
        ws = np.full(nmax + 1, float(weights_scale))
    else:
        ws = np.asarray(weights_scale, dtype=float)[:nmax + 1]

    plan = plan_longitude(lat, lon, nmax, longitude_fft)
    if report is not None:
        report.update(plan.meta())
    if plan.path == "fft":
        g = plan.grid
        vals = synthesize_grid_fft(g.lat_vec, g.lon_vec,
                                   C3[:nmax + 1, :nmax + 1],
                                   S3[:nmax + 1, :nmax + 1], ws)
        flat = g.from_grid(vals)
        # from_grid already drops a length-1 time axis, so only a 2-D result
        # (ntime > 1) still needs the single-slice squeeze.
        if flat.ndim == 1:
            return flat
        return flat[:, 0] if ntime == 1 else flat

    lam = np.deg2rad(lon)
    out = np.zeros((lat.size, ntime))
    for m, block in legendre_columns(lat, nmax):
        n_idx = np.arange(m, nmax + 1)
        pb = block * ws[n_idx][:, None]                       # (n-m+1, npts)
        c = C3[m:nmax + 1, m, :]                              # (n-m+1, ntime)
        s = S3[m:nmax + 1, m, :]
        acc = pb.T @ c                                        # (npts, ntime)
        if m >= 1:
            cm = np.cos(m * lam)[:, None]
            sm = np.sin(m * lam)[:, None]
            acc = acc * cm
            acc = acc + (pb.T @ s) * sm
        out += acc
    return out[:, 0] if ntime == 1 else out


def synthesize_grid_fft(lat_vec, lon_vec, C, S, weights_scale=1.0,
                        time_chunk: Optional[int] = None) -> np.ndarray:
    """Evaluate the SH series on a grid via one ``irfft`` per component (A4).

    On a whole-circle uniform longitude axis the field is a Fourier series in
    ``lambda``::

        f(theta, lambda) = A_0(theta) + sum_{m>=1} [ A_m cos(m lambda)
                                                   + B_m sin(m lambda) ]
        A_m = sum_n Pbar_nm(sin theta) C_nm ,   B_m = ... S_nm

    so instead of evaluating the ``m``-sum at every one of ``nlon`` longitudes we
    build the complex spectrum once per latitude and transform::

        G[m] = e^{i m lambda_0} (A_m - i B_m) / 2     (m >= 1)
        G[0] = A_0
        f    = nlon * irfft(G, n = nlon)

    The ``e^{i m lambda_0}`` factor is **not optional**: ``irfft`` only ever sees
    ``2*pi*k/nlon``, while the axis is ``lambda_0 + 2*pi*k/nlon``.  Dropping it is
    exact on ``lon = 0, 1, 2, ...`` and wrong by O(1) elsewhere
    (:func:`shkit.lonfft.longitude_phase`, and the regressions in
    ``tests/validate_lonfft.py``).

    Cost: ``O(nSHCS * nlat * ntime) + O(nlat * ntime * nlon log nlon)`` against
    ``O(nSHCS * nlat * nlon * ntime)`` for the direct sweep over every point.

    ``lat_vec``/``lon_vec`` must be the sorted unique axes and ``lon_vec`` must be
    whole-circle and uniform; the caller checks that
    (:func:`shkit.lonfft.fft_path_applicable`).
    """
    from .lonfft import longitude_phase

    lat = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    C3 = C[:, :, None] if C.ndim == 2 else C
    S3 = S[:, :, None] if S.ndim == 2 else S
    ntime = C3.shape[2]

    nmax = C3.shape[0] - 1
    if np.ndim(weights_scale) == 0:
        ws = np.full(nmax + 1, float(weights_scale))
    else:
        ws = np.asarray(weights_scale, dtype=float)[:nmax + 1]

    nlat, nlon = lat.size, lon.size
    nf = nlon // 2 + 1
    ph = longitude_phase(float(lon[0]), nmax)
    out = np.empty((nlat, nlon, ntime))
    step = int(time_chunk) if time_chunk else ntime
    for t0 in range(0, ntime, step):
        t1 = min(t0 + step, ntime)
        g = np.zeros((nf, nlat, t1 - t0), dtype=complex)
        for m, block in legendre_columns(lat, nmax):
            n_idx = np.arange(m, nmax + 1)
            pb = block * ws[n_idx][:, None]                   # (n-m+1, nlat)
            a = pb.T @ C3[m:nmax + 1, m, t0:t1]               # (nlat, ntc)
            if m == 0:
                # B_0 has no meaning: sin(0*lambda) == 0 for every S_n0.
                g[0] = a
                continue
            b = pb.T @ S3[m:nmax + 1, m, t0:t1]
            g[m] = ph[m] * (a - 1j * b) / 2.0
        out[:, :, t0:t1] = (nlon * np.fft.irfft(g, n=nlon, axis=0)).transpose(1, 0, 2)
    return out[:, :, 0] if ntime == 1 else out
