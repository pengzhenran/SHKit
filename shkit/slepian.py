# -*- coding: utf-8 -*-
"""
shkit.slepian
=============

Spherical Slepian (spatio-spectral concentration) analysis for **regional**
data.

Why this module exists
----------------------
Regional data cannot determine global coefficients: the problem has a null
space (every band-limited function that vanishes inside the region).  Plain
quadrature pretends the field is zero outside and leaks across the boundary;
an unregularised least-squares solve amplifies noise into the null space and
produces absurd coefficients.  The Slepian basis fixes both by construction:
it is the basis that is *optimally concentrated* inside the region.

Implementation
--------------
The concentration problem needs no external library here, because the
concentration matrix **is** the weighted Gram matrix already used by
:mod:`shkit.diagnostics`:

.. math::

    K = \\frac{1}{4\\pi}A^{T} W A

with :math:`A` the real (cos/sin) design matrix and :math:`W` the area elements
of the region.  For a band-limited field :math:`g = A c`,

.. math::

    \\lambda = \\frac{\\int_R g^2\\,\\mathrm{d}\\Omega}{\\int_\\Omega g^2\\,\\mathrm{d}\\Omega}
             = \\frac{c^{T} K c}{c^{T} c}

so the Slepian functions are the eigenvectors of ``K`` and the concentration
factors are its eigenvalues.  Everything then has a closed form: with
orthonormal tapers :math:`t_\\alpha` and eigenvalues :math:`\\lambda_\\alpha`,
the weighted least-squares fit in the taper subspace is **diagonal**,

.. math::

    s_\\alpha = \\frac{t_\\alpha^{T} A^{T} W f}{4\\pi\\,\\lambda_\\alpha},
    \\qquad c = \\sum_\\alpha s_\\alpha t_\\alpha

which makes the ill-posedness explicit: only tapers with
:math:`\\lambda_\\alpha \\approx 1` are usable.  Truncating at
:math:`\\lambda_\\alpha > \\lambda_{min}` *is* the regularisation.

The number of eigenvalues above 1/2 is the **Shannon number**
:math:`(L+1)^2 A/(4\\pi)` - the number of coefficients a region of that area can
genuinely support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .analysis import _as_2d, _unpack
from .basis import design_matrix_full, synthesize
from .coeffs import SHCoeffs
from .diagnostics import AnalysisReport
from .weights import FOUR_PI, WeightSet, compute_weights

__all__ = [
    "SlepianBasis",
    "slepian_basis",
    "slepian_analysis",
    "slepian_expand",
]

_MAX_DENSE_NCOEF = 6000        # eigh cost warning threshold


@dataclass
class SlepianBasis:
    """Slepian functions (in SH coefficient space) for one region."""

    tapers: np.ndarray                  # (ncoef, ntaper), orthonormal columns
    eigenvalues: np.ndarray             # (ntaper,), descending
    nmax: int
    region_fraction: float              # area of the region / 4*pi
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ properties
    @property
    def ncoef(self) -> int:
        return (self.nmax + 1) ** 2

    @property
    def ntaper(self) -> int:
        return self.tapers.shape[1]

    @property
    def shannon(self) -> float:
        """Degrees of freedom the region supports, ``(L+1)**2 * A/(4 pi)``."""
        return (self.nmax + 1) ** 2 * self.region_fraction

    def usable(self, lam_min: float = 0.5) -> int:
        """Number of tapers above a concentration threshold."""
        return int(np.count_nonzero(self.eigenvalues > lam_min))

    # -------------------------------------------------------------- helpers
    def to_shcoeffs(self, s: np.ndarray, meta: Optional[dict] = None) -> SHCoeffs:
        """Expand Slepian coefficients into SH coefficients."""
        s = np.atleast_2d(np.asarray(s, dtype=float))
        if s.shape[0] != self.ntaper:
            s = s.T
        C, S = _unpack(self.tapers @ s, self.nmax)
        if C.shape[2] == 1:
            C, S = C[:, :, 0], S[:, :, 0]
        m = {"basis": "slepian", "region_fraction": self.region_fraction}
        m.update(meta or {})
        return SHCoeffs(C, S, m)

    def spectrum(self) -> str:
        ev = self.eigenvalues
        lines = [f"Slepian spectrum ({self.ntaper} tapers, nmax={self.nmax}, "
                 f"region = {self.region_fraction:.4f} of the sphere)"]
        for lo in (0.99, 0.9, 0.5, 0.1, 0.01):
            lines.append(f"  lambda > {lo:<5}: {int(np.count_nonzero(ev > lo)):5d}")
        lines.append(f"  Shannon number = {self.shannon:.1f}  "
                     f"(eigenvalues in (0.5, 1]: {self.usable():d})")
        return "\n".join(lines)

    def describe(self) -> str:
        return (f"SlepianBasis(nmax={self.nmax}, ncoef={self.ncoef}, "
                f"ntaper={self.ntaper}, region_fraction={self.region_fraction:.5f})\n"
                + self.spectrum())


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def slepian_basis(lat_deg, lon_deg, nmax: int,
                  weights: Optional[WeightSet] = None,
                  rule: str = "auto",
                  subset=None,
                  ntaper: Optional[int] = None,
                  lam_min: float = 1e-8,
                  normalise: str = "region",
                  user_w=None) -> SlepianBasis:
    """Eigen-decompose the concentration problem for a region.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Samples **inside the region** (or the full set together with ``subset``).
    nmax : int
        Band limit.  Cost is dominated by an ``(nmax+1)**2`` dense symmetric
        eigenproblem, so ``nmax`` around 60 (3721x3721) is a practical ceiling
        on a laptop.
    weights : WeightSet, optional
        Integration elements; ``rule``/``subset`` are used otherwise.
    ntaper : int, optional
        Keep only this many tapers (the best-concentrated ones).  Defaults to
        all eigenvalues above ``lam_min``.
    lam_min : float
        Eigenvalue floor for automatic taper selection.
    normalise : {'region', 'global', 'none'}
        ``'region'`` (default) keeps the geometric region area, which is what
        makes the eigenvalues true concentration factors in ``[0, 1]``.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    if weights is None:
        weights = compute_weights(lat, lon, rule=rule, subset=subset,
                                  normalise=normalise, user_w=user_w)
    w = np.asarray(weights.w, dtype=float)
    if w.size != lat.size:
        raise ValueError("weights do not match the points")

    ncoef = (nmax + 1) ** 2
    A = design_matrix_full(lat, lon, nmax)
    K = (A * w[:, None]).T @ A / FOUR_PI
    K = 0.5 * (K + K.T)                     # symmetrise away round-off

    ev, evec = np.linalg.eigh(K)
    order = np.argsort(ev)[::-1]
    ev = ev[order]
    evec = evec[:, order]

    if ntaper is None:
        ntaper = int(np.count_nonzero(ev > lam_min))
        ntaper = max(ntaper, 1)
    ntaper = min(int(ntaper), evec.shape[1])

    meta = {
        "weight_rule": weights.rule,
        "weight_sum": weights.total,
        "lam_min": lam_min,
        "max_eigenvalue": float(ev[0]),
        "cond_warning": None,
    }
    if ncoef > _MAX_DENSE_NCOEF:
        meta["cond_warning"] = (
            f"ncoef={ncoef} is large: the dense eigendecomposition is O(ncoef^3) "
            f"({ncoef**3:.1e} flops).  Consider lowering nmax or using "
            "method='wlsq' with regularization instead.")
    return SlepianBasis(tapers=evec[:, :ntaper], eigenvalues=ev[:ntaper],
                        nmax=int(nmax),
                        region_fraction=float(weights.total / FOUR_PI),
                        meta=meta)


# ---------------------------------------------------------------------------
# analysis / synthesis
# ---------------------------------------------------------------------------
def slepian_expand(lat_deg, lon_deg, f, basis: SlepianBasis,
                   weights: Optional[WeightSet] = None,
                   lam_min: float = 0.5,
                   rule: str = "auto",
                   user_w=None) -> np.ndarray:
    """Slepian coefficients ``s_alpha`` (one row per time slice)."""
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    f2, _single = _as_2d(f, lat.size)
    if weights is None:
        weights = compute_weights(lat, lon, rule=rule, normalise="region",
                                  user_w=user_w)
    w = np.asarray(weights.w, dtype=float)

    A = design_matrix_full(lat, lon, basis.nmax)
    proj = A.T @ (w[:, None] * f2)          # (ncoef, ntime)
    lam = basis.eigenvalues
    keep = lam > lam_min
    s = np.zeros((basis.ntaper, f2.shape[1]))
    s[keep] = (basis.tapers[:, keep].T @ proj) / (FOUR_PI * lam[keep])[:, None]
    return s


def slepian_analysis(lat_deg, lon_deg, f, basis: Optional[SlepianBasis] = None,
                     nmax: Optional[int] = None,
                     weights: Optional[WeightSet] = None,
                     rule: str = "auto",
                     subset=None,
                     ntaper: Optional[int] = None,
                     lam_min: float = 0.5,
                     user_w=None) -> tuple:
    """Regional SH analysis in the Slepian basis.

    Returns ``(SHCoeffs, AnalysisReport, SlepianBasis)``.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    f2, single = _as_2d(f, lat.size)
    if weights is None:
        weights = compute_weights(lat, lon, rule=rule, subset=subset,
                                  normalise="region", user_w=user_w)
    if basis is None:
        if nmax is None:
            raise ValueError("pass either basis= or nmax=")
        basis = slepian_basis(lat, lon, nmax, weights=weights, ntaper=ntaper,
                              lam_min=min(lam_min, 1e-8))

    s = slepian_expand(lat, lon, f2, basis, weights=weights, lam_min=lam_min,
                       user_w=user_w)
    used = int(np.count_nonzero(basis.eigenvalues > lam_min))
    coeffs = basis.to_shcoeffs(s, meta={"lam_min": lam_min, "ntaper_used": used})

    fit = synthesize(lat, lon, coeffs.C, coeffs.S)
    fit = np.asarray(fit).reshape(f2.shape)
    rep = AnalysisReport(
        method="slepian", n_points=lat.size, nmax=basis.nmax,
        ntime=f2.shape[1], weight_rule=weights.rule,
        weight_sum=weights.total, coverage=weights.total / FOUR_PI,
        residual_rms=float(np.sqrt(np.mean((fit - f2) ** 2))),
        data_rms=float(np.sqrt(np.mean(f2 ** 2))),
        regularization=f"slepian(lam>{lam_min})",
    )
    rep.meta["ntaper_used"] = used
    rep.meta["ntaper_available"] = basis.ntaper
    rep.meta["shannon"] = basis.shannon
    rep.meta["slepian_eigenvalues"] = basis.eigenvalues.tolist()
    rep.meta["max_eigenvalue"] = float(basis.eigenvalues[0])
    # For a Slepian solve the meaningful conditioning is the ratio of the best
    # to the worst *retained* concentration factor - that is exactly the
    # amplification the diagonal normal matrix applies.
    if used >= 1:
        rep.condition_number = float(basis.eigenvalues[0] /
                                     basis.eigenvalues[used - 1])
    if basis.meta.get("cond_warning"):
        rep.add_warning(basis.meta["cond_warning"])
    if used < 1:
        rep.add_warning(
            f"no Slepian taper has lambda > {lam_min}: the region is too small "
            f"for degree {basis.nmax}.  Shannon number is {basis.shannon:.1f}; "
            "lower nmax or lower lam_min.")
    else:
        rep.add_warning(
            f"Slepian solve keeps {used} of {basis.ntaper} tapers "
            f"(Shannon number {basis.shannon:.1f}); the result is a regional "
            "estimate, not a global solution - outside the region it is "
            "effectively zero by construction.")
    return coeffs, rep.check(), basis
