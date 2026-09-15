# -*- coding: utf-8 -*-
"""
shkit.filters
=============

Degree-domain (isotropic) filtering and the GRACE/GRACE-FO equivalent-water-
height conversion.

Gaussian smoothing follows the standard Jekeli/Wahr formulation used by the
reference implementation in ``m2py``:

.. math::

    b = \\frac{\\ln 2}{1-\\cos(r/R)},\\qquad
    W(\\alpha) = \\frac{b\\,e^{-b(1-\\cos\\alpha)}}{1-e^{-2b}}

.. math::

    W_n = \\frac{1}{2}\\int_{-1}^{1} W(\\cos\\alpha)\\,P_n(\\cos\\alpha)\\,
          \\mathrm{d}(\\cos\\alpha)

computed by Gauss-Legendre quadrature (``method='glq'``, numerically stable for
all degrees) or by the classical forward recurrence (``method='frc'``, fast but
unstable at high degree).  ``W_0 == 1`` by construction.

The filter is applied at **synthesis** time (degree-wise multiplication of the
coefficients), which is exactly what the reference software does.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .coeffs import SHCoeffs

__all__ = [
    "R_EARTH_KM",
    "EARTH_RADIUS_M",
    "RHO_AVE",
    "RHO_WATER",
    "FIELD_UNITS",
    "gaussian_coefficients",
    "apply_degree_filter",
    "apply_gaussian",
    "load_love_numbers",
    "load_lln",
    "love_number",
    "ewh_scaling",
    "field_unit",
    "with_field_unit",
    "to_ewh",
    "to_geopotential",
    "check_ewh_conversion",
]

R_EARTH_KM = 6378.1363          # mean Earth radius used for the filter (km)
EARTH_RADIUS_M = 6378136.46     # equatorial radius used for the EWH factor (m)
RHO_AVE = 5517.0                # mean Earth density (kg/m3)
RHO_WATER = 1000.0              # water density (kg/m3)

_LOVE_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Gaussian filter
# ---------------------------------------------------------------------------
def gaussian_coefficients(radius_km: float, nmax: int, method: str = "glq",
                          ngl: Optional[int] = None) -> np.ndarray:
    """Isotropic Gaussian filter coefficients ``W_n`` for ``n = 0..nmax``.

    Parameters
    ----------
    radius_km : float
        Averaging radius.  ``radius_km <= 0`` returns all ones (no filtering).
    nmax : int
        Maximum degree.
    method : {'glq', 'frc'}
        ``'glq'`` (default) is Gauss-Legendre quadrature - stable at any degree.
        ``'frc'`` is the classical forward recurrence; it is faster but loses
        accuracy above roughly 300 km of averaging radius.
    ngl : int, optional
        Number of quadrature nodes for ``'glq'`` (default ``max(501, 2*nmax+51)``,
        always odd so that the integrand's mass is well sampled).
    """
    if radius_km <= 0:
        return np.ones(nmax + 1)

    b = np.log(2.0) / (1.0 - np.cos(radius_km / R_EARTH_KM))

    if method == "frc":
        W = np.ones(nmax + 1)
        if nmax >= 1:
            e2b = np.exp(-2.0 * b)
            W[1] = (1.0 + e2b) / (1.0 - e2b) - 1.0 / b
            for n in range(1, nmax):
                W[n + 1] = -((2.0 * n + 1.0) / b) * W[n] + W[n - 1]
        return W

    if method != "glq":
        raise ValueError(f"unknown method {method!r}; use 'glq' or 'frc'")

    if ngl is None:
        ngl = max(501, 2 * nmax + 51)
    if ngl % 2 == 0:
        ngl += 1
    x, gw = np.polynomial.legendre.leggauss(ngl)
    Wa = b * np.exp(-b * (1.0 - x)) / (1.0 - np.exp(-2.0 * b))
    P = np.zeros((nmax + 1, ngl))
    P[0] = 1.0
    if nmax >= 1:
        P[1] = x
    for n in range(1, nmax):
        P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
    W = P @ (gw * Wa)
    # The kernel is defined up to a constant that fixes unit gain at degree 0.
    # Imposing W_0 == 1 exactly also removes the quadrature error of the very
    # peaked integrand at small radii (measured 1.1e-10 at r = 100 km).
    return W / W[0]


def apply_degree_filter(coeffs: SHCoeffs,
                        W: np.ndarray) -> SHCoeffs:
    """Multiply every coefficient of degree ``n`` by ``W[n]``."""
    W = np.asarray(W, dtype=float).ravel()
    L = coeffs.nmax
    if W.size < L + 1:
        raise ValueError(f"need at least {L+1} filter coefficients, got {W.size}")
    w = W[:L + 1]
    C = coeffs.C * w[:, None] if coeffs.C.ndim == 2 else \
        coeffs.C * w[:, None, None]
    S = coeffs.S * w[:, None] if coeffs.S.ndim == 2 else \
        coeffs.S * w[:, None, None]
    meta = dict(coeffs.meta)
    meta["degree_filter"] = "applied"
    return SHCoeffs(C, S, meta, coeffs.times)


def apply_gaussian(coeffs: SHCoeffs, radius_km: float,
                   method: str = "glq") -> SHCoeffs:
    """Convenience wrapper: Gaussian-smooth a coefficient set in place."""
    W = gaussian_coefficients(radius_km, coeffs.nmax, method=method)
    out = apply_degree_filter(coeffs, W)
    out.meta["gaussian_km"] = radius_km
    out.meta["gaussian_method"] = method
    return out


# ---------------------------------------------------------------------------
# Load Love numbers and EWH conversion
# ---------------------------------------------------------------------------
def _default_love_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "love_numbers.npy")


def _default_lln_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "load_love_numbers.npz")


def load_love_numbers(path: Optional[str] = None) -> np.ndarray:
    """Load the degree-indexed load Love numbers ``k'_n``.

    Defaults to the table shipped with the software
    (``SHKit/data/love_numbers.npy``), which is the same file the reference
    ``m2py`` / ``gridSHconvert`` uses (Wang 2012 / PREM:
    ``k'_0 = 0``, ``k'_1 = 0.02617`` in the CF frame, ``k'_2 = -0.30516``, ...).

    Use :func:`load_lln` when you also need ``h'_n`` or ``l'_n``.
    """
    p = path or _default_love_path()
    if p not in _LOVE_CACHE:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"load Love number table not found: {p}\n"
                "Pass love_numbers=<path to a .npy/.txt table> explicitly.")
        kl = np.load(p)
        _LOVE_CACHE[p] = np.asarray(kl, dtype=float).ravel()
    return _LOVE_CACHE[p]


def load_lln(path: Optional[str] = None) -> dict:
    """Load the full load Love number set: ``h'``, ``l'`` and ``k'``.

    Returns a dict with keys ``n``, ``h``, ``l``, ``k`` (plus ``model``,
    ``source``, ``asymptote``).  Defaults to
    ``SHKit/data/load_love_numbers.npz``, built from
    ``PREM-LLNs.dat`` (Wang et al. 2012) by ``tools/build_love_numbers.py``.

    ``k'_1`` in that table has the **CE -> CF** frame correction applied,
    ``k'_1 = -(h'_1 + 2 l'_1)/3``, exactly as ``pz_LLN.m`` does, so it agrees
    with the ``m2py`` table to 1e-16 and existing EWH results are unchanged.

    A raw ``PREM-LLNs.dat``-style text file can be passed as ``path``
    (columns ``n h l k nl nk``); the asymptote row with ``n = inf`` is read
    into ``asymptote``.
    """
    p = path or _default_lln_path()
    key = ("lln", p)
    if key in _LOVE_CACHE:
        return _LOVE_CACHE[key]

    if str(p).lower().endswith(".npz"):
        with np.load(p, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
    else:
        out = _read_lln_text(p)
    for k in ("h", "l", "k"):
        if k in out:
            out[k] = np.asarray(out[k], dtype=float).ravel()
    _LOVE_CACHE[key] = out
    return out


def _read_lln_text(path: str) -> dict:
    """Read a ``PREM-LLNs.dat``-style table (``n h l k [nl nk]``)."""
    rows, asym = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vals = [float(x) for x in parts[:6]]
            except ValueError:
                continue                                   # header
            if not np.isfinite(vals[0]):
                asym = vals
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: 没有解析到任何 'n h l k' 数据行")
    a = np.asarray(rows, dtype=float)
    n = a[:, 0].astype(int)
    nmax = int(n.max())
    k = a[:, 3].copy()
    k[0] = -(a[0, 1] + 2.0 * a[0, 2]) / 3.0        # degree 1: CE -> CF

    interpolated = not np.array_equal(n, np.arange(1, nmax + 1))
    if interpolated:
        # The short ``*-LLNs.dat`` tables only list every degree up to 10 and
        # then a handful of nodes (18, 32, 56, 100, 180, ...).  pz_LLN.m fills
        # the gaps with interp1, so we do the same (linear) and say so.
        grid = np.arange(nmax + 1, dtype=float)
        hh = np.interp(grid, n, a[:, 1])
        ll = np.interp(grid, n, a[:, 2])
        kk = np.interp(grid, n, k)
        hh[0] = ll[0] = kk[0] = 0.0
        kk[1] = -(hh[1] + 2.0 * ll[1]) / 3.0       # keep the frame fix exact
        hh[1], ll[1] = a[0, 1], a[0, 2]
    else:
        hh = np.concatenate(([0.0], a[:, 1]))
        ll = np.concatenate(([0.0], a[:, 2]))
        kk = np.concatenate(([0.0], k))

    return {
        "n": np.arange(nmax + 1),
        "h": hh,
        "l": ll,
        "k": kk,
        "asymptote": np.asarray(asym[1:4]) if asym else np.zeros(3),
        "model": os.path.basename(path),
        "source": path,
        "interpolated": interpolated,
        "n_given": int(n.size),
    }


def love_number(n: int, table: Optional[np.ndarray] = None,
                kind: str = "k") -> float:
    """Load Love number of degree ``n`` (0 beyond the end of the table).

    ``kind`` selects which one: ``'k'`` (potential, the default and the
    historical behaviour), ``'h'`` (radial) or ``'l'`` (horizontal).
    """
    if table is not None:
        arr = np.asarray(table, dtype=float).ravel()
    else:
        arr = np.asarray(load_lln()[kind], dtype=float)
    n = int(n)
    return float(arr[n]) if 0 <= n < arr.size else 0.0


def ewh_scaling(nmax: int, love_numbers=None,
                earth_radius_m: float = EARTH_RADIUS_M,
                rho_ave: float = RHO_AVE,
                rho_water: float = RHO_WATER) -> np.ndarray:
    """Degree-wise factor converting geopotential coefficients to EWH.

    .. math::

        A_n = \\frac{R\\,\\bar\\rho_E}{3\\rho_w}\\cdot\\frac{2n+1}{1+k_n}

    Following the reference implementation, the *analysis* direction divides by
    ``A_n`` while the *synthesis* direction multiplies by it.  Note that a
    **Gaussian filter is not** part of this factor - apply it separately.
    """
    kl = load_love_numbers() if love_numbers is None else np.asarray(love_numbers)
    n = np.arange(nmax + 1)
    kn = np.array([kl[i] if i < kl.size else 0.0 for i in n])
    return (earth_radius_m * rho_ave / (3.0 * rho_water)) * (2 * n + 1) / (1.0 + kn)


# ---------------------------------------------------------------------------
# Quantity labels and unit conversion live in :mod:`shkit.units`.
# ---------------------------------------------------------------------------
# These thin re-exports keep the older ``shkit.filters.to_ewh`` spelling working.
# The real implementation (and the full set of quantities: geoid, surface
# density, EWH, radial displacement) is in :mod:`shkit.units`; the import is
# local because ``shkit.units`` imports the physical constants from this module.

from . import units as _units          # noqa: E402  (deliberate late import)


def field_unit(coeffs: SHCoeffs) -> str:
    """Declared physical meaning of the coefficients (see :mod:`shkit.units`)."""
    return _units.field_unit(coeffs)


def with_field_unit(coeffs: SHCoeffs, unit: str, **meta) -> SHCoeffs:
    """Tag coefficients with their physical meaning (metadata only)."""
    return _units.with_field_unit(coeffs, unit, **meta)


def to_ewh(coeffs: SHCoeffs, love_numbers=None) -> SHCoeffs:
    """Dimensionless geopotential coefficients -> EWH coefficients.

    Equivalent to ``shkit.units.convert(coeffs, 'ewh')``.  The factor
    ``A_n = R*rho_ave/(3*rho_w)*(2n+1)/(1+k'_n)`` is **per-degree**, so EWH and
    geopotential coefficients can never be mixed or rescaled by one number.
    """
    return _units.convert(coeffs, "ewh", love_numbers=love_numbers)


def to_geopotential(coeffs: SHCoeffs, love_numbers=None) -> SHCoeffs:
    """Inverse of :func:`to_ewh`."""
    return _units.convert(coeffs, "geopotential", love_numbers=love_numbers)


def check_ewh_conversion(coeffs: SHCoeffs, want_ewh: bool,
                         strict: bool = True):
    """Backwards-compatible wrapper around :func:`shkit.units.require_convertible`.

    Returns ``None`` when applying (or not applying) the EWH factor is
    consistent with the declared ``field_unit``, otherwise a Chinese
    explanation.
    """
    src = _units.field_unit(coeffs)
    if want_ewh:
        problem = _units.require_convertible(src, "ewh")
        if problem is not None:
            return (problem + "\n\n正确做法：在分析时用 field_unit= 声明输入场到底"
                    "是位系数、geoid 还是 EWH，然后统一用 shkit.units.convert() "
                    "做换算。")
    else:
        if src == "geopotential":
            return ("系数是无量纲重力位系数；若要得到等效水高，需要乘以 A_n"
                    "（shkit.units.convert(coeffs, 'ewh')，或 synthesis 时指定 "
                    "target_unit='ewh'）。")
    return None
