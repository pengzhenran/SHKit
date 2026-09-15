# -*- coding: utf-8 -*-
"""
shkit.synthesis
===============

Spherical harmonic **synthesis**: coefficients -> values at arbitrary points or
on arbitrary (lat, lon) grids.

Synthesis needs no integration element - only the correct normalisation - so it
works for any geometry, global or regional, regular or scattered.

Large point sets are processed in chunks: :func:`shkit.basis.legendre_columns`
keeps two ``(nmax+1) x npoints`` working arrays alive, which for 10**6 points and
``nmax=100`` would already be 1.6 GB.  Chunking caps that.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .basis import synthesize as _synthesize_basis
from .basis import synthesize_grid_fft
from .coeffs import SHCoeffs
from .filters import EARTH_RADIUS_M, RHO_AVE, RHO_WATER
from .lonfft import plan_longitude

__all__ = [
    "synthesis",
    "synthesis_grid",
    "regular_grid",
    "degree_scale",
    "synthesis_horizontal",
]


def synthesis_horizontal(lat_deg, lon_deg, coeffs: SHCoeffs, **kwargs) -> dict:
    """Horizontal displacement (north/east/magnitude/azimuth), in metres.

    Thin re-export of :func:`shkit.gradient.synthesis_horizontal`, so that
    "coefficients -> field" stays discoverable next to :func:`synthesis`.
    The real implementation lives in :mod:`shkit.gradient` because the horizontal
    displacement is the *horizontal gradient* of the per-degree-scaled scalar,
    not a per-degree multiplication (it mixes ``(n, m)`` with ``(n-1, m)``).
    """
    from .gradient import synthesis_horizontal as _impl
    return _impl(lat_deg, lon_deg, coeffs, **kwargs)


def _unit_ratio(coeffs: SHCoeffs, target_unit: str, love_numbers,
                love_numbers_h, radius_m, rho_ave, rho_water,
                allow_unit_mismatch: bool) -> np.ndarray:
    """Per-degree factors turning ``coeffs``' declared unit into ``target_unit``.

    Both units are expressed relative to the canonical dimensionless
    geopotential coefficients, so the ratio is ``f_target / f_source``.  The
    guard refuses the two ways this can go silently wrong: an undeclared or
    explicitly non-physical source, and a target that is not a quantity.
    """
    from . import units

    src = units.field_unit(coeffs)
    if src == target_unit:
        return np.ones(coeffs.nmax + 1)

    problem = units.require_convertible(src, target_unit)
    if problem is not None and not allow_unit_mismatch:
        raise ValueError(
            f"synthesis(target_unit={target_unit!r}) 的单位不一致：\n" + problem +
            "\n\n若确实要强行换算，传 allow_unit_mismatch=True。")

    kw = dict(love_numbers=love_numbers, love_numbers_h=love_numbers_h,
              radius_m=radius_m, rho_ave=rho_ave, rho_water=rho_water)
    f_dst = units.degree_factors(target_unit, coeffs.nmax, **kw)
    if src in ("scalar", "unknown"):
        # only reachable with allow_unit_mismatch=True: assume the numbers are
        # already geopotential coefficients
        return f_dst
    f_src = units.degree_factors(src, coeffs.nmax, **kw)
    return f_dst / f_src


def degree_scale(coeffs: SHCoeffs,
                 gaussian_km: float = 0.0,
                 ewh: bool = False,
                 love_numbers=None,
                 gaussian_method: str = "glq",
                 extra: Optional[np.ndarray] = None) -> np.ndarray:
    """Multiplicative per-degree factors applied at synthesis time.

    This is how smoothing and the EWH/geopotential conversion are applied
    **without touching the stored coefficients**, matching what the reference
    ``gridSHconvert`` does (the Gaussian filter is folded in at reconstruction).
    """
    L = coeffs.nmax
    s = np.ones(L + 1)
    if gaussian_km and gaussian_km > 0:
        from .filters import gaussian_coefficients
        s = s * gaussian_coefficients(gaussian_km, L, method=gaussian_method)
    if ewh:
        from .filters import ewh_scaling
        s = s * ewh_scaling(L, love_numbers=love_numbers)
    if extra is not None:
        e = np.asarray(extra, dtype=float).ravel()
        s = s * e[:L + 1]
    return s


def synthesis(lat_deg, lon_deg, coeffs: SHCoeffs,
              nmax: Optional[int] = None,
              gaussian_km: float = 0.0,
              ewh: bool = False,
              love_numbers=None,
              chunk: int = 200_000,
              allow_unit_mismatch: bool = False,
              target_unit: Optional[str] = None,
              love_numbers_h=None,
              radius_m: float = EARTH_RADIUS_M,
              rho_ave: float = RHO_AVE,
              rho_water: float = RHO_WATER,
              longitude_fft: str = "auto",
              report: Optional[dict] = None) -> np.ndarray:
    """Evaluate a SH expansion at arbitrary points.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Target coordinates in degrees.
    coeffs : SHCoeffs
        Coefficients (possibly multi-time).
    nmax : int, optional
        Truncate to this degree (defaults to the coefficient's own ``nmax``).
    gaussian_km : float
        Isotropic Gaussian smoothing radius applied on the fly.
    ewh : bool
        Shorthand for ``target_unit='ewh'`` (kept for compatibility).
    target_unit : str, optional
        **Preferred way to request a unit conversion.**  One of
        ``'geopotential'``, ``'geoid'``, ``'surface_density'``, ``'ewh'``,
        ``'radial_displacement'`` (see :mod:`shkit.units`).

        This is a *conversion*, not a display option, and it is applied
        relative to what the coefficients already are.  If
        ``coeffs.meta['field_unit']`` already equals ``target_unit`` nothing
        happens (so asking for EWH twice is harmless); if the coefficient unit
        is undeclared/``'scalar'``, or the conversion is undefined, SHKit
        raises instead of silently producing numbers that are wrong by
        ``~1e7``.
    chunk : int
        Points processed per block (memory control).
    allow_unit_mismatch : bool
        Proceed even when the recorded ``field_unit`` says the conversion is
        wrong.  Only for deliberate experiments.
    longitude_fft : {'auto', 'fft', 'direct'}
        **经度方向的加速开关。** 点集构成「完整矩形网格 + 整圈均匀经度 +
        ``nlon > 2*nmax``」时用一次 ``irfft`` 代替逐点乘 ``cos/sin``；
        两条路径结果一致到浮点舍入（实测相对 ≤5e-15），所以这纯粹是速度选项。
        ``'fft'`` 在条件不满足时**报错**而不是悄悄回退；``'auto'``（默认）
        回退并把理由写进 ``report``（``longitude_path`` / ``longitude_reason``）。

    Returns
    -------
    ndarray
        ``(npoints,)`` or ``(npoints, ntime)``.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    if lat.size != lon.size:
        raise ValueError("lat and lon must have the same length")

    if target_unit is None and ewh:
        target_unit = "ewh"                 # backwards-compatible shorthand
    L = coeffs.nmax if nmax is None else min(int(nmax), coeffs.nmax)
    ws = degree_scale(coeffs, gaussian_km, 0.0, love_numbers)[:L + 1]
    if target_unit is not None:
        ws = ws * _unit_ratio(coeffs, target_unit, love_numbers,
                              love_numbers_h, radius_m, rho_ave, rho_water,
                              allow_unit_mismatch)[:L + 1]

    # A4: pick the longitude kernel **once, on the whole point set**.  The chunk
    # loop below would otherwise hand each block to the direct sweep, where no
    # single block is a complete grid and the FFT is unreachable no matter how
    # regular the full set is.
    plan = plan_longitude(lat, lon, L, longitude_fft)
    if report is not None:
        report.update(plan.meta())
    if plan.path == "fft":
        g = plan.grid
        vals = synthesize_grid_fft(g.lat_vec, g.lon_vec,
                                   coeffs.C[:L + 1, :L + 1],
                                   coeffs.S[:L + 1, :L + 1], ws)
        return g.from_grid(vals)

    if lat.size <= chunk:
        return _synthesize_basis(lat, lon, coeffs.C, coeffs.S, L,
                                 weights_scale=ws, longitude_fft="direct")
    ntime = coeffs.ntime
    out = np.empty((lat.size, ntime))
    for s in range(0, lat.size, chunk):
        e = min(s + chunk, lat.size)
        out[s:e] = np.atleast_2d(
            _synthesize_basis(lat[s:e], lon[s:e], coeffs.C, coeffs.S, L,
                              weights_scale=ws,
                              longitude_fft="direct").reshape(e - s, -1))
    return out[:, 0] if ntime == 1 else out


def regular_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                 lat_step: float, lon_step: float):
    """Build a regular (lat, lon) sampling, inclusive of both endpoints.

    Returns ``(lat_vec, lon_vec)`` with ``lat`` ascending.
    """
    if lat_step <= 0 or lon_step <= 0:
        raise ValueError("steps must be positive")
    nlat = int(round((lat_max - lat_min) / lat_step)) + 1
    nlon = int(round((lon_max - lon_min) / lon_step)) + 1
    lat = np.linspace(lat_min, lat_max, nlat)
    lon = np.linspace(lon_min, lon_max, nlon)
    return lat, lon


def synthesis_grid(lat_vec, lon_vec, coeffs: SHCoeffs,
                   nmax: Optional[int] = None,
                   gaussian_km: float = 0.0,
                   ewh: bool = False,
                   love_numbers=None,
                   chunk: int = 200_000,
                   allow_unit_mismatch: bool = False,
                   longitude_fft: str = "auto",
                   report: Optional[dict] = None,
                   target_unit: Optional[str] = None,
                   love_numbers_h=None,
                   radius_m: float = EARTH_RADIUS_M,
                   rho_ave: float = RHO_AVE,
                   rho_water: float = RHO_WATER) -> np.ndarray:
    """Evaluate a SH expansion on a regular lat/lon grid.

    Returns an array shaped ``(nlat, nlon)`` or ``(nlat, nlon, ntime)`` with the
    ``lat`` axis matching the order of ``lat_vec`` (and ``lon_vec`` likewise).

    ``ewh=True`` is shorthand for ``target_unit='ewh'``; ``target_unit`` names any
    quantity from :mod:`shkit.units` (``'geoid'``, ``'radial_displacement'`` ...)
    and is guarded exactly as in :func:`synthesis`.  ``longitude_fft``/``report``
    are forwarded there too; on a whole-circle uniform axis with ``nlon > 2*nmax``
    the default ``'auto'`` evaluates the grid with one ``irfft`` per component.
    """
    lat = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    vals = synthesis(LA.ravel(), LO.ravel(), coeffs, nmax=nmax,
                     gaussian_km=gaussian_km, ewh=ewh,
                     love_numbers=love_numbers, chunk=chunk,
                     allow_unit_mismatch=allow_unit_mismatch,
                     longitude_fft=longitude_fft, report=report,
                     target_unit=target_unit, love_numbers_h=love_numbers_h,
                     radius_m=radius_m, rho_ave=rho_ave, rho_water=rho_water)
    if vals.ndim == 1:
        return vals.reshape(lat.size, lon.size)
    return vals.reshape(lat.size, lon.size, vals.shape[1])
