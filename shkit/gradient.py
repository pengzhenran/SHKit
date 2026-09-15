# -*- coding: utf-8 -*-
"""
shkit.gradient
==============

Spherical horizontal gradient of a scalar spherical-harmonic field, and the
**horizontal surface displacement** it gives rise to under surface-mass loading.

Why this is a separate operator (and not a per-degree factor)
-------------------------------------------------------------
Every other quantity in :mod:`shkit.units` is ``coefficients x f_n`` -- a
per-degree multiplication, so ``radial_displacement``, ``ewh``, ``geoid`` ...
all live in the same ``(n, m)`` slot.  The horizontal displacement is the
**horizontal gradient** of that same scalar, so it mixes ``(n, m)`` with
``(n-1, m)`` and ``(n, m+-1)``::

    u_r :  C_nm x f_n                        -> same (n, m)
    u_N :  C_nm x f_n x d/dtheta             -> (n, m) and (n-1, m) mixed
    u_E :  C_nm x f_n x (1/sin) d/dlambda    -> (n, m) and (n, m+-1) mixed

Conventions (frozen; see ``docs/水平形变契约.md``, F1-F11)
---------------------------------------------------------
* 4-pi normalised, **no** Condon-Shortley phase (SHTOOLS ``norm=1, csphase=1``);
* ``theta`` is co-latitude; ``theta-hat`` points **south**, ``lambda-hat`` **east**::

      u_N = -dS/dtheta ,      u_E = +(1/sin theta) dS/dlambda

* per-degree factor ``F^h_n = R * l'_n / (1 + k'_n)`` (``l'_0 = 0``, so degree 0
  is annihilated -- as it must be, a constant has no gradient);
* poles are **exact**, not ``eps`` limits: ``Pbar_n1/sin(theta) -> Q_n1(+-1)``
  (``Q_11 = sqrt(3)``) and ``Pbar_nm/sin(theta) == 0`` for ``m >= 2``;
* ``m = 0`` contributes to ``u_N`` only (``dPbar_n0/dtheta != 0``, while the east
  component carries a factor ``m``).  Dropping it makes ``u_N`` systematically
  wrong (~31% in the reference SHSynth prototype);
* ``magnitude`` is a rotation invariant and is valid at the poles; ``azimuth``
  is ``atan2(u_E, u_N)`` in ``[0, 360)`` clockwise from north.  The pole's
  *direction* is coordinate-singular -- report the numbers, but do not draw
  arrows there.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .basis import legendre_columns_vec
from .coeffs import SHCoeffs
from .filters import EARTH_RADIUS_M, load_lln, load_love_numbers
# One longitude-axis detector for the whole package; analysis (A3) asks the very
# same question, so a divergence in the criterion would show up as two different
# algorithms accepting the same grid.  Re-exported below for compatibility.
from .lonfft import fft_path_applicable as _fft_path_applicable
from .lonfft import longitude_phase

__all__ = [
    "degree_factors_horizontal",
    "fft_path_applicable",
    "sph_gradient",
    "horizontal_field",
    "horizontal_grid",
    "synthesis_horizontal",
    "synthesis_horizontal_grid",
    "canonical_scaled",
]


def _tick(progress, cancel, message: str, fraction: float):
    """报进度并响应取消（与 :func:`shkit.analysis._tick` 同一口径）。

    取消统一抛 ``RuntimeError("cancelled")``（``shkit.series`` 也是这个约定），
    调用方据此区分为"用户取消"而不是真失败。
    """
    if cancel is not None and cancel():
        raise RuntimeError("cancelled")
    if progress is not None:
        progress(message, float(min(max(fraction, 0.0), 1.0)))


def fft_path_applicable(lon_deg, nmax: int, *, rtol: float = 1e-9):  # noqa: F811
    """Can the **longitude FFT** path be used for this longitude axis?

    Thin alias of :func:`shkit.lonfft.fft_path_applicable`, kept here because this
    is where the horizontal path first used it.
    """
    return _fft_path_applicable(lon_deg, nmax, rtol=rtol)


def degree_factors_horizontal(nmax: int, *,
                              love_numbers_l=None,
                              love_numbers_k=None,
                              radius_m: float = EARTH_RADIUS_M) -> np.ndarray:
    """``F^h_n = R * l'_n / (1 + k'_n)`` for ``n = 0..nmax`` (metres).

    ``F^h_0 = 0`` because ``l'_0 = 0``.  The factor is **per-degree** and spans
    about 5.4x over ``n = 1..40`` (6.4e5 m at ``n = 1`` down to 1.2e5 m at
    ``n = 40``), so it can never be replaced by a constant -- and it must never
    be confused with the EWH factor ``A_n`` (they differ by ~1e7).
    """
    lln = None
    if love_numbers_l is None or love_numbers_k is None:
        lln = load_lln()
    if love_numbers_l is None:
        love_numbers_l = lln["l"]
    if love_numbers_k is None:
        love_numbers_k = load_love_numbers()
    ll = np.asarray(love_numbers_l, dtype=float).ravel()
    kl = np.asarray(love_numbers_k, dtype=float).ravel()
    ln = np.array([ll[i] if 0 <= i < ll.size else 0.0 for i in range(nmax + 1)])
    kn = np.array([kl[i] if 0 <= i < kl.size else 0.0 for i in range(nmax + 1)])
    return radius_m * ln / (1.0 + kn)


def _accumulate(lat, lon, C, S, nmax, scale, chunk,
                progress=None, cancel=None, label="球面梯度"):
    """Core sweep: returns ``(dS/dtheta, (1/sin) dS/dlambda)``.

    ``progress``/``cancel`` 是可选的（界面把这一步放到线程/子进程里跑时用）：
    每扫完一个（点块 × 阶）就报一次，取消抛 ``RuntimeError("cancelled")``。
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    if lat.size != lon.size:
        raise ValueError(f"lat{lat.shape} and lon{lon.shape} must match")

    single = C.ndim == 2
    C3 = C[:, :, None] if single else C
    S3 = S[:, :, None] if single else S
    ntime = C3.shape[2]

    if scale is None:
        sc = None
    else:
        sc = np.asarray(scale, dtype=float).ravel()[:nmax + 1]

    out_dth = np.empty((lat.size, ntime))
    out_dlm = np.empty((lat.size, ntime))

    step = lat.size if not chunk or chunk >= lat.size else max(int(chunk), 1)
    nchunk = (lat.size + step - 1) // step
    total = max(nchunk * (nmax + 1), 1)
    done = 0
    for s0 in range(0, lat.size, step):
        s1 = min(s0 + step, lat.size)
        la = lat[s0:s1]
        lam = np.deg2rad(lon[s0:s1])
        dth = np.zeros((s1 - s0, ntime))
        dlm = np.zeros((s1 - s0, ntime))
        for m, P, D, Q in legendre_columns_vec(la, nmax):
            _tick(progress, cancel,
                  f"{label}：阶 m={m}/{nmax}（点 {s1}/{lat.size}）",
                  done / total)
            done += 1
            n0 = m
            cc = C3[n0:nmax + 1, m, :]
            ss = S3[n0:nmax + 1, m, :]
            if sc is not None:
                w = sc[n0:nmax + 1][:, None]
                cc = cc * w
                ss = ss * w
            cm = np.cos(m * lam)[:, None]
            sm = np.sin(m * lam)[:, None]
            dth += (D.T @ cc) * cm                 # m = 0 also contributes here
            if m >= 1:
                dth += (D.T @ ss) * sm
                # (1/sin) dS/dlambda, factor m;  m = 0 has no east term
                dlm += m * ((Q.T @ ss) * cm - (Q.T @ cc) * sm)
        out_dth[s0:s1] = dth
        out_dlm[s0:s1] = dlm

    if single:
        return out_dth[:, 0], out_dlm[:, 0]
    return out_dth, out_dlm


def sph_gradient(lat_deg, lon_deg, C, S, nmax: Optional[int] = None, *,
                 chunk: Optional[int] = None, progress=None, cancel=None):
    """Horizontal gradient of the scalar SH field with coefficients ``C``/``S``.

    Returns ``(dS/dtheta, (1/sin theta) dS/dlambda)``, shaped like ``C``'s time
    axis (``(npoints,)`` for 2-D coefficients, ``(npoints, ntime)`` for 3-D).

    Pole-safe by construction: ``dS/dtheta`` comes from the differentiated
    Legendre recurrence (no ``1/sin``), and the eastern factor uses the exact
    ``Pbar/sin`` limit at the poles.
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    if C.ndim not in (2, 3):
        raise ValueError("C/S must be 2-D or 3-D (with a time axis)")
    L = C.shape[0] - 1
    if nmax is None or nmax > L:
        nmax = L
    return _accumulate(lat_deg, lon_deg, C, S, nmax, None, chunk,
                       progress=progress, cancel=cancel)


def horizontal_field(lat_deg, lon_deg, C, S, degree_factors, *,
                     chunk: Optional[int] = None, progress=None,
                     cancel=None) -> dict:
    """Horizontal displacement for coefficients already scaled per degree.

    This is the **fixture-level** entry point (see ``tools/make_horizontal_golden.py``):
    ``degree_factors`` is ``F^h_n`` (usually from
    :func:`degree_factors_horizontal`), so the returned ``north``/``east`` are in
    metres when the input coefficients are dimensionless geopotential ones.

    Works for **arbitrary points** (scattered or gridded, global or regional).
    For a regular whole-circle longitude grid prefer :func:`horizontal_grid`,
    which can take the FFT path.

    Returns a dict with ``north``, ``east``, ``magnitude``, ``azimuth``.
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    nmax = C.shape[0] - 1
    dth, dlm = _accumulate(lat_deg, lon_deg, C, S, nmax,
                           np.asarray(degree_factors, float), chunk,
                           progress=progress, cancel=cancel)
    return _derived(-dth, dlm)


def canonical_scaled(coeffs: SHCoeffs, *, love_numbers_l=None,
                     love_numbers_k=None, radius_m: float = EARTH_RADIUS_M):
    """``(C, S, F^h_n)``：换算到**经典无量纲位系数**并给出水平逐阶因子。

    **不做单位合法性检查**（各调用方保留自己的提示语）。这就是
    :func:`synthesis_horizontal` / :func:`synthesis_horizontal_grid` 内部那两步；
    界面把水平形变交给子进程算时也先调它 —— 于是线程路径与子进程路径拿到的输入
    **逐位相同**（子进程里只剩纯数值的梯度综合，不再碰单位）。
    """
    from . import units as _units

    src = _units.field_unit(coeffs)
    if src in ("scalar", "unknown"):
        x = coeffs
    else:
        x = _units.convert(coeffs, _units.CANONICAL, love_numbers=love_numbers_k)
    f = degree_factors_horizontal(x.nmax, love_numbers_l=love_numbers_l,
                                  love_numbers_k=love_numbers_k,
                                  radius_m=radius_m)
    return (np.asarray(x.C, dtype=float), np.asarray(x.S, dtype=float), f)


def synthesis_horizontal(lat_deg, lon_deg, coeffs: SHCoeffs, *,
                         want=("north", "east", "magnitude", "azimuth"),
                         love_numbers_l=None, love_numbers_k=None,
                         radius_m: float = EARTH_RADIUS_M,
                         allow_unit_mismatch: bool = False,
                         chunk: Optional[int] = None, progress=None,
                         cancel=None) -> dict:
    """Coefficients -> horizontal surface displacement (metres).

    ``coeffs`` may declare any physical unit: it is first converted to the
    canonical dimensionless geopotential coefficients (the only slot with a
    textbook load-Love-number formula), then scaled by ``F^h_n`` and
    differentiated.  A 3-D (multi-epoch) ``coeffs`` yields ``(npoints, ntime)``
    arrays for every requested component -- batching is free here, exactly as in
    :func:`shkit.synthesis.synthesis`.

    Undeclared (``scalar``/``unknown``) coefficients are **refused** by default,
    matching ``synthesis(target_unit=...)``: the output is a physical vector in
    metres, so silently assuming the input is dimensionless geopotential would be
    the classic "off by A_n" mistake.  Pass ``allow_unit_mismatch=True`` to force
    it, or declare the unit (``.gfc`` files are tagged automatically).

    Raises
    ------
    ValueError
        If the coefficients are declared ``horizontal_displacement`` (a vector
        cannot be a scalar input -- reversal is only possible up to an additive
        ``C00``), or if the unit is undeclared and ``allow_unit_mismatch`` is off.
    """
    from . import units as _units

    src = _units.field_unit(coeffs)
    if src == "horizontal_displacement":
        raise ValueError(
            "系数声明为「水平形变 u_h」——它是**矢量**，不能当作标量场综合。\n"
            "水平形变是位系数的球面梯度：由 (u_N, u_E) 反推位系数只到**差一个常数 "
            "C00**（常数的梯度恒为 0），分量也已丢失、不可逆。\n"
            "请分别导出北分量与东分量，或改用 field_unit=\"radial_displacement\"。")
    if src in ("scalar", "unknown") and not allow_unit_mismatch:
        raise ValueError(
            f"synthesis_horizontal 的单位不一致：源系数的物理含义是"
            f"「{_units.FIELD_UNIT_LABELS.get(src, src)}」。\n"
            "输出是**米**为单位的物理矢量的水平位移，所以源系数必须是"
            "无量纲重力位系数（或可换算到它的量：EWH / geoid / 面密度 ……）。\n"
            "读 .gfc 时会自动标成 geopotential；自己构造的系数请加 "
            ".with_unit(\"geopotential\")。\n"
            "若确实要强行按位系数处理，传 allow_unit_mismatch=True。")
    if src in ("scalar", "unknown"):
        x = coeffs
    else:
        x = _units.convert(coeffs, _units.CANONICAL, love_numbers=love_numbers_k)
    L = x.nmax
    f = degree_factors_horizontal(L, love_numbers_l=love_numbers_l,
                                  love_numbers_k=love_numbers_k,
                                  radius_m=radius_m)
    out = horizontal_field(lat_deg, lon_deg, x.C, x.S, f, chunk=chunk,
                           progress=progress, cancel=cancel)
    return {k: out[k] for k in want if k in out}


# ---------------------------------------------------------------------------
# Regular grids: the longitude FFT path
# ---------------------------------------------------------------------------
def _derived(u_n, u_e):
    mag = np.hypot(u_n, u_e)
    azi = np.degrees(np.arctan2(u_e, u_n)) % 360.0
    azi = np.where(azi >= 360.0, 0.0, azi)      # keep [0, 360) half-open (IEEE)
    return {"north": u_n, "east": u_e, "magnitude": mag, "azimuth": azi}


def _gradient_grid_fft(lat_vec, lon_vec, C3, S3, sc, nmax, time_chunk,
                       progress=None, cancel=None, label="球面梯度(FFT)"):
    """``u_N``/``u_E`` on a regular grid via one IFFT per component.

    On a whole-circle uniform longitude axis the ``m``-sum is a Fourier series in
    ``lambda``, so instead of evaluating it at every grid point we build the two
    complex spectra and transform once::

        A_m      = psi_m - i*chi_m            (psi_m = sum_n f_n Pbar_nm C_nm)
        G_N[m]   = -(psi'_m - i*chi'_m)        # u_N = -dS/dtheta
        G_E[m]   = i*m*(Qpsi_m - i*Qchi_m)     # u_E = (1/sin) dS/dlambda
        u        = nlon * irfft(G, n=nlon)     # G[0] full weight, G[m>=1] halved

    ``m = 0`` fills ``G_N`` **only** -- ``dPbar_n0/dtheta != 0`` but the east
    component carries a factor ``m``, so dropping ``m = 0`` makes ``u_N``
    systematically wrong (measured 31% in the reference SHSynth prototype).

    Cost drops from ``O(nSHCS*nlat*nlon*ntime)`` to
    ``O(nSHCS*nlat*ntime) + O(nlat*ntime*nlon*log nlon)``.
    """
    nlat, nlon = lat_vec.size, lon_vec.size
    ntime = C3.shape[2]
    nf = nlon // 2 + 1
    u_n = np.empty((nlat, nlon, ntime))
    u_e = np.empty_like(u_n)
    # e^{i m lambda_0}: lambda_k = lambda_0 + 2*pi*k/nlon, and irfft only ever
    # sees the second term.  Without this the path is exact for lon = 0,1,2,...
    # and wrong by O(1) for any other axis -- see shkit/lonfft.longitude_phase.
    ph = longitude_phase(lon_vec[0], nmax)
    step = int(time_chunk) if time_chunk else ntime
    nchunk = (ntime + step - 1) // step
    total = max(nchunk * (nmax + 1), 1)
    done = 0
    for t0 in range(0, ntime, step):
        t1 = min(t0 + step, ntime)
        ntc = t1 - t0
        g_n = np.zeros((nf, nlat, ntc), dtype=complex)
        g_e = np.zeros((nf, nlat, ntc), dtype=complex)
        for m, P, D, Q in legendre_columns_vec(lat_vec, nmax):
            _tick(progress, cancel,
                  f"{label}：阶 m={m}/{nmax}（时次 {t1}/{ntime}）",
                  done / total)
            done += 1
            cc = C3[m:nmax + 1, m, t0:t1]
            ss = S3[m:nmax + 1, m, t0:t1]
            if sc is not None:
                w = sc[m:nmax + 1][:, None]
                cc = cc * w
                ss = ss * w
            if m == 0:
                # A_0 = psi_0 (real): S_n0 contributes nothing (sin(0*lambda) == 0)
                g_n[0] = -(D.T @ cc)
                continue
            g_n[m] = ph[m] * -((D.T @ cc) - 1j * (D.T @ ss)) / 2.0
            g_e[m] = ph[m] * 1j * m * ((Q.T @ cc) - 1j * (Q.T @ ss)) / 2.0
        u_n[:, :, t0:t1] = (nlon * np.fft.irfft(g_n, n=nlon, axis=0)).transpose(1, 0, 2)
        u_e[:, :, t0:t1] = (nlon * np.fft.irfft(g_e, n=nlon, axis=0)).transpose(1, 0, 2)
    if ntime == 1:
        return u_n[:, :, 0], u_e[:, :, 0]
    return u_n, u_e


def horizontal_grid(lat_vec, lon_vec, C, S, degree_factors, *,
                    method: str = "auto", report: Optional[dict] = None,
                    time_chunk: Optional[int] = None, progress=None,
                    cancel=None) -> dict:
    """Horizontal displacement on a **regular lat/lon grid**.

    ``method``:

    ``'auto'`` (default)
        use the longitude FFT when :func:`fft_path_applicable` allows it,
        otherwise the direct sweep.  The choice and its reason are written into
        ``report`` -- never switch silently, or the same input would show an
        unexplained speed difference on another machine.
    ``'direct'``
        always the direct sweep (the universal path: any geometry).
    ``'fft'``
        require the FFT path; raises if it does not apply (no silent fallback).

    ``report``, if given, receives ``path`` / ``reason`` / ``nlon`` / ``nmax``.

    Returns ``north``/``east``/``magnitude``/``azimuth`` shaped ``(nlat, nlon)``
    or ``(nlat, nlon, ntime)``.
    """
    lat_vec = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon_vec = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    if C.ndim not in (2, 3):
        raise ValueError("C/S must be 2-D or 3-D (with a time axis)")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C/S must be square in (degree, order)")
    L = C.shape[0] - 1
    C3 = C[:, :, None] if C.ndim == 2 else C
    S3 = S[:, :, None] if S.ndim == 2 else S
    sc = None if degree_factors is None else \
        np.asarray(degree_factors, dtype=float).ravel()[:L + 1]

    ok, why = fft_path_applicable(lon_vec, L)
    if method == "auto":
        method = "fft" if ok else "direct"
    elif method == "fft" and not ok:
        raise ValueError(
            f"method='fft' 在这个经度轴上不适用：{why}\n"
            "（要么改用 method='auto'/'direct'，要么换一张整圈均匀的经度网格）")
    elif method not in ("fft", "direct"):
        raise ValueError(f"method must be 'auto'|'direct'|'fft', got {method!r}")

    if method == "fft":
        u_n, u_e = _gradient_grid_fft(lat_vec, lon_vec, C3, S3, sc, L, time_chunk,
                                      progress=progress, cancel=cancel)
    else:
        LA, LO = np.meshgrid(lat_vec, lon_vec, indexing="ij")
        # pass the ORIGINAL C/S so _accumulate collapses the time axis itself
        dth, dlm = _accumulate(LA.ravel(), LO.ravel(), C, S, L, sc, None,
                               progress=progress, cancel=cancel)
        shape = ((lat_vec.size, lon_vec.size) if C.ndim == 2
                 else (lat_vec.size, lon_vec.size, C.shape[2]))
        u_n = (-dth).reshape(shape)
        u_e = dlm.reshape(shape)

    if report is not None:
        report.update(path=method, reason=why, nlon=int(lon_vec.size), nmax=int(L))
    return _derived(u_n, u_e)


def synthesis_horizontal_grid(lat_vec, lon_vec, coeffs: SHCoeffs, *, method="auto",
                              report: Optional[dict] = None, want=("north", "east",
                                                                    "magnitude", "azimuth"),
                              love_numbers_l=None, love_numbers_k=None,
                              radius_m: float = EARTH_RADIUS_M,
                              allow_unit_mismatch: bool = False,
                              time_chunk: Optional[int] = None, progress=None,
                              cancel=None) -> dict:
    """``synthesis_horizontal`` on a regular grid (same unit guards)."""
    from . import units as _units

    src = _units.field_unit(coeffs)
    if src == "horizontal_displacement":
        raise ValueError(
            "系数声明为「水平形变 u_h」——它是**矢量**，不能当作标量场综合（见 "
            "docs/水平形变契约.md F11）。")
    if src in ("scalar", "unknown") and not allow_unit_mismatch:
        raise ValueError(
            f"synthesis_horizontal_grid 的单位不一致：源系数是"
            f"「{_units.FIELD_UNIT_LABELS.get(src, src)}」。\n"
            "请用 .with_unit(\"geopotential\") 声明，或传 allow_unit_mismatch=True。")
    if src in ("scalar", "unknown"):
        x = coeffs
    else:
        x = _units.convert(coeffs, _units.CANONICAL, love_numbers=love_numbers_k)
    f = degree_factors_horizontal(x.nmax, love_numbers_l=love_numbers_l,
                                  love_numbers_k=love_numbers_k, radius_m=radius_m)
    out = horizontal_grid(lat_vec, lon_vec, x.C, x.S, f, method=method,
                          report=report, time_chunk=time_chunk,
                          progress=progress, cancel=cancel)
    return {k: out[k] for k in want if k in out}
