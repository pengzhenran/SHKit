# -*- coding: utf-8 -*-
"""
shkit.lonfft
============

**经度方向的 FFT 加速** —— 综合/水平形变（B3）与分析（A3）共用同一套判据与
同一套相位口径。

为什么只加速经度
----------------
球谐函数在经度上本来就是三角级数：``Pbar_nm(sin phi) * (cos m*lambda,
sin m*lambda)``。只要经度是**整圈均匀**采样，把沿经度的求和换成一次 FFT
就得到全部阶 ``m`` 的傅里叶系数，不需要在每个格点上显式乘 ``cos/sin``：

.. math::

    G_j[m] = \\sum_k g_{jk} e^{-i 2\\pi m k / n_{lon}}

于是对 ``m >= 1``

.. math::

    \\sum_k g_{jk}\\cos(m\\lambda_k) &= \\cos\\phi_m\\,\\Re G_j[m] + \\sin\\phi_m\\,\\Im G_j[m] \\\\
    \\sum_k g_{jk}\\sin(m\\lambda_k) &= \\sin\\phi_m\\,\\Re G_j[m] - \\cos\\phi_m\\,\\Im G_j[m]

其中 ``phi_m = m * lambda_0`` 是**起始经度相位**（``lambda_0`` 是网格第一条经线，
不一定是 0，例如 ``0.5`` 或 ``-179.5``）。这一项容易漏：漏掉时 ``lambda_0 = 0``
的网格仍然完全正确，只有带偏移的网格会整体错乱，所以回归用例里必须放一个
``lambda_0 != 0`` 的网格。

纬度方向**不做**任何假设：FFT 只吃掉经度那一层求和，剩下的
``Pbar_nm(sin phi_j)`` 逐行加权求和照旧。因此

* 极点的精确处理、``nlon > 2*nmax`` 之外的混叠判据都不受影响；
* 权重**不需要**与经度可分离。分析方向求的是
  :math:`\\sum_k (w_{jk} f_{jk}) \\cos m\\lambda_k`，所以先乘权重、再对
  ``g = w*f`` 做 FFT 即可，权重随经度变化也照样精确。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "fft_path_applicable",
    "longitude_phase",
    "LongitudeGrid",
    "LongitudePlan",
    "longitude_grid_layout",
    "plan_longitude",
]


def longitude_phase(lam0_deg: float, nmax: int) -> np.ndarray:
    """``e^{i m lambda_0}`` for ``m = 0..nmax`` -- the offset-grid phase.

    A whole-circle axis does **not** have to start at ``0``.  Both the analysis
    and the synthesis direction express the field as a Fourier series in
    ``lambda_k = lambda_0 + 2*pi*k/n``, and one ``rfft``/``irfft`` only ever sees
    the ``2*pi*k/n`` part.  The ``e^{i m lambda_0}`` rotation has to be applied to
    the spectrum explicitly, and **dropping it is invisible on a grid that starts
    at 0**: the same code is exactly right there and wrong by O(1) on any other
    axis.

    Measured on the horizontal path before this was fixed (``nlat=37``,
    ``nmax=20``, comparing FFT against the direct sweep)::

        lam0 =     0   ->  relative 3.0e-15   (looks perfect)
        lam0 =   0.5   ->  relative 5.8e-02
        lam0 = -179.5  ->  relative 1.2e+00
        lam0 =  90.3   ->  relative 1.5e+00

    Hence the offset-grid regressions in ``tests/validate_lonfft.py`` and
    ``tests/validate_horizontal.py``.
    """
    return np.exp(1j * np.deg2rad(float(lam0_deg)) * np.arange(int(nmax) + 1))



def fft_path_applicable(lon_deg, nmax: int, *, rtol: float = 1e-9):
    """Can the **longitude FFT** path be used for this longitude axis?

    Returns ``(ok, reason)`` -- ``reason`` is Chinese and meant to be printed, so
    callers never switch algorithms silently.

    The FFT acts on **longitude only**, so it asks nothing about latitude (a
    latitude *belt* is fine).  It needs:

    1. a strictly increasing, **uniformly spaced**, **whole-circle** axis
       (step ``d`` with ``lon[-1] + d - lon[0] == 360``);
    2. ``nlon > 2*nmax`` -- otherwise order ``m`` exceeds Nyquist and folds into
       low frequencies.  That is not "slightly wrong", it is aliasing, so it is a
       hard requirement.
    """
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    nlon = lon.size
    if nlon < 2:
        return False, "经度点少于 2 个，无法做经度方向的变换"
    d = np.diff(lon)
    if not np.all(d > 0):
        return False, "经度不是严格递增（请先排序并去重）"
    if not np.allclose(d, d[0], rtol=rtol, atol=1e-9):
        return False, "经度不是等间隔：只有均匀整圈采样才能用 FFT"
    span = float(lon[-1] + d[0] - lon[0])
    if not np.isclose(span, 360.0, rtol=rtol, atol=1e-6):
        return False, (f"经度只覆盖 {span:g}°，不是整圈 360°；"
                       "区域经度请用直接法（结果一样，只是慢）")
    if nlon <= 2 * nmax:
        return False, (f"nlon={nlon} ≤ 2·nmax={2 * nmax}：m 会超过 Nyquist 折叠（混叠），"
                       "必须回退直接法")
    return True, f"整圈均匀经度 {nlon} 点，满足 nlon > 2·nmax={2 * nmax}"


@dataclass(frozen=True)
class LongitudeGrid:
    """A complete rectangular ``(lat, lon)`` grid recognised inside a point set.

    ``order`` is the permutation that turns the *original* flat point arrays into
    row-major grid arrays::

        g_flat[order].reshape(nlat, nlon)   # == g_grid

    It is ``None`` when the caller already holds the axes themselves (e.g.
    :func:`shkit.synthesis.synthesis_grid`), in which case the data is a
    ``(nlat, nlon, ...)`` array to begin with and no permutation is involved.

    ``lam0_deg`` is the first longitude of the axis (the phase reference the FFT
    needs); ``lat_vec``/``lon_vec`` are the sorted unique axes.
    """

    nlat: int
    nlon: int
    lat_vec: np.ndarray
    lon_vec: np.ndarray
    order: np.ndarray
    lam0_deg: float

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nlat, self.nlon)

    def to_grid(self, values: np.ndarray) -> np.ndarray:
        """``(npoints, ...)`` -> ``(nlat, nlon, ...)`` using :attr:`order`."""
        v = np.asarray(values)
        rest = v.shape[1:]
        return v[self.order].reshape(self.nlat, self.nlon, *rest)

    def from_grid(self, values: np.ndarray) -> np.ndarray:
        """``(nlat, nlon, ...)`` -> ``(npoints, ...)``: inverse of :meth:`to_grid`."""
        v = np.asarray(values)
        rest = v.shape[2:]
        inv = np.empty(self.order.size, dtype=self.order.dtype)
        inv[self.order] = np.arange(self.order.size)
        return v.reshape(self.nlat * self.nlon, *rest)[inv]


@dataclass(frozen=True)
class LongitudePlan:
    """Which longitude kernel to use, and why.

    ``reason`` is Chinese and printable on purpose: the two paths give the same
    numbers (measured <=1e-12 relative), so a silent switch would only ever show
    up as an unexplained speed difference on someone else's machine.
    """

    path: str                      # 'fft' | 'direct'
    reason: str
    grid: Optional[LongitudeGrid] = None

    def meta(self) -> dict:
        out = {"longitude_path": self.path, "longitude_reason": self.reason}
        if self.grid is not None:
            out["longitude_grid"] = {
                "nlat": self.grid.nlat, "nlon": self.grid.nlon,
                "lam0_deg": self.grid.lam0_deg}
        return out


def plan_longitude(lat_deg, lon_deg, nmax: int, mode: str = "auto") -> LongitudePlan:
    """Decide between the direct sweep and the longitude FFT.

    ``mode`` is ``'auto'`` / ``'fft'`` / ``'direct'``.  ``lat_deg``/``lon_deg`` are
    a flat point set and a complete rectangular grid is looked for inside it.
    ``mode='fft'`` **raises** when the geometry does not allow the fast path --
    never a silent fallback.

    Shared by :mod:`shkit.analysis` (A3) and :mod:`shkit.synthesis` (A4) so the
    two cannot drift apart on what "usable longitude axis" means.
    """
    if mode not in ("auto", "fft", "direct"):
        raise ValueError("longitude_fft must be 'auto'|'fft'|'direct', "
                         f"got {mode!r}")
    if mode == "direct":
        return LongitudePlan("direct", "用户指定 longitude_fft='direct'")
    grid, why_grid = longitude_grid_layout(lat_deg, lon_deg)
    if grid is None:
        if mode == "fft":
            raise ValueError(
                f"longitude_fft='fft' 不适用：{why_grid}\n"
                "（经度 FFT 只吃经度那一层求和，要求点集是**完整矩形网格**；"
                "散点/区域掩膜请用 longitude_fft='auto' 或 'direct'，"
                "结果一样，只是慢）")
        return LongitudePlan("direct", why_grid)
    ok, why = fft_path_applicable(grid.lon_vec, nmax)
    if not ok:
        if mode == "fft":
            raise ValueError(
                f"longitude_fft='fft' 不适用：{why}\n"
                "（这是**混叠**级别的硬条件，不能放宽：nlon 必须 > 2·nmax。"
                "请改用 'auto'/'direct'，或换更密的经度网格）")
        return LongitudePlan("direct", why)
    return LongitudePlan("fft", f"{why_grid}；{why}", grid)


def longitude_grid_layout(lat_deg, lon_deg, *, rtol: float = 1e-9,
                          atol: float = 1e-9) -> Tuple[Optional[LongitudeGrid], str]:
    """Recognise a **complete rectangular lat/lon grid** inside a point set.

    Returns ``(layout, reason)``; ``layout`` is ``None`` when the points are not
    such a grid and ``reason`` says why (Chinese, printable).

    This is deliberately stricter than
    :func:`shkit.weights.looks_like_lattice`: a *lattice subset* (a rasterised
    mask) is a perfectly good analysis input, but it has no full set of
    longitudes, so there is nothing to transform.  The checks are

    1. ``n_unique(lat) * n_unique(lon) == n_points`` -- no missing cell;
    2. every point matches an axis value exactly (no near-duplicate coordinates);
    3. no ``(lat, lon)`` combination appears twice.

    Whether the axis is then *usable* for the FFT (uniform, whole circle,
    ``nlon > 2*nmax``) is :func:`fft_path_applicable`'s question, asked by the
    caller with the ``nmax`` actually in play.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    if lat.size != lon.size:
        return None, f"纬度 {lat.size} 个、经度 {lon.size} 个，长度不一致"
    if lat.size < 4:
        return None, f"只有 {lat.size} 个点，构不成网格"
    lat_u = np.unique(lat)
    lon_u = np.unique(lon)
    ncell = lat_u.size * lon_u.size
    if ncell != lat.size:
        return None, (f"{lat.size} 个点不是完整矩形网格：唯一纬度 {lat_u.size} × "
                      f"唯一经度 {lon_u.size} = {ncell} ≠ {lat.size}")
    i = np.searchsorted(lat_u, lat)
    j = np.searchsorted(lon_u, lon)
    if not (np.allclose(lat_u[i], lat, rtol=rtol, atol=atol)
            and np.allclose(lon_u[j], lon, rtol=rtol, atol=atol)):
        return None, "存在与网格轴不严格相等的坐标（近似重复点），拒绝按网格处理"
    key = i * lon_u.size + j
    if np.unique(key).size != lat.size:
        return None, "同一个 (纬度, 经度) 组合出现了多次（网格有重复点）"
    order = np.argsort(key, kind="stable")
    return (LongitudeGrid(nlat=int(lat_u.size), nlon=int(lon_u.size),
                          lat_vec=lat_u, lon_vec=lon_u, order=order,
                          lam0_deg=float(lon_u[0])),
            f"识别为 {lat_u.size}×{lon_u.size} 完整矩形网格")
