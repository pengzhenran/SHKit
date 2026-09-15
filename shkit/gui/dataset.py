# -*- coding: utf-8 -*-
"""
shkit.gui.dataset
=================

A uniform in-memory view of "whatever the user loaded", so the analysis code
does not care whether the input was a scattered point table or a gridded field.

Both kinds are exposed as flat sample vectors ``(lat, lon, values)`` because
that is what :func:`shkit.analysis.analysis` consumes; a grid additionally keeps
its structured form so it can be re-displayed and re-exported as a grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["Dataset"]


@dataclass
class Dataset:
    kind: str                       # 'points' | 'grid'
    path: str
    lat: np.ndarray                 # (N,) sample latitudes, degrees
    lon: np.ndarray                 # (N,) sample longitudes, degrees
    #: (N,) 或 (N, ntime)。**懒加载的多时次网格这里是 None** —— 上一层的
    #: (nlat, nlon, ntime) 立方体还没读全，硬展平会立刻多占一倍内存。要完整
    #: 矩阵请用 all_values()（它会按需补齐），要某一层用 value_slice(t)。
    values: Optional[np.ndarray] = None
    lat_vec: Optional[np.ndarray] = None
    lon_vec: Optional[np.ndarray] = None
    #: (nlat, nlon) 或 (nlat, nlon, ntime)；也可能是 shkit.io.LazyTimeCube
    grid: Optional[np.ndarray] = None
    times: Optional[object] = None         # TimeAxis when the epochs carry dates
    meta: dict = field(default_factory=dict)
    #: 统计量与摘要的缓存（按**时次**索引；大文件上重复计算很贵，见 stats()）
    _stats: dict = field(default_factory=dict, repr=False, compare=False)
    _summary: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------- builders
    @classmethod
    def from_points(cls, path: str, lat, lon, values, meta=None,
                    times=None) -> "Dataset":
        lat = np.asarray(lat, dtype=float).ravel()
        lon = np.asarray(lon, dtype=float).ravel()
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return cls(kind="points", path=str(path), lat=lat, lon=lon,
                   values=values, times=times, meta=dict(meta or {}))

    @classmethod
    def from_grid(cls, path: str, lat_vec, lon_vec, grid, meta=None,
                  times=None) -> "Dataset":
        lat_vec = np.asarray(lat_vec, dtype=float).ravel()
        lon_vec = np.asarray(lon_vec, dtype=float).ravel()
        lazy = hasattr(grid, "ensure") and hasattr(grid, "start_prefetch")
        if lazy:
            # ⚠️ 这里**不能** np.asarray(grid)：那正是"打开就等 3.5 s"的来源。
            # 首屏只要第 0 层（LazyTimeCube 构造时已读好），其余交给后台补数；
            # 真要全部历元的操作显式调 all_values() / ensure_all_epochs()。
            g = grid
            values = None
        else:
            g = np.asarray(grid, dtype=float)
            if g.ndim == 2:
                g = g[:, :, None]
            values = g.reshape(-1, g.shape[2])
        LA, LO = np.meshgrid(lat_vec, lon_vec, indexing="ij")
        return cls(kind="grid", path=str(path),
                   lat=LA.ravel(), lon=LO.ravel(), values=values,
                   lat_vec=lat_vec, lon_vec=lon_vec, grid=g, times=times,
                   meta=dict(meta or {}))

    # --------------------------------------------------------------- epochs
    def time_axis(self):
        """The :class:`~shkit.timeaxis.TimeAxis` for the epochs, if known.

        Built on demand from ``meta['time']`` (what :func:`shkit.io.read_grid`
        records for a netCDF ``time`` coordinate) so a loaded grid can show real
        dates without the reader having to know about the GUI.  The decoding
        itself lives in :func:`shkit.timeaxis.time_axis_from_meta` -- including
        the case of a **numeric** time coordinate (``days since …``), which some
        real products (CSR mascons, attribute ``Units`` with a capital U) use and
        which used to end up here as "no time information".
        """
        if self.times is not None:
            return self.times
        raw = self.meta.get("time") or self.meta.get("times")
        if raw is None or len(raw) != self.ntime:
            return None
        from ..timeaxis import time_axis_from_meta
        ax = time_axis_from_meta(raw, units=self.meta.get("time_units"),
                                 calendar=self.meta.get("time_calendar"))
        if ax is None or not ax.has_dates:
            return None
        self.times = ax
        return ax

    def epoch_label(self, t: int) -> str:
        """``'第 k/N 个时次（2002-04-18）'``, or just the index when undated."""
        t = int(min(max(t, 0), max(self.ntime - 1, 0)))
        lbl = f"第 {t + 1}/{self.ntime} 个时次"
        ax = self.time_axis()
        if ax is not None and ax.has_dates:
            lbl += f"（{str(ax.values[t])[:10]}）"
        return lbl

    # ------------------------------------------------------------ accessors
    @property
    def npoints(self) -> int:
        return int(self.lat.size)

    @property
    def ntime(self) -> int:
        if self.values is not None:
            return int(self.values.shape[1])
        if self.grid is not None:
            return int(self.grid.shape[2])
        return 0

    def value_slice(self, t: int = 0) -> np.ndarray:
        """第 ``t`` 个时次的 ``(N,)`` 样本向量（懒加载时只读这一层）。"""
        if self.values is not None:
            return self.values[:, t]
        return np.asarray(self.grid[:, :, t]).reshape(-1)

    def all_values(self) -> np.ndarray:
        """完整 ``(N, ntime)`` 样本矩阵。

        ⚠️ 懒加载的网格在这里会**补齐全部时次**（后台补数还没完就同步等），
        所以调用方最好先 :meth:`ensure_all_epochs` 走一遍进度提示。
        """
        if self.values is not None:
            return self.values
        arr = np.asarray(self.grid)          # LazyTimeCube.__array__ → ensure_all()
        return arr.reshape(-1, arr.shape[2])

    def has_time(self) -> bool:
        return self.ntime > 1

    def suggested_rule(self) -> str:
        """A sensible default integration-element rule for this dataset."""
        from ..weights import looks_like_lattice

        lon = self.lon
        span = float(np.unique(lon).max() - np.unique(lon).min())
        if self.kind == "grid":
            nlat_u = np.unique(self.lat).size
            nlon_u = np.unique(lon).size
            if span > 300.0 and nlon_u == 2 * nlat_u and nlat_u % 2 == 0:
                return "dh"
            return "grid"
        if span > 300.0 and self.lat.max() > 60.0 and self.lat.min() < -60.0:
            return "voronoi"
        # a mask sampled on a regular lattice: each point is one lattice cell.
        # 'delaunay' would report the convex hull (67x too big on the Yangtze
        # river mask) and 'grid' would let points next to a missing row absorb
        # half of that gap (+4.59%); both errors land straight in C00.
        if looks_like_lattice(self.lat, lon):
            return "lattice"
        return "delaunay"

    def grid_slice(self, t: int):
        """第 ``t`` 个时次的 ``(nlat, nlon)`` 场。

        懒加载的多时次 nc 会在这里**按需读那一个时次**（约 0.25 s）—— 这正是
        "切换时次/播放"该有的粒度；已经读过的（或后台补齐的）是内存里的视图。
        """
        t = int(min(max(t, 0), max(self.ntime - 1, 0)))
        if self.grid is None:
            return None
        if hasattr(self.grid, "ensure"):                # LazyTimeCube
            self.grid.ensure(t)
        return self.grid[:, :, t]

    def ensure_all_epochs(self, progress=None) -> None:
        """要把**全部历元**读完的操作（批量分析/趋势拟合）先调这个。

        懒加载时它是"补齐"（后台可能已经在补）；非懒加载时是空操作。
        """
        if self.grid is not None and hasattr(self.grid, "ensure_all"):
            self.grid.ensure_all(progress=progress)

    def lazy_loader(self):
        """懒加载容器（没有就返回 None），供界面显示"后台补齐进度"。"""
        g = self.grid
        return g if (g is not None and hasattr(g, "start_prefetch")) else None

    def close(self) -> None:
        """释放懒加载持有的文件句柄与 ASCII 临时副本。"""
        g = self.grid
        if g is not None and hasattr(g, "close"):
            try:
                g.close()
            except Exception:                                    # noqa: BLE001
                pass

    def stats(self, t: int = 0) -> dict:
        """第 ``t`` 个时次的 `(n_finite, n_total, min, max, rms)`，**按需 + 缓存**。

        ⚠️ 口径是**逐时次**（用户要求）：多时次文件（CSR mascon 2.65 亿个值）若对
        整块求范围/RMS，光统计就要 3.4 s（``v ** 2`` 还再开一份 2 GB 临时数组），
        而那个数字对"我正在看的这一层"也没有直接意义。现在只统计**当前时次**
        （1 M 个值，毫秒级），并在摘要里写明是第几个时次；单时次数据退化为整块。
        """
        t = int(min(max(t, 0), max(self.ntime - 1, 0))) if self.ntime else 0
        hit = self._stats.get(t)
        if hit is not None:
            return hit
        # ⚠️ 多时次**散点**也要取那一列（``(N, ntime)``）：以前这里直接用整块
        # ``values``，于是摘要会一边写着「第 k 个时次」一边给整块的范围 —— 那是
        # 假的（mascon 的网格那条路走 grid_slice，散点这条路当年漏了）。
        if self.kind == "grid":
            v = self.grid_slice(t)
        elif self.values is not None and self.values.ndim > 1 \
                and self.values.shape[1] > 1:
            v = self.values[:, t]
        else:
            v = self.values
        if v is None:
            v = self.values
        v = np.asarray(v)
        n_total = int(v.size)
        n_finite = 0
        lo, hi = np.inf, -np.inf
        s2 = 0.0
        # 单层（≤2 维）直接算；多时次按最后一维分块，临时数组只有一块
        chunks = ((v,) if v.ndim <= 2
                  else (v[..., k] for k in range(v.shape[-1])))
        for ch in chunks:
            ch = np.ravel(ch)
            fin = np.isfinite(ch)
            if not fin.all():
                ch = ch[fin]
            n_finite += int(ch.size)
            if ch.size:
                lo = min(lo, float(ch.min()))
                hi = max(hi, float(ch.max()))
                ch64 = ch.astype(np.float64, copy=False)
                s2 += float(np.dot(ch64, ch64))
        rms = float(np.sqrt(s2 / n_finite)) if n_finite else float("nan")
        st = {"n_finite": n_finite, "n_total": n_total,
              "min": lo if n_finite else float("nan"),
              "max": hi if n_finite else float("nan"), "rms": rms, "epoch": t}
        self._stats[t] = st
        return st

    def summary(self, t: int = 0) -> str:
        """载入后给用户看的摘要（**按第 ``t`` 个时次**统计，结果按 t 缓存）。"""
        t = int(min(max(t, 0), max(self.ntime - 1, 0))) if self.ntime else 0
        hit = self._summary.get(t)
        if hit is not None:
            return hit
        st = self.stats(t)
        lines = [
            f"类型        : {'散点' if self.kind == 'points' else '规则网格'}",
            f"文件        : {self.path}",
            f"点数        : {self.npoints}",
            f"时次        : {self.ntime}",
        ]
        # 多时次数据必须在这里说清"时间轴认出来了没有"：认出来就给日期范围，
        # 没认出来就**明说只有序号**。否则用户只看到"时次 256"，到趋势页才发现
        # 没有日期，却不知道是文件没有还是读不懂。
        if self.ntime > 1:
            ax = self.time_axis() if self.has_time() else None
            if ax is not None and ax.has_dates:
                lines.append(f"时间轴      : {str(ax.values[0])[:10]} … "
                             f"{str(ax.values[-1])[:10]}"
                             f"（{ax.meta.get('time_source')}）")
            else:
                src = self.meta.get("time_units")
                lines.append("时间轴      : 未识别到日期（只有历元序号）"
                             + (f"，文件里写的时间单位是 {src!r}" if src else ""))
        lines += [
            f"纬度范围    : {np.nanmin(self.lat):.4f} … {np.nanmax(self.lat):.4f} °",
            f"经度范围    : {np.nanmin(self.lon):.4f} … {np.nanmax(self.lon):.4f} °",
        ]
        if self.kind == "grid":
            lines.append(f"网格        : {self.lat_vec.size} (纬) x "
                         f"{self.lon_vec.size} (经)")
        if st["n_finite"]:
            scope = (f"第 {t + 1}/{self.ntime} 个时次" if self.ntime > 1
                     else "整块")
            lines += [
                f"数值范围    : {st['min']:.6g} … {st['max']:.6g}"
                f"　（{scope}）",
                f"数值 RMS    : {st['rms']:.6g}　（{scope}）",
            ]
            if st["n_finite"] < st["n_total"]:
                lines.append(f"缺测        : {st['n_total'] - st['n_finite']} / "
                             f"{st['n_total']} 个值为 NaN（{100.0 * (1 - st['n_finite'] / st['n_total']):.3g}%）")
        else:
            lines.append("数值范围    : 全为 NaN")
        # 文件里声明的变量单位。注意措辞：SHKit 的正/反变换与各物理量之间的换算
        # **全都是线性的**，所以输入什么单位、输出就是同一套单位的倍数 ——
        # 往返与各物理量之比都不受影响，**不需要换算**。只有打印出来的绝对量
        # （m / mm）以及"真实 GRACE 应在 mm 级"这类按米写的量级自检要按文件单位读。
        vu = str(self.meta.get("variable_units") or "").strip()
        if vu:
            note = ""
            if vu.lower() in ("cm", "mm", "km"):
                note = ("（线性换算，往返与各物理量之比不受影响；"
                        "但打印的绝对量 m/mm 要按此单位读）")
            lines.append(f"变量单位    : {vu}{note}")
        if self.meta.get("warnings"):
            for w in self.meta["warnings"][:6]:
                lines.append(f"提示        : {w}")
        self._summary[t] = "\n".join(lines)
        return self._summary[t]
