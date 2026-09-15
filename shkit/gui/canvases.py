# -*- coding: utf-8 -*-
"""
shkit.gui.canvases
==================

matplotlib canvases embedded in Qt widgets.

Plotting uses **matplotlib** and never Qt Charts: ``QtCharts`` and
``QtDataVisualization`` are GPL-only Qt modules, which would make a closed-source
commercial build impossible.  matplotlib is BSD/PSF-style and carries no such
restriction.  This is a deliberate licensing decision - see
``docs/许可与闭源商用说明.md``.

Coastlines come from a bundled Natural Earth 110m extract
(``SHKit/data/coastline_110m.npz``, public domain), so maps work offline with no
cartopy download on first use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

import matplotlib.patheffects as pe
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

__all__ = ["MapCanvas", "SpectrumCanvas", "load_coastlines", "wrap_longitude",
           "setup_matplotlib_fonts"]

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data")
_COAST_CACHE: dict = {}
_COAST_ERROR: list = []

# matplotlib's bundled DejaVu Sans has no CJK glyphs, so every Chinese axis
# label would render as a row of boxes.  Pick the first CJK font actually
# installed on this machine.
_CJK_CANDIDATES = ("Microsoft YaHei", "SimHei", "SimSun",
                   "Noto Sans CJK SC", "Source Han Sans SC",
                   "PingFang SC", "Hiragino Sans GB", "WenQuanYi Zen Hei",
                   "Arial Unicode MS", "MS Gothic")


def setup_matplotlib_fonts(verbose: bool = False) -> list:
    """Make matplotlib able to draw Chinese text.  Idempotent."""
    import matplotlib
    from matplotlib import font_manager

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    chosen = [n for n in _CJK_CANDIDATES if n in available]
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False   # U+2212 is not in CJK fonts
    if verbose:
        print(f"matplotlib CJK font: {chosen or '未找到，中文可能显示为方框'}")
    return chosen


setup_matplotlib_fonts()


def load_coastlines(which: str = "coastline"):
    """Return ``(lon, lat)`` polyline arrays with NaN separators, or None.

    Reads the bundled Natural Earth 110m extract, so maps work offline with no
    cartopy download on first use.
    """
    if which in _COAST_CACHE:
        return _COAST_CACHE[which]
    path = os.path.join(_DATA_DIR, f"{which}_110m.npz")
    try:
        with np.load(path, allow_pickle=False) as z:
            val = (np.asarray(z["lon"], dtype=float),
                   np.asarray(z["lat"], dtype=float))
    except Exception as exc:                      # noqa: BLE001
        if not _COAST_ERROR:
            _COAST_ERROR.append(f"{path}: {type(exc).__name__}: {exc}")
        val = None
    _COAST_CACHE[which] = val
    return val


def wrap_longitude(lon: np.ndarray) -> np.ndarray:
    """Map longitudes to ``[-180, 180)``."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def prepare_polyline(lon, lat, lon_min: float = -180.0):
    """Wrap longitudes and **break** the polyline at every seam.

    Wrapping each vertex independently is not enough: a vertex sitting exactly on
    ``+180`` maps to ``-180``, which turns a harmless 1-degree step at the
    antimeridian into a 359-degree jump.  matplotlib then draws a line straight
    across the map - the classic "flying coastline".

    Measured on the bundled Natural Earth 110 m extract: the raw data contains
    no jumps at all, but naive wrapping creates **7** spurious 359-degree
    segments in the coastline (and none in the borders, which is why only the
    coastline looked broken).

    This function inserts NaN separators wherever the wrapped longitude jumps by
    more than 180 degrees, so the renderer breaks the line instead of crossing
    the map.  Existing NaN separators are preserved.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    bad = np.isnan(lon) | np.isnan(lat)

    w = (np.where(bad, 0.0, lon) + 180.0) % 360.0 - 180.0
    if lon_min not in (-180.0, 180.0):          # support a 0..360 target too
        w = np.where(bad, 0.0, w % 360.0)

    jump = np.zeros(w.size, dtype=bool)
    if w.size > 1:
        jump[1:] = np.abs(np.diff(w)) > 180.0
    broken = bad | jump

    n_add = int(broken.sum())
    out_lon = np.full(w.size + n_add, np.nan)
    out_lat = np.full(w.size + n_add, np.nan)
    if w.size:
        # ``broken[i]`` means "put a NaN row immediately before vertex i", so
        # vertex i shifts by the number of broken flags among 0..i **inclusive**.
        # Using the exclusive cumulative sum leaves the seam vertices adjacent
        # and the flying line survives - that was a real bug here.
        shifts = np.cumsum(broken)
        pos = np.arange(w.size) + shifts
        good = ~bad
        out_lon[pos[good]] = w[good]
        out_lat[pos[good]] = lat[good]
    return out_lon, out_lat


class _CanvasBase(QWidget):
    """A matplotlib figure plus the standard navigation toolbar."""

    def __init__(self, parent=None, figsize=(6.0, 4.0), dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi, layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavToolbar(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def refresh(self):
        self.canvas.draw_idle()

    def show_placeholder(self, text: str = "等待数据…"):
        """Draw a centred hint instead of an empty (and warning-prone) figure."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center",
                color="0.62", fontsize=12, transform=ax.transAxes)
        self.refresh()


@dataclass
class ColorScale:
    """**Shared** colour-scale state for every map view (D1).

    One instance is handed to all canvases, so "apply to the whole map family"
    really means one thing: the radial grid, the scattered view, the difference
    map and the E/N components of the horizontal page all read from here.

    ``auto=True`` uses the robust 2-98 percentile range (the pre-existing
    behaviour); ``auto=False`` uses ``vmin``/``vmax``, where an unfilled end falls
    back to the automatic bound for that end rather than to 0 or ±inf.

    ``symmetric`` mirrors the interval about 0; ``vcenter`` instead centres it on
    an arbitrary value (e.g. 0 for a difference map, or a regional mean).  When
    both are set ``vcenter`` wins, and that is stated in :meth:`describe` so the
    panel never lies about which one is in force.
    """

    auto: bool = True
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    vcenter: Optional[float] = None
    symmetric: bool = True
    percentile: tuple = (2.0, 98.0)

    # ------------------------------------------------------------- helpers
    def auto_range(self, values) -> tuple:
        """Robust ``(lo, hi)`` from the finite values, or ``(-1, 1)`` if none."""
        v = np.asarray(values, dtype=float)
        finite = np.isfinite(v)
        if not finite.any():
            return -1.0, 1.0
        lo, hi = np.percentile(v[finite], list(self.percentile))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(v[finite])), float(np.nanmax(v[finite]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return lo - 1.0, hi + 1.0
        return float(lo), float(hi)

    def limits(self, values) -> tuple:
        """The ``(vmin, vmax)`` to hand to matplotlib for these ``values``."""
        alo, ahi = self.auto_range(values)
        if self.auto:
            lo, hi = alo, ahi
        else:
            lo = alo if self.vmin is None else float(self.vmin)
            hi = ahi if self.vmax is None else float(self.vmax)
        if self.vcenter is not None:
            c = float(self.vcenter)
            a = max(abs(lo - c), abs(hi - c))
            if a <= 0:
                a = 1.0
            lo, hi = c - a, c + a
        elif self.symmetric:
            a = max(abs(lo), abs(hi))
            if a <= 0:
                a = 1.0
            lo, hi = -a, a
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return alo, ahi
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        return float(lo), float(hi)

    def describe(self, values=None) -> str:
        """One-line Chinese description of what is actually in force."""
        bits = []
        if self.auto:
            bits.append(f"自动 {self.percentile[0]:g}–{self.percentile[1]:g}%")
        else:
            bits.append("手动范围")
        if self.vcenter is not None:
            bits.append(f"以 {self.vcenter:g} 为中心")
        elif self.symmetric:
            bits.append("对称于 0")
        if values is not None:
            lo, hi = self.limits(values)
            bits.append(f"→ [{lo:.6g}, {hi:.6g}]")
        return "，".join(bits)

    def reset(self) -> None:
        self.auto = True
        self.vmin = self.vmax = self.vcenter = None


class MapCanvas(_CanvasBase):
    """Latitude/longitude map for scattered points or regular grids."""

    #: Called as ``hover(lon, lat, value, index)`` on mouse move (D2).  ``index``
    #: is the point index for scattered data and ``None`` for a grid, so the
    #: window can add whatever context only it knows (epoch, raw vs processed).
    hover_callback = None

    def __init__(self, parent=None, color: Optional[ColorScale] = None):
        super().__init__(parent, figsize=(7.0, 4.2))
        self._ax = None
        self._cax = None
        self._mappable = None
        self.show_graticule = True
        self.show_coast = True
        # shared colour-scale state (D1); a private one keeps the class usable
        # standalone in tests
        self.color = color if color is not None else ColorScale()
        # focus view: zoom the axes to the data footprint instead of the whole
        # globe.  Auto-enabled when the data are local (see _new_extent);
        # the user can toggle it with the 聚焦 button in the map control row.
        self._focus = False
        self._extent = None                  # (lon0, lon1, lat0, lat1)
        # hover readout (D2): the data currently drawn, in the same geometry
        self._hover_data = None
        self._hover_artist = None
        self._hover_marker = None
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", self._on_leave)
        self.show_placeholder("载入数据后在此显示地图")

    # ------------------------------------------------------------------ ax
    def _axes(self):
        if self._ax is None:
            self.figure.clear()
            self._ax = self.figure.add_subplot(111)
            self._cax = None
        return self._ax

    # --------------------------------------------------------------- focus
    @staticmethod
    def _nice_ticks(lo, hi, target=6):
        """Graticule ticks spanning ``[lo, hi]`` with a round step."""
        span = float(hi) - float(lo)
        if not np.isfinite(span) or span <= 0:
            return np.array([float(lo)])
        raw = span / max(int(target), 2)
        step = None
        for s in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25,
                  0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0):
            if raw <= s:
                step = s
                break
        if step is None:
            step = 90.0
        start = np.ceil(float(lo) / step) * step
        return np.arange(start, float(hi) + step * 0.5, step)

    @staticmethod
    def _pad_extent(lo0, lo1, la0, la1):
        """Padded (lon0, lon1, lat0, lat1) box, or None if it is global.

        Returns ``None`` when the footprint is global, in which case there is
        nothing to zoom to and the map stays on the whole-globe view.
        """
        # global (or near-global) coverage: focusing would only crop the map
        if (lo1 - lo0) > 350.0 or (la1 - la0) > 170.0:
            return None
        # keep degenerate extents (a meridian transect, a single point) drawable
        if lo1 - lo0 < 0.2:
            c = 0.5 * (lo0 + lo1)
            lo0, lo1 = c - 0.1, c + 0.1
        if la1 - la0 < 0.2:
            c = 0.5 * (la0 + la1)
            la0, la1 = c - 0.1, c + 0.1
        pad_lon = max(0.04 * (lo1 - lo0), 0.05)
        pad_lat = max(0.04 * (la1 - la0), 0.05)
        lon0 = max(-180.0, lo0 - pad_lon)
        lon1 = min(180.0, lo1 + pad_lon)
        lat0 = max(-90.0, la0 - pad_lat)
        lat1 = min(90.0, la1 + pad_lat)
        if lon1 - lon0 <= 0 or lat1 - lat0 <= 0:
            return None
        return (lon0, lon1, lat0, lat1)

    def _new_extent(self, lon, lat):
        """Set the focusable extent for freshly drawn data and auto-focus.

        ``lon`` and ``lat`` need not be the same length (a grid passes its two
        axes), so the limits are taken per axis.
        """
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        lon = lon[np.isfinite(lon)]
        lat = lat[np.isfinite(lat)]
        if lon.size == 0 or lat.size == 0:
            self._extent = None
            self._focus = False
            return None
        ext = self._pad_extent(float(lon.min()), float(lon.max()),
                               float(lat.min()), float(lat.max()))
        self._extent = ext
        # 载入散点或局部网格时自动聚焦到局部范围；全球数据保持全球视图
        self._focus = ext is not None
        return ext

    def can_focus(self) -> bool:
        """True when the current data have a local footprint to zoom to."""
        return self._extent is not None

    def is_focused(self) -> bool:
        return bool(self._focus and self._extent is not None)

    def set_focus(self, on: bool) -> None:
        """Toggle the focus view without re-reading the data."""
        on = bool(on) and self._extent is not None
        if on == self._focus:
            return
        self._focus = on
        self._apply_view()

    def _apply_view(self) -> None:
        """Point the axes at either the data extent or the whole globe."""
        ax = self._ax
        if ax is None:
            return
        if self.is_focused():
            lon0, lon1, lat0, lat1 = self._extent
        else:
            lon0, lon1, lat0, lat1 = -180.0, 180.0, -90.0, 90.0
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
        ratio = (lat1 - lat0) / (lon1 - lon0) if (lon1 - lon0) else 0.5
        try:
            # box aspect and equal data aspect must agree, otherwise
            # constrained-layout letterboxes the map.  0.5 reproduces the
            # whole-globe 2:1 equirectangular box.
            ax.set_box_aspect(min(max(ratio, 0.12), 2.0))
        except Exception:
            pass
        ax.set_aspect("equal", adjustable="box")
        if self.show_graticule:
            ax.set_xticks(self._nice_ticks(lon0, lon1))
            ax.set_yticks(self._nice_ticks(lat0, lat1))
            ax.grid(True, color="0.85", linewidth=0.6, zorder=0)
        self.refresh()

    def _finish(self, title: str, cb_label: str, vmin, vmax, cmap: str):
        ax = self._ax
        ax.set_xlabel("经度 (°)")
        ax.set_ylabel("纬度 (°)")
        if title:
            ax.set_title(title, fontsize=10)
        # 只画**海岸线**，不再叠加国界（`borders`）：国界虚线在看场分布时是噪声，
        # 而且它和海岸线叠在一起会让"这里是海还是陆"更难读。数据文件仍在包里
        # （load_coastlines("borders") 还能用），只是地图上不再画它。
        if self.show_coast:
            c = load_coastlines("coastline")
            if c is not None:
                # note: prepare_polyline, NOT a bare wrap_longitude - the latter
                # leaves 359-degree seams at the antimeridian that matplotlib
                # draws as lines straight across the map
                px, py = prepare_polyline(c[0], c[1])
                ax.plot(px, py, zorder=3, color="0.25", linewidth=0.6)
        if self._mappable is not None:
            if self._cax is None:
                self._cax = self.figure.colorbar(self._mappable, ax=ax,
                                                 shrink=0.85, pad=0.02)
            else:
                self._cax.cla()
                self.figure.colorbar(self._mappable, cax=self._cax)
            try:
                self._cax.set_label(cb_label, fontsize=9)
            except Exception:
                pass
        self._apply_view()

    def clear_data(self):
        self.figure.clear()
        self._ax = None
        self._cax = None
        self._mappable = None
        self._extent = None
        self._focus = False
        self._hover_data = None
        self._hover_artist = None
        self._hover_marker = None
        self.refresh()

    # --------------------------------------------------------------- hover
    def _on_motion(self, event):
        """Nearest-datum readout on mouse move (D2)."""
        if self.hover_callback is None or self._hover_data is None:
            return
        if event.inaxes is not self._ax or event.xdata is None:
            return
        lon, lat = float(event.xdata), float(event.ydata)
        kind, a, b, v = self._hover_data
        idx = None
        if kind == "points":
            # nearest sample in *degree* space; good enough for a readout and
            # O(n) — the point sets here are at most a few 10^5
            j = int(np.argmin((a - lat) ** 2 + (b - lon) ** 2))
            val = float(v[j])
            lat, lon, idx = float(a[j]), float(b[j]), j
        else:
            lon_vec, lat_vec = a, b
            i = int(np.argmin(np.abs(lat_vec - lat)))
            j = int(np.argmin(np.abs(lon_vec - lon)))
            lat, lon = float(lat_vec[i]), float(lon_vec[j])
            val = float(v[i, j])
        self._show_hover_marker(lat, lon)
        try:
            self.hover_callback(lon, lat, val, idx)
        except Exception:                                        # noqa: BLE001
            pass

    def _on_leave(self, _event):
        if self._hover_marker is not None:
            try:
                self._hover_marker.remove()
            except Exception:                                    # noqa: BLE001
                pass
            self._hover_marker = None
            self.refresh()

    def _show_hover_marker(self, lat, lon):
        ax = self._ax
        if ax is None:
            return
        if self._hover_marker is None:
            self._hover_marker, = ax.plot([lon], [lat], marker="o", ms=4.5,
                                          mfc="none", mec="k", mew=1.0,
                                          zorder=6)
        else:
            self._hover_marker.set_data([lon], [lat])
        self.refresh()

    # --------------------------------------------------------------- points
    def show_points(self, lat, lon, values, title: str = "",
                    cmap: str = "RdBu_r", symmetric: Optional[bool] = None,
                    cb_label: str = "", s: float = 8.0):
        self.figure.clear()
        self._ax = None
        self._cax = None
        self._hover_marker = None
        ax = self._axes()
        lat = np.asarray(lat, dtype=float).ravel()
        lon_raw = np.asarray(lon, dtype=float).ravel()
        lon = wrap_longitude(lon_raw)
        v = np.asarray(values, dtype=float).ravel()
        if symmetric is not None:
            self.color.symmetric = bool(symmetric)
        self._new_extent(lon, lat)
        vmin, vmax = self.color.limits(v)
        self._mappable = ax.scatter(lon, lat, c=v, cmap=cmap, s=s,
                                    vmin=vmin, vmax=vmax, linewidths=0,
                                    zorder=2)
        self._hover_data = ("points", lat, lon, v)
        self._finish(title, cb_label, vmin, vmax, cmap)

    # ----------------------------------------------------------------- grid
    def show_grid(self, lat_vec, lon_vec, grid, title: str = "",
                  cmap: str = "RdBu_r", symmetric: Optional[bool] = None,
                  cb_label: str = ""):
        self.figure.clear()
        self._ax = None
        self._cax = None
        self._hover_marker = None
        ax = self._axes()
        lat = np.asarray(lat_vec, dtype=float).ravel()
        lon = wrap_longitude(np.asarray(lon_vec, dtype=float).ravel())
        g = np.asarray(grid, dtype=float)
        if g.ndim == 3:
            g = g[:, :, 0]
        if symmetric is not None:
            self.color.symmetric = bool(symmetric)
        # the footprint is the grid's axes extent, not just the cell centres
        self._new_extent(lon, lat)
        order = np.argsort(lon)
        lon_s = lon[order]
        g_s = g[:, order]
        vmin, vmax = self.color.limits(g)
        self._mappable = ax.pcolormesh(lon_s, lat, g_s, cmap=cmap,
                                       vmin=vmin, vmax=vmax, shading="auto",
                                       zorder=2)
        # sorted axes, so the readout's column index matches what is drawn
        self._hover_data = ("grid", lon_s, lat, g_s)
        self._finish(title, cb_label, vmin, vmax, cmap)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _range(v, symmetric: bool):
        """Kept for callers/tests that only want the 2-98 percentile rule."""
        cs = ColorScale(symmetric=bool(symmetric))
        return cs.limits(v)


class HorizontalCanvas(_CanvasBase):
    """Horizontal surface displacement: north, east and the vector field.

    Contract notes (``docs/水平形变契约.md``):

    * ``magnitude`` is a rotation invariant and is valid at the poles, but the
      pole's **direction** is coordinate-singular -- so **no arrows are drawn at
      the poles** (``|lat| = 90``), and the two component maps use a symmetric
      colour scale like every other field in this program;
    * the quiver view is thinned to a readable arrow count; the thinning stride
      is reported so the figure is not silently misleading.
    """

    MAX_ARROWS_PER_ROW = 24

    def __init__(self, parent=None, color: Optional[ColorScale] = None):
        # 三行（u_N / u_E / |u_h|）按**原尺寸**画在一张"很高的"画布上，外层用滚动区
        # 承载：以前挤成 2×2 布局，第三行的标题会压到上一行的色标上，分量图也被压扁。
        super().__init__(parent, figsize=(7.6, 13.2))
        self.show_coast = True
        self._drawn = None
        # the same shared scale as the map tab (D1): one panel drives every map,
        # including the north/east components below
        self.color = color if color is not None else ColorScale()
        self.show_placeholder("运行分析后在此显示水平形变（北 / 东 / 水平模）")

    def _sym(self, v):
        """Symmetric limits for one component, via the shared colour scale."""
        cs = self.color
        was = cs.symmetric
        cs.symmetric = True                    # 分量图必须对称于 0（契约 F3/F4）
        try:
            return cs.limits(v)
        finally:
            cs.symmetric = was

    def _maps(self, north, east, lat, lon, *, is_grid: bool, title=""):
        """三行：北分量、东分量、水平模（叠加矢量 + 比例尺）。

        ``lat``/``lon`` are the *axes* for a grid, or the per-point coordinates
        for scattered data (``is_grid`` says which).

        三处刻意的选择（都来自"图不好读"的反馈）：

        * **两个分量共用一个色标范围**（取两者绝对值的公共上限，对称于 0）：
          各画各的时 u_N 与 u_E 的红/蓝色阶含义不同，摆在一起会读成"东向比北向小"；
        * **第三行画水平模** ``|u_h| = hypot(u_N, u_E)``（顺序色标，从 0 起），
          并**叠加矢量场**与一把 quiver 比例尺 —— 只有箭头没有比例尺的矢量图
          没法定量读，而只有模又丢掉了方向；
        * 极点行不画箭头（那里的东/北是坐标奇异的），并在标题里写明。

        ⚠️ 网格的经度轴**必须**先化成 ``[-180, 180)`` 再排序（与
        :meth:`MapCanvas.show_grid` 同一口径）。真实产品常把经度写成 ``0…360``
        （CSR mascon 是 0.125…359.875），而海岸线数据在 ``-180…180``：直接把两套
        坐标画在同一张图上，x 轴会被撑到 -180…360，**场与海岸线错位半个地球**，
        看起来像"形变画错了"。
        """
        self.figure.clear()
        axes = [self.figure.add_subplot(3, 1, i + 1) for i in range(3)]
        axn, axe, axq = axes
        north = np.asarray(north, dtype=float)
        east = np.asarray(east, dtype=float)
        if is_grid:
            lon = np.asarray(lon, dtype=float).ravel()
            lat = np.asarray(lat, dtype=float).ravel()
            order = np.argsort(wrap_longitude(lon))
            lon = wrap_longitude(lon)[order]
            north = north[:, order]
            east = east[:, order]
        else:
            # 散点也要包到 [-180,180)：数据常来自 0…360 的文件，而海岸线在
            # -180…180 —— 不包的话同一个"东西半球半张脸"的错位照样出现。
            # 顺序无关紧要（scatter 逐点画），值跟着坐标一起走即可。
            lon = wrap_longitude(np.asarray(lon, dtype=float).ravel())
            lat = np.asarray(lat, dtype=float).ravel()

        # 分量共用范围：取两者的公共对称上限（读起来才可比）
        lo, hi = self._sym(np.concatenate([np.ravel(north), np.ravel(east)]))
        mag = np.hypot(north, east)
        mlo, mhi = 0.0, float(np.nanmax(mag)) if np.size(mag) else 1.0

        panels = [
            (axn, north, "北分量 u_N（向上为正）", "RdBu_r", (lo, hi), True),
            (axe, east, "东分量 u_E（向东为正）", "RdBu_r", (lo, hi), True),
            (axq, mag, "水平模 |u_h|（叠加矢量；极点不画箭头）", "viridis",
             (mlo, mhi), False),
        ]
        for ax, v, name, cmap, (v0, v1), sym in panels:
            if is_grid:
                m = ax.pcolormesh(lon, lat, v, cmap=cmap, vmin=v0, vmax=v1,
                                  shading="auto")
            else:
                m = ax.scatter(lon, lat, c=np.ravel(v), cmap=cmap, s=6,
                               vmin=v0, vmax=v1, linewidths=0)
            cb = self.figure.colorbar(m, ax=ax, shrink=0.92, pad=0.015)
            cb.set_label("m", fontsize=8)
            cb.ax.tick_params(labelsize=8)
            ax.set_title(name, fontsize=10, pad=6)
            ax.set_xlabel("经度 (°)", fontsize=8)
            ax.set_ylabel("纬度 (°)", fontsize=8)
            ax.tick_params(labelsize=8)
            if is_grid:
                ax.set_xlim(float(lon[0]), float(lon[-1]))
                ax.set_ylim(float(lat[0]), float(lat[-1]))
            self._coast(ax)
        if is_grid:
            # 两个分量共用范围时说一句，避免"看起来东向小"的误读
            axes[0].text(0.005, 1.02, f"u_N 与 u_E 共用色标 ±{max(abs(lo), abs(hi)):.4g} m",
                         transform=axes[0].transAxes, fontsize=8, color="0.3")

        # ---- vectors on the magnitude panel (thinned; poles skipped) ---------
        if is_grid:
            ilat = max(1, np.size(lat) // self.MAX_ARROWS_PER_ROW)
            ilon = max(1, np.size(lon) // (2 * self.MAX_ARROWS_PER_ROW))
            LA, LO = np.meshgrid(lat[::ilat], lon[::ilon], indexing="ij")
            un, ue = north[::ilat, ::ilon], east[::ilat, ::ilon]
            stride = f"每 {ilat}×{ilon} 格点 1 个"
        else:
            step = max(1, north.size // 600)
            # thin the coordinates together with the components, or the pole
            # mask below would be sized against the un-thinned arrays
            LA = np.asarray(lat)[::step]
            LO = np.asarray(lon)[::step]
            un, ue = north[::step], east[::step]
            stride = f"每 {step} 点 1 个" if step > 1 else "全部点"
        keep = np.abs(np.abs(LA) - 90.0) > 1e-9          # 极点不画箭头（契约 F4）
        self._quiver_key = None
        if np.any(keep):
            # 箭头用**白色**（在 viridis 的深色端最清楚）；给一层很细的深色描边，
            # 这样落在亮黄端时也不会糊掉。比例尺文字同样白色 + 深色描边。
            q = axq.quiver(LO[keep], LA[keep], ue[keep], un[keep],
                           facecolor="white", edgecolor="0.15", linewidth=0.25,
                           pivot="tail", width=0.0016, minshaft=2.0,
                           scale_units="xy", angles="xy",
                           scale=None if is_grid else 1.0)
            # 比例尺：没有它，箭头长短无法定量（这是"图不好读"的一半原因）。
            # 放在**轴内**左下角：放在轴外会和标题挤在一起。
            span = max(float(np.nanmax(mag)) if np.size(mag) else 1.0, 1e-30)
            # 留一个引用：测试要断言"比例尺真的画了"，而 QuiverKey 不进 ax.texts
            self._quiver_key = axq.quiverkey(
                q, 0.03, 0.07, span * 0.5, f"{span * 0.5:.3g} m　{stride}",
                labelpos="E", coordinates="axes",
                fontproperties={"size": 8}, labelsep=0.06,
                color="white", labelcolor="white")
            _key_text = getattr(self._quiver_key, "text", None)
            if _key_text is not None:
                try:
                    _key_text.set_path_effects(
                        [pe.withStroke(linewidth=2.0, foreground="0.15")])
                except Exception:                                # noqa: BLE001
                    pass
        axq.set_title("水平模 |u_h|（叠加矢量；极点不画箭头 —— 极点的东/北是坐标奇异的）",
                      fontsize=10, pad=6)
        if title:
            self.figure.suptitle(title, fontsize=11)
        # 不要 tight_layout：画布用的是 matplotlib 的 constrained layout
        # （见 _CanvasBase），两者混用会互相打架。
        self._drawn = (np.shape(mag),
                       float(np.nanmax(mag)) if np.size(mag) else 0.0)
        self.refresh()

    def _coast(self, ax):
        if not self.show_coast:
            return
        for which, style in (("coastline", dict(color="0.25", linewidth=0.6)),):
            c = load_coastlines(which)
            if c is not None:
                px, py = prepare_polyline(c[0], c[1])
                ax.plot(px, py, zorder=3, **style)


class SpectrumCanvas(_CanvasBase):
    """Per-degree amplitude / error curves on a log axis."""
    def __init__(self, parent=None):
        super().__init__(parent, figsize=(7.0, 4.2))
        self._ax = None
        self.show_placeholder("运行分析后在此显示逐阶振幅")

    def show(self, curves: dict, title: str = "逐阶振幅 (degree RMS)",
             ylabel: str = "RMS", log: bool = True, xlabel: str = "球谐阶 n"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for label, y in curves.items():
            y = np.asarray(y, dtype=float).ravel()
            if y.size == 0:
                continue
            x = np.arange(y.size)
            ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.2,
                    label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        if log:
            ax.set_yscale("log")
        ax.grid(True, which="both", color="0.9", linewidth=0.6)
        if curves:
            ax.legend(fontsize=8, frameon=False)
        self.refresh()


class SeriesCanvas(_CanvasBase):
    """A point/region time series with its fitted trend and annual cycle (D6).

    The x axis is **decimal years** (the label the legacy pipeline and every
    published GRACE figure use), while the fit itself is done on true elapsed
    time -- the two are related by :mod:`shkit.timeaxis`, and the axis note says
    which convention was used.
    """

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(7.2, 4.2))
        self.show_placeholder("载入多时次数据后，在这里选点看时间序列")

    def show(self, years, values, *, label: str = "", band=None,
             trend=None, annual=None, title: str = "", ylabel: str = "",
             note: str = "", markers: bool = True):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        y = np.asarray(values, dtype=float).ravel()
        x = np.asarray(years, dtype=float).ravel()
        if band is not None:
            b = np.asarray(band, dtype=float).ravel()
            ax.fill_between(x, y - b, y + b, color="0.82", alpha=0.7, zorder=1,
                            label="拟合残差 ±1σ（不是观测误差）")
        ax.plot(x, y, marker="o" if markers else None, markersize=3.0,
                linewidth=0.9, color="#1f4e79", label=label or "序列", zorder=3)
        if trend is not None:
            ax.plot(x, np.asarray(trend).ravel(), linewidth=1.8, color="#c00000",
                    label="线性趋势", zorder=4)
        if annual is not None:
            ax.plot(x, np.asarray(annual).ravel(), linewidth=1.4, linestyle="--",
                    color="#2e7d32", label="周年拟合", zorder=4)
        ax.set_xlabel("十进制年")
        ax.set_ylabel(ylabel or "数值")
        ax.set_title(title or "时间序列", fontsize=10)
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.legend(fontsize=8, frameon=False, ncols=2)
        if note:
            ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8,
                    color="0.35", va="bottom")
        self.refresh()


class TrendCanvas(_CanvasBase):
    """Trend / seasonal amplitude / seasonal phase maps (D7).

    Two honesty rules, both learned from user reports:

    * **每一个拟合了的周期都要画出来。** 用户勾了「含半年周期」却只看到周年场，
      会以为半年项没算 —— 其实它被拟合了、只是没显示。现在 ``periods=(1.0, 0.5)``
      会出 5 个面板（趋势 + 每个周期的振幅/相位）。
    * ``phase`` 在振幅 ~0 处没有意义，所以**逐周期**取阈值把低振幅格子遮掉，
      并在标题里写明遮的是哪一档、阈值多少。
    """

    #: Below this fraction of the 98th-percentile amplitude, the phase is masked.
    PHASE_MIN_FRAC = 0.05

    def __init__(self, parent=None, color: Optional[ColorScale] = None):
        super().__init__(parent, figsize=(7.4, 6.6))
        self.color = color if color is not None else ColorScale()
        self.show_coast = True
        self._axes = []
        self._last = None                 # 最近的 (maps, lat, lon, is_grid, title, unit)
        self.show_placeholder("载入多时次数据后，在这里看趋势场与周年振幅/相位场")

    # ------------------------------------------------------------- coastline
    def _coast(self, ax):
        if not self.show_coast:
            return
        c = load_coastlines("coastline")
        if c is not None:
            px, py = prepare_polyline(c[0], c[1])
            ax.plot(px, py, zorder=3, color="0.25", linewidth=0.6)

    def set_coast(self, on: bool) -> None:
        """海岸线开关：立刻按上次的数据重画（不重新拟合）。"""
        self.show_coast = bool(on)
        if self._last is not None:
            self.show(**self._last)          # show() 的 is_grid/title/unit 是 keyword-only

    @staticmethod
    def _period_label(period: float) -> str:
        """``1.0 -> '周年'``、``0.5 -> '半年'``，其它给出周期本身。"""
        from ..timeseries import PERIOD_NAMES
        tag = PERIOD_NAMES.get(float(period))
        if tag == "annual":
            return "周年"
        if tag == "semiannual":
            return "半年"
        return f"{float(period):g} 年周期"

    def show(self, maps: dict, lat, lon, *, is_grid: bool, title: str = "",
             unit: str = ""):
        self.figure.clear()
        if is_grid:
            lat = np.asarray(lat, dtype=float).ravel()
            lon_raw = np.asarray(lon, dtype=float).ravel()
            order = np.argsort(wrap_longitude(lon_raw))
            lon_s = wrap_longitude(lon_raw)[order]
        else:
            order = None
            # 散点：同样要包到 [-180,180)，否则 lon=0…360 的点有一半画在
            # xlim(-180,180) 之外（整段 180…360 直接看不见）。
            lon_raw = wrap_longitude(np.asarray(lon, dtype=float).ravel())
            lon_s = lon_raw
        # 存**原始**轴：show() 会重新 wrap+sort，存排好序的那份会让"重画"时
        # order 变成恒等映射、而 maps 还是原顺序 → 场与经度错位 180°。
        self._last = dict(maps=maps, lat=lat, lon=lon_raw, is_grid=is_grid,
                          title=title, unit=unit)

        # 逐周期面板：趋势 + 每个周期的（振幅, 相位）
        seas = maps.get("seasonal")
        if not seas:
            amp = np.asarray(maps.get("annual_amplitude"), dtype=float)
            seas = [{"tag": "annual", "period": 1.0, "amplitude": amp,
                     "phase_doy": np.asarray(maps.get("annual_phase_doy"),
                                             dtype=float)}]
        panels = [("趋势场", maps.get("trend"), "RdBu_r",
                   (f"每{unit or '单位'}/年" if unit else "每年"), True)]
        for s in seas:
            lab = self._period_label(s.get("period", 1.0))
            amp = np.asarray(s.get("amplitude"), dtype=float)
            fin = amp[np.isfinite(amp)]
            thr = (self.PHASE_MIN_FRAC * float(np.percentile(fin, 98))
                   if fin.size else 0.0)
            ph = np.where(amp > thr, np.asarray(s.get("phase_doy"), dtype=float),
                          np.nan)
            panels.append((f"{lab}振幅场", amp, "viridis", (unit or ""), False))
            panels.append((f"{lab}相位场（峰值日；振幅 < {thr:.3g} 的格子不画）",
                           ph, "twilight", "一年中的第几天", False))

        n = len(panels)
        ncol = 2
        nrow = int(np.ceil(n / ncol))
        axes = [self.figure.add_subplot(nrow, ncol, i + 1) for i in range(n)]
        for ax, (name, v, cmap, cbl, _sym) in zip(axes, panels):
            v = np.asarray(v, dtype=float)
            if is_grid and order is not None:
                v = v[:, order]
            if is_grid:
                lo, hi = self.color.limits(v)
                m = ax.pcolormesh(lon_s, lat, v, cmap=cmap, vmin=lo, vmax=hi,
                                  shading="auto")
                ax.set_xlim(float(lon_s[0]), float(lon_s[-1]))
                ax.set_ylim(float(lat[0]), float(lat[-1]))
            else:
                vv = v.ravel()
                lo, hi = self.color.limits(vv)
                m = ax.scatter(lon_s, lat, c=vv, cmap=cmap, s=6,
                               vmin=lo, vmax=hi, linewidths=0)
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
            self.figure.colorbar(m, ax=ax, shrink=0.85, pad=0.02).set_label(
                cbl, fontsize=8)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("经度 (°)", fontsize=8)
            ax.set_ylabel("纬度 (°)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_aspect("equal", adjustable="box")
            self._coast(ax)
        if title:
            self.figure.suptitle(title, fontsize=10)
        self._axes = axes
        self.refresh()
