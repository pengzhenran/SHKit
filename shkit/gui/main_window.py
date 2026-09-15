# -*- coding: utf-8 -*-
"""
shkit.gui.main_window
=====================

The SHKit desktop window.

Layout::

    ┌─────────────┬───────────────────────────────┬─────────────┐
    │ 数据 dock   │  地图 / 诊断 / 逐阶谱 / 系数   │ 参数 dock   │
    └─────────────┴───────────────────────────────┴─────────────┘
    status bar: message + progress

All heavy work happens in :mod:`shkit.gui.workers`; this module only builds the
interface and wires signals.
"""

from __future__ import annotations

import os
import re
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import (QAction, QDesktopServices, QKeySequence, QPixmap,
                           QTextCursor)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMainWindow, QMessageBox,
                               QProgressBar, QProgressDialog, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QTabWidget,
                               QTableWidget, QTableWidgetItem, QTextBrowser,
                               QVBoxLayout, QWidget)

from .. import __version__
from .. import io as shio
from ..diagnostics import harmonic_resolution_km
from ..weights import WEIGHT_RULES
from . import coeff_cache
from .canvases import ColorScale, HorizontalCanvas, MapCanvas, SpectrumCanvas
from .dataset import Dataset
from .workers import (AnalysisWorker, ExportWorker, HorizontalWorker, LoadWorker,
                      SeriesWorker, SubprocessRunner)

__all__ = ["MainWindow"]

# SHKit project root (…/SHKit), used to locate licenses/ and docs/ in both the
# source tree and a PyInstaller --onedir bundle.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _pyside_version() -> str:
    try:
        import PySide6
        return str(PySide6.__version__)
    except Exception:                                  # noqa: BLE001
        return "?"


# 「估计方法」—— 默认第一项：直接求积（效率优先）。
# GUI 不提供 'auto'：选哪条由使用者明确指定，软件不替他改。
# 库与 CLI 仍然保留 method='auto'（先直接求积＋体检，不健康才升级），
# 需要那套自动升级时用 API / `shkit analyze --method auto`。
_METHODS = [("quadrature（直接求积，快；默认）", "quadrature"),
            ("iterative（求积 + 迭代校正）", "iterative"),
            ("projection（零填充全球投影：区域数据的目标 A）", "projection"),
            ("wlsq（加权最小二乘，需全球覆盖）", "wlsq"),
            ("cg（矩阵无关共轭梯度）", "cg")]
_REGS = [("不启用", None), ("tikhonov（最小范数）", "tikhonov"),
         ("kaula（逐阶幂律先验）", "kaula")]
_NORMS = [("auto（推荐）", "auto"), ("global（强制 sum w = 4π）", "global"),
          ("region（保留几何面积）", "region")]

# 「输入是」—— 正变换公式的选择器（决定导出的 C_nm 是什么）
# 第一项是默认档：普通网格，f_u = 1（网格本身就是那套无量纲位系数场）。
# 注：库/CLI 里仍然支持 field_unit='scalar'（区域平均核这类无物理量纲的场）
# 与 'surface_density'（面密度 σ，只是 EWH 的另一种单位），GUI 不再列出。
_FIELD_UNITS = [
    ("普通网格", "geopotential"),
    ("水准面高 ΔN (geoid)", "geoid"),
    ("等效水高 EWH", "ewh"),
    ("径向形变 u_r", "radial_displacement"),
]

# 「输出为」—— 反变换公式的选择器（决定输出场是什么物理量）
# 空串 = 用输入那一档的 f_u 反算，输出场就是重建场
_TARGET_UNITS = [
    ("不换算（按输入公式反算，重建回原场）", ""),
    ("等效水高 EWH", "ewh"),
    ("水准面高 ΔN (geoid)", "geoid"),
    ("普通球谐系数", "geopotential"),
    ("径向形变 u_r", "radial_displacement"),
]

_UNIT_HINTS = {
    "scalar": "正变换公式：f_u = 1（没有物理公式）。导出的就是该场自身的系数，"
              "不能换算成 EWH/geoid —— 那会被拒绝。"
              "区域平均核、任意无量纲网格走这一档（GUI 里不列出，用 API/CLI 指定）。",
    "geopotential": "「普通网格」= 那串数字就是<b>无量纲重力位系数 C_nm/S_nm</b>"
                    "（GRACE Level-2 公布的那一套）。正变换公式 f_u = 1，"
                    "所以导出的系数与输入数值一致。",
    "geoid": "正变换公式：C_nm = a_nm ÷ R（R = 6378136.46 m，<b>常数</b>，不逐阶）。"
             "同一块网格若改声明成 EWH，除的是 Aₙ，两者第 2 阶差 13.2 倍。",
    "surface_density": "正变换公式：C_nm = a_nm ÷ [R·ρ̄/3·(2n+1)/(1+k′ₙ)]。"
                       "σ 与 EWH 是<b>同一个物理量的两种单位</b>（σ = ρ_w·EWH，差常数 1000）。",
    "ewh": "正变换公式：C_nm = a_nm ÷ Aₙ，Aₙ = R·ρ̄/(3ρ_w)·(2n+1)/(1+k′ₙ)，"
           "<b>逐阶、不是常数</b>（0→6 阶跨 14.3 倍）。",
    "radial_displacement": "正变换公式：C_nm = a_nm ÷ [R·h′ₙ/(1+k′ₙ)]，"
                           "h′ 取自随包的 PREM 载荷勒夫数表。"
                           "注意 h′₀ = 0 ⇒ 0 阶不可反推，该阶不为 0 时会报错。",
}

_FORMULA_LABELS = {
    "geopotential": "普通网格 / 普通球谐系数（无量纲位系数 C_nm/S_nm）",
}


def _gui_unit_label(key: str, role: str = "field") -> str:
    """界面里用的物理量名。

    「普通网格」（输入）与「普通球谐系数」（输出）就是那套无量纲重力位系数
    ``C_nm/S_nm``；文件头与报告里仍写技术上更明确的
    ``units.FIELD_UNIT_LABELS``，避免歧义。
    """
    if key == "geopotential":
        return "普通网格" if role == "recon" else "普通球谐系数"
    from .. import units as _units
    return _units.FIELD_UNIT_LABELS.get(key, key)


# ============================================================ 开发者 / 身份信息
# 与同组的 GRACE_Downloader 系列保持一致（grace_downloader_gui_release.py）
APP_NAME = "SHKit"
AUTHOR_NAME_CN = "彭桢燃"
AUTHOR_NAME_EN = "Zhenran Peng"
AUTHOR_EMAIL = "zhenran.peng@cug.edu.cn"
AUTHOR_PHONE = "15927402265"
AUTHOR_AFFILIATION_CN = "中国地质大学（武汉）"
AUTHOR_AFFILIATION_EN = "China University of Geosciences (Wuhan)"

WECHAT_ACCOUNT = "地球重力与人类生活"
WECHAT_ACCOUNT_EN = "TVGG"
WECHAT_QR_FILENAME = "地球重力与人类生活TVGG.jpg"
WECHAT_QR_CAPTION = f"课题组公众号：{WECHAT_ACCOUNT}（{WECHAT_ACCOUNT_EN}）"

# 随包的文档目录与 HTML 使用说明（含界面截图与示例配图）。
# docs/ 由 packaging/shkit.spec 一起打进 onedir 发行包，所以打包后仍然能找到。
DOCS_DIRNAME = "docs"
GUIDE_HTML_NAME = "使用说明.html"
GUIDE_MD_NAME = "使用说明_GUI.md"


def _first_existing(*rel_parts) -> str:
    """First existing path among ``rel_parts`` (relative to the project root)
    or the current working directory.  Returns "" when nothing is found."""
    for rel in rel_parts:
        for base in (_PROJECT_ROOT, os.getcwd()):
            cand = os.path.join(base, rel)
            if os.path.exists(cand):
                return cand
    return ""


def guide_html_path() -> str:
    """Path of the screenshot-based HTML user guide, or "" if absent."""
    return _first_existing(os.path.join(DOCS_DIRNAME, GUIDE_HTML_NAME),
                           GUIDE_HTML_NAME)


def guide_md_path() -> str:
    """Path of the Markdown user guide (fallback when the HTML is missing)."""
    return _first_existing(os.path.join(DOCS_DIRNAME, GUIDE_MD_NAME),
                           GUIDE_MD_NAME)


def qr_image_path() -> str:
    """Path of the WeChat official-account QR image, or "" if absent."""
    return _first_existing(WECHAT_QR_FILENAME,
                           os.path.join(DOCS_DIRNAME, WECHAT_QR_FILENAME))



_HINT_STYLE = "color:#777;font-size:9pt;"
_HINT_OK = "color:#2a7;font-size:9pt;"
_HINT_BAD = "color:#b35;font-size:9pt;"


def _hint_label(text: str = "", style: str = _HINT_STYLE,
                bold: "bool | None" = True) -> QLabel:
    """A word-wrapping hint label that does **not** widen its dock.

    带 ``wordWrap`` 的 QLabel 仍然把"整行宽度"报成 ``sizeHint``，而
    QMainWindow 是按内容的 ``sizeHint`` 给停靠面板定宽的——所以提示/预览文字
    一填长，右侧参数面板就会变宽、把地图挤窄（分析完成后布局突然一跳）。
    把水平方向的 sizeHint 忽略掉，标签就只占面板给它的宽度并在那里换行。

    ``bold=False`` 用来把「快速上手」这类清单正文压回常规字重（条目编号自己带
    ``<b>``），否则整段都会继承 QGroupBox 的粗体，读起来很吵。``bold=True``
    （默认）保持原有行为，所以别处的提示文字不受影响。
    """
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(("" if bold else "font-weight:normal;") + style)
    sp = lbl.sizePolicy()
    sp.setHorizontalPolicy(QSizePolicy.Ignored)
    lbl.setSizePolicy(sp)
    lbl.setMinimumWidth(0)
    return lbl


_RULE_LABELS = {
    "auto": "auto（按采样自动判断）", "dh": "dh（全球等经纬网格，精确）",
    "grid": "grid（任意经纬网格的精确球带面积）",
    "lattice": "lattice（规则格网上的掩膜：每点一个格元）",
    "voronoi": "voronoi（全球散点：球面 Voronoi 胞面积）",
    "delaunay": "delaunay（区域散点：球面 Delaunay 1/3）",
    "uniform": "uniform（等面积 4π/N，慎用）",
}


class MainWindow(QMainWindow):
    """SHKit main window."""

    #: 「显示」下拉里**需要逐历元系数**的那几项（0 = 原始数据不需要）。
    #: 它决定"切显示 / 播放 / 导出"要不要先保证手上真有这个历元的系数。
    EPOCH_VIEWS = (1, 2, 3)

    #: 播放时等一个历元的按需补算最多等多久（秒）。超过就停下来报错，
    #: 而不是永远卡在"正在补算…"上（真实 mascon、nmax=60 时一个历元可能要几十秒，
    #: 所以按**墙钟**而不是按定时器拍数计，帧率高低不影响这个上限）。
    PLAY_WAIT_SECONDS_MAX = 300.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"SHKit {__version__} — 球谐分析（任意散点 / 任意网格）")
        self.resize(1440, 900)

        self.dataset: Dataset | None = None
        self.coeffs = None
        self.report = None
        self.recon = None          # 重建场：与输入同物理量（f_t = f_u）
        self.outfield = None       # 输出场：按「输出为」的物理量（f_t）
        #: 多历元系数时按历元缓存的重建/输出场（播放时避免每个历元都重算）
        self._epoch_cache: dict = {}
        #: 「运行分析」解的是**哪一个**历元。``self.recon`` 只对它成立 —— 以前
        #: 换时次后仍然拿它当"当前历元的重建场"画，用户看到的就是"播放不动 /
        #: 播的还是原始图"。多时次数据走"一次解完全部历元"（见 _start_series），
        #: 这条只对**单时次**数据集有意义。
        self._single_t = None
        #: 系数版本号 + 水平形变那张图的指纹：水平形变**只在用户点按钮时算**，
        #: 系数/时次一变就把旧图作废（不自动重算）。
        self._coeff_version = 0
        self._horiz_sig = None
        self._horiz_data = None
        #: 正在载入数据（on_loaded 期间）：此时**不允许自动开跑**整条序列 ——
        #: 否则"上一份数据停在重建场"会让新数据一打开就在后台解 256 个历元
        #: （懒加载的文件还要被整块读进来），用户根本没点过任何按钮。
        self._loading = False
        #: 正在跑的逐历元分析是干什么用的（进度提示里写明"运行分析/批量分析/
        #: 导出 GIF…"），以及它用的参数（写缓存时必须用**当时**那套参数）。
        self._series_reason = ""
        self._series_params = None
        #: 当前这套逐历元系数是从哪个缓存文件读回来的（None = 本次真算的）
        self._series_cache = None
        #: 地图上现在这张图属于哪个数据集 / 是哪一个「显示」（换数据后不能拿旧图顶数）
        self._map_ds = None
        self._map_view_idx = None
        #: 播放时等按需补算的起点（秒；None = 没在等）
        self._play_wait_since = None
        self.load_worker = None
        self.analysis_worker = None
        self.series_worker = None
        #: 水平形变（球面梯度）的 worker —— 散点上要几十秒，绝不能在界面线程里跑
        self.horiz_worker = None
        self._horiz_pending_t = 0
        self._t0 = 0.0
        #: 懒加载立方体的后台补数监视（见 _on_lazy_tick）：只用来显示进度 +
        #: 补齐后刷新"需要全部历元"的页面（时间序列/趋势），不参与数据读取本身。
        self._lazy_ld = None
        self._lazy_timer = QTimer(self)
        self._lazy_timer.setInterval(300)
        self._lazy_timer.timeout.connect(self._on_lazy_tick)

        self._build_central()
        self._build_data_dock()
        self._build_param_dock()
        self.resizeDocks([self._data_dock, self.param_dock], [300, 360],
                         Qt.Horizontal)
        self._build_menus()
        self._build_status()
        self._update_actions()
        self.status("就绪。用「文件 → 打开散点/网格」载入数据，或「生成示例数据」试用。")

    # ==================================================================== UI
    def _build_central(self):
        self.tabs = QTabWidget(self)
        # --- D1: ONE colour-scale state shared by every map view --------------
        self.color_scale = ColorScale()
        # D3: the value table is refilled lazily (a 65k-cell table costs seconds)
        self._values_dirty = True
        self._last_hover = None
        self._values_dirty = True
        self.map_canvas = MapCanvas(self, color=self.color_scale)
        self.map_canvas.hover_callback = self._on_map_hover
        self.spectrum_canvas = SpectrumCanvas(self)
        self.report_view = QTextBrowser(self)
        self.report_view.setOpenExternalLinks(True)

        # --- map tab wraps the canvas with a small control row ---------------
        map_tab = QWidget()
        mv = QVBoxLayout(map_tab)
        mv.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(QLabel("显示："))
        self.map_field = QComboBox()
        self.map_field.addItems(["原始数据", "重建场（与输入同量）",
                                 "差值 (原始 − 重建)", "输出场（按「输出为」）"])
        self.map_field.setToolTip(
            "重建场与输入网格<b>同物理量</b>，可以直接逐点比对（差值图就是它）；\n"
            "输出场是按「输出为」换算出来的另一个物理量的场。\n"
            "「输出为 = 不换算」时两者是同一个场。")
        self.map_field.currentIndexChanged.connect(lambda _i: self._refresh_map())
        row.addWidget(self.map_field)
        self.map_symmetric = QCheckBox("对称色标")
        self.map_symmetric.setChecked(True)
        self.map_symmetric.toggled.connect(self._on_symmetric_toggled)
        self.map_symmetric.setToolTip(
            "把色标区间做成关于 0 对称（正负同量级时最公平）。\n"
            "与下面的「以 vcenter 为中心」互斥：填了中心值就以它为准。")
        row.addWidget(self.map_symmetric)
        self.map_coast = QCheckBox("海岸线")
        self.map_coast.setChecked(True)
        self.map_coast.toggled.connect(self._toggle_coast)
        row.addWidget(self.map_coast)
        self.btn_focus = QPushButton("聚焦到数据")
        self.btn_focus.setCheckable(True)
        self.btn_focus.setEnabled(False)
        self.btn_focus.setToolTip(
            "把地图缩放到数据的经纬范围。\n"
            "载入散点或局部网格时会自动聚焦；全球数据没有可聚焦的局部范围，\n"
            "此时按钮为灰色。再点一次即取消聚焦，回到全球视图。")
        self.btn_focus.toggled.connect(self._on_focus_toggled)
        row.addWidget(self.btn_focus)
        row.addStretch(1)
        mv.addLayout(row)
        mv.addLayout(self._build_color_row())
        mv.addLayout(self._build_play_row())
        # 逐历元视图的提示（"这张图是哪个历元的场/正在补算第几个"）。单独占一行，
        # 免得跟播放行的控件抢宽度导致中文被截断。
        self.map_note = _hint_label("")
        mv.addWidget(self.map_note)
        mv.addWidget(self.map_canvas, 1)
        self.tabs.addTab(map_tab, "地图")

        self.tabs.addTab(self.spectrum_canvas, "逐阶谱")
        self.tabs.addTab(self.report_view, "诊断报告")
        self.tabs.addTab(self._build_coeff_tab(), "系数统计")
        self.tabs.addTab(self._build_values_tab(), "数值表")
        self.tabs.addTab(self._build_epochs_tab(), "逐历元诊断")
        self.tabs.addTab(self._build_series_tab(), "时间序列")
        self.tabs.addTab(self._build_trend_tab(), "趋势与周年")
        self.tabs.addTab(self._build_horizontal_tab(), "水平形变")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

    # ------------------------------------------------------------------ D9
    def _build_play_row(self) -> QVBoxLayout:
        """Playback + per-epoch movie export (D9).

        刻意排成**两行**而不是一行：一行放不下八个控件时，Qt 会把它们压到最小
        宽度以下，中文后缀（"4 帧/秒"）和带省略号的按钮文字就被切掉。两行之后每个
        控件都能拿到自己的 sizeHint，窗口再窄也不会出现"显示不全"。
        """
        box = QVBoxLayout()
        box.setSpacing(2)
        row = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip(
            "按帧率逐个历元播放**当前视图**（原始/重建/差值/输出场都行）。\n"
            "播放到时间窗口末端自动停，不循环 —— 免得看漏最后一帧。")
        self.btn_play.toggled.connect(self._on_play_toggled)
        row.addWidget(self.btn_play)
        self.sp_fps = QSpinBox()
        self.sp_fps.setRange(1, 30)
        self.sp_fps.setValue(4)
        self.sp_fps.setSuffix(" 帧/秒")
        # 不要给 spinbox 设小于 sizeHint 的 maximumWidth：中文后缀在 125%/150%
        # 缩放下会比 sizeHint 更宽，设死 90 px 就会把"秒"或上下箭头切掉。
        self.sp_fps.setMinimumWidth(
            self.sp_fps.fontMetrics().horizontalAdvance("30 帧/秒") + 44)
        self.sp_fps.valueChanged.connect(self._apply_play_interval)
        row.addWidget(self.sp_fps)
        self.btn_gif = QPushButton("导出 GIF…")
        self.btn_gif.setToolTip(
            "把时间窗口内的历元逐帧渲染成 GIF（用 Pillow；不依赖 ffmpeg）。\n"
            "帧数 = 时间窗口长度，渲染时可取消。")
        self.btn_gif.clicked.connect(self._export_gif)
        row.addWidget(self.btn_gif)
        self.play_note = _hint_label("")
        row.addWidget(self.play_note, 1)
        box.addLayout(row)

        row2 = QHBoxLayout()
        self.edit_export_prefix = QLineEdit("epoch")
        self.edit_export_prefix.setToolTip("逐时次导出的文件名前缀（默认 epoch）")
        self.edit_export_prefix.setMinimumWidth(
            self.edit_export_prefix.fontMetrics().horizontalAdvance("epoch0000") + 20)
        row2.addWidget(QLabel("逐时次导出前缀")); row2.addWidget(self.edit_export_prefix)
        self.cb_export_ext = QComboBox()
        self.cb_export_ext.addItem("每历元 .nc", ".nc")
        self.cb_export_ext.addItem("每历元 .grd", ".grd")
        self.cb_export_ext.addItem("每历元 .csv", ".csv")
        self.cb_export_ext.setToolTip("逐时次导出写什么格式")
        row2.addWidget(self.cb_export_ext)
        self.btn_export_epochs = QPushButton("逐时次导出…")
        self.btn_export_epochs.setToolTip(
            "把时间窗口内的每个历元各自写成一个文件（文件名 = 前缀_0001.nc）。\n"
            "导出过程中可以**暂停/继续/停止**；停下来的已写文件都是完整历元。")
        self.btn_export_epochs.clicked.connect(self.export_epochs)
        row2.addWidget(self.btn_export_epochs)
        row2.addStretch(1)
        box.addLayout(row2)

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(250)
        self._play_timer.timeout.connect(self._play_step)
        return box

    def _apply_play_interval(self):
        self._play_timer.setInterval(int(1000 / max(self.sp_fps.value(), 1)))

    def _stop_play(self, note: str = "") -> None:
        """Stop playback and (optionally) explain why in the play row."""
        self._play_timer.stop()
        self._play_wait_since = None
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶ 播放")
        if note:
            self.play_note.setText(note)

    def _on_play_toggled(self, on: bool):
        if on:
            if self.dataset is None or not self.dataset.has_time():
                self.btn_play.setChecked(False)
                self.play_note.setText("需要多时次数据才能播放。")
                return
            self._apply_play_interval()
            self._play_wait_since = None
            lo, hi = self.time_window()
            if self.time_slider.value() >= hi:
                self.time_slider.setValue(lo)          # 从窗口起点重播
            # 逐历元视图：手上还没有逐历元系数就**先一次解完**，别假装在播
            which = self.map_field.currentIndex()
            t0 = self.time_slider.value()
            if not self._epoch_view_ready(which, t0):
                name = self._map_view_name(which)
                if self.dataset.has_time():
                    self._ensure_series_async("播放")
                    self.play_note.setText(
                        f"正在一次解完 {self.dataset.ntime} 个历元（{name} 需要它们）…")
                else:
                    self.btn_play.setChecked(False)
                    self.play_note.setText(
                        f"「{name}」还没有系数，没法播放：先点「运行分析」。")
                    return
            else:
                self.play_note.setText(f"播放中（{lo + 1}–{hi + 1}）…")
            self.btn_play.setText("⏸ 暂停")
            self._play_timer.start()
        else:
            self._play_timer.stop()
            self._play_wait_since = None
            self.btn_play.setText("▶ 播放")

    def _play_step(self):
        """Advance one epoch; stop (do not wrap) at the window end.

        逐历元视图在**还没有逐历元系数**时不前进：以前它照常往前走，而每一帧画的
        都还是分析时那一个旧场（用户看到的就是"播放不起作用"或"播的还是原始图"）。
        现在先一次解完整条序列（几乎不比重算一个历元贵），解完再走。
        """
        _lo, hi = self.time_window()
        t = self.time_slider.value() + 1
        if t > hi:
            self._stop_play(f"播放结束（停在最后一个历元 {hi + 1}）。")
            return
        which = self.map_field.currentIndex()
        if not self._epoch_view_ready(which, t):
            name = self._map_view_name(which)
            if not self.dataset.has_time():
                self._stop_play(f"「{name}」还没有系数，播放无法继续：先点「运行分析」。")
                return
            if self._play_wait_since is None:
                self._play_wait_since = time.time()
            waited = time.time() - self._play_wait_since
            self._ensure_series_async("播放")
            w = self.series_worker
            if w is not None and not w.isRunning() \
                    and self._epoch_source_coeffs() is None:
                self._stop_play(f"逐历元分析没有成功，播放已停止"
                                "（详见状态栏/诊断报告）。")
                return
            self.play_note.setText(
                f"正在一次解完 {self.dataset.ntime} 个历元（{name} 需要它们）…"
                f"（已等 {waited:.0f} 秒）")
            if waited > self.PLAY_WAIT_SECONDS_MAX:
                self._stop_play(
                    f"逐历元分析等了 {waited:.0f} 秒还没结束，已停止播放。"
                    "可以先在「逐历元诊断」页点「批量分析」看进度，再回来播放。")
            return
        self._play_wait_since = None
        self.time_slider.setValue(t)

    def _render_map_frame(self):
        """Render the map canvas **synchronously** and return a PIL image."""
        from PIL import Image
        c = self.map_canvas.canvas
        c.draw()                     # draw_idle() would return before painting
        arr = np.asarray(c.buffer_rgba())
        return Image.fromarray(arr[:, :, :3].copy())

    def _export_gif(self):
        """Per-epoch GIF of the current view (frames = the time window)."""
        if self.dataset is None or not self.dataset.has_time():
            self.status("需要多时次数据才能导出动画。")
            return
        which = self.map_field.currentIndex()
        # 逐历元视图：先把整条序列解完（一次、可取消、会落盘缓存），然后每一帧只是
        # 从系数**综合**出来 —— 所以不会再有"逐帧补算"的等待，也不会重复同一帧。
        if which in self.EPOCH_VIEWS and self._epoch_source_coeffs() is None:
            if not self._ensure_series_sync("导出 GIF"):
                self.status("导出已取消：逐历元系数没有算完。")
                return
        lo, hi = self.time_window()
        n = hi - lo + 1
        path, _ = QFileDialog.getSaveFileName(self, "导出 GIF", "series.gif",
                                              "GIF (*.gif)")
        if not path:
            return
        keep = self.time_slider.value()
        dlg = QProgressDialog(f"正在渲染 {n} 帧…", "取消", 0, n, self)
        dlg.setWindowTitle("导出 GIF")
        dlg.setMinimumDuration(0)
        frames = []
        try:
            for k in range(n):
                dlg.setValue(k)
                QApplication.processEvents()
                if dlg.wasCanceled():
                    self.status("导出已取消。")
                    return
                self.time_slider.setValue(lo + k)
                QApplication.processEvents()
                frames.append(self._render_map_frame())
            dlg.setValue(n)
            if not frames:
                self.status("时间窗口里没有历元。")
                return
            fps = max(self.sp_fps.value(), 1)
            frames[0].save(path, save_all=True, append_images=frames[1:],
                           duration=int(1000 / fps), loop=0, optimize=False)
            self.status(f"已导出 {len(frames)} 帧 GIF：{path}（{fps} 帧/秒）")
        except Exception as exc:                                 # noqa: BLE001
            QMessageBox.warning(self, "导出 GIF", f"导出失败：{exc}")
        finally:
            dlg.close()
            self.time_slider.setValue(keep)

    # ------------------------------------------------------------------ D10
    def export_epochs(self):
        """Export one file per epoch, with pause/stop (D10)."""
        ds = self.dataset
        if ds is None or not ds.has_time():
            self.status("逐时次导出需要多时次数据（ntime > 1）。")
            return
        which = self.map_field.currentIndex()
        view = self.map_field.currentText()
        # ★ 「显示」选的是逐历元视图时，每一历元写出的必须是**它自己**的场：先把整条
        # 序列解完（一次、可取消、会落盘缓存）再导。以前只做过单历元分析时会拿那
        # 一个场写给每个历元 —— 一份"看起来正常、其实全错"的序列。
        if which in self.EPOCH_VIEWS and self._epoch_source_coeffs() is None:
            if not self._ensure_series_sync("逐时次导出"):
                self.status("逐时次导出已取消：逐历元系数没有算完。")
                return
        co_multi = None if which == 0 else self._epoch_source_coeffs()
        if which in self.EPOCH_VIEWS and co_multi is None:
            QMessageBox.information(
                self, "逐时次导出",
                f"逐时次导出「{self._map_view_name(which)}」需要**逐历元**系数，"
                "而这次没有算出来，为避免写出错误文件已中止。")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择逐时次导出的目录")
        if not out_dir:
            return
        lo, hi = self.time_window()
        ext = str(self.cb_export_ext.currentData() or ".nc")
        # 散点只能写文本表：.nc/.grd 是**网格**格式。选错了就明确纠正并说出来
        # （以前会一路跑到写文件才失败，用户看到 0 个文件却不知道为什么）。
        if ds.kind == "points" and ext in (".nc", ".grd"):
            ext = ".csv"
            self.cb_export_ext.setCurrentIndex(2)
            self.status("散点数据只能用 .csv 逐时次导出（.nc/.grd 是网格格式）"
                        "——已把导出格式改成 .csv。")
        prefix = self.edit_export_prefix.text().strip() or "epoch"
        # 单位在界面线程里读一次：导出 worker 在别的线程里跑，不该去碰控件
        unit_u = self.cb_field_unit.currentData() or "scalar"
        unit_t = self.cb_target_unit.currentData() or unit_u

        def _recon_for(k: int, unit: str):
            co_t = (co_multi.time_slice(k) if hasattr(co_multi, "time_slice")
                    else co_multi)
            return self._reconstruct_for(co_t, unit)

        def _field(k):
            base = (ds.grid[:, :, min(k, ds.grid.shape[2] - 1)] if ds.kind == "grid"
                    else ds.value_slice(min(k, ds.ntime - 1)))
            if which == 0 or co_multi is None:
                return base
            recon = _recon_for(k, unit_u)          # 这一历元自己的重建场
            if which == 1:
                return recon
            if which == 2:
                r = np.asarray(recon)
                return base - (r if r.shape == np.shape(base) else r.reshape(-1))
            if which == 3:
                return recon if unit_t == unit_u else _recon_for(k, unit_t)
            return base

        meta = {"source_file": ds.path, "view": view}

        def _write(k: int, out: str):
            vals = np.asarray(_field(k))
            m = dict(meta, epoch_index=k + 1)
            if ds.kind == "grid":
                shio.write_grid(out, ds.lat_vec, ds.lon_vec, vals, var="value",
                                meta=m, long_name="SHKit per-epoch field")
            else:
                shio.write_points(out, ds.lat, ds.lon, vals,
                                  comment=f"# epoch {k + 1}\n# view {view}")

        w = ExportWorker(_write, out_dir, prefix, lo, hi, ext=ext, var="value",
                         meta=meta, parent=self)
        dlg = ExportDialog(self, w, out_dir)
        self._export_dialog = dlg
        w.progressed.connect(dlg.on_progress)
        w.succeeded.connect(dlg.on_done)
        w.failed.connect(dlg.on_failed)
        w.cancelled.connect(dlg.on_cancelled)
        w.finished.connect(dlg.on_finished)
        dlg.show()
        w.start()

    # ------------------------------------------------------------------ D5
    def _build_epochs_tab(self) -> QWidget:
        """Per-epoch diagnostics page (D5)."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.btn_epochs_run = QPushButton("批量分析（逐历元）")
        self.btn_epochs_run.setToolTip(
            "对**所有**历元做一次批量球谐分析（一次调用算完，积分元与 Gram 诊断只算一次）。\n"
            "逐历元残差来自 report_fit=True。可疑历元只**高亮**，绝不自动剔除 —— "
            "剔除与否是科学判断，不是软件的默认。")
        self.btn_epochs_run.clicked.connect(self.run_series_analysis)
        row.addWidget(self.btn_epochs_run)
        self.btn_epochs_stop = QPushButton("停止")
        self.btn_epochs_stop.setEnabled(False)
        self.btn_epochs_stop.clicked.connect(self.stop_series_analysis)
        row.addWidget(self.btn_epochs_stop)
        self.chk_epochs_highlight = QCheckBox("高亮可疑历元（MAD 3σ）")
        self.chk_epochs_highlight.setChecked(True)
        self.chk_epochs_highlight.toggled.connect(
            lambda _b: self._fill_epochs_table())
        row.addWidget(self.chk_epochs_highlight)
        self.chk_subprocess = QCheckBox("独立进程计算（崩溃不影响界面）")
        self.chk_subprocess.setChecked(True)
        self.chk_subprocess.setToolTip(
            "在**子进程**里跑批量分析：求解器若是原生崩溃/被系统杀掉，"
            "倒下的只是子进程，界面还能继续用（线程做不到这一点）。\n"
            "当前环境不支持 spawn 时会自动回退到界面内线程，并明确写出来。")
        row.addWidget(self.chk_subprocess)
        row.addStretch(1)
        v.addLayout(row)
        self.epochs_note = _hint_label(
            "多时次数据可在这里看逐历元诊断；点某一行会把地图跳到那个历元。")
        v.addWidget(self.epochs_note)
        self.epochs_table = QTableWidget(0, 6)
        self.epochs_table.setHorizontalHeaderLabels(
            ["#", "日期", "相对 RMSE", "C00", "残差 RMS", "可疑"])
        self.epochs_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.epochs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.epochs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.epochs_table.cellClicked.connect(self._on_epoch_row_clicked)
        v.addWidget(self.epochs_table, 1)
        return w

    def run_series_analysis(self):
        """逐历元诊断页的「批量分析」按钮：一次解完整条序列。"""
        ds = self.dataset
        if ds is None:
            self.status("先载入数据。")
            return
        if not ds.has_time():
            self.epochs_note.setText(
                "逐历元诊断需要多时次数据（ntime > 1）——单时次没有「逐历元」。")
            return
        self._start_series("批量分析", ask_cache=True)

    def _start_series(self, reason: str, use_cache: bool = True,
                      ask_cache: bool = False) -> bool:
        """解**整条序列**（固定成本只付一次），必要时先吃本地缓存。

        D12：优先放在独立子进程里（原生崩溃不带走界面）；子进程起不来就**回退到
        线程并写明原因**，不静默换地方。

        ``ask_cache=True``（只有用户**显式点**「运行分析 / 批量分析」才传）时，命中
        缓存会先弹一句"读缓存还是重算"让用户定 —— 见 :meth:`_ask_series_cache`。
        导出 GIF / 切视图那些**自动补算**的路径不传（那里弹模态框会把人卡住）。

        返回 True = 已经有结果 / 已经开跑；False = 现在给不出结果（含用户取消）。
        """
        ds = self.dataset
        if ds is None or not ds.has_time():
            return False
        if self.series_worker is not None and self.series_worker.isRunning():
            self.status(f"{reason}：已经有一次逐历元分析在跑，等它结束。")
            return False
        params = self._collect_params()
        if use_cache:
            hit = coeff_cache.load(ds, params)
            if hit is not None and ask_cache:
                decision = self._ask_series_cache(hit, params)
                if decision == "cancel":
                    self.status(f"{reason}：已取消 —— 本地缓存没有读，也没有重算"
                                "（内存里的结果保持原样）。")
                    return False
                if decision == "recompute":
                    hit = None            # 落到下面真算，算完覆盖缓存
            if hit is not None:
                self._mark_cache_hit_progress()
                self._apply_series(hit["coeffs"], rep=None, reason=reason,
                                   table=hit.get("table"),
                                   shared=hit.get("shared"), cache=hit)
                return True
        use_sub = self.chk_subprocess.isChecked()
        fallback = ""
        if use_sub:
            from .subproc import spawn_available
            ok, why = spawn_available()
            if not ok:
                fallback = (f"独立进程不可用（{why}）→ 已回退到界面内线程计算；"
                            "结果一样，只是求解器崩溃会连带界面。")
                use_sub = False
        # 独立进程那条路要**先在界面进程里**把整条序列取出来 pickle 过去，所以必须
        # 先把懒加载补齐（带进度提示）。放在禁用按钮之前做，免得这里提前 return
        # 留下"运行按钮灰着"的状态。
        if use_sub and not self._ensure_all_values(reason):
            return False
        self.btn_epochs_run.setEnabled(False)
        self.btn_epochs_stop.setEnabled(True)
        if use_sub:
            # 1 GB 的 payload 要 pickle 进管道，这 2–3 s 没有任何百分比可报 ——
            # 给忙碌条 + 一句话，而不是一根 0% 的空标签条。
            est = int(getattr(ds, "npoints", 0) or 0) * int(ds.ntime) * 4
            self._progress_busy(
                f"{reason}：正在把 {ds.ntime} 个历元的数据交给独立进程"
                f"（约 {est / 1e6:.0f} MB）…")
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)
            self.progress.setVisible(True)
            self.progress_label.setText(f"{reason}：一次解完 {ds.ntime} 个历元…")
        self._t0 = time.time()
        self._series_reason = reason
        self._series_params = dict(params)
        self._series_cache = None
        self._series_fallback_note = fallback
        if use_sub:
            self.epochs_note.setText(
                f"{reason}：独立进程中一次解完 {ds.ntime} 个历元…（崩溃不影响界面）")
            self.series_worker = SubprocessRunner(
                "analyze_series",
                {"lat": ds.lat, "lon": ds.lon, "values": ds.all_values(),
                 "nmax": int(params["nmax"]),
                 "method": params.get("method", "quadrature"),
                 "rule": params.get("rule", "auto"),
                 "field_unit": params.get("field_unit", "scalar"),
                 "target_unit": params.get("target_unit"),
                 "gaussian_km": float(params.get("gaussian_km", 0.0) or 0.0),
                 "report_fit": True, "times": ds.time_axis(),
                 "longitude_fft": params.get("longitude_fft", "auto")},
                parent=self)
        else:
            self.epochs_note.setText(
                f"{reason}：一次解完 {ds.ntime} 个历元…"
                + (f"　⚠ {fallback}" if fallback else ""))
            self.series_worker = SeriesWorker(ds, params, parent=self)
        self.series_worker.progressed.connect(self.on_progress)
        self.series_worker.succeeded.connect(self.on_series_done)
        self.series_worker.failed.connect(self.on_analysis_failed)
        self.series_worker.cancelled.connect(self.on_analysis_cancelled)
        self.series_worker.finished.connect(self._on_series_finished)
        self.series_worker.start()
        self.status(f"{reason}：正在一次解完 {ds.ntime} 个历元"
                    "（固定成本只付一次；结果会缓存到本地）…")
        self._update_actions()
        return True

    def _on_series_finished(self):
        self.btn_epochs_run.setEnabled(True)
        self.btn_epochs_stop.setEnabled(False)
        self.progress.setVisible(False)
        self._update_actions()

    def stop_series_analysis(self):
        w = self.series_worker
        if w is not None and w.isRunning():
            w.cancel()
            self.epochs_note.setText("已请求停止（当前历元结束后退出）…")

    def _apply_series(self, coeffs, rep=None, reason: str = "", table=None,
                      shared=None, cache=None):
        """把"整条序列的解"装进界面（缓存命中与真算两条路共用）。

        ``rep`` 是真算出来的 :class:`SeriesReport`；缓存命中时只有 ``table`` /
        ``shared``（报表本来就是从它们渲染的），此时用它们重建一个同型对象。
        """
        from ..series import SeriesReport

        if rep is None:
            rep = SeriesReport(times=self.dataset.time_axis(),
                               table=dict(table or {}), per_epoch=[],
                               shared=dict(shared or {}),
                               meta={"from_cache": True})
        self.series_coeffs = coeffs
        self.series_report = rep
        self._series_cache = cache
        self._epoch_cache.clear()          # 换了一套系数 → 逐历元重建缓存作废
        self._coeff_version += 1           # 系数变了：水平形变那张图作废（不自动重算）
        self._fill_epochs_table()
        t = int(min(self.time_slider.value(), max(int(coeffs.ntime) - 1, 0)))
        self._apply_epoch_views(t)         # 谱 / 系数表 / 报告 / 单历元场
        self._refresh_map()
        # ⚠️ **不**在这里算水平形变（网格上一次全球梯度综合，256 个历元时很贵）：
        # 只把旧图作废并提示用户点按钮。
        self._mark_horizontal_stale(
            f"{reason}完成（{coeffs.ntime} 个历元，nmax={coeffs.nmax}）："
            "水平形变与这套系数不再对应。")
        head = (f"{reason}：一次解完 {coeffs.ntime} 个历元（nmax={coeffs.nmax}）"
                if cache is None else
                f"{reason}：逐历元系数来自本地缓存（{cache.get('created') or '?'}"
                f"，{os.path.basename(str(cache.get('path')))}），本次没有重算")
        self.status(head + "。地图/播放/GIF/逐时次导出都用它，按历元切即可。")
        if cache is not None:
            self.epochs_note.setText(
                f"逐历元系数来自本地缓存：{cache.get('path')}"
                f"（写于 {cache.get('created') or '?'}）。"
                "要按当前参数重算，点「批量分析」或「运行分析」。")
        self._update_actions()

    def _apply_epoch_views(self, t: int) -> None:
        """把"当前历元那一解"送进逐阶谱 / 系数统计 / 诊断报告 / 单历元场。

        多时次数据只有**一套**系数（3-D），所以换时次时这几页跟着切片，不需要
        再解一次；单时次数据集仍走 ``AnalysisWorker``（``self.report`` 由它给）。
        """
        ds = self.dataset
        co = self._epoch_source_coeffs()
        if ds is None or co is None:
            return
        n = int(np.asarray(co.C).shape[2])
        t = int(min(max(t, 0), max(n - 1, 0)))
        self.coeffs = co.time_slice(t) if hasattr(co, "time_slice") else co
        self._single_t = t
        rep = self.series_report
        if rep is not None and getattr(rep, "per_epoch", None):
            chunk = max(int((rep.shared or {}).get("epoch_chunk") or n), 1)
            base = rep.per_epoch[min(t // chunk, len(rep.per_epoch) - 1)]
            self.report = _epoch_report_from(base, rep.table, t)
        self.report_view.setHtml(self._series_report_html(t))
        self.recon, self.outfield, _ = self._epoch_fields(t)
        self._fill_spectrum()
        self._fill_coeff_table()
        self._update_unit_preview()
        self._values_dirty = True          # 数值表懒加载，切过去再填
        self._update_actions()

    def _series_report_html(self, t: int) -> str:
        """诊断报告页：整条序列的共享诊断 + 当前历元那一行。"""
        rep = self.series_report
        if rep is None:
            return ""
        n = max(int(rep.ntime), 1)
        t = int(min(max(t, 0), n - 1))
        tab = rep.table or {}
        labels = [("index", "#"), ("time", "日期"), ("data_rms", "数据 RMS"),
                  ("c00", "C00"), ("residual_rms", "残差 RMS"),
                  ("fit_rmse_rel", "相对 RMSE"), ("coverage", "覆盖率"),
                  ("gram_deviation", "max|K−I|")]
        rows = ["<style>td,th{padding:3px 10px 3px 0;}</style>",
                f"<h3>诊断报告（整条序列 · {n} 个历元，当前第 {t + 1} 个）</h3>",
                "<p style='color:#555;'>「逐阶谱 / 系数统计 / 导出系数」显示的是"
                f"<b>第 {t + 1} 个历元</b>那一解；全部历元的逐历元诊断见"
                "「逐历元诊断」页。</p>", "<table>"]
        for key, label in labels:
            v = tab.get(key)
            if v is None or t >= len(v):
                continue
            val = v[t]
            txt = (str(val) if isinstance(val, str)
                   else _fmt(float(val)) if key != "index"
                   else f"{int(val) + 1}")
            rows.append(f"<tr><th>{label}</th><td>{txt}</td></tr>")
        rows.append("</table>")
        html = "".join(rows)
        if self._series_cache is not None:
            html += ("<p style='color:#b35;'>⚠ 这套系数是<b>本地缓存</b>读回来的"
                     f"（{self._series_cache.get('created') or '?'}），"
                     "参数与缓存写入时一致；点「批量分析」可按当前参数重算。</p>")
        html += "<h3>序列摘要</h3><pre>" + rep.summary() + "</pre>"
        if getattr(rep, "per_epoch", None) and self.report is not None:
            base = _report_to_html(self.report, self.dataset)
            html += ("<h3>与历元无关的共享诊断</h3>"
                     "<p style='color:#555;'>积分元 / Gram / 推荐阶数这些量对每个"
                     "历元都一样，只算了一次（上面那条序列摘要里写了调用次数）。</p>"
                     + base)
        return html

    def on_series_done(self, coeffs, rep):
        # 批量分析可能是在**上一个**数据集上发起、跑到现在才回来（中途换了数据）：
        # 那样这套系数属于旧数据，装进来会让地图与逐历元表都画别人的东西。
        sender = self.sender()
        if getattr(sender, "dataset", None) is not None \
                and sender.dataset is not self.dataset:
            self.status("逐历元分析结果属于上一个数据集，已丢弃（当前数据未做该分析）。")
            return
        reason = self._series_reason or "批量分析"
        self._apply_series(coeffs, rep=rep, reason=reason)
        # 落盘：下次打开（或换视图、导 GIF、逐时次导出）直接读，不用再解一遍
        params = self._series_params or self._collect_params()
        path, note = coeff_cache.save(self.dataset, params, coeffs, rep)
        self.status(f"{reason}完成：{coeffs.ntime} 个历元，nmax={coeffs.nmax}。{note}")
        self._refresh_cache_hint()


    def _outlier_epochs(self):
        rep = getattr(self, "series_report", None)
        if rep is None or not self.chk_epochs_highlight.isChecked():
            return set()
        try:
            # outliers() 返回的是**历元序号列表**（不是 (序号, 分数) 对）
            return {int(i) for i in rep.outliers()}
        except Exception as exc:                                 # noqa: BLE001
            self.epochs_note.setText(f"离群检出失败（不影响表格）：{exc}")
            return set()

    def _fill_epochs_table(self):
        rep = getattr(self, "series_report", None)
        if rep is None:
            self.epochs_table.setRowCount(0)
            return
        table = rep.table
        n = len(table["index"]) if "index" in table else len(rep.per_epoch)
        bad = self._outlier_epochs()
        self.epochs_table.setRowCount(n)
        for i in range(n):
            def cell(key, fmt="{:.6g}"):
                v = table.get(key)
                if v is None or i >= len(v):
                    return "—"
                try:
                    fv = float(v[i])
                except (TypeError, ValueError):
                    return str(v[i])
                return "—" if not np.isfinite(fv) else fmt.format(fv)

            date = "—"
            ax = None if self.dataset is None else self.dataset.time_axis()
            if ax is not None and ax.has_dates and i < len(ax.values):
                date = str(ax.values[i])[:10]
            vals = [str(i + 1), date, cell("fit_rmse_rel"), cell("C00"),
                    cell("residual_rms"), "⚠" if i in bad else ""]
            for j, txt in enumerate(vals):
                item = QTableWidgetItem(txt)
                if i in bad:
                    item.setBackground(Qt.yellow)
                self.epochs_table.setItem(i, j, item)
        red = float(rep.meta.get("rms_reduction", float("nan")))
        fb = getattr(self, "_series_fallback_note", "")
        self.epochs_note.setText(
            f"{n} 个历元　权重规则 {rep.shared.get('weight_rule')}"
            f"　Gram 只算 {rep.shared.get('fill_gram', 0)} 次"
            f"（复用 {rep.shared.get('fill_gram_reused', 0)} 次）"
            + (f"　可疑历元 {len(bad)} 个（**只高亮、不剔除**）" if bad else "")
            + (f"　⚠ {fb}" if fb else "")
            + "　点某一行跳到该历元的地图")

    def _on_epoch_row_clicked(self, row: int, _col: int):
        """Jump the map to the clicked epoch."""
        if self.dataset is None or row >= self.dataset.ntime:
            return
        self.time_slider.setValue(row)
        self.tabs.setCurrentIndex(0)              # 地图页
        self.status(f"已跳到第 {row + 1} 个历元。")

    # ------------------------------------------------------------------ D6
    def _build_series_tab(self) -> QWidget:
        """Point/region time-series page (D6)."""
        from .canvases import SeriesCanvas

        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(QLabel("位置："))
        self.sp_series_lon = QDoubleSpinBox()
        self.sp_series_lat = QDoubleSpinBox()
        for s, lo, hi, dflt, tip in (
                (self.sp_series_lon, -180.0, 360.0, 100.0, "取样经度（度）"),
                (self.sp_series_lat, -90.0, 90.0, 30.0, "取样纬度（度）")):
            s.setRange(lo, hi)
            s.setDecimals(2)
            s.setValue(dflt)
            s.setToolTip(tip)
            s.valueChanged.connect(lambda _v: self._refresh_series_page())
        row.addWidget(QLabel("lon")); row.addWidget(self.sp_series_lon)
        row.addWidget(QLabel("lat")); row.addWidget(self.sp_series_lat)
        self.chk_series_region = QCheckBox("区域平均（±窗口）")
        self.chk_series_region.setToolTip(
            "勾选后不取单点，而是在该点周围 ±窗口 的经纬框内做**面积加权**平均"
            "（覆盖了球面多少会写在图下方）。")
        self.chk_series_region.toggled.connect(lambda _b: self._refresh_series_page())
        row.addWidget(self.chk_series_region)
        self.sp_series_half = QDoubleSpinBox()
        self.sp_series_half.setRange(0.5, 60.0)
        self.sp_series_half.setDecimals(1)
        self.sp_series_half.setValue(5.0)
        self.sp_series_half.setToolTip("区域窗口的半宽（度）")
        self.sp_series_half.valueChanged.connect(lambda _v: self._refresh_series_page())
        row.addWidget(QLabel("±")); row.addWidget(self.sp_series_half)
        self.chk_series_fit = QCheckBox("叠加趋势+周年拟合")
        self.chk_series_fit.setChecked(True)
        self.chk_series_fit.toggled.connect(lambda _b: self._refresh_series_page())
        row.addWidget(self.chk_series_fit)
        self.btn_series_pick = QPushButton("用悬停点")
        self.btn_series_pick.setToolTip("把坐标设成地图上最后悬停的那个点。")
        self.btn_series_pick.clicked.connect(self._use_last_hover_point)
        row.addWidget(self.btn_series_pick)
        row.addStretch(1)
        v.addLayout(row)
        self.series_canvas = SeriesCanvas(self)
        v.addWidget(self.series_canvas, 1)
        self.series_note = _hint_label("载入多时次数据后自动出曲线。")
        v.addWidget(self.series_note)
        return w

    def _use_last_hover_point(self):
        if getattr(self, "_last_hover", None) is None:
            self.status("还没有悬停过任何位置；先把鼠标移到地图上。")
            return
        lon, lat = self._last_hover
        self.sp_series_lon.setValue(float(lon))
        self.sp_series_lat.setValue(float(lat))

    def _series_xy(self):
        """``(years, values, label, note)`` for the current selection, or None."""
        ds = self.dataset
        if ds is None or not ds.has_time():
            return None
        lo, hi = self.time_window()
        lo = max(0, min(lo, ds.ntime - 1))
        hi = max(lo, min(hi, ds.ntime - 1))
        lon0 = float(self.sp_series_lon.value())
        lat0 = float(self.sp_series_lat.value())
        half = float(self.sp_series_half.value())
        ax = ds.time_axis()

        def _years(k):
            if ax is not None and ax.has_dates:
                try:
                    return float(np.asarray(ax.decimal_years)[k])
                except Exception:                                # noqa: BLE001
                    pass
            return float(k)

        years_all = np.array([_years(k) for k in range(ds.ntime)])
        # 到这里一定已经"全部历元可用"（_refresh_series_page 挡在前面），所以
        # all_values() 只是把内存里的立方体整形成一个视图，不会再触发读取。
        allv = ds.all_values()
        if self.chk_series_region.isChecked():
            m = (np.abs(ds.lat - lat0) <= half) & (np.abs(ds.lon - lon0) <= half)
            if not m.any():
                return None
            yy = allv[m, :]
            if ds.kind == "grid":
                sub_lat, sub_lon = ds.lat[m], ds.lon[m]
                # 面积加权（球带上 cos(lat) 是权重的一部分）—— 与 basin_average 同一口径
                from ..weights import compute_weights
                try:
                    w = np.asarray(compute_weights(
                        sub_lat, sub_lon, rule="auto").w, dtype=float)
                except Exception:                                # noqa: BLE001
                    w = np.ones(sub_lat.size)
            else:
                w = np.ones(int(m.sum()))
            w = w / max(w.sum(), 1e-300)
            vals = (yy * w[:, None]).sum(axis=0)
            label = f"区域平均 ±{half:g}°（{int(m.sum())} 点）"
            note = (f"区域平均：覆盖球面 "
                    f"{100.0 * m.sum() / max(ds.npoints, 1):.3g}% 的点")
        else:
            j = int(np.argmin((ds.lat - lat0) ** 2 + (ds.lon - lon0) ** 2))
            vals = allv[j, :]
            label = f"最近点 #{j}（lon={ds.lon[j]:.2f}, lat={ds.lat[j]:.2f}）"
            note = f"最近点距所选位置 {np.hypot(ds.lat[j] - lat0, ds.lon[j] - lon0):.3f}°"
        sl = slice(lo, hi + 1)
        return years_all[sl], np.asarray(vals, dtype=float)[sl], label, note, ax, sl

    def _refresh_series_page(self):
        ds = self.dataset
        ld = ds.lazy_loader() if ds is not None else None
        if ld is not None and ld.n_filled < ld.ntime:
            # 懒加载还没补完：**不要**在这里同步补齐（会卡界面好几秒），先画占位，
            # 后台补齐后由 _on_lazy_tick 自动重画。
            self.series_canvas.show_placeholder(
                f"正在后台读取全部时次（{ld.n_filled}/{ld.ntime}）…\n"
                f"读完后这里自动显示时间序列")
            self.series_note.setText(
                "数据量较大，时间序列需要全部历元；后台补齐中（地图/时次切换"
                "不受影响）。")
            return
        got = self._series_xy()
        if got is None:
            self.series_canvas.show_placeholder(
                "载入**多时次**数据后在这里显示时间序列\n（单时次数据没有曲线可画）")
            self.series_note.setText("需要多时次数据（ntime > 1）。")
            return
        years, vals, label, note, ax, sl = got
        trend = annual = band = None
        extra = ""
        if self.chk_series_fit.isChecked() and vals.size >= 4 and np.isfinite(vals).all():
            from ..timeseries import design_time
            tt = np.asarray(ax.values, dtype="datetime64[s]")[sl] if (
                ax is not None and ax.has_dates) else None
            if tt is not None:
                mid = tt[0] + (tt[-1] - tt[0]) / 2
                t = ((tt - mid).astype("timedelta64[s]").astype(float)
                     / (86400.0 * 365.25))
                names, A = design_time(t, poly_order=1, periods=(1.0,))
                sol, *_ = np.linalg.lstsq(A, vals, rcond=None)
                model = A @ sol
                trend = A[:, names.index("const")] * sol[names.index("const")] + \
                    A[:, names.index("trend")] * sol[names.index("trend")]
                annual = model - trend + sol[names.index("const")] * 0.0
                resid = vals - model
                band = np.full(vals.size, float(np.sqrt(np.mean(resid ** 2))))
                amp, doy = np.hypot(sol[names.index("annual_cos")],
                                    sol[names.index("annual_sin")]), None
                extra = (f"趋势 {sol[names.index('trend')]:+.4g}/年，"
                         f"周年振幅 {amp:.4g}，残差 RMS {band[0]:.3g}")
        self.series_canvas.show(years, vals, label=label, band=band,
                               trend=trend, annual=annual,
                               title="时间序列（十进制年）",
                               ylabel="数值", note=note)
        self.series_note.setText(f"{label}　{note}" + (f"　{extra}" if extra else ""))

    # ------------------------------------------------------------------ D7
    def _build_trend_tab(self) -> QWidget:
        """Trend / seasonal amplitude / seasonal phase maps (D7)."""
        from .canvases import TrendCanvas

        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.btn_trend = QPushButton("计算趋势与周年场")
        self.btn_trend.setToolTip(
            "对每个格点做「常数+趋势+周期项」的最小二乘拟合：\n"
            "趋势场 + **每个拟合了的周期**各一对（振幅场 / 相位场）。\n"
            "勾「含半年周期」就会多出半年振幅场与半年相位场 —— 拟合里一直有它，\n"
            "以前只是没画出来。\n"
            "与在系数域拟合再综合是**等价**的（拟合是逐历元线性的），"
            "实测相对差 ~1e-15。")
        self.btn_trend.clicked.connect(self._compute_trend_maps)
        row.addWidget(self.btn_trend)
        self.chk_trend_seasonal = QCheckBox("含半年周期")
        self.chk_trend_seasonal.setToolTip(
            "拟合里再加上 0.5 年的谐波；勾上后**多出**半年振幅场与半年相位场两张图。")
        row.addWidget(self.chk_trend_seasonal)
        self.chk_trend_coast = QCheckBox("海岸线")
        self.chk_trend_coast.setChecked(True)
        self.chk_trend_coast.setToolTip(
            "趋势/振幅/相位图上叠加随程序分发的离线海岸线（不联网）。")
        self.chk_trend_coast.toggled.connect(self._on_trend_coast)
        row.addWidget(self.chk_trend_coast)
        self.trend_note = _hint_label("尚未计算。")
        row.addWidget(self.trend_note, 1)
        v.addLayout(row)
        self.trend_canvas = TrendCanvas(self, color=self.color_scale)
        v.addWidget(self.trend_canvas, 1)
        return w

    def _on_trend_coast(self, on: bool) -> None:
        """海岸线开关：按上次的数据重画，不重新拟合。"""
        if getattr(self, "trend_canvas", None) is not None:
            self.trend_canvas.set_coast(on)

    def _compute_trend_maps(self):
        from ..timeseries import fit_series_maps

        ds = self.dataset
        if ds is None or not ds.has_time():
            self.trend_note.setText("需要多时次数据（ntime > 1）。")
            self.trend_canvas.show_placeholder("需要多时次数据")
            return
        ax = ds.time_axis()
        if ax is None or not ax.has_dates:
            self.trend_note.setText(
                "这条数据没有日期（只有时次序号），趋势/周年拟合需要真实时间 —— "
                "拒绝按等间隔硬算。请载入带 time 坐标的 nc。")
            self.trend_canvas.show_placeholder("缺少日期，无法拟合")
            return
        periods = (1.0, 0.5) if self.chk_trend_seasonal.isChecked() else (1.0,)
        # 趋势/周年是"每个格点对全部历元做最小二乘"——必须整条序列在内存里。
        if not self._ensure_all_values("趋势与周年场"):
            return
        # 这条是**同步**计算（补齐 + 拟合都在界面线程里），所以忙碌条要在这儿收工，
        # 否则它会一直转下去 —— 而下面这段拟合在大数据上确实要几秒。
        self._progress_busy("趋势与周年场：正在对全部历元逐格点做最小二乘…")
        try:
            if ds.kind == "grid":
                maps = fit_series_maps(np.asarray(ds.grid), ax, poly_order=1,
                                       periods=periods)
                self.trend_canvas.show(maps, ds.lat_vec, ds.lon_vec, is_grid=True,
                                       title=f"趋势与周年场（{ds.epoch_label(0)} … "
                                             f"{ds.epoch_label(ds.ntime - 1)}）")
            else:
                maps = fit_series_maps(ds.all_values(), ax, poly_order=1,
                                       periods=periods)
                self.trend_canvas.show(maps, ds.lat, ds.lon, is_grid=False,
                                       title="趋势与周年场（散点）")
        except Exception as exc:                                 # noqa: BLE001
            self.trend_note.setText(f"计算失败：{exc}")
            return
        finally:
            self._progress_idle()
        used = maps["n_cells_used"]
        total = int(np.size(maps["trend"]))
        nper = len(maps.get("seasonal") or [])
        per_txt = ("周年" if nper == 1 else "周年 + 半年" if nper == 2
                   else f"{nper} 个周期")
        self.trend_note.setText(
            f"拟合：{per_txt}　条件数 {maps['cond']:.3g}　用了 {used}/{total} 个格点"
            + (f"（{total - used} 个因缺测被排除）" if used < total else "")
            + (f"　共 {1 + 2 * nper} 张图" if nper else "")
            + "　每个周期的相位场各自把振幅<5%的格子遮掉（那里没有可谈的相位）")

    def _refresh_trend_page(self):
        """Time changed: the maps themselves do not depend on it, so just note it."""
        if getattr(self, "trend_canvas", None) is not None and \
                self.trend_canvas._axes:
            self.trend_note.setText(
                self.trend_note.text().split("　（当前时次")[0]
                + "　（当前时次变了不影响这些图：它们用了全部历元）")

    # ------------------------------------------------- 多历元系数：逐时次重建
    def _epoch_source_coeffs(self):
        """多历元系数从哪来：批量分析结果优先，其次单次分析留下的 3-D 系数。

        ``analyze_series``（逐历元诊断页的「批量分析」）一次解出**所有**历元，
        所以它一旦存在，地图上的重建/差值/输出场就应该跟着时次一起变 —— 这正是
        用户期望的行为（此前只有"原始"随滑块变，其余三张图冻在分析时那个历元）。
        """
        for co in (getattr(self, "series_coeffs", None), self.coeffs):
            if co is None:
                continue
            arr = np.asarray(co.C)
            if arr.ndim == 3 and arr.shape[2] > 1:
                return co
        return None

    def _reconstruct_for(self, coeffs, target: str):
        """与 ``AnalysisWorker._reconstruct`` 同一口径地把系数解回采样几何。"""
        from ..synthesis import synthesis, synthesis_grid
        ds = self.dataset
        if ds.kind == "grid":
            out = synthesis_grid(ds.lat_vec, ds.lon_vec, coeffs,
                                 target_unit=target or "scalar")
            return np.asarray(out)
        vals = np.asarray(synthesis(ds.lat, ds.lon, coeffs,
                                    target_unit=target or "scalar", chunk=100_000))
        if vals.ndim == 2 and vals.shape[1] == 1:
            vals = vals[:, 0]
        return vals.ravel()

    def _has_any_coeffs(self) -> bool:
        """手上到底有没有系数（逐历元的那一套，或单时次数据集的那一解）。"""
        return self.coeffs is not None or self._epoch_source_coeffs() is not None

    @staticmethod
    def _map_view_name(which: int) -> str:
        """「显示」下拉的短名字（提示文字里用）。"""
        return {0: "原始数据", 1: "重建场", 2: "差值", 3: "输出场"}.get(int(which),
                                                                  "该视图")

    def _epoch_view_ready(self, which: int, t: int) -> bool:
        """第 ``t`` 个历元上，``which`` 这个视图现在**画得出来**吗？

        "画得出来"的严格含义：手上有**属于这个历元**的系数/场。只有当视图是
        原始数据时才算无条件成立。
        """
        if which not in self.EPOCH_VIEWS or self.dataset is None:
            return which not in self.EPOCH_VIEWS
        return bool(self._epoch_fields(int(t))[2])

    def _epoch_fields(self, t: int):
        """当前时次的 ``(重建场, 输出场, 是否属于这个历元)``。

        第三项是关键：``self.recon`` 只是**某一个**历元（``_single_t``）的结果。
        以前无论滑块在哪都把它当成"当前历元的重建场"返回，于是换时次/播放时
        用户看到的是同一个旧场（甚至被当成"原始图"）。现在只有 t 真的对上才认，
        否则明说"没有"（``None, None, False``），由上层去补算或如实提示。

        缓存很关键：播放会按帧率连续换时次，每次重算 1 M 点的综合会让动画卡死。
        只留最近几个历元，避免把整条序列的重建都堆在内存里（256 个历元 × 1 M 点
        是 GB 量级）。
        """
        t = int(t)
        co = self._epoch_source_coeffs()
        if co is None:
            # 只有**单时次**数据集会走到这里：那时 self.recon 就是第 0 个历元的场
            if (self.recon is not None and self._single_t is not None
                    and t == int(self._single_t)):
                return self.recon, self.outfield, True
            return None, None, False
        n = int(np.asarray(co.C).shape[2])
        t = int(min(max(t, 0), n - 1))
        hit = self._epoch_cache.get(t)
        if hit is not None:
            return hit[0], hit[1], True
        u = self.cb_field_unit.currentData() or "scalar"
        tgt = self.cb_target_unit.currentData() or u
        co_t = co.time_slice(t) if hasattr(co, "time_slice") else co
        recon = self._reconstruct_for(co_t, u)
        outfield = recon if tgt == u else self._reconstruct_for(co_t, tgt)
        if len(self._epoch_cache) >= 4:                  # 只留最近 4 个历元
            self._epoch_cache.clear()
        self._epoch_cache[t] = (recon, outfield)
        return recon, outfield, True

    def _epoch_note(self, t: int) -> str:
        """这一张图用的是哪个历元的系数（多历元时说清来源）。"""
        if self._epoch_source_coeffs() is not None:
            src = "批量系数" if self._series_cache is None else "本地缓存的批量系数"
            return f"　{src} · 第 {t + 1} 个历元"
        if self.recon is not None and self._single_t is not None \
                and int(t) == int(self._single_t):
            return f"　单次分析系数 · 第 {t + 1} 个历元"
        return ""

    # ------------------------------------------- 逐历元系数：一次解完，不逐个补
    def _ensure_series_async(self, reason: str = "按需") -> bool:
        """需要逐历元系数但手上没有 → **一次解完整条序列**（不是逐个补算）。

        这是"运行分析/切视图/播放"共用的入口：整条序列的固定成本（积分元、Gram、
        勒让德递推）只付一次，每历元边际成本实测 1.4 ms（203 历元 / 1° 全球 /
        nmax=60），所以"解一个历元"和"解全部历元"几乎一样贵 —— 那就一次解完，
        顺带落盘缓存。以前这里是"给当前历元补算一次单历元分析"，播放时一帧一个
        历元，跟不上帧率（用户报的"播放不生效"）。
        """
        if self.dataset is None or not self.dataset.has_time():
            return False
        if self._epoch_source_coeffs() is not None:
            return True
        if self.series_worker is not None and self.series_worker.isRunning():
            return False
        return self._start_series(reason)

    def _ensure_series_sync(self, label: str) -> bool:
        """同步把整条序列解完（导出前用；进度对话框可取消）。

        返回 False = 用户取消 / 解不出来 —— 调用方必须**中止**，不能拿一份不完整的
        结果去写文件（那正是"把同一个场写给每个历元"的来源）。
        """
        ds = self.dataset
        if ds is None or not ds.has_time():
            return False
        if self._epoch_source_coeffs() is not None:
            return True
        dlg = QProgressDialog(f"{label}：正在一次解完 {ds.ntime} 个历元…",
                              "取消", 0, 1000, self)
        dlg.setWindowTitle(label)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.setLabelText(f"{label}：需要先解出逐历元系数（共 {ds.ntime} 个历元）…")
        w = self.series_worker
        if not (w is not None and w.isRunning()):
            if not self._start_series(label):
                dlg.close()
                return False
            w = self.series_worker
        if w is not None:
            w.progressed.connect(
                lambda msg, frac: (dlg.setLabelText(f"{label}：{msg}"),
                                   dlg.setValue(int(max(0.0, min(1.0, frac)) * 1000))))
        try:
            while w is not None and w.isRunning():
                QApplication.processEvents()
                if dlg.wasCanceled():
                    self.stop_series_analysis()
                    self.status(f"{label}：已取消（逐历元系数没有算完）。")
                    return False
                time.sleep(0.02)
            QApplication.processEvents()
        finally:
            dlg.close()
        return self._epoch_source_coeffs() is not None

    def _coeffs_for_epoch(self, t: int):
        """当前时次要用的系数：逐历元那一套的切片 → 单时次数据集的那一解。

        ⚠️ 单次分析结果只对 ``_single_t`` 那个历元成立 —— 不是那个历元就返回
        ``None``（上层会如实提示），不能拿它冒充"当前历元的系数"。
        """
        co = self._epoch_source_coeffs()
        if co is not None:
            n = int(np.asarray(co.C).shape[2])
            return (co.time_slice(min(t, n - 1)) if hasattr(co, "time_slice")
                    else co)
        if self._single_t is not None and int(t) == int(self._single_t):
            return self.coeffs
        return None

    def _unit_scale_note(self) -> str:
        """文件声明的变量单位不是米时，说清**什么会变、什么不会变**。

        ⚠️ 这条以前写成"结果会整体差 100 倍"，是**错的**（用户当场指出并复验）：
        SHKit 的正变换 ``C = a/f_u``、反变换 ``场 = C·f``、以及 EWH↔geoid↔形变
        之间的比值**全都是线性**的，所以输入什么单位，输出就是同一套单位的倍数 ——
        往返、各物理量之间的比例都不受单位影响，**不需要任何换算**。
        实测：同一个场按 m 与按 cm 各走一遍，系数之比恒为 100，重建场各自精确回到
        输入，水平形变之比恒为 100，geoid/EWH 的比值两次完全相同。

        真正与单位有关的是**绝对物理标注**：打印出来的 m / mm，以及
        "真实 GRACE 应在 mm 级"这类按 SI 写的量级自检 —— 文件若是 cm，
        这些数字要按 cm 读（÷100 才是米）。这里只提醒这一点。
        """
        ds, co = self.dataset, (self._coeffs_for_epoch(
            int(self.time_slider.value()) if self.time_slider is not None else 0))
        if ds is None or co is None:
            return ""
        vu = str((ds.meta or {}).get("variable_units") or "").strip().lower()
        if vu not in ("cm", "mm", "km"):
            return ""
        declared = getattr(self, "_declared_field_unit", None) \
            or self.cb_field_unit.currentData() or "scalar"
        if declared in ("geopotential", "scalar", "unknown", None):
            return ""
        factor = {"cm": 100.0, "mm": 1000.0, "km": 0.001}.get(vu, 1.0)
        return (f"　单位：文件声明 {vu} —— 本行的绝对量（mm/m）是按『这串数是米』"
                f"换算的，按 {vu} 读请 ÷{factor:g}；"
                "往返与各物理量之间的比例不受单位影响，无需换算。")

    # ------------------------------------------------------------------ D1
    def _build_color_row(self) -> QVBoxLayout:
        """Colour-scale panel (D1): auto / manual range / centre / apply.

        控件一行、**说明文字单独一行**：说明里带着实际数值（例如
        「自动 2–98%，对称于 0，→ [-22.0534, 22.0534]」）会很长，塞在「应用色标」
        右边时只剩一条窄缝，于是被折成五行贴在角落 —— 读起来像出错信息。
        单独一行整幅宽、自动折行才正常。
        """
        box = QVBoxLayout()
        box.setSpacing(2)
        row = QHBoxLayout()
        row.addWidget(QLabel("色标："))
        self.chk_color_auto = QCheckBox("Auto Range（2–98%）")
        self.chk_color_auto.setChecked(True)
        self.chk_color_auto.setToolTip(
            "勾选 = 用数据的 2–98% 分位数（稳健，抗离群点）；\n"
            "取消勾选 = 用右边的 vmin/vmax 手动值，留空的一端仍按自动值。")
        self.chk_color_auto.toggled.connect(self._on_color_auto_toggled)
        row.addWidget(self.chk_color_auto)

        self.edit_vmin = QLineEdit()
        self.edit_vmax = QLineEdit()
        self.edit_vcenter = QLineEdit()
        for e, tip in ((self.edit_vmin, "手动色标下界（留空 = 该端自动）"),
                       (self.edit_vmax, "手动色标上界（留空 = 该端自动）"),
                       (self.edit_vcenter, "以该值为中心对称展开（留空 = 不以某值居中）")):
            e.setMaximumWidth(110)
            e.setPlaceholderText("自动")
            e.setToolTip(tip)
            e.returnPressed.connect(self._on_apply_color)
        row.addWidget(QLabel("vmin")); row.addWidget(self.edit_vmin)
        row.addWidget(QLabel("vmax")); row.addWidget(self.edit_vmax)
        row.addWidget(QLabel("vcenter")); row.addWidget(self.edit_vcenter)
        self.btn_color_apply = QPushButton("应用色标")
        self.btn_color_apply.setToolTip(
            "把上面的设置应用到**地图页与趋势页**（原始/重建/差值/输出场、趋势/振幅/相位）。\n"
            "⚠️ 水平形变页**不跟随**这个色标：它有自己的一套（按本页数据自动定 2–98%、\n"
            "分量对称于 0）—— 地图的量级（例如 EWH 的 ±700 cm）套到毫米级的形变上，\n"
            "会把三张分量图压成一片颜色，看起来就是「画不出来」。")
        self.btn_color_apply.clicked.connect(self._on_apply_color)
        row.addWidget(self.btn_color_apply)
        row.addStretch(1)
        box.addLayout(row)

        self.color_note = _hint_label("")
        self.color_note.setWordWrap(True)
        box.addWidget(self.color_note)
        # start in the Auto state, i.e. with the manual fields真的灰掉
        self.edit_vmin.setEnabled(False)
        self.edit_vmax.setEnabled(False)
        return box

    def _parse_float(self, edit: QLineEdit, name: str):
        txt = edit.text().strip()
        if not txt:
            return None
        try:
            return float(txt)
        except ValueError:
            raise ValueError(f"{name} 不是数字：{txt!r}") from None

    def _on_color_auto_toggled(self, on: bool):
        self.color_scale.auto = bool(on)
        for e in (self.edit_vmin, self.edit_vmax):
            e.setEnabled(not on)
        self._refresh_map()

    def _on_symmetric_toggled(self, on: bool):
        self.color_scale.symmetric = bool(on)
        self._refresh_map()

    def _on_apply_color(self):
        try:
            self.color_scale.vmin = self._parse_float(self.edit_vmin, "vmin")
            self.color_scale.vmax = self._parse_float(self.edit_vmax, "vmax")
            c = self._parse_float(self.edit_vcenter, "vcenter")
        except ValueError as exc:
            QMessageBox.warning(self, "色标", str(exc))
            return
        if (self.color_scale.vmin is not None and self.color_scale.vmax is not None
                and self.color_scale.vmin >= self.color_scale.vmax):
            QMessageBox.warning(self, "色标", "vmin 必须小于 vmax。")
            return
        self.color_scale.vcenter = c
        if c is not None and self.map_symmetric.isChecked():
            # vcenter wins; keep the checkbox honest about what is in force
            self.map_symmetric.setChecked(False)
        self._refresh_map()
        # 水平形变页有自己的色标（不再与地图共用），所以这里**不重算**它
        self.status(f"色标已应用：{self.color_scale.describe()}")

    def _update_color_note(self, values) -> None:
        """说明当前色标是按什么定出来的（放在色标控件**下面**一整行）。"""
        try:
            self.color_note.setText("当前色标："
                                    + self.color_scale.describe(values))
        except Exception:                                        # noqa: BLE001
            self.color_note.setText("")

    # ------------------------------------------------------------------ D2
    def _on_map_hover(self, lon: float, lat: float, val: float, idx) -> None:
        """Hover readout: 原始 / Proc / (lon, lat) / epoch (D2).

        "原始" is the input datum at that location and "Proc" is whatever the map
        is currently showing, so the two are comparable at a glance -- which is the
        whole point of putting the difference map in the same family.
        """
        parts = []
        raw = self._hover_raw_at(lat, lon, idx)
        self._last_hover = (lon, lat)
        if raw is not None:
            parts.append(f"原始={_fmt(raw)}")
        parts.append(f"Proc={_fmt(val)}")
        parts.append(f"(lon={lon:.4f}°, lat={lat:.4f}°)")
        if idx is not None:
            parts.append(f"点 #{idx}")
        t = self._hover_time_label()
        if t:
            parts.append(t)
        self.status("　".join(parts))
        if getattr(self, "hover_label", None) is not None:
            self.hover_label.setText("　".join(parts))

    def _hover_raw_at(self, lat, lon, idx):
        """The input dataset's value at that location, or ``None``."""
        ds = self.dataset
        if ds is None or self.recon is None and self.outfield is None:
            pass
        if ds is None:
            return None
        t = min(self.time_slider.value(), max(ds.ntime - 1, 0))
        try:
            if idx is not None:
                return float(ds.value_slice(t)[idx])
            i = int(np.argmin(np.abs(np.asarray(ds.lat_vec) - lat)))
            j = int(np.argmin(np.abs(np.asarray(ds.lon_vec) - lon)))
            return float(ds.grid[i, j, min(t, ds.grid.shape[2] - 1)])
        except Exception:                                        # noqa: BLE001
            return None

    def _hover_time_label(self) -> str:
        ds = self.dataset
        if ds is None or not ds.has_time():
            return ""
        return ds.epoch_label(self.time_slider.value())

    def _build_horizontal_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.chk_horiz_coast = QCheckBox("海岸线")
        self.chk_horiz_coast.setChecked(True)
        self.chk_horiz_coast.toggled.connect(self._on_horiz_coast)
        row.addWidget(self.chk_horiz_coast)
        self.btn_horiz = QPushButton("计算水平形变")
        self.btn_horiz.setToolTip(
            "由当前系数算出水平地表位移 u_N / u_E（米）。**只在点这个按钮时算**：\n"
            "运行分析 / 批量分析 / 换时次都不会顺手把这张图算掉（网格上一次全球\n"
            "梯度综合，256 个历元时很贵），它们只把旧图作废并在下面写明。\n"
            "水平形变是位系数的<b>球面梯度</b>（不是逐阶乘法），因子为 R·l′ₙ/(1+k′ₙ)。\n"
            "规则整圈经度网格会自动走 FFT 经度路径（快 10× 以上），并在下面写明走了哪条。")
        self.btn_horiz.clicked.connect(lambda: self._refresh_horizontal(force=True))
        row.addWidget(self.btn_horiz)
        self.horiz_note = _hint_label("尚未计算。")
        row.addWidget(self.horiz_note, 1)
        v.addLayout(row)
        # ★ 本页有**自己的**色标（不再与地图页共用）：地图页的色标常是原始数据的
        # 量级（例如 EWH 的 ±700 cm），拿它来画毫米级的形变分量 → 全图一片颜色，
        # 看起来就是"画不出来"。本页按自己的数据自动定 2–98%（分量对称于 0）。
        self.horiz_color_scale = ColorScale()
        self.horiz_color_scale.symmetric = True
        self.horiz_canvas = HorizontalCanvas(self, color=self.horiz_color_scale)
        # 三行图按原尺寸画在一张很高的画布上，放进滚动区：以前塞进页签时会被压扁，
        # 第三行的标题还会盖住上面一行的色标。
        self.horiz_scroll = QScrollArea()
        self.horiz_scroll.setWidgetResizable(True)
        self.horiz_scroll.setWidget(self.horiz_canvas)
        self.horiz_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 画布按**自然像素高**（figsize × dpi）给最小高度，滚动区才会滚动而不是
        # 把三行图重新压扁 —— widgetResizable 单独用是不够的。
        _fig = self.horiz_canvas.figure
        self.horiz_canvas.setMinimumHeight(
            int(_fig.get_figheight() * _fig.dpi) + 40)
        v.addWidget(self.horiz_scroll, 1)
        return w

    def _on_tab_changed(self, idx: int):
        """Compute lazily: both the horizontal sweep and the value table cost real work."""
        try:
            name = self.tabs.tabText(idx)
            if name == "水平形变":
                # ★ 水平形变**只在点「计算水平形变」时算**：切到这个页签只显示
                # "点按钮算"（或上一次的结果，若它仍对应当前系数/时次）。
                if self._horiz_data is None or self._horiz_sig != self._horiz_key():
                    self._mark_horizontal_stale(
                        "水平形变按需计算（不会在运行分析时自动算）。"
                        if self._horiz_data is None else
                        "系数或时次已变：这张图对应的不是当前状态。")
            elif name == "数值表" and self._values_dirty:
                self._fill_values_table()
                self._values_dirty = False
            elif name == "时间序列":
                self._refresh_series_page()
        except Exception:                                    # noqa: BLE001
            pass

    def _on_horiz_coast(self, on: bool):
        """海岸线开关：只重画，**不重算**（网格上的一次全球梯度综合不便宜）。"""
        self.horiz_canvas.show_coast = bool(on)
        d = self._horiz_data
        if d is not None and self.horiz_canvas._drawn is not None:
            self.horiz_canvas._maps(d["north"], d["east"], d["lat"], d["lon"],
                                    is_grid=d["is_grid"], title=d["title"])

    # -------------------------------------------------------------- D3 数值表
    #: Above this many cells the table shows a strided subset (and says so) --
    #: a QTableWidget with 65k items takes seconds to fill, which a "look at the
    #: numbers" tab must not cost.
    MAX_TABLE_CELLS = 30000

    def _build_values_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.values_note = QLabel("尚未运行分析。")
        self.values_note.setStyleSheet("color:#555;")
        row.addWidget(self.values_note, 1)
        self.btn_values_refresh = QPushButton("刷新")
        self.btn_values_refresh.clicked.connect(self._fill_values_table)
        self.btn_values_copy = QPushButton("复制")
        self.btn_values_copy.setToolTip("把整张表以制表符分隔复制到剪贴板"
                                        "（可直接粘进 Excel）。")
        self.btn_values_copy.clicked.connect(self._copy_values)
        self.btn_values_export = QPushButton("导出 CSV…")
        self.btn_values_export.clicked.connect(self._export_values)
        for b in (self.btn_values_refresh, self.btn_values_copy,
                  self.btn_values_export):
            row.addWidget(b)
        v.addLayout(row)
        self.values_table = QTableWidget(0, 0)
        self.values_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.values_table, 1)
        # (header, rows) of whatever is currently shown; copy/export/tests read
        # this instead of scraping the widget
        self._values_payload = ([], [])
        return w

    def _current_field_for_table(self):
        """``(kind, label, lat/lon, values)`` for the current map view, or None."""
        ds = self.dataset
        if ds is None:
            return None
        t = min(self.time_slider.value(), max(ds.ntime - 1, 0))
        which = self.map_field.currentIndex()
        # 与地图同一口径：有多历元系数时，重建/差值/输出场都按当前历元；没有这个
        # 历元的系数时**不能**拿原始数据冒充，表头里要写明。
        recon, outfield, ready = self._epoch_fields(t)
        epo = self._epoch_note(t)
        if which in self.EPOCH_VIEWS and not ready:
            recon = outfield = None
            epo = (f"（第 {t + 1} 个历元的{self._map_view_name(which)}还没有系数，"
                   "暂列原始数据）")
        if ds.kind == "grid":
            base = ds.grid[:, :, min(t, ds.grid.shape[2] - 1)]
            if which == 0 or recon is None:
                return ("grid", f"原始网格，第 {t + 1} 个时次{epo}",
                        (ds.lat_vec, ds.lon_vec), base)
            if which == 1:
                return ("grid", f"重建网格，第 {t + 1} 个时次{epo}",
                        (ds.lat_vec, ds.lon_vec), np.asarray(recon))
            if which == 2:
                return ("grid", f"差值（原始 − 重建），第 {t + 1} 个时次{epo}",
                        (ds.lat_vec, ds.lon_vec), base - np.asarray(recon))
            if outfield is None:
                return ("grid", f"原始网格（无输出场），第 {t + 1} 个时次",
                        (ds.lat_vec, ds.lon_vec), base)
            return ("grid", f"输出场，第 {t + 1} 个时次{epo}",
                    (ds.lat_vec, ds.lon_vec), np.asarray(outfield))
        base = ds.value_slice(t)
        if which == 0 or recon is None:
            return ("points", f"原始散点，第 {t + 1} 个时次{epo}", (ds.lat, ds.lon), base)
        if which == 1:
            return ("points", f"重建场，第 {t + 1} 个时次{epo}", (ds.lat, ds.lon),
                    np.asarray(recon).ravel())
        if which == 2:
            return ("points", f"差值，第 {t + 1} 个时次{epo}", (ds.lat, ds.lon),
                    base - np.asarray(recon).ravel())
        if outfield is None:
            return ("points", f"原始散点（无输出场），第 {t + 1} 个时次",
                    (ds.lat, ds.lon), base)
        return ("points", f"输出场，第 {t + 1} 个时次{epo}", (ds.lat, ds.lon),
                np.asarray(outfield).ravel())

    def _fill_values_table(self):
        item = self._current_field_for_table()
        if item is None:
            self.values_table.clear()
            self.values_table.setRowCount(0)
            self.values_table.setColumnCount(0)
            self._values_payload = ([], [])
            self.values_note.setText("尚未运行分析。")
            return
        kind, label, axes, vals = item
        if kind == "grid":
            lat_vec, lon_vec = (np.asarray(a, dtype=float) for a in axes)
            g = np.asarray(vals, dtype=float)
            g = np.where(np.isfinite(g), g, np.nan)
            ilat = ilon = 1
            n = g.shape[0] * g.shape[1]
            if n > self.MAX_TABLE_CELLS:
                stride = int(np.ceil(np.sqrt(n / self.MAX_TABLE_CELLS)))
                ilat = ilon = max(stride, 1)
            lat_show = lat_vec[::ilat]
            lon_show = lon_vec[::ilon]
            sub = g[::ilat, ::ilon]
            header = ["lat \\ lon"] + [f"{x:g}" for x in lon_show]
            rows = [[f"{y:g}"] + [_fmt(v) for v in row]
                    for y, row in zip(lat_show, sub)]
            note = (f"{label}　网格 {g.shape[0]}×{g.shape[1]}"
                    + (f" → 按每 {ilat}×{ilon} 个格点抽样显示 {sub.shape[0]}×{sub.shape[1]}"
                       "（表太大，抽的是显示、不是数据）" if ilat > 1 else "（全部格点）"))
            self._set_table(header, rows)
            self.values_note.setText(note)
        else:
            lat, lon = (np.asarray(a, dtype=float).ravel() for a in axes)
            v = np.asarray(vals, dtype=float).ravel()
            step = max(1, int(np.ceil(v.size / self.MAX_TABLE_CELLS)))
            idx = np.arange(0, v.size, step)
            header = ["#", "lon", "lat", "value"]
            rows = [[str(i), f"{lon[i]:.6f}", f"{lat[i]:.6f}", _fmt(v[i])]
                    for i in idx]
            self._set_table(header, rows)
            self.values_note.setText(
                f"{label}　散点 {v.size} 个"
                + (f" → 每 {step} 个点显示 1 个（{len(idx)} 行）" if step > 1 else ""))

    def _set_table(self, header, rows) -> None:
        self._values_payload = (list(header), [list(r) for r in rows])
        tbl = self.values_table
        tbl.clear()
        tbl.setColumnCount(len(header))
        tbl.setRowCount(len(rows))
        tbl.setHorizontalHeaderLabels([str(h) for h in header])
        for i, row in enumerate(rows):
            for j, txt in enumerate(row):
                tbl.setItem(i, j, QTableWidgetItem(str(txt)))
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for j in range(len(header)):
            tbl.resizeColumnToContents(j)
            if tbl.columnWidth(j) > 160:
                tbl.setColumnWidth(j, 160)
        tbl.setVerticalHeaderLabels([str(i) for i in range(len(rows))])

    def _values_tsv(self) -> str:
        header, rows = self._values_payload
        if not rows:
            return ""
        return "\n".join(["\t".join(map(str, header))]
                         + ["\t".join(map(str, r)) for r in rows])

    def _copy_values(self):
        txt = self._values_tsv()
        if not txt:
            self.status("数值表为空，先载入数据并运行分析。")
            return
        QApplication.clipboard().setText(txt)
        self.status(f"已复制 {len(self._values_payload[1])} 行到剪贴板（制表符分隔）。")

    def _export_values(self):
        txt = self._values_tsv()
        if not txt:
            self.status("数值表为空，先载入数据并运行分析。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出数值表", "values.csv",
                                              "CSV (*.csv);;文本 (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            import csv as _csv
            fh.write("# " + self.values_note.text() + "\n")
            w = _csv.writer(fh)
            header, rows = self._values_payload
            w.writerow(header)
            w.writerows(rows)
        self.status(f"数值表已导出：{path}（{len(self._values_payload[1])} 行）")

    def _build_coeff_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        self.coeff_note = QLabel("尚未运行分析。")
        self.coeff_note.setStyleSheet("color:#555;")
        v.addWidget(self.coeff_note)
        self.coeff_table = QTableWidget(0, 5)
        self.coeff_table.setHorizontalHeaderLabels(
            ["阶 n", "逐阶 RMS", "power", "半波长 (km)", "占总功率比例"])
        self.coeff_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.coeff_table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.coeff_table, 1)
        return w

    # ------------------------------------------------------------ data dock
    def _build_data_dock(self):
        dock = QDockWidget("数据", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        body = QWidget()
        v = QVBoxLayout(body)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("未载入文件")
        v.addWidget(self.path_edit)

        row = QHBoxLayout()
        b_open_pts = QPushButton("打开散点…")
        b_open_pts.clicked.connect(lambda: self.open_file("points"))
        b_open_grid = QPushButton("打开网格…")
        b_open_grid.clicked.connect(lambda: self.open_file("grid"))
        row.addWidget(b_open_pts)
        row.addWidget(b_open_grid)
        v.addLayout(row)

        self.var_row = QWidget()
        fr = QHBoxLayout(self.var_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.addWidget(QLabel("变量："))
        self.var_combo = QComboBox()
        self.var_combo.setMinimumWidth(120)
        fr.addWidget(self.var_combo, 1)
        b_var = QPushButton("载入")
        b_var.clicked.connect(self.reload_with_var)
        fr.addWidget(b_var)
        self.var_row.setVisible(False)
        v.addWidget(self.var_row)

        self.time_row = QWidget()
        tr = QVBoxLayout(self.time_row)
        tr.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(QLabel("时次："))
        self.btn_time_prev = QPushButton("◀")
        self.btn_time_prev.setMaximumWidth(34)
        self.btn_time_prev.setToolTip("上一个时次（← 键同样可用）")
        self.btn_time_prev.clicked.connect(lambda: self._step_time(-1))
        head.addWidget(self.btn_time_prev)
        self.btn_time_next = QPushButton("▶")
        self.btn_time_next.setMaximumWidth(34)
        self.btn_time_next.setToolTip("下一个时次（→ 键同样可用）")
        self.btn_time_next.clicked.connect(lambda: self._step_time(+1))
        head.addWidget(self.btn_time_next)
        self.btn_time_summary = QPushButton("时间轴信息")
        self.btn_time_summary.setToolTip("弹出 TimeAxis.summary()：跨度、间隔、疑似缺测、"
                                         "重复历元与时间来源。")
        self.btn_time_summary.clicked.connect(self._show_time_summary)
        head.addWidget(self.btn_time_summary)
        head.addStretch(1)
        tr.addLayout(head)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        tr.addWidget(self.time_slider)
        # 起止范围：两个滑块圈出用户关心的窗口（只影响时间序列页/播放，
        # 不动地图当前时次 —— 地图永远画 slider 指的那个历元）
        rng_row = QHBoxLayout()
        rng_row.addWidget(QLabel("范围："))
        self.time_lo = QSlider(Qt.Horizontal)
        self.time_hi = QSlider(Qt.Horizontal)
        for s, tip in ((self.time_lo, "时间窗口起点（时间序列页/播放用）"),
                       (self.time_hi, "时间窗口终点（时间序列页/播放用）")):
            s.setMinimum(0)
            s.setMaximum(0)
            s.setToolTip(tip)
            s.valueChanged.connect(self._on_time_range_changed)
        rng_row.addWidget(self.time_lo, 1)
        rng_row.addWidget(self.time_hi, 1)
        self.btn_time_all = QPushButton("全部")
        self.btn_time_all.setToolTip("把范围设回全部历元")
        self.btn_time_all.clicked.connect(self._reset_time_range)
        rng_row.addWidget(self.btn_time_all)
        tr.addLayout(rng_row)
        self.time_label = QLabel("—")
        self.time_label.setWordWrap(True)
        tr.addWidget(self.time_label)
        self.time_row.setVisible(False)
        v.addWidget(self.time_row)

        self.data_summary = QTextBrowser()
        self.data_summary.setMinimumHeight(220)
        v.addWidget(self.data_summary, 1)

        b_demo = QPushButton("生成示例数据（球面散点合成场）")
        b_demo.clicked.connect(self.make_demo)
        v.addWidget(b_demo)

        dock.setWidget(body)
        dock.setMinimumWidth(300)
        self._data_dock = dock
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.data_dock = dock

    # ----------------------------------------------------------- param dock
    def _build_param_dock(self):
        dock = QDockWidget("分析参数", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        body = QWidget()
        v = QVBoxLayout(body)

        box = QGroupBox("球谐分析")
        form = QFormLayout(box)
        self.sp_nmax = QSpinBox()
        self.sp_nmax.setRange(1, 3000)
        self.sp_nmax.setValue(60)
        self.sp_nmax.setToolTip("最大阶数。系数个数 = (nmax+1)²。")
        form.addRow("最大阶数 nmax", self.sp_nmax)

        self.cb_rule = QComboBox()
        for r in ("auto", "dh", "grid", "lattice", "voronoi", "delaunay",
                  "uniform"):
            self.cb_rule.addItem(_RULE_LABELS[r], r)
        self.cb_rule.setToolTip(
            "积分元（立体角元）规则。\n"
            "· 全球等经纬网格用 dh；完整网格用 grid\n"
            "· 规则格网上的掩膜（区域散点但坐标等间距）用 lattice：\n"
            "  每个点代表一个格元，Σw = 点数 × 格元面积\n"
            "· 全球散点用 voronoi；不规则区域散点用 delaunay\n"
            "· uniform 只在等面积采样时正确")
        form.addRow("积分元规则", self.cb_rule)

        self.cb_method = QComboBox()
        for label, val in _METHODS:
            self.cb_method.addItem(label, val)
        self.cb_method.setToolTip(
            "估计方法（默认 <b>quadrature 直接求积</b>，效率优先）。\n"
            "GUI 不提供 auto —— 选哪条由你明确指定，软件不替你换。\n"
            "· quadrature：一次加权投影，最快；采样不是好的求积规则时会有混叠\n"
            "  （报告里的 gram max|K−I| 会告诉你）。\n"
            "· iterative：求积 + 理查森迭代校正（散点上 1 次常改善 ~250 倍）。\n"
            "· projection：区域数据的正确目标（区域外按 0 处理的全球投影）。\n"
            "· wlsq / cg：加权最小二乘 / 矩阵无关 CG，欠定或含缺口时用。\n"
            "需要「先求积、不健康才自动升级」时用 API 或 "
            "<code>shkit analyze --method auto</code>。")
        self.cb_method.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("估计方法", self.cb_method)

        self.sp_niter = QSpinBox()
        self.sp_niter.setRange(0, 20)
        self.sp_niter.setValue(0)      # 效率优先：只做一次直接求积
        self.sp_niter.setToolTip(
            "理查森迭代校正次数。\n"
            "<b>只对「估计方法 = iterative」生效</b>（其它方法下这一项是灰的）。\n"
            "0 = 只做一次直接求积；1 次在散点上实测可把往返精度改善约 250 倍，\n"
            "代价是多扫一遍数据。")
        form.addRow("迭代校正次数", self.sp_niter)

        self.ck_tau = QCheckBox("启用（只修数据看得见的方向）")
        self.ck_tau.setChecked(False)
        self.ck_tau.setToolTip(
            "仅对 projection 生效。\n"
            "不勾选 = 纯零填充全球投影：残差就是截断泄漏，\n"
            "  C00 = 面积占比，不做任何反演。\n"
            "勾选 = 只对集中因子 λ >= τ 的那部分方向做校正（除以 λ），\n"
            "  λ < τ 的方向保持原样，避免把区域外的噪声放大。\n"
            "τ 越接近 0 越接近未正则化的加权最小二乘（区域数据上会失效）。")
        form.addRow("采样偏差校正", self.ck_tau)

        self.sp_tau = QDoubleSpinBox()
        self.sp_tau.setRange(0.01, 1.0)
        self.sp_tau.setSingleStep(0.05)
        self.sp_tau.setDecimals(2)
        self.sp_tau.setValue(0.5)
        self.sp_tau.setToolTip("τ：只校正集中因子 >= τ 的方向。")
        form.addRow("校正阈值 τ", self.sp_tau)

        self.cb_norm = QComboBox()
        for label, val in _NORMS:
            self.cb_norm.addItem(label, val)
        form.addRow("权重归一化", self.cb_norm)

        self.cb_reg = QComboBox()
        for label, val in _REGS:
            self.cb_reg.addItem(label, val)
        form.addRow("正则化", self.cb_reg)

        self.ed_alpha = QLineEdit()
        self.ed_alpha.setPlaceholderText("留空 = 自动（L 曲线）")
        form.addRow("正则参数 alpha", self.ed_alpha)
        v.addWidget(box)

        box2 = QGroupBox("重建 / 显示")
        form2 = QFormLayout(box2)
        self.sp_gauss = QDoubleSpinBox()
        self.sp_gauss.setRange(0.0, 5000.0)
        self.sp_gauss.setDecimals(1)
        self.sp_gauss.setSingleStep(50.0)
        self.sp_gauss.setValue(0.0)
        self.sp_gauss.setSuffix(" km")
        self.sp_gauss.setToolTip(
            "各向同性高斯平滑半径（0.5 幅度半宽）；0 表示不平滑。\n"
            "分析完成后<b>直接乘到系数上</b>：导出的系数、逐阶谱、系数表、\n"
            "重建场与输出场都是同一个平滑结果（只施加一次）。\n"
            "注意半径相对于 nmax 的意义：nmax 很低时（如 12 阶），\n"
            "300 km 只压掉约 1%，看起来几乎没变化；半径取大一些才明显。")
        form2.addRow("高斯平滑", self.sp_gauss)

        self.chk_auto_map = QCheckBox("分析完成后自动显示重建场")
        self.chk_auto_map.setChecked(True)
        form2.addRow("", self.chk_auto_map)
        v.addWidget(box2)

        # ---- 物理量与公式（正变换 / 反变换）--------------------------------
        box3 = QGroupBox("物理量与公式")
        form3 = QFormLayout(box3)
        note = _hint_label(
            "两个下拉各选 <b>一条公式</b>，中间是普通球谐系数 C_nm：\n"
            "　正变换（输入是）：网格 → C_nm = a_nm / f_u\n"
            "　反变换一：重建场 = C_nm × f_u（永远与输入同量）\n"
            "　反变换二（输出为）：输出场 = C_nm × f_t\n"
            "导出的系数文件装的永远是 C_nm。",
            style="color:#666;font-size:9pt;", bold=False)
        form3.addRow(note)

        self.cb_field_unit = QComboBox()
        for label, val in _FIELD_UNITS:
            self.cb_field_unit.addItem(label, val)
        self.cb_field_unit.setToolTip(
            "【正变换公式】这串数字物理上是什么。\n"
            "它决定把该网格自身的系数 a_nm 除以哪个 f_u 得到普通球谐系数 C_nm：\n"
            "　普通网格 → f_u = 1（网格本身就是那套无量纲位系数场）\n"
            "　geoid → ÷R；EWH → ÷Aₙ；径向形变 → ÷(R h′ₙ/(1+k′ₙ))\n"
            "所以改这一项<b>会改变导出的系数</b>：同一块网格声明成 geoid 与声明成 EWH，"
            "是两个不同的物理场，第 2 阶差 A₂/R ≈ 13.2 倍（逐阶不同）。")
        self.cb_field_unit.currentIndexChanged.connect(self._on_input_unit_changed)
        form3.addRow("输入是", self.cb_field_unit)

        self.lbl_unit_hint = _hint_label()
        form3.addRow("", self.lbl_unit_hint)

        self.cb_target_unit = QComboBox()
        for label, val in _TARGET_UNITS:
            self.cb_target_unit.addItem(label, val)
        self.cb_target_unit.setToolTip(
            "【反变换公式·输出场】输出场要是什么物理量（乘哪个 f_t）。\n"
            "它既不改变导出的系数（永远是 C_nm），也不改变重建场（恒与输入同量），\n"
            "只决定输出场放大的倍数：\n"
            "　不换算 = 乘输入那一档的 f_u，输出场就是重建场。\n"
            "换算按逐阶因子进行：C_nm→EWH 乘 Aₙ；C_nm→水准面 乘常数 R。")
        self.cb_target_unit.currentIndexChanged.connect(self._on_target_changed)
        form3.addRow("输出为", self.cb_target_unit)

        self.lbl_target_hint = _hint_label()
        form3.addRow("", self.lbl_target_hint)

        # 实时数值预览：把两条公式具体到某个系数上，直接显示成数字
        self.lbl_unit_preview = _hint_label(style="color:#357;font-size:9pt;")
        form3.addRow("", self.lbl_unit_preview)
        v.addWidget(box3)
        self._on_field_unit_changed()

        row = QHBoxLayout()
        self.btn_run = QPushButton("运行分析")
        self.btn_run.setDefault(True)
        self.btn_run.setToolTip(
            "多时次数据：一次解完**全部**历元（积分元与法方程只组装一次，每历元\n"
            "边际成本实测毫秒级），并把逐历元系数**缓存到数据文件旁边**；之后地图、\n"
            "播放、导出 GIF、逐时次导出都只是「切片 + 综合」，不必再解方程。\n"
            "单时次数据：解当前这一个历元。")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_run, 2)
        row.addWidget(self.btn_stop, 1)
        v.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        # ★ 出厂就藏起来。QProgressBar 塞进布局就是**可见**的，而它以前只在
        # 「序列线程跑完」那条路上被隐藏 —— 于是"本地已有缓存 → 运行分析"
        # （不跑线程）会让一根 0% 的条永远挂在界面上，看起来就是"卡在 0 不动"。
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        self.progress_label = _hint_label(style="color:#555;")
        v.addWidget(self.progress_label)

        # 这一行回答"缓存现在是什么状态、占了多少"（文件菜单里能就地清理）
        self.cache_hint = _hint_label(style="color:#666;font-size:9pt;")
        self.cache_hint.setText("本地暂无该数据的缓存。")
        v.addWidget(self.cache_hint)

        v.addStretch(1)
        dock.setWidget(body)
        dock.setMinimumWidth(320)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.param_dock = dock

    # --------------------------------------------------------------- menus
    def _build_menus(self):
        mb = self.menuBar()

        m = mb.addMenu("文件(&F)")
        self.act_open_pts = QAction("打开散点…", self)
        self.act_open_pts.setShortcut(QKeySequence.Open)
        self.act_open_pts.triggered.connect(lambda: self.open_file("points"))
        m.addAction(self.act_open_pts)

        self.act_open_grid = QAction("打开网格…", self)
        self.act_open_grid.triggered.connect(lambda: self.open_file("grid"))
        m.addAction(self.act_open_grid)

        m.addSeparator()
        self.act_demo = QAction("生成示例数据", self)
        self.act_demo.triggered.connect(self.make_demo)
        m.addAction(self.act_demo)

        m.addSeparator()
        self.act_exp_coeffs = QAction("导出球谐系数…", self)
        self.act_exp_coeffs.triggered.connect(self.export_coeffs)
        m.addAction(self.act_exp_coeffs)

        self.act_exp_recon = QAction("导出重建场（与输入同量）…", self)
        self.act_exp_recon.triggered.connect(lambda: self.export_field("recon"))
        m.addAction(self.act_exp_recon)

        self.act_exp_outfield = QAction("导出输出场（按「输出为」）…", self)
        self.act_exp_outfield.triggered.connect(lambda: self.export_field("outfield"))
        m.addAction(self.act_exp_outfield)

        self.act_exp_report = QAction("导出诊断报告…", self)
        self.act_exp_report.triggered.connect(self.export_report)
        m.addAction(self.act_exp_report)

        self.act_exp_horiz = QAction("导出水平形变（当前历元）…", self)
        self.act_exp_horiz.setToolTip(
            "把当前历元的水平地表位移写出来：每个历元 3 个文件\n"
            "（u_N 北分量 / u_E 东分量 / |u_h| 合矢量，单位 m）。\n"
            "水平形变是位系数的球面梯度，不是逐阶乘法，所以单独一项。")
        self.act_exp_horiz.triggered.connect(
            lambda: self.export_horizontal("current"))
        m.addAction(self.act_exp_horiz)

        self.act_exp_horiz_series = QAction("导出水平形变（时间窗内逐时次）…", self)
        self.act_exp_horiz_series.setToolTip(
            "时间窗口内每个历元各写 3 个文件（可暂停/停止）。\n"
            "散点上每个历元要直接扫一遍梯度，很慢 —— 会先给出估算量让你确认。")
        self.act_exp_horiz_series.triggered.connect(
            lambda: self.export_horizontal("window"))
        m.addAction(self.act_exp_horiz_series)

        m.addSeparator()
        self.act_cache = QAction("本地缓存…", self)
        self.act_cache.setToolTip(
            "看看系数缓存占了多少地方，以及就地清理（缓存只是省时间，删掉不影响结果）")
        self.act_cache.triggered.connect(self.show_cache_manager)
        m.addAction(self.act_cache)

        m.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)

        m = mb.addMenu("分析(&A)")
        self.act_run = QAction("运行分析", self)
        self.act_run.setShortcut("F5")
        self.act_run.triggered.connect(self.run_analysis)
        m.addAction(self.act_run)

        self.act_stop = QAction("停止", self)
        self.act_stop.setShortcut("Esc")
        self.act_stop.triggered.connect(self.stop_analysis)
        m.addAction(self.act_stop)

        m = mb.addMenu("帮助(&H)")
        act_guide = QAction("📘 使用说明", self)
        act_guide.setShortcut("F1")
        act_guide.triggered.connect(self.show_guide)
        m.addAction(act_guide)

        act_about = QAction("ℹ️ 关于 / 作者信息", self)
        act_about.triggered.connect(self.show_about)
        m.addAction(act_about)

        act_wx = QAction("📱 课题组公众号", self)
        act_wx.triggered.connect(self.show_wechat)
        m.addAction(act_wx)

        m.addSeparator()

        act_lic = QAction("第三方许可与声明…", self)
        act_lic.triggered.connect(self.show_licenses)
        m.addAction(act_lic)

        act_lgpl = QAction("GNU LGPL v3 全文…", self)
        act_lgpl.triggered.connect(lambda: self.show_license_text(
            "licenses/LGPL-3.0.txt", "GNU Lesser General Public License v3"))
        m.addAction(act_lgpl)

        act_gpl = QAction("GNU GPL v3 全文…", self)
        act_gpl.triggered.connect(lambda: self.show_license_text(
            "licenses/GPL-3.0.txt", "GNU General Public License v3"))
        m.addAction(act_gpl)

    def _build_status(self):
        sb = self.statusBar()
        self.status_label = QLabel("就绪")
        sb.addWidget(self.status_label, 1)
        # D2: the hover readout lives here so it never fights the status text
        self.hover_label = QLabel("")
        self.hover_label.setStyleSheet("color:#333;")
        self.hover_label.setToolTip(
            "鼠标在地图上移动时显示：原始数据值 / 当前显示值（Proc）/ 坐标 / "
            "第几个时次（日期）。\"Proc\" 是图上正在画的那个场（原始/重建/差值/输出场）。")
        sb.addPermanentWidget(self.hover_label, 2)
        self.status_time = QLabel("")
        sb.addPermanentWidget(self.status_time)

    # =============================================================== helpers
    def status(self, text: str):
        self.status_label.setText(text)

    def _update_actions(self):
        has_data = self.dataset is not None
        has_result = self.coeffs is not None
        running = ((self.analysis_worker is not None
                    and self.analysis_worker.isRunning())
                   or (self.series_worker is not None
                       and self.series_worker.isRunning()))
        horiz_running = (self.horiz_worker is not None
                         and self.horiz_worker.isRunning())
        self.btn_run.setEnabled(has_data and not running)
        self.act_run.setEnabled(has_data and not running)
        self.btn_stop.setEnabled(running or horiz_running)
        self.act_stop.setEnabled(running or horiz_running)
        self.act_exp_coeffs.setEnabled(has_result)
        self.act_exp_recon.setEnabled(has_result and self.recon is not None)
        self.act_exp_outfield.setEnabled(has_result and self.outfield is not None)
        self.act_exp_report.setEnabled(has_result)
        # 水平形变导出：有系数就能导当前历元；逐时次那条要求逐历元系数
        self.act_exp_horiz.setEnabled(has_result)
        self.act_exp_horiz_series.setEnabled(
            has_result and self.dataset is not None and self.dataset.has_time()
            and self._epoch_source_coeffs() is not None)
        self.cb_rule.setEnabled(not running)
        self.cb_method.setEnabled(not running)
        self.sp_nmax.setEnabled(not running)
        # 迭代校正次数只对 iterative 有意义，其它方法下灰掉，避免"设了没反应"
        self.sp_niter.setEnabled(not running
                                 and self.cb_method.currentData() == "iterative")

    def _on_method_changed(self, *_):
        """方法换了：迭代校正次数只对 iterative 有效，其余灰掉。"""
        m = self.cb_method.currentData()
        self.sp_niter.setEnabled(m == "iterative")
        if m == "quadrature":
            self.status("估计方法 = quadrature（直接求积，默认）。"
                        "若采样不是好的求积规则，诊断报告里的 gram max|K−I| "
                        "会提示改用 wlsq/cg。")
        elif m == "projection":
            self.status("估计方法 = projection（零填充全球投影）——"
                        "区域数据的正确目标，C00 = 面积占比。")
        else:
            self.status(f"估计方法 = {m}")

    # ============================================================ data input
    def open_file(self, kind: str):
        if kind == "grid":
            filt = "网格数据 (*.nc *.grd *.npy *.csv *.txt);;所有文件 (*)"
        else:
            filt = ("散点数据 (*.csv *.txt *.dat *.tsv *.npy *.xlsx);;"
                    "所有文件 (*)")
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", filt)
        if path:
            self.load_path(path, kind)

    def load_path(self, path: str, kind: str = "auto", var=None):
        self._stop_workers_for_reload()
        self.status(f"正在读取 {os.path.basename(path)} …")
        self.load_worker = LoadWorker(path, kind, var=var, parent=self)
        self.load_worker.succeeded.connect(self.on_loaded)
        self.load_worker.failed.connect(self.on_load_failed)
        self.load_worker.start()
        # ⚠️ 这里**不再**为了列变量而再开一次这个 nc 文件：那会与上面的 worker
        # 并发打开同一份文件，而 netCDF4/HDF5 的 C 库是进程级全局状态 —— 实测
        # 症状就是"第一次打开 mascon 必然报 RuntimeError: NetCDF: Not a valid ID"。
        # 变量名单改由读文件时顺带带出来（meta['nc_variables']），载入完成后填充。
        self.var_row.setVisible(str(path).lower().endswith(".nc"))

    def _stop_workers_for_reload(self):
        """换文件/换变量前的收尾：先停 worker，再关掉旧数据集的懒加载句柄。

        懒加载的立方体持有 nc 文件句柄**和**一份 ASCII 临时副本（netCDF4 打不开
        非 ASCII 路径，只能先复制过去），不关就会一直占着磁盘与句柄；换变量 /
        重新载入都会再开一份。⚠️ 顺序不能反：正在读的线程若撞上已关闭的数据集，
        会拿到 ``NetCDF: Not a valid ID``。
        """
        for w in (self.load_worker, self.analysis_worker, self.series_worker,
                  self.horiz_worker,
                  getattr(self, "export_worker", None)):
            if w is None or not w.isRunning():
                continue
            try:
                if hasattr(w, "cancel"):
                    w.cancel()
                elif hasattr(w, "requestInterruption"):
                    w.requestInterruption()
                w.wait(5000)
            except Exception:                                    # noqa: BLE001
                pass
        self._lazy_timer.stop()
        self._lazy_ld = None
        # 换数据：上一份的半截进度条没有意义了（可能停在 0%/100%）。隐藏它，
        # 免得刚载入新数据就看见一根"不动的条"。
        self._progress_idle()
        old = self.dataset
        self.dataset = None
        if old is not None:
            old.close()

    def _fill_var_combo(self, ds) -> None:
        """把文件里的变量名填进下拉框（数据来自同一次读取，不再开文件）。"""
        m = ds.meta or {}
        names = list(m.get("nc_variables") or m.get("data_variables") or [])
        if not names:
            return
        cur = str(m.get("variable") or "")
        blocked = self.var_combo.blockSignals(True)
        try:
            self.var_combo.clear()
            self.var_combo.addItems(names)
            if cur in names:
                self.var_combo.setCurrentText(cur)
        finally:
            self.var_combo.blockSignals(blocked)

    def reload_with_var(self):
        if self.dataset is not None:
            self.load_path(self.dataset.path, self.dataset.kind,
                           var=self.var_combo.currentText() or None)

    def on_load_failed(self, tb: str):
        self.status("读取失败")
        QMessageBox.critical(self, "读取失败", tb.strip().splitlines()[-1]
                             if tb.strip() else "未知错误")
        self._update_actions()

    def on_loaded(self, ds: Dataset):
        self.dataset = ds
        self._loading = True               # 载入期间不自动开跑（见 _epoch_view_unavailable）
        self.coeffs = self.report = self.recon = self.outfield = None
        self._single_t = None              # 新数据 → 不知道任何历元的系数
        self._map_ds = None                # 画布上那张图不再是"当前数据"的
        self._map_view_idx = None
        self._play_wait_since = None
        self._declared_field_unit = None
        self.series_coeffs = None          # 换数据集 → 逐历元结果也作废
        self.series_report = None
        self._series_cache = None
        self._series_reason = ""
        self._series_params = None
        self._epoch_cache = {}             # 逐历元重建缓存同样作废
        self._coeff_version += 1           # 新数据：水平形变那张图也作废
        self._horiz_data = None
        self._horiz_sig = None
        if getattr(self, "horiz_canvas", None) is not None:
            self.horiz_canvas.show_placeholder(
                "运行分析后在「计算水平形变」里按需计算\n"
                "（不会在运行分析时自动算）")
            self.horiz_note.setText("尚未计算。")
        # 换了数据集，上一份已知真值即失效。「真实场」曲线只对 make_demo()
        # 合成的示例场成立；残留下来会被 _fill_spectrum() 按「阶数相同」误判，
        # 画到任意真实数据（无真值）的逐阶谱上。
        self._real_truth = None
        self.path_edit.setText(ds.path)
        self.path_edit.setToolTip(ds.path)
        self._fill_var_combo(ds)           # nc 变量名单来自同一次读取（不再开文件）

        # auto-pick the integration-element rule for this sampling
        rule = ds.suggested_rule()
        idx = self.cb_rule.findData(rule)
        if idx >= 0:
            self.cb_rule.setCurrentIndex(idx)

        self.time_slider.setMaximum(max(ds.ntime - 1, 0))
        self.time_slider.setValue(0)
        # 逐时次导出的默认格式跟着数据种类走：网格 .nc，散点 .csv（散点写不了 nc/grd）
        _ext_idx = 0 if ds.kind == "grid" else 2
        if 0 <= _ext_idx < self.cb_export_ext.count():
            self.cb_export_ext.setCurrentIndex(_ext_idx)
        for s in (self.time_lo, self.time_hi):
            s.blockSignals(True)
            s.setMaximum(max(ds.ntime - 1, 0))
            s.blockSignals(False)
        self._reset_time_range()
        self.time_row.setVisible(ds.has_time())
        self._update_time_label()
        # D6/D7 依赖全部历元：换数据后重新出图
        self._refresh_series_page()
        self.trend_canvas.show_placeholder("多时次数据：点「计算趋势与周年场」")
        self.trend_note.setText("尚未计算。")

        self.data_summary.setPlainText(ds.summary())
        self.report_view.setPlainText(
            "尚未运行分析。\n\n载入成功：\n" + ds.summary())
        self._refresh_summary(0)
        self.coeff_table.setRowCount(0)
        self.coeff_note.setText("尚未运行分析。")
        self.spectrum_canvas.figure.clear()
        self.spectrum_canvas.refresh()
        # 本地缓存：上次用**同一套参数**解过这条序列就直接读回来（用户不必再等
        # 一次求解；缓存键含源文件指纹与全部影响系数的参数，所以错配不会命中）。
        # 命中与否都会在状态栏写明，不会让人以为"这是本次刚算的"。
        #
        # ★ 没命中但**这份数据有缓存**时，把参数切回缓存那一套再读一次。这条是
        # 用户点名要的：「识别到缓存时，输入是 这个应该自动切换吧」。不切就会**来回
        # 打转**：界面的默认「输入是」= geopotential，而 `field_unit` 是键的一部分
        # （同一块网格声明成 geoid/EWH/geopotential 是三个不同的物理场，除的 f_u
        # 不同，**系数真的不一样**），于是默认值永远对不上 → 每次点运行都重算，而重算
        # 又把缓存写成默认那套 → 下次打开还是对不上。实测那份 mascon 就这样重算了两次。
        cache_note = ""
        self._cache_adopt_note = ""       # 本次载入"按缓存切过参数"的摘要（常驻提示用）
        if ds.has_time():
            hit = coeff_cache.load(ds, self._collect_params())
            if hit is None:
                found = coeff_cache.find_any(ds)
                if found is not None:
                    summary = self._adopt_cached_params(found)
                    self._cache_adopt_note = summary
                    hit = coeff_cache.load(ds, self._collect_params())
                    if hit is not None:
                        cache_note = (f"这份数据有本地缓存（{found['created']} 写的："
                                      f"{summary}）—— 已把参数切回缓存那一套并直接读它，"
                                      "本次没有重算。")
                    else:
                        cache_note = (f"这份数据有本地缓存（{summary}），参数已按它切回，"
                                      "但缓存读不回来（可能已损坏）→ 点「运行分析」会重算。")
            if hit is not None:
                self._mark_cache_hit_progress()
                self._apply_series(hit["coeffs"], rep=None, reason="载入",
                                   table=hit.get("table"),
                                   shared=hit.get("shared"), cache=hit)
        self._refresh_cache_hint()
        self._refresh_map()
        self.tabs.setCurrentIndex(0)
        self._update_actions()
        self._loading = False
        # 懒加载：首屏已经能用（第 0 层读好了），剩下的**后台**按块补。这里只负责
        # 报进度，补齐后把"需要全部历元"的页面（时间序列）刷新出来。
        ld = ds.lazy_loader()
        self._lazy_ld = ld
        _base = (f"已载入 {os.path.basename(ds.path)}：{ds.npoints} 个采样点，"
                 f"{ds.ntime} 个时次{self._time_span_hint(ds)}")
        if ld is not None and ld.n_filled < ld.ntime:
            ld.start_prefetch(from_t=self.time_slider.value())
            self._lazy_timer.start()
            self.status(f"{_base}；首层已可显示，其余 "
                        f"{ld.ntime - ld.n_filled} 个时次后台读取中…"
                        + (f"　{cache_note}" if cache_note else ""))
        else:
            self.status(f"{_base}；建议积分元规则 '{rule}'"
                        + (f"　{cache_note}" if cache_note else ""))

    # ------------------------------------------------- 懒加载：后台补数进度
    def _on_lazy_tick(self):
        """轮询后台补齐进度（只报进度，不缺数据 —— 地图/时次切换随时可用）。"""
        ds = self.dataset
        ld = self._lazy_ld
        if ds is None or ld is None or ds.lazy_loader() is not ld:
            self._lazy_timer.stop()
            return
        n, total = int(ld.n_filled), int(ld.ntime)
        if n < total:
            if self._lazy_ld is not None:
                ld.start_prefetch()            # 保险：线程异常退出时重启
            self.status(f"后台读取时次 {n}/{total}…"
                        f"（地图与滑块不受影响，可直接用）")
            return
        self._lazy_timer.stop()
        self.status(f"全部 {total} 个时次已读入内存：时间序列 / 趋势 / 批量分析"
                    f"现在都是内存操作。")
        self._refresh_series_page()

    def _ensure_all_values(self, why: str) -> bool:
        """要让"全部历元"的操作前先把懒加载补齐（带进度提示）。

        返回 False 表示没有数据。补齐可能要几秒（实测 256 时次约 3–7 s），所以
        这里给出状态栏文字并让事件循环转一圈，不做"静默卡住"。
        """
        ds = self.dataset
        if ds is None:
            return False
        ld = ds.lazy_loader()
        if ld is None or ld.n_filled >= ld.ntime:
            return True
        # ★ 先把进度条（忙碌态）摆出来，**再**开始补数：实测真实 mascon（256 历元 /
        # 1 GB）这一步约 3 s，放在网络盘/同步盘上更久。以前它发生在
        # ``setVisible(True)`` **之前**，于是"点了运行 → 界面发呆 → 进度条 0%"
        # 就是这么来的。
        self._progress_busy(f"{why}：正在读取全部 {ld.ntime} 个时次"
                            f"（{ld.n_filled}/{ld.ntime}）…")
        self.status(f"{why}：正在读取全部 {ld.ntime} 个时次"
                    f"（{ld.n_filled}/{ld.ntime}）…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ds.ensure_all_epochs(progress=lambda d, n: self._lazy_progress(d, n, why))
        finally:
            QApplication.restoreOverrideCursor()
        self._lazy_timer.stop()
        self.status(f"{why}：{ld.ntime} 个时次已全部读入。")
        return True

    def _lazy_progress(self, done: int, total: int, why: str):
        msg = f"{why}：正在读取全部时次 {done}/{total}…"
        self.status(msg)
        if getattr(self, "progress", None) is not None and self.progress.isVisible():
            self.progress_label.setText(msg)     # 忙碌条上的文字跟着走
        QApplication.processEvents()

    @staticmethod
    def _time_span_hint(ds) -> str:
        """``'（2002-04-18 … 2026-03-16）'``，或明确写出"没认出时间轴"。"""
        if not ds.has_time():
            return ""
        ax = ds.time_axis()
        if ax is not None and ax.has_dates:
            return f"（{str(ax.values[0])[:10]} … {str(ax.values[-1])[:10]}）"
        return "（时间轴未识别：只有历元序号，日期相关功能不可用）"

    def _refresh_summary(self, t: int = 0):
        """左侧「数据摘要」按**当前时次**重算。

        摘要里的「数值范围 / RMS」现在是逐时次的（用户要求：多时次文件整块求
        范围没有意义，而且整块统计在大文件上要几秒）。所以换时次就得刷新这两行，
        否则用户看到的数字跟自己正在看的那张图对不上。
        """
        ds = self.dataset
        if ds is None:
            return
        try:
            txt = ds.summary(int(t))
        except Exception as exc:                                 # noqa: BLE001
            txt = f"统计失败：{exc}"
        self.data_summary.setPlainText(txt)

    def _on_time_changed(self, _v: int):
        self._update_time_label()
        ds = self.dataset
        if self._epoch_source_coeffs() is not None:
            self.status("时次已切换：重建/差值/输出场按该历元的系数重算"
                        "（系数一次解完，这里只做切片）。")
            # 逐阶谱 / 系数统计 / 报告里的"那一解"也跟着时次走
            self._apply_epoch_views(self.time_slider.value())
        elif ds is not None and ds.has_time():
            self.status("时次已切换：还没有逐历元系数 —— 切到重建/差值/输出场时"
                        "会一次解完全部历元（或现在点「运行分析」）。")
        self._refresh_map()
        self._refresh_summary(self.time_slider.value())
        # 水平形变是**逐历元**的量：换了时次，之前那张图就不再对应 —— 但**不自动
        # 重算**（用户要求"点了计算再算"），只作废并提示。
        if getattr(self, "horiz_canvas", None) is not None \
                and self._horiz_data is not None:
            self._mark_horizontal_stale(
                f"时次已切到第 {self.time_slider.value() + 1} 个历元："
                "水平形变那张图是上一个历元的。")
        self._refresh_series_page()
        self._refresh_trend_page()

    def _step_time(self, delta: int):
        """◀/▶ buttons (and ←/→ keys): move one epoch."""
        v = int(self.time_slider.value()) + int(delta)
        v = max(self.time_slider.minimum(), min(self.time_slider.maximum(), v))
        self.time_slider.setValue(v)

    def _on_time_range_changed(self, _v: int = 0):
        """Keep ``lo <= hi`` and refresh whatever uses the window."""
        lo, hi = self.time_lo.value(), self.time_hi.value()
        if lo > hi:
            # 谁被拖动就把另一端让开，避免两个滑块互相打架
            if self.sender() is self.time_lo:
                self.time_hi.blockSignals(True)
                self.time_hi.setValue(lo)
                self.time_hi.blockSignals(False)
            else:
                self.time_lo.blockSignals(True)
                self.time_lo.setValue(hi)
                self.time_lo.blockSignals(False)
        self._update_time_label()
        self._refresh_series_page()

    def _reset_time_range(self):
        n = max(self.dataset.ntime - 1, 0) if self.dataset is not None else 0
        for s, v in ((self.time_lo, 0), (self.time_hi, n)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self._update_time_label()
        self._refresh_series_page()

    def time_window(self) -> tuple:
        """``(i0, i1)`` inclusive epoch window from the 范围 sliders."""
        if self.dataset is None:
            return 0, 0
        return (int(self.time_lo.value()), int(self.time_hi.value()))

    def _update_time_label(self):
        if self.dataset is None:
            self.time_label.setText("—")
            return
        ds = self.dataset
        i = min(self.time_slider.value(), max(ds.ntime - 1, 0))
        ax = ds.time_axis()
        parts = [f"第 {i + 1} / {ds.ntime} 个时次"]
        if ax is not None and ax.has_dates:
            iso = str(ax.values[i])[:19].replace("T", " ")
            parts.append(f"ISO {iso}")
            try:
                dy = float(np.asarray(ax.decimal_years)[i])
                parts.append(f"十进制年 {dy:.6f}")
            except Exception:                                    # noqa: BLE001
                pass
        else:
            parts.append("（无日期：只有序号，日期相关操作会被拒绝）")
        lo, hi = self.time_window()
        if (lo, hi) != (0, max(ds.ntime - 1, 0)):
            parts.append(f"范围 {lo + 1}–{hi + 1}")
        self.time_label.setText("　".join(parts))

    def _show_time_summary(self):
        ds = self.dataset
        ax = None if ds is None else ds.time_axis()
        if ax is None:
            QMessageBox.information(
                self, "时间轴", "当前数据没有可用的时间轴（只有历元序号）。\n"
                "要让这里显示跨度/缺测/重复，请载入带 time 坐标的 nc。")
            return
        QMessageBox.information(self, "时间轴", ax.summary())

    def make_demo(self):
        """Build a synthetic scattered dataset so the GUI can be tried instantly."""
        n, L = 2500, 12
        i = np.arange(n) + 0.5
        lat = 90.0 - np.rad2deg(np.arccos(1.0 - 2.0 * i / n))
        lon = np.mod(np.pi * (1 + 5 ** 0.5) * i, 2 * np.pi) * 180.0 / np.pi
        rng = np.random.default_rng(12345)
        from ..coeffs import SHCoeffs
        from ..synthesis import synthesis
        C = np.zeros((L + 1, L + 1))
        S = np.zeros((L + 1, L + 1))
        for deg in range(L + 1):
            for m in range(deg + 1):
                C[deg, m] = rng.standard_normal() / max(deg, 1) ** 2
                if m:
                    S[deg, m] = rng.standard_normal() / max(deg, 1) ** 2
        f = synthesis(lat, lon, SHCoeffs(C, S))
        f = f + rng.standard_normal(n) * 0.01 * np.std(f)
        self.dataset = Dataset.from_points(
            "<示例数据>", lat, lon, f,
            {"warnings": ["合成场：真实系数带限到 12 阶 + 1% 噪声"]})
        self.path_edit.setText("<示例数据（球面 Fibonacci 散点，真实带限 12 阶）>")
        self._real_truth = SHCoeffs(C, S)
        idx = self.cb_rule.findData("voronoi")
        if idx >= 0:
            self.cb_rule.setCurrentIndex(idx)
        self.sp_nmax.setValue(12)
        self.time_row.setVisible(False)
        self.data_summary.setPlainText(self.dataset.summary())
        self.status("已生成示例数据（2500 个球面散点，真实场带限 12 阶 + 1% 噪声）")
        self._refresh_map()
        self._update_actions()

    # ============================================================== analysis
    def _collect_params(self) -> dict:
        alpha_txt = self.ed_alpha.text().strip()
        return {
            "nmax": self.sp_nmax.value(),
            "rule": self.cb_rule.currentData(),
            "method": self.cb_method.currentData(),
            "niter": self.sp_niter.value(),
            "tau": (float(self.sp_tau.value()) if self.ck_tau.isChecked()
                    else None),
            "normalise": self.cb_norm.currentData(),
            "reg": self.cb_reg.currentData(),
            "alpha": float(alpha_txt) if alpha_txt else None,
            "gaussian_km": self.sp_gauss.value(),
            "field_unit": self.cb_field_unit.currentData() or "scalar",
            "target_unit": self.cb_target_unit.currentData() or None,
        }

    def _on_input_unit_changed(self, *_):
        """「输入是」= 正变换公式，换了它导出的系数就变，所以要重跑。"""
        self._on_field_unit_changed()
        if self.coeffs is not None:
            self.status("「输入是」已改变：正变换公式变了，导出的系数与两个场"
                        "都要重新运行分析才会更新。")
        self._refresh_map()

    def _on_target_changed(self, *_):
        self._on_field_unit_changed()
        if self.coeffs is not None:
            self.status("「输出为」已改变：导出的系数不变（永远是 C_nm），"
                        "但输出场要重新运行分析才会刷新。")
        self._refresh_map()

    def _on_field_unit_changed(self, *_):
        """把两条公式的说明与实时数值刷新一遍。

        这里的分工是最容易读错的地方：

        * **输入是 = 正变换公式。** ``C_nm = a_nm / f_u``。同一块网格声明成
          geoid 和声明成 EWH 是**两个不同的物理场**，除的因子不同，所以
          **改这一项确实会改变导出的系数**（第 2 阶差 ``A₂/R ≈ 13.2`` 倍，
          且逐阶不同）。
        * **输出为 = 反变换公式。** ``重建场 = C_nm × f_t``。它不改变导出的
          系数（那永远是 ``C_nm``），只决定重建出来是什么物理量；取「不换算」
          时 ``f_t = f_u``，正反互相抵消，重建精确回到原网格。
        """
        u = self.cb_field_unit.currentData() or "scalar"
        tgt = self.cb_target_unit.currentData() or ""
        hint = _UNIT_HINTS.get(u, "")
        hint += "\n（这一项决定导出系数：换一个声明，除的 f_u 就变了。）"
        self.lbl_unit_hint.setText(hint)

        if not tgt:
            self.lbl_target_hint.setText(
                "反变换公式取输入那一档的 f_u —— 正反互相抵消，重建精确回到"
                "原网格（「网格 → 系数 → 网格」就走这一档）。"
                "导出的系数仍然是 C_nm。")
            self.lbl_target_hint.setStyleSheet(_HINT_STYLE)
            self._update_unit_preview()
            return
        if tgt == u:
            msg = ("输出与输入同档 —— 反变换正好抵消正变换，重建回到原网格"
                   "（不会重复乘）。导出的系数仍是 C_nm。")
            style = _HINT_OK
        elif u in ("scalar", "unknown"):
            from .. import units as _units
            msg = _units.require_convertible(u, tgt) or "无法换算"
            msg = msg.split("为什么不能替你猜")[0].strip()
            style = _HINT_BAD
        else:
            try:
                from .. import units as _units
                f = _units.degree_factors(tgt, 6)
                if tgt == "radial_displacement":
                    msg = (f"反变换：C_nm × 逐阶因子（n=0..6 约 {f[0]:.3g} … {f[6]:.3g} m，"
                           "h′ 取自随包的 PREM 表；负值表示正载荷下沉）。"
                           "导出的系数仍是 C_nm。")
                elif tgt == "geoid":
                    msg = (f"反变换：C_nm × 常数 R = {f[0]:.6g} m（不是逐阶因子）。"
                           "导出的系数仍是 C_nm。")
                else:
                    msg = (f"反变换：C_nm × 逐阶因子（n=0..6 约 {f[0]:.3g} … {f[6]:.3g}"
                           f"，跨 {f[6] / f[0]:.1f} 倍，不能用单一常数代替）。"
                           "导出的系数仍是 C_nm。")
                style = _HINT_OK
            except Exception as exc:                   # noqa: BLE001
                msg = f"无法换算：{exc}"
                style = _HINT_BAD
        self.lbl_target_hint.setText(msg)
        self.lbl_target_hint.setStyleSheet(style)
        self._update_unit_preview()

    def _update_unit_preview(self):
        """把两条公式具体到某一个系数上，直接显示成数字。

        这里把同一块网格在当前声明下的**正变换**摆出来（导出 C_nm = a_nm / f_u），
        再把**两个反变换场**都摆出来：

        * **重建场** = `C_nm × f_u` —— 与输入同物理量，可直接与原数据比对；
        * **输出场** = `C_nm × f_t` —— 按「输出为」换算出来的另一个物理量。

        改「输入是」时 `f_u` 变、导出系数就变；改「输出为」时只有输出场变。
        """
        from .. import units as _units
        c = self.coeffs
        u = self.cb_field_unit.currentData() or "scalar"
        tgt = self.cb_target_unit.currentData() or ""

        if c is None:
            self.lbl_unit_preview.setText(
                "运行分析后这里会显示正变换（导出）与反变换（重建）的实际数字。")
            return

        # 挑一个非零的、有代表性的系数（优先低阶、m≠0）
        L = c.nmax
        pick = None
        for n in range(1, min(L, 6) + 1):
            for m in range(1, n + 1):
                if abs(c.C[n, m]) > 1e-12:
                    pick = (n, m)
                    break
            if pick:
                break
        if pick is None:
            self.lbl_unit_preview.setText("")
            return
        n, m = pick

        lines = []
        try:
            f_u = _units.forward_factors(u, L)
            C_nm = float(c.C[n, m])
            a = C_nm * f_u[n]                 # 该网格自身的系数
            if u in ("scalar", "unknown"):
                lines.append(f"正变换 f_u = 1：导出 C[{n},{m}] = 网格系数 {a:.6g}"
                             "（无物理公式，就是该场自身的系数）")
            else:
                lines.append(
                    f"正变换（÷ f_u，n={n} 档 f_u = {f_u[n]:.6g}）："
                    f"网格系数 {a:.6g} → 导出 C[{n},{m}] = {C_nm:.6g}")
        except Exception as exc:                       # noqa: BLE001
            lines.append(f"正变换不可用：{str(exc).splitlines()[0][:48]}")

        try:
            f_u = _units.forward_factors(u, L)
            f_t = _units.forward_factors(tgt or u, L)
            val = float(c.C[n, m])
            lines.append(
                f"重建场（× f_u = {f_u[n]:.6g}）：该点 = {val * f_u[n]:.6g}"
                f"（= 原网格值，与输入同物理量，可直接比对）")
            if not tgt or tgt == u:
                lines.append(
                    "输出场（× f_t = 输入的 f_u）：与重建场相同"
                    "（「输出为」与「输入是」同档，正反抵消）")
            else:
                ratio = f_t[n] / f_u[n] if f_u[n] else float("nan")
                lines.append(
                    f"输出场（× f_t = {f_t[n]:.6g}，相对输入 × {ratio:.6g}）："
                    f"该点 = {val * f_t[n]:.6g}")
        except Exception as exc:                       # noqa: BLE001
            lines.append(f"反变换不可用：{str(exc).splitlines()[0][:48]}")

        self.lbl_unit_preview.setText("\n".join(lines))

    def run_analysis(self):
        """「运行分析」。

        多时次数据：**一次解完全部历元**（与前端的"点一次就把数据处理完"一致），
        结果落盘缓存，之后地图/播放/GIF/逐时次导出都只是切片 + 综合，不再重算。
        单时次数据：还是原来的单解（没有"全部历元"可言）。
        """
        ds = self.dataset
        if ds is None:
            QMessageBox.information(self, "提示", "请先载入数据（或生成示例数据）。")
            return
        if ds.has_time():
            self._start_series("运行分析", ask_cache=True)
            return
        if self.analysis_worker is not None and self.analysis_worker.isRunning():
            return
        if self.series_worker is not None and self.series_worker.isRunning():
            self.status("已经有一次逐历元分析在跑，等它结束。")
            return
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.progress_label.setText("")
        self.status("分析中 …")
        self._t0 = time.time()
        # 记下**用户声明**的「输入是」：正变换一做完，系数与报告里的 field_unit 都会
        # 变成 geopotential，之后再想判断"这串数原本是 cm 还是 m"就没依据了。
        self._declared_field_unit = self.cb_field_unit.currentData() or "scalar"
        self._epoch_cache.clear()          # 参数可能变了 → 逐历元重建缓存作废
        self.analysis_worker = AnalysisWorker(
            self.dataset, self._collect_params(),
            time_index=self.time_slider.value(), parent=self)
        w = self.analysis_worker
        w.progressed.connect(self.on_progress)
        w.succeeded.connect(self.on_analysis_done)
        w.failed.connect(self.on_analysis_failed)
        w.cancelled.connect(self.on_analysis_cancelled)
        w.finished.connect(self._on_single_finished)
        w.start()
        self._update_actions()

    def _on_single_finished(self):
        """单历元线程收尾（``finished`` 总会发，成功/失败/取消/静默结束都算）。

        进度条的隐藏以前只挂在"序列线程跑完"上，单历元这条路成功后会把一根
        100% 的条永远留在界面上；这里统一收工。
        """
        self._progress_idle()
        self._update_actions()

    def stop_analysis(self):
        """停「运行分析」：单时次走 analysis_worker，多时次走 series_worker，
        水平形变走 horiz_worker。"""
        if self.analysis_worker is not None and self.analysis_worker.isRunning():
            self.analysis_worker.cancel()
            self.status("已请求停止 …（当前 NumPy 运算结束后生效）")
        elif self.series_worker is not None and self.series_worker.isRunning():
            self.series_worker.cancel()
            self.status("已请求停止 …（当前分块结束后生效）")
        elif self.horiz_worker is not None and self.horiz_worker.isRunning():
            self.horiz_worker.cancel()
            self.horiz_note.setText("已请求停止 …（当前阶/分块结束后生效）")
            self.status("已请求停止水平形变 …")

    # ------------------------------------------------------------ 缓存管理
    def _cache_report_text(self, rep: dict) -> str:
        import time as _time

        def _mb(n):
            return f"{n / 1e6:.2f} MB" if n >= 1e6 else f"{n / 1024:.0f} kB"

        lines = ["<b>当前数据的缓存</b>"]
        if rep["data_path"] is None:
            lines.append("　（当前数据没有磁盘文件，不会写缓存）")
        elif not rep["data_exists"]:
            lines.append(f"　还没有：{rep['data_path']}")
        else:
            when = _time.strftime("%Y-%m-%d %H:%M",
                                  _time.localtime(rep["data_mtime"]))
            lines.append(f"　{_mb(rep['data_size'])}　{when}")
            lines.append(f"　{rep['data_path']}")
            if rep.get("data_match") is True:
                lines.append("　<b>当前文件与参数：能命中</b>"
                             "（「运行分析」直接读它，不重算）")
            elif rep.get("data_match") is False:
                lines.append("　<b>当前文件与参数：对不上，不会命中</b>"
                             "（键里有源文件指纹与全部参数；点「运行分析」会重算并覆盖）")
        lines.append("")
        lines.append("<b>用户缓存目录</b>（数据目录不可写时的退路）")
        lines.append(f"　{rep['user_files']} 份，共 {_mb(rep['user_total'])}"
                     f"（上限 {_mb(rep['user_limit'])}，超出按最久未用自动清理）")
        lines.append(f"　{rep['user_dir']}")
        lines.append("")
        lines.append(f"单份缓存上限：{_mb(rep['entry_limit'])}；"
                     "缓存只是省时间，删掉不影响已经算出来的结果，"
                     "下次「运行分析」会重算并重写。")
        return "<br>".join(lines)

    def show_cache_manager(self):
        """「文件 → 本地缓存…」：看清占用 + 就地清理（回答"缓存会不会无限涨"）。"""
        rep = coeff_cache.report(
            self.dataset,
            self._collect_params() if self.dataset is not None else None)
        box = QMessageBox(self)
        box.setWindowTitle("本地缓存")
        box.setIcon(QMessageBox.Information)
        box.setTextFormat(Qt.RichText)
        box.setText(self._cache_report_text(rep))
        b_ds = box.addButton("删除当前数据的缓存", QMessageBox.DestructiveRole)
        b_user = box.addButton("清空用户缓存目录", QMessageBox.DestructiveRole)
        b_close = box.addButton("关闭", QMessageBox.AcceptRole)
        box.setDefaultButton(b_close)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_ds:
            params = self._collect_params() if self.dataset is not None else None
            removed = coeff_cache.clear(self.dataset, params)
            self.status(f"已删除 {len(removed)} 份缓存"
                        + ("；内存里的结果不受影响（本次会话照常可用）。"
                           if self.series_coeffs is not None or
                           self.coeffs is not None else "。"))
            self._refresh_cache_hint()
        elif clicked is b_user:
            n, freed = coeff_cache.clear_user_cache()
            self.status(f"用户缓存目录已清空：删除 {n} 份，释放 {freed / 1e6:.2f} MB"
                        + ("；内存里的结果不受影响。" if self.coeffs is not None
                           else "。"))
            self._refresh_cache_hint()

    # ------------------------------------------------- 缓存里的参数 → 界面控件
    def _params_human(self, p: dict) -> str:
        """把一套参数说成人话（对话框/提示里用）。"""
        bits = []
        if p.get("field_unit") is not None:
            bits.append(f"输入是={p['field_unit']}")
        if p.get("target_unit"):
            bits.append(f"输出为={p['target_unit']}")
        if p.get("nmax") is not None:
            bits.append(f"nmax={p['nmax']}")
        if p.get("rule"):
            bits.append(f"积分元={p['rule']}")
        if p.get("method"):
            bits.append(f"估计方法={p['method']}")
        if p.get("niter"):
            bits.append(f"迭代={p['niter']}")
        if p.get("tau") is not None:
            bits.append(f"tau={p['tau']:g}")
        if p.get("normalise") and p["normalise"] != "auto":
            bits.append(f"归一化={p['normalise']}")
        if p.get("reg") and p["reg"] != "none":
            bits.append(f"正则={p['reg']}")
        if p.get("alpha") is not None:
            bits.append(f"alpha={p['alpha']:g}")
        if p.get("gaussian_km"):
            bits.append(f"高斯平滑={p['gaussian_km']:g} km")
        return "、".join(bits) or "（默认）"

    def _ask_series_cache(self, hit: dict, params: dict) -> str:
        """有缓存 → 问一句"读缓存还是重算"。返回 ``'cache'`` / ``'recompute'`` / ``'cancel'``。

        用户要求：「有缓存时，点击运行分析弹出提示，用户决定是否重跑」。
        默认按钮是**读缓存**（最快的那个），"重算"是 DestructiveRole（会覆盖缓存）。
        只有**用户显式点**「运行分析 / 批量分析」时才会走到这里；导出/切视图那条
        自动补算的路不弹框。
        """
        cached = dict(hit.get("params") or {})
        same = all(cached.get(k) == params.get(k)
                   for k in coeff_cache._PARAM_KEYS)
        try:
            size = coeff_cache._fmt_size(os.path.getsize(hit["path"]))
        except OSError:
            size = "?"
        where = "数据文件旁边" if hit.get("source") == "data-dir" else "用户缓存目录"
        in_mem = (self._series_cache is not None
                  and str((self._series_cache or {}).get("path")) == hit["path"])
        lines = [
            f"这份数据已经有一份本地缓存（{where}，{size}，写于 "
            f"{hit.get('created') or '?'}）：",
            f"　参数：{self._params_human(cached)}",
        ]
        if same:
            lines.append("　与当前界面参数<b>一致</b>。")
        else:
            lines.append("　与当前界面参数<b>不同</b>："
                         f"{self._params_human(params)}")
        lines.append("　当前内存里的结果：" + (
            "就是这份缓存读回来的（本次会话还没有重算过）。" if in_mem
            else "还没有 / 不是这一份缓存。"))
        lines.append("")
        lines.append("直接读缓存最快；重新计算会按当前参数解一遍整条序列，"
                     "并把缓存<b>覆盖</b>成新的。")

        box = QMessageBox(self)
        box.setWindowTitle("本地已有缓存")
        box.setIcon(QMessageBox.Question)
        box.setTextFormat(Qt.RichText)
        box.setText("<br>".join(lines))
        b_cache = box.addButton("直接读缓存（不重算）", QMessageBox.AcceptRole)
        b_re = box.addButton("重新计算（覆盖缓存）", QMessageBox.DestructiveRole)
        b_cancel = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(b_cache)
        box.setEscapeButton(b_cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_cache:
            return "cache"
        if clicked is b_re:
            return "recompute"
        return "cancel"

    def _adopt_cached_params(self, found: dict) -> str:
        """把界面参数切回**缓存那一套**，返回一句人话摘要（给状态栏/提示用）。

        用户的要求原话：「识别到缓存时，「输入是」这个应该自动切换吧」。要点：

        * ``field_unit``（「输入是」）**必须**跟着切：它是缓存键的一部分，因为
          同一块网格声明成 geoid / EWH / geopotential 除的 ``f_u`` 不同，**系数真的
          不一样** —— 不切就会每次点运行都重算（默认值永远对不上缓存）。
        * 其余影响系数的参数（nmax/规则/方法/迭代/tau/归一化/正则/alpha/高斯半径）
          一起切，否则照样对不上。切完状态栏会写明"参数已切到缓存那一套"，
          用户想用别的参数直接改就行（改了就会按新参数重算并**覆盖**缓存）。
        * 控件里没有那个取值（版本差异/缓存是别的构建写的）→ **不静默**：跳过并
          在摘要里标出来，界面表现是"没完全切回去"，用户能看出原因。
        """
        p = dict(found.get("params") or {})
        applied, skipped = [], []

        def _label(key, val):
            if key == "field_unit":
                return f"输入是={val}"
            if key == "target_unit":
                return f"输出为={val or '不换算'}"
            if key == "gaussian_km":
                return f"高斯半径={val:g} km"
            if key == "nmax":
                return f"nmax={val}"
            return f"{key}={val}"

        def _spin(box, key, setter):
            val = p.get(key)
            if val is None:
                return
            try:
                setter(val)
                applied.append(_label(key, val))
            except Exception:                                    # noqa: BLE001
                skipped.append(f"{key}={val}")

        def _combo(box, key):
            val = p.get(key)
            if val is None and key in ("target_unit", "reg", "alpha"):
                pass                        # None 是合法取值，按下面各自处理
            idx = box.findData(val)
            if idx < 0 and key == "target_unit":
                idx = box.findData("")      # 「不换算」
            if idx < 0:
                if val is not None:
                    skipped.append(f"{key}={val}")
                return
            box.setCurrentIndex(idx)
            applied.append(_label(key, val))

        # 顺序要紧：先定"估计方法"，迭代次数/ tau 的可用性由它决定；再填数值。
        _combo(self.cb_method, "method")
        _combo(self.cb_rule, "rule")
        _combo(self.cb_norm, "normalise")
        _combo(self.cb_reg, "reg")
        _combo(self.cb_field_unit, "field_unit")
        _combo(self.cb_target_unit, "target_unit")
        _spin(self.sp_nmax, "nmax", lambda v: self.sp_nmax.setValue(int(v)))
        _spin(self.sp_niter, "niter", lambda v: self.sp_niter.setValue(int(v)))
        _spin(self.sp_gauss, "gaussian_km",
              lambda v: self.sp_gauss.setValue(float(v)))
        # tau：控件是"勾选 + 数值"两件套，None = 没勾
        tau = p.get("tau")
        if "tau" in p:
            self.ck_tau.setChecked(tau is not None)
            if tau is not None:
                self.sp_tau.setValue(float(tau))
            applied.append(_label("tau", "不启用" if tau is None else tau))
        alpha = p.get("alpha")
        if "alpha" in p:
            self.ed_alpha.setText("" if alpha is None else f"{float(alpha):g}")
            applied.append(f"alpha={alpha if alpha is not None else '不启用'}")
        reg_power = p.get("reg_power")
        if reg_power is not None:
            applied.append(f"正则幂={reg_power}")     # 界面没有这个控件，只说明
        QApplication.processEvents()
        self._on_field_unit_changed()                 # 刷新两条公式的说明与预览
        txt = "、".join(applied) if applied else "（无可切换项）"
        if skipped:
            txt += "；⚠ 这些取值当前界面没有、未切：" + "、".join(skipped)
        return txt

    def _refresh_cache_hint(self):
        """把参数面板里那句"缓存什么状态、占多少"刷新一下（写/删/换数据后都叫它）。

        ★ 判据是**能不能命中**，不是"文件在不在"：旁边躺着一份键对不上的缓存时
        （实测用户那份 mascon 缓存的键里写的是旧路径 ``D:\\myds\\…``），说"直接读它、
        不重算"就是撒谎 —— 这里必须说清楚"对不上，会重算并覆盖"。
        """
        try:
            rep = coeff_cache.report(self.dataset)
            hit = (coeff_cache.probe(self.dataset, self._collect_params())
                   if self.dataset is not None else None)
        except Exception:                                        # noqa: BLE001
            return
        if not hasattr(self, "cache_hint"):
            return

        def _mb(n):
            return (f"{n / 1e6:.2f} MB" if n >= 1e6 else f"{n / 1024:.0f} kB")

        if hit is not None:
            head = (f"本地这份缓存「能用」（{_mb(rep['data_size'])}，写于 "
                    f"{hit.get('created') or '?'}）——「运行分析」直接读它，不重算。")
        elif rep["data_exists"]:
            head = (f"旁边有一份缓存（{_mb(rep['data_size'])}），但与当前文件/参数"
                    "「对不上 → 不会命中」；「运行分析」会重算并覆盖它。")
        else:
            head = "本地暂无该数据的缓存。"
        if rep["user_files"]:
            head += (f"\n用户缓存目录另有 {rep['user_files']} 份 / "
                     f"{_mb(rep['user_total'])}（上限 {_mb(rep['user_limit'])}）。")
        # 载入时"按缓存把参数切回来了"这件事写在常驻提示里：状态栏那句很快会被
        # 懒加载后台进度顶掉，而这件事用户是要能随时看见的。
        _adopt = getattr(self, "_cache_adopt_note", "")
        if _adopt and hit is not None:
            head += f"\n已按这份缓存切回参数：{_adopt}。"
        head += "　文件 → 「本地缓存…」可查看/清理。"
        self.cache_hint.setText(head)

    def on_progress(self, msg: str, frac: float):
        if self.progress.maximum() == 0:
            # 之前是"忙碌条"（不确定进度）→ 现在有真实百分比了，切回来
            self.progress.setRange(0, 1000)
        self.progress.setValue(int(frac * 1000))
        self.progress_label.setText(msg)

    def _progress_busy(self, label: str):
        """**不确定进度**：忙碌条 + 一句话。

        有些阶段本来就没有可信的百分比，硬按 0% 显示就是"卡住"：实测真实 mascon
        （1 M 点 × 256 历元）点上「运行分析」后，前 3 s 在补齐全历元、接着约 2 s 在把
        1 GB 数据交给子进程 —— 这 5 s 里进度条原来是"可见、0%、标签空白"，用户看到
        的就是卡住。忙碌条会自己动，含义是"在干活，还不知道多久"。
        """
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.progress_label.setText(label)

    def _progress_idle(self):
        """收工：把进度条藏起来（空闲时界面上不该留一根条）。

        所有"这次不跑线程 / 已经跑完"的出口都必须走这里：隐藏动作以前只写在
        :meth:`_on_series_finished` 里，于是**缓存命中**（``_start_series`` 提前
        return）、**单历元分析完成**、失败与取消这几条路都把一根 0% 或 100% 的
        条留在界面上 —— 用户报的"运行后进度条卡在 0 不动"就是这么来的。

        （``getattr`` 是给"控件还没建好就被叫到"的构造期路径留的退路。）
        """
        p = getattr(self, "progress", None)
        if p is not None:
            if p.maximum() == 0:            # 忙碌条要还原成百分比条，否则下次没法显示进度
                p.setRange(0, 1000)
            p.setVisible(False)

    def _mark_cache_hit_progress(self):
        """缓存命中：不跑线程，进度条得**手动收工**。

        把条推到 100% 再藏起来，并在标签上写明"这次没有重算"。缺这一步，用户点完
        「运行分析」看到的就是一根停在 0% 不动的条（他的原话）。
        ``on_loaded``（载入即命中）与 ``_start_series``（点运行才命中）共用。
        """
        if self.progress.maximum() == 0:      # 万一还停在忙碌态，先还原量程
            self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.progress_label.setText("逐历元系数直接读本地缓存（本次未重算）")
        self._progress_idle()

    def on_analysis_cancelled(self):
        self.status("分析已取消")
        self.progress.setValue(0)
        self.progress_label.setText("已取消")
        self._progress_idle()
        self._update_actions()

    def on_analysis_failed(self, tb: str):
        which = self._series_reason if (self.series_worker is self.sender()
                                        and self._series_reason) else "分析"
        self.status(f"{which}失败")
        self.progress.setValue(0)
        self.progress_label.setText("失败")
        self._progress_idle()
        # 关键：清掉上一次的结果。否则「导出系数」会静默写出**上一次**的旧结果，
        # 用户会以为那是刚算出来的（实测确实会这样）。逐历元那一套同理：它是用
        # 旧参数算的，留着会让地图画"上次那套参数"的场。
        self.coeffs = self.report = self.recon = self.outfield = None
        self._single_t = None
        self.series_coeffs = None
        self.series_report = None
        self._series_cache = None
        self._epoch_cache.clear()
        self._coeff_version += 1
        self._horiz_data = None
        self._horiz_sig = None
        self.horiz_canvas.show_placeholder("还没有结果 —— 先运行一次分析。")
        self.horiz_note.setText("尚未计算。")
        short = _short_error(tb)
        QMessageBox.critical(
            self, "分析失败",
            f"{short}\n\n（上一次的结果已清除，导出按钮已禁用；"
            "完整回溯见控制台）")
        print(tb)
        self.coeff_table.setRowCount(0)
        self.coeff_note.setText("分析失败，无结果。")
        self.report_view.setPlainText("分析失败：\n\n" + tb)
        self._update_unit_preview()
        self._update_actions()

    def on_analysis_done(self, coeffs, report, recon, outfield=None):
        # 同一个理由：中途换了数据集的话，这次分析的结果不属于当前数据。
        sender = self.sender()
        if getattr(sender, "dataset", None) is not None \
                and sender.dataset is not self.dataset:
            self.status("分析结果属于上一个数据集，已丢弃。")
            return
        self.coeffs, self.report = coeffs, report
        self.recon, self.outfield = recon, outfield
        # ★ 记住这一次解的是**哪个历元**：self.recon 只对它成立。以前换时次后
        # 仍然拿它当作"当前历元的重建场"，于是播放时每帧都是同一个旧场。
        self._single_t = int(getattr(self.analysis_worker, "time_index", 0))
        self._epoch_cache.clear()          # 新结果 → 缓存的逐历元重建作废
        # 新的一套系数：水平形变那张图作废（不自动重算，等用户点按钮）
        self._coeff_version += 1
        self._horiz_data = None
        if getattr(self, "horiz_canvas", None) is not None:
            self._mark_horizontal_stale("分析完成：水平形变与这套系数不再对应。")
        dt = time.time() - self._t0
        self.progress.setValue(1000)
        g_km = float(report.meta.get("gaussian_km") or 0.0)
        if g_km > 0:
            w_n = float(report.meta.get("gaussian_Wnmax", 1.0))
            self.progress_label.setText(
                f"完成（高斯平滑 {g_km:g} km，W(nmax)={w_n:.4g}）")
        else:
            self.progress_label.setText("完成")
        self.status(f"分析完成（{dt:.2f} s，方法 {report.method}，"
                    f"权重规则 {report.weight_rule}"
                    + (f"，高斯平滑 {g_km:g} km（系数已按 W 逐阶相乘）"
                       if g_km > 0 else "") + "）")
        self._progress_idle()
        self.status_time.setText(f"用时 {dt:.2f} s")
        self.report_view.setHtml(_report_to_html(report, self.dataset))
        self._fill_spectrum()
        self._fill_coeff_table()
        self._update_unit_preview()
        if self.chk_auto_map.isChecked() and self.recon is not None:
            self.map_field.setCurrentIndex(1)
        self._refresh_map()
        self._update_actions()

    # =============================================================== display
    # ------------------------------------------------------- 水平形变（按需算）
    def _horiz_key(self):
        """当前系数/时次的指纹：画出来的图只对它有效。"""
        ds = self.dataset
        if ds is None:
            return None
        t = int(self.time_slider.value()) if ds.ntime > 1 else 0
        return (self._coeff_version, id(ds), t)

    def _mark_horizontal_stale(self, why: str) -> None:
        """系数或时次变了：**不重算**，只把已有图作废并说清楚。

        用户的要求是「水平形变点了计算再算」：运行分析/批量分析/换时次都不该顺手
        把这张图（网格上是一次全球梯度综合，256 个历元时很贵）一起算掉。
        """
        self._horiz_sig = None
        if self.horiz_canvas._drawn is not None:
            self.horiz_canvas.show_placeholder(
                f"{why}\n点「计算水平形变」按当前系数重算（本页不会自动重算）")
        else:
            self.horiz_canvas.show_placeholder(
                f"{why}\n点「计算水平形变」开始计算（本页不会在运行分析时自动算）")
        self.horiz_note.setText(f"{why}　点「计算水平形变」重算。")
        self._horiz_data = None

    def _refresh_horizontal(self, force: bool = False):
        """Compute and draw horizontal deformation for the current coefficients.

        **只在用户点「计算水平形变」时算**（``force=True``）——运行分析、批量分析、
        切时次都只是把旧图作废，见 :meth:`_mark_horizontal_stale`。

        ★ 计算**不在这里做**：位系数的球面梯度在散点上实测 79 s/历元（百万点、
        nmax=60；规则网格走经度 FFT 是 0.08 s），同步做会把整个窗口冻住。所以这里只
        负责备料 + 起 :class:`HorizontalWorker`（或子进程，规则网格且勾了独立进程时），
        结果由 :meth:`_on_horiz_ready` 画出来；进度与取消都走既有那套。
        """
        if self.dataset is None or (self.coeffs is None
                                    and self._epoch_source_coeffs() is None):
            self.horiz_canvas.show_placeholder("还没有结果 —— 先运行一次分析。")
            self.horiz_note.setText("尚未计算。")
            self._horiz_sig = None
            return
        if self.horiz_worker is not None and self.horiz_worker.isRunning():
            self.status("水平形变正在算，等它结束（或点「停止」）。")
            return
        ds = self.dataset
        t = int(self.time_slider.value()) if ds.ntime > 1 else 0
        if self._coeffs_for_epoch(t) is None:
            # 这个历元还没有系数：能解就解（一次解完整条序列），然后**如实说**
            if ds.has_time():
                running = (self.series_worker is not None
                           and self.series_worker.isRunning())
                if not running:
                    self._ensure_series_async("水平形变")
                    running = (self.series_worker is not None
                               and self.series_worker.isRunning())
                self.horiz_canvas.show_placeholder(
                    (f"正在一次解完 {ds.ntime} 个历元…" if running else
                     "点「运行分析」会一次解完全部历元"))
                self.horiz_note.setText(
                    ("逐历元系数还没有：正在一次解完全部历元…" if running else
                     "逐历元系数还没有：点「运行分析」即可一次解完。"))
            else:
                self.horiz_canvas.show_placeholder(
                    "还没有结果 —— 先点「运行分析」。")
                self.horiz_note.setText("尚未计算。")
            return
        try:
            from ..gradient import canonical_scaled
            # 有逐历元系数就用当前历元那一片；单时次数据集用那一解
            co = self._coeffs_for_epoch(t)
            if co is None:
                raise ValueError(
                    f"第 {t + 1} 个历元还没有系数：点「运行分析」会一次解完全部历元。")
            if np.asarray(co.C).ndim == 3:         # one epoch at a time
                n = int(np.asarray(co.C).shape[2])
                co = co.time_slice(min(t, n - 1)) if hasattr(co, "time_slice") else co
            # ★ 单位检查/换算 + 逐阶因子在**界面进程**里做一次（报错语句与库一致），
            # 再把纯数值的活交给 worker：线程路径与子进程路径拿到逐位相同的输入。
            C, S, fac = canonical_scaled(co)
        except Exception as exc:                             # noqa: BLE001
            self.horiz_canvas.show_placeholder(
                f"水平形变计算失败：\n{_short_error(str(exc))}")
            self.horiz_note.setText(f"失败：{type(exc).__name__}")
            return

        self._horiz_pending_t = int(t)
        self.btn_horiz.setEnabled(False)
        lat_src = ds.lat_vec if ds.kind == "grid" else ds.lat
        lon_src = ds.lon_vec if ds.kind == "grid" else ds.lon
        use_sub = bool(self.chk_subprocess.isChecked())
        sub_note = ""
        if use_sub:
            from .subproc import spawn_available
            ok, why = spawn_available()
            if not ok:
                use_sub = False
                sub_note = f"独立进程不可用（{why}）→ 已回退到界面内线程；结果一样。"
        if use_sub:
            kind = "horizontal_grid" if ds.kind == "grid" else "horizontal_points"
            self.horiz_worker = SubprocessRunner(
                kind,
                {"lat": lat_src, "lon": lon_src, "C": C, "S": S,
                 "degree_factors": fac, "method": "auto", "chunk": 100_000},
                parent=self)
        else:
            self.horiz_worker = HorizontalWorker(
                ds.kind, lat_src, lon_src, C, S, fac, method="auto", parent=self)
        w = self.horiz_worker
        w.progressed.connect(self.on_progress)
        # ⚠️ SubprocessRunner.succeeded 是**两个**参数（object, object），
        # HorizontalWorker.succeeded 是**一个** —— 用 1 参槽接 2 参信号会被 Qt
        # 静默丢掉（项目里踩过这个坑），所以分成两个槽。
        if isinstance(w, SubprocessRunner):
            w.succeeded.connect(self._on_horiz_ready_sub)
        else:
            w.succeeded.connect(self._on_horiz_ready)
        w.failed.connect(self._on_horiz_failed)
        w.cancelled.connect(self._on_horiz_cancelled)
        w.finished.connect(self._on_horiz_finished)
        self._progress_busy("水平形变：正在算球面梯度…")
        self.horiz_note.setText(
            "正在计算水平形变（球面梯度）…"
            + ("　散点上这一步比网格慢很多，可以点「停止」。" if ds.kind != "grid"
               else "") + ("　" + sub_note if sub_note else ""))
        self.btn_stop.setEnabled(True)
        w.start()
        self._update_actions()

    # --------------------------------------------------- 水平形变：结果与收尾
    def _on_horiz_ready_sub(self, out, _why=None):
        """子进程那条路的适配器（``SubprocessRunner.succeeded`` 是两个参数）。"""
        self._on_horiz_ready(out)

    def _on_horiz_ready(self, out: dict):
        """worker 算完了：画图 + 写清路径/量级自检 + 记住"这张图对哪套系数有效"。"""
        ds = self.dataset
        if ds is None:
            return
        t = int(getattr(self, "_horiz_pending_t", 0))
        try:
            north = np.asarray(out["north"])
            east = np.asarray(out["east"])
            if north.ndim == 3:
                north, east = north[:, :, 0], east[:, :, 0]
            north = np.asarray(north).reshape(-1) if ds.kind != "grid" \
                else np.asarray(north)
            east = np.asarray(east).reshape(-1) if ds.kind != "grid" \
                else np.asarray(east)
        except Exception as exc:                                 # noqa: BLE001
            self._on_horiz_failed(f"结果形状不对：{exc}")
            return
        report = dict(out.get("report") or {})
        path = report.get("path") or "?"
        rms = float(np.sqrt(np.mean(north ** 2 + east ** 2)))
        note = f"RMS = {rms * 1e3:.3f} mm　路径：{path}　{report.get('reason', '')}"
        if rms > 1.0:
            # 量级自检：真实 GRACE 位系数（|C|~1e-10）给 mm 级水平形变。
            # 差了 3 个数量级以上，几乎一定是"这串数不是位系数"或"是示例数据"。
            note += "　⚠ 量级异常（>1 m）：真实 GRACE 位系数应在 mm 级"
        note += self._unit_scale_note()
        self.horiz_note.setText(note)
        g_km = float(self.report.meta.get("gaussian_km") or 0.0) if self.report else 0.0
        parts = []
        if ds.ntime > 1:
            parts.append(f"第 {t + 1} 个时次")
        if g_km > 0:
            parts.append(f"高斯 {g_km:g} km（已作用于系数）")
        suffix = ("，" + "，".join(parts)) if parts else ""
        title = f"水平形变 u_h{suffix}"
        lat_d = np.asarray(ds.lat_vec) if ds.kind == "grid" else np.asarray(ds.lat)
        lon_d = np.asarray(ds.lon_vec) if ds.kind == "grid" else np.asarray(ds.lon)
        self.horiz_canvas._maps(north, east, lat_d, lon_d,
                                is_grid=(ds.kind == "grid"), title=title)
        # 记住"这张图是对哪一套系数、哪个历元算的"，以及原始数据（海岸线开关
        # 只重画，不重算 —— 网格上一次全球梯度综合不便宜）
        self._horiz_sig = self._horiz_key()
        self._horiz_data = dict(north=north, east=east, lat=lat_d, lon=lon_d,
                                is_grid=(ds.kind == "grid"), title=title)
        self.progress_label.setText(f"水平形变已算出（{path}）")
        self.status(f"水平形变已计算（{path}）")

    def _on_horiz_failed(self, tb: str):
        self.horiz_canvas.show_placeholder(
            f"水平形变计算失败：\n{_short_error(tb)}")
        self.horiz_note.setText(f"失败：{_short_error(tb).splitlines()[0][:60]}")
        self.status("水平形变计算失败（窗口不受影响，可直接重试）。")
        print(tb)

    def _on_horiz_cancelled(self):
        self.horiz_note.setText("水平形变：已取消（这张图没有更新）。")
        self.status("水平形变已取消。")

    def _on_horiz_finished(self):
        self.btn_horiz.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._progress_idle()
        self._update_actions()

    def _toggle_coast(self, on: bool):
        self.map_canvas.show_coast = bool(on)
        self._refresh_map()

    def _on_focus_toggled(self, on: bool):
        """The 聚焦 button: zoom the map to the data footprint, or go global."""
        if getattr(self, "_focus_guard", False):
            return
        self.map_canvas.set_focus(bool(on))
        self._sync_focus_button()

    def _sync_focus_button(self):
        """Match the button to the canvas state (it may have auto-focused)."""
        if not hasattr(self, "btn_focus"):
            return
        can = self.map_canvas.can_focus()
        focused = self.map_canvas.is_focused()
        self._focus_guard = True
        try:
            self.btn_focus.setEnabled(can)
            self.btn_focus.setChecked(focused)
        finally:
            self._focus_guard = False
        self.btn_focus.setText("取消聚焦" if focused else "聚焦到数据")
        if not can:
            self.btn_focus.setToolTip(
                "当前数据覆盖全球（或范围接近全球），没有可聚焦的局部范围。\n"
                "载入散点或局部网格时本按钮会自动生效。")
        else:
            self.btn_focus.setToolTip(
                f"当前视图已聚焦到数据范围 {self.map_canvas._extent}。\n"
                "点击回到全球视图。"
                if focused else
                "把地图缩放到数据的经纬范围。")

    def _refresh_map(self):
        ds = self.dataset
        if ds is None:
            self.map_canvas.clear_data()
            self._map_ds = None
            self._map_view_idx = None
            self._sync_focus_button()
            return
        which = self.map_field.currentIndex()
        t = self.time_slider.value()
        sym = self.map_symmetric.isChecked()
        self.color_scale.symmetric = bool(sym) and self.color_scale.vcenter is None
        # 有多历元系数（批量分析）时，重建/差值/输出场都按**当前历元**算
        recon, outfield, ready = self._epoch_fields(t)
        if which in self.EPOCH_VIEWS and not ready:
            # ★ 「显示」切到重建/差值/输出场，而手上没有**这个历元**的系数时，
            # 以前会悄悄画上"分析时那一个历元的场"，或者干脆画原始数据 ——
            # 用户看到的就是"切了显示、按了播放，图还是原始图/根本不动"。
            # 现在：该补算就去补算，算不出来就**说清楚**，绝不拿别的场顶替。
            self._epoch_view_unavailable(t, which)
            return
        epo = self._epoch_note(t)

        if ds.kind == "grid":
            base = ds.grid[:, :, min(t, ds.grid.shape[2] - 1)]
            if which == 0 or recon is None:
                field = base
                self.map_canvas.show_grid(ds.lat_vec, ds.lon_vec, base,
                                          title="原始网格", symmetric=sym)
            elif which == 1:
                field = recon
                self.map_canvas.show_grid(
                    ds.lat_vec, ds.lon_vec, recon,
                    title=f"重建网格（{_gui_unit_label(self.cb_field_unit.currentData() or 'scalar', 'recon')}，与输入同量）" + epo,
                    symmetric=sym)
            elif which == 2:
                field = base - recon
                self.map_canvas.show_grid(ds.lat_vec, ds.lon_vec, field,
                                          title="差值（原始 − 重建，同一物理量）" + epo,
                                          symmetric=True)
            elif outfield is None:
                field = base
                self.map_canvas.show_grid(ds.lat_vec, ds.lon_vec, base,
                                          title="原始网格（无输出场）",
                                          symmetric=sym)
            else:
                field = outfield
                self.map_canvas.show_grid(
                    ds.lat_vec, ds.lon_vec, outfield,
                    title=f"输出场（{_gui_unit_label(self.cb_target_unit.currentData() or self.cb_field_unit.currentData() or 'scalar', 'out')}）"
                          + ("　= 重建场（「输出为」与「输入是」同档）"
                             if outfield is recon else "") + epo,
                    symmetric=sym)
        else:
            base = ds.value_slice(min(t, ds.ntime - 1))
            if which == 0 or recon is None:
                field = base
                self.map_canvas.show_points(ds.lat, ds.lon, base,
                                            title="原始散点", symmetric=sym)
            elif which == 1:
                field = np.asarray(recon).ravel()
                self.map_canvas.show_points(
                    ds.lat, ds.lon, recon,
                    title=f"重建场（{_gui_unit_label(self.cb_field_unit.currentData() or 'scalar', 'recon')}，与输入同量）" + epo,
                    symmetric=sym)
            elif which == 2:
                field = base - np.asarray(recon).ravel()
                self.map_canvas.show_points(
                    ds.lat, ds.lon, field,
                    title="差值（原始 − 重建，同一物理量）" + epo, symmetric=True)
            elif outfield is None:
                field = base
                self.map_canvas.show_points(ds.lat, ds.lon, base,
                                            title="原始散点（无输出场）",
                                            symmetric=sym)
            else:
                field = np.asarray(outfield).ravel()
                self.map_canvas.show_points(
                    ds.lat, ds.lon, field,
                    title=f"输出场（{_gui_unit_label(self.cb_target_unit.currentData() or self.cb_field_unit.currentData() or 'scalar', 'out')}）"
                          + ("　= 重建场（「输出为」与「输入是」同档）"
                             if outfield is recon else "") + epo,
                    symmetric=sym)
        self._update_color_note(np.asarray(field, dtype=float))
        self._values_dirty = True
        self._map_ds = ds
        self._map_view_idx = which
        self.map_note.setText(self._epoch_note_line(t, which))
        self._sync_focus_button()

    # ------------------------------------------------- 「显示」× 逐历元：不撒谎
    def _epoch_note_line(self, t: int, which: int) -> str:
        """地图上方那行提示：这张图到底是谁的场。"""
        if which not in self.EPOCH_VIEWS:
            return ""
        return (f"当前显示：{self._map_view_name(which)}"
                f"（第 {t + 1}/{self.dataset.ntime} 个历元）"
                + self._epoch_note(t))

    def _epoch_view_unavailable(self, t: int, which: int) -> None:
        """逐历元视图还没有系数：**一次解完整条序列** + 如实提示，不画别的场。

        契约（说明书 §8）：不能悄悄沿用旧结果。多时次数据下"没有系数"只可能是
        "还没解过" —— 那就现在解（整条序列，固定成本只付一次），并把进度写在
        提示行上；单时次数据集没有"逐历元"可言，只能请用户先运行分析。
        """
        name = self._map_view_name(which)
        t = int(t)
        nt = max(int(self.dataset.ntime), 1)
        if self.dataset.has_time() and not self._loading:
            running = (self.series_worker is not None
                       and self.series_worker.isRunning())
            if not running:
                self._ensure_series_async(f"「{name}」")
                running = (self.series_worker is not None
                           and self.series_worker.isRunning())
            detail = (f"正在一次解完 {nt} 个历元…" if running else
                      "点「运行分析」会一次解完全部历元")
            note = (f"当前显示：{name} —— 逐历元系数还没有，正在一次解完 {nt} 个历元"
                    "…（解完自动换成该历元的场；进度见右侧进度条）" if running else
                    f"当前显示：{name} —— 还没有逐历元系数。点「运行分析」即可"
                    "一次解完全部历元。")
        elif self.dataset.has_time():
            # 正在**载入**数据：只提示，不自动开工（否则"上一份数据停在重建场"
            # 会让新数据一打开就后台解完所有历元，懒加载的文件还会被整块读进来）
            detail = "点「运行分析」会一次解完全部历元"
            note = (f"当前显示：{name} —— 新载入的数据还没有系数。"
                    "点「运行分析」即可一次解完全部历元。")
        else:
            detail = "先点「运行分析」"
            note = (f"当前显示：{name} —— 还没有系数，画不出来。"
                    "先点「运行分析」。")
        self.map_canvas.clear_data()
        self._map_ds = None
        self._map_view_idx = None
        self.map_canvas.show_placeholder(
            f"第 {t + 1}/{nt} 个历元的{name}还没有系数\n{detail}")
        self.map_note.setText(note)
        self._values_dirty = True
        self._sync_focus_button()

    def _fill_spectrum(self):
        if self.coeffs is None:
            return
        curves = {"重建（球谐解）": self.coeffs.degree_rms()}
        truth = getattr(self, "_real_truth", None)
        if truth is not None and truth.nmax == self.coeffs.nmax:
            curves = {"真实场": truth.degree_rms(), **curves}
        elif self.recon is not None and self.dataset is not None:
            pass
        self.spectrum_canvas.show(curves, title="逐阶振幅 (degree RMS)",
                                  ylabel="RMS")

    def _fill_coeff_table(self):
        if self.coeffs is None:
            return
        rms = self.coeffs.degree_rms()
        power = self.coeffs.power()
        total = float(power.sum()) or 1.0
        L = self.coeffs.nmax
        self.coeff_table.setRowCount(L + 1)
        for n in range(L + 1):
            vals = [f"{n}", f"{rms[n]:.6e}",
                    f"{power[n]:.6e}",
                    f"{harmonic_resolution_km(n):.1f}" if n > 0 else "inf",
                    f"{100.0 * power[n] / total:.3f} %"]
            for c, txt in enumerate(vals):
                self.coeff_table.setItem(n, c, QTableWidgetItem(txt))
        t = int(self.time_slider.value()) if self.dataset is not None else 0
        recon, _out, ready = (self._epoch_fields(t) if self.dataset is not None
                              else (None, None, False))
        if ready and recon is not None:
            base = (self.dataset.grid[:, :, min(t, self.dataset.grid.shape[2] - 1)]
                    if self.dataset.kind == "grid"
                    else self.dataset.value_slice(t))
            res = np.asarray(base).ravel() - np.asarray(recon).ravel()
            self.coeff_note.setText(
                f"nmax = {L}，系数个数 {(L+1)**2}；第 {t + 1} 个历元重建残差 RMS = "
                f"{np.sqrt(np.nanmean(res**2)):.6e}；"
                f"数据 RMS = {np.sqrt(np.nanmean(np.asarray(base)**2)):.6e}")
        else:
            self.coeff_note.setText(f"nmax = {L}，系数个数 {(L + 1) ** 2}")

    # ================================================================ export
    def _need_result(self) -> bool:
        if self.coeffs is None:
            QMessageBox.information(self, "提示", "请先运行一次分析。")
            return False
        return True

    def export_coeffs(self):
        if not self._need_result():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出球谐系数", "shkit_coeffs.sh",
            "三角布局 (m2py/gridSHconvert 兼容) (*.sh *.txt *.csv);;"
            "ICGEM/GFZ (*.gfc);;NumPy 数组 (*.npy);;压缩 npz (*.npz)")
        if not path:
            return
        try:
            from .. import io as shio
            layout = "gfc" if path.lower().endswith(".gfc") else \
                ("npy" if path.lower().endswith(".npy") else
                 ("npz" if path.lower().endswith(".npz") else "triangle"))
            # .gfc 一个文件只放一个历元。``self.coeffs`` 现在**始终**是"当前历元
            # 那一解"（2-D 切片），所以不能再把滑块下标当 time 传进去 —— 那会
            # 在"第 5 个历元导出一个 ntime=1 的对象"时直接报 time out of range。
            kw = {}
            if layout == "gfc" and np.asarray(self.coeffs.C).ndim == 3:
                kw["time"] = self.time_slider.value()
            shio.write_coeffs(self.coeffs, path, layout=layout, **kw)
            epoch_note = (f"（第 {self._single_t + 1} 个历元）"
                          if (self._single_t is not None
                              and self.dataset is not None
                              and self.dataset.has_time()) else "")
            self.status(f"已导出系数{epoch_note} → {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------- 水平形变导出（B）
    #: 散点直算的经验速率：实测 1 036 800 点 / nmax=60 → 79.25 s，
    #: 即每（点 × 阶）约 1.27 µs（规则网格走 FFT 是 0.08 s/历元，另算）。
    _HORIZ_SEC_PER_POINT_DEG = 1.27e-6

    def _horiz_epoch_estimate(self, ds) -> float:
        """单个历元水平形变的**估算**耗时（秒）。

        只对**散点直算**有意义 —— 规则网格走经度 FFT（实测 720×1440 / nmax=60 =
        0.08 s/历元），快得不需要提示；散点是"点 × 阶"的直接扫，慢了三个数量级。
        """
        if ds.kind == "grid":
            return 0.0
        nmax = int(self.sp_nmax.value())
        return float(ds.npoints) * (nmax + 1) * self._HORIZ_SEC_PER_POINT_DEG

    def export_horizontal(self, scope: str = "current"):
        """导出水平形变：``scope='current'`` 只导当前历元，``'window'`` 逐时次。

        每个历元写 **3 个文件**（u_N / u_E / |u_h|，单位 m）—— 一个文件只装一个变量，
        这样任何工具都读得进去；文件名带分量后缀，进度条上会如实列出写了什么。
        """
        from ..gradient import canonical_scaled, horizontal_grid, horizontal_field

        ds = self.dataset
        if ds is None or (self.coeffs is None
                          and self._epoch_source_coeffs() is None):
            QMessageBox.information(self, "提示", "请先运行一次分析（要有系数）。")
            return
        multi = ds.has_time() and self._epoch_source_coeffs() is not None
        if scope == "window":
            if not multi:
                QMessageBox.information(
                    self, "导出水平形变",
                    "逐时次导出需要**逐历元**系数（多时次数据且已运行分析）。")
                return
            lo, hi = self.time_window()
        else:
            if multi and self._coeffs_for_epoch(
                    int(self.time_slider.value())) is None:
                self.status("当前历元还没有系数：先点「运行分析」。")
                return
            lo = hi = int(self.time_slider.value()) if multi else 0
        n_ep = hi - lo + 1

        # 散点直算很慢：先把估算量说清楚，让用户自己决定
        t_one = self._horiz_epoch_estimate(ds)
        if ds.kind != "grid" and n_ep * t_one > 20.0:
            ans = QMessageBox.question(
                self, "导出水平形变（散点较慢）",
                f"散点上水平形变只能直接扫：{ds.npoints} 个点、nmax="
                f"{self.sp_nmax.value()}，\n每个历元约 {t_one:.1f} 秒，"
                f"{n_ep} 个历元合计约 {n_ep * t_one / 60:.1f} 分钟"
                f"（每个历元 3 个文件）。\n\n"
                "可以：只导当前历元，或把时间窗调小。要继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                self.status("导出水平形变：已取消（散点上估算耗时较长）。")
                return
        out_dir = QFileDialog.getExistingDirectory(
            self, "选择水平形变导出的目录")
        if not out_dir:
            return
        ext = str(self.cb_export_ext.currentData() or ".nc")
        if ds.kind == "points" and ext in (".nc", ".grd"):
            ext = ".csv"
            self.cb_export_ext.setCurrentIndex(2)
            self.status("散点数据只能用 .csv 导出水平形变（.nc/.grd 是网格格式）"
                        "——已把导出格式改成 .csv。")
        prefix = self.edit_export_prefix.text().strip() or "uh"
        comps = (("uN", "north"), ("uE", "east"), ("umag", "magnitude"))
        base_meta = {"source_file": ds.path, "field": "horizontal_displacement"}

        def _horiz_for(k: int) -> tuple:
            """第 k 个历元的 (C, S, F^h_n)（单位在 worker 线程里换算，控件不碰）。"""
            co = (self._coeffs_for_epoch(k) if multi else self.coeffs)
            if co is None:
                raise ValueError(f"第 {k + 1} 个历元没有系数。")
            if np.asarray(co.C).ndim == 3:
                n = int(np.asarray(co.C).shape[2])
                co = co.time_slice(min(k, n - 1)) if hasattr(co, "time_slice") else co
            return canonical_scaled(co)

        def _write(k: int, out: str):
            C, S, fac = _horiz_for(k)
            if ds.kind == "grid":
                res = horizontal_grid(ds.lat_vec, ds.lon_vec, C, S, fac,
                                      method="auto")
            else:
                res = horizontal_field(ds.lat, ds.lon, C, S, fac, chunk=100_000)
            stem = os.path.splitext(str(out))[0]
            for tag, key in comps:
                arr = np.asarray(res[key])
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                m = dict(base_meta, epoch_index=k + 1, component=key)
                if ds.kind == "grid":
                    shio.write_grid(f"{stem}_{tag}{ext}", ds.lat_vec, ds.lon_vec,
                                    arr, var=tag, meta=m, units="m",
                                    long_name=f"horizontal displacement {key}")
                else:
                    shio.write_points(f"{stem}_{tag}{ext}", ds.lat, ds.lon,
                                      np.asarray(arr).ravel(),
                                      comment=f"# epoch {k + 1}\n# {key} [m]")

        def _label(k: int) -> str:
            return (f"{prefix}_{k + 1:04d}_"
                    + "/".join(t for t, _ in comps) + ext)

        w = ExportWorker(_write, out_dir, prefix, lo, hi, ext=ext, var="value",
                         meta=base_meta, parent=self, label_of=_label)
        dlg = ExportDialog(self, w, out_dir)
        dlg.setWindowTitle("导出水平形变")
        self._export_dialog = dlg
        w.progressed.connect(dlg.on_progress)
        w.succeeded.connect(dlg.on_done)
        w.failed.connect(dlg.on_failed)
        w.cancelled.connect(dlg.on_cancelled)
        w.finished.connect(dlg.on_finished)
        dlg.show()
        self.status(f"正在导出水平形变（{n_ep} 个历元 × 3 个分量 → {out_dir}）…")
        w.start()

    def export_field(self, which: str = "recon"):
        arr = self.recon if which == "recon" else self.outfield
        label = "重建场" if which == "recon" else "输出场"
        if not self._need_result() or arr is None:
            QMessageBox.information(self, "提示", f"没有可导出的{label}。")
            return
        ds = self.dataset
        stem = "recon" if which == "recon" else "outfield"
        default = f"{stem}.nc" if ds.kind == "grid" else f"{stem}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, f"导出{label}", default,
            "NetCDF (*.nc);;Surfer 网格 (*.grd);;三列文本 (*.csv *.txt);;"
            "NumPy (*.npy)")
        if not path:
            return
        try:
            from .. import io as shio
            unit = self._field_unit_label(which)
            if ds.kind == "grid":
                shio.write_grid(path, ds.lat_vec, ds.lon_vec, arr,
                                long_name=label, units=unit)
            else:
                shio.write_points(path, ds.lat, ds.lon, arr)
            self.status(f"已导出{label}（{unit}）→ {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", f"{type(exc).__name__}: {exc}")

    def _field_unit_label(self, which: str) -> str:
        """重建场/输出场各自的物理量标签（用于导出的 units 属性与标题）。"""
        from .. import units as _units
        u = self.cb_field_unit.currentData() or "scalar"
        if which == "recon":
            key = u
        else:
            key = self.cb_target_unit.currentData() or u
        return _units.FIELD_UNIT_LABELS.get(key, key)

    def export_report(self):
        if not self._need_result():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出诊断报告", "shkit_report.json",
            "JSON (*.json);;Markdown (*.md)")
        if not path:
            return
        try:
            from .. import io as shio
            shio.write_report(path, self.report)
            self.status(f"已导出报告 → {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", f"{type(exc).__name__}: {exc}")

    # ================================================================= about
    def show_about(self):
        """关于 / 作者信息 + 快速上手（与 GRACE Downloader 系列一致的版式）。"""
        dlg = AboutDialog(self)
        dlg.exec()

    def show_guide(self):
        """帮助 → 使用说明（F1）：打开截图版 HTML 说明书。

        优先渲染随包的 ``docs/使用说明.html``（内含 23 张图：整窗界面截图、
        左右面板原始像素特写、示例配图与公众号二维码）；
        如果发行包漏带了 HTML，就退回 Markdown 版，至少内容不丢。
        """
        dlg = GuideDialog(self)
        dlg.exec()

    def show_wechat(self):
        """课题组公众号：显示二维码；没放图片时给出公众号名称与放置说明。"""
        WeChatDialog(self).exec()

    def show_licenses(self):
        """Show the third-party notice shipped in ``licenses/NOTICE.txt``."""
        self._show_text_dialog("第三方组件与许可声明", "licenses/NOTICE.txt",
                               prefer_markdown=False)

    def show_license_text(self, rel: str, title: str):
        self._show_text_dialog(title, rel, prefer_markdown=False)

    def _show_text_dialog(self, title: str, rel: str, prefer_markdown: bool):
        path = os.path.join(_PROJECT_ROOT, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.warning(
                self, title,
                f"未找到 {rel}：{exc}\n\n"
                "若这是打包后的发行版，说明发行包缺少必需的文件；"
                "请重新使用 tools/check_licensing.py 检查发行包完整性。")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(900, 720)
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setLineWrapMode(QTextBrowser.WidgetWidth)
        if prefer_markdown:
            # 文档里的表格 / 标题 / 代码块按 Markdown 渲染（QTextBrowser 支持
            # CommonMark 子集）；渲染不了时退化为等宽纯文本，至少内容不丢。
            try:
                view.setMarkdown(text)
            except Exception:                          # noqa: BLE001
                view.setPlainText(text)
        else:
            view.setPlainText(text)
        view.moveCursor(QTextCursor.Start)
        lay.addWidget(view)
        row = QHBoxLayout()
        hint = QLabel(f"<small style='color:#777;'>{rel}</small>")
        row.addWidget(hint)
        row.addStretch(1)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        row.addWidget(btn)
        lay.addLayout(row)
        dlg.exec()

    # ================================================================ events
    def closeEvent(self, event):  # noqa: N802
        # 关窗时必须等所有 worker 收手，**包括**逐历元（series_worker）与导出
        # worker：QThread 还在跑就被销毁，Qt 会直接 fail-fast
        # （实测踩到：脚本/程序退出时 `QThread: Destroyed while thread is still
        # running` + 进程以 0xC0000409 结束）。
        for w in (self.analysis_worker, self.load_worker, self.series_worker,
                  self.horiz_worker,
                  getattr(self, "export_worker", None)):
            if w is None or not w.isRunning():
                continue
            try:
                if hasattr(w, "cancel"):
                    w.cancel()
                elif hasattr(w, "requestInterruption"):
                    w.requestInterruption()
                w.wait(5000)
            except Exception:                                    # noqa: BLE001
                pass
        # 懒加载的立方体有一条后台读文件的守护线程 + 一个打开的 nc 句柄（非
        # ASCII 路径还有一份临时副本）。关窗不放手，进程退出时那条线程还在读。
        self._lazy_timer.stop()
        if self.dataset is not None:
            self.dataset.close()
        event.accept()


# ====================================================================== HTML
_EXC_PREFIX = ("ValueError", "RuntimeError", "TypeError", "KeyError",
               "FileNotFoundError", "IndexError", "ZeroDivisionError",
               "AttributeError", "OSError", "MemoryError", "ImportError")


# =========================================================== 关于 / 作者信息
# 富文本（供「关于」对话框与 HTML 说明书共用同一份文字）。
# 注意：这里不用 Markdown —— QLabel 不认 ``code`` / ``**``，直接写标签。
_QUICK_START = [
    ("1  载入数据", "「文件 → 打开散点/网格」，或点「生成示例数据」先试一遍。"),
    ("2  选公式", "「输入是」选<b>正变换公式</b>（决定导出什么系数）；"
                  "「输出为」选<b>输出场</b>是什么物理量。只做往返就保持默认。"),
    ("3  定阶数", "「最大阶数 nmax」。想确认该取多少："
                  "<code>shkit nmax --coeffs m.sh</code>"
                  "（三个独立约束：数据带宽 / 采样 / 区域 Shannon）。"),
    ("4  跑分析", "点「运行分析」（F5）。积分元规则<b>默认 auto</b>："
                  "自动判断积分元；估计方法默认 <code>quadrature</code>（直接求积），最快。"
                  "多时次数据会<b>一次解完全部历元</b>并缓存到本地，之后看/播/导都不再重算。"),
    ("5  看结果", "地图页签：原始 / <b>重建场（与输入同量）</b> / 差值 / 输出场；"
                  "另有逐阶谱、诊断报告、系数统计。"),
    ("6  导出", "文件菜单：系数（.sh/.gfc/.npz）、重建场、输出场、诊断报告。"
                "导出系数永远是普通球谐系数 C_nm。"),
]


class ExportDialog(QDialog):
    """Progress dialog for D10 with **pause / resume / stop**.

    ``QProgressDialog`` only offers Cancel, and a 200-epoch export is long enough
    that "cancel and start over" is not a real option.
    """

    def __init__(self, parent=None, worker=None, out_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("逐时次导出")
        self.worker = worker
        self.out_dir = out_dir
        v = QVBoxLayout(self)
        self.label = QLabel(f"正在导出到 {out_dir}")
        self.label.setWordWrap(True)
        v.addWidget(self.label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        v.addWidget(self.bar)
        row = QHBoxLayout()
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self._toggle_pause)
        row.addWidget(self.btn_pause)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_stop)
        self.btn_close = QPushButton("关闭")
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_close)
        row.addStretch(1)
        v.addLayout(row)
        self.resize(460, 150)

    def _toggle_pause(self):
        if self.worker is None:
            return
        now = not self.worker._paused
        self.worker.set_paused(now)
        self.btn_pause.setText("继续" if now else "暂停")
        self.label.setText(("已暂停（已写出的历元都是完整的，不会出现半截文件）"
                            if now else "继续导出…"))

    def _stop(self):
        if self.worker is not None:
            self.worker.cancel()
            self.label.setText("已请求停止（当前历元写完后退出）…")

    def on_progress(self, msg: str, frac: float):
        self.bar.setValue(int(max(0.0, min(1.0, frac)) * 100))
        self.label.setText(msg)

    def on_done(self, written: int, out_dir: str):
        self.bar.setValue(100)
        self.label.setText(f"完成：写出 {written} 个文件到 {out_dir}")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def on_cancelled(self):
        self.label.setText(f"已停止：写出 {self.worker.written} 个文件"
                           "（每个都是完整历元）")

    def on_failed(self, tb: str):
        """Report a failure **without** a modal box.

        A ``QMessageBox`` here would block the GUI thread from inside a queued
        signal handler -- fine when a human clicks OK, but it wedges any
        unattended run (and made this feature untestable offscreen).  The dialog
        label, the header line and the status bar carry the same information.
        """
        msg = _short_error(tb)
        self.label.setText("导出失败：" + msg)
        self.bar.setStyleSheet("QProgressBar::chunk {background:#c62828;}")
        win = self.parent()
        if win is not None and hasattr(win, "status"):
            win.status(f"逐时次导出失败：{msg}")

    def on_finished(self):
        self.btn_close.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)


class AboutDialog(QDialog):
    """关于 / 作者信息 + 公众号二维码 + 快速上手（版式对齐 GRACE Downloader 系列）。

    个人信息（作者、单位、邮箱、电话、公众号）与二维码都在这一屏里：左侧是
    联系方式表，右侧直接显示公众号二维码，不用再点开二级对话框；点图片仍可
    放大。底部有直通截图版说明书的按钮。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ℹ️  关于 / About")
        self.resize(880, 800)
        self.setMinimumSize(620, 460)
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(18, 16, 18, 16)

        header = QLabel(
            f"<h2>{APP_NAME} {__version__}</h2>"
            "<p>任意散点 / 任意网格 ↔ 球谐系数。<br>"
            "球谐解算、积分元诊断、区域投影、物理量换算。</p>")
        header.setWordWrap(True)
        root.addWidget(header)

        # ---------------------------------------------------------- 作者信息
        grp = QGroupBox("作者信息  /  Author")
        outer = QHBoxLayout(grp)
        outer.setSpacing(16)

        # 左：联系方式表
        left = QWidget()
        grid = QGridLayout(left)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        rows = [
            ("作者  Author", f"{AUTHOR_NAME_CN}  （{AUTHOR_NAME_EN}）"),
            ("单位  Affiliation",
             f"{AUTHOR_AFFILIATION_CN}<br>{AUTHOR_AFFILIATION_EN}"),
            ("邮箱  E-mail", f'<a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a>'),
            ("电话  Phone", AUTHOR_PHONE),
            ("公众号  WeChat", f"{WECHAT_ACCOUNT}（{WECHAT_ACCOUNT_EN}）"),
            ("软件  Version", f"v{__version__}"),
        ]
        for r, (key, value) in enumerate(rows):
            k = QLabel(f"<b>{key}</b>")
            k.setAlignment(Qt.AlignRight | Qt.AlignTop)
            v = QLabel(value)
            v.setWordWrap(True)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse
                                      | Qt.LinksAccessibleByMouse)
            v.setOpenExternalLinks(True)
            grid.addWidget(k, r, 0)
            grid.addWidget(v, r, 1)
        grid.setColumnStretch(1, 1)

        btns = QHBoxLayout()
        b_mail = QPushButton("✉️  发邮件  (E-mail)")
        b_mail.setMinimumHeight(30)
        b_mail.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl(f"mailto:{AUTHOR_EMAIL}")))
        b_wx = QPushButton("📱  放大二维码")
        b_wx.setMinimumHeight(30)
        b_wx.clicked.connect(self._show_wechat)
        b_guide = QPushButton("📘  使用说明（截图版）")
        b_guide.setMinimumHeight(30)
        b_guide.setToolTip("打开带界面截图与示例配图的 HTML 使用说明（F1）")
        b_guide.clicked.connect(self._show_guide)
        for b in (b_mail, b_wx, b_guide):
            btns.addWidget(b)
        btns.addStretch(1)
        grid.addLayout(btns, len(rows), 0, 1, 2)
        outer.addWidget(left, stretch=1)

        # 右：公众号二维码（直接显示，不必再点）
        outer.addWidget(self._qr_panel(), stretch=0)
        root.addWidget(grp)

        # ---------------------------------------------------------- 使用说明
        guide = QGroupBox("使用说明（快速上手）  /  Quick start")
        gl = QVBoxLayout(guide)
        for title, body in _QUICK_START:
            lbl = _hint_label(f"<b>{title}</b>　{body}",
                              style="font-size:9.5pt;", bold=False)
            gl.addWidget(lbl)
        gl.addWidget(_hint_label(
            "完整说明：<b>帮助 → 使用说明</b>（快捷键 <b>F1</b>）"
            "—— 截图版 HTML（界面截图 + 示例配图），随包文件 "
            f"<code>docs/{GUIDE_HTML_NAME}</code>；"
            f"纯文本版 <code>docs/{GUIDE_MD_NAME}</code>。",
            style="color:#357;font-size:9pt;"))
        root.addWidget(guide)

        # ------------------------------------------------------ 许可 / 致谢
        credit = _hint_label(
            "<small>图形界面：Qt for Python（<b>PySide6-Essentials</b>，GNU LGPL v3）"
            "—— 动态链接、未修改；刻意未安装 Qt Charts / Qt Data Visualization "
            "等 GPL-only 模块，绘图改用 matplotlib。LGPLv3 全文见 "
            "licenses/LGPL-3.0.txt，第三方声明见 licenses/NOTICE.txt。"
            "SHKit 自身代码采用 MIT 许可。<br>"
            "载荷勒夫数表来自 PREM-LLNs.dat（Wang et al. 2012）；"
            "海岸线来自 Natural Earth 110m。本软件仅供科研与教学使用。</small>",
            style="color:#666;font-size:9pt;")
        root.addWidget(credit)
        root.addStretch(1)

        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(self.reject)
        root.addWidget(box)

    # ------------------------------------------------------------- 子部件
    def _qr_panel(self) -> QWidget:
        """二维码小面板：有图就显示，没图就给出公众号名称与放置说明。"""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = QLabel(f"<b>{WECHAT_ACCOUNT}</b><br>"
                     f"<small style='color:#666;'>{WECHAT_ACCOUNT_EN} · "
                     "微信扫一扫</small>")
        cap.setAlignment(Qt.AlignCenter)
        lay.addWidget(cap)

        path = qr_image_path()
        pix = QPixmap(path) if path else QPixmap()
        if not pix.isNull():
            img = QLabel()
            img.setPixmap(pix.scaled(180, 180, Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation))
            img.setAlignment(Qt.AlignCenter)
            img.setToolTip("点击放大二维码")
            img.setCursor(Qt.PointingHandCursor)
            img.mousePressEvent = lambda _ev: self._show_wechat()  # noqa: ARG005
            lay.addWidget(img)
        else:
            miss = _hint_label(
                f"未找到二维码图片。<br>把 <code>{WECHAT_QR_FILENAME}</code> 放到"
                "程序目录或 <code>docs/</code> 下即可显示。",
                style="color:#777;font-size:8.5pt;")
            miss.setFixedWidth(190)
            miss.setAlignment(Qt.AlignCenter)
            lay.addWidget(miss)
        return panel

    def _show_guide(self):
        GuideDialog(self).exec()

    def _show_wechat(self):
        WeChatDialog(self).exec()


class WeChatDialog(QDialog):
    """课题组公众号二维码（关于对话框里的「放大二维码」也走这里）。"""

    def __init__(self, parent=None, size: int = 360):
        super().__init__(parent)
        self.setWindowTitle(f"📱  {WECHAT_QR_CAPTION}")
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(16, 16, 16, 16)

        t = QLabel(f"<h3>{WECHAT_QR_CAPTION}</h3>")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        hint = QLabel("微信扫一扫，关注课题组公众号。")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

        panel = QLabel()
        panel.setAlignment(Qt.AlignCenter)
        panel.setMinimumSize(size, size)
        path = qr_image_path()
        pix = QPixmap(path) if path else QPixmap()
        if not pix.isNull():
            panel.setPixmap(pix.scaled(size, size, Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))
            panel.setToolTip(path)
        else:
            panel.setText(
                f"❌  未找到二维码图片\n\n"
                f"请把 {WECHAT_QR_FILENAME} 放到程序目录（或 docs/）下，"
                "重新打开即可。")
            panel.setWordWrap(True)
        lay.addWidget(panel, stretch=1)

        row = QHBoxLayout()
        row.addStretch(1)
        b = QPushButton("关闭")
        b.clicked.connect(self.accept)
        row.addWidget(b)
        lay.addLayout(row)


class GuideDialog(QDialog):
    """截图版 HTML 使用说明（``docs/使用说明.html``）的阅读窗口。

    页面里的图片是相对路径，靠 ``setSearchPaths`` 指向 docs/ 才能显示。HTML 里
    **不写死图片宽度**：这里在渲染后按真实视口宽度逐张定尺寸（只在超过可读宽度
    时才缩小、绝不放大过原图），所以窗口拉大图片跟着变大、不会溢出，浏览器里
    打开同一份 HTML 也仍然是排版好的（HTML 里带 ``max-width`` 兜底）。

    ``figure`` 与 ``figcaption`` 在 Qt 富文本里没有样式，所以文档用的是
    ``<p align="center">`` + 一行居中小字题注，Qt 与浏览器都居中。

    缺 HTML 时退回 Markdown 版；两者都没有才报错。
    """

    #: 文档两侧留出的余量（文档边距 + 竖直滚动条 + 表格边框），避免多出横向滚动条。
    #: 实测 80 还不够（窄窗口下仍会溢出几个像素），留 120 有余量。
    SIDE_MARGIN_PX = 120

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📘  使用说明  /  User Guide")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        if avail is not None:
            width = max(720, min(1080, avail.width() - 60))
            height = max(520, min(920, avail.height() - 60))
        else:
            width, height = 1000, 880
        self.resize(width, height)
        self.setMinimumSize(640, 480)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 10, 12, 10)

        self._view = QTextBrowser()
        self._view.setOpenExternalLinks(True)
        self._view.setFrameShape(QFrame.StyledPanel)
        root.addWidget(self._view, stretch=1)

        self._status = _hint_label("", style="color:#666;font-size:9pt;",
                                   bold=False)
        root.addWidget(self._status)

        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("关闭")
        btn.clicked.connect(self.accept)
        row.addWidget(btn)
        root.addLayout(row)

        self._shown = False
        self._html = ""
        self._base_dir = ""
        self._natural = {}                     # basename -> (w, h)
        self.refresh()

    def showEvent(self, event):  # noqa: N802
        """第一次显示后用真实视口宽度再渲染一次（图片才不会切边）。"""
        super().showEvent(event)
        if not self._shown:
            self._shown = True
            QTimer.singleShot(0, self.refresh)

    def resizeEvent(self, event):  # noqa: N802
        """窗口变大时把图片重新算一遍，否则图片会一直停在小尺寸。"""
        super().resizeEvent(event)
        if self._shown:
            QTimer.singleShot(80, self._refit)

    def _refit(self):
        """Re-fit and re-render the guide (used after the window was resized)."""
        source = getattr(self, "_qt_html", "") or self._html
        if not source:
            return
        html, _n = self._fit_images(source, self._fit_width())
        self._view.setHtml(html)

    def _fit_width(self) -> int:
        """Available width for the reading column, avoiding a spurious h-scrollbar.

        竖直滚动条本身要占掉十几个像素；不算进去，窄窗口下就会出现一条没有
        用的横向滚动条。
        """
        view = self._view.viewport()
        width = view.width()
        if width <= 0:
            return 860
        if self._view.verticalScrollBar().isVisible():
            width -= self._view.verticalScrollBar().sizeHint().width()
        return width

    # ------------------------------------------------------------ 图片缩放
    def _natural_size(self, name: str):
        """Image size in device-independent pixels (0, 0 when unreadable).

        ``name`` 是 HTML 里的相对路径，可能带子目录（如
        ``使用说明_img/gui_01.png``），所以按完整相对路径查，不能只取 basename。
        """
        key = name.replace("\\", "/")
        if key not in self._natural:
            pix = QPixmap(os.path.join(self._base_dir, *key.split("/")))
            if pix.isNull():
                self._natural[key] = (0, 0)
            else:
                dpr = float(pix.devicePixelRatio()) or 1.0
                self._natural[key] = (pix.width() / dpr, pix.height() / dpr)
        return self._natural[key]

    def _fit_images(self, html: str, viewport_px: int) -> tuple:
        """Return ``(html, n)``: every ``<img>`` sized to the reading column.

        Qt 的富文本引擎只在**解析 HTML 时**读 ``width``/``height`` 属性；解析完
        再用 ``QTextCursor`` 改图片格式是没用的（实测无效，图片仍按原始宽度排版
        并溢出）。所以这里在 ``setHtml`` 之前把尺寸写进标签里。

        只在图片比可读宽度还宽时才缩小，绝不放大到超过原始像素——放大就会糊。
        """
        limit = max(240.0, float(viewport_px) - self.SIDE_MARGIN_PX)
        count = 0

        def repl(match: "re.Match") -> str:
            nonlocal count
            src = match.group(1)
            nat_w, nat_h = self._natural_size(src)
            if nat_w > 0:
                scale = min(1.0, limit / nat_w)
            else:
                # 图片读不出来（发行包丢图）：按栏宽占位，至少不溢出
                scale = min(1.0, limit / 860.0)
                nat_h = 0.0
            target_w = int(round(nat_w * scale)) if nat_w else int(limit)
            target_h = int(round(nat_h * scale)) if nat_h else 0
            count += 1
            size = f'width="{max(64, target_w)}"'
            if target_h:
                size += f' height="{target_h}"'
            return f'<img src="{src}" {size}'

        return re.sub(r'<img src="([^"]+)"', repl, html), count

    # ----------------------------------------------------------- Qt 兼容处理
    @staticmethod
    def _for_qt(html: str) -> str:
        """Strip the outer HTML skeleton before handing the page to Qt.

        Qt 的富文本引擎有个坑：只要文档里出现 ``<head>``（哪怕只是包着
        ``<style>``），后面**每张图片**都会被多留出一大块空白 —— 它按
        ``图高 × 文档宽 / 图宽`` 预留行高，而不是按缩放后的真实高度。实测同一张
        600×820 的图：``<style>`` + body 时图与题注间距 12 px，套上 ``<head>``
        后变成 463 px，看起来就像「图片和标题之间空了半屏」。

        所以渲染前把 ``<!DOCTYPE>`` / ``<html>`` / ``<head>`` / ``<body>`` 外壳
        去掉，只把 ``<style>`` 的内容内联到开头；``body`` 上的 CSS 改挂到一个
        ``<div>`` 上，免得不套 ``<body>`` 就丢掉字体与字号。文件本身仍是完整合法
        的 HTML（浏览器打开时排版照旧），这里只改喂给 Qt 的那一份。
        """
        styles = re.findall(r"<style[^>]*>(.*?)</style>", html,
                            flags=re.S | re.I)
        body_attrs = ""
        match = re.search(r"<body([^>]*)>(.*?)</body>", html, flags=re.S | re.I)
        if match:
            attrs, body = match.group(1), match.group(2)
            style = re.search(r'style\s*=\s*"([^"]*)"', attrs, flags=re.I)
            body_attrs = f' style="{style.group(1)}"' if style else ""
        else:
            body = html
        head = "".join(f"<style>{s}</style>" for s in styles)
        return f"{head}<div{body_attrs}>{body}</div>"

    # -------------------------------------------------------------- 渲染
    def refresh(self):
        html_path = guide_html_path()
        if html_path:
            if os.path.abspath(html_path) != getattr(self, "_loaded", ""):
                try:
                    with open(html_path, encoding="utf-8") as fh:
                        self._html = fh.read()
                except Exception as exc:                # noqa: BLE001
                    self._view.setPlainText(f"无法读取 {html_path}：{exc}")
                    self._status.setText("")
                    return
                self._base_dir = os.path.dirname(html_path)
                self._loaded = os.path.abspath(html_path)
                self._natural = {}
                self._qt_html = self._for_qt(self._html)
            try:
                self._view.setSearchPaths([self._base_dir])
            except Exception:                          # noqa: BLE001
                pass
            html, _n = self._fit_images(self._qt_html, self._fit_width())
            self._view.setHtml(html)
            self._status.setText(
                f"截图版说明书：{html_path}"
                "　（在本窗口内滚动阅读；图片随包分发，无需联网）")
            return

        md_path = guide_md_path()
        if md_path:
            try:
                with open(md_path, encoding="utf-8") as fh:
                    self._view.setMarkdown(fh.read())
                self._status.setText(
                    f"未找到 docs/{GUIDE_HTML_NAME}，已退回 Markdown 版：{md_path}")
            except Exception as exc:                    # noqa: BLE001
                self._view.setPlainText(f"无法读取 {md_path}：{exc}")
            return

        self._view.setPlainText(
            f"未找到使用说明文件。\n\n"
            f"发行包应包含 docs/{GUIDE_HTML_NAME} 与其配套图片目录；"
            "请用 tools/check_licensing.py 检查发行包完整性，或重新解压。")
        self._status.setText("")


def _short_error(tb: str) -> str:
    """Pick the most informative line out of a traceback for the error dialog.

    The last line of a traceback is the exception line, e.g.
    ``ValueError: analysis(output_unit='ewh') 无法完成换算：`` — that, plus the
    first line of the message body, is what the user needs.  Naively taking the
    last line containing a colon used to surface a mid-message closing sentence
    ("...把目标设为「不换算」即可。") instead of the actual cause, so prefer a
    line that *starts* with an exception name and only fall back to a colon.
    """
    lines = [ln.strip() for ln in (tb or "").strip().splitlines() if ln.strip()]
    if not lines:
        return "未知错误"
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(_EXC_PREFIX):
            out = lines[i]
            if i + 1 < len(lines) and not lines[i + 1].startswith(_EXC_PREFIX):
                body = lines[i + 1]
                out += "\n" + (body if len(body) <= 140 else body[:139] + "…")
            return out
    for ln in reversed(lines):
        if "：" in ln:
            return ln
    return lines[-1]


def _fmt(v, nd=6):
    """Human-friendly number: plain decimal in the normal range, scientific
    outside it.  ``nd`` is decimal places (not significant digits), so
    2001.5 with ``nd=1`` renders as ``2,001.5`` rather than ``2e+03``."""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(f):
        return "—"
    if f == 0:
        return "0"
    a = abs(f)
    if a >= 1e6 or a < 1e-3:
        return f"{f:.{max(min(int(nd), 6), 2)}e}"
    if a >= 1000:
        return f"{f:,.1f}"
    k = max(min(int(nd), 8), 1)
    s = f"{f:.{k}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def _sci(v, nd=2):
    """Always-scientific formatting for quantities spanning many decades."""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(f):
        return "—"
    return "0" if f == 0 else f"{f:.{nd}e}"


def _report_to_html(report, dataset=None) -> str:
    r = report.report()
    rows = [
        ("估计方法", r["method"]),
        ("积分元规则", r["weight_rule"]),
        ("采样点数", r["n_points"]),
        ("nmax / 系数个数", f"{r['nmax']} / {r['ncoef']}"),
        ("超定比", _fmt(r["overdetermination"], 3)),
        ("权重和 sum(w)", f"{_fmt(r['weight_sum'])} （4π = 12.566370614）"),
        ("覆盖率", _fmt(r["coverage"], 6)),
        ("Shannon 数", _fmt(r["shannon"], 2)),
        ("求积完备性 max|K−I|", _sci(r["gram_deviation"])),
        ("条件数", _sci(r["condition_number"])),
        ("拟合残差 RMS", _fmt(r["residual_rms"])),
        ("数据 RMS", _fmt(r["data_rms"])),
        ("相对拟合误差", _fmt(r["fit_rmse_rel"], 3)),
        ("正则化", f"{r['regularization'] or '未启用'} "
                   f"(alpha = {_fmt(r['alpha'], 3)})"),
        ("nmax 处半波长分辨率", f"{_fmt(r['resolution_km'], 1)} km"),
        ("推荐 nmax", r["lmax_recommended"]),
    ]
    # 系数的物理含义 —— 这是最容易出错、也最需要写明的一项
    try:
        from .. import units as _units
        fu = getattr(report, "meta", {}).get("field_unit", "unknown")
        rows.insert(3, ("系数物理含义",
                        _units.FIELD_UNIT_LABELS.get(fu, str(fu))))
    except Exception:                                  # noqa: BLE001
        pass
    html = ["<style>td{padding:3px 10px 3px 0;} "
            "th{text-align:left;padding-right:10px;}</style>",
            "<h3>诊断报告</h3><table>"]
    for k, v in rows:
        html.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
    html.append("</table>")

    warnings = r.get("warnings") or []
    if warnings:
        html.append("<h3 style='color:#b35;'>警告</h3><ul>")
        for w in warnings:
            html.append(f"<li style='color:#b35;'>{w}</li>")
        html.append("</ul>")
    else:
        html.append("<p style='color:#2a7;'>无警告：采样与阶数看起来是健康的。</p>")

    if r.get("auto_note") or report.meta.get("auto_choice"):
        html.append(f"<p>自动选择：<code>{report.meta.get('auto_choice')}</code>"
                    "</p>")
    extra = {k: v for k, v in report.meta.items()
             if k in ("alpha_selection", "cond_note", "gram_note",
                      "cg_info", "ntaper_used", "shannon")
             and not isinstance(v, (list, dict))}
    if extra:
        html.append("<h3>补充信息</h3><table>")
        for k, v in extra.items():
            html.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
        html.append("</table>")

    html.append("<p style='color:#888;font-size:9pt;'>"
                "提示：<b>max|K−I|</b> 小说明采样近似正交，纯求积即可；"
                "大则说明必须用最小二乘。<b>覆盖率</b> &lt; 1 说明只覆盖了部分球面，"
                "此时结果为「区域外为 0」的全球系数，需谨慎解读。</p>")
    return "".join(html)


def _epoch_report_from(base, table: dict, t: int):
    """把整条序列的**共享**报告补成"第 t 个历元"的那一份。

    ``analyze_series`` 只把每个分块的**首个** ``AnalysisReport`` 留下来（积分元 /
    Gram / 推荐阶数这些量与历元无关），逐历元数值在 ``table`` 里。诊断报告页要显示
    "当前历元"的数字，所以这里用 ``dataclasses.replace`` 造一份同型的副本 ——
    不改动原对象（它被所有历元共用）。
    """
    from dataclasses import replace
    kw = {}
    for field_name, col in (("residual_rms", "residual_rms"),
                            ("data_rms", "data_rms"),
                            ("dc_mean_expected", "dc_mean_expected"),
                            ("dc_mean_got", "c00")):
        v = (table or {}).get(col)
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if t < arr.size and np.isfinite(arr[t]):
            kw[field_name] = float(arr[t])
    if not kw:
        return base
    try:
        return replace(base, **kw)
    except Exception:                                        # noqa: BLE001
        return base
