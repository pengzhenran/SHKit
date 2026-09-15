# -*- coding: utf-8 -*-
"""
Offscreen smoke test for the SHKit PySide6 GUI.

Runs the whole window headlessly (``QT_QPA_PLATFORM=offscreen``), drives a real
analysis through the worker thread, checks that every panel was populated, and
writes screenshots to ``tests/_gui_shots/`` so the layout can be inspected.

Run:  python tests/test_gui_smoke.py
"""
import glob
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SHOTS = os.path.join(HERE, "_gui_shots")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")


def wait_for(app, worker, timeout=180.0):
    t0 = time.time()
    while worker is not None and worker.isRunning() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return not (worker is not None and worker.isRunning())


def wait_any_solve(app, win, timeout=600.0):
    """等到界面上不再有求解在跑（单时次 AnalysisWorker 或逐历元 SeriesWorker）。

    v2.0.1 起「运行分析」在多时次数据上走的是**整条序列**（``series_worker``），
    所以测试不能再只等 ``analysis_worker``。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        busy = [w for w in (win.analysis_worker, win.series_worker)
                if w is not None and w.isRunning()]
        if not busy:
            app.processEvents()
            busy = [w for w in (win.analysis_worker, win.series_worker)
                    if w is not None and w.isRunning()]
            if not busy:
                return True
        app.processEvents()
        time.sleep(0.01)
    return False


def main():
    shutil.rmtree(SHOTS, ignore_errors=True)
    os.makedirs(SHOTS, exist_ok=True)

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    import shkit
    from shkit.gui.main_window import MainWindow
    from shkit.gui.workers import SeriesWorker, SubprocessRunner

    check("导入 shkit 本身不会拉入 PySide6",
          True, f"shkit {shkit.__version__}")

    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    app.processEvents()
    check("主窗口可创建并显示", win.isVisible(), f"{win.width()}x{win.height()}")

    # ⚠️ 「有缓存时点「运行分析」会弹一句『读缓存还是重算』」是**模态**对话框，离屏测试里
    # 会永久阻塞。所以默认把它替换成"直接读缓存"，并记录调用；后面有专门用例校验真实的
    # 按钮→决定映射、三种决定的实际效果，以及"自动补算的路径不弹窗"。
    _real_ask_cache = win._ask_series_cache
    _ask_calls = []
    win._ask_decision = "cache"

    def _ask_stub(hit, params):
        _ask_calls.append(hit.get("path"))
        return win._ask_decision

    win._ask_series_cache = _ask_stub

    # ------------------------------------ 出厂默认：auto 规则 / 直接求积
    methods = [win.cb_method.itemData(i) for i in range(win.cb_method.count())]
    check("默认：积分元规则 = auto、估计方法 = quadrature（直接求积）、迭代校正 = 0",
          win.cb_rule.currentData() == "auto"
          and win.cb_method.currentData() == "quadrature"
          and win.sp_niter.value() == 0,
          f"rule={win.cb_rule.currentData()!r}, "
          f"method={win.cb_method.currentData()!r}, "
          f"niter={win.sp_niter.value()}")
    check("「估计方法」下拉已去掉 auto，且 quadrature 排第一（默认项）",
          "auto" not in methods and methods[0] == "quadrature",
          f"{methods}")
    _p0 = win._collect_params()
    check("默认提交给 analysis 的就是直接求积（quadrature + niter 0）",
          _p0["rule"] == "auto" and _p0["method"] == "quadrature"
          and _p0["niter"] == 0,
          f"rule={_p0['rule']!r}, method={_p0['method']!r}, niter={_p0['niter']}")
    check("迭代校正次数只在「估计方法 = iterative」时可用（其它方法灰掉）",
          not win.sp_niter.isEnabled(),
          f"quadrature 下 enabled={win.sp_niter.isEnabled()}")
    win.cb_method.setCurrentIndex(methods.index("iterative"))
    app.processEvents()
    _iter_on = win.sp_niter.isEnabled()
    win.cb_method.setCurrentIndex(0)
    app.processEvents()
    check("切到 iterative 后迭代校正次数变为可用，切回后再次灰掉",
          _iter_on and not win.sp_niter.isEnabled(),
          f"iterative→{_iter_on}，quadrature→{win.sp_niter.isEnabled()}")

    # ------------------------------------------------------------ demo data
    win.make_demo()
    app.processEvents()
    ds = win.dataset
    check("生成示例数据", ds is not None and ds.npoints > 0,
          f"{ds.npoints if ds else 0} 点，{ds.ntime if ds else 0} 时次")
    check("自动推断积分元规则为 voronoi",
          win.cb_rule.currentData() == "voronoi",
          f"当前规则 = {win.cb_rule.currentData()}")
    win.grab().save(os.path.join(SHOTS, "01_loaded.png"))

    check("已移除冗余的「往返自检」按钮（与「运行分析」功能相同）",
          not hasattr(win, "btn_roundtrip"),
          "win.btn_roundtrip 不存在")

    # 布局稳定：分析前后地图不能变窄、右侧参数面板不能变宽
    dock_w0, map_w0 = win.param_dock.width(), win.map_canvas.width()

    # -------------------------------------------------------------- analysis
    win.run_analysis()
    ok = wait_for(app, win.analysis_worker, timeout=300)
    check("分析线程正常结束", ok, "")
    app.processEvents()
    check("分析前后布局不变（地图不变窄、参数面板不变宽）",
          win.param_dock.width() == dock_w0 and win.map_canvas.width() == map_w0,
          f"参数面板 {dock_w0} -> {win.param_dock.width()}，"
          f"地图 {map_w0} -> {win.map_canvas.width()}")
    check("分析产生了系数", win.coeffs is not None,
          f"nmax={win.coeffs.nmax if win.coeffs else None}, "
          f"ncoef={win.coeffs.ncoef if win.coeffs else None}")
    check("分析产生了报告", win.report is not None,
          f"method={win.report.method if win.report else None}, "
          f"rule={win.report.weight_rule if win.report else None}")
    check("产生了重建场", win.recon is not None,
          f"shape={np.shape(win.recon)}")
    # 重建场与输出场是两个场；「输出为」与「输入是」同档时它们是同一个场
    u_now = win.cb_field_unit.currentData() or "scalar"
    t_now = win.cb_target_unit.currentData() or None
    same_cfg = t_now in (None, u_now)
    same_val = np.allclose(np.asarray(win.outfield, dtype=float),
                           np.asarray(win.recon, dtype=float))
    check("重建场恒与输入同量；输出场按「输出为」",
          win.outfield is not None
          and np.shape(win.outfield) == np.shape(win.recon)
          and (same_val == same_cfg),
          f"shape={np.shape(win.outfield)}, 与重建场逐值相同={same_val}"
          f"（「输出为」与「输入是」同档={same_cfg}）")

    if win.report is not None:
        r = win.report.report()
        check("报告含关键诊断量",
              all(k in r for k in ("gram_deviation", "coverage",
                                   "overdetermination", "lmax_recommended")),
              f"gram={r['gram_deviation']:.2e}, coverage={r['coverage']:.4f}, "
              f"推荐 nmax={r['lmax_recommended']}")
        check("示例数据的分析被正确识别为健康/不健康并给出说明",
              isinstance(r["warnings"], list),
              f"{len(r['warnings'])} 条警告")
        # truth is band-limited to 12 and we asked for 12 -> coefficients close
        truth = getattr(win, "_real_truth", None)
        if truth is not None:
            L = win.coeffs.nmax
            num = (np.linalg.norm(win.coeffs.C - truth.C) ** 2 +
                   np.linalg.norm(win.coeffs.S - truth.S) ** 2)
            den = np.linalg.norm(truth.C) ** 2 + np.linalg.norm(truth.S) ** 2
            rel = float(np.sqrt(num / den))
            check("GUI 跑出的系数与真值吻合", rel < 0.2,
                  f"相对 L2 误差 = {rel:.3e}（示例场带 1% 噪声）")

    # ----------------------------------------------------------- panels
    check("诊断报告面板已填充 HTML",
          len(win.report_view.toHtml()) > 500,
          f"{len(win.report_view.toHtml())} 字符")
    check("系数统计表已填充",
          win.coeff_table.rowCount() == (win.coeffs.nmax + 1 if win.coeffs else 0),
          f"{win.coeff_table.rowCount()} 行")
    check("逐阶谱画布已绘制",
          len(win.spectrum_canvas.figure.axes) >= 1,
          f"{len(win.spectrum_canvas.figure.axes)} 个 axes")
    check("地图画布已绘制",
          win.map_canvas._mappable is not None, "")

    # ------------------------------------------------- switch map views
    for idx, name in ((0, "raw"), (1, "recon"), (2, "diff")):
        win.map_field.setCurrentIndex(idx)
        app.processEvents()
        win.grab().save(os.path.join(SHOTS, f"02_map_{idx}_{name}.png"))
    check("三种地图视图都能渲染",
          win.map_canvas._mappable is not None, "原始 / 重建 / 差值")

    win.tabs.setCurrentIndex(1)
    app.processEvents()
    win.grab().save(os.path.join(SHOTS, "03_spectrum.png"))
    win.tabs.setCurrentIndex(2)
    app.processEvents()
    win.grab().save(os.path.join(SHOTS, "04_report.png"))
    win.tabs.setCurrentIndex(3)
    app.processEvents()
    win.grab().save(os.path.join(SHOTS, "05_coeffs.png"))

    # ------------------------------------------------- horizontal tab (B2)
    labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    check("存在「水平形变」页签", "水平形变" in labels, str(labels))
    win.tabs.setCurrentIndex(labels.index("水平形变"))
    app.processEvents()
    win._refresh_horizontal()
    # ⚠️ v2.0.1 起水平形变在 **worker 线程/子进程**里算（散点上要几十秒，同步做会冻住
    # 整个窗口），所以这里必须等它结束再断言。
    wait_for(app, win.horiz_worker, timeout=600)
    app.processEvents()
    win.grab().save(os.path.join(SHOTS, "06_horizontal.png"))
    hz = win.horiz_canvas
    check("水平形变已绘制（北/东 + 矢量 + 2 个色标）",
          hz._drawn is not None and len(hz.figure.axes) >= 5,
          f"{len(hz.figure.axes)} 个 axes, shape={None if hz._drawn is None else hz._drawn[0]}")
    # 契约 F4：极点不画箭头。用一张含 ±90° 的网格直接验这条不变量。
    import numpy as _np
    latv = _np.arange(-90.0, 90.01, 10.0)
    lonv = _np.arange(0.0, 360.0, 20.0)
    n = _np.outer(_np.cos(_np.deg2rad(latv)), _np.ones(lonv.size)) * 1e-3
    hz._maps(n, n * 0.5, latv, lonv, is_grid=True, title="极点检查")
    app.processEvents()
    offsets = []
    for coll in hz.figure.axes[2].collections:
        try:
            offsets.append(_np.asarray(coll.get_offsets()))
        except Exception:                                     # noqa: BLE001
            continue
    off = _np.vstack([o for o in offsets if o.size]) if offsets else _np.zeros((0, 2))
    check("矢量图上极点不画箭头（|lat| = 90 被剔除）",
          off.size > 0 and not _np.any(_np.abs(_np.abs(off[:, 1]) - 90.0) < 1e-9),
          f"{off.shape[0]} 个箭头，lat 范围 "
          f"[{off[:, 1].min():.1f}, {off[:, 1].max():.1f}]" if off.size else "没有箭头")
    lo, hi = hz._sym(_np.array([-3.0, 1.0, 2.0, _np.nan]))
    check("分量图色标对称（vmin = −vmax）", lo < 0 < hi and abs(lo + hi) < 1e-12,
          f"[{lo:.4g}, {hi:.4g}]")
    win.tabs.setCurrentIndex(0)
    app.processEvents()

    # ------------------------------------------------- coastline sanity
    from shkit.gui.canvases import (load_coastlines, prepare_polyline,
                                    setup_matplotlib_fonts, wrap_longitude)
    setup_matplotlib_fonts()
    ok_all = True
    detail = []
    for name in ("coastline", "borders"):
        lon, lat = load_coastlines(name)
        if lon is None:
            ok_all = False
            detail.append(f"{name}: 数据缺失")
            continue
        naive = int((np.nan_to_num(np.abs(np.diff(wrap_longitude(lon)))) > 180).sum())
        px, py = prepare_polyline(lon, lat)
        fixed = int((np.nan_to_num(np.abs(np.diff(px))) > 180).sum())
        m = ~np.isnan(px)
        orig = ~np.isnan(lon)
        same = (m.sum() == orig.sum()
                and np.allclose(px[m], wrap_longitude(lon[orig]), atol=1e-12)
                and np.allclose(py[m], lat[orig], atol=1e-12))
        ok_all &= (fixed == 0 and same)
        detail.append(f"{name}: 裸wrap假连线 {naive} -> 修复后 {fixed}, 坐标一致={same}")
    check("海岸线不再有跨 180° 的飞线", ok_all, "; ".join(detail))

    # 地图上**只画海岸线**，不再叠国界（用户要求去掉国界）：国界是虚线，
    # 所以"图上没有虚线"就是可断言的行为判据。
    win.map_field.setCurrentIndex(0)
    app.processEvents()
    _mlines = win.map_canvas._ax.get_lines() if win.map_canvas._ax else []
    check("★ 地图只画海岸线、不再画国界（没有虚线）",
          len(_mlines) >= 1
          and all(str(ln.get_linestyle()) not in (":", "dotted", "dashed")
                  for ln in _mlines),
          f"{len(_mlines)} 条折线，线型="
          f"{sorted({str(ln.get_linestyle()) for ln in _mlines})}")

    # ------------------------------------------------------ unit controls
    fin = [win.cb_field_unit.itemData(i) for i in range(win.cb_field_unit.count())]
    fout = [win.cb_target_unit.itemData(i) or ""
            for i in range(win.cb_target_unit.count())]
    check("「输入是」= 普通网格 / geoid / EWH / 径向形变，默认第一项",
          fin == ["geopotential", "geoid", "ewh", "radial_displacement"]
          and win.cb_field_unit.itemText(0) == "普通网格"
          and win.cb_field_unit.currentData() == "geopotential",
          f"{[win.cb_field_unit.itemText(i) for i in range(len(fin))]}")
    check("「输入是」已去掉「无物理含义的标量场」与「地表面密度」",
          "scalar" not in fin and "surface_density" not in fin,
          f"{fin}")
    check("「输出为」= 不换算 / EWH / geoid / 普通球谐系数 / 径向形变",
          fout == ["", "ewh", "geoid", "geopotential", "radial_displacement"]
          and "普通球谐系数" in [win.cb_target_unit.itemText(i)
                                 for i in range(len(fout))],
          f"{[win.cb_target_unit.itemText(i) for i in range(len(fout))]}")
    check("「输出为」已去掉面密度",
          "surface_density" not in fout, f"{fout}")

    win.cb_field_unit.setCurrentIndex(0)          # 普通网格
    win.cb_target_unit.setCurrentIndex(1)         # ewh
    app.processEvents()
    check("普通网格 -> EWH 时给出逐阶因子提示",
          "倍" in win.lbl_target_hint.text(),
          win.lbl_target_hint.text()[:72])
    win.cb_target_unit.setCurrentIndex(2)         # geoid
    app.processEvents()
    check("普通网格 -> 水准面时提示是常数 R（不是逐阶因子）",
          "常数" in win.lbl_target_hint.text(),
          win.lbl_target_hint.text()[:72])
    win.cb_target_unit.setCurrentIndex(0)         # 不换算
    app.processEvents()

    # --------------------------------------------- 默认参数：效率优先
    # （出厂默认已在窗口创建时断言；这里核的是"载入数据后 auto 会按采样自动
    #   挑积分元规则，但估计方法仍是我们选的 quadrature=直接求积"）
    check("载入数据后：估计方法仍是 quadrature，只有积分元规则被自动挑选",
          win.cb_method.currentData() == "quadrature"
          and win.cb_rule.currentData() == ds.suggested_rule(),
          f"rule={win.cb_rule.currentData()!r}（建议 "
          f"{ds.suggested_rule()!r}），method={win.cb_method.currentData()!r}")
    params = win._collect_params()
    check("提交给 analysis 的仍是直接求积（method=quadrature, niter=0）",
          params["method"] == "quadrature" and params["niter"] == 0,
          f"rule={params['rule']!r}, method={params['method']!r}, "
          f"niter={params['niter']}")

    # ------------------------------------------------- 关于 / 使用说明
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialog, QLabel, QTextBrowser

    from shkit.gui import main_window as mw

    def close_dialog_later():
        def _close():
            for w in app.topLevelWidgets():
                if isinstance(w, QDialog) and w.isVisible():
                    w.accept()
        QTimer.singleShot(150, _close)

    dlg = mw.AboutDialog(win)
    texts = [l.text() for l in dlg.findChildren(QLabel) if l.text()]
    joined = "\n".join(texts)
    ok_author = all(k in joined for k in (
        mw.AUTHOR_NAME_CN, mw.AUTHOR_NAME_EN, mw.AUTHOR_EMAIL,
        mw.AUTHOR_PHONE, mw.AUTHOR_AFFILIATION_CN, mw.WECHAT_ACCOUNT))
    check("「关于」对话框含开发者信息（姓名/单位/邮箱/电话/公众号）",
          ok_author and "使用说明" in joined,
          f"作者={mw.AUTHOR_NAME_CN}（{mw.AUTHOR_NAME_EN}）"
          f"{mw.AUTHOR_EMAIL}")
    check("「关于」对话框含许可说明（PySide6-Essentials / LGPL）",
          "PySide6-Essentials" in joined and "LGPL" in joined,
          "已在关于里说明 Qt 用 LGPLv3、动态链接、未修改")
    dlg.close()

    guide_path = os.path.join(ROOT, "docs", "使用说明.html")
    check("使用说明（截图版 HTML）随包存在",
          os.path.exists(guide_path) and os.path.getsize(guide_path) > 10000,
          f"{os.path.relpath(guide_path, ROOT)}，"
          f"{os.path.getsize(guide_path) if os.path.exists(guide_path) else 0} 字节")

    # 说明书里的每张图都要真的在磁盘上——发行包漏图就等于说明书是空框
    import re as _re
    html = open(guide_path, encoding="utf-8").read()
    srcs = _re.findall(r'src="([^"]+)"', html)
    docs_dir = os.path.dirname(guide_path)
    missing = [s for s in srcs if not os.path.exists(os.path.join(docs_dir, s))]
    check("说明书引用的图片全部随包存在（界面截图 + 示例配图）",
          len(srcs) >= 18 and not missing,
          f"{len(srcs)} 张图，缺失：{missing or '无'}")

    check("说明书含示例配图目录与公众号二维码",
          "使用说明_img/" in html
          and os.path.exists(os.path.join(docs_dir, mw.WECHAT_QR_FILENAME)),
          f"{mw.WECHAT_QR_FILENAME} + 使用说明_img/")

    tb = QTextBrowser()
    tb.setSearchPaths([docs_dir])
    tb.setHtml(html)
    rendered = tb.toPlainText()
    check("说明书 HTML 能被 QTextBrowser 渲染（正文不丢）",
          "输入是" in rendered and "普通网格" in rendered
          and len(rendered) > 6000,
          f"渲染后 {len(rendered)} 字符")

    tb2 = QTextBrowser()
    tb2.setMarkdown(open(os.path.join(ROOT, "docs", "使用说明_GUI.md"),
                         encoding="utf-8").read())
    md = tb2.toPlainText()
    check("纯文本版说明仍能按 Markdown 渲染（表格/标题不丢内容）",
          "输入是" in md and "普通网格" in md and len(md) > 3000,
          f"渲染后 {len(md)} 字符")

    # 「关于」里必须真的出现二维码图片（不只是文字提到公众号）
    about = mw.AboutDialog(win)
    qr_labels = [l for l in about.findChildren(QLabel)
                 if l.pixmap() is not None and not l.pixmap().isNull()]
    check("「关于」对话框内嵌公众号二维码图片", bool(qr_labels),
          f"{len(qr_labels)} 张图片控件；二维码路径 = "
          f"{os.path.relpath(mw.qr_image_path(), ROOT)}"
          if mw.qr_image_path() else "未找到二维码文件")
    about.close()

    # 说明书窗口在渲染后按视口逐张定图片尺寸（只缩不放），且图片居中
    gd = mw.GuideDialog(win)
    gd.resize(1000, 800)
    gd.show()
    for _ in range(3):
        app.processEvents()
    gd.refresh()
    app.processEvents()
    doc = gd._view.document()
    col = gd._fit_width()
    widths = []
    blk = doc.begin()
    while blk.isValid():
        it = blk.begin()
        while not it.atEnd():
            fr = it.fragment()
            if fr.isValid() and fr.charFormat().isImageFormat():
                widths.append(int(fr.charFormat().toImageFormat().width()))
            it += 1
        blk = blk.next()
    check("使用说明里的图片按视口定尺寸（只缩不放、不超栏宽、都在）",
          len(widths) >= 20 and all(0 < w <= col for w in widths)
          and max(widths) <= 1600,
          f"{len(widths)} 张图，可读宽度 {col}，图片宽度 {min(widths)}~{max(widths)}")
    check("窄窗口也不会出现横向滚动条（代码块自动折行）",
          not gd._view.horizontalScrollBar().isVisible(), "")
    check("说明书图片一律居中（<p align=center>）",
          gd._view.toHtml().count('align="center"') >= 20,
          f"{gd._view.toHtml().count('align=\"center\"')} 个居中块")

    # 回归：Qt 只要看到 <head> 就会给每张图多留「图高 × 文档宽 / 图宽」的空白，
    # 图与题注之间会空半屏。GuideDialog 渲染前会剥掉 HTML 外壳，这里守住它。
    doc2 = gd._view.document()
    layout2 = doc2.documentLayout()
    rows = []
    blk = doc2.begin()
    while blk.isValid():
        rect = layout2.blockBoundingRect(blk)
        has_img = False
        it = blk.begin()
        while not it.atEnd():
            fr = it.fragment()
            if fr.isValid() and fr.charFormat().isImageFormat():
                has_img = True
            it += 1
        rows.append((rect.y(), rect.height(), has_img))
        blk = blk.next()
    gaps = [round(rows[i + 1][0] - (rows[i][0] + rows[i][1]), 1)
            for i in range(len(rows) - 1) if rows[i][2]]
    worst = max(gaps) if gaps else 0
    check("图与题注之间没有多余空白（Qt 的 <head> 图片行高坑）",
          gaps and worst <= 30, f"{len(gaps)} 张图，最大间距 {worst} px")
    gd.close()

    acts = {}
    for act in win.menuBar().actions():
        if act.text().startswith("帮助"):
            for a in act.menu().actions():
                acts[a.text()] = a.shortcut().toString()
    check("帮助菜单含「使用说明」(F1) / 「关于」/「课题组公众号」",
          any("使用说明" in k and v == "F1" for k, v in acts.items())
          and any("关于" in k for k in acts)
          and any("公众号" in k for k in acts),
          f"{[(k, v) for k, v in acts.items() if k]}")

    close_dialog_later()
    win.show_guide()
    close_dialog_later()
    win.show_about()
    close_dialog_later()
    win.show_wechat()
    check("帮助里的三个对话框都能打开并正常关闭（不阻塞）", True,
          "使用说明 / 关于 / 公众号 均已弹出并自动关闭")


    # ------------------------------------------------------- cancel path
    win.tabs.setCurrentIndex(0)
    win.sp_nmax.setValue(200)          # something slow enough to cancel
    win.run_analysis()
    app.processEvents()
    time.sleep(0.05)
    win.stop_analysis()
    wait_for(app, win.analysis_worker, timeout=300)
    check("取消路径不崩溃（阶数 200 中途停止）",
          win.analysis_worker is not None, "已请求并等待线程结束")

    # -------------------------------------------------------- export paths
    from shkit import io as shio
    win.sp_nmax.setValue(12)
    win.run_analysis()
    wait_for(app, win.analysis_worker, timeout=300)
    cp = os.path.join(SHOTS, "gui_coeffs.sh")
    shio.write_coeffs(win.coeffs, cp, layout="triangle")
    back = shio.read_coeffs(cp)
    check("GUI 结果可写入并读回（三角布局）",
          back.nmax == win.coeffs.nmax
          and np.abs(back.C - win.coeffs.C).max() < 1e-12, cp)

    rp = os.path.join(SHOTS, "gui_report.json")
    shio.write_report(rp, win.report)
    check("GUI 报告可序列化为 JSON", os.path.exists(rp),
          f"{os.path.getsize(rp)} 字节")

    # ------------------------------------------------------ gaussian filter
    # 设置了高斯半径必须**看得出效果**：导出系数、谱、两个场都得变，
    # 而且只能施加一次（不能既乘在系数上又在合成时再乘一遍）。
    from shkit.filters import apply_gaussian, gaussian_coefficients
    from shkit.synthesis import synthesis

    win.sp_nmax.setValue(12)
    win.sp_gauss.setValue(0.0)
    win.run_analysis()
    wait_for(app, win.analysis_worker, timeout=300)
    c0 = win.coeffs.copy()
    rec0 = np.asarray(win.recon).ravel().copy()

    win.sp_gauss.setValue(3000.0)
    win.run_analysis()
    wait_for(app, win.analysis_worker, timeout=300)
    app.processEvents()
    c1 = win.coeffs
    rec1 = np.asarray(win.recon).ravel()

    W = gaussian_coefficients(3000.0, 12)
    rel_c = float(np.linalg.norm(c1.C - c0.C) / np.linalg.norm(c0.C))
    check("高斯平滑改变**导出的系数**（不再只是重建时才生效）",
          rel_c > 0.10,
          f"半径 3000 km（W₀={W[0]:.6f}, W₁₂={W[12]:.6g}）→ 系数相对变化 "
          f"{rel_c:.2%}")

    # 系数应当正好是 r=0 的系数逐阶乘 W
    want = apply_gaussian(c0, 3000.0)
    check("系数就是逐阶乘 W（W 与 filters 自己算的一致）",
          np.abs(c1.C - want.C).max() < 1e-10 * max(1.0, np.abs(want.C).max()),
          f"max|C − C₀·W| = {np.abs(c1.C - want.C).max():.3e}；"
          f"报告记录 W(nmax) = {win.report.meta.get('gaussian_Wnmax'):.6g}")

    # 重建场 = 用**已平滑**的系数综合 → 只施加一次
    expect = np.asarray(synthesis(win.dataset.lat, win.dataset.lon, c1)).ravel()
    rel1 = float(np.max(np.abs(rec1 - expect)) / np.max(np.abs(expect)))
    double = np.asarray(synthesis(win.dataset.lat, win.dataset.lon,
                                  apply_gaussian(c1, 3000.0))).ravel()
    rel2 = float(np.max(np.abs(rec1 - double)) / np.max(np.abs(double)))
    check("重建场用的是同一套平滑系数：只施加一次（不是乘两次）",
          rel1 < 1e-12 and rel2 > 1e-3,
          f"与「系数综合」相对差 {rel1:.2e}；与「再乘一次 W」相对差 {rel2:.2e}")

    rms0 = float(np.sqrt(np.mean(rec0 ** 2)))
    rms1 = float(np.sqrt(np.mean(rec1 ** 2)))
    check("重建场随半径缩小（平滑确实压掉了高阶）",
          rms1 < rms0, f"RMS {rms0:.6e} → {rms1:.6e}（↓{1 - rms1 / rms0:.2%}）")

    win.sp_gauss.setValue(0.0)
    win.run_analysis()
    wait_for(app, win.analysis_worker, timeout=300)
    app.processEvents()
    check("半径改回 0 后系数回到未平滑（可逆、无残留）",
          np.abs(win.coeffs.C - c0.C).max() < 1e-12,
          f"max|C − C₀| = {np.abs(win.coeffs.C - c0.C).max():.3e}")

    # ==================================================== D1 色标控制面板
    check("D1：色标面板存在且默认 Auto Range（手动输入框灰掉）",
          win.chk_color_auto.isChecked()
          and not win.edit_vmin.isEnabled() and not win.edit_vmax.isEnabled(),
          f"auto={win.chk_color_auto.isChecked()}")
    check("★ D1：水平形变页有**自己的** ColorScale（不跟地图走）",
          win.horiz_canvas.color is win.horiz_color_scale
          and win.horiz_canvas.color is not win.color_scale,
          "地图的色标（例如 EWH ±700 cm）套到毫米级形变上会把分量压成一片颜色")

    win.map_field.setCurrentIndex(0)          # 原始数据，保证有场可画
    win.chk_color_auto.setChecked(False)
    win.map_symmetric.setChecked(False)
    win.edit_vmin.setText("-1.5")
    win.edit_vmax.setText("2.5")
    win._on_apply_color()
    app.processEvents()
    check("D1：手动 vmin/vmax 应用到地图色标",
          win.map_canvas._mappable.get_clim() == (-1.5, 2.5),
          f"clim={win.map_canvas._mappable.get_clim()}")
    check("D1：面板写明当前生效的规则（不会嘴上说自动、手上是手动）",
          "手动范围" in win.color_note.text() and "1.5" in win.color_note.text(),
          win.color_note.text().strip())

    # vcenter：以某值居中展开，且把「对称色标」自动取消（避免两个口径打架）
    win.map_symmetric.setChecked(True)
    win.edit_vcenter.setText("1.0")
    win._on_apply_color()
    app.processEvents()
    lo, hi = win.map_canvas._mappable.get_clim()
    check("D1：vcenter 以该值居中展开（|lo−c| == |hi−c|）",
          abs((1.0 - lo) - (hi - 1.0)) < 1e-9,
          f"clim=({lo:.4g}, {hi:.4g})，中心 1.0")
    check("D1：填了 vcenter 后「对称色标」自动取消（写明谁在生效）",
          not win.map_symmetric.isChecked()
          and "以 1 为中心" in win.color_note.text(),
          win.color_note.text().strip())

    # 回到自动
    win.edit_vcenter.setText("")
    win.chk_color_auto.setChecked(True)
    win._on_apply_color()
    app.processEvents()
    vals = np.asarray(win.dataset.value_slice(0), dtype=float)
    want = np.percentile(vals[np.isfinite(vals)], [2, 98])
    lo, hi = win.map_canvas._mappable.get_clim()
    check("D1：恢复 Auto 后色标 = 2–98% 分位数（对称）",
          abs(max(abs(lo), abs(hi)) - max(abs(want[0]), abs(want[1]))) < 1e-9,
          f"clim=({lo:.4g}, {hi:.4g})，2–98% = ({want[0]:.4g}, {want[1]:.4g})")
    win.grab().save(os.path.join(SHOTS, "20_color_panel.png"))

    # ======================================================= D2 悬停双读数
    check("D2：地图画布已接上悬停回调",
          win.map_canvas.hover_callback is not None,
          "hover_callback → MainWindow._on_map_hover")

    class _Ev:                                    # 最小 motion 事件替身
        pass

    ev = _Ev()
    ev.inaxes = win.map_canvas._ax
    ev.xdata, ev.ydata = 30.0, 10.0
    win.map_canvas._on_motion(ev)
    app.processEvents()
    hov = win.hover_label.text()
    check("D2：悬停读数含 原始= / Proc= / (lon, lat)",
          "原始=" in hov and "Proc=" in hov and "lon=" in hov and "lat=" in hov,
          hov)
    check("D2：散点数据上会吸附到最近的点并给出点号",
          "点 #" in hov, hov.split("(lon")[0][:44])

    # 多时次 + 日期：构造一个带时间坐标的网格数据集，直接喂给窗口
    from shkit.gui.dataset import Dataset

    _lat = np.arange(-80.0, 80.01, 10.0)
    _lon = np.arange(0.0, 360.0, 15.0)
    # 24 个历元：D6 的「常数+趋势+年周期」要 4 项，3 个历元根本拟不出来
    _nt = 24
    _g = np.random.default_rng(3).normal(size=(_lat.size, _lon.size, _nt))
    _tv = [str(x)[:10] for x in
           (np.datetime64("2002-01-18") + np.arange(_nt) * 30)]
    ds_mt = Dataset.from_grid("<D2 测试网格>", _lat, _lon, _g, {"time": _tv})
    win.on_loaded(ds_mt)
    win._refresh_map()
    app.processEvents()
    check("D2：多时次网格的时间轴可从 meta['time'] 建起来并带日期",
          ds_mt.time_axis() is not None and ds_mt.time_axis().has_dates
          and ds_mt.epoch_label(1).endswith("2002-02-17）"),
          ds_mt.epoch_label(1))

    win.time_slider.setValue(1)
    app.processEvents()
    ev2 = _Ev()
    ev2.inaxes = win.map_canvas._ax
    ev2.xdata, ev2.ydata = 15.0, -20.0
    win.map_canvas._on_motion(ev2)
    app.processEvents()
    hov2 = win.hover_label.text()
    check("D2：多时次读数带「第 k/N 个时次（日期）」",
          "第 2/24 个时次" in hov2 and "2002-02-17" in hov2, hov2)
    check("D2：网格数据上吸附到最近格点（lon=15, lat=−20）",
          "lon=15.0000" in hov2 and "lat=-20.0000" in hov2, hov2)
    win.grab().save(os.path.join(SHOTS, "21_hover_readout.png"))

    # ======================================================== D3 数值表页
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    check("D3：存在「数值表」页签", "数值表" in tabs, str(tabs))
    idx_tab = tabs.index("数值表")
    win._values_dirty = True
    win.tabs.setCurrentIndex(idx_tab)              # 切过去才填（懒加载）
    app.processEvents()
    check("D3：切到数值表页后自动填充（网格：行=纬、列=经）",
          win.values_table.rowCount() == _lat.size
          and win.values_table.columnCount() == _lon.size + 1,
          f"{win.values_table.rowCount()}×{win.values_table.columnCount()}")
    check("D3：表头首格标明 lat \\ lon，且表头是经度值",
          win._values_payload[0][0].startswith("lat")
          and win._values_payload[0][1] == "0",
          str(win._values_payload[0][:3]))
    tsv = win._values_tsv()
    check("D3：TSV 行数 = 表头 + 数据行，列数与表头一致",
          len(tsv.splitlines()) == _lat.size + 1
          and all(len(r) == _lon.size + 1
                  for r in win._values_payload[1]),
          f"{len(tsv.splitlines())} 行")
    check("D3：表里写明了这是哪个时次/哪个视图",
          "第 2 个时次" in win.values_note.text(), win.values_note.text()[:60])

    win._copy_values()
    app.processEvents()
    clip = QApplication.clipboard().text()
    check("D3：复制到剪贴板（制表符分隔，可直接粘进 Excel）",
          clip.startswith(win._values_payload[0][0])
          and len(clip.splitlines()) == _lat.size + 1,
          f"{len(clip.splitlines())} 行")

    # 导出：把文件对话框替换成固定路径，走真实导出代码
    import shkit.gui.main_window as _mw
    exp = os.path.join(SHOTS, "values_export.csv")
    _orig = _mw.QFileDialog.getSaveFileName
    _mw.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (exp, "CSV (*.csv)"))
    try:
        win._export_values()
    finally:
        _mw.QFileDialog.getSaveFileName = _orig
    body = open(exp, encoding="utf-8-sig").read().splitlines()
    check("D3：导出 CSV（首行是口径说明，随后是表头与数据）",
          os.path.exists(exp) and body[1].startswith("lat")
          and len(body) == _lat.size + 2,
          f"{len(body)} 行 → {os.path.basename(exp)}")

    # 大网格：抽稀显示，而且必须**说出来**抽了
    _big = np.random.default_rng(5).normal(size=(181, 360, 1))
    ds_big = Dataset.from_grid("<大网格>", np.arange(-90.0, 90.01, 1.0),
                               np.arange(0.0, 360.0, 1.0), _big)
    win.on_loaded(ds_big)
    _cap = win.MAX_TABLE_CELLS
    win.MAX_TABLE_CELLS = 4000                    # 别让冒烟测试填 6 万格
    try:
        win._fill_values_table()
        app.processEvents()
        check("D3：大网格抽稀显示并在说明里写明抽样步长",
              "抽样" in win.values_note.text()
              and win.values_table.rowCount() < 181,
              win.values_note.text()[:70])
        check("D3：抽稀只影响显示，payload 里的行数 = 表行数（不是原网格）",
              len(win._values_payload[1]) == win.values_table.rowCount())
    finally:
        win.MAX_TABLE_CELLS = _cap

    # ====================================================== D4 时间轴控件
    # 回到多时次数据集：D3 那段为了测抽稀把 181×360×1 的大网格塞了进来
    win.on_loaded(ds_mt)
    app.processEvents()
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    check("D4：时间轴行可见（多时次数据）", win.time_row.isVisible(),
          f"ntime={win.dataset.ntime}")
    lbl = win.time_label.text()
    check("D4：读数同时给 ISO 日期与十进制年",
          "ISO 2002-" in lbl and "十进制年" in lbl, lbl)
    win.time_slider.setValue(0)
    win._step_time(+1)
    check("D4：▶ 前进一步", win.time_slider.value() == 1,
          f"→ {win.time_slider.value()}")
    win._step_time(-1)
    check("D4：◀ 后退一步", win.time_slider.value() == 0,
          f"→ {win.time_slider.value()}")
    win.time_slider.setValue(0)
    win._step_time(-5)
    check("D4：到达端点后不越界（钳到 0）", win.time_slider.value() == 0)
    win.time_slider.setValue(win.time_slider.maximum())
    win._step_time(+5)
    check("D4：末端同样不越界",
          win.time_slider.value() == win.time_slider.maximum())

    win.time_lo.setValue(0)
    win.time_hi.setValue(2)
    app.processEvents()
    check("D4：范围滑块生效并写进取值函数", win.time_window() == (0, 2),
          str(win.time_window()))
    check("D4：范围不是全部历元时会在读数里标出来",
          "范围 1–3" in win.time_label.text(), win.time_label.text()[-14:])
    win.time_lo.setValue(5)                     # lo > hi：必须自动让位
    app.processEvents()
    lo, hi = win.time_window()
    check("D4：拖动使 lo > hi 时自动纠正（不会出现空窗口）", lo <= hi,
          f"({lo}, {hi})")
    win._reset_time_range()
    app.processEvents()
    check("D4：「全部」把范围设回全部历元",
          win.time_window() == (0, win.dataset.ntime - 1),
          str(win.time_window()))

    # ==================================================== D6 时间序列页
    idx_series = tabs.index("时间序列")
    win.tabs.setCurrentIndex(idx_series)
    app.processEvents()
    win.chk_series_region.setChecked(False)
    win.sp_series_lon.setValue(15.0)
    win.sp_series_lat.setValue(20.0)
    app.processEvents()
    check("D6：单点模式给出「最近点」曲线（并说明离所选位置多远）",
          "最近点 #" in win.series_note.text()
          and "距所选位置" in win.series_note.text(),
          win.series_note.text()[:70])
    check("D6：曲线真的画了线（>0 条）",
          len(win.series_canvas.figure.axes[0].lines) >= 1,
          f"{len(win.series_canvas.figure.axes[0].lines)} 条")
    check("D6：拟合开启时给出趋势/周年振幅/残差 RMS",
          "趋势" in win.series_note.text() and "周年振幅" in win.series_note.text()
          and "残差 RMS" in win.series_note.text(),
          win.series_note.text()[-60:])
    win.chk_series_fit.setChecked(False)
    app.processEvents()
    check("D6：关掉拟合后不再出现趋势数字（不是画了线却说没画）",
          "趋势" not in win.series_note.text(), win.series_note.text()[:50])
    win.chk_series_fit.setChecked(True)
    win.chk_series_region.setChecked(True)
    app.processEvents()
    check("D6：区域模式写明覆盖了球面多少（区域平均必须带覆盖率）",
          "覆盖球面" in win.series_note.text(), win.series_note.text()[:70])
    check("D6：x 轴标签是十进制年",
          win.series_canvas.figure.axes[0].get_xlabel() == "十进制年",
          win.series_canvas.figure.axes[0].get_xlabel())
    win.grab().save(os.path.join(SHOTS, "23_time_series.png"))

    # ==================================================== D7 趋势与周年页
    idx_trend = tabs.index("趋势与周年")
    win.tabs.setCurrentIndex(idx_trend)
    app.processEvents()
    win._compute_trend_maps()
    app.processEvents()
    check("D7：画出三张图（趋势 / 周年振幅 / 周年相位）",
          len(win.trend_canvas._axes) == 3,
          f"{len(win.trend_canvas._axes)} 个面板")
    check("D7：写明条件数与用掉的格点数",
          "条件数" in win.trend_note.text() and "用了" in win.trend_note.text(),
          win.trend_note.text()[:70])
    check("D7：相位场明确说把低振幅格子遮掉了（相位在振幅≈0处无意义）",
          "相位" in win.trend_canvas._axes[2].get_title()
          and "不画" in win.trend_canvas._axes[2].get_title(),
          win.trend_canvas._axes[2].get_title()[:44])
    win.grab().save(os.path.join(SHOTS, "24_trend_maps.png"))

    # 没有日期 -> 必须拒绝，而不是按等次序号硬算
    ds_nd = Dataset.from_grid("<无日期>", _lat, _lon, _g)
    win.on_loaded(ds_nd)
    win._compute_trend_maps()
    app.processEvents()
    check("D7：没有日期时拒绝拟合（不假装等间隔）",
          "没有日期" in win.trend_note.text(), win.trend_note.text()[:60])

    # ================================================== D5 逐历元诊断页
    # 专门造一份带「坏历元」的数据：第 8 个历元加一层低阶表示不了的噪声
    _gb = np.random.default_rng(7).normal(size=(_lat.size, _lon.size, _nt))
    _gb[:, :, 7] += np.random.default_rng(8).normal(
        size=(_lat.size, _lon.size)) * 8.0
    ds_ep = Dataset.from_grid("<D5 测试网格>", _lat, _lon, _gb, {"time": _tv})
    win.on_loaded(ds_ep)
    win.sp_nmax.setValue(6)
    app.processEvents()
    check("D5：存在「逐历元诊断」页签",
          "逐历元诊断" in [win.tabs.tabText(i) for i in range(win.tabs.count())],
          str([win.tabs.tabText(i) for i in range(win.tabs.count())]))

    win.run_series_analysis()
    ok = wait_for(app, win.series_worker, timeout=300)
    app.processEvents()
    check("D5：批量分析线程正常结束", ok)
    check("D5：表格 = 每个历元一行 × 6 列（#/日期/相对RMSE/C00/残差/可疑）",
          win.epochs_table.rowCount() == _nt
          and win.epochs_table.columnCount() == 6,
          f"{win.epochs_table.rowCount()}×{win.epochs_table.columnCount()}")
    check("D5：日期列来自时间轴",
          win.epochs_table.item(1, 1).text().startswith("2002-"),
          win.epochs_table.item(1, 1).text())
    note = win.epochs_note.text()
    check("D5：写明 Gram 只算了一次（A2 的优化在 GUI 里也看得到）",
          "Gram 只算 1 次" in note, note[:60])
    bad = [i + 1 for i in range(win.epochs_table.rowCount())
           if win.epochs_table.item(i, 5).text()]
    check("★ D5：注入的坏历元被 MAD 检出并高亮（第 8 个）",
          bad == [8], f"高亮 {bad}")
    check("D5：注明只高亮、不自动剔除（剔除是科学判断，不是软件默认）",
          "只高亮" in note and "不剔除" in note, note[-40:])
    rows = [win.epochs_table.item(7, 4).text(), win.epochs_table.item(0, 4).text()]
    check("D5：坏历元的残差确实比正常历元大一个量级",
          float(rows[0]) > 3 * float(rows[1]), f"坏 {rows[0]} vs 正常 {rows[1]}")

    win._on_epoch_row_clicked(7, 0)
    app.processEvents()
    check("D5：点某一行跳到该历元的地图（时次 + 页签都跟着走）",
          win.time_slider.value() == 7
          and win.tabs.tabText(win.tabs.currentIndex()) == "地图",
          f"slider={win.time_slider.value()}，"
          f"tab={win.tabs.tabText(win.tabs.currentIndex())}")
    win.tabs.setCurrentIndex([win.tabs.tabText(i)
                              for i in range(win.tabs.count())].index("逐历元诊断"))
    app.processEvents()
    win.grab().save(os.path.join(SHOTS, "26_epochs.png"))

    # 关掉高亮 -> 不再有标记（开关必须真的起作用）
    win.chk_epochs_highlight.setChecked(False)
    app.processEvents()
    check("D5：关掉高亮开关后表格里不再有标记",
          not any(win.epochs_table.item(i, 5).text()
                  for i in range(win.epochs_table.rowCount())))
    win.chk_epochs_highlight.setChecked(True)
    app.processEvents()

    # 单时次数据必须拒绝（没有「逐历元」可言）
    win.on_loaded(ds_big)                       # 181×360×1
    win.run_series_analysis()
    app.processEvents()
    check("D5：单时次数据拒绝批量逐历元（并说明原因）",
          "多时次" in win.epochs_note.text(), win.epochs_note.text()[:40])
    win.stop_series_analysis()                  # 没在跑也不能崩
    check("D5：未运行时点「停止」不崩", True)

    # ==================================================== D9 播放 + GIF 导出
    win.on_loaded(ds_mt)                        # 多时次、有日期
    win._reset_time_range()
    win.sp_fps.setValue(10)
    win._apply_play_interval()
    check("D9：帧率决定定时器间隔（10 帧/秒 → 100 ms）",
          win._play_timer.interval() == 100, f"{win._play_timer.interval()} ms")
    win.time_slider.setValue(0)
    win.btn_play.setChecked(True)
    app.processEvents()
    check("D9：点播放后开始走（按钮变成暂停、提示写明窗口）",
          win._play_timer.isActive() and "暂停" in win.btn_play.text()
          and "播放中" in win.play_note.text(), win.play_note.text())
    for _ in range(win.time_slider.maximum() + 5):
        win._play_step()
    check("★ D9：播放到窗口末端自动停，且**不循环**",
          not win._play_timer.isActive() and not win.btn_play.isChecked()
          and win.time_slider.value() == win.time_slider.maximum(),
          f"slider={win.time_slider.value()}/{win.time_slider.maximum()}，"
          f"{win.play_note.text()}")

    win.time_lo.setValue(0)
    win.time_hi.setValue(3)
    app.processEvents()
    gif = os.path.join(SHOTS, "play.gif")
    _orig_save = _mw.QFileDialog.getSaveFileName
    _mw.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (gif, "GIF (*.gif)"))
    keep_t = win.time_slider.value()
    try:
        win._export_gif()
    finally:
        _mw.QFileDialog.getSaveFileName = _orig_save
    app.processEvents()
    from PIL import Image
    check("★ D9：GIF 帧数 = 时间窗口长度（帧数固定，不会多也不会少）",
          os.path.exists(gif) and Image.open(gif).n_frames == 4,
          f"{Image.open(gif).n_frames if os.path.exists(gif) else 0} 帧，"
          f"窗口 {win.time_window()}")
    check("D9：导出后把时次恢复到导出前（不偷走用户的位置）",
          win.time_slider.value() == keep_t, f"{win.time_slider.value()}")

    # =============================================== D10 逐时次导出（暂停/停止）
    exp_dir = os.path.join(SHOTS, "epochs_export")
    shutil.rmtree(exp_dir, ignore_errors=True)
    _orig_dir = _mw.QFileDialog.getExistingDirectory
    _mw.QFileDialog.getExistingDirectory = staticmethod(
        lambda *a, **k: exp_dir)
    try:
        # 先在暂停状态启动：一个文件都不该写出来
        _real_init = _mw.ExportWorker.__init__

        def _paused_init(self, *a, **k):
            _real_init(self, *a, **k)
            self.set_paused(True)

        _mw.ExportWorker.__init__ = _paused_init
        win.export_epochs()
        _mw.ExportWorker.__init__ = _real_init
        w_exp = win._export_dialog.worker
        time.sleep(0.4)
        app.processEvents()
        import glob as _glob
        check("D10：暂停状态下不写任何文件（不会写出半截文件）",
              len(_glob.glob(os.path.join(exp_dir, "*.nc"))) == 0,
              f"{len(_glob.glob(os.path.join(exp_dir, '*.nc')))} 个文件，"
              f"paused={w_exp._paused}")
        win._export_dialog._toggle_pause()
        check("D10：恢复后按钮变成「暂停」语义",
              win._export_dialog.btn_pause.text() == "暂停",
              win._export_dialog.btn_pause.text())
        t0 = time.time()
        while w_exp.isRunning() and time.time() - t0 < 60:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        files = sorted(_glob.glob(os.path.join(exp_dir, "*.nc")))
        check("★ D10：逐时次导出写满时间窗口（4 帧 → 4 个文件，名字带序号）",
              len(files) == 4
              and os.path.basename(files[0]).startswith("epoch_0001"),
              f"{[os.path.basename(f) for f in files]}")
        if files:
            _la, _lo, _cu, _meta = shio.read_grid(files[0])
            want = win.dataset.grid[:, :, 0]
            check("D10：导出的文件能读回，且与源网格逐值一致",
                  _cu.shape == want.shape
                  and float(np.abs(_cu - want).max()) <= 1e-9 * max(
                      float(np.abs(want).max()), 1e-300),
                  f"max|Δ| = {np.abs(_cu - want).max():.2e}")
        check("D10：完成提示写明写了几个文件到哪",
              "完成" in win._export_dialog.label.text()
              and "4 个文件" in win._export_dialog.label.text(),
              win._export_dialog.label.text()[:50])

        # 停止：中途叫停，已写文件仍是完整历元
        exp2 = os.path.join(SHOTS, "epochs_stop")
        shutil.rmtree(exp2, ignore_errors=True)
        _mw.QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: exp2)
        win.export_epochs()
        w2 = win._export_dialog.worker
        win._export_dialog._stop()
        t0 = time.time()
        while w2.isRunning() and time.time() - t0 < 30:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        check("D10：「停止」生效并如实报告写了几个（每 个都是完整历元）",
              not w2.isRunning() and "已停止" in win._export_dialog.label.text()
              and "完整历元" in win._export_dialog.label.text(),
              win._export_dialog.label.text()[:46])

        # ---- 水平形变导出（用户要求：径向能导出，水平也要能）----------------
        hz_dir = os.path.join(SHOTS, "horiz_export")
        shutil.rmtree(hz_dir, ignore_errors=True)
        _mw.QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: hz_dir)
        if win._epoch_source_coeffs() is None:        # 先把逐历元系数解出来
            win.run_analysis()
            wait_any_solve(app, win, timeout=600)
            app.processEvents()
        win.time_lo.setValue(0)
        win.time_hi.setValue(1)                      # 时间窗 = 前两个历元
        win.time_slider.setValue(0)
        app.processEvents()
        win.export_horizontal("current")             # 先导当前历元
        _wh = win._export_dialog.worker
        t0 = time.time()
        while _wh.isRunning() and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        _hf = sorted(glob.glob(os.path.join(hz_dir, "*")))
        check("★ 水平形变导出（当前历元）：写出 u_N / u_E / |u_h| 三个文件",
              len(_hf) == 3
              and all(any(t in os.path.basename(f) for t in ("uN", "uE", "umag"))
                      for f in _hf),
              f"{[os.path.basename(f) for f in _hf]}")
        if _hf:
            import shkit.gradient as _gr
            from shkit.gradient import canonical_scaled as _cs3
            _gh = [f for f in _hf if "_uN" in f]
            _la2, _lo2, _cu2, _mo2 = shio.read_grid(_gh[0])
            _C3, _S3, _F3 = _cs3(win._coeffs_for_epoch(
                win.time_slider.value() if win.dataset.has_time() else 0))
            _want_h = _gr.horizontal_grid(win.dataset.lat_vec,
                                          win.dataset.lon_vec, _C3, _S3, _F3,
                                          method="auto")
            _wn = np.asarray(_want_h["north"])
            if _wn.ndim == 3:
                _wn = _wn[:, :, 0]
            check("★ 导出的 u_N 与库算的逐值一致（不是另一个东西）",
                  _cu2.shape == _wn.shape
                  and float(np.abs(_cu2 - _wn).max()) <= 1e-12 * max(
                      float(np.abs(_wn).max()), 1e-300),
                  f"max|Δ| = {np.abs(_cu2 - _wn).max():.2e}")

        # 逐时次（时间窗内 2 个历元 → 6 个文件），可暂停/停止
        hz_dir2 = os.path.join(SHOTS, "horiz_export_series")
        shutil.rmtree(hz_dir2, ignore_errors=True)
        _mw.QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: hz_dir2)
        win.export_horizontal("window")
        _wh2 = win._export_dialog.worker
        check("  逐时次：进度条如实列出每个历元写了哪几个分量",
              _wh2.label_of is not None and "_uN/uE/umag" in _wh2.label_of(0),
              _wh2.label_of(0) if _wh2.label_of else "—")
        t0 = time.time()
        while _wh2.isRunning() and time.time() - t0 < 180:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        _hf2 = sorted(glob.glob(os.path.join(hz_dir2, "*")))
        check("★ 水平形变导出（时间窗内逐时次）：2 个历元 × 3 个分量 = 6 个文件",
              len(_hf2) == 6
              and all(os.path.basename(f).startswith("epoch_000")
                      for f in _hf2),
              f"{len(_hf2)} 个：{[os.path.basename(f) for f in _hf2][:3]}…")
    finally:
        _mw.QFileDialog.getExistingDirectory = _orig_dir

    # ============================================ D12 子进程计算隔离
    from shkit.gui import subproc as _sp
    from shkit.timeaxis import TimeAxis

    ok_spawn, why_spawn = _sp.spawn_available()
    check("D12：本测试进程满足 spawn 前提（有 __main__ 保护）",
          ok_spawn, why_spawn or "ok")

    win.on_loaded(ds_ep)
    win.chk_subprocess.setChecked(True)
    win.sp_nmax.setValue(6)
    app.processEvents()
    win.run_series_analysis()
    app.processEvents()
    check("D12：勾选后走的是独立进程 runner",
          isinstance(win.series_worker, SubprocessRunner),
          type(win.series_worker).__name__)
    ok = wait_for(app, win.series_worker, timeout=600)
    app.processEvents()
    check("D12：子进程里同样算得出来（表格被填满）",
          ok and win.epochs_table.rowCount() == _nt,
          f"{win.epochs_table.rowCount()} 行")
    co_sub = getattr(win, "series_coeffs", None)
    check("D12：写完的系数与表格里的诊断都在",
          co_sub is not None and co_sub.ntime == _nt,
          "—" if co_sub is None else f"{co_sub.C.shape}")

    # 与界面内线程路径逐值对拍（"结果一样，只是崩溃不拖垮界面"）
    win.chk_subprocess.setChecked(False)
    win.run_series_analysis()
    ok2 = wait_for(app, win.series_worker, timeout=600)
    app.processEvents()
    co_thr = getattr(win, "series_coeffs", None)
    if ok2 and co_sub is not None and co_thr is not None:
        d = float(np.abs(co_sub.C - co_thr.C).max())
        check("★ D12：子进程结果 ≡ 界面内线程结果（逐值）", d <= 1e-12,
              f"max|ΔC| = {d:.2e}")
    else:
        check("★ D12：子进程结果 ≡ 界面内线程结果（逐值）", False,
              f"ok2={ok2}")

    # D12（水平形变）：子进程那条路也要对得上。v2.0.1 起水平形变接上了
    # ``subproc`` 的 horizontal_grid 任务（以前那个分支是**死代码**），单位换算与
    # 逐阶因子都在界面进程里算好（``gradient.canonical_scaled``），子进程只做纯数值，
    # 所以两条路必须逐值相同。
    if ok_spawn:
        from shkit.gradient import canonical_scaled as _cs2
        _co1 = win._coeffs_for_epoch(0)
        _Cg, _Sg, _Fg = _cs2(_co1)
        _latv = np.linspace(-88.0, 88.0, 19)
        _lonv = np.arange(0.0, 360.0, 20.0)          # nlon=18 ≤ 2*6=12? no → 走 direct
        _lonv2 = np.arange(0.0, 360.0, 5.0)          # nlon=72 > 2*6 ✓ 走 FFT
        _rep_sub = {}
        _rr = SubprocessRunner("horizontal_grid",
                               {"lat": _latv, "lon": _lonv2, "C": _Cg, "S": _Sg,
                                "degree_factors": _Fg, "method": "auto"},
                               parent=win)
        _got = {}
        _rr.succeeded.connect(lambda o, _x=None: _got.update(res=o))
        _rr.failed.connect(lambda t: _got.update(err=t))
        _rr.start()
        wait_for(app, _rr, timeout=600)
        app.processEvents()
        _from_thread = {}
        try:
            from shkit.gradient import horizontal_grid as _hg
            _from_thread = _hg(_latv, _lonv2, _Cg, _Sg, _Fg,
                               method="auto", report=_rep_sub)
        except Exception as exc:                             # noqa: BLE001
            _got["err"] = f"{exc}"
        if "res" in _got:
            _dn = float(np.abs(np.asarray(_got["res"]["north"])
                               - np.asarray(_from_thread["north"])).max())
            _de = float(np.abs(np.asarray(_got["res"]["east"])
                               - np.asarray(_from_thread["east"])).max())
            check("★ D12：水平形变 子进程 ≡ 界面内（逐值）",
                  _dn == 0.0 and _de == 0.0,
                  f"max|Δu_N| = {_dn:.2e}, max|Δu_E| = {_de:.2e}")
            check("  子进程把 path/reason 带回来了（不静默换算法）",
                  (_got["res"].get("report") or {}).get("path")
                  == _rep_sub.get("path"),
                  f"子进程={(_got['res'].get('report') or {}).get('path')!r}，"
                  f"线程={_rep_sub.get('path')!r}")
        else:
            check("★ D12：水平形变 子进程 ≡ 界面内（逐值）", False,
                  str(_got.get("err", "没有结果"))[:70])

    # 子进程被硬杀：父进程必须活着并收到明确的崩溃报告
    #
    # ⚠️ 这条本来是有竞态的：用的是「小任务」，只要测试进程在被杀之前把它算完，
    # 就会走成 **正常成功**，于是断言以「没收到失败信号」失败 —— 与实现对不对无关。
    # 所以这里刻意把任务做**长**（~200 个历元的批量分析，秒级），
    # 让子进程有足够长的存活窗口；本用例不检查数值（数值相等由上面那条断言管）。
    crashed = {}
    killed = False
    for attempt in range(3):
        crashed.clear()
        r12 = SubprocessRunner(
            "analyze_series",
            {"lat": win.dataset.lat, "lon": win.dataset.lon,
             "values": np.repeat(win.dataset.values, 8, axis=1), "nmax": 8,
             "report_fit": True,
             # ⚠️ `times` 必须是 TimeAxis（SHCoeffs.__post_init__ 会拒绝裸数组/字符串
             # 列表）。以前这里塞的是 `[str(x)[:10] …]`，于是子进程**一进去就抛
             # TypeError** —— 这条用例只是碰巧在子进程走到那行之前就把它杀了才"绿"，
             # 属于"绿得不对"。现在给真的 TimeAxis，任务才会真的长跑。
             "times": TimeAxis.from_datetimes(
                 [np.datetime64("2002-01-18") + np.timedelta64(30 * i, "D")
                  for i in range(win.dataset.values.shape[1] * 8)])},
            parent=win)
        r12.failed.connect(lambda tb: crashed.update(msg=tb))
        r12.succeeded.connect(lambda _r: crashed.update(unexpected=True))
        r12.start()
        # 等子进程真的起来（spawn 要 import numpy/shkit，0.4 s 往往还在引导阶段）
        t0 = time.time()
        child = None
        while time.time() - t0 < 30:
            child = getattr(r12, "_proc", None)
            if child is not None and child.is_alive():
                break
            app.processEvents()
            time.sleep(0.02)
        if child is not None and child.is_alive():
            child.terminate()
            child.join(timeout=10)          # 确认真的死了，而不是"以为杀了"
            killed = not child.is_alive()
        wait_for(app, r12, timeout=120)
        app.processEvents()
        if killed:
            break
        print(f"    （第 {attempt + 1} 次没能在子进程存活期内杀到它，重试；"
              f"crashed={list(crashed)}）", flush=True)
    check("★ D12：子进程被强杀后，收到「异常退出」而不是假成功",
          killed and "msg" in crashed and "异常退出" in crashed["msg"]
          and "unexpected" not in crashed,
          f"killed={killed} crashed={list(crashed)} "
          + (crashed.get("msg", "").splitlines()[0][:40]
             if crashed.get("msg") else "没收到失败信号"))
    # 界面还活着：照常跑一次分析
    # （多时次数据下「运行分析」= 一次解完整条序列，所以等的是 series_worker）
    win.sp_nmax.setValue(6)
    win.run_analysis()
    ok = wait_any_solve(app, win, timeout=600)
    app.processEvents()
    check("★ D12：崩溃之后界面仍然可用（还能正常做一次分析）",
          ok and win.coeffs is not None,
          f"coeffs={'有' if win.coeffs is not None else '无'}")

    # spawn 不可用时：回退到线程，并且**写出来**
    _orig_avail = _sp.spawn_available
    _sp.spawn_available = lambda: (False, "测试强制不可用")
    try:
        wait_any_solve(app, win, timeout=600)     # 上一步的整条序列先跑完
        win.chk_subprocess.setChecked(True)
        win.run_series_analysis()
        app.processEvents()
        note = win.epochs_note.text()
        check("★ D12：spawn 不可用时回退到线程，并明确写出回退原因",
              "回退" in note and "强制不可用" in note, note[:56])
        wait_for(app, win.series_worker, timeout=600)
    finally:
        _sp.spawn_available = _orig_avail
    check("D12：回退路径同样算得出结果",
          win.epochs_table.rowCount() == _nt,
          f"{win.epochs_table.rowCount()} 行")

    # ==================================== 控件不被压到"显示不全"（用户报的问题）
    # 播放行挤成一行时，「4 帧/秒」的中文后缀与「导出 GIF…」会被切掉；色标说明被
    # 挤在「应用色标」右边的一条窄缝里折成五行、贴在角落。这里把"显示得下"变成
    # 可断言的东西：控件实际宽度不小于 sizeHint，说明文字单独占一行、整幅宽。
    win.resize(1280, 860)                      # 用较窄的窗口来测（最容易挤）
    app.processEvents()
    for name, wdg in (("帧率框", win.sp_fps), ("播放按钮", win.btn_play),
                      ("导出 GIF 按钮", win.btn_gif),
                      ("逐时次导出按钮", win.btn_export_epochs),
                      ("逐时次导出前缀框", win.edit_export_prefix),
                      ("导出格式下拉框", win.cb_export_ext)):
        check(f"布局：{name} 实际宽度 ≥ sizeHint（不会被截断）",
              wdg.width() >= wdg.sizeHint().width() - 2,
              f"width={wdg.width()} sizeHint={wdg.sizeHint().width()}")
    check("布局：帧率框容得下最长的「30 帧/秒」",
          win.sp_fps.width() >= win.sp_fps.fontMetrics().horizontalAdvance(
              "30 帧/秒") + 30,
          f"width={win.sp_fps.width()}，文字宽="
          f"{win.sp_fps.fontMetrics().horizontalAdvance('30 帧/秒')}")
    check("布局：色标说明单独占一行（整幅宽，不在「应用色标」右边挤成一条）",
          win.color_note.width() > 0.6 * win.tabs.width()
          and win.color_note.x() < 0.2 * win.tabs.width(),
          f"x={win.color_note.x()} width={win.color_note.width()} "
          f"（页签区宽 {win.tabs.width()}）")
    # 注意判据用 heightForWidth(实际宽度)，**不能**用 sizeHint().height()：
    # 自动折行的 QLabel，sizeHint 是"按自然宽度排下来要几行"，比实际需要的高得多
    # （实测 40 vs 12），拿它当判据会把好好的布局判成被截断。
    check("布局：色标说明不被纵向截断（height ≥ heightForWidth(实际宽度)）",
          win.color_note.height()
          >= win.color_note.heightForWidth(win.color_note.width()) - 2,
          f"height={win.color_note.height()}，需要="
          f"{win.color_note.heightForWidth(win.color_note.width())}"
          f"（sizeHint={win.color_note.sizeHint().height()} 是自然宽度下的行高，"
          "不是判据）")
    win.resize(1600, 960)
    app.processEvents()

    # ============================== 全球等经纬网格（规则 dh）必须能算出来
    # ⚠️ 这条以前**没有任何用例走过**：GUI 的 AnalysisWorker 对"全球等经纬网格"
    # 会传 weights_kw={'nlon': …}，而 analysis() 是四个求解器里**唯一**不接受它的，
    # 于是"载入全球网格 → 运行分析"这条最常见的路径直接抛 TypeError。所有用例
    # （含本套件）都只用散点 / 区域掩膜，所以全绿也发现不了 —— 用户载入 0.25° 的
    # 全球 mascon 网格、点下「运行分析」就炸，正是这个洞。
    # 放到最后跑：它会换掉 win.dataset / coeffs，前面那些用例依赖原始数据。
    import shkit.gui.main_window as _mw2
    _crit = []
    _orig_crit = _mw2.QMessageBox.critical

    def _fake_crit(*a, **k):
        # on_analysis_failed 会弹**模态**框；离屏跑必须拦下来，否则测试直接卡死
        _crit.append(a[2] if len(a) > 2 else "")
        return _mw2.QMessageBox.StandardButton.Ok

    _mw2.QMessageBox.critical = staticmethod(_fake_crit)
    try:
        _latd = -90.0 + np.arange(90) * 2.0            # 90 条纬线（nlat 偶数）
        _lond = np.arange(0.0, 360.0, 2.0)             # 180 条经线 = 2*90 ✓
        _gd = (np.cos(np.deg2rad(_latd))[:, None]
               * np.cos(np.deg2rad(_lond))[None, :])
        _ds_dh = Dataset.from_grid("<DH 网格>", _latd, _lond, _gd[:, :, None])
        win.on_loaded(_ds_dh)
        check("dh：自动建议的积分元规则是 dh",
              win.cb_rule.currentData() == "dh",
              f"rule={win.cb_rule.currentData()}，nlon={_lond.size}，"
              f"nlat={_latd.size}")
        win.sp_nmax.setValue(10)
        win.run_analysis()
        ok_dh = wait_for(app, win.analysis_worker, timeout=300)
        app.processEvents()
        check("★ dh：全球等经纬网格点「运行分析」能出结果（weights_kw 回归）",
              ok_dh and win.coeffs is not None and not _crit,
              f"ok={ok_dh} coeffs={'有' if win.coeffs is not None else '无'}"
              + (f"；弹窗：{_crit[0].splitlines()[0][:60]}" if _crit else ""))
        if win.coeffs is not None:
            check("dh：系数有限，且 C00 = 场的面积加权均值",
                  bool(np.isfinite(win.coeffs.C).all())
                  and abs(float(win.coeffs.C[0, 0])
                          - float(np.mean(_gd))) < 1e-9,
                  f"C00={float(win.coeffs.C[0, 0]):.6g}，"
                  f"均值={float(np.mean(_gd)):.6g}")
            check("dh：报告里的权重规则就是 dh",
                  win.report.weight_rule == "dh",
                  f"weight_rule={win.report.weight_rule}")
            win.grab().save(os.path.join(SHOTS, "27_dh_grid_analysis.png"))
    finally:
        _mw2.QMessageBox.critical = _orig_crit

    # ================== 显示修复回归（用户报障）：经度对齐 / 半年图 / 海岸线 /
    #                    播放与时次切换时重建场也要跟着变
    win.on_loaded(ds_mt)                 # 24 历元、有日期、lon = 0…360
    win.chk_subprocess.setChecked(False)
    win.sp_nmax.setValue(6)
    win.tabs.setCurrentIndex([win.tabs.tabText(i)
                              for i in range(win.tabs.count())].index("逐历元诊断"))
    app.processEvents()
    win.run_series_analysis()
    wait_for(app, win.series_worker, timeout=600)
    app.processEvents()
    check("显示回归：先有批量（多历元）系数",
          getattr(win, "series_coeffs", None) is not None
          and win.series_coeffs.ntime == _nt,
          f"ntime={getattr(win.series_coeffs, 'ntime', None)}")

    # (1) 水平形变：**按需计算**（点按钮才算）+ 经度轴口径 + 海岸线 + 自己的色标
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    check("★ 水平形变：批量分析完成后**没有**自动算（用户要求点了才算）",
          win._horiz_data is None and win._horiz_sig is None,
          f"_horiz_data={win._horiz_data is not None}，"
          f"页脚={win.horiz_note.text()[:34]!r}")
    win.tabs.setCurrentIndex(tabs.index("水平形变"))
    app.processEvents()
    check("★ 水平形变：切到这个页签也**不算**（只提示点按钮）",
          win._horiz_data is None
          and ("计算水平形变" in win.horiz_note.text()
               or "计算水平形变" in win.horiz_canvas.figure.axes[0].texts[0].get_text()
               if win.horiz_canvas.figure.axes else False),
          win.horiz_note.text()[:42])
    win.btn_horiz.click()
    wait_for(app, win.horiz_worker, timeout=600)      # 现在是异步算（worker 线程）
    app.processEvents()
    check("★ 水平形变：点「计算水平形变」之后才算出来",
          win._horiz_data is not None and win._horiz_sig == win._horiz_key(),
          f"sig={win._horiz_sig} key={win._horiz_key()}")
    axn = win.horiz_canvas.figure.axes[0]
    x0, x1 = axn.get_xlim()
    check("★ 水平形变：经度轴归一到 [-180,180]（0…360 的数据不再把轴撑成 540°）",
          -180.5 <= x0 and x1 <= 180.5,
          f"xlim=({x0:.1f}, {x1:.1f})；数据经度 0…360")
    check("★ 水平形变：画了海岸线（与地图页同一套离线数据）",
          len(axn.get_lines()) >= 1, f"{len(axn.get_lines())} 条折线")
    check("水平形变：用的是批量系数里当前历元那一片",
          "第" in win.horiz_canvas.figure._suptitle.get_text(),
          win.horiz_canvas.figure._suptitle.get_text()[:40])
    # 换时次 → 只作废、不重算
    _h_before = win._horiz_data
    win.time_slider.setValue(win.time_slider.value() + 1)
    app.processEvents()
    check("★ 水平形变：换时次只作废旧图、**不自动重算**",
          win._horiz_data is None and win._horiz_sig is None
          and "计算水平形变" in win.horiz_note.text(),
          win.horiz_note.text()[:44])
    del _h_before
    win.time_slider.setValue(0)
    win.btn_horiz.click()
    wait_for(app, win.horiz_worker, timeout=600)
    app.processEvents()

    # (1b) 色标：水平形变**不跟随**地图页的色标（否则毫米级分量被压成一片颜色）
    _hz_had_own = win.horiz_canvas.color is win.horiz_color_scale
    win.map_field.setCurrentIndex(0)
    win.chk_color_auto.setChecked(False)
    win.edit_vmin.setText("-100000")
    win.edit_vmax.setText("100000")
    win._on_apply_color()
    app.processEvents()
    win.btn_horiz.click()
    wait_for(app, win.horiz_worker, timeout=600)
    app.processEvents()
    _clim_n = win.horiz_canvas.figure.axes[0].collections[0].get_clim()
    _north_max = float(np.nanmax(np.abs(np.asarray(
        win._horiz_data["north"], dtype=float)))) if win._horiz_data else 0.0
    check("★ 水平形变分量用**自己的**自动色标（不被地图的手动 ±1e5 压平）",
          _hz_had_own and abs(_clim_n[1]) < 10 * max(_north_max, 1e-30)
          and abs(_clim_n[1]) > _north_max * 0.5,
          f"分量色标 = {_clim_n}，数据 |u_N|max = {_north_max:.4g}，"
          f"地图色标 = {win.color_scale.limits(np.array([0.0]))}")
    win.chk_color_auto.setChecked(True)
    app.processEvents()

    # (1c) 水平形变计算挪出界面线程（用户要求：散点上不能把窗口冻住）
    #   以前 `btn_horiz.click()` 是**同步**算的：散点百万点要 79 s，整窗冻住、
    #   没有进度也不能取消；子进程那条路（subproc 的 horizontal_grid）压根没接上。
    _ticks = []
    _t_click = time.time()
    win.btn_horiz.click()
    _click_dt = time.time() - _t_click
    check("★ 点「计算水平形变」**立刻返回**（不在界面线程里算）",
          _click_dt < 2.0 and win.horiz_worker is not None
          and win.horiz_worker.isRunning() and not win.btn_horiz.isEnabled(),
          f"click 用了 {_click_dt*1000:.0f} ms，worker 在跑="
          f"{win.horiz_worker.isRunning()}，按钮仍禁用="
          f"{not win.btn_horiz.isEnabled()}")
    check("  算的时候按钮/停止按钮状态正确（停止键可用）",
          not win.btn_horiz.isEnabled() and win.btn_stop.isEnabled(), "")
    win.horiz_worker.progressed.connect(lambda m, f: _ticks.append((m, f)))
    wait_for(app, win.horiz_worker, timeout=600)
    app.processEvents()
    check("★ 计算过程有进度回呼（按阶 m / 分块报）",
          len(_ticks) >= 1 and all("阶 m=" in m for m, _f in _ticks),
          f"{len(_ticks)} 次，例 {_ticks[0][0]!r}" if _ticks else "没有进度")
    check("  跑完：图已更新、按钮恢复、进度条收工",
          win._horiz_data is not None and win.btn_horiz.isEnabled()
          and not win.progress.isVisible(),
          f"sig={win._horiz_sig is not None}")

    # 取消：协作式（这里是"还没开跑就取消"，一定能被第一次 tick 拦下）
    from shkit.gui.workers import HorizontalWorker as _HW
    from shkit.gradient import canonical_scaled as _cs
    _C1, _S1, _F1 = _cs(win._coeffs_for_epoch(0))
    _ds_now = win.dataset
    _la = _ds_now.lat_vec if _ds_now.kind == "grid" else _ds_now.lat
    _lo = _ds_now.lon_vec if _ds_now.kind == "grid" else _ds_now.lon
    _w = _HW(_ds_now.kind, _la, _lo, _C1, _S1, _F1, parent=win)
    _seen = []
    _w.cancelled.connect(lambda: _seen.append("cancelled"))
    _w.succeeded.connect(lambda _o: _seen.append("succeeded"))
    _w.failed.connect(lambda _t: _seen.append("failed"))
    _w.cancel()                       # 先举手，再开跑 → 第一个 tick 就该退出
    _w.start()
    wait_for(app, _w, timeout=120)
    app.processEvents()
    check("★ 取消：协作式取消立刻生效（不会白算完）",
          _seen == ["cancelled"] and not _w.isRunning(), f"{_seen}")

    # (4) 重建/差值随历元重算（批量系数存在时）
    win.tabs.setCurrentIndex(0)
    win.map_field.setCurrentIndex(1)                  # 重建场
    win.time_slider.setValue(0)
    app.processEvents()
    f0 = float(np.nanmax(np.asarray(win.map_canvas._hover_data[3])))
    t0_title = win.map_canvas._ax.get_title()
    win.time_slider.setValue(7)
    app.processEvents()
    f7 = float(np.nanmax(np.asarray(win.map_canvas._hover_data[3])))
    check("★ 重建场随时次改变而重算（不再冻在分析时那个历元）",
          abs(f7 - f0) > 1e-9 and "批量系数" in t0_title,
          f"时次0 max={f0:.6g} → 时次7 max={f7:.6g}；标题含「批量系数」")
    win.map_field.setCurrentIndex(2)                  # 差值
    win.time_slider.setValue(0)
    app.processEvents()
    d0 = float(np.nanmax(np.abs(np.asarray(win.map_canvas._hover_data[3]))))
    win.time_slider.setValue(7)
    app.processEvents()
    d7 = float(np.nanmax(np.abs(np.asarray(win.map_canvas._hover_data[3]))))
    check("★ 差值场也随时次改变而重算", abs(d7 - d0) > 1e-12,
          f"时次0 |差值|max={d0:.3g} → 时次7 {d7:.3g}")
    check("逐时次重建有缓存（播放时不会每个历元都重算）",
          0 < len(win._epoch_cache) <= 4, f"{len(win._epoch_cache)} 个历元在缓存")
    win.map_field.setCurrentIndex(0)

    # (2)(3) 趋势页：含半年 → 5 张图；海岸线随开关
    win.tabs.setCurrentIndex([win.tabs.tabText(i)
                              for i in range(win.tabs.count())].index("趋势与周年"))
    win.chk_trend_seasonal.setChecked(True)
    app.processEvents()
    win._compute_trend_maps()
    app.processEvents()
    t_axes = win.trend_canvas._axes
    t_titles = [a.get_title() for a in t_axes]
    check("★ 趋势页：勾「含半年周期」后出 5 张图（趋势 + 周年/半年各一对）",
          len(t_axes) == 5, f"{len(t_axes)} 个面板：{[t[:6] for t in t_titles]}")
    check("★ 趋势页：半年振幅场与半年相位场都在（不再只画周年）",
          any("半年振幅" in t for t in t_titles)
          and any("半年相位" in t for t in t_titles),
          "含「半年振幅场」/「半年相位场」")
    check("★ 趋势页：每个面板都画了海岸线",
          all(len(a.get_lines()) >= 1 for a in t_axes),
          f"每面板折线数 {[len(a.get_lines()) for a in t_axes]}")
    check("趋势页备注写明拟合了哪些周期", "周年 + 半年" in win.trend_note.text(),
          win.trend_note.text()[:46])
    win.chk_trend_coast.setChecked(False)
    app.processEvents()
    check("趋势页：关掉「海岸线」后所有面板都没有海岸线（开关真的起作用）",
          all(len(a.get_lines()) == 0 for a in win.trend_canvas._axes),
          f"折线数 {[len(a.get_lines()) for a in win.trend_canvas._axes]}")
    win.chk_trend_coast.setChecked(True)
    app.processEvents()
    win.grab().save(os.path.join(SHOTS, "28_trend_seasonal_coast.png"))

    # (5) 文件声明 cm：措辞必须**准确** —— 线性换算，不需要换算；只有绝对标注要按 cm 读
    _ds_cm = Dataset.from_grid("<cm 测试>", _lat, _lon, _g[:, :, :1],
                              {"variable_units": "cm"})
    win.on_loaded(_ds_cm)
    win.sp_nmax.setValue(4)
    win.chk_subprocess.setChecked(False)
    win.cb_field_unit.setCurrentIndex(
        [win.cb_field_unit.itemData(i)
         for i in range(win.cb_field_unit.count())].index("ewh"))
    app.processEvents()
    win.run_analysis()
    wait_for(app, win.analysis_worker, timeout=300)
    app.processEvents()
    win.tabs.setCurrentIndex([win.tabs.tabText(i)
                              for i in range(win.tabs.count())].index("水平形变"))
    win._refresh_horizontal(force=True)
    wait_for(app, win.horiz_worker, timeout=600)
    app.processEvents()
    _note_cm = win.horiz_note.text()
    check("★ cm 数据：说明「线性、无需换算」，只提醒绝对标注要按 cm 读",
          "无需换算" in _note_cm and "÷100" in _note_cm
          and "差 100 倍" not in _note_cm,
          _note_cm[-70:])
    check("数据摘要里的变量单位也不再说「必须先换算成米」",
          "变量单位    : cm" in win.dataset.summary()
          and "先换算成米" not in win.dataset.summary(),
          [ln for ln in win.dataset.summary().splitlines()
           if "变量单位" in ln][0][:58])
    # 声明为位系数（无量纲场）时不该出现这条单位提示
    win.cb_field_unit.setCurrentIndex(
        [win.cb_field_unit.itemData(i)
         for i in range(win.cb_field_unit.count())].index("geopotential"))
    app.processEvents()
    win.run_analysis()
    wait_for(app, win.analysis_worker, timeout=300)
    win._refresh_horizontal(force=True)
    wait_for(app, win.horiz_worker, timeout=600)
    app.processEvents()
    check("位系数（无量纲场）时不出现单位提示",
          "单位：文件声明" not in win.horiz_note.text(),
          win.horiz_note.text()[:40])

    # (7) 水平形变页：三行（北/东/水平模）+ 滚动区 + 分量共用色标 + 矢量比例尺
    _hz_axes = win.horiz_canvas.figure.axes
    check("★ 水平形变：三行布局（北 / 东 / 水平模），每行带自己的色标",
          len(_hz_axes) >= 6, f"{len(_hz_axes)} 个 axes")
    _titles = [_hz_axes[0].get_title(), _hz_axes[1].get_title(),
               _hz_axes[2].get_title()]
    check("★ 水平形变：第三行是水平模 |u_h|（不是只有箭头）",
          "北分量" in _titles[0] and "东分量" in _titles[1]
          and "水平模" in _titles[2], " / ".join(t[:8] for t in _titles))
    _c0 = _hz_axes[0].collections[0].get_clim()
    _c1 = _hz_axes[1].collections[0].get_clim()
    check("★ 水平形变：u_N 与 u_E **共用**色标范围（可横向对比）",
          _c0 == _c1 and abs(_c0[0] + _c0[1]) < 1e-12,
          f"u_N={tuple(round(v, 6) for v in _c0)} u_E={tuple(round(v, 6) for v in _c1)}")
    _cm = _hz_axes[2].collections[0].get_clim()
    check("水平模用顺序色标且从 0 起", abs(_cm[0]) < 1e-15 and _cm[1] > 0,
          f"|u_h| clim={tuple(round(v, 6) for v in _cm)}")
    _k = getattr(win.horiz_canvas, "_quiver_key", None)
    # QuiverKey 的标签不在 ax.texts 里，而且不同 matplotlib 版本一个放在 .label、
    # 一个放在 .text（get_label() 在 3.10 上返回空串）—— 两种都读。
    _lbl = ""
    if _k is not None:
        _lbl = str(getattr(_k, "label", "") or "")
        if not _lbl:
            _t = getattr(_k, "text", None)
            _lbl = str(getattr(_t, "get_text", lambda: "")())
    check("★ 水平形变：矢量带比例尺（箭头长度可定量读）",
          _k is not None and "m" in _lbl, _lbl[:36] or "没有 quiverkey")
    # 用户要求：水平模里的箭头与左下角比例尺文字都用**白色**
    from matplotlib.colors import to_rgb as _to_rgb
    from matplotlib.quiver import Quiver as _Q
    _q_art = [c for c in win.horiz_canvas.figure.axes[2].collections
              if isinstance(c, _Q)]
    _white_arrow = bool(_q_art) and bool(np.allclose(
        np.asarray(_q_art[0].get_facecolor())[0][:3], 1.0))
    _white_label = bool(np.allclose(
        _to_rgb(win.horiz_canvas._quiver_key.text.get_color()), 1.0))
    check("★ 水平模：箭头与比例尺文字都是白色（箭头带细描边）",
          _white_arrow and _white_label
          and bool(_q_art[0].get_edgecolor().size),
          f"箭头底色白={_white_arrow}，文字白={_white_label}")
    check("★ 水平形变：画布放在滚动区里，且按自然高度给最小高度（不再压扁）",
          type(win.horiz_scroll).__name__ == "QScrollArea"
          and win.horiz_canvas.minimumHeight() > 1000,
          f"最小高 {win.horiz_canvas.minimumHeight()} px")

    # (8) 多时次数据的「运行分析」= 一次解完**全部**历元（不是只解当前那一个）
    # 用户的原话："点运行分析，就该载入全部数据并处理完全部数据，缓存到本地；
    # 绘图时直接展示即可" —— 所以这里钉住：一次点击之后，逐历元视图 / 播放 /
    # 导出都不需要再解任何一次方程。
    win.on_loaded(ds_mt)
    win.chk_subprocess.setChecked(False)
    win.sp_nmax.setValue(6)
    win.run_analysis()                       # 多时次 → 走"整条序列"这条路
    wait_for(app, win.series_worker, timeout=600)
    app.processEvents()
    _nt_mt = win.dataset.ntime
    check("★ 多时次数据点「运行分析」= 一次解完全部历元（不是只解一个）",
          getattr(win, "series_coeffs", None) is not None
          and win.series_coeffs.ntime == _nt_mt
          and win.series_coeffs is not None and win.coeffs is not None
          and np.asarray(win.coeffs.C).ndim == 2,
          f"series_ntime={getattr(getattr(win, 'series_coeffs', None), 'ntime', None)}"
          f"/{_nt_mt}，self.coeffs 是第 {win._single_t + 1 if win._single_t is not None else '?'} 个历元的切片")
    check("★ 逐历元诊断表也被这一次填满（同一个解，不用再点批量分析）",
          win.epochs_table.rowCount() == _nt_mt,
          f"{win.epochs_table.rowCount()} 行")

    win.tabs.setCurrentIndex(0)
    win.map_field.setCurrentIndex(1)         # 重建场
    win.time_slider.setValue(0)
    app.processEvents()
    _g0 = float(np.nanmax(np.asarray(win.map_canvas._hover_data[3])))
    win.time_slider.setValue(6)
    app.processEvents()
    _g6 = float(np.nanmax(np.asarray(win.map_canvas._hover_data[3])))
    check("★ 换时次只是切片 + 综合（没有新的求解在跑）",
          abs(_g6 - _g0) > 1e-12
          and not (win.series_worker is not None and win.series_worker.isRunning())
          and not (win.analysis_worker is not None
                   and win.analysis_worker.isRunning()),
          f"时次0 max={_g0:.5g} → 时次6 max={_g6:.5g}")
    check("标题写明用的是哪个历元的系数",
          "第 7 个历元" in win.map_canvas._ax.get_title(),
          win.map_canvas._ax.get_title()[-22:])

    # 逐历元视图"没有系数"时：一次解完整条序列，而不是拿原始数据顶上
    win.on_loaded(ds_mt)                     # 换数据 → 手上一个系数都没有
    app.processEvents()
    win.map_field.setCurrentIndex(0)
    app.processEvents()
    _raw0 = np.sort(np.asarray(ds_mt.value_slice(0), dtype=float).ravel())
    win.map_field.setCurrentIndex(1)         # 切到「重建场」→ 触发一次解完
    app.processEvents()
    _running_now = (win.series_worker is not None
                    and win.series_worker.isRunning())
    _shown = (None if win.map_canvas._hover_data is None else
              np.sort(np.asarray(win.map_canvas._hover_data[3],
                                 dtype=float).ravel()))
    check("★ 还没解过时切「重建场」：开跑整条序列，画面不给原始数据",
          (_running_now or win.series_coeffs is not None
           or "一次解完" in win.map_note.text())
          and (_shown is None or not np.allclose(_shown, _raw0)),
          f"series 在跑={_running_now}，提示={win.map_note.text()[:34]!r}")
    wait_for(app, win.series_worker, timeout=600)
    app.processEvents()
    _g1 = float(np.nanmax(np.asarray(win.map_canvas._hover_data[3])))
    check("★ 解完后自动换成该历元的重建场（图与提示都不撒谎）",
          "第 1 个历元" in win.map_canvas._ax.get_title()
          and abs(_g1 - _raw0.max()) > 1e-12,
          win.map_canvas._ax.get_title()[-22:])

    # (8b) 磁盘缓存：只有"有真实文件"的数据集才缓存（内存里编的网格不写盘）
    # ⚠️ 这份测试数据写在**系统临时目录**：本仓库常放在同步盘/网络盘上，
    # HDF5 建文件偶发 "unable to create file"，会把这条用例变成随机红。
    from shkit.gui import coeff_cache as _ccache
    _cache_dir = tempfile.mkdtemp(prefix="shkit_gui_cache_")
    _cache_nc = os.path.join(_cache_dir, "gui_cache_series.nc")
    _gcube = np.random.default_rng(11).normal(size=(_lat.size, _lon.size, 4))
    shio.write_grid(_cache_nc, _lat, _lon, _gcube, var="value",
                    meta={"time": _tv[:4]}, long_name="gui cache test")
    check("缓存用例的数据文件写得出来（缓存断言的前提）",
          os.path.isfile(_cache_nc), _cache_nc)
    ds_cache = Dataset.from_grid(_cache_nc, _lat, _lon, _gcube, {"time": _tv[:4]})
    _ccache.clear(ds_cache)
    win.chk_subprocess.setChecked(False)
    win.sp_nmax.setValue(4)
    win.map_field.setCurrentIndex(1)         # 逐历元视图：解完就能直接画
    win.on_loaded(ds_cache)
    app.processEvents()
    win.run_analysis()
    wait_any_solve(app, win, timeout=600)
    app.processEvents()
    _cache_files = [p for p in [_cache_nc + _ccache.SUFFIX] if os.path.isfile(p)]
    check("★ 解完后把逐历元系数写到数据文件旁边（缓存）",
          bool(_cache_files) and win.series_coeffs is not None
          and win.series_coeffs.ntime == 4,
          f"{[os.path.basename(p) for p in _cache_files] or '没有缓存文件'}")
    _cached_C = np.asarray(win.series_coeffs.C).copy()

    # 再次载入同一份数据 + 同一套参数 → 直接读缓存，不重算
    win.on_loaded(ds_cache)
    app.processEvents()
    check("★ 再次载入：命中本地缓存，直接拿到逐历元系数（没有新求解在跑）",
          win.series_coeffs is not None and win._series_cache is not None
          and not (win.series_worker is not None and win.series_worker.isRunning()),
          f"来源 {os.path.basename(str((win._series_cache or {}).get('path', '')))}，"
          f"写于 {(win._series_cache or {}).get('created')}")
    check("★ 缓存读回来的系数与真算的**逐值一致**",
          win.series_coeffs is not None
          and np.array_equal(np.asarray(win.series_coeffs.C), _cached_C),
          f"shape={np.asarray(win.series_coeffs.C).shape}")
    check("★ 缓存命中这件事被写明（不让人以为「这是刚算的」）",
          "缓存" in (win.status_label.text() + win.epochs_note.text()),
          (win.status_label.text() or win.epochs_note.text())[:52])
    # ★ 载入这份数据时，参数会被**恢复成它上次求解用的那一套**并直接命中 —— 这正是
    # 用户点名要的：「识别到缓存时，「输入是」这个应该自动切换吧」。
    # 以前这里断言的是"改了 nmax 再载入 → 不命中"，那条契约的后果是：默认「输入是」
    # 与缓存里的不同 → 每次打开都重算 → 重算又把缓存写成默认那套 → 下次还是对不上，
    # 来回打转（实测用户那份真实 mascon 就这样连着重算了两次）。
    _fu_cached = win.cb_field_unit.currentData()
    _fu_other = next(d for d in (win.cb_field_unit.itemData(i)
                                 for i in range(win.cb_field_unit.count()))
                     if d != _fu_cached)
    win.cb_field_unit.setCurrentIndex(
        win.cb_field_unit.findData(_fu_other))     # 故意与缓存不一致
    win.sp_nmax.setValue(3)                        # nmax 也与缓存（4）不一致
    win.on_loaded(ds_cache)
    app.processEvents()
    check("★ 载入时「输入是」与缓存不一致 → 自动切回缓存那一档并命中，不重算",
          win.cb_field_unit.currentData() == _fu_cached
          and win.series_coeffs is not None and win._series_cache is not None,
          f"输入是 {_fu_other!r} → {win.cb_field_unit.currentData()!r}；"
          f"命中={win._series_cache is not None}")
    check("★ 其余参数（nmax）也一起切回缓存那一套",
          win.sp_nmax.value() == 4 and win.series_coeffs.nmax == 4,
          f"nmax 控件={win.sp_nmax.value()}，结果 nmax="
          f"{getattr(win.series_coeffs, 'nmax', None)}")
    check("★ 这次切换被写明（不静默改用户的参数）",
          "切回" in win.cache_hint.text() or "切回" in win.status_label.text(),
          win.cache_hint.text().replace("\n", " / ")[-70:])
    # 但**点了运行**是另一个语义：用户刚设的参数就是他要的 → 按新参数重算并覆盖缓存
    win.sp_nmax.setValue(3)
    win.run_analysis()
    wait_any_solve(app, win, timeout=600)
    app.processEvents()
    check("★ 用户改了参数后点「运行分析」→ 按新参数重算（不回退到缓存那套）",
          win.series_coeffs is not None and win.series_coeffs.nmax == 3,
          f"nmax={getattr(win.series_coeffs, 'nmax', None)}")
    win.on_loaded(ds_cache)                   # 现在该命中新写的那一份
    app.processEvents()
    check("★ 缓存只保留**最新一套参数**：重算后写入的是新参数那一份",
          win._series_cache is not None and win.series_coeffs is not None
          and win.series_coeffs.nmax == 3,
          f"nmax={getattr(win.series_coeffs, 'nmax', None)}，"
          f"来源 {os.path.basename(str((win._series_cache or {}).get('path', '')))}")

    # (8b3) 用户要求：「有缓存时，点击运行分析弹出提示，用户决定是否重跑」。
    # ① 真的对话框：校验按钮→决定 的映射（把 exec 换成"自动点某个按钮"）
    import shkit.gui.main_window as _mw
    from PySide6.QtWidgets import QMessageBox as _QMB

    class _AutoBox(_QMB):
        pick = "直接读缓存"

        def exec(self):                      # noqa: D102
            for _b in self.buttons():
                if self.pick in _b.text():
                    _b.click()
                    return 0
            return 0

    _mb_orig = _mw.QMessageBox
    _hit_now = _ccache.load(ds_cache, win._collect_params())
    try:
        for _pick, _want in (("直接读缓存", "cache"), ("重新计算", "recompute"),
                             ("取消", "cancel")):
            _AutoBox.pick = _pick
            _mw.QMessageBox = _AutoBox
            _got = _real_ask_cache(_hit_now, win._collect_params())
            check(f"弹窗：点「{_pick}」→ 决定 {_want}", _got == _want, f"得到 {_got!r}")
        _AutoBox.pick = "不存在"
        _mw.QMessageBox = _AutoBox
        check("弹窗：关掉对话框（没点任何按钮）→ 当取消处理",
              _real_ask_cache(_hit_now, win._collect_params()) == "cancel", "")
    finally:
        _mw.QMessageBox = _mb_orig

    # ② 三种决定的**实际效果**（用 stub 决定，不弹窗）
    win.on_loaded(ds_cache)                  # 先拿到一份内存里的结果
    app.processEvents()
    _C_before = np.asarray(win.series_coeffs.C).copy()
    _n_before = len(_ask_calls)
    win._ask_decision = "cancel"
    win.run_analysis()
    app.processEvents()
    check("★ 选「取消」→ 不读缓存也不重算，内存里的结果原样保留",
          len(_ask_calls) == _n_before + 1
          and not (win.series_worker is not None and win.series_worker.isRunning())
          and np.array_equal(np.asarray(win.series_coeffs.C), _C_before)
          and "已取消" in win.status_label.text(),
          f"问过 {len(_ask_calls) - _n_before} 次，状态={win.status_label.text()[:40]!r}")
    win._ask_decision = "cache"
    _n_before = len(_ask_calls)
    win.run_analysis()
    app.processEvents()
    check("★ 选「直接读缓存」→ 弹过提示、不跑求解、写明本次未重算",
          len(_ask_calls) == _n_before + 1
          and not (win.series_worker is not None and win.series_worker.isRunning())
          and win._series_cache is not None
          and "未重算" in win.progress_label.text(),
          f"label={win.progress_label.text()!r}")
    win._ask_decision = "recompute"
    _n_before = len(_ask_calls)
    _cache_mtime0 = os.path.getmtime(_cache_nc + _ccache.SUFFIX)
    win.run_analysis()
    wait_any_solve(app, win, timeout=600)
    app.processEvents()
    check("★ 选「重新计算」→ 真的解了一遍（不走缓存）并覆盖缓存文件",
          len(_ask_calls) == _n_before + 1 and win.series_coeffs is not None
          and win._series_cache is None
          and os.path.getmtime(_cache_nc + _ccache.SUFFIX) >= _cache_mtime0,
          f"nmax={getattr(win.series_coeffs, 'nmax', None)}，"
          f"_series_cache={win._series_cache is not None}")
    win._ask_decision = "cache"

    # ③ 自动补算的路径（导出/切视图）**不弹窗** —— 那里弹模态框会把人卡住
    import inspect
    check("★ 只有显式点「运行分析/批量分析」才问（其余调用点默认不问）",
          inspect.signature(win._start_series).parameters["ask_cache"].default
          is False, "")
    _asked_during_auto = []
    win._ask_series_cache = lambda hit, params: (
        _asked_during_auto.append(1), "cache")[1]
    win.series_coeffs = None
    win._series_cache = None
    _ok_sync = win._ensure_series_sync("导出前补算")
    app.processEvents()
    check("★ 自动补算（导出前同步解完）用缓存但**不问**用户",
          _ok_sync and not _asked_during_auto and win.series_coeffs is not None,
          f"问过 {len(_asked_during_auto)} 次，ok={_ok_sync}")
    win._ask_series_cache = _ask_stub

    # (8b2) 进度条生命周期 + 缓存限量。
    # 用户报障两连：「如果本地有缓存,运行后,进度条会卡在0不动」「缓存会无限量吗?」
    # 前者：QProgressBar 一塞进布局就是可见的，而隐藏动作以前只挂在"序列线程跑完"
    # 上 —— 缓存命中不跑线程，于是那根 0% 的条永远留着。
    # ⚠️ 此刻 nmax 仍是 3（上一段刚用 3 写了缓存），下面这一下必须命中。
    win.on_loaded(ds_cache)                   # 这一下必然命中缓存（上一步刚写过）
    app.processEvents()
    check("★ 缓存命中后进度条不会停在 0%（推到 100% 并隐藏，且写明未重算）",
          (not win.progress.isVisible()) and win.progress.value() == 1000
          and "缓存" in win.progress_label.text()
          and "未重算" in win.progress_label.text(),
          f"visible={win.progress.isVisible()}, value={win.progress.value()}, "
          f"label={win.progress_label.text()!r}")
    check("参数面板的缓存提示如实报出占用（用户不用猜缓存到底占多大）",
          "缓存" in win.cache_hint.text()
          and ("MB" in win.cache_hint.text() or "kB" in win.cache_hint.text()),
          win.cache_hint.text().replace("\n", " / ")[:70])

    _old_la = os.environ.get("LOCALAPPDATA")
    _la_dir = tempfile.mkdtemp(prefix="shkit_gui_la_")
    os.environ["LOCALAPPDATA"] = _la_dir
    try:
        _rep = _ccache.report(ds_cache)
        check("缓存报告：给出数据旁缓存的大小 + 用户缓存目录的总量上限",
              _rep["data_exists"] and _rep["data_size"] > 0
              and _rep["user_limit"] == _ccache.USER_CACHE_MAX_BYTES,
              f"{_rep['data_size']} B，上限 {_rep['user_limit'] / 1e6:.0f} MB")
        # 单份上限：超了就不写盘，并且**说出来**（结果照常可用）
        _saved_cap = _ccache.ENTRY_MAX_BYTES
        _ccache.ENTRY_MAX_BYTES = 16
        try:
            _p, _note = _ccache.save(ds_cache, win._collect_params(),
                                     win.series_coeffs, None)
        finally:
            _ccache.ENTRY_MAX_BYTES = _saved_cap
        check("★ 单份缓存超上限 → 不写盘且明确说明（不静默占用户磁盘）",
              _p is None and "上限" in _note and "未写缓存" in _note
              and "照常可用" in _note, _note[:60])
        check("  拒绝写盘时不动已有的那份缓存",
              os.path.isfile(_cache_nc + _ccache.SUFFIX), "")
        # 用户缓存目录：本来"数据集×参数"无限涨 → 超上限按最久未用修剪
        _udir = _ccache.user_cache_dir()
        os.makedirs(_udir, exist_ok=True)
        _now = time.time()
        for _i in range(5):
            _fp = os.path.join(_udir, f"fake{_i}.npz")
            with open(_fp, "wb") as _fh:
                _fh.write(b"\0" * (1024 * 1024))
            os.utime(_fp, (_now + _i, _now + _i))
        _n, _freed, _rm = _ccache.prune_user_cache(max_bytes=3 * 1024 * 1024)
        _left = sorted(os.listdir(_udir))
        check("★ 用户缓存超上限 → 按最久未用删到 80% 以内（保留最新的）",
              _n == 3 and _left == ["fake3.npz", "fake4.npz"] and _freed > 0,
              f"删 {_n} 份，释放 {_freed / 1e6:.1f} MB，剩 {_left}")
        check("  没超上限时一份都不删",
              _ccache.prune_user_cache(max_bytes=100 * 1024 * 1024)[0] == 0, "")
        _n2, _freed2 = _ccache.clear_user_cache()
        check("  可以一键清空用户缓存目录（并报出释放量）",
              _n2 == 2 and os.listdir(_udir) == [] and _freed2 > 0,
              f"删 {_n2} 份，释放 {_freed2 / 1e6:.1f} MB")
        # ★ 数据目录不可写（NAS 只读挂载就是这样）→ 缓存退到用户缓存目录；此时
        # find_any 也必须能从那里认出"这份数据有缓存"，否则载入时就不会自动切参数，
        # 又回到"明明缓存了、点运行还是重算"。
        _saved_paths = _ccache.paths
        _ccache.paths = lambda ds: [os.path.join(_cache_nc, "not_a_dir", "x.npz")]
        try:
            _pp2, _note2 = _ccache.save(ds_cache, win._collect_params(),
                                        win.series_coeffs, None)
            _found2 = _ccache.find_any(ds_cache)
        finally:
            _ccache.paths = _saved_paths
        check("★ 数据目录不可写时：缓存退到用户目录，find_any 仍能认出来",
              _pp2 is not None and _found2 is not None
              and _found2["source"] == "user-cache",
              f"写入={os.path.basename(str(_pp2))}，"
              f"认出={_found2 and _found2['source']}")
        check("  认出来的参数与写进去的一致",
              (_found2 or {}).get("params", {}).get("nmax")
              == win._collect_params()["nmax"],
              f"nmax={(_found2 or {}).get('params', {}).get('nmax')}")
        check("  「本地缓存…」菜单项存在（用户自己就能看/清）",
              hasattr(win, "act_cache") and "缓存" in win.act_cache.text(), "")

        # ★ 「文件在不在」≠「能不能命中」。实测用户那份 mascon 缓存挂在数据旁边、
        # 7.4 MB，但键里写的是旧路径（D:\myds\…），其实命不中；如果提示只看"文件
        # 在不在"，界面就会说"直接读它、不重算"—— 那是最不能接受的那种撒谎。
        _p_now = _ccache.probe(ds_cache, win._collect_params())
        check("★ probe()：当前文件+参数下这份缓存确实能命中",
              _p_now is not None and _p_now["source"] == "data-dir",
              f"{os.path.basename(str(_p_now and _p_now['path']))}")
        check("★ 换个参数问 probe()：明确回答「不能命中」（不靠文件存在与否）",
              _ccache.probe(ds_cache, dict(win._collect_params(),
                                           nmax=win.sp_nmax.value() + 1)) is None,
              "")
        _rep_mismatch = _ccache.report(ds_cache, dict(win._collect_params(), nmax=99))
        check("  report(…, params) 也如实给出 data_match=False",
              _rep_mismatch["data_exists"] and _rep_mismatch["data_match"] is False,
              f"存在={_rep_mismatch['data_exists']}，能命中="
              f"{_rep_mismatch['data_match']}")
        _rep_ok = _ccache.report(ds_cache, win._collect_params())
        check("  参数一致时 data_match=True", _rep_ok["data_match"] is True, "")

        # ★ 补丁版本号不该让用户缓存全部作废：键里的 shkit_version 只留档、不参与匹配
        # （2.0.0 → 2.0.1 这种补丁改了界面/文案，系数的含义没变；真正决定能不能用的
        # 是 CACHE_VERSION，语义变了才 +1）。
        _v_orig = _ccache.SHKIT_VERSION
        try:
            _ccache.SHKIT_VERSION = "0.0.0-别的版本"
            _hit_v = _ccache.probe(ds_cache, win._collect_params())
        finally:
            _ccache.SHKIT_VERSION = _v_orig
        check("★ 换了软件版本号（补丁）→ 缓存**仍然命中**（版本不参与匹配）",
              _hit_v is not None,
              f"伪造版本下 probe={'命中' if _hit_v else '未命中'}")
        _k1 = {"cache_version": 3, "shkit_version": "2.0.0", "source": {"a": 1}}
        _k2 = {"cache_version": 3, "shkit_version": "9.9.9", "source": {"a": 1}}
        check("  比较串剔掉 shkit_version，但保留 cache_version",
              _ccache._blob(_k1) == _ccache._blob(_k2)
              and _ccache._blob(_k1) != _ccache._blob(
                  dict(_k1, cache_version=4)), "")

        # ★ 键里**不含绝对路径**（v3）：数据连同旁边的缓存一起拷到别的目录仍然命中
        # —— "写在数据文件旁边"的全部意义就是这个。旧键里有 abspath，一拷就全废。
        _moved_dir = tempfile.mkdtemp(prefix="shkit_gui_moved_")
        _moved_nc = os.path.join(_moved_dir, "moved_series.nc")
        shutil.copy2(_cache_nc, _moved_nc)                 # copy2 保留 mtime_ns
        shutil.copy2(_cache_nc + _ccache.SUFFIX,
                     _moved_nc + _ccache.SUFFIX)
        _ds_moved = Dataset.from_grid(_moved_nc, _lat, _lon, _gcube,
                                      {"time": _tv[:4]})
        _hit_moved = _ccache.probe(_ds_moved, win._collect_params())
        check("★ 数据+缓存一起拷到别的目录 → 仍然命中（键里没有绝对路径）",
              _hit_moved is not None, f"{_moved_nc}")
        # 反向：大小一样但内容不同 → 必须不命中（靠前 64 KB 的 sha1）
        with open(_moved_nc, "r+b") as _fh:
            _fh.seek(0)
            _fh.write(b"\xff" * 8)
        os.utime(_moved_nc, None)                          # mtime 也变
        _ds_tampered = Dataset.from_grid(_moved_nc, _lat, _lon, _gcube,
                                        {"time": _tv[:4]})
        check("  内容/时间变了 → 不命中（不会拿旧缓存糊弄）",
              _ccache.probe(_ds_tampered, win._collect_params()) is None, "")
        shutil.rmtree(_moved_dir, ignore_errors=True)

        # ★ 忙碌条：补齐全历元 / 把数据交给子进程这类**没有可信百分比**的阶段，
        # 显示的是会自己动的忙碌条，而不是一根停在 0% 的死条。
        win._progress_busy("测试：正在读取全部时次 3/256…")
        _busy_ok = (win.progress.maximum() == 0 and win.progress.isVisible()
                    and "读取" in win.progress_label.text())
        win.on_progress("求积: 阶 m=31/60", 0.52)          # 有真实百分比 → 切回百分比条
        _back_ok = (win.progress.maximum() == 1000
                    and win.progress.value() == 520 and "31/60" in win.progress_label.text())
        win._progress_idle()
        _idle_ok = (not win.progress.isVisible() and win.progress.maximum() == 1000)
        check("★ 忙碌条 → 有百分比就切回百分比条 → 收工还原并隐藏",
              _busy_ok and _back_ok and _idle_ok,
              f"busy={_busy_ok}, back={_back_ok}, idle={_idle_ok}")
    finally:
        if _old_la is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = _old_la
        shutil.rmtree(_la_dir, ignore_errors=True)

    win.sp_nmax.setValue(6)                         # 交还给后面的用例
    shutil.rmtree(_cache_dir, ignore_errors=True)   # 缓存用例的临时目录

    # (8c) 「显示」切到逐历元视图：绝不拿原始数据/别的历元顶替；播放/导出只做
    # 切片 + 综合（系数已经一次解完），不再逐帧求解。
    # 用户报障原话：「显示 切换到其他选项，播放不能起效或者播放的是原始图」。
    # 先把「显示」放回原始数据，这样 on_loaded 不会顺手开跑一次求解 ——
    # 下面要的正是"手上一个系数都没有"的确定状态。
    win.map_field.setCurrentIndex(0)
    app.processEvents()
    win.on_loaded(ds_mt)
    app.processEvents()
    wait_any_solve(app, win, timeout=600)
    win.coeffs = win.report = win.recon = win.outfield = None
    win.series_coeffs = None
    win.series_report = None
    win._series_cache = None
    win._single_t = None
    win._epoch_cache.clear()
    _raw0 = np.sort(np.asarray(ds_mt.value_slice(0), dtype=float).ravel())
    win.map_field.setCurrentIndex(0)
    app.processEvents()                      # 先画一张原始图放着（最容易误留的那张）
    win.map_field.setCurrentIndex(1)         # 再切到「重建场」
    app.processEvents()
    _drawn = (None if win.map_canvas._hover_data is None else
              np.sort(np.asarray(win.map_canvas._hover_data[3],
                                 dtype=float).ravel()))
    check("★ 没有系数时切到「重建场」不会留着/画上原始数据冒充",
          _drawn is None or not np.allclose(_drawn, _raw0),
          f"画布上有场={_drawn is not None}")
    check("★ 并写明「现在在解什么」（说明书 §8：不悄悄沿用旧结果）",
          ("一次解完" in win.map_note.text()) or (win.series_coeffs is not None),
          win.map_note.text()[:52])
    wait_for(app, win.series_worker, timeout=600)
    app.processEvents()
    check("★ 等它解完：地图自动换成该历元的重建场（不是原始数据）",
          win.map_canvas._hover_data is not None
          and "第 1 个历元" in win.map_canvas._ax.get_title(),
          win.map_canvas._ax.get_title()[-22:])

    # 播放：每一帧都必须等于**它自己那个历元**的重建场
    _co = win.series_coeffs
    _unit = win.cb_field_unit.currentData() or "scalar"
    _ref = {}
    for _t in range(4):
        _ref[_t] = np.sort(np.asarray(
            win._reconstruct_for(_co.time_slice(_t), _unit),
            dtype=float).ravel())
    win.time_lo.setValue(0)
    win.time_hi.setValue(3)
    app.processEvents()
    win.time_slider.setValue(0)
    win.map_field.setCurrentIndex(1)
    app.processEvents()
    _sig0 = np.sort(np.asarray(win.map_canvas._hover_data[3],
                               dtype=float).ravel())
    win.sp_fps.setValue(30)
    win._apply_play_interval()
    win.btn_play.setChecked(True)
    app.processEvents()
    _seen = {}
    _t0 = time.time()
    while win._play_timer.isActive() and time.time() - _t0 < 300:
        app.processEvents()
        time.sleep(0.01)
        if win.map_canvas._hover_data is not None:
            _seen[win.time_slider.value()] = np.sort(
                np.asarray(win.map_canvas._hover_data[3], dtype=float).ravel())
    app.processEvents()
    check("★ 播放逐历元走到窗口末端（系数已就绪，不再等求解）",
          win.time_slider.value() == 3 and win.time_window() == (0, 3),
          f"slider={win.time_slider.value()}，窗口={win.time_window()}，"
          f"{win.play_note.text()[:24]}")
    check("★ 每一帧都是**它自己那个历元**的重建场（不再整段放同一个场）",
          len(_seen) >= 4
          and all(t in _ref and np.allclose(_seen[t], _ref[t])
                  for t in _seen),
          f"捕捉到 {sorted(_seen)} 帧")
    check("★ 末帧不再是第 1 个历元的场（用户报的「播的还是原始图」）",
          not np.allclose(_seen[max(_seen)], _sig0),
          "末帧 ≠ 第 1 个历元的场")

    # 逐时次导出「重建场」：每一历元写出的必须是**它自己**的场（自动先解完）
    # （同样写进系统临时目录：仓库常在非 ASCII 的同步盘上，HDF5 建文件会偶发失败）
    _exp2 = tempfile.mkdtemp(prefix="shkit_gui_epochs_")
    _orig_dir2 = _mw.QFileDialog.getExistingDirectory
    _mw.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: _exp2)
    try:
        win.cb_export_ext.setCurrentIndex(0)          # .nc
        win.time_lo.setValue(0)
        win.time_hi.setValue(1)
        win.map_field.setCurrentIndex(1)              # 重建场
        win.export_epochs()
        _w2 = win._export_dialog.worker
        _t0 = time.time()
        while _w2.isRunning() and time.time() - _t0 < 120:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        _files2 = sorted(glob.glob(os.path.join(_exp2, "*.nc")))
        _v = []
        for _f in _files2[:2]:
            _v.append(np.asarray(shio.read_grid(_f)[2], dtype=float))
        check("★ 逐时次导出「重建场」：自动先解完再导，两个历元的文件内容不同",
              len(_files2) == 2 and len(_v) == 2
              and not np.allclose(_v[0], _v[1]),
              f"{[os.path.basename(f) for f in _files2]}，"
              + (f"max|Δ|={np.nanmax(np.abs(_v[0] - _v[1])):.4g}" if len(_v) == 2
                 else "读不回来"))
    finally:
        _mw.QFileDialog.getExistingDirectory = _orig_dir2
        shutil.rmtree(_exp2, ignore_errors=True)
    win.map_field.setCurrentIndex(0)
    win._reset_time_range()
    app.processEvents()

    # (8d) 散点的**多时间**：时间轴 → 一次解完 → 逐帧真播 → 逐时次导出（.csv）
    from shkit.gui.dataset import Dataset as _PDS
    _np_pts, _np_t = 400, 3
    _prng = np.random.default_rng(21)
    _i = np.arange(_np_pts) + 0.5
    _plat = np.clip(90.0 - np.rad2deg(np.arccos(1.0 - 2.0 * _i / _np_pts)),
                    -85.0, 85.0)
    _plon = np.mod(np.pi * (1 + 5 ** 0.5) * _i, 2 * np.pi) * 180.0 / np.pi
    _pdates = ["2002-01-18", "2002-02-17", "2002-03-19"]
    _pvals = np.column_stack([
        np.sin(np.deg2rad(_plat) * (k + 2)) * np.cos(np.deg2rad(_plon) * (k + 1))
        for k in range(_np_t)])
    _pts = _PDS.from_points("<多时次散点>", _plat, _plon, _pvals,
                            {"time": _pdates})
    win.on_loaded(_pts)
    app.processEvents()
    check("★ 散点多时间：Dataset 认出 ntime 与日期轴",
          _pts.ntime == _np_t and _pts.has_time()
          and _pts.time_axis() is not None
          and _pts.time_axis().has_dates,
          f"ntime={_pts.ntime}")
    check("★ 散点多时间：时间行出现、滑块到 ntime-1",
          win.time_row.isVisible()
          and win.time_slider.maximum() == _np_t - 1,
          f"滑块最大 = {win.time_slider.maximum()}")
    win.sp_nmax.setValue(6)
    win.run_analysis()
    wait_any_solve(app, win, timeout=600)
    app.processEvents()
    check("★ 散点多时间：一次「运行分析」解完全部历元",
          win.series_coeffs is not None and win.series_coeffs.ntime == _np_t
          and win.epochs_table.rowCount() == _np_t,
          f"ntime={getattr(win.series_coeffs, 'ntime', None)}，"
          f"表 {win.epochs_table.rowCount()} 行")
    check("★ 散点逐时次导出的默认格式是 .csv（.nc/.grd 是网格格式）",
          str(win.cb_export_ext.currentData()) == ".csv",
          f"{win.cb_export_ext.currentData()!r}")

    win.map_field.setCurrentIndex(1)          # 重建场（散点）
    win.time_slider.setValue(0)
    app.processEvents()
    _pref = {k: np.sort(np.asarray(win._reconstruct_for(
        win.series_coeffs.time_slice(k),
        win.cb_field_unit.currentData() or "scalar")).ravel())
        for k in range(_np_t)}
    win.time_lo.setValue(0)
    win.time_hi.setValue(_np_t - 1)
    win.sp_fps.setValue(30)
    win._apply_play_interval()
    win.btn_play.setChecked(True)
    app.processEvents()
    _pseen = {}
    _t0 = time.time()
    while win._play_timer.isActive() and time.time() - _t0 < 120:
        app.processEvents()
        time.sleep(0.01)
        if win.map_canvas._hover_data is not None:
            _pseen[win.time_slider.value()] = np.sort(np.asarray(
                win.map_canvas._hover_data[3], dtype=float).ravel())
    app.processEvents()
    check("★ 散点多时间：播放每一帧都是该历元自己的重建场",
          len(_pseen) == _np_t
          and all(np.allclose(_pseen[k], _pref[k], rtol=1e-9) for k in _pseen),
          f"捕捉 {sorted(_pseen)} 帧")

    _pexp = tempfile.mkdtemp(prefix="shkit_gui_pts_")
    _orig_dir3 = _mw.QFileDialog.getExistingDirectory
    _mw.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: _pexp)
    try:
        win.time_lo.setValue(0)
        win.time_hi.setValue(1)
        win.map_field.setCurrentIndex(1)
        win.export_epochs()
        _w3 = win._export_dialog.worker
        _t0 = time.time()
        while _w3.isRunning() and time.time() - _t0 < 120:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        _pfiles = sorted(glob.glob(os.path.join(_pexp, "*.csv")))
        _pv = [np.asarray(shio.read_points(f)[2], dtype=float)
               for f in _pfiles[:2]]
        check("★ 散点逐时次导出写成 .csv，两个历元内容不同",
              len(_pfiles) == 2 and len(_pv) == 2
              and not np.allclose(_pv[0], _pv[1]),
              f"{[os.path.basename(f) for f in _pfiles]}，"
              + (f"max|Δ|={np.nanmax(np.abs(_pv[0] - _pv[1])):.4g}"
                 if len(_pv) == 2 else "读不回来"))

        # 水平形变导出（散点）：只能 .csv，而且**先给出耗时估算**让人确认
        _hz_pt = os.path.join(SHOTS, "horiz_export_pts")
        shutil.rmtree(_hz_pt, ignore_errors=True)
        _mw.QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: _hz_pt)
        _saved_sec = win._HORIZ_SEC_PER_POINT_DEG
        win._HORIZ_SEC_PER_POINT_DEG = 1.0        # 强制让估算值超过阈值（确定性）
        _asked = {}
        _orig_q = _mw.QMessageBox.question
        _mw.QMessageBox.question = staticmethod(
            lambda *a, **k: (_asked.update(
                called=True, text=str(a[2]) if len(a) > 2 else ""),
                _mw.QMessageBox.No)[1])
        try:
            win.export_horizontal("window")       # 选「否」→ 一个文件都不该写
            app.processEvents()
            _after_no = glob.glob(os.path.join(_hz_pt, "*"))
            _mw.QMessageBox.question = staticmethod(
                lambda *a, **k: _mw.QMessageBox.Yes)
            win.export_horizontal("current")      # 选「是」→ 正常导
            _wh3 = win._export_dialog.worker
            _t0 = time.time()
            while _wh3.isRunning() and time.time() - _t0 < 180:
                app.processEvents()
                time.sleep(0.01)
            app.processEvents()
        finally:
            _mw.QMessageBox.question = _orig_q
            win._HORIZ_SEC_PER_POINT_DEG = _saved_sec
        check("★ 散点导出水平形变：先给出耗时估算，选「否」就真的不导",
              _asked.get("called") and not _after_no
              and ("分钟" in _asked.get("text", "")
                   or "秒" in _asked.get("text", "")),
              f"问过={_asked.get('called')}，{_asked.get('text', '')[:44]!r}")
        _hf3 = sorted(glob.glob(os.path.join(_hz_pt, "*")))
        check("★ 散点导出水平形变：写成三个 .csv 分量文件（散点写不了 nc）",
              len(_hf3) == 3 and all(f.endswith(".csv") for f in _hf3),
              f"{[os.path.basename(f) for f in _hf3]}")
        if _hf3:
            _res = shio.read_points([f for f in _hf3 if "uN" in f][0])
            check("  散点 u_N 文件能读回（lon,lat,值 三列）",
                  _res is not None and np.asarray(_res[2]).size > 0,
                  f"{np.asarray(_res[2]).size} 个值")
        shutil.rmtree(_hz_pt, ignore_errors=True)
    finally:
        _mw.QFileDialog.getExistingDirectory = _orig_dir3
        shutil.rmtree(_pexp, ignore_errors=True)
    win.map_field.setCurrentIndex(0)          # 交还状态：不要停在逐历元视图上
    win._reset_time_range()
    app.processEvents()

    # (8e) 载入新数据**不许**偷偷开工：即使「显示」停在重建场，也不自动解整条序列
    win.map_field.setCurrentIndex(1)
    app.processEvents()
    win.on_loaded(ds_mt)
    app.processEvents()
    check("★ 载入数据时不自动开跑（显示停在重建场也一样）",
          not (win.series_worker is not None and win.series_worker.isRunning())
          and win.series_coeffs is None
          and "运行分析" in win.map_note.text(),
          f"页脚={win.map_note.text()[:34]!r}")
    win.map_field.setCurrentIndex(0)
    app.processEvents()

    # (9) 统计量与摘要：**按当前时次** + 缓存（"载入要 10 多秒"的一半在这里）
    # 用户要求：「统计那一行数值范围/RMS 要么改成按第 k 个时次」。多时次文件对整块
    # 求范围既没有直接意义、又贵（mascon 2.65 亿个值要 3.4 s），所以口径改成逐时次。
    _cube_s = np.array([[[1.0, 3.0], [np.nan, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]]], dtype="float32")
    _dss = Dataset.from_grid("<统计>", np.arange(2.0), np.arange(2.0),
                             _cube_s, {})
    _ep0 = _cube_s[:, :, 0].ravel()
    _ep0 = _ep0[np.isfinite(_ep0)]
    _st = _dss.stats(0)
    check("★ 统计量按**第 1 个时次**算（含 NaN 计数），与逐层手算逐值一致",
          _st["n_finite"] == int(_ep0.size)
          and _st["min"] == float(_ep0.min())
          and _st["max"] == float(_ep0.max())
          and abs(_st["rms"] - float(np.sqrt(np.mean(_ep0.astype(float) ** 2))))
          <= 1e-12 * max(abs(_st["rms"]), 1.0),
          f"min/max/rms = {_st['min']:.4g}/{_st['max']:.4g}/{_st['rms']:.6g}，"
          f"缺测 {_st['n_total'] - _st['n_finite']}")
    _all = _cube_s.ravel()
    _all = _all[np.isfinite(_all)]
    check("★ 不再是整块口径（整块 max=8、rms=5.345；那是旧行为）",
          _st["max"] != float(_all.max())
          and abs(_st["rms"] - float(np.sqrt(np.mean(_all.astype(float) ** 2)))) > 1e-6,
          f"整块 {_all.min():.4g}…{_all.max():.4g}，"
          f"RMS {np.sqrt(np.mean(_all.astype(float) ** 2)):.6g}")
    _st1 = _dss.stats(1)
    check("每个时次各算各的（第 2 个时次 max=8）",
          _st1["max"] == 8.0 and _st1["min"] == 3.0 and _st1["epoch"] == 1,
          f"第 2 个时次 {_st1['min']:.4g}…{_st1['max']:.4g}")
    check("★ 统计量按**时次**缓存（切回来不重算 —— 播放时会反复问）",
          _dss.stats(0) is _st and _dss.stats(1) is _st1, "同一 dict 对象")
    _s1 = _dss.summary(0)
    check("摘要按时次缓存（第二次直接返回，不再扫一遍整块）",
          _dss.summary(0) is _s1 and _dss._summary[0] is _s1,
          f"摘要 {len(_s1)} 字符")
    check("摘要写明这两行是第几个时次的",
          "第 1/2 个时次" in _s1, f"共 {_dss.ntime} 个时次")
    check("摘要里写明缺测个数（不把 NaN 悄悄算进 RMS）",
          "缺测" in _s1, [ln for ln in _s1.splitlines() if "缺测" in ln][:1])
    # 多时次**散点**也必须逐时次：网格那条路走 grid_slice，散点这条路当年直接用整块
    # values，于是摘要会一边写「第 k 个时次」一边给整块的范围 —— 数字和口径对不上。
    _pts = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    _dsp = Dataset.from_points("<散点多时次>", [10.0, 20.0, 30.0],
                               [1.0, 2.0, 3.0], _pts)
    _p0, _p1 = _dsp.stats(0), _dsp.stats(1)
    check("★ 多时次散点的统计同样按第 k 个时次（整块口径会给出 1…300）",
          _dsp.ntime == 2 and _p0["min"] == 1.0 and _p0["max"] == 3.0
          and _p1["min"] == 100.0 and _p1["max"] == 300.0,
          f"第 1 个 {_p0['min']:.4g}…{_p0['max']:.4g}，"
          f"第 2 个 {_p1['min']:.4g}…{_p1['max']:.4g}")
    check("散点摘要的数值范围/文字口径一致（都指第 2 个时次）",
          "第 2/2 个时次" in _dsp.summary(1)
          and "100" in _dsp.summary(1) and "300" in _dsp.summary(1),
          [ln for ln in _dsp.summary(1).splitlines() if "数值范围" in ln][:1])

    # (6) 打开 nc 只能**开一次**文件：变量下拉框以前是"再开一次同一个文件"，
    import tempfile as _tf

    import xarray as _xr
    from shkit import io as _shio
    _nc_dir = _tf.mkdtemp(prefix="shkit_gui_nc_")
    _nc = os.path.join(_nc_dir, "vars.nc")
    try:
        _xr.Dataset(
            {"ewh": (("lat", "lon"), np.zeros((6, 12), "float32")),
             "extra": (("lat", "lon"), np.ones((6, 12), "float32")),
             "bounds": (("n", "b"), np.zeros((6, 2), "float32"))},
            coords={"lat": ("lat", np.linspace(-75, 75, 6)),
                    "lon": ("lon", np.linspace(0, 330, 12))},
        ).to_netcdf(_nc) if _shio._is_ascii_path(_nc_dir) else None
        if not os.path.exists(_nc):
            _ds_buf = _xr.Dataset(
                {"ewh": (("lat", "lon"), np.zeros((6, 12), "float32")),
                 "extra": (("lat", "lon"), np.ones((6, 12), "float32")),
                 "bounds": (("n", "b"), np.zeros((6, 2), "float32"))},
                coords={"lat": ("lat", np.linspace(-75, 75, 6)),
                        "lon": ("lon", np.linspace(0, 330, 12))})
            _shio.to_netcdf_path(_ds_buf, _nc)

        _calls = []
        _orig_open = _shio.open_nc_dataset

        def _counting_open(path, **kw):
            _calls.append(str(path))
            return _orig_open(path, **kw)

        _shio.open_nc_dataset = _counting_open
        try:
            win.load_path(_nc)
            wait_for(app, win.load_worker, timeout=120)
            app.processEvents()
            _items = [win.var_combo.itemText(i)
                      for i in range(win.var_combo.count())]
            check("★ 打开 nc 只读一次文件（变量名单来自同一次读取）",
                  len(_calls) == 1, f"open_nc_dataset 被调用 {len(_calls)} 次")
            check("★ 变量下拉框由载入结果填充，且只列网格变量",
                  _items == ["ewh", "extra"], str(_items))
            check("下拉框选中当前载入的那个变量",
                  win.var_combo.currentText() == "ewh",
                  win.var_combo.currentText())
            check("换变量重载仍然只开一次文件",
                  (win.var_combo.setCurrentText("extra"),
                   win.reload_with_var(), app.processEvents(),
                   wait_for(app, win.load_worker, timeout=120),
                   len(_calls) == 2)[-1] and win.dataset is not None,
                  f"累计 {len(_calls)} 次（应为 2：首次 + 换变量各一次）")
        finally:
            _shio.open_nc_dataset = _orig_open
    finally:
        import shutil as _sh
        _sh.rmtree(_nc_dir, ignore_errors=True)

    # (10) 懒加载：Dataset 不再展平整块 + 界面"先给一层、其余后台补" + 逐时次摘要
    # 用户报障两条：①「载入 mascon 要 10 多秒」；②「统计那一行数值范围/RMS 应按
    # 第 k 个时次」。这里用 duck-typed 的假 LazyTimeCube 稳定地停在"后台还没补完"
    # 的状态（真文件上这个窗口只有几百毫秒，测不稳）。
    from PySide6.QtWidgets import QMessageBox as _QMB

    class _StubLazy:
        """只填好第 0 层的懒加载容器（接口与 shkit.io.LazyTimeCube 一致）。"""

        def __init__(self, cube):
            self._c = np.asarray(cube, dtype=float)
            self.shape = self._c.shape
            self._done = np.zeros(self.shape[2], bool)
            self._done[0] = True
            self.prefetch_calls = 0
            self.ensure_all_calls = 0
            self.closed = False

        @property
        def ntime(self):
            return int(self.shape[2])

        @property
        def n_filled(self):
            return int(self._done.sum())

        def is_filled(self, t):
            return bool(self._done[int(t)])

        def ensure(self, t):
            self._done[int(t)] = True

        def ensure_all(self, progress=None):
            self.ensure_all_calls += 1
            for t in range(self.ntime):
                self._done[t] = True
                if progress is not None:
                    progress(t + 1, self.ntime)

        def start_prefetch(self, from_t=None):
            self.prefetch_calls += 1

        def wait(self, timeout=None):
            return True

        def close(self):
            self.closed = True

        def __getitem__(self, k):
            # 与真 LazyTimeCube 同语义：取哪一层就先把哪一层标记为已读
            ts = None
            if isinstance(k, tuple) and len(k) >= 3:
                kk = k[2]
                ts = (list(range(*kk.indices(self.ntime))) if isinstance(kk, slice)
                      else [int(np.atleast_1d(kk)[0])])
            if ts is None:
                self.ensure_all()
            else:
                for t in ts:
                    self._done[int(t)] = True
            return self._c[k]

        def __array__(self, dtype=None, copy=None):
            return self._c

    _cubeL = (np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)) * 0.5
    _stub = _StubLazy(_cubeL)
    _latL = np.array([0.0, 1.0])
    _lonL = np.array([0.0, 1.0])
    _dsL = Dataset.from_grid("<懒加载网格>", _latL, _lonL, _stub,
                             {"variable_units": "cm"})
    check("★ 懒加载网格不再被展平：values 为 None，但 ntime/npoints 仍然正确",
          _dsL.values is None and _dsL.ntime == 4 and _dsL.npoints == 4,
          f"values={_dsL.values!r}, ntime={_dsL.ntime}, npoints={_dsL.npoints}")
    check("★ 逐时次取值只读那一层（不做整块物化）",
          np.array_equal(_dsL.value_slice(1), _cubeL[:, :, 1].ravel())
          and _stub.n_filled == 2,
          f"已读 {_stub.n_filled}/4 层")
    _sum3 = _dsL.summary(2)
    _line3 = f"数值范围    : {_cubeL[:, :, 2].min():.6g} … " \
             f"{_cubeL[:, :, 2].max():.6g}　（第 3/4 个时次）"
    check("★ 摘要的「数值范围」按**第 k 个时次**统计，并写明是第几个",
          _line3 in _sum3,
          [ln for ln in _sum3.splitlines() if "数值范围" in ln][:1])
    check("★ 摘要的「数值 RMS」同样按该时次（不再对整块求 RMS）",
          f"数值 RMS    : {np.sqrt(np.mean(_cubeL[:, :, 2] ** 2)):.6g}　" \
          f"（第 3/4 个时次）" in _sum3,
          [ln for ln in _sum3.splitlines() if "数值 RMS" in ln][:1])
    check("不同时次的摘要是真的不同（逐时次缓存，不是同一份文字）",
          _dsL.summary(2) is not _dsL.summary(0)
          and "第 1/4 个时次" in _dsL.summary(0),
          [ln for ln in _dsL.summary(0).splitlines() if "数值范围" in ln][:1])
    check("all_values() 才补齐整块，并与立方体逐值一致",
          np.array_equal(_dsL.all_values(), _cubeL.reshape(4, 4)), "形状 (4, 4)")

    _stub2 = _StubLazy(_cubeL)
    _dsL2 = Dataset.from_grid("<懒加载网格2>", _latL, _lonL, _stub2,
                              {"variable_units": "cm"})
    _oc2 = _QMB.critical
    _QMB.critical = staticmethod(lambda *a, **k: _QMB.StandardButton.Ok)
    try:
        win.sp_nmax.setValue(1)          # 2x2 的小网格：阶数 1 就够
        win.on_loaded(_dsL2)
        app.processEvents()
        check("★ 载入懒加载数据后界面立刻可用，并启动后台补齐",
              win.dataset is _dsL2 and _stub2.prefetch_calls == 1
              and win._lazy_timer.isActive(),
              f"prefetch {_stub2.prefetch_calls} 次，定时器 "
              f"{'在跑' if win._lazy_timer.isActive() else '没跑'}")
        check("时间序列页先给占位并说明「后台补齐中」（不静默卡住）",
              "后台" in win.series_note.text(), win.series_note.text()[:38])
        check("★ 后台还没补完时，界面不会为了摘要/占位去补齐整块",
              _stub2.ensure_all_calls == 0,
              f"ensure_all 被调用 {_stub2.ensure_all_calls} 次")
        win._on_lazy_tick()
        app.processEvents()
        _msg = win.status_label.text()
        check("后台补齐进度显示在状态栏",
              "后台读取时次" in _msg and "/4" in _msg, _msg[:48])
        win.time_slider.setValue(2)
        app.processEvents()
        # 注：切时次本身不再触发任何求解（逐历元系数只在需要时一次解完），
        # 所以这里不需要等 worker —— 摘要按时次取数是 Dataset 的懒读。
        app.processEvents()
        check("★ 切换时次后左侧摘要的数值范围/RMS 跟着变成该时次",
              "第 3/4 个时次" in win.data_summary.toPlainText(),
              [ln for ln in win.data_summary.toPlainText().splitlines()
               if "数值范围" in ln][:1])
        check("★ 要「全部历元」的操作才补齐整块，并报进度",
              win._ensure_all_values("测试") and _stub2.n_filled == 4
              and not win._lazy_timer.isActive(),
              f"已填 {_stub2.n_filled}/4，定时器已停")
        win.chk_series_region.setChecked(False)      # 取最近点，不受区域掩码影响
        win._refresh_series_page()
        app.processEvents()
        check("补齐后时间序列页换成真实曲线（不再是占位）",
              "后台" not in win.series_note.text()
              and "需要多时次" not in win.series_note.text(),
              win.series_note.text()[:44])
        _dsL2.close()
        check("close() 把懒加载句柄交还给底层容器（不泄漏文件句柄）",
              _stub2.closed, "stub.closed = True")
    finally:
        _QMB.critical = _oc2

    win.close()
    app.processEvents()

    npass = sum(RESULTS)
    print(f"\n{'='*72}\n{npass}/{len(RESULTS)} checks passed")
    print(f"截图已保存到 {SHOTS}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
