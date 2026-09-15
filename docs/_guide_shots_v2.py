# -*- coding: utf-8 -*-
"""Regenerate the GUI screenshots used by 使用说明.html / 使用说明_GUI.md.

Why this is separate from ``build_guide.py``
--------------------------------------------
``build_guide.py`` captures the *dialogs* and the two dock close-ups (that is what
its ``--only-panels`` / ``--only-dialogs`` modes are for).  The **main-window**
shots come from here, because they need real data in the window: a Fibonacci demo
run, a 24-epoch synthetic grid with dates, and the Yangtze mask in two methods.

The checked-in shots in ``docs/使用说明_img/gui_*.png`` were taken *before* v2.0
added five tabs (数值表 / 逐历元诊断 / 时间序列 / 趋势与周年 / 水平形变), so the
shipped guide was illustrating a window that no longer exists.  Running this
script replaces them with shots of the current UI.

Run:  python docs/_guide_shots_v2.py          （约 1 分钟，离线）
"""
import json
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

HERE = os.path.dirname(os.path.abspath(__file__))
SHKIT = os.path.dirname(HERE)
sys.path.insert(0, SHKIT)
os.chdir(SHKIT)

import numpy as np                                             # noqa: E402
from PySide6.QtGui import QFont                                # noqa: E402
from PySide6.QtWidgets import QApplication                     # noqa: E402

IMG = os.path.join(HERE, "使用说明_img")
os.makedirs(IMG, exist_ok=True)
RESULTS = []
SAVED = {}


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}", flush=True)


app = QApplication.instance() or QApplication([])
app.setFont(QFont("SimHei", 10))

from shkit.gui.dataset import Dataset                           # noqa: E402
from shkit.gui.main_window import MainWindow                     # noqa: E402


def wait(worker, timeout=900.0):
    t0 = time.time()
    while worker is not None and worker.isRunning() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return not (worker is not None and worker.isRunning())


def shot(win, name):
    for _ in range(3):
        app.processEvents()
    p = os.path.join(IMG, name)
    win.grab().save(p)
    SAVED[name] = os.path.getsize(p)
    print(f"    saved {name}  ({os.path.getsize(p)/1024:.0f} kB)", flush=True)


def pick(combo, value):
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return True
    return False


def new_win(w=1600, h=960):
    win = MainWindow()
    win.resize(w, h)
    win.show()
    app.processEvents()
    return win


YANGTZE = os.path.join(SHKIT, "Yangtze_River.txt")



def _run() -> int:
    """抓图主流程（整段包在函数里，理由见 main()）。"""
    # --------------------------------------------------------------- A. 示例数据
    print("== A. 内置示例（2500 点 Fibonacci，真值带限 12 阶 + 1% 噪声）==")
    win = new_win()
    win.make_demo()
    app.processEvents()
    check("示例数据已载入", win.dataset is not None and win.dataset.npoints > 0,
          f"{win.dataset.npoints} 点")
    win.tabs.setCurrentIndex(0)
    win._refresh_map()
    shot(win, "gui_01_demo_main.png")

    win.run_analysis()
    wait(win.analysis_worker)
    check("示例数据分析完成", win.coeffs is not None,
          f"method={win.report.method} rule={win.report.weight_rule} "
          f"nmax={win.coeffs.nmax}")
    TABS = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    check("页签数 = 9（v2.0）", len(TABS) == 9, str(TABS))

    win.tabs.setCurrentIndex(TABS.index("逐阶谱"))
    shot(win, "gui_02_demo_spectrum.png")
    win.tabs.setCurrentIndex(TABS.index("诊断报告"))
    shot(win, "gui_03_demo_report.png")
    win.tabs.setCurrentIndex(TABS.index("系数统计"))
    shot(win, "gui_04_demo_coeffs.png")
    win.tabs.setCurrentIndex(TABS.index("地图"))
    win._refresh_map()
    shot(win, "gui_05_demo_map.png")

    # D3 数值表（懒加载：先标脏再切页）
    win._values_dirty = True
    win.tabs.setCurrentIndex(TABS.index("数值表"))
    app.processEvents()
    check("数值表已填充", win.values_table.rowCount() > 0
          and win.values_table.columnCount() > 1,
          f"{win.values_table.rowCount()}×{win.values_table.columnCount()}")
    shot(win, "gui_11_values.png")

    # B2 水平形变（需要已有的系数）
    win.tabs.setCurrentIndex(TABS.index("水平形变"))
    app.processEvents()
    win._refresh_horizontal()
    # ★ v2.0.1 起水平形变在 worker 线程/子进程里算（散点上几十秒，同步做会冻窗口），
    # 所以这里要**等它算完**再截图，否则拍到的是占位图。
    wait(win.horiz_worker, timeout=600)
    app.processEvents()
    check("水平形变已绘制", win.horiz_canvas._drawn is not None,
          f"{len(win.horiz_canvas.figure.axes)} 个 axes")
    shot(win, "gui_15_horizontal.png")

    # --------------------------------------------------- B. 多时次（合成带日期）
    print()
    print("== B. 24 个历元的合成网格（带日期）：趋势 + 年周期 ==")
    lat = np.arange(-80.0, 80.01, 10.0)
    lon = np.arange(0.0, 360.0, 15.0)
    nt = 24
    tv = np.datetime64("2002-01-18") + np.arange(nt) * 30
    t = np.arange(nt) / 12.0
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    # 两个「信号源」：南亚（趋势）与亚马逊（年周期），数值量级 ~0.1
    asia = 0.9 * np.exp(-((LA - 25.0) / 18.0) ** 2 - ((LO - 80.0) / 30.0) ** 2)
    amaz = 0.8 * np.exp(-((LA + 5.0) / 16.0) ** 2 - ((LO + 60.0) / 25.0) ** 2)
    cube = (asia[:, :, None] * 0.30 * t[None, None, :]
            + amaz[:, :, None] * 0.35 * np.cos(2 * np.pi * (t[None, None, :] - 0.25))
            + 0.005 * np.random.default_rng(5).normal(size=(lat.size, lon.size, nt)))
    # 人为把一个历元的噪声放大，好让「逐历元诊断」的离群高亮有东西可标
    cube[:, :, 7] += np.random.default_rng(6).normal(
        size=(lat.size, lon.size)) * 0.40
    ds = Dataset.from_grid("<合成多时次>", lat, lon, cube, {"time": [str(x)[:10] for x in tv]})
    check("时间轴可建起来且带日期",
          ds.time_axis() is not None and ds.time_axis().has_dates,
          f"ntime={ds.ntime}，首个历元 {ds.epoch_label(1)}")

    win.on_loaded(ds)
    win._reset_time_range()
    win.time_slider.setValue(9)
    win._refresh_map()
    app.processEvents()
    shot(win, "gui_16_multitime_map.png")

    # D5 逐历元诊断（批量，走 worker）
    win.tabs.setCurrentIndex(TABS.index("逐历元诊断"))
    win.sp_nmax.setValue(8)
    app.processEvents()
    win.run_series_analysis()
    ok = wait(win.series_worker, timeout=600)
    check("批量逐历元分析完成", ok and win.epochs_table.rowCount() == nt,
          f"{win.epochs_table.rowCount()} 行")
    check("检出被放大的那个历元（第 8 个）",
          [i + 1 for i in range(win.epochs_table.rowCount())
           if win.epochs_table.item(i, 5).text()] == [8],
          win.epochs_note.text()[-30:])
    app.processEvents()
    shot(win, "gui_12_epochs.png")

    # D6 时间序列（单点）
    TABS = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    win.tabs.setCurrentIndex(TABS.index("时间序列"))
    app.processEvents()
    win.chk_series_region.setChecked(False)
    win.sp_series_lat.setValue(25.0)
    win.sp_series_lon.setValue(80.0)
    app.processEvents()
    check("时间序列已画出", len(win.series_canvas.figure.axes[0].lines) >= 1,
          win.series_note.text()[:50])
    shot(win, "gui_13_series.png")

    # D7 趋势与周年（勾上「含半年周期」：说明书要展示 5 张图那一档）
    win.tabs.setCurrentIndex(TABS.index("趋势与周年"))
    app.processEvents()
    win.chk_trend_seasonal.setChecked(True)
    win.chk_trend_coast.setChecked(True)
    app.processEvents()
    win._compute_trend_maps()
    app.processEvents()
    check("趋势/振幅/相位图（含半年 → 5 张、每张都有海岸线）",
          len(win.trend_canvas._axes) == 5
          and all(len(a.get_lines()) >= 1 for a in win.trend_canvas._axes),
          win.trend_note.text()[:50])
    shot(win, "gui_14_trend.png")

    meta = {"tabs": TABS, "ntime": int(nt), "saved": SAVED}
    with open(os.path.join(HERE, "_guide_shots_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    # ------------------------------------------------ C. 长江掩膜：projection / cg
    print()
    print("== C. 长江流域掩膜（142 846 点）：projection 与 cg ==")
    w2 = new_win()
    w2.load_path(YANGTZE)
    wait(w2.load_worker)
    check("长江掩膜已载入", w2.dataset is not None, f"{w2.dataset.npoints} 点")
    w2.sp_nmax.setValue(12)
    check("方法可选 projection", pick(w2.cb_method, "projection"), "")
    check("规则可选 lattice", pick(w2.cb_rule, "lattice"), "")
    app.processEvents()
    w2.run_analysis()
    wait(w2.analysis_worker)
    check("projection 跑完", w2.coeffs is not None,
          f"method={w2.report.method} rule={w2.report.weight_rule}")
    T2 = [w2.tabs.tabText(i) for i in range(w2.tabs.count())]
    w2.tabs.setCurrentIndex(T2.index("诊断报告"))
    shot(w2, "gui_06_yangtze_projection_report.png")
    w2.tabs.setCurrentIndex(T2.index("逐阶谱"))
    shot(w2, "gui_07_yangtze_projection_spectrum.png")
    w2.tabs.setCurrentIndex(T2.index("地图"))
    w2._refresh_map()
    shot(w2, "gui_08_yangtze_projection_map.png")

    check("方法可选 cg", pick(w2.cb_method, "cg"), "")
    app.processEvents()
    w2.run_analysis()
    wait(w2.analysis_worker)
    check("cg 跑完且 C00 守卫报警", w2.coeffs is not None
          and any("weighted mean" in m for m in w2.report.warnings),
          f"C00={float(w2.coeffs.C[0, 0]):.3e}")
    w2.tabs.setCurrentIndex(T2.index("诊断报告"))
    shot(w2, "gui_09_yangtze_cg_report.png")
    w2.tabs.setCurrentIndex(T2.index("地图"))
    w2._refresh_map()
    shot(w2, "gui_10_yangtze_cg_map.png")

    npass = sum(1 for ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} checks passed，写出 {len(SAVED)} 张图到 使用说明_img/")
    return 0 if npass == len(RESULTS) else 1


def main() -> int:
    """入口。

    ⚠️ 必须有这层 `if __name__ == '__main__'` 保护：GUI 的批量分析在 Windows 上
    用 `spawn` 起子进程，而 spawn 会**重新导入主模块**。没有保护时（脚本里连
    "__main__" 这个字符串都不出现）`spawn_available()` 会判定"可以 spawn"，
    于是子进程把本脚本从头再跑一遍 —— 表现为批量分析永远拿不到结果（0 行），
    并在退出时崩溃（QThread 被销毁 / 0xC0000409）。
    """
    return _run()


if __name__ == "__main__":
    sys.exit(main())
