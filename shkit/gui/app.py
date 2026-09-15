# -*- coding: utf-8 -*-
"""
shkit.gui.app
=============

Application bootstrap for the SHKit desktop GUI.

Run it with::

    python -m shkit.gui
    # or, after `pip install -e .`:
    shkit-gui

Graphics libraries
------------------
Only **PySide6-Essentials** is required.  Qt is used under **LGPLv3**, which
permits closed-source commercial distribution provided the library stays
dynamically linked and can be replaced by the user; this application never
modifies Qt and never statically links it.  The GPL-only Qt modules (Qt Charts,
Qt Data Visualization, ...) are deliberately *not* used - all plotting goes
through matplotlib.  See ``docs/许可与闭源商用说明.md``.
"""

from __future__ import annotations

import os
import sys
import traceback

__all__ = ["main", "check_pyside", "MISSING_HINT"]

MISSING_HINT = """\
SHKit 的图形界面需要 PySide6。

请安装 **Essentials** 版本（不是完整的 PySide6）：

    pip install PySide6-Essentials

为什么不装完整的 PySide6：它还会带上 PySide6-Addons，其中 Qt Charts、
Qt Data Visualization 等模块是 GPL-only，会让闭源商用分发变得不可能。
Essentials 只有 LGPLv3 模块，配套 matplotlib 绘图即可。
"""


def check_pyside() -> tuple:
    """Return ``(ok, message)`` without importing the GUI modules."""
    try:
        import PySide6  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"未找到 PySide6（{type(exc).__name__}: {exc}）。\n\n{MISSING_HINT}"
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, (f"未找到 matplotlib（{type(exc).__name__}: {exc}）。\n"
                       "绘图需要 matplotlib（BSD 风格许可，可闭源商用）：\n"
                       "    pip install matplotlib")
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, (f"matplotlib 的 Qt 后端不可用（{type(exc).__name__}: {exc}）。\n"
                       "请把 matplotlib 升级到 3.5 以上。")
    return True, "ok"


def main(argv=None) -> int:
    """Start the GUI.  Returns a process exit code."""
    argv = list(sys.argv if argv is None else argv)

    # D12: the batch analysis can run in a child process.  Under a frozen
    # (PyInstaller) bundle that is only safe if the runtime hook runs first, so
    # this must happen before any child is spawned.  No-op when not frozen.
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:                                            # noqa: BLE001
        pass

    # 自检模式：打包后用来验证冻结版真的能起来（建窗口、画地图、读说明书），
    # 而不是只看进程有没有活着。见 packaging/shkit_launcher.py。
    self_test = "--self-test" in argv

    ok, msg = check_pyside()
    if not ok:
        print(msg, file=sys.stderr)
        return 2

    # Headless / offscreen support: QT_QPA_PLATFORM=offscreen lets the GUI be
    # smoke-tested in CI without a display.
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("SHKit")
    app.setOrganizationName("SHKit")
    from .. import __version__
    app.setApplicationVersion(__version__)

    # keep the app alive when a slot raises
    def _hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
    sys.excepthook = _hook

    from .main_window import MainWindow
    win = MainWindow()
    win.show()

    if self_test:
        return _self_test(win, app)

    # optional command-line convenience: open a file straight away
    rest = [a for a in argv[1:] if not a.startswith("-")]
    if rest and os.path.exists(rest[0]):
        win.load_path(rest[0])

    return app.exec()


def _self_test(win, app) -> int:
    """跑一遍关键路径后退出：0 = 通过。给打包脚本与 CI 用。

    刻意走得深一点：建窗口 → 生成示例数据 → 跑真实分析线程 → 画地图/谱图/报告
    → 渲染说明书 HTML（含图片）→ 导出系数。任何一个环节在冻结环境里缺依赖，
    都会在这里变成非 0 退出码，而不是等用户点下去才崩。
    """
    import time

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QTextDocument

    from .main_window import GuideDialog, guide_html_path

    # 冻结版（windowed）的 stdout 很可能是 GBK 管道，日志会变乱码；直接强制 UTF-8，
    # 不依赖外部环境变量（打包脚本里设 PYTHONIOENCODING 实测对冻结版无效）。
    for _stream in ("stdout", "stderr"):
        try:
            getattr(sys, _stream).reconfigure(encoding="utf-8")
        except Exception:                              # noqa: BLE001
            pass

    failures = []

    def step(name, fn):
        try:
            detail = fn()
            print(f"[PASS] {name}" + (f"  {detail}" if detail else ""))
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    def wait_worker(timeout=300.0):
        t0 = time.time()
        while (win.analysis_worker is not None and win.analysis_worker.isRunning()
               and time.time() - t0 < timeout):
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    step("主窗口显示", lambda: f"{win.width()}x{win.height()}")
    step("生成示例数据", lambda: (win.make_demo(), f"{win.dataset.npoints} 点")[1])
    step("运行分析", lambda: (win.run_analysis(), wait_worker(),
                              f"method={win.report.method}")[2])
    step("渲染地图", lambda: (win.tabs.setCurrentIndex(0), win._refresh_map(), "ok")[2])
    step("渲染逐阶谱", lambda: (win.tabs.setCurrentIndex(1),
                                win._fill_spectrum(), "ok")[2])
    # 报告与系数表在分析完成时已自动填充，这里只确认真的填上了
    step("诊断报告已填充", lambda: (
        win.tabs.setCurrentIndex(2),
        f"{len(win.report_view.toPlainText())} 字符")[1])
    step("系数表已填充", lambda: (win.tabs.setCurrentIndex(3),
                                  f"{win.coeff_table.rowCount()} 行")[1])

    def guide():
        path = guide_html_path()
        if not path:
            raise FileNotFoundError("找不到说明书 HTML")
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
        # 说明书在 Qt 里要先剥掉 HTML 外壳（<head> 会让图片行高错乱），
        # 这里复用界面上那套处理，确保打包后的渲染路径和真实使用一致。
        dlg = GuideDialog(win)
        fitted, n = dlg._fit_images(dlg._for_qt(html), dlg._fit_width() or 1000)
        doc = QTextDocument()
        doc.setBaseUrl(QUrl.fromLocalFile(os.path.dirname(path) + os.sep))
        doc.setHtml(fitted)
        img = doc.resource(QTextDocument.ResourceType.ImageResource,
                           QUrl("使用说明_img/gui_01_demo_main.png"))
        if img is None or img.isNull():
            raise RuntimeError("说明书配图读不出来")
        dlg.close()
        return f"{n} 张图，正文 {len(doc.toPlainText())} 字符"

    step("渲染说明书（含配图）", guide)

    # ---- 安装/数据路径里的**空格**（用户明确要求：打包时验证这一条） ----------
    # 冻结版常常装在 `C:\Program Files\SHKit` 或 `%LOCALAPPDATA%\Programs\...`，
    # 用户名/目录名里也可能带空格。任何"把路径拼进命令字符串"或"没引号的相对
    # 路径"都会在这里露出来 —— 所以这里在**带空格**（还带中文）的目录里走一遍
    # 写→读→逐值比对→本地缓存，而不是只检查进程有没有活着。
    _spaced_state = {}

    def spaces_nc():
        import tempfile as _tf

        import numpy as _np

        from .. import io as _shio
        base = os.path.join(_tf.gettempdir(), "shkit 自检 space")
        os.makedirs(base, exist_ok=True)
        lat = _np.arange(-80.0, 80.01, 20.0)
        lon = _np.arange(0.0, 360.0, 30.0)
        g = _np.random.default_rng(3).normal(size=(lat.size, lon.size, 3))
        p = os.path.join(base, "网格 文件.nc")
        _shio.write_grid(p, lat, lon, g, var="value",
                         meta={"time": ["2002-01-18", "2002-02-17",
                                        "2002-03-19"]},
                         long_name="selftest")
        _la, _lo, back, _meta = _shio.read_grid(p)
        d = float(_np.nanmax(_np.abs(_np.asarray(back, dtype=float) - g)))
        if d != 0.0:
            raise RuntimeError(f"往返不一致 max|Δ| = {d}")
        _spaced_state.update(base=base, path=p, lat=lat, lon=lon, cube=g)
        return (f"{os.path.basename(p)} 逐值一致；"
                f"可执行文件 {sys.executable}"
                f"（含空格={(' ' in sys.executable)}）")

    step("带空格路径：网格 nc 写→读逐值一致", spaces_nc)

    def spaces_cache():
        import numpy as _np

        from ..coeffs import SHCoeffs
        from . import coeff_cache
        from .dataset import Dataset
        st = _spaced_state
        if not st:
            raise RuntimeError("前一步没有跑成")
        ds = Dataset.from_grid(st["path"], st["lat"], st["lon"], st["cube"],
                               {"time": ["2002-01-18", "2002-02-17",
                                         "2002-03-19"]})
        params = {"nmax": 4, "gaussian_km": 0.0, "field_unit": "scalar"}
        # 缓存装的是**逐历元**系数（3-D），与真实用途一致
        C = _np.zeros((5, 5, 3))
        S = _np.zeros((5, 5, 3))
        C[0, 0, :] = [1.0, 2.0, 3.0]
        co3 = SHCoeffs(C, S, {"field_unit": "scalar"}, ds.time_axis())
        coeff_cache.clear(ds)
        path, note = coeff_cache.save(ds, params, co3, None)
        if not path:
            raise RuntimeError(f"缓存没写出来：{note}")
        hit = coeff_cache.load(ds, params)
        if hit is None:
            raise RuntimeError("写出去的缓存读不回来（键不一致？）")
        same = bool(_np.array_equal(hit["coeffs"].C, C))
        coeff_cache.clear(ds)
        if not same:
            raise RuntimeError("缓存读回来的系数与写出去的不一致")
        return f"{os.path.basename(path)}：{note.split('（')[-1].rstrip('）')}"

    step("带空格路径：逐历元系数缓存写→读命中", spaces_cache)

    def child_process():
        """冻结版里的**独立子进程**（D12 的崩溃隔离）必须真的能起来。

        打包后这条最容易坏：进程入口变成 SHKit.exe，spawn 要重新拉起自己，
        路径里的空格、`freeze_support()` 有没有先调、队列能不能建，任何一处
        不对都会变成"子进程异常退出"。
        """
        import numpy as _np

        from .subproc import run_in_child, spawn_available
        ok, why = spawn_available()
        if not ok:
            raise RuntimeError(f"spawn_available 说不可用：{why}")
        lat = _np.array([-40.0, -20.0, 0.0, 20.0, 40.0])
        lon = _np.array([0.0, 90.0, 180.0, 270.0, 350.0])
        payload = {"lat": _np.repeat(lat, 4), "lon": _np.tile(lon, 4),
                   "values": _np.arange(20.0).reshape(-1, 1),
                   "nmax": 2, "method": "quadrature", "rule": "uniform",
                   "field_unit": "scalar", "report_fit": False}
        proc, q = run_in_child("analyze_series", payload)
        import queue as _queue
        import time as _time
        res = err = None
        t0 = _time.time()
        while _time.time() - t0 < 120:
            app.processEvents()
            try:
                kind, body = q.get(timeout=0.2)
            except _queue.Empty:
                if not proc.is_alive() and not q.qsize():
                    break
                continue
            if kind == "r":
                res = body
                break
            if kind == "e":
                err = body
                break
        if proc.is_alive():
            proc.terminate()
        if err is not None:
            raise RuntimeError(f"子进程报错：{err[0]}")
        if res is None:
            raise RuntimeError("子进程没有返回结果（也没有报错）")
        coeffs, _rep = res
        return (f"子进程 pid={proc.pid} 解出 nmax={coeffs.nmax}，"
                f"可执行文件 {os.path.basename(sys.executable)}")

    step("独立子进程能起来并返回结果", child_process)

    def export():
        from .. import io as shio
        out = os.path.join(os.environ.get("TEMP", "."), "_shkit_selftest.sh")
        shio.write_coeffs(win.coeffs, out)
        size = os.path.getsize(out)
        os.remove(out)
        return f"{size} 字节"

    step("导出系数文件", export)

    win.close()
    app.processEvents()
    print(f"\n自检结果：{'全部通过' if not failures else str(len(failures)) + ' 项失败'}")
    for f in failures:
        print("  -", f)
    return 0 if not failures else 1


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
