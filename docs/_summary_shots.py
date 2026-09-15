# -*- coding: utf-8 -*-
"""Regenerate current GUI screenshots for the method-summary document.

The checked-in shots in ``tests/_gui_shots/`` are unusable: that offscreen Qt run
had no font directory, so every Chinese glyph came out as a tofu box.  Here Qt is
pointed at the Windows font directory and SimHei is set explicitly.

Nothing under shkit/, tests/ or tools/ is touched, and the checked-in
screenshots are left alone: everything lands in docs/SHKit方法总结_figs/.
"""
import json
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

SHKIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SHKIT)
os.chdir(SHKIT)

import numpy as np                                             # noqa: E402
from PySide6.QtGui import QFont                                # noqa: E402
from PySide6.QtWidgets import QApplication                     # noqa: E402

FIGS = os.path.join(SHKIT, "docs", "SHKit方法总结_figs")
os.makedirs(FIGS, exist_ok=True)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:52s} {detail}", flush=True)


app = QApplication.instance() or QApplication([])
app.setFont(QFont("SimHei", 10))

from shkit.gui.main_window import MainWindow                   # noqa: E402


def wait(worker, timeout=900.0):
    t0 = time.time()
    while worker is not None and worker.isRunning() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return not (worker is not None and worker.isRunning())


def shot(win, name):
    app.processEvents()
    p = os.path.join(FIGS, name)
    win.grab().save(p)
    print(f"    saved {name}  ({os.path.getsize(p)/1024:.0f} kB)", flush=True)
    return p


def pick(combo, value):
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return True
    return False


YANGTZE = os.path.join(SHKIT, "Yangtze_River.txt")

# ------------------------------------------------------------------- demo
print("== A. demo dataset (global Fibonacci 2500, truth band-limited 12 + 1%) ==")
win = MainWindow()
win.resize(1600, 960)
win.show()
app.processEvents()
win.make_demo()
app.processEvents()
check("demo loaded", win.dataset is not None and win.dataset.npoints > 0,
      f"{win.dataset.npoints} points, spinner rule="
      f"{win.cb_rule.currentData()}")
win.tabs.setCurrentIndex(0)
shot(win, "gui_01_demo_main.png")

win.run_analysis()
wait(win.analysis_worker)
check("demo analysis finished", win.coeffs is not None,
      f"method={win.report.method}, rule={win.report.weight_rule}, "
      f"nmax={win.coeffs.nmax}")
demo = {"method": win.report.method, "rule": win.report.weight_rule,
        "n_warn": len(win.report.warnings),
        "npoints": int(win.dataset.npoints),
        "suggested_rule": win.dataset.suggested_rule()}
win.tabs.setCurrentIndex(1)
shot(win, "gui_02_demo_spectrum.png")
win.tabs.setCurrentIndex(2)
shot(win, "gui_03_demo_report.png")
win.tabs.setCurrentIndex(3)
shot(win, "gui_04_demo_coeffs.png")
win.tabs.setCurrentIndex(0)
shot(win, "gui_05_demo_map.png")

# ------------------------------------------------- Yangtze: projection/lattice
print()
print("== B. Yangtze mask, method=projection (auto rule -> lattice) ==")
w2 = MainWindow()
w2.resize(1600, 960)
w2.show()
app.processEvents()
w2.load_path(YANGTZE)
wait(w2.load_worker)
check("Yangtze loaded", w2.dataset is not None,
      f"{w2.dataset.npoints} points, suggested rule="
      f"{w2.dataset.suggested_rule()}")
w2.sp_nmax.setValue(12)
check("method combo offers 'projection'", pick(w2.cb_method, "projection"), "")
check("rule combo offers 'lattice'", pick(w2.cb_rule, "lattice"), "")
app.processEvents()
w2.run_analysis()
wait(w2.analysis_worker)
proj = {"method": w2.report.method, "rule": w2.report.weight_rule,
        "c00": float(w2.coeffs.C[0, 0]),
        "n_warn": len(w2.report.warnings),
        "warn_c00": any("disagrees with the weighted mean" in m
                        for m in w2.report.warnings),
        "resid": float(w2.report.residual_rms),
        "resid_w": float(w2.report.residual_rms_weighted)}
check("projection ran", w2.coeffs is not None,
      f"method={proj['method']}, rule={proj['rule']}, C00={proj['c00']:.6e}, "
      f"warn_c00={proj['warn_c00']}")
w2.tabs.setCurrentIndex(2)
shot(w2, "gui_06_yangtze_projection_report.png")
w2.tabs.setCurrentIndex(1)
shot(w2, "gui_07_yangtze_projection_spectrum.png")
w2.tabs.setCurrentIndex(0)
shot(w2, "gui_08_yangtze_projection_map.png")

# ------------------------------------------------------- Yangtze: cg (broken)
print()
print("== C. Yangtze mask, method=cg (the path the old .gfc used) ==")
check("method combo offers 'cg'", pick(w2.cb_method, "cg"), "")
app.processEvents()
w2.run_analysis()
wait(w2.analysis_worker)
cg = {"method": w2.report.method, "rule": w2.report.weight_rule,
      "c00": float(w2.coeffs.C[0, 0]),
      "n_warn": len(w2.report.warnings),
      "warn_c00": any("disagrees with the weighted mean" in m
                      for m in w2.report.warnings),
      "warn_collapsed": any("collapsed" in m for m in w2.report.warnings),
      "resid": float(w2.report.residual_rms)}
check("cg ran and the C00 guard fired", cg["warn_c00"],
      f"C00={cg['c00']:.6e}, residual={cg['resid']:.3e}, "
      f"{cg['n_warn']} warning(s)")
w2.tabs.setCurrentIndex(2)
shot(w2, "gui_09_yangtze_cg_report.png")
w2.tabs.setCurrentIndex(0)
shot(w2, "gui_10_yangtze_cg_map.png")

meta = {"demo": demo, "projection": proj, "cg": cg}
with open(os.path.join(FIGS, "_gui_meta.json"), "w", encoding="utf-8") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=1)
print(json.dumps(meta, ensure_ascii=False, indent=1))
npass = sum(1 for ok in RESULTS if ok)
print(f"\n{npass}/{len(RESULTS)} checks passed")
sys.exit(0 if npass == len(RESULTS) else 1)
