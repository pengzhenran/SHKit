# -*- coding: utf-8 -*-
"""Supplementary: (a) font-fallback check, (b) the two states of the .gfc.

(a) SimHei lacks U+00B2 (²), U+2212 (−) and U+1D40 (ᵀ).  matplotlib 3.10 does
    per-glyph fallback through the font.sans-serif list, so listing Microsoft
    YaHei after SimHei recovers ² and −.  ᵀ exists in neither, so it is written
    as ^T in every label.
(b) ``yantze_shkit_coeffs.gfc`` was regenerated during this session (mtime
    2026-09-12 17:55:55) and now holds a ``method=projection`` solution with
    ``weight_rule=delaunay``.  The original file it replaced held a
    ``method=cg`` / ``reg=None`` solution.  We therefore (i) record the current
    file, and (ii) REPRODUCE the original by re-running that exact configuration,
    so the document quotes a live computation rather than a lost file.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

SHKIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SHKIT)
FIGS = os.path.join(SHKIT, "docs", "SHKit方法总结_figs")

from shkit import io, weights as W                             # noqa: E402
from shkit.analysis import analysis_wlsq, analysis_projection  # noqa: E402
from shkit.filters import EARTH_RADIUS_M as R                  # noqa: E402

H = {}

# ---------------------------------------------------------------- (a) fonts
import warnings                                                # noqa: E402
fig, ax = plt.subplots(figsize=(3, 1))
ax.text(0.05, 0.5, "km² −1.2e−4 AᵀWA (L+1)² 中文测试")
ax.axis("off")
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    fig.savefig(os.path.join(FIGS, "_fontcheck.png"), dpi=100)
plt.close(fig)
glyph_warn = [str(w.message) for w in wlist if "Glyph" in str(w.message)]
H["font_warnings"] = glyph_warn
print("font glyph warnings:", glyph_warn or "(none)")

# ------------------------------------------------------------ (b) the .gfc
p = np.genfromtxt(os.path.join(SHKIT, "Yangtze_River.txt"), delimiter="\t",
                  names=True, encoding="utf-8")
lat = np.asarray(p["latitude"], float)
lon = np.asarray(p["longitude"], float)
f = np.asarray(p["value"], float)
LY = 12
ws_lat = W.compute_weights(lat, lon, rule="lattice")
ws_del = W.compute_weights(lat, lon, rule="delaunay")
frac_lat = ws_lat.total / W.FOUR_PI
frac_del = ws_del.total / W.FOUR_PI

cur = io.read_coeffs(os.path.join(SHKIT, "yantze_shkit_coeffs.gfc"))
hdr = cur.meta.get("gfc_header", {})
H["current_file"] = {
    "modelname": hdr.get("modelname"), "method": hdr.get("method"),
    "weight_rule": hdr.get("weight_rule"),
    "weight_sum": hdr.get("weight_sum"),
    "max_degree": hdr.get("max_degree"),
    "target": hdr.get("target"), "tau": hdr.get("tau"),
    "c00": float(cur.C[0, 0]),
    "own_weight_sum": float(hdr.get("weight_sum", "nan") or "nan"),
    "dc_expected_from_own_weights": (
        float(hdr.get("weight_sum", "nan") or "nan") / W.FOUR_PI),
    "ratio_vs_lattice": float(cur.C[0, 0]) / frac_lat,
    "ratio_vs_delaunay": float(cur.C[0, 0]) / frac_del,
}
H["current_file"]["self_consistent"] = bool(
    abs(H["current_file"]["c00"] /
        H["current_file"]["dc_expected_from_own_weights"] - 1.0) < 1e-12)

# reproduce the ORIGINAL stored solution: method=cg, reg=None, delaunay weights
c_orig, r_orig = analysis_wlsq(lat, lon, f, LY, weights=ws_del, method="cg",
                               reg=None, alpha=None)
fit = np.asarray(__import__("shkit.basis", fromlist=["x"]).synthesize(
    lat, lon, c_orig.C, c_orig.S, LY)).ravel()
H["reproduced_original_cg_delaunay"] = {
    "weight_rule": ws_del.rule, "c00": float(c_orig.C[0, 0]),
    "ratio_vs_lattice": float(c_orig.C[0, 0]) / frac_lat,
    "implied_area_km2": float(c_orig.C[0, 0]) * 4 * np.pi * R ** 2 / 1e6,
    "residual_rms": float(np.sqrt(np.mean((fit - f) ** 2))),
    "cg_info": r_orig.meta.get("cg_info"),
    "n_warnings": len(r_orig.warnings),
    "warn_c00": any("disagrees with the weighted mean" in m
                    for m in r_orig.warnings),
}
print(json.dumps(H, ensure_ascii=False, indent=1))
with open(os.path.join(FIGS, "_history.json"), "w", encoding="utf-8") as fh:
    json.dump(H, fh, ensure_ascii=False, indent=1)
print("wrote _history.json")
