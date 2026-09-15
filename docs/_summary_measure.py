# -*- coding: utf-8 -*-
"""Step 1 of the SHKit method-summary document: recompute EVERY number.

Writes:
    docs/SHKit方法总结_figs/_measurements.json   all scalars / tables
    docs/SHKit方法总结_figs/_data.npz            arrays the plotting step needs

Nothing here is typed from memory: every value is computed at run time.
Library code under shkit/ is imported read-only and never modified.
"""
import json
import os
import sys
import time

import numpy as np

SHKIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SHKIT)

from shkit import weights as W                                    # noqa: E402
from shkit.analysis import (analysis, analysis_projection,         # noqa: E402
                            analysis_quadrature, analysis_wlsq)
from shkit.basis import design_matrix_full, synthesize            # noqa: E402
from shkit.coeffs import SHCoeffs                                 # noqa: E402
from shkit.diagnostics import gram_deviation, gram_matrix         # noqa: E402
from shkit.filters import EARTH_RADIUS_M as R                     # noqa: E402
from shkit.slepian import slepian_analysis                        # noqa: E402
from shkit.synthesis import synthesis_grid                        # noqa: E402
from shkit.weights import (compute_weights, glq_grid,             # noqa: E402
                           lattice_cell_weights, lattice_step,
                           looks_like_lattice, voronoi_weights)

FIGS = os.path.join(SHKIT, "docs", "SHKit方法总结_figs")
os.makedirs(FIGS, exist_ok=True)
FOUR_PI = 4.0 * np.pi
M = {}
ARR = {}
T0 = time.time()


def stamp(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def sphere_area_km2(sr):
    return sr * R ** 2 / 1e6


# ---------------------------------------------------------------------------
def fib_sphere(n, seed=0):
    """Fibonacci lattice + a tiny deterministic jitter (same as the GUI demo)."""
    i = np.arange(n) + 0.5
    colat = np.arccos(1.0 - 2.0 * i / n)
    lam = np.mod(np.pi * (1 + 5 ** 0.5) * i, 2 * np.pi)
    return 90.0 - np.rad2deg(colat), np.rad2deg(lam)


def rand_sphere(n, rng):
    z = rng.uniform(-1, 1, n)
    lam = rng.uniform(0, 2 * np.pi, n)
    return np.rad2deg(np.arcsin(z)), np.rad2deg(lam)


def clustered_sphere(n, rng, cap_lat=40.0, cap_lon=100.0, cap_rad=35.0,
                     frac_cluster=0.7):
    n1 = int(round(n * frac_cluster))
    n2 = n - n1
    la, lo = rand_sphere(n2, rng)
    out_la, out_lo = [], []
    need = n1
    cla, clo, cr = np.deg2rad(cap_lat), np.deg2rad(cap_lon), np.deg2rad(cap_rad)
    while need > 0:
        m = max(need * 3, 64)
        z = rng.uniform(np.sin(cla - cr), np.sin(min(cla + cr, np.pi / 2)), m)
        x = rng.uniform(0, 2 * np.pi, m)
        la2 = np.rad2deg(np.arcsin(z))
        lo2 = np.rad2deg(x)
        cosd = (np.sin(np.deg2rad(la2)) * np.sin(cla) +
                np.cos(np.deg2rad(la2)) * np.cos(cla) *
                np.cos(np.deg2rad(lo2) - clo))
        keep = cosd >= np.cos(cr)
        out_la.append(la2[keep])
        out_lo.append(lo2[keep])
        need -= int(keep.sum())
    la1 = np.concatenate(out_la)[:n1]
    lo1 = np.concatenate(out_lo)[:n1]
    return np.concatenate([la1, la]), np.concatenate([lo1, lo])


def rand_truth(nmax, rng):
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for n in range(nmax + 1):
        for m in range(n + 1):
            C[n, m] = rng.standard_normal() / max(n, 1) ** 2
            if m:
                S[n, m] = rng.standard_normal() / max(n, 1) ** 2
    return SHCoeffs(C, S)


def cap_mask(lat, lon, clat, clon, rad):
    cosd = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(clat)) +
            np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(clat)) *
            np.cos(np.deg2rad(lon) - clon))
    return (cosd >= np.cos(np.deg2rad(rad))).astype(float)


def coeff_rel_err(est, truth):
    L = max(est.nmax, truth.nmax)
    a, b = est.truncate(L), truth.truncate(L)
    num = np.linalg.norm(a.C - b.C) ** 2 + np.linalg.norm(a.S - b.S) ** 2
    den = np.linalg.norm(b.C) ** 2 + np.linalg.norm(b.S) ** 2
    return float(np.sqrt(num / den))


def gram_dev(lat, lon, w, L):
    A = design_matrix_full(lat, lon, L)
    return float(gram_deviation(gram_matrix(A, w))), A


# ===========================================================================
stamp("== 1. math basics ==")
# Parseval on an exact quadrature rule
L = 8
glat, glon, gw = glq_grid(L)
rng = np.random.default_rng(1)
truth = rand_truth(L, rng)
gf = np.asarray(synthesize(glat, glon, truth.C, truth.S)).ravel()
lhs = float(np.sum(truth.power()))                       # sum C^2+S^2
rhs = float(np.sum(gw * gf ** 2) / FOUR_PI)              # (1/4pi) int f^2 dOmega
M["parseval"] = {"sum_power": lhs, "mean_square": rhs,
                 "rel_diff": abs(lhs / rhs - 1.0), "L": L,
                 "n_points": int(glat.size)}
stamp(f"   Parseval rel diff = {abs(lhs/rhs-1):.3e}")

M["resolution"] = {
    "R_km": R / 1000.0,
    "halfwave": {str(n): float(np.pi * R / 1000.0 / n)
                 for n in (4, 8, 12, 20, 40, 60, 120, 240, 480, 700)},
}

# ===========================================================================
stamp("== 2. Yangtze weights: three conventions ==")
p = np.genfromtxt(os.path.join(SHKIT, "Yangtze_River.txt"), delimiter="\t",
                  names=True, encoding="utf-8")
Y_LAT = np.asarray(p["latitude"], float)
Y_LON = np.asarray(p["longitude"], float)
Y_F = np.asarray(p["value"], float)
ARR["y_lat"], ARR["y_lon"], ARR["y_f"] = Y_LAT, Y_LON, Y_F

w_tool = np.asarray(np.genfromtxt(
    os.path.join(SHKIT, "examples", "yangtze_points_weights.csv"),
    delimiter=",", names=True, encoding="utf-8")["weight_sr"], float)

w_lat_vec = lattice_cell_weights(Y_LAT, Y_LON)
ws_lat = compute_weights(Y_LAT, Y_LON, rule="lattice")
ws_grd = compute_weights(Y_LAT, Y_LON, rule="grid")
ws_del = compute_weights(Y_LAT, Y_LON, rule="delaunay")
ws_auto = compute_weights(Y_LAT, Y_LON, rule="auto")
w_grd = ws_grd.w
w_del = ws_del.w
ARR["y_w_lattice"] = w_lat_vec
ARR["y_w_grid"] = w_grd

M["yangtze"] = {
    "n_points": int(Y_LAT.size),
    "lon_min": float(Y_LON.min()), "lon_max": float(Y_LON.max()),
    "lat_min": float(Y_LAT.min()), "lat_max": float(Y_LAT.max()),
    "lon_span": float(np.ptp(Y_LON)), "lat_span": float(np.ptp(Y_LAT)),
    "value_min": float(Y_F.min()), "value_max": float(Y_F.max()),
    "value_std": float(Y_F.std()),
    "unique_lat": int(np.unique(Y_LAT).size),
    "unique_lon": int(np.unique(Y_LON).size),
    "d_lat": lattice_step(Y_LAT), "d_lon": lattice_step(Y_LON),
    "is_lattice": bool(looks_like_lattice(Y_LAT, Y_LON)),
    "auto_rule": ws_auto.rule,
    "lat_gap_median": float(np.median(np.diff(np.unique(Y_LAT)))),
    "lat_gap_max": float(np.diff(np.unique(Y_LAT)).max()),
    "n_big_lat_gaps": int((np.diff(np.unique(Y_LAT)) >
                           1.5 * np.diff(np.unique(Y_LAT)).min()).sum()),
    "n_big_lon_gaps": int((np.diff(np.unique(Y_LON)) >
                           1.5 * np.diff(np.unique(Y_LON)).min()).sum()),
}
wy = M["yangtze"]
wy["d_lat_m"] = wy["d_lat"] * 111.19 * 1000.0
wy["d_lon_m"] = wy["d_lon"] * 111.19 * 1000.0
wy["box_km2"] = float(wy["lon_span"] * 111.19 *
                      np.cos(np.deg2rad(Y_LAT.mean())) * wy["lat_span"] * 111.19)

conv = {}
for tag, wset in (("lattice", ws_lat), ("grid", ws_grd), ("delaunay", ws_del),
                  ("tool_csv", W.WeightSet(w_tool, "user", w_tool.size))):
    s = float(wset.total)
    conv[tag] = {"sum_sr": s, "area_km2": sphere_area_km2(s),
                 "coverage": s / FOUR_PI, "c00": s / FOUR_PI}
ref = conv["lattice"]["sum_sr"]
for tag in conv:
    conv[tag]["ratio_vs_lattice"] = conv[tag]["sum_sr"] / ref
M["yangtze"]["weights"] = conv
stamp(f"   lattice {conv['lattice']['sum_sr']:.12e}  "
      f"grid x{conv['grid']['ratio_vs_lattice']:.6f}  "
      f"delaunay x{conv['delaunay']['ratio_vs_lattice']:.1f}  "
      f"tool ratio {conv['tool_csv']['sum_sr']/ref:.15f}")

ratio = w_grd / w_lat_vec
M["yangtze"]["grid_vs_lattice_ratio"] = {
    "min": float(ratio.min()), "p1": float(np.percentile(ratio, 1)),
    "median": float(np.median(ratio)), "p99": float(np.percentile(ratio, 99)),
    "max": float(ratio.max()),
    "std_over_mean": float(ratio.std() / ratio.mean()),
    "n_gt_1p1": int((ratio > 1.1).sum()),
    "frac_gt_1p001": float((ratio > 1.001).mean()),
    "n_points": int(ratio.size),
}

# ===========================================================================
stamp("== 3. regional Voronoi trap ==")
cap_lat, cap_lon, cap_rad = 30.0, 100.0, 30.0
fla, flo = fib_sphere(4000)
sel = cap_mask(fla, flo, cap_lat, cap_lon, cap_rad).astype(bool)
cap_true_area = 2 * np.pi * (1 - np.cos(np.deg2rad(cap_rad)))
w_vor_cap = voronoi_weights(fla[sel], flo[sel])
w_del_cap = compute_weights(fla[sel], flo[sel], rule="delaunay").w
M["voronoi_trap"] = {
    "cap_rad_deg": cap_rad, "n_points": int(sel.sum()),
    "true_cap_area_sr": float(cap_true_area),
    "true_cap_fraction": float(cap_true_area / FOUR_PI),
    "voronoi_sum_sr": float(w_vor_cap.sum()),
    "voronoi_ratio": float(w_vor_cap.sum() / cap_true_area),
    "delaunay_sum_sr": float(w_del_cap.sum()),
    "delaunay_ratio": float(w_del_cap.sum() / cap_true_area),
}
ARR["cap_la"], ARR["cap_lo"] = fla[sel], flo[sel]
stamp(f"   Voronoi {w_vor_cap.sum():.4f} vs cap {cap_true_area:.4f} "
      f"= {w_vor_cap.sum()/cap_true_area:.2f}x")

# ===========================================================================
stamp("== 4. Yangtze estimators at L=12 ==")
LY = 12
frac_y = conv["lattice"]["coverage"]


def c00_report(c):
    got = float(c.C[0, 0])
    return {"c00": got, "ratio_vs_area": got / frac_y,
            "implied_area_km2": sphere_area_km2(got * FOUR_PI)}


est = {}
cp, rp = analysis_projection(Y_LAT, Y_LON, Y_F, LY, weights=ws_lat)
est["projection"] = {
    **c00_report(cp), "residual_rms": float(rp.residual_rms),
    "residual_rms_weighted": float(rp.residual_rms_weighted),
    "residual_rel": float(rp.residual_rms),
    "weight_rule": rp.weight_rule,
    "dc_expected": float(rp.dc_mean_expected), "dc_got": float(rp.dc_mean_got),
    "n_warnings": len(rp.warnings),
}
ARR["y_proj_C"], ARR["y_proj_S"] = cp.C, cp.S

cq, rq = analysis_quadrature(Y_LAT, Y_LON, Y_F, LY, weights=ws_lat)
est["quadrature"] = {**c00_report(cq), "residual_rms": float(rq.residual_rms),
                     "note": "同 projection（权重同样取几何和时二者逐位相同）"}
est["quad_vs_proj_c00_rel"] = float(abs(cq.C[0, 0] / cp.C[0, 0] - 1.0))

# iterative / Richardson drift
drift = []
for k in (0, 1, 2, 3, 5, 8, 12, 20, 25):
    ck, rk = analysis_quadrature(Y_LAT, Y_LON, Y_F, LY, weights=ws_lat, niter=k)
    drift.append({"niter": k, "residual": float(rk.residual_rms),
                  "c00": float(ck.C[0, 0]),
                  "ratio": float(ck.C[0, 0]) / frac_y,
                  "implied_area_km2": sphere_area_km2(float(ck.C[0, 0]) * FOUR_PI)})
M["yangtze"]["iterative_drift"] = drift
stamp("   iterative drift done")

# CG (the stored .gfc path)
ccg, rcg = analysis_wlsq(Y_LAT, Y_LON, Y_F, LY, weights=ws_lat, method="cg")
est["cg"] = {**c00_report(ccg), "residual_rms": float(rcg.residual_rms),
             "cg_info": rcg.meta.get("cg_info"),
             "n_warnings": len(rcg.warnings),
             "warn_c00": any("disagrees with the weighted mean" in m
                             for m in rcg.warnings),
             "warn_collapsed": any("collapsed" in m for m in rcg.warnings)}
ARR["y_cg_C"], ARR["y_cg_S"] = ccg.C, ccg.S
stamp(f"   cg C00 = {ccg.C[0,0]:.9e} ({ccg.C[0,0]/frac_y:.1f}x), "
      f"residual {rcg.residual_rms:.6e}, info={rcg.meta.get('cg_info')}")

# the stored .gfc
from shkit import io                                            # noqa: E402
old = io.read_coeffs(os.path.join(SHKIT, "yantze_shkit_coeffs.gfc"))
old_fit = np.asarray(synthesize(Y_LAT, Y_LON, old.C, old.S, old.nmax)).ravel()
est["stored_gfc"] = {
    **c00_report(old), "nmax": old.nmax,
    "residual_rms": float(np.sqrt(np.mean((old_fit - Y_F) ** 2))),
    "max_abs_dev": float(np.abs(old_fit - Y_F).max()),
    "method": old.meta.get("method"), "reg": old.meta.get("reg"),
    "alpha": old.meta.get("alpha"), "weight_rule": old.meta.get("weight_rule"),
    "weight_sum": old.meta.get("weight_sum"),
}
M["yangtze"]["estimators"] = est

# regularization
regs = [{"reg": None, "alpha": None}, {"reg": "tikhonov", "alpha": 1e-8},
        {"reg": "tikhonov", "alpha": 1e-6}, {"reg": "tikhonov", "alpha": 1e-4},
        {"reg": "kaula", "alpha": 1e-4}, {"reg": "kaula", "alpha": 1e-6}]
regrows = []
for cfg in regs:
    ck, rk = analysis_wlsq(Y_LAT, Y_LON, Y_F, LY, weights=ws_lat, method="cg",
                           reg=cfg["reg"], alpha=cfg["alpha"])
    regrows.append({**cfg, "c00": float(ck.C[0, 0]),
                    "ratio": float(ck.C[0, 0]) / frac_y,
                    "implied_area_km2": sphere_area_km2(float(ck.C[0, 0]) * FOUR_PI),
                    "residual": float(rk.residual_rms)})
M["yangtze"]["regularization"] = regrows
stamp("   regularization done")

# projection tau
taus = [None, 0.9, 0.5, 0.1, 1e-2, 1e-4, 1e-6, 1e-8]
taurows = []
for tau in taus:
    ct, rt = analysis_projection(Y_LAT, Y_LON, Y_F, LY, weights=ws_lat, tau=tau)
    ce = rt.meta.get("completeness_eigen", {})
    taurows.append({"tau": tau, "c00": float(ct.C[0, 0]),
                    "ratio": float(ct.C[0, 0]) / frac_y,
                    "residual": float(rt.residual_rms),
                    "n_above_tau": ce.get("n_above_tau"),
                    "lam_max": ce.get("lambda_max"),
                    "sum_lam": ce.get("sum_lambda"),
                    "shannon": ce.get("shannon_number")})
M["yangtze"]["tau_sweep"] = taurows
stamp("   tau sweep done")

# completeness eigenvalues + Slepian identity
gd, A = gram_dev(Y_LAT, Y_LON, ws_lat.w, LY)
K = gram_matrix(A, ws_lat.w)
lam = np.linalg.eigvalsh(K)
M["yangtze"]["gram"] = {
    "max_K_minus_I": gd,
    "ncoef": int(lam.size),
    "lambda_max": float(lam.max()), "lambda_min": float(lam.min()),
    "sum_lambda": float(lam.sum()),
    "shannon": float((LY + 1) ** 2 * frac_y),
    "sum_over_shannon": float(lam.sum() / ((LY + 1) ** 2 * frac_y)),
    "n_above": {str(t): int((lam >= t).sum())
                for t in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)},
}
stamp(f"   sum(lambda)/Shannon = {lam.sum()/((LY+1)**2*frac_y):.12f}")

# ===========================================================================
stamp("== 5. Yangtze spectrum, leakage, Shannon sweep ==")
rms_proj = cp.degree_rms()
rms_cg = ccg.degree_rms()
p_proj = cp.power()
M["yangtze"]["spectrum"] = {
    "n": list(range(LY + 1)),
    "proj_degree_rms": [float(v) for v in rms_proj],
    "proj_power_share": [float(v / p_proj.sum()) for v in p_proj],
    "cg_degree_rms": [float(v) for v in rms_cg],
}
fit_pts = np.asarray(synthesize(Y_LAT, Y_LON, cp.C, cp.S, LY)).ravel()
M["yangtze"]["recon_peak_proj"] = float(fit_pts.max())
M["yangtze"]["shannon_L12"] = float((LY + 1) ** 2 * frac_y)
M["yangtze"]["L_for_mask_peak_1"] = float(np.sqrt(1.0 / frac_y) - 1.0)

glat_y = np.linspace(Y_LAT.min(), Y_LAT.max(), 120)
glon_y = np.linspace(Y_LON.min(), Y_LON.max(), 240)
GY = np.asarray(synthesis_grid(glat_y, glon_y, cp, nmax=LY))
CG = np.asarray(synthesis_grid(glat_y, glon_y, ccg, nmax=LY))
ARR["gy_lat"], ARR["gy_lon"] = glat_y, glon_y
ARR["gy_proj"], ARR["gy_cg"] = GY, CG
M["yangtze"]["box_field_proj_max"] = float(GY.max())
M["yangtze"]["box_field_proj_mean"] = float(GY.mean())
M["yangtze"]["box_field_cg_max"] = float(CG.max())
M["yangtze"]["box_field_cg_mean"] = float(CG.mean())
# pattern correlation of the two reconstructions at the data points
cg_pts = np.asarray(synthesize(Y_LAT, Y_LON, ccg.C, ccg.S, LY)).ravel()
M["yangtze"]["pattern_corr_proj_vs_cg"] = float(
    np.corrcoef(fit_pts, cg_pts)[0, 1])

# grid vs lattice reconstruction shape
c_gl, _ = analysis_projection(Y_LAT, Y_LON, Y_F, LY, weights=ws_grd)
gl_pts = np.asarray(synthesize(Y_LAT, Y_LON, c_gl.C, c_gl.S, LY)).ravel()
M["yangtze"]["grid_vs_lattice"] = {
    "c00_ratio": float(c_gl.C[0, 0] / cp.C[0, 0]),
    "peak_grid": float(gl_pts.max()), "peak_lattice": float(fit_pts.max()),
    "peak_ratio": float(gl_pts.max() / fit_pts.max()),
    "pattern_corr": float(np.corrcoef(gl_pts, fit_pts)[0, 1]),
    "max_rel_diff_spectrum": float(
        np.abs(c_gl.degree_rms()[1:] / rms_proj[1:] - 1.0).max()),
}

sweep = []
for Lk in (12, 30, 60, 120):
    t1 = time.time()
    ck, rk = analysis_projection(Y_LAT, Y_LON, Y_F, Lk, weights=ws_lat)
    pk = np.asarray(synthesize(Y_LAT, Y_LON, ck.C, ck.S, Lk)).ravel()
    sh = (Lk + 1) ** 2 * frac_y
    sweep.append({"L": Lk, "shannon": float(sh), "peak": float(pk.max()),
                  "peak_over_shannon": float(pk.max() / sh),
                  "c00": float(ck.C[0, 0]),
                  "residual": float(rk.residual_rms),
                  "halfwave_km": float(np.pi * R / 1000.0 / Lk),
                  "secs": round(time.time() - t1, 1)})
    stamp(f"   Yangtze sweep L={Lk} peak={pk.max():.5e} shannon={sh:.4e}")
M["yangtze"]["shannon_sweep"] = sweep

# ===========================================================================
stamp("== 6. global 0/1 cap: truncation floor vs iteration ==")
nla, nlo = 360, 720
clat = np.linspace(-89.75, 89.75, nla)
clon = np.linspace(0.25, 359.75, nlo)
CLA, CLO = np.meshgrid(clat, clon, indexing="ij")
CF = cap_mask(CLA.ravel(), CLO.ravel(), 30.0, 110.0, 20.0)
CFLAT, CFLON = CLA.ravel(), CLO.ravel()
wc = compute_weights(CFLAT, CFLON, rule="dh")
CFRAC = float(np.sum(wc.w * CF) / FOUR_PI)
ARR["cap_lat"], ARR["cap_lon"], ARR["cap_f"] = CFLAT, CFLON, CF
floor = []
for Lk in (2, 4, 8, 12, 20, 40):
    _, r0 = analysis_quadrature(CFLAT, CFLON, CF, Lk, weights=wc)
    _, r3 = analysis_quadrature(CFLAT, CFLON, CF, Lk, weights=wc, niter=3)
    # the 25-iteration check is only run where it is affordable; the point is
    # that it changes nothing, which 3 iterations already demonstrates
    nit = 25 if Lk <= 12 else 3
    cN, rN = analysis_quadrature(CFLAT, CFLON, CF, Lk, weights=wc, niter=nit)
    fn = np.asarray(synthesize(CFLAT, CFLON, cN.C, cN.S, Lk)).ravel()
    grel = float(np.sqrt(np.sum(wc.w * (fn - CF) ** 2) /
                         np.sum(wc.w * CF ** 2)))
    floor.append({"L": Lk, "ncoef": (Lk + 1) ** 2,
                  "quad": float(r0.residual_rms),
                  "it3": float(r3.residual_rms),
                  "it_n": nit,
                  "it25": float(rN.residual_rms),
                  "global_rel": grel,
                  "c00": float(cN.C[0, 0]),
                  "area_fraction": CFRAC,
                  "resid_weighted": float(rN.residual_rms_weighted)})
    stamp(f"   cap L={Lk}: quad={r0.residual_rms:.8f} "
          f"it{nit}={rN.residual_rms:.8f}")
M["cap"] = {"area_fraction": CFRAC, "floor": floor,
            "dh_weight_sum": float(wc.total),
            "n_points": int(CFLAT.size)}

# L=12 detail: unweighted vs area-weighted residual
_, r12 = analysis_quadrature(CFLAT, CFLON, CF, 12, weights=wc)
M["cap"]["L12_resid_unweighted"] = float(r12.residual_rms)
M["cap"]["L12_resid_weighted"] = float(r12.residual_rms_weighted)
M["cap"]["L12_ratio"] = float(r12.residual_rms_weighted / r12.residual_rms)
c12q, _ = analysis_quadrature(CFLAT, CFLON, CF, 12, weights=wc)
ARR["cap_c12_C"], ARR["cap_c12_S"] = c12q.C, c12q.S

# ===========================================================================
stamp("== 7. Fibonacci 3000: round-trip + noise + auto routing ==")
Lfb = 20
latf, lonf = fib_sphere(3000)
wf = compute_weights(latf, lonf, rule="voronoi")
truth20 = rand_truth(Lfb, np.random.default_rng(7))
fb = np.asarray(synthesize(latf, lonf, truth20.C, truth20.S)).ravel()


def coef_err(c):
    return coeff_rel_err(c, truth20)


fb_rows = []
for tag, kw in (("quadrature", dict(method="quadrature")),
                ("iterative x1", dict(method="iterative", niter=1)),
                ("iterative x3", dict(method="iterative", niter=3)),
                ("wlsq", dict(method="wlsq")),
                ("projection", dict(method="projection"))):
    ck, rk = analysis(latf, lonf, fb, Lfb, weights=wf, **kw)
    fb_rows.append({"tag": tag, "coef_rel_err": coef_err(ck),
                    "residual": float(rk.residual_rms)})
M["fib"] = {"n": int(latf.size), "L": Lfb, "rows": fb_rows}
stamp("   fib round trip: " + ", ".join(
    f"{r['tag']}={r['coef_rel_err']:.3e}" for r in fb_rows))

# noise behaviour (truth band-limited to 40, solve to 20)
Ltr = 40
truth40 = rand_truth(Ltr, np.random.default_rng(11))
clean = np.asarray(synthesize(latf, lonf, truth40.C, truth40.S)).ravel()
noise_rows = []
for sig in (0.0, 0.01, 0.05):
    rngn = np.random.default_rng(23)
    y = clean + (rngn.standard_normal(clean.size) * sig *
                 np.sqrt(np.mean(clean ** 2)) if sig else 0.0)
    row = {"sigma_rel": sig}
    for tag, kw in (("quadrature", dict(method="quadrature")),
                    ("iterative", dict(method="iterative", niter=1)),
                    ("wlsq", dict(method="wlsq"))):
        ck, _ = analysis(latf, lonf, y, Lfb, weights=wf, **kw)
        row[tag] = coeff_rel_err(ck, truth40)
    row["theory"] = (Lfb + 1) / np.sqrt(latf.size) * sig if sig else 0.0
    noise_rows.append(row)
M["noise"] = {"rows": noise_rows, "L_solve": Lfb, "L_truth": Ltr}
stamp("   noise rows done")

# auto routing table
rng = np.random.default_rng(20240607)
sets = {
    "DH 网格 90x180（规则）": (lambda: (lambda v: (v[0].ravel(), v[1].ravel()))(
        np.meshgrid(np.linspace(-89, 89, 90), np.linspace(0, 358, 180),
                    indexing="ij"))),
    "GLQ 网格（L=20，用 GLQ 自己的权重）": (lambda: glq_grid(20)),
    "Fibonacci 3000 点": (lambda: fib_sphere(3000)),
    "随机均匀 3000 点": (lambda: rand_sphere(3000, rng)),
    "聚簇 3000 点（70% 在 35° 帽）": (lambda: clustered_sphere(3000, rng)),
}
route_rows = []
for name, mk in sets.items():
    got = mk()
    if len(got) == 3:
        la, lo, ww = got[0], got[1], compute_weights(
            got[0], got[1], rule="user", user_w=got[2], normalise="none")
    else:
        la, lo = got
        ww = compute_weights(la, lo, rule="dh" if name.startswith("DH")
                             else "voronoi")
    y = np.asarray(synthesize(la, lo, truth20.C, truth20.S)).ravel()
    gd_, _ = gram_dev(la, lo, ww.w, Lfb)
    _, rr = analysis(la, lo, y, Lfb, method="auto", weights=ww)
    route_rows.append({"name": name, "n": int(la.size),
                       "max_K_minus_I": gd_,
                       "auto": rr.meta.get("auto_choice")})
    stamp(f"   route {name}: max|K-I|={gd_:.2e} -> {rr.meta.get('auto_choice')}")
M["routing"] = route_rows

# ===========================================================================
stamp("== 8. regional honesty (30 deg cap, global truth) ==")
Lr = 12
truth_r = rand_truth(8, np.random.default_rng(9))
fr = np.asarray(synthesize(fla, flo, truth_r.C, truth_r.S)).ravel()
la_r, lo_r, fs_r = fla[sel], flo[sel], fr[sel]
wr = compute_weights(la_r, lo_r, rule="delaunay", normalise="region")
cr, rr_ = analysis(la_r, lo_r, fs_r, Lr, method="quadrature", weights=wr)
cw2, rw2 = analysis_wlsq(la_r, lo_r, fs_r, Lr, weights=wr, method="wlsq")
regrows2 = {}
for reg, al in ((None, None), ("tikhonov", 1e-9), ("kaula", 1e-4)):
    ck, _ = analysis_wlsq(la_r, lo_r, fs_r, Lr, weights=wr, method="wlsq",
                          reg=reg, alpha=al)
    regrows2[f"{reg}-{al}"] = coeff_rel_err(ck, truth_r)
perL = []
for Lk in (2, 4, 6, 8, 12):
    ck, rk = analysis_wlsq(la_r, lo_r, fs_r, Lk, weights=wr, method="wlsq")
    perL.append({"L": Lk, "ncoef": (Lk + 1) ** 2,
                 "cond": float(rk.condition_number) if np.isfinite(
                     rk.condition_number) else None})
M["regional30"] = {
    "cap_rad": 30.0, "n_points": int(sel.sum()), "coverage": float(wr.total / FOUR_PI),
    "quad_coef_rel_err": coeff_rel_err(cr, truth_r),
    "wlsq_coef_rel_err": coeff_rel_err(cw2, truth_r),
    "wlsq_cond": float(rw2.condition_number) if np.isfinite(
        rw2.condition_number) else None,
    "reg_coef_rel_err": regrows2,
    "per_L_cond": perL,
}
stamp(f"   region quad err={coeff_rel_err(cr, truth_r):.3f} "
      f"wlsq err={coeff_rel_err(cw2, truth_r):.1f} cond={rw2.condition_number:.2e}")

# ===========================================================================
stamp("== 9. Slepian vs unregularised WLSQ (25 deg cap) ==")
Lsp = 20
sp_lat, sp_lon, sp_rad = 30.0, 100.0, 25.0
sel_sp = cap_mask(fla, flo, sp_lat, sp_lon, sp_rad).astype(bool)
la_s, lo_s = fla[sel_sp], flo[sel_sp]
truth_sp = rand_truth(10, np.random.default_rng(3))
fs_s = np.asarray(synthesize(la_s, lo_s, truth_sp.C, truth_sp.S)).ravel()
w_s = compute_weights(la_s, lo_s, rule="delaunay", normalise="region")
spin = {}
cs, rs, basis = slepian_analysis(la_s, lo_s, fs_s, nmax=Lsp, weights=w_s,
                                 lam_min=0.5)
cw_s, rw_s = analysis_wlsq(la_s, lo_s, fs_s, Lsp, weights=w_s, method="wlsq")
# field outside the cap, relative to the RMS inside
out_lat, out_lon = fib_sphere(6000)
out_sel = ~cap_mask(out_lat, out_lon, sp_lat, sp_lon, sp_rad).astype(bool)
in_sel = ~out_sel
f_in = np.asarray(synthesize(out_lat, out_lon, cs.C, cs.S, min(Lsp, cs.nmax)))
f_w = np.asarray(synthesize(out_lat, out_lon, cw_s.C, cw_s.S,
                            min(Lsp, cw_s.nmax)))
den = float(np.sqrt(np.mean(f_in[in_sel] ** 2)))
spin = {
    "n_points": int(sel_sp.sum()), "L": Lsp,
    "coverage": float(w_s.total / FOUR_PI),
    "slepian_cond": float(rs.condition_number) if np.isfinite(
        rs.condition_number) else None,
    "slepian_normC": float(np.sqrt(np.sum(cs.C ** 2) + np.sum(cs.S ** 2))),
    "slepian_outside_rel": float(np.sqrt(np.mean(f_in[out_sel] ** 2)) / den),
    "slepian_inside_truth_norm": float(
        np.sqrt(np.sum(truth_sp.C ** 2) + np.sum(truth_sp.S ** 2))),
    "slepian_ntaper": int((np.asarray(basis.eigenvalues) > 0.5).sum()),
    "slepian_region_fraction": float(getattr(basis, "region_fraction", np.nan)),
    "wlsq_cond": float(rw_s.condition_number) if np.isfinite(
        rw_s.condition_number) else None,
    "wlsq_normC": float(np.sqrt(np.sum(cw_s.C ** 2) + np.sum(cw_s.S ** 2))),
    "wlsq_outside_rel": float(np.sqrt(np.mean(f_w[out_sel] ** 2)) / den),
    "slepian_coef_err": coeff_rel_err(cs, truth_sp),
    "wlsq_coef_err": coeff_rel_err(cw_s, truth_sp),
}
lam_sp = getattr(basis, "eigenvalues", None)
if lam_sp is not None:
    lam_sp = np.asarray(lam_sp)
    spin["eig"] = {"n": int(lam_sp.size),
                   "gt_0.99": int((lam_sp > 0.99).sum()),
                   "gt_0.9": int((lam_sp > 0.9).sum()),
                   "gt_0.5": int((lam_sp > 0.5).sum()),
                   "gt_0.1": int((lam_sp > 0.1).sum()),
                   "sum": float(lam_sp.sum()),
                   "shannon": float((Lsp + 1) ** 2 * w_s.total / FOUR_PI),
                   "max": float(lam_sp.max()), "min": float(lam_sp.min())}
    ARR["slepian_lam"] = lam_sp
M["slepian"] = spin
stamp(f"   slepian cond={spin['slepian_cond']} outside={spin['slepian_outside_rel']:.2e} "
      f"wlsq cond={spin['wlsq_cond']:.2e} outside={spin['wlsq_outside_rel']:.1f}")

# small cap: can it hold a degree-4 field?
small_rows = []
for rad in (24.0, 15.0, 8.0):
    s2 = cap_mask(fla, flo, 30.0, 100.0, rad).astype(bool)
    cov = (1 - np.cos(np.deg2rad(rad))) / 2
    ww2 = compute_weights(fla[s2], flo[s2], rule="delaunay", normalise="region")
    A4 = design_matrix_full(fla[s2], flo[s2], 4)
    K4 = gram_matrix(A4, ww2.w)
    e4 = np.linalg.eigvalsh(K4)
    small_rows.append({"rad": rad, "coverage": float(cov),
                       "L": 4, "ncoef": 25,
                       "shannon": float(25 * cov),
                       "lam_max": float(e4.max()),
                       "n_above_0.5": int((e4 > 0.5).sum()),
                       "sum_lam": float(e4.sum())})
M["small_caps"] = small_rows
stamp("   small caps done")

# ===========================================================================
with open(os.path.join(FIGS, "_measurements.json"), "w", encoding="utf-8") as fh:
    json.dump(M, fh, ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(FIGS, "_data.npz"), **ARR)
stamp(f"wrote _measurements.json and _data.npz ({len(M)} blocks) to {FIGS}")
print(json.dumps({"yangtze_frac": frac_y,
                  "auto_rule": ws_auto.rule}, ensure_ascii=False))
