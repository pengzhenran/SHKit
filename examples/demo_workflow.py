# -*- coding: utf-8 -*-
"""
SHKit end-to-end demonstration.

Runs four scenarios on synthetic data with a known answer, and prints what each
integration-element / estimator choice actually costs:

  1. global regular grid              -> Driscoll-Healy, exact
  2. global quasi-uniform scatter     -> Voronoi vs a flat area
  3. global clustered scatter         -> Voronoi vs a flat area
  4. regional scatter (a cap)         -> Delaunay + quadrature / WLSQ / Slepian

Run:  python examples/demo_workflow.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shkit.analysis import analysis, analysis_quadrature, analysis_wlsq
from shkit.coeffs import SHCoeffs, triangle_order
from shkit.filters import apply_gaussian
from shkit.slepian import slepian_analysis, slepian_basis
from shkit.synthesis import synthesis, synthesis_grid
from shkit.weights import (compute_weights, delaunay_weights, dh_lat_weights,
                           glq_grid, uniform_weights, voronoi_weights)


# ---------------------------------------------------------------- utilities
def banner(text):
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def fib_sphere(n):
    """Quasi-uniform (Fibonacci) points covering the whole sphere."""
    i = np.arange(n) + 0.5
    colat = np.arccos(1.0 - 2.0 * i / n)
    lon = np.mod(np.pi * (1 + 5 ** 0.5) * np.arange(n), 2 * np.pi)
    return 90.0 - np.rad2deg(colat), np.rad2deg(lon)


def synthetic_field(nmax, band, seed=0):
    """Kaula-like random coefficients, non-zero only up to `band`."""
    rng = np.random.default_rng(seed)
    m, n = triangle_order(nmax)
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for k in range(len(m)):
        if n[k] > band:
            continue
        C[n[k], m[k]] = rng.standard_normal() / max(n[k], 1) ** 2
        if m[k] >= 1:
            S[n[k], m[k]] = rng.standard_normal() / max(n[k], 1) ** 2
    return SHCoeffs(C, S)


def coeff_error(est, truth):
    L = max(est.nmax, truth.nmax)
    a, b = est.truncate(L), truth.truncate(L)
    num = np.linalg.norm(a.C - b.C) ** 2 + np.linalg.norm(a.S - b.S) ** 2
    den = np.linalg.norm(b.C) ** 2 + np.linalg.norm(b.S) ** 2
    return float(np.sqrt(num / den))


ROWS = []


def record(scenario, rule, method, err, note=""):
    ROWS.append((scenario, rule, method, err, note))
    print(f"    {rule:<10} {method:<12} coeff rel.err = {err:9.3e}   {note}")


# ---------------------------------------------------------------- 1. grid
def scenario_global_grid():
    banner("1. 全球规则网格（等经纬 90 x 180）")
    nlat, nlon = 90, 180
    L, band = 20, 20
    lat = -90.0 + np.arange(nlat) * (180.0 / nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    truth = synthetic_field(L, band, seed=1)
    f = synthesis(LA.ravel(), LO.ravel(), truth).reshape(nlat, nlon)

    a = dh_lat_weights(lat)
    print(f"  Driscoll-Healy 纬向权重 sum = {a.sum():.12f}  (应为 2)")
    w = compute_weights(LA.ravel(), LO.ravel(), rule="dh", nlon=nlon)
    print(f"  每点权重 sum = {w.total:.12f}   coverage = {w.coverage:.12f}")

    est, rep = analysis(LA.ravel(), LO.ravel(), f.ravel(), L,
                        method="quadrature", weights=w)
    record("global grid", "dh", "quadrature", coeff_error(est, truth),
           f"gram max|K-I| = {rep.gram_deviation:.2e}")

    # exact Gauss-Legendre reference grid
    glat, glon, gw = glq_grid(L)
    fg = synthesis(glat, glon, truth)
    est_g, _ = analysis(glat, glon, fg, L, method="quadrature", rule="user",
                        user_w=gw)
    record("global grid", "glq", "quadrature", coeff_error(est_g, truth),
           f"{glat.size} points, exact quadrature")


# ---------------------------------------------------- 2/3. scattered global
def scenario_scatter(name, lat, lon, truth, L, seed):
    banner(name)
    f = synthesis(lat, lon, truth)
    print(f"  N = {lat.size} 真实场带限到 {truth.nmax} 阶")

    wv = voronoi_weights(lat, lon)
    print(f"  Voronoi 胞面积和 = {wv.sum():.10f} (4pi = {4*np.pi:.10f})，"
          f"异质性 max/min = {wv.max()/wv.min():.2f}")

    for rule, w, tag in (
        ("uniform", uniform_weights(lat.size), "旧做法：统一面积"),
        ("voronoi", wv, "球面 Voronoi 胞面积"),
    ):
        ws = compute_weights(lat, lon, rule="user", user_w=w,
                             normalise="global")
        est, rep = analysis_quadrature(lat, lon, f, L, weights=ws)
        record(name, rule, "quadrature", coeff_error(est, truth), tag)

    # iterative correction on the Voronoi solution
    ws = compute_weights(lat, lon, rule="user", user_w=wv, normalise="global")
    est_i, rep_i = analysis_quadrature(lat, lon, f, L, weights=ws, niter=3)
    record(name, "voronoi", "iterative x3", coeff_error(est_i, truth),
           "理查森迭代校正")

    est_w, rep_w = analysis_wlsq(lat, lon, f, L, weights=ws, method="wlsq")
    record(name, "voronoi", "wlsq", coeff_error(est_w, truth),
           f"cond = {rep_w.condition_number:.2e}")

    est_a, rep_a = analysis(lat, lon, f, L, method="auto", weights=ws)
    print(f"    -> method='auto' 选择 '{rep_a.meta.get('auto_choice')}'，"
          f"gram = {rep_a.gram_deviation:.2e}, 超定比 = "
          f"{rep_a.overdetermination:.1f}")


def scenario_random_scatter():
    rng = np.random.default_rng(5)
    N, L = 3000, 12
    lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, N)))
    lon = rng.uniform(0, 360, N)
    scenario_scatter("2. 全球随机散点", lat, lon,
                     synthetic_field(L, 8, seed=2), L, 2)


def scenario_clustered_scatter():
    rng = np.random.default_rng(6)
    N, L = 3000, 12
    n1 = int(0.7 * N)
    cla, clo, cr = np.deg2rad(40.0), np.deg2rad(100.0), np.deg2rad(35.0)
    pts = []
    while sum(len(p[0]) for p in pts) < n1:
        z = rng.uniform(np.sin(cla - cr), np.sin(cla + cr), 2048)
        x = rng.uniform(0, 2 * np.pi, 2048)
        la2 = np.rad2deg(np.arcsin(z))
        lo2 = np.rad2deg(x)
        cosd = (np.sin(np.deg2rad(la2)) * np.sin(cla) +
                np.cos(np.deg2rad(la2)) * np.cos(cla) *
                np.cos(np.deg2rad(lo2) - clo))
        pts.append((la2[cosd >= np.cos(cr)], lo2[cosd >= np.cos(cr)]))
    la1 = np.concatenate([p[0] for p in pts])[:n1]
    lo1 = np.concatenate([p[1] for p in pts])[:n1]
    lat = np.concatenate([la1, np.rad2deg(np.arcsin(rng.uniform(-1, 1, N - n1)))])
    lon = np.concatenate([lo1, rng.uniform(0, 360, N - n1)])
    scenario_scatter("3. 全球聚簇散点（70% 落在 35° 球冠内）", lat, lon,
                     synthetic_field(L, 8, seed=3), L, 3)


# ---------------------------------------------------------------- 4. region
def scenario_region():
    banner("4. 区域散点（25° 球冠）—— 这里才是问题的本质")
    L = 20
    lat_all, lon_all = fib_sphere(6000)
    cap_lat, cap_lon, cap_rad = 30.0, 100.0, 25.0
    cla, clo, cr = np.deg2rad(cap_lat), np.deg2rad(cap_lon), np.deg2rad(cap_rad)
    cosd = (np.sin(np.deg2rad(lat_all)) * np.sin(cla) +
            np.cos(np.deg2rad(lat_all)) * np.cos(cla) *
            np.cos(np.deg2rad(lon_all) - clo))
    sel = cosd >= np.cos(cr)
    lat, lon = lat_all[sel], lon_all[sel]
    cap_area = 2 * np.pi * (1 - np.cos(cr))
    print(f"  区域内 {lat.size} 个点；真实球冠面积 = {cap_area:.6f}，"
          f"占全球 {cap_area/(4*np.pi):.4f}")

    wv = voronoi_weights(lat, lon)
    wd = delaunay_weights(lat, lon)
    print(f"  ⚠ 区域点集上 Voronoi 面积和 = {wv.sum():.6f}  "
          f"（虚高 {wv.sum()/cap_area:.1f} 倍，因为 Voronoi 恒铺满全球）")
    print(f"  ✔ Delaunay 1/3 面积和        = {wd.sum():.6f}  "
          f"（球面凸包，比球冠小 {abs(wd.sum()-cap_area)/cap_area:.1%}）")

    truth = synthetic_field(L, 10, seed=4)
    f = synthesis(lat, lon, truth)
    w = compute_weights(lat, lon, rule="delaunay", normalise="region")

    est_q, rep_q = analysis(lat, lon, f, L, method="quadrature", weights=w)
    print(f"\n  [求积投影]")
    print(f"    coverage = {rep_q.coverage:.4f}  "
          f"（真实覆盖 {cap_area/(4*np.pi):.4f}）")
    record("region", "delaunay", "quadrature", coeff_error(est_q, truth),
           f"coverage = {rep_q.coverage:.4f}")

    est_w, rep_w = analysis_wlsq(lat, lon, f, L, weights=w, method="wlsq")
    record("region", "delaunay", "wlsq(未正则)", coeff_error(est_w, truth),
           f"cond = {rep_w.condition_number:.2e}")

    est_r, rep_r = analysis_wlsq(lat, lon, f, L, weights=w, method="wlsq",
                                 reg="kaula", alpha=1e-6)
    record("region", "delaunay", "wlsq+kaula", coeff_error(est_r, truth),
           f"alpha = {rep_r.alpha:.1e}")

    coeffs_s, rep_s, basis = slepian_analysis(lat, lon, f, nmax=L, weights=w,
                                              lam_min=0.5)
    record("region", "delaunay", "slepian", coeff_error(coeffs_s, truth),
           f"使用了 {rep_s.meta['ntaper_used']}/{rep_s.meta['ntaper_available']} 个 taper")

    print("\n  Slepian 谱：")
    for line in basis.spectrum().splitlines():
        print("    " + line)

    print("\n  区域外行为（相对区域内 RMS）：")
    denom = np.sqrt(np.mean(
        np.asarray(synthesis(lat_all, lon_all, truth))[sel] ** 2))
    for tag, c in (("slepian", coeffs_s), ("wlsq(未正则)", est_w),
                   ("quadrature", est_q)):
        v = np.asarray(synthesis(lat_all, lon_all, c))
        print(f"    {tag:<14} 区域外 = "
              f"{np.sqrt(np.mean(v[~sel] ** 2))/denom:9.3e}   "
              f"||C|| = {np.linalg.norm(c.C):9.3e}")
    print(f"    {'真值':<14} 区域外 = "
          f"{np.sqrt(np.mean(np.asarray(synthesis(lat_all, lon_all, truth))[~sel] ** 2))/denom:9.3e}   "
          f"||C|| = {np.linalg.norm(truth.C):9.3e}")

    print("\n  为什么区域反演救不了低阶？")
    print(f"    常数场在该区域的集中因子 = 面积占比 = "
          f"{cap_area/(4*np.pi):.4f}（不是 1）——低阶场本就弥散在全球。")
    print(f"    Shannon 数 = (L+1)^2·A/4pi = {basis.shannon:.1f}，"
          f"而系数个数 = {basis.ncoef}。")

    # ---------------------------------------------------------- remove-restore
    print("\n  ── 可落地做法：remove–restore ──")
    print("    先用全球模型（这里用真值的 0–4 阶代替）扣掉长波，")
    print("    只在区域内分析残差，最后加回。")
    low = truth.truncate(4)
    f_res = f - synthesis(lat, lon, low)
    coeffs_res, rep_res, _ = slepian_analysis(lat, lon, f_res, nmax=L,
                                              weights=w, lam_min=0.5)
    total = low + coeffs_res
    fit_res = np.sqrt(np.mean(
        (np.asarray(synthesis(lat, lon, coeffs_res)) - f_res) ** 2)) / \
        np.sqrt(np.mean(f_res ** 2))
    print(f"    残差场的区域内相对拟合误差 = {fit_res:.3e}")
    print(f"    残差系数范数 = {np.linalg.norm(coeffs_res.C):.4e}  "
          f"（未正则 WLSQ 在同一区域是 "
          f"{np.linalg.norm(est_w.C):.4e}）")
    v = np.asarray(synthesis(lat_all, lon_all, total))
    print(f"    区域外/区域内 RMS = "
          f"{np.sqrt(np.mean(v[~sel]**2))/denom:.3e}"
          f"  <- 长波由模型提供，区域外不再发散")
    print("    诚实解读：remove–restore 让解**稳定、物理有界**，")
    print("    但它**不创造信息**。25° 球冠的 Shannon 数只有 19，")
    print("    残差场（5–10 阶）的集中因子同样只有 ~0.047，拟合误差不会消失。")
    print("    要真正提高区域分辨率，只能：缩小阶数范围 / 扩大区域 / 引入先验。")


# ------------------------------------------------------------ 5. synthesis
def scenario_synthesis():
    banner("5. 综合：球谐系数 -> 任意点 / 任意网格 / 高斯平滑")
    L = 20
    truth = synthetic_field(L, 20, seed=7)
    rng = np.random.default_rng(9)
    lat = rng.uniform(-90, 90, 500)
    lon = rng.uniform(0, 360, 500)
    f1 = synthesis(lat, lon, truth)
    f2 = synthesis(lat, lon, truth, nmax=10)
    print(f"  任意 500 点：nmax=20 时 RMS = {np.sqrt(np.mean(f1**2)):.6f}，"
          f"截断到 10 阶后 RMS = {np.sqrt(np.mean(f2**2)):.6f}")
    g = synthesis_grid(np.arange(-90, 91, 1.0), np.arange(0, 360, 1.0), truth)
    print(f"  1°x1° 全球网格 shape = {g.shape}")
    gs = synthesis_grid(np.arange(-90, 91, 1.0), np.arange(0, 360, 1.0), truth,
                        gaussian_km=300.0)
    print(f"  300 km 高斯平滑后 RMS: {np.sqrt(np.mean(g**2)):.6f} -> "
          f"{np.sqrt(np.mean(gs**2)):.6f}")


def summary_table():
    banner("汇总")
    print(f"  {'场景':<34} {'积分元':<10} {'方法':<14} {'系数相对误差':>12}")
    print("  " + "-" * 74)
    for sc, rule, method, err, note in ROWS:
        print(f"  {sc:<34} {rule:<10} {method:<14} {err:>12.3e}")
    print("\n  结论：")
    print("   * 全球规则网格 + DH 权重 = 机器精度（这是旧软件的场景，没问题）")
    print("   * 散点用统一面积（旧做法）误差可达 1e-1~1e0 量级；Voronoi 好一个数量级以上")
    print("   * 区域数据：求积有系统偏差、未正则 WLSQ 条件数 1e16 而失控，")
    print("     Slepian 是唯一把 ||C|| 和区域外能量都压住的方案")


def main():
    t0 = time.time()
    scenario_global_grid()
    scenario_random_scatter()
    scenario_clustered_scatter()
    scenario_region()
    scenario_synthesis()
    summary_table()
    print(f"\n总耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
