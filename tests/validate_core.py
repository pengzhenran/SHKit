# -*- coding: utf-8 -*-
"""
Numerical validation of the shkit core.

Run:  python tests/validate_core.py
Every check prints PASS/FAIL with the measured quantity, so the numbers can be
quoted in the documentation.
"""
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, __file__.rsplit("tests", 1)[0])

from shkit.basis import (design_matrix_full, legendre_pbar, synthesize)
from shkit.coeffs import SHCoeffs, triangle_order
from shkit.weights import (compute_weights, delaunay_weights, dh_lat_weights,
                           glq_grid, grid_cell_weights, lonlat_to_xyz,
                           uniform_weights, voronoi_weights, FOUR_PI)
from shkit.analysis import analysis, analysis_quadrature, analysis_wlsq

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")


def fib_sphere(n):
    i = np.arange(n) + 0.5
    colat = np.arccos(1.0 - 2.0 * i / n)
    lon = np.mod(np.pi * (1 + 5 ** 0.5) * np.arange(n), 2 * np.pi)
    return 90.0 - np.rad2deg(colat), np.rad2deg(lon)


def rand_truth(nmax, rng):
    m, n = triangle_order(nmax)
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for k in range(len(m)):
        v = rng.standard_normal() / max(n[k], 1) ** 2
        C[n[k], m[k]] = v
        if m[k] >= 1:
            S[n[k], m[k]] = rng.standard_normal() / max(n[k], 1) ** 2
    return SHCoeffs(C, S)


def coeff_rel_err(a: SHCoeffs, b: SHCoeffs):
    L = max(a.nmax, b.nmax)
    a, b = a.truncate(L), b.truncate(L)
    num = np.linalg.norm(a.C - b.C) ** 2 + np.linalg.norm(a.S - b.S) ** 2
    den = np.linalg.norm(b.C) ** 2 + np.linalg.norm(b.S) ** 2
    return float(np.sqrt(num / den))


# ---------------------------------------------------------------------------
# 1. Legendre
# ---------------------------------------------------------------------------
def t_legendre():
    P = legendre_pbar([0.0, 30.0, 90.0], 4)
    m, n = triangle_order(4)
    def val(nn, mm, col):
        return P[col][np.where((n == nn) & (m == mm))[0][0]]
    exp = {
        (0, 0, 0): 1.0,
        (2, 0, 0): -np.sqrt(5) / 2,
        (2, 0, 1): np.sqrt(5) / 2 * (3 * 0.25 - 1),
        (2, 0, 2): np.sqrt(5),
        (1, 1, 0): np.sqrt(3),
        (2, 2, 0): np.sqrt(15) / 2,
    }
    worst = 0.0
    for (nn, mm, c), want in exp.items():
        worst = max(worst, abs(val(nn, mm, c) - want))
    check("Legendre matches closed-form 4pi-normalised values", worst < 1e-13,
          f"max abs err {worst:.2e}")


# ---------------------------------------------------------------------------
# 2. Driscoll-Healy global equiangular grid: exact quadrature
# ---------------------------------------------------------------------------
def t_dh_exact():
    rng = np.random.default_rng(1)
    nlat, nlon = 90, 180
    L = 20                                   # < nlat/2 - 1 = 44
    lat = -90.0 + np.arange(nlat) * (180.0 / nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    truth = rand_truth(L, rng)
    f = synthesize(LA.ravel(), LO.ravel(), truth.C, truth.S).reshape(nlat, nlon)

    a = dh_lat_weights(lat)
    check("Driscoll-Healy latitude weights sum to 2", abs(a.sum() - 2) < 1e-12,
          f"sum = {a.sum():.12f}")

    w = compute_weights(LA.ravel(), LO.ravel(), rule="dh", nlon=nlon)
    check("DH per-point weights sum to 4*pi", abs(w.total - FOUR_PI) < 1e-10,
          f"sum = {w.total:.12f}, coverage = {w.coverage:.12f}")

    est, rep = analysis(LA.ravel(), LO.ravel(), f.ravel(), L, method="quadrature",
                        weights=w)
    err = coeff_rel_err(est, truth)
    check("DH quadrature recovers a band-limited field", err < 1e-8,
          f"coeff rel err = {err:.3e}, gram max|K-I| = {rep.gram_deviation:.2e}")
    return lat, lon, LA, LO


# ---------------------------------------------------------------------------
# 3b. 截断就是截断：解到 nmax 必须**逐值等于**高阶解截断到 nmax
# ---------------------------------------------------------------------------
def t_truncation_is_exact():
    """真实 mascon 那种网格上，「60 阶看起来太光滑」不许是求解器压低出来的。

    用户报障：「现在重建的 60 阶场，有点不像 60 阶，信号泄露偏低很多」。
    在这份真实数据（CSR RL0603 mascon，720×1440，**纬度是单元中心** −89.875…89.875，
    nlon = 2·nlat 所以 auto 会选 ``dh``）上逐条量过：

    * 带限场闭环：nmax=60 的系数恢复到 3.7e-16，``gram max|K−I|`` ≈ 1.5e-15；
    * 直接解 60 阶 ≡ 240 阶解截断到 60 阶（max|ΔC| = 1e-28，重建场相对差 5.5e-18）；
    * 重建 RMS/输入 RMS 随阶数单调上升：60 阶 0.785 → 120 阶 0.904 → 240 阶 0.948。
      也就是说那 21.5% 是**真的在 60 阶以上**，不是被漏掉/压低的。

    这里用同构的小网格 + 宽带场把最后那条性质钉住：**解到 L 就是"高阶解截断到 L"**。
    任何"顺手压低/平滑"的改动（错的权重、多乘一次滤波、少一个归一化）都会让它变红。
    """
    rng = np.random.default_rng(11)
    nlat, nlon = 72, 144                     # nlon = 2*nlat、nlat 偶 → auto 选 dh
    step = 180.0 / nlat
    lat = 90.0 - step * (np.arange(nlat) + 0.5)      # 单元中心（与真实 mascon 同构）
    lon = 0.5 * step + step * np.arange(nlon)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    lat_u, lon_u = LA.ravel(), LO.ravel()
    f = rng.standard_normal(lat_u.size)              # 宽带场：功率一直到很高阶
    w = compute_weights(lat_u, lon_u, rule="dh", nlon=nlon)
    check("★ 单元中心 + dh：权重仍然归一（sum = 4π）",
          abs(w.total - FOUR_PI) < 1e-9,
          f"sum = {w.total:.10f}（纬度是单元中心，不含极点）")

    sols = {}
    rms = {}
    for L in (6, 12, 18):
        co, rep = analysis(lat_u, lon_u, f, L, method="quadrature", weights=w,
                           longitude_fft="auto")
        sols[L] = co
        back = synthesize(lat_u, lon_u, co.C, co.S)
        rms[L] = float(np.sqrt(np.mean(back ** 2)) / np.sqrt(np.mean(f ** 2)))
        if L == 18:
            check("宽带场在 18 阶上的 gram 仍接近单位（求积近似精确）",
                  float(rep.gram_deviation) < 1e-6,
                  f"max|K−I| = {rep.gram_deviation:.2e}")

    # 高阶解截断到 L，必须与直接解到 L 逐值相同
    C18 = np.asarray(sols[18].C)
    S18 = np.asarray(sols[18].S)
    err = 0.0
    for L in (6, 12):
        Cl = C18[:L + 1, :L + 1].copy()
        Sl = S18[:L + 1, :L + 1].copy()
        trunc = SHCoeffs(Cl, Sl)
        err = max(err, coeff_rel_err(sols[L], trunc))
    check("★ 解到 L ≡ 高阶解截断到 L（求解器不额外压低任何东西）",
          err < 1e-10, f"最大相对差 = {err:.3e}")

    check("★ 重建 RMS/输入 RMS 随阶数**单调上升**（少的那部分是真的在更高阶）",
          rms[6] < rms[12] < rms[18],
          "、".join(f"L={L}: {rms[L]:.4f}" for L in (6, 12, 18)))

    # 反向对照：如果谁"顺手"乘了一层级联滤波（例如把系数再乘一次 W），
    # 重建 RMS 就会比真截断低 —— 这里把那种偏差的尺度指出来。
    from shkit.filters import gaussian_coefficients
    W = gaussian_coefficients(300.0, 18)
    over = SHCoeffs(np.asarray(sols[18].C) * W[:, None],
                    np.asarray(sols[18].S) * W[:, None])
    back_over = synthesize(lat_u, lon_u, over.C, over.S)
    r_over = float(np.sqrt(np.mean(back_over ** 2)) / np.sqrt(np.mean(f ** 2)))
    check("对照：多乘一次 300 km 高斯会明显压低（>5%），所以上一条有区分度",
          r_over < 0.95 * rms[18],
          f"正常 {rms[18]:.4f} → 多乘一次 W {r_over:.4f}")


# ---------------------------------------------------------------------------
# 3. Gauss-Legendre x equiangular grid: exact quadrature
# ---------------------------------------------------------------------------
def t_glq_exact():
    rng = np.random.default_rng(2)
    L = 20
    lat, lon, w = glq_grid(L)
    truth = rand_truth(L, rng)
    f = synthesize(lat, lon, truth.C, truth.S)
    est, rep = analysis(lat, lon, f, L, method="quadrature", rule="user",
                        user_w=w)
    err = coeff_rel_err(est, truth)
    check("GLQ grid recovers a band-limited field exactly", err < 1e-9,
          f"n = {lat.size} points, coeff rel err = {err:.3e}")
    check("GLQ weights sum to 4*pi (user rule)", abs(w.sum() - FOUR_PI) < 1e-10,
          f"sum = {w.sum():.12f}")


# ---------------------------------------------------------------------------
# 3b. ``analysis()`` 也必须接受 weights_kw（GUI 的 dh 路径就靠它）
# ---------------------------------------------------------------------------
def t_analysis_weights_kw():
    """``analysis(..., rule='dh', weights_kw={'nlon': nlon})`` 必须能用。

    这条来自一个真实事故：GUI 的 ``AnalysisWorker`` 对全球等经纬网格会传
    ``weights_kw={'nlon': nlon}``（``dh`` 规则没有它取不到精确节点），而
    ``analysis()`` 是四个求解器里**唯一**漏掉这个参数的，于是"载入全球等经纬网格
    → 运行分析"这条最常见的路径直接抛
    ``TypeError: analysis() got an unexpected keyword argument 'weights_kw'``。
    三个兄弟函数（quadrature/wlsq/cg）一直都有它，所以这里也顺手比一遍：
    同一份数据经 ``analysis`` 与直接给 ``weights`` 的两条路必须逐值一致。
    """
    rng = np.random.default_rng(11)
    nlat, nlon = 60, 120
    L = 12
    lat = -90.0 + np.arange(nlat) * (180.0 / nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    truth = rand_truth(L, rng)
    f = synthesize(LA.ravel(), LO.ravel(), truth.C, truth.S)

    est, rep = analysis(LA.ravel(), LO.ravel(), f, L, method="quadrature",
                        rule="dh", weights_kw={"nlon": nlon},
                        report_fit=False)
    check("analysis(rule='dh', weights_kw={'nlon':…}) 不再抛 TypeError",
          est is not None and rep.weight_rule == "dh",
          f"rule = {rep.weight_rule}")
    check("weights_kw 传下去的 dh 权重确实覆盖全球",
          abs(rep.coverage - 1.0) < 1e-9, f"coverage = {rep.coverage:.12f}")
    check("weights_kw 路径恢复带限场（与先算 weights 的写法等价）",
          coeff_rel_err(est, truth) < 1e-8,
          f"coeff rel err = {coeff_rel_err(est, truth):.3e}")

    w = compute_weights(LA.ravel(), LO.ravel(), rule="dh", nlon=nlon)
    ref, _ = analysis(LA.ravel(), LO.ravel(), f, L, method="quadrature", weights=w,
                      report_fit=False)
    d = float(np.max(np.abs(est.C - ref.C)))
    check("``weights_kw`` 与显式 ``weights`` 两条路逐值相同", d < 1e-15,
          f"max|ΔC| = {d:.2e}")

    # 不给 nlon：库按 nlon = 2*nlat 推断（dh 的定义就是这个网格），
    # 对"恰好 nlon = 2*nlat"的网格结果必须与显式传入**逐值相同**；
    # 而网格不满足该条件时 **必须在报告里留下说明** —— 权重按 global 归一化到
    # 4π 之后覆盖率永远是 1.0，光看数字看不出规则已经不再精确。
    est2, rep2 = analysis(LA.ravel(), LO.ravel(), f, L, method="quadrature",
                          rule="dh", report_fit=False)
    d2 = float(np.max(np.abs(est2.C - est.C)))
    check("dh 不传 nlon 时按 2*nlat 推断，结果与显式传入逐值相同", d2 == 0.0,
          f"max|ΔC| = {d2:.2e}")

    nlon_bad = 100                       # 2*nlat = 120 != 100
    lon_bad = np.arange(nlon_bad) * (360.0 / nlon_bad)
    LAb, LOb = np.meshgrid(lat, lon_bad, indexing="ij")
    fb = synthesize(LAb.ravel(), LOb.ravel(), truth.C, truth.S)
    _, rep_bad = analysis(LAb.ravel(), LOb.ravel(), fb, 6, method="quadrature",
                          rule="dh", report_fit=False)
    hit = [w for w in rep_bad.warnings if "2*nlat" in w]
    check("dh 用在 nlon != 2*nlat 的网格上时报告里出现提醒（不再被吞掉）",
          bool(hit), (hit[0][:70] if hit else f"warnings={len(rep_bad.warnings)} 条"))


# ---------------------------------------------------------------------------
# 4. Voronoi vs uniform area on random / clustered points
# ---------------------------------------------------------------------------
def t_weight_rules():
    rng = np.random.default_rng(3)
    L = 12
    N = 3000
    truth = rand_truth(8, rng)

    lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, N)))
    lon = rng.uniform(0, 360, N)
    f = synthesize(lat, lon, truth.C, truth.S)

    wv = voronoi_weights(lat, lon)
    wu = uniform_weights(N)
    check("Voronoi cell areas sum to 4*pi", abs(wv.sum() - FOUR_PI) < 1e-9,
          f"sum = {wv.sum():.12f}, heterogeneity = {wv.max()/wv.min():.1f}")
    check("Delaunay 1/3 weights also sum to 4*pi on a global set",
          abs(delaunay_weights(lat, lon).sum() - FOUR_PI) < 1e-9,
          f"sum = {delaunay_weights(lat, lon).sum():.12f}")

    r = {}
    for tag, w in (("uniform", wu), ("voronoi", wv)):
        est, rep = analysis_quadrature(lat, lon, f, L,
                                       weights=compute_weights(lat, lon,
                                                               rule="user",
                                                               user_w=w,
                                                               normalise="global"))
        r[tag] = coeff_rel_err(est, truth)
    check("Voronoi weights beat a flat area on random scattered points",
          r["voronoi"] < r["uniform"] / 3,
          f"uniform {r['uniform']:.3e} -> voronoi {r['voronoi']:.3e} "
          f"(x{r['uniform']/r['voronoi']:.1f} better)")

    # clustered: 70% inside a 35-degree cap
    n1 = int(0.7 * N)
    cla, clo, cr = np.deg2rad(40.0), np.deg2rad(100.0), np.deg2rad(35.0)
    pts = []
    while sum(len(p[0]) for p in pts) < n1:
        z = rng.uniform(np.sin(cla - cr), np.sin(cla + cr), 2048)
        x = rng.uniform(0, 2 * np.pi, 2048)
        la2 = np.rad2deg(np.arcsin(z))
        lo2 = np.rad2deg(x)
        cosd = (np.sin(np.deg2rad(la2)) * np.sin(cla) +
                np.cos(np.deg2rad(la2)) * np.cos(cla) * np.cos(np.deg2rad(lo2) - clo))
        pts.append((la2[cosd >= np.cos(cr)], lo2[cosd >= np.cos(cr)]))
    la1 = np.concatenate([p[0] for p in pts])[:n1]
    lo1 = np.concatenate([p[1] for p in pts])[:n1]
    la = np.concatenate([la1, np.rad2deg(np.arcsin(rng.uniform(-1, 1, N - n1)))])
    lo = np.concatenate([lo1, rng.uniform(0, 360, N - n1)])
    f2 = synthesize(la, lo, truth.C, truth.S)
    r2 = {}
    for tag, w in (("uniform", uniform_weights(N)),
                   ("voronoi", voronoi_weights(la, lo))):
        est, _ = analysis_quadrature(la, lo, f2, L,
                                     weights=compute_weights(la, lo, rule="user",
                                                             user_w=w,
                                                             normalise="global"))
        r2[tag] = coeff_rel_err(est, truth)
    check("Voronoi weights beat a flat area on clustered points",
          r2["voronoi"] < r2["uniform"] / 3,
          f"uniform {r2['uniform']:.3e} -> voronoi {r2['voronoi']:.3e} "
          f"(x{r2['uniform']/r2['voronoi']:.1f} better)")


# ---------------------------------------------------------------------------
# 5. The regional Voronoi trap: Voronoi still sums to 4pi, Delaunay does not
# ---------------------------------------------------------------------------
def t_regional_weights():
    lat, lon = fib_sphere(4000)
    cla, clo, cr = np.deg2rad(30.0), np.deg2rad(100.0), np.deg2rad(30.0)
    cosd = (np.sin(np.deg2rad(lat)) * np.sin(cla) +
            np.cos(np.deg2rad(lat)) * np.cos(cla) * np.cos(np.deg2rad(lon) - clo))
    sel = cosd >= np.cos(cr)
    cap_area = 2 * np.pi * (1 - np.cos(cr))

    wv = voronoi_weights(lat[sel], lon[sel])
    wd = delaunay_weights(lat[sel], lon[sel])
    check("REGIONAL Voronoi areas wrongly sum to 4*pi", abs(wv.sum() - FOUR_PI) < 1e-6,
          f"sum = {wv.sum():.6f} but true cap area = {cap_area:.6f} "
          f"(inflation x{wv.sum()/cap_area:.1f})")
    check("Delaunay weights on a region sum to ~the cap area (spherical hull)",
          abs(wd.sum() - cap_area) / cap_area < 0.15,
          f"sum = {wd.sum():.6f} vs cap area {cap_area:.6f} "
          f"(rel err {abs(wd.sum()-cap_area)/cap_area:.2%}; the spherical "
          "convex hull is inscribed in the cap, so it is systematically "
          "smaller - do not treat it as an exact region area)")

    # grid-cell weights on a regional equiangular grid
    la = np.arange(-20.0, 20.0 + 1e-9, 1.0)
    lo = np.arange(95.0, 105.0 + 1e-9, 1.0)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    wg = grid_cell_weights(LA.ravel(), LO.ravel())
    # cell edges: lat -20.5..20.5, lon 94.5..105.5 -> dlambda = 11 deg
    want = np.deg2rad(11.0) * (np.sin(np.deg2rad(20.5)) - np.sin(np.deg2rad(-20.5)))
    check("grid-cell weights on a regional grid match the band area",
          abs(wg.sum() - want) < 1e-12,
          f"sum = {wg.sum():.10f} vs exact {want:.10f}")
    # and the classic dlat*dlon*cos(lat) shortcut is biased HIGH by (dlat)^2/24
    w_naive = np.deg2rad(1.0) ** 2 * np.cos(np.deg2rad(LA.ravel()))
    bias = (w_naive.sum() - wg.sum()) / wg.sum()
    check("dlat*dlon*cos(lat) overestimates by +(dlat)^2/24",
          abs(bias - (np.deg2rad(1.0) ** 2) / 24) < 1e-8,
          f"measured bias {bias:.3e} vs theory {np.deg2rad(1.0)**2/24:.3e}")


# ---------------------------------------------------------------------------
# 6. Least squares / CG
# ---------------------------------------------------------------------------
def t_lsq():
    rng = np.random.default_rng(6)
    lat, lon = fib_sphere(1500)
    L = 12
    truth = rand_truth(L, rng)
    f = synthesize(lat, lon, truth.C, truth.S)

    w = compute_weights(lat, lon, rule="voronoi")
    e_w, r_w = analysis_wlsq(lat, lon, f, L, weights=w, method="wlsq")
    e_c, r_c = analysis_wlsq(lat, lon, f, L, weights=w, method="cg")
    err_w = coeff_rel_err(e_w, truth)
    err_c = coeff_rel_err(e_c, truth)
    check("WLSQ recovers a band-limited field", err_w < 1e-8,
          f"n={lat.size}, ncoef={(L+1)**2}, coeff rel err = {err_w:.3e}, "
          f"cond = {r_w.condition_number:.2e}")
    check("matrix-free CG matches the direct solve", err_c < 1e-6,
          f"coeff rel err = {err_c:.3e}, cg_info = {r_c.meta.get('cg_info')}")

    # noise: error should follow (L+1)/sqrt(N) / sqrt(snr)
    rng2 = np.random.default_rng(7)
    data_rms = float(np.sqrt(np.mean(f ** 2)))
    for lvl in (0.01, 0.05):
        fn = f + rng2.standard_normal(lat.size) * lvl * data_rms
        e, r = analysis_wlsq(lat, lon, fn, L, weights=w, method="wlsq")
        rel = coeff_rel_err(e, truth)
        pred = (L + 1) / np.sqrt(lat.size) * lvl
        print(f"        noise {lvl:.0%}: measured coeff rel err {rel:.3e}, "
              f"order-of-magnitude prediction {pred:.3e}")


# ---------------------------------------------------------------------------
# 7. Synthesis round trip + triangle I/O
# ---------------------------------------------------------------------------
def t_roundtrip():
    rng = np.random.default_rng(8)
    lat, lon = fib_sphere(1200)
    L = 10
    truth = rand_truth(L, rng)
    f = synthesize(lat, lon, truth.C, truth.S)
    w = compute_weights(lat, lon, rule="voronoi")

    res = {}
    for tag, kw in (("quadrature", dict(method="quadrature")),
                    ("iterative x1", dict(method="iterative", niter=1)),
                    ("iterative x3", dict(method="iterative", niter=3)),
                    ("wlsq", dict(method="wlsq"))):
        est, rep = analysis(lat, lon, f, L, weights=w, **kw)
        f2 = synthesize(lat, lon, est.C, est.S)
        res[tag] = float(np.sqrt(np.mean((f2 - f) ** 2)) /
                         np.sqrt(np.mean(f ** 2)))
    check("round trip converges with iterative correction",
          res["iterative x1"] < 1e-5 and res["iterative x3"] < 1e-8,
          "relative RMS: " + ", ".join(f"{k}={v:.2e}" for k, v in res.items()))
    check("WLSQ round trip is exact on scattered points",
          res["wlsq"] < 1e-12, f"relative RMS = {res['wlsq']:.2e}")
    check("iterative correction beats plain quadrature on scattered points",
          res["iterative x1"] < res["quadrature"] / 5,
          f"{res['quadrature']:.2e} -> {res['iterative x1']:.2e} "
          f"(x{res['quadrature']/res['iterative x1']:.1f})")

    est_auto, rep_auto = analysis(lat, lon, f, L, method="auto", weights=w)
    check("method='auto' picks a defensible branch",
          rep_auto.meta.get("auto_choice") in ("quadrature", "projection",
                                               "wlsq", "cg"),
          f"picked '{rep_auto.meta.get('auto_choice')}'")

    tri = truth.to_triangle()
    back = SHCoeffs.from_triangle(tri, L)
    check("triangle (m2py) layout round trip is lossless",
          coeff_rel_err(back, truth) < 1e-15,
          f"2*NC rows = {tri.shape[0]}, NC = {(L+1)*(L+2)//2}")


# ---------------------------------------------------------------------------
# 8. Regional honesty: what a regional solve can and cannot do
# ---------------------------------------------------------------------------
def t_regional_honesty():
    rng = np.random.default_rng(9)
    lat, lon = fib_sphere(4000)
    L = 12
    truth = rand_truth(8, rng)
    f = synthesize(lat, lon, truth.C, truth.S)
    cla, clo, cr = np.deg2rad(30.0), np.deg2rad(100.0), np.deg2rad(30.0)
    cosd = (np.sin(np.deg2rad(lat)) * np.sin(cla) +
            np.cos(np.deg2rad(lat)) * np.cos(cla) * np.cos(np.deg2rad(lon) - clo))
    sel = cosd >= np.cos(cr)
    la, lo, fs = lat[sel], lon[sel], f[sel]

    wd = compute_weights(la, lo, rule="delaunay", normalise="region")
    est, rep = analysis(la, lo, fs, L, method="quadrature", weights=wd)
    check("regional quadrature reports its reduced coverage",
          abs(rep.coverage - (1 - np.cos(cr)) / 2) < 0.02,
          f"coverage = {rep.coverage:.4f} (true cap fraction "
          f"{(1-np.cos(cr))/2:.4f})")
    check("regional quadrature raises the coverage warning",
          any("of the sphere" in m for m in rep.warnings),
          f"{len(rep.warnings)} warning(s)")

    wls = analysis_wlsq(la, lo, fs, L, weights=wd, method="wlsq")
    _, rep_w = wls
    check("regional WLSQ is ill-conditioned and says so",
          rep_w.condition_number > 1e6 or rep_w.warnings,
          f"cond = {rep_w.condition_number:.2e}, "
          f"recommended nmax = {rep_w.lmax_recommended}, "
          f"warnings = {len(rep_w.warnings)}")


def t_multitime():
    """Regression: the three-dimensional code path (nmax+1, nmax+1, ntime).

    A naive ``S[..., :, 0] = 0`` in SHCoeffs.__post_init__ used to zero the
    whole first time slice instead of the m=0 column.
    """
    L = 6
    lat, lon, w = glq_grid(L)
    truths = [rand_truth(L, np.random.default_rng(s)) for s in (1, 2, 3)]
    F = np.column_stack([synthesize(lat, lon, t.C, t.S) for t in truths])
    est, rep = analysis(lat, lon, F, L, method="quadrature", rule="user",
                        user_w=w)
    check("multi-time analysis keeps every time slice",
          est.ntime == 3 and est.C.shape == (L + 1, L + 1, 3),
          f"C shape {est.C.shape}, ntime {est.ntime}")
    worst_C = worst_S = 0.0
    for k, t in enumerate(truths):
        worst_C = max(worst_C, float(np.abs(est.C[:, :, k] - t.C).max()))
        worst_S = max(worst_S, float(np.abs(est.S[:, :, k] - t.S).max()))
    check("multi-time coefficients are correct in C and in S",
          worst_C < 1e-12 and worst_S < 1e-12,
          f"max|dC| = {worst_C:.2e}, max|dS| = {worst_S:.2e} "
          "(dS used to be O(1) on the first slice)")
    check("multi-time sine part of order 0 stays structurally zero",
          np.all(est.S[:, 0, :] == 0.0),
          f"max|S[:,0,:]| = {np.abs(est.S[:, 0, :]).max():.1e}")

    tri = est.to_triangle()
    back = SHCoeffs.from_triangle(tri, L)
    check("multi-time triangle round trip is lossless",
          back.ntime == 3 and np.abs(back.C - est.C).max() < 1e-15
          and np.abs(back.S - est.S).max() < 1e-15,
          f"triangle shape {tri.shape}")

    e2, _ = analysis_wlsq(lat, lon, F, L, weights=compute_weights(
        lat, lon, rule="user", user_w=w, normalise="global"), method="wlsq")
    check("multi-time WLSQ is correct too",
          max(np.abs(e2.C[:, :, k] - t.C).max()
              for k, t in enumerate(truths)) < 1e-12,
          "max|dC| = "
          f"{max(np.abs(e2.C[:, :, k] - t.C).max() for k, t in enumerate(truths)):.2e}")


def t_residual_reporting():
    """Regression: (N,) - (N,1) used to broadcast to (N,N) in the quadrature
    branch, inflating every reported residual."""
    L = 10
    lat, lon = fib_sphere(800)
    truth = rand_truth(L, np.random.default_rng(31))
    f = synthesize(lat, lon, truth.C, truth.S)
    w = compute_weights(lat, lon, rule="voronoi")
    data_rms = float(np.sqrt(np.mean(f ** 2)))
    for tag, kw in (("quadrature", dict(method="quadrature")),
                    ("iterative", dict(method="iterative", niter=2)),
                    ("wlsq", dict(method="wlsq"))):
        est, rep = analysis(lat, lon, f, L, weights=w, **kw)
        fit = np.asarray(synthesize(lat, lon, est.C, est.S)).ravel()
        independent = float(np.sqrt(np.mean((fit - f) ** 2)))
        # compare against the data scale: two ~1e-15 residuals cannot be
        # compared relatively
        spread = abs(independent - rep.residual_rms) / data_rms
        check(f"reported residual matches an independent computation ({tag})",
              spread < 1e-10,
              f"report {rep.residual_rms:.6e} vs independent {independent:.6e} "
              f"(abs diff / data RMS = {spread:.1e})")

    # and the same for a multi-time-slice call
    F = np.column_stack([f, synthesize(lat, lon, rand_truth(L, np.random.default_rng(32)).C,
                                       rand_truth(L, np.random.default_rng(32)).S)])
    est, rep = analysis(lat, lon, F, L, method="quadrature", weights=w)
    fit = np.asarray(synthesize(lat, lon, est.C, est.S))
    independent = float(np.sqrt(np.mean((fit - F) ** 2)))
    check("reported residual is right for multi-time data too",
          abs(independent - rep.residual_rms) /
          max(float(np.sqrt(np.mean(F ** 2))), 1e-300) < 1e-10,
          f"report {rep.residual_rms:.6e} vs independent {independent:.6e}")


def t_packaging_entry():
    """打包入口必须是包外的 launcher，否则冻结后一启动就崩。

    PyInstaller 把**入口脚本**当顶层脚本执行（没有父包）。若拿
    ``shkit/gui/app.py`` 当入口，它里面的 ``from .. import __version__``
    会直接抛 ``ImportError: attempted relative import with no known parent package``
    —— 而且因为没有控制台窗口，用户只看到程序"闪一下"什么都没发生。
    所以入口固定在 ``packaging/shkit_launcher.py``，这里守住这条约束。
    """
    import ast
    import os

    root = __file__.rsplit("tests", 1)[0]
    spec = os.path.join(root, "packaging", "shkit.spec")
    launch = os.path.join(root, "packaging", "shkit_launcher.py")

    check("打包入口文件存在", os.path.isfile(launch),
          os.path.relpath(launch, root))
    check("spec 用包外 launcher 当入口（不用 shkit/gui/app.py）",
          os.path.isfile(spec)
          and "shkit_launcher.py" in open(spec, encoding="utf-8").read()
          and "shkit\", \"gui\", \"app.py" not in open(spec, encoding="utf-8").read(),
          "Analysis([...]) 指向 packaging/shkit_launcher.py")

    # launcher 必须真的导入 shkit.gui.app 再调 main，且自身不能有相对导入
    if os.path.isfile(launch):
        src = open(launch, encoding="utf-8").read()
        tree = ast.parse(src, launch)
        rel = [n for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and (n.level or 0) > 0]
        check("launcher 自身不含相对导入（它是顶层脚本）", not rel,
              f"{len(rel)} 处相对导入" if rel else "无")
        check("launcher 调用 shkit.gui.app.main",
              "from shkit.gui.app import main" in src, "from shkit.gui.app import main")

    # app.py 提供 --self-test（打包脚本靠它验证冻结版）
    app = os.path.join(root, "shkit", "gui", "app.py")
    check("app.main 支持 --self-test（打包后自检用）",
          os.path.isfile(app) and '"--self-test"' in open(app, encoding="utf-8").read(),
          "打包脚本用 SHKit.exe --self-test 验证冻结版")

    # 打包脚本要跑这个自检
    build = os.path.join(root, "packaging", "build_installer.ps1")
    check("打包脚本会跑冻结版自检",
          os.path.isfile(build) and "--self-test" in open(build, encoding="utf-8-sig").read(),
          "build_installer.ps1 里执行 SHKit.exe --self-test")


def main():
    t0 = time.time()
    tests = [t_legendre, t_dh_exact, t_glq_exact, t_truncation_is_exact,
             t_analysis_weights_kw,
             t_weight_rules,
             t_regional_weights, t_lsq, t_roundtrip, t_multitime,
             t_residual_reporting, t_regional_honesty, t_packaging_entry]
    for fn in tests:
        print(f"\n=== {fn.__name__} " + "=" * (60 - len(fn.__name__)))
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(fn.__name__, False, "raised an exception")
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'='*76}\n{npass}/{len(RESULTS)} checks passed in {time.time()-t0:.1f}s")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
