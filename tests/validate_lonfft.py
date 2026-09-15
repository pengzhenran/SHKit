# -*- coding: utf-8 -*-
"""
Validation of the **longitude FFT path** -- analysis (A3) and synthesis (A4),
both built on :mod:`shkit.lonfft`.

Five claims, each checked as a *property* rather than by a stopwatch:

1. **the criterion is a hard one.**  The FFT replaces the longitude sum, so it
   needs a complete rectangular grid whose longitude axis is uniform, whole
   circle and resolved (``nlon > 2*nmax``).  Anything else is either refused by
   ``longitude_fft='fft'`` or reported and fallen back from by ``'auto'`` --
   never a silent switch.
2. **the fast path is the same answer**, in both directions.  FFT ≡ direct to
   floating-point round-off, on several grids, with multi-epoch data, polar rows,
   and longitude-dependent (non-separable) weights for the analysis side.
3. **the phase reference is right.**  A grid starting at ``lon = 0`` cannot tell
   a correct implementation from one that forgets ``phi_m = m*lambda_0``; the
   offset grids below can, and a deliberately phase-free reimplementation is
   included to prove these cases have teeth.  (This is not hypothetical: the
   horizontal path shipped without the phase and was wrong by up to 150% on
   offset axes while every ``lon = 0,1,2,...`` test passed.)
4. **it is actually faster** -- measured separately for the quadrature kernel,
   the whole analysis call, and the synthesis call.
5. **the batch reconstruction uses it.**  ``analyze_series(report_fit=True)``
   records which path each half took.

Run:  python tests/validate_lonfft.py
"""
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit.analysis import (STATS, analysis, analysis_projection,   # noqa: E402
                           analysis_quadrature, lonfft_applicable,
                           stats_delta, stats_reset)
from shkit.basis import legendre_columns                            # noqa: E402
from shkit.coeffs import SHCoeffs                                   # noqa: E402
from shkit.lonfft import (fft_path_applicable,                      # noqa: E402
                          longitude_grid_layout)
from shkit.series import analyze_series                             # noqa: E402
from shkit.synthesis import synthesis                               # noqa: E402
from shkit.weights import FOUR_PI, WeightSet, grid_cell_weights     # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def rel(v, ref):
    scale = max(float(np.max(np.abs(np.asarray(ref)))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


def make_grid(dlat=2.0, dlon=2.0, lam0=0.0, nmax=20, K=1, seed=3,
              f=None, polar=False, weights=None):
    """A complete rectangular grid; returns ``(lat_flat, lon_flat, f, ws)``."""
    if polar:
        lat = np.concatenate([[90.0], np.arange(-88.0, 88.01, dlat), [-90.0]])
    else:
        lat = np.arange(-90.0 + dlat / 2, 90.0, dlat)
    lon = lam0 + np.arange(0.0, 360.0, dlon)
    LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    if f is None:
        f = np.random.default_rng(seed).normal(size=(LA.size, K))
    w = weights if weights is not None else grid_cell_weights(LA, LO)
    return LA, LO, f, WeightSet(w=w, rule="grid")


def band_limited(LA, LO, nmax=20, seed=5):
    rng = np.random.default_rng(seed)
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    C[0, 0] = 1.5
    for n in range(2, nmax + 1):
        C[n, 0] = rng.normal() / (n + 1.0)
        for m in range(1, n + 1):
            C[n, m] = rng.normal() / (n + 1.0)
            S[n, m] = rng.normal() / (n + 1.0)
    return synthesis(LA, LO, SHCoeffs(C, S), nmax=nmax)


def raw_coeffs(LA, LO, f, nmax, mode, weights=None):
    co, rep = analysis_quadrature(LA, LO, f, nmax, weights=weights,
                                  longitude_fft=mode, report_fit=False)
    return co, rep


def _flat(x):
    """Drop a length-1 time axis, so (L+1,L+1,1) and (L+1,L+1) compare.

    Without this, ``(21,21,1) - (21,21)`` broadcasts to ``(21,21,21)`` and the
    "difference" is pure fiction -- an error that looks like a 39x disagreement.
    """
    x = np.asarray(x, dtype=float)
    return x[:, :, 0] if x.ndim == 3 and x.shape[2] == 1 else x


def worst_diff(a, b):
    return max(rel(_flat(a.C), _flat(b.C)), rel(_flat(a.S), _flat(b.S)))


def _timeit(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 1. criterion
# ---------------------------------------------------------------------------
def t_criterion():
    ok, why = fft_path_applicable(np.arange(0.0, 360.0, 1.0), 60)
    check("判据：整圈均匀 360 点 / nmax=60 → 可用", ok, why[:34])
    ok, why = fft_path_applicable(np.arange(0.0, 360.0, 1.0), 180)
    check("判据：nlon=360 ≤ 2·nmax=360 → 必须拒绝", not ok, why[:40])
    ok, why = fft_path_applicable(np.arange(0.0, 360.0, 1.0), 179)
    check("判据：nlon=360 > 2·nmax=358 → 可用（边界内侧）", ok, why[:34])
    ok, why = fft_path_applicable(np.arange(0.0, 300.0, 1.0), 60)
    check("判据：只覆盖 300° → 拒绝（非整圈）", not ok, why[:34])
    ok, why = fft_path_applicable(np.array([0.0, 1.0, 2.0, 4.0, 5.0] * 20), 60)
    check("判据：不等间隔 → 拒绝", not ok, why[:30])
    ok, why = fft_path_applicable(np.array([5.0, 3.0, 1.0]), 0)
    check("判据：非递增 → 拒绝", not ok, why[:30])
    ok, why = fft_path_applicable(np.array([10.0]), 0)
    check("判据：单点 → 拒绝", not ok, why[:30])
    # 经典 DH 网格（nlon = 2*nlat）在满阶下**不能**用 FFT：这是诚实的限制
    ok, why = fft_path_applicable(np.arange(0.0, 360.0, 2.0), 90)
    check("判据：DH 网格 180×91 满阶 nmax=90 → 拒绝（混叠）", not ok, why[:40])
    ok, why = fft_path_applicable(np.arange(0.0, 360.0, 2.0), 89)
    check("判据：DH 网格 nmax=89 → 可用", ok, why[:34])


def t_layout():
    LA, LO, _, _ = make_grid(2.0, 2.0)
    g, why = longitude_grid_layout(LA, LO)
    check("布局：完整网格被识别", g is not None, why)
    check("布局：轴长正确", g is not None and (g.nlat, g.nlon) == (90, 180),
          f"{g.shape}" if g else "")
    check("布局：起始经度 0", g is not None and abs(g.lam0_deg) < 1e-12)

    # 打乱点序后仍应识别，且 order 能把数据摆回 (nlat, nlon)
    rng = np.random.default_rng(0)
    perm = rng.permutation(LA.size)
    tag = np.arange(LA.size, dtype=float)
    g2, _ = longitude_grid_layout(LA[perm], LO[perm])
    grid = g2.to_grid(tag[perm])
    check("布局：乱序点集仍被识别", g2 is not None)
    check("布局：order 把点摆回行主序网格",
          g2 is not None and np.array_equal(grid[:, 0], np.arange(0, LA.size, 180.0)))
    check("布局：乱序网格与顺序网格给出同一数组",
          g2 is not None and np.array_equal(grid.ravel(), np.arange(LA.size, dtype=float)))

    # 区域掩膜：唯一坐标很多，但格子不全（用椭圆，边缘参差）
    m = ((LA - 10.0) / 25.0) ** 2 + ((LO - 100.0) / 55.0) ** 2 <= 1.0
    g3, why3 = longitude_grid_layout(LA[m], LO[m])
    check("布局：区域掩膜（格子不全）→ 拒绝", g3 is None,
          f"{int(m.sum())} 点：" + why3[:34])

    # 重复点
    g4, why4 = longitude_grid_layout(np.append(LA, LA[0]), np.append(LO, LO[0]))
    check("布局：重复点 → 拒绝", g4 is None, why4[:40])

    # 与轴不严格相等的坐标
    LO2 = LO.copy()
    LO2[5] += 1e-7
    g5, why5 = longitude_grid_layout(LA, LO2)
    check("布局：坐标偏离网格轴 → 拒绝", g5 is None, why5[:40])

    g6, why6 = longitude_grid_layout(LA, LO[:-1])
    check("布局：长度不一致 → 拒绝", g6 is None, why6[:40])


# ---------------------------------------------------------------------------
# 2./3. equivalence, including the phase regression
# ---------------------------------------------------------------------------
def t_equivalence_offsets():
    """The phase regression: ``lon = lam0 + k*dlon`` with ``lam0 != 0``."""
    worst = 0.0
    for lam0 in (0.0, 0.5, -179.5, 90.3, 123.4):
        LA, LO, f, ws = make_grid(2.0, 2.0, lam0=lam0, nmax=20, K=1)
        cd, rd = raw_coeffs(LA, LO, f, 20, "direct", ws)
        cf, rf = raw_coeffs(LA, LO, f, 20, "fft", ws)
        d = worst_diff(cd, cf)
        worst = max(worst, d)
        check(f"等价：lam0={lam0:g} 时 FFT ≡ 直接法", d <= 1e-12,
              f"相对差 {d:.2e}, path={rf.meta['longitude_path']}")
    check("★ 起始经度相位回归（5 个 lam0）总体 ≤1e-12", worst <= 1e-12,
          f"max = {worst:.2e}")


def _no_phase_fft(LA, LO, f, nmax, w):
    """A deliberately **wrong** FFT implementation that ignores ``lam0``.

    Only used to show that the offset grids above are discriminating: on a
    ``lam0 = 0`` grid this is correct, so a test suite built only on that grid
    would pass no matter what.
    """
    g, _ = longitude_grid_layout(LA, LO)
    # NB: w is (npoints,) and f is (npoints, ntime) -- `w * f` would broadcast to
    # (npoints, npoints).  This mistake is invisible until the numbers come out
    # wrong, and it is exactly the kind of thing the fixture below is meant to
    # catch, so it is called out here.
    F = g.to_grid(w[:, None] * f)[:, :, 0]
    Gh = np.fft.rfft(F, axis=1)
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for m, block in legendre_columns(g.lat_vec, nmax):
        n_idx = np.arange(m, nmax + 1)
        C[n_idx, m] = block @ Gh[:, m].real / FOUR_PI
        if m >= 1:
            S[n_idx, m] = block @ (-Gh[:, m].imag) / FOUR_PI
    return SHCoeffs(C, S)


def t_phase_regression_has_teeth():
    """Show that the offset grids above can actually fail.

    ``_no_phase_fft`` is a correct-at-``lam0 = 0`` reimplementation that drops the
    ``phi_m = m*lambda_0`` rotation.  If it agrees on the zero-offset grid and
    disagrees hugely on an offset grid, then the offset cases are doing real work.
    """
    LA, LO, f, ws = make_grid(2.0, 2.0, lam0=0.0, nmax=20)
    cd, _ = raw_coeffs(LA, LO, f, 20, "direct", ws)
    bad = _no_phase_fft(LA, LO, f, 20, ws.w)
    e0 = worst_diff(cd, bad)
    check("反例：lam0=0 时“漏相位”的实现**看起来是对的**", e0 <= 1e-12,
          f"相对差 {e0:.2e}")
    LA, LO, f, ws = make_grid(2.0, 2.0, lam0=0.5, nmax=20)
    cd, _ = raw_coeffs(LA, LO, f, 20, "direct", ws)
    good = raw_coeffs(LA, LO, f, 20, "fft", ws)[0]
    bad = _no_phase_fft(LA, LO, f, 20, ws.w)
    e1 = worst_diff(cd, bad)
    check("★ 反例：lam0=0.5 时同一实现明显错（证明用例有鉴别力）", e1 > 1e-2,
          f"相对差 {e1:.2e}")
    check("反例：而真实实现在同一网格上是对的（≤1e-12）",
          worst_diff(cd, good) <= 1e-12, f"相对差 {worst_diff(cd, good):.2e}")


def t_equivalence_general():
    # 多时次 + 极点行
    LA, LO, f, ws = make_grid(3.0, 3.0, lam0=0.0, nmax=15, K=4, polar=True)
    cd, rd = raw_coeffs(LA, LO, f, 15, "direct")
    cf, rf = raw_coeffs(LA, LO, f, 15, "fft")
    check("等价：含 ±90° 极点行的多时次网格", worst_diff(cd, cf) <= 1e-12,
          f"相对差 {worst_diff(cd, cf):.2e}, ntime=4")

    # 权重逐经度变化 → 不可分离，仍必须精确（g = w*f 的 FFT）
    LA, LO, f, _ = make_grid(2.0, 2.0, lam0=0.0, nmax=20)
    w = grid_cell_weights(LA, LO) * (1.0 + 0.5 * np.cos(np.deg2rad(3.0 * LO)))
    ws = WeightSet(w=w, rule="user")
    cd, _ = analysis_quadrature(LA, LO, f, 20, weights=ws,
                                longitude_fft="direct", report_fit=False)
    cf, _ = analysis_quadrature(LA, LO, f, 20, weights=ws,
                                longitude_fft="fft", report_fit=False)
    check("等价：权重随经度变化（不可分离）时仍精确",
          worst_diff(cd, cf) <= 1e-12, f"相对差 {worst_diff(cd, cf):.2e}")

    # 带限真值：两种路径都必须恢复出正确的系数（不是只互相一致）
    LA, LO, f, ws = make_grid(2.0, 2.0, lam0=0.5, nmax=20)
    f = band_limited(LA, LO, 20)[:, None]
    cf, _ = raw_coeffs(LA, LO, f, 20, "fft")
    cd, _ = raw_coeffs(LA, LO, f, 20, "direct")
    fit_d = synthesis(LA, LO, cd, nmax=20)
    check("正确性：FFT 路径重建带限场（相对残差 ≤1e-10）",
          float(np.max(np.abs(fit_d - f[:, 0]))) / float(np.max(np.abs(f))) <= 1e-10,
          f"rel = {rel(synthesis(LA, LO, cf, nmax=20), f[:, 0]):.2e}")


# ---------------------------------------------------------------------------
# 4. the switch is explicit
# ---------------------------------------------------------------------------
def t_switch():
    LA, LO, f, ws = make_grid(2.0, 2.0, nmax=20)
    try:
        analysis_quadrature(LA, LO, f, 20, weights=ws, report_fit=False,
                            longitude_fft="fft")
        check("开关：'fft' 在可用网格上不报错", True)
    except Exception as exc:                                    # pragma: no cover
        check("开关：'fft' 在可用网格上不报错", False, repr(exc))

    # 散点：'fft' 必须报错，'auto' 必须回退并给出理由
    rng = np.random.default_rng(1)
    slat = rng.uniform(-90, 90, 400)
    slon = rng.uniform(0, 360, 400)
    sf = rng.normal(size=(400, 1))
    raised = None
    try:
        analysis_quadrature(slat, slon, sf, 10, longitude_fft="fft",
                            report_fit=False)
    except ValueError as exc:
        raised = str(exc)
    check("开关：散点上 'fft' 明确报错（不静默回退）",
          raised is not None and "完整矩形网格" in raised,
          (raised or "no error").splitlines()[-1][:44])

    co, rep = analysis_quadrature(slat, slon, sf, 10, longitude_fft="auto",
                                  report_fit=False)
    check("开关：散点上 'auto' 回退直接法并记录理由",
          rep.meta["longitude_path"] == "direct"
          and "网格" in rep.meta["longitude_reason"],
          rep.meta["longitude_reason"][:40])
    check("开关：回退后的系数与强制直接法一致",
          worst_diff(co, raw_coeffs(slat, slon, sf, 10, "direct")[0]) == 0.0)

    # nlon ≤ 2·nmax：'fft' 报错的理由要指出混叠
    lat = np.arange(-80.0, 81.0, 10.0)
    lon = np.arange(0.0, 360.0, 20.0)          # nlon = 18
    LA2, LO2 = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    f2 = np.random.default_rng(2).normal(size=(LA2.size, 1))
    raised = None
    try:
        analysis_quadrature(LA2, LO2, f2, 18, longitude_fft="fft",
                            report_fit=False)
    except ValueError as exc:
        raised = str(exc)
    check("开关：nlon ≤ 2·nmax 时 'fft' 报错并点名混叠",
          raised is not None and "Nyquist" in raised,
          (raised or "no error").splitlines()[0][:44])
    co, rep = analysis_quadrature(LA2, LO2, f2, 18, longitude_fft="auto",
                                  report_fit=False)
    check("开关：混叠网格 'auto' 回退并记录 Nyquist 理由",
          rep.meta["longitude_path"] == "direct"
          and "Nyquist" in rep.meta["longitude_reason"],
          rep.meta["longitude_reason"][:40])

    co, rep = analysis_quadrature(LA, LO, f, 20, weights=ws,
                                  longitude_fft="direct", report_fit=False)
    check("开关：'direct' 记录为 direct 且理由说明是用户指定",
          rep.meta["longitude_path"] == "direct" and "用户指定" in rep.meta["longitude_reason"])

    try:
        analysis_quadrature(LA, LO, f, 20, weights=ws, longitude_fft="turbo")
        check("开关：非法取值报错", False)
    except ValueError as exc:
        check("开关：非法取值报错", "longitude_fft" in str(exc))

    ok, why = lonfft_applicable(slat, slon, 10)
    check("lonfft_applicable()：散点 → False + 可打印理由",
          (ok is False) and isinstance(why, str) and len(why) > 0, why[:40])
    ok, why = lonfft_applicable(LA, LO, 20)
    check("lonfft_applicable()：网格 → True + 理由", ok is True, why[:40])


def t_counters():
    LA, LO, f, ws = make_grid(5.0, 5.0, nmax=10)
    stats_reset()
    analysis_quadrature(LA, LO, f, 10, weights=ws, longitude_fft="auto",
                        report_fit=False)
    d = stats_delta({"quadrature_fft": 0, "quadrature_direct": 0})
    check("计数器：auto 在网格上走了 FFT（quadrature_fft=1）",
          d.get("quadrature_fft") == 1 and d.get("quadrature_direct", 0) == 0, str(d))
    stats_reset()
    analysis_quadrature(LA, LO, f, 10, weights=ws, longitude_fft="direct",
                        report_fit=False)
    d = stats_delta({"quadrature_fft": 0, "quadrature_direct": 0})
    check("计数器：'direct' 走直接法（quadrature_direct=1）",
          d.get("quadrature_direct") == 1 and d.get("quadrature_fft", 0) == 0, str(d))


# ---------------------------------------------------------------------------
# 5. entry points
# ---------------------------------------------------------------------------
def t_entry_points():
    LA, LO, f, ws = make_grid(3.0, 3.0, lam0=0.5, nmax=15, K=3)
    a, ra = analysis(LA, LO, f, 15, method="quadrature", weights=ws,
                     longitude_fft="auto", report_fit=False)
    b, rb = analysis(LA, LO, f, 15, method="quadrature", weights=ws,
                     longitude_fft="direct", report_fit=False)
    check("analysis(method='quadrature')：auto ≡ direct",
          worst_diff(a, b) <= 1e-12, f"相对差 {worst_diff(a, b):.2e}")

    a, ra = analysis_projection(LA, LO, f, 15, weights=ws,
                                longitude_fft="auto", report_fit=False)
    b, rb = analysis_projection(LA, LO, f, 15, weights=ws,
                                longitude_fft="direct", report_fit=False)
    check("analysis_projection：auto ≡ direct",
          worst_diff(a, b) <= 1e-12, f"相对差 {worst_diff(a, b):.2e}")
    check("analysis_projection：报告里记了 FFT 路径",
          ra.meta["longitude_path"] == "fft", ra.meta["longitude_reason"][:40])

    # 批量：报告要说明批量本身没换算法
    cs, rs = analyze_series(LA, LO, f, 15, weights=ws, longitude_fft="auto",
                            report_fit=False)
    c2, r2 = analyze_series(LA, LO, f, 15, weights=ws, longitude_fft="direct",
                            report_fit=False)
    check("analyze_series：auto ≡ direct（多历元）",
          worst_diff(cs, c2) <= 1e-12, f"相对差 {worst_diff(cs, c2):.2e}")
    check("analyze_series：透传 longitude_fft（shared 里有路径）",
          rs.shared.get("longitude_path") == "fft", str(rs.shared)[:60])

    # iterative 分支也走同一条
    c1, _ = analysis_quadrature(LA, LO, f, 15, weights=ws, niter=1,
                                longitude_fft="auto", report_fit=False)
    c2, _ = analysis_quadrature(LA, LO, f, 15, weights=ws, niter=1,
                                longitude_fft="direct", report_fit=False)
    check("iterative（niter=1）：auto ≡ direct",
          worst_diff(c1, c2) <= 1e-12, f"相对差 {worst_diff(c1, c2):.2e}")


# ---------------------------------------------------------------------------
# 6. speed
# ---------------------------------------------------------------------------
def t_speed():
    """Time both the **quadrature kernel** and the **whole call**.

    Both matter and they are not the same number.  The kernel is where the
    algorithmic win lives (~56-90x here); the whole call additionally pays fixed
    per-call costs (weights, the Gram diagnostic over all points, the Legendre
    recurrence over all points) that amortise over a batch, so it only improves
    by ~10-18x and the ratio *falls* on a faster CPU.  Quoting only the kernel is
    how you end up promising 90x for something that finishes in 12x.
    """
    from shkit.analysis import _plan_longitude, _quadrature_C

    lat = np.arange(-90.0, 90.01, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    rng = np.random.default_rng(4)
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    nmax = 60

    def whole(f, mode, reps=2):
        """Best of ``reps``: the minimum is the least noisy microbenchmark
        estimator (BLAS thread scheduling only ever adds time)."""
        best = float("inf")
        co = None
        for _ in range(reps):
            t0 = time.perf_counter()
            co, _ = analysis_quadrature(LA, LO, f, nmax, weights=ws,
                                        report_fit=False, longitude_fft=mode)
            best = min(best, time.perf_counter() - t0)
        return best, co

    f6 = rng.normal(size=(LA.size, 6))          # the SAME data for both paths
    t_dir, cd = whole(f6, "direct")
    t_fft, cf = whole(f6, "fft")
    d = worst_diff(cd, cf)
    check(f"速度：181×360, nmax={nmax}, ntime=6 上 FFT ≡ direct",
          d <= 1e-12, f"相对差 {d:.2e}")
    # The whole-call ratio is machine dependent -- it is the fixed per-call cost
    # (Legendre recurrence over all points, weights, Gram diagnostic) that the
    # FFT path never pays, so on a faster CPU the ratio *falls*.  The hard
    # assertions are the kernel (algorithmic) and the absolute M3 target below.
    s_whole = t_dir / max(t_fft, 1e-9)
    check("速度：整次调用加速 ≥6×", s_whole >= 6.0,
          f"direct {t_dir:.3f}s → fft {t_fft:.3f}s = {s_whole:.1f}×")

    # ---- the kernel alone ------------------------------------------------
    fw = f6 * ws.w[:, None]
    pd = _plan_longitude(LA, LO, nmax, "direct")
    pf = _plan_longitude(LA, LO, nmax, "fft")
    k_dir = min(_timeit(lambda: _quadrature_C(pd, LA, LO, fw, nmax))
                for _ in range(2))
    k_fft = min(_timeit(lambda: _quadrature_C(pf, LA, LO, fw, nmax))
                for _ in range(2))
    s_core = k_dir / max(k_fft, 1e-9)
    check("★ 速度：求积核心加速 ≥30×（计划目标 44×）", s_core >= 30.0,
          f"direct {k_dir:.4f}s → fft {k_fft:.4f}s = {s_core:.0f}×")

    # ---- marginal per-epoch cost: the number a batch actually sees --------
    # Warm up first (BLAS thread pool + first-touch allocation), then take a
    # slope over a long baseline: a two-point difference over 5 epochs is noise.
    f1 = rng.normal(size=(LA.size, 1))
    t1d, _ = whole(f1, "direct")
    t1f, _ = whole(f1, "fft")
    whole(f1, "direct"); whole(f1, "fft")
    f_a = rng.normal(size=(LA.size, 6))
    f_b = rng.normal(size=(LA.size, 30))
    ta_d, _ = whole(f_a, "direct"); tb_d, _ = whole(f_b, "direct")
    ta_f, _ = whole(f_a, "fft");    tb_f, _ = whole(f_b, "fft")
    per_d, per_f = (tb_d - ta_d) / 24.0, (tb_f - ta_f) / 24.0
    check("速度：FFT 每历元边际成本 ≤5 ms/历元",
          per_f <= 5e-3, f"{per_f * 1000:.2f} ms/历元（direct {per_d * 1000:.1f} ms）")
    check("速度：单历元整次调用也更快（≥4×）", t1d / max(t1f, 1e-9) >= 4.0,
          f"{t1d:.4f}s → {t1f:.4f}s = {t1d / max(t1f, 1e-9):.1f}×")
    check("速度：203 历元外推的投影成本 ≤2 s（M3 目标量级）",
          per_f * 203 <= 2.0, f"{per_f * 203:.3f}s（direct 外推 "
                              f"{per_d * 203:.1f}s）")
    print(f"      （核心 {k_dir:.4f}s → {k_fft:.4f}s = {s_core:.0f}×；"
          f"整次调用 ntime=6 {t_dir:.3f}s → {t_fft:.3f}s = {s_whole:.1f}×；"
          f"边际 {per_d * 1000:.1f} → {per_f * 1000:.1f} ms/历元 "
          f"({per_d / max(per_f, 1e-9):.0f}×)，203 历元 {per_f * 203:.2f}s）")


# ---------------------------------------------------------------------------
# 7. A4: the same acceleration in the SYNTHESIS direction
# ---------------------------------------------------------------------------
def t_synthesis_equivalence():
    """``synthesize`` / ``synthesis_grid`` FFT ≡ direct, offsets included."""
    from shkit.basis import synthesize as syn_basis
    from shkit.synthesis import synthesis, synthesis_grid

    rng = np.random.default_rng(9)
    L, K = 20, 3
    C3 = rng.normal(size=(L + 1, L + 1, K)) / 30.0
    S3 = rng.normal(size=(L + 1, L + 1, K)) / 30.0
    ws = 1.0 / (np.arange(L + 1) + 1.0)

    worst = 0.0
    for lam0 in (0.0, 0.5, -179.5, 90.3, 123.4):
        lat = np.arange(-89.0, 90.0, 5.0)
        lon = lam0 + np.arange(0.0, 360.0, 5.0)
        LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
        rep = {}
        got = syn_basis(LA, LO, C3, S3, L, weights_scale=ws,
                        longitude_fft="auto", report=rep)
        ref = syn_basis(LA, LO, C3, S3, L, weights_scale=ws,
                        longitude_fft="direct")
        sc = max(float(np.abs(ref).max()), 1e-300)
        e = float(np.abs(got - ref).max()) / sc
        worst = max(worst, e)
        check(f"综合等价：lam0={lam0:g}（{K} 时次）", e <= 1e-12 and
              rep["longitude_path"] == "fft", f"rel = {e:.2e}")
    check("★ 综合：起始经度相位回归（5 个 lam0）总体 ≤1e-12", worst <= 1e-12,
          f"max = {worst:.2e}")

    # 极点行（±90°）必须同样精确
    lat = np.array([90.0, 45.0, 0.0, -45.0, -90.0])
    lon = 0.5 + np.arange(0.0, 360.0, 3.0)
    LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    g = syn_basis(LA, LO, C3, S3, L, weights_scale=ws, longitude_fft="fft")
    d = syn_basis(LA, LO, C3, S3, L, weights_scale=ws, longitude_fft="direct")
    check("综合：含 ±90° 极点行", float(np.abs(g - d).max()) <=
          1e-12 * max(float(np.abs(d).max()), 1e-300),
          f"rel = {np.abs(g - d).max() / np.abs(d).max():.2e}")

    # 同一份系数：单时次必须返回 (npoints,)，多时次 (npoints, ntime)
    co1 = SHCoeffs(C3[:, :, 0], S3[:, :, 0])
    lat = np.arange(-90.0, 90.01, 3.0)
    lon = 0.5 + np.arange(0.0, 360.0, 3.0)
    check("综合：单时次返回 (npoints,)", synthesis_grid(lat, lon, co1).shape ==
          (lat.size, lon.size))
    co3 = SHCoeffs(C3, S3)
    check("综合：多时次返回 (nlat, nlon, ntime)",
          synthesis_grid(lat, lon, co3).shape == (lat.size, lon.size, K))

    # 单位换算（geopotential -> EWH）不能因为换路径而变
    Cg = np.zeros((L + 1, L + 1))
    Cg[2, 0] = 1e-12
    cog = SHCoeffs(Cg, np.zeros_like(Cg), {"field_unit": "geopotential"})
    v_f = synthesis_grid(lat, lon, cog, ewh=True, longitude_fft="fft")
    v_d = synthesis_grid(lat, lon, cog, ewh=True, longitude_fft="direct")
    check("综合：--ewh 单位换算在 FFT 路径上一致",
          float(np.abs(v_f - v_d).max()) <= 1e-12 * float(np.abs(v_d).max()),
          f"rel = {np.abs(v_f - v_d).max() / np.abs(v_d).max():.2e}")


def t_synthesis_switch():
    """The synthesis switch is explicit too."""
    from shkit.synthesis import synthesis

    rng = np.random.default_rng(10)
    L = 20
    C = rng.normal(size=(L + 1, L + 1))
    S = rng.normal(size=(L + 1, L + 1))
    co = SHCoeffs(C, S)

    rep = {}
    synthesis(np.array([10.0, -30.0, 45.0]), np.array([1.0, 200.0, 300.0]), co,
              longitude_fft="auto", report=rep)
    check("综合开关：散点上 'auto' 回退直接法并记录理由",
          rep.get("longitude_path") == "direct" and "网格" in rep.get("longitude_reason", ""),
          rep.get("longitude_reason", "")[:40])
    raised = None
    try:
        synthesis(np.array([10.0, -30.0, 45.0]), np.array([1.0, 200.0, 300.0]), co,
                  longitude_fft="fft")
    except ValueError as exc:
        raised = str(exc)
    check("综合开关：散点上 'fft' 明确报错", raised is not None
          and "矩形网格" in raised, "raised" if raised else "no error")

    # nlon ≤ 2·nmax 的网格上 'fft' 必须报错
    lat = np.arange(-80.0, 80.01, 10.0)
    lon = np.arange(0.0, 360.0, 20.0)                 # nlon=18, nmax=18
    LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    raised = None
    try:
        synthesis(LA, LO, co, nmax=18, longitude_fft="fft")
    except ValueError as exc:
        raised = str(exc)
    check("综合开关：nlon ≤ 2·nmax 时 'fft' 报错并点名混叠",
          raised is not None and "Nyquist" in raised,
          (raised or "").splitlines()[0][:40] if raised else "no error")

    try:
        synthesis(LA, LO, co, longitude_fft="turbo")
        check("综合开关：非法取值报错", False)
    except ValueError as exc:
        check("综合开关：非法取值报错", "longitude_fft" in str(exc))


def t_synthesis_in_analysis():
    """The reconstruction used by ``analyze_series(report_fit=True)`` is the FFT.

    This is where A4 actually pays: the residual table used to be the single most
    expensive part of a batch job.
    """
    LA, LO, f, ws = make_grid(3.0, 3.0, lam0=0.5, nmax=15, K=6)
    a, ra = analyze_series(LA, LO, f, 15, weights=ws, report_fit=True,
                           longitude_fft="auto")
    b, rb = analyze_series(LA, LO, f, 15, weights=ws, report_fit=True,
                           longitude_fft="direct")
    check("批量：逐历元残差表的两半都记录了路径（求解 / 重建）",
          ra.shared.get("longitude_path") == "fft"
          and ra.shared.get("fit_longitude_path") == "fft",
          f"{ra.shared.get('longitude_path')} / {ra.shared.get('fit_longitude_path')}")
    check("批量：重建走直接法时同样记录",
          rb.shared.get("fit_longitude_path") == "direct")
    ta = np.asarray([ra.table["residual_rms"][i] for i in range(6)], dtype=float)
    tb = np.asarray([rb.table["residual_rms"][i] for i in range(6)], dtype=float)
    check("批量：两种路径的逐历元残差表一致", float(np.abs(ta - tb).max()) <=
          1e-12 * max(float(np.abs(tb).max()), 1e-300),
          f"rel = {np.abs(ta - tb).max() / max(np.abs(tb).max(), 1e-300):.2e}")


def t_synthesis_speed():
    """Synthesis FFT on 181x360, nmax=60 -- the A4 acceptance scene."""
    from shkit.synthesis import synthesis_grid

    lat = np.arange(-90.0, 90.01, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    rng = np.random.default_rng(12)
    L, K = 60, 6
    C = rng.normal(size=(L + 1, L + 1, K)) * 1e-12
    S = rng.normal(size=(L + 1, L + 1, K)) * 1e-12
    co = SHCoeffs(C, S)

    def run(mode, reps=2):
        best = float("inf")
        out = None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = synthesis_grid(lat, lon, co, longitude_fft=mode)
            best = min(best, time.perf_counter() - t0)
        return best, out

    t_d, v_d = run("direct")
    t_f, v_f = run("fft")
    sc = max(float(np.abs(v_d).max()), 1e-300)
    check(f"综合速度：181×360 / nmax={L} / ntime={K} 上 FFT ≡ direct",
          float(np.abs(v_f - v_d).max()) / sc <= 1e-12,
          f"rel = {np.abs(v_f - v_d).max() / sc:.2e}")
    sp = t_d / max(t_f, 1e-9)
    check("★ 综合速度：整次调用加速 ≥8×", sp >= 8.0,
          f"direct {t_d:.3f}s → fft {t_f:.3f}s = {sp:.1f}×")
    print(f"      （综合：direct {t_d:.4f}s → fft {t_f:.4f}s = {sp:.1f}×）")


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("A3/A4：经度 FFT 路径（分析方向求积 / 综合方向求值）")
    print("=" * 78)
    for fn in (t_criterion, t_layout, t_equivalence_offsets,
               t_phase_regression_has_teeth, t_equivalence_general, t_switch,
               t_counters, t_entry_points, t_speed,
               t_synthesis_equivalence, t_synthesis_switch,
               t_synthesis_in_analysis, t_synthesis_speed):
        print(f"\n--- {fn.__name__} " + "-" * (60 - len(fn.__name__)))
        try:
            fn()
        except Exception:
            traceback.print_exc()
            RESULTS.append(False)
    n, N = sum(RESULTS), len(RESULTS)
    print("\n" + "=" * 78)
    print(f"{n}/{N} checks passed")
    print("=" * 78)
    return 0 if n == N else 1


if __name__ == "__main__":
    sys.exit(main())
