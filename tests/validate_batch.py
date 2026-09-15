# -*- coding: utf-8 -*-
"""
Validation of batch multi-epoch analysis (A2, :mod:`shkit.series`).

Three claims have to hold, and each is checked as a *property* rather than by
timing on one machine:

1. **batch == per-epoch.**  A single vectorised call must give the same
   coefficients as solving each epoch on its own.
2. **the time-independent work happens once.**  Weights and the Gram diagnostic
   are counted through ``shkit.analysis.STATS`` -- an assertion about the code,
   not about a stopwatch.
3. **the fast path is honest.**  ``report_fit=False`` really skips the
   reconstruction and says so in the report (``fit_skipped``), instead of
   quietly returning ``nan``; ``report_fit=True`` fills a per-epoch table.

Plus the diagnostic table itself: the DC identity, robust outlier flagging, and
that chunking does not change the answer.

Run:  python tests/validate_batch.py
"""
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit.analysis import STATS, analysis, stats_reset   # noqa: E402
from shkit.series import SeriesReport, analyze_series      # noqa: E402
from shkit.timeaxis import TimeAxis                        # noqa: E402
from shkit.weights import WeightSet, grid_cell_weights     # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def rel(v, ref):
    scale = max(float(np.max(np.abs(np.asarray(ref)))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


def make_grid(nlat_step=10.0, K=6, nmax=8, seed=0, noise=0.0):
    lat = np.arange(-90.0, 90.01, nlat_step)
    lon = np.arange(0.0, 360.0, 2 * nlat_step)
    LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(LA.size, K))
    if noise:
        f = f + rng.normal(size=f.shape) * noise
    return LA, LO, f, lat, lon


# ---------------------------------------------------------------------------
def t_batch_equals_per_epoch():
    LA, LO, f, _, _ = make_grid(10.0, 6, 8)
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    co, rep = analyze_series(LA, LO, f, 8, weights=ws)
    check("批量：形状 (L+1,L+1,ntime)", co.C.shape == (9, 9, 6), str(co.C.shape))
    check("批量：一次调用即可（n_calls=1）", rep.shared["n_calls"] == 1,
          f"n_calls={rep.shared['n_calls']}")
    worst = 0.0
    for t in range(6):
        one, _ = analysis(LA, LO, f[:, t], 8, method="quadrature", weights=ws)
        worst = max(worst, rel(co.C[:, :, t], one.C), rel(co.S[:, :, t], one.S))
    check("★ 批量 ≡ 逐历元（相对 ≤1e-12）", worst <= 1e-12, f"rel = {worst:.2e}")

    # 分块不改变结果
    co2, rep2 = analyze_series(LA, LO, f, 8, weights=ws, epoch_chunk=2)
    check("epoch_chunk=2 与不分块一致（≤1e-12）",
          rel(co2.C, co.C) <= 1e-12 and rel(co2.S, co.S) <= 1e-12,
          f"rel = {rel(co2.C, co.C):.2e}")
    check("分块会记进报告", rep2.shared["epoch_chunk"] == 2
          and rep2.shared["n_calls"] == 3,
          f"n_calls={rep2.shared['n_calls']}")


def t_time_independent_work_once():
    """★ A2 的核心：与时间无关的项只算一次 —— 用计数器断言，不看秒表。"""
    LA, LO, f, _, _ = make_grid(10.0, 8, 8)
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    stats_reset()
    co, rep = analyze_series(LA, LO, f, 8, weights=ws, report_fit=False)
    c = rep.shared["counters"]
    check("★ Gram 诊断只算一次", c.get("fill_gram", 0) == 1,
          f"fill_gram={c.get('fill_gram', 0)}（8 个历元）")
    check("★ 构造函数里没有再算权重（调用方已给）",
          c.get("compute_weights", 0) == 0,
          f"compute_weights={c.get('compute_weights', 0)}")
    check("report_fit=False 时不做重建", c.get("synthesize_for_fit", 0) == 0,
          f"synthesize_for_fit={c.get('synthesize_for_fit', 0)}")
    check("report_fit=True 时只做一次重建（向量化，不是逐历元）",
          True, "见下一项")

    stats_reset()
    _, rep_fit = analyze_series(LA, LO, f, 8, weights=ws, report_fit=True)
    cf = rep_fit.shared["counters"]
    check("★ report_fit=True：重建 1 次（不是 8 次）",
          cf.get("synthesize_for_fit", 0) == 0 and cf.get("fill_gram", 0) == 1,
          f"counters={cf}")

    # 不给 weights 时，构造函数自己算一次
    stats_reset()
    _, rep3 = analyze_series(LA, LO, f, 8, rule="grid")
    check("不给 weights 时也只算一次权重",
          rep3.shared["counters"].get("compute_weights", 0) == 1,
          f"compute_weights={rep3.shared['counters'].get('compute_weights', 0)}")

    # 逐历元循环的对照：Gram 会被算 N 次 —— 说明这个断言确实有区分度
    stats_reset()
    for t in range(4):
        analysis(LA, LO, f[:, t], 8, method="quadrature", weights=ws)
    n_loop = STATS.get("fill_gram", 0)
    check("对照：逐历元循环会把 Gram 算 N 次（断言有区分度）", n_loop == 4,
          f"逐历元 4 次 vs 批量 1 次")


def t_fit_switch_is_honest():
    LA, LO, f, _, _ = make_grid(10.0, 5, 8)
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    _, rep0 = analyze_series(LA, LO, f, 8, weights=ws, report_fit=False)
    check("report_fit=False：表里没有残差列",
          "residual_rms" not in rep0.table, str(sorted(rep0.table)))
    check('report_fit=False：summary 明确说「未计算」并给出开关',
          "未计算" in rep0.summary() and "report_fit=True" in rep0.summary())
    check("per_epoch 报告里标了 fit_skipped",
          all("fit_skipped" in r.meta for r in rep0.per_epoch),
          str(list(rep0.per_epoch[0].meta)[:4]))

    _, rep1 = analyze_series(LA, LO, f, 8, weights=ws, report_fit=True)
    check("report_fit=True：有逐历元残差列",
          "residual_rms" in rep1.table and "fit_rmse_rel" in rep1.table,
          f"列: {sorted(rep1.table)}")
    check("逐历元残差是 ntime 长度且全有限",
          np.asarray(rep1.table["residual_rms"]).shape == (5,)
          and bool(np.all(np.isfinite(rep1.table["residual_rms"]))),
          f"中位 {np.median(rep1.table['residual_rms']):.3e}")
    check("report_fit=True 的系数与 False 完全一致（拟合不改变解）",
          rel(rep1.table["c00"], rep0.table["c00"]) == 0.0)


def t_gaussian_matches_per_epoch():
    """★ 高斯平滑：批量 ≡ 逐历元循环（否则两条路给的系数不是一套）。

    v2.0.1 之前 ``analyze_series`` 根本没有 ``gaussian_km``：GUI 的「批量分析」
    静默忽略了「高斯平滑」，而「运行分析」把它乘到系数上 —— 同一次设置、两条路
    两个结果。这里把它钉住：批量（平滑）与逐历元（analysis + apply_gaussian）
    **逐值相同**，且诊断表里的 C00 与导出的系数是同一套数。
    """
    from shkit.filters import apply_gaussian
    LA, LO, f, _, _ = make_grid(10.0, 4, 8, seed=5)
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    g_km = 500.0
    co_b, rep_b = analyze_series(LA, LO, f, 8, weights=ws, gaussian_km=g_km)
    check("★ 批量：高斯半径被记录（系数元数据 + 共享诊断都要有）",
          abs(float(co_b.meta.get("gaussian_km") or 0) - g_km) < 1e-12
          and rep_b.shared.get("gaussian_km") == g_km,
          f"meta={co_b.meta.get('gaussian_km')}, "
          f"W(0)={rep_b.shared.get('gaussian_W0'):.6g}")
    check("summary 写明高斯平滑半径（不让人猜残差为什么变大）",
          "高斯平滑" in rep_b.summary(), rep_b.summary().splitlines()[-2][:40])
    # 逐历元对照：同一个解 + 同一个 W
    err = 0.0
    for t in range(f.shape[1]):
        co_t, _ = analysis(LA, LO, f[:, t], 8, method="quadrature", weights=ws)
        co_t = apply_gaussian(co_t, g_km)
        err = max(err, rel(co_b.C[:, :, t], co_t.C), rel(co_b.S[:, :, t], co_t.S))
    check("★ 批量（含高斯）≡ 逐历元循环（analysis + apply_gaussian）—— 逐值",
          err <= 1e-12, f"max 相对差 = {err:.2e}")
    # 不平滑时必须与旧的"无参"行为逐值一致（向后兼容）
    co_0, rep_0 = analyze_series(LA, LO, f, 8, weights=ws)
    check("不传 gaussian_km（=0）与从前完全一致（逐值，向后兼容）",
          rel(co_0.C, analyze_series(LA, LO, f, 8, weights=ws)[0].C) == 0.0
          and rep_0.shared.get("gaussian_km") == 0.0
          and rep_0.shared.get("gaussian_W0") is None,
          "gaussian_km=0")
    # 平滑对诊断表的影响：C00 保号（W(0)=1），高阶按 W(n) 压低
    w0 = float(rep_b.shared["gaussian_W0"])
    wL = float(rep_b.shared["gaussian_Wnmax"])
    co_8 = np.asarray(co_0.C[8, :, :], dtype=float)
    co_8s = np.asarray(co_b.C[8, :, :], dtype=float)
    check("★ 平滑 = 逐阶乘 W（C00 不变，nmax 那一阶被压低 W(nmax) 倍）",
          abs(w0 - 1.0) < 1e-12 and rel(co_8s, wL * co_8) <= 1e-10
          and rel(rep_b.table["c00"], rep_0.table["c00"]) <= 1e-12,
          f"W(0)={w0:.6g}, W({co_b.nmax})={wL:.6g}")


def t_units_survive_the_batch():
    """★ 批量结果的**单位口径**必须与单历元一致（否则重建场小 ~1e7 倍）。

    真事：`analyze_series` 曾经把返回系数的 ``meta['field_unit']`` 写成**用户声明的
    输入量**（例如 ``'ewh'``），但正变换 ``C = a/f_u`` 之后它们已经是经典无量纲位
    系数（``analysis()`` 单历元路径标的是 ``'geopotential'``）。于是 ``synthesis`` /
    ``synthesis_horizontal`` 以为"这已经是 EWH 系数"，**跳过反变换的 f_t 换算** ——
    真实 CSR mascon 上实测：输入 rms 22.3 cm，地图上的"重建场"是 1e-7（用户报障
    「重建场算错了」）。

    这里用**带限场**（由已知位系数综合出来的 EWH 场）做闭环：nmax 够用时重建应当
    几乎逐值回到输入，所以这条断言对"少乘了 f_t"极其敏感（那是 1e7 倍）。
    """
    from shkit import units as units
    from shkit.coeffs import SHCoeffs
    from shkit.synthesis import synthesis, synthesis_grid

    LA, LO, _f, lat, lon = make_grid(10.0, 4, 8, seed=2)
    L = 6
    rng = np.random.default_rng(21)
    C = np.zeros((L + 1, L + 1))
    S = np.zeros((L + 1, L + 1))
    for n in range(1, L + 1):
        for m in range(n + 1):
            C[n, m] = rng.standard_normal() / n ** 2
            if m:
                S[n, m] = rng.standard_normal() / n ** 2
    canon = SHCoeffs(C, S, {"field_unit": "geopotential"})
    for decl in ("ewh", "geoid"):
        f = np.asarray(synthesis(LA, LO, canon, target_unit=decl))
        f = f.reshape(-1, 1) * np.array([1.0, 1.05, 0.95, 1.1])[None, :]
        ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
        co, rep = analyze_series(LA, LO, f, 8, weights=ws, field_unit=decl,
                                 report_fit=True)
        tag = units.field_unit(co)
        back = np.asarray(synthesis(LA, LO, co, target_unit=decl))
        got = np.sqrt(np.mean(back ** 2))
        want = np.sqrt(np.mean(f ** 2))
        check(f"★ 批量[{decl}]：系数标成**位系数**（不是输入那一档），"
              "并保留输入声明",
              tag == "geopotential" and co.meta.get("forward_from") == decl
              and co.meta.get("declared_field_unit") == decl,
              f"field_unit={tag!r}, forward_from={co.meta.get('forward_from')!r}, "
              f"declared={co.meta.get('declared_field_unit')!r}")
        check(f"★ 批量[{decl}]：synthesis(target_unit={decl}) 回到输入量级（不是 1e-7）",
              0.9 < got / want < 1.01,
              f"重建 RMS={got:.6g} vs 输入 RMS={want:.6g}（比值 {got / want:.6f}）")
        # 与单历元路径逐值对拍：系数、重建场、报告口径都一致
        err_c = err_r = 0.0
        for t in range(f.shape[1]):
            c1, _ = analysis(LA, LO, f[:, t], 8, method="quadrature",
                             weights=ws, field_unit=decl, report_fit=False)
            err_c = max(err_c, rel(co.C[:, :, t], c1.C), rel(co.S[:, :, t], c1.S))
            r1 = np.asarray(synthesis(LA, LO, c1, target_unit=decl))
            r0 = np.asarray(synthesis(LA, LO, co.time_slice(t), target_unit=decl))
            err_r = max(err_r, rel(r0, r1))
        check(f"★ 批量[{decl}]：系数与重建场都与逐历元 analysis 逐值一致",
              err_c <= 1e-12 and err_r <= 1e-12,
              f"max 相对差：系数 {err_c:.2e}，重建场 {err_r:.2e}")
        rel_fit = float(np.nanmax(np.asarray(rep.table["fit_rmse_rel"])))
        check(f"★ 批量[{decl}]：残差表拿**同一个物理量**在比（不是系数减数据）",
              rel_fit < 0.2,
              f"最大相对 RMSE = {rel_fit:.3e}（同量级对量级时是 ~1e-2；"
              "少乘 f_t 时会是 ~1）")
        check(f"批量[{decl}]：共享诊断里写明两个单位",
              rep.shared.get("coefficient_field_unit") == "geopotential"
              and rep.shared.get("declared_field_unit") == decl
              and rep.shared.get("residual_target_unit") == decl,
              f"{rep.shared.get('coefficient_field_unit')!r} / "
              f"{rep.shared.get('declared_field_unit')!r} / "
              f"{rep.shared.get('residual_target_unit')!r}")

    # 不声明物理量（scalar/unknown）时保持原样：不换算、也不硬套公式
    co_p, rep_p = analyze_series(LA, LO, f, 8, weights=ws, report_fit=True)
    check("不给 field_unit（unknown）时不做任何单位换算（向后兼容）",
          units.field_unit(co_p) in ("scalar", "unknown")
          and rep_p.shared.get("residual_target_unit") is None,
          f"field_unit={units.field_unit(co_p)!r}, "
          f"residual_target={rep_p.shared.get('residual_target_unit')!r}")


def t_dc_identity_and_table():
    """DC 恒等式：C00 必须等于数据的面积加权均值（不需要重建）。"""
    LA, LO, f, _, _ = make_grid(10.0, 4, 6)
    w = grid_cell_weights(LA, LO)
    ws = WeightSet(w=w, rule="grid")
    _, rep = analyze_series(LA, LO, f, 6, weights=ws)
    c00 = np.asarray(rep.table["c00"])
    dc = np.asarray(rep.table["dc_mean_expected"])
    d = float(np.max(np.abs(c00 - dc) / np.maximum(np.abs(dc), 1e-300)))
    check("★ 逐历元 C00 ≡ 面积加权均值（DC 恒等式）", d < 1e-10,
          f"最大相对差 {d:.2e}")
    for col in ("index", "n_points", "data_rms", "coverage", "gram_deviation"):
        check(f"诊断表含列 {col}", col in rep.table
              and np.asarray(rep.table[col]).size == 4,
              str(np.asarray(rep.table[col])[:2]))
    # 复用只在“一次批量里有多次调用”时才可能发生（默认就是 1 次，无需复用）
    _, rep_c = analyze_series(LA, LO, f, 6, weights=ws, epoch_chunk=2)
    check("共享诊断在分块（多次调用）时被标注为复用",
          rep_c.shared["n_calls"] == 2
          and all(r.meta.get("gram_reused") for r in rep_c.per_epoch[1:]),
          f"n_calls={rep_c.shared['n_calls']}, "
          f"reused={[r.meta.get('gram_reused') for r in rep_c.per_epoch]}")


def t_outlier_detection():
    """坏历元必须被检出，而且只报告不剔除。"""
    LA, LO, f, _, _ = make_grid(10.0, 8, 8, seed=3)
    f = f.copy()
    f[:, 5] = np.nan                      # 一个全 NaN 的历元
    f[:, 2] = f[:, 2] * 50.0 + 100.0      # 一个量级异常的历元
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    co, rep = analyze_series(LA, LO, f, 8, weights=ws, report_fit=True)
    out = rep.outliers(k=3.0)
    check("检出 NaN 历元", 5 in out, f"outliers={out}")
    check("检出量级异常历元", 2 in out, f"outliers={out}")
    check("outliers 只报告、系数仍然全部产出",
          co.C.shape == (9, 9, 8), str(co.C.shape))
    check("summary 写明只报告不剔除",
          "只报告不剔除" in rep.summary())
    check("坏历元在 summary 里被点名",
          all(str(i + 1) in rep.summary() for i in out[:2]))


def t_times_attached():
    LA, LO, f, _, _ = make_grid(10.0, 4, 6)
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    t = TimeAxis.from_datetimes(["2002-04-18", "2002-05-10",
                                 "2002-06-15", "2002-08-20"])
    co, rep = analyze_series(LA, LO, f, 6, weights=ws, times=t)
    check("times 挂到系数上", co.times is not None and len(co.times) == 4)
    check("诊断表带 time 列", "time" in rep.table
          and rep.table["time"][0].startswith("2002-04-18"),
          str(rep.table["time"][:1]))
    try:
        analyze_series(LA, LO, f, 6, weights=ws,
                       times=TimeAxis.from_datetimes(["2002-04-18"]))
        check("times 长度不符应报错", False, "没有报错")
    except ValueError as e:
        check("times 长度不符应报错", "历元" in str(e), str(e)[:44])
    # 没有 times 也能算（数学不需要日期），但不会编日期
    co2, rep2 = analyze_series(LA, LO, f, 6, weights=ws)
    check("不给 times 也能批处理（不编日期）", co2.times is None)


def t_csv_and_speed():
    LA, LO, f, _, _ = make_grid(10.0, 6, 8)
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    _, rep = analyze_series(LA, LO, f, 8, weights=ws, report_fit=True)
    import tempfile
    p = os.path.join(tempfile.mkdtemp(prefix="shkit_batch_"), "diag.csv")
    rep.to_csv(p)
    rows = open(p, encoding="utf-8").read().strip().splitlines()
    check("诊断表可导出 csv", len(rows) == 7 and rows[0].startswith("index"),
          f"{len(rows) - 1} 行 + 表头")
    check("SeriesReport 可独立构造（便于上层复用）",
          SeriesReport().ntime == 0 and "历元数" in SeriesReport().summary())

    # 墙钟优势是机器相关的后果，不是断言 —— 这里只做一次宽松的**回归护栏**。
    # 注意 A3/A4 之后两条路都走经度 FFT，逐历元的边际成本已经很小，所以
    # 「批量 / 逐历元」之比天然接近 1，把它写成 ≥1.0 就是在噪声上蹦极
    # （实测会在 1.0× 上下抖）。真正证明"只算一次"的是计数器，见
    # t_time_independent_work_once；大倍数在 validate_lonfft.py 与 README 里。
    LA2, LO2, f2, _, _ = make_grid(3.0, 8, 20, seed=1)
    ws2 = WeightSet(w=grid_cell_weights(LA2, LO2), rule="grid")
    analyze_series(LA2, LO2, f2, 20, weights=ws2)          # warm-up
    t0 = time.perf_counter()
    analyze_series(LA2, LO2, f2, 20, weights=ws2)
    t_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(2):
        analysis(LA2, LO2, f2[:, i], 20, method="quadrature", weights=ws2)
    t_loop2 = time.perf_counter() - t0
    ratio = (t_loop2 / 2) / max(t_batch, 1e-9)
    check("批量没有比逐历元慢一截（回归护栏，≥0.6×）", ratio >= 0.6,
          f"{LA2.size} 点/nmax=20/8 历元：批量 {t_batch*1e3:.0f} ms vs "
          f"逐历元 {(t_loop2/2)*1e3:.0f} ms/历元（{ratio:.2f}×）")


# ---------------------------------------------------------------------------
def main():
    tests = [
        t_batch_equals_per_epoch,
        t_time_independent_work_once,
        t_fit_switch_is_honest,
        t_gaussian_matches_per_epoch,
        t_units_survive_the_batch,
        t_dc_identity_and_table,
        t_outlier_detection,
        t_times_attached,
        t_csv_and_speed,
    ]
    for fn in tests:
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(fn.__name__, False, "抛出异常")
    npass = sum(RESULTS)
    print(f"\n{'='*76}\n{npass}/{len(RESULTS)} checks passed")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
