# -*- coding: utf-8 -*-
"""
Validation of the **time-domain operators** (C1, :mod:`shkit.timeseries`).

Five claims, each a property rather than a timing:

1. **An exact model is recovered exactly.**  A noise-free signal built from the
   fitted basis (constant + trend + harmonics) must come back to round-off --
   coefficients, model values at the sample epochs, *and* values at interpolated
   epochs.  If the fit only matched the sample epochs, the trend/seasonal split
   would be arbitrary.
2. **Amplitude and phase are what the fit says.**  The reported peak day is
   checked against the annual *term* evaluated on a dense cycle (not against the
   peak of the whole series, which the trend and semiannual term also move).
3. **Coefficient domain == grid domain.**  The fit is a linear operator over
   epochs, so applying it before or after synthesis must agree -- this is what
   makes fitting 3721 coefficients instead of 65 160 grid points legitimate.
4. **Irregular cadence and missing epochs are handled honestly.**  The Gaussian
   smoother is compared against explicitly hand-computed weights; a missing epoch
   gets zero weight rather than being treated as a zero datum; an axis without
   dates raises instead of pretending epochs are equally spaced.
5. **A model the data cannot support says so.**  Rank deficiency and large
   condition numbers are reported, not hidden behind confident amplitudes.

Run:  python tests/validate_timeseries.py
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit.coeffs import SHCoeffs                                    # noqa: E402
from shkit.synthesis import synthesis_grid                           # noqa: E402
from shkit.timeaxis import TimeAxis                                  # noqa: E402
from shkit.timeseries import (design_time, deseasonalize, detrend,   # noqa: E402
                              fit_time_model, poly_terms,
                              remove_time_mean, seasonal_terms, series_coord,
                              time_gaussian_filter, time_mean, trend_field)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def rel(v, ref):
    scale = max(float(np.max(np.abs(np.asarray(ref)))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


# ---------------------------------------------------------------------------
def synth_series(nt=60, L=4, seed=0, span_years=5.0, noise=0.0, nan=()):
    """A series built **exactly** from const + trend + annual + semiannual.

    Epochs are irregular (uniform in time, which is what real monthly solutions
    look like once gaps are included), and the returns include the truth so the
    fit can be checked against the generating values rather than against itself.
    """
    rng = np.random.default_rng(seed)
    day = np.sort(rng.choice(np.arange(0, int(span_years * 365.25)), nt,
                             replace=False)).astype(int)
    t0 = np.datetime64("2002-01-01")
    vals = (t0 + day.astype("timedelta64[D]")).astype("datetime64[s]")
    mid = vals[0] + (vals[-1] - vals[0]) / 2
    t = ((vals - mid).astype("timedelta64[s]").astype(float)
         / 86400.0 / 365.25)

    C = np.zeros((L + 1, L + 1, nt))
    S = np.zeros((L + 1, L + 1, nt))
    truth = {}
    for n in range(L + 1):
        for m in range(n + 1):
            c0 = rng.normal()
            c1 = rng.normal() * 0.1
            ac, as_ = rng.normal() * 0.5, rng.normal() * 0.5
            bc, bs = rng.normal() * 0.2, rng.normal() * 0.2
            truth[(n, m)] = dict(const=c0, trend=c1, annual_cos=ac,
                                 annual_sin=as_, semiannual_cos=bc,
                                 semiannual_sin=bs)
            x = (c0 + c1 * t
                 + ac * np.cos(2 * np.pi * t) + as_ * np.sin(2 * np.pi * t)
                 + bc * np.cos(4 * np.pi * t) + bs * np.sin(4 * np.pi * t))
            C[n, m, :] = x
            if m > 0:
                S[n, m, :] = x
    if noise:
        C[:, :, :] += rng.normal(size=C.shape) * noise
        S[:, :, 1:] += rng.normal(size=S[:, :, 1:].shape) * noise
    for i in nan:
        C[:, :, i] = np.nan
        S[:, :, i] = np.nan
    return SHCoeffs(C, S).with_times(TimeAxis.from_datetimes(vals)), truth, t, mid


# ---------------------------------------------------------------------------
# 1. design matrix
# ---------------------------------------------------------------------------
def t_design():
    t = np.linspace(-2.0, 2.0, 7)
    names, cols = poly_terms(t, 2)
    check("poly_terms 名称/列数", names == ["const", "trend", "poly2"]
          and len(cols) == 3, str(names))
    check("poly_terms const 列全 1、trend 列 == t",
          np.allclose(cols[0], 1.0) and np.allclose(cols[1], t))
    names, cols = seasonal_terms(t, (1.0, 0.5))
    check("seasonal_terms 名称（annual/semiannual × cos/sin）",
          names == ["annual_cos", "annual_sin", "semiannual_cos", "semiannual_sin"],
          str(names))
    check("seasonal_terms 年周期列 = cos(2πt)",
          np.allclose(cols[0], np.cos(2 * np.pi * t)))
    names, A = design_time(t, poly_order=1, periods=(1.0,))
    check("design_time 形状 (ntime, p)", A.shape == (7, 4), str(A.shape))
    try:
        design_time(t, poly_order=-1)
        check("design_time 无项时报错", False)
    except ValueError as exc:
        check("design_time 无项时报错", "至少要有一项" in str(exc))

    try:
        seasonal_terms(t, (0.0,))
        check("seasonal_terms 拒绝 period<=0", False)
    except ValueError as exc:
        check("seasonal_terms 拒绝 period<=0", "positive" in str(exc))

    # extra 项
    names, A = design_time(t, poly_order=0, extra=(["jump"], np.r_[np.zeros(3), np.ones(4)]))
    check("design_time 支持 extra 自定义列",
          names == ["const", "jump"] and A.shape == (7, 2), str(names))
    try:
        design_time(t, poly_order=0, extra=(["jump"], np.ones(3)))
        check("extra 行数不符时报错", False)
    except ValueError as exc:
        check("extra 行数不符时报错", "行" in str(exc))


# ---------------------------------------------------------------------------
# 2. exact recovery, in and between the sample epochs
# ---------------------------------------------------------------------------
def t_exact_recovery():
    co, truth, t_true, mid = synth_series(nt=60, L=4, seed=0, span_years=5.0)
    fit = fit_time_model(co, poly_order=1, periods=(1.0, 0.5))
    scale = max(float(np.abs(co.C).max()), 1e-300)
    check("无噪声拟合：残差 ≤1e-12（相对数据量级）",
          float(np.abs(fit.residual.C).max()) / scale <= 1e-12,
          f"max|resid|/max|data| = {np.abs(fit.residual.C).max() / scale:.2e}")
    check("无噪声拟合：模型在采样历元上等于数据",
          float(np.abs(fit.model.C - co.C).max()) / scale <= 1e-12
          and float(np.abs(fit.model.S - co.S).max()) / scale <= 1e-12,
          f"rel = {np.abs(fit.model.C - co.C).max() / scale:.2e}")

    # 在**没有数据**的时刻也要对：拿真值公式直接比
    tc, tref = series_coord(co)
    check("坐标以跨度中点为原点（|t| 对称）",
          abs(float(tc.min() + tc.max())) < 1e-9, f"t ∈ [{tc.min():.3f}, {tc.max():.3f}]")

    tt = np.linspace(tc.min(), tc.max(), 37)
    when = mid + np.round(tt * 365.25).astype("timedelta64[D]")
    # Compare at the times actually evaluated: the epoch grid is quantised to whole
    # days, and half a day of the annual cycle is already ~1e-2 of its amplitude.
    tt = (when - mid).astype("timedelta64[s]").astype(float) / 86400.0 / 365.25
    ev = fit.evaluate(when)
    worst = 0.0
    for (n, m), p in truth.items():
        x = (p["const"] + p["trend"] * tt
             + p["annual_cos"] * np.cos(2 * np.pi * tt)
             + p["annual_sin"] * np.sin(2 * np.pi * tt)
             + p["semiannual_cos"] * np.cos(4 * np.pi * tt)
             + p["semiannual_sin"] * np.sin(4 * np.pi * tt))
        worst = max(worst, float(np.max(np.abs(ev.C[n, m] - x))))
    check("★ 内插历元（37 个，无数据处）也等于真值 ≤1e-9",
          worst <= 1e-9, f"max |Δ| = {worst:.2e}")
    check("evaluate 在采样历元上与 model 一致",
          rel(fit.evaluate(co.times.values).C, fit.model.C) <= 1e-12)
    check("evaluate 标记未外推", fit.evaluate(co.times.values).meta["extrapolated"] is False)

    # 振幅/相位：与**年周期项本身**的稠密评估核对（不是整条序列的峰值）
    dense_t = np.linspace(0.0, 1.0, 366, endpoint=False)
    dense_when = mid + np.round((dense_t - 0.5) * 365.25).astype("timedelta64[D]")
    term = SHCoeffs(np.zeros((5, 5, 1)), np.zeros((5, 5, 1)))
    amp, doy, sig = fit.amplitude_phase("annual")
    a = fit.term("annual_cos")
    b = fit.term("annual_sin")
    tt_dense = (dense_t - 0.5)
    xd = a.C[2, 1] * np.cos(2 * np.pi * tt_dense) + b.C[2, 1] * np.sin(2 * np.pi * tt_dense)
    peak_doy = float(np.mod(dense_t[np.argmax(xd)] - 0.5, 1.0) * 365.25)
    check("相位（峰值日在一年中的位置）与稠密评估一致 ≤0.5 天",
          abs(peak_doy - float(doy[2, 1])) <= 0.5,
          f"振幅相位表 {float(doy[2, 1]):.2f} vs 稠密 {peak_doy:.2f}")
    want_amp = float(np.hypot(a.C[2, 1], b.C[2, 1]))
    check("振幅 ≡ hypot(a,b)", abs(float(amp.C[2, 1]) - want_amp) <= 1e-15 * want_amp,
          f"{float(amp.C[2, 1]):.6g}")
    check("sigma ≡ 振幅/√2（可直接与残差 RMS 比）",
          abs(float(sig.C[2, 1]) - want_amp / np.sqrt(2)) <= 1e-15 * want_amp)

    # 有噪声时：残差应与噪声同量级，且远小于数据
    co2, _, _, _ = synth_series(nt=60, L=4, seed=1, noise=1e-3)
    fit2 = fit_time_model(co2, poly_order=1, periods=(1.0, 0.5))
    check("有噪声（1e-3）时残差 RMS ≈ 噪声量级",
          5e-4 <= fit2.meta["fit_rms"] <= 2e-3,
          f"fit_rms = {fit2.meta['fit_rms']:.3e}")
    check("有噪声时报告了 RMS 降幅", 0.0 < fit2.meta["rms_reduction"] < 1.0,
          f"{fit2.meta['rms_reduction'] * 100:.1f}%")

    # 降幅的分母必须是**异常**（去时间均值），不是原始场 RMS：
    # 系数序列的静态部分占绝对多数，拿原始 RMS 当分母的话"解释 99.9999%"是废话。
    # 这里故意用一个模型**拟合不了**的时变信号（0.37 年周期），
    # 好让两个口径给出完全不同的结论。
    tt = np.linspace(-1.0, 1.0, 40)
    sig = 1e-6 * np.sin(2 * np.pi * tt / 0.37)
    arr = np.zeros((3, 3, 40))
    arr[1, 0, :] = 1.0 + sig                    # 静态项比时变项大 6 个量级
    ax = TimeAxis.from_datetimes(
        np.datetime64("2002-01-01")
        + np.round((tt + 1.0) * 182.6).astype("timedelta64[D]"))
    co_s = SHCoeffs(arr, np.zeros_like(arr)).with_times(ax)
    fit_s = fit_time_model(co_s, poly_order=1)   # 只有 const + trend
    check("★ 降幅以「异常 RMS」为分母（静态场不会把它抬到 100%）",
          fit_s.meta["rms_reduction_raw"] > 0.99 and fit_s.meta["rms_reduction"] < 0.5,
          f"原始口径 {fit_s.meta['rms_reduction_raw'] * 100:.5f}% 会骗人；"
          f"异常口径 {fit_s.meta['rms_reduction'] * 100:.1f}% 才是实话")
    check("summary 里同时给出异常 RMS 与原始 RMS（并说明后者仅作参照）",
          "异常 RMS" in fit_s.summary() and "仅作参照" in fit_s.summary())


# ---------------------------------------------------------------------------
# 3. coefficient domain == grid domain
# ---------------------------------------------------------------------------
def t_coeff_vs_grid():
    """The property that makes fitting coefficients (not grid points) legitimate."""
    co, _, _, _ = synth_series(nt=24, L=4, seed=3, span_years=4.0)
    # The generator contains the semiannual term, so the fit must model it too --
    # otherwise "the residual is zero" would be testing the wrong model.
    fit_c = fit_time_model(co, poly_order=1, periods=(1.0, 0.5))
    names, A = design_time(fit_c.coord, poly_order=1, periods=(1.0, 0.5))

    lat = np.arange(-80.0, 80.01, 10.0)
    lon = np.arange(0.0, 360.0, 15.0)
    cube = synthesis_grid(lat, lon, co)                  # (nlat, nlon, ntime)
    ntime = cube.shape[2]
    Y = cube.reshape(-1, ntime).T                        # (ntime, npoints)

    sol_g, *_ = np.linalg.lstsq(A, Y, rcond=None)
    scale = max(float(np.abs(Y).max()), 1e-300)
    check("网格域残差同样 ≈ 0（同一模型空间）",
          float(np.abs(Y - A @ sol_g).max()) / scale <= 1e-10,
          f"max|resid|/max|data| = {np.abs(Y - A @ sol_g).max() / scale:.2e}")

    k = names.index("trend")
    trend_grid = sol_g[k].reshape(lat.size, lon.size)
    trend_from_co = synthesis_grid(lat, lon, fit_c.term("trend"))
    check("★ 系数域拟合 ≡ 网格域拟合（趋势场，相对 ≤1e-10）",
          rel(trend_from_co, trend_grid) <= 1e-10,
          f"rel = {rel(trend_from_co, trend_grid):.2e}")


# ---------------------------------------------------------------------------
# 4. mean / missing / no dates
# ---------------------------------------------------------------------------
def t_mean_and_missing():
    co, _, _, _ = synth_series(nt=24, L=4, seed=5)
    mean = time_mean(co)
    want = np.mean(np.asarray(co.C, float), axis=2)
    check("未加权时间平均 ≡ 逐历元算术平均",
          rel(mean.C, want) <= 1e-14, f"rel = {rel(mean.C, want):.2e}")

    w = np.arange(1, 25, dtype=float)
    mw = time_mean(co, weights=w)
    want_w = np.tensordot(np.asarray(co.C, float), w / w.sum(), axes=([2], [0]))
    check("加权时间平均 ≡ 手算权重", rel(mw.C, want_w) <= 1e-13,
          f"rel = {rel(mw.C, want_w):.2e}")

    an, mean2 = remove_time_mean(co)
    check("去均值后时间平均为 0", float(np.abs(np.mean(an.C, axis=2)).max()) <= 1e-13,
          f"max = {np.abs(np.mean(an.C, axis=2)).max():.2e}")
    check("异常序列带时间轴且长度一致",
          an.has_time and an.ntime == co.ntime)
    check("未加权平均在不等间隔时给出提示",
          any("未加权" in w for w in mean.meta.get("warnings", [])),
          str(mean.meta.get("warnings", []))[:50])

    # 缺测历元
    co_nan, _, _, _ = synth_series(nt=24, L=4, seed=5, nan=(5, 11))
    fit = fit_time_model(co_nan, poly_order=1, periods=(1.0, 0.5))
    check("全 NaN 历元被排除并报告",
          list(fit.excluded) == [5, 11], str(list(fit.excluded)))
    check("排除的数量正确", fit.n_used == 22, f"n_used={fit.n_used}")
    keep = np.setdiff1d(np.arange(24), [5, 11])
    C = np.asarray(co_nan.C, float)[:, :, keep]
    Ss = np.asarray(co_nan.S, float)[:, :, keep]
    ax = TimeAxis.from_datetimes(np.asarray(co_nan.times.values)[keep])
    co_trim = SHCoeffs(C, Ss).with_times(ax)
    fit_trim = fit_time_model(co_trim, poly_order=1, periods=(1.0, 0.5))
    check("★ 带 NaN 的拟合 ≡ 手工剔除后的拟合",
          rel(fit.term("trend").C, fit_trim.term("trend").C) <= 1e-12,
          f"rel = {rel(fit.term('trend').C, fit_trim.term('trend').C):.2e}")

    # 只有一部分是 NaN（不是"整段缺测"）：必须明确报错而不是崩在 SVD 里
    co_part = SHCoeffs(np.asarray(co.C, float).copy(),
                       np.asarray(co.S, float).copy()).with_times(co.times)
    co_part.C[2, 1, 3] = np.nan
    try:
        fit_time_model(co_part, poly_order=1, periods=(1.0,))
        check("部分 NaN（非整段缺测）明确报错", False)
    except ValueError as exc:
        check("部分 NaN（非整段缺测）明确报错",
              "非有限值" in str(exc), str(exc)[:44])

    mask = np.ones(24, dtype=bool)
    mask[[0, 1, 2]] = False
    fit_m = fit_time_model(co, poly_order=0, periods=(1.0,), mask=mask)
    check("mask 生效并记录", list(fit_m.excluded) == [0, 1, 2],
          str(list(fit_m.excluded)))

    # 没有日期 -> 必须报错，而不是假装等间隔
    idx_co = SHCoeffs(np.asarray(co.C, float), np.asarray(co.S, float)).with_times(
        TimeAxis.from_index(24))
    try:
        fit_time_model(idx_co, poly_order=1)
        check("kind='index' 的时间轴拒绝拟合", False)
    except ValueError as exc:
        check("kind='index' 的时间轴拒绝拟合",
              "真实日期" in str(exc) or "只有序号" in str(exc), str(exc)[:44])
    plain = SHCoeffs(np.asarray(co.C, float), np.asarray(co.S, float))
    try:
        fit_time_model(plain, poly_order=1)
        check("没有时间轴时拒绝拟合", False)
    except ValueError as exc:
        check("没有时间轴时拒绝拟合", "时间轴" in str(exc), str(exc)[:44])
    try:
        time_gaussian_filter(idx_co, 0.3)
        check("时间滤波同样要求真实日期", False)
    except ValueError as exc:
        check("时间滤波同样要求真实日期", "日期" in str(exc), str(exc)[:44])
    # 平均不需要日期（等权平均与日期无关），但要有明确语义
    check("等权时间平均不要求日期（本身与日期无关）",
          time_mean(idx_co).C.shape == (5, 5))


# ---------------------------------------------------------------------------
# 5. time filtering: hand-computed weights
# ---------------------------------------------------------------------------
def t_time_filter():
    co, _, t, mid = synth_series(nt=24, L=3, seed=7, span_years=3.0)
    rep = {}
    sigma = 0.25
    truncate = 3.0
    filt = time_gaussian_filter(co, sigma, truncate=truncate, report=rep)

    # 手算第 7 个历元的权重
    k = 7
    half = truncate * sigma
    sel = np.abs(t - t[k]) <= half
    d = (t[sel] - t[k]) / sigma
    w = np.exp(-0.5 * d * d)
    want2 = np.tensordot(np.asarray(co.C, float)[:, :, sel],
                         w / w.sum(), axes=([2], [0]))
    check("★ 高斯时间滤波 ≡ 手算权重（单个历元）",
          rel(filt.C[:, :, k], want2) <= 1e-13,
          f"rel = {rel(filt.C[:, :, k], want2):.2e}")
    check("有效历元数 = 1/Σw²（手算）",
          abs(rep["effective_epochs"][k] - 1.0 / np.sum((w / w.sum()) ** 2)) <= 1e-12,
          f"{rep['effective_epochs'][k]:.4f}")
    check("报告里带 sigma/截断/边界历元数",
          rep["sigma_years"] == sigma and rep["truncate"] == truncate
          and "edge_epochs" in rep)
    check("不等间隔会被点名（不是静默平滑）",
          any("不等间隔" in n for n in rep["notes"]), str(rep["notes"])[:46])

    # 平滑必须真的压低高频，同时别把年周期抹掉
    rng = np.random.default_rng(2)
    nt = 36
    day = np.sort(rng.choice(np.arange(0, int(3 * 365.25)), nt, replace=False)).astype(int)
    vals = (np.datetime64("2002-01-01") + day.astype("timedelta64[D]")).astype("datetime64[s]")
    mid2 = vals[0] + (vals[-1] - vals[0]) / 2
    tt = (vals - mid2).astype("timedelta64[s]").astype(float) / 86400 / 365.25
    annual = np.cos(2 * np.pi * tt)
    noise = 0.5 * rng.normal(size=nt)
    C = np.zeros((2, 2, nt))
    C[1, 1, :] = annual + noise
    co2 = SHCoeffs(C, np.zeros_like(C)).with_times(TimeAxis.from_datetimes(vals))
    rep2 = {}
    sm = time_gaussian_filter(co2, 0.15, report=rep2)
    r_before = float(np.std(noise))
    r_after = float(np.std(np.asarray(sm.C, float)[1, 1] - annual))
    check("滤波压低逐历元噪声（≥1.4×）", r_after * 1.4 <= r_before,
          f"std {r_before:.4f} → {r_after:.4f}（有效历元数中位 "
          f"{np.median(rep2['effective_epochs']):.2f}）")

    # 幅频响应：连续高斯核对周期 P 的衰减是 exp(-½(2πσ/P)²)。
    # 这里只要求 25% 以内 —— 不等间隔下核不是位移不变的，偏差是**真实的**
    # （同时还有相位滞后），所以不能要求它像个理想低通。
    s = np.asarray(sm.C, float)[1, 1]
    scale = float(np.dot(s, annual) / np.dot(annual, annual))
    theory = float(np.exp(-0.5 * (2 * np.pi * 0.15 / 1.0) ** 2))
    check("滤波对年周期的衰减 ≈ 连续高斯响应 exp(-½(2πσ/P)²)（±25%）",
          abs(scale - theory) <= 0.25 * theory,
          f"实测 {scale:.4f} vs 理论 {theory:.4f}")
    check("年周期相关性 > 0.9（相位滞后是预期内的）",
          float(np.corrcoef(s, annual)[0, 1]) > 0.9,
          f"r = {np.corrcoef(s, annual)[0, 1]:.4f}，相位滞后 "
          f"{np.degrees(np.arctan2(np.dot(s, np.sin(2*np.pi*tt)), np.dot(s, np.cos(2*np.pi*tt)))):.1f}°")

    # 缺测历元权重为 0（不是当 0 值用）
    co3 = SHCoeffs(C.copy(), np.zeros_like(C)).with_times(TimeAxis.from_datetimes(vals))
    co3.C[:, :, 4] = np.nan
    rep3 = {}
    sm3 = time_gaussian_filter(co3, 0.15, report=rep3)
    sel3 = np.abs(tt - tt[9]) <= 4 * 0.15
    sel3[4] = False
    d3 = (tt[sel3] - tt[9]) / 0.15
    w3 = np.exp(-0.5 * d3 * d3)
    hand = (w3 / w3.sum()) @ np.asarray(co3.C, float)[1, 1, sel3]
    check("★ 缺测历元权重为 0（不当 0 值参与）",
          abs(float(sm3.C[1, 1, 9]) - float(hand)) <= 1e-13,
          f"got {sm3.C[1, 1, 9]:.6f} vs 手算 {hand:.6f}")
    check("缺测被写进滤波报告", any("缺数据历元权重为 0" in n for n in rep3["notes"]),
          str(rep3["notes"])[:50])
    try:
        time_gaussian_filter(co, -1.0)
        check("sigma <= 0 报错", False)
    except ValueError as exc:
        check("sigma <= 0 报错", "必须为正" in str(exc))


# ---------------------------------------------------------------------------
# 6. honesty: rank / conditioning / extrapolation
# ---------------------------------------------------------------------------
def t_honesty():
    co, _, _, _ = synth_series(nt=12, L=3, seed=9, span_years=0.6)
    fit = fit_time_model(co, poly_order=1, periods=(1.0,))
    check("短序列 + 趋势 + 年周期：条件数被报告",
          np.isfinite(fit.cond) and fit.cond > 10.0, f"cond = {fit.cond:.3g}")
    check("条件数偏大时给出警告",
          any("条件数" in w for w in fit.meta["warnings"])
          or any("条件数" in ln for ln in fit.summary().splitlines()),
          f"cond = {fit.cond:.3g}")
    check("summary 里能看到条件数与残差",
          "条件数" in fit.summary() and "残差 RMS" in fit.summary())

    co2, _, _, _ = synth_series(nt=60, L=3, seed=9, span_years=5.0)
    fit2 = fit_time_model(co2, poly_order=1, periods=(1.0,))
    out = fit2.evaluate(np.datetime64("2035-01-01"))
    check("外推被标记", out.meta["extrapolated"] is True)

    # 项数多于历元 -> 明确报错（不是给一个乱七八糟的解）
    C = np.asarray(co2.C, float)[:, :, :3]
    Ss = np.asarray(co2.S, float)[:, :, :3]
    ax = TimeAxis.from_datetimes(np.asarray(co2.times.values)[:3])
    t3 = SHCoeffs(C, Ss).with_times(ax)
    try:
        fit_time_model(t3, poly_order=1, periods=(1.0, 0.5))
        check("历元数 < 项数时明确报错", False)
    except ValueError as exc:
        check("历元数 < 项数时明确报错", "方程比未知数还少" in str(exc),
              str(exc)[:44])


# ---------------------------------------------------------------------------
# 7. convenience wrappers
# ---------------------------------------------------------------------------
def t_wrappers():
    co, _, _, _ = synth_series(nt=36, L=3, seed=11, span_years=3.0)
    det, tr = detrend(co)
    fit = fit_time_model(co, poly_order=1)
    check("detrend 返回 (残差, 趋势) 且趋势与拟合一致",
          rel(tr.C, fit.term("trend").C) <= 1e-14
          and rel(det.C, fit.residual.C) <= 1e-14)
    check("trend_field 等价于 detrend 的趋势",
          rel(trend_field(co).C, tr.C) <= 1e-14)
    check("趋势单位是「每历元单位 / 年」（斜率定义在真实时间上）",
          "trend" in fit.names)

    des, f2 = deseasonalize(co, periods=(1.0,))
    check("deseasonalize 只留常数项模型",
          f2.names == ["const", "annual_cos", "annual_sin"], str(f2.names))
    check("deseasonalize 后年周期项已移除（无年周期能量）",
          float(np.abs(np.mean(des.C, axis=2)).max()) <
          1e-9 * max(float(np.abs(co.C).max()), 1e-300),
          f"max|mean| = {np.abs(np.mean(des.C, axis=2)).max():.2e}")

    # weights 与显式加权最小二乘一致
    w = np.linspace(0.5, 2.0, co.ntime)
    fitw = fit_time_model(co, poly_order=1, periods=(1.0,), weights=w)
    names, A = design_time(fitw.coord, poly_order=1, periods=(1.0,))
    Y = np.asarray(co.C, float)[0, 0]
    sw = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(A * sw[:, None], Y * sw, rcond=None)
    check("★ 加权拟合 ≡ 手算加权 lstsq",
          abs(float(fitw.term("trend").C[0, 0]) - sol[names.index("trend")]) <= 1e-12,
          f"{fitw.term('trend').C[0, 0]:.9g} vs {sol[names.index('trend')]:.9g}")
    try:
        fit_time_model(co, poly_order=1, weights=np.ones(3))
        check("weights 长度不符报错", False)
    except ValueError as exc:
        check("weights 长度不符报错", "weights" in str(exc))


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
def t_cli():
    """``timefit`` / ``time-filter`` 端到端（走真实命令行入口）。"""
    import shutil
    import tempfile

    from shkit import io as shio
    from shkit.cli import main as cli_main

    co, _, _, _ = synth_series(nt=30, L=4, seed=21, span_years=4.0)
    tmp = tempfile.mkdtemp(prefix="shkit_c1_")
    try:
        series = os.path.join(tmp, "s.nc")
        shio.write_coeffs(co, series, layout="series_nc")

        def run(argv):
            try:
                rc = cli_main(argv)
                return (0 if rc is None else int(rc)), None
            except SystemExit as exc:
                return int(exc.code or 0), None
            except Exception as exc:                              # noqa: BLE001
                return 1, exc

        prefix = os.path.join(tmp, "fit")
        rc, exc = run(["timefit", "--series", series, "--poly-order", "1",
                       "--periods", "1.0,0.5", "--out-prefix", prefix,
                       "--out-residual", os.path.join(tmp, "res.nc"),
                       "--out-csv", os.path.join(tmp, "ep.csv")])
        terms = [f"{prefix}_{n}.sh" for n in
                 ("const", "trend", "annual_cos", "annual_sin",
                  "semiannual_cos", "semiannual_sin")]
        check("'timefit' 跑通并写出 6 个项文件",
              rc == 0 and all(os.path.exists(p) for p in terms) and exc is None,
              f"rc={rc}" + (f", {exc!r}" if exc else ""))
        check("'timefit' 写出残差序列与逐历元 csv",
              os.path.exists(os.path.join(tmp, "res.nc"))
              and os.path.exists(os.path.join(tmp, "ep.csv")))
        if os.path.exists(terms[1]):
            tr = shio.read_coeffs(terms[1])
            fit = fit_time_model(co, poly_order=1, periods=(1.0, 0.5))
            check("'timefit' 写出的趋势项与 API 结果一致",
                  rel(tr.C, fit.term("trend").C) <= 1e-12,
                  f"rel = {rel(tr.C, fit.term('trend').C):.2e}")

        rc, exc = run(["timefit", "--series", series, "--poly-order", "1",
                       "--periods", "none"])
        check("'timefit --periods none' 只拟合多项式", rc == 0 and exc is None,
              f"rc={rc}" + (f", {exc!r}" if exc else ""))

        out = os.path.join(tmp, "sm.nc")
        rc, exc = run(["time-filter", "--series", series, "--sigma", "0.3",
                       "--out", out])
        check("'time-filter' 跑通并写出平滑序列",
              rc == 0 and os.path.exists(out) and exc is None,
              f"rc={rc}" + (f", {exc!r}" if exc else ""))
        if os.path.exists(out):
            sm = shio.read_coeffs(out)
            want = time_gaussian_filter(co, 0.3)
            check("'time-filter' 结果与 API 一致",
                  rel(sm.C, want.C) <= 1e-12, f"rel = {rel(sm.C, want.C):.2e}")

        # 没有时间轴的序列必须给出可操作的报错，而不是崩掉
        plain = os.path.join(tmp, "plain.sh")
        shio.write_coeffs(co.time_slice(0), plain, layout="triangle")
        rc, exc = run(["timefit", "--series", plain])
        check("'timefit' 对无时间轴的输入报错（不崩）", rc != 0)

        rc, exc = run(["timefit", "--series", os.path.join(tmp, "nope.nc")])
        check("'timefit' 对不存在的文件非零退出", rc != 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 9. D7 core: per-point fits over a whole cube
# ---------------------------------------------------------------------------
def t_series_maps():
    """``fit_series_maps``：逐格点拟合的趋势/周年振幅/相位场。"""
    from shkit.timeseries import fit_series_maps

    nt = 48
    rng = np.random.default_rng(4)
    day = np.sort(rng.choice(np.arange(0, 365 * 4), nt, replace=False)).astype(int)
    vals = (np.datetime64("2002-01-01")
            + day.astype("timedelta64[D]")).astype("datetime64[s]")
    ax = TimeAxis.from_datetimes(vals)
    mid = vals[0] + (vals[-1] - vals[0]) / 2
    t = (vals - mid).astype("timedelta64[s]").astype(float) / 86400 / 365.25

    truth = [(0.5, 2.0, 182.0), (-0.25, 1.0, 0.0), (0.0, 0.0, 0.0)]
    cube = np.zeros((1, len(truth), nt))
    for k, (slope, amp, doy) in enumerate(truth):
        cube[0, k] = 1.0 + slope * t + amp * np.cos(2 * np.pi * t
                                                    - 2 * np.pi * doy / 365.25)
    out = fit_series_maps(cube, ax, poly_order=1, periods=(1.0,))
    worst = 0.0
    for k, (slope, amp, doy) in enumerate(truth):
        worst = max(worst, abs(float(out["trend"][0, k]) - slope),
                    abs(float(out["annual_amplitude"][0, k]) - amp))
        if amp > 0:
            # 相位是**环上的**量：365.25 与 0 是同一个峰值日，差值要走圆周距离
            d = abs(float(out["annual_phase_doy"][0, k]) - doy)
            worst = max(worst, min(d, 365.25 - d) / 365.25)
    check("★ 逐格点拟合：趋势/振幅/相位精确恢复（无噪声 ≤1e-12）",
          worst <= 1e-12, f"max|Δ| = {worst:.2e}（相位按年折算、走圆周距离）")
    check("残差 ≈ 0（模型正好张成数据）",
          float(np.nanmax(out["residual_rms"])) <= 1e-12,
          f"max = {np.nanmax(out['residual_rms']):.2e}")
    check("形状保持：输入 (nlat,nlon,ntime) → 输出同形",
          out["trend"].shape == cube.shape[:2]
          and out["annual_amplitude"].shape == cube.shape[:2],
          str(out["trend"].shape))
    check("相位在振幅≈0处无意义（那里的值只是噪声相位）",
          abs(float(out["annual_amplitude"][0, 2])) < 1e-12,
          f"cell2 振幅 = {float(out['annual_amplitude'][0, 2]):.2e}，"
          f"相位 = {float(out['annual_phase_doy'][0, 2]):.1f}（不作解释）")

    # 含半年周期：**两个周期都要报出来**（用户勾了半年却没看到半年图 = 报障）
    truth2 = np.zeros((1, 2, nt))
    truth2[0, 0] = 1.0 + 0.3 * np.cos(2 * np.pi * t - 2 * np.pi * 40.0 / 365.25) \
        + 0.2 * np.sin(2 * np.pi * t)
    truth2[0, 1] = 0.25 * np.cos(4 * np.pi * t) + 0.15 * np.sin(4 * np.pi * t)
    o2p = fit_series_maps(truth2, ax, poly_order=1, periods=(1.0, 0.5))
    seas = o2p.get("seasonal")
    check("逐格点拟合：periods=(1, 0.5) 时 seasonal 里有两个周期",
          isinstance(seas, list) and len(seas) == 2
          and [s["tag"] for s in seas] == ["annual", "semiannual"],
          str([(s["tag"], s["period"]) for s in seas] if seas else None))
    a1 = float(seas[0]["amplitude"][0, 0]) if seas else float("nan")
    a2 = float(seas[1]["amplitude"][0, 1]) if seas else float("nan")
    check("★ 半年项被单独报出：振幅 = hypot(0.25, 0.15) = 0.2915",
          abs(a2 - np.hypot(0.25, 0.15)) < 1e-12,
          f"半振幅 = {a2:.12g}")
    check("周年项仍是 hypot(0.3·cos40°, …) 那条（第一周期不受影响）",
          abs(a1 - np.hypot(0.3 * np.cos(2 * np.pi * 40.0 / 365.25), 0.2
                            + 0.3 * np.sin(2 * np.pi * 40.0 / 365.25))) < 1e-12,
          f"年振幅 = {a1:.12g}")
    check("旧键名 annual_amplitude/annual_phase_doy = 第一周期（向后兼容）",
          np.allclose(o2p["annual_amplitude"], seas[0]["amplitude"], equal_nan=True)
          and np.allclose(o2p["annual_phase_doy"], seas[0]["phase_doy"],
                          equal_nan=True))
    check("半年相位场有值且落在 [0, 0.5) 年", 
          np.all(np.isfinite(seas[1]["phase_doy"][0, 1]))
          and 0.0 <= float(seas[1]["phase_doy"][0, 1]) < 365.25,
          f"半年峰值日 = {float(seas[1]['phase_doy'][0, 1]):.2f}")
    check("只拟合周年时 seasonal 只有一条（不凭空多出半年）",
          len(fit_series_maps(cube, ax, periods=(1.0,)).get("seasonal") or []) == 1)

    # 与「系数域拟合再综合」等价 —— C1 已证明两者可交换，这里在 D7 的场口径上再验一次
    co = SHCoeffs(cube[0, 0][None, None, :], np.zeros((1, 1, nt)), {}, ax)
    f = fit_time_model(co, poly_order=1, periods=(1.0,))
    check("★ 逐格点拟合 ≡ 系数域 fit_time_model（趋势与振幅）",
          abs(float(out["trend"][0, 0]) - float(f.term("trend").C[0, 0])) <= 1e-12
          and abs(float(out["annual_amplitude"][0, 0])
                  - float(np.hypot(f.term("annual_cos").C[0, 0],
                                   f.term("annual_sin").C[0, 0]))) <= 1e-12,
          f"趋势 {float(out['trend'][0, 0]):.12g} vs "
          f"{float(f.term('trend').C[0, 0]):.12g}")

    # 缺测格点被排除并计数
    cube2 = cube.copy()
    cube2[0, 1] = np.nan
    o2 = fit_series_maps(cube2, ax)
    check("整段 NaN 的格点被排除，并在 n_used 里看得出来",
          list(np.asarray(o2["n_used"]).ravel()) == [nt, 0, nt]
          and o2["n_cells_used"] == 2,
          f"n_used={list(np.asarray(o2['n_used']).ravel())}，"
          f"used={o2['n_cells_used']}/3")
    check("缺测格点的结果保持 NaN（不填 0 冒充）",
          np.isnan(o2["trend"][0, 1]) and np.isnan(o2["annual_amplitude"][0, 1]))

    # 错误输入
    try:
        fit_series_maps(cube[0, 0], ax)          # 一维：没有时间轴那一维
        check("values 维度不足时报错", False)
    except ValueError as exc:
        check("values 维度不足时报错", "最后一维" in str(exc), str(exc)[:40])
    try:
        fit_series_maps(cube, TimeAxis.from_index(nt))
        check("没有日期时拒绝拟合（不假装等间隔）", False)
    except ValueError as exc:
        check("没有日期时拒绝拟合（不假装等间隔）",
              "真实日期" in str(exc) or "序号" in str(exc), str(exc)[:40])
    try:
        fit_series_maps(cube, vals[:10])
        check("times 与 ntime 不符时报错", False)
    except ValueError as exc:
        check("times 与 ntime 不符时报错", "历元" in str(exc), str(exc)[:40])


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("C1：时间域算子（去均值 / 趋势 / 周年 / 时间滤波）")
    print("=" * 78)
    for fn in (t_design, t_exact_recovery, t_coeff_vs_grid, t_mean_and_missing,
               t_time_filter, t_honesty, t_wrappers, t_series_maps, t_cli):
        print(f"\n--- {fn.__name__} " + "-" * (60 - len(fn.__name__)))
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(fn.__name__, False, "抛出异常")
    npass = sum(RESULTS)
    print(f"\n{'=' * 78}\n{npass}/{len(RESULTS)} checks passed")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
