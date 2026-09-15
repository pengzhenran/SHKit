# -*- coding: utf-8 -*-
"""
Validation of the *package* horizontal-deformation stack (B1).

``test_horizontal_golden.py`` pins the frozen cross-package contract.  This suite
covers what only the package can be asked about:

* the gradient as an operator (finite differences against ``shkit.synthesis``);
* the units layer (``horizontal_displacement`` as a vector quantity, per-degree
  factor from ``l'``, irreversible conversions refused);
* ``synthesis_horizontal`` end to end, including multi-epoch and chunking;
* the pole behaviour actually implemented (not just the reference's);
* **and one correction**: Gaussian smoothing does **not** commute with the
  horizontal gradient on the sphere (measured here), so only one order is
  offered and it is the one applied to the coefficients.

Run:  python tests/validate_horizontal.py
"""
import inspect
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit import units                                        # noqa: E402
from shkit.analysis import analysis_quadrature                 # noqa: E402
from shkit.coeffs import SHCoeffs                              # noqa: E402
from shkit.filters import (EARTH_RADIUS_M, gaussian_coefficients,  # noqa: E402
                           load_lln)
from shkit.gradient import (degree_factors_horizontal, horizontal_field,  # noqa: E402
                            sph_gradient, synthesis_horizontal)
from shkit.synthesis import synthesis                          # noqa: E402
from shkit.weights import WeightSet, grid_cell_weights          # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def rel(v, ref):
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


def rand_coeffs(L=20, seed=7, scale=1.0):
    rng = np.random.default_rng(seed)
    C = np.zeros((L + 1, L + 1))
    S = np.zeros((L + 1, L + 1))
    for n in range(L + 1):
        for m in range(n + 1):
            C[n, m] = rng.standard_normal() * scale / (n + 1) ** 2
            if m:
                S[n, m] = rng.standard_normal() * scale / (n + 1) ** 2
    S[:, 0] = 0.0
    return C, S


LAT_PTS = np.array([0.0, 12.0, 37.0, -44.0, 61.0, -8.0, 89.999, -89.999])
LON_PTS = np.array([0.0, 40.0, 130.0, 250.0, 300.0, 80.0, 15.0, 200.0])
C20, S20 = rand_coeffs(20, seed=7)
FH20 = degree_factors_horizontal(20)


# ---------------------------------------------------------------------------
def t_gradient_iron_law():
    """∂/∂θ 与 (1/sinθ)∂/∂λ ≡ 标量场的中心差分（用包内自己的 synthesis）。"""
    co = SHCoeffs(C20, S20)
    lat_ok = np.abs(np.abs(LAT_PTS) - 90.0) > 1e-9
    away = lat_ok & (np.abs(LAT_PTS) <= 89.0)
    near = lat_ok & (np.abs(LAT_PTS) > 89.0)
    h = 1e-4
    dth, dlm = sph_gradient(LAT_PTS, LON_PTS, C20, S20)

    def f(la, lo):
        return np.asarray(synthesis(la, lo, co)).ravel()

    for mask, tag, tol in ((away, "中低纬", 1e-9), (near, "近极点", 1e-5)):
        if not mask.any():
            continue
        la, lo = LAT_PTS[mask], LON_PTS[mask]
        d_dlat = -(f(la + h, lo) - f(la - h, lo)) / (2 * np.deg2rad(h))
        d_dlon = (f(la, lo + h) - f(la, lo - h)) / (2 * np.deg2rad(h))
        scale = max(float(np.max(np.abs(d_dlat))), 1e-300)
        r1 = float(np.max(np.abs(dth[mask] - d_dlat))) / scale
        r2 = float(np.max(np.abs(dlm[mask] - d_dlon / np.cos(np.deg2rad(la))))) / \
            max(float(np.max(np.abs(d_dlon / np.cos(np.deg2rad(la))))), 1e-300)
        check(f"sph_gradient ∂/∂θ ≡ 差分（{tag}）", r1 < tol, f"rel = {r1:.2e}")
        check(f"sph_gradient (1/sinθ)∂/∂λ ≡ 差分（{tag}）", r2 < tol, f"rel = {r2:.2e}")


def t_zero_degree_annihilated():
    """只改 C00 时水平场逐位不变（0 阶被抹掉 = F^h_0 = 0）。"""
    C, S = rand_coeffs(12, seed=3)
    a = horizontal_field(LAT_PTS, LON_PTS, C, S, degree_factors_horizontal(12))
    C2 = C.copy()
    C2[0, 0] += 1.0e3
    b = horizontal_field(LAT_PTS, LON_PTS, C2, S, degree_factors_horizontal(12))
    d = max(float(np.max(np.abs(a["north"] - b["north"]))),
            float(np.max(np.abs(a["east"] - b["east"]))))
    check("只改 C00 → 水平场逐位不变", d == 0.0, f"max|diff| = {d:.1e}")
    check("F^h_0 = 0（l′_0 = 0）", degree_factors_horizontal(4)[0] == 0.0)


def t_degree_factor_uses_l():
    """逐阶因子来自 l′（不是 h′、不是 EWH 的 Aₙ）。"""
    lln = load_lln()
    l, h, k = (np.asarray(lln[x], float) for x in "lhk")
    fh = units.degree_factors("horizontal_displacement", 20)
    want = EARTH_RADIUS_M * l[1:21] / (1.0 + k[1:21])
    check("units 因子 = R·l′ₙ/(1+k′ₙ)（手算）",
          float(np.max(np.abs(fh[1:21] - want))) == 0.0,
          f"F^h_1 = {fh[1]:.6e}")
    fr = units.degree_factors("radial_displacement", 20)
    fe = units.degree_factors("ewh", 20)
    check("水平与径向**不同**（l′ ≠ h′）", rel(fh[1:], fr[1:]) > 0.5,
          f"|l′/h′| ≈ {abs(l[6] / h[6]):.4f} (n=6)")
    # 混用 EWH 的 Aₙ 会差 2~4 个数量级（逐阶不同，所以不能"乘一个常数补救"）
    ratio_ae = fe[1:21] / fh[1:21]
    check("EWH 因子 Aₙ 与 F^hₙ 差 10²~10⁴（混用即错几个数量级）",
          10.0 < float(np.min(ratio_ae)) and float(np.max(ratio_ae)) < 1e5,
          f"Aₙ/F^hₙ = {np.min(ratio_ae):.1f} .. {np.max(ratio_ae):.1f}（逐阶变化）")
    check("l′ 非单调（2 阶凹陷）", l[2] < l[3], f"l′₂={l[2]:.5f} < l′₃={l[3]:.5f}")


def t_amplitude_anchor_with_radial():
    """量级锚：GRACE 量级的系数下，水平形变与径向形变**同量级**（比值 1e-2~1）。

    这条是最省事的自检：**谁把水平乘上了 Aₙ 或 (2n+1)/3，比值立刻偏 3~6 个数量级。**
    （水平梯度带来一个 ~n 的因子、l′/h′ ≈ −0.02~−0.07，所以比值落在百分之几到几十。）
    """
    Cg, Sg = rand_coeffs(20, seed=21, scale=1e-10)      # GRACE 位系数量级
    co = SHCoeffs(Cg, Sg).with_unit("geopotential")
    ur = np.asarray(synthesis(LAT_PTS, LON_PTS, co,
                              target_unit="radial_displacement")).ravel()
    uh = synthesis_horizontal(LAT_PTS, LON_PTS, co, want=("north", "east"))
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2)))
    r_h = max(rms(uh["north"]), rms(uh["east"]))
    r_r = rms(ur)
    ratio = r_h / r_r
    check("|u_h|/|u_r| ∈ [1e-2, 1]（GRACE 量级下同量级）",
          1e-2 < ratio < 1.0, f"rms 比 = {ratio:.4f}  (u_h={r_h:.3e} m, u_r={r_r:.3e} m)")
    check("水平形变量级为 mm 级（|C|~1e-10）", 1e-6 < r_h < 1e-1,
          f"rms(u_h) = {r_h * 1e3:.4f} mm")


def t_units_registry():
    """单位注册表：矢量量、必需 l′、标签与公式。"""
    check("'horizontal_displacement' 在 FIELD_UNITS",
          "horizontal_displacement" in units.FIELD_UNITS)
    check("UNITS_NEEDING_LOVE_L 只含水平形变",
          units.UNITS_NEEDING_LOVE_L == ("horizontal_displacement",),
          str(units.UNITS_NEEDING_LOVE_L))
    check("标签写着「矢量」", "矢量" in units.FIELD_UNIT_LABELS["horizontal_displacement"])
    check("formula_text 说明不是逐阶乘法",
          "不是逐阶乘法" in units.formula_text("horizontal_displacement"))


def t_reversal_refused():
    """F11：水平形变不能当输入、也不能被换算 —— 报错要点明 C00 / 矢量。"""
    co = SHCoeffs(*rand_coeffs(8, seed=1))
    try:
        units.convert(co, "horizontal_displacement")
        check("convert → horizontal 被拒绝", False, "没有报错")
    except ValueError as e:
        check("convert → horizontal 被拒绝（矢量装不进 SHCoeffs）",
              "矢量" in str(e), str(e).splitlines()[0][:56])
    coh = co.with_unit("horizontal_displacement")
    try:
        units.convert(coh, "ewh")
        check("horizontal → ewh 被拒绝", False, "没有报错")
    except ValueError as e:
        check("horizontal → ewh 被拒绝（含 C00 说明）",
              "C00" in str(e) and "矢量" in str(e), str(e).splitlines()[0][:56])
    try:
        synthesis_horizontal(LAT_PTS, LON_PTS, coh)
        check("synthesis_horizontal(水平形变系数) 被拒绝", False, "没有报错")
    except ValueError as e:
        check("synthesis_horizontal(水平形变系数) 被拒绝",
              "C00" in str(e), str(e).splitlines()[0][:56])


def t_end_to_end_from_ewh():
    """synthesis_horizontal 从「EWH 声明」的系数出发 == 手工链条。"""
    C, S = rand_coeffs(14, seed=11)
    A = units.degree_factors("ewh", 14)
    co = SHCoeffs(C * A[:, None], S * A[:, None]).with_unit("ewh")
    got = synthesis_horizontal(LAT_PTS, LON_PTS, co, want=("north", "east"))
    want = horizontal_field(LAT_PTS, LON_PTS, C, S, degree_factors_horizontal(14))
    check("EWH 声明 → 水平形变 == 手工链条 (u_N)",
          rel(got["north"], want["north"]) <= 1e-12,
          f"rel = {rel(got['north'], want['north']):.2e}")
    check("EWH 声明 → 水平形变 == 手工链条 (u_E)",
          rel(got["east"], want["east"]) <= 1e-12,
          f"rel = {rel(got['east'], want['east']):.2e}")
    # 未声明 → 默认**拒绝**（与 synthesis(target_unit=...) 一致）；强行才放行
    co_s = SHCoeffs(C, S)
    try:
        synthesis_horizontal(LAT_PTS, LON_PTS, co_s, want=("north",))
        check("未声明系数默认被拒绝", False, "没有报错")
    except ValueError as e:
        check("未声明系数默认被拒绝（提示 .with_unit）",
              "geopotential" in str(e), str(e).splitlines()[0][:56])
    got_s = synthesis_horizontal(LAT_PTS, LON_PTS, co_s, want=("north",),
                                 allow_unit_mismatch=True)
    check("allow_unit_mismatch=True 时不做换算是恒等",
          rel(got_s["north"], want["north"]) <= 1e-12,
          f"rel = {rel(got_s['north'], want['north']):.2e}")


def t_derived_components():
    """magnitude / azimuth 与 want 过滤。"""
    co = SHCoeffs(C20, S20).with_unit("geopotential")
    out = synthesis_horizontal(LAT_PTS, LON_PTS, co)
    check("默认返回四个分量",
          set(out) == {"north", "east", "magnitude", "azimuth"}, str(sorted(out)))
    check("magnitude = hypot(north, east)",
          rel(out["magnitude"], np.hypot(out["north"], out["east"])) < 1e-15)
    check("azimuth ∈ [0,360)", bool(np.all(out["azimuth"] >= 0) and np.all(out["azimuth"] < 360)),
          f"{out['azimuth'].min():.3f} .. {out['azimuth'].max():.3f}")
    check("want= 可裁剪输出",
          set(synthesis_horizontal(LAT_PTS, LON_PTS, co, want=("east",))) == {"east"})


def t_poles_in_package():
    """包内极点：Q_11 = √3；纯 C_11 在极点 u_E = F^h_1·(−√3)（非 0、非 NaN）。"""
    from shkit.basis import q_at_pole_m1
    for t_pole, tag in ((1.0, "北极"), (-1.0, "南极")):
        Q = q_at_pole_m1(20, t_pole)
        check(f"Q_11({tag}) = √3", abs(Q[1] - np.sqrt(3.0)) < 1e-15, f"{Q[1]:.15f}")
    C = np.zeros((3, 3)); S = np.zeros((3, 3)); C[1, 1] = 1.0
    out = horizontal_field(np.array([90.0]), np.array([90.0]), C, S,
                           degree_factors_horizontal(2))
    want = degree_factors_horizontal(2)[1] * (-np.sqrt(3.0))
    check("极点 u_E = F^h_1·(−√3)（非 0）",
          abs(float(out["east"][0]) - want) / abs(want) < 1e-14 and abs(want) > 1.0,
          f"u_E = {float(out['east'][0]):.6f} m")
    check("极点 u_N = 0", abs(float(out["north"][0])) / abs(want) < 1e-15)
    check("极点 magnitude 有限", bool(np.isfinite(out["magnitude"][0])))


def t_multitime_and_chunking():
    """3D 系数一次算 ≡ 逐历元；chunk 只切点、不改结果。"""
    C, S = rand_coeffs(12, seed=5)
    C3 = np.stack([C, C * 2.0, C * 0.5], axis=2)
    S3 = np.stack([S, S * 2.0, S * 0.5], axis=2)
    fh = degree_factors_horizontal(12)
    got = horizontal_field(LAT_PTS, LON_PTS, C3, S3, fh)
    check("3D 系数返回 (npoints, ntime)",
          got["north"].shape == (LAT_PTS.size, 3), str(got["north"].shape))
    worst = 0.0
    for t in range(3):
        one = horizontal_field(LAT_PTS, LON_PTS, C3[:, :, t], S3[:, :, t], fh)
        worst = max(worst, rel(got["north"][:, t], one["north"]),
                    rel(got["east"][:, t], one["east"]))
    check("3D 一次算 ≡ 逐历元（≤1e-12）", worst <= 1e-12, f"rel = {worst:.2e}")
    big = np.arange(-88.0, 88.01, 2.0)
    lon = np.full(big.size, 33.0)
    base = horizontal_field(big, lon, C, S, fh)
    wc = 0.0
    for ch in (1, 7, 13):
        got_c = horizontal_field(big, lon, C, S, fh, chunk=ch)
        wc = max(wc, rel(got_c["north"], base["north"]), rel(got_c["east"], base["east"]))
    check("chunk 不改变结果（≤1e-15）", wc <= 1e-15, f"rel = {wc:.2e}")


def t_smoothing_does_not_commute():
    """⚠️ 更正：球面上高斯平滑与水平梯度**不可交换**。

    ``∇_H`` 把阶 ``ℓ`` 混到 ``ℓ±1``，而空间高斯是**逐阶**乘子 ``W_ℓ``，
    所以"先乘 ``W`` 再求梯度" ≠ "先求梯度再平滑"。实测两者可差 ~24%。
    （圆环上 ``d/dθ`` 在 Fourier 基下是对角的，才会交换 —— 直觉在这里会骗人。）

    因此 v2.0 **只提供一种顺序**：平滑施加在**系数**上（与径向共用 ``gaussian_km``，
    只施加一次），水平形变是那个已平滑标量场的梯度。
    """
    L = 20
    C, S = rand_coeffs(L, seed=7)
    W = gaussian_coefficients(800.0, L)
    fh = degree_factors_horizontal(L)
    la, lo = np.array([30.0]), np.array([40.0])

    a = horizontal_field(la, lo, C * W[:, None], S * W[:, None], fh)   # 先平滑再求梯度

    lat = np.arange(-89.5, 89.51, 2.0)
    lon = np.arange(0.0, 360.0, 2.0)
    LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    b = {}
    for comp in ("north", "east"):
        g = horizontal_field(LA, LO, C, S, fh)[comp]
        co, _ = analysis_quadrature(LA, LO, g, L, weights=ws)
        b[comp] = np.asarray(synthesis(la, lo, SHCoeffs(co.C * W[:, None],
                                                        co.S * W[:, None]))).ravel()
    ratios = {c: float(b[c][0] / a[c][0]) for c in ("north", "east")}
    check("两种顺序**不**等价（|ratio−1| > 5%）",
          all(abs(r - 1.0) > 0.05 for r in ratios.values()),
          "  ".join(f"{c}: {r:.4f}" for c, r in ratios.items()))
    check("synthesis_horizontal 不接受 gaussian_km（平滑只能施加在系数上）",
          "gaussian_km" not in inspect.signature(synthesis_horizontal).parameters,
          str(sorted(inspect.signature(synthesis_horizontal).parameters)))


def t_scattered_and_regional():
    """散点与区域子集都能算，结果有限。"""
    rng = np.random.default_rng(2)
    la = rng.uniform(-85, 85, 400)
    lo = rng.uniform(0, 360, 400)
    out = horizontal_field(la, lo, C20, S20, FH20)
    check("随机散点：有限",
          bool(np.all(np.isfinite(out["north"])) and np.all(np.isfinite(out["east"]))),
          f"|u_N|max = {np.max(np.abs(out['north'])):.3e} m")
    reg = (la > 20) & (la < 50) & (lo > 70) & (lo < 140)
    check("区域子集（覆盖不足）不崩、结果有限",
          reg.sum() > 0 and bool(np.all(np.isfinite(horizontal_field(
              la[reg], lo[reg], C20, S20, FH20)["north"]))),
          f"{int(reg.sum())} 点")


def t_progress_and_cancel():
    """进度回呼 + 协作式取消（界面把这一步挪出界面线程后全靠这两条）。"""
    from shkit.gradient import (horizontal_field, horizontal_grid,
                                synthesis_horizontal)

    ticks = []
    a = horizontal_field(LAT_PTS, LON_PTS, C20, S20, FH20,
                         progress=lambda m, f: ticks.append((m, f)))
    check("散点路径：progress 被调用且比例单调不减、落在 [0,1]",
          len(ticks) >= 2
          and all(0.0 <= f <= 1.0 for _m, f in ticks)
          and all(ticks[i][1] <= ticks[i + 1][1] for i in range(len(ticks) - 1)),
          f"{len(ticks)} 次，最后 {ticks[-1][1]:.3f}")
    check("  进度文字里带得出阶数与偏移（不是空字符串）",
          all("阶 m=" in m for m, _f in ticks), f"例：{ticks[0][0]!r}")

    # 网格 FFT 路径也要报（它按 time_chunk × 阶 分块）
    ticks_g = []
    horizontal_grid(np.arange(-90.0, 90.01, 5.0), np.arange(0.0, 360.0, 5.0),
                    C20, S20, FH20, method="fft",
                    progress=lambda m, f: ticks_g.append(f))
    check("网格 FFT 路径：progress 也被调用", len(ticks_g) >= 2,
          f"{len(ticks_g)} 次")

    for who, fn in (("散点", lambda **kw: horizontal_field(
                        LAT_PTS, LON_PTS, C20, S20, FH20, **kw)),
                    ("网格", lambda **kw: horizontal_grid(
                        np.arange(-90.0, 90.01, 5.0), np.arange(0.0, 360.0, 5.0),
                        C20, S20, FH20, **kw))):
        try:
            fn(cancel=lambda: True)
            raised = ""
        except RuntimeError as exc:
            raised = str(exc)
        check(f"{who}路径：cancel 立刻生效（抛 RuntimeError('cancelled')）",
              raised == "cancelled", f"{raised!r}")
    # synthesis_horizontal 这一层也要能取消（界面走的就是它）
    co_geo = SHCoeffs(C20, S20).with_unit("geopotential")
    try:
        synthesis_horizontal(LAT_PTS, LON_PTS, co_geo, cancel=lambda: True)
        raised2 = ""
    except RuntimeError as exc:
        raised2 = str(exc)
    check("synthesis_horizontal：取消同样透传", raised2 == "cancelled",
          f"{raised2!r}")

    # 不传 progress/cancel 时结果与传了**逐值相同**（钩子不能改变数值）
    b = horizontal_field(LAT_PTS, LON_PTS, C20, S20, FH20,
                         progress=lambda m, f: None, cancel=lambda: False)
    check("★ 带钩子与不带钩子的结果逐值相同（钩子不改数值）",
          np.array_equal(a["north"], b["north"])
          and np.array_equal(a["east"], b["east"]), "")


def t_fft_longitude_path():
    """B3：经度 FFT 路径 ≡ 直接法，且**自动选择 + 自动报告 + 可回退**。"""
    import time
    from shkit.gradient import fft_path_applicable, horizontal_grid
    L = 20
    C, S = rand_coeffs(L, seed=7)
    fh = degree_factors_horizontal(L)
    lat = np.arange(-90.0, 90.01, 2.0)
    lon = np.arange(0.0, 360.0, 2.0)                    # nlon=180 > 2*20 ✓

    rep_d, rep_f = {}, {}
    t0 = time.perf_counter()
    d = horizontal_grid(lat, lon, C, S, fh, method="direct", report=rep_d)
    t_dir = time.perf_counter() - t0
    t0 = time.perf_counter()
    f = horizontal_grid(lat, lon, C, S, fh, method="fft", report=rep_f)
    t_fft = time.perf_counter() - t0

    for k in ("north", "east", "magnitude"):
        check(f"FFT ≡ 直接法：{k}（相对 ≤1e-14）", rel(f[k], d[k]) <= 1e-14,
              f"rel = {rel(f[k], d[k]):.2e}")
    poles = np.abs(np.abs(lat) - 90.0) < 1e-12
    check("FFT ≡ 直接法：极点行",
          rel(f["north"][poles], d["north"][poles]) <= 1e-14
          and rel(f["east"][poles], d["east"][poles]) <= 1e-14,
          f"north {rel(f['north'][poles], d['north'][poles]):.1e}")
    check("报告写明走了哪条路", rep_d.get("path") == "direct" and rep_f.get("path") == "fft",
          f"direct→{rep_d.get('path')}, fft→{rep_f.get('path')}")
    check("FFT 更快（≥3×，宽松下限防波动）", t_dir / max(t_fft, 1e-9) >= 3.0,
          f"direct {t_dir*1e3:.1f} ms → fft {t_fft*1e3:.1f} ms（{t_dir/max(t_fft,1e-9):.0f}×）")

    # ---- 起始经度相位回归（B3 曾经的真缺陷）-------------------------------
    # 上面的用例全部用 lon = 0,1,2,...，那里 e^{i m·0} = 1，所以**漏掉相位也全对**。
    # 实测过：修之前 lam0=0.5 相对差 5.8e-02、lam0=-179.5 达 1.2e+00。
    worst = 0.0
    for lam0 in (0.0, 0.5, -179.5, 90.3, 123.4, 359.0):
        lo = lam0 + np.arange(0.0, 360.0, 5.0)
        dd = horizontal_grid(lat, lo, C, S, fh, method="direct")
        ff = horizontal_grid(lat, lo, C, S, fh, method="fft")
        sc = max(float(np.abs(dd["north"]).max()),
                 float(np.abs(dd["east"]).max()), 1e-300)
        e = max(float(np.abs(ff["north"] - dd["north"]).max()),
                float(np.abs(ff["east"] - dd["east"]).max())) / sc
        worst = max(worst, e)
        check(f"★ 起始经度 lam0={lam0:g} 时 FFT ≡ 直接法", e <= 1e-12,
              f"rel = {e:.2e}")
    check("★ 起始经度相位回归（6 个 lam0）总体 ≤1e-12", worst <= 1e-12,
          f"max = {worst:.2e}")

    # ---- m=0 规则：纯 C_10 且 S=0 → u_E ≡ 0，u_N = F_1·√3·cos φ（两条路都要对）
    Z = np.zeros((L + 1, L + 1))
    C10 = Z.copy(); C10[1, 0] = 1.0
    for tag, m in (("direct", "direct"), ("fft", "fft")):
        o = horizontal_grid(lat, lon, C10, Z, fh, method=m)
        want = fh[1] * np.sqrt(3.0) * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
        check(f"纯 C_10 ({tag})：u_N = F^h_1·√3 cos φ（m=0 有贡献）",
              rel(o["north"], want) <= 1e-13, f"rel = {rel(o['north'], want):.2e}")
        check(f"纯 C_10 ({tag})：u_E ≡ 0", float(np.max(np.abs(o["east"]))) == 0.0,
              f"max|u_E| = {float(np.max(np.abs(o['east']))):.1e}")

    # ---- 回退与拒绝
    cases = [("区域经度", np.arange(70.0, 141.0, 2.0)),
             ("非均匀经度", np.array([0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0])),
             ("nlon ≤ 2·nmax", np.arange(0.0, 360.0, 10.0))]
    for tag, lo in cases:
        ok, why = fft_path_applicable(lo, L)
        check(f"{tag}：判定为不可用且给出原因", (not ok) and len(why) > 6, why[:44])
        rep = {}
        horizontal_grid(lat, np.sort(lo), C, S, fh, method="auto", report=rep)
        check(f"{tag}：auto 回退直接法并报告", rep.get("path") == "direct",
              str(rep.get("reason"))[:44])
        try:
            horizontal_grid(lat, np.sort(lo), C, S, fh, method="fft")
            check(f"{tag}：method='fft' 应拒绝", False, "没有报错")
        except ValueError as e:
            check(f"{tag}：method='fft' 明确拒绝（不静默回退）",
                  "不适用" in str(e), str(e).splitlines()[0][:44])

    # ---- 多时次：auto 与 direct 一致
    C3 = np.stack([C, C * 2.0, C * 0.3], axis=2)
    S3 = np.stack([S, S * 2.0, S * 0.3], axis=2)
    a = horizontal_grid(lat, lon, C3, S3, fh, method="auto")
    b = horizontal_grid(lat, lon, C3, S3, fh, method="direct")
    check("3D 网格：auto(FFT) ≡ direct", a["north"].shape == (lat.size, lon.size, 3)
          and rel(a["north"], b["north"]) <= 1e-14 and rel(a["east"], b["east"]) <= 1e-14,
          f"shape {a['north'].shape}, rel = {rel(a['north'], b['north']):.2e}")


# ---------------------------------------------------------------------------
def main():
    tests = [
        t_gradient_iron_law,
        t_zero_degree_annihilated,
        t_degree_factor_uses_l,
        t_amplitude_anchor_with_radial,
        t_units_registry,
        t_reversal_refused,
        t_end_to_end_from_ewh,
        t_derived_components,
        t_poles_in_package,
        t_multitime_and_chunking,
        t_smoothing_does_not_commute,
        t_scattered_and_regional,
        t_progress_and_cancel,
        t_fft_longitude_path,
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
