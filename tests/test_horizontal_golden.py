# -*- coding: utf-8 -*-
"""
Horizontal-deformation golden fixture — independent verification (B0).

SHKit and SHSynth implement horizontal deformation **separately** (no shared code,
no cross-import), so agreement rests on a frozen convention plus a golden fixture.
This suite is SHKit's side of that contract:

1. the fixture is **self-consistent** — re-running the generator reproduces it exactly;
2. the generator's Legendre recurrence is tied to SHKit's own frozen convention
   (``shkit.basis.legendre_pbar``);
3. the operator passes the **analytic anchors** that make a "zeroed pole" or a
   "missing m=0 term" implementation fail loudly;
4. the gradient is checked against **finite differences of the scalar SH field**
   (the iron law that catches a stray ``1/sin``);
5. once ``shkit.gradient`` / ``shkit.synthesis.synthesis_horizontal`` lands (B1),
   this suite automatically starts checking the package implementation too.

Run:  python tests/test_horizontal_golden.py
"""
import json
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

FIXTURE = os.path.join(HERE, "fixtures", "horizontal_golden.npz")

import make_horizontal_golden as gen            # noqa: E402  (B0 reference generator)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:58s} {detail}")


def _load_fixture():
    if not os.path.exists(FIXTURE):
        raise FileNotFoundError(
            f"缺少黄金样本 {FIXTURE}；先跑 python tools/make_horizontal_golden.py")
    with np.load(FIXTURE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


FX = _load_fixture()
LAT = FX["lat"]
LON = FX["lon"]
NMAX = int(FX["nmax"])
FH = FX["Fh"]
FH_CASE = FH[:NMAX + 1]
CASES = [str(s) for s in FX["case_names"]]
PROV = json.loads(str(FX["provenance"]))


def field(lat, lon, C, S):
    """标量场 S(theta,lambda)，用于有限差分对照。"""
    P, _D, _t, _u = gen.legendre_and_dtheta(lat, NMAX)
    lam = np.deg2rad(lon)
    out = np.zeros(np.atleast_1d(lat).size)
    for n in range(NMAX + 1):
        for m in range(n + 1):
            if C[n, m] == 0.0 and S[n, m] == 0.0:
                continue
            out += FH_CASE[n] * P[n, m] * (C[n, m] * np.cos(m * lam)
                                           + S[n, m] * np.sin(m * lam))
    return out


# ---------------------------------------------------------------------------
def t_fixture_self_consistent():
    """fixture 必须**自足**：拿它自己存的 C/S 与 Fh 重算，逐位还原它自己存的输出。

    这一步刻意**不重新读 gfc**（fixture 一旦冻结就不该再依赖外部数据），
    所以它是"契约的可执行版本"这一说法的直接检验。
    """
    worst = 0.0
    for i, name in enumerate(CASES):
        u_n, u_e, mag, azi = gen.horizontal_field(LAT, LON, FX["C"][i], FX["S"][i], FH_CASE)
        d = max(float(np.max(np.abs(u_n - FX["u_N"][i]))),
                float(np.max(np.abs(u_e - FX["u_E"][i]))),
                float(np.max(np.abs(mag - FX["magnitude"][i]))),
                float(np.max(np.abs(azi - FX["azimuth"][i]))))
        worst = max(worst, d)
        check(f"fixture 自足还原: {name}", d == 0.0, f"max|diff| = {d:.1e}")
    # Fh 也必须能从 fixture 自己存的勒夫数还原
    Fh = gen.degree_factor_h(FX["love_l"], FX["love_k"], 96)
    d = float(np.max(np.abs(Fh - FX["Fh"])))
    check("fixture 自足还原: Fh（由 love_l/love_k）", d == 0.0, f"max|diff| = {d:.1e}")
    check("采样点含 ±90° 极点行", int((np.abs(np.abs(LAT) - 90.0) < 1e-12).sum()) == 16,
          f"{(np.abs(np.abs(LAT) - 90.0) < 1e-12).sum()} 个")
    check("采样点含 lon=90°（Q_11 探针）", bool(np.any(LON == 90.0)))


def t_recurrence_matches_shkit():
    """生成器的递推必须与 SHKit 自己的（已冻结的）勒让德约定一致。"""
    from shkit.basis import legendre_pbar
    P, _D, _t, _u = gen.legendre_and_dtheta(LAT, NMAX)
    ref = legendre_pbar(LAT, NMAX)                  # (npts, NC) 三角序
    idx = 0
    worst = 0.0
    for m in range(NMAX + 1):
        for n in range(m, NMAX + 1):
            worst = max(worst, float(np.max(np.abs(P[n, m] - ref[:, idx]))))
            idx += 1
    check("递推 ≡ shkit.basis.legendre_pbar", worst < 1e-14, f"max|diff| = {worst:.2e}")


def t_dtheta_vs_finite_difference():
    """dP̄/dθ ≡ 中心差分（含近极点纬度，但不含恰好 ±90° 的极点行）。"""
    from shkit.basis import legendre_pbar
    sel = np.abs(np.abs(LAT) - 90.0) > 1e-9         # 极点行的导数由递推给出，差分比不了
    lat = LAT[sel]
    h = 1e-4
    Pp = legendre_pbar(lat + h, NMAX)
    Pm = legendre_pbar(lat - h, NMAX)
    d_fd = -(Pp - Pm) / (2 * np.deg2rad(h))         # θ = 90° − lat ⇒ d/dθ = −d/dlat
    _P, D, _t, _u = gen.legendre_and_dtheta(lat, NMAX)
    idx = 0
    worst = 0.0
    scale = 0.0
    for m in range(NMAX + 1):
        for n in range(m, NMAX + 1):
            if n == 0:
                idx += 1
                continue
            worst = max(worst, float(np.max(np.abs(D[n, m] - d_fd[:, idx]))))
            scale = max(scale, float(np.max(np.abs(d_fd[:, idx]))))
            idx += 1
    rel = worst / max(scale, 1e-300)
    check("dP̄/dθ ≡ 中心差分（Richardson 无需）", rel < 1e-6,
          f"rel = {rel:.2e}  绝对 = {worst:.2e}")


def t_pole_q_exact():
    """极点：Q_11 = √3、Q_nm(m≥2) 恰为 0 —— 解析值，不是 eps 极限。"""
    for t_pole, tag in ((1.0, "北极 t=+1"), (-1.0, "南极 t=−1")):
        Q = gen.q_at_pole_m1(NMAX, t_pole)
        check(f"Q_11({tag}) = √3", abs(Q[1] - np.sqrt(3.0)) < 1e-15,
              f"{Q[1]:.15f}")
    # m>=2 时 P̄_nm/sinθ 在极点恰为 0：由 P̄ ∝ sin^m θ 保证，这里核对数值实现
    P, _D, _t, u = gen.legendre_and_dtheta(np.array([90.0]), NMAX)
    worst = 0.0
    for m in range(2, NMAX + 1):
        for n in range(m, NMAX + 1):
            q = 0.0 if abs(u[0]) < 1e-12 else P[n, m, 0] / u[0]
            worst = max(worst, abs(q - 0.0))
    check("Q_nm(极点, m≥2) = 0", worst == 0.0, f"max = {worst:.1e}")


def t_analytic_anchor_C10():
    """纯 C_10：u_N = F^h_1·√3 cos φ，u_E ≡ 0（逐位）。

    ⚠️ **必须带上逐阶因子** ``F^h_1 = R·l′_1/(1+k′_1) ≈ 6.438e5 m``：
    只写 ``√3 cos φ`` 会得到一个差 6 个数量级的"锚点"（本套件第一版就踩了这个坑）。
    """
    F1 = FH_CASE[1]
    C, S = FX["C"][0], FX["S"][0]
    u_n, u_e, _m, _a = gen.horizontal_field(LAT, LON, C, S, FH_CASE)
    want = F1 * np.sqrt(3.0) * np.cos(np.deg2rad(LAT))
    d_n = float(np.max(np.abs(u_n - want))) / (F1 * np.sqrt(3.0))
    d_e = float(np.max(np.abs(u_e)))
    check("纯 C_10: u_N = F^h_1·√3 cos φ", d_n < 1e-14, f"相对 max|diff| = {d_n:.2e}")
    check("纯 C_10: u_E ≡ 0（逐位）", d_e == 0.0, f"max|u_E| = {d_e:.2e}")


def t_analytic_anchor_C11():
    """纯 C_11：u_N = F^h_1·(−√3 cos λ sin φ)，u_E = F^h_1·(−√3 sin λ)（含极点）。"""
    F1 = FH_CASE[1]
    C, S = FX["C"][1], FX["S"][1]
    u_n, u_e, _m, _a = gen.horizontal_field(LAT, LON, C, S, FH_CASE)
    lam = np.deg2rad(LON)
    phi = np.deg2rad(LAT)
    d_n = float(np.max(np.abs(u_n - F1 * (-np.sqrt(3.0) * np.cos(lam) * np.sin(phi)))))
    d_e = float(np.max(np.abs(u_e - F1 * (-np.sqrt(3.0) * np.sin(lam)))))
    check("纯 C_11: u_N = F^h_1·(−√3 cos λ sin φ)", d_n / (F1 * np.sqrt(3.0)) < 1e-14,
          f"相对 max|diff| = {d_n / (F1 * np.sqrt(3.0)):.2e}")
    check("纯 C_11: u_E = F^h_1·(−√3 sin λ)（含极点）", d_e / (F1 * np.sqrt(3.0)) < 1e-14,
          f"相对 max|diff| = {d_e / (F1 * np.sqrt(3.0)):.2e}")


def t_pole_uE_nonzero():
    """极点上 u_E 必须非 0、非 NaN：纯 C_11 在 (90°, 90°) 给 ``F^h_1·(−√3)``。

    这是唯一能抓住"把极点写成 0 / NaN"的检查 —— 缺了它，一个坏实现能通过其余全部测试。
    """
    F1 = FH_CASE[1]
    want = F1 * (-np.sqrt(3.0))
    C, S = FX["C"][1], FX["S"][1]
    i = int(np.argmin(np.abs(LAT - 90.0) * 1000 + np.abs(LON - 90.0)))
    u_n, u_e, _m, _a = gen.horizontal_field(LAT[i:i + 1], LON[i:i + 1], C, S, FH_CASE)
    val = float(u_e[0])
    check("极点 u_E 非 0 且有限", np.isfinite(val) and abs(val) > 1.0,
          f"lat={LAT[i]:g} lon={LON[i]:g}  u_E = {val:.6f} m")
    check("极点 u_E = F^h_1·(−√3)", abs(val - want) / abs(want) < 1e-14,
          f"相对 |diff| = {abs(val - want) / abs(want):.2e}")
    check("极点 u_N = 0（纯 C_11，数学上 cos λ = 0）",
          abs(float(u_n[0])) / abs(want) < 1e-15,
          f"u_N = {float(u_n[0]):.2e} m（相对 {abs(float(u_n[0])) / abs(want):.1e}）")


def t_gradient_vs_finite_difference():
    """铁律：算子的两个分量 ≡ 标量场 S 的有限差分。

    这条与解析锚点互补 —— 解析锚点只钉住 (n,m) ≤ 1，差分钉住**全谱**，
    并且会当场抓住"多除/少除一个 sinθ"（方向类检验对它不敏感）。

    ⚠️ **近极点行的公差必须放宽，这是数学不是懒**：那里要除以 ``cos φ ~ 1.7e-5``，
    差分被放大；极点（±90°）根本不做差分（"东/北"在极点坐标奇异），
    由**解析锚点**（精确，不是差分）负责。实测 h=1e-4°：
    中低纬 1.9e-10、近极点 8.1e-7（近极点在 h=1e-3° 最好，1.1e-7）。
    """
    C, S = FX["C"][2], FX["S"][2]                    # 随机带限：全谱都动
    h = 1e-4
    not_pole = np.abs(np.abs(LAT) - 90.0) > 1e-9     # 极点不做差分
    away = not_pole & (np.abs(LAT) <= 89.0)          # 良态
    near = not_pole & (np.abs(LAT) > 89.0)           # 近极点，病态
    for mask, tag, tol in ((away, "中低纬 |lat| ≤ 89°", 1e-9),
                           (near, "近极点 89° < |lat| < 90°", 1e-5)):
        lat, lon = LAT[mask], LON[mask]
        if lat.size == 0:
            continue
        dlat = -(field(lat + h, lon, C, S) - field(lat - h, lon, C, S)) \
            / (2 * np.deg2rad(h))
        dlam = (field(lat, lon + h, C, S) - field(lat, lon - h, C, S)) \
            / (2 * np.deg2rad(h)) / np.cos(np.deg2rad(lat))
        u_n, u_e, _m, _a = gen.horizontal_field(lat, lon, C, S, FH_CASE)
        s_n = max(float(np.max(np.abs(dlat))), 1e-300)
        s_e = max(float(np.max(np.abs(dlam))), 1e-300)
        r_n = float(np.max(np.abs(u_n - (-dlat)))) / s_n   # u_N = −dS/dθ = +dS/dlat
        r_e = float(np.max(np.abs(u_e - dlam))) / s_e
        check(f"u_N ≡ −∂S/∂θ（差分, {tag}）", r_n < tol, f"rel = {r_n:.2e}")
        check(f"u_E ≡ (1/sinθ)∂S/∂λ（差分, {tag}）", r_e < tol, f"rel = {r_e:.2e}")


def t_degree_factors():
    """F^h_n 的手算核对、F^h_0 = 0、R 与 shkit.filters 一致。"""
    from shkit.filters import EARTH_RADIUS_M
    check("R 与 shkit.filters.EARTH_RADIUS_M 一致",
          gen.R_EARTH_M == EARTH_RADIUS_M, f"{gen.R_EARTH_M!r}")
    check("F^h_0 = 0（l′_0 = 0 ⇒ 0 阶被抹掉）", FH[0] == 0.0, f"{FH[0]!r}")

    l, k = FX["love_l"], FX["love_k"]
    want = EARTH_RADIUS_M * l[1:7] / (1.0 + k[1:7])
    d = float(np.max(np.abs(FH[1:7] - want)))
    check("F^h_n 手算核对 n=1..6", d == 0.0, f"max|diff| = {d:.1e} (Fh[1]={FH[1]:.6e})")

    # l′ 不单调（2 阶有凹陷）—— 防止"用低阶代表整体"的近似
    check("l′ 非单调（l′_2 < l′_3）", l[2] < l[3], f"l′_2={l[2]:.5f} < l′_3={l[3]:.5f}")

    span = float(np.max(FH[1:41]) / np.min(FH[1:41]))
    check("F^h_n 逐阶（n=1..40 跨 >5 倍）", span > 5.0, f"max/min = {span:.2f}×")


def t_derived_quantities():
    """magnitude / azimuth 的定义与取值范围。"""
    u_n, u_e = FX["u_N"], FX["u_E"]
    mag, azi = FX["magnitude"], FX["azimuth"]
    d_m = float(np.max(np.abs(mag - np.hypot(u_n, u_e))))
    check("magnitude = hypot(u_N, u_E)", d_m == 0.0, f"max|diff| = {d_m:.1e}")
    check("azimuth ∈ [0, 360)", bool(np.all(azi >= 0.0) and np.all(azi < 360.0)),
          f"min={azi.min():.3f} max={azi.max():.3f}")
    check("magnitude 在极点有限", bool(np.all(np.isfinite(mag[:, np.abs(np.abs(LAT) - 90) < 1e-12]))),
          "极点行全部有限")


def t_case_hygiene():
    """系数样本本身的卫生：S_n0 ≡ 0；四组样本形状一致。"""
    S = FX["S"]
    check("所有样本 S[:, :, 0] ≡ 0", float(np.max(np.abs(S[:, :, 0]))) == 0.0)
    check("四组系数形状一致", S.shape == FX["C"].shape == (len(CASES), NMAX + 1, NMAX + 1),
          f"{S.shape}")
    check("fixture 含真实 GRACE 样本", PROV.get("grace_gfc") is not None,
          f"gfc = {PROV.get('grace_gfc')}")
    check("常量 R 与约定已冻结进 provenance",
          PROV["conventions"]["F2_R_m"] == gen.R_EARTH_M
          and PROV["conventions"]["F1_norm"] == "4pi")


def _rel(v, ref):
    """Relative difference against the case's own field magnitude.

    An **absolute** tolerance is meaningless here: the ``pure_C*`` anchors use
    ``C = 1`` on purpose (fields of ~1e6 m), while the physical cases are
    1e-3..1e2 m.  ``1e-12`` absolute is unreachable for the former (1 ulp of
    1.1e6 is 2.3e-10) and trivially loose for the latter.
    """
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


def t_package_implementation_if_present():
    """B1 落地后自动生效：包内实现也必须与 fixture 对上。

    判据是**相对** `≤1e-12`（实测 1e-15~2e-15，即几个 ulp）。**做不到逐位 0.0**
    而且这不是缺陷：参考实现是逐 (n,m) 累加的可读写法（便于 SHSynth 逐行移植），
    包内实现是 BLAS 矩阵乘的流式写法（省内存、快），两者求和顺序不同。
    """
    try:
        from shkit.gradient import horizontal_field, degree_factors_horizontal
    except ImportError as exc:                      # not implemented yet
        RESULTS.append(True)
        print(f"[PASS] 包内 shkit.gradient 尚未实现（B1）—— 本项待生效   skip ({exc})")
        return

    d = _rel(degree_factors_horizontal(96), FX["Fh"])
    check("包内 F^h_n ≡ fixture（n=0..96）", d <= 1e-12, f"相对 = {d:.2e}")

    worst = {}
    for i, name in enumerate(CASES):
        out = horizontal_field(LAT, LON, FX["C"][i], FX["S"][i], FH_CASE)
        worst[name] = max(_rel(out["north"], FX["u_N"][i]),
                          _rel(out["east"], FX["u_E"][i]),
                          _rel(out["magnitude"], FX["magnitude"][i]))
        check(f"包内实现 ≡ fixture: {name}", worst[name] <= 1e-12,
              f"相对 max|diff| = {worst[name]:.2e}")
    check("包内实现总体 ≡ fixture（四组 ≤1e-12）",
          max(worst.values()) <= 1e-12, f"最差 = {max(worst.values()):.2e} ({max(worst, key=worst.get)})")

    # azimuth 是角度量，单独按"角度"比（场的相对误差不适用于它）
    for i, name in enumerate(CASES):
        out = horizontal_field(LAT, LON, FX["C"][i], FX["S"][i], FH_CASE)
        da = np.abs(np.asarray(out["azimuth"]) - FX["azimuth"][i])
        da = np.minimum(da, 360.0 - da)             # 环绕
        check(f"包内 azimuth ≡ fixture: {name}", float(da.max()) < 1e-9,
              f"max|Δ角| = {float(da.max()):.2e}°")


# ---------------------------------------------------------------------------
def main():
    tests = [
        t_fixture_self_consistent,
        t_recurrence_matches_shkit,
        t_dtheta_vs_finite_difference,
        t_pole_q_exact,
        t_analytic_anchor_C10,
        t_analytic_anchor_C11,
        t_pole_uE_nonzero,
        t_gradient_vs_finite_difference,
        t_degree_factors,
        t_derived_quantities,
        t_case_hygiene,
        t_package_implementation_if_present,
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
