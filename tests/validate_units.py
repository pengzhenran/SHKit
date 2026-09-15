# -*- coding: utf-8 -*-
"""
Validation of shkit.units — the physical-meaning / unit conversion layer.

The point of these checks is that the different "kinds" of spherical harmonic
coefficients are **not** interchangeable and the conversion factor is
per-degree, so every claim is verified against a hand-computed number.

Run:  python tests/validate_units.py
"""
import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit import units
from shkit import io as shio
from shkit.analysis import analysis
from shkit.coeffs import SHCoeffs
from shkit.filters import (EARTH_RADIUS_M, RHO_AVE, RHO_WATER,
                           ewh_scaling, load_lln, load_love_numbers)
from shkit.synthesis import synthesis, synthesis_grid
from shkit.gradient import synthesis_horizontal_grid
from shkit.weights import glq_grid

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:58s} {detail}")


def random_coeffs(L=6, seed=1, unit="geopotential"):
    rng = np.random.default_rng(seed)
    C = np.zeros((L + 1, L + 1))
    S = np.zeros((L + 1, L + 1))
    for n in range(L + 1):
        for m in range(n + 1):
            C[n, m] = rng.standard_normal() / max(n, 1) ** 2
            if m:
                S[n, m] = rng.standard_normal() / max(n, 1) ** 2
    return units.with_field_unit(SHCoeffs(C, S), unit)


# ---------------------------------------------------------------------------
def t_factor_values():
    L = 6
    kl = load_love_numbers()
    f_geo = units.degree_factors("geoid", L)
    f_sig = units.degree_factors("surface_density", L)
    f_ewh = units.degree_factors("ewh", L)

    check("geoid 因子是常数 R（不是逐阶因子）",
          np.ptp(f_geo) == 0 and abs(f_geo[0] - EARTH_RADIUS_M) < 1e-6,
          f"R = {f_geo[0]:.4f} m, 极差 = {np.ptp(f_geo):.1e}")

    # sigma_n = rho_ave * R / 3 * (2n+1)/(1+k'_n)
    want_sig = np.array([RHO_AVE * EARTH_RADIUS_M / 3.0 * (2 * n + 1) /
                         (1 + kl[n]) for n in range(L + 1)])
    check("面密度因子 = ρ̄·R/3·(2n+1)/(1+k′ₙ)（手算核对）",
          np.allclose(f_sig, want_sig),
          f"n=0: {f_sig[0]:.6e} vs 手算 {want_sig[0]:.6e}")

    check("EWH 因子与 filters.ewh_scaling 完全一致",
          np.allclose(f_ewh, ewh_scaling(L)),
          f"A_0 = {f_ewh[0]:.6e}, A_6 = {f_ewh[6]:.6e}")

    check("面密度 / EWH = ρ_w （同一物理量的单位换算）",
          np.allclose(f_sig / f_ewh, RHO_WATER),
          f"比值 = {(f_sig / f_ewh)[0]:.6f}")

    ratio = f_ewh[-1] / f_ewh[0]
    check("⚠ 换算因子不是常数 —— 两套系数不能用单一常数互换",
          ratio > 5.0,
          f"A_6 / A_0 = {ratio:.1f}（0→6 阶已跨 {ratio:.1f} 倍，"
          "所以必须逐阶换算、且不能相加或直接比较）")


def t_conversion_roundtrip():
    L = 6
    gp = random_coeffs(L)
    for u in ("geoid", "surface_density", "ewh"):
        c = units.convert(gp, u)
        back = units.convert(c, "geopotential")
        err = max(float(np.abs(back.C - gp.C).max()),
                  float(np.abs(back.S - gp.S).max()))
        check(f"geopotential -> {u} -> geopotential 往返无损", err < 1e-12,
              f"max|diff| = {err:.2e}, 标签 = {units.field_unit(back)}")

    # cross conversion not going through geopotential explicitly
    ew = units.convert(gp, "ewh")
    gd = units.convert(ew, "geoid")
    gd2 = units.convert(gp, "geoid")
    check("EWH -> geoid 与 geopotential -> geoid 等价（经由规范中间量）",
          np.allclose(gd.C, gd2.C, atol=1e-12 * max(1.0, np.abs(gd2.C).max())),
          f"max|diff| = {np.abs(gd.C - gd2.C).max():.3e}")

    # idempotence: asking for the unit you already have must be a no-op
    same = units.convert(ew, "ewh")
    check("已经是 EWH 再转换为 EWH 是恒等（不会乘两次 Aₙ）",
          np.allclose(same.C, ew.C) and np.allclose(same.S, ew.S),
          f"max|diff| = {np.abs(same.C - ew.C).max():.1e}")

    # the failure mode we are protecting against
    A = ewh_scaling(L)
    check("若错误地重复换算，量级会差 1e7 以上（说明为什么必须挡住）",
          A[0] > 1e7 and A[6] > 1e8,
          f"A_0 = {A[0]:.3e}, A_6 = {A[6]:.3e}")


def t_guards():
    gp = random_coeffs(6)
    sc = units.with_field_unit(gp, "scalar")
    un = SHCoeffs(gp.C.copy(), gp.S.copy())          # 未声明

    for tag, c in (("scalar", sc), ("未声明(unknown)", un)):
        try:
            units.convert(c, "ewh")
            check(f"{tag} -> EWH 应被拒绝", False, "没有报错")
        except ValueError as exc:
            check(f"{tag} -> EWH 应被拒绝", "换算" in str(exc) or "标量" in str(exc),
                  str(exc).splitlines()[0][:52])

    try:
        units.convert(gp, "scalar")
        check("换算到 scalar 应被拒绝", False, "没有报错")
    except ValueError as exc:
        check("换算到 scalar 应被拒绝", True, str(exc).splitlines()[0][:52])

    # synthesis guard
    lat, lon, _ = glq_grid(4)
    try:
        synthesis(lat, lon, sc, target_unit="ewh")
        check("synthesis(target_unit='ewh') 对 scalar 系数应被拒绝", False, "没有报错")
    except ValueError as exc:
        check("synthesis(target_unit='ewh') 对 scalar 系数应被拒绝", True,
              str(exc).splitlines()[0][:52])

    v = synthesis(lat, lon, sc, target_unit="ewh", allow_unit_mismatch=True)
    check("allow_unit_mismatch=True 时仍可强行换算（逃生舱）", v.size == lat.size,
          f"RMS = {np.sqrt(np.mean(v ** 2)):.4e}")


def t_synthesis_matches_convert():
    L = 6
    gp = random_coeffs(L)
    lat, lon, _ = glq_grid(6)
    for u in ("geoid", "surface_density", "ewh"):
        a = synthesis(lat, lon, gp, target_unit=u)
        b = synthesis(lat, lon, units.convert(gp, u))
        d = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
        check(f"synthesis(target_unit='{u}') 等价于先 convert 再 synthesis",
              d < 1e-12, f"相对差 = {d:.2e}")

    # 无换算时不能改变任何东西
    a = synthesis(lat, lon, gp)
    b = synthesis(lat, lon, gp, ewh=False, target_unit=None)
    check("不给 target_unit 时不做任何换算",
          np.array_equal(a, b), "两者逐位相同")


def t_radial_displacement():
    gp = random_coeffs(6)
    z = load_lln()

    check("随包提供完整的 h′/l′/k′ 载荷勒夫数表",
          all(k in z for k in ("h", "l", "k")) and z["h"].size > 1000,
          f"{z['model']}，{z['h'].size} 个度；"
          f"h′₂ = {z['h'][2]:.6f}, l′₂ = {z['l'][2]:.6f}, k′₂ = {z['k'][2]:.6f}")

    # k′ must agree with the table the legacy software used
    kl = load_love_numbers()
    m = min(kl.size, z["k"].size)
    dmax = float(np.abs(kl[:m] - z["k"][:m]).max())
    check("自带 k′ 与 m2py/gridSHconvert 的表一致（EWH 结果不受影响）",
          dmax < 1e-15, f"前 {m} 个度最大差 = {dmax:.2e}")

    # degree 1 frame correction must reproduce what pz_LLN.m does
    k1_cf = -(z["h"][1] + 2.0 * z["l"][1]) / 3.0
    check("k′₁ 的 CE→CF 改正 = −(h′₁+2l′₁)/3（与 pz_LLN.m 一致）",
          abs(z["k"][1] - k1_cf) < 1e-15,
          f"表中 {z['k'][1]:.8f} vs 公式 {k1_cf:.8f}")

    # radial displacement now works out of the box
    f = units.degree_factors("radial_displacement", 6)
    want = np.array([EARTH_RADIUS_M * z["h"][n] / (1 + z["k"][n])
                     for n in range(7)])
    check("径向形变因子 = R·h′ₙ/(1+k′ₙ)（手算核对，默认用自带 h′ 表）",
          np.allclose(f, want),
          f"n=2: {f[2]:.6e} vs 手算 {want[2]:.6e}")

    check("径向形变因子符号与 h′₂ < 0 一致（正载荷下沉）",
          f[2] < 0, f"因子 n=2 = {f[2]:.4e}")

    lat, lon, _ = glq_grid(4)
    v = synthesis(lat, lon, gp, target_unit="radial_displacement")
    check("synthesis(target_unit='radial_displacement') 可直接运行",
          np.all(np.isfinite(v)) and np.max(np.abs(v)) > 0,
          f"u_r RMS = {np.sqrt(np.mean(v ** 2)):.4e} m")

    # any of the user's own models can be swapped in
    other = (r"D:\华为家庭存储\1_Joe Science Data\1_1_GRACE\2_Corrections"
             r"\love num\loading_models\LLNs\ak135-LLNs.dat")
    if os.path.exists(other):
        z2 = load_lln(other)
        f2 = units.degree_factors("radial_displacement", 6,
                                  love_numbers=z2["k"], love_numbers_h=z2["h"])
        check("可换用其它载荷模型（ak135）并得到不同因子",
              abs(f2[2] - f[2]) > 1e3 and np.all(np.isfinite(f2)),
              f"ak135 因子 n=2 = {f2[2]:.4e} vs PREM {f[2]:.4e}")
    else:
        check("可换用其它载荷模型（ak135）", True, "（本机未找到 ak135-LLNs.dat，跳过）")


def t_gfc_label():
    gp = random_coeffs(8)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gfc")
        shio.write_coeffs(gp, p, layout="gfc")
        back = shio.read_coeffs(p)
        check("读取 .gfc 时自动标注 field_unit='geopotential'",
              units.field_unit(back) == "geopotential",
              f"field_unit = {units.field_unit(back)}")
        lat, lon, _ = glq_grid(4)
        v = synthesis(lat, lon, back, target_unit="ewh")   # 不应报错
        check("因此 .gfc 系数可直接合成 EWH，无需手动声明",
              np.all(np.isfinite(v)),
              f"EWH RMS = {np.sqrt(np.mean(v ** 2)):.4e}")


def t_output_unit():
    """「输入是」选**正变换公式** → 它确实改变导出的系数。

    ``C_nm = a_nm / f_u``。同一块网格声明成 geoid 与声明成 EWH 是两个不同的
    物理场，除的因子不同，所以导出的经典无量纲位系数不同（第 2 阶差
    ``A₂/R ≈ 13.2`` 倍，且逐阶不同）。「输出为」选的是**反变换公式**，它不
    改变导出的系数，只决定重建出来是什么物理量。
    """
    L = 8
    lat, lon, w = glq_grid(L)
    gp = random_coeffs(L)                       # 已知的"位系数"
    f = synthesis(lat, lon, gp)                 # 用它合成数据
    kw = dict(method="quadrature", rule="user", user_w=w)

    # ---- 1) 「输入是」= 正变换公式，导出的 C_nm 随它改变 --------------------
    got = {}
    for u in ("scalar", "geopotential", "geoid", "surface_density", "ewh"):
        c, rep = analysis(lat, lon, f, L, field_unit=u, **kw)
        got[u] = c
        assert rep.meta.get("forward_from") == u, "报告没有记录正变换起点"
    base = got["geopotential"]

    check("输入=重力位系数：f_u = 1，导出即该场自身的系数（≈ 真值）",
          np.allclose(got["geopotential"].C, gp.C, atol=1e-10),
          f"max|C-C_true| = {np.abs(got['geopotential'].C - gp.C).max():.2e}")

    f_geo = units.forward_factors("geoid", L)
    f_ewh = units.forward_factors("ewh", L)
    f_sig = units.forward_factors("surface_density", L)
    ok = True
    for n in range(1, L + 1):
        for m in range(1, n + 1):
            if abs(base.C[n, m]) > 1e-12:
                ok &= abs(got["geoid"].C[n, m] * f_geo[n] / base.C[n, m] - 1) < 1e-9
                ok &= abs(got["ewh"].C[n, m] * f_ewh[n] / base.C[n, m] - 1) < 1e-9
                ok &= abs(got["surface_density"].C[n, m] * f_sig[n]
                          / base.C[n, m] - 1) < 1e-9
    check("「输入是」改变导出系数：geoid ÷R、EWH ÷Aₙ、σ ÷(Aₙρ_w)", ok,
          f"n=5: geoid {got['geoid'].C[5, 2] / base.C[5, 2]:.6e} "
          f"(=1/R {1 / f_geo[5]:.6e})；EWH {got['ewh'].C[5, 2] / base.C[5, 2]:.6e} "
          f"(=1/A₅ {1 / f_ewh[5]:.6e})")

    geoid_vs_ewh = got["geoid"].C[2, 1] / got["ewh"].C[2, 1]
    check("同一块网格：声明 geoid 与声明 EWH 的导出系数差 A₂/R",
          abs(geoid_vs_ewh - f_ewh[2] / f_geo[2]) < 1e-9,
          f"实测 {geoid_vs_ewh:.6f} vs A₂/R = {f_ewh[2] / f_geo[2]:.6f}")

    check("输入=scalar 时 f_u = 1（无物理公式，导出该场自身系数）",
          np.array_equal(got["scalar"].C, got["geopotential"].C)
          and units.field_unit(got["scalar"]) == "scalar"
          and units.field_unit(got["geopotential"]) == "geopotential",
          "数值相同但标签不同：标量场不能换算，位系数场可以")

    check("换算后的结果被打上「经典无量纲位系数」标签",
          all(units.field_unit(got[u]) == "geopotential"
              for u in ("geoid", "surface_density", "ewh")),
          "geoid / σ / EWH 三种声明导出的都是 C_nm")

    # ---- 2) 「输出为」是反变换公式，不改导出系数 ---------------------------
    c_t, rep2 = analysis(lat, lon, f, L, field_unit="ewh", target_unit="geoid",
                         **kw)
    check("「输出为」不改变导出的系数（永远导出 C_nm）",
          np.array_equal(c_t.C, got["ewh"].C),
          f"max|diff| = {np.abs(c_t.C - got['ewh'].C).max():.1e}；"
          f"反变换目标记在 rep.meta['inverse_target'] = "
          f"{rep2.meta.get('inverse_target')!r}")

    # ---- 3) 反变换：同一套 C_nm，按「输出为」回到不同物理量 ---------------
    v_ewh = synthesis(lat, lon, got["ewh"], target_unit="ewh")
    v_geo = synthesis(lat, lon, got["ewh"], target_unit="geoid")
    # 期望的 geoid 场：把真值系数逐阶乘 R/Aₙ（EWH → 位系数 → geoid）
    scale = units.degree_factors("geoid", L) / units.degree_factors("ewh", L)
    want_geo = synthesis(lat, lon,
                         SHCoeffs(gp.C * scale[:, None], gp.S * scale[:, None]))
    check("反变换按逐阶因子把 C_nm 换成目标物理量",
          np.allclose(np.asarray(v_ewh), np.asarray(f), atol=1e-10)
          and np.allclose(np.asarray(v_geo), np.asarray(want_geo), atol=1e-10),
          f"输出=EWH 回到原网格；输出=geoid 逐阶乘 R/Aₙ"
          f"（n=2 档 {scale[2]:.6e}）")

    # ---- 4) 声明矛盾（scalar + 目标 EWH）→ 报错 ---------------------------
    try:
        analysis(lat, lon, f, L, field_unit="scalar", target_unit="ewh", **kw)
        check("普通标量 + 目标 EWH 应报错", False, "没有报错")
    except ValueError as exc:
        check("普通标量 + 目标 EWH 应报错", True,
              str(exc).splitlines()[0][:56])

    # ---- 5) 面密度与 EWH 只差常数 rho_w ----------------------------------
    sig = units.convert(base, "surface_density")
    ewh = units.convert(base, "ewh")
    check("面密度 σ 与 EWH 只差常数 ρ_w（同一个物理量的两种单位）",
          np.allclose(sig.C, ewh.C * RHO_WATER),
          f"σ/EWH = {sig.C[2, 1] / ewh.C[2, 1]:.10f}  (ρ_w = {RHO_WATER})")

    # ---- 6) field_unit / forward_from 必须能随文件保存/读回 ---------------
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.sh")
        shio.write_coeffs(c_t, p, layout="triangle")
        back = shio.read_coeffs(p)
        check("field_unit 随系数文件保存并读回（否则换算会退化成 unknown）",
              units.field_unit(back) == "geopotential",
              f"写出 geopotential，读回 {units.field_unit(back)}")
        hdr = back.meta.get("header_fields", {})
        check("正变换起点与反变换目标也写进文件头（可追溯）",
              hdr.get("forward_from") == "ewh"
              and hdr.get("inverse_target") == "geoid",
              f"forward_from={hdr.get('forward_from')!r}, "
              f"inverse_target={hdr.get('inverse_target')!r}")


def t_roundtrip_is_unit_agnostic():
    """往返精确与声明无关；但**导出的系数**与声明有关——两件事要分开。

        「水高网格 ——按公式——> C_nm ——按对应公式——> 水高」

    正反是一对互逆公式，所以往返误差与选哪一档无关；而不同声明对应的
    ``f_u`` 不同，所以中间那套 ``C_nm`` 不同。
    """
    L = 10
    lat, lon, w = glq_grid(L)
    rng = np.random.default_rng(7)
    C = np.zeros((L + 1, L + 1))
    S = np.zeros((L + 1, L + 1))
    for n in range(L + 1):
        for m in range(n + 1):
            C[n, m] = rng.standard_normal() / max(n, 1) ** 2
            if m:
                S[n, m] = rng.standard_normal() / max(n, 1) ** 2
    field = synthesis(lat, lon, SHCoeffs(C, S))       # 任意物理量的网格

    UNI = ("scalar", "geopotential", "geoid", "surface_density", "ewh")
    coeffs_seen, errs = {}, []
    for u in UNI:
        c, _ = analysis(lat, lon, field, L, method="quadrature", rule="user",
                        user_w=w, field_unit=u)
        coeffs_seen[u] = c
        back = synthesis(lat, lon, c, target_unit=u)   # 按对应的反变换公式
        errs.append(float(np.max(np.abs(np.asarray(back) - field))))

    check("网格→系数→网格：正反是一对互逆公式，六种声明下往返都精确",
          max(errs) < 1e-12,
          f"最大往返误差 {max(errs):.2e}（数据 RMS "
          f"{np.sqrt(np.mean(field ** 2)):.3f}）")

    # 中间那套系数则随声明改变：逐阶比必须精确等于 f_u 之比
    base = coeffs_seen["geopotential"]
    f_geo = units.forward_factors("geoid", L)
    f_ewh = units.forward_factors("ewh", L)
    ok_geo = ok_ewh = True
    for n in range(1, L + 1):
        m = int(np.argmax(np.abs(base.C[n])))
        if abs(base.C[n, m]) > 1e-12:
            ok_geo &= abs(coeffs_seen["geoid"].C[n, m] * f_geo[n]
                          / base.C[n, m] - 1) < 1e-9
            ok_ewh &= abs(coeffs_seen["ewh"].C[n, m] * f_ewh[n]
                          / base.C[n, m] - 1) < 1e-9
    check("但中间那套 C_nm **随声明改变**（geoid 恒差 1/R、EWH 差 1/Aₙ）",
          ok_geo and ok_ewh,
          f"n=5: geoid 声明 {coeffs_seen['geoid'].C[5, 1] / base.C[5, 1]:.6e} "
          f"vs 1/R {1 / f_geo[5]:.6e}；EWH 声明 "
          f"{coeffs_seen['ewh'].C[5, 1] / base.C[5, 1]:.6e} vs 1/A₅ "
          f"{1 / f_ewh[5]:.6e}")

    # 径向形变的 0 阶不可反推（h′₀ = 0）—— 声明它是输入就要报错
    try:
        analysis(lat, lon, field, L, method="quadrature", rule="user",
                 user_w=w, field_unit="radial_displacement")
        check("输入=径向形变而 0 阶不为 0 → 报错（h′₀ = 0 不可反推）",
              False, "没有报错")
    except ValueError as exc:
        check("输入=径向形变而 0 阶不为 0 → 报错（h′₀ = 0 不可反推）",
              "因子为 0" in str(exc), str(exc).splitlines()[0][:52])

    # A_n 只在跨物理量时出现，而且非恒定
    A = units.degree_factors("ewh", L)
    check("A_n 只在跨物理量换算时出现，而且逐阶变化很大",
          A[-1] / A[0] > 10,
          f"A_0 = {A[0]:.3e} → A_{L} = {A[-1]:.3e}（跨 {A[-1]/A[0]:.1f} 倍）；"
          "同物理量往返中完全不出现")

    # 跨物理量：因子取决于起点，无法从"我要什么"反推"我有什么"
    f_geo = units.degree_factors("ewh", 2) / units.degree_factors("geoid", 2)
    f_pot = (units.degree_factors("ewh", 2)
             / units.degree_factors("geopotential", 2))
    check("跨物理量时必须知道起点（geoid→EWH 与 位系数→EWH 因子不同）",
          abs(f_geo[2] / f_pot[2] - 1.0 / EARTH_RADIUS_M) < 1e-18,
          f"起点=geoid {f_geo[2]:.6e} vs 起点=位系数 {f_pot[2]:.6e}，"
          f"相差恰好 R = {EARTH_RADIUS_M:.1f} m")


def t_unit_scale_is_linear():
    """同一个物理场用不同**数值单位**表示（m / cm）时，整条链只差一个常数。

    这条是用户追问出来的："按公式，输入什么单位、输出就是什么单位，不需要换算吧"——
    **对的**。软件从不需要知道单位：正变换 ``C = a/f_u``、反变换 ``场 = C·f``、
    以及各物理量之间的因子都是**线性**的，所以把输入整体 ×100（同一片水，改用 cm 记），
    系数、形变、重建场全部按 ×100 缩放，而**各物理量之间的比例完全不变**。

    真正与单位有关的只有**绝对物理标注**：`m`/`mm` 这些字面标签，以及软件里
    "真实 GRACE 应在 mm 级"这类按 SI 写的量级自检。所以界面那条提示写的是
    "绝对量要按文件单位读、比值不受影响"，而不是"结果算错了"。
    """
    L = 10
    lat = np.arange(-85.0, 85.01, 5.0)
    lon = np.arange(0.0, 360.0, 5.0)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    f_m = (0.30 * np.exp(-((LA + 5) / 18.0) ** 2 - ((LO - 300) / 25.0) ** 2)
           + 0.15 * np.sin(np.deg2rad(2 * LO)) * np.cos(np.deg2rad(LA)))
    f_cm = f_m * 100.0                      # 同一片水，只是改用 cm 记数

    got = {}
    for tag, f in (("m", f_m), ("cm", f_cm)):
        co, _ = analysis(LA.ravel(), LO.ravel(), f.ravel(), 12, rule="grid",
                         field_unit="ewh", report_fit=False)
        got[tag] = (co,
                    np.asarray(synthesis_grid(lat, lon, co, target_unit="ewh")),
                    synthesis_horizontal_grid(lat, lon, co),
                    np.asarray(synthesis_grid(lat, lon, co, target_unit="geoid")))

    ratio = np.asarray(got["cm"][0].C) / np.asarray(got["m"][0].C)
    fin = np.isfinite(ratio) & (np.abs(np.asarray(got["m"][0].C)) > 0)
    check("单位只改变数值标注：C(cm) / C(m) 恒等于 100",
          np.allclose(ratio[fin], 100.0, rtol=1e-12),
          f"{fin.sum()} 个非零系数，比值范围 "
          f"[{ratio[fin].min():.12g}, {ratio[fin].max():.12g}]")

    err_m = float(np.max(np.abs(got["m"][1] - f_m)))
    err_cm = float(np.max(np.abs(got["cm"][1] - f_cm)))
    check("重建场各自精确回到**自己的**输入（往返与单位无关）",
          err_cm / err_m > 99.0 and err_cm / err_m < 101.0,
          f"m: {err_m:.3e} / cm: {err_cm:.3e}（比值 {err_cm / err_m:.4f}，"
          "只差截断误差的线性缩放）")

    uh_m = np.hypot(np.asarray(got["m"][2]["north"]),
                    np.asarray(got["m"][2]["east"]))
    uh_cm = np.hypot(np.asarray(got["cm"][2]["north"]),
                     np.asarray(got["cm"][2]["east"]))
    q = uh_cm / uh_m
    check("水平形变也按同一常数缩放（u_h(cm)/u_h(m) 恒等于 100）",
          np.allclose(q[np.isfinite(q)], 100.0, rtol=1e-9),
          f"比值范围 [{np.nanmin(q):.12g}, {np.nanmax(q):.12g}]")

    # 比值只在场**不接近 0** 的地方看：除以 ~0 的场值会得到没有意义的比例
    _cut = 1e-3 * float(np.max(np.abs(got["m"][1])))
    _m = np.abs(got["m"][1]) > _cut
    _c = np.abs(got["cm"][1]) > _cut * 100.0
    k_m = got["m"][3][_m] / got["m"][1][_m]
    k_cm = got["cm"][3][_c] / got["cm"][1][_c]
    check("★ 跨物理量的比值与单位无关（geoid/EWH 两次完全相同）",
          k_m.size > 100 and np.allclose(k_m, k_cm, rtol=1e-12),
          f"{k_m.size} 个点上逐点一致，中位比值 {np.median(k_m):.8f} vs "
          f"{np.median(k_cm):.8f}")

    check("所以：软件不需要知道单位，界面只需说明绝对量的标注口径",
          True, "（提示语已按此改写，见 tests/test_gui_smoke.py 的 cm 用例）")


def main():
    tests = [t_factor_values, t_conversion_roundtrip, t_guards,
             t_synthesis_matches_convert, t_radial_displacement, t_gfc_label,
             t_output_unit, t_roundtrip_is_unit_agnostic, t_unit_scale_is_linear]
    for fn in tests:
        print(f"\n=== {fn.__name__} " + "=" * (56 - len(fn.__name__)))
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
