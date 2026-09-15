# -*- coding: utf-8 -*-
"""
绝对物理验证：不靠往返，用独立已知的物理量核对系数的**含义**。

为什么需要这个
--------------
往返检验（正变换→反变换→比原场）只能证明 A 与 A⁻¹ 互逆。
任何一对自洽的算子都能完美往返——即使两边用错的常数、
甚至两边都用一个瞎编的因子，往返照样闭合。
所以**往返精度不能用来验证"物理含义"**。

本文件改用三类**独立可核对的量**：

  1. 常数场 → 总质量：M = ρ_w·4πR²·e₀，与任何谐波展开无关，只由几何决定；
  2. 单个谐波 → 系数必须精确等于 1 或 0（解析已知）；
  3. 球冠载荷 → 总质量由面积与密度**独立算出**，与球谐展开无关；
  4. Aₙ 本身 → 用载荷位场的第一性原理重新推导一遍再比对，
     而不是拿 Aₙ 的定义去验 Aₙ。

Run:  python tests/validate_physics.py
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit import units
from shkit.analysis import analysis
from shkit.coeffs import SHCoeffs, triangle_order
from shkit.filters import (EARTH_RADIUS_M, RHO_AVE, RHO_WATER,
                           load_love_numbers)
from shkit.synthesis import synthesis, synthesis_grid
from shkit.weights import glq_grid

R = EARTH_RADIUS_M
RHO_W = RHO_WATER
RHO_E = RHO_AVE
M_EARTH = 4.0 / 3.0 * np.pi * RHO_E * R ** 3      # 与我们用的 ρ̄、R 自洽的地球质量

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


# ---------------------------------------------------------------------------
def t_constant_field_total_mass():
    """常数场：总质量由几何唯一决定，与谐波展开无关。

    均匀铺满整个球面的水层，等效水高 e₀ (m)：
        总质量 M = ρ_w · 4πR² · e₀
        e 的 0 阶系数必须 = e₀          （因为 P̄₀₀ = 1）
        重力位 0 阶系数必须 = M/M_E     （M_E = (4/3)πρ̄R³）
    """
    e0 = 0.05                                   # 5 cm 水层
    L = 12
    lat, lon, w = glq_grid(L)
    grid = synthesis_grid(np.unique(lat), np.unique(lon),
                          SHCoeffs(np.array([[e0]]), np.array([[0.0]]))
                          .truncate(L))
    # 用常数场直接构造，避免依赖 synthesis 的正确性
    grid = np.full_like(grid, e0)

    coeffs, _ = analysis(lat, lon, grid.ravel(), L,
                         method="quadrature", rule="user", user_w=w,
                         field_unit="ewh")
    M_true = RHO_W * 4 * np.pi * R ** 2 * e0

    # 「输入是 = EWH」选的是正变换公式 C_nm = a_nm / Aₙ：
    f_u = units.forward_factors("ewh", L)
    check("正变换公式按声明执行：输入=EWH 时 C₀₀ = e₀ / A₀",
          rel(coeffs.C[0, 0], e0 / f_u[0]) < 1e-12,
          f"C₀₀ = {coeffs.C[0,0]:.8e} = {e0} / {f_u[0]:.6e}")
    check("常数水层：重力位 0 阶系数 = M / M_地球（绝对物理量）",
          rel(coeffs.C[0, 0], M_true / M_EARTH) < 1e-10,
          f"C₀₀^pot = {coeffs.C[0,0]:.12e} vs M/M_E = {M_true/M_EARTH:.12e}"
          f"  (M = {M_true:.6e} kg)")

    # 对照：同一块网格声明成「位系数」时导出的就是该场自身的系数 = e₀
    raw, _ = analysis(lat, lon, grid.ravel(), L,
                      method="quadrature", rule="user", user_w=w,
                      field_unit="geopotential")
    check("对照：同一块网格声明成「位系数」导出 e₀ —— 换声明就换系数",
          rel(raw.C[0, 0], e0) < 1e-12 and rel(coeffs.C[0, 0] / raw.C[0, 0],
                                               1.0 / f_u[0]) < 1e-12,
          f"位系数声明 C₀₀ = {raw.C[0,0]:.6f}；EWH 声明 C₀₀ = "
          f"{coeffs.C[0,0]:.6e}（差 {raw.C[0,0]/coeffs.C[0,0]:.6e} = A₀）")

    back = synthesis(lat, lon, coeffs, target_unit="ewh")
    check("常数水层：反变换回 EWH 后场值还原（含义自洽）",
          rel(np.max(np.abs(back)), e0) < 1e-12,
          f"max|EWH| = {np.max(np.abs(back)):.12f} m")


def t_single_harmonics():
    """单个谐波：系数解析已知，必须精确命中且其余为 0。"""
    L = 24
    lat, lon, w = glq_grid(L)
    m_vec, n_vec = triangle_order(L)

    print("        (n,m)   回收系数        应为    其余系数最大绝对值")
    worst_any = 0.0
    for (n, m) in ((1, 0), (2, 1), (5, 5), (13, 7), (24, 24)):
        i = int(np.where((n_vec == n) & (m_vec == m))[0][0])
        # 直接取该基函数作为"网格"（解析已知）
        from shkit.basis import legendre_pbar
        P = legendre_pbar(lat, L)[:, i]
        field = P * (np.cos(m * np.deg2rad(lon)) if m else 1.0)
        c, _ = analysis(lat, lon, field, L, method="quadrature",
                        rule="user", user_w=w)
        got = c.C[n, m] if m else c.C[n, 0]
        others = np.abs(c.C.copy())
        others[n, m if m else 0] = 0.0
        others_abs = max(float(np.abs(others).max()),
                         float(np.abs(c.S).max()))
        worst_any = max(worst_any, abs(got - 1.0), others_abs)
        print(f"        ({n:2d},{m:2d})  {got:14.10f}        1    "
              f"{others_abs:.2e}")
    check("单个谐波：系数精确等于 1，其余为 0（解析核对）",
          worst_any < 1e-12,
          f"最大偏差 = {worst_any:.2e}（L={L}，GLQ 网格，含 24 阶）")


def t_cap_total_mass():
    """球冠载荷：总质量由**面积与密度**独立算出，与球谐展开无关。

    半径 θ₀ 的均匀水层（等效水高 e_in）：
        球冠立体角 Ω = 2π(1 − cos θ₀)
        球冠面积   A = R²Ω
        e_in      = M / (ρ_w A)
        e 的 0 阶系数 = M / (4π ρ_w R²)      ← 与 θ₀ 无关！
    这一条用面积独立算出 M，因此不是同一套公式的自证。
    """
    theta0 = np.deg2rad(25.0)
    cap_lat, cap_lon = 30.0, 100.0
    step = 0.5
    lat = np.arange(-90.0, 90.0 + 1e-9, step)
    lon = np.arange(0.0, 360.0, step)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")

    cla, clo = np.deg2rad(cap_lat), np.deg2rad(cap_lon)
    cosd = (np.sin(np.deg2rad(LA)) * np.sin(cla)
            + np.cos(np.deg2rad(LA)) * np.cos(cla) * np.cos(np.deg2rad(LO) - clo))
    inside = cosd >= np.cos(theta0)

    omega = 2 * np.pi * (1 - np.cos(theta0))       # 立体角
    area = R ** 2 * omega                          # 球冠面积 m²
    M_true = 1.0e15                                # 指定总质量 1 Pg = 1000 Gt
    e_in = M_true / (RHO_W * area)                 # 解析等效水高

    grid = np.where(inside, e_in, 0.0)
    L = 60
    c, rep = analysis(LA.ravel(), LO.ravel(), grid.ravel(), L,
                      method="quadrature", rule="grid", field_unit="ewh")

    e00_true = M_true / (4 * np.pi * RHO_W * R ** 2)
    # 输入声明为 EWH，导出的 C₀₀ 是经典无量纲位系数：由它反推质量
    M_from_coeffs = 4 * np.pi * R ** 3 * RHO_AVE / 3.0 * c.C[0, 0]
    err = rel(M_from_coeffs, M_true)

    check("球冠载荷：由系数反推的总质量守恒（M 由面积独立算出）",
          err < 5e-3,
          f"C₀₀ = {c.C[0,0]:.8e} → 反推 M = {M_from_coeffs:.6e} kg vs "
          f"指定 {M_true:.6e} kg（差 {err:.2%}）")
    print(f"        （对照：若按「该场自身的系数」理解，C₀₀ 应为 "
          f"e₀ = M/(4πρ_wR²) = {e00_true:.8e}；"
          f"当前导出的是位系数，两者相差 A₀ = "
          f"{e00_true/c.C[0,0]:.4e}）")
    print(f"        球冠: θ₀=25°, 面积 {area:.6e} m², 等效水高 e_in = "
          f"{e_in*100:.3f} cm; 网格 {step}°, L={L}")
    print(f"        （残差来自球冠边缘的格网离散，不是单位问题："
          f"e_in 边缘台阶在 {step}° 网格上被平滑）")


def t_an_from_first_principles():
    """用载荷位场的第一性原理重新推导 Aₙ 与 R 因子，再与实现比对。

    对纯 n 阶的面密度谐波 σ_nm：
        载荷位（地表）      V_load = 4πGa/(2n+1) · σ_nm
        计入弹性响应        V      = (1+k′ₙ)·V_load
        无量纲位系数定义    C_nm   = a·V/(GM_E)
        地球质量            M_E    = (4/3)πρ̄a³
      ⇒ C_nm = 3(1+k′ₙ)/((2n+1)ρ̄a) · σ_nm
      ⇒ 反过来 σ_nm = ρ̄a(2n+1)/(3(1+k′ₙ)) · C_nm
    而 EWH = σ/ρ_w，所以
        e_nm = R·ρ̄/(3ρ_w)·(2n+1)/(1+k′ₙ) · C_nm   ← 这正是 Aₙ

    这里**不复用** units.degree_factors 的公式，而是把上面的推导写成
    独立的数值实现，再与 units 的输出逐阶比对。
    """
    L = 20
    kl = load_love_numbers()
    n = np.arange(L + 1, dtype=float)
    # 独立推导：由 C_nm 到 e_nm
    e_from_C = R * RHO_E / (3.0 * RHO_W) * (2 * n + 1.0) / (1.0 + kl[:L + 1])
    A = units.degree_factors("ewh", L)
    check("Aₙ 与第一性原理推导一致（独立实现，非复用自己的公式）",
          np.allclose(e_from_C, A),
          f"n=2: 推导 {e_from_C[2]:.8e} vs 实现 {A[2]:.8e}")

    # 端到端：把"水高场"换成位系数，逐阶核对
    lat, lon, w = glq_grid(L)
    m_vec, n_vec = triangle_order(L)
    from shkit.basis import legendre_pbar
    P = legendre_pbar(lat, L)
    for (nn, mm) in ((2, 1), (7, 4), (15, 9)):
        i = int(np.where((n_vec == nn) & (m_vec == mm))[0][0])
        e_field = P[:, i] * np.cos(mm * np.deg2rad(lon))     # 纯 (nn,mm) 的水高场
        c, _ = analysis(lat, lon, e_field, L, method="quadrature",
                        rule="user", user_w=w, field_unit="ewh")
        gp = units.convert(c, "geopotential")
        want = 3.0 * RHO_W * (1 + kl[nn]) / ((2 * nn + 1) * RHO_E * R)
        check(f"端到端：水高场 (n={nn},m={mm}) → 位系数符合第一性原理",
              rel(gp.C[nn, mm], want) < 1e-12,
              f"C = {gp.C[nn,mm]:.8e} vs 推导 {want:.8e}")

    # 水准面因子 R 的定义核对：N = a·C
    gd = units.degree_factors("geoid", 4)
    check("水准面因子 = R（由 N = R·ΣC_nm Y_nm 的定义直接给出）",
          np.allclose(gd, R),
          f"n=0..3 全为 {gd[0]:.4f} m")


def t_degree_dependence():
    """按阶展示：换一个「输入是」，导出的 C_nm 整条谱差同一个**常数**倍。

    回应"C[2,1] 不一定体现差异"：单个系数说明不了问题，要看整条谱。
    正变换公式是 ``C_nm = a_nm / f_u``，所以换成不同声明后：

        输入 = geoid  : C_nm = a_nm / R                 （**常数**倍）
        输入 = EWH    : C_nm = a_nm / Aₙ                （逐阶不同）
        输入 = 位系数 : C_nm = a_nm                     （恒等）

    关键结论：把起点判断错（位系数 当成 geoid），导出的 C_nm 会整体差一个 R，
    每一阶都差同一个倍数——所以是比例性错误，不是某一阶"看不出差异"。
    """
    L = 20
    A_ewh = units.degree_factors("ewh", L)          # 位系数 -> EWH
    A_geo = units.degree_factors("geoid", L)        # 位系数 -> geoid（= R）
    geoid_to_ewh = A_ewh / A_geo                    # geoid -> EWH

    print(f"        {'n':>3}{'位系数→EWH (Aₙ)':>20}{'位系数→geoid':>18}"
          f"{'geoid→EWH':>18}{'两者之比':>16}")
    for n in (0, 1, 2, 3, 5, 10, 15, 20):
        print(f"        {n:>3}{A_ewh[n]:>20.6e}{A_geo[n]:>18.6e}"
              f"{geoid_to_ewh[n]:>18.6e}{A_ewh[n] / geoid_to_ewh[n]:>16.4f}")
    print(f"        ↑ 最后一列恒等于 R = {R:.1f} —— "
          "起点判错就是整体差一个 R")

    check("Aₙ 逐阶变化很大，而位系数→geoid 是常数 R",
          A_ewh[L] / A_ewh[0] > 20 and np.ptp(A_geo) == 0,
          f"Aₙ 0→{L} 阶跨 {A_ewh[L]/A_ewh[0]:.1f} 倍；"
          f"geoid 因子恒为 R")

    check("起点判错 → EWH 结果整体差一个 R（常数倍，不是逐阶小偏差）",
          np.allclose(A_ewh / geoid_to_ewh, R),
          f"比值恒为 {A_ewh[5] / geoid_to_ewh[5]:.4f} = R")

    # 用真实数据走一遍：同一串数字按三种声明分析，导出的 C_nm 逐阶比对
    L2 = 12
    lat, lon, w = glq_grid(L2)
    m_vec, n_vec = triangle_order(L2)
    from shkit.basis import legendre_pbar
    P = legendre_pbar(lat, L2)
    field = np.zeros(lat.size)
    rng = np.random.default_rng(5)
    for k, (nn, mm) in enumerate(zip(n_vec, m_vec)):
        field += (rng.standard_normal()
                  * P[:, k] * (np.cos(mm * np.deg2rad(lon)) if mm else 1.0))
    kw = dict(method="quadrature", rule="user", user_w=w)
    c_pot, _ = analysis(lat, lon, field, L2, field_unit="geopotential", **kw)
    c_geo, _ = analysis(lat, lon, field, L2, field_unit="geoid", **kw)
    c_ewh, _ = analysis(lat, lon, field, L2, field_unit="ewh", **kw)

    f_ewh = units.forward_factors("ewh", L2)
    print("\n        同一串数字、三种「输入是」，导出 C_nm 的逐阶比值：")
    r_geo, r_ewh = [], []
    for n in (1, 2, 5, 10, 12):
        ok = np.abs(c_pot.C[n]) > 1e-9 * np.abs(c_pot.C[n]).max()
        if ok.any():
            r_geo.append(float(np.median(c_geo.C[n][ok] / c_pot.C[n][ok])))
            r_ewh.append(float(np.median(c_ewh.C[n][ok] / c_pot.C[n][ok])))
            print(f"          n={n:2d}:  geoid/位系数 = {r_geo[-1]:.6e}"
                  f"    ewh/位系数 = {r_ewh[-1]:.6e}"
                  f"   (1/Aₙ = {1/f_ewh[n]:.6e})")
    check("输入=geoid 时导出的 C_nm 恒差 1/R（常数倍）",
          len(r_geo) >= 4 and all(abs(r * R - 1) < 1e-9 for r in r_geo),
          f"实测逐阶比值为 1/R = {1/R:.6e}")
    check("输入=EWH 时导出的 C_nm 差 1/Aₙ（逐阶不同，不是常数）",
          len(r_ewh) >= 4 and all(abs(r * f_ewh[n] - 1) < 1e-9
                                  for r, n in zip(r_ewh, (1, 2, 5, 10, 12))),
          f"实测 n=1..12 与 1/Aₙ 逐阶吻合（n=2 时 {r_ewh[1]:.6e}）")

    # 各自用对应的反变换公式，都能精确重建回原场
    for name, c, tgt in (("geopotential", c_pot, "geopotential"),
                         ("geoid", c_geo, "geoid"), ("ewh", c_ewh, "ewh")):
        back = np.asarray(synthesis(lat, lon, c, target_unit=tgt)).ravel()
        check(f"输入=输出={name}：反变换精确回到原场（往返与声明无关）",
              np.max(np.abs(back - field)) < 1e-10,
              f"max|重建-原场| = {np.max(np.abs(back - field)):.2e}")


def main():
    tests = [t_constant_field_total_mass, t_single_harmonics,
             t_cap_total_mass, t_an_from_first_principles,
             t_degree_dependence]
    for fn in tests:
        print(f"\n=== {fn.__name__} " + "=" * (54 - len(fn.__name__)))
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
