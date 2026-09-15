# -*- coding: utf-8 -*-
"""
shkit.units
===========

Which physical quantity do a set of spherical harmonic coefficients describe,
and how do you convert between the possibilities?

Why this module exists
----------------------
A grid of numbers has no physical meaning on its own.  The same numbers could be

* a **geoid** undulation in metres,
* an **equivalent water height** in cm,
* a **surface density** in kg/m²,
* a **radial displacement** in mm,
* or a plain **dimensionless** field (e.g. an averaging kernel),

and the spherical harmonic coefficients of each are **different sets of
numbers**.  Going from dimensionless potential coefficients (what GRACE Level-2
publishes) to EWH multiplies degree ``n`` by

.. math::

    A_n = \\frac{R\\,\\bar\\rho_E}{3\\rho_w}\\cdot\\frac{2n+1}{1+k'_n}

which is **not constant**: with the bundled load Love numbers it already runs
from ``1.17e7`` at ``n = 0`` to ``1.68e8`` at ``n = 6``, a factor 14 within six
degrees (and it keeps growing like ``n``).  So the two coefficient sets can
neither be added, compared, nor converted by a single number, and applying
``A_n`` twice inflates the result by ~1e7-1e8.

This module keeps an explicit ``field_unit`` label on every
:class:`~shkit.coeffs.SHCoeffs` and routes all conversions through one
canonical intermediate - the **dimensionless geopotential coefficients**
``C_nm, S_nm`` - so the arithmetic is always unambiguous.

Conversion factors
------------------
From dimensionless geopotential coefficients to a target quantity:

===========================  =================================  ===============
target ``field_unit``        per-degree factor from ``C_nm``     needs
===========================  =================================  ===============
``'geopotential'``           ``1``                              -
``'geoid'``                  ``R``  (constant, not per-degree)  -
``'surface_density'``        ``R·rho_ave/3 · (2n+1)/(1+k'_n)``  ``k'_n``
``'ewh'``                    ``R·rho_ave/(3·rho_w) · (2n+1)/(1+k'_n)``  ``k'_n``
``'radial_displacement'``    ``R·h'_n/(1+k'_n)``                ``k'_n`` and ``h'_n``
===========================  =================================  ===============

and ``'scalar'`` is a **terminal** label: "this is just the field, in whatever
unit the numbers carry".  Nothing can be converted to or from it, because there
is no defined physics.

Derivations (4-pi normalised harmonics, ``R = a``)
--------------------------------------------------
Writing the load response with the load Love numbers ``k'_n`` (potential) and
``h'_n`` (radial):

* a surface density harmonic ``sigma_nm`` produces the potential harmonic
  ``4*pi*G*a/(2n+1) * sigma_nm``;
* dimensionless Stokes coefficients are defined by
  ``V = (GM/r) sum (a/r)^n C_nm Y_nm``, so
  ``C_nm = 3 (1+k'_n) / ((2n+1) rho_ave a) * sigma_nm``;
* hence ``sigma_nm = rho_ave a (2n+1) / (3 (1+k'_n)) * C_nm`` and
  ``EWH_nm = sigma_nm / rho_w``;
* the geoid is ``N = sum a C_nm Y_nm``, i.e. a factor ``a``;
* the radial displacement is ``u_r = h'_n * V_load / g``, which reduces to
  ``u_r,nm = a h'_n / (1 + k'_n) * C_nm``.

``u_r`` is positive **outward**; with the usual (negative) ``h'_n`` for ``n>=2``
a positive mass load therefore gives subsidence.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .coeffs import SHCoeffs
from .filters import (EARTH_RADIUS_M, RHO_AVE, RHO_WATER, load_lln,
                      load_love_numbers)

__all__ = [
    "FIELD_UNITS",
    "FIELD_UNIT_LABELS",
    "UNITS_NEEDING_LOVE_K",
    "UNITS_NEEDING_LOVE_H",
    "field_unit",
    "with_field_unit",
    "degree_factors",
    "convert",
    "describe_conversion",
    "require_convertible",
]

#: Canonical intermediate: every conversion goes through dimensionless
#: geopotential coefficients.
CANONICAL = "geopotential"

#: Allowed values of ``SHCoeffs.meta['field_unit']``.
FIELD_UNITS = (
    "unknown",              # not declared (legacy files) - treated as scalar
    "scalar",               # just a field; unit carried by the numbers
    CANONICAL,              # dimensionless geopotential C_nm, S_nm (GRACE L2)
    "geoid",                # geoid undulation (metres)
    "surface_density",      # surface mass density (kg/m^2)
    "ewh",                  # equivalent water height
    "radial_displacement",  # elastic radial displacement (metres)
    "horizontal_displacement",   # elastic horizontal displacement (metres, VECTOR)
)

FIELD_UNIT_LABELS = {
    "unknown": "未声明",
    "scalar": "普通标量（无量纲或不作换算的场）",
    "geopotential": "无量纲重力位系数 C_nm/S_nm",
    "geoid": "水准面高 ΔN",
    "surface_density": "面密度 σ",
    "ewh": "等效水高 EWH",
    "radial_displacement": "径向形变 u_r",
    "horizontal_displacement": "水平形变 u_h（矢量，北/东分量）",
}

UNITS_NEEDING_LOVE_K = ("geoid", "surface_density", "ewh", "radial_displacement",
                        "horizontal_displacement")
UNITS_NEEDING_LOVE_H = ("radial_displacement",)
#: Quantities whose per-degree factor needs the **horizontal** load Love number
#: ``l'_n`` (they are the only ones; everything else uses ``h'`` and/or ``k'``).
UNITS_NEEDING_LOVE_L = ("horizontal_displacement",)

_SHORT = {
    "geopotential": "无量纲位系数",
    "geoid": "水准面",
    "surface_density": "面密度",
    "ewh": "EWH",
    "radial_displacement": "径向形变",
    "horizontal_displacement": "水平形变",
    "scalar": "普通标量",
    "unknown": "未声明",
}


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def field_unit(coeffs: SHCoeffs) -> str:
    """Return the declared ``field_unit``, defaulting to ``'unknown'``."""
    u = str(coeffs.meta.get("field_unit", "unknown"))
    return u if u in FIELD_UNITS else "unknown"


def with_field_unit(coeffs: SHCoeffs, unit: str, **meta) -> SHCoeffs:
    """Tag a coefficient set with its physical meaning (metadata only)."""
    if unit not in FIELD_UNITS:
        raise ValueError(f"field_unit must be one of {FIELD_UNITS}, got {unit!r}")
    out = coeffs.copy()
    out.meta["field_unit"] = unit
    out.meta.update(meta)
    return out


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------
def degree_factors(target: str, nmax: int, *,
                   love_numbers: Optional[Sequence[float]] = None,
                   love_numbers_h: Optional[Sequence[float]] = None,
                   love_numbers_l: Optional[Sequence[float]] = None,
                   radius_m: float = EARTH_RADIUS_M,
                   rho_ave: float = RHO_AVE,
                   rho_water: float = RHO_WATER) -> np.ndarray:
    """Per-degree factor converting **from** ``geopotential`` **to** ``target``.

    Parameters
    ----------
    target : str
        One of :data:`FIELD_UNITS` except ``'scalar'`` / ``'unknown'``.
    nmax : int
        Maximum degree.
    love_numbers : sequence, optional
        Load Love numbers ``k'_n``.  Defaults to the bundled table
        (``SHKit/data/love_numbers.npy``, PREM / Wang 2012).
    love_numbers_h : sequence, optional
        Load Love numbers ``h'_n``.  Defaults to the bundled
        ``SHKit/data/load_love_numbers.npz`` (PREM, Wang et al. 2012), which
        also carries ``l'_n``.  Only needed for
        ``target='radial_displacement'``.
    love_numbers_l : sequence, optional
        Load Love numbers ``l'_n`` (same bundled file).  Only needed for
        ``target='horizontal_displacement'``.

    Notes
    -----
    ``horizontal_displacement`` is **not** a per-degree-only quantity: the
    factor here is only the radial part ``R*l'_n/(1+k'_n)``; the field itself is
    the *horizontal gradient* of the scaled scalar, which mixes ``(n, m)`` with
    ``(n-1, m)``.  Use :func:`shkit.gradient.synthesis_horizontal` to evaluate it.
    """
    n = np.arange(nmax + 1, dtype=float)

    if target == CANONICAL:
        return np.ones(nmax + 1)
    if target == "geoid":
        return np.full(nmax + 1, float(radius_m))
    if target not in FIELD_UNITS or target in ("scalar", "unknown"):
        raise ValueError(
            f"cannot define a degree factor for field_unit={target!r}; "
            f"usable targets are 'geopotential', 'geoid', 'surface_density', "
            f"'ewh', 'radial_displacement', 'horizontal_displacement'")

    kl = load_love_numbers() if love_numbers is None else np.asarray(love_numbers, float)
    kl = np.asarray(kl, dtype=float)
    kn = np.array([kl[i] if 0 <= i < kl.size else 0.0 for i in range(nmax + 1)])

    if target == "surface_density":
        # sigma_n = rho_ave * R * (2n+1) / (3 (1+k'_n)) * C_n     [kg/m^2]
        return radius_m * rho_ave / 3.0 * (2 * n + 1.0) / (1.0 + kn)

    if target == "ewh":
        # EWH_n = sigma_n / rho_w  =  R rho_ave / (3 rho_w) * (2n+1)/(1+k'_n) * C_n
        #       = A_n * C_n
        return (radius_m * rho_ave / (3.0 * rho_water)
                * (2 * n + 1.0) / (1.0 + kn))

    if target == "radial_displacement":
        if love_numbers_h is None:
            love_numbers_h = load_lln()["h"]
        hl = np.asarray(love_numbers_h, dtype=float)
        hn = np.array([hl[i] if 0 <= i < hl.size else 0.0 for i in range(nmax + 1)])
        # u_r,n = R h'_n/(1+k'_n) C_n ; 向外为正，h'<0 时正载荷下沉
        return radius_m * hn / (1.0 + kn)

    if target == "horizontal_displacement":
        if love_numbers_l is None:
            love_numbers_l = load_lln()["l"]
        ll = np.asarray(love_numbers_l, dtype=float)
        ln = np.array([ll[i] if 0 <= i < ll.size else 0.0 for i in range(nmax + 1)])
        # u_h,n = R l'_n/(1+k'_n) * grad_H(...) ; l'_0 = 0 ⇒ 0 阶被抹掉
        return radius_m * ln / (1.0 + kn)

    raise AssertionError("unreachable")


def forward_factors(src: str, nmax: int, **kw) -> np.ndarray:
    """**正变换**因子 ``f_u``：``C_nm = a_nm / f_u``。

    这里 ``a_nm`` 是该网格（或散点场）**自身**的球谐系数，``C_nm`` 是**经典
    无量纲位系数**（GRACE Level-2 公布的那一套）。也就是说「输入是」选的是
    这一条公式：

    ====================  ==========================================
    ``src``              ``f_u``
    ====================  ==========================================
    ``geopotential``     1（恒等：网格本身就是位系数场）
    ``geoid``            ``R``（常数）
    ``surface_density``  ``R·ρ̄/3 · (2n+1)/(1+k′ₙ)``
    ``ewh``              ``R·ρ̄/(3ρ_w) · (2n+1)/(1+k′ₙ)`` = ``Aₙ``
    ``radial_displacement``  ``R·h′ₙ/(1+k′ₙ)``
    ``scalar`` / ``unknown``  1（没有物理公式，导出的就是它自己的系数）
    ====================  ==========================================

    ``horizontal_displacement`` **不在**上表里：它是矢量，装不进 ``SHCoeffs``，
    不能被声明为「输入是」（契约 F9/F11）。要它请用
    :func:`shkit.gradient.synthesis_horizontal`。

    同一串数字声明成 geoid 还是声明成 EWH，是两个不同的物理场，所以除以的
    因子不同 —— 导出的 ``C_nm`` 也就不同（第 2 阶差 ``A₂/R ≈ 13.2`` 倍）。
    """
    if src in ("scalar", "unknown"):
        return np.ones(nmax + 1)
    return degree_factors(src, nmax, **kw)


_FORMULAS = {
    "geopotential": "1（网格本身就是无量纲位系数场）",
    "scalar": "1（无物理公式，导出的就是该场自身的系数）",
    "unknown": "1（未声明物理量，导出的就是该场自身的系数）",
    "geoid": "R = 6378136.46 m（常数）",
    "surface_density": "R·ρ̄/3 · (2n+1)/(1+k′ₙ)",
    "ewh": "Aₙ = R·ρ̄/(3ρ_w) · (2n+1)/(1+k′ₙ)",
    "radial_displacement": "R·h′ₙ/(1+k′ₙ)",
    "horizontal_displacement": "R·l′ₙ/(1+k′ₙ)  （北/东分量由球面梯度给出，不是逐阶乘法）",
}


def formula_text(unit: str) -> str:
    """Human-readable ``f`` for the forward/inverse formula of ``unit``."""
    return _FORMULAS.get(unit, f"（未知物理量 {unit!r}）")


# ---------------------------------------------------------------------------
# division with zero factors
# ---------------------------------------------------------------------------
ZERO_FACTOR_TOL = 1e-8


def _divide_by(coeffs: SHCoeffs, factors, what: str) -> tuple:
    """``coeffs / factors``，并处理**因子为 0** 的阶。

    因子为 0 的阶（目前只有径向形变的 0 阶：``h′₀ = 0``）在物理上不可反推，
    只能取 0。若数据在该阶明显不为 0，说明它不符合该物理模型，报错而不是
    悄悄地给出 inf/NaN。

    Returns ``(coeffs, zero_degrees)``.
    """
    f = np.asarray(factors, dtype=float).ravel()[: coeffs.nmax + 1]
    if not np.all(np.isfinite(f)):
        raise ValueError(
            f"「{what}」的换算因子含非有限值，无法换算"
            "（通常是载荷勒夫数表在该阶缺失）。")
    C = np.array(coeffs.C, dtype=float, copy=True)
    S = np.array(coeffs.S, dtype=float, copy=True)
    zero = np.flatnonzero(f == 0.0)
    bad = []
    if zero.size:
        scale = float(np.sqrt(np.mean(coeffs.C ** 2 + coeffs.S ** 2))) or 1.0
        for d in zero:
            d = int(d)
            if d >= C.shape[0]:
                continue
            content = max(float(np.max(np.abs(C[d, :]))),
                          float(np.max(np.abs(S[d, :]))))
            if content <= ZERO_FACTOR_TOL * scale:
                if C.ndim == 2:
                    C[d, :] = 0.0
                    S[d, :] = 0.0
                else:
                    C[d, :, :] = 0.0
                    S[d, :, :] = 0.0
            else:
                bad.append((d, content))
    if bad:
        lst = "、".join(f"第 {d} 阶（该阶系数已达 {v:.3g}）" for d, v in bad)
        raise ValueError(
            f"无法完成「{what}」：{lst} 的换算因子为 0，信息在该阶丢失。\n"
            + ("径向形变的 0 阶载荷勒夫数 h′₀ = 0 —— 全球均匀载荷不产生"
               "（相对参考系定义的）径向位移，所以 u_r 的均值与载荷无关，"
               "不能由它反推位系数的 0 阶。\n"
               if "径向形变" in what else "")
            + "请改从位系数 / EWH / geoid 出发，或先用其它信息确定该阶后再合并。")
    finv = np.zeros_like(f)
    ok = f != 0.0
    finv[ok] = 1.0 / f[ok]
    if C.ndim == 2:
        C = C * finv[:, None]
        S = S * finv[:, None]
    else:
        C = C * finv[:, None, None]
        S = S * finv[:, None, None]
    return SHCoeffs(C, S, dict(coeffs.meta), coeffs.times), [int(d) for d in zero]


def to_canonical(coeffs: SHCoeffs, src: Optional[str] = None, **kw) -> SHCoeffs:
    """**正变换**：「某物理量的网格」算出的系数 → 经典无量纲位系数 ``C_nm``。

    这是「输入是」所选的公式，也是导出系数文件里装的东西::

        C_nm = a_nm / f_u           (f_u = forward_factors(src))

    ``scalar`` / ``unknown`` 没有物理公式，因子为 1，导出的就是该数组自己的
    系数（区域平均核这类无物理量纲的场走这一档）。
    """
    if src is None:
        src = field_unit(coeffs)
    out, zeroed = _divide_by(
        coeffs, forward_factors(src, coeffs.nmax, **kw),
        f"从「{FIELD_UNIT_LABELS.get(src, src)}」正变换为位系数")
    out.meta = dict(out.meta)
    out.meta["field_unit"] = (src if src in ("scalar", "unknown") else CANONICAL)
    out.meta["forward_from"] = src
    if zeroed:
        out.meta["zero_degrees"] = zeroed
    return out


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def require_convertible(src: str, dst: str) -> Optional[str]:
    """Return ``None`` if the conversion is defined, else a Chinese explanation."""
    if src == dst:
        return None
    if dst == "horizontal_displacement":
        # 契约 F9：矢量性体现在**综合算子与产品层**，不在容器里。
        # SHCoeffs 装的是"一个标量场的系数"，装不下一个矢量场。
        return (
            "不能把「{}」换算成「水平形变 u_h」——SHCoeffs 装的是**一个标量场**的"
            "系数，而水平形变是**矢量**（北/东两个分量），容器里装不下。\n"
            "正确做法：用 shkit.gradient.synthesis_horizontal(coeffs, ...) 直接综合出"
            "北/东分量；其中的逐阶因子 degree_factors('horizontal_displacement', L) "
            "只负责径向部分 R·l′ₙ/(1+k′ₙ)，真正的水平场由球面梯度给出。".format(
                FIELD_UNIT_LABELS.get(src, src)))
    if src in ("scalar", "unknown"):
        extra = ""
        if dst == "ewh":
            extra = (
                "\n\n为什么不能替你猜：换算到 EWH 的因子取决于起点 —— "
                "位系数 → EWH 要乘 Aₙ（n=2 时 8.44e7），"
                "而 geoid → EWH 只要乘约 13.2，两者差整整一个地球半径 R。"
                "同一串数字，起点不同结果差 1e7 倍，所以必须由你说明。")
        return (
            f"源系数的物理含义是「{FIELD_UNIT_LABELS.get(src, src)}」，"
            f"没有定义到「{FIELD_UNIT_LABELS.get(dst, dst)}」的换算。\n"
            "「无物理含义的标量场」（例如区域平均核、任意无量纲网格）"
            "只代表它自己，没有已知的物理换算。"
            "要换算，请先用 field_unit= 声明它到底是位系数、geoid 还是 EWH。"
            + extra +
            "\n\n如果你只是要做「网格 → 系数 → 网格」，"
            "那本来就不需要换算：把目标设为「不换算」即可。")
    if dst in ("scalar", "unknown"):
        return (
            f"不能把「{FIELD_UNIT_LABELS.get(src, src)}」换算成"
            f"「{FIELD_UNIT_LABELS.get(dst, dst)}」——"
            "「无物理含义的标量场」不是一个可换算的目标。")
    if src == "horizontal_displacement":
        # 契约 F11：水平形变是**矢量**，而且它是位系数的球面梯度 —— 反推只能到
        # 差一个常数 C00（常数的梯度恒为 0）。与「h′₀ = 0，径向形变不能反推 0 阶」
        # 同一种"宁可拒绝也不猜"。
        return (
            "「水平形变 u_h」是**矢量**（北/东两个分量），不能当作标量场换算到"
            f"「{FIELD_UNIT_LABELS.get(dst, dst)}」。\n"
            "而且水平形变是位系数的**球面梯度**：常数项的梯度恒为 0，所以由 "
            "(u_N, u_E) 反推位系数只到**差一个常数 C00**，分量也已在综合时丢失、"
            "过程不可逆。\n"
            "正确做法：分别导出北分量与东分量；要反推位系数请改用径向形变"
            "（field_unit=\"radial_displacement\"），或用显式 c00= 约束另做反演。")
    return None


def convert(coeffs: SHCoeffs, target: str, *,
            love_numbers: Optional[Sequence[float]] = None,
            love_numbers_h: Optional[Sequence[float]] = None,
            radius_m: float = EARTH_RADIUS_M,
            rho_ave: float = RHO_AVE,
            rho_water: float = RHO_WATER,
            allow_unit_mismatch: bool = False) -> SHCoeffs:
    """Convert coefficients from their declared unit to ``target``.

    All conversions are routed through the canonical dimensionless geopotential
    coefficients, so ``ewh -> geoid`` or ``surface_density -> ewh`` work even
    though only the geopotential direction has a textbook formula.

    Raises
    ------
    ValueError
        If the source unit is ``'scalar'``/``'unknown'`` (no physics defined),
        if the target is not a quantity, or if a Love number table that the
        conversion needs is missing.
    """
    if target not in FIELD_UNITS:
        raise ValueError(f"unknown target field_unit {target!r}; "
                         f"choose from {FIELD_UNITS}")
    src = field_unit(coeffs)

    problem = require_convertible(src, target)
    if problem is not None:
        if not (allow_unit_mismatch and src == "unknown" and target == CANONICAL):
            raise ValueError(problem)

    if src == target:
        return coeffs.copy()

    L = coeffs.nmax
    kw = dict(love_numbers=love_numbers, love_numbers_h=love_numbers_h,
              radius_m=radius_m, rho_ave=rho_ave, rho_water=rho_water)

    # source -> canonical
    if src == CANONICAL:
        x = coeffs
    else:
        # 反推与正变换用的是同一个分母；因子为 0 的阶单独处理，不再悄悄地
        # 产生 inf/NaN（见 _divide_by）。
        x, _zeroed = _divide_by(
            coeffs, degree_factors(src, L, **kw),
            f"从「{FIELD_UNIT_LABELS.get(src, src)}」反推")

    # canonical -> target
    if target == CANONICAL:
        out = x
    else:
        f_dst = degree_factors(target, L, **kw)
        out = _scale(x, f_dst)

    out.meta = dict(out.meta)
    out.meta["field_unit"] = target
    out.meta["converted_from"] = src
    out.meta.pop("degree_filter", None)
    return out


def _scale(coeffs: SHCoeffs, factors: np.ndarray) -> SHCoeffs:
    f = np.asarray(factors, dtype=float).ravel()[:coeffs.nmax + 1]
    if coeffs.C.ndim == 2:
        C = coeffs.C * f[:, None]
        S = coeffs.S * f[:, None]
    else:
        C = coeffs.C * f[:, None, None]
        S = coeffs.S * f[:, None, None]
    return SHCoeffs(C, S, dict(coeffs.meta), coeffs.times)


def describe_conversion(nmax: int = 8, **kw) -> str:
    """Human-readable table of the per-degree factors (for docs and tests)."""
    lines = [f"{'n':>3}  " + "".join(f"{_SHORT[t]:>16}" for t in
                                      ("geoid", "surface_density", "ewh"))]
    cols = {t: degree_factors(t, nmax, **kw)
            for t in ("geoid", "surface_density", "ewh")}
    for n in range(nmax + 1):
        lines.append(f"{n:>3}  " + "".join(f"{cols[t][n]:>16.6e}"
                                           for t in ("geoid", "surface_density",
                                                     "ewh")))
    return "\n".join(lines)
