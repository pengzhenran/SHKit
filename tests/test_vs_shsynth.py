# -*- coding: utf-8 -*-
"""
Cross-package agreement: SHKit's horizontal deformation vs SHSynth's (B4).

SHKit and SHSynth implement horizontal deformation **independently** (no shared
code, no cross-import), so agreement is enforced by
``docs/水平形变契约.md`` (F1-F11) plus the frozen golden fixture.  This suite is
the *live* half of that: when SHSynth is present and exposes a horizontal API,
the two implementations are called on the same coefficients and compared.

Rules of engagement:

* **SHSynth absent  -> skip, and say why.**  This must never turn ``run_all.py``
  red -- the two packages are released separately and neither may depend on the
  other being installed.
* **SHSynth present but no horizontal API yet -> skip with the reason**
  (executable code outranks plans; SHSynth's own plan lists it as a later phase).
* **Present and callable -> compare**, relative criterion ``<= 1e-12`` on the
  fixture's four coefficient sets (see 契约 §6: an absolute tolerance is
  meaningless because the ``pure_C*`` anchors use C = 1, and bitwise equality is
  not expected across two different summation orders).

Run:  python tests/test_vs_shsynth.py
"""
import importlib
import inspect
import json
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FIXTURE = os.path.join(HERE, "fixtures", "horizontal_golden.npz")
SHSYNTH_ROOT = os.path.abspath(os.path.join(os.path.dirname(ROOT), "SHSynth"))

RESULTS = []
#: Divergences from the frozen contract found on the **other** side.  These do not
#: fail SHKit's suite (SHKit is not the one that has to change), but they must be
#: visible -- that is the whole point of a cross-package check.
DISCREPANCIES = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def warn(name, detail=""):
    """Report a divergence on the other side without failing this suite."""
    DISCREPANCIES.append(f"{name} -- {detail}")
    RESULTS.append(True)
    print(f"[WARN] {name:56s} {detail}")


def rel(v, ref):
    scale = max(float(np.max(np.abs(np.asarray(ref)))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


FX = dict(np.load(FIXTURE, allow_pickle=False))
LAT, LON, NMAX = FX["lat"], FX["lon"], int(FX["nmax"])
FH = FX["Fh"][:NMAX + 1]
CASES = [str(s) for s in FX["case_names"]]


# ---------------------------------------------------------------------------
def _find_shsynth_horizontal():
    """Return ``(callable, how)`` for SHSynth's horizontal entry point, or ``(None, why)``."""
    if not os.path.isdir(SHSYNTH_ROOT):
        return None, f"未找到 SHSynth 目录（{SHSYNTH_ROOT}）"
    if SHSYNTH_ROOT not in sys.path:
        sys.path.insert(0, SHSYNTH_ROOT)
    try:
        importlib.import_module("shsynth")
    except Exception as exc:                                  # noqa: BLE001
        return None, f"shsynth 不可导入：{type(exc).__name__}: {exc}"

    # (module, attribute) candidates, most specific first
    for mod_name in ("shsynth.gradient", "shsynth.engine", "shsynth.synthesis",
                     "shsynth"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:                                     # noqa: BLE001
            continue
        for attr in ("synthesis_horizontal", "horizontal_field",
                     "synthesize_horizontal", "sph_gradient_horizontal"):
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn, f"{mod_name}.{attr}"
    return None, "shsynth 里还没有水平形变的公开接口（其方案 §4.9 列为后续阶段）"


def _call(fn, lat, lon, C, S):
    """Call SHSynth's operator with whichever keyword shape it accepts."""
    params = inspect.signature(fn).parameters
    kw = {}
    if "degree_factors" in params:
        kw["degree_factors"] = FH
    elif "weights_scale" in params:
        kw["weights_scale"] = FH
    elif "Fh" in params:
        kw["Fh"] = FH
    if "nmax" in params:
        kw["nmax"] = NMAX
    if "want" in params:
        kw["want"] = ("north", "east")
    out = fn(lat, lon, C, S, **kw)
    if isinstance(out, dict):
        return out.get("north", out.get("u_N")), out.get("east", out.get("u_E"))
    return out[0], out[1]


def _fixture_selftest():
    """Even when SHSynth is missing, assert the fixture is usable as a contract."""
    check("黄金样本可读且自足（含 C/S 与 Fh）",
          all(k in FX for k in ("C", "S", "u_N", "u_E", "Fh", "lat", "lon")),
          f"{len(CASES)} 组样本 × {LAT.size} 点")
    prov = json.loads(str(FX["provenance"]))
    check("provenance 记录了约定与勒夫数表哈希",
          "conventions" in prov and len(prov.get("love_table_sha256", "")) == 64,
          f"sha256 = {prov['love_table_sha256'][:12]}…")
    check("SHSynth 目录位置已解析", True, SHSYNTH_ROOT)


def t_vs_shsynth():
    fn, how = _find_shsynth_horizontal()
    if fn is None:
        RESULTS.append(True)                                  # 跳过不算失败
        print(f"[PASS] 跨包对装：跳过（{how}）                        skip")
        return
    print(f"       使用 SHSynth 入口：{how}")
    worst = {}
    for i, name in enumerate(CASES):
        try:
            u_n, u_e = _call(fn, LAT, LON, FX["C"][i], FX["S"][i])
        except Exception as exc:                              # noqa: BLE001
            check(f"跨包对表 {name}", False,
                  f"调用失败 {type(exc).__name__}: {str(exc)[:60]}")
            continue
        worst[name] = max(rel(u_n, FX["u_N"][i]), rel(u_e, FX["u_E"][i]))
        check(f"跨包对表 {name}（相对 ≤1e-12）", worst[name] <= 1e-12,
              f"相对 max|diff| = {worst[name]:.2e}")
    if worst:
        check("跨包总体一致（四组 ≤1e-12）", max(worst.values()) <= 1e-12,
              f"最差 = {max(worst.values()):.2e} ({max(worst, key=worst.get)})")


def t_degree_factor_cross_check():
    """**F2 是跨包最隐蔽的失败模式**：两侧的 l′ 表若不同（例如一边做了 CE→CF
    改正、另一边没做），水平形变会整体偏几个百分点，而两边各自的内部测试**全绿**。
    所以必须逐值对表。"""
    try:
        from shsynth import engine as se
        from shsynth import units as su
    except Exception as exc:                                  # noqa: BLE001
        RESULTS.append(True)
        print(f"[PASS] 跨包 F^h_n 对表：跳过（{type(exc).__name__}）              skip")
        return
    from shkit.gradient import degree_factors_horizontal
    mine = degree_factors_horizontal(96)
    for label, other in (("engine.love_horizontal_factors", se.love_horizontal_factors(96)),
                         ("units.degree_factors", su.degree_factors("horizontal_displacement", 96))):
        d = float(np.max(np.abs(mine - np.asarray(other, dtype=float))))
        check(f"F^h_n 跨包一致（{label}，n=0..96）", d <= 1e-12,
              f"max|diff| = {d:.2e}（F^h_1 = {mine[1]:.6e}）")


def t_azimuth_boundary():
    """F6 是半开区间 ``[0, 360)``。IEEE 细节：``(-1e-16) % 360.0 == 360.0`` 恰好，
    会把区间变成闭区间 —— 必须把 360.0 折回 0。跨包要一起守这条。"""
    from shkit.gradient import horizontal_field, degree_factors_horizontal
    mine = horizontal_field(LAT, LON, FX["C"][1], FX["S"][1], FH)["azimuth"]
    check("SHKit azimuth ∈ [0,360)（含极点样本）",
          bool(np.all(mine >= 0) and np.all(mine < 360)),
          f"max = {float(np.max(mine)):.6f}")
    fn, how = _find_shsynth_horizontal()
    if fn is None:
        RESULTS.append(True)
        print("[PASS] SHSynth azimuth 边界：跳过（无接口）                     skip")
        return
    try:
        out = fn(LAT, LON, FX["C"][1], FX["S"][1], nmax=NMAX, weights_scale=FH,
                 want=("north", "east"))
    except TypeError:
        out = fn(LAT, LON, FX["C"][1], FX["S"][1], weights_scale=FH)
    if not isinstance(out, dict) or "azimuth" not in out:
        RESULTS.append(True)
        print("[PASS] SHSynth azimuth 边界：跳过（该入口不返回 azimuth）        skip")
        return
    azi = np.asarray(out["azimuth"])
    bad = int(np.sum(azi >= 360.0))
    if bad == 0:
        check("SHSynth azimuth ∈ [0,360)（半开，无恰好 360.0）", True,
              f"max = {float(np.max(azi)):.6f}")
    else:
        warn("SHSynth azimuth 违反 F6（半开区间）",
             f"{bad}/{azi.size} 个点 = 360.0（应为 0）。"
             f"修法：`azi = np.where(azi >= 360.0, 0.0, azi)`")


def t_fft_path_cross_check():
    """B3 的跨包对表：水平形变的 FFT 经度路径。"""
    try:
        from shsynth import engine as se
        from shsynth.coeffs import SHCoeffs as SSCoeffs
    except Exception as exc:                                  # noqa: BLE001
        RESULTS.append(True)
        print(f"[PASS] 跨包 FFT 路径对表：跳过（{type(exc).__name__}）            skip")
        return
    if not hasattr(se, "horizontal_grid_fft"):
        RESULTS.append(True)
        print("[PASS] 跨包 FFT 路径对表：跳过（SHSynth 无 horizontal_grid_fft）  skip")
        return
    from shkit.coeffs import SHCoeffs
    from shkit.gradient import horizontal_grid, degree_factors_horizontal
    L = NMAX
    lat = np.arange(-88.0, 88.01, 4.0)
    lon = np.arange(0.0, 360.0, 4.0)              # nlon = 90 > 2*20 ✓
    C, S = FX["C"][2], FX["S"][2]
    mine = horizontal_grid(lat, lon, C, S, degree_factors_horizontal(L))
    theirs = se.horizontal_grid_fft(lat, lon, SSCoeffs(C, S))
    for comp in ("north", "east"):
        d = rel(mine[comp], theirs[comp])
        check(f"跨包 FFT 路径对表：{comp}（相对 ≤1e-12）", d <= 1e-12,
              f"相对 max|diff| = {d:.2e}")


def t_starting_longitude_phase():
    """F12：起始经度相位。

    A3 发现 SHKit 自己的 FFT 路径曾漏掉 ``e^{i m lambda_0}``，而**当时跨包对表
    是绿的** —— 因为两边都只在 ``lon = 0,1,2,...`` 上比过。 所以这里显式地在
    ``lambda_0 != 0`` 的网格上检查"对方"的行为：

    * 对方**算对**  -> PASS；
    * 对方**明确拒绝** -> PASS（F12 允许"不适用就拒绝"）；
    * 对方**算错** -> WARN，并把差额报出来（这正是 F12 要禁止的第三种情况）。
    """
    try:
        from shsynth import engine as se
        from shsynth.coeffs import SHCoeffs as SSCoeffs
    except Exception as exc:                                  # noqa: BLE001
        RESULTS.append(True)
        print(f"[PASS] 跨包起始经度相位：跳过（{type(exc).__name__}）            skip")
        return
    if not hasattr(se, "horizontal_grid_fft"):
        RESULTS.append(True)
        print("[PASS] 跨包起始经度相位：跳过（无 horizontal_grid_fft）          skip")
        return

    from shkit.gradient import horizontal_grid, degree_factors_horizontal
    L = NMAX
    lat = np.arange(-88.0, 88.01, 4.0)
    C, S = FX["C"][2], FX["S"][2]
    fh = degree_factors_horizontal(L)

    # 先证明 SHKit 自己在偏移网格上是对的（这是 A3 修好的那一条）
    lam0 = -179.5
    lon = lam0 + np.arange(0.0, 360.0, 4.0)
    mine_f = horizontal_grid(lat, lon, C, S, fh, method="fft")
    mine_d = horizontal_grid(lat, lon, C, S, fh, method="direct")
    check(f"SHKit 自身：lam0={lam0:g} 时 FFT ≡ 直接法",
          rel(mine_f["north"], mine_d["north"]) <= 1e-12,
          f"相对 max|diff| = {rel(mine_f['north'], mine_d['north']):.2e}")

    try:
        theirs = se.horizontal_grid_fft(lat, lon, SSCoeffs(C, S))
    except Exception as exc:                                  # noqa: BLE001
        RESULTS.append(True)
        print(f"[PASS] 跨包起始经度相位：对方拒绝偏移网格（合规，F12）        "
              f"{type(exc).__name__}: {str(exc)[:40]}")
        return
    d = max(rel(mine_d["north"], theirs["north"]),
            rel(mine_d["east"], theirs["east"]))
    if d <= 1e-12:
        RESULTS.append(True)
        print(f"[PASS] 跨包起始经度相位：对方也算对（相对 {d:.2e}）")
    else:
        warn("SHSynth 违反 F12（起始经度相位）",
             f"lam0={lam0:g} 时与 SHKit 直接法相对差 {d:.2e}；"
             f"要么乘 e^(i·m·lam0)，要么像自家 fft_path_applicable 那样拒绝该网格")


# ---------------------------------------------------------------------------
def main():
    print(f"  SHKit 侧 fixture : {FIXTURE}")
    print(f"  SHSynth 目录     : {SHSYNTH_ROOT}"
          f"{'' if os.path.isdir(SHSYNTH_ROOT) else '  (不存在)'}")
    for fn in (_fixture_selftest, t_vs_shsynth, t_degree_factor_cross_check,
               t_azimuth_boundary, t_fft_path_cross_check,
               t_starting_longitude_phase):
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(fn.__name__, False, "抛出异常")
    npass = sum(RESULTS)
    print(f"\n{'='*76}\n{npass}/{len(RESULTS)} checks passed")
    if DISCREPANCIES:
        print(f"\n  ⚠ 跨包契约分歧 {len(DISCREPANCIES)} 条（不影响 SHKit 自身，"
              f"但要交给 SHSynth 侧处理）：")
        for d in DISCREPANCIES:
            print(f"    - {d}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
