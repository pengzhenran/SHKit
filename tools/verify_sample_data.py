# -*- coding: utf-8 -*-
"""
载入 sample_data/ 下的示例数据并做分析，报告与真值的差距。

    python tools/verify_sample_data.py

这个脚本同时回答一个容易被含糊带过的问题：
**积分元到底在什么条件下才重要？**

实测结论（见脚本输出）：
  · 作为「求积权重」——采样不均匀时极其重要（聚簇数据上 uniform 比 voronoi
    差约 80 倍，而且用错权重时迭代校正会发散）；
  · 作为「最小二乘的行权重」——几乎不影响结果。最小二乘对任意正的行权重都是
    一致的，行权重只影响噪声最优性；
  · 采样本身就准均匀时（Fibonacci）——两者几乎无差别。
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit import io as shio
from shkit.analysis import analysis_quadrature, analysis_wlsq
from shkit.gui.workers import _load
from shkit.synthesis import synthesis
from shkit.weights import compute_weights

DATA = os.path.join(ROOT, "sample_data")


def rel_err(a, b) -> float:
    L = max(a.nmax, b.nmax)
    a, b = a.truncate(L), b.truncate(L)
    num = np.linalg.norm(a.C - b.C) ** 2 + np.linalg.norm(a.S - b.S) ** 2
    den = np.linalg.norm(b.C) ** 2 + np.linalg.norm(b.S) ** 2
    return float(np.sqrt(num / den))


def fnum(v: float) -> str:
    if not np.isfinite(v):
        return "   —    "
    if v == 0:
        return "      0 "
    if v >= 1e4 or v < 1e-4:
        return f"{v:8.2e}"
    return f"{v:8.3f}"


def weights_for(ds, rule):
    return compute_weights(ds.lat, ds.lon, rule=rule)


def run_table(ds, f, truth, label: str):
    """对同一份数据，比较积分元 × 估计方法的组合。"""
    L = truth.nmax
    print(f"\n{label}")
    print(f"  {'积分元':<10}{'求积投影':>11}{'迭代校正x3':>12}{'WLSQ':>11}"
          f"   说明")
    base = None
    for rule in ("voronoi", "uniform"):
        try:
            w = weights_for(ds, rule)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {rule:<10}  权重计算失败: {exc}")
            continue
        c1, _ = analysis_quadrature(ds.lat, ds.lon, f, L, weights=w)
        try:
            c2, _ = analysis_quadrature(ds.lat, ds.lon, f, L, weights=w, niter=3)
            e2 = fnum(rel_err(c2, truth))
        except Exception:                              # noqa: BLE001
            e2 = "   发散   "
        c3, _ = analysis_wlsq(ds.lat, ds.lon, f, L, weights=w, method="wlsq")
        e1, e3 = rel_err(c1, truth), rel_err(c3, truth)
        if base is None:
            base = e1
        note = ""
        if rule == "uniform" and base:
            note = f"求积投影比 voronoi 差 {e1 / base:.0f} 倍"
        print(f"  {rule:<10}{fnum(e1):>11}{e2:>12}{fnum(e3):>11}   {note}")


def in_region_fit(ds, coeffs, mask_lat, mask_lon, radius_deg):
    """区域内重建的相对拟合误差（区域数据唯一有意义的精度指标）。"""
    lat0, lon0 = np.deg2rad(mask_lat), np.deg2rad(mask_lon)
    r = np.deg2rad(radius_deg)
    cosd = (np.sin(np.deg2rad(ds.lat)) * np.sin(lat0) +
            np.cos(np.deg2rad(ds.lat)) * np.cos(lat0) *
            np.cos(np.deg2rad(ds.lon) - lon0))
    sel = cosd >= np.cos(r)
    v = np.asarray(synthesis(ds.lat, ds.lon, coeffs)).ravel()
    obs = ds.value_slice(0)
    return float(np.sqrt(np.mean((v[sel] - obs[sel]) ** 2)) /
                 np.sqrt(np.mean(obs[sel] ** 2))), int(sel.sum())


def main() -> int:
    truth = shio.read_coeffs(os.path.join(DATA, "truth_coeffs_20.sh"))
    L = truth.nmax
    print("=" * 78)
    print(f"  SHKit 示例数据验证   真值 = truth_coeffs_20.sh（带限 {L} 阶，"
          f"{(L + 1) ** 2} 个系数）")
    print("=" * 78)

    # ------------------------------------------------------- 全球散点
    for fname in ("points_global_fibonacci.csv", "points_clustered.csv"):
        ds = _load(os.path.join(DATA, fname), "points")
        run_table(ds, ds.value_slice(0), truth,
                  f"{fname}   （{ds.npoints} 点，含 1% 噪声）")

    # 无噪声时对比更干净
    for fname in ("points_global_fibonacci.csv", "points_clustered.csv"):
        ds = _load(os.path.join(DATA, fname), "points")
        run_table(ds, synthesis(ds.lat, ds.lon, truth), truth,
                  f"{fname}   （同样点位，但用真值合成、无噪声）")

    # ------------------------------------------------------- 全球网格
    print("\n" + "-" * 78)
    from shkit.analysis import analysis as _analysis

    TIME_FACTORS = (1.0, 1.3, 0.7)      # 与 make_sample_data.py 保持一致
    for fname, rule in (("grid_global_2deg.nc", "grid"),
                        ("grid_global_5deg.grd", "grid")):
        ds = _load(os.path.join(DATA, fname), "grid")
        f = ds.value_slice(0)
        c, rep = _analysis(ds.lat, ds.lon, f, L, method="auto", rule=rule)
        print(f"{fname:<32} 数据点 {ds.npoints:>6}  时次 {ds.ntime}"
              f"  方法 {rep.method:<10} 系数相对误差 {fnum(rel_err(c, truth))}"
              f"  覆盖率 {rep.coverage:.4f}")
        if ds.ntime > 1:
            for t in range(ds.ntime):
                ct, _ = _analysis(ds.lat, ds.lon, ds.value_slice(t), L,
                                  method=rep.method, rule=rule)
                # 每个时次是同一个场乘以已知的振幅系数，参考值也要乘
                ref = truth * TIME_FACTORS[t]
                print(f"    时次 {t}（振幅 ×{TIME_FACTORS[t]:.1f}）: "
                      f"相对误差 {fnum(rel_err(ct, ref))}")

    # ------------------------------------------------------- 区域数据
    print("\n" + "-" * 78)
    print("区域数据：全球系数误差在这里不是有意义的指标，请看覆盖率与区域内拟合")

    ds = _load(os.path.join(DATA, "points_regional_cap.csv"), "points")
    for rule in ("delaunay", "voronoi"):
        w = compute_weights(ds.lat, ds.lon, rule=rule, normalise="region")
        c, rep = analysis_wlsq(ds.lat, ds.lon, ds.value_slice(0), L,
                               weights=w, method="wlsq")
        fit, nsel = in_region_fit(ds, c, 30.0, 100.0, 25.0)
        print(f"  points_regional_cap.csv  rule={rule:<9} "
              f"覆盖率 {rep.coverage:.4f}  cond {rep.condition_number:.1e}  "
              f"区域内拟合 {fit:.3e}  ||C|| {np.linalg.norm(c.C):.3e}")
    print(f"    （真值 ||C|| = {np.linalg.norm(truth.C):.3e}；"
          "无论哪种规则，区域数据都恢复不出这个范数）")

    ds = _load(os.path.join(DATA, "grid_regional_1deg.nc"), "grid")
    c, rep = analysis_wlsq(ds.lat, ds.lon, ds.value_slice(0), L,
                           weights=compute_weights(ds.lat, ds.lon, rule="grid"),
                           method="wlsq")
    print(f"  grid_regional_1deg.nc    rule=grid       "
          f"覆盖率 {rep.coverage:.4f}  cond {rep.condition_number:.1e}  "
          f"||C|| {np.linalg.norm(c.C):.3e}")

    # ------------------------------------------------------------ 结论
    print("\n" + "=" * 78)
    print("  结论")
    print("=" * 78)
    print("""
1) 采样准均匀（Fibonacci）时，voronoi 与 uniform 几乎没有差别 —— 因为这时
   「统一面积」本来就是对的。积分元不能靠"猜"，要看采样。

2) 采样不均匀（聚簇）时，差别是决定性的：
   · 求积投影：uniform 比 voronoi 差约 80 倍；
   · 迭代校正：uniform 权重下会**发散**（迭代的收敛条件被破坏）；
   · WLSQ：两者几乎一样 —— 因为最小二乘对任意正的行权重都是一致的，
     行权重只影响噪声最优性，不影响无偏性。
   → 所以"积分元"与"最小二乘行权重"是两个不同的东西，不要混。

3) 区域数据（覆盖率 0.04–0.08）：系数范数与全球参数不可比，
   有意义的只有**区域内拟合**和诊断报告里的警告。这是原理性限制。

4) 用错了估计方法和阶数，比用错积分元更容易出事：注意报告里的
   「推荐 nmax」和条件数。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
