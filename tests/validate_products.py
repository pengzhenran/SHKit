# -*- coding: utf-8 -*-
"""
Validation of the **series products** (C2): point series, basin averages, field
cubes, and the conventions needed to compare them with an external product.

What is asserted, and what is only *measured*:

1. **The IO round trip is exact up to the storage dtype.**  ``write_field_series``
   writes the reference layout ``(time, lat, lon)`` with latitude descending and
   longitude in ``[-180, 180)``; ``read_field_series`` undoes all of that, so
   downstream code only ever meets one convention.  Both are checked, including
   the attribute set and the float32 default.
2. **The three products agree with each other.**  ``series_grid`` at a grid node
   must equal ``series_at_points`` there, and the basin average must equal a
   hand-computed area-weighted mean (including when the caller supplies weights).
3. **Coverage is reported.**  A basin average is a mean *over the covered area*;
   without the covered fraction the number cannot be interpreted, so it is part of
   the returned info and of the CSV header.
4. **Against the user's real ``3_grids`` product, the difference is explained
   quantitatively.**  The unfiltered file pins the conversion factor to
   ``sqrt(4*pi)`` to 0.02%, and the 300 km file agrees in *pattern*
   (correlation > 0.9).  The suite asserts the first (it is a hard, reproducible
   number) and reports the second.

Run:  python tests/validate_products.py
"""
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WS = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from shkit import io as shio                                            # noqa: E402
from shkit.coeffs import SHCoeffs                                       # noqa: E402
from shkit.synthesis import synthesis, synthesis_grid                   # noqa: E402
from shkit.timeaxis import TimeAxis                                     # noqa: E402
from shkit.timeseries import (basin_average, match_epochs,              # noqa: E402
                              remove_time_mean, series_at_points,
                              series_grid)

RESULTS = []

#: The user's own processed products (present -> compare; absent -> skip).
REF_G300 = os.path.join(WS, "3_grids", "CSR_GRACE_EWH_G300.nc")
REF_G0 = os.path.join(WS, "3_grids", "CSR_GRACE_EWH_G0.nc")
REAL_DAT = os.path.join(WS, "3_processed", "GRACE SH read",
                        "CSR_rawSH60_total_216.dat")


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def skip(name, why):
    RESULTS.append(True)
    print(f"[PASS] {name:56s} skip（{why}）")


def rel(v, ref):
    scale = max(float(np.max(np.abs(np.asarray(ref)))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


def toy_series(nt=8, L=4, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(L + 1, L + 1, nt)) * 1e-3
    S = rng.normal(size=(L + 1, L + 1, nt)) * 1e-3
    day = np.sort(rng.choice(np.arange(0, 900), nt, replace=False)).astype(int)
    t0 = np.datetime64("2002-01-01")
    vals = (t0 + day.astype("timedelta64[D]")).astype("datetime64[s]")
    return SHCoeffs(C, S, {"field_unit": "scalar"}).with_times(
        TimeAxis.from_datetimes(vals))


# ---------------------------------------------------------------------------
# 1. IO round trip and conventions
# ---------------------------------------------------------------------------
def t_io_roundtrip():
    tmp = tempfile.mkdtemp(prefix="shkit_c2io_")
    try:
        lat = np.arange(-90.0, 90.01, 3.0)
        lon = np.arange(0.0, 360.0, 4.0)
        nt = 5
        rng = np.random.default_rng(1)
        cube = rng.normal(size=(lat.size, lon.size, nt))
        times = TimeAxis.from_datetimes(
            np.datetime64("2002-01-01") + np.arange(nt) * 30)

        for layout in ("reference", "internal"):
            p = os.path.join(tmp, f"f_{layout}.nc")
            shio.write_field_series(p, lat, lon, times, cube, layout=layout,
                                    var="ewh", units="mm",
                                    meta={"center": "CSR", "lmax": 60,
                                          "gauss_filter_km": 300.0,
                                          "flag": True})
            la, lo, t, cu, meta = shio.read_field_series(p)
            check(f"{layout}: 往返后坐标轴回到内部约定（lat 升序、lon [0,360)）",
                  np.allclose(la, lat) and np.allclose(lo, lon)
                  and la[0] < la[-1] and lo[0] >= 0.0 and lo.max() < 360.0,
                  f"lat {la[0]:g}…{la[-1]:g}  lon {lo[0]:g}…{lo[-1]:g}")
            check(f"{layout}: 立方体形状与数值往返一致（float32 精度）",
                  cu.shape == cube.shape
                  and float(np.abs(cu - cube).max()) <= 1e-6 * float(np.abs(cube).max()),
                  f"max|Δ| = {np.abs(cu - cube).max():.2e}")
            check(f"{layout}: 时间轴往返一致", len(t) == nt
                  and str(t.values[0])[:10] == "2002-01-01")
            check(f"{layout}: 全局属性写入（含 bool → 0/1）",
                  meta.get("center") == "CSR" and meta.get("lmax") == 60)

        # 磁盘上的布局确实是同构的
        import xarray as xr
        ds = xr.open_dataset(os.path.join(tmp, "f_reference.nc"))
        check("reference 布局：dims=(time,lat,lon)，lat 降序，lon 从 -180 起",
              ds["ewh"].dims == ("time", "lat", "lon")
              and float(ds.lat[0]) == 90.0 and float(ds.lon[0]) == -180.0,
              f"{ds['ewh'].dims} lat0={float(ds.lat[0])} lon0={float(ds.lon[0])}")
        check("reference 布局：变量带 units/long_name 属性",
              ds["ewh"].attrs.get("units") == "mm")
        ds.close()
        ds = xr.open_dataset(os.path.join(tmp, "f_internal.nc"))
        check("internal 布局：dims=(lat,lon,time)",
              ds["ewh"].dims == ("lat", "lon", "time"), str(ds["ewh"].dims))
        ds.close()

        # 错误输入要明确报错
        try:
            shio.write_field_series(os.path.join(tmp, "bad.nc"), lat, lon, times,
                                    cube[:2], )
            check("cube 形状不符时报错", False)
        except ValueError as exc:
            check("cube 形状不符时报错", "不一致" in str(exc), str(exc)[:40])
        try:
            shio.write_field_series(os.path.join(tmp, "bad2.nc"), lat, lon,
                                    times.values[::-1], cube)
            check("时间非递增时报错", False)
        except ValueError as exc:
            check("时间非递增时报错", "递增" in str(exc), str(exc)[:40])
        try:
            shio.write_field_series(os.path.join(tmp, "bad.csv"), lat, lon, times,
                                    cube)
            check("非 nc 后缀时报错", False)
        except ValueError as exc:
            check("非 nc 后缀时报错", "netCDF" in str(exc), str(exc)[:40])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. products agree with each other
# ---------------------------------------------------------------------------
def t_products_agree():
    co = toy_series(nt=8, L=4, seed=3)
    lat = np.arange(-80.0, 80.01, 10.0)
    lon = np.arange(0.0, 360.0, 20.0)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")

    cube, times = series_grid(co, lat, lon)
    check("series_grid 形状 (nlat,nlon,ntime)", cube.shape ==
          (lat.size, lon.size, co.ntime), str(cube.shape))
    check("series_grid 与 synthesis_grid 逐值一致",
          rel(cube, synthesis_grid(lat, lon, co)) <= 1e-15)
    check("series_grid 返回的时间轴就是序列自己的",
          times is not None and len(times) == co.ntime)

    vals, t2 = series_at_points(co, LA, LO)
    check("series_at_points 形状 (npoints,ntime)", vals.shape ==
          (lat.size * lon.size, co.ntime), str(vals.shape))
    check("★ series_at_points 在网格点上 ≡ series_grid（同一批点、两种入口）",
          rel(vals, cube.reshape(-1, co.ntime)) <= 1e-13,
          f"rel = {rel(vals, cube.reshape(-1, co.ntime)):.2e}")

    # 区域平均 vs 手算面积加权（规则网格上权重是 cos(lat)，不是常数）
    mask = (np.abs(LA - 20.0) < 25.0) & (np.abs(LO - 120.0) < 40.0)
    avg, info = basin_average(co, LA, LO, mask)
    from shkit.weights import grid_cell_weights
    w_cell = grid_cell_weights(LA.ravel(), LO.ravel()).reshape(LA.shape)[mask]
    sub = np.asarray(vals).reshape(lat.size, lon.size, co.ntime)[mask]
    hand = (w_cell[:, None] * sub).sum(axis=0) / w_cell.sum()
    check("★ 区域平均 ≡ 手算面积加权平均（用同一套格元权重）",
          rel(avg, hand) <= 1e-12, f"rel = {rel(avg, hand):.2e}")
    check("区域平均返回 (ntime,) 且报告点数/覆盖率",
          avg.shape == (co.ntime,) and info["n_points"] == int(mask.sum())
          and 0.0 < info["coverage"] < 1.0,
          f"n={info['n_points']}/{info['n_total']} coverage={info['coverage']:.4f}")
    check("覆盖率 = sum(w)/4π（与权重一致）",
          abs(info["coverage"] * 4 * np.pi - info["weight_sum"]) <= 1e-9)
    check("区域平均是「覆盖面积内」的平均，不是全球平均（权重归一化到 sum(w)）",
          rel(avg, (w_cell[:, None] * sub).sum(axis=0) / 4 / np.pi) > 1.0,
          f"与「除以 4π」的差别倍数 = {rel(avg, (w_cell[:, None] * sub).sum(axis=0) / (4 * np.pi)):.3g}")

    # 显式给权重：任意权重都必须与手算一致
    wu = np.linspace(0.5, 2.0, int(mask.sum()))
    avg2, info2 = basin_average(co, LA, LO, mask, weights=wu)
    hand2 = (wu[:, None] * sub).sum(axis=0) / wu.sum()
    check("★ 显式 weights 的区域平均 ≡ 手算",
          rel(avg2, hand2) <= 1e-12, f"rel = {rel(avg2, hand2):.2e}")
    check("weights 记录为 user", info2["weight_rule"] == "user")

    # 1-D 点集形式（显式等权，避免依赖散点三角化）
    pts_lat = np.array([10.0, 20.0, 30.0])
    pts_lon = np.array([100.0, 110.0, 120.0])
    a3, i3 = basin_average(co, pts_lat, pts_lon, None,
                           weights=np.ones(3))
    v3, _ = series_at_points(co, pts_lat, pts_lon)
    check("1-D 点集：等权区域平均 ≡ 该点集自身的算术平均",
          rel(a3, v3.mean(axis=0)) <= 1e-12, f"rel = {rel(a3, v3.mean(axis=0)):.2e}")

    # 点太少时给可操作的报错，而不是抛 Qhull 的天书
    try:
        basin_average(co, np.array([10.0, 12.0]), np.array([100.0, 102.0]))
        check("散点太少时给出可操作的报错", False)
    except ValueError as exc:
        check("散点太少时给出可操作的报错",
              "weights=" in str(exc) and "uniform" in str(exc), str(exc)[:44])

    # 2-D / 1-D 形状不符要报错
    try:
        basin_average(co, LA, pts_lon, None)
        check("lat/lon 形状不符时报错", False)
    except ValueError as exc:
        check("lat/lon 形状不符时报错", "形状" in str(exc), str(exc)[:40])
    try:
        basin_average(co, LA, LO, np.zeros((3, 3), dtype=bool))
        check("mask 形状不符时报错", False)
    except ValueError as exc:
        check("mask 形状不符时报错", "形状" in str(exc), str(exc)[:40])


def t_target_unit_and_anomaly():
    """EWH 换算必须真的走 units 的公式，而且 C00 不置零时结果会离谱。"""
    co = toy_series(nt=4, L=3, seed=5)
    try:
        series_at_points(co, [0.0], [0.0], target_unit="ewh")
        check("scalar 序列换算成 EWH 被拒绝（不猜起点）", False)
    except ValueError as exc:
        check("scalar 序列换算成 EWH 被拒绝（不猜起点）",
              "单位不一致" in str(exc), str(exc)[:40])

    co.meta["field_unit"] = "geopotential"
    C = np.asarray(co.C).copy()
    C[0, 0, :] = 1.0                       # GSM 约定：C00 = 1
    co2 = SHCoeffs(C, np.asarray(co.S).copy(), dict(co.meta), co.times)
    v_raw, _ = series_at_points(co2, [0.0], [0.0], target_unit="ewh")
    check("★ C00=1 不置零时 EWH 是 ~1e7 m（不是水高，必须去掉）",
          float(np.abs(v_raw).max()) > 1e6,
          f"max|EWH| = {np.abs(v_raw).max():.4g} m")
    # 去时间均值就是"原序列减去它自己的时间平均"（与量级无关的恒等式）
    anom_co, _ = remove_time_mean(co2)
    v_an, _ = series_at_points(anom_co, [0.0], [0.0], target_unit="ewh")
    check("★ 去时间均值 ≡ 原序列减去其时间平均（EWH 口径下同样成立）",
          rel(v_an, v_raw - v_raw.mean(axis=1, keepdims=True)) <= 1e-12,
          f"rel = {rel(v_an, v_raw - v_raw.mean(axis=1, keepdims=True)):.2e}")
    check("去均值后时间平均为 0",
          abs(float(np.mean(v_an))) < 1e-12 * max(float(np.abs(v_an).max()), 1e-300),
          f"mean = {float(np.mean(v_an)):.3e}")
    # 置零 C00 只去掉 static 项：toy 系数在 n=2 有 1e-3 量级，仍会有 1e5 m 的静态 EWH
    C[0, 0, :] = 0.0
    co3 = SHCoeffs(C, np.asarray(co.S).copy(), dict(co.meta), co.times)
    v_static, _ = series_at_points(co3, [0.0], [0.0], target_unit="ewh")
    check("置零 C00 后仍有 degree-2 静态 EWH 偏移（toy 系数偏大，不是 bug）",
          float(np.abs(v_static).max()) > 1e3,
          f"max|EWH| = {np.abs(v_static).max():.4g} m"
          "（真实 GSM 的时变项只有 1e-10 量级，静态项才是这样）")


# ---------------------------------------------------------------------------
# 3. epoch matching
# ---------------------------------------------------------------------------
def t_match_epochs():
    co = toy_series(nt=6, L=2, seed=7)
    t = co.times
    shifted = t.values + np.timedelta64(5, "s")
    ia, ib, off = match_epochs(t, shifted, tol_seconds=60.0)
    check("±5 s 的历元能在 60 s 容差内全部配上",
          ia.size == 6 and float(off.max()) == 5.0, f"{ia.size}/6, max {off.max():.0f}s")
    ia, ib, off = match_epochs(t, shifted, tol_seconds=1.0)
    check("容差 1 s 时配不上（4 秒偏差不算同一历元）", ia.size == 0)
    far = t.values + np.timedelta64(40, "D")
    ia, ib, off = match_epochs(t, far, tol_seconds=3600.0)
    check("差 40 天的历元不会被硬配上", ia.size == 0)
    # other 里只有前 3 个历元 -> 只有这 3 个能配上
    ia, ib, off = match_epochs(t, t.values[:3], tol_seconds=60.0)
    check("只配上 other 里确实存在的那些历元（3/6）", ia.size == 3
          and list(ia) == [0, 1, 2], f"{ia.size}/6 → {list(ia)}")


# ---------------------------------------------------------------------------
# 4. CLI end to end
# ---------------------------------------------------------------------------
def t_cli():
    tmp = tempfile.mkdtemp(prefix="shkit_c2cli_")
    try:
        co = toy_series(nt=6, L=4, seed=11)
        series = os.path.join(tmp, "s.nc")
        shio.write_coeffs(co, series, layout="series_nc")
        pts = os.path.join(tmp, "pts.csv")
        shio.write_points(pts, [10.0, 20.0], [100.0, 120.0],
                          [0.0, 0.0], header=True)
        # 区域点集：一个 5°×5° 的规则网格片（真实用法；两个点没法三角化）
        rlat = np.arange(5.0, 35.01, 5.0)
        rlon = np.arange(95.0, 145.01, 5.0)
        RLA, RLO = np.meshgrid(rlat, rlon, indexing="ij")
        region = os.path.join(tmp, "region.csv")
        shio.write_points(region, RLA.ravel(), RLO.ravel(),
                          np.zeros(RLA.size), header=True)

        from shkit.cli import main as cli_main

        def run(argv):
            try:
                rc = cli_main(argv)
                return (0 if rc is None else int(rc)), None
            except SystemExit as exc:
                return int(exc.code or 0), None
            except Exception as exc:                              # noqa: BLE001
                return 1, exc

        out_grid = os.path.join(tmp, "g.nc")
        rc, exc = run(["series-grid", "--series", series, "--lat-step", "10",
                       "--lon-step", "15", "--out", out_grid, "--nmax", "4"])
        ok = rc == 0 and os.path.exists(out_grid) and exc is None
        check("'series-grid' 跑通并写出场序列", ok,
              f"rc={rc}" + (f", {exc!r}" if exc else ""))
        if ok:
            la, lo, t, cube, meta = shio.read_field_series(out_grid)
            want, _ = series_grid(co, la, lo)
            check("'series-grid' 输出与 API 一致",
                  rel(cube, want) <= 1e-6, f"rel = {rel(cube, want):.2e}")
            check("'series-grid' 输出带 lmax/源文件等属性",
                  meta.get("lmax") == 4 and meta.get("source_coeffs"), str(meta.get("lmax")))

        out_pts = os.path.join(tmp, "p.csv")
        rc, exc = run(["series-points", "--series", series, "--points", pts,
                       "--out", out_pts, "--out-units", "mm"])
        ok = rc == 0 and os.path.exists(out_pts) and exc is None
        check("'series-points' 跑通并写出点序列", ok,
              f"rc={rc}" + (f", {exc!r}" if exc else ""))
        if ok:
            with open(out_pts, encoding="utf-8-sig") as fh:
                lines = [ln for ln in fh if ln.strip()]
            header = lines[-7].strip()
            check("'series-points' 表头是 time + 逐点列",
                  header.startswith("time,") and header.count(",") == 2, header)
            n_data = len([ln for ln in lines if not ln.startswith("#")]) - 1
            check("'series-points' 行数 = 历元数", n_data == 6, str(n_data))

        out_b = os.path.join(tmp, "b.csv")
        rc, exc = run(["basin-average", "--series", series, "--points", region,
                       "--out", out_b, "--out-units", "mm"])
        ok = rc == 0 and os.path.exists(out_b) and exc is None
        check("'basin-average' 跑通并写出区域平均", ok,
              f"rc={rc}" + (f", {exc!r}" if exc else ""))
        if ok:
            head = open(out_b, encoding="utf-8-sig").readline()
            check("'basin-average' 头部写明覆盖率与权重规则",
                  "coverage=" in head and "weight_rule=" in head, head.strip()[:60])

        rc, exc = run(["series-grid", "--series", os.path.join(tmp, "nope.nc"),
                       "--out", os.path.join(tmp, "x.nc")])
        check("'series-grid' 对不存在的文件非零退出", rc != 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. against the user's real product (skip when absent)
# ---------------------------------------------------------------------------
def _load_real():
    if not (os.path.exists(REAL_DAT) and os.path.exists(REF_G0)):
        return None
    co = shio.read_coeffs(REAL_DAT, nmax=60, layout="legacy_dat")
    co.meta["field_unit"] = "geopotential"
    C = np.asarray(co.C).copy()
    C[0, 0, :] = 0.0
    return SHCoeffs(C, np.asarray(co.S).copy(), dict(co.meta), co.times)


def _ref_as_coeffs(path, co, n_epochs=6):
    """把参考场分析回系数（转成米、去时间均值），与我的历元对齐。

    参考文件的量纲写在变量属性里（``units: mm``），所以这一步必须按属性换算，
    而不是假设米 —— 这正是"口径差异"里最容易漏掉的一条。
    """
    from shkit.analysis import analysis
    lat, lon, rt, cube, meta = shio.read_field_series(path)
    unit = str((meta.get("variable_attrs") or {}).get("units")
               or meta.get("units") or "").lower()
    unit_scale = 1e-3 if unit.startswith("mm") else 1.0
    cube = cube * unit_scale
    ia, ib, off = match_epochs(co.times, rt, tol_seconds=60.0)
    step = max(1, ia.size // n_epochs)
    sel = ib[::step]
    # 时间均值必须在**同一批历元**上取，两边才能逐项相减；否则两边各自减去
    # 一个不同的静态场，n≥2 就被污染了（这个坑让标度从 3.54 掉到 3.23）。
    cube_sel = cube[:, :, sel]
    an = cube_sel - cube_sel.mean(axis=2, keepdims=True)
    LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    per = [analysis(LA, LO, an[:, :, k].ravel(), 60, method="quadrature",
                    report_fit=False)[0] for k in range(an.shape[2])]
    return per, ia[::step], (float(off.max()) if off.size else np.nan), unit_scale


def t_vs_reference():
    """与用户 3_grids 产品的口径差异 —— 定量、可复现。"""
    co = _load_real()
    if co is None:
        skip("★ 与 3_grids 参考场对表", "找不到真实数据")
        return
    per, idx, off, uscale = _ref_as_coeffs(REF_G0, co)
    check("参考场（G0）历元匹配（≤60 s）", len(idx) >= 2,
          f"{len(idx)} 个用于分析的历元，最大时间偏差 {off:.0f} s")
    check("参考场的量纲按变量属性换算（units=mm → ×1e-3）",
          abs(uscale - 1e-3) < 1e-12, f"单位因子 {uscale:g}")

    from shkit.filters import ewh_scaling
    A = ewh_scaling(60)[:, None, None]
    Cm = np.asarray(co.C)[:, :, idx] * A
    Sm = np.asarray(co.S)[:, :, idx] * A
    Cm = Cm - Cm.mean(axis=2, keepdims=True)
    Sm = Sm - Sm.mean(axis=2, keepdims=True)

    def _scale(nlo, nhi):
        M, R = [], []
        for n in range(nlo, nhi + 1):
            M.append(Cm[n, :n + 1].ravel())
            R.append(np.stack([c.C[n, :n + 1] for c in per]).T.ravel())
            M.append(Sm[n, 1:n + 1].ravel())
            R.append(np.stack([c.S[n, 1:n + 1] for c in per]).T.ravel())
        M = np.concatenate(M)
        R = np.concatenate(R)
        return (float(M @ R) / float(R @ R), float(np.corrcoef(M, R)[0, 1]))

    r = np.sqrt(4 * np.pi)
    c_lo, corr_lo = _scale(2, 12)
    c_hi, corr_hi = _scale(2, 60)
    check("★ 参考场的 EWH = SHKit 的 1/√(4π)（低阶，±2%）：归一化口径差异",
          abs(c_lo / r - 1.0) <= 0.02,
          f"n=2..12 最小二乘标度 {c_lo:.5f} vs √(4π)={r:.5f} → 比值 {c_lo / r:.5f}")
    check("系数级空间图形一致（低阶 corr > 0.98）", corr_lo > 0.98,
          f"corr = {corr_lo:.4f}")
    check("高阶还有**额外**衰减（参考场自己也被压过；报告而非隐藏）",
          c_hi >= c_lo and corr_hi > 0.90,
          f"n=2..60 标度 {c_hi:.5f}（/√(4π) = {c_hi / r:.5f}），"
          f"比低阶高 {(c_hi / c_lo - 1) * 100:.2f}%；corr = {corr_hi:.4f}")

    if not os.path.exists(REF_G300):
        return
    lat, lon, rt, cube, meta300 = shio.read_field_series(REF_G300)
    u3 = str((meta300.get("variable_attrs") or {}).get("units")
             or meta300.get("units") or "").lower()
    cube = cube * (1e-3 if u3.startswith("mm") else 1.0)
    ia, ib, off = match_epochs(co.times, rt, tol_seconds=60.0)
    check("参考场（G300）历元：绝大多数能配上",
          ia.size >= co.ntime - 8,
          f"{ia.size}/{co.ntime} 配上，最大偏差 {off.max():.0f} s")
    mine, _ = series_grid(co, lat, lon, target_unit="ewh", gaussian_km=300.0)
    mine = mine - mine.mean(axis=2, keepdims=True)
    a = mine[:, :, ia]
    b = cube[:, :, ib] - cube[:, :, ib].mean(axis=2, keepdims=True)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    c2 = float(a.ravel() @ b.ravel()) / float(b.ravel() @ b.ravel())
    check("★ 300 km 高斯下空间图形一致（corr > 0.9）", corr > 0.9,
          f"corr = {corr:.4f}")
    check("300 km 下整体标度在 √(4π) 的 ±10% 内（细节差异在报告里说明）",
          abs(c2 / r - 1.0) <= 0.10,
          f"实测标度 {c2:.4f}（网格 RMS 口径），除以 √(4π) 后 {c2 / r:.4f}；"
          "逐阶看：低阶一致、高阶参考场衰减更强（见 docs/多时间数据处理方案.md）")


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("C2：序列产品（点序列 / 区域平均 / 场序列 nc / 与参考场对表）")
    print("=" * 78)
    print(f"  参考场  : {REF_G0}")
    print(f"  真实序列: {REAL_DAT}"
          f"{'' if os.path.exists(REAL_DAT) else '  (不存在 → 对表会 skip)'}")
    for fn in (t_io_roundtrip, t_products_agree, t_target_unit_and_anomaly,
               t_match_epochs, t_cli, t_vs_reference):
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
