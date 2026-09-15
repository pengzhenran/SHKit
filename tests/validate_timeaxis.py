# -*- coding: utf-8 -*-
"""
Validation of shkit.timeaxis and the ``SHCoeffs.times`` plumbing (A0).

The interesting parts are not the accessors -- they are:

1. **the frozen legacy decimal-year convention**, verified against the *real*
   ``CSR_rawSH60_total_216_TimeInfo.dat`` (200/203 epochs exact to 1e-6), with the
   two counter-examples that must keep failing (no ``+1`` day -> 0/203; leap year
   divided by 365 -> 146/203);
2. **no invented dates**: an axis without dates must refuse date-dependent work
   instead of pretending epochs are evenly spaced;
3. **`times` cannot be silently broadcast**: combining two sets with different
   axes raises;
4. **backwards compatibility**: objects without ``times`` behave bitwise as before.

Run:  python tests/validate_timeaxis.py
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit.coeffs import SHCoeffs                              # noqa: E402
from shkit.filters import apply_gaussian                       # noqa: E402
from shkit import io as shio                                   # noqa: E402
from shkit import units                                        # noqa: E402
from shkit.timeaxis import (TimeAxis, attr_ci, date_of_decimal_year,  # noqa: E402
                            decimal_year_of_date, parse_grace_filename,
                            time_axis_from_meta)

RESULTS = []

#: Real GRACE series, if it is still next to this checkout.
GRACE_DIR = os.path.join(os.path.dirname(ROOT), "2_unzipped", "1_CSR", "01RL06",
                         "01deg60")
LEGACY_TIMEINFO = os.path.join(os.path.dirname(ROOT), "3_processed",
                               "GRACE SH read",
                               "CSR_rawSH60_total_216_TimeInfo.dat")


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def _grace_files():
    if not os.path.isdir(GRACE_DIR):
        return []
    return sorted(os.path.join(GRACE_DIR, f)
                  for f in os.listdir(GRACE_DIR) if f.endswith(".gfc"))


def _legacy_mids():
    if not os.path.exists(LEGACY_TIMEINFO):
        return None
    return np.loadtxt(LEGACY_TIMEINFO)[:, 3]


def rel(v, ref):
    scale = max(float(np.max(np.abs(np.asarray(ref)))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


# ---------------------------------------------------------------------------
def t_filename_parsing():
    p = parse_grace_filename("GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc")
    check("文件名解析：起止积日", p["start"] == "2002095" and p["end"] == "2002120",
          f"{p['start']} .. {p['end']}")
    check("文件名解析：起止日期（结束日是**含**的）",
          str(p["start_date"]) == "2002-04-05" and str(p["end_date"]) == "2002-04-30",
          f"{p['start_date']} .. {p['end_date']}")
    check("文件名解析：中点 = (start + end + 1)/2 = 04-18",
          str(p["mid_date"]) == "2002-04-18", str(p["mid_date"]))
    check("文件名解析：十进制年对上 legacy 首行",
          abs(p["decimal_year"] - 2002.293151) < 1e-6,
          f"{p['decimal_year']:.6f}")
    check("文件名解析：任务/中心/阶型/RL",
          (p["mission"], p["center"], p["degtype"], p["rl"])
          == ("GRAC", "UTCSR", "BA01", "0600"),
          f"{p['mission']}/{p['center']}/{p['degtype']}/{p['rl']}")
    # GRACE-FO 96 阶
    q = parse_grace_filename("GSM-2_2019020-2019048_GRFO_UTCSR_BB01_0603.gfc")
    check("文件名解析：GRFO / BB01 / 0603",
          (q["mission"], q["degtype"], q["rl"]) == ("GRFO", "BB01", "0603"),
          f"{q['mission']}/{q['degtype']}/{q['rl']}")
    # 跨年历元：结束日期在次年
    r = parse_grace_filename("GSM-2_2016346-2017006_GRAC_UTCSR_BA01_0600.gfc")
    check("文件名解析：跨年历元不报错且给出日期",
          str(r["start_date"]) == "2016-12-11" and str(r["end_date"]) == "2017-01-06",
          f"{r['start_date']} .. {r['end_date']}")
    for bad in ("not_a_grace_file.gfc", "GSM-2_2002095_GRAC.gfc"):
        try:
            parse_grace_filename(bad)
            check(f"无法解析应报错: {bad}", False, "没有报错")
        except ValueError:
            check(f"无法解析应报错: {bad}", True, "ValueError")


def t_decimal_year_roundtrip():
    """十进制年 ↔ 日期往返；含闰年与年界。"""
    import datetime as dt
    ok = True
    worst = 0.0
    for d in (dt.date(2002, 4, 18), dt.date(2004, 12, 31), dt.date(2016, 2, 29),
              dt.date(2017, 1, 6), dt.date(1999, 1, 1)):
        back = date_of_decimal_year(decimal_year_of_date(d))
        worst = max(worst, abs((back - d).days))
        ok &= (back == d)
    check("十进制年 ↔ 日期往返（含闰年 2/29、年界）", ok, f"最大偏差 {worst:.0f} 天")
    check("闰年用 366 做分母（2004-12-31 → 365/366）",
          abs(decimal_year_of_date(dt.date(2004, 12, 31)) - (2004 + 365 / 366)) < 1e-12,
          f"{decimal_year_of_date(dt.date(2004, 12, 31)):.9f}")
    check("平年用 365 做分母（2003-12-31 → 364/365）",
          abs(decimal_year_of_date(dt.date(2003, 12, 31)) - (2003 + 364 / 365)) < 1e-12,
          f"{decimal_year_of_date(dt.date(2003, 12, 31)):.9f}")


def t_against_real_legacy():
    """★ A0 的核心验收：与真实 legacy TimeInfo 对表 + 两个反例必须失败。"""
    files = _grace_files()
    leg = _legacy_mids()
    if not files or leg is None:
        RESULTS.append(True)
        print("[PASS] 真实数据对表：跳过（找不到 GRACE 目录或 legacy TimeInfo）  skip")
        return
    ta = TimeAxis.from_grace_filenames(files)
    dv = ta.decimal_years
    d = np.abs(dv[:, None] - leg[None, :]).min(axis=1)
    n_exact, n_ok = int((d <= 1e-6).sum()), int((d <= 1e-5).sum())
    check("★ 203 个真实文件名 → 十进制年对上 legacy TimeInfo",
          n_exact >= 200, f"≤1e-6: {n_exact}/{len(dv)}；≤1e-5: {n_ok}；max {d.max():.2e}")
    bad = np.nonzero(d > 1e-5)[0]
    check("残留偏差只出现在跨年历元",
          all(os.path.basename(files[i]).split("_")[1][:4]
              != os.path.basename(files[i]).split("_")[1].split("-")[1][:4]
              for i in bad),
          f"{len(bad)} 个：" + ", ".join(os.path.basename(files[i]).split("_")[1]
                                        for i in bad))

    # ---- 反例 1：结束日不做"排他"处理（不加 1 天）
    import datetime as dt
    import calendar

    def dec_no_plus1(name):
        m = parse_grace_filename(name)
        s, e = m["start"], m["end"]
        ys, ds = int(s[:4]), int(s[4:])
        ye, de = int(e[:4]), int(e[4:])
        mid = (dt.date(ys, 1, 1).toordinal() + ds - 1
               + dt.date(ye, 1, 1).toordinal() + de - 1) / 2.0
        base = 366 if calendar.isleap(ys) else 365
        return ys + (mid - dt.date(ys, 1, 1).toordinal()) / base

    def dec_leap365(name):
        m = parse_grace_filename(name)
        s, e = m["start"], m["end"]
        ys, ds = int(s[:4]), int(s[4:])
        ye, de = int(e[:4]), int(e[4:])
        mid = (dt.date(ys, 1, 1).toordinal() + ds - 1
               + dt.date(ye, 1, 1).toordinal() + de - 1 + 1) / 2.0
        return ys + (mid - dt.date(ys, 1, 1).toordinal()) / 365.0

    for tag, fn, want_max in (("结束日不 +1 天（反例）", dec_no_plus1, 20),
                              ("闰年用 365（反例）", dec_leap365, 170)):
        e = np.abs(np.array([fn(f) for f in files])[:, None] - leg[None, :]).min(axis=1)
        check(f"反例必须失败：{tag}", int((e <= 1e-6).sum()) <= want_max,
              f"≤1e-6 只有 {int((e <= 1e-6).sum())}/{len(e)}（max {e.max():.2e}）")


def t_grace_header_and_prefer():
    files = _grace_files()
    if not files:
        RESULTS.append(True)
        print("[PASS] gfc 头部解析：跳过（找不到 GRACE 文件）                  skip")
        return
    ta = TimeAxis.from_gfc_headers(files)
    check("gfc 头部解析：历元数与文件数一致", len(ta) == len(files),
          f"{len(ta)} 个")
    check("gfc 头部解析：与文件名口径给出同一天",
          bool(np.all(ta.values == TimeAxis.from_grace_filenames(files).values)),
          "两路 mid 完全一致")
    auto = TimeAxis.from_grace(files, prefer="auto")
    check("from_grace 报告来源与冲突",
          auto.meta.get("time_source") == "header"
          and not [w for w in auto.meta.get("warnings", []) if "相差" in w],
          f"source={auto.meta.get('time_source')}, warnings={len(auto.meta.get('warnings', []))}")


def t_index_axis_is_honest():
    """没有日期时不许编日期：日期相关操作必须报错。"""
    ax = TimeAxis.from_index(5)
    check("from_index 标记为 index 且无日期",
          ax.kind == "index" and not ax.has_dates, f"kind={ax.kind}")
    for what, fn in (("dt_days", lambda: ax.dt_days),
                     ("span", lambda: ax.span),
                     ("decimal_years", lambda: ax.decimal_years),
                     ("missing", lambda: ax.missing()),
                     ("slice", lambda: ax.slice("2002-01-01", "2003-01-01"))):
        try:
            fn()
            check(f"index 轴调用 {what} 应报错", False, "没有报错")
        except ValueError as e:
            check(f"index 轴调用 {what} 应报错", "序号" in str(e), str(e)[:34])
    check("index 轴 summary 说明只有序号", "只有序号" in ax.summary())


def t_queries():
    d = ["2002-04-18", "2002-05-10", "2002-06-15", "2002-08-20"]
    ax = TimeAxis.from_datetimes(d)
    check("is_regular 对不等间隔返回 False", not ax.is_regular(),
          f"间隔 {np.round(ax.dt_days[1:], 1)}")
    check("is_regular(tol=30) 对月度解返回 True",
          TimeAxis.from_datetimes(d).is_regular(tol_days=30.0))
    miss = ax.missing(cadence_days=31, tol_days=6)
    check("missing 找到 2002-06-15 → 08-20 的缺口", len(miss) == 1 and miss[0][2] > 60,
          f"{len(miss)} 段，{miss[0][2]:.0f} 天" if miss else "没找到")
    dup = TimeAxis.from_datetimes(["2002-04-18", "2002-04-18", "2002-05-10"]).duplicates()
    check("duplicates 找到同一天两个文件", len(dup) == 1 and dup[0] == [0, 1], str(dup))
    check("nearest 取最近的历元", ax.nearest("2002-06-20") == 2,
          str(ax.values[ax.nearest("2002-06-20")])[:10])
    mask, note = ax.slice("2002-05-01", "2002-07-01")
    check("slice 给出布尔掩码与说明", list(mask) == [False, True, True, False], note)
    check("select 同步裁剪 labels", len(ax.select([0, 2])) == 2)


def t_legacy_timeinfo_roundtrip(tmp="/tmp"):
    import tempfile
    files = _grace_files()
    if not files:
        RESULTS.append(True)
        print("[PASS] legacy TimeInfo 往返：跳过（无 GRACE 文件）              skip")
        return
    ta = TimeAxis.from_grace_filenames(files[:8])
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "ti.dat")
        ta.to_legacy_timeinfo(out)
        back = TimeAxis.from_legacy_timeinfo(out)
        dt_s = np.abs((back.values - ta.values).astype("timedelta64[s]")
                      .astype(float))
        check("写 legacy TimeInfo → 读回：历元一致（≤20 s，格式本身只有 6 位小数）",
              float(dt_s.max()) <= 20.0, f"max {float(dt_s.max()):.0f} s")
        check("写 legacy TimeInfo → 读回：日期至多差 1 天（6 位小数的固有精度）",
              bool(np.all(np.abs((back.values.astype("datetime64[D]")
                                  - ta.values.astype("datetime64[D]"))
                                 .astype("timedelta64[D]").astype(int)) <= 1)),
              f"{len(back)} 个")
        check("写 legacy TimeInfo → 读回：十进制年一致",
              rel(back.decimal_years, ta.decimal_years) < 1e-7,
              f"rel = {rel(back.decimal_years, ta.decimal_years):.1e}")
        arr = np.loadtxt(out)
        check("legacy TimeInfo 布局 = [idx, y_start, y_end, y_mid]",
              arr.shape[1] == 4 and int(arr[0, 0]) == 1,
              f"shape {arr.shape}")


def t_coeffs_times_plumbing():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(5, 5, 3))
    S = rng.normal(size=(5, 5, 3))
    t = TimeAxis.from_datetimes(["2002-04-18", "2002-05-10", "2002-06-15"])
    co = SHCoeffs(C, S, times=t)
    check("SHCoeffs 接受 times 并与 ntime 一致", co.has_time and len(co.times) == 3)

    # 长度不一致必须报错
    try:
        SHCoeffs(C, S, times=TimeAxis.from_datetimes(["2002-04-18"]))
        check("times 长度与 ntime 不符应报错", False, "没有报错")
    except ValueError as e:
        check("times 长度与 ntime 不符应报错", "历元" in str(e), str(e)[:40])
    # 直接塞数组应报错（引导用 TimeAxis）
    try:
        SHCoeffs(C, S, times=np.array(["2002-04-18"]))
        check("times 塞裸数组应报错", False, "没有报错")
    except TypeError as e:
        check("times 塞裸数组应报错", "TimeAxis" in str(e), str(e)[:40])

    check("time_slice 返回 2D 且 times 清空",
          co.time_slice(1).C.ndim == 2 and co.time_slice(1).times is None)
    check("time_slice 记录 epoch 元信息",
          co.time_slice(1).meta.get("epoch") == "2002-05-10",
          str(co.time_slice(1).meta.get("epoch")))
    check("copy / truncate 保留 times",
          co.copy().times is not None and co.truncate(3).times is not None)
    check("apply_gaussian 保留 times", apply_gaussian(co, 300.0).times is not None)
    check("units.convert 保留 times",
          units.convert(units.with_field_unit(co, "geopotential"), "ewh").times is not None)
    check("__mul__ 保留 times", (co * 2.0).times is not None)

    sel = co.select_times("2002-05-01", "2002-06-20")
    check("select_times 按时间窗取子序列",
          sel.ntime == 2 and bool(np.all(sel.times.values
                                         == np.array(["2002-05-10", "2002-06-15"],
                                                     dtype="datetime64[s]"))),
          f"{sel.ntime} 个历元")

    # 时间轴不一致 → 必须报错
    other = SHCoeffs(C, S, times=TimeAxis.from_datetimes(
        ["2003-04-18", "2003-05-10", "2003-06-15"]))
    for op, fn in (("相加", lambda: co + other), ("相减", lambda: co - other)):
        try:
            fn()
            check(f"时间轴不一致时{op}应报错", False, "没有报错")
        except ValueError as e:
            check(f"时间轴不一致时{op}应报错", "不一致" in str(e), str(e).splitlines()[0][:40])
    # 一侧没有时间轴：容忍，但要留痕
    plain = SHCoeffs(C, S)
    mix = co + plain
    check("一侧无时间轴：沿用另一侧并留痕",
          mix.times is not None and "times_from" in mix.meta,
          str(mix.meta.get("times_from"))[:30])


def t_backwards_compatible():
    """没有 times 的对象行为必须与以前**逐位**相同。"""
    rng = np.random.default_rng(1)
    C = rng.normal(size=(4, 4, 2))
    S = rng.normal(size=(4, 4, 2))
    co = SHCoeffs(C, S)
    check("无 times 时 has_time 为 False", not co.has_time)
    check("无 times：time_slice(0) 回到 2D", co.time_slice(0).C.ndim == 2)
    check("无 times：add/sub 不报错",
          (co + co).C.shape == co.C.shape and (co - co).C.max() == 0.0)
    check("无 times：summary 不含时间行", "历元" not in co.summary())
    check("无 times：copy/truncate 仍为 None",
          co.copy().times is None and co.truncate(2).times is None)


def t_numeric_netcdf_time():
    """数值型 netCDF 时间坐标：``<n> <unit> since <epoch>`` 必须能解出日期。

    这条来自一个真实事故：CSR mascon 文件的 ``time`` 是 float32 + ``days since
    2002-01-01T00:00:00Z``，而且属性名写的是 **``Units``（大写 U）**。旧实现只把小写
    ``units`` 当回事，于是拿去当裸数字处理、构造 datetime 时抛异常，调用方再把异常
    吞成"没有时间轴" —— 一个 256 历元、横跨 2002–2026 的文件在界面上显示"无日期"。
    """
    import xarray as xr

    # ① 逐条单位：秒 / 分 / 时 / 天，且都要支持大写属性名与结尾的 Z
    base = "2002-01-01T00:00:00Z"
    cases = [("days", [0.0, 1.0, 365.0], [0, 24, 8760]),
             ("hours", [0.0, 1.5, 24.0], [0, 1.5, 24]),
             ("minutes", [0.0, 90.0], [0, 1.5]),
             ("seconds", [0.0, 5400.0], [0, 1.5])]
    for unit, vals, want_h in cases:
        coord = xr.DataArray(np.asarray(vals, dtype="float32"),
                             attrs={"Units": f"{unit} since {base}",
                                    "calendar": "gregorian"}, name="time")
        ax = TimeAxis.from_netcdf_coord(coord)
        got = [(ax.values[i] - np.datetime64("2002-01-01T00:00:00"))
               .astype("timedelta64[m]").astype(float) / 60.0
               for i in range(len(vals))]
        check(f"数值时间：{unit} since …（大写 Units + Z）",
              ax.has_dates and np.allclose(got, want_h, atol=1e-3),
              f"{got} vs {want_h}")

    # ② **半分日不能被截断**：GRACE 月度产品的历元是月中点（129.5 天 = 12:00）
    coord = xr.DataArray(np.array([107.0, 129.5, 227.5], dtype="float32"),
                         attrs={"Units": f"days since {base}"}, name="time")
    ax = TimeAxis.from_netcdf_coord(coord)
    check("数值时间：0.5 天保留为 12:00（不截断成整天）",
          str(ax.values[0]) == "2002-04-18T00:00:00"
          and str(ax.values[1]) == "2002-05-10T12:00:00",
          f"{ax.values[0]} / {ax.values[1]}")

    # ③ 长度随日历变化的单位**必须拒绝**，不许猜
    coord = xr.DataArray(np.array([0.0, 1.0, 12.0], dtype="float32"),
                         attrs={"units": "months since 2002-01-01"}, name="time")
    ax = TimeAxis.from_netcdf_coord(coord)
    check("数值时间：months since 拒绝近似并说明（降级为序号）",
          not ax.has_dates and "拒绝近似" in " ".join(map(str, ax.meta.values())),
          str(ax.meta.get("note"))[:48])

    # ④ 只有数值、没有 units → 降级为序号，绝不编日期
    coord = xr.DataArray(np.array([1.0, 2.0, 3.0], dtype="float32"), name="time")
    ax = TimeAxis.from_netcdf_coord(coord)
    check("数值时间：缺 units 时降级为序号（不编日历）", not ax.has_dates,
          str(ax.meta.get("note"))[:40])

    # ⑤ 四个调用点共用的那一个入口：list / dict / DataArray / TimeAxis 都要认
    tm = time_axis_from_meta
    raw = [107.0, 129.5]
    a1 = tm(raw, units=f"days since {base}")
    a2 = tm({"values": raw, "units": f"days since {base}"})
    a3 = tm(xr.DataArray(np.asarray(raw, dtype="float32"),
                         attrs={"Units": f"days since {base}"}, name="time"))
    a4 = TimeAxis.from_datetimes(["2002-04-18", "2002-05-10"])
    check("time_axis_from_meta：list + units ⇒ 真日期",
          a1.has_dates and str(a1.values[1]) == "2002-05-10T12:00:00")
    check("time_axis_from_meta：dict(values/units) ⇒ 真日期",
          a2.has_dates and str(a2.values[0]) == "2002-04-18T00:00:00")
    check("time_axis_from_meta：xarray 坐标 ⇒ 真日期", a3.has_dates)
    check("time_axis_from_meta：TimeAxis 原样返回", tm(a4) is a4)
    check("time_axis_from_meta：None ⇒ None", tm(None) is None)
    check("time_axis_from_meta：裸数字无 units ⇒ 序号（不猜）",
          not tm(raw).has_dates)


def t_io_records_time_units():
    """``read_grid`` 必须把时间坐标的 units 一起带出来（否则下游无从解码）。"""
    import tempfile

    import xarray as xr

    tmp = tempfile.mkdtemp(prefix="shkit_tunits_")
    try:
        p = os.path.join(tmp, "t.nc")
        tvals = [107.0, 129.5, 227.5]
        ds = xr.Dataset(
            {"v": (("time", "lat", "lon"),
                   np.zeros((3, 3, 6), dtype="float32"),
                   {"Units": "cm", "long_name": "LWE thickness"})},
            coords={"time": ("time", np.asarray(tvals, dtype="float32"),
                             {"Units": "days since 2002-01-01T00:00:00Z",
                              "calendar": "gregorian"}),
                    "lat": ("lat", np.array([-45.0, 0.0, 45.0])),
                    "lon": ("lon", np.array([0.0, 60.0, 120.0, 180.0,
                                             240.0, 300.0]))})
        ds.to_netcdf(p)
        lat, lon, grid, meta = shio.read_grid(p)
        check("read_grid 带出 time_units（大写 Units 也认）",
              meta.get("time_units") == "days since 2002-01-01T00:00:00Z",
              repr(meta.get("time_units")))
        check("read_grid 带出 time_calendar",
              meta.get("time_calendar") == "gregorian",
              repr(meta.get("time_calendar")))
        ax = time_axis_from_meta(meta.get("time"), units=meta.get("time_units"),
                                 calendar=meta.get("time_calendar"))
        check("read_grid → TimeAxis 解出真日期（含 12:00 中点）",
              ax.has_dates and str(ax.values[1]) == "2002-05-10T12:00:00",
              str(ax.values[1]))
        check("read_grid 带出**变量自己**的声明单位（大写 Units + 非米）",
              meta.get("variable_units") == "cm",
              repr(meta.get("variable_units")))
        from shkit.gui.dataset import Dataset                     # noqa: E402
        dss = Dataset.from_grid(p, lat, lon, grid, meta)
        sm = dss.summary()
        check("数据摘要里写明变量单位，并说明换算与单位无关",
              "变量单位    : cm" in sm and "线性换算" in sm,
              [ln.strip() for ln in sm.splitlines() if "变量单位" in ln][0][:58])
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def main():
    tests = [
        t_filename_parsing,
        t_decimal_year_roundtrip,
        t_against_real_legacy,
        t_grace_header_and_prefer,
        t_index_axis_is_honest,
        t_queries,
        t_legacy_timeinfo_roundtrip,
        t_coeffs_times_plumbing,
        t_backwards_compatible,
        t_numeric_netcdf_time,
        t_io_records_time_units,
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
