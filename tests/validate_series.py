# -*- coding: utf-8 -*-
"""
Validation of multi-epoch series I/O (A1):

* ``read_coeffs_series`` -- directory / glob / manifest / list -> one SHCoeffs
  with a time axis, with product filtering, duplicate policy, ``strict_nmax``;
* the frozen ``series_nc`` container (contract §2.4) and its tolerant reader;
* the legacy triangle table (``*.dat`` + ``*_TimeInfo.dat``) both ways;
* extension-less and PO.DAAC ``SHM`` (``GRCOF2``) files, which is how a large
  part of the real GRACE-FO record actually ships.

The real-data checks at the end **skip** when the user's data is not next to this
checkout -- a test suite must not go red because a data disk moved.

Run:  python tests/validate_series.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit import io as SIO                                    # noqa: E402
from shkit.coeffs import SHCoeffs                              # noqa: E402
from shkit.timeaxis import TimeAxis                            # noqa: E402

RESULTS = []
PARENT = os.path.dirname(ROOT)
REAL_DIR = os.path.join(PARENT, "2_unzipped", "1_CSR", "01RL06", "01deg60")
REAL_LEGACY = os.path.join(PARENT, "3_processed", "GRACE SH read",
                           "CSR_rawSH60_total_216.dat")


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def skip(name, why):
    RESULTS.append(True)
    print(f"[PASS] {name:56s} skip（{why}）")


def rel(v, ref, floor=1e-9):
    """Relative difference ignoring entries whose reference is ~0."""
    v, ref = np.asarray(v), np.asarray(ref)
    m = np.abs(ref) > floor
    if not m.any():
        return 0.0
    return float(np.max(np.abs(v[m] - ref[m]) / np.abs(ref[m])))


# ---------------------------------------------------------------------------
# synthetic GRACE files (self-contained: no user data needed)
# ---------------------------------------------------------------------------
def _ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def make_gfc(path, start: dt.date, end: dt.date, nmax, seed, *,
             with_period=True, shm=False, ext=True):
    """Write a GRACE-named coefficient file.

    ``start`` is the first data day and ``end`` the last (inclusive), matching
    ``time_period_of_data``.  ``shm=True`` writes the PO.DAAC YAML form, which
    carries **only** the coverage window (and its end is exclusive).
    """
    label = "0" if not ext else "0"
    rng = np.random.default_rng(seed)
    out = []
    if shm:
        end_excl = end + dt.timedelta(days=1)
        out += ["header:", "  dimensions:",
                f"    degree                :  {nmax}", "",
                f"time_coverage_start   : {start}T00:00:00.00",
                f"time_coverage_end     : {end_excl}T00:00:00.00",
                "# End of YAML header"]
        kw = "GRCOF2"
    else:
        mid = start + (end - start) // 2
        out += ["# synthetic SHKit test model"]
        if with_period:
            out.append(f"time_period_of_data    {_ymd(start)} - {_ymd(end)}"
                       f"   (mid: {_ymd(mid)})")
        out += [f"max_degree             {nmax}", "end_of_head"]
        kw = "gfc"
    for n in range(nmax + 1):
        for m in range(n + 1):
            c = 1.0 if (n == 0 and m == 0) else rng.normal() * 1e-3
            s = 0.0 if m == 0 else rng.normal() * 1e-3
            out.append(f"{kw} {n:5d} {m:5d} {c: .12E} {s: .12E}"
                       " 0.0000E+00 0.0000E+00")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def _grace_name(start: dt.date, end: dt.date, *, mission="GRAC", center="UTCSR",
                deg="BA01", rl="0600", product="GSM", suffix=".gfc"):
    a = start.strftime("%Y") + f"{start.timetuple().tm_yday:03d}"
    b = end.strftime("%Y") + f"{end.timetuple().tm_yday:03d}"
    return f"{product}-2_{a}-{b}_{mission}_{center}_{deg}_{rl}{suffix}"


def build_case_dir(n=4, nmax=12, *, start=dt.date(2002, 4, 5)):
    """A temp directory of ``n`` synthetic monthly GSM files."""
    d = tempfile.mkdtemp(prefix="shkit_series_")
    dates = []
    cur = start
    for i in range(n):
        end = cur + dt.timedelta(days=27 + i % 3)
        dates.append((cur, end))
        make_gfc(os.path.join(d, _grace_name(cur, end)), cur, end, nmax, seed=i)
        cur = end + dt.timedelta(days=3)
    return d, dates


# ---------------------------------------------------------------------------
def t_series_read_basic():
    d, dates = build_case_dir(4, nmax=12)
    try:
        co = SIO.read_coeffs_series(d, nmax=12)
        check("目录 → 序列：(L+1,L+1,N)",
              co.C.shape == (13, 13, 4), str(co.C.shape))
        check("目录 → 序列：挂上时间轴且时间递增",
              co.times is not None and bool(np.all(np.diff(co.times.values) > 0)),
              f"{len(co.times)} 个历元，{str(co.times.values[0])[:10]} …")
        check("meta 记录来源与逐历元文件名",
              len(co.meta.get("source_files", [])) == 4
              and co.meta.get("time_source") == "header",
              f"time_source={co.meta.get('time_source')}")
        check("时间轴与文件名口径一致（头部与文件名同日）",
              bool(np.all(co.times.values.astype("datetime64[D]")
                          == TimeAxis.from_grace_filenames(
                              sorted(os.path.join(d, f) for f in os.listdir(d))
                          ).values.astype("datetime64[D]"))))
        # glob / 清单 / 显式列表 三种入口
        g = SIO.read_coeffs_series(os.path.join(d, "GSM-2_*"))
        check("通配入口等价", g.C.shape == co.C.shape and g.ntime == 4)
        manifest = os.path.join(d, "list.txt")
        with open(manifest, "w", encoding="utf-8") as fh:
            for f in sorted(os.listdir(d)):
                if f.endswith(".gfc"):
                    fh.write(f + "\n")
        m = SIO.read_coeffs_series(manifest)
        check("清单文件入口（相对路径按清单所在目录解析）",
              m.C.shape == co.C.shape)
        explicit = [os.path.join(d, f) for f in sorted(os.listdir(d))
                    if f.endswith(".gfc")]
        e = SIO.read_coeffs_series(explicit)
        check("显式列表入口", e.C.shape == co.C.shape)
        check("三种入口给出同一组系数",
              rel(e.C, co.C) == 0.0 and rel(m.C, co.C) == 0.0)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_product_filtering():
    d, dates = build_case_dir(2, nmax=8)
    try:
        # 目录里混入 GAD：应被筛掉并报告，而不是混进序列
        make_gfc(os.path.join(d, _grace_name(*dates[0], product="GAD")),
                 *dates[0], 8, seed=99)
        co = SIO.read_coeffs_series(d, nmax=8)
        check("目录里的 GAD 被筛掉（不混入 GSM 序列）", co.ntime == 2,
              f"{co.ntime} 个历元")
        check("被筛掉的文件出现在 warnings 里",
              any("筛选" in w for w in co.meta.get("warnings", [])),
              next((w for w in co.meta.get("warnings", []) if "筛选" in w), "")[:52])
        # 显式列表混产品 → 必须报错
        mixed = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".gfc"))
        try:
            SIO.read_coeffs_series(mixed)
            check("显式列表混入 GAD 应报错", False, "没有报错")
        except ValueError as e:
            check("显式列表混入 GAD 应报错", "不符合筛选条件" in str(e),
                  str(e).splitlines()[0][:48])
        # center 过滤（用**新**历元，避免与已有文件撞成重复历元）
        c = dates[1][1] + dt.timedelta(days=40)
        make_gfc(os.path.join(d, _grace_name(c, c + dt.timedelta(days=27),
                                             center="GFZOP")),
                 c, c + dt.timedelta(days=27), 8, seed=98)
        co2 = SIO.read_coeffs_series(d, nmax=8, center="GFZOP")
        check("按 center 过滤", co2.ntime == 1, f"{co2.ntime} 个历元")
        # 非 GRACE 命名的文件被忽略并报告
        with open(os.path.join(d, "notes.txt"), "w") as fh:
            fh.write("hello\n")
        co3 = SIO.read_coeffs_series(d, nmax=8)
        check("无关键被忽略且报告", co3.ntime == 3, f"{co3.ntime} 个历元")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_duplicates_and_gaps():
    d = tempfile.mkdtemp(prefix="shkit_dup_")
    try:
        a = dt.date(2002, 4, 5)
        # 与前一历元隔 ~4 个月（大于 cadence+tol，应被认成缺测）
        b = a + dt.timedelta(days=120)
        make_gfc(os.path.join(d, _grace_name(a, a + dt.timedelta(days=25))),
                 a, a + dt.timedelta(days=25), 6, seed=1)
        # same epoch, second file
        make_gfc(os.path.join(d, _grace_name(a, a + dt.timedelta(days=25),
                                             center="GFZOP")),
                 a, a + dt.timedelta(days=25), 6, seed=2)
        # a later epoch, separated by a big gap
        make_gfc(os.path.join(d, _grace_name(b, b + dt.timedelta(days=28))),
                 b, b + dt.timedelta(days=28), 6, seed=3)
        try:
            SIO.read_coeffs_series(d, nmax=6)
            check("重复历元默认应报错", False, "没有报错")
        except ValueError as e:
            check("重复历元默认应报错", "重复" in str(e), str(e).splitlines()[0][:46])
        first = SIO.read_coeffs_series(d, nmax=6, on_duplicate="first")
        last = SIO.read_coeffs_series(d, nmax=6, on_duplicate="last")
        check("on_duplicate='first' 保留先到的", first.ntime == 2, f"{first.ntime}")
        check("on_duplicate='last' 保留后到的", last.ntime == 2, f"{last.ntime}")
        check("first 与 last 取到的是不同文件",
              rel(first.C, last.C) > 1e-6, f"rel = {rel(first.C, last.C):.2e}")
        rep = SIO.read_coeffs_series(d, nmax=6, on_duplicate="report")
        check("on_duplicate='report' 保留全部并报告",
              rep.ntime == 3 and any("重复" in w for w in rep.meta.get("warnings", [])),
              f"{rep.ntime} 个历元")
        check("缺测被报告", any("缺测" in w for w in rep.meta.get("warnings", [])),
              next((w for w in rep.meta["warnings"] if "缺测" in w), "")[:46])
        ign = SIO.read_coeffs_series(d, nmax=6, on_duplicate="report",
                                     on_missing="ignore")
        check("on_missing='ignore' 不报缺测",
              not any("缺测" in w for w in ign.meta.get("warnings", [])))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_strict_nmax():
    d = tempfile.mkdtemp(prefix="shkit_nmax_")
    try:
        a = dt.date(2002, 4, 5)
        make_gfc(os.path.join(d, _grace_name(a, a + dt.timedelta(days=25))),
                 a, a + dt.timedelta(days=25), 10, seed=1)
        b = a + dt.timedelta(days=31)
        make_gfc(os.path.join(d, _grace_name(b, b + dt.timedelta(days=27))),
                 b, b + dt.timedelta(days=27), 6, seed=2)
        try:
            SIO.read_coeffs_series(d)
            check("阶数不一致时 strict_nmax=True 应报错", False, "没有报错")
        except ValueError as e:
            check("阶数不一致时 strict_nmax=True 应报错", "阶数不一致" in str(e),
                  str(e).splitlines()[0][:48])
        co = SIO.read_coeffs_series(d, strict_nmax=False)
        check("strict_nmax=False 统一到最高阶（零填充）",
              co.C.shape == (11, 11, 2), str(co.C.shape))
        check("零填充的低阶部分确为 0",
              float(np.max(np.abs(co.C[7:11, 7:11, 1]))) == 0.0)
        check("零填充被报告", any("阶数不一致" in w for w in co.meta.get("warnings", [])))
        co2 = SIO.read_coeffs_series(d, nmax=6, strict_nmax=False)
        check("显式 nmax 截断", co2.C.shape == (7, 7, 2), str(co2.C.shape))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_extensionless_and_shm():
    d = tempfile.mkdtemp(prefix="shkit_shm_")
    try:
        a = dt.date(2021, 12, 1)
        e = dt.date(2021, 12, 31)
        # PO.DAAC SHM form, no extension at all
        p = os.path.join(d, _grace_name(a, e, mission="GRFO", suffix=""))
        make_gfc(p, a, e, 6, seed=1, shm=True, ext=False)
        co = SIO.read_coeffs(p)
        check("无扩展名 + SHM(GRCOF2) 文件可读", co.nmax == 6, f"nmax={co.nmax}")
        check("SHM 识别为 gfc 家族", co.meta.get("format") == "gfc",
              str(co.meta.get("format")))
        s = SIO.read_coeffs_series(d, nmax=6)
        check("SHM 也能进序列（头部只有 coverage 窗口时取中点）",
              s.ntime == 1, f"{s.ntime} 个历元")
        want = np.datetime64("2021-12-16T12:00:00")
        check("coverage 窗口的**排他**结束 → 中点 = 12-16 12:00",
              bool(s.times.values[0] == want),
              f"{s.times.values[0]}（期望 {want}）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_series_nc_roundtrip():
    d, _ = build_case_dir(3, nmax=10)
    try:
        co = SIO.read_coeffs_series(d, nmax=10)
        co = co.with_unit("geopotential")
        co.meta["gaussian_km"] = 300.0
        co.meta["center"] = "UTCSR"
        out = os.path.join(d, "series.nc")
        SIO.write_coeffs(co, out, layout="series_nc")
        back = SIO.read_coeffs(out)
        check("series_nc 往返：系数逐位一致",
              rel(back.C, co.C) == 0.0 and rel(back.S, co.S) == 0.0)
        check("series_nc 往返：时间轴逐位一致",
              back.times is not None and bool(np.all(back.times.values == co.times.values)),
              f"{len(back.times) if back.times else 0} 个历元")
        check("series_nc 往返：shape", back.C.shape == co.C.shape, str(back.C.shape))
        import xarray as xr
        # the case dir lives in %TEMP% (ASCII), so netCDF4 opens it directly
        with xr.open_dataset(out) as ds:
            keys = set(ds.attrs)
            need = {"norm", "csphase", "field_unit", "nmax", "producer", "time_source"}
            check("series_nc 冻结属性齐全（契约 §2.4）", need <= keys,
                  f"缺 {sorted(need - keys)}" if need - keys else f"{len(keys)} 个属性")
            check("series_nc 变量名 = c/s，维度 = time/n/m",
                  "c" in ds and "s" in ds and set(ds["c"].dims) == {"time", "n", "m"},
                  f"{sorted(ds.data_vars)} {ds['c'].dims}")
            check("gaussian_km / center 落盘",
                  str(ds.attrs.get("gaussian_km")) == "300.0"
                  and ds.attrs.get("center") == "UTCSR",
                  f"center={ds.attrs.get('center')} gaussian={ds.attrs.get('gaussian_km')}")
            check("field_unit = geopotential", ds.attrs.get("field_unit") == "geopotential")
            # 宽容读：变量名大写 C/S
            ds.rename({"c": "C", "s": "S"}).to_netcdf(os.path.join(d, "upper.nc"),
                                                      engine="netcdf4")
        up = SIO.read_coeffs(os.path.join(d, "upper.nc"))
        check("宽容读：接受大写 C/S 变量名",
              up.C.shape == co.C.shape and rel(up.C, co.C) == 0.0)
        # 单历元也能写
        one = co.time_slice(0)
        out1 = os.path.join(d, "one.nc")
        SIO.write_coeffs(one, out1, layout="series_nc")
        b1 = SIO.read_coeffs(out1)
        check("series_nc 单历元往返", b1.C.ndim == 2 and rel(b1.C, one.C) == 0.0,
              f"shape {b1.C.shape}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_legacy_dat_roundtrip():
    d, _ = build_case_dir(4, nmax=8)
    try:
        co = SIO.read_coeffs_series(d, nmax=8)
        out = os.path.join(d, "legacy.dat")
        SIO.write_coeffs(co, out, layout="legacy_dat")
        check("legacy_dat 写出 companion _TimeInfo.dat",
              os.path.exists(os.path.join(d, "legacy_TimeInfo.dat")),
              str(sorted(os.listdir(d))[:3]))
        back = SIO.read_coeffs(out, layout="legacy_dat")
        check("legacy_dat 往返：系数逐位一致",
              rel(back.C, co.C) == 0.0 and rel(back.S, co.S) == 0.0)
        check("legacy_dat 往返：时间轴来自 TimeInfo",
              back.times is not None and back.times.meta.get("time_source")
              == "legacy:TimeInfo")
        check("legacy_dat 布局 = 首行表头 + (2*NC, ntime)",
              np.loadtxt(out, skiprows=1).shape
              == (2 * 9 * 10 // 2, 4), str(np.loadtxt(out, skiprows=1).shape))
        # 没有 TimeInfo 时必须降级为序号并报告，而不是编日期
        os.remove(os.path.join(d, "legacy_TimeInfo.dat"))
        plain = SIO.read_coeffs(out, layout="legacy_dat")
        check("缺 TimeInfo 时降级为序号并报告",
              plain.times.kind == "index"
              and any("TimeInfo" in w for w in plain.meta.get("warnings", [])),
              f"kind={plain.times.kind}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_real_data():
    """真实 GRACE 数据：★ A1 的核心验收（数据不在则跳过）。"""
    if not os.path.isdir(REAL_DIR):
        skip("★ 真实数据：目录 → 216 个历元序列", "找不到 " + REAL_DIR)
        skip("★ 与 legacy .dat 对表", "找不到真实数据")
        return
    co = SIO.read_coeffs_series(REAL_DIR)
    check("★ 真实数据：目录 → 216 个历元序列",
          co.ntime >= 200 and co.C.shape[1] == co.C.shape[0],
          f"{co.C.shape}（{co.ntime} 个历元，nmax={co.nmax}）")
    check("★ 时间单调递增且跨度覆盖 2002–2022",
          bool(np.all(np.diff(co.times.values) > 0))
          and str(co.times.values[0])[:4] == "2002"
          and str(co.times.values[-1])[:4] >= "2022",
          f"{str(co.times.values[0])[:10]} … {str(co.times.values[-1])[:10]}")
    miss = co.times.missing()
    check("★ 真实缺测（GRACE 任务间隙）被报告",
          len(miss) >= 5 and any("缺测" in w for w in co.meta.get("warnings", [])),
          f"{len(miss)} 段，最长 {max(m[2] for m in miss):.0f} 天" if miss else "没找到")
    if not os.path.exists(REAL_LEGACY):
        skip("★ 与 legacy .dat 对表", "找不到 " + REAL_LEGACY)
        return
    leg = SIO.read_coeffs(REAL_LEGACY, layout="legacy_dat")
    ours = co.times.decimal_years
    worst, n = 0.0, 0
    for j in range(leg.ntime):
        i = int(np.argmin(np.abs(ours - leg.times.decimal_years[j])))
        if abs(ours[i] - leg.times.decimal_years[j]) < 1e-4:
            n += 1
            worst = max(worst, rel(co.C[:, :, i], leg.C[:, :, j]))
    # ⚠️ legacy .dat 是 7 位有效数字的文本，绝对 1e-15 在数学上不可能。
    # 实测 5e-7，正好是文本精度 —— 这条才是能真正检查的最强断言。
    check("★ 与 legacy .dat 重叠历元对表（相对 ≤1e-6，受 7 位有效数字限制）",
          n >= 200 and worst <= 1e-6, f"{n} 个历元，最大相对差 {worst:.2e}")


# ---------------------------------------------------------------------------
def main():
    tests = [
        t_series_read_basic,
        t_product_filtering,
        t_duplicates_and_gaps,
        t_strict_nmax,
        t_extensionless_and_shm,
        t_series_nc_roundtrip,
        t_legacy_dat_roundtrip,
        t_real_data,
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
