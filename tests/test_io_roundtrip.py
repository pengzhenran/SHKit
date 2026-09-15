# -*- coding: utf-8 -*-
"""
Independent round-trip checks for shkit.io (written separately from the
module's own test suite, so a bug in one does not hide a bug in the other).

Run:  python tests/test_io_roundtrip.py
"""
import glob
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from shkit.coeffs import SHCoeffs
from shkit import io as SIO

TMP = os.path.join(HERE, "_tmp_roundtrip")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:50s} {detail}")


def _nc_temp_dirs():
    """SHKit 为"非 ASCII 路径读 nc"建的临时目录（残留就是泄漏）。"""
    return sorted(p for p in glob.glob(os.path.join(tempfile.gettempdir(),
                                                    "shkit_nc_*"))
                  if os.path.isdir(p))


def build_coeffs(L=6, ntime=3):
    rng = np.random.default_rng(4)
    C = np.zeros((L + 1, L + 1, ntime))
    S = np.zeros((L + 1, L + 1, ntime))
    for t in range(ntime):
        for n in range(L + 1):
            for m in range(n + 1):
                C[n, m, t] = rng.standard_normal() / max(n, 1) ** 2
                if m:
                    S[n, m, t] = rng.standard_normal() / max(n, 1) ** 2
    return SHCoeffs(C, S)


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    co = build_coeffs()
    # 快照：本套件只负责证明**自己这一轮**没有泄漏。别的进程（用户正在用的界面、
    # 崩掉的脚本）留下的残留不该把这条判成红 —— 那会让"真泄漏"被淹没在噪声里。
    _nc_dirs_before = set(_nc_temp_dirs())

    print("== coefficient layouts ==")
    for layout, ext, tol in (("triangle", "sh", 1e-10),
                             ("triangle", "csv", 1e-10),
                             ("gmfcsv", "csv", 1e-10),
                             ("npy", "npy", 1e-12),
                             ("npz", "npz", 1e-12)):
        path = os.path.join(TMP, f"m_{layout}.{ext}")
        try:
            SIO.write_coeffs(co, path, layout=layout)
            back = SIO.read_coeffs(path)
            d = max(float(np.abs(back.C - co.C).max()),
                    float(np.abs(back.S - co.S).max()))
            check(f"coeffs {layout}.{ext} round trip", d < tol and back.ntime == 3,
                  f"ntime={back.ntime}, max|diff| = {d:.3e}")
        except Exception as exc:      # noqa: BLE001
            check(f"coeffs {layout}.{ext} round trip", False,
                  f"{type(exc).__name__}: {exc}")

    # gfc holds exactly one epoch: multi-epoch must be refused, not silently
    # concatenated
    p_gfc = os.path.join(TMP, "m.gfc")
    try:
        SIO.write_coeffs(co, p_gfc, layout="gfc")
        check("gfc refuses a multi-epoch object", False, "no error raised")
    except ValueError:
        check("gfc refuses a multi-epoch object", True,
              "ValueError with guidance (avoids silent data loss)")
    SIO.write_coeffs(co, p_gfc, layout="gfc", time=1)
    back1 = SIO.read_coeffs(p_gfc)
    d1 = max(float(np.abs(back1.C - co.C[:, :, 1]).max()),
             float(np.abs(back1.S - co.S[:, :, 1]).max()))
    check("gfc single-epoch round trip (time=1)", d1 < 1e-10,
          f"max|diff| = {d1:.3e}")

    print("\n== points ==")
    rng = np.random.default_rng(5)
    lat = rng.uniform(-80, 80, 300)
    lon = rng.uniform(0, 360, 300)
    val = rng.standard_normal(300)
    p = os.path.join(TMP, "pts.csv")
    SIO.write_points(p, lat, lon, val)
    la2, lo2, v2, meta = SIO.read_points(p)
    # note: the default text format is %.10g, so a lat of 80 deg round-trips to
    # ~5e-9 deg (about 0.5 mm) - pass fmt='%.17g' for bit-exact text
    check("points csv round trip",
          np.abs(la2 - lat).max() < 1e-8 and np.abs(v2 - val).max() < 1e-8,
          f"max dlat {np.abs(la2-lat).max():.2e} deg "
          f"(~{np.abs(la2-lat).max()*111e3:.2e} m), max dv {np.abs(v2-val).max():.2e}")

    pe = os.path.join(TMP, "pts_exact.csv")
    SIO.write_points(pe, lat, lon, val, fmt="%.17g")
    _, _, v2e, _ = SIO.read_points(pe)
    check("points csv round trip is exact to double precision with fmt='%.17g'",
          np.abs(v2e - val).max() < 1e-14,
          f"max dv {np.abs(v2e-val).max():.2e}")

    pn = os.path.join(TMP, "pts.npy")
    SIO.write_points(pn, lat, lon, val)
    la3, lo3, v3, _ = SIO.read_points(pn)
    check("points npy round trip",
          np.abs(la3 - lat).max() < 1e-12 and np.abs(v3 - val).max() < 1e-12,
          f"max dv {np.abs(v3-val).max():.2e}")

    pc = os.path.join(TMP, "中文目录", "散点.csv")
    os.makedirs(os.path.dirname(pc), exist_ok=True)
    SIO.write_points(pc, lat, lon, val)
    la4, lo4, v4, _ = SIO.read_points(pc)
    check("points round trip through a non-ASCII path",
          np.abs(v4 - val).max() < 1e-9,
          f"max dv {np.abs(v4-val).max():.2e}")

    print("\n== 散点的多时间（宽表 / 矩阵） ==")
    # 用户问："散点支持多时间吗" —— 这里把每条路都钉住：矩阵 .npy、value/value1/…、
    # 任意列名用 val_col=[…]、日期列名当时间轴、以及**长表**必须给出警告。
    from shkit.gui.dataset import Dataset as _PDS
    _N, _NT = 60, 3
    _rng2 = np.random.default_rng(9)
    _plat = _rng2.uniform(-80, 80, _N)
    _plon = _rng2.uniform(0, 360, _N)
    _pvals = _rng2.normal(size=(_N, _NT))
    _pdir = os.path.join(TMP, "points_mt")
    os.makedirs(_pdir, exist_ok=True)
    for _ext in ("csv", "txt", "npy"):
        _p = os.path.join(_pdir, f"mt.{_ext}")
        SIO.write_points(_p, _plat, _plon, _pvals)
        _la, _lo, _v, _m = SIO.read_points(_p)
        _tol = 1e-9 if _ext != "npy" else 0.0
        check(f"★ 散点多时次 {_ext}：写→读仍是 (N, ntime) 且逐值一致",
              _v.shape == (_N, _NT) and _m["ntime"] == _NT
              and float(np.abs(np.asarray(_v) - _pvals).max()) <= _tol,
              f"shape={_v.shape}, ntime={_m['ntime']}, "
              f"max|Δ|={float(np.abs(np.asarray(_v) - _pvals).max()):.2e}")

    # 任意列名：val_col=[…] 显式选多列（以前给 list 直接 TypeError）
    _p = os.path.join(_pdir, "generic.csv")
    np.savetxt(_p, np.column_stack([_plon, _plat, _pvals]), delimiter=",",
               header="lon,lat,e2002,e2003,e2004", comments="", fmt="%.10g")
    _la, _lo, _v, _m = SIO.read_points(_p, val_col=["e2002", "e2003", "e2004"])
    check("★ 任意列名：val_col=[名字…] 读成多时次",
          _v.shape == (_N, _NT) and _m["ntime"] == _NT,
          f"shape={_v.shape}，列={_m['value_columns']}")
    _la, _lo, _v2, _m2 = SIO.read_points(_p, val_col=[2, 4])
    check("val_col 也能给序号（并支持只取其中两个历元）",
          _v2.shape == (_N, 2)
          and float(np.abs(np.asarray(_v2)[:, 1] - _pvals[:, 2]).max()) < 1e-9,
          f"shape={_v2.shape}，列={_m2['value_columns']}")

    # 日期列名 → 直接当历元时间轴（散点也能有真实日期，趋势页才拟合得起来）
    _p = os.path.join(_pdir, "dated.csv")
    np.savetxt(_p, np.column_stack([_plon, _plat, _pvals]), delimiter=",",
               header="lon,lat,2002-01-18,2002-02-17,2002-03-19",
               comments="", fmt="%.10g")
    _la, _lo, _v, _m = SIO.read_points(_p)
    _ax = _PDS.from_points(_p, _la, _lo, _v, _m).time_axis()
    check("★ 日期列名自动当历元日期（日写法）",
          _v.shape == (_N, _NT)
          and _m.get("time") == ["2002-01-18", "2002-02-17", "2002-03-19"]
          and _ax is not None and _ax.has_dates,
          f"meta[time]={_m.get('time')}，has_dates="
          f"{None if _ax is None else _ax.has_dates}")
    for _hdr, _tag in ((["2002-01", "2002-02", "2002-03"], "月"),
                       (["2002", "2003", "2004"], "年")):
        _p2 = os.path.join(_pdir, f"dated_{_tag}.csv")
        np.savetxt(_p2, np.column_stack([_plon, _plat, _pvals]), delimiter=",",
                   header="lon,lat," + ",".join(_hdr), comments="",
                   fmt="%.10g")
        _l2, _o2, _v3, _m3 = SIO.read_points(_p2)
        _a3 = _PDS.from_points(_p2, _l2, _o2, _v3, _m3).time_axis()
        check(f"★ 日期列名：{_tag}写法也能当时间轴",
              _v3.shape == (_N, _NT) and _a3 is not None and _a3.has_dates,
              f"meta[time]={_m3.get('time')}")
    # 列名不像日期时**不许**编造日期
    _p = os.path.join(_pdir, "nodesc.csv")
    np.savetxt(_p, np.column_stack([_plon, _plat, _pvals]), delimiter=",",
               header="lon,lat,a,b,c", comments="", fmt="%.10g")
    _l4, _o4, _v4, _m4 = SIO.read_points(_p, val_col=["a", "b", "c"])
    _a4 = _PDS.from_points(_p, _l4, _o4, _v4, _m4).time_axis()
    check("列名不是日期时不编造日期（只给历元序号）",
          "time" not in _m4 and (_a4 is None or not _a4.has_dates),
          f"meta 里有 time 吗：{'time' in _m4}")

    # 长表（一行一个「点×历元」）必须**说出来**，不能悄悄只读第一列
    _rows = np.column_stack([
        np.repeat(_plon, _NT), np.repeat(_plat, _NT), _pvals.ravel()])
    _p = os.path.join(_pdir, "long.csv")
    np.savetxt(_p, _rows, delimiter=",", header="lon,lat,value",
               comments="", fmt="%.10g")
    _l5, _o5, _v5, _m5 = SIO.read_points(_p)
    check("★ 长表格式给出明确警告（避免悄悄只读第一列）",
          any("长表" in w for w in _m5["warnings"]),
          ([w for w in _m5["warnings"] if "长表" in w][0][:46]
           if any("长表" in w for w in _m5["warnings"]) else "没有警告"))

    print("\n== grids ==")
    latv = np.arange(-90.0, 91.0, 10.0)
    lonv = np.arange(0.0, 360.0, 20.0)
    g = np.outer(np.cos(np.deg2rad(latv)), np.ones(lonv.size)) + 1.0
    for ext, tol in (("nc", 1e-12), ("grd", 1e-9), ("csv", 1e-9)):
        path = os.path.join(TMP, f"g.{ext}")
        try:
            SIO.write_grid(path, latv, lonv, g)
            lv, lw, g2, _ = SIO.read_grid(path)
            d = float(np.abs(g2 - g).max()) if g2.shape == g.shape else np.inf
            check(f"grid {ext} round trip", d < tol,
                  f"shape {g2.shape}, max|diff| = {d:.3e}")
        except Exception as exc:      # noqa: BLE001
            check(f"grid {ext} round trip", False, f"{type(exc).__name__}: {exc}")

    print("\n== 带空格的路径（安装/数据目录的常态） ==")
    # 用户要求：打包时必须确认"安装路径带空格"不会让程序跑不起来。运行期真正会
    # 碰到空格的地方就是**文件路径**（数据、导出、逐历元缓存都在这些目录里），
    # 所以这里在带空格（并且带中文）的目录里把每条读写路径走一遍。
    _sp_root = os.path.join(tempfile.gettempdir(), "shkit space 带空格 dir")
    shutil.rmtree(_sp_root, ignore_errors=True)
    os.makedirs(_sp_root, exist_ok=True)
    _sp_nc = os.path.join(_sp_root, "带 空格 的 网格.nc")
    _sp_cube = np.stack([g, g * 1.1], axis=2)
    try:
        SIO.write_grid(_sp_nc, latv, lonv, _sp_cube, var="lwe",
                       meta={"time": ["2002-01-18", "2002-02-17"]},
                       long_name="space test")
        _lv, _lw, _b, _m = SIO.read_grid(_sp_nc)
        check("★ 带空格路径：网格 nc 写入→读回逐值一致",
              _b.shape == _sp_cube.shape
              and float(np.abs(_b - _sp_cube).max()) == 0.0,
              f"{os.path.basename(_sp_nc)}，max|Δ| = "
              f"{float(np.abs(_b - _sp_cube).max()):.3e}")
    except Exception as exc:                                     # noqa: BLE001
        check("★ 带空格路径：网格 nc 写入→读回逐值一致", False,
              f"{type(exc).__name__}: {exc}")

    try:
        _lz = SIO.read_grid(_sp_nc, lazy_time=True)[2]
        _ok = isinstance(_lz, SIO.LazyTimeCube) and np.array_equal(
            np.asarray(_lz)[:, :, 0], g)
        _lz.close()
        check("★ 带空格路径：懒加载读第 1 层逐值一致", bool(_ok),
              type(_lz).__name__)
    except Exception as exc:                                     # noqa: BLE001
        check("★ 带空格路径：懒加载读第 1 层逐值一致", False,
              f"{type(exc).__name__}: {exc}")

    try:
        _sp_sh = os.path.join(_sp_root, "系数 输出.sh")
        SIO.write_coeffs(co, _sp_sh, layout="triangle")
        _back = SIO.read_coeffs(_sp_sh)
        _d = max(float(np.abs(_back.C - co.C).max()),
                 float(np.abs(_back.S - co.S).max()))
        _sp_grd = os.path.join(_sp_root, "网格 输出.grd")
        SIO.write_grid(_sp_grd, latv, lonv, g)
        _g2 = SIO.read_grid(_sp_grd)[2]
        check("★ 带空格路径：系数与 grd 导出→读回都一致",
              _d < 1e-10 and float(np.abs(_g2 - g).max()) < 1e-9,
              f"系数 max|Δ| = {_d:.2e}，grd max|Δ| = "
              f"{float(np.abs(_g2 - g).max()):.2e}")
    except Exception as exc:                                     # noqa: BLE001
        check("★ 带空格路径：系数与 grd 导出→读回都一致", False,
              f"{type(exc).__name__}: {exc}")

    # TEMP 本身就带空格时（用户名里有空格就会这样）非 ASCII 兜底的副本也落在
    # 带空格的临时目录里 —— 这条路径以前没人走过。
    _old_temp = tempfile.tempdir
    _sp_temp = os.path.join(tempfile.gettempdir(), "shkit space temp")
    try:
        os.makedirs(_sp_temp, exist_ok=True)
        tempfile.tempdir = _sp_temp
        _sp_cn = os.path.join(_sp_root, "中文 目录", "带空格 x.nc")
        os.makedirs(os.path.dirname(_sp_cn), exist_ok=True)
        SIO.write_grid(_sp_cn, latv, lonv, g)
        _c2 = SIO.read_grid(_sp_cn)[2]
        check("★ 带空格路径：TEMP 也带空格时的非 ASCII 兜底仍然正确",
              float(np.abs(_c2 - g).max()) < 1e-9,
              f"tempfile.tempdir = ...{os.sep}{os.path.basename(_sp_temp)}")
    except Exception as exc:                                     # noqa: BLE001
        check("★ 带空格路径：TEMP 也带空格时的非 ASCII 兜底仍然正确", False,
              f"{type(exc).__name__}: {exc}")
    finally:
        tempfile.tempdir = _old_temp
        shutil.rmtree(_sp_temp, ignore_errors=True)
        shutil.rmtree(_sp_root, ignore_errors=True)

    print("\n== report ==")
    import json
    rp = os.path.join(TMP, "rep.json")
    SIO.write_report(rp, {"method": "quadrature", "nmax": 12, "警告": ["测试"]})
    with open(rp, encoding="utf-8") as fh:
        d = json.load(fh)
    check("report json round trip", d.get("nmax") == 12 and "警告" in d.keys(),
          f"keys = {sorted(d.keys())}")

    print("\n== netCDF 变量选择 / 元数据 / 并发打开 ==")
    # ⚠️ 这一组来自一个真实报障："第一次打开 mascon 总是报
    # RuntimeError: NetCDF: Not a valid ID"。真因是 GUI 为了填变量下拉框，
    # 在读数据的 worker 之外**又开了一次同一个文件**：netCDF4/HDF5 的 C 库带
    # 进程级全局状态，两个线程同时开同一份文件就会这样炸，而且这个异常不是
    # OSError，原来的"非 ASCII 路径退临时副本"兜底根本兜不住。
    multi = os.path.join(TMP, "multi.nc")
    nt_s, la_s, lo_s = 4, 3, 5
    try:
        import xarray as xr
        cube = np.arange(nt_s * la_s * lo_s, dtype="float32").reshape(
            nt_s, la_s, lo_s)
        ds_buf = xr.Dataset(
            {"lwe": (("time", "lat", "lon"), cube,
                     {"Units": "cm", "long_name": "LWE"}),
             "time_bounds": (("time", "b"), np.zeros((nt_s, 2), "float32"))},
            coords={"time": ("time", np.arange(nt_s, dtype="float32") * 30.0,
                             {"Units": "days since 2002-01-01"}),
                    "lat": ("lat", np.linspace(-60, 60, la_s)),
                    "lon": ("lon", np.linspace(0, 288, lo_s))},
        )
        # 写盘也要走 SHKit 的兜底：TMP 在中文路径下，直接 to_netcdf 会 PermissionError
        SIO.to_netcdf_path(ds_buf, multi)
        la, lo, g, m = SIO.read_grid(multi)
        check("自动探测：带上 nc_variables（只列网格变量）",
              m.get("nc_variables") == ["lwe"],
              str(m.get("nc_variables")))
        check("自动探测：带上时间轴与变量单位",
              m.get("time_units") == "days since 2002-01-01"
              and m.get("variable_units") == "cm",
              f"{m.get('time_units')} / {m.get('variable_units')}")
        la2, lo2, g2, m2 = SIO.read_grid(multi, var="lwe")
        check("指定变量读：网格与自动探测逐值一致",
              np.array_equal(np.asarray(g), np.asarray(g2)))
        check("★ 指定变量读：元数据与自动探测同样完整（以前丢时间轴）",
              m2.get("time_units") == m.get("time_units")
              and m2.get("variable_units") == m.get("variable_units")
              and m2.get("nc_variables") == m.get("nc_variables"),
              f"time_units={m2.get('time_units')!r}")
        try:
            SIO.read_grid(multi, var="nope")
            check("变量不存在时报错（并列出可用变量）", False, "没有报错")
        except ValueError as exc:
            check("变量不存在时报错（并列出可用变量）", "lwe" in str(exc),
                  str(exc)[:52])

        # 并发打开同一个文件（旧 GUI 的就是这么炸的）
        import threading
        errs = []

        def _open_read():
            try:
                SIO.read_grid(multi, var="lwe")
            except Exception as exc:                             # noqa: BLE001
                errs.append(f"{type(exc).__name__}: {exc}")

        def _open_names():
            try:
                with SIO.open_nc_dataset(multi) as ds:
                    list(ds.data_vars)
            except Exception as exc:                             # noqa: BLE001
                errs.append(f"{type(exc).__name__}: {exc}")

        for _ in range(8):
            ts = [threading.Thread(target=_open_read),
                  threading.Thread(target=_open_names)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        check("★ 并发打开同一份 nc 不再抛 NetCDF: Not a valid ID",
              not errs, errs[:1] or "8 轮并发全部成功")

        # 大文件保留 float32：省一次全量 float64 拷贝（实测 CSR mascon 上 0.9 s
        # 与一倍内存；曾被 read_grid 末尾那句 np.asarray(..., dtype=float) 抵消掉）。
        # 这里把阈值调小，让小文件也能走到这条分支。
        _old_min = SIO._KEEP_DTYPE_MIN
        try:
            SIO._KEEP_DTYPE_MIN = 10
            _l1, _o1, g32, m32 = SIO.read_grid(multi)
            check("★ 超过阈值时保留文件原精度 float32（省一次全量拷贝）",
                  np.asarray(g32).dtype == np.float32
                  and m32.get("dtype_policy") == "float32",
                  f"dtype={np.asarray(g32).dtype} policy={m32.get('dtype_policy')}")
            check("并在提示里写明（不静默改精度）",
                  any("float32" in w for w in m32.get("warnings", [])),
                  next((w[:44] for w in m32.get("warnings", [])
                        if "float32" in w), "无提示"))
            SIO._KEEP_DTYPE_MIN = 10 ** 18
            _l2, _o2, g64, m64 = SIO.read_grid(multi)
            check("低于阈值时按老规矩转 float64",
                  np.asarray(g64).dtype == np.float64
                  and m64.get("dtype_policy") == "float64",
                  f"dtype={np.asarray(g64).dtype}")
            check("两种精度下数值一致（只差表示精度）",
                  np.allclose(np.asarray(g32, dtype=float), np.asarray(g64),
                              rtol=1e-6, atol=0),
                  "float32 保留不影响结果")
        finally:
            SIO._KEEP_DTYPE_MIN = _old_min

        # ---- 懒加载多时次 nc：先给一层、其余按需/后台补，补齐后与整块读一致 ----
        # 用户报障："载入 mascon .nc 要 10 多秒"。实测打开时把 1.06 GB 全读进来
        # 要 2.4–3.0 s，而**一个时次**只要 253 ms；Panoply 快就是因为它只读一层。
        _tmp_before = _nc_temp_dirs()
        _la, _lo, cube, _ml = SIO.read_grid(multi, var="lwe", lazy_time=True)
        check("★ 多时次 nc 懒加载：返回 LazyTimeCube，且只读了第 1 个时次",
              isinstance(cube, SIO.LazyTimeCube)
              and cube.shape == (la_s, lo_s, nt_s)
              and cube.n_filled == 1 and not cube.is_filled(1),
              f"{type(cube).__name__} shape={cube.shape} "
              f"已填 {getattr(cube, 'n_filled', '?')}/{nt_s}")
        check("懒加载信息写进 meta（界面据此显示后台补齐进度）",
              (_ml.get("lazy_time") or {}).get("ntime") == nt_s,
              str(_ml.get("lazy_time")))
        _eager = np.asarray(g64)                     # 上面读的整块（float64）
        check("★ 懒加载的第 1 个时次与整块读逐值一致",
              np.array_equal(np.asarray(cube[:, :, 0], dtype=float), _eager[:, :, 0]),
              "max|Δ| = %.3g" % float(np.max(np.abs(
                  np.asarray(cube[:, :, 0], dtype=float) - _eager[:, :, 0]))))
        cube.ensure(2)
        check("★ ensure(t) 之后那一层可用，且与整块读逐值一致",
              cube.is_filled(2) and np.array_equal(
                  np.asarray(cube[:, :, 2], dtype=float), _eager[:, :, 2]),
              f"已填 {cube.n_filled}/{cube.ntime}")
        _prog = []
        cube.ensure_all(progress=lambda d, n: _prog.append((d, n)))
        check("★ ensure_all() 补齐后与整块读**逐值一致**（懒加载不改变数值）",
              cube.n_filled == nt_s
              and np.array_equal(np.asarray(cube, dtype=float), _eager),
              f"{cube.n_filled}/{cube.ntime}，进度回调 {len(_prog)} 次")
        check("懒加载容器的元数据与整块读一样完整（时间轴 / 变量单位）",
              _ml.get("time_units") == m.get("time_units")
              and _ml.get("variable_units") == "cm",
              f"{_ml.get('time_units')!r} / {_ml.get('variable_units')!r}")
        cube.close()
        check("★ close() 释放 nc 句柄并清掉 ASCII 临时副本（不泄漏）",
              _nc_temp_dirs() == _tmp_before,
              f"残留 {[os.path.basename(p) for p in _nc_temp_dirs()]}")

        # 回退：单时次 nc 不该为了"1 个时次"上懒加载，且必须**写明**回退理由
        single = os.path.join(TMP, "single.nc")
        SIO.to_netcdf_path(xr.Dataset(
            {"lwe": (("lat", "lon"), np.zeros((la_s, lo_s), "float32"))},
            coords={"lat": ("lat", np.linspace(-60, 60, la_s)),
                    "lon": ("lon", np.linspace(0, 288, lo_s))}), single)
        _l3, _o3, g_s, m_s = SIO.read_grid(single, var="lwe", lazy_time=True)
        check("★ 单时次 nc 自动回退到整块读，并把理由写进 meta（不静默切换）",
              not isinstance(g_s, SIO.LazyTimeCube)
              and bool(m_s.get("lazy_fallback")),
              f"{type(g_s).__name__}，理由 {m_s.get('lazy_fallback')}")
        check("回退后仍然读出网格（回退不是失败）",
              np.asarray(g_s).shape == (la_s, lo_s), str(np.asarray(g_s).shape))
    except ImportError as exc:                                   # noqa: BLE001
        check("netCDF 变量选择回归（缺 xarray）", False, str(exc)[:40])

    print("\n== netCDF non-ASCII workaround hygiene ==")
    # TMP lives under a non-ASCII tree, so the .nc round trip above exercised the
    # temp-copy fallback on BOTH read and write.  A leaked copy is a real bug:
    # an unclosed netCDF handle makes rmtree fail silently on Windows.
    leftovers = [p for p in _nc_temp_dirs() if p not in _nc_dirs_before]
    check("非 ASCII .nc 兜底不留临时目录", not leftovers,
          f"本轮新增残留 {len(leftovers)} 个: {leftovers[:2]}" if leftovers
          else f"本轮新增 0 个（临时目录里另有 {len(_nc_dirs_before)} 个"
               "是别的会话留下的）")
    check("兜底确实被触发过（本路径是中文路径）",
          not SIO._is_ascii_path(TMP), f"TMP = ...{os.sep}{os.path.basename(TMP)}")

    npass = sum(RESULTS)
    print(f"\n{'='*70}\n{npass}/{len(RESULTS)} checks passed")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
