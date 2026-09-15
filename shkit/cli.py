# -*- coding: utf-8 -*-
"""
shkit.cli
=========

Command line front end for :mod:`shkit`.  Everything here is a thin wrapper
around the library - no numerical logic is re-implemented in this file.

Usage
-----
::

    shkit weights --points data.csv --rule voronoi [--nmax 60] [--out weights.csv]
    shkit analyze  --points data.csv --nmax 60 [--method auto] [--rule auto]
                   [--niter 2] [--reg kaula] [--alpha 1e-6]
                   [--out-prefix out/run1] [--out-grid grid.nc]
                   [--gaussian-km 300] [--reconstruct]
    shkit synth    --coeffs model.sh --out-grid out.nc --lat-step 1 --lon-step 1
    shkit synth    --coeffs model.sh --points pts.csv
    shkit roundtrip --points data.csv --nmax 60 --rule voronoi
    shkit glq-grid --lmax 40 --out grid.nc
    shkit info     --coeffs model.sh

Conventions
-----------
* latitude / longitude in **degrees**, longitude written as ``[0, 360)``
* weights are solid-angle elements in steradian (``sum = 4*pi`` globally)
* coefficients follow :mod:`shkit.coeffs` (4-pi normalised associated Legendre
  functions, no Condon-Shortley phase)
* every path is read through :mod:`shkit.io`, so Chinese paths, ``.sh``
  triangle files, ICGEM ``.gfc`` and netCDF grids all work here too

Exit codes
----------
``0`` success, ``1`` unexpected internal error, ``2`` usage error (bad option,
missing file, unreadable input).  Every failure prints a Chinese explanation on
stderr, never a bare traceback.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional, Sequence

import numpy as np

from . import io as shio
from .analysis import analysis
from .basis import synthesize
from .coeffs import SHCoeffs
from .diagnostics import harmonic_resolution_km
from .weights import WEIGHT_RULES, compute_weights, glq_grid

__all__ = ["main", "build_parser"]

PROG = "shkit"

#: extensions read as a *grid* by --points (everything else goes to read_points)
_GRID_EXTS = (".nc", ".nc4", ".cdf", ".grd")

_METHODS = ("auto", "quadrature", "iterative", "projection", "wlsq", "cg")
_RULES = ("auto", "dh", "grid", "lattice", "voronoi", "delaunay", "uniform",
          "user")
_REGS = ("kaula", "tikhonov")
#: 物理量（见 shkit.units）：'scalar' = 无物理含义的标量场，不做换算；
#: 'geopotential' = 无量纲重力位系数 C_nm/S_nm（GRACE Level-2）。
_FIELD_UNITS = ("scalar", "geopotential", "geoid", "surface_density",
                "ewh", "radial_displacement")


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def _force_utf8() -> None:
    """Best-effort UTF-8 stdio so a Chinese report never raises on a GBK console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if getattr(stream, "encoding", "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _out(msg: str = "") -> None:
    print(msg)


def _hr(title: str = "", width: int = 76) -> None:
    if title:
        print(f"--- {title} " + "-" * max(width - len(title) - 5, 3))
    else:
        print("-" * width)


class CliError(Exception):
    """Usage-level failure: printed as a Chinese message, exit code 2."""


def _check_rule(rule: str) -> str:
    if rule not in _RULES:
        raise CliError(
            f"未知的权重规则 {rule!r}；可选: {', '.join(_RULES)}"
            f"（库内 WEIGHT_RULES = {WEIGHT_RULES}）")
    return rule


def _looks_like_grid_file(path: str) -> bool:
    return os.fspath(path).lower().endswith(_GRID_EXTS)


# ---------------------------------------------------------------------------
# shared loaders
# ---------------------------------------------------------------------------
def _load_points(path, lon_col=None, lat_col=None, val_col=None):
    """Read a scatter file; returns ``(lat, lon, values, meta)`` (values 1-D)."""
    lat, lon, vals, meta = shio.read_points(path, lat_col=lat_col,
                                            lon_col=lon_col, val_col=val_col)
    if vals.ndim == 2 and vals.shape[1] > 1:
        raise CliError(
            f"{path}: 读到 {vals.shape[1]} 个数值列（多时次）。本命令需要单列数值，"
            "请用 --val-col 指定列名或列序号。")
    return lat, lon, np.asarray(vals).ravel(), meta


def _load_field(path, var=None, lon_col=None, lat_col=None, val_col=None):
    """Load scattered **or** gridded data.

    Returns ``(lat, lon, values, meta, grid)`` where ``values`` is flat
    (``(nlat*nlon,)`` or ``(nlat*nlon, ntime)``) and ``grid`` is
    ``(lat_vec, lon_vec, cube)`` for a gridded file, else ``None``.
    """
    if _looks_like_grid_file(path):
        lat_vec, lon_vec, cube, meta = shio.read_grid(path, var=var)
        meta["input_kind"] = "grid"
        nlat, nlon = lat_vec.size, lon_vec.size
        if cube.ndim == 2:
            values = cube.ravel()
            ntime = 1
        else:
            ntime = cube.shape[2]
            values = cube.reshape(nlat * nlon, ntime)
        lat = np.repeat(lat_vec, nlon)
        lon = np.tile(lon_vec, nlat)
        meta["input_flat_order"] = "lat outer, lon inner (row-major)"
        if cube.ndim == 3 and ntime == 1:
            values = values[:, 0]
        return lat, lon, values, meta, (lat_vec, lon_vec, cube)

    lat, lon, values, meta = _load_points(path, lon_col, lat_col, val_col)
    meta["input_kind"] = "points"
    return lat, lon, values, meta, None


def _read_user_weights(path, n: int) -> np.ndarray:
    """Read a per-point weight column (last column of a csv/npy or 1-D text)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        w = np.asarray(np.load(path), dtype=float).ravel()
    else:
        import pandas as pd
        df = pd.read_csv(path, comment="#", engine="python", sep=None)
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            raise CliError(f"{path}: 没有找到数值列作为逐点权重")
        w = np.asarray(num.iloc[:, -1], dtype=float).ravel()
    if w.size != n:
        raise CliError(
            f"{path}: 权重个数 {w.size} 与点数 {n} 不一致")
    return w


def _make_weights(lat, lon, rule, user_path=None, nlon=None):
    """Wrap :func:`shkit.weights.compute_weights` with CLI-friendly errors."""
    kw = {}
    if rule == "user":
        if not user_path:
            raise CliError("--rule user 需要同时给出 --user-weights <文件>"
                           "（每点一个权重，csv/npy/txt 均可）")
        kw["user_w"] = _read_user_weights(user_path, np.asarray(lat).size)
    if rule == "dh":
        if nlon is None:
            raise CliError(
                "--rule dh（Driscoll-Healy）只适用于全球等角网格 "
                "(nlon == 2*nlat，nlat 为偶数)。散点数据请用 "
                "--rule voronoi / delaunay / grid。")
        kw["nlon"] = int(nlon)
    if rule == "glq":
        raise CliError(
            "--rule glq 不能从点集直接构造：请先用 'shkit glq-grid --lmax L "
            "--out grid.nc' 生成参考网格，再用 --rule user --user-weights 或 "
            "直接对网格文件做分析（规则 'dh'/'grid'）。")
    try:
        return compute_weights(lat, lon, rule=rule, **kw)
    except CliError:
        raise
    except Exception as exc:
        raise CliError(
            f"计算权重失败 (rule={rule!r}): {type(exc).__name__}: {exc}") from exc


def _method_kwargs(args) -> dict:
    """Translate the CLI's --method/--niter/--tau into ``analysis()`` keywords."""
    m = args.method
    kw = {}
    if m == "iterative":
        kw["niter"] = int(args.niter) if args.niter is not None else 1
    elif args.niter:
        kw["niter"] = int(args.niter)
    tau = getattr(args, "tau", None)
    if tau is not None:
        kw["tau"] = float(tau)
    return kw


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------
def cmd_weights(args) -> int:
    """Print the WeightSet diagnostics, optionally writing per-point weights."""
    _check_rule(args.rule)
    lat, lon, _vals, meta = _load_points(args.points, args.lon_col,
                                         args.lat_col, args.val_col)

    nlon = None
    if args.rule == "dh":
        nlon = meta.get("nlon") or int(np.unique(lon).size)
    ws = _make_weights(lat, lon, args.rule, args.user_weights, nlon=nlon)

    _out(f"输入文件   : {meta['source_file']}")
    _out(f"点数       : {lat.size}   ({meta.get('format')})")
    _out(f"纬度范围   : {np.min(lat):.6f} .. {np.max(lat):.6f} deg")
    _out(f"经度范围   : {np.min(lon):.6f} .. {np.max(lon):.6f} deg")
    if meta.get("warnings"):
        _out("读取警告   :")
        for wmsg in meta["warnings"]:
            _out(f"  - {wmsg}")
    _hr()
    _out(ws.describe())
    _hr()
    _out("权重单位   : 立体角元 (球面度, sr)；全球采样 sum(w) = 4*pi = "
         f"{4 * np.pi:.6f}")
    _out("提示       : 面元权重是*求积*权重，不是统计权重；"
         "最小二乘要用 1/sigma_i^2（CLI 里暂无 --sigma）")

    if args.out:
        lines = [f"# shkit weights: rule={ws.rule!r} n={ws.n_points} "
                 f"sum(w)={ws.total:.12g} 4*pi={4 * np.pi:.12g}",
                 "lon,lat,weight"]
        with np.errstate(all="ignore"):
            for i in range(lat.size):
                lines.append(f"{lon[i]:.10g},{lat[i]:.10g},{ws.w[i]:.12g}")
        shio._text_write(args.out, "\n".join(lines) + "\n")   # noqa: SLF001
        _out(f"已写出逐点权重: {args.out}  （列: lon, lat, weight，单位 sr）")
    return 0


def cmd_analyze(args) -> int:
    """Analyse scattered points or a grid into SH coefficients."""
    _check_rule(args.rule)
    if args.method not in _METHODS:
        raise CliError(f"未知的 --method {args.method!r}；可选: {', '.join(_METHODS)}")
    if args.reg is not None and args.reg not in _REGS:
        raise CliError(f"未知的 --reg {args.reg!r}；可选: {', '.join(_REGS)}")

    lat, lon, values, meta, grid = _load_field(args.points, args.var,
                                                args.lon_col, args.lat_col,
                                                args.val_col)
    ntime = 1 if values.ndim == 1 else values.shape[1]
    _out(f"输入       : {meta['source_file']}  ({meta.get('input_kind')})")
    _out(f"点数       : {lat.size}  (ntime={ntime})")
    if meta.get("lat_order_flipped"):
        _out("注意       : 输入网格纬度轴是降序，读取时已翻转为升序")
    for w in meta.get("warnings", []):
        _out(f"读取警告   : {w}")

    nlon = meta.get("nlon") if meta.get("input_kind") == "grid" else None
    ws = _make_weights(lat, lon, args.rule, args.user_weights, nlon=nlon)

    sigma = None
    if args.sigma_col is not None:
        import pandas as pd
        try:
            df = pd.read_csv(args.points, comment="#", engine="python", sep=None)
        except Exception as exc:
            raise CliError(f"--sigma-col 只支持散点 csv 输入: {exc}") from exc
        try:
            col = (df[args.sigma_col] if isinstance(args.sigma_col, str)
                   and args.sigma_col in df.columns
                   else df.iloc[:, int(args.sigma_col)])
        except Exception as exc:
            raise CliError(
                f"--sigma-col {args.sigma_col!r} 无法解析；文件列为 "
                f"{list(df.columns)}（也可给 0 起始的列序号）") from exc
        sigma = np.asarray(col, dtype=float)
        if sigma.size != lat.size:
            raise CliError(
                f"--sigma-col 长度 {sigma.size} 与点数 {lat.size} 不一致")

    kw = _method_kwargs(args)
    reg = args.reg
    if reg is None and args.alpha is not None:
        reg = "tikhonov"
    coeffs, report = analysis(lat, lon, values, args.nmax, method=args.method,
                              weights=ws, sigma=sigma, reg=reg,
                              alpha=args.alpha,
                              field_unit=getattr(args, "field_unit", "scalar"),
                              target_unit=getattr(args, "target_unit", None),
                              longitude_fft=getattr(args, "longitude_fft", "auto"),
                              **kw)

    # 高斯平滑：直接乘到系数上，只施加一次。
    # 导出的 .sh、重建场、输出场都用同一套平滑系数；若改成只在重建时施加
    # （旧做法），存下来的系数就是未平滑的，用户改 --gaussian-km 会觉得"没效果"；
    # 两处都施加则乘两次。
    Wg = None
    if args.gaussian_km:
        from .filters import apply_gaussian
        Wg = shio.gaussian_degree_weights(coeffs.nmax, args.gaussian_km)
        coeffs = apply_gaussian(coeffs, float(args.gaussian_km))
        coeffs.meta["gaussian_radius_km"] = float(args.gaussian_km)
        coeffs.meta["gaussian_W0"] = float(Wg[0])
        coeffs.meta["gaussian_Wnmax"] = float(Wg[-1])
        Wg = None            # 系数已平滑，后面不许再乘一次

    _hr("AnalysisReport")
    _out(report.describe())
    if args.gaussian_km:
        Wshow = shio.gaussian_degree_weights(coeffs.nmax, args.gaussian_km)
        _hr("Gaussian 平滑（已乘到系数上，只施加一次）")
        _out(f"半径 {args.gaussian_km:g} km（0.5 幅度半宽）："
             f"W(0)={Wshow[0]:.6f}, W(1)={Wshow[1]:.6g}, "
             f"W(10)={Wshow[min(10, len(Wshow) - 1)]:.6g}, "
             f"W(nmax)={Wshow[-1]:.6g}")
        _out("导出的系数与重建场/输出场都是这套已平滑的系数，不会重复施加。")
    _out(f"半波长分辨率 @nmax={coeffs.nmax}: "
         f"{harmonic_resolution_km(coeffs.nmax):.1f} km")

    # ------------------------------------------------------------ reconstruct
    # 重建场与输入同物理量（C_nm × f_u），所以先把系数换回输入那一档
    recon_coeffs = _input_unit_coeffs(coeffs)
    recon_grid = None
    recon_points = None
    if args.reconstruct:
        if grid is not None:
            recon_grid = _resample(recon_coeffs, grid, Wg, ntime)
        else:
            fit = np.asarray(synthesize(lat, lon, recon_coeffs.C, recon_coeffs.S,
                                        weights_scale=Wg if Wg is not None else 1.0))
            recon_points = (lat, lon, fit)
            orig = np.asarray(values)
            rel = float(np.sqrt(np.mean((fit - orig) ** 2)) /
                        max(np.sqrt(np.mean(orig ** 2)), 1e-300))
            _hr("重建检查（在输入点上）")
            _out(f"  点数 {lat.size}      相对 RMS 误差 = {rel:.6e}")

    # ---------------------------------------------------------------- outputs
    if args.out_prefix:
        d = os.path.dirname(os.path.abspath(args.out_prefix)) or "."
        name = os.path.basename(args.out_prefix)
        if not name:
            raise CliError(f"--out-prefix {args.out_prefix!r} 缺少文件名部分")
        res = shio.save_result(d, name, coeffs, report,
                               grid=recon_grid, points=recon_points,
                               extra={"cli_command": _argv_string(),
                                      "input_file": str(meta["source_file"]),
                                      "gaussian_km": args.gaussian_km})
        _hr("落盘结果")
        for k in ("coeffs", "coeffs_npz", "report_json", "report_md",
                  "grid", "points"):
            if res.get(k):
                _out(f"  {k:12s}: {res[k]}")

    if args.out_grid:
        _write_field(args, coeffs, report, Wg, ntime, grid,
                     recon_grid, recon_points, lat, lon,
                     args.out_grid, target=None, kind="重建场")
    if getattr(args, "out_field", None):
        _write_field(args, coeffs, report, Wg, ntime, grid,
                     recon_grid, recon_points, lat, lon,
                     args.out_field, target=args.target_unit, kind="输出场")
    return 0


def _write_field(args, coeffs, report, Wg, ntime, grid,
                 recon_grid, recon_points, lat, lon, path,
                 target, kind):
    """写出重建场（target=None，与输入同量）或输出场（target=「输出为」）。"""
    from . import units as _units
    if target:
        # 输出场 = C_nm × f_t：先把系数换算成目标量自己的系数，再综合
        cc = _units.convert(coeffs, target)
    else:
        # 重建场 = C_nm × f_u：换回输入那一档，与输入网格同物理量
        cc = _input_unit_coeffs(coeffs)
    unit_label = _units.FIELD_UNIT_LABELS.get(_units.field_unit(cc),
                                              _units.field_unit(cc))
    if os.fspath(path).lower().endswith(".grd") and ntime > 1:
        raise CliError(f"{kind} 的 .grd 只能存单时次；多时次请用 .nc")
    if grid is not None and target is None:
        la, lo, cube = (_resample(cc, grid, Wg, ntime)
                        if recon_grid is None else recon_grid)
        shio.write_grid(path, la, lo, cube, var="reconstructed",
                        meta={"nmax": coeffs.nmax, "method": report.method,
                              "weight_rule": report.weight_rule,
                              "gaussian_km": args.gaussian_km,
                              "field_unit": _units.field_unit(cc),
                              "forward_from": coeffs.meta.get("forward_from"),
                              "shkit_cli": f"shkit analyze --out-{'grid' if kind == '重建场' else 'field'}"},
                        long_name=f"SH {kind}",
                        units=unit_label)
        _out(f"{kind}已写出: {path}  ({la.size} x {lo.size}"
             f"{'' if cube.ndim == 2 else ' x ' + str(cube.shape[2])})"
             f"  [{unit_label}]")
        return
    if grid is not None:
        la, lo = grid[0], grid[1]
        vals = np.asarray(synthesize(la, lo, cc.C, cc.S,
                                     weights_scale=Wg if Wg is not None else 1.0))
        shape = ((la.size, lo.size) if vals.ndim == 1
                 else (la.size, lo.size, vals.shape[1]))
        shio.write_grid(path, la, lo, vals.reshape(shape), var="output_field",
                        meta={"nmax": coeffs.nmax, "field_unit": _units.field_unit(cc)},
                        long_name=f"SH {kind}", units=unit_label)
        _out(f"{kind}已写出: {path}  ({la.size} x {lo.size})  [{unit_label}]")
        return
    if target is None and recon_points is not None:
        fit = recon_points[2]
    else:
        fit = np.asarray(synthesize(lat, lon, cc.C, cc.S,
                                    weights_scale=Wg if Wg is not None else 1.0))
    shio.write_points(path, lat, lon, fit,
                      comment=(f"shkit analyze {kind}\n"
                               f"nmax={coeffs.nmax} "
                               f"method={report.method} "
                               f"rule={report.weight_rule}\n"
                               f"field_unit={_units.field_unit(cc)}"))
    _out(f"{kind}散点已写出: {path}  ({lat.size} 点)  [{unit_label}]")
    return 0


def _input_unit_coeffs(coeffs):
    """把导出的经典位系数换回**输入那一档**物理量自己的系数。

    重建场必须与输入网格同物理量（``C_nm × f_u``），否则没法逐点比对。
    """
    from . import units as _units
    src = coeffs.meta.get("forward_from") or _units.field_unit(coeffs)
    if src == _units.field_unit(coeffs):
        return coeffs
    return _units.convert(coeffs, src)


def _resample(coeffs, grid, Wg, ntime):
    """Synthesise the field on the axes of an existing grid."""
    la, lo, cube = grid
    vals = synthesize(la, lo, coeffs.C, coeffs.S,
                      weights_scale=Wg if Wg is not None else 1.0)
    vals = np.asarray(vals)
    shape = (la.size, lo.size) if vals.ndim == 1 else (la.size, lo.size, vals.shape[1])
    return (la, lo, vals.reshape(shape))


def cmd_synth(args) -> int:
    """Synthesise a field from coefficients at grid nodes or at given points."""
    coeffs = shio.read_coeffs(args.coeffs, nmax=args.nmax, layout=args.layout)
    Wg = None
    if args.gaussian_km:
        Wg = shio.gaussian_degree_weights(coeffs.nmax, args.gaussian_km)
    _out(coeffs.summary())
    _out(f"布局       : {coeffs.meta.get('layout')}   "
         f"ntime={coeffs.ntime}   ncoef={(coeffs.nmax + 1) ** 2}")
    if Wg is not None:
        _out(f"Gaussian   : 半径 {args.gaussian_km:g} km, W(nmax)={Wg[-1]:.6g}")

    if args.points:
        lat, lon, _v, meta = _load_points(args.points, args.lon_col,
                                          args.lat_col, args.val_col)
        frep: dict = {}
        vals = np.asarray(synthesize(lat, lon, coeffs.C, coeffs.S,
                                     weights_scale=Wg if Wg is not None else 1.0,
                                     longitude_fft=args.longitude_fft,
                                     report=frep))
        _out(f"经度路径   : {frep.get('longitude_path')} —— {frep.get('longitude_reason')}")
        out = args.out or _derive_out(args.points, "_synth")
        shio.write_points(out, lat, lon, vals,
                          comment=(f"# shkit synth 结果\n"
                                   f"# coeffs={coeffs.meta.get('source_file')}\n"
                                   f"# nmax={coeffs.nmax} ntime={coeffs.ntime}"))
        _out(f"已合成 {lat.size} 个点 -> {out}")
        _out(f"  范围 = [{np.nanmin(vals):.6g}, {np.nanmax(vals):.6g}]")
        return 0

    out = args.out_grid
    if not out:
        raise CliError("请给出 --out-grid（规则网格）或 --points（指定点上合成）")
    lat_vec = np.arange(-90.0, 90.0 + 1e-9, args.lat_step)
    lon_vec = np.arange(0.0, 360.0 - 1e-9, args.lon_step)
    LA, LO = np.meshgrid(lat_vec, lon_vec, indexing="ij")
    grep: dict = {}
    vals = np.asarray(synthesize(LA.ravel(), LO.ravel(), coeffs.C, coeffs.S,
                                 weights_scale=Wg if Wg is not None else 1.0,
                                 longitude_fft=args.longitude_fft,
                                 report=grep))
    _out(f"经度路径   : {grep.get('longitude_path')} —— {grep.get('longitude_reason')}")
    if vals.ndim == 1:
        cube = vals.reshape(lat_vec.size, lon_vec.size)
    else:
        cube = vals.reshape(lat_vec.size, lon_vec.size, vals.shape[1])
    shio.write_grid(out, lat_vec, lon_vec, cube, var=args.var,
                    meta={"nmax": coeffs.nmax, "ntime": coeffs.ntime,
                          "source_coeffs": str(coeffs.meta.get("source_file")),
                          "gaussian_km": args.gaussian_km,
                          "shkit_cli": "shkit synth"},
                    long_name="SH synthesis")
    _out(f"网格       : {lat_vec.size} x {lon_vec.size}"
         f"{'' if cube.ndim == 2 else ' x ' + str(cube.shape[2])}"
         f"  -> {out}")
    _out(f"  范围 = [{np.nanmin(cube):.6g}, {np.nanmax(cube):.6g}]")
    return 0


def _derive_out(path: str, suffix: str) -> str:
    root, _ext = os.path.splitext(os.fspath(path))
    return root + suffix + ".csv"


def cmd_roundtrip(args) -> int:
    """Self-check: analyse -> synthesise -> report the error."""
    _check_rule(args.rule)
    if args.method not in _METHODS:
        raise CliError(f"未知的 --method {args.method!r}；可选: {', '.join(_METHODS)}")

    truth = None
    if args.coeffs:
        truth = shio.read_coeffs(args.coeffs, nmax=args.nmax, layout=args.layout)
        _out(truth.summary())
        if args.points:
            lat, lon, _v, meta = _load_points(args.points, args.lon_col,
                                              args.lat_col, args.val_col)
        else:
            lat, lon, meta = _fib_points(args.npoints, seed=args.seed)
            meta = {"source_file": f"<fibonacci {lat.size} points>",
                    "input_kind": "points", "warnings": [
                        f"未给 --points，使用确定性 Fibonacci 球面采样 "
                        f"N={lat.size}"]}
        values = np.asarray(synthesize(lat, lon, truth.C, truth.S)).ravel()
    else:
        lat, lon, values, meta, _grid = _load_field(args.points, args.var,
                                                     args.lon_col, args.lat_col,
                                                     args.val_col)
        truth = None

    ntime = 1 if values.ndim == 1 else values.shape[1]
    _out(f"输入       : {meta['source_file']}")
    _out(f"点数       : {lat.size}  (ntime={ntime})")
    for w in meta.get("warnings", []):
        _out(f"读取警告   : {w}")

    ws = _make_weights(lat, lon, args.rule, args.user_weights)
    coeffs, report = analysis(lat, lon, values, args.nmax, method=args.method,
                              weights=ws, **_method_kwargs(args))

    fit = np.asarray(synthesize(lat, lon, coeffs.C, coeffs.S))
    data = np.asarray(values)
    r_res = float(np.sqrt(np.mean((fit - data) ** 2)) /
                  max(float(np.sqrt(np.mean(data ** 2))), 1e-300))

    _hr("AnalysisReport")
    _out(report.describe())
    _hr("往返误差（analysis -> synthesis，在输入点上）")
    _out(f"  点数            : {lat.size}")
    _out(f"  数据 RMS        : {float(np.sqrt(np.mean(data ** 2))):.6e}")
    _out(f"  残差 RMS        : {float(np.sqrt(np.mean((fit - data) ** 2))):.6e}")
    _out(f"  相对 RMS        : {r_res:.6e}")
    _out(f"  最大绝对偏差    : {float(np.max(np.abs(fit - data))):.6e}")

    if truth is not None:
        L = max(truth.nmax, coeffs.nmax)
        a, b = truth.truncate(L), coeffs.truncate(L)
        num = np.linalg.norm(a.C - b.C) ** 2 + np.linalg.norm(a.S - b.S) ** 2
        den = max(np.linalg.norm(b.C) ** 2 + np.linalg.norm(b.S) ** 2, 1e-300)
        cerr = float(np.sqrt(num / den))
        _hr("系数恢复误差（相对真实系数）")
        _out(f"  系数相对误差    : {cerr:.6e}")
        _out(f"  nmax            : {coeffs.nmax}  (ncoef={(coeffs.nmax + 1) ** 2})")
        _out(f"  判定            : "
             + ("优秀（< 1e-10）" if cerr < 1e-10 else
                "良好（< 1e-6）" if cerr < 1e-6 else
                "可接受（< 1e-3）" if cerr < 1e-3 else
                "偏差较大：检查权重规则/采样密度/条件数"))
    else:
        _out("提示       : 未给 --coeffs，只报告数据残差；"
             "给 --coeffs 可直接给出系数恢复误差")

    if args.out:
        shio.write_report(args.out, report,
                          extra={"roundtrip_rel_rms": r_res,
                                 "n_points": int(lat.size)})
        _out(f"报告已写出: {args.out}")
    return 0


def _fib_points(n: int, seed: int = 0):
    """Deterministic quasi-uniform sphere sampling (Fibonacci lattice)."""
    n = max(int(n), 8)
    i = np.arange(n) + 0.5 + (seed % 7) * 1e-3
    colat = np.arccos(1.0 - 2.0 * i / n)
    lon = np.mod(np.pi * (1 + 5 ** 0.5) * np.arange(n), 2 * np.pi)
    return 90.0 - np.rad2deg(colat), np.rad2deg(lon), {}


def cmd_glq_grid(args) -> int:
    """Write the exact Gauss-Legendre quadrature reference grid."""
    lat, lon, w = glq_grid(args.lmax)
    lat_u = np.unique(lat)
    lon_u = np.unique(lon)
    nlat, nlon = lat_u.size, lon_u.size
    if lat_u.size * lon_u.size != lat.size:
        raise CliError("glq_grid 返回的不是规则网格，无法整理为二维网格")
    W = np.asarray(w).reshape(nlat, nlon)
    lat2 = np.asarray(lat).reshape(nlat, nlon)[:, 0]
    lon2 = np.asarray(lon).reshape(nlat, nlon)[0, :]
    shio.write_grid(args.out, lat2, lon2, W, var="weight",
                    meta={"lmax": int(args.lmax),
                          "weight_rule": "glq",
                          "sum_weight": float(np.sum(W)),
                          "exact_for_degree": f"<= {args.lmax}",
                          "shkit_cli": "shkit glq-grid"},
                    long_name="Gauss-Legendre solid angle element",
                    units="sr")
    _hr("GLQ 参考网格")
    _out(f"  lmax            : {args.lmax}")
    _out(f"  网格            : {nlat} (Gauss 纬度) x {nlon} (等间隔经度) = "
         f"{nlat * nlon} 点")
    _out(f"  权重和 sum(w)   : {float(np.sum(W)):.12f}   "
         f"(4*pi = {4 * np.pi:.12f})")
    _out(f"  可精确恢复到    : n <= {args.lmax} 阶的带限场")
    _out(f"  已写出          : {args.out}")
    _out("  用法提示        : 读取该网格做分析时请用 --rule user "
         "--user-weights <每点权重>，或直接用 weights 列（列名 weight）")
    return 0


def cmd_info(args) -> int:
    """Print a coefficient file's summary and its per-degree RMS table."""
    coeffs = shio.read_coeffs(args.coeffs, nmax=args.nmax, layout=args.layout)
    m = coeffs.meta
    _hr("系数文件")
    _out(f"  文件            : {m.get('source_file')}")
    _out(f"  布局            : {m.get('layout')}")
    _out(f"  nmax            : {coeffs.nmax}")
    _out(f"  ntime           : {coeffs.ntime}")
    _out(f"  真值系数个数    : {coeffs.ncoef} = (nmax+1)^2")
    _out(f"  三角布局行数    : {2 * (coeffs.nmax + 1) * (coeffs.nmax + 2) // 2} "
         f"= 2*NC")
    for k in ("norm", "modelname", "earth_gravity_constant", "radius",
              "tide_system", "max_degree_in_file", "has_sigmas"):
        if k in m:
            _out(f"  {k:16s}: {m[k]}")
    for k in ("weight_rule", "method", "weight_sum", "coverage",
              "fit_rmse_rel", "lmax_recommended", "resolution_km",
              "gaussian_radius_km"):
        if k in m:
            _out(f"  {k:16s}: {m[k]}")
    if m.get("warnings"):
        _out("  读取警告        :")
        for w in m["warnings"]:
            _out(f"    - {w}")
    _hr()
    _out(coeffs.summary())
    _hr(f"逐阶 RMS（degrees 0 .. {coeffs.nmax}）")

    rms = np.atleast_2d(coeffs.degree_rms().reshape(1, -1))
    if coeffs.C.ndim == 3:
        C3, S3 = coeffs.C, coeffs.S
        rms = np.zeros((coeffs.nmax + 1, coeffs.ntime))
        for n in range(coeffs.nmax + 1):
            vals = np.concatenate([C3[n, :n + 1, :], S3[n, :n + 1, :]], axis=0)
            rms[n, :] = np.sqrt(np.mean(vals ** 2, axis=0))
        rms = rms.T                       # (ntime, nmax+1)
    power = coeffs.power()
    _out(f"  {'deg':>4}  " + "  ".join(
        f"{('RMS_t%d' % t) if coeffs.ntime > 1 else 'RMS':>14}" for t in range(rms.shape[0]))
        + f"  {'power':>14}  {'res_km':>9}")
    for n in range(coeffs.nmax + 1):
        cells = "  ".join(f"{rms[t, n]:14.6e}" for t in range(rms.shape[0]))
        _out(f"  {n:>4}  {cells}  {power[n]:14.6e}  "
             f"{harmonic_resolution_km(n):9.1f}")
    if coeffs.nmax > 0:
        _out(f"  半波长分辨率 @nmax = {coeffs.nmax}: "
             f"{harmonic_resolution_km(coeffs.nmax):.1f} km"
             f"  (R = 6371 km)")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def _add_cols(p) -> None:
    p.add_argument("--lon-col", default=None, metavar="SPEC",
                   help="经度列的列名或列序号（默认按表头/位置推断：第1列=经度）")
    p.add_argument("--lat-col", default=None, metavar="SPEC",
                   help="纬度列的列名或列序号（默认按表头/位置推断：第2列=纬度）")
    p.add_argument("--val-col", default=None, metavar="SPEC",
                   help="数值列的列名或列序号（默认：表头里的 value/z，否则第3列）")


def _add_rule(p, help_extra: str = "") -> None:
    p.add_argument("--rule", default="auto", choices=list(_RULES),
                   help="权重规则: auto 自动判断；dh 全球等角网格（精确求积）；"
                        "grid 任意网格的精确球带面积；lattice 规则格网上的掩膜"
                        "（每点一个格元，Σw = 点数 × 格元面积）；voronoi 任意散点的"
                        "球面 Voronoi 面积（全球，sum=4pi）；delaunay 不规则区域点集"
                        "（球面凸包内）；uniform 等面积 4pi/N；user 需配合 "
                        "--user-weights。" + help_extra)


def _add_longitude_fft(p) -> None:
    p.add_argument("--longitude-fft", default="auto",
                   choices=["auto", "fft", "direct"], metavar="MODE",
                   help="经度方向加速（只影响速度，不影响结果：两条路径实测相对 "
                        "≤1e-12）：auto（默认）在点集是完整矩形网格、经度整圈均匀"
                        "且 nlon > 2*nmax 时用一次 rfft 代替逐点乘 cos/sin；"
                        "fft 要求快路径（条件不满足时报错，不静默回退）；"
                        "direct 强制通用直接法。选了什么、为什么，报告里都会写。")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` tree (one function so tests can introspect it)."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "SHKit 命令行工具：球谐分析 / 合成 / 权重 / 往返自检。\n"
            "坐标单位为度；经度按 [0,360) 归一化；权重为立体角元（球面度），\n"
            "全球采样满足 sum(w) = 4*pi；系数遵循 4pi 归一化、无 Condon-Shortley 相位。"),
        epilog=(
            "示例:\n"
            "  shkit weights   --points data.csv --rule voronoi --out w.csv\n"
            "  shkit analyze   --points data.csv --nmax 60 --rule voronoi "
            "--out-prefix out/run1 --reconstruct\n"
            "  shkit analyze   --points grid.nc --nmax 60 --rule grid "
            "--out-grid recon.nc\n"
            "  shkit synth     --coeffs model.sh --out-grid out.nc --lat-step 1 "
            "--lon-step 1\n"
            "  shkit synth     --coeffs model.sh --points pts.csv --out synth.csv\n"
            "  shkit roundtrip --points data.csv --nmax 60 --rule voronoi\n"
            "  shkit roundtrip --coeffs model.sh --nmax 40 --rule voronoi "
            "--points data.csv\n"
            "  shkit glq-grid  --lmax 40 --out glq.nc\n"
            "  shkit info      --coeffs model.sh\n"))
    sub = parser.add_subparsers(dest="command", metavar="<子命令>")

    # ------------------------------------------------------------- weights
    w = sub.add_parser(
        "weights", help="计算并打印逐点积分权重（WeightSet 诊断）",
        description="计算样本点的立体角元权重并打印诊断；"
                    "可用 --out 写出逐点权重（列: lon,lat,weight，单位 sr）。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    w.add_argument("--points", required=True, metavar="FILE",
                   help="散点文件（csv/txt/dat/npy/xlsx）。网格文件请改用 analyze")
    _add_cols(w)
    _add_rule(w)
    w.add_argument("--nmax", type=int, default=None,
                   help="目标最大阶数（仅用于提示该采样能否支撑，不参与权重计算）")
    w.add_argument("--out", default=None, metavar="FILE",
                   help="写出逐点权重 csv（列: lon,lat,weight）")
    w.add_argument("--user-weights", default=None, metavar="FILE",
                   help="--rule user 时使用的逐点权重文件（csv/npy，取最后一个数值列）")
    w.set_defaults(func=cmd_weights)

    # ------------------------------------------------------------- analyze
    a = sub.add_parser(
        "analyze", help="球谐分析：散点/网格 -> 系数 + 诊断报告",
        description="读入散点或网格 -> shkit.analysis.analysis() -> 打印 "
                    "AnalysisReport -> 落盘三角系数、JSON/MD 报告，"
                    "可选重建场写回。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--points", required=True, metavar="FILE",
                   help="输入数据：散点文件（csv/txt/dat/npy/xlsx）"
                        "或网格文件（nc/grd/cdf）")
    a.add_argument("--var", default=None, metavar="NAME",
                   help="网格文件的变量名（nc/grd；默认取第一个含 lat/lon 的变量）")
    _add_cols(a)
    a.add_argument("--nmax", type=int, required=True,
                   help="求解的最大阶数（未知数 (nmax+1)^2 个）")
    a.add_argument("--method", default="auto", choices=list(_METHODS),
                   help="估计算法: auto 自动；quadrature 加权投影（快，需好的求积规则）；"
                        "iterative 投影+Richardson 修正；projection 零填充全球投影"
                        "（区域数据的目标 A：min ∫_Ω(f-Ax)²dΩ，f 区域外为 0；"
                        "残差即截断泄漏，C00 = 面积占比）；wlsq 加权最小二乘"
                        "（仅适用于全球覆盖但采样不规则的数据，区域数据上未正则化必崩）；"
                        "cg 矩阵无关 CG")
    _add_rule(a)
    a.add_argument("--tau", type=float, default=None, metavar="LAMBDA",
                   help="仅 projection：开启采样偏差校正，只对集中因子 >= LAMBDA 的 "
                        "Slepian 方向除以 λ（默认关闭 = 纯投影）")
    a.add_argument("--niter", type=int, default=None,
                   help="iterative 的迭代次数（默认 1）")
    _add_longitude_fft(a)
    a.add_argument("--reg", default=None, choices=list(_REGS),
                   help="正则化: kaula（相对 Kaula 律）或 tikhonov（单位阵阻尼）")
    a.add_argument("--alpha", type=float, default=None,
                   help="正则化强度；不给了 reg 时默认按 L 曲线自动选")
    a.add_argument("--sigma-col", default=None, metavar="SPEC",
                   help="逐点观测标准差所在列（仅散点 csv；最小二乘行权重变成 w/sigma^2）")
    a.add_argument("--field-unit", default="scalar", choices=list(_FIELD_UNITS),
                   metavar="UNIT",
                   help="【正变换公式】输入网格/散点是什么物理量，决定 "
                        "C_nm = a_nm / f_u 里的 f_u：geoid -> /R、EWH -> /A_n、"
                        "面密度 -> /(A_n*rho_w)、径向形变 -> /(R*h'/(1+k'))；"
                        "scalar 时 f_u=1，导出的就是该场自身的系数。"
                        "换一个声明，导出的 C_nm 就跟着变。 可选: "
                        + ", ".join(_FIELD_UNITS))
    a.add_argument("--target-unit", "--output-unit", default=None,
                   dest="target_unit", choices=list(_FIELD_UNITS), metavar="UNIT",
                   help="【反变换公式】重建场要换成哪个物理量，决定 "
                        "场 = C_nm * f_t 里的 f_t。它不改变导出的系数"
                        "（导出的永远是经典无量纲位系数 C_nm），只影响重建；"
                        "默认与 --field-unit 相同（正反抵消，往返精确）。"
                        "面密度与 EWH 之间只差常数 rho_w。")
    a.add_argument("--user-weights", default=None, metavar="FILE",
                   help="--rule user 时的逐点权重文件")
    a.add_argument("--out-prefix", default=None, metavar="PATH",
                   help="结果前缀，如 out/run1；将落盘 out/run1.sh、"
                        "out/run1.json、out/run1.md、out/run1.npz 等")
    a.add_argument("--out-grid", default=None, metavar="FILE",
                   help="重建场输出：散点输入写 csv；网格输入写 nc/grd")
    a.add_argument("--gaussian-km", type=float, default=None, metavar="KM",
                   help="高斯平滑半径（0.5 幅度半宽，km）；仅影响重建/合成，"
                        "不改变求解出的系数")
    a.add_argument("--reconstruct", action="store_true",
                   help="写出**重建场**（与输入同物理量，可直接与原数据比对）："
                        "散点写 csv，网格写 nc（配合 --out-prefix）")
    a.add_argument("--out-field", default=None, metavar="FILE",
                   help="写出**输出场**（按 --target-unit 选的物理量，"
                        "逐阶乘 f_t）。与 --out-grid 同时给就写两份。")
    a.set_defaults(func=cmd_analyze)

    # --------------------------------------------------------------- synth
    s = sub.add_parser(
        "synth", help="球谐合成：系数 -> 规则网格或指定点上的场",
        description="读入系数文件，在规则网格（--out-grid，配合 --lat-step/"
                    "--lon-step）或给定点（--points）上合成场并写出。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("--coeffs", required=True, metavar="FILE",
                   help="系数文件：.sh/三角 txt、gmfcsv、.gfc、.npy、.npz")
    s.add_argument("--points", default=None, metavar="FILE",
                   help="只在给定散点上合成，并把结果写成 csv")
    s.add_argument("--out", default=None, metavar="FILE",
                   help="--points 模式的输出文件（默认 <points 名>_synth.csv）")
    s.add_argument("--out-grid", default=None, metavar="FILE",
                   help="网格模式输出文件（nc/grd/csv/txt/npy）")
    s.add_argument("--lat-step", type=float, default=1.0, metavar="DEG",
                   help="纬度步长（度，默认 1；网格范围 -90..90）")
    s.add_argument("--lon-step", type=float, default=1.0, metavar="DEG",
                   help="经度步长（度，默认 1；网格范围 0..360）")
    s.add_argument("--var", default="value", metavar="NAME",
                   help="输出网格的变量名（默认 value）")
    s.add_argument("--nmax", type=int, default=None,
                   help="只用到该阶（默认用文件里的全部阶；会截断更高阶）")
    s.add_argument("--layout", default="auto",
                   choices=["auto", "triangle", "matrix", "gmfcsv", "npy", "gfc"],
                   help="强制系数布局（默认 auto 由文件推断）")
    s.add_argument("--gaussian-km", type=float, default=None, metavar="KM",
                   help="高斯平滑半径（km），按阶加权（不改写系数文件）")
    _add_longitude_fft(s)
    _add_cols(s)
    s.set_defaults(func=cmd_synth)

    # ----------------------------------------------------------- roundtrip
    r = sub.add_parser(
        "roundtrip", help="往返自检：analysis -> synthesis -> 打印误差",
        description="对同一批点做分析再合成，打印相对 RMS 误差；"
                    "若同时给出 --coeffs（真实系数），还会打印系数恢复误差。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    r.add_argument("--points", default=None, metavar="FILE",
                   help="散点文件；配合 --coeffs 时可省略，此时自动用 Fibonacci 球面采样")
    r.add_argument("--coeffs", default=None, metavar="FILE",
                   help="可选：真实系数文件（用于给出系数恢复误差）")
    _add_cols(r)
    r.add_argument("--nmax", type=int, required=True, help="求解的最大阶数")
    _add_rule(r)
    r.add_argument("--method", default="auto", choices=list(_METHODS),
                   help="估计算法（同 analyze）")
    r.add_argument("--niter", type=int, default=None,
                   help="iterative 的迭代次数")
    r.add_argument("--user-weights", default=None, metavar="FILE",
                   help="--rule user 时的逐点权重文件")
    r.add_argument("--npoints", type=int, default=3000, metavar="N",
                   help="未给 --points 时自动生成的 Fibonacci 点数（默认 3000）")
    r.add_argument("--seed", type=int, default=0,
                   help="Fibonacci 采样的小扰动种子（默认 0，保证可复现）")
    r.add_argument("--var", default=None, help="网格输入时的变量名")
    r.add_argument("--layout", default="auto",
                   choices=["auto", "triangle", "matrix", "gmfcsv", "npy", "gfc"],
                   help="--coeffs 的布局（默认 auto）")
    r.add_argument("--out", default=None, metavar="FILE",
                   help="把诊断报告写成 json/md")
    r.set_defaults(func=cmd_roundtrip)

    # ---------------------------------------------------------- glq-grid
    g = sub.add_parser(
        "glq-grid", help="生成精确求积参考网格（Gauss-Legendre x 等间隔经度）",
        description="生成 lmax 阶带限场可被*精确*恢复的参考网格，"
                    "并把每点立体角元写成网格文件的 weight 变量。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g.add_argument("--lmax", type=int, required=True,
                   help="目标最大阶数（纬度节点数 = lmax+1，经度 = 2*lmax+1）")
    g.add_argument("--out", required=True, metavar="FILE",
                   help="输出文件（nc/grd/csv/txt/npy）")
    g.set_defaults(func=cmd_glq_grid)

    # --------------------------------------------------------------- info
    i = sub.add_parser(
        "info", help="查看系数文件：summary + 逐阶 RMS 表",
        description="打印系数文件的元信息、summary() 以及逐阶 RMS / power / "
                    "对应半波长分辨率。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    i.add_argument("--coeffs", required=True, metavar="FILE", help="系数文件")
    i.add_argument("--nmax", type=int, default=None,
                   help="只读到该阶（默认按文件推断）")
    i.add_argument("--layout", default="auto",
                   choices=["auto", "triangle", "matrix", "gmfcsv", "npy", "gfc"],
                   help="强制布局（默认 auto）")
    i.set_defaults(func=cmd_info)

    # --------------------------------------------------------------- nmax
    n = sub.add_parser(
        "nmax", help="该用多少阶？三个独立约束 + 截断误差表",
        description="读系数文件（或跑一遍分析），打印：\n"
                    "  ① 截断：每个 nmax 的保留功率比与重建误差 "
                    "sqrt(1-保留比)（Parseval 精确，与采样密度无关）\n"
                    "  ② 采样/覆盖：recommend_lmax\n"
                    "  ③ 区域：Shannon 数 (L+1)^2*coverage，以及使 Shannon≈1 的阶数\n"
                    "覆盖比与采样点数默认从系数文件的元信息读取。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    n.add_argument("--coeffs", metavar="FILE", help="系数文件（与 --points 二选一）")
    n.add_argument("--points", metavar="FILE",
                   help="数据文件：先跑一遍分析再诊断（配合 --nmax/--rule/--method）")
    n.add_argument("--nmax", type=int, default=None,
                   help="--points 模式下的分析阶数（默认 60）；--coeffs 模式下"
                        "只读到该阶")
    n.add_argument("--layout", default="auto",
                   choices=["auto", "triangle", "matrix", "gmfcsv", "npy", "gfc"],
                   help="系数文件布局（默认 auto）")
    n.add_argument("--var", default=None, metavar="NAME",
                   help="--points 为网格文件时的变量名")
    n.add_argument("--tol", type=float, default=0.01,
                   help="目标重建相对误差（默认 0.01 = 1%%），用来给"
                        "「① 需要多少阶」")
    n.add_argument("--coverage", type=float, default=None,
                   help="覆盖比 sum(w)/4pi；默认从系数文件元信息读")
    n.add_argument("--npoints", type=int, default=None,
                   help="采样点数；默认从系数文件元信息读")
    n.add_argument("--list", default=None, metavar="N1,N2,...",
                   help="只在指定的这些阶数上列截断表")
    _add_rule(n)
    n.add_argument("--method", default="auto", choices=list(_METHODS),
                   help="--points 模式的估计算法（默认 auto）")
    n.add_argument("--field-unit", default="scalar", choices=list(_FIELD_UNITS),
                   metavar="UNIT", help="--points 模式的输入物理量（默认 scalar）")
    _add_cols(n)
    n.set_defaults(func=cmd_nmax)

    # --------------------------------------------------------- series-read
    s = sub.add_parser(
        "series-read", help="目录/清单 → 多历元系数序列（series_nc）",
        description="把一批单历元系数文件（GRACE GSM 的 .gfc / 无扩展名 SHM）"
                    "读成**一条**多历元序列，并落盘。\n"
                    "时间来自 gfc 头部（优先）或文件名；缺测/重复/被筛掉的文件"
                    "一律打印，不静默丢弃。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("--source", required=True, metavar="PATH",
                   help="目录 / 通配 / 清单 .txt / 单个文件，或逗号分隔的路径列表")
    s.add_argument("--pattern", default=None,
                   help="--source 是目录时的过滤模式（默认 *）")
    s.add_argument("--product", default="GSM",
                   help="允许的产品码，逗号分隔（默认 GSM；目录里的其它产品会被筛掉）")
    s.add_argument("--mission", default=None, help="任务过滤，逗号分隔（GRAC,GRFO）")
    s.add_argument("--center", default=None, help="中心过滤，逗号分隔（UTCSR,GFZOP,JPLEM）")
    s.add_argument("--rl", default=None, help="版本过滤，逗号分隔（0600,0603）")
    s.add_argument("--epoch-from", default="auto",
                   choices=["auto", "header", "filename", "index"],
                   help="历元时间来源（默认 auto：头部优先，缺则文件名）")
    s.add_argument("--sort", default="time", choices=["time", "name", "none"])
    s.add_argument("--on-duplicate", default="error",
                   choices=["error", "first", "last", "report"],
                   help="同一历元两个文件怎么办（默认报错）")
    s.add_argument("--on-missing", default="report", choices=["report", "ignore"])
    s.add_argument("--nmax", type=int, default=None, help="截断到该阶（默认取各文件最大）")
    s.add_argument("--no-strict-nmax", action="store_true",
                   help="各文件阶数不一致时统一到最高阶（零填充）而不是报错")
    s.add_argument("--list", default=None, metavar="FILE",
                   help="把逐历元清单（时间/来源文件/警告）写到该文件")
    s.add_argument("--out", required=True, metavar="FILE",
                   help="输出序列文件（.nc 为 series_nc；.dat 为 legacy 三角表）")
    s.add_argument("--summary", action="store_true", help="打印时间轴摘要")
    s.set_defaults(func=cmd_series_read)

    # ------------------------------------------------------ analyze-series
    b = sub.add_parser(
        "analyze-series", help="多历元数据 → 一条系数序列（批量，权重只算一次）",
        description="把带时间轴的网格/散点**一次**解成多历元系数序列。\n"
                    "与逐历元循环 `analyze` 相比：权重与 Gram 诊断只算一次，"
                    "网格上还会自动走经度 FFT。1°全球/nmax=60 的 203 个历元实测："
                    "逐历元循环 ~13 s → 批量直接法 5.1 s → 批量 + 经度 FFT 0.5 s"
                    "（加 --report-fit 求逐历元残差表则约 9 s，瓶颈转到综合）。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    b.add_argument("--points", required=True, metavar="FILE",
                   help="数据文件：多时次网格（nc，带 time 维）或散点表（多数值列=多时次）")
    b.add_argument("--var", default=None, help="网格变量名（默认自动）")
    b.add_argument("--nmax", type=int, required=True, help="最大阶数")
    b.add_argument("--epochs", default="all",
                   help="历元范围：all（默认）或 i0:i1（0 起始，含 i0 不含 i1）")
    b.add_argument("--epoch-chunk", type=int, default=None,
                   help="每次调用最多解多少历元（默认一次全解；只影响内存与求和顺序）")
    b.add_argument("--report-fit", action="store_true",
                   help="额外算逐历元残差（一次向量化综合，约占一次调用 45%%，"
                        "换残差表与离群检测）")
    b.add_argument("--field-unit", default="scalar", choices=list(_FIELD_UNITS))
    b.add_argument("--out-series", required=True, metavar="FILE",
                   help="系数序列输出（.nc 为 series_nc）")
    b.add_argument("--out-summary", default=None, metavar="FILE",
                   help="逐历元诊断表输出（csv）")
    b.add_argument("--reconstruct", default=None, metavar="FILE",
                   help="可选：把重建场写成多时次网格 nc")
    _add_longitude_fft(b)
    _add_rule(b)
    _add_cols(b)
    b.set_defaults(func=cmd_analyze_series)

    # ------------------------------------------------------------- timefit
    tf = sub.add_parser(
        "timefit", help="时间域拟合：常数 / 趋势 / 周年（可含半年）",
        description="对一条**带时间轴的系数序列**做最小二乘时间拟合。\n"
                    "拟合坐标是「距跨度中点的真实经过年数 / 365.25」，不是 legacy 小数年；"
                    "历元不等间隔、有缺测都按真实时间处理（缺测历元被排除并报告，"
                    "不会当成 0 值）。输出每项的系数文件 + 逐历元残差表。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    tf.add_argument("--series", required=True, metavar="FILE",
                    help="系数序列：series_nc（.nc）或 legacy 三角表（.dat，"
                         "需同目录有 <名字>_TimeInfo.dat）")
    tf.add_argument("--nmax", type=int, default=None, help="截断到该阶")
    tf.add_argument("--poly-order", type=int, default=1,
                    help="多项式阶数：0=只拟合常数，1=加线性趋势（默认），2=再加二次项")
    tf.add_argument("--periods", default="1.0",
                    help="周期项（年），逗号分隔：默认 1.0（年周期）；"
                         "1.0,0.5 表示年周期 + 半年周期；none 表示只拟合多项式")
    tf.add_argument("--weights-file", default=None, metavar="FILE",
                    help="逐历元权重（每行一个数，个数必须等于历元数）")
    tf.add_argument("--out-prefix", default=None, metavar="PATH",
                    help="每项写一个 <前缀>_<项名>.sh（const/trend/annual_cos…）")
    tf.add_argument("--out-residual", default=None, metavar="FILE",
                    help="残差序列（series_nc）")
    tf.add_argument("--out-model", default=None, metavar="FILE",
                    help="模型在采样历元上的值（series_nc）")
    tf.add_argument("--out-csv", default=None, metavar="FILE",
                    help="逐历元残差表 csv")
    tf.set_defaults(func=cmd_timefit)

    # --------------------------------------------------------- time-filter
    tg = sub.add_parser(
        "time-filter", help="时间维高斯平滑（按真实经过时间，不是按序号）",
        description="沿**真实经过时间**做高斯平滑：每个输出历元是窗口内的加权平均。\n"
                    "⚠️ 这是平滑器而不是截止频率明确的低通：历元不等间隔时核不是"
                    "位移不变的，会同时轻微改变振幅与相位（报告里会写明）。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    tg.add_argument("--series", required=True, metavar="FILE")
    tg.add_argument("--nmax", type=int, default=None)
    tg.add_argument("--sigma", type=float, required=True, metavar="YEARS",
                    help="高斯核 sigma（年）")
    tg.add_argument("--truncate", type=float, default=4.0, metavar="SIGMA",
                    help="窗口截断倍数（默认 4σ）")
    tg.add_argument("--out", required=True, metavar="FILE",
                    help="平滑后的序列（.nc 为 series_nc，.dat 为 legacy 三角表）")
    tg.set_defaults(func=cmd_time_filter)

    # ------------------------------------------------------- C2 产品子命令
    sg = sub.add_parser(
        "series-grid", help="系数序列 → 多时次场 nc（与 3_grids/*.nc 同构）",
        description="把一条系数序列合成成**多时次网格场**，落盘为 (time, lat, lon) 的 nc："
                    "lat 降序、lon [-180,180)、float32，与用户 3_grids/*.nc 同构。\n"
                    "⚠️ GRACE GSM 的 C00 = 1（地球总质量），做 EWH 异常要加 --drop-c00，"
                    "通常还要 --remove-time-mean 才是参考场那个口径。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_product_args(sg)
    sg.add_argument("--lat-step", type=float, default=1.0, metavar="DEG",
                    help="纬度步长（默认 1；范围 -90..90）")
    sg.add_argument("--lon-step", type=float, default=1.0, metavar="DEG",
                    help="经度步长（默认 1；范围 0..360）")
    sg.add_argument("--var", default="ewh", metavar="NAME", help="变量名（默认 ewh）")
    sg.add_argument("--out", required=True, metavar="FILE", help="输出 .nc")
    sg.set_defaults(func=cmd_series_grid)

    sp = sub.add_parser(
        "series-points", help="系数序列 → 指定点上的时间序列（csv）",
        description="在给定点上合成时间序列，落盘为宽表 csv（行=历元，列=点）。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_product_args(sp)
    sp.add_argument("--points", required=True, metavar="FILE",
                    help="点位文件（csv/txt：lon,lat[,...]，与 analyze 同一套列推断）")
    _add_cols(sp)
    sp.add_argument("--out", required=True, metavar="FILE", help="输出 csv")
    sp.set_defaults(func=cmd_series_points)

    sb = sub.add_parser(
        "basin-average", help="系数序列 → 区域面积加权平均时间序列（csv）",
        description="对给定点集（就是一个区域掩膜）做**面积加权**平均，每个历元一个数。\n"
                    "输出头里写明覆盖了球面的百分之几 —— 区域平均是覆盖面积内的平均，"
                    "不写覆盖率的话这个数没法解释。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_product_args(sb)
    sb.add_argument("--points", required=True, metavar="FILE",
                    help="区域点集（csv/txt：lon,lat[,...]；每一行就是一个格点）")
    _add_cols(sb)
    sb.add_argument("--out", required=True, metavar="FILE", help="输出 csv")
    sb.set_defaults(func=cmd_basin_average)
    return parser


def cmd_analyze_series(args) -> int:
    """Batch-analyse a multi-epoch data set."""
    from . import io as shio
    from .series import analyze_series

    lat, lon, values, meta, grid = _load_field(args.points, args.var,
                                               args.lon_col, args.lat_col,
                                               args.val_col)
    ntime = 1 if values.ndim == 1 else values.shape[1]
    _out(f"输入       : {meta['source_file']}  ({meta.get('input_kind')})")
    _out(f"点数       : {lat.size}   历元: {ntime}")

    # ---- 时间轴（有就用，没有就不编） -----------------------------------
    times = None
    tmeta = meta.get("time") or meta.get("times")
    if tmeta is not None and len(tmeta) == ntime:
        from .timeaxis import time_axis_from_meta
        times = time_axis_from_meta(tmeta, units=meta.get("time_units"),
                                    calendar=meta.get("time_calendar"))
        if times is not None and times.has_dates:
            _out(f"时间轴     : {times.meta.get('time_source')}，"
                 f"{str(times.values[0])[:10]} … {str(times.values[-1])[:10]}")
        else:
            times = None
    if times is None:
        _out("时间轴     : 无（不编日期；后续按日期操作会报错）")

    # ---- 历元切片 --------------------------------------------------------
    sl = slice(None)
    if args.epochs and args.epochs != "all":
        try:
            a, b_ = args.epochs.split(":")
            sl = slice(int(a) if a else None, int(b_) if b_ else None)
        except ValueError as exc:
            raise CliError(f"--epochs 需要 all 或 i0:i1，得到 {args.epochs!r}") from exc
        values = values[..., sl] if values.ndim > 1 else values
        if times is not None:
            times = times.select(range(*sl.indices(ntime)))
        _out(f"历元范围   : {args.epochs}")

    ws = _make_weights(lat, lon, args.rule, None,
                       nlon=meta.get("nlon") if meta.get("input_kind") == "grid"
                       else None)
    coeffs, rep = analyze_series(
        lat, lon, values, args.nmax, times=times, weights=ws,
        epoch_chunk=args.epoch_chunk, report_fit=args.report_fit,
        method="quadrature", field_unit=args.field_unit,
        longitude_fft=args.longitude_fft,
        progress=lambda msg, frac: None)
    _hr("SeriesReport")
    _out(rep.summary())

    shio.write_coeffs(coeffs, args.out_series, layout="series_nc")
    _hr("落盘")
    _out(f"  系数序列  : {args.out_series}")
    if args.out_summary:
        rep.to_csv(args.out_summary)
        _out(f"  逐历元诊断: {args.out_summary}")
    if args.reconstruct:
        from .synthesis import synthesis_grid
        cube = synthesis_grid(meta["grid"][0], meta["grid"][1], coeffs)
        shio.write_grid(args.reconstruct, meta["grid"][0], meta["grid"][1], cube,
                        var="reconstructed")
        _out(f"  重建场序列: {args.reconstruct}")
    return 0


def _read_any_series(path: str, nmax=None):
    """Read a coefficient **series** in any of the formats we write.

    ``.nc``/``.npz`` go through the generic reader (a ``series_nc`` file is a
    coefficient file like any other); a legacy ``.dat`` needs its explicit
    ``layout`` **and** its companion ``<名字>_TimeInfo.dat`` next to it, which is
    the only place the epoch dates live in that format.
    """
    from . import io as shio

    low = str(path).lower()
    if low.endswith(".dat"):
        if not os.path.exists(str(path)[:-4] + "_TimeInfo.dat"):
            raise CliError(
                f"{os.path.basename(path)} 是 legacy 三角表，里面没有历元日期；"
                f"同目录还需要 {os.path.basename(str(path)[:-4])}_TimeInfo.dat。"
                "若手上只有 .dat，请先用 series-read 从 gfc 目录重建带时间轴的 .nc。")
        return shio.read_coeffs(path, nmax=nmax, layout="legacy_dat")
    if low.endswith((".nc", ".npz")):
        return shio.read_coeffs(path, nmax=nmax)
    return shio.read_coeffs_series(path, nmax=nmax, strict_nmax=False,
                                   progress=lambda msg, frac: None)


def cmd_timefit(args) -> int:
    """Fit a time model (trend / seasonal) to a coefficient series."""
    from . import io as shio
    from .timeseries import fit_time_model

    coeffs = _read_any_series(args.series, args.nmax)
    _out(f"输入序列   : {args.series}   nmax={coeffs.nmax}  历元={coeffs.ntime}")
    if not coeffs.has_time:
        raise CliError(
            "这条序列没有时间轴，无法做时间域拟合。"
            "若读的是 .dat，请确保同目录有 <名字>_TimeInfo.dat；"
            "或用 series-read 从 gfc 目录重建带时间轴的 .nc。")

    raw = str(args.periods).strip().lower()
    periods = (() if raw in ("", "none", "-", "0")
               else tuple(float(x) for x in raw.split(",") if x.strip()))
    fit = fit_time_model(coeffs, poly_order=args.poly_order, periods=periods,
                         weights=(None if args.weights_file is None
                                  else np.loadtxt(args.weights_file)))
    _hr("TimeFit")
    _out(fit.summary())

    prefix = args.out_prefix
    if prefix:
        _hr("落盘")
        for nm, tm in zip(fit.names, fit.terms):
            p = f"{prefix}_{nm}.sh"
            shio.write_coeffs(tm, p, layout="triangle")
            _out(f"  {nm:16s}: {p}")
        if args.out_residual:
            shio.write_coeffs(fit.residual, args.out_residual, layout="series_nc")
            _out(f"  残差序列        : {args.out_residual}")
        if args.out_model:
            shio.write_coeffs(fit.model, args.out_model, layout="series_nc")
            _out(f"  模型序列        : {args.out_model}")
        if args.out_csv:
            _write_fit_csv(fit, args.out_csv)
            _out(f"  逐历元残差      : {args.out_csv}")
    return 0


def _write_fit_csv(fit, path: str) -> None:
    import csv
    vals = fit.model.times.values if fit.model.times is not None else None
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "time", "residual_rms", "used"])
        for i in range(fit.coord.size):
            w.writerow([i,
                        "" if vals is None else str(vals[i]),
                        "" if not np.isfinite(fit.rms_per_epoch[i])
                        else f"{fit.rms_per_epoch[i]:.10g}",
                        0 if i in set(fit.excluded.tolist()) else 1])


def cmd_time_filter(args) -> int:
    """Gaussian smoothing along the time axis of a coefficient series."""
    from . import io as shio
    from .timeseries import time_gaussian_filter

    coeffs = _read_any_series(args.series, args.nmax)
    rep: dict = {}
    out = time_gaussian_filter(coeffs, args.sigma, truncate=args.truncate,
                               report=rep)
    _hr("TimeFilter")
    _out(f"输入序列   : {args.series}   nmax={coeffs.nmax}  历元={coeffs.ntime}")
    eff = rep["effective_epochs"]
    _out(f"sigma      : {rep['sigma_years']:g} 年（截断 {rep['truncate']:g}σ）")
    _out(f"有效历元数 : 中位 {np.median(eff):.2f}  最小 {eff.min():.2f}  "
         f"最大 {eff.max():.2f}")
    _out(f"边界历元   : {rep['edge_epochs']}")
    for n in rep["notes"]:
        _out(f"提示       : {n}")
    layout = "legacy_dat" if str(args.out).lower().endswith(".dat") else "series_nc"
    shio.write_coeffs(out, args.out, layout=layout)
    _hr("落盘")
    _out(f"  平滑序列  : {args.out}  (layout={layout})")
    return 0


def _add_product_args(p) -> None:
    """Options shared by the C2 product subcommands (series-points/basin/series-grid)."""
    p.add_argument("--series", required=True, metavar="FILE",
                   help="系数序列：series_nc（.nc）或 legacy 三角表（.dat，"
                        "需同目录有 <名字>_TimeInfo.dat）")
    p.add_argument("--nmax", type=int, default=None, help="截断到该阶")
    p.add_argument("--field-unit", default=None, metavar="UNIT",
                   choices=list(_FIELD_UNITS),
                   help="**声明这串系数是什么**（正变换的起点）。legacy .dat 里不带这个信息，"
                        "所以要做 EWH/geoid 换算就必须声明（通常是 geopotential）")
    p.add_argument("--target-unit", default=None, metavar="UNIT",
                   choices=list(_FIELD_UNITS),
                   help="要输出成什么物理量（反变换公式）。默认不换算")
    p.add_argument("--gaussian-km", type=float, default=0.0, metavar="KM",
                   help="高斯平滑半径（km，按阶加权；参考场 G300 对应 300）")
    p.add_argument("--drop-c00", action="store_true",
                   help="把 C00 置零。GRACE GSM 的 C00 恒为 1（地球总质量），"
                        "不置零的话 EWH 会得到 ~1.2e7 m 这种数")
    p.add_argument("--remove-time-mean", action="store_true",
                   help="减去时间平均，得到「相对时间均值的异常」"
                        "（参考场 3_grids/*.nc 就是这个口径）")
    p.add_argument("--out-units", default="m", choices=["m", "mm"],
                   help="输出单位：m（默认）或 mm（×1000，参考场用 mm）")
    p.add_argument("--longitude-fft", default="auto",
                   choices=["auto", "fft", "direct"], metavar="MODE",
                   help="经度方向加速（只影响速度，不影响结果）")


def _prepare_series(args):
    """Load a series and apply the shared C2 pre-processing options.

    Returns ``(coeffs, notes)``; every transformation is named in ``notes`` so the
    printed summary shows exactly which conventions were applied before comparing
    with anybody else's product.
    """
    from . import units as shunits

    co = _read_any_series(args.series, args.nmax)
    notes = [f"输入序列   : {args.series}   nmax={co.nmax}  历元={co.ntime}"]
    if args.field_unit:
        co = shunits.with_field_unit(co, args.field_unit)
        notes.append(f"声明为     : {args.field_unit}（正变换的起点）")
    notes.append(f"系数单位   : {co.meta.get('field_unit') or '未声明'}")
    if args.drop_c00:
        C = np.asarray(co.C, dtype=float).copy()
        c00_was = float(C[0, 0, 0]) if C.ndim == 3 else float(C[0, 0])
        C[0, 0, ...] = 0.0
        co = SHCoeffs(C, np.asarray(co.S, dtype=float).copy(), dict(co.meta),
                      co.times)
        notes.append(f"C00 置零   : 原值 {c00_was:g}"
                     "（GSM 的 C00 = 1 是地球总质量，EWH 异常必须去掉；"
                     "若原值不是 1，说明这串数本来就是异常，置零会引入偏移）")
    if args.remove_time_mean:
        from .timeseries import remove_time_mean
        co, mean = remove_time_mean(co)
        notes.append("去时间均值 : 已减（→「相对时间均值的异常」）")
    if args.out_units == "mm":
        notes.append("输出单位   : mm（数值 ×1000）")
    return co, notes


def _scale_out(args):
    return 1000.0 if args.out_units == "mm" else 1.0


def cmd_series_grid(args) -> int:
    """Write a field-series netCDF (the C2 grid product)."""
    from . import io as shio
    from .timeseries import series_grid

    co, notes = _prepare_series(args)
    for n in notes:
        _out(n)
    lat = np.arange(-90.0, 90.0 + 1e-9, args.lat_step)
    lon = np.arange(0.0, 360.0 - 1e-9, args.lon_step)
    rep: dict = {}
    cube, times = series_grid(co, lat, lon, target_unit=args.target_unit,
                              gaussian_km=args.gaussian_km,
                              longitude_fft=args.longitude_fft, report=rep)
    cube = cube * _scale_out(args)
    _out(f"网格       : {lat.size} x {lon.size} x {cube.shape[2]}")
    _out(f"经度路径   : {rep.get('longitude_path')} —— {rep.get('longitude_reason')}")
    _out(f"数值范围   : [{np.nanmin(cube):.6g}, {np.nanmax(cube):.6g}] {args.out_units}")

    units = ("mm equivalent water height" if (args.out_units == "mm"
                                              and args.target_unit == "ewh")
             else ("m equivalent water height" if args.target_unit == "ewh"
                   else args.out_units))
    meta = {"title": f"SHKit field series ({args.target_unit or 'no conversion'})",
            "lmax": co.nmax,
            "gauss_filter_km": float(args.gaussian_km),
            "grid_resolution_deg": float(args.lat_step),
            "units": units,
            "long_name": (f"{args.target_unit or 'field'} anomaly"
                          if args.remove_time_mean else (args.target_unit or "field")),
            "source_coeffs": str(args.series),
            "time_source": (None if times is None else times.meta.get("time_source")),
            "c00_dropped": bool(args.drop_c00),
            "time_mean_removed": bool(args.remove_time_mean)}
    shio.write_field_series(args.out, lat, lon, times, cube, var=args.var,
                            units=units, long_name=meta["long_name"], meta=meta)
    _hr("落盘")
    _out(f"  场序列    : {args.out}   （(time, lat, lon)，lat 降序、lon [-180,180)，"
         "与 3_grids/*.nc 同构）")
    return 0


def cmd_series_points(args) -> int:
    """Point time series from a coefficient series (the C2 point product)."""
    from .timeseries import series_at_points

    co, notes = _prepare_series(args)
    lat, lon, _v, pmeta = _load_points(args.points, args.lon_col,
                                       args.lat_col, args.val_col)
    for n in notes:
        _out(n)
    vals, times = series_at_points(co, lat, lon, target_unit=args.target_unit,
                                   gaussian_km=args.gaussian_km)
    vals = vals * _scale_out(args)
    _out(f"点数       : {lat.size}   历元: {vals.shape[1]}")

    labels = [f"{la:g}_{lo:g}" for la, lo in zip(lat, lon)]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# shkit series-points  units={args.out_units}"
                 f" target_unit={args.target_unit} gaussian_km={args.gaussian_km}"
                 f" c00_dropped={bool(args.drop_c00)}"
                 f" time_mean_removed={bool(args.remove_time_mean)}\n")
        for n in notes:
            fh.write(f"# {n}\n")
        fh.write("time," + ",".join(labels) + "\n")
        tv = (times.values if times is not None
              else np.arange(vals.shape[1]).astype("datetime64[s]"))
        for k in range(vals.shape[1]):
            fh.write(str(tv[k]) + "," +
                     ",".join(f"{v:.10g}" for v in vals[:, k]) + "\n")
    _hr("落盘")
    _out(f"  点序列    : {args.out}   形状 {vals.shape}（行=历元，列=点）")
    _hr("TimeAxis")
    if times is not None:
        _out(times.summary())
    else:
        _out("（这条序列没有时间轴，输出第一列是序号）")
    return 0


def cmd_basin_average(args) -> int:
    """Area-weighted regional average time series (the C2 basin product)."""
    from .timeseries import basin_average

    co, notes = _prepare_series(args)
    lat, lon, _v, pmeta = _load_points(args.points, args.lon_col,
                                       args.lat_col, args.val_col)
    _out(notes[0])
    avg, info = basin_average(co, lat, lon, None, target_unit=args.target_unit,
                              gaussian_km=args.gaussian_km)
    avg = avg * _scale_out(args)
    for n in notes[1:]:
        _out(n)
    _hr("区域平均")
    _out(f"点数       : {info['n_points']}/{info['n_total']}")
    _out(f"覆盖       : {info['coverage'] * 100:.6g}% 的球面"
         "（区域平均是**覆盖面积内**的平均，不是全球平均）")
    _out(f"权重规则   : {info['weight_rule']}   sum(w) = {info['weight_sum']:.6g} sr")
    _out(f"纬经范围   : lat [{info['lat_range'][0]:.2f}, {info['lat_range'][1]:.2f}]"
         f"  lon [{info['lon_range'][0]:.2f}, {info['lon_range'][1]:.2f}]")
    _out(f"数值范围   : [{np.nanmin(avg):.6g}, {np.nanmax(avg):.6g}] {args.out_units}")
    if info.get("note"):
        _out(f"提示       : {info['note']}")

    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# shkit basin-average  points={pmeta.get('source_file')}"
                 f"  n_points={info['n_points']}/{info['n_total']}"
                 f"  coverage={info['coverage']:.8g}"
                 f"  weight_rule={info['weight_rule']}"
                 f"  units={args.out_units}  target_unit={args.target_unit}"
                 f"  gaussian_km={args.gaussian_km}\n")
        fh.write("time,value\n")
        tv = (co.times.values if co.times is not None
              else np.arange(avg.size).astype("datetime64[s]"))
        for k in range(avg.size):
            fh.write(f"{tv[k]},{avg[k]:.10g}\n")
    _hr("落盘")
    _out(f"  区域平均序列: {args.out}")
    return 0


def cmd_series_read(args) -> int:
    """Read a directory/manifest of single-epoch files into one series."""
    from . import io as shio

    src = args.source
    if "," in src and not os.path.exists(src):
        src = [x.strip() for x in src.split(",") if x.strip()]

    def _csv(v):
        return None if v is None else [x.strip() for x in str(v).split(",") if x.strip()]

    _out(f"来源       : {args.source}")
    coeffs = shio.read_coeffs_series(
        src, nmax=args.nmax, pattern=args.pattern,
        product=_csv(args.product) or None,
        mission=_csv(args.mission), center=_csv(args.center), rl=_csv(args.rl),
        epoch_from=args.epoch_from, sort=args.sort,
        on_duplicate=args.on_duplicate, on_missing=args.on_missing,
        strict_nmax=not args.no_strict_nmax,
        progress=lambda msg, frac: None)
    _out(f"历元数     : {coeffs.ntime}   阶数: {coeffs.nmax}")
    _out(f"时间来源   : {coeffs.meta.get('time_source')}")
    if coeffs.meta.get("coverage_start"):
        _out(f"时间跨度   : {coeffs.meta['coverage_start']} … {coeffs.meta['coverage_end']}")
    for w in coeffs.meta.get("warnings", []):
        _out(f"提示       : {w}")
    if args.summary and coeffs.times is not None:
        _hr("TimeAxis")
        _out(coeffs.times.summary())

    out = args.out
    layout = "legacy_dat" if str(out).lower().endswith(".dat") else "series_nc"
    shio.write_coeffs(coeffs, out, layout=layout)
    _hr("落盘")
    _out(f"  序列      : {out}  (layout={layout})")
    if coeffs.times is not None and coeffs.times.has_dates and layout == "legacy_dat":
        _out(f"  时间表    : {str(out)[:-4]}_TimeInfo.dat")
    if args.list:
        with open(args.list, "w", encoding="utf-8", newline="") as fh:
            fh.write("epoch,time,source_file\n")
            srcs = coeffs.meta.get("source_files") or [""] * coeffs.ntime
            for i in range(coeffs.ntime):
                t = (str(coeffs.times.values[i]) if coeffs.times is not None else "")
                fh.write(f"{i + 1},{t},{srcs[i] if i < len(srcs) else ''}\n")
        _out(f"  逐历元清单: {args.list}")
    return 0


def cmd_nmax(args) -> int:
    """Print the three independent nmax constraints plus the truncation table."""
    from .diagnostics import describe_nmax, diagnose_nmax

    if bool(args.coeffs) == bool(args.points):
        raise CliError("请给出 --coeffs 或 --points（二选一）")

    n_points = args.npoints
    coverage = args.coverage
    if args.coeffs:
        coeffs = shio.read_coeffs(args.coeffs, nmax=args.nmax,
                                  layout=args.layout)
        _hr("系数文件")
        _out(f"  {coeffs.meta.get('source_file')}  nmax={coeffs.nmax}  "
             f"ntime={coeffs.ntime}")
    else:
        from .analysis import analysis
        lat, lon, values, meta, _grid = _load_field(
            args.points, args.var, args.lon_col, args.lat_col, args.val_col)
        if values.ndim == 2 and values.shape[1] > 1:
            raise CliError(
                f"{args.points}: 读到 {values.shape[1]} 个时次；"
                "请用 --val-col 指定单列，或先导出某一时次。")
        values = np.asarray(values).ravel()
        nmax = args.nmax or 60
        n_points = lat.size
        nlon = (meta.get("nlon")
                if meta.get("input_kind") == "grid" else None)
        rule = args.rule
        if rule == "dh" and nlon is None:
            rule = "auto"
        kwargs = {"method": args.method, "field_unit": args.field_unit}
        if nlon is not None and rule == "dh":
            kwargs["weights_kw"] = {"nlon": int(nlon)}
        try:
            coeffs, rep = analysis(lat, lon, values, nmax, rule=rule, **kwargs)
        except ValueError as exc:
            _out(f"（rule={rule!r} 失败：{str(exc).splitlines()[0][:60]}；"
                 "改用 wlsq）")
            coeffs, rep = analysis(lat, lon, values, nmax, rule=rule,
                                   method="wlsq", field_unit=args.field_unit)
        coverage = rep.coverage
        _hr("先跑一遍分析")
        _out(f"  nmax={nmax}  rule={rep.weight_rule}  method={rep.method}  "
             f"coverage={coverage:.6e}  fit_rmse_rel={rep.fit_rmse_rel:.3e}")

    wanted = None
    if args.list:
        wanted = [int(x) for x in args.list.split(",") if x.strip()]
    diag = diagnose_nmax(coeffs, coverage=coverage, n_points=n_points,
                         tol=args.tol, nmax_list=wanted)
    _out(describe_nmax(diag))
    if diag["nmax_for_tolerance"] is not None:
        _out(f"  ① 误差 ≤ {args.tol:.2%} 所需的最小 nmax = "
             f"{diag['nmax_for_tolerance']}"
             + ("（已触及当前阶数上限，真实需求可能更高）"
                if diag.get("tolerance_hits_ceiling") else ""))
    else:
        _out(f"  ① 在 ≤{coeffs.nmax} 阶范围内，没有任何阶数能达到误差 "
             f"≤ {args.tol:.2%}（谱太宽）")
    return 0


_ARGV_FOR_REPORT: list = []


def _argv_string() -> str:
    return " ".join(_ARGV_FOR_REPORT) if _ARGV_FOR_REPORT else "(unknown)"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI.  Returns the process exit code (0 ok, 1 error, 2 usage)."""
    _force_utf8()
    global _ARGV_FOR_REPORT
    _ARGV_FOR_REPORT = [PROG] + list(sys.argv[1:] if argv is None else argv)

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        return int(args.func(args) or 0)
    except CliError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"错误: 文件不存在 -> {exc}", file=sys.stderr)
        return 2
    except IsADirectoryError as exc:
        print(f"错误: 期望文件但给的是目录 -> {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"错误: 输入无法解析 -> {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"错误: 缺少可选依赖 -> {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断 (Ctrl-C)", file=sys.stderr)
        return 130
    except MemoryError:
        print("错误: 内存不足 —— 请降低 --nmax、减少点数，或对散点使用 "
              "--method cg（矩阵无关）", file=sys.stderr)
        return 1
    except Exception as exc:                                   # noqa: BLE001
        import numpy as _np
        if isinstance(exc, _np.linalg.LinAlgError):
            print("错误: 线性代数求解失败（矩阵奇异/病态）——请降低 --nmax、"
                  "提高过定比、或加正则化 --reg kaula", file=sys.stderr)
            return 1
        print(f"内部错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
