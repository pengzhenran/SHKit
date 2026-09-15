# -*- coding: utf-8 -*-
"""
shkit.io
========

File interchange for :mod:`shkit`: reading and writing the four objects that
make up a spherical-harmonic workflow.

``points``   scattered samples ``(lat, lon, value)``   -> csv / txt / dat / npy / xlsx
``grid``     regular ``(lat_vec, lon_vec, value)``     -> nc / grd / npy / csv
``coeffs``   :class:`shkit.coeffs.SHCoeffs`            -> sh / txt / csv / gfc / npy / npz
``report``   :class:`shkit.diagnostics.AnalysisReport` -> json / md

Units and conventions (identical everywhere in this module)
-----------------------------------------------------------
*input / output latitude*   degrees, geocentric, ``-90 <= lat <= 90``
*input / output longitude*  degrees, ``0 <= lon < 360`` after reading is
                            normalised by ``mod 360`` (a warning is recorded
                            when the file held negatives, so ``-180..180``
                            geographic data becomes ``0..360``)
*values*                    whatever the caller's field is (EWH, geodetic
                            height, gravity anomaly, ...); this module never
                            rescales or converts a value
*weights*                   solid-angle elements in steradian, ``sum(w) == 4*pi``
                            for a global sampling (see :mod:`shkit.weights`)
*coefficients*              :mod:`shkit.coeffs` convention: 4-pi normalised
                            associated Legendre functions, **no
                            Condon-Shortley phase**, ``S[:, 0] == 0``
*grid shape*                ``(nlat, nlon)`` or ``(nlat, nlon, ntime)``, with
                            ``lat_vec`` strictly **increasing** and
                            ``lon_vec`` increasing; a descending latitude axis
                            found on disk is flipped (data included) and
                            ``meta['lat_order_flipped'] = True`` is recorded
*nmax*                      maximum spherical harmonic degree; a matrix
                            layout file of ``nmax = L`` occupies an
                            ``(L+1, L+1)`` array and carries
                            ``(L+1)**2`` real coefficients
*time axis*                 an SHCoeffs with ``ntime = 1`` is written and read
                            back as a 2-D (single-slice) object, so a
                            1-slice round trip returns a 2-D ``SHCoeffs``

Layouts for spherical harmonic coefficients
-------------------------------------------
``triangle``    the ``to_triangle()`` layout of the reference ``m2py`` /
                ``gridSHconvert`` software: ``[C; S]`` stacked, ``2*NC`` rows
                with ``NC = (nmax+1)(nmax+2)/2`` and one column per time step,
                ordered ``m`` outer / ``n`` inner:
                ``(0,0), (0,1), ..., (0,N), (1,1), (1,2), ...``.
``gmfcsv``      one row per coefficient, ``n,m,C,S`` (an optional 5th/6th
                column holding ``sigmaC,sigmaS`` is ignored on read).
``gfc``         ICGEM / GFZ text format, ``gfc n m C S sigmaC sigmaS`` lines
                with ``#`` comments and ``begin/end`` blocks; the field name
                and constants of the header are kept in ``meta``.
``matrix``      the native dense layout as an ``.npy`` or ``.npz`` array
                (``C``/``S`` of shape ``(nmax+1, nmax+1[, ntime])``).

Windows / Chinese paths
-----------------------
Every path is passed through :func:`os.fspath` and handled as ``str``; all text
I/O uses explicit UTF-8 (with a BOM on written csv so that Excel opens Chinese
column names correctly).  **Helper files are normally created next to their
target**, but ``.nc`` is an exception: netCDF4's C library (HDF5) cannot touch a
path containing non-ASCII characters on Windows, which breaks *both* reading
(``FileNotFoundError`` even though the file exists) and writing
(``PermissionError``).  :func:`open_nc_dataset` / :func:`to_netcdf_path` route
such paths through an ASCII temp directory; see their docstrings.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import glob
import gzip
import io as _io
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .coeffs import SHCoeffs, triangle_order
from .timeaxis import (TimeAxis, attr_ci, parse_grace_filename,
                       time_axis_from_meta)

__all__ = [
    "SHKIT_VERSION",
    "POINT_EXTENSIONS",
    "GRID_EXTENSIONS",
    "COEFF_EXTENSIONS",
    "REPORT_EXTENSIONS",
    "read_points",
    "write_points",
    "read_grid",
    "write_grid",
    "open_nc_dataset",
    "to_netcdf_path",
    "read_coeffs",
    "write_coeffs",
    "write_report",
    "save_result",
    "gaussian_degree_weights",
    "detect_coeff_layout",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
#: Version string written into every file header we produce.  v2.0 adds the
#: multi-time axis (A*/C*), horizontal deformation (B*) and the display upgrade
#: (D*); the on-disk *formats* are unchanged from 1.0 except that ``series_nc``
#: and the field-series ``(time, lat, lon)`` layout now exist.
SHKIT_VERSION = "2.0.1"

POINT_EXTENSIONS = (".csv", ".txt", ".dat", ".tsv", ".npy", ".xlsx")
GRID_EXTENSIONS = (".nc", ".nc4", ".cdf", ".grd", ".npy", ".csv", ".txt", ".dat")
COEFF_EXTENSIONS = (".sh", ".txt", ".csv", ".dat", ".tsv", ".gfc", ".npy", ".npz",
                    ".nc")
REPORT_EXTENSIONS = (".json", ".md")

#: ASCII column names used on write (portable across Excel / MATLAB / R).
COL_LON, COL_LAT, COL_VAL = "lon", "lat", "value"

#: Recognised header aliases.  Longitude is resolved before latitude, and the
#: matcher is anchored on ``^``/``$`` so that ``lon_sigma`` is not mistaken for
#: ``lon``.  Chinese aliases are accepted even though the writer only emits the
#: ASCII ones (the BOM written by :func:`_text_write` makes them safe to edit in
#: Excel).
_LON_NAMES = ("lon", "long", "longitude", "lon_deg", "longitude_deg",
              "x", "xlon", "glon", "经度", "东经")
_LAT_NAMES = ("lat", "latitude", "lat_deg", "latitude_deg",
              "y", "ylat", "glat", "纬度", "北纬")
_VAL_NAMES = ("value", "val", "z", "data", "field", "ewh", "height", "h",
              "sigma", "obs", "slm", "trend", "anomaly", "值", "数值")

_DELIMS = [(",", "逗号"), (";", "分号"), ("\t", "制表符"), ("|", "竖线"),
           (None, "空白")]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _p(path) -> Path:
    """Path -> :class:`pathlib.Path` (accepts ``str``, ``Path``, os.PathLike)."""
    return Path(os.fspath(path))


def _suffix(path) -> str:
    """Lower-case extension including the dot."""
    return _p(path).suffix.lower()


def _text_write(path, text: str, gzip_ok: bool = True, bom: bool = True) -> Path:
    """Write UTF-8 text (gzip when the name ends in ``.gz``), creating parents.

    ``bom=True`` prepends a UTF-8 BOM.  That is deliberate for *tabular* output:
    it is what makes a Chinese ``lon,lat`` header open correctly in Excel on a
    zh-CN Windows install, and every reader here strips it.  Structured formats
    (JSON, Markdown) are written without a BOM instead, because ``json.load``
    and most Markdown tooling reject a leading BOM.
    """
    p = _p(path)
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8-sig" if bom else "utf-8")
    if gzip_ok and str(p).lower().endswith(".gz"):
        with gzip.open(p, "wb") as fh:
            fh.write(data)
    else:
        with open(p, "wb") as fh:
            fh.write(data)
    return p


def _text_read(path) -> str:
    """Read UTF-8 text, transparently decompressing ``.gz``."""
    p = _p(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if str(p).lower().endswith(".gz"):
        with gzip.open(p, "rb") as fh:
            raw = fh.read()
    else:
        raw = p.read_bytes()
    return raw.decode("utf-8-sig", errors="replace")


def _require_file(path) -> Path:
    p = _p(path)
    if not p.exists():
        raise FileNotFoundError(
            f"文件不存在: {p}  (文件不存在 / not found; "
            f"当前工作目录 = {os.getcwd()})")
    if p.is_dir():
        raise IsADirectoryError(f"这是一个目录，不是文件: {p}")
    return p


#: Data line of an ICGEM ``gfc`` **or** a PO.DAAC ``SHM`` (``GRCOF2``) file.
#: Both put ``n m C S [sigmaC sigmaS ...]`` in the same first five columns, so a
#: single reader covers them -- SHM just uses a YAML header instead of ICGEM keys.
_COEFF_LINE_RE = re.compile(r"(?m)^\s*(?:gfc[t]?|GRCOF2)\s+\d+\s+\d+\s+[-+.\d]", re.I)


def _head_text(p, n: int = 8192) -> str:
    """First ``n`` characters of a (possibly gzipped) text file."""
    try:
        if str(p).lower().endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
                return fh.read(n)
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(n)
    except OSError:
        return ""


def _sniff_coeff_ext(p) -> str:
    """``'.gfc'`` for ICGEM/SHM content, else ``'.sh'`` (triangle text).

    GRACE Level-2 files frequently ship with **no extension at all**
    (``GSM-2_2021335-2021365_GRFO_UTCSR_BA01_0600``), and the CSR/GFZ ones are in
    PO.DAAC ``SHM`` form (``GRCOF2`` lines) rather than ICGEM ``gfc``.
    """
    return ".gfc" if _COEFF_LINE_RE.search(_head_text(p)) else ".sh"


def _ensure_ext(path, allowed: Sequence[str], what: str) -> str:
    """Return the lower-case extension, or raise listing the supported ones."""
    ext = _suffix(path)
    if ext not in allowed:
        raise ValueError(
            f"{what}: 不支持的扩展名 {ext!r}（文件 {path}）。"
            f"支持: {', '.join(sorted(allowed))}")
    return ext


# ---------------------------------------------------------------------------
# netCDF4 / HDF5 non-ASCII path workaround (Windows)
# ---------------------------------------------------------------------------
# netCDF4's C library (HDF5) cannot touch a path containing non-ASCII characters
# on Windows.  Reading fails with FileNotFoundError even though the file exists;
# **writing fails with PermissionError**, which is actively misleading.  Measured
# on this machine with `D:\华为家庭存储\...\x.nc` (both xarray and netCDF4 direct).
# The remedy (same idea as the reference `m2py` viewer) is an ASCII temp copy.
_NC_TEMP_DIRS: list = []

#: netCDF4/HDF5 的 C 库是**进程级全局状态**，多线程同时 open/read 会互相踩
#: （实测症状：`RuntimeError: NetCDF: Not a valid ID`）。所有 nc 打开/读取都过这把锁。
_NC_OPEN_LOCK = threading.RLock()


def _is_ascii_path(path) -> bool:
    """True when the *absolute* path is pure ASCII (what the C library needs)."""
    try:
        os.path.abspath(os.fspath(path)).encode("ascii")
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


@atexit.register
def _cleanup_nc_temp_dirs() -> None:
    for d in list(_NC_TEMP_DIRS):
        shutil.rmtree(d, ignore_errors=True)
        if not os.path.exists(d):
            _NC_TEMP_DIRS.remove(d)


def _new_nc_temp_dir() -> str:
    d = tempfile.mkdtemp(prefix="shkit_nc_")
    _NC_TEMP_DIRS.append(d)
    return d


def _drop_nc_temp_dir(d: str) -> None:
    """Delete a temp dir, retrying briefly: on Windows a just-closed netCDF file
    can stay locked for a moment, and a silent failure here leaks the copy.

    Only deregisters when the directory is really gone, so the ``atexit`` sweep
    gets another chance instead of forgetting about it.
    """
    for _ in range(5):
        shutil.rmtree(d, ignore_errors=True)
        if not os.path.exists(d):
            break
        time.sleep(0.05)
    if not os.path.exists(d) and d in _NC_TEMP_DIRS:
        _NC_TEMP_DIRS.remove(d)


def open_nc_dataset(path, **kwargs):
    """``xarray.open_dataset`` that survives a non-ASCII path on Windows.

    Tries the path directly first (so nothing changes on the many machines and
    builds where it works).  If the C library refuses **and** the file really
    exists **and** the path is non-ASCII, the file is copied into an ASCII temp
    directory and read **eagerly** from there -- eager because the temp copy has
    to be deleted before we return, and a lazy handle would then be dangling.
    ``attrs['shkit_path_workaround']`` records that this happened, so it is
    never silent.

    Callers should use this instead of ``xr.open_dataset`` for every ``.nc``.

    ⚠️ **为什么要串行化（`_NC_OPEN_LOCK`）**：netCDF4/HDF5 的 C 库带**进程级全局
    状态**，两个线程同时 open/read 同一份文件时，其中一个会抛
    ``RuntimeError: NetCDF: Not a valid ID``（这是实测到的真实故障：GUI 一边在
    worker 线程里读数据、一边在主线程里读变量名，第一次打开 mascon 必然报错）。
    这个异常**不是** ``OSError``，所以以前那条"非 ASCII 路径就退到临时副本"的兜底
    根本兜不住它。现在：① 所有 nc 打开/读取都过同一把锁；② 非 ASCII 路径的兜底
    也认 ``RuntimeError``。
    """
    import xarray as xr

    p = _require_file(path)
    with _NC_OPEN_LOCK:
        try:
            return xr.open_dataset(p, **kwargs)
        except (OSError, RuntimeError) as exc:
            # FileNotFoundError / PermissionError（非 ASCII 路径的经典症状），
            # 以及并发下的 "NetCDF: Not a valid ID"
            if not os.path.exists(p) or _is_ascii_path(p):
                raise
            tmp_dir = _new_nc_temp_dir()
            try:
                tmp = os.path.join(tmp_dir, "data.nc")
                shutil.copy2(p, tmp)
                ds = None
                last = None
                for attempt in range(3):        # 并发下的 "not a valid ID" 是可重试的
                    try:
                        ds = xr.open_dataset(tmp, **kwargs)
                        ds.load()   # eager: the temp copy is about to be deleted
                        break
                    except (OSError, RuntimeError) as e2:     # noqa: PERF203
                        last = e2
                        time.sleep(0.05 * (attempt + 1))
                if ds is None:
                    raise last if last is not None else exc
                ds.close()          # ...and the handle has to go first, or Windows locks it
            finally:
                _drop_nc_temp_dir(tmp_dir)
            ds.attrs["shkit_path_workaround"] = (
                f"non-ASCII path -> ASCII temp copy ({type(exc).__name__} from netCDF4)")
            return ds


def open_nc_dataset_lazy(path, **kwargs):
    """打开 nc 并**保持句柄**（懒加载用），返回 ``(ds, closer)``。

    与 :func:`open_nc_dataset` 的唯一区别：**不** eager load、**不**立刻关句柄 ——
    懒加载稍后还要按时次切片读。非 ASCII 路径仍走"复制到 ASCII 临时目录"，
    但那份副本要留到 ``closer()`` 才删（否则句柄悬空）。

    实测（CSR mascon 112 MB / 256 时次）：只读结构 64 ms、读**一个**时次 253 ms、
    读**全部** 2.4 s —— 所以"先给当前时次、其余后台补"能把首屏从 3.5 s 降到 0.3 s。
    """
    import xarray as xr

    p = _require_file(path)
    try:
        return xr.open_dataset(p, **kwargs), (lambda: None)
    except (OSError, RuntimeError):
        if not os.path.exists(p) or _is_ascii_path(p):
            raise
    tmp_dir = _new_nc_temp_dir()
    tmp = os.path.join(tmp_dir, "data.nc")
    shutil.copy2(p, tmp)
    try:
        ds = xr.open_dataset(tmp, **kwargs)
    except Exception:                                            # noqa: BLE001
        _drop_nc_temp_dir(tmp_dir)
        raise

    def _closer():
        try:
            ds.close()
        except Exception:                                        # noqa: BLE001
            pass
        _drop_nc_temp_dir(tmp_dir)

    return ds, _closer


class LazyTimeCube:
    """多时次网格的**懒加载**容器：先给第一个时次，其余按需/后台补。

    为什么要有它：Panoply 打开文件只读"目录"（64 ms），再读**一个切片**（4 MB）
    就出图；而 SHKit 原来在打开时把 1.06 GB 全读进来（2.4 s）。实测一个时次只要
    253 ms，所以"先给当前时次、其余后台按块补"能让首屏从 3.5 s 降到 ~0.3 s，
    同时**保留"全历元都在内存里"**（补齐后与一次性读入逐值一致）。

    它像数组一样用（所以调用方不必改）：

    * ``cube[:, :, t]`` / ``cube[:, :, a:b]`` —— 按需读那几个时次后返回视图；
    * ``np.asarray(cube)`` —— 补齐全部后返回底层数组；
    * ``shape`` / ``ndim`` / ``dtype`` / ``ntime`` / ``n_filled``；
    * ``ensure(t)`` / ``ensure_all()`` / ``start_prefetch()`` / ``wait()`` / ``close()``。

    ⚠️ 因此**不会**出现"读到一半的零值被当成数据"：任何按索引取用都会先把那一块读进来。
    """

    #: 后台补数时一次读几个时次。一次读大的块远比"每个时次单独读"便宜
    #: （实测：单独读 253 ms/时次；连续读 256 个时次共 2.4 s ≈ 9.5 ms/时次）。
    #: 取 64：既摊薄了每次 HDF5 调用的固定开销，又保证"按需插队"最多等一块
    #: （64 个时次 ≈ 0.6 s），拖滑块不会明显顿。
    PREFETCH_BLOCK = 64

    def __init__(self, ds, closer, var: str, shape, *, dtype=np.float32,
                 t0: int = 0, row_flip: bool = False, perm=(1, 2, 0)):
        self._ds, self._closer, self._var = ds, closer, var
        self._arr = np.zeros(shape, dtype=dtype)
        self._done = np.zeros(shape[2], dtype=bool)
        self._row_flip = bool(row_flip)
        #: 文件里的维度顺序 → (lat, lon, time)。真实文件常见 (time, lat, lon)，
        #: 不转置就直接赋值会得到 "could not broadcast (1,720,1440) into
        #: (720,1440,1)"（实测踩到）。
        self._perm = tuple(int(x) for x in perm)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._wanted: list = []          # 需要**优先**读的时次（LIFO）
        self.ensure(int(t0))

    # ------------------------------------------------------------------ 形状
    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return 3

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ntime(self) -> int:
        return int(self._arr.shape[2])

    @property
    def n_filled(self) -> int:
        return int(self._done.sum())

    def is_filled(self, t: int) -> bool:
        return bool(self._done[int(t)])

    def __len__(self):
        return int(self._arr.shape[0])

    def __repr__(self):                                          # pragma: no cover
        return (f"<LazyTimeCube {self.shape} {self.dtype.__name__} "
                f"已填 {self.n_filled}/{self.ntime}>")

    # -------------------------------------------------------------------- 读
    def _read_block(self, i0: int, i1: int):
        """读 ``[i0, i1)`` 这几个时次并写进底层数组（一次 HDF5 读一块）。"""
        with _NC_OPEN_LOCK:
            blk = np.asarray(self._ds[self._var].isel(time=slice(i0, i1)).values)
        blk = np.transpose(blk, self._perm)      # → (lat, lon, time)
        if self._row_flip:
            blk = blk[:, ::-1, :]
        self._arr[:, :, i0:i1] = blk.astype(self._arr.dtype, copy=False)
        self._done[i0:i1] = True

    def ensure(self, t: int) -> None:
        """保证第 ``t`` 个时次已读（没读就同步读一次，约 250 ms）。"""
        t = int(t)
        if not (0 <= t < self.ntime) or self._done[t]:
            return
        with self._lock:
            if self._done[t]:
                return
            self._read_block(t, t + 1)

    def ensure_all(self, progress=None) -> None:
        """补齐全部时次（按块读）。``progress(done, total)`` 可选。"""
        total = self.ntime
        while True:
            with self._lock:
                left = np.flatnonzero(~self._done)
                if left.size == 0:
                    return
                i0 = int(left[0])
                i1 = min(i0 + self.PREFETCH_BLOCK, total)
                self._read_block(i0, i1)
                done = int(self._done.sum())
            if progress is not None:
                try:
                    progress(done, total)
                except Exception:                                # noqa: BLE001
                    pass

    def start_prefetch(self, from_t: Optional[int] = None) -> None:
        """后台把剩下的时次按块补进来（不阻塞界面）。

        顺序：先从**当前时次**往后补（用户最可能接着看的），再从 0 补起；
        ``ensure()`` 请求过而还没读的时次插队。
        """
        if self._thread is not None and self._thread.is_alive():
            return
        t0 = 0 if from_t is None else int(from_t)

        def _work():
            order = list(range(t0, self.ntime)) + list(range(0, t0))
            i = 0
            while not self._stop and i < len(order):
                with self._lock:
                    if self._wanted:
                        t = int(self._wanted.pop())
                        if not self._done[t]:
                            self._read_block(t, t + 1)
                            continue
                    t = order[i]
                    i += 1
                    if self._done[t]:
                        continue
                    end = min(t + self.PREFETCH_BLOCK, self.ntime)
                    self._read_block(t, end)

        self._thread = threading.Thread(target=_work, name="shkit-nc-prefetch",
                                        daemon=True)
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等后台补数结束（测试与"要全部历元"的操作会用到）。"""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self._closer()
        except Exception:                                        # noqa: BLE001
            pass

    # ------------------------------------------------------------ 数组式访问
    def _key_times(self, key):
        """从 ``key`` 里取出时次索引（``None`` 表示"整块"）。"""
        if not isinstance(key, tuple) or len(key) < 3:
            return None
        k = key[2]
        if isinstance(k, slice):
            return list(range(*k.indices(self.ntime)))
        arr = np.atleast_1d(np.asarray(k))
        return [int(x) % self.ntime for x in arr]

    def __getitem__(self, key):
        ts = self._key_times(key)
        if ts is None:
            self.ensure_all()
            return self._arr[key]
        for t in sorted({t for t in ts if not self._done[t]}):
            self._wanted.append(t)
            self.ensure(t)
        return self._arr[key]

    def __array__(self, dtype=None, copy=None):                  # noqa: A003
        self.ensure_all()
        return self._arr if dtype is None else self._arr.astype(dtype, copy=False)


def to_netcdf_path(ds, path, **kwargs):
    """``Dataset.to_netcdf`` that survives a non-ASCII path on Windows.

    netCDF4 cannot *create* a file under a non-ASCII path either; here the write
    goes to an ASCII temp file which is then moved into place.  Falls back only
    when the path is non-ASCII -- an ASCII write that fails is a real error and
    is re-raised rather than retried.
    """
    p = os.fspath(path)
    with _NC_OPEN_LOCK:            # 与读同一把锁：netCDF4 的 C 库状态是进程级的
        try:
            ds.to_netcdf(p, **kwargs)
            return p
        except OSError:
            if _is_ascii_path(p):
                raise
    parent = os.path.dirname(os.path.abspath(p))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_dir = _new_nc_temp_dir()
    try:
        tmp = os.path.join(tmp_dir, "out.nc")
        with _NC_OPEN_LOCK:
            ds.to_netcdf(tmp, **kwargs)
        shutil.move(tmp, p)
    finally:
        _drop_nc_temp_dir(tmp_dir)
    return p


def _write_nc(ds, p) -> str:
    """Write a Dataset to ``.nc``, preferring the netCDF4 engine.

    The ``ImportError``/``ValueError`` fallback to the default (scipy) engine is
    the pre-existing behaviour and is kept.
    """
    try:
        return to_netcdf_path(ds, p, engine="netcdf4")
    except (ImportError, ValueError):
        return to_netcdf_path(ds, p)


def _to_float_array(a, name: str, ndim: Optional[int] = None) -> np.ndarray:
    """Coerce to a float ndarray, raising a Korean-free, explicit error."""
    try:
        out = np.asarray(a, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"列 {name!r} 不是数值列: {exc}") from exc
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"列 {name!r} 期望 {ndim} 维，实际 {out.ndim} 维")
    return out


def _resolve_col(col, columns, ncols: int, semantics: str,
                 default: int) -> tuple:
    """Resolve a user column specifier to an index.

    ``col`` may be ``None`` (use ``default``), an ``int`` (positional), the name
    of a column, or a 1-based index given as a string (``"1"``).  Returns
    ``(index, note_or_None)`` where ``note`` records how the column was found.
    """
    if col is None:
        if default >= ncols:
            raise ValueError(
                f"按位置取{default + 1}列作为{semantics}，但文件只有 {ncols} 列。"
                "请显式指定列名或列序号。")
        return default, f"{semantics} = 第 {default + 1} 列（位置默认）"

    if isinstance(col, (int, np.integer)):
        idx = int(col)
        if idx < 0:
            idx += ncols
        if not 0 <= idx < ncols:
            raise ValueError(f"{semantics} 列序号 {col} 超出范围（共 {ncols} 列）")
        return idx, f"{semantics} = 第 {idx + 1} 列（按序号给出）"

    if isinstance(col, str):
        key = col.strip()
        if key in columns:
            return columns.index(key), f"{semantics} = 列名 {key!r}"
        if re.fullmatch(r"-?\d+", key):
            idx = int(key)
            if idx < 0:
                idx += ncols
            if not 0 <= idx < ncols:
                raise ValueError(f"{semantics} 列序号 {key} 超出范围（共 {ncols} 列）")
            return idx, f"{semantics} = 第 {idx + 1} 列（按序号给出）"
        low = key.lower()
        for i, c in enumerate(columns):
            if str(c).strip().lower() == low:
                return i, f"{semantics} = 列名 {c!r}（忽略大小写）"
        raise ValueError(
            f"找不到{semantics}列 {col!r}。文件中的列为: {list(columns)}")

    raise TypeError(f"{semantics} 列说明符类型不支持: {type(col).__name__}")


def _guess_named_column(columns, candidates: Sequence[str]) -> Optional[int]:
    """Return the index of the first header entry matching ``candidates``."""
    norm = [str(c).strip().lower().replace(" ", "_") for c in columns]
    for cand in candidates:
        for i, c in enumerate(norm):
            if c == cand:
                return i
    return None


def _looks_like_dates(labels: Sequence[str]) -> bool:
    """True when a list of column names can serve as an epoch axis.

    Accepts full ISO dates (``2002-01-18``), month stamps (``2002-01``),
    year stamps (``2002``) and the ``2002_01`` / ``2002/01`` spellings people
    actually export.  Requires **all** entries to parse and to be strictly
    increasing — otherwise a column called ``value1``…``value3`` or an
    unrelated numeric header would be mistaken for a calendar.
    """
    txt = [str(s).strip().replace("_", "-").replace("/", "-") for s in labels]
    if not txt or any(not re.fullmatch(r"\d{4}(-\d{1,2}(-\d{1,2})?)?", t)
                      for t in txt):
        return False
    try:
        import datetime as _dt
        parsed = []
        for t in txt:
            parts = [int(p) for p in t.split("-")]
            parsed.append(_dt.date(parts[0],
                                   parts[1] if len(parts) > 1 else 1,
                                   parts[2] if len(parts) > 2 else 1))
    except Exception:                                            # noqa: BLE001
        return False
    return all(b > a for a, b in zip(parsed, parsed[1:]))


def _flip_grid_if_needed(lat_vec: np.ndarray, grid: np.ndarray, meta: dict,
                         warnings: list):
    """Force an increasing latitude axis and flip the data to match."""
    if lat_vec.size > 1 and lat_vec[0] > lat_vec[-1]:
        meta["lat_order_flipped"] = True
        warnings.append(
            "输入网格的纬度轴是降序（北->南），已翻转为升序；数据同步翻转。"
            "meta['lat_order_flipped'] = True")
        lat_vec = lat_vec[::-1].copy()
        grid = grid[::-1, ...].copy()
    else:
        meta.setdefault("lat_order_flipped", False)
    return lat_vec, grid


def _looks_like_grid(lat: np.ndarray, lon: np.ndarray) -> tuple:
    """True when the (lat, lon) sample set is a complete rectangular grid.

    Returns ``(ok, lat_vec, lon_vec)`` with the sorted unique axes when ok.
    """
    lat_u = np.unique(lat)
    lon_u = np.unique(lon)
    if lat_u.size < 2 or lon_u.size < 2:
        return False, lat_u, lon_u
    if lat_u.size * lon_u.size != lat.size:
        return False, lat_u, lon_u
    # every (lat, lon) pair must occur exactly once
    keys = {(round(float(a), 9), round(float(b), 9)) for a, b in zip(lat, lon)}
    if len(keys) != lat.size:
        return False, lat_u, lon_u
    return True, lat_u, lon_u


def _regular_spacing(v: np.ndarray, tol: float = 1e-6) -> bool:
    """True when the axis is (near) uniformly spaced."""
    if v.size < 3:
        return True
    d = np.diff(v)
    return bool(np.allclose(d, d[0], rtol=1e-6, atol=tol))


def _json_meta(meta: Mapping[str, Any]) -> dict:
    """JSON-safe copy of a metadata dict (paths -> str, arrays -> list)."""
    out = {}
    for k, v in dict(meta).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            out[k] = v.item()
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x.item() if isinstance(x, (np.integer, np.floating,
                                                 np.bool_)) else x for x in v]
        elif isinstance(v, dict):
            out[k] = _json_meta(v)
        else:
            out[k] = str(v)
    return out


def _promote_target(H: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    nlat: int, nlon: int, ntime: int, what: str):
    """(N,) or (N, ntime) values -> (nlat, nlon[, ntime]) on the grid axes."""
    H = np.asarray(H, dtype=float)
    if H.ndim == 1:
        H = H[:, None]
    if H.shape[0] != lat.size:
        raise ValueError(f"{what}: 数值个数 {H.shape[0]} 与坐标个数 {lat.size} 不一致")
    if H.shape[1] != 1 and H.shape[1] != ntime:
        raise ValueError(
            f"{what}: 数值有 {H.shape[1]} 个时次，与期望的 ntime={ntime} 不符")
    li = np.unique(lat, return_inverse=True)[1]
    loi = np.unique(lon, return_inverse=True)[1]
    out = np.full((nlat, nlon, H.shape[1]), np.nan)
    out[li, loi, :] = H
    if np.isnan(out).any():
        raise ValueError(
            f"{what}: 存在没有数据的 (lat, lon) 网格单元，无法整理成规则网格")
    return out[:, :, 0] if H.shape[1] == 1 else out


# ---------------------------------------------------------------------------
# tabular input (csv / txt / dat / tsv) and xlsx
# ---------------------------------------------------------------------------
def _sniff_delimiter(path) -> tuple:
    """Guess the field separator from the first data-bearing line.

    Returns ``(delim_or_None, human_name)``; ``None`` means "any whitespace".
    """
    text = _text_read(path)
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.startswith("%"):
            continue
        counts = [(s.count(d), d, nm) for d, nm in _DELIMS if d is not None]
        counts = [c for c in counts if c[0] > 0]
        if not counts:
            return None, "空白"
        # tabs are the strongest hint; otherwise the most frequent separator
        tabs = [c for c in counts if c[1] == "\t"]
        if tabs:
            return "\t", "制表符"
        best = max(counts, key=lambda c: c[0])
        return best[1], best[2]
    return None, "空白"


def _read_table(path, delim=None) -> tuple:
    """Read a text table with an optional header; returns ``(df, meta, warn)``."""
    import pandas as pd

    guessed, dname = _sniff_delimiter(path)
    if delim is None:
        delim = guessed
        how = f"自动嗅探 = {dname}"
    else:
        how = f"用户指定 = {delim!r}"

    text = _text_read(path)
    lines = text.splitlines()
    n_comment = sum(1 for ln in lines
                    if ln.strip().startswith(("#", "//", "%")))
    body = [ln for ln in lines
            if ln.strip() and not ln.lstrip().startswith(("#", "//", "%"))]
    if not body:
        raise ValueError(f"{path}: 文件里没有有效数据行（只有注释/空行）")

    # header detection: the first data line must not be fully numeric
    def _row_is_numeric(line):
        fields = line.split(delim) if delim else line.split()
        if not fields:
            return False
        for f in fields:
            tok = f.strip().strip('"').strip("'")
            if tok == "":
                continue
            try:
                float(tok)
            except ValueError:
                return False
        return True

    has_header = not _row_is_numeric(body[0])

    sep = rf"\s+" if delim is None else delim
    df = pd.read_csv(_io.StringIO(text), sep=sep, engine="python",
                     comment="#", skip_blank_lines=True,
                     header=0 if has_header else None)
    if not has_header:
        # drop comment lines that are not '#'-prefixed
        df = df[~df.apply(
            lambda r: str(r.iloc[0]).strip().startswith(("//", "%")), axis=1)]
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    meta = {
        "delimiter": sep if delim is not None else "whitespace",
        "delimiter_label": dname if delim is None else delim,
        "delimiter_source": how,
        "has_header": bool(has_header),
        "columns": list(df.columns),
        "n_comment_lines": int(n_comment),
    }
    warn = []
    if not has_header:
        warn.append("未检测到表头，列按位置 col1, col2, ... 命名")
    return df, meta, warn


def _read_table_xlsx(path) -> tuple:
    """Read the first sheet of an Excel workbook; needs ``openpyxl``."""
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "读取 .xlsx 需要 openpyxl，但当前解释器没有安装。请执行:\n"
            "    pip install openpyxl\n"
            f"（原始错误: {exc}）") from exc
    import pandas as pd

    p = _require_file(path)
    try:
        df = pd.read_excel(p, sheet_name=0, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"读取 Excel 失败: {p}: {exc}") from exc

    df = df.dropna(how="all")
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise ValueError(f"{p}: 工作表为空")
    if all(re.fullmatch(r"\d+", str(c)) for c in df.columns):
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
        warn = ["Excel 首行是数值，未检测到表头，列按位置命名"]
    else:
        warn = []
    df.columns = [str(c).strip() for c in df.columns]
    meta = {"columns": list(df.columns), "sheet": 0, "format": "xlsx"}
    return df, meta, warn


# ---------------------------------------------------------------------------
# READ / WRITE  points
# ---------------------------------------------------------------------------
def read_points(path, lat_col=None, lon_col=None, val_col=None,
                delim=None) -> tuple:
    """Read scattered samples from csv / txt / dat / npy / xlsx.

    Parameters
    ----------
    path : str or Path
        Input file.  Recognised extensions: ``.csv .txt .dat .tsv .npy .xlsx``
        (``.gz`` is accepted on top of the text ones).  Chinese paths are fine.
    lat_col, lon_col : int or str, optional
        Column index (0-based; negative counts from the end) or column name.
        A numeric string is interpreted as an *index* (``"1"`` = second
        column), not as a name.  When omitted the resolver uses the header if
        there is one (``lat``/``latitude``/``纬度``, ``lon``/``longitude``/
        ``经度``) and otherwise falls back to the common geographic convention:
        column 1 = **longitude**, column 2 = **latitude**, column 3 = value.
        This is recorded in ``meta['warnings']``.
    val_col : int, str, or sequence of them, optional
        The value column(s).  **Passing a sequence makes every entry an epoch**,
        which is how a scatter time series with arbitrary column names is read
        (``val_col=["e2002", "e2003"]`` or ``val_col=[2, 3, 4]``).  A single
        spec keeps the old behaviour.
    delim : str, optional
        Force a field separator (``','``, ``';'``, ``'\\t'``, ``'|'``).  By
        default it is sniffed from the first non-comment line.

    Returns
    -------
    lat, lon, values, meta : ndarray, ndarray, ndarray, dict
        ``lat``/``lon`` in **degrees** (longitude is normalised to ``[0, 360)``);
        ``values`` has shape ``(N,)`` for one time step or ``(N, ntime)``.
        ``meta`` holds the resolved columns, the delimiter, the inferred layout
        and a ``warnings`` list - read it before trusting the numbers.

    Notes
    -----
    **多时次散点的两种通行写法**（都会读成 ``(N, ntime)``）::

        # 1) 矩阵式 .npy：(N, 2 + ntime)，第 1 列经度、第 2 列纬度
        # 2) 宽表 csv/txt：lon,lat,value,value1,value2   ← write_points 写的就是这个
        #    列名任意时用 val_col=["e2002","e2003",…] 显式指定

    若选中的**数值列名本身能当日期用**（``2002-01-18``、``2002-01``、``2002``），
    就直接把它们当作历元日期写进 ``meta['time']``（时间序列 / 趋势页因此能按真实
    日期拟合）；解析不出日期就**只给历元序号**，不编造。
    ``.npy`` 一律只有序号（数组里没有列名）。

    ⚠️ **长表（tidy）格式不支持**：一行一个"点×历元"、带一列日期的表，会被当成
    宽表解释（只取第一列数值）。这种情况会给出明确的警告，提示先透视成宽表。

    Raises
    ------
    FileNotFoundError
        The path does not exist (the message repeats the full path).
    ValueError
        Unknown extension, or the requested columns are absent.
    ImportError
        ``.xlsx`` without ``openpyxl``.
    """
    p = _require_file(path)
    ext = _suffix(p)
    warnings: list = []

    if ext not in POINT_EXTENSIONS:
        raise ValueError(
            f"read_points: 不支持的扩展名 {ext!r}（文件 {p}）。"
            f"支持: {', '.join(POINT_EXTENSIONS)}")

    meta: dict = {"source_file": str(p), "format": ext.lstrip(".")}

    if ext == ".npy":
        arr = np.load(p, allow_pickle=False)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(
                f"{p}: .npy 必须是 (N,3) 或 (N,2+Nt) 的二维数组，"
                f"实际 shape = {arr.shape}")
        meta["array_shape"] = list(arr.shape)
        meta["layout"] = "lon,lat,value[,...]（第1列经度、第2列纬度）"
        warnings.append(
            ".npy 按第1列=经度、第2列=纬度、其余列=数值解释；"
            "可用 lon_col/lat_col 覆盖")
        lon = arr[:, 0].copy()
        lat = arr[:, 1].copy()
        vals = arr[:, 2:].copy()
        meta["has_header"] = False
        meta["columns"] = [f"col{i + 1}" for i in range(arr.shape[1])]
    else:
        if ext == ".xlsx":
            df, tmeta, warn = _read_table_xlsx(p)
        else:
            df, tmeta, warn = _read_table(p, delim=delim)
        meta.update(tmeta)
        warnings.extend(warn)
        cols = list(df.columns)
        c_lon = c_lat = c_val = None
        if lon_col is None:
            c_lon = _guess_named_column(cols, _LON_NAMES)
        if lat_col is None:
            c_lat = _guess_named_column(cols, _LAT_NAMES)

        if lon_col is None and c_lon is not None:
            i_lon, note_lon = c_lon, f"经度 = 列名 {cols[c_lon]!r}"
        else:
            i_lon, note_lon = _resolve_col(lon_col, cols, len(cols), "经度", 0)
        if lat_col is None and c_lat is not None:
            i_lat, note_lat = c_lat, f"纬度 = 列名 {cols[c_lat]!r}"
        else:
            i_lat, note_lat = _resolve_col(lat_col, cols, len(cols), "纬度", 1)
        if i_lat == i_lon:
            raise ValueError(
                f"经度列和纬度列指向同一列（{cols[i_lon]!r}）；请显式指定 lat_col/lon_col")

        if val_col is None:
            # 1) an explicit value header wins, and then only that one column
            i_val = _guess_named_column(cols, _VAL_NAMES)
            if i_val in (i_lon, i_lat):
                i_val = None
            if i_val is not None:
                note_val = f"数值 = 列名 {cols[i_val]!r}"
                # 2) our writer appends value1, value2, ... for extra time steps
                chain = []
                k = 1
                while True:
                    nxt = f"{cols[i_val]}{k}"
                    j = next((t for t, c in enumerate(cols)
                              if str(c).strip().lower() == nxt.lower()), None)
                    if j is None:
                        break
                    chain.append(j)
                    k += 1
                if chain:
                    note_val += ("（并含 "
                                 + ",".join(repr(str(cols[j])) for j in chain)
                                 + " 作为后续时次）")
                    i_val = [i_val] + chain
            else:
                rest = [i for i in range(len(cols)) if i not in (i_lon, i_lat)]
                if not rest:
                    raise ValueError(
                        f"{p}: 除经度/纬度列外没有数值列；"
                        "请用 val_col= 指定数值列")
                # ★ 列名能当日期用时（lon,lat,2002-01-18,2002-02-17,…），
                # 把这些列全部当成历元 —— 不然只会读第一列，用户手里就只剩
                # 一个时次却看不出来。
                _rest_labels = [str(cols[j]).strip() for j in rest]
                if len(rest) > 1 and _looks_like_dates(_rest_labels):
                    i_val = list(rest)
                    note_val = (f"数值 = 其余 {len(rest)} 列（列名可当日期用："
                                f"{_rest_labels[0]} … {_rest_labels[-1]}），"
                                "按多时次读")
                else:
                    i_val = [rest[0]]
                    note_val = f"数值 = 第 {i_val[0] + 1} 列（位置默认）"
                    if len(rest) > 1:
                        note_val += ("；还有未使用的列 "
                                     + ",".join(str(j + 1) for j in rest[1:])
                                     + "（多时次请用 val_col=[列名或序号, …] 指定）")
        else:
            # ★ val_col 也可以是**一串**列说明符：每一个就是一個历元。
            # 真实散点时间序列的列名常常是 e2002/e2003 或 2002-01/2002-02，
            # 单个 val_col 根本选不全（以前这里遇到 list 直接 TypeError）。
            specs = list(val_col) if isinstance(val_col, (list, tuple)) \
                else [val_col]
            if not specs:
                raise ValueError("val_col 是空序列：至少要给一个数值列")
            i_val, _notes = [], []
            for spec in specs:
                idx, _note = _resolve_col(spec, cols, len(cols), "数值", 2)
                if int(idx) in (int(i_lon), int(i_lat)):
                    raise ValueError(
                        f"val_col 里的 {spec!r} 指向经度/纬度列（{cols[int(idx)]!r}）")
                i_val.append(int(idx))
                _notes.append(_note)
            note_val = "；".join(_notes) if len(_notes) > 1 else _notes[0]

        if isinstance(i_val, int):
            i_val = [i_val]

        lat = _to_float_array(df.iloc[:, i_lat], cols[i_lat], 1)
        lon = _to_float_array(df.iloc[:, i_lon], cols[i_lon], 1)
        vals = np.column_stack(
            [_to_float_array(df.iloc[:, k], cols[k], 1) for k in i_val])

        meta.update({
            "lat_column": str(cols[i_lat]), "lat_column_index": int(i_lat),
            "lon_column": str(cols[i_lon]), "lon_column_index": int(i_lon),
            "value_columns": [str(cols[k]) for k in i_val],
            "value_column_indices": [int(k) for k in i_val],
            "inference": [note_lat, note_lon, note_val],
            "margin_convention": "第1列=经度、第2列=纬度、其余=数值",
        })
        warnings.extend(meta["inference"])
        if i_lat == 1 and i_lon == 0 and val_col is None and lat_col is None \
                and lon_col is None:
            warnings.append(
                "按地理位置默认约定解释：第1列经度、第2列纬度；"
                "若文件实际是 (lat, lon, value)，请用 lon_col=1 lat_col=0 或 "
                "lon_col='lon' lat_col='lat' 明确指定")

        # ★ 数值列名若能当日期用（2002-01-18 / 2002-01 / 2002），直接当历元日期。
        # 这样散点的多时次也能给时间序列/趋势页一个**真实**时间轴；解析不出来就
        # 只给序号，绝不编造日期。
        if len(i_val) > 1:
            labels = [str(cols[k]).strip() for k in i_val]
            if _looks_like_dates(labels):
                meta["time"] = labels
                warnings.append(
                    "数值列名可以当日期用，已作为历元时间轴："
                    f"{labels[0]} … {labels[-1]}（共 {len(labels)} 个历元）")

    n_bad = int(np.count_nonzero(~np.isfinite(lat) | ~np.isfinite(lon)))
    if n_bad:
        keep = np.isfinite(lat) & np.isfinite(lon)
        warnings.append(f"丢弃 {n_bad} 行：坐标缺失或非有限")
        lat, lon, vals = lat[keep], lon[keep], vals[keep]

    # ⚠️ 长表（tidy）保护：一行一个"点×历元"、带一列日期的表，会被当成宽表解释
    # （只取第一列数值）——那不是报错，而是**悄悄读错**。坐标大量重复就说明很可能
    # 是长表，直接说清楚，别让用户拿着半份数据去分析。
    if lat.size:
        n_uniq = int(np.unique(np.round(np.column_stack([lat, lon]), 9),
                               axis=0).shape[0])
        if n_uniq < 0.6 * lat.size:
            warnings.append(
                f"坐标大量重复（{lat.size} 行里只有 {n_uniq} 个不同的 (lon,lat)）："
                "这看起来是**长表**（一行一个「点×历元」）。本读取器按宽表解释，"
                "只会取第一列数值 —— 请先在 Excel/pandas 里透视成宽表"
                "（同一坐标一行、每个历元一列），或用 (N, 2+ntime) 的 .npy。")

    if np.any(lon < 0) or np.any(lon >= 360.0):
        warnings.append(
            "经度存在 <0 或 >=360 的值，已按 mod 360 归一化到 [0, 360)；"
            "如果原数据是 -180..180 地理经度，这是预期的")
        lon = np.mod(lon, 360.0)

    if vals.shape[1] == 1:
        vals = vals[:, 0]
    if np.all(~np.isfinite(vals)):
        warnings.append("数值列全部是 NaN/Inf，请检查列选择是否正确")

    meta.update({
        "n_points": int(lat.size),
        "ntime": 1 if vals.ndim == 1 else int(vals.shape[1]),
        "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))] if lat.size else [],
        "lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))] if lon.size else [],
        "units": {"lat": "degrees_north (geocentric)",
                  "lon": "degrees_east [0,360)", "value": "user field units"},
        "warnings": warnings,
    })
    return lat, lon, vals, meta


def write_points(path, lat, lon, values, header: bool = True,
                 fmt: str = "%.10g", comment: Optional[str] = None,
                 delim=None, extra_cols: Optional[Mapping[str, Any]] = None,
                 **kw) -> Path:
    """Write scattered samples to csv / txt / dat / npy.

    Columns are ``lon, lat, value[, value2, ...]`` (the same convention
    :func:`read_points` assumes by default).  ``lat``/``lon`` are degrees; no
    normalisation is applied here, the numbers are written as given.

    Parameters
    ----------
    path : str or Path
        ``.csv`` (comma), ``.txt``/``.dat`` (whitespace), ``.npy`` (one 2-D
        float array), ``.gz`` allowed for the text forms.
    lat, lon : array_like
        Degrees, same length ``N``.
    values : array_like
        ``(N,)`` or ``(N, ntime)``.
    header : bool
        Write a ``lon,lat,value...`` header line (ignored for ``.npy``).
    fmt : str
        ``numpy.savetxt`` format, default ``%.10g`` (10 significant digits).
    comment : str, optional
        Extra ``#``-prefixed lines placed above the header.
    delim : str, optional
        Field separator; inferred from the extension when omitted.
    extra_cols : mapping, optional
        Additional ``name -> array`` columns appended on the right.

    Returns
    -------
    pathlib.Path
        The path actually written.
    """
    p = _p(path)
    ext = _suffix(p)
    if ext == ".gz":
        ext = Path(p.stem).suffix.lower()
    if ext not in (".csv", ".txt", ".dat", ".tsv", ".npy"):
        raise ValueError(
            f"write_points: 不支持的扩展名 {_suffix(p)!r}（文件 {p}）。"
            f"支持: {', '.join(POINT_EXTENSIONS[:4])}, .npy")

    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    if vals.shape[0] != lat.size or lat.size != lon.size:
        raise ValueError(
            f"长度不一致: lat {lat.size}, lon {lon.size}, values {vals.shape[0]}")

    if ext == ".npy":
        if str(p).lower().endswith(".gz"):
            raise ValueError("write_points: .npy 不支持 .gz 后缀")
        arr = np.column_stack([lon, lat, vals])
        if str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        return p

    names = [COL_LON, COL_LAT] + [COL_VAL if i == 0 else f"{COL_VAL}{i}"
                                  for i in range(vals.shape[1])]
    mat = [lon, lat] + [vals[:, k] for k in range(vals.shape[1])]
    if extra_cols:
        for k, v in extra_cols.items():
            v = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
            if v.size != lat.size:
                raise ValueError(f"extra_cols[{k!r}] 长度 {v.size} != {lat.size}")
            names.append(str(k))
            mat.append(v)
    data = np.column_stack(mat)

    if delim is None:
        delim = {".csv": ",", ".tsv": "\t", ".txt": " ", ".dat": " "}.get(ext, ",")
    sep = delim

    lines = []
    hdr = ("# " + comment.replace("\n", "\n# ")) if comment else None
    if hdr:
        lines.append(hdr)
    if header:
        lines.append(sep.join(names))
    with np.errstate(all="ignore"):
        body = "\n".join(
            sep.join(fmt % v if np.isfinite(v) else "nan" for v in row)
            for row in data)
    lines.append(body)
    _text_write(p, "\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# READ / WRITE  grid
# ---------------------------------------------------------------------------
def _clean_axis(v: np.ndarray, warnings: list, name: str) -> np.ndarray:
    """Tidy one coordinate axis read from disk.

    Two real-world annoyances are fixed here, both of which silently break a
    later ``unique`` / quadrature step:

    * a **float32** coordinate axis (very common in netCDF written by MATLAB or
      by ``float32`` model output) reads back as ``-89.00000190734863`` etc., so
      the axis is not equal-spaced any more and ``np.unique`` sees more nodes
      than there are;
    * an axis written as ``0, 0.9999999, 1.9999999, ...`` which should be an
      integer-degree grid.

    Near-equal-spaced axes are snapped to a common spacing when that moves every
    value by less than 1e-6 of the spacing.  Returns the axis unchanged when it
    is not near-equal-spaced (no silent resampling of irregular axes).
    """
    v = np.asarray(v, dtype=float).ravel()
    if v.size < 3:
        return v
    d = np.diff(v)
    med = float(np.median(d))
    if not np.isfinite(med) or med == 0.0:
        return v
    if float(np.max(np.abs(d - med))) > 1e-4 * abs(med):
        return v                                  # genuinely irregular: leave it
    step = float(round(med, 9))
    if step == 0.0:
        return v
    base = float(round(v[0] / step))
    snapped = (base + np.arange(v.size)) * step
    if float(np.max(np.abs(snapped - v))) > 1e-6 * abs(step):
        return v
    if not np.array_equal(snapped, v):
        warnings.append(
            f"{name} 轴不是严格等间距（可能来自 float32 坐标或十进制舍入），"
            f"已按步长 {step:g} 吸附到 {snapped[0]:g} 起的规则轴；"
            "这样 np.unique/求积规则才能按节点数工作")
    return snapped


def _read_grid_netcdf(path) -> tuple:
    """Open a netCDF/CF file and pull out ``lat``, ``lon`` and the data cube."""
    try:
        import xarray as xr
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "读取 .nc 需要 xarray + netCDF4，请执行: pip install xarray netCDF4"
        ) from exc

    p = _require_file(path)
    warnings: list = []
    with open_nc_dataset(p) as ds:
        dims = list(ds.dims)
        lat_name = _first_present((ds.coords, ds.variables),
                                  ("lat", "latitude", "LAT", "Lat", "ylat", "lat_deg"))
        lon_name = _first_present((ds.coords, ds.variables),
                                  ("lon", "longitude", "LON", "Lon", "xlon", "lon_deg"))
        if lat_name is None or lon_name is None:
            raise ValueError(
                f"{p}: 找不到纬度/经度变量（尝试过 lat/latitude, lon/longitude）。"
                f"文件中的变量: {list(ds.variables)}")
        lat_da = ds[lat_name]
        lon_da = ds[lon_name]

        time_name = None
        for cand in ("time", "t", "nt", "epoch"):
            if cand in ds.coords or cand in ds.dims:
                time_name = cand
                break

        data_vars = [v for v in ds.data_vars
                     if v not in (lat_name, lon_name, time_name)
                     and ds[v].dims and set(ds[v].dims) <= set(dims)]
        if not data_vars:
            raise ValueError(f"{p}: 没有找到可用的数据变量（只有坐标变量）")
        # prefer a variable that actually depends on lat and lon
        want = {lat_da.dims[0], lon_da.dims[0]}
        chosen = None
        for v in data_vars:
            if want <= set(ds[v].dims):
                chosen = v
                break
        if chosen is None:
            chosen = data_vars[0]
            warnings.append(
                f"数据变量 {chosen!r} 的维度 {ds[chosen].dims} 未同时包含 "
                f"{sorted(want)}；按原样读出，可能不是规则网格")
        da = ds[chosen]
        meta = _nc_var_meta(ds, chosen, lat_name, lon_name, time_name,
                            data_vars=data_vars)
        lat = np.asarray(lat_da.values, dtype=float).ravel()
        lon = np.asarray(lon_da.values, dtype=float).ravel()
        # order dims as (lat, lon, [time])
        order = [lat_da.dims[0], lon_da.dims[0]]
        if time_name is not None and time_name in da.dims:
            order.append(time_name)
        for d in da.dims:
            if d not in order:
                order.append(d)
                warnings.append(f"额外维度 {d!r} 被保留在输出数组的最后")
        arr = _cast_grid_dtype(da.transpose(*order).values, meta, warnings)
    return lat, lon, arr, meta, warnings


#: 超过这么多个值时**保留文件自己的精度**（float32），不再整块升到 float64。
#: 实测依据（CSR mascon 265 M 个值 / 112 MB）：整块 float32→float64 要 0.9 s、
#: 峰值内存多 1 GB；而所有数值路径本来就会按需转 float64
#: （`analysis()` / `fit_series_maps()` / `synthesis()` 入口都做 `dtype=float`），
#: 所以保留原精度不影响任何结果，只是少一次全量拷贝。
_KEEP_DTYPE_MIN = 20_000_000


def _cast_grid_dtype(values, meta: dict, warnings: list):
    """把读到的数组转成工作精度，并对大文件保留文件原精度（**写在元数据里**）。"""
    v = np.asarray(values)
    if v.dtype == np.float32 and v.size >= _KEEP_DTYPE_MIN:
        meta["dtype_policy"] = "float32"
        warnings.append(
            f"数据量 {v.size / 1e6:.3g} M 个值：保留文件原精度 float32"
            "（不整块升到 float64，省一次全量拷贝与一倍内存）；"
            "所有分析/拟合入口都会按需转 float64，结果不受影响")
        return v
    meta["dtype_policy"] = "float64"
    return np.asarray(v, dtype=float)


def _first_present(sources, names):
    """First name from ``names`` that exists in any of the ``sources`` (a tuple)."""
    for src in sources:
        for n in names:
            if n in src:
                return n
    return None


def _attr_scalar(v):
    """Keep a netCDF attribute as a Python scalar, or stringify it.

    ``isinstance(v, int)`` is **False** for ``np.int64``, which is what xarray
    hands back for an integer attribute -- so a naive check silently turns
    ``lmax = 60`` into the string ``'60'``, and every later ``== 60`` comparison
    fails while the value *looks* right when printed.  Hence the explicit numpy
    scalar branches (and the conversion to plain Python types, so the value
    behaves the same in ``json.dumps`` and in comparisons).
    """
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _read_grid_surfer(path) -> tuple:
    """Surfer ASCII grid (``DSAA``, best effort ``DSBB``)."""
    text = _text_read(path)
    lines = [ln.strip() for ln in text.splitlines()]
    head_idx = None
    kind = None
    for i, ln in enumerate(lines):
        tag = ln.upper()
        if tag.startswith("DSAA"):
            head_idx, kind = i, "DSAA"
            break
        if tag.startswith("DSBB"):
            head_idx, kind = i, "DSBB"
            break
    if head_idx is None:
        raise ValueError(
            f"{path}: 不是 Surfer ASCII 网格（未找到 DSAA/DSBB 头）。"
            "二维三列数据请用 read_points()。")
    warnings: list = []
    if kind == "DSBB":
        warnings.append("读到 DSBB 头：按 DSAA 的 (nx, ny) + 4 个边界字段解释，"
                        "仅尽力而为，请核对坐标范围")

    def grab(i):
        return [float(t) for t in lines[i].split()]

    try:
        nx, ny = (int(v) for v in grab(head_idx + 1)[:2])
        xlo, xhi = grab(head_idx + 2)[:2]
        ylo, yhi = grab(head_idx + 3)[:2]
        zlo, zhi = grab(head_idx + 4)[:2]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: Surfer 网格头解析失败: {exc}") from exc
    if nx < 2 or ny < 2:
        raise ValueError(f"{path}: Surfer 网格尺寸非法 nx={nx}, ny={ny}")

    vals = []
    for ln in lines[head_idx + 5:]:
        if not ln:
            continue
        vals.extend(float(t) for t in ln.split())
    if len(vals) < nx * ny:
        raise ValueError(
            f"{path}: Surfer 网格声明 {nx}x{ny}={nx * ny} 个值，实际只有 {len(vals)} 个")
    if len(vals) > nx * ny:
        warnings.append(f"多余 {len(vals) - nx * ny} 个数值被忽略")
    grid = np.asarray(vals[:nx * ny], dtype=float).reshape(ny, nx)

    lat = np.linspace(ylo, yhi, ny)
    lon = np.linspace(xlo, xhi, nx)
    if xlo > xhi or ylo > yhi:
        warnings.append("Surfer 头里的坐标范围是降序；已按 linspace 生成升序轴，"
                        "并同步翻转数据")
        if ylo > yhi:
            grid = grid[::-1, :]
            lat = np.sort(lat)
        if xlo > xhi:
            grid = grid[:, ::-1]
            lon = np.sort(lon)
    lat = np.sort(lat)

    meta = {
        "format": "surfer_ascii",
        "header": kind,
        "nx": int(nx), "ny": int(ny),
        "x_range": [float(min(xlo, xhi)), float(max(xlo, xhi))],
        "y_range": [float(min(ylo, yhi)), float(max(ylo, yhi))],
        "z_range": [float(zlo), float(zhi)],
        "notes": ("Surfer 的行序是 y 递增（南->北），本模块按同一顺序写出；"
                  "读入后纬度轴保证升序"),
    }
    return lat, lon, grid, meta, warnings


def write_field_series(path, lat_vec, lon_vec, times, cube, *, var: str = "ewh",
                       units: Optional[str] = None,
                       long_name: Optional[str] = None,
                       meta: Optional[Mapping[str, Any]] = None,
                       layout: str = "reference",
                       dtype: str = "float32") -> Path:
    """Write a **time series of grids** (a field cube) to netCDF.

    Two layouts, and the difference matters when comparing with someone else's
    product:

    ``'reference'`` (default)
        Exactly the layout of ``3_grids/CSR_GRACE_EWH_G300.nc`` and friends:
        dims ``(time, lat, lon)``, latitude **descending** (90 -> -90), longitude
        wrapped to ``[-180, 180)``, float32.  Global attributes are written with
        their plain names (``center``/``lmax``/``gauss_filter_km``/...) so the
        file is a drop-in neighbour of those files.
    ``'internal'``
        SHKit's own convention: dims ``(lat, lon, time)``, axes exactly as given.

    ``cube`` is always accepted in the **internal** order ``(nlat, nlon, ntime)``
    with ``lat_vec``/``lon_vec`` matching, so callers never have to pre-flip
    anything; the reordering (and its inverse on read) lives here alone.

    Parameters
    ----------
    times : TimeAxis or array_like of datetime64
        One epoch per time slice; must be strictly increasing.
    var, units, long_name : str
        Data-variable name and CF attributes.
    meta : mapping, optional
        Global attributes (``title``, ``center``, ``lmax``, ``gauss_filter_km``,
        ``grid_resolution_deg``, ``rho_earth_kgm3``, ...).  Written verbatim.
    dtype : str
        ``'float32'`` (default, matches the reference files and halves the size)
        or ``'float64'``.
    """
    from .timeaxis import TimeAxis

    p = _p(path)
    if _suffix(p) not in (".nc", ".nc4", ".cdf"):
        raise ValueError(
            f"write_field_series 只写 netCDF（{_suffix(p)!r} 不支持）："
            "多时次网格请用 .nc；单层网格用 write_grid")
    lat = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"cube 必须是 (nlat, nlon, ntime)，实际 {cube.shape}")
    if cube.shape[:2] != (lat.size, lon.size):
        raise ValueError(
            f"cube.shape={cube.shape} 的前两维与坐标轴（nlat={lat.size}, "
            f"nlon={lon.size}）不一致")
    tv = np.asarray(times.values if hasattr(times, "values") else times,
                    dtype="datetime64[s]").ravel()
    if tv.size != cube.shape[2]:
        raise ValueError(f"有 {tv.size} 个时间，但 cube 有 {cube.shape[2]} 层")
    if tv.size > 1 and not np.all(np.diff(tv.astype("int64")) > 0):
        raise ValueError("时间必须严格递增（先排序并去重）")

    if layout == "reference":
        if lat.size > 1 and lat[0] < lat[-1]:
            lat = lat[::-1].copy()
            cube = cube[::-1, :, :].copy()
        lw = np.mod(lon + 180.0, 360.0) - 180.0
        order = np.argsort(lw)
        lw = lw[order]
        cube = cube[:, order, :].copy()
        arr = np.moveaxis(cube, 2, 0)                 # (time, lat, lon)
        dims = ("time", "lat", "lon")
        coords = {"time": tv, "lat": lat, "lon": lw}
    elif layout == "internal":
        arr = cube
        dims = ("lat", "lon", "time")
        coords = {"time": tv, "lat": lat, "lon": lon}
    else:
        raise ValueError(f"layout 必须是 'reference' 或 'internal'，得到 {layout!r}")

    import xarray as xr

    da = xr.DataArray(np.asarray(arr, dtype=dtype), dims=dims, coords=coords,
                      name=var)
    da["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    da["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    if units:
        da.attrs["units"] = units
    if long_name:
        da.attrs["long_name"] = long_name
    ds = da.to_dataset()
    m = dict(meta or {})
    for k, v in m.items():
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            # netCDF attributes have no boolean type; store 0/1
            ds.attrs[k] = int(v)
        elif isinstance(v, (str, int, float, np.number)):
            ds.attrs[k] = v
        else:
            ds.attrs[k] = json.dumps(v, ensure_ascii=False, default=str)
    ds.attrs.setdefault("Conventions", "CF-1.8")
    ds.attrs["shkit_version"] = SHKIT_VERSION
    ds.attrs["shkit_layout"] = layout
    ds.attrs.setdefault("source",
                        f"SHKit {SHKIT_VERSION} (shkit.io.write_field_series)")
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    _write_nc(ds, p)
    return p


def read_field_series(path, var: Optional[str] = None) -> tuple:
    """Read a time series of grids back, **normalised to SHKit's convention**.

    Returns ``(lat_vec, lon_vec, times, cube, meta)`` where ``lat_vec`` is
    ascending, ``lon_vec`` lies in ``[0, 360)``, ``cube`` is
    ``(nlat, nlon, ntime)`` and ``times`` is a :class:`~shkit.timeaxis.TimeAxis`.

    Every convention difference against the reference files (descending latitude,
    ``[-180, 180)`` longitude, ``(time, lat, lon)`` dimension order) is undone
    here, so downstream code only ever sees one convention.  The original order is
    recorded in ``meta['read_from_layout']``.
    """
    lat, lon, arr, meta = read_grid(path, var=var)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    ntime = arr.shape[2]

    tmeta = meta.get("time") or meta.get("times")
    times = None
    if tmeta is not None:
        times = time_axis_from_meta(tmeta, units=meta.get("time_units"),
                                    calendar=meta.get("time_calendar"))
    if times is None or len(times) != ntime:
        times = TimeAxis.from_index(ntime)

    # 经度折到 [0, 360)（参考文件用 [-180, 180)）
    lon = np.mod(np.asarray(lon, dtype=float), 360.0)
    order = np.argsort(lon)
    lon = lon[order]
    arr = arr[:, order, :]
    meta = dict(meta)
    # Flatten the netCDF global attributes to the top level so a caller can just
    # ask for meta['lmax'] / meta['gauss_filter_km'] / meta['center']; the nested
    # original stays available under meta['attrs'].
    for k, v in (meta.get("attrs") or {}).items():
        meta.setdefault(k, v)
    meta["read_from_layout"] = ("reference-like"
                                if meta.get("lat_order_flipped") else "as-written")
    meta["lon_convention"] = "[0, 360)"
    return lat, lon, times, arr, meta


def read_grid(path, var: Optional[str] = None, lazy_time: bool = False) -> tuple:
    """Read a regular lat/lon grid from nc / grd / npy / csv / txt.

    Parameters
    ----------
    path : str or Path
        ``.nc``/``.nc4``/``.cdf`` (netCDF-CF via xarray), ``.grd`` (Surfer
        ASCII ``DSAA``, ``DSBB`` best effort), ``.npy`` (a stack of
        ``lat_vec, lon_vec, grid`` or just the data cube), or a two/three-column
        ``.csv``/``.txt``/``.dat`` of ``lat lon value`` that is pivoted into a
        grid.
    var : str, optional
        netCDF variable to read; the first lat/lon-dependent variable is used
        otherwise (the choice is reported in ``meta['variable']``).
    lazy_time : bool
        **多时次 nc 的懒加载**（默认关）。开启时只读第一个时次就返回，其余时次
        按需/后台补（返回 :class:`LazyTimeCube`，用法与数组一致）。实测把
        112 MB / 256 时次 mascon 的首屏从 3.5 s 降到 ~0.3 s。只在
        "nc + ntime > 1 + 经度轴本来就规范"时生效，其余情况**静默回退**到整块读
        （回退理由写进 ``meta['lazy_fallback']``）。

    Returns
    -------
    lat_vec, lon_vec, grid, meta : ndarray, ndarray, ndarray or LazyTimeCube, dict
        ``lat_vec`` strictly increasing, ``lon_vec`` increasing, ``grid`` of
        shape ``(nlat, nlon)`` or ``(nlat, nlon, ntime)``.  A descending
        latitude axis on disk is flipped (data included) and
        ``meta['lat_order_flipped'] = True`` records it.

    Raises
    ------
    FileNotFoundError, ValueError
        Missing file / unknown extension / the points do not actually form a
        rectangular grid (in which case :func:`read_points` is suggested).
    """
    p = _require_file(path)
    ext = _suffix(p)
    if ext not in GRID_EXTENSIONS:
        raise ValueError(
            f"read_grid: 不支持的扩展名 {ext!r}（文件 {p}）。"
            f"支持: {', '.join(GRID_EXTENSIONS)}")
    warnings: list = []
    meta: dict = {"source_file": str(p)}

    if ext in (".nc", ".nc4", ".cdf"):
        # 指定了变量就只读那一个：以前先跑一遍自动探测、再跑一遍指定变量，
        # 同一份大文件被**读两遍**（非 ASCII 路径还要复制两遍 112 MB 的临时副本）。
        lazy = None
        if lazy_time:
            lazy = _try_lazy_grid(p, var, meta, warnings)
        if lazy is not None:
            lat, lon, arr, ncmeta, w = lazy
        elif var is None:
            lat, lon, arr, ncmeta, w = _read_grid_netcdf(p)
        else:
            lat, lon, arr, ncmeta, w = _read_grid_netcdf_var(p, var)
        meta.update(ncmeta)
        warnings.extend(w)
        grid = arr
    elif ext == ".grd":
        lat, lon, grid, gmeta, warnings = _read_grid_surfer(p)
        meta.update(gmeta)
    elif ext == ".npy":
        arr = np.load(p, allow_pickle=False)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            raise ValueError(f"{p}: .npy 是 object 数组，无法安全读取")
        arr = np.asarray(arr, dtype=float)
        meta["array_shape"] = list(arr.shape)
        if arr.ndim == 2 and arr.shape[0] >= 3 and \
                np.allclose(np.diff(arr[:, 0]), np.diff(arr[:, 0])[0]) and \
                np.allclose(arr[:, 1], arr[0, 1]):
            # (3, nlon): [lon; lat; value] row layout
            lon, lat, grid = arr[0], arr[1], arr[2:].reshape(-1, arr.shape[1])
            warnings.append(".npy 按 (3, nlon) 行布局 [lon; lat; value] 解释")
        elif arr.ndim == 3 and arr.shape[0] >= 3:
            lon, lat, grid = arr[0], arr[1], arr[2:]
            if grid.shape[0] == 1:
                grid = grid[0]
            warnings.append(".npy 按 (3, nlat, nlon) 行布局 [lon; lat; grid] 解释")
        elif arr.ndim == 2:
            raise ValueError(
                f"{p}: (nlat, nlon) 的 .npy 不含坐标轴。请用 .npz 保存 "
                "lat/lon/grid，或提供 lat_vec/lon_vec 参数改用 read_grid_npy()；"
                "散点数据请用 read_points()")
        else:
            raise ValueError(f"{p}: 无法解释的 .npy shape {arr.shape}")
        meta["format"] = "npy"
    else:                                   # csv / txt / dat
        df, tmeta, w = _read_table(p)
        meta.update(tmeta)
        warnings.extend(w)
        cols = list(df.columns)
        i_lat, note_lat = _resolve_col(None, cols, len(cols), "纬度", 0)
        i_lon, note_lon = _resolve_col(None, cols, len(cols), "经度", 1)
        i_val, note_val = _resolve_col(None, cols, len(cols), "数值", 2)
        warnings.extend([note_lat + "（三列网格文件按第1列纬度、第2列经度）",
                         note_lon, note_val])
        lat_c = _to_float_array(df.iloc[:, i_lat], cols[i_lat], 1)
        lon_c = _to_float_array(df.iloc[:, i_lon], cols[i_lon], 1)
        val_c = _to_float_array(df.iloc[:, i_val], cols[i_val], 1)
        ok, lat_u, lon_u = _looks_like_grid(lat_c, lon_c)
        if not ok:
            raise ValueError(
                f"{p}: 三列数据不构成完整规则网格（{np.unique(lat_c).size} 个纬度 x "
                f"{np.unique(lon_c).size} 个经度 != {lat_c.size} 行）。"
                "散点数据请改用 read_points()；若确实是网格但缺少某些格点，"
                "请补齐或改用 read_points() 后自行插值。")
        li = np.unique(lat_c, return_inverse=True)[1]
        loi = np.unique(lon_c, return_inverse=True)[1]
        grid = np.full((lat_u.size, lon_u.size), np.nan)
        grid[li, loi] = val_c
        if np.isnan(grid).any():            # pragma: no cover - guarded above
            raise ValueError(f"{p}: 网格存在空缺单元")
        lat, lon = lat_u, lon_u
        meta["format"] = "text_grid"
        meta["inference"] = [note_lat, note_lon, note_val]

    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    # 大文件在读取阶段已按 _cast_grid_dtype 保留了 float32；这里**不要**再统一升到
    # float64 —— 那会把省下来的 0.9 s 与一倍内存又花回去（实测被这一步抵消）。
    # 懒加载容器要**原样保留**：np.asarray() 会触发 __array__ → 全量补齐，
    # 那正好把懒加载的意义抹掉。
    if isinstance(grid, LazyTimeCube):
        pass
    else:
        grid = np.asarray(grid)
        if grid.dtype != np.float32:
            grid = np.asarray(grid, dtype=float)
    if grid.ndim == 2:
        pass
    elif grid.ndim == 3:
        pass
    else:
        raise ValueError(f"{p}: 数据必须是 2 维或 3 维，实际 {grid.ndim} 维")
    if grid.shape[0] != lat.size or grid.shape[1] != lon.size:
        raise ValueError(
            f"{p}: 网格 shape {grid.shape} 与坐标轴（nlat={lat.size}, "
            f"nlon={lon.size}）不一致")

    lat, grid = _flip_grid_if_needed(lat, grid, meta, warnings)
    if np.any(lon < 0) or np.any(lon >= 360.0):
        warnings.append("经度已按 mod 360 归一化到 [0, 360)")
        order = np.argsort(np.mod(lon, 360.0))
        lon = np.mod(lon, 360.0)[order]
        grid = grid[:, order, ...]

    meta.update({
        "nlat": int(lat.size), "nlon": int(lon.size),
        "ntime": 1 if grid.ndim == 2 else int(grid.shape[2]),
        "lat_range": [float(lat.min()), float(lat.max())] if lat.size else [],
        "lon_range": [float(lon.min()), float(lon.max())] if lon.size else [],
        "lat_step_deg": float(np.median(np.diff(lat))) if lat.size > 1 else None,
        "lon_step_deg": float(np.median(np.diff(lon))) if lon.size > 1 else None,
        "regular_lat": _regular_spacing(lat), "regular_lon": _regular_spacing(lon),
        "units": {"lat": "degrees_north (geocentric)",
                  "lon": "degrees_east [0,360)",
                  "value": "user field units"},
        "warnings": warnings,
        "grid": (lat, lon, grid),
    })
    if not meta["regular_lat"] or not meta["regular_lon"]:
        warnings.append("坐标轴不是等间距的；SH 分析请相应选择权重规则"
                        "（rule='grid' 对非等距网格同样正确）")
    return lat, lon, grid, meta


def _try_lazy_grid(path, var, meta: dict, warnings: list):
    """试着用懒加载打开一个多时次 nc；不适用就返回 ``None``（**并写明原因**）。

    只在这一种情形下生效：nc + 有 time 维 + ntime > 1 + 变量本来就是 float32 或
    可安全转 float32 + 经度轴不需要重排（重排要逐切片做列置换，不值得）。
    纬度降序是支持的：翻转作为读取时的一个行索引交给 :class:`LazyTimeCube`，
    所以 ``meta['lat_order_flipped']`` 与整块读**完全一致**。
    """
    warnings: list = warnings
    try:
        ds, closer = open_nc_dataset_lazy(path)
    except Exception as exc:                                     # noqa: BLE001
        meta["lazy_fallback"] = f"打不开（{type(exc).__name__}）"
        warnings.append(f"懒加载回退到整块读：{meta['lazy_fallback']}")
        return None
    try:
        return _build_lazy_grid(ds, closer, path, var, meta, warnings)
    except _LazyNotApplicable as exc:
        closer()
        meta["lazy_fallback"] = str(exc)
        warnings.append(f"懒加载回退到整块读：{exc}")
        return None
    except Exception:                                            # noqa: BLE001
        closer()
        raise


class _LazyNotApplicable(Exception):
    """懒加载在这个文件/变量上不适用（回退到整块读，理由写进 meta）。"""


def _build_lazy_grid(ds, closer, path, var, meta: dict, warnings: list):
    lat_name = _first_present((ds.coords, ds.variables),
                              ("lat", "latitude", "LAT", "Lat", "ylat", "lat_deg"))
    lon_name = _first_present((ds.coords, ds.variables),
                              ("lon", "longitude", "LON", "Lon", "xlon", "lon_deg"))
    if lat_name is None or lon_name is None:
        raise _LazyNotApplicable(f"{path}: 找不到 lat/lon 变量")
    time_name = None
    for cand in ("time", "t", "nt", "epoch"):
        if cand in ds.coords or cand in ds.dims:
            time_name = cand
            break
    if time_name is None:
        raise _LazyNotApplicable("文件里没有 time 维")
    # 选变量：与 _read_grid_netcdf 同一套规则（优先同时依赖 lat 与 lon 的）
    if var is not None:
        if var not in ds.data_vars:
            raise _LazyNotApplicable(
                f"变量 {var!r} 不存在（可用 {list(ds.data_vars)}）")
        chosen = var
    else:
        want = {ds[lat_name].dims[0], ds[lon_name].dims[0]}
        cands = [v for v in ds.data_vars
                 if v not in (lat_name, lon_name, time_name)
                 and want <= set(ds[v].dims)]
        if not cands:
            raise _LazyNotApplicable("没有同时依赖 lat/lon 的数据变量")
        chosen = cands[0]
    da = ds[chosen]
    if time_name not in da.dims:
        raise _LazyNotApplicable("数据变量没有 time 维")
    ntime = int(ds.sizes[time_name])
    if ntime <= 1:
        raise _LazyNotApplicable("只有一个时次（懒加载没有意义）")
    if da.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise _LazyNotApplicable(f"变量 dtype 是 {da.dtype}，不做隐式转换")
    lat = np.asarray(ds[lat_name].values, dtype=float).ravel()
    lon = np.asarray(ds[lon_name].values, dtype=float).ravel()
    if np.any(lon < 0) or np.any(lon >= 360.0):
        raise _LazyNotApplicable("经度轴需要重排（懒得逐切片做列置换）")
    # 纬度降序 → 读取时翻行；lat 轴同步翻转，与整块读口径一致
    row_flip = bool(lat.size > 1 and lat[0] > lat[-1])
    if row_flip:
        meta["lat_order_flipped"] = True
        warnings.append(
            "输入网格的纬度轴是降序（北->南），已翻转为升序；数据同步翻转。"
            "meta['lat_order_flipped'] = True")
        lat = lat[::-1].copy()
    else:
        meta.setdefault("lat_order_flipped", False)
    shape = (lat.size, lon.size, ntime)
    lat_dim, lon_dim = ds[lat_name].dims[0], ds[lon_name].dims[0]
    dims = list(da.dims)
    if set(dims) != {lat_dim, lon_dim, time_name}:
        raise _LazyNotApplicable(
            f"变量维度 {dims} 不是 (lat, lon, time) 三元组")
    perm = (dims.index(lat_dim), dims.index(lon_dim), dims.index(time_name))
    cube = LazyTimeCube(ds, closer, chosen, shape, dtype=np.dtype(da.dtype),
                        t0=0, row_flip=row_flip, perm=perm)
    meta.update(_nc_var_meta(ds, chosen, lat_name, lon_name, time_name))
    meta["dtype_policy"] = str(np.dtype(da.dtype))
    meta["lazy_time"] = {"ntime": ntime, "block": LazyTimeCube.PREFETCH_BLOCK}
    warnings.append(
        f"多时次大文件：已按**懒加载**打开 —— 首屏只读第 1 个时次"
        f"（共 {ntime} 个），其余在后台按块补（补齐后与整块读逐值一致）")
    # 返回**空列表**：这些提示已经写进调用方的 warnings 了，再传一份会重复
    return lat, lon, cube, meta, []


def _nc_var_meta(ds, name: str, lat_name, lon_name, time_name,
                 data_vars=None) -> dict:
    """Shared metadata for one netCDF data variable (single place, both readers).

    Includes the three things that used to be easy to lose and each of which broke
    something real:

    * ``time`` / ``time_units`` / ``time_calendar`` —— 数值型时间坐标（CF
      ``<unit> since <epoch>``，属性名还可能写成大写 ``Units``）必须连 units
      一起带出去，否则下游只拿到一串裸数字、日期解不出来；
    * ``variable_units`` —— 文件声明的单位常常不是米（CSR mascon 是 ``cm``），
      而 SHKit 的换算公式按米；
    * ``nc_variables`` —— 文件里的**全部**数据变量名。GUI 的变量下拉框以前为了
      拿这份名单会**再开一次同一个文件**，与读数据的 worker 并发，
      netCDF4 直接抛 ``RuntimeError: NetCDF: Not a valid ID``。
    """
    da = ds[name]
    # 下拉框只该列出**真正的网格变量**：像 mascon 的 `time_bounds`（只有 time 维）
    # 也出现在 data_vars 里，选中它只会得到一个莫名其妙的网格。没有匹配的再退回全列。
    try:
        want = {ds[lat_name].dims[0], ds[lon_name].dims[0]}
        grid_vars = [str(v) for v in ds.data_vars if want <= set(ds[v].dims)]
    except Exception:                                            # noqa: BLE001
        grid_vars = []
    meta = {
        "format": "netcdf",
        "variable": name,
        "dims": list(da.dims),
        "attrs": {k: _attr_scalar(v) for k, v in ds.attrs.items()},
        "variable_attrs": {k: _attr_scalar(v) for k, v in da.attrs.items()},
        "lat_name": lat_name, "lon_name": lon_name, "time_name": time_name,
        "nc_variables": grid_vars or [str(v) for v in ds.data_vars],
    }
    if data_vars is not None:
        meta["data_variables"] = list(data_vars)
    meta["variable_units"] = attr_ci(da.attrs, "units")
    meta["variable_long_name"] = attr_ci(da.attrs, "long_name")
    if time_name is not None and time_name in da.dims:
        tvar = ds[time_name]
        tv = np.asarray(tvar.values)
        meta["time"] = (tv.astype(str).tolist() if tv.dtype.kind in "mMSO"
                        else tv.tolist())
        tattrs = dict(getattr(tvar, "attrs", {}) or {})
        meta["time_units"] = attr_ci(tattrs, "units")
        meta["time_calendar"] = attr_ci(tattrs, "calendar")
    return meta


def _read_grid_netcdf_var(path, var: str):
    """Read ``var`` from a netCDF file using the same lat/lon detection.

    与 :func:`_read_grid_netcdf` 共用 :func:`_nc_var_meta`，所以**指定变量**读出来的
    元数据（时间轴、变量单位、变量名单）与自动探测那条路完全一致 —— 以前这条路
    只给一个最小 meta，用户在下拉框里换个变量就会丢掉时间轴。
    """
    warnings: list = []
    with open_nc_dataset(path) as ds:
        if var not in ds.data_vars:
            raise ValueError(
                f"{path}: 变量 {var!r} 不存在。可用: {list(ds.data_vars)}")
        lat_name = _first_present((ds.coords, ds.variables),
                                  ("lat", "latitude", "LAT", "Lat", "lat_deg"))
        lon_name = _first_present((ds.coords, ds.variables),
                                  ("lon", "longitude", "LON", "Lon", "lon_deg"))
        if lat_name is None or lon_name is None:
            raise ValueError(f"{path}: 找不到 lat/lon 变量")
        time_name = None
        for cand in ("time", "t", "nt", "epoch"):
            if cand in ds.coords or cand in ds.dims:
                time_name = cand
                break
        order = [ds[lat_name].dims[0], ds[lon_name].dims[0]]
        if time_name is not None and time_name in ds[var].dims:
            order.append(time_name)
        for d in ds[var].dims:
            if d not in order:
                order.append(d)
        arr = np.asarray(ds[var].transpose(*order).values)
        lat = np.asarray(ds[lat_name].values, dtype=float).ravel()
        lon = np.asarray(ds[lon_name].values, dtype=float).ravel()
        meta = _nc_var_meta(ds, var, lat_name, lon_name, time_name)
        arr = _cast_grid_dtype(arr, meta, warnings)
    return lat, lon, arr, meta, warnings

def write_grid(path, lat_vec, lon_vec, grid, var: str = "value",
               meta: Optional[Mapping[str, Any]] = None,
               long_name: Optional[str] = None,
               units: Optional[str] = None,
               comment: Optional[str] = None,
               fmt: str = "%.10g") -> Path:
    """Write a regular lat/lon grid to nc / grd / csv / txt / npy.

    Parameters
    ----------
    path : str or Path
        ``.nc`` (netCDF4 via xarray, CF-ish, with ``shkit`` and ``meta`` global
        attributes), ``.grd`` (Surfer ASCII ``DSAA``), ``.csv``/``.txt``
        (three columns ``lat,lon,value``), ``.npy`` (a stack of
        ``lat_vec, lon_vec, grid``/``grid``).
    lat_vec, lon_vec : array_like
        Degrees.  ``lat_vec`` is written **increasing**; if it is given
        decreasing it is flipped (data included) and
        ``meta['lat_order_flipped'] = True`` is recorded in the caller's ``meta``
        dict (the grid itself is stored increasing, which is also the written
        order of a Surfer ``.grd``).
    grid : array_like
        ``(nlat, nlon)`` or ``(nlat, nlon, ntime)``.
    var : str
        netCDF variable name (default ``value``).
    meta : mapping, optional
        Free-form provenance; serialised into the netCDF global attributes and,
        for the text formats, into the ``#`` comment block.  Mutated in place
        with ``lat_order_flipped``.
    long_name, units : str, optional
        CF attributes of the data variable.

    Returns
    -------
    pathlib.Path
        The path actually written.
    """
    p = _p(path)
    ext = _suffix(p)
    if ext == ".gz":
        ext = Path(p.stem).suffix.lower()
    if ext not in GRID_EXTENSIONS:
        raise ValueError(
            f"write_grid: 不支持的扩展名 {_suffix(p)!r}（文件 {p}）。"
            f"支持: {', '.join(GRID_EXTENSIONS)}")

    lat = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    grid = np.asarray(grid, dtype=float)
    if grid.ndim not in (2, 3):
        raise ValueError(f"grid 必须是 2 维或 3 维，实际 {grid.ndim} 维")
    if grid.shape[0] != lat.size or grid.shape[1] != lon.size:
        raise ValueError(
            f"grid shape {grid.shape} 与坐标轴（nlat={lat.size}, nlon={lon.size}）不一致")
    flipped = bool(lat.size > 1 and lat[0] > lat[-1])
    if flipped:
        lat = lat[::-1].copy()
        grid = grid[::-1, ...].copy()
        if isinstance(meta, dict):
            meta["lat_order_flipped"] = True
    m = dict(meta or {})

    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)

    if ext in (".nc", ".nc4", ".cdf"):
        import xarray as xr

        dims = ("lat", "lon") if grid.ndim == 2 else ("lat", "lon", "time")
        da = xr.DataArray(grid, dims=dims,
                          coords={"lat": lat, "lon": lon},
                          name=var)
        da["lat"].attrs = {"units": "degrees_north",
                           "standard_name": "latitude",
                           "long_name": "geocentric latitude"}
        da["lon"].attrs = {"units": "degrees_east",
                           "standard_name": "longitude",
                           "long_name": "longitude"}
        if units:
            da.attrs["units"] = units
        if long_name:
            da.attrs["long_name"] = long_name
        for k, v in _json_meta(m).items():
            try:
                da.attrs[f"shkit_{k}"] = v if isinstance(
                    v, (str, int, float)) else json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                da.attrs[f"shkit_{k}"] = str(v)
        da.attrs["shkit_version"] = SHKIT_VERSION
        ds = da.to_dataset()
        ds.attrs["Conventions"] = "CF-1.8"
        ds.attrs["title"] = f"shkit {var} on a regular lat/lon grid"
        ds.attrs["source"] = f"SHKit {SHKIT_VERSION} (shkit.io.write_grid)"
        ds.attrs["history"] = comment or "written by shkit.io.write_grid"
        ds.attrs["shkit_meta"] = json.dumps(_json_meta(m), ensure_ascii=False)
        _write_nc(ds, p)
        return p

    if ext == ".grd":
        ny, nx = lat.size, lon.size
        lines = ["DSAA", f"{nx} {ny}",
                 f"{lon.min():.10g} {lon.max():.10g}",
                 f"{lat.min():.10g} {lat.max():.10g}"]
        finite = grid[np.isfinite(grid)]
        zlo = float(finite.min()) if finite.size else 1.0
        zhi = float(finite.max()) if finite.size else 1.0
        lines.append(f"{zlo:.10g} {zhi:.10g}")
        if comment or m:
            for ln in (comment or "").splitlines():
                lines.append(f"# {ln}")
        if grid.ndim == 3:
            raise ValueError(
                "write_grid: Surfer .grd 只有单层，无法写 3 维网格；"
                "请对每个时次分别写出，或改用 .nc")
        g = np.where(np.isfinite(grid), grid, 1.701410009187828e38)
        body = "\n".join(" ".join(fmt % v for v in row) for row in g)
        _text_write(p, "\n".join(lines) + "\n" + body + "\n")
        return p

    if ext == ".npy":
        if grid.ndim == 2:
            arr = np.vstack([lon[None, :], lat[:, None], grid])
        else:
            arr = np.concatenate([lon[None, None, :].repeat(lat.size, 0),
                                  lat[:, None, None].repeat(lon.size, 2),
                                  grid], axis=0)
        np.save(p, np.asarray(arr, dtype=float))
        return p

    # csv / txt / dat: three columns
    if grid.ndim == 3:
        if grid.shape[2] != 1:
            raise ValueError(
                "write_grid: 三列文本格式只支持单个时次；"
                "多时次请用 .nc 或逐个时次写出")
        grid = grid[:, :, 0]
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    out = np.column_stack([LA.ravel(), LO.ravel(), grid.ravel()])
    lines = []
    if comment:
        lines.append("# " + comment.replace("\n", "\n# "))
    if m:
        lines.append("# shkit_meta " + json.dumps(_json_meta(m), ensure_ascii=False))
    lines.append("lat,lon,value")
    with np.errstate(all="ignore"):
        lines.append("\n".join(
            ",".join(fmt % v if np.isfinite(v) else "nan" for v in row)
            for row in out))
    _text_write(p, "\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# coeffs
# ---------------------------------------------------------------------------
def _infer_nmax_from_rows(nrows: int) -> int:
    """Solve ``2*(L+1)(L+2)/2 == nrows`` for the integer ``L``."""
    L = int(round((-3.0 + np.sqrt(1.0 + 4.0 * nrows)) / 2.0))
    for cand in (L, L + 1, L - 1):
        if cand >= 0 and 2 * (cand + 1) * (cand + 2) // 2 == nrows:
            return cand
    raise ValueError(
        f"三角布局的行数 {nrows} 不能写成 2*(L+1)(L+2)/2；"
        "请用 nmax= 明确指定")


def detect_coeff_layout(path, nmax: Optional[int] = None) -> str:
    """Guess the coefficient layout of a file (``'triangle'`` etc.).

    Useful before calling :func:`read_coeffs` when you want to know what the
    heuristic decided.  Raises :class:`ValueError` for an unknown extension.
    """
    p = _require_file(path)
    ext = _suffix(p)
    if ext not in COEFF_EXTENSIONS:
        raise ValueError(
            f"detect_coeff_layout: 不支持的扩展名 {ext!r}（文件 {p}）。"
            f"支持: {', '.join(COEFF_EXTENSIONS)}")
    return _detect_layout(p, ext, nmax)[0]


def _detect_layout(p: Path, ext: str, nmax: Optional[int],
                   head_only: bool = False) -> tuple:
    """Return ``(layout, nmax_hint, payload)``; ``payload`` caches what was read."""
    if ext == ".npy":
        arr = np.load(p, allow_pickle=False)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 3 and arr.shape[0] == 2:
            return "matrix", arr.shape[1] - 1, arr
        if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
            return "matrix", arr.shape[1] - 1, arr
        if arr.ndim == 2:
            NC = 0
            L = None
            for cand in range(0, 4097):
                if 2 * (cand + 1) * (cand + 2) // 2 == arr.shape[0]:
                    L = cand
                    break
            if L is not None and (nmax is None or L == nmax):
                return "triangle", L, arr
        return "matrix", (arr.shape[1] - 1 if arr.ndim >= 2 else None), arr

    if ext == ".npz":
        z = np.load(p, allow_pickle=False)
        if "C" in z.files and "S" in z.files:
            return "matrix", np.asarray(z["C"]).shape[0] - 1, z
        if "flat_cs" in z.files:
            arr = np.asarray(z["flat_cs"], dtype=float)
            return "triangle", None, {"flat_cs": arr,
                                      "meta": _npz_meta(z)}
        if "triangle" in z.files:
            arr = np.asarray(z["triangle"], dtype=float)
            return "triangle", None, {"flat_cs": arr, "meta": _npz_meta(z)}
        raise ValueError(
            f"{p}: .npz 必须包含 'C'/'S'（或 'flat_cs'/'triangle'）数组，"
            f"实际内容: {list(z.files)}")

    if ext == ".gfc":
        return "gfc", None, None

    # text: peek at the first data-bearing line
    text = _text_read(p)
    first = None
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        first = s
        break
    if first is None:
        raise ValueError(f"{p}: 文件没有有效数据行")
    toks = [t for t in re.split(r"[,\s;]+", first) if t]
    low = first.lower()
    if low.startswith("gfc") or low.startswith("gfct") or low.startswith("end"):
        return "gfc", None, None
    # header or gmfcsv?
    numeric = True
    try:
        for t in toks[:4]:
            float(t)
    except ValueError:
        numeric = False
    if not numeric and len(toks) >= 4:
        norm = [t.strip().lower() for t in toks[:4]]
        if norm[0] in ("n", "degree", "l") and norm[1] in ("m", "order"):
            return "gmfcsv", None, None
    if numeric and len(toks) == 4 and \
            all(float(t).is_integer() for t in toks[:2]) and \
            abs(float(toks[0])) < 1e6:
        # could still be a 4-row triangle chunk; only trust it when nmax agrees
        pass
    # count data rows
    rows = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        toks2 = [t for t in re.split(r"[,\s;]+", s) if t]
        try:
            rows.append([float(t) for t in toks2])
        except ValueError:
            rows.append(None)
    data = [r for r in rows if r is not None]
    if not data:
        raise ValueError(f"{p}: 文件没有可解析的数值行")
    nrow, ncol = len(data), max(len(r) for r in data)
    if ncol == 4:
        try:
            L4 = _infer_nmax_from_rows(nrow)
        except ValueError:
            L4 = None
        if L4 is None:
            return "gmfcsv", None, None
        if nmax is not None:
            return ("triangle", nmax, None) if 2 * (nmax + 1) * (nmax + 2) // 2 == nrow \
                else ("gmfcsv", None, None)
        return "triangle", L4, None
    L = None
    try:
        L = _infer_nmax_from_rows(nrow)
    except ValueError:
        L = None
    if nmax is not None:
        if nrow == 2 * (nmax + 1) * (nmax + 2) // 2:
            return "triangle", nmax, None
        if nrow == nmax + 1:
            return "matrix", nmax, None
        return "triangle", None, None
    if L is not None:
        return "triangle", L, None
    if nrow >= 1 and ncol >= 3:
        return "gmfcsv", None, None
    raise ValueError(f"{p}: 无法判断系数布局（{nrow} 行 x {ncol} 列）")


def _npz_meta(z) -> dict:
    if "meta" in z.files:
        try:
            v = z["meta"]
            if getattr(v, "dtype", None) is not None and v.dtype.kind in "US":
                return json.loads(str(v.item()) if v.ndim == 0 else str(v))
            if v.ndim == 0:
                return json.loads(str(v.item()))
        except Exception:
            pass
    return {}


def _read_triangle_text(p: Path, nmax: Optional[int], delim=None) -> tuple:
    """Read a ``2*NC x ntime`` triangle file (with or without a header)."""
    text = _text_read(p)
    header_lines = []
    data_rows = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            if s.startswith("#"):
                header_lines.append(s.lstrip("#").strip())
            continue
        toks = [t for t in re.split(r"[,\s;]+", s) if t]
        try:
            data_rows.append([float(t) for t in toks])
        except ValueError:
            header_lines.append(s)
    if not data_rows:
        raise ValueError(f"{p}: 没有数值行")
    ncol = max(len(r) for r in data_rows)
    if any(len(r) != ncol for r in data_rows):
        raise ValueError(f"{p}: 各行列数不一致，不是纯三角系数表")
    arr = np.asarray(data_rows, dtype=float)
    if arr.shape[1] == 1:
        arr = arr[:, 0]
    L = nmax if nmax is not None else _infer_nmax_from_rows(arr.shape[0])
    return arr, L, header_lines


def _header_field(header_lines, key: str):
    """Pull ``key = value`` / ``# key: value`` out of a comment header."""
    pat = re.compile(rf"^\s*{re.escape(key)}\s*[:=]\s*(.+)$", re.IGNORECASE)
    for ln in header_lines:
        m = pat.match(ln)
        if m:
            return m.group(1).strip()
    return None


def read_coeffs(path, nmax: Optional[int] = None, layout: str = "auto") \
        -> SHCoeffs:
    """Read spherical harmonic coefficients from any supported layout.

    Parameters
    ----------
    path : str or Path
        Recognised extensions: ``.sh .txt .csv .dat .tsv .gfc .npy .npz``
        (``.gz`` allowed on the text forms).
    nmax : int, optional
        Maximum degree.  Required only when the layout cannot be inferred
        (e.g. a triangle file whose row count does not match any degree, or a
        matrix ``.npy``).  Overrides the inference when given.
    layout : {'auto', 'triangle', 'matrix', 'gmfcsv', 'npy', 'gfc'}
        ``'auto'`` inspects the file (see :func:`detect_coeff_layout`).
        ``'npy'`` forces the numpy container path (``.npy``/``.npz``).

    Returns
    -------
    SHCoeffs
        ``meta`` always carries ``source_file``, ``layout``, ``nmax`` and
        ``ntime``; a ``warnings`` list records anything that was inferred
        rather than stated (e.g. coefficients above ``nmax`` dropped from a
        triangle file).

    Raises
    ------
    FileNotFoundError, ValueError
        Missing file / unknown extension / unparsable layout.
    """
    p = _require_file(path)
    ext = _suffix(p)
    if ext == "":
        # no extension: sniff the content (see _sniff_coeff_ext)
        ext = _sniff_coeff_ext(p)
        if layout == "auto":
            layout = "gfc" if ext == ".gfc" else "triangle"
    if ext not in COEFF_EXTENSIONS:
        raise ValueError(
            f"read_coeffs: 不支持的扩展名 {ext!r}（文件 {p}）。"
            f"支持: {', '.join(COEFF_EXTENSIONS)}")
    if layout not in ("auto", "triangle", "matrix", "gmfcsv", "npy", "gfc",
                      "series_nc", "legacy_dat"):
        raise ValueError(
            "layout 必须是 'auto'|'triangle'|'matrix'|'gmfcsv'|'npy'|'gfc'"
            "|'series_nc'|'legacy_dat'，"
            f"实际 {layout!r}")

    # multi-epoch containers first: they carry their own time axis
    if ext == ".nc" or layout == "series_nc":
        return _read_series_nc(p, nmax=nmax)
    if layout == "legacy_dat":
        return _read_legacy_dat(p, nmax=nmax)

    warnings: list = []
    if layout == "auto":
        layout, hint, payload = _detect_layout(p, ext, nmax)
        if hint is not None and nmax is None:
            nmax = hint
    else:
        payload = None
    if layout == "npy":
        layout = "matrix" if ext == ".npz" or _npy_is_matrix(p) else "triangle"
        payload = None

    meta: dict = {"source_file": str(p), "layout": layout}

    # ---------------------------------------------------------------- gfc
    if layout == "gfc" or (layout == "auto" and ext == ".gfc") or (
            layout == "auto" and _COEFF_LINE_RE.search(_head_text(p))):
        C, S, gmeta, w = _read_gfc(p, nmax)
        warnings.extend(w)
        meta.update(gmeta)
        # A .gfc file is an ICGEM/GFZ gravity model: its C_nm/S_nm are
        # dimensionless geopotential coefficients, NOT water height.  Labelling
        # them here means synthesis(target_unit='ewh') applies A_n automatically
        # instead of demanding that the user declare it.
        meta.setdefault("field_unit", "geopotential")
        out = SHCoeffs(C, S, meta)
        meta.update({"nmax": out.nmax, "ntime": out.ntime,
                     "warnings": warnings})
        out.meta = meta
        return out

    # ------------------------------------------------------------ triangle
    if layout == "triangle":
        if ext in (".npy",):
            arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
            header_lines = []
        elif payload is not None and isinstance(payload, dict) and "flat_cs" in payload:
            arr = payload["flat_cs"]
            header_lines = []
            meta.update(payload.get("meta", {}))
        elif ext == ".npz":
            z = np.load(p, allow_pickle=False)
            key = "flat_cs" if "flat_cs" in z.files else "triangle"
            arr = np.asarray(z[key], dtype=float)
            header_lines = []
            meta.update(_npz_meta(z))
        else:
            arr, L, header_lines = _read_triangle_text(p, nmax)
            if nmax is None:
                nmax = L
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] > 1 and arr.shape[0] != 2 * (nmax + 1) * (nmax + 2) // 2 \
                and arr.shape[1] == 2 * (nmax + 1) * (nmax + 2) // 2:
            arr = arr.T
            warnings.append("系数表是 (ntime, 2*NC) 布局，已转置成 (2*NC, ntime)")
        if nmax is None:
            nmax = _infer_nmax_from_rows(arr.shape[0])
        nrows = arr.shape[0]
        need = 2 * (nmax + 1) * (nmax + 2) // 2
        if nrows > need:
            nmax_full = _infer_nmax_from_rows(nrows)
            warnings.append(
                f"三角表有 {nrows} 行 = 完整到 {nmax_full} 阶；按 nmax={nmax} "
                f"只读取前 {need} 行（更高阶被丢弃）。传 nmax={nmax_full} 可全部读入")
            warnings.append(
                f"truncated_from_degree={nmax_full} on read with nmax={nmax}")
            arr = arr[:need, :]
        elif nrows < need:
            raise ValueError(
                f"{p}: 三角表有 {nrows} 行，但 nmax={nmax} 需要 {need} 行 "
                f"(2*(L+1)(L+2)/2)")
        coeffs = SHCoeffs.from_triangle(arr, int(nmax), meta)
        hdr = {}
        for key in ("nmax", "ntime", "field_unit", "forward_from",
                    "inverse_target", "output_unit",
                    "weight_rule", "method", "weight_sum",
                    "coverage", "fit_rmse_rel", "created",
                    "gaussian_radius_km", "gaussian_W0", "gaussian_Wnmax",
                    "n_points"):
            v = _header_field(header_lines, key)
            if v is not None:
                hdr[key] = v
        if hdr:
            meta["header_fields"] = hdr
        # promote the physical-meaning label to the top level: it must
        # survive a save/load round trip, otherwise units.units convert
        # silently degrades to "unknown" and conversions start failing.
        if hdr.get("field_unit"):
            meta["field_unit"] = str(hdr["field_unit"])
        if hdr.get("output_unit"):
            meta["output_unit"] = str(hdr["output_unit"])
        # 高斯平滑半径也要能读回：否则读回来的系数看不出已经被平滑过
        if hdr.get("gaussian_radius_km") is not None:
            try:
                meta["gaussian_radius_km"] = float(hdr["gaussian_radius_km"])
            except (TypeError, ValueError):
                pass
        # 覆盖比 / 采样点数：'shkit nmax' 要用它判断区域约束与定解裕度
        for key, cast in (("coverage", float), ("n_points", int)):
            if hdr.get(key) is not None:
                try:
                    meta[key] = cast(hdr[key])
                except (TypeError, ValueError):
                    pass
        if header_lines:
            meta["comment_header"] = header_lines
        meta.update({"nmax": coeffs.nmax, "ntime": coeffs.ntime})
        meta["warnings"] = warnings
        coeffs.meta = meta
        return coeffs

    # -------------------------------------------------------------- npz/npy
    if ext == ".npz" and layout == "matrix":
        z = np.load(p, allow_pickle=False)
        if "C" not in z.files or "S" not in z.files:
            raise ValueError(
                f"{p}: 矩阵布局的 .npz 需要 'C' 与 'S' 数组，实际 {list(z.files)}")
        C = np.asarray(z["C"], dtype=float)
        S = np.asarray(z["S"], dtype=float)
        meta.update(_npz_meta(z))
        if nmax is not None and C.shape[0] != nmax + 1:
            warnings.append(
                f"文件里是 nmax={C.shape[0] - 1} 的矩阵，与 nmax={nmax} 不一致；"
                "以文件为准（不做补零/截断），需要时请调用 coeffs.truncate()")
        coeffs = SHCoeffs(C, S, meta)
        meta.update({"nmax": coeffs.nmax, "ntime": coeffs.ntime,
                     "warnings": warnings})
        coeffs.meta = meta
        return coeffs

    if ext == ".npy" and layout == "matrix":
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
        C, S, w = _split_matrix_array(arr, nmax)
        warnings.extend(w)
        coeffs = SHCoeffs(C, S, meta)
        meta.update({"nmax": coeffs.nmax, "ntime": coeffs.ntime,
                     "warnings": warnings})
        coeffs.meta = meta
        return coeffs

    # --------------------------------------------------------------- gmfcsv
    if layout in ("gmfcsv", "matrix"):
        df, tmeta, w = _read_table(p)
        warnings.extend(w)
        meta.update(tmeta)
        cols = list(df.columns)
        n_idx = _guess_named_column(cols, ("n", "degree", "l"))
        m_idx = _guess_named_column(cols, ("m", "order"))
        c_idx = _guess_named_column(cols, ("c", "clm", "cos", "c_nm"))
        s_idx = _guess_named_column(cols, ("s", "slm", "sin", "s_nm"))
        if None in (n_idx, m_idx, c_idx, s_idx):
            if len(cols) >= 4:
                n_idx, m_idx, c_idx, s_idx = 0, 1, 2, 3
                warnings.append("未识别到 n,m,C,S 表头，按前 4 列解释")
            else:
                raise ValueError(
                    f"{p}: 矩阵布局需要 n,m,C,S 四列，文件只有 {len(cols)} 列: {cols}")
        nn = _to_float_array(df.iloc[:, n_idx], cols[n_idx], 1)
        mm = _to_float_array(df.iloc[:, m_idx], cols[m_idx], 1)
        if not np.all(np.isfinite(nn)) or not np.all(np.isfinite(mm)):
            raise ValueError(f"{p}: n/m 列含缺失值")
        # optional extra time steps written as C_t1,S_t1,C_t2,S_t2,...
        extra = []
        tcols = {str(c).strip().lower(): i for i, c in enumerate(cols)}
        t = 1
        while f"c_t{t}" in tcols and f"s_t{t}" in tcols:
            extra.append((tcols[f"c_t{t}"], tcols[f"s_t{t}"]))
            t += 1
        ntime = 1 + len(extra)
        cvals = [_to_float_array(df.iloc[:, c_idx], cols[c_idx], 1)]
        svals = [_to_float_array(df.iloc[:, s_idx], cols[s_idx], 1)]
        for ic, is_ in extra:
            cvals.append(_to_float_array(df.iloc[:, ic], cols[ic], 1))
            svals.append(_to_float_array(df.iloc[:, is_], cols[is_], 1))
        L_file = int(np.nanmax(nn))
        if nmax is None:
            nmax = L_file
        elif L_file > nmax:
            warnings.append(
                f"文件含到 {L_file} 阶的系数，但 nmax={nmax}；更高阶被丢弃。"
                f"传 nmax={L_file} 可全部读入")
            warnings.append(f"truncated_from_degree={L_file} on read with nmax={nmax}")
        if ntime == 1:
            C = np.zeros((nmax + 1, nmax + 1))
            S = np.zeros((nmax + 1, nmax + 1))
        else:
            C = np.zeros((nmax + 1, nmax + 1, ntime))
            S = np.zeros((nmax + 1, nmax + 1, ntime))
            warnings.append(f"矩阵 csv 含 {ntime} 个时次（C,S 及其后的 C_t1/S_t1...）")
        keep = (nn <= nmax) & (mm <= nmax)
        for row in np.nonzero(keep)[0]:
            i, j = int(round(nn[row])), int(round(mm[row]))
            if j > i:
                warnings.append(f"忽略不合法的 (n={i}, m={j})（m > n）")
                continue
            for k in range(ntime):
                if ntime == 1:
                    C[i, j] = cvals[0][row]
                    S[i, j] = svals[0][row]
                else:
                    C[i, j, k] = cvals[k][row]
                    S[i, j, k] = svals[k][row]
        coeffs = SHCoeffs(C, S, meta)
        meta.update({"nmax": coeffs.nmax, "ntime": coeffs.ntime,
                     "max_degree_in_file": L_file, "warnings": warnings})
        coeffs.meta = meta
        return coeffs

    raise ValueError(f"{p}: 无法用 layout={layout!r} 读取")


def _npy_is_matrix(p: Path) -> bool:
    """True when a ``.npy``/``.npz`` holds the dense matrix layout."""
    try:
        if _suffix(p) == ".npz":
            z = np.load(p, allow_pickle=False)
            return "C" in z.files and "S" in z.files
        arr = np.load(p, allow_pickle=False)
        return arr.ndim == 3
    except Exception:
        return False


def _split_matrix_array(arr: np.ndarray, nmax: Optional[int]) -> tuple:
    """``(2, L+1, L+1[, ntime])`` or ``(L+1, L+1, 2...)`` -> ``(C, S, warnings)``."""
    warnings: list = []
    if arr.ndim == 3:
        if arr.shape[0] == 2:
            C, S = arr[0], arr[1]
        else:
            raise ValueError(
                f"矩阵 .npy 的 shape {arr.shape} 无法解释；"
                "期望 (2, L+1, L+1) 或 (L+1, L+1, 2)")
    elif arr.ndim == 4 and arr.shape[0] == 2:
        C, S = arr[0], arr[1]
        warnings.append("(2, L+1, L+1, ntime) 布局：C/S 各带一个时次轴")
    elif arr.ndim == 4 and arr.shape[1] == 2:
        C, S = arr[:, 0], arr[:, 1]
        warnings.append("(L+1, L+1, 2, ntime) 布局")
    else:
        raise ValueError(f"矩阵 .npy 的 shape {arr.shape} 无法解释")
    if C.shape != S.shape or C.shape[0] != C.shape[1]:
        raise ValueError(f"C{C.shape} 与 S{S.shape} 必须是同形的方阵")
    if nmax is not None and C.shape[0] != nmax + 1:
        warnings.append(
            f"文件里是 nmax={C.shape[0] - 1}，与 nmax={nmax} 不一致；以文件为准")
    if C.ndim == 3 and C.shape[2] == 1:
        C, S = C[:, :, 0], S[:, :, 0]
        warnings.append("单时次矩阵已压成 2 维")
    return C, S, warnings


# ---------------------------------------------------------------- gfc -------
_GFC_KEYS = ("modelname", "earth_gravity_constant", "radius", "max_degree",
             "errors", "norm", "tide_system", "format", "ref_frame",
             "institution", "product_type", "value_type", "gapvalue",
             "descriptor", "unit")


def _read_gfc(p: Path, nmax: Optional[int]) -> tuple:
    """Parse an ICGEM/GFZ ``.gfc`` file into dense ``C``/``S`` matrices.

    Lines are ``gfc n m C S [sigmaC sigmaS]`` (``gfct`` is accepted as well);
    ``#`` comments, ``begin/end`` markers and the ``key value`` header lines are
    handled.  ``nmax`` defaults to the largest degree in the file; if it is
    smaller, the higher degrees are dropped and a warning is recorded.
    """
    text = _text_read(p)
    warnings: list = []
    meta: dict = {}
    triples = []
    max_deg_in_file = 0
    n_sigma = 0
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        toks = s.split()
        low = toks[0].lower()
        if low in ("begin", "end", "begin_model", "end_model", "begin_of_head",
                   "end_of_head", "begin_of_data", "end_of_data"):
            continue
        if low in ("gfc", "gfct", "grcof2", "grcf"):
            if len(toks) < 5:
                warnings.append(f"忽略字段不足的 gfc 行: {s[:60]!r}")
                continue
            try:
                n_ = int(toks[1])
                m_ = int(toks[2])
                c_ = float(toks[3])
                s_ = float(toks[4])
            except ValueError:
                warnings.append(f"忽略无法解析的 gfc 行: {s[:60]!r}")
                continue
            sig = None
            if len(toks) >= 7:
                try:
                    sig = (float(toks[5]), float(toks[6]))
                    n_sigma += 1
                except ValueError:
                    sig = None
            triples.append((n_, m_, c_, s_, sig))
            max_deg_in_file = max(max_deg_in_file, n_)
            continue
        # header / constant lines: key value [value ...]
        if len(toks) >= 2:
            key = toks[0].rstrip(":").lower()
            if key in _GFC_KEYS or re.fullmatch(r"[a-z_]{3,}", key):
                meta.setdefault(key, " ".join(toks[1:]))
        # anything else is ignored
    if not triples:
        raise ValueError(
            f"{p}: 没有解析到任何系数数据行；确认这是 ICGEM ``gfc`` 或 PO.DAAC "
            "``SHM`` 格式（数据行首列为 gfc / gfct / GRCOF2）")

    if nmax is None:
        if meta.get("max_degree"):
            try:
                nmax = int(float(str(meta["max_degree"]).split()[0]))
            except ValueError:
                nmax = max_deg_in_file
        else:
            nmax = max_deg_in_file
    if max_deg_in_file > nmax:
        warnings.append(
            f"文件含到 {max_deg_in_file} 阶的系数，但 nmax={nmax}；更高阶被丢弃。"
            f"传 nmax={max_deg_in_file} 可全部读入")
        warnings.append(f"truncated_from_degree={max_deg_in_file} on read with nmax={nmax}")

    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    sigC = np.zeros((nmax + 1, nmax + 1)) if n_sigma else None
    sigS = np.zeros((nmax + 1, nmax + 1)) if n_sigma else None
    n_used = 0
    for n_, m_, c_, s_, sig in triples:
        if n_ > nmax or m_ > nmax:
            continue
        if m_ > n_:
            warnings.append(f"忽略不合法的 (n={n_}, m={m_})（m > n）")
            continue
        C[n_, m_] = c_
        S[n_, m_] = s_
        if sig is not None and sigC is not None:
            sigC[n_, m_], sigS[n_, m_] = sig
        n_used += 1
    meta_out = {
        "format": "gfc",
        "gfc_header": dict(meta),
        "gfc_lines_used": int(n_used),
        "max_degree_in_file": int(max_deg_in_file),
        "has_sigmas": bool(n_sigma),
    }
    for k in ("modelname", "earth_gravity_constant", "radius", "norm",
              "tide_system", "product_type", "value_type", "institution"):
        if k in meta:
            meta_out[k] = meta[k]
    if sigC is not None:
        meta_out["sigmaC"] = sigC
        meta_out["sigmaS"] = sigS
    if str(meta.get("norm", "4pi")).lower() not in ("4pi", "fully_normalized",
                                                    "fully_normalised", "unnormalized",
                                                    "unnormalised"):
        warnings.append(
            f"文件声明的归一化是 {meta.get('norm')!r}；本模块只按 4-pi 归一化"
            "（SHTOOLS norm=1, csphase=1）解释，直接使用前请核对")
    return C, S, meta_out, warnings


def _fmt_float(v: float, fmt: str) -> str:
    if v is None or not np.isfinite(v):
        return "nan"
    return fmt % v


def write_coeffs(coeffs: SHCoeffs, path, layout: str = "triangle",
                 comment: Optional[str] = None, fmt: str = "%.16g",
                 header: bool = True, with_sigma: bool = False,
                 meta_extra: Optional[Mapping[str, Any]] = None,
                 time: Optional[int] = None) -> Path:
    """Write :class:`SHCoeffs` to disk in one of five layouts.

    Parameters
    ----------
    coeffs : SHCoeffs
        Object to write.  ``ntime = 1`` produces a single column.
    path : str or Path
        ``.sh .txt .csv .dat .tsv`` (text), ``.gfc`` (ICGEM), ``.npy``,
        ``.npz``.  A ``.gz`` suffix switches on gzip compression for the text
        and gfc forms.
    layout : {'triangle', 'gmfcsv', 'gfc', 'npy', 'npz'}
        ``'triangle'`` is the reference-software (``m2py`` /
        ``gridSHconvert``) ``[C; S]`` stack of ``2*NC`` rows: column-stacked
        order ``m`` outer / ``n`` inner, one column per time step.  It is the
        layout to hand back to the legacy tools, and it is what
        :func:`read_coeffs` detects by default.
        ``'gmfcsv'`` writes ``n,m,C,S`` rows, ``'gfc'`` the ICGEM text format,
        ``'npy'`` a single dense array, ``'npz'`` ``C``/``S``/``meta``.
    comment : str, optional
        Free text placed in the ``#`` header (text layouts only).
    fmt : str
        Format string for every number; the default ``%.16g`` round-trips
        doubles exactly through decimal text.
    header : bool
        Write the ``#`` header block (text layouts only).
    with_sigma : bool
        ``'gfc'`` only: also emit the ``sigmaC sigmaS`` columns when
        ``coeffs.meta['sigmaC']``/``['sigmaS']`` are present (otherwise zeros).
    meta_extra : mapping, optional
        Extra key/value pairs merged into the written header / ``meta``.
    time : int, optional
        ``'gfc'`` only.  A ``.gfc`` file holds exactly one epoch, so writing a
        multi-epoch object without choosing one raises ``ValueError`` instead of
        silently concatenating (which would read back as a single corrupted
        model).

    Returns
    -------
    pathlib.Path
        The path actually written.

    Notes
    -----
    For ``.npy`` the array is ``(2, L+1, L+1)`` (single slice) or
    ``(2, L+1, L+1, ntime)``; :func:`read_coeffs` reverses exactly this.
    """
    if not isinstance(coeffs, SHCoeffs):
        raise TypeError(f"coeffs 必须是 SHCoeffs，实际 {type(coeffs).__name__}")
    p = _p(path)
    ext = _suffix(p)
    if layout not in ("triangle", "gmfcsv", "gfc", "npy", "npz", "auto",
                      "series_nc", "legacy_dat"):
        raise ValueError(
            "layout 必须是 'triangle'|'gmfcsv'|'gfc'|'npy'|'npz'|'series_nc'"
            f"|'legacy_dat'，实际 {layout!r}")
    if layout == "auto":
        # derive from the extension for the numpy containers, triangle otherwise
        layout = {"npy": "npy", "npz": "npz", "gfc": "gfc",
                  "nc": "series_nc"}.get(ext, "triangle")
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)

    if layout == "series_nc" or (layout == "auto" and ext == ".nc"):
        return _write_series_nc(coeffs, p, comment=comment, meta_extra=meta_extra)
    if layout == "legacy_dat":
        return _write_legacy_dat(coeffs, p, comment=comment, fmt=fmt,
                                 meta_extra=meta_extra)

    L, ntime = coeffs.nmax, coeffs.ntime
    m_vec, n_vec = triangle_order(L)
    C3 = coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C
    S3 = coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S

    meta = dict(coeffs.meta)
    meta.update(dict(meta_extra or {}))

    # ------------------------------------------------------------- npz / npy
    if layout in ("npz", "npy") or ext in (".npz", ".npy"):
        if ext == ".npy" or layout == "npy":
            if coeffs.C.ndim == 2:
                arr = np.stack([coeffs.C, coeffs.S])
            else:
                arr = np.stack([coeffs.C, coeffs.S])
            if str(p).lower().endswith(".gz"):
                raise ValueError("write_coeffs: .npy 不支持 .gz")
            np.save(p, arr)
            return p
        np.savez(p,
                 C=np.asarray(coeffs.C, dtype=float),
                 S=np.asarray(coeffs.S, dtype=float),
                 meta=json.dumps(_json_meta(meta), ensure_ascii=False),
                 nmax=L, ntime=ntime, layout="matrix",
                 shkit_version=SHKIT_VERSION)
        return p

    if ext == ".gfc" or layout == "gfc":
        sigC = meta.get("sigmaC")
        sigS = meta.get("sigmaS")
        out = []
        out.append("# SHKit %s  ICGEM/GFZ formatted spherical harmonic model"
                   % SHKIT_VERSION)
        out.append("# Written by shkit.io.write_coeffs(layout='gfc')")
        if comment:
            out.extend("# " + ln for ln in comment.splitlines())
        model_name = str(meta.get("modelname") or p.stem)
        out.append(f"modelname            {model_name}")
        out.append(f"earth_gravity_constant {meta.get('earth_gravity_constant', '0.0')}")
        out.append(f"radius                {meta.get('radius', '6378137.0')}")
        out.append(f"max_degree            {L}")
        out.append(f"errors                {'formal' if sigC is not None else 'no'}")
        out.append("norm                  4pi")
        out.append(f"tide_system           {meta.get('tide_system', 'unknown')}")
        out.append("format                icgem")
        out.append("shkit_layout          triangle-source")
        for k, v in _json_meta(meta).items():
            if k in _GFC_KEYS or k in ("sigmaC", "sigmaS"):
                continue
            try:
                out.append(f"{k:<22}{v if isinstance(v, (int, float)) else str(v)}")
            except Exception:
                pass
        out.append("end_of_head")
        if ntime > 1 and time is None:
            raise ValueError(
                f"write_coeffs(layout='gfc'): an ICGEM/GFZ .gfc file holds "
                f"exactly one epoch, but this object has ntime={ntime}. "
                "Writing every epoch into one file would produce a file that "
                "reads back as a single corrupted model.  Options: "
                "(a) pass time=<k> to export one epoch, "
                "(b) use layout='triangle' or 'npz', which keep every epoch, "
                "(c) write one .gfc per epoch in a loop.")
        t_sel = 0 if time is None else int(time)
        if not 0 <= t_sel < ntime:
            raise ValueError(f"time={time} out of range for ntime={ntime}")
        for t in (t_sel,):
            for k in range(len(m_vec)):
                n_, m_ = int(n_vec[k]), int(m_vec[k])
                c_ = float(C3[n_, m_, t])
                s_ = float(S3[n_, m_, t])
                if with_sigma and sigC is not None:
                    sc = float(np.asarray(sigC)[n_, m_]) if np.ndim(sigC) >= 2 else 0.0
                    ss = float(np.asarray(sigS)[n_, m_]) if np.ndim(sigS) >= 2 else 0.0
                    out.append("gfc %3d %3d %s %s %s %s" % (
                        n_, m_, _fmt_float(c_, fmt), _fmt_float(s_, fmt),
                        _fmt_float(sc, fmt), _fmt_float(ss, fmt)))
                else:
                    out.append("gfc %3d %3d %s %s" % (
                        n_, m_, _fmt_float(c_, fmt), _fmt_float(s_, fmt)))
        out.append("end_of_data")
        _text_write(p, "\n".join(out) + "\n")
        return p

    if ext not in (".sh", ".txt", ".csv", ".dat", ".tsv") and not \
            str(p).lower().endswith(".gz"):
        raise ValueError(
            f"write_coeffs: 不支持的扩展名 {ext!r}（文件 {p}）。"
            f"支持: {', '.join(COEFF_EXTENSIONS)}, .gz")

    sep = "," if ext == ".csv" else ("\t" if ext == ".tsv" else " ")

    if layout == "triangle":
        tri = coeffs.to_triangle()
        lines = []
        if header:
            lines.append(f"# SHKit {SHKIT_VERSION} spherical harmonic "
                         "coefficients (triangle layout: [C; S], m outer, n inner)")
            lines.append(f"# nmax = {L}")
            lines.append(f"# ntime = {ntime}")
            lines.append(f"# ncoef_triangle = {tri.shape[0] // 2}")
            lines.append(f"# rows = 2*ncoef_triangle = {tri.shape[0]}  (C rows first)")
            lines.append("# norm = 4pi (SHTOOLS norm=1, csphase=1), S[:,0] == 0")
            lines.append("# units = user field units, lat/lon in degrees")
            for k in ("field_unit", "forward_from", "inverse_target",
                      "output_unit", "weight_rule", "gaussian_radius_km",
                      "gaussian_W0", "gaussian_Wnmax",
                      "method", "weight_sum", "coverage", "n_points",
                      "fit_rmse_rel", "lmax_recommended", "resolution_km"):
                if k in meta and meta[k] is not None:
                    lines.append(f"# {k} = {meta[k]}")
            if "report" in meta and isinstance(meta["report"], Mapping):
                lines.append("# report = " + json.dumps(
                    _json_meta(meta["report"]), ensure_ascii=False))
            if comment:
                lines.extend("# " + ln for ln in comment.splitlines())
            lines.append("# columns: " + sep.join(
                [f"t{t}" for t in range(ntime)]))
        with np.errstate(all="ignore"):
            for r in range(tri.shape[0]):
                lines.append(sep.join(_fmt_float(v, fmt) for v in tri[r]))
        _text_write(p, "\n".join(lines) + "\n")
        return p

    # gmfcsv
    lines = []
    if header:
        lines.append(f"# SHKit {SHKIT_VERSION} coefficients, layout n,m,C,S")
        lines.append(f"# nmax = {L}   ntime = {ntime}   "
                     "(C,S are the first time step; extra columns hold later steps)")
        if comment:
            lines.extend("# " + ln for ln in comment.splitlines())
        names = ["n", "m", "C", "S"]
        for t in range(1, ntime):
            names.extend([f"C_t{t}", f"S_t{t}"])
        lines.append(sep.join(names))
    for k in range(len(m_vec)):
        n_, m_ = int(n_vec[k]), int(m_vec[k])
        row = [str(n_), str(m_), _fmt_float(C3[n_, m_, 0], fmt),
               _fmt_float(S3[n_, m_, 0], fmt)]
        for t in range(1, ntime):
            row.extend([_fmt_float(C3[n_, m_, t], fmt),
                        _fmt_float(S3[n_, m_, t], fmt)])
        lines.append(sep.join(row))
    _text_write(p, "\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# reports and bundles
# ---------------------------------------------------------------------------
def _report_dict(report) -> dict:
    """Accept a dict, an AnalysisReport, a WeightSet or anything with report()."""
    if report is None:
        return {}
    if isinstance(report, Mapping):
        return _json_meta(report)
    if hasattr(report, "report") and callable(report.report):
        return _json_meta(report.report())
    raise TypeError(
        "report 必须是 dict / AnalysisReport / 任何带 report() 的对象，"
        f"实际 {type(report).__name__}")


def write_report(path, report, extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Write an analysis report as JSON or Markdown.

    Parameters
    ----------
    path : str or Path
        ``.json`` (``ensure_ascii=False``, ``indent=2``, so a Chinese description
        stays readable) or ``.md`` (a human-readable key/value table plus the
        warning list).  ``.gz`` is allowed for either.
    report : dict or AnalysisReport or WeightSet
        Anything with ``report()`` is accepted; a plain dict is passed through.
    extra : mapping, optional
        Extra top-level entries merged into the JSON document (ignored for
        ``.md``, where they appear as a trailing "附加信息" table).
    """
    p = _p(path)
    ext = _suffix(p)
    if ext == ".gz":
        ext = Path(p.stem).suffix.lower()
    if ext not in REPORT_EXTENSIONS:
        raise ValueError(
            f"write_report: 不支持的扩展名 {_suffix(p)!r}（文件 {p}）。"
            f"支持: {', '.join(REPORT_EXTENSIONS)}")
    doc = _report_dict(report)
    ex = dict(extra or {})
    if ex:
        doc = dict(doc)
        doc["extra"] = _json_meta(ex)

    if ext == ".json":
        _text_write(p, json.dumps(doc, ensure_ascii=False, indent=2,
                                  allow_nan=True) + "\n", bom=False)
        return p

    lines = ["# shkit analysis report", ""]
    if ex:
        lines.append("## 附加信息")
        lines.append("")
        lines.append("| 键 | 值 |")
        lines.append("| --- | --- |")
        for k, v in ex.items():
            lines.append(f"| `{k}` | {_md_cell(v)} |")
        lines.append("")
    lines.append("## 诊断量")
    lines.append("")
    lines.append("| 键 | 值 |")
    lines.append("| --- | --- |")
    for k, v in doc.items():
        if k == "warnings":
            continue
        lines.append(f"| `{k}` | {_md_cell(v)} |")
    warn = doc.get("warnings") or []
    lines.append("")
    lines.append("## 警告" if warn else "## 警告（无）")
    lines.append("")
    for w in warn:
        lines.append(f"- {w}")
    if not warn:
        lines.append("- 无")
    lines.append("")
    _text_write(p, "\n".join(lines), bom=False)
    return p


def _md_cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
        return f"{v:.6g}"
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).replace("|", "\\|").replace("\n", " ")


def save_result(dirpath, name: str, coeffs: SHCoeffs, report,
                grid: Optional[tuple] = None, resampled: Optional[np.ndarray] = None,
                points: Optional[tuple] = None, extra: Optional[Mapping] = None,
                grid_var: str = "value", gzip_coeffs: bool = False) -> dict:
    """Write a complete analysis result bundle into ``dirpath``.

    Files written
    -------------
    ``<name>.sh``        triangle (m2py-compatible) coefficients; with
                         ``gzip_coeffs=True`` the name becomes ``<name>.sh.gz``
    ``<name>.json``      the analysis report (JSON, Chinese-safe)
    ``<name>.md``        the same report as a Markdown table
    ``<name>_grid.nc``   the resampled field, when ``grid`` is given as
                         ``(lat_vec, lon_vec, values)``
    ``<name>_points.csv`` the reconstructed field at the input samples, when
                         ``points`` is given as ``(lat, lon, values)``
    ``<name>.npz``       ``C``/``S``/``meta`` (always written; cheap, and it is
                         the only lossless container)

    Parameters
    ----------
    dirpath : str or Path
        Output directory (created when missing).
    name : str
        Base name of the bundle; path separators are rejected.
    coeffs : SHCoeffs
    report : dict or AnalysisReport
    grid : tuple(lat_vec, lon_vec, values), optional
    resampled : array_like, optional
        Alias of the third element of ``grid`` when only values are known -
        kept for callers that resample separately.  Rarely needed.
    points : tuple(lat, lon, values), optional
    extra : mapping, optional
        Merged into the coefficient header and the JSON report.
    grid_var : str
        netCDF variable name of the resampled grid.

    Returns
    -------
    dict
        ``{'dir', 'coeffs', 'coeffs_npz', 'report_json', 'report_md',
        'grid', 'points', 'coefficients': SHCoeffs}`` - values are ``str``
        paths (or ``None``).
    """
    if not isinstance(coeffs, SHCoeffs):
        raise TypeError("save_result: coeffs 必须是 SHCoeffs")
    base = str(name).strip()
    if not base:
        raise ValueError("name 不能为空")
    if any(sep in base for sep in ("/", "\\", ":", "*", "?", '"', "<", ">", "|")):
        raise ValueError(f"name 里不能有路径分隔符或非法字符: {name!r}")
    d = _p(dirpath)
    d.mkdir(parents=True, exist_ok=True)

    ex = dict(extra or {})
    meta = dict(coeffs.meta)
    meta.update(_json_meta(ex))
    doc = _report_dict(report)
    if doc:
        meta["report"] = doc
    c2 = SHCoeffs(coeffs.C.copy(), coeffs.S.copy(), meta)

    sh_name = f"{base}.sh.gz" if gzip_coeffs else f"{base}.sh"
    p_sh = write_coeffs(c2, d / sh_name, layout="triangle", meta_extra=ex)
    p_json = write_report(d / f"{base}.json", report, extra=ex)
    p_md = write_report(d / f"{base}.md", report, extra=ex)
    p_npz = write_coeffs(c2, d / f"{base}.npz", layout="npz", meta_extra=ex)

    p_grid = None
    if grid is not None:
        if resampled is not None:
            la, lo = grid
            vals = resampled
        else:
            if len(grid) != 3:
                raise ValueError("grid 必须是 (lat_vec, lon_vec, values) 三元组")
            la, lo, vals = grid
        p_grid = write_grid(d / f"{base}_grid.nc", la, lo, vals, var=grid_var,
                            meta=ex or meta, units=meta.get("units"))

    p_pts = None
    if points is not None:
        if len(points) != 3:
            raise ValueError("points 必须是 (lat, lon, values) 三元组")
        la, lo, vals = points
        p_pts = write_points(d / f"{base}_points.csv", la, lo, vals)

    return {
        "dir": str(d),
        "coeffs": str(p_sh),
        "coeffs_npz": str(p_npz),
        "report_json": str(p_json),
        "report_md": str(p_md),
        "grid": None if p_grid is None else str(p_grid),
        "points": None if p_pts is None else str(p_pts),
        "coefficients": c2,
    }


# ---------------------------------------------------------------------------
# misc numerics that belong to the I/O-side workflow
# ---------------------------------------------------------------------------
def gaussian_degree_weights(nmax: int, radius_km: float,
                            r_earth_km: float = 6371.0,
                            n_quad: int = 4096) -> np.ndarray:
    """Jekeli/Wahr Gaussian smoothing coefficients ``W_n`` by degree.

    ``W_n`` is the degree amplitude of an isotropic Gaussian average over a
    spherical cap, in the classical integral form (Jekeli 1981; Wahr, Molenaar
    & Bryan 1998, eq. 6):

    .. math::

        W_n = \\frac{1}{2\\pi}\\int_0^{\\pi}
              \\frac{b\\exp\\!\\bigl(-b(1-\\cos\\psi)\\bigr)}
                   {1-\\exp(-2b)}\\;\\bar P_n(\\cos\\psi)\\sin\\psi\\,\\mathrm d\\psi
            = \\int_0^{1} k(x)\\,\\bar P_n(x)\\,\\mathrm dx, \\qquad
        b = \\frac{2\\ln 2}{r^2}, \\quad r = \\frac{\\text{radius\\_km}}{R}

    ``radius_km`` is the **half-width** radius: the kernel falls to 0.5 of its
    central value at that spherical distance (``k(1)-k(-1) = ln 2``).  The
    integral is evaluated with Gauss-Legendre quadrature, which is stable for
    every ``nmax``; the textbook difference recursion
    ``W_{n+1} = ((2n+1)/(n+1)) b W_n + (n/(n+1)) W_{n-1}`` cancels
    catastrophically here for a small kernel (measured: it returns 1.0 at
    ``n = 2`` for ``radius = 300 km``, where the true value is 0.19).

    The result is meant for :func:`shkit.basis.synthesize`'s ``weights_scale=``
    argument or for damping coefficients in place; it changes neither the
    normalisation nor the ``4*pi`` convention.

    Returns
    -------
    ndarray, shape ``(nmax+1,)``
        ``W_n``, ``W_0 = 1`` and non-increasing.

    Notes
    -----
    ``radius_km <= 0`` (or ``None``) returns all ones, i.e. no smoothing.
    A small kernel makes ``W_n`` underflow to exactly 0 beyond a finite degree
    (that is the correct "the filter removes everything above this degree"
    behaviour, not a loss of precision).
    """
    nmax = int(nmax)
    W = np.ones(nmax + 1)
    if radius_km is None or radius_km <= 0:
        return W
    R = float(r_earth_km)
    r = float(radius_km) / R
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"radius_km={radius_km!r} 必须是正数（单位 km）")
    b = 2.0 * np.log(2.0) / r ** 2

    # kernel on [-1, 1]:  k(x) = b*exp(-b(1-x)) / (2*pi*(1-exp(-2b)))
    norm = 2.0 * np.pi * (1.0 - np.exp(-2.0 * b))
    x, gw = np.polynomial.legendre.leggauss(int(n_quad))
    k = b * np.exp(-b * (1.0 - x)) / norm
    k = np.where(np.isfinite(k), k, 0.0)

    # Legendre recurrence, scaling down to avoid overflow at large n
    p_prev = np.ones_like(x)
    W[0] = float(np.sum(gw * k * p_prev))
    p_cur = x.copy()
    if nmax >= 1:
        W[1] = float(np.sum(gw * k * p_cur))
    for n in range(1, nmax):
        p_next = ((2 * n + 1) * x * p_cur - n * p_prev) / (n + 1)
        mx = float(np.max(np.abs(p_next)))
        if mx > 1e150:
            p_next = p_next / mx
            p_cur = p_cur / mx
            p_prev = p_prev / mx
        W[n + 1] = float(np.sum(gw * k * p_next))
        p_prev, p_cur = p_cur, p_next

    W = np.clip(W, 0.0, None)
    W[0] = 1.0
    np.minimum.accumulate(W, out=W)      # enforce monotonicity against round-off
    return W


# ===========================================================================
# Multi-epoch containers and series construction (A1)
# ===========================================================================
#: Attributes frozen by ``docs/水平形变契约.md`` §2.4 / SHSynth §12.1.
SERIES_NC_ATTRS = ("norm", "csphase", "field_unit", "nmax", "gaussian_km",
                   "tide_system", "center", "mission", "product", "rl",
                   "source_files", "time_source", "coverage_start",
                   "coverage_end", "created", "producer")


def _series_attrs(coeffs: SHCoeffs, meta: Mapping[str, Any]) -> dict:
    """NetCDF global attributes for a coefficient series (the frozen key set)."""
    out: dict = {}
    out["norm"] = str(meta.get("norm") or "4pi")
    try:
        out["csphase"] = int(meta.get("csphase", 1))
    except (TypeError, ValueError):
        out["csphase"] = 1
    out["field_unit"] = str(meta.get("field_unit") or "geopotential")
    out["nmax"] = int(coeffs.nmax)
    out["producer"] = f"SHKit {SHKIT_VERSION}"
    out["created"] = meta.get("created") or _dt.datetime.now().astimezone().isoformat(
        timespec="seconds")
    for k in SERIES_NC_ATTRS:
        if k in ("norm", "csphase", "field_unit", "nmax", "producer", "created"):
            continue
        if k in meta and meta[k] is not None:
            v = meta[k]
            out[k] = (json.dumps(v, ensure_ascii=False)
                      if isinstance(v, (list, tuple, dict)) else str(v))
    if coeffs.times is not None:
        out.setdefault("time_source", str(coeffs.times.meta.get("time_source", "?")))
        src = coeffs.times.source or coeffs.times.labels
        if src:
            out["source_files"] = json.dumps([str(s) for s in src], ensure_ascii=False)
        if coeffs.times.has_dates:
            out["coverage_start"] = str(coeffs.times.values[0])[:10]
            out["coverage_end"] = str(coeffs.times.values[-1])[:10]
    return out


def _write_series_nc(coeffs: SHCoeffs, p: Path, *, comment=None,
                     meta_extra=None, dtype="float64") -> Path:
    """Write the frozen ``series_nc`` container: ``c(time,n,m)`` / ``s(time,n,m)``.

    This is the **authoritative** multi-epoch format: unlike the legacy triangle
    table (6-decimal decimal years, ~16 s) it carries a real ``datetime64`` axis,
    which is what both SHKit and SHSynth read back.
    """
    import xarray as xr

    if str(p).lower().endswith(".gz"):
        raise ValueError("write_coeffs: series_nc 不支持 .gz")
    L, ntime = coeffs.nmax, coeffs.ntime
    C3 = coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C
    S3 = coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S
    nn = np.arange(L + 1)
    meta = dict(coeffs.meta)
    meta.update(dict(meta_extra or {}))

    if coeffs.times is not None and coeffs.times.has_dates:
        tvals = coeffs.times.values
    else:
        # honest fallback: an integer epoch index with explicit units, so a
        # reader cannot mistake it for a calendar date
        tvals = np.arange(ntime)
    ds = xr.Dataset(
        data_vars={
            # on-disk order is (time, n, m); the in-memory arrays are (n, m, time)
            "c": (("time", "n", "m"),
                  np.asarray(C3, dtype=dtype).transpose(2, 0, 1)),
            "s": (("time", "n", "m"),
                  np.asarray(S3, dtype=dtype).transpose(2, 0, 1)),
        },
        coords={"time": tvals, "n": nn, "m": nn},
        attrs=_series_attrs(coeffs, meta))
    if coeffs.times is not None and not coeffs.times.has_dates:
        ds["time"].attrs["units"] = "epoch index (no calendar date available)"
    if comment:
        ds.attrs["history"] = str(comment)
    _write_nc(ds, p)
    return p


def _read_series_nc(p: Path, *, nmax: Optional[int] = None) -> SHCoeffs:
    """Read a ``series_nc`` container, tolerantly (per the frozen contract).

    Tolerated: variable names ``c``/``s`` **or** ``C``/``S``; any dimension order;
    a missing ``n``/``m`` coordinate (dense square is then assumed); a missing or
    integer ``time`` coordinate (degrades to an index axis + a warning).
    """
    with open_nc_dataset(p) as ds:
        cname = "c" if "c" in ds else ("C" if "C" in ds else None)
        sname = "s" if "s" in ds else ("S" if "S" in ds else None)
        if cname is None or sname is None:
            raise ValueError(
                f"{p}: 不是 series_nc 容器（需要 c/s 或 C/S 变量；"
                f"文件里的变量: {list(ds.data_vars)}）")
        Cda, Sda = ds[cname], ds[sname]
        L = int(Cda.sizes["n"]) - 1
        if "time" in Cda.dims:
            order = ["time", "n", "m"]
            if Cda.ndim != 3 or set(Cda.dims) != set(order):
                raise ValueError(f"{p}: c 的维度 {Cda.dims} 不是 (time,n,m)")
            # on-disk (time,n,m) -> in-memory (n,m,time)
            C3 = np.asarray(Cda.transpose(*order).values, dtype=float).transpose(1, 2, 0)
            S3 = np.asarray(Sda.transpose(*order).values, dtype=float).transpose(1, 2, 0)
        else:                                        # single epoch
            C3 = np.asarray(Cda.transpose("n", "m").values, dtype=float)[:, :, None]
            S3 = np.asarray(Sda.transpose("n", "m").values, dtype=float)[:, :, None]
        warnings: list = []
        attrs = {k: ds.attrs.get(k) for k in SERIES_NC_ATTRS if k in ds.attrs}
        times = None
        if "time" in ds.coords:
            times = TimeAxis.from_netcdf_coord(ds["time"])
            if times.kind == "index":
                warnings.append(
                    "time 坐标是数值且缺少可用 units，已降级为**序号**；"
                    "按日期切片/求趋势会报错（不要用序号当时间）")
        else:
            warnings.append("文件没有 time 坐标，已降级为序号时间轴")
            times = TimeAxis.from_index(C3.shape[2])
    if times is not None and times.kind == "index" and C3.shape[2] != len(times):
        times = TimeAxis.from_index(C3.shape[2])
    if nmax is not None and nmax < L:
        C3, S3, L = C3[:nmax + 1, :nmax + 1], S3[:nmax + 1, :nmax + 1], int(nmax)
    C = C3[:, :, 0] if C3.shape[2] == 1 else C3
    S = S3[:, :, 0] if S3.shape[2] == 1 else S3
    meta = {"format": "series_nc", "source_file": str(p), "nmax": L,
            "ntime": int(C3.shape[2] if C.ndim == 3 else 1)}
    for k, v in attrs.items():
        if v is None:
            continue
        meta[k] = v
        if k == "nmax":
            try:
                meta[k] = int(float(v))
            except (TypeError, ValueError):
                pass
    if "source_files" in meta and isinstance(meta["source_files"], str):
        try:
            meta["source_files"] = json.loads(meta["source_files"])
        except (ValueError, TypeError):
            pass
    meta["warnings"] = warnings
    return SHCoeffs(C, S, meta, times)


# ---------------------------------------------------------------------------
# legacy triangle table (m2py / the user's 3_processed/*.dat)
# ---------------------------------------------------------------------------
def _read_legacy_dat(p: Path, *, nmax: Optional[int] = None,
                     timeinfo: Optional[Path] = None,
                     timeinfo_path: Optional[str] = None) -> SHCoeffs:
    """Read the legacy ``<name>.dat`` (+ ``<name>_TimeInfo.dat``) pair.

    Layout: first line = epoch labels, body = ``(2*NC, ntime)`` -- i.e. exactly
    ``SHCoeffs.to_triangle().T`` with a header row.  The companion
    ``_TimeInfo.dat`` holds ``[idx, y_start, y_end, y_mid]`` (see
    :meth:`shkit.timeaxis.TimeAxis.from_legacy_timeinfo`).
    """
    txt = _text_read(p)
    lines = txt.splitlines()
    if not lines:
        raise ValueError(f"{p}: 空文件")
    header = lines[0].split()
    body = np.loadtxt(lines[1:], ndmin=2)
    ntime = body.shape[1]
    if len(header) not in (0, ntime):
        raise ValueError(
            f"{p}: 首行有 {len(header)} 个历元标签，但主体有 {ntime} 列，对不上")
    rows = body.shape[0]
    L = _infer_nmax_from_rows(rows)
    if nmax is not None and nmax < L:
        L = int(nmax)
    coeffs = SHCoeffs.from_triangle(body, L, meta={
        "format": "legacy_dat", "source_file": str(p),
        "legacy_header_labels": header})
    if coeffs.C.ndim == 3 and coeffs.C.shape[2] != ntime:      # pragma: no cover
        raise ValueError(f"{p}: 三角布局解析出的 ntime {coeffs.C.shape[2]} != {ntime}")

    # ---- time axis from the companion TimeInfo file, if present
    ti = timeinfo_path or timeinfo
    if ti is None:
        cand = str(p)
        for suf in (".dat", ".txt", ".DAT"):
            if cand.endswith(suf):
                ti = cand[:-len(suf)] + "_TimeInfo" + suf
                break
        else:
            ti = None
    meta = dict(coeffs.meta)
    warns = list(meta.get("warnings", []))
    times = None
    if ti and os.path.exists(ti):
        times = TimeAxis.from_legacy_timeinfo(str(ti))
        if len(times) != ntime:
            warns.append(
                f"TimeInfo 有 {len(times)} 个历元，系数表有 {ntime} 列 —— "
                "时间轴不可用，已降级为序号")
            times = TimeAxis.from_index(ntime)
        else:
            meta["timeinfo_file"] = str(ti)
    else:
        warns.append("找不到同名的 _TimeInfo.dat，时间轴降级为**序号**"
                     "（按日期操作会报错）")
        times = TimeAxis.from_index(ntime)
    meta["warnings"] = warns
    return SHCoeffs(coeffs.C, coeffs.S, meta, times)


def _write_legacy_dat(coeffs: SHCoeffs, p: Path, *, comment=None,
                      fmt: str = "%.6E", meta_extra=None,
                      timeinfo_path: Optional[str] = None) -> Path:
    """Write ``<name>.dat`` (+ ``<name>_TimeInfo.dat``) in the legacy layout."""
    tri = coeffs.to_triangle()                      # (2*NC, ntime)
    header = (coeffs.times.labels if (coeffs.times is not None
                                      and coeffs.times.labels)
              else [str(i + 1) for i in range(tri.shape[1])])
    lines = [" ".join(str(h) for h in header)]
    for row in tri:
        lines.append(" ".join(fmt % v for v in row))
    txt = "\n".join(lines) + "\n"
    if comment:
        txt = "".join(f"# {ln}\n" for ln in comment.splitlines()) + txt
    _text_write(p, txt, gzip_ok=False)
    ti = timeinfo_path
    if ti is None and coeffs.times is not None and coeffs.times.has_dates:
        ti = str(p)[:-4] + "_TimeInfo.dat" if str(p).endswith(".dat") else None
    if ti and coeffs.times is not None and coeffs.times.has_dates:
        coeffs.times.to_legacy_timeinfo(str(ti))
    return p


# ---------------------------------------------------------------------------
# series construction
# ---------------------------------------------------------------------------
def _resolve_series_inputs(source, *, pattern=None, recursive=True) -> tuple:
    """``source`` -> ``(paths, how)`` for a directory / glob / list / manifest."""
    if isinstance(source, (str, os.PathLike)):
        s = os.fspath(source)
        if os.path.isdir(s):
            pat = pattern or "*"
            hits = sorted(glob.glob(os.path.join(s, "**", pat), recursive=recursive)
                          if recursive else glob.glob(os.path.join(s, pat)))
            return hits, f"目录 {s}（pattern={pat!r}）"
        if os.path.isfile(s) and _suffix(s) in (".txt", ".list") and pattern is None:
            with open(s, encoding="utf-8", errors="replace") as fh:
                items = [ln.strip() for ln in fh
                         if ln.strip() and not ln.lstrip().startswith("#")]
            base = os.path.dirname(os.path.abspath(s))
            out = [it if os.path.isabs(it) else os.path.join(base, it) for it in items]
            return out, f"清单文件 {s}（{len(out)} 行）"
        if any(ch in s for ch in "*?["):
            return sorted(glob.glob(s, recursive=recursive)), f"通配 {s!r}"
        return [s], "单个文件"
    items = [os.fspath(x) for x in source]
    return items, f"显式列表（{len(items)} 个）"


def read_coeffs_series(source, *, nmax: Optional[int] = None, layout: str = "auto",
                       pattern: Optional[str] = None, product=("GSM",),
                       mission=None, center=None, rl=None,
                       epoch_from: str = "auto", sort: str = "time",
                       on_duplicate: str = "error", on_missing: str = "report",
                       strict_nmax: bool = True, recursive: bool = True,
                       timeinfo_path: Optional[str] = None,
                       progress=None) -> SHCoeffs:
    """Read many single-epoch coefficient files into **one** multi-epoch set.

    Parameters
    ----------
    source : path or sequence
        A directory, a glob, a ``.txt`` manifest (one path per line), a single
        file, or an explicit list of paths.
    pattern : str, optional
        ``glob`` pattern used when ``source`` is a directory (default ``"*"``).
    product : sequence of str
        Allowed product codes (``GSM``/``GAC``/``GAD``/``GAA`` ...).  Directories
        are **filtered** and the dropped files are reported; an explicit file
        list is **rejected** when it mixes products, so a GAD never sneaks into a
        GSM series unnoticed.
    mission, center, rl : str or sequence, optional
        Further name-based filters (``GRAC``/``GRFO``, ``UTCSR``/``GFZOP``/``JPLEM``,
        ``0600``/``0603``).
    epoch_from : {'auto', 'header', 'filename', 'index'}
        Where each epoch's date comes from.  ``'auto'`` prefers the gfc header
        when every file has one, else the file name; ``'index'`` keeps only the
        order (and then date-dependent work will refuse rather than guess).
    sort : {'time', 'name', 'none'}
    on_duplicate : {'error', 'first', 'last', 'report'}
        What to do when two files land on the same epoch (``<1`` day apart).
    on_missing : {'report', 'ignore'}
        Whether to list suspicious gaps in ``meta['warnings']``.
    strict_nmax : bool
        ``True`` (default) requires every file to resolve to the same degree;
        ``False`` truncates/zero-pads to the largest and reports which files
        were adjusted.

    Returns
    -------
    SHCoeffs
        ``(L+1, L+1, N)`` with ``times`` attached and a ``Warnings`` trail in
        ``meta``.  Nothing is ever dropped silently.
    """
    paths, how = _resolve_series_inputs(source, pattern=pattern, recursive=recursive)
    if not paths:
        raise FileNotFoundError(
            f"没有找到任何文件：{how}。"
            "（目录/通配没匹配到，或清单文件为空）")
    explicit = how.startswith("显式列表") or how == "单个文件"
    warnings: list = []

    # ---- name-based filtering -------------------------------------------
    keep, info = [], []
    for f in paths:
        try:
            p = parse_grace_filename(f)
        except ValueError:
            info.append((f, None))
            continue
        info.append((f, p))
    def _wanted(p) -> bool:
        if p is None:
            return False
        prod = [str(x).upper() for x in (product or [])]
        if prod and p["product"] not in prod:
            return False
        for key, want in (("mission", mission), ("center", center), ("rl", rl)):
            if want is None:
                continue
            allowed = [str(x).upper() for x in
                       (want if isinstance(want, (list, tuple, set)) else [want])]
            if str(p[key]).upper() not in allowed:
                return False
        return True

    dropped = [(f, "文件名不符合 GRACE 命名规则") for f, p in info if p is None]
    for f, p in info:
        if p is not None and not _wanted(p):
            dropped.append((f, f"{p['product']}/{p['mission']}/{p['center']}/{p['rl']}"))
        elif p is not None:
            keep.append(f)
    if dropped:
        if explicit:
            raise ValueError(
                f"显式给出的 {len(dropped)} 个文件不符合筛选条件 "
                f"(product={product}, mission={mission}, center={center}, rl={rl})：\n  "
                + "\n  ".join(os.path.basename(f) + f"  [{why}]"
                              for f, why in dropped[:8])
                + "\n如果确实要把它们读成一条序列，请放宽 product/mission/center/rl。")
        warnings.append(
            f"{len(dropped)} 个文件被名称筛选掉（product={product} 等），"
            f"例如：" + ", ".join(os.path.basename(f) for f, _ in dropped[:3]))
    if not keep:
        raise ValueError(
            f"{how} 里没有符合条件的系数文件（筛选后为 0）。"
            f"被筛掉的例子：" + (", ".join(os.path.basename(f) for f, _ in dropped[:3])
                                or "（无）"))

    # ---- time axis -------------------------------------------------------
    if epoch_from == "index":
        times = TimeAxis.from_index(len(keep))
    else:
        prefer = {"auto": "auto", "header": "header",
                  "filename": "filename"}[epoch_from]
        times = TimeAxis.from_grace(keep, prefer=prefer)
        warnings.extend(times.meta.get("warnings", []))

    # ---- order -----------------------------------------------------------
    if sort == "time" and times.has_dates:
        order = np.argsort(times.values, kind="stable")
        if not np.array_equal(order, np.arange(len(keep))):
            warnings.append("已按时间重排（原顺序与时间顺序不一致）")
    elif sort == "name":
        order = np.argsort([os.path.basename(f) for f in keep])
    else:
        order = np.arange(len(keep))
    keep = [keep[i] for i in order]
    times = times.select(order)

    # ---- duplicates ------------------------------------------------------
    if times.has_dates:
        groups = times.duplicates(tol_days=1.0)
        if groups:
            msg = "; ".join(f"{os.path.basename(keep[i])}" for i in groups[0][:4])
            if on_duplicate == "error":
                raise ValueError(
                    f"时间轴上有 {len(groups)} 组重复/过近历元（同一天两个文件）：{msg}。\n"
                    "请用 on_duplicate='first'|'last' 明确取舍，或先清理输入目录。")
            if on_duplicate in ("first", "last"):
                drop = sorted({(i if on_duplicate == "last" else i - 1)
                               for g in groups for i in g[1:]})
                keep = [f for i, f in enumerate(keep) if i not in set(drop)]
                times = times.select([i for i in range(len(times))
                                      if i not in set(drop)])
                warnings.append(f"按 on_duplicate={on_duplicate!r} 丢弃了 "
                                f"{len(drop)} 个重复历元")
            else:
                warnings.append(f"发现 {len(groups)} 组重复/过近历元（保留全部）：{msg}")
        if on_missing == "report":
            miss = times.missing()
            if miss:
                gaps = "; ".join(f"{str(a)[:10]}→{str(b)[:10]} 缺 {g:.0f} 天"
                                 for a, b, g in miss[:4])
                warnings.append(f"{len(miss)} 段疑似缺测：{gaps}")

    # ---- read and stack --------------------------------------------------
    mats, Ls, done = [], [], 0
    for f in keep:
        co = read_coeffs(f, nmax=nmax, layout=layout)
        if co.C.ndim == 3:
            if co.ntime != 1:
                raise ValueError(
                    f"{os.path.basename(f)}: 这是多历元文件（ntime={co.ntime}），"
                    "不能作为一个历元塞进序列；请先拆成单历元文件")
            co = co.time_slice(0)
        mats.append(co)
        Ls.append(co.nmax)
        done += 1
        if progress is not None:
            progress(f"读取 {done}/{len(keep)}：{os.path.basename(f)}",
                     done / max(len(keep), 1))
    Lmax, Lmin = max(Ls), min(Ls)
    if Lmax != Lmin:
        if strict_nmax:
            bad = [os.path.basename(keep[i]) for i, L in enumerate(Ls) if L != Lmax]
            raise ValueError(
                f"各历元的阶数不一致（{Lmin}..{Lmax}），例如 {bad[:3]}。\n"
                "传 strict_nmax=False 会统一到最高阶（低阶零填充）并逐条报告，"
                "或用 nmax= 显式截断。")
        warnings.append(
            f"各历元阶数不一致（{Lmin}..{Lmax}），已统一到 {Lmax} 阶"
            f"（低阶零填充）：" + ", ".join(
                os.path.basename(keep[i]) for i, L in enumerate(Ls) if L != Lmax)[:3])
    L = Lmax if nmax is None else int(nmax)
    C = np.zeros((L + 1, L + 1, len(mats)))
    S = np.zeros((L + 1, L + 1, len(mats)))
    for i, co in enumerate(mats):
        n = min(L, co.nmax)
        C[:n + 1, :n + 1, i] = co.C[:n + 1, :n + 1]
        S[:n + 1, :n + 1, i] = co.S[:n + 1, :n + 1]

    meta = {
        "format": "series", "nmax": L, "ntime": len(mats),
        "n_epochs": len(mats), "source_kind": how,
        "pattern": pattern, "product": list(product or []),
        "mission": mission, "center": center, "rl": rl,
        "epoch_from": epoch_from,
        "field_unit": mats[0].meta.get("field_unit", "geopotential"),
        "tide_system": mats[0].meta.get("tide_system"),
        "radius": mats[0].meta.get("radius"),
        "earth_gravity_constant": mats[0].meta.get("earth_gravity_constant"),
        "source_files": [os.path.basename(f) for f in keep],
        "warnings": warnings,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    if times.has_dates:
        meta["coverage_start"] = str(times.values[0])[:10]
        meta["coverage_end"] = str(times.values[-1])[:10]
    meta["time_source"] = times.meta.get("time_source")
    return SHCoeffs(C, S, meta, times)


def read_coeffs_pair(path, nmax=None, layout="auto"):
    """Read a file that holds **two** stacked triangle blocks (C and S apart).

    Convenience for legacy exports where ``C`` and ``S`` were written as two
    separate triangle blocks separated by a blank line or a ``# S`` header.
    Returns ``(SHCoeffs, meta)``.  When the file is a normal single block this
    simply delegates to :func:`read_coeffs`.
    """
    p = _require_file(path)
    text = _text_read(p)
    blocks = re.split(r"\n\s*\n", text.strip())
    if len(blocks) == 2:
        try:
            c = read_coeffs(path, nmax=nmax, layout=layout)
            return c, c.meta
        except ValueError:
            pass
    c = read_coeffs(p, nmax=nmax, layout=layout)
    return c, c.meta
