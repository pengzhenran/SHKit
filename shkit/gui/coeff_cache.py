# -*- coding: utf-8 -*-
"""
shkit.gui.coeff_cache
=====================

Local cache for **per-epoch coefficients** (the expensive part of the GUI).

Why
---
「运行分析 / 批量分析」一次解出整条序列要花的是**固定成本**（积分元、Gram、
勒让德递推）＋极小的每历元边际成本（实测 203 历元 / 1° 全球 / nmax=60：
批量 + 经度 FFT 0.53 s）。既然一次就能全部解完，就没有理由让用户为"再看一遍
同一份数据"再等一次 —— 所以结果落盘，下次打开（或换视图、导 GIF、逐时次导出）
直接读。

What is cached
--------------
``C`` / ``S``（3-D，``(nmax+1, nmax+1, ntime)``）＋ 逐历元诊断表（JSON）＋
命中所需的**键**。键包含：

* 源数据：字节数、``mtime_ns``、**开头 64 KB 的 sha1**、变量名/种类、点数、历元数
  （**不含绝对路径** —— 拷贝/搬家/换机器之后缓存仍然有效，见 :func:`cache_key`）；
* 参数：nmax、积分元规则、估计方法、迭代次数、tau、归一化、正则、alpha、
  正则幂、**高斯平滑半径**、输入/输出物理量；
* 缓存格式版本（:data:`CACHE_VERSION`）—— 这个**参与匹配**；SHKit 版本号只写进键里
  **留档**、不参与匹配（补丁号不该让用户的缓存全部作废，见 :func:`_blob`）。

任一项不同即视为**未命中**（重算并覆盖），所以"改了参数却拿到旧结果"不会发生。

Where
-----
优先写在数据文件**旁边**（``<name>.shkit-coeffs.npz``），这样把数据一起拷走
缓存也跟着走（v3 起键里不含绝对路径，所以这一条**真的**成立）；数据目录只读或
无权限时退到用户缓存目录（``%LOCALAPPDATA%/SHKit/cache``），并且**把这件事说出来**
（返回值里带 ``note``），不静默换地方。

How big can it get
------------------
两处的增长性**不一样**，这里说清楚：

* 数据文件旁边：**一个数据集一份**（改参数就覆盖同一个文件），大小
  ≈ ``(nmax+1)(nmax+2)/2 × ntime × 16 B``（压缩前）。实测 nmax=60 / 256 历元
  = 7.0 MB。它跟着数据走、用户看得见，所以**只受单份上限**
  :data:`ENTRY_MAX_BYTES` 约束，不会自动删（要删有「文件 → 本地缓存…」）。
* 用户缓存目录：按"数据集 × 参数"散列，**天然会越攒越多**，所以那里有总量上限
  :data:`USER_CACHE_MAX_BYTES`，每次写盘后按最久未用自动修剪到 80%，并在状态栏
  写明删了几份、释放多少。

Honesty
-------
读回来的东西一律带 ``source``（``'data-dir'`` / ``'user-cache'``），界面必须据此
说明"这是本地缓存，不是本次重算"，因为缓存命中的前提是**参数完全一致**，而这一点
用户看不见。
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from .. import __version__ as SHKIT_VERSION
from ..coeffs import SHCoeffs

__all__ = ["CACHE_VERSION", "SUFFIX", "ENTRY_MAX_BYTES", "USER_CACHE_MAX_BYTES",
           "cache_key", "load", "save", "paths", "clear", "estimate_bytes",
           "prune_user_cache", "clear_user_cache", "report", "user_cache_dir",
           "probe", "find_any"]

#: Bump whenever the on-disk layout or the meaning of the key changes.
#: v2: ``analyze_series`` 修复了系数单位元数据（此前把「输入是」那一档写成
#: ``field_unit``，导致读回来的系数被当成 EWH/geoid 系数而跳过反变换换算）。
#: 旧缓存里的 meta 是错的，必须整体作废。
#: v3: 键里**去掉绝对路径**、改用「前 64 KB 的 sha1」当内容指纹 —— 数据（连同旁边
#: 的缓存）换个目录/换台机器仍然命中。旧键的字段形状不同，自动全部未命中并重写。
CACHE_VERSION = 3

#: Suffix written next to the data file.
SUFFIX = ".shkit-coeffs.npz"

#: 单份缓存的大小上限。C/S 是 ``(nmax+1, nmax+2)/2 × ntime × 2 × 8 B``，所以
#: 它随 nmax²·ntime 长：实测 nmax=60 / 256 历元 = 7.0 MB（压缩后），
#: nmax=120 / 256 = 27 MB，nmax=200 / 256 = 75 MB。256 MiB 足够覆盖正经用法，
#: 超过就**不写盘并说出来**（而不是悄悄在用户的数据盘上放一个几 GB 的文件）。
ENTRY_MAX_BYTES = 256 * 1024 ** 2

#: 用户缓存目录（数据目录不可写时的退路）的总量上限。那里是"每个
#: 数据集 × 每组参数一份"，本来会无限涨 —— 超过就按最久未用删到 80%。
USER_CACHE_MAX_BYTES = 512 * 1024 ** 2

#: 修剪后保留的比例（留一点余量，免得每写一份就立刻再修剪）。
_PRUNE_TARGET_FRAC = 0.8

#: Only these parameters change the coefficients themselves (``target_unit``
#: changes the *output field* only, but a request that asks for another output
#: unit is cheap to recompute from ``C``/``S`` -- still, include it so a cache
#: can never serve a differently-flavoured job silently).
_PARAM_KEYS = ("nmax", "rule", "method", "niter", "tau", "normalise", "reg",
               "alpha", "reg_power", "gaussian_km", "field_unit", "target_unit")


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------
def _head_hash(path: str, nbytes: int = 65536):
    """文件开头 ``nbytes`` 字节的 sha1（读不到就是 None）。

    这是"内容指纹"：拷走文件后仍然相同，所以键里可以**不用绝对路径**也不会认错
    文件（见 :func:`cache_key` 的说明）。
    """
    try:
        with open(path, "rb") as fh:
            return hashlib.sha1(fh.read(nbytes)).hexdigest()
    except OSError:
        return None


def _file_identity(path: str) -> dict:
    try:
        st = os.stat(path)
        return {"path": os.path.abspath(path), "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "head": _head_hash(path)}
    except OSError:
        return {"path": os.path.abspath(path), "size": None, "mtime_ns": None,
                "head": None}


def cache_key(dataset, params: dict) -> dict:
    """Everything a hit must match, as a plain JSON-able dict.

    ⚠️ **匹配不看绝对路径**。v2 的键里有 ``os.path.abspath(path)``，于是"把数据
    连同旁边的缓存一起拷到别的目录/别的盘"就**全部命不中** —— 而"写在数据文件
    旁边"的全部意义正是**拷走缓存跟着走**（实测用户那份 7 MB 缓存就是这么废掉
    的：键里写的是 ``D:\\myds\\…``，文件现在躺在 ``D:\\华为家庭存储\\…``）。
    文件身份改用 ``size + mtime_ns + head(前 64 KB 的 sha1) + kind + npoints +
    ntime + variable``：

    * 拷贝（``copy2``/同步盘/资源管理器）会保留 mtime_ns、内容不变 → 指纹相同 → **命中**；
    * 内容改了（size / mtime 变，或开头字节变）→ **一定不命中**；
    * 同一个文件被搬到别处 → **命中**（不再白算一遍）。

    路径不再进键，所以用户缓存目录那份文件名（按键散列）也跟着变得可搬。
    """
    meta = dict(getattr(dataset, "meta", None) or {})
    ds_id = getattr(dataset, "path", None)
    # ``<D5 测试网格>`` 这类内存数据集没有真文件：没有身份信息，也不写盘
    # （``save()`` 会拒绝）。in-memory 数据集的 size/mtime/head 为 None。
    ident = _file_identity(ds_id) if ds_id and os.path.exists(str(ds_id)) \
        else {"path": str(ds_id), "size": None, "mtime_ns": None, "head": None}
    source = {
        "size": ident["size"],
        "mtime_ns": ident["mtime_ns"],
        "head": ident["head"],
        "kind": str(getattr(dataset, "kind", "")),
        "npoints": int(getattr(dataset, "npoints", 0) or 0),
        "ntime": int(getattr(dataset, "ntime", 0) or 0),
        "variable": meta.get("nc_variable") or meta.get("variable"),
    }
    return {
        "cache_version": CACHE_VERSION,
        "shkit_version": SHKIT_VERSION,
        "source": source,
        "params": {k: params.get(k) for k in _PARAM_KEYS},
    }


def _digest(key: dict) -> str:
    blob = _blob(key)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _blob(key: dict) -> str:
    """键的**比较串**：``shkit_version`` 只留档，不参与匹配。

    为什么：键里带软件版本会让**补丁号**（2.0.0 → 2.0.1）把用户所有缓存一次作废 ——
    而补丁往往只是界面/文案修复，系数的含义没变。真正决定"缓存还能不能用"的是
    :data:`CACHE_VERSION`（缓存格式/系数语义变了才 +1，见那里的注释），所以版本号
    只写进 payload 供诊断，比较时剔掉。
    """
    d = {k: v for k, v in dict(key or {}).items() if k != "shkit_version"}
    return json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)


def user_cache_dir() -> str:
    base = (os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(base, "SHKit", "cache")


def paths(dataset) -> list:
    """Candidate cache paths, best first: next to the data, then user cache."""
    out = []
    src = getattr(dataset, "path", None)
    if src and os.path.exists(str(src)):
        out.append(os.path.abspath(str(src)) + SUFFIX)
    return out


def _user_path(dataset, key: dict) -> str:
    return os.path.join(user_cache_dir(), _digest(key) + ".npz")


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------
def _jsonable_table(table: dict) -> dict:
    out = {}
    for col, v in (table or {}).items():
        arr = np.asarray(v)
        if arr.dtype.kind in "US" or arr.dtype == object:
            out[col] = [str(x) for x in arr.ravel()]
        else:
            out[col] = [float(x) for x in arr.ravel()]
    return out


def _fmt_size(nbytes: float) -> str:
    if nbytes < 1024:
        return f"{nbytes:.0f} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.0f} kB"
    return f"{nbytes / 1e6:.1f} MB"


def estimate_bytes(coeffs) -> int:
    """写盘前估算 C/S 的裸字节数（压缩前；压缩比通常 2–50×）。"""
    try:
        return int(np.asarray(coeffs.C).size + np.asarray(coeffs.S).size) * 8
    except Exception:                                            # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# size accounting / pruning (the cache must not grow without bound)
# ---------------------------------------------------------------------------
def _entries() -> list:
    """User-cache files as ``[(path, size, mtime)]``, oldest first."""
    out = []
    d = user_cache_dir()
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".npz"):
            continue
        p = os.path.join(d, name)
        try:
            st = os.stat(p)
        except OSError:
            continue
        out.append((p, int(st.st_size), float(st.st_mtime)))
    out.sort(key=lambda e: e[2])
    return out


def prune_user_cache(max_bytes: int = None) -> tuple:
    """把用户缓存目录修剪到上限内（最久未用的先删）。

    Returns ``(removed_count, freed_bytes, removed_paths)``.  只在**写入用户缓存
    之后**调用；数据文件旁边那份缓存不归这里管（它跟着数据走，用户自己能看见
    也自己能删）。
    """
    cap = int(USER_CACHE_MAX_BYTES if max_bytes is None else max_bytes)
    items = _entries()
    total = sum(e[1] for e in items)
    if cap <= 0 or total <= cap:
        return 0, 0, []
    target = int(cap * _PRUNE_TARGET_FRAC)
    removed, freed = [], 0
    # 从最旧的开始删，直到降到 target 以内；刚写进来的那份是最新的，留到最后。
    for path, size, _mtime in items:
        if total - freed <= target:
            break
        try:
            os.remove(path)
        except OSError:
            continue
        removed.append(path)
        freed += size
    return len(removed), freed, removed


def clear_user_cache() -> tuple:
    """清空用户缓存目录。Returns ``(removed_count, freed_bytes)``."""
    n, freed = 0, 0
    for path, size, _mtime in _entries():
        try:
            os.remove(path)
        except OSError:
            continue
        n += 1
        freed += size
    return n, freed


def _read_key(path: str):
    """只读 npz 里的 ``key`` / ``created`` / ``source_path``（**不解压** C/S）。"""
    try:
        with np.load(path, allow_pickle=False) as z:
            if "key" not in z:
                return None
            return {"key": json.loads(str(z["key"])),
                    "created": str(z["created"]) if "created" in z else "",
                    "source_path": (str(z["source_path"])
                                    if "source_path" in z else "")}
    except Exception:                                            # noqa: BLE001
        return None                     # 损坏/半截/读不动一律当"没有"


def _candidate_files(dataset) -> list:
    """所有可能装着这份数据缓存的（路径, 来源, 字节数, mtime）。

    数据旁边那个在前（它跟着数据走，优先信它），用户缓存目录里所有 ``.npz`` 在后
    —— 那个目录文件名是键的散列，没法直接算出来，只能逐个读 ``key`` 成员筛。
    """
    out = []
    for p in paths(dataset):
        try:
            st = os.stat(p)
        except OSError:
            continue
        out.append((p, "data-dir", int(st.st_size), float(st.st_mtime)))
    for p, size, mtime in _entries():
        out.append((p, "user-cache", size, mtime))
    return out


def find_any(dataset) -> dict:
    """这份**数据**有没有缓存？**不要求参数一致**。

    Returns ``{"params", "path", "source", "created", "mtime"}`` 或 ``None``
    （同一份数据有多份时：数据旁边那份优先，否则取最新的）。

    为什么需要它：``field_unit``（「输入是」）是键的一部分 —— 同一块网格声明成
    geoid / EWH 是两个不同的物理场，除的 ``f_u`` 不同，**系数确实不一样**，所以
    它必须在键里。但这会带来一个死循环：界面的默认「输入是」与缓存里的不同 →
    重算 → 缓存被写成默认那套 → 下次默认又对不上……（实测用户那份 mascon 就这么
    来回重算了两次）。界面应该在**载入时**据此把参数切回缓存那一套，然后命中。
    """
    try:
        want_src = cache_key(dataset, {})["source"]
    except Exception:                                            # noqa: BLE001
        return None
    best = None
    for path, source, _size, mtime in _candidate_files(dataset):
        got = _read_key(path)
        if not got:
            continue
        key = got["key"]
        if key.get("cache_version") != CACHE_VERSION:
            continue                    # 旧格式：参数可能已失效，不拿来恢复
        if key.get("source") != want_src:
            continue                    # 不是这份数据（文件指纹不同）
        if source == "data-dir":
            return {"params": dict(key.get("params") or {}), "path": path,
                    "source": source, "created": got["created"], "mtime": mtime}
        if best is None or mtime > best["mtime"]:
            best = {"params": dict(key.get("params") or {}), "path": path,
                    "source": source, "created": got["created"], "mtime": mtime}
    return best


def probe(dataset, params: dict):
    """**这份缓存到底能不能命中？** —— 只看键，不解压数组。

    Returns ``{"path", "source", "created", "source_path"}`` 或 ``None``。

    为什么单独要这个：早先参数面板那句提示是按"**文件在不在**"写的，于是旁边躺着
    一份**键对不上**的旧缓存时（实测：用户那份 7.38 MB 的缓存键里写的是
    ``D:\\myds\\…``、文件已在 ``D:\\华为家庭存储\\…``），界面会说"运行分析直接读它，
    不重算"，而实际上会重算 —— 这正是最不能接受的那种撒谎。存在 ≠ 能用。
    """
    want = cache_key(dataset, params)
    want_blob = _blob(want)
    for path, source, _size, _mtime in _candidate_files(dataset):
        got = _read_key(path)
        if not got:
            continue
        if _blob(got["key"]) != want_blob:
            continue
        return {"path": path, "source": source, "created": got["created"],
                "source_path": got["source_path"]}
    return None


def report(dataset=None, params: dict = None) -> dict:
    """缓存现状（给界面「本地缓存…」用）。

    Keys: ``data_path`` / ``data_exists`` / ``data_size`` / ``data_mtime`` /
    ``data_match``（给了 ``params`` 才是 True/False，否则 None）,
    ``user_dir`` / ``user_files`` / ``user_total`` / ``user_limit`` /
    ``entry_limit``.
    """
    out = {"data_path": None, "data_exists": False, "data_size": 0,
           "data_mtime": None,
           "user_dir": user_cache_dir(), "user_files": 0, "user_total": 0,
           "user_limit": int(USER_CACHE_MAX_BYTES),
           "entry_limit": int(ENTRY_MAX_BYTES)}
    if dataset is not None:
        for p in paths(dataset):
            out["data_path"] = p
            try:
                st = os.stat(p)
            except OSError:
                continue
            out["data_exists"] = True
            out["data_size"] = int(st.st_size)
            out["data_mtime"] = float(st.st_mtime)
    items = _entries()
    out["user_files"] = len(items)
    out["user_total"] = int(sum(e[1] for e in items))
    # 「文件在不在」和「能不能命中」是两件事：给了 params 就顺便如实回答后者。
    out["data_match"] = None
    if dataset is not None and params is not None:
        out["data_match"] = probe(dataset, params) is not None
    return out


def save(dataset, params: dict, coeffs, report=None) -> tuple:
    """Write the cache.  Returns ``(path_or_None, note)``.

    ``note`` is a short sentence for the status bar; it never claims success when
    the file was not written.
    """
    key = cache_key(dataset, params)
    if not paths(dataset):
        return None, "数据集没有磁盘文件，未写缓存。"
    est = estimate_bytes(coeffs)
    if est > ENTRY_MAX_BYTES:
        return None, (f"逐历元系数约 {_fmt_size(est)}，超过单份缓存上限 "
                      f"{_fmt_size(ENTRY_MAX_BYTES)}，本次未写缓存"
                      "（结果照常可用，只是下次还要重算）。")
    payload = {
        "C": np.asarray(coeffs.C, dtype=float),
        "S": np.asarray(coeffs.S, dtype=float),
        "key": json.dumps(key, ensure_ascii=False, sort_keys=True, default=str),
        "meta": json.dumps(dict(coeffs.meta or {}), ensure_ascii=False,
                           default=str),
        "table": json.dumps(_jsonable_table(getattr(report, "table", {}) or {}),
                            ensure_ascii=False),
        "shared": json.dumps(getattr(report, "shared", {}) or {},
                             ensure_ascii=False, default=str),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        # 记录当时的数据文件路径**仅供诊断**：它不参与命中判断（键里没它），
        # 所以拷走/搬家之后缓存照样命中，而"这份缓存本来是给哪个文件算的"仍查得到。
        "source_path": str(getattr(dataset, "path", "") or ""),
    }
    errs = []
    src_paths = paths(dataset)
    for path in src_paths + [_user_path(dataset, key)]:
        where = "数据文件旁边" if path in src_paths else "用户缓存目录"
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez_compressed(path, **payload)
            mb = _fmt_size(os.path.getsize(path))
            note = f"逐历元系数已缓存到{where}（{mb}）：{path}"
            if where == "用户缓存目录":
                # 用户缓存目录是"数据集×参数"一份、本来无限涨：写完立刻修剪，
                # 并把删了什么、为什么删说出来（不静默丢用户的东西）。
                n, freed, _ = prune_user_cache()
                if n:
                    note += (f"　⚠ 用户缓存超出上限 "
                             f"{_fmt_size(USER_CACHE_MAX_BYTES)}，"
                             f"已按最久未用删除 {n} 份旧缓存（释放 "
                             f"{_fmt_size(freed)}）。")
                else:
                    note += (f"　（用户缓存目录上限 "
                             f"{_fmt_size(USER_CACHE_MAX_BYTES)}，"
                             "超出会按最久未用自动清理）")
            return path, note
        except Exception as exc:                                 # noqa: BLE001
            errs.append(f"{where} {path}: {exc}")
    return None, "缓存写入失败（不影响结果）：" + errs[0]


def load(dataset, params: dict):
    """Return a hit as a dict, or ``None``.

    Keys: ``coeffs`` (SHCoeffs, ``times`` re-attached from the dataset),
    ``table`` / ``shared`` (from the cached report, may be ``{}``), ``path``,
    ``source`` (``'data-dir'`` / ``'user-cache'``), ``created``,
    ``params``（缓存里记的那套参数 —— 界面要拿它跟当前界面对照，好让用户在
    "读缓存 / 重算"之间做决定时知道差别在哪）。
    """
    want = cache_key(dataset, params)
    want_blob = _blob(want)
    candidates = [(p, "data-dir") for p in paths(dataset)]
    candidates.append((_user_path(dataset, want), "user-cache"))
    for path, source in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                got = json.loads(str(z["key"]))
                if _blob(got) != want_blob:
                    continue
                C = np.asarray(z["C"], dtype=float)
                S = np.asarray(z["S"], dtype=float)
                meta = json.loads(str(z["meta"])) if "meta" in z else {}
                table = json.loads(str(z["table"])) if "table" in z else {}
                shared = json.loads(str(z["shared"])) if "shared" in z else {}
                created = str(z["created"]) if "created" in z else ""
        except Exception:                                        # noqa: BLE001
            continue                    # 损坏/半截文件一律当未命中，下次覆盖
        if C.ndim != 3 or C.shape[0] != C.shape[1] or S.shape != C.shape:
            continue
        if int(C.shape[2]) != int(getattr(dataset, "ntime", 0) or 0):
            continue
        times = dataset.time_axis()
        if times is not None and len(times) != C.shape[2]:
            times = None
        coeffs = SHCoeffs(C, S, dict(meta), times)
        tab = {k: (np.array(v, dtype=float) if v and
                   not isinstance(v[0], str) else np.array(v))
               for k, v in (table or {}).items()}
        return {"coeffs": coeffs, "table": tab, "shared": shared,
                "path": path, "source": source, "created": created,
                "params": dict(got.get("params") or {})}
    return None


def clear(dataset, params: dict = None) -> list:
    """Delete the cache files of this dataset.  Returns what was removed.

    给了 ``params`` 就连**用户缓存目录里当前参数那一份**一起删（只删这一份：
    那个目录按"数据集×参数"散列，别的参数是别的文件）。
    """
    targets = list(paths(dataset))
    if params is not None:
        try:
            targets.append(_user_path(dataset, cache_key(dataset, params)))
        except Exception:                                        # noqa: BLE001
            pass
    removed = []
    for path in targets:
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed.append(path)
        except OSError:
            pass
    return removed
