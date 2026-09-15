# -*- coding: utf-8 -*-
"""
shkit.timeaxis
==============

A first-class description of the **time axis** that a multi-epoch coefficient
set or field series lives on.

Design decisions (see ``docs/多时间数据处理方案.md`` §3 and §4.1):

* **Canonical internal form is ``datetime64``**, not decimal years.  GRACE monthly
  solutions are 27-35 days apart with gaps and two-solutions-per-year, so any
  arithmetic on "months" must use real elapsed time.  Decimal years are a *label
  and interchange* format only.
* **The legacy decimal-year convention is frozen and non-obvious** (verified
  against the real ``CSR_rawSH60_total_216_TimeInfo.dat``: 200/203 epochs exact
  to 1e-6):

  .. code-block:: text

      start = ordinal(YDDD_start)          # e.g. 2002095
      end   = ordinal(YDDD_end)            # e.g. 2002120  (inclusive last day)
      mid   = (start + end + 1) / 2        # <- the +1 makes the end exclusive
      base  = 366 if isleap(year_of_start) else 365
      decimal_year = year_of_start + (mid - ordinal(year_of_start, 1, 1)) / base

  Getting either the ``+1`` or the leap-year ``366`` wrong puts every epoch off
  by up to a day.  The 3 cross-year epochs (e.g. ``2016346-2017006``) use the
  start year as the base and still carry a ~2e-5 residual -- those are the only
  known deviation, and they are why an explicit hint is stored rather than always
  re-derived.
* **Dates are never invented.**  When an epoch's date cannot be established the
  axis degrades to ``kind='index'`` and any date-dependent operation raises,
  instead of quietly pretending epochs are equally spaced.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["TimeAxis", "parse_grace_filename", "GRACE_NAME_RE",
           "time_axis_from_meta", "attr_ci",
           "decimal_year_of_date", "date_of_decimal_year"]


def attr_ci(attrs, name: str, default=None):
    """Case-insensitive attribute lookup (``units`` vs ``Units``).

    Real netCDF products are not all CF-clean: the CSR mascon files write
    ``Units`` with a capital U for the time *and* the lat/lon axes.  A case
    sensitive ``attrs.get('units')`` silently returns ``None`` there, and the
    time axis then degrades to "no dates" -- which is how a file with 256 monthly
    epochs ends up telling the user it has no time information.
    """
    if not attrs:
        return default
    low = str(name).lower()
    for k, v in attrs.items():
        if str(k).lower() == low:
            return v
    return default


def time_axis_from_meta(values, units=None, calendar=None,
                        name: str = "time") -> Optional["TimeAxis"]:
    """Build a :class:`TimeAxis` from whatever a reader put in ``meta``.

    The **single** entry point for this job.  Before it existed, four call sites
    each re-implemented the same fragile dance (``hasattr(raw, 'values') ? ... :
    TimeAxis.from_datetimes(raw)``), and all four broke the same way on a
    numeric time coordinate: the numbers went to ``from_datetimes``, which
    raises, and the caller swallowed it into "no time axis".

    Accepts: an existing ``TimeAxis``; an xarray/netCDF coordinate (has
    ``.values``); a mapping with ``values``/``units``/``calendar``; a sequence of
    ``datetime64``, ISO strings, or plain numbers (with ``units`` supplied
    separately, e.g. from ``meta['time_units']``).
    """
    if values is None:
        return None
    if isinstance(values, TimeAxis):
        return values
    # ⚠️ 顺序与 `callable` 判断都不能省：`dict` 也有 `.values` **方法**，
    # 若先做 `hasattr(values, "values")` 就会把 {"values":…, "units":…} 这种
    # 描述字典当成 xarray 坐标，然后拿 bound method 去构造日期 —— 实测会静默
    # 降级成"没有时间轴"。xarray 的 `.values` 是属性（不可调用），据此区分。
    if isinstance(values, dict):
        units = values.get("units", units)
        calendar = values.get("calendar", calendar)
        values = values.get("values")
        if values is None:
            return None
    elif hasattr(values, "values") and not callable(getattr(values, "values")):
        try:
            return TimeAxis.from_netcdf_coord(values)
        except Exception:                                        # noqa: BLE001
            return None
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return TimeAxis(arr, "datetime", meta={"time_source": "meta:datetime"})
    if arr.dtype.kind in "USO":
        try:
            return TimeAxis.from_datetimes(arr)
        except Exception:                                        # noqa: BLE001
            return None
    if arr.dtype.kind in "iuf" and units:
        return TimeAxis.from_numeric_time(arr, units, calendar=calendar,
                                          source=f"meta:{name}:numeric")
    # 纯序号：不编日期（这是设计约定，不是失败）
    return TimeAxis.from_index(arr.size,
                               note=f"{name} 只有序号、没有可解码的时间信息")

#: ``GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc``
GRACE_NAME_RE = re.compile(
    r"^(?P<product>[A-Za-z]{3})-?(?P<version>\d*)_?"
    r"(?P<start>\d{7})-(?P<end>\d{7})_"
    r"(?P<mission>[A-Z]{4})_(?P<center>[A-Z]{5})_"
    r"(?P<degtype>[A-Z]{2}\d{2})_(?P<rl>\d{4})",
    re.IGNORECASE)

_DAY = np.timedelta64(1, "D")
_SEC = np.timedelta64(1, "s")


# ---------------------------------------------------------------------------
# decimal-year convention
# ---------------------------------------------------------------------------
def _ordinal(year: int, doy: int) -> int:
    """Ordinal of day-of-year ``doy`` (1-based) in ``year``."""
    return _dt.date(year, 1, 1).toordinal() + doy - 1


def _year_length(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def decimal_year_of_date(when, *, base_year: Optional[int] = None) -> float:
    """Legacy decimal year of a date, using the frozen convention.

    ``base_year`` lets a cross-year epoch be expressed against its *start* year
    (which is what the legacy pipeline does); by default the date's own year is
    used.  The divisor is that year's length (366 in a leap year).
    """
    if isinstance(when, np.datetime64):
        when = when.astype("datetime64[s]").astype(_dt.datetime)
    if isinstance(when, _dt.datetime):
        date, frac = when.date(), (when.hour * 3600 + when.minute * 60
                                   + when.second + when.microsecond * 1e-6) / 86400.0
    else:
        date, frac = when, 0.0
    y = base_year if base_year is not None else date.year
    doy = (date - _dt.date(y, 1, 1)).days + 1
    return y + (doy - 1 + frac) / _year_length(y)


def date_of_decimal_year(value: float, *, base_year: Optional[int] = None):
    """Inverse of :func:`decimal_year_of_date` at **day** resolution.

    Rounds to the nearest day: ``decimal_year_of_date`` divided by 365/366 is
    rarely exact in binary, and truncating a ``106.9999999``-day offset lands one
    day early (that bug cost one epoch in the round-trip test).
    """
    y = int(np.floor(value)) if base_year is None else int(base_year)
    doy0 = (value - y) * _year_length(y)          # doy0 = doy - 1
    return _dt.date(y, 1, 1) + _dt.timedelta(days=int(round(doy0)))


def datetime_of_decimal_year(value: float, *, base_year: Optional[int] = None):
    """Inverse of :func:`decimal_year_of_date`, keeping the **fractional day**.

    GRACE monthly spans are often an odd number of days, so the midpoint lands on
    ``12:00``; that half day is part of the legacy decimal year and must survive
    the round trip (``date_of_decimal_year`` would drop it).  Rounding the total
    *seconds* rather than the fraction avoids the same 1-day truncation.
    """
    y = int(np.floor(value)) if base_year is None else int(base_year)
    total_s = int(round((value - y) * _year_length(y) * 86400.0))
    days, secs = divmod(total_s, 86400)
    return _dt.datetime(y, 1, 1) + _dt.timedelta(days=days, seconds=secs)


def _epoch_from_ordinals(start_ord: int, end_ord: int, start_year: int) -> tuple:
    """``(mid_datetime, decimal_year)`` from two ordinals and the start year.

    The **single** place implementing the frozen convention -- both the file-name
    and the gfc-header paths go through it, so they cannot drift apart:

        mid = (start + end + 1) / 2        # end_ord is the INCLUSIVE last day
        decimal_year = start_year + (mid - jan1(start_year)) / len(start_year)

    Callers must convert their own date syntax to ordinals first, because the two
    sources use different ones: file names use ``YYYYDDD`` (day-of-year) while gfc
    headers use ``YYYYMMDD``.  Feeding a ``MMDD`` number into a day-of-year slot
    silently lands ~a year away, which is exactly the bug this signature prevents.

    The header's own ``(mid: YYYYMMDD)`` is deliberately **not** trusted for the
    value: it is rounded to whole days and would lose the 12:00 midpoint that the
    legacy ``TimeInfo`` file actually carries.
    """
    mid_ord = (start_ord + end_ord + 1) / 2.0
    whole = int(np.floor(mid_ord))
    mid_dt = (_dt.datetime.fromordinal(whole)
              + _dt.timedelta(seconds=int(round((mid_ord - whole) * 86400.0))))
    dec = (start_year
           + (mid_ord - _dt.date(start_year, 1, 1).toordinal())
           / _year_length(start_year))
    return mid_dt, dec


def _ordinal_from_ymd(ymd: str) -> tuple:
    """``('20020405') -> (ordinal, year)``.  The gfc header's date syntax."""
    y, m, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
    return _dt.date(y, m, d).toordinal(), y


def parse_grace_filename(path_or_name: str) -> dict:
    """Parse a GRACE/GRACE-FO GSM/GAC/GAD file name.

    Returns a dict with ``start``/``end`` (``YYYYDDD`` strings), the derived
    ``start_date``/``end_date`` (the end is the **inclusive last day**),
    ``mid_date`` and ``decimal_year``, plus mission/center/degree/RL fields.

    Raises ``ValueError`` when the name does not match at all -- never guesses.
    """
    name = os.path.basename(str(path_or_name))
    m = GRACE_NAME_RE.match(name)
    if not m:
        raise ValueError(f"无法从文件名解析历元（不符合 GRACE 命名规则）: {name}")
    s, e = m.group("start"), m.group("end")
    ys, ds = int(s[:4]), int(s[4:])
    ye, de = int(e[:4]), int(e[4:])
    start_date = _dt.date(ys, 1, 1) + _dt.timedelta(days=ds - 1)
    end_date = _dt.date(ye, 1, 1) + _dt.timedelta(days=de - 1)
    mid_dt, dec = _epoch_from_ordinals(_ordinal(ys, ds), _ordinal(ye, de), ys)
    mid_date = mid_dt.date()
    return {
        "product": m.group("product").upper(),
        "mission": m.group("mission").upper(),
        "center": m.group("center").upper(),
        "degtype": m.group("degtype").upper(),
        "rl": m.group("rl"),
        "start": s, "end": e,
        "start_date": start_date, "end_date": end_date,
        "mid_date": mid_date, "mid_datetime": mid_dt,
        # expressed against the START year (matters for cross-year epochs)
        "decimal_year": dec,
    }


_HEAD_KEYS = ("time_coverage_start", "time_coverage_end", "time_period_of_data")
_PERIOD_RE = re.compile(r"(\d{8})\s*-\s*(\d{8})(?:\s*\(mid:\s*(\d{8})\))?")


def parse_gfc_time_header(path: str, *, head_bytes: int = 8192) -> Optional[dict]:
    """Pull the epoch information out of a gfc/YAML header, or ``None``.

    Recognised: ``time_coverage_start`` / ``time_coverage_end`` (ISO) and
    ``time_period_of_data 20020405 - 20020430 (mid: 20020418)``.  Values in the
    converted GRACE gfc files carry a leading ``": "`` which is stripped.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(head_bytes)
    except OSError:
        return None
    found: dict = {}
    for line in head.splitlines():
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        for key in _HEAD_KEYS:
            if not s.lower().startswith(key):
                continue
            value = s[len(key):].lstrip().lstrip(":").strip()
            if key == "time_period_of_data":
                mm = _PERIOD_RE.search(value)
                if mm:
                    # NOTE: the header uses YYYYMMDD, file names use YYYYDDD.
                    found["start_ymd"] = mm.group(1)
                    found["end_ymd"] = mm.group(2)       # inclusive last day
                    found["mid_ymd"] = mm.group(3)
            else:
                found[key] = value
            break
    if not found:
        return None
    # prefer the explicit data period; fall back to the coverage window
    if "start_ymd" in found:
        s_ord, ys = _ordinal_from_ymd(found["start_ymd"])
        e_ord, _ = _ordinal_from_ymd(found["end_ymd"])
        mid_dt, dec = _epoch_from_ordinals(s_ord, e_ord, ys)
        found["mid_datetime"] = mid_dt
        found["mid_date"] = mid_dt.date()
        found["decimal_year"] = dec
    elif "time_coverage_start" in found:
        # PO.DAAC SHM files (``GRCOF2``) carry only the coverage window, and its
        # END is *exclusive* (``2021-12-01 .. 2022-01-01`` is December), unlike
        # ``time_period_of_data``'s inclusive last day.  So no "+1" here -- the
        # two branches really do need different arithmetic.
        try:
            t0 = np.datetime64(found["time_coverage_start"][:19])
            t1 = np.datetime64(found.get("time_coverage_end",
                                         found["time_coverage_start"])[:19])
            mid = t0 + (t1 - t0) / 2
            found["mid_datetime"] = mid.astype("datetime64[s]").astype(_dt.datetime)
            found["mid_date"] = found["mid_datetime"].date()
            found["decimal_year"] = decimal_year_of_date(
                found["mid_datetime"], base_year=found["mid_datetime"].year)
        except (ValueError, TypeError):
            return found
    return found


# ---------------------------------------------------------------------------
# TimeAxis
# ---------------------------------------------------------------------------
@dataclass
class TimeAxis:
    """The time axis of a multi-epoch data set.

    Attributes
    ----------
    values : ndarray of datetime64
        Canonical epochs.  ``NaT`` marks an epoch whose date is unknown; a ``
        kind='index'`` axis is represented as ``NaT`` throughout.
    kind : {'datetime', 'index'}
        ``'datetime'`` means every epoch carries a real date; ``'index'`` is the
        honest degradation when dates are unavailable (then date-dependent
        operations raise rather than invent dates).
    labels, source : list, optional
        Original label (file name) and provenance per epoch.
    decimal_year_hint : ndarray, optional
        Explicit decimal years using the **frozen legacy convention**.  Stored
        rather than re-derived because the 3 cross-year epochs cannot be
        recovered from the date alone.
    meta : dict
        ``time_source`` (``header`` / ``filename`` / ``netcdf`` / ``legacy`` …),
        plus any conflicts found while parsing.
    """

    values: np.ndarray
    kind: str = "datetime"
    labels: Optional[list] = None
    source: Optional[list] = None
    decimal_year_hint: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- init
    def __post_init__(self):
        v = np.asarray(self.values)
        if v.dtype.kind == "M":
            self.values = v.astype("datetime64[s]")
        elif v.dtype.kind in "iuf":
            raise ValueError(
                "TimeAxis.values 必须是 datetime64；数值时间请用 "
                "from_decimal_years() / from_index() 明确语义")
        else:
            raise ValueError(f"无法把 {v.dtype} 当作时间轴")
        if self.decimal_year_hint is not None:
            self.decimal_year_hint = np.asarray(self.decimal_year_hint, dtype=float).ravel()
            if self.decimal_year_hint.size != self.values.size:
                raise ValueError("decimal_year_hint 长度与 values 不一致")
        if self.labels is not None and len(self.labels) != self.values.size:
            raise ValueError("labels 长度与 values 不一致")

    # ---------------------------------------------------------- accessors
    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        return (f"TimeAxis({self.kind}, n={len(self)}, "
                f"{self.meta.get('time_source', '?')})")

    @property
    def has_dates(self) -> bool:
        return self.kind == "datetime" and not bool(np.all(np.isnat(self.values)))

    def _require_dates(self, what: str) -> None:
        if not self.has_dates:
            raise ValueError(
                f"{what} 需要真实日期，但这条时间轴只有序号"
                f"（kind={self.kind!r}，meta.time_source={self.meta.get('time_source')!r}）。"
                "请先给出历元日期，不要用序号当时间。")

    @property
    def decimal_years(self) -> np.ndarray:
        """Legacy-convention decimal years (the format the old pipeline writes)."""
        if self.decimal_year_hint is not None:
            return np.asarray(self.decimal_year_hint, dtype=float).copy()
        self._require_dates("decimal_years")
        out = np.empty(len(self), dtype=float)
        dt = self.values.astype("datetime64[s]").astype(_dt.datetime)
        for i, t in enumerate(np.atleast_1d(dt)):
            out[i] = decimal_year_of_date(t)
        return out

    @property
    def dt_days(self) -> np.ndarray:
        """Gap to the previous epoch in days (first entry is ``nan``)."""
        self._require_dates("dt_days")
        d = np.diff(self.values).astype("timedelta64[s]").astype(float) / 86400.0
        return np.concatenate([[np.nan], d])

    @property
    def span(self) -> tuple:
        self._require_dates("span")
        return (self.values[0], self.values[-1])

    def is_regular(self, tol_days: float = 3.0) -> bool:
        """True when consecutive gaps stay within ``tol_days`` of the median."""
        d = self.dt_days[1:]
        if d.size == 0:
            return True
        med = float(np.median(d))
        return bool(np.all(np.abs(d - med) <= tol_days))

    def summary(self) -> str:
        lines = [f"历元数      : {len(self)}",
                 f"时间来源    : {self.meta.get('time_source', '未记录')}"]
        if self.has_dates:
            d = self.dt_days[1:]
            lines += [
                f"跨度        : {str(self.values[0])[:10]} … {str(self.values[-1])[:10]}",
                f"间隔 (天)   : 中位 {np.median(d) if d.size else float('nan'):.2f}"
                f"  最小 {d.min() if d.size else float('nan'):.2f}"
                f"  最大 {d.max() if d.size else float('nan'):.2f}",
                f"等间隔      : {'是' if self.is_regular() else '否（月度解本来就不等间隔）'}",
            ]
            miss = self.missing()
            if miss:
                lines.append(f"疑似缺测    : {len(miss)} 段，最长 "
                             f"{max(m[2] for m in miss):.0f} 天"
                             f"（例如 {str(miss[0][0])[:10]} → {str(miss[0][1])[:10]}）")
            dup = self.duplicates()
            if dup:
                lines.append(f"疑似重复    : {len(dup)} 组，例如 {dup[0]}")
        else:
            lines.append("跨度        : （只有序号，没有日期）")
        for k in ("warnings",):
            for w in self.meta.get(k, [])[:4]:
                lines.append(f"提示        : {w}")
        return "\n".join(lines)

    # ------------------------------------------------------------ queries
    def select(self, idx) -> "TimeAxis":
        idx = np.atleast_1d(idx)
        return TimeAxis(
            self.values[idx], self.kind,
            labels=None if self.labels is None else [self.labels[i] for i in idx],
            source=None if self.source is None else [self.source[i] for i in idx],
            decimal_year_hint=(None if self.decimal_year_hint is None
                               else self.decimal_year_hint[idx]),
            meta=dict(self.meta))

    def slice(self, t0=None, t1=None) -> tuple:
        """Return ``(mask, note)`` for the inclusive window ``[t0, t1]``."""
        self._require_dates("slice")
        lo = np.datetime64("1900-01-01") if t0 is None else np.datetime64(t0)
        hi = np.datetime64("2999-01-01") if t1 is None else np.datetime64(t1)
        mask = (self.values >= lo) & (self.values <= hi)
        return mask, f"{int(mask.sum())}/{len(self)} 个历元落在 [{lo}, {hi}]"

    def nearest(self, when) -> int:
        """Index of the epoch closest to ``when`` (ties go to the earlier one)."""
        self._require_dates("nearest")
        w = np.datetime64(when)
        return int(np.argmin(np.abs(self.values - w)))

    def missing(self, cadence_days: float = 31.0,
                tol_days: float = 6.0) -> list:
        """Gaps that look like a dropped epoch: ``[(before, after, gap_days)]``."""
        self._require_dates("missing")
        d = self.dt_days
        out = []
        for i in range(1, len(self)):
            if np.isfinite(d[i]) and d[i] > cadence_days + tol_days:
                out.append((self.values[i - 1], self.values[i], float(d[i])))
        return out

    def duplicates(self, tol_days: float = 1.0) -> list:
        """Groups of epochs closer together than ``tol_days``."""
        self._require_dates("duplicates")
        d = self.dt_days
        groups, cur = [], []
        for i in range(1, len(self)):
            if np.isfinite(d[i]) and d[i] < tol_days:
                if not cur:
                    cur = [i - 1]
                cur.append(i)
            elif cur:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        return groups

    # ------------------------------------------------------------- output
    def to_legacy_timeinfo(self, path: str) -> str:
        """Write the legacy ``[idx, y_start, y_end, y_mid]`` table.

        Column 0 is 1-based, matching the reference file.  ``y_start``/``y_end``
        come from the stored labels when they are GRACE file names (the only way
        to reproduce the reference exactly); otherwise the epoch day is used for
        all three.

        .. warning::
           The legacy format stores decimal years with 6 decimals, i.e. ~16 s
           resolution, so a write/read round trip is **not** exact at the second
           level.  Keep the canonical axis in ``series_nc`` (datetime64) and use
           this table for interop only.
        """
        rows = []
        for i in range(len(self)):
            ys = ye = None
            if self.labels is not None:
                try:
                    p = parse_grace_filename(self.labels[i])
                    ys, ye = p["start_date"], p["end_date"]
                except ValueError:
                    pass
            mid = self.values[i].astype("datetime64[s]").astype(_dt.datetime)
            ys = ys or mid.date()
            ye = ye or mid.date()
            rows.append([i + 1,
                         decimal_year_of_date(ys),
                         decimal_year_of_date(ye),
                         self.decimal_years[i]])
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write("%d %.6f %.6f %.6f \n" % tuple(r))
        return path

    @classmethod
    def from_legacy_timeinfo(cls, path: str) -> "TimeAxis":
        """Read the legacy ``[idx, y_start, y_end, y_mid]`` table."""
        arr = np.loadtxt(path)
        arr = np.atleast_2d(arr)
        mid = np.asarray(arr[:, 3], dtype=float)
        vals = np.array([datetime_of_decimal_year(v) for v in mid])
        return cls(vals.astype("datetime64[s]"), "datetime", decimal_year_hint=mid,
                   meta={"time_source": "legacy:TimeInfo"})

    def to_dict(self) -> dict:
        """JSON-friendly form (used in ``meta`` and npz sidecars)."""
        naive = self.values.astype("datetime64[s]").astype("int64")
        return {
            "kind": self.kind,
            "values_s": [int(v) for v in naive],
            "decimal_year_hint": (None if self.decimal_year_hint is None
                                  else [float(v) for v in self.decimal_year_hint]),
            "labels": list(self.labels) if self.labels else None,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TimeAxis":
        vals = np.array([np.datetime64(int(v), "s") for v in d["values_s"]])
        hint = d.get("decimal_year_hint")
        return cls(vals, d.get("kind", "datetime"), labels=d.get("labels"),
                   decimal_year_hint=None if hint is None else np.asarray(hint, float),
                   meta=dict(d.get("meta") or {}))

    # --------------------------------------------------------- constructors
    @classmethod
    def from_index(cls, n: int, **meta) -> "TimeAxis":
        """An honest "I only know the order" axis (no invented dates)."""
        v = np.full(int(n), np.datetime64("NaT"), dtype="datetime64[s]")
        return cls(v, "index", meta={"time_source": "index", **meta})

    @classmethod
    def from_datetimes(cls, values, **meta) -> "TimeAxis":
        return cls(np.asarray(values, dtype="datetime64[s]"), "datetime",
                   meta={"time_source": "datetime", **meta})

    @classmethod
    def from_decimal_years(cls, years, **meta) -> "TimeAxis":
        y = np.asarray(years, dtype=float).ravel()
        dts = np.array([datetime_of_decimal_year(v) for v in y])
        return cls(dts.astype("datetime64[s]"), "datetime",
                   decimal_year_hint=y,
                   meta={"time_source": "decimal_year", **meta})

    @classmethod
    def from_netcdf_coord(cls, coord, **meta) -> "TimeAxis":
        """From an xarray/netCDF ``time`` coordinate (datetime64 or numeric).

        Numeric coordinates are decoded through :meth:`from_numeric_time`, so the
        CF ``units`` attribute is what supplies the calendar.  Attribute names are
        looked up **case-insensitively**: real products are not all CF-clean —
        the CSR mascon files spell it ``Units`` (capital U), and their epoch
        values are half-days (the midpoint of the month), which a naive
        ``timedelta64[D]`` conversion truncates by 12 h.
        """
        v = np.asarray(coord.values if hasattr(coord, "values") else coord)
        name = getattr(coord, "name", "time")
        if v.dtype.kind == "M":
            return cls(v, "datetime", meta={"time_source": f"netcdf:{name}", **meta})
        attrs = dict(getattr(coord, "attrs", {}) or {})
        units = attr_ci(attrs, "units")
        calendar = attr_ci(attrs, "calendar")
        if v.dtype.kind in "iuf" and units:
            return cls.from_numeric_time(v, units, calendar=calendar,
                                         source=f"netcdf:{name}", **meta)
        # numeric but no usable units -> do NOT guess a calendar
        return cls.from_index(v.size, note=f"坐标 {name} 是数值且缺少 units，降级为序号")

    @classmethod
    def from_numeric_time(cls, values, units: str, calendar: Optional[str] = None,
                          source: str = "numeric", **meta) -> "TimeAxis":
        """Decode ``<n> <unit> since <epoch>`` values (CF / COARDS ``units``).

        Supported units: ``seconds``/``minutes``/``hours``/``days``.  Anything
        else -- notably ``months since`` / ``years since``, whose length is
        calendar-dependent -- is **refused** rather than approximated: a wrong
        month length silently moves every epoch, which is exactly the class of
        bug this module exists to prevent.  The result is an *index* axis with an
        explicit note in that case, so the caller can say why.

        Values are converted at **second** resolution, so the half-day midpoints
        that GRACE-style monthly products carry survive (``days since`` values
        like ``129.5`` are ordinary there).
        """
        v = np.asarray(values, dtype=float).ravel()
        txt = str(units).strip()
        m = re.match(r"^\s*([A-Za-z]+)\s+since\s+(.+?)\s*$", txt, re.I)
        if not m:
            return cls.from_index(v.size, note=f"读不懂的时间单位 {txt!r}，降级为序号")
        unit = m.group(1).lower()
        epoch = m.group(2).strip().rstrip("Zz").strip()   # 「…T00:00:00Z」→ 朴素时间
        per = {"seconds": 1.0, "second": 1.0, "secs": 1.0, "sec": 1.0,
               "minutes": 60.0, "minute": 60.0, "mins": 60.0, "min": 60.0,
               "hours": 3600.0, "hour": 3600.0, "hrs": 3600.0, "hr": 3600.0,
               "days": 86400.0, "day": 86400.0}.get(unit)
        if per is None:
            return cls.from_index(
                v.size,
                note=(f"时间单位 {unit!r} 的长度随日历变化（不是固定秒数），"
                      "拒绝近似；已降级为序号，请先把时间换算成 days/hours/seconds"))
        try:
            t0 = np.datetime64(epoch)
        except Exception:                                        # noqa: BLE001
            return cls.from_index(v.size,
                                  note=f"读不懂时间起点 {epoch!r}，降级为序号")
        if t0.dtype.kind != "M":
            return cls.from_index(v.size,
                                  note=f"时间起点 {epoch!r} 不是日期，降级为序号")
        secs = np.round(v * per).astype("int64")
        vals = t0.astype("datetime64[s]") + secs.astype("timedelta64[s]")
        return cls(vals, "datetime",
                   meta={"time_source": source, "units": txt,
                         "calendar": calendar, **meta})

    @classmethod
    def from_grace_filenames(cls, paths: Sequence[str], **meta) -> "TimeAxis":
        """Parse the epoch of each GRACE file from its **name**."""
        paths = list(paths)
        parsed = [parse_grace_filename(p) for p in paths]
        vals, decades = [], []
        for p in parsed:
            mid = p["mid_datetime"]
            vals.append(np.datetime64(mid))
            decades.append(p["decimal_year"])
        return cls(np.array(vals, dtype="datetime64[s]"), "datetime",
                   labels=[os.path.basename(p) for p in paths],
                   source=[str(p) for p in paths],
                   decimal_year_hint=np.asarray(decades, float),
                   meta={"time_source": "filename", **meta})

    @classmethod
    def from_gfc_headers(cls, paths: Sequence[str], **meta) -> "TimeAxis":
        """Parse the epoch of each GRACE file from its **gfc header**.

        Raises when any file lacks usable header time information -- a partially
        dated series is worse than an obvious failure.
        """
        paths = list(paths)
        vals, decades, labels = [], [], []
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"from_gfc_headers 需要能打开的**完整路径**，但找不到 {p!r}"
                    "（如果手上只有文件名，请传 os.path.join(目录, 文件名)，"
                    "或改用 epoch_from='filename'）")
            h = parse_gfc_time_header(p)
            if not h or "mid_date" not in h:
                raise ValueError(
                    f"{os.path.basename(p)}: 头部没有可用的时间信息"
                    "（time_coverage_* / time_period_of_data）；"
                    "请改用 epoch_from='filename' 或显式给出时间轴")
            mid = h.get("mid_datetime")
            if mid is None:
                # coverage-window fallback: only a day is known
                mid = _dt.datetime.combine(h["mid_date"], _dt.time())
                if "decimal_year" not in h:
                    h["decimal_year"] = decimal_year_of_date(h["mid_date"])
            vals.append(np.datetime64(mid))
            decades.append(float(h["decimal_year"]))
            labels.append(os.path.basename(p))
        return cls(np.array(vals, dtype="datetime64[s]"), "datetime",
                   labels=labels, source=[str(p) for p in paths],
                   decimal_year_hint=np.asarray(decades, float),
                   meta={"time_source": "header", **meta})

    @classmethod
    def from_grace(cls, paths: Sequence[str], *, prefer: str = "auto",
                   **meta) -> "TimeAxis":
        """Epochs from file names and/or headers, with conflicts reported.

        ``prefer``: ``'header'`` (authoritative), ``'filename'``, or ``'auto'``
        (header when *every* file has one, else filename).  When both sources are
        available their mid-days are compared and any disagreement is recorded in
        ``meta['warnings']`` rather than silently resolved.
        """
        paths = list(paths)
        warns: list = []
        by_name = None
        try:
            by_name = cls.from_grace_filenames(paths)
        except ValueError as exc:
            warns.append(f"文件名解析失败，回退到头部：{exc}")
        by_head = None
        if prefer in ("auto", "header"):
            try:
                by_head = cls.from_gfc_headers(paths)
            except (ValueError, OSError) as exc:
                if prefer == "header":
                    raise
                warns.append(f"头部解析失败，回退到文件名：{exc}")
        if prefer == "filename" or (prefer == "auto" and by_head is None):
            if by_name is None:
                raise ValueError("文件名与头部都无法解析历元时间")
            out = by_name
        else:
            out = by_head if by_head is not None else by_name
        if by_name is not None and by_head is not None:
            d = np.abs((by_name.values - by_head.values)
                       .astype("timedelta64[s]").astype(float)) / 86400.0
            bad = int(np.sum(d > 1.0))
            if bad:
                warns.append(
                    f"{bad}/{len(paths)} 个历元的**文件名与 gfc 头部时间相差 >1 天**；"
                    f"已按 prefer={prefer!r} 取 {out.meta.get('time_source')}，"
                    "请人工核对（例如 " +
                    ", ".join(os.path.basename(paths[i]) for i in np.nonzero(d > 1.0)[0][:3]) + "）")
        out = out.select(range(len(out)))
        out.meta["warnings"] = list(out.meta.get("warnings", [])) + warns
        return out


def sort_and_check(paths: Iterable[str], times: TimeAxis) -> tuple:
    """Reorder files into time order and report duplicates.

    Returns ``(order, warnings)`` where ``order`` is the index permutation.
    """
    order = np.argsort(times.values, kind="stable")
    warns = []
    if not np.all(np.diff(order) == 1):
        warns.append("文件顺序与时间顺序不一致，已按时间重排")
    dup = times.select(order).duplicates()
    if dup:
        warns.append(f"时间轴上发现 {len(dup)} 组重复/过近历元（同一天两个文件？）："
                     + "; ".join(str(g) for g in dup[:3]))
    return order, warns
