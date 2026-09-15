# -*- coding: utf-8 -*-
"""
shkit.timeseries
================

**Time-domain operators** on a multi-epoch coefficient series: mean removal,
polynomial trend, annual/semiannual harmonics, and time filtering.

Why this is not "just an FFT along the time axis"
-------------------------------------------------
GRACE/GRACE-FO monthly solutions are **27-35 days apart, with gaps and two
solutions in some months**.  Anything that assumes a uniform cadence (an FFT, a
moving average counted in "months") silently mixes frequencies.  So every
operator here works on the **true elapsed time** of each epoch and is either a
*least-squares* fit or an *explicitly weighted* kernel:

* the fit coordinate is ``t = (values - values[0]) / 365.25`` **years since the
  first epoch** -- not the legacy decimal year.  Decimal years are a label and
  interchange format (see :mod:`shkit.timeaxis`): they carry the leap-year rule
  and a ~2e-5 residual on cross-year epochs, neither of which belongs inside a
  fit;
* a missing epoch (all-NaN, or a ``mask``) is **excluded and reported**, never
  zero-filled -- zero is a perfectly good datum to a least-squares fit and would
  drag the trend;
* an axis without dates (``kind='index'``) makes date-dependent operators
  **raise**, instead of pretending the epochs are equally spaced.  A plain
  unweighted mean does not need dates and is therefore still allowed.

Coefficient domain vs grid domain
---------------------------------
Fitting is a linear combination **over epochs** with the same weights in every
spatial slot, and synthesis is linear **per epoch**, so the two commute:
:func:`fit_time_model` returns exactly the same field whether it is applied
before or after :func:`shkit.synthesis.synthesis_grid`.  That is asserted at
machine precision in ``tests/validate_timeseries.py``, and it is the property that
makes it legitimate to fit 3721 coefficients instead of 65 160 grid points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .coeffs import SHCoeffs
from .timeaxis import TimeAxis

__all__ = [
    "TimeFit",
    "series_coord",
    "design_time",
    "poly_terms",
    "seasonal_terms",
    "fit_time_model",
    "time_mean",
    "remove_time_mean",
    "detrend",
    "deseasonalize",
    "trend_field",
    "time_gaussian_filter",
    "TimeFilterReport",
    # C2 products
    "series_at_points",
    "series_grid",
    "basin_average",
    "match_epochs",
    # D7 helper
    "fit_series_maps",
]

_DAYS_PER_YEAR = 365.25
_SEC_PER_DAY = 86400.0

#: Named periods, so ``periods=(1.0, 0.5)`` produces readable column names.
_PERIOD_NAMES = {1.0: "annual", 0.5: "semiannual"}
#: Public alias —— GUI 要用它把周期写成「周年 / 半年」而不是裸数字。
PERIOD_NAMES = _PERIOD_NAMES


# ---------------------------------------------------------------------------
# time coordinate
# ---------------------------------------------------------------------------
def series_coord(coeffs: SHCoeffs, *, require: bool = True):
    """``(t, t_ref)``: elapsed **years from the middle of the span**, and that instant.

    Two deliberate choices:

    * **True elapsed time / 365.25**, not the legacy decimal year.  Decimal years
      are a label and interchange format (see :mod:`shkit.timeaxis`): they carry
      the leap-year rule and a ~2e-5 residual on cross-year epochs, neither of
      which belongs inside a fit.
    * **Centred** on the middle of the span, not on the first epoch.  A 20-year
      series spans ``t in [-10, 10]`` instead of ``[0, 20]``, which cuts the
      ``const``/``trend`` collinearity (condition number 15 -> 4 on a pure
      ``[1, t]`` design) and makes ``const`` the level at mid-span -- i.e. close
      to the series mean -- rather than "value at whatever the first epoch was".
      The seasonal phases are unaffected: ``cos(2*pi*(t-c))`` is a rotation of the
      ``cos``/``sin`` pair, so the fitted function space is identical.

    ``t_ref`` is the ``datetime64`` origin, so a fitted model can be evaluated at
    real dates.  Raises when the series has no usable time axis; ``require=False``
    returns ``(None, None)`` instead (for "is this available?" branches).
    """
    if not coeffs.has_time:
        if not require:
            return None, None
        raise ValueError(
            "这条系数序列没有可用的时间轴（times 缺失或长度与 ntime 不符）。"
            "时间域拟合需要每个历元的真实日期：请用 SHCoeffs.with_times(TimeAxis…) "
            "挂上时间轴；若手上只有历元序号，请先用 TimeAxis.from_decimal_years() "
            "把语义写清楚，不要用序号当时间。")
    ax = coeffs.times
    ax._require_dates("时间域拟合")
    v = np.asarray(ax.values, dtype="datetime64[s]")
    if np.any(np.isnat(v)):
        raise ValueError("时间轴上有 NaT（日期未知的历元），无法做时间域拟合；"
                         "请先剔除或补上这些历元。")
    mid = v[0] + (v[-1] - v[0]) / 2
    dt_days = (v - mid).astype("timedelta64[s]").astype(float) / _SEC_PER_DAY
    return dt_days / _DAYS_PER_YEAR, mid


def _coord_of(when, t0) -> np.ndarray:
    v = np.atleast_1d(np.asarray(
        when.values if hasattr(when, "values") else when, dtype="datetime64[s]"))
    return ((v - np.datetime64(t0)).astype("timedelta64[s]").astype(float)
            / (_SEC_PER_DAY * _DAYS_PER_YEAR))


def _irregular(t, tol_days: float = 3.0) -> bool:
    d = np.diff(np.asarray(t, dtype=float)) * _DAYS_PER_YEAR
    if d.size == 0:
        return False
    return bool(np.max(np.abs(d - np.median(d))) > tol_days)


# ---------------------------------------------------------------------------
# design matrix
# ---------------------------------------------------------------------------
def poly_terms(t, order: int = 1) -> tuple:
    """``(names, columns)`` for a polynomial of ``order`` in ``t``.

    ``order=0`` is the constant, ``1`` adds the linear trend, ``2`` the quadratic.
    """
    t = np.asarray(t, dtype=float).ravel()
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")
    names = ["const" if k == 0 else ("trend" if k == 1 else f"poly{k}")
             for k in range(order + 1)]
    return names, [t ** k for k in range(order + 1)]


def seasonal_terms(t, periods: Sequence[float] = (1.0,),
                   names: Optional[Sequence[str]] = None) -> tuple:
    """``(names, columns)`` for ``cos/sin`` pairs at the given periods (years).

    ``periods=(1.0,)`` is the annual cycle, ``(1.0, 0.5)`` adds the semiannual.
    The phase is :math:`2\\pi t/\\text{period}` in *true elapsed years*, so an
    irregularly sampled year is handled correctly.
    """
    t = np.asarray(t, dtype=float).ravel()
    out_names, cols = [], []
    for i, p in enumerate(periods):
        p = float(p)
        if not np.isfinite(p) or p <= 0:
            raise ValueError(f"period must be positive and finite, got {p!r}")
        tag = (names[i] if names is not None and i < len(names)
               else _PERIOD_NAMES.get(p, f"period{p:g}"))
        ph = 2.0 * np.pi * t / p
        out_names += [f"{tag}_cos", f"{tag}_sin"]
        cols += [np.cos(ph), np.sin(ph)]
    return out_names, cols


def design_time(t, *, poly_order: int = 0, periods: Sequence[float] = (),
                extra: Optional[tuple] = None) -> tuple:
    """Assemble ``(names, A)`` with ``A`` of shape ``(ntime, p)``.

    ``extra`` is ``(names, matrix)`` for anything not covered above -- a step
    function for a known jump, an external index, a GIA model.  Note that a
    custom column cannot be re-evaluated by :meth:`TimeFit.evaluate`, which says
    so rather than guessing.
    """
    t = np.asarray(t, dtype=float).ravel()
    names, cols = [], []
    if poly_order >= 0:
        n0, c0 = poly_terms(t, poly_order)
        names += n0
        cols += c0
    if len(periods):
        n1, c1 = seasonal_terms(t, periods)
        names += n1
        cols += c1
    if extra is not None:
        xn, xm = extra
        xm = np.asarray(xm, dtype=float)
        if xm.ndim == 1:
            xm = xm[:, None]
        if xm.shape[0] != t.size:
            raise ValueError(f"extra 项有 {xm.shape[0]} 行，但历元是 {t.size} 个")
        if len(xn) != xm.shape[1]:
            raise ValueError(f"extra 项名字 {len(xn)} 个与列数 {xm.shape[1]} 不符")
        names += list(xn)
        cols += [xm[:, j] for j in range(xm.shape[1])]
    if not cols:
        raise ValueError("至少要有一项：给 poly_order>=0 或 periods/extra")
    return names, np.column_stack(cols)


# ---------------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------------
def _flatten(coeffs: SHCoeffs):
    """``(Y, shape)`` with ``Y`` of shape ``(ntime, 2*ncoef)`` (C | S).

    Concatenation is along **axis 1**, so ``Y``'s first axis is time -- the shape
    the design matrix multiplies.  (``np.column_stack`` of two ``(ncoef, ntime)``
    blocks would silently give ``(ncoef, 2*ntime)`` instead; that mistake shows up
    as an ``IndexError`` only once an epoch index exceeds ``ncoef``.)
    """
    C, S = np.asarray(coeffs.C, float), np.asarray(coeffs.S, float)
    if C.ndim != 3:
        raise ValueError("时间域运算需要多历元系数，形状 (L+1, L+1, ntime)")
    ntime = C.shape[2]
    Y = np.concatenate([C.reshape(-1, ntime).T, S.reshape(-1, ntime).T], axis=1)
    return Y, C.shape


def _unflatten(Y: np.ndarray, shape, meta: dict) -> SHCoeffs:
    """Inverse of :func:`_flatten` (keeps a length-1 time axis)."""
    L = shape[0]
    n = L * L
    nt = Y.shape[0]
    C = Y[:, :n].T.reshape(L, L, nt)
    S = Y[:, n:].T.reshape(L, L, nt)
    if nt == 1:
        return SHCoeffs(C[:, :, 0], S[:, :, 0], dict(meta))
    return SHCoeffs(C, S, dict(meta))


def _missing_epochs(coeffs: SHCoeffs) -> np.ndarray:
    """Epochs with no data: every **C** coefficient at that epoch is non-finite.

    Only ``C`` is consulted, deliberately.  ``S[:, 0]`` is identically zero by
    convention and many other ``S`` entries are legitimately finite zeros, so
    requiring *both* halves to be non-finite would never flag anything: a
    NaN-filled epoch would slip through and blow up much later inside the SVD.
    """
    C = np.asarray(coeffs.C, float)
    if C.ndim != 3:
        return np.array([], dtype=int)
    flat = C.reshape(-1, C.shape[2])
    return np.nonzero(~np.isfinite(flat).any(axis=0))[0]


def _with_times_like(co: SHCoeffs, src: SHCoeffs) -> SHCoeffs:
    if src.times is not None and len(src.times) == co.ntime:
        return co.with_times(src.times)
    return co


def _binop(a: SHCoeffs, b: SHCoeffs, fn) -> SHCoeffs:
    """``fn`` on two coefficient sets, broadcasting a single-epoch ``b`` over time.

    ``a`` is a series ``(L+1, L+1, ntime)`` and ``b`` is often a *single* set (a
    time mean, a trend), so ``b`` needs an explicit trailing axis -- without it
    numpy broadcasts ``(L+1, L+1, ntime) - (L+1, L+1)`` against the *first* two
    axes and either raises or, worse, silently produces something else.
    """
    def _fix(x):
        x = np.asarray(x, dtype=float)
        return x[:, :, None] if (np.ndim(a.C) == 3 and x.ndim == 2) else x

    out = SHCoeffs(fn(_fix(a.C), _fix(b.C)), fn(_fix(a.S), _fix(b.S)),
                   dict(a.meta))
    return _with_times_like(out, a)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
@dataclass
class TimeFit:
    """A least-squares time model of a coefficient series.

    Attributes
    ----------
    names : list of str
        One per column of the design matrix.
    terms : list of SHCoeffs
        Fitted coefficient set per column (``terms[1]`` is usually the trend, in
        *coefficient units per year*).
    model, residual : SHCoeffs
        Model at the fitted epochs, and ``data - model`` (NaN where excluded).
    coord, t0 : ndarray, datetime64
        Fit coordinate (years since ``t0``) and its origin.
    excluded : ndarray of int
        Epochs left out of the fit (missing data or ``mask``).
    rms_per_epoch : ndarray
        RMS over all coefficients of the residual at each epoch.
    """

    names: list
    terms: list
    model: SHCoeffs
    residual: SHCoeffs
    coord: np.ndarray
    t0: np.datetime64
    weights: Optional[np.ndarray] = None
    excluded: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    rms_per_epoch: np.ndarray = field(default_factory=lambda: np.array([]))
    design: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- queries
    @property
    def n_used(self) -> int:
        return int(self.coord.size - self.excluded.size)

    @property
    def cond(self) -> float:
        """Condition number of the (weighted) design matrix.

        Reported, not hidden: a trend plus an annual term fitted to a one-year
        series are nearly collinear and the fit will happily return huge,
        meaningless amplitudes.  A large value is the honest warning that the
        series is too short for the model asked for.
        """
        return float(self.meta.get("cond", np.nan))

    @property
    def rank(self) -> int:
        return int(self.meta.get("rank", 0))

    def term(self, name: str) -> SHCoeffs:
        if name not in self.names:
            raise KeyError(f"没有名为 {name!r} 的项；现有 {self.names}")
        return self.terms[self.names.index(name)]

    @property
    def trend(self) -> Optional[SHCoeffs]:
        """The linear term if the model has one, else ``None``."""
        return self.term("trend") if "trend" in self.names else None

    def amplitude_phase(self, tag: str = "annual", period: Optional[float] = None):
        """``(amplitude, doy_of_max, sigma)`` of a seasonal ``cos/sin`` pair.

        ``amplitude = hypot(a, b)``; the phase is reported as the **day of year
        of the maximum** (0-365.25), which is what a reader can compare with a
        water-balance seasonality.  ``sigma = amplitude/sqrt(2)`` is the pair's
        RMS contribution, directly comparable with the residual RMS.
        """
        a, b = self.term(f"{tag}_cos"), self.term(f"{tag}_sin")
        per = float(period if period is not None
                    else self.meta.get("periods", {}).get(tag, 1.0))
        amp = SHCoeffs(np.hypot(a.C, b.C), np.hypot(a.S, b.S), dict(a.meta))
        phi = np.arctan2(b.C, a.C)
        doy = np.mod(np.degrees(phi) / 360.0 * per, per) * _DAYS_PER_YEAR
        return amp, doy, SHCoeffs(amp.C / np.sqrt(2.0), amp.S / np.sqrt(2.0),
                                  dict(a.meta))

    # ------------------------------------------------------------ evaluate
    def _columns(self, t) -> np.ndarray:
        """Design columns at arbitrary ``t``, from the fit's own recipe."""
        t = np.asarray(t, dtype=float).ravel()
        cols = []
        for nm in self.names:
            if nm == "const":
                cols.append(np.ones_like(t))
            elif nm == "trend":
                cols.append(t)
            elif nm.startswith("poly") and nm[4:].isdigit():
                cols.append(t ** int(nm[4:]))
            elif nm.endswith("_cos") or nm.endswith("_sin"):
                tag = nm.rsplit("_", 1)[0]
                per = float(self.meta["periods"][tag])
                ph = 2.0 * np.pi * t / per
                cols.append(np.cos(ph) if nm.endswith("_cos") else np.sin(ph))
            else:
                raise ValueError(
                    f"项 {nm!r} 不是可解析的内置项（自定义 extra 列无法在拟合之外"
                    "重新求值）。请用 terms[...] 自行组合，或改用内置项。")
        return np.column_stack(cols)

    def evaluate(self, when) -> SHCoeffs:
        """Evaluate the fitted model at real dates.

        ``when`` is anything ``np.datetime64`` accepts (date strings, a
        ``datetime64`` array, or a :class:`~shkit.timeaxis.TimeAxis`).  Dates
        outside the fitted span are extrapolation -- allowed, but flagged in
        ``meta['extrapolated']``.
        """
        v = np.atleast_1d(np.asarray(
            when.values if hasattr(when, "values") else when,
            dtype="datetime64[s]"))
        t = ((v - np.datetime64(self.t0)).astype("timedelta64[s]").astype(float)
             / (_SEC_PER_DAY * _DAYS_PER_YEAR))
        A = self._columns(t)
        Cc = np.stack([tm.C for tm in self.terms])            # (p, L, L)
        Sc = np.stack([tm.S for tm in self.terms])
        C = np.moveaxis(np.tensordot(A, Cc, axes=([1], [0])), 0, -1)
        S = np.moveaxis(np.tensordot(A, Sc, axes=([1], [0])), 0, -1)
        if v.size == 1:
            C, S = C[:, :, 0], S[:, :, 0]
        out = SHCoeffs(C, S, dict(self.model.meta))
        out = out.with_times(TimeAxis.from_datetimes(v))
        out.meta["extrapolated"] = bool(
            np.any(t < self.coord.min() - 1e-9)
            or np.any(t > self.coord.max() + 1e-9))
        return out

    # -------------------------------------------------------------- report
    def summary(self) -> str:
        ar = float(self.meta.get("anomaly_rms", np.nan))
        dr = float(self.meta.get("data_rms", np.nan))
        lines = [
            f"时间域拟合 : {len(self.names)} 项  {', '.join(self.names)}",
            f"历元       : 用 {self.n_used}/{self.coord.size}"
            + (f"（排除 {self.excluded.size} 个：{list(self.excluded[:6])}）"
               if self.excluded.size else ""),
            f"设计矩阵   : 条件数 {self.cond:.3g}，秩 {self.rank}/{len(self.names)}",
            f"残差 RMS   : 中位 {float(np.nanmedian(self.rms_per_epoch)):.6g}"
            f"  最大 {float(np.nanmax(self.rms_per_epoch)):.6g}",
            # The raw field RMS is dominated by the static part, so it is quoted
            # for reference only -- the reduction is measured against the anomaly.
            f"异常 RMS   : {ar:.6g}（去时间均值后）"
            f"  → 模型解释了 {float(self.meta.get('rms_reduction', np.nan)) * 100:.1f}%",
            f"参考       : 原始场 RMS {dr:.6g}"
            f"（含静态部分，仅作参照；对它谈降幅没有意义）",
        ]
        if self.rank < len(self.names):
            lines.append(
                "⚠ 设计矩阵**秩亏**：这些项在历元分布上不可分辨（例如序列太短却"
                "同时拟合趋势与年周期）。振幅不可信，请减少项数。")
        elif self.cond > 100.0:
            lines.append(f"⚠ 条件数 {self.cond:.3g} 偏大：项之间接近共线，"
                         "振幅会被放大，解释要谨慎。")
        for w in self.meta.get("warnings", []):
            lines.append(f"提示       : {w}")
        return "\n".join(lines)


def fit_time_model(coeffs: SHCoeffs, *, poly_order: int = 1,
                   periods: Sequence[float] = (), extra: Optional[tuple] = None,
                   weights=None, mask=None) -> TimeFit:
    """Least-squares fit of ``data ≈ sum_k A[:, k] * term_k`` over epochs.

    Parameters
    ----------
    coeffs : SHCoeffs
        Multi-epoch set with a usable :class:`~shkit.timeaxis.TimeAxis`.
    poly_order : int
        ``0`` constant only, ``1`` adds a linear trend (default), ``2`` quadratic.
    periods : sequence of float
        Seasonal periods in years; ``(1.0,)`` annual, ``(1.0, 0.5)`` semiannual.
    extra : (names, matrix), optional
        Additional design columns.
    weights : array_like, optional
        Per-epoch weights ``w_i``; the solver minimises ``sum w_i r_i^2``.
    mask : array_like of bool, optional
        Epochs to **exclude**.  All-NaN epochs are excluded automatically; both
        are reported in :attr:`TimeFit.excluded`.

    Returns
    -------
    TimeFit
        Carries ``cond``/``rank`` so a model the data cannot support is visible
        instead of producing confident nonsense.

    Notes
    -----
    A single ``lstsq`` solves **all** ``(n,m)`` slots at once -- design matrix
    ``(ntime, p)`` against ``(ntime, 2*ncoef)`` -- so the cost does not grow with
    the number of grid points the series came from.
    """
    t, t0 = series_coord(coeffs)
    ntime = coeffs.ntime
    if ntime < 2:
        raise ValueError("至少要 2 个历元才能谈趋势/周期")

    bad = _missing_epochs(coeffs)
    if mask is not None:
        mk = np.asarray(mask, dtype=bool).ravel()
        if mk.size != ntime:
            raise ValueError(f"mask 有 {mk.size} 个，但历元是 {ntime} 个")
        bad = np.union1d(bad, np.nonzero(~mk)[0])
    keep = np.setdiff1d(np.arange(ntime), bad)

    names, A = design_time(t[keep], poly_order=poly_order, periods=periods,
                           extra=extra)
    if keep.size < len(names):
        raise ValueError(
            f"只有 {keep.size} 个可用历元，却要拟合 {len(names)} 项"
            f"（{', '.join(names)}）—— 方程比未知数还少。请减少项数或补历元。")

    Y, shape = _flatten(coeffs)
    Yk = Y[keep, :]
    # An epoch flagged missing clears every C value, but stray non-finite S
    # entries (or a *partially* NaN epoch) would otherwise reach the SVD and fail
    # with "SVD did not converge" -- a message that says nothing about the data.
    if not np.all(np.isfinite(Yk)):
        bad_cols = np.nonzero(~np.isfinite(Yk).all(axis=0))[0]
        raise ValueError(
            f"参与拟合的数据里仍有 {bad_cols.size} 个非有限值（列 {list(bad_cols[:6])}"
            "…，C 与 S 各占一半列）。缺数据的历元要求**整段 C 全为 NaN** 才会被"
            "自动排除；只有一部分是 NaN 时请显式给 mask，或先修好数据。")
    w = None if weights is None else np.asarray(weights, float).ravel()
    if w is not None:
        if w.size != ntime:
            raise ValueError(f"weights 有 {w.size} 个，但历元是 {ntime} 个")
        if not np.all(np.isfinite(w[keep])) or np.any(w[keep] < 0):
            raise ValueError("weights 必须是非负有限值")
        sw = np.sqrt(w[keep])[:, None]
        Aw, Yw = A * sw, Yk * sw
    else:
        Aw, Yw = A, Yk

    sol, _res, rank, sv = np.linalg.lstsq(Aw, Yw, rcond=None)

    model = np.full_like(Y, np.nan)
    model[keep, :] = A @ sol
    resid = Y - model

    meta = dict(coeffs.meta)
    terms = [_unflatten(sol[j][None, :], shape, meta) for j in range(len(names))]
    model_co = _with_times_like(_unflatten(model, shape, meta), coeffs)
    res_co = _with_times_like(_unflatten(resid, shape, meta), coeffs)

    with np.errstate(invalid="ignore"):
        rms = np.sqrt(np.nanmean(resid ** 2, axis=1))
    # Two different denominators, and only one of them is meaningful.
    #
    # For a coefficient series the *static* field dominates: on the real
    # 216-epoch CSR series the raw RMS is 1.2e-2 while the month-to-month
    # variation is ~1e-10, so "explains 100% of the RMS" against the raw data is
    # vacuous -- any model with a constant term scores ~100%.  The honest
    # denominator is the **anomaly** RMS (data minus its time mean): that is the
    # part a time model is supposed to explain.  Both are reported.
    data_rms = float(np.sqrt(np.nanmean(Y[keep, :] ** 2)))
    fit_rms = float(np.sqrt(np.nanmean(resid[keep, :] ** 2)))
    anom = Y[keep, :] - np.mean(Y[keep, :], axis=0, keepdims=True)
    anomaly_rms = float(np.sqrt(np.nanmean(anom ** 2)))

    per_map = {_PERIOD_NAMES.get(float(p), f"period{float(p):g}"): float(p)
               for p in periods}
    tf = TimeFit(
        names=names, terms=terms, model=model_co, residual=res_co,
        coord=t, t0=t0, weights=w, excluded=bad, rms_per_epoch=rms, design=A,
        meta={"cond": float(np.linalg.cond(Aw)), "rank": int(rank),
              "singular_values": [float(s) for s in np.atleast_1d(sv)],
              "periods": per_map, "n_used": int(keep.size),
              "data_rms": data_rms, "fit_rms": fit_rms,
              "anomaly_rms": anomaly_rms,
              "rms_reduction": ((1.0 - fit_rms / anomaly_rms)
                                if anomaly_rms > 0 else np.nan),
              "rms_reduction_raw": ((1.0 - fit_rms / data_rms)
                                    if data_rms > 0 else np.nan),
              "coord_units": "years since t0",
              "warnings": list(coeffs.meta.get("warnings", []))})
    if rank < len(names):
        tf.meta["warnings"].append(
            f"设计矩阵秩亏（{rank}/{len(names)}）：项之间不可分辨，振幅不可信")
    if bad.size:
        tf.meta["warnings"].append(
            f"{bad.size} 个历元被排除出拟合（缺数据或 mask）：{list(bad[:8])}")
    if _irregular(t[keep]):
        tf.meta["warnings"].append(
            "历元不等间隔：这里的拟合按真实经过时间做，所以仍然正确，"
            "但有效自由度低于名义历元数，振幅的置信区间会比等间隔情形宽。")
    return tf


# ---------------------------------------------------------------------------
# convenience operators
# ---------------------------------------------------------------------------
def time_mean(coeffs: SHCoeffs, *, weights=None) -> SHCoeffs:
    """Mean over epochs (missing epochs excluded and counted).

    A plain mean of monthly solutions is a mean of *solutions*, not of *time*:
    two solutions in one month get double weight and a gap counts for nothing.
    Pass ``weights`` (e.g. each epoch's span in days) when that matters -- the
    report says which was done, and an irregular cadence without weights is
    flagged rather than silently accepted.
    """
    Y, shape = _flatten(coeffs)
    ntime = Y.shape[0]
    ok = np.ones(ntime, dtype=bool)
    bad = _missing_epochs(coeffs)
    ok[bad] = False
    if not ok.any():
        raise ValueError("所有历元都是缺数据，无法求时间平均")
    if weights is None:
        w = np.ones(int(ok.sum()))
        weighted = False
    else:
        wu = np.asarray(weights, float).ravel()
        if wu.size != ntime:
            raise ValueError(f"weights 有 {wu.size} 个，但历元是 {ntime} 个")
        w = wu[ok]
        weighted = True
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("权重之和必须为正")
    mean = _unflatten((Y[ok, :] * (w / s)[:, None]).sum(axis=0)[None, :],
                      shape, coeffs.meta)
    mean.meta["time_mean_n_used"] = int(ok.sum())
    mean.meta["time_mean_weighted"] = weighted
    if bad.size:
        mean.meta["time_mean_excluded"] = [int(i) for i in bad]
    if not weighted and coeffs.has_time:
        try:
            if not coeffs.times.is_regular():
                mean.meta.setdefault("warnings", []).append(
                    "历元不等间隔却用了**未加权**平均：历元多的月份被重复计入，"
                    "缺测段则完全没有权重。要按时间平均请传 weights。")
        except ValueError:
            pass
    return mean


def remove_time_mean(coeffs: SHCoeffs, *, weights=None) -> tuple:
    """``(anomaly, mean)`` -- the series minus its time mean."""
    mean = time_mean(coeffs, weights=weights)
    return _binop(coeffs, mean, lambda a, b: a - b), mean


def detrend(coeffs: SHCoeffs, **kw) -> tuple:
    """``(detrended, trend)``, with the trend in coefficient units **per year**."""
    fit = fit_time_model(coeffs, poly_order=1, **kw)
    return fit.residual, fit.term("trend")


def deseasonalize(coeffs: SHCoeffs, periods=(1.0,), **kw) -> tuple:
    """``(deseasonalized, fit)`` removing the mean and the given harmonics."""
    fit = fit_time_model(coeffs, poly_order=0, periods=periods, **kw)
    return fit.residual, fit


def trend_field(coeffs: SHCoeffs, **kw) -> SHCoeffs:
    """Per-year trend as a single-epoch ``SHCoeffs`` (the C2 product form)."""
    return fit_time_model(coeffs, poly_order=1, **kw).term("trend")


# ---------------------------------------------------------------------------
# C2: series products (points / basins / field cubes)
# ---------------------------------------------------------------------------
def series_at_points(coeffs: SHCoeffs, lat, lon, *, nmax: Optional[int] = None,
                     target_unit: Optional[str] = None,
                     gaussian_km: float = 0.0, love_numbers=None,
                     love_numbers_h=None, allow_unit_mismatch: bool = False,
                     chunk: int = 200_000):
    """Time series of the field at given points.

    Returns ``(values, times)`` with ``values`` shaped ``(npoints, ntime)`` --
    one row per point, one column per epoch -- and ``times`` the series'
    :class:`~shkit.timeaxis.TimeAxis` (``None`` if the series has none).

    ``lat``/``lon`` may have **any matching shape** (a 1-D point list, or the 2-D
    meshgrid of a regular grid); they are flattened, row-major, to give the rows of
    ``values``.

    ``target_unit`` is the **inverse** conversion of :mod:`shkit.units`
    (``'ewh'``, ``'geoid'``, ``'radial_displacement'`` ...); it is applied to the
    coefficients once, and the guard rails are the same as
    :func:`shkit.synthesis.synthesis` (a ``'scalar'`` series cannot be turned into
    EWH, and the call raises rather than guessing).

    ⚠️ **A GRACE GSM series is not itself an anomaly.**  Its ``C00`` is exactly 1
    (the total mass of the Earth), so a plain ``target_unit='ewh'`` synthesis of a
    raw GSM series returns ~1.2e7 m, not water heights.  Subtract a time mean first
    (:func:`remove_time_mean`, or the CLI's ``--remove-time-mean``), which is what
    "EWH anomaly relative to the temporal mean" means in the reference products.
    """
    from .synthesis import synthesis

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape:
        raise ValueError(f"lat 形状 {lat.shape} 与 lon 形状 {lon.shape} 不一致")
    lat = lat.ravel()
    lon = lon.ravel()
    vals = synthesis(lat, lon, coeffs, nmax=nmax, gaussian_km=gaussian_km,
                     target_unit=target_unit, love_numbers=love_numbers,
                     love_numbers_h=love_numbers_h, chunk=chunk,
                     allow_unit_mismatch=allow_unit_mismatch)
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    return vals, coeffs.times


def series_grid(coeffs: SHCoeffs, lat_vec, lon_vec, *, nmax: Optional[int] = None,
                target_unit: Optional[str] = None, gaussian_km: float = 0.0,
                love_numbers=None, chunk: int = 200_000,
                allow_unit_mismatch: bool = False,
                longitude_fft: str = "auto", report: Optional[dict] = None):
    """Field **cube** of the series on a regular grid.

    Returns ``(cube, times)`` with ``cube`` shaped ``(nlat, nlon, ntime)`` in
    SHKit's internal convention (``lat`` ascending, ``lon`` as given).  Write it
    with :func:`shkit.io.write_field_series` to get the ``(time, lat, lon)``
    reference layout.
    """
    from .synthesis import synthesis_grid

    cube = synthesis_grid(lat_vec, lon_vec, coeffs, nmax=nmax,
                          gaussian_km=gaussian_km, target_unit=target_unit,
                          love_numbers=love_numbers, chunk=chunk,
                          allow_unit_mismatch=allow_unit_mismatch,
                          longitude_fft=longitude_fft, report=report)
    cube = np.asarray(cube, dtype=float)
    if cube.ndim == 2:
        cube = cube[:, :, None]
    return cube, coeffs.times


def basin_average(coeffs: SHCoeffs, lat, lon, mask=None, *, weights=None,
                  rule: str = "auto", normalise: str = "auto",
                  target_unit: Optional[str] = None, gaussian_km: float = 0.0,
                  nmax: Optional[int] = None, love_numbers=None,
                  chunk: int = 200_000,
                  allow_unit_mismatch: bool = False):
    """Area-weighted mean of the field over a region, one value per epoch.

    Returns ``(values, info)``: ``values`` is ``(ntime,)`` and ``info`` records
    what the average actually was, because a regional mean without its coverage is
    not interpretable:

    ``n_points``/``n_total``
        How many of the input points fell inside the region.
    ``coverage``
        Covered fraction of the sphere, ``sum(w)/4pi``.  A basin average is
        ``sum(w f)/sum(w)`` -- i.e. the mean over the **covered area**, not over
        the sphere -- so a small region and a large one are not comparable
        without this number.
    ``weight_rule``/``weight_sum``
        Which integration elements were used (``'grid'`` for a regular grid,
        ``'delaunay'`` for scattered points, ...).  With ``rule='auto'`` the
        choice is made by :func:`shkit.weights.compute_weights` and reported.

    ``weights`` overrides the integration elements entirely (e.g. a mascon area
    file); the region is then still ``mask``.

    ``lat``/``lon``/``mask`` may be 1-D point lists or the 2-D meshgrid of a
    regular grid (any matching shape); they are flattened row-major and the mask
    selects the region.  ``weights``, if given, may be shaped like the coordinates
    or like the selected region.
    """
    from .synthesis import synthesis
    from .weights import FOUR_PI, compute_weights

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape:
        raise ValueError(f"lat 形状 {lat.shape} 与 lon 形状 {lon.shape} 不一致")
    shape = lat.shape
    lat = lat.ravel()
    lon = lon.ravel()
    n_all = lat.size
    sub = None
    if mask is not None:
        sub = np.asarray(mask, dtype=bool)
        if sub.shape != shape:
            raise ValueError(f"mask 形状 {sub.shape} 与坐标形状 {shape} 不一致")
        sub = sub.ravel()
    sel = slice(None) if sub is None else sub
    lat_u, lon_u = lat[sel], lon[sel]
    n_pts = int(lat_u.size)
    if n_pts == 0:
        raise ValueError("区域里一个点都没有：请检查 mask 的坐标顺序是否与 lat/lon 一致")

    if weights is None:
        try:
            ws = compute_weights(lat, lon, rule=rule, subset=sub,
                                 normalise=normalise)
        except Exception as exc:                                 # noqa: BLE001
            raise ValueError(
                f"积分元计算失败（rule={rule!r}）：{exc}\n"
                "区域点太少、太窄或共线时，'delaunay'/'voronoi' 无法三角形化。三条出路：\n"
                "  1) 显式给每一点的面积权重 weights=…（最可控，也是 mascon 数据的做法）；\n"
                "  2) 传 rule='uniform'（等权 —— 只有当每个点代表**等面积**格元时才正确）；\n"
                "  3) 若这是一个规则网格区域，直接传完整的 2-D 网格坐标，"
                "rule='grid' 会按球带面积算。") from exc
        w = np.asarray(ws.w, dtype=float).ravel()
        rule_used, wsum = ws.rule, float(ws.total)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size == n_all and sub is not None:
            w = w[sub]
        if w.size != n_pts:
            raise ValueError(
                f"weights 有 {w.size} 个，区域里有 {n_pts} 个点"
                "（weights 要么与全部坐标等长，要么与区域等长）")
        if np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("weights 必须是非负有限值")
        rule_used, wsum = "user", float(w.sum())
    if wsum <= 0:
        raise ValueError("权重之和为 0，无法求区域平均")

    vals = synthesis(lat_u, lon_u, coeffs, nmax=nmax, gaussian_km=gaussian_km,
                     target_unit=target_unit, love_numbers=love_numbers,
                     chunk=chunk, allow_unit_mismatch=allow_unit_mismatch)
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    avg = (w[:, None] * vals).sum(axis=0) / wsum

    info = {
        "n_points": n_pts, "n_total": int(lat.size),
        "coverage": wsum / FOUR_PI,
        "weight_rule": rule_used, "weight_sum": wsum,
        "lat_range": (float(lat_u.min()), float(lat_u.max())),
        "lon_range": (float(lon_u.min()), float(lon_u.max())),
        "target_unit": target_unit, "gaussian_km": float(gaussian_km),
    }
    if rule_used in ("delaunay", "lattice", "voronoi") or rule_used == "auto":
        info["note"] = (
            f"权重规则 {rule_used!r}：区域平均是**覆盖面积内**的平均，"
            f"覆盖了全球的 {info['coverage'] * 100:.4g}%"
            "（不是全球平均，也不是区域面积归一化的通量）。")
    return avg, info


# ---------------------------------------------------------------------------
# C2 辅助：与外部产品对表
# ---------------------------------------------------------------------------
def match_epochs(times: TimeAxis, other, tol_seconds: float = 60.0):
    """Match two epoch lists by **nearest neighbour within a tolerance**.

    Returns ``(idx_self, idx_other, offsets_seconds)``.  Exact equality is the
    wrong test: an axis rebuilt from legacy decimal years carries only ~16 s of
    resolution (and cross-year epochs ~2e-5 year), so the same physical epoch can
    differ by seconds.  Epochs with no partner inside ``tol_seconds`` are dropped
    and *reported* by the caller rather than silently paired with a neighbour a
    month away.
    """
    a = np.asarray(times.values if hasattr(times, "values") else times,
                   dtype="datetime64[s]").astype("datetime64[s]").astype(float)
    b = np.asarray(other.values if hasattr(other, "values") else other,
                   dtype="datetime64[s]").astype("datetime64[s]").astype(float)
    ia, ib, off = [], [], []
    for i, v in enumerate(a):
        j = int(np.argmin(np.abs(b - v)))
        d = abs(b[j] - v)
        if d <= tol_seconds:
            ia.append(i); ib.append(j); off.append(d)
    return np.array(ia, dtype=int), np.array(ib, dtype=int), np.array(off, float)


# ---------------------------------------------------------------------------
# D7: per-point time fits over a whole cube (trend / annual amplitude+phase)
# ---------------------------------------------------------------------------
def fit_series_maps(values, times, *, poly_order: int = 1,
                    periods: Sequence[float] = (1.0,), mask=None) -> dict:
    """Fit every spatial cell of a series and return the **maps** (D7).

    ``values`` is shaped ``(..., ntime)`` -- a grid cube ``(nlat, nlon, ntime)``
    or a point series ``(npoints, ntime)``; the leading shape is preserved in the
    outputs.  ``times`` is a :class:`~shkit.timeaxis.TimeAxis`, a
    ``datetime64`` array, or anything :meth:`TimeAxis.from_datetimes` accepts.

    Returns a dict with ``trend`` (units per **year**, at the centre of the span),
    ``annual_amplitude``, ``annual_phase_doy`` (day of year of the maximum,
    0-365.25), ``const`` (level at mid-span), ``residual_rms``,
    ``n_used`` and ``cond``.

    ⚠️ ``annual_phase_doy`` is **meaningless where the amplitude is ~0** (a zero
    sinusoid has no peak; the fit still returns ``atan2`` of two noise numbers).
    Callers that draw a phase map must mask it by amplitude -- see
    :func:`shkit.gui.canvases.TrendCanvas`.

    One ``lstsq`` solves every cell at once, so this is O(ncells · ntime · p) with
    no Python loop over cells -- which is what makes the three maps affordable in
    the GUI and in ``shkit series-grid``.

    Missing cells (all-NaN along time) are excluded and counted in ``n_used``;
    a partial NaN makes the whole design row drop out for that cell, and the count
    says how many epochs survived, so a map built from 3 of 200 epochs is visible
    rather than looking authoritative.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim < 2:
        raise ValueError("values 至少要是 (…, ntime)，也就是时间轴在最后一维")
    ntime = v.shape[-1]
    lead = v.shape[:-1]
    Y = v.reshape(-1, ntime)                      # (ncells, ntime)

    if isinstance(times, TimeAxis):
        ax = times
    else:
        ax = TimeAxis.from_datetimes(times)
    ax._require_dates("时间域拟合")
    vv = np.asarray(ax.values, dtype="datetime64[s]")
    if vv.size != ntime:
        raise ValueError(f"times 有 {vv.size} 个历元，但数据最后一维是 {ntime}")
    mid = vv[0] + (vv[-1] - vv[0]) / 2
    t = ((vv - mid).astype("timedelta64[s]").astype(float)
         / (_SEC_PER_DAY * _DAYS_PER_YEAR))
    names, A = design_time(t, poly_order=poly_order, periods=periods)

    finite = np.isfinite(Y)
    n_used = finite.sum(axis=1)
    ok = n_used == ntime                            # 整段可用才参与（与 fit_time_model 一致）
    out = {k: np.full(Y.shape[0], np.nan) for k in
           ("const", "trend", "annual_amplitude", "annual_phase_doy",
            "residual_rms")}
    #: 每个周期的振幅/相位都要能拿到：勾了「含半年周期」却只画周年场，用户会以为
    #: 半年项没算 —— 其实它一直被拟合着，只是从没被报出来。
    seasonal = [{"tag": _PERIOD_NAMES.get(float(p), f"period{float(p):g}"),
                 "period": float(p),
                 "amplitude": np.full(Y.shape[0], np.nan),
                 "phase_doy": np.full(Y.shape[0], np.nan)} for p in periods]
    cond = float(np.linalg.cond(A))
    if ok.any():
        sol, *_ = np.linalg.lstsq(A, Y[ok].T, rcond=None)      # (p, n_ok)
        model = A @ sol
        resid = Y[ok].T - model
        if "const" in names:
            out["const"][ok] = sol[names.index("const")]
        if "trend" in names:
            out["trend"][ok] = sol[names.index("trend")]
        for s in seasonal:
            tag = s["tag"]
            cn, sn = f"{tag}_cos", f"{tag}_sin"
            if cn not in names or sn not in names:
                continue
            a = sol[names.index(cn)]
            b = sol[names.index(sn)]
            per = float(s["period"])
            s["amplitude"][ok] = np.hypot(a, b)
            # 峰值日：模型 a·cos + b·sin 在 t = atan2(b,a)/(2π)·P 处最大
            s["phase_doy"][ok] = np.mod(
                np.degrees(np.arctan2(b, a)) / 360.0 * per, per) * _DAYS_PER_YEAR
        if seasonal:
            # 第一周期同时保留旧键名（annual_*），历史调用与测试不必改
            out["annual_amplitude"] = seasonal[0]["amplitude"]
            out["annual_phase_doy"] = seasonal[0]["phase_doy"]
        out["residual_rms"][ok] = np.sqrt(np.mean(resid ** 2, axis=0))

    res = {k: arr.reshape(lead) for k, arr in out.items()}
    res["seasonal"] = [{**{k: v for k, v in s.items()
                           if k in ("tag", "period")},
                        "amplitude": s["amplitude"].reshape(lead),
                        "phase_doy": s["phase_doy"].reshape(lead)}
                       for s in seasonal]
    res["n_used"] = n_used.reshape(lead)
    res["cond"] = cond
    res["names"] = list(names)
    res["n_cells_used"] = int(ok.sum())
    res["t0"] = mid
    return res


@dataclass
class TimeFilterReport:
    """What a time filter actually did, so it is never a silent smoothing."""

    sigma_years: float
    truncate: float
    n_epochs: int
    effective_epochs: np.ndarray
    edge_epochs: int
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        e = self.effective_epochs
        return "\n".join([
            f"时间滤波   : 高斯核 sigma = {self.sigma_years:g} 年，"
            f"截断 {self.truncate:g}σ",
            f"历元数     : {self.n_epochs}   边界历元（窗口不完整）: {self.edge_epochs}",
            f"有效历元数 : 中位 {np.median(e):.2f}  最小 {e.min():.2f}  "
            f"最大 {e.max():.2f}（1/Σw²；不规则采样下它远小于名义窗口）",
            *[f"提示       : {n}" for n in self.notes],
        ])


def time_gaussian_filter(coeffs: SHCoeffs, sigma_years: float, *,
                         truncate: float = 4.0, weights=None,
                         report: Optional[dict] = None) -> SHCoeffs:
    """Gaussian smoothing **along true elapsed time**.

    Each output epoch is ``sum_i w_i x_i / sum_i w_i`` with
    ``w_i = exp(-0.5 ((t_i - t_k)/sigma)^2)`` restricted to
    ``|t_i - t_k| <= truncate*sigma``; missing epochs get zero weight.

    ⚠️ **This is a smoother, not a low-pass filter with a cutoff.**  On an
    irregular cadence the kernel is not shift-invariant, so it also slightly
    changes amplitude and phase, and it cannot remove a periodic signal without
    attenuating its neighbours.  Honest for visualisation and for suppressing
    month-to-month noise; use :func:`fit_time_model` when you want a *model*.

    ``report``, if given, receives the fields of :class:`TimeFilterReport`.
    """
    sigma = float(sigma_years)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_years 必须为正")
    t, t0 = series_coord(coeffs)
    Y, shape = _flatten(coeffs)
    ok = np.ones(Y.shape[0], dtype=bool)
    ok[_missing_epochs(coeffs)] = False
    ntime = Y.shape[0]

    w_user = None if weights is None else np.asarray(weights, float).ravel()
    if w_user is not None and w_user.size != ntime:
        raise ValueError(f"weights 有 {w_user.size} 个，但历元是 {ntime} 个")

    out = np.zeros_like(Y)
    eff = np.zeros(ntime)
    n_edge = 0
    half = truncate * sigma
    for k in range(ntime):
        sel = ok & (np.abs(t - t[k]) <= half)
        if not sel.any():
            out[k, :] = Y[k, :]
            eff[k] = 1.0
            n_edge += 1
            continue
        d = (t[sel] - t[k]) / sigma
        w = np.exp(-0.5 * d * d)
        if w_user is not None:
            w = w * w_user[sel]
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            out[k, :] = Y[k, :]
            eff[k] = 1.0
            n_edge += 1
            continue
        wn = w / s
        out[k, :] = wn @ Y[sel, :]
        eff[k] = 1.0 / float(np.sum(wn ** 2))
        if float(np.abs(t[sel] - t[k]).max()) >= half - 1e-9:
            n_edge += 1

    rep = TimeFilterReport(sigma_years=sigma, truncate=float(truncate),
                           n_epochs=ntime, effective_epochs=eff,
                           edge_epochs=n_edge)
    if eff.min() < 1.5:
        rep.notes.append(
            f"有历元的有效历元数只有 {eff.min():.2f}：窗口内几乎没有别的历元"
            "（缺测段或序列两端），这些历元基本等于没滤。")
    if _irregular(t):
        rep.notes.append("时间轴不等间隔：高斯核在时间上不是位移不变的，"
                         "滤波会同时轻微改变振幅与相位。")
    if not ok.all():
        rep.notes.append(
            f"{int((~ok).sum())} 个缺数据历元权重为 0（没有把它们当 0 值参与平滑）")
    if report is not None:
        report.update({"sigma_years": sigma, "truncate": float(truncate),
                       "effective_epochs": eff, "edge_epochs": n_edge,
                       "n_epochs": ntime, "notes": list(rep.notes)})

    filt = _with_times_like(_unflatten(out, shape, dict(coeffs.meta)), coeffs)
    filt.meta["time_filter_sigma_years"] = sigma
    filt.meta["time_filter_truncate"] = float(truncate)
    filt.meta["time_filter_effective_epochs"] = eff.tolist()
    filt.meta["time_filter_notes"] = list(rep.notes)
    filt.meta["time_filter"] = rep.summary()
    return filt
