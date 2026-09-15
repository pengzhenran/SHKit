# -*- coding: utf-8 -*-
"""
shkit.series
============

Batch analysis of a **multi-epoch** data set: one call for the whole series,
plus an honest per-epoch diagnostic table.

Why a single call (measured, 1 deg global grid, nmax=60, 24 epochs)
-------------------------------------------------------------------
``analysis()`` already vectorises over a trailing time axis, but calling it once
per epoch re-pays everything that does not depend on time:

===========================  ==========  ==============
item                         per epoch   one call
===========================  ==========  ==============
wall clock                   2.68 s      0.176 s
weights (Voronoi/Delaunay)   every time  once
Gram completeness diagnostic every time  once
===========================  ==========  ==============

That is a **15x** penalty, and it is not the BLAS: the projection itself is only
about a quarter of a call.  The rest is ``synthesize``-for-fit (45%) and the
Gram diagnostic (13%), neither of which belongs inside a per-epoch loop.  So:

* the weights are computed once and passed in;
* the Gram diagnostic is computed once and shared through ``gram_cache``
  (flagged per report as ``gram_reused``, so it is never mistaken for a
  per-epoch measurement);
* the reconstruction used for the residual table is **one** vectorised
  ``synthesize`` over all epochs, and can be skipped entirely
  (``report_fit=False``, the default) when only the coefficients are wanted.

Per-epoch reports reuse the shared diagnostics, so a reader can always tell
which numbers are per-epoch facts and which are shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .analysis import STATS, analysis, stats_delta, stats_snapshot
from .coeffs import SHCoeffs
from .timeaxis import TimeAxis
from .weights import FOUR_PI, compute_weights

__all__ = ["SeriesReport", "analyze_series"]


@dataclass
class SeriesReport:
    """Per-epoch diagnostics for a batch analysis.

    Attributes
    ----------
    times : TimeAxis, optional
        The axis the epochs were solved on (``None`` when the caller gave none).
    table : dict of str -> ndarray
        Column-oriented, one entry per epoch.  Always present: ``index``,
        ``n_points``, ``data_rms``, ``c00``, ``dc_mean_expected``, ``coverage``,
        ``gram_deviation``.  Present when ``report_fit=True``: ``residual_rms``,
        ``fit_rmse_rel``.
    per_epoch : list of AnalysisReport
        The first epoch's report object per chunk, kept so callers can inspect
        the full structure; the shared diagnostics are identical across them.
    shared : dict
        What was computed **once** (weights rule, Gram check, call counters) --
        the audit trail for the optimisation, not a promise.
    """

    times: Optional[TimeAxis] = None
    table: dict = field(default_factory=dict)
    per_epoch: list = field(default_factory=list)
    shared: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ api
    @property
    def ntime(self) -> int:
        return int(np.asarray(self.table.get("index", [])).size)

    def summary(self) -> str:
        t = self.table
        lines = [f"历元数        : {self.ntime}",
                 f"权重规则      : {self.shared.get('weight_rule')}"
                 f"（{'调用方提供' if self.shared.get('weights_provided') else '本次计算'}"
                 f"，调用 {self.shared.get('n_calls', 0)} 次）",
                 f"Gram 诊断     : {self.shared.get('gram_checked')}"
                 f"（计算 {self.shared.get('fill_gram', 0)} 次，"
                 f"复用 {self.shared.get('fill_gram_reused', 0)} 次）"]
        if "residual_rms" in t and np.isfinite(t["residual_rms"]).any():
            r = np.asarray(t["residual_rms"], dtype=float)
            rel = np.asarray(t["fit_rmse_rel"], dtype=float)
            lines += [f"逐历元残差 RMS: 中位 {np.nanmedian(r):.4e}"
                      f"  最小 {np.nanmin(r):.4e}  最大 {np.nanmax(r):.4e}",
                      f"相对 RMSE     : 中位 {np.nanmedian(rel):.3e}"
                      f"  最大 {np.nanmax(rel):.3e}"]
        else:
            lines.append("逐历元残差    : 未计算（report_fit=False —— "
                         "需要残差表就传 report_fit=True）")
        g_km = float(self.shared.get("gaussian_km") or 0.0)
        if g_km > 0:
            lines.append(
                f"高斯平滑      : {g_km:g} km（W(0)={self.shared.get('gaussian_W0'):.6g}，"
                f"W(nmax)={self.shared.get('gaussian_Wnmax'):.6g}）已逐阶乘到系数上；"
                "残差表用的是平滑后的系数")
        c00 = np.asarray(t.get("c00", []), dtype=float)
        if c00.size:
            lines.append(f"C00 范围      : {np.nanmin(c00):.4e} … {np.nanmax(c00):.4e}")
        out = self.outliers()
        if out:
            lines.append(f"可疑历元      : {len(out)} 个（稳健 MAD 离群，只报告不剔除）："
                         + ", ".join(str(i + 1) for i in out[:8]))
        return "\n".join(lines)

    def outliers(self, k: float = 3.0) -> list:
        """Epoch indices whose diagnostic is a robust outlier (MAD, never removes).

        Uses ``residual_rms`` when the fit was computed, otherwise ``data_rms``
        (which still catches an epoch that is all-NaN or wildly rescaled).
        """
        col = "residual_rms" if "residual_rms" in self.table else "data_rms"
        v = np.asarray(self.table.get(col, []), dtype=float)
        if v.size == 0:
            return []
        good = np.isfinite(v)
        if good.sum() < 4:
            return []
        med = float(np.median(v[good]))
        mad = float(np.median(np.abs(v[good] - med)))
        if mad <= 0:
            return []
        z = 0.6745 * (v - med) / mad
        return [int(i) for i in np.nonzero(~good | (np.abs(z) > k))[0]]

    def to_csv(self, path) -> str:
        cols = list(self.table)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(cols) + "\n")
            for i in range(self.ntime):
                row = []
                for c in cols:
                    v = np.asarray(self.table[c])[i]
                    if isinstance(v, (np.floating, float)):
                        row.append("" if not np.isfinite(v) else f"{float(v):.10g}")
                    elif isinstance(v, bytes):
                        row.append(v.decode("utf-8", "replace"))
                    else:
                        row.append(str(v))
                fh.write(",".join(row) + "\n")
        return path


def analyze_series(lat, lon, values, nmax: int, *,
                   times: Optional[TimeAxis] = None,
                   weights=None,
                   epoch_chunk: Optional[int] = None,
                   report_fit: bool = False,
                   gaussian_km: float = 0.0,
                   rule: str = "auto",
                   method: str = "quadrature",
                   field_unit: str = "unknown",
                   target_unit=None,
                   longitude_fft: str = "auto",
                   progress=None,
                   cancel=None,
                   **analysis_kwargs) -> tuple:
    """Solve a whole series in one (or a few) vectorised calls.

    Parameters
    ----------
    lat, lon : array_like
        Sampling geometry, shared by every epoch.
    values : array_like
        ``(npoints, ntime)`` (or ``(npoints,)`` for a single epoch).
    nmax : int
        Maximum degree.
    times : TimeAxis, optional
        Attached to the returned coefficients.  **Not required** -- the math does
        not need dates, and nothing is invented if they are absent.
    weights : WeightSet, optional
        Pre-computed integration elements.  Passing them is what makes the batch
        cheap; when omitted they are computed **once** here (not per epoch).
    epoch_chunk : int, optional
        Solve at most this many epochs per call.  ``None`` (default) is one call
        for everything, which is the fastest; chunking bounds peak memory and
        changes nothing but floating-point summation order.
    report_fit : bool
        Also evaluate the reconstruction to fill ``residual_rms`` /
        ``fit_rmse_rel`` per epoch.  That is one vectorised ``synthesize`` --
        measured at ~45% of a plain call -- so it is **off by default**; the
        per-epoch residual table is the only thing it buys.
    gaussian_km : float
        Isotropic Gaussian smoothing radius (0.5 amplitude half-width).  Applied
        to the **coefficients** once, exactly like the single-epoch path
        (``analysis`` + ``apply_gaussian``), and before the residual table is
        evaluated -- so the batch and the per-epoch loop stay bit-identical when
        the same radius is requested.  Until v2.0.1 this parameter did not exist
        here, which meant 批量分析 silently ignored 「高斯平滑」 while
        「运行分析」 applied it.
    **analysis_kwargs
        Forwarded to :func:`shkit.analysis.analysis` (``reg``, ``alpha``,
        ``niter``, ``sigma`` ...).
    longitude_fft : {'auto', 'fft', 'direct'}
        The longitude kernel for **both** halves of the job -- the solve and (when
        ``report_fit=True``) the reconstruction used for the residual table.  On a
        regular grid both become one ``rfft``/``irfft`` per order, which is where
        the batch win comes from (measured on 181x360, nmax=60: 203 epochs
        5.07 s -> 0.53 s without the residual table, 13.4 s -> 9.3 s with it).
        ``SeriesReport.shared`` records which path each half took
        (``longitude_path`` / ``fit_longitude_path``).

    Returns
    -------
    (SHCoeffs, SeriesReport)
        Coefficients are ``(L+1, L+1, ntime)`` with ``times`` attached.
    """
    from .analysis import analysis as _analysis

    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    f = np.asarray(values, dtype=float)
    single = f.ndim == 1
    if single:
        f = f.reshape(-1, 1)
    if f.shape[0] != lat.size:
        raise ValueError(f"values{f.shape} 的第一维与点数 {lat.size} 不一致")
    ntime = f.shape[1]
    if times is not None and len(times) != ntime:
        raise ValueError(
            f"times 有 {len(times)} 个历元，但数据是 {ntime} 列 —— 二者必须一致")

    # ---- everything that does not depend on time, exactly once -----------
    # the snapshot must be taken BEFORE the weights, or their cost falls outside
    # the measured window and the counter would under-report
    before = stats_snapshot()
    weights_given = weights is not None
    if weights is None:
        STATS["compute_weights"] += 1          # counted here, not inside analysis
        weights = compute_weights(lat, lon, rule=rule)
    gram_cache: dict = {}

    step = ntime if not epoch_chunk else max(int(epoch_chunk), 1)
    C = None
    S = None
    per_epoch: list = []
    unit_meta: dict = {}
    for t0 in range(0, ntime, step):
        t1 = min(t0 + step, ntime)
        if progress is not None:
            progress(f"批量求解 {t0 + 1}–{t1}/{ntime}", t0 / max(ntime, 1))
        co, rep = _analysis(lat, lon, f[:, t0:t1], nmax, weights=weights,
                            method=method, field_unit=field_unit,
                            target_unit=target_unit,
                            gram_cache=gram_cache, report_fit=False,
                            longitude_fft=longitude_fft,
                            progress=progress, cancel=cancel,
                            **analysis_kwargs)
        per_epoch.append(rep)
        if not unit_meta:
            # ★ 沿用 analysis() 给系数的**单位元数据**：正变换 C = a/f_u 已经除过
            # f_u，所以系数是经典无量纲位系数（``field_unit='geopotential'``）。
            # 以前这里自己写 ``"field_unit": field_unit``（把「输入是」那一档当成
            # 系数自身的物理含义），于是 synthesis / synthesis_horizontal 以为
            # "已经是 EWH 系数"而**跳过 f_t 换算** —— 重建场直接小约 1e7 倍
            # （实测真实 mascon：输入 rms 22.3 cm，重建 1e-7）。
            unit_meta = dict(co.meta)
        cc = co.C[:, :, None] if co.C.ndim == 2 else co.C
        ss = co.S[:, :, None] if co.S.ndim == 2 else co.S
        if C is None:
            C = np.zeros((cc.shape[0], cc.shape[1], ntime))
            S = np.zeros_like(C)
        C[:, :, t0:t1] = cc
        S[:, :, t0:t1] = ss

    if cancel is not None and cancel():
        raise RuntimeError("cancelled")
    meta = dict(unit_meta)
    meta.update({"method": method, "weight_rule": weights.rule,
                 "n_points": int(lat.size), "ntime": ntime,
                 "batch": True, "epoch_chunk": step,
                 # 用户声明的「输入是」单独记一笔，别覆盖 field_unit
                 "declared_field_unit": field_unit})
    coeffs = SHCoeffs(C[:, :, 0] if single else C,
                      S[:, :, 0] if single else S,
                      meta, times)

    # ---- Gaussian smoothing, exactly once, before any reconstruction ------
    # 与单历元路径同一口径（AnalysisWorker：先解、再逐阶乘 W、再用平滑后的系数
    # 做重建与残差）。放在这里而不是调用方，是为了让"批量 ≡ 逐历元循环"这条
    # 契约在高斯半径非 0 时也成立。
    g_km = float(gaussian_km or 0.0)
    if g_km > 0:
        from .filters import apply_gaussian, gaussian_coefficients
        if progress is not None:
            progress(f"高斯平滑 {g_km:g} km", 0.88)
        W = gaussian_coefficients(g_km, coeffs.nmax)
        coeffs = apply_gaussian(coeffs, g_km)
        coeffs.meta["gaussian_km"] = g_km
        coeffs.meta["gaussian_W0"] = float(W[0])
        coeffs.meta["gaussian_Wnmax"] = float(W[-1])
        # 下面的诊断表用**平滑后**的系数（否则表里的 C00 与用户导出的系数不是
        # 同一套数；平滑后 C00/DC 均值之比正好是 W(0)，summary() 会写明）。
        C = C * W[:C.shape[0], None, None]
        S = S * W[:S.shape[0], None, None]
    else:
        W = None

    # ---- per-epoch table -------------------------------------------------
    w = np.asarray(weights.w, dtype=float)
    sw = float(w.sum())
    # An epoch that is entirely NaN is a legitimate (and diagnosed) case, so the
    # all-NaN-slice warning would be noise.
    with np.errstate(invalid="ignore"):
        table = {
            "index": np.arange(ntime),
            "n_points": np.full(ntime, lat.size, dtype=int),
            "data_rms": np.sqrt(np.nanmean(f ** 2, axis=0)),
            "coverage": np.full(ntime, sw / FOUR_PI),
            "gram_deviation": np.full(ntime, float(per_epoch[0].gram_deviation)),
        }
    # C00 and the weighted mean need no reconstruction (that is the point of the
    # DC identity): cheap enough to always report.
    table["c00"] = np.asarray(C[0, 0, :], dtype=float)
    table["dc_mean_expected"] = np.asarray((w[:, None] * f).sum(axis=0) / FOUR_PI,
                                           dtype=float)
    if times is not None:
        table["time"] = np.array([str(v) for v in times.values])

    if report_fit:
        if progress is not None:
            progress("逐历元残差（一次向量化综合）", 0.9)
        from .synthesis import synthesis as _synthesis
        # ★ 残差要拿"数据所在的那个物理量"跟数据比：系数是位系数，而 f 可能是
        # EWH / geoid / 形变 —— 不做反变换就相当于拿 1e-8 量级的系数去减 20 cm 的
        # 数据（以前正是这样：`target_unit` 没传，`fit` 少乘了整个 f_t）。
        # ``scalar``/``unknown``（无物理公式，f_u = 1）保持不换算。
        fit_target = (field_unit if field_unit not in ("scalar", "unknown", None)
                      else None)
        fit_rep: dict = {}
        fit = np.asarray(_synthesis(lat, lon, coeffs,
                                    target_unit=fit_target,
                                    longitude_fft=longitude_fft,
                                    report=fit_rep)).reshape(f.shape)
        resid = fit - f
        with np.errstate(invalid="ignore"):
            table["residual_rms"] = np.sqrt(np.nanmean(resid ** 2, axis=0))
            table["fit_rmse_rel"] = table["residual_rms"] / np.where(
                table["data_rms"] > 0, table["data_rms"], np.nan)
    else:
        fit_rep = {}
        fit_target = None

    shared = {
        "weight_rule": weights.rule,
        "weight_sum": sw,
        "coverage": sw / FOUR_PI,
        "weights_provided": weights_given,
        "gram_checked": per_epoch[0].meta.get("gram_checked"),
        "gram_deviation": float(per_epoch[0].gram_deviation),
        "longitude_path": per_epoch[0].meta.get("longitude_path"),
        "longitude_reason": per_epoch[0].meta.get("longitude_reason"),
        "fit_longitude_path": fit_rep.get("longitude_path"),
        "epoch_chunk": step,
        "report_fit": bool(report_fit),
        "residual_target_unit": fit_target,
        "declared_field_unit": field_unit,
        "coefficient_field_unit": meta.get("field_unit"),
        "gaussian_km": g_km,
        "gaussian_W0": (float(W[0]) if W is not None else None),
        "gaussian_Wnmax": (float(W[-1]) if W is not None else None),
        "n_calls": len(per_epoch),
    }
    # flatten the counters so the summary can print them without a second lookup
    shared.update(stats_delta(before))
    shared["counters"] = stats_delta(before)
    rep = SeriesReport(times=times, table=table, per_epoch=per_epoch,
                       shared=shared,
                       meta={"method": method, "field_unit": field_unit,
                             "declared_field_unit": field_unit,
                             "coefficient_field_unit": meta.get("field_unit"),
                             "n_points": int(lat.size), "ntime": ntime})
    if progress is not None:
        progress("完成", 1.0)
    return coeffs, rep
