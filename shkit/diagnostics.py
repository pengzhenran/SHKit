# -*- coding: utf-8 -*-
"""
shkit.diagnostics
=================

Honest, quantitative reporting for a spherical harmonic analysis.

An SH analysis of scattered or incomplete data is only as trustworthy as the
conditioning of the problem, so every solver in :mod:`shkit.analysis` returns an
:class:`AnalysisReport` carrying:

``gram_deviation``
    ``max|K - I|`` with ``K = A^T W A / (4 pi)``.  ``K == I`` is exactly the
    condition under which weighted quadrature and weighted least squares agree
    and quadrature has no aliasing.  Values below ~1e-2 mean plain quadrature is
    safe; large values mean quadrature is aliased and a least-squares solve is
    required.
``condition_number``
    ``cond(A sqrt(W))``.  Above ~1e8-1e10 the normal equations stop being
    solvable in double precision and regularisation becomes mandatory.
``shannon_number``
    ``(L+1)**2 * coverage`` - the number of coefficients a region of that size
    can actually support.  ``shannon_number ``<<`` ncoef`` means the requested
    degree is far beyond what the data can resolve.
``lmax_recommended``
    Largest degree that the sampling/coverage supports at the requested
    overdetermination, subject to the ``(L+1)**2 <= n_points`` hard limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .weights import FOUR_PI, lonlat_to_xyz

__all__ = [
    "AnalysisReport",
    "gram_matrix",
    "gram_deviation",
    "shannon_number",
    "recommend_lmax",
    "harmonic_resolution_km",
    "min_point_spacing_deg",
    "degrees_of_freedom",
    "truncation_table",
    "diagnose_nmax",
]


# ---------------------------------------------------------------------------
# core quantities
# ---------------------------------------------------------------------------
def gram_matrix(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted Gram matrix ``K = A^T diag(w) A / (4 pi)``.

    ``A`` is the design matrix ``(npoints, ncoef)`` of real (cos/sin)
    coefficients and ``w`` the area elements.  When the sampling plus weights
    form an exact quadrature rule, ``K`` is the identity.
    """
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    return (A * w[:, None]).T @ A / FOUR_PI


def gram_deviation(K: np.ndarray, block: int = 512) -> float:
    """``max|K - I|`` computed blockwise (never materialises K - I whole)."""
    n = K.shape[0]
    worst = 0.0
    for s in range(0, n, block):
        e = min(s + block, n)
        sub = K[s:e].copy()
        idx = np.arange(s, e)
        sub[np.arange(e - s), idx] -= 1.0
        worst = max(worst, float(np.abs(sub).max()))
    return worst


def degrees_of_freedom(lmax: int, coverage: float = 1.0) -> float:
    """Shannon number: coefficients a region of that area can support."""
    return (lmax + 1) ** 2 * float(coverage)


def shannon_number(lmax: int, coverage: float = 1.0) -> float:
    """Alias of :func:`degrees_of_freedom`."""
    return degrees_of_freedom(lmax, coverage)


def recommend_lmax(n_points: int, coverage: float = 1.0,
                   oversampling: float = 2.0,
                   quadrature_safe: bool = False,
                   hard_cap: Optional[int] = None) -> int:
    """Largest defensible degree for the available sampling.

    Parameters
    ----------
    n_points : int
        Number of usable samples.
    coverage : float
        Fraction of the sphere covered (``sum(w)/(4 pi)``).
    oversampling : float
        Required ``n_points / (L+1)**2`` ratio.  2 is a reasonable minimum;
        use >= 4 when the data are noisy (noise amplification is
        ``(L+1)/sqrt(n) * sqrt(1/coverage)``).
    quadrature_safe : bool
        Additionally cap at ``0.5*sqrt(n_points/coverage)``, the conservative
        limit for *plain quadrature on scattered points* (as opposed to a least
        squares solve, which is bounded only by the counting limit).
    """
    cov = max(float(coverage), 1e-12)
    n = max(int(n_points), 1)
    l_count = int(np.floor(np.sqrt(n / (cov * oversampling)))) - 1
    lim = max(l_count, 0)
    if quadrature_safe:
        l_quad = int(np.floor(0.5 * np.sqrt(n / cov)))
        lim = min(lim, max(l_quad, 0))
    if hard_cap is not None:
        lim = min(lim, int(hard_cap))
    return int(max(lim, 0))


def harmonic_resolution_km(lmax: int, radius_km: float = 6371.0) -> float:
    """Half-wavelength resolution ``pi*R/lmax`` in km."""
    if lmax <= 0:
        return np.inf
    return float(np.pi * radius_km / lmax)


def min_point_spacing_deg(lat_deg, lon_deg) -> float:
    """Nearest-neighbour angular separation (degrees), robust to duplicates."""
    from scipy.spatial import cKDTree

    xyz = lonlat_to_xyz(lat_deg, lon_deg)
    if xyz.shape[0] < 2:
        return np.nan
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=2)
    return float(np.rad2deg(2.0 * np.arcsin(np.clip(d[:, 1] / 2.0, 0, 1))).min())


def arcmin_spacing_deg(lat_deg, lon_deg) -> float:
    """Median nearest-neighbour separation (degrees)."""
    from scipy.spatial import cKDTree

    xyz = lonlat_to_xyz(lat_deg, lon_deg)
    if xyz.shape[0] < 2:
        return np.nan
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=2)
    return float(np.rad2deg(2.0 * np.arcsin(np.clip(d[:, 1] / 2.0, 0, 1))).median())


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@dataclass
class AnalysisReport:
    """Diagnostics returned alongside every analysis result."""

    method: str = "quadrature"
    n_points: int = 0
    nmax: int = 0
    ntime: int = 1
    weight_rule: str = ""
    weight_sum: float = np.nan
    coverage: float = 1.0
    gram_deviation: float = np.nan
    condition_number: float = np.nan
    residual_rms: float = np.nan
    data_rms: float = np.nan
    residual_rms_weighted: float = np.nan
    dc_mean_expected: float = np.nan
    dc_mean_got: float = np.nan
    regularization: Optional[str] = None
    alpha: float = 0.0
    n_iterations: int = 0
    meta: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    # -------------------------------------------------------------- derived
    @property
    def ncoef(self) -> int:
        return (self.nmax + 1) ** 2

    @property
    def overdetermination(self) -> float:
        return self.n_points / self.ncoef if self.ncoef else np.nan

    @property
    def fit_rmse_rel(self) -> float:
        if not np.isfinite(self.residual_rms) or not np.isfinite(self.data_rms) \
                or self.data_rms == 0:
            return np.nan
        return self.residual_rms / self.data_rms

    @property
    def shannon(self) -> float:
        return degrees_of_freedom(self.nmax, self.coverage)

    @property
    def lmax_recommended(self) -> int:
        return recommend_lmax(self.n_points, self.coverage)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    # --------------------------------------------------------------- checks
    def check(self) -> "AnalysisReport":
        """Append warnings for every diagnostic that looks unhealthy."""
        w = self.warnings
        if (np.isfinite(self.weight_sum) and abs(self.coverage - 1.0) > 1e-3
                and not self.method.startswith("slepian")):
            if self.coverage < 0.999:
                w.append(
                    f"weights cover only {self.coverage:.4f} of the sphere "
                    f"(sum(w) = {self.weight_sum:.6f}, 4*pi = {FOUR_PI:.6f}).  "
                    "For a regional dataset the honest reading is target A -- the "
                    "zero-filled global field (0 outside the region) -- and its "
                    "coefficients come from method='projection' with the geometric "
                    "sum, which keeps C00 = area fraction and shows the real "
                    "truncation leakage.  Do NOT reach for an unregularised "
                    "least-squares solve here: with this coverage its normal "
                    "matrix is numerically singular, so it returns the "
                    "minimum-norm member of a near-null space rather than the "
                    "field.  If you need target B (a global field constrained by "
                    "the region only), regularise (reg='kaula'), use the Slepian "
                    "basis, or remove-restore a global long-wavelength model.")
        if np.isfinite(self.gram_deviation) and self.gram_deviation > 1e-2:
            w.append(
                f"quadrature completeness max|K-I| = {self.gram_deviation:.3e}: "
                "the sampling is not an accurate quadrature rule for this degree, "
                "so the plain projection carries a sampling bias.  If the data "
                "cover the sphere, that bias is a discretisation error and is "
                "fixable: use method='wlsq' with reg='kaula', or "
                "method='projection' with tau set so that only the well-"
                "concentrated Slepian directions are corrected.  If they cover "
                "only a region, do NOT invert K -- the bias is not what limits "
                "you, the truncation leakage is.")
        if (np.isfinite(self.dc_mean_expected)
                and np.isfinite(self.dc_mean_got)
                and abs(self.dc_mean_expected) > 0.0):
            rel = abs(self.dc_mean_got - self.dc_mean_expected) \
                / abs(self.dc_mean_expected)
            if rel > 0.1:
                w.append(
                    f"C00 = {self.dc_mean_got:.6e} disagrees with the weighted "
                    f"mean of the data = {self.dc_mean_expected:.6e} "
                    f"(ratio {rel:.3e}).  C00 *is* that mean -- it is the mean of "
                    "the field over the sphere, and truncating at any degree "
                    "preserves it -- so this solve has not represented the field.  "
                    "A small sample residual is no defence: it was minimised at "
                    "the samples only, which excludes the implicit zeros.  Check "
                    "the integration elements (rule and normalise), then use "
                    "method='projection'.")
        if (self.coverage < 0.999
                and self.method in ("wlsq", "cg", "iterative")
                and np.isfinite(self.fit_rmse_rel) and self.fit_rmse_rel < 1e-3):
            w.append(
                f"sample residual has collapsed to {self.fit_rmse_rel:.2e} while "
                f"the data cover only {self.coverage:.3e} of the sphere.  "
                f"{self.method!r} minimises the misfit AT THE SAMPLES only, and a "
                "finite-degree truncation leaves an irreducible leakage floor, so "
                "a residual this small indicates that the solve has exploited a "
                "near-null space -- not that the field was recovered.  Compare "
                "against method='projection' and check C00 before trusting it.")
        if np.isfinite(self.condition_number) and self.condition_number > 1e8:
            w.append(
                f"condition number {self.condition_number:.2e} > 1e8: double "
                "precision is exhausted.  Lower nmax, raise the "
                "overdetermination, or supply regularization "
                "(reg='kaula' and/or alpha>0).")
        if self.overdetermination < 2:
            w.append(
                f"overdetermination n_points/ncoef = {self.overdetermination:.2f} "
                "< 2: the solution is barely determined and will trade "
                "resolution for noise.  Recommended nmax for this sampling "
                f"is {self.lmax_recommended}.")
        if self.shannon < 0.5 * self.ncoef:
            w.append(
                f"Shannon number {self.shannon:.0f} << ncoef {self.ncoef}: the "
                "covered area cannot support this degree.")
        return self

    # ------------------------------------------------------------- printing
    def report(self) -> dict:
        return {
            "method": self.method,
            "auto_choice": self.meta.get("auto_choice"),
            "n_points": self.n_points,
            "nmax": self.nmax,
            "ncoef": self.ncoef,
            "ntime": self.ntime,
            "weight_rule": self.weight_rule,
            "weight_sum": self.weight_sum,
            "coverage": self.coverage,
            "overdetermination": self.overdetermination,
            "shannon": self.shannon,
            "gram_deviation": self.gram_deviation,
            "condition_number": self.condition_number,
            "residual_rms": self.residual_rms,
            "data_rms": self.data_rms,
            "residual_rms_weighted": self.residual_rms_weighted,
            "dc_mean_expected": self.dc_mean_expected,
            "dc_mean_got": self.dc_mean_got,
            "fit_rmse_rel": self.fit_rmse_rel,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "n_iterations": self.n_iterations,
            "lmax_recommended": self.lmax_recommended,
            "resolution_km": harmonic_resolution_km(self.nmax),
            "warnings": list(self.warnings),
        }

    def describe(self) -> str:
        r = self.report()
        lines = [
            f"AnalysisReport(method={r['method']})",
            f"  samples            : {r['n_points']}  (ntime={r['ntime']})",
            f"  nmax / ncoef       : {r['nmax']} / {r['ncoef']}   "
            f"overdetermination = {r['overdetermination']:.2f}",
            f"  weight rule / sum  : {r['weight_rule']!r}  "
            f"sum={r['weight_sum']:.6f}  coverage={r['coverage']:.6f}",
            f"  Shannon number     : {r['shannon']:.1f}",
            f"  gram max|K-I|      : {r['gram_deviation']:.3e}",
            f"  condition number   : {r['condition_number']:.3e}",
        ]
        if r.get("auto_choice"):
            why = {
                "quadrature": "求积健康（采样是够精度的求积规则）→ 直接用**直接求积**，最快",
                "projection": "区域覆盖 → 用零填充全球投影（不是最小二乘）",
                "wlsq": "求积不健康 → 升级到加权最小二乘",
                "cg": "求积不健康 或 矩阵太大 → 升级到矩阵无关共轭梯度",
            }.get(r["auto_choice"], "")
            lines.append(f"  auto 选择          : {r['auto_choice']}"
                         + (f" —— {why}" if why else ""))
        if np.isfinite(r["residual_rms"]):
            lines.append(f"  fit residual       : {r['residual_rms']:.6e}  "
                         f"(relative {r['fit_rmse_rel']:.3e})")
        if np.isfinite(r["residual_rms_weighted"]):
            lines.append(f"  fit residual (area-weighted) : "
                         f"{r['residual_rms_weighted']:.6e}   "
                         "<- the honest L2 norm")
        if np.isfinite(r["dc_mean_expected"]):
            ratio = (r["dc_mean_got"] / r["dc_mean_expected"]
                     if r["dc_mean_expected"] else np.nan)
            lines.append(f"  C00 / weighted mean : {r['dc_mean_got']:.6e} / "
                         f"{r['dc_mean_expected']:.6e}  (ratio {ratio:.6e})")
        if r["regularization"]:
            lines.append(f"  regularization     : {r['regularization']} "
                         f"alpha={r['alpha']:.3e}")
        lines.append(f"  resolution @nmax   : {r['resolution_km']:.1f} km "
                     f"half-wavelength")
        lines.append(f"  recommended nmax   : {r['lmax_recommended']}")
        for msg in r["warnings"]:
            lines.append(f"  WARNING: {msg}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# "which nmax should I use?" -- truncation vs sampling vs region size
# ---------------------------------------------------------------------------
def degree_power(coeffs) -> np.ndarray:
    """Total power of each degree: ``sum_m (C_nm^2 + S_nm^2)``."""
    C = np.asarray(coeffs.C, dtype=float)
    S = np.asarray(coeffs.S, dtype=float)
    if C.ndim == 3:                       # multi-epoch: aggregate over time
        C = np.sqrt(np.mean(C ** 2, axis=2)) * C.shape[2] ** 0.5
        S = np.sqrt(np.mean(S ** 2, axis=2)) * S.shape[2] ** 0.5
    return np.array([np.sum(C[n] ** 2) + np.sum(S[n] ** 2)
                     for n in range(C.shape[0])])


def truncation_table(coeffs, nmax_list=None) -> list:
    """Truncating to ``nmax`` costs exactly ``sqrt(1 - retained_power)`` (Parseval).

    Returns a list of dicts with ``nmax``, ``retained`` (fraction of the total
    power kept) and ``error`` (relative RMS of the reconstruction you would get
    from that truncation).  The relation ``error = sqrt(1 - retained)`` is exact
    for an orthogonal basis, so it needs no experiment - and it is **not**
    affected by how dense the sampling is.
    """
    p = degree_power(coeffs)
    L = p.size - 1
    total = float(p.sum())
    if total <= 0:
        return []
    if nmax_list is None:
        step = max(1, (L + 1) // 10)
        nmax_list = sorted(set(list(range(step, L + 1, step)) + [L]))
    out = []
    for n in nmax_list:
        n = int(min(max(n, 0), L))
        keep = float(p[:n + 1].sum()) / total
        out.append({"nmax": n, "retained": keep,
                    "error": float(np.sqrt(max(1.0 - keep, 0.0))),
                    "resolution_km": harmonic_resolution_km(n) if n else np.inf})
    return out


def diagnose_nmax(coeffs, coverage: Optional[float] = None,
                  n_points: Optional[int] = None,
                  tol: float = 0.01, nmax_list=None) -> dict:
    """Answer "which nmax should I use?" from three **independent** limits.

    1. **Truncation** (data bandwidth): to keep the reconstruction error below
       ``tol`` you need ``retained_power >= 1 - tol**2``.  This limit comes from
       the field's own spectrum and is independent of the sampling.
    2. **Sampling / coverage**: ``lmax_recommended`` from :func:`recommend_lmax`;
       below it the solve is under-determined or noise-amplifying.
    3. **Region size** (only when ``coverage < 1``): the Shannon number
       ``(L+1)**2 * coverage`` counts how many coefficients the footprint can
       actually support.  ``Shannon < 1`` means the degree is far beyond what the
       region can carry.

    ``coverage`` / ``n_points`` default to what the coefficients themselves
    recorded (``meta``), so a coefficient file read back from disk works as-is.
    """
    meta = getattr(coeffs, "meta", {}) or {}
    hdr = meta.get("header_fields", {}) or {}
    if coverage is None:
        for src in (meta, hdr):
            v = src.get("coverage")
            if v is not None:
                try:
                    coverage = float(v)
                    break
                except (TypeError, ValueError):
                    pass
    if n_points is None:
        for src in (meta, hdr):
            v = src.get("n_points")
            if v is not None:
                try:
                    n_points = int(v)
                    break
                except (TypeError, ValueError):
                    pass

    table = truncation_table(coeffs, nmax_list)
    L = coeffs.nmax

    # (1) smallest nmax whose truncation error is within tol.  Exact search on
    #     the cumulative power (not just the printed rows): error <= tol
    #     <=> retained >= 1 - tol**2.
    #     Note the spectrum is only known up to the degree the coefficients
    #     already have, so hitting L means "even the full set barely makes it".
    p = degree_power(coeffs)
    nmax_tol = None
    if p.sum() > 0:
        cum = np.cumsum(p) / p.sum()
        idx = int(np.searchsorted(cum, 1.0 - tol ** 2, side="left"))
        nmax_tol = int(min(idx, L))
    at_ceiling = bool(nmax_tol is not None and nmax_tol >= L and L > 0)

    # (2) sampling limit.  recommend_lmax divides by the coverage, which is the
    #     right counting limit when the samples are spread over the sphere; for
    #     a small region it becomes enormous and meaningless (it counts
    #     coefficients the region cannot support at all), so it is reported only
    #     as a counting bound and flagged.
    nmax_sampling = None
    sampling_is_binding = True
    if n_points:
        cov = float(coverage) if coverage else 1.0
        nmax_sampling = recommend_lmax(int(n_points), cov)
        if cov < 1e-3:
            sampling_is_binding = False

    # (3) region limit
    nmax_shannon = None
    shannon_at_L = None
    if coverage and coverage < 1.0 - 1e-9:
        shannon_at_L = (L + 1) ** 2 * float(coverage)
        nmax_shannon = int(np.ceil(np.sqrt(1.0 / float(coverage)) - 1.0))

    notes = []
    if coverage and coverage < 1.0 - 1e-9:
        notes.append(
            f"权重只覆盖全球的 {coverage:.3e}（Σw = 4π×{coverage:.3e}）——这是"
            "**区域**问题：上面的系数是「区域内 = 场、区域外 = 0」的投影，"
            "不是可反演的全球解。")
    _p = degree_power(coeffs)
    if _p.sum() > 0:
        flat = float(_p[1:].max() / max(_p[1:].mean(), 1e-300)) if L >= 1 else 1.0
        if flat < 3.0:
            notes.append(
                "逐阶功率几乎是**平的**（最大/平均 = "
                f"{flat:.2f}）——这是小尺度/掩膜型场的特征，谱天生很宽，"
                "截断会丢掉大量功率。")
        elif flat > 20.0:
            notes.append(
                f"逐阶功率很**红**（最大/平均 = {flat:.1f}）——低阶主导，"
                "适度截断的代价不大。")

    return {
        "nmax_of_coeffs": L,
        "coverage": coverage,
        "n_points": n_points,
        "tol": tol,
        "nmax_for_tolerance": nmax_tol,
        "tolerance_hits_ceiling": at_ceiling,
        "nmax_sampling": nmax_sampling,
        "sampling_is_binding": sampling_is_binding,
        "nmax_region_shannon": nmax_shannon,
        "shannon_at_nmax": shannon_at_L,
        "table": table,
        "notes": notes,
    }


def describe_nmax(diag: dict) -> str:
    """Render :func:`diagnose_nmax` as plain text."""
    d = diag
    lines = ["=" * 72, "  用哪个 nmax？三个互相独立的约束", "=" * 72]
    L = d["nmax_of_coeffs"]
    lines.append(f"  系数本身的阶数 nmax = {L}"
                 f"（半波长 {harmonic_resolution_km(L):.1f} km）")
    if d["coverage"] is not None:
        lines.append(f"  覆盖 / 采样点数     : {d['coverage']:.6e} / "
                     f"{d['n_points'] if d['n_points'] else '—'}")
    lines.append("")
    lines.append(f"  ① 截断（数据带宽）：保留功率比 → 重建误差 = sqrt(1−保留比)")
    lines.append(f"     {'nmax':>6}{'保留功率比':>14}{'重建误差':>14}"
                 f"{'半波长 (km)':>15}")
    for row in d["table"]:
        lines.append(f"     {row['nmax']:>6}{row['retained']:>13.2%} "
                     f"{row['error']:>13.4e}{row['resolution_km']:>15.1f}")
    lines.append("     ↑ 精确关系（Parseval），与采样密度无关")
    if d.get("tolerance_hits_ceiling"):
        lines.append(f"     ⚠ 在这套系数能看到的范围（≤{L} 阶）内，没有更小的阶数"
                     f"能把误差压到 {d['tol']:.2%} 以内——")
        lines.append("       真实带宽 ≥ 当前阶数，需要更高阶的分析才能定；"
                     "谱越平，这个代价越大。")
    lines.append("")
    lines.append("  ② 采样/覆盖够不够定解：")
    if d["nmax_sampling"] is not None:
        lines.append(f"     lmax_recommended = {d['nmax_sampling']}"
                     "（recommend_lmax：n_points/((L+1)²·coverage) ≥ 2）")
        if not d.get("sampling_is_binding", True):
            lines.append("     ⚠ 覆盖比 ≪ 1 时这只是**计数上限**——点数再多也"
                         "变不出区域装不下的自由度，有效约束是 ③。")
    else:
        lines.append("     （没记录采样点数，无法判断；传 --points 或看诊断报告）")
    lines.append("")
    lines.append("  ③ 区域能承载多少自由度（Shannon = (L+1)²·覆盖比）：")
    if d["nmax_region_shannon"] is None:
        lines.append("     （覆盖全球，此约束不适用）")
    else:
        lines.append(f"     nmax={L} 时 Shannon = {d['shannon_at_nmax']:.4e}"
                     f"（<< 1 表示远超出区域能承载的）")
        lines.append(f"     要让 Shannon ≈ 1 → nmax ≈ {d['nmax_region_shannon']}")
    if d["notes"]:
        lines.append("")
        for n in d["notes"]:
            lines.append(f"  · {n}")
    lines.append("")
    if d["nmax_region_shannon"] is None:
        lines.append(f"  一句话：① 要求 nmax ≥ {d['nmax_for_tolerance']}；"
                     "③ 不适用（覆盖全球）。")
        lines.append("        宁小勿大：阶数越高噪声放大越厉害，"
                     "取到①的下限就够。")
    else:
        lines.append(f"  一句话：① 要求 nmax ≥ {d['nmax_for_tolerance']}（或更高），"
                     f"③ 要求 nmax ≲ {d['nmax_region_shannon']}；")
        if (d["nmax_for_tolerance"] is not None
                and d["nmax_for_tolerance"] > d["nmax_region_shannon"]):
            lines.append("        两者**冲突** → 数据本身不支持你要的分辨率："
                         "该用 Slepian 局部化，或降低目标、加正则化。")
        else:
            lines.append("        两者不冲突时取区间内的值即可。")
    return "\n".join(lines)
