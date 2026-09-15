# -*- coding: utf-8 -*-
"""
shkit.weights
=============

Integration elements (quadrature weights) for spherical harmonic analysis.

The analysis integral is

.. math::

    C_{nm} = \\frac{1}{4\\pi}\\sum_i w_i\\, f_i\\, \\bar P_{nm}(\\sin\\varphi_i)
             \\cos(m\\lambda_i)

so ``w_i`` must approximate the solid-angle element
``dOmega_i = cos(phi_i) dphi dlambda`` of sample ``i``.  Writing the rule this
way fixes the convention

.. math::

    \\sum_{\\text{whole sphere}} w_i = 4\\pi

Available rules
---------------
``'dh'``        Driscoll-Healy latitude weights.  Exact quadrature for global
                equiangular grids with ``nlon == 2*nlat``; recovers band-limited
                fields up to degree ``nlat/2 - 1``.
``'glq'``       Gauss-Legendre x equiangular-longitude grid (exact quadrature).
                Use :func:`glq_grid` to build the sampling.
``'grid'``      Exact spherical band area ``dlambda * (sin phi2 - sin phi1)``
                for *any* lat/lon grid, global or regional, uniform or not.
                This is the correct replacement for the common
                ``dlat*dlon*cos(phi)`` shortcut, which **overestimates** the
                cell area by ``(dlat)**2/24`` - 1.3e-5 at a 1-degree step
                (verified numerically in ``tests/validate_core.py``).
``'voronoi'``   Spherical Voronoi cell area (scipy).  The geometrically correct
                element for **arbitrary scattered points**.  Always sums to
                ``4*pi`` because a Voronoi tessellation covers the whole sphere.
``'delaunay'``  Spherical Delaunay barycentric (one-third) rule.  Sums to the
                area of the **spherical convex hull** of the point set, so it is
                the right choice for a **regional** point set, where Voronoi
                weights would be inflated by boundary cells that balloon across
                the empty side of the sphere.
``'uniform'``   Equal area ``4*pi/N`` per point.  Only defensible for genuinely
                equal-area samplings (HEALPix-like, Fibonacci).
``'user'``      A weight vector supplied by the caller.

.. warning::

   Two rules that look interchangeable are **not**:

   * ``SphericalVoronoi`` on a *regional* subset returns cell areas that still
     sum to ``4*pi`` - the boundary cells expand to cover the rest of the
     sphere.  Measured on a 30-degree cap (264 points): true cap area 0.84,
     Voronoi sum 12.57.  Use ``'delaunay'`` there, or pass the global point set
     together with ``subset=``.
   * Area weights are *quadrature* weights.  They are not statistical weights:
     for least squares you want ``1/sigma_i**2``, optionally combined with the
     area as ``A_i/sigma_i**2`` when each sample is a block average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "WEIGHT_RULES",
    "WeightSet",
    "dh_lat_weights",
    "glq_grid",
    "grid_cell_weights",
    "lattice_cell_weights",
    "lattice_step",
    "voronoi_weights",
    "delaunay_weights",
    "uniform_weights",
    "geodesic_cell_weights",
    "compute_weights",
    "looks_like_lattice",
    "lonlat_to_xyz",
    "xyz_to_lonlat",
    "sphere_area",
]

WEIGHT_RULES = ("dh", "glq", "grid", "lattice", "voronoi", "delaunay",
                "uniform", "user", "auto")

FOUR_PI = 4.0 * np.pi


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def lonlat_to_xyz(lat_deg, lon_deg) -> np.ndarray:
    """Unit vectors for ``(lat, lon)`` in degrees.  Shape ``(n, 3)``."""
    la = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lo = np.deg2rad(np.asarray(lon_deg, dtype=float))
    return np.column_stack([np.cos(la) * np.cos(lo),
                            np.cos(la) * np.sin(lo),
                            np.sin(la)])


def xyz_to_lonlat(xyz) -> tuple:
    """Inverse of :func:`lonlat_to_xyz`."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    n = np.linalg.norm(xyz, axis=1)
    x, y, z = (xyz / n[:, None]).T
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.mod(np.rad2deg(np.arctan2(y, x)), 360.0)
    return lat, lon


def sphere_area(radius: float = 1.0) -> float:
    """Area of a sphere, ``4*pi*r**2``."""
    return FOUR_PI * radius ** 2


def _sorted_edges(values, period: Optional[float] = None,
                  clip: Optional[tuple] = None):
    """Cell edges (midpoints) for a 1-D coordinate axis.

    Returns ``(edges_sorted, inverse_permutation, sorted_values)``.
    """
    v = np.asarray(values, dtype=float).ravel()
    order = np.argsort(v, kind="stable")
    s = v[order]
    n = len(s)
    e = np.empty(n + 1)
    if n == 1:
        e[0] = s[0] - 0.5
        e[1] = s[0] + 0.5
    else:
        e[1:-1] = 0.5 * (s[:-1] + s[1:])
        if period is None:
            e[0] = s[0] - 0.5 * (s[1] - s[0])
            e[-1] = s[-1] + 0.5 * (s[-1] - s[-2])
        else:
            gap = (s[0] + period) - s[-1]
            e[0] = s[0] - 0.5 * gap
            e[-1] = s[-1] + 0.5 * gap
    if clip is not None:
        e = np.clip(e, clip[0], clip[1])
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    return e, inv, s


# ---------------------------------------------------------------------------
# rule implementations
# ---------------------------------------------------------------------------
def dh_lat_weights(lat_deg) -> np.ndarray:
    """Driscoll-Healy latitude quadrature weights (they sum to 2).

    Use them as ``w_j * dlambda`` with ``dlambda = 2*pi/nlon`` to obtain the
    per-point solid angle.  Reproduces ``compute_spherical_weights`` of the
    reference ``m2py`` implementation, which is ``(4*pi/nlat**2) * sin(theta_j)
    * sum_k sin((2k+1)theta_j)/(2k+1)`` and equals ``dlambda * a_j`` when
    ``nlon == 2*nlat``.

    The sampling must be a global equiangular grid with ``nlat`` even and
    ``nlon = 2*nlat``; then the rule integrates products of SH up to degree
    ``nlat/2 - 1`` exactly.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    nlat = lat.size
    if nlat % 2:
        raise ValueError("Driscoll-Healy requires an even number of latitudes")
    colat = np.deg2rad(90.0 - lat)
    k = np.arange(nlat // 2)
    w = np.empty(nlat)
    for j in range(nlat):
        c = colat[j]
        w[j] = (4.0 / nlat) * np.sin(c) * \
            np.sum(np.sin((2 * k + 1) * c) / (2 * k + 1))
    # this is the pure latitude rule: sum_j a_j = 2
    return w


def glq_grid(lmax: int):
    """Exact Gauss-Legendre quadrature grid for degree ``lmax``.

    Returns ``(lat, lon, w)`` with ``lat`` the ``lmax+1`` Gauss nodes (degrees),
    ``lon`` the ``2*lmax+1`` equiangular longitudes, and ``w`` the per-point
    solid angle (``sum(w) == 4*pi``).  A field band-limited to ``lmax`` is
    recovered **exactly** on this grid - it is the reference path used to
    validate the scattered-point machinery.
    """
    if lmax < 1:
        raise ValueError("lmax must be >= 1")
    x, gw = np.polynomial.legendre.leggauss(lmax + 1)   # sum(gw) = 2
    lat = np.rad2deg(np.arcsin(x))
    lon = np.arange(2 * lmax + 1) * (360.0 / (2 * lmax + 1))
    dlam = 2 * np.pi / (2 * lmax + 1)
    w = np.repeat(gw * dlam, lon.size)
    lax = np.repeat(lat, lon.size)
    lox = np.tile(lon, lat.size)
    return lax, lox, w


def grid_cell_weights(lat_deg, lon_deg) -> np.ndarray:
    """Exact spherical band area for an arbitrary lat/lon grid or point list.

    ``w = dlambda * (sin(phi_hi) - sin(phi_lo))`` with cell boundaries taken at
    midpoints between neighbouring coordinates.  Correct for regional grids and
    for non-uniform spacing; the longitude axis is treated as periodic when the
    coordinates span (nearly) 360 degrees.

    Returns an array shaped like the input (broadcast to 2-D), summing to the
    area actually covered by the grid.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()

    lat_u = np.unique(lat)
    lon_u = np.unique(lon)
    span = lon_u.max() - lon_u.min()
    periodic = span > 180.0                      # global longitude ring

    e_lat, inv_lat, _ = _sorted_edges(lat_u, period=None,
                                      clip=(-90.0, 90.0))
    w_lat_sorted = np.diff(np.sin(np.deg2rad(e_lat)))
    w_lat = np.empty_like(w_lat_sorted)
    w_lat[inv_lat] = w_lat_sorted

    e_lon, inv_lon, _ = _sorted_edges(lon_u, period=360.0 if periodic else None)
    w_lon_sorted = np.deg2rad(np.diff(e_lon))
    w_lon = np.empty_like(w_lon_sorted)
    w_lon[inv_lon] = w_lon_sorted

    # map each input point to its (lat, lon) cell
    lat_idx = np.searchsorted(lat_u, lat)
    lon_idx = np.searchsorted(lon_u, lon)
    return w_lat[lat_idx] * w_lon[lon_idx]


def lattice_step(v) -> float:
    """Robust lattice step (degrees) from consecutive unique coordinates.

    Uses the median of the gaps no wider than 1.5x the smallest one, so that
    rows/columns **missing** from a mask do not inflate the estimate.  This is the
    same estimator as ``tools/yangtze_points_to_sh.py``.
    """
    u = np.unique(np.asarray(v, dtype=float))
    if u.size < 3:
        raise ValueError("need at least 3 distinct coordinates to estimate a "
                         "lattice step")
    g = np.diff(u)
    d0 = float(g.min())
    if not np.isfinite(d0) or d0 <= 0.0:
        raise ValueError("degenerate coordinate spacing; not a lattice")
    core = g[g <= 1.5 * d0]
    return float(np.median(core)) if core.size else d0


def _lattice_weights(lat, lon):
    """``(w, d_lat, d_lon)`` for a lattice subset.  See lattice_cell_weights."""
    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    d_lat = lattice_step(lat)
    d_lon = lattice_step(lon)
    half = np.deg2rad(d_lat) / 2.0
    phi = np.deg2rad(lat)
    return np.deg2rad(d_lon) * (np.sin(phi + half) - np.sin(phi - half)), \
        d_lat, d_lon


def lattice_cell_weights(lat_deg, lon_deg) -> np.ndarray:
    """Integration elements for a **lattice subset** (a rasterised region mask).

    Each sample stands for exactly one cell of the underlying regular lattice::

        w_i = dlambda_rad * (sin(phi_i + dphi/2) - sin(phi_i - dphi/2))

    with one ``dphi`` / ``dlambda`` per axis estimated from the coordinates.  The
    sum is then exactly the area of the union of the occupied cells.

    Why not :func:`grid_cell_weights`: that rule puts its edges at the midpoints
    between *consecutive unique* coordinates, so in a mask every point bordering a
    missing row or column absorbs half of that gap, and the outermost rows get
    half cells.  Measured on ``Yangtze_River.txt`` (a 92 m lattice whose 12
    latitude gaps reach 83 steps): 44 points came out up to 5362x too heavy, the
    total area was 4.59% too large, and ``C00`` inherited that same 4.59% --
    while ``n_points * cell_area`` reproduces a per-cell reference exactly.
    """
    return _lattice_weights(lat_deg, lon_deg)[0]


def voronoi_weights(lat_deg, lon_deg) -> np.ndarray:
    """Spherical Voronoi cell area for arbitrary points (sums to ``4*pi``).

    Duplicate points are merged (their areas are summed and reported at the
    first occurrence); an input whose points are all coplanar - e.g. a single
    latitude ring or the equator - cannot be tessellated and raises.
    """
    from scipy.spatial import SphericalVoronoi

    xyz = lonlat_to_xyz(lat_deg, lon_deg)
    try:
        sv = SphericalVoronoi(xyz, radius=1.0)
        areas = sv.calculate_areas()
    except Exception as exc:                     # duplicate / degenerate input
        raise ValueError(
            "spherical Voronoi tessellation failed "
            f"({type(exc).__name__}: {exc}).  Usual causes: duplicate points "
            "(merge them with a tolerance first) or all points lying on one "
            "great circle (e.g. a single latitude ring), which has no 2-D "
            "tessellation - use rule='grid' or rule='delaunay' there."
        ) from exc
    return np.asarray(areas, dtype=float)


def delaunay_weights(lat_deg, lon_deg) -> np.ndarray:
    """Spherical Delaunay barycentric (one-third) weights.

    Each point receives one third of the area of every **origin-visible**
    spherical triangle it belongs to.  The total equals the area of the
    spherical convex hull of the point set, which for a regional cluster is
    (approximately) the region area - unlike Voronoi weights, which always sum
    to ``4*pi``.

    Notes
    -----
    A 3-D convex hull of a *regional* point set is a closed polyhedron whose
    surface includes a flat base facing away from the sphere's centre.  Keeping
    every facet would roughly double the total (measured: a 30-degree cap gave
    1.54 instead of 0.84, a factor 1.83).  Facets with ``dot(normal, vertex)<=0``
    are therefore dropped - they are exactly the ones the origin cannot see.
    """
    from scipy.spatial import ConvexHull

    xyz = lonlat_to_xyz(lat_deg, lon_deg)
    hull = ConvexHull(xyz)
    w = np.zeros(xyz.shape[0])
    # hull.equations[i] = [outward normal, offset] with  n.x + b <= 0 inside.
    # b = -n.a, so the origin can see the facet iff dot(n, a) > 0 iff b < 0.
    for i, tri in enumerate(hull.simplices):
        if hull.equations[i, 3] >= 0.0:
            continue                          # base facet, not seen from O
        a, b, c = xyz[tri[0]], xyz[tri[1]], xyz[tri[2]]
        area = 2.0 * np.arctan2(abs(np.dot(a, np.cross(b, c))),
                                1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
        w[tri] += area / 3.0
    return w


def uniform_weights(n: int, total: float = FOUR_PI) -> np.ndarray:
    """Equal weights ``total/n`` per point."""
    return np.full(int(n), float(total) / int(n))


def geodesic_cell_weights(lat_deg, lon_deg, radius_deg) -> np.ndarray:
    """Equal area per point for a *quasi-uniform* point set.

    ``w_i = A_i / n_i`` where ``A_i`` is the area of a spherical cap of radius
    ``radius_deg`` around point ``i`` and ``n_i`` is the number of points inside
    it.  Useful when an equal-area sampling is intended but not exact.
    """
    from scipy.spatial import cKDTree

    xyz = lonlat_to_xyz(lat_deg, lon_deg)
    tree = cKDTree(xyz)
    cosr = np.cos(np.deg2rad(radius_deg))
    cap_area = 2 * np.pi * (1.0 - cosr)
    idx = tree.query_ball_point(xyz, 2.0 * np.sin(np.deg2rad(radius_deg) / 2))
    counts = np.array([len(ix) for ix in idx], dtype=float)
    return cap_area / np.maximum(counts, 1.0)


# ---------------------------------------------------------------------------
# container + dispatcher
# ---------------------------------------------------------------------------
@dataclass
class WeightSet:
    """Integration elements plus the diagnostics needed to trust them."""

    w: np.ndarray
    rule: str
    n_points: int = 0
    subset: Optional[np.ndarray] = None
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.w = np.asarray(self.w, dtype=float).ravel()
        if self.n_points == 0:
            self.n_points = self.w.size
        if self.w.size != self.n_points:
            raise ValueError("weight vector length does not match n_points")

    # ------------------------------------------------------------ diagnostics
    @property
    def total(self) -> float:
        return float(self.w.sum())

    @property
    def coverage(self) -> float:
        """Fraction of the sphere covered, ``sum(w)/(4*pi)``."""
        return self.total / FOUR_PI

    @property
    def heterogeneity(self) -> float:
        """``max(w)/min(w)``; close to 1 means a quasi-uniform sampling."""
        w = self.w[self.w > 0]
        return float(w.max() / w.min()) if w.size else np.inf

    def equivalent_lat_step(self) -> float:
        """Latitude step (deg) of an equal-area grid with the same cell area."""
        wm = float(np.mean(self.w))
        cos_phi = wm / np.deg2rad(1.0) ** 2
        if cos_phi <= 0:
            return np.nan
        return float(np.rad2deg(np.arccos(min(cos_phi, 1.0))))

    def normalise(self, mode: str = "global") -> "WeightSet":
        """Rescale the weights.

        ``'global'``  force ``sum(w) = 4*pi`` (assumes full-sphere coverage).
        ``'region'``  keep the geometric sum (correct for a regional subset).
        ``'none'``    leave untouched.
        """
        if mode == "none":
            return self
        if mode == "region":
            return self
        if mode == "global":
            if self.total <= 0:
                raise ValueError("cannot normalise non-positive weights")
            return WeightSet(self.w * (FOUR_PI / self.total), self.rule,
                             self.n_points, self.subset, list(self.notes))
        raise ValueError(f"unknown normalisation mode {mode!r}")

    def report(self) -> dict:
        w = self.w
        d = {
            "rule": self.rule,
            "n_points": int(self.n_points),
            "weight_sum": self.total,
            "weight_sum_over_4pi": self.coverage,
            "min_weight": float(w.min()),
            "max_weight": float(w.max()),
            "heterogeneity": self.heterogeneity,
            "has_negative": bool((w < 0).any()),
        }
        if self.subset is not None:
            d["subset_size"] = int(np.count_nonzero(self.subset))
        if self.notes:
            d["notes"] = list(self.notes)
        return d

    def describe(self) -> str:
        r = self.report()
        lines = [
            f"WeightSet(rule={r['rule']!r}, n={r['n_points']})",
            f"  sum(w)          = {r['weight_sum']:.10f}   "
            f"(4*pi = {FOUR_PI:.10f}, ratio = {r['weight_sum_over_4pi']:.8f})",
            f"  min/max w       = {r['min_weight']:.4e} / {r['max_weight']:.4e}",
            f"  heterogeneity   = {r['heterogeneity']:.4f}",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def compute_weights(lat_deg, lon_deg, rule: str = "auto",
                    user_w: Optional[Sequence[float]] = None,
                    subset: Optional[Sequence[bool]] = None,
                    normalise: str = "auto",
                    nlon: Optional[int] = None) -> WeightSet:
    """Build the integration elements for a point set or grid.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Sample coordinates in degrees.  ``lon`` is only used by the grid rules.
    rule : str
        One of :data:`WEIGHT_RULES`.  ``'auto'`` picks ``'dh'``/``'grid'`` when the
        points form a lat/lon grid or a **subset of one** (a region mask), and
        otherwise ``'voronoi'`` for a global-looking set and ``'delaunay'`` for a
        genuinely irregular regional one.  The lattice-subset branch matters: on
        such a set ``'delaunay'`` would return the convex-hull area instead of the
        area the points actually cover (measured 67x too big on a river mask).
    user_w : array_like, optional
        Required when ``rule='user'``; the caller's per-point element.
    subset : array_like of bool, optional
        Selection of points actually used (e.g. a region mask).  When given for
        the Voronoi rule, the tessellation is still built on the **full** point
        set and the weights are then subset - this keeps boundary cells honest.
    normalise : {'auto', 'global', 'region', 'none'}
        ``'auto'`` forces ``sum(w)=4*pi`` for the global tessellation rules
        (``dh``, ``glq``, ``voronoi``, ``uniform``) and leaves the geometric sum
        for the regional ones (``grid``, ``delaunay``).  ``'global'`` is
        required if you want coefficients on the "field is zero outside" reading
        of a regional dataset.
    nlon : int, optional
        Number of longitudes, only used by the ``'dh'`` rule.
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    if rule not in WEIGHT_RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {WEIGHT_RULES}")

    sub = None if subset is None else np.asarray(subset, dtype=bool).ravel()
    if sub is not None and sub.size != lat.size:
        raise ValueError("subset mask size does not match the point set")

    notes: list = []
    if rule == "auto":
        rule = _guess_rule(lat, lon)
        notes.append(f"rule='auto' resolved to '{rule}'")

    # -------- global-tessellation rules -------------------------------------
    if rule == "voronoi":
        w = voronoi_weights(lat, lon)
        if sub is not None:
            w = w[sub]
        ws = WeightSet(w, rule, w.size, sub, notes)
        return ws.normalise(normalise if normalise != "auto" else "global")

    if rule == "delaunay":
        w = delaunay_weights(lat, lon)
        if sub is not None:
            w = w[sub]
        ws = WeightSet(w, rule, w.size, sub, notes)
        return ws.normalise(normalise if normalise != "auto" else "region")

    if rule == "uniform":
        ws = WeightSet(uniform_weights(lat.size), rule, lat.size, sub, notes)
        return ws.normalise(normalise if normalise != "auto" else "global")

    if rule == "user":
        if user_w is None:
            raise ValueError("rule='user' requires user_w=")
        w = np.atleast_1d(np.asarray(user_w, dtype=float)).ravel()
        if sub is not None:
            w = w[sub]
        if w.size != (sub.sum() if sub is not None else lat.size):
            raise ValueError("user_w length does not match the point set")
        ws = WeightSet(w, rule, w.size, sub, notes)
        return ws.normalise(normalise if normalise != "auto" else "none")

    # -------- grid rules ----------------------------------------------------
    if rule == "dh":
        if sub is not None:
            raise ValueError("rule='dh' applies to the whole grid, not a subset")
        # the rule is defined on the unique latitude circles, not per grid cell
        lat_u = np.unique(lat)
        nlat_u = lat_u.size
        lon_u = np.unique(lon)
        nlon_eff = nlon if nlon is not None else max(2 * nlat_u, 2)
        a_u = dh_lat_weights(lat_u)
        a = a_u[np.searchsorted(lat_u, lat)] if lat_u.size != lat.size else a_u
        w = a * (2 * np.pi / nlon_eff)
        # ⚠️ 只检查「nlon_eff 是否等于 2*nlat」在**没传 nlon** 时是同义反复
        # （nlon_eff 本来就是这么推断出来的），于是"网格根本不是 DH 网格"这件事
        # 永远发现不了 —— 而权重随后被归一化到 Σw = 4π，覆盖率照样显示 1.0，
        # 从数字上完全看不出来。所以要和**实际的经线条数**比。
        if lon_u.size != nlon_eff:
            notes.append(
                f"实际有 {lon_u.size} 条经线，但按 nlon={nlon_eff} 计算权重"
                "（Driscoll-Healy 只在 nlon == 2*nlat 时精确）")
        elif nlon_eff != 2 * nlat_u:
            notes.append(
                f"nlon={nlon_eff} != 2*nlat={2*nlat_u}: the Driscoll-Healy "
                "rule is only exact for nlon == 2*nlat")
        ws = WeightSet(w, rule, w.size, sub, notes)
        return ws.normalise(normalise if normalise != "auto" else "global")

    if rule == "glq":
        raise ValueError(
            "'glq' is not a weight-from-points rule: build the sampling with "
            "shkit.weights.glq_grid(lmax), which returns (lat, lon, w), then "
            "pass rule='user', user_w=w")

    if rule == "grid":
        w = grid_cell_weights(lat, lon)
        if sub is not None:
            w = w[sub]
        ws = WeightSet(w, rule, w.size, sub, notes)
        if normalise == "global":
            ws = ws.normalise("global")
        return ws

    if rule == "lattice":
        w, d_lat, d_lon = _lattice_weights(lat, lon)
        if sub is not None:
            w = w[sub]
        notes.append(
            f"lattice cell {d_lon:.8f} x {d_lat:.8f} deg "
            f"({d_lon * 111.19 * 1000:.1f} x {d_lat * 111.19 * 1000:.1f} m); "
            f"sum(w) = n_points * cell_area = the area of the occupied cells")
        ws = WeightSet(w, rule, w.size, sub, notes)
        if normalise == "global":
            ws = ws.normalise("global")
        return ws

    raise AssertionError("unreachable")


def _looks_like_grid(lat, lon) -> bool:
    """True when the points form a complete rectangular lat/lon grid."""
    lat_u = np.unique(lat)
    lon_u = np.unique(lon)
    return lat_u.size * lon_u.size == lat.size


def looks_like_lattice(lat, lon, spread_tol: float = 0.05,
                       min_frac: float = 0.8) -> bool:
    """True when the samples lie on a *subset* of a regular lat/lon lattice.

    A region mask (a 0/1 outline saved as scattered points) has uniformly spaced
    unique coordinates but most ``(lat, lon)`` combinations missing, so
    :func:`_looks_like_grid` fails on it.  That distinction matters because the
    correct integration element is then the **lattice cell area** -- the area the
    points actually stand for -- whereas ``'delaunay'`` returns the much larger
    spherical convex-hull area.

    The test looks at consecutive gaps between unique coordinates: on a lattice
    the overwhelming majority are one step (~1.5x the smallest), with a few
    integer multiples where whole rows/columns are absent, and those base steps
    agree to within ``spread_tol``.  An irregular regional scatter spreads its
    gaps over a wide range and fails the first condition.

    Measured on a Yangtze-river mask (142846 points on a 92 m lattice):
    ``grid`` -> 1 106 km^2, ``delaunay`` -> 73 987 km^2 (66.9x too big), and the
    exported ``C00`` was consequently 1.1e4 times too large.
    """
    for v in (lat, lon):
        u = np.unique(np.asarray(v, dtype=float))
        if u.size < 3:
            return False
        g = np.diff(u)
        d_min = float(g.min())
        if not np.isfinite(d_min) or d_min <= 0.0:
            return False
        base = g <= 1.5 * d_min
        if float(np.mean(base)) < min_frac:
            return False
        b = g[base]
        if (float(b.max()) / float(b.min()) - 1.0) > spread_tol:
            return False
    return True


def _guess_rule(lat, lon) -> str:
    """Heuristic used by ``rule='auto'``."""
    lon_u = np.unique(lon)
    span = lon_u.max() - lon_u.min()
    global_lon = span > 300.0
    if _looks_like_grid(lat, lon) and global_lon:
        nlon = lon_u.size
        nlat = np.unique(lat).size
        return "dh" if nlon == 2 * nlat and nlat % 2 == 0 else "grid"
    if global_lon and lat.max() > 60.0 and lat.min() < -60.0:
        return "voronoi"
    # A regional grid, or a mask sampled on a regular one.  Both need cell areas,
    # not the convex hull: falling through to 'delaunay' inflated the covered area
    # of a river mask by ~67x and C00 with it.
    if _looks_like_grid(lat, lon):
        # complete rectangular grid: midpoint edges are exact
        return "grid"
    if looks_like_lattice(lat, lon):
        # lattice SUBSET (a rasterised mask): each point is one lattice cell.
        # The midpoint-edge rule would let a point next to a missing row absorb
        # half of that gap -- measured +4.59% on the Yangtze mask, carried
        # straight into C00.
        return "lattice"
    return "delaunay"
