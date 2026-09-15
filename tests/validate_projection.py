# -*- coding: utf-8 -*-
"""Zero-filled global projection -- the objective for regional / mask data.

Covers the two things that were wrong before this suite existed:

1. ``rule='auto'`` sent a **mask sampled on a regular lattice** to ``'delaunay'``,
   which reports the spherical convex-hull area instead of the area the points
   stand for.  On a river mask that inflated the covered area 67x and put ``C00``
   1.1e4 times too large.
2. ``method='auto'`` then escalated a regional dataset to an unregularised
   least-squares solve, whose objective is defined at the samples only.  Its
   sample residual collapsed to ~1e-6 while ``C00`` drifted to a physically
   impossible value -- ``C00`` is the field's mean over the sphere, so for a 0/1
   mask it *must* equal the covered-area fraction.

Run:  python tests/validate_projection.py
"""
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit import weights as W                                   # noqa: E402
from shkit.analysis import (analysis, analysis_projection,        # noqa: E402
                            analysis_quadrature, analysis_wlsq)
from shkit.basis import design_matrix_full, synthesize          # noqa: E402
from shkit.coeffs import SHCoeffs                               # noqa: E402
from shkit.diagnostics import gram_matrix                       # noqa: E402
from shkit.filters import EARTH_RADIUS_M as R                   # noqa: E402
from shkit.weights import (compute_weights, glq_grid,        # noqa: E402
                           lattice_cell_weights, looks_like_lattice)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:60s} {detail}")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def lattice_mask(d=0.005):
    """A 0/1 mask sampled on a regular lattice: a blob plus a filament.

    Structurally the same as ``Yangtze_River.txt`` -- uniformly spaced unique
    coordinates with most (lat, lon) combinations absent.
    """
    la = np.linspace(30.0, 30.0 + 199 * d, 200)
    lo = np.linspace(106.0, 106.0 + 399 * d, 400)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    blob = (LA - 30.12) ** 2 + (LO - 106.30) ** 2 <= 0.12 ** 2
    fil = np.abs(LA - (30.05 + 0.20 * np.sin((LO - 106.0) * 6.0))) < 0.008
    keep = (blob | fil).ravel()
    return LA.ravel()[keep], LO.ravel()[keep], d


def global_cap(dlat=0.5, dlon=0.5, rad=20.0):
    nlat, nlon = 360, 720
    la = np.linspace(-89.75, 89.75, nlat)
    lo = np.linspace(0.25, 359.75, nlon)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    cosd = (np.sin(np.deg2rad(LA)) * np.sin(np.deg2rad(rad)) +
            np.cos(np.deg2rad(LA)) * np.cos(np.deg2rad(rad)) *
            np.cos(np.deg2rad(LO - 110.0)))
    return (LA.ravel(), LO.ravel(),
            (cosd >= np.cos(np.deg2rad(rad))).astype(float).ravel())


def fib_sphere(n):
    i = np.arange(n) + 0.5
    colat = np.arccos(1.0 - 2.0 * i / n)
    lam = np.mod(np.pi * (1 + 5 ** 0.5) * np.arange(n), 2 * np.pi)
    return 90.0 - np.rad2deg(colat), np.rad2deg(lam)


# ---------------------------------------------------------------------------
# 1. the integration element for a lattice subset
# ---------------------------------------------------------------------------
def t_lattice_rule():
    lat, lon, d = lattice_mask()
    n = lat.size
    check("mask points lie on a regular lattice (looks_like_lattice)",
          looks_like_lattice(lat, lon), f"{n} points, step {d} deg")

    fla, flo = fib_sphere(2000)
    cla, clo, cr = np.deg2rad(30.0), np.deg2rad(100.0), np.deg2rad(25.0)
    cosd = (np.sin(np.deg2rad(fla)) * np.sin(cla) +
            np.cos(np.deg2rad(fla)) * np.cos(cla) *
            np.cos(np.deg2rad(flo) - clo))
    cap = cosd >= np.cos(cr)
    check("an irregular regional scatter is NOT taken for a lattice",
          not looks_like_lattice(fla[cap], flo[cap]),
          f"{int(cap.sum())} cap points")

    ws = compute_weights(lat, lon, rule="auto")
    check("rule='auto' picks 'lattice' for a lattice mask",
          ws.rule == "lattice", f"rule = {ws.rule}")

    wd = compute_weights(lat, lon, rule="delaunay")
    ratio = wd.total / ws.total
    check("'lattice' area is the cell count x cell area, 'delaunay' the hull",
          ratio > 3.0,
          f"hull/cells = {ratio:.1f}x ({ws.total*R**2/1e6:,.0f} vs "
          f"{wd.total*R**2/1e6:,.0f} km^2; on the Yangtze river mask it is 67x)")

    # each point is one lattice cell, so the sum is exact -- not a midpoint
    # approximation.  ('grid' gets this wrong: a point next to a missing row
    # absorbs half of that gap.  Measured +4.59% on the Yangtze mask.)
    expect = (lattice_cell_weights(lat, lon))
    check("'lattice' sum == n_points x cell area (exact)",
          abs(ws.total / expect.sum() - 1.0) < 1e-12,
          f"ratio {ws.total/expect.sum():.15f}")
    wg = compute_weights(lat, lon, rule="grid")
    check("'grid' would over-count when rows/columns are missing",
          wg.total > ws.total * 1.001,
          f"grid/lattice = {wg.total/ws.total:.6f}")

    # the library must agree with the hand-built reference in tools/
    ref = os.path.join(ROOT, "examples", "yangtze_points_weights.csv")
    yat = os.path.join(ROOT, "Yangtze_River.txt")
    if os.path.exists(yat) and os.path.exists(ref):
        p = np.genfromtxt(yat, delimiter="\t", names=True, encoding="utf-8")
        w_ref = np.asarray(np.genfromtxt(ref, delimiter=",", names=True,
                                         encoding="utf-8")["weight_sr"], float)
        w_lat = lattice_cell_weights(np.asarray(p["latitude"], float),
                                     np.asarray(p["longitude"], float))
        check("lattice weights reproduce tools/ per-cell reference",
              abs(w_lat.sum() / w_ref.sum() - 1.0) < 1e-12,
              f"ratio {w_lat.sum()/w_ref.sum():.15f}")


# ---------------------------------------------------------------------------
# 2. the objective: C00 is a geometric fact
# ---------------------------------------------------------------------------
def t_c00_is_geometric():
    lat, lon, _d = lattice_mask()
    f = np.ones(lat.size)
    ws = compute_weights(lat, lon, rule="auto")
    frac = ws.total / W.FOUR_PI

    c, rep = analysis_projection(lat, lon, f, 12)
    check("projection C00 == covered area fraction (exact)",
          abs(c.C[0, 0] / frac - 1.0) < 1e-12,
          f"C00 = {c.C[0,0]:.9e}, area fraction = {frac:.9e}")
    check("C00 equals the weighted mean of the data",
          abs(rep.dc_mean_got / rep.dc_mean_expected - 1.0) < 1e-12,
          f"{rep.dc_mean_got:.6e} vs {rep.dc_mean_expected:.6e}")
    check("no C00 inconsistency warning on the projection",
          not any("disagrees with the weighted mean" in m for m in rep.warnings),
          f"{len(rep.warnings)} warning(s)")

    # the truncation cannot reach 1 -- the peak is bounded by the Shannon number
    peak = float(np.abs(np.asarray(synthesize(lat, lon, c.C, c.S, 12))).max())
    shannon = 169 * frac
    check("at L=12 the mask cannot reach 1 (peak ~ Shannon number)",
          peak < 10 * shannon,
          f"peak {peak:.4e}, Shannon {shannon:.4e}")

    # residual is the leakage and is reported area-weighted too
    check("area-weighted residual is reported and is the leakage",
          np.isfinite(rep.residual_rms_weighted) and rep.residual_rms_weighted > 0.5,
          f"unweighted {rep.residual_rms:.6f}, area-weighted "
          f"{rep.residual_rms_weighted:.6f}")

    # leakage decreases with L but never vanishes
    prev = None
    floors = []
    for L in (4, 8, 16):
        _cl, rl = analysis_projection(lat, lon, f, L)
        floors.append(rl.residual_rms_weighted)
    check("the truncation floor decreases with L and stays non-zero",
          floors[0] > floors[1] > floors[2] > 0.0,
          "floors(L=4,8,16) = " + ", ".join(f"{v:.4f}" for v in floors))


# ---------------------------------------------------------------------------
# 3. Slepian / completeness invariant and the bias correction
# ---------------------------------------------------------------------------
def t_completeness_invariant():
    lat, lon, _d = lattice_mask()
    ws = compute_weights(lat, lon, rule="auto")
    for L in (8, 12):
        A = design_matrix_full(lat, lon, L)
        K = gram_matrix(A, ws.w)
        lam = np.linalg.eigvalsh(K)
        shannon = (L + 1) ** 2 * (ws.total / W.FOUR_PI)
        check(f"sum of K eigenvalues == Shannon number (L={L})",
              abs(lam.sum() / shannon - 1.0) < 1e-9,
              f"sum lambda {lam.sum():.6e} vs Shannon {shannon:.6e}")

    # well-determined directions do not exist at this degree -> tau is a no-op
    f = np.ones(lat.size)
    c0, r0 = analysis_projection(lat, lon, f, 12)
    c1, r1 = analysis_projection(lat, lon, f, 12, tau=0.5)
    ce = r1.meta["completeness_eigen"]
    check("lambda_max << 1 at L=12 for a tiny footprint",
          ce["lambda_max"] < 1e-2, f"lambda_max = {ce['lambda_max']:.4e}")
    check("tau=0.5 is a no-op and says why",
          abs(c1.C[0, 0] / c0.C[0, 0] - 1.0) < 1e-12
          and any("corrected nothing" in m for m in r1.warnings),
          f"n_above_tau = {ce['n_above_tau']}")
    check("tau must be in (0, 1]",
          _raises(lambda: analysis_projection(lat, lon, f, 12, tau=0.0)),
          "tau=0 rejected")


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# 4. well-posed case: the projection introduces no bias of its own
# ---------------------------------------------------------------------------
def t_wellposed():
    L = 8
    lat, lon, w = glq_grid(L)
    rng = np.random.default_rng(5)
    Ct = np.zeros((L + 1, L + 1))
    St = np.zeros((L + 1, L + 1))
    for n in range(L + 1):
        for m in range(n + 1):
            Ct[n, m] = rng.standard_normal() / max(n, 1) ** 2
            if m:
                St[n, m] = rng.standard_normal() / max(n, 1) ** 2
    truth = SHCoeffs(Ct, St)
    f = np.asarray(synthesize(lat, lon, truth.C, truth.S)).ravel()

    c, rep = analysis_projection(lat, lon, f, L,
                                 weights=compute_weights(
                                     lat, lon, rule="user", user_w=w,
                                     normalise="none"))
    err = np.abs(c.C - Ct).max()
    check("on an exact quadrature rule the projection recovers the field",
          err < 1e-10, f"max|dC| = {err:.2e}")
    check("K is the identity there, so its eigenvalues sum to ncoef",
          abs(sum(np.linalg.eigvalsh(
              gram_matrix(design_matrix_full(lat, lon, L), w))) - (L + 1) ** 2)
          < 1e-8, f"ncoef = {(L+1)**2}")

    # a global 0/1 cap: correct C00, and a truncation floor that stays put
    glat, glon, F = global_cap()
    wg = compute_weights(glat, glon, rule="dh")
    frac = np.sum(wg.w * F) / W.FOUR_PI
    cg, rg = analysis_projection(glat, glon, F, 12, weights=wg)
    check("global cap projection C00 == cap area fraction",
          abs(cg.C[0, 0] / frac - 1.0) < 1e-8,
          f"ratio {cg.C[0,0]/frac:.10f}")
    rel = rg.residual_rms_weighted / np.sqrt(frac)
    check("global cap leaks ~39% of the field at L=12",
          0.2 < rel < 0.6, f"relative truncation error = {rel:.4f}")
    check("no C00 warning on the global case",
          not any("disagrees with the weighted mean" in m for m in rg.warnings))


# ---------------------------------------------------------------------------
# 5. routing and the guards
# ---------------------------------------------------------------------------
def t_routing_and_guards():
    lat, lon, _d = lattice_mask()
    f = np.ones(lat.size)

    c, rep = analysis(lat, lon, f, 12, method="auto")
    check("method='auto' routes a regional dataset to 'projection'",
          rep.meta.get("auto_choice") == "projection",
          f"picked {rep.meta.get('auto_choice')}")
    check("method='projection' is accepted by analysis()",
          analysis(lat, lon, f, 12, method="projection")[1].method
          == "projection", "")

    ws = compute_weights(lat, lon, rule="auto")
    cw, repw = analysis_wlsq(lat, lon, f, 12, weights=ws, method="wlsq")
    fired = any("disagrees with the weighted mean" in m for m in repw.warnings)
    check("C00 consistency guard fires on the unregularised regional solve",
          fired, f"C00 = {cw.C[0,0]:.4e}, weighted mean = "
                 f"{repw.dc_mean_expected:.4e}, {len(repw.warnings)} warning(s)")
    check("collapsed-residual alarm fires on the same solve",
          any("collapsed" in m for m in repw.warnings), "")
    check("the guard does NOT fire on the projection",
          not any("disagrees with the weighted mean" in m
                  for m in analysis_projection(lat, lon, f, 12)[1].warnings), "")


def main():
    t0 = time.time()
    tests = [t_lattice_rule, t_c00_is_geometric, t_completeness_invariant,
             t_wellposed, t_routing_and_guards]
    for fn in tests:
        print(f"\n=== {fn.__name__} " + "=" * (60 - len(fn.__name__)))
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(fn.__name__, False, "raised an exception")
    npass = sum(1 for ok in RESULTS if ok)
    print(f"\n{'='*76}\n{npass}/{len(RESULTS)} checks passed in {time.time()-t0:.1f}s")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
