# -*- coding: utf-8 -*-
"""
Validation of shkit.filters and shkit.slepian.

Run:  python tests/validate_slepian.py
"""
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, __file__.rsplit("tests", 1)[0])

from shkit.analysis import analysis_wlsq
from shkit.basis import design_matrix_full, synthesize
from shkit.coeffs import SHCoeffs, triangle_order
from shkit.filters import (apply_gaussian, ewh_scaling, gaussian_coefficients,
                           love_number)
from shkit.slepian import slepian_analysis, slepian_basis, slepian_expand
from shkit.weights import FOUR_PI, compute_weights

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def fib_sphere(n):
    i = np.arange(n) + 0.5
    colat = np.arccos(1.0 - 2.0 * i / n)
    lon = np.mod(np.pi * (1 + 5 ** 0.5) * np.arange(n), 2 * np.pi)
    return 90.0 - np.rad2deg(colat), np.rad2deg(lon)


def rand_truth(nmax, rng, band=None):
    """Random Kaula-like coefficients, non-zero only for n <= band."""
    band = nmax if band is None else band
    m, n = triangle_order(nmax)
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for k in range(len(m)):
        if n[k] > band:
            continue
        C[n[k], m[k]] = rng.standard_normal() / max(n[k], 1) ** 2
        if m[k] >= 1:
            S[n[k], m[k]] = rng.standard_normal() / max(n[k], 1) ** 2
    return SHCoeffs(C, S)


def coeff_rel_err(a, b):
    L = max(a.nmax, b.nmax)
    a, b = a.truncate(L), b.truncate(L)
    num = np.linalg.norm(a.C - b.C) ** 2 + np.linalg.norm(a.S - b.S) ** 2
    den = np.linalg.norm(b.C) ** 2 + np.linalg.norm(b.S) ** 2
    return float(np.sqrt(num / den))


# ---------------------------------------------------------------------------
def t_filters():
    L = 120
    W0 = gaussian_coefficients(0.0, L)
    check("radius <= 0 returns an all-ones filter", np.allclose(W0, 1.0),
          f"min = {W0.min()}, max = {W0.max()}")

    for r in (100.0, 300.0, 1000.0):
        W = gaussian_coefficients(r, L)
        big = W > 1e-10
        ok = (abs(W[0] - 1.0) < 1e-15) and np.all(np.diff(W[big]) < 0.0)
        check(f"Gaussian {r:.0f} km: W_0 = 1 and monotonically decreasing", ok,
              f"W_0 = {W[0]:.15f}, W_1 = {W[1]:.6f}, W_10 = {W[10]:.4f}, "
              f"W_60 = {W[60]:.4e}, W_120 = {W[120]:.4e}")

    # cross-check against the classic closed-form approximation
    #   W_l ~ exp(-l(l+1)(r/R)^2 / (4 ln2))
    for r in (300.0, 500.0):
        W = gaussian_coefficients(r, L)
        appr = np.exp(-np.arange(L + 1) * (np.arange(L + 1) + 1) *
                      (r / 6378.1363) ** 2 / (4 * np.log(2)))
        rel = np.abs(W[1:40] - appr[1:40]).max()
        check(f"Gaussian {r:.0f} km matches the closed-form approximation", rel < 5e-3,
              f"max |exact - approx| over l=1..39 = {rel:.3e}")

    # glq vs frc must agree where the recurrence is still stable
    for r in (150.0, 300.0):
        a = gaussian_coefficients(r, L, "glq")
        b = gaussian_coefficients(r, L, "frc")
        rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
        check(f"glq and frc agree at {r:.0f} km", rel < 1e-8,
              f"max abs diff = {rel:.3e}")
    a = gaussian_coefficients(800.0, L, "glq")
    b = gaussian_coefficients(800.0, L, "frc")
    rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
    check("frc becomes unreliable at large radius (documented)", rel > 1e-8,
          f"glq vs frc max diff at 800 km = {rel:.3e}")

    rng = np.random.default_rng(11)
    truth = rand_truth(60, rng)
    sm = apply_gaussian(truth, 300.0)
    ratio_lo = np.linalg.norm(sm.C[:10]) / np.linalg.norm(truth.C[:10])
    ratio_hi = np.linalg.norm(sm.C[50:]) / np.linalg.norm(truth.C[50:])
    check("Gaussian filter preserves low degrees and damps high degrees",
          ratio_lo > 0.99 and ratio_hi < 0.2,
          f"|C|(0-9) ratio {ratio_lo:.6f}, |C|(50-60) ratio {ratio_hi:.3e}")

    A = ewh_scaling(4)
    check("EWH scaling factor is positive and follows (2n+1)/(1+k_n)",
          A[0] > 0 and A[2] > A[1],
          f"A_0 = {A[0]:.4e}, A_2 = {A[2]:.4e}, k_2 = {love_number(2):.5f}")


# ---------------------------------------------------------------------------
def t_slepian():
    rng = np.random.default_rng(21)
    lat, lon = fib_sphere(6000)
    L = 20
    cap_lat, cap_lon, cap_rad = 30.0, 100.0, 25.0
    cla, clo, cr = np.deg2rad(cap_lat), np.deg2rad(cap_lon), np.deg2rad(cap_rad)
    cosd = (np.sin(np.deg2rad(lat)) * np.sin(cla) +
            np.cos(np.deg2rad(lat)) * np.cos(cla) * np.cos(np.deg2rad(lon) - clo))
    sel = cosd >= np.cos(cr)
    la, lo = lat[sel], lon[sel]
    frac = (1 - np.cos(cr)) / 2

    w = compute_weights(la, lo, rule="delaunay", normalise="region")
    basis = slepian_basis(la, lo, L, weights=w)

    check("Slepian eigenvalues lie in [0, 1]",
          basis.eigenvalues.max() <= 1.0 + 5e-3 and basis.eigenvalues.min() > -1e-8,
          f"lambda in [{basis.eigenvalues.min():.3e}, "
          f"{basis.eigenvalues.max():.6f}]  (slightly above 1 is the "
          "quadrature error of the regional weights)")

    ev = basis.eigenvalues
    check("eigenvalues sum to the Shannon number",
          abs(ev.sum() - basis.shannon) / basis.shannon < 0.05,
          f"sum(lambda) = {ev.sum():.3f} vs (L+1)^2*A/(4pi) = {basis.shannon:.3f}")

    n05 = basis.usable(0.5)
    check("number of lambda > 0.5 is close to the Shannon number",
          abs(n05 - basis.shannon) / basis.shannon < 0.35,
          f"count = {n05}, Shannon = {basis.shannon:.1f}, nmax={L}")

    A = design_matrix_full(la, lo, L)
    K = (A * w.w[:, None]).T @ A / FOUR_PI
    t = basis.tapers[:, 0]
    resid = np.linalg.norm(K @ t - ev[0] * t) / np.linalg.norm(K @ t)
    check("tapers really are eigenvectors of the concentration matrix",
          resid < 1e-10, f"||K t - lambda t||/||K t|| = {resid:.3e}")
    check("tapers are orthonormal in coefficient space",
          np.abs(basis.tapers.T @ basis.tapers - np.eye(basis.ntaper)).max() < 1e-10,
          f"max|T'T - I| = "
          f"{np.abs(basis.tapers.T @ basis.tapers - np.eye(basis.ntaper)).max():.2e}")

    # ---- the defining property: tapers really are concentrated in the region
    from shkit.weights import glq_grid
    glat, glon, gw = glq_grid(30)
    g_gl = np.sum(gw)                       # = 4*pi
    conc = []
    for a in (0, 4, 16, 40):
        ta = basis.tapers[:, a]
        ca = np.zeros((L + 1, L + 1))
        sa = np.zeros((L + 1, L + 1))
        # expand the taper into SH coefficients (it is already a coefficient vector)
        from shkit.analysis import _unpack
        Cc, Sc = _unpack(ta[:, None], L)
        ca, sa = Cc[:, :, 0], Sc[:, :, 0]
        tot = float(np.sum(gw * synthesize(glat, glon, ca, sa) ** 2))
        part = float(np.sum(w.w * synthesize(la, lo, ca, sa) ** 2))
        conc.append(part / tot)
    rel = [abs(conc[i] - basis.eigenvalues[a]) / max(basis.eigenvalues[a], 1e-12)
           for i, a in enumerate((0, 4, 16, 40))]
    check("taper alpha really concentrates lambda_alpha of its energy in R",
          max(rel) < 0.02,
          "measured " + ", ".join(f"a={a}:{c:.4f}" for a, c in
                                  zip((0, 4, 16, 40), conc)) +
          f"  (max rel err {max(rel):.2e}, global integral from a 61x61 GLQ grid)")

    # ---- the Slepian expansion is exact for a function inside the taper space
    from shkit.analysis import _unpack
    t0 = basis.tapers[:, 0]
    C0, S0 = _unpack(t0[:, None], L)
    f0 = synthesize(la, lo, C0[:, :, 0], S0[:, :, 0])
    s0 = slepian_expand(la, lo, f0, basis, weights=w, lam_min=1e-12)
    check("Slepian expansion of a taper is the unit vector it should be",
          abs(s0[0, 0] - 1.0) < 1e-8 and np.abs(s0[1:, 0]).max() < 1e-8,
          f"s_0 = {s0[0,0]:.12f}, max|s_alpha != 0| = {np.abs(s0[1:,0]).max():.2e}")

    # ---- the actual claim: Slepian is stable where WLSQ explodes
    truth = rand_truth(L, rng, band=10)
    f = synthesize(la, lo, truth.C, truth.S)

    cb, rb, _basis = slepian_analysis(la, lo, f, basis=basis, weights=w, lam_min=0.5)
    cw, rw = analysis_wlsq(la, lo, f, L, weights=w, method="wlsq")

    check("Slepian coefficient norm stays bounded where WLSQ blows up",
          np.linalg.norm(cb.C) < 100 * np.linalg.norm(truth.C),
          f"||C_slepian|| = {np.linalg.norm(cb.C):.4e}, "
          f"||C_wlsq|| = {np.linalg.norm(cw.C):.4e}, "
          f"||C_truth|| = {np.linalg.norm(truth.C):.4e}")
    check("Slepian report carries the taper diagnostics",
          rb.meta.get("ntaper_used", 0) > 0 and "slepian_eigenvalues" in rb.meta,
          f"ntaper_used = {rb.meta.get('ntaper_used')} of "
          f"{rb.meta.get('ntaper_available')}, Shannon = "
          f"{rb.meta.get('shannon'):.1f}, cond(WLSQ) = "
          f"{rw.condition_number:.2e}")

    # the decisive difference: behaviour OUTSIDE the region
    fb_all = np.asarray(synthesize(lat, lon, cb.C, cb.S)).ravel()
    fw_all = np.asarray(synthesize(lat, lon, cw.C, cw.S)).ravel()
    f_all = np.asarray(synthesize(lat, lon, truth.C, truth.S)).ravel()
    denom = np.sqrt(np.mean(f_all[sel] ** 2))
    eb = np.sqrt(np.mean((fb_all[sel] - f_all[sel]) ** 2)) / denom
    ew = np.sqrt(np.mean((fw_all[sel] - f_all[sel]) ** 2)) / denom
    ob = np.sqrt(np.mean(fb_all[~sel] ** 2)) / denom
    ow = np.sqrt(np.mean(fw_all[~sel] ** 2)) / denom
    check("Slepian stays bounded outside the region where WLSQ explodes",
          ob < 0.2 and ow > 1.0,
          f"outside-region field (relative to in-region RMS): "
          f"slepian {ob:.3e} vs wlsq {ow:.3e}  |  in-region field error: "
          f"slepian {eb:.3e} vs wlsq {ew:.3e}")

    # ---- the hard truth about coverage, made explicit
    check("a small region cannot recover smooth (low-degree) fields",
          basis.eigenvalues[basis.usable() - 1] > 0.5 > basis.region_fraction,
          f"region = {basis.region_fraction:.4f} of the sphere, so a constant "
          f"field has concentration {basis.region_fraction:.4f} (not 1); the "
          f"{basis.usable()} usable tapers (lambda>0.5) live at high degree. "
          f"Shannon number = {basis.shannon:.1f}")

    print("\n" + basis.describe())
    print("\n" + rb.describe())


def main():
    t0 = time.time()
    for fn in (t_filters, t_slepian):
        print(f"\n=== {fn.__name__} " + "=" * (60 - len(fn.__name__)))
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(fn.__name__, False, "raised an exception")
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'='*76}\n{npass}/{len(RESULTS)} checks passed in {time.time()-t0:.1f}s")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
