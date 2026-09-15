# -*- coding: utf-8 -*-
"""
Smoke tests for the shkit command line front end.

Runs every subcommand in-process and asserts a zero exit code plus a couple of
output artefacts.  Deliberately independent of shkit.io's own behaviour beyond
its public API.

Run:  python tests/test_cli.py
"""
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit.cli import main as cli_main
from shkit import io as SIO
from shkit.coeffs import SHCoeffs
from shkit.synthesis import synthesis

TMP = os.path.join(HERE, "_tmp_cli")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")


def run(argv):
    """Call the CLI, returning (exit_code, raised_exception)."""
    try:
        rc = cli_main(argv)
        return (0 if rc is None else int(rc)), None
    except SystemExit as exc:                      # argparse calls sys.exit
        return (int(exc.code or 0)), None
    except Exception as exc:                       # noqa: BLE001
        return 1, exc


def make_inputs(L=10, N=1500):
    i = np.arange(N) + 0.5
    lat = 90.0 - np.rad2deg(np.arccos(1.0 - 2.0 * i / N))
    lon = np.mod(np.pi * (1 + 5 ** 0.5) * i, 2 * np.pi) * 180.0 / np.pi
    rng = np.random.default_rng(11)
    C = np.zeros((L + 1, L + 1))
    S = np.zeros((L + 1, L + 1))
    for n in range(L + 1):
        for m in range(n + 1):
            C[n, m] = rng.standard_normal() / max(n, 1) ** 2
            if m:
                S[n, m] = rng.standard_normal() / max(n, 1) ** 2
    truth = SHCoeffs(C, S)
    f = synthesis(lat, lon, truth)
    pts = os.path.join(TMP, "points.csv")
    SIO.write_points(pts, lat, lon, f, fmt="%.17g")
    coeffs_path = os.path.join(TMP, "truth.sh")
    SIO.write_coeffs(truth, coeffs_path, layout="triangle")
    return pts, coeffs_path


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    pts, truth_path = make_inputs()
    L = 10

    # ---------------------------------------------------------------- help
    for sub in ("weights", "analyze", "synth", "roundtrip", "glq-grid", "info"):
        rc, exc = run([sub, "-h"])
        check(f"'{sub} -h' exits cleanly", rc == 0 and exc is None,
              f"rc={rc}" + (f", {exc!r}" if exc else ""))

    # ------------------------------------------------------------- weights
    out_w = os.path.join(TMP, "w.csv")
    rc, exc = run(["weights", "--points", pts, "--rule", "voronoi",
                   "--out", out_w])
    check("'weights' runs and writes a weight file",
          rc == 0 and os.path.exists(out_w) and exc is None,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    # ------------------------------------------------------------- analyze
    prefix = os.path.join(TMP, "run1")
    rc, exc = run(["analyze", "--points", pts, "--nmax", str(L),
                   "--out-prefix", prefix])
    produced = [prefix + e for e in (".sh", ".json")]
    check("'analyze' runs and writes coeffs + report",
          rc == 0 and all(os.path.exists(p) for p in produced) and exc is None,
          f"rc={rc}, files={[os.path.basename(p) for p in produced if os.path.exists(p)]}"
          + (f", {exc!r}" if exc else ""))
    if os.path.exists(prefix + ".sh"):
        back = SIO.read_coeffs(prefix + ".sh")
        truth = SIO.read_coeffs(truth_path)
        num = (np.linalg.norm(back.C - truth.C) ** 2 +
               np.linalg.norm(back.S - truth.S) ** 2)
        den = np.linalg.norm(truth.C) ** 2 + np.linalg.norm(truth.S) ** 2
        rel = float(np.sqrt(num / den))
        check("'analyze' recovers the synthetic coefficients (relative L2)",
              rel < 5e-3,
              f"relative L2 error = {rel:.3e} on 1500 scattered points, L=10")

    # ------------------------------------------------ --longitude-fft (A3)
    # 完整矩形网格 + 整圈均匀经度 + nlon > 2*nmax -> auto 走 FFT；
    # 结果必须与强制直接法一致；条件不满足时 'fft' 必须报错而不是回退。
    glat = np.arange(-80.0, 80.01, 10.0)
    glon = np.arange(0.0, 360.0, 10.0)                 # nlon=36 > 2*8
    GLA, GLO = map(np.ravel, np.meshgrid(glat, glon, indexing="ij"))
    gf = synthesis(GLA, GLO, SHCoeffs(
        np.random.default_rng(5).normal(size=(9, 9)) / 4.0,
        np.random.default_rng(6).normal(size=(9, 9)) / 4.0), nmax=8)
    gpts = os.path.join(TMP, "grid.csv")
    SIO.write_points(gpts, GLA, GLO, gf, fmt="%.17g")

    outs = {}
    for mode in ("auto", "direct"):
        pfx = os.path.join(TMP, f"fft_{mode}")
        rc, exc = run(["analyze", "--points", gpts, "--nmax", "8",
                       "--rule", "grid", "--longitude-fft", mode,
                       "--out-prefix", pfx])
        outs[mode] = pfx + ".sh"
    ok = all(os.path.exists(outs[m]) for m in outs)
    check("'analyze --longitude-fft auto|direct' 都能跑", ok,
          f"auto={os.path.exists(outs['auto'])}, "
          f"direct={os.path.exists(outs['direct'])}")
    if ok:
        ca, cd = SIO.read_coeffs(outs["auto"]), SIO.read_coeffs(outs["direct"])
        d = max(float(np.abs(ca.C - cd.C).max()), float(np.abs(ca.S - cd.S).max()))
        check("'--longitude-fft auto' ≡ 'direct'（≤1e-12）", d <= 1e-12,
              f"max abs diff = {d:.2e}")

    # nlon <= 2*nmax 的网格上强制 'fft' 必须失败
    pfx = os.path.join(TMP, "fft_bad")
    rc, exc = run(["analyze", "--points", gpts, "--nmax", "18",
                   "--rule", "grid", "--longitude-fft", "fft",
                   "--out-prefix", pfx])
    check("'--longitude-fft fft' 在 nlon ≤ 2·nmax 时报错（不静默回退）",
          rc != 0, f"rc={rc}")

    # 散点上同理
    rc, exc = run(["analyze", "--points", pts, "--nmax", str(L),
                   "--longitude-fft", "fft", "--out-prefix",
                   os.path.join(TMP, "fft_pts")])
    check("'--longitude-fft fft' 在散点上报错（不静默回退）", rc != 0,
          f"rc={rc}")

    # ---- 'synth' 也要有这个开关（A4：综合方向的经度 FFT）------------------
    outs = {}
    for mode in ("auto", "direct"):
        p = os.path.join(TMP, f"synth_{mode}.nc")
        rc, exc = run(["synth", "--coeffs", truth_path, "--out-grid", p,
                       "--lat-step", "6", "--lon-step", "6",
                       "--longitude-fft", mode])
        outs[mode] = p
    ok = all(os.path.exists(outs[m]) for m in outs)
    check("'synth --longitude-fft auto|direct' 都能跑", ok,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    # -------------------------------------------- 重建场 vs 输出场（两条公式）
    # 输入声明 EWH、输出声明 geoid：重建场必须仍然与输入同量（EWH），
    # 输出场才是 geoid。两者逐阶比值 = f_t/f_u。
    r_recon = os.path.join(TMP, "f_recon.csv")
    r_out = os.path.join(TMP, "f_out.csv")
    rc, exc = run(["analyze", "--points", pts, "--nmax", str(L),
                   "--rule", "voronoi", "--field-unit", "ewh",
                   "--target-unit", "geoid",
                   "--out-grid", r_recon, "--out-field", r_out])
    ok_files = (rc == 0 and os.path.exists(r_recon) and os.path.exists(r_out)
                and exc is None)
    check("'analyze --out-grid/--out-field' 同时写出重建场与输出场",
          ok_files, f"rc={rc}" + (f", {exc!r}" if exc else ""))

    if ok_files:
        def _rd(p):
            rows = []
            with open(p, encoding="utf-8-sig") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln and not ln.startswith("#") and not ln.startswith("lon"):
                        rows.append([float(x) for x in ln.split(",")])
            return np.asarray(rows)

        a, b = _rd(r_recon), _rd(r_out)
        rms_r = float(np.sqrt((a[:, 2] ** 2).mean()))
        rms_o = float(np.sqrt((b[:, 2] ** 2).mean()))
        # 重建场应与输入同量：直接和输入数据比相对 RMS
        inp = _rd(pts)
        rel = float(np.sqrt(np.mean((a[:, 2] - inp[:, 2]) ** 2)) /
                    np.sqrt(np.mean(inp[:, 2] ** 2)))
        check("重建场与输入同物理量（可与原数据直接比对）",
              rel < 0.05 and rms_r > 0,
              f"重建 vs 原数据 相对 RMS = {rel:.3e}；重建场 RMS = {rms_r:.4e}")
        check("输出场是与重建场不同的物理量（geoid，逐阶乘 f_t/f_u）",
              abs(rms_o - rms_r) > 1e-6 * rms_r and rms_o > 0,
              f"重建场 RMS = {rms_r:.4e}，输出场 RMS = {rms_o:.4e}，"
              f"比值 = {rms_o / rms_r:.6f}")

    # ------------------------------------------------------- gaussian filter
    # --gaussian-km 必须作用于**存下来的系数**，而且只施加一次。
    gp = os.path.join(TMP, "g0.sh")
    rc0, _ = run(["analyze", "--points", pts, "--nmax", str(L),
                  "--rule", "voronoi", "--out-prefix", os.path.join(TMP, "g0")])
    rc1, exc1 = run(["analyze", "--points", pts, "--nmax", str(L),
                     "--rule", "voronoi", "--gaussian-km", "3000",
                     "--out-prefix", os.path.join(TMP, "g1"),
                     "--out-grid", os.path.join(TMP, "g1_recon.csv")])
    c0 = SIO.read_coeffs(os.path.join(TMP, "g0.sh"))
    c1 = SIO.read_coeffs(os.path.join(TMP, "g1.sh"))
    from shkit.filters import apply_gaussian, gaussian_coefficients
    W = gaussian_coefficients(3000.0, L)
    want = apply_gaussian(c0, 3000.0)
    check("'--gaussian-km' 改变**存下来的系数**（不再只是重建时生效）",
          rc1 == 0 and exc1 is None
          and np.abs(c1.C - want.C).max() < 1e-10 * np.abs(want.C).max()
          and np.abs(c1.C - c0.C).max() > 1e-6,
          f"半径 3000 km（W_nmax={W[-1]:.6g}）→ max|C − C₀| = "
          f"{np.abs(c1.C - c0.C).max():.4e}")
    check("平滑半径随系数文件保存（可追溯）",
          abs(float(c1.meta.get("gaussian_radius_km", 0)) - 3000.0) < 1e-9,
          f"meta gaussian_radius_km = {c1.meta.get('gaussian_radius_km')}")

    # --------------------------------------------------------------- synth
    gpath = os.path.join(TMP, "recon.nc")
    rc, exc = run(["synth", "--coeffs", truth_path, "--out-grid", gpath,
                   "--lat-step", "5", "--lon-step", "10"])
    check("'synth' runs and writes a grid", rc == 0 and os.path.exists(gpath)
          and exc is None,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    spath = os.path.join(TMP, "synth.csv")
    rc, exc = run(["synth", "--coeffs", truth_path, "--points", pts,
                   "--out", spath])
    check("'synth' at the sample points runs", rc == 0 and os.path.exists(spath)
          and exc is None,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    # ----------------------------------------------------------- roundtrip
    rc, exc = run(["roundtrip", "--points", pts, "--nmax", str(L),
                   "--rule", "voronoi"])
    check("'roundtrip' runs", rc == 0 and exc is None,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    # ------------------------------------------------------------ glq-grid
    glq = os.path.join(TMP, "glq.nc")
    rc, exc = run(["glq-grid", "--lmax", "8", "--out", glq])
    check("'glq-grid' runs and writes an exact quadrature grid",
          rc == 0 and os.path.exists(glq) and exc is None,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    # ---------------------------------------------------------------- info
    rc, exc = run(["info", "--coeffs", truth_path])
    check("'info' runs", rc == 0 and exc is None,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    # ---------------------------------------------------------------- nmax
    rc, exc = run(["nmax", "--coeffs", truth_path])
    check("'nmax' runs on a coefficient file",
          rc == 0 and exc is None,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    rc, exc = run(["nmax", "--points", pts, "--nmax", "12", "--rule", "voronoi"])
    check("'nmax --points' 先跑分析再诊断", rc == 0 and exc is None,
          f"rc={rc}" + (f", {exc!r}" if exc else ""))

    # 两条路径互斥
    rc, exc = run(["nmax", "--coeffs", truth_path, "--points", pts])
    check("'nmax' 同时给 --coeffs 与 --points 应报错",
          rc != 0 or exc is not None,
          f"rc={rc}, exc={type(exc).__name__ if exc else None}")

    # 诊断本身的数学正确性（不走 CLI 子进程）
    # NB: SHCoeffs 已经在模块顶部导入；这里若再局部 import 一次，Python 会把
    # 整个 main() 里的 SHCoeffs 当成局部名，前面的使用就会 UnboundLocalError。
    from shkit.diagnostics import diagnose_nmax, truncation_table

    # 造一套功率已知的系数：只有 n=0..4 有能量
    C = np.zeros((10, 10))
    S = np.zeros((10, 10))
    for n in range(5):
        C[n, 0] = 1.0 / (n + 1)
    known = SHCoeffs(C, S)
    tab = truncation_table(known, [4, 5, 10])
    check("truncation_table：保留功率比与 sqrt(1-保留比) 精确一致",
          abs(tab[0]["retained"] - 1.0) < 1e-15
          and abs(tab[1]["retained"] - 1.0) < 1e-15
          and all(abs(r["error"] - np.sqrt(max(1 - r["retained"], 0))) < 1e-15
                  for r in tab),
          f"n=4 保留 {tab[0]['retained']:.17f}，误差 {tab[0]['error']:.2e}")

    d = diagnose_nmax(known, coverage=1.0, n_points=1000, tol=0.01)
    check("diagnose_nmax：带限数据的 nmax 就是它真实的带宽（不触顶）",
          d["nmax_for_tolerance"] == 4 and not d["tolerance_hits_ceiling"],
          f"真实带宽 4 → 诊断给出 {d['nmax_for_tolerance']}，"
          f"ceiling={d['tolerance_hits_ceiling']}")

    # 平谱（掩膜型场）：任何小于 L 的截断都丢很多功率 → 必然触顶
    Cf = np.zeros((10, 10))
    Sf = np.zeros((10, 10))
    for n in range(10):
        Cf[n, 0] = 1.0
    flat = SHCoeffs(Cf, Sf)
    d_flat = diagnose_nmax(flat, coverage=1.0, n_points=100000, tol=0.01)
    check("diagnose_nmax：平谱数据触顶并给出「真实带宽 ≥ 当前阶数」的提示",
          d_flat["tolerance_hits_ceiling"]
          and d_flat["nmax_for_tolerance"] == flat.nmax,
          f"平谱（nmax={flat.nmax}）→ nmax={d_flat['nmax_for_tolerance']}，"
          f"ceiling={d_flat['tolerance_hits_ceiling']}")

    d2 = diagnose_nmax(known, coverage=2.0e-6, n_points=1000)
    check("diagnose_nmax：区域数据给出 Shannon 约束并标记计数上限不具约束力",
          d2["nmax_region_shannon"] == int(np.ceil(np.sqrt(1 / 2.0e-6) - 1))
          and d2["sampling_is_binding"] is False,
          f"Shannon≈1 需 nmax≈{d2['nmax_region_shannon']}，"
          f"sampling_is_binding={d2['sampling_is_binding']}")

    # 覆盖比 / 采样点数要能随文件读回（nmax 命令依赖它）
    from shkit import io as shio2
    p2 = os.path.join(TMP, "cov.sh")
    tagged = SHCoeffs(C, S, {"coverage": 2.068019e-06, "n_points": 142846})
    shio2.write_coeffs(tagged, p2, layout="triangle")
    back2 = shio2.read_coeffs(p2)
    check("coverage / n_points 随系数文件保存并读回",
          abs(back2.meta.get("coverage", 0) - 2.068019e-06) < 1e-18
          and back2.meta.get("n_points") == 142846,
          f"coverage={back2.meta.get('coverage')}, "
          f"n_points={back2.meta.get('n_points')}")

    # ------------------------------------------------------- bad input path
    rc, exc = run(["info", "--coeffs", os.path.join(TMP, "does_not_exist.sh")])
    check("a missing file exits non-zero instead of crashing silently",
          rc != 0 or exc is not None, f"rc={rc}, exc={type(exc).__name__ if exc else None}")

    npass = sum(RESULTS)
    print(f"\n{'='*70}\n{npass}/{len(RESULTS)} checks passed")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
