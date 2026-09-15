# -*- coding: utf-8 -*-
"""
make_horizontal_golden.py — 生成「水平形变黄金样本」(B0)
========================================================

SHKit 与 SHSynth **各自实现**水平形变（不共享代码、不互相 import），
因此一致性只能靠**冻结约定 + 黄金样本对表**。本脚本是那份"契约的可执行版本"：
它用**最直白的写法**算出参考值并冻结成一个 ``.npz``，两个包各存一份**逐位相同**的副本，
各自断言自己的实现与冻结值一致（``≤1e-12``，目标 ``0.0``）。

设计约束（都是为了"能被另一个包逐行移植"）：

* **只依赖 numpy**：勒让德递推、``dP̄/dθ``、``Q=P̄/sinθ``、gfc 解析全部写在本文件里，
  不 import ``shkit``。勒夫数表用 ``np.load`` 直接读 ``.npz``。
  （唯一与 SHKit 的耦合是"勒夫数表文件"这个**数据**——而它的哈希会写进 provenance。）
* **可读性优先，不是性能**：这里用全矩阵 ``P[n, m, i]`` 写法。
  **包内实现必须流式**（见规划 §3.11，全矩阵会多吃 1 GB、代价 4.81×），
  但**数值结果必须与本文件的冻结值逐位对得上**。
* **极点用精确值，不用 ε 极限**：``P̄/sinθ`` 在极点用 ``Q_n1(t=±1)`` 递推给出解析值
  （``Q_11=√3``），``m≥2`` 恰为 0。``dP̄/dθ`` 由**递推两边对 θ 求导**得到，全程无 ``1/sinθ``。

用法::

    python tools/make_horizontal_golden.py                 # 写 tests/fixtures/horizontal_golden.npz
    python tools/make_horizontal_golden.py --out /tmp/g.npz
    python tools/make_horizontal_golden.py --grace-gfc path/to/GSM-....gfc
    python tools/make_horizontal_golden.py --check         # 只校验已有 fixture，不覆盖

冻结约定见 ``docs/水平形变契约.md``（F1–F11）。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(ROOT, "tests", "fixtures", "horizontal_golden.npz")
LOVE_NPZ = os.path.join(ROOT, "data", "load_love_numbers.npz")

# ---------------------------------------------------------------- F2: 冻结常数
R_EARTH_M = 6378136.46          # EARTH_RADIUS_M，与 shkit.filters 同一个值
NMAX = 20                       # 四组系数统一到 nmax=20
SEED = 20240501
GRACE_SCALE = 1e-10             # GRACE 位系数量级

CONVENTIONS = {
    "F1_norm": "4pi",
    "F1_csphase": 1,
    "F2_degree_factor": "R*l'_n/(1+k'_n)",
    "F2_R_m": R_EARTH_M,
    "F2_love_model": "PREM-LLN / Wang et al. 2012 (load l')",
    "F3_sign": "u_N = -dS/dtheta ; u_E = +(1/sin th) dS/dlambda "
               "(theta-hat points SOUTH, lambda-hat points EAST)",
    "F4_pole": "exact: Q_n1(t=+-1) from the m=1 recurrence seeded Q_11=sqrt(3); "
               "Q_nm=0 exactly for m>=2",
    "F5_m0": "m=0 contributes to u_N only (dPbar_n0/dtheta != 0); u_E has no m=0 term",
    "F6_derived": "magnitude=hypot(u_N,u_E); azimuth=atan2(u_E,u_N) in [0,360) clockwise from north",
    "F7_fields": ["north_displacement", "east_displacement", "magnitude", "azimuth"],
    "F8_units": "m",
    "F9_field_unit": "horizontal_displacement",
    "F10_smoothing": "single gaussian_km shared with radial, applied to the COEFFICIENTS "
                     "once.  NOTE: grad_H does NOT commute with per-degree smoothing on "
                     "the sphere (measured ratio 1.24 / 0.91 for an 800 km Gaussian); only "
                     "'smooth the coefficients, then differentiate' is defined, and "
                     "synthesis_horizontal takes no gaussian_km parameter at all.",
    "F11_irreversible": "horizontal -> geopotential only up to an additive C00 constant",
}


# ===========================================================================
# 1. 勒让德递推 + 对 theta 求导（自足、可移植）
# ===========================================================================
def legendre_and_dtheta(lat_deg, nmax):
    """返回 ``(P, D, t, u)``。

    ``P[n, m, i] = Pbar_nm(sin lat_i)``，``D[n, m, i] = dPbar_nm/dtheta``。
    ``t = sin(lat) = cos(theta)``，``u = cos(lat) = sin(theta)``。

    递推取自 SHKit/m2py 的跨阶形式（``method 3``）；``D`` 是把**每条递推两边对
    theta 求导**得到的（``t`` 与 ``u`` 的导数分别是 ``-u`` 与 ``t``），
    因此**全程不出现 ``1/sin(theta)``，极点天然安全**。
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    npts = lat.size
    t = np.sin(np.deg2rad(lat))
    u = np.cos(np.deg2rad(lat))
    dt = -u                      # d(cos theta)/dtheta
    du = t                       # d(sin theta)/dtheta

    P = np.zeros((nmax + 1, nmax + 1, npts))
    D = np.zeros_like(P)
    P[0, 0] = 1.0
    D[0, 0] = 0.0

    # ---- m = 0 -----------------------------------------------------------
    if nmax >= 1:
        P[1, 0] = np.sqrt(3.0) * t
        D[1, 0] = np.sqrt(3.0) * dt
        for n in range(2, nmax + 1):
            c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / (n * n))
            num = (2 * n + 1) * (n - 1) * (n - 1)
            c2 = np.sqrt(num / ((2 * n - 3) * n * n)) if num > 0 else 0.0
            P[n, 0] = c1 * t * P[n - 1, 0] - c2 * P[n - 2, 0]
            D[n, 0] = c1 * (dt * P[n - 1, 0] + t * D[n - 1, 0]) - c2 * D[n - 2, 0]

    # ---- m = 1 -----------------------------------------------------------
    if nmax >= 1:
        P[1, 1] = np.sqrt(3.0) * u
        D[1, 1] = np.sqrt(3.0) * du
        for n in range(2, nmax + 1):
            c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - 1) * (n + 1)))
            num = (2 * n + 1) * n * (n - 2)
            c2 = np.sqrt(num / ((2 * n - 3) * (n + 1) * (n - 1))) if num > 0 else 0.0
            P[n, 1] = c1 * t * P[n - 1, 1] - c2 * P[n - 2, 1]
            D[n, 1] = c1 * (dt * P[n - 1, 1] + t * D[n - 1, 1]) - c2 * D[n - 2, 1]

    # ---- m >= 2 ----------------------------------------------------------
    for m in range(2, nmax + 1):
        for n in range(m, nmax + 1):
            a1 = np.sqrt((2 * n + 1) * (n - m) * (n - m - 1) /
                         ((2 * n - 3) * (n + m) * (n + m - 1)))
            if m == 2:
                g1 = np.sqrt(2.0) * np.sqrt((n - m + 1) * (n - m + 2) /
                                            ((n + m) * (n + m - 1)))
                b1 = np.sqrt(2.0) * np.sqrt((2 * n + 1) * (n + m - 2) * (n + m - 3) /
                                            ((2 * n - 3) * (n + m) * (n + m - 1)))
            else:
                g1 = np.sqrt((n - m + 1) * (n - m + 2) / ((n + m) * (n + m - 1)))
                b1 = np.sqrt((2 * n + 1) * (n + m - 2) * (n + m - 3) /
                             ((2 * n - 3) * (n + m) * (n + m - 1)))
            P[n, m] = a1 * P[n - 2, m] + b1 * P[n - 2, m - 2] - g1 * P[n, m - 2]
            D[n, m] = a1 * D[n - 2, m] + b1 * D[n - 2, m - 2] - g1 * D[n, m - 2]
    return P, D, t, u


def q_at_pole_m1(nmax, t_pole):
    """``Q_n1 = Pbar_n1/sin(theta)`` 在极点（``t = +-1``）的**精确**值。

    ``Pbar_n1 = u * Q_n1(t)``（``u`` 与 n 无关），所以把 m=1 的递推两边除以 ``u``
    就得到 ``Q`` 的同一条递推，种子 ``Q_11 = sqrt(3)``。``Q_n1`` 是 ``t`` 的 ``n-1`` 次
    多项式，在极点直接求值即解析极限 —— 不需要任何 ``eps`` 偏移。
    """
    Q = np.zeros(nmax + 1)
    if nmax >= 1:
        Q[1] = np.sqrt(3.0)
    for n in range(2, nmax + 1):
        c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - 1) * (n + 1)))
        num = (2 * n + 1) * n * (n - 2)
        c2 = np.sqrt(num / ((2 * n - 3) * (n + 1) * (n - 1))) if num > 0 else 0.0
        Q[n] = c1 * t_pole * Q[n - 1] - c2 * Q[n - 2]
    return Q


# ===========================================================================
# 2. 水平形变算子（参考实现）
# ===========================================================================
def degree_factor_h(love_l, love_k, nmax, radius_m=R_EARTH_M):
    """``F^h_n = R * l'_n / (1 + k'_n)``，``n = 0..nmax``。

    ``l'_0 = 0`` ⇒ ``F^h_0 = 0``：**0 阶被自动抹掉**（与水平梯度抹掉常数的性质一致）。
    """
    n = np.arange(nmax + 1)
    ln = np.array([love_l[i] if i < love_l.size else 0.0 for i in n])
    kn = np.array([love_k[i] if i < love_k.size else 0.0 for i in n])
    return radius_m * ln / (1.0 + kn)


def horizontal_field(lat_deg, lon_deg, C, S, Fh, pole_tol=1e-12):
    """系数 → ``(u_N, u_E, magnitude, azimuth)``，单位米。

    ``S(theta, lambda) = sum_n Fh[n] * sum_m Pbar_nm [C cos m lam + S sin m lam]``；
    ``u_N = -dS/dtheta``，``u_E = +(1/sin th) dS/dlambda``（F3）。
    """
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    nmax = C.shape[0] - 1
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))

    P, D, t, u = legendre_and_dtheta(lat, nmax)
    lam = np.deg2rad(lon)
    pole = np.abs(u) < pole_tol                     # 恰好 +-90°
    qpole = np.array([q_at_pole_m1(nmax, tt) for tt in t[pole]])   # (npole, nmax+1)

    dth = np.zeros(lat.size)                        # dS/dtheta
    dlm = np.zeros(lat.size)                        # (1/sin th) dS/dlambda
    for n in range(0, nmax + 1):
        fn = float(Fh[n])
        if fn == 0.0:
            continue
        for m in range(0, n + 1):
            yc = fn * C[n, m]
            ys = fn * S[n, m]
            if yc == 0.0 and ys == 0.0:
                continue
            cm = np.cos(m * lam)
            sm = np.sin(m * lam)
            dth += D[n, m] * (yc * cm + ys * sm)     # F5: m=0 也进 u_N
            if m >= 1:
                q = np.divide(P[n, m], u, out=np.zeros_like(u), where=~pole)
                if pole.any():
                    if m == 1:
                        q[pole] = qpole[:, n]
                    else:
                        q[pole] = 0.0                # Pbar_nm ∝ sin^m th ⇒ m>=2 恰为 0
                dlm += q * m * (ys * cm - yc * sm)
    u_n = -dth
    u_e = dlm
    mag = np.hypot(u_n, u_e)
    azi = np.degrees(np.arctan2(u_e, u_n)) % 360.0   # F6
    # IEEE: (-1e-16) % 360.0 == 360.0 exactly，会把 [0,360) 变成 [0,360]。
    # 契约是**半开区间**，所以把恰好 360.0 折回 0.0。
    azi = np.where(azi >= 360.0, 0.0, azi)
    return u_n, u_e, mag, azi


# ===========================================================================
# 3. 采样点、系数样本、勒夫数表
# ===========================================================================
def golden_points():
    """固定采样点：``lat x lon`` 全交叉。

    **必须包含**：``lat = ±90°``（极点探针）、``lon = 90°``
    （纯 ``C_11`` 场在极点给出 ``u_E = -sqrt(3)`` 的那一点）、
    以及 ``±89.999°``（近极点，走除法路径）。
    """
    lats = np.array([-90.0, -89.999, -80.0, -60.0, -45.0, -30.0, -10.0, 0.0,
                     10.0, 30.0, 45.0, 60.0, 80.0, 89.999, 90.0])
    lons = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return LA.ravel(), LO.ravel()


def _parse_gfc(path, nmax):
    """极简 ICGEM/gfc 解析（``gfc n m C S``），只为让本脚本不依赖 shkit。"""
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    used = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tok = line.split()
            if len(tok) < 5 or tok[0].lower() not in ("gfc", "gfct"):
                continue
            try:
                n, m = int(tok[1]), int(tok[2])
                c, s = float(tok[3]), float(tok[4])
            except ValueError:
                continue
            if n > nmax or m > n:
                continue
            C[n, m], S[n, m] = c, s
            used += 1
    if not used:
        raise ValueError(f"{path}: 没有解析到 'gfc n m C S' 数据行")
    return C, S


def _auto_grace_gfc():
    """在 SHKit 上一级目录里找一个真实 GRACE GSM 文件（找不到就返回 None）。"""
    pats = [
        os.path.join(os.path.dirname(ROOT), "2_unzipped", "1_CSR", "01RL06", "01deg60",
                     "GSM-*.gfc"),
        os.path.join(os.path.dirname(ROOT), "2_unzipped", "**", "GSM-*.gfc"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def build_cases(grace_gfc=None):
    """四组系数（都与 ``NMAX`` 同尺寸，便于堆成一个数组）。"""
    C = np.zeros((4, NMAX + 1, NMAX + 1))
    S = np.zeros((4, NMAX + 1, NMAX + 1))
    names = ["pure_C10", "pure_C11", "random_bandlimited", "grace_gfc_trunc20"]
    notes = ["解析锚点：u_N=sqrt(3)cos(phi), u_E=0",
             "解析锚点：u_N=-sqrt(3)cos(lam)sin(phi), u_E=-sqrt(3)sin(lam)",
             f"随机带限, seed={SEED}, scale={GRACE_SCALE:g}",
             ""]

    C[0, 1, 0] = 1.0
    C[1, 1, 1] = 1.0

    rng = np.random.default_rng(SEED)
    for n in range(NMAX + 1):
        for m in range(n + 1):
            C[2, n, m] = rng.standard_normal() * GRACE_SCALE
            if m >= 1:
                S[2, n, m] = rng.standard_normal() * GRACE_SCALE
    # 真实 GRACE 信号的主体是低阶；用 1/(n+1) 衰减让样本更接近真实谱形
    for n in range(NMAX + 1):
        C[2, n, :n + 1] /= (n + 1)
        S[2, n, :n + 1] /= (n + 1)

    path = grace_gfc if grace_gfc is not None else _auto_grace_gfc()
    if path and os.path.exists(path):
        Cg, Sg = _parse_gfc(path, NMAX)
        C[3], S[3] = Cg, Sg
        notes[3] = f"真实 GRACE gfc（截断到 {NMAX} 阶）: {os.path.basename(path)}"
    else:
        notes[3] = "跳过：未找到真实 GRACE gfc（传 --grace-gfc 指定）"
        print(f"[warn] 未找到 GRACE gfc，第 4 组系数将为全 0"
              f"（用 --grace-gfc 指定；这会让 fixture 少一个真实样本）", file=sys.stderr)

    S[:, :, 0] = 0.0                                  # S_n0 ≡ 0
    return C, S, names, notes, (path if path and os.path.exists(path) else None)


def load_love():
    """直接读 ``data/load_love_numbers.npz``（不 import shkit）。"""
    if not os.path.exists(LOVE_NPZ):
        raise FileNotFoundError(f"找不到勒夫数表: {LOVE_NPZ}")
    with np.load(LOVE_NPZ, allow_pickle=False) as z:
        out = {k: np.asarray(z[k]) for k in z.files}
    with open(LOVE_NPZ, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return out, digest


# ===========================================================================
# 4. 生成 / 校验
# ===========================================================================
def build(nmax=NMAX, grace_gfc=None):
    love, lhash = load_love()
    l = np.asarray(love["l"], dtype=float).ravel()
    k = np.asarray(love["k"], dtype=float).ravel()
    Fh = degree_factor_h(l, k, 96)                     # 冻结到 n=0..96（SHSynth 侧也要对）
    Fh_case = Fh[:nmax + 1]

    lat, lon = golden_points()
    C, S, names, notes, used = build_cases(grace_gfc)

    u_n = np.zeros((len(names), lat.size))
    u_e = np.zeros_like(u_n)
    mag = np.zeros_like(u_n)
    azi = np.zeros_like(u_n)
    for i in range(len(names)):
        u_n[i], u_e[i], mag[i], azi[i] = horizontal_field(lat, lon, C[i], S[i], Fh_case)

    prov = {
        "generator": "tools/make_horizontal_golden.py",
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "nmax": int(nmax),
        "n_points": int(lat.size),
        "seed": SEED,
        "grace_gfc": (os.path.basename(used) if used else None),
        "love_table": os.path.relpath(LOVE_NPZ, ROOT).replace("\\", "/"),
        "love_table_sha256": lhash,
        "love_model": str(love.get("model", "unknown")),
        "love_source": str(love.get("source", "unknown")),
        "n_love_available": int(l.size),
        "conventions": CONVENTIONS,
        "case_notes": notes,
        "note": ("两边各自实现时以本文件的数组为准（不是文件字节）："
                 "跨包对表判据 <=1e-12，目标 0.0。"),
    }
    return {
        "lat": lat, "lon": lon,
        "nmax": np.int64(nmax),
        "case_names": np.array(names),
        "C": C, "S": S,
        "u_N": u_n, "u_E": u_e, "magnitude": mag, "azimuth": azi,
        "Fh": Fh, "love_l": l[:97], "love_k": k[:97],
        "provenance": np.array(json.dumps(prov, ensure_ascii=False, indent=1)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成 / 校验水平形变黄金样本")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--grace-gfc", default=None,
                    help="真实 GRACE gfc（截断到 nmax）；省略则自动在上级目录里找")
    ap.add_argument("--nmax", type=int, default=NMAX)
    ap.add_argument("--check", action="store_true",
                    help="只把新算结果与已有 fixture 比对，不覆盖")
    args = ap.parse_args(argv)

    data = build(nmax=args.nmax, grace_gfc=args.grace_gfc)

    if args.check:
        if not os.path.exists(args.out):
            print(f"[FAIL] 没有 fixture: {args.out}")
            return 1
        with np.load(args.out, allow_pickle=False) as z:
            bad = 0
            for key in ("C", "S", "u_N", "u_E", "magnitude", "azimuth", "Fh", "lat", "lon"):
                d = float(np.max(np.abs(np.asarray(z[key]) - data[key])))
                flag = "OK  " if d == 0.0 else "DIFF"
                if d != 0.0:
                    bad += 1
                print(f"  {flag} {key:12s} max|diff| = {d:.3e}")
        print("  逐位一致。" if not bad else f"  {bad} 个数组不一致。")
        return 0 if not bad else 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"写入 {args.out}")
    print(f"  系数样本 : {list(data['case_names'])}")
    print(f"  采样点   : {data['lat'].size}"
          f"  (含 lat=±90° 极点行 {(np.abs(np.abs(data['lat']) - 90) < 1e-12).sum()} 个)")
    print(f"  Fh       : n=0..96, Fh[1]={data['Fh'][1]:.6e}, Fh[0]={data['Fh'][0]:.1f}")
    print(f"  |u_N|max : {np.max(np.abs(data['u_N'])):.6e} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
