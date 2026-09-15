# -*- coding: utf-8 -*-
"""
Yangtze_River 散点 → 球谐系数，并**算出**"对应空间分辨率的阶数"。

数据：``Yangtze_River.mat`` 只有 ``L``（2 × 142846，行 0 经度、行 1 纬度），
没有数值场，所以这里转的是**区域指示函数**（区域内 = 1，区域外 = 0）——
即作者 ``cal_kernalSH.m`` / ``scatter2SHs.m`` 里那个 ``input_mask`` 的球谐系数。

积分元：每个点代表一个 ``Δlon × Δlat`` 球面格，
``w_i = Δlon_rad · (sin φ_hi − sin φ_lo)``；Δ 由数据本身稳健估计，
不硬编码作者脚本里的 0.00083。

用法：
    python tools/yangtze_points_to_sh.py --lmax 60
    python tools/yangtze_points_to_sh.py --lmax 60 --sweep 30,60,120
    python tools/yangtze_points_to_sh.py --mat 水位变化new.mat --values CH --epoch 0
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DEFAULT_MAT = r"D:\myds\3_research\他人工作\zhengyuhao\Yangtze_River.mat"


def load_points(path):
    """读 .mat（v7.3/HDF5 或 v7），返回 (变量名, lon, lat, 其它变量 dict)。"""
    extra = {}
    try:
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            key = "L" if "L" in keys else keys[0]
            arr = np.array(f[key], dtype=float)
            for k in keys:
                if k != key:
                    extra[k] = np.array(f[k])
    except (OSError, ImportError):
        from scipy.io import loadmat
        m = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
        key = "L" if "L" in m else sorted(m)[0]
        arr = np.asarray(m[key], dtype=float)
        extra = {k: v for k, v in m.items() if k != key}
    if arr.ndim != 2:
        raise ValueError(f"{key} 形状 {arr.shape} 不是二维")
    if arr.shape[0] == 2:
        lon, lat = arr[0], arr[1]
    elif arr.shape[1] == 2:
        lon, lat = arr[:, 0], arr[:, 1]
    else:
        raise ValueError(f"{key} 形状 {arr.shape} 无法判断经纬度方向")
    if lat.max() > 90.0 and lon.max() <= 90.0:      # 防呆：装反了
        lon, lat = lat, lon
    return key, lon, lat, extra


def robust_cell_size(v, name=""):
    """相邻唯一坐标间隔的稳健格距（度）。"""
    u = np.unique(v)
    gaps = np.diff(u)
    d0 = float(gaps.min())
    core = gaps[gaps < 1.5 * d0]
    d = float(np.median(core)) if core.size else d0
    if name:
        kinds = np.unique(np.round(core, 8))
        print(f"    {name}: 唯一值 {u.size}，格距 {d:.8f}°"
              f"（{d * 111.19:.4f} km），邻近间隔只有 {kinds.size} 种取值"
              " → 规则格网")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default=DEFAULT_MAT)
    ap.add_argument("--lmax", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(ROOT, "examples"))
    ap.add_argument("--sweep", default="30,60,120",
                    help="额外在这些阶数上做一遍，量重建出来的掩膜峰值")
    ap.add_argument("--values", default=None,
                    help="数值场所在变量名（默认用掩膜 = 1）")
    ap.add_argument("--epoch", type=int, default=0,
                    help="--values 给的是 (N, ntime) 时的时次下标")
    ap.add_argument("--cell", type=float, default=None, metavar="DEG",
                    help="强制积分元的格距（度），例如作者脚本里的 0.00083；"
                         "默认由数据稳健估计")
    args = ap.parse_args()

    from shkit import io as shio
    from shkit.analysis import analysis
    from shkit.diagnostics import harmonic_resolution_km
    from shkit.filters import EARTH_RADIUS_M as R
    from shkit.synthesis import synthesis

    t0 = time.time()
    print("=" * 78)
    print(f"  读入 {args.mat}")
    print("=" * 78)
    key, lon, lat, extra = load_points(args.mat)
    n = lon.size
    print(f"  变量 {key}: {n} 个散点；同文件其它变量: {list(extra)}")
    print(f"  经度 [{lon.min():.6f}, {lon.max():.6f}]  "
          f"纬度 [{lat.min():.6f}, {lat.max():.6f}]")

    print("\n  估计格距（积分元）：")
    dlon = robust_cell_size(lon, "经度")
    dlat = robust_cell_size(lat, "纬度")
    if args.cell:
        print(f"    （--cell 强制为 {args.cell:g}°；实测估计值 "
              f"{dlon:.8f} / {dlat:.8f}°，相对差 "
              f"{abs(args.cell / dlon - 1):.2%} / {abs(args.cell / dlat - 1):.2%}）")
        dlon = dlat = float(args.cell)

    lo_rad = np.deg2rad(dlon)
    phi = np.deg2rad(lat)
    half = np.deg2rad(dlat) / 2.0
    w = lo_rad * (np.sin(phi + half) - np.sin(phi - half))      # 立体角 sr
    omega = float(w.sum())
    frac = omega / (4 * np.pi)
    area_km2 = omega * R ** 2 / 1e6
    print(f"\n  单元面积中位数 = {np.median(w) * R ** 2 / 1e6:.6f} km²")
    print(f"  总覆盖面积     = {area_km2:.2f} km²")
    print(f"  占全球立体角   = {frac:.6e}（{frac * 100:.6f}%）")

    flat = (R ** 2 * lo_rad * np.deg2rad(dlat) * np.cos(phi)).sum()
    print(f"  平面近似（同作者 calParea 的量级）= {flat / 1e6:.2f} km²，"
          f"与球面精确值相对差 {abs(flat / (omega * R ** 2) - 1):.2e}")

    # ---------------- 区域尺度 → 阶数 -------------------------------------
    ext_lon = (lon.max() - lon.min()) * 111.19 * np.cos(np.deg2rad(lat.mean()))
    ext_lat = (lat.max() - lat.min()) * 111.19
    diag = float(np.hypot(ext_lon, ext_lat))
    box_area = ext_lon * ext_lat
    eq_box = 2 * float(np.sqrt(box_area / np.pi))
    geo_mean = float(np.sqrt(ext_lon * ext_lat))
    eq_water = 2 * float(np.sqrt(area_km2 / np.pi))

    print("\n" + "=" * 78)
    print("  区域尺度 → 球谐阶数（半波长 = πR/n，R = 6378.136 km）")
    print("=" * 78)
    print(f"  经向 {ext_lon:.1f} km，纬向 {ext_lat:.1f} km，"
          f"外接矩形 {box_area:.0f} km²")
    print(f"  {'特征尺度':22s}{'长度 (km)':>12s}{'对应阶数 n':>14s}")
    for name, Lc in (("采样点间距", dlat * 111.19),
                     ("长江主槽宽度（约）", 1.0),
                     ("水体等面积圆直径", eq_water),
                     ("区域短边", min(ext_lon, ext_lat)),
                     ("矩形等面积圆直径", eq_box),
                     ("边长几何平均", geo_mean),
                     ("外接矩形对角线", diag)):
        nn = np.pi * R / 1000.0 / Lc
        note = "  ← 全球展开不可行" if nn > 2000 else ""
        print(f"  {name:20s}{Lc:>14.2f}{nn:>14.1f}{note}")

    # ---------------- Shannon 数（解析，精确） ----------------------------
    print("\n" + "=" * 78)
    print("  这个区域能承载多少个独立自由度？（Shannon 数 N ≈ (L+1)²·Ω/4π）")
    print("=" * 78)
    print(f"  {'L':>6}{'自由度 (L+1)²':>16}{'Shannon 数':>14}"
          f"{'半波长 (km)':>14}")
    for L in (10, 30, 60, 120, 240, 480, 700, 1000):
        print(f"  {L:>6}{(L + 1) ** 2:>16}{(L + 1) ** 2 * frac:>14.4e}"
              f"{harmonic_resolution_km(L):>14.1f}")
    L1 = np.sqrt(1.0 / frac) - 1.0
    print(f"\n  N = 1（掩膜能「立起来」的最起码阶数）→ L ≈ {L1:.0f}")
    print("  含义：L 远小于它时，区域掩膜截断后的峰值只有 N 量级，"
          "不可能接近 1。")

    # ---------------- 正变换 -------------------------------------------------
    field = np.ones(n)
    if args.values:
        if args.values not in extra:
            raise SystemExit(f"{args.mat} 里没有变量 {args.values}；"
                             f"有 {list(extra)}")
        a = np.asarray(extra[args.values], dtype=float)
        if a.shape[0] != n and a.shape[1] == n:
            a = a.T
        if a.ndim == 2:
            a = a[:, args.epoch]
        field = a.ravel()
        print(f"\n  数值场 = {args.values}（时次 {args.epoch}），"
              f"范围 [{field.min():.4g}, {field.max():.4g}]")

    print("\n" + "=" * 78)
    print(f"  正变换 → C_nm（nmax={args.lmax}，rule=user，normalise=none）")
    print("=" * 78)
    c, rep = analysis(lat, lon, field, args.lmax, method="quadrature",
                      rule="user", user_w=w, normalise="none",
                      field_unit="scalar")
    print(f"  用时 {time.time() - t0:.1f} s；Σw = {omega:.6e} sr "
          f"（= 4π × {frac:.6e}）")
    print(f"  半波长分辨率 = {harmonic_resolution_km(args.lmax):.1f} km")
    if args.values is None:
        print(f"  C[0,0] = {c.C[0, 0]:.10e}  应为 Ω/4π = {frac:.10e}"
              f"  （相对差 {abs(c.C[0, 0] / frac - 1):.2e}）")

    back = np.asarray(synthesis(lat, lon, c)).ravel()
    print(f"\n  区域内重建（掩膜应≈1）：{back.min():.4f} / "
          f"{back.mean():.4f} / {back.max():.4f}  (min/mean/max)")
    rng = np.random.default_rng(0)
    n_out = 30000
    o_lon = rng.uniform(lon.min() - 2, lon.max() + 2, n_out)
    o_lat = rng.uniform(lat.min() - 1.5, lat.max() + 1.5, n_out)
    keep = ~((o_lat >= lat.min()) & (o_lat <= lat.max())
             & (o_lon >= lon.min()) & (o_lon <= lon.max()))
    oo = np.asarray(synthesis(o_lon[keep], o_lat[keep], c)).ravel()
    print(f"  区域外重建：mean|.| = {np.abs(oo).mean():.3e}，"
          f"P95 = {np.percentile(np.abs(oo), 95):.3e}")

    # ---------------- 阶数扫描：掩膜峰值随 L 的变化 ------------------------
    if args.sweep and args.values is None:
        print("\n" + "=" * 78)
        print("  阶数扫描：截断到 L 时，掩膜重建的峰值能到多少")
        print("=" * 78)
        print(f"  {'L':>6}{'半波长 (km)':>14}{'Shannon 数':>13}"
              f"{'实测峰值':>12}{'峰值/Shannon':>15}")
        for L in [int(x) for x in args.sweep.split(",") if x.strip()]:
            t1 = time.time()
            cc, _ = analysis(lat, lon, field, L, method="quadrature",
                             rule="user", user_w=w, normalise="none",
                             field_unit="scalar")
            bb = np.asarray(synthesis(lat, lon, cc)).ravel()
            sh = (L + 1) ** 2 * frac
            print(f"  {L:>6}{harmonic_resolution_km(L):>14.1f}"
                  f"{sh:>13.4e}{bb.max():>12.4f}{bb.max() / sh:>15.3f}"
                  f"   ({time.time() - t1:.0f}s)")

    # ---------------- 保存 --------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    tag = f"yangtze_mask_n{args.lmax}"
    if args.values:
        tag = f"yangtze_{args.values}_e{args.epoch}_n{args.lmax}"
    p_sh = os.path.join(args.out, tag + ".sh")
    shio.write_coeffs(c, p_sh, layout="triangle")
    print(f"\n  已写出：{p_sh}")

    p_w = os.path.join(args.out, "yangtze_points_weights.csv")
    if not os.path.exists(p_w):
        with open(p_w, "w", encoding="utf-8") as fh:
            fh.write("lon,lat,weight_sr,area_m2\n")
            for i in range(n):
                fh.write(f"{lon[i]:.8f},{lat[i]:.8f},{w[i]:.12e},"
                         f"{w[i] * R ** 2:.6f}\n")
        print(f"  已写出：{p_w}  （{n} 行，w 为立体角 sr）")

    p_npz = os.path.join(args.out, tag + ".npz")
    np.savez_compressed(p_npz, C=c.C, S=c.S, lon=lon, lat=lat, w=w)
    print(f"  已写出：{p_npz}")
    print(f"\n  总用时 {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
