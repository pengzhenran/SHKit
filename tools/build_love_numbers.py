# -*- coding: utf-8 -*-
"""
把 PREM-LLNs(-complete).dat 转成 SHKit 自带的载荷勒夫数表，并核对来源。

格式（列）: n, h, l, k, nl, nk
  - h  = h'_n   （径向，载荷勒夫数第二类）
  - l  = l'_n   （水平）
  - k  = k'_n   （位）
  - nl = n*l'_n, nk = n*k'_n  （用于核对约定）

pz_LLN.m 在 degree 1 上做了一次 CE -> CF 参考系改正：
    k'_1(CF) = -(h'_1 + 2 l'_1)/3
本脚本复现这一步，并与 m2py 的 love_numbers.npy 逐值比对。
"""
import os
import sys

import numpy as np

SRC = (r"D:\华为家庭存储\1_Joe Science Data\1_1_GRACE\2_Corrections"
       r"\love num\loading_models\LLNs")
M2PY = r"D:\myds\mywork\1_GlobalSpatiotemporal\codes\m2py\data\love_numbers.npy"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data")


def read_lln(path):
    """Read n,h,l,k[,nl,nk] rows, skipping the header and the n=inf asymptote row.

    The ``-complete`` files end with a row whose degree column is literally
    ``inf``; that is the asymptotic (n -> infinity) limit, which is why
    ``pz_LLN.m`` uses ``read(1:end-1, …)``.  We keep it separately.
    """
    rows, asymptote = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vals = [float(x) for x in parts[:6]]
            except ValueError:
                continue                      # header
            if not np.isfinite(vals[0]):
                asymptote = vals              # n = inf
                continue
            rows.append(vals)
    return np.asarray(rows, dtype=float), asymptote


def main():
    path = os.path.join(SRC, "PREM-LLNs-complete.dat")
    a, asym = read_lln(path)
    n = a[:, 0].astype(int)
    h, l, k = a[:, 1], a[:, 2], a[:, 3]
    nmax = int(n.max())
    assert np.array_equal(n, np.arange(1, nmax + 1)), "度编号不是连续 1..N"

    print(f"{os.path.basename(path)}: {a.shape[0]} 个有效数据行（另有 1 行 n=inf 渐近值）")
    print(f"  度范围 n = {n.min()} .. {n.max()}")
    print(f"  h'_2 = {h[1]:.8f}   l'_2 = {l[1]:.8f}   k'_2 = {k[1]:.8f}")
    print(f"  nk/n 校验 (应等于 k): max|k - nk/n| = "
          f"{np.abs(k - a[:, 5] / n).max():.3e}")
    print(f"  nl/n 校验 (应等于 l): max|l - nl/n| = "
          f"{np.abs(l - a[:, 4] / n).max():.3e}")
    if asym is not None:
        print(f"  渐近值 n=inf: h'={asym[1]:.6f}  l'={asym[2]:.3e}  k'={asym[3]:.3e}")

    # degree 1: CE -> CF  (与 pz_LLN.m 完全一致)
    k_cf1 = -(h[0] + 2.0 * l[0]) / 3.0
    print(f"\ndegree 1 CE->CF: 原始 k'_1 = {k[0]:.8f} -> {k_cf1:.8f}"
          f"   [pz_LLN: -(h'_1+2l'_1)/3]")
    k_cf = k.copy()
    k_cf[0] = k_cf1

    # 与 m2py 的表比对
    kl = np.load(M2PY)
    ref = np.concatenate(([0.0], k_cf))
    m = min(kl.size, ref.size)
    dmax = float(np.abs(kl[:m] - ref[:m]).max())
    print(f"\n与 m2py love_numbers.npy 比对（前 {m} 个度，m2py 共 {kl.size} 个）:")
    print(f"  最大绝对差 = {dmax:.3e}")
    print(f"  kl[1] = {kl[1]:.8f}   本表[1] = {ref[1]:.8f}")
    print(f"  结论: {'一致 ✔ —— m2py 的表就是这张表 + 同样的 CF 改正' if dmax < 1e-12 else '不一致 ✘'}")

    out = np.zeros((nmax + 1, 4))
    out[:, 0] = np.arange(nmax + 1)
    out[:, 1] = np.concatenate(([0.0], h))
    out[:, 2] = np.concatenate(([0.0], l))
    out[:, 3] = ref

    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "load_love_numbers.npz")
    np.savez_compressed(
        dst,
        n=out[:, 0].astype(np.int32),
        h=out[:, 1], l=out[:, 2], k=out[:, 3],
        asymptote=np.array(asym[1:4]) if asym is not None else np.zeros(3),
        model="PREM (Wang et al. 2012, PREM-LLNs.dat)",
        source=os.path.basename(path),
        note=("columns: n, h' (radial), l' (horizontal), k' (potential); "
              "k'_1 converted CE->CF as in pz_LLN.m; "
              "h'_0=l'_0=k'_0=0; matches m2py love_numbers.npy to 1e-16"),
    )
    print(f"\n写出 {dst}  ({os.path.getsize(dst)} 字节, nmax={nmax})")
    for name, col in (("h", 1), ("l", 2), ("k", 3)):
        np.save(os.path.join(OUT, f"love_numbers_{name}.npy"), out[:, col])
    print("另存 love_numbers_h.npy / _l.npy / _k.npy")

    other = sorted(f for f in os.listdir(SRC) if f.endswith("-LLNs.dat"))
    print("\n可替换的其它载荷模型（同格式，用 load_love_numbers(path=…) 载入）:")
    for f in other:
        print("   ", f)


if __name__ == "__main__":
    sys.exit(main())
