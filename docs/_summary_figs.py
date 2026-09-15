# -*- coding: utf-8 -*-
"""Step 2: all matplotlib figures for SHKit方法总结.html.

Reads _measurements.json / _data.npz produced by _summary_measure.py.
Chinese labels need SimHei.  Every figure is wrapped so one failure does not
kill the rest.
"""
import json
import os
import sys
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SHKIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SHKIT)
FIGS = os.path.join(SHKIT, "docs", "SHKit方法总结_figs")

M = json.load(open(os.path.join(FIGS, "_measurements.json"), encoding="utf-8"))
D = np.load(os.path.join(FIGS, "_data.npz"), allow_pickle=False)
T = M["yangtze"]
FAILS = []


def _ascii_ticks(fig):
    """Force plain-ASCII log tick labels.

    matplotlib renders log ticks with mathtext (``$10^{-2}$``), whose minus sign
    is U+2212 -- a glyph SimHei does not have.  ``axes.unicode_minus=False`` does
    not cover that path, so the major formatter is replaced outright.
    """
    import matplotlib.ticker as mticker
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                axis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
                axis.set_minor_formatter(mticker.NullFormatter())


def save(fig, name):
    p = os.path.join(FIGS, name)
    _ascii_ticks(fig)
    fig.tight_layout()
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"    wrote {name} ({os.path.getsize(p)/1024:.0f} kB)", flush=True)


def guard(fn):
    def wrap():
        try:
            fn()
        except Exception:
            traceback.print_exc()
            FAILS.append(fn.__name__)
    return wrap


# ---------------------------------------------------------------- 01 weights
@guard
def f01_weights_bar():
    w = T["weights"]
    tags = ["lattice\n(每点一个格元)", "grid\n(中点格边)", "delaunay\n(球面凸包)"]
    keys = ["lattice", "grid", "delaunay"]
    sums = [w[k]["sum_sr"] for k in keys]
    areas = [w[k]["area_km2"] for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    b = ax[0].bar(tags, sums, color=["#2b7a4b", "#c98a2b", "#a33b3b"])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Σw  (立体角 sr，对数轴)")
    ax[0].set_title("长江掩膜：三种积分元口径的总权重")
    for r, k in zip(b, keys):
        ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() * 1.25,
                   f"{w[k]['sum_sr']:.6e}\n{w[k]['ratio_vs_lattice']:.4g}×",
                   ha="center", va="bottom", fontsize=8.5)
    ax[0].set_ylim(top=max(sums) * 12)
    b2 = ax[1].bar(tags, areas, color=["#2b7a4b", "#c98a2b", "#a33b3b"])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("覆盖面积 (km$^2$，对数轴)")
    ax[1].set_title("换算成面积")
    for r, k in zip(b2, keys):
        ax[1].text(r.get_x() + r.get_width() / 2, r.get_height() * 1.25,
                   f"{w[k]['area_km2']:,.0f}", ha="center", va="bottom",
                   fontsize=8.5)
    ax[1].set_ylim(top=max(areas) * 12)
    fig.suptitle("图1  长江掩膜的积分元：lattice 是唯一正确的口径"
                 f"（参考值 {w['tool_csv']['area_km2']:,.2f} km$^2$，由逐点格元独立算出）",
                 fontsize=11)
    save(fig, "fig01_weights_bar.png")


# ------------------------------------------------------------ 02 weight ratio
@guard
def f02_weight_ratio_hist():
    w_lat = D["y_w_lattice"]
    w_grd = D["y_w_grid"]
    r = w_grd / w_lat
    st = T["grid_vs_lattice_ratio"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    bins = np.logspace(np.log10(r.min() * 0.999), np.log10(r.max() * 1.05), 160)
    ax[0].hist(r, bins=bins, color="#c98a2b", edgecolor="none")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].axvline(1.0, color="#2b7a4b", lw=1.4, ls="--",
                  label="理想比 1.0")
    ax[0].axvline(r.max(), color="#a33b3b", lw=1.4,
                  label=f"最大 {r.max():.0f}×")
    ax[0].set_xlabel("w_grid / w_lattice  （对数轴）")
    ax[0].set_ylabel("点数（对数轴）")
    ax[0].set_title("逐点权重比值：不是纯比例缩放")
    ax[0].legend(fontsize=8.5, loc="center left")
    ax[0].text(0.60, 0.96,
               f"median {st['median']:.6f}\n"
               f"p99 {st['p99']:.6f}\n"
               f"std/mean {st['std_over_mean']:.3f}\n"
               f">{1.1}× 的点：{st['n_gt_1p1']} 个"
               f"（{st['n_gt_1p1']/st['n_points']*100:.3f}%）",
               transform=ax[0].transAxes, va="top", fontsize=8.5,
               bbox=dict(fc="white", alpha=0.88, ec="none", pad=2.5))
    lat_u = np.unique(D["y_lat"])
    gaps = np.diff(lat_u)
    ax[1].semilogy(np.arange(gaps.size), gaps, ".", ms=2.5, color="#5b6b7c")
    ax[1].axhline(np.median(gaps), color="#2b7a4b", lw=1.2, ls="--",
                  label=f"格距中位数 {np.median(gaps):.7f}°")
    ax[1].set_xlabel("纬度序号")
    ax[1].set_ylabel("相邻唯一纬度的间隔 (°，对数轴)")
    ax[1].set_title(f"纬度空档：{T['n_big_lat_gaps']} 条宽于一步，最宽 "
                    f"{gaps.max()/np.median(gaps):.0f} 倍")
    ax[1].legend(fontsize=8.5)
    fig.suptitle("图2  grid 把格边放在相邻现存坐标的中点，紧邻空档的点于是吞掉"
                 "空档的一半 → 不仅尺度偏大 4.59%，形状也变", fontsize=11)
    save(fig, "fig02_weight_ratio_hist.png")


# ------------------------------------------------------- 03 objective domains
@guard
def f03_objective_domain():
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.6))
    rng = np.random.default_rng(0)
    for k, (title, sub) in enumerate([
            ("零填充全球 L2 投影  method='projection' / 'quadrature'",
             "min ∫$_{\\Omega}$ ( f - A x )$^2$ dΩ      Ω = 整个球面"),
            ("逐点加权最小二乘  method='wlsq' / 'cg'",
             "min Σ$_i$ w$_i$ ( (A x)$_i$ - f$_i$ )$^2$      求和只在测点上")]):
        a = ax[k]
        a.add_patch(Circle((0, 0), 1.0, facecolor="#eef3f7",
                           edgecolor="#5b6b7c", lw=1.2))
        th = rng.uniform(0, 2 * np.pi, 900)
        rr = np.sqrt(rng.uniform(0, 1, 900))
        a.plot(rr * np.cos(th), rr * np.sin(th), ".", ms=1.2,
               color="#9fb0c0", alpha=0.5)
        cx, cy = 0.30, 0.28
        th2 = rng.uniform(0, 2 * np.pi, 120)
        rr2 = np.sqrt(rng.uniform(0, 1, 120)) * 0.16
        px, py = cx + rr2 * np.cos(th2), cy + rr2 * np.sin(th2)
        if k == 0:
            a.plot(px, py, ".", ms=3.0, color="#a33b3b")
            a.add_patch(Circle((cx, cy), 0.20, facecolor="none",
                               edgecolor="#a33b3b", lw=1.4, ls="--"))
            a.text(0, -1.20, "区域外 f ≡ 0 —— 这些隐含的 0 在目标函数里\n"
                             "（占球面 99.9998%，通过 4π 归一化进入分母）",
                   ha="center", fontsize=9, color="#a33b3b")
            a.text(-0.62, 0.62, "测点 f=1", fontsize=9, color="#a33b3b")
        else:
            a.plot(px, py, ".", ms=3.0, color="#2b7a4b")
            a.text(0, -1.20, "区域外 f ≡ 0 —— 这些隐含的 0 不在目标函数里\n"
                             "（没有求和项，求解器看不到它们）",
                   ha="center", fontsize=9, color="#a33b3b")
            a.text(-0.62, 0.62, "测点 f=1", fontsize=9, color="#2b7a4b")
        a.set_xlim(-1.25, 1.25)
        a.set_ylim(-1.45, 1.15)
        a.set_aspect("equal")
        a.axis("off")
        a.set_title(f"{title}\n{sub}", fontsize=9.5)
    fig.suptitle("图3  两个目标函数的定义域不同——这是全部问题的根源", fontsize=11)
    save(fig, "fig03_objective_domain.png")


# ---------------------------------------------------------- 04 K eigenspectrum
@guard
def f04_k_spectrum():
    from shkit import weights as W
    from shkit.basis import design_matrix_full
    from shkit.diagnostics import gram_matrix
    ws = W.compute_weights(D["y_lat"], D["y_lon"], rule="lattice")
    A = design_matrix_full(D["y_lat"], D["y_lon"], 12)
    lam = np.linalg.eigvalsh(gram_matrix(A, ws.w))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax[0].semilogy(np.arange(1, lam.size + 1), np.sort(lam)[::-1], "o-",
                   ms=3, lw=0.8, color="#2b5f8c")
    ax[0].axhline(1.0, color="#2b7a4b", ls="--", lw=1.2, label="λ = 1")
    ax[0].axhline(0.5, color="#c98a2b", ls="--", lw=1.2, label="λ = 0.5")
    ax[0].axvline(17, color="#a33b3b", lw=1.2, ls=":",
                  label="第 17 大之后全部 < 1e-2")
    ax[0].set_xlabel("特征值序号（降序）")
    ax[0].set_ylabel("K 的特征值 λ（对数轴）")
    ax[0].set_title(f"长江 L=12 的完备性矩阵谱\n"
                    f"λ$_{{max}}$ = {lam.max():.3e}，Σλ = {lam.sum():.6e} "
                    f"= Shannon 数")
    ax[0].legend(fontsize=8.5)
    flags = sorted([(k, v) for k, v in T["gram"]["n_above"].items()],
                   key=lambda x: -float(x[0]))
    ax[1].bar([f"λ ≥ {k}" for k, _ in flags], [v for _, v in flags],
              color="#2b5f8c")
    ax[1].axhline(169, color="#5b6b7c", ls="--", lw=1,
                  label="系数总数 169")
    ax[1].set_ylabel("满足条件的特征值个数")
    ax[1].set_title("只有极少数方向对数据“可见”")
    for i, (_, v) in enumerate(flags):
        ax[1].text(i, v + 2, str(v), ha="center", fontsize=8.5)
    ax[1].legend(fontsize=8.5)
    plt.setp(ax[1].get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.suptitle("图4  K = A$^T$WA/4π 的谱：绝大多数方向数值上不可见 —— "
                 "许多不同的系数向量能同样好地拟合样本", fontsize=11)
    save(fig, "fig04_k_spectrum.png")


# ------------------------------------------------------- 05 iteration drift
@guard
def f05_iteration_drift():
    d = T["iterative_drift"]
    e = T["estimators"]
    n = [r["niter"] for r in d]
    res = [r["residual"] for r in d]
    ratio = [r["ratio"] for r in d]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax[0].plot(n, res, "o-", color="#2b5f8c", label="Richardson 迭代")
    ax[0].plot([max(n) * 1.35], [e["cg"]["residual_rms"]], "s",
               color="#a33b3b", ms=9,
               label=f"CG 收敛 ({e['cg']['cg_info']})")
    ax[0].set_xlabel("迭代次数 niter")
    ax[0].set_ylabel("样本残差 RMS（对数轴）")
    ax[0].set_yscale("log")
    ax[0].set_title("样本残差：Richardson 只从 1.00 压到 0.53；\nCG 收敛那一步才趋零")
    ax[0].legend(fontsize=8.5)
    ax[0].annotate("趋零是 CG 干的，不是迭代的功劳",
                   xy=(max(n) * 1.35, e["cg"]["residual_rms"]), fontsize=8.5,
                   color="#a33b3b", ha="center", va="bottom",
                   xytext=(max(n) * 0.52, 3e-5),
                   arrowprops=dict(arrowstyle="->", color="#a33b3b", lw=1))
    ax[1].plot(n, ratio, "o-", color="#a33b3b")
    ax[1].plot([max(n) * 1.35], [e["cg"]["ratio_vs_area"]], "s",
               color="#a33b3b", ms=9)
    ax[1].axhline(1.0, color="#2b7a4b", ls="--", lw=1.4,
                  label="几何要求的 C00（= 面积占比）")
    ax[1].set_xlabel("迭代次数 niter")
    ax[1].set_ylabel("C00 / 几何要求值（对数轴）")
    ax[1].set_yscale("log")
    ax[1].set_title("同一批迭代里，C00 单调膨胀")
    ax[1].legend(fontsize=8.5, loc="center left")
    ax[1].set_ylim(0.7, e["cg"]["ratio_vs_area"] * 60)
    ax[1].annotate(f"CG 收敛：{e['cg']['ratio_vs_area']:.0f}×\n"
                   f"隐含 {e['cg']['implied_area_km2']:,.0f} km$^2$",
                   xy=(max(n) * 1.35, e["cg"]["ratio_vs_area"]), fontsize=8.5,
                   color="#a33b3b", ha="center", va="bottom",
                   xytext=(max(n) * 0.55, e["cg"]["ratio_vs_area"] * 1.6),
                   arrowprops=dict(arrowstyle="->", color="#a33b3b", lw=1))
    fig.suptitle("图5  残差与物理误差严格反相关：残差每降一档，C00 就远离真值一档",
                 fontsize=11)
    save(fig, "fig05_iteration_drift.png")


# ------------------------------------------------------- 06 regularization
@guard
def f06_regularization():
    rows = T["regularization"]
    lab = [("无正则\n(reg=None)" if r["reg"] is None
            else f"{r['reg']}\nα={r['alpha']:g}") for r in rows]
    area = [r["implied_area_km2"] for r in rows]
    cols = ["#a33b3b" if r["reg"] is None or r["reg"] == "tikhonov"
            else "#2b7a4b" for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))
    b = ax[0].bar(lab, area, color=cols)
    ax[0].axhline(T["weights"]["lattice"]["area_km2"], color="#2b7a4b",
                  ls="--", lw=1.4,
                  label=f"几何真值 {T['weights']['lattice']['area_km2']:,.0f} km$^2$")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("C00 隐含的 1-set 面积 (km$^2$，对数轴)")
    ax[0].set_title("长江 L=12：正则化对 C00 的影响")
    ax[0].legend(fontsize=8.5)
    for r, v in zip(b, area):
        ax[0].text(r.get_x() + r.get_width() / 2, v * 1.3, f"{v:,.0f}",
                   ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax[0].set_ylim(top=max(area) * 40)
    plt.setp(ax[0].get_xticklabels(), fontsize=7.5)
    rr = M["regional30"]["reg_coef_rel_err"]
    keys = list(rr.keys())
    b2 = ax[1].bar([k.replace("-", "\n") for k in keys],
                   [rr[k] for k in keys],
                   color=["#a33b3b", "#c98a2b", "#2b7a4b"])
    ax[1].axhline(1.0, color="#5b6b7c", ls="--", lw=1.2,
                  label="误差 1 = 完全没恢复")
    ax[1].set_ylabel("系数相对误差（对数轴）")
    ax[1].set_yscale("log")
    ax[1].set_title(f"30° 球冠区域反演（覆盖 "
                    f"{M['regional30']['coverage']:.3f}）")
    ax[1].legend(fontsize=8.5)
    for r, k in zip(b2, keys):
        ax[1].text(r.get_x() + r.get_width() / 2, r.get_height() * 1.3,
                   f"{rr[k]:.3g}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylim(top=max(rr.values()) * 30)
    plt.setp(ax[1].get_xticklabels(), fontsize=7.5)
    fig.suptitle("图6  正则化：kaula 有效，tikhonov 在区域数据上几乎不起作用",
                 fontsize=11)
    save(fig, "fig06_regularization.png")


# ---------------------------------------------------------- 07 cap floor
@guard
def f07_cap_floor():
    fl = M["cap"]["floor"]
    Ls = [r["L"] for r in fl]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax[0].semilogy(Ls, [r["quad"] for r in fl], "o-", color="#2b5f8c",
                   label="求积（0 次迭代）")
    ax[0].semilogy(Ls, [r["it3"] for r in fl], "x--", color="#c98a2b",
                   label="+3 次迭代")
    ax[0].semilogy(Ls, [r["it25"] for r in fl], "+:", color="#a33b3b",
                   label="+25 次迭代（L≤12）")
    ax[0].plot([], [], " ", label="三条线完全重合（8 位小数不动）")
    ax[0].set_xlabel("截断阶数 L")
    ax[0].set_ylabel("样本残差 RMS（对数轴）")
    ax[0].set_title("全局 0/1 帽：残差停在截断地板，迭代动不了它")
    ax[0].legend(fontsize=8.5, loc="center right")
    for x, r in zip(Ls, fl):
        ax[0].annotate(f"{r['quad']:.4f}", (x, r["quad"]),
                       textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=7.5)
    ax[0].set_ylim(top=max(r["quad"] for r in fl) * 3.2)
    ax[1].semilogy(Ls, [r["global_rel"] for r in fl], "s-", color="#2b7a4b")
    ax[1].set_xlabel("截断阶数 L")
    ax[1].set_ylabel("面积加权相对截断误差（对数轴）")
    ax[1].set_title("这就是泄漏：随 L 下降，但永不消失")
    for x, y in zip(Ls, [r["global_rel"] for r in fl]):
        ax[1].annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                       xytext=(4, 6), fontsize=8)
    ax[1].set_ylim(top=max(r["global_rel"] for r in fl) * 6)
    fig.suptitle("图7  良态对照（球面覆盖率 = 1.000000）：截断泄漏是必然的，"
                 "而迭代算法尊重这个地板", fontsize=11)
    save(fig, "fig07_cap_floor.png")


# ---------------------------------------------------------------- 08 noise
@guard
def f08_noise():
    rows = M["noise"]["rows"]
    sig = [r["sigma_rel"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for tag, c, mk in (("quadrature", "#2b5f8c", "o-"),
                       ("iterative", "#c98a2b", "s-"),
                       ("wlsq", "#a33b3b", "^-")):
        ax[0].semilogy(sig, [max(r[tag], 1e-17) for r in rows], mk,
                       color=c, label=tag)
    ax[0].semilogy(sig, [max(r["theory"], 1e-17) for r in rows], "k--",
                   label="理论 (L+1)/√N·σ$_{rel}$")
    ax[0].set_xlabel("相对噪声 σ")
    ax[0].set_ylabel("系数相对误差（对数轴）")
    ax[0].set_title(f"噪声行为（真值带限 40 阶，求解 {M['noise']['L_solve']} 阶，"
                    f"N={M['fib']['n']}）")
    ax[0].legend(fontsize=8.5)
    ax[0].set_xticks(sig)
    ax[1].axis("off")
    txt = ["σ = 0：三法差别只在截断偏差",
           f"  quadrature {rows[0]['quadrature']:.3e}",
           f"  +迭代      {rows[0]['iterative']:.3e}",
           f"  wlsq       {rows[0]['wlsq']:.3e}",
           "",
           "σ = 1%：三法到第 4 位有效数字以内",
           f"  {rows[1]['quadrature']:.4e} / {rows[1]['iterative']:.4e} / "
           f"{rows[1]['wlsq']:.4e}",
           f"  理论 {rows[1]['theory']:.4e}",
           "",
           "σ = 5%：同样几乎无差别",
           f"  {rows[2]['quadrature']:.4e} / {rows[2]['iterative']:.4e} / "
           f"{rows[2]['wlsq']:.4e}",
           f"  理论 {rows[2]['theory']:.4e}",
           "",
           "结论：有噪声时起决定作用的是阶数，不是估计量。"]
    ax[1].text(0.0, 0.95, "\n".join(txt), va="top", fontsize=9,
               family="SimHei")
    fig.suptitle("图8  噪声主导时，三种估计量等价", fontsize=11)
    save(fig, "fig08_noise.png")


# ---------------------------------------------------------- 09 fib roundtrip
@guard
def f09_fib_roundtrip():
    rows = M["fib"]["rows"]
    tags = [r["tag"] for r in rows]
    errs = [max(r["coef_rel_err"], 1e-17) for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    b = ax.bar(tags, errs, color=["#2b5f8c", "#c98a2b", "#8a6fb0", "#a33b3b",
                                 "#2b7a4b"])
    ax.set_yscale("log")
    ax.set_ylabel("系数相对误差（对数轴）")
    ax.set_title(f"准均匀全球散点（Fibonacci {M['fib']['n']} 点，L={M['fib']['L']}）："
                 "迭代校正逐步逼近最小二乘")
    for r, v in zip(b, errs):
        ax.text(r.get_x() + r.get_width() / 2, v * 1.4, f"{v:.3e}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(top=max(errs) * 60)
    ax.text(0.02, 0.06,
            "求积 → +1 次迭代 → +3 次迭代 → wlsq 逐级收敛；\n"
            "这正是“迭代校正的不动点就是加权最小二乘解”。",
            transform=ax.transAxes, fontsize=9, va="bottom")
    fig.suptitle("图9  采样准正交时（max|K-I| = "
                 f"{[r['max_K_minus_I'] for r in M['routing'] if 'Fibonacci' in r['name']][0]:.1e}）"
                 "迭代校正在数值上等价于最小二乘", fontsize=10.5)
    save(fig, "fig09_fib_roundtrip.png")


# -------------------------------------------------------------- 10 routing
@guard
def f10_routing():
    rows = M["routing"]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ypos = np.arange(len(rows))[::-1]
    vals = [r["max_K_minus_I"] for r in rows]
    cols = ["#2b7a4b" if r["auto"] in ("quadrature", "projection")
            else "#a33b3b" for r in rows]
    ax.barh(ypos, vals, color=cols)
    ax.set_xscale("log")
    ax.axvline(1e-2, color="#5b6b7c", ls="--", lw=1.4,
               label="auto 阈值 max|K-I| = 1e-2")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['name']}\nN={r['n']}" for r in rows], fontsize=8.5)
    ax.set_xlabel("max|K - I|（对数轴）")
    for y, r in zip(ypos, rows):
        ax.text(r["max_K_minus_I"] * 1.6, y,
                f"{r['max_K_minus_I']:.2e} → {r['auto']}", va="center",
                fontsize=9)
    ax.set_xlim(left=min(vals) * 0.2, right=max(vals) * 900)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("图10  method='auto' 的选路：max|K-I| < 1e-2 且覆盖率为 1 时才用求积",
                 fontsize=11)
    save(fig, "fig10_routing.png")


# ------------------------------------------------------ 11 yangtze spectrum
@guard
def f11_yangtze_spectrum():
    sp = T["spectrum"]
    n = sp["n"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax[0].semilogy(n, sp["proj_degree_rms"], "o-", color="#2b7a4b",
                   label="projection（正确：零填充全球投影）")
    ax[0].semilogy(n, sp["cg_degree_rms"], "s--", color="#a33b3b",
                   label="cg（存储的 .gfc 路径）")
    ax[0].set_xlabel("阶数 n")
    ax[0].set_ylabel("逐阶 RMS（对数轴）")
    ax[0].set_title("长江掩膜的逐阶谱：n = 0..12 近乎平坦")
    ax[0].legend(fontsize=8.5)
    ax[0].axhline(T["shannon_L12"] / (13 * 13) ** 0.5, color="#5b6b7c",
                  ls=":", lw=1)
    ax[1].bar(n, [v * 100 for v in sp["proj_power_share"]], color="#2b5f8c")
    ax[1].set_xlabel("阶数 n")
    ax[1].set_ylabel("功率占比 (%)")
    ax[1].set_title("功率分布在所有阶上（0/1 掩膜本来就是宽谱）")
    ax[1].text(0.35, 0.80,
               f"n=0 占 {sp['proj_power_share'][0]*100:.2f}%\n"
               f"n=1..12 合计 {sum(sp['proj_power_share'][1:])*100:.2f}%",
               transform=ax[1].transAxes, fontsize=9)
    fig.suptitle("图11  0/1 掩膜没有“衰减的谱”：L=12 只留下很小一片能量",
                 fontsize=11)
    save(fig, "fig11_yangtze_spectrum.png")


# ----------------------------------------------------------- 12 yangtze map
@guard
def f12_yangtze_map():
    glat, glon = D["gy_lat"], D["gy_lon"]
    GY, CG = D["gy_proj"], D["gy_cg"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    sub = slice(None, None, 6)
    axes[0].plot(D["y_lon"][sub], D["y_lat"][sub], ".", ms=0.35,
                 color="#333333")
    axes[0].set_title("输入：142846 个点，value 全为 1\n（隐含：别处全为 0）",
                      fontsize=9.5)
    vmax = max(abs(GY).max(), 1e-12)
    im1 = axes[1].pcolormesh(glon, glat, GY, cmap="viridis",
                             vmin=-vmax, vmax=vmax, shading="auto")
    axes[1].set_title("projection, L=12 重建\n"
                      f"峰值仅 {GY.max():.3e}（真值 1）", fontsize=9.5)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    vmax2 = max(abs(CG).max(), 1e-12)
    im2 = axes[2].pcolormesh(glon, glat, CG, cmap="viridis",
                             vmin=-vmax2, vmax=vmax2, shading="auto")
    axes[2].set_title("cg, L=12 重建（错误路径）\n"
                      f"峰值 {CG.max():.3f}——“看起来对了”", fontsize=9.5)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for a in axes:
        a.set_xlabel("经度 (°E)")
        a.set_ylabel("纬度 (°N)")
        a.tick_params(labelsize=8)
    fig.suptitle("图12  同一个 L=12：正确的投影只给出 ~3.5e-4 的团（这就是泄漏）；"
                 "CG 把整块区域抬到 ≈1，但代价是 C00 错 1 万倍\n"
                 "注意左右两幅的色标量级不同（左 ~1e-4，右 ~1）",
                 fontsize=10.5)
    save(fig, "fig12_yangtze_map.png")


# --------------------------------------------------------- 13 shannon sweep
@guard
def f13_shannon_sweep():
    sw = T["shannon_sweep"]
    Ls = [r["L"] for r in sw]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax[0].loglog(Ls, [r["shannon"] for r in sw], "o-", color="#2b5f8c",
                 label="Shannon 数 (L+1)$^2$·Ω/4π")
    ax[0].loglog(Ls, [r["peak"] for r in sw], "s-", color="#2b7a4b",
                 label="实测重建峰值")
    ax[0].axhline(1.0, color="#a33b3b", ls="--", lw=1.4,
                  label="掩膜要“立”到 1")
    Lneed = T["L_for_mask_peak_1"]
    ax[0].axvline(Lneed, color="#a33b3b", ls=":", lw=1.2)
    ax[0].annotate(f"L ≈ {Lneed:.0f}", (Lneed, 1.0), fontsize=9,
                   color="#a33b3b", xytext=(Lneed * 0.16, 3.0),
                   arrowprops=dict(arrowstyle="->", color="#a33b3b", lw=1))
    ax[0].set_xlabel("截断阶数 L")
    ax[0].set_ylabel("幅度（对数轴）")
    ax[0].set_title("峰值紧贴 Shannon 数")
    ax[0].legend(fontsize=8.5)
    ax[1].loglog(Ls, [r["peak"] / r["shannon"] for r in sw], "o-",
                 color="#8a6fb0")
    ax[1].set_xlabel("截断阶数 L")
    ax[1].set_ylabel("峰值 / Shannon 数")
    ax[1].set_title("比值 ≈ 1：截断掩膜的峰值就是 Shannon 数")
    for x, r in zip(Ls, sw):
        ax[1].annotate(f"{r['peak']/r['shannon']:.3f}", (x, r["peak"]/r["shannon"]),
                       textcoords="offset points", xytext=(3, 6), fontsize=8)
    ax[1].set_ylim(0.3, 3)
    fig.suptitle("图13  长江掩膜：L=12 时峰值 3.46e-4，而 Shannon 数 3.50e-4 —— "
                 "这块区域在这个阶数下连一个自由度都撑不住", fontsize=10.5)
    save(fig, "fig13_shannon_sweep.png")


# -------------------------------------------------------------- 14 cap map
@guard
def f14_cap_map():
    lats = np.linspace(-89.5, 89.5, 180)
    lons = np.linspace(0.5, 359.5, 360)
    LO, LA = np.meshgrid(lons, lats)
    from shkit.synthesis import synthesis_grid
    from shkit.coeffs import SHCoeffs
    co = SHCoeffs(D["cap_c12_C"], D["cap_c12_S"])
    G = np.asarray(synthesis_grid(lats, lons, co, nmax=12))
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.4))
    ax[0].pcolormesh(lons, lats, _cap_field(LA, LO), cmap="Greys",
                     vmin=0, vmax=1, shading="auto")
    ax[0].set_title("真值：20° 半径的 0/1 帽\n面积占比 "
                    f"{M['cap']['area_fraction']:.5f}", fontsize=9.5)
    im = ax[1].pcolormesh(lons, lats, G, cmap="RdBu_r",
                          vmin=-np.abs(G).max(), vmax=np.abs(G).max(),
                          shading="auto")
    ax[1].set_title("L=12 投影重建\n面积加权相对截断误差 "
                    f"{M['cap']['floor'][3]['global_rel']:.4f}", fontsize=9.5)
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    for a in ax:
        a.set_xlabel("经度 (°E)")
        a.set_ylabel("纬度 (°N)")
        a.tick_params(labelsize=8)
    fig.suptitle("图14  良态对照：即使覆盖全球，L=12 也重建不出一个 20° 的 0/1 帽"
                 "——这就是截断泄漏的真面目", fontsize=10.5)
    save(fig, "fig14_cap_map.png")


def _cap_field(LA, LO):
    cosd = (np.sin(np.deg2rad(LA)) * np.sin(np.deg2rad(30.0)) +
            np.cos(np.deg2rad(LA)) * np.cos(np.deg2rad(30.0)) *
            np.cos(np.deg2rad(LO - 110.0)))
    return (cosd >= np.cos(np.deg2rad(20.0))).astype(float)


# ------------------------------------------------- 15 slepian vs wlsq
@guard
def f15_slepian_vs_wlsq():
    s = M["slepian"]
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.8))
    groups = [("条件数", s["slepian_cond"], s["wlsq_cond"]),
              ("‖C‖（真值 %.2f）" % s["slepian_inside_truth_norm"],
               s["slepian_normC"], s["wlsq_normC"]),
              ("区域外场量 / 区域内 RMS",
               s["slepian_outside_rel"], s["wlsq_outside_rel"])]
    for a, (title, a1, a2) in zip(ax, groups):
        b = a.bar(["Slepian\n(λ>0.5)", "未正则\nWLSQ"],
                  [max(a1, 1e-30), max(a2, 1e-30)],
                  color=["#2b7a4b", "#a33b3b"])
        a.set_yscale("log")
        a.set_title(title, fontsize=9.5)
        for r, v in zip(b, [a1, a2]):
            a.text(r.get_x() + r.get_width() / 2, v * 1.5, f"{v:.3g}",
                   ha="center", va="bottom", fontsize=8.5)
        a.set_ylim(top=max(a1, a2) * 60)
    fig.suptitle(f"图15  {s['L']} 阶、{s['n_points']} 点、25° 球冠（覆盖 "
                 f"{s['coverage']:.3f}）：Slepian 截断把病态显式化后，每一项都"
                 "低几个数量级", fontsize=10.5)
    save(fig, "fig15_slepian_vs_wlsq.png")


# ------------------------------------------------- 16 voronoi trap
@guard
def f16_voronoi_trap():
    v = M["voronoi_trap"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    vals = [v["true_cap_area_sr"], v["delaunay_sum_sr"], v["voronoi_sum_sr"]]
    labs = ["球冠真实面积\n2π(1-cos r)", "delaunay\n(球面凸包)",
            "区域子集做\nVoronoi"]
    b = ax.bar(labs, vals, color=["#2b7a4b", "#c98a2b", "#a33b3b"])
    ax.axhline(4 * np.pi, color="#5b6b7c", ls="--", lw=1.2, label="4π")
    for r, val, ratio in zip(b, vals,
                             [1.0, v["delaunay_ratio"], v["voronoi_ratio"]]):
        ax.text(r.get_x() + r.get_width() / 2, val + 0.3,
                f"{val:.4f} sr\n{ratio:.2f}×", ha="center", fontsize=9)
    ax.set_ylabel("Σw（立体角 sr）")
    ax.set_ylim(top=4 * np.pi * 1.22)
    ax.legend(fontsize=8.5)
    ax.set_title(f"30° 球冠（{v['n_points']} 点，真实面积 "
                 f"{v['true_cap_area_sr']:.4f} sr = 占全球 "
                 f"{v['true_cap_fraction']*100:.2f}%）\n"
                 "区域点子集做球面 Voronoi，胞永远铺满整个球面 → Σw 恒等于 4π",
                 fontsize=10)
    fig.suptitle("图16  区域 Voronoi 陷阱：虚高 14.93 倍，绝对不能用", fontsize=11)
    save(fig, "fig16_voronoi_trap.png")


for fn in (f01_weights_bar, f02_weight_ratio_hist, f03_objective_domain,
           f04_k_spectrum, f05_iteration_drift, f06_regularization,
           f07_cap_floor, f08_noise, f09_fib_roundtrip, f10_routing,
           f11_yangtze_spectrum, f12_yangtze_map, f13_shannon_sweep,
           f14_cap_map, f15_slepian_vs_wlsq, f16_voronoi_trap):
    print(f"  -> {fn.__name__}", flush=True)
    fn()

print(f"\nfailures: {FAILS if FAILS else 'none'}")
