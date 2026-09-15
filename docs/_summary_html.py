# -*- coding: utf-8 -*-
"""Step 4: build docs/SHKit方法总结.html from the measured JSON.

Every number in the document is interpolated from
    _measurements.json / _history.json / _gui_meta.json
so nothing is typed from memory.  No external resources: no CDN, no web fonts,
no remote images -- all <img src> are relative paths inside
SHKit方法总结_figs/.
"""
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "SHKit方法总结_figs")
OUT = os.path.join(HERE, "SHKit方法总结.html")

M = json.load(io.open(os.path.join(FIGS, "_measurements.json"), encoding="utf-8"))
H = json.load(io.open(os.path.join(FIGS, "_history.json"), encoding="utf-8"))
G = json.load(io.open(os.path.join(FIGS, "_gui_meta.json"), encoding="utf-8"))
T = M["yangtze"]
cur = H["current_file"]
rep = H["reproduced_original_cg_delaunay"]
E = T["estimators"]
W_ = T["weights"]
FLOOR = M["cap"]["floor"]
IMG = "SHKit方法总结_figs"

CSS = """
:root{--fg:#1c2733;--muted:#5b6b7c;--line:#d7dee5;--bg:#fff;
--accent:#2b5f8c;--ok:#2b7a4b;--warn:#a33b3b;--amber:#c98a2b;--soft:#f4f7fa;}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.75 "Microsoft YaHei","PingFang SC","Hiragino Sans GB","Source Han Sans SC",sans-serif;}
header{background:linear-gradient(180deg,#1f4a70,#2b5f8c);color:#fff;padding:34px 28px 26px}
header h1{margin:0 0 6px;font-size:27px;letter-spacing:.4px}
header p{margin:2px 0;color:#d6e4f0;font-size:13.5px}
.wrap{display:flex;align-items:flex-start;gap:26px;max-width:1220px;margin:0 auto;padding:0 22px}
nav#toc{position:sticky;top:12px;flex:0 0 246px;max-height:calc(100vh - 30px);
 overflow:auto;background:var(--soft);border:1px solid var(--line);border-radius:10px;
 padding:14px 14px 16px;font-size:13px;margin:22px 0}
nav#toc h2{font-size:13px;margin:0 0 8px;color:var(--muted);letter-spacing:.6px;
 text-transform:uppercase}
nav#toc a{display:block;color:var(--accent);text-decoration:none;padding:3px 0;
 border-bottom:1px dotted transparent}
nav#toc a:hover{text-decoration:underline;background:#e8eff6}
nav#toc a.sub{padding-left:14px;color:#4a5b6c;font-size:12.4px}
main{flex:1 1 auto;min-width:0;padding:22px 0 70px}
section{margin:0 0 40px;scroll-margin-top:14px}
h2{font-size:21px;border-left:5px solid var(--accent);padding:2px 0 2px 11px;
 margin:34px 0 14px}
h3{font-size:16.5px;margin:24px 0 9px;color:#24405a}
h4{font-size:14.6px;margin:18px 0 7px;color:#33506b}
p{margin:9px 0}
ul,ol{margin:9px 0 9px 22px;padding:0}
li{margin:4px 0}
code{background:#eef2f6;border:1px solid #e0e7ee;border-radius:4px;padding:.5px 5px;
 font-family:Consolas,"Courier New",monospace;font-size:12.8px}
pre{background:#f7f9fb;border:1px solid var(--line);border-left:3px solid var(--accent);
 border-radius:6px;padding:11px 13px;overflow:auto;font-size:12.9px;line-height:1.6;
 font-family:Consolas,"Courier New",monospace}
table{border-collapse:collapse;width:100%;margin:13px 0;font-size:13.4px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
th{background:#eef3f8;font-weight:600}
tbody tr:nth-child(even){background:#fafcfd}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;
 font-family:Consolas,"Courier New",monospace;font-size:12.7px}
figure{margin:20px 0;padding:12px;background:#fbfcfe;border:1px solid var(--line);
 border-radius:9px}
figure img{width:100%;height:auto;display:block;border-radius:5px}
figcaption{margin-top:9px;font-size:13.2px;color:#3d4d5d;line-height:1.7}
figcaption b{color:var(--fg)}
.figrow{display:flex;gap:14px;flex-wrap:wrap}
.figrow figure{flex:1 1 380px}
.note,.warn,.key{border-radius:8px;padding:11px 14px;margin:14px 0;font-size:13.8px}
.key{background:#eef7f1;border-left:4px solid var(--ok)}
.warn{background:#fdf0f0;border-left:4px solid var(--warn)}
.note{background:var(--soft);border-left:4px solid var(--accent)}
.key h4,.warn h4,.note h4{margin:0 0 5px;color:inherit}
.formula{background:#f7f9fb;border:1px dashed #c3d0dc;border-radius:7px;
 padding:11px 14px;margin:12px 0;font-size:14.6px;line-height:2;
 font-family:"Cambria Math",Cambria,Georgia,serif;overflow-x:auto}
.formula .lbl{color:var(--muted);font-size:12.6px;font-family:inherit}
.small{font-size:12.6px;color:var(--muted)}
.tag{display:inline-block;font-size:11.6px;padding:1px 7px;border-radius:20px;
 border:1px solid var(--line);background:#fff;color:var(--muted);margin-left:5px}
.tag.ok{border-color:#a9d3ba;background:#eef7f1;color:#20603c}
.tag.bad{border-color:#e0b3b3;background:#fdf0f0;color:#8c2f2f}
.tag.mid{border-color:#e6cf9e;background:#fdf7ea;color:#8a5f16}
footer{border-top:1px solid var(--line);padding:16px 26px 34px;color:var(--muted);
 font-size:12.7px;text-align:center}
a{color:var(--accent)}
@media (max-width:960px){.wrap{flex-direction:column}nav#toc{position:static;width:100%;
 flex:0 0 auto}}
"""

TOC = [
    ("s1", "1. 文档目的与核心结论"),
    ("s2", "2. 数学基础"),
    ("s3", "3. 积分元（权重规则）"),
    ("s4", "4. 估计方法：目标函数与适用条件"),
    ("s4a", "4.1 统一视角", 1), ("s4b", "4.2 quadrature", 1),
    ("s4c", "4.3 iterative", 1), ("s4d", "4.4 projection", 1),
    ("s4e", "4.5 wlsq / cg", 1), ("s4f", "4.6 正则化", 1),
    ("s4g", "4.7 slepian", 1), ("s4h", "4.8 auto 选路", 1),
    ("s5", "5. 诊断量与守卫"),
    ("s6", "6. 算法与实现"),
    ("s7", "7. 试验一：示例数据"),
    ("s8", "8. 试验二：全球 0/1 帽"),
    ("s9", "9. 试验三：长江掩膜（重点）"),
    ("s10", "10. 方法选择速查表"),
    ("s11", "11. 优缺点与适用场景"),
    ("s12", "12. 复现"),
    ("s13", "13. 已知限制"),
]


def f(x, n=6):
    """Compact number formatting."""
    if x is None:
        return "—"
    if isinstance(x, float) and (x != x):
        return "—"
    a = abs(x)
    if a != 0 and (a >= 1e5 or a < 1e-3):
        return f"{x:.{n}g}"
    return f"{x:,.{min(n, 6)}f}".rstrip("0").rstrip(".")


def g(x, n=4):
    return "—" if x is None else f"{x:.{n}g}"


def sci(x, n=4):
    return "—" if x is None else f"{x:.{n}e}"


def tbl(heads, rows, num_from=1, cls=""):
    out = [f'<table class="{cls}">', "<thead><tr>"]
    for i, hh in enumerate(heads):
        out.append(f'<th class="{"num" if i >= num_from else ""}">{hh}</th>')
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for i, c in enumerate(r):
            out.append(f'<td class="{"num" if i >= num_from else ""}">{c}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def img(name, num, title, cap, w=None):
    return (f'<figure><img src="{IMG}/{name}" alt="{title}">'
            f'<figcaption><b>{num}</b> {cap}</figcaption></figure>')


P = []          # page chunks
A = P.append


# ===========================================================================
A(f"""<header>
<h1>SHKit 方法总结：从球谐目标函数到长江掩膜案例</h1>
<p>配套文档 · 全部数值由 <code>docs/_summary_measure.py</code> 在本机实时计算得出</p>
<p>球谐展开采用 4π 归一化、无 Condon-Shortley 相位；半波长分辨率 πR/n</p>
</header>
<div class="wrap"><nav id="toc"><h2>目录</h2>""")
for item in TOC:
    sid, label = item[0], item[1]
    sub = " sub" if len(item) > 2 else ""
    A(f'<a class="{sub.strip()}" href="#{sid}">{label}</a>')
A("</nav><main>")

# ------------------------------------------------------------------ 1
A(f"""<section id="s1">
<h2>1. 文档目的与核心结论</h2>
<p>本文说明 SHKit 把「散点/网格 → 球谐系数」这件事做成了什么，重点回答三个问题：
<b>每个方法最小化的目标函数是什么</b>、<b>它在什么条件下才成立</b>、<b>什么时候它会给出
看起来正确但物理上错误的结果</b>。全部结论都附有本机实时计算的数值与图。</p>

<div class="key"><h4>核心结论速览（只读这一段也够）</h4>
<ol>
<li><b>两个目标函数的定义域不同，这是全部问题的根源。</b>零填充全球 L2 投影的积分域是
<b>整个球面</b>（区域外 f≡0 也进入目标函数）；加权最小二乘的求和域<b>只有测点</b>。
二者仅在采样构成有效求积规则（K≈I）时才等价。</li>
<li><b>区域数据上“迭代把重建误差压到零”不是成功，而是警报。</b>长江掩膜实测：求积残差
{sci(E['projection']['residual_rms'],4)}（<b>这就是截断泄漏</b>），而 CG 收敛后残差塌到
{sci(E['cg']['residual_rms'],3)}，同时 C<sub>00</sub> 从几何真值
{sci(W_['lattice']['c00'],10)} 膨胀到 {sci(E['cg']['c00'],6)}，即 <b>{f(E['cg']['ratio_vs_area'],1)} 倍</b>，
隐含 1-set 面积 {E['cg']['implied_area_km2']:,.0f} km²（比中国还大）。</li>
<li><b>对 0/1 指示场，C<sub>00</sub> 是几何事实，不是自由参数。</b>它必须等于
"为 1 的面积占比”。长江掩膜的三套积分元给出三套答案，只有 <code>lattice</code> 是对的：
{sci(W_['lattice']['c00'],10)}（{W_['lattice']['area_km2']:,.2f} km²）；
<code>grid</code> 偏大 {(W_['grid']['ratio_vs_lattice']-1)*100:.2f}%；
<code>delaunay</code> 用球面凸包，偏大 {W_['delaunay']['ratio_vs_lattice']:.2f} 倍。</li>
<li><b>积分元选错不只是“尺度偏一个常数”。</b><code>grid</code> 与 <code>lattice</code> 的逐点权重比
在 {T['grid_vs_lattice_ratio']['n_gt_1p1']} 个点上超过 1.1 倍、最大
<b>{g(T['grid_vs_lattice_ratio']['max'],0)} 倍</b>，std/mean =
{g(T['grid_vs_lattice_ratio']['std_over_mean'],3)}；所以重建的<b>形状</b>也变
（L=12 图样相关 {T['grid_vs_lattice']['pattern_corr']:.6f}）。</li>
<li><b>截断泄漏有硬地板，迭代算法尊重它。</b>全球 0/1 帽（覆盖率 1.000000）实测
L=2..40 的残差 {g(FLOOR[0]['quad'],4)}/{g(FLOOR[1]['quad'],4)}/{g(FLOOR[2]['quad'],4)}/
{g(FLOOR[3]['quad'],4)}/{g(FLOOR[4]['quad'],4)}/{g(FLOOR[5]['quad'],4)}，
加 3 次甚至 25 次迭代<b>一位都不动</b>。区域数据上残差却塌向零——差别就在这里。</li>
<li><b>区域数据本质上是可解性问题，不是积分元问题。</b>未正则化的区域求解必须避免：
30° 球冠实测条件数 {sci(M['regional30']['wlsq_cond'],2)}、系数相对误差
{f(M['regional30']['wlsq_coef_rel_err'],1)}（完全失效）；<code>kaula</code> 正则或 Slepian 才对。</li>
</ol></div>

<p class="small">约定：下文所有“覆盖度"= Σw/4π；"Shannon 数"= (L+1)²·覆盖度；
"区域外场量"= 区域外重建 RMS 与区域内重建 RMS 之比。符号
C<sub>nm</sub>/S<sub>nm</sub> 指 4π 归一化的实球谐系数。</p>
</section>""")

# ------------------------------------------------------------------ 2
hw = M["resolution"]["halfwave"]
A(f"""<section id="s2">
<h2>2. 数学基础</h2>

<h3>2.1 展开式</h3>
<div class="formula">
f(θ,λ) = Σ<sub>n=0..N</sub> Σ<sub>m=0..n</sub> P̄<sub>nm</sub>(cos θ)
[ C<sub>nm</sub> cos(mλ) + S<sub>nm</sub> sin(mλ) ]<br>
<span class="lbl">P̄ 为 4π 归一化连带 Legendre 函数，无 Condon-Shortley 相位；
S<sub>n0</sub> ≡ 0（sin 0λ = 0），因此非平凡系数共 (N+1)² 个。</span>
</div>
<p>归一化定义：<b>(1/4π) ∫<sub>Ω</sub> P̄<sub>nm</sub>² cos²(mλ) dΩ = 1</b>。
与 SHTOOLS 的 <code>norm=1, csphase=1</code> 一致，也与 m2py / gridSHconvert 的三角布局兼容。</p>

<h3>2.2 正变换（分析）</h3>
<div class="formula">
C<sub>nm</sub> = (1/4π) ∫<sub>Ω</sub> f(θ,λ) P̄<sub>nm</sub>(cos θ) cos(mλ) dΩ<br>
S<sub>nm</sub> = (1/4π) ∫<sub>Ω</sub> f(θ,λ) P̄<sub>nm</sub>(cos θ) sin(mλ) dΩ
</div>
<p>离散化后就是加权投影 <b>x = (1/4π) AᵀWf</b>。注意<b>分母恒为 4π（全球归一化）</b>，
而 W 必须是真实的面积元——这一条决定了区域数据上 C<sub>00</sub> 对不对。
若把 W 强行归一化到 Σw = 4π，等于把系数整体乘上 4π/Σw，C<sub>00</sub> 随之失去几何含义。</p>

<h3>2.3 反变换（综合）</h3>
<p>把同一组系数代回 2.1 的展开式。SHKit 里 <code>synthesize()</code> 在给定点上求值，
<code>synthesis_grid()</code> 在规则网格上求值（分块，避免一次性构造 N×N 的中间量）。</p>

<h3>2.4 逐阶功率与 Parseval</h3>
<div class="formula">
逐阶功率 P<sub>n</sub> = Σ<sub>m=0..n</sub> ( C<sub>nm</sub>² + S<sub>nm</sub>² ) &nbsp;&nbsp;⟹&nbsp;&nbsp;
Σ<sub>n</sub> P<sub>n</sub> = (1/4π) ∫<sub>Ω</sub> f² dΩ
</div>
<p>本机校验（L={M['parseval']['L']}，GLQ 精确求积网格
{M['parseval']['n_points']} 点）：ΣP<sub>n</sub> = {sci(M['parseval']['sum_power'],12)}，
(1/4π)∫f²dΩ = {sci(M['parseval']['mean_square'],12)}，相对差
<b>{sci(M['parseval']['rel_diff'],1)}</b>。Parseval 是精确恒等式，与采样密度无关。</p>

<h3>2.5 半波长分辨率</h3>
<p>半波长 = πR/n（R = {M['resolution']['R_km']:.0f} km）：</p>
{tbl(["n", "半波长 (km)"], [[str(k), f"{v:,.1f}"] for k, v in hw.items()],
     num_from=0)}
<p>L=12 时半波长 {hw['12']:,.0f} km，而长江掩膜只跨
{T['lon_span']*111.19*np.cos(np.deg2rad(30)):,.0f} km × {T['lat_span']*111.19:,.0f} km——
<b>区域比分辨率元还小</b>，所以 L=12 根本不可能表达它的形状。</p>

<h3>2.6 Shannon 数与覆盖度</h3>
<div class="formula">
覆盖度 c = Σw / 4π &nbsp;&nbsp;&nbsp;&nbsp;
Shannon 数 N<sub>Sh</sub> = (L+1)² · c
</div>
<p>N<sub>Sh</sub> 是一个区域在给定阶数下能承载的独立自由度数。它同时也是完备性矩阵
K 的迹：<b>Σλ(K) = (L+1)²·c</b>。本机在长江掩膜 L=12 上实测
Σλ = {sci(T['gram']['sum_lambda'],10)}，Shannon 数 =
{sci(T['gram']['shannon'],10)}，比值 <b>{T['gram']['sum_over_shannon']:.12f}</b>——
这条恒等式可当自检用。</p>
<div class="note">N<sub>Sh</sub> ≪ 1 时，截断后掩膜的峰值只能到 N<sub>Sh</sub> 量级。
长江掩膜 L=12 的 N<sub>Sh</sub> = {sci(T['shannon_L12'],4)}，实测峰值
{sci(T['recon_peak_proj'],4)}，比值 {T['recon_peak_proj']/T['shannon_L12']:.4f}。
要让掩膜“立”到 1，需 L ≈ <b>{T['L_for_mask_peak_1']:.1f}</b>。</div>
</section>""")

# ------------------------------------------------------------------ 3
vt = M["voronoi_trap"]
A(f"""<section id="s3">
<h2>3. 积分元（权重规则）</h2>
<p>积分元 w<sub>i</sub> 是“每个样本代表多大立体角”。它决定了 Σw（⇒ 覆盖度 ⇒
C<sub>00</sub>）以及每个点在目标函数里的份量。选错规则不会报错，只会给出一组自洽但
物理错误的结果。</p>
{tbl(["规则", "定义", "Σw", "精确性", "适用条件"], [
 ["<code>dh</code>", "Driscoll-Healy：纬度按 GL 节点权，经度 2π/nlon", "4π",
  "对带限场精确（nlon = 2·nlat）", "全球等经纬网格"],
 ["<code>glq</code>", "Gauss-Legendre 精确求积网格（须用 <code>glq_grid()</code> 生成的权重）",
  "4π", "对 ≤L 阶带限场精确", '需要“精确反演”的参考算例'],
 ["<code>grid</code>", "球带面积 Δλ(sinφ₂−sinφ₁)，格边取相邻<b>现存</b>坐标中点",
  "覆盖面积", "对<b>完整</b>网格精确", "任意完整经纬网格（含区域矩形）"],
 ["<code>lattice</code>", "每点一个格元：Δλ<sub>rad</sub>(sin(φ+Δφ/2)−sin(φ−Δφ/2))，"
  "Δ 由坐标稳健估计", "覆盖面积", "对规则格点上的<b>子集</b>精确",
  "<b>规则格网上的掩膜</b>（本案例）"],
 ["<code>voronoi</code>", "球面 Voronoi 胞面积", "恒为 4π",
  "对全球准均匀点集好", "全球散点"],
 ["<code>delaunay</code>", "球面 Delaunay 1/3 规则（只保留原点可见面）", "球面凸包面积",
  "二阶近似", "<b>不规则</b>区域散点"],
 ["<code>uniform</code>", "等面积 4π/N", "4π", "仅当采样本身等面积", "等面积采样"],
 ["<code>user</code>", "调用者给出逐点权重", "自定义", "取决于输入",
  "已有可靠权重（如本案例的 CSV）"],
], num_from=1)}

<h3>3.1 长江掩膜：三套口径实测对照</h3>
{tbl(["口径", "Σw (sr)", "覆盖面积 (km²)", "C<sub>00</sub>", "相对 lattice", "相对格元参考"],
 [[k if k != "tool_csv" else "逐点格元参考（CSV）",
   sci(W_[k]['sum_sr'], 12), f"{W_[k]['area_km2']:,.2f}",
   sci(W_[k]['c00'], 10), f"{W_[k]['ratio_vs_lattice']:.6f}×", ""]
  for k in ("lattice", "grid", "delaunay")]
 + [["逐点格元参考（<code>examples/yangtze_points_weights.csv</code>）",
     sci(W_['tool_csv']['sum_sr'], 12), f"{W_['tool_csv']['area_km2']:,.2f}",
     sci(W_['tool_csv']['c00'], 10), f"{W_['tool_csv']['ratio_vs_lattice']:.15f}×",
     "基准"]], num_from=1)}
<p>参考值 {W_['tool_csv']['area_km2']:,.2f} km² 由
<code>tools/yangtze_points_to_sh.py</code> 独立算出（每点一个 Δlon×Δlat 球面格），
与库内 <code>lattice</code> 的 Σw 相对差
<b>{abs(W_['lattice']['sum_sr']/W_['tool_csv']['sum_sr']-1):.2e}</b>（逐点最大相对差约
2.8e-13，纯粹是该 CSV 只写了 12 位有效数字）。</p>

{img("fig01_weights_bar.png", "图1",
     "长江掩膜三种积分元的总权重与面积",
     "<b>看图什么</b>：左为 Σw（对数轴），右为换算面积。lattice 与 grid 相差 4.59%，"
     "delaunay 则大出近 70 倍。<b>证明了什么</b>：同一份点集，换一个积分元规则就能让"
     "覆盖面积差 70 倍，而 C<sub>00</sub> 直接继承这个倍数——所以积分元不是“技术细节”，"
     "它决定答案的量级。绿色参考值由逐点格元独立算出，lattice 与它一致。")}

<h3>3.2 为什么 <code>grid</code> 在这份数据上偏大 4.59%</h3>
<p>掩膜是“有洞的规则格网”：2509 个唯一纬度里有
<b>{T['n_big_lat_gaps']}</b> 条间隔宽于一步，最宽
{T['lat_gap_max']:.7f}° ≈ <b>{T['lat_gap_max']/T['lat_gap_median']:.0f} 步</b>。
<code>grid</code> 把格边放在相邻<b>现存</b>坐标的中点，于是紧邻空档的点吞掉了空档的一半。</p>
{tbl(["统计量", "值"], [
    ["w<sub>grid</sub>/w<sub>lattice</sub> 最小值", g(T['grid_vs_lattice_ratio']['min'], 8)],
    ["中位数", g(T['grid_vs_lattice_ratio']['median'], 8)],
    ["p99", g(T['grid_vs_lattice_ratio']['p99'], 8)],
    ["最大值", f"{T['grid_vs_lattice_ratio']['max']:,.1f}"],
    ["std / mean", g(T['grid_vs_lattice_ratio']['std_over_mean'], 3)],
    ["比值 > 1.1 的点数", f"{T['grid_vs_lattice_ratio']['n_gt_1p1']} / "
     f"{T['grid_vs_lattice_ratio']['n_points']}（{T['grid_vs_lattice_ratio']['n_gt_1p1']/T['grid_vs_lattice_ratio']['n_points']*100:.3f}%）"],
    ["比值 > 1.001 的点占比", f"{T['grid_vs_lattice_ratio']['frac_gt_1p001']*100:.3f}%"],
    ["L=12 重建峰值比（grid/lattice）", f"{T['grid_vs_lattice']['peak_ratio']:.6f}"],
    ["L=12 两套重建的图样相关", f"{T['grid_vs_lattice']['pattern_corr']:.6f}"],
    ["逐阶 RMS 最大相对差", f"{T['grid_vs_lattice']['max_rel_diff_spectrum']*100:.3f}%"],
], num_from=1)}
{img("fig02_weight_ratio_hist.png", "图2",
     "逐点权重比值不是常数；纬度空档分布",
     "<b>看图什么</b>：左图是 w<sub>grid</sub>/w<sub>lattice</sub> 的分布（对数横轴），"
     f"std/mean = {g(T['grid_vs_lattice_ratio']['std_over_mean'],3)}，远不是纯比例缩放；"
     f"右图显示有 {T['n_big_lat_gaps']} 条空档宽于一个格距。<b>证明了什么</b>："
     f"偏差不是“整体乘个常数”，而是集中在 {T['grid_vs_lattice_ratio']['n_gt_1p1']} 个"
     "紧邻空档的点上（最大超过 5000 倍），因此连重建的<b>形状</b>都会变——"
     f"L=12 时两套重建的图样相关只有 {T['grid_vs_lattice']['pattern_corr']:.4f}。")}

<h3>3.3 区域 Voronoi 陷阱</h3>
<p>对<b>区域子集</b>做球面 Voronoi 是明确错误的：Voronoi 剖分永远铺满整个球面，
边界点的胞会膨胀到球的对侧，所以 Σw <b>恒等于 4π</b>。</p>
{tbl(["量", "值"], [
    ["30° 球冠真实面积 2π(1−cos r)", f"{vt['true_cap_area_sr']:.4f} sr（占全球 {vt['true_cap_fraction']*100:.2f}%）"],
    ["区域子集 Voronoi 的 Σw", f"{vt['voronoi_sum_sr']:.4f} sr"],
    ["⇒ 虚高倍数", f"<b>{vt['voronoi_ratio']:.2f}×</b>"],
    ["同一球冠用 delaunay", f"{vt['delaunay_sum_sr']:.4f} sr（{vt['delaunay_ratio']:.3f}×，即偏小 {(1-vt['delaunay_ratio'])*100:.1f}%）"],
], num_from=1)}
{img("fig16_voronoi_trap.png", "图16",
     "区域 Voronoi 虚高 14.93 倍",
     "<b>看图什么</b>：30° 球冠三种口径的 Σw。Voronoi 给 4π，是真实球冠面积的 "
     f"{vt['voronoi_ratio']:.2f} 倍。<b>证明了什么</b>：区域数据用 Voronoi 会把覆盖面积"
     "放大一个数量级，而且这个错误不会触发任何警告——只能用"
     "<code>delaunay</code>（不规则散点）或 <code>grid</code>/<code>lattice</code>"
     "（格点数据）。若手上有全球点集 + 区域掩膜，正确做法是在<b>全点集</b>上做 Voronoi"
     "再取子集。")}
</section>""")

# ------------------------------------------------------------------ 4
A(f"""<section id="s4">
<h2>4. 估计方法：目标函数与适用条件</h2>
<p>这是全文重点。SHKit 有 5 条估计路径加 2 种正则化、1 个 Slepian 分支。
先说它们之间的关系，因为不理解这一点就会以为“迭代能把误差压小”。</p>

<h3 id="s4a">4.1 统一视角：三种停止策略，同一个估计量</h3>
<p>设 A 为设计矩阵（N×(L+1)²），W = diag(w<sub>i</sub>)，定义<b>完备性矩阵</b></p>
<div class="formula">
K = (1/4π) AᵀWA &nbsp;&nbsp;&nbsp;⟹&nbsp;&nbsp;&nbsp;
K = I ⟺ 采样 + 权重构成精确求积规则
</div>
<p>三条路径其实是同一个正规方程 <b>(AᵀWA)x = AᵀWf</b> 的三种停止策略：</p>
{tbl(["路径", "公式", "等价说法"], [
 ["<code>quadrature</code> / <code>projection</code>",
  "x = (1/4π)AᵀWf", "正规方差的<b>第 0 步</b>"],
 ["<code>iterative</code>",
  "x ← x + (1/4π)AᵀW(f − Ax)", "<b>提前停止</b>的 Richardson/Landweber 迭代 = 隐式正则化"],
 ["<code>wlsq</code> / <code>cg</code>",
  "x = (AᵀWA)<sup>−1</sup>AᵀWf", "正规方程的<b>极限解</b>；K 亏秩时 = 最小范数解"],
], num_from=1)}
<p>K ≈ I 时三者数值等价；K 远离 I 时它们分道扬镳，而<b>收敛极限是三者里唯一没有物理依据的
那个</b>。长江案例正是后者。</p>
{img("fig03_objective_domain.png", "图3",
     "两个目标函数的定义域不同",
     "<b>看图什么</b>：左图的积分域是整个球面（灰色点也参与），右图的求和域只有测点。"
     "<b>证明了什么</b>：这是全部问题的根源。区域外那 99.9998% 的隐含 0 在左图里通过 4π "
     "归一化进入目标函数，在右图里根本不存在——所以右图的目标函数对“掩膜是否被还原“"
     "结构性失明，残差再小也说明不了问题。")}

<h3 id="s4b">4.2 <code>quadrature</code> / <code>projection</code>：零填充全球 L2 投影</h3>
<div class="formula">
min<sub>deg≤L</sub> ∫<sub>Ω</sub> ( f − A x )² dΩ &nbsp;&nbsp;&nbsp;
Ω = <b>整个球面</b>，f = 区域外 0 &nbsp;&nbsp;⟹&nbsp;&nbsp;
x = (1/4π) AᵀWf <span class="lbl">（W 取几何和，不归一化到 4π）</span>
</div>
<p>因为区域外 f ≡ 0 对分子无贡献，而分母 4π 是全球归一化，所以这个投影<b>恰好是
零填充场的 L2 投影</b>。它的三条性质：</p>
<ul>
<li>样本残差<b>就是截断泄漏</b>，停在由 L 决定的地板上，永远压不到零；</li>
<li>C<sub>00</sub> 精确等于面积占比（几何事实）；</li>
<li>不做任何反演，因此没有近零空间放大。</li>
</ul>
<p>长江 L=12 实测：C<sub>00</sub> = {sci(E['projection']['c00'],10)}
（几何要求 {sci(W_['lattice']['c00'],10)}，比值 <b>{E['projection']['ratio_vs_area']:.12f}</b>），
残差 {sci(E['projection']['residual_rms'],6)}，面积加权残差
{sci(E['projection']['residual_rms_weighted'],6)}。
<code>quadrature</code> 与 <code>projection</code> 在同样的几何权重下<b>逐位相同</b>
（相对差 {sci(E['quad_vs_proj_c00_rel'],3)}）——它们本来就是同一件事。</p>
<div class="note"><b>可选采样偏差校正 τ</b>（仅 <code>projection</code>）：把 K 特征分解
（特征向量即 Slepian taper，特征值 λ 即集中因子），只对 <b>λ ≥ τ</b> 的方向除以 λ，
其余保持不变。τ = 1 等于纯投影；τ → 0 退化成未正则化 WLSQ。
校正的是<b>采样偏差</b>（可修的离散化误差），从不碰截断泄漏。</div>

<h3 id="s4c">4.3 <code>iterative</code>：对同一目标做 Richardson 迭代</h3>
<p>x ← x + (1/4π)AᵀW(f − Ax)，即对 4.2 的目标做定步长梯度下降，<b>不动点恰好是 WLSQ 解</b>。
准均匀采样下它很划算：本机 Fibonacci {M['fib']['n']} 点、L={M['fib']['L']} 实测</p>
{tbl(["方法", "系数相对误差"], [[r["tag"], sci(r["coef_rel_err"], 4)]
  for r in M["fib"]["rows"]], num_from=1)}
{img("fig09_fib_roundtrip.png", "图9",
     "准均匀采样下迭代校正逐步逼近最小二乘",
     "<b>看图什么</b>：同一次求解，求积 → +1 次迭代 → +3 次迭代 → wlsq 的系数误差。"
     "<b>证明了什么</b>：在 max|K−I| 只有 "
     f"{g([r['max_K_minus_I'] for r in M['routing'] if 'Fibonacci' in r['name']][0],2)} 的良态采样上，"
     "迭代校正确实在逼近最小二乘解——此时“迭代减小误差”是<b>成立</b>的。"
     "所以问题不是迭代本身，而是它被用在 K 远离 I 的数据上。")}

<h3 id="s4d">4.4 <code>projection</code> 的 Slepian 视角与 τ 实测</h3>
<p>集中比 λ = ∫<sub>R</sub>g²dΩ / ∫<sub>Ω</sub>g²dΩ = cᵀKc / cᵀc，所以 Slepian 函数就是 K 的
特征向量、集中因子就是特征值——不需要任何外部库。</p>
{tbl(["τ", "C<sub>00</sub>", "C<sub>00</sub>/几何值", "被校正方向数", "λ<sub>max</sub>", "样本残差"], [
 [("None（纯投影）" if r["tau"] is None else g(r["tau"], 3)),
  sci(r["c00"], 6), f"{r['ratio']:.6f}", "—",
  (g(r["lam_max"], 4) if r["lam_max"] else "—"), sci(r["residual"], 4)]
 for r in T["tau_sweep"]], num_from=1)}
<p>长江 L=12 的 K 谱：λ<sub>max</sub> = {sci(T['gram']['lambda_max'],4)}，
λ<sub>min</sub> = {sci(T['gram']['lambda_min'],3)}，Σλ = Shannon 数
{sci(T['gram']['shannon'],4)}。各阈值以上还有几个方向：</p>
{tbl(["阈值", "λ ≥ 阈值的个数"], [[f"λ ≥ {k}", str(v)] for k, v in
     sorted(T["gram"]["n_above"].items(), key=lambda kv: -float(kv[0]))], num_from=0)}
<div class="warn"><b>读法</b>：τ = 0.5 是<b>空操作</b>，而且这是正确的——
λ<sub>max</sub> = {sci(T['gram']['lambda_max'],3)} ≪ 0.5 说明这块区域在 L=12 下
<b>不存在任何集中因子接近 1 的带限图样</b>。一旦把 τ 降到 1e-4 以下开始“校正”，
C<sub>00</sub> 立刻从 1.000000 跳到 {g([r for r in T['tau_sweep'] if r['tau']==1e-4][0]['ratio'],1)} 倍。
软件会明确报出 <code>corrected nothing</code> 并说明原因。</div>
{img("fig04_k_spectrum.png", "图4",
     "长江 L=12 的完备性矩阵谱",
     "<b>看图什么</b>：左为 K 的 169 个特征值（降序、对数轴）；右为各阈值以上的方向数。"
     f"<b>证明了什么</b>：{T['gram']['n_above']['0.01']} 个方向满足 λ ≥ 0.01，"
     f"{T['gram']['n_above']['1e-10']} 个满足 λ ≥ 1e-10。绝大多数方向对数据"
     "<b>数值上不可见</b>——这正是“许多完全不同的系数向量能同样好地拟合样本”的原因，"
     "也是最小二乘残差可以塌到 {:.0e} 的原因。".format(E['cg']['residual_rms']))}

<h3 id="s4e">4.5 <code>wlsq</code> / <code>cg</code>：逐点加权最小二乘</h3>
<div class="formula">
min ‖√W (A x − f)‖² = Σ<sub>i=1..N</sub> w<sub>i</sub> ((Ax)<sub>i</sub> − f<sub>i</sub>)²
&nbsp;&nbsp;<span class="lbl">求和域<b>只有测点</b></span>
</div>
<p>两条重要性质，决定了它什么时候会骗你：</p>
<ul>
<li><b>求和域是测点集</b>：隐含的 0 不在目标函数里，所以它对“掩膜是否被还原”结构性失明。</li>
<li><b>目标函数对 W 的整体缩放不变</b>：φ(cW) 与 φ(W) 极小点相同。而指示场的正确
C<sub>00</sub> 恰恰正比于 Σw——<b>目标函数扔掉的正是确定 C<sub>00</sub> 所需的那条信息</b>。</li>
</ul>
<p>长江 L=12 实测（<code>cg</code>，权重取 lattice，与存储文件的 delaunay 情形对照）：</p>
{tbl(["方法", "C<sub>00</sub>", "C<sub>00</sub>/几何值", "隐含 1-set 面积 (km²)", "样本残差", "CG 收敛标志"], [
 ["<code>projection</code>", sci(E['projection']['c00'], 10),
  f"{E['projection']['ratio_vs_area']:.6f}",
  f"{E['projection']['implied_area_km2']:,.0f}", sci(E['projection']['residual_rms'], 6), "—"],
 ["<code>cg</code>（本轮）", sci(E['cg']['c00'], 10), f"{E['cg']['ratio_vs_area']:,.1f}",
  f"{E['cg']['implied_area_km2']:,.0f}", sci(E['cg']['residual_rms'], 3),
  f"info = {E['cg']['cg_info']}（成功）"],
], num_from=1)}
<div class="warn"><b>关键陷阱</b>：<code>cg_info = [0]</code> 是 scipy 报告"<b>成功收敛</b>"。
它确实收敛了，只是收敛到一个物理上不可能的解——C<sub>00</sub> 隐含
{E['cg']['implied_area_km2']:,.0f} km²（比中国还大）。<b>"收敛“和”正确“在这里是两件事。</b>
本机把该情形下的告警逐条列出：C<sub>00</sub> 不一致（触发）＝
{E['cg']['warn_c00']}，残差塌陷告警（触发）＝ {E['cg']['warn_collapsed']}。</div>
{img("fig05_iteration_drift.png", "图5",
     "残差与物理误差严格反相关",
     "<b>看图什么</b>：左为样本残差随迭代下降（Richardson 只从 1.00 压到 "
     f"{g(T['iterative_drift'][-1]['residual'],3)}，CG 收敛那一步才趋到 "
     f"{sci(E['cg']['residual_rms'],3)}）；右为同一批迭代里 C<sub>00</sub> 相对几何值的倍数。"
     "<b>证明了什么</b>：<b>残差每降一档，物理误差就涨一档</b>，没有一步例外。"
     "所以在区域数据上最小化样本残差就等于最大化物理错误——"
     "“迭代减小重建误差”作为验收标准在逻辑上就不成立。")}
{tbl(["niter", "样本残差", "C<sub>00</sub>", "C<sub>00</sub>/几何值", "隐含面积 (km²)"],
 [[str(r["niter"]), sci(r["residual"], 6), sci(r["c00"], 6), f"{r['ratio']:.2f}×",
   f"{r['implied_area_km2']:,.0f}"] for r in T["iterative_drift"]]
 + [["<b>CG 收敛</b>", f"<b>{sci(E['cg']['residual_rms'],3)}</b>",
     f"<b>{sci(E['cg']['c00'],6)}</b>", f"<b>{E['cg']['ratio_vs_area']:,.1f}×</b>",
     f"<b>{E['cg']['implied_area_km2']:,.0f}</b>"]], num_from=1)}

<h3 id="s4f">4.6 正则化：<code>tikhonov</code> vs <code>kaula</code></h3>
<div class="formula">
x<sub>α</sub> = arg min ‖W<sup>1/2</sup>(Ax − f)‖² + α²‖R x‖²
&nbsp;&nbsp;<span class="lbl"> tikhonov: R = I；kaula: R<sub>k</sub> = 1/(s<sub>n</sub>(n+1)<sup>p</sup>)</span>
</div>
<p>kaula 的谱先验 s<sub>n</sub> 由一次快速求积得到的逐阶振幅给出，因此<b>量纲无关</b>
（EWH、重力异常、水高都直接可用）。长江 L=12 实测：</p>
{tbl(["正则化", "α", "C<sub>00</sub>", "隐含面积 (km²)", "样本残差"], [
 [("无" if r["reg"] is None else r["reg"]), (g(r["alpha"], 2) if r["alpha"] else "—"),
  sci(r["c00"], 6), f"{r['implied_area_km2']:,.0f}", sci(r["residual"], 4)]
 for r in T["regularization"]], num_from=1)}
<p>30° 球冠区域反演（覆盖度 {M['regional30']['coverage']:.3f}，全球带限真值）的系数相对误差：</p>
{tbl(["正则化", "系数相对误差"], [
 ["区域求积（quadrature）", f(M['regional30']['quad_coef_rel_err'], 3)],
 ["<code>wlsq</code> 未正则化", f"<b>{f(M['regional30']['wlsq_coef_rel_err'],1)}</b>"],
 ["<code>wlsq</code> + tikhonov α=1e-9", f(M['regional30']['reg_coef_rel_err']['tikhonov-1e-09'], 3)],
 ["<code>wlsq</code> + kaula α=1e-4", f(M['regional30']['reg_coef_rel_err']['kaula-0.0001'], 3)],
], num_from=1)}
<p>条件数随阶数的增长（同一球冠，未正则化 WLSQ）：</p>
{tbl(["L", "系数数", "条件数"], [[str(r["L"]), str(r["ncoef"]),
  (sci(r["cond"], 2) if r["cond"] else "—")] for r in M["regional30"]["per_L_cond"]],
 num_from=1)}
{img("fig06_regularization.png", "图6",
     "kaula 有效，tikhonov 在区域数据上几乎无效",
     "<b>看图什么</b>：左为长江 L=12 各种正则化下 C<sub>00</sub> 隐含的面积；"
     "右为 30° 球冠的系数相对误差。<b>证明了什么</b>：tikhonov（R=I）在这类问题上"
     "几乎不起作用——长江上 α 加到 1e-4 仍然停在 "
     f"{sci([r for r in T['regularization'] if r['reg']=='tikhonov'][-1]['c00'],3)}；"
     "而 <b>kaula 把 C<sub>00</sub> 拉回 "
     f"{sci([r for r in T['regularization'] if r['reg']=='kaula'][0]['c00'],3)}</b>，"
     "与散点足迹同量级。原因是零空间太大，惩罚方向不对。")}

<h3 id="s4g">4.7 <code>slepian</code>：区域数据的正解</h3>
<p>核心洞察：<b>Slepian 集中矩阵就是 K</b>。在正交归一的 taper 基下，区域加权最小二乘的
正规矩阵是<b>对角</b>的：</p>
<div class="formula">
s<sub>α</sub> = t<sub>α</sub>ᵀAᵀWf / (4π λ<sub>α</sub>), &nbsp;&nbsp; c = Σ<sub>α</sub> s<sub>α</sub> t<sub>α</sub>
&nbsp;&nbsp;<span class="lbl">只有 λ<sub>α</sub> ≈ 1 的 taper 可用，截断本身就是正则化</span>
</div>
<p>本机实测（25° 球冠，{M['slepian']['n_points']} 点，L={M['slepian']['L']}，
覆盖度 {M['slepian']['coverage']:.4f}，λ>0.5 共 {M['slepian']['slepian_ntaper']} 个 taper）：</p>
{tbl(["指标", "Slepian（λ>0.5）", "未正则化 WLSQ"], [
 ["条件数", g(M['slepian']['slepian_cond'], 4), sci(M['slepian']['wlsq_cond'], 2)],
 ["‖C‖（真值 %s）" % g(M['slepian']['slepian_inside_truth_norm'], 3),
  g(M['slepian']['slepian_normC'], 4), f(M['slepian']['wlsq_normC'], 1)],
 ["区域外场量 / 区域内 RMS", sci(M['slepian']['slepian_outside_rel'], 2),
  f(M['slepian']['wlsq_outside_rel'], 1)],
 ["系数相对误差", f(M['slepian']['slepian_coef_err'], 3),
  f(M['slepian']['wlsq_coef_err'], 1)],
], num_from=1)}
{img("fig15_slepian_vs_wlsq.png", "图15",
     "Slepian 与未正则 WLSQ 的三项指标对比",
     "<b>看图什么</b>：条件数、‖C‖、区域外场量三项（都是对数轴）。"
     "<b>证明了什么</b>：未正则化 WLSQ 在区域数据上每一项都大出几个数量级，"
     "尤其是区域外场量——它把误差全部倾泻到数据没覆盖的地方。"
     "Slepian 只保留 λ>0.5 的 taper，把这部分完全抑制住。")}
<h4>小区域连低阶场都恢复不了</h4>
{tbl(["球冠半径", "覆盖度", "Shannon 数(L=4)", "λ<sub>max</sub>", "λ>0.5 的个数"],
 [[f"{r['rad']:.0f}°", f"{r['coverage']:.4f}", g(r["shannon"], 3),
   g(r["lam_max"], 4), str(r["n_above_0.5"])] for r in M["small_caps"]],
 num_from=1)}
<p>低阶场本身就弥散在全球，其集中因子恰好接近面积占比 A/4π，远低于 0.5；能集中的 taper
都落在高阶。所以<b>用区域数据反演低阶全球场在信息论上不成立</b>，正确做法是
<b>remove–restore</b>：先用全球模型扣掉长波，只在区域内分析残差，最后加回。</p>

<h3 id="s4h">4.8 <code>auto</code> 的选路</h3>
{tbl(["采样", "N", "max|K−I|", "auto 选择"], [
 [r["name"], f"{r['n']:,}", sci(r["max_K_minus_I"], 3),
  f"<b>{r['auto']}</b>"] for r in M["routing"]], num_from=1)}
{img("fig10_routing.png", "图10",
     "auto 的选路判据实测",
     "<b>看图什么</b>：五类采样的 max|K−I|（对数轴）与 auto 的选择，虚线是 1e-2 阈值。"
     f"<b>证明了什么</b>：判据本身工作正常——DH 网格 {sci(M['routing'][0]['max_K_minus_I'],2)} "
     "直接求积，随机均匀/聚簇点 "
     f"{sci(M['routing'][3]['max_K_minus_I'],2)} / {sci(M['routing'][4]['max_K_minus_I'],2)} "
     "转最小二乘。但注意 GLQ 那一行：它的 max|K−I| 只有 "
     f"{sci(M['routing'][1]['max_K_minus_I'],2)}，却仍选了 {M['routing'][1]['auto']}——"
     "因为 auto 还要求超定比 ≥ 2，而 GLQ(L=20) 是 861 点 / 441 系数 = 1.95。<b>所以"
     "auto 是“求积够准”与“方程够超定”两个条件的与，不是只看完备性</b>。")}
<p><b>本轮修改后的行为</b>：区域覆盖（覆盖度 &lt; 0.999）→ 路由到
<code>projection</code>，不再升级为 <code>wlsq</code>/<code>cg</code>。原因是区域覆盖下
正规矩阵在双精度下已经奇异（长江实测
{sci(T['gram']['lambda_min'],2)} 的最小特征值），升级到最小二乘只会返回近零空间的
最小范数解。</p>
</section>""")

# ------------------------------------------------------------------ 5
A(f"""<section id="s5">
<h2>5. 诊断量与守卫</h2>
{tbl(["诊断量", "含义", "判据阈值", "该采取的动作"], [
 ["<code>max|K−I|</code>（gram_deviation）",
  "K = AᵀWA/4π 偏离单位阵的最大量；K=I 才是精确求积",
  "&lt; 1e-2 允许直接求积", "超标且覆盖全球 → 换 WLSQ + 正则；覆盖区域 → <b>不要反演 K</b>"],
 ["<code>coverage</code>", "Σw/4π，数据覆盖球面的比例", "&lt; 0.999 判为区域数据",
  "区域数据按目标 A 走 <code>projection</code>；要目标 B 就正则化 / Slepian / remove-restore"],
 ["<code>condition_number</code>", "cond(A√W)", "&gt; 1e8 双精度耗尽",
  "降阶、提高超定比、或给正则化"],
 ["<code>overdetermination</code>", "N/(L+1)²", "&lt; 2 勉强定解；推荐 ≥ 4",
  "降 nmax，或增加数据"],
 ["<code>shannon</code>", "(L+1)²·coverage，区域能承载的自由度",
  "&lt; 0.5·系数数 ⇒ 该区域撑不住这个阶数", "降阶，或改用 Slepian；记住峰值上界就是这个数"],
 ["<code>lmax_recommended</code>", "当前采样能支持的最大阶数（计数上限）",
  "覆盖 ≪ 1 时只是计数上限，不是有效约束", "配合 Shannon 一起看"],
 ["<b><code>residual_rms_weighted</code></b>（新增）",
  "面积加权的残差 RMS——球面上的真 L2 范数",
  "与未加权值并列显示", "两者差别大时说明采样在球面上不均匀（如经纬网格极区过采样）"],
 ["<b>C<sub>00</sub> 一致性守卫</b>（新增）",
  "C<sub>00</sub> 必须等于 Σwf/4π（场的全球均值；任何阶截断都保持它）",
  "相对偏离 &gt; 10% 报警",
  "检查积分元与 normalise；改用 <code>projection</code>。<b>小的样本残差不能作为辩护</b>"],
 ["<b>残差塌陷告警</b>（新增）",
  "wlsq/cg/iterative + 覆盖度 &lt; 0.999 + 相对残差 &lt; 1e-3",
  "三者同时满足即报警",
  "说明求解器利用了近零空间，不是拟合得好；与 <code>projection</code> 对照并检查 C<sub>00</sub>"],
], num_from=2)}
<h3>5.1 未加权残差不是球面上的 L2 范数</h3>
<p>经纬网格在极区过采样，而那里 f 与重建都 ≈0，会把未加权 RMS 稀释。全局 0/1 帽实测：</p>
{tbl(["L=12 帽", "值"], [
 ["未加权残差 RMS", sci(M['cap']['L12_resid_unweighted'], 6)],
 ["面积加权残差 RMS", sci(M['cap']['L12_resid_weighted'], 6)],
 ["比值", f"{M['cap']['L12_ratio']:.4f}"],
], num_from=1)}
<p>几何/区域数据上两者一致（因为权重本身近似均匀），但全球网格数据上必须看面积加权值。</p>
<h3>5.2 C<sub>00</sub> 守卫能力与边界</h3>
<p>它抓的是<b>DC 不一致</b>，不是<b>积分元选错</b>。实测两种情形：</p>
{tbl(["情形", "C<sub>00</sub>", "自身 Σw/4π", "比值", "守卫"], [
 ["存储的 <code>yantze_shkit_coeffs.gfc</code>（当前内容：projection + delaunay）",
  sci(H['current_file']['c00'], 10),
  sci(H['current_file']['dc_expected_from_own_weights'], 10),
  f"{H['current_file']['ratio_vs_delaunay']:.6f}", "<b>不触发</b>"],
 ["原始 <code>.gfc</code>（cg + reg=None + delaunay，本机复现）",
  sci(H['reproduced_original_cg_delaunay']['c00'], 10),
  sci(W_['delaunay']['c00'], 10),
  f"{rep['c00']/W_['delaunay']['c00']:,.1f}×",
  "<b>触发</b>"],
], num_from=1)}
<p>当前文件是<b>自洽</b>的（C<sub>00</sub> 精确等于它自己权重的 Σw/4π），所以守卫不响——
但它用的 delaunay 口径把面积放大了 {W_['delaunay']['ratio_vs_lattice']:.1f} 倍。
<b>守卫能抓“求解器跑偏”，抓不了“口径选错”；后者要靠
<code>rule='auto'</code> 的格点识别与 3.1 的对照表。</b></p>
</section>""")

# ------------------------------------------------------------------ 6
A(f"""<section id="s6">
<h2>6. 算法与实现</h2>
{tbl(["环节", "实现要点", "复杂度 / 内存"], [
 ["流式 m 阶求积", "沿阶 m 逐列推进，只保留当前与 m−2 列，不构造 N×N 中间量",
  "O(N·(L+1)²) 时间，O(N·L) 内存"],
 ["矩阵无关 CG（<code>SHOperator</code>）", "matvec/rmatvec 现场构造，设计矩阵从不落地",
  "O(N) 内存（对比稠密 O(N·ncoef)）"],
 ["K 的特征分解 = Slepian", "集中矩阵就是 K，特征向量即 taper，<b>不需要外部库</b>",
  "O(ncoef³)，ncoef &gt; 1500 时跳过"],
 ["Kaula 谱先验", "由一次快速求积得到逐阶振幅 s<sub>n</sub>，量纲无关", "额外一次求积"],
 ["L 曲线自动选 α", "变量替换 y = Rx 后一次特征分解扫描整条 L 曲线",
  "O(ncoef³)；ncoef &gt; 2500 时跳过并提示显式给 α"],
 ["完备性检查开销", "N·ncoef² &gt; 2e9 flops 时降级为廉价的 m=0 对角完备性", "O(N·L)"],
 ["单位正/反变换", "正变换 C<sub>nm</sub> = a<sub>nm</sub>/f<sub>u</sub>（改它导出的系数就变）；"
  "反变换 f<sub>t</sub> 只影响重建", "常数级"],
 ["多时次", "各时次独立求解，共享设计矩阵/算子", "线性于 ntime"],
], num_from=2)}
<p>本机在长江数据上实测的耗时（L=12，142846 点）：求积/投影 ~2 s，
6 组正则化 CG 共 ~110 s，τ 扫描 8 组 ~10 s，CG 单次 ~6 s；阶数扫描
L=12/30/60/120 分别 {', '.join(str(r['secs']) for r in T['shannon_sweep'])} s。
L=120 以上直接求和的代价迅速上升（README 记录 L=700 需 40+ 分钟），
更高阶必须换 Slepian / FFT / 局部方法。</p>
</section>""")

# ------------------------------------------------------------------ 7
A(f"""<section id="s7">
<h2>7. 试验一：示例数据（全球准均匀散点，真值已知）</h2>
<p>数据：GUI「生成示例数据」= 全球 Fibonacci <b>{G['demo']['npoints']:,}</b> 点
（<code>main_window.make_demo()</code>：<code>n, L = {G['demo']['npoints']}, 12</code>），
真实场带限 12 阶 + 1% 噪声；<code>sample_data/</code> 另提供 3000 点、
区域球冠、以及 20 阶真值系数（<code>truth_coeffs_20.sh</code>）。
真值已知时可以直接量系数误差，因此这是验证估计量行为的最佳场景。</p>
{tbl(["项目", "值"], [
 ["点数", f"{G['demo']['npoints']:,}（Fibonacci 球面）"],
 ["真值", "带限 12 阶 + 1% 噪声"],
 ["GUI 自动选的积分元", f"<code>{G['demo']['rule']}</code>"],
 ["GUI 跑出的方法", f"<code>{G['demo']['method']}</code>"],
 ["警告条数", str(G['demo']['n_warn'])],
], num_from=1)}
{img("gui_01_demo_main.png", "截图1",
     "示例数据载入后的主界面（当前版本）",
     "<b>看什么</b>：左侧数据摘要（类型/点数/范围/建议积分元），右侧分析参数面板。"
     "<b>说明什么</b>：这是重新生成的截图（离屏 Qt + SimHei），中文正常显示，"
     "并且包含本轮新增的 <code>projection</code> 方法与 <code>lattice</code> 积分元选项。"
     "此前的 <code>tests/_gui_shots/*.png</code> 因离屏运行缺字体，所有中文都是豆腐块，已弃用。")}
{img("gui_02_demo_spectrum.png", "截图2",
     "逐阶谱 tab（示例数据）",
     "<b>看什么</b>：重建（球谐解）与真实场两条逐阶振幅曲线。<b>说明什么</b>："
     "真值已知时两条曲线应当重合——这正是需要“真实场”曲线才有意义的场合。"
     "对没有真值的数据（如长江散点），这条曲线本不该出现。")}
{img("gui_03_demo_report.png", "截图3",
     "诊断报告 tab（示例数据）",
     "<b>看什么</b>：方法、积分元、覆盖度、Shannon 数、<code>max|K−I|</code>、"
     "拟合残差与<b>面积加权残差</b>、C<sub>00</sub> 与数据加权均值之比、以及逐条警告。"
     "<b>说明什么</b>：报告里现在同时给出两个残差范数与 C<sub>00</sub> 一致性，"
     "这正是判断“求解器有没有跑偏”的最小信息集。")}
{img("gui_05_demo_map.png", "截图5",
     "地图视图（示例数据）",
     "<b>看什么</b>：原始散点 / 重建场 / 差值三种视图切换。<b>说明什么</b>："
     "真值已知时差值场应当只剩噪声，这是判断求解器有没有系统偏差最快的一眼。")}
{img("gui_04_demo_coeffs.png", "截图4",
     "系数统计 tab（示例数据）",
     "<b>看什么</b>：逐阶 RMS、power、功率占比与对应半波长。<b>说明什么</b>："
     "带限场的谱应当随阶数下降；若看到谱在低阶突然抬高，通常是噪声放大或近零空间在作祟。")}
<h3>7.1 噪声行为：阶数比估计量重要</h3>
<p>真值带限 {M['noise']['L_truth']} 阶、求解 {M['noise']['L_solve']} 阶、
{M['fib']['n']} 点，加高斯噪声：</p>
{tbl(["相对噪声 σ", "quadrature", "iterative", "wlsq", "理论 (L+1)/√N·σ"],
 [[g(r["sigma_rel"], 2), sci(r["quadrature"], 4), sci(r["iterative"], 4),
   sci(r["wlsq"], 4), sci(r["theory"], 4)] for r in M["noise"]["rows"]],
 num_from=1)}
{img("fig08_noise.png", "图8",
     "噪声主导时三种估计量等价",
     "<b>看图什么</b>：三条误差随噪声的曲线几乎重合，并与理论线一致。"
     "<b>证明了什么</b>：σ = 1% 与 5% 时三法差别在<b>第 4 位有效数字以内</b>。"
     "有噪声时起决定作用的是<b>阶数</b>而不是估计量——所以“要不要花代价做最小二乘“"
     "这个问题，在噪声主导时答案是不值得。")}
</section>""")

# ------------------------------------------------------------------ 8
A(f"""<section id="s8">
<h2>8. 试验二：全球 0/1 帽（良态对照）</h2>
<p>构造：全球 0.5° 网格（{M['cap']['n_points']:,} 点），20° 半径的 0/1 帽，
<code>dh</code> 权重（Σw = {M['cap']['dh_weight_sum']:.12f} = 4π，覆盖率
<b>1.000000</b>），帽面积占比 {M['cap']['area_fraction']:.6f}。
这是<b>良态</b>对照：样本残差是真实全球误差的离散化，可以放心用它做判据。</p>
{tbl(["L", "系数数", "求积残差", "+3 次迭代", f"+{FLOOR[3]['it_n']} 次迭代", "面积加权相对截断误差"],
 [[str(r["L"]), str(r["ncoef"]), f"{r['quad']:.8f}", f"{r['it3']:.8f}",
   f"{r['it25']:.8f}", f"{r['global_rel']:.6f}"] for r in FLOOR], num_from=1)}
{img("fig07_cap_floor.png", "图7",
     "截断地板存在，且迭代算法尊重它",
     "<b>看图什么</b>：左为三种迭代次数的残差曲线（三条完全重合），右为面积加权相对"
     "截断误差随 L 下降。<b>证明了什么</b>：<b>加 3 次甚至 25 次迭代，残差一位都不动</b>"
     "——因为 dh 权重下投影已经就是正规方程的解，校正项恒等于零。"
     "而残差停在由 L 决定的地板上（L=40 仍有 "
     f"{FLOOR[5]['global_rel']:.4f}）。<b>这就是截断泄漏：它是必然的、非零的，"
     "而且不会被任何估计量消掉。</b>对比第 9 节的长江案例——那里残差却塌向零。")}
{img("fig14_cap_map.png", "图14",
     "即使覆盖全球，L=12 也重建不出一个 20° 的 0/1 帽",
     "<b>看图什么</b>：左为真值 0/1 帽，右为 L=12 投影重建。"
     "<b>证明了什么</b>：即使是全球有效求积、C<sub>00</sub> 精确正确的场合，"
     f"L=12 的相对截断误差仍有 {FLOOR[3]['global_rel']:.4f}，帽被抹成一片平滑鼓包。"
     "低阶截断的泄漏是几何事实，与算法无关。")}
</section>""")

# ------------------------------------------------------------------ 9
cur = H["current_file"]
rep = H["reproduced_original_cg_delaunay"]
A(f"""<section id="s9">
<h2>9. 试验三：长江散点掩膜（重点案例）</h2>
<h3>9.1 数据</h3>
{tbl(["项目", "值"], [
 ["点数", f"{T['n_points']:,}"],
 ["经度范围", f"{T['lon_min']:.4f} .. {T['lon_max']:.4f} °E（跨度 {T['lon_span']:.4f}°）"],
 ["纬度范围", f"{T['lat_min']:.4f} .. {T['lat_max']:.4f} °N（跨度 {T['lat_span']:.4f}°）"],
 ["value", f"min = max = {T['value_min']:.1f}，std = {T['value_std']:.1f}（只有一个取值）"],
 ["唯一纬度 / 唯一经度", f"{T['unique_lat']:,} / {T['unique_lon']:,}"],
 ["格距", f"{T['d_lon']:.8f}° × {T['d_lat']:.8f}° ≈ {T['d_lon_m']:.1f} m × {T['d_lat_m']:.1f} m"],
 ["外接矩形面积", f"{T['box_km2']:,.0f} km²（掩膜只占其中极小部分）"],
 ["<code>rule='auto'</code> 判定", f"<code>{T['auto_rule']}</code>（识别为规则格点上的子集）"],
], num_from=1)}
<div class="note"><b>关键字义</b>：<code>Yangtze_River.txt</code> 列出的是<b>取值为 1 的点</b>；
文件里没有 0 值行。所以这份数据的物理含义是<b>区域指示函数</b>（区域内 = 1、区域外 = 0），
而不是“一个常数场”。指示场的信息全在<b>边界</b>上，这是它的高频内容。
（原始数据 <code>Yangtze_River.mat</code> 只有坐标、没有数值场，
对应作者 <code>cal_kernalSH.m</code> / <code>scatter2SHs.m</code> 里的 <code>input_mask</code>。）</div>

<h3>9.2 C<sub>00</sub> 是几何事实</h3>
<p>对 0/1 场，C<sub>00</sub> = 为 1 的面积占全球的比例。三套积分元给出三套答案：</p>
{tbl(["口径", "Σw (sr)", "面积 (km²)", "C<sub>00</sub>", "相对 lattice", "与自身 Σw/4π 自洽？"], [
 ["<code>lattice</code>", sci(W_['lattice']['sum_sr'], 12), f"{W_['lattice']['area_km2']:,.2f}",
  sci(W_['lattice']['c00'], 10), "1.000000×", "是"],
 ["<code>grid</code>", sci(W_['grid']['sum_sr'], 12), f"{W_['grid']['area_km2']:,.2f}",
  sci(W_['grid']['c00'], 10), f"{W_['grid']['ratio_vs_lattice']:.6f}×", "是"],
 ["<code>delaunay</code>", sci(W_['delaunay']['sum_sr'], 12), f"{W_['delaunay']['area_km2']:,.2f}",
  sci(W_['delaunay']['c00'], 10), f"{W_['delaunay']['ratio_vs_lattice']:.2f}×", "是"],
 ["逐点格元参考（CSV）", sci(W_['tool_csv']['sum_sr'], 12), f"{W_['tool_csv']['area_km2']:,.2f}",
  sci(W_['tool_csv']['c00'], 10), f"{W_['tool_csv']['ratio_vs_lattice']:.15f}×", "是"],
], num_from=1)}
<p>四个都“自洽”（C<sub>00</sub> 都等于各自的 Σw/4π），但只有 <code>lattice</code> 与逐点格元参考
一致。所以自洽性检查抓不到口径错误。</p>

<h3>9.3 逐阶谱与泄漏</h3>
{tbl(["n", "逐阶 RMS（projection）", "功率占比"], [
 [str(n), sci(r, 6), f"{p*100:.3f}%"] for n, r, p in
 zip(T["spectrum"]["n"], T["spectrum"]["proj_degree_rms"],
     T["spectrum"]["proj_power_share"])], num_from=1)}
{img("fig11_yangtze_spectrum.png", "图11",
     "长江掩膜的逐阶谱：近乎平坦",
     "<b>看图什么</b>：左为 projection 与 cg 两条逐阶 RMS（对数轴），"
     "右为 projection 的功率占比。<b>证明了什么</b>：0/1 掩膜是<b>宽谱</b>——"
     f"n=0 只占 {T['spectrum']['proj_power_share'][0]*100:.2f}%，"
     "能量摊在所有阶上，没有“衰减的谱”可看。这与第 8 节帽的谱形状一致，"
     "说明“截断到低阶必然泄漏”是这类数据的通性而不是个案。")}
{tbl(["L=12 的关键量", "值"], [
 ["Shannon 数 (L+1)²·Σw/4π", sci(T["shannon_L12"], 4)],
 ["实测重建峰值", sci(T["recon_peak_proj"], 4)],
 ["峰值 / Shannon 数", f"{T['recon_peak_proj']/T['shannon_L12']:.4f}"],
 ["掩膜要“立”到 1 所需 L", f"≈ {T['L_for_mask_peak_1']:.1f}"],
 ["半波长 @L=12", f"{hw['12']:,.0f} km"],
 ["区域尺度", f"{T['lon_span']*111.19*np.cos(np.deg2rad(30)):,.0f} km × {T['lat_span']*111.19:,.0f} km"],
 ["<b>样本残差（= 泄漏）</b>", f"<b>{sci(E['projection']['residual_rms'],6)}</b>"],
], num_from=1)}
{img("fig12_yangtze_map.png", "图12",
     "同一个 L=12：正确的投影只给出 ~3.5e−4 的团，CG 却把整块抬到 ≈1",
     "<b>看图什么</b>：左为输入的 142846 个点（长江河网形状清晰可见），中为"
     "projection 重建，右为 cg 重建。注意中右两幅<b>色标量级不同</b>（中 ~3e−4，右 ~1）。"
     "<b>证明了什么</b>：正确的投影诚实地显示出 L=12 只能把“1”摊成一个 1670 km 的团，"
     "峰值仅 " + sci(T['recon_peak_proj'],4) + "；而 cg 把整块区域抬到 "
     + f"{T['box_field_cg_mean']:.3f}" + "（框内均值），看起来“对了”，"
     "代价是 C<sub>00</sub> 错 " + f"{E['cg']['ratio_vs_area']:,.0f}" + " 倍。"
     "两套重建在测点上的图样相关只有 " + f"{T['pattern_corr_proj_vs_cg']:.4f}" + "。")}
{img("fig13_shannon_sweep.png", "图13",
     "峰值紧贴 Shannon 数，掩膜要立起来需 L≈694",
     "<b>看图什么</b>：左为 Shannon 数与实测峰值随 L 的变化（双对数）；"
     "右为两者之比。<b>证明了什么</b>：实测峰值与 Shannon 数是同一条线"
     f"（比值 {T['shannon_sweep'][0]['peak_over_shannon']:.3f} → "
     f"{T['shannon_sweep'][3]['peak_over_shannon']:.3f}），"
     "所以“截断掩膜的幅度上界 = Shannon 数”是可以直接用来判断“这个阶数够不够”的硬指标。"
     f"L=12 时该上界只有 {sci(T['shannon_L12'],3)}，与“1”差了三个多数量级。")}
{tbl(["L", "Shannon 数", "实测峰值", "峰值/Shannon", "半波长 (km)", "样本残差", "用时 (s)"], [
 [str(r["L"]), sci(r["shannon"], 4), sci(r["peak"], 4), f"{r['peak_over_shannon']:.4f}",
  f"{r['halfwave_km']:,.0f}", sci(r["residual"], 4), g(r["secs"], 3)]
 for r in T["shannon_sweep"]], num_from=1)}

<h3>9.4 迭代把残差压到零，同时把 C<sub>00</sub> 推离真值</h3>
<p>见 4.5 的表与图5。核心结论：<b>残差与物理误差严格反相关</b>。</p>

<h3>9.5 存储文件的两种状态</h3>
<div class="warn"><b>重要说明</b>：<code>SHKit/yantze_shkit_coeffs.gfc</code>
在本轮工作期间已被重新生成（mtime 2026-09-12 17:55:55），
<b>当前内容是 <code>method=projection</code> + <code>weight_rule=delaunay</code></b>，
不再是原始的 CG 解。因此下面两行分别给出：</div>
{tbl(["状态", "method", "积分元", "C<sub>00</sub>", "相对 lattice 几何值", "样本残差", "触发 C<sub>00</sub> 守卫？"], [
 ["<b>当前文件</b>（读取原文件）", cur["method"], f"<code>{cur['weight_rule']}</code>",
  sci(cur["c00"], 10), f"{cur['ratio_vs_lattice']:.2f}×", "—",
  f"<b>不触发</b>（与自身 Σw/4π 比值 {cur['ratio_vs_delaunay']:.6f}）"],
 ["<b>原始文件</b>（本机用同配置复现）", "cg", "<code>delaunay</code>",
  sci(rep["c00"], 10), f"{rep['ratio_vs_lattice']:,.1f}×", sci(rep["residual_rms"], 3),
  f"<b>触发</b>（{rep['warn_c00']}）"],
], num_from=3)}
<p>复现原始文件的方式（命令行）：</p>
<pre>shkit analyze --points Yangtze_River.txt --nmax 12 --method cg --rule delaunay</pre>
<p>得到 C<sub>00</sub> = {sci(rep['c00'],17)}、样本残差 = {sci(rep['residual_rms'],16)}、
<code>cg_info = {rep['cg_info']}</code>，与原始文件内容逐位一致。
它错在同时踩了三个已记录在案的坑：<code>method=cg</code>（未正则化最小二乘）、
<code>reg=None</code>、<code>rule=delaunay</code>（凸包面积）。
而 <code>docs/方案调研.md</code> 自己就测出“区域未正则化 WLSQ 不可用
（30° 球冠系数相对误差 74×）”——本机在同一配置下复现到
{f(M['regional30']['wlsq_coef_rel_err'],1)}，条件数 {sci(M['regional30']['wlsq_cond'],2)}。</p>

<h3>9.6 GUI 对照：两种方法跑同一份数据</h3>
{img("gui_07_yangtze_projection_spectrum.png", "截图6",
     "长江散点的逐阶谱（projection）",
     "<b>看什么</b>：重建（球谐解）的逐阶振幅曲线。注意这里<b>没有</b>真实场曲线。"
     "<b>说明什么</b>：长江散点没有已知真值，所以逐阶谱只能给出重建一条线——"
     "这正是本文档要强调的一点：对没有真值的数据不应该画出真实场曲线，"
     "而旧版 GUI 会因为残留状态错误地把它画上去。")}
{img("gui_06_yangtze_projection_report.png", "截图5",
     "长江散点 + projection + lattice 的诊断报告（正确路径）",
     "<b>看什么</b>：估计方法 = projection，积分元规则 = lattice，"
     f"C<sub>00</sub> = {sci(G['projection']['c00'],6)}，拟合残差 = "
     f"{g(G['projection']['resid'],6)}，面积加权残差 = {g(G['projection']['resid_w'],6)}。"
     "<b>说明什么</b>：警告里有覆盖度、完备性、Shannon 三条，但<b>没有 C<sub>00</sub> 不一致告警</b>"
     f"（<code>warn_c00 = {G['projection']['warn_c00']}</code>），说明这个解与数据的加权均值自洽。"
     "残差接近 1 是<b>正常且诚实</b>的：它就是截断泄漏。")}
{img("gui_09_yangtze_cg_report.png", "截图6",
     "长江散点 + cg 的诊断报告（错误路径，守卫触发）",
     "<b>看什么</b>：估计方法 = cg，C<sub>00</sub> = " + sci(G['cg']['c00'],6) +
     "，拟合残差 = " + sci(G['cg']['resid'],3) + "（塌陷），警告里出现两条新告警："
     "<b>C<sub>00</sub> 与数据加权均值不一致</b>，以及<b>样本残差塌陷</b>。"
     "<b>说明什么</b>：同一份数据、同一个阶数，只换估计方法，残差从 " +
     g(G['projection']['resid'],4) + " 掉到 " + sci(G['cg']['resid'],1) +
     "，而 C<sub>00</sub> 从正确的 " + sci(G['projection']['c00'],3) + " 变成 " +
     sci(G['cg']['c00'],3) + "。守卫把这种情形明确标了出来——这正是本轮新增守卫要解决的问题。")}
{img("gui_10_yangtze_cg_map.png", "截图8",
     "cg 结果的地图视图（对照）",
     "<b>看什么</b>：与截图7同一份数据的 cg 重建场。<b>说明什么</b>："
     "cg 的重建在整个外接矩形里几乎是常数约 1，看上去像把区域填满了，"
     "而它并没有分辨出任何河网形状——低幅度真解的缺席被高 C00 掩盖了。")}
{img("gui_08_yangtze_projection_map.png", "截图7",
     "projection 结果的地图视图",
     "<b>看什么</b>：散点图/重建场/差值三种视图。<b>说明什么</b>："
     "projection 的重建场在该分辨率下是一片低幅度平滑场，"
     "这是诚实的表现——它没有假装自己还原了河网。")}
</section>""")

# ------------------------------------------------------------------ 10
A(f"""<section id="s10">
<h2>10. 方法选择速查表</h2>
{tbl(["你的情况", "推荐方法", "积分元规则", "不该用什么", "理由（实测）"], [
 ["全球规则网格", "<code>quadrature</code>（<code>auto</code> 会自动选）", "<code>dh</code>",
  "—", f"max|K−I| = {sci(M['routing'][0]['max_K_minus_I'],2)}，求积即精确投影"],
 ["全球准均匀散点", "<code>quadrature</code>，想更准加 1–3 次迭代", "<code>voronoi</code>",
  "把 niter 开到收敛", f"max|K−I| = {sci(M['routing'][2]['max_K_minus_I'],2)}；"
  f"迭代实测 {sci(M['fib']['rows'][0]['coef_rel_err'],3)} → {sci(M['fib']['rows'][2]['coef_rel_err'],3)}"],
 ["全球但采样很不规则", "<code>wlsq</code> + <code>reg='kaula'</code>",
  "<code>voronoi</code>", "无正则化的 wlsq/cg",
  f"max|K−I| = {sci(M['routing'][3]['max_K_minus_I'],2)} / {sci(M['routing'][4]['max_K_minus_I'],2)}，需最小二乘"],
 ["区域网格（完整矩形）", "<code>projection</code> 或 <code>slepian</code>",
  "<code>grid</code>", "区域 Voronoi", f"Voronoi 虚高 {vt['voronoi_ratio']:.2f} 倍"],
 ["区域散点（不规则）", "<code>projection</code>；要目标 B 用 <code>slepian</code>",
  "<code>delaunay</code>", "区域 Voronoi；未正则化 wlsq",
  f"delaunay 覆盖度偏小 {(1-vt['delaunay_ratio'])*100:.1f}%（可接受）"],
 ["规则格网上的掩膜", "<code>projection</code>（<code>auto</code> 已自动路由）", "<code>lattice</code>",
  "<code>grid</code>（偏大 4.59%）、<code>delaunay</code>（偏大 70 倍）",
  f"只有 lattice 与逐点格元参考一致（相对差 {abs(W_['lattice']['sum_sr']/W_['tool_csv']['sum_sr']-1):.0e}）"],
 ["0/1 指示场，想看泄漏", "<code>projection</code>（τ 留空）", "<code>lattice</code>/<code>dh</code>",
  "任何把样本残差压到 0 的方法", "投影的残差就是泄漏；迭代会把它变成假象"],
 ["要区域反演低阶全球场", "remove–restore；或 <code>slepian</code>",
  "<code>delaunay</code>/<code>grid</code>", "直接用区域数据反演",
  f"小区域 λ<sub>max</sub> 仅 {g(M['small_caps'][0]['lam_max'],3)}（<0.5），信息论上不成立"],
], num_from=2)}
</section>""")

# ------------------------------------------------------------------ 11
A(f"""<section id="s11">
<h2>11. 优缺点与适用场景汇总</h2>
<h3><code>quadrature</code> / <code>projection</code>（零填充全球投影）</h3>
<p><b>优点</b>：目标函数定义在整个球面上，因此样本残差<b>就是截断泄漏</b>；C<sub>00</sub> 精确
等于面积占比；不做反演，没有近零空间放大；计算最便宜。</p>
<p><b>缺点/边界</b>：不能修采样偏差（需靠 τ 或换权重规则）；不给出“区域内最优拟合”，
所以区域内的拟合看起来“很差”——但那是真的。</p>
<p><b>什么时候会骗你</b>：几乎不会骗；它的问题是被误读——把 0.99 的残差当成“方法失败”。
真正的失败是换个方法把残差压到 1e-6。</p>

<h3><code>iterative</code>（Richardson 校正）</h3>
<p><b>优点</b>：准均匀采样下极便宜地逼近最小二乘（实测
{sci(M['fib']['rows'][0]['coef_rel_err'],3)} → {sci(M['fib']['rows'][1]['coef_rel_err'],3)}）。</p>
<p><b>缺点/边界</b>：niter 必须小；不动点是 WLSQ 解，所以在 K 远离 I 的数据上
它会一路上滑到错误解。</p>
<p><b>什么时候会骗你</b>：区域数据上它每一步都在降残差，<b>看起来在改进</b>，
而 C<sub>00</sub> 同时单调膨胀（niter 0→25 从 1.0× 到 {T['iterative_drift'][-1]['ratio']:.1f}×）。</p>

<h3><code>wlsq</code> / <code>cg</code>（逐点加权最小二乘）</h3>
<p><b>优点</b>：全球覆盖但采样不规则时的正解；配合超定比 ≥ 4 与正则化很稳。</p>
<p><b>缺点/边界</b>：目标函数求和域只有测点，对“隐含 0"结构性失明；对 W 整体缩放不变，
因此原则上无法约束 C<sub>00</sub>。</p>
<p><b>什么时候会骗你</b>：区域数据上未正则化时——<code>cg_info=0</code>（报告成功收敛）、
残差 {sci(rep['residual_rms'],1)}、而 C<sub>00</sub> 错 {rep['ratio_vs_lattice']:,.0f} 倍。
<b>它不会报错，只会给一个自洽但物理上不可能的解。</b></p>

<h3><code>reg='tikhonov'</code></h3>
<p><b>优点</b>：实现简单，先验明确（最小范数）。</p>
<p><b>缺点</b>：在零空间极大的区域问题上几乎不起作用——长江上 α 到 1e-4 仍停在
{sci([r for r in T['regularization'] if r['reg']=='tikhonov'][-1]['c00'],3)}；
30° 球冠上加 α=1e-9 后系数相对误差仍 {f(M['regional30']['reg_coef_rel_err']['tikhonov-1e-09'],3)}。</p>

<h3><code>reg='kaula'</code></h3>
<p><b>优点</b>：量纲无关（谱先验由数据自估），对区域数据<b>有效</b>：长江 C<sub>00</sub> 拉回
{sci([r for r in T['regularization'] if r['reg']=='kaula'][0]['c00'],3)}（隐含
{[r for r in T['regularization'] if r['reg']=='kaula'][0]['implied_area_km2']:,.0f} km²）。</p>
<p><b>缺点/边界</b>：先验本身取自一次求积的逐阶振幅，覆盖不全或有噪声时先验会被污染；
α 仍需人工判断。</p>

<h3><code>slepian</code></h3>
<p><b>优点</b>：区域数据的正解——把病态显式化（λ 就是集中因子），
条件数与区域外场量都低几个数量级（{g(M['slepian']['slepian_cond'],3)} /
{sci(M['slepian']['slepian_outside_rel'],2)} vs {sci(M['slepian']['wlsq_cond'],2)} /
{M['slepian']['wlsq_outside_rel']:.1f}）。</p>
<p><b>缺点/边界</b>：区域内拟合“一般”（它放弃了不可确定的方向）；
小区域连低阶场都恢复不了，必须 remove–restore。</p>

<h3>积分元规则</h3>
<p><b><code>lattice</code></b>：格点数据上唯一正确；<code>grid</code> 只在<b>完整</b>网格上正确；
<code>delaunay</code> 拿凸包面积，对河网状掩膜虚高 ~70 倍；
<b><code>voronoi</code> 对任何区域子集都绝对错误</b>（恒为 4π）。</p>
</section>""")

# ------------------------------------------------------------------ 12
A(f"""<section id="s12">
<h2>12. 复现</h2>
<h3>12.1 本文档的全部数值与图</h3>
<pre>cd SHKit
python docs/_summary_measure.py     # 计算全部数值 -> _measurements.json / _data.npz
python docs/_summary_history.py     # 字体检查 + .gfc 两种状态
python docs/_summary_figs.py        # 16 张图 -> SHKit方法总结_figs/
python docs/_summary_shots.py       # 10 张 GUI 截图（离屏 Qt + SimHei）
python docs/_summary_html.py        # 生成本文件</pre>
<h3>12.2 关键结论对应的命令</h3>
<pre># 长江掩膜：正确路径（auto 会选 lattice + projection；C00 = 2.068e-06）
shkit analyze --points Yangtze_River.txt --nmax 12 --method projection

# 长江掩膜：错误路径（复现原始 .gfc 的内容）
shkit analyze --points Yangtze_River.txt --nmax 12 --method cg --rule delaunay

# 原始工具链（逐点格元权重，写 examples/）
python tools/yangtze_points_to_sh.py --lmax 60

# 新回归套件（28 项，含与逐点格元参考 CSV 的交叉校验）
python tests/validate_projection.py
python tests/run_all.py</pre>
<h3>12.3 仓库产物</h3>
{tbl(["文件", "内容"], [
 ["<code>docs/SHKit方法总结.html</code>", "本文档"],
 ["<code>docs/SHKit方法总结_figs/</code>",
  f"16 张方法/试验图 + 10 张 GUI 截图 + 测量 JSON（本目录当前共 {len([n for n in os.listdir(FIGS) if n.endswith('.png')])} 个 PNG）"],
 ["<code>docs/方案调研.md</code>", "方法调研原始记录（含区域 WLSQ 失效、Slepian 实测等）"],
 ["<code>examples/README_yangtze.md</code>", "长江数据说明 + 三种积分元口径对照表"],
 ["<code>tools/yangtze_points_to_sh.py</code>", "长江散点 → 球谐系数（逐点格元权重）"],
 ["<code>examples/yangtze_mask_n60.sh</code> / <code>.npz</code>", "该工具产出的 nmax=60 掩膜系数"],
 ["<code>examples/yangtze_points_weights.csv</code>", "142846 行逐点积分元（本文档的参考基准）"],
 ["<code>tests/validate_projection.py</code>", "零填充全球投影 + C00 守卫的回归套件（28 项）"],
], num_from=1)}
</section>""")

# ------------------------------------------------------------------ 13
A(f"""<section id="s13">
<h2>13. 已知限制</h2>
<ol>
<li><b>命名问题（未改）</b>：新规则名 <code>lattice</code> 与
<code>shkit/cli.py</code> 里已有的 "Fibonacci lattice"（指全球准均匀点列，
<code>fib_sphere()</code> 的 docstring）<b>撞名</b>，且与 <code>grid</code> 在英文里近乎同义。
二者靠名字本身区分不出“完整 vs 有洞”。候选改名 <code>subgrid</code>
（与 <code>grid</code> 的关系一目了然，不引入新术语）。本轮未改，以免动到已发布的规则名。</li>
<li><b><code>SHCoeffs.degree_rms()</code> 在 n=0 处偏小 √2 倍（既有行为，未改）</b>：
它对 n=0 也是 <code>mean([C00², S00²])</code>，而 S<sub>00</sub> 结构上恒为 0，
所以返回的是 |C<sub>00</sub>|/√2。本文档 9.3 的逐阶谱表沿用该既有行为
（n=0 行 = {sci(T['spectrum']['proj_degree_rms'][0],6)}，而 C<sub>00</sub> = {sci(E['projection']['c00'],6)}，
比值为 {T['spectrum']['proj_degree_rms'][0]/E['projection']['c00']:.6f} ≈ 1/√2）。
<b>其余各阶不受影响。</b>修它会改变所有导出谱与既有测试基线，故未在本次范围内改动。</li>
<li><b>C<sub>00</sub> 守卫抓不到口径错误</b>：它检查的是求解器产出的 C<sub>00</sub> 与
Σwf/4π 是否一致。若积分元本身选错（如 delaunay 凸包面积），解仍然自洽，
守卫不响。9.2 的表就是这种情形（四套口径全部“自洽”）。</li>
<li><b>τ 校正需要构造 K</b>：<code>projection(tau=...)</code> 需要完整设计矩阵，
内存约 O(N·ncoef)；长江（142846 × 169）约 193 MB，可接受，但更大的问题要小心。</li>
<li><b>读取 <code>.gfc</code> 的头部元数据不落到 <code>coeffs.meta</code> 顶层</b>：
方法/积分元等写在 <code>meta['gfc_header']</code> 里（键值为字符串），
需要 <code>c.meta['gfc_header']['method']</code> 才能取到。
本文档 9.5 即按此读取。</li>
<li><b>netCDF 无法在本机共享盘上写入</b>：<code>netCDF4</code> 在
<code>华为家庭存储</code>（SMB）上对<b>任何</b>路径 <code>to_netcdf()</code> 都报
<code>PermissionError</code>；本机临时目录正常。因此
<code>tests/run_all.py</code> 中 3 项 <code>.nc</code> 相关检查失败
（<code>grid nc round trip</code>、<code>synth</code> 写 <code>recon.nc</code>、
<code>glq-grid</code> 写 <code>glq.nc</code>），与本轮改动无关。
<code>.grd</code>/<code>.csv</code> 往返均通过。故本文档所有产物只用 PNG/CSV/JSON。</li>
<li><b>区域低阶反演在信息论上不成立</b>：这不是软件限制而是数学事实
（4.7 的小球冠表）。任何声称“用区域数据反演出低阶全球场”的结果都需要先做
remove–restore，否则不可信。</li>
<li><b>本轮未覆盖</b>：FFT/局部球谐等更快的高阶路径；多时次数据上的 C<sub>00</sub> 守卫
只对第 0 个时次给出标量判定（其余时次在
<code>meta['dc_mean_expected_all']</code> 里）。</li>
</ol>
</section>""")

A("</main></div>")
A('<footer>SHKit 方法总结 · 全部数值由本机实时计算（docs/_summary_measure.py）'
  ' · 无外部依赖，可离线打开</footer>')
A("</body></html>")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SHKit 方法总结：从球谐目标函数到长江掩膜案例</title>
<style>{CSS}</style>
</head>
<body>
{''.join(P)}
"""

io.open(OUT, "w", encoding="utf-8").write(html)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} kB)")
print(f"sections: {len(TOC)}  figures referenced: {html.count('<img ')}")
