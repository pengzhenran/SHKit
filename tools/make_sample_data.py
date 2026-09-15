# -*- coding: utf-8 -*-
"""
生成 SHKit 的示例/测试数据。

    python tools/make_sample_data.py

输出到 SHKit/sample_data/，包含散点、网格、球谐系数三类，
每个数据集都由同一个已知的真值场生成，因此结果可以逐项核对。

数据集体设计目标：
  · points_global_fibonacci.csv  全球准均匀散点  -> 演示 rule='voronoi'
  · points_clustered.csv         全球聚簇散点    -> 演示"统一面积"为什么不行
  · points_regional_cap.csv      区域散点(球冠)  -> 演示 rule='delaunay' 与区域警告
  · grid_global_2deg.nc          全球网格 3 时次 -> 演示 rule='grid' + 时次滑块
  · grid_regional_1deg.nc        区域网格        -> 演示非全球网格
  · grid_global_5deg.grd         Surfer 格式     -> 演示 .grd 读取
  · truth_coeffs_20.sh           真值系数        -> 用于对比 / 试 synth
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shkit import io as shio
from shkit.coeffs import SHCoeffs
from shkit.synthesis import synthesis, synthesis_grid

OUT = os.path.join(ROOT, "sample_data")
LTRUTH = 20                      # 真值场的带限阶数
SEED = 20240911


# ---------------------------------------------------------------------------
def kaula_truth(nmax: int, seed: int = SEED) -> SHCoeffs:
    """随机 Kaula 型系数：振幅 ~ 1/n²，带限到 nmax。"""
    rng = np.random.default_rng(seed)
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for n in range(nmax + 1):
        amp = 1.0 if n == 0 else 1.0 / n ** 2
        for m in range(n + 1):
            C[n, m] = rng.standard_normal() * amp
            if m >= 1:
                S[n, m] = rng.standard_normal() * amp
    return SHCoeffs(C, S, {"note": "SHKit 示例真值场，带限 20 阶"})


def fib_sphere(n: int) -> tuple:
    """准均匀（Fibonacci）全球散点。"""
    i = np.arange(n) + 0.5
    lat = 90.0 - np.rad2deg(np.arccos(1.0 - 2.0 * i / n))
    lon = np.mod(np.pi * (1 + 5 ** 0.5) * np.arange(n), 2 * np.pi) * 180.0 / np.pi
    return lat, lon


def clustered_sphere(n: int, seed: int = 7, frac: float = 0.7,
                     cap_lat=40.0, cap_lon=100.0, cap_rad=35.0) -> tuple:
    """70% 的点挤在一个球冠里 —— 用来暴露"统一面积"权重的问题。"""
    rng = np.random.default_rng(seed)
    n1 = int(round(n * frac))
    cla, clo, cr = np.deg2rad(cap_lat), np.deg2rad(cap_lon), np.deg2rad(cap_rad)
    la, lo = [], []
    got = 0
    while got < n1:
        z = rng.uniform(np.sin(cla - cr), np.sin(cla + cr), max(n1 * 3, 256))
        x = rng.uniform(0, 2 * np.pi, z.size)
        lat = np.rad2deg(np.arcsin(z))
        lon = np.rad2deg(x)
        cosd = (np.sin(np.deg2rad(lat)) * np.sin(cla) +
                np.cos(np.deg2rad(lat)) * np.cos(cla) *
                np.cos(np.deg2rad(lon) - clo))
        keep = cosd >= np.cos(cr)
        la.append(lat[keep]); lo.append(lon[keep])
        got = sum(a.size for a in la)
    la1 = np.concatenate(la)[:n1]
    lo1 = np.concatenate(lo)[:n1]
    la2, lo2 = fib_sphere(n - n1)
    return np.concatenate([la1, la2]), np.concatenate([lo1, lo2])


def cap_points(seed: int = 11, cap_lat=30.0, cap_lon=100.0,
               cap_rad=25.0, n: int = 900) -> tuple:
    """球冠内的散点（模拟一个区域观测网）。"""
    rng = np.random.default_rng(seed)
    cla, clo, cr = np.deg2rad(cap_lat), np.deg2rad(cap_lon), np.deg2rad(cap_rad)
    la, lo = [], []
    got = 0
    while got < n:
        # 在球冠内均匀采样
        u = rng.uniform(0, 1, 4096)
        colat = np.arccos(1 - u * (1 - np.cos(cr)))
        az = rng.uniform(0, 2 * np.pi, 4096)
        lat = np.rad2deg(np.arcsin(np.sin(cla) * np.cos(colat) +
                                   np.cos(cla) * np.sin(colat) * np.cos(az)))
        lon = np.rad2deg(clo + np.arctan2(np.sin(az) * np.sin(colat) * np.cos(cla),
                                          np.cos(colat) - np.sin(cla) * np.sin(np.deg2rad(lat))))
        la.append(lat); lo.append(lon)
        got = sum(a.size for a in la)
    return np.concatenate(la)[:n], np.concatenate(lo)[:n]


# ---------------------------------------------------------------------------
def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    truth = kaula_truth(LTRUTH)
    rng = np.random.default_rng(99)

    print(f"真值场: 带限 {LTRUTH} 阶，{(LTRUTH + 1) ** 2} 个系数")
    print(f"输出目录: {OUT}\n")

    # ---------------------------------------------------------- 真值系数
    p = os.path.join(OUT, "truth_coeffs_20.sh")
    shio.write_coeffs(
        truth, p, layout="triangle",
        comment=("SHKit 示例真值场（带限 20 阶）\n"
                 "三角布局 [C; S]，与 m2py / gridSHconvert 的 out_SHCS 兼容"))
    print(f"  写出 {os.path.basename(p)}")

    # ------------------------------------------------- 全球准均匀散点
    lat, lon = fib_sphere(3000)
    f = synthesis(lat, lon, truth) + rng.standard_normal(3000) * 0.01
    p = os.path.join(OUT, "points_global_fibonacci.csv")
    shio.write_points(p, lat, lon, f, fmt="%.10g",
                      comment="全球准均匀散点（Fibonacci，3000 点），真值带限 20 阶 + 1% 噪声\n"
                              "建议: 积分元规则 = voronoi")
    print(f"  写出 {os.path.basename(p)}  ({lat.size} 点)")

    # --------------------------------------------------------- 聚簇散点
    lat, lon = clustered_sphere(3000)
    f = synthesis(lat, lon, truth) + rng.standard_normal(3000) * 0.01
    p = os.path.join(OUT, "points_clustered.csv")
    shio.write_points(p, lat, lon, f, fmt="%.10g",
                      comment="聚簇散点（70% 落在 40N/100E 半径 35° 的球冠内），3000 点\n"
                              "用途: 对比 voronoi 与 uniform 两种积分元（uniform 会明显变差）\n"
                              "建议: 积分元规则 = voronoi（因为全球都有覆盖）")
    print(f"  写出 {os.path.basename(p)}  ({lat.size} 点)")

    # --------------------------------------------------------- 区域散点
    lat, lon = cap_points(n=900)
    f = synthesis(lat, lon, truth) + rng.standard_normal(lat.size) * 0.01
    p = os.path.join(OUT, "points_regional_cap.csv")
    shio.write_points(p, lat, lon, f, fmt="%.10g",
                      comment="区域散点：30N/100E、半径 25° 的球冠内 900 点\n"
                              "用途: 演示区域数据的正确处理方式\n"
                              "建议: 积分元规则 = delaunay（千万不要用 voronoi）\n"
                              "注意: 分析后请看「覆盖率」与警告 —— 区域数据无法给出全球系数")
    print(f"  写出 {os.path.basename(p)}  ({lat.size} 点)")

    # --------------------------------------------------------- 全球网格
    latv = np.arange(-90.0, 90.0 + 1e-9, 2.0)      # 91
    lonv = np.arange(0.0, 360.0, 2.0)               # 180
    nt = 3
    # 时次之间用**乘性**振幅变化（而不是加性趋势），这样每个时次都是同一个带限场
    # 的缩放，可以直接与 truth_coeffs_20.sh 比较。第 0 个时次振幅恰为 1.0。
    factors = (1.0, 1.3, 0.7)
    cube = np.empty((latv.size, lonv.size, nt))
    base = synthesis_grid(latv, lonv, truth)
    for t in range(nt):
        cube[:, :, t] = (factors[t] * base
                         + rng.standard_normal(base.shape) * 0.005)
    p = os.path.join(OUT, "grid_global_2deg.nc")
    shio.write_grid(p, latv, lonv, cube,
                    var="mass_anomaly",
                    long_name="示例场（3 个时次，振幅系数 1.0 / 1.3 / 0.7）",
                    units="cm",
                    meta={"note": "SHKit sample data; every time step is the "
                                  "same degree-20 field scaled by "
                                  "1.0 / 1.3 / 0.7, plus 0.5% noise"})
    print(f"  写出 {os.path.basename(p)}  ({latv.size} x {lonv.size} x {nt})")

    # --------------------------------------------------------- 区域网格
    latv2 = np.arange(5.0, 55.0 + 1e-9, 1.0)        # 5N..55N
    lonv2 = np.arange(60.0, 140.0 + 1e-9, 1.0)      # 60E..140E
    g2 = synthesis_grid(latv2, lonv2, truth)
    g2 = g2 + rng.standard_normal(g2.shape) * 0.01
    p = os.path.join(OUT, "grid_regional_1deg.nc")
    shio.write_grid(p, latv2, lonv2, g2, var="mass_anomaly",
                    long_name="东亚区域网格 5-55N / 60-140E",
                    units="cm",
                    meta={"note": "regional grid: use rule='grid'; read the "
                                  "coverage warning"})
    print(f"  写出 {os.path.basename(p)}  ({latv2.size} x {lonv2.size})")

    # --------------------------------------------------- Surfer .grd 格式
    latv3 = np.arange(-90.0, 90.0 + 1e-9, 5.0)
    lonv3 = np.arange(0.0, 360.0, 5.0)
    g3 = synthesis_grid(latv3, lonv3, truth)
    p = os.path.join(OUT, "grid_global_5deg.grd")
    shio.write_grid(p, latv3, lonv3, g3)
    print(f"  写出 {os.path.basename(p)}  ({latv3.size} x {lonv3.size}, Surfer ASCII)")

    # ------------------------------------------------------------- 说明
    p = os.path.join(OUT, "README.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(_README)
    print(f"  写出 {os.path.basename(p)}")

    print(f"\n完成。共 {len(os.listdir(OUT))} 个文件在 {OUT}")
    return 0


_README = """# SHKit 示例数据

全部由 `tools/make_sample_data.py` 生成，**真值已知**（带限 20 阶的随机 Kaula 场，
固定随机种子），所以结果可以逐项核对。重新生成：

```bash
python tools/make_sample_data.py
```

做完分析后想知道"应该得到什么结果"，跑：

```bash
python tools/verify_sample_data.py
```

## 文件一览

| 文件 | 内容 | 载入方式 | 建议积分元 | 用来演示什么 |
| --- | --- | --- | --- | --- |
| `points_global_fibonacci.csv` | 全球准均匀散点 3000 点 | 打开散点 | `voronoi` | 标准散点球谐分析（误差 ~2e-3） |
| `points_clustered.csv` | 聚簇散点 3000 点（70% 在一个球冠里） | 打开散点 | `voronoi` | 积分元什么时候真的重要 |
| `points_regional_cap.csv` | 区域球冠 900 点（30N/100E，半径 25°） | 打开散点 | **`delaunay`** | 区域数据的正确处理与警告 |
| `grid_global_2deg.nc` | 全球 2° 网格，91×180，**3 个时次** | 打开网格 | `grid` | 时次滑块、网格分析 |
| `grid_regional_1deg.nc` | 东亚区域网格 5–55N / 60–140E，1° | 打开网格 | `grid` | 非全球网格（不能用 FFT） |
| `grid_global_5deg.grd` | 全球 5° 网格，Surfer ASCII | 打开网格 | `grid` | `.grd` 格式读取 |
| `truth_coeffs_20.sh` | **真值球谐系数**（带限 20 阶） | 命令行 `shkit info` | — | 与反演结果比对 |

`grid_global_2deg.nc` 的 3 个时次是同一个场乘以 **1.0 / 1.3 / 0.7**
再叠加 0.5% 噪声（乘性，不是加性趋势），所以每个时次都能和真值直接比较。

## 推荐的上手流程

1. 双击 `启动SHKit.bat` 打开界面。
2. **打开散点…** → `points_global_fibonacci.csv`。
   软件会自动把积分元规则设为 `voronoi`。
3. 最大阶数填 **20**（真值就是 20 阶），点 **运行分析**。
4. 看四个页签：
   - **地图** → 重建场与原图几乎一致；
   - **逐阶谱** → 重建曲线与真值曲线高度重合；
   - **诊断报告** → 看 `求积完备性 max|K−I|`、`覆盖率`、`推荐 nmax`；
   - **系数统计** → 逐阶 RMS 与半波长分辨率。
5. 换成 `grid_global_2deg.nc`，拖动**时次滑块**再运行，观察振幅随 1.0→1.3→0.7 变化。
6. 换成 `points_regional_cap.csv`，积分元选 **`delaunay`**，运行后
   **重点看诊断报告里的「覆盖率」和红色警告**。

## 积分元到底什么时候重要？（实测，别被含糊说法误导）

在 `points_clustered.csv`（采样明显不均匀）上实测，系数相对误差：

| 估计方法 | `voronoi` | `uniform` | 差距 |
| --- | --- | --- | --- |
| 求积投影 | 1.4e-02 | **1.089** | **79 倍** |
| 迭代校正 ×3 | 3.2e-03 | **444**（发散） | — |
| WLSQ | 3.2e-03 | 3.1e-03 | 基本无差别 |

三点结论：

1. **积分元作为"求积权重"时非常关键**——采样越不均匀越关键；用错权重时
   迭代校正甚至不收敛（收敛条件被破坏）。
2. **积分元作为"最小二乘的行权重"时几乎不影响结果**。最小二乘对任意正的
   行权重都是一致的；行权重只影响噪声最优性，不影响无偏性。
   所以"积分元"和"统计权重"是两个不同的东西，不要混。
3. **采样本身就准均匀时（`points_global_fibonacci.csv`）两者几乎无差别**——
   因为这时"统一面积"本来就是对的做法。

> 换句话说：**积分元不能靠猜，要看采样几何**。软件的 `rule='auto'` 就是按这个原则
> 自动判断的。

## 真值系数长什么样

`truth_coeffs_20.sh` 是三角布局的 `[C; S]` 堆叠（与 m2py / gridSHconvert 兼容），
2·NC = 462 行。用命令行查看：

```bash
shkit info --coeffs sample_data/truth_coeffs_20.sh
```

用它和反演得到的系数做差，就是真实的系数误差。

## 关于噪声

散点加了 1% 量级的高斯噪声，全球 2° 网格加了 0.5%，因此**拟合残差不会为零**，
这是正常的。按 `(L+1)/√N` 的经验式，20 阶、3000 点时噪声放大约 0.38 倍，
系数相对误差落在 1e-3 量级属正常；2° 网格有 16380 个点，精度更高。
"""


if __name__ == "__main__":
    sys.exit(main())
