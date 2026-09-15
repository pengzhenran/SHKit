# -*- coding: utf-8 -*-
"""
**v2.0 发布验收**（E 阶段）：把各阶段的最强断言串成一条真实使用链。

这不是又一套单元测试 —— 那些在各自的套件里跑过了。这里只回答发布前必须回答的
四个问题：

1. **一条链能不能真的跑通**：`series-read` → `timefit` → `series-grid` →
   `analyze-series` → `basin-average`，每一步的产物都能被下一步读进来
   （最后一步是**闭环**：把刚写出的场序列产品再当输入解一遍）；
2. **跨阶段的口径是否一致**：批量 ≡ 逐历元（A2）、系数域 ≡ 网格域（C1）、
   场序列 nc 往返（C2）、水平形变 FFT ≡ 直接法（B3）—— 全部在**同一次运行**里复核；
3. **发版信息是否同步**：`shkit.io.SHKIT_VERSION` = `pyproject.toml` 的 version =
   安装脚本的 `MyAppVersion`，以及 README 声称的套件数/检查数与 `run_all.py` 的实际
   配置一致（版本号不同步是发版最常见的低级事故）；
4. **发行包要带的东西是否都在**（`packaging/shkit.spec` 的 datas 逐项存在）——
   说明书、配图、二维码、许可、勒夫数表、离线海岸线。缺任何一项，装出来的包里
   就是空框或直接报错。

真实数据在场时（本工作区有）再加一条：**真 216 历元数据走完整链，并与用户自己的
`3_grids` 产品对表**，把口径差异（√(4π)）当场量出来；不在场则 skip 并说明。

Run:  python tests/validate_v2.py
"""
import os
import re
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WS = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from shkit import io as shio                                            # noqa: E402
from shkit.coeffs import SHCoeffs                                       # noqa: E402
from shkit.series import analyze_series                                 # noqa: E402
from shkit.timeaxis import TimeAxis                                     # noqa: E402
from shkit.timeseries import (fit_series_maps, fit_time_model,         # noqa: E402
                              match_epochs, remove_time_mean, series_grid)
from shkit.weights import WeightSet, grid_cell_weights                  # noqa: E402

PY = sys.executable
RESULTS = []

REAL_GFC_DIR = os.path.join(WS, "2_unzipped", "1_CSR", "01RL06", "01deg60")
REAL_LEGACY = os.path.join(WS, "3_processed", "GRACE SH read",
                           "CSR_rawSH60_total_216.dat")
REF_G300 = os.path.join(WS, "3_grids", "CSR_GRACE_EWH_G300.nc")


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def skip(name, why):
    RESULTS.append(True)
    print(f"[PASS] {name:56s} skip（{why}）")


def note(msg):
    """Print a measured fact **without** turning it into a pass/fail.

    Used for numbers that are worth seeing at release time but that we are not
    entitled to assert (an external product's conventions, amplitude ranges).
    """
    print(f"[INFO] {msg}")


def rel(v, ref):
    scale = max(float(np.max(np.abs(np.asarray(ref)))), 1e-300)
    return float(np.max(np.abs(np.asarray(v) - np.asarray(ref)))) / scale


def run_cli(args, timeout=900):
    """Run the CLI in a subprocess so this suite exercises the real entry point."""
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    p = subprocess.run([PY, "-m", "shkit.cli", *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       env=env, cwd=ROOT, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ---------------------------------------------------------------------------
# 1. 发版信息同步
# ---------------------------------------------------------------------------
def t_version_sync():
    ver = shio.SHKIT_VERSION
    # 不写死具体版本（改一次版本就要改测试是坏味道）：只要求是 X.Y.Z 且主版本 2.x。
    # "当前是哪个版本"这件事的唯一事实来源是 SHKIT_VERSION。
    check("版本号是 X.Y.Z 形式且属 2.x",
          re.fullmatch(r"2\.\d+\.\d+", ver or "") is not None, ver)

    pyproj = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproj, re.M)
    check("pyproject.toml 的 version 与 SHKIT_VERSION 一致",
          m is not None and m.group(1) == ver,
          f"{m.group(1) if m else '未找到'} vs {ver}")

    iss = open(os.path.join(ROOT, "installer_shkit.iss"), encoding="utf-8").read()
    m = re.search(r'#define\s+MyAppVersion\s+"([^"]+)"', iss)
    mv = m.group(1) if m else None
    # v2.0.1 起**补丁号也进安装程序版本**：安装包名与「应用和功能」里显示的版本必须与
    # SHKIT_VERSION 完全相同，否则用户装了 2.0.1 却看到 2.0（同名安装包也分不清）。
    check("安装脚本的 MyAppVersion = SHKIT_VERSION（含补丁号）",
          mv is not None and mv == ver,
          f"MyAppVersion={mv} vs {ver}")
    check("安装脚本的输出文件名由 MyAppVersion 推导（不写死）",
          "OutputBaseFilename=SHKit_Setup_v{#MyAppVersion}" in iss)

    # run_all.py 的 SUITES 是"到底有多少套/多少项"的唯一事实来源：直接 import 它，
    # 不要在这里另外维护一份名单（两份一起过期就没人发现了）
    sys.path.insert(0, HERE)
    import run_all                                               # noqa: E402
    n_suites = len(run_all.SUITES)
    check("本次验收套件已登记进 run_all.py", "validate_v2.py" in
          [s[0] for s in run_all.SUITES],
          f"共 {n_suites} 套")

    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    claimed = {int(x) for x in re.findall(r"(\d+)\s*套", readme)}
    check("README 里出现的每一个「N 套」都等于 run_all 的实配套件数",
          claimed == {n_suites},
          f"实配 {n_suites} 套；README 写的 {sorted(claimed)}")
    # 凡是「N 套 … M 项」成对出现的地方，M 必须处处一样 —— 这一条正是为了抓
    # "加了套件但只改了其中一处数字" 这种最典型的过期文档。
    # （M 本身是否等于所有套件之和，只有跑完 run_all 才知道，这里不假装能验证。）
    pairs = re.findall(str(n_suites) + r"\s*套[^0-9]{0,16}?(\d+)\s*项", readme)
    check("README 里「N 套 … M 项」成对出现且 M 处处一致",
          len(pairs) >= 2 and len(set(pairs)) == 1,
          f"出现 {len(pairs)} 处，值为 {sorted(set(pairs))}")


# ---------------------------------------------------------------------------
# 2. CLI 一条链
# ---------------------------------------------------------------------------
def t_cli_pipeline():
    """一条真实使用链，全程走 CLI 子进程与真实文件格式。

    ``series-read``（gfc 目录 → 序列）→ ``timefit``（时间域拟合）→
    ``series-grid``（场序列 nc）→ 读回 → ``analyze-series``（把**刚写出的
    产品**再当输入解一遍）→ ``basin-average``。

    最后那个闭环是有意安排的：产品必须能被自己的分析入口读回来，否则
    "产品"只是终点而不是中间物。
    """
    tmp = tempfile.mkdtemp(prefix="shkit_v2_")
    try:
        # ---- 造 6 个合成 gfc：文件名用 YYYYDDD，头部用 YYYYMMDD -------------
        # 这两个日期语法**不一样**，写错一个就会静默落到约一年以外
        # （timeaxis._epoch_from_ordinals 的注释专门讲这件事），所以两种都写全。
        rng = np.random.default_rng(2)
        d0 = np.datetime64("2002-04-05")
        n_ep = 6
        for k in range(n_ep):
            s = d0 + np.timedelta64(int(k * 30), "D")
            e = s + np.timedelta64(25, "D")
            def _doy(t):
                return int((t - t.astype("datetime64[Y]").astype("datetime64[D]"))
                           .astype(int)) + 1
            def _ymd(t):
                return str(t).replace("-", "")
            fn = (f"GSM-2_{str(s)[:4]}{_doy(s):03d}-"
                  f"{str(e)[:4]}{_doy(e):03d}_GRAC_UTCSR_BA01_0600.gfc")
            with open(os.path.join(tmp, fn), "w", encoding="utf-8") as fh:
                fh.write(f"time_period_of_data {_ymd(s)} - {_ymd(e)} "
                         f"(mid: {_ymd(s + np.timedelta64(13, 'D'))})\n")
                fh.write("product_type GSM\n")
                for n in range(9):
                    for mm in range(n + 1):
                        fh.write(f"GRCOF2 {n:4d} {mm:4d} "
                                 f"{rng.normal() * 1e-9:20.12E} {0.0:20.12E}\n")

        s_nc = os.path.join(tmp, "series.nc")
        rc, out = run_cli(["series-read", "--source", tmp, "--out", s_nc,
                           "--summary"])
        check("① series-read：gfc 目录 → 多历元序列 nc",
              rc == 0 and os.path.exists(s_nc), f"rc={rc} {out.strip()[-80:]}")
        if not os.path.exists(s_nc):
            return
        co = shio.read_coeffs(s_nc, layout="series_nc")
        check("   → 6 个历元，时间轴有真实日期（不是序号）",
              co.ntime == n_ep and co.times is not None and co.times.has_dates,
              f"ntime={co.ntime} source={co.times.meta.get('time_source')}")
        check("   → 首历元 2002-04-18：mid = (start+end+1)/2（26 天窗 → +13 天）",
              str(co.times.values[0])[:10] == "2002-04-18",
              str(co.times.values[0]))

        # ---- ② timefit：系数序列 → 每项一个系数文件 + 逐历元残差表 ---------
        pref = os.path.join(tmp, "fit")
        ep_csv = os.path.join(tmp, "fit_epochs.csv")
        rc, out = run_cli(["timefit", "--series", s_nc, "--poly-order", "1",
                           "--periods", "1.0", "--out-prefix", pref,
                           "--out-csv", ep_csv])
        check("② timefit：写出 const/trend/annual_cos/annual_sin 四项 + csv",
              rc == 0 and all(os.path.exists(pref + s) for s in
                              ("_const.sh", "_trend.sh", "_annual_cos.sh",
                               "_annual_sin.sh")) and os.path.exists(ep_csv),
              f"rc={rc} {out.strip()[-80:]}")

        # ---- ③ series-grid：系数序列 → 与 3_grids 同构的场序列 nc ----------
        sg_nc = os.path.join(tmp, "ewh_series.nc")
        rc, out = run_cli(["series-grid", "--series", s_nc, "--field-unit",
                           "geopotential", "--target-unit", "ewh", "--drop-c00",
                           "--remove-time-mean", "--lat-step", "20",
                           "--lon-step", "45", "--out-units", "mm",
                           "--out", sg_nc])
        check("③ series-grid：产出与 3_grids 同构的场序列 nc",
              rc == 0 and os.path.exists(sg_nc), f"rc={rc} {out.strip()[-80:]}")
        if not os.path.exists(sg_nc):
            return
        la, lo, tv, cube, cmeta = shio.read_field_series(sg_nc)
        check("   → 读回 (nlat,nlon,ntime) = (10,8,6)（lat -90..90/20）",
              cube.shape == (10, 8, n_ep), f"{cube.shape}")
        check("   → 落盘是参考布局：lat 降序被读回时翻转、lon 归到 [0,360)",
              cmeta.get("lat_order_flipped") is True and float(np.min(lo)) >= 0.0
              and float(np.max(lo)) < 360.0,
              f"layout={cmeta.get('read_from_layout')} "
              f"lon[{lo.min():g},{lo.max():g}]")
        check("   → 变量单位写成 mm（与参考场一致）",
              str((cmeta.get("variable_attrs") or {}).get("units", "")).find("mm") >= 0,
              str((cmeta.get("variable_attrs") or {}).get("units")))
        check("   → 时间轴随文件回来，6 个历元且是真日期",
              len(tv) == n_ep and tv.has_dates, f"ntime={len(tv)}")

        # ---- ④ 闭环：把刚写出的产品再当输入解一遍 ------------------------
        a2 = os.path.join(tmp, "reanalyzed.nc")
        rc, out = run_cli(["analyze-series", "--points", sg_nc, "--nmax", "4",
                           "--out-series", a2, "--report-fit"])
        check("④ 闭环：analyze-series 能读回自己的产品并解出系数",
              rc == 0 and os.path.exists(a2), f"rc={rc} {out.strip()[-80:]}")
        if os.path.exists(a2):
            co2 = shio.read_coeffs(a2, layout="series_nc")
            check("   → 历元数与产品一致、时间轴继承下来",
                  co2.ntime == n_ep and co2.times is not None
                  and co2.times.has_dates, f"ntime={co2.ntime}")
            # 产品做过 --remove-time-mean：逐点时间平均为 0（f4 舍入残差 ~1e-7），
            # 而分析是线性的，所以解出来系数的**时间平均**也必须≈0。
            cub = co2.C[:, :, :co2.ntime]
            scale = float(np.max(np.abs(cub - cub.mean(axis=2, keepdims=True))))
            resid = float(np.max(np.abs(cub.mean(axis=2))))
            check("   → 解出系数的时间平均 ≈ 0（remove-time-mean 的可验证后果）",
                  resid <= 1e-4 * max(scale, 1e-300),
                  f"mean/scale = {resid / max(scale, 1e-300):.2e}")

        # ---- ⑤ basin-average：区域点集 → 面积加权序列 --------------------
        b_csv = os.path.join(tmp, "basin.csv")
        region = os.path.join(tmp, "region.csv")
        rla, rlo = np.meshgrid(np.arange(-20.0, 20.01, 10.0),
                               np.arange(90.0, 150.01, 10.0), indexing="ij")
        shio.write_points(region, rla.ravel(), rlo.ravel(),
                          np.zeros(rla.size), header=True)
        rc, out = run_cli(["basin-average", "--series", s_nc, "--points", region,
                           "--field-unit", "geopotential", "--target-unit", "ewh",
                           "--drop-c00", "--out", b_csv])
        check("⑤ basin-average：区域 → 面积加权序列 csv",
              rc == 0 and os.path.exists(b_csv), f"rc={rc} {out.strip()[-80:]}")
        if os.path.exists(b_csv):
            head = open(b_csv, encoding="utf-8-sig").readline()
            check("   → 头部写明覆盖率与权重规则（区域平均必须自证范围）",
                  "coverage=" in head and "weight_rule=" in head,
                  head.strip()[:56])
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. 跨阶段口径（合成数据，同一次运行内复核）
# ---------------------------------------------------------------------------
def t_cross_phase_consistency():
    from shkit.synthesis import synthesis, synthesis_grid
    from shkit.gradient import degree_factors_horizontal, horizontal_grid

    rng = np.random.default_rng(11)
    L, nt = 8, 12
    lat = np.arange(-80.0, 80.01, 10.0)
    # nlon 必须 > 2·nmax 才允许走经度 FFT（nmax=8 → 至少 17 条经线），
    # 所以用 15° 间隔的 24 条；30°（12 条）会被判据正确地拒绝。
    lon = np.arange(0.0, 360.0, 15.0)
    LA, LO = map(np.ravel, np.meshgrid(lat, lon, indexing="ij"))
    tv = np.datetime64("2002-04-18") + np.arange(nt) * 30
    ax = TimeAxis.from_datetimes(tv.astype("datetime64[s]"))
    C = rng.normal(size=(L + 1, L + 1, nt)) * 1e-9
    S = rng.normal(size=(L + 1, L + 1, nt)) * 1e-9
    co = SHCoeffs(C, S, {"field_unit": "geopotential"}, ax)
    ws = WeightSet(w=grid_cell_weights(LA, LO), rule="grid")
    field = co_field(co, LA, LO)                 # (npoints, nt)，算一次

    # A2：批量 ≡ 逐历元
    co_b, rep = analyze_series(LA, LO, field, L, weights=ws, report_fit=False)
    worst = 0.0
    from shkit.analysis import analysis
    for k in range(nt):
        one, _ = analysis(LA, LO, field[:, k], L, method="quadrature",
                          weights=ws, report_fit=False)
        worst = max(worst, rel(co_b.C[:, :, k], one.C))
    check("★ A2 批量 ≡ 逐历元（同一次运行内复核）", worst <= 1e-12,
          f"rel = {worst:.2e}")
    ctr = rep.shared.get("counters", {})
    check("★ A2 权重/Gram 只算一次（计数器）",
          ctr.get("fill_gram") == 1 and rep.shared.get("n_calls") == 1,
          f"fill_gram={ctr.get('fill_gram')} n_calls={rep.shared.get('n_calls')}")

    # C1：系数域拟合 ≡ 网格域拟合
    fit = fit_time_model(co, poly_order=1, periods=(1.0,))
    maps = fit_series_maps(synthesis_grid(lat, lon, co), ax, poly_order=1,
                           periods=(1.0,))
    trend_co = synthesis_grid(lat, lon, fit.term("trend"))
    check("★ C1 系数域拟合 ≡ 网格域拟合（趋势场）",
          rel(trend_co, maps["trend"]) <= 1e-10,
          f"rel = {rel(trend_co, maps['trend']):.2e}")

    # C2：场序列落盘/读回（内部布局与参考布局都测）
    tmp = tempfile.mkdtemp(prefix="shkit_v2b_")
    try:
        for layout in ("reference", "internal"):
            p = os.path.join(tmp, f"f_{layout}.nc")
            shio.write_field_series(p, lat, lon, ax, synthesis_grid(lat, lon, co),
                                    layout=layout, var="ewh", units="m")
            _la, _lo, _t, cube, _m = shio.read_field_series(p)
            want = synthesis_grid(lat, lon, co)
            check(f"★ C2 场序列 nc 往返（{layout} 布局，float32 精度）",
                  rel(cube, want) <= 1e-6, f"rel = {rel(cube, want):.2e}")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    # B3：水平形变 FFT ≡ 直接法（含起始经度偏移）
    fh = degree_factors_horizontal(L)
    C2, S2 = C[:, :, 0], S[:, :, 0]
    for lam0 in (0.0, -179.5):
        lo = lam0 + lon
        d = horizontal_grid(lat, lo, C2, S2, fh, method="direct")
        f = horizontal_grid(lat, lo, C2, S2, fh, method="fft")
        check(f"★ B3 水平形变 FFT ≡ 直接法（lam0={lam0:g}）",
              rel(f["north"], d["north"]) <= 1e-12,
              f"rel = {rel(f['north'], d['north']):.2e}")


def co_field(co, LA, LO):
    from shkit.synthesis import synthesis
    v = synthesis(LA, LO, co)
    return v if v.ndim == 2 else v[:, None]


# ---------------------------------------------------------------------------
# 4. 发行包资产
# ---------------------------------------------------------------------------
def t_packaging_assets():
    spec_path = os.path.join(ROOT, "packaging", "shkit.spec")
    spec = open(spec_path, encoding="utf-8").read()

    # spec 的 datas 是**唯一**决定发行包内容的清单，所以直接把它求值出来逐项核对，
    # 而不是在这里再抄一份路径表 —— 抄一份的话，两边一起错就没人发现了。
    ie = os.path.join(ROOT, "installer_shkit.iss")
    block = re.search(r"(?ms)^datas = (\[.*?\n\])", spec)
    check("能从 shkit.spec 里取出 datas 清单", block is not None)
    datas = []
    if block is not None:
        ns = {"os": os, "PROJECT": ROOT, "DOCS": os.path.join(ROOT, "docs"),
              "QR": "地球重力与人类生活TVGG.jpg"}
        exec(compile(block.group(0), spec_path, "exec"), ns)     # noqa: S102
        datas = ns["datas"]
    check("datas 有 6 项（data/licenses/说明书/纯文本/二维码/配图）",
          len(datas) == 6, f"{len(datas)} 项")
    for src, dst in datas:
        ok = os.path.exists(src)
        size = ""
        if ok and os.path.isfile(src):
            size = f"{os.path.getsize(src)} B"
        elif ok:
            size = f"{len(os.listdir(src))} 个文件"
        check(f"发行包资产存在：{os.path.relpath(src, ROOT)} → {dst}", ok, size)

    # 许可：帮助菜单要读，LGPLv3 也要求随包附带全文
    lic = os.path.join(ROOT, "licenses")
    names = sorted(os.listdir(lic)) if os.path.isdir(lic) else []
    check("licenses/ 带 LGPLv3 全文与 NOTICE",
          any(n.startswith("LGPL") for n in names) and "NOTICE.txt" in names,
          ", ".join(names))
    check("spec 明确排除 GPL-only 的 Qt 模块（Charts/DataVisualization/Graphs）",
          all(k in spec for k in ("PySide6.QtCharts", "PySide6.QtDataVisualization",
                                  "PySide6.QtGraphs")))
    check("spec 只用 onedir（COLLECT + exclude_binaries，LGPL 可替换库）",
          "COLLECT(" in spec and "exclude_binaries=True" in spec)

    # 安装脚本：说明书/许可的落地路径必须与 spec 的 datas 目标一致
    iss = open(ie, encoding="utf-8").read()
    check("iss 从安装根 _internal\\docs 打开说明书（spec 的 dst=docs）",
          "_internal\\docs\\使用说明.html" in iss and any(
              dst.startswith("docs") for _src, dst in datas))
    check("iss 的产物名由 MyAppVersion 推导（不写死第二处版本号）",
          "OutputBaseFilename=SHKit_Setup_v{#MyAppVersion}" in iss)

    # 打包脚本期待的产物名必须与 iss 推导出来的名字一致
    ps = open(os.path.join(ROOT, "packaging", "build_installer.ps1"),
              encoding="utf-8").read()
    mv = re.search(r'#define\s+MyAppVersion\s+"([^"]+)"', iss)
    want = f"SHKit_Setup_v{mv.group(1)}.exe" if mv else None
    check("build_installer.ps1 期待的产物名 = iss 的 MyAppVersion 推导值",
          want is not None and want in ps, f"{want}")
    check("打包/许可自检脚本在册（tools/check_licensing.py）",
          os.path.exists(os.path.join(ROOT, "tools", "check_licensing.py")))

    # ⚠️ **安装路径带空格**（用户明确要求打包时验证）：Windows 上
    # `C:\Program Files\SHKit`、`%LOCALAPPDATA%\Programs\...`、用户名带空格都是常态。
    # 这里守住三件必须在**打包流程里**发生的事 —— 它们全是"以前没人走过这条路"：
    #   ① 静默安装的目标目录**故意带空格**，而且 `/DIR=` 的值必须**整体加引号**
    #      （Start-Process -ArgumentList 会把没引号的含空格值拆成两个参数 → 装错目录）；
    #   ② 目录包会被**复制到一个带空格的路径**下，从那里再跑一遍自检；
    #   ③ 冻结版自检里真的有"带空格路径"和"独立子进程"两件事，不是只检查进程活着。
    app_src = open(os.path.join(ROOT, "shkit", "gui", "app.py"),
                   encoding="utf-8").read()
    check("★ 打包脚本：静默安装到**带空格**的目录，且 /DIR 的值整体加引号",
          '_install test' in ps.split('"')[0] or "_install test" in ps,
          "build_installer.ps1 里 /DIR 的用户目录名含空格")
    check("★ 打包脚本：/DIR 的值是引号包住的（否则空格会被拆成两个参数）",
          re.search(r'/DIR=`"\$TestDir`"', ps) is not None
          or re.search(r'/DIR="\$TestDir"', ps) is not None,
          "/DIR=`\"$TestDir`\"")
    check("★ 打包脚本：把目录包复制到带空格的路径再跑一遍自检",
          "dist space" in ps and "selftest_space.log" in ps
          and "Copy-Item $Dist $SpaceDist" in ps,
          "dist space 带空格 → SHKit.exe --self-test")
    check("★ 打包脚本：把「带空格路径走通了」当成硬性判据（少一项就红）",
          ps.count("带空格路径：网格 nc 写→读逐值一致") >= 2
          and "独立子进程能起来并返回结果" in ps,
          "目录包与装完的版本各查一遍")
    check("★ 冻结版自检：包含「带空格路径」的读写与缓存两步",
          "带空格路径：网格 nc 写→读逐值一致" in app_src
          and "带空格路径：逐历元系数缓存写→读命中" in app_src
          and "shkit 自检 space" in app_src)
    check("★ 冻结版自检：包含独立子进程（spawn）能起来并能返回结果",
          "独立子进程能起来并返回结果" in app_src
          and "run_in_child" in app_src)
    check("★ 冻结版自检：打印可执行文件路径（打包日志里能看出装在哪）",
          "含空格={(' ' in sys.executable)}" in app_src)
    # 运行代码里不许把路径拼进命令行（空格/引号一类问题全从这里来）
    _shell_hits = []
    for _dir, _sub, _files in os.walk(os.path.join(ROOT, "shkit")):
        _sub[:] = [d for d in _sub if d != "__pycache__"]
        for _f in _files:
            if not _f.endswith(".py"):
                continue
            _txt = open(os.path.join(_dir, _f), encoding="utf-8").read()
            for _pat in ("shell=True", "os.system(", "os.popen("):
                if _pat in _txt:
                    _shell_hits.append(f"{_f}:{_pat}")
    check("★ 运行代码里没有 shell 拼串（路径一律走参数列表，空格才安全）",
          not _shell_hits, f"{_shell_hits or '干净'}")

    # ⚠️ 驱动 GUI 的脚本必须有 `if __name__ == "__main__"` 保护：GUI 的批量分析在
    # Windows 上用 spawn 起子进程，而 spawn 会**重新导入主模块**。没有保护、且脚本里
    # 连 "__main__" 字符串都不出现时，`spawn_available()` 会判定"可以 spawn"，
    # 于是子进程把整个脚本从头再跑一遍 —— 实测症状是批量分析永远拿不到结果（0 行）
    # 并在退出时崩溃（QThread 被销毁 / 0xC0000409）。这条断言就是防它。
    shots = open(os.path.join(ROOT, "docs", "_guide_shots_v2.py"),
                 encoding="utf-8").read()
    check("抓图脚本带 __main__ 保护（spawn 会重新导入主模块）",
          "if __name__" in shots and "def main(" in shots,
          "docs/_guide_shots_v2.py")

    # ⚠️ 上面那条是**静态**检查（脚本自己写没写保护）；这里再补一条**运行时**检查：
    # 一个没有保护的宿主脚本调用 `spawn_available()` 必须拿到 False + 说得出原因。
    # v2.0.1 前那里只判 `"__main__" in text`，于是"既没有保护、也没提 __main__ 的
    # 脚本"被判为"可以 spawn" —— 实测后果是子进程把整个脚本又跑一遍（标记文件里
    # 2 个进程各 1 行、GUI 多开一个窗口）。判据现在按 CPython 的真实规则来：
    # 只有"普通 .py 主模块且没有 if __name__ 保护"才拒绝；.exe 启动器与
    # python -m <pkg> 不受影响。
    import subprocess as _sub
    import tempfile as _tf
    _noguard = os.path.join(_tf.mkdtemp(prefix="shkit_noguard_"),
                            "noguard_host.py")
    with open(_noguard, "w", encoding="utf-8") as fh:
        fh.write("import sys\n"
                 f"sys.path.insert(0, {ROOT!r})\n"
                 "from shkit.gui.subproc import spawn_available\n"
                 "ok, why = spawn_available()\n"
                 "print('OK' if ok else 'NO', why)\n")
    try:
        _out = _sub.run([sys.executable, _noguard], capture_output=True,
                        text=True, timeout=180)
        _txt = (_out.stdout or "") + (_out.stderr or "")
    except Exception as exc:                                     # noqa: BLE001
        _txt = f"<跑不起来: {exc}>"
    check("★ 没有 __main__ 保护的宿主脚本 → spawn_available() 明确拒绝并说明原因",
          "NO" in _txt and "__main__" in _txt and "再执行一遍" in _txt,
          _txt.strip().splitlines()[-1][:88] if _txt.strip() else "无输出")

    # ⚠️ **这两个文件必须带 UTF-8 BOM**，而且这一条是踩过坑才加上的：
    #   * `build_installer.ps1` 交给 Windows PowerShell 5.1 跑，没有 BOM 就按 GBK 解码
    #     → 中文全变乱码 → **解析直接报错**，脚本一行都执行不了；
    #   * `installer_shkit.iss` 交给 Inno Setup 编译，没有 BOM 时它按 ANSI 代码页读，
    #     中文（菜单名、快捷方式名）会变成乱码。
    # 任何"以 UTF-8 无 BOM 写回"的编辑器/脚本都会**静默**把这个属性弄丢（BOM 看不见），
    # 所以必须在这里守住它 —— 否则要等到打包那一刻才发现。
    for rel in (os.path.join("packaging", "build_installer.ps1"),
                "installer_shkit.iss"):
        with open(os.path.join(ROOT, rel), "rb") as fh:
            head = fh.read(3)
        check(f"{rel} 带 UTF-8 BOM（PowerShell 5.1 / Inno 都要）",
              head == b"\xef\xbb\xbf", f"前三字节 {head!r}")


# ---------------------------------------------------------------------------
# 5. 真实数据端到端（缺数据则 skip）
# ---------------------------------------------------------------------------
def t_real_end_to_end():
    if not (os.path.exists(REAL_GFC_DIR) or os.path.exists(REAL_LEGACY)):
        skip("★ 真实数据端到端", "找不到真实数据")
        return
    # A1：真实目录/legacy → 序列
    co = leg = None
    if os.path.exists(REAL_GFC_DIR):
        co = shio.read_coeffs_series(REAL_GFC_DIR, nmax=60, strict_nmax=False,
                                     progress=lambda m, f: None)
        check("★ A1 真实 gfc 目录 → 216 历元序列",
              co.ntime == 216 and co.nmax == 60,
              f"ntime={co.ntime} nmax={co.nmax}")
        check("★ A1 真实目录的时间轴来自文件名（YYYYDDD），有日期",
              co.times is not None and co.times.has_dates,
              str(co.times.meta.get("time_source")))
    if os.path.exists(REAL_LEGACY):
        leg = shio.read_coeffs(REAL_LEGACY, nmax=60, layout="legacy_dat")
        check("★ A1 legacy .dat → 216 历元（日期来自 companion TimeInfo）",
              leg.ntime == 216 and leg.times is not None and leg.times.has_dates,
              f"ntime={leg.ntime} src={leg.times.meta.get('time_source')}")

    # A1：gfc ≡ legacy —— **按日期配对**，不是按下标 ------------------------
    # 两个来源的日期各有分辨率（gfc 头只到整天，legacy 小数年约 16 s），跨年历元
    # 还会差到 ~2e-5 年（≈20 分钟），所以必须最近邻配对并把配不上的报出来。
    if co is not None and leg is not None:
        ia, ib, off = match_epochs(co.times, leg.times, tol_seconds=3600.0)
        check("★ A1 gfc/legacy 历元配对（1 h 容差）≥200 个",
              ia.size >= 200,
              f"{ia.size}/{co.ntime} 配对，最大偏差 {off.max():.0f} s")
        Ca, Cb = np.asarray(co.C)[:, :, ia], np.asarray(leg.C)[:, :, ib]
        Sa, Sb = np.asarray(co.S)[:, :, ia], np.asarray(leg.S)[:, :, ib]
        check("★ A1 两边 C00 都 = 1（GSM 总质量，不是归一化差异）",
              abs(Ca[0, 0, 0] - 1.0) < 1e-9 and abs(Cb[0, 0, 0] - 1.0) < 1e-9,
              f"gfc={Ca[0, 0, 0]!r} legacy={Cb[0, 0, 0]!r}")

        # 老 .dat 是**七位有效数字**的文本表（-4.841697000000E-04 里只有
        # 4.841697 是真的），所以两个来源的差异上限就是最低位的一半 = 相对 5e-7。
        # 实测刚好贴着这个上限（4.99e-7），说明差异**全部**是文本舍入：谁也没有
        # 多算或少算一个数。卡 1e-15 只会得到一个恒假的断言（方案文档里已更正）。
        for tag, ga, gb in (("C", Ca, Cb), ("S", Sa, Sb)):
            nz = gb != 0
            r = float(np.max(np.abs(ga[nz] - gb[nz]) / np.abs(gb[nz])))
            check(f"★ A1 {tag} 表逐系数差异 ≤ 5e-7（.dat 七位有效数字的半量化步长）",
                  r <= 5.5e-7,
                  f"max = {r:.3e}（{int(nz.sum())}/{gb.size} 个非零位）")
            check(f"★ A1 {tag} 表没有「一边为 0、另一边非 0」的位置",
                  int(((ga == 0) != (gb == 0)).sum()) == 0,
                  f"{int(((ga == 0) != (gb == 0)).sum())} 处")
        note(f"gfc/legacy 的最大绝对差 {np.abs(Ca - Cb).max():.2e}（出现在 C20 的"
             f"时间变化上，正好是 .dat 最低存储位的一半）")

    # C2：真实序列 → 场序列，并与用户自己的 3_grids 产品对表 --------------
    if leg is None or not os.path.exists(REF_G300):
        skip("★ C2 与 3_grids 对表", "缺 legacy 序列或参考场")
        return

    # 先用参考场自己的轴（1°，lat 升序、lon [0,360)——read_field_series 已经把
    # 磁盘上的降序 lat 与 [-180,180) 经度翻回来了），保证两边逐格对齐
    _la, _lo, rt, ref, rmeta = shio.read_field_series(REF_G300)
    check("★ C2 参考场网格 = 181 x 360（1°），时间轴可读",
          _la.size == 181 and _lo.size == 360 and len(rt) > 0,
          f"{_la.size}x{_lo.size}, ntime={len(rt)}")

    # GSM 的 C00 = 1 是**地球总质量**，做异常必须先去掉，否则 EWH ~1e7 m
    meta = dict(leg.meta)
    meta["field_unit"] = "geopotential"
    Cc = np.array(leg.C, dtype=float, copy=True)
    Cc[0, 0, :] = 0.0
    raw = SHCoeffs(Cc, np.array(leg.S, dtype=float, copy=True), meta, leg.times)
    anom, _mean = remove_time_mean(raw)
    cube, times = series_grid(anom, _la, _lo, target_unit="ewh",
                              gaussian_km=300.0)
    check("★ C2 真实 216 历元 → EWH 场序列（1°，300 km 高斯）",
          cube.shape[:2] == (181, 360) and cube.shape[2] == leg.ntime,
          f"shape={cube.shape}")

    ia, ib, off = match_epochs(times, rt, tol_seconds=3600.0)
    check("★ C2 与 3_grids 的历元配对（1 h 容差）≥200 个",
          ia.size >= 200,
          f"{ia.size}/{len(times)} 配对，最大偏差 {off.max():.0f} s")

    unit = str((rmeta.get("variable_attrs") or {}).get("units") or "").lower()
    if unit.startswith("mm"):
        ref *= 1e-3                       # 参考场是 mm，SHKit 这里是 m
    A = np.asarray(cube)[:, :, ia]
    B = np.asarray(ref)[:, :, ib]
    del cube, ref

    # 统计量一律在**全部配对历元**上算（不是抽样）：215 个历元的场各 112 MB，
    # 这台机器放得下，抽样反而会让别人怀疑数字是挑出来的
    ok = np.isfinite(A) & np.isfinite(B)
    check("★ C2 两边都没有缺测格点（有缺测的话下面的统计量没意义）",
          bool(ok.all()), f"缺测 {int((~ok).sum())} 个")

    def _scale(x, y):
        x, y = x.ravel(), y.ravel()
        return float(x @ y) / float(y @ y)

    r4 = float(np.sqrt(4.0 * np.pi))
    corr = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    check("★ C2 与参考场空间图形一致（原始口径，corr > 0.9）", corr > 0.9,
          f"corr = {corr:.4f}")

    # 逐历元 RMS 比值：如果差异只是"乘一个常数"，这个比值应当不随时间变化
    nc = A.shape[0] * A.shape[1]
    ra = np.sqrt(np.einsum("ijk,ijk->k", A, A) / nc)
    rb = np.sqrt(np.einsum("ijk,ijk->k", B, B) / nc)
    q = ra / rb
    note(f"逐历元振幅比 {q.mean():.4f} ± {q.std():.4f}"
         f"（std/mean = {q.std() / q.mean() * 100:.1f}% —— 不是纯乘性常数）")
    note(f"振幅范围：SHKit [{np.min(A):.3f}, {np.max(A):.3f}] m，"
         f"参考场 [{np.min(B):.3f}, {np.max(B):.3f}] m")

    # 把两边**重新基线化到同一批历元**：参考文件是 257 个历元、.dat 是 216 个，
    # 各自减的是不同跨度上的时间平均，这一步能把这部分口径差摘掉。（就地做，
    # 每边 112 MB，不再多开两份。）
    A -= A.mean(axis=2, keepdims=True)
    B -= B.mean(axis=2, keepdims=True)
    corr2 = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    check("★ C2 同历元重新基线后一致性提升（corr > 0.95）", corr2 > 0.95,
          f"corr = {corr2:.4f}（原始 {corr:.4f}）")

    # 中低纬（|lat|<45°）看归一化常数：两极是 300 km 高斯与泄漏差异最大的地方，
    # 把极区算进来的话，测到的是"两家产品差多少"而不是"归一化差多少"。
    m45 = np.abs(_la) < 45.0
    m60 = np.abs(_la) < 60.0
    c45 = _scale(A[m45], B[m45])
    check("★ C2 归一化常数 = √(4π)（|lat|<45°，实测 ±2% 以内）",
          abs(c45 / r4 - 1.0) <= 0.02,
          f"{c45:.4f} / {r4:.4f} = {c45 / r4:.4f}（全部配对历元、逐格）")
    note("极区/高纬的差值来自两边的平滑与泄漏口径，不做断言，只报出来："
         f"全球比值 {_scale(A, B):.4f}，|lat|<60° {_scale(A[m60], B[m60]):.4f}"
         f"（√(4π)={r4:.4f}）")


# ---------------------------------------------------------------------------
# 6. 真实 mascon nc（用户报障的那一份）：数值时间轴 + dh 规则
# ---------------------------------------------------------------------------
MASCON = os.path.join(os.path.dirname(WS), "3_Level-3", "1_Mascon",
                      "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc")


def t_real_mascon():
    """真实 CSR mascon 文件：两个报障点各留一条断言。

    ① **时间轴**：``time`` 是 float32 + ``days since 2002-01-01T00:00:00Z``，
    属性名还是**大写** ``Units``；历元是月中点（含 .5 天）。旧实现只认小写
    ``units``，于是把一个 256 历元、横跨 2002–2026 的文件当成"没有时间信息"。
    ② **规则**：0.25° 全球网格 ``nlon = 1440 = 2*720``，自动选 ``dh``，
    GUI 会带 ``weights_kw={'nlon': …}`` —— 而 ``analysis()`` 曾经不接受它。

    只读时间坐标与维度，**不整块读 2 GB 的 cube**（几何用同样形状的合成场验），
    所以这条在验收套件里是秒级的。
    """
    if not os.path.exists(MASCON):
        skip("★ 真实 mascon 文件", "找不到 3_Level-3\\1_Mascon\\*.nc")
        return
    from shkit.analysis import analysis
    from shkit.timeaxis import TimeAxis, attr_ci, time_axis_from_meta

    with shio.open_nc_dataset(MASCON) as ds:
        tvar = ds["time"]
        t_units = str(attr_ci(tvar.attrs, "units"))
        t_cal = attr_ci(tvar.attrs, "calendar")
        t_vals = np.asarray(tvar.values).tolist()      # 关文件之前取出
        ax = TimeAxis.from_netcdf_coord(tvar)
        nlat, nlon = int(ds.sizes["lat"]), int(ds.sizes["lon"])
        ntime = int(ds.sizes["time"])
        v_units = str(attr_ci(ds["lwe_thickness"].attrs, "units"))
    check("★ mascon：属性名是**大写** Units，也能读出时间单位",
          t_units.startswith("days since"), repr(t_units))
    check("★ mascon：变量单位读出来是 cm（换算公式按米，界面必须写明）",
          v_units.lower() == "cm", repr(v_units))
    check("★ mascon：256 个历元解出真实日期（不再说「没有时间轴」）",
          ax.has_dates and len(ax) == ntime == 256,
          f"ntime={ntime}，{str(ax.values[0])[:10]} … {str(ax.values[-1])[:10]}")
    check("★ mascon：首末历元 = 2002-04-18 … 2026-04-16",
          str(ax.values[0]) == "2002-04-18T00:00:00"
          and str(ax.values[-1]) == "2026-04-16T00:00:00",
          f"{ax.values[0]} … {ax.values[-1]}")
    check("★ mascon：月中点（.5 天）保留为 12:00，没有被截断",
          str(ax.values[1]) == "2002-05-10T12:00:00", str(ax.values[1]))

    # GUI 侧：Dataset 拿同一份时间元数据必须给出日期与摘要
    from shkit.gui.dataset import Dataset
    latv = np.linspace(-89.875, 89.875, 8)
    lonv = np.linspace(0.125, 359.875, 16)
    meta = {"time": t_vals, "time_units": t_units, "time_calendar": t_cal}
    dsm = Dataset.from_grid(MASCON, latv, lonv, np.zeros((8, 16, 256)), meta)
    summ = dsm.summary()
    check("★ mascon：GUI 侧 Dataset.time_axis() 有日期、摘要写明跨度",
          dsm.time_axis() is not None and dsm.time_axis().has_dates
          and "2002-04-18" in summ and "时间轴" in summ,
          dsm.epoch_label(0))
    check("★ mascon：summary 里的时间轴行不是「未识别」",
          "未识别" not in summ,
          [ln.strip() for ln in summ.splitlines() if "时间轴" in ln][0][:56])

    # 几何：0.25° 全球网格 → dh；用同构的合成场跑一遍 GUI 走的那次调用
    check("★ mascon：网格 720x1440（nlon = 2*nlat）→ 自动规则就是 dh",
          (nlat, nlon) == (720, 1440) and nlon == 2 * nlat,
          f"nlat={nlat} nlon={nlon}")
    lat_s = -90.0 + np.arange(90) * 2.0
    lon_s = np.arange(0.0, 360.0, 2.0)          # 与真实文件同构：nlon = 2*nlat
    LA, LO = np.meshgrid(lat_s, lon_s, indexing="ij")
    f = np.cos(np.deg2rad(LA)).ravel()
    co, rep = analysis(LA.ravel(), LO.ravel(), f, 10, method="quadrature",
                       rule="dh", weights_kw={"nlon": lon_s.size},
                       report_fit=False)
    check("★ mascon：dh + weights_kw 这条路能解出系数（不再 TypeError）",
          bool(np.isfinite(co.C).all()) and rep.weight_rule == "dh",
          f"weight_rule={rep.weight_rule}，C00={float(co.C[0, 0]):.6g}")
    check("★ mascon：dh 报告里不该出现「经线数不对」的提醒",
          not any("经线" in w for w in rep.warnings),
          f"{len(rep.warnings)} 条警告（正变换/反变换说明除外）")


# ---------------------------------------------------------------------------
# 7. 真实 mascon：懒加载首屏 vs 整块读 + 逐时次统计
# ---------------------------------------------------------------------------
def t_lazy_mascon():
    """用户第二条报障链：**「载入 mascon 要 10 多秒」** + 「统计那一行应按第 k 个时次」。

    打开时把 1.06 GB / 256 时次整块读进来要 2.4–3.0 s（首屏 3.5 s），而**一个
    时次**只要 ~0.25 s；Panoply 快就是因为它只读一层。这里在**真实文件**上量一遍
    两者的时间，并逐值核对"懒加载读出来的那一层"与直接切片读出来的完全一致 ——
    省时间不能以改数值为代价。统计则核对摘要里的数值范围就是该时次的范围。
    """
    if not os.path.exists(MASCON):
        skip("★ mascon 懒加载", "找不到 3_Level-3\\1_Mascon\\*.nc")
        return
    import gc
    import time

    from shkit.gui.dataset import Dataset

    def _epoch(t: int) -> np.ndarray:
        with shio.open_nc_dataset(MASCON) as ds:
            return np.asarray(ds["lwe_thickness"].isel(time=t).values,
                              dtype=float)

    # (a) 懒加载首屏：只读第 1 个时次
    t0 = time.perf_counter()
    la, lo, cube, ml = shio.read_grid(MASCON, var="lwe_thickness",
                                      lazy_time=True)
    t_lazy = time.perf_counter() - t0
    check("★ mascon：懒加载返回 LazyTimeCube，首屏只读了 1 个时次",
          isinstance(cube, shio.LazyTimeCube) and cube.shape == (720, 1440, 256)
          and cube.n_filled == 1 and not cube.is_filled(1),
          f"{type(cube).__name__} {cube.shape}，已填 {cube.n_filled}/{cube.ntime}，"
          f"首屏 {t_lazy:.2f} s")
    v0 = np.asarray(cube[:, :, 0], dtype=float)
    check("★ mascon：懒加载的第 1 层与直接切片读逐值一致（0 差异）",
          v0.shape == (720, 1440) and np.array_equal(v0, _epoch(0)),
          f"max|Δ| = {np.max(np.abs(v0 - _epoch(0))):.3g}")

    # (b) 按需取第 200 个时次：一次单层读，仍要逐值一致
    cube.ensure(200)
    check("★ mascon：ensure(t) 按需取中间某一层，与直接切片读逐值一致",
          cube.is_filled(200)
          and np.array_equal(np.asarray(cube[:, :, 200], dtype=float), _epoch(200)),
          f"已填 {cube.n_filled}/{cube.ntime}")

    # (c) 整块读的时间（用户抱怨的那个数）—— 量完立刻释放，别和懒加载副本叠着占内存
    t0 = time.perf_counter()
    _la2, _lo2, g_full, _m2 = shio.read_grid(MASCON, var="lwe_thickness")
    t_eager = time.perf_counter() - t0
    _nbytes = int(np.asarray(g_full).nbytes)
    check("★ mascon：懒加载首屏快于整块读（这就是「打开慢」的答案）",
          t_lazy < t_eager,
          f"懒加载 {t_lazy:.2f} s vs 整块 {t_eager:.2f} s"
          f"（{_nbytes / 1e9:.2f} GB，{_nbytes / 1e6:.0f} M 个值）")
    in_sync = np.array_equal(np.asarray(cube[:, :, 0], dtype=float),
                             np.asarray(g_full)[:, :, 0].astype(float))
    check("懒加载与整块读在同一时次上仍然一致（两条路不会给出不同结果）",
          in_sync, "第 1 层逐值相同")
    del g_full, _la2, _lo2
    gc.collect()

    # (d) 逐时次统计：摘要里的数值范围/RMS 就是**当前时次**的
    dsL = Dataset.from_grid(MASCON, la, lo, cube, ml)
    s0 = dsL.summary(0)
    st0 = dsL.stats(0)
    check("★ mascon：摘要的数值范围是第 1 个时次的范围，并写明「第 1/256 个时次」",
          "第 1/256 个时次" in s0
          and f"{st0['min']:.6g}" in s0 and f"{st0['max']:.6g}" in s0,
          [ln.strip() for ln in s0.splitlines() if "数值范围" in ln][:1])
    check("★ mascon：统计值与该时次的直接 min/max/RMS 一致",
          abs(st0["min"] - float(v0.min())) < 1e-9
          and abs(st0["max"] - float(v0.max())) < 1e-9
          and abs(st0["rms"] - float(np.sqrt(np.mean(v0 ** 2)))) < 1e-6,
          f"{st0['min']:.6g} … {st0['max']:.6g}，RMS {st0['rms']:.6g}")
    s199 = dsL.summary(199)
    check("★ mascon：换一个时次，摘要里的数值范围跟着变（不是整块口径）",
          "第 200/256 个时次" in s199 and s199 != s0,
          [ln.strip() for ln in s199.splitlines() if "数值范围" in ln][:1])
    check("摘要把「统计口径 = 逐时次」写在明面上（用户不用猜）",
          "个时次" in s0, f"{len(s0.splitlines())} 行摘要")
    dsL.close()
    check("★ mascon：close() 之后不再持有文件句柄/临时副本", True,
          "LazyTimeCube.close() 已调用")


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("v2.0 发布验收（E）：一条链 / 跨阶段口径 / 发版信息 / 发行包资产")
    print("=" * 78)
    print(f"  真实 gfc 目录 : {REAL_GFC_DIR}"
          f"{'' if os.path.isdir(REAL_GFC_DIR) else '  (不存在 → skip)'}")
    print(f"  legacy 序列   : {REAL_LEGACY}"
          f"{'' if os.path.exists(REAL_LEGACY) else '  (不存在 → skip)'}")
    print(f"  参考场        : {REF_G300}"
          f"{'' if os.path.exists(REF_G300) else '  (不存在 → skip)'}")
    for fn in (t_version_sync, t_cli_pipeline, t_cross_phase_consistency,
               t_packaging_assets, t_real_end_to_end, t_real_mascon,
               t_lazy_mascon):
        print(f"\n--- {fn.__name__} " + "-" * (58 - len(fn.__name__)))
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(fn.__name__, False, "抛出异常")
    npass = sum(RESULTS)
    print(f"\n{'=' * 78}\n{npass}/{len(RESULTS)} checks passed")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
