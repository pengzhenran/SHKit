# -*- coding: utf-8 -*-
"""
Run every SHKit test suite in one go and print a single summary table.

Each suite is a standalone script that prints ``<n>/<N> checks passed`` and
exits non-zero on any failure, so they are run as subprocesses (this also keeps
the Qt GUI suites from leaking state into the numeric ones).

Run:  python tests/run_all.py            # everything
      python tests/run_all.py --fast     # skip the slow GUI suites
      python tests/run_all.py validate_core test_cli   # only these
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (file, short description, slow?)
SUITES = [
    ("validate_core.py", "核心：基函数 / 权重 / 积分元 / 求解器", False),
    ("validate_projection.py", "零填充全球投影目标函数 + C00 守卫", False),
    ("validate_slepian.py", "球面 Slepian 局部化", False),
    ("validate_units.py", "物理量与单位换算", False),
    ("validate_physics.py", "物理正演校验（质量、单一谐波、帽状区域）", False),
    ("validate_timeaxis.py", "时间轴与 times 贯通（A0：legacy 口径 / 反例 / 不静默广播）", False),
    ("validate_series.py", "多历元序列读写（A1：series_nc / legacy .dat / SHM）", False),
    ("validate_batch.py", "批量多历元分析（A2：等价性 / 只算一次 / report_fit）", False),
    ("validate_lonfft.py", "经度 FFT（A3 分析 / A4 综合：判据 / 相位回归 / 自动回退 / 加速）", False),
    ("validate_timeseries.py", "时间域算子（C1：趋势/周年/时间滤波/缺测/系数域≡网格域）", False),
    ("validate_products.py", "序列产品（C2：点序列/区域平均/场序列 nc/与参考场对表）", False),
    ("test_horizontal_golden.py", "水平形变黄金样本（跨包冻结契约 B0）", False),
    ("validate_horizontal.py", "水平形变包内实现 + 单位体系（B1/B3）", False),
    ("test_vs_shsynth.py", "跨包对表 SHSynth（咨询性：对方缺陷不阻塞本包）", False),
    ("test_io_roundtrip.py", "文件读写往返", False),
    ("test_cli.py", "命令行接口", False),
    ("validate_v2.py", "v2.0 发布验收（E：一条链 / 跨阶段口径 / 发版信息 / 发行包资产）", False),
    ("test_gui_units.py", "GUI 单位单选框语义（真实窗口路径）", True),
    ("test_gui_smoke.py", "GUI 离屏冒烟 + 截图", True),
]

SUMMARY = re.compile(r"(\d+)/(\d+)\s*checks passed")

#: Suites that compare against an *external* package.  A failure there means the
#: other side diverges from the frozen contract -- worth reporting, but it must
#: not turn this package's run red (the two are released separately).
ADVISORY = {"test_vs_shsynth.py"}


def run_one(fname, desc, quiet=False):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        return fname, desc, 0, 0, False, "缺失", 0.0
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", path], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    dt = time.time() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    m = None
    for m in SUMMARY.finditer(out):
        pass                                   # take the last one
    if m:
        npass, total = int(m.group(1)), int(m.group(2))
    else:
        npass, total = 0, 0
    fails = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[FAIL]")]
    ok = (proc.returncode == 0) and total > 0 and npass == total
    if not quiet and not ok:
        print(out)
    note = f"rc={proc.returncode}"
    if fails:
        note = f"{len(fails)} 项失败"
    return fname, desc, npass, total, ok, note, dt


def main(argv):
    want = [a for a in argv if not a.startswith("-")]
    slow_ok = "--fast" not in argv and "--no-gui" not in argv
    suites = [s for s in SUITES
              if (not s[2] or slow_ok)
              and (not want or any(w in s[0] for w in want))]

    print("=" * 78)
    print(f"  SHKit 全量测试（{len(suites)} 个套件）"
          f"{'' if slow_ok else '  [--fast: 已跳过 GUI 套件]'}")
    print("=" * 78)

    rows = []
    for fname, desc, _slow in suites:
        print(f"  ... {fname}", flush=True)
        rows.append(run_one(fname, desc))

    print()
    print(f"  {'套件':<24s} {'通过':>10s}  {'用时':>7s}  说明")
    print("  " + "-" * 74)
    tp = tt = 0
    nbad = 0
    nadvisory = 0
    for fname, desc, npass, total, ok, note, dt in rows:
        advisory = fname in ADVISORY and not ok
        flag = "OK  " if ok else ("WARN" if advisory else "FAIL")
        print(f"  {fname:<24s} {npass:>4d}/{total:<4d} {dt:>6.1f}s  {flag} {desc}")
        tp += npass
        tt += total
        if not ok:
            if advisory:
                nadvisory += 1
            else:
                nbad += 1
    print("  " + "-" * 74)
    print(f"  合计 {tp}/{tt} 项通过，失败套件 {nbad} 个"
          + (f"，咨询性告警 {nadvisory} 个（跨包，不影响本包）" if nadvisory else ""))

    if nadvisory:
        print("  ⚠ 咨询性告警来自跨包对表 —— 请把上面的 WARN 转给 SHSynth 侧。")
    if (tt and tp != tt) or nbad:
        print("\n  有失败项，请查看上面的详细输出。")
        return 1
    print("\n  全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
