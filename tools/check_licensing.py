# -*- coding: utf-8 -*-
"""
SHKit 闭源商用许可自检 / Licensing self-check for closed-source distribution.

在**正式打包用的那个 Python 环境**里运行：

    python tools/check_licensing.py

检查三件事：
  1. 环境里没有 GPL-only 的 Qt 模块（Qt Charts / Qt Data Visualization / ...）；
  2. 需要的 LGPL 模块都在；
  3. 发行包必需的许可文本文件已就位。

退出码 0 = 通过，1 = 有风险项。
"""

from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# GPL-only（或商业许可）的 Qt 模块：闭源分发绝对不能用
GPL_ONLY = [
    "QtCharts",
    "QtDataVisualization",
    "QtGraphs",
    "QtVirtualKeyboard",
    "QtQuick3D",
]

# SHKit 实际需要 / 允许的 LGPLv3 模块
LGPL_OK = [
    "QtCore", "QtGui", "QtWidgets", "QtSvg", "QtNetwork",
    "QtPrintSupport", "QtSql", "QtXml", "QtConcurrent", "QtOpenGL",
    "QtUiTools", "QtTest", "QtQml", "QtQuick", "QtHelp",
]

# 发行包必须携带的文件（相对项目根目录）
# 说明书（截图版 HTML + 配图）与公众号二维码也必须随包：帮助菜单和「关于」
# 直接读它们，少了就是空框。
REQUIRED_FILES = [
    "licenses/LGPL-3.0.txt",
    "licenses/GPL-3.0.txt",
    "licenses/NOTICE.txt",
    "docs/许可与闭源商用说明.md",
    "docs/使用说明.html",
    "docs/地球重力与人类生活TVGG.jpg",
    "docs/使用说明_img/gui_01_demo_main.png",
    "docs/使用说明_img/dlg_about.png",
]

problems = []
notes = []


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> int:
    section("环境 / environment")
    print("Python :", sys.version.split()[0])
    print("执行文件:", sys.executable)

    # ---------------------------------------------------------------- PySide6
    section("1. PySide6 与 Qt 模块")
    try:
        import PySide6
        print("PySide6:", PySide6.__version__)
        ver = PySide6.__version__
    except ImportError:
        print("PySide6 未安装 —— 若只用于命令行/库，可以忽略此节。")
        ver = None

    if ver is not None:
        present_gpl = []
        for m in GPL_ONLY:
            try:
                importlib.import_module(f"PySide6.{m}")
                present_gpl.append(m)
            except ImportError:
                pass
        if present_gpl:
            problems.append(
                f"检测到 GPL-only 模块: {present_gpl} —— 闭源分发有风险。"
                " 修复: pip uninstall PySide6 PySide6-Addons && "
                "pip install PySide6-Essentials")
            print("GPL-only 模块:", present_gpl, "  <-- 有问题")
        else:
            print("✅ 未检测到任何 GPL-only 的 Qt 模块")

        ok, missing = 0, []
        for m in LGPL_OK:
            try:
                importlib.import_module(f"PySide6.{m}")
                ok += 1
            except ImportError:
                missing.append(m)
        print(f"LGPL 模块可用: {ok}/{len(LGPL_OK)}")
        if missing:
            notes.append(f"未安装（可忽略，按需）: {missing}")

        # 是否误装了 addons
        try:
            import importlib.metadata as md
            dists = {d.metadata["Name"].lower() for d in md.distributions()}
            if "pyside6-addons" in dists:
                problems.append(
                    "已安装 pyside6-addons（完整版 PySide6 的一部分），"
                    "其中含 GPL-only 模块。建议卸载：pip uninstall PySide6-Addons")
            elif "pyside6" in dists:
                problems.append(
                    "已安装完整的 'pyside6' 元包（通常会把 Addons 一起带来）。"
                    "建议只保留 PySide6-Essentials")
            else:
                print("✅ 只安装了 PySide6-Essentials（未发现 pyside6 / addons 元包）")
        except Exception as exc:                      # noqa: BLE001
            notes.append(f"无法枚举已安装发行包: {exc}")

    # --------------------------------------------------------------- 绘图库
    section("2. 绘图库")
    try:
        import matplotlib
        print("matplotlib:", matplotlib.__version__,
              "—— PSF/BSD 风格许可，可闭源商用 ✅")
    except ImportError:
        print("matplotlib 未安装（GUI 需要）")
        notes.append("GUI 需要 matplotlib")

    # 检查代码里有没有误用 GPL 的 Qt 模块
    section("3. 源码扫描：有没有 import GPL 模块")
    hits = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "shkit")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:                          # noqa: BLE001
                continue
            for m in GPL_ONLY:
                if f"Qt{m}" in text:
                    hits.append(f"{os.path.relpath(p, ROOT)}: Qt{m}")
    if hits:
        problems.append("源码中引用了 GPL-only 模块: " + "; ".join(hits))
        print("❌", hits)
    else:
        print("✅ shkit/ 下没有任何对 GPL-only Qt 模块的引用")

    # ------------------------------------------------------------ 许可文件
    section("4. 发行包必需的许可文本")
    for rel in REQUIRED_FILES:
        p = os.path.join(ROOT, rel)
        ok = os.path.exists(p)
        size = os.path.getsize(p) if ok else 0
        print(f"{'✅' if ok else '❌'} {rel}"
              + (f"  ({size} 字节)" if ok else "  <-- 缺失"))
        if not ok:
            problems.append(f"缺少 {rel}")

    # --------------------------------------------------------------- 汇总
    section("结论")
    if notes:
        print("提示：")
        for n in notes:
            print("  ·", n)
    if problems:
        print("\n❌ 发现以下问题，闭源商用前必须解决：")
        for p in problems:
            print("  ·", p)
        return 1
    print("✅ 通过：当前环境与发行包满足闭源商用所需的许可条件。")
    print("   提醒：本脚本只做技术自检，不构成法律意见；")
    print("   正式发行前请核对 docs/许可与闭源商用说明.md 第 5.3 节的自查清单。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
