# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for SHKit — closed-source commercial packaging.

Build with::

    pip install PyInstaller
    pyinstaller --noconfirm --clean packaging/shkit.spec

Deliberate choices (see ../docs/许可与闭源商用说明.md):

* **onedir, never onefile.**  PySide6/Qt is LGPLv3; the licence requires the
  end user to be able to replace the library.  In a ``--onedir`` bundle the Qt
  DLLs sit next to the executable as separate files and can be swapped; in a
  ``--onefile`` bundle they are packed inside the archive and silently
  unpacked to a temp directory, which is the community-recognised relinking
  risk.  This spec therefore never produces a one-file build.
* **GPL-only Qt modules are excluded** so that even an accidental installation
  of the full ``PySide6`` (which pulls ``PySide6-Addons``) cannot leak
  Qt Charts / Qt Data Visualization / Qt Graphs into the distribution.
* ``licenses/`` and ``docs/`` are bundled because LGPLv3 obliges us to ship the
  licence texts and a prominent notice.  ``docs/`` also carries what the Help
  menu reads at runtime - ``使用说明.html`` (screenshot-based user guide),
  ``使用说明_img/`` (its pictures) and the WeChat QR image - so a bundle that
  drops ``docs/`` shows an empty guide and an empty QR panel.
  ``tools/check_licensing.py`` verifies those files are present.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_submodules

PROJECT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# 需要看启动期报错时：  set SHKIT_CONSOLE=1  再打包，会出带控制台窗口的 SHKit.exe
CONSOLE = os.environ.get("SHKIT_CONSOLE", "") not in ("", "0", "false")

# --- 绝不进包的模块 ----------------------------------------------------------
# 1) 其它 Qt 绑定：PyInstaller 一旦同时发现 PySide6 与 PyQt5/PyQt6 就直接中止。
#    发行包只用 PySide6；如果打包环境里恰好还装了 PyQt5，这几条就是保险。
# 2) GPL-only 的 Qt 模块（见 GPL_ONLY）：即使误装了完整 PySide6 也不会漏进来。
#
# 注：用 packaging/build_installer.ps1 里的精简虚拟环境打包时（推荐），
# NumPy/SciPy 是 PyPI 的 OpenBLAS 构建，环境里没有 pandas 的绘图链路、
# pyarrow、h5py 等重依赖，也不必再靠 excludes 去兜。
ALWAYS_EXCLUDE = [
    "PyQt5", "PyQt5.sip", "PyQt6", "PyQt6.sip", "qtpy", "PySide2",
    "_tkinter", "IPython", "jupyter", "notebook",
]

# --- GPL-only or unwanted Qt modules: never ship these ----------------------
GPL_ONLY = [
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtQuick3D",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtSerialPort",
    "PySide6.QtSerialBus",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSensors",
    "PySide6.QtTextToSpeech",
    "PySide6.QtSpatialAudio",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtHttpServer",
    "PySide6.QtWebSockets",
    "PySide6.QtWebChannel",
    "PySide6.QtWebView",
    # heavy helpers we do not use
    "PySide6.scripts",
    "PySide6.assistant",
    "PySide6.designer",
    "PySide6.linguist",
    "tkinter",
    "IPython",
    "notebook",
]

# --- data files that must travel with the executable ------------------------
# 发行包只带**运行时真正需要**的东西：
#   data/                  勒夫数表 + 离线海岸线
#   licenses/              LGPLv3 等许可全文（帮助菜单要读；LGPL 也要求随包附带）
#   docs/使用说明.html      帮助 → 使用说明（F1）渲染的那份说明书
#   docs/使用说明_img/       说明书里的图（缺了就是空框）
#   docs/地球重力与人类生活TVGG.jpg   公众号二维码（「关于」里直接显示）
#   docs/使用说明_GUI.md     纯文本兜底版（HTML 读不到时退回它）
# 其余开发资料（方案调研、方法总结、生成脚本、探测脚本…）一律不进发行包。
DOCS = os.path.join(PROJECT, "docs")
QR = "地球重力与人类生活TVGG.jpg"

datas = [
    (os.path.join(PROJECT, "data"), "data"),
    (os.path.join(PROJECT, "licenses"), "licenses"),
    (os.path.join(DOCS, "使用说明.html"), "docs"),
    (os.path.join(DOCS, "使用说明_GUI.md"), "docs"),
    (os.path.join(DOCS, QR), "docs"),
    (os.path.join(DOCS, "使用说明_img"), os.path.join("docs", "使用说明_img")),
]

hiddenimports = collect_submodules("shkit")

# --- conda 版 Python 的外置运行库 --------------------------------------------
# conda 的 python3xx.dll 会依赖若干放在 <env>\Library\bin 里的 DLL，其中最要命的
# 是 `_ctypes.pyd` 依赖的 **ffi-8.dll**：PyInstaller 只扫 DLLs/ 与 site-packages，
# 扫不到 Library\bin，结果冻结后的程序在解释器刚启动、加载 site 之前就
# `ImportError: DLL load failed while importing _ctypes` 直接崩掉（连报错窗口都没有）。
# 这里把该目录里的 DLL 显式收进来；用纯 pip 环境打包时该目录不存在，自动跳过。
CONDA_BIN = os.path.join(sys.prefix, "Library", "bin")
binaries = []
if os.path.isdir(CONDA_BIN):
    for _name in sorted(os.listdir(CONDA_BIN)):
        if _name.lower().endswith(".dll"):
            binaries.append((os.path.join(CONDA_BIN, _name), "."))

# 入口必须是**包外**的 launcher：PyInstaller 把入口当顶层脚本跑，
# 直接拿 shkit/gui/app.py 当入口会让里面的相对导入全炸
# （ImportError: attempted relative import with no known parent package）。
a = Analysis(
    [os.path.join(SPECPATH, "shkit_launcher.py")],
    pathex=[PROJECT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=GPL_ONLY + ALWAYS_EXCLUDE,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SHKit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=CONSOLE,         # 默认窗口程序；SHKIT_CONSOLE=1 时带控制台便于排错
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SHKit",
)
