# -*- coding: utf-8 -*-
"""SHKit 打包入口（PyInstaller 用）。

为什么要单独一个入口文件
------------------------
PyInstaller 会把**入口脚本**当顶层脚本执行（``__name__ == "__main__"``、
没有父包）。如果直接拿 ``shkit/gui/app.py`` 当入口，它里面的相对导入
（``from .. import __version__``、``from .main_window import MainWindow``）
就会炸：

    ImportError: attempted relative import with no known parent package

所以打包入口放在包外，先 ``import shkit.gui.app``（包被正常导入，相对导入全部
成立）再调用它。用户直接 ``python packaging/shkit_launcher.py`` 也能用。

另外这里把 matplotlib 的配置目录指到临时目录：冻结版若去写
``%USERPROFILE%\\.matplotlib``，在只读或受管环境里会报错，而 SHKit 并不需要
保存 matplotlib 配置。
"""
from __future__ import annotations

import os
import sys
import tempfile

__all__ = ["main"]


def _isolate_matplotlib_config() -> None:
    """Point matplotlib's config dir at a temp folder (written before it loads)."""
    tmp = os.path.join(tempfile.gettempdir(), "shkit-mplconfig")
    try:
        os.makedirs(tmp, exist_ok=True)
    except OSError:
        return
    os.environ.setdefault("MPLCONFIGDIR", tmp)


def main(argv=None) -> int:
    _isolate_matplotlib_config()
    # 项目根目录加入 sys.path：源码运行时能找到 shkit 包；
    # 冻结后 PyInstaller 已经把 shkit 打进包，这一步是无害的兜底。
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.dirname(here)):
        if cand not in sys.path and os.path.isdir(os.path.join(cand, "shkit")):
            sys.path.insert(0, cand)
            break

    from shkit.gui.app import main as gui_main
    return gui_main(list(sys.argv if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
