# -*- coding: utf-8 -*-
"""
shkit.gui
=========

PySide6 desktop interface for SHKit.

The GUI is an **optional** layer: importing :mod:`shkit` never imports PySide6,
so the library and the CLI keep working on machines without Qt installed.

    python -m shkit.gui                 # launch
    python -m shkit.gui data.csv        # launch with a file preloaded

Licensing note (important for closed-source commercial use)
-----------------------------------------------------------
The GUI uses **PySide6-Essentials** under **LGPLv3**, dynamically linked and
unmodified.  ``PySide6`` (the full metapackage) is deliberately avoided because
``PySide6-Addons`` ships GPL-only modules.  All plotting uses **matplotlib**.
See ``docs/许可与闭源商用说明.md`` for the full compliance checklist.
"""

from __future__ import annotations

__all__ = ["main", "check_pyside", "MainWindow", "Dataset"]


def main(argv=None) -> int:
    """Launch the SHKit GUI (imports PySide6 lazily)."""
    from .app import main as _main
    return _main(argv)


def check_pyside():
    """Return ``(ok, message)`` describing whether the GUI can start."""
    from .app import check_pyside as _c
    return _c()


def __getattr__(name):        # lazy re-exports: keep PySide6 out of `import shkit`
    if name == "MainWindow":
        from .main_window import MainWindow
        return MainWindow
    if name == "Dataset":
        from .dataset import Dataset
        return Dataset
    raise AttributeError(name)
