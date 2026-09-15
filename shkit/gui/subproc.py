# -*- coding: utf-8 -*-
"""
shkit.gui.subproc
=================

Run the expensive solves in a **separate process** so that a hard crash (a bad
BLAS build, an OOM kill, a segfault in a native library) takes down the child and
not the window (D12).

Why a process and not a thread
------------------------------
The existing workers are ``QThread``\\ s: they keep the UI responsive but they
share the interpreter, so a native crash or ``MemoryError`` inside the solver
kills the GUI with it.  A process also lets the solver be killed outright, which
a thread cannot be.

Protocol
--------
Deliberately small and boring -- a ``multiprocessing.Queue`` of tuples:

===== ==========================================
kind  payload
===== ==========================================
``p`` progress ``(message, fraction)``
``r`` result (any picklable object)
``e`` error ``(short message, traceback)``
===== ==========================================

The child sends exactly one terminal message (``r``/``e``) **if it survives long
enough to send one**.  A child that is killed outright (OOM, segfault, a
``terminate()`` from the parent) obviously cannot report anything, so the parent
does **not** wait for a message: when the queue goes empty *and* the process is
gone, :class:`~shkit.gui.workers.SubprocessRunner` reports
``子进程异常退出（exitcode=…）``.  That inference is the reason the parent must
never treat "no message" as success.

Windows uses the ``spawn`` start method, so the child re-imports this module and
the entry point must be importable and picklable -- hence the flat
:func:`run_job` taking a plain dict.
"""

from __future__ import annotations

import os
import re
import sys
import traceback

__all__ = ["run_job", "JOB_KINDS", "spawn_available", "run_in_child"]

#: Jobs the child knows how to run.  Keeping this a whitelist (rather than
#: pickling an arbitrary callable) means anything that reaches the child is
#: serialisable, and the failure mode of an unknown name is a clear message.
JOB_KINDS = ("analyze_series", "horizontal_grid", "horizontal_points")


def spawn_available() -> tuple:
    """``(ok, reason)`` -- can we start a child process at all?

    ``spawn`` needs to re-import ``__main__``, so a host that runs the GUI without
    an ``if __name__ == "__main__"`` guard (or a frozen bundle that did not opt in
    via ``multiprocessing.freeze_support()``) cannot use it.  We check rather than
    find out the hard way, and the caller falls back to the in-thread path **and
    says so** instead of silently running somewhere else.

    ``__main__`` 会被重新执行的**确切条件**（实测过，不是猜的）：

    * ``__main__.__file__`` 以 ``.exe`` 结尾（pip 的 console-script 启动器、
      PyInstaller 的冻结包）→ CPython 的 ``multiprocessing.get_preparation_data``
      **不会**把主模块路径交给子进程，所以不存在重复执行；
    * ``__main__`` 是包里的 ``__main__`` 模块（``python -m shkit.gui``）→ 子进程
      按模块名导入，顶层只有 ``from .app import main``，也没有副作用；
    * 其它情况（普通 ``.py`` 脚本）→ 子进程会把**整个脚本的顶层代码再执行一遍**。
      实测：没有保护的脚本跑一次，标记文件里出现 **2 个进程各 1 行**；带上保护就
      只有 1 行。GUI 宿主脚本被重跑一次 = 多开一个窗口 / 重复算一遍，所以这种情况
      必须拒绝并说明原因。
    """
    try:
        import multiprocessing as mp
    except Exception as exc:                                     # noqa: BLE001
        return False, f"无法导入 multiprocessing：{exc}"
    if "spawn" not in mp.get_all_start_methods():
        return False, "当前平台没有 spawn 启动方式"
    main_mod = __import__("__main__")
    src = getattr(main_mod, "__file__", None)
    if src is None:
        # interactive / embedded interpreter: spawn will re-import nothing useful
        return False, "主模块没有 __file__（交互式或嵌入式解释器），无法安全 fork/spawn"
    # .exe 启动器与冻结包：multiprocessing 不会重跑主模块（见上面的说明）
    if getattr(sys, "frozen", False) or str(src).lower().endswith(".exe"):
        return True, ""
    if getattr(getattr(main_mod, "__spec__", None), "name", None):
        # python -m <pkg>：子进程按模块名导入，顶层无副作用
        return True, ""
    try:
        with open(src, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return True, ""
    if not re.search(r"if\s+__name__\s*==", text):
        return False, (f"主模块 {os.path.basename(str(src))} 里没有 "
                       "if __name__ == '__main__' 保护：spawn 的子进程会把脚本顶层"
                       "代码**再执行一遍**（实测会多开一个窗口、把分析重复算一遍）")
    return True, ""


# ---------------------------------------------------------------------------
# child side
# ---------------------------------------------------------------------------
def run_job(kind: str, payload: dict, q) -> None:
    """Child entry point: run one job, stream progress, send one terminal message.

    Never raises: any exception becomes an ``e`` message, because a traceback in
    the child would otherwise be invisible to the parent.
    """
    try:
        if kind not in JOB_KINDS:
            q.put(("e", (f"未知任务 {kind!r}", "")))
            return

        def progress(msg, frac):
            try:
                q.put(("p", (str(msg), float(frac))))
            except Exception:                                    # noqa: BLE001
                pass

        if kind == "analyze_series":
            from ..series import analyze_series

            coeffs, rep = analyze_series(
                payload["lat"], payload["lon"], payload["values"],
                int(payload["nmax"]),
                method=payload.get("method", "quadrature"),
                rule=payload.get("rule", "auto"),
                field_unit=payload.get("field_unit", "scalar"),
                target_unit=payload.get("target_unit"),
                gaussian_km=float(payload.get("gaussian_km", 0.0) or 0.0),
                report_fit=bool(payload.get("report_fit", True)),
                times=payload.get("times"),
                longitude_fft=payload.get("longitude_fft", "auto"),
                progress=progress, cancel=lambda: False)
            q.put(("r", (coeffs, rep)))
        elif kind == "horizontal_grid":
            # 单位换算/逐阶因子已由**界面进程**算好（`gradient.canonical_scaled`），
            # 这里只剩纯数值的梯度综合 —— 线程路径与子进程路径输入逐位相同。
            from ..gradient import horizontal_grid

            rep = {}
            out = horizontal_grid(payload["lat"], payload["lon"],
                                  payload["C"], payload["S"],
                                  payload["degree_factors"],
                                  method=payload.get("method", "auto"),
                                  report=rep,
                                  time_chunk=payload.get("time_chunk"),
                                  progress=progress)
            out = dict(out)
            out["report"] = rep          # path/reason 要带回界面（别静默换算法）
            q.put(("r", out))
        else:                                                    # horizontal_points
            from ..gradient import horizontal_field

            out = horizontal_field(payload["lat"], payload["lon"],
                                   payload["C"], payload["S"],
                                   payload["degree_factors"],
                                   chunk=payload.get("chunk"),
                                   progress=progress)
            out = dict(out)
            out["report"] = {"path": "direct",
                             "reason": "散点：经度没有均匀轴，只能直接法"}
            q.put(("r", out))
    except Exception as exc:                                     # noqa: BLE001
        q.put(("e", (f"{type(exc).__name__}: {exc}", traceback.format_exc())))


def run_in_child(kind: str, payload: dict):
    """Start ``run_job`` in a fresh process and return ``(process, queue)``.

    Raises whatever ``multiprocessing`` raises when the child cannot be started;
    the caller decides whether to fall back.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=run_job, args=(kind, payload, q), daemon=True)
    p.start()
    return p, q
