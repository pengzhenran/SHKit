# -*- coding: utf-8 -*-
"""
shkit.gui.workers
=================

Background :class:`QThread` workers.

Every long-running step (file loading, analysis, round-trip self-test) runs off
the GUI thread and reports back through signals, so the window never freezes.
Cancellation is **cooperative**: the workers pass a ``cancel`` callable into
:func:`shkit.analysis.analysis`, which polls it inside the order-``m`` quadrature
sweep, the normal-equation assembly and the CG iterations.  NumPy work that is
already running cannot be interrupted, so pressing Stop may take a moment.
"""

from __future__ import annotations

import os
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..analysis import AnalysisCancelled, analysis
from ..synthesis import synthesis
from .. import io as shio
from .dataset import Dataset

__all__ = ["LoadWorker", "AnalysisWorker", "SeriesWorker", "ExportWorker",
           "SubprocessRunner"]


class LoadWorker(QThread):
    """Read a points or grid file into a :class:`Dataset`."""

    succeeded = Signal(object)          # Dataset
    failed = Signal(str)

    def __init__(self, path: str, kind: str = "auto", var=None, parent=None):
        super().__init__(parent)
        self.path = path
        self.kind = kind
        self.var = var

    def run(self):  # noqa: D102
        try:
            ds = _load(self.path, self.kind, self.var)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.succeeded.emit(ds)


def _load(path: str, kind: str = "auto", var=None) -> Dataset:
    """Dispatch to the right reader and wrap the result in a Dataset."""
    lower = str(path).lower()
    is_grid_like = lower.endswith((".nc", ".grd"))
    if kind == "auto":
        kind = "grid" if is_grid_like else "points"

    if kind == "grid":
        # 多时次 nc 用**懒加载**：首屏只读第 1 个时次（实测 112 MB/256 时次的
        # mascon：3.5 s → ~1.2 s），其余由界面调 start_prefetch() 在后台按块补。
        # 不适用的文件会自动回退到整块读，理由写在 meta['lazy_fallback']。
        lat_vec, lon_vec, grid, meta = shio.read_grid(path, var=var,
                                                      lazy_time=True)
        return Dataset.from_grid(path, lat_vec, lon_vec, grid, meta)

    lat, lon, values, meta = shio.read_points(path)
    return Dataset.from_points(path, lat, lon, values, meta)


class SeriesWorker(QThread):
    """Batch-analyse every epoch of a multi-time dataset (D5).

    Uses :func:`shkit.series.analyze_series`, i.e. **one** call for the whole
    series, so the integration elements and the Gram diagnostic are computed once
    (that is the A2 optimisation, and it is what makes the "solve all epochs"
    button usable at all).  The per-epoch residual table comes from
    ``report_fit=True``.
    """

    progressed = Signal(str, float)
    # SHCoeffs, SeriesReport
    succeeded = Signal(object, object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, dataset: Dataset, params: dict, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.params = dict(params)
        self._cancel_flag = False

    def cancel(self):
        self._cancel_flag = True

    def _cancelled(self) -> bool:
        return self._cancel_flag

    def run(self):  # noqa: D102
        ds, p = self.dataset, self.params
        try:
            from ..series import analyze_series

            if not ds.has_time():
                raise ValueError(
                    "逐历元诊断需要多时次数据（ntime > 1）；"
                    "单时次数据没有「逐历元」可言。")
            kwargs = dict(
                method=p.get("method", "quadrature"),
                rule=p.get("rule", "auto"),
                field_unit=p.get("field_unit", "scalar"),
                target_unit=p.get("target_unit"),
                # 高斯平滑必须走**系数**（与单历元 AnalysisWorker 同一口径），
                # 否则「运行分析」平滑了、批量分析没平滑，两条路的结果不一致。
                gaussian_km=float(p.get("gaussian_km", 0.0) or 0.0),
                report_fit=True,                 # 要逐历元残差表
                times=ds.time_axis(),
                progress=lambda msg, frac: self.progressed.emit(msg, frac),
                cancel=self._cancelled,
                longitude_fft=p.get("longitude_fft", "auto"),
            )
            nlon = int(np.unique(ds.lon).size) if ds.kind == "grid" else None
            if nlon is not None and kwargs["rule"] == "dh":
                kwargs.setdefault("weights_kw", {"nlon": nlon})
            # 逐历元分析要**全部**时次：懒加载的立方体可能还在后台补，先把补齐
            # 进度报出去（否则界面只看到"开始分析"然后干等几秒）。
            if ds.lazy_loader() is not None:
                ld = ds.lazy_loader()
                if ld.n_filled < ld.ntime:
                    self.progressed.emit(
                        f"读取全部时次 {ld.n_filled}/{ld.ntime}…", 0.01)
                ds.ensure_all_epochs(progress=lambda d, n: self.progressed.emit(
                    f"读取全部时次 {d}/{n}…", 0.01 + 0.09 * d / max(n, 1)))
            coeffs, rep = analyze_series(ds.lat, ds.lon, ds.all_values(),
                                         int(p["nmax"]), **kwargs)
            if self._cancelled():
                self.cancelled.emit()
                return
            self.succeeded.emit(coeffs, rep)
        except Exception as exc:                                 # noqa: BLE001
            if self._cancelled():
                self.cancelled.emit()
            else:
                import traceback
                self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ExportWorker(QThread):
    """Write one file per epoch, with **pause and stop** (D10).

    Per-epoch files are what people actually hand to other software, and a
    200-epoch export takes long enough that "cancel" alone is not enough -- hence
    pause.  The loop writes file ``k`` only after passing the pause gate, so a
    paused export has written exactly the epochs up to the pause, never a torn
    partial set (only whole epochs are ever on disk).
    """

    progressed = Signal(str, float)
    # 写出的文件数、输出目录
    succeeded = Signal(int, str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, path_of_epoch, out_dir: str, prefix: str,
                 i0: int, i1: int, *, ext: str = ".nc", var: str = "value",
                 meta=None, parent=None, label_of=None):
        super().__init__(parent)
        self._write = path_of_epoch        # (k) -> Path，实际写盘在这里发生
        self.out_dir = str(out_dir)
        self.prefix = str(prefix)
        self.i0, self.i1 = int(i0), int(i1)
        self.ext = ext
        self.var = var
        self.meta = dict(meta or {})
        #: ``(k) -> str``：进度条上"第 k 个时次 → ?"里那个"?"。默认是文件名；一个历元
        #: 写好几个文件时（水平形变会写 u_N/u_E/|u_h|）由调用方给一句更准确的话。
        self.label_of = label_of
        self.written = 0
        self._paused = False
        self._cancel_flag = False

    # -------------------------------------------------------------- control
    def set_paused(self, on: bool):
        self._paused = bool(on)

    def cancel(self):
        self._cancel_flag = True

    def file_of(self, k: int) -> str:
        return os.path.join(self.out_dir, f"{self.prefix}_{k + 1:04d}{self.ext}")

    def _wait_if_paused(self) -> bool:
        """Block while paused; return True when the user asked to stop."""
        while self._paused and not self._cancel_flag:
            self.msleep(50)
        return self._cancel_flag

    # ------------------------------------------------------------------ run
    def run(self):  # noqa: D102
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            total = self.i1 - self.i0 + 1
            for n, k in enumerate(range(self.i0, self.i1 + 1)):
                if self._wait_if_paused():
                    self.cancelled.emit()
                    return
                out = self.file_of(k)
                self._write(k, out)
                self.written += 1
                desc = (self.label_of(k) if self.label_of is not None
                        else os.path.basename(out))
                self.progressed.emit(f"第 {k + 1} 个时次 → {desc}",
                                     (n + 1) / total)
            self.succeeded.emit(self.written, self.out_dir)
        except Exception as exc:                                 # noqa: BLE001
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class HorizontalWorker(QThread):
    """水平形变（球面梯度）放到**界面线程之外**算。

    为什么需要它：位系数的球面梯度不是逐阶乘法，规则整圈经度网格可以走经度 FFT
    （实测真实 mascon 720×1440、nmax=60：**0.08 s/历元**），但**散点**只能直接扫，
    同样规模要 **79 s/历元**（510 倍）。以前这一步是同步跑在界面线程里的 —— 散点上
    一点「计算水平形变」，整个窗口就冻住，没有进度也不能取消。现在统一走这里：
    进度按"阶 m / 点块"报，取消是协作式的（抛 ``RuntimeError("cancelled")``）。
    """

    progressed = Signal(str, float)
    succeeded = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, kind: str, lat, lon, C, S, degree_factors, *,
                 method: str = "auto", path: str = "", reason: str = "",
                 chunk=None, parent=None):
        super().__init__(parent)
        self.kind = str(kind)                 # 'grid' | 'points'
        self.lat, self.lon = lat, lon
        self.C, self.S = C, S
        self.degree_factors = degree_factors
        self.method = method
        self.path, self.reason = path, reason
        self.chunk = chunk
        self._cancel_flag = False

    def cancel(self):
        self._cancel_flag = True

    def _cancelled(self) -> bool:
        return self._cancel_flag

    def run(self):  # noqa: D102
        try:
            from ..gradient import horizontal_field, horizontal_grid

            report = {"path": self.path, "reason": self.reason}
            if self.kind == "grid":
                # chunk 不传：FFT 路径按 time_chunk 分块，直接路径一次扫完（与原行为一致）
                out = horizontal_grid(self.lat, self.lon, self.C, self.S,
                                      self.degree_factors, method=self.method,
                                      report=report,
                                      progress=lambda m, f: self.progressed.emit(m, f),
                                      cancel=self._cancelled)
            else:
                out = horizontal_field(self.lat, self.lon, self.C, self.S,
                                       self.degree_factors, chunk=self.chunk,
                                       progress=lambda m, f: self.progressed.emit(m, f),
                                       cancel=self._cancelled)
                report.update(path="direct",
                              reason="散点：经度没有均匀轴，只能直接法")
            if self._cancelled():
                self.cancelled.emit()
                return
            res = dict(out)
            res["report"] = report
            self.succeeded.emit(res)
        except RuntimeError as exc:
            if str(exc) == "cancelled" or self._cancelled():
                self.cancelled.emit()
            else:
                self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
        except Exception:                                        # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class SubprocessRunner(QThread):
    """Drive :mod:`shkit.gui.subproc` and translate it into Qt signals (D12).

    A crash in the child (segfault, OOM kill, ``os._exit``) surfaces as
    ``failed("子进程异常退出 …")`` while this window keeps running -- which is the
    entire point of paying for a process instead of a thread.

    The queue is drained **before** ``join()``: a child that fills a pipe buffer
    and blocks in ``put`` would deadlock a parent that joins first.  The child
    always sends one terminal message, so "the process ended and no result
    arrived" is itself the crash verdict.
    """

    progressed = Signal(str, float)
    # Two objects, exactly like SeriesWorker.succeeded -- a one-argument signal
    # connected to a two-argument slot is dropped **silently** by Qt, which is
    # how "the child succeeded but the table stayed empty" happens.
    succeeded = Signal(object, object)
    failed = Signal(str)
    cancelled = Signal()

    POLL_MS = 20

    def __init__(self, kind: str, payload: dict, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.payload = payload
        self._proc = None
        self._q = None
        self._cancel_flag = False
        self._sentinel_seen = None

    def cancel(self):
        self._cancel_flag = True

    def _terminate(self):
        try:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)
        except Exception:                                        # noqa: BLE001
            pass

    def run(self):  # noqa: D102
        from . import subproc

        try:
            self._proc, self._q = subproc.run_in_child(self.kind, self.payload)
        except Exception as exc:                                 # noqa: BLE001
            self.failed.emit(f"无法启动子进程：{type(exc).__name__}: {exc}")
            return
        import queue as _queue

        while True:
            if self._cancel_flag:
                self._terminate()
                self.cancelled.emit()
                return
            try:
                msg = self._q.get(timeout=self.POLL_MS / 1000.0)
            except _queue.Empty:
                if not self._proc.is_alive():
                    # 进程没了：先把队列里剩下的消息读完，再判定
                    try:
                        while True:
                            msg = self._q.get_nowait()
                            self._handle(msg)
                            if self._sentinel_seen:
                                return
                    except _queue.Empty:
                        pass
                    code = self._proc.exitcode
                    self.failed.emit(
                        f"子进程异常退出（exitcode={code}），没有返回结果。\n"
                        "界面本身不受影响：可以直接重试，或改用界面内计算。\n"
                        "常见原因：原生库崩溃（BLAS/OpenMP）、内存不足被系统杀掉。")
                    return
                continue
            self._handle(msg)
            if self._sentinel_seen:
                return

    def _handle(self, msg):
        try:
            kind, body = msg
        except Exception:                                        # noqa: BLE001
            # 队列里出现了不是 (kind, body) 的东西：**必须**在这里报出来。
            # 若让异常逃出 run()，QThread 会静默结束，父进程既收不到成功也收不到
            # 失败 —— 那正是最难查的一类故障（界面等着，什么提示都没有）。
            self._sentinel_seen = "unknown"
            self.failed.emit(f"子进程发来无法解析的消息：{msg!r}")
            return
        if kind == "p":
            self.progressed.emit(body[0], body[1])
        elif kind == "r":
            self._sentinel_seen = "ok"
            try:
                if self._proc is not None:
                    self._proc.join(timeout=5)
            except Exception:                                    # noqa: BLE001
                pass
            res = body
            if isinstance(res, tuple) and len(res) == 2:
                self.succeeded.emit(res[0], res[1])
            else:
                self.succeeded.emit(res, None)
        elif kind == "e":
            self._sentinel_seen = "err"
            self.failed.emit(f"{body[0]}\n\n{body[1]}")
        else:
            self._sentinel_seen = "unknown"
            self.failed.emit(f"子进程发来未知消息：{msg!r}")


class AnalysisWorker(QThread):
    """Run :func:`shkit.analysis.analysis` (and the reconstruction) in a thread."""

    progressed = Signal(str, float)
    # SHCoeffs, AnalysisReport, 重建场（与输入同量）, 输出场（按「输出为」）
    succeeded = Signal(object, object, object, object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, dataset: Dataset, params: dict, time_index: int = 0,
                 make_reconstruction: bool = True, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.params = dict(params)
        self.time_index = int(time_index)
        self.make_reconstruction = bool(make_reconstruction)
        self._cancel_flag = False

    # -------------------------------------------------------------- control
    def cancel(self):
        self._cancel_flag = True

    def _cancelled(self) -> bool:
        return self._cancel_flag

    # ------------------------------------------------------------------ run
    def run(self):  # noqa: D102
        """正变换，然后算**两个**反变换场。

        * 导出的系数 = 经典无量纲位系数 ``C_nm``（由「输入是」选的正变换公式得到）；
        * **重建场** = ``C_nm × f_u``，用**输入那一档**的公式反算 —— 它与输入网格
          同物理量，所以可以直接逐点比对（差值图、往返校验都看它）；
        * **输出场** = ``C_nm × f_t``，用「输出为」选的公式反算 —— 它是"另一个
          物理量的场"。

        「输出为 = 不换算」时 ``f_t = f_u``，两个场是同一个，只算一次。
        """
        ds, p = self.dataset, self.params
        try:
            f = ds.value_slice(self.time_index)
            lat = ds.lat
            lon = ds.lon
            nlon = int(np.unique(lon).size) if ds.kind == "grid" else None

            kwargs = dict(
                method=p.get("method", "auto"),
                rule=p.get("rule", "auto"),
                niter=int(p.get("niter", 0)),
                tau=p.get("tau"),
                normalise=p.get("normalise", "auto"),
                reg=p.get("reg"),
                alpha=p.get("alpha"),
                reg_power=float(p.get("reg_power", 2.0)),
                field_unit=p.get("field_unit", "scalar"),
                progress=lambda msg, frac: self.progressed.emit(msg, frac),
                cancel=self._cancelled,
            )
            if nlon is not None and kwargs["rule"] == "dh":
                kwargs["weights_kw"] = {"nlon": nlon}
            # 「输入是」= 正变换公式：C_nm = a_nm / f_u，所以导出的系数会随它变。
            # 「输出为」= 反变换公式，作用于下面的输出场。
            coeffs, report = analysis(lat, lon, f, int(p["nmax"]),
                                      target_unit=p.get("target_unit"),
                                      **kwargs)

            # 高斯平滑：直接乘到系数上，只做一次。
            # 这样导出的系数、逐阶谱、系数表、重建场/输出场都是**同一个**平滑结果；
            # 若改成只在合成时施加（旧做法），导出的系数就不是平滑过的，用户改半径
            # 会发现"设置了没效果"；两处都施加则会乘两次。
            g_km = float(p.get("gaussian_km", 0.0) or 0.0)
            if g_km > 0:
                from ..filters import apply_gaussian, gaussian_coefficients
                self.progressed.emit(f"高斯平滑 {g_km:g} km", 0.94)
                W = gaussian_coefficients(g_km, coeffs.nmax)
                coeffs = apply_gaussian(coeffs, g_km)
                report.meta["gaussian_km"] = g_km
                report.meta["gaussian_W0"] = float(W[0])
                report.meta["gaussian_Wnmax"] = float(W[-1])
                report.add_warning(
                    f"高斯平滑：半径 {g_km:g} km（0.5 幅度半宽）。"
                    f"W(0)={W[0]:.6f}，W({coeffs.nmax})={W[-1]:.6g}；"
                    "系数已逐阶乘 W 保存，重建场不再重复施加。")

            recon = outfield = None
            if self.make_reconstruction:
                u = p.get("field_unit") or "scalar"
                t = p.get("target_unit") or u
                self.progressed.emit("重建场（与输入同量）", 0.96)
                recon = self._reconstruct(coeffs, u)
                if t == u:
                    outfield = recon      # 两档相同：同一个场，不用重算
                else:
                    self.progressed.emit("输出场（按「输出为」）", 0.98)
                    outfield = self._reconstruct(coeffs, t)
            self.succeeded.emit(coeffs, report, recon, outfield)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

    # ------------------------------------------------------------ synthesis
    def _reconstruct(self, coeffs, target: str):
        """Evaluate the solution back on the original sampling geometry.

        ``coeffs`` are the canonical dimensionless geopotential coefficients
        (正变换 already applied by ``analysis(field_unit=…)``).  ``target``
        selects the **inverse formula**::

            场 = C_nm × f_target

        * ``target = 「输入是」``（``f_t = f_u``）→ 反变换正好抵消正变换，
          得到的就是**重建场**，与输入网格同物理量、可直接比对；
        * 其它 ``target`` → 得到「输出为」选的物理量对应的**输出场**。

        **高斯平滑不在这里做**：``coeffs`` 已经在 :meth:`run` 里乘过 W 了，
        再传 ``gaussian_km`` 会乘两次。所以 :func:`synthesis` 这里只做单位反变换。
        """
        ds, _p = self.dataset, self.params
        vals = synthesis(
            ds.lat, ds.lon, coeffs,
            target_unit=target or "scalar",
            chunk=100_000,
        )
        arr = np.asarray(vals)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if ds.kind == "grid":
            return arr.reshape(ds.lat_vec.size, ds.lon_vec.size)
        return arr.ravel()
