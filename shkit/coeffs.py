# -*- coding: utf-8 -*-
"""
shkit.coeffs
============

Geodesy-convention spherical harmonic coefficients.

Convention (identical to the reference implementation in ``m2py`` /
``gridSHconvert``, and to SHTOOLS ``norm=1, csphase=1``):

.. math::

    f(\\theta,\\lambda)=\\sum_{n=0}^{N}\\sum_{m=0}^{n}
        \\bar P_{nm}(\\cos\\theta)\\,
        \\bigl[C_{nm}\\cos m\\lambda + S_{nm}\\sin m\\lambda\\bigr]

with the **4-pi normalised** associated Legendre functions

.. math::

    \\frac{1}{4\\pi}\\int_{\\Omega}\\bar P_{nm}^2(\\cos\\theta)
        \\cos^2(m\\lambda)\\;\\mathrm{d}\\Omega = 1

and **no Condon-Shortley phase** (``csphase = 1``).

Coefficients are stored as two arrays ``C`` and ``S`` of shape
``(nmax+1, nmax+1)`` (or ``(nmax+1, nmax+1, ntime)``); entries with ``m > n``
are identically zero.  ``S[:, 0]`` is identically zero by definition
(``sin(0*lambda) == 0``) and is always kept at zero.

Two flat layouts are supported for interchange:

``'matrix'`` (native)
    ``C[n, m]``, ``S[n, m]``.

``'triangle'`` (m2py / gridSHconvert compatible)
    ``[C; S]`` stacked, each of length ``(nmax+1)(nmax+2)/2``, ordered
    ``m`` outer / ``n`` inner, i.e. ``(0,0), (0,1), ..., (0,N),
    (1,1), (1,2), ...``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .timeaxis import TimeAxis

__all__ = ["SHCoeffs", "mn_index", "triangle_index", "triangle_order"]


def triangle_order(nmax: int):
    """Return ``(m_vec, n_vec)`` for the m2py triangle ordering (m outer)."""
    m_list, n_list = [], []
    for m in range(nmax + 1):
        for n in range(m, nmax + 1):
            m_list.append(m)
            n_list.append(n)
    return np.asarray(m_list, dtype=int), np.asarray(n_list, dtype=int)


def triangle_index(nmax: int):
    """Return ``(rows, cols)`` giving where each triangle entry sits in C/S."""
    m_vec, n_vec = triangle_order(nmax)
    return n_vec, m_vec


def mn_index(nmax: int):
    """Alias of :func:`triangle_order` (m outer, n inner)."""
    return triangle_order(nmax)


@dataclass
class SHCoeffs:
    """Spherical harmonic coefficients (4-pi normalised, no CS phase).

    Attributes
    ----------
    C, S : ndarray
        ``(nmax+1, nmax+1)`` or ``(nmax+1, nmax+1, ntime)``.  ``S[:, 0] == 0``.
    nmax : int
        Maximum degree actually resolved.
    meta : dict
        Free-form provenance (source file, weight rule, diagnostics, ...).
    times : TimeAxis, optional
        The time axis of the trailing dimension (see :mod:`shkit.timeaxis`).
        ``None`` means "no time information", which is the historical behaviour,
        so single-epoch code is unaffected.  Operations that combine two
        coefficient sets check that the axes agree.
    """

    C: np.ndarray
    S: np.ndarray
    meta: dict = field(default_factory=dict)
    times: Optional[TimeAxis] = None

    # ------------------------------------------------------------------ init
    def __post_init__(self) -> None:
        self.C = np.asarray(self.C, dtype=float)
        self.S = np.asarray(self.S, dtype=float)
        if self.C.shape != self.S.shape:
            raise ValueError(f"C{S.C.shape} and S{self.S.shape} must have the same shape")
        if self.C.ndim not in (2, 3):
            raise ValueError("C/S must be 2-D (nmax+1, nmax+1) or 3-D with a time axis")
        if self.C.shape[0] != self.C.shape[1]:
            raise ValueError("C/S must be square in (degree, order)")
        if self.times is not None:
            if not isinstance(self.times, TimeAxis):
                raise TypeError(
                    f"times 必须是 shkit.timeaxis.TimeAxis，得到 "
                    f"{type(self.times).__name__}（不要直接塞数组，用 TimeAxis.from_* 构造）")
            if len(self.times) != self.ntime:
                raise ValueError(
                    f"times 有 {len(self.times)} 个历元，但系数的时间轴是 {self.ntime}"
                    " —— 二者必须一致")
        # S[:, 0] is structurally zero (sin(0*lambda) == 0).  Index the ORDER
        # axis explicitly: for 3-D arrays (nmax+1, nmax+1, ntime) the trailing
        # axis is time, so a naive S[..., :, 0] would wipe the first time slice.
        if self.S.ndim == 2:
            self.S[:, 0] = 0.0
        else:
            self.S[:, 0, :] = 0.0

    # ------------------------------------------------------------- geometry
    @property
    def nmax(self) -> int:
        return self.C.shape[0] - 1

    @property
    def ntime(self) -> int:
        return 1 if self.C.ndim == 2 else self.C.shape[2]

    @property
    def ncoef(self) -> int:
        """Number of non-trivial real coefficients, ``(nmax+1)**2``."""
        return (self.nmax + 1) ** 2

    def shape(self):
        return self.C.shape

    def copy(self) -> "SHCoeffs":
        return SHCoeffs(self.C.copy(), self.S.copy(), dict(self.meta), self.times)

    # ------------------------------------------------------------------ time
    @property
    def has_time(self) -> bool:
        """True when a time axis is attached *and* it has one entry per epoch."""
        return self.times is not None and len(self.times) == self.ntime

    def with_times(self, times: Optional[TimeAxis]) -> "SHCoeffs":
        """Return a copy tagged with a time axis (validated against ``ntime``)."""
        return SHCoeffs(self.C, self.S, dict(self.meta), times)

    def time_slice(self, t: int) -> "SHCoeffs":
        """One epoch as a plain 2-D coefficient set (``times`` becomes ``None``)."""
        if self.C.ndim == 2:
            if t not in (0, -1):
                raise IndexError(f"this object holds a single time slice (asked {t})")
            return SHCoeffs(self.C, self.S, dict(self.meta), None)
        meta = dict(self.meta)
        if self.times is not None:
            meta["epoch"] = str(self.times.values[t])[:10]
            meta["epoch_index"] = int(t)
        return SHCoeffs(self.C[:, :, t], self.S[:, :, t], meta, None)

    def select_times(self, t0=None, t1=None) -> "SHCoeffs":
        """Sub-series inside the inclusive date window ``[t0, t1]``."""
        if self.times is None:
            raise ValueError("这个系数集没有时间轴，无法按时间切片（先挂上 times=）")
        mask, _ = self.times.slice(t0, t1)
        if not mask.any():
            raise ValueError(f"窗口 [{t0}, {t1}] 内没有历元")
        idx = np.nonzero(mask)[0]
        if self.C.ndim == 2:
            return self.copy()
        return SHCoeffs(self.C[:, :, idx], self.S[:, :, idx], dict(self.meta),
                        self.times.select(idx))

    def _combined_times(self, other: "SHCoeffs", op: str) -> Optional[TimeAxis]:
        """Common time axis for ``self op other``, or raise when they disagree.

        Exactly one side missing an axis is tolerated (legacy files have none)
        but recorded in ``meta['times_from']`` so it is visible rather than
        silent; two *different* axes are an error.
        """
        a, b = self.times, other.times
        if a is None and b is None:
            return None
        if a is None or b is None:
            return a if a is not None else b
        if len(a) != len(b) or not np.array_equal(a.values, b.values):
            raise ValueError(
                f"{op}: 两边的时间轴不一致，拒绝静默广播。\n"
                f"  左：{len(a)} 个历元，{a.meta.get('time_source')}，"
                f"{str(a.values[0])[:10] if len(a) else '-'} …\n"
                f"  右：{len(b)} 个历元，{b.meta.get('time_source')}，"
                f"{str(b.values[0])[:10] if len(b) else '-'} …\n"
                "请先用 select_times() 对齐，或明确用 with_times() 指定。")
        return a

    # ------------------------------------------------------------ accessors
    def matrix(self, time: int = 0):
        """Return ``(C, S)`` for one time slice as ``(nmax+1, nmax+1)`` arrays."""
        if self.C.ndim == 2:
            if time not in (0, -1):
                raise IndexError(f"this object holds a single time slice (asked {time})")
            return self.C, self.S
        return self.C[:, :, time], self.S[:, :, time]

    def flat_cs(self, time: int = 0):
        """Return ``(C_flat, S_flat)`` in the ``(L+1)**2`` ordering used by solvers.

        Ordering: cosine part for all ``(n, m)`` with ``m <= n`` (m outer, n
        inner), then the sine part for ``m >= 1``.  ``S[:, 0]`` is not stored.
        """
        C, S = self.matrix(time)
        m_vec, n_vec = triangle_order(self.nmax)
        cf = C[n_vec, m_vec]
        sel = m_vec >= 1
        sf = S[n_vec[sel], m_vec[sel]]
        return cf, sf

    @classmethod
    def from_flat_cs(cls, cf: np.ndarray, sf: np.ndarray, nmax: int,
                     meta: Optional[dict] = None) -> "SHCoeffs":
        m_vec, n_vec = triangle_order(nmax)
        sel = m_vec >= 1
        C = np.zeros((nmax + 1, nmax + 1))
        S = np.zeros((nmax + 1, nmax + 1))
        C[n_vec, m_vec] = cf
        S[n_vec[sel], m_vec[sel]] = sf
        return cls(C, S, dict(meta or {}))

    def to_triangle(self) -> np.ndarray:
        """``[C; S]`` stacked triangle vectors (m2py ``out_SHCS`` layout).

        Shape ``(2*NC, ntime)`` with ``NC = (nmax+1)(nmax+2)/2``.
        """
        C, S = self.C, self.S
        ntime = self.ntime
        if C.ndim == 2:
            C = C[:, :, None]
            S = S[:, :, None]
        m_vec, n_vec = triangle_order(self.nmax)
        NC = len(m_vec)
        out = np.zeros((2 * NC, ntime))
        for t in range(ntime):
            out[:NC, t] = C[n_vec, m_vec, t]
            out[NC:, t] = S[n_vec, m_vec, t]
        return out

    @classmethod
    def from_triangle(cls, arr: np.ndarray, nmax: int,
                      meta: Optional[dict] = None) -> "SHCoeffs":
        """Inverse of :meth:`to_triangle`."""
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        if arr.shape[0] == 2 * (nmax + 1) * (nmax + 2) // 2:
            pass
        elif arr.shape[1] in (2 * (nmax + 1) * (nmax + 2) // 2,):
            arr = arr.T                    # accept (ntime, 2*NC) too
        NC = (nmax + 1) * (nmax + 2) // 2
        if arr.shape[0] != 2 * NC:
            raise ValueError(f"expected {2*NC} rows (2*NC), got {arr.shape[0]}")
        m_vec, n_vec = triangle_order(nmax)
        ntime = arr.shape[1]
        C = np.zeros((nmax + 1, nmax + 1, ntime))
        S = np.zeros((nmax + 1, nmax + 1, ntime))
        for t in range(ntime):
            C[n_vec, m_vec, t] = arr[:NC, t]
            S[n_vec, m_vec, t] = arr[NC:, t]
        if ntime == 1:
            C = C[:, :, 0]
            S = S[:, :, 0]
        return cls(C, S, dict(meta or {}))

    # ------------------------------------------------------------- analysis
    def degree_rms(self) -> np.ndarray:
        """Per-degree (deg 0..nmax) RMS amplitude over all orders."""
        C, S = self.C, self.S
        out = np.zeros(self.nmax + 1)
        for n in range(self.nmax + 1):
            c = np.atleast_1d(C[n, :n + 1])
            s = np.atleast_1d(S[n, :n + 1])
            if C.ndim == 3:
                c = c.reshape(-1)
                s = s.reshape(-1)
            out[n] = np.sqrt(np.mean(np.concatenate([c, s]) ** 2))
        return out

    def power(self) -> np.ndarray:
        """Per-degree total power (sum of squared coefficients)."""
        C, S = self.C, self.S
        out = np.zeros(self.nmax + 1)
        for n in range(self.nmax + 1):
            out[n] = np.sum(C[n, :n + 1] ** 2) + np.sum(S[n, :n + 1] ** 2)
        return out

    def truncate(self, lmax: int) -> "SHCoeffs":
        """Return a copy truncated (or zero-padded) to degree ``lmax``."""
        L = max(lmax, 0)
        ntime = self.ntime
        shape = (L + 1, L + 1) if self.C.ndim == 2 else (L + 1, L + 1, ntime)
        C = np.zeros(shape)
        S = np.zeros(shape)
        n = min(L, self.nmax)
        if self.C.ndim == 2:
            C[:n + 1, :n + 1] = self.C[:n + 1, :n + 1]
            S[:n + 1, :n + 1] = self.S[:n + 1, :n + 1]
        else:
            C[:n + 1, :n + 1, :] = self.C[:n + 1, :n + 1, :]
            S[:n + 1, :n + 1, :] = self.S[:n + 1, :n + 1, :]
        meta = dict(self.meta)
        meta["truncated_from"] = self.nmax
        return SHCoeffs(C, S, meta, self.times)

    # ------------------------------------------------------------- units
    def to_ewh(self, love_numbers=None) -> "SHCoeffs":
        """Convert dimensionless geopotential coefficients to EWH coefficients.

        Equivalent to :func:`shkit.filters.to_ewh`.  The two are **not**
        interchangeable with a constant factor: the conversion multiplies degree
        ``n`` by ``A_n = R*rho_ave/(3*rho_w) * (2n+1)/(1+k_n)``, which already
        spans a factor of ~14 between degree 0 and degree 6.
        """
        from .filters import to_ewh as _to_ewh
        return _to_ewh(self, love_numbers=love_numbers)

    def to_geopotential(self, love_numbers=None) -> "SHCoeffs":
        """Convert EWH coefficients back to dimensionless geopotential ones."""
        from .filters import to_geopotential as _to_geopotential
        return _to_geopotential(self, love_numbers=love_numbers)

    @property
    def field_unit(self) -> str:
        """Physical meaning of these coefficients (see ``shkit.filters``)."""
        from .filters import field_unit as _fu
        return _fu(self)

    def with_unit(self, unit: str, **meta) -> "SHCoeffs":
        """Return a copy tagged with a physical meaning (metadata only)."""
        from .filters import with_field_unit
        return with_field_unit(self, unit, **meta)

    # ----------------------------------------------------------- arithmetic
    def __add__(self, other: "SHCoeffs") -> "SHCoeffs":
        times = self._combined_times(other, "相加")
        L = max(self.nmax, other.nmax)
        a, b = self.truncate(L), other.truncate(L)
        meta = {"op": "add"}
        if (self.times is None) != (other.times is None):
            meta["times_from"] = "一侧没有时间轴（已沿用另一侧）"
        return SHCoeffs(a.C + b.C, a.S + b.S, meta, times)

    def __sub__(self, other: "SHCoeffs") -> "SHCoeffs":
        times = self._combined_times(other, "相减")
        L = max(self.nmax, other.nmax)
        a, b = self.truncate(L), other.truncate(L)
        meta = {"op": "sub"}
        if (self.times is None) != (other.times is None):
            meta["times_from"] = "一侧没有时间轴（已沿用另一侧）"
        return SHCoeffs(a.C - b.C, a.S - b.S, meta, times)

    def __mul__(self, scalar: float) -> "SHCoeffs":
        return SHCoeffs(self.C * scalar, self.S * scalar, dict(self.meta), self.times)

    __rmul__ = __mul__

    # ------------------------------------------------------------- printing
    def __repr__(self) -> str:
        return (f"SHCoeffs(nmax={self.nmax}, ntime={self.ntime}, "
                f"ncoef={self.ncoef})")

    def summary(self) -> str:
        lines = [repr(self)]
        rms = self.degree_rms()
        nz = np.nonzero(rms)[0]
        if len(nz):
            lines.append(f"  degree RMS: n={nz[0]} -> {rms[nz[0]]:.6e} ... "
                         f"n={nz[-1]} -> {rms[nz[-1]]:.6e}")
        for k in ("weight_rule", "method", "weight_sum_ratio", "fit_rmse_rel"):
            if k in self.meta:
                lines.append(f"  {k}: {self.meta[k]}")
        return "\n".join(lines)
