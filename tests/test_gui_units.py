# -*- coding: utf-8 -*-
"""
GUI「输入是 / 输出为」的语义回归测试（真实窗口 + 真实工作线程 + 真实导出）。

这对下拉各选**一条公式**，中间是经典无量纲位系数：

    正变换（由「输入是」决定）：  C_nm = a_nm / f_u
    反变换（由「输出为」决定）：  场 = C_nm × f_t

导出的系数文件装的永远是 ``C_nm``。

所以本测试断言三件事：

  A. 改「输入是」**会改变导出的系数**：同一块网格声明成 geoid / EWH / σ /
     位系数，导出的 C_nm 互不相同，且逐阶比值精确等于 1/f_u
     （geoid 恒差 1/R；EWH 差 1/Aₙ，逐阶不同）。
  B. 改「输出为」**不改变导出的系数**（永远是 C_nm），只改变重建场的物理量；
     「不换算」= 用输入的 f_u 反算，正反抵消，重建精确回到原网格。
  C. 出错时清掉上一次结果，不会静默写出旧系数。

另加：预览栏必须把两条公式都写成具体数字。

Run:  python tests/test_gui_units.py
"""
import os
import shutil
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TMP = os.path.join(HERE, "_gui_units_tmp")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)

    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication([])

    # 分析失败时 GUI 弹**模态**对话框，在无人值守脚本里永久阻塞。
    seen_dialogs = []
    QMessageBox.critical = staticmethod(
        lambda *a, **k: seen_dialogs.append(str(a[-1]) if a else ""))

    from shkit import io as shio
    from shkit import units as shunits
    from shkit.gui.main_window import MainWindow

    win = MainWindow()
    win.show()

    def pump(n=30):
        for _ in range(n):
            app.processEvents()
            time.sleep(0.005)

    def analyse(timeout=180.0):
        win.run_analysis()
        w = win.analysis_worker
        t0 = time.time()
        while w is not None and w.isRunning() and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.01)
        pump(30)

    def export(tag):
        p = os.path.join(TMP, f"{tag}.sh")
        shio.write_coeffs(win.coeffs, p, layout="triangle")
        return shio.read_coeffs(p)

    def pick(unit):
        for i in range(win.cb_field_unit.count()):
            if win.cb_field_unit.itemData(i) == unit:
                return i
        raise AssertionError(f"「输入是」没有 {unit}")

    def pick_t(unit):
        for i in range(win.cb_target_unit.count()):
            if (win.cb_target_unit.itemData(i) or "") == unit:
                return i
        raise AssertionError(f"「输出为」没有 {unit!r}")

    pump()
    win.make_demo()
    win.sp_nmax.setValue(12)
    pump()

    UNITS = [win.cb_field_unit.itemData(i)
             for i in range(win.cb_field_unit.count())]
    UNITS_TEXT = [win.cb_field_unit.itemText(i)
                  for i in range(win.cb_field_unit.count())]
    TARGETS_DATA = [win.cb_target_unit.itemData(i) or ""
                    for i in range(win.cb_target_unit.count())]
    TARGETS_TEXT = [win.cb_target_unit.itemText(i)
                    for i in range(win.cb_target_unit.count())]
    check("「输入是」= 普通网格 / geoid / EWH / 径向形变（4 项，默认普通网格）",
          UNITS == ["geopotential", "geoid", "ewh", "radial_displacement"]
          and win.cb_field_unit.currentIndex() == 0,
          f"{UNITS_TEXT}（默认 {UNITS_TEXT[win.cb_field_unit.currentIndex()]}）")
    check("「输入是」不含标量场与面密度",
          "scalar" not in UNITS and "surface_density" not in UNITS,
          f"{UNITS}")
    check("「输出为」= 不换算 / EWH / geoid / 普通球谐系数 / 径向形变",
          TARGETS_DATA == ["", "ewh", "geoid", "geopotential",
                           "radial_displacement"],
          f"{TARGETS_TEXT}")
    check("「输出为」不含面密度",
          "surface_density" not in TARGETS_DATA, f"{TARGETS_DATA}")

    L = 12
    R = shunits.EARTH_RADIUS_M
    f_geo = shunits.forward_factors("geoid", L)
    f_ewh = shunits.forward_factors("ewh", L)

    # ---------------------------------------------------------------- A
    print("\n" + "=" * 78)
    print("  A. 改「输入是」→ 导出的系数应当跟着变（正变换公式）")
    print("=" * 78)
    idx_none = pick_t("")
    got, failed = {}, {}
    t0 = time.time()
    for u in UNITS:
        win.cb_field_unit.setCurrentIndex(pick(u))
        win.cb_target_unit.setCurrentIndex(idx_none)
        pump(10)
        seen_dialogs.clear()
        analyse()
        if win.coeffs is None:
            failed[u] = seen_dialogs[-1] if seen_dialogs else "无消息"
            print(f"       {u:<22s} 报错（已捕获）: "
                  f"{failed[u].splitlines()[0][:46]}")
            continue
        c = export(f"a_{u}")
        got[u] = c
        print(f"       {u:<22s} field_unit={shunits.field_unit(c):<13s} "
              f"C[2,1] = {c.C[2, 1]:+.12e}")

    base = got.get("geopotential")
    check("A1 「普通网格」声明：f_u = 1，导出就是该场自身的系数",
          base is not None, f"C[2,1] = {base.C[2, 1]:+.12e}" if base is not None else "失败")

    ok = True
    detail = []
    for u, f_u in (("geoid", f_geo), ("ewh", f_ewh)):
        c = got.get(u)
        if c is None:
            ok = False
            detail.append(f"{u}: 无结果")
            continue
        r = c.C[2, 1] * f_u[2] / base.C[2, 1]     # 应恒为 1
        ok &= abs(r - 1) < 1e-9
        detail.append(f"{u}: ×{f_u[2]:.4e} 归一 = {r:.12f}")
    check("A2 导出系数与声明严格吻合：geoid ÷R、EWH ÷A_n", ok, "; ".join(detail))

    if got.get("geoid") is not None and got.get("ewh") is not None:
        ratio = got["geoid"].C[2, 1] / got["ewh"].C[2, 1]
        want = f_ewh[2] / f_geo[2]
        check("A3 同一块网格：声明 geoid 与声明 EWH 的系数差 A₂/R",
              abs(ratio - want) < 1e-9,
              f"实测 {ratio:.6f} vs A₂/R = {want:.6f}（≈ {ratio:.1f} 倍）")
        # 逐阶都不同：A_n 不是常数
        per_deg = [got["ewh"].C[n, 1] / got["geoid"].C[n, 1]
                   for n in (1, 2, 3, 6)]
        check("A4 这个差别**逐阶不同**（A_n 不是常数）",
              max(per_deg) / min(per_deg) > 1.5,
              "ewh/geoid 逐阶 = " + ", ".join(f"{v:.4e}" for v in per_deg))
    else:
        check("A3 同一块网格声明 geoid 与 EWH 的系数差 A₂/R", False, "缺结果")
        check("A4 这个差别逐阶不同", False, "缺结果")

    check("A5 输入=径向形变时报错（h′₀ = 0，0 阶不可反推）",
          "radial_displacement" in failed
          and ("因子为 0" in failed["radial_displacement"]),
          failed.get("radial_displacement", "未失败").splitlines()[0][:56])

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 78)
    print("  B. 改「输出为」→ 导出系数不变，只换**输出场**；重建场恒与输入同量")
    print("=" * 78)
    win.cb_field_unit.setCurrentIndex(pick("ewh"))
    same_as_none = {}
    base_recon = None
    for t in ("ewh", "geoid", "geopotential"):
        win.cb_target_unit.setCurrentIndex(pick_t(t))
        pump(10)
        analyse()
        c = export(f"b_{t}")
        same_as_none[t] = (c is not None and got.get("ewh") is not None
                           and np.array_equal(c.C, got["ewh"].C)
                           and np.array_equal(c.S, got["ewh"].S))
        r = np.asarray(win.recon).ravel()
        if base_recon is None:
            base_recon = r.copy()
        recon_rms = float(np.sqrt(np.mean(r ** 2))) if win.recon is not None else float("nan")
        out_rms = (float(np.sqrt(np.mean(np.asarray(win.outfield) ** 2)))
                   if win.outfield is not None else float("nan"))
        print(f"       输入=ewh, 输出={t:<16s} 导出 C[2,1] = {c.C[2, 1]:+.12e}  "
              f"重建场 RMS = {recon_rms:.6e}  输出场 RMS = {out_rms:.6e}")

    check("B1 改「输出为」不改变导出的系数（永远导出 C_nm）",
          all(same_as_none.values()) and len(same_as_none) == 3,
          f"{same_as_none}")

    # 关键新语义：换「输出为」时**重建场一动不动**（它恒与输入同量）
    win.cb_target_unit.setCurrentIndex(pick_t("geoid"))
    pump(10)
    analyse()
    same_recon = np.array_equal(np.asarray(win.recon).ravel(), base_recon)
    check("B2 重建场与「输出为」无关（恒与输入同物理量）",
          same_recon,
          f"输出=ewh 与输出=geoid 两次的重建场逐位相同: {same_recon}")

    # 而输出场按 f_t 变化：geoid 输出场 = EWH 输出场 × (R/A_n)（逐阶）
    win.cb_target_unit.setCurrentIndex(pick_t("ewh"))
    pump(10)
    analyse()
    o_ewh = np.asarray(win.outfield).ravel()
    win.cb_target_unit.setCurrentIndex(pick_t("geoid"))
    pump(10)
    analyse()
    o_geo = np.asarray(win.outfield).ravel()
    k = float(np.median(o_geo / o_ewh))
    lo = min(f_geo[n] / f_ewh[n] for n in range(L + 1))
    hi = max(f_geo[n] / f_ewh[n] for n in range(L + 1))
    check("B3 输出场按 f_t 逐阶换算（geoid 输出场 = EWH 输出场 × R/A_n，逐阶）",
          lo < k < hi,
          f"实测等效倍率 {k:.6f} ∈ [{lo:.6f}, {hi:.6f}]（R/A₂ = "
          f"{f_geo[2] / f_ewh[2]:.6f}）")

    # 导出系数与输入系数域核对：C_nm × f_u 逐位还原成 a_nm
    from shkit.coeffs import SHCoeffs
    from shkit.synthesis import synthesis as _synth
    f_u = shunits.forward_factors("ewh", L)
    raw = SHCoeffs(win.coeffs.C * f_u[:, None], win.coeffs.S * f_u[:, None])

    # 「输出为 = 不换算」时 f_t = f_u，两个场是同一个
    win.cb_target_unit.setCurrentIndex(idx_none)
    pump(10)
    analyse()
    back = np.asarray(win.recon).ravel()
    check("B4 「输出为=不换算」时输出场就是重建场（f_t = f_u）",
          win.outfield is win.recon,
          f"同一个对象: {win.outfield is win.recon}；"
          f"重建场 = C_nm × f_u 展开后 RMS {np.sqrt(np.mean(back ** 2)):.4e}")

    # 重建场本身就等于「用该网格自身的系数 a_nm 综合」
    expect = np.asarray(_synth(win.dataset.lat, win.dataset.lon, raw)).ravel()
    rel = float(np.max(np.abs(back - expect)) / np.max(np.abs(expect)))
    check("B5 重建场 = C_nm × f_u 逐位还原成 a_nm 的重建（正反抵消）",
          rel < 1e-12, f"max|重建 - a_nm 重建| / max = {rel:.2e}")

    # 与「输出=geoid」相比：输出场等价于把导出的 C_nm **逐阶**乘 f_t
    win.cb_target_unit.setCurrentIndex(pick_t("geoid"))
    pump(10)
    analyse()
    geo = np.asarray(win.outfield).ravel()
    want_geo = np.asarray(_synth(win.dataset.lat, win.dataset.lon,
                                 SHCoeffs(win.coeffs.C * f_geo[:, None],
                                          win.coeffs.S * f_geo[:, None]))).ravel()
    rel_geo = float(np.max(np.abs(geo - want_geo)) / np.max(np.abs(want_geo)))
    # 逐阶因子确实是"逐阶"的：整条谱的等效倍率必须落在 min/max(f_t/f_u) 之间
    # （含 0 阶：A₀ = R·ρ̄/(3ρ_w) 最小，所以 R/A₀ 反而是最大的一档）
    lo = min(f_geo[n] / f_ewh[n] for n in range(L + 1))
    hi = max(f_geo[n] / f_ewh[n] for n in range(L + 1))
    eff = float(np.sqrt(np.mean(geo ** 2)) / np.sqrt(np.mean(back ** 2)))
    check("B6 输出场 = C_nm 逐阶乘 f_t（geoid 时 ×R，且逐阶而非常数）",
          rel_geo < 1e-12 and lo < eff < hi,
          f"逐阶吻合 {rel_geo:.2e}；等效倍率 {eff:.4e} 严格落在 "
          f"[{lo:.4e}, {hi:.4e}] 内（跨 {hi / lo:.1f} 倍，常数倍会贴到边界）")

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 78)
    print("  C. 分析失败必须清掉上一次结果")
    print("=" * 78)
    win.cb_field_unit.setCurrentIndex(pick("geopotential"))
    win.cb_target_unit.setCurrentIndex(idx_none)
    pump(10)
    analyse()
    check("C1 正常分析后有结果", win.coeffs is not None)
    n_before = win.coeff_table.rowCount()

    # 用「输入是 = 径向形变」触发失败（h′₀ = 0，demo 数据 0 阶不为 0）
    win.cb_field_unit.setCurrentIndex(pick("radial_displacement"))
    win.cb_target_unit.setCurrentIndex(idx_none)
    pump(10)
    seen_dialogs.clear()
    analyse()
    check("C2 失败后 coeffs/recon/outfield/report 全部清空",
          win.coeffs is None and win.recon is None and win.outfield is None
          and win.report is None,
          f"coeffs={win.coeffs!r}, recon={win.recon!r}, "
          f"outfield={win.outfield!r}")
    check("C3 失败后系数表被清空（不再展示旧结果）",
          win.coeff_table.rowCount() == 0, f"{n_before} → {win.coeff_table.rowCount()}")
    check("C4 失败弹窗给出真正的原因（异常行 + 消息首句，不是收尾句）",
          bool(seen_dialogs)
          and seen_dialogs[-1].splitlines()[0].startswith("ValueError:")
          and "把目标设为" not in seen_dialogs[-1].splitlines()[0],
          (seen_dialogs[-1].splitlines()[0][:60] if seen_dialogs else "无弹窗"))

    # ---------------------------------------------------------------- D
    print("\n" + "=" * 78)
    print("  D. 预览栏把两条公式写成具体数字")
    print("=" * 78)
    win.cb_field_unit.setCurrentIndex(pick("ewh"))
    win.cb_target_unit.setCurrentIndex(idx_none)
    pump(10)
    analyse()
    prev = win.lbl_unit_preview.text()
    print("       预览：")
    for ln in prev.splitlines():
        print("         " + ln)
    check("D1 预览给出正变换（÷ f_u）的实际数字",
          "正变换" in prev and "f_u" in prev and "C[" in prev,
          prev.replace("\n", " | ")[:80])
    check("D2 预览分别给出重建场（× f_u）与输出场（× f_t）的数字",
          "重建场" in prev and "输出场" in prev and "f_t" in prev
          and "与重建场相同" in prev,
          prev.splitlines()[-1][:80])

    # 换声明后预览里的 f_u 必须跟着变 —— 这就是"输入是 起作用"的直接证据
    win.cb_field_unit.setCurrentIndex(pick("geoid"))
    pump(10)
    analyse()
    prev_geo = win.lbl_unit_preview.text()
    check("D3 换「输入是」后预览里的 f_u 数值确实改变",
          prev_geo != prev and "6.378" in prev_geo.replace("e+06", "e+06"),
          prev_geo.splitlines()[0][:80])

    # 选一个真正的输出量：输出场那一行必须与重建场不同
    win.cb_field_unit.setCurrentIndex(pick("ewh"))
    win.cb_target_unit.setCurrentIndex(pick_t("geoid"))
    pump(10)
    analyse()
    prev_geo2 = win.lbl_unit_preview.text()
    for ln in prev_geo2.splitlines():
        print("         " + ln)
    check("D4 选了「输出为」后，输出场数字与重建场不同（且标明倍数）",
          "输出场" in prev_geo2 and "相对输入 ×" in prev_geo2
          and "与重建场相同" not in prev_geo2,
          prev_geo2.splitlines()[-1][:80])

    # 「输出为 = 普通球谐系数」时输出场就是 ×1（f_t = 1）
    win.cb_target_unit.setCurrentIndex(pick_t("geopotential"))
    pump(10)
    analyse()
    prev_pot = win.lbl_unit_preview.text()
    check("D5 「输出为 = 普通球谐系数」时 f_t = 1",
          "输出场" in prev_pot and "× f_t = 1" in prev_pot,
          prev_pot.splitlines()[-1][:80])

    # ---------------------------------------------------------------- E
    print("\n" + "=" * 78)
    print("  E. 多时次（批量/逐历元）路径的单位口径：重建场不许小 1e7 倍")
    print("=" * 78)
    # 用户报障：真实 mascon（输入是 = EWH）点「运行分析」后，地图上的「重建场」
    # 是 ~1e-7（输入 rms 22 cm）—— 根因是批量系数被标成「输入那一档」的系数，
    # 于是 synthesis 跳过反变换的 f_t 换算。这里用多时次 EWH 数据在 GUI 里走一遍。
    from shkit.coeffs import SHCoeffs as _SHC
    from shkit.synthesis import synthesis_grid as _sg
    from shkit.gui.dataset import Dataset as _DS

    _L = 6
    _rng = np.random.default_rng(5)
    _C = np.zeros((_L + 1, _L + 1))
    _S = np.zeros((_L + 1, _L + 1))
    for _n in range(1, _L + 1):
        for _m in range(_n + 1):
            _C[_n, _m] = _rng.standard_normal() / _n ** 2
            if _m:
                _S[_n, _m] = _rng.standard_normal() / _n ** 2
    _lat_m = np.arange(-80.0, 80.01, 10.0)
    _lon_m = np.arange(0.0, 360.0, 15.0)
    _ewh = np.asarray(_sg(_lat_m, _lon_m,
                          _SHC(_C, _S, {"field_unit": "geopotential"}),
                          target_unit="ewh"))
    _cube = np.concatenate([(_ewh * s)[:, :, None] for s in (1.0, 1.05, 0.95)],
                           axis=2)
    _ds_mt = _DS.from_grid("<多时次 EWH>", _lat_m, _lon_m, _cube,
                           {"time": ["2002-01-18", "2002-02-17", "2002-03-19"]})
    win.on_loaded(_ds_mt)
    win.sp_nmax.setValue(8)
    win.cb_field_unit.setCurrentIndex(pick("ewh"))
    win.cb_target_unit.setCurrentIndex(idx_none)
    win.map_field.setCurrentIndex(1)          # 重建场
    pump(30)
    win.run_analysis()                        # 多时次 → 一次解完整条序列
    _t0 = time.time()
    while time.time() - _t0 < 180:
        busy = [w for w in (win.analysis_worker, win.series_worker)
                if w is not None and w.isRunning()]
        if not busy:
            app.processEvents()
            busy = [w for w in (win.analysis_worker, win.series_worker)
                    if w is not None and w.isRunning()]
            if not busy:
                break
        app.processEvents()
        time.sleep(0.01)
    pump(30)
    _drawn = np.asarray(win.map_canvas._hover_data[3], dtype=float)
    _input = np.asarray(_cube[:, :, 0], dtype=float)
    _r_drawn = float(np.sqrt(np.nanmean(_drawn ** 2)))
    _r_input = float(np.sqrt(np.nanmean(_input ** 2)))
    print(f"       输入 rms={_r_input:.6g}  地图重建场 rms={_r_drawn:.6g}  "
          f"比值={_r_drawn / _r_input:.6f}")
    check("E1 多时次路径：系数标成位系数，输入声明单独记着",
          win.series_coeffs is not None
          and shunits.field_unit(win.series_coeffs) == "geopotential"
          and win.series_coeffs.meta.get("forward_from") == "ewh",
          f"field_unit={shunits.field_unit(win.series_coeffs)!r}, "
          f"forward_from={win.series_coeffs.meta.get('forward_from')!r}")
    check("★ E2 地图上的重建场与输入**同量级**（不是小 1e7 倍的系数）",
          0.5 < _r_drawn / _r_input < 1.5,
          f"比值 {_r_drawn / _r_input:.6f}（漏乘 f_t 时约为 1e-8）")
    _ref_recon = np.asarray(_sg(_lat_m, _lon_m,
                                win.series_coeffs.time_slice(0),
                                target_unit="ewh"), dtype=float)
    check("E3 地图画的就是该历元的重建场（逐值一致）",
          np.allclose(np.sort(_drawn.ravel()), np.sort(_ref_recon.ravel()),
                      rtol=1e-12, atol=0),
          f"shape={_drawn.shape} vs {_ref_recon.shape}")

    win.close()
    app.processEvents()
    shutil.rmtree(TMP, ignore_errors=True)

    n_ok = sum(RESULTS)
    print("\n" + "=" * 78)
    print(f"{n_ok}/{len(RESULTS)} checks passed")
    print("=" * 78)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
