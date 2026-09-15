# -*- coding: utf-8 -*-
"""
fix_utf8_codepage_damage.py — undo "UTF-8 file read through the ANSI codepage".

The accident
------------
``Get-Content -Raw`` in **Windows PowerShell 5.1** decodes with the ANSI code
page, not UTF-8.  Reading a UTF-8 source file and writing it back therefore
re-encodes every non-ASCII character:

* valid CP936 byte pairs become mojibake (``求积`` -> ``姹傜Н``) -- **invertible**;
* bytes the decoder cannot map become ``?`` or private-use characters -- the two
  bytes are **lost**, and everything after them is shifted, so a naive
  ``gb18030`` round trip leaves ``U+FFFD`` holes.

Recipe
------
1. ``damaged.encode('gb18030').decode('utf-8')`` restores everything invertible;
2. lines that still contain ``U+FFFD`` need their text restored from an oracle;
3. **use a compiled ``.pyc`` of the pre-damage file as that oracle** -- it holds
   every string constant and docstring exactly.  This tool reports which lines
   are holed; patch them (they are almost always string literals, because a lost
   byte pair inside a comment is harmless but inside a literal breaks the quote);
4. verify by recompiling and comparing the **complete** multiset of string
   constants against the ``.pyc`` -- "it looks right" is not verification.

Usage::

    python tools/fix_utf8_codepage_damage.py path/to/file.py [--pyc path.pyc] [--write]

Without ``--write`` it only reports.  ``--pyc`` defaults to the matching file in
``__pycache__``, which must be **older** than the damage (check the timestamp).
"""
from __future__ import annotations

import argparse
import glob
import marshal
import os
import re
import sys
import types

BAD = re.compile("[\ufffd?]")


def _code_from_pyc(path: str):
    with open(path, "rb") as fh:
        magic = fh.read(4)
        fh.read(12)                      # flags + mtime + size (PEP 552)
        return magic, marshal.load(fh)


def _strings(code) -> set:
    out = set()

    def walk(co):
        for c in co.co_consts:
            if isinstance(c, str) and c.strip():
                out.add(c)
            elif isinstance(c, types.CodeType):
                walk(c)

    walk(code)
    return out


def _compile_text(text: str, filename: str):
    """Compile source text the way ``py_compile`` does -- **not** the way
    ``compile()`` does by default.

    ``compile()`` inherits the ``__future__`` flags of the *calling* module
    unless ``dont_inherit=True``.  This tool starts with
    ``from __future__ import annotations``, so a bare ``compile()`` here compiles
    the target with PEP 563 in force while the ``.pyc`` oracle was compiled
    without it.  The two then disagree on compiler-generated constants (for
    example the ``'return'`` key of a function's ``__annotations__``), and the
    safety gate below reports a **healthy** file as damaged.  Measured on
    ``tests/validate_lonfft.py``: 121 vs 122 constants, single difference
    ``'return'`` -- and every module in this package uses that future import, so
    the gate was wrong for all of them.

    ``dont_inherit=True`` makes the comparison apples-to-apples.
    """
    import ast
    return compile(ast.parse(text), filename, "exec", dont_inherit=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("source")
    ap.add_argument("--pyc", default=None,
                    help="pre-damage .pyc（默认到 __pycache__ 里找）")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--patch", default=None, metavar="JSON",
                    help="可选：{行号: 正确内容} 的 JSON，用于补 U+FFFD 行")
    args = ap.parse_args(argv)

    src = args.source
    with open(src, encoding="utf-8") as fh:
        damaged = fh.read()

    pyc = args.pyc
    if pyc is None:
        stem = os.path.splitext(os.path.basename(src))[0]
        cands = sorted(glob.glob(os.path.join(os.path.dirname(src) or ".",
                                              "__pycache__", stem + ".*.pyc")))
        pyc = cands[-1] if cands else None
    oracle = None
    if pyc and os.path.exists(pyc):
        try:
            _, oracle = _code_from_pyc(pyc)
        except Exception as exc:                              # noqa: BLE001
            print(f"（读不了 {pyc}: {exc}）")

    # ---- SAFETY GATE ------------------------------------------------------
    # Running the transform on a healthy file silently destroys it, and the
    # damaged/healthy cases look alike from the holes alone.  So: if the input
    # ALREADY matches the oracle, it is not damaged -- stop.
    oracle_verdict = None
    if oracle is not None:
        try:
            cur = _compile_text(damaged, src)
            if _strings(cur) == _strings(oracle):
                print("输入已经与 .pyc 一致 —— 文件是好的，不做任何修改。")
                return 0
            oracle_verdict = sorted(_strings(oracle) - _strings(cur))
        except SyntaxError:
            pass                       # a syntax error is itself damage evidence
    else:
        print("⚠ 没有 .pyc 可对照：无法先证明文件已损坏。"
              "请显式传 --pyc，否则本工具拒绝改写。")
        if args.write:
            return 2

    # A mismatch here is NOT a damage verdict: any edit made after the .pyc was
    # written produces one.  Say so plainly, and refuse to write -- otherwise a
    # healthy file gets "repaired" into mojibake by a helpful-looking report.
    if oracle_verdict:
        print(f"⚠ 无法判定：源码与 .pyc 的字符串常量不一致"
              f"（少 {len(oracle_verdict)} 个，例如 {oracle_verdict[:3]}）。\n"
              "   常见原因只是**源码在 .pyc 之后被改过**，与编码损坏无关。\n"
              "   下面的“损坏”报告是在“假设已损坏”的前提下生成的，**不可信**；\n"
              "   请用与源码同时期的 .pyc，或先确认文件确实读不出来。")
        if args.write:
            print("   --write 已拒绝。")
            return 2

    text = damaged.encode("gb18030").decode("utf-8", errors="replace")
    n_holes = text.count("\ufffd")
    print(f"（假设已损坏时的修复结果，仅供参考）"
          f"读入 {len(damaged)} 字符，恢复后仍有 {n_holes} 个 U+FFFD")
    lines = text.splitlines()
    holed = [i + 1 for i, ln in enumerate(lines) if BAD.search(ln)]
    if holed:
        print(f"需要人工/查表修补的行（{len(holed)}）：")
        for i in holed:
            print(f"  L{i}: {lines[i-1].strip()[:92]}")

    if args.patch:
        import json
        patch = {int(k): v for k, v in json.loads(args.patch).items()}
        for ln, content in patch.items():
            old = lines[ln - 1]
            indent = old[:len(old) - len(old.lstrip())]
            lines[ln - 1] = indent + content
        text = "\n".join(lines) + "\n"
        print(f"已应用 {len(patch)} 行补丁，剩余 U+FFFD: {text.count(chr(0xFFFD))}")

    # ---- verification against the oracle -----------------------------------
    ok = True
    if oracle is not None:
        try:
            new = _compile_text(text, src)
            a, b = _strings(new), _strings(oracle)
            print(f"\n对照 {os.path.basename(pyc)}："
                  f"常量 现在 {len(a)} / 原版 {len(b)}")
            if a == b:
                print("★ 字符串常量完全一致")
            else:
                ok = False
                print(f"★ 不一致：多了 {len(a - b)}，少了 {len(b - a)}")
                for c in list(a - b)[:5]:
                    print("   多:", repr(c[:70]))
                for c in list(b - a)[:5]:
                    print("   少:", repr(c[:70]))
        except SyntaxError as exc:
            ok = False
            print(f"\n语法错误 L{exc.lineno}: {exc.msg}")
            print("  ->", repr(lines[(exc.lineno or 1) - 1]))
    else:
        print("\n没有 .pyc 可对照 —— 只能靠语法检查，请格外小心")

    if ok and args.write:
        with open(src, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        print("\n已写回", src)
    elif not args.write:
        print("\n（未写回；加 --write 生效）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
