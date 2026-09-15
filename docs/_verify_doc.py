# -*- coding: utf-8 -*-
"""Verify the generated document: local-only refs, all images present, parses."""
import io
import os
import re
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "SHKit方法总结.html")
FIGS = os.path.join(HERE, "SHKit方法总结_figs")
html = io.open(DOC, encoding="utf-8").read()
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


# ---------------------------------------------------------------- references
refs = re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', html)
check("no remote src/href", not any(r.startswith(("http://", "https://", "//"))
                                   for r in refs),
      f"{len(refs)} refs, all local" if not any(
          r.startswith(("http://", "https://", "//")) for r in refs)
      else "REMOTE FOUND")
check("no <link>/<script>/@import", not re.search(
    r"<link\b|<script\b|@import", html, re.I), "no external CSS/JS")
check("no data: URIs", "data:" not in html, "")

imgs = re.findall(r'<img\s+src="([^"]+)"', html)
missing = [s for s in imgs if not os.path.isfile(os.path.join(HERE, s))]
check("every <img src> exists on disk", not missing,
      f"{len(imgs)} images, missing: {missing or 'none'}")

for s in imgs:
    p = os.path.join(HERE, s)
    check(f"  exists: {os.path.basename(s)}", os.path.isfile(p),
          f"{os.path.getsize(p)/1024:.0f} kB" if os.path.isfile(p) else "MISSING")

# ------------------------------------------------------------------- parsing
class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []
        self.counts = {}
        self.void = {"img", "br", "hr", "meta", "input", "link", "area", "base",
                     "col", "embed", "source", "track", "wbr"}
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.counts[tag] = self.counts.get(tag, 0) + 1
        self.tags.append(tag)
        if tag not in self.void:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}>")
        elif self.stack[-1] == tag:
            self.stack.pop()
        else:
            if tag in self.stack:
                while self.stack and self.stack[-1] != tag:
                    self.errors.append(f"unclosed <{self.stack.pop()}>")
                self.stack.pop()
            else:
                self.errors.append(f"unmatched </{tag}>")


p = P()
p.feed(html)
p.close()
check("HTML parses with balanced tags",
      not p.errors and not p.stack,
      f"errors={p.errors[:4]}, unclosed={p.stack[:4]}")

check("has <!DOCTYPE html>", html.lstrip().startswith("<!DOCTYPE html>"), "")
check("declares UTF-8", 'charset="utf-8"' in html, "")
check("has embedded <style>", "<style>" in html and "</style>" in html, "")
check("no <style src> / external css", "stylesheet" not in html.lower(), "")

# ------------------------------------------------------------------- anchors
ids = set(re.findall(r'\bid="([^"]+)"', html))
anchors = re.findall(r'href="#([^"]+)"', html)
bad = [a for a in anchors if a not in ids]
check("every TOC anchor resolves", not bad, f"{len(anchors)} links, bad={bad or 'none'}")

# ------------------------------------------------------------------- content
n_sections = len(re.findall(r"<section\b", html))
check("section count", n_sections == 13, f"{n_sections} sections")
check("figure/table density", html.count("<table") >= 18 and html.count("<figure") >= 20,
      f"{html.count('<table')} tables, {html.count('<figure')} figures")

sus = [w for w in ("nan", "None", "nan%", "{", "}") if w in html]
body = re.sub(r"<style>.*?</style>", "", html, flags=re.S)
check("no un-substituted template braces outside <style>",
      "{" not in body and "}" not in body,
      f"found: {sum(body.count(c) for c in '{}')} brace(s) in the body")
low = [w for w in ("nan", "None,") if re.search(rf"\b{w}\b", html)]
check("no 'nan'/'None' leaked into values", not low, f"{low or 'none'}")

pngs = sorted(n for n in os.listdir(FIGS) if n.endswith(".png"))
print(f"\nPNGs in figs dir: {len(pngs)}")
used = {os.path.basename(s) for s in imgs}
unused = [n for n in pngs if n not in used and not n.startswith("_")]
print(f"  referenced by the document: {len(used)}")
print(f"  present but unreferenced: {unused or 'none'}")
print(f"\nDOC: {DOC}  ({os.path.getsize(DOC)/1024:.1f} kB)")
print(f"FIGDIR: {FIGS}")

npass = sum(1 for ok in RESULTS if ok)
print(f"\n{npass}/{len(RESULTS)} checks passed")
sys.exit(0 if npass == len(RESULTS) else 1)
