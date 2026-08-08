#!/usr/bin/env python3
"""Cut the "markup signature ≠ English" debt into agent-sized units (page-file format).

  python prep_sigfix.py <每单元权重> [--pack <name.json>] [--min <字符>]

What this debt is
-----------------
`apply_translations.py` accepts a translation only when its markup multiset equals
the English one. But that gate runs **at write time**, so anything that entered the
library before the gate existed -- or was written when the English still looked
different -- can sit there permanently violating it. Measured on
`ember.crucible-adventure`: **357 entries / ~791 K chars**.

Neither `measure_8c.py` nor `measure_stale_extra.py` sees this: they compare
`<p>`/`<li>` COUNTS, and these pages usually have the right number of blocks.
What is wrong is inside them -- most often:

* the `<sup class="system-swap-inline"><sub data-system="dnd5e">…</sub>
  <sub data-system="crucible">…</sub></sup>` dual-display pair was collapsed to a
  single branch, so the other system's reader sees nothing;
* `[[/skillCheck awareness 15]]` drifted to a dnd5e-style `[[/check 15 awareness]]`
  (or vice versa) after an upstream edit;
* an `@UUID[...]` now points at a different id than the English does.

Why it is worth a batch of its own (three payoffs at once)
----------------------------------------------------------
1. It is a **shipped defect**: broken links and a missing dual-system branch are
   things players see.
2. It **unblocks other work**. The 2026-08-09 term unification had 15 entries it
   could not write, purely because their pre-existing markup already failed the
   gate (verified: feeding the *unmodified* current values through the gate
   rejects the same 15).
3. It **frees ~713 K characters of the twin pack**. `fill_twin.py` drops a TM hit
   when the crucible Chinese's markup does not match the twin's English, and all
   143 dropped journal pages are exactly these entries. Fix here, re-run
   `fill_twin.py`, and the twin fills itself -- the same ordering lesson as
   "land batch 6 before filling the twin" (554 → 143 drops, +421 filled).

Emits the same page-file layout as `prep_realign.py`, so `collect_realign.py`,
`diff_realign.py`, `tagseq.py` and `prose_survival.py` all work unchanged.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ROOT = os.environ.get("EMBER_PARALLEL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "parallel")
CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')

_spec = importlib.util.spec_from_file_location(
    "_apply", os.path.join(P, "3-常用脚本", "qa", "apply_translations.py"))
_apply = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_apply)
markup_signature = _apply.markup_signature


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weight", type=int, nargs="?", default=90000)
    ap.add_argument("--pack", default="ember.crucible-adventure.json",
                    help='包名；"*" 表示仓库里所有包')
    ap.add_argument("--repo", default=os.path.join(P, "1-Ember汉化插件"))
    ap.add_argument("--prefix", default="sigfix")
    ap.add_argument("--min", type=int, default=0)
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    packs = ([f for f in sorted(os.listdir(en_dir))
              if f.endswith('.json') and os.path.exists(os.path.join(cn_dir, f))]
             if a.pack == "*" else [a.pack])

    rows = []
    for pack in packs:
        o = []
        walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8'))['entries'],
             json.load(open(os.path.join(cn_dir, pack), encoding='utf-8'))['entries'], [], o)
        for path, e, c in o:
            if not (c and CJK.search(c)):
                continue
            se, sc = markup_signature(e), markup_signature(c)
            if se == sc:
                continue
            if len(TAG.sub(' ', e)) < a.min:
                continue
            diff = {k: (se.get(k, 0), sc.get(k, 0))
                    for k in set(se) | set(sc) if se.get(k, 0) != sc.get(k, 0)}
            rows.append({
                "pack": pack, "path": path, "en": e, "cn": c, "diff": diff,
                # missing = English has it and Chinese does not; those usually mean
                # a whole branch or link was dropped and must be written back
                "missing": sum(max(0, w - g) for w, g in diff.values()),
                "surplus": sum(max(0, g - w) for w, g in diff.values()),
                "weight": round(0.15 * (len(e) + len(c))),
            })

    # A unit must never mix packs: `collect_realign.py` emits one batch.json and
    # `apply_translations.py --pack` lands exactly one pack. Chunk within a pack.
    units = []
    for pack in packs:
        pr = sorted([r for r in rows if r["pack"] == pack], key=lambda r: -r["weight"])
        if not pr:
            continue
        n = max(1, round(sum(r["weight"] for r in pr) / a.weight))
        buckets = [[] for _ in range(n)]
        load = [0] * n
        for r in pr:
            i = load.index(min(load))
            buckets[i].append(r)
            load[i] += r["weight"]
        units += [b for b in buckets if b]

    manifest = []
    for i, ch in enumerate(units, 1):
        if not ch:
            continue
        name = f"{a.prefix}-{i}"
        d = os.path.join(ROOT, name)
        pages = os.path.join(d, "pages")
        os.makedirs(pages, exist_ok=True)
        index = []
        for j, r in enumerate(ch, 1):
            for side in ("en", "cn"):
                with open(os.path.join(pages, f"{j}.{side}.html"), "w",
                          encoding="utf-8", newline="") as f:
                    f.write(r[side])
            index.append({"id": j, "path": r["path"], "diff": r["diff"],
                          "missing": r["missing"], "surplus": r["surplus"]})
        json.dump({"unit": name, "pack": ch[0]["pack"], "pages": index},
                  open(os.path.join(d, "index.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump({str(r["id"]): ch[k]["cn"] for k, r in enumerate(index)},
                  open(os.path.join(d, "_original_cn.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        manifest.append({"unit": name, "dir": d, "pack": ch[0]["pack"],
                         "pages": len(ch),
                         "missing": sum(r["missing"] for r in ch),
                         "surplus": sum(r["surplus"] for r in ch)})
        print(f'{name:<12}{len(ch):>4} 条  {ch[0]["pack"]:<30} '
              f'缺标记 {sum(r["missing"] for r in ch):>4} / 多标记 {sum(r["surplus"] for r in ch):>4}')

    out = os.path.join(ROOT, "manifest.sigfix.json")
    json.dump(manifest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f'\n{len(manifest)} 单元 / {sum(m["pages"] for m in manifest)} 条')
    print("->", out)


main()
