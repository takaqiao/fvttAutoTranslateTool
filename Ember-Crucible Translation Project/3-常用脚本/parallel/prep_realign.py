#!/usr/bin/env python3
"""Cut items 8c + 8j into agent-sized units, as *page files* rather than a todo list.

Why this replaces `prep_8c.py`
------------------------------
8c (中文缺块) and 8j (中文多出块) are the same defect seen from two sides: the
Chinese was translated against an older English. 32 pages are on BOTH lists --
upstream rewrote them, adding some blocks and deleting others. Handing those to
two different agents means two batch files claiming the same path, and whichever
lands second silently wins. So they are merged here into one task: **re-align
the page to today's English**.

Two measurements forced the format change:

1. `prep_8c.py` sizes units by `est_chars` (the MISSING content, 260 K), but an
   agent emitting a full replacement value must reproduce the WHOLE page --
   1.45 M chars. A 5.6x under-estimate. Stage 23's `fill-8` was cut off after
   4 of 12 entries; this is why.
2. Re-emitting a whole page puts every already-vetted sentence back through the
   model, which is exactly how carefully reviewed Chinese gets quietly reworded.
   Stage 23 made "有没有被无故重写" the reviewer's second check item.

So each page ships as two files: `<i>.en.html` (today's English) and
`<i>.cn.html` (**a byte-exact copy of the current Chinese**). The agent edits
`<i>.cn.html` in place -- inserting the missing blocks, deleting the ones the
English no longer has, touching nothing else. Untouched text is preserved by
construction rather than by good intentions, and authored output falls to the
~260 K that genuinely needs writing.

`collect_realign.py` assembles the edited files back into a batch.json.

  python prep_realign.py <每单元权重> [--min <单条最小估算字符>]
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"

# Reuse the gate's own path resolver rather than re-implementing it. Page keys
# like `Patch 0.2.0` contain dots; a naive split('.') loses them, which is the
# bug stage 21 already fixed once inside apply_translations.py.
_spec = importlib.util.spec_from_file_location(
    "_apply", os.path.join(P, "3-常用脚本", "qa", "apply_translations.py"))
_apply = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_apply)
split_path = _apply.split_path
ROOT = os.environ.get("EMBER_PARALLEL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "parallel")
PACK = "ember.crucible-adventure.json"
REPORTS = os.path.join(P, "5-其他内容", "reports", "ember")
MISSING = os.path.join(REPORTS, "missing_blocks.json")
EXTRA = os.path.join(REPORTS, "stale_extra_blocks.json")

# Context cost is dominated by READING both sides; authoring cost by the blocks
# that actually change. Deletions are near-free to author, hence the low weight.
W_READ = 0.15
W_FILL = 1.0
W_TRIM = 0.4


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    min_chars = 0
    if "--min" in sys.argv:
        min_chars = int(sys.argv[sys.argv.index("--min") + 1])

    miss = {i["path"]: i for i in json.load(open(MISSING, encoding="utf-8"))["items"]}
    extra = {i["path"]: i for i in json.load(open(EXTRA, encoding="utf-8"))["items"]}
    repo = os.path.join(P, "1-Ember汉化插件", "compendium")
    en = json.load(open(os.path.join(repo, "en", PACK), encoding="utf-8"))["entries"]
    cn = json.load(open(os.path.join(repo, "cn", PACK), encoding="utf-8"))["entries"]

    rows, skipped = [], []
    for path in sorted(set(miss) | set(extra)):
        m, x = miss.get(path, {}), extra.get(path, {})
        fill_c, trim_c = m.get("est_chars", 0), x.get("est_chars", 0)
        if fill_c + trim_c < min_chars:
            continue
        e = get_at(en, split_path(en, path))
        c = get_at(cn, split_path(cn, path))
        if not isinstance(e, str) or not isinstance(c, str):
            skipped.append(path)
            continue
        rows.append({
            "path": path, "en": e, "cn": c,
            "missing_blocks": m.get("missing_blocks", 0),
            "extra_blocks": x.get("extra_blocks", 0),
            "en_blocks": m.get("en_blocks") or x.get("cn_blocks", 0),
            "short": m.get("short", {}), "over": x.get("over", {}),
            "fill_chars": fill_c, "trim_chars": trim_c,
            "weight": round(W_READ * (len(e) + len(c)) + W_FILL * fill_c + W_TRIM * trim_c),
        })

    # Largest first, then greedy fill: keeps any single monster page from
    # landing on top of an already-full unit.
    rows.sort(key=lambda r: -r["weight"])
    n_units = max(1, round(sum(r["weight"] for r in rows) / chunk))
    units = [[] for _ in range(n_units)]
    load = [0] * n_units
    for r in rows:
        i = load.index(min(load))
        units[i].append(r)
        load[i] += r["weight"]

    manifest = []
    for i, ch in enumerate(units, 1):
        if not ch:
            continue
        name = f"realign-{i}"
        d = os.path.join(ROOT, name)
        pages = os.path.join(d, "pages")
        os.makedirs(pages, exist_ok=True)
        index = []
        for j, r in enumerate(ch, 1):
            with open(os.path.join(pages, f"{j}.en.html"), "w", encoding="utf-8", newline="") as f:
                f.write(r["en"])
            with open(os.path.join(pages, f"{j}.cn.html"), "w", encoding="utf-8", newline="") as f:
                f.write(r["cn"])
            index.append({k: r[k] for k in (
                "path", "missing_blocks", "extra_blocks", "en_blocks",
                "short", "over", "fill_chars", "trim_chars")} | {"id": j})
        json.dump({"unit": name, "pack": PACK, "pages": index},
                  open(os.path.join(d, "index.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        # byte-exact originals, so collect_realign.py can report what was touched
        json.dump({str(r["id"]): ch[k]["cn"] for k, r in enumerate(index)},
                  open(os.path.join(d, "_original_cn.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        fills = sum(1 for r in ch if r["missing_blocks"])
        trims = sum(1 for r in ch if r["extra_blocks"])
        both = sum(1 for r in ch if r["missing_blocks"] and r["extra_blocks"])
        manifest.append({"unit": name, "dir": d, "pages": len(ch),
                         "weight": load[i - 1],
                         "fill_chars": sum(r["fill_chars"] for r in ch),
                         "trim_chars": sum(r["trim_chars"] for r in ch)})
        print(f"{name:<14}{len(ch):>4} 页  权重 {load[i-1]:>7}  "
              f"补 {sum(r['fill_chars'] for r in ch):>6} / 删 {sum(r['trim_chars'] for r in ch):>5} 字符"
              f"   (补{fills} 删{trims} 兼{both})")

    out = os.path.join(ROOT, "manifest.realign.json")
    json.dump(manifest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n{len(manifest)} 单元 / {sum(m['pages'] for m in manifest)} 页 / "
          f"补 {sum(m['fill_chars'] for m in manifest):,} + 删 {sum(m['trim_chars'] for m in manifest):,} 字符")
    if skipped:
        print(f"跳过 {len(skipped)} 条（英文或中文侧不是字符串）: {skipped[:5]}")
    print("->", out)


main()
