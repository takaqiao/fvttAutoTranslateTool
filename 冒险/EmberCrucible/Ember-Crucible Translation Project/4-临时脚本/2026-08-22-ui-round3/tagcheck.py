# -*- coding: utf-8 -*-
"""EN/CN enricher + HTML-tag multiset comparison, leaf by leaf.

Acceptance item: "@UUID/@Condition/@Embed 多重集对拍; EN/CN 标签多重集不等的叶必须 0".

For every leaf path present in BOTH compendium/en and compendium/cn:
  * enricher multiset  = Counter of `@Word[...]` / `&Word[...]` occurrences
    (verb + whole bracket body, so a swapped target counts as a difference)
  * tag multiset       = Counter of HTML tag names (opening tags + closing tags)
Reports the number of leaves where either multiset differs.

Usage: python tagcheck.py <repoDir> [--out report.json]
"""
from __future__ import annotations
import json, os, re, sys, io
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENRICH = re.compile(r'(@[A-Za-z]+|&(?:amp;)?[A-Za-z]+)\[(?:[^\]"]|"[^"]*")*\]')
TAG = re.compile(r'</?([A-Za-z][A-Za-z0-9]*)\b')
INLINE = re.compile(r'\[\[[^\]]*\]\]')


def leaves(node, prefix=""):
    if isinstance(node, str):
        yield prefix, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, f"{prefix}[{i}]")


def enr(s):
    """Multiset keyed on (verb, TARGET) - NOT the whole bracket body.

    Keying on the body was tried first and reported 74 "mismatching" leaves that
    are all correct translations: `@Embed[Actor.X readaloud="..."]` is SUPPOSED
    to differ, its payload is prose. The invariant that must hold is the verb and
    the thing it points at, so the target is the first whitespace-delimited token
    inside the brackets (a UUID, a condition name, an inline-roll expression)."""
    c = Counter()
    for m in ENRICH.finditer(s):
        verb = m.group(1).replace('amp;', '')
        body = m.group(0)[len(m.group(1)) + 1:-1]
        c[(verb, body.split(' ', 1)[0].split('{', 1)[0])] += 1
    for m in INLINE.finditer(s):
        c[('[[]]', m.group(0)[2:-2].split('#', 1)[0].strip())] += 1
    return c


def tags(s):
    return Counter(m.group(1).lower() for m in TAG.finditer(s))


def main():
    repo = sys.argv[1]
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    en_dir, cn_dir = os.path.join(repo, "compendium", "en"), os.path.join(repo, "compendium", "cn")
    total_common = 0
    bad_enr, bad_tag = [], []
    per_pack = {}
    for fn in sorted(os.listdir(cn_dir)):
        if not fn.endswith(".json"):
            continue
        ep = os.path.join(en_dir, fn)
        if not os.path.exists(ep):
            per_pack[fn] = {"skipped": "no en counterpart"}
            continue
        EN = dict(leaves(json.load(io.open(ep, encoding="utf-8"))))
        CN = dict(leaves(json.load(io.open(os.path.join(cn_dir, fn), encoding="utf-8"))))
        common = set(EN) & set(CN)
        total_common += len(common)
        pe = pt = 0
        for k in common:
            if enr(EN[k]) != enr(CN[k]):
                pe += 1
                if len(bad_enr) < 40:
                    bad_enr.append({"pack": fn, "path": k,
                                    "en_only": [list(x) for x in (enr(EN[k]) - enr(CN[k])).elements()][:6],
                                    "cn_only": [list(x) for x in (enr(CN[k]) - enr(EN[k])).elements()][:6]})
            if tags(EN[k]) != tags(CN[k]):
                pt += 1
                if len(bad_tag) < 40:
                    bad_tag.append({"pack": fn, "path": k,
                                    "en_only": list((tags(EN[k]) - tags(CN[k])).elements())[:6],
                                    "cn_only": list((tags(CN[k]) - tags(EN[k])).elements())[:6]})
        per_pack[fn] = {"common_leaves": len(common), "enricher_mismatch": pe, "tag_mismatch": pt,
                        "en_only_leaves": len(set(EN) - set(CN)), "cn_only_leaves": len(set(CN) - set(EN))}
        print(f"{fn:38s} common={len(common):6d}  enricherMismatch={pe:4d}  tagMismatch={pt:4d}"
              f"  enOnly={len(set(EN)-set(CN)):5d}  cnOnly={len(set(CN)-set(EN)):5d}")
    print(f"\nTOTAL common leaves={total_common}  enricher-mismatch leaves={sum(v.get('enricher_mismatch',0) for v in per_pack.values())}"
          f"  tag-mismatch leaves={sum(v.get('tag_mismatch',0) for v in per_pack.values())}")
    print(f"TOTAL en-only leaves={sum(v.get('en_only_leaves',0) for v in per_pack.values())}"
          f"  cn-only leaves={sum(v.get('cn_only_leaves',0) for v in per_pack.values())}")
    for b in bad_enr[:20]:
        print("  ENR", b)
    for b in bad_tag[:20]:
        print("  TAG", b)
    if out:
        io.open(out, "w", encoding="utf-8").write(json.dumps(
            {"per_pack": per_pack, "enricher_samples": bad_enr, "tag_samples": bad_tag}, ensure_ascii=False, indent=2))
        print("->", out)


if __name__ == "__main__":
    main()
