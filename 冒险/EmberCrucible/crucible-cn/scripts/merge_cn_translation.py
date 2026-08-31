#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge an older Chinese translation into a newer English compendium.

For every string field in the new English compendium:
  - If the same path existed in the old English with an identical string,
    and the old Chinese has a translation at that path, use that Chinese.
  - Otherwise, keep the new English string (so the user knows it needs
    (re)translation).

Structural fields (label / mapping / folders / entries / nested dicts / arrays)
are walked recursively; only leaf strings are swapped.

Inputs  (relative to project root):
  compendium/en/      - baseline old English (e.g. 0.9.0)
  compendium/cn/      - matching old Chinese (e.g. 0.9.0)
  compendium/en_new/  - new English (e.g. 0.9.1), from extract_en_compendium.mjs

Output:
  compendium/cn_new/  - merged; English preserved wherever content changed
                       or is new. Never touches the three input dirs.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def dump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


class Counter:
    __slots__ = ('reused', 'kept_english_changed', 'kept_english_new', 'total_strings')

    def __init__(self):
        self.reused = 0
        self.kept_english_changed = 0
        self.kept_english_new = 0
        self.total_strings = 0


def merge(new_en, old_en, old_cn, c: Counter):
    """Return a merged value mirroring new_en's structure."""
    if isinstance(new_en, str):
        c.total_strings += 1
        if isinstance(old_en, str):
            if new_en == old_en:
                if isinstance(old_cn, str) and old_cn:
                    c.reused += 1
                    return old_cn
                # Old CN missing for this path — treat as untranslated.
                c.kept_english_new += 1
                return new_en
            # English content changed at this path.
            c.kept_english_changed += 1
            return new_en
        # Path did not exist in old English.
        c.kept_english_new += 1
        return new_en

    if isinstance(new_en, dict):
        out = {}
        for k, v in new_en.items():
            oe = old_en.get(k) if isinstance(old_en, dict) else None
            oc = old_cn.get(k) if isinstance(old_cn, dict) else None
            out[k] = merge(v, oe, oc, c)
        return out

    if isinstance(new_en, list):
        out = []
        for i, v in enumerate(new_en):
            oe = old_en[i] if isinstance(old_en, list) and i < len(old_en) else None
            oc = old_cn[i] if isinstance(old_cn, list) and i < len(old_cn) else None
            out.append(merge(v, oe, oc, c))
        return out

    # Numbers, bools, null — pass through as-is.
    return new_en


def merge_pack(new_en_path: Path, old_en_path: Path | None, old_cn_path: Path | None, out_path: Path) -> dict:
    new_en = load(new_en_path)
    old_en = load(old_en_path) if (old_en_path and old_en_path.exists()) else {}
    old_cn = load(old_cn_path) if (old_cn_path and old_cn_path.exists()) else {}

    c = Counter()
    merged = merge(new_en, old_en, old_cn, c)
    dump(out_path, merged)

    return {
        'file': new_en_path.name,
        'total': c.total_strings,
        'reused_cn': c.reused,
        'new_english_changed': c.kept_english_changed,
        'new_english_added': c.kept_english_new,
        'old_en_present': old_en_path is not None and old_en_path.exists(),
        'old_cn_present': old_cn_path is not None and old_cn_path.exists(),
    }


def main():
    ap = argparse.ArgumentParser(description='Merge old CN translation into new EN compendium.')
    ap.add_argument('--old-en', default=str(PROJECT_ROOT / 'compendium' / 'en'),
                    help='Old English compendium dir (baseline)')
    ap.add_argument('--old-cn', default=str(PROJECT_ROOT / 'compendium' / 'cn'),
                    help='Old Chinese compendium dir (matching the baseline)')
    ap.add_argument('--new-en', default=str(PROJECT_ROOT / 'compendium' / 'en_new'),
                    help='New English compendium dir (target)')
    ap.add_argument('--out', default=str(PROJECT_ROOT / 'compendium' / 'cn_new'),
                    help='Output dir for merged result')
    args = ap.parse_args()

    old_en_dir = Path(args.old_en)
    old_cn_dir = Path(args.old_cn)
    new_en_dir = Path(args.new_en)
    out_dir = Path(args.out)

    if not new_en_dir.exists():
        print(f'ERROR: new English dir missing: {new_en_dir}', file=sys.stderr)
        sys.exit(1)

    files = sorted(p for p in new_en_dir.iterdir() if p.suffix == '.json')
    if not files:
        print(f'ERROR: no .json files in {new_en_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Old EN : {old_en_dir}')
    print(f'Old CN : {old_cn_dir}')
    print(f'New EN : {new_en_dir}')
    print(f'Output : {out_dir}')
    print()

    totals = {'total': 0, 'reused_cn': 0, 'new_english_changed': 0, 'new_english_added': 0}

    header = f'{"file":40s} {"strings":>8s} {"reused":>8s} {"changed":>8s} {"added":>8s}  baseline'
    print(header)
    print('-' * len(header))

    for new_f in files:
        old_en_f = old_en_dir / new_f.name
        old_cn_f = old_cn_dir / new_f.name
        out_f = out_dir / new_f.name
        r = merge_pack(new_f, old_en_f, old_cn_f, out_f)
        for k in totals:
            totals[k] += r[k]
        baseline = []
        if not r['old_en_present']:
            baseline.append('NEW PACK (no old EN)')
        elif not r['old_cn_present']:
            baseline.append('no old CN')
        print(f'{r["file"]:40s} {r["total"]:8d} {r["reused_cn"]:8d} {r["new_english_changed"]:8d} {r["new_english_added"]:8d}  {" ".join(baseline)}')

    print('-' * len(header))
    print(f'{"TOTAL":40s} {totals["total"]:8d} {totals["reused_cn"]:8d} {totals["new_english_changed"]:8d} {totals["new_english_added"]:8d}')
    print()
    if totals['total']:
        pct = 100.0 * totals['reused_cn'] / totals['total']
        print(f'Reused existing CN: {pct:.1f}% of translatable strings.')


if __name__ == '__main__':
    main()
