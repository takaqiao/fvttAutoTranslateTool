#!/usr/bin/env python3
"""Rescue orphaned translations after an upstream rename.

`validate_translations.py` flags CN values sitting at paths the English baseline
no longer has. Most are upstream renames rather than deletions — crucible 0.10.1
renamed the whole `Lightning` rune family to `Storm`, for example — and the
Chinese for those is still perfectly good.

This script applies a rename table to orphan paths, moves the translation to the
new path when the new path exists in the English baseline and is still
untranslated, and reports whatever it could not place so nothing disappears
silently.

Usage:
  python port_orphans.py --repo <pluginRepo> --rules <rules.json> [--dry]

rules.json: {"substitutions": [["Lightning", "Storm"], ...]}
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def dump(p, o):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(o, f, ensure_ascii=False, indent=2)
        f.write('\n')


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


def leaf_paths(node, prefix=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaf_paths(v, prefix + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaf_paths(v, prefix + (str(i),))
    elif isinstance(node, str) and node.strip():
        yield prefix, node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--rules', required=True)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    subs = [tuple(s) for s in load(a.rules)['substitutions']]
    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')

    ported = unplaced = already = 0
    detail = []

    for fn in sorted(f for f in os.listdir(en_dir) if f.endswith('.json') and not f.startswith('_')):
        cn_path = os.path.join(cn_dir, fn)
        if not os.path.exists(cn_path):
            continue
        en = load(os.path.join(en_dir, fn)).get('entries', {})
        cn_doc = load(cn_path)
        cn = cn_doc.get('entries', {})
        changed = False

        for parts, value in list(leaf_paths(cn)):
            if not CJK.search(value):
                continue
            if get_at(en, list(parts)) is not None:
                continue                                  # not an orphan

            for old, new in subs:
                renamed = [p.replace(old, new) for p in parts]
                if renamed == list(parts):
                    continue
                if get_at(en, renamed) is None:
                    continue                              # rename target absent
                existing = get_at(cn, renamed)
                if isinstance(existing, str) and CJK.search(existing):
                    already += 1
                    detail.append({'file': fn, 'from': '.'.join(parts),
                                   'to': '.'.join(renamed), 'result': 'target already translated'})
                    break

                node = cn
                for i, p in enumerate(renamed[:-1]):
                    shape = get_at(en, renamed[:i + 1])
                    if isinstance(node, list):
                        node = node[int(p)]
                        continue
                    if p not in node or not isinstance(node[p], (dict, list)):
                        node[p] = [] if isinstance(shape, list) else {}
                    node = node[p]
                    if isinstance(node, list):
                        idx = int(renamed[i + 1]) if renamed[i + 1].isdigit() else None
                        if idx is not None:
                            while len(node) <= idx:
                                node.append({})
                if isinstance(node, list):
                    idx = int(renamed[-1])
                    while len(node) <= idx:
                        node.append({})
                    node[idx] = value
                else:
                    node[renamed[-1]] = value

                ported += 1
                changed = True
                detail.append({'file': fn, 'from': '.'.join(parts),
                               'to': '.'.join(renamed), 'result': 'ported'})
                break
            else:
                unplaced += 1
                detail.append({'file': fn, 'from': '.'.join(parts),
                               'result': 'no rename rule matched — left in place'})

        if changed and not a.dry:
            dump(cn_path, cn_doc)

    print(f'ported                      {ported}')
    print(f'target already translated   {already}')
    print(f'orphans with no rule        {unplaced}')
    for d in detail[:40]:
        if d['result'] == 'ported':
            print(f"  {d['file']}: {d['from']}  ->  {d['to']}")
    unmatched = [d for d in detail if d['result'].startswith('no rename')]
    if unmatched:
        print('\n  orphans left in place (dead translations, review manually):')
        for d in unmatched[:30]:
            print(f"    {d['file']}: {d['from']}")
        if len(unmatched) > 30:
            print(f'    ... and {len(unmatched) - 30} more')
    if a.dry:
        print('\n(dry run: nothing written)')


main()
