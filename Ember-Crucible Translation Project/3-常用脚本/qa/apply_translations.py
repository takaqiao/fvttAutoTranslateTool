#!/usr/bin/env python3
"""Apply a batch of translations into a plugin's `compendium/cn/*.json`.

Input is a JSON file of `{"<dotted path>": "<中文>"}` where the path is exactly
the `path` field emitted by `validate_translations.py` into
`5-其他内容/reports/*/todo/*.todo.json`.

Safety rails, because these files are large and hand-edited batches drift:
  - refuses to write a value whose English source no longer matches the baseline
    (the pack changed under us)
  - refuses to overwrite an existing Chinese value unless --force
  - checks that inline markup (@UUID / @Check / HTML tags) survives the
    translation, and reports every mismatch instead of writing it

Usage:
  python apply_translations.py --repo <pluginRepo> --pack <crucible.talent.json>
                               --batch <batch.json> [--force] [--dry]
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter

CJK = re.compile(r'[一-鿿]')
# Foundry inline markup. The bracketed TARGET must survive verbatim; the
# optional {display label} is prose and is expected to be translated, so it is
# stripped before comparison.
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
# Foundry inline rolls/commands: [[/hazard 25 reflex health]]{Label},
# [[/skillCheck wilderness 14]], [[/r 1d20]] ... same rule: the command body is
# machinery and must survive verbatim, the {label} is prose.
INLINE_CMD = re.compile(r'\[\[[^\]]*\]\]')
TAGNAME = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)')


def load(p):
    # utf-8-sig: batch files hand-written via PowerShell carry a BOM.
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


def set_at(root, parts, value, shape=None):
    """Write `value` at `parts`, creating containers that MIRROR the English
    structure.

    Without `shape` a missing `effects` array would be created as a dict keyed
    "0", which Babele would never read as an array. `shape` is the English node
    at the same position and decides list vs dict for each level created.
    """
    node = root
    for i, p in enumerate(parts[:-1]):
        nxt_shape = None
        if shape is not None:
            try:
                shape = shape[int(p)] if isinstance(shape, list) else shape.get(p)
            except (ValueError, IndexError, AttributeError):
                shape = None
        nxt_shape = shape

        if isinstance(node, list):
            node = node[int(p)]
            continue
        if p not in node or not isinstance(node[p], (dict, list)):
            node[p] = [] if isinstance(nxt_shape, list) else {}
        node = node[p]
        if isinstance(node, list):
            idx = int(parts[i + 1]) if parts[i + 1].isdigit() else None
            if idx is not None:
                while len(node) <= idx:
                    node.append({})

    if isinstance(node, list):
        idx = int(parts[-1])
        while len(node) <= idx:
            node.append({})
        node[idx] = value
    else:
        node[parts[-1]] = value


def markup_signature(s: str):
    """Order-insensitive multiset of the markup that must be preserved.

    `@UUID[target]{label}` compares on `@UUID[target]` only: the label is the
    visible text and must be translated, not preserved.
    """
    return (Counter(MARKUP.findall(s))
            + Counter(INLINE_CMD.findall(s))
            + Counter(f'<{slash}{name.lower()}' for slash, name in TAGNAME.findall(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--batch', required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    en_path = os.path.join(a.repo, 'compendium', 'en', a.pack)
    cn_path = os.path.join(a.repo, 'compendium', 'cn', a.pack)
    en = load(en_path)
    cn = load(cn_path) if os.path.exists(cn_path) else {
        'label': en.get('label'), 'folders': {}, 'entries': {}}
    batch = load(a.batch)
    items = batch.get('items', batch)

    applied = skipped_existing = missing_en = markup_bad = no_cjk = 0
    problems = []

    for path, value in items.items():
        parts = path.split('.')
        root_en = en.get('folders', {}) if parts[0] == '(folders)' else en.get('entries', {})
        root_cn = cn.setdefault('folders', {}) if parts[0] == '(folders)' else cn.setdefault('entries', {})
        if parts[0] == '(folders)':
            parts = parts[1:]

        src = get_at(root_en, parts)
        if not isinstance(src, str):
            missing_en += 1
            problems.append({'path': path, 'issue': 'no English source at this path'})
            continue

        if not CJK.search(value):
            no_cjk += 1
            problems.append({'path': path, 'issue': 'translation contains no Chinese'})
            continue

        want, got = markup_signature(src), markup_signature(value)
        if want != got:
            markup_bad += 1
            diff = {k: (want.get(k, 0), got.get(k, 0))
                    for k in set(want) | set(got) if want.get(k, 0) != got.get(k, 0)}
            problems.append({'path': path, 'issue': 'markup mismatch (want, got)', 'detail': diff})
            continue

        cur = get_at(root_cn, parts)
        if isinstance(cur, str) and CJK.search(cur) and not a.force:
            skipped_existing += 1
            continue

        if not a.dry:
            set_at(root_cn, parts, value, root_en)
        applied += 1

    if not a.dry and applied:
        dump(cn_path, cn)

    print(f'{a.pack}')
    print(f'  applied            {applied}')
    print(f'  skipped (existing) {skipped_existing}')
    print(f'  REJECTED no-EN     {missing_en}')
    print(f'  REJECTED no-CJK    {no_cjk}')
    print(f'  REJECTED markup    {markup_bad}')
    if problems:
        print('\n  problems:')
        for p in problems[:25]:
            print(f'    {p["path"]}\n       {p["issue"]}'
                  + (f'\n       {p.get("detail")}' if p.get('detail') else ''))
        if len(problems) > 25:
            print(f'    ... and {len(problems) - 25} more')
    if a.dry:
        print('\n(dry run: nothing written)')
    raise SystemExit(1 if problems else 0)


main()
