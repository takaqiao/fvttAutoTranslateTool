#!/usr/bin/env python3
"""Restore `&Reference[...]` keys that were translated into Chinese.

  python fix_reference_keys.py [--write]

The defect
----------
`&Reference[Restrained]` became `&Reference[受拘束]`, `&Reference[poisoned]`
became `&Reference[中毒]`, and so on -- 75 entries across the two campaign packs.

Per `BRIEF.md` §2 the bracket content is a **dnd5e reference key**, not prose:
the enricher looks it up and renders the localized name itself. Translating the
key breaks the lookup, so the player sees raw text instead of a rules link. It is
the same rule as the 2026-08-06 decision "enricher 方括号内的目标与参数一律照抄".

Why nothing caught it
---------------------
`apply_translations.markup_signature` builds `MARKUP` from `@[A-Za-z]+\\[...\\]`
-- only `@`-prefixed markers. `&Reference[...]` starts with `&`, so it never
entered the signature and the gate reported 0 rejections while the key was being
rewritten. `scan_markup_targets.py` looks for Chinese inside `@`-markers and
misses it for the same reason. Found by the batch-7 cross-check agent.

The repair is positional: EN and CN carry the same NUMBER of `&Reference`
tokens, so the i-th CN token takes the i-th EN token's key. Entries where the
counts differ are reported and left alone -- those need a human read, not a
mechanical swap.
"""
from __future__ import annotations
import json
import os
import re
import sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ("1-Ember汉化插件", "2-Crucible汉化插件")
# keep the entity form (&amp; vs &) the entry already uses; only the key changes
REF = re.compile(r'(&(?:amp;)?[Rr]eference\[)([^\]]*)(\])')
CJK = re.compile(r'[一-鿿]')


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
    write = '--write' in sys.argv
    total = fixed = skipped = 0
    batches = {}
    for repo in REPOS:
        d = os.path.join(P, repo, 'compendium')
        en_dir, cn_dir = os.path.join(d, 'en'), os.path.join(d, 'cn')
        for pack in sorted(os.listdir(en_dir)):
            if not pack.endswith('.json') or not os.path.exists(os.path.join(cn_dir, pack)):
                continue
            o = []
            walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {}),
                 json.load(open(os.path.join(cn_dir, pack), encoding='utf-8')).get('entries', {}),
                 [], o)
            for path, e, c in o:
                if not (c and CJK.search(c)):
                    continue
                en_keys = [m.group(2) for m in REF.finditer(e)]
                cn_keys = [m.group(2) for m in REF.finditer(c)]
                if not en_keys and not cn_keys:
                    continue
                if en_keys == cn_keys:
                    continue
                total += 1
                if len(en_keys) != len(cn_keys):
                    skipped += 1
                    print(f'  跳过（数量不等 {len(en_keys)} vs {len(cn_keys)}）: {path[:95]}')
                    continue
                it = iter(en_keys)
                new = REF.sub(lambda m: m.group(1) + next(it) + m.group(3), c)
                if new != c:
                    fixed += 1
                    batches.setdefault((repo, pack), {})[path] = new
    print(f'\n不一致 {total} 条：可机械修复 {fixed}，跳过 {skipped}')
    if write:
        out_dir = os.path.join(P, '5-其他内容', 'reports', 'ember')
        os.makedirs(out_dir, exist_ok=True)
        for (repo, pack), b in batches.items():
            f = os.path.join(out_dir, f'reffix.{repo[0]}.{pack}')
            json.dump(b, open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(f'  {repo} / {pack}: {len(b)} 条 -> {f}')
        print('\n用 apply_translations.py --force 落盘（目标路径本来就有中文）')
    else:
        print('(未加 --write，未产出 batch)')


main()
