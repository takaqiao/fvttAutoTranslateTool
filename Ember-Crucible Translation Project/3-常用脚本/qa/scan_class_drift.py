#!/usr/bin/env python3
"""Scan for HTML `class` attributes that drifted between the English and the Chinese.

  python scan_class_drift.py --repo <repo> [--pack <name.json>] [--out <json>] [--top N]

The blind spot this covers
--------------------------
`apply_translations.py`'s markup gate builds its signature from TAG NAMES only
(`TAGNAME = <\\s*(/?)([a-zA-Z][a-zA-Z0-9]*)`). So `<ul class="complex-check">`
and a bare `<ul>` are indistinguishable to it, and a translation that dropped
the class sails through with 0 rejections. `scan_markup_drift.py` does not look
at classes either. Batch 6's `tagseq.py` found the first instances by accident.

In Crucible/Ember these classes are functional, not decorative:

* `section.block gamemaster` -- hides GM-only prose from players. **Losing this
  class shows the players text they were never meant to read.** This is the one
  to fix first; it is a content leak, not a style nit.
* `ul.complex-check`, `li.advantage`, `li.critical-success`,
  `li.automatic-success` -- render check outcomes as distinct rows. Without the
  class the reader cannot tell an automatic success from a normal one.
* `sup.system-swap-inline`, `div.system-swap-block` -- the dnd5e/crucible dual
  display. Losing it shows both systems' text at once, or neither.

Note on interpretation: a good share of the drift is NOT translator error but
the same root cause as items 8c/8j -- the Chinese was translated against an
older English whose markup differed (e.g. upstream moved `div.answer` into
`section.block answer`). Re-run this after a realign batch lands; the count
should fall on its own.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter

CLS = re.compile(r'<(\w+)[^>]*?\sclass="([^"]*)"')
CJK = re.compile(r'[一-鿿]')
# Losing these changes what the player can see or how a rule reads.
FUNCTIONAL = ('gamemaster', 'system-swap', 'complex-check', 'advantage',
              'critical-success', 'automatic-success', 'readaloud', 'hazard')


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


def sig(s):
    return Counter(f'{t}.{k}' for t, k in CLS.findall(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--pack')
    ap.add_argument('--out')
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    packs = [a.pack] if a.pack else sorted(
        f for f in os.listdir(en_dir) if f.endswith('.json') and not f.startswith('_'))

    tot = bad = 0
    miss, extra, rows = Counter(), Counter(), []
    for pack in packs:
        cn_p = os.path.join(cn_dir, pack)
        if not os.path.exists(cn_p):
            continue
        en = json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {})
        cn = json.load(open(cn_p, encoding='utf-8')).get('entries', {})
        o = []
        walk(en, cn, [], o)
        for path, e, c in o:
            if not (c and CJK.search(c)):
                continue
            ea, ca = sig(e), sig(c)
            if not ea and not ca:
                continue
            tot += 1
            if ea == ca:
                continue
            bad += 1
            lost = ea - ca
            gained = ca - ea
            for k in lost.elements():
                miss[k] += 1
            for k in gained.elements():
                extra[k] += 1
            rows.append({'pack': pack, 'path': path,
                         'missing': dict(lost), 'extra': dict(gained),
                         'functional': any(f in k for k in lost for f in FUNCTIONAL)})

    fn = [r for r in rows if r['functional']]
    print(f'带 class 的已译条目 {tot}，class 与英文不一致 {bad}')
    print(f'  其中丢掉了**功能性** class 的 {len(fn)} 条 ← 优先修')
    print('\n中文缺失的 class（英文有、中文没有）:')
    for k, v in miss.most_common(a.top):
        mark = '  ←功能性' if any(f in k for f in FUNCTIONAL) else ''
        print(f'   {v:>5}  {k}{mark}')
    print('\n中文多出的 class:')
    for k, v in extra.most_common(a.top):
        print(f'   {v:>5}  {k}')

    gm = [r for r in rows if any('gamemaster' in k for k in r['missing'])]
    if gm:
        print(f'\n⚠ 丢了 section.block gamemaster 的 {len(gm)} 条 —— GM 专属内容会暴露给玩家:')
        for r in gm[:10]:
            print(f'   {r["path"][:100]}')

    if a.out:
        json.dump({'total': tot, 'drifted': bad, 'functional': len(fn),
                   'missing': dict(miss), 'extra': dict(extra), 'items': rows},
                  open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
