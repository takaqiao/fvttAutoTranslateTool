#!/usr/bin/env python3
"""P7: GM-bearing FIELDS (not classes).

Ember journal pages carry sibling fields whose visibility differs:
  contentOverview / contentGamemaster, public / private, overview / summary ...
Cross-field contamination (the GM text written into the player field, or the
two swapped) is invisible to every multiset gate because both fields are
translated and both keep their own markup.

Test: for every parent object holding >1 of these sibling fields, check that the
CN value of field F derives from the EN value of the SAME field F, using
translation-invariant anchors (@UUID / [[ ]] / src / numbers / latin runs).
If CN[F] matches EN[G] better than EN[F], the fields were swapped/copied.
"""
import sys, os, re, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, CJK

VIS_FIELDS = ['contentOverview', 'contentGamemaster', 'public', 'private',
              'overview', 'summary', 'exposition', 'text', 'description',
              'secret', 'gm', 'gamemaster', 'notes']
ANCH = re.compile(r'@UUID\[[^\]]+\]|@Embed\[[^\]]+\]|\[\[[^\]]+\]\]|src="[^"]+"'
                  r'|\b\d+(?:d\d+)?\b|[A-Z][a-zA-Z\'\-]{3,}')


def anchors(s):
    return set(ANCH.findall(re.sub(r'<[^>]+>', ' ', s)))


def jac(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def collect(en, cn, path, out):
    """Yield (path, dict_of_visfields_en, dict_of_visfields_cn)."""
    if isinstance(en, dict):
        present = [k for k in en if k in VIS_FIELDS and isinstance(en[k], str) and en[k].strip()]
        if len(present) > 1 and isinstance(cn, dict):
            out.append(('.'.join(path), {k: en[k] for k in present},
                        {k: cn.get(k) for k in present}))
        for k, v in en.items():
            collect(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            collect(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                    path + [str(i)], out)


groups = []
for rname, repo in REPOS.items():
    en_dir = os.path.join(repo, 'compendium', 'en')
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    for pack in sorted(os.listdir(en_dir)):
        if not pack.endswith('.json') or pack.startswith('_'):
            continue
        cn_p = os.path.join(cn_dir, pack)
        if not os.path.exists(cn_p):
            continue
        en = json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {})
        cn = json.load(open(cn_p, encoding='utf-8')).get('entries', {})
        o = []
        collect(en, cn, [], o)
        for p, ef, cf in o:
            groups.append((rname, pack, p, ef, cf))

print(f'objects holding >1 visibility-bearing field: {len(groups)}')
combo = defaultdict(int)
for _, _, _, ef, _ in groups:
    combo[tuple(sorted(ef))] += 1
for k, v in sorted(combo.items(), key=lambda x: -x[1]):
    print(f'  {v:>5}  {k}')

print('\n=== mis-sourced / swapped ===')
bad = 0
for rname, pack, p, ef, cf in groups:
    keys = list(ef)
    for k in keys:
        c = cf.get(k)
        if not c or not CJK.search(c):
            continue
        own = jac(anchors(ef[k]), anchors(c))
        for g in keys:
            if g == k:
                continue
            other = jac(anchors(ef[g]), anchors(c))
            if other > own and other >= 0.5:
                bad += 1
                print(f'\n{rname}/{pack} {p}')
                print(f'  CN[{k}] looks sourced from EN[{g}]  (jac own={own:.2f} other={other:.2f})')
                print(f'  EN[{k}]: {ef[k][:180]}')
                print(f'  EN[{g}]: {ef[g][:180]}')
                print(f'  CN[{k}]: {c[:180]}')
print(f'\nhits: {bad}')
