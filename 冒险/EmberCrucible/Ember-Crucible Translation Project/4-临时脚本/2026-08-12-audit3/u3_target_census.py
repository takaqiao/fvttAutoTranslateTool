# -*- coding: utf-8 -*-
"""U3: every leaf that links a given target id, with its EN and CN label.

`scan_uuid_swap` reports only an aggregate ("majority X, support N"); to decide
whether the majority or the `name` field is the wrong side you need the actual
leaves, English label included.  Usage:  u3_target_census.py <id> [<id> ...]
"""
import json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]
LINK = re.compile(r'@([A-Za-z]+)\[([^\]\n]*)\]\{([^}\n]*)\}')


def flat(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            flat(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            flat(v, path + [str(i)], out)
    elif isinstance(node, str):
        out['.'.join(path)] = node


def links(s):
    out = []
    for m in LINK.finditer(s):
        tok = m.group(2).strip().split(' ')[0].split('#')
        out.append((tok[0].split('.')[-1], tok[1] if len(tok) > 1 else '', m.group(3)))
    return out


packs = {}
for repo in REPOS:
    for side in ('en', 'cn'):
        d = os.path.join(P, repo, 'compendium', side)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json') or fn.startswith('_'):
                continue
            o = json.load(open(os.path.join(d, fn), encoding='utf-8'))
            f = {}
            flat(o, [], f)
            packs[(repo, fn, side)] = f

for q in sys.argv[1:]:
    print(f"\n############ {q}")
    rows, c = [], Counter()
    for (repo, fn, side), f in packs.items():
        if side != 'cn':
            continue
        en = packs.get((repo, fn, 'en'), {})
        for path, s in f.items():
            if q not in s:
                continue
            cn_ls = [x for x in links(s) if x[0] == q]
            if not cn_ls:
                continue
            en_ls = [x for x in links(en.get(path, '')) if x[0] == q]
            en_by_anchor = {}
            for k, a, lab in en_ls:
                en_by_anchor.setdefault(a, []).append(lab)
            for k, a, lab in cn_ls:
                e = en_by_anchor.get(a) or ['<none>']
                rows.append((fn, path, a, e[0], lab))
                c[(a, e[0], lab)] += 1
    for (a, e, lab), n in c.most_common():
        print(f"  {n:>3}  anchor={a!r:26} EN={e!r:34} CN={lab!r}")
    print("  --- leaves ---")
    for fn, path, a, e, lab in sorted(rows):
        print(f"     {fn:34} {path[:78]:78} #{a or '-'} EN={e!r} CN={lab!r}")
