"""Probe: (a) the two already-translated label=, (b) leaves where attr counts differ."""
import json, os, re
from collections import Counter

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [("ember", os.path.join(P, "1-Ember汉化插件")),
         ("crucible", os.path.join(P, "2-Crucible汉化插件"))]
CJK = re.compile(r'[\u4e00-\u9fff]')
ATTRN = re.compile(r'(?<![\w-])([a-zA-Z_][\w:-]{0,30})\s*=\s*"')


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


print('===== (a) label= values that DO contain CJK =====')
for tag, repo in REPOS:
    for f in sorted(os.listdir(os.path.join(repo, 'compendium', 'cn'))):
        if not f.endswith('.json') or f.startswith('_'):
            continue
        cn = json.load(open(os.path.join(repo, 'compendium', 'cn', f), encoding='utf-8')).get('entries', {})
        o = []
        walk(cn, [], o)
        for path, s in o:
            for m in re.finditer(r'(?<![\w-])label\s*=\s*"([^"]*)"', s):
                if CJK.search(m.group(1)):
                    i = m.start()
                    print(f'[{tag}/{f}] {path}')
                    print(f'   ...{s[max(0,i-160):i+90]}...')

print()
print('===== (b) leaves whose attribute-name multiset differs en vs cn =====')
for tag, repo in REPOS:
    en_d = os.path.join(repo, 'compendium', 'en')
    cn_d = os.path.join(repo, 'compendium', 'cn')
    for f in sorted(os.listdir(en_d)):
        if not f.endswith('.json') or f.startswith('_'):
            continue
        if not os.path.exists(os.path.join(cn_d, f)):
            continue
        en = json.load(open(os.path.join(en_d, f), encoding='utf-8')).get('entries', {})
        cn = json.load(open(os.path.join(cn_d, f), encoding='utf-8')).get('entries', {})
        eo, co = [], []
        walk(en, [], eo)
        walk(cn, [], co)
        cnmap = dict(co)
        for path, s in eo:
            c = cnmap.get(path)
            if not isinstance(c, str):
                continue
            ea = Counter(m.group(1).lower() for m in ATTRN.finditer(s))
            ca = Counter(m.group(1).lower() for m in ATTRN.finditer(c))
            if ea != ca:
                print(f'[{tag}/{f}] {path}')
                print(f'   en-only: {dict(ea - ca)}   cn-only: {dict(ca - ea)}')
