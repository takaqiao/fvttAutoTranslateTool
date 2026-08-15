#!/usr/bin/env python3
"""Survey which HTML attribute names actually appear in the library."""
import json, os, re, sys
from collections import Counter, defaultdict

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]

# attr="value" or attr='value' or attr=bare
ATTR = re.compile(r'''(?<![\w-])([a-zA-Z_][\w:-]*)\s*=\s*("([^"]*)"|'([^']*)')''')
TAGRE = re.compile(r'<([a-zA-Z][\w:-]*)((?:[^>"\']|"[^"]*"|\'[^\']*\')*?)/?>')

def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))

cnt = Counter()
samples = defaultdict(list)
for repo in REPOS:
    for side in ('en', 'cn'):
        d = os.path.join(repo, 'compendium', side)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json') or f.startswith('_'): continue
            data = json.load(open(os.path.join(d, f), encoding='utf-8')).get('entries', {})
            leaves = []
            walk(data, [], leaves)
            for path, s in leaves:
                if '<' not in s: continue
                for m in TAGRE.finditer(s):
                    tag, attrs = m.group(1), m.group(2)
                    for am in ATTR.finditer(attrs):
                        name = am.group(1).lower()
                        val = am.group(3) if am.group(3) is not None else am.group(4)
                        key = f'{side}:{name}'
                        cnt[key] += 1
                        if len(samples[name]) < 6:
                            samples[name].append((side, tag, val[:120]))

names = sorted({k.split(':',1)[1] for k in cnt})
print(f'{"attr":24} {"en":>7} {"cn":>7}')
for n in names:
    print(f'{n:24} {cnt.get("en:"+n,0):>7} {cnt.get("cn:"+n,0):>7}')
print()
for n in names:
    print(f'--- {n} ---')
    for side, tag, v in samples[n]:
        print(f'   [{side}] <{tag}> = {v!r}')
