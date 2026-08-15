import json, os, re, sys
from collections import Counter
P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
ATTR = re.compile(r'(?<![\w-])([a-zA-Z_][\w:-]{0,30})\s*=\s*\\"')
cnt = Counter()
for repo in REPOS:
    for side in ('en', 'cn'):
        d = os.path.join(repo, 'compendium', side)
        if not os.path.isdir(d):
            print('MISSING', d); continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json') or f.startswith('_'): continue
            raw = open(os.path.join(d, f), encoding='utf-8').read()
            for m in ATTR.finditer(raw):
                cnt[(side, m.group(1).lower())] += 1
names = sorted({n for _, n in cnt})
print(f'{"attr":26}{"en":>8}{"cn":>8}')
for n in names:
    print(f'{n:26}{cnt.get(("en",n),0):>8}{cnt.get(("cn",n),0):>8}')
