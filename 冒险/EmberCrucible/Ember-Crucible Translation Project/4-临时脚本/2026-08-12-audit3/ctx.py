import json, os, re, sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [("ember", os.path.join(P, "1-Ember汉化插件")),
         ("crucible", os.path.join(P, "2-Crucible汉化插件"))]
name = sys.argv[1]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 12
pat = re.compile(r'.{0,90}' + re.escape(name) + r'\s*=\s*\\"[^"]{0,110}', re.S)
n = 0
for tag, repo in REPOS:
    for side in ('en', 'cn'):
        d = os.path.join(repo, 'compendium', side)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json') or f.startswith('_'):
                continue
            raw = open(os.path.join(d, f), encoding='utf-8').read()
            for m in pat.finditer(raw):
                n += 1
                if n > lim:
                    sys.exit()
                print(f'[{tag}/{side}/{f}] ...{m.group(0)}...')
                print()
