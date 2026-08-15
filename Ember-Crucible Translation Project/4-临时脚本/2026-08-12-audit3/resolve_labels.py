"""For each @Embed[... label="X"], report the target UUID and that actor's EN/CN name."""
import json, os, re
from collections import defaultdict

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [("ember", os.path.join(P, "1-Ember汉化插件")),
         ("crucible", os.path.join(P, "2-Crucible汉化插件"))]
EMB = re.compile(r'@Embed\[([^\]]*?)\s+label\s*=\s*"([^"]*)"([^\]]*)\]')


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


# build id -> (en name, cn name) index over every actor-ish container in both repos
idx = {}
for tag, repo in REPOS:
    for side in ('en', 'cn'):
        d = os.path.join(repo, 'compendium', side)
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json') or f.startswith('_'):
                continue
            data = json.load(open(os.path.join(d, f), encoding='utf-8')).get('entries', {})
            # entries are keyed by name; actors live under <adventure>.actors.<Name>
            o = []
            walk(data, [], o)
            for path, s in o:
                if path.endswith('.name') or path.endswith('.tokenName'):
                    idx.setdefault((tag, path), {})[side] = s

hits = defaultdict(list)
for tag, repo in REPOS:
    for f in sorted(os.listdir(os.path.join(repo, 'compendium', 'cn'))):
        if not f.endswith('.json') or f.startswith('_'):
            continue
        cn = json.load(open(os.path.join(repo, 'compendium', 'cn', f), encoding='utf-8')).get('entries', {})
        o = []
        walk(cn, [], o)
        for path, s in o:
            for m in EMB.finditer(s):
                hits[(m.group(1).strip(), m.group(2))].append((tag, f, path))

print(f'{"uuid":42} {"label":32} n  where')
for (uuid, label), where in sorted(hits.items(), key=lambda x: x[0][1]):
    print(f'{uuid:42} {label:32} {len(where)}  {where[0][2][:70]}')
