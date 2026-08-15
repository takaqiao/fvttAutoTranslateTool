"""Probe: full anchor HTML around data-tooltip / data-tooltip-text, and the /item activity lines."""
import json, os, re

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [("ember", os.path.join(P, "1-Ember汉化插件")),
         ("crucible", os.path.join(P, "2-Crucible汉化插件"))]


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


TAG = re.compile(r'<(?:a|span)\b[^>]*data-tooltip[^>]*>(?:[^<]*)(?:</(?:a|span)>)?')
ITEM = re.compile(r'.{0,40}\[\[/item[^\]]*\]\](?:\{[^}]*\})?')

seen = set()
print('===== data-tooltip anchors (CN side, dedup by shape) =====')
for tag, repo in REPOS:
    for f in sorted(os.listdir(os.path.join(repo, 'compendium', 'cn'))):
        if not f.endswith('.json') or f.startswith('_'):
            continue
        cn = json.load(open(os.path.join(repo, 'compendium', 'cn', f), encoding='utf-8')).get('entries', {})
        o = []
        walk(cn, [], o)
        for path, s in o:
            for m in TAG.finditer(s):
                shape = re.sub(r'[0-9a-zA-Z]{16}', 'ID', m.group(0))
                if shape in seen:
                    continue
                seen.add(shape)
                print(f'[{tag}/{f}] {path}')
                print(f'   {m.group(0)}')
                print()

print('===== [[/item ...]] with activity= (EN vs CN) =====')
for tag, repo in REPOS:
    en_d, cn_d = os.path.join(repo, 'compendium', 'en'), os.path.join(repo, 'compendium', 'cn')
    for f in sorted(os.listdir(en_d)):
        if not f.endswith('.json') or f.startswith('_') or not os.path.exists(os.path.join(cn_d, f)):
            continue
        en = json.load(open(os.path.join(en_d, f), encoding='utf-8')).get('entries', {})
        cn = json.load(open(os.path.join(cn_d, f), encoding='utf-8')).get('entries', {})
        eo, co = [], []
        walk(en, [], eo)
        walk(cn, [], co)
        cm = dict(co)
        for path, s in eo:
            if 'activity=' not in s:
                continue
            print(f'[{tag}/{f}] {path}')
            for m in ITEM.finditer(s):
                print(f'   EN {m.group(0)}')
            c = cm.get(path, '')
            for m in ITEM.finditer(c):
                print(f'   CN {m.group(0)}')
            print()
