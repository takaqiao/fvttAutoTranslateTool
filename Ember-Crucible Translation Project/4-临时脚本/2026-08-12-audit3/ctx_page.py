"""Dump the CN text of a given leaf path (and the EN) so labels can be judged in context."""
import json, os, re, sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPO = os.path.join(P, "1-Ember汉化插件")
PACK = "ember.adventure.json"
frag = sys.argv[1]
win = int(sys.argv[2]) if len(sys.argv) > 2 else 700


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


for side in ('en', 'cn'):
    data = json.load(open(os.path.join(REPO, 'compendium', side, PACK), encoding='utf-8')).get('entries', {})
    o = []
    walk(data, [], o)
    for path, s in o:
        if frag not in path:
            continue
        print(f'##### [{side}] {path}')
        print(s[:win])
        print()
