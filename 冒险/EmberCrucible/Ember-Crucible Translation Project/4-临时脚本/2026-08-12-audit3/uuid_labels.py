"""Collect EN->CN @UUID label pairs for a set of actor ids (strongest evidence for @Embed label=)."""
import json, os, re, sys
from collections import Counter, defaultdict

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [("ember", os.path.join(P, "1-Ember汉化插件")),
         ("crucible", os.path.join(P, "2-Crucible汉化插件"))]
IDS = sys.argv[1:] or ['6nFbfFrRVb2lGW3F', 'Q4dB8frYvdhTbUYk', 'I4omhcBZxC8PP9jI',
                       '9plFRf3Hurd9r7ol', 'BJ0TzuQIdpqKTm5R', 'zCaW2IABpatj2TDs']


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


pairs = defaultdict(Counter)
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
            c = cm.get(path)
            if not isinstance(c, str):
                continue
            for i in IDS:
                rx = re.compile(r'@UUID\[[^\]]*' + i + r'[^\]]*\]\{([^}]*)\}')
                es = rx.findall(s)
                cs = rx.findall(c)
                for j, e in enumerate(es):
                    pairs[i][(e, cs[j] if j < len(cs) else '<none>')] += 1

for i in IDS:
    print(f'--- Actor.{i}')
    for (e, c), n in pairs[i].most_common():
        print(f'   {n:>4}  {e!r}  ->  {c!r}')
