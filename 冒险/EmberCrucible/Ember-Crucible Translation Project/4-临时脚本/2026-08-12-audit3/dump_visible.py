"""Dump every occurrence of the candidate player-visible attributes, en vs cn."""
import json, os, re, sys
from collections import Counter, defaultdict

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [("ember", os.path.join(P, "1-Ember汉化插件")),
         ("crucible", os.path.join(P, "2-Crucible汉化插件"))]

VIS = ('data-tooltip', 'data-tooltip-text', 'label', 'readaloud', 'title',
       'alt', 'aria-label', 'placeholder', 'data-label', 'activity')
ATTR = re.compile(r'(?<![\w-])(' + '|'.join(VIS) + r')\s*=\s*"([^"]*)"', re.I)


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


for tag, repo in REPOS:
    en_d = os.path.join(repo, 'compendium', 'en')
    cn_d = os.path.join(repo, 'compendium', 'cn')
    for f in sorted(os.listdir(en_d)):
        if not f.endswith('.json') or f.startswith('_'):
            continue
        cnp = os.path.join(cn_d, f)
        if not os.path.exists(cnp):
            continue
        en = json.load(open(os.path.join(en_d, f), encoding='utf-8')).get('entries', {})
        cn = json.load(open(cnp, encoding='utf-8')).get('entries', {})
        eo, co = [], []
        walk(en, [], eo)
        walk(cn, [], co)
        cnmap = dict(co)
        for path, s in eo:
            ems = ATTR.findall(s)
            if not ems:
                continue
            c = cnmap.get(path)
            cms = ATTR.findall(c) if isinstance(c, str) else []
            for i, (an, av) in enumerate(ems):
                cv = cms[i][1] if i < len(cms) and cms[i][0].lower() == an.lower() else '<<NO-PAIR>>'
                print(f'{tag}\t{f}\t{an.lower()}\t{path}\n   EN: {av}\n   CN: {cv}')
