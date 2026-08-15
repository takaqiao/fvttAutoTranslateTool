#!/usr/bin/env python3
"""P13: do the CN packs carry any NON-TEXT value that Babele will write back?

Babele merges the translation object into the document data. Any key whose value
is not a string is still merged if the mapping asks for it -- and a boolean like
`hidden`, `visible`, `secret`, or an `ownership` map would silently change who
can see the document. List every non-string leaf and every visibility-shaped key
present on the CN side.
"""
import sys, os, json, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS

VIS_KEYS = re.compile(r'ownership|hidden|visible|secret|gmOnly|private|public|permission'
                      r'|displayName|disposition|gmNotes|gamemaster', re.I)

nonstr = Counter()
viskeys = Counter()
samples = {}


def walk(o, path, bag_nonstr, bag_vis, sam):
    if isinstance(o, dict):
        for k, v in o.items():
            if VIS_KEYS.search(k):
                bag_vis[k] += 1
                sam.setdefault(k, ('.'.join(path + [k]), repr(v)[:120]))
            walk(v, path + [k], bag_nonstr, bag_vis, sam)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, path + [str(i)], bag_nonstr, bag_vis, sam)
    elif not isinstance(o, str):
        key = path[-1] if path else '?'
        bag_nonstr[f'{key}={type(o).__name__}'] += 1
        sam.setdefault(f'NONSTR:{key}', ('.'.join(path), repr(o)[:80]))


for rname, repo in REPOS.items():
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    for pack in sorted(os.listdir(cn_dir)):
        if not pack.endswith('.json'):
            continue
        d = json.load(open(os.path.join(cn_dir, pack), encoding='utf-8'))
        walk(d.get('entries', {}), [rname + ':' + pack], nonstr, viskeys, samples)

print('=== non-string leaves on the CN side ===')
for k, v in nonstr.most_common(40):
    print(f'{v:>6}  {k}   e.g. {samples.get("NONSTR:"+k.split("=")[0], ("", ""))}')
print('\n=== visibility-shaped KEYS present on the CN side ===')
for k, v in viskeys.most_common(40):
    print(f'{v:>6}  {k}   e.g. {samples.get(k)}')
