# -*- coding: utf-8 -*-
"""Cross-layer term split probe: EN-gated occurrence counts of UI-enum terms
inside compendium prose, paired leaf-by-leaf with the CN side.
READ ONLY. Assumes compendium/en and compendium/cn share key paths."""
import json, os, re, io, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
ROOT = os.path.abspath(ROOT)
REPOS = ['1-Ember\u6c49\u5316\u63d2\u4ef6', '2-Crucible\u6c49\u5316\u63d2\u4ef6']

def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            for x in walk(v, p + '.' + k if p else k): yield x
    elif isinstance(o, list):
        for i, v in enumerate(o):
            for x in walk(v, p + '[%d]' % i): yield x
    elif isinstance(o, str):
        yield (p, o)

def load(repo, side):
    d = {}
    base = os.path.join(ROOT, repo, 'compendium', side)
    if not os.path.isdir(base): return d
    for fn in sorted(os.listdir(base)):
        if not fn.endswith('.json'): continue
        data = json.load(open(os.path.join(base, fn), encoding='utf-8'))
        for k, v in walk(data):
            d[fn + '|' + k] = v
    return d

# term -> (EN regex, list of CN variants to look for)
TERMS = {
 'Pronouns':      (r'\bPronouns?\b',           [u'\u4ee3\u79f0', u'\u4ee3\u8bcd']),
 'Age':           (r'<strong>Age</strong>',    [u'\u5e74\u4ee3', u'\u5e74\u9f84']),
 'Weight-person': (r'<strong>Weight</strong>', [u'\u91cd\u91cf', u'\u4f53\u91cd']),
 'PublicBio':     (r'Public Biography',        [u'\u516c\u5f00\uff08Public\uff09\u4f20\u8bb0', u'\u516c\u5f00\u4f20\u8bb0']),
 'PrivateBio':    (r'Private Biography',       [u'\u79c1\u5bc6\uff08Private\uff09\u4f20\u8bb0', u'\u79c1\u4eba\u4f20\u8bb0']),
 'ScaledPrice':   (r'Scaled Price',            [u'\u7f29\u653e\u4ef7\u683c', u'\u6309\u6bd4\u4f8b\u5b9a\u4ef7']),
 'BiographyTab':  (r'Biography tab',           [u'\u4f20\u8bb0', u'\u751f\u5e73']),
 'GroupType':     (r'<strong>Group</strong>|Group Actor|Group type', [u'\u7fa4\u7ec4', u'\u56e2\u961f']),
 'HeavyArmor':    (r'Heavy Armor',             [u'\u91cd\u578b\u62a4\u7532', u'\u91cd\u7532']),
 'Mundane':       (r'\bMundane\b',             [u'\u51e1\u4fd7', u'\u51e1\u54c1']),
 'Minion':        (r'\bMinions?\b',            [u'\u5589\u56c9', u'\u722a\u7259', u'\u4ec6\u4ece']),
 'WallTarget':    (r'<h4>Wall</h4>|Wall type', [u'\u5899\u58c1', u'\u5899']),
 'WaxWane':       (r'\bWaxing\b|\bWaning\b',   [u'\u76c8\u6708', u'\u4e8f\u6708', u'\u6e10\u76c8', u'\u6e10\u4e8f']),
}

out = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xlayer_report.txt'), 'w', encoding='utf-8')
for repo in REPOS:
    EN = load(repo, 'en'); CN = load(repo, 'cn')
    out.write('##### %s   en leaves=%d cn leaves=%d\n' % (repo, len(EN), len(CN)))
    for name, (rx, variants) in TERMS.items():
        hits = []
        for k, v in EN.items():
            n = len(re.findall(rx, v))
            if not n: continue
            c = CN.get(k)
            if c is None:
                hits.append((k, n, None, {})); continue
            counts = dict((var, c.count(var)) for var in variants)
            hits.append((k, n, c, counts))
        if not hits: continue
        out.write('\n=== %s  (EN /%s/)  leaves=%d  EN-occurrences=%d\n' % (name, rx, len(hits), sum(h[1] for h in hits)))
        for k, n, c, counts in hits:
            out.write('   %-90s enN=%d  %s\n' % (k[:90], n, ' '.join('%s=%d' % (a, b) for a, b in counts.items())))
out.close()
print('written')
