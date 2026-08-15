#!/usr/bin/env python3
"""P4: does each translation-invariant ANCHOR sit inside/outside the GM block on
both sides?

Anchors that survive translation verbatim:
  * @UUID[...]            document links
  * [[ ... ]]             inline rolls / enrichers
  * id="slug"             heading anchors (project-added, 1642 of them)
  * <img src="...">       image paths
  * @Embed / @Check etc.

Method: for each string, find every <section> region; classify a region as
GM-hidden if its own class OR any ancestor's class contains `gamemaster` or is
exactly/contains `secret`.  Then map anchor -> hidden? on each side and diff.

False-positive modes:
  * an anchor duplicated N times inside and M outside -> we compare multisets of
    (anchor, hidden) so a pure duplication change also trips. Printed for eyeball.
  * anchors the translator legitimately added/removed (should be 0: anchor gate
    is already green project-wide).
"""
import sys, os, re, json
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs, CJK
from htmlblocks import regions, balanced

ANCHOR = re.compile(
    r'@UUID\[[^\]]+\]'
    r'|@Embed\[[^\]]+\]'
    r'|\[\[[^\]]+\]\]'
    r'|id="[^"]+"'
    r'|src="[^"]+"'
)
HID = re.compile(r'gamemaster|secret', re.I)


def hidden_spans(html):
    """List of (start,end) char ranges that are GM-hidden (union of section regions
    whose class matches HID, including everything nested inside)."""
    sp = []
    for cls, ins, ine, outs, oute in regions(html, 'section'):
        if HID.search(cls or ''):
            sp.append((outs, oute))
    return sp


def is_hidden(pos, spans):
    return any(s <= pos < e for s, e in spans)


def profile(html):
    spans = hidden_spans(html)
    c = Counter()
    for m in ANCHOR.finditer(html):
        c[(m.group(0), is_hidden(m.start(), spans))] += 1
    return c, spans


unbal = []
hits = []
tot = 0
gm_strings = 0
for rname, repo in REPOS.items():
    for pack, rows in pairs(repo):
        for path, e, c in rows:
            if not (c and CJK.search(c)):
                continue
            if '<section' not in e and '<section' not in c:
                continue
            tot += 1
            for h, side in ((e, 'EN'), (c, 'CN')):
                ok, o, cl = balanced(h, 'section')
                if not ok:
                    unbal.append((rname, pack, path, side, o, cl))
            pe, se_ = profile(e)
            pc, sc_ = profile(c)
            if not se_ and not sc_:
                continue
            gm_strings += 1
            if pe == pc:
                continue
            lost = pe - pc
            gained = pc - pe
            hits.append({'repo': rname, 'pack': pack, 'path': path,
                         'en_only': [f'{"HID" if h else "OPEN"} {a}' for (a, h), n in lost.items() for _ in range(n)],
                         'cn_only': [f'{"HID" if h else "OPEN"} {a}' for (a, h), n in gained.items() for _ in range(n)],
                         })

print(f'strings with <section> and CJK translation: {tot}')
print(f'  of which some GM/secret region exists: {gm_strings}')
print(f'unbalanced <section> occurrences: {len(unbal)}')
for u in unbal[:20]:
    print('   ', u)
print(f'\nanchor-membership mismatches: {len(hits)}')
for h in hits:
    print('\n' + '=' * 70)
    print(f'{h["repo"]}/{h["pack"]}  {h["path"]}')
    print('  EN-side only:', h['en_only'][:12])
    print('  CN-side only:', h['cn_only'][:12])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p4_membership.json')
json.dump({'unbalanced': unbal, 'hits': hits}, open(out, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\n->', out)
