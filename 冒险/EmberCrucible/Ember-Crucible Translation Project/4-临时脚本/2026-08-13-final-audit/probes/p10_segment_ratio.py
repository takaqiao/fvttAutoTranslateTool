#!/usr/bin/env python3
"""P10: per-text-node length profile inside vs outside GM blocks.

p6 proved the ordered tag stream is identical for all but 35 strings (all
inline word-order). That lets us zip the EN and CN *text nodes* 1:1 and compare
each pair. A sentence dragged out of `section.block gamemaster` into the
adjacent player paragraph keeps every tag and every class -- only the text-node
lengths move. This is the last mechanism by which GM prose can leak without
tripping any existing gate.

Flag: within one string, a text node whose CN/EN length ratio departs far from
that string's own median ratio, when the node is inside a GM/secret region on
one side and its neighbour is not.
False positives: legitimately terse or expansive renderings of one sentence;
that is why we require a large departure AND print the pair.
"""
import sys, os, re, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs, CJK
from htmlblocks import regions

TAGRE = re.compile(r'<[^>]+>')
EMPTY_P = re.compile(r'<p>\s*</p>')
HID = re.compile(r'gamemaster|secret', re.I)


def nodes(html):
    """[(text, char_offset)] of text between tags, non-blank only."""
    out, pos = [], 0
    for m in TAGRE.finditer(html):
        t = html[pos:m.start()]
        if t.strip():
            out.append((t, pos))
        pos = m.end()
    if html[pos:].strip():
        out.append((html[pos:], pos))
    return out


def spans(html):
    return [(o, e) for cls, i, j, o, e in regions(html, 'section') if HID.search(cls or '')]


def hid(p, sp):
    return any(s <= p < e for s, e in sp)


hits = []
scanned = 0
for rname, repo in REPOS.items():
    for pack, prs in pairs(repo):
        for path, e, c in prs:
            if not (c and CJK.search(c)):
                continue
            e2, c2 = EMPTY_P.sub('', e), EMPTY_P.sub('', c)
            se, sc = spans(e2), spans(c2)
            if not se and not sc:
                continue
            ne, nc = nodes(e2), nodes(c2)
            if len(ne) != len(nc) or len(ne) < 3:
                continue
            scanned += 1
            ratios = []
            for (te, pe), (tc, pc) in zip(ne, nc):
                le, lc = len(te.strip()), len(tc.strip())
                if le >= 40:
                    ratios.append(lc / le)
            if len(ratios) < 4:
                continue
            med = statistics.median(ratios)
            if med <= 0:
                continue
            for i, ((te, pe), (tc, pc)) in enumerate(zip(ne, nc)):
                le, lc = len(te.strip()), len(tc.strip())
                if le < 60:
                    continue
                r = lc / le
                he, hc = hid(pe, se), hid(pc, sc)
                if he != hc:
                    hits.append({'kind': 'NODE_SIDE_FLIP', 'repo': rname, 'pack': pack,
                                 'path': path, 'i': i, 'en': te[:200], 'cn': tc[:200]})
                    continue
                if r > 2.6 * med or r < 0.38 * med:
                    hits.append({'kind': 'RATIO', 'repo': rname, 'pack': pack, 'path': path,
                                 'i': i, 'hidden': he, 'r': round(r, 2), 'med': round(med, 2),
                                 'en': te[:220], 'cn': tc[:220]})

print(f'GM-bearing strings with 1:1 zippable text nodes: {scanned}')
flip = [h for h in hits if h['kind'] == 'NODE_SIDE_FLIP']
rat = [h for h in hits if h['kind'] == 'RATIO']
print(f'text node that is inside GM on one side and outside on the other: {len(flip)}')
for h in flip[:40]:
    print(f'\n  {h["repo"]}/{h["pack"]} {h["path"]} #{h["i"]}')
    print('   EN:', h['en'])
    print('   CN:', h['cn'])
print(f'\nlength-ratio outliers: {len(rat)}')
for h in sorted(rat, key=lambda x: abs(x['r'] - x['med']), reverse=True)[:40]:
    print(f'\n  {h["repo"]}/{h["pack"]} {h["path"]} #{h["i"]} hidden={h["hidden"]} '
          f'r={h["r"]} med={h["med"]}')
    print('   EN:', h['en'])
    print('   CN:', h['cn'])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p10_segments.json')
json.dump(hits, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n->', out)
