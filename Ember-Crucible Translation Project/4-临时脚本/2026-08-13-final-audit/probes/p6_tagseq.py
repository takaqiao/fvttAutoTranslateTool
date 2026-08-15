#!/usr/bin/env python3
"""P6: FULL ORDERED tag sequence, EN vs CN.

Why this is not already covered: scan_markup_drift's BLOCK check is a
`Counter` of tag names -- unordered. Moving a <p> from inside
`<section class="block gamemaster">` to just after `</section>` leaves the
counter identical, so the existing gate is blind to it by construction.
scan_class_drift is likewise a multiset, and only of class-bearing tags.

Here we build the ordered open/close token stream (class attached for
section/div/ul/li/sup/span so a re-attach shows up) and diff it.

Normalisation applied (matching the house gate's own conventions):
  * empty <p></p> dropped (EN typography padding the CN never copies -- this is
    already normalised away in scan_markup_drift.tag_counter)
  * `id="..."` on headings ignored: the 1642 project-added anchors are known and
    intentional, and they do not change nesting.
False-positive modes: a translator legitimately swapping two inline spans
(word order), and paragraph merge/split. Both are printed with a diff so they
can be told apart from a real structural move.
"""
import sys, os, re, json, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs, CJK

EMPTY_P = re.compile(r'<p>\s*</p>')
TAG = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*)>')
WITH_CLASS = {'section', 'div', 'ul', 'ol', 'li', 'sup', 'span', 'p', 'h1', 'h2', 'h3', 'h4'}


def seq(html):
    html = EMPTY_P.sub('', html)
    out = []
    for slash, name, attrs in TAG.findall(html):
        n = name.lower()
        if slash:
            out.append('/' + n)
            continue
        if n in WITH_CLASS:
            m = re.search(r'class="([^"]*)"', attrs)
            out.append(f'{n}.{m.group(1)}' if m else n)
        else:
            out.append(n)
    return out


hits = []
tot = 0
for rname, repo in REPOS.items():
    for pack, rows in pairs(repo):
        for path, e, c in rows:
            if not (c and CJK.search(c)):
                continue
            if '<' not in e:
                continue
            se, sc = seq(e), seq(c)
            if not se and not sc:
                continue
            tot += 1
            if se == sc:
                continue
            same_ms = sorted(se) == sorted(sc)
            sm = difflib.SequenceMatcher(None, se, sc)
            ops = [(t, se[i1:i2], sc[j1:j2]) for t, i1, i2, j1, j2 in sm.get_opcodes()
                   if t != 'equal']
            touches_gm = any('gamemaster' in x or 'secret' in x
                             for t, a, b in ops for x in list(a) + list(b))
            hits.append({'repo': rname, 'pack': pack, 'path': path,
                         'same_multiset': same_ms, 'gm': touches_gm,
                         'ops': [[t, a, b] for t, a, b in ops],
                         'en': e, 'cn': c})

print(f'markup-bearing translated strings: {tot}')
print(f'ordered-sequence mismatches: {len(hits)}')
print(f'  same multiset (pure move/reorder): {sum(1 for h in hits if h["same_multiset"])}')
print(f'  diff touches a gamemaster/secret boundary: {sum(1 for h in hits if h["gm"])}')
for h in hits:
    print('\n' + '=' * 70)
    print(f'{"GM! " if h["gm"] else ""}{h["repo"]}/{h["pack"]}  {h["path"]}  '
          f'same_multiset={h["same_multiset"]}')
    for t, a, b in h['ops'][:8]:
        print(f'   {t:<8} EN{a}  ->  CN{b}')
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p6_tagseq.json')
json.dump(hits, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n->', out)
