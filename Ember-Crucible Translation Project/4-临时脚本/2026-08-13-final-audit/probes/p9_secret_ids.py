#!/usr/bin/env python3
"""P9: Foundry `<section class="secret">` handling.

Foundry's secret-reveal UI keys off the section's `id` (HTMLSecretConfiguration
matches `section.secret[id]`). Losing/renaming that id makes the eye toggle
unable to persist a reveal, and a `revealed` class present on one side but not
the other flips who sees the text. Neither is visible to a class multiset gate
(the class token is identical) nor to the anchor gate (id="" is excluded there
as a project-added heading anchor).
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs, CJK

SEC = re.compile(r'<section[^>]*class="([^"]*)"[^>]*>')
SECFULL = re.compile(r'<section[^>]*>')

rows = []
for rname, repo in REPOS.items():
    for pack, prs in pairs(repo):
        for path, e, c in prs:
            if 'secret' not in e.lower() and (not c or 'secret' not in c.lower()):
                continue
            en_secs = [m.group(0) for m in SECFULL.finditer(e) if 'secret' in m.group(0).lower()]
            cn_secs = [m.group(0) for m in SECFULL.finditer(c or '') if 'secret' in m.group(0).lower()]
            if not en_secs and not cn_secs:
                continue
            rows.append((rname, pack, path, en_secs, cn_secs))

print(f'strings with a <section ... secret ...>: {len(rows)}')
bad = 0
for rname, pack, path, es, cs in rows:
    if es == cs:
        continue
    bad += 1
    print(f'\n{rname}/{pack} {path}')
    print('  EN:', es)
    print('  CN:', cs)
print(f'\nopening-tag mismatches: {bad}')

print('\n--- sample of the opening tags found ---')
seen = set()
for rname, pack, path, es, cs in rows:
    for t in es:
        if t not in seen:
            seen.add(t)
            print('  EN', t)
print(f'distinct EN secret <section> opening tags: {len(seen)}')

# revealed class anywhere?
print('\n--- `revealed` occurrences ---')
for rname, repo in REPOS.items():
    for pack, prs in pairs(repo):
        for path, e, c in prs:
            if 'revealed' in e or (c and 'revealed' in c):
                print(f'  {rname}/{pack} {path}  EN={"revealed" in e} CN={bool(c) and "revealed" in c}')
