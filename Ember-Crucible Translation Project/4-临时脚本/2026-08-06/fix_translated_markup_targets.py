#!/usr/bin/env python3
"""Repair Chinese that leaked into Foundry markup targets (see
qa/scan_markup_targets.py for what this class of defect is).

Method: for every CN string that has a translated bracket body, line up the
markup of the CN string with the markup of the English baseline *of the same
kind, in order*, and copy the English body back. No guessing -- if the kinds
don't line up 1:1 the string is skipped and listed.

Deliberately left alone:
  - `readaloud="..."` / `label="..."` parameter values -- visible prose
  - `[[/r ...#flavor]]` -- the part after `#` is the visible roll flavor
  - `[[/item <名称>]]` -- dnd5e resolves this against the actor's item names,
    which Babele has already translated, so the Chinese form may be the
    correct one. Flagged, never rewritten.

Emits a batch for qa/apply_translations.py, which re-checks every value against
the English markup signature before writing.

  python fix_translated_markup_targets.py --repo <repo> --pack <pack.json> --out <batch.json>
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')
CJK_RUN = re.compile(r'[一-鿿]+')
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]')
# Bodies that carry visible prose rather than an identifier.
KEEP = re.compile(r'=\s*"|#')
SKIP_CMD = re.compile(r'^\[\[/(?:item|r|roll|damage|check|save)\b')


def body_of(m: str) -> str:
    return m[m.index('[') + 1:m.rindex(']')]


def same_skeleton(cn_body: str, en_body: str) -> bool:
    """Is the CN body just the EN body with some identifiers translated?

    Each Chinese run stands in for one token, so it may not swallow whitespace.
    This is what stops the repair from eating prose: a CN body that added or
    dropped structure (a `readaloud="..."` parameter, a `和` where the English
    has ` and `) will not match and is left for a human.
    """
    pat = ''.join(re.escape(p) if i % 2 == 0 else r'[^\s\]]*'
                  for i, p in enumerate(_split_keep(cn_body)))
    return re.fullmatch(pat, en_body) is not None


def _split_keep(s: str):
    """['literal', 'CJKrun', 'literal', ...] -- even indices are literals."""
    out, last = [], 0
    for m in CJK_RUN.finditer(s):
        out.append(s[last:m.start()])
        out.append(m.group(0))
        last = m.end()
    out.append(s[last:])
    return out


def kind_of(m: str) -> str:
    if m.startswith('@'):
        return m.split('[', 1)[0].lower()
    body = m[2:-2].strip()
    return '[[' + (body.split(' ', 1)[0] if body else '')


def repairable(m: str) -> bool:
    body = body_of(m)
    if not CJK.search(body):
        return False
    if KEEP.search(body):
        return False
    if SKIP_CMD.match(m):
        return False
    return True


def repair(cn: str, en: str):
    """Return (new_cn, notes). Only rewrites bodies that are clearly identifiers."""
    cn_marks = list(MARKUP.finditer(cn))
    en_marks = [m.group(0) for m in MARKUP.finditer(en)]
    targets = [m for m in cn_marks if repairable(m.group(0))]
    if not targets:
        return cn, []

    # Positional alignment within each kind.
    en_by_kind = {}
    for m in en_marks:
        en_by_kind.setdefault(kind_of(m), []).append(m)
    cn_by_kind = {}
    for m in cn_marks:
        cn_by_kind.setdefault(kind_of(m.group(0)), []).append(m)

    notes, out, last = [], [], 0
    for m in targets:
        k = kind_of(m.group(0))
        peers = cn_by_kind[k]
        if len(peers) != len(en_by_kind.get(k, [])):
            notes.append(f'kind {k}: CN {len(peers)} vs EN {len(en_by_kind.get(k, []))} -- skipped')
            continue
        idx = peers.index(m)
        want = en_by_kind[k][idx]
        if not same_skeleton(body_of(m.group(0)), body_of(want)):
            notes.append(f'{m.group(0)[:70]} !~ {want[:70]} -- skipped (shape differs)')
            continue
        out.append(cn[last:m.start()])
        out.append(want)
        last = m.end()
        notes.append(f'{m.group(0)[:60]} -> {want[:60]}')
    out.append(cn[last:])
    return ''.join(out), notes


def walk(cn, en, path, batch, log):
    if isinstance(cn, dict):
        for k, v in cn.items():
            walk(v, en.get(k) if isinstance(en, dict) else None, path + [str(k)], batch, log)
    elif isinstance(cn, list):
        for i, v in enumerate(cn):
            walk(v, en[i] if isinstance(en, list) and i < len(en) else None, path + [str(i)], batch, log)
    elif isinstance(cn, str) and isinstance(en, str):
        if not any(repairable(m.group(0)) for m in MARKUP.finditer(cn)):
            return
        new, notes = repair(cn, en)
        if new != cn:
            batch['.'.join(path)] = new
        log.append({'path': '.'.join(path), 'notes': notes})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    en = json.load(open(os.path.join(a.repo, 'compendium', 'en', a.pack), encoding='utf-8'))
    cn = json.load(open(os.path.join(a.repo, 'compendium', 'cn', a.pack), encoding='utf-8'))

    batch, log = {}, []
    walk(cn.get('entries', {}), en.get('entries', {}), [], batch, log)

    for e in log:
        print(e['path'])
        for n in e['notes']:
            print('   ', n)
    print(f'\nrepairs: {len(batch)} strings')
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    print('->', a.out)


main()
