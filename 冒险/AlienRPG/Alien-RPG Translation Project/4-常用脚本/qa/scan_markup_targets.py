#!/usr/bin/env python3
"""Find Chinese that leaked INSIDE Foundry markup targets/parameters.

`@Embed[JournalEntry.x.JournalEntryPage.y inline]`, `@UUID[Actor.x.Item.y]`,
`[[/skillCheck awareness 14]]` -- everything between the brackets is machinery.
Only a trailing `{label}` is visible text. A translated target silently breaks
the link or embed: the page still renders, just without the block, and every
other check passes (the path has a Chinese value, so coverage says 100%; the
markup gate only runs on values written *through* it, so legacy translations
never get inspected).

This is the mechanical form of the standing decision "方括号内的目标与参数一律照抄".

Usage:
  python scan_markup_targets.py --repo <pluginRepo> [--repo <another>] [--json <out>]
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')
# Bracketed machinery. The `{label}` that may follow is deliberately not matched.
MARKUP = re.compile(r'@[A-Za-z]+\[([^\]]*)\]|\[\[([^\]]*)\]\]')
# Chinese inside a bracket is EXPECTED in these three shapes:
#   `readaloud="..."` / `label="..."`  parameter values are visible prose
#   `[[/r 1d6#分身被摧毁]]`             everything after `#` is the roll flavor
#   `[[/item 战镐]]`                    dnd5e resolves this against the actor's
#                                       item names, which Babele has already
#                                       translated -- the Chinese form is the
#                                       one that exists at runtime
BY_DESIGN_BODY = re.compile(r'=\s*"|#')
BY_DESIGN_CMD = re.compile(r'^\[\[/(?:item|r|roll|damage|check|save)\b')


def by_design(markup: str, body: str) -> bool:
    return bool(BY_DESIGN_BODY.search(body) or BY_DESIGN_CMD.match(markup))


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        for m in MARKUP.finditer(node):
            body = m.group(1) if m.group(1) is not None else m.group(2)
            if CJK.search(body):
                out.append({'path': '.'.join(path), 'markup': m.group(0),
                            'verdict': 'by-design' if by_design(m.group(0), body) else 'BROKEN'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--json')
    a = ap.parse_args()

    all_hits = []
    for repo in a.repo:
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        if not os.path.isdir(cn_dir):
            continue
        for fn in sorted(f for f in os.listdir(cn_dir) if f.endswith('.json')):
            with open(os.path.join(cn_dir, fn), encoding='utf-8') as f:
                data = json.load(f)
            hits = []
            walk(data, [], hits)
            for h in hits:
                h['pack'] = fn
                h['repo'] = os.path.basename(os.path.normpath(repo))
            all_hits += hits
            broken = sum(1 for h in hits if h['verdict'] == 'BROKEN')
            print(f'{os.path.basename(os.path.normpath(repo))}/{fn:<38}'
                  f'{broken:>5} broken {len(hits) - broken:>5} by-design')

    bad = [h for h in all_hits if h['verdict'] == 'BROKEN']
    print(f'\nBROKEN {len(bad)}   by-design {len(all_hits) - len(bad)}')
    for h in bad[:40]:
        print(f'  {h["pack"]} :: {h["path"]}\n      {h["markup"][:120]}')
    if len(bad) > 40:
        print(f'  ... and {len(bad) - 40} more')

    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(all_hits, f, ensure_ascii=False, indent=2)
        print(f'\n-> {a.json}')
    raise SystemExit(1 if bad else 0)


main()
