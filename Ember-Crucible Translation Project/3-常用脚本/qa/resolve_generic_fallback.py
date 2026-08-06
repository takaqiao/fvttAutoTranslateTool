#!/usr/bin/env python3
"""Subtract what Babele's generic fallback will translate for free.

`validate_translations.py` counts every translatable leaf, including embedded
documents that Babele will resolve from a DIFFERENT translated pack.

Babele 2.9.1's `DocumentConverter._genericTranslationSource` calls
`runtime.translatedPackFor(documentType, data)`, which scans every registered
translated pack for one whose `hasTranslation(data)` is true. Match candidates
are `_id` -> `name` -> `sourceId`, so an embedded item merely NAMED
"Backstab" resolves against `crucible.talent`'s "Backstab" entry even with no
`compendiumSource` on it at all.

This script reports, per pack, how much of the remaining todo is covered that
way and what the true residual is.

Usage:
  python resolve_generic_fallback.py --repo <pluginRepo> [--also <otherRepo>...]
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')
# Embedded-document segments whose key is a document NAME Babele can match on.
EMBEDDED = re.compile(r'\.(items|effects|actors)\.([^.]+)')


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def translated_names(repos):
    """Every entry name that some translated pack can resolve, by doc type."""
    names = set()
    for repo in repos:
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        if not os.path.isdir(cn_dir):
            continue
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith('.json'):
                continue
            try:
                doc = load(os.path.join(cn_dir, fn))
            except Exception:
                continue
            for k, v in (doc.get('entries') or {}).items():
                if isinstance(v, dict) and isinstance(v.get('name'), str) and CJK.search(v['name']):
                    names.add(k)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--also', nargs='*', default=[])
    ap.add_argument('--reports', help='报告目录；不给则按仓库目录名推断')
    a = ap.parse_args()

    resolvable = translated_names([a.repo, *a.also])
    print(f'names resolvable from translated packs: {len(resolvable)}\n')

    # 必须只看仓库**目录名**：项目根目录本身就叫 "Ember-Crucible Translation Project"，
    # 拿整条路径判断会让 ember 仓库也命中 'Crucible'，静默去读 crucible 的清单。
    if a.reports:
        todo_dir = os.path.join(a.reports, 'todo')
    else:
        leaf = os.path.basename(os.path.normpath(a.repo))
        todo_dir = os.path.join(a.repo, '..', '5-其他内容', 'reports',
                                'crucible' if 'Crucible' in leaf else 'ember', 'todo')
    todo_dir = os.path.normpath(todo_dir)
    print(f'todo dir: {todo_dir}\n')
    if not os.path.isdir(todo_dir):
        print(f'no todo dir at {todo_dir}')
        return

    print(f"{'pack':<34}{'todo':>7}{'auto':>7}{'residual':>10}{'res chars':>11}")
    T = [0, 0, 0, 0]
    residual_out = {}
    for fn in sorted(os.listdir(todo_dir)):
        if not fn.endswith('.todo.json'):
            continue
        items = load(os.path.join(todo_dir, fn))['items']
        auto, res = [], []
        for it in items:
            m = EMBEDDED.search('.' + it['path'])
            if m and m.group(2) in resolvable:
                auto.append(it)
            else:
                res.append(it)
        chars = sum(x['chars'] for x in res)
        print(f"{fn[:-10]:<34}{len(items):>7}{len(auto):>7}{len(res):>10}{chars:>11}")
        T = [T[0] + len(items), T[1] + len(auto), T[2] + len(res), T[3] + chars]
        if res:
            residual_out[fn[:-10]] = res

    print(f"{'TOTAL':<34}{T[0]:>7}{T[1]:>7}{T[2]:>10}{T[3]:>11}")
    out = os.path.join(todo_dir, '_residual_after_fallback.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'_meta': {'todo': T[0], 'autoResolved': T[1], 'residual': T[2], 'residualChars': T[3],
                             'note': 'auto = Babele 会从其他已翻译包按名字取到译文，无需内联翻译'},
                   'packs': residual_out}, f, ensure_ascii=False, indent=2)
    print(f'\nresidual -> {out}')


main()
