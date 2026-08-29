#!/usr/bin/env python3
"""Find (and optionally strip) foreign-script contamination in translations.

Machine-translation passes occasionally splice fragments of unrelated scripts
into Chinese output — Armenian, Thai, Cyrillic, Arabic, Devanagari. They are
invisible in a diff and survive every structural check, but players see them.

Real examples caught in this project:
  crucible lang : `掘穴 շարժ作`            (Armenian "շարժ" = movement)
  ember  campaign: `卡罗ว์ Carrow` ×15      (Thai "ว์" glued to a proper noun)
  ember  campaign: `发现一处 недавно被某头`  (untranslated Russian word)

`--fix` deletes bare contamination. Cases where a foreign fragment REPLACED a
real word (the Russian one) cannot be auto-repaired and are reported for manual
translation instead.

Usage:
  python scan_foreign_script.py --repo <pluginRepo> [--repo <another>] [--fix]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re

# Scripts that have no business appearing in a zh-Hans translation of an
# English game. Latin, CJK, kana, punctuation and symbols are all allowed.
FOREIGN = re.compile(
    r'['
    r'Ѐ-ӿ'   # Cyrillic
    r'԰-֏'   # Armenian
    r'֐-׿'   # Hebrew
    r'؀-ۿ'   # Arabic
    r'ऀ-ॿ'   # Devanagari
    r'฀-๿'   # Thai
    r'Ⴀ-ჿ'   # Georgian
    r']+'
)
# A run of 3+ foreign letters is a whole word that was never translated;
# 1-2 characters is stray noise glued onto otherwise-correct output.
WORD_LEN = 3


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def walk(node, path, hits):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], hits)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], hits)
    elif isinstance(node, str) and FOREIGN.search(node):
        for m in FOREIGN.finditer(node):
            hits.append({'path': '.'.join(path), 'frag': m.group(0),
                         'context': node[max(0, m.start() - 40):m.end() + 40],
                         'word': len(m.group(0)) >= WORD_LEN})


def strip_in(node, path, fixed, manual):
    """Return node with short foreign fragments removed."""
    if isinstance(node, dict):
        return {k: strip_in(v, path + [k], fixed, manual) for k, v in node.items()}
    if isinstance(node, list):
        return [strip_in(v, path + [str(i)], fixed, manual) for i, v in enumerate(node)]
    if isinstance(node, str) and FOREIGN.search(node):
        out = node
        for m in list(FOREIGN.finditer(node)):
            if len(m.group(0)) >= WORD_LEN:
                manual.append({'path': '.'.join(path), 'frag': m.group(0),
                               'context': node[max(0, m.start() - 40):m.end() + 40]})
                continue
            out = out.replace(m.group(0), '')
            fixed.append({'path': '.'.join(path), 'frag': m.group(0)})
        return out
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--fix', action='store_true')
    a = ap.parse_args()

    total_hits = total_fixed = 0
    manual_all = []
    for repo in a.repo:
        for pattern in ('compendium/cn/*.json', 'lang/cn.json'):
            for f in sorted(glob.glob(os.path.join(repo, pattern))):
                try:
                    doc = load(f)
                except Exception:
                    continue
                hits = []
                walk(doc, [], hits)
                if not hits:
                    continue
                total_hits += len(hits)
                print(f'\n{os.path.relpath(f, repo)}  —  {len(hits)} 处')
                for h in hits[:6]:
                    kind = '整词未译' if h['word'] else '杂散字符'
                    print(f"  [{kind}] {h['path']}")
                    print(f"      …{h['context']}…")
                if len(hits) > 6:
                    print(f'  … 另有 {len(hits) - 6} 处')

                if a.fix:
                    fixed, manual = [], []
                    doc = strip_in(doc, [], fixed, manual)
                    if fixed:
                        with open(f, 'w', encoding='utf-8') as fh:
                            json.dump(doc, fh, ensure_ascii=False, indent=2)
                            fh.write('\n')
                        total_fixed += len(fixed)
                        print(f'  -> 清除杂散字符 {len(fixed)} 处')
                    manual_all.extend([{**m, 'file': os.path.relpath(f, repo)} for m in manual])

    print(f'\n合计发现 {total_hits} 处')
    if a.fix:
        print(f'自动清除 {total_fixed} 处')
        if manual_all:
            print(f'\n需人工翻译（整词被外语替换，无法自动修复）: {len(manual_all)}')
            for m in manual_all:
                print(f"  {m['file']}  {m['path']}")
                print(f"      {m['frag']}  ←  …{m['context']}…")


main()
