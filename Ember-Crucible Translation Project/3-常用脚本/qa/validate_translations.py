#!/usr/bin/env python3
"""Validate a plugin's translations against the English baseline.

The English baseline in `compendium/en/` is produced by interpreting the very
mapping Babele is given at runtime, so "does this CN path exist in the EN file"
is the same question as "will Babele find this translation". That makes this
both the QA gate and the source of the remaining-work numbers.

Reports per pack:
  covered / todo      translatable leaves with / without a Chinese value
  orphans             CN values at paths the EN baseline does not have -> dead
                      translation, Babele will never apply it
  residue             CN values that are byte-identical to the English -> never
                      actually translated, just copied

Writes a machine-readable todo list so translation batches can be driven from
it directly.

Usage:
  python validate_translations.py --repo <pluginRepo> [--out <reportDir>] [--quiet]
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')
# Paths that are structural keys rather than prose (folders map EN->CN names).
PLAIN_MAP_FIELDS = {'folders', 'categories'}


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def dump(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(o, f, ensure_ascii=False, indent=2)
        f.write('\n')


def textlen(s: str) -> int:
    return len(TAG.sub(' ', s))


def walk(en, cn, path, todo, covered, residue, stats):
    """Walk the EN structure; CN is looked up at the same path."""
    if isinstance(en, dict):
        for k, v in en.items():
            sub_cn = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub_cn, path + [k], todo, covered, residue, stats)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub_cn = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub_cn, path + [str(i)], todo, covered, residue, stats)
    elif isinstance(en, str):
        if not en.strip():
            return
        stats['leaves'] += 1
        stats['chars'] += textlen(en)
        if isinstance(cn, str) and CJK.search(cn):
            covered[0] += 1
            if cn.strip() == en.strip():
                residue.append({'path': '.'.join(path), 'text': en[:160]})
        else:
            todo.append({'path': '.'.join(path), 'en': en, 'chars': textlen(en)})


def find_orphans(en, cn, path, out):
    """CN values sitting at paths the EN baseline does not define."""
    if isinstance(cn, dict):
        en_d = en if isinstance(en, dict) else {}
        for k, v in cn.items():
            if k.startswith('_'):            # _legacyActions etc: parked on purpose
                continue
            if k not in en_d:
                n = count_strings(v)
                if n:
                    out.append({'path': '.'.join(path + [k]), 'strings': n})
                continue
            find_orphans(en_d[k], v, path + [k], out)


def count_strings(v) -> int:
    if isinstance(v, str):
        return 1 if v.strip() else 0
    if isinstance(v, dict):
        return sum(count_strings(x) for x in v.values())
    if isinstance(v, list):
        return sum(count_strings(x) for x in v)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    out_dir = a.out or os.path.join(a.repo, '..', '5-其他内容', 'reports')

    files = sorted(f for f in os.listdir(en_dir) if f.endswith('.json') and not f.startswith('_'))

    # Clear stale todo files: a pack that reached 100% must not leave last
    # run's list behind for the next batch to pick up.
    todo_dir = os.path.join(out_dir, 'todo')
    if os.path.isdir(todo_dir):
        for stale in os.listdir(todo_dir):
            if stale.endswith('.todo.json'):
                os.remove(os.path.join(todo_dir, stale))
    report = {}
    grand = dict(leaves=0, chars=0, covered=0, todo=0, todo_chars=0, orphans=0, residue=0)

    print(f"{'pack':<34}{'leaves':>8}{'done':>8}{'cov%':>6}{'todo':>7}{'todo chars':>12}{'orphan':>8}{'residue':>8}")
    for fn in files:
        en = load(os.path.join(en_dir, fn))
        cn_path = os.path.join(cn_dir, fn)
        cn = load(cn_path) if os.path.exists(cn_path) else {'entries': {}}

        todo, residue, orphans = [], [], []
        covered = [0]
        stats = dict(leaves=0, chars=0)
        walk(en.get('entries', {}), cn.get('entries', {}), [], todo, covered, residue, stats)
        walk(en.get('folders', {}), cn.get('folders', {}), ['(folders)'], todo, covered, residue, stats)
        find_orphans(en.get('entries', {}), cn.get('entries', {}), [], orphans)

        pct = 100.0 * covered[0] / stats['leaves'] if stats['leaves'] else 0.0
        todo_chars = sum(t['chars'] for t in todo)
        print(f"{fn:<34}{stats['leaves']:>8}{covered[0]:>8}{pct:>5.0f}%{len(todo):>7}"
              f"{todo_chars:>12}{len(orphans):>8}{len(residue):>8}")

        report[fn] = {
            'leaves': stats['leaves'], 'chars': stats['chars'],
            'covered': covered[0], 'coverage_pct': round(pct, 1),
            'todo': len(todo), 'todo_chars': todo_chars,
            'orphans': len(orphans), 'orphan_paths': orphans[:80],
            'residue': len(residue), 'residue_samples': residue[:40],
            'cn_file_present': os.path.exists(cn_path),
        }
        grand['leaves'] += stats['leaves']; grand['chars'] += stats['chars']
        grand['covered'] += covered[0]; grand['todo'] += len(todo)
        grand['todo_chars'] += todo_chars; grand['orphans'] += len(orphans)
        grand['residue'] += len(residue)

        if todo:
            dump(os.path.join(out_dir, 'todo', f'{fn[:-5]}.todo.json'),
                 {'_meta': {'pack': fn, 'todo': len(todo), 'chars': todo_chars},
                  'items': todo})

    pct = 100.0 * grand['covered'] / grand['leaves'] if grand['leaves'] else 0
    print(f"{'TOTAL':<34}{grand['leaves']:>8}{grand['covered']:>8}{pct:>5.0f}%{grand['todo']:>7}"
          f"{grand['todo_chars']:>12}{grand['orphans']:>8}{grand['residue']:>8}")

    name = os.path.basename(os.path.normpath(a.repo))
    dump(os.path.join(out_dir, f'validate.{name}.json'), {'总计': grand, 'packs': report})
    print(f"\nreport -> {os.path.normpath(os.path.join(out_dir, f'validate.{name}.json'))}")
    print(f"todo    -> {os.path.normpath(os.path.join(out_dir, 'todo'))}/")


main()
