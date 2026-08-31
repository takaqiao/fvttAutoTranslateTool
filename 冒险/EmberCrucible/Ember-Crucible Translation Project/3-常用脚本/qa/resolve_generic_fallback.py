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

The match is SCOPED BY DOCUMENT TYPE, and this script has to be too. Babele
2.9.1 `script/compendium/mapped-compendium.js`:

    hasTranslation(data, documentType = this.metadata?.type ?? null, runtime)
    if (documentType && this.metadata?.type && this.metadata.type !== documentType)
      return false;

and `compendium-runtime.js::translatedPackFor` filters candidate packs with
exactly that. So an embedded ITEM named "Backstab" resolves against
`crucible.talent` (type `Item`) but NOT against `crucible.rules` (type
`JournalEntry`) or an `Adventure` pack that happens to hold an adventure of the
same name. A name-only, type-blind check reports every todo as auto-resolved and
a residual of 0, which is what the two shipped reports used to say.

The pack -> documentType map comes from the UPSTREAM manifests (the translation
modules declare no packs of their own), keyed `<pkgId>.<packName>` which is
exactly the `compendium/cn/<file>.json` naming.

Usage:
  python resolve_generic_fallback.py --repo <pluginRepo> [--also <otherRepo>...]
                                     [--package <foundryPkgDir>...] [--reports <dir>]
"""
from __future__ import annotations
import argparse
import json
import os
import re

CJK = re.compile(r'[一-鿿]')
# Embedded-document segments whose key is a document NAME Babele can match on,
# and the documentType each one carries.
EMBEDDED = re.compile(r'\.(items|effects|actors)\.([^.]+)')
SEGMENT_TYPE = {'items': 'Item', 'effects': 'ActiveEffect', 'actors': 'Actor'}

# Where the upstream packages normally live; only used when --package is absent.
DEFAULT_PACKAGES = [
    os.path.expandvars(r'%LOCALAPPDATA%\FoundryVTT\Data\systems\crucible'),
    os.path.expandvars(r'%LOCALAPPDATA%\FoundryVTT\Data\modules\ember'),
]


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def pack_types(package_dirs):
    """`<pkgId>.<packName>` -> documentType, read from the upstream manifests."""
    types = {}
    for pkg in package_dirs:
        manifest = next((os.path.join(pkg, f) for f in ('system.json', 'module.json')
                         if os.path.exists(os.path.join(pkg, f))), None)
        if not manifest:
            continue
        try:
            m = load(manifest)
        except Exception:
            continue
        pid = m.get('id')
        for p in m.get('packs') or []:
            if p.get('name') and p.get('type'):
                types[f"{pid}.{p['name']}"] = p['type']
    return types


def translated_names(repos, types):
    """`entry name` -> set(documentType) it can be resolved as.

    The type is the OWNING PACK's declared type, because that is what Babele
    compares against (`this.metadata?.type`). A pack whose type is unknown is
    reported and contributes nothing -- guessing here would re-introduce the
    type-blind behaviour this function exists to remove.
    """
    names: dict[str, set] = {}
    unknown = []
    for repo in repos:
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        if not os.path.isdir(cn_dir):
            continue
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith('.json'):
                continue
            dtype = types.get(fn[:-len('.json')])
            if dtype is None:
                unknown.append(fn)
                continue
            try:
                doc = load(os.path.join(cn_dir, fn))
            except Exception:
                continue
            for k, v in (doc.get('entries') or {}).items():
                if isinstance(v, dict) and isinstance(v.get('name'), str) and CJK.search(v['name']):
                    names.setdefault(k, set()).add(dtype)
    return names, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--also', nargs='*', default=[])
    ap.add_argument('--package', nargs='*', default=None,
                    help='上游包目录（含 system.json / module.json），用来读 packs[].type；'
                         '不给则用 %%LOCALAPPDATA%%\\FoundryVTT\\Data 下的 crucible / ember')
    ap.add_argument('--reports', help='报告目录；不给则按仓库目录名推断')
    a = ap.parse_args()

    pkgs = a.package if a.package is not None else DEFAULT_PACKAGES
    types = pack_types(pkgs)
    if not types:
        print('! 一个上游 packs[].type 都没读到；没有类型就无法复现 babele 的类型闸。')
        print(f'  找过：{pkgs}')
        print('  用 --package 指向 Foundry 里的 crucible 系统目录 / ember 模块目录。')
        raise SystemExit(2)
    print(f'pack types from manifests: {len(types)}')

    resolvable, unknown = translated_names([a.repo, *a.also], types)
    if unknown:
        print(f'! 这些 cn 包在上游清单里找不到对应的 packs[].type，未计入可解析名单：'
              f'{", ".join(unknown)}')
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
            # findall + [-1]，不是 search：`search` 取的是**最外层**的
            # items/effects/actors 段，而 babele 要解析的是最深的那一层内嵌文档
            # （`actors.X.items.Y.effects.Z` 要按 ActiveEffect「Z」去查，不是按
            #  Actor「X」）。
            segs = EMBEDDED.findall('.' + it['path'])
            if segs:
                seg_key, name = segs[-1]
                need_type = SEGMENT_TYPE[seg_key]
            else:
                name = need_type = None
            if name is not None and need_type in resolvable.get(name, set()):
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
        json.dump({'_meta': {
            'todo': T[0], 'autoResolved': T[1], 'residual': T[2], 'residualChars': T[3],
            'note': 'auto = 该内嵌文档名在**同一 documentType** 的已翻译包里可解析，'
                    'babele 的类型闸（mapped-compendium.js:156-158）会放行，无需内联翻译；'
                    '其余需内联翻译。类型取自上游 packs[].type，'
                    '内嵌层取路径里**最深**的 items/effects/actors 段。',
            'packTypes': dict(sorted(types.items())),
            'packsWithUnknownType': unknown,
        }, 'packs': residual_out}, f, ensure_ascii=False, indent=2)
    print(f'\nresidual -> {out}')


main()
