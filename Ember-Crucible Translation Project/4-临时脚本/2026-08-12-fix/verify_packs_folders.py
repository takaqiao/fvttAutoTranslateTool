#!/usr/bin/env python3
"""Gate for a babele `<packageId>._packs-folders.json` payload.

`qa/apply_translations.py` cannot check this file: its three gates all start
from an English baseline under `compendium/en/<pack>.json` walked through
`entries.<doc>....`, and a `_packs-folders` file is synthetic -- it has no
document entries and no English counterpart in the extractor's output.

So this replicates the checks that actually matter for it, straight off the
babele 2.9.1 source:

  * translation-loader.js#translationSourcesByCollection
        collection = basename minus '.json'
  * pack-folder-translations-catalog.js#isFolderTranslationCollection
        collection must end with '._packs-folders'
  * export/pack-folders-export.js#data
        payload shape is exactly {"entries": {<folder name>: <translation>}}
  * compendium/folder-translations.js#translateSystemPackFolders
        lookup is an exact match on the *untranslated* folder name

and then does the thing no format check can do: compares every key against the
folder names the live system actually declares, so a typo or an upstream rename
fails loudly instead of silently translating nothing.

Usage:
  python verify_packs_folders.py --file <batch.json> --package <foundry 包目录>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

CJK = re.compile(r'[一-鿿]')
MARKUP = re.compile(r'[<>{}\[\]]|@[A-Za-z]+')
SUFFIX = '._packs-folders'


def declared_folder_names(package_dir: str) -> list[str]:
    """Every folder name in the package manifest's `packFolders` tree.

    Foundry materializes these as Compendium-type Folder documents; they are
    exactly the names babele looks up.
    """
    for manifest in ('system.json', 'module.json'):
        path = os.path.join(package_dir, manifest)
        if os.path.exists(path):
            break
    else:
        sys.exit(f'no system.json / module.json under {package_dir}')

    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)

    names: list[str] = []

    def walk(nodes):
        for node in nodes or []:
            name = node.get('name')
            if name:
                names.append(name)
            walk(node.get('folders'))

    walk(data.get('packFolders'))
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--package', required=True)
    args = ap.parse_args()

    rejects: list[str] = []
    base = os.path.basename(args.file)
    collection = base[:-len('.json')] if base.endswith('.json') else base

    print(f'file       : {args.file}')
    print(f'collection : {collection}   (babele: basename minus .json)')

    # utf-8-sig, same as apply_translations.py: PowerShell-written batches carry a BOM.
    with open(args.file, encoding='utf-8-sig') as fh:
        raw = fh.read()
    payload = json.loads(raw)

    if raw.startswith('﻿'):
        rejects.append('BOM present (babele fetches with the browser JSON parser, which rejects it)')

    if not collection.endswith(SUFFIX):
        rejects.append(f'collection {collection!r} does not end with {SUFFIX!r} '
                       '-> PackFolderTranslationsCatalog will ignore this file')

    if set(payload) - {'entries'}:
        rejects.append(f'unexpected top-level keys: {sorted(set(payload) - {"entries"})} '
                       '(exporter emits only "entries")')

    entries = payload.get('entries')
    if not isinstance(entries, dict) or not entries:
        rejects.append('no non-empty "entries" object')
        entries = {}

    declared = declared_folder_names(args.package)
    print(f'package    : {args.package}')
    print(f'declared   : {len(declared)} folder name(s) in packFolders -> {declared}')
    print()

    for key, value in entries.items():
        if key not in declared:
            rejects.append(f'key {key!r} is not a folder name this package declares '
                           '-> would never match at runtime')
        if not isinstance(value, str) or not value.strip():
            rejects.append(f'{key!r}: empty translation')
            continue
        if MARKUP.search(value):
            rejects.append(f'{key!r}: markup characters in a folder name -> {value!r}')
        if '\n' in value or '\t' in value:
            rejects.append(f'{key!r}: newline/tab in a folder name -> {value!r}')
        if CJK.search(key):
            rejects.append(f'key {key!r} is Chinese; keys must be the untranslated name')

    missing = [name for name in declared if name not in entries]
    if missing:
        rejects.append(f'folder name(s) declared but not covered: {missing}')

    translated = [k for k, v in entries.items() if CJK.search(str(v))]
    passthrough = [k for k, v in entries.items() if not CJK.search(str(v))]

    print(f'entries    : {len(entries)}')
    print(f'translated : {len(translated)} -> ' +
          ', '.join(f'{k} -> {entries[k]}' for k in translated))
    print(f'passthrough: {len(passthrough)} -> ' +
          (', '.join(f'{k} -> {entries[k]}' for k in passthrough) or '(none)'))
    print(f'coverage   : {len(entries)}/{len(declared)} declared folder names')
    print()

    if rejects:
        print(f'rejected {len(rejects)}:')
        for reject in rejects:
            print(f'  - {reject}')
        return 1

    print('applied 0 / rejected 0  (dry run: writes nothing; target file is placed by 主控)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
