#!/usr/bin/env python3
"""Fill the dnd5e twin pack `ember.adventure` from the crucible pack by exact-match TM.

The two campaigns ship the same adventure for two systems; the journal prose is
byte-identical. So most of the twin needs no translator at all -- only a lookup.
As the crucible side reached 98%, TM coverage of the twin rose 88% -> **98%**
(13,114 leaves / 8.17 M chars). What is left is 1,213 leaves / 142 K chars that
genuinely exist only in the dnd5e version (mostly `actors`).

This script does NOT write `compendium/cn`. It emits a batch file, and
`apply_translations.py` lands it -- so the fill goes through the exact same
three gates as human translation (English drift / no Chinese / markup mismatch)
instead of getting a private, unchecked write path.

  python fill_twin.py [--out <batch.json>] [--report <json>]
  python "..\\qa\\apply_translations.py" --repo <repo> --pack ember.adventure.json --batch <batch.json> --dry

Two things it is careful about:

* **Same English, different Chinese.** A naive `dict[en] = cn` takes whichever
  leaf was walked first. Stage 23 established that `.name` fields follow three
  different conventions depending on where they sit (item name / action name /
  effect name), so the lookup is keyed on `(结构路径, 英文)` -- see `shape_of` --
  and falls back to the English alone. Keying on the LAST SEGMENT is not enough:
  it collapses all three `.name` populations back together, which is the very
  mistake stage 23 warned about. Switching from last-segment to structural path
  cut ambiguous keys from 697 to 530. Where a key still has competing Chinese,
  the majority wins and the conflict is reported rather than silently resolved.
* **Markup.** A TM hit whose markup multiset differs from the twin's English is
  dropped, not filled. The two systems' pages differ exactly where the rules
  differ (`[[/skillCheck …]]` vs `[[/check …]]`), and those are the leaves where
  copying the crucible Chinese would be wrong.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SRC_PACK = "ember.crucible-adventure.json"
DST_PACK = "ember.adventure.json"
CJK = re.compile(r'[一-鿿]')

_spec = importlib.util.spec_from_file_location(
    "_apply", os.path.join(P, "3-常用脚本", "qa", "apply_translations.py"))
_apply = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_apply)
markup_signature = _apply.markup_signature


# Structural segments; everything else in a path is an entity/action/effect id.
STRUCT = {
    'journals', 'pages', 'actors', 'items', 'actions', 'effects', 'folders',
    'macros', 'scenes', 'tables', 'results', 'text', 'content', 'system',
    'description', 'public', 'private', 'name', 'biography', 'prototypeToken',
    'overview', 'exposition', 'summary', 'terrain', 'gamemaster', 'subtitle',
    'pronunciation', 'caption', 'label', 'outcomes', 'notes', 'journal',
}


def shape_of(path):
    """The structural skeleton of a path, with entity names dropped.

    Stage 23 established that `.name` obeys three different conventions
    depending on where it sits -- `items.X.name` (item name),
    `items.X.actions.<id>.name` (action name) and `effects[].name` are separate
    populations, and grouping them together yields the wrong majority. Keying TM
    on the last segment alone ('name') does exactly that, so the key is the
    filtered structural path instead:

        actors.Kalasak the Cutter.items.Caustic Phial.actions.causticPhial.name
        -> actors.items.actions.name
    """
    return '.'.join(p for p in path.split('.')
                    if p in STRUCT or p.isdigit())


def leaves(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            leaves(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            leaves(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                   path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(P, "1-Ember汉化插件"))
    ap.add_argument('--out', default=os.path.join(
        P, "5-其他内容", "reports", "ember", "twin_fill.batch.json"))
    ap.add_argument('--report', default=os.path.join(
        P, "5-其他内容", "reports", "ember", "twin_fill.report.json"))
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')

    def load(d, f):
        p = os.path.join(d, f)
        return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}

    src = []
    leaves(load(en_dir, SRC_PACK).get('entries', {}),
           load(cn_dir, SRC_PACK).get('entries', {}), [], src)

    # (structural shape, english) -> Counter of candidate Chinese
    shaped = defaultdict(Counter)
    plain = defaultdict(Counter)
    for path, e, c in src:
        if not (c and CJK.search(c)):
            continue
        shaped[(shape_of(path), e.strip())][c] += 1
        plain[e.strip()][c] += 1

    conflicts = [{'key': list(k), 'variants': len(v),
                  'chosen': v.most_common(1)[0][0][:120],
                  'others': [x[0][:120] for x in v.most_common()[1:4]]}
                 for k, v in shaped.items() if len(v) > 1]

    dst = []
    leaves(load(en_dir, DST_PACK).get('entries', {}),
           load(cn_dir, DST_PACK).get('entries', {}), [], dst)

    batch = {}
    stat = Counter()
    dropped_markup = []
    for path, e, c in dst:
        if c and CJK.search(c):
            stat['already'] += 1
            continue
        key = e.strip()
        shape = shape_of(path)
        hit = shaped.get((shape, key)) or plain.get(key)
        if not hit:
            stat['no_tm'] += 1
            continue
        cn = hit.most_common(1)[0][0]
        if markup_signature(e) != markup_signature(cn):
            stat['markup_drop'] += 1
            if len(dropped_markup) < 200:
                dropped_markup.append({'path': path, 'en': e[:160], 'cn': cn[:160]})
            continue
        stat['filled'] += 1
        stat['shaped' if shaped.get((shape, key)) else 'plain'] += 1
        batch[path] = cn

    json.dump(batch, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump({'stat': dict(stat), 'conflicts': conflicts[:500],
               'conflict_total': len(conflicts), 'dropped_markup': dropped_markup},
              open(a.report, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print(f'TM 条目  按路径形状 {len(shaped)}   仅按英文 {len(plain)}')
    print(f'  同键多译（取多数，已记进报告）  {len(conflicts)}')
    print(f'\n孪生包叶子 {len(dst)}')
    print(f'  已有中文        {stat["already"]:>7}')
    print(f'  TM 命中并填充   {stat["filled"]:>7}   （形状键 {stat["shaped"]} / 英文键 {stat["plain"]}）')
    print(f'  标记不符，弃填  {stat["markup_drop"]:>7}   ← 两个系统规则不同的地方，不能照搬')
    print(f'  无 TM，需真翻   {stat["no_tm"]:>7}')
    print(f'\nbatch  -> {a.out}')
    print(f'report -> {a.report}')
    print('\n下一步（务必先 --dry）：')
    print(f'  python "{P}\\3-常用脚本\\qa\\apply_translations.py" '
          f'--repo "{a.repo}" --pack {DST_PACK} --batch "{a.out}" --dry')


if __name__ == '__main__':
    main()
