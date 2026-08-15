#!/usr/bin/env python3
"""Scan HTML/enricher ATTRIBUTE VALUES for player-visible English that never got translated.

  python scan_attr_text.py --repo <repo> [--repo <repo2>] [--out report.json] [--show N]

The blind spot this covers
--------------------------
Every existing attribute-aware check compares the English and the Chinese as a
*multiset* (`scan_class_drift.py` on `class`, `scan_markup_drift.py` on tag
names). A multiset comparison can only see that a translation BROKE something.
It is structurally incapable of seeing that a translation never happened: if the
Chinese leaf copies `data-tooltip="Page"` verbatim out of the English, the two
multisets are equal and every gate reports 0.

Meanwhile the text living inside those attributes is on screen:

* `data-tooltip` / `data-tooltip-text` -- the hover tooltip of a content link.
* `label=` inside `@Embed[Actor.x label="..."]` -- the caption printed above an
  embedded stat block. PROJECT.md already rules that the `{label}` half of
  `@UUID[target]{label}` is translated; an @Embed caption is the same thing
  spelled differently, and the library itself proves the intent -- one page
  writes `label="烽灯旅巡逻者"` while nine sibling pages leave it English.
* `readaloud=` inside `@Embed[...]` -- box text the GM reads to the table.
  (Already 48/48 translated; kept in the allowlist so a regression is caught.)

What is deliberately NOT reported
---------------------------------
* Mechanical attributes (`class`, `id`, `href`, `src`, `style`, `data-uuid`,
  `data-system`, `data-type`, `colspan`, `draggable`, ...) -- see MECHANICAL.
* **Anything inside a `[[/command ...]]` body.** PROJECT.md hard rule: inline
  command bodies are copied byte for byte. `[[/item Actor.x.Item.y
  activity="Befoul Waters"]]{投掷格布林}` is CORRECT as it stands -- `activity=`
  is a resolver key, only the trailing `{label}` is translated. These are
  counted and reported under `resolver_verbatim` so the number is visible, never
  under `findings`.
* Values that are identifiers / URLs / numbers / booleans.

Buckets in the output
---------------------
* `PROSE`       -- >=2 English words, at least one of them an ordinary English
                   word. A real omission.
* `SINGLE_WORD` -- exactly one English word and it is ordinary English. Real.
* `PROPER_NOUN` -- every token is a name the English corpus never uses in lower
                   case (`Vesk`, `Rakavi`, `Liestra Grann`). Listed separately:
                   the project writes proper nouns bilingually, so English here
                   *may* be intentional. Judge these by hand.

Two further attribute-level defects are reported alongside, because they live in
the same blind spot and cost nothing to check here:

* `broken_attr_names` -- the attribute NAME itself got translated
  (`target="_blank"` -> `目标="_blank"`). The browser ignores the unknown
  attribute, so whatever it controlled is silently gone. A multiset comparison
  cannot see this: after the rename the two sides no longer share a key.
* `attr_name_drift` -- per leaf, the multiset of attribute NAMES in the Chinese
  differs from the English. `scan_class_drift.py` only compares `class` values;
  an attribute that vanished entirely is invisible to it.

  ⚠ 2026-08-13（第九轮）加了**一条**白名单：`id` 的**净增加**不报。
  第八轮为了修 `@UUID[...#锚点]` 断链（标题一译成中文，Foundry 用来解析锚点的
  slug 就变了，234 处锚点链接全断），有意给 112 叶的 `<hN>` 补了显式 `id=`
  （PROJECT.md §8 2026-08-13f）。那是**功能性添加**，不是漂移，却让本项由 0 变 113。
  只放行净增加：`id` 的净**减少**照报不误 —— 丢了 id 就是丢了锚点，那才是缺陷。
  放行掉的数量会照常打印出来（不做静默全绿），`--strict-id` 可以关掉白名单。

The "ordinary English word" test is derived from the data, not hardcoded: a
token counts as ordinary if the English baseline uses it in lower case at least
`--vocab-min` times. Proper nouns essentially never appear lower-cased.

Coverage denominator: the report carries `ground_truth`, the count of every
player-visible attribute occurrence on the ENGLISH side, so the finding count
can be read as a rate rather than an absolute.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --------------------------------------------------------------------------
# Attribute classification. The names below were chosen by first surveying what
# actually occurs in the two repos (4-临时脚本/2026-08-12-audit3/raw_attr_survey.py),
# not by guessing. Names with zero occurrences today are kept so that an upstream
# release that starts using them does not sail through unnoticed.
# --------------------------------------------------------------------------

# Player-visible free text. Report English-only values here.
VISIBLE = {
    'data-tooltip',       # 16 in EN today
    'data-tooltip-text',  # 14
    'data-tooltip-html',  # 0 -- future proofing
    'label',              # 44 (inside @Embed only; see ENRICHER_TEXT)
    'readaloud',          # 48 (inside @Embed only)
    'title',              # 0
    'alt',                # 0
    'aria-label',         # 0
    'placeholder',        # 0
    'data-label',         # 0
    'caption',            # 0 (string form; boolean form filtered by value test)
    'summary',            # 0
}

# Identifiers, geometry, styling, wiring. Never translated.
MECHANICAL = {
    'class', 'classes', 'id', 'href', 'src', 'style', 'type', 'name', 'rel',
    'target', 'draggable', 'start', 'colspan', 'rowspan', 'width', 'height',
    'data-uuid', 'uuid', 'data-type', 'data-id', 'data-link', 'data-anchor',
    'data-system', 'data-colwidth', 'data-start', 'data-end', 'data-dobid',
    'data-index-in-node', 'data-path-to-node', 'count', 'rollable', 'cite',
    'inline', 'format', 'passive', 'tool', 'group', 'activity',
}

# Inside an @Embed[...] body only these two carry display text; the rest of the
# bracket is a target/flag and is copied verbatim.
ENRICHER_TEXT = {'label', 'readaloud'}

TAG = re.compile(r'<[a-zA-Z][\w:-]*((?:[^>"\']|"[^"]*"|\'[^\']*\')*?)/?>')
EMBED = re.compile(r'@[Ee]mbed\[((?:[^\]"]|"[^"]*")*)\]')
INLINE_CMD = re.compile(r'\[\[/((?:[^\]"]|"[^"]*")*)\]\]')
# ⚠ 第三个分支（无引号值）是 2026-08-13 补的。Foundry 的 enricher 配置解析
# (`parseEmbedConfig`) 接受 `label=Squish` 这种不带引号的写法，而只认引号的正则
# 会把它整个排除在定义域之外 —— 既不进 ground_truth，也永远不会被报成缺漏。
# 实测本库正有 2 处（`@Embed[Actor.LUptsqBgGJVWcg9v label=Squish]`），
# 于是分母显示 122 而真值是 124，那 2 处 100% 静默漏检。
ATTR = re.compile(
    r'(?<![\w-])([a-zA-Z_][\w:-]*)\s*=\s*'
    r'("([^"]*)"|\'([^\']*)\'|([^\s"\'<>\]]+))')
# An attribute name is ASCII by definition; anything else is a translated name.
BAD_NAME = re.compile(r'([^\s<>=/"\']*[^\x00-\x7f][^\s<>=/"\']*)\s*=\s*"([^"]*)"')
# `@Ref[target]{label}` / `&Reference[target]{label}` / `&amp;Reference[…]{…}`
ENRICHER_LABEL = re.compile(
    r'(@[A-Za-z]+|&(?:amp;)?[A-Za-z]+)\[((?:[^\]"]|"[^"]*")*)\]\{([^}]*)\}')

CJK = re.compile(r'[㐀-䶿一-鿿豈-﫿]')
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
LOWER_TOKEN = re.compile(r"\b[a-z][a-z'’-]{1,}\b")

# A value that is plainly not prose.
IDENTIFIERISH = re.compile(r'''^(
      \s*
    | [-+]?\d+(\.\d+)?\s*(px|em|rem|%|pt)?
    | (true|false|null|undefined)
    | https?://\S+
    | [\w./-]*\.(webp|png|jpg|jpeg|svg|gif|json|mjs|js|css)
    | [A-Za-z0-9_-]*\d[A-Za-z0-9_-]*
    | [a-z0-9]+(-[a-z0-9]+)+
    | [\w-]+(\.[\w-]+)+
)\s*$''', re.X | re.I)


def walk(node, path, out):
    """Collect every string leaf. Arrays ARE descended -- prune_dead/fill_missing
    were both bitten by walking dicts and strs only (PROJECT.md 3.6)."""
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


def load_entries(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh).get('entries', {})


def packs_of(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith('.json') and not f.startswith('_'))


def attr_value(m):
    """The value of an ATTR match: double-quoted, single-quoted, or bare."""
    for g in (3, 4, 5):
        if m.group(g) is not None:
            return m.group(g)
    return ''


def extract(text):
    """Yield (attr_name, value, origin) for every attribute in one leaf.

    origin is 'html' | 'embed' | 'inline'. Inline-command attributes are emitted
    so they can be counted, but callers must never treat them as translatable.
    """
    out = []
    for m in INLINE_CMD.finditer(text):
        for am in ATTR.finditer(m.group(1)):
            v = attr_value(am)
            out.append((am.group(1).lower(), v, 'inline'))
    for m in EMBED.finditer(text):
        for am in ATTR.finditer(m.group(1)):
            v = attr_value(am)
            out.append((am.group(1).lower(), v, 'embed'))
    for m in TAG.finditer(text):
        for am in ATTR.finditer(m.group(1)):
            v = attr_value(am)
            out.append((am.group(1).lower(), v, 'html'))
    return out


def translatable(name, origin):
    if origin == 'inline':
        return False                       # inline command bodies are verbatim
    if name in MECHANICAL:
        return False
    if origin == 'embed' and name not in ENRICHER_TEXT:
        return False
    return name in VISIBLE


def build_vocab(repos, vocab_min):
    """Ordinary-English vocabulary, learned from the English baseline."""
    c = Counter()
    for repo in repos:
        d = os.path.join(repo, 'compendium', 'en')
        for f in packs_of(d):
            leaves = []
            walk(load_entries(os.path.join(d, f)), [], leaves)
            for _, s in leaves:
                c.update(LOWER_TOKEN.findall(s))
    return {w for w, n in c.items() if n >= vocab_min}


def classify(value, vocab):
    """-> None if the value needs no translation, else a bucket name."""
    if CJK.search(value):
        return None
    if IDENTIFIERISH.match(value):
        return None
    words = WORD.findall(value)
    if not words:
        return None
    ordinary = [w for w in words if w.lower() in vocab]
    if not ordinary:
        return 'PROPER_NOUN'
    if len(words) >= 2:
        return 'PROSE'
    return 'SINGLE_WORD'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True,
                    help='repeatable; each is a plugin repo containing compendium/{en,cn}')
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=40)
    ap.add_argument('--vocab-min', type=int, default=3)
    ap.add_argument('--strict-id', action='store_true',
                    help='连 id= 的净增加也报（关掉 2026-08-13 的锚点白名单）')
    a = ap.parse_args()

    vocab = build_vocab(a.repo, a.vocab_min)

    findings, proper, broken, drift, labels = [], [], [], [], []
    id_added = Counter()         # 白名单放行的 id= 净增加，按包计数（仍然要可见）
    seen_attr = Counter()        # cn side, every attribute name
    gt = Counter()               # en side, player-visible attributes only
    cn_seen = Counter()          # cn side, player-visible attributes only (MEASURED)
    resolver = Counter()

    for repo in a.repo:
        tag = os.path.basename(os.path.normpath(repo))
        en_d = os.path.join(repo, 'compendium', 'en')
        cn_d = os.path.join(repo, 'compendium', 'cn')

        # ---- ground truth: how many player-visible attribute values exist in EN
        for f in packs_of(en_d):
            leaves = []
            walk(load_entries(os.path.join(en_d, f)), [], leaves)
            for _, s in leaves:
                for n, v, o in extract(s):
                    if o == 'inline':
                        if n not in MECHANICAL or n == 'activity':
                            resolver[n] += 1
                        continue
                    if translatable(n, o):
                        gt[n] += 1

        # ---- the scan itself, over the Chinese
        for f in packs_of(cn_d):
            leaves = []
            walk(load_entries(os.path.join(cn_d, f)), [], leaves)
            en_leaves = {}
            if os.path.exists(os.path.join(en_d, f)):
                tmp = []
                walk(load_entries(os.path.join(en_d, f)), [], tmp)
                en_leaves = dict(tmp)
            for path, s in leaves:
                for n, v, o in extract(s):
                    seen_attr[n] += 1
                    if not translatable(n, o):
                        continue
                    cn_seen[n] += 1
                    bucket = classify(v, vocab)
                    if bucket is None:
                        continue
                    rec = {'repo': tag, 'pack': f, 'path': path, 'attr': n,
                           'origin': o, 'value': v, 'bucket': bucket}
                    (proper if bucket == 'PROPER_NOUN' else findings).append(rec)

                # enricher 的 {标签}（@UUID 以外）：与属性值同属「方括号语法里的
                # 可见文本」。scan_markup_drift 为了让标签可译，专门把 {…} 剥掉再比对；
                # scan_uuid_swap 只认 @UUID —— 于是 `&Reference[surprise]{Surprised}`
                # 这种没译的标签在全套检查里没有任何判据。实测 10 处。
                for m in ENRICHER_LABEL.finditer(s):
                    head, lab = m.group(1), m.group(3)
                    if head.lower().endswith('uuid'):
                        continue          # @UUID 归 scan_uuid_swap 管
                    if CJK.search(lab) or not WORD.search(lab):
                        continue
                    if classify(lab, vocab) is None:
                        continue
                    labels.append({'repo': tag, 'pack': f, 'path': path,
                                   'enricher': head, 'target': m.group(2),
                                   'label': lab, 'markup': m.group(0)[:120]})

                # attribute NAME translated into non-ASCII -> dead attribute
                for m in TAG.finditer(s):
                    for bm in BAD_NAME.finditer(m.group(1)):
                        broken.append({'repo': tag, 'pack': f, 'path': path,
                                       'attr': bm.group(1), 'value': bm.group(2),
                                       'tag': m.group(0)[:200]})

                # attribute NAME multiset drift vs the English at the same leaf.
                # ⚠ 2026-08-13：原来只比 origin=='html'，于是译文把
                # `@Embed[... readaloud="…"]` 整个 readaloud= 删掉也看不见 ——
                # 而那正是本判据分母最大的两个属性所在的位置。现在三种 origin 都比。
                e = en_leaves.get(path)
                if isinstance(e, str):
                    for orig in ('html', 'embed', 'inline'):
                        en_names = Counter(n for n, _, o in extract(e) if o == orig)
                        cn_names = Counter(n for n, _, o in extract(s) if o == orig)
                        if en_names == cn_names:
                            continue
                        lost = en_names - cn_names
                        gained = cn_names - en_names
                        if not a.strict_id and gained.get('id'):
                            id_added[f'{tag}/{f}'] += gained['id']
                            del gained['id']
                        if not lost and not gained:
                            continue
                        drift.append({'repo': tag, 'pack': f, 'path': path,
                                      'origin': orig,
                                      'lost': dict(lost), 'gained': dict(gained)})

    tot_gt = sum(gt.values())
    tot_cn = sum(cn_seen.values())
    print(f'英文基准里「面向玩家可见」的属性值共 {tot_gt} 处（这是覆盖率的分母）:')
    for n, c in gt.most_common():
        miss = gt[n] - cn_seen[n]
        flag = f'   ⚠ 中文侧只有 {cn_seen[n]}，少 {miss}' if miss else ''
        print(f'   {c:>5}  {n}{flag}')
    print(f'   中文侧同类属性实测 {tot_cn} 处'
          f'{"（与英文一致）" if tot_cn == tot_gt else f"（与英文差 {tot_cn - tot_gt}，见属性名漂移）"}')
    print()
    # 覆盖率的分子按**中文侧实测**算，不是拿分母减缺漏数倒推 ——
    # 倒推的话，译文把 readaloud= 整个删掉仍然会显示 100%。
    ok = tot_cn - len(findings) - len(proper)
    print(f'中文侧仍是纯英文的:  确认缺漏 {len(findings)} 处 / 专名待判 {len(proper)} 处'
          f'  ->  已译 {ok}/{tot_gt}'
          f' = {100.0 * ok / max(tot_gt, 1):.1f}%')
    print()
    by = Counter((r['attr'], r['bucket']) for r in findings)
    for (n, b), c in by.most_common():
        print(f'   {c:>5}  {n:20} {b}')
    print()
    print('--- 确认缺漏（无中文、含英文散文） ---')
    for r in findings[:a.show]:
        print(f'   [{r["repo"]}/{r["pack"]}] {r["attr"]}={r["value"]!r}')
        print(f'        {r["path"][:120]}')
    print()
    print(f'--- 专名单列（{len(proper)} 处，本项目专名双语并列，可能有意保留，需人判） ---')
    for r in proper[:a.show]:
        print(f'   [{r["repo"]}/{r["pack"]}] {r["attr"]}={r["value"]!r}')
        print(f'        {r["path"][:120]}')
    print()
    print('--- 内联命令体内的属性（照抄，不算缺陷，仅计数） ---')
    for n, c in resolver.most_common():
        print(f'   {c:>5}  {n}')
    print()
    print(f'--- ⚠ 属性名本身被译成中文（属性直接失效） {len(broken)} 处 ---')
    for r in broken[:a.show]:
        print(f'   [{r["repo"]}/{r["pack"]}] {r["attr"]}="{r["value"]}"')
        print(f'        {r["path"][:120]}')
        print(f'        {r["tag"]}')
    print()
    print(f'--- 属性名多重集与英文不一致 {len(drift)} 处 ---')
    for r in drift[:a.show]:
        print(f'   [{r["repo"]}/{r["pack"]}] ({r["origin"]}) 丢失={r["lost"]} 多出={r["gained"]}')
        print(f'        {r["path"][:120]}')
    if id_added:
        print(f'   （白名单放行：中文侧**净增加**的 id= 共 {sum(id_added.values())} 个 '
              f'/ {len(id_added)} 个包 —— 第八轮为修 `#锚点` 断链有意补的，见 PROJECT.md §8 2026-08-13f。')
        print('     净**减少**的 id 仍然会报：丢了 id 就是丢了锚点。要连净增加也看，加 --strict-id）')
        for k, v in id_added.most_common():
            print(f'     {v:>5}  {k}')
    print()
    print(f'--- ⚠ enricher 的 {{标签}} 仍是英文（@UUID 除外）{len(labels)} 处 ---')
    print('    （scan_markup_drift 为了让标签可译特地把 {…} 剥掉，scan_uuid_swap 只认 @UUID，')
    print('      所以这一档在全套检查里此前没有任何判据）')
    for r in labels[:a.show]:
        print(f'   [{r["repo"]}/{r["pack"]}] {r["markup"]}')
        print(f'        {r["path"][:120]}')

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({
                'ground_truth_en': dict(gt),
                'ground_truth_total': tot_gt,
                'visible_attrs_seen_cn': dict(cn_seen),
                'visible_attrs_total_cn': tot_cn,
                'translated_ok': ok,
                'attr_names_seen_cn': dict(seen_attr),
                'resolver_verbatim': dict(resolver),
                'findings': findings,
                'proper_noun_watchlist': proper,
                'broken_attr_names': broken,
                'attr_name_drift': drift,
                'enricher_label_english': labels,
            }, fh, ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')

    return 1 if (findings or broken or drift or labels) else 0


if __name__ == '__main__':
    sys.exit(main())
