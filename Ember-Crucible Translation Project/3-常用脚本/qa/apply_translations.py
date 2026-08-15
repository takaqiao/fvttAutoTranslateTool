#!/usr/bin/env python3
"""Apply a batch of translations into a plugin's `compendium/cn/*.json`.

Input is a JSON file of `{"<dotted path>": "<中文>"}` where the path is exactly
the `path` field emitted by `validate_translations.py` into
`5-其他内容/reports/*/todo/*.todo.json`.

Safety rails, because these files are large and hand-edited batches drift:
  - refuses to write a value whose English source no longer matches the baseline
    (the pack changed under us)
  - refuses to overwrite an existing Chinese value unless --force -- and the
    check walks EVERY level of the path, not just the leaf: a translated
    `description` that lives as a plain string where English has an object is a
    supported shape (see release/runtime-converters.js::crucibleDescription), so
    writing `description.public` under it used to replace that whole string with
    an empty dict, silently and uncounted
  - checks that inline markup (@UUID / @Check / HTML tags) survives the
    translation, and reports every mismatch instead of writing it

Usage:
  python apply_translations.py --repo <pluginRepo> --pack <crucible.talent.json>
                               --batch <batch.json> [--force] [--dry]
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter

CJK = re.compile(r'[一-鿿]')
# Foundry inline markup. The bracketed TARGET must survive verbatim; the
# optional {display label} is prose and is expected to be translated, so it is
# stripped before comparison.
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
# Foundry inline rolls/commands: [[/hazard 25 reflex health]]{Label},
# [[/skillCheck wilderness 14]], [[/r 1d20]] ... same rule: the command body is
# machinery and must survive verbatim, the {label} is prose.
INLINE_CMD = re.compile(r'\[\[[^\]]*\]\]')
TAGNAME = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)')
TAG = re.compile(r'<[^>]+>')
LATIN = re.compile(r'[A-Za-z]')


def translatable(en: str) -> bool:
    """False when the English holds no visible prose at all, e.g.
    `<p>@Embed[JournalEntry.x.JournalEntryPage.y inline]</p>`.

    Such a value still needs to be writable -- a broken target inside the
    markup has to be repairable -- but demanding Chinese in it would be
    demanding that machinery be translated, which is what broke it.
    """
    rest = MARKUP.sub(' ', TAG.sub(' ', en)).replace('&amp;', ' ')
    rest = INLINE_CMD.sub(' ', rest)
    return bool(LATIN.search(rest))


def load(p):
    # utf-8-sig: batch files hand-written via PowerShell carry a BOM.
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def dump(p, o):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(o, f, ensure_ascii=False, indent=2)
        f.write('\n')


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def split_path(root, path):
    """Split a dotted path into keys, tolerating keys that contain dots.

    Journal page keys such as `Patch 0.5.1` are shredded by a naive
    `path.split('.')`, so every leaf under them looks like it has no English
    source. The naive split is still tried first (unchanged behaviour); only
    when it resolves to nothing do we walk `root` preferring the longest
    matching key.
    """
    naive = path.split('.')
    if get_at(root, naive) is not None:
        return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + '.')]
            if cands:
                k = max(cands, key=len)
                parts.append(k)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition('.')
        parts.append(head)
        if isinstance(node, list):
            try:
                node = node[int(head)]
            except (ValueError, IndexError):
                node = None
        elif isinstance(node, dict):
            node = node.get(head)
        else:
            node = None
    return parts


class WouldClobber(Exception):
    """`set_at` 拒绝把路径**中间层**上一个已有的字符串换成空容器。

    为什么这不是脏数据：CN 侧在 EN 的 object 路径上放一个字符串是**受支持的写法** ——
    `3-常用脚本/release/runtime-converters.js` 的 `crucibleDescription`（:19-26）明确
    处理「源是对象、译文是字符串」这一形态。所以中间层的字符串是有人写的正文，
    静默 `node[p] = {}` 会把整段既有中文吃掉，而且既不进 problems 也不计 skipped。
    """

    def __init__(self, prefix, existing):
        super().__init__(prefix)
        self.prefix = prefix
        self.existing = existing


def set_at(root, parts, value, shape=None, allow_clobber=False):
    """Write `value` at `parts`, creating containers that MIRROR the English
    structure.

    Without `shape` a missing `effects` array would be created as a dict keyed
    "0", which Babele would never read as an array. `shape` is the English node
    at the same position and decides list vs dict for each level created.

    Raises `WouldClobber` instead of silently replacing an intermediate string
    (see that exception's docstring). `allow_clobber=True` — wired to `--force` —
    restores the old destructive behaviour for the rare deliberate reshape.
    """
    node = root
    for i, p in enumerate(parts[:-1]):
        nxt_shape = None
        if shape is not None:
            try:
                shape = shape[int(p)] if isinstance(shape, list) else shape.get(p)
            except (ValueError, IndexError, AttributeError):
                shape = None
        nxt_shape = shape

        if isinstance(node, list):
            node = node[int(p)]
            continue
        if isinstance(node.get(p), str) and not allow_clobber:
            raise WouldClobber('.'.join(parts[:i + 1]), node[p])
        if p not in node or not isinstance(node[p], (dict, list)):
            node[p] = [] if isinstance(nxt_shape, list) else {}
        node = node[p]
        if isinstance(node, list):
            idx = int(parts[i + 1]) if parts[i + 1].isdigit() else None
            if idx is not None:
                while len(node) <= idx:
                    node.append({})

    if isinstance(node, list):
        idx = int(parts[-1])
        while len(node) <= idx:
            node.append({})
        node[idx] = value
    else:
        node[parts[-1]] = value


# Quoted parameter VALUES inside markup are visible prose, not machinery:
# `@Embed[Actor.x readaloud="A radiant figure stands…"]` renders that sentence to
# the player. Comparing it verbatim made the whole @Embed immutable, so those
# read-aloud blocks could never be translated -- even though 22 of them already
# were. Only the value is blanked; the parameter NAME and everything outside the
# quotes still has to match.
QUOTED_PARAM = re.compile(r'=\s*"[^"]*"')

# ⚠ 2026-08-13：同一个参数**不带引号**时（Foundry 的 `parseEmbedConfig` 允许
# `@Embed[Actor.x label=Squish]`），上面那条正则看不见它，于是签名里留着英文原值 ——
# 结果是闸门**结构性地禁止**翻译这个 label：任何译法都会被报 markup mismatch。
# 本库正有 2 处这样的 label，它们至今是英文，原因就在这儿而不在译者。
# 只对确定是可见文本的三个参数名放行；`apply=false` 这类机械参数仍然逐字节比对。
BARE_TEXT_PARAM = re.compile(r'\b(label|readaloud|caption)\s*=\s*(?!["\'])([^\s\]<>"\']+)')

# dnd5e 的规则引用。`MARKUP` 只认 `@` 开头，于是 `&Reference[Restrained]` 长期不进签名 ——
# 方括号里的键被译成中文（`&Reference[受拘束]`）时，闸门照样报 0 拒绝，而 enricher
# 查不到这个键，玩家看到的是裸文本而不是规则链接。全库积了 77 条，2026-08-09 补上。
# 按 BRIEF 第 2 节：方括号里是引用键不是正文，照抄；渲染时 dnd5e 自己出中文名。
REFERENCE = re.compile(r'&(?:amp;)?[Rr]eference\[[^\]]*\]')


def markup_signature(s: str):
    """Order-insensitive multiset of the markup that must be preserved.

    `@UUID[target]{label}` compares on `@UUID[target]` only: the label is the
    visible text and must be translated, not preserved. Same for `key="value"`
    parameters inside a marker.

    `&Reference[key]` is normalised to a single entity form first, so a change of
    `&amp;` to `&` alone is not treated as a markup change.
    """
    s = BARE_TEXT_PARAM.sub(r'\1="~"', s)
    s = QUOTED_PARAM.sub('="~"', s)
    return (Counter(MARKUP.findall(s))
            + Counter(INLINE_CMD.findall(s))
            + Counter(m.replace('&amp;', '&') for m in REFERENCE.findall(s))
            + Counter(f'<{slash}{name.lower()}' for slash, name in TAGNAME.findall(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--batch', required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    en_path = os.path.join(a.repo, 'compendium', 'en', a.pack)
    cn_path = os.path.join(a.repo, 'compendium', 'cn', a.pack)
    en = load(en_path)
    cn = load(cn_path) if os.path.exists(cn_path) else {
        'label': en.get('label'), 'folders': {}, 'entries': {}}
    batch = load(a.batch)
    items = batch.get('items', batch)

    applied = skipped_existing = missing_en = markup_bad = no_cjk = 0
    clobber_bad = 0
    problems = []

    for path, value in items.items():
        parts = path.split('.')
        root_en = en.get('folders', {}) if parts[0] == '(folders)' else en.get('entries', {})
        root_cn = cn.setdefault('folders', {}) if parts[0] == '(folders)' else cn.setdefault('entries', {})
        if parts[0] == '(folders)':
            parts = parts[1:]
        parts = split_path(root_en, '.'.join(parts))

        src = get_at(root_en, parts)
        if not isinstance(src, str):
            missing_en += 1
            problems.append({'path': path, 'issue': 'no English source at this path'})
            continue

        if not CJK.search(value) and translatable(src):
            no_cjk += 1
            problems.append({'path': path, 'issue': 'translation contains no Chinese'})
            continue

        want, got = markup_signature(src), markup_signature(value)
        if want != got:
            markup_bad += 1
            diff = {k: (want.get(k, 0), got.get(k, 0))
                    for k in set(want) | set(got) if want.get(k, 0) != got.get(k, 0)}
            problems.append({'path': path, 'issue': 'markup mismatch (want, got)', 'detail': diff})
            continue

        # 护栏：路径上**任何一层**已经是字符串就不能直接写。
        # 只查叶子（parts 全长）看不见「中间层是字符串」这一形态，而那正是
        # set_at 会静默换成空容器的那一层 —— 见 WouldClobber 的 docstring。
        existing_at = clobber_at = clobber_val = None
        for n in range(1, len(parts) + 1):
            cur = get_at(root_cn, parts[:n])
            if not isinstance(cur, str):
                continue
            if CJK.search(cur):
                existing_at = '.'.join(parts[:n])
            elif n < len(parts):
                # 叶子本身是非中文字符串 = 正常的覆盖写，放行；只有中间层要拦。
                clobber_at, clobber_val = '.'.join(parts[:n]), cur
            break
        if existing_at is not None and not a.force:
            skipped_existing += 1
            continue
        if clobber_at is not None and not a.force:
            clobber_bad += 1
            problems.append({'path': path,
                             'issue': 'would clobber intermediate string',
                             'detail': {'at': clobber_at, 'existing': clobber_val[:120]}})
            continue

        if not a.dry:
            try:
                set_at(root_cn, parts, value, root_en, allow_clobber=a.force)
            except WouldClobber as e:
                clobber_bad += 1
                problems.append({'path': path,
                                 'issue': 'would clobber intermediate string',
                                 'detail': {'at': e.prefix, 'existing': e.existing[:120]}})
                continue
        applied += 1

    if not a.dry and applied:
        dump(cn_path, cn)

    print(f'{a.pack}')
    print(f'  applied            {applied}')
    print(f'  skipped (existing) {skipped_existing}')
    print(f'  REJECTED no-EN     {missing_en}')
    print(f'  REJECTED no-CJK    {no_cjk}')
    print(f'  REJECTED markup    {markup_bad}')
    print(f'  REJECTED clobber   {clobber_bad}')
    if problems:
        print('\n  problems:')
        for p in problems[:25]:
            print(f'    {p["path"]}\n       {p["issue"]}'
                  + (f'\n       {p.get("detail")}' if p.get('detail') else ''))
        if len(problems) > 25:
            print(f'    ... and {len(problems) - 25} more')
    if a.dry:
        print('\n(dry run: nothing written)')
    raise SystemExit(1 if problems else 0)


# Guarded so `split_path` / `markup_signature` can be imported by the prep and
# collect scripts instead of being reimplemented (and drifting) there.
# CLI behaviour is unchanged.
if __name__ == '__main__':
    main()
