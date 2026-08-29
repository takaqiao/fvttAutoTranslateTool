#!/usr/bin/env python3
"""把某次提交里修好的译文，传播到**英文逐字相同**的同源副本上。

  python propagate_fix.py --repo <repo> --english <英文包目录> --since <commit> [--write]

为什么需要它
------------
Crucible 把同一条能力抄进了好几个包：一个天赋在 `crucible.talent.json` 里有定义，
在 `crucible.playtest.json` / `crucible.pregens.json` 的预生角色身上又各有一份**逐字相同**
的副本。上游改规则时往往只改其中一份的英文措辞，另外几份的英文一个字没动。

于是「英文变了→中文没跟上」那套三方 diff 只会挑出被改动的那一份。修完之后，
**同一条规则的其它副本仍带着旧错**，而且再也没有任何扫描能发现它们
—— 它们的英文和中文都「自洽」，只是自洽在错误的旧规则上。

做法：拿 `--since` 之后改过的 (英文, 新中文) 当权威样本，去全库找英文逐字相同、
中文却不同的条目，把新中文推过去。英文相同是硬条件，所以不存在语境误伤。
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_translations import split_path, markup_signature, get_at as _get_at  # noqa: E402


def walk(obj, prefix=''):
    """产出 (点号路径, 字符串值)。路径与 apply_translations 的 split_path 互逆。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{prefix}.{k}' if prefix else k)
    elif isinstance(obj, str) and obj:
        yield prefix, obj


def role_of(path):
    """路径的**字段角色** = 最后一段非数字的键名。

    `name` / `adjective` / `text` / `public` 各有各的格式约定，跨角色传播必错 ——
    见 index 处的注释。数组下标要跳过（`effects.0.name` 的角色是 `name`）。
    """
    for seg in reversed(path.split('.')):
        if not seg.isdigit():
            return seg
    return path


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def get_at(root, path):
    """点号路径取值。用闸门的 split_path —— 键里带点的页名（`Patch 0.5.1`）必须靠它。"""
    v = _get_at(root, split_path(root, path))
    return v if isinstance(v, str) else None


def set_at(root, path, val):
    segs = split_path(root, path)
    node = _get_at(root, segs[:-1])
    node[segs[-1]] = val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--english', required=True, help='英文包目录（compendium/en 或基线目录）')
    ap.add_argument('--since', required=True, help='从这个 commit 之后改过的条目当权威样本')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    packs = sorted(f for f in os.listdir(cn_dir) if f.endswith('.json'))

    cn_data = {p: load(os.path.join(cn_dir, p)) for p in packs}
    en_data = {}
    for p in packs:
        ep = os.path.join(a.english, p)
        if os.path.exists(ep):
            en_data[p] = load(ep)

    # 全库索引：(英文原文, 字段角色) -> [(包, 路径, 中文)]
    #
    # **角色必须进键**。只按英文原文配对会跨字段角色传播，而不同角色的**格式约定不同**：
    #   `name`      是双语并列（「辉耀 Luminary」，见第 3.4 节）
    #   `adjective` 是拼装用的裸中文（`{prefixes}{name}` → 「辉耀长剑」，见第 8 节）
    # 两者英文都是 `Luminary`，不加这道闸就会把 name 推进 adjective，
    # 运行时拼出「辉耀 Luminary长剑」；反方向则会把 name 的英文尾巴剥掉。
    # 2026-08-13 实测：crucible 侧 10 条候选里有 7 条正是这种跨角色配对。
    index = {}
    for p, en in en_data.items():
        for path, en_text in walk(en.get('entries', en)):
            cn_text = get_at(cn_data[p].get('entries', cn_data[p]), path)
            if cn_text:
                index.setdefault((en_text, role_of(path)), []).append((p, path, cn_text))

    # 权威样本：--since 之后中文发生变化的条目
    changed = subprocess.run(
        ['git', '-C', a.repo, 'diff', '--name-only', a.since, 'HEAD', '--', 'compendium/cn'],
        capture_output=True, text=True).stdout.split()
    authority = {}
    for rel in changed:
        pack = os.path.basename(rel)
        if pack not in en_data:
            continue
        old = subprocess.run(['git', '-C', a.repo, 'show', f'{a.since}:{rel}'],
                             capture_output=True, text=True, encoding='utf-8').stdout
        if not old:
            continue
        old_json = json.loads(old)
        for path, en_text in walk(en_data[pack].get('entries', en_data[pack])):
            new_cn = get_at(cn_data[pack].get('entries', cn_data[pack]), path)
            old_cn = get_at(old_json.get('entries', old_json), path)
            if new_cn and old_cn and new_cn != old_cn:
                authority[(en_text, role_of(path))] = (new_cn, f'{pack}::{path}')

    hits = []
    for (en_text, role), (good_cn, src) in authority.items():
        for p, path, cn_text in index.get((en_text, role), []):
            if cn_text == good_cn or f'{p}::{path}' == src:
                continue
            # 闸门同款校验：标记签名必须一致，否则不推
            if markup_signature(good_cn) != markup_signature(en_text):
                continue
            hits.append((p, path, cn_text, good_cn, src))

    by_pack = {}
    for p, path, _old, new, _src in hits:
        by_pack.setdefault(p, {})[path] = new

    print(f'权威样本 {len(authority)} 条（{a.since}..HEAD 改过的译文）')
    print(f'英文逐字相同、中文还是旧的：{len(hits)} 条，分布 {len(by_pack)} 个包')
    for p, d in sorted(by_pack.items()):
        print(f'  {p:<42} {len(d)}')
    for p, path, old, new, src in hits[:6]:
        print(f'\n  {p}::{path}\n    源: {src}\n    旧: {old[:100]}\n    新: {new[:100]}')

    if a.write:
        for p, d in by_pack.items():
            for path, new in d.items():
                set_at(cn_data[p].get('entries', cn_data[p]), path, new)
            with open(os.path.join(cn_dir, p), 'w', encoding='utf-8') as f:
                json.dump(cn_data[p], f, ensure_ascii=False, indent=2)
        print(f'\n已写回 {len(hits)} 条')
    else:
        print('\n(未加 --write，未改动)')


if __name__ == '__main__':
    main()
