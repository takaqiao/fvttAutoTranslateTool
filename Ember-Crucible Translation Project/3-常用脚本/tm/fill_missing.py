#!/usr/bin/env python3
"""用全库译文记忆补上**中文侧根本不存在的键**。

  python fill_missing.py --repo <repo> [--repo <repo2>] --out-dir <批次目录> [--report <json>]

与 fill_twin.py 的分工
----------------------
`fill_twin.py` 只做 crucible-adventure -> ember.adventure 这一个方向（同一场
战役的两套规则版本）。本脚本处理的是另一类缺口：**某个键在英文包里有，中文包里
整条不存在**。

这类缺口任何既有扫描都发现不了 —— 覆盖率、残留、标记签名、drift 全都是拿
「中文里的某条」去比对，中文里压根没有的条目不在它们的定义域内。所以库里一直报
「覆盖率 99%」，而 crucible 的两个预生角色（Fizzit / Zarajah）几乎整体没译。
实测两个仓库合计 572 条、约 8.6 万英文字符。

译文记忆按 `(结构路径, 英文)` 建，退化到只按英文 —— 与 fill_twin 同一套理由：
`.name` 按所处位置分三种写法（物品名 / 动作名 / 效果名），只按最后一段建键会把三
类混成一堆，取出错误的多数派。同一个键下若有互相竞争的中文，取多数并把冲突报出来，
不静默裁决。

本脚本**不直接写** compendium/cn，只出批次；由 apply_translations.py 落库，
走与人工翻译完全相同的三道闸。
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter, defaultdict

MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
INLINE_CMD = re.compile(r'\[\[[^\]]*\]\]')
TAGNAME = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)')
QUOTED_PARAM = re.compile(r'=\s*"[^"]*"')
CJK = re.compile(r'[一-鿿]')
WORD = re.compile(r'[A-Za-z]{3,}')

# 结构段：路径里保留这些，其余（实体名/页名）一律丢弃
STRUCT = {'entries', 'actors', 'items', 'actions', 'effects', 'pages', 'journals',
          'results', 'folders', 'biography', 'description', 'name', 'text',
          'label', 'public', 'private', 'appearance', 'condition', 'tokenName'}


def sig(s: str):
    s = QUOTED_PARAM.sub('="~"', s)
    return (Counter(MARKUP.findall(s)) + Counter(INLINE_CMD.findall(s))
            + Counter(f'<{sl}{n.lower()}' for sl, n in TAGNAME.findall(s)))


def shape_of(path):
    return '.'.join(p for p in path.split('.') if p in STRUCT or p.isdigit())


def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{p}.{k}' if p else k)
    elif isinstance(o, str) and o.strip():
        yield p, o


def to_batch_path(path):
    """文档根路径 -> apply_translations 的批次路径。

    闸门是以 `entries` 为根解析批次 key 的（`folders` 走 `(folders)` 前缀），
    不是文档根。少了这一步，整批 500 条会被判成 `REJECTED no-EN` —— 拒绝理由
    看着像「英文里没有这个键」，其实是路径根对不上，很容易被误读成源数据的问题。
    """
    if path.startswith('entries.'):
        return path[len('entries.'):]
    if path.startswith('folders.'):
        return '(folders).' + path[len('folders.'):]
    return None   # label 之类的顶层标量，不走批次


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--report')
    a = ap.parse_args()

    tm_shape = defaultdict(Counter)   # (shape, en) -> Counter(cn)
    tm_plain = defaultdict(Counter)   # en -> Counter(cn)
    missing = defaultdict(dict)       # (repo, pack) -> {path: en}

    for repo in a.repo:
        en_dir = os.path.join(repo, 'compendium', 'en')
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json'):
                continue
            cnp = os.path.join(cn_dir, fn)
            if not os.path.exists(cnp):
                continue
            en = dict(walk(load(os.path.join(en_dir, fn))))
            cn = dict(walk(load(cnp)))
            for path, src in en.items():
                tgt = cn.get(path)
                if tgt and CJK.search(tgt):
                    tm_shape[(shape_of(path), src)][tgt] += 1
                    tm_plain[src][tgt] += 1
                elif tgt is None:
                    bare = re.sub(r'<[^>]+>', '', MARKUP.sub('', INLINE_CMD.sub('', src)))
                    if WORD.search(bare):
                        missing[(repo, fn)][path] = src

    os.makedirs(a.out_dir, exist_ok=True)
    filled = unresolved = conflicts = 0
    report = {'filled': {}, 'unresolved': {}, 'conflicts': []}
    for (repo, pack), items in sorted(missing.items()):
        batch = {}
        rest = {}
        for path, src in items.items():
            cands = tm_shape.get((shape_of(path), src)) or tm_plain.get(src)
            if not cands:
                rest[path] = src
                continue
            if len(cands) > 1:
                conflicts += 1
                report['conflicts'].append({'pack': pack, 'path': path, 'en': src[:120],
                                            'candidates': dict(cands)})
            best = cands.most_common(1)[0][0]
            # 标记必须与本条英文一致，否则宁可留给人翻
            if sig(best) != sig(src):
                rest[path] = src
                continue
            bp = to_batch_path(path)
            if bp is None:
                continue
            batch[bp] = best
        if batch:
            out = os.path.join(a.out_dir, f'tm.{pack}')
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False, indent=1)
            filled += len(batch)
        if rest:
            out = os.path.join(a.out_dir, f'todo.{pack}')
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(rest, f, ensure_ascii=False, indent=1)
            unresolved += len(rest)
        report['filled'][f'{repo}/{pack}'] = len(batch)
        report['unresolved'][f'{repo}/{pack}'] = len(rest)
        chars = sum(len(v) for v in rest.values())
        print(f'  {pack:<38} TM 命中 {len(batch):>4} / 仍需人翻 {len(rest):>4}（{chars} 字符）')

    print(f'\n合计：TM 补上 {filled} 条 / 需人翻 {unresolved} 条 / 同键竞争 {conflicts} 处')
    if a.report:
        with open(a.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'  → {a.report}')


if __name__ == '__main__':
    main()
