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

译文记忆**只**按 `(结构路径, 英文)` 建，且结构路径必定含最后一段（字段角色）：
`.name` 按所处位置分三种写法（物品名 / 动作名 / 效果名），只按最后一段建键会把三
类混成一堆，取出错误的多数派 —— 而**只按英文**建键（连角色都没有）更糟，它会把
`name` 的双语并列「护盾术 Shield」灌进 `tokenName` / `adjective` 的裸中文槽。
2026-08-14 起取消「退化到只按英文」这一分支：查不到就交给人翻，并把无角色键
本会给出的建议记进报告的 `roleblind_suggestions`，供人过目而不落库。
同一个键下若有互相竞争的中文，取多数并把冲突报出来，不静默裁决。

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

# 结构段：路径里保留这些，其余（实体名/页名）一律丢弃。
# ⚠ 这张白名单只是「中间层容器」的加分项，**不是**角色维度的保障 —— 角色维度由
# shape_of 的「最后一段永远保留」提供。2026-08-14 普查（两仓 en 全量）发现原表漏掉
# 的容器/角色段有：scenes 2806 / regions 1508 / tables 1302 / outcomes 1096 /
# summary 1042 / overview 818 / behaviors 796 / exposition 778 /
# contentOverview 664 / notes 574 / categories 552 / levels 517 /
# pronunciation 210 / adjective 172 / subtitle 72 —— 其中 regions.name 是裸中文
# 约定、顶层 name 是双语并列约定，漏掉 regions 就把两个互斥约定装进同一个桶。
STRUCT = {'entries', 'actors', 'items', 'actions', 'effects', 'pages', 'journals',
          'results', 'folders', 'biography', 'description', 'name', 'text',
          'label', 'public', 'private', 'appearance', 'condition', 'tokenName',
          'scenes', 'regions', 'tables', 'behaviors', 'tokens', 'outcomes',
          'levels', 'notes', 'categories', 'adjective', 'caption', 'subtitle',
          'pronunciation', 'overview', 'summary', 'exposition', 'navName',
          'contentOverview', 'contentGamemaster'}


def sig(s: str):
    s = QUOTED_PARAM.sub('="~"', s)
    return (Counter(MARKUP.findall(s)) + Counter(INLINE_CMD.findall(s))
            + Counter(f'<{sl}{n.lower()}' for sl, n in TAGNAME.findall(s)))


def shape_of(path):
    """路径的角色骨架。**最后一段永远保留** —— 它就是字段角色（name / tokenName /
    adjective / levels / …），而白名单永远追不上上游新增的角色名：白名单漏掉谁，
    谁的角色维度就整个消失，于是两套互斥的书写约定（`name` 双语并列「护盾术 Shield」
    与 `tokenName`/`adjective` 裸中文「护盾术」）被装进同一个桶，取出错误的多数派。
    保留最后一段最坏只是让键过窄（少命中、落到人翻），绝不会张冠李戴。
    """
    parts = path.split('.')
    return '.'.join(p for i, p in enumerate(parts)
                    if p in STRUCT or p.isdigit() or i == len(parts) - 1)


def walk(o, p=''):
    """遍历到叶子。**必须下钻 list** —— 索引作为一段路径，与 `apply_translations.py`
    的 `split_path` / `get_at` 同一套语义（它们对 list 就是 `node[int(p)]`）。

    2026-08-12 之前这里只认 dict 与 str，数组内的叶子**整体不在定义域**：
    `outcomes[0].label`、`effects[1].name`、`changes[…]` 这些一条都看不见。
    后果是本脚本对着有缺口的库报「需人翻 0」，而 ember 侧实测有 22 条数组内叶子
    中文整条不存在（审计 2026-08-12 第 2.1 条）。本脚本的存在意义就是覆盖
    「所有其它检查都覆盖不到的那个方向」，自己再留一个方向就没意义了。
    """
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{p}.{k}' if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f'{p}.{i}' if p else str(i))
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
    report = {'filled': {}, 'unresolved': {}, 'conflicts': [],
              'roleblind_suggestions': []}
    for (repo, pack), items in sorted(missing.items()):
        batch = {}
        rest = {}
        for path, src in items.items():
            cands = tm_shape.get((shape_of(path), src))
            if not cands:
                # 这里曾经退化成 `tm_plain.get(src)`：纯英文键、零角色的全库多数派。
                # 留一法实测：3437 条会走这条退化路径，其中 232 条写出的中文与人写的
                # 不同，且 232/232 的多数派来自**别的角色**（218 条是「双语并列 ↔
                # 裸中文」互换，如 `Ifton Shepp.tokenName` 会被写成「伊夫顿·谢普
                # Ifton Shepp」、`Bewilderment.adjective` 会被写成「迷惘 Bewilderment」
                # 而后者会被 item-physical.mjs:324 拼进物品名）。裸名字两侧的 markup
                # 签名都是空集，:143 的 sig 闸与 apply_translations.py 的三道闸
                # 一条都拦不住。现在不再写值，只把建议记进报告供人过目。
                blind = tm_plain.get(src)
                if blind:
                    report['roleblind_suggestions'].append(
                        {'pack': pack, 'path': path, 'en': src[:120],
                         'no_role_majority': blind.most_common(1)[0][0][:120]})
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
            # ⚠ 待译清单的 key **必须**和 tm.* 一样走 to_batch_path。
            # 原先这里直接写文档根路径（带 `entries.` 前缀），而 tm.* 那边转过 ——
            # 于是「机器能填的那一半」键是对的、「留给人翻的那一半」键是错的。
            # 后果：照 PROJECT.md「批次 key ＝ 待译清单里的 path」把 todo 填完直接喂
            # apply_translations，会整批报 `REJECTED no-EN`，而那个理由看着像
            # 「英文里没有这个键」，极易被误读成源数据的问题。
            # 实测：第十四轮 6 路翻译 agent **每一路都先踩了一次**才发现要剥前缀。
            todo = {}
            for path, src in rest.items():
                bp = to_batch_path(path)
                if bp is None:
                    continue        # label 之类的顶层标量本来就不走批次
                todo[bp] = src
            out = os.path.join(a.out_dir, f'todo.{pack}')
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(todo, f, ensure_ascii=False, indent=1)
            unresolved += len(todo)
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
