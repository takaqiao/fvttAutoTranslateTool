#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""落盘前的恒等核验探针（落盘人自己写、落盘、再跑 —— 不用 heredoc）。

对每个 noop 批次的每一条：
  1. 报出「我这次扫了多少条」（反空转：条数必须 = 批次条数）
  2. 逐条把批次里的新值与库里现值做**逐字节**比较
  3. 顺带把「同叶别的术语有没有被顺手改掉」这件事量化：
     不是靠肉眼，而是靠 IDENTICAL 这个更强的结论 —— 全等则同叶一切都没动。
  4. 对非全等的条目打印 unified diff，供人判。
反向自检（证明探针真的在验）：把每条值人为改一个字符，必须全部报 DIFF。
"""
import json, os, sys, difflib

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(P, '3-常用脚本', 'qa'))
from apply_translations import split_path, get_at, load  # noqa: E402

BATCH_DIR = os.path.join(P, '4-临时脚本', '2026-08-16-round22', 'batches')
JOBS = [
    ('2-Crucible汉化插件', 'crucible.rules.json', 'noop_verify_crucible.rules.json'),
    ('2-Crucible汉化插件', 'crucible.equipment.json', 'noop_verify_crucible.equipment.json'),
]


def check(mutate=False):
    total = ident = diff = missing = 0
    details = []
    for repo, pack, bf in JOBS:
        en = load(os.path.join(P, repo, 'compendium', 'en', pack))
        cn = load(os.path.join(P, repo, 'compendium', 'cn', pack))
        items = load(os.path.join(BATCH_DIR, bf))
        items = items.get('items', items)
        print(f'--- {bf} -> {repo}/compendium/cn/{pack} : 批次 {len(items)} 条')
        for path, val in items.items():
            total += 1
            parts = split_path(en.get('entries', {}), path)
            cur = get_at(cn.get('entries', {}), parts)
            if mutate:
                val = val + 'X'
            if cur is None:
                missing += 1
                details.append((pack, path, 'NO-CN-LEAF', None, None))
                continue
            if cur == val:
                ident += 1
            else:
                diff += 1
                details.append((pack, path, 'DIFF', cur, val))
    print(f'\n扫过 {total} 条 · 逐字节全等 {ident} · 不等 {diff} · 库里无此叶 {missing}')
    for pack, path, kind, cur, val in details:
        print(f'\n### {kind}  {pack} :: {path}')
        if kind == 'DIFF':
            for line in list(difflib.unified_diff(
                    cur.splitlines(), val.splitlines(),
                    fromfile='库里现值', tofile='批次新值', lineterm=''))[:40]:
                print('   ' + line)
    return total, ident, diff, missing


if __name__ == '__main__':
    print('=== 正向：批次 vs 库 ===')
    t, i, d, m = check(False)
    print('\n=== 反向自检：把每条值改一个字符，必须条条报 DIFF ===')
    t2, i2, d2, m2 = check(True)
    ok = (d == 0 and m == 0 and t > 0 and d2 == t2 and i2 == 0)
    print(f'\n探针自证：正向 {i}/{t} 全等 · 反向 {d2}/{t2} 报不等 · 结论 {"OK" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)
