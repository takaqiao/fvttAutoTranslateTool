#!/usr/bin/env python3
"""`tokenName` 与 `name` 的中文不一致 —— 玩家在地图上看到的名字和角色卡对不上。

判据（严格版）：**英文侧 `name` 与 `tokenName` 逐字节相同**时，中文侧也必须一致
（`tokenName` 取 `name` 去掉双语英文尾巴后的中文头，库内约定是裸中文 533/537）。
英文侧本来就不同的（`Kalasak the Cutter` 的 token 叫 `Kalasak`）是作者有意的短称，
**不在判据内**。

为什么单独做一个闸
------------------
`tokenName` 是**玩家在地图上直接看到**的名字，比角色卡上的 name 还显眼，
而它不在任何既有判据的配对范围里：
  * `scan_label_vs_name` 只比 `@UUID{标签}` ↔ 目标文档 `name`；
  * `scan_name_binding` 只比表结果/场景针脚；
  * 覆盖率、标记、数字、外来文字全都只看单个叶子自身。

2026-08-13d 首测：英文侧同名的 481 个 actor 里，**中文侧 32 个不一致**，
其中不乏 `Thayloc Courser` 的 tokenName 是「非玩家角色」（占位符没删）、
`Sporix Host` 的是「受折磨的荆棘幼体」（完全另一种生物）这种。

用法：
  python scan_token_name.py --repo <repo> [--repo <另一个>] [--out <json>] [--fix-batch-dir <目录>]
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 双语并列的英文尾巴：「拉斯特·索恩 Raster Thorn」-> 拉斯特·索恩
BILINGUAL_TAIL = re.compile(r'\s+[^一-鿿　-〿＀-￯]+$')


def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f'{path}.{i}')
    elif isinstance(obj, str) and obj:
        yield path, obj


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def head(s):
    return BILINGUAL_TAIL.sub('', s).strip() or s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out')
    ap.add_argument('--fix-batch-dir', help='顺便产出「tokenName := name 的中文头」批次')
    ap.add_argument('--show', type=int, default=40)
    a = ap.parse_args()

    findings, stats = [], collections.Counter()
    batches = collections.defaultdict(dict)
    for repo in a.repo:
        en_dir = os.path.join(repo, 'compendium', 'en')
        tag = 'ember' if repo.startswith('1') else 'crucible'
        for pack in sorted(os.listdir(en_dir)):
            if not pack.endswith('.json') or pack == '_source.json':
                continue
            cn_p = os.path.join(repo, 'compendium', 'cn', pack)
            if not os.path.exists(cn_p):
                continue
            en = dict(walk(load(os.path.join(en_dir, pack)).get('entries', {})))
            cn = dict(walk(load(cn_p).get('entries', {})))
            for path, v in en.items():
                if not path.endswith('.tokenName'):
                    continue
                npath = path[:-len('.tokenName')] + '.name'
                if en.get(npath) != v:
                    stats['英文侧本来就不同名（by design）'] += 1
                    continue
                stats['英文侧同名'] += 1
                cnt, cnn = cn.get(path), cn.get(npath)
                if not cnt or not cnn:
                    stats['一侧没有中文'] += 1
                    continue
                if head(cnt) == head(cnn):
                    stats['一致'] += 1
                    continue
                stats['**不一致**'] += 1
                findings.append({
                    'repo': repo, 'pack': pack, 'actor': npath, 'batch_path': path,
                    'english': v, 'cn_name': cnn, 'cn_token': cnt, 'should_be': head(cnn),
                })
                batches[(tag, pack)][path] = head(cnn)

    print('统计：')
    for k, v in stats.most_common():
        print(f'  {k:32s} {v}')
    print(f'\n**tokenName 与 name 中文不一致** 共 {len(findings)} 条')
    for f in findings[:a.show]:
        print(f'  {f["pack"][:24]:26s} {f["actor"].split(".actors.")[-1][:32]:34s} '
              f'{f["cn_token"][:20]:22s} -> {f["should_be"]}')

    if a.fix_batch_dir and batches:
        os.makedirs(a.fix_batch_dir, exist_ok=True)
        for (tag, pack), items in batches.items():
            p = os.path.join(a.fix_batch_dir, f'TKN__{tag}__{pack[:-5]}.json')
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=1)
            print(f'-> {os.path.basename(p)}  ({len(items)} 条)')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'stats': dict(stats), 'findings': findings}, f,
                      ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
