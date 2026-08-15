#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5：时间单位 Round(轮) / Turn(回合) 是否在不同页被互换。

定译写死了 `Round`=轮、`Turn`=回合，两者在规则里是**不同长度的时间**
（一轮 = 全场每人各一个回合）。互换会直接改变规则含义，而现有任何判据都不查它：
数字判据只看数值，术语判据 glossary_ec 里 round/turn 是通用词被排除在外。

判据（块级，双向）
------------------
EN 侧只在**明确的时间名词**语境里计数（`for 3 rounds` / `each round` /
`Round 2` / `at the end of its turn` / `on your turn` …），排除
`round table`、`turn the valve`、`in turn` 这些非机制用法。
CN 侧计 `轮` 与 `回合`，并剔除 `轮` 的非机制用法（`一轮月亮` `六轮货车` `轮廓`
`轮流` `车轮` `轮盘` `轮换` `轮番` `轮椅` `转轮`）。

报两类：
 R2T  EN 只说 round，CN 只出现「回合」
 T2R  EN 只说 turn，CN 只出现「轮」

假阳性模式
----------
* 中文把「在其回合开始时」并进上一句、本块只剩另一个词 —— 相邻块会同时出现互补的一条。
* EN 同块里 round 与 turn 都出现时本探针不报（无法判定对应关系），属漏报。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_common import CJK, plain, split_blocks, load_all

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'findings')
os.makedirs(OUT, exist_ok=True)

QUANT = (r'\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
         r'each|every|per|this|that|next|following|another|first|last|same|'
         r'a|an|the|its|your|their|his|her|one|additional|extra|several|many|'
         r'consecutive|full|entire|subsequent|current|previous')
EN_ROUND = re.compile(
    r'\b(?:(?:' + QUANT + r')\s+)?rounds?\b(?!\s*(?:table|of\s+(?:applause|drinks)))'
    r'|\bround\s+\d+\b|\b(?:end|start|beginning|top) of (?:the |each |every |'
    r'its |your |their )?round\b|\bper[- ]round\b', re.I)
EN_TURN = re.compile(
    r'\b(?:(?:' + QUANT + r')\s+)?turns?\b|\b(?:end|start|beginning) of '
    r'(?:the |each |every |its |your |their )?turn\b', re.I)
# turn 的非机制用法：turn the / turn into / turn to / in turn / turn away …
EN_TURN_BAD = re.compile(
    r'\bturns?\s+(?:the|a|an|into|to|toward|towards|away|back|on|off|over|around|'
    r'up|down|in|out|left|right|red|black|pale|it|them|him|her|his|their)\b'
    r'|\bin turn\b|\bturn\s+of\s+(?:the|events|phrase)\b|\btaking turns\b'
    r'|\bby turns\b|\bturns?\s+out\b|\bturned\b|\bturning\b', re.I)
EN_ROUND_BAD = re.compile(r'\bround(?:ed|ish|ly)\b|\bround\s+(?:table|room|shield|'
                          r'chamber|window|stone|tower|building|shape|door)\b'
                          r'|\ba round of\b|\bround\s+the\b|\bgo\s+round\b', re.I)

CN_TURN = re.compile(r'回合')
CN_ROUND_BAD = re.compile(r'轮(?:廓|流|换|番|椅|盘|子|月|明月|货车|辐|回|值|船)|'
                          r'(?:车|齿|转|滚|飞|水|风|年|巨|大|圆)轮|'
                          r'[零〇一二三四五六七八九十百\d]+\s*轮(?:月|明月|货车|马车|大车)')
CN_ROUND = re.compile(r'轮')


def cn_round_count(s: str):
    masked = CN_ROUND_BAD.sub(lambda m: '×' * len(m.group()), s)
    return len(CN_ROUND.findall(masked))


def en_count(s: str, good, bad):
    masked = bad.sub(lambda m: ' ' * len(m.group()), s)
    return len(good.findall(masked))


def page_of(path):
    p = path.split('.')
    return f'{p[1]}/{p[3]}' if len(p) >= 4 and p[2] == 'pages' else '.'.join(p[:2])


def main():
    rows = load_all()
    out = []
    nb = 0
    for repo, pack, path, en, cn in rows:
        if not cn or not CJK.search(cn):
            continue
        eb, cb = split_blocks(en), split_blocks(cn)
        if len(eb) != len(cb):
            continue
        for i, (e, c) in enumerate(zip(eb, cb)):
            pe, pc = plain(e), plain(c)
            if not pe or not CJK.search(pc):
                continue
            er = en_count(pe, EN_ROUND, EN_ROUND_BAD)
            et = en_count(pe, EN_TURN, EN_TURN_BAD)
            if not er and not et:
                continue
            nb += 1
            cr, ct = cn_round_count(pc), len(CN_TURN.findall(pc))
            kind = None
            loose = '--loose' in sys.argv
            if er and not et and ct and not cr:
                kind = 'R2T'
            elif et and not er and cr and not ct:
                kind = 'T2R'
            elif loose and er and not cr:
                kind = 'R-MISSING'
            elif loose and et and not ct:
                kind = 'T-MISSING'
            elif loose and cr and not er:
                kind = 'R-EXTRA'
            elif loose and ct and not et:
                kind = 'T-EXTRA'
            if kind:
                out.append({'kind': kind, 'repo': repo, 'pack': pack, 'path': path,
                            'page': page_of(path), 'block': i,
                            'en_round': er, 'en_turn': et, 'cn_round': cr, 'cn_turn': ct,
                            'en': pe[:400], 'cn': pc[:400]})
    seen, uniq = set(), []
    for r in out:
        k = (r['page'], r['block'], r['en'][:160])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print(f'含 round/turn 机制词的块：{nb}')
    print(f'疑似单位互换：{len(out)}（孪生去重后 {len(uniq)}）')
    for r in uniq[:60]:
        print('\n' + '-' * 88)
        print(f'[{r["kind"]}] {r["repo"]} / {r["page"]} [b{r["block"]}] '
              f'en(r{r["en_round"]}/t{r["en_turn"]}) cn(轮{r["cn_round"]}/回合{r["cn_turn"]})')
        print('   EN:', r['en'][:300])
        print('   CN:', r['cn'][:300])
    json.dump(uniq, open(os.path.join(OUT, 'xp_p5_round_turn.json'), 'w',
                         encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', os.path.join(OUT, 'xp_p5_round_turn.json'))


if __name__ == '__main__':
    main()
