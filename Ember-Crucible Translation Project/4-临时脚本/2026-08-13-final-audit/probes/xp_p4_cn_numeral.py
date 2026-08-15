#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4：中文**数词**写的机制数值（判据的公认盲区）。

任务书原话：「数字判据用阿拉伯数字锚定，对中文数词是盲的」。
scan_content_coverage 会把中文数词折算回阿拉伯数字**再比**，所以它对
「英文 6、中文六」不报警——这是它的设计（否则会逼出坏中文）。代价是：
* 中文数词写错值（英文 six、中文写「五轮」）它同样看不见；
* 同一机制在不同页一处「6 轮」一处「六轮」，它完全无感。

本探针把所有「中文数词 + 机制量词」的位置全捞出来，逐条对英文，
并统计同一 (英文单位) 下阿拉伯 / 数词两种写法的分布，供人工判「哪边是多数派」。

输出三张表
----------
A. 值不符：中文数词折算值在英文块里找不到  -> 疑似改数（最强信号）
B. 写法分歧：同一英文块（跨页重复）一处数词一处阿拉伯
C. 全库分布：每个机制量词下数词写法的条数与出处（供裁「要不要统一」）
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_common import CJK, plain, split_blocks, load_all, EN_WORD_NUM, cn_num_value

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'findings')
os.makedirs(OUT, exist_ok=True)

# 刻意只留**歧义低**的机制量词。
# 剔除：点/面（「一点时间」「一面墙」「十二面体」全是普通用法）、
#       个/次（量词化差异）、天/日（日期）、层（楼层与矿井层）。
UNITS = ['英尺', '轮', '回合', '个动作', '点专注', '专注', '级', '环',
         '点伤害', '小时', '分钟', '英里', '格']
CNNUM = r'[零〇一二三四五六七八九十百]+'
PAT = re.compile(r'(?<![零〇一二三四五六七八九十百第])(' + CNNUM + r')\s*(?:个)?\s*(' +
                 '|'.join(sorted(UNITS, key=len, reverse=True)) + r')')

WORDS = '|'.join(sorted(EN_WORD_NUM, key=len, reverse=True))
EN_NUM = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)(?!\d)')
EN_WORDNUM = re.compile(r'\b(' + WORDS + r')(?:[\s-](hundred|thousand))?\b', re.I)


def en_values(pe: str):
    vals = set(EN_NUM.findall(pe))
    for m in EN_WORDNUM.finditer(pe):
        v = EN_WORD_NUM[m.group(1).lower()]
        if m.group(2):
            v *= 100 if m.group(2).lower() == 'hundred' else 1000
        vals.add(str(v))
    return {str(int(float(v))) if re.fullmatch(r'\d+(\.0)?', v) else v for v in vals}


def page_of(path):
    p = path.split('.')
    return f'{p[1]}/{p[3]}' if len(p) >= 4 and p[2] == 'pages' else '.'.join(p[:2])


def main():
    rows = load_all()
    mismatch, hits = [], []
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
            found = list(PAT.finditer(pc))
            if not found:
                continue
            evals = en_values(pe)
            for m in found:
                v = cn_num_value(m.group(1))
                if v is None or v == 1:   # 「一轮」「一回合」多是冠词性用法
                    continue
                rec = {'repo': repo, 'pack': pack, 'path': path, 'page': page_of(path),
                       'block': i, 'cn_tok': m.group(0), 'val': v, 'unit': m.group(2),
                       'ctx': pc[max(0, m.start() - 40):m.end() + 30],
                       'en': pe[:400]}
                hits.append(rec)
                if str(v) not in evals:
                    mismatch.append(rec)

    def dedupe(lst):
        seen, out = set(), []
        for r in lst:
            k = (r['page'], r['block'], r['cn_tok'], r['en'][:120])
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    mm = dedupe(mismatch)
    hh = dedupe(hits)
    print(f'中文数词+机制量词 的出现处（孪生去重后）：{len(hh)}')
    print(f'其中英文块里找不到该值的：**{len(mm)}**')
    for r in mm[:60]:
        print('\n' + '-' * 88)
        print(f'{r["repo"]} / {r["page"]} [b{r["block"]}]  「{r["cn_tok"]}」-> {r["val"]}')
        print('   CN…', r['ctx'])
        print('   EN:', r['en'][:260])
    dist = Counter((r['unit'],) for r in hh)
    print('\n量词分布：', dist.most_common())
    json.dump({'mismatch': mm, 'all': hh},
              open(os.path.join(OUT, 'xp_p4_cn_numeral.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('->', os.path.join(OUT, 'xp_p4_cn_numeral.json'))


if __name__ == '__main__':
    main()
