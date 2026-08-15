#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3：**块级双向**数值比对（含英文数词与中文数词）。

与 scan_content_coverage 的三处差别 —— 每一处都是它的盲区：
1. **粒度**：它按整叶比多重集，一页 journal 里几十个数字混成一个池子，
   「这一段的 30 被另一段的 30 顶掉」是常态。本探针按**块**（p/li/td/dt/dd/hN…）比。
2. **方向**：它只查 EN->CN。CN 侧改动/新增的数字它看不见。本探针双向。
3. **数词**：它只认 EN 侧阿拉伯数字。EN 写 `two`、`six hundred`、`fifty feet` 时它零需求，
   于是 `two` -> 「3 名」这类改数完全静默（第十三轮那处是人肉读出来的）。
   本探针把 EN 英文数词（含 six-hundred 这类复合）、CN 中文数词都折算成值再比。

判据（保守）
------------
* 只比**带机制量词**的数值。量词表刻意做小：LEN(feet/英尺)、ROUND(轮)、TURN(回合)、
  ACTION(动作)、FOCUS(专注)、HOUR/MIN、LEVEL(级/环)、DC。
  `里`(英里) 因与「里程碑」冲突整体剔除；`天` 加 `赋` 负向前瞻（「4 天赋点」不是 4 天）。
* 两侧都支持「数在前」「单位在前」两种词序（EN `Level 4` / `4 rounds`；CN `第4天` / `4级`）。
* 值 1 全免（英文冠词 a/an 隐含的 1、中文量词化差异噪声太大）。
* 中文出现「半」的块跳过 LEN（`five and a half feet` -> 「五英尺半」）。

假阳性模式
----------
* 中文把相邻两块的信息搬家 —— 会同时报一个 EN 独有和一个 CN 独有，看相邻块即可辨认。
* EN 用括号补注等级序列 `levels 5 (2d8), 11 (3d8), 17 (4d8)`：11/17 后没有 level 词，
  已用「levels 后跟逗号分隔数列」规则吸收；仍可能有变体漏吸。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_common import CJK, plain, split_blocks, load_all, EN_WORD_NUM, cn_num_value

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'findings')
os.makedirs(OUT, exist_ok=True)

SMALL = {k: v for k, v in EN_WORD_NUM.items() if k not in
         ('half', 'quarter', 'both', 'pair', 'couple', 'single', 'once', 'twice',
          'thrice', 'sole', 'lone', 'dual', 'double', 'triple', 'duo', 'trio',
          'decade', 'century', 'score', 'dozen')}
WORDS = '|'.join(sorted(SMALL, key=len, reverse=True))
NUMTOK = r'(?:\d+(?:\.\d+)?|(?:' + WORDS + r')(?:[\s-](?:hundred|thousand))?)'

UNIT_AFTER = {          # 数字在前：`30 feet`
    r'feet|foot|ft\.?': 'LEN',
    r'rounds?': 'ROUND',
    r'turns?': 'TURN',
    r'actions?': 'ACTION',
    r'focus': 'FOCUS',
    r'hours?': 'HOUR',
    r'minutes?': 'MIN',
}
UNIT_BEFORE = {         # 单位在前：`Level 4`（DC 单独走 EN_DC，放这里会双计）
    r'levels?': 'LEVEL',
}
EN_AFTER = re.compile(r'(?<![\w.])(' + NUMTOK + r')\s*[-‑]?\s*(' +
                      '|'.join(UNIT_AFTER) + r')\b', re.I)
EN_BEFORE = re.compile(r'\b(' + '|'.join(UNIT_BEFORE) + r')\s+(' + NUMTOK + r')(?![\w])',
                       re.I)
EN_ORD = re.compile(r'\b(\d+)(?:st|nd|rd|th)[\s-]level\b', re.I)
EN_ORDW = re.compile(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|'
                     r'tenth|twelfth|twentieth)[\s-]level\b', re.I)
ORDW = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
        'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, 'twelfth': 12,
        'twentieth': 20}
# `levels 5 (2d8), 11 (3d8), and 17 (4d8)` —— 序列里后续的数也是等级
EN_LEVELSEQ = re.compile(r'\blevels?\s+' + NUMTOK + r'\b(.{0,120}?)(?=[.;]|$)', re.I | re.S)

CN_AFTER = {'英尺': 'LEN', '尺': 'LEN', '轮': 'ROUND', '回合': 'TURN',
            '动作': 'ACTION', '专注': 'FOCUS', '小时': 'HOUR', '分钟': 'MIN',
            '级': 'LEVEL', '环': 'LEVEL', '层': 'LEVEL'}
CN_TOK = r'(?:\d+(?:\.\d+)?|[零〇一二三四五六七八九十百]+)'
CN_PAT = re.compile(r'(?<![\d.])(' + CN_TOK + r')\s*(?:点|个)?\s*(' +
                    '|'.join(sorted(CN_AFTER, key=len, reverse=True)) + r')')
CN_DC = re.compile(r'DC\s*(?:值\s*)?(?:为|是)?\s*(\d+)')
EN_DC = re.compile(r'\bDC\s*(\d+)', re.I)


def wordval(tok: str):
    t = tok.strip().lower()
    if re.fullmatch(r'\d+(\.\d+)?', t):
        return t
    m = re.fullmatch(r'(' + WORDS + r')(?:[\s-](hundred|thousand))?', t)
    if not m:
        return None
    v = SMALL.get(m.group(1))
    if v is None:
        return None
    if m.group(2):
        v *= 100 if m.group(2) == 'hundred' else 1000
    return str(v)


def en_pairs(s: str):
    c = Counter()
    for m in EN_AFTER.finditer(s):
        v = wordval(m.group(1))
        if v is None:
            continue
        u = m.group(2).lower()
        tag = next(t for p, t in UNIT_AFTER.items() if re.fullmatch(p, u, re.I))
        c[(v, tag)] += 1
    for m in EN_BEFORE.finditer(s):
        v = wordval(m.group(2))
        if v is None:
            continue
        u = m.group(1).lower()
        tag = next(t for p, t in UNIT_BEFORE.items() if re.fullmatch(p, u, re.I))
        c[(v, tag)] += 1
    for m in EN_ORD.finditer(s):
        c[(m.group(1), 'LEVEL')] += 1
    for m in EN_ORDW.finditer(s):
        c[(str(ORDW[m.group(1).lower()]), 'LEVEL')] += 1
    for m in EN_DC.finditer(s):
        c[(m.group(1), 'DC')] += 1
    return c


def en_level_seq(s: str):
    """`levels 5 (2d8), 11 (3d8), and 17 (4d8)` 里 11/17 也算等级。"""
    out = set()
    for m in EN_LEVELSEQ.finditer(s):
        for n in re.findall(r'(?<![\dd])(\d+)(?![\dd])', m.group(1)):
            out.add(n)
    return out


def cn_pairs(s: str):
    c = Counter()
    for m in CN_PAT.finditer(s):
        raw, unit = m.group(1), m.group(2)
        if raw[0].isdigit():
            v = raw
        else:
            val = cn_num_value(raw)
            if val is None:
                continue
            v = str(val)
        c[(v, CN_AFTER[unit])] += 1
    for m in CN_DC.finditer(s):
        c[(m.group(1), 'DC')] += 1
    return c


def norm(v):
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def page_of(path):
    parts = path.split('.')
    if len(parts) >= 4 and parts[2] == 'pages':
        return f'{parts[1]}/{parts[3]}'
    return '.'.join(parts[:2])


def main():
    rows = load_all()
    findings = []
    nblocks = 0
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
            ep, cp = en_pairs(pe), cn_pairs(pc)
            if not ep and not cp:
                continue
            nblocks += 1
            ep = Counter({(norm(v), u): n for (v, u), n in ep.items() if norm(v) != '1'})
            cp = Counter({(norm(v), u): n for (v, u), n in cp.items() if norm(v) != '1'})
            # 等级序列吸收
            seq = en_level_seq(pe)
            for v in seq:
                if (v, 'LEVEL') in cp and (v, 'LEVEL') not in ep:
                    ep[(v, 'LEVEL')] = cp[(v, 'LEVEL')]
            # 「半」：五英尺半 vs five and a half feet
            if '半' in pc:
                ep = Counter({k: n for k, n in ep.items() if k[1] != 'LEN'})
                cp = Counter({k: n for k, n in cp.items() if k[1] != 'LEN'})
            only_en = ep - cp
            only_cn = cp - ep
            if not only_en and not only_cn:
                continue
            findings.append({
                'repo': repo, 'pack': pack, 'path': path, 'page': page_of(path),
                'block': i,
                'only_en': [f'{v}{u}x{n}' for (v, u), n in sorted(only_en.items())],
                'only_cn': [f'{v}{u}x{n}' for (v, u), n in sorted(only_cn.items())],
                'en': pe[:500], 'cn': pc[:500],
            })
    seen = {}
    for f in findings:
        k = (f['page'], f['block'], f['en'][:200])
        seen.setdefault(k, []).append(f)
    uniq = [dict(v[0], dupes=len(v)) for v in seen.values()]
    uniq.sort(key=lambda f: -(len(f['only_en']) + len(f['only_cn'])))
    print(f'比对了 {nblocks} 个含机制数值的块')
    print(f'两侧 (值,单位) 多重集不等的块：{len(findings)}（孪生去重后 {len(uniq)}）')
    for f in uniq[:80]:
        print('\n' + '=' * 90)
        print(f'{f["repo"]} / {f["page"]} [b{f["block"]}] x{f["dupes"]}')
        print(f'   EN独有: {f["only_en"]}   CN独有: {f["only_cn"]}')
        print('   EN:', f['en'][:300])
        print('   CN:', f['cn'][:300])
    json.dump(uniq, open(os.path.join(OUT, 'xp_p3_block_num.json'), 'w',
                         encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', os.path.join(OUT, 'xp_p3_block_num.json'))


if __name__ == '__main__':
    main()
