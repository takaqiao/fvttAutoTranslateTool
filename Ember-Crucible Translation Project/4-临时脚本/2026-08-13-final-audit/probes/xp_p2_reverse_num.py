#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2：**反向**数值闸 —— 中文里带机制量词的数字，英文侧根本没有。

为什么是盲区
------------
`scan_content_coverage` 只走 EN -> CN 一个方向（「英文有的数字中文得有」）。
CN 侧凭空多出或改动的数字它一律看不见：英文写 `two Jahud Assassins`（无阿拉伯数字），
中文写「3 名」，EN 需求集为空，判据静默。第十三轮那一处是人肉读出来的，不是判据抓的。

判据
----
对每个叶子：
  EN 允许集 = EN 阿拉伯数字 ∪ EN 英文数词折算值 ∪ 量词换算（dozen/score/decade…）
              ∪ 每个阿拉伯数字的 ±（简单加减派生：n、n±1 不算，只放宽到 100 的倍数换算）
  CN 需求   = 中文里「数字 + 机制量词」的数字（量词见 CN_UNIT）
  报出 CN 需求 - EN 允许集。

假阳性模式
----------
* 机关参数里的数字（`[[/dc 15]]`、`@Advantage[2]`、UUID）两侧都已剥掉，不参与。
* 中文把 `a d6` 写成「1d6」、把 `once per round` 写成「每轮 1 次」—— 这类由英文冠词/
  单数隐含的 1 会被报出来，故 **数字 1 与 2 默认降权**（`--min-val` 控制，默认 3）。
* 中文合并单位（`10 feet` -> 「10 英尺」没问题；但 `ten-foot` -> 「10 英尺」时 EN 侧
  的 ten 已由数词表覆盖）。
* 百分比、年份、页码等非机制数字：量词表不含它们，天然不进。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_common import CJK, NUM, plain, load_all, EN_WORD_NUM, EN_WORD_RE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'findings')
os.makedirs(OUT, exist_ok=True)

# 中文机制量词：数字紧跟这些字时，它一定是个游戏数值而不是编号/年份
CN_UNIT = ('英尺|尺|码|英里|轮|回合|个动作|动作|专注|点伤害|点|级|环|层|名|人|只|次|'
           '小时|分钟|天|日|周|年|骰|d\\d|格|面|倍|成|份|把|件|枚|颗|条|张|扇|座|头')
CN_NUM_UNIT = re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s*(?:点)?\s*(?:' + CN_UNIT + r')')

SCALE = {'decade': 10, 'dozen': 12, 'score': 20, 'century': 100,
         'millennium': 1000, 'hundred': 100, 'thousand': 1000, 'half': 2}


def en_allowed(pe: str):
    allow = set(NUM.findall(pe))
    ints = set()
    for a in list(allow):
        try:
            f = float(a)
            if f == int(f):
                ints.add(int(f))
        except ValueError:
            pass
    for m in EN_WORD_RE.finditer(pe):
        w = m.group(1).lower()
        v = EN_WORD_NUM[w]
        allow.add(str(v))
        ints.add(v)
    # 复合数词 twenty-five / twenty five
    for m in re.finditer(r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s-]'
                         r'(one|two|three|four|five|six|seven|eight|nine)\b', pe, re.I):
        allow.add(str(EN_WORD_NUM[m.group(1).lower()] + EN_WORD_NUM[m.group(2).lower()]))
    # N dozen / N score / N hundred …
    for m in re.finditer(r'(\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b)\s+'
                         r'(dozens?|scores?|decades?|centur(?:y|ies)|hundreds?|thousands?)',
                         pe, re.I):
        base = m.group(1)
        b = int(base) if base.isdigit() else EN_WORD_NUM[base.lower()]
        u = m.group(2).lower().rstrip('s')
        for k, v in SCALE.items():
            if u.startswith(k[:5]):
                allow.add(str(b * v))
    # 尺寸/骰面等常见派生：`d6` 里的 6 已在 NUM 里
    for i in ints:
        allow.add(str(i))
    return allow


def page_of(path):
    parts = path.split('.')
    if len(parts) >= 4 and parts[2] == 'pages':
        return f'{parts[1]}/{parts[3]}'
    return '.'.join(parts[:2])


def main():
    min_val = 3
    for i, a in enumerate(sys.argv):
        if a == '--min-val':
            min_val = int(sys.argv[i + 1])
    rows = load_all()
    findings = []
    checked = 0
    for repo, pack, path, en, cn in rows:
        if not cn or not CJK.search(cn):
            continue
        pe, pc = plain(en), plain(cn)
        if not pe:
            continue
        checked += 1
        allow = en_allowed(pe)
        extra = []
        for m in CN_NUM_UNIT.finditer(pc):
            n = m.group(1)
            if n in allow:
                continue
            try:
                if float(n) < min_val:
                    continue
            except ValueError:
                continue
            ctx = pc[max(0, m.start() - 30):m.end() + 18]
            extra.append((n, ctx))
        if extra:
            findings.append({'repo': repo, 'pack': pack, 'path': path,
                             'page': page_of(path),
                             'extra': [{'n': n, 'ctx': c} for n, c in extra[:6]],
                             'en': pe[:600], 'cn': pc[:600]})
    findings.sort(key=lambda f: -len(f['extra']))
    print(f'查了 {checked} 条已译叶子')
    print(f'中文侧「数字+机制量词」而英文侧没有该数字的：**{len(findings)}** 条')
    for f in findings[:60]:
        print('\n' + '=' * 90)
        print(f'{f["repo"]} / {f["pack"]} / {f["path"][:120]}')
        for e in f['extra']:
            print(f'   [{e["n"]}] …{e["ctx"]}…')
        print('   EN:', f['en'][:300])
    json.dump(findings, open(os.path.join(OUT, 'xp_p2_reverse_num.json'), 'w',
                             encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', os.path.join(OUT, 'xp_p2_reverse_num.json'))


if __name__ == '__main__':
    main()
