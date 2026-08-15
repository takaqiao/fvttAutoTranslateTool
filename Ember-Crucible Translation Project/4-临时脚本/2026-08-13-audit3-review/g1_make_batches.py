#!/usr/bin/env python3
"""G1 批次生成：以 compendium/cn 现值为底，做定点字符串替换，产出扁平批次 JSON。

每条替换都带 expect（预期命中次数），命中数不符就直接报错退出 ——
批次值是整叶覆盖，替换错了会静默毁掉一整页译文。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
OUT = r'C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches'
EMBER = os.path.join(P, '1-Ember汉化插件')

GOOD = 'Ember Early Access.journals.Ooze Control.pages.Good Ooze, Bad Ooze.text'
ALCH = 'Ember Early Access.journals.Ooze Control.pages.Alchemical Decisions.text'
WAYW = 'Ember Early Access.journals.Ooze Control.pages.Wayward Sampler.text'
HARR = 'Ember Early Access.journals.Crumbling Sanctuary.pages.Harrowed Crossing.text'
QUAY = "Ember Early Access.journals.Flotsam Canal Market.pages.Merchant's Quay.text"
BARG = 'Ember Early Access.journals.Flotsam Canal Market.pages.The Bread Barge.text'
ADEL = 'Ember Early Access.actors.Adelyne Goss.items.Anachraenum Member.description'
FERN = 'Ember Early Access.actors.Fernis Ossa.items.Anachraenum Member.description'

SURP = ('&amp;Reference[surprise]{Surprised}', '&amp;Reference[surprise]{突袭}')
SURP2 = ('&amp;Reference[Surprise]{Surprised}', '&amp;Reference[Surprise]{突袭}')
SURP3 = ('&amp;Reference[surprise]{surprised}', '&amp;Reference[surprise]{措手不及}')
MEALS = ('计作 Poor 或 Modest &amp;Reference[food]{Meals}',
         '计作贫寒或小康级的&amp;Reference[food]{餐食}')
BARGE = ('而且总是 eager to make a good impression on new patrons。',
         '而且总是热切地想给新顾客留下好印象。')

SQUISH = 'SQUISH'          # 特殊算子：Squish -> 压扁 并收拾中英之间的空格
# @Embed 的 label 改用**带引号**形态，与库内其余 44 处 `label="…"` 一致。
# 不加引号的话 scan_markup_targets 的 BY_DESIGN_BODY(`=\s*"|#`) 认不出这是参数值，
# 会把方括号里的中文报成「标记被译坏」（5.4 第 2b 项）。
LABEL = ('label=压扁]', 'label="压扁"]')

# (pack, path, [(old, new, expect), ...])
EDITS = [
    ('ember.adventure.json', GOOD, [SURP + (2,), (SQUISH, '', 10), LABEL + (1,)]),
    ('ember.crucible-adventure.json', GOOD, [SURP + (2,), (SQUISH, '', 10), LABEL + (1,)]),
    ('ember.adventure.json', ALCH, [(SQUISH, '', 8)]),
    ('ember.crucible-adventure.json', ALCH, [(SQUISH, '', 8)]),
    ('ember.adventure.json', WAYW, [(SQUISH, '', 4)]),
    ('ember.crucible-adventure.json', WAYW, [(SQUISH, '', 4)]),
    ('ember.adventure.json', HARR, [SURP2 + (2,)]),
    ('ember.adventure.json', QUAY, [MEALS + (1,)]),
    ('ember.crucible-adventure.json', QUAY, [MEALS + (1,)]),
    ('ember.adventure.json', BARG, [BARGE + (1,)]),
    ('ember.crucible-adventure.json', ADEL, [SURP3 + (1,)]),
    ('ember.crucible-adventure.json', FERN, [SURP3 + (1,)]),
]

CJKC = '\u4e00-\u9fff'
SQ = re.compile(r'\bSquish\b')
# 替换后收空格：中文/中文标点与「压扁」之间不留空格（本库既有风格）
SP1 = re.compile(r'(?<=[' + CJKC + r'“”‘’（）、，。！？；：])[ \u00a0]+(?=压扁)')
SP2 = re.compile(r'(?<=压扁)[ \u00a0]+(?=[' + CJKC + r'“”‘’（）、，。！？；：])')


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


def main():
    os.makedirs(OUT, exist_ok=True)
    by_pack = defaultdict(dict)
    for pack, path, ops in EDITS:
        p = os.path.join(EMBER, 'compendium', 'cn', pack)
        L = []
        walk(json.load(open(p, encoding='utf-8')).get('entries', {}), [], L)
        d = dict(L)
        if path not in d:
            sys.exit(f'路径不存在: {pack} :: {path}')
        s = d[path]
        for op in ops:
            old, new, expect = op
            if old == SQUISH:
                n = len(SQ.findall(s))
                if n != expect:
                    sys.exit(f'Squish 命中 {n} != {expect}  ({pack} :: {path})')
                s = SQ.sub('压扁', s)
                s = SP1.sub('', s)
                s = SP2.sub('', s)
            else:
                n = s.count(old)
                if n != expect:
                    sys.exit(f'命中 {n} != {expect}: {old!r}  ({pack} :: {path})')
                s = s.replace(old, new)
        if s == d[path]:
            sys.exit(f'无变化: {pack} :: {path}')
        by_pack[pack][path] = s

    for pack, batch in by_pack.items():
        f = os.path.join(OUT, f'G1__ember__{pack[:-5]}.json')
        with open(f, 'w', encoding='utf-8') as fh:
            json.dump(batch, fh, ensure_ascii=False, indent=1)
        print(f'-> {f}   {len(batch)} 条')


if __name__ == '__main__':
    main()
