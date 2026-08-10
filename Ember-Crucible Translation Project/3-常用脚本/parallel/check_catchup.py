#!/usr/bin/env python3
"""「追平」批专用的完成度自检：中文有没有覆盖今天英文里的数字与篇幅。

  python check_catchup.py <单元目录> [<页 id>]

为什么这批不能拿闸门当完成信号
------------------------------
`apply_translations.py` 的闸门比的是**标记多重集**。而「追平」批的条目是按**内容**
漂移挑出来的 —— 它们的标记本来就是对的，所以闸门在动手前后**都是 0 拒绝**。
把「REJECTED 全 0」当交付标准，等于没有标准（第一次发这批时就写错了）。

真正的信号有两个，都在这里：

* **数字覆盖** —— 英文正文里的阿拉伯数字（DC、尺数、伤害、轮数、Size 级差、次数），
  中文里必须还在。这是最硬的一条：丢了数字就是丢了规则。
  比对前会把中文数字折算成阿拉伯数字（三层→3、两个→2、第一→1），
  **不折算的话这条规则会逼出坏中文** —— 首版就把全库 16 处的「三层矿井」逼成了「3 层矿井」。
* **长度比** —— 中文纯文本 / 英文纯文本。本库中位 **0.31**，正常区间 0.25–0.45。
  低于 0.25 基本可以断定还在缩写。

两条都过 + `collect_realign.py` 报 `UNTOUCHED 0`，才算这一单元做完。
"""
from __future__ import annotations
import json
import os
import re
import sys

TAG = re.compile(r'<[^>]+>')
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]|&(?:amp;)?[Rr]eference\[[^\]]*\]')
# 不能用 \w 做边界：中文里数字紧贴汉字（「其AC为17」），汉字算 \w 会把中文侧数字全挡掉
NUM = re.compile(r'(?<!\d)(\d+(?:\.\d+)?)(?!\d)')


CN_DIGIT = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9}
CN_NUM = re.compile(r'[零〇一二三四五六七八九十百千两]+')


def cn_numerals_to_digits(s: str) -> str:
    """把中文数字也折算成阿拉伯数字后再比对。

    **这一步是必须的，否则本检查会逼出坏中文。** 首版只认阿拉伯数字，于是译者为了
    让「英文里的 3」出现在中文里，把全库 16 处都写作「三层矿井」的地名改成了
    「3 层矿井」；`two` 被写成「2 个战斗阶段」、`1st Host` 写成「第 1 军团」、
    `2 decades` 写成「2 个十年」。中文本来就该写三/两/第一 —— 是规则错了，不是译文错了。

    只做粗折算（够用即可）：十=10、二十=20、三层=3、第一=1。
    """
    def conv(m):
        t = m.group()
        if t == '十':
            return '10'
        if len(t) == 2 and t[0] == '十':          # 十一 .. 十九
            return str(10 + CN_DIGIT.get(t[1], 0))
        if len(t) == 2 and t[1] == '十':          # 二十 .. 九十
            return str(CN_DIGIT.get(t[0], 0) * 10)
        if len(t) == 3 and t[1] == '十':          # 二十五
            return str(CN_DIGIT.get(t[0], 0) * 10 + CN_DIGIT.get(t[2], 0))
        if len(t) == 1:
            return str(CN_DIGIT.get(t, ''))
        return t
    return CN_NUM.sub(lambda m: ' ' + conv(m) + ' ', s)


# HTML 实体里带数字：`Jeweler&#x27;s` 的 `&#x27;` 会被当成数字 27 报缺失。
# 必须先解实体再抽数字，否则每个带撇号的英文名都会造一个假警报。
ENTITY = re.compile(r'&#?\w{1,8};')


def plain(s, cn=False):
    out = TAG.sub(' ', ENTITY.sub(' ', MARKUP.sub(' ', s)))
    return cn_numerals_to_digits(out) if cn else out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        raise SystemExit('usage: check_catchup.py <单元目录> [<页 id>]')
    unit = args[0]
    only = args[1] if len(args) > 1 else None

    index = json.load(open(os.path.join(unit, 'index.json'), encoding='utf-8'))
    bad = 0
    for page in index['pages']:
        i = str(page['id'])
        if only is not None and i != str(only):
            continue
        pe = os.path.join(unit, 'pages', f'{i}.en.html')
        pc = os.path.join(unit, 'pages', f'{i}.cn.html')
        if not (os.path.exists(pe) and os.path.exists(pc)):
            continue
        e = plain(open(pe, encoding='utf-8-sig').read())
        c = plain(open(pc, encoding='utf-8-sig').read(), cn=True)
        miss = sorted(set(NUM.findall(e)) - set(NUM.findall(c)), key=lambda x: -len(x))
        ratio = len(c) / max(len(e), 1)
        problems = []
        if miss:
            problems.append(f'缺数字 {miss[:10]}')
        if ratio < 0.25:
            problems.append(f'长度比 {ratio:.2f} < 0.25（还在缩写）')
        if problems:
            bad += 1
            print(f'  [{i}] ' + ' | '.join(problems))
            print(f'       {page["path"][-72:]}')
            for r in page.get('reason', [])[:2]:
                print(f'       ← {r}')

    print(f'\n{index["unit"]}: {bad} 页未达标（数字覆盖 + 长度比）')
    print('达标后还要 collect_realign.py 报 UNTOUCHED 0 才算做完')
    raise SystemExit(1 if bad else 0)


main()
