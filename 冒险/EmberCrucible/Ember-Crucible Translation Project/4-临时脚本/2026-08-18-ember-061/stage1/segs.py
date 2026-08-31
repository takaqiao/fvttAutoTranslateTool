# -*- coding: utf-8 -*-
"""把一片叶切成 M（不可译标记）/ T（可译文本）交替的段序列。"""
import re

M_RE = re.compile(
    r'<[^>]+>'                       # HTML 标签
    r'|\[\[[^\]]*\]\]'               # [[/roll ...]]
    r'|&amp;Reference\[[^\]]*\]'     # &amp;Reference[Dodge]
    r'|&Reference\[[^\]]*\]'
    r'|@\w+\[[^\]]*\]'               # @UUID[...] 头（{标签} 归 T）
)


def seg(s):
    out = []
    i = 0
    for m in M_RE.finditer(s):
        if m.start() > i:
            out.append(('T', s[i:m.start()]))
        out.append(('M', m.group(0)))
        i = m.end()
    if i < len(s):
        out.append(('T', s[i:]))
    return out


def mseq(s):
    return [t for k, t in seg(s) if k == 'M']


def join(segs):
    return ''.join(t for _, t in segs)
