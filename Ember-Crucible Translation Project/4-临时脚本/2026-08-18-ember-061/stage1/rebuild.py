# -*- coding: utf-8 -*-
"""结构重建器：拿 EN061 的 HTML 骨架 + CN 的既有译文，拼出新的 CN，
并把「新出现 / 被改写」的文本槽标出来，等人工填。

前提（已实测 212/214 成立）：htags(EN060) == htags(CN)（去掉 id/data-anchor 属性后逐个相等），
所以 EN060 与 CN 的 [内容, 标签, 内容, 标签 …] 流是同长同位的。
"""
import re, difflib

TAG = re.compile(r'<[^>]+>')
IDATTR = re.compile(r'\s+(?:id|data-anchor)="[^"]*"')


def norm_tag(t):
    return IDATTR.sub('', t)


def stream(s):
    """-> [c0, t0, c1, t1, …, cn]  （偶数位是内容，奇数位是 HTML 标签）"""
    out = []
    i = 0
    for m in TAG.finditer(s):
        out.append(s[i:m.start()])
        out.append(m.group(0))
        i = m.end()
    out.append(s[i:])
    return out


def norm_stream(st):
    return [norm_tag(x) if k % 2 else x for k, x in enumerate(st)]


def _has_letter(t):
    return bool(re.search(r'[A-Za-z一-鿿]', t))


class Slot:
    """需要人工填的内容槽。"""
    def __init__(self, kind, en_new, en_old=None, cn_old=None):
        self.kind = kind          # 'NEW' 新增内容 / 'PATCH' 改写内容
        self.en_new = en_new
        self.en_old = en_old
        self.cn_old = cn_old


def rebuild(old_en, new_en, cn):
    """返回 (parts, slots)：parts 是字符串与 Slot 交替的列表；join 后即新 CN。"""
    S0, S1, SC = stream(old_en), stream(new_en), stream(cn)
    assert len(S0) == len(SC), f"EN060 与 CN 流长不等 {len(S0)} vs {len(SC)}"
    N0, N1 = norm_stream(S0), norm_stream(S1)
    parts, slots = [], []
    sm = difflib.SequenceMatcher(None, N0, N1, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i1, i2):
                parts.append(SC[k])          # 用 CN 的（标签保留 CN 的 id）
        elif tag == 'delete':
            continue
        elif tag == 'insert':
            for k in range(j1, j2):
                if k % 2:                     # 新标签
                    parts.append(S1[k])
                else:
                    if not _has_letter(S1[k]):
                        parts.append(S1[k])   # 纯符号/空白，直接抄
                    else:
                        s = Slot('NEW', S1[k]); slots.append(s); parts.append(s)
        else:  # replace
            # 逐位配：能对上位置的按位配，多出来的走 NEW / 丢弃
            n0, n1 = i2 - i1, j2 - j1
            for d in range(n1):
                k1 = j1 + d
                k0 = i1 + d if d < n0 else None
                if k1 % 2:
                    parts.append(S1[k1])      # 标签一律用 EN061 的
                    continue
                aligned = k0 is not None and k0 % 2 == 0
                if aligned and S0[k0] == S1[k1]:
                    parts.append(SC[k0])      # 内容其实没变，照抄 CN
                    continue
                if not _has_letter(S1[k1]):
                    parts.append(S1[k1])
                    continue
                if aligned and _has_letter(S0[k0]):
                    s = Slot('PATCH', S1[k1], S0[k0], SC[k0])
                else:
                    s = Slot('NEW', S1[k1])
                slots.append(s); parts.append(s)
    return parts, slots


def render(parts, fill=None):
    """fill: {slot_index: 中文串}；未填的槽用 ⟦…⟧ 标出来。"""
    out = []
    si = 0
    for p in parts:
        if isinstance(p, Slot):
            if fill is not None and si in fill:
                out.append(fill[si])
            else:
                out.append(f"⟦{si}:{p.kind}⟧")
            si += 1
        else:
            out.append(p)
    return ''.join(out)


def reattach_ids(new_cn, cn_old):
    """EN061 新引入的标题标签没有 id；按标题文字把 CN 原有的 id 补回去。"""
    ids = {}
    for m in re.finditer(r'<(h[1-6])([^>]*)>(.*?)</\1>', cn_old, re.S):
        idm = IDATTR.search(m.group(2))
        if idm:
            ids[m.group(3).strip()] = idm.group(0)
    if not ids:
        return new_cn

    def rep(m):
        if IDATTR.search(m.group(2)):
            return m.group(0)
        a = ids.get(m.group(3).strip())
        return f"<{m.group(1)}{m.group(2)}{a}>{m.group(3)}</{m.group(1)}>" if a else m.group(0)
    return re.sub(r'<(h[1-6])([^>]*)>(.*?)</\1>', rep, new_cn, flags=re.S)
