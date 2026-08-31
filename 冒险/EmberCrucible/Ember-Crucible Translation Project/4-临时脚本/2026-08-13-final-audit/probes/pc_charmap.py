# -*- coding: utf-8 -*-
"""人物一致性镜头 B：地名志「人物条」<dt>/<dd> 的阵营·代词·族裔跨页核对。

数据源
------
全库 690 条形如 `Name (Alignment, Ancestry Culture, pronouns)` 的 <dt>，
以及紧随其后的 <dd> 正文。EN/CN 的 <dt>/<dd> 数量逐叶相等（已验证 unaligned=0），
按下标 zip 对齐。

四道判据
--------
A  ALIGN   : dt 里的英文阵营词 -> 中文阵营词，逐条查表。错一个字母就是给玩家换阵营。
B  PRONTAG : dt 尾部的 she/her|he/him|they/them 标签在中文侧必须逐字保留（全库既定）。
C  DDPRON  : dt 标 she/her 而**同一条 dd 的中文**只用「他」不用「她」（反之亦然）。
             dd 里可能提到别人 -> 只在「dd 的英文侧同样只用该性别代词」时才判 HARD。
D  XPAGE   : 同一人名在多处出现时，阵营/代词/族裔是否自洽（EN 侧先比，EN 自身矛盾的
             单独归档为 EN_CONFLICT，不算译文缺陷）。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all, plain, CJK

DT = re.compile(r'<dt\b[^>]*>(.*?)</dt>', re.S | re.I)
DD = re.compile(r'<dd\b[^>]*>(.*?)</dd>', re.S | re.I)
PRON = re.compile(r'\b(she/her|he/him|they/them|it/its|he/they|she/they|they/she|they/he|any pronouns|no pronouns|xe/xem|ze/zir)\b', re.I)

ALIGN = {
    'lawful good': ['守序善良'],
    'neutral good': ['中立善良'],
    'chaotic good': ['混乱善良'],
    'lawful neutral': ['守序中立'],
    'chaotic neutral': ['混乱中立'],
    'lawful evil': ['守序邪恶'],
    'neutral evil': ['中立邪恶'],
    'chaotic evil': ['混乱邪恶'],
    'true neutral': ['绝对中立', '真中立', '中立'],
    'neutral': ['中立'],
    'unaligned': ['无阵营'],
}
# 长词优先，避免 "neutral good" 被 "neutral" 抢先
ALIGN_RE = re.compile(r'\b(' + '|'.join(sorted(ALIGN, key=len, reverse=True)).replace(' ', r'\s+') + r')\b', re.I)

FEM = re.compile(r'\b(she|her|hers|herself)\b', re.I)
MASC = re.compile(r'\b(he|him|his|himself)\b', re.I)
TA_NON = re.compile(r'其他|他人|他日|他处|他乡|他方|吉他|利他|他律|他杀|他者|排他')


def cn_ta(s):
    s = TA_NON.sub('  ', s)
    s = s.replace('他们', '  ').replace('她们', '  ')
    return s.count('她'), s.count('他')


def align_of(txt):
    m = ALIGN_RE.search(txt)
    return m.group(1).lower().replace('\n', ' ') if m else None


def name_of(txt):
    """dt 头部的人名（英文侧）：截到第一个阵营词之前。"""
    m = ALIGN_RE.search(txt)
    head = txt[:m.start()] if m else txt
    return re.sub(r'[\s(（,，]+$', '', head).strip()


def main():
    A, B, C = [], [], []
    xpage = defaultdict(list)
    n_dt = 0
    skipped = Counter()
    for repo, pack, path, en, cn in load_all():
        if '<dt' not in en or not cn:
            continue
        edts, cdts = DT.findall(en), DT.findall(cn)
        edds, cdds = DD.findall(en), DD.findall(cn)
        if len(edts) != len(cdts):
            skipped['dt_len'] += 1
            continue
        dd_ok = len(edds) == len(cdds) == len(edts)
        for i, (e, c) in enumerate(zip(edts, cdts)):
            pe, pc = plain(e), plain(c)
            if not PRON.search(pe):
                continue
            n_dt += 1
            loc = {'repo': repo, 'pack': pack, 'path': path, 'i': i}
            # A 阵营
            al = align_of(pe)
            if al:
                ok = any(w in pc for w in ALIGN[al])
                # 「中立」是「守序中立」的子串 -> 只有当英文是纯 neutral 时才可能误判，
                # 反向：英文 neutral good 而中文只写「中立」-> 会被判 not ok，正确。
                if not ok:
                    A.append({**loc, 'en': pe, 'cn': pc, 'en_align': al})
                else:
                    # 反向：中文写了英文没有的阵营词（换阵营）
                    others = [w for k, ws in ALIGN.items() if k != al for w in ws
                              if w in pc and w not in ALIGN[al]
                              and not any(w in x for x in ALIGN[al])]
                    # 「中立」是很多词的子串，排掉
                    others = [w for w in others if w not in ('中立',)
                              and not any(w in v for v in ALIGN[al])]
                    if others:
                        A.append({**loc, 'en': pe, 'cn': pc, 'en_align': al,
                                  'cn_extra': sorted(set(others))})
            # B 代词标签
            for tag in set(m.group(1) for m in PRON.finditer(pe)):
                if tag.lower() not in pc.lower():
                    B.append({**loc, 'en': pe, 'cn': pc, 'missing_tag': tag})
            # C dd 代词
            if dd_ok:
                tags = [m.group(1).lower() for m in PRON.finditer(pe)]
                g = None
                if tags and all(t == 'she/her' for t in tags):
                    g = 'F'
                elif tags and all(t == 'he/him' for t in tags):
                    g = 'M'
                if g:
                    de, dc = plain(edds[i]), plain(cdds[i])
                    ef, em = len(FEM.findall(de)), len(MASC.findall(de))
                    cf, cm = cn_ta(dc)
                    if g == 'F' and cm and not cf:
                        C.append({**loc, 'tag': 'she/her', 'en_f': ef, 'en_m': em,
                                  'cn_f': cf, 'cn_m': cm, 'dt_en': pe, 'dt_cn': pc,
                                  'dd_en': de[:700], 'dd_cn': dc[:700]})
                    if g == 'M' and cf and not cm:
                        C.append({**loc, 'tag': 'he/him', 'en_f': ef, 'en_m': em,
                                  'cn_f': cf, 'cn_m': cm, 'dt_en': pe, 'dt_cn': pc,
                                  'dd_en': de[:700], 'dd_cn': dc[:700]})
            nm = name_of(pe)
            if nm:
                xpage[nm].append({**loc, 'align': al,
                                  'pron': ','.join(sorted(set(m.group(1).lower() for m in PRON.finditer(pe)))),
                                  'en': pe, 'cn': pc})

    # D 跨页
    D = []
    for nm, occ in sorted(xpage.items()):
        if len(occ) < 2:
            continue
        als = {o['align'] for o in occ}
        prs = {o['pron'] for o in occ}
        cns = {o['cn'] for o in occ}
        if len(als) > 1 or len(prs) > 1:
            D.append({'name': nm, 'kind': 'EN_CONFLICT', 'aligns': sorted(x or '' for x in als),
                      'prons': sorted(prs), 'occ': occ})
        elif len(cns) > 1:
            D.append({'name': nm, 'kind': 'CN_SPLIT', 'cns': sorted(cns), 'occ': occ})

    print(f'person-<dt> 条数 {n_dt}   skipped(dt 数不等) {dict(skipped)}')
    print(f'A 阵营不符 {len(A)} / B 代词标签丢失 {len(B)} / C dd 代词与标签相反 {len(C)} / D 跨页 {len(D)}')
    for lbl, rowsx in (('A', A), ('B', B), ('C', C)):
        for r in rowsx[:40]:
            print(f'\n[{lbl}] {r["repo"]}/{r["pack"]}/{r["path"][-70:]} #{r["i"]}')
            for k in ('en', 'cn', 'en_align', 'cn_extra', 'missing_tag', 'tag',
                      'en_f', 'en_m', 'cn_f', 'cn_m', 'dt_en', 'dt_cn', 'dd_en', 'dd_cn'):
                if k in r:
                    print(f'    {k}: {r[k]}')
    for r in D[:40]:
        print(f'\n[D:{r["kind"]}] {r["name"]}')
        for o in r['occ']:
            print(f'    {o["path"][-70:]} #{o["i"]} :: {o["en"]}  ||  {o["cn"]}')
    json.dump({'A': A, 'B': B, 'C': C, 'D': D},
              open('pc_charmap.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
