# -*- coding: utf-8 -*-
"""人物一致性镜头 C：**阵营标记**（缩写与全称）EN→CN 逐条核对 + 写法统计。

背景：第十轮的 Vinarith 事故就是阵营字母被改（NG→NE），把 GM 专属的反转泄给玩家。
本探针把全库所有「人物条」括号里的阵营 token 抽出来，按下标对齐 EN/CN。

抽取方式（不依赖 <dt>，避免嵌套 <dl> 造成的下标错位）
------------------------------------------------------
把整叶按 `PERSON_TAG` 正则扫：`(<align>, <ancestry…>, <pronouns>)`，
其中 align ∈ {9 个缩写, 9 个全称}，pronouns ∈ {she/her, he/him, they/them, …}。
EN 与 CN 各自扫出一串 person-tag，**只有条数相等的叶子才逐条对齐**（不等的单独计数）。

判据
----
1  WRONG  : CN 的阵营与 EN 不是同一个阵营（缩写→缩写、缩写→中文、全称→中文都查）
2  STYLE  : CN 把 EN 的缩写展开成中文，或把 EN 的全称缩成字母（写法漂移，非事实错误）
3  MISS   : CN 侧完全找不到阵营 token
"""
from __future__ import annotations
import json, os, re, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all

TAG = re.compile(r'<[^>]+>')
# 与 common.plain 不同：**保留** @UUID[...]{标签} 的标签文字，人名常在里面
UUIDL = re.compile(r'@[A-Za-z]+\[[^\]]*\]\{([^}]*)\}')
UUIDN = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]')
ENT = re.compile(r'&#?\w{1,8};')

ABBR = {'LG': 'lawful good', 'NG': 'neutral good', 'CG': 'chaotic good',
        'LN': 'lawful neutral', 'N': 'neutral', 'TN': 'neutral', 'CN': 'chaotic neutral',
        'LE': 'lawful evil', 'NE': 'neutral evil', 'CE': 'chaotic evil',
        'U': 'unaligned', 'NN': 'neutral'}
ZH = {'lawful good': '守序善良', 'neutral good': '中立善良', 'chaotic good': '混乱善良',
      'lawful neutral': '守序中立', 'neutral': '中立', 'chaotic neutral': '混乱中立',
      'lawful evil': '守序邪恶', 'neutral evil': '中立邪恶', 'chaotic evil': '混乱邪恶',
      'unaligned': '无阵营'}
FULL = sorted(ZH, key=len, reverse=True)

ALIGN_TOK = (r'(?:' + '|'.join(re.escape(a) for a in sorted(ABBR, key=len, reverse=True))
             + '|' + '|'.join(f.replace(' ', r'\s+') for f in FULL)
             + '|' + '|'.join(sorted(set(ZH.values()), key=len, reverse=True)) + r')')
PRONS = r'(?:she/her|he/him|they/them|it/its|he/they|she/they|they/she|they/he|xe/xem|ze/zir|any pronouns|no pronouns)'
# 括号可为半角/全角，逗号可为半角/全角
PERSON = re.compile(r'[(（]\s*(' + ALIGN_TOK + r')\s*[,，]([^)）]*?)[,，]\s*(' + PRONS + r')\s*[)）]', re.I)
# 有些条目没有括号：`Name Lawful Good, Arcturian Wirrun, she/her`
PERSON2 = re.compile(r'(?<![(（\w])(' + ALIGN_TOK + r')\s*[,，]([^()（）]{1,60}?)[,，]\s*(' + PRONS + r')(?![\w/])', re.I)


def plain(s):
    s = UUIDL.sub(lambda m: ' ' + m.group(1) + ' ', s)
    s = UUIDN.sub(' ', s)
    s = ENT.sub(' ', s)
    s = TAG.sub(' ', s)
    return re.sub(r'[ \t]+', ' ', s)


def norm(tok):
    """阵营 token -> 规范英文键；认不出返回 None。"""
    t = re.sub(r'\s+', ' ', tok.strip())
    if t in ZH.values():
        for k, v in ZH.items():
            if v == t:
                return k
    up = t.upper()
    if up in ABBR:
        return ABBR[up]
    lo = t.lower()
    if lo in ZH:
        return lo
    if lo == 'true neutral':
        return 'neutral'
    return None


def kind(tok):
    t = tok.strip()
    if t in ZH.values():
        return 'zh'
    if t.upper() in ABBR and t.upper() == t and len(t) <= 2:
        return 'abbr'
    return 'full'


def scan(txt):
    out, seen = [], set()
    for rx in (PERSON, PERSON2):
        for m in rx.finditer(txt):
            key = (m.start(), m.end())
            if any(s <= m.start() < e for s, e in seen):
                continue
            seen.add(key)
            out.append((m.start(), m.group(1), m.group(2).strip(), m.group(3)))
    out.sort()
    return out


def main():
    wrong, style, miss, unequal = [], [], [], []
    styles = Counter()
    total = 0
    for repo, pack, path, en, cn in load_all():
        if not cn:
            continue
        pe, pc = plain(en), plain(cn)
        se, sc = scan(pe), scan(pc)
        if not se:
            continue
        if len(se) != len(sc):
            unequal.append({'repo': repo, 'pack': pack, 'path': path,
                            'n_en': len(se), 'n_cn': len(sc),
                            'en_toks': [x[1] for x in se], 'cn_toks': [x[1] for x in sc]})
            continue
        for (pos_e, ae, anc_e, pr_e), (pos_c, ac, anc_c, pr_c) in zip(se, sc):
            total += 1
            ne, nc = norm(ae), norm(ac)
            ctx_e = pe[max(0, pos_e - 90):pos_e + 60].strip()
            ctx_c = pc[max(0, pos_c - 60):pos_c + 60].strip()
            rec = {'repo': repo, 'pack': pack, 'path': path,
                   'en_tok': ae, 'cn_tok': ac, 'en_norm': ne, 'cn_norm': nc,
                   'en_anc': anc_e, 'cn_anc': anc_c, 'pron_en': pr_e, 'pron_cn': pr_c,
                   'ctx_en': ctx_e, 'ctx_cn': ctx_c}
            if nc is None:
                miss.append(rec)
            elif ne != nc:
                wrong.append(rec)
            else:
                ke, kc = kind(ae), kind(ac)
                styles[f'{ke}->{kc}'] += 1
                if ke == 'abbr' and kc != 'abbr':
                    style.append(rec)
                elif ke == 'full' and kc != 'zh':
                    style.append(rec)
            if pr_e.lower() != pr_c.lower():
                wrong.append({**rec, 'pron_mismatch': True})

    print(f'对齐成功的人物条 {total}；EN/CN 条数不等的叶 {len(unequal)}')
    print('写法分布', dict(styles))
    print(f'阵营/代词不符 {len(wrong)}   写法漂移 {len(style)}   中文缺阵营 {len(miss)}')
    for lbl, rs in (('WRONG', wrong), ('STYLE', style), ('MISS', miss)):
        for r in rs[:60]:
            print(f'\n[{lbl}] {r["repo"]}/{r["pack"]}/{r["path"][-70:]}')
            print(f'   EN tok={r["en_tok"]!r} anc={r["en_anc"]!r} pron={r["pron_en"]}')
            print(f'   CN tok={r["cn_tok"]!r} anc={r["cn_anc"]!r} pron={r["pron_cn"]}')
            print(f'   ctxEN: ...{r["ctx_en"]}')
            print(f'   ctxCN: ...{r["ctx_cn"]}')
    for u in unequal[:20]:
        print(f'\n[UNEQ] {u["repo"]}/{u["pack"]}/{u["path"][-70:]} en={u["n_en"]} cn={u["n_cn"]}')
        print('   EN', u['en_toks'])
        print('   CN', u['cn_toks'])
    json.dump({'wrong': wrong, 'style': style, 'miss': miss, 'unequal': unequal,
               'styles': dict(styles), 'total': total},
              open('pc_align.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
