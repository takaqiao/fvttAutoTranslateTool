#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针 2：角色级「双语并列」约定普查 + 违约叶清单 + 哪些判据结构性看不见它。

背景
----
本项目的角色约定：`*.name` 双语并列「护盾术 Shield」，
`tokenName` / `adjective` / `tables…results.name` / `scenes…levels` / `categories` 裸中文。

三个相关判据在比对**之前**都会剥掉双语尾巴：
  scan_same_en_split.strip_bilingual_tail（第十二轮新增）
  scan_name_splits.head()
  scan_token_name.head()
剥完之后「该带尾巴却没带」这一维度**从判据的定义域里消失**。
本探针把它捞回来：按角色统计双语率，找出高双语率角色里的裸中文叶。

只读。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
CJK = re.compile(r'[\u4e00-\u9fff]')
LATIN_WORD = re.compile(r'[A-Za-z]{2,}')
# scan_same_en_split 的归一（逐字复刻）
WRAPS = [("", ""), ("(", ")"), ("（", "）"), ("[", "]"), ("【", "】")]
SEPS = " \t\r\n　-—–~·:：,，、;；/|(（[【"


def strip_bilingual_tail(cn, en):
    s, e = (cn or "").strip(), (en or "").strip()
    if not e or s == e:
        return s
    for lb, rb in WRAPS:
        pat = lb + e + rb
        if len(pat) < len(s) and s.endswith(pat):
            head = s[:-len(pat)].rstrip(SEPS)
            if head and CJK.search(head):
                return head
    return s


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{p}.{k}' if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f'{p}.{i}' if p else str(i))
    elif isinstance(o, str) and o.strip():
        yield p, o


CONTAINERS = ('items', 'actions', 'effects', 'pages', 'journals', 'actors',
              'results', 'folders', 'tables', 'scenes', 'macros', 'regions',
              'levels', 'tokens', 'outcomes', 'categories', 'entries')


def subrole(top, path):
    """角色 = 末段键；`name` 再按其直接容器细分成 items.name / actions.name / …"""
    segs = [s for s in path.split('.') if not s.isdigit()]
    last = segs[-1] if segs else path
    if last != 'name':
        return last
    for s in reversed(segs[:-1]):
        if s in CONTAINERS:
            return f'{s}.name'
    return f'{top}.name'


def main():
    rows = []
    for repo in REPOS:
        rn = os.path.basename(repo)
        en_dir, cn_dir = os.path.join(repo, 'compendium', 'en'), os.path.join(repo, 'compendium', 'cn')
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json') or fn == '_source.json':
                continue
            cnp = os.path.join(cn_dir, fn)
            if not os.path.exists(cnp):
                continue
            en_doc, cn_doc = load(os.path.join(en_dir, fn)), load(cnp)
            for top in ('entries', 'folders'):
                cn_flat = dict(walk(cn_doc.get(top) or {}))
                for path, e in walk(en_doc.get(top) or {}):
                    c = cn_flat.get(path)
                    if not c or not CJK.search(c):
                        continue
                    rows.append((rn, fn, top, path, subrole(top, path), e, c))

    # 只看「短标签型」叶：英文含拉丁词、长度 <= 60、无 HTML/换行 —— 双语并列只适用于名称
    cand = [r for r in rows if len(r[5]) <= 60 and '<' not in r[5]
            and '\n' not in r[5] and LATIN_WORD.search(r[5])]
    print(f'名称型叶（英文<=60、含拉丁词、有中文）：{len(cand)}')

    bi = defaultdict(lambda: [0, 0, []])   # role -> [双语, 裸, 裸样本]
    for rn, fn, top, path, role, e, c in cand:
        if strip_bilingual_tail(c, e) != c.strip():
            bi[role][0] += 1
        else:
            bi[role][1] += 1
            bi[role][2].append((rn, fn, path, e, c))

    print(f'\n{"角色":<24}{"双语":>7}{"裸":>7}{"双语率":>9}')
    for role, (b, n, _) in sorted(bi.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        if b + n < 5:
            continue
        print(f'{role:<24}{b:>7}{n:>7}{b/(b+n):>9.1%}')

    # 违约：双语率 >= 90% 且样本 >= 20 的角色里的裸中文叶
    print('\n\n===== 高双语率角色（>=90%, n>=20）里的裸中文叶 =====')
    viol = []
    for role, (b, n, samples) in sorted(bi.items()):
        tot = b + n
        if tot < 20 or b / tot < 0.90 or n == 0:
            continue
        print(f'\n-- {role}  双语 {b} / 裸 {n}  ({b/tot:.1%})')
        for rn, fn, path, e, c in samples:
            viol.append({'repo': rn, 'pack': fn, 'path': path, 'en': e, 'cn': c, 'role': role})
            print(f'   {rn[:1]}/{fn:<32} {path[:96]}')
            print(f'      EN={e!r}  CN={c!r}')

    # 这些违约叶里，有多少会被 scan_same_en_split 的归一「收掉」
    # （＝同一英文串在别处是双语，归一后两者相同 -> 不再报为分叉）
    by_en = defaultdict(lambda: defaultdict(list))
    for rn, fn, top, path, role, e, c in rows:
        by_en[e][c].append(path)
    hidden = []
    for v in viol:
        d = by_en[v['en']]
        if len(d) < 2:
            continue
        merged = {strip_bilingual_tail(cn, v['en']) for cn in d}
        if len(merged) < 2:            # 归一后只剩一个变体 -> 该组不再被报
            hidden.append(v)
    print(f'\n\n违约叶共 {len(viol)} 条；其中 {len(hidden)} 条所在的同英文组'
          f'**因双语尾巴归一而整组不再被 scan_same_en_split 报出**：')
    for v in hidden:
        print(f'   {v["pack"]}::{v["path"]}  EN={v["en"]!r} CN={v["cn"]!r}')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'role_bilingual_violations.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'violations': viol, 'hidden_by_normalisation': hidden}, f,
                  ensure_ascii=False, indent=1)
    print(f'\n-> {out}')


main()
