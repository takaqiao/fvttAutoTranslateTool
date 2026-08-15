# -*- coding: utf-8 -*-
r"""
scan_roleblind_glossary_pool.py
—— 「对异质集合施加同一处变换，不做逐成员角色判据」这一类，落在**术语表构建**上的实例。

判据抽象（与已确认实例同形）
----------------------------
已确认实例：`tm/fill_missing.py:133` 的退化分支 `tm_plain.get(src)` —— 按英文原文取
全库多数派，把 name / tokenName / adjective / folders 这些**书写约定互不相同**的字段
当成同一个池子。

同一形状在 `tm/build_glossary.py:harvest()` 上出现，而且**没有退化分支这一说**：
它的投票池 `pairs[en][cn] += 1`（:113）从一开始就只有「英文原文」一个维度，
`walk_pairs()`（:59-69）平铺 entries 与 folders 的**全部**并行叶子，
`harvested[en] = ranked[0][0]`（:149）取全库多数派。产物 glossary_ec.json 是
项目对外的**唯一术语权威**（PROJECT 交接文档与各轮 brief 都写「术语表 glossary_ec.json」），
并被 tm/fill_twin_names.py:127-137 直接读取。

本探针做三件事（只读）
----------------------
A. 复刻 harvest() 的取数，但给每一票带上**字段角色**（路径最后一段非数字键名；
   folders 层记为 'folders'）。统计每个术语的票来自哪些角色。
B. 判定每一票的**书写约定**：中文里含该英文原文 => 双语并列（name 约定）；
   否则 => 裸中文（tokenName / adjective / levels / 表结果行 约定）。
   找出「同一英文在不同角色下约定相反」的术语 —— 这些术语在词表里只能有一个答案，
   于是对另一半角色一定是错的。
C. 核对 build_glossary.py:169 的 `format_only` 判据
   `(cn.startswith(b) and en in cn) or (b in cn) or (cn in b)`：
   第二、三个分支是**子串包含**，会把「工艺 vs 工艺品」这种真正的用词分歧
   判成「只是双语格式差异」，于是静默用 shipped 顶掉已裁决的 base 值，不进 disputes。

假阳性模式（必须知道）
----------------------
1. 「双语并列」的判定用「中文里含英文原文」，对英文本身是纯符号/数字的条目会误判 ——
   已用 `re.search('[A-Za-z]', en)` 排掉。
2. 同一角色内部本来就允许异写（G1 豁免的 14 组），所以本探针**只报跨角色**的分歧，
   同角色内的多写法单独计数、不进结论。
3. 角色取「路径最后一段非数字键名」，与 qa/propagate_fix.py:39-48 的 role_of 同语义；
   `pages.X.name` 与 `items.Y.name` 都算 name（这正是项目的约定：name 一律双语）。
4. 词表的消费方是人和 fill_twin_names，本探针不能证明「某条已经被用错」，
   只能证明「词表对这些术语给出的答案对一部分角色必然是错的」。
   fill_twin_names.py:132 `v.split(' ')[0]` 会剥掉英文尾巴，属于**下游各自打补丁**，
   不是词表本身有角色维度。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
BASE_DIR = os.path.join(P, '5-其他内容', 'english-baseline')
GLOSS = os.path.join(P, '5-其他内容', 'glossary', 'glossary_ec.json')
PROV = os.path.join(P, '5-其他内容', 'glossary', 'glossary_ec.provenance.json')

CJK = re.compile(r'[\u4e00-\u9fff]')
HTML = re.compile(r'<[^>]+>')
MAX_TERM_LEN = 60

PAIRS = [
    (os.path.join(BASE_DIR, 'crucible-0.10.1'),
     os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn'), 'crucible'),
    (os.path.join(BASE_DIR, 'ember-0.6.0'),
     os.path.join(P, '1-Ember汉化插件', 'compendium', 'cn'), 'ember'),
]


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def is_term(s):
    """与 build_glossary.py:44-56 逐字相同。"""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or len(s) > MAX_TERM_LEN:
        return False
    if HTML.search(s) or '\n' in s:
        return False
    if s[-1] in '.!?:;,':
        return False
    return len(s.split()) <= 8


def walk_pairs(en, cn, out, path=()):
    """build_glossary.walk_pairs 的带路径版本。"""
    if isinstance(en, dict) and isinstance(cn, dict):
        for k, v in en.items():
            if k in cn:
                walk_pairs(v, cn[k], out, path + (k,))
    elif isinstance(en, list) and isinstance(cn, list):
        for i, (a, b) in enumerate(zip(en, cn)):
            walk_pairs(a, b, out, path + (str(i),))
    elif isinstance(en, str) and isinstance(cn, str):
        out.append((en, cn, path))


def role_of(path, en_s=None):
    """与 qa/propagate_fix.py:39-48 同语义：最后一段非数字键名。

    额外一条：nameCollection / textCollection 形态（folders / categories / levels /
    tokens）的**键就是英文原文本身**（extract_en.mjs:194-218 `map[it.name] ??= it.name`），
    这时最后一段是条目名不是字段名，角色要取上一段，否则会冒出
    `Introduction` / `The Dives` 这种假角色。"""
    segs = [s for s in path if not s.isdigit()]
    if not segs:
        return path[-1] if path else '?'
    if en_s is not None and segs[-1] == en_s and len(segs) >= 2:
        return segs[-2]
    return segs[-1]


def main():
    # votes[en][cn] = count ; roles[en][role][cn] = count
    votes = defaultdict(Counter)
    roles = defaultdict(lambda: defaultdict(Counter))
    where = defaultdict(list)

    for en_dir, cn_dir, label in PAIRS:
        if not (os.path.isdir(en_dir) and os.path.isdir(cn_dir)):
            print(f'!! missing dir: {en_dir} / {cn_dir}')
            continue
        for fn in sorted(f for f in os.listdir(en_dir)
                         if f.endswith('.json') and not f.startswith('_')):
            cnp = os.path.join(cn_dir, fn)
            if not os.path.exists(cnp):
                continue
            try:
                en_doc = load(os.path.join(en_dir, fn))
                cn_doc = load(cnp)
            except Exception as e:
                print(f'  ! unreadable {fn}: {e}')
                continue
            got = []
            walk_pairs(en_doc.get('entries', {}), cn_doc.get('entries', {}), got, ('entries',))
            walk_pairs(en_doc.get('folders', {}), cn_doc.get('folders', {}), got, ('folders',))
            for en_s, cn_s, path in got:
                if not is_term(en_s) or not CJK.search(cn_s):
                    continue
                r = role_of(path, en_s)
                if len(path) == 2 and path[0] == 'folders':
                    r = 'folders'
                votes[en_s][cn_s] += 1
                roles[en_s][r][cn_s] += 1
                if len(where[en_s]) < 60:
                    where[en_s].append((fn, '.'.join(path), cn_s))

    gloss = load(GLOSS)
    print(f'harvest 复刻：{len(votes)} 个术语，{sum(sum(c.values()) for c in votes.values())} 票')
    print(f'glossary_ec.json 现有 {len(gloss)} 条\n')

    def bilingual(en_s, cn_s):
        return en_s.lower() in cn_s.lower()

    cross, same_role_only = [], 0
    for en_s, per_role in roles.items():
        if not re.search(r'[A-Za-z]', en_s):
            continue
        conv = {}   # role -> set of conventions
        for r, cns in per_role.items():
            conv[r] = {bilingual(en_s, c) for c in cns}
        # 只保留「角色 A 全部双语、角色 B 全部裸中文」这种**跨角色**的相反约定
        bi_roles = [r for r, s in conv.items() if s == {True}]
        bare_roles = [r for r, s in conv.items() if s == {False}]
        if bi_roles and bare_roles:
            winner = gloss.get(en_s)
            cross.append({
                'en': en_s,
                'glossary': winner,
                'glossary_is_bilingual': bilingual(en_s, winner) if winner else None,
                'bilingual_roles': {r: dict(per_role[r]) for r in bi_roles},
                'bare_roles': {r: dict(per_role[r]) for r in bare_roles},
                'samples': where[en_s][:8],
            })
        elif len(votes[en_s]) > 1:
            same_role_only += 1

    print(f'A/B —— 跨角色书写约定相反的术语：{len(cross)}')
    print(f'      （同角色内多写法、不进结论的：{same_role_only}）')
    bi_win = [c for c in cross if c['glossary_is_bilingual'] is True]
    bare_win = [c for c in cross if c['glossary_is_bilingual'] is False]
    missing = [c for c in cross if c['glossary'] is None]
    print(f'      词表给出**双语并列**（对 adjective/tokenName 等裸中文角色必错）：{len(bi_win)}')
    print(f'      词表给出**裸中文**（对 name 角色必错）：{len(bare_win)}')
    print(f'      词表里没有这一条：{len(missing)}')

    # D. 剥掉英文尾巴之后**仍然不同**的：这类是语义分歧，不是格式分歧，
    #    下游那句 `v.split(' ')[0]` 救不了。
    def strip_en(en_s, cn_s):
        low = cn_s.lower()
        i = low.rfind(en_s.lower())
        return (cn_s[:i] + cn_s[i + len(en_s):]).strip() if i >= 0 else cn_s.strip()

    semantic = []
    for c in cross:
        bi_forms = {strip_en(c['en'], k) for d in c['bilingual_roles'].values() for k in d}
        bare_forms = {k for d in c['bare_roles'].values() for k in d}
        if bi_forms & bare_forms:
            continue                      # 剥完就一致 —— 纯格式差异
        semantic.append({**c, 'bilingual_stripped': sorted(bi_forms),
                         'bare_forms': sorted(bare_forms)})
    print(f'\nD —— 其中剥掉英文尾巴后**中文本身仍不同**（语义分歧，格式修补救不了）：{len(semantic)}')
    for c in semantic[:25]:
        print(f'        {c["en"]!r}: 词表 {c["glossary"]!r} | name侧剥后 {c["bilingual_stripped"]} '
              f'| 裸中文角色 {c["bare_forms"]} {list(c["bare_roles"])}')

    role_pairs = Counter()
    for c in cross:
        for a in c['bilingual_roles']:
            for b in c['bare_roles']:
                role_pairs[f'{a}(双语) x {b}(裸)'] += 1
    print('\n      角色组合 top：')
    for k, v in role_pairs.most_common(12):
        print(f'        {v:>4}  {k}')

    # E. 稳定性：多数派的**格式**由票数决定，票差 <=1 的条目，上游多一条同名文档就翻面
    margins = Counter()
    flippable = []
    for c in cross:
        nb = sum(sum(d.values()) for d in c['bilingual_roles'].values())
        nr = sum(sum(d.values()) for d in c['bare_roles'].values())
        m = abs(nb - nr)
        margins[min(m, 5)] += 1
        if m <= 1:
            flippable.append((c['en'], c['glossary'], nb, nr))
    print(f'\nE —— 双语票 vs 裸票的票差分布（0..5+）：{dict(sorted(margins.items()))}')
    print(f'      票差 <=1（上游再多一条同名文档就会让词表里这一条的格式翻面）：{len(flippable)}')
    for e, g, nb, nr in flippable[:10]:
        print(f'        {e!r} -> {g!r}   双语票 {nb} : 裸票 {nr}')

    print('\n      样例（词表给双语、库里有裸中文角色）：')
    for c in bi_win[:12]:
        print(f'        {c["en"]!r} -> 词表 {c["glossary"]!r}')
        for r, d in list(c['bare_roles'].items())[:2]:
            print(f'            裸中文角色 {r}: {d}')

    # ---- C. format_only 子串判据 ----
    prov = load(PROV)
    bvs = prov.get('baseVsShipped', {})
    subs_only = []
    for en_s, d in bvs.items():
        if d.get('kind') != 'format':
            continue
        b, s = d['base'], d['shipped']
        suffix_form = s.startswith(b) and en_s in s      # 真·双语后缀形态
        if not suffix_form:
            subs_only.append((en_s, b, s))
    print(f'\nC —— provenance 里被判成「只是双语格式差异」而静默采用 shipped 的：{len(bvs)} 条中 '
          f'kind=format {sum(1 for d in bvs.values() if d.get("kind") == "format")} 条')
    print(f'      其中**不是**「base + 空格 + 英文」这种后缀形态、纯靠子串包含过关的：{len(subs_only)}')
    for en_s, b, s in subs_only[:20]:
        print(f'        {en_s!r}: base {b!r} -> shipped {s!r}')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'roleblind_glossary_pool.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'crossRoleTerms': cross, 'semanticCrossRole': semantic,
                   'formatOnlyBySubstring': subs_only}, f, ensure_ascii=False, indent=1)
    print(f'\n-> {out}')


main()
