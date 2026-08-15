#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定义列表 / 表格 / 项目符号的**行列错位**：中文条目相对英文整体或局部挪了位。

来历
----
第七轮顺带撞见过一条阻断：`Ordain Gazetteer / Scholar's Nook` 的 landmark `<dt>` 名
**整体错位一格**（玩家按地名志找任何一家店都会走错门）。当时只写了针对那一页的
一次性修复脚本 `2026-08-12-audit3/fix_gazetteer_dt.py`，**从未做成全库判据**。

为什么现有闸门全盲
------------------
* `scan_markup_drift` 的 BLOCK 只数**块级标签个数**。错位不改变个数，永远相等；
  而且它的 BLOCK_TAGS 里**根本没有 `dt` / `dd`** —— 定义列表连"少一项"都数不出来。
* 覆盖率按叶算，错位的叶子里中文满满当当，算 100% 已译。
* 数字覆盖看的是数字**多重集**，整列换位后多重集不变。
* 标记五项 / class 漂移 / 外来文字 / tokenName 全都只看单叶自身或单个字段。

五类判据
--------
DL_COUNT   `<dt>` 或 `<dd>` 条数中英不等（BLOCK 的盲区，dt/dd 不在它的标签表里）。
SHAPE      **每个容器各自**的子元素条数不同。BLOCK 只比整叶总数：同一叶里三个 `<ul>`
           从 4/3/5 变成 4/4/4，或表格行从 3/3/3 变成 4/2/3，总数不变，BLOCK 全绿。
DT_BARE    `<dt>` 条目整条没译（同列表里别的条目已译，说明不是有意保留英文）。
SHIFT      **锚点错位**：位置 i 的中文条目里的锚点（数字 / 拉丁词 / `@UUID[...]` 目标 /
           `[[/...]]`）在位置 i 的英文里一个都找不到，却在位置 j≠i 的英文里命中。
           两道去噪，每一道都是实测逼出来的：
             * 锚点先做**同列表内去重**，只留在本列表英文侧只出现一次的锚点，
               `he/him`、`Level`、`DC` 这种每条都有的词自动出局，不用维护停用词表；
             * 判定按**列表级投票**：同一列表里 >=2 条各自独立指向**同一位移量**才报。
               单锚点证据太弱 —— 英文写 "three"、中文写 "3"，那个 3 恰好出现在别的
               条目里就会误报（min-anchor=1 无投票时全库 8 条独立误报全是这个模式）。
LEX_SHIFT  **词表错位**：给纯中文条目用的。用全库其它叶子里成对出现的
           (英文条目 → 中文条目) 建一张词表（**排除本叶自身**，否则永远自证），
           若 cn[i] 是 en[j] (j≠i) 的已知译法、而不是 en[i] 的，就报。
           这是唯一能抓「一串纯中文店名整体挪一格」的信号 —— 上面那条阻断就是这一类。
           **已知局限**：该词条在库里必须有**独立的第二次出现**才判得出来。
           拿 git 里修复前的版本回测：孪生包未同步时能精确报出 Scholar's Nook 的
           3 条错位 dt；两个孪生包都错时（历史真实情形）报不出来。

`--tags` 可以把同一套错位判据推广到行内层（`--tags p,h2,h3,h4,strong`）。
`<strong>` 那一路的主要误报模式是**中文语序倒装相邻粗体**
（EN `resistance of +5 to Slashing` → CN `对挥砍伤害具有 +5 抗性`，两个粗体互换位置，
是对的），逐条看时先排除这一类。

用法
----
  python scan_list_alignment.py --repo <repo> [--repo <另一个>] [--out <json>]
                                [--kind DL_COUNT,SHIFT,LEX_SHIFT] [--show 40]
只读，不写 compendium。
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TAGS = ('dt', 'dd', 'li', 'th', 'td')
CJK = re.compile(r'[一-鿿㐀-䶿]')
TAG_RE = re.compile(r'<[^>]+>')
MARKUP = re.compile(r'@[A-Za-z]+\[([^\]]*)\]')
INLINE_CMD = re.compile(r'\[\[([^\]]*)\]\]')
ANY_MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]')
NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')
LATIN = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'’­-]{2,}")
WS = re.compile(r'\s+')

# 每条列表项都可能带的、毫无区分度的拉丁串（同列表去重已能干掉绝大多数，
# 这里只兜底跨列表也高频的几个）
NOISE_WORDS = {'the', 'and', 'she', 'her', 'hers', 'him', 'his', 'they', 'them',
               'their', 'theirs', 'you', 'your', 'strong', 'span', 'class'}


def leaves(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f'{prefix}.{k}' if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f'{prefix}.{i}')
    elif isinstance(obj, str) and obj:
        yield prefix, obj


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def items_of(s, tag):
    """取某个标签的**最内层**序列（`<li>` 里可能嵌 `<ul><li>`，内层单独成项，
    两侧同一套规则，配对仍成立）。"""
    rx = re.compile(r'<%s\b[^>]*>((?:(?!</?%s[\s>]).)*)</%s>' % (tag, tag, tag),
                    re.S | re.I)
    return [m.group(1) for m in rx.finditer(s)]


def plain(s):
    """可见文字：去标记（含 `{标签}` 以外的部分）、去标签、压空白。"""
    s = ANY_MARKUP.sub(' ', s)
    s = TAG_RE.sub(' ', s)
    s = s.replace('&nbsp;', ' ').replace('&amp;', '&')
    return WS.sub(' ', s).strip()


def anchors(s):
    """位置无关的、中英两侧应当照抄的锚点。"""
    a = set()
    for m in MARKUP.finditer(s):
        a.add('M:' + WS.sub('', m.group(1)))
    for m in INLINE_CMD.finditer(s):
        a.add('R:' + WS.sub('', m.group(1)))
    t = plain(s)
    for n in NUM.findall(t):
        a.add('N:' + n.replace(',', ''))
    for w in LATIN.findall(t):
        # 所有格与复数要归一：英文写 `Jasper's Argument`、中文写「Jasper 的主张」，
        # 不归一就会把 `jasper's` 与 `jasper` 当成两个词，判成错位（实测的一类误报）。
        wl = re.sub(r'[’\']s$', '', w.lower())
        wl = re.sub(r'(?<=[a-z]{3})s$', '', wl)
        if len(wl) >= 3 and wl not in NOISE_WORDS:
            a.add('W:' + wl)
    return a


VOID = {'br', 'hr', 'img', 'input', 'meta', 'link', 'col', 'source', 'wbr'}
TOKEN = re.compile(r'<(/?)\s*([a-zA-Z][a-zA-Z0-9]*)[^>]*?(/?)\s*>')
CONTAINERS = ('ul', 'ol', 'dl', 'table', 'tr')
CHILDREN = {'ul': ('li',), 'ol': ('li',), 'dl': ('dt', 'dd'),
            'table': ('tr',), 'tr': ('td', 'th')}


def shape(s):
    """每个容器**各自**的直接子元素条数，按开标签的文档顺序排。

    这是 `scan_markup_drift` 的 BLOCK 看不见的一层：BLOCK 只比整叶的
    `li`/`td`/`tr` **总数**。同一叶里三个 `<ul>` 从 4/3/5 变成 4/4/4 时总数不变，
    BLOCK 全绿，可页面上第二个清单已经多吞了第三个清单的一项。
    `<tr>` 的列数分布同理：3/3/3 变 4/2/3 总数还是 9，表格却已经错行。
    """
    stack, done = [], []
    for m in TOKEN.finditer(s):
        closing, name, selfclose = m.group(1), m.group(2).lower(), m.group(3)
        if name in VOID or selfclose:
            continue
        if closing:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == name:
                    del stack[k:]
                    break
        else:
            if stack:
                stack[-1][1][name] += 1
            stack.append([name, collections.Counter()])
            if name in CONTAINERS:
                done.append((name, stack[-1][1]))
    return [(n, tuple(c[k] for k in CHILDREN[n])) for n, c in done]


def norm_item(s):
    """列表条目的归一化文本，用来当词表的键 / 值。"""
    t = plain(s)
    t = re.sub(r'[\s　（）()，,。.、；;：:！!？?“”"\'’·\-—…]+', '', t)
    return t


# ---------------------------------------------------------------- 主流程
def collect(repos):
    """一次性把两个仓库的所有叶子对读进来。返回 (含列表的叶, 全部叶)。"""
    out, every = [], []
    for repo in repos:
        en_dir = os.path.join(repo, 'compendium', 'en')
        for pack in sorted(os.listdir(en_dir)):
            if not pack.endswith('.json') or pack.startswith('_'):
                continue
            cn_p = os.path.join(repo, 'compendium', 'cn', pack)
            if not os.path.exists(cn_p):
                continue
            en = dict(leaves(load(os.path.join(en_dir, pack)).get('entries', {})))
            cn = dict(leaves(load(cn_p).get('entries', {})))
            for path, s in en.items():
                t = cn.get(path)
                if not t or not isinstance(t, str):
                    continue
                every.append((repo, pack, path, s, t))
                if any('<%s' % g in s for g in TAGS):
                    out.append((repo, pack, path, s, t))
    return out, every


UUID_LABEL = re.compile(r'(@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])\s*\{([^}]*)\}')
STRONG = re.compile(r'<strong[^>]*>((?:(?!</?strong[\s>]).)*)</strong>', re.S | re.I)


def build_lexicon(pairs, all_pairs):
    """en条目 -> {cn条目: set(来源叶 id)}，来源用于排除自证。

    四路料源，越多独立料源，纯中文列表越有可能被判出错位：
      1. 各列表/表格条目本身（同 tag、条数相等才配对）
      2. 全库**短叶**（`name` / `tokenName` / 页名 / 针脚名 …）—— 专名的主要来源
      3. `@UUID[目标]{标签}` 的标签，按**目标**配对（不靠位置，最可靠）
      4. `<strong>` 行内片段（条数相等才配对）—— 正文里加粗的专名
    """
    lex = collections.defaultdict(lambda: collections.defaultdict(set))

    def add(e, c, idx):
        ne, nc = norm_item(e), norm_item(c)
        if ne and nc and len(ne) >= 3 and LATIN.search(ne):
            lex[ne][nc].add(idx)

    for idx, (_r, _p, _path, s, t) in enumerate(pairs):
        for tag in TAGS:
            es, cs = items_of(s, tag), items_of(t, tag)
            if len(es) == len(cs):
                for e, c in zip(es, cs):
                    add(e, c, idx)

    # 短叶 + UUID 标签 + <strong>：用 all_pairs（不限于含列表的叶），
    # 叶 id 用负数，保证与 pairs 的 idx 不冲突（自证排除只针对 pairs 的 idx）。
    for k, (_r, _p, _path, s, t) in enumerate(all_pairs):
        oid = -1 - k
        if len(s) <= 80 and LATIN.search(s) and CJK.search(t):
            add(s, t, oid)
        el = {m.group(1): m.group(2) for m in UUID_LABEL.finditer(s)}
        cl = {m.group(1): m.group(2) for m in UUID_LABEL.finditer(t)}
        for tgt, lab in el.items():
            if tgt in cl:
                add(lab, cl[tgt], oid)
        eb, cb = STRONG.findall(s), STRONG.findall(t)
        if len(eb) == len(cb):
            for e, c in zip(eb, cb):
                add(e, c, oid)
    return lex


def uniq_anchor_sets(seq):
    """只保留在本列表英文侧**只出现一次**的锚点 —— 自动过滤每条都有的套话。"""
    df = collections.Counter()
    sets = [anchors(x) for x in seq]
    for a in sets:
        df.update(a)
    return [{x for x in a if df[x] == 1} for a in sets], sets


def main():
    global TAGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out')
    ap.add_argument('--kind', default='DL_COUNT,SHAPE,DT_BARE,SHIFT,LEX_SHIFT')
    ap.add_argument('--tags', default=','.join(TAGS),
                    help='要配对的标签序列，默认 dt,dd,li,th,td；'
                         '可加 p,h2,h3,h4,strong 把同样的错位判据推广到段落层')
    ap.add_argument('--min-anchor', type=int, default=1,
                    help='SHIFT：单条命中位置 j 所需的锚点数下限')
    ap.add_argument('--min-votes', type=int, default=2,
                    help='SHIFT：同一列表内指向同一位移量的条目数下限')
    ap.add_argument('--strong-anchor', type=int, default=3,
                    help='SHIFT：单条锚点数达到这个数就不需要凑票')
    ap.add_argument('--show', type=int, default=40)
    a = ap.parse_args()
    kinds = set(a.kind.split(','))
    TAGS = tuple(x.strip() for x in a.tags.split(',') if x.strip())

    pairs, every = collect(a.repo)
    stats = collections.Counter()
    stats['带列表/表格标签的叶'] = len(pairs)
    stats['全部叶对'] = len(every)
    findings = []

    lex = build_lexicon(pairs, every) if 'LEX_SHIFT' in kinds else {}
    stats['词表条目(英文键)'] = len(lex)

    for idx, (repo, pack, path, s, t) in enumerate(pairs):
        bpath = path[len('entries.'):] if path.startswith('entries.') else path

        # ---- SHAPE：每个容器各自的子元素条数（BLOCK 只比总数）------------
        if 'SHAPE' in kinds:
            se, sc = shape(s), shape(t)
            stats['容器数'] += len(se)
            if se != sc:
                ten, tcn = [x[0] for x in se], [x[0] for x in sc]
                findings.append({
                    'kind': 'SHAPE', 'repo': repo, 'pack': pack,
                    'batch_path': bpath,
                    'why': '容器序列不同' if ten != tcn else '容器内条数分布不同',
                    'en_shape': [f'{n}{c}' for n, c in se],
                    'cn_shape': [f'{n}{c}' for n, c in sc],
                })
                stats['**SHAPE**'] += 1

        # ---- DT_BARE：<dt> 条目整条没译（同列表里别的条目已译）------------
        if 'DT_BARE' in kinds:
            ed, cd = items_of(s, 'dt'), items_of(t, 'dt')
            if len(ed) == len(cd) and len(ed) >= 2 and any(CJK.search(x) for x in cd):
                bare = [(i, plain(ed[i]), plain(cd[i])) for i in range(len(ed))
                        if plain(cd[i]) and not CJK.search(cd[i]) and LATIN.search(cd[i])]
                if bare:
                    findings.append({
                        'kind': 'DT_BARE', 'repo': repo, 'pack': pack,
                        'batch_path': bpath, 'n': len(ed),
                        'detail': [{'i': i, 'en': e[:90], 'cn': c[:90]}
                                   for i, e, c in bare[:10]],
                    })
                    stats['**DT_BARE**'] += len(bare)

        for tag in TAGS:
            es, cs = items_of(s, tag), items_of(t, tag)
            if not es:
                continue
            stats['列表数(%s)' % tag] += 1
            stats['条目数(%s)' % tag] += len(es)
            if len(es) != len(cs):
                if tag in ('dt', 'dd') and 'DL_COUNT' in kinds:
                    findings.append({
                        'kind': 'DL_COUNT', 'repo': repo, 'pack': pack,
                        'batch_path': bpath, 'tag': tag,
                        'en_n': len(es), 'cn_n': len(cs),
                        'en_items': [plain(x)[:90] for x in es],
                        'cn_items': [plain(x)[:90] for x in cs],
                    })
                    stats['**DL_COUNT**'] += 1
                else:
                    stats['条数不等(%s,交给BLOCK)' % tag] += 1
                continue
            if len(es) < 2:
                continue

            # ---- SHIFT：锚点错位 -------------------------------------
            if 'SHIFT' in kinds:
                eu, _ = uniq_anchor_sets(es)
                cu, call = uniq_anchor_sets(cs)
                moved = []
                for i in range(len(es)):
                    ca = call[i] & set().union(*eu) if eu else set()
                    if not ca:
                        stats['锚点盲(%s)' % tag] += 1
                        continue
                    stats['锚点可判(%s)' % tag] += 1
                    hits = [(len(ca & eu[j]), j) for j in range(len(es))]
                    self_n = hits[i][0]
                    best_n, best_j = max(hits, key=lambda x: (x[0], -abs(x[1] - i)))
                    # 唯一性：最好的那个 j 不能和第二好的并列
                    tie = sum(1 for n, _ in hits if n == best_n)
                    if (best_j != i and self_n == 0 and best_n >= a.min_anchor
                            and tie == 1):
                        moved.append((i, best_j, best_n, sorted(ca & eu[best_j])[:6]))
                # **列表级投票**：单条只有一个锚点时证据很弱（英文写 "three"、
                # 中文写 "3"，那个 3 恰好也出现在别的条目里，就会误报一条）。
                # 但若同一列表里 >=2 条**各自独立**指向**同一个位移量**，那就是真错位。
                # 实测：min-anchor=1 全库 8 条独立误报里，7 条只有 1 票，
                # 剩下 1 条（Patch 0.4.2）3 票但位移量 -16/-17/-19 各不相同，一并出局。
                deltas = collections.Counter(j - i for i, j, _, _ in moved)
                if deltas:
                    d, votes = deltas.most_common(1)[0]
                    grp = [m for m in moved if m[1] - m[0] == d]
                    strong = max((m[2] for m in grp), default=0)
                    if votes >= a.min_votes or strong >= a.strong_anchor:
                        findings.append({
                            'kind': 'SHIFT', 'repo': repo, 'pack': pack,
                            'batch_path': bpath, 'tag': tag, 'n': len(es),
                            'moved': len(grp), 'votes': votes,
                            'delta': d,
                            'detail': [{'i': i, 'best_j': j, 'anchors': k, 'shared': sh,
                                        'cn': plain(cs[i])[:110],
                                        'en_i': plain(es[i])[:110],
                                        'en_j': plain(es[j])[:110]}
                                       for i, j, k, sh in grp[:8]],
                        })
                        stats['**SHIFT**'] += 1

            # ---- LEX_SHIFT：词表错位（纯中文条目也能查）---------------
            if 'LEX_SHIFT' in kinds:
                ne = [norm_item(x) for x in es]
                nc = [norm_item(x) for x in cs]
                moved = []
                for i in range(len(es)):
                    if not ne[i] or not nc[i] or len(ne[i]) < 3:
                        continue
                    # 本叶之外，en[i] 的已知译法
                    def others(k):
                        d = lex.get(ne[k])
                        if not d:
                            return set()
                        return {c for c, src in d.items() if src - {idx}}
                    if nc[i] in others(i):
                        continue
                    if not others(i):
                        continue          # en[i] 在别处没出现过，无从判断
                    for j in range(len(es)):
                        if j == i or not ne[j]:
                            continue
                        if nc[i] in others(j):
                            moved.append((i, j, plain(cs[i])[:80],
                                          plain(es[i])[:80], plain(es[j])[:80]))
                            break
                if moved:
                    findings.append({
                        'kind': 'LEX_SHIFT', 'repo': repo, 'pack': pack,
                        'batch_path': bpath, 'tag': tag, 'n': len(es),
                        'moved': len(moved),
                        'delta': collections.Counter(j - i for i, j, *_ in moved).most_common(1)[0],
                        'detail': [{'i': i, 'best_j': j, 'cn': c, 'en_i': ei, 'en_j': ej}
                                   for i, j, c, ei, ej in moved[:8]],
                    })
                    stats['**LEX_SHIFT**'] += 1

    print('规模：')
    for k, v in sorted(stats.items()):
        print(f'  {k:34s} {v}')
    print(f'\n共 {len(findings)} 条')
    for f in findings[:a.show]:
        print(f'\n[{f["kind"]}] {f["pack"][:28]}  <{f.get("tag","-")}>  {f["batch_path"][:100]}')
        if f['kind'] == 'DL_COUNT':
            print(f'   en={f["en_n"]} cn={f["cn_n"]}')
        elif f['kind'] == 'SHAPE':
            print(f'   {f["why"]}\n   en {f["en_shape"]}\n   cn {f["cn_shape"]}')
        elif f['kind'] == 'DT_BARE':
            for d in f['detail'][:6]:
                print(f'   #{d["i"]}  EN {d["en"][:50]:52s} CN {d["cn"][:50]}')
        else:
            print(f'   n={f["n"]} moved={f["moved"]} delta={f["delta"]} votes={f.get("votes")}')
            for d in f['detail'][:4]:
                print(f'   #{d["i"]}->#{d["best_j"]}  CN: {d["cn"][:60]}')
                print(f'        en[i]: {d["en_i"][:60]}')
                print(f'        en[j]: {d["en_j"][:60]}')

    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'stats': dict(stats), 'findings': findings}, f,
                      ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
