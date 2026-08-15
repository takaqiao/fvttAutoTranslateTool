#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同一个英文专名在**正文**里有两套以上中文译名。

`scan_name_splits.py` 只比 `name` 字段（5449 个英文名，当前剩 7 个分裂）。
玩家真正读到的绝大部分文字在正文（description / journal page text），
那里的专名分裂**至今没有判据**。`scan_bare_english_names.py` 查的是「正文里留着裸英文」，
方向不同（它查没译，本闸查译了但两处译法不一致）。

判据
----
1. 从 `*.name` 叶子成对提取「英文专名 -> 中文译名」词典（沿用 scan_bare_english_names 的做法）。
2. 对每个英文专名 N，用**英文闸**取出正文（非 name）叶子里英文侧确实出现 N 的那些叶，
   并做**长名遮蔽**：`Ring of Heroism` 命中的叶子不算 `Ring` 也不算 `Heroism` 的证据。
3. 按中文侧用了哪个译名分桶：`hit`（含词典译名）/ `miss`（一个都不含）。
4. 在 miss 桶里发现「对手译名」，两条引擎：
   - **A 近形**：与词典译名字形相近的同长度串（音译分裂：阿克图里亚/阿克图里安、基瓦赫/基瓦尔）。
   - **B 自由词**：miss 叶里反复出现、且「只跟着 N 走」的 CJK n-gram（另起炉灶的意译）。
   两条都要过**边界熵**（左右邻字各 >= 2 种）—— 这是把「及可被其擒抱的生」这类
   整句碎片挡在外面的关键；句子片段在语料里左右邻字永远相同。

为什么必须做「发现」而不是只匹配已知译名
----------------------------------------
绝大多数英文名在 name 字段只有**一个**中文译名（name 侧已被 scan_name_splits 清干净），
只匹配已知译名的话 |V|=1 永远报不出分裂。正文里的第二种译法恰恰是词典里没有的那个。

⚠ 本闸只报分裂，**不给方向建议**
--------------------------------
本项目实测多数派常常是错的那边（`Signborn Lineage` 星兆血统 3:1 胜出但已裁「印记裔」；
`Arcturian` 阿克图里亚 1090 : 阿克图里安 248，最终取少数派）。方向按依据阶梯逐条人判：
同名条目 name > 同卷已译页 > 全库多数 > glossary_ec.json，且 name 本身也可能是错的那边。

⚠ 合法分裂要人排除
------------------
同一英文词在不同语境下确实该有不同中文（`Shield` 法术护盾术 / 装备盾牌；
`Freezing` 状态「冰冻」/ 形容词「冰封酷寒」；`Trading House Cevher` 商会 /
`House Cevher` 家族）。判据看不出词义。

回测（2026-08-13 第八轮）
-------------------------
* 特异度：Ember 全库报 164 组，按 sim 排序取头部 50 组人工逐条核，**22 组是真分裂**（44%）。
  假阳性两大类：(1) 整句碎片被当成词（「取决于其制作」「一颗宝石」）；
  (2) 语境合法的不同译法（`Lantern`提灯 vs `lantern oil`灯油、`Mirror`镜子 vs `prism`棱镜）。
* 灵敏度：往 Crucible 副本注入三条已知分裂（扎拉贾->扎拉迦 / 反制法术->破法术 /
  艾利奥文->艾莉奥雯，各 3 片正文叶），**三条全部报出且都排在该专名对手词第一/第二位**；
  未注入的对照跑只报 5 组，注入后 8 组，无附带误报。

用法：
  python scan_body_name_splits.py --repo <repo> [--repo <另一个>] --out <json>
  python scan_body_name_splits.py --repo <repo> --engine near --out x.json   # 只跑近形
灵敏度回测见同目录 inject_body_name_split.py（只改临时副本，绝不碰 compendium/）。
运行时间：Ember+Crucible 全库约 6-8 分钟，瓶颈在 Corpus.tally 的全库 n-gram 计数。
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CJK_RE = re.compile(r'[一-鿿]')
CJK_RUN = re.compile(r'[一-鿿·]{2,}')
TAG = re.compile(r'<[^>]+>')
BILINGUAL_TAIL = re.compile(r'\s+[^一-鿿　-〿＀-￯]+$')
WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

STOP = {
    'Green', 'Ship', 'Empty', 'Ground', 'Surface', 'Entry', 'Basement', 'Library',
    'Gardens', 'Markets', 'Temple', 'Mine', 'Ocean', 'Roofs', 'Fields', 'Overlook',
    'Traps', 'Caste', 'Gala', 'Lookout', 'Promenade', 'Vineyard', 'Pathways',
    'Name', 'None', 'Other', 'Text', 'Item', 'Type', 'Notes', 'Level', 'Damage',
    'Effect', 'Effects', 'Action', 'Actions', 'Attack', 'Move', 'Rest', 'Round',
    'Human', 'Cosmos', 'Small', 'Large', 'Medium', 'Common', 'Rare', 'Simple',
}
CN_STOP = {'描述', '效果', '动作', '攻击', '伤害', '等级', '物品', '技能', '法术',
           '人类', '生物', '角色', '玩家'}
SMALL = {'of', 'the', 'and', 'in', 'on', 'a', 'to', 'for', 'at', 'from', 'with',
         'or', 'de', 'la', 'le', 'du', 'des', 'von', "o'", 'ur'}


def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f'{path}.{i}')
    elif isinstance(obj, str) and obj:
        yield path, obj


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def head(s):
    return BILINGUAL_TAIL.sub('', s).strip() or s.strip()


def plain_en(s):
    return TAG.sub(' ', re.sub(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]', ' ', s))


def plain_cn(s):
    s = re.sub(r'@[A-Za-z]+\[[^\]]*\]', ' ', s)
    s = re.sub(r'\[\[[^\]]*\]\]', ' ', s)
    return TAG.sub(' ', s)


def looks_proper(en: str) -> bool:
    if len(en) < 4 or en in STOP:
        return False
    ws = en.split(' ')
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*", w) for w in ws):
        return False
    if not ws[0][0].isupper():
        return False
    return all(w[0].isupper() or w.lower() in SMALL for w in ws[1:])


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


class Corpus:
    """全库中文语料，用来算对手词的背景频次与左右邻字多样性。

    背景频次**必须一次性批量算**：逐个 `str.count` 在 600 万字语料上每次 ~3ms，
    候选上万条就是几十分钟。这里改成「先把全部候选攒起来，再扫一遍语料」。
    """

    def __init__(self, parts):
        self.text = '\x00'.join(parts)
        self._cnt = {}
        self._br = {}

    def tally(self, cands):
        """一次扫描把 cands 里所有串的全库出现次数算出来。"""
        todo = {g for g in cands if g not in self._cnt}
        if not todo:
            return
        by_len = collections.defaultdict(set)
        for g in todo:
            by_len[len(g)].add(g)
            self._cnt[g] = 0
        t = self.text
        n = len(t)
        for k, pool in by_len.items():
            cnt = self._cnt
            for i in range(n - k + 1):
                g = t[i:i + k]
                if g in pool:
                    cnt[g] += 1

    def count(self, g):
        v = self._cnt.get(g)
        if v is None:
            v = self._cnt[g] = self.text.count(g)
        return v

    def branching(self, g):
        """边界完整性：只在「**每一次**出现都紧贴同一个汉字」时判否。

        那种情况说明 g 只是 `c+g`（或 `g+c`）的一截 —— 整句碎片就长这样。
        ⚠ 早期版本要求「左右邻字各 >= 2 种」，把标点/空格/叶边界都折叠成一个记号 '#'，
        结果**专名恰恰全被误杀**：专名几乎总是紧跟标点或独占整叶，左侧永远是 '#'，
        于是集合大小恒为 1。灵敏度回测（注入 扎拉贾->扎拉迦 / 艾利奥文->艾莉奥雯）
        三条注入漏报两条，就是这条规则写反了造成的。
        """
        v = self._br.get(g)
        if v is not None:
            return v
        L, R, start, n = set(), set(), 0, 0
        t = self.text
        while n < 300:
            i = t.find(g, start)
            if i < 0:
                break
            n += 1
            lc = t[i - 1] if i else '\x00'
            rc = t[i + len(g)] if i + len(g) < len(t) else '\x00'
            L.add(lc if CJK_RE.match(lc) else '#')
            R.add(rc if CJK_RE.match(rc) else '#')
            start = i + 1
        stuck = (lambda S: len(S) == 1 and next(iter(S)) != '#')
        v = self._br[g] = not (stuck(L) or stuck(R))
        return v


def build_containment(names):
    """n -> 包含 n 的更长专名集合（按词，用于长名遮蔽的快速预筛）。"""
    by_word = collections.defaultdict(set)
    for n in names:
        for w in n.split(' '):
            by_word[w].add(n)
    cont = {}
    for n in names:
        ws = n.split(' ')
        cand = set(by_word[ws[0]])
        for w in ws[1:]:
            cand &= by_word[w]
        cand = {m for m in cand if len(m) > len(n) and
                re.search(r'(?<![A-Za-z])' + re.escape(n) + r'(?![A-Za-z])', m)}
        if cand:
            cont[n] = cand
    return cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=60)
    ap.add_argument('--engine', default='both', choices=['both', 'near', 'free'])
    ap.add_argument('--min-total', type=int, default=3, help='正文英文命中叶数下限')
    ap.add_argument('--min-spec', type=float, default=0.5,
                    help='对手词特异度下限 = 该词在 N 的正文叶里出现数 / 全库出现数')
    ap.add_argument('--max-cn', type=int, default=1500, help='用于挖词的 miss 叶长度上限')
    ap.add_argument('--max-miss', type=int, default=14, help='每个专名最多用几片 miss 叶挖词')
    ap.add_argument('--min-sim', type=float, default=0.0,
                    help='只留最佳对手词与词典译名字形相似度 >= 此值的组。'
                         '实测（Ember 全库）：0 报 164 组，0.5 报 50 组且其中约 44%% 是真分裂；'
                         '但 Lake Jinro(锦露湖) / Hallows(圣堂区) 这类**意译**分裂 sim 只有 0.33，'
                         '所以别把这个阈值当成过滤器用，它只是排序辅助')
    ap.add_argument('--extra-cn', help='灵敏度回测：额外注入的 {path: cn} 覆盖（JSON）')
    a = ap.parse_args()

    # ---------- 1. 装载 ----------
    override = json.load(open(a.extra_cn, encoding='utf-8')) if a.extra_cn else {}
    packs = []
    for repo in a.repo:
        en_dir = os.path.join(repo, 'compendium', 'en')
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        for pack in sorted(os.listdir(en_dir)):
            if not pack.endswith('.json') or pack == '_source.json':
                continue
            cn_p = os.path.join(cn_dir, pack)
            if not os.path.exists(cn_p):
                continue
            en = dict(walk(load(os.path.join(en_dir, pack)).get('entries', {})))
            cn = dict(walk(load(cn_p).get('entries', {})))
            for k, v in override.get(pack, {}).items():
                cn['entries.' + k if not k.startswith('entries.') else k] = v
            packs.append((repo, pack, en, cn))

    # ---------- 2. name 词典 ----------
    votes = collections.defaultdict(collections.Counter)
    for repo, pack, en, cn in packs:
        for path, v in en.items():
            if not path.endswith('.name'):
                continue
            c = cn.get(path)
            if not c or not CJK_RE.search(c) or not looks_proper(v):
                continue
            h = head(c)
            if len(h) >= 2 and h not in CN_STOP:
                votes[v][h] += 1
    print(f'英文专名 -> 中文译名 词典 {len(votes)} 条'
          f'（name 侧本身分裂 {sum(1 for c in votes.values() if len(c) > 1)} 条）')

    names = list(votes)
    first_idx = collections.defaultdict(list)
    RX = {}
    for n in names:
        first_idx[n.split(' ')[0].lower()].append(n)
        RX[n] = re.compile(r'(?<![A-Za-z])' + re.escape(n) + r'(?![A-Za-z])')
    CONT = build_containment(names)

    # ---------- 3. 正文叶 + 语料 ----------
    body, corpus_parts = [], []
    for repo, pack, en, cn in packs:
        for path, v in en.items():
            c = cn.get(path)
            if not c or not CJK_RE.search(c):
                continue
            cp = plain_cn(c)
            corpus_parts.append(cp)
            if path.endswith('.name') or path.endswith('.tokenName'):
                continue
            body.append((repo, pack, path, plain_en(v), cp))
    CO = Corpus(corpus_parts)
    print(f'正文叶 {len(body)}   全库中文语料 {len(CO.text)} 字')

    # ---------- 4. 英文闸（含长名遮蔽） ----------
    occ = collections.defaultdict(list)
    for i, (_r, _p, _pt, ep, _cp) in enumerate(body):
        cand = set()
        for t in {w.lower() for w in WORD.findall(ep)}:
            if t in first_idx:
                cand.update(first_idx[t])
        if not cand:
            continue
        hit = {}
        for n in cand:
            m = RX[n].search(ep)
            if m:
                hit[n] = n
        for n in hit:
            shadow = CONT.get(n)
            if shadow and shadow & hit.keys():
                # n 的每一次出现是否都被更长的专名包住
                spans = [m.span() for m in RX[n].finditer(ep)]
                big = []
                for m in shadow & hit.keys():
                    big += [s.span() for s in RX[m].finditer(ep)]
                if all(any(b <= s and e <= f for b, f in big) for s, e in spans):
                    continue
            occ[n].append(i)
    print(f'英文闸命中 {len(occ)} 个专名 / {sum(len(v) for v in occ.values())} 组(专名,叶)')

    # ---------- 5a. 分桶 + 候选攒集 ----------
    pending, n_suspect = [], 0
    allcand = set()
    for n, idxs in sorted(occ.items()):
        if len(idxs) < a.min_total:
            continue
        known = sorted(votes[n], key=len, reverse=True)
        hits, miss = collections.Counter(), []
        for i in idxs:
            cp = body[i][4]
            got = [z for z in known if z in cp]
            if got:
                hits[max(got, key=len)] += 1
            else:
                miss.append(i)
        if len(miss) < 2:
            continue
        n_suspect += 1
        usable = [i for i in miss if len(body[i][4]) <= a.max_cn][:a.max_miss]
        if len(usable) < 2:
            continue
        need = max(2, (len(usable) + 1) // 2)

        Z = known[0]
        df = collections.Counter()
        for i in usable:
            seen = set()
            for run in CJK_RUN.findall(body[i][4]):
                L = len(run)
                ks = set()
                if a.engine in ('both', 'free'):
                    ks |= {k for k in range(2, 7) if k <= L}
                if a.engine in ('both', 'near'):
                    ks |= {k for k in (len(Z) - 1, len(Z), len(Z) + 1) if 2 <= k <= L}
                for k in ks:
                    for j in range(L - k + 1):
                        seen.add(run[j:j + k])
            for g in seen:
                df[g] += 1

        scope = '\x00'.join(body[i][4] for i in idxs)
        cands = []
        for g, d in df.items():
            if d < need or any(g in z or z in g for z in known):
                continue
            fg = scope.count(g)
            if fg < need:
                continue
            cands.append((g, d, fg))
        if not cands:
            continue
        allcand.update(g for g, _, _ in cands)
        pending.append((n, idxs, known, hits, miss, usable, cands))

    print(f'有 >=2 片 miss 叶的专名 {n_suspect} 个；待定候选串 {len(allcand)} 条'
          f'（{len(pending)} 个专名）')
    CO.tally(allcand)

    # ---------- 5b. 特异度 + 边界熵 + 极大化 ----------
    findings = []
    for n, idxs, known, hits, miss, usable, cands in pending:
        Z = known[0]
        keep = []
        for g, d, fg in cands:
            bg = CO.count(g)
            if not bg or fg / bg < a.min_spec:
                continue
            if not CO.branching(g):
                continue
            keep.append((g, d, fg, bg, round(lcs(g, Z) / max(len(g), len(Z)), 2)))
        if not keep:
            continue
        keep.sort(key=lambda t: -len(t[0]))
        maximal = []
        for rec in keep:
            g, d, fg, bg, sim = rec
            if any(g in G and d == D and fg == F for G, D, F, _, _ in maximal):
                continue
            maximal.append(rec)
        maximal.sort(key=lambda t: (-t[4], -t[1], -len(t[0])))
        rivals = maximal[:6]

        total = sum(hits.values()) + len(miss)
        minor = min(sum(hits.values()), len(miss))
        findings.append({
            'english': n,
            'body_leaves': len(idxs),
            'name_variants': dict(votes[n]),
            'hit': dict(hits),
            'miss_leaves': len(miss),
            'balance': round(minor / total, 3),
            'rivals': [{'cn': g, 'miss_df': d, 'fg': fg, 'bg': bg,
                        'spec': round(fg / bg, 2), 'sim_to_dict': s}
                       for g, d, fg, bg, s in rivals],
            'samples': [{
                'repo': os.path.basename(body[i][0]), 'pack': body[i][1],
                'path': body[i][2],
                'batch_path': body[i][2][8:] if body[i][2].startswith('entries.')
                              else body[i][2],
                'en_ctx': _ctx(body[i][3], n),
                'cn_ctx': body[i][4][:260],
            } for i in usable[:4]],
        })

    if a.min_sim > 0:
        findings = [f for f in findings
                    if max((r['sim_to_dict'] for r in f['rivals']), default=0) >= a.min_sim]
    findings.sort(key=lambda f: (-max((r['sim_to_dict'] for r in f['rivals']), default=0),
                                 -f['balance'], -f['body_leaves']))
    print(f'报出对手译名的 **{len(findings)}** 个专名\n')
    for f in findings[:a.show]:
        rv = ' | '.join(f'{r["cn"]}×{r["miss_df"]}(sim{r["sim_to_dict"]},sp{r["spec"]})'
                        for r in f['rivals'][:3])
        hv = ' | '.join(f'{k}×{v}' for k, v in f['hit'].items()) or '(正文从不用词典译名)'
        print(f'  {f["english"][:28]:30s} bal={f["balance"]:.2f} 叶{f["body_leaves"]:4d} '
              f'词典:{hv[:34]:36s} 对手:{rv}')
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'dict_size': len(votes), 'body_leaves': len(body),
                       'gated_names': len(occ), 'suspects': n_suspect,
                       'findings': findings}, fh, ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


def _ctx(text, name, w=70):
    m = re.search(r'(?<![A-Za-z])' + re.escape(name) + r'(?![A-Za-z])', text)
    return text[:160] if not m else text[max(0, m.start() - w):m.end() + w]


if __name__ == '__main__':
    main()
