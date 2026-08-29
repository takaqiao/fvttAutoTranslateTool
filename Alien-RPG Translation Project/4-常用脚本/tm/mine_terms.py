#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按**英文闸**批量开采字幕语料，产出**逐术语 / 逐血脉**的投票表。

    python mine_terms.py [-o out.json] [--print KEY]

与 term_vote.py 的关系
----------------------
直接 `import term_vote` 复用它的 load()，因此**血脉折叠与去重完全继承**：
CnSCG 的异形 1/2/3/4 仍然只算一票，镜像文件（chs&eng / eng&chs、ass / srt）仍然只留一份。
本脚本不是平行工具，只是把 term_vote 的单术语查询扩成整表 + 加了三样东西：

  1. **提升度（lift）过滤**：裸取汉字块会把「我们/这个/他们」当候选。
     lift = 命中行里的出现率 / 全语料里的出现率。只有显著偏向命中行的写法才是译法。
  2. **拉丁候选**：LV-426 / APC / EEV / UPP 这类中文侧原样保留的，汉字正则永远抓不到。
  3. **逐血脉众数 + 新旧分歧标记**：新优先级把 Romulus2024 / AlienEarth2025 放在 CnSCG 之上，
     所以「新血脉和 CnSCG 不一致」必须显式报出来，而不是混在总票里被 CnSCG 的行数淹没。

排序口径：**血脉数第一，行数第二**（PROJECT.md §3.3）。
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import term_vote  # noqa: E402  复用血脉折叠 + 去重

NEW_LINEAGES = {'Romulus2024', 'AlienEarth2025'}   # 优先级 2：最新、面向大陆
CNSCG = 'CnSCG(异形1/2/3/4·同源)'                   # 优先级 6：四部折一票

CJK = r'一-鿿'
RUN_RE = re.compile(f'[{CJK}][{CJK}·・‧‐-―\\-]*[{CJK}]|[{CJK}]')
LATIN_RE = re.compile(r'[A-Za-z][A-Za-z0-9]*(?:[\-.][A-Za-z0-9]+)*|[0-9]+(?:[\-][A-Za-z0-9]+)+')
MAX_NGRAM = 8

# 中文侧纯功能词/代词，永远不是术语译法
# 功能字：出现即说明这块是句子碎片而不是术语（上/下/中 故意不列，军衔要用）
# 才/会/好 曾在此表里，把 Prodigy=天才 这类真术语误杀了，已移除。
FUNC = set('的了吗呢吧啊呀哦嗯我你您他她它们这那哪什么怎谁很就都也还又再只已却竟被把给对从向跟和与或但而且因所如若虽然使让请别没不是在要得着过之其此于并说想知觉应该可需')

STOP = set('我们 你们 他们 她们 它们 这个 那个 什么 怎么 为什么 可以 不是 就是 没有 一个 自己 现在 已经 因为 所以 如果 但是 还有 知道 觉得 应该 需要 可能 时候 这样 那样 出来 进去 起来 一下 这里 那里 东西 事情'.split())


def ngrams(text, lo=1, hi=MAX_NGRAM):
    out = set()
    for run in RUN_RE.findall(text):
        for n in range(lo, min(hi, len(run)) + 1):
            for i in range(len(run) - n + 1):
                g = run[i:i + n]
                if g[0].isalnum() or '一' <= g[0] <= '鿿':
                    out.add(g)
    return out


def latin_toks(text):
    return {t for t in LATIN_RE.findall(text) if len(t) >= 2 and not t.isdigit()}


def build_index(rows):
    """全语料基线：每个候选出现在多少行（用于 lift 分母）。"""
    base = Counter()
    for _lin, _rel, _en, zh in rows:
        for g in ngrams(zh):
            base[g] += 1
        for t in latin_toks(zh):
            base['~L' + t] += 1
    return base


def mine(rows, base, total_corpus, spec):
    en_re = re.compile(spec['en'], 0 if spec.get('cs') else re.I)
    hits = [r for r in rows if en_re.search(r[2])]
    by_lin_hits = Counter(r[0] for r in hits)

    rec = {
        'key': spec['key'],
        'category': spec['cat'],
        'english_regex': spec['en'],
        'case_sensitive': bool(spec.get('cs')),
        'total_hits': len(hits),
        'hits_by_lineage': dict(sorted(by_lin_hits.items())),
        'no_evidence': not hits,
        'evidence_grade': ('none' if not hits else 'thin' if len(hits) <= 3 else 'usable'),
        'candidates': [],
        'per_lineage_top': {},
        'new_vs_cnscg': {'status': 'no_evidence'},
        'examples': [],
    }
    if not hits:
        rec['note'] = 'ZERO English-gated hits - NOT evidence. Goes to pending.json, never into the spine.'
        return rec

    # 候选计票
    lines = defaultdict(int)
    lins = defaultdict(set)
    perlin = defaultdict(Counter)
    kind = {}
    for lin, _rel, _en, zh in hits:
        for g in ngrams(zh):
            lines[g] += 1
            lins[g].add(lin)
            perlin[g][lin] += 1
            kind[g] = 'han'
        for t in latin_toks(zh):
            k = '~L' + t
            lines[k] += 1
            lins[k].add(lin)
            perlin[k][lin] += 1
            kind[k] = 'latin'

    n = len(hits)
    kept = {}
    for g, lc in lines.items():
        disp = g[2:] if kind[g] == 'latin' else g
        if disp in STOP:
            continue
        if kind[g] == 'han' and any(ch in FUNC for ch in disp):
            continue
        b = base.get(g, lc) or lc
        lift = (lc / n) / (b / total_corpus)
        exclusive = (b == lc)
        # 稀疏术语（命中极少）放宽：只要该写法几乎只出现在命中行里就留
        ok = (lift >= 4.0 and lc >= 2) or (exclusive and lc >= 2) or (n <= 4 and lift >= 8.0)
        if kind[g] == 'han' and len(disp) == 1 and not (lift >= 25.0 and lc >= 2):
            ok = False
        if ok:
            kept[g] = (disp, lc, lift, b, exclusive)

    # 单字若只是某个存活多字候选的碎片（如 检疫规定 里的「定」），毙掉；
    # 但独立成词的单字（卵/酸/蛋）保留。不做「同计数只留长的」剪枝——那会把
    # 机关枪 这种真术语吃掉，只留下 拿出机关枪 这种整句碎片。
    drop = set()
    multi = [k for k in kept if kind[k] == 'han' and len(k) >= 2]
    for a in list(kept):
        if kind[a] == 'han' and len(a) == 1 and any(a in m for m in multi):
            drop.add(a)

    # 极大化折叠：n-gram 会把「焚化炉」切成 焚化/焚化炉/化炉 三个位移窗口。
    # 若 c 只出现在更长的 c' 内部（行数与血脉集合全同），c 不携带新信息，毙掉。
    # c' 限长 6：本领域术语与音译都在 6 字内（合成人/焚化炉/诺斯特罗莫/普罗米修斯），
    # 再长的等长同计数串基本是整句碎片，不许它吞掉真术语。
    for a in list(kept):
        if kind[a] != 'han' or a in drop:
            continue
        for b2 in kept:
            if b2 == a or kind[b2] != 'han' or len(b2) <= len(a) or len(b2) > 6:
                continue
            if a in b2 and kept[a][1] == kept[b2][1] and lins[a] == lins[b2]:
                drop.add(a)
                break

    cands = []
    for g, (disp, lc, lift, b, exclusive) in kept.items():
        if g in drop:
            continue
        cands.append({
            'zh': disp,
            'kind': kind[g],
            'lineage_count': len(lins[g]),
            'line_count': lc,
            'lineages': sorted(lins[g]),
            'per_lineage_lines': dict(sorted(perlin[g].items())),
            'corpus_lines': b,
            'lift': round(lift, 1),
            'exclusive_to_hits': exclusive,
        })
    # 血脉数第一，行数第二
    cands.sort(key=lambda c: (-c['lineage_count'], -c['line_count'], -c['lift']))
    rec['candidates'] = cands[:14]

    # ---- 人工审定候选：精确计票（0 也照报）----
    probes = []
    for h in spec.get('zh') or []:
        hr = re.compile(h)
        hl = [r for r in hits if hr.search(r[3])]
        probes.append({
            'zh': h,
            'lineage_count': len({r[0] for r in hl}),
            'line_count': len(hl),
            'lineages': sorted({r[0] for r in hl}),
            'per_lineage_lines': dict(sorted(Counter(r[0] for r in hl).items())),
        })
    probes.sort(key=lambda c: (-c['lineage_count'], -c['line_count'], -len(c['zh'])))
    if probes:
        rec['audited_candidates'] = probes
        rec['audited_zero'] = [c['zh'] for c in probes if c['line_count'] == 0]

    # ---- 逐血脉众数（审定优先）----
    # 规则：在该血脉内覆盖过半命中行的审定候选里取**最长**的。
    # 只取最高计数会选出「威兰」而不是「威兰汤谷」——短串必然被长串包含，
    # 计数永远不低，所以必须先卡覆盖率再比长度。
    for lin in sorted(by_lin_hits):
        pool = [c for c in probes if c['per_lineage_lines'].get(lin)]
        if not pool:
            continue
        need = max(1, by_lin_hits[lin] * 0.5)
        major = [c for c in pool if c['per_lineage_lines'][lin] >= need]
        best = (max(major, key=lambda c: (len(c['zh']), c['per_lineage_lines'][lin]))
                if major else
                max(pool, key=lambda c: (c['per_lineage_lines'][lin], len(c['zh']))))
        rec['per_lineage_top'][lin] = {
            'zh': best['zh'], 'lines': best['per_lineage_lines'][lin],
            'of_hits': by_lin_hits[lin], 'source': 'audited'}
    rec['per_lineage_top_auto'] = {}

    # 逐血脉众数：该血脉内行数最高者，同分取更长（更具体）
    for lin in sorted(by_lin_hits):
        pool = [c for c in cands if lin in c['per_lineage_lines']]
        if not pool:
            rec['per_lineage_top'][lin] = None
            continue
        best = max(pool, key=lambda c: (c['per_lineage_lines'][lin], c['lift'], -len(c['zh'])))
        rec['per_lineage_top_auto'][lin] = {
            'zh': best['zh'], 'lines': best['per_lineage_lines'][lin]}
        if not probes:
            rec['per_lineage_top'].setdefault(lin, dict(
                rec['per_lineage_top_auto'][lin], source='auto-mined'))

    for lin in sorted(by_lin_hits):
        rec['per_lineage_top'].setdefault(lin, None)

    # 新血脉 vs CnSCG
    newtops = {l: v for l, v in rec['per_lineage_top'].items() if l in NEW_LINEAGES and v}
    cn = rec['per_lineage_top'].get(CNSCG)
    if not newtops and not cn:
        rec['new_vs_cnscg'] = {'status': 'no_lineage_top'}
    elif not newtops:
        rec['new_vs_cnscg'] = {'status': 'cnscg_only', 'cnscg': cn}
    elif not cn:
        rec['new_vs_cnscg'] = {'status': 'new_only', 'new': newtops}
    else:
        agree = all(v['zh'] == cn['zh'] or v['zh'] in cn['zh'] or cn['zh'] in v['zh']
                    for v in newtops.values())
        rec['new_vs_cnscg'] = {
            'status': 'agree' if agree else 'DISAGREE',
            'new': newtops, 'cnscg': cn,
            'ruling': None if agree else
            'New lineages (priority 2) outrank CnSCG (priority 6) - prefer the new rendering.',
        }

    # 人工审定候选：精确计票，绕过 lift 闸（0 也照报，"没有"本身是结论）
    if False:
        probes = []
        for h in spec['zh']:
            hr = re.compile(h)
            hl = [r for r in hits if hr.search(r[3])]
            probes.append({
                'zh': h,
                'lineage_count': len({r[0] for r in hl}),
                'line_count': len(hl),
                'lineages': sorted({r[0] for r in hl}),
                'per_lineage_lines': dict(sorted(Counter(r[0] for r in hl).items())),
            })
        probes.sort(key=lambda c: (-c['lineage_count'], -c['line_count']))
        rec['audited_candidates'] = probes

    seen_lin = Counter()
    for lin, _rel, en, zh in hits:
        if seen_lin[lin] < 3:
            seen_lin[lin] += 1
            rec['examples'].append({'lineage': lin, 'en': en[:150], 'zh': zh[:150]})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--print', dest='pr', default=None)
    args = ap.parse_args()

    from terms_spec import TERMS
    rows = term_vote.load()
    total = len(rows)
    base = build_index(rows)
    lin_sizes = Counter(r[0] for r in rows)

    out = {
        'meta': {
            'corpus_dir': term_vote.CORPUS,
            'total_pairs': total,
            'lineages': dict(sorted(lin_sizes.items())),
            'lineage_folding': 'inherited from term_vote.load() - CnSCG Alien 1/2/3/4 = ONE vote',
            'max_lineage_count': len(lin_sizes),
            'ranking': 'lineage_count first, line_count second',
            'new_lineages': sorted(NEW_LINEAGES),
            'method': 'English-gated hits; Chinese candidates = char n-grams (1..8) + Latin tokens, '
                      'filtered by lift = P(cand|hit)/P(cand|corpus); substring pruning.',
        },
        'terms': [],
    }
    for spec in TERMS:
        out['terms'].append(mine(rows, base, total, spec))

    zero = [t['key'] for t in out['terms'] if t['no_evidence']]
    dis = [t['key'] for t in out['terms'] if t['new_vs_cnscg'].get('status') == 'DISAGREE']

    out['zero_hit_unusable'] = [{
        'key': t['key'], 'category': t['category'], 'english_regex': t['english_regex'],
        'verdict': 'ZERO English-gated hits - NOT evidence. Do not put in the spine; send to pending.json.',
    } for t in out['terms'] if t['no_evidence']]

    out['new_lineage_vs_cnscg_disagreements'] = [{
        'key': t['key'], 'category': t['category'], 'total_hits': t['total_hits'],
        'new_lineages': {k: v for k, v in t['new_vs_cnscg'].get('new', {}).items()},
        'cnscg': t['new_vs_cnscg'].get('cnscg'),
        'ruling': 'Romulus2024 / AlienEarth2025 sit at priority 2; CnSCG at priority 6. '
                  'The new rendering outranks CnSCG unless a higher source overrides.',
    } for t in out['terms'] if t['new_vs_cnscg'].get('status') == 'DISAGREE']

    out['thin_evidence'] = [t['key'] for t in out['terms']
                            if t['evidence_grade'] == 'thin']
    out['meta']['zero_hit_terms'] = zero
    out['meta']['zero_hit_count'] = len(zero)
    out['meta']['disagreement_terms'] = dis
    out['meta']['term_count'] = len(out['terms'])

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f'wrote {args.out}  ({len(out["terms"])} terms, {len(zero)} zero-hit, {len(dis)} disagreements)')

    if args.pr:
        for t in out['terms']:
            if t['key'].lower() == args.pr.lower():
                print(json.dumps(t, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
