#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""叶内重复块 —— 复制粘贴事故（第八轮新判据）

要抓什么
--------
历轮 59+ 个并行批次三方合并落盘，`apply_translations.py` 是**整叶覆盖**。
merge 把一段内容重放两遍、或译者复制上一段忘了改标签，都会在中文侧留下
「两块一模一样」而英文侧对应的两块其实不同。

为什么既有判据全都不响
----------------------
* BLOCK 闸（scan_markup_drift）比的是**块标签多重集**，重复块本身就是合法的 `<p>`，多重集不变；
* 数字覆盖比的是**整叶数字多重集**，块内标签被复制时数字往往还都在；
* scan_content_coverage 比长度比值，一叶里多一段少一段影响不到阈值；
* 人通读时看到两段一样的中文，往往以为原文就长这样。

子判据
------
R1 EXACT_DUP     中文两块逐字节相同，英文两块**有实义差异**（数字/专名/enricher 载荷）。
R2 LABEL_COLLAPSE 中文两块高度相似（>=0.80），块内**同序位**的 `<strong>/<b>/<em>` 行内标签
                 英文不同而中文相同 —— 典型的「品质档位标签复制粘贴没改」。
R3 LEAF_DUP      两条**不同路径**的中文叶逐字节相同，而英文差异大（TM 预填留下的整叶串味）。
R4 LADDER        （opt-in，`--rule R4`）R2 那一类的**穷尽版**：凡是英文块以品质档位词起头的，
                 都拿中文块的起头标签跟全库定译核一遍。R2 只能抓到「两块相邻且相似」的，
                 R4 连不相邻、不相似的漂移一起抓。默认不跑，因为它已经越界到术语一致性了。

R1/R2/R4 的定位前提：英中两侧按块切分后**块数必须相等**（本库 15358 个含标签叶 100% 相等，
所以按序位配对是安全的）。块数不等的叶直接跳过并计数。

**R2 跳过纯数值行内标签**（`2 feet` / `4 rounds` / `+1`）：中文经常把
「move 2 feet per 20 feet of descent」倒装成「每下降 20 英尺，可移动 2 英尺」，
序位一配就错位，实测这是 R2 唯一的假阳性来源；而数值重复本来就有数字覆盖闸兜底。

用法：
  python scan_dup_block.py --repo "1-Ember汉化插件" --repo "2-Crucible汉化插件" --out out.json
  python scan_dup_block.py --repo ... --rule R4 --show 60
  # 灵敏度回测（只在内存副本上注入，绝不写盘）：
  python scan_dup_block.py --repo ... --rule R1 \
      --inject "entries.Rallying Elixir.actions.rallyingElixir.description" \
      --inject-mode block --inject-from 1 --inject-at 2
"""
from __future__ import annotations
import argparse
import collections
import difflib
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SKIP_KEYS = {'_id', 'path', '_variants', '_when'}

# 块边界：出现即切一刀，保留刀与刀之间的正文。对嵌套（blockquote>p）天然安全。
BLOCK = re.compile(
    r'</?(?:p|li|ul|ol|dl|dd|dt|h[1-6]|blockquote|section|div|table|tbody|thead'
    r'|tr|td|th|hr|br|figure|figcaption|aside|article|header|footer)\b[^>]*/?>',
    re.I)
CJK = re.compile(r'[一-鿿]')
INLINE = re.compile(r'<(strong|b|em|i)\b[^>]*>(.*?)</\1>', re.S | re.I)
TAG = re.compile(r'<[^>]+>')
WORD = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")
NUM = re.compile(r'\d+(?:\.\d+)?')
# enricher 载荷：@X[...] 的方括号内容、[[/cmd ...]] 的整条
ENRICH = re.compile(r'@\w+\[([^\]]*)\]|\[\[/([^\]]*)\]\]')
CAPWORD = re.compile(r"\b[A-Z][A-Za-z'-]{2,}\b")
LEAD_STRONG = re.compile(r'^\s*<(strong|b)\b[^>]*>(.*?)</\1>', re.S | re.I)
# 纯数值/量纲行内标签：中文倒装后序位对不上，R2 一律跳过（数字覆盖闸已兜底）
NUMERIC_SPAN = re.compile(
    r'^\s*[+\-]?\d+(?:\.\d+)?\s*'
    r'(?:feet|foot|ft|squares?|spaces?|rounds?|turns?|hours?|hrs?|minutes?|mins?|days?'
    r'|Health|Morale|Focus|Wounds?|Madness|damage|%)?\s*$', re.I)

# 品质/附魔档位定译（全库多数写法，见 README 里的统计：粗糙100 标准123 精良131 卓越127 大师级135）
LADDER = {
    'Shoddy': '粗糙', 'Standard': '标准', 'Fine': '精良',
    'Superior': '卓越', 'Masterwork': '大师级',
}
# 句首/常见虚词大写，出现在 sym-diff 里没有实义
CAP_STOP = {
    'The', 'This', 'That', 'These', 'Those', 'Any', 'All', 'Each', 'Every',
    'And', 'But', 'For', 'Not', 'Once', 'When', 'While', 'With', 'Without',
    'Though', 'Although', 'However', 'Finally', 'Additionally', 'Instead',
    'They', 'Their', 'You', 'Your', 'His', 'Her', 'Its', 'She', 'One', 'Two',
    'Characters', 'Character', 'Alternatively', 'Otherwise', 'Because',
    'After', 'Before', 'During', 'Upon', 'Should', 'May', 'Can', 'Will',
    'Regardless', 'Furthermore', 'Meanwhile', 'Then', 'There', 'Here',
    'Anyone', 'Someone', 'Successful', 'Success', 'Failure', 'Read', 'Note',
}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str):
        p = '.'.join(path)
        out.append({
            'path': p,
            'batch_path': p[len('entries.'):] if p.startswith('entries.') else p,
            'en': en,
            'cn': cn if isinstance(cn, str) else None,
        })


def blocks(s):
    return [x.strip() for x in BLOCK.split(s) if x.strip()]


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def toks(s):
    return WORD.findall(s.lower())


def en_salient(s):
    """英文块的实义标记：数字 + enricher 载荷 + 非句首常见词的大写词。"""
    out = set(NUM.findall(s))
    for a, b in ENRICH.findall(s):
        out.add('@' + (a or b).strip())
    plain = TAG.sub(' ', s)
    for w in CAPWORD.findall(plain):
        if w in CAP_STOP:
            continue
        out.add(w.rstrip('s') if len(w) > 4 and w.endswith('s') else w)
    return out


def norm_label(s):
    """行内标签归一：去标签、去空白标点、小写、去复数尾 s。"""
    t = TAG.sub('', s)
    t = re.sub(r'[\s:：。，,.;；、·\-—()（）]+', '', t).lower()
    if len(t) > 3 and t.endswith('es'):
        t = t[:-2]
    elif len(t) > 3 and t.endswith('s'):
        t = t[:-1]
    return t


def inline_spans(s):
    return [m.group(2) for m in INLINE.finditer(s)]


def lead_label(s):
    m = LEAD_STRONG.match(s)
    if not m:
        return None
    return TAG.sub('', m.group(2)).strip().strip(':：').strip()


def collect(repos):
    rows = []
    for repo in repos:
        en_dir = os.path.join(repo, 'compendium', 'en')
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json') or fn == '_source.json':
                continue
            cp = os.path.join(repo, 'compendium', 'cn', fn)
            if not os.path.exists(cp):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding='utf-8-sig')).get('entries', {})
            cn = json.load(open(cp, encoding='utf-8-sig')).get('entries', {})
            sub = []
            walk(en, cn, ['entries'], sub)
            for r in sub:
                r['repo'], r['pack'] = repo, fn
            rows.extend(sub)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out')
    ap.add_argument('--rule', action='append',
                    help='只跑某条子判据：R1 / R2 / R3（默认全跑）')
    ap.add_argument('--cn-sim', type=float, default=0.80, help='R2 中文块相似度下限')
    ap.add_argument('--min-block', type=int, default=12, help='R1 中文块最小字数')
    ap.add_argument('--show', type=int, default=30)
    ap.add_argument('--inject', metavar='PATH',
                    help='灵敏度回测：把该叶中文侧第 --inject-from 块复制覆盖第 --inject-at 块。'
                         '只改内存里的副本，绝不写盘。')
    ap.add_argument('--inject-from', type=int, default=0)
    ap.add_argument('--inject-at', type=int, default=1)
    ap.add_argument('--inject-mode', default='block', choices=['block', 'label'],
                    help='block=整块复制（打 R1）；label=只复制块首粗体标签（打 R2）')
    a = ap.parse_args()
    rules = set(a.rule) if a.rule else {'R1', 'R2', 'R3'}

    rows = collect(a.repo)
    stats = collections.Counter()
    stats['叶总数'] = len(rows)
    stats['有中文的叶'] = sum(1 for r in rows if r['cn'])

    # ---- 注入（只改内存中的副本，绝不写盘） ----
    if a.inject:
        tgt = [r for r in rows if r['path'] == a.inject or r['batch_path'] == a.inject]
        if not tgt:
            print(f'!! 注入目标未找到：{a.inject}')
            return
        r = dict(tgt[0])                      # 副本，原 row 不动
        rows = [r if x is tgt[0] else x for x in rows]
        br = blocks(r['cn'])
        src, dst = br[a.inject_from], br[a.inject_at]
        if a.inject_mode == 'block':
            r['cn'] = r['cn'].replace(dst, src, 1)
            print(f'[注入·整块] {r["path"]}\n   块{a.inject_at}「{dst[:60]}」\n   <- 块'
                  f'{a.inject_from}「{src[:60]}」')
        else:
            ls, ld = lead_label(src), lead_label(dst)
            r['cn'] = r['cn'].replace(dst, dst.replace(ld, ls, 1), 1)
            print(f'[注入·标签] {r["path"]}\n   块{a.inject_at} 首标签「{ld}」<- 块'
                  f'{a.inject_from}「{ls}」')

    findings = []

    # ---------------- R1 / R2：叶内 ----------------
    if rules & {'R1', 'R2'}:
        for r in rows:
            e, c = r['en'], r['cn']
            if not c or '<' not in e:
                continue
            er, cr = blocks(e), blocks(c)
            if len(er) != len(cr):
                stats['块数不等（跳过）'] += 1
                continue
            if len(cr) < 2:
                continue
            stats['进入叶内比对的叶'] += 1
            n = len(cr)
            for i in range(n):
                if len(cr[i]) < a.min_block or not CJK.search(cr[i]):
                    continue
                for j in range(i + 1, n):
                    if len(cr[j]) < a.min_block:
                        continue
                    stats['比对过的块对'] += 1
                    if 'R1' in rules and cr[i] == cr[j]:
                        stats['中文两块逐字节相同'] += 1
                        if er[i] == er[j]:
                            stats['  ..英文也相同（上游如此，不报）'] += 1
                        else:
                            si, sj = en_salient(er[i]), en_salient(er[j])
                            if (si - sj) and (sj - si):
                                stats['**R1 命中**'] += 1
                                findings.append({
                                    'rule': 'R1_EXACT_DUP', 'repo': r['repo'], 'pack': r['pack'],
                                    'path': r['path'], 'batch_path': r['batch_path'],
                                    'block_i': i, 'block_j': j,
                                    'en_i': er[i], 'en_j': er[j], 'cn_both': cr[i],
                                    'en_diff': sorted(si ^ sj),
                                })
                            else:
                                stats['  ..英文只是措辞差异（不报）'] += 1
                        continue
                    if 'R2' not in rules:
                        continue
                    csim = sim(cr[i], cr[j])
                    if csim < a.cn_sim:
                        continue
                    stats['中文两块高度相似'] += 1
                    ei_s, ej_s = inline_spans(er[i]), inline_spans(er[j])
                    ci_s, cj_s = inline_spans(cr[i]), inline_spans(cr[j])
                    if not ei_s or not (len(ei_s) == len(ej_s) == len(ci_s) == len(cj_s)):
                        stats['  ..行内标签数不齐（跳过）'] += 1
                        continue
                    bad = []
                    for k in range(len(ei_s)):
                        if NUMERIC_SPAN.match(TAG.sub('', ei_s[k])) \
                                or NUMERIC_SPAN.match(TAG.sub('', ej_s[k])):
                            stats['  ..纯数值标签（跳过，中文倒装会错位）'] += 1
                            continue
                        if norm_label(ei_s[k]) != norm_label(ej_s[k]) \
                                and norm_label(ci_s[k]) == norm_label(cj_s[k]):
                            bad.append({'k': k, 'en_i': ei_s[k], 'en_j': ej_s[k],
                                        'cn_both': ci_s[k]})
                    if not bad:
                        stats['  ..标签有区分（不报）'] += 1
                        continue
                    stats['**R2 命中**'] += 1
                    findings.append({
                        'rule': 'R2_LABEL_COLLAPSE', 'repo': r['repo'], 'pack': r['pack'],
                        'path': r['path'], 'batch_path': r['batch_path'],
                        'block_i': i, 'block_j': j, 'cn_sim': round(csim, 3),
                        'en_i': er[i], 'en_j': er[j], 'cn_i': cr[i], 'cn_j': cr[j],
                        'collapsed': bad,
                    })

    # ---------------- R4：品质档位阶梯穷尽核对（opt-in） ----------------
    if 'R4' in rules:
        for r in rows:
            e, c = r['en'], r['cn']
            if not c or '<strong' not in e:
                continue
            er, cr = blocks(e), blocks(c)
            if len(er) != len(cr):
                continue
            for b, (eb, cb) in enumerate(zip(er, cr)):
                lab = lead_label(eb)
                if lab not in LADDER:
                    continue
                stats['R4 档位行'] += 1
                clab = lead_label(cb)
                want = LADDER[lab]
                if clab == want:
                    stats['  ..合规'] += 1
                    continue
                stats['**R4 命中**'] += 1
                findings.append({
                    'rule': 'R4_LADDER', 'repo': r['repo'], 'pack': r['pack'],
                    'path': r['path'], 'batch_path': r['batch_path'], 'block': b,
                    'en_tier': lab, 'cn_label': clab, 'should_be': want,
                    'en_block': eb, 'cn_block': cb,
                })

    # ---------------- R3：整叶跨路径 ----------------
    if 'R3' in rules:
        g = collections.defaultdict(list)
        for r in rows:
            if r['cn'] and len(r['cn']) >= 20:
                g[r['cn']].append(r)
        for c, lst in g.items():
            if len(lst) < 2:
                continue
            ens = sorted({r['en'] for r in lst})
            if len(ens) == 1:
                stats['R3 同中文同英文（合法）'] += 1
                continue
            stats['R3 同中文异英文'] += 1
            worst = min(sim(toks(ens[x]), toks(ens[y]))
                        for x in range(len(ens)) for y in range(x + 1, len(ens)))
            if worst >= 0.85:
                stats['  ..英文只是拼写/措辞异体（不报）'] += 1
                continue
            stats['**R3 命中**'] += 1
            findings.append({
                'rule': 'R3_LEAF_DUP', 'en_sim': round(worst, 3), 'cn': c,
                'paths': [{'repo': r['repo'], 'pack': r['pack'], 'path': r['path'],
                           'batch_path': r['batch_path'], 'en': r['en']} for r in lst],
            })

    print('\n统计：')
    for k, v in stats.items():
        print(f'  {k:34s} {v}')
    byrule = collections.Counter(f['rule'] for f in findings)
    print(f'\n命中 {len(findings)} 条  {dict(byrule)}')
    for f in findings[:a.show]:
        print('=' * 78)
        if f['rule'] == 'R3_LEAF_DUP':
            print(f'[{f["rule"]}] en_sim={f["en_sim"]}')
            print('  CN :', f['cn'][:160])
            for p in f['paths'][:4]:
                print('  EN :', p['en'][:160])
                print('     @', p['pack'], p['path'][:110])
            continue
        if f['rule'] == 'R4_LADDER':
            print(f'[{f["rule"]}] {f["pack"]} 块{f["block"]}  '
                  f'{f["en_tier"]} -> 「{f["cn_label"]}」 应为「{f["should_be"]}」')
            print('  ', f['path'][:150])
            print('   EN:', f['en_block'][:120])
            print('   CN:', f['cn_block'][:120])
            continue
        print(f'[{f["rule"]}] {f["pack"]}  块[{f["block_i"]},{f["block_j"]}]')
        print('  ', f['path'][:150])
        if f['rule'] == 'R1_EXACT_DUP':
            print('  CN(两块相同):', f['cn_both'][:200])
            print('  EN_i        :', f['en_i'][:200])
            print('  EN_j        :', f['en_j'][:200])
            print('  英文实义差异 :', f['en_diff'])
        else:
            for b in f['collapsed']:
                print(f'  标签#{b["k"]}  EN「{b["en_i"]}」vs「{b["en_j"]}」 -> CN 都是「{b["cn_both"]}」')
            print('  CN_i:', f['cn_i'][:200])
            print('  CN_j:', f['cn_j'][:200])

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({'stats': dict(stats), 'findings': findings},
                  open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
