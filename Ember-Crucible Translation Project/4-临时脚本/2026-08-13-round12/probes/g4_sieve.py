#!/usr/bin/env python3
"""G4 筛子：给 scan_en_drift 的 `changed` 桶排优先级。

判据换掉了 —— 不用长度比（§8 已证在短叶子上零鉴别力），改用
**「上游这次动过的、翻译不掉的 token，在中文里在不在」**：

  A 信号  新英文独有的 token 不在中文里   → 中文没跟上新增/改动
  B 信号  旧英文独有的 token 还在中文里   → 中文留着上游已删的内容

token 只取「过了翻译也不变形」的四类，因此不受译笔影响：

  ENR   `@Check[...]` / `@UUID[...]` / `@Damage[...]`  —— 整串照抄，含目标不含 {标签}
  ANCH  `#some-slug`  页内锚点
  NUM   数字与骰式 `18` `2d6` `+3`
  PROP  首字母大写的专名，且**在 glossary_ec 里查得到**（否则不算，避免句首词噪声）
        —— 中文里出现「中文译名」或「英文原名」任一即算命中

输出按 (A 类 ENR/ANCH > A 类 NUM/PROP > B 类) 分档，便于按档抽样估假阳性。
"""
from __future__ import annotations
import argparse, json, os, re

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')
ENRICHER = re.compile(r'@(\w+)\[([^\]]*)\]')
ANCHOR = re.compile(r'#([A-Za-z0-9][A-Za-z0-9\-_]{2,})')
# 边界必须写成 ASCII 类。用 `\w` 会把 CJK 也算进单词字符，于是「30英尺」里的 30
# 在中文侧永远提取不到 —— 首轮 38 条 A_num 大半栽在这。
NUM = re.compile(r'(?<![0-9A-Za-z.])(\d+d\d+|\d+)(?![0-9A-Za-z])')
CAPSEQ = re.compile(r'\b[A-Z][a-zA-Z\'’]+(?:[ \-][A-Z][a-zA-Z\'’]+)*\b')


def load_json(path):
    raw = open(path, encoding='utf-8-sig').read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r',(\s*[}\]])', r'\1', raw))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out['.'.join(path)] = node


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def baseline_packs(bdir):
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        key = ('ember.crucible-adventure.json' if f == '_repaired.json'
               else f.replace('-en.json', '.json'))
        out.setdefault(key, os.path.join(bdir, f))
    return out


# ---------- token 提取 ----------

LABELPARAM = re.compile(r'\s*\b(?:label|readaloud|name|title|text)\s*=\s*"[^"]*"')


def enrichers(s):
    """`@Type[target]`。两处必须抹掉，否则全是假阳性：

    * `{显示标签}` —— 要翻的
    * `label="…"` / `readaloud="…"` 参数 —— 同样要翻（首轮筛子 13 条 T1 全栽在这）
    """
    out = set()
    for t, tgt in ENRICHER.findall(s):
        out.add(f'@{t}[{norm(LABELPARAM.sub("", tgt))}]')
    return out


def anchors(s):
    return {'#' + m for m in ANCHOR.findall(s)}


def _strip(s):
    """去掉标签与 enricher 目标，只留会被读到的正文。"""
    s = ENRICHER.sub(lambda m: ' ', s)
    return TAG.sub(' ', s)


def numbers(s):
    return set(NUM.findall(_strip(s)))


STOP = set('''The A An And But Or If When While This That These Those You Your They Their He She It
Each Any All Some One Two Three Four Five Six Seven Eight Nine Ten In On At To For From With Without
By Of As Is Are Was Were Be Been Being Do Does Did Can Could May Might Must Should Would Will Shall
Not No Yes So Then Than There Here What Which Who Whom Whose How Why Where After Before During Until
Once Also Even Only Just Still Yet Both Either Neither Every Other Another Such Same Own More Most
Less Least Many Much Few Several First Second Third Last Next Previous New Old Good Bad Great Small
Large Long Short High Low Left Right Up Down Out Over Under Above Below Between Through Against Upon
Note Success Failure Critical Special Effect Trigger Requirements Frequency Cost Duration Range Area
Targets Level Actions Reaction Free Action Passive Round Turn Player Players Character Characters
GM Game Master Chapter Page Table Read Aloud Boxed Text If The'''.split('\n'))
STOP = {w for line in STOP for w in line.split()}


def props(s, gl, common):
    """大写序列里**能在术语表查到、且确实是专名**的那些。

    两个必要的收紧（首轮各贡献大量假阳性）：

    1. 取**所有连续子跨度**而非最大跨度 —— 否则新英文的「The Coat Room」查不到，
       旧英文的「Coat Room」查得到，凭空造出一条 B 信号。
    2. 单词术语必须是专名：拿全语料统计，凡该词的**小写形式**在英文里出现过
       （light / north / hidden / coins / thief…），就不是专名，丢掉。
    """
    out = {}
    for m in CAPSEQ.findall(_strip(s)):
        w = re.split(r'[ \-]', m)
        for i in range(len(w)):
            for j in range(i + 1, min(i + 5, len(w)) + 1):
                span = ' '.join(w[i:j])
                if span in STOP or (j - i == 1 and span.lower() in common):
                    continue
                v = gl.get(span)
                if v:
                    out[span] = v
    return out


def label_only(term, s):
    """该词在 s 里是否**只**以 `@Enricher[...]{term}` 的显示标签形态出现过。"""
    esc = re.escape(term)
    n_all = len(re.findall(esc, s))
    n_lab = len(re.findall(r'\]\{' + esc + r'\}', s))
    return n_all > 0 and n_all == n_lab


def cn_zh(val):
    """术语表值形如 `中文 English`，取其中的中文段（英文段另算命中）。"""
    parts = re.findall(r'[一-鿿　-〿“”‘’·、（）]+', val)
    return max(parts, key=len) if parts else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--glossary', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-len', type=int, default=40)
    a = ap.parse_args()

    gl = load_json(a.glossary)
    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')

    # 「小写形式出现过 = 不是专名」的语料表，扫全部当前英文包一次建好
    common = set()
    for f in os.listdir(en_dir):
        if not f.endswith('.json') or f == '_source.json':
            continue
        tmp = {}
        leaves(load_json(os.path.join(en_dir, f)).get('entries', {}), [], tmp)
        for v in tmp.values():
            common.update(re.findall(r'\b[a-z]{2,}\b', _strip(v)))

    rows, total_changed = [], 0
    for pack, oldpath in baseline_packs(a.baseline).items():
        cur_p = os.path.join(en_dir, pack)
        if not os.path.exists(cur_p):
            continue
        o, n, c = {}, {}, {}
        leaves(load_json(oldpath).get('entries', {}), [], o)
        leaves(load_json(cur_p).get('entries', {}), [], n)
        cnp = os.path.join(cn_dir, pack)
        if os.path.exists(cnp):
            leaves(load_json(cnp).get('entries', {}), [], c)
        for path, new_en in n.items():
            old_en = o.get(path)
            if old_en is None or norm(old_en) == norm(new_en):
                continue
            if len(TAG.sub('', new_en)) < a.min_len:
                continue
            cn = c.get(path)
            if not (cn and CJK.search(cn)):
                continue
            total_changed += 1

            ev = {}
            # --- A: 新英文独有 token 缺席于中文 ---
            ne, oe, ce = enrichers(new_en), enrichers(old_en), enrichers(cn)
            miss = sorted((ne - oe) - ce)
            if miss:
                ev['A_enricher'] = miss
            na, oa, ca = anchors(new_en), anchors(old_en), anchors(cn)
            missa = sorted((na - oa) - ca)
            if missa:
                ev['A_anchor'] = missa
            nn, on, cnum = numbers(new_en), numbers(old_en), numbers(cn)
            missn = sorted((nn - on) - cnum)
            if missn:
                ev['A_num'] = missn
            npr, opr = props(new_en, gl, common), props(old_en, gl, common)
            missp = []
            for term, val in npr.items():
                # `term in opr` 不够：该词可能在旧英文里出现过、只是被 enricher/小写过滤挡掉。
                # 判据是「上游这次**新写进来**的词」，所以拿原串再兜一次。
                if term in opr or term in old_en:
                    continue
                zh = cn_zh(val)
                if term in cn or (zh and zh in cn):
                    continue
                missp.append(f'{term}->{val[:40]}')
            if missp:
                ev['A_prop'] = sorted(missp)

            # --- B: 旧英文独有 token 还留在中文里 ---
            keep = sorted((oe - ne) & ce)
            if keep:
                ev['B_enricher'] = keep
            keepa = sorted((oa - na) & ca)
            if keepa:
                ev['B_anchor'] = keepa
            keepn = sorted((on - nn) & cnum)
            if keepn:
                ev['B_num'] = keepn
            keepp = []
            for term, val in opr.items():
                if term in npr or term in new_en:
                    continue
                # 上游把 `@UUID[x]{Label}` 的显示标签摘掉、只留裸 UUID —— 中文那边
                # 保留自己的标签完全合法（裸 UUID 本来也渲染成目标的中文名）。
                # 该词若在旧英文里**只**以标签形态出现过，就不算证据。抽样里 B_prop
                # 的假阳性大半是这一种。
                if label_only(term, old_en):
                    continue
                zh = cn_zh(val)
                if term in cn or (zh and zh in cn):
                    keepp.append(f'{term}->{val[:40]}')
            if keepp:
                ev['B_prop'] = sorted(keepp)

            if not ev:
                continue
            hard = bool({'A_enricher', 'A_anchor', 'B_enricher', 'B_anchor'} & ev.keys())
            tier = ('T1' if hard else
                    'T2' if {'A_num', 'A_prop'} & ev.keys() else 'T3')
            rows.append({
                'tier': tier, 'pack': pack, 'path': path,
                'ev': ev,
                'n_ev': sum(len(v) for v in ev.values()),
                'en_len_old': len(TAG.sub('', old_en)),
                'en_len_new': len(TAG.sub('', new_en)),
                'cn_len': len(TAG.sub('', cn)),
                'old_en': old_en, 'new_en': new_en, 'cn': cn,
            })

    rows.sort(key=lambda r: (r['tier'], -r['n_ev']))
    from collections import Counter
    tc = Counter(r['tier'] for r in rows)
    evc = Counter(k for r in rows for k in r['ev'])
    print(f'{os.path.basename(a.repo)}  changed(有中文,>={a.min_len}) {total_changed}'
          f'  -> 命中 {len(rows)}   {dict(tc)}')
    print('  证据类型分布', dict(evc))
    json.dump({'total_changed': total_changed, 'hit': len(rows),
               'tiers': dict(tc), 'ev_kinds': dict(evc), 'items': rows},
              open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('->', a.out)


if __name__ == '__main__':
    main()
