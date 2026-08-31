#!/usr/bin/env python3
"""drift_aligned.py 的降噪版：把「上游只动了标点/一个词」的块压成一行。

为什么要降噪：51 条 stale 的完整块级 diff 有 520 KB，而其中大半是
`towards→toward`、`- → —`、`Yllith→Ylith` 这种改动 —— 它们不影响中文对错。
真正要人读的是**散文被改写**的块（相似度低）与**新增**的块。

另外记录一个判据上的事实（决定了本脚本为什么不做机械判定）：
全库 LINK/BLOCK 漂移与数字覆盖都已是 0，也就是说
「中文的链接多重集 / 数字多重集 == 当前英文」是**既成事实**。
所以 `@UUID`、`[[/…]]`、数字这三种锚点在这里一律会显示「跟新」，不含信息量。
剩下的 stale 只可能是**纯散文**层面的，只能人读。

用法同 drift_aligned.py，多两个参数：
  --sim 0.88     相似度阈值，高于此视作「微改」压成一行
  --show-del     是否打印被删块的首 80 字符（默认只计数）
"""
from __future__ import annotations
import argparse, json, os, re, sys, difflib

sys.stdout.reconfigure(encoding='utf-8')

BLOCK_RE = re.compile(r'<(p|li|h[1-6]|td|th|figcaption|blockquote)\b[^>]*>(.*?)</\1>', re.S | re.I)
TAG = re.compile(r'<[^>]+>')


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


def baseline_packs(bdir):
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        key = ('ember.crucible-adventure.json' if f == '_repaired.json'
               else f.replace('-en.json', '.json'))
        out.setdefault(key, os.path.join(bdir, f))
    return out


def blocks(html):
    if not html:
        return []
    out = []
    for m in BLOCK_RE.finditer(html):
        txt = TAG.sub('', m.group(2))
        out.append(re.sub(r'\s+', ' ', txt).strip())
    if not out:
        out.append(re.sub(r'\s+', ' ', TAG.sub('', html)).strip())
    return out


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def worddiff(a, b, cap=200):
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, [w.lower().strip('.,;:—-') for w in aw],
                                 [w.lower().strip('.,;:—-') for w in bw])
    bits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        o = ' '.join(aw[i1:i2]); nn = ' '.join(bw[j1:j2])
        bits.append(f'«{o}»→«{nn}»')
    s = ' '.join(bits)
    return s if len(s) <= cap else s[:cap] + ' …'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=51)
    ap.add_argument('--max-block', type=int, default=300)
    ap.add_argument('--cn-width', type=int, default=220)
    ap.add_argument('--sim', type=float, default=0.88)
    ap.add_argument('--show-del', action='store_true')
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'][a.start:a.start + a.limit]
    old_map = baseline_packs(a.baseline)
    cache = {}

    def pack_leaves(pack):
        if pack in cache:
            return cache[pack]
        o, n, c = {}, {}, {}
        if pack in old_map:
            leaves(load_json(old_map[pack]).get('entries', {}), [], o)
        pe = os.path.join(a.repo, 'compendium', 'en', pack)
        pc = os.path.join(a.repo, 'compendium', 'cn', pack)
        if os.path.exists(pe):
            leaves(load_json(pe).get('entries', {}), [], n)
        if os.path.exists(pc):
            leaves(load_json(pc).get('entries', {}), [], c)
        cache[pack] = (o, n, c)
        return cache[pack]

    def clip(s, w):
        return s if len(s) <= w else s[:w] + ' …'

    for i, it in enumerate(items, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        oe, ne, cn = o.get(p) or '', n.get(p) or '', c.get(p) or ''
        ob, nb, cb = blocks(oe), blocks(ne), blocks(cn)
        print('\n' + '=' * 96)
        print(f'[{i}] {p.split(".journals.")[-1]}  块 旧{len(ob)}/新{len(nb)}/CN{len(cb)}'
              f'  chars {it["en_len_old"]}→{it["en_len_new"]} cn {it["cn_len"]}')
        if len(nb) != len(cb):
            print('  !! CN 块数与新英文不等，位置对齐不可用')

        sm = difflib.SequenceMatcher(None, [norm(x) for x in ob], [norm(x) for x in nb])
        ndel = 0
        dels = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            pairs = min(i2 - i1, j2 - j1) if tag == 'replace' else 0
            for k in range(j1, j2):
                oldtxt = ob[i1 + (k - j1)] if (tag == 'replace' and (k - j1) < pairs) else None
                cnt = cb[k] if k < len(cb) else '(缺)'
                if oldtxt is not None:
                    r = difflib.SequenceMatcher(None, norm(oldtxt), norm(nb[k])).ratio()
                    if r >= a.sim:
                        print(f'  #{k} 微改 {worddiff(oldtxt, nb[k])}')
                        print(f'      CN {clip(cnt, 110)}')
                        continue
                    print(f'  #{k} 改写(sim {r:.2f})')
                    print(f'      OLD {clip(oldtxt, a.max_block)}')
                    print(f'      NEW {clip(nb[k], a.max_block)}')
                    print(f'      CN  {clip(cnt, a.cn_width)}')
                else:
                    print(f'  #{k} 新增')
                    print(f'      NEW {clip(nb[k], a.max_block)}')
                    print(f'      CN  {clip(cnt, a.cn_width)}')
            extra = ob[i1 + (j2 - j1):i2] if tag == 'replace' else (ob[i1:i2] if tag == 'delete' else [])
            for x in extra:
                ndel += 1
                dels.append(x)
        if ndel:
            print(f'  [上游删除 {ndel} 块]')
            if a.show_del:
                for x in dels:
                    print(f'      -DEL {clip(x, 90)}')


if __name__ == '__main__':
    main()
