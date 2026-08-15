#!/usr/bin/env python3
"""在**已有中文**的叶子里找残留的整段英文散文（属性扫描的姊妹判据）。

5.4 的第 4 项只查西里尔/亚美尼亚等外来文字，**英文残留没有任何判据** ——
而英文残留恰恰是最容易发生的（漏译一个从句、一个 <dd> 项）。

判据：把标记全部挖空（HTML 标签体、@Enricher[...]、[[/cmd]]、&Ref[...]、
{标签} 里的内容保留，因为标签是要译的），在剩下的可见文本里找
「连续 >=N 个普通英文词」的段落，且该叶整体含中文（纯英文叶属于「整条没译」，
由 fill_missing 覆盖，不在这里报）。

  python g1_en_residue.py --repo <r> [--repo <r2>] [--min-words 5] --out <json>
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CJK = re.compile(r'[一-鿿]')
LOWER_TOKEN = re.compile(r"\b[a-z][a-z'’-]{1,}\b")
TAG = re.compile(r'<[^>]*>')
ENRICH_BODY = re.compile(r'(@[A-Za-z]+|&(?:amp;)?[A-Za-z]+)\[(?:[^\]"]|"[^"]*")*\]')
INLINE = re.compile(r'\[\[[^\]]*\]\]')
ENTITY = re.compile(r'&[a-zA-Z#0-9]+;')
RUN = re.compile(r"[A-Za-z][A-Za-z'’]*(?:[ ,;:'’\-]+[A-Za-z][A-Za-z'’]*)+")


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


def load(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh).get('entries', {})


def packs(d):
    return sorted(f for f in os.listdir(d) if f.endswith('.json') and not f.startswith('_')) if os.path.isdir(d) else []


def blank(rx, t):
    return rx.sub(lambda m: ' ' * len(m.group(0)), t)


def visible_text(s):
    s = blank(INLINE, s)
    s = blank(ENRICH_BODY, s)   # 只挖 [...] 体，{标签} 留着（标签本来该译）
    s = blank(TAG, s)
    s = blank(ENTITY, s)
    return s


def build_vocab(repos, vocab_min=3):
    c = Counter()
    for repo in repos:
        d = os.path.join(repo, 'compendium', 'en')
        for f in packs(d):
            L = []
            walk(load(os.path.join(d, f)), [], L)
            for _, t in L:
                c.update(LOWER_TOKEN.findall(t))
    return {w for w, n in c.items() if n >= vocab_min}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--min-words', type=int, default=5)
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=80)
    a = ap.parse_args()
    vocab = build_vocab(a.repo)
    hits = []
    for repo in a.repo:
        tag = os.path.basename(os.path.normpath(repo))
        cn = os.path.join(repo, 'compendium', 'cn')
        en = os.path.join(repo, 'compendium', 'en')
        for f in packs(cn):
            L = []
            walk(load(os.path.join(cn, f)), [], L)
            em = {}
            if os.path.exists(os.path.join(en, f)):
                t = []
                walk(load(os.path.join(en, f)), [], t)
                em = dict(t)
            for path, s in L:
                if not CJK.search(s):
                    continue          # 整条没译 -> fill_missing 的定义域
                vis = visible_text(s)
                for m in RUN.finditer(vis):
                    toks = [w for w in LOWER_TOKEN.findall(m.group(0)) if w in vocab]
                    if len(toks) < a.min_words:
                        continue
                    hits.append({'repo': tag, 'pack': f, 'path': path,
                                 'run': m.group(0).strip(),
                                 'ctx': s[max(0, m.start() - 90):m.end() + 90],
                                 'en_leaf_exists': path in em})
    print(f'含中文的叶子里，残留 >= {a.min_words} 个普通英文词的连续片段：{len(hits)} 处')
    for h in hits[:a.show]:
        print(f'  [{h["repo"]}/{h["pack"]}] {h["path"][:120]}')
        print(f'      RUN: {h["run"][:160]}')
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'min_words': a.min_words, 'hits': hits}, fh, ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
