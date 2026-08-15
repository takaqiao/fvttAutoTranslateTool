#!/usr/bin/env python3
"""G1 对抗性复核：独立重做「HTML 属性里的可见文本」的普查，用来验 qa/scan_attr_text.py 的判据。

不复用 scan_attr_text 的任何正则/名单，故意用更松的抓取方式，看它有没有漏掉定义域。

  python g1_probe.py --repo <repo> [--repo <repo2>] --out <json>

四件事：
  A. 裸普查：任意字符串叶子里所有 name="value" / name='value' / name=bare，
     不管在不在标签内，EN 与 CN 分别计数 -> 能看出 scan_attr_text 的 TAG/EMBED/INLINE
     三个正则是不是覆盖了全部出现位置。
  B. 叶级配对：同一 path 的 EN/CN 叶，把可见属性按出现顺序对齐，
     报「CN 值与 EN 值逐字节相同且含普通英文词」= 真·没译（不依赖词表分类）。
  C. 静默丢失：EN 叶里有的可见属性，CN 叶里整个不见了（scan_attr_text 的
     attr_name_drift 只看 origin=='html'，@Embed 体内的 label/readaloud 丢了看不见）。
  D. 半译：CN 值里既有中文又有连续 >=3 个普通英文词。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 故意比 scan_attr_text 松：不要求在标签内
RAW_ATTR = re.compile(r'''(?<![\w:-])([A-Za-z_][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'<>\]]+))''')
CJK = re.compile(r'[㐀-䶿一-鿿豈-﫿]')
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
LOWER_TOKEN = re.compile(r"\b[a-z][a-z'’-]{1,}\b")

VISIBLE = {'data-tooltip', 'data-tooltip-text', 'data-tooltip-html', 'label',
           'readaloud', 'title', 'alt', 'aria-label', 'placeholder', 'data-label',
           'caption', 'summary'}
# 内联命令体 [[/... ]] 内的属性照抄，不算
INLINE_CMD = re.compile(r'\[\[/(?:[^\]"]|"[^"]*")*\]\]')


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
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith('.json') and not f.startswith('_'))


def mask_inline(text):
    """把内联命令体挖空（用等长空格），这样其中的属性不会被抓到，偏移量还保持。"""
    out = list(text)
    for m in INLINE_CMD.finditer(text):
        for i in range(m.start(), m.end()):
            out[i] = ' '
    return ''.join(out)


def attrs_of(text, skip_inline=True):
    src = mask_inline(text) if skip_inline else text
    res = []
    for m in RAW_ATTR.finditer(src):
        v = m.group(2)
        if v is None:
            v = m.group(3)
        if v is None:
            v = m.group(4)
        res.append((m.group(1).lower(), v, m.start()))
    return res


def build_vocab(repos, vocab_min=3):
    c = Counter()
    for repo in repos:
        d = os.path.join(repo, 'compendium', 'en')
        for f in packs(d):
            leaves = []
            walk(load(os.path.join(d, f)), [], leaves)
            for _, s in leaves:
                c.update(LOWER_TOKEN.findall(s))
    return {w for w, n in c.items() if n >= vocab_min}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()

    vocab = build_vocab(a.repo)

    en_names, cn_names = Counter(), Counter()
    en_vis, cn_vis = Counter(), Counter()
    identical, dropped, partial, cn_extra = [], [], [], []
    cn_vis_samples = defaultdict(list)

    for repo in a.repo:
        tag = os.path.basename(os.path.normpath(repo))
        en_d = os.path.join(repo, 'compendium', 'en')
        cn_d = os.path.join(repo, 'compendium', 'cn')

        for f in packs(en_d):
            leaves = []
            walk(load(os.path.join(en_d, f)), [], leaves)
            for _, s in leaves:
                for n, v, _o in attrs_of(s):
                    en_names[n] += 1
                    if n in VISIBLE:
                        en_vis[n] += 1

        for f in packs(cn_d):
            cn_leaves = []
            walk(load(os.path.join(cn_d, f)), [], cn_leaves)
            en_map = {}
            if os.path.exists(os.path.join(en_d, f)):
                t = []
                walk(load(os.path.join(en_d, f)), [], t)
                en_map = dict(t)
            for path, s in cn_leaves:
                ca = attrs_of(s)
                for n, v, _o in ca:
                    cn_names[n] += 1
                    if n in VISIBLE:
                        cn_vis[n] += 1
                        if len(cn_vis_samples[n]) < 400:
                            cn_vis_samples[n].append(v)
                e = en_map.get(path)
                if not isinstance(e, str):
                    continue
                ea = attrs_of(e)
                # 按属性名分组、按出现顺序对齐
                eg, cg = defaultdict(list), defaultdict(list)
                for n, v, _o in ea:
                    if n in VISIBLE:
                        eg[n].append(v)
                for n, v, _o in ca:
                    if n in VISIBLE:
                        cg[n].append(v)
                for n in set(eg) | set(cg):
                    el, cl = eg.get(n, []), cg.get(n, [])
                    if len(cl) < len(el):
                        dropped.append({'repo': tag, 'pack': f, 'path': path, 'attr': n,
                                        'en_count': len(el), 'cn_count': len(cl),
                                        'en_values': el, 'cn_values': cl})
                    elif len(cl) > len(el):
                        cn_extra.append({'repo': tag, 'pack': f, 'path': path, 'attr': n,
                                         'en_count': len(el), 'cn_count': len(cl),
                                         'en_values': el, 'cn_values': cl})
                    for ev, cv in zip(el, cl):
                        if ev == cv and not CJK.search(cv):
                            ws = WORD.findall(cv)
                            ord_ = [w for w in ws if w.lower() in vocab]
                            if ord_:
                                identical.append({'repo': tag, 'pack': f, 'path': path,
                                                  'attr': n, 'value': cv,
                                                  'ordinary_words': ord_})
                        elif CJK.search(cv):
                            # 半译：中文里还剩连续 >=3 个普通英文词
                            for run in re.findall(r"(?:[A-Za-z][A-Za-z'’-]*[ ,]+){2,}[A-Za-z][A-Za-z'’-]*", cv):
                                toks = [w for w in WORD.findall(run) if w.lower() in vocab]
                                if len(toks) >= 3:
                                    partial.append({'repo': tag, 'pack': f, 'path': path,
                                                    'attr': n, 'value': cv, 'run': run})
                                    break

    print('== A. 裸普查：EN 侧全部属性名（不限标签内） ==')
    for n, c in en_names.most_common():
        mark = ' <== VISIBLE' if n in VISIBLE else ''
        print(f'  {c:>6}  {n}{mark}')
    print(f'  属性名总数 EN={len(en_names)}  CN={len(cn_names)}')
    print()
    print('== CN 侧独有的属性名（EN 没有的） ==')
    for n in sorted(set(cn_names) - set(en_names)):
        print(f'  {cn_names[n]:>6}  {n}')
    print('== EN 侧独有的属性名（CN 没有的） ==')
    for n in sorted(set(en_names) - set(cn_names)):
        print(f'  {en_names[n]:>6}  {n}')
    print()
    print('== 可见属性 EN vs CN 计数 ==')
    for n in sorted(set(en_vis) | set(cn_vis)):
        d = cn_vis[n] - en_vis[n]
        print(f'  {n:20} EN={en_vis[n]:>5}  CN={cn_vis[n]:>5}  差={d:+d}')
    print()
    print(f'== B. CN 与 EN 逐字节相同且含普通英文词：{len(identical)} 处 ==')
    for r in identical[:60]:
        print(f'  [{r["repo"]}/{r["pack"]}] {r["attr"]}={r["value"]!r}')
        print(f'       {r["path"][:140]}')
    print()
    print(f'== C. CN 叶里整个丢失的可见属性：{len(dropped)} 处 ==')
    for r in dropped[:60]:
        print(f'  [{r["repo"]}/{r["pack"]}] {r["attr"]} EN={r["en_count"]} CN={r["cn_count"]}')
        print(f'       {r["path"][:140]}')
        print(f'       EN值={r["en_values"]}')
    print(f'== C2. CN 叶里多出来的可见属性：{len(cn_extra)} 处 ==')
    for r in cn_extra[:30]:
        print(f'  [{r["repo"]}/{r["pack"]}] {r["attr"]} EN={r["en_count"]} CN={r["cn_count"]}')
        print(f'       {r["path"][:140]}')
        print(f'       CN值={r["cn_values"]}')
    print()
    print(f'== D. 半译（中文里仍有连续 >=3 个普通英文词）：{len(partial)} 处 ==')
    for r in partial[:60]:
        print(f'  [{r["repo"]}/{r["pack"]}] {r["attr"]}={r["value"]!r}')
        print(f'       {r["path"][:140]}')

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'en_attr_names': dict(en_names), 'cn_attr_names': dict(cn_names),
                       'en_visible': dict(en_vis), 'cn_visible': dict(cn_vis),
                       'identical': identical, 'dropped': dropped, 'cn_extra': cn_extra,
                       'partial': partial,
                       'cn_visible_samples': {k: v for k, v in cn_vis_samples.items()}},
                      fh, ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
