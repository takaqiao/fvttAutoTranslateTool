# -*- coding: utf-8 -*-
r"""
scan_dotted_key_ambiguity.py —— 「同一处变换套在异质集合上、不按成员判身份」的
**路径解析**形态：批次格式用点号拼路径，而条目名本身可以含点，于是同一个路径串
可能对应多个真实位置；四个工具各用各的猜法。

四种解析（都在库里活着）：
  A qa/apply_translations.py:split_path  —— 先试朴素 split('.')，**能解析出非 None 就用它**；
                                            否则再按「最长键优先」走一遍
  B qa/unify_terms.py:187-198            —— 直接「最长键优先」，没有朴素优先这一档
  C qa/port_orphans.py:get_at / fill_missing.walk / prune_dead.leaves
                                         —— 纯朴素 split('.')，含点键一律被切碎
  D tm/fill_missing.to_batch_path        —— 把含点键原样拼进批次 key（无转义）

本探针只回答两个可证伪的问题：
  Q1 库里到底有多少个**含点的键**？（＝批次格式里天然有歧义的键）
  Q2 有没有**真歧义**：同一个父节点下既有键 `X` 又有键 `X.Y`（或 `X` 下有子键 `Y`），
     使得路径串 `X.Y....` 的两种解析都能落到实节点？有 -> A 与 B 会写到不同地方。

假阳性模式：
  * Q1 只是暴露面，不含点键的路径没有歧义；
  * Q2 判定的是「两种解析都能落到实节点」，不代表批次里真出现过这条路径。
只读。
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield path, k, node
            yield from walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, path + (str(i),))


def main():
    dotted = Counter()
    dotted_samples = []
    ambiguous = []
    files = 0
    for repo in REPOS:
        for side in ('en', 'cn'):
            d = os.path.join(P, repo, 'compendium', side)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith('.json') or fn.startswith('_'):
                    continue
                files += 1
                doc = load(os.path.join(d, fn))
                for parent_path, key, parent in walk(doc):
                    if '.' not in key:
                        continue
                    dotted[f'{repo}/{side}/{fn}'] += 1
                    if len(dotted_samples) < 25:
                        dotted_samples.append(f'{fn} :: {".".join(parent_path + (key,))}')
                    # 真歧义：同一父节点里存在前缀键，且沿前缀走下去也能落到实节点
                    head, _, tail = key.partition('.')
                    while head:
                        if head in parent and isinstance(parent[head], (dict, list)):
                            node = parent[head]
                            ok = True
                            for seg in tail.split('.'):
                                if isinstance(node, dict) and seg in node:
                                    node = node[seg]
                                elif isinstance(node, list) and seg.isdigit() and int(seg) < len(node):
                                    node = node[int(seg)]
                                else:
                                    ok = False
                                    break
                            if ok:
                                ambiguous.append({'file': f'{repo}/{side}/{fn}',
                                                  'parent': '.'.join(parent_path),
                                                  'key': key, 'prefix': head})
                        nh, _, nt = tail.partition('.')
                        if not nt and nh:
                            head, tail = f'{head}.{nh}', ''
                            break
                        head, tail = f'{head}.{nh}', nt
                        if not nt:
                            break
    print(f'扫了 {files} 个 json（两仓 en+cn）')
    print(f'Q1 含点的键：{sum(dotted.values())} 个，分布 {len(dotted)} 个文件')
    for k, v in dotted.most_common(10):
        print(f'      {v:>4}  {k}')
    print('    样例：')
    for s in dotted_samples[:12]:
        print(f'      {s}')
    print(f'\nQ2 真歧义（同父节点下既有 `X.Y` 键、又能沿 `X`->`Y` 走通）：{len(ambiguous)}')
    for a in ambiguous[:20]:
        print(f'      {a}')


main()
