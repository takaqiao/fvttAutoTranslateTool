#!/usr/bin/env python3
"""专查：中文里还留着阵营缩写 `（NG，` 而当前英文已经改成 `Neutral Good` 全称。

G4 从 `A Conflagration of Lumé` 一条挖出来的模式，回扫全库看还有多少条同型。
顺带核对**字母本身有没有被改掉**（NG 写成 NE 那类）。
"""
import json, os, re, sys

ABB = {'LG': 'Lawful Good', 'NG': 'Neutral Good', 'CG': 'Chaotic Good',
       'LN': 'Lawful Neutral', 'N': 'True Neutral', 'CN': 'Chaotic Neutral',
       'LE': 'Lawful Evil', 'NE': 'Neutral Evil', 'CE': 'Chaotic Evil'}
ZH = {'Lawful Good': '守序善良', 'Neutral Good': '中立善良', 'Chaotic Good': '混乱善良',
      'Lawful Neutral': '守序中立', 'True Neutral': '绝对中立', 'Neutral': '绝对中立',
      'Chaotic Neutral': '混乱中立', 'Lawful Evil': '守序邪恶', 'Neutral Evil': '中立邪恶',
      'Chaotic Evil': '混乱邪恶'}
CNPAT = re.compile(r'（(' + '|'.join(ABB) + r')[，,]')
ENPAT = re.compile(r'\((' + '|'.join(list(ABB) + list(ZH)) + r')[,，]')


def load(p):
    return json.loads(open(p, encoding='utf-8-sig').read())


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out['.'.join(path)] = node


for repo in sys.argv[1:]:
    en_d = os.path.join(repo, 'compendium', 'en')
    cn_d = os.path.join(repo, 'compendium', 'cn')
    for f in sorted(os.listdir(cn_d)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        if not os.path.exists(os.path.join(en_d, f)):
            continue
        cn, en = {}, {}
        leaves(load(os.path.join(cn_d, f)).get('entries', {}), [], cn)
        leaves(load(os.path.join(en_d, f)).get('entries', {}), [], en)
        for path, v in cn.items():
            hits = CNPAT.findall(v)
            if not hits:
                continue
            e = en.get(path, '')
            print(f'{repo[0]} | {f} | {path}')
            print(f'   CN 缩写 {hits}   EN 同位 {ENPAT.findall(e)}')
            for m in CNPAT.finditer(v):
                print('   …', v[max(0, m.start() - 30):m.start() + 30].replace('\n', ' '))
