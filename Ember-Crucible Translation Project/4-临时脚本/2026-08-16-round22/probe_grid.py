#!/usr/bin/env python3
"""阈值网格：干净库上的假阳性 × 「删一半」时的抓获率。落盘再跑，别用 heredoc。"""
import json
import math
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
CJK = re.compile(r'[一-鿿]')
EN_SENT = re.compile(r'[.!?](?:\s|$)')
CN_SENT = re.compile(r'[。！？；]')

rows = json.load(open(os.path.join(HERE, 'readaloud_probe.json'), encoding='utf-8'))['rows']
gp = os.path.join(os.path.dirname(os.path.dirname(HERE)), '5-其他内容', 'glossary', 'glossary_ec.json')
gloss = {}
for k, v in json.load(open(gp, encoding='utf-8')).items():
    zh = v.split(' ')[0].strip() if isinstance(v, str) else ''
    if len(k) >= 5 and len(zh) >= 2 and CJK.search(zh) and not re.search(r'[A-Za-z]', zh):
        gloss[k] = zh

for r in rows:
    r['en_s'] = len(EN_SENT.findall(r['en']))
    r['cn_s'] = len(CN_SENT.findall(r['cn']))

print('EN 句数分布:', sorted({r['en_s'] for r in rows}))
print('句数比取值:', sorted({round(r['cn_s'] / max(r['en_s'], 1), 2) for r in rows}))
print('readaloud 值里含 HTML 标签的段:', sum(1 for r in rows if '<' in r['en']))
print('readaloud 值里含 @ 增强器的段:', sum(1 for r in rows if '@' in r['en']))


def half(c):
    """⚠ **真的删一半**：按字符取前 50%。

    第一版按句切、`parts[:max(1, n//2)]`，对**单句段**（EN 句数=1 的 10 段）等于原样返回 ——
    「半比 = 原比」，测的是个寂寞。这就是探针假绿的第 (a) 形态，落盘跑才看得出来。
    """
    return c[:max(1, len(c) // 2)]


B = chr(92) + 'b'      # ⚠ 不经改写脚本传 \b（会被吃成退格符）
ANCH = {}
for _i, _r in enumerate(rows):
    ANCH[_i] = [v for k, v in gloss.items() if re.search(B + re.escape(k) + B, _r['en'])]
print('每段锚点已预算：命中段', sum(1 for v in ANCH.values() if v),
      '· 总命中', sum(len(v) for v in ANCH.values()))


def judge(i, cn, thr, f):
    """返回 (比值响, 句数响, 锚点响)"""
    r = rows[i]
    v_ratio = len(cn) / max(len(r['en']), 1) < thr
    v_sent = len(CN_SENT.findall(cn)) < math.floor(r['en_s'] * f)
    v_anchor = any(v not in cn for v in ANCH[i])
    return v_ratio, v_sent, v_anchor


print(f'\n{"thr":>5} {"f":>5} | 干净假阳(比/句/锚/任一) | 删一半抓获(比/句/锚/任一) | 全删抓获')
for thr in (0.15, 0.18, 0.20, 0.22):
    for f in (0.5, 0.6, 0.75, 1.0):
        cl = [0, 0, 0, 0]
        hf = [0, 0, 0, 0]
        fu = 0
        for idx, r in enumerate(rows):
            a = judge(idx, r['cn'], thr, f)
            for j in range(3):
                cl[j] += a[j]
            cl[3] += any(a)
            b = judge(idx, half(r['cn']), thr, f)
            for j in range(3):
                hf[j] += b[j]
            hf[3] += any(b)
            c = judge(idx, '', thr, f)
            fu += any(c)
        print(f'{thr:>5} {f:>5} | {cl[0]:2d}/{cl[1]:2d}/{cl[2]:2d}/{cl[3]:2d}'
              f'              | {hf[0]:2d}/{hf[1]:2d}/{hf[2]:2d}/{hf[3]:2d}'
              f'                | {fu:2d}/48')

# 删一半后仍然全部信号都不响的段，逐条列出来（要在报告里说清楚为什么）
print('\n阈值 0.20 / 系数 0.6（floor）下，删一半仍不响的段:')
for i, r in enumerate(rows):
    b = judge(i, half(r['cn']), 0.20, 0.6)
    if not any(b):
        h = half(r['cn'])
        print(f"  EN句{r['en_s']} CN句{r['cn_s']} 原比{r['ratio']:.3f} "
              f"半比{len(h)/len(r['en']):.3f} 半句{len(CN_SENT.findall(h))} 锚{r['anchor_hits']} "
              f"{r['path'][-50:]}")
