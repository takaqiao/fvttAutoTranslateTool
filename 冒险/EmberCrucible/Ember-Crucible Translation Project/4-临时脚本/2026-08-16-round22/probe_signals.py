#!/usr/bin/env python3
"""摊开候选信号在真库 48 段上的分布：句数、比值、锚点、数字。

目的：在写闸之前先知道每个信号「干净时的富余度」和「删一半时的判别力」。
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
CJK = re.compile(r'[一-鿿]')
EN_SENT = re.compile(r'[.!?](?:\s|$)')
CN_SENT = re.compile(r'[。！？；]')
NUM = re.compile(r'(?<!\d)(\d+(?:\.\d+)?)(?!\d)')

rows = json.load(open(os.path.join(HERE, 'readaloud_probe.json'), encoding='utf-8'))['rows']
print(f'读入 {len(rows)} 段（探针的产物，与库同一次运行；下面每个数都来自这 {len(rows)} 段）')

worst_s = 9e9
for r in rows:
    e, c = r['en'], r['cn']
    r['en_s'] = len(EN_SENT.findall(e))
    r['cn_s'] = len(CN_SENT.findall(c))
    r['s_ratio'] = r['cn_s'] / max(r['en_s'], 1)
    r['en_num'] = len(NUM.findall(e))
    worst_s = min(worst_s, r['s_ratio'])

print('\n句数比 (CN句/EN句) 分布:')
sr = sorted(r['s_ratio'] for r in rows)
print('  min', round(sr[0], 3), 'p25', round(sr[len(sr)//4], 3),
      '中位', round(sr[len(sr)//2], 3), 'max', round(sr[-1], 3))
print('  CN 句数 < EN 句数的段:', sum(1 for r in rows if r['cn_s'] < r['en_s']))
print('  英文含阿拉伯数字的段:', sum(1 for r in rows if r['en_num']))
print('\n最低的 8 段句数比:')
for r in sorted(rows, key=lambda r: r['s_ratio'])[:8]:
    print(f"  {r['s_ratio']:.2f}  EN句{r['en_s']:2d} CN句{r['cn_s']:2d} "
          f"比{r['ratio']:.3f} 锚{r['anchor_hits']} {r['path'][-52:]}")

# 「删一半」= 只保留中文前一半字符（按句边界切）
print('\n删一半（中文只留前半，按句切）后各信号的判别力:')


def half(c):
    parts = re.split(r'(?<=[。！？])', c)
    parts = [p for p in parts if p]
    keep = parts[:max(1, len(parts) // 2)]
    return ''.join(keep)


gp = os.path.join(os.path.dirname(os.path.dirname(HERE)), '5-其他内容', 'glossary', 'glossary_ec.json')
gloss = {}
for k, v in json.load(open(gp, encoding='utf-8')).items():
    zh = v.split(' ')[0].strip() if isinstance(v, str) else ''
    if len(k) >= 5 and len(zh) >= 2 and CJK.search(zh) and not re.search(r'[A-Za-z]', zh):
        gloss[k] = zh

for thr in (0.15, 0.18, 0.20):
    hit_ratio = hit_anchor = hit_sent = hit_any = 0
    for r in rows:
        h = half(r['cn'])
        ok_ratio = len(h) / max(len(r['en']), 1) < thr
        anchors = [(k, v) for k, v in gloss.items()
                   if re.search(r'\b' + re.escape(k) + r'\b', r['en'])]
        ok_anchor = any(v not in h for _, v in anchors)
        ok_sent = len(CN_SENT.findall(h)) < r['en_s'] * 0.6
        hit_ratio += ok_ratio
        hit_anchor += ok_anchor
        hit_sent += ok_sent
        hit_any += (ok_ratio or ok_anchor or ok_sent)
    print(f'  阈值 {thr}: 比值抓 {hit_ratio}/{len(rows)} · 锚点抓 {hit_anchor}/{len(rows)} · '
          f'句数(<0.6EN)抓 {hit_sent}/{len(rows)} · 任一抓 {hit_any}/{len(rows)}')

print('\n干净库上句数闸 (<0.6*EN句) 的假阳性:',
      sum(1 for r in rows if r['cn_s'] < r['en_s'] * 0.6))
for f in (0.5, 0.55, 0.6, 0.7):
    print(f'  系数 {f}: 干净时假阳 {sum(1 for r in rows if r["cn_s"] < r["en_s"] * f)}')
