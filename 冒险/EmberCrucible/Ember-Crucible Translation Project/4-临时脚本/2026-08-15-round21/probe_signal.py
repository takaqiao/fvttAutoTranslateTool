#!/usr/bin/env python3
r"""纳入之后，这 16966 字符到底给判据喂进去了多少**可判的信号**？

本脚本回答三个问题（不回答就等于不知道自己修了什么）：
  1. EN readaloud 值里有多少个阿拉伯数字？—— 决定数字闸能不能咬到它。
  2. EN readaloud 值里命中多少个 glossary_ec 定译专名？—— 决定 --with-terms 能不能咬到。
  3. 逐段的中英字符比是多少？—— 拿来判断有没有整段没译的。
"""
import json
import os
import re
import sys

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
sys.path.insert(0, os.path.join(P, '3-常用脚本', 'qa'))
import scan_content_coverage as S   # noqa: E402

REPOS = [os.path.join(P, '1-Ember汉化插件'), os.path.join(P, '2-Crucible汉化插件')]
ENR_ONE = re.compile(r'(?:readaloud|label)\s*=\s*"([^"]*)"')
RA = re.compile(r'readaloud\s*=\s*"([^"]*)"')

gloss = {}
gp = os.path.join(P, '5-其他内容', 'glossary', 'glossary_ec.json')
for k, v in json.load(open(gp, encoding='utf-8')).items():
    zh = v.split(' ')[0].strip() if isinstance(v, str) else ''
    if len(k) >= 5 and len(zh) >= 2 and S.CJK.search(zh) and not re.search(r'[A-Za-z]', zh):
        gloss[k] = zh
print(f'词表锚点（与 scan_content_coverage 同口径）: {len(gloss)} 条')

pairs = []
for repo in REPOS:
    en_dir = os.path.join(repo, 'compendium', 'en')
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    for pack in sorted(os.listdir(en_dir)):
        if not pack.endswith('.json') or not os.path.exists(os.path.join(cn_dir, pack)):
            continue
        o = []
        S.walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {}),
               json.load(open(os.path.join(cn_dir, pack), encoding='utf-8')).get('entries', {}),
               [], o)
        for path, e, c in o:
            if not c or 'readaloud' not in e:
                continue
            en_ra, cn_ra = RA.findall(e), RA.findall(c)
            for i, t in enumerate(en_ra):
                ct = cn_ra[i] if i < len(cn_ra) else ''
                pairs.append((pack, path, t, ct))

print(f'readaloud 段对: {len(pairs)}  (EN {sum(len(t) for _,_,t,_ in pairs)} 字 / '
      f'CN {sum(len(c) for _,_,_,c in pairs)} 字)')

with_num = [p for p in pairs if re.search(r'(?<!\d)\d+(?!\d)', p[2])]
print(f'1. EN 段里含阿拉伯数字的: {len(with_num)} 段')
for p in with_num[:10]:
    print('   ', re.findall(r'(?<!\d)\d+(?!\d)', p[2]), p[1][-60:])

hits = 0
det = []
for pack, path, t, ct in pairs:
    ms = [k for k in gloss if re.search(r'\b' + re.escape(k) + r'\b', t)]
    if ms:
        hits += 1
        miss = [f'{k}->{gloss[k]}' for k in ms if gloss[k] not in ct]
        if miss:
            det.append((path, miss[:5]))
print(f'2. EN 段命中词表锚点的: {hits} 段；其中中文没有对应定译的: {len(det)} 段')
for d in det[:15]:
    print('   ', d[1], d[0][-60:])

print('3. 逐段中英比（低位 10 段）:')
r = sorted(((len(c) / max(len(t), 1)), len(t), len(c), path)
           for pack, path, t, c in pairs)
for x in r[:10]:
    print(f'   {x[0]:.3f}  EN {x[1]:4d} / CN {x[2]:4d}  {x[3][-64:]}')
tot_e = sum(len(t) for _, _, t, _ in pairs)
tot_c = sum(len(c) for _, _, _, c in pairs)
print(f'   合计中英比 {tot_c / tot_e:.3f}')
print(f'   CN 为空的段: {sum(1 for _,_,_,c in pairs if not c.strip())} 段')
print(f'   CN 里没有汉字的段: {sum(1 for _,_,_,c in pairs if not S.CJK.search(c))} 段')
