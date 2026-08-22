# -*- coding: utf-8 -*-
"""把 691 条缺口拆成形态素，看这活到底有多大、有多系统。

拆法与上屏那条正则同口径：`(?<!^)([A-Z1-9])` 前断开。
纯数字段（symbol 那族 10..82）单独拎出来 —— 它们上屏就是「10」，本来就不用翻。
"""
import io, json, re, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
g = json.load(io.open('recon/gap.json', encoding='utf-8'))
segs = sorted({s for v in g['groups'].values() for s in v})
assert len(segs) == g['missing'], (len(segs), g['missing'])

numeric = [s for s in segs if s.isdigit()]
words = [s for s in segs if not s.isdigit()]
print(f"缺口 {len(segs)}：纯数字 {len(numeric)}（{numeric[0]}..{numeric[-1]}，上屏就是数字，不用翻）"
      f" · 需要翻的 {len(words)}")

SPLIT = re.compile(r'(?<!^)(?=[A-Z1-9])')
tok = collections.Counter()
for s in words:
    for t in SPLIT.split(s):
        tok[t] += 1
print(f"\n形态素 {len(tok)} 种，出现 {sum(tok.values())} 次"
      f"（平均每条 {sum(tok.values())/len(words):.1f} 段）")
print("出现 ≥5 次的：", len([t for t,c in tok.items() if c>=5]))
print("只出现 1 次的：", len([t for t,c in tok.items() if c==1]))
print("\n前 60 高频：")
for t, c in tok.most_common(60):
    print(f"   {t:<22}{c}")
io.open('recon/morphemes.json','w',encoding='utf-8').write(
    json.dumps({'numeric':numeric,'words':words,
                'tokens':dict(sorted(tok.items(), key=lambda kv:(-kv[1],kv[0])))},
               ensure_ascii=False, indent=1))
