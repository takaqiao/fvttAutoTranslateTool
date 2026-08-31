# -*- coding: utf-8 -*-
"""风险核：本轮那些**给了图层专属含义**的单词键，会不会同时出现在别的图层上。

`TOKEN_MAKER_PARTS` 是一张**扁平**表，键是显示名 —— 一个键只能有一个值。
所以「Bone 在 pauldron 层译成骨肩甲」这种译法，只有在 `Bone` **只出现在 pauldron** 时才成立。
出现在两个图层就是硬伤：另一个图层会显示错的中文。
"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')
U = json.load(io.open('recon/universe.json', encoding='utf-8'))
segs = U['segs']
rows = [l.rstrip('\n').split('\t') for l in io.open('review.tsv', encoding='utf-8')]
# 只看「拿到了图层专属含义」的：本轮译文里带图层中心词（肩甲/袖/裤/盔/腰饰/披肩…）
MARK = ('肩甲', '肩饰', '袖', '裤', '盔', '腰饰', '披肩', '覆体', '石质', '臂刃')
risky = [(fam, seg, cn) for fam, seg, cn in rows if any(k in cn for k in MARK)]
bad = []
for fam, seg, cn in risky:
    layers = segs.get(seg, [])
    # `pantsL` / `pantsR` 是**同一个逻辑图层**的左右两片，不算「出现在两个图层」——
    # 第一版没归一，82 条里 81 条是这种假警报。
    fams = sorted({l[:-1] if l.endswith(('L', 'R')) and l[:-1] else l for l in layers})
    if len(fams) > 1:
        bad.append((seg, cn, fams))
print(f'带图层专属含义的译文 {len(risky)} 条；其中该段同时出现在多个图层的：{len(bad)}')
for seg, cn, layers in bad:
    print(f'   {seg:<28}{cn:<16}{layers}')
