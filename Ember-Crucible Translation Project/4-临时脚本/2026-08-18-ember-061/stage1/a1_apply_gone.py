# -*- coding: utf-8 -*-
"""A1: 删除桶落地 —— 迁移 16 叶 + 真删 298 叶。
迁移目标值取自 4-临时脚本/2026-08-18-ember-061/terms/GLOSSARY-061.md 的既定裁决。
所有被删的 CN 值先归档到 deleted_cn_archive.json（一个字都不丢）。
"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

BOTH = 'BOTH'
# (src, dst, 新 CN 值 或 None=原样搬)  —— None 表示 EN 未变、译文照搬
MIG = [
    (BOTH, '/entries/Ember Early Access/items/Amberal Bronze/name',
           '/entries/Ember Early Access/items/Ambral Bronze/name',
           '安布拉尔青铜 Ambral Bronze'),          # 拼写修正；GLOSSARY-061 §主表
    (BOTH, '/entries/Ember Early Access/scenes/Amerasp Grove/levels/Amerasp Grove',
           '/entries/Ember Early Access/scenes/Bloodwoods/levels/Amerasp Grove',
           None),                                  # 独立场景折进血林场景当层级
    (BOTH, '/entries/Ember Early Access/scenes/Vista: Arcturel/levels/Rock Bottom',
           '/entries/Ember Early Access/scenes/Vista: Arcturel/levels/Rock Bottom Causeway',
           '石底镇堤道'),
    (BOTH, '/entries/Ember Early Access/scenes/Vista: Sinkhole Depths/levels/Rock Bottom',
           '/entries/Ember Early Access/scenes/Vista: Sinkhole Depths/levels/Rock Bottom Slum',
           '石底镇贫民窟'),
    (BOTH, '/entries/Ember Early Access/scenes/Vista: Skybrush/levels/Skybrush - Cheery',
           '/entries/Ember Early Access/scenes/Vista: Skybrush/levels/Cheery',
           '欢欣'),
    (BOTH, '/entries/Ember Early Access/scenes/Vista: Skybrush/levels/Skybrush - Gloomy',
           '/entries/Ember Early Access/scenes/Vista: Skybrush/levels/Gloom',
           '阴郁'),
    ('ember.crucible-adventure.json',
           "/entries/Ember Early Access/actors/Edivel Sprout/items/Edivel's Book of Spells/description/private",
           "/entries/Ember Early Access/actors/Edivel Sprout/items/Edivel's Book of Spells/description/public",
           None),
    ('ember.crucible-adventure.json',
           '/entries/Ember Early Access/items/A Farewell Note/description/private',
           '/entries/Ember Early Access/items/A Farewell Note/description/public',
           None),
    ('ember.crucible-adventure.json',
           '/entries/Ember Early Access/items/Petalzon/description/private',
           '/entries/Ember Early Access/items/Petalzon/description/public',
           None),
    ('ember.crucible-adventure.json',
           '/entries/Ember Early Access/items/Potion of Climbing/description',
           '/entries/Ember Early Access/items/Potion of Climbing/description/public',
           '<blockquote><p>这瓶药水分成棕色、银色和灰色的层次，如同石头的条带一般。摇晃瓶子也无法将这些颜色混合起来。</p>'
           '</blockquote><p>当你饮下这瓶药水时，你暂时获得以下效果：</p>'),
]

delta = load_delta()
mig_by_file = collections.defaultdict(list)
for scope, src, dst, val in MIG:
    for f in (FILES if scope == BOTH else [scope]):
        if src in delta[f]['gone']:
            mig_by_file[f].append((src, dst, val))

archive = {}
summary = {}
for f in FILES:
    gone = delta[f]['gone']
    if not gone:
        continue
    path = os.path.join(CN, f)
    cn = jload(path)
    en061 = jload(os.path.join(EN061, f))
    migs = mig_by_file[f]
    migsrc = {m[0] for m in migs}
    arc = {}

    # 1) 迁移
    for src, dst, val in migs:
        sp, dp = split_path(src), split_path(dst)
        old = get_at(cn, sp)
        assert old is not None, f"迁移源在 CN 里没有: {src}"
        assert has_leaf(en061, dp), f"迁移目标在 EN061 里不是叶: {dst}"
        new = old if val is None else val
        assert not has_at(cn, dp), f"迁移目标 CN 已存在: {dst}"
        # ⚠ 必须先删源、再写目标：`items/X/description` → `items/X/description/public`
        #   这种「目标是源的后代」的迁移，先写后删会被 del_at 连着新写的目标一起删掉
        #   （2026-08-18 实测踩到：Potion of Climbing 的译文被当场抹掉，靠 v1_verify 的
        #   「多了几叶」自证抓出来）。
        del_at(cn, sp)
        set_at(cn, dp, new)
        assert get_at(cn, dp) == new, f"迁移目标写入后不见了: {dst}"
        assert not has_leaf(cn, sp), f"迁移源没删干净: {src}"
        arc['MIGRATED ' + src] = {"to": dst, "cn_old": old, "cn_new": new}

    # 2) 真删
    ndel = 0
    for g in gone:
        if g in migsrc:
            continue
        gp = split_path(g)
        v = get_at(cn, gp)
        if v is None:
            continue
        arc['DELETED ' + g] = v
        assert del_at(cn, gp), g
        ndel += 1

    save_cn(cn, path)
    archive[f] = arc
    summary[f] = (len(migs), ndel)
    print(f"{f}: 迁移 {len(migs)} / 真删 {ndel}")

jdump(archive, os.path.join(WORK, 'deleted_cn_archive.json'))
print("合计 迁移", sum(s[0] for s in summary.values()), "/ 真删", sum(s[1] for s in summary.values()))
