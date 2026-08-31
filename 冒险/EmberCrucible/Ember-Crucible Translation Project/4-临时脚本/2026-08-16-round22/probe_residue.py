#!/usr/bin/env python3
"""残留探针：上游从旧英文里**删掉/改掉**的关键词，其中文写法是否还留在译文里。

不复写判据 —— 三方文本直接取自 three_way_14.json（由 dump14.py 用 scan_en_drift
自己的 load_json/leaves 生成）。本探针只做一件事：在**中文全文**里数关键词次数。

自证在跑：先对每条打印「中文字数 / 本条检查了几个词」，再打印命中。
另有一组**阳性对照词**（明知一定在中文里），若对照没命中说明探针本身空转。
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {(r['pack'], r['path']): r for r in
        json.load(open(os.path.join(HERE, 'three_way_14.json'), encoding='utf-8'))}

R = 'crucible.rules.json'
E = 'crucible.equipment.json'

# path -> (旧版专有的中文写法[应当为 0], 阳性对照[应当 >0])
CHECK = {
    (E, 'Cloak of Kindly Visage.description.private'):
        (['和善面容', '亲和面容', '恩惠', '轻微', '强效', '传奇', '已注魔', '穿戴者'], ['前缀', '后缀', '示例']),
    (E, 'Common Clothing.description.public'):
        (['护甲', '不能', '无法'], ['衬衫', '靴子']),
    (R, 'Character Mechanics.pages.Defenses.text'):
        (['偏斜', '已偏斜', '擦身一击', '掠过一击', '/ 2)', '/2)'], ['擦击', '/ 4)', '溢出']),
    (R, 'Combat.pages.Engagement and Flanking.text'):
        (['谨慎移动', '小心移动', '脱离交战打击', '脱离打击', '主手武器', '1点专注', '1 点专注', '专注点',
          '主角'], ['reactiveStrike', '英雄', '@Condition[flanked]', '近战']),
    (R, 'Conditions.pages.Broken.text'):
        (['主角'], ['英雄', '破碎']),
    (R, 'Conditions.pages.Incapacitated.text'):
        ([], ['失能']),
    (R, 'Conditions.pages.Stunned.text'):
        (['踉跄', '蹒跚'], ['震慑']),
    (R, 'Conditions.pages.Weakened.text'):
        (['主角'], ['英雄', '虚弱']),
    (R, 'Crafting.pages.Tradeskills Overview.text'):
        (['早期开发', '可能会变更', '可能有所变动', '预览', '尚未完成'], ['积极开发', '预期设计', '自动化']),
    (R, 'Equipment.pages.Weapons.text'):
        # 注：'稀有度修正' 不能当旧词 —— 品质/附魔两张表的表头本来就叫它，新英文里仍在。
        # 上游删掉的是「每种特性 +1 稀有度修正」这句，故只查 '+1 稀有度' 这种搭配。
        (['延伸', '敏锐', '可靠', '+1 稀有度', '+1稀有度', '尚未完全自动化', '很罕见', '在极少数情况下'],
         ['伏击', '缠斗', '直觉', '招架', '格挡', '多用', '投掷', '超大尺寸']),
    (R, 'Spellcraft.pages.Inflections.text'):
        (['试玩测试 1', '试玩测试1', 'Playtest 1', '第一次试玩', '后续阶段'], ['未来版本', '目前仅实现']),
    (R, 'Welcome To Crucible.pages.Module Recommendations.text'):
        (['预购', '众筹', 'Kickstarter', 'Beta', '测试阶段', '巨大'], ['抢先体验', '沉浸式', '开放世界']),
    (R, 'Welcome To Crucible.pages.Providing Feedback.text'):
        (['谷歌', 'Google', '表单', '试玩测试的一部分', '暂不收集'], ['GitHub', 'Discord', '问题追踪器']),
    (R, 'Welcome To Crucible.pages.What is Crucible.text'):
        (['crucible-text-dark', 'Kickstarter', '预购', '众筹', 'Alpha', '阿尔法', '多轮反馈', '还有很长的路'],
         ['banner-full.webp', '抢先体验', '当前试玩测试', '不是最终成品']),
}

assert len(CHECK) == 14, len(CHECK)
tot_words = tot_hits = tot_ctrl_miss = 0
print(f'共 {len(CHECK)} 条叶进探针')
for key, (stale, ctrl) in CHECK.items():
    cn = rows[key]['cn']
    hits = [(w, cn.count(w)) for w in stale if w in cn]
    miss = [w for w in ctrl if w not in cn]
    tot_words += len(stale) + len(ctrl)
    tot_hits += len(hits)
    tot_ctrl_miss += len(miss)
    print(f"· {key[1][:52]:<52} 中文 {len(cn):>5} 字 · 旧词 {len(stale)} 个 · 对照 {len(ctrl)} 个"
          f" · 旧词命中 {hits if hits else '无'}"
          + (f" · ⚠对照未命中 {miss}" if miss else ''))
print(f'合计检查词 {tot_words} 个 · 旧版残留命中 {tot_hits} · 阳性对照未命中 {tot_ctrl_miss}')
print('探针有效性：阳性对照全部命中 =>', tot_ctrl_miss == 0)
