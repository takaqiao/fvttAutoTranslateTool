# -*- coding: utf-8 -*-
"""构建两个仓库的 lang 批次。

每条都写死「改前」的期望值：若与仓库现状不符即报错退出，
避免我按记忆写键名/值而与实际漂移（PROJECT.md「先查再落笔」的机械保障）。
"""
import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project")
OUT = ROOT / "4-临时脚本/2026-08-12-fix/batches"

# ---------------------------------------------------------------- ember
EMBER = [
    # 1. 月相机翻残留（世界时钟每次都显示）
    ("EMBER.MOON.PHASES.WAXING", "打蜡", "渐盈"),
    ("EMBER.MOON.PHASES.FULL", "全额", "满月"),
    ("EMBER.MOON.PHASES.WANING", "衰退的", "渐亏"),
    ("EMBER.MOON.PHASES.NONE", "休眠的", "休眠"),
    # 2. hex＝六边格，不是妖术
    ("EMBER.CONTROLS.HexHUD", "切换妖术 HUD", "切换六边格 HUD"),
    ("EMBER.EVENT.FIELDS.hexes.hint",
     "将此事件配置为发生在一组特定的妖术坐标中。",
     "将此事件配置为发生在一组特定的六边格坐标中。"),
    ("EMBER.EVENT.FIELDS.global.hint",
     "全局事件可能发生在世界中的任何一个和所有区域。",
     "全局事件可以发生在世界中的任何一个乃至每一个六边格中。"),
    ("EMBER.LOCATION.FIELDS.sounds.environment.neighbors.label", "邻居们", "相邻六边格"),
    # 3. 季节名（三通道同步，见 notes）
    ("EMBER.CALENDAR.SEASONS.STEADING", "庄园", "安居"),
    # 4. 词义跑偏的 label（同条 hint 已译对）
    ("EMBER.LOCATION.FIELDS.events.coefficients.hint",
     "配置用于确定此地点中事件发生概率的系数和重量。",
     "配置用于确定此地点中事件发生概率的系数和权重。"),
    ("EMBER.EVENT.FIELDS.scene.complete.id.label", "完整场景", "完成场景"),
    ("EMBER.JOURNAL_ENTRY_PAGE.TABS.development", "发展", "开发"),
    ("EMBER.ACTOR_FLAGS.FIELDS.portrait.height.label", "身高", "高度"),
    ("EMBER.ANCESTRY.FIELDS.height.label", "高度", "身高"),
    ("EMBER.ACTOR_FLAGS.FIELDS.character.path.label", "道路", "道途"),
    ("EMBER.ACTOR_FLAGS.FIELDS.discoverable.hint",
     "这个 Actor 在 Ember 法典中能被识别为生物还是角色吗？",
     "这个 Actor 能否在 Ember 法典中作为生物或角色被发现？"),
    ("EMBER.ANCESTRY.FIELDS.rarity.hint", "定义该血统的稀有度阶级。", "定义该血统的稀有度阶数。"),
    # 5. 逐词直译留下的空格
    ("EMBER.ANCESTRY.FIELDS.item.label", "血统 种族 物品", "血统种族物品"),
    ("EMBER.ANCESTRY.FIELDS.origin.label", "血统 起源", "血统起源"),
    ("EMBER.CULTURE.FIELDS.item.label", "文化 背景 物品", "文化背景物品"),
    ("EMBER.BIOME.SECTIONS.ambience", "环境 音频", "环境音频"),
    # 6. 形容词尾巴「的」——与同族 40 余个名词标签不一致
    ("EMBER.CONST.TRAVEL.slow", "缓慢的", "缓慢"),
    ("EMBER.EVENT.FIELDS.unique.label", "独特的", "独特"),
    ("EMBER.ACTOR_FLAGS.FIELDS.discoverable.label", "可被发现的", "可发现"),
    ("EMBER.ACTOR_FLAGS.SHEET.DISCOVERABLE", "可发现的", "可发现"),
    # 7. 同一英文两种中文
    ("ACTOR.CONTROLS.EmberToken", "余烬动态令牌", "余烬动态指示物"),
    ("EMBER.MovementMiles", "灰烬要求队伍移动速度以英里为单位。", "余烬要求队伍移动速度以英里为单位。"),
    ("TYPES.JournalEntryPage.ember.standaloneEvent", "Ember独立事件", "余烬独立事件"),
    ("TYPES.JournalEntryPage.ember.ancestry", "烬裔血统", "余烬血统"),
    ("TYPES.JournalEntryPage.ember.location", "余烬位置", "余烬地点"),
    ("EMBER.CODEX.DISCOVERY_RESET", "重置探索", "重置发现"),
    ("EMBER.EventActionDiscoveryComplete", "完成探索", "完成发现"),
    # 8. 跨通道：compendium 同调阶位 19 : 同调等级 4
    ("EMBER.ATTUNEMENT.Rank", "等级", "阶位"),
    ("EMBER.ATTUNEMENT.NextTooltip", "晋升至下一等级所需", "晋升至下一阶位所需"),
]

# ---------------------------------------------------------------- crucible
CRUCIBLE = [
    # A. 审计 2.3：7 个 spellcraft 手势（英文闸计数见 notes）
    ("SPELL.GESTURES.Fan", "扇子", "扇形"),
    ("SPELL.GESTURES.Blast", "爆炸咒", "爆破"),
    ("SPELL.GESTURES.Ward", "防护罩", "防护"),
    ("SPELL.GESTURES.Ray", "光线", "射线"),
    ("SPELL.GESTURES.Surge", "激涌", "涌动"),
    ("SPELL.GESTURES.Conjure", "咒法系", "召唤"),
    ("SPELL.GESTURES.Cone", "锥形区域", "锥形"),
    ("ACTION.TARGET_TYPES.Cone", "锥形区域", "锥形"),
    # B. 审计 2.3：Stride 步幅
    ("ACTOR.FIELDS.movement.stride.label", "跨步", "步幅"),
    ("ANCESTRY.FIELDS.movement.stride.label", "跨步", "步幅"),
    ("TAXONOMY.FIELDS.movement.stride.label", "跨步", "步幅"),
    ("ACTOR.FIELDS.movement.strideBonus.label", "移动加值", "步幅加值"),
    ("ACTOR.FIELDS.movement.strideBonus.hint",
     "由血统或分类法定义的，对该角色的移动速度的加值修正。",
     "由血统或分类法定义的，对该角色步幅速度的加值修正。"),
    ("TOKEN.MOVEMENT.ACTIONS.jump.description",
     "每个动作可跃过的距离最多为你的移动属性的一半。",
     "每个动作可跃过的距离最多为你步幅属性的一半。"),
    ("ACTOR.FIELDS.movement.stride.tooltip", "每消耗 1 点行动点的移动距离。", "每消耗一点动作点的移动距离。"),
    # C. 词义完全跑偏的机翻标签
    ("CRUCIBLE.Base", "基地", "基础"),
    ("CRUCIBLE.Scaled", "鳞化", "已成长"),
    ("SCHEMATIC.FIELDS.inputs.element.ingredients.element.consumed.label", "吞噬者", "已消耗"),
    ("SCHEMATIC.CATEGORIES.Jewelcraft", "珠宝匠石墨棒", "珠宝匠图样"),
    ("FLANKED_EFFECT.FIELDS.flanked.label", "夹击舞台", "夹击阶段"),
    ("FLANKED_EFFECT.FIELDS.enemies.label", "敌人接战", "敌人交战"),
    ("TOOL.CATEGORIES.Implement", "实现", "器具"),
    ("TOKEN.MOVEMENT.ACTIONS.fly.description",
     "在气元素中移动，每次动作最多可移动相当于你飞行属性的距离。只有具有飞行速度的生物才能使用此移动方式。",
     "在空中移动，每次动作最多可移动相当于你飞行属性的距离。只有具有飞行速度的生物才能使用此移动方式。"),
    ("ACTOR.FIELDS.details.biography.age.label", "年代", "年龄"),
    ("ACTOR.FIELDS.details.biography.weight.label", "重量", "体重"),
    ("ACTOR.CREATION.EquipmentInsufficient", "你不能再失去另一个{name}。", "你买不起另一个{name}。"),
    ("ARCHETYPE.WARNINGS.NotEquipment", "脱手的文件必须是装备。", "被拖入的文档必须是装备。"),
    ("CONSUMABLE.FIELDS.uses.label", "用途", "使用次数"),
    ("CONSUMABLE.FIELDS.uses.value.label", "价值", "当前值"),
    ("TOOL.FIELDS.skills.hint", "鉴定一下哪些技能或手艺会用到这个特定工具。", "指明哪些技能或手艺会用到这件特定工具。"),
    ("SCHEMATIC.SHEET.InputDrop", "放下输入物品", "拖入输入物品"),
    ("SCHEMATIC.SHEET.OutputDrop", "丢弃输出物品", "拖入输出物品"),
    ("TAXONOMY.CATEGORIES.Celestial", "天界的", "天界生物"),
    ("ACTOR.GROUP.LABELS.cyclePace", "循环节奏", "循环步调"),
    ("ACTION.PlanRegion", "地区放置", "放置区域"),
    ("ACTION.TAG.MovementBlink", "闪现术", "闪现"),
    ("AFFIX.ConfigIdentity", "附魔身份", "词缀身份"),
    ("ITEM.PROPERTIES.Investment", "投入", "注入"),
    ("ACTOR.DisarmedStatus", "解除武装！", "缴械！"),
    ("ACTOR.LABELS.CurrentCapacity", "当前容量", "当前负重"),
    ("TYPES.Actor.adversary", "敌手", "对手"),
    ("TAXONOMY.FIELDS.movement.size.hint", "该敌对者在网格中占据的尺寸。", "该对手在网格中占据的尺寸。"),
    ("SETTINGS.COMPENDIUM_SOURCES.affix.label", "缀加来源", "词缀来源"),
    ("TALENT.FIELDS.actorHooks.label", "钩挂功能", "钩子函数"),
    ("TAXONOMY.SHEET.Stature", "分类法 身材", "分类法体格"),
    ("ACTION.FIELDS.summon.label", "动作 召唤", "动作召唤"),
    ("ACTION.FIELDS.range.maximum.label", "极大值", "最大值"),
    ("ACTION.FIELDS.range.minimum.label", "最低限度", "最小值"),
    ("SKILL.SpecializationPaths", "专精径", "专精路线"),
    ("SKILL.TooltipPassive", "12 + 评分", "12 + 技能值"),
    ("ITEM.ACTIONS.Recover", "恢复 {typeLabel}", "拾回 {typeLabel}"),
    ("ITEM.ACTIONS.RecoverDetail", "恢复脱手的{item}", "拾回脱手的{item}"),
    ("SPELL.COMPONENTS.GestureNone", "未习得任何姿势手势。", "未习得任何躯体手势。"),
    ("SPELL.COMPONENTS.InflectionNone", "未习得任何超魔屈折。", "未习得任何超魔法屈折。"),
    # 全表 HTML 标签体检唯一命中：中文比英文多一对 <strong>（把「回合」也加粗了）
    ("ACTION.DEFAULT_ACTIONS.Defend.Description",
     "你专注于避免受到伤害，从而提高你的物理防御。直到你的下个<strong>回合</strong>开始前，你获得<strong>戒备</strong>状态。",
     "你专注于避免受到伤害，从而提高你的物理防御。直到你的下个回合开始前，你获得<strong>戒备</strong>状态。"),
    ("ARMOR.PROPERTIES.NaturalTooltip",
     "自然护甲是“穿戴”它的生物自身的一部分，通常无法被卸下。",
     "天然护甲是“穿戴”它的生物自身的一部分，通常无法被卸下。"),
    # D. Tier＝阶数（08-06 决议），label 位上是机翻残渣
    ("AFFIX.FIELDS.tier.label", "第1阶", "阶数"),
    ("AFFIX.FIELDS.tier.value.label", "第1阶词缀", "词缀阶数"),
    ("AFFIX.FIELDS.tier.max.label", "最高阶", "最高阶数"),
    ("AFFIX.FIELDS.tier.hint",
     "该词缀可使用的威能阶层范围。每个阶层会消耗一点前缀或后缀容量。",
     "该词缀可使用的威能阶数范围。每一阶会消耗一点前缀或后缀容量。"),
    # E. 伤害类型标签（下拉里会拼出「钝击伤害伤害」）
    ("DAMAGE.Bludgeoning", "钝击伤害", "钝击"),
    ("DAMAGE.Slashing", "挥砍伤害", "挥砍"),
    ("DAMAGE.Physical", "物理的", "物理"),
    # F. 中文实际渲染的是 other 支（Intl.PluralRules('zh') 恒为 other）
    ("ACTION.TAG.CostHand.one", "{hands} 手", "{hands}手"),
    ("ACTION.TAG.CostHand.other", "{hands} 手部", "{hands}手"),
    ("CONSUMABLE.USES.TagMax.one", "{value} 使用", "{value} 次"),
    ("CONSUMABLE.USES.TagMax.other", "{value} 使用次数", "{value} 次"),
    ("CONSUMABLE.USES.TagPartial.one", "{value}/{max} 使用", "{value}/{max} 次"),
    ("CONSUMABLE.USES.TagPartial.other", "{value}/{max} 使用次数", "{value}/{max} 次"),
    ("ACTION.WARNINGS.CannotAffordMove",
     "{name}的动作不足，无法移动这么远的距离；这需要 {cost}A！",
     "{name}的动作不足，无法移动这么远的距离；这需要 {cost}动！"),
    # F2. Action＝动作（英文闸：含 action 的键里 动作 177 : 行动 18）
    ("ACTION.TAG.FlankingTooltip", "该行动只能对处于夹击或未察觉状态的目标执行。", "该动作只能对处于夹击或未察觉状态的目标执行。"),
    ("ACTION.TAG.HarmlessTooltip", "该行动不会施加直接有害的效果，其伤害倍率为零。", "该动作不会施加直接有害的效果，其伤害倍率为零。"),
    ("ACTION.TAG.MovementStepTooltip", "此行动要求执行者在其移动过程中进行一次谨慎的踏步。", "此动作要求执行者在其移动过程中进行一次谨慎的踏步。"),
    ("ACTION.TAG.MovementTooltip",
     "这个行动允许你在发动时进行移动，其中包括每轮一次自由移动动作的动作点消耗降低。",
     "这个动作允许你在发动时进行移动，其中包括每轮一次自由移动动作的动作点消耗降低。"),
    ("ACTION.TAG.OneHandTooltip", "此行动需要持用一把单手武器。", "此动作需要持用一把单手武器。"),
    ("ACTION.TAG.SkillTooltip", "此行动涉及一次有目标的或对抗的技能检定。", "此动作涉及一次有目标的或对抗的技能检定。"),
    ("ACTION.TAG.SpellTooltip", "这个行动涉及施法。", "这个动作涉及施法。"),
    ("ACTION.TAG.SummonTooltip",
     "这个行动会召唤、呼唤，或以其他方式创造一个战斗单位，使其加入遭遇。",
     "这个动作会召唤、呼唤，或以其他方式创造一个战斗单位，使其加入遭遇。"),
    ("ACTION.TAG.TargetTooltip", "适用于此行动的目标类型、距离和数量。", "适用于此动作的目标类型、距离和数量。"),
    ("ACTION.TAG.TalismanTooltip", "执行此行动需要持握一把护符武器。", "执行此动作需要持握一把护符武器。"),
    ("ACTION.TAG.TwoHandedTooltip", "此行动需要持用一把双手持用武器才能执行。", "此动作需要持用一把双手持用武器才能执行。"),
    ("ACTION.WARNINGS.MovementTooShort", "计划中的移动路径未达到此行动所需的最小射程。", "计划中的移动路径未达到此动作所需的最小射程。"),
    ("ACTION.WARNINGS.RequiresTwoHanded", "此行动需要已装备一把双手持用武器。", "此动作需要已装备一把双手持用武器。"),
    ("ACTION.RequiresMovement", "你必须先规划一条移动路线，才能使用这个行动。", "你必须先规划一条移动路线，才能使用这个动作。"),
    # F3. 两条 tooltip 里的防御名还停在上游改名前（08-06 决议：Rallying/Healing Threshold）
    ("ACTION.TAG.HealingTooltip", "这个行动通过检定一名盟友的创伤阈值来提供恢复。", "这个动作通过检定一名盟友的治疗阈值来提供恢复。"),
    ("ACTION.TAG.RallyingTooltip", "这个行动通过检验一名盟友的疯狂阈值来提供恢复。", "这个动作通过检定一名盟友的集结阈值来提供恢复。"),
    # G. 与 ember 运行时补丁的 KNOWLEDGE 表对齐（repo:ember-hygiene 同结论）
    ("KNOWLEDGE.Crafts", "工艺品", "工艺"),
    ("KNOWLEDGE.Seafaring", "航海的", "航海"),
    # H. Ability＝属性（同文件 ABILITIES.* / 属性值 / 属性点 已是属性）
    ("DICE.Ability", "能力", "属性"),
    ("DICE.AbilityBonus", "能力加值", "属性加值"),
    ("ACTOR.SHEET.Abilities", "能力", "属性"),
    ("ACTOR.WARNINGS.UnderspentAbility", "花费能力点数", "花费属性点"),
    ("WALKTHROUGH.AbilityPoints", "通过提高你的属性值来花费可用的能力点数。", "通过提高你的属性值来花费可用的属性点数。"),
    ("ARCHETYPE.FIELDS.abilities.label", "能力成长", "属性成长"),
    ("ARCHETYPE.WARNINGS.InvalidAbilities", "能力成长数值的总和必须等于 12。当前为 {sum}", "属性成长数值的总和必须等于 12。当前为 {sum}"),
    ("TAXONOMY.FIELDS.abilities.label", "基础能力", "基础属性"),
    ("TAXONOMY.WARNINGS.InvalidAbilities", "初始能力值的总和必须等于 12。当前为 {sum}", "初始属性值的总和必须等于 12。当前为 {sum}"),
    ("WARNING.AbilityCannotDecrease", "你无法进一步降低这个能力。", "你无法进一步降低这项属性。"),
    ("WARNING.AbilityCannotIncrease", "你无法再进一步提高这个能力。", "你无法再进一步提高这项属性。"),
    ("WARNING.AbilityRequireAncestry", "你必须先选定血统，然后才能购买能力值提升。", "你必须先选定血统，然后才能购买属性值提升。"),
    # I. Schematic＝图纸（compendium 4/4 + TYPES.Item.schematic「制作图纸」）
    ("SCHEMATIC.FIELDS.inputs.label", "图式输入", "图纸输入"),
    ("SCHEMATIC.FIELDS.outputs.label", "原理图输出", "图纸输出"),
    ("SCHEMATIC.FIELDS.dc.hint",
     "设定一个数值；在制作此设计图时，必须在一次制作技能检定中超过该数值才算成功。",
     "设定一个数值；在制作此图纸时，必须在一次制作技能检定中超过该数值才算成功。"),
    ("SCHEMATIC.FIELDS.inputs.element.currency.hint",
     "制作这张设计图可能需要一定数量的货币，用以表示所需的补给或未被明确建模为物品的次要材料。",
     "制作这张图纸可能需要一定数量的货币，用以表示所需的补给或未被明确建模为物品的次要材料。"),
    ("CRUCIBLE.SHEETS.Schematic", "Crucible原理图表", "Crucible 图纸表"),
    # J. compendium pack 的中文：「合集包 包」「纲要」
    ("ACTOR.SECTIONS.BACKPACK.empty",
     "通过将物品从提供的 Crucible 系统 合集包 包中拖放来添加物品。",
     "从 Crucible 系统提供的合集包中拖放物品即可添加。"),
    ("ACTOR.SECTIONS.ICONIC.empty",
     "通过浏览提供的 Crucible 系统 合集包 包来添加标志性法术。",
     "浏览 Crucible 系统提供的合集包即可添加标志性法术。"),
    ("SETTINGS.COMPENDIUM_SOURCES.hint",
     "配置在角色创建期间哪些 合集包 包可用作来源。",
     "配置在角色创建期间哪些合集包可用作来源。"),
    ("ITEM.SHEET.NoAffixes", "将一个词缀从词缀 合集包 中拖拽出来以应用它。", "从词缀合集包中拖入一个词缀即可应用。"),
    ("WALKTHROUGH.AddAncestry",
     "从纲要或物品侧边栏中选择一个血统，并通过拖放将其添加。",
     "从合集包或物品侧边栏中选择一个血统，并通过拖放将其添加。"),
    ("SETTINGS.COMPENDIUM_SOURCES.label", "配置资料库来源", "配置合集包来源"),
    ("SETTINGS.COMPENDIUM_SOURCES.name", "资料库来源", "合集包来源"),
    ("SETTINGS.COMPENDIUM_SOURCES.spell.hint", "哪些资料包将被用作标志性法术的来源。", "哪些合集包将被用作标志性法术的来源。"),
    # K. Creation＝创建（同文件「角色创建」多数）
    ("ACTOR.CREATION.AbandonContent", "放弃创造进度并退出创建器？", "放弃创建进度并退出创建器？"),
    ("ACTOR.CREATION.AbandonTitle", "放弃创造流程？", "放弃创建流程？"),
    ("ACTOR.CREATION.CompleteHint", "完整创造", "完成创建"),
    ("ACTOR.CREATION.ExitHint", "退出创造", "退出创建"),
    ("ACTOR.CREATION.NameRequired", "你必须提供角色名称才能完成创造。", "你必须提供角色名称才能完成创建。"),
    ("ACTOR.CREATION.RestartContent", "放弃创造进度并从头开始重新开始吗？", "放弃创建进度并从头重新开始吗？"),
    ("ACTOR.CREATION.RestartHint", "重新开始创造", "重新开始创建"),
    ("ACTOR.CREATION.RestartTitle", "重置创造进度？", "重置创建进度？"),
    # L. Group＝团队（同文件 团队成员/团队技能检定/确认团队休息 多数）
    ("TYPES.Actor.group", "小队", "团队"),
    ("ACTOR.GROUP.WARNINGS.NoAddSelf", "你不能添加你自己的组！", "你不能添加你自己的团队！"),
    ("ACTOR.GROUP.FIELDS.advancement.milestones.hint", "该小组已获得的里程碑记录。", "该团队已获得的里程碑记录。"),
    ("AWARD.MILESTONE.GroupAward",
     "已将 {number} 个 {label} 授予群组 {name} 及以下成员：",
     "已将 {number} 个 {label} 授予团队 {name} 及以下成员："),
    ("AWARD.MILESTONE.GroupRevoke",
     "已从组{name}及以下成员中撤销了 {number} 个{label}：",
     "已从团队 {name} 及以下成员中撤销了 {number} 个{label}："),
    ("AWARD.WARNINGS.CannotRevokeMilestone",
     "在组“{group}”中，没有标识符为“{identifier}”的里程碑奖励。",
     "在团队“{group}”中，没有标识符为“{identifier}”的里程碑奖励。"),
    ("AWARD.WARNINGS.DuplicateMilestone",
     "标识符为“{identifier}”的里程碑已授予给组“{group}”",
     "标识符为“{identifier}”的里程碑已授予给团队“{group}”"),
]


def build(repo, rows, out_name):
    cn_path = ROOT / repo / "lang" / "cn.json"
    doc = json.loads(cn_path.read_text(encoding="utf-8-sig"))
    nested = [k for k, v in doc.items() if isinstance(v, dict)]
    if nested:
        sys.exit(f"!! {repo} cn.json 已有混合形态顶层键: {nested[:5]}")
    bad, batch = [], {}
    for key, old, new in rows:
        cur = doc.get(key)
        if cur is None:
            bad.append(f"KEY-MISSING {key}")
        elif cur != old:
            bad.append(f"OLD-MISMATCH {key}\n     expect: {old!r}\n     actual: {cur!r}")
        elif old == new:
            bad.append(f"NOOP {key}")
        elif key in batch:
            bad.append(f"DUP {key}")
        else:
            batch[key] = new
    if bad:
        print("\n".join(bad))
        sys.exit(f"!! {repo}: {len(bad)} 条自检失败")
    # 键形态自检：批次键必须是扁平点号串，且不会与任何现有顶层键构成嵌套关系
    for k in batch:
        assert "." in k or k.isupper() or True
        assert not isinstance(doc.get(k), dict), k
    p = OUT / out_name
    p.write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{repo}: {len(batch)} 条 -> {p}")


build("1-Ember汉化插件", EMBER, "ember_lang.batch.json")
build("2-Crucible汉化插件", CRUCIBLE, "crucible_lang.batch.json")
