# -*- coding: utf-8 -*-
"""拼串规则拼不好的那些，逐条手改。

⚠ 这份表是**逐条读完 611 行初稿**之后写的，不是预先猜的。规则能覆盖八成，
  剩下两成全是「英文里两个中心词叠在一起」「族名本身省略了中心词」
  「同一个词在这一族里另有专名」这三类，规则一动就顾此失彼 —— 手改更诚实。

`TAIL` 是**系统性**的那一类：sleeve / pants / pauldron 三族里有大批 id **省掉了中心词**
（`ClothSimpleShort` 就是「简约短布*袖*」），靠所在图层补回来，不是一条条硬写。
"""

# 图层默认中心词：这一族里凡是拼不出中心词的，补上它
TAIL = {
    "sleeve": "袖",
    "pants": "裤",
    "pauldron": "肩甲",
}

OVERRIDES = {
    # ── 两个中心词叠在一起：英文用「上位词 + 具体形制」，中文只留具体的那个 ──
    "ShieldBucklerBronzeStandard": "标准青铜小圆盾",
    "ShieldBucklerSteelStandard": "标准钢小圆盾",
    "ShieldHeaterSteelStandard": "标准钢熨斗盾",
    "ShieldKiteSteelStandard": "标准钢鸢形盾",
    "ShieldRoundOrnateFine": "精良华饰圆盾",
    "ShieldRoundMetal": "金属圆盾",
    "ShieldWoodenRound": "木制圆盾",
    "ShieldMetalHeater": "金属熨斗盾",
    "ShieldLightSteelStandard": "标准钢轻盾",
    "ClubBoneCudgel": "骨短棒",
    "ClubBoneMace": "骨硬头锤",
    "BookLeatherTome": "皮革典籍",
    "BagCloth": "布袋",
    "ClothRags": "破布衣",
    "PeltFurThick": "厚毛皮",
    "ArmorBoneMandible": "颚骨护甲",

    # ── 中文里该合成一个词，别把修饰拆出来 ──
    "AxeGreatSteelStandard": "标准钢巨斧",
    "ClubGreatStandard": "标准巨棒",
    "ClubGreatSuperior": "卓越巨棒",
    "AxeHandStandard": "标准手斧",
    "AxeHandSuperior": "卓越手斧",
    "CrossbowHandStandard": "标准手弩",
    "HammerWarStandard": "标准战锤",
    "HammerWarLightStandard": "标准轻战锤",
    "SickleWarBronzeShoddy": "粗糙青铜战镰",
    "HammerPoleSteelFine": "精良钢长柄锤",      # pole hammer＝长柄锤，不是「旗杆」
    "ChainHeavySuperior": "卓越重型锁子甲",      # 这一族的 Chain 指 chain mail
    "ChainMetalHeavy1": "重型金属锁子甲 1",
    "RingMailFine": "精良环甲",                  # ring mail＝环甲，不是「锁甲戒指」
    "TatteredLayeredMail": "褴褛层叠锁甲",
    "ShirtDoubleBreasted": "双排扣衬衫",
    "ShirtDoubleBreastedLower": "双排扣衬衫下段",
    "TopSilkLow": "低领丝上装",                  # Low 在这里是低领口，不是低帮
    "PlateMetalBurnishedHand": "手工抛光金属板甲",
    "BootMetalPlate": "金属板甲长靴",
    "BracerMetalPlate1": "金属板甲护腕 1",
    "GauntletMetalPlate1": "金属板甲臂铠 1",
    "SandalOpen": "露趾凉鞋",
    "BlindfoldCloth": "布蒙眼罩",
    "BlindfoldClothFancy": "华丽布蒙眼罩",
    "BlindfoldClothThin": "薄布蒙眼罩",
    "SkirtMetalFaulds1": "金属腰甲裙 1",
    "SkirtMetalFaulds2": "金属腰甲裙 2",
    "SkirtMetalFaulds3": "金属腰甲裙 3",
    "SkirtMetalFaulds4": "金属腰甲裙 4",
    "SkirtStuddedLeather": "镶钉皮革裙",
    "DressClothStola": "布斯托拉长裙",
    "DressClothStolaSleeveless": "无袖布斯托拉长裙",
    "NecklaceStoneAmulet": "石护符项链",
    "NecklaceBoneTeethSmall": "小骨齿项链",
    "NecklaceMetalRingedThin": "细环金属项链",
    "CloakClothShoulders": "及肩布斗篷",
    "CloakClothShoulder": "布斗篷肩片",   # pauldron 层；与 neck 层的 CloakClothShoulders（及肩布斗篷）分开
    "BundleSticks": "柴捆",
    "HatMetalMinerLamp": "带灯矿工金属帽",
    "HatClothDrapedTail": "垂尾布帽",
    "HoodJeweledFullKithil": "基希尔全覆宝石兜帽",
    "HoodClothDown1": "垂下布兜帽 1",
    "HoodClothDown2": "垂下布兜帽 2",
    "BowLongFancyQuiver": "华丽长弓配箭袋",
    "QuiverLeatherArrows": "带箭皮革箭袋",
    "FlameThrower": "火焰喷射器",
    "MissilePod": "导弹发射巢",
    "BladeArm": "刃臂",
    "Blade": "臂刃",                              # wrist 族，单独一条
    "SleeveClothBracer1": "带护腕布袖 1",
    "SleeveClothBracer2": "带护腕布袖 2",
    "SleeveClothBracer3": "带护腕布袖 3",
    "BeltRopePouch": "带小袋绳腰带",
    "BeltClothSash": "布饰带腰带",
    "TogaClothRightNecklace": "右布托加袍配项链",
    "ShouldersMetalOaken": "橡木金属护肩",
    "ShoulderStrapped": "绑带披肩",
    "ShoulderFurInner": "毛皮内衬肩饰",
    "TartanClothDraped": "垂坠格纹布巾",
    "PaddedClothQuilted": "绗缝衬垫布",
    "StuddedLeatherStrapped": "绑带镶钉皮甲",
    "BandedLeatherStudded": "镶钉束带皮甲",
    "BandedMetalPlated": "镀面束带金属甲",
    "HideHeavyCollared": "带领重型兽皮甲",
    "HeavyBoneArmor": "重型骨护甲",
    "LightBoneArmor": "轻型骨护甲",
    "ClothLayeredTrimmed": "镶饰层叠布衣",
    "SkullBoneLion": "狮骨颅骨",
    "SkullBoneLionMutated": "变异狮骨颅骨",
    "SkullBoneScalemawFull": "完整鳞颚兽骨颅骨",
    "LizardSkull": "蜥颅骨",
    "HornedSkull": "有角颅骨",
    "BrokenBone": "断骨",
    "Bony": "嶙峋骨盔",
    "Feathers": "羽饰",
    "MetalBattered": "打损金属盔",
    "MetalLayeredDirtied": "污渍层叠金属盔",
    "MetalRoundSpiked": "尖刺圆金属盔",
    "MetalCoif": "金属锁甲头罩",
    "MetalCoifDirtied": "污渍金属锁甲头罩",
    "Mushrooms": "蘑菇肩饰",
    "Fungi1": "菌类肩饰 1",
    "Fungi2": "菌类肩饰 2",
    "Fungal": "菌质覆体",
    "Stone": "石质",
    "Basic": "基础腰饰",
    "MetalWater": "金属（水面）",
    "WoodWater": "木（水面）",
    "Metal": "金属",
    "BowlWoodenWater": "盛水木碗",
    "BowlWoodenGrain": "盛谷木碗",
    "BowlWoodenEmpty": "空木碗",
    "SpellSimpleGlow": "简约施法辉光",
    "SpellSmallFlame": "小型施法火焰",
    "MetalSpikedAction": "尖刺金属裤（动作）",
    "Resting": "静置",
}

# ── 第二轮手改：读完 611 行初稿之后补的 ──────────────────────────────────
# ① `FurLined`＝毛皮内衬。规则把 Lined(修饰) 倒到 Fur(材质) 前面，拼出「内衬毛皮X」——
#    意思不错但语序别扭，中文习惯是「毛皮内衬X」。这一族逐条列出来，不动规则。
OVERRIDES.update({
    "CloakFurLined": "毛皮内衬斗篷",
    "CollarFurLined": "毛皮内衬领饰",
    "HatLeatherFurLined": "毛皮内衬皮革帽",
    "BootLeatherFurLined": "毛皮内衬皮革长靴",
    "BracerMetalFurLined": "毛皮内衬金属护腕",
    "PlateMetalHeavyFurLined": "毛皮内衬重型金属板甲",
})
# ② 头盔的 `Full` 是「全罩」（覆盖整张脸），不是数量上的「全」
OVERRIDES.update({
    "HelmFullTayan": "塔扬全罩头盔",
    "HelmSteelFullClosed": "闭合全罩钢头盔",
    "HelmSteelFullHorned": "有角全罩钢头盔",
    "HelmSteelFullOpen": "敞开全罩钢头盔",
    "MaskFullDoomsayer": "末日预言者全罩面具",
})
# ③ 形制有现成中文名的，别硬拼
OVERRIDES.update({
    "HatBambooRound": "竹斗笠",
    "CapClothSun": "遮阳布便帽",
    "ShawlClothHalfRays": "半幅光芒布披肩",
    "ClothRobeLarge": "大布长袍袖",     # sleeve 层：中心词是 Robe，兜底补不上「袖」
    "ClothRobeShort": "短布长袍袖",
    "ApronClothPatternedHeavy": "厚重纹样布围裙",   # 这里的 Heavy/Light 说的是厚薄
    "ApronClothPatternedLight": "轻薄纹样布围裙",
})

# ④ 图层兜底**不该加**的情形：拼出来的末尾本身已经是一件护甲的名字，
#    再补图层中心词就成了「板甲肩甲」。实测这一族有 8 条（pauldron 层的 Plate*）。
TAIL_SKIP_SUFFIX = {
    "pauldron": ("板甲", "锁子甲", "鳞甲", "锁甲", "夹板甲", "护甲"),
}

# ⑤ 唯一一处跨图层撞名：chest 的 `RobeClothAgrimage` 与 waist 的 `RobesClothAgrimage`
#    是同一件长袍的上下两段，拼出来都是「农艺法师布长袍」。分开写，免得选择器里
#    两行长得一模一样。
OVERRIDES["RobesClothAgrimage"] = "农艺法师布长袍下摆"
