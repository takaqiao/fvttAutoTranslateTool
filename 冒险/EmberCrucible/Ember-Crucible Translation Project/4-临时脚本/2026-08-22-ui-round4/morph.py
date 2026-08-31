# -*- coding: utf-8 -*-
"""装备族部件名的**形态素词典**（覆盖 611 个待译显示名）。

三条来源纪律（本项目既定的术语优先级）：
  ① **系统定译最高**：Crucible 的品质四档直接取 crucible-cn 的 `ITEM.Quality*`
     （对应 `crucible/lang/en.json:1834` 那一族）—— Shoddy 粗糙 / Standard 标准 /
     Fine 精良 / Superior 卓越。这四个词出现 87 次，是本轮最大的一处「不能自己拍脑袋」。
  ② 其次是**发布中的部件表**里已定过的同名段（48 个）。
  ③ 再次是 glossary_ec（94 个命中）—— 但**逐条核过语境，不许直接套**。实测有六条套了就错：
       Shield   → 词表给「护盾术」，那是**法术**；这里是 front/item 上的**盾牌**
       Point    → 词表给「岬」，那是**世界地图地名**的裁决（R-point-cape）；这里是帽子的**尖顶**
       Split    → 词表给「分裂」；这里 `ShirtClothSplit` 是**开衩**
       Sticks   → 词表给「斯蒂克斯」（专名）；这里 `BundleSticks` 是一捆**木棍**
       Water    → 词表给「水域」；这里 `BowlWoodenWater` 是碗里盛的**水**
       Alchemist→ 词表给「阿克图里安」（一眼串行的脏数据）；这里作**炼金术士**
     ⇒ 「词表里有」不等于「这条能用」。同一个英文词在不同语域本来就该分裂 ——
       这正是本项目 term_domains 那一族判据存在的理由。

组合规则见 compose.py。`HEADS` 殿后，`MODS` 前置，`SUFFIX` 原样带到最后。
"""

QUALITY = {
    "Shoddy": "粗糙", "Standard": "标准", "Fine": "精良", "Superior": "卓越",
}

MATERIAL = {
    "Cloth": "布", "Leather": "皮革", "Metal": "金属", "Steel": "钢", "Bone": "骨",
    "Fur": "毛皮", "Silk": "丝", "Stone": "石", "Wood": "木", "Wooden": "木制",
    "Bronze": "青铜", "Coral": "珊瑚", "Clay": "陶", "Bamboo": "竹", "Rope": "绳",
    "Ropes": "绳索", "Crystal": "水晶", "Crystaline": "晶质", "Diamond": "钻石",
    "Chainmail": "锁子甲", "Mail": "锁甲", "Scalemail": "鳞甲", "Plate": "板甲",
    "Chain": "锁链", "Chains": "锁链", "Chained": "缀链", "Hide": "兽皮", "Pelt": "毛皮",
    "Birch": "桦木", "Oaken": "橡木", "Straw": "干草", "Sticks": "木棍", "Branch": "枝条",
    "Fungal": "菌质", "Fungi": "菌类", "Mushrooms": "蘑菇", "Plants": "草木",
    "Feathers": "羽毛", "Feathered": "羽饰", "Teeth": "齿", "Skull": "颅骨",
    "Beads": "串珠", "Beaded": "串珠", "Woven": "编织", "Quilted": "绗缝", "Padded": "衬垫",
    "Tartan": "格纹",
}

ARMOR = {
    "Helm": "头盔", "Coif": "锁甲头罩", "Hat": "帽", "Cap": "便帽", "Hood": "兜帽",
    "Headband": "头带", "Headwrap": "头巾", "Turban": "缠头巾", "Halo": "光环",
    "Mask": "面具", "Masked": "覆面", "Veil": "面纱", "Blindfold": "蒙眼布",
    "Eyepatch": "眼罩", "Glasses": "眼镜", "Goggles": "护目镜", "Bandage": "绷带",
    "Breastplate": "胸甲", "Brigandine": "布面甲", "Gambeson": "缉甲衣", "Jerkin": "皮外衣",
    "Jacket": "夹克", "Coat": "外套", "Vest": "背心", "Tunic": "束腰衣", "Blouse": "罩衫",
    "Shirt": "衬衫", "Shirts": "衬衫", "Dress": "连衣裙", "Stola": "斯托拉长袍",
    "Toga": "托加袍", "Robe": "长袍", "Robes": "长袍", "Shawl": "披肩", "Scarf": "围巾",
    "Cape": "短披风", "Cloak": "斗篷", "Mantle": "披氅", "Collar": "领饰",
    "Necklace": "项链", "Amulet": "护符", "Ring": "戒指", "Pauldron": "肩甲",
    "Bracer": "护腕", "Gauntlet": "臂铠", "Glove": "手套", "Sleeve": "袖", "Sleeveless": "无袖",
    "Apron": "围裙", "Belt": "腰带", "Sash": "饰带", "Cord": "束绳", "Faulds": "腰甲裙",
    "Skirt": "裙", "Skirted": "带裙摆", "Shorts": "短裤", "Trousers": "长裤",
    "Greaves": "胫甲", "Boot": "长靴", "Shoe": "鞋", "Sandal": "凉鞋", "Slipper": "便鞋",
    "Suspenders": "背带", "Bandolier": "弹药带", "Backpack": "背包", "Bedroll": "铺盖卷",
    "Bag": "布袋", "Pouch": "小袋", "Quiver": "箭袋", "Basket": "篮", "Bundle": "捆",
    "Wings": "翼", "Armor": "护甲", "Splint": "夹板甲", "Scale": "鳞片", "Wrap": "缠裹",
}

WEAPON = {
    "Axe": "斧", "Hatchet": "手斧", "Sword": "剑", "Longsword": "长剑", "Shortsword": "短剑",
    "Greatsword": "巨剑", "Katana": "武士刀", "Rapier": "细剑", "Scimitar": "弯刀",
    "Dagger": "匕首", "Blade": "利刃", "Club": "棍棒", "Greatclub": "巨棒", "Cudgel": "短棒",
    "Mace": "硬头锤", "Morningstar": "钉头锤", "Hammer": "锤", "Maul": "巨槌",
    "Flail": "链枷", "Whip": "长鞭", "Spear": "矛", "Javelin": "标枪", "Trident": "三叉戟",
    "Halberd": "戟", "Glaive": "长柄刀", "Scythe": "长柄镰", "Sickle": "镰刀",
    "Pick": "镐", "Shovel": "铲", "Bow": "弓", "Crossbow": "弩", "Sling": "投石索",
    "Arrows": "箭矢", "Shield": "盾牌", "Buckler": "小圆盾", "Heater": "熨斗盾",
    "Kite": "鸢形盾", "Staff": "长杖", "Rod": "短杖", "Wand": "魔杖", "Orb": "法珠",
    "Book": "书", "Tome": "典籍", "Spellbook": "法术书", "Scroll": "卷轴",
    "Torch": "火把", "Lantern": "提灯", "Lamp": "灯", "Jug": "壶", "Bowl": "碗",
    "Lyre": "里拉琴", "Mandolin": "曼陀林", "Flag": "旗",
    "Pole": "旗杆", "Missile": "导弹", "Pod": "发射巢", "Thrower": "喷射器",
    "Backbones": "脊骨", "Mandible": "颚骨", "Arm": "臂", "Tail": "尾", "Horns": "角",
    "Potions": "药水", "Tools": "工具", "Smithing": "锻造", "Mining": "采矿",
    "Cutters": "剪钳", "Button": "纽扣", "Straps": "束带", "Circles": "圆环",
    "Rays": "光芒", "Stripe": "条纹", "Flames": "火焰", "Flame": "火焰", "Glow": "辉光",
    "Sun": "太阳", "Lion": "狮", "Wolf": "狼", "Boar": "野猪", "Lizard": "蜥",
    "Water": "水", "Grain": "谷物", "Rags": "破布", "Shoulders": "双肩", "Shoulder": "肩",
    "Top": "上装", "Inner": "内衬", "Breather": "呼吸器",
}

MOD = {
    "Long": "长", "Short": "短", "Large": "大", "Small": "小", "Full": "全", "Half": "半",
    "Heavy": "重型", "Light": "轻型", "Thin": "薄", "Thick": "厚", "Low": "低帮",
    "Round": "圆", "Circular": "圆形", "Oval": "椭圆", "Domed": "穹顶", "Flat": "平顶",
    "Point": "尖顶", "Pointed": "尖头", "Flanged": "凸棱", "Serrated": "锯齿",
    "Hooked": "带钩", "Toothed": "带齿", "Spiked": "尖刺", "Spikded": "尖刺",
    "Great": "巨型", "Double": "双排", "Breasted": "襟", "Divided": "分片", "Split": "开衩",
    "Twotailed": "双尾", "Winged": "带翼", "Flared": "外张", "Sweeping": "曳地",
    "Billowing": "鼓风", "Flowing": "飘垂", "Draped": "垂坠", "Rolled": "卷起",
    "Unrolled": "展开", "Looped": "环扣", "Tied": "系结", "Laced": "系带",
    "Wrapped": "缠绕", "Bloused": "束口", "Gathered": "抽褶", "Ruffled": "荷叶边",
    "Frilled": "褶边", "Cuffed": "翻边", "Rimmed": "镶边", "Lined": "内衬",
    "Collared": "带领", "Banded": "束带", "Ringed": "环箍", "Riveted": "铆钉",
    "Studded": "镶钉", "Strapped": "绑带", "Stitched": "缝线", "Embossed": "压纹",
    "Layered": "层叠", "Scalloped": "扇贝纹", "Rippled": "波纹", "Wavy": "波浪",
    "Patterned": "纹样", "Decorated": "装饰", "Jeweled": "宝石镶嵌", "Burnished": "抛光",
    "Plated": "镀面", "Reinforced": "加固", "Closed": "闭合", "Open": "敞开",
    "Partial": "半覆", "Both": "双侧", "Left": "左", "Right": "右", "Upper": "上段",
    "Lower": "下段", "Inspection": "检视", "Vigilante": "义警", "Travel": "旅行",
    "Work": "工作", "Battle": "战", "War": "战争", "Common": "寻常", "Basic": "基础",
    "Plain": "素面", "Simple": "简约", "Fancy": "华丽", "Ornate": "华饰",
    "Intricate": "精巧", "Natural": "原生", "Default": "默认", "None": "无",
    "Torn": "破洞", "Tattered": "褴褛", "Ragged": "破烂", "Frayed": "毛边",
    "Ripped": "撕裂", "Battered": "打损", "Broken": "断裂", "Burned": "烧焦",
    "Dirtied": "污渍", "Bloated": "臃肿", "Sodden": "浸水", "Dense": "厚密",
    "Empty": "空", "Lit": "点燃", "Unlit": "未燃", "Bandaged": "包扎", "Opaque": "不透光",
    "Bony": "嶙峋", "Horned": "有角", "Feline": "猫形", "Furred": "覆毛", "Scaled": "覆鳞",
    "Nosed": "尖鼻", "Soft": "柔软", "Trimmed": "镶饰", "Gothic": "哥特",
    "Ethereal": "以太", "Resting": "静置", "Spell": "施法", "Action": "动作",
    "Down": "向下", "Forward": "前跨", "Backward": "后撤", "Neutral": "常态", "Sitting": "坐姿",
    "Miner": "矿工", "Sailor": "水手", "Alchemist": "炼金术士", "Hand": "单手",
    # `Kettle` 全集里只出现在 `HelmSteelKettle` 一条上 —— 那是**壶盔**（kettle hat）
    # 这个形制，不是「水壶」。所以它归修饰、不归中心词。
    "Kettle": "壶形",
    "Domino": "多米诺", "Capped": "覆顶", "Mutated": "变异",
    # 与发布中的部件表同词（那张表里 Feminine 阴柔 / Masculine 阳刚 已定过）
    "Feminine": "阴柔", "Masculine": "阳刚",
}

PROPER = {
    "Agrimage": "农艺法师", "Arcturian": "阿克图里安", "Bejak": "贝雅克",
    "Cindaric": "辛达里克", "Corpuleth": "尸团怪", "Doomsayer": "末日预言者",
    "Kithil": "基希尔", "Lumek": "卢梅克", "Otherhood": "幸运异姊会",
    "Scalemaw": "鳞颚兽", "Serethus": "塞雷苏斯", "Starmage": "星法师",
    "Strider": "疾行者", "Tayan": "塔扬", "Waerd": "瓦尔德", "Warden": "守林者",
    "Wicked": "邪祟",
}

SUFFIX = {str(i): str(i) for i in range(0, 10)}
SUFFIX.update({c: c for c in "ABCDEF"})

HEADS = {}
HEADS.update(ARMOR)
HEADS.update(WEAPON)

MODS = {}
MODS.update(QUALITY)
MODS.update(MATERIAL)
MODS.update(MOD)
MODS.update(PROPER)
