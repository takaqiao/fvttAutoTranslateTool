/**
 * ember-hardcoded-cn.mjs
 *
 * 翻译 Ember 里 **babele 够不到** 的硬编码字符串。
 *
 * Babele 只能翻合集内容（compendium documents）；Foundry 的 i18n 只能翻模块自己
 * 用 lang key 声明的字符串。但 Ember 有一大批文本是直接写死在 `scripts/ember.mjs`
 * 与模板里的 —— 比如 TextEditor 富文本增强器拼出来的 `Attunement: Aura`、
 * 角色卡上的分节标题、事件按钮、确认对话框标题。这些两条通道都碰不到，
 * 只能在运行时替换。
 *
 * 设计原则：
 *   1. **只读不写**：所有替换都发生在渲染出来的 DOM 或增强器返回的节点上，
 *      不改 Ember 的任何数据，卸载本模块即恢复原状。
 *   2. **防御式**：每个补丁点都先探测 API 是否存在，包在 try/catch 里，
 *      失败只在控制台留一条警告，绝不影响开世界。
 *   3. **数据与逻辑分离**：TRANSLATIONS 是纯数据，方便后续增补与校对。
 *
 * 注意：`AC / AB / AT / AS` 这几个纪年缩写有意不译 —— 它们是 Ember 历法的
 * 纪元代号，和 DC 一样属于约定俗成的记号，译开反而认不出来。
 */

const MODULE = "ember_cn_unofficial";
const log = (...a) => console.log(`${MODULE} |`, ...a);
const warn = (...a) => console.warn(`${MODULE} |`, ...a);

/* ============================================================ */
/*  1. 翻译数据                                                  */
/* ============================================================ */

/**
 * 同调（11 轮元素之月 / 界域），译名与 compendium、lang 保持一致。
 *
 * 两套键都要有：合集页名（`Attunement: ${page.name}` 走这一套）与 `ember.CONST.ATTUNEMENTS[x].label`
 * 的短名（`Activate Attunement: ${label}` 走这一套，dnd5e-async.mjs:406 定义为 Abyss / Heart，不带 The / of Ember）。
 * 2026-08-13 第三轮：`Aura` 原译「灵气」是错的 —— 它是月亮专名，Cosmos 页 name 字段即「奥拉 Aura」，
 * 同一份月亮清单里 Mayis/Cora/Ragen/Orbis/Akon 全是音译；「灵气」是 `Aura Spellcraft` 手势的 adjective，不是月名。
 */
const ATTUNEMENTS = {
  "The Abyss": "深渊",
  "Abyss": "深渊",
  "Akon": "阿肯",
  "Aura": "奥拉",
  "Cora": "科拉",
  "Heart of Ember": "余烬之心",
  "Heart": "余烬之心",
  "Luxarum": "卢克萨鲁姆",
  "Mayis": "玛伊斯",
  "Orbis": "奥比斯",
  "Primordis": "普里莫迪斯",
  "Ragen": "拉根",
  "Signara": "西格纳拉"
};

/** 语言。Common / Sign 来自 crucible 本体，其余是 Ember 新增 */
const LANGUAGES = {
  "Common": "通用语",
  "Sign": "手语",
  "Arcden": "奥克登语",
  "Cascal": "卡斯卡尔语",
  "Forest Speech": "森林语",
  "Hardac": "哈达克语",
  "Imperial": "帝国语",
  "Solical": "索利卡尔语",
  "Mithia": "米西亚语",
  "Luma": "卢玛语",
  "Kaziric": "卡兹里克语",
  "Scripta": "斯克里普塔语",
  "Wyrdic": "维尔迪克语",
  "Pathward": "径道语",
  "Scor": "斯科语",
  "Towyr": "托维尔语",
  "Windclaw": "风爪语",
  "Abyssal": "深渊语",
  "Draconic": "龙语",
  "Druidic": "德鲁伊语",
  "Lunix": "卢尼克斯语",
  "Caligon": "卡利贡语",
  "Eonic": "永世语",
  "Harmos": "哈莫斯语",
  "Thieves' Cant": "盗贼黑话"
};

/**
 * `[[/language …]]` 引用了、但**上游根本没有**的语言 id。
 *
 * crucible.CONFIG.languages（ember.mjs:126693 起那张 23 条表）里没有 borel / kost，
 * 于是 enrichLanguage 走 `if (!language) return new Text(match)`（ember.mjs:126542），
 * 正文原样吐出字面量 `[[/language borel]]`。合集实测：borel×2、kost×1（孪生包各一份）。
 * 这里按 **id** 兜底，配合 PATTERNS 末尾那条 `^\[\[\/language …]]$` 把裸标记换成中文；
 * 能生效的前提是增强器包装那边用 `result instanceof Node`（Text 节点也要收），见 patchEnrichers。
 *
 * `moiré` 拿不到这个入口：增强器的 pattern 是 `(\w+)`、无 u 标志，é 不算 \w，
 * 连增强器都不会被调用，字面量停在正文里 —— 那两处只能在 compendium 译文里改掉。
 */
const MISSING_LANGUAGES = {
  "borel": "博雷尔语",
  "kost": "科斯特语"
};

/**
 * 知识领域。前 31 条 crucible 本体的 lang 已经有译名，这里重复一份是因为
 * Ember 的增强器不走 lang key、直接拼英文 label，我们只能按英文原文匹配。
 * 译名与 crucible lang 的 KNOWLEDGE.* 逐条对齐，改一处要两边一起改。
 *
 * 2026-08-12 对表：31 条共有键里漂了 2 条，裁决**以本表为准**，crucible lang 那两条要改过来：
 *   - `Crafts` 本表「工艺」/ lang「工艺品」→ 取**工艺**。Crafts 是知识领域不是成品器物；
 *     crucible.rules 的 Character Mechanics/Background 页那张背景表里
 *     `Crafts, Trade` 译的就是「工艺、贸易」。
 *   - `Seafaring` 本表「航海」/ lang「航海的」→ 取**航海**。其余 30 条 KNOWLEDGE.* 全是
 *     不带「的」的名词，带「的」是机翻形容词残留（与 08-12 裁掉的 `Auditory` 听觉的→听觉
 *     同一类问题）。
 * 2026-08-13 第九轮再裁 6 条（lang 与本表**同时**改过，两边现已逐条一致）：
 *   Crime 罪行→犯罪(英文闸 60叶 : 40叶) · Forensics 法医学→法证学(151 处 [[/knowledge forensics]]
 *   的语境是痕迹勘验不是尸检，且「法医学」全库 0 叶) · Intrigue 阴谋→权谋(「阴谋」已被 conspiracy
 *   占用 22 叶) · Legends 传奇→传说(232叶 : 83叶) · Machines 机械装置→机械(84叶 : 13叶) ·
 *   Undeath 亡灵化→不死(94叶 : 4叶)。Artifacts 保持「神器」（「遗物」已被 Relic 占死 182 叶）。
 * 末尾 4 条是 Ember 新增的领域，crucible lang 里没有对应键。
 */
const KNOWLEDGE = {
  "Alchemy": "炼金术", "Ancients": "远古者", "Artifacts": "神器", "Arts": "艺术",
  "Beasts": "野兽", "Celestials": "天界生物", "Cosmology": "宇宙学", "Crafts": "工艺",
  "Crime": "犯罪", "Dragons": "巨龙", "Elementals": "元素生物", "Fey": "妖精",
  "Fiends": "邪魔", "Forensics": "法证学", "Gods": "诸神", "Intrigue": "权谋",
  "Legends": "传说", "Machines": "机械", "Monsters": "怪物", "Outsiders": "外来者",
  "Plants": "植物", "Politics": "政治", "Rituals": "仪式", "Seafaring": "航海",
  "Souls": "灵魂", "Subterranea": "地下世界", "Tracking": "追踪", "Trade": "贸易",
  "Undeath": "不死", "Warfare": "战争", "Weather": "天气",
  // 以下四条为 Ember 新增
  "Abyssals": "深渊裔", "Aedir": "艾迪尔", "Leviathans": "利维坦", "Shent": "申特"
};

/**
 * 音乐氛围。**只有这两档** —— `EmberSoundscape.MOODS`（ember.mjs:15606）就是
 * `{CALM: "calm", TENSION: "tension"}`，enricher 拼的是 `Music Mood: ${mood.titleCase()}`。
 * 原来那五个键（战斗/探索/环境/旅行/休息）在 ember 0.6.x 里一个都不会出现。
 * 译名取 lang/cn.json 的 `EMBER.SoundscapeMoodCalm` / `EMBER.SoundscapeMoodTension`。
 */
const MOODS = {
  "Calm": "平静", "Tension": "紧张"
};

/**
 * 音景「编排」名。arrangement.label 是 ember.mjs 里的硬编码常量
 * （5694 / 5787 / 12393 / 14064 / 14748 各 var 块），babele 与 i18n 两条通道都够不到。
 */
const ARRANGEMENTS = {
  "Reset": "默认",
  "Ancient Ruins": "远古遗迹",
  "Shent Ruins": "申特遗迹",
  "Shent Ruins Tension": "申特遗迹 · 紧张",
  "Ankarist Theme": "安卡里斯特的主题",
  "Lyla Theme": "莱拉的主题",
  "Sin Theme": "辛的主题",
  "The Pit Trap - Intense": "陷坑 · 激烈",
  "The Pit Trap - Relaxed": "陷坑 · 舒缓"
};

/** 带前缀的标签：`前缀: 名字` → `中文前缀：中文名字` */
const PREFIXED = [
  { en: "Attunement", cn: "同调", table: ATTUNEMENTS },
  { en: "Language", cn: "语言", table: LANGUAGES },
  { en: "Knowledge", cn: "知识", table: KNOWLEDGE },
  { en: "Music Mood", cn: "音乐氛围", table: MOODS },
  // 下面四条的叶子名都已经是中文：三个 character-option 增强器取的是 compendium index 的
  // `name`（babele 已译），crucible 的 enrichTalent 取的是 talentIndex.name（同样已译），
  // 所以 table 用空表 —— 只换前缀，名字原样保留。
  //   ember.mjs:22934 `Ancestry: ${name}` / :22954 `Culture: ${name}` / :22986 `Path: ${name}`
  //   crucible-compiled.mjs:46838 `Talent: ${talentIndex.name}`（相邻的 knowledge/language 都走
  //   _loc，只有 talent 这条漏了 i18n，crucible 汉化插件那边又没有运行时字符串层）
  // 注意：EXACT 里那三个裸词 Ancestry/Culture/Path **不是**给这里用的，见 EXACT 的注释。
  { en: "Ancestry", cn: "血统", table: {} },
  { en: "Culture", cn: "文化", table: {} },
  { en: "Path", cn: "道途", table: {} },
  { en: "Talent", cn: "天赋", table: {} },

  // 音景增强器的**前两支**（S1 补）：`EmberSoundscape.enricherHTML`（ember.mjs:16250-16278）
  // 有三个互斥分支，只有第三支是 `Music Mood: …`，前两支拼的是
  //   16255  `${channel.capitalize()}: Reset`               → `Music: Reset`
  //   16266  `${channel.capitalize()}: ${arrangement.label}` → `Music: Ankarist Theme`
  // 已发布语料实测：每包 23 颗按钮里 21 颗落在前两支，`mood=` 只有 2 处。
  // channel 只有 music / environment 两个（ember.mjs:15643）。
  { en: "Music", cn: "音乐", table: ARRANGEMENTS },
  { en: "Environment", cn: "环境音", table: ARRANGEMENTS },

  // 六边形 HUD 的四条 data-tooltip（templates/applications/hex-hud.hbs:13/47/50/52）。
  // 宿主 EmberHexHUD 的 classes 含 "ember"，闸放行、translateNode 也走到了，
  // 只是查表形状对不上带动态尾巴的串。尾巴是 babele 已译的地名/群系名/地形名，故用空表。
  { en: "Area Map", cn: "区域地图", table: {} },
  { en: "Location", cn: "地点", table: {} },
  { en: "Biome", cn: "生物群系", table: {} },
  { en: "Terrain", cn: "地形", table: {} }
];

/**
 * Ember 弹出的原生 DialogV2 的窗口标题（英文原文）。
 *
 * 这张表有两个用途，缺一不可：
 *   ① 标题译文 —— 本表会被 spread 进 EXACT，走 EXACT 的老通道照旧生效；
 *   ② **认框** —— patchRenderedApplications 里的 DialogV2 例外分支拿窗口标题跟这张表比对，
 *      认出「这是 Ember 弹的框」之后才对整个窗口跑 translateNode(root, DIALOG_UI)。
 * 所以每加一条标题，那个框的正文与按钮才跟着解锁；**漏了标题，正文按钮就仍然是英文**。
 * 行末是 modules/ember/scripts/ember.mjs 的行号。
 */
const DIALOG_TITLES = {
  "Add to Party?": "加入队伍？",                                     // 84
  "Re-combine Caravans?": "重新合并商队？",                           // 18788 / 18823
  "Ember: Create Weather": "余烬：创建天气",                           // 22719
  "Find Text in Journals": "在日志中查找文本",                         // 23291
  "Reset Event": "重置事件",                                         // 36938（同时是事件页上的按钮文本）
  "Initiate Event": "启动事件",                                      // 36956
  "Select Outcome": "选择结果",                                      // 36842
  "Delete Saved Composition?": "删除已保存的构图？",                    // 34488
  "Clear Vista": "清空远景",                                         // 34723
  "Import Configuration": "导入配置",                                // 34750
  "Summarize Token Maker Part Usage": "统计令牌制作器部件用量",          // 49532
  "Ember: Teleport Destination": "余烬：传送目的地",                    // 61789
  "Elevator Controls": "升降机控制",                                 // 67247
  "Toggle Corpuleth Damage": "切换尸团怪 Corpuleth 伤害状态",           // 73312
  "Aedir Signalpost Generator Room Switch": "艾迪尔信号哨站 Aedir Signalpost 发电机房开关", // 95126
  "Elevator Destination": "升降机目的地",                             // 95361
  "Steam Cleansing Cutoff": "蒸汽净化切断",                           // 95691
  "Machine": "机器",                                                // 96495
  "Bastion Apex: Orb of Lantyr": "堡垒顶点 Bastion Apex：兰提尔法珠 Orb of Lantyr", // 97170
  "Bastion Apex: Barrier Pillar": "堡垒顶点 Bastion Apex：屏障石柱",     // 97255
  "Bleak Archive Light Beams": "黯淡秘库 Bleak Archive 光束",          // 97558
  "Transition to Pathways?": "转入通路？",                            // 99652
  "Dredging Valve": "疏浚阀门",                                      // 99852
  "Redwalk Ramble - Illusion Control": "红行漫步园 Redwalk Ramble - 幻象控制", // 108413
  "Temple Lunarium": "神殿月辉宫 Temple Lunarium",                    // 109033
  "Ring Alarm Bell?": "敲响警钟？",                                  // 110323
  "Modify Flow Control Valve?": "调整流量控制阀？",                     // 110376
  "Forcefield Control Orb": "力场控制法珠",                           // 110864
  "Vortest Tower Transporter": "沃特斯特塔 Vortest Tower 传送装置",      // 110958
  "Mine Cart Destination": "矿车目的地",                              // 112047 / 112072
  "Install Junction Wheel": "安装枢纽轮盘",                           // 112217
  "Construct Elevator": "构装体升降机",                               // 114356
  "Awaken Vampyre Body?": "唤醒吸血鬼躯体？",                          // 115784
  "Unspent Ability Points": "未分配的属性点",                          // 123317（dnd5e 分支）
  "Apply Soulbound Progression": "应用魂缚进程"                        // 126638 / 126659
  // 缺席说明：ember.mjs:95615 那个 `dialog:{title,icon,description}` 少写了 window 这一层，
  // DialogV2 读不到，标题实际落到基类兜底的 `Interactable: ${id}`（62795），
  // 所以 "Aedir Signalpost Stealth Field Generator" 不在本表，由 DIALOG_TITLE_PATTERNS 认。
};

/** 动态拼出来的 Ember 对话框标题，只用来认框（能翻的那几条在 PATTERNS 里） */
const DIALOG_TITLE_PATTERNS = [
  /^(?:Award|Revoke|Activate) Attunement: /,   // ember.mjs:3051 / 3178 / 3142 / 23181
  /^Token Maker Part Usage: /,                 // ember.mjs:49557
  /^Interactable: /                            // ember.mjs:62795 基类兜底标题
];

/** 标题已由 lang 键译成中文的 Ember 对话框：只用来认框（它们的正文与按钮仍是裸英文） */
const DIALOG_TITLE_I18N = [
  "EMBER.CONTROLS.VistaComposition"            // ember.mjs:32940 / 63640，正文 Composition + 按钮 Change 全英
];

/**
 * 只在**已认出是 Ember 弹的** DialogV2 子树里生效的作用域表。
 *
 * 这里的词大多太通用（Close / Ring / Change / Active / Actor…），进全局 EXACT 会误伤别的模块，
 * 所以单独一张，由 patchRenderedApplications 认框成功后作为 extra 传给 translateNode。
 * 行末是 modules/ember/scripts/ember.mjs 的行号。
 */
const DIALOG_UI = {
  // 通用按钮
  "Interact": "交互",                        // 62794，EmberInteractable 无显式按钮时的兜底 OK
  "Close": "关闭",                           // 49558 / 112049
  "Confirm": "确认",                         // 112074
  "Change": "更改",                          // 32946 / 63640
  "Import": "导入",                          // 34758
  "Search": "搜索",                          // 23301 / 49533
  "Activate": "激活",                        // 110866
  // 升降机 / 矿车 / 转运
  "Move": "移动", "Ascend": "上行", "Descend": "下行", "Call": "呼叫",  // 67273-67274 / 95344-95355
  "Clockwise": "顺时针旋转",                  // 95333，与日志里「左转→顺时针旋转」的说法对齐
  "Counter-Clockwise": "逆时针旋转",           // 95336
  "Forwards": "前进方向", "Backwards": "后退方向", "Unreachable": "无法到达", // 112063-112065
  "Tradeway": "贸易道", "Underbelly": "底腹区", "Construct Assembly": "构装体装配区", // 114360-114362
  // 场景机关按钮
  "Ring": "敲响", "Destroy": "破坏",           // 110325-110327
  "Enable Flow": "开启流量", "Disable Flow": "关闭流量",                // 110378-110379
  "Close Valve": "关闭阀门", "Open Valve": "打开阀门",                  // 95692-95693
  "Fill": "注满", "Purge": "排空", "Befoul": "污染", "Cleanse": "净化", // 100211-100213（疏浚阀门动词）
  "Disable Generator": "关闭发电机", "Restore Power": "恢复供电",        // 95622-95623
  "Machine On": "机器开启", "Machine Off": "机器关闭", "Machine Destroyed": "机器损毁", // 96499-96501
  "Defenses Inactive": "防御未激活", "Defenses Active": "防御已激活", "Orb Destroyed": "法珠已毁", // 97174-97176
  "Broken": "破碎", "Damaged": "已受损", "Repaired": "已修复",          // 97259-97261
  "Reset Vault": "重置宝库", "Activate Beams": "激活光束",              // 97562-97563
  "Reset Body": "重置躯体", "Awaken Vampyre": "唤醒吸血鬼",             // 115788-115789
  "Lock": "锁定", "Unlock": "解锁",                                   // 110891 / 110893
  "Fully Armored": "全副武装", "Helm Broken": "头盔破损", "Armor Broken": "护甲破损", // 73314-73316
  // 表单标签 / 正文
  "Composition": "构图",                      // 32943 / 63637
  "Search Term": "搜索词",                    // 23271
  "Document Types": "文档类型",                // 23282
  "Case Sensitive": "区分大小写",              // 23285
  "Journal Entry": "日志条目", "Actor": "角色", "Item": "物品", "Roll Table": "随机表", // 23277-23280
  "Usage": "用途", "Static": "固定", "Randomization": "随机化",         // 49551-49552
  "Token Maker Part": "令牌制作器部件",                                // 49527
  "Enter a part id as template/layer/part, for example kiska/eyes/Fluffy2":
    "以 模板/层/部件 的形式输入部件 id，例如 kiska/eyes/Fluffy2",       // 49528
  "No world Actors use this part.": "世界中没有角色使用该部件。",         // 49554
  "Select Characters": "选择角色",                                    // 23176
  "Select the characters who should receive the award.": "选择应当获得此项奖励的角色。", // 23177
  "Do you wish to recombine the Party into the Strayhearth Caravan?":
    "是否将队伍重新并入迷炉商队 Strayhearth Caravan？",                  // 18789 / 18824
  "Activate this elevator?": "启动这台升降机？",                        // 67249
  "Direct this elevator to a destination.": "为这台升降机指定一个目的地。", // 114359
  "Activate the force-field control orb?": "激活力场控制法珠？",         // 110871
  "Activate the hidden Generator Room switch?": "启动隐藏的发电机房开关？", // 95131
  "Set the machine's operating state.": "设置该机器的运行状态。",         // 96497
  "The junction wheel is missing. Install a replacement?": "枢纽轮盘缺失。是否安装替换件？", // 112218
  "Are you sure you want to completely clear this vista composition?": "确定要彻底清空该远景构图吗？", // 34730
  "No destinations are currently reachable. Adjust the track levers and try again.":
    "当前没有可到达的目的地。请调整轨道拉杆后重试。",                      // 112042
  "Activate this mine cart with no passenger?": "在无乘客的情况下启动这辆矿车？", // 112066
  "Resetting the event step for this event may introduce critical errors into your Ember game state. Are you sure you wish to proceed?":
    "重置该事件的步骤可能给你的余烬战役状态引入严重错误。确定要继续吗？",    // 36935（模板串跨行，靠折叠空白后命中）
  "Beginning this event may introduce critical errors into your Ember game state. Are you sure you wish to proceed?":
    "开始该事件可能给你的余烬战役状态引入严重错误。确定要继续吗？"         // 36952（同上）
};

/** 完全匹配即可替换的字符串 */
const EXACT = {
  // 英雄创建向导顶栏的步骤标签。上游把 label 写成裸英文（crucible-async.mjs:25/34/44/63），
  // 经 crucible 的 templates/sheets/creation/header.hbs:7 `{{localize step.label}}` 上屏，
  // 而 Foundry core 与两个插件的 lang 里都没有这四个裸键，localize 原样返回，所以能被这里接住。
  // ⚠ 这三行**不是**「富文本增强器前缀单独出现」—— 那个场景不存在：三个增强器拼的永远是
  //    `Ancestry: 名字` 整串，走的是 PREFIXED（见上）。原来的注释认错了来源，一直没人补 PREFIXED。
  "Ancestry": "血统",
  "Culture": "文化",
  "Path": "道途",
  "Attunement": "同调",
  "Token": "令牌",

  // 恩惠 / 祸骰
  "-3 Banes": "-3 祸骰", "-2 Banes": "-2 祸骰", "-1 Banes": "-1 祸骰",
  "+1 Boons": "+1 恩惠骰", "+2 Boons": "+2 恩惠骰", "+3 Boons": "+3 恩惠骰",
  "Critical Success": "大成功",
  "Critical Failure": "严重失败",

  // 事件状态提示
  "Event Completed": "事件已完成",
  "Event Not Completed": "事件未完成",
  "Event Outcome Completed": "事件结果已完成",
  "Event Outcome Not Completed": "事件结果未完成",

  // 角色卡 / 日志分节标题
  "Gamemaster Information": "游戏主持人信息",
  "Ancestry Details": "血统详情",
  "Culture Details": "文化详情",
  "Notable Inhabitants": "知名居民",
  "Secret Lore": "秘辛",
  "At a Glance": "概览",
  "Setting the Scene": "场景设定",
  "Event Details": "事件详情",
  "Journal Summary": "日志摘要",
  "Event Outcomes": "事件结果",
  "Quest Details": "任务详情",
  "Involved Locations": "涉及地点",
  "Event Summary": "事件摘要",
  "Biome Details": "生物群系详情",
  "Locations": "地点",
  "Location Details": "地点详情",
  "Biomes": "生物群系",
  "Related Locations": "相关地点",
  "Events": "事件",
  "Quest Overview": "任务概览",
  "Standalone Event": "独立事件",
  "Quest Event": "任务事件",

  // 操作按钮
  "Begin Event": "开始事件",
  // "Reset Event" 挪进了 DIALOG_TITLES —— 它同时是 ember.mjs:36938 那个确认框的标题，
  // 认框要靠它；DIALOG_TITLES 已 spread 进本表，事件页上的按钮照旧命中。
  "Complete Event": "完成事件",
  "Mark as Discovered": "标记为已发现",
  "Reset Discovery": "重置发现",
  "Award Attunements": "授予同调",
  "Attunements Awarded": "同调已授予",
  "No Awarded Attunements": "无可授予的同调",
  "Award Milestone": "授予里程碑",
  "Milestone Awarded": "里程碑已授予",

  // 按钮浮窗
  "Granted attunement points require awarding.": "已获得的同调点数尚待授予。",
  "All granted attunement points have been awarded.": "所有已获得的同调点数都已授予。",
  "No attunement points have been awarded.": "尚未授予任何同调点数。",
  "Award a milestone point for the completion of this event.": "为完成此事件授予一点里程碑。",
  "The milestone point for this event has already been awarded.": "此事件的里程碑点数已经授予过了。",

  // 对话框标题统一收在 DIALOG_TITLES（那张表同时是「这是不是 Ember 弹的框」的识别依据），
  // 这里 spread 进来，保证走 EXACT 的老通道不变。
  // 顺带订正一处：Install Junction Wheel 原译「安装路口轮盘」，
  // 与合集里的「雅科什塔枢纽轮盘 Yakoshta Junction Wheel」不一致，改「安装枢纽轮盘」。
  ...DIALOG_TITLES,

  // Ember 自己的应用窗口标题与页脚按钮：根 class 含 "ember"，DOM 遍历够得到，
  // 但上游没给 i18n 键 —— ember.mjs:51613 那句 `_loc("Save Changes")` 更是把
  // 「这里就是 i18n 通道」写在脸上，只是没有键，localization.mjs 查不到就原样返回。
  "Ember Vista Configuration": "余烬远景配置",                        // ember.mjs:33836
  "Ember Dynamic Token Randomization Configuration": "余烬动态令牌随机化配置", // ember.mjs:51486
  "Add Part": "添加部件",                                            // ember.mjs:51608
  "Add Color": "添加颜色",                                           // ember.mjs:51611
  "Save Changes": "保存更改",                                        // ember.mjs:51613
  "Exit": "退出", "Exit Creation": "退出创建",                        // ember.mjs:122292
  "Complete": "完成", "Complete Creation": "完成创建",                // ember.mjs:122293-122294
  "Create Weather": "创建天气",                                      // ember.mjs:22744（带 ember class，过得了主闸）
  "Teleport": "传送",                                               // ember.mjs:61810（同上）

  // crucible 的 enrichSpell 给每个法术标签挂的 tooltip（crucible-compiled.mjs:46724，不走 _loc）。
  // 只有把 crucibleSpell 放进增强器包装的闸里才够得到，见 patchEnrichers。
  "Spell tooltips are still TO-DO.": "法术悬浮提示尚未实现。"
};

/** 掷骰结果档位。Ember 用 `Result of X` 的形式作为结局标题 */
const RESULTS = {
  "Success": "成功", "Failure": "失败",
  "Critical Success": "大成功", "Critical Failure": "严重失败"
};

/** 需要按模式改写的（保留其中的动态部分） */
const PATTERNS = [
  { re: /^Result of (.+)$/, cn: (m) => `结果：${EXACT[m[1]] ?? RESULTS[m[1]] ?? m[1]}` },
  { re: /^Award Attunement: (.+)$/, cn: (m) => `授予同调：${m[1]}` },
  { re: /^Revoke Attunement: (.+)$/, cn: (m) => `撤销同调：${m[1]}` },
  { re: /^Activate Attunement: (.+)$/, cn: (m) => `激活同调：${translateLeaf(m[1], ATTUNEMENTS)}` },
  // 世界时钟拼的是整串 `Day 43 - 12:00`（ember.mjs:24576），法典日志表头是纯 `Day 43`
  // （ember.mjs:25243），一条正则同时吃掉两种。原先那条 `^Day\b(.*)$` 兜底会把整串译成
  // 「日 43 - 12:00」，还会误伤远景资源名 `Day, Generic` / `Day, Clear`（ember.mjs:32102），已删。
  { re: /^Day (\d+)\b(.*)$/, cn: (m) => `第 ${m[1]} 天${m[2]}` },

  // 上游没有 borel / kost 这两个语言 id，enrichLanguage 直接 `return new Text(match)`
  // （ember.mjs:126542），正文原样吐出裸标记 `[[/language borel]]`
  { re: /^\[\[\/language (\w+)]]$/, cn: (m) => `语言：${MISSING_LANGUAGES[m[1]] ?? m[1]}` },

  // 动态拼出来的窗口标题
  { re: /^Interactable: (.+)$/, cn: (m) => `可交互物：${m[1]}` },          // ember.mjs:62795 兜底标题
  { re: /^Token Maker Part Usage: (.+)$/, cn: (m) => `令牌制作器部件用量：${m[1]}` }, // ember.mjs:49557

  // 对话框正文里带插值的整句。都是长句，进全局表不会误伤别的模块。
  { re: /^Are you sure you wish to proceed and delete the "(.+)" composition\? This cannot be undone\.$/,
    cn: (m) => `确定要删除「${m[1]}」构图吗？此操作无法撤销。` },                       // ember.mjs:34489
  { re: /^Activate this mine cart with (.+) as its passenger\?$/,
    cn: (m) => `以 ${m[1]} 为乘客启动这辆矿车？` },                                  // ember.mjs:112067
  { re: /^There are downstream events of (.+) which have been started or completed\.$/,
    cn: (m) => `${m[1]} 存在已开始或已完成的下游事件。` },                             // ember.mjs:36934
  { re: /^The (.+) event is not currently available because its prerequisites are not satisfied\.$/,
    cn: (m) => `${m[1]} 事件当前不可用，其前置条件尚未满足。` },                        // ember.mjs:36951
  { re: /^Do you want to (complete this event and )?transition the Party to the Pathways section of the Region map\?$/,
    cn: (m) => `是否${m[1] ? "完成此事件并" : ""}将队伍转移到区域地图的通路 Pathways 区段？` } // ember.mjs:99653
];

/**
 * Ember 历法的月名与星期名。
 *
 * 这是实测反馈出来的一个坑：日历里 **seasons 走 i18n、months 不走**。
 * Ember 的 `EMBER_CALENDAR_CONFIG` 是这么写的：
 *
 *     months:  { values: [{name: "Blooming"}, …] }                       ← 裸英文
 *     seasons: { values: [{name: "EMBER.CALENDAR.SEASONS.BLOOMING"}, …] } ← i18n 键
 *
 * 所以我们在 `lang/cn.json` 里把 `EMBER.CALENDAR.SEASONS.*` 全译了也没用 ——
 * 日历条上显示的日期串由 `formatEmberDate` 用**月名**拼出来，那一份 i18n 够不着。
 *
 * 译名与 `lang/cn.json` 的 `EMBER.CALENDAR.SEASONS.*` **逐字一致**：同一个词在
 * 「季节」和「月份」两处必须同名，否则玩家会以为是两回事。
 *
 * 2026-08-12：`Steading` 原译「庄园」是错的 —— 它是季节名不是建筑，而「庄园」在本库
 * 已被 `Grange` / `Manor` 占用（英文闸：英文写 Grange/Manor 的 120 条叶子中文 120 条全是
 * 「庄园」，如 Dradley Grange 德拉德利庄园），于是正文出现「庄园被称为工业季节」这种句子。
 * 改为 **耕耘**：
 *   - 英文侧 `History/Steading` 页原文 "The Steading is known as the Season of Industry …
 *     the period of the year when people are happiest being productive and working with
 *     their hands"，Gleaning 页又写 "the quiet duty found in the Steading"，
 *     讲的是脚踏实地的劳作季，「耕耘」正是这个意思且兼有勤勉义；
 *   - 与另外五个同为两字动名词的季节名（播种/绽放/拾取/凋零/寂止）保持同一构词与农事语域；
 *   - 库内「耕耘」原本只有 2 处，不与既有译名相撞。
 * **同名的另外两处必须一起改**（否则玩家会同屏看到两套词）：
 *   ① `lang/cn.json` 的 `EMBER.CALENDAR.SEASONS.STEADING`；
 *   ② compendium 里 `History/Steading` 页名与正文的「庄园（季）」。
 */
const CALENDAR_MONTHS = {
  "Seeding": "播种",
  "Blooming": "绽放",
  "Steading": "耕耘",
  "Gleaning": "拾取",
  "Withering": "凋零",
  "Stilling": "寂止"
};

/** 星期名同样是裸英文（`days.values[].name`） */
const CALENDAR_DAYS = {
  "Monday": "周一", "Tuesday": "周二", "Wednesday": "周三", "Thursday": "周四",
  "Friday": "周五", "Saturday": "周六", "Sunday": "周日"
};
const CALENDAR_DAY_ABBR = {
  "Mon": "一", "Tues": "二", "Wed": "三", "Thu": "四",
  "Fri": "五", "Sat": "六", "Sun": "日"
};

/**
 * Ember 塞进 **crucible 自己的** HeroSheet 里的「同调」页签
 * （ember.mjs:124717 addAttunementTab，模板 modules/ember/templates/crucible/tab-attunement.hbs）。
 *
 * 宿主 HeroSheet 的根 classes 是 ["crucible","actor","standard-form","themed","theme-dark"]
 * （crucible base-actor-sheet.mjs:10）、构造函数名是 "HeroSheet"，渲染钩子那道 ember 闸
 * 两个判据都不成立 —— 整张英雄卡在闸外，**补 EXACT 键一样不会生效**，只能按注入点的
 * 选择器单独放行子树，见 INJECTED_SUBTREES。
 *
 * 模板取的是 `{{attunement.label}}`，也就是 ember.CONST.ATTUNEMENTS[].label 那 11 个英文短名
 * （dnd5e-async.mjs:406-416），不是 babele 已译的 `.name`（创角向导的 crucible-attunement.hbs
 * 取的才是 name，同一份数据两处取法不一致），所以这里要带上 ATTUNEMENTS 那张短名表。
 */
const ATTUNEMENT_TAB = {
  ...ATTUNEMENTS,
  "Cosmological Attunements": "寰宇同调",  // tab-attunement.hbs:4；「寰宇」取 lang 的 TYPES.…ember.cosmos
  "Make Active": "设为激活",               // tab-attunement.hbs:38 的 aria-label
  "Active": "激活中"                       // ember.mjs:124757，拼进 tags 的那半截（另半截 Rank 已走 i18n）
};

/**
 * 「Ember 注入到别人窗口里的子树」→「只在这棵子树里生效的表」。
 * 渲染钩子的宿主闸拦下非 Ember 应用之后，会拿这张表逐条 querySelectorAll 补翻。
 */
const INJECTED_SUBTREES = [
  ["section.tab.attunement", ATTUNEMENT_TAB]
];

/* ============================================================ */
/*  2. 翻译引擎                                                  */
/* ============================================================ */

function translateLeaf(name, table) {
  return table[name] ?? name;
}

/**
 * 把一段界面文字翻成中文。翻不了就原样返回 —— 宁可露出英文，也不要猜。
 * @param {string} text
 * @returns {string}
 */
export function translateText(text, extra = null) {
  if (typeof text !== "string") return text;
  const raw = text.trim();
  if (!raw) return text;

  // extra 是「只在某棵子树里生效」的作用域表：已认出的 Ember 对话框、Ember 注入到
  // 别人窗口里的页签。里头装的是 Close / Ring / Change / Active / Actor 这种太通用、
  // 进了全局 EXACT 就会误伤别的模块的词，只有确认过归属之后才查它。
  if (extra && (raw in extra)) return text.replace(raw, extra[raw]);

  if (raw in EXACT) return text.replace(raw, EXACT[raw]);

  for (const { en, cn, table } of PREFIXED) {
    if (raw.startsWith(`${en}: `)) {
      const leaf = raw.slice(en.length + 2);
      return text.replace(raw, `${cn}：${translateLeaf(leaf, table)}`);
    }
  }

  for (const { re, cn } of PATTERNS) {
    const m = raw.match(re);
    if (m) return text.replace(raw, cn(m));
  }

  return text;
}

/**
 * 递归翻译一棵 DOM 子树里的所有文本节点与 tooltip 属性。
 * @param {Node} node
 * @param {Record<string, string>|null} [extra]  只在这棵子树里生效的作用域表，见 translateText
 */
function translateNode(node, extra = null) {
  if (!node) return;
  if (node.nodeType === Node.TEXT_NODE) {
    let t = translateText(node.nodeValue, extra);
    if (t === node.nodeValue) {
      // 上游有一批对话框正文是模板字符串拼的，源码里的换行 + 缩进被原样带进文本节点，
      // 例如 ember.mjs:36934 那段 `…game state. \n            Are you sure…`。
      // 折叠内部空白后再查一次；命中就把 trim 后的整段换掉（首尾空白保留）。
      const flat = node.nodeValue.trim().replace(/\s+/g, " ");
      const c = translateText(flat, extra);
      if (c !== flat) t = node.nodeValue.replace(node.nodeValue.trim(), c);
    }
    if (t !== node.nodeValue) node.nodeValue = t;
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return;
  // v14 的 tooltip 取值顺序是 tooltipHtml > tooltipText > tooltip（tooltip-manager.mjs:138）。
  // ember 的事件状态提示走的正是 data-tooltip-text（ember.mjs:23042/23047），
  // 漏掉它等于「事件已完成 / 事件未完成」那几条永不生效。
  for (const attr of ["data-tooltip", "data-tooltip-text", "data-tooltip-html", "title", "aria-label"]) {
    const v = node.getAttribute?.(attr);
    if (v) {
      const t = translateText(v, extra);
      if (t !== v) node.setAttribute(attr, t);
    }
  }
  for (const child of Array.from(node.childNodes)) translateNode(child, extra);
}

/* ============================================================ */
/*  3. 补丁                                                      */
/* ============================================================ */

function applyOnce(target, flag, fn, label) {
  try {
    if (!target || target[flag]) return false;
    fn();
    Object.defineProperty(target, flag, { value: true, enumerable: false });
    return true;
  } catch (err) {
    warn(`补丁「${label}」失败，已跳过：`, err);
    return false;
  }
}

/**
 * 包住 Ember 注册的富文本增强器。
 * 这些增强器把 `[[/attunement aura]]`、`[[/knowledge alchemy]]` 之类展开成
 * 带英文标签的元素，标签是在 JS 里拼的，两条汉化通道都够不到。
 */
function patchEnrichers() {
  const enrichers = CONFIG.TextEditor?.enrichers;
  if (!Array.isArray(enrichers)) return warn("找不到 CONFIG.TextEditor.enrichers，增强器补丁跳过。");
  let n = 0;
  for (const entry of enrichers) {
    if (typeof entry?.enricher !== "function" || entry.__emberCnWrapped) continue;
    const src = String(entry.pattern ?? "");
    // 按 pattern 里的关键词点名放行。注意这道闸事实上**也会**包住 crucible 自己的几个增强器
    // （crucibleKnowledge / crucibleLanguage / crucibleTalent / crucibleSpell）—— 这是故意的：
    // crucible 的 enrichTalent（crucible-compiled.mjs:46838）拼的是裸模板 `Talent: ${name}`，
    // enrichSpell（:46724）把 "Spell tooltips are still TO-DO." 写进 data-tooltip，两处都不走
    // _loc（相邻的 enrichKnowledge / enrichLanguage 走了），而 crucible 汉化插件那边只有
    // babele-register.js、没有运行时字符串层，只能在这里兜。
    if (!/attunement|language|knowledge|soundscape|ancestry|culture|path|talent|spell|eventState|outcome|Advantage|Critical|date/i.test(src)) continue;
    const original = entry.enricher;
    entry.enricher = async function (...args) {
      const result = await original.apply(this, args);
      try {
        // 判据用 Node 而不是 HTMLElement：增强器解析不出目标时返回的是 `new Text(match)`
        // （crucible-compiled.mjs:46815 / ember.mjs:126542），那是 Text 节点、不是 HTMLElement，
        // 原来这一支直接漏过去，正文就把 `[[/language borel]]` 这种裸标记原样吐给玩家。
        if (result instanceof Node) translateNode(result);
        else if (typeof result === "string") return translateText(result);
      } catch (err) {
        warn("增强器结果翻译失败：", err);
      }
      return result;
    };
    entry.__emberCnWrapped = true;
    n++;
  }
  log(`已包装 ${n} 个 Ember 富文本增强器。`);
}

/**
 * Ember 往 crucible.CONFIG 里塞了自己的语言与知识领域，label 是硬编码英文。
 * 这些 label 会出现在角色卡的下拉框里，改数据是唯一的办法。
 */
function patchCrucibleConfig() {
  const cfg = globalThis.crucible?.CONFIG;
  if (!cfg) return warn("找不到 crucible.CONFIG，配置补丁跳过。");
  let n = 0;
  for (const [key, table] of [["languages", LANGUAGES], ["knowledge", KNOWLEDGE]]) {
    const group = cfg[key];
    if (!group) continue;
    for (const entry of Object.values(group)) {
      if (entry && typeof entry.label === "string" && table[entry.label]) {
        entry.label = table[entry.label];
        n++;
      }
    }
  }
  log(`已改写 crucible.CONFIG 里 ${n} 条 Ember 新增的语言/知识标签。`);
}

/**
 * 改写历法里的月名与星期名。
 *
 * 这两处是**数据**不是文案：`CONFIG.time.worldCalendarConfig` 里存的就是裸英文，
 * 日历条的日期串由它拼出来，i18n 与 babele 两条通道都碰不到。
 *
 * 两个对象都要改：`CONFIG.time.worldCalendarConfig` 是原始配置，
 * `game.time.calendar` 是已经实例化出来的日历 —— 实例可能持有配置的深拷贝，
 * 只改其中一个可能不生效。两边都改，且都做「翻不到就不动」的保护。
 */
function patchCalendarNames() {
  const targets = [CONFIG?.time?.worldCalendarConfig, game?.time?.calendar].filter(Boolean);
  if (!targets.length) return warn("找不到历法配置，月名补丁跳过。");
  let n = 0;
  for (const cal of targets) {
    for (const [key, table] of [["months", CALENDAR_MONTHS], ["days", CALENDAR_DAYS]]) {
      for (const v of cal?.[key]?.values ?? []) {
        if (v && typeof v.name === "string" && table[v.name]) { v.name = table[v.name]; n++; }
        if (v && typeof v.abbreviation === "string" && CALENDAR_DAY_ABBR[v.abbreviation]) {
          v.abbreviation = CALENDAR_DAY_ABBR[v.abbreviation];
        }
      }
    }
  }
  // 日历条多半已经渲染过了，改完要让它重画一次，否则要等下一次时间变动才显示中文
  try {
    for (const app of Object.values(ui.windows ?? {})) {
      if (/calendar/i.test(app?.constructor?.name ?? "")) app.render(false);
    }
    document.querySelector("#ember-calendar")?.dispatchEvent(new Event("change"));
  } catch { /* 重画失败不影响下次开界面 */ }
  log(`已改写历法里 ${n} 个月名/星期名。`);
}

/**
 * 把「第 43 天」翻在**源头**，而不是翻在 DOM 上。
 *
 * 世界时钟那行字是 `EmberCalendarUI#animate()` 直接写 innerText 的
 * （ember.mjs:24576-24578 `this.#elements.timeLabel.innerText = \`${campaignDay} - ${time}\``），
 * 而 animate() 的调用方全是非渲染路径（ember.mjs:3878 时间推进的每一帧、:28978 天气变化），
 * **一次 renderApplicationV2 都不发**。首屏之所以看着是对的，只是因为 _onRender 末尾调了一次
 * animate（:24551）、而 Foundry 的 _doEvent 先跑 handler 再派钩子，我们正好接在后面；
 * 此后任何一次时间推进都会被 animate 用英文覆盖回去，且没有第二次翻译机会。
 *
 * 所以改挂在格式化函数上：`calendar.format(t, "emberDay")` 先查 CONFIG.time.formatters
 * （core client/data/calendar.mjs:198），包住它以后 `Day 43` 这个英文串根本不会产生，
 * 时钟条（:24576）和法典日志的日期表头（:25243）两处一起解决。
 */
function patchCalendarFormatters() {
  const formatters = CONFIG?.time?.formatters;
  const original = formatters?.emberDay;
  if (typeof original !== "function") return warn("找不到 CONFIG.time.formatters.emberDay，日期格式补丁跳过。");
  formatters.emberDay = function (...args) {
    return translateText(original.apply(this, args));
  };
  log("已包住历法的 emberDay 格式化函数。");
}

/**
 * Ember 各类应用（角色卡、任务面板、日历）渲染出来的分节标题与按钮同样是硬编码。
 * 在 renderApplication 之后对根元素做一次 DOM 遍历。
 */
function patchRenderedApplications() {
  const handler = (app, element) => {
    try {
      const root = element instanceof HTMLElement ? element : element?.[0];
      if (!root) return;
      const cls = root.className ?? "";
      const id = app?.constructor?.name ?? "";
      // 只处理 Ember 自己的界面，避免把别的模块的英文也一起改了
      if (!/ember/i.test(cls) && !/^Ember/.test(id)) {
        // 例外一：Ember 的确认框走的是**原生 DialogV2**（根元素 class 只有 "dialog"、
        // 类名就是 "DialogV2"），标题、正文、按钮全是硬编码英文，babele 与 i18n 两条通道
        // 都够不着，而上面那道 ember 闸会把它整个挡掉。
        //
        // 先按窗口标题**认框**：认得出是 Ember 弹的，就连正文和按钮一起翻，用作用域表
        // DIALOG_UI（`Ring` / `Close` / `Change` 这类词不能进全局 EXACT）；认不出来就只翻
        // 标题 —— EXACT 里有 Path / Culture / Events 这类通用词，别的模块的窗口恰好同名会被误改。
        // 认框在改标题**之前**做，所以这段是幂等的：重复渲染时标题已是中文，认不出来也不会再动。
        if (id === "DialogV2" || /(^|\s)dialog(\s|$)/.test(cls)) {
          const title = root.querySelector?.(".window-title");
          const rawTitle = title?.textContent?.trim() ?? "";
          const mine = (rawTitle in DIALOG_TITLES)
            || DIALOG_TITLE_PATTERNS.some(re => re.test(rawTitle))
            || DIALOG_TITLE_I18N.some(key => game.i18n?.localize(key) === rawTitle);
          if (mine) {
            translateNode(root, DIALOG_UI);  // 标题也在这棵树里，一并翻掉
            return;
          }
          if (title && !title.children.length) {
            const t = translateText(title.textContent);
            if (t !== title.textContent) title.textContent = t;
          }
          return;
        }
        // 例外二：Ember 把自己的页签塞进**别人的**窗口 —— crucible HeroSheet 的「同调」页
        // （ember.mjs:124717）。宿主 class 是 crucible 的、类名是 HeroSheet，上面两个判据都
        // 不成立，整张卡在闸外。按注入点选择器单独放行子树，同样只用作用域表，
        // 不会顺手把宿主自己的界面改了。
        for (const [selector, table] of INJECTED_SUBTREES) {
          for (const sub of root.querySelectorAll?.(selector) ?? []) translateNode(sub, table);
        }
        return;
      }
      translateNode(root);
    } catch (err) {
      warn("界面文本翻译失败：", err);
    }
  };
  Hooks.on("renderApplicationV2", handler);
  Hooks.on("renderApplication", handler);
  log("已挂上界面渲染钩子。");
}

/* ============================================================ */
/*  4. 入口                                                      */
/* ============================================================ */

Hooks.once("ready", () => {
  if (!game.modules.get("ember")?.active) return;
  applyOnce(CONFIG, "__emberCnEnrichers", patchEnrichers, "富文本增强器");
  applyOnce(globalThis.crucible?.CONFIG ?? {}, "__emberCnConfig", patchCrucibleConfig, "crucible.CONFIG");
  applyOnce(CONFIG, "__emberCnCalendar", patchCalendarNames, "历法月名");
  applyOnce(CONFIG, "__emberCnDayFormat", patchCalendarFormatters, "历法日期格式");
  applyOnce(CONFIG, "__emberCnRender", patchRenderedApplications, "界面渲染");
  log("Ember 硬编码字符串补丁已就绪。");
});
