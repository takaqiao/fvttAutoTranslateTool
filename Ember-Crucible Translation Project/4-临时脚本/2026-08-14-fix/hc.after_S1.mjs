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
  // 2026-08-13 第十一轮补：这三个语言 [[/language …]] 有调用但表里缺键，缺了会渲染成英文
  "Moiré": "莫伊雷语", "Borel": "博雷尔语", "Kost": "科斯特语",
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
 * `{CALM: "calm", TENSION: "tension"}`，enricher 的**第三支**拼的是 `Music Mood: ${mood.titleCase()}`。
 * ⚠ 2026-08-14 订正：enricher 一共**三支**，另外两支拼的是 `Music: …` / `Environment: …`，
 * 原注释漏掉了它们，详见下面 ARRANGEMENTS 的说明。
 * 原来那五个键（战斗/探索/环境/旅行/休息）在 ember 0.6.x 里一个都不会出现。
 * 译名取 lang/cn.json 的 `EMBER.SoundscapeMoodCalm` / `EMBER.SoundscapeMoodTension`。
 */
const MOODS = {
  "Calm": "平静", "Tension": "紧张"
};

/**
 * 音景「编排」名。`EmberSoundscape.enricherHTML`（ember.mjs:16250-16278）有三个互斥分支：
 *
 *     16255  label = `${dataset.channel.capitalize()}: Reset`                  → `Music: Reset`
 *     16266  label = `${dataset.channel.capitalize()}: ${arrangement.label}`   → `Music: Ankarist Theme`
 *     16271  label = `Music Mood: ${mood.titleCase()}`
 *
 * 前两支不带 `Music Mood: ` 前缀，原来一条都没收。已发布语料实测（两个孪生包形状相同）：
 * `[[/soundscape music reset]]` 7 + lylaTheme 5 + sinTheme 4 + ankaristTheme 3 +
 * pitTrap/intense 1 + ancientRuins/shentRuins 1 = **21 处**落在前两支，`mood=` 只有 2 处。
 * channel 只有 music / environment 两个（ember.mjs:15643）。
 * arrangement.label 是 ember.mjs 里的硬编码常量（5694 / 5787 / 12393 / 14064 / 14748 各 var 块），
 * babele 与 i18n 两条通道都够不到。
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

/**
 * 尾巴不需要查表的前缀共用这张空表：leaf 由 translateLeaf 原样透传。
 * 用在 [[/ancestry]] / [[/culture]] / [[/path]]（尾巴是 babele 已译的合集条目名）
 * 与 hex-hud 的四条 tooltip（尾巴是 babele 已译的地名 / 群系名 / 地形名）上。
 */
const PASSTHROUGH = {};

/** 带前缀的标签：`前缀: 名字` → `中文前缀：中文名字` */
const PREFIXED = [
  { en: "Attunement", cn: "同调", table: ATTUNEMENTS },
  { en: "Language", cn: "语言", table: LANGUAGES },
  { en: "Knowledge", cn: "知识", table: KNOWLEDGE },
  { en: "Music Mood", cn: "音乐氛围", table: MOODS },

  // 角色选项增强器：ember.mjs:22934 `tag.innerHTML = `Ancestry: ${name}`` /
  // :22954 `Culture: ${name}` / :22986 `Path: ${name}`。这三个增强器**永远**吐
  // `前缀: 名字` 复合串，从不单独吐裸词 —— 下面 EXACT 里那三条裸词打的是角色创建卡
  // #STEPS 的 label（ember.mjs:121863 / 121876 / 121882），两个用途形状不同，两张表都要有。
  // 用量：ember.crucible-adventure.json 里 ancestry 127 + culture 111 + path 107 = 345 处。
  { en: "Ancestry", cn: "血统", table: PASSTHROUGH },
  { en: "Culture", cn: "文化", table: PASSTHROUGH },
  { en: "Path", cn: "道途", table: PASSTHROUGH },

  // 音景增强器的前两支（见 ARRANGEMENTS）。顺序无所谓：
  // `"Music Mood: Calm".startsWith("Music: ")` 为假，两条前缀不会互相吃掉。
  { en: "Music", cn: "音乐", table: ARRANGEMENTS },
  { en: "Environment", cn: "环境音", table: ARRANGEMENTS },

  // 六边形 HUD 的四条 data-tooltip（templates/applications/hex-hud.hbs:13 / 47 / 50 / 52）。
  // 宿主 EmberHexHUD 过闸（classes:["ember"]，ember.mjs:25415），data-tooltip 也在
  // translateNode 的属性白名单里 —— 遍历得到，只是查表形状对不上带动态尾巴的串。
  { en: "Area Map", cn: "区域地图", table: PASSTHROUGH },
  { en: "Location", cn: "地点", table: PASSTHROUGH },
  { en: "Biome", cn: "生物群系", table: PASSTHROUGH },
  { en: "Terrain", cn: "地形", table: PASSTHROUGH }
];

/** 完全匹配即可替换的字符串 */
const EXACT = {
  // 角色创建卡 EmberCharacterCreationSheet 的 #STEPS 裸标签
  //（ember.mjs:121863 / 121876 / 121882 的 `label: "Ancestry" | "Culture" | "Path"`）。
  // ⚠ 这**不是**富文本增强器前缀 —— 增强器吐的是 `Ancestry: 名字` 复合串，
  // 整串永远不等于裸词，那一支在 PREFIXED 里（两张表都要有，别再合并）。
  "Ancestry": "血统",
  "Culture": "文化",
  "Path": "道途",

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
  "Reset Event": "重置事件",
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

  // 对话框标题
  "Add to Party?": "加入队伍？",
  "Re-combine Caravans?": "重新合并商队？",
  "Initiate Event": "启动事件",
  "Select Outcome": "选择结果",
  "Delete Saved Composition?": "删除已保存的构图？",
  "Transition to Pathways?": "转入通路？",
  "Ring Alarm Bell?": "敲响警钟？",
  "Modify Flow Control Valve?": "调整流量控制阀？",
  "Mine Cart Destination": "矿车目的地",
  "Install Junction Wheel": "安装路口轮盘",
  "Elevator Controls": "升降机控制",
  "Elevator Destination": "升降机目的地",
  "Steam Cleansing Cutoff": "蒸汽净化切断",
  "Unspent Ability Points": "未分配的属性点",
  "Apply Soulbound Progression": "应用魂缚进程",

  // crucible 英雄卡的「同调」页签（templates/crucible/tab-attunement.hbs:4 与 :38）。
  // 这块是 Ember splice 进 crucible 自己的 HeroSheet 的，走 patchRenderedApplications
  // 里新加的「外来宿主注入块」那条路才够得到；11 个月名与激活标签在
  // translateAttunementPart 里单独处理，不进本表。
  "Cosmological Attunements": "宇宙同调",
  "Make Active": "设为激活",

  // 注入 Playlists 侧栏的「Ember 音乐」控制面板
  //（ember.mjs:15886 / 15889 / 15892，与 #updateSoundscapeForm 的 15929 / 15934 两个 blank 项）
  "Ember Music": "余烬音乐",
  "Ember Environment": "余烬环境音",
  "Ember Default": "余烬默认",
  "Rearrange Music": "重排乐曲"
};

/**
 * 对话框正文、按钮与场景交互物的窗口标题。
 *
 * **单独一张表，刻意不并进 EXACT**：这些串要在「非 Ember 宿主的 DialogV2」分支里查，
 * 而那条分支会碰到别的模块的对话框；EXACT 里有 Path / Culture / Events / Locations
 * 这类通用词，混进去就会误改别人的窗口（原注释担心的正是这一点）。
 *
 * 为什么必须做：Ember 的确认框走原生 DialogV2，`content` 是硬编码英文 HTML 串，
 * babele 与 i18n 都够不到。按钮 label 与窗口标题**本来**是 i18n 槽
 *（core dialog.mjs:249 `span.innerText = _loc(label)`、application.mjs:319
 * `get title(){ return _loc(this.options.window.title) }`），但 Ember 全写了裸英文，
 * 所以这里一并按 DOM 文本兜住。原先这条分支只取 `.window-title` 译一行就 return，
 * 于是玩家看到「中文标题 + 英文正文 + 英文按钮」的半截框。
 *
 * ⚠ 查表前把串内空白折成单个空格（translateWith 负责），因为上游好几处 content 是
 * 跨行模板字符串，换行加缩进会原样进 DOM。所以下面的键一律写成单行。
 */
const DIALOG_TEXT = {
  // ── 正文（EmberInteractable 家族的 static DEFAULT_CONFIG.dialog.content）
  "Activate this elevator?": "启动这台升降机？",
  "Activate the hidden Generator Room switch?": "启动隐藏的发电机室开关？",
  "This lever has already been activated.": "这根拉杆已经被扳动过了。",
  "Restore or disable power to the Stealth Field Generator?": "恢复还是切断隐形立场发生器的电力？",
  "Set the machine's operating state.": "设定这台机器的运行状态。",
  "Activate the force-field control orb?": "启动力场控制球？",
  "Direct this elevator to a destination.": "为这台升降机指定目的地。",

  // ── 正文（散落的 DialogV2 调用点）
  "Do you wish to recombine the Party into the Strayhearth Caravan?": "你希望把队伍重新并入迷炉商队 Strayhearth Caravan 吗？",
  "Are you sure you want to completely clear this vista composition?": "确定要彻底清空这幅远景构图吗？",
  "The junction wheel is missing. Install a replacement?": "路口轮盘缺失，是否安装一个替换件？",
  "Resetting the event step for this event may introduce critical errors into your Ember game state. Are you sure you wish to proceed?": "重置此事件的事件步骤可能会给你的余烬 Ember 游戏状态引入严重错误。确定要继续吗？",
  "Beginning this event may introduce critical errors into your Ember game state. Are you sure you wish to proceed?": "开始此事件可能会给你的余烬 Ember 游戏状态引入严重错误。确定要继续吗？",
  "No destinations are currently reachable. Adjust the track levers and try again.": "当前没有可到达的目的地。请调整轨道拉杆后重试。",
  "Activate this mine cart with no passenger?": "在没有乘客的情况下启动这辆矿车？",
  "Do you wish to proceed and forego these increases?": "你希望继续并放弃这些提升吗？",

  // ── 矿车目的地对话框的分组 legend（ember.mjs:112021 buildGroup 造 <legend>）
  "Forwards": "向前",
  "Backwards": "向后",
  "Unreachable": "无法到达",

  // ── 按钮。`Interact` 是 _configureDialog（ember.mjs:62794）的兜底 ok 标签，
  //    出现在**每一个**未显式声明 buttons 的交互物上，是全冒险被点得最多的英文按钮。
  "Interact": "交互",
  "Cancel": "取消",
  "Close": "关闭",
  "Confirm": "确认",
  "Move": "平移",
  "Ascend": "上升",
  "Descend": "下降",
  "Seal": "封闭",
  "Disable Generator": "关闭发生器",
  "Restore Power": "恢复供电",
  "Close Valve": "关闭阀门",
  "Open Valve": "打开阀门",
  "Machine On": "启动机器",
  "Machine Off": "关闭机器",
  "Machine Destroyed": "机器已损毁",
  "Defenses Inactive": "防御未启动",
  "Defenses Active": "防御已启动",
  "Orb Destroyed": "法珠已损毁",
  "Broken": "已破损",
  "Damaged": "已受损",
  "Repaired": "已修复",
  "Repair": "修复",
  "Reset Vault": "重置宝库",
  "Activate Beams": "启动光束",
  "Ring": "敲响",
  "Destroy": "摧毁",
  "Enable Flow": "开启水流",
  "Disable Flow": "截断水流",
  "Activate": "启动",
  "Tradeway": "贸易道 Tradeway",
  "Underbelly": "底腹区 Underbelly",
  "Construct Assembly": "构装体装配区 Construct Assembly",
  "Reset Body": "重置躯体",
  "Awaken Vampyre": "唤醒吸血孽裔",
  "Engage Lockdown": "启动封锁",
  "Lift Lockdown": "解除封锁",

  // ── 场景交互物的窗口标题（EXACT 里那 15 条对话框标题只覆盖了本族 4 条）
  "Aedir Signalpost Generator Room Switch": "艾迪尔信号哨站 发电机室开关",
  "Aedir Signalpost Stealth Field Generator": "艾迪尔信号哨站 隐形立场发生器",
  "Awaken Vampyre Body?": "唤醒吸血孽裔的躯体？",
  "Bastion Apex: Barrier Pillar": "堡垒顶点 Bastion Apex：屏障之柱",
  "Bastion Apex: Orb of Lantyr": "堡垒顶点 Bastion Apex：兰提尔法珠",
  "Bleak Archive Light Beams": "黯淡秘库 光束",
  "Construct Elevator": "构装体升降机",
  "Dredging Valve": "疏浚阀",
  "Forcefield Control Orb": "力场控制球",
  "Machine": "机器",
  "Redwalk Ramble - Illusion Control": "红行漫步园 - 幻象控制",
  "Silver Beam Security Control": "银光束 安保控制",
  "Temple Lunarium": "神殿月辉宫",
  "Vortest Tower Transporter": "沃特斯特塔 传送器",
  "Clear Vista": "清空远景"
};

/** 掷骰结果档位。Ember 用 `Result of X` 的形式作为结局标题 */
const RESULTS = {
  "Success": "成功", "Failure": "失败",
  "Critical Success": "大成功", "Critical Failure": "严重失败"
};

/**
 * 历法纪元名。`[[/date …]]` 增强器把 tooltip 写进 `data-tooltip`：
 * `EmberCalendar.parseDate`（ember.mjs:4130-4134）拼
 * ``${resolvedAge.label} - ${yearsAgo ? `${yearsAgo} ${relativeLabel}` : relativeLabel}``，
 * 增强器再 `Object.assign(span.dataset, {year, tooltip})`（ember.mjs:129416）。
 * `resolvedAge.label` 取自 dnd5e-async.mjs:145-150 的 CALENDAR_AGES，是硬编码英文。
 *
 * 通道本来就是通的（patchEnrichers 的白名单正则含 `date`，translateNode 的属性白名单含
 * `data-tooltip`）—— 缺的一直是表项。内联可见的 `date.label`（AT24571 这类纪元缩写）
 * 按本文件开头的约定有意保留英文，不在此列。
 * 用量：ember.crucible-adventure.json 里 `[[/date` 111 处。
 * 译名与 compendium 对齐：高塔时代 Age of the Tower / 野兽时代 Age of Beasts /
 * 大破裂 The Shattering。
 */
const CALENDAR_AGES = {
  "Age of Creation": "创世时代",
  "Age of Beasts": "野兽时代",
  "Age of the Tower": "高塔时代",
  "After Shattering": "大破裂之后"
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
  // `[[/date …]]` 的 data-tooltip：`${age.label} - ${yearsAgo} ${relativeLabel}`（ember.mjs:4134）。
  // yearsAgo 为 0 时上游只拼 `${age.label} - Current Year`；为负时会拼出
  // `… - -5 Years From Now`（上游把负数直接插进去了），正则一并吃掉。
  // 尾巴必须整段等于那三个短语之一，所以不会误伤别的「X - Y」形状的串。
  {
    re: /^(.+?) - (?:(-?\d+) )?(Current Year|Years Ago|Years From Now)$/,
    cn: (m) => {
      const age = CALENDAR_AGES[m[1]] ?? m[1];
      if ( m[3] === "Current Year" ) return `${age} - 本年`;
      return `${age} - ${Math.abs(Number(m[2]))} 年${m[3] === "Years Ago" ? "前" : "后"}`;
    }
  }
];

/**
 * Ember 历法的月名与星期名。
 *
 * ⚠ 2026-08-14 订正 —— 原注释在这里写着「日历条上显示的日期串由 formatEmberDate 用**月名**
 * 拼出来，所以 lang/cn.json 里把 `EMBER.CALENDAR.SEASONS.*` 全译了也没用」。**因果正好反了**，
 * 照它推理会改错地方，最坏情况是有人据此删掉那 6 个 SEASONS 键、世界时钟当场变回英文。
 * ember 0.6.0 + Foundry v14.365 逐行核对的实况：
 *
 *     months:  { values: [{name: "Blooming"}, …] }                       ← 裸英文，但**没有读者**
 *     seasons: { values: [{name: "EMBER.CALENDAR.SEASONS.BLOOMING"}, …] } ← i18n 键，日期串读的是它
 *
 *   - `EmberCalendar.formatEmberDate`（ember.mjs:4063）返回的是
 *     ``${dayOfMonth+1} ${_loc(season.name)}, ${abbr}${ageYear}`` —— 取 **season**，走 i18n；
 *     日历条 `EmberCalendarNavigation.animate()`（ember.mjs:24595）正是用它填 dateLabel。
 *     **所以世界时钟读的是 `lang/cn.json` 的 `EMBER.CALENDAR.SEASONS.*`，那 6 个键一个都不能删。**
 *   - `months` 在整个 ember.mjs 里只出现 1 次（3626 行配置字面量自身），`days.values` 0 次；
 *     core 唯一读 months 的是 `client/data/calendar.mjs:386 formatTimestamp`，取的是
 *     `month.ordinal`（数字）不是 `name`，`client/applications/` 下 0 命中。
 *     下面两张表因此**当前零读者**，`patchCalendarNames` 打印的「已改写 N 个」
 *     只说明数据被改到了，不代表界面上有任何变化。保留是备上游改用月名，
 *     不要反过来把 SEASONS 的 i18n 键当成多余的。
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
export function translateText(text) {
  if (typeof text !== "string") return text;
  const raw = text.trim();
  if (!raw) return text;

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

/** 递归翻译一棵 DOM 子树里的所有文本节点与 tooltip 属性 */
function translateNode(node) {
  if (!node) return;
  if (node.nodeType === Node.TEXT_NODE) {
    const t = translateText(node.nodeValue);
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
      const t = translateText(v);
      if (t !== v) node.setAttribute(attr, t);
    }
  }
  for (const child of Array.from(node.childNodes)) translateNode(child);
}

/**
 * 按**指定的一张表**翻一棵子树（文本节点 + tooltip 属性）。
 * 与 translateNode 有两点不同：
 *   ① 只查传进来的表，不碰 EXACT / PREFIXED / PATTERNS —— 用在会碰到别的模块界面的
 *      DialogV2 分支上，避免 EXACT 里的通用词误伤；
 *   ② 比对前把串内空白折成单个空格 —— 上游好几处 `content` 是跨行模板字符串，
 *      换行与缩进会原样进 DOM，按字面比对必然落空。
 * @param {Node} node
 * @param {Record<string, string>} table
 */
function translateWith(node, table) {
  if (!node) return;
  if (node.nodeType === Node.TEXT_NODE) {
    const raw = node.nodeValue.trim();
    if (!raw) return;
    const cn = table[raw.replace(/\s+/g, " ")];
    if (cn) node.nodeValue = node.nodeValue.replace(raw, cn);
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return;
  for (const attr of ["data-tooltip", "data-tooltip-text", "title", "aria-label"]) {
    const v = node.getAttribute?.(attr);
    const cn = v && table[v.trim().replace(/\s+/g, " ")];
    if (cn) node.setAttribute(attr, cn);
  }
  for (const child of Array.from(node.childNodes)) translateWith(child, table);
}

/**
 * crucible 英雄卡的「同调」页签。
 *
 * 这块界面是 Ember 塞进 **crucible 自己的** HeroSheet 的：
 * `addAttunementTab()`（ember.mjs:124717）往 `cls.TABS.sheet.tabs` splice 一项，
 * 并设 `cls.PARTS.attunement = {id: "attunement", template: "modules/ember/templates/crucible/tab-attunement.hbs"}`。
 * 宿主类名是 `HeroSheet`、根 class 是 ["crucible","actor","standard-form",…]，
 * 两道闸都不放行 —— 只能按 core 打在 part 根元素上的 `data-application-part`
 *（client/applications/api/handlebars-application.mjs:175）单独捞出来。
 *
 * 页内 11 个 `{{attunement.label}}`（hbs:13 与 :16）走的是 dnd5e-async.mjs:405-417 的
 * 硬编码短名 Abyss/Akon/Aura/Cora/Heart/Luxarum/Mayis/Orbis/Primordis/Ragen/Signara：
 * `initializeAttunementOptions()`（ember.mjs:129702-129719）只把 `name` 覆盖成已译的
 * Cosmos 页名，**`label` 原封不动**，而模板取的正是 label，所以只能按 ATTUNEMENTS 逐个换。
 * 激活标签 `"Active"`（ember.mjs:124757 `active ? "Active" : ""`）在 ember 里只此一处，
 * 就地处理，不进 EXACT，免得这个通用词从 DialogV2 分支漏到别的模块去。
 * @param {HTMLElement} part
 */
function translateAttunementPart(part) {
  translateNode(part);
  for (const li of part.querySelectorAll("li.attunement")) {
    const h4 = li.querySelector(".title h4");
    if (h4) {
      const cn = ATTUNEMENTS[h4.textContent.trim()];
      if (cn) h4.textContent = cn;
    }
    const img = li.querySelector("img.icon");
    const alt = img?.getAttribute("alt")?.trim();
    if (alt && ATTUNEMENTS[alt]) img.setAttribute("alt", ATTUNEMENTS[alt]);
    for (const tag of li.querySelectorAll(".tags .tag")) {
      if (tag.textContent.trim() === "Active") tag.textContent = "已激活";
    }
  }
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
    // 只包 Ember 自己的增强器，别去动 crucible 与 Foundry 本体的
    if (!/attunement|language|knowledge|soundscape|ancestry|culture|path|eventState|outcome|Advantage|Critical|date/i.test(src)) continue;
    const original = entry.enricher;
    entry.enricher = async function (...args) {
      const result = await original.apply(this, args);
      try {
        if (result instanceof HTMLElement) translateNode(result);
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
  log(`已改写历法里 ${n} 个月名/星期名（备用路径：ember 0.6.0 无读者，日期串走 EMBER.CALENDAR.SEASONS.*）。`);
}

/**
 * 世界时钟的「第 N 天」与月亮浮窗 —— 这两处不能靠 DOM 补丁。
 *
 * `EmberCalendarNavigation#animate()` **逐帧直写 DOM**：
 *   ember.mjs:24576  `this.#elements.timeLabel.innerText = `${campaignDay} - ${time}``
 *   ember.mjs:24628  `m.dataset.tooltip = `${moon.name} ${moon.phaseLabel}``
 * 而 animate() 挂在 `CanvasAnimation.animate(…, {ontick})` 上（ember.mjs:3874-3879），
 * 完全不经过 ApplicationV2 的渲染生命周期。渲染时序上 `_onRender` 先调 animate()（24551）
 * 再触发 render 钩子，所以首帧确实是中文 —— 但只要世界时间动一次就永久写回英文，
 * `renderApplicationV2` 不会有第二次机会。所以要堵在源头：
 *
 *   1. `Day N` 由 `CONFIG.time.formatters.emberDay` 产出（ember.mjs:129028 注册），
 *      core `CalendarData#format`（client/data/calendar.mjs:198）**每次调用都从 CONFIG 里取**，
 *      包一层即一劳永逸；顺带也覆盖法典日志表头（ember.mjs:25243 用的是同一个 formatter）。
 *      产出的 `Day 43` 交给 PATTERNS 的 `^Day (\d+)\b(.*)$` 变成「第 43 天」。
 *   2. 月名是 `EmberMoon` 实例上的裸英文（ember.mjs:52821 起 Aura/Cora/Ragen/Mayis/Orbis/
 *      Akon/Signara/Luxarum/Primordis）。`EmberCalendar#initialize()`（ember.mjs:3769）
 *      只在 setup 钩子里跑一次，ready 时改实例不会被重建覆盖；`moon.name` 在 ember.mjs 里
 *      只有 24628 一个读者，改它不影响贴图与光照（那些走 `moon.texture` / `moon.id`）。
 *      译名与 ATTUNEMENTS 同一套（月亮与同调是同一批专名）。
 */
function patchTimeFormatters() {
  const formatters = CONFIG?.time?.formatters;
  const original = formatters?.emberDay;
  if (typeof original !== "function") return warn("找不到 CONFIG.time.formatters.emberDay，日期格式补丁跳过。");
  formatters.emberDay = (...args) => translateText(original(...args));
  let n = 0;
  try {
    for (const moon of Object.values(globalThis.ember?.calendar?.moons ?? {})) {
      const cn = ATTUNEMENTS[moon?.name];
      if (cn) { moon.name = cn; n++; }
    }
  } catch (err) {
    warn("月亮名改写失败（日期格式已生效）：", err);
  }
  log(`已接管 emberDay 日期格式，并改写 ${n} 个月亮名。`);
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

        // 例外一：Ember 会把**整块界面注入别人的宿主**。注入块自己带 ember 标记，
        // 但上面那道闸只测宿主**根元素**的 className，看不见后代，于是整块永不进 translateNode。
        // 现有两处（ember/templates 下 68 个 .hbs 按宿主枚举过，只有这两个落在外来宿主上）：
        //   · Playlists 侧栏的「Ember 音乐」面板 —— ember.mjs:15883 往
        //     `#playlists .currently-playing` 前插 `<form id="ember-mood" class="ember global-control …">`；
        //   · crucible HeroSheet 的「同调」页签 —— ember.mjs:124725 塞 PARTS，
        //     core 会给 part 根元素打上 `data-application-part="attunement"`。
        // 时序没问题：v14 的 #callHooks 沿 inheritanceChain 从最派生类往上叫
        //（client/applications/api/application.mjs:1726），renderPlaylistDirectory / renderHeroSheet
        // 都先于 renderApplicationV2 触发，我们跑的时候注入块已经在 DOM 里了。
        for (const injected of root.querySelectorAll?.(".ember") ?? []) translateNode(injected);
        const attunement = root.querySelector?.('[data-application-part="attunement"]');
        if (attunement) translateAttunementPart(attunement);

        // 例外二：Ember 的确认框走的是**原生 DialogV2**（根元素 class 只有 "dialog"、
        // 类名就是 "DialogV2"），标题、正文与按钮全是硬编码英文，babele 与 i18n 两条通道
        // 都够不着，而上面那道 ember 闸会把它整个挡掉。
        // 闸只放行 DialogV2 这一档，不放行全世界的 ApplicationV2 ——
        // EXACT 里有 Path / Culture / Events 这类通用词，别的模块的窗口标题恰好同名就会被误改。
        if (id !== "DialogV2" && !/(^|\s)dialog(\s|$)/.test(cls)) return;
        const title = root.querySelector?.(".window-title");
        if (title && !title.children.length) {
          const t = translateText(title.textContent);
          if (t !== title.textContent) title.textContent = t;
        }
        // 正文与按钮：原来译完标题就 return，`.dialog-content` 与 `.form-footer button`
        // 从不被遍历，于是玩家看到「中文标题 + 英文正文 + 英文按钮」的半截框。
        // 这一趟只查 DIALOG_TEXT（本族专有的整句、按钮名与窗口标题），不查 EXACT，理由同上。
        translateWith(root, DIALOG_TEXT);
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
  applyOnce(CONFIG, "__emberCnFormatters", patchTimeFormatters, "世界时钟日期格式");
  applyOnce(CONFIG, "__emberCnRender", patchRenderedApplications, "界面渲染");
  log("Ember 硬编码字符串补丁已就绪。");
});
