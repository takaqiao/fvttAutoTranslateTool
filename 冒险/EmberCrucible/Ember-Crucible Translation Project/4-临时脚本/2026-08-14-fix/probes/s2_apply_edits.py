#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 编辑清单：构造 -> 校验唯一命中 -> 试打补丁 -> node --check。不改原文件。"""
import io, json, os, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")

SRC = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
REL = "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_patched.mjs")

E = []


def add(sig, old, new, why):
    E.append({"sig": sig, "file": REL, "old": old, "new": new, "why": why})


# ------------------------------------------------------------------ #
# 1. LANGUAGES 里的三个死键
# ------------------------------------------------------------------ #
add(
    "dead-key|LANGUAGES/Moire-Borel-Kost",
    '''const LANGUAGES = {
  // 2026-08-13 第十一轮补：这三个语言 [[/language …]] 有调用但表里缺键，缺了会渲染成英文
  "Moiré": "莫伊雷语", "Borel": "博雷尔语", "Kost": "科斯特语",
  "Common": "通用语",''',
    '''const LANGUAGES = {
  "Common": "通用语",''',
    "这张表按 `language.label` 查（enrichLanguage 拼的是 `Language: ${language.label}`），"
    "而 crucible.CONFIG.languages（ember.mjs:126693 那张 23 条表）里压根没有 borel/kost/moiré 三个 id，"
    "它们连 label 都不存在 —— 三个键永远等不到 `Language: Borel` 这种输入。挪到按 id 兜底的 MISSING_LANGUAGES。",
)

add(
    "dead-key|LANGUAGES/Moire-Borel-Kost",
    '''  "Thieves' Cant": "盗贼黑话"
};''',
    '''  "Thieves' Cant": "盗贼黑话"
};

/**
 * `[[/language …]]` 引用了、但**上游根本没有**的语言 id。
 *
 * crucible.CONFIG.languages（ember.mjs:126693 起那张 23 条表）里没有 borel / kost，
 * 于是 enrichLanguage 走 `if (!language) return new Text(match)`（ember.mjs:126542），
 * 正文原样吐出字面量 `[[/language borel]]`。合集实测：borel×2、kost×1（孪生包各一份）。
 * 这里按 **id** 兜底，配合 PATTERNS 末尾那条 `^\\[\\[\\/language …]]$` 把裸标记换成中文；
 * 能生效的前提是增强器包装那边用 `result instanceof Node`（Text 节点也要收），见 patchEnrichers。
 *
 * `moiré` 拿不到这个入口：增强器的 pattern 是 `(\\w+)`、无 u 标志，é 不算 \\w，
 * 连增强器都不会被调用，字面量停在正文里 —— 那两处只能在 compendium 译文里改掉。
 */
const MISSING_LANGUAGES = {
  "borel": "博雷尔语",
  "kost": "科斯特语"
};''',
    "把 Moiré/Borel/Kost 从「按 label 查」的死键改成「按 id 兜底」的活键，"
    "并把 moiré 为什么修不了写进注释，免得下一轮又照着表补一遍。",
)

# ------------------------------------------------------------------ #
# 2. PREFIXED 缺 Ancestry / Culture / Path / Talent
# ------------------------------------------------------------------ #
add(
    "闸/选择器失配|PREFIXED缺Ancestry/Culture/Path三条前缀",
    '''/** 带前缀的标签：`前缀: 名字` → `中文前缀：中文名字` */
const PREFIXED = [
  { en: "Attunement", cn: "同调", table: ATTUNEMENTS },
  { en: "Language", cn: "语言", table: LANGUAGES },
  { en: "Knowledge", cn: "知识", table: KNOWLEDGE },
  { en: "Music Mood", cn: "音乐氛围", table: MOODS }
];''',
    '''/** 带前缀的标签：`前缀: 名字` → `中文前缀：中文名字` */
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
  { en: "Talent", cn: "天赋", table: {} }
];''',
    "三个增强器从不单独输出前缀，永远是 `前缀: 名字` 一个文本节点；EXACT 要求全等、PREFIXED 又没登记，"
    "于是 Crucible 侧 345 处（孪生包另 343 处）全部漏成「Ancestry: 人类 Human」。"
    "crucible 的 [[/talent]] 68 处同构。补 PREFIXED 是唯一能命中的通道。",
)

# ------------------------------------------------------------------ #
# 3. EXACT 头部：注释事实订正 + 创建向导缺的两步
# ------------------------------------------------------------------ #
add(
    "missing-key|EXACT/creation-steps-Attunement-Token",
    '''/** 完全匹配即可替换的字符串 */
const EXACT = {
  // 富文本增强器前缀（单独出现时）
  "Ancestry": "血统",
  "Culture": "文化",
  "Path": "道途",
''',
    '''/**
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
''',
    "① 创建向导顶栏 attunement/token 两步没键（crucible-async.mjs:44/63），顶栏混排成"
    "「血统/文化/道途/Attunement/天赋/装备/Token」；② 原注释断言这三个裸词来自增强器前缀，"
    "事实相反，注释直接导致 PREFIXED 一直没补；③ 顺带把 DIALOG_TITLES/DIALOG_UI 等作用域表定义在 EXACT 之前"
    "（EXACT 要 spread DIALOG_TITLES，必须先定义）。",
)

# ------------------------------------------------------------------ #
# 4. "Reset Event" 挪进 DIALOG_TITLES
# ------------------------------------------------------------------ #
add(
    "gate-scope|dialogv2-body-and-buttons-outside-title-only-branch",
    '''  "Begin Event": "开始事件",
  "Reset Event": "重置事件",
  "Complete Event": "完成事件",''',
    '''  "Begin Event": "开始事件",
  // "Reset Event" 挪进了 DIALOG_TITLES —— 它同时是 ember.mjs:36938 那个确认框的标题，
  // 认框要靠它；DIALOG_TITLES 已 spread 进本表，事件页上的按钮照旧命中。
  "Complete Event": "完成事件",''',
    "避免 EXACT 里出现重复键：Reset Event 既是事件页按钮又是确认框标题，统一放在 DIALOG_TITLES，"
    "由 spread 保证两个surface 都能命中。",
)

# ------------------------------------------------------------------ #
# 5. EXACT 尾部：对话框标题段 -> spread + 应用窗口/页脚 + spell tooltip
# ------------------------------------------------------------------ #
add(
    "i18n-keyless-literal|ember non-interactable DialogV2 title+footer",
    '''  // 对话框标题
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
  "Apply Soulbound Progression": "应用魂缚进程"
};''',
    '''  // 对话框标题统一收在 DIALOG_TITLES（那张表同时是「这是不是 Ember 弹的框」的识别依据），
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
};''',
    "把 15 条对话框标题收进 DIALOG_TITLES 并 spread 回来（认框需要），"
    "同时补上 9 个 Ember 应用窗口标题/页脚按钮与 crucible 法术标签的 TODO tooltip —— "
    "这些都在 i18n 通道上却没有键，四张覆盖表逐串相减后全部落空。",
)

# ------------------------------------------------------------------ #
# 6. PATTERNS 补充
# ------------------------------------------------------------------ #
add(
    "gate-selector-mismatch|dialogv2-body-and-buttons",
    '''  { re: /^Day (\\d+)\\b(.*)$/, cn: (m) => `第 ${m[1]} 天${m[2]}` }
];''',
    '''  { re: /^Day (\\d+)\\b(.*)$/, cn: (m) => `第 ${m[1]} 天${m[2]}` },

  // 上游没有 borel / kost 这两个语言 id，enrichLanguage 直接 `return new Text(match)`
  // （ember.mjs:126542），正文原样吐出裸标记 `[[/language borel]]`
  { re: /^\\[\\[\\/language (\\w+)]]$/, cn: (m) => `语言：${MISSING_LANGUAGES[m[1]] ?? m[1]}` },

  // 动态拼出来的窗口标题
  { re: /^Interactable: (.+)$/, cn: (m) => `可交互物：${m[1]}` },          // ember.mjs:62795 兜底标题
  { re: /^Token Maker Part Usage: (.+)$/, cn: (m) => `令牌制作器部件用量：${m[1]}` }, // ember.mjs:49557

  // 对话框正文里带插值的整句。都是长句，进全局表不会误伤别的模块。
  { re: /^Are you sure you wish to proceed and delete the "(.+)" composition\\? This cannot be undone\\.$/,
    cn: (m) => `确定要删除「${m[1]}」构图吗？此操作无法撤销。` },                       // ember.mjs:34489
  { re: /^Activate this mine cart with (.+) as its passenger\\?$/,
    cn: (m) => `以 ${m[1]} 为乘客启动这辆矿车？` },                                  // ember.mjs:112067
  { re: /^There are downstream events of (.+) which have been started or completed\\.$/,
    cn: (m) => `${m[1]} 存在已开始或已完成的下游事件。` },                             // ember.mjs:36934
  { re: /^The (.+) event is not currently available because its prerequisites are not satisfied\\.$/,
    cn: (m) => `${m[1]} 事件当前不可用，其前置条件尚未满足。` },                        // ember.mjs:36951
  { re: /^Do you want to (complete this event and )?transition the Party to the Pathways section of the Region map\\?$/,
    cn: (m) => `是否${m[1] ? "完成此事件并" : ""}将队伍转移到区域地图的通路 Pathways 区段？` } // ember.mjs:99653
];''',
    "补三类：① 上游缺 id 的语言裸标记；② 两个动态窗口标题；③ 五条带插值、EXACT 接不住的对话框正文。"
    "全是长句或带冒号前缀的整串，放全局 PATTERNS 不会误伤。",
)

# ------------------------------------------------------------------ #
# 7. 引擎：translateText 作用域表
# ------------------------------------------------------------------ #
add(
    "gate-selector-mismatch|dialogv2-body-and-buttons",
    '''export function translateText(text) {
  if (typeof text !== "string") return text;
  const raw = text.trim();
  if (!raw) return text;

  if (raw in EXACT) return text.replace(raw, EXACT[raw]);''',
    '''export function translateText(text, extra = null) {
  if (typeof text !== "string") return text;
  const raw = text.trim();
  if (!raw) return text;

  // extra 是「只在某棵子树里生效」的作用域表：已认出的 Ember 对话框、Ember 注入到
  // 别人窗口里的页签。里头装的是 Close / Ring / Change / Active / Actor 这种太通用、
  // 进了全局 EXACT 就会误伤别的模块的词，只有确认过归属之后才查它。
  if (extra && (raw in extra)) return text.replace(raw, extra[raw]);

  if (raw in EXACT) return text.replace(raw, EXACT[raw]);''',
    "给替换引擎加一层作用域，才能在不污染全局 EXACT 的前提下翻对话框按钮与同调页签里的通用词。",
)

# ------------------------------------------------------------------ #
# 8. 引擎：translateNode 作用域 + 折叠空白重试
# ------------------------------------------------------------------ #
add(
    "gate-selector-mismatch|dialogv2-body-and-buttons",
    '''/** 递归翻译一棵 DOM 子树里的所有文本节点与 tooltip 属性 */
function translateNode(node) {
  if (!node) return;
  if (node.nodeType === Node.TEXT_NODE) {
    const t = translateText(node.nodeValue);
    if (t !== node.nodeValue) node.nodeValue = t;
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return;''',
    '''/**
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
      // 例如 ember.mjs:36934 那段 `…game state. \\n            Are you sure…`。
      // 折叠内部空白后再查一次；命中就把 trim 后的整段换掉（首尾空白保留）。
      const flat = node.nodeValue.trim().replace(/\\s+/g, " ");
      const c = translateText(flat, extra);
      if (c !== flat) t = node.nodeValue.replace(node.nodeValue.trim(), c);
    }
    if (t !== node.nodeValue) node.nodeValue = t;
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return;''',
    "作用域表要一路传下去；顺带解决模板字符串跨行导致「整句在表里但文本节点带内部换行」永不命中的问题。",
)

add(
    "gate-selector-mismatch|dialogv2-body-and-buttons",
    '''    const v = node.getAttribute?.(attr);
    if (v) {
      const t = translateText(v);
      if (t !== v) node.setAttribute(attr, t);
    }
  }
  for (const child of Array.from(node.childNodes)) translateNode(child);
}''',
    '''    const v = node.getAttribute?.(attr);
    if (v) {
      const t = translateText(v, extra);
      if (t !== v) node.setAttribute(attr, t);
    }
  }
  for (const child of Array.from(node.childNodes)) translateNode(child, extra);
}''',
    "属性与递归也要带上作用域表，否则 tab-attunement.hbs:38 那个 aria-label=Make Active 之类够不到。",
)

# ------------------------------------------------------------------ #
# 9. 同调页签作用域表 + 注入点登记
# ------------------------------------------------------------------ #
add(
    "gate-mismatch|crucible-HeroSheet-attunement-tab",
    '''const CALENDAR_DAY_ABBR = {
  "Mon": "一", "Tues": "二", "Wed": "三", "Thu": "四",
  "Fri": "五", "Sat": "六", "Sun": "日"
};''',
    '''const CALENDAR_DAY_ABBR = {
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
];''',
    "同调页签的 14 条英文既不是 i18n key、也不是 compendium 文档、宿主又过不了闸，"
    "三条通道全灭。加一张按选择器放行的注入点登记表，是唯一能够到的办法。",
)

# ------------------------------------------------------------------ #
# 10. 增强器闸 + Text 节点
# ------------------------------------------------------------------ #
add(
    "gate-selector|enricher-pattern-gate|crucibleTalent+crucibleSpell",
    '''    const src = String(entry.pattern ?? "");
    // 只包 Ember 自己的增强器，别去动 crucible 与 Foundry 本体的
    if (!/attunement|language|knowledge|soundscape|ancestry|culture|path|eventState|outcome|Advantage|Critical|date/i.test(src)) continue;''',
    '''    const src = String(entry.pattern ?? "");
    // 按 pattern 里的关键词点名放行。注意这道闸事实上**也会**包住 crucible 自己的几个增强器
    // （crucibleKnowledge / crucibleLanguage / crucibleTalent / crucibleSpell）—— 这是故意的：
    // crucible 的 enrichTalent（crucible-compiled.mjs:46838）拼的是裸模板 `Talent: ${name}`，
    // enrichSpell（:46724）把 "Spell tooltips are still TO-DO." 写进 data-tooltip，两处都不走
    // _loc（相邻的 enrichKnowledge / enrichLanguage 走了），而 crucible 汉化插件那边只有
    // babele-register.js、没有运行时字符串层，只能在这里兜。
    if (!/attunement|language|knowledge|soundscape|ancestry|culture|path|talent|spell|eventState|outcome|Advantage|Critical|date/i.test(src)) continue;''',
    "闸的关键词表里没有 talent / spell，crucibleTalent 的 pattern `[[/talent ([\\\\w-.]+)]]` 与 "
    "crucibleSpell 的 `@Spell\\\\[([\\\\w.]+)]` 一个词都不含，包装从不发生 —— "
    "68 处「Talent: 中文名」与 81 处英文 TODO tooltip 全部外泄。",
)

add(
    "dead-key|LANGUAGES/Moire-Borel-Kost",
    '''        if (result instanceof HTMLElement) translateNode(result);
        else if (typeof result === "string") return translateText(result);''',
    '''        // 判据用 Node 而不是 HTMLElement：增强器解析不出目标时返回的是 `new Text(match)`
        // （crucible-compiled.mjs:46815 / ember.mjs:126542），那是 Text 节点、不是 HTMLElement，
        // 原来这一支直接漏过去，正文就把 `[[/language borel]]` 这种裸标记原样吐给玩家。
        if (result instanceof Node) translateNode(result);
        else if (typeof result === "string") return translateText(result);''',
    "上游 enrich* 在目标不存在时返回 Text 节点，旧判据接不住，裸标记外泄。",
)

# ------------------------------------------------------------------ #
# 11. 渲染钩子：DialogV2 认框 + 注入子树
# ------------------------------------------------------------------ #
add(
    "gate-scope|dialogv2-body-and-buttons-outside-title-only-branch",
    '''      // 只处理 Ember 自己的界面，避免把别的模块的英文也一起改了
      if (!/ember/i.test(cls) && !/^Ember/.test(id)) {
        // 例外：Ember 的十五个确认框走的是**原生 DialogV2**（根元素 class 只有 "dialog"、
        // 类名就是 "DialogV2"），标题是硬编码英文、babele 与 i18n 两条通道都够不着，
        // 而上面那道 ember 闸会把它整个挡掉。
        // 闸只放行 DialogV2 这一档，不放行全世界的 ApplicationV2 ——
        // EXACT 里有 Path / Culture / Events 这类通用词，别的模块的窗口标题恰好同名就会被误改。
        if (id !== "DialogV2" && !/(^|\\s)dialog(\\s|$)/.test(cls)) return;
        const title = root.querySelector?.(".window-title");
        if (title && !title.children.length) {
          const t = translateText(title.textContent);
          if (t !== title.textContent) title.textContent = t;
        }
        return;
      }
      translateNode(root);''',
    '''      // 只处理 Ember 自己的界面，避免把别的模块的英文也一起改了
      if (!/ember/i.test(cls) && !/^Ember/.test(id)) {
        // 例外一：Ember 的确认框走的是**原生 DialogV2**（根元素 class 只有 "dialog"、
        // 类名就是 "DialogV2"），标题、正文、按钮全是硬编码英文，babele 与 i18n 两条通道
        // 都够不着，而上面那道 ember 闸会把它整个挡掉。
        //
        // 先按窗口标题**认框**：认得出是 Ember 弹的，就连正文和按钮一起翻，用作用域表
        // DIALOG_UI（`Ring` / `Close` / `Change` 这类词不能进全局 EXACT）；认不出来就只翻
        // 标题 —— EXACT 里有 Path / Culture / Events 这类通用词，别的模块的窗口恰好同名会被误改。
        // 认框在改标题**之前**做，所以这段是幂等的：重复渲染时标题已是中文，认不出来也不会再动。
        if (id === "DialogV2" || /(^|\\s)dialog(\\s|$)/.test(cls)) {
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
      translateNode(root);''',
    "例外分支只取 .window-title 一个节点就 return，从不 translateNode(root) —— "
    "34 个对话框的正文/按钮结构性不可达，其中 ~15 个标题已被译成中文，同屏「中文标题+英文按钮」。"
    "同一段还顺手补上 Ember 注入到非 Ember 宿主的子树（同调页签）。",
)

# ------------------------------------------------------------------ #
# 12. 日历格式化函数：animate 不走 render 钩子
# ------------------------------------------------------------------ #
add(
    "dead-guard|ember-hardcoded-cn.mjs:472-render-hook-misses-animate-rewrite",
    '''  log(`已改写历法里 ${n} 个月名/星期名。`);
}

/**
 * Ember 各类应用（角色卡、任务面板、日历）渲染出来的分节标题与按钮同样是硬编码。''',
    '''  log(`已改写历法里 ${n} 个月名/星期名。`);
}

/**
 * 把「第 43 天」翻在**源头**，而不是翻在 DOM 上。
 *
 * 世界时钟那行字是 `EmberCalendarUI#animate()` 直接写 innerText 的
 * （ember.mjs:24576-24578 `this.#elements.timeLabel.innerText = \\`${campaignDay} - ${time}\\``），
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
 * Ember 各类应用（角色卡、任务面板、日历）渲染出来的分节标题与按钮同样是硬编码。''',
    "翻译挂在 render 钩子上，而这行字是 animate() 直接写 innerText 的；改包格式化函数，"
    "让英文串不再产生，时钟与法典日志两处一并修好。",
)

add(
    "dead-guard|ember-hardcoded-cn.mjs:472-render-hook-misses-animate-rewrite",
    '''  applyOnce(CONFIG, "__emberCnCalendar", patchCalendarNames, "历法月名");
  applyOnce(CONFIG, "__emberCnRender", patchRenderedApplications, "界面渲染");''',
    '''  applyOnce(CONFIG, "__emberCnCalendar", patchCalendarNames, "历法月名");
  applyOnce(CONFIG, "__emberCnDayFormat", patchCalendarFormatters, "历法日期格式");
  applyOnce(CONFIG, "__emberCnRender", patchRenderedApplications, "界面渲染");''',
    "注册新补丁点。",
)

# ------------------------------------------------------------------ #
src = io.open(SRC, encoding="utf-8").read()
buf = src
bad = []
for i, e in enumerate(E):
    n = buf.count(e["old"])
    if n != 1:
        bad.append((i, e["sig"], n, e["old"][:70]))
    else:
        buf = buf.replace(e["old"], e["new"], 1)

if bad:
    print("!! 命中数不为 1：")
    for b in bad:
        print("  #%d %s -> %d  %r" % b)
    sys.exit(1)

io.open(OUT, "w", encoding="utf-8", newline="\n").write(buf)
print("edits:", len(E), "-> wrote", OUT)

r = subprocess.run(["node", "--check", OUT], capture_output=True, text=True, encoding="utf-8")
print("node --check rc=", r.returncode)
print(r.stdout)
print(r.stderr)

io.open(os.path.join(os.path.dirname(OUT), "s2_edits.json"), "w", encoding="utf-8").write(
    json.dumps(E, ensure_ascii=False, indent=1)
)
