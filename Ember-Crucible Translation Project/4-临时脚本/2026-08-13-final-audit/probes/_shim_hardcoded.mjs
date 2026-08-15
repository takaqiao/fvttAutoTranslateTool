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
 * `{CALM: "calm", TENSION: "tension"}`，enricher 拼的是 `Music Mood: ${mood.titleCase()}`。
 * 原来那五个键（战斗/探索/环境/旅行/休息）在 ember 0.6.x 里一个都不会出现。
 * 译名取 lang/cn.json 的 `EMBER.SoundscapeMoodCalm` / `EMBER.SoundscapeMoodTension`。
 */
const MOODS = {
  "Calm": "平静", "Tension": "紧张"
};

/** 带前缀的标签：`前缀: 名字` → `中文前缀：中文名字` */
const PREFIXED = [
  { en: "Attunement", cn: "同调", table: ATTUNEMENTS },
  { en: "Language", cn: "语言", table: LANGUAGES },
  { en: "Knowledge", cn: "知识", table: KNOWLEDGE },
  { en: "Music Mood", cn: "音乐氛围", table: MOODS }
];

/** 完全匹配即可替换的字符串 */
const EXACT = {
  // 富文本增强器前缀（单独出现时）
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
  "Apply Soulbound Progression": "应用魂缚进程"
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
  { re: /^Day (\d+)\b(.*)$/, cn: (m) => `第 ${m[1]} 天${m[2]}` }
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
  log(`已改写历法里 ${n} 个月名/星期名。`);
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
        // 例外：Ember 的十五个确认框走的是**原生 DialogV2**（根元素 class 只有 "dialog"、
        // 类名就是 "DialogV2"），标题是硬编码英文、babele 与 i18n 两条通道都够不着，
        // 而上面那道 ember 闸会把它整个挡掉。
        // 闸只放行 DialogV2 这一档，不放行全世界的 ApplicationV2 ——
        // EXACT 里有 Path / Culture / Events 这类通用词，别的模块的窗口标题恰好同名就会被误改。
        if (id !== "DialogV2" && !/(^|\s)dialog(\s|$)/.test(cls)) return;
        const title = root.querySelector?.(".window-title");
        if (title && !title.children.length) {
          const t = translateText(title.textContent);
          if (t !== title.textContent) title.textContent = t;
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

