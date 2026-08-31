# -*- coding: utf-8 -*-
"""把 ember 侧 40 条玩家可见硬编码串接进表。"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'

# ── ① 新表：crucible 掷骰卡上的加值/减值来源名（ember 自己塞进去的那几条）──
anchor = 'const CHAT_UI = {'
assert s.count(anchor) == 1
BOON = nl.join([
'/**',
' * **crucible 掷骰卡**上的加值 / 减值**来源名** —— ember 往 `usage.boons` / `usage.banes`',
' * 里塞的那几条（`ember.mjs:140804` 屠龙 / `:141167` 余烬之火花 / `:142148-142149` 混沌折射 …）。',
' *',
' * ⚠ 为什么不能靠下面那道 `flags.ember` 旗标闸：渲染它们的是 **crucible 的**',
' *   `templates/dice/partials/standard-check-details.hbs:6/18`（`{{localize boon.label}}`），',
' *   那是 crucible 自己的掷骰卡，**不带 ember 旗标** —— 旗标闸一条都接不住。',
' *   所以改按**结构选择器**认：`.boon-details .boon > .label` 是 crucible 模板独有的形状，',
' *   命中即归属。',
' *',
' * ⚠ 与 `crucible-cn 0.9.18` 的 `BOON_BANE_LABELS` **不重叠**：那边收的是 crucible 自己的',
' *   11 条（Special / Elite / Boss / Bulky Armor …），这边只收 ember 加的 7 条。',
' *   两个模块各翻各的串，同一棵 DOM 上跑两遍互不影响（查不到的原样返回）。',
' *',
' * 译名一律取 glossary_ec 已定的中文段（合集里角色卡上就是这些字），不另造：',
' *   混沌折射 / 混沌继承者 / 湮解印记 / 钻研反制 / 余烬之火花 已在词表；',
' *   `Nir\'ae`＝尼尔艾、`Drakonbane Poison`＝屠龙毒药 是词表里的既定词根。',
' */',
'const BOON_BANE_LABELS = {',
'  "Chaotic Refraction": "混沌折射",     // ember.mjs:142148 / :142149（加值与减值同名）',
'  "Inheritors of Chaos": "混沌继承者",   // 词表 Inheritors of Chaos',
'  "Mark of Unmaking": "湮解印记",       // 词表 Mark of Unmaking',
"  \"Nir'ae Resistance\": \"尼尔艾抗性\",    // 词表 Nir'ae＝尼尔艾",
'  "Studious Counter": "钻研反制",       // 词表 Studious Counter',
'  "Drakonbane": "屠龙",                // ember.mjs:140804；词表 Drakonbane Poison＝屠龙毒药',
'  "Spark of Ember": "余烬之火花",        // ember.mjs:141167；词表 Spark of Ember',
'};',
'',
])
s = s.replace(anchor, BOON + anchor)

# ── ② 聊天钩子：在旗标闸**之前**加一条结构规则 ──
OLD = nl.join([
'  Hooks.on("renderChatMessageHTML", (msg, html) => {',
'    try {',
'      if (!msg?.flags?.ember) return;',
'      translateNode(html instanceof HTMLElement ? html : html?.[0], CHAT_UI);',
])
NEW = nl.join([
'  Hooks.on("renderChatMessageHTML", (msg, html) => {',
'    try {',
'      const chatRoot = html instanceof HTMLElement ? html : html?.[0];',
'      if (!chatRoot) return;',
'      // ① 结构规则，跑在旗标闸**之前**：crucible 掷骰卡上的加值/减值来源名。',
'      //    它们不带 ember 旗标（卡是 crucible 弹的），只能按结构认。见 BOON_BANE_LABELS。',
'      for (const el of chatRoot.querySelectorAll?.(".boon-details .boon > .label, .bane-details .bane > .label") ?? []) {',
'        if (el.children.length) continue;',
'        const raw = el.textContent.trim();',
'        const cn = BOON_BANE_LABELS[raw];',
'        if (cn && cn !== raw) el.textContent = cn;',
'      }',
'      // ② 旗标闸：其余只翻**打了 ember 旗标**的卡（那张卡是 :2982 造的）。',
'      if (!msg?.flags?.ember) return;',
'      translateNode(chatRoot, CHAT_UI);',
])
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)

# ── ③ EMBER_WINDOW_UI 补 30 条 ──
tail_anchor = nl + '};' + nl + nl + '/**' + nl + ' * 指示物制作器'
idx = s.index('const EMBER_WINDOW_UI = {')
end = s.index(nl + '};', idx)
ADD = nl + nl.join([
'',
'  // ══ 2026-08-22 第七轮｜硬编码全量审计挑出来的**玩家可见**缺口 ══════════════',
'  // 来源：`4-临时脚本/2026-08-22-hardcoded-audit/`。审计扫五个通道，ember 侧剩 3401 条未覆盖，',
'  // 其中约 3080 条是 Vista 建图器的资产/图层名（只在 vista-config-assets.hbs 渲染、',
'  // 由场景控件打开＝GM 建图工具，项目所有者已裁**不做**），199 条是区域地图场景定义、',
'  // 40 条是 GM 配置/编辑器 ⇒ 真正玩家可见的就下面这些。',
'  // ⚠ 这些窗口的类名全部以 `Ember` 开头，过得了本文件那道 `/^Ember/.test(id)` 闸，',
'  //   所以放在本表（Ember 窗口作用域）就够得着；不需要另开注入点。',
'',
'  // ── 独立事件页（EmberStandaloneEventPage，ember.mjs 内）',
'  "Not Completed": "未完成", "Repeating": "可重复", "Scene and Level": "场景与层",',
'  "Specific Hexes": "指定六边格", "Unique": "唯一",',
'',
'  // ── 生物群系页（EmberBiomePageSheet）；`Biome`＝生物群系 / `Vista`＝远景 取自 glossary_ec',
'  "Area Map": "区域地图", "Misconfigured Area": "区域配置有误",',
'  "Misconfigured Vista": "远景配置有误", "Not Discovered": "未发现", "Vista": "远景",',
'',
'  // ── 血统页（EmberAncestryPageSheet）',
'  "Not Playable": "不可选用", "Playable": "可选用",',
'',
'  // ── 创角向导（EmberCharacterCreationSheet）；`Aster`＝阿斯特 / `Soulbound`＝魂缚 取自 glossary_ec',
'  "Aster Features": "阿斯特特性", "Soulbound Features": "魂缚特性",',
'',
'  // ── 日历条（EmberCalendarNavigation）的两个分类',
'  "Discovery": "发现", "Event": "事件",',
'',
'  // ── 派系名（ember.mjs:142616 起）。`Anachraenum`＝阿纳克瑞纽姆 与 `Trading House`＝商会',
'  //    取自 glossary_ec；`Graven\'s Rest` 取词表 `Graven\'s Rest Refresher`＝格雷文之憩提神饮 的词根；',
'  //    `Cerulean` 取 `The Cerulean Bloom`＝蔚蓝绽放 的词根；`Ashka`＝阿什卡。',
'  "Cerulean Sails Company": "蔚蓝之帆商会", "Graven\'s Rest": "格雷文之憩",',
'  "Rejarh Ashka": "雷雅尔·阿什卡", "The Anachraenum": "阿纳克瑞纽姆", "Trading Houses": "商会",',
'',
'  // ── 旅行事件（ember.mjs 的 events 表）',
'  "Combat Encounter": "战斗遭遇", "Harvesting Opportunity": "采集机会",',
'  "No Event": "无事件", "Wandering Merchant": "游商",',
'',
'  // ── 区域地图：两个切片名与地形图例。切片名是玩家在地区地图上直接看到的分区。',
'  "Surface of Ember": "余烬地表", "The Pathways": "通路", "Barrier": "屏障",',
'',
'  // ── 区域地图**层名的兜底**（`s.levels.get(id)?.name ?? cfg.compositions[id]?.label`，:2376）。',
'  //    正常情况下玩家看到的是 Scene 文档那一份（已由 Babele 的 SCENE_LEVELS 翻成双语），',
'  //    这里只在那一份取不到时才上屏 —— 所以**照抄双语形式**，免得两条路显示不一致。',
'  "Repurposed Quarry - Upper Level": "改造采石场 - 上层 Repurposed Quarry - Upper",',
'  "Repurposed Quarry - Middle Level": "改造采石场 - 中层 Repurposed Quarry - Middle",',
'  "Repurposed Quarry - Lower Level": "改造采石场 - 下层 Repurposed Quarry - Lower",',
'  "Chamber of Agaseros - Reservoir": "阿加瑟罗斯之室 - 蓄水池 Chamber of Agaseros - Reservoir",',
])
s = s[:end] + ADD + s[end:]

# ── ④ DIALOG_TITLES 补一条（认框 + 标题翻译）──
DT = '  "Ember: Teleport Destination": "余烬：传送目的地",'
assert s.count(DT) == 1
s = s.replace(DT, DT + nl
  + '  // ember.mjs:108743 的 DialogV2 标题；`Aedir Signalpost`＝艾迪尔信号哨站 取自 glossary_ec' + nl
  + '  "Aedir Signalpost Stealth Field Generator": "艾迪尔信号哨站隐形立场发生器",')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('已接入')
