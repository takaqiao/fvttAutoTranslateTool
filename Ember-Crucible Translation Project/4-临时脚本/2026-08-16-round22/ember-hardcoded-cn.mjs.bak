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
 *
 * ------------------------------------------------------------------
 * `scan_cross_channel` B 段 `MJS_ORPHAN_CN` 的裁决索引（2026-08-15 第十七轮逐条判完）
 * ------------------------------------------------------------------
 * 这一档从没被登记过（§1 那张收口表只列了 `MJS_LANG_DRIFT`），第十七轮实跑
 * ember 10 条 / crucible 4 条、去重 14 条，**全部判完**，理由都写在各自的表旁边：
 *
 *   改 .mjs（合集有 GM 指南级的同域证据，判据这次是对的）—— 3 条，改完该档 14 → 11：
 *     `DIALOG_UI.Ascend`  上行 → 上升    ┐ Aedir Signalpost「Rotating Elevator」页逐字写过
 *     `DIALOG_UI.Descend` 下行 → 下降    ┘（与 Forwards/Backwards/Unreachable 同一先例）
 *     `DIALOG_UI.Seal`    封闭 → 封堵      Disturbed Earth「操作焦油坑」一节逐条列过按钮
 *       ↳ 顺带补 `Open`/`Spawn` 两个按钮，与 DIALOG_TITLES 的 `Tar Pit`（先前是**死键**，
 *         该框认不出来、标题与三个按钮全英）
 *
 *   保持 .mjs、compendium 也不动（判据的多数写法是共现噪声或另一个义项）—— 8 条：
 *     `DIALOG_UI.Close` / `DIALOG_UI.Usage` / `DIALOG_UI.Make` / `DIALOG_UI.Ring` /
 *     `EXACT.Exit` / `LANGUAGES.Sign` / `EMBER_WINDOW_UI.Sprites` / `ATTUNEMENT_TAB.Active`
 *
 *   保持 .mjs、**合集那边错**，只有 **1 条**：`WEATHER.Tempest`
 *     —— 已出批次 `4-临时脚本/2026-08-15-round17/batches/r17-weather-tiers-*.json` 并已落盘。
 *     ⚠ 上一版这里写「3 条：Dust / Kindling / Tempest」是**说大了**：WEATHER 表上方的注释
 *     自己写明 `Dust` 与 `Kindling` 的天气义在合集里 **0 叶**、两边都不该动，本不需要批次。
 *
 * ⚠ 剩下的 11 条**已全部裁为「保持」，下一轮不要再当新缺陷重查**，也不要为了把计数清零
 *   去改任何一边 —— 它们是判据的定义域问题（枚举名 / UI 按钮 / 句首碎片 vs 散文里的同形词）。
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
 *
 * ⚠ 长名那两条（`The Abyss` / `Heart of Ember`）**只在「英文存量世界」里可命中**，别当活键去调译文
 *   （2026-08-15 第十六轮查实，原注释说的「合集页名走这一套」会误导）：
 *   `Attunement: ${attunement.name}` 里的 name 由 initializeAttunementOptions（ember.mjs:129701-129718）
 *   从**世界里的** JournalEntry `emberCosmos00000` 取 page.name。装了本汉化之后那个页名已经是
 *   babele 译好的「深渊 The Abyss」/「余烬之心 Heart of Ember」（双语），查表必然落空、叶子原样带回；
 *   只有「先用英文导入过、之后才装汉化」的世界才会命中。保留是因为成本为零、且正是那种世界的兜底。
 * 2026-08-13 第三轮：`Aura` 原译「灵气」是错的 —— 它是月亮专名，Cosmos 页 name 字段即「奥拉 Aura」，
 * 同一份月亮清单里 Mayis/Cora/Ragen/Orbis/Akon 全是音译；「灵气」是 `Aura Spellcraft` 手势的 adjective，不是月名。
 * ⚠ 由此，B 段拿 crucible lang 的 `SPELL.GESTURES.Aura` / `ACTION.TARGET_TYPES.Aura`＝灵气 来比本表的
 *   「奥拉」必报 `MJS_LANG_DRIFT`，**是既定的 `Aura` 三分，不是缺陷**（PROJECT.md：Ember 月亮／同调／屈折
 *   ＝奥拉、`Auric`＝奥拉的；Crucible 的手势与目标类型＝灵气；泛用 `Fear/Command Aura`＝灵气）。别统一。
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

/**
 * 11 个「同调天赋」物品的合集译名（ember.crucible-character 的 `Abyss Attunement` 等 11 条 entries）。
 *
 * 为什么要单列一张而不是拿 ATTUNEMENTS 拼：合集里 `Heart Attunement` 定的是「心之同调」，
 * 不是「余烬之心 + 同调」，靠短名表拼会跟合集打架。
 *
 * 用途有二（都是 ATTUNEMENTS 那张短名表接不住的形状）：
 *   ① HeroSheet 天赋列表里的物品名 —— 上游 prepareAbilities 每轮 prepareData 都用
 *      `item.name = \`Abyss Attunement (Rank ${ATTUNEMENT_RANK_NUMERALS[rank]})\``（ember.mjs:126104 等 11 处）
 *      把 babele 译名覆盖掉，合集里的译文根本留不住，只能在 DOM 上兜（见 HERO_ITEM_NAMES）；
 *   ② 同调奖励聊天卡的表头 `${config.label} 同调`（ember.mjs:2971，见 CHAT_UI）。
 *
 * ⚠ **`scan_cross_channel` B 段会为这 11 条各报两次，两次都是判据的定义域问题，不是缺陷**
 *   （2026-08-15 第十六轮逐条判完，ember 侧 22 条 / crucible 侧 22 条，占两边 B 段报数的大半）：
 *     · `MJS_LANG_DRIFT [ATTUNEMENTS]`：判据按**英文串**在整个 .mjs 里找同名键，于是
 *       ATTUNEMENTS["Abyss"]＝深渊 与本表 ["Abyss"]＝「深渊同调 Abyss Attunement」被算成
 *       「.mjs 内部同英文异译」。两张表喂的是**两种形状**：前者是 `ATTUNEMENTS[x].label` 那个短名，
 *       后者是天赋物品的整名（带双语尾），键名撞在一起只是因为上游拿同一个词当 id。
 *     · `MJS_LANG_DRIFT [ATTUNEMENT_ITEM_NAMES]`：拿本表的值去比 lang `SPELL.INFLECTIONS.<x>`
 *       （en 就是 `Abyss` 这种裸词），比的同样是「整名 vs 短名」，不同域。
 *   两条都别再统一 —— 统一任何一边都会让上面 ① ② 那两处上屏文本错位。
 *   `Heart` 那条差得最远（短名「余烬之心」／整名「心之同调」），成因见上一段，合集英文闸下
 *   「心之同调」57 叶有据，也不是笔误。
 */
const ATTUNEMENT_ITEM_NAMES = {
  "Abyss": "深渊同调 Abyss Attunement",
  "Akon": "阿肯同调 Akon Attunement",
  "Aura": "奥拉同调 Aura Attunement",
  "Cora": "科拉同调 Cora Attunement",
  "Heart": "心之同调 Heart Attunement",
  "Luxarum": "卢克萨鲁姆同调 Luxarum Attunement",
  "Mayis": "玛伊斯同调 Mayis Attunement",
  "Orbis": "奥比斯同调 Orbis Attunement",
  "Primordis": "普里莫迪斯同调 Primordis Attunement",
  "Ragen": "拉根同调 Ragen Attunement",
  "Signara": "西格纳拉同调 Signara Attunement"
};

/**
 * Ember 的十个月亮名（ember.mjs:52821 起 `cosmos.moons[]` 的 `name` 字段，裸英文）。
 *
 * 日历面板的月亮 tooltip 由 `#refreshMoons()` 每帧写成 `${moon.name} ${moon.phaseLabel}`
 * （ember.mjs:24628），相位那半截走 i18n 已是中文、月名那半截是数据 —— 三张查表都是整串匹配，
 * 接不住这种「英文名 + 空格 + 中文相位」的复合串，而 #refreshMoons 由 animate() 每帧调用、
 * 不发渲染钩子，DOM 层就算翻了下一帧也会被写回。所以改数据（patchMoonNames）。
 * `moon.name` 全上游只有 24628 这一处消费点（查找一律用 `moon.id`，如 :125773 / :125958），改名安全。
 * 第十个 `Ember` 不是同调月，按 PROJECT.md 既定 Ember(世界)=余烬。
 */
const MOON_NAMES = {
  ...ATTUNEMENTS,
  "Ember": "余烬"
};

/**
 * 语言。Common / Sign 来自 crucible 本体，其余是 Ember 新增。
 *
 * ⚠ 头两条（Common / Sign）在 **Crucible 世界里是死键**：crucible 本体登记的是 i18n 键
 * （crucible-compiled.mjs:1007-1012 `label: "LANGUAGES.Common"`），i18nInit 的 :47782
 * `localizeConfigObject(crucible.CONFIG.languages)` 早于 ready 就把它们换成了 crucible-cn 的译文，
 * patchCrucibleConfig 再查 `table[entry.label]` 必然落空。它们只在 **dnd5e** 世界里有用
 * （那边 Ember 自己拼英文 label）。其余 23 条是 ember.mjs:126693 起用裸英文 label 注册的，两边都活。
 */
const LANGUAGES = {
  // ⚠ `Common` 在本文件里**有意两译**：这里是语言名＝通用语，RARITIES 那张表里是稀有度＝常见。
  //   同理 `Draconic`（语言＝龙语 / 术士起源＝龙脉）。B 段会报成「.mjs 内部同英文异译」，不是缺陷。
  "Common": "通用语",
  // ⚠ `Sign` 是 crucible 本体那门**手语**（crucible-compiled.mjs:1007-1012 的 languages 表）。
  //   B 段 `MJS_ORPHAN_CN` 报的多数写法「荆棘」（6/10）纯属共现（同叶有 brambles/thorny）；
  //   英文闸 `\bSign\b` 的 10 叶是 `The Clever Fox (Engraved Sign)` 那件雕刻招牌、
  //   「Sign on the wall? 墙上的招牌」、「a single, obscure Sign 一个晦涩的符号」、
  //   「Sign your party's name 签上队伍的名字」—— 招牌 / 符号 / 签名三个义项，无一是语言名。
  //   2026-08-15 第十七轮已裁：保持「手语」，compendium 一叶不动。
  //   （另注：本条与上面的 `Common` 在 **Crucible 世界里是死键**，见上一段。）
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
  // ⚠ `Abyssal` 与 ember lang 的 `SPELL.INFLECTIONS.AbyssAdj`（en 也是 `Abyssal`）＝深渊的 **不同域**：
  //   那条是同调**屈折词**（法术名 `Abyssal Ray` 那一支，同族还有 Akonic/Coran/Emberite/Luxaran…），
  //   这里是 crucible.CONFIG.languages 里的**语言名**。B 段按英文串比，会报成 MJS_LANG_DRIFT，不是缺陷。
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
 * 语言分组名。Ember 往 `crucible.CONFIG.languageCategories` 里塞了两个**裸英文** label
 * （ember.mjs:126691-126692 `.ancient = {label: "Ancient Languages"}` / `.obscure = {label: "Obscure Languages"}`），
 * 不是 i18n 键，crucible 的 `localizeConfigObject`（crucible-compiled.mjs:47781）对它们原样返回。
 * 这两个分组名出现在语言下拉的 optgroup 上（crucible-compiled.mjs:15608 与 :23770），
 * 而 crucible 自带的 spoken / nonSpoken 走 i18n 键、已由 crucible 汉化插件译成
 * 「口语语言 / 非口语语言」—— 不补这张表就会四个分组两中两英。
 * 由 patchCrucibleConfig 在 ready 时改写；core 自带那两条此时 label 已是中文，查表落空、不会被动。
 */
const LANGUAGE_CATEGORIES = {
  "Ancient Languages": "远古语言",
  "Obscure Languages": "冷僻语言"
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
  "kost": "科斯特语",
  // `moiré` 与上面两条成因不同，值得单记一笔：
  // 上游增强器的 pattern 是 `/\[\[\/language (\w+)]]/g`（ember.mjs:129447）—— **没有 `u` 标志**，
  // 而 `é` 不在 `\w` 里，所以这条标记**根本不会被增强器接管**，正文直接吐出字面量
  // `[[/language moiré]]`。也就是说它不是「查不到 id」，是「压根没进增强器」。
  // 曾经考虑过改 compendium 把标记删成纯文本，但那会让中英标记多重集不等 ——
  // `apply_translations` 的标记闸必拒、`scan_markup_drift` 的 INLINE 从 0 变 4，
  // 还要为此加一条永久豁免。改成在**我们自己这条兜底 PATTERN 上放宽字符类**，
  // 一行解决，compendium 一个字都不用动，也不需要绕任何闸。
  "moiré": "莫伊雷语"
};

/**
 * `[[/knowledge …]]` 引用了、但**两个系统都没有**的知识领域 id —— 同 MISSING_LANGUAGES 的成因。
 *
 * 逐条核过上游：crucible 的 DEFAULT_KNOWLEDGE（crucible-compiled.mjs:586 起 31 条）与 Ember 追加的
 * 4 条（ember.mjs:126683 abyssals/aedir/leviathans/shent）里都没有 stars / soul / beats，
 * `Shent` 则是大小写写错（真实 id 全小写）。enrichKnowledge 查不到就 `return new Text(match)`
 * （crucible-compiled.mjs:46815），正文原样吐出 `[[/knowledge stars]]` 这种裸标记。
 * 合集实测（两个孪生包合计）：stars×4、Shent×2、soul×2、beats×2，共 10 处。
 *
 * 三条是上游拼错（soul→souls、beats→beasts、Shent→shent），译名照抄 KNOWLEDGE 里的正主，
 * 不另起炉灶；`stars` 是 dnd5e 侧 KNOWLEDGE_TYPES 有、crucible 侧没有的领域（dnd5e-async.mjs:549）。
 * 兜底文案与 crucible 汉化插件的 `ACTOR.KnowledgeSpecific = 知识：{knowledge}` 同形，
 * 这样它跟同一段正文里正常渲染的 `[[/knowledge undeath]]` 看上去是一回事。
 */
const MISSING_KNOWLEDGE = {
  "stars": "星辰",
  "Shent": "申特",
  "soul": "灵魂",
  "beats": "野兽"
};

/**
 * 知识领域。
 *
 * ⚠ **作用域**（2026-08-14 第十四轮订正，原注释把因果写反了）：
 *   - 在 **Crucible 世界里，本表只有末尾那 4 条是活的**（Abyssals / Aedir / Leviathans / Shent）——
 *     它们是 ember.mjs:126683 用裸英文 label 注册进 crucible.CONFIG.knowledge 的，由
 *     patchCrucibleConfig 在 ready 时改写。前 31 条与 PREFIXED 里的 `Knowledge` 前缀在这边**一律不生效**：
 *     ① Ember 的 knowledge 增强器只在 **dnd5e** 专用的 registerEnrichers$1（ember.mjs:123656，
 *        `Knowledge: ${k.label}` 在 :123707）里注册；crucible 侧 enrichers 模块只导出
 *        `{enrichLanguage, onRenderPassiveCheck}`（ember.mjs:126618），:129483 的
 *        `if (ember.system.enrichers.registerEnrichers)` 条件为假，压根不挂。
 *     ② Crucible 下真正上屏的是 crucible 自己的 enrichKnowledge（crucible-compiled.mjs:46809-46820），
 *        它走 `_loc("ACTOR.KnowledgeSpecific", {knowledge: knowledge.label})`，而 label 是 i18n 键
 *        （:587 起 `alchemy: {label: "KNOWLEDGE.Alchemy"}`），i18nInit 的 :47780 就已换成 crucible-cn 的译文，
 *        patchCrucibleConfig 再查 `table[entry.label]` 恒为 undefined。
 *        ⇒ **Crucible 世界要改这 31 条，只能改 crucible 汉化插件的 lang/cn.json，改本表 0 效果。**
 *   - 前 31 条保留不删：它们在 **dnd5e** 世界里是活的（那边 Ember 自己拼英文 label）。
 *   - 2026-08-14 第十四轮删掉了 `Outsiders`：它在**两个系统里都够不到** —— crucible 侧被
 *     ember.mjs:126682 `delete crucible.CONFIG.knowledge.outsiders` 删掉，dnd5e 侧的
 *     KNOWLEDGE_TYPES（dnd5e-async.mjs:528-560，共 35 条）里压根没有 outsiders 这一项
 *     （只有 abyssals 的 aliases 里留着这个旧名）。原注释「仅 dnd5e 侧可能残留」是错的。
 *   - `Stars` 是反过来的情况：dnd5e-async.mjs:549 有 `stars: {label: "Stars"}`，crucible 侧没有，
 *     所以它在 dnd5e 世界里是活键；crucible 世界那 4 处裸标记走 MISSING_KNOWLEDGE。
 *
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
  "Legends": "传说", "Machines": "机械", "Monsters": "怪物",
  "Plants": "植物", "Politics": "政治", "Rituals": "仪式", "Seafaring": "航海",
  "Souls": "灵魂", "Stars": "星辰", "Subterranea": "地下世界", "Tracking": "追踪", "Trade": "贸易",
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
 * 音景「组名」＝播放列表侧栏音景面板两个 `<select>` 里的 `<optgroup label>`。
 * `#updateSoundscapeForm()` 把 `{value, label: 编排名, group: 音景 label}` 交给
 * createSelectInput（ember.mjs:15916-15922），`group` 取的就是音景对象自己的 `label`。
 *
 * 全集是**实测**的，不是估计：`var soundscapes = Object.freeze({…})`（ember.mjs:15260）
 * 注册 44 个音景，其中 `type` 为 music(41) / environment(1) 的 **42 个**才进这两个下拉
 * （落选的两个是 `Ember Events` / `Ember Weather`），42 个 label 互不重样。
 * 探针 `4-临时脚本/2026-08-15-round21/probe_groups.mjs` 用**两套互不相干的方法**各抠一遍
 * —— (A) 手写大括号配对 + 正则取 label/type；(B) 按顶层 `var <名> = {…}` 切片交给 **node:vm**，
 * 让真正的 JS 解析器把对象建出来、并要求每个对象自己的 `id` 等于注册表里的键 ——
 * 自报 `resolved: A=44 B=44 BAD=0` 且两套结果逐条相同；任一条对不上就退出而不是给半份结果。
 * **本表 42 条＝全部组名**（第二十一轮补齐：组名恒英文 37 → 0）。
 *
 * ⚠ 与 ARRANGEMENTS **各自查重**：实测有 **7 条**字符串既是组名又是编排名 ——
 * `Ancient Ruins` / `Ankarist Theme` / `Lyla Theme` / `Marlstone Gala` / `Ordain` /
 * `Sin Theme` / `The Pit Trap`（`Shent Ruins` 则只是编排名，不在本表）。
 * 一张扁平表分不出 optgroup 与 option，同一个英文串只可能有一个译文，所以这 7 条
 * **只在本表里定义一次**，ARRANGEMENTS 不再重复写；编排名那一路要用时由
 * ARRANGEMENT_LEAVES 按键取回（见其表内注）。
 *   ↳ 副作用记一笔：组名补齐之后，`Marlstone Gala` / `Ordain` / `The Pit Trap` 这 3 条
 *     **编排名**也跟着被翻了，编排名恒英文 **204 → 201**。这是同串同译的必然结果，
 *     不是对「204 条编排名暂不补」那条裁决的翻案，下一轮别把它当成偷补。
 *
 * 译名依据（专名一律先过 compendium 英文闸，大小写逐条单独判；括号内为命中的英文串）：
 *   · 生物类取 crucible lang `TAXONOMY.CATEGORIES.*`（最权威的一档）：`Beast`野兽 /
 *     `Celestial`天界生物 / `Construct`构装体 / `Monstrosity`畸怪 / `Ooze`软泥怪 /
 *     `Undead`不死生物 / `ElementalEarth`土元素 · `ElementalFire`火元素 · `ElementalFrost`冰霜元素。
 *     `Abyssal` 不在该表，取合集英文闸的压倒性写法「深渊」（`Abyssal Monster`深渊怪物 26 叶级）；
 *     注意不是 KNOWLEDGE 里那条 `Abyssals`＝深渊裔（那是族名，这里是形容词）。
 *   · 地名 / 专名走合集英文闸：`Arctus Plateau`阿克图斯高原、`House Bastilla`巴斯蒂拉家族、
 *     `Marlstone Gala`马尔石晚会、`Ordain`奥尔丹、`Ordani`奥尔达尼、`Seawall`海堤、
 *     `Forest of Stone`石之森林、`Kadisos`卡迪索斯、`Inner Realms`内界（Cosmos 页分类名）、
 *     `Cosmos`寰宇、`The Pit Trap`陷坑、`Arcturian`阿克图里安、`Aedir`艾迪尔、
 *     `Graven's Rest`格雷文之憩（`Graven's Rest Refresher`＝格雷文之憩提神饮）。
 *   · `Cindaric Temple`＝辛达里克神殿：合集正文 3 处逐字如此。**别写成「辛达林神殿」**——
 *     那是另一个英文串 `Cindarin Temple` 的既定译名（上游对同一栋楼有两种写法），
 *     英文闸按键判，本表的键是 `Cindaric Temple`。
 *   · `Water Temple`＝水之神殿：石之森林地名志正文「然而，水之神殿要古老得多」。
 *   · `Solemn Folk`＝肃穆民谣：GM 指南补丁 0.4.5 **逐字点名过这个音景**
 *     （「加入了名为“肃穆民谣”的新乐曲音景」）；同一段还给出 `Dungeon`＝地下城、
 *     `The Pit Trap`＝“陷坑”。`Seydiri Theme`＝塞迪里主题：另一版补丁说明作「塞迪里主题乐曲」。
 *   · `Music`＝乐曲：glossary_ec 定稿 `Ember Music`＝余烬乐曲，GM 指南同作「…乐曲」。
 *   · `X Theme` 的「的」字看是不是人名：合集正文写人物主题用的是所有格
 *     （`Music: Ankarist's Theme` / `Lyla's Theme` / `Sin's Theme`），故人名三条作「X的主题」；
 *     族名 / 概念（Aedir / Seydiri / Arcane）上游不带所有格，作「X主题」。
 *   · 合集查不到的只有 3 条，依据只到「同族词 + 构词」这一档，写明以免下轮误当定稿：
 *     `Bard Troupe`吟游诗人剧团（`Bard`＝吟游诗人 7 叶）、`Mystical Dungeon`神秘地下城、
 *     `Arcturian Town`阿克图里安城镇。
 */
const SOUNDSCAPE_GROUPS = {
  "Abyssal Combat": "深渊战斗",
  "Aedir Theme": "艾迪尔主题",
  "Ancient Ruins": "远古遗迹",                    // 也是编排名（7 条同串之一）
  "Ankarist Theme": "安卡里斯特的主题",             // 也是编排名
  "Arcane Theme": "奥术主题",
  "Arcturian Folk": "阿克图里安民谣",
  "Arcturian Town": "阿克图里安城镇",
  "Arctus Plateau Music": "阿克图斯高原乐曲",
  "Bard Troupe": "吟游诗人剧团",
  "Beast Combat": "野兽战斗",
  "Celestial Combat": "天界生物战斗",
  "Cindaric Temple": "辛达里克神殿",
  "Construct Combat": "构装体战斗",
  "Cosmos Music": "寰宇乐曲",
  "Elemental Combat - Combined": "元素战斗 · 综合",
  "Elemental Combat - Earth": "元素战斗 · 土",
  "Elemental Combat - Fire": "元素战斗 · 火",
  "Elemental Combat - Frost": "元素战斗 · 冰霜",
  // `Ember Environment` 同时是 :15892 那条 `<header>` 的文案，一个键盖两处，
  // 所以 MOOD_PANEL 里不再单独写它。译名取 glossary_ec 定稿。
  "Ember Environment": "余烬环境",
  "Forest of Stone Exploration": "石之森林探索",
  "Graven's Rest Music": "格雷文之憩乐曲",
  "House Bastilla": "巴斯蒂拉家族",
  "Illusory Combat": "幻象战斗",                  // `IllusionAdj: Illusory`＝幻象的（crucible lang）
  "Inner Realms Music": "内界乐曲",
  "Kadisos Exploration": "卡迪索斯探索",
  "Lyla Theme": "莱拉的主题",                      // 也是编排名
  "Marlstone Gala": "马尔石晚会",                  // 也是编排名
  "Monstrosity Combat": "畸怪战斗",
  "Mutagenic Combat": "诱变战斗",                  // `Mutagenic Affliction`＝诱变病症
  "Mystical Dungeon": "神秘地下城",
  "Ooze Combat": "软泥怪战斗",
  "Ordain": "奥尔丹",                             // 也是编排名
  "Ordani Folk": "奥尔达尼民谣",
  "Pirate Combat": "海盗战斗",
  "Raider Combat": "劫掠者战斗",                   // `Raider`＝劫掠者、`Otherhood Raider`＝异姊会劫掠者
  "Seawall": "海堤",
  "Seydiri Theme": "塞迪里主题",
  "Sin Theme": "辛的主题",                         // 也是编排名
  "Solemn Folk": "肃穆民谣",
  "The Pit Trap": "陷坑",                          // 也是编排名
  "Undead Combat": "不死生物战斗",
  "Water Temple": "水之神殿"
};

/**
 * 音景「编排」名。arrangement.label 是 ember.mjs 里的硬编码常量
 * （5694 / 5787 / 12393 / 14064 / 14748 各 var 块），babele 与 i18n 两条通道都够不到。
 *
 * ⚠ 这张表**只装编排名**（外加 `Reset`，理由见下）。第二十一轮以前它一张表同时兜两档
 * （9 键里 4 个其实是 optgroup 组名），混着长下去迟早出误命中，已拆开：组名一律进
 * SOUNDSCAPE_GROUPS。实测 42 个组名 / 212 个去重编排名里有 **7 条同串**，同串只写一次、
 * 写在组名表；本表现在的 4 条编排名（`Shent Ruins` / `Shent Ruins Tension` /
 * `The Pit Trap - Intense` / `The Pit Trap - Relaxed`）都**只是编排名、不是组名**，与组名表零重叠。
 * 覆盖面没变：编排名 212 条里翻了 11 条（本表 4 ＋ 组名表里同串的 7），恒英文 201 条。
 */
const ARRANGEMENTS = {
  // `Reset` 既不是组名也不是编排名：它是 ember.mjs:16255 `${channel.capitalize()}: Reset`
  // 那一支增强器的叶子（通道重置），因为与编排名走同一个叶子位置，故放在本表。
  // 2026-08-15：原译「默认」与 lang 的 `EMBER.CONTENT_CONFIG.RESET`＝重置 漂移（scan_cross_channel B 段第 1 条），
  // 且英文闸下 compendium 的 `Reset` 8 叶全作「重置」、0 叶作「默认」。上游这一支的注释也是
  // 「// Reset channel to default」（ember.mjs:16253），按动作译成「重置」两条通道即一致。
  // ⚠ 记一笔方向：这一条把 B 段的 `Reset` 由 DRIFT 变成 AGREE（净收益 −1 条漂移），
  //   与同一轮 `Usage` 那条 MJS_ORPHAN_CN +1 是两笔相反的变化，B 段总数 33 → 32 是两者相抵后的结果。
  //   下一轮对 B 段计数时别把这两笔当成没发生。
  "Reset": "重置",
  "Shent Ruins": "申特遗迹",
  "Shent Ruins Tension": "申特遗迹 · 紧张",
  "The Pit Trap - Intense": "陷坑 · 激烈",
  "The Pit Trap - Relaxed": "陷坑 · 舒缓"
};

/**
 * 增强器那一路要查的「编排名叶子」表：本表 = ARRANGEMENTS ＋ 组名表里那 7 条同串。
 *
 * `EmberSoundscape.enricherHTML` 拼出来的叶子只可能是 `arrangement.label` 或 `Reset`
 * （ember.mjs:16255 / 16266），其中 7 个 label 与组名同串、译文写在 SOUNDSCAPE_GROUPS 里，
 * 所以这里**按键取回**而不是重抄一遍 —— 一个英文串在本文件里只有一个定义处。
 * ⚠ 别图省事写成 `{...SOUNDSCAPE_GROUPS, ...ARRANGEMENTS}`：那会把 35 条**不是编排名**的
 * 组名也塞进增强器的查表面，属于无谓放宽。
 */
const ARRANGEMENT_LEAVES = {
  ...ARRANGEMENTS,
  ...Object.fromEntries(
    ["Ancient Ruins", "Ankarist Theme", "Lyla Theme", "Marlstone Gala",
     "Ordain", "Sin Theme", "The Pit Trap"]
      .filter((k) => k in SOUNDSCAPE_GROUPS)
      .map((k) => [k, SOUNDSCAPE_GROUPS[k]])
  )
};

/**
 * 神祇页上的三类子职标签（`EmberDeityPageSheet._getTags()`，ember.mjs:36354/36359/36364）。
 *
 * 上游拼的是 `Divine Domain: ${domain}` / `Warlock Patron: ${patron}` / `Sorcerous Origin: ${origin}`，
 * 值取自 journal page 的 `system.{clerics,warlocks,sorcerers}`。这三个字段既没进 babele mapping
 * （3-常用脚本/extract/mappings.mjs 的 ember.deity 页型里没有它们），也不是 i18n 键，只能在这里兜。
 * 宿主类名以 Ember 开头，patchRenderedApplications 的闸放行，补进 PREFIXED 即生效。
 *
 * 值集是从 modules/ember 的 LevelDB 里实测导出的（148 个页面、130 叶、26 个唯一值）：
 *   clerics   Knowledge6 Life10 Light4 Order6 Other24 Peace6 Time2 Trickery4 Twilight10 War12
 *   warlocks  Archfae2 Archfey2 "The Archfey"2 Celestial4 Fathomless2 Fiend4 "The Fiend"2 Genie2
 *             "Great Old One"2 "The Great Old One"6 Hexblade2 "The Undead"6 Other2
 *   sorcerers Draconic2 Spellfire2 "Wild Magic"4
 * 译名取合集里《Character Classes》页已有的小标题译法（en/cn 逐条对齐）：
 *   Cleric 页 X Domain→X 领域、Warlock 页 X Patron→X 宗主、Sorcerer 页 Draconic→龙脉/Spellfire→法术火焰。
 * `Time` / `Other` 该页没有小节，取 glossary_ec 的「时间 / 其他」。
 *
 * ⚠ `Hexblade` **故意不在下面的 WARLOCK_PATRONS 表里** —— 已裁，不是漏了，别再补回来。
 *   裁决：PROJECT.md §8 `2026-08-14d`「`Hexblade` 从表里删掉（不译）」，原始记录见
 *   `4-临时脚本/2026-08-14-round14/RESOLUTIONS.md` R5。全库只此一处裁决，无翻案记录。
 *   理由：`Spellblade` 在合集里**已定稿**为「咒刃 Spellblade」（`1-Ember汉化插件/compendium/cn/
 *   ember.crucible-adventure.json` 以及 crucible 侧 archetype / talent / playtest / pregens 四份），
 *   再把 `Hexblade` 也译成「咒刃」就是 **1:2 撞名**：同一个中文「咒刃」指两样东西
 *   （Crucible 的战士**原型** vs dnd5e 的邪术师**宗主**），而后者是 dnd5e 侧附带项、
 *   上游 Warlock 类页面**已删该子职**、合集里也没有对应译法。为它制造一处主线撞名不划算。
 *   ⚠ 撞名判据是「同一个中文串在库里指两样东西」，**不是**「两者会同屏」—— 2026-08-15 实测
 *   两者根本不同屏：`Hexblade` 只在**神祇日志页**的表头标签区，`Spellblade` 在 crucible-adventure
 *   的角色 `Veiled Chain Investigator` 身上（及 crucible 系统的原型/天赋表），落在角色卡一侧。
 *   引用这条裁决时别把理由写成「同屏」，那是可证伪的、且实测为假。
 *   ⚠ 它**不是死键**（删的是有效键，不是清死代码）：2026-08-15 探针
 *   `4-临时脚本/2026-08-15-round20/probe_patrons2.mjs` 只读打开两个孪生包的 LevelDB，
 *   `Hexblade` 仍在 `system.warlocks` 里、**2 叶**，两包各 1 叶，都在神祇页 **`Ku'arta`**；
 *   同一次跑出 148 页 / 130 叶 / 26 唯一值，与上面那份值集清单逐字相符（探针自证没空转）。
 *   所以后果是确定的：`Ku'arta` 页的标签恒显示「邪术师宗主：Hexblade」，共 2 处，**故意留英**。
 *   下一轮的英文残留 / 覆盖率扫描若报到这 2 处，属**预期内**，按本条驳回，别当缺陷补。
 */
// ⚠ `Light` 与 crucible lang 的 `WEAPON.CATEGORIES.Light`＝轻型 同英文不同域（神祇领域「光」
//   vs 武器分类「轻型」）；B 段 2026-08-15 报过，已裁：不统一。
const DIVINE_DOMAINS = {
  "Knowledge": "知识", "Life": "生命", "Light": "光", "Order": "秩序",
  "Peace": "和平", "Time": "时间", "Trickery": "诡术", "Twilight": "暮光",
  "War": "战争", "Other": "其他"
};
// ⚠ `Celestial` 与 crucible lang 的 `TAXONOMY.CATEGORIES.Celestial`＝天界生物 同英文不同域：
//   这里是**邪术师宗主**（上屏拼作「天界宗主」，见上面的取名依据），那里是生物分类学的类别名。
//   本文件 KNOWLEDGE 表里的 `Celestials`＝天界生物 才是与那条对应的。B 段报的这条不是缺陷。
const WARLOCK_PATRONS = {
  "Archfae": "至高妖精", "Archfey": "至高妖精", "The Archfey": "至高妖精",
  "Celestial": "天界", "Fathomless": "深海", "Fiend": "邪魔", "The Fiend": "邪魔",
  "Genie": "巨灵", "Great Old One": "远古旧日支配者", "The Great Old One": "远古旧日支配者",
  // `Hexblade` 有意缺席（§8 2026-08-14d 已裁：不译）。理由与实测后果见本表上方的注释块。
  "The Undead": "不死", "Other": "其他"
};
// ⚠ `Draconic` 在本文件里**有意两译**：这里是术士起源＝龙脉（合集 Sorcerer 页的小节译法），
//   LANGUAGES 那张表里是语言名＝龙语。`scan_cross_channel` B 段会把它报成「.mjs 内部同英文异译」，
//   是判据的定义域问题（按英文串比，不看归属），**不是缺陷**，别再统一。
const SORCEROUS_ORIGINS = {
  "Draconic": "龙脉", "Spellfire": "法术火焰", "Wild Magic": "狂野魔法"
};

/**
 * 血统稀有度。上游 crucible-async.mjs:188 `traits.tags.push({text: \`Rarity: ${rarity.titleCase()}\`})`，
 * 值域就 3 个（dnd5e-async.mjs:370-386 的 ANCESTRY_RARITIES = common / rare / extinct，titleCase 后如下）。
 */
const RARITIES = {
  "Common": "常见", "Rare": "稀有", "Extinct": "已灭绝"
};

/**
 * `[[/ancestry …]]` / `[[/culture …]]` / `[[/path …]]` 里**上游根本不存在**的 identifier。
 *
 * 成因同 MISSING_LANGUAGES：真实 id 一律带 `ember` 前缀（如 `emberHuman`），这 18 个不带前缀的
 * 在合集里查不到，`enrichAncestry` 等三个增强器的兜底是 `const name = ix?.name || id`
 * （ember.mjs:22934 / 22954 / 22986），于是把 id 原样当名字渲染成「血统：Kavir」——
 * 前缀中文、叶子英文，夹在中文正文里。实测两个孪生包合计 78 处（每包 39 处）。
 *
 * 译名不是新拟的，逐条取自**合集里同名文档已定稿的双语 name**（ember.crucible-character /
 * ember.character 的 entries，机械核对过），所以跟旁边正常渲染的 `[[/ancestry emberHuman]]`
 * →「人类 Human」写法完全一致：
 *   Arcturian 阿克图里安 · Bejak 贝雅克 · Human 人类 · Kessian 凯西安 · Keth 凯思 · Kivahr 基瓦尔 ·
 *   Lumek 卢梅克 · Oaken 奥肯 · Ordani 奥尔达尼 · Waerd 瓦尔德 · Wirrun 威伦 ·
 *   Anchorite Marine 隐修士陆战队员 · Cindaric Initiate 辛达里克入门者 · Flameguard Militia 焰卫民兵 ·
 *   Nightwatch 夜巡者 · Shard God Devotee 碎片之神信徒。
 * 两条例外：`CindaricAdherent` 合集里没有同名文档，取 glossary_ec 的「辛达里克信徒」；
 * `Kavir` 全库（含英文基线）只有这 8 处悬空引用、无任何定稿，音译作「卡维尔」暂定。
 *
 * ⚠ 只登记这些**悬空 id**。真实 id 的叶子已经是 babele 译好的中文，查表落空原样返回，不会被改。
 */
const MISSING_ANCESTRIES = {
  "Arcturian": "阿克图里安 Arcturian",
  "Human": "人类 Human",
  "Kavir": "卡维尔 Kavir",
  "Keth": "凯思 Keth",
  "Kivahr": "基瓦尔 Kivahr",
  "Lumek": "卢梅克 Lumek",
  "Oaken": "奥肯 Oaken",
  "Wirrun": "威伦 Wirrun"
};
const MISSING_CULTURES = {
  "Arcturian": "阿克图里安 Arcturian",
  "Bejak": "贝雅克 Bejak",
  "Kessian": "凯西安 Kessian",
  "Ordani": "奥尔达尼 Ordani",
  "Waerd": "瓦尔德 Waerd"
};
const MISSING_PATHS = {
  "AnchoriteMarine": "隐修士陆战队员 Anchorite Marine",
  "CindaricAdherent": "辛达里克信徒 Cindaric Adherent",
  "CindaricInitiate": "辛达里克入门者 Cindaric Initiate",
  "FlameguardMilitia": "焰卫民兵 Flameguard Militia",
  "Nightwatch": "夜巡者 Nightwatch",
  "ShardGodDevotee": "碎片之神信徒 Shard God Devotee"
};

/** 带前缀的标签：`前缀: 名字` → `中文前缀：中文名字` */
const PREFIXED = [
  { en: "Attunement", cn: "同调", table: ATTUNEMENTS },
  { en: "Language", cn: "语言", table: LANGUAGES },
  { en: "Knowledge", cn: "知识", table: KNOWLEDGE },
  { en: "Music Mood", cn: "音乐氛围", table: MOODS },
  // 下面四条的叶子名**绝大多数**已经是中文：三个 character-option 增强器取的是 compendium index 的
  // `name`（babele 已译），crucible 的 enrichTalent 取的是 talentIndex.name（同样已译）。
  //   ember.mjs:22934 `Ancestry: ${name}` / :22954 `Culture: ${name}` / :22986 `Path: ${name}`
  //   crucible-compiled.mjs:46838 `Talent: ${talentIndex.name}`（相邻的 knowledge/language 都走
  //   _loc，只有 talent 这条漏了 i18n，crucible 汉化插件那边又没有运行时字符串层）
  // 前三条挂的不是空表而是 MISSING_* —— 那三个增强器查不到 identifier 时会把 **id 当名字**
  // 渲染（`ix?.name || id`），合集里有 78 处这种悬空引用，只能按 id 兜底，见 MISSING_ANCESTRIES。
  // 注意：EXACT 里那三个裸词 Ancestry/Culture/Path **不是**给这里用的，见 EXACT 的注释。
  { en: "Ancestry", cn: "血统", table: MISSING_ANCESTRIES },
  { en: "Culture", cn: "文化", table: MISSING_CULTURES },
  { en: "Path", cn: "道途", table: MISSING_PATHS },
  { en: "Talent", cn: "天赋", table: {} },

  // 音景增强器的前两支**不在这里**：它们的叶子可能带 ` (Calm)` / ` (Tension)` 尾巴，
  // PREFIXED 是整串查表、接不住复合叶，已改成 PATTERNS 里那条 `^(Music|Environment): …` 正则。

  // 六边形 HUD 的四条 data-tooltip（templates/applications/hex-hud.hbs:13/47/50/52）。
  // 宿主 EmberHexHUD 的 classes 含 "ember"，闸放行、translateNode 也走到了，
  // 只是查表形状对不上带动态尾巴的串。尾巴是 babele 已译的地名/群系名/地形名，故用空表。
  { en: "Area Map", cn: "区域地图", table: {} },
  { en: "Location", cn: "地点", table: {} },
  { en: "Biome", cn: "生物群系", table: {} },
  { en: "Terrain", cn: "地形", table: {} },

  // 神祇页的三类子职标签（ember.mjs:36354 / 36359 / 36364，见上面三张表的注释）
  { en: "Divine Domain", cn: "神圣领域", table: DIVINE_DOMAINS },
  { en: "Warlock Patron", cn: "邪术师宗主", table: WARLOCK_PATRONS },
  { en: "Sorcerous Origin", cn: "术士起源", table: SORCEROUS_ORIGINS },

  // 创角向导「血统」步骤的两条 trait 标签（crucible-async.mjs:188-189）。
  // 宿主 EmberHeroCreationSheet 类名以 Ember 开头，闸放行，只是原先没有这两条前缀。
  // lifespan 的值已由 babele 译成中文（mappings.mjs:293 有 lifespan 映射），故用空表只换前缀。
  { en: "Rarity", cn: "稀有度", table: RARITIES },
  { en: "Lifespan", cn: "寿命", table: {} },

  // 事件/地点页的标识符标签（ember.mjs:35842 `Identifier: ${identifier}`）。
  // 尾巴是数据 id，不译，故用空表。
  { en: "Identifier", cn: "标识符", table: {} },

  // 法典日志给每条已完成事件挂的 context 串（ember.mjs:25236
  // `const context = event.root ? \`Quest: ${event.root.label}\` : "Standalone Event"`，:25250 消费）。
  // 同组的 "Standalone Event" 早就在 EXACT 里（→独立事件），只差这条前缀；EXACT 里那个裸
  // `Quest`（codex/quests.hbs:7 的分类标题）是整串匹配，接不住 `Quest: 任务名` 这种形状。
  // 叶子是 babele 已译的任务名，故用空表。
  { en: "Quest", cn: "任务", table: {} }
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
  "Summarize Token Maker Part Usage": "统计指示物制作器部件用量",          // 49532
  "Ember: Teleport Destination": "余烬：传送目的地",                    // 61789
  "Elevator Controls": "升降机控制",                                 // 67247
  "Toggle Corpuleth Damage": "切换尸团怪 Corpuleth 伤害状态",           // 73312
  "Aedir Signalpost Generator Room Switch": "艾迪尔信号哨站 Aedir Signalpost 发电机房开关", // 95126
  "Elevator Destination": "升降机目的地",                             // 95361
  "Steam Cleansing Cutoff": "蒸汽净化切断阀",                         // 95691（按钮是 Close/Open Valve，Cutoff 在这里是阀）
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
  "Apply Soulbound Progression": "应用魂缚进阶",                       // 126638 / 126659
  // 2026-08-14 第十四轮补：这五个框原先认不出来，整框（含正文与按钮）都留在英文
  "Aedir Elevator Control": "艾迪尔 Aedir 升降机控制",                  // 65249（appv1 Dialog.prompt）
  "Select Destination": "选择目的地",                                 // 96220
  "Arcturel Elevator": "阿克图瑞尔 Arcturel 升降机",                   // 96401
  "Arcturel Lift": "阿克图瑞尔 Arcturel 升降台",                       // 96462
  "Silver Beam Security Control": "银光束 Silver Beam 安保控制",       // 113769
  // 2026-08-15 第十七轮补：裁 `Seal` 那条 `MJS_ORPHAN_CN` 时顺带查出来的**死键**。
  // `EmberTarPit#_configureDialog`（ember.mjs:107610）是在方法体里写 `config.window.title = "Tar Pit"`，
  // 不是 DEFAULT_CONFIG 里的静态标题，先前几轮枚举 `window: {title:` 时整类漏掉。
  // 结果这个框一直认不出来，标题与三个按钮（Open / Spawn / Seal，:107606-107608）全是英文，
  // 而 DIALOG_UI 里那条 `Seal` 只对 CorpinSanctuaryElevator 生效（它继承基类标题
  // `Elevator Controls`，在本表里）。补上标题，三个按钮跟着解锁。
  "Tar Pit": "焦油坑"                                                 // 107610
  // 缺席说明：ember.mjs:95615 那个 `dialog:{title,icon,description}` 少写了 window 这一层，
  // DialogV2 读不到，标题实际落到基类兜底的 `Interactable: ${id}`（62795），
  // 所以 "Aedir Signalpost Stealth Field Generator" 不在本表，由 DIALOG_TITLE_PATTERNS 认。
};

/** 动态拼出来的 Ember 对话框标题，只用来认框（能翻的那几条在 PATTERNS 里） */
const DIALOG_TITLE_PATTERNS = [
  /^(?:Award|Revoke|Activate) Attunement: /,   // ember.mjs:3051 / 3178 / 3142 / 23181
  /^Token Maker Part Usage: /,                 // ember.mjs:49557
  /^Interactable: /,                           // ember.mjs:62795 基类兜底标题
  /^Vantage Point: /,                          // ember.mjs:67414
  /^".+" - \d+ match(?:es)? in \d+ location(?:s)?$/  // ember.mjs:23332 日志搜索结果框
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
  // 同调短名：`Activate Attunement: X` 那个框的正文里，label 被 <strong> 单独切成一个文本节点
  // （ember.mjs:3134），只能靠这张作用域表接住。放在最前面，后面的显式键优先级更高。
  ...ATTUNEMENTS,
  // 通用按钮
  "Interact": "交互",                        // 62794，EmberInteractable 无显式按钮时的兜底 OK
  "Cancel": "取消",                          // 113780
  "Observe": "观察",                         // 67415
  // ⚠ `Close` 是**关掉这个框**的 OK 按钮（49559 `ok:{label:"Close"}` 指示物制作器部件用量框、
  //   112049 同形状的矿车目的地框），不是任何游戏术语。B 段 `MJS_ORPHAN_CN` 报的多数写法「解除」
  //   （6/10）纯属共现：英文闸 `\bClose\b` 全库 10 叶里没有一叶是 UI 按钮 —— 是「Close ties 密切
  //   关系」「Close by 附近」「Up Close and Dangerous 近身而危险」，以及那条追踪距离档
  //   「In Reach – In View – Nearby – Close – Distant」。判据不分词性也不分域。
  //   2026-08-15 第十七轮已裁：保持「关闭」，compendium 一叶不动。
  "Close": "关闭",                           // 49558 / 112049
  "Confirm": "确认",                         // 112074
  "Change": "更改",                          // 32946 / 63640
  "Import": "导入",                          // 34758
  "Search": "搜索",                          // 23301 / 49533
  "Activate": "激活",                        // 110866
  // 升降机 / 矿车 / 转运
  // ⚠ `Ascend` / `Descend` 2026-08-15 第十七轮由「上行 / 下行」改为「**上升 / 下降**」——
  //   与 `Forwards/Backwards/Unreachable`（下面 112063-112065 那三条）**同一类依据、同一个先例**：
  //   GM 指南把这台升降机的按钮**逐字写过**。`Aedir Signalpost` 的 `Rotating Elevator` 页
  //   （孪生包各一份）正文里那张符文表就是 `Top / Sky / Ascend` →「上方 / 天空 / **上升**」，
  //   同页还有「升降机**上升**一层」「**上升**到顶部层级」「你会看到一个**下降**选项提示」
  //   （英文原句 `you will be prompted with an option to descend`，说的正是这个对话框）
  //   「识别其含义为『**下降**』」。玩家/GM 照着日志去按按钮，「上行 / 下行」两条对不上。
  //   B 段 `MJS_ORPHAN_CN` 给的多数写法（Ascend→「触碰」4/6、Descend→「荆棘」8/12）是共现噪声：
  //   Descend 的 12 叶里 8 叶来自 `The Expedition Challenge` 那首谜语（同叶有 thorny/brambles），
  //   Ascend 的 6 叶里 2 叶是 `Cosmos.Ascendancy`（飞升，另一个域）、2 叶是「攀登山路」的散文义。
  //   真正同域的证据只有 Rotating Elevator 那 2 叶，取它。
  "Move": "移动", "Ascend": "上升", "Descend": "下降", "Call": "呼叫",  // 67273-67274 / 95344-95355
  // ⚠ `Seal` 有**两个**宿主，取的是有 GM 指南背书的那一个：
  //     · 107606-107608 焦油坑（`Tar Pit`）菜单：closed 时是 `Open`，否则 `Spawn` / `Seal`；
  //     · 99592 CorpinSanctuaryElevator 的第三个按钮。
  //   `Disturbed Earth` 的 `Muddy Waters` / `Death on the Vine` 两页（孪生包共 4 叶）在
  //   「操作焦油坑」一节把这三个按钮**逐条列过**：「**打开**：此按钮会播放一段动画…」
  //   「**封堵**：此按钮会播放一段动画…」「**生成**：把 Actor 拖进…」。
  //   所以 2026-08-15 第十七轮把「封闭」改成「**封堵**」，并按同一段补上 Open / Spawn / Tar Pit
  //   （补之前那个菜单是「打开(英) / 封闭 / 生成(英)」三选二半英半中）。
  //   B 段报的「保护」（8/12）是共现：`\bSeal\b` 的 12 叶是 `Burnished Seal` 烫金印记、
  //   `Jade Seal` 玉印、Ordain 的铜币面额 `Seal`，全是名词义，与按钮动词不同域。
  "Seal": "封堵",                             // 107608（焦油坑）/ 99592（CorpinSanctuaryElevator）
  "Open": "打开",                             // 107606，焦油坑关闭状态下的唯一按钮
  "Spawn": "生成",                            // 107607，焦油坑的生成按钮
  // （窗口标题 `Tar Pit` 在 DIALOG_TITLES 里 —— 那张表兼管**认框**，不加进去这三个按钮
  //   连解锁的机会都没有，见 DIALOG_TITLES 的注释。）
  // `Unseal` 只出现在 CorpinSanctuaryElevator 的 state=2（99689），此时按钮**只有它一个**，
  // 与上面的「封堵」永不同屏（state 0/1 才出 Ascend/Descend/Seal），GM 指南也没写过它。
  // 故不为了配对硬改成「解除封堵」，保持「解封」。
  "Unseal": "解封",                           // 99689，同一台升降机「已封闭」状态下的唯一按钮
  "Repair": "修复",                           // 110357，警钟被破坏后 state=2 时替换掉的按钮
  "Raise Elevator": "升起升降机", "Spawn Construct": "生成构装体", "None": "无", // 65250 / 65253 / 65255
  "Clockwise": "顺时针旋转",                  // 95333，与日志里「左转→顺时针旋转」的说法对齐
  "Counter-Clockwise": "逆时针旋转",           // 95336
  // 112063-112065 这三个字段名 GM 指南**逐字写过**：Yakoshta Mine 的 Area Overview /
  // Blue Track / Red Track 三页（两个孪生包共 6 叶）都是 `<li><p>后退</p></li><li><p>前进</p></li>
  // <li><p>无法抵达</p></li>` —— 玩家照着日志找矿车对话框，原译「前进方向 / 后退方向 /
  // 无法到达」三条全对不上（英文闸下「后退方向」全库 0 叶、「无法到达」2 叶且都是别处正文，
  // 「无法抵达」15 叶）。2026-08-15 主控裁决批次落地时改到合集定稿。
  "Forwards": "前进", "Backwards": "后退", "Unreachable": "无法抵达", // 112063-112065
  "Tradeway": "贸易道", "Underbelly": "底腹区", "Construct Assembly": "构装体装配区", // 114360-114362
  // 场景机关按钮
  // ⚠ `Ring` 是**动词**：宿主框标题就是 `window: {title: "Ring Alarm Bell?"}`（ember.mjs:110323），
  //   三个按钮 label 是 Ring / Ring / Destroy。B 段的 `MJS_ORPHAN_CN` 拿合集英文闸下的多数写法
  //   「戒指」来比（30 叶全是名词义的 ring），是判据不分词性，不是缺陷；这里只能是「敲响」。
  "Ring": "敲响", "Destroy": "破坏",           // 110325-110327
  "Enable Flow": "开启流量", "Disable Flow": "关闭流量",                // 110378-110379
  "Close Valve": "关闭阀门", "Open Valve": "打开阀门",                  // 95692-95693
  // 100211-100213（疏浚阀门动词）。`Befoul` 2026-08-15 由「污染」改「污化」：英文闸 `\bBefoul\b`
  // 3 叶全作「污化」（斯莱思的动作名叶 `污化水域 Befoul Waters` + 传记里「污化土地、水域与生灵」），
  // 「污染」在库内 196 叶全是 contamination / contaminant 的名词义（同一座塔的
  // 「处理污染」「净化污染物」就在隔壁），拿它译 Befoul 会跟同场景的 Contamination 撞车。
  "Fill": "注满", "Purge": "排空", "Befoul": "污化", "Cleanse": "净化",
  // 「供电」全库 0 叶：合集一律作「供能」（31 叶），而且正是同一座塔的同一件事 ——
  // Aedir Signalpost 的 Generator Room 页 `<h2>恢复供能</h2>`「为了恢复发电机的供能」，
  // Signal of Intent 的 Traversing the Tower 页 `<h4>卢克萨鲁姆同调：恢复供能</h4>`。
  // 2026-08-15 主控裁决批次落地时改齐。
  "Disable Generator": "关闭发电机", "Restore Power": "恢复供能",        // 95622-95623
  "Restore or disable power to the Stealth Field Generator?": "恢复还是切断隐形力场发生器的供能？", // 95618
  "Engage Lockdown": "启动封锁", "Lift Lockdown": "解除封锁",            // 113776
  // 银光束安保控制的两段正文：`<strong>lockdown</strong>` 把整段切成三个文本节点
  // ⚠ 这三条是**句子碎片**，不是术语。B 段拿 crucible lang 的
  //   `ACTOR.FIELDS.movement.engagement.labelShort`＝交战 来比这里的 `Engage`＝启动，比的是
  //   「启动封锁」的动词与战斗中的「交战」状态，不同域；已裁：不统一。
  "Engage": "启动", "Lift the": "解除", "lockdown": "封锁",              // 113770-113774
  "? Security doors lock, the alarm sounds, and both construct elevators descend to the Construct Assembly.":
    "？安保门将上锁，警报鸣响，两台构装体升降机将下降至构装体装配区。",      // 113771-113772
  "? Security doors unlock, the alarm silences, and both construct elevators return to the Tradeway.":
    "？安保门将解锁，警报停止，两台构装体升降机将返回贸易道。",            // 113773-113774
  "Machine On": "机器开启", "Machine Off": "机器关闭", "Machine Destroyed": "机器损毁", // 96499-96501
  "Defenses Inactive": "防御未激活", "Defenses Active": "防御已激活", "Orb Destroyed": "法珠已摧毁", // 97174-97176
  // 97256-97261 是同一个框里并排的三个按钮（贴图 TowerBroken/TowerDamaged/TowerRepaired），
  // 原译「破碎」与另两条的体式不齐（那两条都带「已」），而且裸「破碎」正是 crucible 的状态名
  // （crucible.rules 的 Conditions.pages.Broken＝破碎 Broken）—— 这里是石柱的受损档位，不是状态。
  // 取「已破碎」：既与 已受损/已修复 同体式，又保住词根（英文闸 `\bBroken\b` 75 叶：破碎 53 /
  // 破损 10 / **已破碎 2** / 损毁 2，「已破碎」在库内有据，不是新造）。
  // ⚠ 正因如此，B 段拿 crucible lang 的 `ACTIVE_EFFECT.STATUSES.Broken` / `ITEM.FIELDS.broken.label`
  //   ＝破碎 来比必报 `MJS_LANG_DRIFT` —— **那正是本条要避开的撞名**，已裁：不统一，别改回「破碎」。
  "Broken": "已破碎", "Damaged": "已受损", "Repaired": "已修复",         // 97259-97261
  "Reset Vault": "重置宝库", "Activate Beams": "激活光束",              // 97562-97563
  "Reset Body": "重置躯体", "Awaken Vampyre": "唤醒吸血鬼",             // 115788-115789
  "Lock": "锁定", "Unlock": "解锁",                                   // 110891 / 110893
  "Fully Armored": "全副武装", "Helm Broken": "头盔破损", "Armor Broken": "护甲破损", // 73314-73316
  // 表单字段标签。这几条只在**已认出归属**的框里查，不进全局 EXACT：
  // Type / Size / Speed / Strength 是通用词，Create Weather 那个框（22730-22734）虽然带
  // "ember-hex-selection-dialog" 类名走的是主闸，但主闸这一支现在也把 DIALOG_UI 一并传下去了。
  // `Hex` 取「六边格」：glossary_ec 的 `Hexes`＝六边格 / `Hex Attributes`＝六边格属性 / `Hex Key`＝六边格键值，
  // compendium 英文闸 `\bhex(es)?\b` 下 六边格 62 : 六角格 38（name 叶 2 : 0）。原译「六角格」是少数派。
  // ⚠ `Strength` / `Size` 在这个框里说的是**天气的强度**与**影响范围的尺寸**（Create Weather 表单），
  //   与 crucible lang 的 `ABILITIES.Strength`＝力量、`ACTOR.FIELDS.movement.size.label`＝体型
  //   不是一回事 —— 后两条是角色属性与角色体型。B 段 2026-08-15 报过这两条，已裁：不统一。
  "Origin Hex": "起始六边格", "Type": "类型", "Strength": "强度", "Size": "尺寸", "Speed": "速度", // 22729-22734
  "Illusions": "幻象",                                                                     // 108562
  "Select which illusions are currently active": "选择当前处于激活状态的幻象",                  // 108562
  "Elemental Orbs": "元素法珠",                                                             // 109105
  "How many elemental orbs have been depleted between 0 (shield fully powered) to 6 (shield disabled).":
    "已耗尽的元素法珠数量，0 表示护盾全功率、6 表示护盾停摆。",                                  // 109107
  // 表单标签 / 正文
  "Composition": "构图",                      // 32943 / 63637
  "Search Term": "搜索词",                    // 23271
  "Document Types": "文档类型",                // 23282
  "Case Sensitive": "区分大小写",              // 23285
  "Journal Entry": "日志条目", "Actor": "角色", "Item": "物品", "Roll Table": "随机表", // 23277-23280
  // `Usage` 是结果表的**列头**（49551 `<th>Actor</th><th>Usage</th>`），单元格里填的是
  // "Static, Randomization" —— 问的是「怎么用的」不是「有什么用」，原译「用途」是义项选错了。
  // （compendium 英文闸 `\bUsage\b` 只有 5 叶、其中 3 叶作「用途」，全是正文里的一般义，
  //   与这个列头不同域，不构成反证。）
  // ⚠ **这一改是有意让 `scan_cross_channel` B 段 `MJS_ORPHAN_CN` 12 → 13 的**：新值「使用方式」
  //   在 compendium 英文闸下 0 叶支持，判据必报孤儿。主控 2026-08-15 认过：`Usage` 在这里是
  //   `<th>` 列头、compendium 那 5 叶是正文一般义，两者不同域，B 段这 +1 不是缺陷。
  //   **下一轮不要把它当新漂移重查，也不要为了消掉 +1 把值改回「用途」。**
  // ⚠ 同一条在 **crucible 侧** B 段还会报一次 `MJS_LANG_DRIFT`（对 `ACTION.TABS.usage`＝用法）：
  //   那是 crucible 动作卡的**页签名**（这个动作怎么用），本条是 Ember 指示物制作器结果表的 `<th>`
  //   列头（这个部件被怎么用着），两者不同域，一并裁为不统一。
  "Usage": "使用方式", "Static": "固定", "Randomization": "随机化",       // 49551-49552
  "Token Maker Part": "指示物制作器部件",                                // 49527
  "Enter a part id as template/layer/part, for example kiska/eyes/Fluffy2":
    "以 模板/层/部件 的形式输入部件 id，例如 kiska/eyes/Fluffy2",       // 49528
  "No world Actors use this part.": "世界中没有角色使用该部件。",         // 49554
  // 49555 的 `Part <strong>X</strong> is used by <strong>N</strong> world Actor(s).`
  // 被两个 <strong> 切成三个文本节点，只能按碎片建键；拼回去正好是
  // 「部件 X 被 N 个世界角色使用。」，语序对得上。
  "Part": "部件", "is used by": "被", "world Actor(s).": "个世界角色使用。",  // 49555
  "Select Characters": "选择角色",                                    // 23176
  "Select the characters who should receive the award.": "选择应当获得此项奖励的角色。", // 23177
  "Do you wish to recombine the Party into the Strayhearth Caravan?":
    "是否将队伍重新并入迷炉商队 Strayhearth Caravan？",                  // 18789 / 18824
  "Activate this elevator?": "启动这台升降机？",                        // 67249
  "Direct this elevator to a destination.": "为这台升降机指定一个目的地。", // 114359
  "Choose a destination.": "选择一个目的地。",                          // 96220
  // 升降机目的地下拉（ArcturelElevatorTransit / ArcturelDepthsTransit 的 DESTINATIONS，
  // ember.mjs:96390-96392 等）。⚠ `Region Map` 与 `Area Map` **出现在同一个下拉框里**，
  // 而全库现在两者都译作「区域地图」—— 照现有译名会渲染出两个一模一样的按钮，比留英文更糟。
  // 这里就地区分：`Region Map`＝地区地图（世界六边格大图）/ `Area Map`＝区域地图（局部场景）。
  // 全库层面的拆分（159 叶）见 RESOLUTIONS.md R4，本轮判为不做；这六条是**局部消歧**，
  // 与 R4 将来的落地方向一致，届时不需要回改。
  "Tradeway (Region Map)": "贸易道（地区地图）",                          // 96390
  "Tradeway (Area Map)": "贸易道（区域地图）",                            // 96391
  "Tradeway (Vista)": "贸易道（远景）",                                   // 96392
  "Rock Bottom (Region Map)": "石底镇（地区地图）",
  "Rock Bottom (Vista)": "石底镇（远景）",
  "Arcturel Caverns (Vista)": "阿克图瑞尔洞窟（远景）",
  "The elevator rises to the Tradeway. Select a destination for the party.":
    "升降机将升往贸易道。请为队伍选择一个目的地。",                        // 96401-96402
  "The lift descends into the depths. Select a destination for the party.":
    "升降台将下降至深处。请为队伍选择一个目的地。",                        // 96462-96463
  // 魂缚进阶确认框（126639 / 126660）的正文同样被 <strong> 切碎，按碎片建键；
  // 尾段源码里带换行缩进，靠 translateNode 的空白折叠回退命中。
  // 拼回去是「将 <天赋> 天赋添加给 <角色> ，阶位 1（次等魂印）？」/
  //         「将 <天赋> 天赋于 <角色> 身上升至阶位 2（高等魂印）？」
  "Add the": "将", "talent to": "天赋添加给",                           // 126639
  "at rank 1 (Lesser Soulmark)?": "，阶位 1（次等魂印）？",              // 126639-126640
  "Upgrade the": "将", "talent on": "天赋于",                           // 126660
  "to rank 2 (Greater Soulmark)?": "身上升至阶位 2（高等魂印）？",        // 126660-126661
  "to rank 3 (Deathly Soulmark)?": "身上升至阶位 3（死亡魂印）？",        // 同上（nextRank 只可能是 2 或 3）
  // `Make <strong>X</strong> the active attunement for Y.` 的首段（ember.mjs:3134）；
  // 其余部分是插值整句，走 PATTERNS
  // 2026-08-15 第十六轮删掉了 `"and lose": "并失去"` —— **上游永远不会产出这个片段**：
  // ember.mjs:3135-3138 是 `const clauses=[]; if(loseItem) clauses.push(\`lose @UUID…\`);
  // if(gainItem) clauses.push(\`gain @UUID…\`); content += \` You will ${clauses.join(" and ")}.\`;`，
  // 压栈顺序写死 lose 在前，所以两个 @UUID 锚点之间的那个文本节点只可能是 ` and gain `。
  // （单独一条 lose 的情形走 PATTERNS 里 `You will (lose|gain)` 那条可选组，与本键无关。）
  // ⚠ `Make` 是**句首碎片**（拼回去是「将 X 设为 Y 的当前同调。」），不是术语；B 段的
  //   `MJS_ORPHAN_CN` 拿合集里 `\bmake\b` 的多数写法（「成功」，来自 make a check 之类）来比，
  //   是判据不认碎片键，已裁：保持。
  "Make": "将", "and gain": "并获得",                                   // 3134-3137
  ".": "。",                                  // 同上：@UUID 链接后面那个单独成节点的句号
  // 「未分配的属性点」框正文（ember.mjs:123311-123317，dnd5e 分支）。
  // 同样被 <strong>Class</strong> / <strong>Path</strong> 切碎；Path 走 EXACT 的「道途」。
  "Class": "职业", "selection.": "的选择中分配。",                        // 123313 / 123316
  "Do you wish to proceed and forego these increases?": "确定要继续并放弃这些提升吗？", // 123317
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
  // 英雄创建向导顶栏的步骤标签。上游把 label 写成裸英文（crucible-async.mjs:24/34/43/63），
  // 经 crucible 的 templates/sheets/creation/header.hbs:7 `{{localize step.label}}` 上屏，
  // 而 Foundry core 与两个插件的 lang 里都没有这几个裸键，localize 原样返回，所以能被这里接住。
  // ⚠ 这几行**不是**「富文本增强器前缀单独出现」—— 那个场景不存在：三个增强器拼的永远是
  //    `Ancestry: 名字` 整串，走的是 PREFIXED（见上）。原来的注释认错了来源，一直没人补 PREFIXED。
  // ⚠ **`Ancestry` 不在这四条里，已于 2026-08-15 移进 EMBER_WINDOW_UI**（原注释把它和另四条
  //    混为一谈，是错的）：crucible-compiled.mjs:16397-16405 的 `STEPS.ancestry.label` 是 i18n 键
  //    `TYPES.Item.ancestry`，而 crucible-async.mjs:18-21 的 `ancestry: {...super.STEPS.ancestry,
  //    template: …}` **只覆盖了 template、没覆盖 label** —— 对照 :24 culture / :34 path /
  //    :43 attunement / :63 token 四条才是裸英文。也就是说 Crucible 世界里 localize 吐出的是
  //    crucible-cn 的中文，这个裸键永远接不到自己人，却仍会去改任何第三方窗口里孤零零的
  //    "Ancestry" 文本节点 —— 是越界风险而不只是死重量。它只在 dnd5e 世界活（ember.mjs:121864
  //    `label: "Ancestry"`），那边宿主是 EmberHeroCreationSheet，作用域表照样够得到。
  "Culture": "文化",
  "Path": "道途",
  "Attunement": "同调",
  // ⚠ `Token`＝**指示物**，2026-08-15 主控裁决，本文件内无例外（另有 15 处 Token 相关键
  //   与 3 处通知模板同批改过）。依据是**宿主 UI 用词**而不是裸词频：Foundry 核心中文包
  //   `modules/foundry_chn/cn.json` 的 TOKEN.* 相关键 指示物 53 : 令牌 0，玩家控件上写的
  //   就是「指示物」；ember lang 侧 11/11 也是指示物（ACTOR.CONTROLS.EmberToken
  //   ＝余烬动态指示物 / EMBER.CONTROLS.Teleport＝传送指示物 / …SHEET.TOKEN＝指示物）。
  //   全库「令牌 212 : 指示物 32」那个多数派**不作数** —— 那 212 处几乎全挤在 GM 指南
  //   更新日志一处，而 compendium 里凡指代 Foundry 对象的 32/32 都是指示物。
  //   连带：Dynamic Token＝动态指示物 · Token Maker＝指示物制作器 · Teleport Token＝传送指示物；
  //   「代币」是错译，不许出现。若某处 Token 是**故事内的实体信物**（青铜信物之类）
  //   而不是 Foundry 对象，不在本裁决范围内，单独列出来别跟着改。
  "Token": "指示物",

  // 恩惠 / 祸骰：原先这里硬列 ±1..±3 六个键，只盖住 18 个取值里的 6 个 ——
  // 上游 ember.mjs:129470 注册的 pattern 是 `@Advantage\[(-?\d)]`，enrichAdvantage(:22890) 拼
  // `+${n} Boons` / `${n} Banes`，定义域是 ±1..±9；合集实测 `@Advantage[-6]` 两个孪生包各 1 处，
  // 落在枚举外，屏幕上就是「-6 Banes」；反过来 ±3 两档全库 0 处，是死键。
  // 现已整体挪进 PATTERNS 的 `/^([+-]?\d+) (Boons|Banes)$/`，一次盖住全值域。
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
  "Ember Dynamic Token Randomization Configuration": "余烬动态指示物随机化配置", // ember.mjs:51486
  "Add Part": "添加部件",                                            // ember.mjs:51608
  "Add Color": "添加颜色",                                           // ember.mjs:51611
  "Save Changes": "保存更改",                                        // ember.mjs:51613
  // ⚠ `Exit` 是创角向导页脚那个 `{action:"close", label:"Exit", tooltip:"Exit Creation"}`
  //   （122292）的**动词**按钮。B 段 `MJS_ORPHAN_CN` 报的多数写法「失败」（12/28）是共现噪声
  //   （同叶有 Failed Save / failed check）；英文闸 `\bExit\b` 的 28 叶**全部**是名词义的出口
  //   或事件标题 —— `Secret Exit` / `Culvert Exit` / `A Misty Exit` / `A Promised Exit` /
  //   `Meandering Exit` / `Forced Exit` / `Clearing the Exit`，一叶也不是 UI 按钮。
  //   动词「退出」与名词「出口」不同域，2026-08-15 第十七轮已裁：保持，compendium 一叶不动。
  "Exit": "退出", "Exit Creation": "退出创建",                        // ember.mjs:122292
  "Complete": "完成", "Complete Creation": "完成创建",                // ember.mjs:122293-122294
  "Create Weather": "创建天气",                                      // ember.mjs:22744（带 ember class，过得了主闸）
  "Teleport": "传送",                                               // ember.mjs:61810（同上）

  // 法典（EmberCodex，ember.mjs:24810）与创角向导（EmberHeroCreationSheet）的模板裸串。
  // 两个宿主都命中 patchRenderedApplications 的 `/^Ember/` 闸，纯粹是原先表里没有这些键。
  "Entry Date": "条目日期",                                          // codex/journal.hbs:7
  "Quest": "任务",                                                  // codex/quests.hbs:7
  "Select a quest from the left menu.": "请从左侧菜单选择一个任务。",      // codex/quests.hbs:40
  "Select a discovered creature from the left menu.": "请从左侧菜单选择一个已发现的生物。", // codex/bestiary.hbs:46
  "Select a character from the left menu.": "请从左侧菜单选择一个角色。", // codex/characters.hbs:48
  "Select a biome or location from the left menu.": "请从左侧菜单选择一个生物群系或地点。", // codex/discoveries.hbs:46
  "Uncategorized": "未分类",                                         // ember.mjs:124593 的兜底分类名
  "Ability Scores": "属性值",                                        // creation/crucible-path.hbs:54
  // 模板是 `<span class="points">{{pool}}</span> Points`，上屏就是「9 点」这种数量后缀，
  // 所以不能取 lang `EMBER.MILESTONE.Dialog.Legend`＝点数（那条是 fieldset 的 legend）。
  // B 段会报这一对，已裁：不统一。crucible 侧同理会报 `TALENT.LABELS.Points.*`＝点数 ——
  // 那是天赋点余额的**独立名词**（「剩余 3 点数」不通，那条本来就该是名词），本条是数量后缀，不同形状。
  "Points": "点",                                                   // creation/crucible-path.hbs:55
  // 下面这三条**与 crucible 本体的 i18n 键逐字同英文**（crucible lang/en.json：
  //   ACTOR.ACTIONS.IncreaseAbility = "Increase Ability Score"
  //   ACTOR.ACTIONS.DecreaseAbility = "Decrease Ability Score"
  //   ACTOR.CREATION.SpendAbilities = "Spend 9 points across 6 ability scores, allocating up to 3 points per ability."）
  // —— Ember 的 creation/crucible-path.hbs 把这三句**硬编码成英文字面量**而没有走键，于是同一台
  // 创角向导上「crucible 自己的按钮」与「Ember 抄过去的按钮」并排出现，两条通道必须一个字都不差。
  // 2026-08-15 第十六轮 B 段查实：Decrease 两边本就同为「降低属性值」，Increase 与那句提示漂了。
  // 方向取 **lang**（crucible-cn 的 ACTOR.ACTIONS.*／ACTOR.CREATION.*）：一来 lang 是整条通道的定本、
  // 覆盖面远大于这一个模板，二来「提高／降低」成对，原值「提升」与隔壁「降低」不成对。
  "Increase Ability Score": "提高属性值",                             // creation/crucible-path.hbs:63（＝ACTOR.ACTIONS.IncreaseAbility）
  "Decrease Ability Score": "降低属性值",                             // creation/crucible-path.hbs:66（＝ACTOR.ACTIONS.DecreaseAbility）
  "Spend 9 points across 6 ability scores, allocating up to 3 points per ability.":
    "将 9 点分配到 6 项属性值中，每项属性值最多可分配 3 点。",           // crucible-path.hbs:71（＝ACTOR.CREATION.SpendAbilities）

  // crucible 的 enrichSpell 给每个法术标签挂的 tooltip（crucible-compiled.mjs:46724，不走 _loc）。
  // 只有把 crucibleSpell 放进增强器包装的闸里才够得到，见 patchEnrichers。
  "Spell tooltips are still TO-DO.": "法术悬浮提示尚未实现。"
};

/**
 * Ember 历法的四个纪元名。硬编码在 dnd5e-async.mjs:145-150 的 CALENDAR_AGES 里，
 * 由 EmberCalendar.parseDate（ember.mjs:4134）拼进 `[[/date …]]` 标记的 data-tooltip：
 *   `${resolvedAge.label} - ${yearsAgo ? \`${yearsAgo} ${relativeLabel}\` : relativeLabel}`
 * relativeLabel 三档同样硬编码（:4130-4132 Current Year / Years Ago / Years From Now）。
 * 译名取合集里已有的写法：AB「野兽时代」、AT「高塔时代」、AS「大破裂之后」
 * （《设定总览》的纪元表，en 侧 `AC = After Creation` 对应 cn「创造之后」；
 *  这里的 label 是 `Age of Creation`，按 `Age of X → X时代` 的构词取「创造时代」）。
 * 注意：标记正文里的 AC/AB/AT/AS 缩写按文件头 :19-20 的约定不译。
 */
const DATE_AGES = {
  "Age of Creation": "创造时代",
  "Age of Beasts": "野兽时代",
  "Age of the Tower": "高塔时代",
  "After Shattering": "大破裂之后"
};

/** 需要按模式改写的（保留其中的动态部分） */
const PATTERNS = [
  // `Result of X` **只在 dnd5e 分支产出**（ember.mjs:22909/22912，crucible 分支走
  // 22910/22913 直接输出 Critical Failure / Critical Success，那两串在 EXACT 里）。
  // 叶子只可能是 `18+` / `8-` 这类 DC 数字串，所以不查表，原样带回。
  { re: /^Result of (.+)$/, cn: (m) => `结果：${m[1]}` },
  // 恩惠骰 / 祸骰。上游 enrichAdvantage(ember.mjs:22890) 拼 `+${n} Boons` / `${n} Banes`，
  // n 自带负号，故符号位写成可选。取代原先 EXACT 里 ±1..±3 那六个枚举键。
  { re: /^([+-]?\d+) (Boons|Banes)$/, cn: (m) => `${m[1]} ${m[2] === "Boons" ? "恩惠骰" : "祸骰"}` },
  // `[[/date …]]` 的 data-tooltip（ember.mjs:4134）。年份随世界时间变动，枚举表盖不住。
  { re: /^(Age of Creation|Age of Beasts|Age of the Tower|After Shattering) - (?:(-?\d+) (Years Ago|Years From Now)|Current Year)$/,
    cn: (m) => m[2]
      ? `${DATE_AGES[m[1]]} · ${m[2].replace("-", "")} 年${m[3] === "Years Ago" ? "前" : "后"}`
      : `${DATE_AGES[m[1]]} · 本年` },
  // 音景增强器的前两支（`EmberSoundscape.enricherHTML`，ember.mjs:16250-16278 三个互斥分支）。
  //   16255  `${channel.capitalize()}: Reset`                → `Music: Reset`
  //   16266  `${channel.capitalize()}: ${arrangement.label}` → `Music: Ankarist Theme`
  //   16267-16268 **同一支**在 dataset.mood 存在时再拼一段：`label += \` (${mood.titleCase()})\``
  //             → `Music: Shent Ruins (Tension)`
  // 最后那种复合叶是原先 PREFIXED 两条接不住的（整串查表落空 → 前缀中文、叶子全英）。
  // channel 只有 music / environment 两个（ember.mjs:15643），mood 只有 calm / tension（:15606），
  // 所以尾巴写成穷举的可选组，不会把别的「…（X）」形状吃掉；叶子查不到照旧原样带回。
  // 已发布语料里 46 个 `[[/soundscape …]]` 标记暂无一条同时带 soundscapeId 与 mood，
  // 这条是给 GM 自己写的标记兜底的。
  { re: /^(Music|Environment): (.+?)(?: \((Calm|Tension)\))?$/,
    cn: (m) => `${m[1] === "Music" ? "音乐" : "环境音"}：${translateLeaf(m[2], ARRANGEMENT_LEAVES)}`
             + (m[3] ? `（${MOODS[m[3]]}）` : "") },

  { re: /^Award Attunement: (.+)$/, cn: (m) => `授予同调：${m[1]}` },
  { re: /^Revoke Attunement: (.+)$/, cn: (m) => `撤销同调：${m[1]}` },
  { re: /^Activate Attunement: (.+)$/, cn: (m) => `激活同调：${translateLeaf(m[1], ATTUNEMENTS)}` },
  // 世界时钟拼的是整串 `Day 43 - 12:00`（ember.mjs:24576），法典日志表头是纯 `Day 43`
  // （ember.mjs:25243），一条正则同时吃掉两种。原先那条 `^Day\b(.*)$` 兜底会把整串译成
  // 「日 43 - 12:00」，还会误伤远景资源名 `Day, Generic` / `Day, Clear`（ember.mjs:32102），已删。
  { re: /^Day (\d+)\b(.*)$/, cn: (m) => `第 ${m[1]} 天${m[2]}` },

  // 上游没有 borel / kost 这两个语言 id，enrichLanguage 直接 `return new Text(match)`
  // （ember.mjs:126542），正文原样吐出裸标记 `[[/language borel]]`
  // 字符类必须比上游宽：上游用 `(\w+)` 且无 `u`，`[[/language moiré]]` 因此完全不进增强器，
  // 我们这条兜底要是照抄 `\w` 就同样接不住它（实测 4 处一直漏在正文里）。用 `\p{L}` + `u`。
  // ⚠ 加了 `u` 之后裸 `]` 是语法错误（Lone quantifier brackets），必须写成 `\]\]`。
  { re: /^\[\[\/language ([\p{L}\w-]+)\]\]$/u, cn: (m) => `语言：${MISSING_LANGUAGES[m[1]] ?? m[1]}` },

  // 同一形状的知识领域裸标记：crucible 的 enrichKnowledge 查不到 id 时同样
  // `return new Text(match)`（crucible-compiled.mjs:46815）。文案与 crucible-cn 的
  // `ACTOR.KnowledgeSpecific = 知识：{knowledge}` 同形。MISSING_KNOWLEDGE 查不到再查 KNOWLEDGE
  // （id 首字母大写的写法，如 `[[/knowledge Shent]]`，两张表任一命中即可），都查不到就原样带回 id。
  { re: /^\[\[\/knowledge (\w+)]]$/,
    cn: (m) => `知识：${MISSING_KNOWLEDGE[m[1]] ?? KNOWLEDGE[m[1]] ?? m[1]}` },

  // 动态拼出来的窗口标题
  { re: /^Interactable: (.+)$/, cn: (m) => `可交互物：${m[1]}` },          // ember.mjs:62795 兜底标题
  { re: /^Token Maker Part Usage: (.+)$/, cn: (m) => `指示物制作器部件用量：${m[1]}` }, // ember.mjs:49557

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
    // ⚠ 英文是 `Region map`：按既定裁决 `Region Map`＝地区地图 / `Area Map`＝区域地图，
    //   原译「区域地图」撞了 Area Map（同一批裁决见 DIALOG_UI 里升降机目的地那六条）。
    cn: (m) => `是否${m[1] ? "完成此事件并" : ""}将队伍转移到地区地图的通路 Pathways 区段？` }, // ember.mjs:99653

  // 日志搜索框：标题与「无结果」两句
  { re: /^"(.+)" - (\d+) match(?:es)? in (\d+) location(?:s)?$/,
    cn: (m) => `“${m[1]}” — 在 ${m[3]} 处位置找到 ${m[2]} 个匹配` },                  // ember.mjs:23332
  { re: /^No results found for [“"](.+)[”"]\.$/, cn: (m) => `未找到与“${m[1]}”匹配的内容。` }, // ember.mjs:23351（源码是 &ldquo;/&rdquo; 实体，DOM 里是弯引号）
  { re: /^Vantage Point: (.+)$/, cn: (m) => `制高点：${m[1]}` },                      // ember.mjs:67414

  // `Make <strong>X</strong> the active attunement for Y.` 的后半段（ember.mjs:3134-3137）。
  // 尾巴 `You will lose/gain …` 只有在有 gainItem/loseItem 时才拼上，故写成可选组。
  { re: /^the active attunement for (.+)\.(?: You will (lose|gain))?$/,
    cn: (m) => `设为 ${m[1]} 的激活同调。${m[2] ? (m[2] === "lose" ? "你将失去" : "你将获得") : ""}` },

  // 法典生物条目的「威胁」行（ember.mjs:124594 `Threat ${actor.threat}`）。
  // ⚠ actor.threat 取的是 crucible 的 `system.advancement.threat`
  // （crucible-compiled.mjs:36437 / 42613 `adv.threatLevel * adv.threatFactor`），是**数字**不是
  // minion/normal/elite/boss 那种档位键，别按档位表写。措辞与 crucible-cn 的
  // `ACTOR.ADVERSARY.ThreatLevelSpecific`「威胁等级 {threat}」对齐。
  { re: /^Threat (-?[\d.]+)$/, cn: (m) => `威胁等级 ${m[1]}` },

  // 创角向导血统步骤的图注兜底（crucible-async.mjs:180，page.system.banner.caption 为空时用它）
  { re: /^An example (.+) character\.$/, cn: (m) => `一位${m[1]}角色的示例。` },

  // 「未分配的属性点」框正文的插值段（ember.mjs:123312 / 123315，dnd5e 分支）
  { re: /^You have (\d+) unspent ability (points|increases) to allocate as part of your$/,
    cn: (m) => `你还有 ${m[1]} 点未分配的属性${m[2] === "points" ? "点" : "提升"}，需要在` },

  { re: /^Outcome (\d+)$/, cn: (m) => `结果 ${m[1]}` },                                // ember.mjs:36990 新增结果的默认名
  { re: /^(\d+) Others$/, cn: (m) => `其他 ${m[1]} 项` },                              // ember.mjs:25551 六边形 HUD 的事件折叠项
  // ember.mjs:23955 `${att.label} Rank ${r}`。表里查不到就返回整串原文（等于不动），
  // 这样这条宽正则不会误伤别的「… Rank N」。「阶位」取 lang 的 EMBER.ATTUNEMENT.Rank。
  { re: /^(.+) Rank ([1-5])$/, cn: (m) => (ATTUNEMENTS[m[1]] ? `${ATTUNEMENTS[m[1]]} 阶位 ${m[2]}` : m[0]) },

  // 日历条风向箭头的 tooltip（ember.mjs:24673 `${label} (${speed} mph)`）。
  // 前半截的档位名已经由 patchWeatherLabels 在数据侧换成中文了，这里只处理拼死在模板串里的
  // 单位 `mph` —— 它不是数据，改配置碰不到。写成宽正则不怕误伤：PATTERNS 只在 Ember 自己的
  // 窗口、放行过的注入子树和增强器结果上跑，够不到别的模块的界面。
  { re: /^(.+) \(([\d.]+) mph\)$/, cn: (m) => `${m[1]}（${m[2]} 英里/时）` }
];

/* ------------------------------------------------------------------ *
 * 历法月名 / 星期名：**故意不在这里做**（2026-08-14 第十四轮删除三张表 + patchCalendarNames）
 *
 * 原先这里有 CALENDAR_MONTHS / CALENDAR_DAYS / CALENDAR_DAY_ABBR 三张表，配一个
 * patchCalendarNames() 去改写 `CONFIG.time.worldCalendarConfig` 与 `game.time.calendar` 的
 * `months.values[].name` / `days.values[].name` / `.abbreviation`。那段注释还断言
 * 「seasons 走 i18n 但没用，日期串由**月名**拼出来」—— **因果整个写反了**，且改的数据全库没有读者：
 *
 *   ① 日期串真正的通道是 **seasons + i18n**：
 *      ember.mjs:4063-4068 `formatEmberDate` → `` `${dayOfMonth+1} ${_loc(season.name)}, ${age.abbreviation}${年}` ``，
 *      :4077-4084 `formatEmberDateTime` 同构；season 取 `calendar.seasons.values[components.season]`，
 *      而 seasons 的 name 在 EMBER_CALENDAR_CONFIG（ember.mjs:3650-3660）里存的就是
 *      `EMBER.CALENDAR.SEASONS.SEEDING` 这类 i18n 键，months 的 name 才是裸英文（:3626-3634）。
 *      ⇒ **`lang/cn.json` 里那 6 条 `EMBER.CALENDAR.SEASONS.*`（播种/绽放/耕耘/拾取/凋零/寂止）
 *        是活的、唯一生效的通道，任何时候都不许删。**
 *   ② 月名 / 星期名 **零读者**（v14.365 + ember 0.6.x 实测，判据可复跑）：
 *      modules/ember/scripts 下 `grep -rE "months\.values|days\.values|month\.name|monthName|dayOfWeek|weekday"`
 *        → 1 命中（ember.mjs:35870 `weekday: "long"`，是 Intl 选项，与本表无关）；
 *      modules/ember/templates 下 `grep -riE "month|weekday|abbrevi"` → 0；
 *      systems/crucible 下同组模式 → 0；
 *      Foundry core `grep -rE "months\.values|days\.values" client/ common/` → 只有长度/序号算术
 *        （client/data/calendar.mjs:81/256/257/265/282/296/297 与 documents/active-effect.mjs:356-357），
 *        `formatTimestamp`（calendar.mjs:384-392）取的是 `month.ordinal` 不是 name；
 *      core `grep -r "\.abbreviation" client/ common/` → 0。
 *      ⇒ 改了没有任何渲染方，而 `log("已改写历法里 N 个月名/星期名")` 还会打出非零的 N，
 *        看起来像是生效了。这是删掉它的直接理由。
 *
 * 「`Steading` → **耕耘**」这条裁决本身仍然成立（2026-08-12b），只是落点在 **SEASONS.STEADING**
 * 这个 i18n 键上、不在月名表上。理由存档，改动时对齐三处：
 *   - 原译「庄园」是错的：它是季节名不是建筑，而「庄园」在本库已被 `Grange` / `Manor` 占用
 *     （英文写 Grange/Manor 的 120 条叶子中文 120 条全是「庄园」，如 Dradley Grange 德拉德利庄园），
 *     于是正文出现「庄园被称为工业季节」这种句子；
 *   - 英文侧 `History/Steading` 页原文 "The Steading is known as the Season of Industry … the period
 *     of the year when people are happiest being productive and working with their hands"，
 *     Gleaning 页又写 "the quiet duty found in the Steading"，讲的是脚踏实地的劳作季；
 *   - 与另外五个同为两字动名词的季节名（播种/绽放/拾取/凋零/寂止）同构词、同农事语域。
 *   同改点只有两处：`lang/cn.json` 的 `EMBER.CALENDAR.SEASONS.STEADING`，以及 compendium 里
 *   `History/Steading` 页名与正文（**不是**三处 —— 第三处那张月名表已随本轮删除）。
 *
 * 若上游哪天真的改用 `month.name` 上屏，再按上面的 grep 判据复验一次、把表加回来即可。
 * ------------------------------------------------------------------ */

/**
 * 天气与风力的档位名。
 *
 * 这些 label 是**数据**不是文案：写死在区域切片配置里（surface: ember.mjs:119622-119740、
 * pathways: :120076-120105 的 `weather` 对象），既不是 i18n 键、也进不了 babele。
 * 三处消费点全都拿它拼串或直接上屏，DOM 层都够不到或留不住：
 *   ① 日历条的天气图标 tooltip：`icon.dataset.tooltip = str?.label ?? cfg.label`（ember.mjs:24656）；
 *   ② 日历条的风向箭头 tooltip：`` `${windCfg.strengths[wind.strength]?.label} (${wind.speed} mph)` ``（:24673）；
 *      这两处由 `#refreshWeather()` 写，调用方是 `animate()`（:24584），不发任何渲染钩子；
 *   ③ 天气图例 EmberWeatherLegend（:35131 `weather[type] = {...cfg, …}`）与「创建天气」框的
 *      类型下拉（:22708-22712 `typeChoices[k] = cfg`）—— 两处直接读同一个 cfg.label。
 * 所以走 patchWeatherLabels() 在源头改数据，三处一起变中文、不会分裂。
 *
 * 值域是穷举的（两个切片合计 26 个唯一 label，脚本核过）。译名取法：
 *   - 合集既有写法优先：Gale 疾风（ember.character「Aura 2: Gale」→「奥拉 2：疾风 Aura 2: Gale」）、
 *     Breeze 微风（Chill Breeze→凛冽微风）、Pollen 花粉、Pollen Storm 花粉风暴（glossary_ec 已定）、
 *     Storm 风暴、Arcane 奥术、Dust 尘（尘埃形态 / 尘卷风）；
 *   - 同族三档必须能分出强弱，所以 Mist/Fog/Dense Fog 取 薄雾/雾/浓雾（`Misted` 在 glossary_ec
 *     里正是「薄雾」），Drizzle/Rain/Storm/Tempest 取 细雨/雨/风暴/狂风暴雨，
 *     Kindling/Wildfire/Inferno 取 起火/野火/烈焰，Calm/Breeze/Windy/Gale/Squall 取 无风/微风/有风/疾风/狂风。
 *   - `Calm` 这里是**风力 0 级**，与 MOODS 里那个音乐氛围的 `Calm`「平静」不是一回事；
 *     两张表作用域不同（本表只喂 weather 配置对象），不会互相污染。
 *
 * ⚠ **本表的 `Dust` / `Kindling` / `Tempest` 三条会被 `scan_cross_channel` B 段报
 *   `MJS_ORPHAN_CN`，2026-08-15 第十七轮逐条查过英文闸，三条全部保持，理由如下**
 *   （档位名是**枚举**，判据拿全库多数写法来比，比的却是同一串的其它义项，属于定义域问题）：
 *
 *     · `Dust`（判据：合集多数「尘土」10/52）——「尘土」那 10 叶的最强一叶是
 *       `scenes.The World of Ember.notes.Dust` 的 name「尘土 Dust」，但那是**世界地图上的
 *       一个地名**（同一批 notes 里全是 Crown 王冠 / Mordant 莫丹特 / Red Rhuin 红鲁因 这种
 *       聚落名），不是天气类型；其余是 `Concentrated Ore Dust` 浓缩矿尘 / `Choking Fog Dust`
 *       窒息迷雾粉尘 / `Slowing Dust` 滞缓之尘 / `Dust Mote` 尘埃微粒 / `Dust Devil` 尘卷风
 *       这类复合专名，以及散文里的「灰尘 / 尘埃」。天气义的**自由词** `Dust` 全库 0 叶。
 *       本族要能分强弱（扬尘 / 尘云 / 沙尘暴 / 哈布沙暴），「尘土」既非气象词也压不住档位差。
 *
 *     · `Kindling`（判据：合集多数「神圣」6/8）—— 英文闸 8 叶**全部**是同一件物品
 *       `Divine Kindling`「神圣引火物」；判据取的「神圣」只是那个复合名的前两字，
 *       是子串artifact不是译名。天气义 0 叶。起火 / 野火 / 烈焰 三档保持。
 *
 *     · `Tempest`（判据：合集多数「风暴」20/21，且带 name/uuid 证据，看着最像该改的一条）——
 *       实查那 20 叶：13 叶是玛伊斯的月亮尊号 `The Tempest Moon`「风暴之月」，
 *       5 叶是 `Geography.pages.Tempest` 那座**城市**（`the great city of Tempest`）的
 *       name「风暴 Tempest」及正文提及，只有 **2 叶**（孪生包各一份）真在天气域里。
 *       前两类都不是天气档位，且本表 `Storm` 已占「风暴」，改过去会让四档里有两档同名。
 *       而那 2 叶同域证据本身（`Players' Guide.pages.Weather`：“the weakest strength is a
 *       Drizzle while the greatest strength is a Tempest”）现译「最弱…毛毛雨…最强…风暴」，
 *       **是合集那边错**（毛毛雨↔本表细雨、风暴↔本表狂风暴雨，而且「最强」写成了中间档的名字）。
 *       已出 compendium 批次 `4-临时脚本/2026-08-15-round17/batches/r17-weather-tiers-*.json`（两包各一份），
 *       .mjs 侧不动。
 */
const WEATHER = {
  "Clear": "晴朗",
  "Rain": "雨", "Drizzle": "细雨", "Storm": "风暴", "Tempest": "狂风暴雨",
  "Fog": "雾", "Mist": "薄雾", "Dense Fog": "浓雾", "Arcane Fog": "奥术之雾",
  "Pollen": "花粉", "Pollen Spores": "花粉孢子", "Pollen Cloud": "花粉云", "Pollen Storm": "花粉风暴",
  "Dust": "扬尘", "Dust Clouds": "尘云", "Dust Storm": "沙尘暴", "Haboob": "哈布沙暴",
  "Wildfire": "野火", "Kindling": "起火", "Inferno": "烈焰",
  "Wind": "风", "Calm": "无风", "Breeze": "微风", "Windy": "有风", "Gale": "疾风", "Squall": "狂风"
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
  // 2026-08-15 主控裁决 R-B：**按英文词分裂** —— `Cosmological`/`Cosmology`＝宇宙、`Cosmos`＝寰宇。
  // 原值「寰宇同调」是照 lang 的 `TYPES.JournalEntryPage.ember.cosmos`＝余烬寰宇 取的，但那条英文是
  // `Cosmos` 不是 `Cosmological`，取错了源。锚点是合集里 `Players' Guide.pages.Cosmology` 的中文 name
  // 「宇宙观 Cosmology」，指向它的 `@UUID[…]{宇宙同调}` 标签也是「宇宙」—— 标签跟锚点走。
  // `Cosmos` 那一支（TYPES.…ember.cosmos＝余烬寰宇 / EMBER.CALENDAR.COSMOS＝寰宇地图）保持不动。
  "Cosmological Attunements": "宇宙同调",  // tab-attunement.hbs:4
  "Make Active": "设为激活",               // tab-attunement.hbs:38 的 aria-label
  // `Active` 这里是**状态标签**（哪一个同调当前处于激活状态），所以带「中」；lang 的
  // `EMBER.EVENTS.Active`＝激活 是事件列表的分类名，两者不同域。B 段会报这一条，已裁：不统一。
  "Active": "激活中"                       // ember.mjs:124757，拼进 tags 的那半截（另半截 Rank 已走 i18n）
};

/**
 * crucible HeroSheet「天赋」页里被上游**每轮 prepareData 重写**的 12 个物品名。
 *
 * ember.mjs:126016 `item.name = \`Soulbound (${rankLabel} Soulmark)\`` 与另外 11 处
 * `item.name = \`Xxx Attunement (Rank ${ATTUNEMENT_RANK_NUMERALS[rank] ?? "I"})\``
 * （:126104/126165/126190/126234/126259/126284/126309/126337/126374/126399/126424）都在
 * prepareAbilities 里，babele 在合集里译好的 name 每次数据准备都会被覆盖掉，
 * 合集侧无论怎么改都留不住 —— 只能在 DOM 上兜。
 *
 * 宿主是 crucible 的 HeroSheet（不是 Ember 的窗口），走 INJECTED_SUBTREES 按页签选择器放行；
 * 表是**穷举生成**的（11 同调 × 5 阶 + 3 档魂印 + 11 个不带阶位的动作名），不用宽正则，
 * 免得「… Rank N」这种形状把别的系统的物品名也吃掉。
 * 阶位数字保留上游的罗马数字；「阶位」取 lang 的 EMBER.ATTUNEMENT.Rank。
 * 三档魂印取 glossary_ec 的定译（次等/高等/死亡魂印），与 DIALOG_UI 里那三条确认框文案一致。
 */
const HERO_ITEM_NAMES = (() => {
  const t = {};
  for (const [en, cn] of Object.entries(ATTUNEMENT_ITEM_NAMES)) {
    t[`${en} Attunement`] = cn;                                   // ember.mjs:124858 的动作名
    // ATTUNEMENT_RANK_NUMERALS（ember.mjs:125808）= ["", "I", "II", "III", "IV", "V"]
    for (const numeral of ["I", "II", "III", "IV", "V"]) {
      t[`${en} Attunement (Rank ${numeral})`] = `${cn}（阶位 ${numeral}）`;
    }
  }
  for (const [en, cn] of Object.entries({Lesser: "次等魂印", Greater: "高等魂印", Deathly: "死亡魂印"})) {
    t[`Soulbound (${en} Soulmark)`] = `魂缚 Soulbound（${cn}）`;   // ember.mjs:126016
  }
  return t;
})();

/**
 * 同调奖励聊天卡（ember.mjs:2957-2985）。
 *
 * 表头是 `${config.label} ${_loc("EMBER.ATTUNEMENT.Attunement")}`（:2971）—— 前半截是
 * dnd5e-async.mjs:406-416 的 11 个裸英文短名、后半截已经是中文，拼出来是「Abyss 同调」这种
 * 半英半中的整串，PREFIXED 的 `Attunement: ` 前缀吃不到、ATTUNEMENTS 短名表也对不上形状。
 * 这些复合键在 ready 时由 buildChatKeys() 按当前 lang 的实际译文拼出来（写死 "Abyss 同调"
 * 会在别人改了 EMBER.ATTUNEMENT.Attunement 之后失效）。
 * 聊天面只对 `msg.flags.ember` 的消息开放，绝不无条件翻整个聊天栏。
 */
const CHAT_UI = {
  ...ATTUNEMENTS,
  ...HERO_ITEM_NAMES
};

/**
 * 只在**Ember 自己的窗口**（主闸已放行的那些）里生效的作用域表。
 *
 * 这些词单独看都太通用（Overview / Class / Type / Anchor / Points…），进全局 EXACT 会顺手
 * 改掉别的模块的窗口，甚至被 DialogV2 认框失败那一支拿去改别人的标题；但在 Ember 自己的
 * 窗口里含义是确定的。行末是上游出处。
 */
const EMBER_WINDOW_UI = {
  // 日志页的次级内容页签（EmberPageSheet.secondaryContentTabs，ember.mjs:35987/35989/36094 等，
  // 由 :35711 splice 进 TABS.sheet.tabs 当页签标题）。同组的另外 5 条已在 EXACT 里。
  "Topic Overview": "主题概览",
  "Overview": "概览",
  "Secret Information": "机密信息",
  // 事件 / 任务日志页的模板裸串
  "Event Flowchart": "事件流程图",                                        // journal/pages/flowchart-view.hbs:5
  "Quest is not properly configured for flowchart view.":
    "该任务未正确配置流程图视图。",                                          // flowchart-view.hbs:39
  "Last Modified": "最后修改",                                            // journal/partials/tab-development.hbs:5
  "Add Outcome": "添加结果",                                              // tab-event-outcomes.hbs:5
  "Outcome ID": "结果 ID",                                               // tab-event-outcomes.hbs:19
  "Outcome Label": "结果标签",                                            // tab-event-outcomes.hbs:26
  "Outcome Summary": "结果摘要",                                          // tab-event-outcomes.hbs:33
  "Allow Retry?": "允许重试？",                                           // tab-event-outcomes.hbs:38
  "There are no hooks defined for this event.": "此事件未定义任何钩子。",     // tab-event-hooks.hbs:5
  // 与 crucible 的 `HOOKS.ToggleSource`（en＝"Toggle Source"）是**同一个按钮**：Ember 的
  // templates/journal/partials/hook.hbs 是 crucible templates/sheets/partials/hook.hbs 的照抄，
  // 连 `data-action="hookToggleSource"`、`fa-code` 图标、下面那个 `<code-mirror language="javascript">`
  // 都一样，只是 tooltip 写成了英文字面量而没走键。2026-08-15 B 段报的这条漂移**方向在 lang 那边**：
  // 按钮展开的是钩子的 JavaScript 源码，crucible-cn 原译「切换来源」把 source 当成了「来源」义项。
  // 已出批次改 crucible lang（batches-seal/r16b-lang-crucible-toggle-source.json），本表保持「切换源码」。
  "Toggle Source": "切换源码",                                            // journal/partials/hook.hbs:5（＝crucible HOOKS.ToggleSource）
  "Event Probabilities": "事件概率",                                      // applications/hex-hud.hbs:9
  // 创角向导
  // `Ancestry` 2026-08-15 从全局 EXACT 挪进来（理由见 EXACT 的注释）：它在 Crucible 世界里
  // 接不到自己人，放在全局表只会去改别人窗口里的裸 "Ancestry"；宿主 EmberHeroCreationSheet
  // 过得了 `/^Ember/` 主闸，放作用域表里 dnd5e 世界照旧生效、Crucible 世界不再越界。
  "Ancestry": "血统",                                                    // ember.mjs:121864（dnd5e 分支的步骤名）
  "Class": "职业",                                                       // ember.mjs:121870 的步骤名（dnd5e 分支）
  "Aster Progression": "阿斯特进阶",                                       // crucible-async.mjs:210
  "Soulbound Progression": "魂缚进阶",                                     // crucible-async.mjs:233
  "Character Name": "角色名称",                                           // creation/header.hbs:24 的 placeholder
  // 创角向导的属性分配步骤（**dnd5e 分支**的 creation/class.hbs 与 creation/path.hbs；
  // Crucible 分支那份 crucible-path.hbs 的同类串已在 EXACT 里）
  "Base Ability Scores": "基础属性值",                                     // creation/class.hbs:42
  "Points Remaining": "剩余点数",                                         // creation/class.hbs:43
  "Ability scores will be further boosted when choosing a Path.":
    "选择道途时属性值还会进一步提升。",                                      // creation/class.hbs:60
  "Ability Score Increases": "属性值提升",                                 // creation/path.hbs:46
  "Increases Remaining": "剩余提升次数",                                   // creation/path.hbs:47
  "Further increase ability scores with no more than two increases assigned to one ability.":
    "继续提升属性值，单项属性最多分配两次提升。",                             // creation/path.hbs:63
  // 远景配置 / 日历 / 指示物制作器（都要靠属性白名单里新加的 placeholder / alt 才够得到）
  "Unique Identifier": "唯一标识符",                                       // vista-config-scene.hbs:9
  "Displayed Label": "显示名称",                                          // vista-config-scene.hbs:10
  "Custom Level Name": "自定义层名",                                       // vista-config-scene.hbs:15
  "Wind Direction": "风向",                                              // calendar-visual.hbs:7 的 alt
  // 指示物制作器的动画开关。模板里是 "Play Animation"（body.hbs:28），但 #refresh() 会在
  // 渲染钩子**之后**把 data-tooltip 写回英文（ember.mjs:50999/51002），
  // 所以除了补键还挂了一个 MutationObserver，见 patchRenderedApplications。
  "Play Animation": "播放动画",
  "Stop Animation": "停止动画",
  // 指示物制作器 EmberDynamicTokenConfig（ember.mjs:50481，classes 含 "ember"，主闸放行）。
  // 2026-08-15 第十六轮补：原先整个窗口只登记了上面这两条动画开关，其余按钮的 data-tooltip
  // 与两根分栏标题全是裸英文。判据是把 templates/ 下的字面量与本文件的键集对差跑出来的。
  "Ember Token Maker": "余烬指示物制作器",                                   // ember.mjs:50508 窗口标题
  "Randomization Rules": "随机化规则",                                     // ember.mjs:50513 窗口头部控件
  "Build": "体型",                                                       // ember.mjs:50742（layers.hbs 的 build.title）
  "Stance": "姿态",                                                      // ember.mjs:50759（同上 stance.title）
  "Layers": "层", "Colors": "颜色",                                       // layers.hbs:2 / colors.hbs:2 分栏标题
  "Reset All": "全部重置",                                                // body.hbs:3
  "Randomize": "随机生成",                                                // body.hbs:14
  "Save Dynamic Token": "保存动态指示物",                                    // body.hbs:19
  "Rotate Left": "向左旋转", "Rotate Right": "向右旋转",                    // body.hbs:20 / 27
  "Shift Left": "向左平移", "Shift Right": "向右平移",                      // body.hbs:21 / 26
  "Shift Up": "向上平移", "Shift Down": "向下平移",                         // body.hbs:23 / 24
  "Previous": "上一个", "Next": "下一个",                                  // layers.hbs:6/12/17/23
  "Previous Option": "上一个选项", "Next Option": "下一个选项",              // layers.hbs:29 / 40
  "Randomize Layer": "随机该层", "Clear Layer": "清空该层",                 // layers.hbs:31 / 38
  // 指示物随机化配置 EmberRandomTokenConfig（ember.mjs:51469，类名以 Ember 开头，主闸放行）
  "Anatomy": "身体", "Equipment": "装备",                                  // token-randomization.hbs:11/28/55/69
  "Copy options from the opposite side": "从对侧复制选项",                   // token-randomization.hbs:17（提示语见 NOTIFICATIONS「对侧…」）
  "Import current selection from Token Maker": "从指示物制作器导入当前选择",     // token-randomization.hbs:19
  "Remove Layer": "移除该层", "Remove Color": "移除该颜色",                  // token-randomization.hbs:21 / 62
  "Choose templates to configure part restrictions.": "请先选择模板，再配置部件限制。",   // token-randomization.hbs:45
  "Choose templates to configure color restrictions.": "请先选择模板，再配置颜色限制。", // token-randomization.hbs:83
  // 远景配置 EmberVistaConfiguration 的另外三张模板（窗口标题早已在 EXACT 里）
  "Place Assets": "放置素材",                                             // vista-config-assets.hbs:15
  "Replace Placement": "替换摆放", "Delete Placement": "删除摆放",          // vista-config-assets.hbs:22 / 23
  "Create Custom Vista": "创建自定义远景",                                  // vista-config-scene.hbs:5
  // 下面这条模板里带换行 + 缩进，键写成折叠空白后的单行形态，靠 translateNode 的回退命中
  "Create a new custom vista composition, starting from your currently viewed one, that will be used as the starting point from which you can make further changes.":
    "以你当前查看的远景构图为起点，创建一个新的自定义远景构图，作为后续修改的基础。",       // vista-config-scene.hbs:6
  "Level Name": "层名",                                                   // vista-config-scene.hbs:13（与「自定义层名」同族）
  "Create": "创建",                                                       // vista-config-scene.hbs:21
  "Token Scaling": "指示物缩放",                                            // vista-config-scene.hbs:27
  // ⚠ `Sprites` 是远景图层里的**贴图精灵**（与 SFX / Special Effects 并列的一档美术资源），
  //   B 段 `MJS_ORPHAN_CN` 拿合集英文闸下的「精灵」来比是把它当成了 fey/sprite 那个生物义；
  //   带「图」正是为了跟生物撇清。已裁：保持「精灵图」。
  "Special Effects": "特殊效果", "SFX": "特效", "Sprites": "精灵图",         // vista-config-scene.hbs:34/38/42
  "Coordinates": "坐标", "Scale": "缩放", "Skew": "斜切",                   // vista-config-placement.hbs:5/21/36
  "Precolorization": "预着色",                                            // vista-config-placement.hbs:66
  // 角色旗标配置：上游模板写的是 {{localize "Anchor"}}，而 modules/ember/lang/en.json 里
  // 根本没有这个键（自己 grep 计数 0），core 也没有，localize 原样返回英文。
  // 不塞 lang/cn.json：那是无点号顶层键，会打破发版前 flatten_lang.py 的三数相等，
  // 且 Anchor 太通用，顶层键是全局生效的。
  "Anchor": "锚点"                                                       // applications/actor-flags.hbs:29
};

/** Ember 自己弹的框：作用域表要把「对话框专用」和「Ember 窗口通用」两张合起来用 */
const EMBER_DIALOG_UI = {...DIALOG_UI, ...EMBER_WINDOW_UI};

/**
 * 播放列表侧栏里 Ember 注入的音景面板（`<form id="ember-mood">`，ember.mjs:15874-15898）。
 * 宿主是 core 的 PlaylistDirectory，主闸两个判据都不成立，靠 INJECTED_SUBTREES 放行子树。
 * 本表 **50 键 = SOUNDSCAPE_GROUPS(42) ＋ ARRANGEMENTS(5) ＋ 下面 3 条面板自有文案**，三部分零重叠
 * （`Ember Environment` 既是组名又是 :15892 的 `<header>` 文案，只在组名表里定义一次，
 * 所以这里不再重写它 —— 别看着 header 没登记就往回加，那会造出第二个定义处）。
 * 译名取 glossary_ec 的定稿（Ember Music 余烬乐曲 / Ember Environment 余烬环境）。
 *
 * 组名的 `<optgroup label>` 是**属性**不是文本节点，走得通全靠 translateNode 的属性白名单里
 * 有 `label` 那一条（见该函数）—— 把 `label` 从白名单里删掉，这 42 条会整体失效。
 *
 * ⚠ 覆盖率是 2026-08-15 **实测值，不是估计**，并在第二十一轮**换一套方法复算过**
 * （原探针 `4-临时脚本/2026-08-15-round20/probe_soundscapes.mjs` 是手写大括号配对 + 按固定缩进
 * 正则抠 `label`；复算不用正则碰内容：从冻结注册表 `var soundscapes=Object.freeze({…})`
 * （ember.mjs:15260）取 44 个变量名，按打包器「顶层声明以行首 `}` 收尾」的排版切片，交给
 * **node:vm** 让真正的 JS 解析器把对象建出来，再要求每个对象自己的 `id` 等于注册表里的键 ——
 * 自报 `resolved=44 / BAD=0`，任一条不成立就退出而不是给半份结果。两套方法数字逐条相同。
 * 第二十一轮的复算脚本落在 `4-临时脚本/2026-08-15-round21/probe_groups.mjs`，两法同时跑并互校）。
 * 上屏的两档东西**要分开数**（`#updateSoundscapeForm()`，ember.mjs:15916-15922：
 * `{value, label: 编排名, group: 音景 label}` 交给 createSelectInput）：
 *   ① **optgroup 组名** = 音景对象的 `label`。上游 44 个音景里 42 个 `type` 是 music(41)/environment(1)
 *      才进这两个下拉（落选的两个是 `Ember Events` / `Ember Weather`），42 个 label 全不重样
 *      → **42 条**上屏组名。
 *   ② **option 编排名** = `s.arrangements[*].label`。这 42 个音景合计 **267 条**，去重 **212 条**。
 *
 * ⚠ **「上屏留不留英」必须按这棵子树真正注册的那张表判** —— 不是按 `ARRANGEMENTS`，
 * 更不是按不带 extra 的全局 `translateText`。两点都是实测：
 *   · INJECTED_SUBTREES 里 `form#ember-mood` 注册的是 **MOOD_PANEL**，不是 ARRANGEMENTS；
 *   · 拿真 `translateText` 复核时，**不传作用域表则 42 个组名一条都不翻**（212 个编排名同样一条不翻）——
 *     MOOD_PANEL 是作用域表，键**不在全局 `EXACT` 里**。拿全局通道去量这块覆盖率只会量出假的「0 覆盖」。
 *
 * 按 MOOD_PANEL 判的当前账（第二十一轮实测，探针 `probe_moodpanel.mjs`，
 * 它是**在 vm 里跑本文件真源码**取到的 MOOD_PANEL，再喂给本文件导出的真 `translateText`）：
 *   · 组名 42 条 → 覆盖 **42**、**恒英文 0**（本轮把 37 条补齐了）。
 *   · 编排名去重 212 条 → 覆盖 **11**、**恒英文 201**（上一轮是覆盖 8 / 恒英文 204；
 *     多出来的 3 条是 `Marlstone Gala` / `Ordain` / `The Pit Trap`，它们与组名同串，
 *     是补组名的**必然副作用**，不是偷偷补了编排名，见 SOUNDSCAPE_GROUPS 表内注）。
 *   · 三条面板自有文案（`Ember Music` / `Rearrange Music` / `Ember Default`）与 `Reset`
 *     两边都对不上，它们不是音景名。
 * 排期时**别把两档并成一档**：组名已清零，剩下的 201 条编排名是另一件事，
 * 多为「Bandit Fight Chorus / Celestial Combat Section 3」这类内部段落编号，
 * 主控第二十一轮裁的是**暂不补**。补不补不在本文件裁。
 */
const MOOD_PANEL = {
  ...SOUNDSCAPE_GROUPS,
  ...ARRANGEMENTS,
  "Ember Music": "余烬乐曲",          // 15886
  "Rearrange Music": "重新编排音乐",   // 15889 的 data-tooltip
  "Ember Default": "余烬默认"         // 15929 / 15934 两个 select 的 blank 选项
};

/**
 * Ember 塞进 core NoteConfig 的「Ember Type」表单组
 * （ember.mjs:64962-64975，选项表 EmberRegionNote.TYPES 在 :64822-64831，8 条裸英文无 i18n 键）。
 * City / Dragon / God 都是通用词，只能走作用域表；Leviathan 与 KNOWLEDGE 的「利维坦」对齐。
 */
const NOTE_TYPES = {
  "Ember Type": "余烬类型",
  "City": "城市", "Culture": "文化", "Dragon": "巨龙", "Geography": "地理",
  "Metropolis": "大都会", "Ruin": "废墟", "Leviathan": "利维坦", "God": "神祇"
};

/**
 * 设置面板 / 按键绑定面板里 Ember 命名空间那一段（ember.mjs:129274-129318 四条 config:true 的
 * 设置，:128565/128572 两条按键绑定）。core 的 SettingsConfig 与 ControlsConfig 都是
 * CategoryBrowser，按命名空间分段渲染成 `<section data-category="ember">`
 * （applications/settings/config.mjs:145-157 与 sidebar/apps/controls-config.mjs 的
 * #categorizeEntry，category.id 就是模块 id），所以选择器按 data-category 限定即可，
 * 既不用认类名（v14 里按键面板的类名是 ControlsConfig 而不是 KeybindingsConfig），
 * 也不会翻到别的模块或 core 自己的设置。
 */
const SETTINGS_UI = {
  // `Gazetteer`＝地名志：glossary_ec 定稿（`Gazetteer`＝地名志 Gazetteer、五本 X Gazetteer 同形），
  // compendium 英文闸 98 叶全部作「地名志」，0 叶作「地名录」。原译「地名录」是本文件独有的孤例。
  "Gazetteer Location Journal Entries": "地名志地点日志条目",
  "Additional Journal Entries which provide custom gazetteer Location pages that should be added to the Ember environment.":
    "为余烬环境额外提供自定义地名志「地点」页面的日志条目。",
  "Standalone Event Journal Entries": "独立事件日志条目",
  "Additional Journal Entries which contain Standalone Event pages which should be added to the Ember event engine.":
    "为余烬事件引擎额外提供「独立事件」页面的日志条目。",
  "Clock Time Format": "时钟时间格式",
  "The clock format used to display the in-world time of day.": "用于显示世界内当日时间的时钟格式。",
  "Custom Cursors": "自定义光标",
  "Use custom Ember stylized mouse cursors instead of default browser cursors?":
    "使用余烬风格的自定义鼠标光标，替代浏览器默认光标？",
  "Flip Vista Placement": "翻转远景摆放",
  "When placing an asset in the Vista Configuration screen, flip it horizontally":
    "在远景配置界面摆放素材时，将其水平翻转",
  "Lock Vista Placement Elevation": "锁定远景摆放高度",
  "When placing an asset in the Vista Configuration screen, lock its elevation so it can be moved vertically.":
    "在远景配置界面摆放素材时锁定其高度，使其只能垂直移动。"
};

/**
 * 场景控制栏上 Ember 加的那一个工具按钮（ember.mjs:113453-113456 `title: "Show Tracks"`，
 * 裸英文、无 i18n 键）。core 的模板把它渲染成 `aria-label="{{localize tool.title}}"`
 * （templates/ui/scene-controls-tools.hbs:5），localize 查不到键就原样返回。
 * 按 `data-tool="tracks"` 精确定位，只带这一个键，不碰控制栏上别的按钮。
 */
const SCENE_CONTROL_UI = {
  "Show Tracks": "显示轨道"
};

/**
 * 「Ember 注入到别人窗口里的子树」→「只在这棵子树里生效的表」。
 * 渲染钩子的宿主闸拦下非 Ember 应用之后，会拿这张表逐条 querySelectorAll 补翻。
 * 第三个元素是可选项：`observe` 表示这棵子树在渲染钩子之后还会被上游改写，
 * 需要挂 MutationObserver 盯着（见 observeSubtree）。
 */
const INJECTED_SUBTREES = [
  ["section.tab.attunement", ATTUNEMENT_TAB],
  // crucible HeroSheet 的天赋页：12 个被 prepareAbilities 重写的物品名（见 HERO_ITEM_NAMES）
  ["section.tab.talents", HERO_ITEM_NAMES],
  // 播放列表侧栏的音景面板。renderPlaylistDirectory 早于 renderApplicationV2 派发
  // （core application.mjs:1724-1730 #callHooks 沿 inheritanceChain 从子类往上走），
  // 所以我们跑的时候 form 已经在 DOM 里了。选项由 #updateSoundscapeForm() 在 change 之后
  // 异步整段重写 innerHTML、且不发任何渲染钩子，故要 observe。
  ["form#ember-mood", MOOD_PANEL, {observe: true}],
  // core NoteConfig 里 Ember 注入的「Ember Type」表单组（renderNoteConfig 同样早于 renderApplicationV2）
  ['.form-group:has([name="flags.ember.type"])', NOTE_TYPES],
  // 设置面板 / 按键绑定面板里 ember 命名空间那一段
  ['section[data-category="ember"]', SETTINGS_UI],
  // 场景控制栏上 Ember 那一个工具按钮
  ['button.tool[data-tool="tracks"]', SCENE_CONTROL_UI]
];

/**
 * `ui.notifications` 的提示语。
 *
 * 这条通道**没有宿主可挂**：core 的 Notifications（client/applications/ui/notifications.mjs:30）
 * 不继承 Application，没有任何 render 钩子，DOM 遍历那一套完全够不到；
 * 而 ember.mjs 里 65 处都是字面量、没有 i18n 键。只能包住 `ui.notifications.notify`
 * （info / warn / error 三个方法最后都调它，:152/:163/:174）。
 *
 * ⚠ 包的是**全局**方法，别的模块的提示也会流过来，所以这里**不复用 translateText** ——
 * 那会把全局 EXACT（含 Path / Culture / Complete 这类通用词）套到别人的提示上。
 * 只查本表与 NOTIFICATION_PATTERNS，查不到原样返回，见 translateNotification。
 * 行末是 modules/ember/scripts/ember.mjs 的行号。
 */
const NOTIFICATIONS = {
  "No other events are available at the moment.": "目前没有其他可用的事件。",                    // 24767
  "This event does not configure a Scene to preload.": "该事件没有配置需要预加载的场景。",         // 24782
  "Copied vista configuration to clipboard as JSON data": "已将远景配置以 JSON 数据复制到剪贴板", // 34537
  "You must provide a name to create a new custom composition.": "创建新的自定义构图需要先填写名称。", // 34586
  "Invalid Vista Configuration JSON provided.": "提供的远景配置 JSON 无效。",                    // 34768
  "No current page detected": "未检测到当前页面",                                              // 35339
  "Only a Gamemaster user may initiate a group rest from the Party Sheet.":
    "只有游戏主持人才能从队伍卡发起集体休息。",                                                  // 37929 / 37984
  "You may not initiate a rest while an Event is in-progress.": "事件进行期间不能发起休息。",      // 37932 / 37987
  "Completed a short rest without incident!": "短休顺利完成！",                                 // 38007 / 120588
  "Completed a long rest without incident!": "长休顺利完成！",                                  // 120636
  "The opposite side has no configured options to copy from.": "对侧没有可供复制的已配置选项。",    // 51913
  "You may not move the Party caravan while the game is paused.": "游戏暂停期间不能移动队伍商队。", // 60967
  "You may not move the Party caravan during an active discovery.": "发现进行期间不能移动队伍商队。", // 60971
  "You may not move the Party caravan during an ongoing event.": "事件进行期间不能移动队伍商队。",  // 60975
  "You may not take this action unless there is a Gamemaster user present.":
    "没有游戏主持人在场时不能执行此操作。",                                                      // 62745
  "Only the Gamemaster can install the junction wheel.": "只有游戏主持人才能安装枢纽轮盘。",       // 112213
  "Resetting Ember game state data. Please be patient for several seconds.":
    "正在重置余烬的战役状态数据，请耐心等待几秒。",                                               // 129502
  "Ember game state successfully reset!": "余烬战役状态已重置！",                                // 129519
  "You may only use this macro with a Corpuleth token controlled.":
    "只有在控制着尸团怪 Corpuleth 指示物时才能使用该宏。",                                          // 73303
  "This Corpuleth token does not have an Ember Dynamic Token configured!":
    "该尸团怪 Corpuleth 指示物没有配置余烬动态指示物！",                                             // 73308
  "This macro may only be used in the Bronze Rask Theater.":
    "该宏只能在青铜拉斯克剧院 Bronze Rask Theater 中使用。",                                     // 73349
  "Changed Wandren and Ruffian adversaries in the Bronze Rask Theater to hostile!":
    "已将青铜拉斯克剧院 Bronze Rask Theater 里的万德伦与恶棍敌人切换为敌对！",                     // 73366
  // 61461 是 `"…which will" + " automatically…"` 两段字符串相加，运行时是一整行；
  // 源码里的 \" 到了运行时就是普通的半角引号。任务名取合集定稿「有遮蔽的营地 Sheltered Campsite」。
  "When you are ready to begin the Ember game, activate this Scene which will automatically begin the first quest event, \"The Sheltered Campsite\".":
    "准备好开始余烬战役时，激活本场景即可自动开启第一个任务事件「有遮蔽的营地 The Sheltered Campsite」。", // 61461
  // 魂缚进阶宏的三条前置检查（126622 / 126626 同样是两段相加）
  "The Soulbound Progression macro can only be used by a Gamemaster user.":
    "魂缚进阶宏只能由游戏主持人使用。",                                                          // 126622
  "The Soulbound Progression macro must be used while viewing a specific character sheet for a Hero or Adversary.":
    "使用魂缚进阶宏时必须正在查看某个英雄或敌人的角色卡。",                                        // 126626
  "Source compendium data for the Soulbound talent was not found!":
    "找不到魂缚 Soulbound 天赋的合集源数据！"                                                    // 126635
};

/** 带插值的提示语。同样只在 translateNotification 里查，不进全局 PATTERNS。 */
const NOTIFICATION_PATTERNS = [
  { re: /^Attunement activation is not yet implemented for system "(.+)"\.$/,
    cn: (m) => `系统「${m[1]}」尚未实现同调激活。` },                                            // 3149
  { re: /^The (.+) Vista is not yet configured to support the placement of Tokens\.$/,
    cn: (m) => `远景「${m[1]}」尚未配置为支持放置指示物。` },                                       // 33205
  { re: /^Deleted saved Vista composition "(.+)"\.$/, cn: (m) => `已删除保存的远景构图「${m[1]}」。` }, // 34509
  { re: /^Updated composition for Level "(.+)"\.$/, cn: (m) => `已更新层「${m[1]}」的构图。` },     // 34557
  { re: /^Saved composition to Scene with identifier "(.+)"\.$/,
    cn: (m) => `已将构图保存到场景，标识符为「${m[1]}」。` },                                      // 34570
  { re: /^Imported composition to Level "(.+)"\.$/, cn: (m) => `已将构图导入到层「${m[1]}」。` },   // 34779
  { re: /^Awarded attunement progression points to Actors: (.+)$/,
    cn: (m) => `已向以下角色授予同调进阶点数：${m[1]}` },                                         // 36885
  { re: /^Completed resting for (\d+) hours without incident!$/,
    cn: (m) => `顺利完成了 ${m[1]} 小时的休息！` },                                              // 37966
  { re: /^Your rest was interrupted after (\d+) hours by the (.+) event!$/,
    cn: (m) => `休息在 ${m[1]} 小时后被「${m[2]}」事件打断！` },                                  // 37970
  { re: /^Your rest was interrupted by the (.+) event!$/, cn: (m) => `休息被「${m[1]}」事件打断！` }, // 38009
  { re: /^Your short rest has been interrupted after (\d+) minutes!$/,
    cn: (m) => `短休在 ${m[1]} 分钟后被打断！` },                                                // 120582
  { re: /^Your long rest has been interrupted after (\d+) hours!$/,
    cn: (m) => `长休在 ${m[1]} 小时后被打断！` },                                                // 120627
  { re: /^Saved Ember Dynamic Token configuration for (.+)$/,
    cn: (m) => `已保存 ${m[1]} 的余烬动态指示物配置` },                                            // 51386
  { re: /^Saved Ember Dynamic Token randomization parameters to Actor (.+)$/,
    cn: (m) => `已将余烬动态指示物的随机化参数保存到角色 ${m[1]}` },                                 // 51839
  { re: /^Applied fog exploration from the (.+) vantage point!$/,
    cn: (m) => `已应用来自制高点「${m[1]}」的迷雾探索！` },                                       // 61882
  { re: /^Discovered vantage point, (.+)!$/, cn: (m) => `发现了制高点：${m[1]}！` },              // 67418
  { re: /^Added the Soulbound talent to (.+) at rank 1\.$/,
    cn: (m) => `已为 ${m[1]} 添加阶位 1 的魂缚 Soulbound 天赋。` },                              // 126644
  { re: /^Upgraded Soulbound talent on (.+) to rank (\d+)\.$/,
    cn: (m) => `已将 ${m[1]} 的魂缚 Soulbound 天赋升至阶位 ${m[2]}。` },                         // 126665
  // 2026-08-14 第十四轮补：把 ember.mjs 里剩下的带插值提示语一次收齐。
  // 判据是脚本枚举 `ui.notifications.(info|warn|error|notify)(` 的 76 处调用，
  // 扣掉 9 处非字面量（直接丢 Error / 变量）与 5 处 i18n 键（EMBER.* 那几条由 lang/cn.json 负责，
  // core notifications.mjs:121 的 `message = _loc(message, format)` 是无条件执行的，不看 localize 选项）。
  // 36922 与 126652 的模板串里带换行 + 缩进，靠 translateNotification 的空白折叠回退命中。
  { re: /^Event (.+) is not a recognized gameplay event\. This likely indicates some issue with the Ember module installation\.$/,
    cn: (m) => `事件 ${m[1]} 不是可识别的游戏事件。这多半说明余烬模块的安装有问题。` },              // 36922
  { re: /^Ember \| "(.+)" is not a known Token Maker part in any template layer\.$/,
    cn: (m) => `Ember | 「${m[1]}」不是任何模板层里已知的指示物制作器部件。` },                       // 49480
  // kind 只可能是 "layer" / "color"（ember.mjs:51903 / 51938 两个调用点写死）
  { re: /^This (layer|color) is not available in the Token Maker's current template preview\.$/,
    cn: (m) => `该${m[1] === "layer" ? "层" : "颜色"}在指示物制作器当前的模板预览中不可用。` },        // 51760
  { re: /^"(.+)" is already registered for this (layer|color)\.$/,
    cn: (m) => `「${m[1]}」已在该${m[2] === "layer" ? "层" : "颜色"}上注册过了。` },                // 51766
  { re: /^You cannot create multiple tokens for the "(.+)" group actor\.$/,
    cn: (m) => `不能为群组角色「${m[1]}」创建多个指示物。` },                                        // 60902
  { re: /^Adjacent hex (.+) is not directly reachable from current hex (.+)\.$/,
    cn: (m) => `相邻六边格 ${m[1]} 无法从当前六边格 ${m[2]} 直达。` },                             // 61049
  // ⚠ `Region Map`＝地区地图（既定裁决），原译「区域地图」是 Area Map 的译名
  { re: /^You are not allowed to delete the (.+) Token from the Region Map\.$/,
    cn: (m) => `你无权从地区地图上删除指示物 ${m[1]}。` },                                          // 61232
  { re: /^The Scene "(.+)" does not have compositions defined\.$/,
    cn: (m) => `场景「${m[1]}」没有定义任何构图。` },                                             // 63556
  { re: /^User "(.+)" wants to modify interactable "(.+)" in Scene (.+)\. You must be present in this Scene to acknowledge this operation\.$/,
    cn: (m) => `用户「${m[1]}」想要修改场景 ${m[3]} 中的可交互物「${m[2]}」。你必须身处该场景才能确认此操作。` }, // 63865
  { re: /^Ember \| Transit destination scene "(.+)" was not found\.$/,
    cn: (m) => `Ember | 找不到转运目的地场景「${m[1]}」。` },                                     // 96361
  { re: /^Mirror (.+) does not exist!$/, cn: (m) => `镜子 ${m[1]} 不存在！` },                    // 98432
  { re: /^Attunement feat for (.+) rank (\d+) could not be resolved\.$/,
    cn: (m) => `无法解析 ${m[1]} 阶位 ${m[2]} 的同调专长。` },                                    // 121323（dnd5e 分支）
  { re: /^(.+) cannot progress their Soulbound rank further as they already bear a Deathly Soulmark\.$/,
    cn: (m) => `${m[1]} 已经带有死亡魂印，魂缚阶位无法再提升。` }                                  // 126652
];

/**
 * 画布上的滚动文字（PIXI 对象，不在任何 DOM 子树里，translateNode 天然够不到）。
 * `canvas.interface.createScrollingText(origin, content, style)` 的第二参数：
 *   ember.mjs:20807 `Discovery!\n${discovery.label}`（发现新地点）
 *   ember.mjs:2566 陷阱行为的 `message` 字段，initial 就是 "Trap Triggered!"（:2629 调用）
 * 只翻整行相等的那两条固定串，动态的地点名原样保留。
 */
const SCROLLING_TEXT = {
  "Discovery!": "有所发现！",
  "Trap Triggered!": "陷阱触发！"
};

/**
 * Ember 往 **core 的 CONFIG** 里塞的两个裸英文 label。
 * 它们出现在环境光动画下拉（AmbientLightConfig）与场景天气下拉（SceneConfig）里，
 * 那两个窗口都是 core 自己的、主闸放行不了，而这两个值又不是 i18n 键 —— 改数据是唯一通道。
 * 第三处 `registerSheet(..., {label: "Ember Adventure Importer"})`（ember.mjs:129353）
 * 存在 DocumentSheetConfig 的私有静态表里，改数据够不到，留观。
 */
const CORE_CONFIG_LABELS = [
  ["Canvas.lightAnimations.emberFlame", "Ember Small Torch", "余烬小火把"],   // ember.mjs:52098
  ["weatherEffects.ember", "Ember", "余烬"]                                 // ember.mjs:129004
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
  // 2026-08-14 第十四轮补三个：
  //   placeholder —— creation/header.hbs:24「Character Name」、vista-config-scene.hbs:9/10/15 三条；
  //   alt         —— calendar-visual.hbs:7「Wind Direction」，另外 tab-attunement.hbs:11 的
  //                  alt="{{attunement.label}}" 随 ATTUNEMENT_TAB 一起被收掉；
  //   label       —— `<optgroup label="…">`（语言下拉的分类组名、音景面板的音景组名）。
  //                  这三个属性只出现在 input/img/optgroup 这类元素上，不会误伤正文。
  for (const attr of ["data-tooltip", "data-tooltip-text", "data-tooltip-html", "title", "aria-label",
                      "placeholder", "alt", "label"]) {
    const v = node.getAttribute?.(attr);
    if (v) {
      const t = translateText(v, extra);
      if (t !== v) node.setAttribute(attr, t);
    }
  }
  for (const child of Array.from(node.childNodes)) translateNode(child, extra);
}

/**
 * 翻一条 `ui.notifications` 提示。
 *
 * 刻意**不走 translateText** —— 那道口子是全局的（EXACT 里有 Path / Culture / Complete /
 * Exit 这类通用词），而我们包的是所有模块共用的 `ui.notifications.notify`，
 * 拿全局表去套别人的提示就是越界。只查 Ember 自己那两张提示语表。
 * @param {string} text
 * @returns {string}
 */
function translateNotification(text) {
  if (typeof text !== "string") return text;
  const raw = text.trim();
  if (!raw) return text;
  const lookup = (s) => {
    if (s in NOTIFICATIONS) return NOTIFICATIONS[s];
    for (const { re, cn } of NOTIFICATION_PATTERNS) {
      const m = s.match(re);
      if (m) return cn(m);
    }
    return null;
  };
  let hit = lookup(raw);
  // 上游有几条提示语是跨行的模板串（ember.mjs:36922 / 126652），源码里的换行 + 缩进
  // 原样进了消息文本。折叠内部空白后再查一次，与 translateNode 的回退同策略。
  if (hit === null) {
    const flat = raw.replace(/\s+/g, " ");
    if (flat !== raw) hit = lookup(flat);
  }
  return hit === null ? text : text.replace(raw, hit);
}

/**
 * 盯住一棵**渲染钩子之后还会被上游改写**的子树，改一次翻一次。
 *
 * 两个用例：音景面板的两个 `<select>`（`#updateSoundscapeForm()` 在 change 之后 **异步**
 * 整段重写 innerHTML，不发任何渲染钩子）、指示物制作器的动画按钮
 * （`#refresh()` 在 `await this.render()` 之后才写回 data-tooltip，比渲染钩子晚一步）。
 *
 * 不会自激：我们自己只改 nodeValue 与属性值，`childList` 不会因此变动；
 * 属性那一路虽然会被自己的 setAttribute 再触发一次，但第二次查表已经查不到中文，
 * 不再 set，最多多跑一轮就收敛。
 * @param {HTMLElement} el
 * @param {Record<string, string>} table
 * @param {string[]} [attributeFilter]  传了就同时盯这些属性
 */
function observeSubtree(el, table, attributeFilter = null) {
  if (!el || el.__emberCnObserved) return;
  try {
    const obs = new MutationObserver(() => translateNode(el, table));
    obs.observe(el, attributeFilter
      ? {childList: true, subtree: true, attributes: true, attributeFilter}
      : {childList: true, subtree: true});
    Object.defineProperty(el, "__emberCnObserved", {value: true, enumerable: false});
  } catch (err) {
    warn("挂子树监听失败：", err);
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
 * 允许包装的富文本增强器 id 白名单。
 * Ember 侧：ember.mjs:129421-129476 的十条 + dnd5e 专用的 emberKnowledge。
 * ⚠ `emberKnowledge` 的出处 2026-08-15 订正为 **ember.mjs:123659**（旧注释写 dnd5e-async.mjs，是错的；
 *   它注册在 dnd5e 专用的 registerEnrichers$1 里，但那段代码在 ember.mjs 内）。
 */
const EMBER_ENRICHER_IDS = new Set([
  "emberAncestry", "emberCulture", "emberPath", "emberAttunement", "emberLanguage",
  "emberSoundscape", "emberEventState", "emberEventOutcome", "emberAdvantage",
  "emberCriticalResult", "emberKnowledge"
]);

/** crucible 本体里**故意**一起包住的四条（理由见 patchEnrichers 里的注释） */
const CRUCIBLE_OPT_IN_ENRICHER_IDS = new Set([
  "crucibleKnowledge", "crucibleLanguage", "crucibleTalent", "crucibleSpell"
]);

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
    // 按**归属**放行，不再拿 entry.pattern 的字符串形式做子串匹配。
    // 原来那道闸的关键词表里有 `talent` / `spell` / `path` / `date` 这种极常见的子串，
    // 任何第三方增强器只要正则里出现这些字就会被无声包住，返回节点还会被套上全局 EXACT
    // （里头有 Token / Path / Culture / Events / Complete 这类通用词）。实测越界过一个：
    // crucible 的 crucibleCounterspell（pattern 含子串 spell），以及 dnd5e 本体的 dnd5e-lookup
    // （dnd5e.mjs:20157，pattern 含 language）—— 本模块 module.json 不限系统，dnd5e 世界里真的开着。
    // 归属信息是现成的：上游给每个增强器都写了 id（ember.mjs:129421-129476 十个、
    // dnd5e-async 的 emberKnowledge、crucible-compiled.mjs:46114-46190 十二个）。
    // crucible 那四个是**故意 opt-in** 的：enrichTalent（46838）拼裸模板 `Talent: ${name}`、
    // enrichSpell（46724）把 "Spell tooltips are still TO-DO." 写进 data-tooltip，两处都不走 _loc，
    // 而 crucible 汉化插件那边只有 babele-register.js（它自己也按 id === "crucibleTalent" 兜了一次，
    // 这里重复包一层是幂等的：先跑的那层已经译成中文，后跑的查表落空）。
    const id = String(entry.id ?? "");
    // Ember 的日历 date 增强器（ember.mjs:129404）是**唯一没有 id** 的一条，按 pattern 头部精确认。
    const isEmberDate = !id && String(entry.pattern?.source ?? "").startsWith("\\[\\[\\/date ");
    if (!EMBER_ENRICHER_IDS.has(id) && !CRUCIBLE_OPT_IN_ENRICHER_IDS.has(id) && !isEmberDate) continue;
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
 * Ember **自己**往 crucible.CONFIG 里塞的语言 / 知识领域 / 语言分类，label 是硬编码英文。
 * 这些 label 会出现在角色卡的下拉框与 optgroup 组名上，改数据是唯一的办法。
 *
 * 三张 key 白名单逐条抄自 ember.mjs:126681-126718 的写入点 —— 这道闸原先是**按 label 的
 * 英文值查表**决定改谁，不看归属：本模块的 KNOWLEDGE 表前 31 条与 LANGUAGES 表的
 * Common/Sign 都是 **crucible 本体**的条目（crucible-compiled.mjs:586 / :1007），
 * 单装 ember 汉化（module.json 不 require crucible-cn）时会把它们一起改掉，属于越界。
 * 按 key 认归属之后，crucible 自己的 33 条一律不碰。
 * 注意 `Thieves' Cant` 的 key 是 `cant` 不是 `thieves`。
 */
const EMBER_LANGUAGE_KEYS = new Set([
  "arcden", "cascal", "forest", "hardac", "imperial", "solical", "mithia", "luma", "kaziric",
  "scripta", "wyrdic", "pathward", "scor", "towyr", "windclaw", "abyssal", "draconic", "druidic",
  "lunix", "caligon", "eonic", "harmos", "cant"
]);
const EMBER_KNOWLEDGE_KEYS = new Set(["abyssals", "aedir", "leviathans", "shent"]);
const EMBER_LANGUAGE_CATEGORY_KEYS = new Set(["ancient", "obscure"]);

function patchCrucibleConfig() {
  const cfg = globalThis.crucible?.CONFIG;
  if (!cfg) return warn("找不到 crucible.CONFIG，配置补丁跳过。");
  let n = 0;
  for (const [key, table, allowed] of [
    ["languages", LANGUAGES, EMBER_LANGUAGE_KEYS],
    ["knowledge", KNOWLEDGE, EMBER_KNOWLEDGE_KEYS],
    ["languageCategories", LANGUAGE_CATEGORIES, EMBER_LANGUAGE_CATEGORY_KEYS]
  ]) {
    const group = cfg[key];
    if (!group) continue;
    for (const [id, entry] of Object.entries(group)) {
      if (!allowed.has(id)) continue;
      if (entry && typeof entry.label === "string" && table[entry.label]) {
        entry.label = table[entry.label];
        n++;
      }
    }
  }
  log(`已改写 crucible.CONFIG 里 ${n} 条 Ember 新增的语言/知识标签。`);
}

/**
 * 把 core CONFIG 里 Ember 塞的两个裸英文 label 换成中文（见 CORE_CONFIG_LABELS）。
 * 只在**值还是那个英文原串**时才改，重复执行或上游改了文案都不会误伤。
 */
function patchCoreConfig() {
  let n = 0;
  for (const [path, en, cn] of CORE_CONFIG_LABELS) {
    const target = foundry.utils.getProperty(CONFIG, path);
    if (target && target.label === en) { target.label = cn; n++; }
  }
  log(`已改写 core CONFIG 里 ${n} 条 Ember 的 label。`);
}

/**
 * 把十个月亮的名字改成中文（见 MOON_NAMES）。
 * 日历面板的月亮 tooltip 是 `${moon.name} ${moon.phaseLabel}`（ember.mjs:24628），
 * 由 animate() 每帧写回、不发渲染钩子，DOM 层翻了也留不住 —— 只能改数据。
 * moons 在 setup 钩子里就实例化好了（ember.mjs:3772 EmberCalendar#initialize），ready 时改得到。
 */
function patchMoonNames() {
  const moons = globalThis.ember?.calendar?.moons;
  if (!moons) return warn("找不到 ember.calendar.moons，月亮名补丁跳过。");
  let n = 0;
  for (const moon of Object.values(moons)) {
    if (moon && typeof moon.name === "string" && MOON_NAMES[moon.name]) {
      moon.name = MOON_NAMES[moon.name];
      n++;
    }
  }
  log(`已改写 ${n} 个月亮名。`);
}

/**
 * 包住 `ui.notifications.notify`（info / warn / error 三个方法最后都调它）。
 * 这是 Ember 那 65 处提示语唯一够得到的地方，见 NOTIFICATIONS 的注释。
 * `message` 可能是 Error 对象（ember.mjs:3063 等处直接把 err 丢进来），
 * 那种一律原样透传 —— core 靠 `message instanceof Error` 决定要不要打堆栈。
 */
function patchNotifications() {
  const n = globalThis.ui?.notifications;
  if (!n || typeof n.notify !== "function") return warn("找不到 ui.notifications.notify，提示语补丁跳过。");
  const original = n.notify;
  n.notify = function (message, type, options) {
    // `localize: true` 的那几条传的是 i18n 键（ember.mjs:24053 / 121506 / 121538 / 129796），
    // 归 lang/cn.json 管，这里一律不碰。
    // 注：core 的 notify 是**无条件**跑 `_loc(message, format)` 的（notifications.mjs:121），
    // localize 选项只影响 clean，所以即便漏了这道判断，键形字符串也查不中我们的表；
    // 显式跳过只是把意图写清楚。
    if (typeof message === "string" && !options?.localize) message = translateNotification(message);
    return original.call(this, message, type, options);
  };
  log("已包住 ui.notifications.notify。");
}

/**
 * 包住画布滚动文字。PIXI 文本不在 DOM 里，DOM 遍历那一套够不到。
 * 包在**类原型**上而不是 `canvas.interface` 实例上：切场景会重建 InterfaceCanvasGroup，
 * 挂在实例上会掉。第二参数可能带换行（ember.mjs:20807 `Discovery!\n${label}`），逐行查表。
 */
/**
 * Ember 三个 RegionBehavior 子类型的整张配置表单是裸英文：`defineSchema()` 直接把
 * 英文原文写进了 `label` / `hint` / `choices`（ember.mjs:2554-2570 / 2685-2704 / 2765-2776），
 * 合计 13 个 label + 12 条 hint + 5 个 choices。
 *
 * ⚠ **不能按 i18n 键修。** 这批串里 `Once` / `Locked` / `Discovered` / `Script` /
 * `Material` / `Grass` / `Metal` / `Stone` / `Water` / `Wood` 全是通用词，
 * 写进全局 i18n 表以后任何模块 `localize("Water")` 都会拿到我们的译文 ——
 * 正是 PROJECT.md §8 `2026-08-14c` 已经否决过的做法，而且这里通用词的比例更高。
 *
 * 做法：从 `CONFIG.RegionBehavior.dataModels` 按 type 取到类，就地改写它自己 schema 上的
 * 那几个字段。作用域精确到 Ember 的三个子类型，不碰全局 i18n 表，也不依赖表单 DOM 选择器。
 * 字段名对不上（上游改过 schema）就跳过那一条并告警，不静默。
 */
const REGION_BEHAVIOR_FIELDS = {
  "ember.trapTrigger": {
    once: { label: "仅一次", hint: "触发一次后，触发器是否自动停用？" },
    locked: { label: "已锁定", hint: "触发器能否被解除？若已锁定，则无法解除该触发机关。" },
    discovered: { label: "已发现", hint: "该陷阱是否已被发现？" },
    behaviors: { label: "被触发的行为", hint: "填写本区域或其他区域中应由此陷阱触发的行为的 UUID。" },
    script: { label: "脚本", hint: "陷阱被触发时执行的自定义 JavaScript。" },
    message: { label: "触发文本", hint: "陷阱被触发时显示的滚动消息文本。" },
    pause: { label: "暂停游戏", hint: "陷阱被触发时是否自动暂停游戏？" }
  },
  "ember.areaEffect": {
    // img 的 label 是 "EFFECT.Image" —— 那是**真的** i18n 键，不要动
    description: { label: "聊天消息描述", hint: "应用此区域效果时，在聊天消息中显示的 HTML 描述文本。" },
    save: { hint: "配置此区域效果所需的豁免检定。若未配置属性值，则不需要豁免。" },
    "save.ability": { label: "属性值" },
    "save.dc": { label: "豁免 DC" },
    damage: { label: "伤害公式", hint: "按 {type: string, formula: string} 的格式定义一组伤害公式部分。" },
    effects: { label: "效果数据", hint: "一组 ActiveEffect 数据，会应用到受此区域效果影响的 Actor 上。" }
  },
  "ember.footstepSurface": {
    material: {
      label: "材质",
      hint: "该表面的材质类型。",
      choices: { grass: "草地", metal: "金属", stone: "石头", water: "水", wood: "木头" }
    }
  }
};

function patchRegionBehaviorSchemas() {
  const models = globalThis.CONFIG?.RegionBehavior?.dataModels;
  if (!models) return warn("找不到 CONFIG.RegionBehavior.dataModels，区域行为表单补丁跳过。");

  let patched = 0;
  let missed = 0;
  for (const [type, fieldSpecs] of Object.entries(REGION_BEHAVIOR_FIELDS)) {
    const schema = models[type]?.schema;
    if (!schema) {
      missed += Object.keys(fieldSpecs).length;
      warn(`区域行为 ${type} 不在 CONFIG.RegionBehavior.dataModels 里，跳过。`);
      continue;
    }
    for (const [path, spec] of Object.entries(fieldSpecs)) {
      // SchemaField 的子字段走 a.b；schema.getField 认这种点号路径
      const field = schema.getField?.(path) ?? schema.fields?.[path];
      if (!field) {
        missed += 1;
        warn(`区域行为 ${type} 的字段「${path}」不存在（上游改过 schema？），跳过。`);
        continue;
      }
      if (spec.label !== undefined) field.label = spec.label;
      if (spec.hint !== undefined) field.hint = spec.hint;
      if (spec.choices !== undefined) {
        // 只改**已存在**的选项值，多出来的键说明上游换了枚举，宁可留英文也不要凭空造选项
        for (const [k, v] of Object.entries(spec.choices)) {
          if (k in (field.choices ?? {})) field.choices[k] = v;
          else { missed += 1; warn(`区域行为 ${type}.${path} 没有选项「${k}」，跳过。`); }
        }
      }
      patched += 1;
    }
  }
  log(`区域行为配置表单：改写 ${patched} 个字段${missed ? `，${missed} 处对不上已告警` : ""}。`);
}

/**
 * 远景摆放页（`Ember Vista Configuration` 的 placement 页签）的裸英文 field label。
 *
 * 与上面 `REGION_BEHAVIOR_FIELDS` **完全同一个机制**：`EmberVistaConfiguration.PLACEMENT_SCHEMA`
 * （ember.mjs:33925-33944，外加 :33918-33923 的 `ILLUMINATION_SCHEMA`）是一个 `SchemaField`，
 * 每个子字段的 `label` 直接写着英文原文；模板 `templates/applications/vista-config-placement.hbs`
 * 一半靠 `{{fields.x.label}}` 直接吐、一半靠 `{{formGroup fields.sort …}}` 吐，
 * 而 `_prepareContext` 把 `fields: EmberVistaConfiguration.PLACEMENT_SCHEMA.fields`
 * 原样塞进上下文（:34141）—— 也就是说模板读的就是**这一批对象**，改 schema 三处一起变。
 *
 * ⚠ **不能按 i18n 键修，也不该走 EXACT / EMBER_WINDOW_UI 补键**：
 *   `Sort` / `Angle` / `Alpha` / `Tint` / `Only` 全是通用词，进全局 i18n 表会越界改别的模块
 *   （PROJECT.md §8 `2026-08-14c` 已否决过），进 EMBER_WINDOW_UI 则会作用到**所有** Ember
 *   窗口的同名文本节点上 —— 那张表现在有 90 多条、宿主是整个 Ember UI，`Only` 这种词放进去
 *   风险不对等。改 schema 的作用域精确到这一张表单，且不依赖 DOM 结构。
 *
 * ⚠ **`x` / `y` / `scaleX` / `scaleY` / `skewX` / `skewY` 这 6 条有意不动**：它们的 label
 *   就是单个字母 `X` / `Y`（坐标轴记号，和 DC / AC 一样属于不译的记号），译了反而认不出。
 *   模板里包着它们的三个小标题 `Coordinates` / `Scale` / `Skew` 是**模板里的裸串**、
 *   不是 schema label，已在 EMBER_WINDOW_UI 里（坐标 / 缩放 / 斜切），DOM 遍历接得住。
 *
 * `Elevation`＝高度 与本文件既有的 `Lock Vista Placement Elevation`「锁定远景摆放高度」
 * （按键绑定名，同一个远景摆放域）对齐。`Colorization`＝着色 与模板裸串
 * `Precolorization`「预着色」对齐。
 * `Illumination.only` 上游没给 hint，按它在 `Illumination` 分组里、与 Luminosity /
 * Blur Strength 并列的位置读作「只渲染照明层」，故译「仅照明」——**这一条是推断，不是实证**，
 * 哪天能进游戏看到实际效果再复核。
 */
const VISTA_PLACEMENT_FIELDS = {
  elevation: { label: "高度" },                     // ember.mjs:33928
  sort: { label: "排序" },                          // :33929
  angle: { label: "角度" },                         // :33934
  alpha: { label: "不透明度" },                     // :33935
  tint: { label: "染色" },                          // :33936
  illumination: { label: "照明" },                  // :33923
  "illumination.luminosity": { label: "亮度" },     // :33920
  "illumination.only": { label: "仅照明" },         // :33921
  "illumination.blurStrength": { label: "模糊强度" }, // :33922
  colorize: { label: "着色" },                      // :33943
  "colorize.texture": { label: "自定义颜色贴图" },   // :33939
  "colorize.r": { label: "红色通道" },              // :33940
  "colorize.g": { label: "绿色通道" },              // :33941
  "colorize.b": { label: "蓝色通道" }               // :33942
};

/**
 * 改写 `EmberVistaConfiguration.PLACEMENT_SCHEMA` 上那 14 个 field 的 label（见上）。
 *
 * 类从 `ember.api.applications` 取（ember.mjs:128987-129001 在 init 里挂上，51969 那个
 * 命名空间对象虽然 `Object.freeze` 过，但冻的是命名空间本身，类与 schema 字段都没冻）。
 * 走 `schema.getField(path)`，与 patchRegionBehaviorSchemas 一致，认 `a.b` 点号路径。
 * 只在 label **还是英文原串**时才改，所以重复执行、上游改文案都不会误伤；
 * 字段名对不上就跳过那一条并告警，不静默。
 */
const VISTA_PLACEMENT_EN = {
  elevation: "Elevation", sort: "Sort", angle: "Angle", alpha: "Alpha", tint: "Tint",
  illumination: "Illumination",
  "illumination.luminosity": "Luminosity",
  "illumination.only": "Only",
  "illumination.blurStrength": "Blur Strength",
  colorize: "Colorization",
  "colorize.texture": "Custom Color Texture",
  "colorize.r": "Red Channel",
  "colorize.g": "Green Channel",
  "colorize.b": "Blue Channel"
};

function patchVistaPlacementSchema() {
  const schema = globalThis.ember?.api?.applications?.EmberVistaConfiguration?.PLACEMENT_SCHEMA;
  if (!schema) return warn("找不到 EmberVistaConfiguration.PLACEMENT_SCHEMA，远景摆放表单补丁跳过。");

  let patched = 0;
  let missed = 0;
  for (const [path, spec] of Object.entries(VISTA_PLACEMENT_FIELDS)) {
    const field = schema.getField?.(path) ?? schema.fields?.[path];
    if (!field) {
      missed += 1;
      warn(`远景摆放字段「${path}」不存在（上游改过 schema？），跳过。`);
      continue;
    }
    const en = VISTA_PLACEMENT_EN[path];
    if (field.label !== en && field.label !== spec.label) {
      missed += 1;
      warn(`远景摆放字段「${path}」的 label 是「${field.label}」而非「${en}」（上游改过文案？），跳过。`);
      continue;
    }
    field.label = spec.label;
    patched += 1;
  }
  log(`远景摆放配置表单：改写 ${patched} 个字段${missed ? `，${missed} 处对不上已告警` : ""}。`);
}

/**
 * CDT（content-development-toolkit）的 ProseMirror 插入菜单里，Ember 注册的 15 个模板块标题。
 *
 * 上游在 `registerProseMirrorBlocks()`（ember.mjs:128719-128737）里往
 * `CONFIG.CDT.PROSEMIRROR.blocks` push 15 条 `{action, title, html}`，`title` 是裸英文。
 * 只在 `ember.developmentMode`（= `game.data.options.debug`）且 CDT 模块启用时才注册
 * （:129140-129143 / :128709），所以这是**内容作者侧**的界面，普通开世界看不到。
 *
 * ⚠ **不走 `INJECTED_SUBTREES` 选择器**（第十六轮的升级建议原本提的是这条路）：
 *   ① 宿主确实过不了主闸（CDT 的菜单根既不含 `ember` 类名、类名也不以 `Ember` 开头），
 *      但**这里根本不需要 DOM** —— `blocks` 是一个我们够得到的运行时数组，CDT 每次建菜单
 *      时现读它，跟 WEATHER / REGION_BEHAVIOR 一样属于「改数据」那一档，比 DOM 稳。
 *   ② CDT 未安装在本机（`modules/` 下没有 content-development-toolkit），
 *      写一条**无法实测**的选择器等于埋一条恒不命中或误命中的规则；而 `title` 很可能被 CDT
 *      渲染成 `title=` / `data-tooltip=` 属性，那样 DOM 文本遍历本来就够不到。
 *   ③ 按 `action`（稳定 id）认条目、不按标题文本认，上游改英文文案也不会误改。
 *
 * 译名取法：`Block`＝区块（glossary_ec 的 `Hazard block`＝危害区块）；
 * `WIP`＝制作中（合集里 `Work In Progress!`＝制作中！）；`Gamemaster`＝游戏主持人（glossary_ec）；
 * `divider` 是 `<h2 class="divider">` 那种分隔标题，不是横线，故作「分隔标题」。
 */
const PROSEMIRROR_BLOCK_TITLES = {
  "insert-block-readaloud": "朗读区块",
  "insert-block-hazard": "危害区块",
  "insert-block-exploration": "探索区块",
  "insert-block-social": "社交区块",
  "insert-block-complex-skill": "复合技能检定",
  "insert-block-qna": "问答区块",
  "insert-block-gamemaster": "游戏主持人区块",
  "insert-block-attunement": "同调区块",
  "insert-block-wip": "制作中区块",
  "insert-h2-divider": "H2 分隔标题",
  "insert-h3-divider": "H3 分隔标题",
  "insert-definition-list": "定义列表",
  "insert-block-system": "系统切换区块",
  "insert-inline-system": "系统切换内联",
  "insert-skill-check": "技能检定内联"
};

function patchProseMirrorBlocks() {
  // CDT 没装 / 世界不在 debug 模式时上游根本不会注册，这不是异常，静默返回。
  const blocks = globalThis.CONFIG?.CDT?.PROSEMIRROR?.blocks;
  if (!Array.isArray(blocks)) return;

  let patched = 0;
  for (const block of blocks) {
    const cn = PROSEMIRROR_BLOCK_TITLES[block?.action];
    // 只认 Ember 自己注册的 action，且只在 title 还是英文时改（幂等）
    if (cn && typeof block.title === "string" && block.title !== cn) {
      block.title = cn;
      patched += 1;
    }
  }
  const total = Object.keys(PROSEMIRROR_BLOCK_TITLES).length;
  if (patched) log(`ProseMirror 模板块菜单：改写 ${patched}/${total} 条标题。`);
  else warn(`CONFIG.CDT.PROSEMIRROR.blocks 在，但没认出 Ember 的 ${total} 条模板块（上游改过 action？）。`);
}

function patchScrollingText() {
  const proto = foundry.canvas?.groups?.InterfaceCanvasGroup?.prototype;
  if (typeof proto?.createScrollingText !== "function") {
    return warn("找不到 InterfaceCanvasGroup#createScrollingText，滚动文字补丁跳过。");
  }
  const original = proto.createScrollingText;
  proto.createScrollingText = function (origin, content, options) {
    if (typeof content === "string") {
      content = content.split("\n").map(line => SCROLLING_TEXT[line.trim()] ?? line).join("\n");
    }
    return original.call(this, origin, content, options);
  };
  log("已包住画布滚动文字。");
}

/**
 * 把两个区域切片的天气 / 风力档位名改成中文（见 WEATHER）。
 *
 * 目标是 `ember.scenes.region.slices[*].weather` —— 那正是 `ember.weather.getConfig()`
 * （ember.mjs:21825 `ember.region.slices[event.slice]?.config.weather[event.type]`）拿到的**同一批对象**：
 * EmberRegionMap#configureHexes（:59481-59486）建 slice 时 `{…, config}` 直接引用配置字面量，没有深拷贝。
 * 走静态配置而不是 `ember.region.*`，是因为 ready 时区域地图不一定初始化过，而 `ember.scenes`
 * 在 init 钩子里就赋好了（ember.mjs:129093）且这批对象没有被 freeze（全文件只有 `ember.CONST` 被冻）。
 *
 * 只在**值还是那个英文原串**时才改（查表落空就不动），所以重复执行、上游改文案、
 * 或者别的模块先改过，都不会被误伤。
 */
/**
 * 日历条右侧那排天气图标与风向箭头的**悬浮提示**。
 *
 * ⚠ 路径曾经写错，整个补丁一直是空转（改写 0 条，玩家看到的一直是英文）。
 * 真实取值链是 `EmberWeatherManager#getConfig()`（ember.mjs:21825）：
 *   `ember.region.slices[<sliceId>].config.weather[<type>]`
 * 也就是 **`slice.config.weather`**，而不是 `slice.weather` —— 后者在 slice 上根本不存在
 * （`slice.weather` 只在 Vista 场景定义里出现，而且只有 `elevation` 一个键，没有 label）。
 * 渲染点：`#refreshWeather()`（:24656 `icon.dataset.tooltip = str?.label ?? cfg.label`）
 * 与 :24673（`windArrow.dataset.tooltip = \`${strengths[..].label} (${speed} mph)\``）。
 * 后者是**拼接串**，所以补 i18n 键没用，只能改数据本身。
 *
 * 两条路径都走一遍：`config.weather` 是当前上游的形状，`weather` 留作兜底，
 * 上游哪天把它挪回去也不至于又静默空转。改写 0 条时**告警**，不再静默。
 */
function patchWeatherLabels() {
  const slices = globalThis.ember?.scenes?.region?.slices;
  if (!slices) return warn("找不到 ember.scenes.region.slices，天气名补丁跳过。");

  let n = 0;
  const seen = new Set();
  const relabel = (o) => {
    if (!o || typeof o.label !== "string" || seen.has(o)) return;
    seen.add(o);
    const cn = WEATHER[o.label];
    if (cn) { o.label = cn; n++; }
  };
  const walkWeather = (table) => {
    for (const cfg of Object.values(table ?? {})) {
      relabel(cfg);
      for (const str of Object.values(cfg?.strengths ?? {})) relabel(str);
    }
  };

  for (const slice of Object.values(slices)) {
    walkWeather(slice?.config?.weather);   // ← 当前上游的真实位置
    walkWeather(slice?.weather);           // ← 兜底
  }

  if (!n) warn("天气/风力档位名一条都没改到 —— 上游多半又挪了 slices 的形状，去核对 getConfig() 的取值链。");
  else log(`已改写 ${n} 条天气/风力档位名（含风力，走 config.weather）。`);
}

/**
 * 让日历条重画一次。
 *
 * patchMoonNames / patchWeatherLabels 改的都是**数据**，而写上屏的
 * `#refreshMoons()`（ember.mjs:24614）与 `#refreshWeather()`（:24637）都挂在 `animate()` 下面，
 * 只有时间推进或重新渲染时才跑。日历条多半在 ready 之前就渲染过了，不重画一次就要等到
 * 下一次时间变动才显示中文。
 *
 * ⚠ 这里**不要**遍历 `ui.windows`：它只装 AppV1 popOut 实例（core appv1/api/application-v1.mjs:415
 * 是全库唯一写入点，client/ui.mjs:19-21 的类型就是 `Record<string, appv1.api.Application>`），
 * 而目标 EmberCalendarNavigation 是 AppV2（ember.mjs:24382 `HandlebarsApplicationMixin(ApplicationV2)`，
 * DEFAULT_OPTIONS 还是 `tag:"aside"` / `window.frame:false`），两个集合不相交 —— 老写法恒 0 命中，
 * 反倒会把第三方 AppV1 里类名带 calendar 的窗口强制重画。上游自己重画日历用的是
 * `ember.ui.calendar.render()`（:129518 / :129693），按归属直接取实例即可。
 * 老写法里那句对 `#ember-calendar` 派发 `new Event("change")` 也已删除：该 `<aside>` 上没有任何
 * change 监听（ember.mjs 全库 `addEventListener("change"` 六处无一在日历上），且 bubbles 默认 false。
 *
 * `render(false)` 传的是 `{force:false}`（core application.mjs:501 会把布尔转成 force），
 * 语义正是「已经开着才刷新，没开就什么都不做」（:521 `if (options.isFirstRender && !options.force) return this`），
 * 所以本模块 ready 若跑在 ember 首次 render 之前，这里是安全的空操作。
 */
function refreshCalendarUI() {
  try {
    const seen = new Set();
    for (const app of [globalThis.ember?.ui?.calendar, ...foundry.applications.instances.values()]) {
      if (!app || seen.has(app)) continue;
      seen.add(app);
      if (app.constructor?.name === "EmberCalendarNavigation") app.render(false);
    }
  } catch (err) {
    // 老写法这里是个空 catch，坏了零信号。至少留一条警告。
    warn("日历条重画失败（不影响下次开界面）：", err);
  }
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
        // 例外二：Ember 把自己的东西塞进**别人的**窗口 —— crucible HeroSheet 的「同调」页与
        // 天赋列表（ember.mjs:124717 / 126104 起）、播放列表侧栏的音景面板（:15874）、
        // NoteConfig 的「Ember Type」表单组（:64962）、设置面板与按键面板里 ember 命名空间那一段、
        // 场景控制栏上的 Show Tracks。这些宿主的 class 是别人的、类名也不以 Ember 开头，
        // 上面两个判据都不成立，整个窗口在闸外。按注入点选择器单独放行子树，
        // 且一律只用作用域表，不会顺手把宿主自己的界面改了。
        for (const [selector, table, opts] of INJECTED_SUBTREES) {
          let subs;
          try {
            subs = root.querySelectorAll?.(selector) ?? [];
          } catch (err) {
            // 选择器语法在某个环境里不被支持（例如 :has）也只跳过这一条，别把后面几条一起废掉
            warn(`选择器「${selector}」不可用：`, err);
            continue;
          }
          for (const sub of subs) {
            translateNode(sub, table);
            // 渲染钩子之后还会被上游改写的子树要盯住，见 observeSubtree
            if (opts?.observe) observeSubtree(sub, table);
          }
        }
        return;
      }
      // Ember 自己的窗口：全局三张表 + 一张「只在 Ember 窗口里才敢用」的作用域表。
      // 带 dialog 类名的（如 Create Weather 那个 classes 含 "ember-hex-selection-dialog" 的框）
      // 顺带把 DIALOG_UI 一起带上 —— 归属已经确定，那张表里的按钮词在这里同样安全。
      const isDialog = id === "DialogV2" || /(^|\s)dialog(\s|$)/.test(cls);
      translateNode(root, isDialog ? EMBER_DIALOG_UI : EMBER_WINDOW_UI);
      // 指示物制作器的动画开关：#refresh()（ember.mjs:50994-51004）在 `await this.render()`
      // **之后**才把 data-tooltip 写回英文，也就是比渲染钩子晚一步，而且它只重画
      // parts:["layers","colors"]、不重画 body，模板里那句译了也会被覆盖。挂 observer 盯着。
      if (id === "EmberDynamicTokenConfig") {
        const anim = root.querySelector?.('[data-action="toggleAnimation"]');
        if (anim) observeSubtree(anim, EMBER_WINDOW_UI, ["data-tooltip"]);
      }
      // 日历条的风向箭头：`#refreshWeather()`（ember.mjs:24669-24673）在 animate() 里写
      // `data-tooltip = ${档位名} (${速度} mph)`，不发渲染钩子，渲染时翻过的那一次会被覆盖回去。
      // 档位名那半截已由 patchWeatherLabels 在数据侧解决，剩下的 `mph` 走 PATTERNS，
      // 所以这里挂一个只盯 data-tooltip 的观察者补上。
      // ⚠ 只盯这**一个**元素，不要盯日历根：`#refreshMoons()` 每帧重写 10 个月亮的 data-tooltip，
      //    盯根就是每帧一次全树遍历。风向那条只在天气真的变了才写（:24641 的早退判据）。
      if (id === "EmberCalendarNavigation") {
        const windArrow = root.querySelector?.("#ember-calendar-wind-arrow");
        if (windArrow) observeSubtree(windArrow, EMBER_WINDOW_UI, ["data-tooltip"]);
      }
    } catch (err) {
      warn("界面文本翻译失败：", err);
    }
  };
  Hooks.on("renderApplicationV2", handler);
  Hooks.on("renderApplication", handler);

  // 第三个入口：聊天卡。同调奖励卡（ember.mjs:2957-2985）的表头拼的是
  // `${config.label} ${_loc("EMBER.ATTUNEMENT.Attunement")}`（:2971），前半截是那 11 个裸英文
  // 短名，屏幕上就是「Abyss 同调」这种半英半中。聊天栏没有 Application 宿主，
  // 前面两个渲染钩子都够不到。
  // ⚠ 只翻**打了 ember 旗标**的消息（那张卡是 :2982 `flags:{ember:{attunementAward:true}}` 造的），
  //    绝不无条件翻整个聊天栏 —— 那会把别的模块的卡也一起改了。
  Hooks.on("renderChatMessageHTML", (msg, html) => {
    try {
      if (!msg?.flags?.ember) return;
      translateNode(html instanceof HTMLElement ? html : html?.[0], CHAT_UI);
    } catch (err) {
      warn("聊天卡文本翻译失败：", err);
    }
  });
  log("已挂上界面渲染钩子。");
}

/**
 * 用 lang 里**当前生效的**译文拼出聊天卡表头那种「英文短名 + 空格 + 中文词」的复合键。
 * 写死 "Abyss 同调" 会在有人改了 EMBER.ATTUNEMENT.Attunement 之后失效，所以在 ready 里现拼。
 * 值取 ATTUNEMENT_ITEM_NAMES 的中文段（合集里 `Heart Attunement` 定的是「心之同调」，
 * 不是「余烬之心」+「同调」，拿短名表拼会跟合集打架）。
 */
function buildChatKeys() {
  const att = game.i18n?.localize("EMBER.ATTUNEMENT.Attunement");
  if (!att || att === "EMBER.ATTUNEMENT.Attunement") return warn("取不到同调一词的译文，聊天卡表头补丁跳过。");
  let n = 0;
  for (const [en, cn] of Object.entries(ATTUNEMENT_ITEM_NAMES)) {
    CHAT_UI[`${en} ${att}`] = cn.split(" ")[0];   // "深渊同调 Abyss Attunement" → "深渊同调"
    n++;
  }
  log(`已为聊天卡表头生成 ${n} 条复合键。`);
}

/* ============================================================ */
/*  4. 入口                                                      */
/* ============================================================ */

Hooks.once("ready", () => {
  if (!game.modules.get("ember")?.active) return;
  applyOnce(CONFIG, "__emberCnEnrichers", patchEnrichers, "富文本增强器");
  // 幂等标记要落在**长期存活**的对象上。原先传的是 `globalThis.crucible?.CONFIG ?? {}`：
  // dnd5e 世界里 crucible 缺席，`?? {}` 每次求值都造一个新对象，标记写在临时对象上，闸恒不生效。
  // crucible 缺席的判断由 patchCrucibleConfig 自己做（见函数开头的 warn 返回）。
  applyOnce(CONFIG, "__emberCnConfig", patchCrucibleConfig, "crucible.CONFIG");
  applyOnce(CONFIG, "__emberCnCoreConfig", patchCoreConfig, "core CONFIG");
  applyOnce(CONFIG, "__emberCnMoons", patchMoonNames, "月亮名");
  applyOnce(CONFIG, "__emberCnWeather", patchWeatherLabels, "天气档位名");
  applyOnce(CONFIG, "__emberCnDayFormat", patchCalendarFormatters, "历法日期格式");
  applyOnce(CONFIG, "__emberCnNotifications", patchNotifications, "通知提示语");
  applyOnce(CONFIG, "__emberCnScrollingText", patchScrollingText, "画布滚动文字");
  applyOnce(CONFIG, "__emberCnRegionBehaviors", patchRegionBehaviorSchemas, "区域行为配置表单");
  applyOnce(CONFIG, "__emberCnVistaPlacement", patchVistaPlacementSchema, "远景摆放配置表单");
  applyOnce(CONFIG, "__emberCnProseMirror", patchProseMirrorBlocks, "ProseMirror 模板块菜单");
  applyOnce(CONFIG, "__emberCnChatKeys", buildChatKeys, "聊天卡复合键");
  applyOnce(CONFIG, "__emberCnRender", patchRenderedApplications, "界面渲染");
  // 上面三条改的都是日历条读的数据（月亮名 / 天气档位名 / 日期格式），改完统一重画一次。
  // 已经开着才刷新，没开就是空操作，见 refreshCalendarUI。
  refreshCalendarUI();
  log("Ember 硬编码字符串补丁已就绪。");
});
