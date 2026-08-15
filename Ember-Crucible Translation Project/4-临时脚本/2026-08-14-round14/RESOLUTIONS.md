# 第十四轮 · 主控裁决

## R1 · 两个 `lang` newkeys 批次**全部驳回**（63 条键）

`lang` 单元诚实地报了 `gate_clean: false`：两个**主**批次（crucible 13 条 / ember 8 条）
`--dry` 零拒绝且拍平三数相等（1842 / 486），**予以采纳**；
两个 `newkeys` 批次（crucible 5 条 / ember 58 条）被 `apply_lang.py` 第 1 道闸
（键必须存在于上游 `en.json`）结构性全拒。

**裁决：驳回，不放行闸。** 逐类理由：

| 类别 | 条数 | 为什么驳回 |
|---|---|---|
| ember 的 26 条 `ui.notifications` 整句 | 28 | **已经被覆盖了**。同轮 `hc-mjs` 的第二/三段建好了 `patchNotifications()`，枚举了 `ember.mjs` 里全部 76 处 notification 调用（62 处裸英文），`NOTIFICATION_PATTERNS` 18→31、`NOTIFICATIONS` 补 6 条。再走一遍 lang 就是**同一件事的第二条全局通道**，两边一旦分叉必然同屏矛盾 |
| RegionBehavior schema 的 30 串（idx 264） | 30 | 见 R2 —— 该走 schema 就地改写，不是 i18n 键 |
| `Show Tracks` | 1 | 见 R3 —— 该走 `getSceneControlButtons` |
| `None` / `Unknown` | 2 | **裸通用词**，正是 §8 `2026-08-14c` 已否决的做法 |
| `AFFIX.Affix` | 1 | 上游 `crucible/lang/en.json` 里 `AFFIX.*` 有 22 个键，**没有** `AFFIX.Affix` —— 死键，Foundry 永远查不到 |
| `TOKEN.MOVEMENT.ACTIONS.{blink,displace}.description` | 2 | 上游只声明了这两个动作的 `.label`（`.description` 只有 `burrow` 有）。这是**核心 Foundry** 的键，属 `foundry_chn` 的辖区，不是 crucible 系统汉化该伸手的地方 |

**这条闸不该为了本轮的方便被放开。** 它的存在理由是 1.1.0 那次 `lang` 有 77% 静默失效
（第 1 节「发版前必做 `flatten_lang`」就是同一件事的另一半）。
「键不在上游 en.json」在绝大多数情况下正是「这条译文永远不会被读到」的同义词。

> ⚠ 附带订正一条会误导下一轮的 evidence：finding 说 notifications「29 条」，
> `lang` 单元实测**去重后 26 条唯一串**（37929==37984、37932==37987、38007==120588），
> 且其中 3 条是**跨行字符串拼接**，按字面加键救不了，必须用拼接后的整句当键。

---

## R2 · idx 264（triage 漏掉的那条）—— 走 schema 就地改写

Ember 三个 RegionBehavior 子类型的配置表单共 30 个裸英文串
（`ember.mjs:2554-2570 / 2685-2704 / 2765-2776`）。

**不走 lang 键**：这批串里 `Once` / `Locked` / `Discovered` / `Script` / `Material` /
`Grass` / `Metal` / `Stone` / `Water` / `Wood` 全是通用词，塞进全局 i18n 表以后，
**任何**模块 `localize("Water")` 都会拿到我们的译文。通用词比例比 `2026-08-14c` 否决的那 7 个还高。

**做法**：`init` 阶段从 `CONFIG.RegionBehavior.dataModels[<ember 的三个 type>]` 拿到类，
遍历 `schema.fields` 就地改写 `label` / `hint` / `choices`。作用域精确到 Ember 自己的三个子类型，
不碰全局 i18n 表，也不依赖表单 DOM 选择器。

---

## R3 · `Show Tracks`（idx 159 / 147 / 191 三处认领）—— 唯一实现方定为 `.mjs` 的场景控件钩子

三个单元都认领了这一条，按分工无人落地。裁决：**在源头改，不在 DOM 上改**，
判据必须写成「值等于 `Show Tracks` 才改」—— 这样上游哪天补了 i18n 键，
补丁会自动失效而不是把键名顶成中文。

---

## R4 · `Region Map` / `Area Map` 术语撞车

`hc-mjs-1` 报：`ArcturelElevatorTransit` 的 `DESTINATIONS`（`ember.mjs:96390-96392`）里
`Tradeway (Region Map)` 与 `Tradeway (Area Map)` **出现在同一个下拉框**，
而本库现有译文两者**都是「区域地图」**（lang 的 `EMBER.CALENDAR.REGION`、
`EMBER.BIOME/LOCATION.FIELDS.scenes.area`、合集里的 `Area Maps`、`.mjs` 的 PREFIXED 与 PATTERNS 各占一份）。
照现有译名会渲染出两个一模一样的按钮，**比留英文更糟**，所以该单元没落，是对的。

**裁决（两段）**：

**方向定为** `Region Map` → **地区地图**、`Area Map` → **区域地图**。
理由：改动面小的那边优先（§1 的既定原则）—— 英文闸实测 `Region Map` 命中 159 叶（全部现作「区域地图」），
`Area Map` 命中 268 叶（266 作「区域地图」，另有 2 叶已作「地区地图」）。改 Region 侧 161 叶，
改 Area 侧要 266 叶。且 Ember 的 region 是**大区**（世界六边格图）、area 是**局部场景**，中文语感相符。

**但全库拆分本轮判为「不做」。** 实测后否掉了自己上面的执行方案：

- `unify_terms.py` 的叶级英文闸在这里**不够用** —— 全库有 **46 叶英文同时含两个词**，
  而中文两边现在都是「区域地图」，整叶替换会把同一叶里的 Area Map 一起改错。
- 改用本项目在 `@UUID` 标签上验证过的**逐位对齐**（脚本 `probes/split_region_area_map.py`，已入库）：
  只有「英文出现数 == 中文出现数」时才逐位配对。实测 **115 叶可对齐 / 72 叶计数不等**。
  不等的成因是中文重复了术语而英文用了代词或 the map（例 `EN=1['Area'] CN=2`、
  `Region Exploration` 页 `EN=16 CN=17`）。
- **只落 115 叶比不落更糟**：那样库里会同时存在「Region Map＝地区地图」和「Region Map＝区域地图」
  两种写法 —— 正是 §8 `2026-08-13i` 那条教训（「单元制审计会把术语分叉切成两半，
  剩下的叶子里往往同段同时出现两种说法，比不改还糟」）。要做就得连那 72 叶一起人工过。

**本轮实际落地的是「局部消歧」**：真正会坏掉的只有那个同框下拉，
已在 `.mjs` 的 `DIALOG_UI` 里就地区分 6 条（贸易道/石底镇/阿克图瑞尔洞窟 的
地区地图 / 区域地图 / 远景），与上面定下的方向一致，将来全库拆分落地时不需要回改。
批次已生成后**主动删除未落盘**。

**留给下一轮**：72 叶逐条人工判 + 115 叶机械落盘，一起做。

---

## R5 · `Hexblade` 暂译「咒刃」与 `Spellblade` 撞名 —— **从表里删掉**

合集里 `Spellblade` → 「咒刃 Spellblade」（2 处，已定稿）。
`Hexblade` 上游 Warlock 页**已删该子职**，合集无对应译法，deity 数据里只剩 2 叶。

**裁决**：不译，从 `WARLOCK_PATRONS` 里删掉该条（显示「邪术师宗主：Hexblade」）。
理由：1:2 撞名会让玩家在两处看到同一个中文指两样东西，而这是 **dnd5e 侧**（附带项）
的一个**上游已删**的子职 —— 为它制造一处主线撞名不划算。

同批的 `Divine Domain` 神圣领域 / `Warlock Patron` 邪术师宗主 / `Sorcerous Origin` 术士起源
三个**前缀**译名：叶子值都有合集出处、只有前缀本身没有。**予以采纳**
（`Warlock`＝邪术师是 §8 `2026-08-13j` 的定稿，`Sorcerer`＝术士，两者已区分）。

---

## R6 · 「进程 / 进阶」与「阶位 / 等级」两组同屏分歧 —— 统一

- `Soulbound Progression` → **魂缚进阶**。`glossary_ec` 定稿即「进阶」；
  `DIALOG_TITLES` 里早前轮次写的 `"Apply Soulbound Progression": "应用魂缚进程"` **改为「进阶」**。
- `Rank` → **阶位**。`lang` 的 `EMBER.ATTUNEMENT.Rank` 已是「阶位」，
  `DIALOG_UI` 里写的「等级 1（次等魂印）」**改为「阶位 1」**。
  （注意与 `Tier`＝阶、以及角色 `level`＝等级 区分，三者不是一回事。）

---

## R7 · `[[/language moiré]]` 4 处 —— 走合集批次，不走 `.mjs`

`hc-mjs-2` 实测：上游增强器的 pattern 是 `(\w+)` **无 `u` 标志**，`é` 不算 `\w`，
所以这个增强器**根本不会被调用** —— `.mjs` 侧任何补丁都吃不到。
4 处（`ember.adventure.json` ×2、`ember.crucible-adventure.json` ×2）只能在 `compendium/cn` 里
把字面量换掉。

**裁决：本轮不做，留待下一轮。** 换掉字面量意味着把 `[[/language moiré]]` 整个标记删成纯文本，
这会让中英标记多重集不等 —— `apply_translations.py` 的第 3 道闸（标记破损）必然拒收，
而 `scan_markup_drift` 的 `INLINE` 也会从 0 变成 4。
要落地就得**同时**：① 绕过一道本来在保护我们的闸；② 给 `EXCLUSIONS` 加一条永久豁免。
为 4 处观感问题在一轮的收尾阶段做这两件事，性价比不成立。

现状对玩家的影响：这 4 处渲染出的是字面量 `[[/language moiré]]`（增强器不触发），
确实难看，但不影响功能。下一轮连同 R4 一起处理。

---

## R8 · `Kavir` 译名 —— 采纳「卡维尔 Kavir」

8 处悬空引用，`glossary_ec`、两个合集、英文基线**全库无任何定稿**。
`hc-mjs-2` 按音译暂定「卡维尔」。证据不足属 §1 的例外情形，
按音译落定并**写进本记录**（下一轮不要再当未决项翻出来）。

---

## R9 · dnd5e 侧的裸 `Advantage` / `Disadvantage`（1927 处）—— **不动**

`ember.mjs:22889` 在 dnd5e 分支产出这两个裸词。进全局 `EXACT` 会与 dnd5e 系统的中文模块撞车，
而这一侧是 §1 明定的**附带项**。维持「不得为 dnd5e 侧牺牲主线」的作用域裁决，记录备查。

---

## R10 · 日历条的天气 / 风向没翻译 —— `patchWeatherLabels` 一直在空转（已修）

**用户报的现象**：画面上方那条日历里，天气与风向的悬浮提示是英文。

**根因**：补丁走错了取值链，**改写 0 条**，而且以前不告警，所以一直没人发现。

真实链路（`EmberWeatherManager#getConfig()`，`ember.mjs:21825`）：

```js
ember.region.slices[<sliceId>].config.weather[<type>]
```

也就是 **`slice.config.weather`**。而补丁写的是 `slice?.weather` ——
那个键在 region slice 上**根本不存在**（`slice.weather` 只出现在 Vista 场景定义里，
而且只有 `{elevation: -50}`，没有任何 label）。于是 `Object.values(undefined ?? {})` 空转，
`n` 恒为 0，日志打「已改写 0 条」而没人看。

**渲染点**（两处都在 `#refreshWeather()`）：

- `:24656` `icon.dataset.tooltip = str?.label ?? cfg.label` —— 天气图标
- `:24673` `windArrow.dataset.tooltip = \`${windCfg.strengths[wind.strength]?.label} (${wind.speed} mph)\`` —— 风向箭头

后者是**拼接串**，所以补 i18n 键没有用（core 的 tooltip 管理器只对整串做 `game.i18n.has`），
只能改数据本身 —— 这也正是当初选择「改 config 数据」这条路的原因，路径写错才是问题。

**修法**：`config.weather` 与 `weather` 两条路都走（前者是当前形状，后者留作兜底），
并且**改写 0 条时告警**，不再静默。回测 `probes/weather_harness.mjs` **13/13**，
含两条反向断言（旧路径 `slice.weather` 不被误伤、不再静默空转）。

**词表本来就是全的**：上游 weather 配置块里 27 个唯一 label，除 `Surface of Ember`（slice 名，不属这一档）
外 26 个 `WEATHER` 表全部已有 —— 所以这次只改了一个函数，没动表。

> ⚠ **仍是英文的一小截**：风向提示里的 `(12 mph)`。单位与数字是在 `#refreshWeather()` 里
> 硬编码拼进模板串的，要中文化就得包 `#refreshWeather` 或整个 `EmberCalendarNavigation`，
> 影响面大于收益，本轮不做。天气名、风力档位名（无风/微风/有风/疾风/狂风）现在都是中文。

---

## R11 · hex 图上队伍 token 变透明 —— **与汉化插件无关**（已证）

**证明方式是查我们自己的代码，不是推测**：对两个模块的全部运行时代码
（`register.js` / `babele-mappings.js` / `ember-hardcoded-cn.mjs` / `babele-register.js`）
grep `alpha|hidden|elevation|occlu|renderFlags|refreshMesh|refreshState|\.mesh|visible =|TokenDocument|canvas.tokens`，
**命中 5 处，逐条看过，全部是被翻译的英文界面串或注释**：

| 命中 | 实际是什么 |
|---|---|
| `ember-hardcoded-cn.mjs:591` | 译文 `"Activate the hidden Generator Room switch?"` |
| `ember-hardcoded-cn.mjs:1061` / `register.js:87` | 译文 `"…lock its elevation so it can be moved vertically."` |
| 两个 `babele-mappings.js:213` | **注释**，说明转换器会原样保留 `_id/texture/x/y/elevation/rotation/flags/delta/disposition` |

**零处代码**碰 token 的 alpha / hidden / elevation / 遮挡 / 渲染标志。

另外把唯一一条「理论上可能相关」的通道也查了：我们确实翻 `Scene.levels[].name` 与 `Scene.tokens[].name`
（层级带 `bottom`/`top` 决定 token 遮挡）。babele 2.9.1 的 `nameCollection` 实现是
`fieldCollection`（`babele/script/converter/converters.js:99-114`），做的是
`foundry.utils.mergeObject(data, {name: translation, translated: true})` —— **合并**，
`bottom`/`top`/`_id` 结构上不可能被动到，只多一个 `translated: true` 标记。

**真正的来源在 Ember 自己**：hex 图上队伍 token 的外观由 Ember 的 `EmberToken` 子类接管
（`_refreshMesh` / `_refreshState` / `getParallaxPosition` / `emberDynamicToken`），
从 vista 切回 region 时走的是 `EmberRegionManager` 的初始化：
`_applyHexRules(ember.partyToken, …)` → `party.renderFlags.set({refreshState, refreshMesh})`
→ `party.applyRenderFlags()`（`ember.mjs:59636-59642`）。
队伍 token 在 hex 图上被放在 `elevation: -1`（`:17069` / `:17816`），
而 `canvas.tokens.occlusionMode = TOKEN_OCCLUSION_MODES.VISIBLE`（`:59577`）。
透明多半出在这条链上（层级/遮挡状态没被正确重置），属上游。

---

## R12 · idx 34/41 内容缺口**已补完**（encounterTokens 894 处等价物）

**管线侧**：英文基线用「**只并新路径、已有路径一律不动**」的方式重抽并入
（`merge_new_en_fields.py`，已入库）。这样 `LOCAL-PATCHES.md` 的四条上游笔误补丁
**结构上不可能被回退** —— 已逐条复验：`(Persuasion)` 已补 / `athletics 15]]` 已补 /
坏拼写 `C0nsortium` 不在 / `Prison` 页完整。
每包新增 245 条：**194 encounterTokens + 42 sounds + 9 regions**，无一条意外字段。

**翻译侧**：175 个唯一英文串，库内先例只有 17 条（9%）。6 路分片 + 1 路统稿。
落盘 **245 叶主线 + 245 叶孪生 = 490 叶**，三道闸零拒绝。
复验 `en == cn`：encounterTokens 194/194、sounds 46/46，两包皆是。
`fill_missing` 反方向复跑 **0 条**——缺口真的清了。

**统稿改了 17 条**（6 路各写各的，同族兄弟串不同构）：

| 组 | 裁决 | 依据 |
|---|---|---|
| Waterfall Exterior/Interior × 8 | 瀑布外部/内部 + 方位，**不加空格** | 6 路出了 5 种格式。库内同型先例 `Culvert Interior`＝涵洞内部 / `Exterior`＝涵洞外部 |
| Crowd Murmur × 3 | 场所前置：微光酒廊/悬崖边/聚归馆 + 人群低语 | 中文修饰语必须前置，「人群低语 悬崖边」不成词；`Glimmer` 补出「酒廊」（页 name 已定稿「微光酒廊」）|
| Ooze/Tar Bubbles × 4 | 一律「气泡」，**不加「声」** | 库内唯一 Bubbles 先例 `Tar Bubbles`＝焦油气泡 |
| Ambient Sound × 2 | 环境音效 / 环境音效（2），括注**全角** | 词根取 `Patch 0.5.1` 正文「新的环境音效放置」；库内 name 括注 81 条全是全角 |
| `X Triggered!` × 2 | 压力板触发！（删「已」）| 同句式必须同构 |
| Hallows Passerby A/C/D | 编号字母前留**半角空格** | 库内 name 惯例（`菌丝旷野底图 A`）|

**`Scout` 的处理与 §8 的 G15 豁免一致**：token 层取「斥候」（同卷正文 62:0），
原型层保持「侦察兵 Scout」。`same_en_split` 因此 `Scout` 组由 5 叶增至 7 叶 ——
**组数仍是 15，无新增组**，全部为已归档的合法分叉。

**统稿另查出 2 处库内既存矛盾**（非本批引入，留待下一轮）：
`Erisa Wandren` 埃丽莎/埃里萨（1:1）· `Chef` 主厨(3)/大厨(1)。

---

## R13 · `fill_missing.py` 的 `todo.*` 键形态错了（已修）

**症状**：6 路翻译 agent **每一路都先踩了一次**——照 PROJECT.md「批次 key ＝
待译清单里的 path」把 todo 填完直接喂 `apply_translations`，整批报 `REJECTED no-EN`。

**根因**：同一个函数里，`tm.*`（机器能填的那一半）走了 `to_batch_path()` 把
`entries.` 前缀剥掉，而 `todo.*`（留给人翻的那一半）**直接写了文档根路径**。
于是「两半键形态不一致」，而 `apply_translations` 的根是 `en['entries']`，
带前缀等于去找 `entries.entries.…`。

**这个拒绝理由特别会骗人**：`REJECTED no-EN` 字面意思是「英文里没有这个键」，
很容易被读成源数据的问题，而真正的原因是路径根对不上 ——
`to_batch_path()` 的 docstring 里其实早就写明了这一点，只是没用在 `todo` 这一支上。

**已修**：写 `todo.*` 时同样过 `to_batch_path()`，返回 None 的（顶层标量）跳过。

---

## R14 · R4 推翻重做：`Region Map`/`Area Map` 全库拆分**已完成**

**上一节（R4）的「本轮不做」结论是错的，因为它建立在一个错的测量上。**

`probes/split_region_area_map.py` 的英文侧正则写的是 `\b(Region|Area) Maps?\b` ——
**区分大小写**。而 Ember 英文正文里小写形态（`the region map`、`each new area map`、
`mega-region map`）非常多，于是英文侧计数系统性偏小，大批叶子被误判成「计数不等，需人判」。

改成 `re.IGNORECASE` + 允许连字符之后：

| | 旧（区分大小写）| 新（不敏感）|
|---|---|---|
| 可机械对齐 | 115 叶 | **169 叶** |
| 需人判 | **72 叶** | **12 叶**（6 唯一 ×2 孪生）|

**四路人判单元各自独立发现了这个问题并报了上来** —— 这是本轮最有价值的一次回报：
分片 agent 不只是执行者，它们撞到判据本身的毛病时会说话。

**落地**：机械 85 叶 + 人判 23 叶（重叠 19 叶中 18 叶结论一致）+ 主控补 4 叶 = **93 叶**，
孪生同步 94 叶。全批过守卫：每叶 `len(new) == len(old)` 且
`new.replace('地区地图','区域地图') == old.replace(...)` —— 除该术语外一个字符都没动。

**主控补的那 4 叶**是另一个盲区：脚本只数「区域地图」，**看不见已经写着「地区地图」的叶子**。
全库 5 叶已含「地区地图」，逐条查后 **4 叶是错的、1 叶是假阳性**：

- `Verdant Paths` / `Crystal Carving Cavern` / `Redrak Fields`：英文是 `area map`，中文却写「地区地图」→ 反了，改回
- `Ancient Paths / Emergence`：英文 `occurs on the region map without a specific area map`，中文两处**正好互换**→ 对调
- ⚠ `Ordain Gazetteer / Trident's Point`：**假阳性，绝不能动** ——
  中文是「一幅本**地区**+**地图**」（英文 `a map of the local region`），
  「地区地图」只是碰巧的子串。机械替换会把它改成「本区域地图」，意思全变

**完整性验证**（新写的方向验证器，逐位配对英文 kind 与中文词）：
**221 叶方向全对**，唯一报错的 `Patch 0.3.2` 是**验证器自己**的假阳性 ——
英文 `for vistas and area maps in addition to the region map`，中文把「除了…外」提前，
语序与英文相反但**两处都译对了**。这正是 u3 事先点名「逐位对齐会判反」的那一叶。
全库现状：地区地图 381 处 / 区域地图 628 处。

**方法论**：逐位对齐的已知失效形态是**中文把「in addition to X」提到句首**，
221 叶里出现 1 次。所以它可以当**筛子**，不能当**判据** —— 必须配人判兜底。

---

## R15 · moiré 换了个修法：不动 compendium，放宽我们自己的兜底正则

R7 判的是「改 compendium 字面量 → 必须绕标记闸 + 加永久豁免 → 不划算」。
但那个前提本身可以绕开：`.mjs` 里本来就有一条兜底 PATTERN 专门接「增强器没接管的
`[[/language …]]`」，而它的字符类**照抄了上游的 `\w`**，所以同样接不住 `é`。

改成 `/^\[\[\/language ([\p{L}\w-]+)\]\]$/u` 并把 `moiré: 莫伊雷语` 加进 `MISSING_LANGUAGES`：
**compendium 一个字都不用改**，不绕任何闸，不加任何豁免。
实测 `[[/language moiré]]` → 「语言：莫伊雷语」，borel/kost 不受影响。

⚠ 加 `u` 之后裸 `]` 是语法错误（Lone quantifier brackets），必须写 `\]\]`。
