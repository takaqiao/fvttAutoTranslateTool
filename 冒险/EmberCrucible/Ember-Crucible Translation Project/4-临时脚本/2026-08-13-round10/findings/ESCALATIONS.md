# 第十轮 A 段 escalate

## J04 — 《Gamemaster's Guide》（非 Patch 页）

两处需要主控裁决、我没有擅自改的：(1) Gameplay Overview 页的 @Embed 上游 typo — 英文侧本身写坏了（`@Embed[…hLp3EAST47aVpwVW]i inline]`），中文侧擅自改写成 `@Embed[…]{玩法概览}`。照抄英文＝把渲染故障也搬到中文，改成 `inline`＝翻译越权改标记参数，两条路都要一个明确授权。(2) The Hallows 的「幽圣所（正文 397）vs 圣堂区（name 字段 + 远景场景名）」正好撞上裁决阶梯的两条相反规则（name 优先 vs 改动面小 + 全库多数），跨多个单元，需要一次性定调再由某个单元统一执行。另：本单元 24 个 Patch 页按指令整体跳过，但 Patch 0.1.3 里有一处与我修的 Audio Engine 完全同源的 hex→妖术 错译，请确保 Z4 收到（历史上 J05 撞见的 Ordain Gazetteer 阻断就是这样漏掉一整轮的）。

## J06 —《Cosmos》寰宇（23 页 / 114 叶）

两件需要项目级裁决、我没有擅改的事：(1) glossary_ec 的 `Ascendancy → 优势地位 Ascendancy` 与 `Ascendants → 飞升者` 互相打架，而「优势地位」用来指「凡人吸收心之水晶成神」这个机制，中文读作「支配/优势的地位」，实义严重偏离；我本轮按「name 字段 + glossary」保留了「优势地位」做名词、「飞升」做动词，但建议整个项目把 Ascendancy 改成「飞升」或「登神」，那要连 Ascendancy.name 和 glossary 一起动。(2) glossary_ec 自身有错条目 `Signarans → 希格纳兰人 Signarans`，与同一份 glossary 里的 `Signaran Opal → 西格纳兰蛋白石` 以及项目已定译名「西格纳兰」矛盾；只要这条不修，以后任何自动套词都会把「希格纳兰」再塞回去。

## J11 —— 《Players' Guide》（玩家指南）

Token 的译法在 lang 层与 compendium 层是对立的，且这条对立会直接造成「指南教玩家点的菜单项在界面上不存在」——但修 lang/ 超出铁律 1，我不能写，必须主控裁决。

事实：
- `1-Ember汉化插件/lang/cn.json` 里 Token 一律作「指示物」，9/9 处，无一例外。关键三条：
  `ACTOR.CONTROLS.EmberToken` = 「余烬动态指示物」
  `EMBER.ACTOR_FLAGS.FIELDS.disableDynamicToken.label` = 「禁用动态指示物」
  `EMBER.ACTOR_FLAGS.FIELDS.disableDynamicToken.hint` = 「……用 Ember 动态指示物替换该角色的指示物……」
- compendium 层相反：含 Token 的 269 条英文里，中文用「令牌」202 条、用「指示物」31 条。Players' Guide 的页名也是「动态令牌 Dynamic Token」。

后果（实测）：Players' Guide 的 Dynamic Token 页最后一句英文是「click the three small vertical dots ... then select Ember Dynamic Token」，中文写「然后选择余烬动态令牌」。但玩家在汉化后的界面里看到的菜单项是 lang 层给出的「余烬动态指示物」。玩家照指南找「动态令牌」找不到。这属于跨层不一致导致的可操作性缺陷，任何单层的机械闸都查不出来。

我在本单元做的处理：按裁决阶梯（page `.name`「动态令牌 Dynamic Token」> 全库多数 202:31）把 Players' Guide 内部统一到「令牌」，顺带修掉了 10 处裸英文 Token 和 1 处「代币」。这只是让本单元自洽，没有解决跨层对立。

请主控裁决二选一，并统一执行：
(A) 全项目用「令牌」——改动面：lang/cn.json 9 处 + compendium 31 处「指示物」，约 40 处；但与 Foundry 核心中文（Token=指示物）不一致。
(B) 全项目用「指示物」——与 Foundry 核心一致、与玩家实际看到的界面一致；改动面：compendium 约 202 处（含本单元我刚统一的 19 处，需要回改）+ page name「动态令牌 Dynamic Token」。
另注：`EMBER.ACTOR_FLAGS.FIELDS.disableDynamicToken.hint` 里还留着两处裸「Ember」（「阻止 Ember 用 Ember 动态指示物……」），与 glossary_ec 定的「余烬 Ember」冲突，一并请主控在写 lang/ 时处理。

## Z5 — 同源串分叉的剩余 470 组

## A. 两组我不敢单方面拍板的

**1）`Hallows` —— 组织名 vs 城区名的系统性拆分（第 360 组，4 叶，但牵动 255 叶）**
英文只有一个 "the Hallows"（既是组织，也是它得名的那个城区）。中文被拆成了两个词：
- 组织：`Organizations.pages.Hallows.name`=「幽圣所 Hallows」，全库 **177 叶**用「幽圣所」
- 城区：`Ordain Gazetteer.pages.The Hallows.name`=「圣堂区 The Hallows」、`scenes.Vista: The Hallows.levels.Hallows`=「圣堂区」，全库 **78 叶**用「圣堂区」

这是**大规模、成体系、指称正确**的拆分（每一处都归对了组织/城区），所以我把第 360 组判为 legit、没有出批次。
**但是**组织页正文里那句「幽圣所主要从**与其同名的**圣堂区开展运作」（英文 eponymous）在中文里自相矛盾——「幽圣所」和「圣堂区」并不同名。
要么改那句话（最小改动，1 句 ×2 包，例如改成「幽圣所主要从与其同源得名的圣堂区开展运作」），要么放弃拆分统一成一个词（255 叶，代价极大）。**我建议只改那句话，请主控定夺。**

**2）`Senses` 文件夹（第 469 组，2 叶，1:1 平票）**
`crucible.adversary-talents.json folders.Senses`=「感知」 vs `crucible.affixes.json folders.Senses`=「感官」。同一种东西（感官类天赋/词缀的分组），必须统一，但信号互相打架：
- 支持「感知」：全库多数（`Gesture: Sense`=手势：感知 9 叶、`Sense Spellcraft`=感知施法）、glossary_ec 里 `Senses→感知`、`Sense→感知`
- 支持「感官」：glossary 已定 **`Wisdom`=感知**，在词缀包里「感知」这个文件夹紧挨着 `Skills`=技能 / `Defenses`=防御，玩家会读成「感知属性相关词缀」，与属性名撞车

改动面 1:1，没有 name 字段可依。**我倾向「感官」（避开 Wisdom 撞车），但这与 glossary 相反，所以交给主控。**

---

## B. 请写进 PROJECT.md 的 legit 清单 —— **字段排版约定表**

第九轮留下的 470 组里，**441 组 / 2351 叶的分叉不是分叉，是字段排版约定**。以后任何同源串分叉扫描都应先按下表扣除，不要再逐轮重裁。全部数字是本轮在全库统计出来的（分母为含中文的叶）：

| 字段类别 | 约定 | 实测 |
|---|---|---|
| `actors.*.tokenName` | **纯中文**（地图上不占两倍宽） | 557 : 18 |
| `*.adjective`（词缀前缀形） | **纯中文** | 172 : 0 |
| `scenes.*.levels.*`（楼层选择器） | **纯中文** | 509 : 8（8 例是「底图 A/B」的字母） |
| `scenes.*.regions.*.name`（地形区域） | **纯中文** | 1508 : 0 |
| `scenes.*.tokens.*`（场上 token） | **纯中文** | 14 : 0 |
| `*.outcomes.*.label` | **纯中文** | 548 : 0 |
| `*.condition`（动作触发条件） | **纯中文** | 253 : 3（3 例是整句） |
| `journals.*.categories.*` | ember 两包**纯中文**；`crucible.rules.json`**双语** | 542 : 0 / 0 : 10 |
| `folders.*` | ember.adventure / ember.character / ember.crucible-adventure / ember.crucible-adversary / ember.crucible-character **双语**；其余（ember.crucible-effects / ember.crucible-items / ember.dnd5e-effects / 全部 crucible.*）**纯中文** | 336 : 1 / 1 : 177 |
| `label`（包名） | ember 包用 **「中文 English」空格**；crucible 包用 **「中文\nEnglish」换行** | 9 / 15，各自 100% |
| `entries/actors/journals/pages/items/effects/results/taxonomy/archetype/actions .name` | **双语「中文 English」** | 均 ≥99.5% |

**只按这张表，470 组里 441 组（2351 叶）一次性判 legit。** 最常见的四种组合：`actors.name`(双语)+`tokenName`(纯中文) 192 组；`scenes.name`(双语)+`levels`(纯中文) 28 组；`entries.name`(双语)+`adjective`(纯中文) 22 组；`folders`(ember 双语)+`folders`(crucible 纯中文) 14 组。

### 另外 14 组是**语境不同**的真合法分叉（人工复核过，永久排除）

- `Shield` —— 施法者身上=护盾术法术 / 战士与战利品表=盾牌装备 / `crucible.talent.json folders.Shield`=盾牌（纯中文包）〔已裁过〕
- `Arcturian` —— 文化条目=阿克图里安 / `actors.Arcturian`=阿克图里安**人**（一个人）〔已裁过〕
- `Luminous` —— `ember.character.json`(dnd5e 侧)=辉耀 / `crucible.affixes.json`=明光，两包永不同载〔已裁过〕
- `Spirited` —— 同上，昂扬 / 精神焕发〔已裁过〕
- `Color Commentary` —— 壁画页=彩色解说 / 战场解说表结果=精彩解说〔已裁过〕
- **`Light`** —— 法术/物品 Light=「光 Light」（26 叶） / `crucible.equipment.json folders.Light`=「轻型」（护甲重量级，兄弟是 Heavy=重型、Medium=中型）
- **`Water`** —— `scenes.*.regions.Water.name`=「水域」（地形） / `folders.Water`=「水元素 Water」（元素生物文件夹，兄弟是火元素/气元素）
- **`Ooze`** —— `scenes.Verdant Paths.regions.Ooze.name`=「软泥」（地形物质） / `crucible.taxonomy/adversary-talents folders.Ooze`=「软泥怪」（生物类型）
- **`Aura`** —— `journals.Cosmos.pages.Aura`=「奥拉 Aura」（一颗月亮，别名空心之月） / `crucible.affixes.json Aura Spellcraft.adjective`=「灵气」（Crucible 手势，对应 `Gesture: Aura`=手势：灵气）
- **`West` / `East`** —— `journals.Steed's Point.categories.*`=「西部/东部」（地点的东西两半） / `tables.Change of Direction.results.*`=「西/东」（罗盘方位，兄弟是北/东北/东南）
- **`Rest`** —— `actions.*.condition`=「休息」（纯中文字段） / `crucible.macros.json entries.Rest.name`=「休息 Rest」（宏名，双语字段）
- **`Clarion Fork`** —— `outcomes.fork.label`=「谐鸣叉」（纯中文字段） / `items.Clarion Fork.name`=「谐鸣叉 Clarion Fork」
- **`Hallows`** —— 见上文 A-1，组织/城区拆分，指称正确
- **`<p>While [[lookup @name]] … his AC includes his Charisma modifier.</p>`** —— 英文把代词硬写成 his，中文按角色性别分成「他」（Brackus von Tet、Grim Assembly）与「她」（Zira Hestidero）。这是**语境驱动的合法分叉**，不要统一。

### 顺带记一笔（不是缺陷，只是提醒）
`辉耀` 同时被三个英文占用：`Lucent`（NPC 棋士构装体，ember 两包）、`Luminary`（Crucible 词缀）、`Luminous`（仅 ember.character.json 那一叶）。三处都带英文并列，屏幕上不歧义，且分属永不同载的包，所以本轮不动；但 glossary_ec 里 `Luminary→辉耀者`，与包里的「辉耀」对不上，哪天要清 glossary 时留意。

## J09 — 《Ancestries》血统

两条需要主控做全库级裁决、我不敢单方面动的分叉：

1. **Age of Sunlight = 阳光时代 还是 阳光年代**。全库 阳光年代 7 / 阳光时代 5，近乎五五开，且是同一个 UUID（o5o7MQGq9nVQlIaQ）的标签。我按裁决阶梯最强一档（History 里兄弟条目的 name 字段：野兽时代 / 高塔时代 / 重新发现时代，3/3 用「时代」）+ 同卷已译页（Ashka.text 用「阳光时代」），把我这单元的 2 处改成「阳光时代」。改完全库变成 时代 7 / 年代 5。**剩下 5 处「阳光年代」在 Arctus Plateau Gazetteer、Cosmos/Pathways、History/Age of the Tower，需要另派人扫。** 若主控裁定反过来，我这 2 处要回滚。

2. **Kessian = 凯西安 还是 凯西亚人**。name 字段（Cultures/Kessian.name）写「凯西安 Kessian」，但全库正文压倒性用「凯西亚人」（Ordain Gazetteer/Sunhaven 一页就有十几处）。历轮统计说 name 有 41% 概率是错的那一边，这次很可能就是。我这一单元 2 处跟的是 name 字段，**没有改**，等主控裁决后统一。若裁定为「凯西亚人」，则要同时改 name 字段 + 我这 2 处 + 大陆名保持「凯西亚」。

另外提请注意 out_of_scope 第 3 条：**Sorcerer 与 Warlock 的 name 字段都是「术士」**，这是 name 字段本身的硬冲突，任何以 name 为最高裁决依据的流程都会被它污染，建议优先修。

## J07 —— 《Character Classes》(entries.Ember

【必须由主控统一裁决、我没有单方面改】

1. Warlock 与 Sorcerer 全库同为「术士」——这是本单元查到的最严重问题，但改名面远超我的单元。
   证据：Sorcerer.contentOverview 的原句译完变成「术士不必像术士那样缔约；术士的起源驱动术士」，字面自相矛盾；
   Classes Overview 的核心职业清单里连着两个纯文字相同的「术士」链接；两页的 name 字段是「术士 Sorcerer」「术士 Warlock」；
   glossary_ec.json 里 Warlock 和 Sorcerer 两条的值也一模一样。
   改动面：ember.crucible-adventure.json 里含 Warlock 的叶子 78 个（其中 51 个在本单元内，27 个在 Deities / A Brush With Death /
   Unfinished Business / The Bleak Archive / Kalion Stadium Underworks / Ordain Gazetteer / 多个 actor biography 里），
   孪生包同数，再加 lang/cn.json 三条 UI 字段和 glossary_ec.json 两条。
   建议：Warlock →「邪术师」（5e 中文通行译名），Sorcerer 保留「术士」。要么整体一次改完，要么全都别改——半改比不改更坏。
   我本批次只做了两处最小消歧（Sorcerer.contentOverview 加英文括注；Classes Overview 的两条链接标签改成与各自 name 字段一致的
   「术士 Sorcerer」「术士 Warlock」），如果主控决定整体改名，这两处请一并覆盖掉。

2. Arcageris =「奥术巨龙」是凭空造义（Arc- 词根在同段的「弧行者」「御弧宗师」里都保留着），共 3 个叶子：
   Organizations.pages.The Arcageris.name（同轮 J10 批次已改为「阿卡杰里斯」）、本单元 Monk.text（我已按 J10 的写法改为
   「阿卡杰里斯 Arcageris」）、Notable Figures.pages.Viola Key.contentOverview（无人认领）。
   请确认 J10 的批次会落地，并把第三处一起改掉，否则会留下一半「阿卡杰里斯」一半「奥术巨龙」。

3. 两条全局替换伤，本轮各单元都会撞到，建议由主控出一次性清单而不是各自为战：
   (a) 「路径」→「径」：本单元 2 处、全库另约 29 处（见 out_of_scope）。
   (b) 英文普通词 boon →「恩惠骰」：本单元 5 处、全库另 5+ 处。两者都不能再用整词替换修，必须逐条判是机制术语还是普通词。

4. 工具环境提醒：scratchpad 目录是各 agent 共用的，我写在
   ...\scratchpad\mkbatch.py 的建批脚本在运行途中被另一个 agent（J10 Organizations）用同名文件覆盖了。
   我的批次 JSON 已先落盘所以没受影响，但后续轮次建议给 scratchpad 脚本名加单元前缀。

## J10 —《Organizations》（ember.crucible-adve

两条需要主控拍板的跨单元决定：

1. **Otherhood of Fortune 的名字（最大的一条）**。全库「异缘会」186 处 / 约 50 叶，「幸运异姊会」44 处 / 29 叶，而页面 `.name` 与 folder 名都是「幸运异姊会」。我按裁决阶梯（name 字段 > 全库多数）把本单元 3 叶统一到「异姊会」，但这意味着界外约 47 叶必须跟着改，否则分叉从「页内矛盾」变成「跨页矛盾」。如果主控反过来决定统一到「异缘会」，那需要同时改掉页面 name 与 folder name（4 处），我这 3 叶的改动作废即可，无害。**请尽快派单，不要重演 Ordain Gazetteer 那次「出界所以整轮没人管」。**

2. **The Arcageris → 阿卡杰里斯**。这是我新拟的音译，因为原译「奥术巨龙」是硬伤（把武僧修道院译成龙），全库 3 处都错、没有可用的正确多数派，glossary_ec 里那条也是错的。我只改了本单元的 page name，另 2 处（Viola Key、Monk）出界。如果主控更倾向意译（如「弧宗」「引弧会」），请统一定夺后一次性覆盖这 3 处 + glossary。

另外两点提醒：
- `Cindaric Sages.text` 里 Vinarith 的 NE 我判为**阻断**，理由不只是阵营错，而是它出现在玩家可见的 `text` 字段，把只写在 GM secret 块里的主线反转提前透给了玩家。建议主控把「玩家可见字段里出现只应在 GM 块中的信息」单独立为一类判据去全库扫。
- 本单元的 `<dt>` 缺陷（3 处括号被整段删、5 处括号全英文）与简报里 Ordain Gazetteer 的模式完全同型，说明这不是单本的问题。建议做一个专用扫描：EN 侧 `<dt>` 内匹配 `\)$` 或 `\([A-Z]{1,2}[NGEL]?,` 的条目，比对 CN 侧同位置是否也有括号、括号内是否含 CJK。这类缺陷现有五道闸门一道都不响。

## J03 —《The Winding Trail》（Ember Early Acc

两点需要主控决策：

1) **Firebug's Leather 的 item name 必须与我的修改同轮落地**（见 out_of_scope 第 1 条）。历史上 J05 撞见的 Ordain Gazetteer 阻断因为超出 scope 整整一轮没人修——这条同理，而且比那更急：我已把 Dusktide Rising.text 的链接标签改成 {纵火虫的皮革}，如果那 4 个 name 叶子不同步改，就会从「一致但错」变成「不一致」。要么主控派人改这 4 个叶子，要么把我这条 revert 掉。我按「改动面小的那边优先」原则本可以直接把 4 个 name 一起写进批次（它们在同两个包文件里），但那越出了我的单元、可能与别的 agent 的批次撞车，所以留给主控裁决。

2) **`问道自然` 是全库级的系统性误译**：把普通动词 commune 直接换成 D&D 法术名。crucible 包 8 个叶子，我修了单元内的 2 个，剩 6 个在单元外。这种「用法术名顶替普通词」的模式（同类还有本轮抓到的 daylight→昼明术）机械判据完全看不见，建议下一轮专门起一个横切扫描：拿 D&D/Crucible 法术名中文表去扫中文侧，凡是英文侧没有对应 @UUID/@Spell/法术名的，都是嫌疑。本单元 471,906 字符里就抓到 2 个不同的法术名顶替，密度不低。

## Z4 —— Gamemaster's Guide 下 24 个 Patch 0.

## 一、判定为「不译」的英文串（分类 + 理由，下一轮不必重裁）

裁定依据：**凡是没有出现在 `1-Ember汉化插件/lang/en.json`（486 条）里的 UI 串，玩家在界面上看到的就是英文**，补丁说明照抄英文才对得上；出现在 lang 里的（已译成中文的）必须跟着译。据此：

**A. 界面/API 上确实显示英文，保留（已 grep 全模块 scripts/、lang/、babele-mappings.js，均无这些 key）**
- `Preserve World State`（0.2.1、0.4.3）—— 冒险重导入选项，不在 Ember lang 里
- `Gazetteer Location Journal Entries`（0.4.3）—— 游戏设置名，Ember lang 无 SETTINGS 段
- `Pull to Scene`（0.5.3）—— 已并入 Foundry core 14.362 的钩子名
- `Adaptive Attenuation` / `Natural Attenuation`（0.3.1）—— 原文明说将成为 Foundry VTT V14 核心选项
- 枚举/常量：`ALL/ANY/NONE`（0.3.1）、`SOME/ANY`（0.5.2）、`FRIENDLY/HOSTILE`（0.5.2）、`SECRET`（0.5.5）、`NEVER`（0.6.0）、`` `Impossible` ``/`` `Unknown` ``（0.4.0，原文即用反引号标为标识符）、`false`（0.5.0）
- `trained`（0.4.3）—— Crucible 技能训练等级标识符；Crucible 中文档里未找到确定译法，**留给懂 Crucible 术语表的单元裁**（建议「已训练」）

**B. 外部专名 / 第三方产品 / 真实人名，保留**
- `Jess Levine` / `jessfrom.online` / `Jumpgate Games` / `going rogue 2e` / `I Have the High Ground` / `PLANET FIST` / `Bluesky` / `Shawandasse Tula, Osage, Monongahela`（0.4.7）
- `Caeora`（0.4.6）—— 真实美术总监（Players' Guide 制作名单里同样保留英文）
- `Raxray`（0.4.3）—— 社区贡献者 ID
- `Forge of the Artificer` / `Tasha's Cauldron of Everything`（0.4.2）、`Ravenloft: The Horrors Within`（0.5.5）—— 第三方 D&D 模组/书名
- `PAX Unplugged` / `Kickstarter` / `CRIT Awards`（0.3.0、0.4.7）
- `Foundry VTT` / `Foundry Virtual Tabletop` / `D&D5E` / `Crucible` / `Ember` / `Alpha One` / `Beta One` / `Ember Beta Two`

**C. 系统标识符 / 代码，保留**
`EmberVistaToken`、`VistaConfig`、`EmberDynamicTokenConfig`、`EmberSceneManager`、`EmberEventsLayer`、`EmberEventsControl`、`EmberCalendarNavigation`、`EmberRegionTokenRuler`、`EmberNarrativeEvent#spawnEncounter`、`EmberAreaMap#getActorSpawns`、`EmberBeamLightSource`、`MagicalPlatformShader`、`WaterSamplerShader`、`EmberPlatform`、`EmberElevator`、`AreaEffectRegionBehavior`、`EmberEventEncounter`、`EmberActorFlagData`、`combatTheme`、`ember.partyToken`、`ember.api.actors.combineGroups`、`splitGroups API`、`CONFIG.Canvas.fogManager`、`CONFIG.Token.objectClass`、`CalendarData.format`、`dnd5e.registry.classes`、`discoveryHexes`、`lunarAmbience`、`spriteOptions`、`routePrefix`、`sceneClip`、`physicalVvista`、`geometryMeshes`、`particleEmitters`、`FILES_UPLOAD`、`KTX2/UASTC/WEBP/PIXI/GPU/SFX/CSS/HTML/style/span/spritesheet/codemirror/TODO/POI/HUD/UI/UX`

**D. 新拟译名（库内无既有译名，下一轮如要改请一并改这些位置）**
| 英文 | 新拟 | 出现处 |
|---|---|---|
| The Boy Who Played With Boats | 玩船的男孩 | 0.3.2、0.4.0 |
| Impassable Ground | 不可通行地面 | 0.1.3 |
| Water Rising / Flood | 水位上涨 / 洪水 | 0.3.1 |
| Seydiri | 塞迪里 | 0.3.0 |
| Generic Dungeon | 通用地下城 | 0.4.5 |
| Solemn Folk | 肃穆民谣 | 0.4.5 |
| Quarry Mutagen | 采石场诱变剂 | 0.4.5 |
| Mixed Ancestries | 混合血统 | 0.4.2 |
| First Soulmark | 首个魂印 | 0.4.3 |
| Vertical Hand | 竖握手型 | 0.4.3 |
| Magical Forces | 魔法之力 | 0.4.2 |

**E. 一处需要主控点头的取舍**
`Ember Actor Flags`（0.4.6、0.4.7、0.5.2×2）我译成 **「Ember 角色标记」**，而 lang/cn.json 的 `ACTOR.CONTROLS.EmberFlags` 是 **「余烬角色标记」**。选前者是为了跟同一句里既有的「Ember 动态令牌」并列（该处 lang 又作「余烬动态指示物」）。**根因是 `Ember` 在 lang 里译「余烬」、在 compendium 补丁页里保留「Ember」**，需要一次全局裁决；裁完这 4 处要跟着走。

## 二、本轮**故意未动**、需要项目级决定的两组
1. **Token = 令牌/代币/棋子/Token（+lang 的「指示物」）**：24 页里四分，共 21 处非「令牌」写法（0.2.2×1、0.3.0×2、0.3.1×4、0.3.3×5、0.4.2×4、0.4.3×5）。改动面大且方向未定，只报不改。
2. **Levels = 等级/楼层/层级**：0.4.7 全页「等级」（11 处，且与 Rank=等级 撞车）、0.5.0「楼层」（4 处）、0.5.1 起「层级」（30+ 处）。建议统一到「层级」，但同样属项目级决定。

## 三、把 Patch 0.4.7 的判例记下来
2026-08-13e 的教训在本轮复核成立：0.4.7 那 15 个英文串里，`Jess Levine`/`going rogue 2e`/`PLANET FIST`/`I Have the High Ground`/`Bluesky` 等**本来就不该译**，真正该补的只有 `<h3>Actors</h3>` 和 `Ember Actor Flags` 两处。**该页的中英比例判据永远算不准，别再用它。**

## J01 —《Deities》神祇设定集

两点需要主控裁一下。(1) 成员表名字列：本书五张万神殿成员表（Auris Bor / Sentina / Solaru / Sunalin / The Tanir）里，凡是没有 @UUID 链接的神名全部保留英文，而同叶正文里这些名字大多已有中文。我按「同叶正文已译」为准补了正文里出现过的（沃伦、赫律默、艾伦、瓦罗斯、阿斯特里克、吉尔达蕾、内里萨、埃里萨戈萨、奥里宗、维斯塔埃克、凯娅、奥林、奥尔西亚、埃塔拉、科尔瓦克、诺杜罗），并为正文里从未出现、只在表中露过一次的少数名字新造了音译（皮里欧萨/皮里欧萨萨、维诺苏尔、克纳斯、苏茨、阿斯卡林、莫雷利奥、狂野者莫斯贾克、火匣 Firex、鲁恩弗 Ruenfoe、孔顿德鲁姆、棱光结界、暴虐血魔）。这批新造音译没有任何既有依据，如果项目有统一的造名规矩或想留给别的单元处理，请回退这十几条。Solaru 表里还剩 Kestra'sul、Khasu、Virim、Thunderis、Vae'oris the Gilded、Elar'vai、Valen the Fair、The Moon Wraith、Thalric the Blind、Kalaru、Red Sulina、Nexaaris、Mor'festara、Solcastra the Worm、Lifestealer 等约 15 个纯表内名字我没有动，等裁决后可一次性补齐。(2) Wyrm 归一：我按 name 字段「古龙 Wyrms」把 Alar 页 4 处「龙蛇」、Finor 与 Auris Bor 各 1 处「巨龙」改成「古龙」。这动了 6 处，若主控认为 Wyrm 更该用「龙蛇」（改 name 字段那一边），需要反向回滚，但那会波及全库 14 处「古龙」以及其他单元。

## J05 —《Cultures》（entries.Ember Early Acce

三件事需要主控裁决，我在批次里已按最小改动+多数派做了选择，但都可能被推翻：

1) **Arcden 定名：奥克登 还是 阿克登？** term_gate 全库 18 行：奥克登 14 叶、阿克登 4 叶。我按「多数派 + 改动面小」把我单元里的 4 处（Languages.text 共 5 个词）改成了**奥克登语**，这样全库只剩 Ordain Gazetteer/All-Fable Keep 一处（见 out_of_scope）需要跟改。**反对意见成立的可能性不低**：Arc- 词根在本项目其余专名里一律是「阿克」（阿克图里安 Arcturian、阿克图斯高原 Arctus），奥克登破坏了词根一致性，而且 Arcden 正是阿克图里安人的语言。若主控认为词根一致优先，正确做法是反向拉平——把那 14 叶改成阿克登语，我这一批的 Languages.text 需回退。请给一个定论再让下游动。

2) **我在本批里新造了 10 个中文名，需要过目**（都按「中文 English」双语并列，英文保留）：碎光者 Lightbreakers、铁霜圣堂武士 Ironfrost Templars、马基纳林 Machinarims、谢宁 Shenin、艾卡特 Ekat、米西亚语 Mithia、卢玛语 Luma、钱币语 Coins、尤克鱼 Jurk Fish、大地懒 Megatheria。term_gate 确认这些词全库原本没有任何中文对应，所以不存在「与既有译名分叉」的风险，但它们会成为既定译名，建议进 glossary_ec。其中 **卢玛语 Luma / 钱币语 Coins 是修阻断与严重缺陷所必须**（不造词就没法把 Luma 从「龙语」上摘下来、也没法把 Common/Coins 拆开），另外 8 个是可退让项。
   另有三个我**故意没造**、只报不改：Railen 页的三位碎片神 Szure / Tazarl / Lucon（有名有姓的现役神祇，全裸英文，但本书的人名一律不译，我不想单方面破例）、Waerd 页的 Euntwé 与 Kinthyr、Maziran 页的 Jagkat/Tekrel。请示下是否要一并造名。

3) **固定小标题与 At a Glance 字段名需要一张全项目对照表。** 本书 30 页里同一批标题有 2–3 种写法（饮食与膳食/菜肴/食谱、服装与装饰/饰物、习俗与传统/风俗与传统、大城市/伟大城市、首都/首都城市、名字/名称/姓名、技术/科技）。我只在 Maziran、Varún 两页把**完全未译**的英文标签补成了与多数派一致的写法，没有去动那些「已译但用词不同」的，因为一旦各单元各自拉平就会互相打架。建议主控出一张 12 行左右的标题对照表，再由一个人跨全库一次性刷。

## J02 —《Arctus Plateau Gazetteer》阿克图斯高原地名志

两点需要主控拍板：
（1）**代词标记的处理方向**。本包 66 个 `she/her`-类代词标记里 55 个保留英文原样（本子 44 + Forest of Stone Gazetteer 11），11 个被译成中文，且这 11 个全在本子（Brevin 5 / Rortwark 4 / Brimtown 1 / Rock Bottom 1）。我按「多数派 + 改动面小」把这 11 个还原成英文原样，顺带消掉了 Rortwark 那处 she/her→「他/他」的性别错。若主控的既定方针相反（要求代词一律汉化），那就该反过来改另外 55 处，届时请回退我批次里的这一组，我不再自行改向。
（2）**新拟译名**。批次里有 5 个专名在全包找不到任何中文先例，是我按构词法新拟并采用「中文 English」双语并列形式的：断尾蜥 Docktail Lizard、烬籽藤 Cinderseed Ivy、谷地巨蜥 Vale Monitor、洛多克塞罗斯 Lodoxeros、血奔兽 bloodrunners（另外「编织体 Woven」是从既有的「编织构装体」回推的）。它们原本都是裸英文夹在中文句子里，不动不行；但新拟名请复核，若 glossary_ec 里另有定译请直接覆盖。其余所有译名改动都有全包多数或 name 字段做依据，未凭空创造。

## J08 —《History》（Ember Early Access · jour

需要主控做三件事，都超出单本能解决的范围：

1) **Wyrms=古龙 / Dragons=巨龙 的全库分离**。这是所有条目里唯一会让设定读不通的：Abyssal Shear 与 Forsaken War 明确讲「阿拉尔把垂死的古龙 Wyrms 改造成最初的巨龙 Dragons」，而全库 46 行把 Wyrm 也叫「巨龙」。两个 name 字段（Bestiary.pages.Wyrms.name=古龙、Bestiary.pages.Dragons.name=巨龙）已经给了裁决，只差执行；多数派是错的那一边，机械闸永远查不出来。建议开一个专项 term pass。

2) **Sunfire Empire 的统一裁决**。6 种译名、无 name 字段、最大簇只占 39%。我在 History 内按最大簇统一到「阳炎帝国」，如果主控另有定夺，全库替换时把 History 一并覆盖即可，不会有冲突。同类还有 Inner Realms（4 种）、Sun Tower（3 种）。

3) **专名「地图标签 vs 正文」体系性对不上**。本轮撞见三例：Yarino（地图雅里诺 / 正文亚里诺）、Wick（地图烛芯 / 首都清单灯芯 / 正文威克）、SAURVEK（地图绍维克 / 正文索维克）。scenes.*.notes.* 是玩家在世界地图上直接看到的字符串，跟正文不一致等于地名志错位的轻量版。建议单独跑一遍「scene notes ↔ 正文」对账，这一类目前没有任何闸门覆盖。

另外提请注意本单元发现的**闸门盲区**：单个英文词（hush / Jarn / Akon / Crown / Luen）留在中文里，裸英文专名闸因为要求 ≥2 词而全部放行；Age of Rediscovery.contentGamemaster 那条时间线 28 个专名未译、叶内中文占比却很高，覆盖率闸也放行。如果要加闸，建议加一条「中文侧出现首字母大写的英文单词且该词在英文侧存在」的检测，白名单放行 AS/AC/AB/AT 与括号内的英文注解。

## J00 — 《Ordain Gazetteer》(ember.crucible-

两处需要主控拍板方向，因为改动面大且我选的是「name 字段那一边」而不是「多数派那一边」：

(1) **The Hallows：我统一成「圣堂区」，改了 23 叶次跨 14 页。** 依据是裁决阶梯里 name 字段（「圣堂区 The Hallows」）+ 同卷已译页（它自己那一页 12 次全是圣堂区）压过全库多数（幽圣所 23）。若按「改动面小的那边优先」则应反向（改 13 处成幽圣所）。我认为「幽圣所」的「幽」在英文里无出处、「所」也不是城区后缀，且 Ordain Spires 一页内两者并存已经是硬伤，所以选了圣堂区 —— 但这一条如果复核方向相反，只需把批次里的 23 处替换反转即可，其余 31 叶不受影响。

(2) **Highgate 海门 → 高门 改动了 name 字段本身**（并波及 glossary_ec）。name 字段是阶梯最强一档，我推翻它的依据是英文内证三连：\"gate district connecting Ordain to the Redrak Fields\"、\"a proud northern portal to the Redrak Fields\"、Lastgate 的 \"the austerity of Highgate to the north\" —— 它是通往内陆的北方陆门，和海无关，「海」只可能是 High 的听写误差。请复核者确认这一推翻是否被接受。

另需提醒：**本单元的所有修改在孪生包 ember.adventure.json 里逐字同源**（已验证 Gazetteer 223 叶在两包中英文 100% 一致），两个批次内容完全相同，必须一起落，否则 sync_twin_packs 会报。
