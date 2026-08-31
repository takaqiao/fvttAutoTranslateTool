# 第八轮：需项目所有者/主控裁决的事项

## g00

1) 【必裁·知识领域 7 词方向】三通道现状：compendium 表格 遗物/犯罪/法证学/权谋/传说/机械/不死 · lang/cn.json 的 KNOWLEDGE.* 与 scripts/ember-hardcoded-cn.mjs 的 KNOWLEDGE（两者逐条一致）神器/罪行/法医学/阴谋/传奇/机械装置/亡灵化 · glossary_ec 的 `knowledge X` 键 古器物/犯罪/法医/权谋/传说/机械/亡灵。**订正探针的错误前提：glossary_ec 不是没有这 7 个词，而是有 26 个 `knowledge X` 键**，它在 crime/intrigue/legends/machines 上站表格、在 artifacts/forensics/undeath 上两边都不站（它自己还在 cosmology=宇宙论、ancients=古代、dragons=龙类、outsiders=异界生物 上跟两边都冲突，是一份陈旧遗留表，只能当第四档弱证据）。逐词硬证据：Undeath —— 英文 \bUndeath\b 在 enricher 之外命中 27 叶，中文「不死」25 叶、「亡灵化」0 叶，全库「亡灵化」只有 4 叶（Cultures/Ossarchate 与 History/The Evernight 两对孪生），同卷散文（阶梯第 2 档）压倒性支持表格；Intrigue —— 同卷 ember.crucible-character 把 intrigue 译「权谋」（Cosmopolitan Fashionista）、把 conspiracy 译「阴谋」（Nightwatch），lang 的「阴谋」会跟 conspiracy 撞车；Crime / Legends / Machines —— 表格＋glossary 两票对 lang 一票；Artifacts —— 三方三个词，最乱；Forensics —— 唯一方向清楚的一条，「法证学」全库只在这张表 2 叶、「法医学」全库 0 叶，建议按 lang。渲染面：[[/knowledge X]] 全库实测 2316 处（不是 2175），这 7 个词占 833 处（rituals 177 / intrigue 153 / forensics 151 / machines 143 / crime 117 / artifacts 105 / legends 97 / undeath 67）。改动成本：表格侧 14 格（7×孪生 2 包），lang 侧 7 键 + mjs 7 键。**lang/ 与 scripts/ 都在我的禁写范围，无论往哪边裁都必须主控落手。**

2) 【顺带发现，不在我 findings 里、我没出批次】同属 Cold 术语分叉、英文闸抓得到但没进 g00 列表的另外 2 叶：(a) 2-Crucible/crucible.summons.json :: Frost Visitor.items.Cold Absorption.description 写「<strong>冰寒</strong>抗性」，而该条目自己的 name 是「寒冷吸收 Cold Absorption」—— 裁决阶梯第 1 档（同名条目 name）直接打脸描述，是最干净的一条；(b) 2-Crucible/crucible.affixes.json :: Cold Conversion.description 两处「<strong>冰寒</strong>伤害」，但这一条是三方冲突（name 与 glossary 都写「冷能转换」、lang DAMAGE.Cold 写「寒冷」、描述写「冰寒」），我不敢单方面定，建议跟词缀组一起裁。

3) 【管线】pair_dump.py / term_gate.py 的 --repo 走 os.path.join(repo,'compendium','en')，用 Git Bash 传中文目录名会被字符集打烂（我第一次跑就报 FileNotFoundError: '2-Crucible�������\compendium\en'，看起来像「包不存在」）。本轮我所有脚本改用 PowerShell + 绝对路径 + PYTHONIOENCODING=utf-8 跑通。建议把这一条写进后续 agent 的说明，否则容易把编码故障误判成数据缺失 —— 跟主控这一轮踩的 glob 反斜杠那个坑同类。

## g01

四件需要主控裁决/分派，我都没动：

1. **crucible.affixes.json 的 `等级 3` —— 这是术语缺陷，不是排版，且不在 g01 findings 里。**
   `Cold / Corruption / Electricity / Void Conversion.description` 四条，同一个叶子里英文
   `<strong>Tier 1:</strong>` / `<strong>Tier 2:</strong>` / `<strong>Tier 3:</strong>` 三行完全同构，
   中文却是 `<strong>第1阶：</strong>` / `<strong>第2阶：</strong>` / `<strong>等级 3:</strong>` ——
   第三行既把 `Tier` 从「阶」改成了「等级」（违背既定译名 Tier=阶/阶数，且同叶上文就写着「取决于此前缀的<strong>阶</strong>」），
   又用了半角冒号。我落盘模拟后全库仅剩的 4 处半角中文 strong 标签就是这 4 处。
   建议派给做术语的那一组，或主控直接裁 `第3阶：`。

2. **Lightning Strikes Twice 与 True Vault 是同一类缺陷，但探针只报了后者。**
   `Ember Early Access.journals.The Expedition Challenge.pages.Lightning Strikes Twice.text`（孪生 ×2）
   中文朗读段落结尾挂着一个**孤立的右 ASCII 引号**：`…才让我们做到这一点。"`（前面没有对应的开引号，
   同叶另一处 `\"Residential Corridor\"` 已经译成 `“住宅区走廊”`）。
   在我的 blockquote 构造闸里，全库 349 全角 : 4 ASCII，这 4 处正是 True Vault ×2 + 它 ×2。
   我按「存活的才出批次、不自己加 finding」的规矩没写进批次，但它和我改掉的 True Vault 是同一处理，
   留着下一轮还会被扫出来。建议主控直接补进某个批次。

3. **Brackus / Helice 的「短语 N」残留半个分歧。**
   我把 Brackus 的引号改成了 `“短语 1。”`，Helice 那边是 `“短语1。”` —— 引号已对齐，
   中文与数字之间的空格仍不同。没动是因为这一维度全库 5303:5710 无多数（见 verdict 里那条推翻）。
   若主控想彻底对齐，方向应是 Brackus 去空格（改动面 1 个叶子 3 处），不要反过来给 Helice 加空格。

4. **V'Mar 的译名**（详见 verdicts 里那条）：8 个同类音译一律丢撇号，只有它留了弯撇 `’` 且保留裸拉丁 `V`。
   需要一个译名裁决而不是排版修复，我没出批次。

另外记一条给后续 agent 的坑（和主控这轮踩的 glob 反斜杠是同一类「校验脚本自己是错的」）：
**数中文引号时如果不先剥掉 HTML 标签，`class=\"block gamemaster\"` 这类属性引号会把统计彻底带偏。**
我第一次跑出来是「1485 叶英文带 ASCII 引号，4196 叶中文仍保留」这种自相矛盾的数（保留数大于总数），
剥掉 `<[^>]*>` 与 `@X[...]`/`[[...]]` 之后才是真实的 1485 : 44。探针这次是对的，但它的转述里
「只剩 6 叶没转」也不准 —— 实际 44 叶，其中 28 叶是身高栏的英寸符（`4'4\" to 6'8\"`）、
6 叶是双语名的英文半边、2 叶是 Credits 里的英文绰号，都不是缺陷。

## g02

无跨单元冲突, 无管线问题。两条供主控参考的旁支观察(均在我的判据下不构成缺陷, 未出批次, 也不建议本轮动):

1. Grim Assembly 的人称。ember.crucible-adventure.json :: Ember Early Access.actors.Grim Assembly.items.Eldritch Defense.description 中文作「他的 AC 包含他的魅力调整值」。这忠实于英文模板的 his, 按判据1 不算缺陷; 但英文散文里 Grim Assembly 一律是 it/its, 是 'a flesh-warped creature of the Abyss' 这个非人造物, 实体正确的中文应是「它的」。属于「英文源自己有模板瑕疵」的类别, 要不要顺着英文走由主控定, 我倾向不动(改动面小的那边优先, 且英文源就是这么写的)。

2. 专名双语并列格式。本轮我改到的三个叶子里, 卡琳·卡里塞特 / 忒弥斯 / 阿拉尔 都是纯中文, 没按既定的「中文 English」双语并列写。这是格式类问题, 不在 g02 的判据范围内, 而且属于全库量级(涉及面远超这 3 个叶子), 我没有顺手改 —— 如果主控要统一, 应当交给专做双语并列的那一路统一处理, 免得跟本批次在同一叶子上撞车(Wardcall.text 与 The Abyss.contentGamemaster 两个叶子会同时被两路碰到, 落盘时需要三方合并)。

## g03

1) **合并冲突预警**：`crucible.playtest.json` 的 `…Day Five - Void Harbingers.text` 一叶同时承载了本组三条 finding（2/4/6），我出的是**合并后的单个新叶值**；若别的组也对这一叶出批次，必须三方合并、不能覆盖。同理 `Kern / Gesture: Aura.description` 一叶同时承载 finding 3 与 5。

2) **手势模板需要一次专门的统稿（本轮我故意没做）**：`Gesture: Aura/Blast/Cone/Pulse/Surge/Conjure` 的 description 在四个包里有 61 份英文逐字相同的副本，中文却各写各的 —— 同一句 `scales with` 有 增强/缩放/扩展/变化/成长 五种，`Token` 有 指示物/令牌/Token 三种，`emanation` 有 放射区域/散发区域/扩散区域/放射范围 四种，`Maintained` 有 维持/被维持 两种。没有任何一份具备权威性（`crucible.talent.json` 那份自己也不干净：`基础消耗为5动作和1专注` 两个粗体全丢）。这属于模板去重统稿，不是错位类缺陷，建议主控单独立项，不要塞进本轮。

3) **本轮顺手发现、超出 g03 判据故未改的三处**（请转给对应探针或直接立项）：
   - `Mira Wavehorn / Gesture: Surge.description` 与 `Avwynn Taol / Gesture: Surge.description`（均在 `ember.crucible-adventure.json`）正文写「Surge手势」裸英文，而 `crucible.talent.json` 同模板写「涌动手势」—— 属 scan_bare_english_names 范畴。
   - `Players' Guide.pages.Welcome to Ember.text`（孪生两包）："要开始游玩 Ember，主要有两条**径**" 掉了「路」字（应为 两条路径）；同叶 "player engagement" 译成「玩家**交战**」（应为 玩家参与度）。这两处是我核 uncertain #8 时撞见的，与列表错位无关，没动。

4) **glossary_ec 需要修两处**：键 `Liliman’s Bar Grille` 把 `&` 吃掉了，与正文的 `Liliman’s Bar &amp; Grille` 结构性不匹配（这就是它活到第八轮的原因）；其值「利利曼酒吧烧烤馆」的姓氏译法又与同页已定的「达丽莎·莉莉曼」冲突。建议改键为 `Liliman’s Bar &amp; Grille`、改值为「莉莉曼酒吧烧烤馆」，与本轮批次对齐。

5) 若主控不认同我对 uncertain #9（`吞噬思维 Devour Thoughts`）的推翻，那就是「正文内的能力名算不算专名、要不要带双语尾巴」这条**全局体例**待裁决，改的应该是全库而不是这一处。

## g04

1) **我没改但主控应转给术语组的 4 处**（品质阶梯译名不一致，属术语一致性类而非叶内重复块类，不在 g04 判据内，我刻意没碰以免与别的 agent 撞车）。这 4 处是我做双仓穷尽阶梯扫描（EN 同叶出现 ≥3 个阶梯词）时顺带扫出来的，不在 g04 的 findings 里：
   - `2-Crucible汉化插件/crucible.equipment.json :: Gem Of Conjured Flame.actions.gemOfConjuredFlame.description` —— Shoddy 译作「劣质」（全库 Shoddy→粗糙 109）。
   - `2-Crucible汉化插件/crucible.rules.json :: Equipment.pages.Affixes.text` —— 词缀预算表 Shoddy 行译作「粗制」。
   - `2-Crucible汉化插件/crucible.rules.json :: Equipment.pages.Weapons.text` —— 武器品质表 Shoddy→「粗制滥造」、Superior→「优异」、Masterwork→「大师之作」。
   - `2-Crucible汉化插件/crucible.rules.json :: Equipment.pages.Armor.text` —— 护甲品质表三档译名与既定阶梯都对不上（我 grep 了 粗制/劣质/优异/高等/精制/杰作 六种写法均无命中，需要人眼看一遍这张表到底用了什么词）。
   注意：`crucible.rules.json` 的 Weapons/Armor/Affixes 三张表是**规则书主表**，玩家查品质档位最先看它们。它们错了的话，我这一轮把物品侧统一到 粗糙/标准/精良/卓越/大师级 反而会让主表与物品对不上号 —— 建议**优先**处理这三张表，且与我的批次同批落盘。

2) **文件撞车提示**：我的批次覆盖 `crucible.equipment.json`（3 叶）、`crucible.pregens.json`、`crucible.playtest.json`、`ember.crucible-adventure.json`（19 叶）、`ember.adventure.json`（1 叶）。其中 `ember.adventure.json` / `ember.crucible-adventure.json` 的 `Players' Guide.pages.Creation Overview.text` 是**整叶覆盖**（表格行重排），若有别的探针也在改同一叶，三方合并时必须以整叶为单位裁决、不能按行合并。

3) **一条方法论坑，建议记进 PROJECT.md**：本项目的「标记破损」闸只比对标签的**多重集**。Healing Tonic 那 10 叶丢了 Shoddy/Superior 两个 `<strong>` 对，又在别处多出两对，总数相等，闸完全静默。所以「粗体是否落在正确的词上」这一类缺陷现有闸门结构性看不见，需要位置敏感的检查器才能覆盖。同一机制的反面后果是：**凡当前 CN 签名已与 EN 相等的叶子，任何「顺手补个 <strong>」的建议都必然被闸拒绝**，g04 里就有一条 finding（Periapt 补粗体）栽在这上面。

## g05

三件事需要主控裁决/协调：

【1｜叶子重叠，合并时请取我的超集值】crucible.playtest.json 与 crucible.pregens.json 的两片 Blast Flask 叶子，我在**同一片叶值**里同时修了两处：否定从句缺失（我的判据）＋ Fine 误作「卓越」（本属术语/重复标签那一路）。若术语组也对这两个 path 出批次，其值只含 Fine 修正、**不含**从句修正。三方合并时请取我的值（超集），不要取术语组的，否则从句修正会被覆盖回去。
   受影响 path：
   - crucible.playtest.json :: Playtest 1 - The Ring of Valor.actors.Eliorwen.items.Blast Flask.actions.blastFlask.description
   - crucible.pregens.json :: Eliorwen.items.Blast Flask.actions.blastFlask.description

【2｜同类缺陷的漏网 3 条，超出我的探针范围，未出批次】用「五阶品质表」精确切片扫全库时，除上述 2 条外还有 **3 条同样把 Fine 译成「卓越」、导致表内两个「卓越」**的叶子。它们与否定判据无关，我按「宁可少改不可错改」没有认领，请派给术语组（或直接授权我补批次）：
   - 1-Ember汉化插件 / ember.crucible-adventure.json :: Ember Early Access.actors.Rala Ushna.items.Rallying Elixir.actions.rallyingElixir.description（卓越-24 / 卓越-48）
   - 2-Crucible汉化插件 / crucible.equipment.json :: Paralytic Vial.actions.paralyticIngest.description（卓越-4轮 / 卓越-8轮）
   - 2-Crucible汉化插件 / crucible.equipment.json :: Rallying Elixir.actions.rallyingElixir.description（卓越-24 / 卓越-48）
   判定依据可复现：全库含 Shoddy+Standard+Fine+Superior+Masterwork 的五阶表共 119 片，114 片用「精良」，仅这 5 片缺。

【3｜两处术语分歧，我看到但没动，供别路参考】
   - 同一句里的机制词 `Blast`，6 份 Blast Flask 副本给出三种译法：「爆破」（equipment + ember 三份）/「爆炸」（pregens）/「爆裂」（playtest）。「爆裂」是全库孤例。不属我的判据，未改。
   - `Otherhood` 在 Otherhood of Fortune 页里有两种中文：全称「幸运异姊会」、简称「异缘会」。若属有意的全称/简称之分则是 by-design，但「异姊」与「异缘」用字不同源，像是两次独立翻译。建议术语组核一下。

另附一条给报告方的方法论提醒：本组 5 条 finding 里有 **2 条的 en_excerpt 存在转述失真**——Blast Flask 那条凭空多了 `<b>…</b>` 标签，Shent Moon Temple 那条把 `<sup class="system-swap-inline">` 双系统切换块简化成了裸 `@Reference[restrained]`。两次都不影响实质结论，但都足以让下游按转述直接改而破坏标记。后续轮次建议探针一律回填原始叶值而非人工摘录。

## g06

1) **【阻断项，需主控先动英文基线】** `ember.adventure.json` 与 `ember.crucible-adventure.json` 的 `Ember Early Access.journals.Yakoshta Mine.pages.Elevator.text`，英文基准里 `[[/skillCheck athletics 15 check.` 缺 `]]`。`3-常用脚本/qa/apply_translations.py` 的 `INLINE_CMD = re.compile(r'\[\[[^\]]*\]\]')` 会从这个 `[[` 吞到下一个 `]]`（`[[/skill arcana 12]]`），把中间那句未译英文整个吞进一个 markup token，**任何译法都必被判 markup mismatch**（我实测 REJECTED markup 1）。这不只是「记进 LOCAL-PATCHES 就行」的上游笔误，它正在阻止一处真实漏译（严重）被修复。建议：先在英文侧补 `]]`，再重跑本叶的翻译批次。同类风险普遍存在——凡英文有不闭合 `[[`，其后到下一个 `]]` 之间的散文都被闸门永久锁成不可译，值得全库扫一遍 `\[\[(?:[^\]]|\](?!\]))*$` 类模式。

2) **同族缺陷溢出我这条 finding 的路径范围**：裸英文人名不止 Alchemical Decisions 一叶。屏蔽 enricher 后全库扫描（两仓库）得到的完整清单只有 Ooze Control 这三叶（×2 孪生包）：
   - `Ooze Control.pages.Alchemical Decisions.text` —— Jasper×34 / Tauric×31 / Sellen×4（**已在我的批次中修掉**）
   - `Ooze Control.pages.Good Ooze, Bad Ooze.text` —— Tauric×10（未动，不在我的 finding 路径内）
   - `Ooze Control.pages.Wayward Sampler.text` —— Tauric×16 / Sellen×2（未动，同上）
   两条未动的请指派或并入某一组，否则 `sync_twin_packs` / 裸英文闸日后仍会报。同时请把「Squish 全库 0/32 从不翻译，属 by-design」记进约定，避免后续 agent 顺手「补译」。

3) **关于本探针 verdict=no-signal 却报 7 条 confirmed 的自相矛盾——查清了，verdict 没标错。** 情态强度这个判据本身确实零信号：它的 41 条 modal_invented 候选我抽检 10 条全是噪声，两条情态类 uncertain（should→必须、凭空加「可以」）也都被推翻，没有一条真缺陷出自该判据。那 7 条 confirmed 全部是**判据外**的偶然发现——permit_drop 子扫描把三条 `May contain …` 并排列出来时才看见档名错配，其余是逐叶通读时撞上的漏译与旧英文残留。所以「判据无效」与「报告里有真缺陷」两件事同时成立，不冲突。真正值得主控注意的推论是：这 7 条里有 3 条（Ooze Control 两叶 overview + Glint of Gossamer）属 PROJECT.md 第 7 节 B 项 `scan_en_drift` 的 changed 桶 —— 那 1216 条至今只抽查过。本组仅凭偶然翻到就命中 3 条真阳性，说明该桶的真阳性密度不低，建议单独立项系统清理，别再靠别的探针捎带。

4) 我改正了 finding 自带的两处事实错误，主控如果拿 findings 原文去做交叉核对会对不上：`Dripstones` 正确译名是**滴石笋**（全库 87 例）不是 finding 写的「滴石区」（0 例）；`Living Quarters` 的感知 DC 是 **13/11** 不是 finding 写的 16/14（那是 Central Lunarium 的）。批次按库里实际数据出，不按 finding 转述。

## g07

**结论先说：本组 18 条，推翻 2 条（U1 Scene 灯光/音效名＝GM-only；U6 crafting/dnd5e-items＝本来就不是缺陷），其余 16 条全部推翻不了，但一条批次都出不了 —— 它们无一例外是「字段根本没进英文基准」的管线缺口，回写闸门实测 4/4 no-EN 拒。修法只能是改 mapping + 重抽基准，按任务约束我没动 mappings.mjs。**

需要主控裁决的四件事：

**1（最要紧，是个新查出来的管线错，不在原 finding 里）：`BABELE_DEFAULTS` 与 Babele 2.9.1 真默认的三处偏差，其中一处是活的 bug。**
mappings.mjs 的 BABELE_DEFAULTS 自称「Kept minimal and in sync with babele/script/mapping/default-mappings.js (2.9.1)」，实测三处对不上：
- `RegionBehavior` 只写了 `{name}`，Babele 真默认（default-mappings.js:228-239）还带 `_variants`：displayScrollingText → `text:'system.text'`，teleportToken → `revealedDialog/unrevealedDialog`。**结果是运行时会去查 `text` 这个键、抽取器却永远不产出它** —— 与 2026-08-12 §2.2 的 ActiveEffect 事故同一根因，只是方向相反。
- `Actor` 整型缺席，Babele 真默认有 `description:'system.details.biography.value'`；因为项目层 CRUCIBLE_ACTOR 没有 `description` 同名键，这条默认在**运行时依然生效**，同样是「运行时查一个抽取器不产出的键」（＝本组 F2）。
- `Item` 整型缺席（Babele 默认 `description:'system.description.value'`）。**注意这一条补回去没用**：项目层 CRUCIBLE_ITEM 用同一个键 `description` 挂 crucibleDescription，`effectiveMappings()` 与 Babele 的 `#mergedDefinition` 都是按键覆盖，补了照样被顶掉。F1 的 suggested_cn 把这个当备选方案是**错的**，只有改 crucibleDescription 转换器一条路。
建议：把 BABELE_DEFAULTS 真正对齐 2.9.1（含 `_variants`），并加一个「抽取键集 ⊆ 运行时查找键集」的自动断言，否则这类不对称还会再出。

**2：F1 的兜底假设是错的，而且错的方向会写坏数据。**
F1 说 dnd5e 侧 802,719 字「理论上能被 dnd5e 中文 Babele 模块兜底」。我照 runtime 代码走了一遍：`crucibleDescription(value={value,chat}, translation="中文字符串")` 命中 `if (isStr(translation)) return mergeObject(value,{public:translation})` —— 中文被塞进 `system.description.public`（dnd5e 从不读），`system.description.value` 仍是英文。所以**source-pack fallback 对 dnd5e 侧物品不是「兜得住」而是「静默污染」**。如果哪天真挂了 dnd5e 中文 Babele 模块，这 80 万字会在数据里多出一个假 public 键。改 crucibleDescription 时必须连这个分支一起处理。

**3：范围与优先级需要拍板 —— 16 条存活里只有 3 条在主线。**
- 主线（crucible-adventure）：F5 encounter tokenData.name（814 处 / 143 唯一，两包各改一次）、F11 RegionBehavior（18 处，含上面第 1 点的 displayScrollingText）、F12 deity 三列表（130 处）、以及 U2 的 adversary 那 122 处 pronunciation。
- dnd5e 附带项：F1（150 万字）、F2（55.7 万字）、F3、F4、F6–F10、U3、U5。按 PROJECT.md ⚑「dnd5e 侧顺带一起翻，不得为它牺牲主线」，F1+F2 加起来 200 万字符，量级已经超过前七轮落盘总和（190 万），**这不是「顺带」能吸收的**。请明确：是只补管线让基准里有这些键（覆盖率数字会从 100% 掉下来），还是连管线一起先不动、把它记成已知作用域外。
- 建议的最小主线动作：F11 补默认表（几乎零成本）＋ F5 新 converter（唯一名只有 143 个，翻译量很小但玩家可见度最高）。

**4：请一次性裁掉两条「以后不要再报」的。**
- `ActiveEffect.enchantment.system.changes[].value.condition`（U4）：渲染路径我走通了、玩家确实看得到，但全库只有 **2 条非空**，而要抽它得改 extract_en 的 structured 语义（现在只实现 mapping 形、没实现 key+valuePath 形，且注释里写明是有意的）。建议裁「不抽」，写进 mappings.mjs 的 NOT-translatable 注释块，和 `system.terrain` 放一起。
- `Scene.lights[].name` / `sounds[].name`（U1）：GM-only（Foundry v14 BaseAmbientLight.getUserLevel 对非 GM 返回 NONE），已推翻。若主控认为 GM 界面也算交付面，请单独立项，别让它再混进「玩家可见」判据。

另附两条给下一轮的方法论修正（都已在正文举证）：① `flags.*` 一刀切会漏掉模块挂在 flags 下的展示字段（U2 就是这么漏的）；② 「同一逻辑字段在不同包里容器形状不同」是本库的常态（`system.advancement` 在 ember.character 是数组、在 ember.adventure 是按 id 建键的对象），任何新 mapping 都要两种形状都吃。

## g08

四件事需要主控裁决：

【1｜探针漏项，我已放进批次，可一键撤】Helkas Drake Moments.results.4-4.description 的『持续六轮』（EN: lasts for six turns）不在 g08 的 21 条里，是我独立重扫撞出来的同类严重缺陷，两个孪生包各一条 entry（g08.1.ember.crucible-adventure.json.json 与 g08.1.ember.adventure.json.json）。如果主控坚持「只出存活 finding 的批次」，删这 2 条 entry 即可，其余 21 条不受影响。顺带：这条证伪了 g08 uncertain 笔记②的「全库无漏网」保证——探针的穷举用阿拉伯数字锚定，中文数词『六轮』漏了。**建议提醒其他做数值类判据的探针同样存在中文数词盲区。**

【2｜措辞取值与 findings 的建议不同，可能与他组批次冲突】14 处 Gesture 维持消耗，findings 建议统一改成「此后每一轮」；我改成了机械替换 回合→轮次（如「此后每个后续回合」→「此后每个后续轮次」）。理由：(a) 这是最小改动，只动单位词，前导连接词一字未动；(b)「轮次」正是 crucible.talent.json 里这两个 talent 源条目对逐字节相同英文的现有译法，也是 crucible.rules.json 对 'subsequent rounds' 的现有译法「在后续轮次中如常行动」——按裁决阶梯，同名源条目比同卷旁证更硬。副作用：改完后该 pack 内会是 14 处「轮次」+ 2 处「每一轮」（Amalthea Stonecraft / Larissa Toth 原本就对，我没动）。若主控要求全 pack 统一成「每一轮」，我可以重出批次，但那会连带改动 2 条本来正确的叶。

【3｜跨探针移交，我没改】(a) Kern.items.Gesture: Sense 中文残留英文「Sense手势会随存在成长」——注意 findings 说是 7 处，实测只有 1 处，另 6 处都译成了「感知手势」/「“感知”手势」，请勿按 7 处派工。(b) 同组 Sigil.items.Gesture: Sense 把 'scales with Presence' 译成「会随感知力而成长」，Presence 定译「存在」，其余 6 处都写「存在」——这是术语错，归 term 类探针。

【4｜全库风格，超出本组】英尺 / 尺 不统一，且不只是跨仓库：crucible.talent.json 内部 Gesture: Sense 写「半径 30 尺」而 Gesture: Aura 写「半径20英尺」；Ember 侧同一句英文（30 feet radius）有「半径 30 英尺」也有「半径30英尺」（连数字前后空格都不一致）。两种都是 feet 的合法译法，不是规则性错误，我按「宁可少改不可错改」没动。要不要全库统一（以及统一成哪个、要不要顺带统一数字空格）请主控定，这是一次性全库批处理的量级。

另：本组无跨单元冲突，管线无问题。pair_dump.py / term_gate.py / apply_translations.py 三个脚本都用 os.path.join 拼路径，没有主控提到的 Windows 反斜杠 glob 那个坑。

## g09

1) 【超出我分组、但同类且已验证，请主控裁决是否收】ember.crucible-adventure.json 的 Troubling Reports 物品把 Arcturel 的 "Level 3 mine" 译成「3级矿场 / 3级矿井」，与 Arcturel Dives journal 内一致的「三层矿井」冲突（journal 侧 5 处：categories.Level 3 Mine、pages.Level 3 Mine Office.name、Area Overview.text ×2、The Last Pit.text、Zodi Trask 页；item 侧 2 处例外）。我没有把它塞进批次，因为它不在 g09 的 findings 里，硬塞属于扩大范围（本项目吃过这个亏）。若要收，两条叶为：
   Ember Early Access.items.Troubling Reports.description.public：「…关于阿克图瑞尔3级矿场所收到的报告。」→ 改「3级矿场」为「三层矿井」
   Ember Early Access.items.Troubling Reports.description.private：「…不少关于3级矿井中怪事的备注。」→ 改「3级矿井」为「三层矿井」
   孪生安全：ember.adventure.json 里该物品只有 .name 叶，没有 description 叶，只需改 crucible 侧。

2) 【仅备案，我判定不必改】Marlstone Manor journal 的 categories 'First Floor'→「一楼」/'Second Floor'→「二楼」（两 pack 共 4 叶）与该副本场景侧的「一层/二层」不同调。我没动它：journal 目录与场景列表是两个界面，「一楼/二楼」本身是好中文，且不在 finding 内，改它属于我自己发明判据。若主控想全卷统一可另开。

3) 【仅备案】同 journal 内 pages.Central Elevator.name 译「中央升降机」，而 Gameplay Details 正文同一物译「中央升降梯」。属术语一致性，不属方位/序数，未处理。

4) 无跨单元冲突：我只碰 1 号仓库的两个孪生 adventure pack，且限定在 Spellbreaker Tower 七个场景 + Marlstone Manor 二层场景名这一小片路径内，与其他组的 glossary 类改动无重叠预期。

## g10

需要主控裁决的 6 项（都是全局命名/术语表决定，不是单叶缺陷）：

1) **Marlstone 复合专名要不要换词根**。现状：城区+石材 = 马尔石（gazetteer 页 name，且正文明说城区得名于该石材，「铺着马尔石」「马尔石不仅是建造这片城区所用的材料」都靠「石」字成立）；8 个复合专名 name = 马尔斯通（庄园/晚会/装饰挎包/职员钥匙/两张场景/入口/远景），涉及约 130 叶。二选一：把复合名改成马尔石庄园/马尔石晚会（改 ~130 叶，但石材句子保住）；或另造一个既能当石头又能当地名的译名。我这批只把 3 处光杆城区拉回马尔石，没动复合名。

2) **Hallows 组织与城区要不要共用词根**。英文两者同名 \"the Hallows\"，中文拆成 幽圣所（组织，109 叶）/ 圣堂区（城区，138 叶）。除我改的 2 叶外，还有约 40 处**指组织却写成圣堂区**（「圣堂区代表」= Hallows Representative、「帷幕锁链直接向圣堂区汇报」、「佩戴圣堂区徽章」、「圣堂区监狱」、「他把萨尔瓦告到了圣堂区」、「圣堂区负责城内的免费邮递」等）。要么统一词根（如城区改幽圣区），要么逐条把指组织的 40 处改回幽圣所——后者判断量大、错改风险高，我没做。

3) **The Nineteen 取十九人还是十九神**。name 与多数派都在十九人（60 叶），但英文明说是 pantheon of gods，「十九人」字面读作十九个人。改十九神要动 60 叶 + name 叶，与阶梯默认方向相反。glossary_ec 里 'The Nineteen'→十九人 与 'Nineteen'→十九神 两条并存，也需一并订正。

4) **Protector / Guardian 撞车**。glossary_ec 现在 `Protector`→守护者、`Guardian Beast`→守护野兽、`Mythspire Guardian`→神话尖塔守护者、`Guardian of Myth`→神话守护者，而 actor name `Burnished Hand Protector`→辉手守卫者、`The Derelict Protector`→失职守卫者。英文闸 `\\bProtectors?\\b` 55 叶（守卫者 46 / 守护者 18）、`\\bGuardian\\b` 34 叶（守护者 31 / 守卫者 4），双向串味。建议先定死 Guardian=守护者、Protector=守卫者，再批量落（含 `Svala Bronwen.archetype.name` 与 `Burnished Hand Protector.archetype.name` 两个「守护者 Protector」）。

5) **Lesser Restoration 三种写法**（次级恢复 13 个 name 叶 / 次级复原术 6 叶 / 次等复原术 1 叶）。冲突的是两个权威：本库 name 阶梯说次级恢复，而同库 `Greater Restoration` 的 name 是「高等复原术」+5e 官方中文是「次级复原术」。这实质是「dnd5e 侧法术/物品名是否一律跟 5e 官方中文包」的政策问题——我这批 F22 的四组恰好两个权威一致才敢落，这一组不敢。

6) **Ossuary 四种写法**：`Ossuary`→骨库 / `Silent Ossuaries`→寂静藏骨堂（但该页正文写「这座寂静的骨库」，自相矛盾）/ `Ossuary Loot`→藏骨所战利品 / `Disturbed Ossuary`→受扰乱的藏骨堂。`Ossuary Loot` 这张表在全库英文里无任何引用，判不出它归属哪个 ossuary，故未动。

另附一条给主控的落盘提醒：**glossary_ec.json 目前是当前译文的镜像，本身带着这些分裂**（'Lake Jinro'→金罗湖 但 'Lake Jinro Lunar Shrine'→锦露湖月神殿；'Orbis'→奥比斯 但 'Well of Orbis'→欧比斯之井、'Altar of Orbis'→欧比斯祭坛；'Giant Moonstone'→巨型月长石；'The Gem of Orbis'→奥比斯之石；'The Hallows'→圣堂区 与 'Hallows'→幽圣所）。我的批次落盘后，这些 glossary 条目需要同步订正，否则下一轮又会被当成「权威」反向回灌。

## g11

四件要主控裁的/要知道的：

**1（必须裁）修 id 还是改锚点。** 我选了修 id，理由是两条既定规矩都指这边：BRIEF 铁律 2「方括号内的目标与参数照抄不译」——把 234 处 `@UUID[…#anchor]` 改成中文 slug 会直接违反它；以及「改动面小的那边优先」（112 叶 vs 234 叶）。加 id 对三道闸全隐形我也逐个核过：apply_translations 的 `TAGNAME` 正则只数标签名不看属性；scan_class_drift 只看 class；scan_attr_text 把 `id` 列在 MECHANICAL 白名单里；scan_en_residue 先 `blank(TAG)` 把整个标签抹掉，所以英文 id 不会被当英文残留。如果主控要反过来改锚点，我的批次作废，需要重出。

**2（落盘顺序）与 g05 有 2 个叶子撞车。** `Ember Early Access.journals.Shent Moon Temple.pages.Temple Interior.text` 在 g05.1.ember.adventure / g05.1.ember.crucible-adventure 里也被改了（我扫批次目录时其他 agent 还在写，撞车数可能继续涨）。我的批次值是**建批时刻的整叶快照**，直接后落会把 g05 的改动盖掉。好在我的改动是纯 `id="…"` 插入、可确定性重放，建议二选一：
   - **把 g11 放在所有批次最后落**，落之前用我的生成器对着合并后的 CN 重跑一遍（脚本在 `C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\892fdb28-d096-4415-94d1-99d12c38ef86\scratchpad\build_batches.mjs`，读 `findings/anchor_fix_plan.json` + 现行 CN，幂等）；
   - 或者三方合并时对这 2 个叶子按「在 g05 的新值上再插 id」处理。

**3（孪生对齐，我主动多加了 1 处）** plan 里 `Yakoshta Mine.pages.Makeshift Barricade` 只挂在 crucible-adventure 下（引用它的 item `Yakoshta Blue Track Lever` 只存在于 crucible 侧）。但两包该叶的 EN 与 CN 都逐字节相同，只补一边会被 `sync_twin_packs` 报成「英文同、中文异」。按 BRIEF「同一处缺陷两包各改一次」，我在 ember.adventure.json 也补了同一个 id（该包无引用，纯属无害对齐）。所以两个 ember 批次都是 55 叶、文件大小完全一致（438081 字节）。如果主控不想要这处冗余，删掉 g11.1.ember.adventure.json.json 里的这条即可，其余不受影响。

**4（顺手挖出来的、不属于 g11、我没动）**
   - **上游死链 2 组，英文侧同样死，入 LOCAL-PATCHES**：`JournalEntry.emberCosmos00000.JournalEntryPage.0DQdqtsLn3M1ZZA2#fundamental-forces`（页 id 在 LevelDB 索引里根本不存在，Character Classes.pages.Monk，两包各 1 处）；`Compendium.crucible.rules.JournalEntry.QhZgmBrdLAGwYy5c#movement-cost-calculation`（挂在 **JournalEntry 级** UUID 上的锚点，`JournalEntry._onClickDocumentLink` 压根不读 hash，Shent Moon Temple.pages.Area Overview，两包各 1 处）。四处英文侧字符串完全相同 → 上游问题。
   - **上游作者的 `data-anchor=` 是无效的**：库里 8 种标题开标签带 `data-anchor="…"`（如 `<h4 data-anchor="under-construction">`、`<h2 data-anchor="location-discovery" class="divider">`），但 Foundry 的 `data-anchor` 是 `_renderPageTOC` 建完 toc 之后**回写**上去的输出，不是输入，`buildTOC` 从不读它。目前没有任何引用用 `#location-discovery` 这类 slug（我查过 = 0 处），所以不构成死链，但说明作者对机制有误解，升级 Ember 后值得复查。
   - **同页重名标题（不是我这一档，但落在同一批页面上）**：两处把英文两个不同标题译成了同一个中文，触发 Foundry `_flattenTOC` 的 `$N` 去重 —— `Writhing Grave.pages.Waterfall Nexus`：`Over the Moon Quest` 与 `Over the Moon Side Quest` 都成了「月上之旅任务」；`Lightless Halls.pages.Runekey Altar`：`Deciphering a Glyph` 与 `Deciphering the Glyphs` 都成了「破译字形」。这两组目前没有锚点引用（所以不是死链，我没改），但属于「两个英文合并成一个中文」的译文缺陷，归口给管重名/合并那一档的 agent 或主控裁。同页面上其余 $N 去重（宇宙宝石/无可逃避的命运/幼年加甘萨斯战术/开发中/太阳星）英文侧本来就重复，是 by-design，别去动。

## g12

【一】Mazira 音译全库二分（唯一存活项，请主控裁决方向后统一处理，我未落批次）

事实（英文闸 `Mazira(?![nsA-Za-z])`，两仓库全包，含散文非链接处）：
  马齐拉 17 处 / 马兹拉 14 处，跨 7 个不同叶子（×2 孪生包）：
  - 马齐拉：Ancestries/Human.contentGamemaster(1)、Ancestries/Cor'ak.text(6)、History/Age of Rediscovery.text(1)、actors/Rorhim Iron-Cask.biography.private(1)
  - 马兹拉：Cultures/Tayan.text(1)、Cultures/Maziran.text(2)、Crumbling Sanctuary/Corpin Arrival.text(2)、Organizations/Muzseri.contentOverview(1)、actors/Serethus(1)、actors/Scalemaw(1)

关键点：scan_uuid_swap 只看见 @UUID 标签这一小块（该目标下 马齐拉 3 唯一处 vs 马兹拉 1 唯一处），于是建议把 Tayan 的 马兹拉 → 马齐拉。**这个方向很可能是反的**：
  - 同一页文档 name = `Maziran` → 「马兹兰 Maziran」（glossary_ec 有条目，全库 36 处零例外）
  - 裁决阶梯最高一级（目标文档 name）+ glossary 都指向词干「马兹」，因此 `Mazira` 应作 马兹拉，17 处 马齐拉 才是错的那一边
  - 但那 17 处大多在我这 35 条 finding 之外，改动面 6 叶×2 包，属全局术语决策
建议主控：先定「马兹拉 / 马齐拉」，再一次性全库统一（连同散文，不能只改 @UUID 标签）。查过 4-临时脚本/2026-08-12-fix/reports/label_map.json 与 resolutions.json，此前无裁决记录。

【二】Sunalins 三种译法（不在本组 finding 的「错的那一侧」，顺带上报）
英文闸 `Sunalins`：7 唯一叶×2 包 = 14 处，中文为
  苏纳林斯 3 唯一处（Deities/Spectra、Deities/Alar、Deities/Sockets，均在 "Pantheons:" 列表槽位）
  苏纳林诸神 2 唯一处（Cultures/Kessian、Deities/Sunalin.contentOverview）
  苏纳林 2 唯一处（Deities/Lantyr、Deities/Orvaath，同样是 "Pantheons:" 槽位）
报告只标了 苏纳林诸神（少数派），方向同样是反的：glossary_ec `Sunalin`→「苏纳林 Sunalin」，页面 name 也是「苏纳林 Sunalin」，Sunalin 页自身正文与 contentOverview 都用 苏纳林诸神。**把英文复数 -s 音译成「斯」的 苏纳林斯 才是可疑的那一侧**，且三处 苏纳林斯 与两处 苏纳林 处在完全相同的 "Pantheons:" 槽位却不一致。建议主控定一个（我倾向 苏纳林诸神 或统一为 苏纳林），再一次性处理这 5 唯一叶×2 包。

【三】Kessian 作定语修饰地名时的一处偏离（极轻，可不动）
英文闸 `Kessia`→凯西亚 26/26 全一致；`Kessian`→凯西安 18/20、凯西亚 2。那 2 处是 Deities/Sunalin.text 的 "the Kessian continent"→「凯西亚大陆」，指大陆而非族群，中文这样读更顺（同叶下一句 "Kessian identity"→「凯西安人的身份认同」是对的）。不在 finding 内，判 by-design，仅备案。

【四】对 scan_uuid_swap 判据本身的意见（管线问题）
这一档 68 条 UNCERTAIN 的真阳率是 1/35（约 3%），假阳性有四个固定成因，建议下轮改判据：
  a) 同目标多英文标签（Draconic/Dragons、Shard God/Shard Gods、Elder God/Elder Goddess、Big Liz/Baradom、her journal/Avwynn's Journal、Sin/Sin Marmot、Gameplay Details/Arcturel Tradeway、Grand Kalion Stadium/Arena Ridge、Milestone/Milestone Progression、Writhing Tendrils/Writhing Whisperer Tendril、Moiran/Blood Barons、Mazira/Maziran、Kessia/Kessian、Ordain/Ordani、Eternas/Spiritlands）—— 占绝大多数
  b) `#锚点` 指小节，标签本就该是小节名而非页名（The Eternal Soul/Soul Transference、Fighting Kaftor、Failing to Escape/The Veiled Chain Trap、Encounter Details/Searching the Aftermath、Using the Locator Rod）
  c) 英文源把一个通用 Actor 复用成多个具名 NPC（Actor.9plFRf3Hurd9r7ol 一个目标带 Emelyn Arvoda / Arcturians / Eolas Hathwick / Calandra / Hob Korell 五个标签）
  d) `en_label` 只取叶内首个英文标签 → 张冠李戴（本轮 68 条里错标 30 条以上）
最小修复：报告改为**按位/按目标配出本处真实英文标签**，并在 majority 计算里按 (target, en_label) 分桶而不是只按 target；再把带 `#anchor` 的目标与裸目标分开统计。
