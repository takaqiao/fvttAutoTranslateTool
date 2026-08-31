# 第十二轮 escalate

## G1

以下要改术语表/工具，我不能写，请代改（键与新值都列全了）：

A. `C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json`（base 层）与 `C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\glossary\glossary_ec.json`（产物），两个文件都要改：
  1. "Senses"：base 现为 "感知"、产物现为 "感知" -> 都改成 "感官"。（"Sense" 保持 "感知" 不动 —— 那是 Crucible 的 Sense 手势，「感知施法 Sense Spellcraft」已成体系。）
  2. "Monstrosities"：base 现为 "怪物"、产物现为 "怪物 Monstrosities" -> 改成 "畸怪" / "畸怪 Monstrosities"。定译是畸怪，包内 ember.crucible-adversary.json 的 folders.Monstrosities 早已是「畸怪 Monstrosities」，词表却还是「怪物」（原因见 C）。"Monstrosity" 已是 "畸怪"，无需动。
  3. "Celestial"：base 现为列表 ["天界生物", "天界的"] -> 去掉「天界的」，只留 "天界生物"。产物 glossary_ec 现为 "天界的 Celestial"；G1 批次落地并重建后会自动变成「天界生物 Celestial」（该 folders 位于 entries.Ember Early Access 之下，harvest 走得到），但 base 里的「天界的」不清掉以后仍是隐患。

B. 仅备案、本轮不建议动的两个 base/产物不一致（同形异义，硬统一反而错）：
  - "Aura"：base ["灵气","气场","灵光"] vs 产物 "奥拉 Aura"。天体专名＝奥拉，Crucible 手势＝灵气，两者都对，词表天然无法用一个键表达。
  - "Ooze"：base "软泥怪" vs 产物 "软泥"。生物分类＝软泥怪，地形＝软泥。

C. 工具缺陷（比上面几条更值得修）：`3-常用脚本/tm/build_glossary.py` 的 harvest() 只走 en_doc.get('entries', {})（约第 78、100 行），顶层 folders 从来不采。后果是所有 crucible.* 包以及 ember.character.json / ember.crucible-adversary.json / ember.crucible-items.json 的顶层文件夹名永远进不了 harvest 层，只能被 base 层覆盖 —— Monstrosities 明明包里早就是「畸怪」而词表还是「怪物」就是这么来的；本轮的 Senses 修法同理也不会自动带出。建议补一行 walk_pairs(en_doc.get('folders', {}), cn_doc.get('folders', {}), got)（英文侧 folders 是 {name: name} 平表，结构与 cn 侧一致，直接配对即可），采进去之后这类孤儿键就不需要手改 base 了。该文件不在我本轮可写范围，没动。

## G6

以下均为本轮在 22 页之外发现、需改 compendium/cn 其他页面或术语表 base 层的项，G6 未动：

【1】the Hallows（组织=幽圣所）在 J00 以外仍被大量写成「圣堂区」。ember.adventure.json 全库「圣堂区」77 处，除 The Hallows 城区页自身及其城区正文外，以下语境按英文判断均指组织，应改「幽圣所」（同步 ember.crucible-adventure.json 孪生叶）：
  - 「…但他们也会向帷幕锁链汇报任何重大活动，并代表圣堂区检查当地石构的任何结构问题」→ 幽圣所
  - 「…以及维护并执行审序院法律的圣堂区代表」→ 幽圣所
  - 「…或可以转交给其他权力来源，例如各大商会或圣堂区」→ 幽圣所
  - 「…也加深了审序院与圣堂区之间的相互尊重；后者还协助提供…」→ 幽圣所
  - 「…把那些本可能被圣堂区检查、以确认是否带有危险魔法的…」→ 幽圣所
  - 「…伊里瓦妮掌管着那些可能会让圣堂区皱眉的奥术货物…」→ 幽圣所
  - 「…以避开圣堂区警惕的目光，因为圣堂区担心危险的神器流入城中」→ 幽圣所（2 处）
  - 「…最近更因此被圣堂区推选进入审序院，获得一席之地」→ 幽圣所
  - 「…这引起了数家商会与圣堂区的注意…」→ 幽圣所
  - 「…尽管阿纳克瑞纽姆的一名招募者和圣堂区的一名管理者都曾接触过他们…」→ 幽圣所
  判据：Organizations 分卷 Hallows 页英文 contentOverview 明写 'The Hallows is one of three major arms of the Ordinate'。凡「检查/汇报/推选/皱眉/注意/管理者」等施事语境一律是组织。

【2】「生命状况」误替换 health/wellbeing，J00 以外还有 3 处，建议改「健康状况」：
  - 「…[[/skill Medicine 14]] 用于把伤害降到最低，管理工人的生命状况，并处理那些确实发生的轻微创伤…」（2 处，孪生同文）
  - 「罗尔里的创伤看起来较轻，而且他的生命状况相对不错，可能是因为他还年轻。」

【3】泛指 chamber/room 被写成「之室」，J00 以外存在成句破损，建议逐处改「厅室/房间/大厅」（专名 阿加瑟罗斯之室 Chamber of Agaseros、心之室 Heart Chamber、追忆之室 Chamber of Remembrance、共祭之室 Chamber of Communion、中央立柱之室 Central Pillar Chamber 保持不动）：
  「这间巨大的八角形之室」「这间巨大的圆形之室」「一间巨大的会议之室」「多间殡葬之室」「宽阔入口之室」「一间巨大的拱顶之室」「这间八角形之室」「她拥有自己规模宏大的复合之室」「此地作为集会与冥想之室的用途」「这间宽敞的圆形之室」

【4】术语表 base 层（C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json）+ 产物 glossary_ec.json 需裁定的键：
  - 'Captain' -> 队长 ：中文「队长」是小队长；船长应作「船长」。全库 41 处「队长」/68 处「船长」已混用，且同一条目内并存（Ember's Bounty：「托里什队长…和大多数船长一样」）。建议改判为「船长」（船只语境）并保留「队长」仅用于 Crucible 军衔表那一处，或统一为「船长」。若改判，需重扫全库 41 处。
  - 'The Nineteen' -> 十九人 ：全库 @UUID 标签用「十九人」48 处、散文用「十九神」17 处。该词指一组神祇，散文「十九神」更可读，但链接标签须与神祇分卷页名一致。建议二选一后同步页名与全部标签。
  - 'Hillhome' / 'Luen' 无条目：建议补 'Hillhome' -> 希尔霍姆 Hillhome、'Luen' -> 卢恩 Luen，防止再分叉（本轮已见 山家/希尔霍姆、吕恩/卢恩 各一次）。
  - Flameguard/Flame Guard 同一组织两种英文拼写 → 「焰卫」69 处 / 「火焰卫队」9 处（后者是 Organizations 分卷页名）。若要统一，须同时改页名与全部引用，属发版级决策。

## G2

【一、base 层批量改（188 条，主要工作）】
完整键值表在 `C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round12\G2.glossary_resolution.json` 的 `base_fixes` 字段（每条含 base_old / base_new / en_leaves / shipped_gated / name_field 权威值）。
请把 base 层 `C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json` 里这 188 个键改成 base_new（＝当前 shipped 值）。不改 base 的话，虽然 harvest 压 base 不影响玩家看到的译文，但争议表会永远报这 188 条。
改完后 `glossary_ec.disputes.json` 的 disputes 应从 197 降到 0（188 条消解 + 7 条 context_split 加白名单不再进表 + 2 条见下）。

【二、需要在产物侧压过 harvest 的 2 条（光改 base 无效）】
1) 键 `Frost` —— 产物当前＝「霜寒」，应为「霜冻」。
   原因：harvest 从 `crucible.affixes.json entries.Cold Damage.adjective`（唯一 1 片叶）收到霜寒，压过了 base 的霜冻；而元素/符文侧 115 叶都是霜冻。
   建议：给 build_glossary.py 加一条 harvest 排除规则——`*.adjective` 这类词缀前缀字段不参与 harvest（或至少单叶不得胜出）。affixes 的 adjective 有 100+ 条，全是刻意与本体术语区分的两字前缀（Corrosion=腐蚀 / Freezing=冰冻 / Impact=冲击 …），同类污染还会反复出现。临时办法：给 `Frost` 加硬覆盖 = 霜冻。
2) 键 `Lesser Soulmark` —— base 已是「次等魂印」，正确；shipped 的「次级魂印」在本轮 G2 批次里已改掉。批次落地后重建即自动收敛，**不要**把 base 改成次级魂印。

【三、超出 G2 范围、但本轮扫到的两处，请主控决定是否单独立项】
1) `Frost Elemental` 全库 9 叶＝「冰霜元素」（crucible.taxonomy.json entries.Frost Elemental、Frost Elemental Bane＝冰霜元素祸骰、以及 Temple Invader / Water Mote / Water Sprite / Water Visitor / Water Wanderer 五个 actor 的 taxonomy.name），而同一元素在别处叫「霜冻」——folders 描述 "Elemental creatures corresponding to the element of Frost." 译作「与霜冻元素相对应的元素生物」（8 叶）。即：元素叫霜冻、该元素的生物类别叫冰霜元素。两侧各自内部一致（9 vs 8），我没有动，改哪边都是 9 叶级连锁改名，需要拍板。同理 `Ray of Frost`＝寒霜射线、`Frost Sprite`＝寒霜精灵、`Frost Claw`＝寒霜之爪 用的是第三种「寒霜」。
2) `Consortium` 译法不统一：`Silver Beam Consortium`＝银光束财团，`Agrimage Consortium`＝农艺法师联合会。两者都不是错译，但同一英文词两种机构后缀，值得统一（建议都用财团或都用联合会）。涉及 5 叶。

【四、词表键歧义，建议加白名单而非继续当冲突】
permanently_excluded 里那 7 个键（Water / Night / Warden / Trader / Swarm / Shard God / Reliquary）都是「一个英文键对应两种正确中文」。建议在 build_glossary.py 给它们打 ambiguous 标记，让争议扫描直接跳过，避免第十三轮再重裁一遍。另外 `Shard Goddess` 应当作为独立键存在，值＝碎片女神。

## G11

以下 3 组问题在我的路线边界之外（属 Book of Tales 路线 / Forest of Stone 地名志路线），我没有自行改写，以免两个 agent 对同一键写出不同值。请指派给对应路线或统一裁决：

【A】ember.adventure.json + ember.crucible-adventure.json，键 `Ember Early Access.journals.The Book Of Tales.pages.The Moon Child of Lake Jinro.text` —— 三处：
  A1. 「任何角色在观察阿玛尔忒亚的返年轻仪式时」→「任何角色在观察阿玛尔忒亚的复苏仪式时」。EN 'ritual of rejuvenation'；「返年轻仪式」不是通顺中文，且全库 rejuvenation→复苏 4 次 / 返年轻 2 次（2 次都在这一叶）。同一场仪式在 Lake Jinro Lunar Shrine.pages.Gameplay Details.text 已作「复苏仪式」。
  A2. 本叶 9 处「讲述者」建议统一为「说书人」（含 <h3 class="divider">讲述者的仪式</h3> 与 <h4>辨识讲述者的仪式</h4>；这两个标题在中文侧未注入 id，改文字不影响跳转，但改前请再确认一次）。同页 .exposition 已用 3 次「说书人」，全库 107:38 偏向「说书人」。
  A3. 本叶用「转生」6 处，而同页 .overview / .summary 用「转世」各 1 处（EN 均为 reincarnation）。同页三段须统一，建议统一到「转生」，并顺带处理 The Turn of a Friendly Page 的同型分叉（.text 转生 / .overview + .summary 转世）。

【B】同两包，键 `Ember Early Access.journals.The Book Of Tales.pages.The Moon Child of Lake Jinro.exposition` —— 「她见过神龛那些古老立石的高耸石构」中的「神龛」应为「神殿」。EN 'the Shrine's ancient menhirs' 指的就是 Lake Jinro Lunar Shrine，全路线 Shrine 一律作「月神殿/神殿」，只此一处作「神龛」。

【C】同两包，键 `Ember Early Access.journals.Forest of Stone Gazetteer.pages.Aedir Signalpost.overview` —— 末句有两处删减。EN: 'From this vantage point at the base of the Signalpost, the ever-visible Broken Tower lies as a jagged splinter on the horizon, like a colossal mirror of the broken ruin that still stands tall on this plateau.' 现译「从这样的高度望去，破碎之塔本身如同地平线上的一道锯齿状尖刺，呼应着此地这座依然高高耸立的破败遗迹。」丢了 'at the base of the Signalpost'、'ever-visible'，并把 'colossal mirror' 换成了「呼应着」。建议新值：「从信号哨站塔基这处观景点望去，始终可见的破碎之塔如同地平线上的一道锯齿状尖刺，宛如此地这座依然高高耸立的破败遗迹的巨大镜像。」

【D】词表层：本轮未发现需要改 lang/、.mjs、glossary_ec.json 或 base 词表（glossary_crucible_merged.json）的孤儿键。本路线全部改动都落在 compendium 包叶子上，下次 build_glossary.py 重建会自动带出正确 harvest 词条（Garganthus=加甘萨斯、Aedir Signalpost=艾迪尔信号哨站、Arcturel Tradeway=阿克图瑞尔贸易道、Lake Jinro=金罗湖、Suarrok=苏阿罗克、Selenic Order=月辉教团 等本路线专名已全部一致，无需人工写词表）。

## G3

【一】词表（glossary_ec.json）需新增 174 条，全部键值已写在产物文件里，请直接合并：
C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round12\G3.glossary-escalate.json
  · `addToGlossary` 174 条，值即「中文 English」双语并列形，其中 8 条已按本轮裁决订正（Point→波因特 Point、Hearth→赫斯 Hearth、Wick→威克 Wick、Hoist→霍伊斯特 Hoist、She'lu→谢卢 She'lu、Reysta→雷斯塔 Reysta、Red Rhuin→红鲁因 Red Rhuin、The Crossing Sea→横渡海 The Crossing Sea），与批次 G3 写进包里的值完全一致。
  · `deleteFromPending_machineEnum` 6 条（terrain 枚举），请从 pending 源头剔除。
  · 注意：这 174 条属 harvest 层可自证的词条，**优先做法是修脚本让它下次自动带出**，不必手改 base。若这一轮就要用，两个文件（base `C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json` 与产物 `glossary_ec.json`）都要写。

【二】需改脚本（我按铁律 1 未动）：`3-常用脚本/tm/build_glossary.py` 的 `harvest()`
  现状：用 `5-其他内容/english-baseline/ember-0.6.0` 与 `1-Ember汉化插件/compendium/cn` 做 `walk_pairs`。该冻结基线里 `scenes.<场景>.notes.*` 以文档 id 作键（`5fN2g192NkqHIof9` → 值 `Toraga`），而 compendium/en 与 compendium/cn 都以注记文本作键（`Toraga` → `托拉加 Toraga`）。键不对齐，配对 0 命中，175 个地图地名标签全部误报为「待译」。
  建议：harvest 改为（或额外）用 `compendium/en` 对 `compendium/cn` 配对——这两侧键方案天然一致；english-baseline 只用于判定「上游是否改过原文」，不该拿来做键对齐。改完重跑，pending 应从 181 掉到 ≤6（剩下的即 terrain 枚举，建议一并加进 is_term 的排除名单）。

【三】待裁决 1 条：`Spiritlands / The Spiritlands`（我没有单方面动它）
  库里三种：**灵界** 32 处 / 24 个页面；**灵界荒土** 21 处 / 只在 Cosmos.World of Ember 与 Cosmos.Soul Cycle 两页；**灵魂之地** 2 处（Organizations/House Bastilla 时间线 + 世界地图注记）。
  已确认是硬缺陷：`Geography.Mort'oliss.contentOverview` 与 `Cosmos.World of Ember.text` 含**同一句英文**（"Mort'oliss, often called The Spiritlands"），一处译「灵界」一处译「灵界荒土」。
  之所以不能直接取多数：**「灵界」已经被 `Ethereal` 占用**——`Keywords: Space, Ethereal`→「空间、灵界」，`Ethereal Ocean`→「灵界海洋」，`Ethereal Appendages`→「灵界肢体」（3 个 actor）。统一到「灵界」会让 Spiritlands 与 Ethereal 撞同一个中文。
  我的建议：Spiritlands 取 **灵界荒土**（避开撞车，且已在两页核心宇宙观文本里自洽），把 24 个页面的「灵界」改过来，同时把地图注记与 House Bastilla 的「灵魂之地」一并改掉；「灵界」保留给 Ethereal。反过来选「灵界」则必须先给 Ethereal 另找译名（如「以太」，正文已有『其他名称：大海、以太、灵界海洋』这一处用了以太）。这条跨 25 个页面，且与同源串分叉那条路线重叠，请统一定夺后一次性做。

【四】顺带发现（不属我这一路，交给同源串分叉/术语统一那条路线）：UUID 显示标签层面还有 39 组同一英文标签对两种中文，其中值得看的有 `Milestone Point` 里程碑点数/里程碑点(66:8)、`Ordain Interiors` 奥尔丹室内景/奥尔丹室内(8:10)、`Quests`/`Events` 互串（Quests→事件、Events→任务 各 2）、`Kessian` 凯西安/凯西亚(18:2)、`Kethil` 凯希尔/基希尔、`Maziran` 马兹兰/马兹兰人、`Waterfall Bridge` 瀑桥/入口桥、`Light Motes` 光尘/光微粒、`Tinderboxes` 引火盒/火绒盒、`Abilities` 属性/能力/属性值。完整表在 C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\892fdb28-d096-4415-94d1-99d12c38ef86\scratchpad\g3\uuidlabels.json（键=英文标签，值={中文:次数}）。

## G10

两条，都超出「只写批次文件」的权限：

1) 【base 词表 + 产物，两个文件都要改】叙事性 boon 被术语表无条件带成机制骰，是本轮 7 条里唯一有系统性成因的一条。
   文件 A: C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json（base 层）
   文件 B: C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\glossary\glossary_ec.json（产物）
   现有键值（两文件相同）：
     "Boon": "恩惠骰"
     "boon": "恩惠骰"
   建议新值：
     "Boon": "恩惠骰（机制）/ 恩赐（叙事）"
     "boon": "恩赐（叙事）；仅 +N Boons 等机制文本用「恩惠骰」"
   理由：Boon 在 Crucible 里是掷骰修正（全库形如 <strong>+2 恩惠骰</strong>），但 EN 同词也大量用作叙事「恩赐」，两者已在包内分化为「恩惠骰 / 恩赐」两套译法（Grave Blade Shore、Switchback Path、Ordain Gazetteer.Temple Ward、Thorny Predicaments 均作恩赐）。当前无条件映射会持续把叙事 boon 译成机制骰，J02 与下条的 Forest of Stone 都是它的产物。若不便写成带注释的值，至少请把小写 "boon" 从 base 层删掉，让 harvest 层的上下文译法胜出。

2) 【不属于 G10 的 J02 范围，需另一路或 parent 收】同一处「恩惠骰」误植在别的日志还剩一份，我按铁律没有越界写它：
   仓库：1-Ember汉化插件；包：ember.adventure.json 与 ember.crucible-adventure.json（孪生，两包各一份）
   batch_path：Ember Early Access.journals.Forest of Stone Gazetteer.pages.Giant's Moonstone.text
   现值片段：「时，它的力量尤其强大，并且能够为恰当同调的角色提供一种独特的恩惠骰。」
   建议新值片段：「时，它的力量尤其强大，并且能够为恰当同调的角色提供一份独特的恩赐。」
   EN 对照："...especially potent when the moon is dominant in the sky, and can provide a unique boon for characters who are duly attuned." —— 纯叙事，无机制加骰。

## G9

两条只能在别处收口的，本车道无法自闭环：
(1) `Spiritlands` 全库三套译法——「灵界」独立用例 37、「灵界荒土」21、「灵魂之地」2。本轮只清掉了世界地图注记那 2 处「灵魂之地」；「灵界荒土」集中在 `Cosmos`（Soul Cycle 等）与 `Deities`（Sockets 等）页，属别的车道。建议由负责 Cosmos/Deities 的车道统一为「灵界」，或反过来确认「灵界荒土」是刻意的地名化写法。
(2) 术语表建议新增/确认两条（`glossary_crucible_merged.json` base 层 + 重建产物 `glossary_ec.json` 都要有，若两仓 compendium/cn 已有证据则只需重建）：
    - `Elderbark` = 古皮树（现有证据：ember.adventure `journals.Cultures.pages.Strider.contentOverview`；本批已把 Geography/Corebright Forest 对齐）
    - `The Untamed Edge` = 蛮荒之缘 / `The Hoarfrost Edge` = 霜冻之缘（与既有 `The Swirling Edge` 旋涡之缘、`The Roaring Edge` 咆哮之缘 成套）
    - `The Spiritlands` = 灵界（用于压掉 harvest 层里可能残留的「灵魂之地」）

## G8

无需要改 lang/ .mjs / glossary 基础词表的项。两点供主控知悉（都不需要动手）：(1) glossary_ec 里 `Kerastes -> 角盔蛇 Kerastes`、`Forsaken War -> 被遗弃的战争 Forsaken War`、`Wardcall -> 守护召唤 Wardcall` 都是从包里收割来的 harvest 层词条，本轮未改包内对应词，重建不会漂移。(2) 本轮把 the Nineteen 的正文用法从「十九神」统一到「十九人」，与页名 `十九人 The Nineteen` 及全部链接标签一致；下次 build_glossary 重建会收割到唯一形，无需手改 base 层。

## G5

## 一、需要你跑的一条命令（我无权写 compendium/cn）

    python "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\3-常用脚本\qa\prune_dead.py" ^
           --repo "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件" --write

影子仓已验证：实删 8、复扫 0、diff 确认只动这 8 条（全部是
`entries.Ember Early Access.actors.Jurtak {Hunter,Warrior}.items.Jurtak Poison._legacyActions.{actionname,actiondesc}.{0,1}`）。
删除理由见 confirmed —— 寄存是真的，但打捞早已完成，Poison Vial 的两个动作现已按 id 建键译好；
且 `_legacyActions` 不在任何 mapping 里，Babele 永不读取。crucible 仓死键本来就是 0，不必跑。

## 二、⚠ 批次合并注意（我的值是整叶快照，取自 21:28 的 cn 现状）

我只碰 4 个叶子，但 G4 / G8 等并行 agent 也在写同两个包。若有人先改了同一个叶子，
直接套用我的整叶新值会把对方的改动抹掉。届时请按下列**子串替换**重放，而不是整叶覆盖：

| 叶子（batch_path） | 旧子串 → 新子串 | 次数 |
|---|---|---|
| `Ember Early Access.journals.Geography.pages.Ordain.text` | `...ThsrAwMhXhmowirO]{幽圣所}` → `...ThsrAwMhXhmowirO]{圣堂区}` | 1 |
| `Ember Early Access.journals.Thorny Predicaments.pages.Overview.text` | `@UUID[Actor.qW0lhTVZLQknOayq]{卡尔敏花药}` → `@UUID[Actor.qW0lhTVZLQknOayq]{卡尔敏·安瑟}` | 2 |
| `Ember Early Access.journals.Ooze Control.pages.Good Ooze, Bad Ooze.text` | `@Embed[Actor.LUptsqBgGJVWcg9v label=Squish]` → `@Embed[Actor.LUptsqBgGJVWcg9v label="压扁"]` | 1 |
| `Ember Early Access.tables.Corpse Loot.results.35-35.name`（仅 crucible-adventure） | `撬棒` → `撬棒 Prybar` | 1 |

前三行两个孪生包各一份，第四行只有 crucible-adventure（adventure 包该格英文是 Piton）。
**`label=` 必须带引号**，不带引号会被 scan_markup_targets 判成 BROKEN（实测）。

## 三、待你拍板的一项（我没改）

`Ember Early Access.journals.Glitter in the Dark.pages.Rumors in Rock Bottom.text`
两个孪生包各一处，英文侧逐字就是上游作者忘了删的编辑批注：
`[Recap of description of Entropic Pearl from item here]`。
中文现在照抄英文，是 scan_en_residue 仅剩的 2 处。若决定改善 GM 阅读体验，可直接用：

    [此处复述物品条目中对熵珍珠的描述]

（单层方括号不是 enricher，不进 markup_signature，也不进 scan_markup_targets 的判据，改动零风险。
我没动是因为铁律 2「方括号内照抄不译」的字面读法涵盖了它，且属"替上游补内容"，该由你定。）

## 四、两处判据本身的改进建议（能永久降噪，我按铁律没动 3-常用脚本）

1. **`qa/scan_name_binding.py`** —— 目前 189 条 table-result UNCERTAIN 里 114 条纯属输入不全。
   建议在 id 解析失败时先看 `documentUuid` 的包前缀：`Compendium.dnd5e.*` 直接判 `OUT_OF_SCOPE`
   而不是 `UNCERTAIN`（脚本已有这个档位，只是够不着）。改完这一项，199 → 85。
   要把剩下 73 条也清掉，需要补两份绑定表：
       node qa/dump_bindings.mjs --package <crucible 系统目录> --out bind_crucible.json
       node qa/dump_bindings.mjs --package <dnd5e 系统目录>    --out bind_dnd5e.json
   然后 `--bindings` 三份一起给。**注意**：脚本对 crucible.equipment 的 Longsword / Blast Flask
   报了「目标文档没有中文名」，但那两条中文名（长剑 Longsword / 爆炸瓶 Blast Flask）明明在
   `crucible.equipment.json` 的 `entries` 下 —— `target_cn` 取值路径有假阴性，顺手一并查。

2. **`qa/scan_uuid_swap.py`** —— 剩余 6 处 UNCERTAIN 全是「(目标 id, 英文标签) 语境键票数不足
   min-support(3)，退回只按目标 id 统计」造成的。建议：退回 target-only 统计时，若该叶自己的
   英文标签与多数票所依据的英文标签**不同**，直接不报（或单列成 `alias-label` 档）。
   本轮 6/6 全是这一形态，改完这一项 uuid_swap 可真正清零。

## 五、可直接贴进 PROJECT.md 的永久豁免清单

### §X 永久豁免（第十二轮 G5 终裁，以后各轮不必重裁）

下列报告项**永远非零**，每一条都在 2026-08-13 逐条查过英文原文与目标文档，
改动它们会引入新缺陷。再次看到时直接跳过。

| 报告 | 数量 | 内容 | 为什么永远非零 |
|---|---|---|---|
| `scan_uuid_swap` UNCERTAIN | 6 | Cleric 页 `{Kessia}`→凯西亚 ×4、`{Ordain}`→奥尔丹 ×2 | 上游把页面挂在相关但不同名的英文标签上：`1WoH2TVw0gngrgWL` 是文化页 **Kessian**，英文原文却写 `{Kessia}`（那是另一块地名 `2eLAE5AF2iAMlc0e`）；`RxhlhTWqJqB1cZxY` 是文化页 **Ordani**，英文原文写 `to the shores of {Ordain}`。中文照译标签是对的。判据在 (目标,英文标签) 票数 <3 时退回只按目标 id 统计，于是拿另一英文标签下的多数来比。**改成多数派＝把英文写着 Kessia/Ordain 的地方译错。** |
| `scan_label_vs_name` | 2 | Giant Moonstone 页 `{Maziran}`→马兹兰人（文档名马兹兰） | 英文在此作人称名词（`an enthusiastic Maziran aspiring to...`），中文非加「人」不可；另 5 处作定语（`the brilliant Maziran swordmaster`→马兹兰剑术大师）故不加。判据只比「英文标签==文档英文名」，看不出语法角色。 |
| `scan_name_splits` | 5 | Shield / Arcturian / Luminous / Spirited / Color Commentary | 见下方逐条 |
| `scan_name_binding` UNCERTAIN | 199 | 114 dnd5e + 73 crucible + 10 上游删页 + 2 假阴性 | 判据输入不全（只给了 bindings_ember.json）+ 上游悬空 pageId。**73 条 crucible 目标已离线逐条核对，标签与装备文档中文 name 全部逐字相同。** |
| `scan_en_residue` | 2 | `[Recap of description of Entropic Pearl from item here]` | 上游留在发布版里的编辑批注，英文侧逐字如此。（可选改善见工单） |
| `prune_dead` crucible | 0 | — | 本来就是 0 |

**`scan_name_splits` 五条逐条：**

* **`Shield` 护盾术×23 | 盾牌×8** —— 同形异义。法术侧英文带 `An imperceptible barrier of magical force protects you`；
  装备侧英文只有 `{"name":"Shield"}` 空壳，挂在近战单位与 Corpse Loot 战利品上。
* **`Arcturian` 阿克图里安×8 | 阿克图里安人×2** —— 与 Maziran 同一条语法规律。
  文化页 / 祖裔条目 / 角色卡文化字段 = 阿克图里安（**49 条 @UUID 标签佐证**）；
  NPC 角色卡「一名阿克图里安人」= 阿克图里安人（**46 条标签 + tokenName 佐证**）。
  ⚠ **2026-08-13 实测：把 NPC 卡改成「阿克图里安」会让 `scan_label_vs_name` 从 2 处涨到 20 处。不要再试。**
  （平行的 `Ordani` 角色卡是「奥尔达尼」不带人，22 条标签自洽 —— 两者各自成体系，也不要去"统一"。）
* **`Luminous` 明光×4 | 辉耀×1** / **`Spirited` 精神焕发×1 | 昂扬×1** —— 两套系统的两种机制，
  且中文语法角色不同：crucible 词缀条目另有 `adjective` 字段，要拼到物品名前当形容词；
  ember 那侧是角色进阶路线的等级特性名，独立成词。
* **`Color Commentary` 彩色解说×2 | 精彩解说×2** —— 英文一语双关，两处无关：
  `Local Color`（地方色彩）下的页面讲批判城市的**壁画**，取颜料义；
  `Helkas Drake Moments` 6-6 是吟游诗人战斗中现场演唱，取**体育解说**义。

## G4

【一】批次撞叶子，必须合并而非后写覆盖（唯一一处，但会静默丢改动）
`Ember Early Access.journals.Yakoshta Mine.pages.Supply Cache.text`（ember.adventure.json 与 ember.crucible-adventure.json 两包都撞）同时出现在 **G4** 和 **G12** 的批次里。两边改的是同一叶子的不同位置、互不冲突：
  G4 ：「贾斯珀钥匙环上的储藏室钥匙」→「贾斯珀的钥匙圈上的储藏室钥匙」（插「的」、「环」→「圈」）
  G12：同叶另一处 「盗贼」→「开锁」、「器」→「工具」
两份 batch 的值都是整叶，谁后 apply 谁抹掉对方。请在合并层把两处编辑叠加后再落盘，或让 G12 的值上再套一次 G4 的两字替换。

【二】glossary_ec 的 base 层孤儿键要改（`C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json` 与产物 `5-其他内容\glossary\glossary_ec.json` 两个文件都要改；这些键是复数/词组形态，harvest 层收不到，重建不会自动纠正）。每条都用包内权威名或全库计数验过：
  "Water Sprites"          : 水元素精灵 -> 水妖精            （actor 名「水妖精 Water Sprite」；库内 38:0）
  "Attunement Progression" : 调谐进阶   -> 同调进阶          （库内 70:0；Attunement 全系一律「同调」）
  "Sea Captain"            : 海船船长   -> 海船队长          （actor「阿克图里安海船队长 Arcturian Sea Captain」）
  "Moon Blossoms"          : 月之花     -> 月华花            （库内 82:4）
  "Construct Fluid"        : 构装体流体 -> 构装流体          （item「艾迪尔构装流体 Aedir Construct Fluid」）
  "Interactive Element"    : 可互动要素 -> 可交互元素        （库内两个英文词条中文都用「可交互元素」）
  "Interactable Objects"   : 可交互物件 -> 可交互元素        （同上）
  "Coins"                  : 通用语     -> 删除或改「钱币」  （「通用语」是 Common，明显串行的坏条目，会污染任何用 glossary 做判据的扫描）

【三】G4 之外发现、我没动（避免与其他单元撞叶子），建议专扫一遍：
1. `Terracini` 译名分叉：`Ordain Gazetteer.pages.Lantern Roads.text` 用「泰拉奇尼」7 次（两个孪生包各 7），而页面名 `Terracini's Tavern`／`Terracini Balcony` 与其余正文共 36 次用「特拉奇尼」。页名优先，建议把该地名志叶子的 7 处统一为「特拉奇尼」（术语表当前值「泰拉奇尼」也应随之改）。
2. `地区地图` 残留 5 叶（各 1 处，两个孪生包）：`Forest of Stone Gazetteer.pages.Verdant Paths.text`、`Forest of Stone Gazetteer.pages.Crystal Carving Cavern.text`、`Arctus Plateau Gazetteer.pages.Redrak Fields.text`、`Ordain Gazetteer.pages.Trident's Point.text`、`Ancient Paths.pages.Emergence.text`。全库规范是「区域地图」998 次。（第 6 叶 The Bard's Trail 我已在 G4 批次里改掉。）
3. `Ordain Gazetteer.pages.All-Fable Keep.text`（两包）：4 条 NPC 阵营标签里有 2 条用半角括号逗号 `(NG, 奥尔达尼 凯思, he/him)`，另 2 条用全角。纯排版，低优先。

【四】上游自身的坏链接，我们改不了、也别再试着补 id：
`@UUID[JournalEntry.rycatIw6IR9KlRhK.JournalEntryPage.0LXzxGaXrMFhFREp#entrance-bridge]` 在 en 与 cn 两侧各出现 2 次，但全库**没有任何标题带 `id="entrance-bridge"`** —— 上游把该标题改名成 Waterfall Bridge（slug 变成 waterfall-bridge）却没同步链接。第十一轮的 id 注入按英文 slug 生成，天然补不上这个。属上游缺陷，记档即可。

【五】工具修正建议：`3-常用脚本/qa/scan_en_drift.py` 的 `stale` 判据（`fits_old`）应当作废或换掉。它把「本库译文/英文纯文本长度比」写死为 0.31，而我在这 1217 条上实测中位数是 **0.396**（10–90 分位 0.301–0.488）——中心就偏了，加上短叶子本无鉴别力，这一列既不准也没用。建议把该列换成本轮的 A 信号（新英文独有的 enricher／锚点／数字／专名 token 是否缺席于中文），筛子实现见 `4-临时脚本/2026-08-13-round12/probes/g4_sieve.py`，可直接搬。

## G7

以下 8 项都不在 K7 单元内、或跨多个单元，我没有动，请主控统一裁决执行：

1）【最重要，跨 5 个单元】`Carnal Dragons` → 现译「肉欲龙」，是误导性译名。英文 carnal 在此指「血肉的/兽性的」——同页 contentOverview 明写 "Carnal dragons are bestial, predatory monsters"，别名是 "Fanged Wyrms"（獠牙古龙），整页讲的是捕猎、领地、体型适应，没有任何情欲含义；中文「肉欲」专指情欲。建议改「血肉龙」或「噬肉龙」。
   影响面：13 个叶子 × 2 包 = 26 处，分布在 Bestiary.pages.Carnal Dragons.{name,text,contentOverview,contentGamemaster}、Bestiary.pages.Cruel Dragons.contentGamemaster、Bestiary.pages.Dragons.text、Ancient Paths.pages.Impossible Skin.text、History.pages.Age of Rediscovery.contentGamemaster、History.pages.Ordain.contentGamemaster、Notable Figures.pages.Ryleir.contentOverview、Notable Figures.pages.Siodread.contentOverview、Organizations.pages.Anachraenum.contentGamemaster、Organizations.pages.Mutagists.contentGamemaster。需一次性改，否则立刻产生同源串分叉。

2）`Ember Early Access.journals.Cosmos.pages.The Abyss.contentGamemaster`（×2 包）把 `Heralds` 译成「先驱」，与全库主流「使者」分叉。建议新值：「其中包括它们的使者，据说其力量足以与余烬的上古诸神比肩。在深渊裂斩期间，只有一个使者直接进入了寰宇……」（同叶两处）。

3）`Ember Early Access.journals.Helkas.pages.Helkas Green.text`（×2 包）把 `aithus` 译成「艾瑟斯树」，与「艾苏斯」（Forest of Stone Gazetteer.pages.Helkas.text、Kalion Stadium Underworks.pages.Bathhouse.text）分叉。建议统一为「艾苏斯」。

4）`Ember Early Access.journals.Ordain Gazetteer.pages.Stonework Hollow.text`（×2 包）把 `Crag Raptor` 译成「峭岩迅爪兽」，全库其余 10 叶（含 items.Crag Raptor Feather.name）均作「峭岩猛禽」。建议统一为「峭岩猛禽」。

5）`Ember Early Access.journals.Arctus Plateau Gazetteer.pages.Lake Jinro Lunar Shrine.text`（×2 包）把叙述性 `a temporary cosmic boon` 译成「一项临时的宇宙恩惠骰」，误加了机制词「骰」。全库只有 `+N Boons` 才是恩惠骰。建议改成「一项临时的宇宙恩惠」。（我单元内同类问题已在 Giant's Moonstone 修掉。）

6）物品名与生物名不同源：`items.Cheliceraeth Eye.name` = 「螯肢眼 Cheliceraeth Eye」，而 `actors.Young Cheliceraeth` / `macros.Toggle Cheliceraeth` / Crucible 侧 `entries.Cheliceraeth.name` = 「螯蛛艾斯」。若要统一，物品名与所有引用它的 @UUID 链接标签（Ember's Bounty.pages.Perlin's Powders.text 等）需同步改，属跨单元批次。

7）`Dry Outpost` → 现译「干旱前哨站」，全库 16 处一致。语义上不对：原文强调它是潮湿曲折峡谷中给旅人遮风避雨的落脚处（"These dwellings still offer a small measure of shelter… a welcome sight for travelers navigating the wet and winding canyon paths"），Dry 指「干爽/避雨」不是气候干旱；所在生物群系「翠绿径/滴石笋」恰恰以超自然湿度著称。建议全库改「干燥前哨站」。影响面跨单元：Forest of Stone Gazetteer.pages.Dry Outpost.{name,text}（我的单元）、The Winding Trail.pages.{Giant Moonstone,Skies Above}.text、Gamemaster's Guide.pages.{Starting the Game,Patch 0.3.2}.text、scenes.Vista: Verdant Paths.levels.{Dry Outpost, Dry Outpost - Caravan Camp}，每项 ×2 包。因跨单元，我未单方面改（只改我这一侧会立刻造成分叉）。

8）设定裁决请求：`Forest of Stone Gazetteer.pages.Verdant Paths.text` 中 "a battle between two giant creatures – a Leviathan and a wurm"，小写 wurm 现译「巨蠕虫」。若设定上 wurm 就是 `Wyrm`（第十一轮定「古龙」），需改；但 Bestiary 的 Wyrms 词条把古龙限定为阿肯之月上的生物，且同页 `Wurmwood Trees` 已作「蠕木树」。请设定层裁一次，我按字面保留了原状。

## G13

以下三类需要在我不能写的文件里改，或跨出 K4 分片，请编排者定夺。

【A】base 词表孤儿键（harvest 层无证据，必须 base 与产物两个文件都改）
文件 1：C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json
文件 2：C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\glossary\glossary_ec.json
  1. 键 "Suspended Animation"：现值「悬滞动画」→ 新值「生命悬滞」。
     理由：包内该短语一律译「生命悬滞」（Vortest Tower 三页的 <h4> 小节名 + 正文，全库 26 处）；「悬滞动画」是把 animation 当成「动画片」的直译残留。该短语不是任何叶的整值（只出现在 .text 内部），harvest 收不到，必须手改 base。
  2. 键 "Orb Room"：现值「宝珠室」→ 新值「法珠室」。
     理由：全库 orb 一律「法珠」（411 处）；本批次已把 Crumbling Walkway 里唯一的 {宝珠室} 改为 {法珠室}。该键同样是叶内标签、harvest 收不到，不同步改 base 会在下次重建时把旧值带回。

【B】K4 之外的一处包内叶（在 actors 下，属他人分片，我未写入）
  仓库 1-Ember汉化插件，pack ember.crucible-adventure.json
  键：Ember Early Access.actors.Temple Invader.items.Elemental Summoning.actions.elementalSummonSprite.name
  现值「召唤精灵 Summon Sprite」→ 建议「召唤妖精 Summon Sprite」
  理由：同一遭遇里 Water Sprite =「水妖精」（actor 名 + Temple Interior 正文全部作「妖精」，第十轮 K4 已专门统一过一次），只有这条动作名残留「精灵」，玩家在同一场战斗里会看到「召唤精灵」召出「水妖精」。同步 glossary_ec 的 "Summon Sprite" 键（现值「召唤精灵 Summon Sprite」）。

【C】跨书术语分叉（我只在 K4 内确认，未越界修改，供编排者判断是否单开一路收口）
  1. Abyssal(s)：全库 深渊实体 55 / 深渊生物 49 / 深渊裔 10，而 glossary_ec 与 base 均定「Abyssals → 深渊裔」。K4 的 Central Lunarium 里 Abyssals→深渊裔（合词表）、Abyssal entities→深渊实体、an ancient abyssal→远古深渊实体 三种并存。三方分叉规模大，不宜由单个分片改。
  2. Heroism（Crucible 资源）：全库 英雄气概 73 / 英勇 29。K4 内用的是多数派「英雄气概」，未动。
  3. Star Mage(s)：全库 星辰法师 135 / 星法师 7。K4 内用「星辰法师」，未动。
  4. The Sleeper：Mythspire Central Room 作「沉眠者」（6 处），The Bleak Archive.Mist-Veiled Approach 与 A Brush With Death.Night Terrors 作「沉睡者」（4 处）。词表无此键。两边各在不同分片，需要一个统一裁决（建议「沉眠者」：同页 slumbers 已用「沉睡」作动词，名词用「沉眠者」可区分）。
  5. Frost：全库 霜冻 140 / 寒霜 27 / 冰霜 20。K4 内 Rune: Frost=符文：霜冻、Frost Elemental=冰霜元素、Frigid Swipe=寒霜挥击 均与各自 actor 实名一致，故 K4 内不算错，但全库口径值得单独定。

## G12

【一、不需要主控代改的（走 harvest 自动生效，仅备案）】
本轮所有改动都落在 compendium/cn 的包内叶子上，`tm/build_glossary.py` 下次重建 glossary_ec 时会自动带出。特别是：
- 键 `Yakoshta Junction Wheel`：现值「雅科什塔路口轮盘 Yakoshta Junction Wheel」，与同表 `Junction`＝「枢纽 Junction」自相矛盾。我已改包内 items.Yakoshta Junction Wheel.name 为「雅科什塔枢纽轮盘 Yakoshta Junction Wheel」，重建后该键自动修正，**不要手改 glossary_ec / base 词表**。
- 无需改 lang/、.mjs、base 词表。

【二、需要主控派发的连带修改（不改会在库内留下新分叉）】
1) `1-Ember汉化插件` :: `ember.crucible-adventure.json`（仅此包有）两处 actor 简介仍用少数派「莫赖娅」，本批已把 Kali's Cottage 6 处改为「莫赖亚」：
   - `Ember Early Access.actors.Kali Andrella.biography.private`：「莫赖娅」→「莫赖亚」（1 处）
   - `Ember Early Access.actors.Rattletrap, the Rickety Man.biography.private`：「莫赖娅」→「莫赖亚」（1 处）
   依据：全库 莫赖亚 113:14，glossary_ec `Moriah Foxhaven`＝莫赖亚·福克斯黑文。
2) 若主控否决我对 `items.Yakoshta Junction Wheel.name` 的改动，请把 Old Ore Pit / Ooze Go Boom! 两叶的「枢纽轮盘」一并回退回「路口轮盘」——半边回退会比现状更糟。

【三、全库级用词分叉，超出 K5，建议单开一轮统一】
- `agrimagic / agrimagical`：「农法」46 次 / 22 叶 vs「农艺魔法」55 次 / 29 叶（还有「农法魔法」混合形）。K5 只沾到 Steed's Smithy 一叶（现为「与农法有关联」），我未单方面改动。建议统一到「农艺魔法」（与已定的 agrimage＝农艺法师同源）。
- `frightened`：恐惧 34 / 恐慌 31 / 害怕 8，三分。dnd5e 状态名应为「恐惧」。K5 内 Hedge Maze 一叶写「陷入恐慌」（同叶另一处对译 afraid 的「害怕」是正确的），因是全库级问题未动。
- `dim light`：微光 208 vs 昏暗光照 46。
- `chasm`：K5 内已由本批统一为「裂谷」；全库其余 30 叶仍是 裂谷/深渊/峡谷/裂隙 四分。

【四、英文原文自身的缺陷（中文侧忠实照译，不建议改中文，供上游反馈）】
1) 死锚点：`@UUID[…#hedge-maze-entrance]`（Kali's Cottage，2 处引用）与 `@UUID[…#entrance-bridge]`（Waterfall Bridge，4 处引用）在英文页里**根本没有对应标题**，只有普通 <p>。Foundry 的页内锚点只索引 h1–h6，给 <p> 补 id 无效，必须新增标题才能修——属于加内容，本轮未做。其余三条同类断链（#secret-passage-entrance / #a-conversation-with-kali / #return-visit）已在中文侧补 id 修好。
2) `Steed's Point.Southern Clearing.text` 英文写 “while examining the shed's interior”，但本页是空地不是棚屋（从 Dilapidated Shed 复制粘贴残留）。中文照译「检查棚屋室内」。
3) `Steed's Point.The Mill.text` 英文前文说 “a single Jurtak warrior”，后文 hazard 块说 “The Jurtak hunter has been cornered”。中文照译。
4) `Steed's Point.Overgrown Trapdoor.text` 说活板门在 “northwestern corner”，`Kali's Cottage.text` 两处说秘密通道来自 “Northeast corner”，英文自相矛盾。中文照译。
5) 同一个 UUID `emberNainSide000.JournalEntryPage.hJn9ofVtP4DxSTLe` 在 `Steed's Point Fields.text` 内被英文标成两个不同名字（The Rickety Man / The Wooden Sentinel）。中文照译为「摇摇欲坠的人」「木制哨兵」。
6) `Overgrown Trapdoor.text` crucible 分支英文断句 “…check using a .”（物品引用丢失）。中文按上下文补成「使用开锁工具」。