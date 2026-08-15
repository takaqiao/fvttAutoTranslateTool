# 第九轮：需裁决/后续的事项

## leg1

【必须由主控落手：lang/cn.json 与 scripts/ember-hardcoded-cn.mjs（铁律 1 禁写）】

知识领域 7 词逐词裁决（三通道现状 + 英文闸 + 结论）。compendium 侧新增的第四个通道：`2-Crucible汉化插件/crucible.rules.json :: Character Mechanics.pages.Background.text` 的背景示例表也逐格列了知识领域名 —— 第八轮不知道它的存在，它推翻了「compendium 表格自洽」这个前提（Forensics 在两张表里就是两个词）。

1) Artifacts —— 表格 遗物 / lang·mjs 神器 / glossary 古器物。**裁：神器**。理由：`遗物` 已被 Relic 占死（英文闸 Relic→遗物 182 叶 500 处，vs 神器 18 叶），知识领域再用「遗物」是实义撞车；Artifact 散文侧 神器 9 叶 > 古物 6 叶；lang 已对，只需改 3 叶表格（已在我批次里）。**lang/mjs 无需改动。**
2) Crime —— 表格·背景表·glossary 全是 犯罪 / lang·mjs 罪行。英文闸 crime→犯罪 60 叶 87 处 vs 罪行 40 叶 48 处。**裁：犯罪。→ 改 `KNOWLEDGE.Crime`：罪行 → 犯罪（lang/cn.json 与 mjs 各一处）。**
3) Forensics —— Ember 表 法证学(2 叶) / 背景表 鉴识学(1 叶) / lang·mjs 法医学(全库 0 叶) / glossary 法医。compendium 自身分裂故无权威；151 处 `[[/knowledge forensics]]` 的实际语境是石祭坛、厨房霉变食物、房间字条这类痕迹勘验，**不是尸检**，「法医学」不忠实于英文（阶梯第 4 档）；两位独立译者都落在「法证/鉴识」这一系。**裁：法证学。→ 改 `KNOWLEDGE.Forensics`：法医学 → 法证学。**（这一条与第八轮 g00「建议按 lang」相反，依据是新发现的第四通道 + 渲染点语境。）
4) Intrigue —— 表格·glossary 权谋 / lang·mjs 阴谋。英文闸：conspiracy→阴谋 22 叶 42 处、权谋 0 叶，lang 的「阴谋」会与 conspiracy 撞车。**裁：权谋。→ 改 `KNOWLEDGE.Intrigue`：阴谋 → 权谋。**
5) Legends —— 表格·背景表·glossary 传说 / lang·mjs 传奇。英文闸 legend(s)→传说 232 叶 373 处 vs 传奇 83 叶 120 处。**裁：传说。→ 改 `KNOWLEDGE.Legends`：传奇 → 传说。**
6) Machines —— 表格·背景表·glossary 机械 / lang·mjs 机械装置。英文闸 machine(s)→机械 84 叶 183 处 vs 机械装置 13 叶 13 处。**裁：机械。→ 改 `KNOWLEDGE.Machines`：机械装置 → 机械。**
7) Undeath —— 表格·背景表 不死 / lang·mjs 亡灵化 / glossary 亡灵。英文闸 Undeath→不死 94 叶 198 处 vs 亡灵化 4 叶 4 处（全库「亡灵化」只有那 4 叶）。**裁：不死。→ 改 `KNOWLEDGE.Undeath`：亡灵化 → 不死。**

即：lang/cn.json 与 ember-hardcoded-cn.mjs 各改同样 6 个键 —— Crime 犯罪 / Forensics 法证学 / Intrigue 权谋 / Legends 传说 / Machines 机械 / Undeath 不死；Artifacts 保持 神器。（mjs 注释里那句「译名与 crucible lang 的 KNOWLEDGE.* 逐条对齐，改一处要两边一起改」仍然成立。）

【glossary_ec.json 需同步订正（不在我写盘范围）】
- `knowledge artifacts`：古器物知识 → 神器知识
- `knowledge forensics`：法医知识 → 法证学知识
- `knowledge undeath`：亡灵知识 → 不死知识
- （`knowledge crime/intrigue/legends/machines` 已与裁决一致，无需动）
- Marlstone 相关键：本轮已把全部复合专名统一到「马尔石」词根，glossary_ec 里凡值含「马尔斯通」的条目需一并改成「马尔石」，否则下一轮会被当权威反向回灌。
- Mazira：新增/订正为「马兹拉 Mazira」（与既有「马兹兰 Maziran」同词干）。
- Sunalins：统一为「苏纳林诸神」（`Sunalin`→「苏纳林 Sunalin」保持不变）。
- 第八轮 g03-4 那两条仍未处理：键 `Liliman’s Bar Grille` 改为 `Liliman’s Bar &amp; Grille`、值改「莉莉曼酒吧烧烤馆」。

【仍待裁但我未执行（证据已备齐，见 skipped_detail）】
- The Nineteen 十九人(60 叶) vs 十九神(12 叶)：name 叶与多数派都在「十九人」，语义在「十九神」；跨 name 层改名，请单独立项，并同时订正 glossary_ec 并存的两条。
- Protector=守卫者 / Guardian=守护者：方向我认同 g10 的建议，但双向串味（Protector 55 叶中 18 叶用守护者；Guardian 34 叶中 4 叶用守卫者）且常同叶并存，需逐处配对英文 + 改两个 archetype.name，请派一路专做。
- Lesser Restoration：是政策题（dnd5e 侧法术名是否一律跟 5e 官方中文包），非术语题。
- Ossuary 四写法：需先定复合专名策略；`Ossuary Loot` 在英文侧无引用，归属判不出。

【命名/管线提醒】
- 本轮批次文件名两种惯例并存：我按 BRIEF 字面与第八轮惯例用双 `.json` 尾（`leg1.1.ember.adventure.json.json`），leg2 与全部 t1_*/t2_* 分片用单 `.json` 尾（`t1_0.1.ember.adventure.json`）。收集脚本需兼容，或统一重命名后再落盘。
- 撞车预警：本路的 Marlstone 全量替换覆盖 `ember.adventure.json` / `ember.crucible-adventure.json` 共 139 叶（含 Marlstone Manor 整本 journal、Disgraced House 全组、Lantern Roads、两张 scene name），Hallows 修正覆盖另外 66 叶，两者与 g05/g11 曾点名的 `Shent Moon Temple.pages.Temple Interior.text`、`Players' Guide` 系列有交集。我的批次值是**建批时刻的整叶快照**，若同叶被别路改过，必须三方合并；好在我的改动全是确定性字符串替换（马尔斯通→马尔石、指组织的圣堂区→幽圣所），可在合并后的 CN 上用 `4-临时脚本/.../scratchpad/build_leg1.py` 的规则表幂等重放。
- 顺带证实一条方法论：`term_gate.py` / `pair_dump.py` 在 Git Bash 下传中文目录名会被字符集打烂（第八轮 g00-3 已报）。本路全部脚本改用 PowerShell + 绝对路径 + PYTHONIOENCODING=utf-8，并先把两仓 40058 条 EN/CN 叶对缓存成 pickle 再查询，避免每次重扫。

## leg2

1) **PROJECT.md 有三处因本 leg 而过期，需主控更新**（我无该文件的改动授权）：
   · 第 1 节「⚠ 两个『不是 0 但也不要动』的」：① scan_attr_text 113 处 `gained {id}` → 判据已加白名单，现为 **0**；
     ② scan_uuid_swap UNCERTAIN 68 → 已改逐位对齐，现为 **12**。这两条注记再留着会误导下一轮。
   · 缺陷表 **Z3 可以清掉**（上游笔误补丁已打 + LOCAL-PATCHES.md 第 3 条已记 + 批次已出）。
   · 盲区表 A 行「剩 37 处全是假阳性/已裁 deferred」**是错的**：三类 FP 共 17 处已由判据排掉，
     剩下的 20 处是真缺陷（Patch 页本作内容名），本 leg 已修。建议改写成「闸已归 0，但 Patch 页仍有
     --min-words 2 看不见的单词专名 backlog」。

2) **Patch 页需要一条独立 leg**。0.4.2/0.4.3/0.4.7 三页上仍有十几处裸英文（Ancara / Ortarec Cube /
   Magical Forces / Mixed Ancestries / Cartographer / First Soulmark / Vertical Hand / Preserve World State /
   Initiate Event ⋯），`--min-words 2` 结构性地看不见（多是单词）。**闸门归 0 ≠ 这几页干净** ——
   这正是方法教训 1 的又一个实例。另外 0.4.x 之外还有十几个 Patch 页我没看。

3) **三处需要裁决、本 leg 未改**（详见 real_defects_found 末三条）：
   Sunalins 三分（苏纳林斯 6 / 苏纳林 4 / 苏纳林诸神 2，同目标同英文标签）——
   这是 uuid_swap 68→12 过程中唯一被一起消掉的真缺陷，判据现在报不出来了，**必须人接手**；
   Signaran→西格纳拉 应作 西格纳兰（判据修好后新捞到）；Activating Alerts→触发警报 vs 启动警报。

4) **一条给后续轮次的判据教训**：旧 `scan_uuid_swap` 对 `Shard God` 的建议是「碎片之神 → 碎片诸神」，
   而定译表写死 `Shard God`＝**碎片之神**。也就是说**这个判据当时在建议一个违反定译表的改动** ——
   谁照它的 `suggested` 批量改就会把 30 叶正确译文改坏。逐位对齐后该建议消失。
   推论：任何带 `suggested` 的判据，落盘前都应先过一遍定译表的英文闸。

5) 我改的三个 qa 脚本都保留了退路与可见性，主控复核时可用：
   `scan_attr_text --strict-id` 关掉 id 白名单（复现 113）；
   逆向重建的改前版 uuid_swap 留在
   `C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\892fdb28-d096-4415-94d1-99d12c38ef86\scratchpad\leg2\scan_uuid_swap.OLD.py`
   （已验证能精确复现 UNCERTAIN 68），可直接做 A/B 对照。

## t1_0

三条超出本分片、需要全库统一裁决的事，我按「宁可少改不可错改」都没动：

1. **Monstrosity：怪物 还是 畸怪？** 系统分类包 crucible.taxonomy.json 的 `folders.Monstrosity` = 畸怪（同包 Undead=不死生物、Ooze=软泥怪、Outsiders=外来者 都与正文一致），但正文里 怪物 14 : 畸怪 8 反过来。我这片 G2「Gesture: Sense」的生物类型清单 8 叶全写 怪物 —— 如果只改我这 8 叶会把全库比数变成 6:16，反而更乱。更刺眼的是 ember.crucible-adversary.json 一个包里同时有 `folders.Monstrosity` = 畸怪 和 `folders.Monstrosities` = 怪物 Monstrosities。需要一次全库裁决 + 一次性替换。

2. **Lightning 在 crucible 侧**。定译表说 Electricity=电击、Lightning=闪电，glossary_ec 也是 Lightning→闪电。但 G57「Lunar Shield」的英文写的是 `{Aura} - Lightning`，而 crucible 系统里**根本没有 Lightning 这个伤害类型**，只有 DAMAGE.Electricity=电击；同一份清单里其余五项（Acid/Fire/Cold/Psychic/Piercing）都严格对应 crucible 的伤害类型名。两个变体都写 电击。我判断照定译改成「闪电」会让玩家在描述里读到「闪电」、在抗性 UI 里看到「电击」，反而更糟，所以只修了那条裸英文 Ember，电击 原样保留。请确认这个判断。

3. **dnd5e 侧 Fire = 火元素**。G88/G98/G120 里 fire damage 一律译作「火元素伤害」，所有变体一致，但 crucible 的 DAMAGE.Fire = 火焰。这些是塞在 crucible-adventure 包里的 dnd5e 法术文本，可能是 dnd5e 侧的既定译法。因为没有变体分歧、也不在定译表里，我没碰。若要统一成 火焰 需要单独一轮。

另外三处顺手发现、不在我分片里的小缺陷，供主控扫尾：
- `ember.adventure.json` / `ember.crucible-adventure.json` 的 `folders.Local Color` = 「地域风貌」，是该包 145 个 folder 里唯一没带英文的（其余 144 个双语）。
- `ember.crucible-adversary.json` 的 `folders.Monstrosity` = 「畸怪」，也是该包 22 个 folder 里唯一纯中文的。
- 有 3 个 tokenName 带着英文尾巴，逆 269:3 的规约：'法师 Mage'、'螯蛛艾斯 Cheliceraeth'（ember.adventure）、'晶巉巨像 Crystallath'（ember.crucible-adventure）。

给其他分片的一条操作提示：分片 `paths` 里的 `effects[0].name` 直接当 batch_path 用会被 apply_translations 判成 "no English source"（`[0]` 被当字面 key），必须写成 `effects.0.name`。我这批第一次跑就吃了 14 条，改完 0 拒绝。

## t1_1

1) **「字段惯例」应该固化成判据，否则 same_en_split 每轮都会把它们当分叉报出来。**
本分片 131 组里 **48 组（约 37%）** 的所谓分叉，实质只是「双语并列 vs 纯中文」，而这由**字段类型 + 仓库**决定，不是缺陷。我在两个仓库 24 个 cn 包上实测出的惯例（分母是含中文、≤60 字符的叶）：
   - **双语并列**：`items.*.name` 7797/7810 · `pages.*.name` 3095/3096 · `effects.*.name` 2440/2452 · `actors.*.name` 871/871 · `journals.*.name` 151/151 · 顶层 `scenes.<X>.name`
   - **纯中文**：`*.levels` 0/517 · `categories` 10/552 · `*.adjective` 0/172 · `tokens` 0/14 · `tokenName`（ember 侧 4/537）· `navName` 0/2 · `scenes.*.regions.*` 
   - **`folders` 按仓库分裂**：ember.adventure / ember.crucible-adventure / ember.character / ember.crucible-adversary / ember.crucible-character = **双语**（144/145、21/22、18/18…）；**crucible 全部包 + ember.crucible-items + ember.crucible-effects = 纯中文（0/47、0/31、0/19、0/18…）**
   - **`actions.*.name` 惯例是双语**（crucible.talent 174/174、pregens 100/100、equipment 41/41、adversary 41/41），只有 ember.crucible-adventure 掉队（704/1088）—— 这一类我判为真缺陷并统一了（10 组）。
   建议把上表写进 `same_en_split` 的分组器，跨字段类型的分叉直接不入表。

2) **1 条改不了，仍会分叉**：`2-Crucible / crucible.adversary-talents.json` 的 `Acid Spit.actions.acidSpit.effects[0].name`（现值「酸液喷吐」，应为「酸液喷吐 Acid Spit」）。原因是 **英文基准里没有 `actions.*.effects[].name` 这条路径**（PROJECT.md §盲区 9「抽取器根本没抽的字段」的又一处），`apply_translations` 判 `no English source` 一律拒。同类路径本分片共 9 条，另 8 条现值恰好已是目标值。**要修得先补抽取器白名单。**

3) **`Luxaurum` 的中文名需要项目主裁定。** 全库只有两处 `Luxaurum`（Kynryth 解剖 + History/Withering 页），**两处英文都原样留着没译**。glossary_ec 里有 `Luxarum` ＝「卢克萨鲁姆」（ember.character 的 folders 也是这个），拼写差一个字母，我按同一天体处理、统一写成「卢克萨鲁姆」。如果判定两者不是同一个东西，这一叶要回滚。顺带：Withering 页那处裸英文不在本分片定义域，但同样该修。

4) **`Gesture: Ward` 的定译，glossary 与包内不一致的方向是反的，值得复查全表。** `glossary_ec.json` 里 `Arrow`＝「箭头」，而包内 name 字段已经是「手势：箭矢」——**glossary 比包旧**。我一律以包内 `name` 字段为准（`Gesture: Ward`＝手势：防护、`Gesture: Aspect`＝手势：化相）。建议跑一次 glossary ↔ 包内 name 的反向对齐，否则下一个 agent 查 glossary 会得到过期定译。

5) **`spell slot level above N` 全库「级 / 环」混用**（Invisibility/Fly/Cure Wounds 用「级」，Acid Arrow/Lightning Bolt/Enhance Ability 用「环」）。这不是同一条英文串，不在本轮分叉表里，所以没人会报出来，但玩家同屏能看到。属独立的 dnd5e 侧统稿项。

## t1_2

两点想请主控定夺。① `actions.<x>.name` 全库 946 双语 : 399 裸，**没有干净约定**（ember 内嵌 458:375 几乎对半，crucible 独立条目 307:4 压倒性双语）。我采用的判据是「跟本组同角色的权威副本走」：Motivate(13 叶) 与 Dawn Beacon(2 叶) 因 crucible.talent/crucible.spell 的独立条目是双语而改成双语，Prescient Reflection 因权威副本是裸中文而不动。如果主控打算给 actions.name 立一条全库通则，这 15 叶要跟着通则重判。② crucible.equipment.json 的顶层 `label` 是「装备\n**Equipment**」——包标签里带换行，几乎肯定是错的，但 apply_translations 只认 `entries.` 和 `(folders).` 两个根，我无法把它写进批次。

## t1_3

1) 温度档位名跨条目分裂（超出「同英文串」判据，需要单独裁一次）：权威规则页 crucible.rules.json / Exploration / Temperature 定的是 Gelid 极寒 · Cool 凉爽 · Temperate 适中 · Warm 温暖 · Boiling 灼热；而引用这些档位的 Thermal Vision（crucible.adversary-talents.json 独立条目）与 Ashka Lineage（ember.crucible-character.json 独立条目）用的是 冰寒 · 凉冷 · 温热 · 沸腾。两处英文串不同，同英文串分叉判据看不见。本轮 G4 我按「与同侧权威独立条目一致」取了 冰寒/凉冷/温热/沸腾，但正确的方向多半是反过来把它们拉向规则页。
2) crucible.talent.json 的包级 label 中文是 "天赋\nTalents"（中间是真的换行符），其余两个包级 label 是 "余烬冒险 Ember Adventure" 这种空格分隔。这是真缺陷，但顶层 label 路径 apply_translations 无法寻址（parts[0] 不是 (folders) 时它只查 entries.*），需要主控用别的手段改。
3) apply_translations 的路径覆盖洞：顶层 folders.X 只能用 (folders).X 寻址，顶层 label 完全无法寻址。本轮 same_en_split 分片里直接给出的路径是 folders.X 形式，任何 agent 照抄进批次都会拿到 "no English source at this path"。建议把这条写进 RUNBOOK，或让分片生成器把顶层 folders 路径预先改写成 (folders).X。
4) 还开着的跨英文串术语分裂（本轮不在判据内，全库计数为证）：Cantrip Upgrade 戏法强化 11 : 戏法升级 16（我按更接近英文的「升级」统一了本片内的 2 组）· dnd5e Fire 火元素伤害 31 : 火焰伤害 55（本片 3 个变体一致写「火元素」，没动）· Psychic 心灵伤害 44 : 灵能伤害 22（lang DAMAGE.Psychic = 灵能，说明「心灵伤害」那 44 处才是要改的一边）· reduced-cost 减耗 7 : 降低消耗 17 · foot 单位「尺」与「英尺」混用（本片内已就近取英尺）。
5) 判据建议：本片 131 组里有 51 组（约 39%）是**字段角色格式分裂**（name 双语 vs tokenName/levels/categories/tokens/adjective/behaviors.name/repo2 folders 裸中文），全是假阳性。建议 scan_same_en_split 增加一道「按字段角色归一化后再比」的过滤——剥掉双语英文尾巴后若各变体逐字相同，且分叉正好沿角色边界，就不报。这一条能把全库 1514 组里的一大块噪声直接消掉。

## t1_4

两件建议主控接手的事（都超出本分片，我没动）：

1. **`actions.*.name` 的双语尾巴，全库还差约 330 叶。** 判据：crucible.talent.json 131/131 双语、repo2 合计 422 双语 : 7 裸，而 ember.crucible-adventure 471 双语 : 379 裸、ember.crucible-character 27 : 13。我这一片只按「同英文串」捞到 47 叶（Condense/Offhand Strike/Servitor Sending/Rapid Reload/Regulated Rhythm/Sudden Bite）。剩下的英文串各不相同，`same_en_split` 判据结构上看不见，需要一条**按字段角色比对双语格式**的新判据（角色→格式约定表：`*.name` 双语；`adjective`/`levels`/`tokenName`/`navName` 裸中文；`folders` 与 `categories` 两仓库相反）。

2. **`folders` / `categories` 两个角色在两个仓库里的约定是反的。** repo1：folders 332 双语 : 23 裸、categories 0 : 542 裸；repo2：folders 0 : 160 裸、categories 10 双语 : 0。我按「各自合规」放过了 5 组，但这本身是个待裁决项——合集侧边栏与日志目录里玩家会同屏看到两种风格。涉及约 1067 叶，需要项目所有者定一个方向。

另外两条我按现状保留、请主控确认是否要立判据：
- 权威条目（crucible.talent.json / crucible.ancestry.json 等）里有约 600 处 `中文 <strong>词</strong> 中文` 的半角空格填充（全库 space 603 : nospace 8812）。我这次**原样保留**了权威文本、没有顺手去空格，以免把「统一到权威」变成「统一到我改写过的权威」。如果主控要清，应该是一条独立的机械判据。
- `Anatomy` 这个 `<h2|h3 class="divider">` 小标题全库 4 种译法：构造 53 / 身体构造 41 / 解剖结构 24 / 解剖特征 4。我在长正文组里只做组内统一，没有跨组归一，否则会和别的 agent 的分片打架。

## t1_5

1) **本片最大的一类分叉是「双语并列尾巴」，我按合法分叉跳过了 51 组（约 300 叶）。** 判据不是感觉：`qa/scan_token_name.py` 的 docstring 写明「tokenName 取 name 去掉双语英文尾巴后的中文头，库内约定是裸中文 533/537」；PROJECT.md 2026-08-13b 又明令 `name`（双语「辉耀 Luminary」）与 `adjective`（拼装用裸中文）不得互相传播。数据也印证：`.name` 双语 / `tokenName`·`adjective`·`scenes.levels`·`regions`·journal `categories` 裸中文，20 多组沿同一条轴线整齐分裂，不是随机漂移。**但 folders 与 journal categories 这两类我拿不准**——glossary_ec 里 Ancestries/Creatures/Beasts/Tools/Jewelry/Miscellaneous/Accessories/Armor/Outer Gods/Elevator **全是双语形**，若主控裁定「folders/categories 也统一到 glossary 的双语形」，这十来组可以一次机械统一。这是风格裁决，不该由分片 agent 定。

2) **我确实动了 `actions.*.name` 的双语尾巴（5 组 / 37 叶：Mould、Enkindle、Living Stone、Berserker Rage、Formidable Stamina）。** 依据是阶梯第 1 条（独立条目 + 2-Crucible 优先：`crucible.talent.json` 的 `Rune: Earth.actions.mould.name` 就是「塑形 Mould」）加 glossary_ec 三条词条本身就是双语。若项目实际约定是「动作名裸中文」，请整体回退这 5 组——它们在批次里很好摘（值以 ` Mould`/` Enkindle`/` Living Stone`/` Berserker Rage`/` Formidable Stamina` 结尾）。

3) **`crucible.ancestry.json` 的顶层 `label` = 「血统\n Ancestries」（真有个换行符）写不进去**：`apply_translations.py` 只把路径挂到 `entries` 或 `(folders)` 两个根上，顶层标量键没有入口。请主控手改，或给工具加一个 `(label)` 根。

4) **两个「全库规模、我没敢动」的错译，建议另开条目**：① dnd5e 侧把 Sphere 译成 **天球**（全库 40 处；天球是天文学的 celestial sphere，正确应是球体/球形区域，库内另有 球体 85 / 球形区域 15），本片 G125 两份都是天球，我保留了；② glossary_ec 里 **Surface=地表**（应是场景层名沿用下来的），但 Pinning Shot 的 "pin your target against a nearby surface" 处「地表」语义不对（钉在墙面上也算 surface），我按 glossary 保留了权威条目的「地表」。

5) 顺带一条方法坑给下一轮：分片 `paths` 里的数组下标是 `effects[0]` 形式，而 `apply_translations.py` 的 `split_path` 只按点号走列表，直接照抄会得到 `REJECTED no-EN`。生成批次时要把 `[N]` 改写成 `.N`。

## t1_6

三条给主控的建议（都超出本分片范围）：
(1) glossary_ec.json 有至少三条与实际库相反的过期/污染条目，会持续污染后续翻译：`Study -> 书房 Study`（庄园房间名被当成 dnd5e 的 Study 动作，已造成全库 12 处「书房动作」，我这片修了 6 处，另 6 处在别的分片或未被本轮分组覆盖）；`Inflection: Compose -> 屈折：作曲`（库里实际是「屈折：编构」，41:2）；`Local Color -> 地域风貌`（库里实际是「地方色彩」，40:2）。建议单独跑一遍 glossary 与 packs 的反向核对。
(2) 本轮分组把「字段约定差异」当成了分叉。我这片 131 组里有 52 组（40%）纯粹是 name(双语并列) vs tokenName/adjective/categories/levels/tokens(纯中文) 或 folders 的按包约定。这些是合法的，但会在每一轮都被重新报出来。建议给 same_en_split 的生成脚本加一个「按叶子字段种类分桶后再比」的开关，实测阈值很干净：tokenName 575/19、adjective 172/0、categories 552/10、levels 517/8、tokens 14/0 带拉丁；name 19892/17517 带拉丁；folders 按包分（1-Ember 的 adventure/character/adversary 包双语，其余纯中文）。
(3) `actions.*.name` 的双语并列在 ember.crucible-adventure.json 只有 55% 覆盖（850 条中 471 条），其他所有包都是 91%~100%。我这片只修到了同英文串分叉暴露出的 41 叶，该包剩下的约 340 条纯中文 actions 名还没人管，不会被 same_en_split 发现（它们的英文串在别处没有第二份译法）。建议单开一个按字段约定的扫描任务。

## t1_7

【最重要的方法论发现：本轮"裸英文残留"标记的绝大多数是字段角色约定，不是缺陷】
我先用全库统计标定了每个字段角色的双语/裸中文约定（脚本思路：对 EN 为纯 ASCII 的叶子按路径角色分桶，统计中文侧是否带英文尾巴）：
  tokenName 裸 556:19 · adjective 裸 172:0 · categories 裸 542:10 · levels 裸 509:8 · scenes.tokens 裸 14:0 · regions.name 裸
  page.name 双语 3095:1 · journal.name 双语 151:0 · item.name 双语 1183:0 · actor.name 双语 7517:13 · effect.name 双语 2786:30
  folders：entries.*.folders.* 双语 144:1，包级 folders.* 在 crucible 侧全裸、在 ember 侧多为双语（按包各自一致）
  action.name 全库 946:399，但 crucible 系统包（talent/pregens/playtest/equipment/affixes/spell/summons/adversary-talents）几乎 100% 双语 —— 所以双语是约定，ember.crucible-adventure 的 379 条裸中文是漂移
这条判据把我这片 49 组"裸英文残留"里的 47 组判成合法分叉（例：`Rask` = actor.name 双语 vs tokenName 裸中文）。**建议把这张表写进 PROJECT.md，并让 same_en_split 的下一版按字段角色先分桶再报**，否则每轮都会重复报这 1000+ 叶假阳性。

【另一条包级约定，差点误改】`crucible.rules.json` 的 categories 是**双语**（entries.Conditions.categories.* 全部 8 条双语），与 ember 冒险包的裸中文相反。我最初把 `Overview` 组的 crucible.rules 那条也拉成裸中文，发现后回退了。判据必须按"包内一致"而不是"全库一致"。

【留给主控裁决的一条术语】`Hallows`（我 skip 了）：`Organizations.pages.Hallows` 译"幽圣所"（组织），`Ordain Gazetteer.pages.The Hallows` / `Vista: The Hallows` 译"圣堂区"（奥尔丹的一个街区）。英文里两者其实是同一个东西（民政机构与其所在街区同名）。全库 幽圣所 262 : 圣堂区 245，需要单独设闸统一，不适合在同英文分叉里顺手改。

【一处我做了跨叶判断，主控可复核】`crucible.affixes.json` 全包用"尺"（12 处），`crucible.equipment.json` 全包用"英尺"（39 处），而 Luminous 前缀描述在两包里逐字相同只差这个单位词。我按全库 英尺 2487 处的标准统一到"英尺"，代价是 affixes.json 里 Gliding 前缀仍写"尺" —— 那是本组之外的既有不一致。

## t1_8

需要主控做全库裁决的四件事（都超出单个分组的范围，我在片内只做了不留后患的处理）：

1. **`actions.<key>.name` 的双语 / 裸中文没有统一约定** —— 这是我这片改动量最大的一块（8 组约 46 叶）。数据：crucible 侧权威包近乎全双语（crucible.talent 131:0、pregens 82:0、playtest 115:3、equipment 30:0、spell 15:0、adversary-talents 42:4），而 ember.crucible-adventure.json 是 471 双语 : 379 裸，`ember.crucible-character.json` 27:13——**同一个 actor Sigil 身上就同时有「反制法术 Counterspell」和「揭示」**，是译批漂移不是角色约定。我按「独立条目优先 + 2-Crucible 优先 + items.name 99.8% 双语」统一到双语。全库还剩 ~330 叶同类裸中文没动，要么主控整批推平，要么反过来推翻我这 46 叶——**两种都行，但不能只做一半**。

2. **glossary_ec 有一条被地点名污染的条目：`Study → 书房 Study`**。它来自 `Study Chamber 书房 Study Chamber` 这个房间名，却会让 dnd5e 2024 的 **Study 动作**被译成「书房动作」（第 120 组两叶就是这么坏的）。我按库内既有的 2024 动作译法（搜索动作／利用动作／疾走／脱离／隐藏）补成「**研习动作**」——这是本片唯一一个我新造的术语，请确认。建议同时把 glossary_ec 那条改成带上下文标注。

3. **`Restrained` 在 dnd5e 侧到底叫什么没定**：全库 受缚 139 / 受拘束 58 / 束缚 354，BRIEF 的定译表写「受缚」但那是 crucible 系统状态。第 102、119 两组的 dnd5e 正文都写「受拘束」，我**没有动这个词**（只统一了其余措辞），因为 `&Reference[Restrained]` 是 dnd5e 自己的 enricher，正文用词得跟它渲染出来的一致，而那个译名不在本仓库里。需要先确认 dnd5e 中文包渲染成什么。

4. **`尺` vs `英尺`**：全库 2197 : 318，`英尺` 是事实约定，但 crucible.talent / adversary-talents 里若干**权威条目**用的是「尺」（第 74、81、116、124 组都因此没取权威那份，或取了权威但把「尺」改成「英尺」）。这是我唯一一处在没有语义缺陷时也动了字的地方，只在本来就要重写该叶时顺手改。要不要全库推一遍请主控定。

另：分片 JSON 里的数组下标写成 `effects[0].name`，而 `apply_translations.py` 只认 `effects.0.name`（前者会报 `no English source`）。我在批次里做了规格化，其它分片的 agent 大概率会撞到同一个坑。

## t2_0

四条需要全库级裁决、超出本分片范围的：

1) Sentinel = 哨兵 还是 哨卫（全库分裂，且分裂在同一只怪身上）。actor「Broken Aedir Sentinel」= 损坏的艾迪尔哨兵、archetype「Aedir Sentinel」= 艾迪尔哨兵，但同一只怪的能力描述（Alchemical Recharge / Powered Spin / Repelling Kick）全用「哨卫」，crucible.talent 的独立条目「Sentinel」= 哨卫，「Silver Beam Sentinel」= 银光束哨卫。全库计数 哨兵 51 : 哨卫 47。本分片 #91 我取了「哨卫」以与同批 #68/#141 一致，但 actor 名仍是「哨兵」，需要一次统一裁决 + 全库替换。

2) Hydroxol = 羟氧醇 还是 羟醇。glossary_ec 定 Hydroxol=羟氧醇，怪物描述里也用羟氧醇，但物品名在两个包里一致地写成「羟醇皮革 Hydroxol Hide」「灰色羟醇皮革 Gray Hydroxol Hide」。因为同一条英文串内部两个变体是一致的（不构成分叉），我没有动它——但这是一个跨条目的真实术语不一致，改要连物品名一起改。

3) 「X Journeyman」技能天赋的描述模板全库有 22 种写法：「你获得X技能的…」/「你在X技能中获得…」/「你的X技能获得…」，技能名有的加 <strong> 有的不加，「熟练工」「等级」有的各自加粗有的合并，「+1 技能加值」有的带空格有的不带。本轮按组只能各自统一（#49/#119/#121 都取了 crucible.talent 独立条目），建议单开一个模板统一任务，否则下一轮同样的分叉会以别的英文串再冒出来。

4) 「重击」建议全库扫一遍：Critical 的定译是「暴击」（全库 暴击 204 : 重击 48），本分片 #125 就撞上了两个变体都写「重击」的情况（已自行写正确版）。48 处「重击」里有多少是 Critical 误译、有多少是正常的「沉重一击」，需要一次带英文对照的扫描。

另：本分片有 3 组的正确译文只能靠「同一包内自证」判定——#40 的动作名（描述引用了一个包里根本不存在的动作名）、#153 的手势名、#21 的同句自相矛盾。这类「描述里引用的名字 ≠ 该对象自己的 name 字段」是现有扫描完全看不见的一类缺陷，值得单独做一个交叉校验工具（把 description 里出现的加粗/书名号短语拿去比对同包 name 字段）。

## t2_1

三点建议主控接手：(1) 本片捞到的三个定译错是**全库性**的，我只能改到分片内的叶：`疯狂阈值`→集结阈值 全库 14 处（我改 3）、`创伤阈值`→治疗阈值 全库 7 处（我改 4）、`烬境`→余烬 全库 16 处（我改 1）、`炼金术师`→炼金术士 全库 29 处（我改 1）。这四条都可以直接用 propagate_fix/全库替换收掉，不需要再分片。(2) apply_translations.py 对 `folders.*` 路径要求 batch_path 写成 `(folders).X`，分片里给的是裸 `folders.X`，其它分片如果有 folder 叶会静默吃 REJECTED no-EN —— 建议在分片生成侧统一改写。`label` 这一级 applier 完全没有写入路径，若有 label 分叉只能人工处理。(3) 反复出现的模式：**权威条目本身是错的那一侧**（Gathering Storm 的「增益」、Unarmed Blocking 的漏译从句、Jahud 的「西格纳拉」、三条 Journeyman 的 strong 错位，都在 crucible.talent.json / crucible-adversary.json 这类「独立条目」里）。后续分片的依据阶梯里，第 1 条「独立条目优先」应当明确让位于第 2/4 条，否则会把权威包里的错误反向推广到全库副本。
