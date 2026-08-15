# 第十轮 B 段 escalate

## T4

【一】glossary_ec.json 三处污染源（路径 C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\glossary\glossary_ec.json）——不修则 apply_tm / fill_missing 会把错字重新注回：

1. 行 1541  "Evidence": "证据值"            →  "Evidence": "证据"
   （这是本单元 ③ 全部 47 处「证据值」的唯一来源；英文侧从来只是普通名词 evidence，没有任何数值属性）

2. 行 4049  "Signarans": "希格纳兰人 Signarans"  →  "Signarans": "西格纳兰人 Signarans"
   （与同文件行 4048 "Signaran Opal": "西格纳兰蛋白石" 及已定译名「西格纳兰」冲突，是「希格纳兰」混进库里的源头）

3. 行 4571  "The Arcageris": "奥术巨龙 The Arcageris"  →  "The Arcageris": "阿卡杰里斯 The Arcageris"
   （与 A 段已改的 journals.Organizations.pages.The Arcageris.name＝「阿卡杰里斯 The Arcageris」冲突；Arcageris 是神殿/武僧战团，不是龙）

同步 5-其他内容\glossary\glossary_ec.provenance.json 中对应的 base / shipped / cn / base_was 字段（"The Arcageris" 在行 12327-12329 与 36300-36303，"Evidence" 在行 22335 附近）。

建议新增（可选，防后续回退）：
   "Arcing": "引弧"
   "Conundrum": "谜城 Conundrum"
两个键当前都不在 glossary_ec.json 里。

【二】批次键冲突，必须合并而不是二选一：
  4-临时脚本\2026-08-13-round10\batchesB\T3.1.ember.adventure.json 与 T3.1.ember.crucible-adventure.json
  和我的 T4.1.* 在同一个键上重叠一处：
    Ember Early Access.journals.Ancient Paths.pages.Grim Findings.text
  T3 在该叶做的是「突变派」→「突变学派」（2 处），T4 做的是「证据值」→「证据」（9 处），两组编辑互不相交。
  但批次值是整叶新值，后应用者会覆盖前者：先 T3 后 T4 会把「突变学派」退回「突变派」，先 T4 后 T3 会把「证据值」注回。
  合并结果应满足：该叶同时含「突变学派」且不含「证据值」。最省事的做法是先应用 T3，再对该叶做一次纯字符串替换 证据值→证据（9 处）。

【三】lang / .mjs 侧已核，无需改动：
  1-Ember汉化插件\lang\cn.json 与 2-Crucible汉化插件\lang\cn.json 内不含 Arcageris / Signarans / Conundrum / 奥术巨龙 / 希格纳 / 证据值 / 悖论城。
  唯一相关键 "SPELL.INFLECTIONS.SignaraAdj": "西格纳拉的"（对应 lang/en.json 的 "SignaraAdj": "Signaran"）是 Signara 的形容词形，译法正确，未动。

## T2 —— `Sunfire Empire` + `Age of the Tow

以下都在铁律禁写范围内（glossary），只给出具体键与新值，请主控执行：

【1】5-其他内容/glossary/glossary_ec.json —— 改 1 条
  "Sunfire Descendant": "烈阳后裔 Sunfire Descendant"  →  "阳炎后裔 Sunfire Descendant"
  （与我 T2.1.ember.character.json / T2.1.ember.crucible-character.json 里的 name 字段改动配套；不同步会让下一轮 glossary 闸把我的改动判成分叉。）

【2】5-其他内容/glossary/glossary_ec.provenance.json —— 同步上一条的四处旧值
  行 11928  "base": "烈阳后裔"                       → "阳炎后裔"
  行 11929  "shipped": "烈阳后裔 Sunfire Descendant" → "阳炎后裔 Sunfire Descendant"
  行 35581  "cn": "烈阳后裔 Sunfire Descendant"      → "阳炎后裔 Sunfire Descendant"
  行 35583  "base_was": "烈阳后裔"                   → "阳炎后裔"

【3】5-其他内容/glossary/glossary_ec.json —— 新增 4 条（本轮裁定，glossary 目前完全缺这几个词，下一轮没有锚点会再分叉）
  "Sunfire Empire": "阳炎帝国 Sunfire Empire"
  "Sunfire Throne": "阳炎王座 Sunfire Throne"
  "Age of Sunlight": "阳光时代 Age of Sunlight"
  "Wyrm": "古龙"
  （"Wyrms": "古龙 Wyrms" 已存在，保持不动；新增单数条目是为了让 Wyrmspear/Ice Wyrm/Wyrm of Creation 这类复合词有锚点。）

【4】5-其他内容/glossary/glossary_ec.json —— 建议改 1 条，但请交给拥有该词的单元最终裁定（我未在正文动它）
  "Akonites": "乌头属植物 Akonites"  →  建议 "阿肯体 Akonites"
  理由：aconite（乌头）是机翻误认；正文两种在用写法为「阿肯体」（Bestiary 概述）与「阿肯石裔」（Akon GM 块），需二选一后连 name 字段一起统一。

【5】5-其他内容/glossary/glossary_ec.json —— 改 1 条
  "Cruel Dragons": "残酷巨龙 Cruel Dragons"  →  "残酷龙 Cruel Dragons"
  理由：全库 name 字段与正文一律「残酷龙」，与「肉欲龙 Carnal Dragons」「邪龙 Vile Dragons」构词一致，glossary 是错的那一边。

lang/ 与 scripts/、babele-mappings.js 已 grep 过本单元三条线的全部相关词（烈阳/炎阳/阳光年代/日光年代/獠牙飞龙/火焰巨蛇/Sunfire/Age of Sunlight/Wyrm），零命中，无需改动。

## K7 —— Forest of Stone Gazetteer / Bestia

需要主控代改（我不许写 glossary_ec.json）：

1) **glossary_ec.json 键 `"Carrow"`** 当前值 `卡罗ว์ Carrow` —— **中文值里混进了一个泰文字符 `ว์`**（U+0E27 U+0E4C）。pack 里的 name 是干净的「卡罗 Carrow」，所以外来文字闸对 pack 是 0，但 glossary 本身是脏的，任何以它为源的自动回填都会把泰文字符写回包里。
   新值：`卡罗 Carrow`

2) **glossary_ec.json 键 `"Akonites"`** 当前值 `乌头属植物 Akonites` —— Akonites 是阿肯之月上的构装体，不是乌头属植物。本批次已把 pack 的 name 改成「阿肯体 Akonites」，glossary 需同步。
   新值：`阿肯体 Akonites`

3) **glossary_ec.json 键 `"Wyrms"`** 值 `古龙 Wyrms` 是对的，但库里缺少派生形的条目，导致 `Fanged Wyrms` / `Fire Wyrms` 被各自乱译。建议补两条：
   `"Fanged Wyrms": "獠牙古龙 Fanged Wyrms"`、`"Fire Wyrms": "火焰古龙 Fire Wyrms"`

4) **glossary_ec.json 键 `"Hulg'run Lineage"`** 当前值 `Hulg'run血统 Hulg'run Lineage` —— 中文侧留了裸英文。
   新值：`赫尔格伦血统 Hulg'run Lineage`（与 `"Hulg'run" -> 赫尔格伦` 对齐）

5) **两个 name 字段（在 emberHelkasWalkt 包内，超出我的 scope，见 out_of_scope 第 3、4 条）**：
   - `The Armarium`：`军械库 The Armarium` → 建议 `阿玛留姆 The Armarium`（它是杂货店，且与 Ember's Bounty 的 `The Armory` 军械库撞名）
   - `Helkas Green`：`赫尔卡斯·格林 Helkas Green` → 建议 `赫尔卡斯绿地 Helkas Green`（是公共绿地不是人名）
   这两条改完后，Forest of Stone Gazetteer/Helkas.text 里对应的 UUID 标签也要跟着改，那一叶在我的 scope 内、但我没动，等 name 定了再补一次微批。

## T3 —— Warlock/Sorcerer 撞名（阻断）+ Mutagist 

需要主控代改（我按铁律 1 一个字符都没碰）：

【A】1-Ember汉化插件/lang/cn.json —— 2 个键（sorcerers.* 两键保持原值不动，撞名解开后「术士起源」已无歧义）
  "EMBER.DEITY.FIELDS.warlocks.label"
    旧: "术士契约"
    新: "邪术师契约"
  "EMBER.DEITY.FIELDS.warlocks.hint"
    旧: "通常与该神祇相关联的术士契约（如果有的话）。被该神祇授予力量的术士会遵循其中一种契约。"
    新: "通常与该神祇相关联的邪术师契约（如果有的话）。被该神祇授予力量的邪术师会遵循其中一种契约。"

【B】5-其他内容/glossary/glossary_ec.json —— 18 条（左为现值，右为新值）
  "Warlock"                      "术士 Warlock"                    -> "邪术师 Warlock"
  "Sorcerer"                     "术士 Sorcerer"                   -> 不变
  "Mutagist"                     "突变学派"                        -> 不变
  "Mutagists"                    "突变学派 Mutagists"              -> 不变
  "Mutagist Bombardier"          "突变派爆击手 Mutagist Bombardier" -> "突变学派爆击手 Mutagist Bombardier"
  "Mutagist Bombadier"           "突变学派投弹手"                  -> "突变学派爆击手"      （该键对应原文 tokenName 的拼写错误 "Mutagist Bombadier"，保留键名）
  "Mutagist Vivisector"          "突变派活体解剖师 Mutagist Vivisector" -> "突变学派活体解剖师 Mutagist Vivisector"
  "Mutagist Excisor"             "嬗变师切除者 Mutagist Excisor"    -> "突变学派切除者 Mutagist Excisor"
  "Mutagist Grenadier"           "突变投弹手 Mutagist Grenadier"    -> "突变学派投弹手 Mutagist Grenadier"
  "Mutagist Clothing"            "变异师服装 Mutagist Clothing"     -> "突变学派服装 Mutagist Clothing"
  "Mutagist Scout"               "变异学者斥候 Mutagist Scout"      -> "突变学派斥候 Mutagist Scout"
  "Mutagist Scouts"              "突变学派斥候"                    -> 不变
  "Mutagist Contingent"          "突变剂师分队"                    -> "突变学派分队"
  "Mutagist Alchemical Data"     "突变学派炼金数据 Mutagist Alchemical Data" -> 不变
  "Toothbreakers"                "碎牙者 Toothbreakers"            -> "碎牙帮 Toothbreakers"
  "Toothbreaker Hideout"         "碎牙帮藏身处 Toothbreaker Hideout" -> 不变
  "Toothbreaker Rumors"          "碎牙传闻 Toothbreaker Rumors"     -> "碎牙帮传闻 Toothbreaker Rumors"
  "Toothbreaker Thug"            "碎齿暴徒 Toothbreaker Thug"       -> "碎牙帮暴徒 Toothbreaker Thug"
  "Toothbreaker Scaletamer"      "碎齿驯鳞者 Toothbreaker Scaletamer" -> "碎牙帮驯鳞者 Toothbreaker Scaletamer"
  "Toothbreaker Planning Key"    "碎齿者规划室钥匙 …"               -> "碎牙帮规划室钥匙 Toothbreaker Planning Key"
  "Toothbreaker Prison Key"      "碎齿者监牢钥匙 …"                 -> "碎牙帮监牢钥匙 Toothbreaker Prison Key"
  "Toothbreaker Security Key"    "碎齿者安保钥匙 …"                 -> "碎牙帮安保钥匙 Toothbreaker Security Key"
  "Toothbreaker Storage Key"     "碎齿者储藏室钥匙 …"               -> "碎牙帮储藏室钥匙 Toothbreaker Storage Key"
  "Toothbreaker Throne Room Key" "碎齿者王座室钥匙 …"               -> "碎牙帮王座室钥匙 Toothbreaker Throne Room Key"
  建议新增: "Toothbreaker" -> "碎牙帮 Toothbreaker"

【C】三条定译建议进 PROJECT.md / 已定译名表
  Warlock = 邪术师（区别于 Sorcerer = 术士）
  Mutagist(s) = 突变学派
  Toothbreaker(s) = 碎牙帮（指个体用「碎牙帮成员」，指群体用「碎牙帮众」）

## K1

需要主控执行、K1 不能碰的改动（按优先级）：

一、glossary_ec.json（三条键值，直接给出新值）
1. "Luma": 现值「龙语」 → 改为「卢玛语 Luma」。（"Draconic": "龙语" 保持不变。）现在两个键同值，是 A 段那条实证缺陷的源头。
2. "Earth": 现值「大地 Earth」 → 该条只适用于元素义。建议把键改名为 "Earth (element)"，并新增 "Earth (planet)": "地球"；或至少在条目里加限定。本单元 Introduction 的三处误译就是它引起的。
3. "Anachraenum Member": 现值「阿纳克雷努姆成员」 → 改为「阿纳克瑞纽姆成员 Anachraenum Member」（全库 812:30）。

二、name 字段层（compendium，需要主控统一后再回刷正文）
4. Mutagist 系列 actor/item 名五种写法并存，建议全部收敛到「突变学派」或「突变派」其一：
   Mutagists=突变学派 / Mutagist Scout=突变学派斥候 / Mutagist Alchemical Data=突变学派炼金数据 /
   Mutagist Bombardier=突变派爆击手 / Mutagist Vivisector=突变派活体解剖师 /
   Mutagist Excisor=嬗变师切除者 / Mutagist Grenadier=突变投弹手 / Mutagist Clothing=变异师服装。
5. scenes."The World of Ember".notes 里的四个 Edge 建议按 Geography 侧改（语义正确的一侧）：
   The Swirling Edge 旋涡之刃→旋涡之缘；The Roaring Edge 咆哮之锋→咆哮之缘；
   The Untamed Edge 狂野锋刃→蛮荒边缘；The Hoarfrost Edge 霜锋之刃→霜冻边缘。
   同一批注记里 The Spiritlands「灵魂之地」→「灵界」（全库 46:6）。
6. items."Sajor's Journal" 萨约尔的日志 → 萨乔尔的日志（对齐 actor「萨乔尔·维莱克斯」）。
   items."Clipper's Endless Scroll" 快剪手的无尽卷轴 → 克利珀的无尽卷轴（对齐 actor「克利珀」）。

三、需要主控先裁、再全库批量刷的口径问题（K1 内已尽量不制造新分叉）
7. 长度单位：英尺 2074 : 尺 170，请定一个再全库刷。
8. Great City：大城市 42 : 伟大城市 20 : 伟大的城市 2（K1 三处已统一为「大城市」）。
9. Shardstone：碎片石 2 : 裂片石 2，无多数派，请裁。
10. outsider / otherworldly 是否可用「异界」——Introduction 明写余烬没有其他位面、平行宇宙或维度，全库 247 处「异界」与这条设定冲突。K1 只删了英文侧根本没有对应词的那几处。
11. 神名 Spectra = 「光谱」（name 字段，全库 307 处）是否重裁。物理学名词做上古女神名，在 K1 的对话里出现「我是光谱的女祭司」。同一神的 item「Spectra's Blessing」用的是「斯佩克特拉」。
12. Repurposed Quarry / Compound Lab 里 "Apply Toxin" 与 "Apply Poison" 这类 Foundry 行动名，K1 按「已有 CN 定名的一侧」补译成「施加毒素/施加中毒」；若项目口径是这类 UI 标签一律留英文，请回退这一处。

四、不涉及 lang/.mjs 的改动需求；本单元未发现 lang 侧问题。

## K5 —— Steed's Point / Yakoshta Mine / Ka

需要主控代改 glossary_ec.json（我不写该文件）：
1) 键 "Helkas Green"：现值 "赫尔卡斯·格林 Helkas Green" → 新值 "赫尔卡斯绿地 Helkas Green"。理由：the Green 指村镇中央公共绿地，同本 journal 的 Manfryd's Anvil 页已把 "the Green" 译作「绿地」，Helkas Green.text 自身也说这是 "the settlement's central courtyard area"。
2) 键 "Kadhana Lizard"：现值 "卡达娜蜥蜴 Kadhana Lizard" → 新值 "卡达纳蜥蜴 Kadhana Lizard"。理由：actors.Kadhana Lizard 的 name/tokenName 都是「卡达纳蜥蜴」，包内「卡达娜」0 次、「卡达纳」14 次。
3) 键 "Elder Goddess Spectra"：现值 "上古女神丝珀特拉" → 新值 "上古女神光谱"；键 "Spectra's Blessing"：现值 "斯佩克特拉的祝福" → 新值 "光谱的祝福"。理由：包内 @UUID{Spectra} 的标签、FoS Gazetteer、Helkas.Glinthome 一律作「光谱」，glossary 自身的 "Spectra → 光谱" 与 "Shrine to Spectra → 光谱圣祠" 也是「光谱」，只有这两条走样。（若主控倾向反向统一为音译，则需改的是包内 30+ 处，改动面大得多。）

lang/ 与 .mjs 本轮无需改动。

另请主控决定并派发：越界清单第 1、2 条（「阿克图里安人人类」6 处、「赫尔卡斯·格林」3 处）属于必须与本批同时落地的连带修改，否则会在库内造成新的分叉。

## K6 —— Lightless Halls / Marlstone Manor 

1) glossary_ec.json 建议补/订正三条（我不写 glossary，只报键值）：`Indigo Ray` → `因迪戈·雷`（并注明口令词单用作「因迪戈」，禁用「靛蓝·射线」）；`Lanterelle` → `兰特蕾尔花`（禁用「灯菇花」）；`First Giant` → `最初的巨人`（禁用「第一巨人／第一位巨人」）。2) 建议加一条全库级裁决：`warlock` 现全库 74 条均译「术士」，与 `sorcerer` 的「术士」撞名（另有 178 条 cn_only），需主控定一个区分译名（如 warlock＝邪术师）后统一批改，本单元未动。3) 建议把「房间通行权限」的三个定义词（开放／受限／禁止）与身份名（园丁／装饰工／厨房员工）列入术语表硬约束——Marlstone Manor 一本里就出现了「限制进入／限制通行」「装饰人员／装饰师／装饰工人」等 5 种偏离。4) 本单元发现的一类机械闸盲区，建议加扫描器：中文侧给英文原本没有 `{标签}` 的 @UUID 补写标签（本卷 20+ 处），其中 2 处标签内容与目标文档不符（已修）——建议加一条「CN 有标签而 EN 无标签」的扫描，并对 EN 有标签的做标签一致性比对。

## T5

【必须由主控改 glossary_ec.json：C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\glossary\glossary_ec.json】

A. 订正两条错值（这两条现在会把错误再传播回正文）
  "Luma": "龙语"  ->  "卢玛语 Luma"
      理由：A 段已裁 Luma＝卢玛语、Draconic＝龙语，compendium 侧也已改对（Cultures/Lumek「卢梅克人的语言名为卢玛语 Luma」、Languages 表「卢玛语 Luma」），但 glossary 仍停在错的那一边，且与同文件里已存在的 "Draconic": "龙语" 撞成同义词。
  "Caste": "种姓"  ->  "卡斯特 Caste"
      理由：Patch 0.3.1 原文 "a remote settlement in the northeast of the Arctus Plateau"；Main Quest Overview 已译「海滨小镇卡斯特」；scenes.Vista: Caste.levels.Caste＝「卡斯特」。本轮把 scenes.Vista: Caste.name 从「远景：种姓」改成「远景：卡斯特」，glossary 不同步会被下一轮回滚。

B. 新增本轮定名的语言条目（前 6 条有正文既有译名坐实，其余为本轮新拟）
  "Mithia":   "米西亚语 Mithia"          （Cultures/Kithil 正文）
  "Cascal":   "卡斯卡尔语 Cascal"        （Cultures/Cascilian 正文）
  "Harmos":   "哈莫斯语 Harmos"          （Lantern Roads 三页正文）
  "Eonic":    "永世语 Eonic"             （Arcturian Trinkets 正文）
  "Kaziric":  "卡兹里克语 Kaziric"       （Ember's Bounty / The Smoke Hut 正文）
  "Luxaran":  "卢克萨兰"                 （多处正文；勿与内界 Luxarum 卢克萨鲁姆混淆）
  "Scripta":  "斯克里普塔语 Scripta"
  "Solical":  "索利卡尔语 Solical"
  "Moiré":    "莫伊雷语 Moiré"
  "Windclaw": "风爪语 Windclaw"
  "Scor":     "斯科语 Scor"              （依 Deities/Scoris pronunciation「斯科-里斯」定音）
  "Veax":     "维亚克斯语 Veax"
  "Judega":   "犹德加语 Judega"
  "Caligon":  "卡利贡语 Caligon"
  "Lunix":    "卢尼克斯语 Lunix"
  "Ocana":    "奥卡纳语 Ocana"
  "Asc-seia": "阿斯克-赛亚语 Asc-seia"    （连字符写法沿用 Casla-Brava 卡斯拉-布拉瓦）

C. 一条需要主控决定表示法的多义词
  "Abyssal" 目前不在表里。它有两义且都在用：语言＝深渊语（Languages 表）、生物分类＝深渊裔（adversary folders / Bestiary Abyssals）。若 glossary 是单值结构，建议只收 "Abyssal": "深渊语"，并在 folder/分类语境靠 "Abyssals": "深渊裔"（已存在）兜底；否则本轮刚修好的 folders.Abyssal＝「深渊裔 Abyssal」会被单值 glossary 反向拉回「深渊语」。

【lang/*.json 与 .mjs：本轮无需改动，已核实】
1-Ember汉化插件/lang/en.json（486 键）与 2-Crucible汉化插件/lang/en.json（1842 键）里都没有任何 Ember 语言名键；Crucible 仅有 LANGUAGES.Common＝通用语、LANGUAGES.Sign＝手语、LANGUAGE_CATEGORIES.*，均已正确。正文里的 [[/language xxx]] enricher 由 Ember 模块自身数据渲染，不落在这两个汉化仓库里，故语言表的中文名不会与 enricher 冲突，enricher 参数一律原样照抄未动。

【④ Strayhearth：复核结论为已修好，本轮零改动】
term_gate --en "Strayhearth" 全库 104 叶：gated_hit 迷炉＝104、离火之家＝0，无 en_only、无 cn_only。Players' Guide 两处（Creation Overview / Welcome to Ember，各两包一份）现在是「决定自己的角色为何会与迷炉商队同行」「你的故事开始于跟随迷炉商队一同旅行」，语义与英文 "reason for traveling with the Strayhearth Caravan" 一致，简报提到的「在某地结识的同伴」式改写已不存在。残留的「离火」全部是无关词（远离火源／逃离火灾／距离火焰）。

## K2 — The Bleak Archive / Arcturel Dives 

【需要主控做一次全局裁决，我没有单方面动手】

1) Warlock / Sorcerer 撞名（严重，本单元 Corpin Sanctuary/Profane Altar 已出现「牧师、术士、术士和巫师」）
   现状：
     entries.Ember Early Access.journals.Character Classes.pages.Sorcerer.name = "术士 Sorcerer"
     entries.Ember Early Access.journals.Character Classes.pages.Warlock.name  = "术士 Warlock"
   全库无任何替代词（邪术师 / 契术士 / 魔契师 命中数均为 0）。英文侧 warlock 168 次、sorcerer 52 次；
   本单元 Bleak Archive 里 Tethra Shùl 的 warlock 一律译「术士」。
   建议（二选一，选定后需全库批改）：
     A. Warlock → 邪术师（沿用 5e 官方中文），Sorcerer 保留 术士。改动面：Warlock.name 1 处 + 全库 warlock 出现处。
     B. Sorcerer → 根源术士 / 血脉术士，Warlock 保留 术士。改动面小（52 处），但偏离 5e 官方译名。
   裁决后请一并回填 Corpin Sanctuary/Profane Altar 那句：
     键 `Ember Early Access.journals.Corpin Sanctuary.pages.Profane Altar.text`
     现值片段：「牧师、术士、术士和巫师可以与亵渎祭坛同调」（同调二字我已在本批次改好）
     待改为：「牧师、术士、<Warlock 定名>和巫师可以与亵渎祭坛同调」

2) 中文标题锚点 id 的全库补齐（严重）
   凡被 @UUID[...#slug] 引用的英文标题，中文侧若不显式写 id，锚点必然失效。
   本项目已有成例（the-palimpsest-doors / fiery-doorway-trap / corrupted-guardians / the-device / vorg-infestation）。
   建议做一个批量工具：扫全库所有 @UUID[...#slug]，回溯目标页英文标题，给对应中文标题补 id="<英文 slug>"。
   我单元受影响但改不到的目标页：
     `Ember Early Access.journals.A Brush With Death.pages.Where Evil Lurks.text`
        <h3 class="divider">探索黯淡秘库</h3>          → 需 id="exploring-the-bleak-archive"
        <h3 class="divider">夺取奈瑟赫普提卡斯之指</h3> → 需 id="seizing-the-finger-of-nethehepticas"
        <h3 class="divider">离别赠言</h3>              → 需 id="a-departing-word"
        （另有 Cleansing the Wraith's Remains 一节同样需要 id="cleansing-the-wraiths-remains"）
     `Ember Early Access.journals.Arctus Plateau Gazetteer.pages.The Bleak Archive.text`
     `Ember Early Access.journals.Arctus Plateau Gazetteer.pages.Corpin Sanctuary.text`
        <h3 class="divider">背景</h3> → 需 id="lore"（全库约 140 处同类标题）

3) 「证据值」词表污染的全库回改（严重）
   evidence 被无差别替换成机制词「证据值」，约 20 个叶子。仅 Ancient Paths/Grim Findings 的 18 点计分机制该保留。
   本单元内的 2 处我已改回「证据」。建议按 EN 正则 `\bevidence\b` 且中文含「证据值」逐条复核。

4) glossary_ec.json 建议补录（避免下一轮再分叉）
   Ebbok Zhùr = 艾博克·朱尔（勿作 埃博克 / 祖尔）
   Bleak Archive Relic = 黯淡秘库遗物（勿作 黯晦档案）
   Emelyn Arvoda = 埃梅琳·阿沃达
   Tethra Shùl = 泰斯拉·舒尔（勿作 泰丝拉）
   Age of the Tower = 高塔时代（阳光时代仅作别名）
   retractable wall = 可升降墙壁
   blush blossom = 绯花树
   wardrobe（家具）= 衣柜；wardrobe（剧场更衣室）= 服装间
   Sanctuary = 庇护所 / Sanctorum = 圣所（勿混）
   Yarinu = 雅里努 / Yarino = 亚里诺（两个不同地名）

## K3 — Aedir Signalpost / Arcturel Tradewa

【批次冲突，必须处理】T1 单元的批次与我的 K3 批次在同一个键上都有值：
  键：Ember Early Access.journals.Lake Jinro Lunar Shrine.pages.Garganthus Tunnel.text
  文件：batchesB/T1.1.ember.adventure.json、batchesB/T1.1.ember.crucible-adventure.json 与 batchesB/K3.1.*
  T1 只改了「现身于烬地表」→「现身于余烬地表」；K3 的值是 T1 的超集（同样改了这一处，另外去掉了朗读段里凭空添加的「栖息于湖中的」）。
  处置：apply 时让 K3 覆盖 T1（K3 排在 T1 之后），或直接从 T1 两个批次里删掉这个键。若 T1 排在后面，我这条「凭空增删」修复会被静默回退。

【需主控执行的库外改动】以下叶子不在我单元路径下，但与我批次里的改名强耦合，我已一并写进 K3 批次（两包各 5 条，共 10 条），请确认是否接受；若不接受，请把我批次里对应的页名改动也一并撤掉，否则会出现「地图图钉名 ≠ 页名」「物品名 ≠ 正文引用」：
  1. Ember Early Access.scenes.Lake Jinro.notes.Garganthus Tunnel = "加甘萨斯隧道 Garganthus Tunnel"
  2. Ember Early Access.scenes.Lake Jinro.notes.Grave Blade Shore   = "墓刃湖岸 Grave Blade Shore"
  3. Ember Early Access.scenes.Lake Jinro.notes.Loot-Lined Hollow   = "藏宝树洞 Loot-Lined Hollow"
  4. Ember Early Access.items.Garganthus Hide.name                  = "加甘萨斯兽皮 Garganthus Hide"
  5. Ember Early Access.actors.Juvenile Garganthus.items.Garganthus Hide.name = "加甘萨斯兽皮 Garganthus Hide"
  第 4、5 条落地后，务必同时处理 out_of_scope 第 1 条（The Expedition Challenge / Amazing Brambles 的两处「加甘图斯兽皮」），否则分叉只是换了个地方。

【glossary_ec.json 建议补录】（我未写入，请主控决定）
  - Underforge = 底部熔炉（全库 25:1，唯一反例已在本批修掉）
  - stealth field = 隐形力场
  - Assembly Area / assembly line = 装配区 / 装配线（禁用「集结区域」）
  - Garganthus = 加甘萨斯（禁用 加冈瑟斯 / 加甘图斯）
  - Boon（游戏机制）= 恩惠骰；boon / blessing / benefit（普通名词）= 恩赐 / 祝福 / 裨益，不得套用「恩惠骰」
  - Towyr 词根表列名固定为「基本含义 / 完整翻译」
  - Aedir 信号哨站四拉杆谜题关键词 power = 动力（形容词位「动力十足的」）

## K4 — Vortest Tower / Mythspire Observato

无需改 lang/.mjs/glossary_ec.json 的项。

但有两条建议主控在本轮统一处置（均在我的四本之外，故未写入批次）：
1. `ember.crucible-adventure.json` + `ember.adventure.json` 键 `Ember Early Access.actors.Mythspire Guardian.items.Cosmic Gems.actions.cosmicGemCharge.name`：现值「冲锋宝石 Charge Gem」→ 建议「充能宝石 Charge Gem」。我已在 K4 批次里把 Mythspire Central Room 页引用它的 UUID 标签改成 {充能宝石}，若角色卡不同步改，标签与动作名会对不上。
2. `glossary_ec.json` 可考虑补录本轮定下的三条同本裁决（仅供后续会话参照，不改也不影响本批次）：`suspended animation`＝生命悬滞；`Eonic`＝伊欧尼克语；`Lunarium`＝月辉宫。

## T1 —— 三条最大的跨书术语分叉：`Otherhood of Fortune`

## 一、glossary_ec.json 待改（我不能写，给出确切键与新值）

文件：`C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\glossary\glossary_ec.json`

| 键 | 现值 | 新值 |
|---|---|---|
| `Otherhood` | `异缘会` | `异姊会` |
| `Otherhood Brigand` | `异域母性强盗 Otherhood Brigand` | `异姊会强盗 Otherhood Brigand` |
| `Otherhood Brigands` | `他者会匪徒` | `异姊会强盗 Otherhood Brigands` |
| `Otherhood Raider` | `同袍会劫掠者 Otherhood Raider` | `异姊会劫掠者 Otherhood Raider` |
| `Otherhood of Fortune` | `幸运异姊会 Otherhood of Fortune` | 不变（这条是对的，是本轮的裁决依据） |
| `Pathways Gazetteer` | `《通途公报》 Pathways Gazetteer` | `通路地名志 Pathways Gazetteer` |
| `Pathways Scout Map` | `通道区侦察地图 Pathways Scout Map` | `通路侦察地图 Pathways Scout Map`（包里 name 字段已经是这个，glossary 落后了） |

`Otherhood` 一族现在 glossary 里有五种互不相干的写法（异缘会／异域母性／他者会／同袍会／幸运异姊会），「异域母性」和「同袍会」明显是机翻残留，不清掉的话下一轮 apply_tm 会把它们再灌回包里。

顺带（越界，见 out_of_scope）：`Ordain Gazetteer` = `奥丹地志 Ordain Gazetteer` 两处都错，包里 name 是「奥尔丹地志」，建议 glossary 与包一起改成 `奥尔丹地名志 Ordain Gazetteer`。

lang/cn.json、lang/en.json、scripts/ember-hardcoded-cn.mjs 我全量扫过，三个词都没有需要改的键（`EMBER.SHEETS.GazetteerSheet`＝「余烬地名志」、.mjs 的 `"Transition to Pathways?"`＝「转入通路？」都已经和我的结论一致）。

## 二、⚠ 批次冲突：有 4 条叶子会被别的单元的整叶新值把我的修正冲掉

批次是整叶覆盖，后写的赢。我在交付时（batchesB 目录当时有 K1–K7 / T1–T5 共 27 个文件）比对了全部同名叶子，下面这些**必须按「先应用对方、再应用 T1」或手工合并**，否则术语分叉会重新出现：

| 叶子 | 冲突单元 | 会被冲掉的修正 |
|---|---|---|
| `journals.Ancient Paths.pages.Grim Findings.text`（两包） | **T3、T4** | 「异缘会」→「异姊会」（各 2 处） |
| `journals.Helkas.pages.Overview.text`（两包） | **K5** | 「异缘会」→「异姊会」 |
| `journals.Cosmos.pages.Akon.text`（两包） | **T2** | 「烬地」→「余烬」（3 处） |

另有 20 对叶子与 K1/K3/K4/K7/T2/T5 同时命中但**互相兼容**（对方改的是别的词，我改的术语在对方的值里已经是正确写法，或对方独立做出了同样的修正）——这些只要合并两边的编辑即可，例如 `Bestiary.pages.Wyrms.contentOverview`（K7/T2 在改「巨龙→古龙」，我在改「烬界→余烬」）、`Geography.pages.The Sphere.contentOverview`（K1 修了「比烬的寰宇」，我修了「烬地表」，合并后两边收敛到同一个值）。

我这份统计是我落盘那一刻的快照，其他单元此后若再改批次需要重跑一遍比对。合并脚本建议：以叶子为键收集所有单元的新值，对每个叶子拿原始 CN 做三方 diff，只有编辑区间重叠时才需要人工介入——上表三条就是重叠的那几条。

## 三、一条给下一轮的提醒

`Ember` 这个词在本项目里同时是**世界名**（应为「余烬」）和**产品名**（Ember 抢先体验 / Ember Alpha 补丁 / Ember 设定，惯例保留拉丁）。我这轮只动了确指世界的那些，产品名一处未动。如果以后要做「产品名要不要也中文化」的裁决，范围是全库 626 处裸拉丁 Ember 中的约 600 处，集中在 Gamemaster's Guide 的 Patch 页、Players' Guide 的 License 页和 Introduction 三本，建议整体一次定，不要零散改。

## K0

glossary_ec.json 建议补录（我没动任何 glossary/lang/scripts 文件，具体键值如下，请主控执行）：
1. "Pathways" → "通路"（禁用：通道区）；"Gazetteer" → "地名志"（禁用：公报）
2. "Ember"（世界名） → "余烬"（禁用：烬界 / 烬火 / 安珀 / 裸 Ember）
3. "Chessman/Chessmen" → "棋士"（依 actors.Chessman.name「棋士 Chessman」；禁用：棋子人）
4. "Garganthus/Garganthi" → "加甘萨斯"（依 actors.Juvenile Garganthus.name；禁用：加甘提 / 加甘图斯）
5. "Excavation Pit" → "挖掘坑"；"Glowing Ore Pit" → "发光矿坑"（两个不同房间，务必分开）
6. "Mycelian Expanse" → "菌丝旷野"；"Mycelian Forest" → "菌丝林"（禁用：菌原辽域 / 辽域 / 菌丝荒原 / 菌辉林地）
7. "Stone Life"（Ooze Control 页名） → "石之生命"（原「石料生命」；配套需改 journals.Yakoshta Mine.pages.Ooze Pool.text 里的链接标签）
8. "Ossuary" → 建议全库统一为 "藏骨堂"（现状：藏骨堂 1 / 骨库 5，其中 The Bleak Archive.pages.Ossuary.name 也是骨库，需 T1 拍板后一次性改）
9. "evidence" → "证据"（明确禁用「证据值」；仅 Ancient Paths 的 Evidence Points 计分机制可作「证据点数」）
10. "Long Rest"（system-swap dnd5e 侧） → "长休"

另请主控决定标题锚点方案（见 out_of_scope 第 1 条）：是否统一给中文标题补 id="<英文 slug>"。这会改动 800+ 个叶的 HTML 属性，需要先定规矩再动；本单元内 33 处我按铁律 2 未擅自改。
