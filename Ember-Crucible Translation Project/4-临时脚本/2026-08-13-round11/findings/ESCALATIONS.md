# 第十一轮 escalate

## F5

【A. glossary_ec.json —— 必改，否则「证据值」会被下一次自动套词重新灌回全库】
文件：5-其他内容/glossary/glossary_ec.json
  键 "Evidence"：现值 "证据值" → 新值 "证据"
  理由：这是 29 叶坏替换的根因，词条本身没有任何机制依据 —— 2-Crucible 仓 EN 侧 evidence 命中 0 行，Ember 侧唯一带点数的用法（Ancient Paths/Grim Findings 的 18 点）现译「18 点证据」也不需要「值」。同族键 'Poor/Good/Excellent Evidence Collection' = '低劣/良好/优秀证据收集' 保持不变，与之自洽。
（可选，用于把本轮定下的语言表锁死，防止下一轮再散：新增 "Scripta"="斯克里普塔语 Scripta"、"Solical"="索利卡尔语 Solical"、"Moiré"="莫伊雷语 Moiré"、"Kaziric"="卡兹里克语 Kaziric"、"Windclaw"="风爪语 Windclaw"、"Scor"="斯科语 Scor"、"Veax"="维亚克斯语 Veax"、"Judega"="犹德加语 Judega"、"Eonic"="永世语 Eonic"、"Harmos"="哈莫斯语 Harmos"、"Caligon"="卡利贡语 Caligon"、"Lunix"="卢尼克斯语 Lunix"、"Ocana"="奥卡纳语 Ocana"、"Asc-seia"="阿斯克-赛亚语 Asc-seia"；现 glossary_ec 里这 14 个键一个都没有。）

【B. 1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs 的 const LANGUAGES —— 5 处与刚定案的权威表打架，且全都会真的渲染出来】
这张表同时驱动两条通道：`Language: X` 前缀标签，以及 patchCrucibleConfig() 改写的 crucible.CONFIG.languages[*].label —— 后者就是 [[/language x]] 增强器渲染出的文字。全库 [[/language …]] 调用计数见括号。
  "Solical"：索利卡语 → 索利卡尔语   （4 次调用；compendium EN 闸 4 行全作索利卡尔语，索利卡语 0）
  "Mithia" ：密西亚语 → 米西亚语     （2 次；EN 闸 6 行全作米西亚语，密西亚语 0）
  "Scripta"：书文语   → 斯克里普塔语 （2 次；EN 闸 3 行全作斯克里普塔语，书文语 0）
  "Scor"   ：斯科尔语 → 斯科语       （4 次；EN 闸 2 行作斯科语，斯科尔语 0）
  "Lunix"  ：月语     → 卢尼克斯语   （2 次；EN 闸 2 行作卢尼克斯语，月语 0）
另需新增 3 个缺键，否则这些 [[/language …]] 会在中文正文里渲染出英文：
  "Moiré": "莫伊雷语"   （4 次调用）
  "Borel": "博雷尔语"   （4 次调用）
  "Kost" : "科斯特语"   （2 次调用）
  （这 3 个不在 5-其他内容/reference/ember-fr-recon/ember-hardcoded-strings-en.json 的 25 条 "Language: X" 清单里，说明那份参考清单不全，别照它裁。）
表里其余 17 键与权威表一致，尤其 "Eonic": "永世语" 已经对 —— 这也是我把 Vortest Tower 的「伊欧尼克语」判为错侧的第二条独立证据。

【C. lang/cn.json】
两仓 lang/cn.json 里 `证据` 出现 0 次，语言名相关键也只有 crucible 本体的 LANGUAGES.Common/Sign（通用语/手语，正确）。本族无需改动 lang。

## F4

【glossary_ec.json —— 6 条改值 + 5 条新增，全部与本批已落盘的包内改动配套；不改会在下一次批量套词时把包又污染回去】

改值（键 -> 新值）：
1. "The Armarium": "军械库 The Armarium"                  -> "阿玛留姆 The Armarium"
2. "Helkas Green": "赫尔卡斯·格林 Helkas Green"            -> "赫尔卡斯绿地 Helkas Green"
3. "Pathways Gazetteer": "《通途公报》 Pathways Gazetteer"  -> "通路地名志 Pathways Gazetteer"
4. "Pathways Scout Map": "通道区侦察地图 Pathways Scout Map" -> "通路侦察地图 Pathways Scout Map"
5. "Garganthus Hide": "加甘图斯兽皮 Garganthus Hide"       -> "加甘萨斯兽皮 Garganthus Hide"
6. "Garganthus Tunnel": "加冈瑟斯隧道 Garganthus Tunnel"   -> "加甘萨斯隧道 Garganthus Tunnel"
   （第 6 条是本轮新发现：Garganthus 在词表里竟有第三种音译「加冈瑟斯」，包内 0 命中，属纯词表污染；'Garganthus' -> 加甘萨斯、'Juvenile Garganthus' -> 幼年加甘萨斯 两条本来就是对的。）

新增（键 -> 值）：
7.  "Charge Gem": "充能宝石 Charge Gem"   ← 词表现无此键，而 'Charge' -> 冲锋 会把它再次套错
8.  "The Swirling Edge": "旋涡之缘 The Swirling Edge"
9.  "The Roaring Edge": "咆哮之缘 The Roaring Edge"
10. "The Untamed Edge": "蛮荒边缘 The Untamed Edge"
11. "The Hoarfrost Edge": "霜冻边缘 The Hoarfrost Edge"
    （8-11 词表现在完全没有这四条，注记侧「刃/锋」正是无人把关的产物；'The Edge of the World' -> 世界之缘 指的是烟坊那家锻炉，与这四条无关，不要合并。）

不需要改的（防后来者误合并）：'Armory' -> 军械库 Armory、'The Armory' -> 军械库 The Armory 保持原样——那是 Ember's Bounty / Spellbreaker Tower / Kalion Stadium Underworks 的真军械库，本次拆分正是要让 Armarium 让出这个词。

【lang / .mjs】无需改动：已 grep 1-Ember汉化插件/lang/*.json、1-Ember汉化插件/scripts、2-Crucible汉化插件/lang、3-常用脚本/extract，Armarium / Helkas Green / Garganthus / Charge Gem / *Edge / Pathways Gazetteer 全部 0 命中。

【⚠ 争叶提示】ember.{adventure,crucible-adventure}.json :: entries.Ember Early Access.journals.Forest of Stone Gazetteer.pages.Helkas.text 这一叶我必须整叶提交（Armarium ×3）。人名族若也提交同一叶（ALL_OOS #144 洛蕾提娅），两个批次会互相覆盖。我已在自己的值里把「洛蕾提娅」→「洛雷蒂娅」4 处一并改掉，所以以我的版本为准不会回退任何一侧；若最终以对方版本为准，请务必把 3 处「军械库」→「阿玛留姆」补进去。

## F6

【必须改，我不能写】

1) 1-Ember汉化插件/lang/cn.json —— 只需改 warlocks 两条；sorcerers 两条【不要动】。
   "EMBER.DEITY.FIELDS.warlocks.label"（第 253 行）
     现值: "术士契约"
     新值: "邪术师契约"
     (EN: "Warlock Pacts")
   "EMBER.DEITY.FIELDS.warlocks.hint"（第 254 行）
     现值: "通常与该神祇相关联的术士契约（如果有的话）。被该神祇授予力量的术士会遵循其中一种契约。"
     新值: "通常与该神祇相关联的邪术师契约（如果有的话）。被该神祇授予力量的邪术师会遵循其中一种契约。"
     (EN: "Warlock Pacts which are typically associated with this deity, if any. Warlocks who are granted power by this deity adhere to one of these pacts.")
   "EMBER.DEITY.FIELDS.sorcerers.label"（第 255 行，"术士起源"）与 ".hint"（第 256 行）——【维持原样】。EN 是 Sorcerous Origins / Sorcerers，Sorcerer 定译就是术士；Warlock 改名后这两条已无歧义，且与 compendium 内 Jurtak Geomancer.biography.private 的「术士起源」一致。原 OOS #43 说「三条必须同步」，实测只需同步两条。
   2-Crucible汉化插件/lang/cn.json 无需改（唯一命中 "SPELL.WARNINGS.SorcererNoIconic" 是 Crucible 自己的 Sorcerer 天赋，仍作术士，正确）。

2) 5-其他内容/glossary/glossary_ec.json（第 5138 行）
     "Warlock": "术士 Warlock"   →   "Warlock": "邪术师 Warlock"
   "Sorcerer": "术士 Sorcerer"（第 4178 行）与 "Sorcery Points": "术法点数 Sorcery Points"（第 4179 行）——【维持原样】。
   不修这条的后果与 Signarans 那条同性质：apply_tm / fill_missing 以后会把「术士 Warlock」重新注回去，把刚统一好的 144 叶再撞散。
   建议顺手新增一条（本轮裁定的 Patron 基准需要落点，否则同样会被 TM 反复冲掉）：
     "Warlock Patron": "邪术师宗主 Warlock Patron"
   不建议加裸 "Patron" 条目——全库 191 个 patron 里大多数是酒馆主顾/赞助人义（Ordain Gazetteer 各店铺页、Unfinished Business 等），加了会误伤。

3) .mjs / scripts / babele-mappings.js —— 无需改动。全项目 *.mjs / *.js 递归 grep "warlock|sorcer|施恩者|宗主"，零命中。

## F3

【glossary_ec.json】C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\glossary\glossary_ec.json —— 5 个键必须改，否则 glossary 会把包里刚统一好的译名再拉回去：
1. "Spectra"：            "光谱 Spectra"                            → "斯佩克特拉 Spectra"
2. "Elder Goddess Spectra"："上古女神丝珀特拉"                        → "上古女神斯佩克特拉"
3. "Shrine to Spectra"：   "光谱圣祠 Shrine to Spectra"              → "斯佩克特拉圣祠 Shrine to Spectra"
4. "Tomb of Spectra's Chosen"："光谱之选民的坟墓 Tomb of Spectra's Chosen" → "斯佩克特拉之选民的坟墓 Tomb of Spectra's Chosen"
5. "Akonites"：           "乌头属植物 Akonites"                      → "阿肯体 Akonites"
（"Spectra's Blessing" 已是 "斯佩克特拉的祝福 Spectra's Blessing"，无需改。）

【glossary_ec.json · 低优先，顺手项】
6. "Ember Standalone Event"："Ember独立事件" → "余烬独立事件 Ember Standalone Event"（唯一一条把世界名留成裸英文、且缺双语并列的 Ember 条目）
7. "Ember Type"："余烬类型" → "余烬类型 Ember Type"（补双语并列）
8. "Umber's Pass"："安珀之径" 建议保留原样：Ember 侧的「安珀」已被清空，「安珀」现在全库唯一指向 Umber，ALL_OOS #199 担心的撞车已实际消解；改成「翁伯」反而要动包内 2 叶。

【lang/cn.json 与 .mjs】无需改动：1-Ember汉化插件/lang/cn.json、2-Crucible汉化插件/lang/cn.json 对 "Spectra"/"光谱"/"Akonite"/"烬界" 均 0 命中；3-常用脚本 下的 .mjs 亦 0 命中。

【交付顺序，请主控注意】F3 与并行单元有 3 个叶子撞车，我已把我的替换**叠加在对方批次的新值之上**重建（F1: journals.Helkas.pages.Glinthome.text 双包；F4: journals.Forest of Stone Gazetteer.pages.Helkas.text 双包；F5: actors.Eveis Brightstone.biography.private 仅 crucible 包）。因此 **F3 必须在 F1/F4/F5 之后应用**，否则这 3 叶会丢掉我的 光谱→斯佩克特拉。若 F1/F4/F5 在我落盘后又重写过自己的批次，请对这 3 个键重新叠一次（规则：先替 丝珀特拉→斯佩克特拉，再替 光谱→斯佩克特拉，但保护「完整光谱」「可见光谱」两个串）。anchor.1.* 批次经核对已 100% 落盘（432/432 与当前 CN 一致），不构成冲突。

## F2

【全部落在 5-其他内容/glossary/glossary_ec.json；lang/cn.json、scripts/、*.mjs 经 grep 确认无本族污染，无需改动】

A. Cevher 族（OOS #105(3)(4) / #121）—— 4 个键，改成与 compendium name 字段一致：
  "Funar Cevher"              : "富纳尔·杰夫赫" → "富纳尔·杰夫赫尔 Funar Cevher"
  "House Cevher Mausoleum"    : "杰夫赫家族陵墓 House Cevher Mausoleum" → "杰夫赫尔家族陵墓 House Cevher Mausoleum"
  "House Cevher Signet Ring"  : "切夫赫尔家族印戒 House Cevher Signet Ring" → "杰夫赫尔家族印戒 House Cevher Signet Ring"
  "Lyla Cevher"               : "莱拉 Lyla Cevher" → "莱拉·杰夫赫尔 Lyla Cevher"
  （"Lyla" → "莱拉" 保持不变：裸名确实译莱拉）

B. Mutagist 族（OOS #65 / #127 / #130 / #134）—— 7 个键，统一前缀「突变学派」：
  "Mutagist Bombadier"  : "突变学派投弹手" → "突变学派爆击手"        （英文错拼键，对应 tokenName）
  "Mutagist Bombardier" : "突变派爆击手 Mutagist Bombardier" → "突变学派爆击手 Mutagist Bombardier"
  "Mutagist Vivisector" : "突变派活体解剖师 Mutagist Vivisector" → "突变学派活体解剖师 Mutagist Vivisector"
  "Mutagist Excisor"    : "嬗变师切除者 Mutagist Excisor" → "突变学派切除者 Mutagist Excisor"
  "Mutagist Grenadier"  : "突变投弹手 Mutagist Grenadier" → "突变学派投弹手 Mutagist Grenadier"
  "Mutagist Clothing"   : "变异师服装 Mutagist Clothing" → "突变学派服装 Mutagist Clothing"
  "Mutagist Scout"      : "变异学者斥候 Mutagist Scout" → "突变学派斥候 Mutagist Scout"
  "Mutagist Contingent" : "突变剂师分队" → "突变学派分队"

C. Toothbreaker 族（OOS #67 / #130）—— 9 个键，统一前缀「碎牙帮」：
  "Toothbreakers"                : "碎牙者 Toothbreakers" → "碎牙帮 Toothbreakers"
  "Toothbreaker Rumors"          : "碎牙传闻 …" → "碎牙帮传闻 Toothbreaker Rumors"
  "Toothbreaker Thug"            : "碎齿暴徒 …" → "碎牙帮暴徒 Toothbreaker Thug"
  "Toothbreaker Scaletamer"      : "碎齿驯鳞者 …" → "碎牙帮驯鳞者 Toothbreaker Scaletamer"
  "Toothbreaker Planning Key"    : "碎齿者规划室钥匙 …" → "碎牙帮规划室钥匙 Toothbreaker Planning Key"
  "Toothbreaker Prison Key"      : "碎齿者监牢钥匙 …" → "碎牙帮监牢钥匙 Toothbreaker Prison Key"
  "Toothbreaker Security Key"    : "碎齿者安保钥匙 …" → "碎牙帮安保钥匙 Toothbreaker Security Key"
  "Toothbreaker Storage Key"     : "碎齿者储藏室钥匙 …" → "碎牙帮储藏室钥匙 Toothbreaker Storage Key"
  "Toothbreaker Throne Room Key" : "碎齿者王座室钥匙 …" → "碎牙帮王座室钥匙 Toothbreaker Throne Room Key"

D. Otherhood 族（OOS #3 / #58 / #170）—— 4 个键，统一「异姊会」：
  "Otherhood"          : "异缘会" → "异姊会"
  "Otherhood Brigand"  : "异域母性强盗 Otherhood Brigand" → "异姊会强盗 Otherhood Brigand"
  "Otherhood Brigands" : "他者会匪徒" → "异姊会强盗"
  "Otherhood Raider"   : "同袍会劫掠者 Otherhood Raider" → "异姊会劫掠者 Otherhood Raider"
  （"Otherhood of Fortune" → "幸运异姊会 Otherhood of Fortune" 已正确，保持）

E. Burnished 族 —— 2 个键，配合本轮批次：
  "Burnished Hand Plate" : "抛光手部板甲 …" → "辉手板甲 Burnished Hand Plate"
  "Burnished Seal"       : "锃亮印记 …" → "辉手印记 Burnished Seal"

F. Sanguinary 族（本轮新发现）—— 4 改 1 增，统一「赤血会」：
  "Sanguinary"                : "血裔会" → "赤血会"
  "Sanguinary Salutations"    : "血裔会的问候 …" → "赤血会的问候 Sanguinary Salutations"
  "Sanguinary Druid"          : "血腥德鲁伊" → "赤血会德鲁伊"
  "Sanguinary Correspondence" : "血书往来 …" → "赤血会信函 Sanguinary Correspondence"
  新增 "Sanguinary Warden"    : "赤血会守林者 Sanguinary Warden"（词表当前缺这个键，而 actor 存在）

G. Anachraenum 族（本轮新发现）—— 3 改 1 顺手：
  "Anachraenum Adventurer" : "阿纳克雷努姆冒险者 …" → "阿纳克瑞纽姆冒险者 Anachraenum Adventurer"
  "Anachraenum Member"     : "阿纳克雷努姆成员 …" → "阿纳克瑞纽姆成员 Anachraenum Member"
  "Anachraenum Medallion"  : "阿纳克拉埃努姆徽章 …" → "阿纳克瑞纽姆徽章 Anachraenum Medallion"
  "Anachraenum Aetherial"  : "阿纳克瑞纽姆 以太灵 Anachraenum Aetherial" → 中文里多一个空格，建议去掉 → "阿纳克瑞纽姆以太灵 Anachraenum Aetherial"

H. 建议给 glossary 加机械校验（承接 OOS #130）：同一英文首词（House Cevher* / Mutagist* / Toothbreaker* / Otherhood* / Sanguinar* / Anachraenum* / Burnished*）的所有词条，其中文译名必须共享同一前缀。本轮 30 个键全部是这条规则的违例，且 compendium 已先于 glossary 收敛，词表现在是唯一的分叉源。

I. 另需注意（非本族，供主控派单）：glossary "Bloody Gorge" → "血腥峡谷 Bloody Gorge"，但 compendium actors.Corpuleth.items.Bloody Gorge.name 是「血腥吞噬 Bloody Gorge」（那是个吞噬攻击，不是峡谷）。词表这一条是错的那边。

## F1

全部集中在 5-其他内容/glossary/glossary_ec.json（lang/cn.json 与 *.mjs / *.js 已逐个查过，两个 lang 包只有「血统」这类通用 UI 串，无任何血统专名；脚本侧 0 命中，无需改）。下列 14 个键的中文与实体包 name 字段直接冲突，词表是唯一分叉源：

【血统主名 ↔ Lineage 条目同词根】
1. "Signborn Lineage": "星兆血统 Signborn Lineage" → "印记裔血统 Signborn Lineage"
2. "Kivahr Lineage": "基瓦赫血统 Kivahr Lineage" → "基瓦尔血统 Kivahr Lineage"
3. "Wirrun Lineage": "维伦血统 Wirrun Lineage" → "威伦血统 Wirrun Lineage"
4. "Thornling Lineage": "荆棘裔血统 Thornling Lineage" → "荆芽灵血统 Thornling Lineage"
5. "thornling": "荆棘裔" → "荆芽灵"（全库 Thornling 闸门 荆芽灵 195 : 荆棘裔 0）
6. "Afflicted Thornling": "受折磨的荆棘裔" → "受折磨的荆芽灵"（实体包 actors.Sporix Host.tokenName 已是「受折磨的荆芽灵」）
7. "Hulg'run Lineage": "Hulg'run血统 Hulg'run Lineage" → "赫尔格伦血统 Hulg'run Lineage"（裸英文，实体包 ×7 全作「赫尔格伦血统」）
8. "Vrjnhar Lineage": "Vrjnhar 血统 Vrjnhar Lineage" → "弗尔金哈尔血统 Vrjnhar Lineage"（同上，裸英文）

【族名与实体包 name 冲突】
9. "Signarans": "希格纳兰人 Signarans" → "西格纳兰人 Signarans"（实体包 folders.Signarans 即「西格纳兰人 Signarans」；全库「希格纳兰」0 命中）
10. "Young Cheliceraeth": "幼年螯蛛以太兽 Young Cheliceraeth" → "幼年螯蛛艾斯 Young Cheliceraeth"
11. "Akonites": "乌头属植物 Akonites" → "阿肯体 Akonites"（把阿肯的构装体误认成植物学 aconite；实体包 name 已是「阿肯体」）
12. "The Signborn's Secret": "印记者的秘密 The Signborn's Secret" → "印记裔的秘密 The Signborn's Secret"（实体包 name 与两处正文引用均作「印记裔的秘密」）

【Arcturian 一族音译回归已定译「阿克图里安」——全库「阿克图里亚/阿克图瑞安」0 命中，这 6 条纯属词表遗留】
13. "Arcturians": "阿克图里亚人" → "阿克图里安人 Arcturians"
14. "Arcturian Alchemist": "阿克图里亚炼金术士 …" → "阿克图里安炼金术士 Arcturian Alchemist"
    "Arcturian Jail": "阿克图里亚监狱 …" → "阿克图里安监狱 Arcturian Jail"
    "Arcturian Respirator": "阿克图里亚呼吸器 …" → "阿克图里安呼吸器 Arcturian Respirator"
    "Arcturian Sailor": "阿克图里亚水手 …" → "阿克图里安水手 Arcturian Sailor"
    "Arcturian Sea Captain": "阿克图里亚海船队长 …" → "阿克图里安海船队长 Arcturian Sea Captain"
    "Arcturian Liquor": "阿克图瑞安烈酒 …" → "阿克图里安烈酒 Arcturian Liquor"
（以上 6 条的新值全部照抄实体包对应 name 字段现值，非我另拟。）

另建议（词表机械校验，呼应 ALL_OOS #130）：给 glossary 加一条「同一英文首词的所有词条其中文必须共享同一前缀」的校验——本族 8 条 Lineage 分叉、Arcturian 族 6 条音译分叉都会被它一次性抓出。

## F7

========================================================================
E0. 最重要的机制发现：glossary_ec.json 是**构建产物**，手改会被冲掉
========================================================================
3-常用脚本/tm/build_glossary.py 每次运行都会重建 glossary_ec.json：
  layer1 base = C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json（4646 条）
  layer2 harvest = 从 5-其他内容/english-baseline/{ember-0.6.0,crucible-0.10.1}
                   与两仓 compendium/cn 逐叶配对收割，**收割层压过 base 层**
所以现在词表里那些错值，其实是「上一次构建时包的状态」的化石。

我用同一套逻辑（同 baseline、同 walk_pairs、同 is_term、同多数票）做了重建预测，
对 88 个相关键的结论是：**56 键重建即自愈 / 21 键已一致 / 11 键是孤儿**。
（脚本：probes/f7_rebuild_forecast.py，可复跑）

==> 给主控的执行顺序建议（顺序错了会白做）：
  1) 先让本轮各单元的**包侧**批次全部落库；
  2) 再跑 `python 3-常用脚本/tm/build_glossary.py`  —— 下面 E2 的 56 键全部自动修好；
  3) 然后手改 E1 的 11 个孤儿键（**glossary_ec.json 与 base 词表两处都要改**，
     只改 glossary_ec 的话下次构建就退回去了）；
  4) 最后处理 E3（包侧还错、重建会把错值再固化的）与 E4（Earth 机制）。

========================================================================
E1. 必须手改的 11 个孤儿键（重建永远救不了；两个文件都要改）
========================================================================
文件 A：5-其他内容/glossary/glossary_ec.json
文件 B：C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json（base 层，值不带双语尾巴）

★ 真正的值改动（4 条）
  "Luma"                : "龙语"            -> A:"卢玛语 Luma"      B:"卢玛语"     【严重·阻断】
  "Mutagist Contingent" : "突变剂师分队"      -> A/B:"突变学派分队"
  "thornling"           : "荆棘裔"           -> A/B:"荆芽灵"
  "Cascal Arcden"       : "卡斯卡尔阿克登语"   -> A/B:"卡斯卡尔奥克登语"

★ 仅补双语位（值本身正确，但缺 " English" 尾巴，容易被后来者当成未裁决）
  "Draconic"     : "龙语"          -> A:"龙语 Draconic"        B 保持"龙语"
                   （必须与 Luma 同批改，否则两条还是长得一模一样）
  "Mutagist"     : "突变学派"       -> A:"突变学派 Mutagist"
  "Arcden"       : "奥克登语"       -> A:"奥克登语 Arcden"（全库 18:0，值正确）
  "arcden"       : "奥克登语"       -> 保持
  "Signaran Opal": "西格纳兰蛋白石"  -> A:"西格纳兰蛋白石 Signaran Opal"
  "Mutagist Scouts": "突变学派斥候"  -> 保持（已正确）

★ 待裁决后再改
  "Elder Goddess Spectra" : "上古女神丝珀特拉" -> 见 E5 的 Spectra 裁决

★ 核实后建议保留
  "Umber's Pass" : "安珀之径" -> 保持（Ember→安珀 已清零，全库安珀唯一指向 Umber）

========================================================================
E2. 重建即自愈的 56 键 —— 不要手改，跑 build_glossary.py 就行
========================================================================
（列出便于跑完后核对；括号内是重建会写入的值）
Mutagist 族6：Bombardier(突变学派爆击手) Bombadier(突变学派爆击手·英文错拼键)
  Excisor(突变学派切除者) Scout(突变学派斥候) Clothing(突变学派服装)
  Grenadier(突变学派投弹手) Vivisector(突变学派活体解剖师)
Toothbreaker 族9：Toothbreakers(碎牙帮) Rumors(碎牙帮传闻) Thug(碎牙帮暴徒)
  Scaletamer(碎牙帮驯鳞者) Planning/Prison/Security/Storage/Throne Room Key(碎牙帮X钥匙)
Lineage 族6：Signborn(印记裔血统) Wirrun(威伦血统) Kivahr(基瓦尔血统)
  Thornling(荆芽灵血统) Hulg'run(赫尔格伦血统) Vrjnhar(弗尔金哈尔血统)
Cevher 族4：Mausoleum(杰夫赫尔家族陵墓) Signet Ring(杰夫赫尔家族印戒)
  Funar Cevher(富纳尔·杰夫赫尔) Lyla Cevher(莱拉·杰夫赫尔)
Arcturian 族6(阿克图里安*) · Ordain 族4(奥尔丹*) · Ordani Ruffian(奥尔达尼恶棍)
Pathways 族2(通路地名志/通路侦察地图) · Wandren 族2(万德伦巡逻者/万德伦注视者)
Cascillian 族3(卡斯奇利亚*) · Aedir Wellstone(艾迪尔井石)
Akonites(阿肯体) Signarans(西格纳兰人) The Ordinate(审序院) The Arcageris(阿卡杰里斯)
Cruel Dragons(残酷龙) Kadhana Lizard(卡达纳蜥蜴) Helkas Green(赫尔卡斯绿地)
Afflicted Thornling(受折磨的荆芽灵) The Signborn's Secret(印记裔的秘密)
Young Cheliceraeth(幼年螯蛛艾斯) Cascilian(卡斯奇利亚)

⚠ 一条要留意：Cascilian 重建后会从「卡斯奇利亚人 Cascilian」变成「卡斯奇利亚 Cascilian」
（跟 name 字段走，丢掉「人」）。散文里 "a Cascilian" 该译「卡斯奇利亚人」，
请主控确认是接受，还是给它单独加一条 pending 覆盖。

========================================================================
E3. 包侧先修，否则重建会把错值再固化一遍（词表现在与包一致地错着）
========================================================================
这些键「已一致」，但一致在错的那一边。只改词表会造出新分叉，
只改包不改词表会被重建自动带过来——**必须成对做，且包先词表后**。

(1) Anachraenum 三键【一般】 folder `Anachraenum`=阿纳克瑞纽姆 是最强锚，
    全库 302 : 28 : 8。包侧共 36 叶要改：
      阿纳克雷努姆 28 叶 -> 阿纳克瑞纽姆
        ember.adventure / ember.crucible-adventure 各：
          journals.Gamemaster's Guide.pages.Patch 0.3.0/0.3.1/0.3.3.text
          journals.The Expedition Challenge.pages.An Upcoming Challenge.text
          journals.The Expedition Challenge.pages.Troublemaking Duo.text
          journals.Spellbreaker Tower.pages.Notable Inmates.text
          actors.Anachraenum Adventurer.name / .tokenName
          actors.{Adelyne Goss,Arcos Sarinland,Fernis Ossa,Anachraenum Adventurer}.items.Anachraenum Member.name
          （crucible 包另有 Kazra Steelshift / Leeph / Rorhim Iron-Cask / Sajor Velex 四个）
      阿纳克拉埃努姆 8 叶 -> 阿纳克瑞纽姆
          journals.The Expedition Challenge.pages.Closing Ceremonies.text
          journals.Flotsam Canal Market.pages.Falar's Studio (Upper).text
          items.Anachraenum Medallion.name
          actors.Fernis Ossa.items.Anachraenum Medallion.name
    词表随后自愈为：Adventurer=阿纳克瑞纽姆冒险者 / Member=阿纳克瑞纽姆成员 /
    Medallion=阿纳克瑞纽姆徽章。

(2) Silvered Aedir Warhammer【一般】包侧 2 叶 items.Silvered Aedir Warhammer.name
    = 「镀银的埃迪尔战锤」，与 Aedir 全库 398:0 的「艾迪尔」冲突。
    -> 「镀银的艾迪尔战锤 Silvered Aedir Warhammer」，词表随后自愈。

(3) 其余「死写法」在包内的残余（都属别的单元 scope，附精确路径便于派单，我未动）：
    突变派 4 叶  : {ember.adventure, ember.crucible-adventure}
                   journals.Repurposed Quarry.pages.{Gameplay Details, Chemical Storage}.text
    鲁玛林 7 叶  : Cultures.pages.Lumek.text ×2 / Organizations.pages.Flame Guard.text ×2 /
                   items.Lumarin Steel.name ×2 / crucible items.Lumarin Steel.description.private
    赫尔卡斯·格林 2 叶 : tables.Helkas Raider Moments.results.2-2.description ×2  (OOS #143)
    塞夫赫尔 1 叶 : crucible items.Kilner Notes.description.private              (OOS #49)
    螯蛛以太兽 1 叶: crucible actors.Young Cheliceraeth.biography.private
    埃迪尔 2 叶   : 见 (2)

========================================================================
E4. Earth：手改词表无效，要改的是 build_glossary.py
========================================================================
根因（已读码确认）：build_glossary.py 的 base 层加载写作
    base = {k: v for k, v in base_raw.items() if isinstance(v, str) and CJK.search(v)}
而 base 词表里 `"Earth": ["大地", "土"]` 是**列表**——多义已经建模了，却被
`isinstance(v, str)` 整条丢弃。于是 glossary_ec 的 `"Earth": "大地 Earth"`
其实完全来自收割 folders.Earth，与 base 无关。

三个义项各有独立权威锚点，缺一不可：
    元素义   = 大地   ← 2-Crucible汉化插件/lang/cn.json  SPELL.RUNES.Earth="大地"
                                                        SPELL.RUNES.EarthAdj="大地的"
    生物类义 = 土元素 ← crucible.taxonomy `Earth Elemental`=土元素，
                        兄弟 folder Fire/Water/Air=火/水/气元素
    行星义   = 地球   ← Introduction/Identity, Sex & Gender（K1 已修，gated 2 叶）

⚠ 定时炸弹：若 Z5 按 OOS #21 把 folders.Earth 从「大地 Earth」改成「土元素 Earth」，
   下一次 build_glossary.py 会**静默**把词表键 Earth 翻成「土元素 Earth」，
   于是 apply_tm 会拿土元素去套 Crucible 的元素符文，把 lang/cn.json 的大地体系撞坏。
   Z5 那个 folder 改动本身是对的，所以这条必须在它落库前后处理掉。

两个可选修法（请主控二选一）：
  方案甲（改脚本，推荐）：让 base 层保留 list 值，展开成带义项后缀的键：
      "Earth (element)": "大地 Earth"
      "Earth (creature-type)": "土元素 Earth"
      "Earth (planet)": "地球 Earth"
    并让收割层**不覆盖**带 " (" 后缀的键；同时把裸键 "Earth" 从收割结果里排除
    （加一个 AMBIGUOUS 集合，Earth/Fire/Water/Air 都该进去——它们同样一词多义）。
  方案乙（不改脚本）：在 glossary_ec.pending.json 或一个新的 overrides 文件里
    登记 Earth 为「禁止自动套用」，让 apply_tm/fill_missing 跳过裸 Earth。
无论哪种，**只在 glossary_ec.json 里手写三条是没用的**，下次构建就没了。

========================================================================
E5. 需要主控裁决的（我不下判断，只把全库分布摆出来）
========================================================================
(1) Spectra（魔法上古女神）—— 四分裂，且多数派是意译
      光谱 98 叶 / 丝珀特拉 3 叶 / 斯佩克特拉 4 叶（英文闸 \bSpectra\b=103 叶）
      name 字段站在「光谱」一边：journals.Deities.pages.Spectra.name=光谱 Spectra、
        actors.Spectra.name/tokenName=光谱、Shrine to Spectra=光谱圣祠、
        Tomb of Spectra's Chosen=光谱之选民的坟墓
      少数派：actors.Spectra.biography.public=丝珀特拉（×2）、
        crucible actors.Kazra Steelshift.biography.public=丝珀特拉、
        items.Spectra's Blessing.name=斯佩克特拉的祝福、
        journals.To Fall and Fall Again.pages.Magic Incarnate.text=斯佩克特拉
      我另查了 19 处 "Elder Goddess of Magic" 语境：全部作「光谱」，只有
      actors.Spectra.biography.public 那一处作「丝珀特拉」。
      —— 「光谱」是物理学普通名词当神名用，读起来确实不像名字，但它是压倒性多数
      且占住全部 name 字段。裁「光谱」则改 7 叶 + 词表 2 键；裁音译则改 100 叶 + 词表 5 键。
      词表 5 键：Spectra / Elder Goddess Spectra / Spectra's Blessing /
                Shrine to Spectra / Tomb of Spectra's Chosen
      （注：Elder Goddess Spectra 是孤儿键，无论裁哪边都要手改 + 改 base）

(2) Yakoshta Mine Track Switches —— 切换器 vs 道岔
      轨道切换器 10 叶（Yakoshta Mine journal 正文，33 处）
      轨道道岔 4 叶（tables 的 name+description）+ results 内 16 处「道岔」
      词表键 "Yakoshta Mine Track Switches"="雅科什塔矿井轨道道岔"（与 table name 一致）
      —— 表 name 与 journal 正文对打，两边都自洽。定一个后要同改 6 个 journal 叶
      + 该表 name/description/全部 results + 词表 1 键。

(3) Lumarin 词根 —— 卢马林 vs 鲁玛林（且与已定 Luma=卢玛 不同源）
      英文闸 \bLumarin\b=23 叶：卢马林 16 / 鲁玛林 7 / 卢玛林 0
      ⚠ name 字段站在少数派：items.Lumarin Steel.name=「鲁玛林钢」（两包）
      词表 "Lumarin Steel"="鲁玛林钢 Lumarin Steel"，与包一致地站在 7 那边
      —— 若按多数派裁「卢马林」，要改 7 叶 + 词表 1 键；
         若按词根一致性（Luma 已定为卢玛）裁「卢玛林」，则要改全部 23 叶。
      这是本轮 name 字段第 6 次站错边（已实证模式），建议不要默认信 name。

(4) Mutagen / Mutagenic 裸词（OOS #7）—— 词表侧无裸键，纯包侧裁决
      英文闸 \bMutagen(ic|s)?\b=31 叶：诱变 gated 24 / 诱变剂 4 / 突变剂 3 / 突变药剂 0
      词表里 Mutagenic* 一族已一致作「诱变*」（诱变病症/诱变配方/诱变介质瓶/诱变介质），
      且包内 name 字段全部是诱变*。建议裁「诱变剂」并补一条词表键
        "Mutagen": "诱变剂 Mutagen"
      —— 但这要先把包内 3 叶「突变剂」+ 2 叶「突变药剂」扫平，我未动。
      注意别和已定的组织名「突变学派 Mutagist」混：Mutagist=突变学派，Mutagen=诱变剂，
      两个词不同源，不要求共享前缀。

(5) Cheliceraeth Eye = 「螯肢眼」—— 族内前缀例外
      Cheliceraeth 主名=螯蛛艾斯，但道具 name 用「螯肢眼」（chelicera 的解剖学义）。
      包与词表一致，且作为身体部位道具讲得通。建议**保留**并在
      4-临时脚本/2026-08-12-fix/glossary_ec.disputes.json 留档，
      否则我交付的校验脚本每次都会报它。

========================================================================
E6. lang/ 与 .mjs 的核查结果：**无需任何改动**
========================================================================
  1-Ember汉化插件/lang/cn.json          扫 32 个死写法，0 命中
  2-Crucible汉化插件/lang/cn.json       仅 SPELL.RUNES.Earth="大地" /
                                        SPELL.RUNES.EarthAdj="大地的" —— 正确（元素义），保留
  1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs
      L66  "Luma": "卢玛语",     ← 已经是对的
      L75  "Draconic": "龙语",   ← 已经是对的
      —— 是 glossary_ec 该向 .mjs 看齐，不是反过来。改词表时请以这两行为准。
  两个 babele-mappings.js               0 命中

========================================================================
E7. 交付：机械校验脚本（主控点名要的）
========================================================================
probes/check_glossary_family.py

规则：**同一个专名英文 token 的所有词条，其中文必须共享同一个译名**——
不是简单的「首词前缀」，因为那条规则在本词表上会炸出约 700 条、几乎全是误报
（Light 光 vs Light Armor 轻型护甲、Fire 火元素 vs Alchemist's Fire 炼金火焰
都是**应该**随语境变的）。关键在于把「专名」和「普通英文词」自动分开：

    一个 token 算专名，当且仅当它在两仓英文语料里几乎从不小写出现
    （lowercase 出现数 / 总出现数 < --lower-ratio，默认 0.2）

"light"/"fire" 天天小写出现，"Mutagist"/"Cevher"/"Kivahr" 基本不会。
语料就是 compendium/en 两棵树，所以剧本长大后这个过滤器自己保持正确，
不需要维护任何人工专名表。

三类违例：
  A head-mismatch        token 有主条目（键就是 T / T+s / "The "+T），
                         别的含 T 的条目没带上主条目的中文  → 抓到 Mutagist/Signborn/Anachraenum
  B headless-no-consensus  ≥3 条共享 T、无主条目、且它们的中文毫无公共子串
                         （即没有任何两条对 T 的写法达成一致）→ 抓到 House Cevher
  C english-leak         剥掉双语尾巴后中文里还留着裸英文单词 → 抓到 Hulg'run血统

实测：主控点名的四族**全部命中**
  Mutagist      A 类，family=11，主条目"Mutagist"=突变学派，7 条违例
  Toothbreaker  A 类，family=9， 主条目"Toothbreakers"=碎牙者，9 条违例
  House Cevher  B 类，family=6， 6 条互不一致
  X Lineage     A 类，经 Signborn/Wirrun/Kivahr/Thornling/Hulg'run/Vrjnhar 六个主条目命中

全量输出 111 条（A=77 B=26 C=8），并**额外挖出简报里没有的 5 个已定译名污染**：
Arcturian(608 叶已定阿克图里安，词表 6 键错) / Ordain(1408 叶已定奥尔丹，4 键错) /
Pathways(458 叶已定通路，2 键错) / Wandren(284 叶，2 键错，其中 Watcher 被意译成「流浪注视者」) /
Ordani(656 叶，1 键错)。

用法：
  python check_glossary_family.py                    # 全量，有违例则 exit 1，可当发版闸
  python check_glossary_family.py --token Mutagist   # 单族
  python check_glossary_family.py --kind A           # 只看精度最高的一类
  python check_glossary_family.py --json out.json    # 机器可读

已知误报（约 3 成，建议加白名单后再挂进发版流程）：
  · Hallows / The Hallows —— 组织 vs 城区，有意区分（见上）
  · Abyssal —— 主条目选中了复数名词 "Abyssals"=深渊裔（族群），
               而形容词义就该是「深渊」，属主条目挑选的结构性误报
  · Cheliceraeth Eye —— 身体部位道具，见 E5(5)
  · Journeyman / Frost / Sprite / Beacon / Sanctuary / Warden 等
               —— 小写率刚好卡在阈值上的普通名词，可调 --lower-ratio 或补 STOP
建议：把上述白名单落到脚本旁边一个 family_allow.json 再挂闸，否则会天天红。

========================================================================
E8. 建议一并记进 PROJECT.md 的两条教训
========================================================================
1. **glossary_ec.json 不是手写文件，是构建产物。** 任何人手改它之前必须先看
   build_glossary.py 的层级策略，并确认改的键是不是孤儿；否则改动会在下一次
   构建时静默消失（Highgate 这次是侥幸——包侧先被扫平了，收割层才压住 base 的「海门」）。
2. **词表的错值 = 上一次构建时包的状态。** 所以修复顺序永远是「先包、后重建、
   最后补孤儿」。反过来做（先手改词表再扫包）会做两遍，而且中间态会让
   apply_tm 把旧值注回刚扫干净的叶子。

## F8

【5-其他内容/glossary/glossary_ec.json —— 与本批次直接绑定，不改会被 apply_tm / fill_missing 重新注回】
1. "Anachraenum Adventurer": "阿纳克雷努姆冒险者 Anachraenum Adventurer" → "阿纳克瑞纽姆冒险者 Anachraenum Adventurer"
2. "Anachraenum Member": "阿纳克雷努姆成员 Anachraenum Member" → "阿纳克瑞纽姆成员 Anachraenum Member"（＝原报告 #133）
3. "Anachraenum Medallion": "阿纳克拉埃努姆徽章 Anachraenum Medallion" → "阿纳克瑞纽姆徽章 Anachraenum Medallion"
4. "Anachraenum Aetherial": "阿纳克瑞纽姆 以太灵 Anachraenum Aetherial" → "阿纳克瑞纽姆以太灵 Anachraenum Aetherial"（中文中间多一个空格）
5. "The Armarium": "军械库 The Armarium" → "阿玛留姆 The Armarium"（#117）
6. "Helkas Green": "赫尔卡斯·格林 Helkas Green" → "赫尔卡斯绿地 Helkas Green"（#152①/#118）
7. "Lumarin Steel": "鲁玛林钢 Lumarin Steel" → "卢马林钢 Lumarin Steel"（#39/#53）
8. "Duskmaw's Gavel": "暮颚的法槌 Duskmaw's Gavel" → "暮噬的法槌 Duskmaw's Gavel"（#56）
9. "Chessman construct": "棋兵构装体" → "棋士构装体"（#104）
10. "Elevator Glyphstone": "电梯符石 Elevator Glyphstone" → "升降机符石 Elevator Glyphstone"（#180）
11. "Rotating Elevator": "旋转电梯 Rotating Elevator" → "旋转升降机 Rotating Elevator"（#180）
12. "Mine Elevator": "矿井电梯 Mine Elevator" → "矿井升降机 Mine Elevator"（#180）
13. "Yakoshta Mine Track Switches": "雅科什塔矿井轨道道岔 Yakoshta Mine Track Switches" → "雅科什塔矿井轨道切换器 Yakoshta Mine Track Switches"（#148）
14. "Monstrosities": "怪物 Monstrosities" → "畸怪 Monstrosities"（#160）
15. "Cascillian Marine": "卡斯奇利亚海军 Cascillian Marine" → "卡斯奇利亚海军陆战队员 Cascillian Marine"（#163）
16. "Cascillian Marine Officer": "卡斯奇利安海军军官 Cascillian Marine Officer" → "卡斯奇利亚海军陆战队军官 Cascillian Marine Officer"（#163，兼订正 安→亚）
17. "Cascillian Autotool": "卡斯奇利安自动工具 …" → "卡斯奇利亚自动工具 Cascillian Autotool"；"Cascillian Rebreather": "卡斯奇利安再呼吸器 …" → "卡斯奇利亚再呼吸器 Cascillian Rebreather"（安→亚，与 #164 同族）
18. "Arcturel Investigation": "阿克图雷尔调查" → "阿克图瑞尔调查"（雷→瑞）
19. "Lower Arcturel Mine Effect" 与 "Lower Arcturel Mine Effects"（值均为 "下层阿克图雷尔矿井效果"）→ 建议**删除这两个键**：英文名已过期，现表 name 是 "The Dives Mine Effects"＝矿渊矿井效应；留着会把「阿克图雷尔」重新注回（#171）
20. "Garganthus Hide": "加甘图斯兽皮 Garganthus Hide" → "加甘萨斯兽皮 Garganthus Hide"；"Garganthus Tunnel": "加冈瑟斯隧道 …" → "加甘萨斯隧道 Garganthus Tunnel"（第三种写法，#178/#206）
21. "Signarans": "希格纳兰人 Signarans" → "西格纳兰人 Signarans"（#10；包内已修，词表仍是错的那一边）
22. "Kadhana Lizard": "卡达娜蜥蜴 Kadhana Lizard" → "卡达纳蜥蜴 Kadhana Lizard"（#152②，包内 卡达纳 14 : 卡达娜 0）
23. "Cruel Dragons": "残酷巨龙 Cruel Dragons" → "残酷龙 Cruel Dragons"（#112/#126，包内 name 与全部正文都是残酷龙）
24. "Earth": "大地 Earth" —— 该条按元素义建立却被套到指现实地球的 Earth 上（#132）。建议改键为 "Earth (element)": "土元素 Earth"，并新增 "Earth (planet)": "地球"。注意：包内 (folders).Earth 我已在批次里改成「土元素 Earth」，词表不同步会被注回
25. "Tauric": "牛族 Tauric" —— 与 actor name「陶里克 Tauric」同键异义（#205 的裸英文正是卡在这里）。建议拆键：保留 "Tauric (taxonomy)": "牛族"，新增 "Tauric": "陶里克 Tauric"
26. "Mutagist Bombadier": "突变学派投弹手" —— 键按上游 tokenName 的错拼（少一个 r）建立，值又与 "Mutagist Grenadier"＝突变学派投弹手 撞车、与 "Mutagist Bombardier"＝突变派爆击手 冲突（#127/#130/#134）。建议值改为 "突变学派爆击手"，与 "Mutagist Bombardier" 一并对齐（包内两个字段现已都是「突变学派爆击手」）
27. "Fae": "精灵 Fae" 与 "Fey"＝妖精 并存 —— 我判为需主控裁（#162），**不改**，但建议加注释，防止后来者误合并
28. "Umber's Pass": "安珀之径"（#199）—— 现「安珀」在全库唯一指向 Umber，冲突已消解；仅建议改成不含「珀」的写法（如「翁伯之径 Umber's Pass」）以绝后患。低优先

【1-Ember汉化插件/lang/cn.json —— #43，需与包内 Warlock.name 已改为「邪术师 Warlock」同步】
29. EMBER.DEITY.FIELDS.warlocks.label: "术士契约" → "邪术师契约"
30. EMBER.DEITY.FIELDS.warlocks.hint: 其中的「术士」→「邪术师」
31. EMBER.DEITY.FIELDS.sorcerers.hint: 保持「术士」（Sorcerer 侧不变），但请人工确认该 hint 未混入 Warlock 语义

【.mjs】本轮未发现需要改动的 mappings.mjs 条目。

【机械校验建议（#130 原报告提出，我复核同意）】给 glossary_ec 加一条同族前缀一致性校验：同一英文首词的所有词条，其中文译名必须共享同一前缀。上面第 1–4、15–17、20 条都是这条规则能自动抓到的。