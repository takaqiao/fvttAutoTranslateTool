# GLOSSARY-061 · ember 0.6.0 → 0.6.1 术语裁决表

> 本表是第二阶段（81.9 万字符正文翻译）的**唯一术语依据**。十个译者一律照抄，
> 不许各自另创。凡表里没有的新词，先回来加表，再动译文。
>
> 产出者：第一阶段 · 术语裁决单元 ｜ 日期：2026-08-18
> 输入：`scratchpad/delta_em_060_to_061.json`（三桶差量）· `scratchpad/em061/`（0.6.1 真身抽的英文十包）
> 库：两个插件仓 `compendium/en` × `compendium/cn` 24 对包 + `lang/{en,cn}.json` + `5-其他内容/glossary/glossary_ec.json`

---

## 0. 前置自证（本表的结论建立在这四条上，任何一条不过，下面都不作数）

| # | 断言 | 真值 | 实测 | 结果 |
|---|---|---|---|---|
| A0 | 三桶逐包计数 | adventure 804/40/129 · crucible-adventure 1255/274/204 · crucible-adversary 2/0/1 | 逐包相等 | PASS |
| A1 | **切出来的名称叶条数**（末段 ∈ {`name`,`label`}） | 1122 | 1122（1100 `name` + 22 `label`） | PASS |
| A2-b | **切对了地方**：名称叶里不得含正文特征（`<tag`／`@UUID[`／`@Condition[`／`@Embed[`） | 0 | 0 | PASS |
| A2-c | 名称叶值必须是字符串 | 0 例外 | 0 | PASS |
| A2-d | 补丁说明点名的新专名必须落在切片里 | 15/15 | 10/15 → **查清后 15/15**，见 §1 | PASS（附订正） |
| A2-e | 名称位应当短 | —— | 中位 13 字符 · 最长 30 · 合计 **15 227** 字符（与任务给的「15K 字符」相符） | 佐证 |

**切条数与切地方两件都断言了**（本项目在这上头栽过四次）。脚本：`probe_names.py`。

撞车判据自身也做了正反自证（`check_collisions.py` 开头）：
拿**上一轮真栽过的那对**（`Overrun` / 围攻）做阳性对照，英文方向必须抓到「冲撞 Overrun」、
中文方向必须抓到「冲撞已被 Ram/Overrun 占」，两个方向都 PASS；另拿一个伪造串做阴性对照，必须不报。
**判据抓不到已知真值，就没有资格判未知。**

---

## 1. A2-d 的 5 条「未命中」，逐条查清（这一段比表本身更要紧）

上游补丁说明与包体**对不上**，照说明翻会全错。逐条：

| 补丁说明写的 | 包体真实写法 | 证据 | 处置 |
|---|---|---|---|
| `Kryban` | **`Kyrban`** | 全量差量里 `Kyrban` 306 处、`Kryban` 仅 2 处且**两处都在补丁说明自己那页**里 | 以包体为准，`Kyrban`→凯尔班 |
| `Kadra-Zann`（带连字符） | **`Kadra Zann`**（空格） | journal 名 `/journals/Kadra Zann/name`；库里 0.6.0 已有 `Kadra Zann`→卡德拉赞恩 | 以包体为准 |
| `Amersap Queen` | **`Amerasp Queen`** | 补丁说明自己拼错了 | —— |
| `Casir Cats`（复数） | **`Casir Cat`**（单数） | actor 名 | —— |
| `Sanguinary Warden` 列为「新敌手」 | **0.6.0 就有**，且**早有译名「赤血会守林者 Sanguinary Warden」** | `1-Ember汉化插件/compendium/cn/ember.adventure.json` 的 actors | **直接沿用，不许重译** |
| `Caryx Savannah` / `Elenain Delta` | 只出现在补丁说明正文里，**不产生任何新名称叶** | 是区域地图的绘图更新；`Elenain Delta` 库里已有「埃勒奈因三角洲」 | 无需新裁；`Caryx Savannah` 待它真的进包再裁 |

**另外两条同样重要的「查出来不是新词」：**

* `Arbore Sanctorus` —— 补丁说明列为新兴趣点，**库里 0.6.0 就有译名「圣树庇护所 Arbore Sanctorus」**。直接沿用。
* `Cor'ak` / `Fej` —— 库里早有「科拉克 Cor'ak」（10 处）「费伊杰 Fej」（6 处）。0.6.1 只是把 Serethus 正式介绍为 Cor'ak，**不是新词**。

`Grayling` 与 `Grayce` **没有关系**：`Grayce` 在本项目的 Ember/Crucible 库里 **0 命中**（它属于另一条 SoG 线）。
`Grayling` 是 0.6.1 新造的蛾翅妖精生物，与 `Casia`／`Casir` 一样属于本次血林生态。

---

## 2. 本表怎么用：形态是硬约束，别照抄错通道

库里的形态是**按字段位置**定死的，实测如下（照抄错了主闸的 `no_bilingual_tail` 会红）：

| 字段位置 | 形态 | 实测样本 |
|---|---|---|
| actors / items / effects / actions / journals / journal pages / folders / scenes / macros / **scene notes** | **双语并列**「中文 English」 | `深渊先驱 Abyssal Harbinger`、`第1章事件 Chapter 1 Events`、`墓刃湖岸 Grave Blade Shore` |
| **outcome labels** | **纯中文** | `Alchemical Reagent Withheld` → `扣下炼金试剂` |
| **scene levels / sounds / regions（含 behaviors）** | **纯中文** | `Lower Pools`→`下层水池`、`Waterfall`→`瀑布`、`Surface: Metal`→`地表：金属` |
| **journal categories** | **纯中文** | `Overview`→`概览`、`Arcturel`→`阿克图瑞尔` |
| **encounterTokens** | **纯中文** | `Danith`→`丹尼思`、`Friendly Ooze`→`友善软泥怪` |
| lang/cn.json、ember-hardcoded-cn.mjs | **纯中文** | `Suppress Weather`→`抑制天气`、`Golden Flats Day`→`金色平原 · 白天` |

**前缀族的既定式样**（一个字都别改）：
`Surface: X`→`地表：X` · `Interior: X`→`室内：X` · `Transition: X`→`过渡：X` ·
`Spawn: X`→`出生点：X` · `Vista: X`→`远景：X Vista: X`（scene 名带双语尾） ·
`X Day/Night`→`<中文X> · 白天/夜晚` · `X Fight/Combat Section N`→`<中文X>战斗 · 第 N 段`。
⚠ 冒号是**全角**`：`，`·` 前后各一个半角空格。

---

## 3. 音译习惯（先抽 20 个既有人名归纳出来的，新名字照这个走）

`Zodi Trask`→佐迪·特拉斯克 · `Edivel Sprout`→埃迪维尔·斯普劳特 · `Avwynn Taol`→阿芙温·陶尔 ·
`Sadri Zhalimorne`→萨德里·扎利莫恩 · `Corvana Vortest`→科瓦娜·沃特斯特 · `Liestra Grann`→莉耶丝特拉·格兰 ·
`Steros Kraver`→斯特罗斯·克拉弗 · `Del Kalais`→德尔·卡莱斯 · `Serethus`→塞雷苏斯 · `Kern`→克恩

归纳出三条：
1. **名姓之间用间隔号 `·`**，不用空格、不用点。
2. **音译用新华社通用汉字**（w→芙/瓦、th→斯/苏、-us→斯）。
3. **可解的姓氏意译、纯音的姓氏音译** —— `Mira Wavehorn`→米拉·**波角**、`Amalthea Stonecraft`→阿玛尔忒亚·**石艺**、
   `Kazra Steelshift`→卡兹拉·**钢移**、`Sin Marmot`→辛·**旱獭**；而 `Trask`／`Kraver` 这种无解的走音译。

生物名则**以意译为主**（`Thornling`→荆芽灵、`Trickadee`→戏诈雀、`Gore Bird`→血鸟、`Sporix Host`→孢体宿主、
`Corpuleth`→尸团怪），**唯独已被音译钉死的族名跟着走**（`Amerasp Grove`→阿梅拉斯普林地 ⇒ `Amerasp`→阿梅拉斯普）。

---

## 4. 生物群系 / 地名的风格基线（新地名对齐它们）

`Redrak Fields`→雷德拉克原野 · `Golden Flats`→金色平原 · `Arctus Plateau`→阿克图斯高原 ·
`Bloodwoods`→血林 · `Verdant Paths`→翠绿径 · `Mycelian Expanse`→菌丝旷野 · `Sinkhole Depths`→天坑深渊 ·
`Splinter Canyons`→碎裂峡谷 · `Rustvar Valleys`→鲁斯特瓦尔山谷 · `Elenain Delta`→埃勒奈因三角洲 ·
`Amerasp Grove`→阿梅拉斯普林地 · `Kadra Zann`→卡德拉赞恩 · `Talei`→塔莱 · `Arbore Sanctorus`→圣树庇护所

规律：**专名音译 + 地貌通名意译**（原野/平原/高原/三角洲/林地/峡谷/山谷），
**纯意象名整体意译**（血林、翠绿径、天坑深渊）。
`Ruby Grove` 按这条走 → **红宝林地**（`Ruby`→红宝是库内既定，且 0.6.1 正文确认 Ruby 是一种**果酱状药物**，不是宝石）。

---

## 5. 撞车检测结果（双向都跑了）

227 条候选，双向判据报出 **10 条**，逐条定性如下 —— **没有一条需要改译文**，
其中 1 条是判据阶段就抓出来并当场改掉的真撞车（不在这 10 条里）：

**真撞车（已改）：** `Primordials` 原拟「原初生物」→ 词表里 `Primordial Creatures` 已占该中文 ⇒ 改为 **原初族**。

余下 10 条见 `collisions.json`，分三类：
* **判据取核造成的假阳性 3 条**（`补丁 0.6.1`／`火把 1`／`效果`：剥并列尾时把版本号、序号、单复数一起剥掉了）；
* **库里早就存在的合并 6 条**（`Overview`/`At a Glance`→概览、`Upper`/`Upper Level`→上层、`Lower`/`Lower Level`→下层、
  `Study`/`Study Chamber`→书房、`Gloom`/`Gloomy`→阴郁）—— **不是本轮引入的**，本轮只是又落一个 EN 进同一个桶；
* **本轮主动避开的 1 条**：`Storage Chamber` 没写「储藏室」（`Storage` 已占），改用 **储藏厅**。
  ⚠ 张力照实写：库里 `Study Chamber` 恰恰是**写成了**「书房」（与 `Study` 合并）。两种做法库里都有先例，
  我按任务里的硬规矩（新中文不许撞）选了不合并；owner 若想要一致性可以一键改回「储藏室」。

---

## 6. 需项目所有者拍板（4 组 + 1 条存疑）

> 下面每一条都**给了暂定值**，第二阶段照暂定值先翻不会卡住；
> 拍板后改动都是机械替换，成本可控。**不拍板的代价写在每条后面。**

### 6.1 `Doomsayer` / `Doomsayers` —— 暂定「末日预言者」

影响面：名称位 7 条（`Doomsayer Acolyte`／`Shadewright`／`Barracks`／`Trail Map ×4`／folder `Doomsayers`），
正文里数百处。**这是本轮影响面最大的一条。**

| 候选 | 好处 | 代价 |
|---|---|---|
| **末日预言者**（暂定） | 语义最准、零歧义 | 5 字，做前缀时很长（末日预言者侍僧／末日预言者兵营／末日预言者路径图（完整）） |
| 谶言者 | 3 字，古雅，前缀好看 | 「谶」字冷僻，玩家可能不认 |
| 灾言者 | 3 字，好认 | 「灾」偏结果、「doom」偏预言，语义轻微偏移 |

### 6.2 `Grayling` —— 暂定「灰蛾灵」

依据：0.6.1 正文写它是「蛾翅、触角、琥珀色球状复眼」的妖精，遗骸是玻璃状晶尘（`Grayling Dust`）；
`-ling` 在库里的既定对法是 `Thornling`→荆芽**灵**。

| 候选 | 好处 | 代价 |
|---|---|---|
| **灰蛾灵**（暂定） | 与荆芽灵同族；抓住了蛾翅这个最显著特征 | 「Gray」被读成灰色，而原文的 gray 也可能指它的皮色，属于**解释性翻译** |
| 格雷林 | 纯音译，零解释风险 | 与库里生物名以意译为主的做法不合 |
| 灰翼精 | 不把 moth 坐实 | 「精」在库里未用作 -ling 的对法 |

### 6.3 `Primordial` 词根 —— 暂定「原初」

影响面：`Primordial Weight／Grace／Spores／Buzzing`、`The Primordial Grove／Fragments`、
folder `Primordials`、scene level `Primordian Monument`。

* 暂定：**原初**（对齐词表里既有的 `Primordial Creatures`→原初生物、`Primordial`（语言）→原初语）。
  folder `Primordials` 因为撞了 `Primordial Creatures`，取 **原初族**。
* 另一条路：把它绑到神名上 —— `Primordis`→普里莫迪斯 ⇒ 「普里莫迪斯之重」。
  好处是玩家一眼看出这些能力来自哪位神；代价是每个名字都变长 5 字，且与既有的「原初语／原初生物」割裂。
* **代价**：现在不拍，第二阶段写下去的是「原初」，回头改要动约 8 个名称位 + 正文里所有 `Primordial` 复合词。

### 6.4 `Queen's Ruby` —— 暂定「蜂后红宝」

它是从 `Amerasp Queen`（阿梅拉斯普蜂后）身上提炼的违禁品。
「蜂后红宝」指向明确；「女王红宝」更像街头黑话的口气，但会丢掉「蜂」。两条都说得通。

### 6.5 存疑一条：scene sound `Sphere`

正文里对应的是 Kadra Zann 里申特人造的「**Fin'ian Sphere**」神器（`Fin'ian` 本身也是 0.6.1 新词，属正文层）。
暂定音效名译作 **球体**；等第二阶段把 `Fin'ian` 裁下来之后，若要写成「芬伊安球体」再回改（1 条叶，成本极低）。

### 6.6 顺带报一条**既有译名的事实性错误**（不归本单元改，请 owner 定夺）

`The Silent Pack` 库里译作 **寂静狼群**。0.6.1 的正文把它讲明白了：
> "The Silent Pack — the powerful group of **Casia** on Primordis"

`Casia` 库里是「卡西娅」，是**猫科**存在（`Nimaelle` 就是一头十英尺长的巨猫，`Casir Cat` 是同族小型近亲）。
**「狼群」是事实错误**。建议改「寂静之群」或「寂静猫群」。影响面 3 处名称位 + 正文若干。

---

## 7. 裁决表


### 人名（新角色）

| EN | CN（可直接照抄） | 依据 / 出处 |
|---|---|---|
| `Kyrban` | **凯尔班 Kyrban** | 人名·新角色（Arcturian 法术使用者，末日预言者首脑）。⚠ 补丁说明写作 Kryban，包体实为 Kyrban（306:2），以包体为准 |
| `Maevren` | **梅芙伦 Maevren** | 人名·新角色（塔莱的血林追踪者）。Avwynn→阿芙温 同款 w→芙 |
| `Verno Kreed` | **韦尔诺·克里德 Verno Kreed** | 人名·新角色（塔莱的药贩） |
| `Proctus Caylas` | **普罗克图斯·凯拉斯 Proctus Caylas** | 人名·新角色（年长 Arcturian） |

### 生物 / 敌手

| EN | CN（可直接照抄） | 依据 / 出处 |
|---|---|---|
| `Grayling` | **灰蛾灵 Grayling** | 生物·新敌手（蛾翅妖精，琥珀色球状眼、触角、玻璃质残骸）。⚠ 待拍板 |
| `Shadebranch` | **影枝 Shadebranch** | 生物·新敌手（漆黑虬曲的树怪）。Shade→影（阿加瑟罗斯之影） |
| `Vespinoth` | **韦斯皮诺斯 Vespinoth** | 生物·新敌手（根系＋赤色琥珀的合体，体内嵌着阿梅拉斯普蜂后尸骸）。随 Amerasp→阿梅拉斯普 走音译 |
| `Amerasp` | **阿梅拉斯普 Amerasp** | 生物·新敌手（黑红小蜂）。沿用 Amerasp Grove 阿梅拉斯普林地 的既定音译 |
| `Amerasp Queen` | **阿梅拉斯普蜂后 Amerasp Queen** | 生物·新敌手 |
| `Amerasp Swarm` | **阿梅拉斯普群集 Amerasp Swarm** | 生物·新敌手。Swarm→群集（库内 9 处） |
| `Casir Cat` | **卡西尔猫 Casir Cat** | 生物·新敌手（会影步的小猫） |
| `Doomsayer Acolyte` | **末日预言者侍僧 Doomsayer Acolyte** | 生物·新敌手。Acolyte→侍僧（库内既定）。⚠ Doomsayer 待拍板 |
| `Doomsayer Shadewright` | **末日预言者影匠 Doomsayer Shadewright** | 生物·新敌手。-wright→匠。⚠ Doomsayer 待拍板 |

### 地名 / 区域 / 结构名

| EN | CN（可直接照抄） | 依据 / 出处 |
|---|---|---|
| `Ruby Grove` | **红宝林地 Ruby Grove** | 地名／任务名。Ruby→红宝（库内既定，指果酱状药物而非宝石）；Grove→林地（Amerasp Grove 阿梅拉斯普林地） |
| `Amerasp Caves` | **阿梅拉斯普洞窟 Amerasp Caves** | 地名·Kadra Zann 页。Cavern→洞窟 |
| `Kyrban’s Chancery` | **凯尔班的文书院 Kyrban’s Chancery** | 地名·Kadra Zann 页（法师私人书斋）。Study 书房 已被占，故用文书院 |
| `Doomsayer Barracks` | **末日预言者兵营 Doomsayer Barracks** | 地名·Kadra Zann 页。Barracks→兵营（库内既定） |
| `Vespinoth Pens` | **韦斯皮诺斯兽栏 Vespinoth Pens** | 地名·Kadra Zann 页 |
| `Vespiary` | **蜂巢 Vespiary** | 地名·Kadra Zann 页（vespiary＝胡蜂巢） |
| `Atrium Floor` | **中庭层 Atrium Floor** | 地名·Kadra Zann 页 |
| `Ground Atrium Balcony` | **地面层中庭楼座 Ground Atrium Balcony** | 地名。Balcony→楼座、Ground Level→地面层 均库内既定 |
| `Upper Atrium Balcony` | **上层中庭楼座 Upper Atrium Balcony** | 地名。Upper Level→上层 |
| `Storage Chamber` | **储藏厅 Storage Chamber** | 地名。⚠ 储藏室 已被 Storage 占（库内 11 处），故换字 |
| `Dungeon` | **地牢 Dungeon** | 地名·Kadra Zann 页 |
| `Corrupted Nest` | **堕化巢穴 Corrupted Nest** | 地名／场景 region。⚠ Corrupted→堕化（库内既定），不要写腐化 |
| `Chapter 4 Events` | **第4章事件 Chapter 4 Events** | journal 名。对齐 Chapter 1 Events 第1章事件 |
| `Patch 0.6.1` | **补丁 0.6.1 Patch 0.6.1** | journal page。对齐 Patch 0.1.1〜0.6.0 补丁 x.y.z 系列 |

### 任务页 / 事件页

| EN | CN（可直接照抄） | 依据 / 出处 |
|---|---|---|
| `Ill Met in Talei` | **塔莱恶遇 Ill Met in Talei** | 任务页名 |
| `Local Experts` | **本地行家 Local Experts** | 任务页名 |
| `The Primordial Grove` | **原初林地 The Primordial Grove** | 任务页名。⚠ Primordial 词根待拍板 |
| `Visions of Doom` | **厄运幻象 Visions of Doom** | 任务页名 |
| `Where Shadows Lie` | **暗影潜藏之处 Where Shadows Lie** | 任务页名 |
| `Return to Talei` | **重返塔莱 Return to Talei** | 任务页名 |
| `Capricious Concerns` | **反复无常的顾虑 Capricious Concerns** | 任务页名 |
| `The Ruby Trade` | **红宝贸易 The Ruby Trade** | 任务页名 |
| `Just a Taste` | **浅尝一口 Just a Taste** | 任务页名 |
| `Family Feud` | **家族宿怨 Family Feud** | 第4章事件页名 |
| `Proving Your Metal` | **真金试炼 Proving Your Metal** | 第4章事件页名（metal/mettle 双关，奥肯弯 Ambral 铜棒比试） |
| `Slipping Into Darkness` | **坠入黑暗 Slipping Into Darkness** | 第4章事件页名 |
| `Traversing the Bloodwoods` | **穿越血林 Traversing the Bloodwoods** | 第4章事件页名。Bloodwoods→血林 既定 |

### 物品 / 装备

| EN | CN（可直接照抄） | 依据 / 出处 |
|---|---|---|
| `Ambral Bronze` | **安布拉尔青铜 Ambral Bronze** | 物品。Ambral＝奥肯加德特产魔法金属，专名音译 |
| `Grayling Dust` | **灰蛾灵粉尘 Grayling Dust** | 物品。⚠ 尘土 已被 Dust 占，故用粉尘 |
| `White Aspen Wand` | **白杨魔杖 White Aspen Wand** | 物品。Wand→魔杖 既定 |
| `Shadebranch Missive` | **影枝信笺 Shadebranch Missive** | 物品 |
| `The Primordial Fragments` | **原初残章 The Primordial Fragments** | 物品（典籍）。⚠ Primordial 词根待拍板 |
| `Hodge's Heirloom` | **霍奇的传家宝 Hodge's Heirloom** | 物品 |
| `Queen's Ruby` | **蜂后红宝 Queen's Ruby** | 物品（阿梅拉斯普蜂后提炼的违禁品） |
| `Kadra Zann Skeleton Key` | **卡德拉赞恩万能钥匙 Kadra Zann Skeleton Key** | 物品。Kadra Zann→卡德拉赞恩 既定 |
| `Kyrban's Chancery Key` | **凯尔班文书院钥匙 Kyrban's Chancery Key** | 物品 |
| `Vespine Domino` | **蜂面面具 Vespine Domino** | 物品。⚠ 严格对齐既有 Feline Domino 猫面面具 |
| `Amber Staff` | **琥珀法杖 Amber Staff** | 物品。Amber→琥珀、Staff→法杖 均既定 |
| `Amber Dart` | **琥珀飞镖 Amber Dart** | 武器 |
| `Amber Dirk` | **琥珀短刀 Amber Dirk** | 武器。匕首 Dagger／短剑 Shortsword 已占，故用短刀 |
| `Amber Sputum` | **琥珀唾液 Amber Sputum** | 攻击 |
| `Silver Sword` | **银剑 Silver Sword** | 武器（赛洛克弓手新增） |
| `Pitchvine Net` | **沥青藤网 Pitchvine Net** | 武器。Net→网 既定 |
| `Hunting Trap` | **狩猎陷阱 Hunting Trap** | 物品 |
| `Doomsayer Trail Map (1)` | **末日预言者路径图（1） Doomsayer Trail Map (1)** | 物品 |
| `Doomsayer Trail Map (2)` | **末日预言者路径图（2） Doomsayer Trail Map (2)** | 物品 |
| `Doomsayer Trail Map (3)` | **末日预言者路径图（3） Doomsayer Trail Map (3)** | 物品 |
| `Doomsayer Trail Map (Complete)` | **末日预言者路径图（完整） Doomsayer Trail Map (Complete)** | 物品 |
| `Nimaelle's Blessing` | **尼梅尔的祝福 Nimaelle's Blessing** | 物品。Nimaelle→尼梅尔（库内神祇页既定） |

### 能力 / 攻击 / 法术 / 效果 / 宏 / outcome 标签

| EN | CN（可直接照抄） | 依据 / 出处 |
|---|---|---|
| `Vespine Form` | **蜂形 Vespine Form** | 能力。⚠ 严格对齐既有 Feline Form 猫形 |
| `Aegis of Shadow` | **暗影庇佑 Aegis of Shadow** | 能力（阿芙温·陶尔，5e 侧） |
| `Luminous Strike` | **辉耀打击 Luminous Strike** | 能力。⚠ Luminous 分叉：5e 侧＝辉耀、Crucible 侧＝明光；本条只在 ember.adventure（5e 侧），故取辉耀 |
| `Radiant Teleport` | **光耀传送 Radiant Teleport** | 能力。Radiant→光耀、Teleport→传送 既定 |
| `Ward Against the Beyond` | **彼界防护 Ward Against the Beyond** | 能力。Ward→防护 既定 |
| `Worldfire Cataclysm` | **世界之火浩劫 Worldfire Cataclysm** | 能力 |
| `Bite of the Eclipse` | **日蚀之咬 Bite of the Eclipse** | 能力（尼梅尔） |
| `Shadow Breath` | **暗影吐息 Shadow Breath** | 能力 |
| `Shadow Claw` | **暗影之爪 Shadow Claw** | 能力 |
| `Shadow Illusion` | **暗影幻象 Shadow Illusion** | 能力 |
| `Split Form` | **分裂形态 Split Form** | 能力。Split→分裂、Form→形态 既定 |
| `Unweave Reality` | **解织现实 Unweave Reality** | 能力 |
| `Weaver of Shadow` | **暗影编织者 Weaver of Shadow** | 能力 |
| `Inscrutable Mind` | **高深莫测的心智 Inscrutable Mind** | 能力。Inscrutable→高深莫测 既定 |
| `Primordial Weight` | **原初之重 Primordial Weight** | 能力。⚠ Primordial 词根待拍板 |
| `Primordial Grace` | **原初优雅 Primordial Grace** | 能力。⚠ 同上 |
| `Primordial Spores` | **原初孢子 Primordial Spores** | 能力。⚠ 同上 |
| `Primordial Buzzing` | **原初嗡鸣 Primordial Buzzing** | 能力。⚠ 同上 |
| `Shadowstep` | **暗影步 Shadowstep** | 能力（卡西尔猫） |
| `Greater Shadowstep` | **高等暗影步 Greater Shadowstep** | 能力。Greater→高等 既定 |
| `Mercurial Burst` | **变幻迸发 Mercurial Burst** | 能力。Burst→迸发（术法迸发） |
| `Tenebrous Wisp` | **幽暗鬼火 Tenebrous Wisp** | 能力 |
| `Fey Invocations` | **妖精祈唤 Fey Invocations** | 能力。Fey→妖精 既定 |
| `Phantasmal Force` | **幻影之力 Phantasmal Force** | 5e 法术。对齐 Phantasmal Killer 幻影杀手 |
| `Call Lightning` | **唤雷术 Call Lightning** | 5e 法术 |
| `Summon Undead` | **召唤不死生物 Summon Undead** | 5e 法术。Summon→召唤、Undead→不死生物 既定 |
| `Expert Tracker` | **追踪专家 Expert Tracker** | 能力（梅芙伦） |
| `Hive Mind` | **蜂巢心智 Hive Mind** | 能力 |
| `Constrict` | **绞缠 Constrict** | 能力 |
| `Clawed Branch` | **利爪枝桠 Clawed Branch** | 攻击 |
| `Cloud of Insects` | **虫云 Cloud of Insects** | 能力 |
| `Death Bed Swarm` | **临终群集 Death Bed Swarm** | 能力（死亡时爆出蜂群） |
| `Pillar Blast` | **柱形爆破 Pillar Blast** | 能力。Blast→爆破 既定 |
| `Sap Blast` | **树液爆破 Sap Blast** | 能力 |
| `Gnarled Claw` | **虬结之爪 Gnarled Claw** | 攻击。Claw→爪 既定 |
| `Sting` | **螫刺 Sting** | 攻击 |
| `Stings` | **群螫 Stings** | 攻击（群集版） |
| `Shimmer` | **微光闪耀 Shimmer** | 能力（阿梅拉斯普过光时如火闪烁） |
| `Effervescent Death` | **沸腾之死 Effervescent Death** | 能力（灰蛾灵） |
| `Extract Dart` | **萃取飞镖 Extract Dart** | 能力（灰蛾灵） |
| `Jumper` | **跳跃者 Jumper** | 能力（卡西尔猫） |
| `Born Ready` | **生而备战 Born Ready** | 能力（戏诈雀） |
| `Clumsy Headbutt` | **笨拙头槌 Clumsy Headbutt** | 攻击。Headbutt→头槌 既定 |
| `Flapping Leap` | **扑翼跃击 Flapping Leap** | 能力。Leap→跃击 既定 |
| `Lay Egg` | **产卵 Lay Egg** | 能力（戏诈雀） |
| `Geyser` | **间歇泉 Geyser** | 无尽泉涌石的动作 |
| `Caught in Net` | **落网 Caught in Net** | 效果 |
| `Changed Appearance` | **外貌已改变 Changed Appearance** | 效果（表象 Seeming 的效果） |
| `Clouded` | **笼罩 Clouded** | 效果（虫云） |
| `Grappled and Restrained` | **被擒抱且受缚 Grappled and Restrained** | 效果。被擒抱／受缚 均库内既定状态名 |
| `Seeing Illusions` | **看见幻象 Seeing Illusions** | 效果 |
| `Trapped` | **落入陷阱 Trapped** | 效果（狩猎陷阱） |
| `Egg Delivery` | **卵体植入 Egg Delivery** | 效果（产卵） |
| `Reveal Grayling` | **显现灰蛾灵 Reveal Grayling** | 宏。⚠ 随 Grayling 待拍板 |
| `Reveal Shadebranch` | **显现影枝 Reveal Shadebranch** | 宏 |
| `Banished from the Sanguinaries` | **被赤血会放逐** | outcome label。Sanguinaries→赤血会、Banished→放逐 既定 |
| `Friend to the Sanguinaries` | **赤血会之友** | outcome label |
| `Captive Spared` | **饶过俘虏** | outcome label（被末日预言者关押的赤血会守林者 Cael） |
| `Entered the Arbore Sanctorus` | **已进入圣树庇护所** | outcome label。⚠ Arbore Sanctorus → 圣树庇护所 是库内**既有译名**（0.6.0 就有），直接沿用、不许重译 |
| `Heirloom Returned` | **传家宝已归还** | outcome label |
| `Heirloom Withheld` | **扣下传家宝** | outcome label。对齐 Alchemical Reagent Withheld 扣下炼金试剂 |
| `Kadra Zann Revealed` | **卡德拉赞恩已揭示** | outcome label |
| `Landmark Clue` | **地标线索** | outcome label |
| `One with the Pack` | **与寂静狼群合一** | outcome label。⚠ 此处 the Pack ＝ The Silent Pack 寂静狼群（库内既定，Kyrban 的赞助者、一群 Casia 卡西娅），不是 Pack Tactics 的「群体」 |
| `Signs of Corruption` | **堕化的迹象** | outcome label |
| `Riverside Ruby Works Exploded` | **河畔红宝工坊已爆炸** | outcome label |
| `Interior: Ground Floor` | **室内：地面层** | scene region。Interior:→室内： 既定 |
| `Interior: Lower Floor` | **室内：下层** | scene region |
| `Interior: Upper Floor` | **室内：上层** | scene region |
| `Interior: Vespiaries` | **室内：蜂巢群** | scene region |
| `Surface: Ground` | **地表：地面** | scene region。Surface:→地表： 既定 |
| `Surface: Exterior Ground` | **地表：外部地面** | scene region。Exterior→外部 既定 |
| `Surface: Ground Walkways` | **地表：地面步道** | scene region。Walkways→步道 既定 |
| `Surface: Lower Ground` | **地表：下层地面** | scene region |
| `Surface: Lower Water` | **地表：下层水面** | scene region。⚠ Surface 家族里 Water→水面（地表：水面），不要写水域 |
| `Surface: Upper Water` | **地表：上层水面** | scene region |
| `Surface: Solid Rock` | **地表：坚岩** | scene region |
| `Surface: Upper Grass` | **地表：上层草地** | scene region。Surface: Grass→地表：草地 既定 |
| `Surface: Upper Walkways` | **地表：上层步道** | scene region |
| `Transition: Lower Stairs` | **过渡：下层楼梯** | scene region。Transition:→过渡：、Lower Stairs→下层楼梯 既定 |
| `Transition: Upper Stairs` | **过渡：上层楼梯** | scene region |
| `Transition: Rickety Ladder` | **过渡：摇晃的梯子** | scene region |
| `Transition: Vespiary Hole` | **过渡：蜂巢洞口** | scene region |
| `Transition: Vespiary Tunnel` | **过渡：蜂巢隧道** | scene region |
| `Spawn: Northwest` | **出生点：西北** | scene region。Spawn: Exterior→出生点：外部 既定 |
| `Spawn: Southeast` | **出生点：东南** | scene region |

### 1122 叶之外的名称位（folders / categories / levels / notes / sounds / encounterTokens，共 192 叶）

| EN | CN（可直接照抄） | 形态 | 依据 / 出处 |
|---|---|---|---|
| `Doomsayers` | **末日预言者 Doomsayers** | 双语 | folder（组织）。⚠ 待拍板 |
| `Primordials` | **原初族 Primordials** | 双语 | folder（被普里莫迪斯浸染的新生物族群）。⚠ 不可用「原初生物」—— 词表里 Primordial Creatures 已占该中文。⚠ Primordial 词根待拍板 |
| `Sporix` | **孢体 Sporix** | 双语 | folder。沿用 Sporix Host 孢体宿主 |
| `Acolytes of Thayloc` | **赛洛克侍僧 Acolytes of Thayloc** | 双语 | folder。Thayloc→赛洛克、Acolyte→侍僧 既定 |
| `Talei` | **塔莱 Talei** | 双语 | folder。库内既定 |
| `Ruby Grove` | **红宝林地 Ruby Grove** | 双语 | folder。与 journal 同名 |
| `Tajar` | **塔贾尔** | 纯中文 | category（Arctus Plateau Gazetteer 新分区）。⚠ 新专名，无既有出处 |
| `Bloodwoods` | **血林** | 纯中文 | category。库内既定 |
| `Redrak Fields` | **雷德拉克原野** | 纯中文 | category。库内既定 |
| `Overview` | **概览** | 纯中文 | category。库内既定（120 处） |
| `Appendix` | **附录** | 纯中文 | category。库内既定 |
| `Events` | **事件** | 纯中文 | category。库内既定 |
| `Ground Level` | **地面层** | 纯中文 | category。库内既定 |
| `Lower Level` | **下层** | 纯中文 | category。库内既定 |
| `Upper Level` | **上层** | 纯中文 | category。库内既定 |
| `Clearing` | **空地** | 纯中文 | scene level。沿用 Trapped Clearing 陷阱空地 |
| `Clearing - Camp` | **空地 - 营地** | 纯中文 | scene level。Campsite→营地 既定；连字符样式对齐 天刷镇 - 阴郁 |
| `Mangled Grove` | **残毁林地** | 纯中文 | scene level |
| `Primordian Monument` | **原初纪念碑** | 纯中文 | scene level。⚠ Primordian 词根待拍板 |
| `Template Base, Grassy` | **模板底图，草地** | 纯中文 | scene level（Vista 模板层） |
| `Template Base, Rocky` | **模板底图，岩石** | 纯中文 | scene level |
| `Gloom` | **阴郁** | 纯中文 | scene level。沿用 Skybrush - Gloomy 天刷镇 - 阴郁 |
| `Cheery` | **欢欣** | 纯中文 | scene level。沿用 Skybrush - Cheery 天刷镇 - 欢欣 |
| `Ground` | **地面** | 纯中文 | scene level。库内既定 |
| `Upper` | **上层** | 纯中文 | scene level。库内既定 |
| `Lower` | **下层** | 纯中文 | scene level。库内既定 |
| `Amerasp Grove` | **阿梅拉斯普林地** | 纯中文 | scene level。库内既定 |
| `Hassratch Cave` | **哈斯拉奇洞窟** | 纯中文 | scene level。Hassratch→哈斯拉奇 既定 |
| `Rock Bottom Slum` | **石底镇贫民窟** | 纯中文 | scene level。Rock Bottom→石底镇 既定 |
| `Rock Bottom Causeway` | **石底镇堤道** | 纯中文 | scene level |
| `Town Center` | **镇中心** | 纯中文 | scene level |
| `Barge Explosion` | **驳船爆炸** | 纯中文 | scene level |
| `Corrupted Nest（level）` | **堕化巢穴** | 纯中文 | scene level。与 region／page 同名同译 |
| `Vista: Bloodwoods` | **远景：血林 Vista: Bloodwoods** | 双语 | scene 名。Vista:→远景： 既定（29 处），scene 名带双语尾 |
| `Vista: Talei` | **远景：塔莱 Vista: Talei** | 双语 | scene 名 |
| `Amerasp Caves（note）` | **阿梅拉斯普洞窟 Amerasp Caves** | 双语 | scene note。notes 带双语尾（湖金罗 notes 实测） |
| `Courtyard` | **庭院 Courtyard** | 双语 | scene note。库内既定 |
| `Sanctuary` | **庇护所 Sanctuary** | 双语 | scene note。库内既定 |
| `Study` | **书房 Study** | 双语 | scene note。库内既定 |
| `Workshop` | **工坊 Workshop** | 双语 | scene note。库内既定 |
| `Kyrban's Chancery（note）` | **凯尔班的文书院 Kyrban's Chancery** | 双语 | scene note。⚠ note 键是直撇 Kyrban's，journal page 键是弯撇 Kyrban’s —— 两个键都要覆盖 |
| `Campfire` | **营火** | 纯中文 | scene sound |
| `Sphere` | **球体** | 纯中文 | scene sound。⚠ 上游语义不明，第二阶段进场看资源路径再定 |
| `Torch 1` | **火把 1** | 纯中文 | scene sound（Torch 1〜9 同款）。Torch→火把 既定（29 处）；编号样式对齐 软泥池 1 |
| `Waterfall East Ground` | **瀑布 东侧地面层** | 纯中文 | scene sound（East/West × Ground/Lower/Upper 六条同款）。Waterfall→瀑布 既定 |
| `Buzzing Insects East` | **昆虫嗡鸣 东** | 纯中文 | scene sound（East/North/North East/North West/West 五条同款） |
| `Fleshy Movement East` | **肉质蠕动 东** | 纯中文 | scene sound（五条同款） |
| `Arcos` | **阿科斯** | 纯中文 | encounterToken。沿用 Arcos Sarinland 阿科斯·萨林兰德 |
| `Torra` | **托拉** | 纯中文 | encounterToken。沿用「保护了托拉的构装体」 |
| `Torra's Chessman` | **托拉的棋子** | 纯中文 | encounterToken |
| `Ifton Shepp` | **伊夫顿·谢普** | 纯中文 | encounterToken。库内既定 |
| `Maevren（token）` | **梅芙伦** | 纯中文 | encounterToken |
| `Proctus` | **普罗克图斯** | 纯中文 | encounterToken（Proctus Caylas 的简称） |
| `Verno Kreed（token）` | **韦尔诺·克里德** | 纯中文 | encounterToken |
| `Barka` | **巴尔卡** | 纯中文 | encounterToken·新配角 |
| `Beet` | **比特** | 纯中文 | encounterToken·新配角 |
| `Brond` | **布隆德** | 纯中文 | encounterToken·新配角 |
| `Doran Styme` | **多兰·斯泰姆** | 纯中文 | encounterToken·新配角 |
| `Hodge` | **霍奇** | 纯中文 | encounterToken·新配角（Hodge's Heirloom 的主人） |
| `Nevel Rux` | **内维尔·鲁克斯** | 纯中文 | encounterToken·新配角 |
| `Thura` | **图拉** | 纯中文 | encounterToken·新配角 |
| `Valian` | **瓦利安** | 纯中文 | encounterToken·新配角 |
| `Cael` | **卡埃尔** | 纯中文 | encounterToken·新配角（Arbore Sanctorus 的赤血会守林者） |

### lang/cn.json 与 ember-hardcoded-cn.mjs（compendium 之外的两条通道）

| EN | CN（可直接照抄） | 依据 / 出处 |
|---|---|---|
| `Suppress Environment Filter` | **抑制环境滤镜** | lang 新键 TYPES.RegionBehavior.ember.suppressEnvironmentFilter。对齐同族 Suppress Weather→抑制天气（库内 115 处） |
| `Effects` | **效果** | lang 新键 EMBER.VISTA.TABS.vignettes（Vista 配置工具新增的 Effects 页签）。Effects→效果 库内既定。⚠ 上游键名叫 vignettes、值叫 Effects，以**值**为准 |
| `Arbore Sanctorus Day` | **圣树庇护所 · 白天** | ARRANGEMENTS 新编排名。Arbore Sanctorus→圣树庇护所 库内既定；「X · 白天／夜晚」是表内既定式样 |
| `Arbore Sanctorus Night` | **圣树庇护所 · 夜晚** | ARRANGEMENTS 新编排名 |
| `Talei Day` | **塔莱 · 白天** | ARRANGEMENTS 新编排名。Talei→塔莱 既定 |
| `Talei Night` | **塔莱 · 夜晚** | ARRANGEMENTS 新编排名 |
| `Rortwark Day` | **罗特瓦克 · 白天** | ARRANGEMENTS 新编排名。Rortwark→罗特瓦克 既定 |
| `Rortwark Night` | **罗特瓦克 · 夜晚** | ARRANGEMENTS 新编排名 |
| `Ushna Dredging Day` | **乌什纳疏浚 · 白天** | ARRANGEMENTS 新编排名。Ushna Dredging→乌什纳疏浚 既定 |
| `Ushna Dredging Night` | **乌什纳疏浚 · 夜晚** | ARRANGEMENTS 新编排名 |
| `Earthen Henge Day` | **土石环阵 · 白天** | ARRANGEMENTS 新编排名。Earthen Henge→土石环阵 既定 |
| `Earthen Henge Night` | **土石环阵 · 夜晚** | ARRANGEMENTS 新编排名 |
| `Stonework Hollow Day` | **石工谷地 · 白天** | ARRANGEMENTS 新编排名。Stonework Hollow→石工谷地 既定 |
| `Stonework Hollow Night` | **石工谷地 · 夜晚** | ARRANGEMENTS 新编排名 |
| `Primordis Fight Section 1` | **普里莫迪斯战斗 · 第一段** | ARRANGEMENTS 新编排名。表内既定：X Fight→X战斗、Section N→第 N 段（对齐 Celestial Combat Section 1 天界生物战斗 · 第一段） |
| `Primordis Fight Section 2` | **普里莫迪斯战斗 · 第二段** | ARRANGEMENTS 新编排名 |

## 附表 B：已有译名 —— 直接沿用，不许重译（216 条）

这 216 条英文在 0.6.0 的库里已经有译名。第二阶段十个译者**一律照抄**，任何一条被重译都会在主闸的 `same_en_split` 上炸。

| EN | 库内既定 CN |
|---|---|
| `+3 AC` | AC +3 加值 |
| `Acid Protection` | 强酸防护 Acid Protection |
| `Adjust Darkness Level` | 调整黑暗等级 |
| `Agrimage` | 农艺法师 Agrimage |
| `Agrimagical Relic` | 农艺魔法遗物 Agrimagical Relic |
| `Alchemist's Supplies` | 炼金术师工具 Alchemist's Supplies |
| `Amerasp Grove` | 阿梅拉斯普林地 Amerasp Grove |
| `Amulet of Nimbleness` | 轻捷护符 Amulet of Nimbleness |
| `Ancestral Grove` | 先祖树林 Ancestral Grove |
| `Animal Friendship` | 动物交友 Animal Friendship |
| `Arcturian Training` | 阿克图里安训练 Arcturian Training |
| `Area Overview` | 区域概览 Area Overview |
| `Bane` | 祸骰 Bane |
| `Banished` | 放逐 Banished |
| `Banishment` | 放逐术 Banishment |
| `Bark-like Skin` | 树肤 Bark-like Skin |
| `Barkskin` | 树肤 Barkskin |
| `Bestial Communication` | 野兽交流 Bestial Communication |
| `Bless` | 祝福 Bless |
| `Blessed` | 受祝福 Blessed |
| `Blight` | 枯萎术 Blight |
| `Bloom` | 绽放 Bloom |
| `Bludgeoning Protection` | 钝击防护 Bludgeoning Protection |
| `Book` | 书籍 Book |
| `Canny` | 机敏 Canny |
| `Change Level` | 切换层级 |
| `Charm Person` | 魅惑人类 Charm Person |
| `Charmed` | 魅惑 Charmed |
| `Cloak` | 斗篷 Cloak |
| `Clothes, Fine` | 华服 Clothes, Fine |
| `Clothes, Traveler's` | 旅行者服装 Clothes, Traveler's |
| `Club` | 棍棒 Club |
| `Cold Protection` | 寒冷防护 Cold Protection |
| `Common Clothing` | 普通服装 Common Clothing |
| `Concealed` | 隐蔽 Concealed |
| `Conjure Animals` | 咒唤兽群 Conjure Animals |
| `Conjure Spellcraft` | 召唤施法 Conjure Spellcraft |
| `Corrode Weapon` | 腐蚀武器 Corrode Weapon |
| `Counterspell` | 反制法术 Counterspell |
| `Courtyard` | 庭院 Courtyard |
| `Dagger` | 匕首 Dagger |
| `Dancing Lights` | 舞光术 Dancing Lights |
| `Darkness` | 黑暗 |
| `Define Surface` | 定义地表 |
| `Detect Magic` | 侦测魔法 Detect Magic |
| `Detect Thoughts` | 侦测思想 Detect Thoughts |
| `Devious` | 狡诈 Devious |
| `Difficult Terrain` | 困难地形 |
| `Dimension Door` | 任意门 Dimension Door |
| `Disguise Self` | 易容术 Disguise Self |
| `Disguised` | 易容 Disguised |
| `Dispel Magic` | 解除魔法 Dispel Magic |
| `Druid` | 德鲁伊 Druid |
| `Druidcraft` | 德鲁伊伎俩 Druidcraft |
| `Earth Proficiency` | 大地熟练度 Earth Proficiency |
| `Elder God` | 上古之神 Elder God |
| `Eldritch Blast` | 奥术冲击 Eldritch Blast |
| `Electrified Pseudopod` | 带电拟足 Electrified Pseudopod |
| `Ember Blaze` | 余烬烈焰 Ember Blaze |
| `Energize` | 充能 Energize |
| `Enkindle` | 引燃 Enkindle |
| `Ensnare` | 缠捕 Ensnare |
| `False Life` | 虚假生命 False Life |
| `Fear Aura` | 恐惧灵气 Fear Aura |
| `Fears Manifested` | 恐惧成真 Fears Manifested |
| `Feline Domino` | 猫面面具 Feline Domino |
| `Feline Form` | 猫形 Feline Form |
| `Fire Protection` | 火焰防护 Fire Protection |
| `Fishing Pole` | 钓竿 Fishing Pole |
| `Flowchart` | 流程图 Flowchart |
| `Flyby` | 掠空飞袭 Flyby |
| `Fog Cloud` | 迷雾云团 Fog Cloud |
| `Footstep Surface` | 脚步地表 |
| `Frightened` | 恐慌 Frightened |
| `Gazetteer Reference` | 地名志参考 Gazetteer Reference |
| `Gesture: Arrow` | 手势：箭矢 Gesture: Arrow |
| `Gesture: Aspect` | 手势：化相 Gesture: Aspect |
| `Gesture: Aura` | 手势：灵气 Gesture: Aura |
| `Gesture: Blast` | 手势：爆破 Gesture: Blast |
| `Gesture: Cone` | 手势：锥形 Gesture: Cone |
| `Gesture: Conjure` | 手势：召唤 Gesture: Conjure |
| `Gesture: Create` | 手势：创造 Gesture: Create |
| `Gesture: Fan` | 手势：扇形 Gesture: Fan |
| `Gesture: Influence` | 手势：影响 Gesture: Influence |
| `Gesture: Pulse` | 手势：脉冲 Gesture: Pulse |
| `Gesture: Ray` | 手势：射线 Gesture: Ray |
| `Gesture: Sense` | 手势：感知 Gesture: Sense |
| `Gesture: Step` | 手势：踏步 Gesture: Step |
| `Gesture: Strike` | 手势：打击 Gesture: Strike |
| `Gesture: Surge` | 手势：涌动 Gesture: Surge |
| `Gesture: Ward` | 手势：防护 Gesture: Ward |
| `Growing Thorns` | 生长荆棘 Growing Thorns |
| `Haste` | 加速术 Haste |
| `Hasted` | 加速 Hasted |
| `Healer's Toolkit` | 治疗者工具包 Healer's Toolkit |
| `Healing Word` | 治疗之语 Healing Word |
| `Herbalism Kit` | 草药工具 Herbalism Kit |
| `Human` | 人类 Human |
| `Human Lineage` | 人类血统 Human Lineage |
| `Hypnotic Pattern` | 催眠图纹 Hypnotic Pattern |
| `Hypnotized` | 被催眠 Hypnotized |
| `Illusion Spellcraft` | 幻象施法 Illusion Spellcraft |
| `Imperceptible Barrier` | 无法察觉的屏障 Imperceptible Barrier |
| `Incense` | 熏香 Incense |
| `Inflection: Compose` | 屈折：编构 Inflection: Compose |
| `Inflection: Determine` | 屈折：限定 Inflection: Determine |
| `Inflection: Elude` | 屈折：遁避 Inflection: Elude |
| `Inflection: Extend` | 屈折：延展 Inflection: Extend |
| `Inflection: Negate` | 屈折：否定 Inflection: Negate |
| `Inflection: Pull` | 屈折：拉拽 Inflection: Pull |
| `Inflection: Push` | 屈折：推挤 Inflection: Push |
| `Inflection: Quicken` | 屈折：迅捷 Inflection: Quicken |
| `Inflection: React` | 屈折：反应 Inflection: React |
| `Inflection: Reshape` | 屈折：重塑 Inflection: Reshape |
| `Insect Plague` | 虫灾 Insect Plague |
| `Invisibility` | 隐形 Invisibility |
| `Invisible` | 隐形 Invisible |
| `Kadra Zann` | 卡德拉赞恩 Kadra Zann |
| `Leather Armor` | 皮甲 Leather Armor |
| `Lethal` | 致命 Lethal |
| `Lethargy` | 倦怠 Lethargy |
| `Life Proficiency` | 生命熟练度 Life Proficiency |
| `Life Spellcraft` | 生命施法 Life Spellcraft |
| `Lightning Bolt` | 闪电箭 Lightning Bolt |
| `Lightning Protection` | 闪电防护 Lightning Protection |
| `Longbow` | 长弓 Longbow |
| `Mage Armor` | 法师护甲 Mage Armor |
| `Magic Resistance` | 魔法抗性 Magic Resistance |
| `Magnetic Disarm` | 磁力缴械 Magnetic Disarm |
| `Mender` | 修补匠 Mender |
| `Mind Spike` | 心灵尖刺 Mind Spike |
| `Minor Illusion` | 次级幻影 Minor Illusion |
| `Misty Step` | 迷踪步 Misty Step |
| `Motivate` | 激励 Motivate |
| `Mould` | 塑形 Mould |
| `Multiattack` | 多重攻击 Multiattack |
| `Necrotic Protection` | 死灵防护 Necrotic Protection |
| `Nimaelle` | 尼梅尔 Nimaelle |
| `Nimbleness` | 轻捷 Nimbleness |
| `Nourish` | 催生 Nourish |
| `Overview` | 概览 |
| `Padded Armor` | 衬垫护甲 Padded Armor |
| `Paralyzed` | 麻痹 Paralyzed |
| `Pass without Trace` | 无痕穿行 Pass without Trace |
| `Phantasmal Killer` | 幻影杀手 Phantasmal Killer |
| `Piercing Protection` | 穿刺防护 Piercing Protection |
| `Plane Shift` | 位面转移 Plane Shift |
| `Poison Conversion` | 毒素转化 Poison Conversion |
| `Poison Protection` | 毒素防护 Poison Protection |
| `Poison Resistance` | 毒素抗性 Poison Resistance |
| `Poison Spray` | 毒液喷射 Poison Spray |
| `Poisoned` | 中毒 Poisoned |
| `Polymorph` | 变形术 Polymorph |
| `Potency` | 效力 Potency |
| `Potion of Climbing` | 攀爬药水 Potion of Climbing |
| `Pounce` | 猛扑 Pounce |
| `Predatory` | 掠食 Predatory |
| `Primordis 2: Canny` | 普里莫迪斯 2：机敏 Primordis 2: Canny |
| `Primordis 4: Devious` | 普里莫迪斯 4：狡诈 Primordis 4: Devious |
| `Primordis 5: Predatory` | 普里莫迪斯 5：掠食 Primordis 5: Predatory |
| `Primordis Sorcery` | 普里莫迪斯术法 Primordis Sorcery |
| `Propel` | 推进 Propel |
| `Quarterstaff` | 长棍 Quarterstaff |
| `Radiant Protection` | 光耀防护 Radiant Protection |
| `Reaching Vine` | 延伸藤蔓 Reaching Vine |
| `Refocus` | 重新聚焦 Refocus |
| `Reinforcement` | 强化 Reinforcement |
| `Reliable` | 可靠 Reliable |
| `Resistance` | 抗性 Resistance |
| `Ring of Reinforcement` | 强化戒指 Ring of Reinforcement |
| `Riposte` | 还击 Riposte |
| `Ruby` | 红宝 Ruby |
| `Ruby Influence` | 红宝影响 Ruby Influence |
| `Rune: Control` | 符文：控制 Rune: Control |
| `Rune: Earth` | 符文：大地 Rune: Earth |
| `Rune: Flame` | 符文：火焰 Rune: Flame |
| `Rune: Kinesis` | 符文：念力 Rune: Kinesis |
| `Rune: Life` | 符文：生命 Rune: Life |
| `Rune: Storm` | 符文：风暴 Rune: Storm |
| `Sanctuary` | 庇护所 |
| `Scratch` | 抓挠 Scratch |
| `Seeming` | 表象 Seeming |
| `Shard God` | 碎片之神 Shard God |
| `Shield` | 护盾术 Shield |
| `Shortsword` | 短剑 Shortsword |
| `Siege Monster` | 攻城怪物 Siege Monster |
| `Slashing Protection` | 挥砍防护 Slashing Protection |
| `Sleet Storm` | 雨夹雪风暴 Sleet Storm |
| `Sorcerous Burst` | 术法迸发 Sorcerous Burst |
| `Spawn: Exterior` | 出生点：外部 |
| `Speak with Animals` | 与动物交谈 Speak with Animals |
| `Spell Changes` | 法术变更 Spell Changes |
| `Spellcasting` | 施法 Spellcasting |
| `Spider Climb` | 蛛行术 Spider Climb |
| `Spike Growth` | 荆棘丛生 Spike Growth |
| `Spiked` | 尖刺标记 Spiked |
| `Studded Leather Armor` | 镶钉皮甲 Studded Leather Armor |
| `Study` | 书房 Study |
| `Suppress Weather` | 抑制天气 |
| `Swarm` | 群集 Swarm |
| `Talisman Weapon Proficiency` | 护符武器熟练度 Talisman Weapon Proficiency |
| `Talisman Weapon Training` | 护符武器训练 Talisman Weapon Training |
| `Teleport Token` | 传送指示物 |
| `Tenacity` | 顽强 Tenacity |
| `Thornbark` | 刺棘树皮 Thornbark |
| `Thornling` | 荆芽灵 Thornling |
| `Thornling Lineage` | 荆芽灵血统 Thornling Lineage |
| `Thunder Protection` | 雷鸣防护 Thunder Protection |
| `Trickadee` | 戏诈雀 |
| `Trickadee Egg` | 戏诈雀卵 Trickadee Egg |
| `Wall of Force` | 力场墙 Wall of Force |
| `Warden` | 守林者 Warden |
| `Wildspeak` | 荒野语 Wildspeak |
| `Wildspeaker` | 荒野言者 Wildspeaker |
| `Woodcarver's Tools` | 木雕工具 Woodcarver's Tools |
| `Workshop` | 工坊 Workshop |


---

## 附表 C：0.6.1 的新词但**不在名称位**、第二阶段翻正文时会撞上的

这些只出现在正文里（`text`/`description`/`exposition`…），本单元不裁，但先点名，免得十个译者各译各的：

| EN | 出处 | 备注 |
|---|---|---|
| `Fin'ian Sphere` | Kadra Zann 正文 | 申特人（`Shent`→申特）造的施法媒介神器 |
| `Ambral` | Proving Your Metal 正文 | 奥肯加德（`Oakengarde`→奥肯加德）特产魔法金属；合金名 `Ambral Bronze` 已在主表 |
| `Casia` | Where Shadows Lie 正文 | 库里已有「卡西娅 Casia」，沿用 |
| `Caryx Savannah` | 只在补丁说明页 | 尚未产生名称叶，待它进包再裁 |
| `Hair roots` / `Hair Highlight` | 只在补丁说明页 | 见 §8 |

---

## 8. 代币颜色命名改动（`Hair1`/`Hair2` → `Hair roots`/`Hair Highlight`）的影响面：**零**

实测三处，全部为空：

1. `grep -ri "Hair1|Hair2|Hair roots|Hair Highlight|hairRoot"` 扫整个项目根 —— **0 命中**。
2. `1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs`（24.9 万字符的硬编码译文表）里 `Hair` —— **0 命中**。
3. 上游 `modules/ember/scripts/ember.mjs` 里这批串确实存在（`"Hair Base"` / `"Hair Roots"` / `"Hair Highlights"` /
   `"Hair Glow"` / `"Hair Sparkle"`，另有一条遗留的 `"Hair 1"`），但它们是**代币制作器的颜色槽名**，
   我们的硬编码表从来没有覆盖这一层。

⇒ **本轮不需要跟改任何东西。** 若将来要把代币制作器也汉化，这六个串是新的入表候选
（发根／发色底／挑染／发光／闪粉），届时再裁。

---

## 9. 本单元的产物与脚本

| 文件 | 是什么 |
|---|---|
| `GLOSSARY-061.md` | **本文件**。第二阶段的唯一术语依据 |
| `collisions.json` | 撞车清单（含逐条处置建议） |
| `candidates.json` / `candidates_extra.json` / `candidates_lang_hardcoded.json` | 裁决表的**机器可读源**（撞车判据的输入） |
| `probe_names.py` | 前置自证 + 抽 1122 名称叶 |
| `build_index.py` | 建库的双向索引（EN→CN / CN核心→EN） |
| `check_collisions.py` | 双向撞车判据（自带正反自证） |
| `gen_glossary.py` | 由候选文件渲染本表的表格段 |
| `names_raw.json` / `names_uniq.json` / `known_216.txt` / `new_ctx.txt` | 中间产物 |
| `lib_en2cn.json` / `lib_cn2en.json` / `lib_prov.json` | 库索引（8375 个 EN 键 / 15177 个 CN 核心键，带出处） |
