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
