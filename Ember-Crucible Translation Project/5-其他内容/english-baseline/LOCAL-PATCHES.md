# 对上游英文基准打的本地补丁

> **每次用 `extract_en.mjs` 重抽英文之后，必须回来照这张表重打一遍**，
> 否则会静默回退，相关条目的译文会重新被闸门拒收。

这里只收录**上游英文自身写坏、且坏到会卡住翻译流程**的地方。
上游把它修好之后，对应行可以删掉（重抽后发现不再匹配即说明已修）。

> **正表现有 5 条**（2026-08-15 第二十轮从上游 LevelDB 重抽逐叶对照坐实：
> 孪生两包各 5 叶值不同、无第六叶，命令与实测表见第 5 条末尾）。
> 校验这张表是否还全的唯一办法就是**重抽再逐叶 diff** ——
> **别靠「闸是绿的」判断补丁齐不齐**：第 5 条就是补完闸变绿、然后被漏登记了整整四轮。

---

## 读这张表之前：两个孪生包**不是**同一份英文

下面几条的「影响包」栏写着 `ember.adventure.json`、`ember.crucible-adventure.json`
两个包，并注明**该路径上**两包英文逐字节相同 —— 那是逐条核过的、真的相同，
所以同一处补丁必须两包各打一遍。

⚠ 但**别把它读成「两包整体相同」**。2026-08-16 终段实测（`compendium/en` 侧）：

| 口径 | 实测 |
|---|---|
| `ember.adventure.json` 叶数 | 14 845 |
| `ember.crucible-adventure.json` 叶数 | 20 267 |
| 两包**共有**的路径 | **11 867** |
| 共有路径里**值不同**的 | **243** |
| 仅 `adventure` 有的路径 | 2 978 |
| 仅 `crucible-adventure` 有的路径 | 8 400 |

不同的那 243 条不是噪声，是**按系统分家的内容**：`biography.public` 大量整段重写
（dnd5e 侧是 readaloud 描述、crucible 侧是设定说明）、`tokenName` 前缀不同
（`Pallid Drake` vs `Afflicted Pallid Drake`）、战利品行换物件
（`Piton` 岩钉 vs `Prybar` 撬棒，见 B.3）等等。

**结论**：孪生同步（`sync_twin_packs.py`）与本表的补丁都必须**按路径**核对，
不能按包整体假定相同；反过来，本表各条注明的「该路径两包相同」是**逐条实测过**的，
可以照打。

---

## 1. `Toothbreaker Hideout / Prison` 的 `@Condition[exhaustion` 缺右方括号

- **日期**：2026-08-09
- **影响包**：`ember.crucible-adventure.json`、`ember.adventure.json`（**该路径上**两包英文逐字节相同 —— 逐条实测过；别据此以为两包整体相同，见抬头的表）
- **路径**：`Ember Early Access.journals.Toothbreaker Hideout.pages.Prison.text`
- **上游原文**：

  ```html
  <sub data-system="crucible">@Condition[exhaustion</sub></sup>
  ```

- **补成**：

  ```html
  <sub data-system="crucible">@Condition[exhaustion]</sub></sup>
  ```

**为什么非修不可**：`markup_signature` 的 `MARKUP = @[A-Za-z]+\[[^\]]*\]` 会从
`@Condition[` 一直吞到**下一个** `]`。英文与中文后续的文字不同，被吞进标记的范围就不同，
于是签名永远对不上 —— 即使中文一字不差地照抄了这个坏标记也一样被拒。
这是全库最后一条过不了闸的条目（34859 分之 1）。

补上之后：英文侧标记变成干净的 `@Condition[exhaustion]`，中文照抄同一个标记即可匹配。

**注意：补 `]` 只解决闸门，不产生可点链接。** crucible 里力竭状态的 id 是 `exhausted`
而不是 `exhaustion`（`systems/crucible/crucible-compiled.mjs:7001-7006`，整份文件里
`exhaustion` 出现 **0** 次）。`enrichCondition`（:46672）转调 `enrichRule`（:46736），
后者 `getProperty(SYSTEM.RULES, "condition.exhaustion")` 查空后直接 `return new Text(match)`，
所以玩家看到的仍是裸字面量 `@Condition[exhaustion]`。
若要连渲染一起修好，需另立一条把 en/cn 两侧都改成 `@Condition[exhausted]` 的补丁 ——
那是改上游语义，属另一件事，尚未做。

**同时改的**：`compendium/cn/ember.crucible-adventure.json` 同一路径的中文里
那个照抄来的坏标记也补了 `]`。

---

## 2. `Unfinished Business / Answers From On High` 把 `{Persuasion}` 裸写在正文里

- **日期**：2026-08-12
- **影响包**：`ember.crucible-adventure.json`、`ember.adventure.json`
- **路径**：`Ember Early Access.journals.Unfinished Business.pages.Answers From On High.text`
- **上游原文**：`Agraband automatically succeeds on the Charisma {Persuasion} check.`
- **补成**：`… on the Charisma (Persuasion) check.`

**为什么要修**：这里的 `{Persuasion}` 前面**没有** `@UUID[…]`，也不在任何 enricher 里 ——
作者多半是想写个链接、只留下了标签。Foundry 会把花括号原样渲染给玩家看。

而 `scan_markup_drift` 的 PLACEHOLDER 判据（`\{[A-Za-z_][A-Za-z0-9_.\-]*\}`）分不清
「真占位符」和「作者手滑留下的花括号」，于是把这条永远报成缺失。中文写的是
「魅力（游说）」——比上游还正确，不该为了迁就一条正则改成「魅力{游说}」。
把英文基准补成圆括号，两边就都对了。

---

## 3. `Yakoshta Mine / Elevator` 的 `[[/skillCheck athletics 15 check.` 缺右方括号

- **日期**：2026-08-13
- **影响包**：`ember.adventure.json`、`ember.crucible-adventure.json`（**该路径上**两包英文逐字节相同 —— 逐条实测过；别据此以为两包整体相同，见抬头的表）
- **路径**：`Ember Early Access.journals.Yakoshta Mine.pages.Elevator.text`
- **上游原文**：

  ```html
  <sub data-system="crucible">[[/skillCheck athletics 15 check.</sub>
  ```

- **补成**：

  ```html
  <sub data-system="crucible">[[/skillCheck athletics 15]] check.</sub>
  ```

**为什么非修不可**：与第 1 条同型，只是换成了内联命令那条正则。
`apply_translations.py` 的 `INLINE_CMD = \[\[[^\]]*\]\]` 会从这个没闭合的 `[[` 一路吞到
**下一个** `]]` —— 也就是下一段的 `[[/skill arcana 12]]`。夹在中间的那整句
`If the party has Jasper's Keys, but does not know which one is correct, they can identify
the correct key with a successful …` 就被吞进了同一个标记 token 里。
后果是**这一叶结构性地不可翻译**：只要把那句话译成中文，被吞进 token 的内容就与英文不同，
`markup_signature` 必然对不上，**任何译法都被判 markup mismatch**。
这也正是这一句在库里至今仍是英文的原因 —— 不在译者，在闸门。

补法取的是与同一 `<sup>` 里 dnd5e 兄弟节点完全平行的形态
（`<sub data-system="dnd5e">[[/skill athletics 15]] check.</sub>`），
即作者本来要写的东西；补上之后玩家还能得到一个可点的运动检定，比上游原样更好。

**同时改的**：`compendium/cn` 两包同一路径的中文里那个照抄来的坏标记也补成了 `]]`，
并把因此一直无法回写的那一句补译完整（见第九轮 leg2 批次
`4-临时脚本/2026-08-13-round9/batches/leg2.1.ember.*adventure.json`）。

---

## 4. `Organizations / Silver Beam Consortium` 的 `C0nsortium`（零 vs 字母 o）

- **日期**：2026-08-13
- **影响包**：`ember.adventure.json`、`ember.crucible-adventure.json`（**该路径上**两包英文逐字节相同 —— 逐条实测过；别据此以为两包整体相同，见抬头的表）
- **路径**：`Ember Early Access.journals.Organizations.pages.Silver Beam Consortium.text`
- **上游原文**：`… the Silver Beam C0nsortium are a formidable enterprise …`
- **补成**：`… the Silver Beam Consortium are a formidable enterprise …`

**为什么要修**：上游把 `Consortium` 误打成 `C0nsortium`（数字零），同一叶里正确拼写的
`Consortium` 就在几行之外，是明显的笔误。

原先中文照抄了这个坏词（写作「银光束 C0nsortium」），第十轮把它正确译成「银光束财团」之后，
`scan_content_coverage` 立刻报出 2 条 —— 因为该判据把英文里的 `0` 当成一个**数字**，
而中文「财团」里当然没有 `0`，于是判成「中文丢了英文里的数字」。

这是第 2 条（`{Persuasion}`）的同型：**判据分不清「真数字」与「拼写错误里的字符」，
而正确的中文反倒被判成缺陷**。按既定做法改英文基准而不是迁就判据 —— 否则唯一能让闸门
归零的写法是把「C0nsortium」这个坏词留在中文里给玩家看。

补上之后 `scan_content_coverage` 两个仓库重新归零。

---

## 5. `Paralyzing Bolt / Paralyzed` 的 `reference[Paralyzed]` 缺开头的 `&`

- **日期**：**2026-08-09**，与第 1 条**同一个提交**
  `1-Ember汉化插件` 的 `dd54bdd`「fix(markup): 补上游两处缺字符 + 修最后 10 条丢失的 &Reference」。
  **本条 2026-08-15 第二十轮补登记**（漏登记了六天 / 四轮），见下面「为什么漏了」。
- **影响包**：`ember.adventure.json`、`ember.crucible-adventure.json`
  （**该路径上**两包英文逐字节相同 —— 本轮逐条实测过；别据此以为两包整体相同，见抬头的表）
- **路径**：`Ember Early Access.items.Paralyzing Bolt.effects.Paralyzed.description`
- **上游原文**（重抽出来就是这个，**没有 `&`**）：

  ```html
  <p>The target is reference[Paralyzed] for 1 minute. </p>
  ```

- **补成**（JSON 里存的是字面量 `&amp;`，渲染成 `&`）：

  ```html
  <p>The target is &amp;reference[Paralyzed] for 1 minute. </p>
  ```

**这是手打补丁，不是抽取口径变化。** `extract_en.mjs` 全文不做任何 HTML 转义
（本轮 grep 过：整份文件里没有 escape / 实体替换逻辑），所以重抽出来是什么样，
`compendium/en` 里就该是什么样 —— 现在两者不一样，只可能是有人手改过。
全库 en 侧 `&amp;Reference[` 994 处、`&amp;reference[` 234 处，
**无一例外都带 `&amp;`**（本轮 grep 计数；大小写两种写法上游都在用，
`gi` 下都能被 enricher 认，所以小写不是问题、缺 `&` 才是）。
另可对照第 1 条那一叶里的兄弟节点 `<sub data-system="dnd5e">&amp;Reference[exhaustion]</sub>`
—— 那是**另一叶**，但同样带 `&amp;`。唯独本条这一处不带，是上游漏打。

**为什么非修不可（两条，都实测过）**：

1. **不带 `&` 渲染不出规则链接。** dnd5e 5.3.3 的 enricher 是
   `/&(?<type>Reference)\[(?<config>[^\]]+)](?:{(?<label>[^}]+)})?/gi`
   （`systems/dnd5e/dnd5e.mjs:20163`）—— 前导 `&` 是**必需**的（`gi` 所以小写
   `reference` 没问题，缺 `&` 才是问题）。缺了它玩家看到的是裸字面量 `reference[Paralyzed]`。
2. **重抽后不重打，这一叶的中文会被闸门拒收。** `apply_translations.py:199` 的
   `REFERENCE = &(?:amp;)?[Rr]eference\[[^\]]*\]` 同样要求 `&`。中文按硬约束照抄了
   `&amp;reference[Paralyzed]`；英文一旦退回裸写法，英文侧签名里这个记号数为 0、
   中文侧为 1 → **markup mismatch，任何译法都过不了**。本轮用真闸（import
   `markup_signature`，不复写正则）验过，两包各 1 叶，中文多出的记号都是
   `{'&reference[Paralyzed]': 1}`；探针见 `4-临时脚本/2026-08-15-round20/probe_signature.py`。

**为什么漏了（有据可查，不是猜）**：`dd54bdd` 的提交信息里白纸黑字写着

> 上游英文的两处笔误(临时本地补丁,**已记入 english-baseline/LOCAL-PATCHES.md**):
>   * Toothbreaker Hideout/Prison  `@Condition[exhaustion` 缺右方括号
>   * Paralyzing Bolt/Paralyzed    `reference[Paralyzed]` 缺开头的 `&`

—— **作者以为两条都登记了，实际只写进去了第一条。** 该提交在 en 侧只动了 2 叶
（本轮 `git show dd54bdd^:… vs dd54bdd:…` 逐叶验过：正好这两叶，无第三叶），
第二叶就此从表里消失了六天。

会漏的结构性原因：这条被顺手归进了同一提交里「修最后 10 条丢失的 `&Reference`」那一**批**
（那批是**中文侧**补标记），于是它看起来像"批量作业的一员"而不是"一条独立的英文基准补丁"。
加上它和第 1/3 条不同型 —— 那两条会把闸**报红**逼人去查，这一条补完闸就是**绿**的，
再没人回头看。

**两条可复用的教训**：
1. **「闸绿」不等于「已登记」**；「提交信息说已登记」更不等于已登记 —— 要去表里数一遍。
2. **英文基准改动不要和中文批量作业混在同一个提交里**，混了就会被当成批量的一部分漏掉。

**复现命令**（本轮实跑，别照抄结论、自己跑一遍）：

```bash
cd "<项目根>"
node 3-常用脚本/extract/extract_en.mjs \
  --package <FoundryData>/modules/ember \
  --out 4-临时脚本/<本轮>/reextract --pack adventure
node 3-常用脚本/extract/extract_en.mjs \
  --package <FoundryData>/modules/ember \
  --out 4-临时脚本/<本轮>/reextract --pack crucible-adventure
python 4-临时脚本/2026-08-15-round20/diff_leaves.py \
  1-Ember汉化插件/compendium/en/ember.adventure.json \
  4-临时脚本/<本轮>/reextract/ember.adventure.json
```

**2026-08-15 实测结果**（两包各跑一次，数字应当**完全一致**）：

| 包 | 当前叶 | 重抽叶 | 仅一侧有 | 值不同 |
|---|---|---|---|---|
| `ember.adventure.json` | 14 844 | 14 844 | 0 | **5** |
| `ember.crucible-adventure.json` | 20 266 | 20 266 | 0 | **5** |

那 5 叶就是正表的 1-5 条（本条 + `C0nsortium` + `Condition[exhaustion` +
`{Persuasion}` + `skillCheck`），**再无第六条**。
旁证：`english-baseline/ember-0.6.0/`（旧 mappings 的原始抽取）与当前 en 的**值差**
也正好是这 5 叶 × 孪生两包 = 10 叶，与本表条数吻合。

⚠ **别被「3 条」绊到**：`english-baseline/ember-cn-v1.1.0-shipped-en/README.md` 说
「v1.1.0 → 当前，英文变过的叶只有 **3** 条」。**那句话是对的，和本表的 5 条不矛盾** ——
第 1 条与第 5 条同在 `dd54bdd`（2026-08-09），而 `v1.1.0` 是 2026-08-11 才打的 tag
（`git merge-base --is-ancestor dd54bdd v1.1.0` 成立，本轮验过），
**这两条已经烘进 v1.1.0 的英文里了**，所以从 v1.1.0 起算只剩第 2/3/4 条会显示为差异。
换算关系：**从上游 LevelDB 重抽起算 = 5 条；从 v1.1.0 起算 = 3 条。**
比较基准不同，数字就不同 —— 报数时必须说清是拿哪一侧当基准。

**中文侧不需要改**：cn 两包同一路径本来就照抄着 `&amp;reference[Paralyzed]`，
补的是英文基准，补完两侧一致。

---

# 附录 A：已核实但**尚未落地**的候选补丁

> 上面 1-5 条是**已经打进英文基准**的补丁，重抽后必须照打。
> 本附录**不是**那种东西：这里的条目一处都还没改，重抽时**不要**照着打。
> 它们与正表的区别在于：**不卡翻译流程**（EN/CN 两侧同错，所有闸门恒为 0），
> 但会让玩家在正文里看到裸标记。要不要修、怎么修，需要裁决人先定。

## A.1 `ember.crucible-adventure.json` 的非法 enricher 参数（174 处）

- **日期**：2026-08-14（第十四轮 idx 29 / idx 96）
- **清单与逐条改法**：`5-其他内容/reports/2026-08-14-上游-enricher-参数错误清单.md`
- **实测规模**（en 侧与 cn 侧逐桶同数）：
  `[[/skill]]` 非法首参 150 · `[[/skillCheck]]` 9 · `[[/knowledge]]` 5 · `[[/award]]` 10。
- **其中 148 处可机械订正，26 处必须人判**（D&D 属性名当技能用、`social`、`pathfinding`、`stars`）。
- **落地时必须 en/cn 同一批同时改**，否则这些叶子会从「两侧同错」变成「两侧不同」，
  被 `markup_signature` 全部打成 mismatch。

---

# 附录 B：上游缺陷「已知但不修」栏

> 与正表、与附录 A 都不同：这里的条目**不打补丁、也不打算打**。
> 三条的共同点是 —— **不影响译文正确性**，改动它们要么越过汉化的职责边界、
> 要么会与上游自己的修复撞车。登记在这里是为了**下一轮不必重推**：
> 扫描器每轮都会把它们报出来，看到就来这一栏对号，别再当新发现。
>
> 与 `5-其他内容/EXCLUSIONS.json` 的分工：那张表收**判据层面**的永久豁免
> （扫描器每轮重报但本来就该这样）；这一栏收**上游数据自身写坏**的。
> `B.1` 两处都登记，因为它同时是这两件事。
>
> **本栏 2026-08-16 终段复核：4 条**（B.1 / B.2 / B.3，新增 B.4）。
> 同日 `EXCLUSIONS.json` 实测为 **7 条**（旧注写「增至 6 条」，少数了一条，一并订正），
> 其中 `P-name-split-arcturian`（合法分裂）与
> `P-token-story-souvenir`（故事内实体信物，**已按库账订正为 1 叶**，原写 4 叶）
> ——**这两条都不属于本栏**：上游英文没写坏，是我方判据看不出词义／
> 看不出「Foundry 对象 vs 实体小物件」。别把它们搬进来，也别以为本栏该跟着涨。
> 新增的 B.4 才是本栏该收的那一类：**上游把专名拼错／漏了空格，而中文译对了**。

## B.1 `Character Classes / Cleric` 的 `{Kessia}` `{Ordain}` 标签与目标错配（6 处）

- **日期**：2026-08-15（第十六轮）
- **影响包**：`ember.adventure.json`、`ember.crucible-adventure.json`（孪生两包各 3 处）
- **路径**：`Ember Early Access.journals.Character Classes.pages.Cleric.text`
- **实测**：
  - `{Kessia}`（中文标签「凯西亚」，指**大陆**）挂到 `1WoH2TVw0gngrgWL` = Kessian **文化**页；
    按标签语义应指凯西亚大陆页 `2eLAE5AF2iAMlc0e`。
    全库指向 `2eLAE5AF2iAMlc0e` 的链接 **28 处**，指向 `1WoH2TVw0gngrgWL` 的 **27 处** ——
    后者的标签除本页这 2 处外，**全部**是「凯西安 / 凯西安人」（文化/族群）。
  - `{Ordain}`（中文标签「奥尔丹」，指**城市**）挂到 `RxhlhTWqJqB1cZxY` = Ordani **文化**页；
    按标签语义应指奥尔丹城市页 `4gRSL7Tq1pgccdIW`。
    全库指向 `4gRSL7Tq1pgccdIW` 的 **59 处**，指向 `RxhlhTWqJqB1cZxY` 的 **50 处** ——
    后者的标签除本页这 1 处外，**全部**是「奥尔达尼」。

**为什么不修**：错配在**英文侧就存在**，中文只是照着标签译，译文忠实。
改方括号**内部**的目标 id 等于改上游数据，超出汉化职责；一旦上游自己修了还会撞车。
硬约束也明写「方括号内部照抄英文」。

**扫描器表现**：`scan_uuid_swap` 每轮报这 6 处。已同时登记进
`5-其他内容/EXCLUSIONS.json` 的 `P-uuid-kessia-ordain`。

## B.2 ember 0.6.0 有 46 个唯一场景针脚 `pageId` 悬空

- **日期**：2026-08-15（第十六轮）
- **规模**：46 个唯一 pageId × 孪生两包 = **92 条**；其中 **5 个带标签的**会落到条目首页。
- **表现**：玩家点场景上的针脚，跳到该 journal 条目的第一页而不是目标页。

**为什么不修**：`pageId` 不在翻译通道里（我方 `compendium/{en,cn}` 的抽取只带可译文本，
针脚的 id 字段根本不在其中），改它要动上游 pack 数据。**与译文正确性无关**。

**核实程度**：数字由本轮扫描单元报出；本栏只复核到「不在翻译通道、不影响译文」这一层，
未逐个针脚复算。**上游升版后要重数**，别照抄这个 46。

## B.3 `Corpse Loot` 35-35 引用了 crucible 里不存在的 `prybar` 物品

- **日期**：2026-08-15（第十六轮）
- **路径**：`Ember Early Access.tables.Corpse Loot.results.35-35`
  （`ember.crucible-adventure.json` 侧的名字是 `Prybar`／中文「撬棒 Prybar」；
  同一行在 `ember.adventure.json`＝dnd5e 侧是 `Piton`／「岩钉 Piton」，两包本就不同物）
- **实测**：该结果行指向 `Compendium.crucible.equipment.Item.prybar0000000000`，
  而 crucible 0.10.1 的安装目录里 **grep 不到这个 id**，`crucible.equipment` 英文侧
  连 `Prybar` 这个名字都没有 —— 物品整个不存在。

**为什么不修**：这是上游 ember 引了一个 crucible 尚未提供的物品，属两个包之间的版本错位。
名字叶本身译得对（撬棒），玩家看到的文字没问题，只是那条战利品拖不出实体。
**修它要么等上游 crucible 补物品、要么改 ember 的表数据**，两者都不是汉化能做的。

**复核触发**：crucible 升版后重跑一次 grep；上游一旦补了 `prybar` 物品，本条即作废。

## B.4 `Arcturel` 的三处上游拼写错（`Arcurel` / `Arturel` / `the DivesArcturel`）

- **日期**：2026-08-16（第十六轮终段，做 T-1「the Dives＝矿渊」的读库核对时撞见）
- **实测**（`compendium/en` 侧，孪生两包各一份，所以叶数都是偶数）：

| 上游写法 | 叶数 | 位置与上下文 |
|---|---|---|
| `Arcurel`（漏了 `t`） | 4 | `Arctus Plateau Gazetteer.pages.Arcturel.text`「it is often said that **Arcurel** is their bread and butter」· 同志书 `pages.The Dives.exposition`「more sections of the city of **Arcurel** nestled among the gloom」 |
| `Arturel`（漏了 `c`） | 2 | `Glitter in the Dark.pages.Storming the Consortium.summary`「the Silver Beam headquarters in **Arturel**」 |
| `the DivesArcturel`（漏了空格，两个专名粘在一起） | 2 | `Glitter in the Dark.pages.Chessmen Homecoming.text`「a final meeting with Zodi Trask and Varholomew Chess in **the DivesArcturel**」 |

**中文侧全部译对了**：前两处按城名写「阿克图瑞尔」，第三处按「矿渊」＋「阿克图瑞尔」
两个地名分别写出（该叶中文 矿渊×2 · 阿克图瑞尔×11）。

**为什么不修**：与第 2 / 第 4 条（`{Persuasion}`、`C0nsortium`）看着同型，其实**不同类**——
那两条要修，是因为**判据卡住了正确的中文**（`markup_drift` 把手滑的花括号报成缺失、
`content_coverage` 把拼写错里的 `0` 当数字）。这三处**一条闸都没卡**：
`R-dives-mine` 的 `\bDives\b` 因为词边界不成立而根本匹配不到 `DivesArcturel`，
`R-arcturel-vs-arcturian` 是按英文闸**单向**判的（英文命中才查中文），
拼错的那几叶压根进不了闸，中文写对了也不会被误报。
**没有闸被卡住 = 没有修英文基准的理由**，改了反而多一份重抽后要重打的补丁。

⚠ 特别提醒下一轮：**别把 `the DivesArcturel` 当成一个新专名去查词表**，
也别为了让某条闸命中它而去改英文 —— 它就是上游漏了个空格，
按「矿渊」＋「阿克图瑞尔」两个地名理解，中文照现状即可。

**复核触发**：ember 升版重抽英文后，重跑
`grep -c '\bArcurel\b\|\bArturel\b\|DivesArcturel' compendium/en/*.json`；
上游一旦改对拼写，本条即作废（届时 `R-arcturel-vs-arcturian` 的英文闸命中数
会从 286 叶涨到 292 叶，那是**正常的**，不是缺陷）。


