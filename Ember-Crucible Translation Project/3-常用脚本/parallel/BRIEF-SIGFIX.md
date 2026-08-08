# 标记签名修复批 · 译者须知

**先读 `BRIEF.md`**（标记硬规则、既定译名、文风）。本文件只讲这一批**不一样**的地方。

---

## 1. 这批修的是什么

这些条目**都已经有中文**，覆盖率算 100%，读起来也通顺。问题在**标记**：
中文携带的标记与今天的英文对不上。闸门是在**写入时**校验的，所以闸门存在之前进库的、
或者写的时候英文还是另一副样子的内容，会永久违反它而不被发现。

三种典型（`index.json` 里每条都给了精确的 `diff`）：

| 现象 | 后果 | 怎么修 |
|---|---|---|
| `<sup class="system-swap-inline"><sub data-system="dnd5e">…</sub><sub data-system="crucible">…</sub></sup>` 被压成了单支 | **另一个系统的读者什么都看不到** | 把两支结构补回来，缺的那支正文要译出来 |
| `[[/skillCheck awareness 15]]` 漂成了 `[[/check 15 awareness]]`（或反向） | 检定按钮点不动 / 解析错 | **照抄今天的英文** |
| `@UUID[...]` 指向的 id 与英文不同 | 链接指到别的东西 | 照抄今天英文的 id |

## 2. 你的输入

```
<你的单元>\pages\<i>.en.html   今天的英文，**只读**
<你的单元>\pages\<i>.cn.html   现有中文 ← 你用 Edit 工具改这个
<你的单元>\index.json          每条的 path 与 diff
```

`diff` 的读法：`{"标记": [英文里几个, 中文里几个]}`。
`[1, 0]` = 英文有、中文缺 → **要补**；`[0, 1]` = 中文多出来 → **要删或改对**。

`missing` / `surplus` 是该条缺多少个、多出多少个标记的合计。

## 3. 铁律：只动标记，不要重写译文

这些句子是校对过的。**除了标记本身、以及为补回缺失分支而必须新写的正文，一个字都不要改**——
不许润色、不许换词、不许调整语序、不许「顺手统一」术语。

用 **Edit 工具**做局部修改。**不要用 Write 整个重写 `cn.html`** —— 那正是会把校对过的
译文悄悄洗掉的操作，本批的文件格式就是为了防它。

补回 `system-swap` 分支时要注意：
- **两支的 `data-system` 值不能弄反**。dnd5e 那支写 dnd5e 的规则用语（`[[/check …]]`、
  `@UUID[Compendium.dnd5e…]`），crucible 那支写 crucible 的（`[[/skillCheck …]]`）。
- 方括号内部**一律照抄英文**，只有 `{标签}` 要译（见 `BRIEF.md` 第 2 节）。
- 英文源本身有写坏的（缺右括号、`@UUID[[…]]` 双左括号、`swoopingStrike00}` 用了花括号）。
  **照抄，不要替上游修**，然后在报告里列出来。这类条目可能永远过不了闸 —— 列出来即可，别硬改。

## 4. 自检（改一条查一条）

```powershell
$P   = "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
$PAR = "$P\3-常用脚本\parallel"
$U   = "<你的单元目录>"

python "$PAR\collect_realign.py" $U <i> --quiet
python "$P\3-常用脚本\qa\apply_translations.py" --repo "$P\1-Ember汉化插件" `
  --pack ember.crucible-adventure.json --batch "$U\batch.only.json" --dry --force
```

闸门会打印 `{标记: (英文里几个, 你写了几个)}`，照着补/删到一致。
**`--force` 必须加**（目标路径本来就有中文），`--dry` 保证不写盘。

全单元收尾：

```powershell
python "$PAR\collect_realign.py" $U
python "$P\3-常用脚本\qa\apply_translations.py" --repo "$P\1-Ember汉化插件" `
  --pack ember.crucible-adventure.json --batch "$U\batch.json" --dry --force
```

**交付标准**：`UNTOUCHED 0` 且 `REJECTED 全 0`。
本批的「0 拒绝」含义最直接 —— **签名相等就是这批缺陷的定义本身**，
所以闸门通过 = 这条修好了。改不动的（上游英文自己坏了）单独列出来，不要硬凑。

## 5. 返回什么

不要贴译文。只要结构化摘要：

- 单元名、修了几条、补了多少标记 / 删了多少
- `UNTOUCHED` 与闸门最终数字
- **你补回了哪些 `system-swap` 分支**（列出 path），以及新写的那支正文有多长
- 上游英文本身写坏、你照抄的（列出 path 与坏在哪）
- 拿不准的
