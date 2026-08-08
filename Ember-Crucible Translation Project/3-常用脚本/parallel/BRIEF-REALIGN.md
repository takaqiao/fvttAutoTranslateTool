# 第 8c + 8j 项 · 页面重对齐 · 译者须知

**先读 `BRIEF.md`**（标记硬规则、既定译名、文风）。本文件只讲这一批**不一样**的地方。

---

## 1. 这批的任务不是翻译，是「对齐」

这些页面**都已经有中文了**，覆盖率算它们 100%。问题是那份中文是照着**旧版英文**翻的：

- **缺块（8c）**：上游后来加了段落，中文停在旧版本 → 玩家读不到新增的规则与场景
- **多块（8j）**：上游删了段落，中文还留着 → 玩家读到早已被删除的内容
- **两者都有**：上游把整页改写过。本批有 **32 页**是这种，`index.json` 里
  `missing_blocks` 与 `extra_blocks` **同时非零**的就是

你的任务：**把中文改到与今天的英文逐块对应**。补上缺的，删掉多的，其余一个字都不要动。

## 2. 铁律：已有中文原样保留

那些句子是前几轮校对过的。**除非对应的英文确实变了，否则不许改**——
不许润色、不许换词、不许调整语序、不许"顺手统一"术语。

这就是本批不用「整页重写」而用**逐块编辑**的原因：

```
<你的单元>\pages\<i>.en.html   今天的英文（只读，不要改）
<你的单元>\pages\<i>.cn.html   现在的中文（← 你改这个文件）
<你的单元>\index.json          每页的 path / missing_blocks / extra_blocks / en_blocks
```

`<i>.cn.html` 一开始是**现有译文的逐字节副本**。用 **Edit 工具**做外科手术式修改：
插入缺的块、删除多的块。**不要用 Write 整个重写这个文件**——那正是会把校对过的译文
悄悄洗掉的操作。没动过的字节保持没动过，是靠这个格式保证的，不是靠自觉。

## 3. 怎么对齐

1. 读 `<i>.en.html` 和 `<i>.cn.html`，按**块**（`<p>` / `<li>` / `<h3>` / `<h4>` /
   `<section>` / `<blockquote>`）逐一配对
2. 英文有、中文没有 → 译出来，**插到正确位置**（不是统统追加到末尾）
3. 中文有、英文没有 → 整块删掉（连同它的标签，不要留空 `<p></p>`）
4. 块在但**块内的标记变了** → 按今天的英文改标记，正文照旧

第 4 条比想象中常见，实测已经撞到这几类：

| 现象 | 处理 |
|---|---|
| 英文 `[[/skill perception 13]]`，中文 `[[/check perception 13]]` | 内联命令**照抄今天的英文**。上游改过 dnd5e 侧的命令名 |
| 英文 `[[/damage 1d10 Lightning]]`，中文 `[[/damage fire 1d10]]` | 同上，参数顺序与词都照抄 |
| 中文里 `[[/skill sleightofhand 14 t00l=thief]]`（`t00l` 是零） | 中文侧的既有损坏，按英文改回 `tool=thief` |
| `@UUID[Actor.9plFRf3Hurd9r7ol]` 换成了另一个 id | 上游改了引用对象，照抄新 id |
| `<sup class="system-swap-inline">` 里 `<sub data-system="dnd5e">` / `crucible` 两支数量对不上 | 双轨显示结构，**两支都要在、两支的正文都要译**。数量必须与英文一致 |

## 4. 自检（改一页查一页，别攒到最后）

```powershell
$P   = "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
$PAR = "$P\3-常用脚本\parallel"
$U   = "<你的单元目录>"

# 单页检查：把第 <i> 页单独打成 batch，报告就只讲这一页
python "$PAR\collect_realign.py" $U <i> --quiet
python "$P\3-常用脚本\qa\apply_translations.py" --repo "$P\1-Ember汉化插件" `
  --pack ember.crucible-adventure.json --batch "$U\batch.only.json" --dry --force
```

`REJECTED markup` 会打印 `{标记: (英文里几个, 你写了几个)}`，照着补/删即可。
**`--force` 必须加**：这批的目标路径本来就有中文（那正是缺陷的定义），
不加会被「已有中文则跳过」闸静默跳过，看起来像通过了其实一条没查。
`--dry` 保证不写盘，可以反复跑。

全单元收尾：

```powershell
python "$PAR\collect_realign.py" $U
python "$P\3-常用脚本\qa\apply_translations.py" --repo "$P\1-Ember汉化插件" `
  --pack ember.crucible-adventure.json --batch "$U\batch.json" --dry --force
```

**交付标准（两条都要达到）**：
- `collect_realign.py` 报 **UNTOUCHED 0** —— 每一页都真的动过
- 闸门报 **applied 41 / REJECTED 全 0**

这批的「0 拒绝」比平时更有分量：补齐之前这些页**必然**被拒（缺的块里带着 `@UUID` 与标签），
所以 0 拒绝直接证明块补齐了、多的块删干净了。反过来，
**UNTOUCHED 不为 0 就是没做完**，别报完成。

## 5. 术语

新补的段落里出现的专名，**先查再定**：

```powershell
python "$PAR\probe.py" "Vorg" "Silver Beam"        # 查全库既有中文写法
python "$PAR\probe.py" --names "Lantern Roads"     # 查页名/条目名对照
```

同一页里已有的译法优先级最高——你补的段落要和它上下文一致。
判断依据强弱：同名条目的 `name` 字段 > 本页/本卷已有译文 > 全库多数 > `glossary_ec.json` > BRIEF 的表。

## 6. 返回什么

不要贴译文。只要一段结构化摘要：

- 单元名、处理页数、补了多少块 / 删了多少块
- `UNTOUCHED` 与闸门最终数字（必须 0 / 全 0）
- 你**删掉**的内容里有没有值得一提的（删掉整段规则、整个事件分支之类，列出来）
- 新定的专名（英文 → 中文），后面要做跨单元一致性检查
- 拿不准的地方；尤其是**块配不上**（英文改写太多、无法逐块对应）的页，列出 path
