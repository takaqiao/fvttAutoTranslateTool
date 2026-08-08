# 并行批次运行手册（自动循环用）

> 这份手册是给**每次被定时唤醒、上下文可能已经被摘要过**的自己看的。
> 目标：不靠会话记忆，只靠本文件 + 磁盘状态，就能判断做到哪一步、下一步做什么。
> 背景与硬约束见 `PROJECT.md` 阶段 20/21 日志；本文件只讲操作。

## 0. 固定路径

```
$P    = C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project
$REPO = $P\1-Ember汉化插件                      ← 译文库，只有主控能写
$PAR  = $P\3-常用脚本\parallel                  ← BRIEF.md / probe.py / residue.py / prep_units.py（权威副本）
$WORK = <本会话 scratchpad>\parallel            ← 每个单元的 todo.json / batch.json（临时件）
包名固定 ember.crucible-adventure.json
```

`$WORK` 是会话级临时目录，路径形如
`C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\<会话id>\scratchpad\parallel`。
**发 workflow 时提示词里写的是绝对路径**，所以同一会话内一直有效；换了会话要重新 prep。

### 换会话时怎么接手（重要）

新会话拿不到旧会话的 scratchpad，也**没有旧 workflow 的 runId**，所以：

1. **上一批必须先落盘再换会话**。batch 是临时件，只有落进 `compendium/cn` 并 commit 才算数。
   如果换会话时还有没落盘的 batch，那批就得重做 —— 交接前先确认 `git log` 里有它。
2. 新会话开局照做：读 `PROJECT.md` 第 1 节 → 读本文件 → 跑手册第 1 节的三条命令查状态。
   **不需要**旧会话的任何东西，仓库里的东西是自足的。
3. 重新 prep 时把 `$env:EMBER_PARALLEL_ROOT` 指到新会话的 scratchpad，例如：
   ```powershell
   $base = "$env:LOCALAPPDATA\Temp\claude\C--Users-Taka-Desktop-fvtt\<新会话id>\scratchpad\parallel"
   $env:EMBER_PARALLEL_ROOT = $base
   ```
   然后按第 4 节切单元、发 workflow。`3-常用脚本\parallel\` 下的脚本与 `BRIEF.md` 都在仓库里，
   跟着 clone 走，不依赖会话。
4. **不要试图 resume 旧会话的 workflow** —— runId 是会话内的，新会话只能重发。

## 1. 先判断做到哪一步（每次唤醒的第一件事）

按顺序查这三样，不要凭印象：

```powershell
# a. 有没有 workflow 还在跑 / 上一轮是不是被额度截断
#    看 /workflows，或查最近一次 Workflow 调用的 task 通知里的 failures 段
# b. 各单元的 batch 齐不齐
python "$PAR\batch_status.py"          # 需要先把里面的 WAVE2 列表改成当前批次的目录名
# c. 译文库现状
python "$P\3-常用脚本\qa\validate_translations.py" --repo "$REPO" --out "$P\5-其他内容\reports\ember"
cd "$REPO"; git status --short; git log --oneline -3
```

三种情况：

| 现状 | 该做什么 |
|---|---|
| workflow **还在正常跑**（没有 failures、只是没跑完） | **什么都别做**，尤其不要写 `compendium/cn`。汇报一句「仍在运行」就结束这次唤醒 |
| workflow 被额度截断、部分单元没 batch | **resume**（见第 2 节），不要重发新 workflow |
| batch 齐了，但**审校 agent 大面积没跑成** | 也要 **resume**，不要直接落盘。译者已完成的会从缓存回放，只补跑审校与跨块核对 |
| 所有 batch 齐了、审校也做过、但 git 里没提交 | 走第 3 节落盘流程 |
| 已提交、待译清单还有内容 | 走第 4 节发下一轮 |

> 定时唤醒会**撞上正在跑的 workflow**（一轮要一个多小时）。这是正常的，
> 第一行就是为它准备的 —— 空转一次比插进去写文件安全得多。
>
> **额度总是在审校阶段用完**（译者先跑、吃掉大半窗口，审校排在后面）。所以「batch 齐了」
> 很容易被误读成「这批做完了」。落盘前一定要确认审校真的跑过 —— 对抗式审校是这套并行
> 流程唯一的质量保证，跳过它等于把 30 万字未经复核的译文直接推进库里。

## 2. 断点续跑

```
Workflow({ scriptPath: "<上一轮返回的 Script file 路径>", resumeFromRunId: "<上一轮的 Run ID>" })
```
已完成的 agent 从缓存回放（不重跑、不烧额度），失败的重跑。

**resume 之前先跑一次 `batch_status.py`，把「不完整」的 batch 挪走。**
多数失败的译者是一个字都没写，但**第 5 批出现了写到一半就被掐断的**（fill-8 写了 12 条里的 4 条）。
留着的话，重跑的 agent 可能在半截文件上接着写，最后既不是旧的也不是新的。
挪走（改名成 `batch.partial.bak`，别删）再 resume。

### realign 单元（第 6 批起）反过来：**半截成果要留着，不要挪走**

页文件格式把「被掐断」从事故降级成了普通中断，第 6 批实测：
12 个译者全部被额度杀死（0 个 agent 正常返回、无缓存），但磁盘上
**493 页里有 219 页已完成且过闸**，一页没丢。

能这么做的原因是**完成度是按页判定的，判据就是闸门本身**：

| 每单元 | 含义 |
|---|---|
| 闸门 `applied` 计进的页 | 已完成 —— 块补齐了、多余块删干净了 |
| 闸门 `REJECTED markup` 列出的页 | 没做完（含改了一半的） |
| `collect_realign.py` 报的 `UNTOUCHED` | 完全没开始的页 |

实测每个单元都满足 `applied + rejected = 页数`，且 `rejected` 只比 `UNTOUCHED` 多 0–1 ——
也就是说被掐断在半页中间的情况极少，而且**一旦发生就必然被闸门拒**，不会混进成品。
所以 resume 前**什么都不用挪**，只要在译者提示词里加一段「先跑 collect + 闸门找出还没过的页，
只做那些页、已 applied 的一个字别动」，重跑就是纯增量。

> 老 batch.json 流程必须挪走半截文件，是因为「完成度」只有整个 batch 一个粒度，
> 看不出 12 条里写了几条。页文件流程里这个信息是闸门免费给的。

**第 6 批实测的续跑经济性**（每一轮都是 13 个 agent 全灭、`agents_done` 为 0、无任何缓存）：

| 轮次 | subagent tokens | 时长 | 完成页数（累计） |
|---|---|---|---|
| 第 1 次 | 224 万 | 17 分 | 219 / 493 |
| 第 2 次（续跑） | 383 万 | 36 分 | **422 / 493** |

**「agent 全部失败」不等于「这一轮白跑」** —— 任务通知里的 `agents_error 13 / agents_done 0`
看着像彻底失败，实际两轮推进了 422 页。判断进度**只能看磁盘**（collect + 闸门），
不能看 workflow 的返回值；返回值必然是空的，因为返回值是 agent 正常 return 才有的东西。

同理，**审校 agent 也要留进度记录**：让它每审完一页就把页 id 追加进
`<单元>\_review_progress.json`，开工前先读、已列出的跳过。
否则被掐断后重跑的审校会把 41 页从头再审一遍 —— 译者那边靠闸门天然免疫这个问题，
审校没有等价物，只能自己记。

## 3. 落盘流程（batch 齐了之后）

顺序不能乱，每一步都要看输出：

```powershell
# 3.1 逐个 --dry，必须全部 0 拒绝；有拒绝就先修，别硬写
python "$P\3-常用脚本\qa\apply_translations.py" --repo "$REPO" --pack ember.crucible-adventure.json --batch "$WORK\<单元>\batch.json" --dry
# 3.2 全部干净后去掉 --dry 落盘
#     ⚠ **补缺块(fill-*)与跨卷核对改过的 batch 必须加 --force**。
#       第 8c 项的目标路径本来就有中文(那正是它的定义)，不加 --force 会被
#       「已有中文则跳过」闸**静默跳过**，apply 报 skipped(existing) 而不是报错。
#       第 5 批就这么踩过：8 个 fill 单元一条没写进去，而覆盖率因为同批其它单元
#       在涨，看不出异常 —— **是标记漂移一动不动才暴露的**。
#       补缺块落盘后 BLOCK / TRUNCATED 必须明显下降，没降就是没写进去。
# 3.3 跨卷术语核对：如果 workflow 里的 cross-check agent 挂了，单独补一个 Agent 跑
#     （提示词见第 6 节），它改的是 batch.json，改完用 --force 重新落盘
# 3.4 风格归一
python "$P\4-临时脚本\2026-08-06\normalize_quotes.py" --repo "$REPO" --mode quotes --write
python "$P\4-临时脚本\2026-08-06\normalize_quotes.py" --repo "$REPO" --mode dashes --write
# 3.5 QA 全套
python "$P\3-常用脚本\qa\validate_translations.py"  --repo "$REPO" --out "$P\5-其他内容\reports\ember"
python "$P\3-常用脚本\qa\scan_markup_drift.py"      --repo "$REPO" --out "$P\5-其他内容\reports\ember\markup_drift.json"
python "$P\3-常用脚本\qa\scan_foreign_script.py"    --repo "$REPO"
python "$P\3-常用脚本\qa\scan_markup_targets.py"    --repo "$REPO"
python "$P\3-常用脚本\qa\scan_class_drift.py"      --repo "$REPO" --out "$P\5-其他内容\reports\ember\class_drift.json"
python "$P\4-临时脚本\2026-08-06\detect_swapped_pages.py"
# realign 批次专用：块骨架逐位比对（闸门是无序多重集，抓不到「补的块位置错了」）
for ($i=1; $i -le 12; $i++) { python "$PAR\tagseq.py" "$WORK\realign-$i" --stat }
# 3.6 提交（两个仓库分开提交：$REPO 放译文，Desktop\fvtt 放文档与脚本）
```

**第 6 批新增的两项检查**（都是第 6 批才发现闸门看不见的）：

- `tagseq.py` —— 闸门比对的是**无序多重集**，所以「块补对了但插错位置」照样 0 拒绝，
  玩家读到的段落顺序是乱的。tagseq 逐位比对块骨架（标签+class+闭合序列）。
  注意它会把**块内部的语序调整**也报成 MISMATCH（中文把 `<strong>` 挪到 `<span>` 前面之类），
  那是正常翻译，不是缺陷 —— 要看 opcode 再判。
- `scan_class_drift.py` —— 闸门的签名只取**标签名**（`TAGNAME` 只捕获 `<(/?)(\w+)`），
  于是 `<ul class="complex-check">` 和裸 `<ul>` 在它眼里一模一样。
  而这些 class 在本项目里是功能性的：`section.block gamemaster` 决定 GM 内容是否对玩家隐藏，
  `ul.complex-check` / `li.advantage` / `li.critical-success` 决定检定结果怎么渲染，
  `sup.system-swap-inline` 是 dnd5e/crucible 双轨显示。

  首测：2214 条带 class 的已译条目里 **412 条与英文不一致**。但**其中 365 条（89%）
  就是第 6 批在修的 8c/8j 条目** —— 根因同源（中文照旧版英文翻的），落盘后应自行下降。
  **独立于第 6 批的只有 47 条**。
  > 一开始我把「丢了 `section.block gamemaster`」当成「GM 内容泄露给玩家」，**是错的**：
  > 那 36 条的中文里**根本没有那段 GM 内容**（属第 8c 项缺块），不存在泄露。
  > 判据是去中文里搜该段的 `<span class="reference">` 坐标 —— 搜不到就是缺块，不是丢包裹层。

**验收基线**：标记漂移只许降不许升。**升了就说明这批有问题，别提交，先查。**
外来文字必须是 0。`scan_markup_targets` 的 BROKEN **当前 14**（都是已知有原因的：
`@embed[… inline 概览]` 那批，以及上游英文自己写坏的 `swoopingStrike00}`）。

六轮实测（落盘后）：

| | 起点 | 第 5 批后 | **第 6 批后（当前）** |
|---|---|---|---|
| LINK | 689 | 624 | **232** |
| BLOCK | 584 | 519 | **26** |
| INLINE | 265 | 230 | **112** |
| TRUNCATED | 69 | 51 | **20**（含孪生包 +9） |

> 孪生包填充后 TRUNCATED 由 20 升到 29，**那 9 条不是新缺陷**：是 crucible 侧译文本身已陈旧
> （上游换了内容但块数没变），TM 忠实地复制了过来。换句话说 **`TRUNCATED` 还能当
> 「上游换了内容但块数没变」的探针**用 —— `measure_8c` / `measure_stale_extra` 只比块数，看不见这一类。

## 4. 发下一轮

```powershell
# 4.1 先扣掉 babele 会自动解析的部分，别重复翻译
python "$P\3-常用脚本\qa\resolve_generic_fallback.py" --repo "$REPO" --also "$P\2-Crucible汉化插件"
# 4.2 看还剩什么（改 slice_todo.py 里的路径后运行）
python "$PAR\slice_todo.py"
# 4.3 切单元：大卷按页切块，小卷合并，非 journal 桶按字符切
python "$PAR\prep_units.py" split "<大卷名>" 30000
python "$PAR\prep_units.py" merge "小卷合集" "<卷1>" "<卷2>" ...
python "$PAR\prep_units.py" bucket tables 30000        # 桶：tables / actors / scenes / items
python "$PAR\prep_units.py" journals "<卷名>" ...
# 4.4 把旧路径译文挂进工作目录当底稿
python "$PAR\attach_orphans.py" <manifest 文件名>
# 4.5 发 workflow（结构照抄上一轮的脚本文件，改 UNITS 数组即可）
```

**每轮规模**：10–12 个单元、合计 **27–31 万英文字符**（每个译者约 2–3 万）。
配套 = 同样数量的审校 + 1 个跨单元核对，agent 总数 21–25。这个量实测会用掉大半个额度窗口。

## 5. 不可违背的几条

1. **除主控外，任何 agent 都不许写 `compendium/cn`**。译者/审校只写自己单元目录下的 batch.json。
   落盘只由主控做。这样单个 agent 出问题只需丢掉一个 batch 文件。
2. **译者必须自己把 `apply_translations.py --dry` 跑到 0 拒绝才算交付**。
   标记类错误在返回主控前就被挡掉，主控不必逐条复核。
3. **有 workflow 在跑时，主控不要写 `compendium/cn`** —— agent 正在读它（probe / 闸门），
   会读到半截 JSON。所有归一、修错位之类的写操作都攒到 workflow 结束后做。
4. **术语冲突的判断依据强弱**：同名条目/物品的 `name` 字段 > 同卷已译页 > 全库多数写法 >
   `glossary_ec.json` > `BRIEF.md` 的表。BRIEF 的表是二手摘录，**已经被 agent 查出错过 5 条**
   （Inkaro / Amalthea / 引号 / 破折号 / For Other Fortunes）。发现新的错就直接改 BRIEF。

   **⚑ 2026-08-09 起：不一致由主控自行裁决并统一，不要去问项目所有者。**
   反复核对译文、词义、上下文之后按上面的阶梯定，然后 ① 写进 PROJECT.md 第 8 节决议记录
   ② 用 `qa/unify_terms.py` 执行 ③ 复跑 QA 全套。
   此前「不静默择一、列进 disputes 待裁决」现在**只在证据真的不足时**才适用。

   裁决前**必须先查英文**：中文写法不同 ≠ 错。实测「闪电」167 处里，123 处英文本就是
   `Lightning`（忠实）、44 处是比喻，真正该改的只有英文写 `Electricity` 的 23 处。
   `unify_terms.py` 只在**英文原文确实出现该术语**时才改，正是这一条的机械保障 —— 别绕开它手改。
5. **错译要退回待译，不要留着**。留着的话覆盖率算它已译、永不进待译清单，玩家读到错内容。
6. 额度耗尽时**不要重试**，停下等下次唤醒；已完成的部分先提交，别攒着。

## 6. cross-check agent 的提示词要点（workflow 里那个挂了时单独补）

对照 `$PAR\BRIEF.md` 与 `probe.py`，找：同卷不同块之间的说法分叉 → 跨单元冲突 →
与全库既有译法的冲突 → 前几轮已统一项有没有被违反（因卡罗 / 阿玛尔忒亚 / 同调 / 血统 /
矿渊 / 聚归馆 / 异缘会 / 玛伊斯 / 基希尔 / 为了他人的财富 / 引号 “” / 破折号无空格）。
它改 batch.json，改完每个动过的 batch 重跑 `--dry`（已落盘的加 `--force`）必须 0 拒绝。

## 6.5 第 6 批：8c + 8j 合并为「页面重对齐」（2026-08-08 ✅ 已完成并落盘）

**结果**：12 单元 / 493 页 / 补 259,801 + 删 47,467 字符，25 个 agent 全部完成。
512 条（493 页 + 19 条残余）落盘**零拒绝**；第 8c、8j 两项**双双归零**；
BLOCK 519 → 26、LINK 624 → 232。审校 8 GOOD / 4 FIXABLE，critical 0。
共跑四轮（前两轮被额度全灭、第三轮 1 个 API 断连），**这一节保留下来是因为方法本身要复用**。

这一批换了工作方式，原因有三条实测：

1. **8c 与 8j 有 32 条路径重叠** —— 上游把那些页改写过（既加块又删块）。
   分给两个 agent 会让两份 batch 争同一条 path，落盘后一个静默覆盖另一个。
   两者本来就是同一个缺陷的两面（中文是照旧版英文翻的），**合并成一个任务：
   把页面对齐到今天的英文**。闸门是多重集**相等**比较，本来就同时管两个方向，
   所以「0 拒绝」对补和删一样有效。
2. **`prep_8c.py` 把单元规模低估了 5.6 倍**。它按 `est_chars`（缺失内容 26 万）切，
   但交完整替换值的 agent 必须重新产出**整页** —— 全量 145 万字符。
   阶段 23 的 fill-8 写了 12 条里的 4 条就被掐断，就是这么来的。
3. **整页重写会把已校对的译文洗掉**。每一句都重新过一遍模型，措辞必然漂移；
   阶段 23 已经把「有没有被无故重写」列为审校第二条检查项 —— 那是在用人力兜住格式缺陷。

新格式（`prep_realign.py` / `collect_realign.py` / `diff_realign.py`，都在 `$PAR`）：

| 件 | 作用 |
|---|---|
| `pages\<i>.en.html` | 今天的英文，只读 |
| `pages\<i>.cn.html` | **现有中文的逐字节副本**，agent 用 Edit 工具局部改它 |
| `_original_cn.json` | 改动前的原样存档，供 diff |
| `index.json` | 每页 path / missing_blocks / extra_blocks |

没动过的字节保持没动过是**格式保证的，不是靠自觉**；authored 输出从 145 万降到 26 万。
`collect_realign.py` 把编辑后的页文件装配回 batch.json，并报 **UNTOUCHED**
（还与原文逐字节相同的页 = agent 没做）。**交付标准是 UNTOUCHED 0 且闸门 0 拒绝**。
`collect_realign.py <单元> <页id>` 出单页 batch，闸门报告就只讲那一页（闸门最多打印 25 条问题，
41 页的单元不这么做会看不到尾巴）。
`diff_realign.py` 出块级 diff：`+` 新增 / `-` 删除 / `~` 修改 —— **`~` 是审校要盯的**，
英文没变却改了措辞就是洗译文。

顺带修的：`apply_translations.py` 末尾的 `main()` 加了 `if __name__ == '__main__':` 守卫，
好让 prep 复用它的 `split_path()`（页名带点的 `Patch 0.2.0` 用朴素 `split('.')` 解析不到 ——
就是阶段 21 修过的那个 bug，我在新脚本里又踩了一次）。CLI 行为不变。

### 第 6 批之后还剩什么（以下 1、2 项**均已完成**，保留是为了留下依据；当前待办看第 7 节）

1. ✅ **战役包常规待译实际已归零**：371 条里 352 条 babele 自动解析，真实残余 **19 条 / 1,556 字符**
   （7 个文件夹名 + 1 个宏名 + Caustic Phial 物品）。手册第 7 节原记的 4.8 万是**未扣回退**的数。
   量太小，由主控直接译，不值得发 agent。
2. ✅ **孪生包 `ember.adventure`** —— `3-常用脚本/tm/fill_twin.py`，**已于 2026-08-08 落盘，0% → 90%**。
   （下面这段写于落盘前，结论已被实测证实：先落第 6 批再跑，弃填 554 → 143，多填 421 条。）

   crucible 侧推到 98% 之后，TM 覆盖率从 88% 涨到 **98%**：
   12,560 条 / 约 817 万字符可由精确匹配直接填，闸门 `--dry` 实测 **applied 12560 / 0 拒绝**。
   剩 1,290 条 / 14.7 万字符是孪生包独有（1108 条在 `actors` 桶），才需要真翻。

   脚本**不直接写 `compendium/cn`**，只出 batch，由 `apply_translations.py` 落盘 ——
   这样这 817 万字符照样过三道闸，而不是走一条没人检查的私有写入路径。

   两处它特意做了处理：
   - **同一句英文对多个中文**（694 组）：键用 `(最后一段路径, 英文)`，取不到再退回纯英文键；
     仍冲突的取多数并记进报告，不静默择一。多数是「地点」vs「地点 Locations」这类双语格式不一致（既有缺陷 E）。
   - **标记对不上就弃填**（554 条）：两个系统规则不同的地方照搬会错。

   ⚠ **那 554 条里 66% 正是第 6 批在修的 8c/8j 页** —— 中文与自己那侧英文的标记本来就没对齐，
   自然也匹配不上孪生包。第 6 批落盘后重跑 `fill_twin.py`，这批里会有相当一部分转为可填。
   **顺序反了就白白少填几千条。**

   > 剩下那 34%（如 `Aedir Signalpost` 整卷）是**第三类、目前没有专门测量的欠账**：
   > 块数对得上，但块**内部**的标记漂了（@UUID 换了 id、内联命令改了、`<strong>` 丢了）。
   > `measure_8c.py` / `measure_stale_extra.py` 都只数 `<p>`/`<li>` 块数，看不见它；
   > `scan_markup_drift.py` 的 LINK / INLINE 计数（当前 624 / 230）覆盖的就是这一类。
   > 其中「@UUID 指向了错的 id」不是观感问题，是**链接指向错对象**，应单独排一次。
3. **第 8p 项**：世界地图专名重定（Break→破坏、Crown→王冠、Mordant→腐蚀性的、
   WINDBARE→光秃秃的 等，把专名当普通词译掉了）。量小但显眼。
4. **冒烟验证 + 发版**（见 PROJECT.md 第 1 节）。

## 7. 剩余工作量（随每轮更新）

截至第 6 批 + 孪生包填充**已落盘**（2026-08-08）：

| 项 | 状态 |
|---|---|
| 战役包常规待译 19 条 | ✅ 已落盘 |
| 战役包 `ember.crucible-adventure` **真实残余** | ✅ **0** —— 352 条待译全部由 babele 自动解析 |
| 第 8c 项：中文缺块 339 条 / 1772 块 / 26.0 万字符 | ✅ **归零** |
| 第 8j 项：中文多出内容 186 条 / 762 块 / 4.7 万字符 | ✅ **归零** |
| dnd5e 孪生包 TM 可填部分 | ✅ **12,981 条 / 817 万字符已填**（0% → 90%） |
| dnd5e 孪生包 `actors` 桶独有 | ⬜ 1,019 条 / 11.9 万字符 —— 真正需要翻的那部分 |
| 孪生包 journals 143 条 / **71.3 万字符** | 🔶 **不是翻译债**：卡在 crucible 侧标记失配。第 7 批修完重跑 `fill_twin.py` 即自动填上 |
| 孪生包 tables / items / folders | ⬜ 105 条 / 约 6.8 千字符 |

**全库 58% → 95%，待译字符 836 万 → 90 万。**

### 第 7 批：标记签名与英文对不上的 357 条（2026-08-09 进行中）

`prep_sigfix.py` + `BRIEF-SIGFIX.md`，12 单元 × 30 条，产出与 realign 同一套页文件格式，
所以 `collect_realign.py` / `diff_realign.py` / `tagseq.py` / `prose_survival.py` 全部照用。

**这批修什么**：闸门是**写入时**校验的，所以闸门存在之前进库的、或者写的时候英文还是
另一副样子的内容，会永久违反它而不被发现。实测 `ember.crucible-adventure`
**357 条 / 79.1 万字符**（journals 143 / actors 208 / items 6）。
`measure_8c` 与 `measure_stale_extra` 都只比 `<p>`/`<li>` 块数，看不见这一类 —— 坏的是块**内部**。

最常见的一种是 **`<sup class="system-swap-inline">` 双系统分支被压成单支**，
后果是**另一个系统的读者什么都看不到**；其次是内联命令漂移（`[[/skillCheck …]]` ↔ `[[/check …]]`）
与 `@UUID` 指向变了。

**为什么值得单开一批（三重收益）**

1. 这是**已发布的缺陷**：断链与缺失的分支是玩家直接看得到的。
2. **解锁被挡住的工作**：2026-08-09 那轮术语统一有 15 条写不进去，纯粹因为它们既有的标记
   本来就过不了闸（已用「未修改的现值过闸同样全拒」验证）。
3. **免费释放孪生包约 71.3 万字符**：`fill_twin.py` 在 crucible 中文的标记与孪生包英文
   对不上时会弃填，而**那 143 条被弃填的 journal 就是这 357 条里的**。修完这批重跑
   `fill_twin.py`，孪生包自己就填上了 —— 与「先落第 6 批再填孪生包」（弃填 554→143、
   多填 421 条）是同一条顺序教训。

> ⚠ 本批**最容易出且闸门看不见**的错：**双支放反**（dnd5e 的 `<sub>` 里写了 crucible 用语）。
> 签名只数标记个数，不管它落在哪一支里。所以审校第 2 条就是查这个。
> 另外 `prose_survival.py` 在这批是**有效判据**（与第 6 批相反）：这批只该动标记，
> 存活率应当接近 1.00，明显偏低就是译文被重写了。

### 还剩什么（按优先级）

1. **冒烟验证 + 发版** —— 见 PROJECT.md 第 1 节。这是唯一无法靠脚本证实的环节。
   新增两个要看的点：孪生包 `ember.adventure` 的 cn 文件有 **9.09 MB**，
   确认在 Crucible 世界里 babele 不会白 fetch 它；以及补缺块补回来的 GM 段落是否正常隐藏。
2. **孪生包独有 1,280 条 / 14 万字符** —— 够一轮小并行（4–5 个单元）。
3. **既有残留清理**（跨单元核对第 13 项）：既有未译 `{标签}` 约 80 处、正文英文专名
   （Tauric / Cherish / Caberi 等整页英文）。**本批一个新的都没引入**，全是存量。
4. **全库术语 unify 一轮**：`Concluding the Event` 七种写法（事件结束 449 / 结尾 42 / 收尾 23 /
   结算 7 / 结局 6 / 收束 6 / 结语 4）、`Milestone Point` 四种、`Marlstone Manor` 马尔斯通/马尔石、
   `Fernis Ossa` 费尔尼斯/费尼斯。全部是既有译文，铁律保护下本批未动。
5. **name 字段本身要改的**（正文已统一，name 不改就永久打架）：
   `Vista: Ordain Streets` → 远景：奥尔丹街道（现为「授命街道」，把 ordain 当动词的机翻）；
   `Vista: Yakoshta` / `Vista: Arbore Sanctorus` 根本没译；
   `Young Cheliceraeth`（name「幼年螯蛛以太兽」4 处 vs 全库「幼年螯蛛艾斯」22 处，且
   archetype/macro 都作螯蛛艾斯）；`Tauric`（name 是机翻的「牛族」）。
6. **class 漂移残余 49 条** —— `scan_class_drift.py`，多为 `ul.complex-check` / `li.advantage`。
7. **TRUNCATED 29 条** —— 现在知道它还能当**「上游换了内容但块数没变」的探针**用：
   `measure_8c` / `measure_stale_extra` 只比块数，这一类它们看不见（实证见 PROJECT.md 阶段 24）。

六轮实测：**7500 余条 / 约 170 万英文字符**落盘零拒绝，标记漂移六轮全部只降不升
（LINK 689→232、BLOCK 584→26、INLINE 265→112、TRUNCATED 69→20）。
补缺块类批次的验收判据：**落盘后 BLOCK / TRUNCATED 必须明显下降**，
没降就是没写进去（八成是漏了 `--force`）。第 6 批实测 BLOCK 519→26，判据有效。
