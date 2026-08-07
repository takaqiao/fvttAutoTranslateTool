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
python "$P\4-临时脚本\2026-08-06\detect_swapped_pages.py"
# 3.6 提交（两个仓库分开提交：$REPO 放译文，Desktop\fvtt 放文档与脚本）
```

**验收基线**：标记漂移只许降不许升。前三轮实测每轮都在降
（LINK 689→687→684→682，BLOCK 584→582→579→577）。**升了就说明这批有问题，别提交，先查。**
外来文字必须是 0。`scan_markup_targets` 的 BROKEN 应保持 15（都是已知有原因的）。

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
5. **错译要退回待译，不要留着**。留着的话覆盖率算它已译、永不进待译清单，玩家读到错内容。
6. 额度耗尽时**不要重试**，停下等下次唤醒；已完成的部分先提交，别攒着。

## 6. cross-check agent 的提示词要点（workflow 里那个挂了时单独补）

对照 `$PAR\BRIEF.md` 与 `probe.py`，找：同卷不同块之间的说法分叉 → 跨单元冲突 →
与全库既有译法的冲突 → 前几轮已统一项有没有被违反（因卡罗 / 阿玛尔忒亚 / 同调 / 血统 /
矿渊 / 聚归馆 / 异缘会 / 玛伊斯 / 基希尔 / 为了他人的财富 / 引号 “” / 破折号无空格）。
它改 batch.json，改完每个动过的 batch 重跑 `--dry`（已落盘的加 `--force`）必须 0 拒绝。

## 6.5 下一批（第 6 批）该做什么

第 5 批落盘后，战役包的**常规待译清零**，剩下的都是「覆盖率看不见」的欠账。按优先级：

1. **第 8c 项剩余**（补缺块，约 26 万字符）。用
   `python "$PAR\prep_8c.py" 32000 --min 300` 重新切 —— 它读的是
   `reports/ember/missing_blocks.json`，**必须先重跑 `4-临时脚本/2026-08-06/measure_8c.py`**
   刷新那份报告，否则会把第 5 批已经补好的又发一遍。
2. **第 8j 项**（译文里留着英文早已删除的内容，189 条 / 约 4.8 万字符）。
   清单在 `reports/ember/stale_extra_blocks.json`，由 `measure_stale_extra.py` 生成。
   这一类要**删**不要补，提示词得反过来写。
3. **孪生包 `ember.adventure`**：先写 `tm/fill_twin.py`（88% 可由精确匹配 TM 直接填，
   见 `4-临时脚本/2026-08-06/measure_twin_tm.py` 的实测），填完再看剩多少需要真翻。
   独有部分只有 14.2 万字符。
4. **第 8p 项**：世界地图专名重定（Break→破坏、Crown→王冠、Mordant→腐蚀性的、
   WINDBARE→光秃秃的 等，把专名当普通词译掉了）。量小但显眼。

## 7. 剩余工作量（随每轮更新）

截至第 5 批落盘（2026-08-08）：

| 项 | 字符 | 状态 |
|---|---|---|
| 战役包常规待译 | **4.8 万** | 覆盖率 98%，剩的多是 babele 会自动解析的内嵌物品 |
| 第 8c 项：中文缺块 | **26.0 万** | 已做掉 27.2 万；重切前**先重跑** `measure_8c.py` |
| 第 8j 项：中文多出内容 | 4.8 万 | 未开始，`measure_stale_extra.py` 出清单；这一类要**删**不要补 |
| dnd5e 孪生包独有 | 14.2 万 | 未开始；另有 88% 可由 TM 脚本直接填，需写 `tm/fill_twin.py` |

五轮实测：**7000 余条 / 约 140 万英文字符**落盘零拒绝，标记漂移五轮全部只降不升
（LINK 689→624、BLOCK 584→519、INLINE 265→230、TRUNCATED 69→51）。
