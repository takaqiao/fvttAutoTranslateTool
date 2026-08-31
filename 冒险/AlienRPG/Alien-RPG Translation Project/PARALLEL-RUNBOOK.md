# 并行批次运行手册（异形 RPG 汉化）

> **这份文档是写给一个上下文已经被摘要掉的未来的我看的。**
> 唯一可信的状态源是**磁盘**——不是会话记忆，不是上一条消息，不是任务书里的数字。
> 每次唤醒先跑第 1 节的三条命令，从输出判断做到哪一步。
>
> 方法论承自 `..\Ember-Crucible Translation Project\PARALLEL-RUNBOOK.md`，
> 已按异形项目的实际结构改写。**硬约束在 `PROJECT.md` §3，这里不重复，只讲怎么跑。**

---

## 0. 固定路径

```powershell
$ROOT = "C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"
$HUB  = "$ROOT\1-系统汉化插件"        # alienrpg-cn
$SS   = "$ROOT\2-新手包汉化插件"      # alien-evolved-starterset-cn
$CR   = "$ROOT\3-核心书汉化插件"      # alien-evolved-corerules-cn
$QA   = "$ROOT\4-常用脚本\qa"
$PAR  = "$ROOT\4-常用脚本\parallel"
$DATA = "C:\Users\Taka\AppData\Local\FoundryVTT\Data"
```

**会话级波次工作区**（不在 git 里，会话结束即失联）：

```powershell
$env:ALIEN_PARALLEL_ROOT = "<本会话 scratchpad>\alien\parallel"
```

### 换会话时怎么接手

1. `$env:ALIEN_PARALLEL_ROOT` 指到**新**会话的 scratchpad，然后重跑 prep 脚本重建单元目录。
   旧会话的目录读不到，`runId` 也是会话本地的——**上一个会话没落地的东西就是没了**。
2. **所以：切会话之前，每个 batch 必须先落盘、先提交。**
3. 所有脚本和任务书都在仓里，新会话自给自足；不要依赖"上次那个 agent 说过什么"。

---

## 1. 先判断做到哪一步（每次唤醒的第一件事）

```powershell
# a. 有没有 workflow 还在跑 / 上一轮是不是被额度截断
#    看 /workflows，或查最近一次 Workflow 任务通知里的 failures 段
#    ⚠ 有波次在跑 ⇒ 什么都别写，报「仍在运行」然后停。
#       主控写 compendium/cn 时 agent 正在读它，会读到半截 JSON。

# b. 各单元的 batch 齐不齐
python "$PAR\batch_status.py"

# c. 译文库现状（三个仓分别看）
python "$QA\validate_translations.py" --repo "$HUB"
python "$QA\validate_translations.py" --repo "$SS"
python "$QA\validate_translations.py" --repo "$CR"

# d. 三道异形专属闸（这三道是本项目独有的，EC 没有）
python "$QA\scan_name_lookup_traps.py"
python "$QA\scan_crit_lockstep.py"
python "$QA\scan_table_ref_sync.py"

# e. 工作树状态（判断有没有未提交的成果）
git -C "$HUB" status --short; git -C "$HUB" describe --tags
git -C "$SS"  status --short
git -C "$CR"  status --short
git -C "C:\Users\Taka\Desktop\fvtt" status --short -- "Alien-RPG Translation Project"
```

⚠ **`validate_translations` 的覆盖率数字在跑 `resolve_generic_fallback.py` 之前不可信。**
Babele 会顺着 `_stats.compendiumSource` 自动回源取译文，那部分会被计成"缺口"。
EC 实测：报 97%/99% 带 436 条"缺失"，其中 352 条其实是免费回落，
真残余只有 19 条——照着那 436 条派活会**制造出同英文不同中文的分叉**。

---

## 2. 断点续跑

```powershell
python "$PAR\batch_status.py"   # 每个单元：batch 有没有、条数、是否完整
```

**不完整的 batch 一律挪走，不要删**：`batch.json` → `batch.partial.bak`。
额度掐死的 agent 会留下写了一半的 JSON，重试时被捡起来就产出一个**既不是旧版也不是新版**的东西。

### ⚠ 闸门只是**某些**批次的完成信号，不是所有批次的

- **整叶替换型批次**（新翻）：闸门 0 拒绝 ≈ 完成。
- **页面手术型批次**（重对齐 / 补漏 / 术语统一）：完成信号是
  `collect_realign.py` 报的 **UNTOUCHED == 0**——还与原文逐字节相同的页 = agent 根本没动它。
  闸门对"没动过的页"当然是 0 拒绝，**它看不出没干活**。

### 页面手术型单元反过来：半截成果要留着，不要挪走

那种格式下完成度是**按页**判定的，已经改好的页是有效成果。
把整个目录挪走等于把改好的页一起扔了。让它留在原地，重跑时按页续。

---

## 3. 落盘流程（batch 齐了之后）

```powershell
# 3.0 ⚠ 多个批次基于同一 base 生成时，必须先三方合并再落
#     apply_translations.py 是整叶覆盖；按顺序落 = 先落的被静默回滚
python "$PAR\merge_batches.py" --repo <repo> --units <单元列表>

# 3.1 逐个 --dry，必须全部 0 拒绝；有拒绝先修，不要硬写
python "$QA\apply_translations.py" --repo <repo> --batch <unit>\batch.json --dry

# 3.2 全部干净后去掉 --dry 落盘
#     ⚠⚠ 补漏 / 回填 / 重对齐 / 术语统一批次**必须加 --force**
#        它们按定义就打在已有中文的路径上，不加 --force 会被「已有中文则跳过」闸
#        **静默跳过**，apply 报 skipped(existing) 而不是报错。
#        验收看缺陷不看闸：落盘后 BLOCK / TRUNCATED 必须**肉眼可见地下降**，没降就是没写进去。

# 3.3 跨单元术语核对（workflow 里那个 agent 挂了就单独补，提示词见第 6 节）
#     它改的是 batch.json，改完用 --force 重新落盘

# 3.4 异形专属三闸——每次落盘后都要跑，不是发版前才跑
python "$QA\scan_name_lookup_traps.py"
python "$QA\scan_crit_lockstep.py"
python "$QA\scan_table_ref_sync.py"

# 3.5 QA 全套（见 PROJECT.md §5.4）

# 3.6 页面手术型批次专用：块骨架逐位比对
python "$PAR\tagseq.py" <unit>       # 闸门是无序多重集，抓不到「块补对了但位置错了」
python "$PAR\prose_survival.py" <unit>  # 读 `~` opcode：英文没变却改了措辞 = 洗译文

# 3.7 提交。四个仓分开提交：
#     $HUB / $SS / $CR 放译文，Desktop\fvtt 放文档与脚本
```

---

## 4. 发下一轮

```powershell
# 4.1 先扣掉 babele 会自动解析的部分，别重复翻译
python "$QA\resolve_generic_fallback.py" --repo <repo>

# 4.2 看还剩什么
python "$PAR\slice_todo.py" --repo <repo>

# 4.3 切单元
#     journal 大页按 h2/h3 边界切块；小卷合并；非 journal 桶按字符切
#     ⚠ 按**可见**字数切，不按原始 HTML 字数——本语料可见率 42%–77%，
#        个别页低到 13%（CR-J-14b：59,662 原始 / 8,438 可见）
#     ⚠ 绝不给两个 agent 重叠的路径集
python "$PAR\prep_units.py" --repo <repo> --out "$env:ALIEN_PARALLEL_ROOT"

# 4.4 把旧路径译文挂进工作目录当底稿（改名/搬家后的续命）
python "$QA\port_orphans.py" --repo <repo>
#     ⚠ port_orphans 搬的是**路径**，不是**文字**。上游改名后译文会落在新路径上、
#        正文里还写着旧名字。搬完必须 grep 一遍旧名。

# 4.5 发 workflow（结构照抄上一轮的脚本文件，改 UNITS 数组即可）
```

---

## 5. 不可违背的几条

1. **除主控外没有任何 agent 写 `compendium/cn`。** 译者/审校只写自己单元目录下的 `batch.json`。
   这样一个 agent 出问题只损失一个 batch 文件。
2. **译者必须自己把 `apply_translations.py --dry` 跑到 0 拒绝才算交付。**
   markup 类错误在返回主控之前就被挡掉，主控不必逐条复核。
3. **有 workflow 在跑时主控不写 `compendium/cn`。** 归一、修错位之类的写操作攒到波次结束后做。
4. **术语冲突由主控自行裁决并统一，不要问业主**（证据真的不足时才列进 `disputes.json`）。
   阶梯见 `PROJECT.md` §3.3。裁决前**必须先查英文**——中文写法不同 ≠ 错。
   用 `tm/term_gate.py` 拿英文闸的三桶计数，用 `qa/unify_terms.py` 执行
   （它只在英文原文确实出现该术语时才改，**别绕开它手改**）。
   裁完 ① 写进 `PROJECT.md` §8 ② 执行 ③ 复跑 QA 全套。
5. **错译退回待译，不要留着。** 留着的话覆盖率算它已译、永不进待译清单，玩家读到错内容。
6. **额度耗尽时不要重试**，停下等下次唤醒；已完成的部分先提交，别攒着。
7. **⚑ 异形专属**：任何批次碰到 `PROJECT.md` §1.2 的三条硬耦合面
   （重伤表结果正文 / RollTable 名 / 首次导入查找名）时，
   先读 `7-其他内容\DO-NOT-TRANSLATE.json`，不要靠记忆。

---

## 6. 跨单元术语核对 agent 的提示词要点

对照 `$env:ALIEN_PARALLEL_ROOT\BRIEF.md` 与 `probe.py`，按这个顺序找：

1. 同一卷不同块之间的说法分叉
2. 跨单元冲突
3. 与全库既有译法的冲突
4. **前几轮已统一项有没有被违反**——本项目当前已钉死的：
   异形（生物）/ 外星（形容词）· 抱脸虫 · 破胸体 · 韦兰-尤坦尼集团 · 诺史莫号
   （其余见 `7-其他内容\glossary\glossary_alien.json`，**以文件为准，不以这一行为准**）
5. **三档命名有没有被违反**（`PROJECT.md` §3.4）——
   T-BILINGUAL 用**一个 ASCII 空格**分隔，不是括号；
   T-EXACT 的 12 个 skill-stunts 名**不许带英文尾巴**；
   T-FROZEN 的那批**必须还是英文**。

它改 `batch.json`；改完每个动过的 batch 重跑 `--dry`（已落盘的加 `--force`），必须 0 拒绝。

---

## 7. 本语料特有的切分注意事项

> 数字来源：`7-其他内容\findings\2026-08-29-survey\content-inventory.json`
> 及其对抗式复核 `verify-inventory.json`。**以文件为准，这里只讲形状。**

- **journal 正文占 82.5%**（223 万 / 270 万）。切分主要就是切 journal。
- **先翻的锚点单元**：MU/TH/ER 使用说明（锁死所有卡片标签，必须与 `lang/cn.json` 逐字符一致）
  → 技能与天赋 → 角色 → 异形物种 → 派系与公司名。**锚点没落地，别开并行。**
- **复用面很大，别翻两遍**：
  - 11 张 RollTable 跨包共享，其中 10 张逐字节相同（省约 2.57 万字）
  - Starter Set Rules 有 43% 的段落与核心书第 2/3/4 章逐字节相同
  - Map Pins 的 17 页是 Hope's Last Day 里位置段落的 **100% 复制**（纯粘贴，零新翻译）
  - 怪物 actor 的 `system.general.special.value`：6.69 万字里只有 1.39 万是独特的
    （8 段样板各出现 16 次）
  - **业主的优先级要求 starterset 先于 corerules，所以复用方向反过来**：
    先译 starterset，再用 `tm/fill_twin.py` 灌进 corerules。方向本身无所谓，
    重要的是相同英文得到相同中文。
- **不要派的单元**：9 张表 165 条结果的 description 完全为空（纯骰点范围），只需译表名；
  另有 24 张是标签级（每条 6–39 字）。
- **92 页是纯图片页**（Art of Alien 74 + 18），`image.caption` 全空，只有页名要译。
- **朗读文本**：`class="evblockquoteCenter"` 是居中方框 = 朗读段落标记。
  Hope's Last Day 是全语料唯一带显式"read the following"指示的文档，散文标准最高，**单人负责**。
  `class="evexamplebox"` 是规则示例，不是朗读文本，别混。
- **表格类页面按可见字数派工**：第 11 章 CAMPAIGN PLAY 原始 35.8 万但可见只有 11.6 万。
  ⚠ 该页的"表格字数 / 引用块字数 / 示例框字数"三个数**互相嵌套、重复计数**，不能相加。

---

## 8. 剩余工作量

*（随每轮更新。当前：Phase 0 脚手架进行中，尚未开始任何翻译波次。）*
