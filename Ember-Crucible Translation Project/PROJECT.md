# Ember / Crucible 汉化项目 · 主文档

> 这是本项目的**唯一长期入口**。新会话请先读第 1 节，再按需跳读。
> 阶段日志（第 6 节）只追加、不重写，用来做长期校对与断点续做。

---

## 1. 快速跟进（新会话必读）

**2026-08-14：第十三～十四轮。当前已发布 `crucible-cn 0.9.7` / `ember_cn_unofficial v1.1.10`
（v1.1.8 / v1.1.9 是两条阻断的紧急修复，见第 8 节第十三轮）。
第十四轮把第十三轮遗留的 270 条 finding **逐条对当前 HEAD 重新复核**（OPEN 203 / FIXED 63 /
UPSTREAM 2 / INVALID 1），再按「一个文件只归一个 agent」切成 13 个工作包全部处理完，
落盘 **266 叶 + lang 21 键 + 孪生同步 67 叶**，5.4 全套复验回到全 0。**

> ⚠ **第十四轮有三条明确「本轮不做」的项，别当成已完成**：
> `Region Map`/`Area Map` 全库拆分（159 叶，72 叶无法机械对齐）· `[[/language moiré]]` 4 处 ·
> dnd5e 侧两条管线阻断。理由与数据全在 `4-临时脚本/2026-08-14-round14/RESOLUTIONS.md`。

> 上一版抬头：第八～十二轮，`crucible-cn 0.9.6` / `ember_cn_unofficial v1.1.7`。
第十二轮把**所有仍非零的报告清到零或清成有据可查的永久豁免**（123 条，现 125 条，全表在
`4-临时脚本/2026-08-13-round12/findings/EXCLUSIONS.json`）。

> 上一版抬头：第八～十一轮，`crucible-cn 0.9.5` / `ember_cn_unofficial v1.1.5`。
合计约 9000 叶落盘。**全部 75 本 journal 现已逐句对读完毕**（最后 40 本 380 万字符在第十轮补齐）。
页内锚点断链 590 处修复归零。⚠ **冒烟验证仍未做**，已积压六轮。**

> 上一版状态：第八、九两轮 5341 叶落盘。
第九轮主攻「全库同一英文串 N 种中文」：1514 组 → 470 组，8308 叶 → 2561 叶，
其中 >300 字符长正文分叉（几乎必是缺陷的那一档）**357 组 → 0**。
dnd5e 侧两条管线阻断项目所有者已定「先不管」。**

> 上一版状态：第三～七轮 1000+ 条确认缺陷（5 阻断），已发布为
> `crucible-cn 0.9.4` / `ember_cn_unofficial v1.1.4`，两个仓库覆盖率 100%、5.4 全套全 0。
> **第八轮就是从「5.4 全 0」这个状态出发的**，这本身是方法教训 1 的又一次验证。

> ⚠ **这一版同样是在「没做过冒烟验证」的情况下发出去的**（项目所有者明示先发）。
> 而且**本轮动了 Scene 层 mapping（本项目第一次）与 `.mjs` 的两条运行时机制修复** ——
> 这两类只有开真实世界才能确认。冒烟清单见第 7 节末尾，其中场景层级那五点是本轮新增。

| 轮 | 做了什么 | 产出 |
|---|---|---|
| 三 | 旧英/新英 drift 复核 · `@UUID` 标签 226 条 · 四个新判据 · crucible-fr 交叉校验 | 161 条（3 阻断） |
| 四 | **35 本 journal 逐句复核**，实读 458 万英文字符 | 507 条（2 阻断） |
| 五 | 复核补漏 188 条 · 标签↔name 153 组 · 孪生包存量分叉 484 条 | 800+ 处 |
| 六 | **补 Scene 三字段管线**（`levels`/`tokens`/`navName`）并译完 | 517 处，覆盖率到 100% |
| 七 | 裸英文专名 160→63 · tokenName 32 条 · **同名不同译 54→7** · uuid_swap 98→22 · 闸门对齐洞捞回 7 条 | 200+ 处 |
| **八** | **12 类全新判据探矿**（每类各自写脚本 + 双向回测）＋ 13 路对抗式复核 | **683 叶落盘**，另查出 **2 条 dnd5e 管线阻断** |

### 第八轮（2026-08-13，25 agent）做法与结果

打法与前七轮不同：不是「跑现有检查找残留」，而是**每路 agent 发明一个至今没有判据的错误类别**，
写脚本、做**双向回测**（特异度逐条人看 + 灵敏度往临时副本注入已知错误），再交实证。
12 路探针报 121 条 confirmed / 43 条 uncertain，13 路对抗式复核（默认立场「这条是假阳性」）
**存活 136 / 推翻 63（推翻率 32%）**。

| 新判据 | raw→confirmed | 备注 |
|---|---|---|
| 数值关系词与单位反转 | 24→21 | `at least`↔`up to`、`half`↔`double`、**`Round`轮 / `Turn`回合 互换** |
| 正文专名分裂（name 字段之外） | 188→23 | `scan_name_splits` 只比 name 字段，正文从来没判据 |
| 中文排版约定 | 64→13 | **标记功能区混入全角字符是功能性的**，优先级最高 |
| 抽取器没抽的字段 | 91→12 | **2 阻断**，见下 |
| 叶内重复块 | 43→11 | 中文两块相同而英文两块不同＝丢了一整块 |
| enricher 参数↔相邻中文术语 | 52→10 | 玩家点链接看到 lang 的译名、读正文看到另一个 |
| `@UUID`/`@Embed` 死链（含 `#锚点`） | 1277→9 | 判据关键是**英文侧同路径对照**，英文也死＝上游问题 |
| 情态强度（must/should/may） | 45→7 | |
| 列表/表格行列错位 | 36→6 | 地名志 `<dt>` 错位那条阻断的通用化 |
| 否定/条件反转 | 254→3 | 判据噪声极大 |
| 代词性别错配 | 19→3 | 真缺陷全是**英文侧没有代词、中文补出来且补反了** |
| 方位/方向/序数 | 23→3 | |

**主控自己另查出并已修**：`Confused` 4 处（compendium 写「神志混乱」而 crucible lang 与条件页
都是「混乱」）· `Charged Bite` 4 处（英文 storm energy，权威条目是「带电啃咬」）·
`@UUID{标签}` 首尾空格 6 处（英文侧无空格的那些）。新判据 `probes/scan_status_name.py` 已入库。

**落盘**：43 个批次经三方合并（681 条路径、24 条被多组认领、20 条自动合并、2 条人工裁决）
＋ 后续 2 叶修正 = **683 叶**。复验 5.4 全套（含新判据）见下。

**本轮新增/修好九个判据**（都进第 5.4 节）：`scan_attr_text` · `scan_name_binding` ·
`scan_renamed_terms` · `scan_cross_channel` A/C 段重写 · `scan_label_vs_name` ·
`sync_twin_packs` · `scan_bare_english_names` · `scan_token_name` · `scan_name_splits`。

### 四条贯穿全程的方法教训

1. **全绿不等于没问题。** `0.9.2` 发版时 5.4 全套确实每项都是 0，而下一轮仍查出 8 条阻断 ——
   **判据本身有盲区**（详见 3.6）。看到 5.4 全 0 时正确的问题是
   「**哪一类错误现在还没有对应的判据**」，而不是宣布库是干净的。
2. **术语类结论一律要带英文闸的计数。** 裸词频既会漏也会误伤：库内「电能」25 处、「阶位」160 处
   看着像 `Electricity`/`Tier` 的残留，加英文闸才看出它们译的是 `electrical energy` 与 `Rank`。
   工具：`4-临时脚本/2026-08-12-fix/term_gate.py`（三桶 gated_hit / cn_only / en_only）。
3. **「多数派」和「name 字段」都可能是错的那一边。** 标签↔name 的 153 组分歧里，
   **41% 是 name 错**（`Raster Thorn` 的 name「栅格荆棘」是逐词机翻，他是碎牙帮帮主）。
   依据阶梯要用，但用之前先看那个 name 本身译得对不对。
4. **逐句通读也会漏。** 27 个单元读完 35 本 journal 报了 507 条，挂在后面的复核 agent
   **又补出 188 条**（召回率约 73%），而漏的几乎全是**可以机械化**的一类。
   **通读之后必须再跑一遍机械核查**（`scan_label_vs_name` 就是为此写的）。

5. **（第八轮新增）校验脚本自己也可能是错的 —— 而且它错起来是「静默全绿」。**
   主控这一轮亲手交过一个假结论：扫描器用 `glob.glob()` 取路径再 `.replace('/cn/','/en/')`，
   **Windows 下 glob 返回反斜杠**，替换静默失效 → 拿中文文件跟自己比 → 得出
   「97 处标签空格全部忠实于英文」。改用 `os.path.join` 后真实答案是 **6 处真缺陷**。
   这是 §3.6「校验必须复刻被验证系统的真实语义」的另一面：**先证明你的校验器真的在比两份不同的东西**。
   自查手段：任何 en↔cn 对照脚本，都要先断言「读到的两份文件路径不相等且内容不相等」。

6. **（第八轮新增）语义类判据不能停在「同叶共现」，必须做互斥或指代消解。**
   代词性别那一路的第一版按「同叶共现」判，报 160 条**全假**（`brother` 26 : `sister` 26 的
   对称计数一眼就知道是「兄弟姐妹」同叶并列）。假阳性成因永远是同一个：**这片叶里有好几个人**。
   可用的形态是**实体锚定 + 最近先行词**：先用原文自带的 `(NG, Ordani Keth, he/him)` 声明
   （本库有 977 处，比任何推断都硬）给专名定性，再要求代词的最近先行词就是这个人 —— 做到了 0 假阳性。
   同理：中文代词不能裸匹配，裸「他」会扫进 其他/他们/他人/排他/利他，全库 15449 → 清洗后 10699。

7. **（第八轮新增）数值类判据有「中文数词」盲区。** 穷举用阿拉伯数字锚定的判据，
   会漏掉中文写成「六轮」「三层」的那些。第八轮就有一条 `lasts for six turns`→「持续六轮」
   （`Turn` 误作「轮」）是复核 agent 独立重扫才撞出来的，不在探针的 21 条里。

### 落盘方式（多 agent 并行改同一批文件时**必需**，不是可选优化）

`apply_translations.py` 是**整叶覆盖**，而并行批次是同时基于同一份 base 生成的 ——
直接按顺序落盘会让先落的批次被后落的静默回滚（实测一轮里 350 条路径被 1 个以上批次认领）。
必须先做三方合并：以 `compendium/cn` 现值为 base，把每个批次的编辑当 diff 重放。

```powershell
python "$P-临时脚本6-08-12-fix\merge_batches.py" --manifest <manifest.json> --scan
python "$P-临时脚本6-08-12-fix\merge_batches.py" --manifest <manifest.json> --merge --out-dir <merged>
# 真冲突逐条人工裁决，写进 resolutions.json（附理由），再逐包 apply_translations --force
```

manifest 的产法见 `4-临时脚本/2026-08-12-audit3/prep_manifest*.py`；
冲突多时用同目录 `gen_resolutions.py` 自动生成「取超集 + 叠加其余编辑」的裁决草案。

**顺序**：主线（crucible 侧）全部落完之后，**最后**才跑 `qa/sync_twin_packs.py`
—— 主线每落一批就会制造新的孪生分叉。

### ⚑⚑ 第八轮查出的两条 dnd5e 侧管线阻断（**需项目所有者决策，主控没动 `mappings.mjs`**）

**约 85 万字符的 dnd5e 侧内容从未进过管线，而覆盖率报 100%。**
这是盲区表第 9 项（「抽取器根本没抽的字段」）的又一次命中，也再次证明它是本项目最高产的一类。

| # | 缺口 | 实测 | 根因 |
|---|---|---|---|
| 1 | **dnd5e 侧物品正文** | `ember.adventure` 453 个顶层物品 **0 个**有 description；actor 内嵌物品 **0 个字段**。crucible 侧同批对照：顶层 15.9 万字符/479 字段、内嵌 13.1 万字符/848 字段 | `CRUCIBLE_ITEM` 的 `description` 键**按键覆盖**掉了 babele 内建默认 `Item.description = "system.description.value"`（`mergeObject` 里字符串被对象整个顶替），而 `crucibleDescription` 转换器只认 `string` 与 `{public,private}`，dnd5e 的 `{value, chat}` 落不进任何分支 |
| 2 | **dnd5e 侧 NPC 传记主字段** | 只抽到 `biography.public`（258 actor / 19.7 万字符），`.value` 主传记约 **55.7 万字符**零抽取。crucible 侧对照 `biography.private` 51.1 万字符已译完 | `crucibleNested` 的 extract 白名单是 `['name','description','public','private','appearance']`，**不含 `value`**；且 `BABELE_DEFAULTS` 里根本没有 `Actor` 条目，运行时在查一个抽取器永远不产出的键 |

两条都落在**你 2026-08-09 定的「dnd5e 是附带项」那一侧**，所以是不是要补，是优先级问题不是对错问题。
参照先例：`Scene.levels` 那次你定的是「优先完整汉化，补管线」，但那条在**主线**上。
一个缓解因素：内嵌自 dnd5e 官方合集的物品，babele 源包回退**可能**从 dnd5e 中文模块免费取到译文；
Ember 自有物品与 NPC 传记不会。

另有一条 g07 顺带查出的**新管线错**（不在原 finding 里）：`mappings.mjs` 的 `BABELE_DEFAULTS`
自称「与 babele 默认保持同步」，实测与 babele 2.9.1 真默认有三处偏差，其中一处是活的 bug。
详见 `4-临时脚本/2026-08-13-round8/findings/ESCALATIONS.md` 的 g07 段。

**下一步**（按顺序）：

1. **冒烟验证** —— 清单在第 7 节末尾。**只有项目所有者能做**，已积压五轮。
   第八轮又新增两项（`id` 锚点、`Round`/`Turn`），见清单末尾。
2. **裁决上面两条 dnd5e 管线阻断** —— 补 / 不补 / 只补传记。
3. **裁决 13 组升级事项** —— 全表在 `4-临时脚本/2026-08-13-round8/findings/ESCALATIONS.md`
   （22 KB）。最要紧的是「知识领域 7 个术语」三通道分叉：compendium 表格 / `lang/cn.json` /
   `ember-hardcoded-cn.mjs` 三方不一致，且**探针建议的「按 lang 拉齐」被复核推翻**
   —— glossary_ec 有 26 个 `knowledge X` 键，其中 4 条站表格那边；`Undeath` 更有 25 叶同卷散文
   （裁决阶梯第 2 档，压过 lang）站「不死」。方向清楚的只有 `Forensics`。
4. **继续查缺补漏** —— 从本节末的「下一轮从哪开始」表往下走。

**发版状态**：`crucible-cn 0.9.4` / `ember_cn_unofficial v1.1.4`（2026-08-13，**未做冒烟验证**）。
发版后已按第 5.5 节做下载回包核对：manifest 与 zip 实测 HTTP 200；
**crucible 0.30 MB / 24 文件、ember 7.22 MB / 21 文件**；`compendium/en`、`release/`、
`lang/en.json`、`lang_keep_english.json`、`.git`、`.bak`、嵌套 zip 均未混入；
manifest 声明的每个 `esmodules`/`languages`/`styles` 文件包内都在；
包内抽查确认场景层名已中文、`Thayloc Courser` 的 token 名已由占位符改为「赛洛克疾奔兽」、
地名志 `<dt>` 错位已修、`.mjs` 的两条机制修复与 `Scene.levels` 映射都在包里。
更早的 0.9.0–0.9.3 见第 6 节年表。发版怎么做见第 5.5 节；发版前**必做**
`flatten_lang.py`（不加 `--write`）确认两个仓库「拍平前 == 拍平后 == 英文键数」三者相等
—— 这一步就是为了挡住 1.1.0 那次 lang 有 77% 静默失效。

**怎么干活**：一轮 10–12 个并行单元，译者自检闸门 + 对抗式审校 + 跨单元术语核对。
**操作手册是 `PARALLEL-RUNBOOK.md`**；来龙去脉见 `5-其他内容/STAGE-LOG.md` 阶段 20。
八轮实测约 9000 条 / 190 万英文字符落盘**零拒绝**，标记漂移全程只降不升
（LINK 689→0 / BLOCK 584→0 / INLINE 265→0）。

> **别把 BLOCK / INLINE 漂移一概当观感问题。** `class` 属性漂移
> （`section.block gamemaster` / `ul.complex-check` / `sup.system-swap-inline`）是**功能性**的，
> 而标记闸的签名只取标签名、看不见 class —— 用 `qa/scan_class_drift.py` 单独扫。

### ⚑ 优先级与 dnd5e 来源的作用域（2026-08-09 项目所有者定）

**Ember 是世界包，同时支持 Crucible 与 dnd5e 两套规则。**

- **主线是 `crucible` 系统 + Ember 的 crucible 侧**（`2-Crucible汉化插件` 与
  `ember.crucible-adventure`）。这是项目真正要交付的东西，质量与进度都以它为准。
- **dnd5e 侧（`ember.adventure`）是附带项** —— 顺带一起翻，但**不得为它牺牲主线**。

由此得出一条硬性作用域：
**`dnd-simplified-chinese-babele-patch` 之类的 dnd5e 中文来源，只能作用于 dnd5e 侧。**
它对 `crucible.*` 与 `ember.crucible-adventure` **没有任何权威**。
这不是保守，是规则体系不同：拿模块比对我们已译的 name，752 条不一致里有 **542 条**落在
Crucible 侧 —— 那些英文只是碰巧同名（`Dagger`/`Longbow`），条目却是 Crucible 自己的。
整包照搬会把主线污染掉。`tm/fill_twin_names.py` 因此只填 `ember.adventure` 的**空槽**，
从不覆盖既有译文。

### ⚑ 术语与前后不一致：由主控自行裁决，不要来问（2026-08-09 项目所有者定）

**任何上下文 / 前后译文不一致的情况，反复核对译文、词义、上下文之后自行统一即可，不必上报。**
这条覆盖此前「不静默择一、列进 disputes 待裁决」的做法 —— 那条现在只适用于**证据真的不足**时。

裁决时仍走既定的依据阶梯（强 → 弱）：
**同名条目/物品的 `name` 字段 > 同卷已译页 > 全库多数写法 > `glossary_ec.json` > `BRIEF.md` 的表**。
另外三条实测出来的注意事项：

- **先查英文再判中文**。中文写法不同不等于错 —— 英文本来就不同的场合很多。
  例：库里 167 处「闪电」，其中 123 处英文确实是 `Lightning`（忠实），44 处是「闪电般迅捷」这类比喻，
  真正错的只有**英文写 `Electricity` 却译成「闪电」的 23 处**。不做这一步会误伤大片。
- **改动面小的那边优先**（阶段 20 的 `Inkaro`：改 4 个物品条目名而不是 126 处正文）。
- **`name` 字段与正文冲突时，多数情况该改 `name`**，因为正文改动面通常大得多；
  但要连 `lang/cn.json` 一起看 —— 阶段 23 的教训是 crucible 自己的 lang 与 compendium 条目名就不一致。

裁决后**必须**：① 写进第 8 节决议记录（附证据与计数）；② 用 `qa/unify_terms.py` 执行
（它只在**英文原文确实出现该术语**时才改，正是上面第一条的机械保障）；③ 复跑 QA 全套。

**翻译时必须遵守的既定译名**（避免和已完成的 11 个包冲突）：
`Kinesis`念力 · `Warden`守林者 · `Guardian`守护者 · `Swarm`(archetype)群集 · `Tier`阶 ·
`Electricity`**电击**（状态 `Shocked` 作**感电**；`Lightning` 才是闪电，别混）·
`Bludgeoning`钝击 · `Fire`火焰 · `Corruption`腐化 · `Fortitude`**强韧** ·
`Toughness`坚韧 · `Wisdom`感知 · `Presence`存在 · `Willpower`**意志** · `Health`生命值 ·
`Boon`**恩惠骰**（对 `Bane`祸骰）· `Accurate`精准 · `Stride`步幅 · `Arrow`箭矢 ·
`inflection`屈折 · `gesture`手势 · `rune`符文 · `spellcraft`施法 · `essence`精华 · `compose spells`构筑法术。
完整表见 `5-其他内容/glossary/glossary_ec.json`。

**要做全局验证就跑第 5.4 节那一整套**（每一项都应为 0）。翻译批次的做法在
`PARALLEL-RUNBOOK.md`；回写一律走 `qa/apply_translations.py`（三道闸：英文源漂移 /
无中文 / 标记破损），**任何情况下都不要直接改 `compendium/cn`**。

一条容易踩的：追平类批次改的是**已有中文**的条目，落库必须带 `--force`，
否则是 `applied 0 / skipped(existing) N` 的静默空跑。

改过 `3-常用脚本/extract/mappings.mjs` 或 `release/runtime-converters.js` 之后，必须重跑：
```powershell
node "$P\3-常用脚本\release\generate_runtime.mjs"      # 重新生成两个仓库的 babele-mappings.js
python "$P\4-临时脚本\2026-08-06\crosscheck_vs_crucible_fr.py"   # 交叉校验抽取器
```

**关键路径**：
```
抽英文基准 → 算 diff → 管线改造(声明式 mapping) → TM/回源预填 → 分批翻译 → 三轮校准 → 发版
```

**三个必须知道的事实**（不知道会做错方向）：

1. **babele 2.9.1 会自动回源翻译内嵌文档**。Adventure 里 actor 内嵌的 item，只要带 `_stats.compendiumSource` / `flags.core.sourceId`，babele 就会去它原属合集的译文里取。ember 战役里 82.4% 的内嵌物品字符有来源包 —— 只要用**默认递归 `document` mapping**（而不是手写遍历转换器），这部分自动就翻好了。**不要退回手写转换器**，那会白白丢掉约 60 万字符的免费收益。
2. **英文基准必须存档**。每次系统/模块升级，只有拿旧版英文才能算出"哪些英文原文改了"（drift）。基准存两处：各插件仓库 `compendium/en/`（当前版本，进 git）＋ `5-其他内容/english-baseline/<包>-<版本>/`（历史快照）。
3. **术语表是 `5-其他内容/glossary/glossary_ec.json`**，基底来自 `glossary_crucible_merged.json`（4602 条已裁决译名）。**不要另起炉灶**，也**不要并入 PF2E 主表**（`fvtt\glossary.json`）—— Crucible/Ember 是 Foundry 自有世界观，PF2 的译法会污染（例：`Restrained` 本项目作「受缚」，PF2 作「受制」）。

---

### ⚑ 新会话「查缺补漏」入口（想找现有检查覆盖不到的东西，从这里开始）

**先接受一个前提**：第 5.4 节全绿，只说明「已知的那几类错误不存在」。
阶段 28 的 8 条阻断全部出自 5.4 报 0 的库。所以正确的问题不是「还有没有错」，
而是**「哪一类错误至今没有对应的判据」**。已知的、按价值排序：

| # | 覆盖不到的东西 | 为什么现有检查看不见 | 状态 / 怎么下手 |
|---|---|---|---|
| 1 | **运行时行为**（译文写对了但玩家看不到 / 数据被写坏） | 全部检查都是静态比对 JSON，不加载 Foundry | ⬜ **仍是最高优先级**。**冒烟验证**，第 7 节末尾清单。terrain / 词缀拼装名 / `_packs-folders` 已随 0.9.3 发出去；第三轮又加了 `.mjs` 的两条机制修复（`data-tooltip-text` / DialogV2 标题）与 P-9 待决补丁 |
| 2 | **35 本 journal 的逐句准确性**（占 journal 正文 51%） | 覆盖率只看「有没有中文」，标记闸只看标记，长度比只是启发式 —— 已知的两条凭空增删里有一条比值 0.377 完全正常 | ✅ **第四轮做完**：27 个并行单元实读 458 万英文字符，507 条确认缺陷（2 阻断）。**但复核 agent 又补出 188 条**，说明单次通读召回率约 73% —— 见第 1 节教训 4 |
| 3 | **同一目标的标签命名不一致 226 处** | `scan_uuid_swap` 的 UNCERTAIN 档：不是错位（BROKEN 已 0），是同一文档在不同处用了不同中文标签 | ✅ 第三轮逐条裁完。**方法教训**：`findings` 里的 `en_label` **不可信**（它只取该目标在叶内的第一个英文标签，而同一叶对同一目标用两个英文标签是常态），必须逐位对齐取本处的真实英文标签，否则会把正确译文改坏 |
| 4 | **lang ↔ compendium 的术语一致性** | `scan_cross_channel.py` 的 A/C 段词对齐是坏的（见 3.6） | ✅ A/C 段已重写：候选来源从 n-gram 滑动窗口换成**叶级对照**（整叶等值 / name 字段 / `@UUID` 标签配对 / lang / glossary / `.mjs` / 双语并列锚），四道过滤。回测 13/13：三个坏例消失，且把 lang 换回历史错值能重新报出 |
| 5 | **HTML 属性里的可见文本** | 唯一触及属性的检查用多重集相等，结构上只能发现「译坏」不能发现「根本没译」 | ✅ 新增 `qa/scan_attr_text.py`。本库实际只出现 26 个属性名，**任务当初凭空拟的 `title`/`alt`/`aria-label` 一个都没有**，真正有的是 `@Embed[...]` 体内的 `label=` 与 `readaloud=` —— 先普查再定白名单。ground truth 122 处，修前已译 50（41%），现 **122/122** |
| 6 | **RollTable 结果名 / 场景针脚 与目标文档名是否同名** | `scan_markup_targets` 只管 `@UUID` 方括号，管不到用 `entryId` 绑定、标签独立存放的 Note 与 table result | ✅ 新增 `qa/scan_name_binding.py` + `qa/dump_bindings.mjs`。**关键发现：`entryId`/`documentUuid` 根本不在 babele 译文文件里，只在 LevelDB packs 里**，所以旧探针只能拿「英文标签逐字节等于某文档英文名」当代理判据（实测多报 10 / 漏报 3）。新闸从 packs 导出真实绑定。570 针脚 + 697 表结果，BROKEN 由 16 → **0** |
| 7 | **`ember-hardcoded-cn.mjs` 的中文质量** | `qa/` 下只有 `scan_cross_channel.py` 的 B 段按键比对它，**没有人审过它的中文本身** | ✅ 177 条全部逐条审完。**最重要的结论不是译文错，而是 31 个键在当前 Foundry v14 + ember 0.6.x 上根本不生效**（详见第 8 节三条阻断） |
| 8 | **英文变过、中文没跟上**（跨版本 drift） | 只要上游一升级就立刻回来 | ✅ 第三轮全量复核了 `stale` 桶 278 条。**但要知道它的假阳性率极高**（见第 1 节）。**命门仍是先把当前英文归档到 `english-baseline/`** |
| **9** | **抽取器根本没抽的字段** | 所有静态检查都以「英文基准里有这一条」为起点；**基准里压根没有的字段不在任何检查的定义域内**，与 08-12 的「中文侧整条不存在」是同一类盲区的另一侧 | ✅ 2026-08-13b 补齐 Scene 的三个字段（`levels`/`tokens`/`navName`，见第 7 节第 13 项）。**发现手段值得复用：外部第二实现对照** —— 是拿 crucible-fr 的 `compendium/en` 与我们逐路径比才看见的。**下次上游升级后应重跑这个对照**，它是唯一能发现「我们没抽的字段」的手段 |

---

#### 📌 下一轮从哪开始（2026-08-13b 收尾时的实况）

> ### ✅ 第十二轮之后的收口状态（新会话从这里判断「还剩什么」）
>
> **所有判据当前非零的只剩两项，且都已归档为永久豁免**：
> - `scan_same_en_split` **14 组 / 133 叶** —— 全部在
>   `4-临时脚本/2026-08-13-round12/findings/EXCLUSIONS.json` 里逐条带证据
>   （`Shield` 法术/装备、`Light` 戏法/负重分级、`Water`·`Ooze` 地形/生物分类、
>   `West`·`East` 地点分区/方向表、`Aura` 天体/手势、`Adelyne` 上游硬编码同名等）
> - `scan_label_vs_name` **2 处** —— `Maziran` 马兹兰人 vs 马兹兰，已核实是有意保留的族称
>
> 其余全部为 0：标记五项 / 方括号内标记 / class 漂移 / 数字覆盖 / 外来文字 / **死键（两仓均 0）** /
> 中文侧缺键 / `tokenName` / 状态名 / HTML 属性 / 裸英文专名 / **锚点缺口** / 孪生包分叉 /
> `uuid_swap` BROKEN / `name_binding` BROKEN / lang 四项且拍平三数相等（486 / 1842）。
>
> **75 本 journal 全部逐句对读过**，其中 7 本高产的又做过一轮独立重读。
>
> ⚠ **仍未做的只有两件**：① 真实世界验证（第 7 节清单，项目所有者定「打磨完再统一验证」）；
> ② dnd5e 侧两条管线阻断约 85 万字符（缺陷表 Z1，项目所有者定「先不管」）。

**已经有判据、且当前为 0 的**（跑一遍确认即可，别重复投入）：
5.4 全套十四项、`scan_label_vs_name`（剩 2 处是有意保留的族称 `Maziran`＝马兹兰人）、
`scan_name_binding` BROKEN、`scan_renamed_terms`、`sync_twin_packs`、
`scan_attr_text` 的属性名与 enricher 标签、**`scan_status_name`（新，371 一致 / 0 不一致）**、
`scan_token_name`、`scan_name_splits`（剩 5 个已逐个查实为合法分裂，别再裁）。

> ✅ **第八轮那两个「不是 0 但也不要动」的，第九轮已经从判据层解决**（不要再按旧注记理解）：
> ① `scan_attr_text` 的 113 处 `gained {id}` —— 判据加了 **`id` 净增加白名单**（净减少仍报），现为 **0**。
>    `--strict-id` 可关掉白名单复现 113，证明只是不报、没有丢数据。
> ② `scan_uuid_swap` UNCERTAIN 68 —— 判据改为**逐位对齐**（同一 target 按出现序号配对英中标签），
>    现为 **6**。⚠ 代价记在 §8：这一改**把一处真缺陷一起消掉了**（`Sunalins` 三分），
>    幸而被同轮的同源串统稿覆盖到。**改判据降噪时必须逐条核对消失的那些里有没有真缺陷。**

**第八、九轮已经排除掉的方向**（做过了，已清零，别重做）：
`bare_english` **37 → 0**。第八轮误判为「全是假阳性」，**第九轮证明这个判断是错的** ——
三类假阳性（双语并列语序倒装 `沙纳山脉南部 Southern Shana Mountains` · 出版商名 `Mage Hand Press` ·
enricher 方括号内 `&Reference[Difficult Terrain]`）共 17 处已由判据排掉，
**剩下的 20 处是真缺陷**（Patch 页的本作内容名），已修。灵敏度回测 13/13。

> ⚠ **但闸归 0 ≠ 那几页干净**（方法教训 1 的又一实例）：Patch `0.4.2/0.4.3/0.4.7` 上仍有十几处
> **单词**专名裸英文（`Ancara` / `Ortarec Cube` / `Magical Forces` / `Cartographer` / `First Soulmark` …），
> `--min-words 2` **结构性地看不见**；0.4.x 之外还有十几个 Patch 页没人看过。见下表 Z4。

**还没有判据 / 判据有覆盖洞的**（按预估价值排序）：

| # | 方向 | 现状与下手方式 |
|---|---|---|
| **Z1** | **dnd5e 侧两条管线阻断** | 🔴 **第八轮新查出，最高优先级的「有内容缺口」项**。约 85 万字符。详见第 1 节 ⚑⚑ 那一段。**需项目所有者先裁「补不补」** |
| **Z2** | **13 组升级事项** | 🔴 全表 `4-临时脚本/2026-08-13-round8/findings/ESCALATIONS.md`。最要紧：知识领域 7 词三通道分叉 · `Marlstone` 复合专名词根（~130 叶）· `Hallows` 组织/城区共不共用词根（~40 处）· `Mazira` 音译二分（马齐拉 17 : 马兹拉 14）· 手势模板 61 份同英文副本各译各的 |
| ~~Z3~~ | ~~英文基准里的上游笔误挡住回写~~ | ✅ **第九轮已修**。`Yakoshta Mine / Elevator` 的 `[[/skillCheck athletics 15 check.` 补上 `]]`，已记为 `LOCAL-PATCHES.md` 第 3 条，中文补译完成。**惯例已确认：补丁只打 `compendium/en`，`english-baseline/` 快照保持上游原样** |
| **Z4** | **Patch 页的单词专名裸英文** | 🔶 十几处 `Ancara` / `Ortarec Cube` / `Magical Forces` / `Cartographer` / `First Soulmark` 等，`scan_bare_english_names --min-words 2` **结构性看不见**（单词专名）。0.4.x 之外还有十几个 Patch 页从没人看过。需要一条独立单元，或把闸的 `--min-words` 降到 1 并配一份白名单 |
| **Z5** | **同源串分叉的剩余 470 组 / 2561 叶** | 🔶 第九轮从 1514 组压到 470 组。剩下的 **429 组是 ≤20 字符短标签**（`Overview` 一组就 165 叶），agent 按「短串默认合法分叉」保守留下的。要收口得逐条查语境，性价比中等。长正文那一档（>300 字符）**已经是 0** |
| **A** | ~~中文正文里的裸英文专名~~ | ✅ **第九轮真正收口**：37 → **0**。第八轮判「全是假阳性」是**错的**，其中 20 处是真缺陷（见上）|闸 `qa/scan_bare_english_names.py --min-words 2`（词典 3255 条），报告 `5-其他内容/reports/bare_english.json`。剩下的 63 处是四单元一致 deferred 的 Patch 页条目与假阳性。**闸的已知假阳性模式**：`Mage Hand` 命中的其实是出版商 `Mage Hand Press` —— 交替正则里没有更长的那个名字，短的赢了。修法：把「命中后紧跟大写词」也排除 |
| B | `scan_en_drift` 的 `changed` 非 stale 桶 | 1216 条至今只做过抽查。**但它已被机械闸扫过一遍**（LINK / BLOCK / 数字覆盖 / class / 改名残留都是全库判据且为 0），残余风险只剩「结构与标记都没变的纯散文改写」 |
| C | 剩下约 40 本 journal | 设定集 / 地名志 / 规则页，前几轮读过但**不是逐句**。地名志那一批已被证明有系统性缺陷（70 处 NPC 名整个没译、一条 `<dt>` 错位阻断） |
| D | `scan_uuid_swap` 的 UNCERTAIN | ✅ 226 → 98 → **22**。剩下的是英文标签本来就与文档名不同的（换称呼／代词式简称／`#锚点` 指小节），属 by-design |
| E | `scan_label_vs_name` 自己的覆盖洞 | ✅ **2026-08-13e 已补**。对齐改成「按同一 target 的出现序号」，跳过数 **120 → 20**，捞回 7 条真缺陷；统计口径也修了 —— 原本「目标 id 不在表里 1969」看着像巨大覆盖洞，实际 **1697 条是 dnd5e 系统自己的合集引用**（本项目不负责），真覆盖洞只有 270 |
| F | `scan_name_binding` 的 UNCERTAIN 824 | 目标没有中文 name 的那一档。多数是合理的（目标本来就没条目），但值得抽样确认 |
| **H** | **同一个英文 `name` 有两套中文名** | ✅ 新闸 `qa/scan_name_splits.py`：5449 个英文名里 54 个分裂，已裁到 **7**。剩下的 7 个是**合法分裂**（`Shield` 既是法术护盾术又是装备盾牌、`Luminous`/`Spirited` 分属两套永不同载的规则），判据看不出词义，必须人排除 |

---

## 2. 项目总结

把 Foundry VTT 的 **Crucible 系统**和 **Ember 战役模块**汉化成简体中文，通过两个 Babele 翻译模块交付。

### 版本矩阵

| 组件 | 类型 | 版本 | 位置 |
|---|---|---|---|
| crucible | 系统 | **0.10.1** | `%LOCALAPPDATA%\FoundryVTT\Data\systems\crucible` |
| ember | 模块（**付费/protected**） | **0.6.0** | `…\Data\modules\ember` |
| babele | 模块（翻译框架） | **2.9.1** | `…\Data\modules\babele` |
| crucible-cn | 汉化模块（本项目） | **0.9.7**（2026-08-14 发布） | `2-Crucible汉化插件\` |
| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.10**（2026-08-14 发布；v1.1.8/v1.1.9 为阻断急修） | `1-Ember汉化插件\` |

两个汉化仓库：
- https://github.com/takaqiao/crucible-cn
- https://github.com/takaqiao/ember_cn_unofficial

### 汉化的两条通道

- **`lang/cn.json`** —— 界面字符串（Foundry 原生 i18n），走 module.json 的 `languages` 字段
- **Babele `compendium/cn/*.json`** —— 合集内容（天赋/装备/日志/战役正文），走 `babele.register()`

crucible 侧两条都用；ember 侧两条也都用。

---

## 3. 须知（踩过的坑与硬约束）

### 3.1 babele 2.9.1 的三个真实故障

1. **`crucible-cn/babele-register.js` 有真 bug**
   `adventure_items_converter` 内部调 `game.babele.converters.actions_converter(items, translations)`。
   2.9.1 的 `.converters` getter 返回的是 `ConverterRegistry.snapshot()` —— 值是 **`FunctionalConverter` 对象**，不是函数。
   → 冒险模组里带 `actions` 的内嵌物品会抛 `TypeError: not a function`。

2. **`crucible-cn` 的 `SUPPORTED_PACKS` / `DEFAULT_MAPPINGS.ActiveEffect` 补丁已是死代码**
   2.9.1 的默认 mapping 原生支持 `ActiveEffect`（还带 `changes` 的 `structured` 转换器）。该补丁块可删。

3. **`ember_cn/register.js` 的 `_tableResults` 补丁全是死代码**
   2.9.1 已无 `_tableResults` 转换器，改由 `document` + `documentType: "TableResult"` 处理，identity 是
   `_identity: {export: ["range","_id"], match: ["_id","range"]}`。
   这正是 table results 至今 0% 的原因。

### 3.2 babele 2.9.1 的关键能力（务必用上）

- **注册钩子是 `babele.init`**，不是 Foundry 的 `init`。babele 在自己的 `init` 里 `game.babele = …` 然后同步 `Hooks.callAll('babele.init')`。现在两个模块都挂在 Foundry `init` 上，靠模块加载顺序侥幸能跑 —— 要迁走。
- **`window.Babele` 仍然存在**（`babele.js` 末尾 `window.Babele = BabeleFacade`），旧的 `typeof Babele !== 'undefined'` 守卫不会失效。
- **`registerMapping(mapping)`** —— 声明式追加/覆盖全局文档 mapping 层，比手写转换器好得多。
- **`_variants` + `_when`** —— 按字段值分支 mapping，正好对应 ember 的 13 种 page 子类型：
  ```json
  "_variants": [{ "_when": {"path": "type", "equals": "ember.location"}, "overview": "system.overview" }]
  ```
- **递归 `document` 转换器 + 源包回退**（见第 1 节事实 1）。`fallbackPolicy` 可选
  `source-first`（默认）/ `owner-package-before-generic` / `owner-package-first`。
- **`_packs-folders.json`** —— 能翻合集**文件夹名**。crucible-cn 完全没做，Crucible-FR 做了。
- **多源优先级** —— 同一个 collection 有多份译文文件时可设优先级（`setSourcePriority`）。
- **诊断接口** —— `game.babele.inspectMapping(type)`、`await game.babele.sourceDiagnostics()`、`cacheDiagnostics()`。验收时用。
- **损坏的译文文件不会拖垮整体**：`TranslationLoader.#loadJsonFile` 有 try/catch，只 `console.error` 并跳过该 collection。

### 3.3 ember 的自定义 JournalEntryPage 子类型

ember 的正文**不在 `text.content` 里**。13 种子类型及其正文字段（实测自 ember 0.6.0）：

| 子类型 | 页数 | 正文字段 |
|---|---|---|
| `ember.location` | 115 | `system.overview`、`system.exposition`、`system.terrain` |
| `ember.biome` | 27 | `system.overview`、`system.exposition`、`system.terrain` |
| `ember.lore` | 167 | `system.content.overview`、`system.content.gamemaster`、`system.pronunciation` |
| `ember.deity` | 74 | `system.content.overview`、`system.content.gamemaster`、`system.subtitle`、`system.pronunciation` |
| `ember.questEvent` | 229 | `system.overview`、`system.exposition`、`system.summary`、`system.outcomes[].label\|summary` |
| `ember.standaloneEvent` | 18 | 同上 |
| `ember.culture` | 28 | `system.content.overview`、`system.banner.caption`、`system.pronunciation` |
| `ember.ancestry` | 18 | `system.content.overview`、`system.height`、`system.lifespan`、`system.origin`、`system.pronunciation` |
| `ember.cosmos` | 11 | `system.content.overview`、`system.content.gamemaster`、`system.subtitle` |
| `ember.organization` | 21 | `system.content.overview`、`system.content.gamemaster`、`system.pronunciation` |
| `ember.characterClass` | 13 | `system.content.overview` |
| `ember.quest` | 20 | `system.overview` |
| `ember.questFlowchart` | 20 | 仅 `name` |
| `text`（原生） | 727 | `text.content` |

实测 `system.overview` 与 `text.content` **741 处全部不同**，不是冗余镜像 —— 两者都要翻。

### 3.4 译名规范

- **专有名词用双语并列**：`申特月神殿 Shent Moon Temple`、`古冢 Barrows`。
  这是既有 v1.0.15 已在用的风格，保持一致。
- **术语优先级**：本项目 glossary > 从既有译文提取的 TM > PF2E 主表（仅作建议，命中须人工确认）。
- **HTML/富文本标记必须原样保留**：`@UUID[...]`、`@Check[...]`、`<section class="block gamemaster">`、
  `<span class="reference">⬢ s.3204.2870</span>`、`<figure>/<figcaption>` 等。
  这些是 Foundry 的功能性标记，改坏了会导致链接失效或样式崩。

### 3.6 ⚑ 检查器自己的盲区（2026-08-12 第二轮查出并已修）

**`0.9.2 / 1.1.2` 发版时 5.4 全套确实每项都是 0，而全库仍有 8 条阻断缺陷。**
不是漏跑，是**判据本身看不见那些错误**。四个已修的盲区：

| 脚本 | 原判据 | 看不见什么 | 现状 |
|---|---|---|---|
| `scan_content_coverage.py` | 数字用 **`set()`** 比对（脚本自注「不按次数」） | 同页别处有同一个数字时，`3 Talent Points`→「2点天赋点」完全沉默 | 改**多重集**；配三条宽容规则（块内邻近折叠 / 单双 / 斜杠数对），否则假警报暴涨 |
| `fill_missing.py` `prune_dead.py` | walk 只处理 dict 与 str | **不遍历数组**，数组内叶子整体不在定义域。第 5.4 节第 6 项被称作「其它检查都覆盖不到的方向」，它自己也有覆盖不到的方向 | 都已下钻 list。`prune_dead` 的 `prune()` 只做**尾部截断** —— 删数组中间元素会让后面所有元素索引前移，等于把生效译文整体挪到别人的键上 |
| （无）@UUID 标签/目标错位 | — | 每叶的目标多重集与标签多重集都与英文**相等**，标记五项 / class / `[[ ]]` 全过；中文散文本身通顺，人工通读也发现不了 | 新增 `qa/scan_uuid_swap.py` |
| （无）跨通道一致性 | — | lang / compendium / `.mjs` 三条通道互不校验，玩家却同屏看到 | 新增 `qa/scan_cross_channel.py`（**仅 B 段可信，见下**） |

> ✅ **A/C 段已于 2026-08-13 重写**（下面这段警告保留为病理记录，判据本身已经换掉）。
> 新做法：候选来源从 n-gram 滑动窗口换成**叶级对照**（整叶等值 / name 字段 / `@UUID` 标签配对 /
> lang / glossary / `.mjs` / 双语并列锚），再加互斥度、特异度 lift、通用词 df 上限、归属反查四道过滤。
> 回测 13/13：三个坏例连同「爆炸瓶 / 裂的弧形 / 尺锥 / 奔涌」全部消失；
> **灵敏度也测了** —— 把 lang 换回历史错值（扇子/跨步/庄园/莽撞/样貌/病房）全部重新报出且给出正确对家
> （只测特异度的话「全判 AGREE」也能过，这一半是防自己交假修复的）。硬判据由 197 条降到 31 条。
>
> ⚠ **以下是重写前的病理（留档）：A 段与 C 段曾经不可信。**
> B 段（`.mjs` ↔ lang，按键精确比对）是可靠的，实测抓到 4 条真漂移。
> 但 A/C 段用滑动窗口从 compendium 里取「对应中文」，取出来的是 n-gram 碎片而非词 ——
> 实测 `Ward` 的 compendium 多数被报成「神殿区」、`Aspect` 报成「会替」、`Shoddy` 报成「品质」
> （这是「粗糙品质」的后两字）。**要用它做 A/C 段结论，得先把词对齐换成真正的术语抽取。**
> 它输出的 `lang_support`（lang 值在英文闸命中行里的出现次数）是可信的，可以单看这一列。

**由此得到一条通则**：**校验必须复刻被验证系统的真实语义，而且"全绿"只证明已知错误类型不存在。**
新会话看到 5.4 全 0 时，正确的下一步是问「哪一类错误现在还没有对应的判据」，
而不是宣布库是干净的。第 7 节的覆盖盲区清单就是回答这个问题用的。

### 3.5 其他约束

- **ember 是付费模块**（`module.json` 里 `"protected": true`）。`ember_cn_unofficial` 公开仓库放着整部战役的完整中文正文。
  Padhiver 的 Crucible-FR 有 Ember 转换器但**没有公开发布 Ember 译文**，大概率就是这个原因。是否改私有/只发 diff 由项目所有者决定，此处仅记录。
- **`ember_cn` v1.0.15 发布包里 `ember.crucible-adventure-en.json` 是损坏 JSON**（第 44 行多一个 `}`）。
  它躺在 `compendium/cn/` 里，每次开世界都会被 fetch + 解析失败（11.4 MB 白流量）。要移出到 `compendium/en/` 并重新生成。
- **`ember_cn` 的译文里有 1447 个页面条目带垃圾字段 `path: {}`** —— 是 mapping 保留字 `path` 被当成翻译字段写进去了，要清掉。
- Windows 目录名不能含 `/`，所以项目目录用连字符：`Ember-Crucible Translation Project`。

---

## 4. 目录与脚本索引

```
Ember-Crucible Translation Project\
├── PROJECT.md                  ← 本文件
├── 1-Ember汉化插件\            ← ember_cn_unofficial 的 git clone（可直接发版）
├── 2-Crucible汉化插件\         ← crucible-cn 的 git clone（可直接发版）
├── 3-常用脚本\
│   ├── extract\   从 LevelDB packs 抽英文原版
│   ├── tm\        翻译记忆库构建 / 预填 / 去重
│   ├── qa\        术语校验 / markup 完整性 / 残留英文 / 覆盖率
│   └── release\   打包 zip / 改 module.json / 发 GitHub release
├── 4-临时脚本\<日期>\          ← 一次性探针，按日期归档，不删（可复现结论）
└── 5-其他内容\
    ├── glossary\          glossary_ec.json + 来源谱系 + 冲突裁决
    ├── english-baseline\  历史版本英文快照（跨版本算 drift）
    ├── reports\           每阶段 diff / QA 报告
    └── reference\         babele 2.9.1 API 要点、Crucible-FR 参考实现摘录
```

### 脚本索引

`$P` = 项目根目录；`<repo>` = `1-Ember汉化插件` 或 `2-Crucible汉化插件`。

| 脚本 | 干什么 | 怎么调 |
|---|---|---|
| `extract/mappings.mjs` | **mapping 数据的唯一真源**。抽取器解释它，运行时文件由它生成。改了它必须重跑下面两条 | 不直接执行 |
| `extract/extract_en.mjs` | 解释 mapping，从 LevelDB packs 抽英文基准 | `node extract_en.mjs --package <foundry包目录> --out <输出目录> [--target crucible\|ember] [--pack <名>]` |
| `release/runtime-converters.js` | 三个自定义转换器的**翻译方向**实现（抽取方向在 `extract_en.mjs` 里）。两边必须同一次提交一起改 | 不直接执行 |
| `release/generate_runtime.mjs` | 由上面两个文件**生成**两个仓库的 `babele-mappings.js` | `node generate_runtime.mjs` |
| `tm/build_glossary.py` | 合成 `glossary_ec.json`，并产出待裁决 / 待补清单 | `python build_glossary.py` |
| `qa/validate_translations.py` | **核心验收**：拿英文基准逐路径核对译文，输出覆盖率 + 机读待译清单 | `python validate_translations.py --repo <repo> --out <报告目录>` |
| `qa/lang_gap.py` | `lang/cn.json` 的三方 diff：NEW / DRIFT / UNTRANSLATED / STALE | `python lang_gap.py --repo <repo> --package <foundry包> --out <报告目录> [--sync-baseline]` |
| `qa/apply_lang.py` | lang 批次回写。四道闸：key 不存在 / 占位符 / HTML 标签 / 行内标记；`--clean-stale` 清上游已删的 key | `python apply_lang.py --repo <repo> --package <foundry包> --batch <batch.json> [--clean-stale] [--dry]` |
| `qa/unify_terms.py` | 按规则表统一术语。只在**英文原文确实出现该术语**时才改，支持正则搭配限定 | `python unify_terms.py --repo <repo> [--package <foundry包>] --rules <rules.json> [--review <md>] [--write]` |
| `qa/scan_markup_drift.py` | 扫译文与英文的标记差异：LINK / BLOCK / INLINE / PLACEHOLDER / TRUNCATED | `python scan_markup_drift.py --repo <repo> [--kind LINK,TRUNCATED] [--out <json>]` |
| `qa/scan_markup_targets.py` | 扫**方括号内部**被译成中文的标记（链接/嵌入块会静默失效，覆盖率与漂移检查都看不见）。分 BROKEN / by-design 两类 | `python scan_markup_targets.py --repo <repo> [--repo <另一个>] [--json <out>]` |
| `qa/restore_enrichers.py` | 把被写成裸中文的 `@Condition[...]` 等标记还原回去 | `python restore_enrichers.py --repo <repo> --package <foundry包> --surface-forms <json> [--write]` |
| `qa/resolve_generic_fallback.py` | 从待译数字里**扣掉 babele 会自动解析的部分**。翻译前必跑，否则重复劳动 | `python resolve_generic_fallback.py --repo <repo>` |
| `qa/apply_translations.py` | 批量回写译文。三道闸：英文源漂移 / 无中文 / 标记破损 一律拒 | `python apply_translations.py --repo <repo> --pack <pack.json> --batch <batch.json> [--force] [--dry]` |
| `qa/scan_foreign_script.py` | 扫外来文字污染（西里尔 / 亚美尼亚 / 希伯来 / 泰文等机翻残留） | `python scan_foreign_script.py --repo <repo> [--repo <另一个>] [--fix]` |
| `qa/port_orphans.py` | 上游改名后把孤儿译文移植到新路径；移植不了的**留在原地并列出** | `python port_orphans.py --repo <repo> --rules <rules.json> [--dry]` |
| `qa/migrate_cn_schema.mjs` | 一次性 schema 迁移（已执行完毕，保留备查） | `node migrate_cn_schema.mjs --repo <repo> --package <foundry包> --target <crucible\|ember> [--dry]` |
| `qa/flatten_lang.py` | 把 `lang/cn.json` 拍平成 Foundry 真查得到的扁平点号键，并按英文侧逐键复核。**含 `foundry_lookup()`（复刻 `getProperty`）** | `python flatten_lang.py --repo <repo> --english <英文 lang 文件> [--write]` |
| `qa/scan_content_coverage.py` | 靠「跨语言不变量」（英文正文里的数字）找中文没跟上的条目。**认中文数字与 decade/dozen 这类倍数量词**，否则会逼出「2 个十年」那种坏中文 | `python scan_content_coverage.py --repo <repo> [--out <json>]` |
| `qa/fix_bold_drift.py` | 加粗漂移的机械修复（阶段 20 用过，保留备查） | `python fix_bold_drift.py --repo <repo> [--write]` |
| `tm/fill_twin.py` | crucible-adventure → ember.adventure 单向 TM 填充（两套规则、同一场战役，日志正文逐字节相同） | `python fill_twin.py [--out <batch.json>] [--report <json>]` |
| `qa/prune_dead.py` | 删中文包里英文包已没有的键（babele 永远查不到）。顺带揪出**键名混进中文**的条目 | `python prune_dead.py --repo <repo> [--write]` |
| `qa/propagate_fix.py` | 把某次提交里修好的译文推到**英文逐字相同**的同源副本上。三方 diff 挑不出那些副本（它们的英文没变过） | `python propagate_fix.py --repo <repo> --english <英文包目录> --since <commit> [--write]` |
| `tm/fill_missing.py` | 用全库 TM 补**中文侧整条不存在的键**。这类缺口所有既有扫描都发现不了。**2026-08-12 修了 walk 不下钻数组的盲区** | `python fill_missing.py --repo <repo> [--repo <另一个>] --out-dir <批次目录> [--report <json>]` |
| `qa/scan_uuid_swap.py` | **新（2026-08-12）**。查 `@UUID[目标]{标签}` 的标签挂错目标（链接指向错误文档）。判据：目标的中文标签 ≠ 全库多数标签，且该标签是同叶另一目标的多数标签。分 BROKEN / UNCERTAIN | `python scan_uuid_swap.py --repo <repo> [--repo <另一个>] [--out <json>]` |
| `qa/scan_cross_channel.py` | 2026-08-12 新增，**A/C 段 2026-08-13 重写**。lang ↔ compendium ↔ `ember-hardcoded-cn.mjs` 三通道一致性。三档输出：硬判据 / `LANG_WEAK_SIGNAL` / `LANG_NO_SUPPORT`（后两档是「要人看」不是「有错」） | `python scan_cross_channel.py --repo <repo> --package <包> --mjs <文件> --out <json>` |
| `4-临时脚本/2026-08-12-fix/merge_batches.py` | **并行批次的三方合并**。多 agent 同时基于同一 base 产出的批次会互相整叶覆盖；它以现值为 base 重放各批次的编辑，冲突落 `resolutions.json` 人工裁决 | `python merge_batches.py --scan` / `--merge` |
| `4-临时脚本/2026-08-12-fix/term_gate.py` | **英文闸术语计数**，三桶 gated_hit / cn_only（英文是别的词，多半不是残留）/ en_only（英文是该词但中文换了译法）。术语类结论的机械保障 | `python term_gate.py --repo <repo> --en "\bBoon\b" --cn "恩惠骰,恩惠"` |
| `4-临时脚本/2026-08-12-fix/pair_dump.py` | 成对导出英中叶子，带 `batch_path`（批次文件直接可用的键形态）。`--slice i/k` 可确定性分片，`--grep-path` 可按 journal 切 | `python pair_dump.py --repo <repo> --pack <包> --grep-en <正则> --full` |
| `qa/scan_attr_text.py` | **新（2026-08-13）**。HTML 属性里的**可见文本**没译。先普查库里实际出现的属性名再定白名单（本库 26 个，`title`/`alt`/`aria-label` 一个都没有，真正有的是 `@Embed[]` 体内的 `label=` 与 `readaloud=`）。另报**属性名本身被译成中文**（`target=`→`目标=`，浏览器整个忽略） | `python scan_attr_text.py --repo <repo> [--repo <另一个>] --out <json>` |
| `qa/scan_name_binding.py` + `qa/dump_bindings.mjs` | **新（2026-08-13）**。表结果名 / 场景针脚名 ↔ 它 id 绑定的目标文档名。**绑定关系只在 LevelDB packs 里，不在译文文件里**，所以要先用 `dump_bindings.mjs` 导出。分 BROKEN / UNCERTAIN / BY_DESIGN / NOT_BOUND | `node dump_bindings.mjs --package <foundry包> --out <json>` 然后 `python scan_name_binding.py --repo <repo> --bindings <json> --out <json>` |
| `qa/scan_name_splits.py` | **新（2026-08-13e）**。同一个英文 `name` 在库里有**两套以上中文名**（玩家会在两处看到同一样东西叫两个名字）。只报分裂、**不给建议** —— 实测多数派常常是错的那边（`Signborn Lineage` 星兆血统 3:1 胜出，但 §8 已裁 Signborn＝印记裔）。也有**合法分裂**（`Shield` 既是法术护盾术又是装备盾牌），要人排除。首测 5449 个英文名里 **54 个分裂** | `python scan_name_splits.py --repo <repo> [--repo <另一个>] --out <json>` |
| `qa/scan_token_name.py` | **新（2026-08-13d）**。`tokenName` 与 `name` 的中文不一致 —— **玩家在地图上直接看到的就是 tokenName**，而它不在任何既有判据的配对范围里。判据取严格版：**英文侧 `name` 与 `tokenName` 逐字节相同**时中文侧也必须一致（英文侧本就不同的是作者有意的短称，不算）。首测 481 个同名 actor 里 **32 个不一致** | `python scan_token_name.py --repo <repo> [--repo <另一个>] [--fix-batch-dir <目录>] --out <json>` |
| `qa/scan_bare_english_names.py` | **新（2026-08-13d）**。中文正文里留着的**裸英文专名**（有译名却没用上）。先从 `name` 字段成对提取「英文专名→中文译名」词典，再把全部专名编成**一条**交替正则每叶扫一遍（朴素的 O(专名×叶子) 在本库跑不完）。排除双语并列、标记内部、HTML 属性值，并要求该叶英文里确实有这个专名。首测 `--min-words 2`：词典 3255 条，**160 处 / 55 个专名** | `python scan_bare_english_names.py --repo <repo> [--repo <另一个>] --min-words 2 --out <json>` |
| `qa/scan_same_en_split.py` | **新（2026-08-13 第九轮）**。全库**同一条英文串有 N 种中文**。既有判据的三重盲区：`sync_twin_packs` 只比两包的**同一路径**、`propagate_fix` 只推 `--since` 之后的增量、`scan_name_splits` 只比 `name` 字段 —— 而**同一英文串出现在不同路径上**（`Gesture: Influence.description` 挂在 23 个 actor 上）从来没人比过。首测 **1514 组 / 8308 叶**。**只报分叉不给建议**：实测多数派常是错的那边（`lunge` 的 16 份写「具有灵巧属性的武器」，英文是 `scales using Dexterity`＝以敏捷成长）。带 `--flagged-only` 只看术语分歧/裸英文那 569 组 | `python scan_same_en_split.py --repo <repo> [--repo <另一个>] [--min-en-len 300] [--flagged-only] --out <json>` |
| `qa/sync_twin_packs.py` | **新（2026-08-13b）**。孪生包全库对齐：`ember.crucible-adventure` 与 `ember.adventure` 中**英文逐字节相同、中文却不同**的路径一次性统一到主线。`propagate_fix.py` 只推 `--since` 之后改过的，**存量分叉从没被清过** —— 首测 484 条 | `python sync_twin_packs.py --repo <repo> --out-dir <批次目录>` |
| `qa/scan_label_vs_name.py` | **新（2026-08-13b）**。`@UUID[目标]{标签}` 的中文标签 ≠ 目标文档的中文 `name`。**严格判据**：只报「英文标签本来就等于目标英文 name」的（作者换称呼、代词式简称、`#锚点` 指小节的一律排除）。首测 ember 490 / crucible 14，塌缩成 **153 组唯一配对**。⚠ **方向要逐组判，`name` 也可能是错的那一边**（`拉斯特·索恩`→name`栅格荆棘` 就是 name 机翻） | `node dump_ids.mjs --package <包> --out ids.json` 然后 `python scan_label_vs_name.py --repo <repo> --ids ids.json --out x.json` |
| `qa/scan_renamed_terms.py` | **新（2026-08-13）**。上游改名/删掉的英文专名，**旧译名还留在中文里**。两个探测器：中文里写着的英文串在当前基准 0 次 / 旧基准有而当前 0 次的专名其中文译名仍在库里。必须带英文闸排除同形常用词 | `python scan_renamed_terms.py --repo <repo> --old <旧基准目录> [--ids <id表>] --out <json>` |
| `4-临时脚本/2026-08-12-audit3/drift_dump.py` | 把 `scan_en_drift` 报出的条目按**旧英文 / 新英文 / 现有中文**三元组完整导出（报告里三段都截到 220 字符，只够排序不够判断） | `python drift_dump.py --drift <报告> --repo <repo> --baseline <旧基准> --start N --limit N` |

### 第八轮新判据脚本（`4-临时脚本/2026-08-13-round8/probes/`）

12 类全新判据，各自带**双向回测**（`inject_*.py` / `backtest_*.py` 是灵敏度注入器，
只在 `%TEMP%` 副本上跑，从不碰 `compendium/`）。想复现结论直接重跑对应脚本。
**其中只有 `scan_status_name.py` 建议进常备判据集**（371 一致 / 0 不一致，0 假阳性，秒级）；
`scan_pronoun_gender.py` 的 **G 臂**（实体锚定）也是 0 假阳性、全库约 70 秒，值得常备，A–F 六臂不要单独跑。

| 脚本 | 查什么 | 首测 |
|---|---|---|
| `scan_status_name.py` | 英文 `name` 恰为 crucible 系统状态名的条目，中文须与 `lang` **逐字相同**（不能用子串） | 371 一致 / 4 不一致 |
| `scan_relation_direction.py` | `at least`↔`up to`、`half`↔`double`、**`Round`轮 / `Turn`回合** 互换 | raw 24 → 确认 21 |
| `scan_body_name_splits.py` | **正文**里的专名分裂（`scan_name_splits` 只比 `name` 字段） | raw 188 → 确认 23 |
| `scan_cn_typography.py` | 排版约定；**标记功能区混入全角字符是功能性的** | raw 64 → 确认 13 |
| `scan_unmapped_fields.py` + `census_pack_fields.mjs` | 从 LevelDB packs 枚举全部字符串叶，减去 `mappings.mjs` 的覆盖集 | raw 91 → 确认 12（**2 阻断**） |
| `scan_dup_block.py` | 叶内重复块：中文两块相同而英文两块不同＝丢了一整块 | raw 43 → 确认 11 |
| `scan_enricher_arg_vs_prose.py` | enricher 方括号参数渲染出的中文 ↔ 相邻正文术语 | raw 52 → 确认 10 |
| `scan_uuid_deadlink.py` + `dump_uuid_index.mjs` | 死链（含 `#锚点`）；**判据关键是英文侧同路径对照** | raw 1277 → 确认 9 |
| `scan_modal_strength.py` | `must`/`should`/`may` 跨档冲突 + 施动者错位 | raw 45 → 确认 7 |
| `scan_list_alignment.py` | `<dt>/<dd>/<li>/<th>/<td>` 行列错位（地名志阻断的通用化） | raw 36 → 确认 6 |
| `scan_negation_drift.py` | 否定/条件从句被丢或反转 | raw 254 → 确认 3（噪声极大） |
| `scan_pronoun_gender.py` | 代词性别错配（实体锚定 + 最近先行词） | raw 19 → 确认 3 |
| `scan_orientation.py` | 方位/方向/序数 | raw 23 → 确认 3 |
| `crosscheck_vs_crucible_fr_r8.py` | 与 Crucible-FR 的 `compendium/en` 逐路径比，找「它有我们没有」的字段 | 盲区表第 9 项的复用手段 |

`3-常用脚本/parallel/` 下的 20 余个脚本（切单元 / 收批次 / 逐单元核对）**统一由 `PARALLEL-RUNBOOK.md` 说明**，此处不重复。

**批次文件格式**（喂给 `apply_translations.py`）：扁平 `{"<待译清单里的 path>": "<中文>"}`。
待译清单位于 `5-其他内容/reports/<crucible|ember>/todo/*.todo.json`。

**运行前提**：`node` 能从 `C:/Users/Taka/Desktop/fvtt` 解析到 `classic-level`；Python 3.14。
PowerShell 写批次文件会带 BOM，`apply_translations.py` 已按 `utf-8-sig` 读取。

### 外部参考实现

**Padhiver/Crucible-FR** — https://github.com/Padhiver/Crucible-FR
法语社区汉化，已对齐 crucible 0.10.1，值得抄的地方：

- `Hooks.once("babele.init")` + `game.modules.get("babele")?.active` 守卫
- `module.json` 里 babele 依赖直接写 `minimum: 2.9.1`，`compatibility.maximum: 14.999`
- 仓库里同时有 `compendium/en` 和 `compendium/fr`（英文基准进 git —— 与本项目做法一致）
- **`crucible._packs-folders.json`**（翻合集文件夹名，本项目缺）
- 转换器拆成独立 ES 模块 `scripts/converters-crucible.js` / `converters-ember-core.js`
- 他们的 `compendium/en` 可用来**交叉校验本项目抽取器有没有漏字段**
- 他们也做了 Ember 转换器（`emberPages`/`emberTables`/`emberSceneLevels`/`emberTableResults` 等），
  但**没公开发布 Ember 译文**

他们的策略是**混合式**：`compendium/en/mappings.json` 里只覆写 `Adventure.journals/scenes/macros/tables`、
`JournalEntry.pages`、`RollTable.results` 等，而 **`Adventure.actors` 和 `Actor.items` 保持 babele 默认** ——
所以他们同样拿到了源包回退。（早先记录说他们拿不到，是错的，已更正。）

**本项目与之的差别**：他们对 journal pages 用手写 `pages_converter`，本项目改用声明式子类型键
（`JournalEntryPage.ember.location` 等）。两者都能工作；声明式的好处是抽取器可以直接解释同一份数据，
不会出现「转换器读 A 键、抽取器写 B 键」的漂移。另外作者在 `converters-ember-core.js` 文件头自注
「本轮没在游戏里重测过」，所以他们的实现不宜盲信。

GitHub 上除本项目外**没有任何其他语言的 Ember 翻译**。

---

## 5. 标准操作 SOP

### 5.0 ⚑ 升级追平方案（Ember / Crucible 出新版后照这个走）

> 这一节是**发版之后**要落实的长期方案。核心洞察：
> **找「中文没跟上英文」不要靠启发式，靠旧版英文与新版英文的直接 diff。**

此前用过的间接信号各有盲区，而且盲区**重叠**：

| 检查 | 判据 | 盲在哪 |
|---|---|---|
| `validate_translations.py` | 路径上有没有中文 | 内容对不对完全看不见 |
| `measure_8c` / `measure_stale_extra` | `<p>`/`<li>` **块数** | 上游换内容但块数不变 → 沉默 |
| `scan_markup_drift` 的 `TRUNCATED` | 中文 < 英文 **0.22 倍** | 上游改写但长度相当 → 沉默 |
| 标记签名 | 标记多重集 | 只有标记跟着变才响 |

而 `EN_old != EN_new` 是**直接证据**。工具：`qa/scan_en_drift.py`。

**每次上游升级的标准动作**

```powershell
# 1. 先把「当前」英文归档成「旧版」——这一步漏了，下次就没得比
Copy-Item -Recurse "<repo>\compendium\en" "5-其他内容\english-baseline\<包>-<旧版本号>"
# 2. 抽新版英文，覆盖进 compendium/en
node "$P\3-常用脚本\extract\extract_en.mjs" --package <foundry包目录> --out "<repo>\compendium\en"
# 3. 重打 LOCAL-PATCHES.md 里记的上游笔误补丁（重抽会覆盖掉）
# 4. 三方 diff：英文变过、且中文更贴合旧英文的，就是要重译的
python "$P\3-常用脚本\qa\scan_en_drift.py" --repo <repo> --baseline "5-其他内容\english-baseline\<包>-<旧版本号>" --out <报告>
# 5. 按报告里的 items 切单元、发并行批次（沿用 PARALLEL-RUNBOOK 的页文件工装）
# 6. QA 全套 + 冒烟验证
```

**报告分四档**，按该管的先后：

| 档 | 含义 | 怎么处理 |
|---|---|---|
| `stale` | 英文变过 **且中文长度更贴合旧英文** | **优先重译**。这是主产物 |
| `changed`（非 stale） | 英文变过、但中文长度已贴合新英文 | 多半是上一轮已重译过，抽查即可 |
| `gone` | 上游删了、中文还在 | 死文本，清掉（babele 匹配不到，无害但占体积） |
| `new` | 上游新增 | 走正常翻译流程 |

**两个工具抓的不是同一类东西，都要跑**：

| 工具 | 抓什么 | 抓不到什么 |
|---|---|---|
| `scan_en_drift.py` | 英文**变过**、中文停在旧版 | 英文没变、但译文自始就不全 |
| `scan_content_coverage.py` | 英文里的**数字**中文没有 | 不含数字的漏译 |

那个「更贴合旧英文」的判别很关键：本库译文/英文纯文本长度比中位数是 **0.31**，
比较 `中文/旧英文` 与 `中文/新英文` 哪个更接近 0.31 即可。
**首测就靠它把 crucible 的 295 条压到 44 条、ember 的 921 条压到 280 条** ——
否则上千条无从下手。

> ⚠ **归档旧英文是这套方案的命门。** 没有旧基准，本方案退化成启发式。
> `compendium/en/` 只保存当前版本，历史快照必须另存到
> `5-其他内容/english-baseline/<包>-<版本>/`（第 1 节事实 2 就是讲这个）。

### 5.1 系统/模块升级后的例行检测

1. 记录新版本号，更新第 2 节版本矩阵
2. 用 `3-常用脚本/extract` 抽新版英文 → `5-其他内容/english-baseline/<包>-<新版本>/`
3. 拿**旧版**英文基准 + 新版英文 + 现有译文，算三方 diff：
   - `NEW` 新增条目（要翻）
   - `STALE` 上游删除的条目（可清理）
   - `DRIFT` 英文原文改动（要重翻）
   - `PARTIAL` 条目在但字段缺（要补）
4. 报告落 `5-其他内容/reports/`
5. 把新版英文覆盖进各仓库 `compendium/en/`

### 5.2 翻译批次

1. **先 TM/回源预填，再实译** —— 避免重复劳动与前后不一致
2. 按 pack 或按 journal 切批，单批控制在可校对的规模
3. 每批产出后立刻进三轮校准，不要攒着

### 5.3 三轮校准（每批都做）

- **R1 术语一致性** —— 对照 `glossary_ec.json` 强制校验，命中即替换/告警
- **R2 上下文一致性** —— 同一地点/人物/任务在不同页面的措辞对齐；
  `@UUID` / `@Check` / `<section class>` / `<span class="reference">` 标记完整性
- **R3 机械扫描** —— 残留英文、markup 漂移、长度异常（中译文长度通常为英文 0.4–0.7 倍，超出范围要看）

### 5.4 全库验证（发版前必跑，也是新会话做全局核查的入口）

一次跑完，**每一项都应为 0**。`<repo>` 取 `1-Ember汉化插件` / `2-Crucible汉化插件`，
`<pkg>` 取对应的 Foundry 包目录（`…\Data\systems\crucible` / `…\Data\modules\ember`）。

```powershell
$P = "C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
$Q = "$P\3-常用脚本\qa"

# 1. lang —— 必须四项全 0，尤其 UNREACHABLE（有中文但 Foundry 查不到＝键形态错）
python "$Q\lang_gap.py"      --repo <repo> --package <pkg> --out <reportDir>
#    另跑一遍拍平自检：拍平前 == 拍平后 == 英文键数，三者相等才算过
python "$Q\flatten_lang.py"  --repo <repo> --english <pkg>\lang\en.json

# 2. 标记五项 LINK / BLOCK / INLINE / PLACEHOLDER / TRUNCATED
python "$Q\scan_markup_drift.py"   --repo <repo>
# 2b. 方括号内部被译成中文的标记（链接/嵌入块会静默失效）
python "$Q\scan_markup_targets.py" --repo <repo>
# 2c. class 漂移 —— 闸门的签名只取标签名、看不见 class，
#     而 section.block gamemaster / ul.complex-check / sup.system-swap-inline 都是功能性的
python "$Q\scan_class_drift.py"    --repo <repo> --out <reportDir>\class_drift.json

# 3. 内容覆盖（中文有没有丢掉英文里的数字）
python "$Q\scan_content_coverage.py" --repo <repo>
# 4. 外来文字污染（西里尔/亚美尼亚/希伯来/泰文等机翻残留）
python "$Q\scan_foreign_script.py"   --repo <repo>
# 5. 死键（中文有、英文没有 —— babele 永远查不到，纯占体积）
python "$Q\prune_dead.py"            --repo <repo>          # 加 --write 才真删
# 6. 反方向：英文有、中文整条不存在（**上面每一项都覆盖不到这类缺口**）
python "$P\3-常用脚本\tm\fill_missing.py" --repo <repo> --out-dir <批次目录>

# 7. @UUID 标签挂错目标 —— 链接指向错误文档。上面每一项都全盲（目标与标签的多重集都与英文相等）
python "$Q\scan_uuid_swap.py"      --repo <repo> --out <reportDir>\uuid_swap.json

# 8. 跨通道：lang / compendium / ember-hardcoded-cn.mjs 三方对表
#    A/C 段已于 2026-08-13 重写（叶级对照替代 n-gram 窗口），三档：硬判据 /
#    LANG_WEAK_SIGNAL / LANG_NO_SUPPORT。后两档是「要人看」不是「有错」
python "$Q\scan_cross_channel.py"  --repo <repo> --package <pkg> \
       --mjs "$P\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs" --out <reportDir>\cross.json

# 9. HTML 属性里的可见文本（属性值没译 / 属性名被译成中文）
python "$Q\scan_attr_text.py"      --repo <repo> --out <reportDir>\attr_text.json

# 10. 表结果名 / 场景针脚名 ↔ 目标文档名（绑定在 LevelDB packs 里，先导出）
node   "$Q\dump_bindings.mjs"      --package <pkg> --out <reportDir>\bindings.json
python "$Q\scan_name_binding.py"   --repo <repo> --bindings <reportDir>\bindings.json \
       --out <reportDir>\name_binding.json     # 只看 BROKEN，UNCERTAIN 是没有中文名的目标

# 11. 上游已删的专名，旧译名还留在中文里（改名类 drift 的残留）
python "$Q\scan_renamed_terms.py"  --repo <repo> --old "5-其他内容\english-baseline\<旧版本>" \
       --out <reportDir>\renamed.json

# 12. @UUID 标签 ≠ 目标文档的中文 name（**2026-08-13b 实测最高产的机械核查**）
#     英文侧本来就不同名的已被判据排除；报出来的要逐组判方向，name 也可能是错的那一边
python "$Q\scan_label_vs_name.py" --repo <repo> --ids <reportDir>\all_ids.json \
       --out <reportDir>\label_vs_name.json

# 13. 地名志 <dt> 条目：NPC 名整个没译（叶子里有大量中文，覆盖率算它已译，所有闸门全盲）
python "$P\4-临时脚本\2026-08-12-audit3\fix_gazetteer_dt.py"   # 判据即扫描器，产批次

# 13b. 同一个英文 name 有两套中文名（只报分裂，方向要逐条判）
python "$Q\scan_name_splits.py"   --repo <repo> --out <reportDir>
ame_splits.json

# 14a. tokenName 与 name 的中文不一致（玩家在地图上看到的名字与角色卡对不上）
python "$Q\scan_token_name.py"    --repo <repo> --out <reportDir>	oken_name.json

# 14. 中文正文里的裸英文专名（有译名却没用上）
python "$Q\scan_bare_english_names.py" --repo <repo> --min-words 2 --out <reportDir>are_en.json

# 14b. crucible 系统状态名：compendium 的中文须与 lang 的状态译名逐字相同（2026-08-13f 新增）
#      玩家在 token 状态条上看到的是 lang，在效果名/表结果行上看到的是 compendium，两者必须同字
python "$P\4-临时脚本\2026-08-13-round8\probes\scan_status_name.py" --repo <repo> \
       --lang-en <pkg>\lang\en.json --lang-cn 2-Crucible汉化插件\lang\cn.json

# 16. ⚑ 决议断言 —— 把第 8 节的裁决当**闸**跑，而不是当散文读
#     十四轮里已经出过好几次「悄悄推翻既定裁决」的险，全靠人当场想起来才没出事。
#     这一条把一致性从「靠记忆维持」变成「靠闸维持」。**失败时不要只改断言让它变绿。**
python "$Qssert_resolutions.py"            # 全量；加 --verbose 看每条
python "$Qssert_resolutions.py" --selftest # 判据自身的正反例回测

# 15. 孪生包对齐 —— 英文逐字相同、中文却不同（**最后一步跑**，主线一改就会重新分叉）
python "$Q\sync_twin_packs.py" --repo 1-Ember汉化插件 --out-dir <批次目录>
```

> ⚠ **第 12–14 项有先后顺序**：先把 crucible 侧（主线）改完并落盘，**最后**才跑第 14 项，
> 否则主线每落一批就会制造新的孪生分叉。本轮实测：第五轮开始前分叉 286 条，
> 主线改完之后变成 502 条。

> ⚠ **第 5 项 `prune_dead` 现在会报出 ember 的 8 条 `_legacyActions`，那是有意寄存的内容，别删。**
> `_` 前缀是项目既有约定（`validate_translations.py:97` 明确跳过）。
> 清理类操作前先 `--show 200` 逐条核对判据 —— 教训见第 8 节 2026-08-12 的 textCollection 那条。

> ⚠ **第 6 项是 2026-08-12 才补上的盲区。** 第 1–5 项全都以「中文里的某条」为起点，
> 中文里压根没有的条目不在它们的定义域内 —— 所以库里一度报「覆盖率 99%」，
> 而 crucible 的两个预生角色几乎整体没译。**别再只跑前五项就宣布干净。**

> `port_orphans.py` 只搬路径、不改译文内容。上游改名后**译文里的旧名字不会自动更新** ——
> `Rune: Lightning`→`Rune: Storm` 就是这么留下 28 处「闪电」的。改名类 drift 处理完必须回头
> 搜一遍旧名字。

### 5.5 发版

**两个仓库都有 tag 触发的 `.github/workflows/release.yml`，正常情况不需要手工打包。**

1. 先跑 5.4 全套 + 冒烟验证
2. 改 `module.json` 的 `version` 与 `download`
   —— **tag 形态不同**：crucible 不带 `v`（`0.9.2`），ember 带 `v`（`v1.1.2`），
   两边 `module.json` 的 `download` 就是这么写的，workflow 会逐字校验，写错直接失败
3. 更新 `.github/release-body-template.md`（Actions 拿它当 release 正文）
4. 推 main → 打 tag → push tag，Actions 自动打包建 release
5. 发完**下载回包核对**：manifest/zip HTTP 200；`unzip -l` 确认没有 `compendium/en/`
   与嵌套的上一版副本（1.1.0 就因为 `zip -r` 是**追加**而不是重建，把 22 MB 陈旧副本发了出去）
6. 在第 6 节年表补一行

Actions 不可用时的兜底手工流程：`zip -r` 前先 `rm -f module.zip`，排除
`.git/* .github/* release/* compendium/en/* lang/lang_keep_english.json *.zip *.bak`，
然后 `gh release create <tag> module.json module.zip --notes-file .github/release-body-template.md`。

---

## 6. 阶段年表

> **完整记录已归档到 `5-其他内容/STAGE-LOG.md`**（1509 行，阶段 0–27 的原始测量、
> 走过的弯路、被推翻的判断）。这里只留一句话年表，用来定位「哪一步发生在什么时候」。
> 仍然生效的硬约束看第 3 节，仍然生效的裁决看第 8 节 —— 不要回头去日志里翻结论，
> 那里的数字是当时的状态。

| 阶段 | 日期 | 做了什么 |
|---|---|---|
| 0–2 | 08-06 | 基建、通用抽取器与英文基准、术语表构建 |
| 3–4 | 08-06 | babele 2.9.1 管线改造（声明式 `registerMapping`）+ crucible compendium 收官 |
| 5–7 | 08-06 | crucible `lang` 收官、全库术语统一、ember `lang` 收官 |
| 8–10 | 08-06 | ember 四个小包收尾 |
| 11 | 08-06 | Ember 运行时补丁：硬编码字符串 + 中文字体回退 |
| 12–13 | 08-06 | 标记漂移清扫；查出「显示 100% 实则缺整块」那类欠账 |
| 14–19 | 08-06 | 战役正文首批（Ushna Dredging Docks、Arcturel Dives）+ 一整卷被改名埋掉的旧译文 |
| 20–24 | 08-06~08 | **并行翻译管线**（页文件 + Edit 局部改）第 1–6 批；8c/8j 合并为「页面重对齐」并清零 |
| 25 | 08-09 | 按「自行裁决」新政策统一 9 组术语 |
| 26–27 | 08-09 | 标记签名失配两侧清零；孪生包连带释放 71 万字符 |
| — | 08-10 | **首版发布** `crucible-cn 0.9.0` / `ember_cn_unofficial 1.1.0` |
| — | 08-12 | lang 键形态修复、追平批、TM 补齐缺口、死键清理；发 `0.9.1` / `1.1.1` |
| — | 08-12 | 20 条积压术语分歧裁完、世界地图针脚重译、两处中文与英文不符；PROJECT.md 2239→644 行；发 `0.9.2` / `1.1.2` |
| 28 | 08-12 | **多 agent 全面审计**（13 路并行 + 逐条对抗验证 + 盲区批判，27 agent）：201 条原始 findings → **143 条确认**（8 阻断 / 62 严重）。查出的不只是译文错误，还有**检查器自己的四个盲区**（见 3.6）与**覆盖盲区**（35 本 journal 从未逐句复核，占 journal 正文 51%） |
| 29 | 08-12 | **多 agent 修复**（16 路修复 + 16 路独立复核，32 agent）：8 条阻断全修；工具盲区补齐并新增两个检测器；59 个批次经三方合并（350 条路径冲突、19 条人工裁决）落盘 **2516 叶 + lang 163 键**；复验 5.4 全套 + 新增两项全 0 |
| — | 08-12 | 发 `crucible-cn 0.9.3` / `ember_cn_unofficial v1.1.3`。**冒烟验证仍未做**（项目所有者明示先发） |
| — | 08-13 | **发 `crucible-cn 0.9.4` / `ember_cn_unofficial v1.1.4`**。发版前 5.4 全套 + `flatten_lang` 三数相等（486/1842）全部通过；下载回包核对无混入、抽查正确。**⚠ 冒烟验证仍未做**，且本轮首次改动 Scene 层 mapping。踩到一个坑：crucible 的 `module.json` 除 `version`/`download` 外**还有 `changelog` 字段要指向本 tag**，漏改导致第一次构建 8 秒即失败（ember 没有该字段），删 tag 补提交后重发成功 |
| 36 | 08-13 | **第九轮：全库同源串统稿**（13 agent）：主目标是第八轮顺带发现的最大系统性缺陷 —— **同一条英文串有 N 种中文**（1514 组 / 8308 叶），三重盲区（`sync_twin_packs` 只比同路径 · `propagate_fix` 只推增量 · `scan_name_splits` 只比 name 字段）导致它从没被查过。11 路分片统稿 **1068 组统一 / 483 组判为合法分叉 / 4652 叶落盘**，报出 **225 条实质缺陷**（定译译错 201 叶 / 语义误译 301 叶 / 裸英文 83 叶 / 标记丢失 63 叶）。结果：1514→**470 组**，8308→**2561 叶**，术语分歧 25→**0**，>300 字符长正文分叉 357 组→**0**。另两路：修上游笔误 + 给三个判据收口（`bare_english` 37→0、`attr_text` 113→0、`uuid_swap` 68→6），三个都做了双向灵敏度回测 |
| 35 | 08-13 | **第八轮：多维度新判据探矿**（12 探针 + 13 对抗式复核，25 agent）：不跑现有检查，改为**每路发明一个至今没有判据的错误类别**并做双向回测。121 confirmed → 复核后**存活 136 / 推翻 63**。43 批次经三方合并（681 路径 / 24 条多组认领 / 2 条人工裁决）落 **683 叶**。最大产出是盲区表第 9 项的又一次命中：**dnd5e 侧约 85 万字符从未进过管线**（物品 description + NPC 传记 `.value`），两条阻断都需项目所有者决策。主控另自查出 `Confused` / `Charged Bite` / 标签空格三类共 14 叶，并新增判据 `scan_status_name.py`。**`scan_uuid_swap` 的 UNCERTAIN 档实测 97% 假阳性，判定为已到极限** |
| 34 | 08-13 | **裸英文专名 + tokenName**（5+5 agent）：① `scan_bare_english_names` 报的 160 处逐个判「真缺陷 / Patch 页通则 / 假阳性」，修到 **63 处**（典型：`Del Kalais` 那处**下一句中文就写「德尔随和却技艺高超」**）；② 新写 `qa/scan_token_name.py`，查出 **32 条 tokenName 与 name 不一致**（`Thayloc Courser` 的 token 名是「非玩家角色」这种占位符）；③ `Patch 0.4.7` 整页补全 16 处 —— 主控先按四单元多数剔了 B3，拿到完整复核数据后**改判**（见第 8 节 2026-08-13e） |
| 33 | 08-13 | **补 Scene 三字段管线 + 译完 517 处**（6+6 agent）：项目所有者定「优先完整汉化，补管线」。`levels`（255 唯一层名）/ `tokens`（18 处，已摆在场景上的 token 覆盖名）/ `navName`。用内建 `nameCollection` 而非 FR 的自定义转换器；重抽英文基准**只并新字段**以免回退 `LOCAL-PATCHES`。正文回填反转 118 处。**覆盖率由此到 100%**。另新写 `qa/scan_bare_english_names.py`，实测中文正文里还有 160 处裸英文专名（留给下一轮） |
| 32 | 08-13 | **复核补漏 + 标签↔name + 孪生包对齐**（9+8+3 单元，23 agent）：① 执行对抗式复核补出的 188 条漏项，逐条独立核实后 **接受 ~155 条、驳回一批**（复核 agent 自己也会错：有一条前提事实就是错的，照改会把对的改坏）；② 新写 `qa/scan_label_vs_name.py`，全库 504 处 → 153 组唯一配对，逐组判方向（改标签 88 / **改 name 63** / 不动 2），落盘后复跑 **504 → 2**；③ 补做 J19/J25 两个撞额度没跑成的复核；④ 新写 `qa/sync_twin_packs.py` 清掉 **484 条存量孪生分叉**。合计再落 1284 叶 |
| 31 | 08-13 | **35 本 journal 逐句复核**（27 单元 + 27 对抗式复核，54 agent）：实读 **458 万英文字符**，**507 条确认缺陷**（2 阻断 / 78 严重），复核另补 188 条、驳回 4 条（其中 2 条是**部分**驳回 —— 同叶别的改动是对的，只回退一处）。553 叶经三方合并（12 条冲突、11 条人工裁决）落盘，另 **581 条推送到孪生包**。同批：U1 的 @UUID 标签 43 处 + 复核补漏 12 处、`Critical Success/Failure` 全库统一、lang 4 键对齐 `.mjs` |
| 30 | 08-13 | **第三轮多维度查缺补漏**（15 单元并行 + 逐单元对抗式复核，30 agent）：走**旧英/新英 diff** 这条路复核 `stale` 桶 278 条，另做 226 条 `@UUID` 标签裁决、四个新判据、`.mjs` 177 条逐条审、crucible-fr 交叉校验。**161 条确认缺陷（3 阻断 / 44 严重）**，复核驳回 5 条。24 条路径冲突经三方合并（2 条人工裁决）落盘 **448 叶 + lang 4 键 + .mjs 补丁**。查出的最大新盲区是**抽取器根本没抽 `Scene.levels[].name`**（195 场景 517 处），由 E1 与 H2 两个单元各自独立撞到 |

## 7. 现状与唯一未做项

**排期表已并入第 6 节年表**（原表 48 行全部 ✅，只是历史）。当前状态见第 1 节。

| # | 事项 | 状态 |
|---|---|---|
| 9 | **真实 Foundry 世界冒烟验证** | ⬜ **最高优先级**。脚本证不了，必须开世界看。清单见本节末，**阶段 29 又加了三项，而 0.9.3 / v1.1.3 已经在没做它的情况下发出去了** |
| 10 | **35 本 journal 从未逐句复核** | ✅ **2026-08-13 做完**。27 个并行单元逐句读完 458 万英文字符，**507 条确认缺陷**（2 阻断 / 78 严重），另由对抗式复核补出 188 条。详见第 1 节。**注意**：这不等于这 35 本从此干净 —— 复核 agent 的补漏率说明单次通读的召回率约 73%，真要再压一轮，成本最低的做法是**机械核查**（@UUID 标签 ↔ 目标 name 全量对照）而不是再读一遍 |
| 11 | `scan_cross_channel.py` 的 A/C 段词对齐是坏的 | ✅ **2026-08-13 重写**。见第 1 节盲区表第 4 项 |
| 12 | `@UUID` 标签命名不一致 226 处 | ✅ **2026-08-13 逐条裁完**。见第 1 节盲区表第 3 项（含 `en_label` 不可信这个方法坑） |
| **13** | **Scene 的三个字段完全没被抽取** | ✅ **2026-08-13b 补齐管线**（项目所有者定：优先完整汉化）。三个字段是同一个改造点，一起做的：`levels[].name`（195 场景 / 517 处 / **255 个唯一层名**，玩家在层级选择器里看到的永远是英文）· `tokens[].name`（**已摆在场景上**的 token 的覆盖名，18 处；不能假设「actor 名翻了 token 就跟着变」——实测 3 个 token 名里 2 个与 actor 名不是同一个字符串）· `navName`（画布顶部导航条，2 处）。根因：`Scene` mapping 照抄 babele 2.9.1 默认，而默认里只有 `name/drawings/notes/regions`。**做法**见第 8 节 2026-08-13b 那条：用内建 `nameCollection` 而不是 FR 那样的自定义转换器 |
| 8f | `Arcturel Tradeway` 28 页 | ✅ 2026-08-12 已逐页通读（阶段 28），并修掉了「英文尾巴写着上游已删除的旧名 Arcturel Upper/Lower」那批 |

**35 本从未逐句复核的 journal**（按体量降序，阶段 28 盲区批判点名）：
Glitter in the Dark / The Expedition Challenge / Crumbling Sanctuary / An Old Friend / Disturbed Earth /
Ancient Paths / Smoldering Cinders / To Fall and Fall Again / Thorny Predicaments / Unfinished Business /
Disgraced House / The Book Of Tales / Spreading Sickness / A Brush With Death / Diplomatic Impunity /
Lantern Roads / Signal of Intent / Toothbreaker Hideout / Flotsam Canal Market / Spellbreaker Tower /
Chapter 2 Events / Kalion Stadium Underworks / Traveler's Rest / Chamber of Agaseros / Local Color /
Forgotten Cistern / Burial Grounds / The Bronze Rask Theater / Chapter 1 Events / Jekeroka Villa /
Redwalk Ramble / Oldcraft Lodge / Ushna Dredging Docks / Pit Trap / Chapter 3 Events

> 阶段 29 已按「汉字数/英文字符数」比值（本库中位数 0.31，>0.44 疑增、<0.20 疑漏）在这 35 本里筛过一遍并修了命中的，
> **但比值只是候选筛**：已知的两条阻断里有一条比值就在正常区间内（0.377）。逐句对读没有替代品。

> ⚠️ **不要拿 `validate_translations.py` 的百分比当真实缺口。**
> crucible 显示 97%、ember 显示 99%，但那 436 条待译**全部**由 babele 通用回退
> 从已译包按名字自动取译文。动手前先跑：
> ```powershell
> python "$P\3-常用脚本\qa\resolve_generic_fallback.py" --repo <repo> --also <另一个 repo>
> ```
> 重复翻译只会制造同名异译。**两个包的真实残余都是 0。**

### 待清扫的既有缺陷（2026-08-12 全部清完，A–O 共 15 项）

**全部已修并复验**，逐项的证据与裁决理由在第 8 节（按日期查 2026-08-12 那批）。
这里只保留三条**仍然生效的教训**，其余不再复述：

- **标 ✅ 的项必须拿数据复验，不能照抄上一版状态。** 上一轮曾把 C/D/E/J 四项直接誊成 ✅，
  实测其中 E 与 J **根本没做**，而 C/D 又被误报成没做（裸计数「电能」25／「阶位」160 看着像残留，
  加英文闸后全是别的英文）。
- **`port_orphans.py` 只搬路径、不改译文内容。** 上游改名后译文里的旧名字不会自动更新
  （`Rune: Lightning`→`Storm` 就这么留下 28 处「闪电」）。改名类 drift 处理完必须回头搜旧名字。
- **中文侧整条不存在的键，所有既有扫描都发现不了** —— 覆盖率/残留/签名/drift 全是拿
  「中文里的某条」去比对。反方向的检查是 `tm/fill_missing.py`（第 5.4 节第 6 项）。

### 冒烟验证怎么做（第 9 项）

管线改造过但**没有在真实世界里跑过**。开一个 Crucible 世界，控制台执行：

```js
game.babele.inspectMapping('Item')                    // 应看到 crucibleDescription / crucibleActions
game.babele.inspectMapping('JournalEntryPage', {data:{type:'ember.location'}})  // 应命中子类型层
await game.babele.sourceDiagnostics()                 // 每个 collection 的译文来源与重叠
game.babele.cacheDiagnostics()
```
重点确认：① 合集里天赋/装备显示中文；② 导入 playtest 冒险后，角色身上的内嵌物品也是中文
（这条验证的是源包回退）；③ 控制台无 `TypeError`。

阶段 5 之后补充四个要看的点（都是这轮改动可能出问题的地方）：

- **动作卡上的消耗标签**：`{action}动` / `{focus}专` / `{heroism}英` / `武{action}动`
  原本是 `2A`/`1F`/`W2A`，改成中文后字更宽，确认没有换行或截断
- **物品名拼装**：品质/词缀会拼成「精良长剑」「长剑·腐蚀」，看一眼语序是否可接受
  （`ITEM.COMPOSED_NAME.Prefix` / `.Suffix`）
- **法术名拼装**：`迅捷的 燃烧的打击` —— 屈折形容词与法术名之间的空格是系统硬编码的，改不掉
- **rules「符文」页**：那个原本坏掉的 `@Embed` 现在应该能正常渲染出「符文：风暴」

阶段 29 之后必须补的三个（本轮改动的直接产物，只有开世界才能确认）：

- **世界地图 hex 的地形**：`system.terrain` 是带 `choices` 的枚举 id，此前被当正文翻译了
  （104 条中文值写进了文档数据）。已从 mapping 删除该字段、清掉全部中文值、并同步清了英文基准。
  开一张区域地图，确认水域 / 困难 / 极端地形与移动消耗恢复正常、控制台无 `Terrain ... not defined` 报错。
- **附魔物品的拼装名**：新补了 `ActiveEffect` 的 `adjective` / `actions` mapping 层。
  此前 169 条译文运行时根本不被查，玩家看到 `Acid-Warding长剑`、动作卡 `Replenish Action`。
  拿一件带词缀的装备看名字与动作卡是否已是中文。
- **合集侧边栏的文件夹名**：新增 `compendium/cn/crucible._packs-folders.json`（Crucible / 对手选项 /
  角色选项 / 角色 / 物品）。**babele 的这个能力本项目从未实测过**，文件格式是照
  `babele/script/compendium/folder-translations.js` 与 `translation/pack-folder-translations-catalog.js`
  推出来的（按 `collection` 以 `._packs-folders` 结尾识别），要在世界里确认真的生效。

阶段 30（第三轮）之后必须补的四个 —— 前三个是 `.mjs` 的**机制**修复，
「译文对不对」看不出来，只有开世界才知道到底生没生效：

- **事件状态浮窗**：鼠标停在任意 `[[/eventState …]]` 链接上，浮窗应是「事件已完成 / 事件未完成」。
  这一条验的是属性白名单补了 `data-tooltip-text`（v14 的取值顺序是 tooltipHtml > tooltipText > tooltip，
  ember 用的正是中间那个，原先漏了它 ＝ 那几条永不生效）。
- **确认框标题**：在事件页点「重置事件」，弹窗标题应是中文；在远景配置里删构图，应是「删除已保存的构图？」。
  这一条验的是放行原生 DialogV2（15 个确认框的根元素 class 只有 `dialog`、类名就是 `DialogV2`，
  原先被 ember 闸整个挡掉）。**顺带确认别的模块的窗口标题没有被误改**——闸已收窄到 DialogV2 一档，
  但 `EXACT` 里确实有 `Path` / `Culture` / `Events` 这类通用词。
- **世界时钟**：日期串应是「第 43 天 - 12:00」。若**推进时间后变回英文**，
  说明需要落 `H1_patch.md` 的 **P-9**（`EmberCalendarNavigation.animate()` 直接重写 `innerText`
  且不重渲染，只挂 render 钩子的话第一次时间推进就被刷掉）。P-9 因为要猴补丁第三方原型、
  且作者自己标注「需冒烟验证」，本轮**没有落**，补丁原文在
  `4-临时脚本/2026-08-12-audit3/`（会话产物 `findings/H1_patch.md`）。
- **场景层级选择器**（2026-08-13b 补了管线，这条现在是**验新功能**而不是验现状）：
  随便开一张有多层的 Vista（例 `Vista: Ordain Interiors`），确认：
  ① 层级选择器里的层名是**中文**；② 切换层级正常、地图内容没错位
  （`nameCollection` 只 merge `name` 一个键，`bottom`/`top`/`_id` 应当原样保留）；
  ③ 画布顶部导航条上 `Aedir Signalpost` 显示「塔楼眺望点」；
  ④ `Repurposed Quarry - Middle` 图上已摆好的 token 显示「活体解剖师」「卡拉萨克」，
  ⑤ 控制台无 babele 报错。**这是本项目第一次动 Scene 层 mapping，务必看控制台。**

阶段 35（第八轮）之后必须补的两个：

- **`#锚点` 链接是否真的跳对**（这是本轮**唯一**的运行时行为改动，务必看）：
  给 112 叶的 `<hN>` 补了显式 `id=`，因为标题译成中文后 Foundry 的 slug 变了、234 处
  `@UUID[...#anchor]` 全断。随便找一条带 `#` 的链接点进去（例 `#hex-based-exploration` 21 处、
  `#prison-doors` 13 处），确认**跳到对应小节而不是页首**。若不跳，说明 Foundry 的 TOC 构建
  不认显式 `id`，那 112 叶的 `id` 要撤掉、改走别的路。
- **`Round`/`Turn` 改动后的规则读起来对不对**：本轮把若干处互换的「轮 / 回合」改正
  （`Round`＝轮、`Turn`＝回合）。挑一条改过的怪物能力（例 `Helkas Drake Moments` 6-6
  「持续接下来的六个回合」）与英文对读一遍。

阶段 24–27 之后再补三个（都是这几轮大改动的直接产物）：

- **GM 专属块是否仍对玩家隐藏**：第 6 批补回了大量
  `<section class="block gamemaster">`。以玩家身份看一页 `Area Overview`，
  确认「游戏主持人摘要」没有暴露出来。
- **双系统分支只显示当前系统那一支**：第 7 批补回了大量
  `<sup class="system-swap-inline">`。在 Crucible 世界里看一段带检定的正文，
  确认只出现 `[[/skillCheck …]]` 那一支，不会两套规则并排显示。
- **孪生包不会被白 fetch**：`compendium/cn/ember.adventure.json` 有约 **9 MB**。
  在 **Crucible** 世界里开控制台看 Network，确认 babele 没有去拉它
  （dnd5e 版的包在 Crucible 世界里根本不存在，拉了就是纯浪费流量）。

---

## 8. 决议记录

| 日期 | 决议 | 理由 |
|---|---|---|
| 2026-08-06 | 术语表以 `glossary_crucible_merged.json`（4/16，4602 条）为基底 | 是本地所有 crucible 术语表的超集，且冲突已裁决。另一份 `glossary_adaptive_crucible.json` 与之逐条相同 |
| 2026-08-06 | **不**并入 PF2E 主表 `fvtt\glossary.json`（10942 条） | 不同世界观。例：`Restrained` 本项目「受缚」vs PF2「受制」。仅作低优先级建议源 |
| 2026-08-06 | 术语冲突采用新版裁决 | `Restrained` 受拘束→**受缚**；`Arcden` 阿克登语→**奥克登语**；`jurtak` 尤塔克→**尤尔塔克**；`Hulg'run Lineage` →**赫尔格伦血统**；`Ken Crystals` 感晶→**肯水晶**；`House Cevher` 切夫赫尔→**杰夫赫尔**（共 28 条，全表见 `5-其他内容/glossary/`） |
| 2026-08-06 | 管线用 babele 声明式 `registerMapping` + `_variants`，而非手写遍历转换器 | 手写转换器拿不到 2.9.1 的源包回退，会白丢约 60.9 万字符的免费翻译 |
| 2026-08-06 | 英文基准同时存仓库 `compendium/en/` 与 `5-其他内容/english-baseline/` | 前者是当前版本、跟译文一起进 git；后者是历史快照、用于跨版本算 drift。用途不同，不是重复 |
| 2026-08-06 | 专有名词保持 `中文 English` 双语并列格式 | v1.0.15 既有译文已是此风格，改动会造成大面积不一致 |
| 2026-08-06 | 抽取器改为**解释 mapping 数据**，而非硬编码字段 | 运行时与抽取端共用同一份 mapping，结构上杜绝键名漂移 |
| 2026-08-06 | 保持既有键形状（`description` 多态、`actions` 按 id、`ancestry` 等嵌套对象），不改名 | 改键形状会让 crucible-cn 现有约 4600 条译文全部失配 |
| 2026-08-06 | 每次抽取器改动后必须跑一次 `crosscheck_vs_crucible_fr.py` | 多态 description 那个 bug 单看自己输出完全正常，只有独立实现对照才暴露 |
| 2026-08-06 | ember 的 `-en.json` 从 `compendium/cn/` 移出到 `english-baseline/` | 它们躺在 babele 注册目录里会被每次开世界 fetch（其中一份 11.4 MB 且是损坏 JSON） |
| 2026-08-06 | 各 pack 文件的 `mapping` 块一律删除，改用全局 `registerMapping()` | compendium-local 映射优先级高于注册层，留着会静默盖掉新映射并继续调用已不存在的转换器 |
| 2026-08-06 | 同名文档若内容可合并则合并到名字键，标量冲突才用 `_id` 键 | babele 匹配 `_id` 优先于 `name`；分开发键既翻倍工作量，又会让日后改名字键的译文静默失效 |
| 2026-08-06 | `@UUID[目标]{标签}` 的标签**要翻译**，校验只比对目标 | 标签是玩家看到的可见文字；早期把整段当作不可变标记，导致合法译文被拒 |
| 2026-08-06 | `Kinesis` → **念力**（非术语表里的「念动力」） | 既有译文已用「念力术师 Kineturge」「符文：念力 Rune: Kinesis」「念力熟练度」，服从既有用法 |
| 2026-08-06 | `Warden` → **守林者**（非术语表里的「典狱长」） | 语境是「召唤自然先祖魂灵的战斗德鲁伊」，且要避开 `Guardian`＝守护者的碰撞 |
| 2026-08-06 | `Swarm`（archetype）→ **群集**，非「虫群」 | 该 archetype 描述是「多个生物以单一群集实体行动」，并不限于虫类；具体生物名如 Insect Swarm 仍用「虫群」 |
| 2026-08-06 | 新增译名：`Automaton` 自动人偶 / `Dust Devil` 尘卷风 / `Mud Elemental` 泥浆元素 / `Stone Elemental` 岩石元素 / `Constrictor` 绞杀者 / `Juggernaut` 碾压者 / `Telekinetic` 念力者 / `Deep Behemoth` 深层巨兽 / `Lightweaver` 织光者 / `Prankster` 恶作剧者 / `Ancestral Guardian` 先祖守护者 / `Ancestral Ward` 先祖庇护 / `Ancestral Spirit` 先祖之灵 / `Rune: Storm` 符文：风暴 | 对齐既有 taxonomy 风格（土元素/火元素/冰霜元素、元素微粒）与 archetype 风格（成年龙兽/狂战士/掘穴者） |
| 2026-08-06 | crucible-cn 的 `i18nInit` 里 `sort = "tri"` 改为「排序」 | "tri" 是法语，抄 crucible-fr 时的遗留 |
| 2026-08-06 | `DEFENSES.Madness`/`Wounds` → **集结阈值 / 治疗阈值** | 上游 0.10.1 把这两个防御改名为 Rallying/Healing Threshold；`RESOURCES.Madness`/`Wounds`（疯狂 / 创伤）是另一组东西，不动 |
| 2026-08-06 | `Hazard` → **危害**，`Danger Level` → 危险等级 | 服从 compendium 既有的「环境危害 / 配置危害」 |
| 2026-08-06 | `Inflection` → **屈折**，且 talent 条目名统一为「屈折：X」 | 原译「词缀」与 `Affix`（词缀）完全撞名，玩家无法区分 |
| 2026-08-06 | 屈折/施法构件用词以 `crucible.affixes` 的 `adjective` 为准：编构 / 限定 / 遁避 / 延展 / 否定 / 拉拽 / 推挤 / 迅捷 / 反应 / 重塑 | 那一组 10 条是唯一内部自洽的；talent 侧原本是作曲 / 判定 / 闪避 / 推开 / 迅捷化，且「闪避」与 `Dodge` 撞名 |
| 2026-08-06 | `Critical Hit` → **暴击**（lang 里原有的「重击」改掉） | glossary 与 compendium 正文都是暴击，只有 lang 一处例外；且「重击」易与 `Strike`（打击）混淆 |
| ~~2026-08-06~~ | ~~`Fortitude` 的 lang 标签由「坚韧」改为「坚韧防御」~~ | **已被 2026-08-12 推翻** → `Fortitude` 改 **强韧**。加后缀只是把撞名藏起来：正文里的裸「坚韧」仍然分不清指属性还是指防御 |
| 2026-08-06 | `Signature`（天赋树）→ **招牌**，非「签名」 | `TALENT.WARNINGS.Banned` 早已写作「招牌天赋」，节点标签却是「签名」 |
| 2026-08-06 | 紧凑标签译成中文：`{action}A`→`{action}动`、`F`→专、`H`→英、`W`→武、`{value}R`→`{value}轮` | 目标是完整汉化；但字宽会变，列入冒烟验证清单 |
| 2026-08-06 | `DC` / `∞` / `???` 有意保留原样，记入 `lang/lang_keep_english.json` | DC 是中文桌游圈通用写法；后两个是符号。不进白名单的话每轮都会被报成漏翻 |
| 2026-08-06 | 物品名拼装：前缀 `{prefixes}{name}`、后缀 `{name}·{suffixes}` | 英文的 "Sword of Flame" 语序在中文里不成立；间隔号是国内游戏常见写法，且不会与词缀名内部的字冲突 |
| 2026-08-06 | 中文没有 zero/two/few/many 复数形态，这些键一律填与 `other` 相同的值 | Intl.PluralRules 对 zh 只会取 other；填上只是为了让 `lang_gap.py` 不再报缺口 |
| 2026-08-06 | `Electricity` → **电击**，状态 `Shocked` 连带改为 **感电** | 电力像市电、电能像物理量；伤害类型与状态不能同名 |
| 2026-08-06 | `Radiant` → **光耀**；`Poison`（伤害类型）→ **毒素**；`Psychic` → **灵能** | 都是正文多数写法；辉光/光辉太像，毒药指物不指伤害类型 |
| 2026-08-06 | `Tier` 作独立名词时用 **阶数**（最低阶数 / 阶数配置 / 附魔阶数），计数时用 **阶**（1 阶 / 每阶） | 单说「阶」在 UI 标签里不成词，但正文计数必须与 compendium 的「1 阶」一致 |
| 2026-08-06 | `ABILITIES.*Abbr` 六个缩写定为 敏 / 智 / 存 / 力 / 韧 / 感 | 原本是机翻（`Pre`→「预备」、`Tou`→「图」）；属性框位置窄，单字最合适 |
| 2026-08-06 | `Formidable Presence` 译作 **威严气场**，不套用 `Presence`＝存在 | 这里的 presence 是「气场」的日常义，不是属性值 |
| 2026-08-06 | 被写成裸中文的 enricher 一律还原成 `@Condition[...]` / `@Action[...]` 等标记 | 这类标记不带标签、渲染出来就是 lang 译名；写成裸字等于让玩家失去可点链接与说明浮窗 |
| 2026-08-06 | enricher 方括号内的**目标与参数一律照抄**，只有 `{标签}` 可译 | 已经踩到两次：crucible 的 `@Embed[...runeLightning000...]`、ember 的 `@Embed[... inline overview]` 被译成「概览」。两次都是链接/嵌入块直接失效，且 diff 里看不出来 |
| 2026-08-06 | `Globlin` → **格布林**（音译），不用意译「泥砾精」 | 遇到 `Mud Globlin` / `Paint Globlin` 这类前缀构词时意译没法组合；战役包里还有一处误译成「绘画地精」，而地精是另一种生物 |
| 2026-08-06 | `Waterborne` 作家族名时音译为**沃特伯恩** | 该家族经营酿酒坊；原有一处 `Waterborne Whiskey` 被当普通词译作「水运威士忌」，应改为沃特伯恩威士忌 |
| 2026-08-06 | `The Last Pit` → **最后的矿坑**（此前记为「最后一坑」） | 与同组的 `The Empty Pit` 空矿坑 / `The Active Mine Pit` 在采矿坑 用同一个「矿坑」构词；「最后一坑」的量词读不顺 |
| 2026-08-06 | `Bright Lord` → **辉耀领主** | 沿用 v1.0.15 旧译文里的写法（该角色只在书信中被这样称呼）。`For Other Fortunes` 则保持英文原样，与战役包已译各页一致 |
| 2026-08-06 | 「没有可译文本」的条目（整条都是 `@Embed`/`@UUID` 之类）不计入覆盖率分母 | 它们永远不可能含中文，计进去等于让每个包都永远显示未完成，并把死条目塞进驱动翻译批次的待译清单。全库 298 条 |
| 2026-08-06 | `[[/item 中文名]]` 属**正常**，不算标记被译坏 | dnd5e 的 `/item` 按角色身上的物品名解析，而 babele 已经把那些名字翻成中文了 —— 此处英文反而会失配。`readaloud="…"`、`[[/r …#掷骰说明]]` 同理 |
| 2026-08-06 | 孤儿译文是否移植，看**结构是否逐段对得上**，而不是看有没有译文 | `Arcturel Upper` 28 页 `<p>` 238/240 与今天的英文一致 → 移植；若像阶段 13 那批缺 2200 个 `<li>`，移植只会制造「显示 100% 实则缺整块」的新债 |
| 2026-08-06 | 并行翻译：译者**自己**把 `apply_translations.py --dry` 跑到 0 拒绝才算交付；任何 agent 都不许写 `compendium/cn` | 标记类错误在返回主控前就被挡掉，主控不必逐条复核；落盘只由主控做，单个 agent 出问题只需丢掉一个 batch 文件 |
| 2026-08-06 | 术语冲突的判断依据强弱：**同名条目/物品的 `name` 字段 > 同卷已译页 > 全库多数写法 > glossary_ec > 交给译者的术语表** | 第 1 批三个 agent 各自独立查出我给的术语表有三条（Inkaro/Amalthea/引号）与全库不符，按既有译文处理是对的。术语表是二手摘录，会抽错 |
| 2026-08-06 | `Attunement` → **同调**（战役包 418 处调谐一并改掉） | lang/cn.json 与 character 小包早已是同调，玩家在角色卡上看到的就是它；正文与 UI 不一致比选哪个词更糟 |
| 2026-08-06 | `Inkaro` → **因卡罗**，改 4 个物品条目名而不是 126 处正文 | 统一方向选「改动面小的那边」；统一后 `@UUID` 标签与物品名自然对齐 |
| 2026-08-06 | 引号一律 **“”** | 全库 2618 : 44，44 处是后来一批按风格说明写的，与存量不符 |
| 2026-08-06 | 孤儿页面用**标记指纹**配对，不用页名 | 页名恰恰是上游改掉的那个东西；而 `@UUID`/`[[/…]]` 会被原样抄进译文，是天然指纹。`In The Behemoth's Wake` 只改了大小写就成了孤儿，指纹相似度 0.99 |
| 2026-08-08 | 第 8c 项与第 8j 项**合并**为「页面重对齐」，不分两批做 | 两张清单有 32 条路径重叠（上游改写过的页），分开做会让两份 batch 争同一条 path、落盘时静默互相覆盖；而闸门是多重集**相等**比较，本就同时管两个方向 |
| 2026-08-08 | 并行单元改用**页文件 + Edit 局部改**，不再交整页替换值 | 按「缺失内容」切会把单元规模低估 5.6 倍（26 万 vs 145 万），是阶段 23 fill-8 被掐断的根因；且整页重写会把已校对的译文洗掉。新格式下「没动过的字节保持没动过」是格式保证的。副作用：额度耗尽从事故降成普通中断 |
| 2026-08-08 | `fill_twin.py` 的 TM 键用**结构路径**，不用最后一段路径 | 只取最后一段会把 `items.X.name` / `items.X.actions.<id>.name` / `effects[].name` 三种惯例混成一堆 —— 正是阶段 23 警告过的错。改用结构骨架后歧义键 697→530 |
| 2026-08-08 | 先落第 6 批，**再**跑孪生包 TM 填充 | 被弃填的 554 条里 66% 正是第 6 批在修的 8c/8j 页；顺序反了少填 421 条 |
| **2026-08-09** | **术语与前后不一致由主控自行裁决并统一，不上报项目所有者** | 项目所有者明示。此前「不静默择一、列进 `glossary_ec.disputes.json` 待裁决」的做法降级为**仅证据不足时**适用。裁决仍走既定依据阶梯，且必须写进本表 + 用 `unify_terms.py` 执行 + 复跑 QA |
| 2026-08-09 | `Electricity` → **电击** 再次确认；`Lightning` → 闪电，两者**不是同一个词** | 库内 电击 97 / 电能 48 / 电力 8；crucible `lang/cn.json` 里一个含「电」的键都没有，故按决议＋库内多数。**2026-08-12 复核已归零**：英文写 `Electricity` 的条目里中文全是「电击」；库内残下的「电能」25 / 「电力」8 译的是 `electrical energy` 等别的英文，不是残留 |
| **2026-08-11** | **推翻 08-09 的 `Reaper Ocean`→收割者海洋，改回「劫掠者海洋」** | **我当时判错了。** 上游对同一片海有两种拼写：`Reaver Ocean`（多数、正规）与 `Reaper Ocean`（少数几处，**是拼写错误**）。`Reaver` 就是劫掠者，原译「劫掠者海洋」本来就对。我当时只按 `\bReaper Ocean\b` 取样，没查 `Reaver`，还反过来推理「劫掠者对应 Raider」，把对的改成了错的。全库实测 劫掠者海洋 31 : 收割者海洋 6（后者全是我改出来的），已改回。**教训：定名前先查这个专名在英文侧有没有异体拼写，只按一种拼写取样会取到偏样本。** |
| 2026-08-09 | 库内 167 处「闪电」中，**123 处不动** | 那 123 处英文确实是 `Lightning`（忠实翻译），另有 44 处是「闪电般迅捷」这类比喻。`Rune: Lightning→Storm` 的改名只作用于**英文写 Storm** 的地方。**先查英文再判中文**，否则机械替换会误伤大片 |
| 2026-08-09 | `Electricity` 的「电能」替换加 `unless: electrical energy` | `Rune: Storm` 引导句是 `The chaotic force of raw **electrical energy**`，中文「原始电能的混沌之力」翻的是这个散文短语，而同条里的伤害类型 `deals **Electricity** damage` 中文**本来就是**「电击」。评审抽查抓到的；不加守卫会把正确译文改坏（ember 5 处、crucible 全部 7 处都属此类） |
| 2026-08-09 | `Concluding the Event` → **事件结束**；`Event Outcome` → **事件结果**；两者互加 `unless` | 是两个不同的词组，但在同一条目里大量共现（`Event Outcome` 的 61 条里就有 134 处「事件结束」）。不互相排除就会把对方的译名改坏。同时出现的条目一律跳过 —— 宁可漏改不可错改 |
| 2026-08-09 | `Marlstone Manor` → **马尔斯通庄园**（变体写全「马尔石庄园」而非裸「马尔石」） | name 字段即马尔斯通庄园，英文对得上的条目里 161:22。裸「马尔石」另有街区名之用，替换会误伤 |
| 2026-08-09 | `Fernis Ossa` → **费尔尼斯**；`Horrendor` → **霍伦多尔**；`Yakoshta` → **雅科什塔** | 均以 name 字段为准（204:50、82:13、6 处 name 对 2）。**「惊惧者」是另一个 actor `Harrower` 的名字**，挂在 Horrendor 上属挂错名，`glossary_ec` 里那条也是错的，已改 |
| 2026-08-09 | `Young Cheliceraeth` → **幼年螯蛛艾斯**，**不取 name 字段** | 这是依据阶梯的例外：孤立的 actor name「幼年螯蛛以太兽」只有 8 处，而 archetype name「螯蛛艾斯」、macro「切换螯蛛艾斯」、正文 44 处三方一致。改这一处 name 比改 44 处正文便宜（「改动面小的那边优先」） |
| **2026-08-12** | **复查发现：缺陷表里标 ✅ 的项必须拿数据验，不能照抄上一版状态** | 本轮我一度把 C/D/E/J 四项直接誊成 ✅，实测其中 **E 与 J 根本没做**（disputes 20 条积压、地图针脚 14 个仍是普通词），而 C/D 又被我误报成"没做"（裸计数 电能 25/阶位 160 看着像残留，加英文闸后全是别的英文：`electrical energy`、`Rank`）。**结论：术语类结论一律要带英文闸的计数，裸词频既会漏也会误伤。** |
| **2026-08-12** | 20 条积压 disputes 一次裁完，依据阶梯＝name 字段 > 同卷已译页 > 全库多数 | Agrimage→农艺法师(284:180)、Thornling→荆芽灵(313:148)、Aberin→阿伯林(216:31)、House Cevher→杰夫赫尔(913:6:4)、Ordain→奥尔丹(3062:225)、Arcturian→阿克图里安(248:3)、Reliquary→圣髑匣(29:10)、Wind Raider→风袭劫掠者(26:2)、Hulg'run→赫尔格伦。**注意**：库内的「念动力」译的是 `Telekinesis` 而非 `Kinesis`，两者是不同的词，disputes 里那条已过期 |
| **2026-08-12** | `essence`→**精华**；`Stride`→**步幅**；`maximum Action`→**最大动作** | essence 132:8，同一个英文、无术语/散文之分；Stride 67:28 且 lang 的 `ACTOR.StrideSpecific` 就是「步幅 {stride}」（英文闸必须**区分大小写**，小写 stride 是动词，「所跨步踏过的土地」那处不能动）；maximum Action 按 lang 的 `RESOURCES.Action`＝动作 |
| **2026-08-12** | `The Waterworks` → **水务工程**，与 `The Waterworks Office`（水务办公室）分开 | 是两个地点：前者是城区地下的运河隧道迷宫，后者是那栋楼。中文有 3 处把前者写成了后者，玩家按指示去「水务办公室」会走错地方。取地名志 landmark 名 水务工程 |
| **2026-08-12** | 两处**中文与英文不符**的实质错误 | ① `Wedgelands` 页中文凭空多出「例如切夫赫尔庄园（由同名商会所有）以及」一句，英文只有 Corpin Sanctuary，已删；② `Supplies and Demands` 的旁白整段被重排，且把 `thornling` 译成「农法师」——埃迪维尔是荆芽灵不是农艺法师，还把台词「幸好它是附了魔的」改写成了旁白「若不是埃迪维尔施加的魅惑」。已按英文重译。**两处的块数多重集都与英文相等，所以 BLOCK 检查一声不响 —— 块数相等不等于顺序与内容正确** |
| **2026-08-12** | 两个被截断的页名 | `欢迎来到 Welcome To Crucible`（中文半截是空的）→「欢迎来到 Crucible」；`什么是《 What is Crucible`（残留孤立书名号）→「什么是 Crucible」。库内系统名一律保留英文 Crucible，中文里已含该词，故不再缀英文尾巴 |
| **2026-08-12** | **`lang/cn.json` 一律写扁平点号键，禁止按点建嵌套** | Foundry 的 `getProperty` 先试整键、再按点下探。顶层带点又嵌套的混合形态**两条路都断**：ember 486 键里 372 个（77%）静默失效。`apply_lang.py` 的 `set_path` 当时就是按点建嵌套的，等于每写一条新译文就重造一次这个坑（Boon 统一批亲历：新值成了嵌套副本，与旧扁平键并存，UI 一个字没变还毫无报错），已改为 `root[dotted] = value` |
| **2026-08-12** | **校验必须复刻被验证系统的查找语义** | 上面那个缺陷之所以活了这么久，是因为校验脚本自己会递归展开嵌套，于是一路报「486 键缺口 0」。`lang_gap.py` 现在有 `foundry_lookup()` 与 `UNREACHABLE` 一列 |
| **2026-08-12** | `Boon` → **恩惠骰**（原 恩惠 57 / 惠骰 51 / 恩惠骰 4 三写并存） | lang 的 `DICE.Boons` 就是这三个字，玩家每次掷骰都看得到，且与 `DICE.Banes`＝祸骰对称；`惠骰` 在 lang 里没有任何锚点。**dnd5e 侧例外**：`ember.adventure.json` 里 boon 多是普通英文名词（metaphysical boon / the boons associated with…），只把 system-swap 嵌进来的 4 处「+N Boons」改掉 |
| **2026-08-12** | `Fortitude` → **强韧**（`Toughness` 保持 坚韧） | 缺陷表 B 项的根治。两者共用「坚韧」二字时，正文 120 处裸「坚韧」无从分辨指属性还是指防御，而 Willpower 的公式 `(坚韧+存在)/4` 用的正是属性 |
| **2026-08-12** | `Willpower`→**意志**、`Accurate`→**精准**、`Arrow`→**箭矢** | 意志 79:13；精准是 lang 自己的 `AccurateTooltip` 与 rules 攻击标签表的写法（`ACTION.TAG.Accurate` 的「精确」是同一文件内的自相矛盾）；箭矢是运行时按 `SPELL.GESTURES.Arrow` 渲染出的法术名，且中文的「箭头」指箭镞/光标 |
| **2026-08-12** | lang 标签：`Vocal` 声乐→**言语**、`Auditory` 听觉的→**听觉**、`Mechanical` 机械的→**机械**、`ARMOR.PROPERTIES.Natural` 自然→**天然**、`WEAPON.TAGS.Natural` 自然→**天生** | 声乐是误译（Silenced 页正文写的就是「言语标签」）；其余 40 余个 `ACTION.TAG` 一律是不带「的」的名词，那两个是仅有的例外；两个 Natural 分别对齐同文件的 `ARMOR.CATEGORIES.Natural`＝天然 与正文的「天生武器熟练度」 |
| **2026-08-12** | **抽取器的 `textCollection` 一律按 `text` 建键，不能用 `_id`** | Babele 的 `textCollection` 就是 `fieldCollection("text")`，运行时查 `translations[data.text]`。抽取器写成 `it._id ?? it.text`，于是英文基准按 ID 建键、中文包按文本建键，300 条**正在正常生效**的地图针脚译名被判成死键，`prune_dead` 差一步就整批删掉。**清理类操作前必须先验证「死」的判据本身没错** |
| **2026-08-12** | 数字覆盖检查必须认中文数字与倍数量词 | 只认阿拉伯数字会逼出坏中文：库里真出现过「2 个十年」（2 decades），历史上还逼出过「3 层矿井」「第 1 军团」。中文本来就该写 三/两/第一/二十年 —— 是规则错了，不是译文错了。`decade/dozen/score/century` 这类量词换算后的值也算可接受写法 |
| **2026-08-12** | 上游把 `{Persuasion}` 裸写在正文里，改英文基准而不是改译文 | 它前面没有 `@UUID[…]`、也不在任何 enricher 里，Foundry 会原样渲染花括号。中文写「魅力（游说）」比上游还正确，不该为迁就 PLACEHOLDER 那条正则改成「魅力{游说}」。记入 `LOCAL-PATCHES.md` 第 2 条 |
| **2026-08-12** | 新定专名：`Mial Mountain` 米亚尔山 / `inkaro pearl` 因卡罗珍珠 / `Dusk Hound` 暮猎犬 / `Winged Scavenger` 有翼食腐者 | 补最后 72 条纯缺译时定的，库内此前无任何写法。其余专名一律先查既有译法再落笔 |
| 2026-08-09 | 四个 `Vista:` 场景 name 定名 | `Ordain Streets` 授命街道→**奥尔丹街道**（同一份场景清单里 `Ordain Overview`/`Ordain Interiors` 都作奥尔丹，「授命」是把 ordain 当动词的机翻）；`Yakoshta`、`Arbore Sanctorus` 原本整个没译 → **雅科什塔** / **圣树庇护所**（后者取自既有 name 字段）；`Ordain Interiors` 室内装潢→**室内景**（interiors 指室内场景，上下文是「舒适休息室」「神秘者公寓」这类构图，不是装潢） |

### 阶段 28–29（2026-08-12 第二轮）新增决议

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-12b** | **`Arcturian` 维持「阿克图里安」**，1090 处「阿克图里亚」全部统一过去 | 08-12 上一轮记的比数（248:3）与实况不符，本轮重裁。全库英文闸下 阿克图里亚 1090 处/557 叶 : 阿克图里安 248 处/181 叶，**但 `name`/`tokenName` 字段是 40:16（14 个条目名 : 6 个）**，且 `Cultures/Arcturian` 页 name＝「阿克图里安 Arcturian」、同页 pronunciation 自写「阿克-图里-安（ark-TOOR-ian）」。依据阶梯 name > 同卷 > 全库多数，多数派在此让位。英文侧另有两种拼写 `Acturian(s)` 8 / `Arcturains` 2（上游漏字），闸要一并覆盖 |
| **2026-08-12b** | **`Warden`（宗教/文化义）→ 新定「典守者」**，prison warden 保留典狱长 | 英文闸 52 叶里 35 处确为狱政语境。宗教义（`Wardens of Flame` 章标题、`the Flame Warden` 卢梅尊号 ×13、`Wardens, Priests, Sorcerers`）需另名，而 守护者=Guardian、守望者=Keeper（68 叶中 51 叶）、守林者=Warden(archetype) 三个都已被占用。「典守」＝保管掌管之职，对应 Warden 的 custodian 义，全库 0 次使用无碰撞 |
| **2026-08-12b** | `Sockets`（上古死神）→ **萨克茨**（音译），改 390 处 | 原译「插孔」是把专名当普通名词机翻。同页 `pronunciation` 英文 `SAH-kets`、中文早已写「萨-克茨（SAH-kets）」——库里本来就知道它是专名。英文小写 `sockets`（眼窝）3 叶被闸排除 |
| **2026-08-12b** | `Shard God`（通称）→ **碎片之神**；`Shard Goddess` 才作碎片女神 | 原译一律「碎片女神」，造成「碎片女神贾纳尔…**他**所斩杀」。英文侧 `Shard God(s)` 826 处 vs `Shard Goddess` 127 处，是两个词。308 叶只出现 God → 整叶替换；48 叶只出现 Goddess → 不动；13 叶混合 → 逐处按英文语序改。「碎片之神」是库内既有写法（13 处） |
| **2026-08-12b** | `Steading`（历法季节名）→ **耕耘**，不是「庄园」 | 页内自述 `Season of Industry`、「人们以劳作与产出为乐」，与另五季 播种/绽放/拾取/凋零/寂止 同为二字名词。「庄园」在本库已被 `Grange`/`Manor` 占用。**三处必须同改**：compendium、`lang/cn.json` 的 `EMBER.CALENDAR.SEASONS.STEADING`、`scripts/ember-hardcoded-cn.mjs` 的 `CALENDAR_MONTHS`——世界时钟读的是最后那张表 |
| **2026-08-12b** | `Aura`（月亮专名）→ **奥拉**（音译）；手势 `Gesture: Aura` 保持「灵气」 | 同一份月亮清单里 Mayis 玛伊斯 / Cora 科拉 / Ragen 拉根 / Orbis 奥比斯 / Akon 阿肯 全是音译，只有 Aura 意译；且「灵气」正是 `Aura Spellcraft` 的 adjective，月亮与施法构件同名重演了 Inflection/Affix 撞名。按英文逐处分类：月亮标记改，手势与小写 `aura`（Fear Aura）不改 |
| **2026-08-12b** | `Ordinate` → **审序院**（弃「法序议会」）；`The Hallows` 城区＝**圣堂区**、组织＝**幽圣所** | 都按 name 字段（最强证据层）：组织页 name＝审序院 Ordinate（全库 369:149，UUID 标签 31:1）；城区页 name＝圣堂区、组织页 name＝幽圣所，是两个实体各有其名 |
| **2026-08-12b** | `River Destine` → **德斯廷**（音译，弃「天命河/命运河」） | 该河无独立条目、无 name 可仲裁。英文闸下 德斯廷 38 : 德斯汀 28 : 天命河 11 : 命运河 4，按「专名音译」惯例取多数音译式 |
| **2026-08-12b** | `Elvish` / `Orcish`（祖裔）→ **精灵 / 兽人**，去掉「语」字 | 同一份祖裔清单里 Dwarven 矮人 / Giantkin 巨人裔 / Devilkin 魔裔 / Human 人类 均无「语」字，而 **crucible 系统根本没有语言这个概念**（`crucible.ancestry` 是 Item 类型 ancestry）。角色卡的血统栏此前会显示「兽人语」 |
| **2026-08-12b** | `Electricity Resistance` → **电击抗性**（原「电抗性」） | 同族 11 个 `X Resistance` 全是「<伤害类型>抗性」，只有这条例外；且「电抗」在中文是电工学的 reactance，是另一个词。adjective 的缩写「抗电」与同族 抗火/抗寒/抗毒/抗酸 一致，不动 |
| **2026-08-12b** | `Arcturel Upper` / `Lower` 的双语英文尾巴按现行英文改为 `Tradeway` / `Dives` | 上游早已改名，`Arcturel Upper/Lower` 在今天的英文基准里一个字都不存在——译文却把这两个**已不存在的名字**写在双语并列的英文侧。这是 `port_orphans` 只搬路径不改内容的遗留（第 5.4 节的警告） |
| **2026-08-12b** | `Critical Success/Failure`：`.mjs` 一律服从 crucible `lang` | 英文闸 大成功 74 : 重大成功 15。`.mjs` 文件自己的注释就写着「译名与 crucible lang 逐条对齐」，所以按 lang 改（大成功 / 严重失败）。**遗留**：lang 自身 `大成功` 与 `严重失败` 不对称，是否统一为 大成功/大失败 待定 |
| **2026-08-12b** | **并行批次落盘前必须做三方合并**，不能靠调整落盘顺序 | 16 路 agent 同时基于同一 base 产出 59 个批次，而 `apply_translations.py` 是**整叶覆盖**。实测 350 条路径被 1 个以上批次认领——按任何顺序直接落，都会有一批修复被静默回滚。工具与做法见第 1 节 |
| **2026-08-12b** | `_` 前缀键是**有意寄存**，`prune_dead` 不得删 | `validate_translations.py:97` 明确跳过 `_` 前缀，`migrate_cn_schema.mjs` 把抢救不了的内容 park 在 `_legacyActions` 等键下「留待后续人工抢救」。`prune_dead` 学会下钻数组后第一次报出这 8 条，它们不是死键 |
| **2026-08-12b** | `prune_dead` 删数组元素时**只做尾部截断** | 删掉数组中间的元素会让其后每个元素索引前移一位，等于把一批**正在生效**的译文整体挪到别人的键上——比留着死键坏得多 |
| **2026-08-13e** | **`tokenName` 是独立的一档，必须单独设闸** | 玩家在地图上直接看到的是 `tokenName`，比角色卡上的 `name` 还显眼，而它**不在任何既有判据的配对范围里**（`scan_label_vs_name` 只比 `@UUID{标签}`↔`name`、`scan_name_binding` 只比表结果/针脚、其余闸只看单个叶子自身）。首测：英文侧 `name==tokenName` 的 481 个 actor 里 **32 个中文不一致**，其中 `Thayloc Courser` 的 tokenName 是「非玩家角色」（占位符没删）、`Sporix Host` 是「受折磨的荆棘幼体」（完全另一种生物）、`Raster Thorn` 是「栅格棘刺」（**name 已按 2026-08-13c 改成「拉斯特·索恩」，token 没跟上**）。判据必须取严格版：只在英文侧逐字节相同时才要求中文一致，否则会误伤作者有意的短称（`Kalasak the Cutter` 的 token 就叫 `Kalasak`） |
| **2026-08-13e** | **「同页多数是英文」不能直接当作「该页有意保留英文」** | `Patch 0.4.7` 一页 13 中文 : 15 英文，四个单元据此判「整页 deferred」，只有 B3 主张补。**B3 对**：那 15 个英文里有真实人名（`Jess Levine`）与第三方产品名（`going rogue 2e` / `PLANET FIST`）——**本来就不该译**，把它们计进分母就把比例算反了。判「该不该译」要先按**类别**拆（本作内容名 / 外部专名 / 真实人物），再看同类的比例。补全后该页 13 个 `<em>` 里 10 个中文，剩下 3 个正是那三个产品名 |
| **2026-08-13d** | **补 Scene 的三个字段，用内建 `nameCollection` 而不是自定义转换器** | 项目所有者定「优先完整汉化，补管线」。`levels` / `tokens` 都是「带 `name` 的对象数组」，正好是 babele 内建 `nameCollection`（＝`fieldCollection("name")`）的形状 —— **它比 FR 的自定义 `ember_scene_levels_converter` 更安全**：查不到译文原样返回，查到也只 `mergeObject` 掉 `name` 一个键，`_id` / `bottom` / `top` / `flags` 全保留，对文档数据非破坏。`navName` 是标量，写 `navName: 'navName'` 即可。**放在注册层（`SCENE_LEVELS`）而不是 `BABELE_DEFAULTS`** —— 后者是 babele 默认值的忠实副本，babele 自己没有这三个字段，副本里也不能有；层的合并是**按字段**的，所以写一个键不会顶掉默认的 name/drawings/notes/regions |
| **2026-08-13d** | **重抽英文基准必须走「只并新字段」，不能整份覆盖** | `LOCAL-PATCHES.md` 里记着对上游英文笔误打的本地补丁（`@Condition[exhaustion` 缺右方括号，不补就有条目永远过不了闸门），整份覆盖会**静默回退**它们。本轮实测：加完三个字段后重抽，相对现有基准只多 525 个键、**0 删除**，另有 3 条（×2 包）差异**正是那些本地补丁**。所以写了 `4-临时脚本/2026-08-12-audit3/merge_levels_into_en.py` 只搬新字段。合并后复验：新增 0 / 删除 0 / 变化 6（＝补丁，有意保留） |
| **2026-08-13c** | **孪生包对齐要放在最后一步，方向取主线** | `propagate_fix.py` 只推增量，历轮审计只覆盖当轮动过的叶子，**存量分叉从来没被系统清过**：实测两包 11359 条共有路径里有 **484 条英文逐字相同而中文不同**、18 条一侧没有中文。方向取 crucible 侧（主线）—— 抽样是决定性的：孪生包里还留着「插孔」这种早已裁掉的 `Sockets` 机翻、「档案馆」这类已被修正的写法。**顺序很重要**：主线每落一批就会制造新分叉（本轮 286 → 502），所以 `sync_twin_packs.py` 必须是最后一步 |
| **2026-08-13c** | `scan_label_vs_name` 的裁决**有 41% 是「改 name」** | 153 组配对逐组判方向：改标签 88 / **改 name 63** / 不动 2。这个比例说明「name 字段是最强证据」**不能机械套用** —— 得先看这个 name 本身译得对不对（是不是把专名当普通词逐词机翻）。三个实例：`Raster Thorn`→栅格荆棘（他是碎牙帮帮主，英文闸 86 行里 拉斯特 75 : 栅格荆棘 2）、`Wandren Patroller`→巡逻者旺德伦（**同一 actor 自己的 tokenName 已是「万德伦巡逻者」**）、`Oakengarde`→橡木卫队（地区名被逐词机翻） |
| **2026-08-13b** | **单元制审计会漏掉「单元之外」的缺陷，主控必须接住** | J05 读 `Smoldering Cinders` 时**顺带撞见**并如实上报了一条阻断（`Ordain Gazetteer / Scholar's Nook` 的 landmark `<dt>` 名整体错位一格，玩家按地名志找任何一家店都会走错），但 `Ordain Gazetteer` **不在那 35 本的清单里**，超出它的 scope，所以第四轮**没有任何人修它**。教训：`confirmed` 里 pack/path 落在本单元范围之外的条目，主控要单独接单，否则它会被「已确认」三个字埋掉 |
| **2026-08-13b** | 地名志 `<dt>` 的 NPC 名有 70 处整个没译 | 顺着上一条查下去发现的第二类：432 个带对齐标记的 NPC `<dt>` 条目里，**70 处（22 叶、35 个唯一人名）中文名仍是英文原名**。**所有现有闸门全盲**：叶子里有大量中文，覆盖率算它已译；名字是专名，数字覆盖看不见；标记与 class 都没动，标记五项也不响。判据是「`<dt>` 英文形如 `Name (Alignment, …)` 而中文的名字段仍以英文原名开头」，见 `4-临时脚本/2026-08-12-audit3/fix_gazetteer_dt.py` |
| **2026-08-13b** | 新定专名：`The Crooked Spine` 曲脊巷 / `The Gadrick Estate` 盖德里克宅邸 + 35 个地名志 NPC 名 | 前两个是 `Scholar's Nook` 错位修复时补的（库内此前无任何写法，`<dd>` 正文全部对得上、只有 `<dt>` 名错位，锚点是 `A Message from Sin` 页把 `The Secret Shelf` 译作「秘藏书架」）。35 个 NPC 名逐个查过库内无既有写法后音译；**`Vijin Barriq` 与 `Vujin Barriq` 是两个不同的人**，译名区分为 维金 / 武金 |
| **2026-08-13b** | **逐句通读之后必须再做一遍机械核查** | 27 个单元逐句读完 35 本 journal 报了 507 条，而挂在后面的对抗式复核 agent **又补出 188 条**（平均每单元 7.5 条，单次通读召回率约 73%）。补出来的几乎全是两类：① **同一叶里改了 A 却漏了 B**（同一段里刚把「下层阿克图瑞尔」改成「矿渊」，却漏了同段的人名「凯伦→凯兰」）；② **`@UUID` 标签 ≠ 目标文档的 `name` 字段**。第二类**完全可以机械化**：把该卷所有 `@UUID` 标签与目标文档的中文 name 全量对照即可，比再读一遍便宜得多。**下一轮该先跑这个再读。** |
| **2026-08-13b** | **`Critical Success`→大成功、`Critical Failure`→严重失败，取 lang 不动 lang** | 了结 2026-08-12b 留的那条「大成功/严重失败 不对称是否统一为 大成功/大失败」的遗留。英文闸下 compendium 本来是分裂的（大成功 74 : 重大成功 15 : 严重成功 9；严重失败 7 : 大失败 6），而 lang 的 `ACTION.EFFECT_RESULT_TYPES.*` 就是「大成功 / 严重失败」。**改 lang 会让玩家每次掷骰看到的字变掉，改动面与风险都更大**，所以按「改动面小的那边优先」把 compendium 对齐到 lang。统一后英文闸下 93:0:0 与 15:0 |
| **2026-08-13b** | **`propagate_fix.py` 必须按字段角色配对** | 它原本只按「英文逐字相同」配对，会跨 `name` ↔ `adjective` 传播，而这两个角色的格式约定不同：`name` 是双语并列「辉耀 Luminary」，`adjective` 是拼装用的裸中文（`{prefixes}{name}`）。推过去运行时会拼出「辉耀 Luminary长剑」，反方向则把 name 的英文尾巴剥掉。crucible 侧 10 条候选里 **7 条**是这种跨角色配对，加闸后剩 3 条 |
| **2026-08-13b** | 孪生包的修复走 `propagate_fix`，但**它只能推英文逐字相同的部分** | journal 单元只复核 `ember.crucible-adventure.json`，dnd5e 侧靠 propagate 推（本轮推了 581 条）。但两包**有一批叶子的中文已经分叉**，英文也不同（dnd5e 与 crucible 的规则文本本来就不一样），propagate 推不过去 —— 那部分的同类缺陷要单独查 |
| **2026-08-13** | **`scan_en_drift` 的 `stale` 桶是候选筛，不是判据** | 逐条读完 278 条（crucible 23 + ember 255），真缺陷约 15 条，**C1 单元 23/23、E1 单元 51/51 全是假阳性**。判据「中文长度按旧英文算更接近中位数 0.31」在**短叶子**上没有鉴别力：talent/action 描述普遍 100–350 英文字符，上游删一个从句或替一个专名就能让比值跨过中位数。而这批 drift 的实质内容恰是上游的机械化改造（裸粗体状态名批量改成 `@Condition[...]`、`<ul>`→`<ul class="complex-check">`、`Result of 17+`→`@CriticalSuccess[13]`），译文上一轮已经跟着改过。**有鉴别力的是「新英文独有的 token（enricher / 数字 / 专名）在不在中文里」，不是长度比。** |
| **2026-08-13** | **`Scene.levels[].name` 缺口取「正文回填英文」的安全侧，不改管线** | 两条路互斥：要么正文写英文层名（与今天的运行时一致），要么补 mapping 把层名也翻掉、正文写中文。后者要改 `mappings.mjs` + 写转换器 + 重抽基准 + 译 255 个名字，**且只有开世界才能验**，而前者跟随库内既有先例（`Over The Moon / An Ancient Door` 页本来就保留英文）、零风险、今天就正确。**决定权在项目所有者**，两条路的完整代价写在第 7 节第 13 项 |
| **2026-08-13** | **`scan_uuid_swap` 的 `en_label` 不可信，裁决前必须逐位对齐** | 它只取该目标在叶内的**第一个**英文标签，而同一叶对同一目标用两个英文标签是常态（`{Moiran}`/`{Blood Barons}`、`{Big Liz}`/`{Baradom}`、`{Draconic}`/`{Dragons}`）。仅这一步就把 22 条「看着像错」的判成不改 —— 包括 `Big Liz`→巴拉多姆（实测中文是「大莉兹/巴拉多姆」与英文一一对应，**本来就是对的**）。谁按 `en_label` 直接判就会把正确译文改坏 |
| **2026-08-13** | **`.mjs` 里三条阻断：31 个键此前根本不生效** | 逐条审 177 条时查出的**不是译文错，是机制不通**：① 事件状态浮窗写在 `data-tooltip-text` 上，而属性白名单只有 `data-tooltip`/`title`/`aria-label`（v14 的取值顺序是 tooltipHtml > tooltipText > tooltip）；② 15 个对话框标题 + 3 条 Attunement 正则走的是**原生 DialogV2**，被 `patchRenderedApplications` 的 ember 闸整个挡掉；③ `MOODS` 五个键全是凭空的，ember 只有 calm/tension 两档。**教训：审一张运行时替换表，先验「这个键还能不能被匹配上」，再谈译文对不对。** |
| **2026-08-13** | 表结果名 / 针脚名的**绑定关系不在译文文件里** | `entryId`/`documentUuid` 只存在于 LevelDB packs，Babele 译文文件里根本没有。所以两个一次性探针只能拿「英文标签逐字节等于某文档英文名」当代理判据 —— 实测对本轮 16 条真缺陷**多报 10 条、漏报 3 条**。`scan_name_binding.py` 从 packs 导出真实绑定后才能用 |
| **2026-08-13** | 属性白名单**先普查再定**，不要凭空拟 | 本库实际只出现 26 个属性名，当初凭印象拟的 `title` / `alt` / `aria-label` / `placeholder` / `data-label` **一个都没有**，而真正有可见文本的是 `@Embed[...]` 方括号体内的 `label=` 与 `readaloud=` —— 凭空拟名单会把这两个漏掉。ground truth 122 处、修前只译了 50 处（41%） |
| **2026-08-13** | `Install Junction Wheel` → **安装路口轮盘**（原「安装道岔轮」） | 又一次裸词频翻车：原依据是「道岔＝库内 28 处」，加英文闸后 `\bJunction\b` 命中 11 行、**道岔 gated_hit=0**（那 22 处「道岔」译的是 `Switch(es)`，见条目名「雅科什塔矿井轨道道岔」）。目标物品的 `name` 字段两个包里都是「雅科什塔路口轮盘 Yakoshta Junction Wheel」 |
| **2026-08-13** | 表结果行的中文**要与目标文档的中文 name 逐字相同** | 英文侧行名与文档名本来就逐字相同（`Spear` == `Spear`），中文侧不同名就是不一致。10 条已对齐（含 `长矛`→`矛 Spear`，条目 name 即「矛 Spear」）。双语并列的英文尾巴也要带上——玩家背包里看到的就是「毯子 Blanket」 |
| **2026-08-12b** | 数字覆盖改多重集后必须配三条宽容规则 | 只换多重集会让假警报暴涨（crucible 2→50、ember 0→374）。三条：块内邻近折叠（**只折叠英文侧**，中文长度只有英文 0.35 倍，同 window 在中文侧等于放宽 3 倍；且要认块边界，否则 `<td>3</td><td>3</td>` 被折成一份需求，真缺陷重新被掩盖）、单/双＝1/2、`A/B` 分母豁免（`patrolled 24/7`） |

### 第八轮（2026-08-13 第八轮）新增决议

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-13f** | **`Confused`（crucible 系统状态）→ 混乱**，compendium 的 4 处「神志混乱」对齐到 lang | crucible `lang/cn.json` 的 `ACTIVE_EFFECT.STATUSES.Confused` 与 `crucible.rules.json` 的条件页 name 都是「混乱 Confused」，而 ember 侧 2 个 ActiveEffect + 1 个表结果行写「神志混乱」—— 玩家在 token 状态条上看到「混乱」、在效果名上看到「神志混乱」。方向按 `Critical Success` 的先例（2026-08-13b）：**lang 的有效面比键数大得多**（每个 `@Condition[confused]` 都渲染它），所以把 compendium 拉向 lang。附带：`神志混乱` 其实是更准确的中文，但一致性优先。**新判据 `probes/scan_status_name.py` 已入库**：英文 `name` 恰为 crucible 状态名的叶子，中文剥掉双语尾巴后必须与 lang **逐字相同** —— 371 一致 / 4 不一致，全是这批。⚠ **判据必须严格相等，不能用子串**：「混乱」是「神志混乱」的子串，宽松比较会全绿 |
| **2026-08-13f** | **`Charged Bite` → 带电啃咬**（ember 侧 4 处「充能啃咬」改掉） | 英文 `bites with jaws that crackle with **storm energy**`；权威条目 `crucible.adversary-equipment::Charged Bite.name` 本来就是「带电啃咬 Charged Bite」，且它带完整 description。依据阶梯第 1 档（同名物品的 `name` 字段）。「充能」偏「蓄能/充电中」，与 storm energy 不符 |
| **2026-08-13f** | **`scan_name_splits` 剩下的 7 个里有 2 个是真缺陷，不是「全部合法分裂」** | 上一版文档记的是「剩下的 7 个是合法分裂」。逐个查实：`Confused` 与 `Charged Bite` 是真缺陷（已修）；另 5 个确为合法分裂并已各自留证 —— `Shield`（施法者身上是护盾术、战士身上与战利品表里是盾牌）· `Luminous`/`Spirited`（只存在于 `ember.character.json`，该包是 dnd5e 侧，与 `crucible.affixes.json` 永不同载，已核实 `ember.crucible-character.json` 里没有这两条）· `Color Commentary`（表结果 6-6 是**吟游诗人在战场上的解说**、journal 页是**壁画**，英文是同一个词的两个义）· `Arcturian`（`actors.Arcturian` 是「一个阿克图里安人」，文化/血统条目是「阿克图里安」）。**再次印证：标 ✅ 的项必须拿数据复验** |
| **2026-08-13f** | `@UUID{标签}` 首尾空格：**只改英文侧没有空格的那 6 处** | 全库 123 处标签带首尾空格，其中 **91 处英文侧也带**（上游就这么写的，忠实，不动）、26 处是我们清掉的（无害）、**6 处（3 组 × 孪生包）英文侧无空格** —— ` 深情告别`/` 金色平原`/` 鲁斯特瓦尔山谷`。这一条的价值一半在结论、一半在过程：主控第一版扫描器因 Windows 反斜杠把中文文件跟自己比，报「97 处全部忠实」，差点把 6 处真缺陷放过去（见方法教训 5） |
| **2026-08-13f** | `Spellbreaker Tower (Level N)` → **破法者之塔（第N层）**，内联 `@UUID` 标签一并跟改 | 塔的 Level N 是**楼层**不是等级，原译「（N级）」会让玩家以为是难度等级。g09 改了 7 个场景 name + `levels` + 区域名，但漏了正文里的内联标签，`scan_label_vs_name` 当场从 2 报到 4 把它抓住 —— **这正是该判据存在的意义：改 name 必然制造标签不同步** |
| **2026-08-13f** | 并行批次冲突：`Blast Flask` 两叶取 g05 的值（超集） | g04 只修了品质阶梯用词（`Fine`→精良），g05 在同一叶里同时修了品质用词**和**被漏译的 `who fail to avoid the attack` 从句。实测两者的品质用词完全相同，g05 严格包含 g04。g05 在自己的 escalate 里预先声明了这一点，属实。**这就是为什么落盘前必须三方合并** —— 直接按顺序落，g04 后落就会把从句修正静默回滚 |
| **2026-08-13f** | **接受 g11 给 112 叶的 `<hN>` 补 `id=` 属性，代价是 `scan_attr_text` 的属性名多重集从 0 变 113** | 问题：`@UUID[...#anchor]` 的锚点在 Foundry 里靠标题的 slug 解析，标题一译成中文 slug 就变了，234 处锚点链接全断。两条路：改锚点（234 处，且违反「方括号内照抄」铁律）或给标题补显式 `id`（112 叶）。取后者（改动面小 + 不动方括号）。**已逐个核实生成的 72 个 id 全部命中英文侧真实引用的 anchor**，包括带标点的 `garganthus-attack!` 与 `hallucinatory-terrain:-lava-floe`（上游确实那样引用）。⚠ **g11 自称「对三道闸全隐形」是错的** —— `scan_attr_text` 的属性名多重集会报 113 处 `gained {id}`。这 113 处是**有意的功能性添加**，与 `_legacyActions`、`lang_keep_english.json` 同类，后续轮次不要当缺陷重新裁。⚠ **它是运行时行为改动，只有开世界才能确认锚点真的跳对**，已加进冒烟清单 |
| **2026-08-13f** | `scan_uuid_swap` 的 UNCERTAIN 档**假阳性率约 97%**，不要再按它逐条改 | 第八轮把 34 组唯一情形按文档要求的「逐位对齐取本处真实英文标签」重判，**34 组里 33 组是假阳性，只有 1 组（`Mazira` 音译二分）存活**。判据把四个不同的人（艾梅琳·阿沃达/埃奥拉斯·哈斯威克/卡兰德拉/霍布·科雷尔）全报成同一个 `en_label`，一眼可见 `en_label` 张冠李戴。**结论：这一档已经压到极限，投入产出比很低，下一轮不必再做**，除非 `scan_uuid_swap` 本身改成逐位对齐 |

### 第九轮（2026-08-13）新增决议

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-13g** | **新判据 `qa/scan_same_en_split.py`：同一条英文串在全库只能有一种中文** | 三重盲区的交集：`sync_twin_packs` 只比两包**同一路径**、`propagate_fix` 只推 `--since` 增量、`scan_name_splits` 只比 `name` 字段 —— 而**同一英文串出现在不同路径上**（`Gesture: Influence.description` 挂在 23 个 actor 身上、23 叶 22 种中文）从来没人比过。首测 **1514 组 / 8308 叶**。**判据只报分叉、不给建议**：实测多数派常是错的那边（`Light Weapon Training / lunge` 的 **16 份**写「使用一把具有灵巧属性的武器」，英文是 `scales using Dexterity`＝以敏捷成长）。**分档看**：>300 字符长正文的分叉几乎必是缺陷；≤20 字符短标签默认当合法分叉（`Shield` 法术/装备） |
| **2026-08-13g** | 统稿选基准的阶梯：**独立条目 > 定译正确 > 无裸英文 > 忠实英文 > 多数派**，且**定译正确性压过票数** | 实测三类真缺陷都出在「多数派」那一侧：`Rune: Frost` 的 12 叶多数派写「寒霜符文」（权威物品名是**霜冻**，全库 57:16），还顺带发明了不存在的单位「码尺」；`Rune of Kinesis` 的 15 叶里 9 叶整段留着 `Kinesis`/`Propel` 未译。**顺序里有一处反直觉**：`Lunar Shield` 出问题的恰恰是**独立条目**（`ember.crucible-affixes.json` 写「Ember 诸元素之月」留了裸英文），内嵌副本反而对 —— 所以「无裸英文」压过「独立条目优先」 |
| **2026-08-13g** | **`Presence` → 存在**，清掉「风采」15 处 / 「风范」12 处 / **「感知力」** | 前两个是同义改写，第三个危险得多：**`Wisdom` 的定译就是「感知」**，把 `Presence` 译成「感知力」等于把两个属性混成一个，玩家算不出 `(坚韧+存在)/4` 那类公式 |
| **2026-08-13g** | **泛指句不能写死成具体属性**（`Dustbinder` / `Mesmer` 两处） | 英文 `deals half the ability score your <strong>Rune of Earth</strong> scales with` 是**泛指**「你的大地符文所依据的那项属性」，3 叶（含权威条目 `crucible.talent.json`）写死成「感知」，同时把 `<strong>` 从「大地符文」挪到「感知」。`Mesmer` 那条**四个变体全错**，无一可选，按依据自行写出正确版本。若日后符文改按别的属性成长，写死的那几份直接是错的规则文本 |
| **2026-08-13g** | **`<strong>` 数量相等但位置错了，所有既有闸门全盲** | `Grave Mark` 的权威条目把 `<strong>` 从 `Rest` 挪到了 `Turn`（英文里 `Turn` 不加粗），总数仍是 3；`A Laughing Matter` 四个变体的 `<strong>` 都是 5 个，但只有 1 份加在英文实际加粗的那五处。`scan_markup_drift` 与 `apply_translations` 的闸都只比**多重集**，位置错位一律放行。**这是一类新的已知盲区**，目前只有同源串对读能发现 |
| **2026-08-13g** | 三个判据收口，都做了**双向**灵敏度回测 | ① `scan_bare_english_names` 排掉三类假阳性（双语并列语序倒装 / 出版商名 `Mage Hand Press` / enricher 方括号内），37→20→（修完）**0**，注入 13 处真缺陷 **13/13 报出**；② `scan_attr_text` 加 `id` **净增加**白名单（净减少仍报），113→**0**，`--strict-id` 可复现 113 证明没丢数据，并合成注入验证了净减少与非 id 属性两个方向；③ `scan_uuid_swap` 改**逐位对齐**，68→**6**。⚠ **③ 有代价**：消失的 60 处里**有一处是真缺陷**（`Sunalins` 三分，改后多数占比 0.50 < 阈值 0.6 而不再报）。**通则：改判据降噪后必须逐条核对消失的那些里有没有真缺陷** |
| **2026-08-13g** | `Signaran` → **西格纳兰**（区别于 `Signara` 西格纳拉）；`Activating Alerts` → **启动警报** | 前者是**两个不同的英文词**，中文本来就在区分（`Signara` 31 叶作西格纳拉、`Signaran` 作西格纳兰），只有 `Vortest Tower / Tower Antechamber` 一处混了。旧口径按目标 id 统计多数，反而判它正确 —— 这是「逐位对齐」修好判据后新捞到的。后者 4:2，且 `Activating`＝启动，「触发」是 trigger |
| **2026-08-13g** | `Marlstone` 复合专名统一到 **马尔石**（141 叶） | 城区与石材本来就是马尔石（gazetteer 页 `name`，且正文明说城区得名于该石材，「铺着马尔石」靠「石」字成立）；8 个复合专名原作「马尔斯通」。取词根一致，代价是改 141 叶 |
| **2026-08-13g** | 上游笔误补丁只打 `compendium/en`，**`english-baseline/` 快照保持上游原样** | `Yakoshta Mine / Elevator` 的 `[[/skillCheck athletics 15 check.` 缺 `]]`，会让 `INLINE_CMD` 从这个 `[[` 吞到下一个 `]]`，**任何译法都必被判 markup mismatch**。补成 `...15]] check.`（形态取自同一 `<sup>` 里的 dnd5e 兄弟节点）。已记为 `LOCAL-PATCHES.md` 第 3 条。惯例与前两条补丁一致：`en` 里是修好的、快照里仍是坏的 |
| **2026-08-13g** | ⚠ **带 `suggested` 的判据，落盘前必须先过一遍定译表的英文闸** | 旧 `scan_uuid_swap` 对 `Shard God` 的建议是「碎片之神 → 碎片诸神」，而定译表写死 `Shard God`＝**碎片之神**。**判据在建议一个违反定译表的改动** —— 谁照它的 `suggested` 批量改就会把 30 叶正确译文改坏 |
| **2026-08-13g** | **知识领域 7 词三通道裁完并已执行**（`lang/cn.json` + `ember-hardcoded-cn.mjs` 各改 6 键，compendium 侧走批次） | 第八轮曾建议「一律按 lang 拉齐」，**第九轮推翻**：新发现**第四个通道** —— `crucible.rules.json :: Character Mechanics.pages.Background.text` 的背景示例表也逐格列了知识领域名，`Forensics` 在两张 compendium 表里就是两个词，所以「compendium 表格自洽」这个前提不成立。逐词按英文闸重裁：`Crime` 罪行→**犯罪**(60叶:40叶) · `Forensics` 法医学→**法证学**（151 处 `[[/knowledge forensics]]` 的渲染语境是石祭坛/霉变食物/字条这类**痕迹勘验，不是尸检**，且「法医学」全库 0 叶）· `Intrigue` 阴谋→**权谋**（「阴谋」已被 `conspiracy` 占用 22 叶 42 处，撞车）· `Legends` 传奇→**传说**(232叶:83叶) · `Machines` 机械装置→**机械**(84叶:13叶) · `Undeath` 亡灵化→**不死**(94叶:4叶，「亡灵化」全库仅 4 叶) · `Artifacts` **保持神器**（「遗物」已被 `Relic` 占死 182 叶 500 处）。改完复验 lang 四项全 0、拍平三数相等 1842、lang↔mjs 逐条一致 |
| **2026-08-13g** | `glossary_ec.json` 有三处待同步订正（**尚未做**） | `knowledge artifacts` 古器物知识→神器知识 · `knowledge forensics` 法医知识→法证学知识 · `knowledge undeath` 亡灵知识→不死知识；另 `Marlstone` 相关键凡值含「马尔斯通」的要改「马尔石」，否则**下一轮会被当权威反向回灌**。这是本轮唯一明确留下的尾巴 |

### 第十轮 A 段（2026-08-13）新增决议

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-13h** | **`Highgate` → 高门**（原「海门」，全库 14 处 + glossary） | High-gate 被当成「Hai/海」。英文三处明说它是通往内陆 `Redrak Fields` 的**北方陆上门户**，与海无关。错译已渗透 `name` 字段、`glossary_ec` 与 4 页正文。**其中 4 处落在 `Diplomatic Impunity`（已复核过的 35 本之一），J00 的 scope 之外** —— 又一次印证「越界发现必须主控接单」 |
| **2026-08-13h** | 城门 `gate` **不是**「异界之门」 | `Highgate.overview`「海门曾是一个繁荣的异界之门城区」、`Westgate.exposition`「下方异界之门旁」把凡俗城门写成位面传送门，**直接改动设定**。注意 `Lumiere Wharf` 的「异界之门」是真传送门，不动 |
| **2026-08-13h** | **`Luma` → 卢玛语，`Draconic` → 龙语**（阻断） | `Cultures/Languages` 的「常见语言」表把卢梅克人的语言 `Luma` 译成「龙语」，而真正的 `Draconic` 那一行**原样留着英文**。这是玩家建卡选语言时直接查的表：照此表选「龙语」会选到卢梅克语，想选真龙语的人在表里根本找不到中文。错误已扩散到 `All-Fable Keep` 与 `Performer's Plaza` |
| **2026-08-13h** | **`Vinarith` 的阵营 NE → NG**（阻断，**泄底**） | 玩家可见的 `text` 字段把中立善良写成中立邪恶，而「维纳里斯其实是巨龙 `Zerranyss` 的人类伪装」这一战役核心反转**只写在同页 `contentGamemaster` 的 `<section class="secret">` 里**。玩家翻开组织页看到贤者二把手标着「中立邪恶」，悬念当场泄底。这类「译文把 GM 秘密提前暴露」是**全新的缺陷类别**，此前没有任何判据或人工检查覆盖 |
| **2026-08-13h** | `Carmin Anther` → **卡尔敏·安瑟**（人名，非「卡尔敏花药」）；`Firebug's Leather` → **纵火虫的皮革** | 两处都是 **`name` 字段才是错的那一边**（第三次印证）。`Carmin Anther` 英文明写 `their child, Carmin Anther, … turned to a life of crime`，是个人，而 `anther`（花药）是逐词机翻；该 actor 自己的 `tokenName` 早就是「卡尔敏」。`Firebug` 是 `Bassa the Firebug` 的绰号（`name`＝纵火虫巴萨，英文闸 5:4），他的皮甲不该叫「萤火虫的皮革」 |
| **2026-08-13h** | **`scan_content_coverage` 会被上游拼写错误里的数字骗**（LOCAL-PATCHES 第 4 条） | 上游把 `Consortium` 误打成 **`C0nsortium`**（数字零），同叶几行外就有正确拼写。旧中文照抄了坏词，第十轮正确译成「银光束财团」之后判据立刻报 2 条 —— 因为它把那个 `0` 当成**数字**，而「财团」里没有 `0`。与第 2 条 `{Persuasion}` 同型：**判据分不清「真数字」和「拼写错误里的字符」，正确的中文反被判成缺陷**。按既定做法改英文基准 —— 否则唯一能让闸归零的写法是把 `C0nsortium` 留在中文里给玩家看 |
| **2026-08-13h** | **地名志的 `<dt>` 缺陷换了形态，旧判据抓不到** | J00 用「英文括号里的阵营词 + 代词串」做交叉锚点核对了全书 385 组 `<dt>`，**整体错位一格已不存在**（10 种阵营与 he/him、she/her、they/them 全部对上）。但出现了**更隐蔽的一类**：`<dt>` 里名字已译成中文，紧跟的 `<dd>` 正文里**仍写英文名** —— 同一条目上半段叫「达丽莎」、下半段叫「Darissa」，10 页合计 60+ 处。所有机械闸全盲 |

### 第十轮 B 段（2026-08-13）新增决议

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-13i** | **单元制审计会把术语分叉「切成两半」，必须配一轮全库术语传播** | A 段 12 个单元各自在自己范围内改对了术语，但同一专名横跨十几本书 —— 结果 `Otherhood of Fortune` 全库 186 处「异缘会」只改了 3 叶，**剩下的叶子里往往同段同时出现「幸运异姊会」与「异缘会」**，比不改还糟。B 段为此专设 T1–T5 五路只做传播不做通读，这一层**以后每轮都要有** |
| **2026-08-13i** | 跨单元统一（均以 `name`/folder 字段或英文闸多数为据） | `Otherhood of Fortune`→**幸运异姊会**（页 `.name` 与 folder 均如此）· `Pathways`→**通路**（502:23，「通道区」是少数派）· `Ember`(世界名)→**余烬**（清掉「烬界/安珀/烬火」及裸英文，⚠ 必须用英文闸区分普通名词「余烬/灰烬」）· `Sunfire Empire`→**阳炎帝国**（原有 6 种译名，无 name 可裁，取最大簇）· `Age of the Tower`→**高塔时代**（34 叶；「阳光年代」「日光年代」是错的）· `Arcageris`→**阿卡杰里斯**（它是神殿/武僧战团，原译「奥术巨龙」把它当成了龙）· `Pathward`→**径道语**（26:2，少数派「路径语」并入） |
| **2026-08-13i** | **`Wyrms`＝古龙 / `Dragons`＝巨龙 是两类生物，多数派是错的那一边** | 设定上古龙被阿拉尔改造成了最初的巨龙。`Bestiary.pages.Wyrms.name`＝「古龙 Wyrms」、`Dragons.name`＝「巨龙 Dragons」，但英文含 `Wyrm` 的 50 行里 **46 行译成「巨龙」**、只有 16 行用「古龙」。照多数派统一会把两类生物合并成一类 |
| **2026-08-13i** | 朗读文本（`block readaloud`）是**凭空增删的重灾区** | `Ooze Control / Flying Predator.exposition` 一段就删掉两句（蝙蝠状双翼＋旧伤疤、骨头相碰般的呼吸声）、删掉本事件**核心道具**「两团银灰色胶状物」（下一整节都在讲它），又凭空捏造「苍白的灰色身躯」「剃刀般锋利的爪」，还把 `as if to gauge whether or not you're a threat` 反写成「带着险恶的意图」。`Alchemical Decisions.exposition` 更自相矛盾：先写「默默地并肩大步走来」，两句后又写「边走边激烈地交谈」。**块数多重集相等，BLOCK 闸全程沉默** |
| **2026-08-13i** | 规则被悄悄放宽/收紧 | `Saving Jasper`：`Any character who contributed to`（有出力的那些角色）被译成「队伍中的每名角色」，**同调获得对象被扩大**；同段还凭空加出「以一场致命爆炸」，并给英文里本来没有标签的裸 `@UUID` 补了 `{爆炸瓶箱}`。**给裸 `@UUID` 凭空加标签**是这一轮新认识到的一种破坏方式 |
| **2026-08-13i** | `Excavation Pit`（挖掘坑）与 `Glowing Ore Pit`（发光矿坑）是矿井里**两个不同房间** | `Saving Jasper` 把前者译成后者的名字，会把玩家指向错误地点。两个专名各自的全库一致性都是 100%，正是「同一页后文还写对了」的那种局部错误 |
| **2026-08-13i** | `Stone Life`→石之生命 · `Helkas Green`→赫尔卡斯绿地 · `Gala Security To-Dos`→**晚会**安保待办清单 | 前两条标签向 `name` 对齐（`Helkas Green` 的 `Green` 是绿地不是姓氏，原译「赫尔卡斯·格林」当成了人名）；第三条**反向** —— `Gala` 英文闸 75:10 定译「晚会」，所以改 `name` 而不是改标签 |
| **2026-08-13i** | ⚠ **三方合并导出 `then` 时必须先合并相邻的 diff 操作** | B 段 34 个 K×T 冲突要「取 K 的逐句修复 ＋ 叠加 T 的术语替换」。自动导出 `then` 时踩到：difflib 把「路径语→径道语」拆成 `delete('路')` + `insert('道')` 两个**带空侧**的操作，任何 `old/new` 过滤都会把它们静默丢掉；而中间夹着的 1 个相同字符「径」又会打断朴素的相邻合并。**正确做法是合并相邻非 equal 操作、且不被 ≤4 字符的相同段打断**。第一版因此漏掉了 2 条术语替换 |

### 第十一轮（2026-08-13）新增决议

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-13j** | **标题锚点全线修复：全库 1491 个 `<hN>` 补显式 `id="<英文slug>"`** | Foundry 的 `#anchor` 由**标题文本 slugify** 而来，标题一译成中文 slug 就变了 —— 全库 257 个被引用锚点里 **187 个中文侧解析不了（590 次引用）**，全部落到页首。做法是英中标题**逐位对齐**后给中文标题补上英文 slug。修后我们的缺口 **187 → 0**；剩余 39 个（98 次）**英文侧同样解析不了**，是上游自己的死链。⚠ 这套 id 必须保住：任何后续改标题文字的批次都要**原样保留 `id` 属性**，删掉就会让 590 处链接重新断掉（第十一轮已把这条写进 agent 铁律，F1 还做了「骨架字符串逐叶比对」自证未破坏） |
| **2026-08-13j** | ⚑ **`glossary_ec.json` 是构建产物，手改会被 `build_glossary.py` 冲掉** | 这是本轮最有价值的机制发现。词表 = base 层（`fvtt\glossary_crucible_merged.json`）+ harvest 层（从两仓 `compendium/cn` 逐叶收割），**harvest 压过 base**。所以词表里的错值其实是「上一次构建时包的状态」的化石。**正确顺序**：① 包侧批次全部落库 → ② 跑 `build_glossary.py`（大批键自动自愈）→ ③ 只手改「孤儿键」（harvest 层没有证据的），且**必须两个文件都改**，只改 `glossary_ec.json` 下次构建就退回去。实测：我先前手改的 `knowledge forensics/undeath/artifacts` 三条在重建后**全部被打回**，正是因为只改了产物没改 base |
| **2026-08-13j** | 孤儿键 12 条已在两层同时改定 | `Luma`卢玛语（与 `Draconic`龙语**分开**，这是 A 段那条阻断的根因）· `Evidence`证据（不是「证据值」，29 叶坏替换的源头）· `Otherhood`异姊会 · `thornling`荆芽灵 · `Sanguinary`赤血会 · `Cascal Arcden`卡斯卡尔奥克登语 · `Elder Goddess Spectra`上古女神斯佩克特拉 · `Mutagist Contingent`突变学派分队 · 知识领域三条。⚠ 重建会把 `Luma`/`Draconic` 的双语尾巴去掉（format-only），**值本身正确且两者已区分**，可接受 |
| **2026-08-13j** | 跨单元术语统一（122 个术语族） | `Signborn`印记裔 · `Anachraenum`阿纳克瑞纽姆（三种音译，814:38）· `Spectra`斯佩克特拉（原「光谱」作女神名读成普通名词）· `Akonites`阿肯体（词表原作**「乌头属植物」**，把阿肯的构装体当成了植物学 aconite）· `The Armarium`阿玛留姆（是**杂货店**，原译「军械库」与真军械库 `The Armory` 撞名）· `Mutagist`突变学派（六种）· `Toothbreaker`碎牙帮（五种）· `House Cevher`杰夫赫尔（四种）· `Warlock`**邪术师**（与 `Sorcerer`术士 撞名的阻断，含 lang 两键与 `.mjs`）· `Altyran`阿尔提拉 · `Young Cheliceraeth`幼年螯蛛艾斯 |
| **2026-08-13j** | **替换脚本的 infix 残留是一整类，不是个别** | K5 只点出「阿克图里安人**人**类」9 处，实查同一脚本残留波及 **8 个血统共 35 处**（阿克图里安人阿什卡/费伊杰/基瓦尔/威伦/泽夫/赫尔格伦/尼尔艾）。另有 `<文化><血统>` 标签的「族/人」后缀 688:20:2 三写。**发现一处 infix 残留就要把同构式全查一遍** |
| **2026-08-13j** | `.mjs` 的 `LANGUAGES` 表有 5 处与权威表打架 + 3 个缺键 | 这张表同时驱动 `Language: X` 前缀标签与 `patchCrucibleConfig()` 改写的 `crucible.CONFIG.languages[*].label` —— 后者正是 `[[/language x]]` 渲染出的文字。`Solical`索利卡→**索利卡尔语** · `Mithia`密西亚→**米西亚语** · `Scripta`书文语→**斯克里普塔语** · `Scor`斯科尔→**斯科语** · `Lunix`月语→**卢尼克斯语**；另补 `Moiré`/`Borel`/`Kost` 三个缺键（有 `[[/language …]]` 调用却无表项，会渲染成英文） |
| **2026-08-13j** | `name`(双语并列) vs `tokenName`(裸中文) 的差异**不是** `same_en_split` 的缺陷 | 本轮 13 个「新增分叉」里 12 个是这一形态（`螯蛛艾斯 Cheliceraeth` vs `螯蛛艾斯`），属既定约定，`scan_token_name` 也是这么判的。真缺陷只有 `Monstrosities` 怪物 vs 畸怪（单数 `Monstrosity` 即畸怪）。**以后看 `same_en_split` 报告时先按字段类型分档** |

| — | 08-13 | **发 `crucible-cn 0.9.5` / `ember_cn_unofficial v1.1.5`**。发版前 5.4 全套 + 全部新判据为 0、lang 拍平三数相等（486/1842）、孪生分叉 0、锚点缺口 0。CI 三道闸（version/download/changelog）全过。下载回包核对：**crucible 0.30 MB / 24 文件、ember 7.21 MB / 21 文件**（与上版一致），`compendium/en`、`release/`、`.git`、`.bak`、嵌套 zip 均未混入，manifest 声明的每个文件包内都在。包内抽查：`Warlock`→邪术师契约（`Sorcerer` 仍术士）· `.mjs` 补上 Moiré/Borel/Kost 且 Lunix→卢尼克斯语 · 锚点 id 985 个 · 高门 15/海门 0 · 余烬 2009/烬界 0 · 异姊会 233/异缘会 0 · 证据值 0 · 语言表卢玛语与龙语并存 · 维纳里斯阵营 NG · 寒冷 2/冰寒 0。**⚠ 冒烟验证仍未做**（第 7 节清单已积压六轮） |

### 第十二轮（2026-08-13）新增决议

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-13k** | **`scan_same_en_split` 加双语尾巴归一：479 组 → 14 组** | 本库既定约定是**同一个词按字段类型写两种形态**：`name`＝「螯蛛艾斯 Cheliceraeth」（双语并列）、`tokenName`/表结果行/`levels`＝「螯蛛艾斯」（裸中文）。原判据逐字节比中文，把这条约定整个当缺陷报 —— **479 组里 463 组纯粹是「一处带英文尾巴、一处不带」**，`裸英文残留` 旗标因此在 467 组上无差别亮着，**等于没有信号**。归一规则保守：只剥「与该叶英文值逐字相同」的尾巴，剥完须剩含汉字的头部，`cn==en`（整条未译）不剥。新增 `--raw` 复现旧数字、`--selftest` 注入真分裂（`Monstrosities` 怪物/畸怪）与约定分裂各一条做双向断言 |
| **2026-08-13k** | **123 条永久豁免已归档**，以后不要重裁 | 全表 `4-临时脚本/2026-08-13-round12/findings/EXCLUSIONS.json`，每条都带可复现的硬证据。典型：`Shield`（带 `Imperceptible Barrier` 效果的 23 个＝法术护盾术，裸的 11 个＝装备盾牌，**1:1 无一例外**）· `Light`（actor 身上的戏法＝光 / `crucible.equipment` 的负重分级＝轻型）· `Water`·`Ooze`（场景 `regions` 地形 vs 生物分类 folders）· `West`/`East`（地点分区「西部」vs 随机方向表「西」，两表各自自洽）· `Aura`（天体专名奥拉 vs Crucible 手势灵气）· `Adelyne` 三份（**上游把同一效果复制到三个 actor 且硬编码同一个名字，中文按各自 actor 本地化是对的，统一反而会把上游 bug 复刻给玩家**）|
| **2026-08-13k** | ⚑ **`build_glossary.py` 的 `harvest()` 从不采顶层 `folders`** | 只走 `en_doc['entries']`，导致所有 `crucible.*` 包与 `ember.character`/`crucible-adversary`/`crucible-items` 的**文件夹名永远进不了 harvest 层**，只能被 base 层的陈旧值覆盖 —— 包里 `folders.Monstrosities` 早已是「畸怪」而词表还是「怪物」就是这么来的。已补 `walk_pairs(en_folders, cn_folders, got)`。补完重建后 `Celestial` 自动带出「天界生物 Celestial」，这类孤儿键不必再手改 base |
| **2026-08-13k** | **`prune_dead` 的 8 条死键：「`_legacyActions` 有意寄存」这个说法被证伪，已删** | 2026-08-12b 记的是「`migrate_cn_schema` 把抢救不了的内容 park 在此，留待人工抢救」。实查：这 8 条全在 `actors.Jurtak {Hunter,Warrior}.items.Jurtak Poison._legacyActions` 下，而 **① 该 item 的英文条目现在是 `null`（上游已删）② `_legacyActions` 不在任何 mapping 里，Babele 永不读取 ③ 同一段内容「摄入毒药 Ingest Poison」早已在 `crucible.equipment`/`playtest`/`pregens` 三处按 id 建键译好且有活的英文对照**。抢救早就完成了，寄存说法是过期状态。`prune_dead` 现两仓均为 0 |
| **2026-08-13k** | `scan_name_binding` 的 UNCERTAIN「都是目标没有中文 name 那一档」也被证伪 | 实为 **2/199**，其余各有别的成因，但全部良性，已逐类归因并归档。**「据称」的东西一律要拿数据验** —— 本轮两项「据称合法」一项证实（`Maziran` 族称）一项证伪（`_legacyActions`）|
| **2026-08-13k** | 词表 197 条 base↔shipped 争议全部裁完；181 条 pending 清零 | 174 条补进词表 / 6 条判为机械枚举须删 / 1 条待裁。⚠ **孤儿键必须 base 与产物两处同改**，只改产物下次 `build_glossary.py` 就退回去 |
| **2026-08-13k** | `scan_en_drift` 的 `changed` 桶（缺陷表 B 项）首次正经做完 | 判据按 §8 2026-08-13 的结论选「**新英文独有的 token（enricher/数字/专名）在不在中文里**」，不用长度比（后者在短叶子上零鉴别力，实测两个单元 23/23、51/51 全假阳性）|
| **2026-08-13k** | J00《Ordain Gazetteer》未读的 22 页补完 | 第十轮那本只读了 54 页里的 32 页。补读这 22 页又查出 27 条 —— **「读过」与「读完」是两件事，scope 里写的百分比要当真** |
| **2026-08-13k** | 7 个高产单元对抗式复核，又出 100 条 | 第十轮报最高的 7 个单元独立重读，K5 又出 27 条、K1 22 条、J01 13 条、K4 12 条。**再次印证单次通读召回率约 73%** —— 高产单元读完不等于读干净 |

| — | 08-13 | **发 `crucible-cn 0.9.6` / `ember_cn_unofficial v1.1.6`，随后 ember 补发 `v1.1.7`**。0.9.6/1.1.6 发完后自查发现「施工中」样板句还有第三种译法漏在「样板后面还跟正文」的叶子里 —— 普查全库该句 264 处 / 六种译法，且**上游英文本身就是三句不同的话**（Early Access / Beta / subject to change），按英文分三组各自统一（62 叶），补发 v1.1.7。crucible 侧无改动故不跟版。回包核对：ember 7.20 MB / 21 文件，锚点 id 988、`_legacyActions` 0、样板三种译法各对应一句英文 |

### 第十三轮（2026-08-14）最后一轮全面探测 + 修复

**探测**（84 agent / 4 轮动态迭代 / 1540 万 token）：首次把**插件代码**纳入镜头，
确认 **303 条**（11 阻断 / 88 严重 / 148 一般 / 56 观感）。产出压倒性地来自代码面而非译文面。

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-14a** | ⚑ **`register.js` 把 crucible 十类物品的描述写成 `"[object Object]"`**（11 条阻断的唯一根因，已发 `v1.1.8`） | `normalizeDescriptionValue()` 用「值是不是字符串」判断该不该转成 `{public,private}`。实测 `crucible-compiled.mjs`：**只有 `CruciblePhysicalItem` 是 `SchemaField`，另外十类（talent/spell/ancestry/archetype/background/taxonomy/loot/schematic/spellcraftRune/spellcraftGesture）都是裸 `HTMLField`（字符串）** —— agent 报的 6 类还少算了。于是这十类的**当前正确形状**被当成脏数据，转对象后经 `HTMLField extends StringField` 的 `_cast()`＝`String(value)` 落库成字面量，**不抛错不提示、原文永久丢失**。挂载点两处：`ready` 的全世界迁移 + `preUpdateItem`（每次保存都跑）。**修法：不再猜形状，直接问 schema**（`doc.system.schema.fields.description` 是否 `SchemaField`；只有原始载荷时按 `type` 查 `CONFIG.Item.dataModels`；**问不出来一律不动**）。回测 19/19 |
| **2026-08-14a** | **两个迁移的「只跑一次」护栏从未生效** | `game.world` 是 `foundry.packages.World`（DataModel），`getFlag`/`setFlag` **只定义在 `document.mjs` 与 `data.mjs`，`packages/` 下零命中**。`world?.getFlag?.()` 被可选调用静默吞成 undefined —— 守卫恒不成立、写回恒空操作，连它自己 catch 里的告警都永远打不出来。**每次开世界都重放**。改用 `game.settings`（world scope，在 `init` 注册）。⚠ 通则：**可选调用 `?.()` 会把「方法不存在」和「返回假值」混为一谈**，用在护栏上等于没有护栏 |
| **2026-08-14a** | **那段迁移连它想修的都修不到** | 它读 `item.system.description`（已 prepare 的值），而 SchemaField 类型 prepare 之后**永远是对象**，旧版留下的字符串在那里根本不可见。所以判据既漏掉真目标、又只误伤本来正确的。改读 `_source.system.description` |
| **2026-08-14b** | ⚑ **重导入冒险时 actor 的 `items`/`effects` 被静默丢弃**（已发 `v1.1.9`） | `patchActorUpdateDocuments` 里 `initialPayload = importMode ? prepareSafeActorUpdatesForImport(sanitized) : sanitized`，而后者 `= updates.map(degradeActorUpdatePayload)`，直接 `delete update.items / effects`。**降级被无条件前置到 happy path**，不是注释说的「出错时兜底」——那句注释描述的是下面逐 actor 隔离的循环，**注释与实现不符**，这条因此很容易被读漏。`isAdventureImportInvocation()` 匹配核心 `Adventure.importContent`，任何冒险导入都命中；已存在的 actor 走 `toUpdate` → 内嵌集合被剥。**Ember 的怪物战斗块/天赋/装备全是内嵌 item**，等于把上游推送更新的通道关死一半。修法：第一次尝试用完整载荷，只有单个 actor 确实失败才对它降级并告警。回测 8/8 |
| **2026-08-14c** | **S1/S2 改同一文件的合并裁决**：取 S2 打底 + 手工并入 S1 增量 | 29 处编辑对原文单独匹配全部命中，顺序套用时 7 处失败 —— 两片都看到了同一批问题。取 S2 是因为它把对话框标题重构成 `DIALOG_TITLES`，**既做翻译又做「认框」**（认出是 Ember 弹的框才解锁正文+按钮翻译），与它自己的闸修复耦合；还补了 S1 漏的 `Talent`（crucible 68 处）。S1 独有的 `ARRANGEMENTS` 表 + `Music`/`Environment` 前缀（音景 23 颗按钮里 21 颗）+ hex-hud 四条 tooltip 已手工并回 |
| **2026-08-14c** | **否决「把英文裸词当 i18n key」的修法** | S4 想把 `"Boss"`/`"Broken"`/`"Special"` 等 7 个英文串本身当 key 塞进 `lang/cn.json`。**Foundry 的 i18n 表是全局合并的** —— 任何模块调 `localize("Broken")` 都会拿到「破碎」。风险大于收益。这类「i18n 通道上的无键裸串」应走 `.mjs` 的 `EXACT` 表（有 Ember 宿主闸限定作用域）。已隔离待改 |
| **2026-08-14c** | ⚑ **崩掉的 workflow 会在盘上留下未验证的批次** | 12 路修复轮全部撞额度上限失败，返回 `handled=0`，**但批次文件已经落盘**。这些 agent 从没跑过 `--dry` 闸、也没返回结果，**不能当成成果套用**。已全部隔离到 `4-临时脚本/2026-08-14-fix/unverified/`（25 个）。**通则：只套用 agent 在结构化结果里声明过、且报了 `gate_clean` 的批次**，别扫目录 |
| **2026-08-14c** | 修复面外溢到工具本身 | 74 处编辑里有 22 处落在 `fill_missing.py` / `fill_twin.py` / `build_glossary.py` / 两份 `normalize_adventure_translation.py` —— 那些是会**持续制造回归**的源头（如 `build_glossary` 的分类器把「绝对不能归一的角色约定」判成「可脚本批量归一」）。改 `mappings.mjs` 后已按规矩重跑 `generate_runtime.mjs`，新增 `crucibleTokenName` / `emberEncounterTokenNames` 两个转换器 |

### 第十四轮（2026-08-14）复核 + 修复

**复核**（14 路并行）：把第十三轮遗留的 `REMAINING.json` **270 条逐条对当前 HEAD 重新核实** ——
那份快照写于 04:16，而两个仓库 05:02 又各发了一版，照抄 finding 会同时重复劳动、
并按过期描述把已改对的地方改坏。结果 **OPEN 203 / FIXED 63 / UPSTREAM 2 / INVALID 1**。
OPEN 分档：严重 13 / 一般 135 / 观感 55；落点**压倒性在代码面**（`ember-hardcoded-cn.mjs` 一家 90 条）。

**修复**（13 个工作包）：**FIXED 119 / DUPLICATE 60 / ESCALATED 19 / SKIPPED 4**。
落盘 **compendium 266 叶 + lang 21 键 + 孪生同步 67 叶**，全部零拒绝。

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-14d** | ⚑ **工作包按「一个文件只归一个 agent」切**，`ember-hardcoded-cn.mjs` 那 90 条切成**三段顺序**跑 | 落盘方式那一节的教训（并行批次会静默互相回滚）此前只用在 compendium 上；这一轮把它推广到**代码文件**。同一个 `.mjs` 三段（表 / 闸 / 通道）顺序执行、后段拿前段的产出摘要，实测零冲突。compendium 与 lang 一律只产批次不直接落盘，三方合并前先查路径重叠 —— 本轮 11 个批次 **0 条路径冲突**，因此没有触发人工裁决 |
| **2026-08-14d** | **驳回两个 `lang` newkeys 批次（63 条键）**，不为它们放开 `apply_lang.py` 第 1 道闸 | 该闸（键必须存在于上游 `en.json`）挡的正是 1.1.0 那次 lang 77% 静默失效。逐类看：26 条 `ui.notifications` 整句**已被同轮 `.mjs` 的 `patchNotifications()` 覆盖**（枚举了 76 处调用），再走 lang 就是同一件事的第二条**全局**通道；30 条 RegionBehavior schema 串与 `None`/`Unknown` 是**裸通用词**，正是 `2026-08-14c` 已否决的做法；`AFFIX.Affix` 是死键（上游 `AFFIX.*` 22 个键里没有它）；两条 `TOKEN.MOVEMENT.ACTIONS.*.description` 是**核心 Foundry** 的键，属 `foundry_chn` 辖区 |
| **2026-08-14d** | **RegionBehavior 的 30 个裸英文走「schema 就地改写」，不走 i18n 键** | Ember 三个子类型（`ember.trapTrigger`/`areaEffect`/`footstepSurface`）的 `defineSchema()` 把英文写死在 `label`/`hint`/`choices` 里。做法：从 `CONFIG.RegionBehavior.dataModels` 取到类，遍历 `schema.fields` 就地改写。作用域精确到这三个子类型，不碰全局 i18n 表，也不依赖表单 DOM 选择器。**字段对不上就跳过并告警，不静默、不凭空造选项** —— 离线回测 12/12（含「缺失字段不被创建」「多出的选项不被凭空加」「真 i18n 键 `EFFECT.Image` 不被碰」三条反向断言），脚本 `4-临时脚本/2026-08-14-round14/probes/region_behavior_harness.mjs` |
| **2026-08-14d** | ⚑ **`Region Map`/`Area Map` 全库拆分：方向定了，但本轮判「不做」** | 方向是 `Region Map`＝**地区地图** / `Area Map`＝**区域地图**（英文闸 159 叶 vs 268 叶，改 Region 侧面小）。但 `unify_terms.py` 的叶级闸在这里不够用 —— **46 叶英文同时含两个词**而中文都是「区域地图」。改用逐位对齐后实测 **115 叶可对齐 / 72 叶计数不等**（中文重复了术语而英文用代词或 the map）。**只落 115 叶比不落更糟**，正是 `2026-08-13i` 那条「把术语分叉切成两半比不改还糟」。本轮只做**局部消歧**：那个同框下拉的 6 条 `DESTINATIONS` 在 `.mjs` 里就地区分，方向与将来一致、不需回改。批次已生成后主动删除未落盘 |
| **2026-08-14d** | `Soulbound Progression`→**魂缚进阶**（清「进程」5 处）；魂印确认框的 `rank N`→**阶位 N**（清「等级 N」5 处） | 两组都是同屏可见的分歧：`glossary_ec` 定稿是「进阶」而 `DIALOG_TITLES` 早前写「进程」；`lang` 的 `EMBER.ATTUNEMENT.Rank` 是「阶位」而 `DIALOG_UI` 写「等级 1（次等魂印）」。注意 `Rank`阶位 / `Tier`阶 / `level`等级 是三样东西 |
| **2026-08-14d** | `Hexblade` **从表里删掉**（不译）；`Divine Domain`/`Warlock Patron`/`Sorcerous Origin` 三个前缀采纳 | 合集里 `Spellblade` 已定稿「咒刃」，把 `Hexblade` 也译「咒刃」是 1:2 撞名；而上游 Warlock 页**已删该子职**、合集无对应译法、只剩 2 叶，且属 dnd5e 侧（附带项）。为它制造一处主线撞名不划算 |
| **2026-08-14d** | `[[/language moiré]]` 4 处**本轮不做** | 上游增强器 pattern 是 `(\w+)` **无 `u` 标志**，`é` 不算 `\w`，增强器根本不触发 —— `.mjs` 侧吃不到，只能改 compendium 字面量。但删掉整个标记会让中英标记多重集不等：`apply_translations` 第 3 道闸必拒、`scan_markup_drift` 的 `INLINE` 从 0 变 4。为 4 处观感问题**同时**绕过一道闸并加一条永久豁免，性价比不成立 |
| **2026-08-14d** | `scan_same_en_split` 第 15 组 `Scout` 判**合法分叉**并归档（豁免表 123→**125** 条） | 两侧各自内部一致：权威条目 `crucible.archetype.json::Scout.name`＝「侦察兵 Scout」（crucible 角色**原型**）；`actors.Mutagist Scout.tokenName`（英文侧就是裸 `Scout`）＝「斥候」，与同卷正文 `Scout(s)` 的 **62 处译法 0 例外**一致（故事里的敌方斥候）。同页 `scouting party` 另译「侦察队」也忠实。⚠ `scan_token_name` **结构性看不见**这一组：它要求英文 `name` 与 `tokenName` 逐字节相同，而这里是 `Mutagist Scout` vs `Scout` |
| **2026-08-14d** | 删除 `2-Crucible汉化插件/scripts/fix_word_leaks.py`；`build_zip.py` 改为**直接 `raise SystemExit` 的废弃桩** | 前者：全库零引用、`ALLOW` 白名单是死代码、词表与本项目定译冲突（`Tier`→等级，定译是「阶」），而 `main()` 无 `--dry-run` 直接覆盖写 `compendium/cn/*.json` 与 `lang/cn.json` —— 同时踩本项目两条硬约束。后者：输出名/目录层级/内容清单三项都与 `module.json` 对不上，跑它只会产出装不上的包 |
| **2026-08-14d** | crucible 侧 `release.yml` 补上**回包断言**（ember 侧早就有） | 原来只有 `ls -la` + `unzip -l | tail -10`。那串 `-x` 是**声明不是保证**：zip 的追加语义、通配符写错一个字符、或仓库里新冒出一个 `.py`，都会让排除清单静默失效，而体积看不出来 |

| — | 08-14 | **第十三轮**（84 agent / 4 轮动态迭代）首次把**插件代码**纳入镜头，确认 303 条。急修两条阻断并发 `v1.1.8`（`register.js` 把 crucible 十类物品描述写成 `"[object Object]"`）与 `v1.1.9`（重导入冒险时 actor 的 `items`/`effects` 被静默丢弃），随后发 `crucible-cn 0.9.7` / `ember_cn_unofficial v1.1.10`（严重档第一批） |
| — | 08-14 | **第十四轮**：270 条逐条重核 → 203 条 OPEN → 13 包全部处理。落 266 叶 + 21 lang 键 + 67 孪生。**5.4 全套复验全 0**（lang 四项 + 拍平三数相等 1842/486 · 标记五项 · 方括号内标记 · class 漂移 · 数字覆盖 · 外来文字 · 死键 · 中文侧缺键 · `uuid_swap` BROKEN · `attr_text` · `token_name` · `bare_english` · 孪生分叉），`same_en_split` 15 组全部为已归档的合法分叉，`name_splits` 5 组即已知的合法分裂。**⚠ 冒烟验证仍未做** |

### 第十四轮 B 段（2026-08-14）内容缺口收口 + 两条现场问题

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-14e** | ⚑ **缺陷表 Z 项里的 idx 34/41 已清零**：`encounterTokens` 管线缺口补完 | 第十三轮补了 mapping 与转换器，但**英文基线从未重抽**，所以 cn 侧一条译文都没有、GM 一放怪满地英文。本轮重抽并入：每包新增 **194 encounterTokens + 42 sounds + 9 regions**，翻译落盘 **490 叶**（主线 245 + 孪生 245），三道闸零拒绝。复验 `en == cn`（194/194、46/46 两包皆是），`fill_missing` 反方向 0 条 |
| **2026-08-14e** | ⚑ **重抽英文基线一律走「只并新路径、已有路径不动」**，不要整体覆盖 | 新脚本 `4-临时脚本/2026-08-14-round14/merge_new_en_fields.py`。整体覆盖会静默回退 `LOCAL-PATCHES.md` 的四条上游笔误补丁，而那几条是「**任何译法都必被判 markup mismatch**」的阻断级。单向并入让补丁**结构上不可能被回退**，比「重抽后再复打补丁」可靠 —— 后者漏一次就是一次阻断。已逐条复验四条补丁健在 |
| **2026-08-14e** | 6 路分片翻译必须配一路**统稿**，且统稿要查的是「**同族兄弟串是否同构**」而不是「同一英文是否重复」 | 本批 175 条**没有一条严格重复**（分片互不重叠），但同族兄弟串出了 5 组分叉、17 条要改：光 `Waterfall Exterior/Interior` 8 条就写出 **5 种格式**（瀑布 - 室内东北 / 室内西北瀑布 / 瀑布 外部 东北 / 瀑布 外部西北 / 瀑布室外西南）。分片之间互相看不见，这类分叉是**结构性**的，不能指望分片自觉 |
| **2026-08-14e** | ⚑ **`fill_missing.py` 的 `todo.*` 键形态是错的**（已修） | 同一函数里 `tm.*` 走了 `to_batch_path()` 剥掉 `entries.` 前缀，`todo.*` 却直接写文档根路径 —— **两半键形态不一致**。照 PROJECT.md「批次 key ＝ 待译清单里的 path」直接用会整批 `REJECTED no-EN`，而这个理由字面像「英文里没有这个键」，极易被误读成源数据问题。**6 路 agent 每一路都先踩了一次**。这类「工具自己制造回归」的缺陷优先级要高于译文缺陷 |
| **2026-08-14e** | ⚑ **`patchWeatherLabels` 走错取值链，一直空转**（已修，玩家可见） | 日历条的天气与风向悬浮提示一直是英文。真实链路是 `EmberWeatherManager#getConfig()`（`ember.mjs:21825`）→ **`slices[x].config.weather[type]`**，补丁写的是 `slice.weather` —— 那个键在 region slice 上根本不存在（只在 Vista 场景定义里出现且只有 `elevation`）。于是 `Object.values(undefined ?? {})` 空转、`n` 恒为 0，而**旧代码 0 条时不告警**，所以没人发现。修法：两条路都走 + **0 条时告警**。回测 13/13。词表本来就是全的（上游 27 个 label 除 slice 名外 26 个都在表里）。⚠ 风向里的 `(12 mph)` 是拼接串，仍是英文，中文化要包整个 `#refreshWeather`，不划算 |
| **2026-08-14e** | **hex 图上队伍 token 变透明与本项目无关**（已证，非推断） | 对两个模块全部运行时代码 grep `alpha\|hidden\|elevation\|occlu\|renderFlags\|refreshMesh\|.mesh\|canvas.tokens`，命中 5 处**全部是被翻译的英文界面串或注释**，零处代码。另查了唯一理论相关通道：我们确实翻 `Scene.levels[].name`（层级 `bottom`/`top` 决定遮挡），但 babele 的 `nameCollection` 实现是 `mergeObject(data, {name, translated:true})`（`babele/script/converter/converters.js:99-114`），**合并**，`bottom`/`top`/`_id` 碰不到。真正来源是 Ember 自己的 `EmberToken` 与 `_applyHexRules`→`renderFlags`（`ember.mjs:59636-59642`），队伍 token 在 hex 图上是 `elevation:-1` + `occlusionMode: VISIBLE` |

| — | 08-14 | **第十四轮 B 段**：补完 idx34/41 内容缺口（+490 叶）、修 `patchWeatherLabels` 空转与 `fill_missing` 键形态、新增 `patchRegionBehaviorSchemas`。5.4 全套仍全 0；`same_en_split` 15 组**无新增**（`Scout` 组 5→7 叶，属已归档 G15）。三个离线回测：区域行为 12/12、天气 13/13、临时修补插件 20/20 |

| — | 08-14 | **发 `crucible-cn 0.9.8` / `ember_cn_unofficial v1.1.11`**。发版前 5.4 全套全 0、`flatten_lang` 三数相等（1842 / 486）、孪生分叉 0。下载回包核对：manifest 与 zip 均 HTTP 200，**crucible 320 KB / 26 文件、ember 7.3 MB / 21 文件**，`compendium/en`、`lang/en.json`、`lang_keep_english.json`、`release/`、`.py`、嵌套 zip 均未混入，manifest 声明的每个文件包内都在。⚠ **踩到两个 CI 坑，都记在这里**：① crucible 首次构建 12 秒失败 —— 本轮新加的回包断言写的是 `unzip -l \| grep -E "…\|\.zip$"`，而 `unzip -l` 第一行是 `Archive:  module.zip`，**模式匹配到了自己的表头**，断言恒失败（包本身干净）。改用 `unzip -Z1` 只列成员名。② 修①时把 `^scripts/` 一并抄进 ember 仓，**但两仓含义相反** —— ember 的 `module.json` esmodules 含 `scripts/ember-hardcoded-cn.mjs`（必须进包），crucible 的 `scripts/` 才全是维护脚本；照抄会让**下一次 ember 发版直接失败**。是靠「下载回包核对」这一步发现的，已改回并做双向回测 |

### 第十四轮 C 段（2026-08-14）剩余未做项收口

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-14f** | ⚑ **推翻 R4 的「本轮不做」** —— `Region Map`/`Area Map` 全库拆分**已完成**（两包 187 叶）| 上一段判「做不了」是**建立在错的测量上**：探针正则 `\b(Region\|Area) Maps?\b` **区分大小写**，而英文正文里小写 `region map` / `area map` 很多，英文侧计数系统性偏小，于是 72 叶被误报成「需人判」。改 `IGNORECASE` 后：可机械对齐 115→**169 叶**，需人判 72→**12 叶**。**四路人判单元各自独立发现并报了这个问题** —— 分片 agent 撞到判据本身的毛病时会说话，这是本轮最有价值的一次回报 |
| **2026-08-14f** | 逐位对齐是**筛子不是判据**，已知失效形态是「中文把 in addition to X 提到句首」 | 新写的方向验证器逐位配对 221 叶，报错 1 叶（`Patch 0.3.2`）—— 而那正是人判单元**事先点名**「逐位对齐会判反」的那一叶：英文 `for vistas and area maps in addition to the region map`，中文语序相反但**两处都译对了**。所以真实正确率是 **221/221**，报错的是验证器自己。221 叶里出现 1 次，比例低但必须有人判兜底 |
| **2026-08-14f** | 全库 5 叶原本已含「地区地图」，**4 叶方向是反的、1 叶是绝不能动的假阳性** | 脚本只数「区域地图」，**看不见已经写着「地区地图」的叶子**——这是同一个探针的第二个盲区。`Verdant Paths`/`Crystal Carving Cavern`/`Redrak Fields` 英文是 `area map` 却写「地区地图」，`Emergence` 两处正好互换，均已订正。⚠ `Trident's Point` 是**假阳性**：中文「一幅本**地区**+**地图**」（英文 `a map of the local region`），「地区地图」只是碰巧的子串，机械替换会把意思改坏 |
| **2026-08-14f** | `moiré` 换修法：**不动 compendium，放宽我们自己的兜底正则** | R7 判「改字面量要绕标记闸 + 加永久豁免，不划算」——但那个前提可以绕开。`.mjs` 里本来就有一条兜底 PATTERN 接「增强器没接管的 `[[/language …]]`」，而它**照抄了上游的 `\w`**，所以同样接不住 `é`。改成 `([\p{L}\w-]+)` + `u` 标志即可，compendium 一个字未改、不绕闸、不加豁免。⚠ 加 `u` 之后裸 `]` 是语法错误，必须写 `\]\]` |
| **2026-08-14f** | `term_gate` 的裸词计数**必须带 `\b`** | 实测 `--en "Erisa"` 报「埃里萨 10 : 埃丽莎 4」，看着像该反过来统一 —— 实际是把 `Erisagosa` 当子串误收。加词边界后主线包只剩 3 叶，才是真正的 1:1。这类专名闸一律带 `\b` |

| — | 08-14 | **发 `ember_cn_unofficial v1.1.12`**（crucible 侧本段无改动故不跟版）。回包核对：manifest/zip 均 200，7.3 MB / 21 文件，无混入；包内抽查地区地图 191 / 区域地图 314、`moiré` 在表内、兜底正则已带 `u`、区域行为补丁与 `config.weather` 均在包里、无「魂缚进程」残留 |

### 第十五轮（2026-08-15）四个新方向 · ④ 决议断言化

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-15a** | ⚑ **第 8 节的裁决编译成可执行断言**：`qa/assert_resolutions.py` + `5-其他内容/RESOLUTIONS.assertions.json`，进 §5.4 第 16 项 | 第 8 节有上百条裁决、**全是散文**，没有任何机制阻止下一轮悄悄推翻某一条。十四轮里已出过多次险：`scan_uuid_swap` 改判据降噪时消掉过一条真缺陷 · `Scout` 差点与已归档豁免冲突 · crucible 的 CI 断言照抄进 ember 差点让下次发版失败 · `Trident's Point` 的「本地区+地图」差点被术语替换改坏 —— **全靠人当场想起来才没出事**。首版 16 条断言，8 种类型 |
| **2026-08-15a** | ⚑ **断言必须做双向回测，只测特异度等于没测** | 首次全量跑就抓到**判据自己写错了**：双语尾巴那条把「圣堂区路人 A」「菌丝旷野底图 B」这类**编号后缀**全报成尾巴（14 处全假阳性），而那恰恰是既定约定要求的写法。改为「结尾拉丁串至少 2 个字母」并加 `--selftest`（10 条正反例）。灵敏度回测走 `--root <副本>`：注入 8 类违规，**每一种断言类型都确认会响**（term_gated 要整叶改完才响、单处改不响 —— 这条局限已写进脚本头） |
| **2026-08-15a** | 断言**不追求全覆盖**，覆盖不到的要写明 | 三条已知局限记在脚本头：① `term_gated` 是叶级的，叶内部分错译抓不到；② `cn_absent` 可能误伤合法用法，加新条目前必须先数全库；③ 「改动面小的那边优先」这类**方法性**裁决本质上表达不成断言，仍然只能靠人读第 8 节。**写清楚覆盖不到什么，比假装全覆盖有用** |

### 第十五轮（2026-08-15）三个新方向 · ①②③

**237 条 findings（4 阻断 / 49 严重 / 142 一般 / 42 观感），落 185 叶。**
这一轮的价值不在数量，在于**三个方向此前都没有任何判据**。

| 日期 | 决议 | 理由 |
|---|---|---|
| **2026-08-15b** | ⚑ **① 按「一场戏」读，而不是按「一片叶」读** —— 跨叶指代断裂是主导缺陷类，且叶级判据**结构性**看不见它 | 前十四轮全部判据的单位都是叶。改成按 GM 备课顺序整链通读三条事件链后，30 条跨叶指代断裂全部出自「**两侧英文串本来就不同**」——所以 `scan_same_en_split` 这类同串比对定义域之外。典型：`catwalk` 四种译名且分裂线正好落在「GM 朗读文本 vs GM 指令段」之间（GM 念一个词、看另一个词）· 任务奖励 GM 念「聚焦珍珠」实发「专注珍珠」· `Bonny Captain`↔`the Captain` 的伏笔链被「船长/队长」切断 · 边缘镇在一条朗读文本里成了全库唯一的「炽城」 |
| **2026-08-15b** | **零结果也是结果，而且要机械化证明** | 三路都报了可证伪的零结果：**顺序错乱 0**（按 `</p></li></h*>` 切块对齐，17 页块数逐叶相等）· **指令性丢失 0**（扫全部 If/Unless/Once/Should 块，逐块查中文条件标记）· **规则数字 0 出入**（剔除 enricher 后逐叶数字集合比对）。**说「没查到」和说「查了、用这个方法、确实没有」是两回事** |
| **2026-08-15b** | ⚑ **② 中文可读性是一条独立的轴，与保真度无关** | 一句话可以 100% 忠实同时是很难读的翻译腔。9000+ 叶主要由 LLM 产出，从没人以「中文读者」而非「对照校对者」的身份读过。按**具名病症**润色三本玩家通读物：长定语 53 · 词序照搬 30 · 被字滥用 30 · 的的不休 11 · 同义堆砌 8。硬约束：术语/数字/标记一律不动，语义零变化 —— **只改能说出病名的地方**，说不出病名就不改 |
| **2026-08-15b** | ⚑ **③ 词表会把错误洗成权威 —— 4 条阻断，全是 `source='base'`** | 词表 = base + harvest 且 **harvest 压过 base**，所以 base 层的陈旧错值**从不被覆盖**。包里改对了、词表停在旧值，下一轮拿词表当依据反向回灌就把包改坏。实测：`Kinesis`=念动力（违背定译且与 `Telekinesis` 撞车，而包里「念力」134:0）· `Fortitude`=坚韧（与 `Toughness` 撞车，包里 136:30）· `Rank`=等级（`Rank`/`level` 在词表里彻底不可分）· `Region Map`=区域地图（**会把第十四轮刚拆开的「地区地图」洗回去**）。base 与产物**两层同改**，并补 `Toughness`/`Tier`/`Level` 三条缺失锚点——这一族原先**没有任何正确锚点**，所以才会被反复抢占 |
| **2026-08-15b** | ③ 报的「包内残留 182 处未清漂移」是**假警报** | 那 182 次全部是同叶里正确的 `Area Map`。叶级计数看不出来，逐位核验后 `区域地图` 次数 > 同叶英文 `Area Map` 次数的叶 = **0**。**agent 的计数类结论一律要自己复算一遍** |
| **2026-08-15c** | 单元制审计又一次「把术语分叉切成两半」，靠**发版后回包核对**才发现 | `catwalk` 统一是在 `Glitter in the Dark` 这一本里做的，而该词横跨全库 —— 发完 v1.1.13 做回包抽查时才发现另外 3 本 journal 里还是「栈道」。已补齐并加进断言。⚠ 注意 `boardwalk`＝木栈道是合法的，替换必须用 `(?<!木)栈道` 才不误伤 |
