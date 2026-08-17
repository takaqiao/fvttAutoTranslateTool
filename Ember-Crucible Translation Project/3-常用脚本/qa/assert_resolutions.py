#!/usr/bin/env python3
"""把 PROJECT.md 第 8 节的裁决**编译成可执行断言**，跑在 5.4 全套后面。

  python assert_resolutions.py [--rules <json>] [--repo <repo> ...] [--verbose]

为什么需要这个
--------------
第 8 节现在有上百条裁决，**全是散文**。没有任何机制阻止下一轮悄悄推翻某一条 ——
十四轮里已经出过好几次险：`scan_uuid_swap` 改判据降噪时消掉过一条真缺陷；
`Scout` 的处理差点与已归档豁免冲突；给 crucible 的 CI 断言照抄进 ember 差点让下一次发版失败；
`Trident's Point` 的「本地区+地图」差点被术语替换改坏。**这些都是靠人当场想起来才没出事。**

散文裁决 → 机器断言，一致性就从「靠记忆维持」变成「靠闸维持」。

断言类型
--------
`term_gated`      英文侧命中 `en` 的叶，中文必须含 `cn_required`（且不得含 `cn_forbidden`）
`cn_absent`       某个中文串在 compendium **与 lang** 全库不得出现（已清零的错译不许回潮）
`sense_gated`     同一个英文词有**机制义／普通名词义**两支，按上下文窗口分类后只闸得住的那两个方向
`distinct_terms`  一组术语的中文必须两两不同，**且每个术语都要过英文闸读库核对**（防撞名 + 防空转）
`term_domains`    同一个英文词按域分裂成多个中文，逐域钉死（防下一轮「顺手统一」）
`lang_parity`     两仓 `lang/cn.json` 的键数必须等于英文侧键数
`anchor_ids`      标题上的显式 `id=` 数量不得低于阈值（撑着锚点链接）
`no_bilingual_tail` 指定字段的中文不得带「中文 English」双语尾巴
`exclusions_closed` `same_en_split` 的分组必须全部在已归档豁免表内
`leaf_literal`    指定叶必须/不得包含某个字面串（用于登记「绝不能动」的假阳性）
`block_aligned_gate` **叶内**判据：按块级标签把中英切块后逐块比术语类别（第十八轮 Y2 新增）
`block_sense_gate`   同上，但块内做机制义／普通名词义分类，块内单一义项时上正向闸
`enricher_slot_gate` **槽位级**判据：按 (动词,目标) 把两侧 `@X[…]{标签}` 配对，逐个比标签里的术语
                     （第十九轮 Y6 新增，补 `split_blocks` 把标签连花括号一起涂空留下的洞）
`enricher_text_coverage` **段级**判据：`@Embed[… readaloud="…"]` 里那整段朗读正文的中文
                     有没有跟上英文（缺失／中英字符比／段内定译专名／句数四支）。
                     第二十二轮新增 —— 那 16966 字符上一轮**已经进了口径却没有任何闸**：
                     英文 48 段里阿拉伯数字 0 个，默认的数字判据对它必然 0 命中，
                     实测把最长的一段中文删空也一声不响。见 `a_enricher_text_coverage`。
`translate_cases`  **真的 import 发布中的 `translateText` / `translateNotification`（ESM `.mjs`）
                     跑译文用例**。第二十六轮新增、第二十七轮扩面。被判的是**三张**「命中即整串
                     替换」的表：`PATTERNS`(28 条正则) · `PREFIXED`(19 条前缀) · `NOTIFICATION_PATTERNS`
                     (31 条正则)。自检面板的 D 档按设计核不了正则表（对 PATTERNS+PREFIXED 这 47 个键
                     直接报 ⛔），而第二十六轮以前 61 条断言里 `PATTERNS` 出现 **0 次**。
                     ⚠ 别再写「唯一能主动改坏译文的两张表」—— 第三张 `NOTIFICATION_PATTERNS` 挂在
                     `ui.notifications.notify` 这个**所有模块共用的全局钩子**上，爆炸半径比 PATTERNS
                     （只在 Ember 自己认出的子树里跑）**更大**，第二十六轮漏在闸外整整一轮。
                     四段用例：反例（上游产不出的近似串、别的模块发的通知，一个字都不许动）·
                     正例（上游真产出的串必须翻得动且**译文逐字符相符**）· **覆盖归因**（三张表的
                     每一条都必须被至少一条正例触到，漏一条报违规 —— 第二十六轮 PREFIXED 只触到
                     2/19 而闸名写着它，就是靠人眼数漏的）· 全量护栏（编排名从**上游 ember.mjs 现抠**
                     × 2 通道逐条过）。见 `a_translate_cases` 与 `translate_cases_runner.mjs`。
                     ⚠ 脚注（第二十九轮，上面那个「(31 条正则)」不改、在这里更正）：
                     `NOTIFICATION_PATTERNS` 现在是 **27 条** —— 被判文件按「全局通知表每条正则
                     必须有厂商锚点或封闭定义域」的裁决删掉了 4 条（见它 :2132 起的注释块），
                     `recorded.np_size` 随之 31→27，来龙去脉记在规则的 `recorded._why_29`。
                     ⚠ 第二十九轮补的**反例结构护栏**（作弊路径 C 的一部分）：正例那侧早有
                     「每条表项都必须有正例」的恒等式，**反例那侧此前只有一个可调的数**，
                     清空反例 + 把数改 0 就全绿。现在两道，都从表长现算 ——
                     (N1) 一条表项一条**专属**的近似反例（最大二分匹配必须完美，
                     「近似」＝反例串逐字包含该表项的**字面骨架**）；
                     (N2) 机械近似探针 `Zzq <骨架> Zzq`（零维护，抓「不锚定串首/串尾」的正则）。
                     见 `translate_cases_runner.mjs` 的 §④b。
`panel_liveness`  **现跑**自检面板的 D 档（node 子进程，离线桩 fetch/game/Hooks），把它自己算的
                     数与记录值对一遍（第二十八轮新增）。上一条只钉档名那一个字符串 ——
                     **719 个键的结论漂成什么样都不会红**，而唯一能跑出那组数的执行体是一份
                     **未入库的一次性探针**：结论本身就是快照（形态 (c) 在更高一层复发）。
                     ⚠ 记录值是双向的（覆盖侧「不得低于」· miss 侧「不得高于」），但**真正扛事的
                     是几条不含阈值的恒等式**：逐行 checked 之和 == 合计 rawChecked · 模板三路记账
                     == 抓到的份数 · 喂几张表就得有几行 · 6 个自造串必须 6/6 报出（匹配器变恒真
                     时只有它会响）· 子串边界仍复现 · 现跑吐出的档名与上一条钉的那行一致。
                     见 `a_panel_liveness` 与 `selfcheck_panel_runner.mjs`。
`tracked_inputs`  断言当**运行时输入**读的文件必须都在 git 里（第二十八轮新增）。
                     同型缺陷已经第四次（EXCLUSIONS.json → 第十五轮 findings → english-baseline/
                     → `translate_cases_runner.mjs`）：本机跑得好好的，clean checkout 上整条塌掉。
                     清单**从规则集自身推**（含各 kind 的**默认 runner 名** —— 第四次那个缺陷正好
                     藏在默认值里），外加 `3-常用脚本/qa/` 全目录 sweep 与 `must_include` 点名兜底。
                     ⚠ 按**拥有文件的那个仓**判：两个插件目录各自是独立仓。见 `a_tracked_inputs`。
`ruleset_shape`   **规则集自身的形状**：断言条数 / kind 数 / 各 kind 的容器块（第二十九轮新增）。
                     堵的是**作弊路径 C**（三条路径的定义见 `RULESET_SHAPE` 上面那段注释）：
                     A、B 都堵住之后，最省事的绕法不再是改数字，而是**把装数字的容器端走** ——
                     复核实测七种**单点**容器层删除，改完主闸仍 65/0/0 全绿，其中最刺眼的一条是
                     **删掉整条断言（65→64）全项目没有任何东西数一数还剩几条**。
                     ⚠ 要害是它自己的自指问题：**记录值绝不能住在它所守护的容器里**，
                     否则删容器就把守卫一起删了。⇒ 地板住在**本文件的模块级常量 `RULESET_SHAPE`**，
                     另有一道**不是断言**的存在性检查写在 `main()` 里（`_require_ruleset_shape`），
                     防的正是「连守卫一起删」。绕过它要改哪几处、同时改这几处仍然能过，
                     都写在那段注释与规则的 `why` 里 —— **不许写成「不可能被绕过」**。
`source_literal`  指定**源码文件**里必须出现 / 不许出现的字面量（第二十七轮新增）。
                     只为一件事：把自检面板 D 档的档名「上游字面量存在性（仅供人工复核）」钉死。
                     叫回「键活性」的代价是第二十四轮 77 条被当真缺陷追了一整轮，而 `cn_absent`
                     只扫 compendium + lang、够不到 `.mjs`，`twin_files` 两份一起漂回去也照样绿。
`glossary_value`  词表的 base 层与产物层都必须是某个值（防「词表把错误洗成权威」）
`version_matrix`  PROJECT.md 抬头与版本矩阵必须与两仓 module.json 一致（这一行漂过两次）

每条断言都带 `decision`（对应第 8 节哪一天的裁决）与 `why`，失败时一并打印 ——
让下一个人看到的不是「断言 R-xx 挂了」，而是「你违反了 2026-08-13j 定的那条，理由是……」。

第十六轮补的通则：**任何断言都必须能说出「我扫了多少个叶 / 多少个键」**
------------------------------------------------------------------
说不出来的那种，是**自检**，不是断言。这条通则是被咬出来的，本项目已经实测到
**八种不同的空转形态（a）–（h）**——写新断言前请拿这八条当自检清单逐条过一遍
（每加一种都是被实测咬出来的，别只看前三条就以为过完了；(h) 是唯一一种「判据没坏」的）：

（a）**判据写坏了** —— `R-catwalk`：`en` 在 JSON 里写成 `"\bcatwalk"`（单反斜杠），
     被 JSON 当成退格符吃掉，正则变成 `\x08catwalk`，一个都匹配不到，断言一路报绿，
     而库里实际有 10 叶违规。
     ▸ 防法：`min_hits`。**这一类 `min_hits` 能防。**

（b）**判据压根没读库** —— `R-region-area-map`：旧版 `distinct_terms` 的注释白纸黑字
     自陈「纯配置自检，不读库」，只比较一组术语的中文两两不同。于是它一直全绿，而
     ember lang 的 `EMBER.CALENDAR.REGION` 反着写了「区域地图」**四个发布版没人发现**。
     ▸ 防法：强制声明 `scan`（lang / compendium）。**`min_hits` 对这一类无效** ——
       它连「命中数」这个概念都没有，没读库就没有命中数可言。

（c）**读的是别人早先写下的报告快照** —— `R-exclusions-closed`：判据读磁盘上的
     `4-临时脚本/2026-08-15-round15/qa2/same_en_final.json`（第十五轮产物，15 组），
     而第十六轮库里实跑 `scan_same_en_split` 是 **14 组 / 123 叶**。这条断言因此
     **只与「上次谁跑过扫描并写了那个文件」一样新**：库里新分叉出来，它不会变红。
     ▸ 防法：**现场调用扫描器**（本轮改法），或至少校验报告 mtime 晚于两仓
       `compendium/cn` 的最新 mtime、不满足就判失败。**`min_hits` 与 `scan` 都防不住这一类** ——
       它「读了库」，只不过读的是库的一张过期照片。

（d）**闸在跑，但豁免表已经空了** —— 第十八轮收尾实测：两条块级断言的 `except_blocks` 里
# （e）**输入缺失但判据照跑** —— 判据依赖的某个输入（词表 / 基准 / bindings）没传或不存在时，
#     它不报错、不告警，直接以空输入跑完，打印一份漂亮的「0 条问题」。
#     2026-08-15 第二十轮实测：`scan_renamed_terms` 的 `--glossary` 默认是空串，
#     不显式传就静默跑成 `glossary={}`，探测器 B 每个候选都因「配不出中文写法」静默 continue，
#     最后 `findings 0`。**主控自己拿这个 0 当过一次「兜底成立」的证据。**
#     防法与前四种都不同：**判据必须在输入为空时吵**，并让 `--strict-coverage` 之类的严格模式非零退出。
#     通则：**「0 条问题」有两种含义 —— 「查过了没问题」和「无从查起」，判据有义务把两者分开。**
#
# （f）**判据自己的正则被转义吃掉，而主闸照样全绿**（第二十一轮实测，主控亲手制造）。
#     改 `_ENR_PARAM` 时用 python 的字符串替换传 ``，被当成**退格符 ** 写进文件，
#     正则开头成了一个退格符、**一条都匹配不到**。此时：
#       · `py_compile` 通过（语法没问题）；
#       · **主断言跑出来 59 通过 / 0 失败 —— 全绿**（正则匹配不到东西 ⇒ 没有可违反的 ⇒ 空转）；
#       · **只有 `--selftest` 报了 14/15**，是它唯一被抓住的地方。
#     ⇒ **通则一：改判据本身之后，光跑主闸不算验过，必须跑 `--selftest`。**
#     ⇒ **通则二：正则里的 `` 不要用改写脚本传** —— 这与 `R-catwalk` 那次被 JSON 转义吃掉
#       是同一类错误，只是这次吃它的是 Python 字符串。直接编辑文件，或用 `chr(92)+'b'` 构造。
#     ⇒ **通则三：一个正则多加一个候选组，所有取值处都要同步改。** 本次只改了
#       `scan_content_coverage.py` 的取值处、漏了本文件，`R-arcturel-arcturian-labels`
#       当场 TypeError —— 那一次是断言**抓住了**主控的 bug，按设计工作。
     合计躺着 **7 条**再也匹配不到的条目（登记的内容欠账都已修完），而全套依旧 52/0 全绿。
     豁免不命中只让 detail 里的 `n_exempt` 变小，**没有任何断言会因此变红，只能靠人记**。
     后果与 `R-dives-mine` 一样：欠账还清了、豁免却留着，遮住了同一处未来的回潮。
     ▸ 防法：`max_unused_exempt`（默认 0，见 `_unused_exempt`）。
       **前三种防法对这一类全部无效** —— 它判据没写坏、读了本次运行时的库、命中数也正常。

（g）**输入不缺，但被静默换成了另一份**（第二十五轮登记，`R-selfcheck-twin` 实测）。
     判据照跑、报告「我比过了 N 对」、结论是绿的，**但它比的根本不是你以为的那份**。
     实证：`a_twin_files` 里写 `ctx.repos.get(pair["repo_a"], pair["repo_a"])`，
     而规则里 `repo_a` 填的是目录名 `1-Ember汉化插件`、`ctx.repos` 的键却是 `ember`，
     于是 `.get(k, k)` **永远走兜底**，退化成裸相对路径按 **cwd** 解析。后果三连：
       · 把 `ctx.repos` 指到**注入了漂移的副本树**，它照样报「比对 1 对文件、0 违规」
         —— 因为它比的是真实树；
       · 它同时满足了 `min_pairs≥1` 那道**专防空转**的闸，还打印「我比过了」；
       · 主闸结论**依赖 cwd**：项目根 61/0，在 `3-常用脚本/qa` 下跑是 60/1。
     与（e）「输入缺失」的区别在于：**（e）少了输入，（g）换了输入**，
     所以（e）的防法「缺就吵」对（g）**完全无效** —— （g）的输入是「有」的。
     ▸ 防法一：**解析输入时不许有静默兜底。** `dict.get(k, k)` 这种「取不到就拿键本身当值」
       的写法是重灾区；取不到就当场失败（见 `_twin_repo_dir` 的 KeyError）。
     ▸ 防法二：**灵敏度回测必须在副本树上做。** 只在真实树上验，等于没验 ——
       这条判据当初「在真实树上验过、会响」，而它在副本树上根本没被驱动到。
     ⚠ 这一条的发现过程本身值得记：它是**被另一个判据的回测流程逼出来的** ——
       有人想给它做灵敏度回测，才发现它压根没在读副本树。
       ⇒ **通则：「这条判据能不能被回测」本身就是一个判据。** 写完新断言，
         先问「我能不能往 `--root` 副本里注入违规让它响」，答不上来的就是（g）的候选。

（h）**判据没空转，空转的是给判据喂输入的那个探针**（第二十六轮登记，第二十五轮 `probe_world_c.mjs` 实测）。
     前七种说的都是「判据自己坏了」。这一种反过来：**判据本身是好的，它验的输入是自己捏的。**
     实证：上一轮想验「真实 Foundry 里 `The Abyss` 会不会被『合集索引条目名』那份语料接住」，
     探针把英文基准里**所有层级**的 name 拍平成顶层 index 条目喂进去，于是「接住了」——
     而真实 Foundry **从不产出那种 index**：`The Abyss` / `Heart of Ember` 是 Adventure 文档
     **内层**的 `journal[].pages[].name`，而 `Adventure.metadata.compendiumIndexFields` 只有
     `_id/name/caption/description/img/sort/folder/flags.core.sheetClass`，crucible 只给
     Item / ActiveEffect 追加过索引字段、**没给 Adventure 加** ⇒ 真实 `pack.index` 里根本没有
     `journal`、更没有 `pages`。真实世界的数与离线一模一样（4 键 / 7 报文），
     而基于那份捏出来的 index 写下的「届时剩 3 键 / 3 报文」是**假的**。
     ▸ 防法：**探针喂给判据的模拟输入，其字段形状必须能指到上游契约的出处**
       （这里是 `compendiumIndexFields` / 上游哪一行 API）。
       **不能从「数据文件里有这个字段」推出「运行时拿得到这个字段」** —— 这两件事之间
       隔着一层上游的投影规则，而那层规则才是契约。
     ▸ 与前七种的关系：`min_hits` / `scan` / 现场重跑 / `max_unused_exempt` / 「缺就吵」
       **全部无效** —— 判据读了库、命中数正常、豁免也活着，坏掉的是**上游那一侧**的模拟。
     ⇒ 本轮的 `translate_cases` 就是按这条写的：全量编排名**不抄我们自己的 ARRANGEMENTS 表**
       （那是被判方），而是从上游 `ember.mjs` 的 `soundscapes` 注册表现抠 `arrangements[].label`
       —— 那正是 `ember.mjs:16266` 里 `${channel.capitalize()}: ${arrangement.label}` 的来源。

⚠ 第二十八轮实测到一个**还没登记**的形态（候选第九种，登记处是 PROJECT.md §3.7，本轮不由
  本文件擅自编号）：**判据在跑、也真的在判，但它的全部强度来自它自己规则文件里的一个可调自报数。**
  实证：`R-patterns-translate-cases` 那套「谁加表项不补用例就当场红」的纪律，复核把
  `coverage.min_prefixed` 从 19 改回 2、`min_negative` / `min_positive` 放到 1，
  **其余一个字节不动 → 违规 0，闸变绿**（对照：删一条用例 4/4 立刻红）。
  它与 (d)「豁免表已空」的区别：(d) 是判据**没东西可判**了，这一种是判据**判了、但门槛低到判不着**。
  ▸ 防法一：**可调的数一律两层**（现算 + 不得低于历史，见 `_two_layer_floor`）——
    「调松阈值」和「删东西」两条路一起断。
  ▸ 防法二（更硬）：**能写成不含阈值的恒等式就别写成阈值**。
    本轮的三处示范：叶子表 ⊆ 上游现抠的编排名（`translate_cases_runner.mjs` ⑥）·
    逐表行 checked 之和 == 合计自报的 rawChecked（`a_panel_liveness`）·
    6 个自造串必须 6/6 报出。恒等式**没有可调的旋钮**，也就没有这条作弊路径。
  ▸ 防法三：**回测要往规则里注入变异**，不只往被判文件里注入 ——
    这条作弊路径的形状是「库和判据都没动，只把规则里的自报数调松」，
    只在被判文件上做灵敏度回测的话，它**一次都不会被触发**（见 `run_translate_selftest` 14–17）。

通则一句话：**任何断言都必须能说出「我这次扫了多少叶 / 多少个键」，说不出来的是自检不是断言。**
推论：判据的数据来源必须是**本次运行时现读的库**；凡是从磁盘上另一个文件里拿结论的，
都要能证明那个文件比库新，否则就是形态 (c)。
推论二（形态 (h)）：**判据的模拟输入必须指得出上游契约的出处**；指不出的，
它证明的只是「在我捏的那个世界里成立」。

所以本轮起：`distinct_terms` 的每个术语都必须声明能读库的英文闸（lang / compendium），
没有 `scan` 的 `distinct_terms` 规则**直接判失败**（见 `a_distinct_terms` 顶部）；
`exclusions_closed` 现场重跑 `scan_same_en_split.py`（见 `a_exclusions_closed` 顶部）。

已知局限（**别把它当成全覆盖**，这是本项目方法教训 1 的又一处应用）
-----------------------------------------------------------------
1. `term_gated` 是**叶级**的：只要求该叶中文里**出现过**定译。一叶里提到该术语 5 次、
   只错 1 次的情况**它抓不到**。回测实测：把一叶里 3 处「邪术师」全改成「术士」才会响，
   只改 1 处不响。要抓叶内部分错译，得做逐位对齐（成本高得多），本轮没做。
   ▸ **第十八轮 Y2 做了**：`block_aligned_gate` / `block_sense_gate` 两个新类型把「按块级
     标签切块、再逐块对齐」固化下来，专门补三条断言各自 why 里写死的叶内盲区
     （R-shard-god 的 70 叶 · R-arcturel-vs-arcturian 的叶内串行 · R-rank-sense-compendium
     的混合叶／无法分类叶与缺失的正向闸）。见 `a_block_aligned_gate` 的 docstring。
     ⚠ 它**不是万能的替代品**：叶级判据仍然管着「整叶一次都没提」这种情况，两者互补。
   ⚠ 反过来，叶级也会**假阳性**：一叶里同时出现 `Shard God` 与 `Shard Gods` 时，
   中文只要重写了其中一种说法，另一条闸就会误报。
   ▸ 第十六轮终段的**绕法**（`R-shard-god`）：不去做逐位对齐，而是把闸只下在
     **英文侧不含歧义**的那些叶上 —— `(?s)^(?!.*<另一形态>).*<本形态>` 这种
     「只含单数 / 只含复数」的负向先行断言。实测单数专属 264 叶 264/264 干净、
     复数专属 117 叶 114/117 干净。同时含两种形态的 70 叶仍然不查，那是本判据的边界。
     这个形态可以复用到任何「同一词根多形态、叶内混排」的裁决上。
2. `cn_absent` 反过来会**误伤合法用法**：某个废弃写法如果在别处是正当中文，会假阳性。
   目前每条都验过全库为 0，加新条目前务必先数一遍。必要时用 `except_paths` 登记豁免
   （可以写成 `{"代币": [...]}` 形态，只豁免某一个词，别把整叶从所有词的检查里摘出去）。
3. 断言只覆盖第 8 节里**能机械表达**的那些裁决。像「改动面小的那边优先」「name 与正文
   冲突时多数该改 name」这类**方法性**裁决，本质上表达不成断言，仍然只能靠人读第 8 节。
4. lang 侧英文闸**先剥占位符**（`{rank}` `{level}` 之类）再匹配。不剥的话
   `HAZARD.TooltipDamage` 那种「英文只在 `{rank}` 里出现、中文当然没有对应汉字」的键
   会被判违规。compendium 侧**不剥** —— 那里的 `@UUID[...]{Shard Gods}` 花括号内是正文标签。

自检与回测
----------
`--selftest` 跑判据自身的正反例（双语尾巴那条最容易写错，第一版就把「圣堂区路人 A」
这种编号后缀全报成了尾巴）。
`--root <另一棵树>` 用于灵敏度回测：往副本里注入违规，确认断言真的会响 ——
**只测特异度（全绿）是不够的**，那样「所有断言都返回空」也能过。

⚠ 第二十五轮补的通则：**「这条断言能不能被 `--root` 回测」本身就是一个判据。**
`R-selfcheck-twin` 就是在有人想给它做回测时，才发现它压根没在读副本树（形态 (g)）。
所以 `twin_files` 现在把副本树回测**直接固化进 `--selftest`**（见 `run_twin_selftest`）：
临时目录里现造一棵副本树、注入漂移、并**断言报出来的 hash 是副本树那两份的** ——
只断言「它红了」不够，红也可能是它跑去比了真实树、而真实树恰好正在漂。

主闸结论**不得依赖 cwd**：在项目根与在 `3-常用脚本/qa` 下跑必须一致。
不一致就说明某条断言在用裸相对路径（形态 (g) 的典型征兆）。
"""
import argparse
import collections
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # 项目根
DEFAULT_RULES = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
REPOS = {"ember": "1-Ember汉化插件", "crucible": "2-Crucible汉化插件"}

# 双语尾巴：中文 …… 空白 + 拉丁串结尾。
#
# ⚠ **必须排除单字母编号。** 第一版写成 `[A-Z][A-Za-z'\-]*`，把「圣堂区路人 A」
# 「菌丝旷野底图 B」这类**编号后缀**全报成双语尾巴（14 处全是假阳性）——
# 而那正是既定约定要求的写法（英文侧就是 `Hallows Passerby A`，字母是编号本体，
# 前面留半角空格是库内 name 的惯例）。
#
# 判据取与 `scan_same_en_split` 的归一规则同源的定义：尾巴是**英文名本身**被附在中文后面，
# 所以要求结尾那串拉丁**至少有 2 个字母**。单个字母（或数字）是编号，不是名字。
_TAIL = re.compile(r"[一-鿿].*?\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)$")


def has_bilingual_tail(text):
    m = _TAIL.match(text.strip())
    if not m:
        return False
    letters = re.sub(r"[^A-Za-z]", "", m.group(1))
    return len(letters) >= 2


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def load_pack_pairs(repo_dir):
    """产出 (pack, path, en, cn)；只收两侧都有的叶。"""
    en_dir = os.path.join(repo_dir, "compendium", "en")
    cn_dir = os.path.join(repo_dir, "compendium", "cn")
    if not os.path.isdir(en_dir):
        return
    for fname in sorted(os.listdir(en_dir)):
        if not fname.endswith(".json") or fname == "_source.json":
            continue
        cn_path = os.path.join(cn_dir, fname)
        if not os.path.exists(cn_path):
            continue
        en = dict(walk(json.load(open(os.path.join(en_dir, fname), encoding="utf-8-sig"))))
        cn = dict(walk(json.load(open(cn_path, encoding="utf-8-sig"))))
        for p, ev in en.items():
            cv = cn.get(p)
            if cv is not None:
                yield fname, p, ev, cv


def load_lang_pairs(repo_dir, upstream_dir):
    """产出 (key, en, cn)；只收两侧都有的键。

    英文侧取的是**上游安装目录**里的 `lang/en.json`（模块/系统本体），
    中文侧是本仓的 `lang/cn.json`。两边都要 `walk` 展平：上游 en.json 是嵌套的，
    而本仓 cn.json 是扁平的（`flatten_lang.py` 的产物），不展平会得到 92 : 486 这种假差异。
    """
    en_p = os.path.join(upstream_dir, "lang", "en.json")
    cn_p = os.path.join(repo_dir, "lang", "cn.json")
    if not (os.path.exists(en_p) and os.path.exists(cn_p)):
        return
    en = dict(walk(json.load(open(en_p, encoding="utf-8-sig"))))
    cn = dict(walk(json.load(open(cn_p, encoding="utf-8-sig"))))
    for k, ev in en.items():
        cv = cn.get(k)
        if cv is not None:
            yield k, ev, cv


class Ctx:
    """一次性把两个仓库读进来，所有断言共用（否则每条断言重读一遍太慢）。"""

    def __init__(self, repos, meta=None):
        self.repos = repos
        self.meta = meta or {}
        self.pairs = {}
        for name, d in repos.items():
            self.pairs[name] = list(load_pack_pairs(d))
        # lang 通道。上游路径写在 rules 的 meta.lang_sources 里（与 R-lang-parity 同源）。
        # ⚠ 这里**故意不受 `--root` 影响**：英文基准永远取真实安装目录，
        # 灵敏度回测只需要往副本树的 `lang/cn.json` 里注入违规就能生效。
        self.lang = {}
        for name, src in (self.meta.get("lang_sources") or {}).items():
            if name in repos:
                self.lang[name] = list(load_lang_pairs(repos[name], os.path.expandvars(src)))

    def all_pairs(self, scope):
        for name in (scope or self.repos.keys()):
            for row in self.pairs.get(name, []):
                yield (name,) + row

    def all_lang(self, scope):
        for name in (scope or self.lang.keys()):
            for k, ev, cv in self.lang.get(name, []):
                yield name, k, ev, cv


# lang 侧英文闸要先剥掉占位符再匹配 —— 见模块 docstring 的「已知局限 4」。
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def _paths_matcher(spec):
    """把 except_paths 编成一个「路径是否豁免」的函数。空 spec 返回 None。"""
    if not spec:
        return None
    pats = [re.escape(s) if not s.startswith("re:") else s[3:] for s in spec]
    rx = re.compile("|".join(pats))
    return lambda path: bool(rx.search(path))


# ============================== 「这次判了几条规矩」计数器（第三十一轮 L5-①·2）
#
# ⚠ **要害在于它数的是什么。** 复核第三十轮末尾点出来的那一句，本轮照它设计：
#     反空转的那个数，必须数的是「**我判了几条规矩**」，不是「我扫了几片叶子」。
#   实证：`R-lantyr` 的 `cn_forbidden` 一清空，detail 里那句「英文闸命中 244 叶」
#   **一个字节都不变** —— 禁用变体那一半悄悄没了，而屏幕上的数照旧。
#   「扫了多少叶」是**语料规模**，与「判据还剩几条」正交；拿前者当后者用，
#   等于用体重证明自己没生病。
#
# 一条「规矩」的定义（**两个条件缺一不可**）：
#   ① 它是**载荷里的一个可判项**（一个禁用词 / 一个域 / 一个字段名 / 一个必含串 …）；
#   ② 它这次**真的被应用到了实数据上** —— 语料是空的、闸一叶都没命中、
#      文件读不到，都不许记。所以这个数同时看住「载荷被掏空」与「输入没到位」两侧。
#
# 计数用**集合**而不是累加：同一条规矩在 4 万叶上判 4 万次仍然只算一条，
# 于是这个数**与语料规模无关**、只与「载荷还剩几项」有关 —— 库里增删译文不会让它抖，
# 而清空一个载荷键会让它当场掉下来。这正是「地板 ＝ 当前实际值」能钉死的前提。
_UNITS = set()


def _unit(*tags):
    """记下「这条规矩这次真的判了」。只在**已经落到实数据上**的位置调用。"""
    _UNITS.update(tags)


# 每个 kind 的执行体这一进程里被调用了几次。`--selftest` 的 kind 覆盖表要用它 ——
# ⚠ 「有一组用例」不等于「那一组真的调到了执行体」：本项目实测，`glossary_value` 组
#   打的是 `glossary_value_matches` 这个**纯函数**、`双语尾巴` 组打的是 `has_bilingual_tail`、
#   `读库闸` 组打的是 `_gate_one` —— 三组都在跑，`a_glossary_value` / `a_no_bilingual_tail` /
#   `a_term_gated` 三个执行体的调用次数却都是 **0**。所以覆盖表必须用计数器验，不能用名册验。
_KIND_CALLS = {}


# ----------------------------------------------------------------- 断言实现

def a_term_gated(rule, ctx):
    # ⚠ 大小写不敏感是**默认**。本项目已经被同一个坑咬过两次：
    # `split_region_area_map.py` 的 `\b(Region|Area) Maps?\b` 漏掉小写形态，
    # 导致 72 叶被误判成「需人判」、主控据此判定「全库拆分做不了」——判据错了结论就跟着错。
    # 专名的大小写在本库里从来不是判据的一部分，所以默认忽略，要区分请显式写 "case_sensitive": true。
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    en_re = re.compile(rule["en"], flags)
    # `cn_required` 是可选的：有些裁决只有「禁用写法」而没有「每叶都必须出现的定译」——
    # 例如 `Token`，345 个命中叶里有 60 叶的英文是普通名词义（a token of good luck），
    # 中文当然不含「指示物」。那种规则写成「闸内零容忍某几个词」才对，
    # 硬要求 cn_required 只会造出 60 条假阳性。
    req = rule.get("cn_required")
    forbid = rule.get("cn_forbidden", [])
    # 有意的例外（例：`Drakeling Scales`=幼龙鳞片 —— 材料名不是生物指称，
    # 「龙兽鳞」不是中文词而「龙鳞」是。登记在这里，免得下一轮「顺手统一」）
    except_re = _paths_matcher(rule.get("except_paths"))
    bad = []
    hits = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if not en_re.search(ev):
            continue
        if except_re and except_re(path):
            continue
        hits += 1
        # 判了几条规矩：`cn_required` 一条 + 每个 `cn_forbidden` 变体各一条。
        # ⚠ 记在**闸命中之后**，所以闸死了这些数一起归零；`cn_forbidden` 清空则少几条。
        if req:
            _unit(f"req:{req}")
        _unit(*(f"forbid:{f}" for f in forbid))
        if req and req not in cv:
            bad.append((repo, pack, path, f"英文命中但中文无「{req}」"))
            continue
        for f in forbid:
            if f in cv:
                bad.append((repo, pack, path, f"中文同时出现禁用写法「{f}」"))
                break

    # ⚑ **命中 0 叶不是「通过」，是「这条断言根本没在跑」。**
    # 实测代价：`R-catwalk` 的 `en` 在 JSON 里写成了 `"\bcatwalk"`（单反斜杠），
    # 被 JSON 当成**退格符**吃掉，正则变成 `\x08catwalk`，匹配不到任何东西 ——
    # 断言一路报绿，而库里实际有 10 叶违规。这正是本项目「静默全绿」那一类失败。
    min_hits = rule.get("min_hits", 1)
    if hits < min_hits:
        bad.append(("-", "-", rule["en"],
                    f"英文闸只命中 {hits} 叶（要求 ≥{min_hits}）—— 这条断言在空转，"
                    f"多半是正则被 JSON 转义吃掉了（`\\b` 要写成 `\\\\b`），或上游改了措辞"))
    return bad, f"英文闸命中 {hits} 叶"


def a_cn_absent(rule, ctx):
    """某些中文写法在全库不得出现。

    第十六轮改了三处：
    1. `cn` 可以是**一组**词（`Token` 那条要同时禁「令牌」和「代币」）。
    2. 扫描面从 compendium 扩到 **compendium + lang**。原来只扫 compendium ——
       与 `distinct_terms` 不读库是同一类盲区，只是没那么显眼。实测本轮 18 个禁用词
       在 lang 侧全为 0，所以这次扩面不产生任何新失败，纯属补洞。
    3. 可选的 `en` 英文闸**只用来算命中数**（`min_hits` 反空转），不改变「全库零容忍」的语义。
       意义是：万一上游把这个词整个删了，`cn_absent` 会永远报绿而没人知道它已经不设防。
    """
    needles = rule["cn"]
    if isinstance(needles, str):
        needles = [needles]
    raw_exc = rule.get("except_paths") or {}
    if isinstance(raw_exc, list):                     # 一份豁免表管所有词
        raw_exc = {n: raw_exc for n in needles}
    exc = {n: _paths_matcher(v) for n, v in raw_exc.items()}

    bad = []
    n_leaf = n_key = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        n_leaf += 1
        for needle in needles:
            if needle in cv and not (exc.get(needle) and exc[needle](path)):
                bad.append((repo, pack, path, f"出现了已废弃的写法「{needle}」"))
    for repo, key, ev, cv in ctx.all_lang(rule.get("scope")):
        n_key += 1
        for needle in needles:
            if needle in cv and not (exc.get(needle) and exc[needle](key)):
                bad.append((repo, "lang/cn.json", key, f"出现了已废弃的写法「{needle}」"))

    # 判了几条规矩：每个禁用词一条 —— 但**只在真的有语料可判时**才记（形态 (b)/(e)）。
    if n_leaf or n_key:
        _unit(*(f"cn:{n}" for n in needles))

    detail = (f"扫 {n_leaf} 叶 + {n_key} 个 lang 键，禁用词 {len(needles)} 个"
              f"（本次判了 {len(needles) if (n_leaf or n_key) else 0} 条规矩）")
    if rule.get("en"):
        flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
        en_re = re.compile(rule["en"], flags)
        hits = sum(1 for _, _, _, ev, _ in ctx.all_pairs(rule.get("scope")) if en_re.search(ev))
        hits += sum(1 for _, _, ev, _ in ctx.all_lang(rule.get("scope"))
                    if en_re.search(_PLACEHOLDER.sub(" ", ev)))
        _unit("en-gate")                            # 反空转的英文闸本身也是一条规矩
        detail += f"；英文闸 `{rule['en']}` 命中 {hits}"
        min_hits = rule.get("min_hits", 1)
        if hits < min_hits:
            bad.append(("-", "-", rule["en"],
                        f"英文闸只命中 {hits}（要求 ≥{min_hits}）—— 这条 cn_absent 已经不设防了："
                        f"要么上游把这个概念删了/改了措辞，要么正则被 JSON 转义吃掉"))
    return bad, detail


_TAGS = re.compile(r"<[^>]+>")


def a_sense_gated(rule, ctx):
    """同一个英文词有**机制义**和**普通名词义**两支，只有机制义该用那个定译。

    为什么需要一个新类型（第十六轮终段补）
    --------------------------------------
    `R-tier-rank-level` 的 `scan` 只写了 `lang`，实测只看住 62 个 lang 键；
    而第十六轮在 **compendium 侧改了 176 叶**（英文闸 IGNORECASE 下 阶位 221 : 等级 51），
    那一整片**没有任何闸看守**。可是 compendium 侧又不能照搬 `distinct_terms`：
    英文 `rank` 在本库里是两个词 ——

      · 机制义：`You gain the Novice rank in Arcana` / `Attunement Rank 1` / `Soulbound Rank`
      · 普通名词义：`denote their civic rank`（公民地位）/ `within the ranks of Shard Gods`（行列）
        / `rank as a Commander`（军衔）/ `rank-and-file`（普通一兵）

    —— 后者**不归那条裁决管**，硬要求它们含「阶位」会造出成片假阳性。

    判据（GAME/COMMON 两张正则表与 `4-临时脚本/2026-08-15-round16/probes/split_rank.py` 同源，
    COMMON 优先级高于 GAME）把每一处出现按**剥标签后的上下文窗口**分类，
    然后**只闸得住的那两个方向**：

      ① 反向闸 `require_en_support`：中文用了定译，英文侧却一个该词都没有 → 违规。
         实测全库 40674 叶里「阶位」出现在 221 叶，**221 叶全部**英文含 `rank`，零违规。
         这一条抓的是「整词替换扫到了别的义项」（把 Tier / level 的叶一起刷成阶位）。
      ② 普通名词义专属叶 `forbid_when_all_common`：该叶所有出现都判 COMMON → 中文不得含定译。
         实测 96 叶，**无一含「阶位」**，零违规。这一条抓的是「顺手全库统一」。

    ⚠ **判据边界，别把绿读成全覆盖**：**故意不做**「GAME → 中文必须含阶位」那个正向闸。
    实测纯 GAME 的 230 叶里有 13 叶中文正当地没有「阶位」——
    `1 rank of exhaustion`（＝1 层力竭）· `close ranks with enemy characters`（＝并肩结阵）·
    `join their ranks`（＝加入他们）· 更新日志里被整段改写的 `Soulbound (rank 1 only)`。
    分类器把它们判成 GAME 是因为窗口里有 `exhaustion` / `skill` 这些词，这是分类器的粗糙处，
    不是译文的错。上正向闸就是 13 条假阳性，所以这条断言**证明不了**「每一处机制义都译对了」。
    """
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    occ = re.compile(rule["en"], flags)
    game = re.compile(rule["sense"]["game"], flags)
    common = re.compile(rule["sense"]["common"], flags)
    win = rule.get("window", 90)
    cn = rule["cn"]
    exc = _paths_matcher(rule.get("except_paths"))
    bad = []
    n_en = n_cn = 0
    kinds_n = {"GAME": 0, "COMMON": 0, "UNKNOWN": 0, "MIX": 0}

    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        has_cn = cn in cv
        if has_cn:
            n_cn += 1
        ms = list(occ.finditer(ev))
        if not ms:
            if has_cn and rule.get("require_en_support", True) and not (exc and exc(path)):
                bad.append((repo, pack, path,
                            f"中文用了机制义定译「{cn}」，而英文侧一个 `{rule['en']}` 都没有 ——"
                            f"多半是整词替换扫到了别的义项（Tier / level）"))
            continue
        n_en += 1
        # 判了几条规矩：出现闸 + 两张分类表 + 开着的那两个方向闸。
        _unit("occ", "sense:game", "sense:common")
        if rule.get("require_en_support", True):
            _unit("dir:require_en_support")
        if rule.get("forbid_when_all_common", True):
            _unit("dir:forbid_when_all_common")
        seen = set()
        for m in ms:
            window = _TAGS.sub(" ", ev[max(0, m.start() - win): m.end() + win])
            if common.search(window):
                seen.add("COMMON")
            elif game.search(window):
                seen.add("GAME")
            else:
                seen.add("UNKNOWN")
        if "UNKNOWN" in seen:
            bucket = "UNKNOWN"
        elif seen == {"COMMON"}:
            bucket = "COMMON"
        elif seen == {"GAME"}:
            bucket = "GAME"
        else:
            bucket = "MIX"
        kinds_n[bucket] += 1
        if bucket == "COMMON" and has_cn and rule.get("forbid_when_all_common", True):
            if not (exc and exc(path)):
                bad.append((repo, pack, path,
                            f"这一叶的 `{rule['en']}` 全部是普通名词义（公民地位／行列／军衔），"
                            f"中文却用了机制义定译「{cn}」"))

    detail = (f"英文闸命中 {n_en} 叶（机制义 {kinds_n['GAME']} / 普通名词义 {kinds_n['COMMON']} / "
              f"混合 {kinds_n['MIX']} / 无法分类 {kinds_n['UNKNOWN']}）· 中文含「{cn}」{n_cn} 叶")
    for field, got, why in (("min_en_leaves", n_en, "英文闸"),
                            ("min_cn_leaves", n_cn, f"中文「{cn}」"),
                            ("min_common_leaves", kinds_n["COMMON"], "普通名词义分桶")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{why}只数到 {got}（要求 ≥{want}）—— 这条断言在空转："
                        f"要么正则被 JSON 转义吃掉，要么上游改了措辞，要么分类表失效"))
    return bad, detail


def _gate_one(term, ctx, scan, scope):
    """一个术语的读库核对。返回 (命中数, 违规列表)。

    `en_gate` 缺省是 `\\b<en>\\b`；`cn_gate` 缺省是定译本身 —— 但允许写一个**更宽的
    可接受形态**：`level` 的定译是「等级」，而 lang 里 `Level {x}` 正当地写成「{x} 级」，
    所以 `cn_gate` 取「级」。放宽的每一处都要在规则的 `why` 里写清为什么。
    """
    en_gate = term.get("en_gate") or rf"\b{re.escape(term['en'])}\b"
    flags = 0 if term.get("case_sensitive") else re.IGNORECASE
    rx = re.compile(en_gate, flags)
    need = term.get("cn_gate", term["cn"])
    forbid = term.get("cn_forbidden", [])
    exc = _paths_matcher(term.get("except_paths"))
    hits = 0
    bad = []

    def _mark():                                   # 闸真的命中一叶/一键之后才记
        _unit(f"gate:{en_gate}→{need}")
        _unit(*(f"gate:{en_gate}✗{f}" for f in forbid))

    if "lang" in scan:
        for repo, key, ev, cv in ctx.all_lang(scope):
            if not rx.search(_PLACEHOLDER.sub(" ", ev)):
                continue
            if exc and exc(key):
                continue
            hits += 1
            _mark()
            if need not in cv:
                bad.append((repo, "lang/cn.json", key,
                            f"英文是 {ev[:60]!r}，中文里没有「{need}」：{cv[:50]!r}"))
            for f in forbid:
                if f in cv:
                    bad.append((repo, "lang/cn.json", key, f"中文出现禁用写法「{f}」：{cv[:50]!r}"))
    if "compendium" in scan:
        for repo, pack, path, ev, cv in ctx.all_pairs(scope):
            if not rx.search(ev):
                continue
            if exc and exc(path):
                continue
            hits += 1
            _mark()
            if need not in cv:
                bad.append((repo, pack, path, f"英文命中 `{en_gate}` 但中文无「{need}」"))
            for f in forbid:
                if f in cv:
                    bad.append((repo, pack, path, f"中文出现禁用写法「{f}」"))
    return hits, bad


def a_distinct_terms(rule, ctx):
    """一组 {en, cn} 的中文必须两两不同，**并且逐个过英文闸读库核对**。

    ⚠ 旧版只做前半句（注释原文：「纯配置自检，不读库」）。后果实测：`R-region-area-map`
    一直全绿，而 ember lang 的 `EMBER.CALENDAR.REGION` 反着写成「区域地图」，
    四个发布版没人发现 —— 断言不读库就等于没有断言。
    所以现在**没有 `scan` 的规则直接判失败**，不给「只做配置自检」留后门。
    """
    seen = {}
    bad = []
    for item in rule["terms"]:
        cn = item["cn"]
        if cn in seen:
            bad.append(("-", "配置", item["en"], f"与「{seen[cn]}」共用中文「{cn}」"))
        seen[cn] = item["en"]

    scan = rule.get("scan")
    if not scan:
        bad.append(("-", "配置", rule["id"],
                    "这条 distinct_terms 没有声明 `scan` —— 它只比了配置、一个字节的库都没读，"
                    "正是 R-region-area-map 空转四个版本的那种形态。请补 lang 或 compendium 闸"))
        return bad, f"{len(rule['terms'])} 个术语（**未读库**）"

    total = 0
    per = []
    for item in rule["terms"]:
        if not item.get("gate", True):               # 显式声明「这个词没有可用闸」的，要写 why
            per.append(f"{item['en']}:—")
            continue
        h, b = _gate_one(item, ctx, scan, rule.get("scope"))
        total += h
        per.append(f"{item['en']}:{h}")
        bad.extend(b)

    # 判了几条规矩：逐个术语的读库闸由 `_gate_one` 自己记；这里补「两两不重名」那一条 ——
    # 它是**配置层**的判据，只有在读库闸真的跑起来之后才算数（否则就是「只比配置、不读库」，
    # 正是 R-region-area-map 空转四个版本的形态）。
    if total:
        _unit(f"distinct:{len(rule['terms'])}")

    min_hits = rule.get("min_hits", 1)
    if total < min_hits:
        bad.append(("-", "配置", rule["id"],
                    f"英文闸合计只命中 {total}（要求 ≥{min_hits}）—— 这条断言在空转"))
    return bad, f"{'+'.join(scan)} 闸命中 {total}（{' '.join(per)}）"


def a_term_domains(rule, ctx):
    """同一个英文词按**域**分裂成多个中文，逐域钉死。

    这是 `distinct_terms` 的孪生形态，区别在于：`distinct_terms` 管的是**不同英文**不许
    共用中文；`term_domains` 管的是**同一个英文**必须按域保持不同中文 ——
    后者更容易被下一轮「按多数派统一」一刀切掉，因为从中文侧看它就像一处分裂。

    每个域可以由 lang 键（`lang`）或 compendium 英文闸（`gates`）界定，
    并各自带 `cn_forbidden`（＝别的域的中文不许渗进来）。
    """
    bad = []
    total = 0
    per = []
    for dom in rule["domains"]:
        name = dom.get("name", "?")
        n = 0
        forbid = dom.get("cn_forbidden", [])
        for repo, keys in (dom.get("lang") or {}).items():
            langmap = {k: (ev, cv) for _, k, ev, cv in ctx.all_lang([repo])}
            for key, want in keys.items():
                if key not in langmap:
                    bad.append((repo, "lang/cn.json", key,
                                f"这个 lang 键不见了（上游改键名？）—— 域「{name}」失去看守"))
                    continue
                n += 1
                # 判了几条规矩：每个 lang 键的「必须含」一条，加上该域每个禁用写法各一条。
                _unit(f"dom:{name}:{key}")
                _unit(*(f"dom:{name}✗{f}" for f in forbid))
                ev, cv = langmap[key]
                if want not in cv:
                    bad.append((repo, "lang/cn.json", key,
                                f"域「{name}」要求中文含「{want}」，实为 {cv[:40]!r}（英文 {ev[:40]!r}）"))
                for f in forbid:
                    if f in cv:
                        bad.append((repo, "lang/cn.json", key,
                                    f"域「{name}」里混进了别域的写法「{f}」：{cv[:40]!r}"))
        for g in (dom.get("gates") or []):
            # ⚠ 字段名要翻译一次：`gates[].en` 与 term_gated 一样是**原始正则**，
            # 而 `_gate_one` 里的 `en` 是**字面词**（会被 re.escape 后套上 \b）。
            # 第一版直接 `{**g}` 传过去，`\b(Fear|Command) Aura\b` 被整个 escape 成字面串，
            # 闸命中 0 —— 幸亏 min_hits 把它抓出来了，这正是护栏该干的事。
            h, b = _gate_one({"en_gate": g["en"], "cn": g.get("cn_required", ""),
                              "cn_forbidden": g.get("cn_forbidden", []),
                              "except_paths": g.get("except_paths"),
                              "case_sensitive": g.get("case_sensitive")},
                             ctx, ["compendium"], rule.get("scope"))
            n += h
            if h:                                  # 域内每条 compendium 闸各算一条规矩
                _unit(f"dom:{name}:gate:{g['en']}")
            bad.extend(b)
        total += n
        per.append(f"{name}:{n}")

    min_hits = rule.get("min_hits", 1)
    if total < min_hits:
        bad.append(("-", "配置", rule["id"],
                    f"三域合计只命中 {total}（要求 ≥{min_hits}）—— 这条断言在空转"))
    return bad, f"合计 {total}（{' / '.join(per)}）"


def a_lang_parity(rule, ctx):
    bad = []
    detail = []
    for name, pkg in rule["packages"].items():
        # ⚠ 形态 (g) 的同类洞：仓名写错时**不许静默跳过**（那会让这条断言无声地
        #   一个包都不查、照样返回空、照样绿）。写错 = 硬错误；被 `--repo` 限定掉 = 跳过。
        if name not in REPOS:
            raise KeyError(f"packages 里的仓名 {name!r} 不在 REPOS（可选：{sorted(REPOS)}）"
                           f" —— 规则写错了，不许静默跳过（空转形态 (g)）")
        repo = ctx.repos.get(name)
        if not repo:                                   # 被 `--repo` 限定掉，不是写错
            continue
        cn_p = os.path.join(repo, "lang", "cn.json")
        en_p = os.path.join(os.path.expandvars(pkg), "lang", "en.json")
        if not (os.path.exists(cn_p) and os.path.exists(en_p)):
            bad.append((name, "-", cn_p, "lang 文件找不到，无法核对"))
            continue
        cn = dict(walk(json.load(open(cn_p, encoding="utf-8-sig"))))
        en = dict(walk(json.load(open(en_p, encoding="utf-8-sig"))))
        _unit(f"parity:{name}")                    # 每个包的键数对比各算一条规矩
        detail.append(f"{name}: cn {len(cn)} / en {len(en)}")
        if len(cn) != len(en):
            bad.append((name, "-", "lang/cn.json", f"键数 {len(cn)} != 英文侧 {len(en)}"))
    return bad, " | ".join(detail)


def a_anchor_ids(rule, ctx):
    pat = re.compile(r"<h[1-6][^>]*\sid=", re.IGNORECASE)
    n = n_leaf = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        n_leaf += 1
        n += len(pat.findall(cv))
    if n_leaf:                                     # 有语料可数，「不得低于阈值」这条规矩才算判了
        _unit("anchor:min")
    if n < rule["min"]:
        return [("-", "-", "-", f"标题显式 id 只有 {n} 个，低于阈值 {rule['min']}")], f"实测 {n}"
    return [], f"实测 {n} 个（阈值 {rule['min']}）"


def a_no_bilingual_tail(rule, ctx):
    fields = tuple(rule["fields"])
    bad = []
    n = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        seg = path.replace("[", ".").split(".")
        # 字段名可能是最后一段（tokenName），也可能是倒数第二段（encounterTokens.<名>）
        hit = (seg[-1] if seg[-1] in fields
               else seg[-2] if (len(seg) >= 2 and seg[-2] in fields) else None)
        if hit is None:
            continue
        n += 1
        _unit(f"field:{hit}")                      # 每个**真的匹配到叶**的字段名各算一条规矩
        if has_bilingual_tail(cv):
            bad.append((repo, pack, path, f"带双语尾巴：{cv[:40]!r}"))
    # ⚠ 这里就是复核点名的那一处：「检查 0 个叶」早就印在屏幕上了，缺的只是一句判定。
    #   但**光判叶数不够** —— 叶数是语料规模；`fields` 从 4 个砍到 1 个，叶数只掉一点点，
    #   而看守的字段少了 3 个。所以真正的地板挂在上面那个 `_unit`（判了几条规矩）上，
    #   叶数只作为「输入到位没到位」的第二个数一起报出来。
    return bad, (f"检查 {n} 个该约定下的叶；"
                 f"看守字段 {len(fields)} 个，本次真的判到叶的 "
                 f"{len({t for t in _UNITS if t.startswith('field:')})} 个")


def _exc_key(en, rule):
    """把一个分叉组的英文串压成能在豁免表里查的键。

    短串（专名 / 短标签）直接用原串；长正文用配置里给的**特征词**（`long_keys`），
    因为豁免表记的是「`Adelyne` 那条」而不是整段英文。找不到特征词就退回前 30 字符。
    """
    if len(en) < rule.get("short_len", 40):
        return en
    m = re.search("|".join(rule.get("long_keys", ["Adelyne", r"lookup @name"])), en)
    return m.group(0) if m else en[:30]


# ============================================================ 块对齐（第十八轮 Y2）
#
# 为什么要有这一层：`term_gated` / `distinct_terms` / `sense_gated` 全是**叶级**的，
# 而本项目最大的三块无闸区恰恰都在叶**内部**（三条的 why 里各自写着）：
#   ① R-rank-sense-compendium：混合叶 8 / 无法分类叶 57 整叶不查，纯 GAME 叶不上正向闸
#   ② R-shard-god：同叶单复数并存的 70 叶，负向先行断言结构上就不进闸
#   ③ R-arcturel-vs-arcturian：96% 的叶两个词都有，叶级判据抓不到叶内单处串行
#
# 判据形态取自 `4-临时脚本/2026-08-15-round16/probes/split_dives.py`：标签是机械的、
# 两侧逐字节相同，所以切出来的块数两侧应当相等；不等的**报出来**而不是静默跳过。
#
# ⚠ 与 split_dives 的两处**有意不同**，都是本轮实测逼出来的：
#
# 1. **只按块级标签切，行内标签剥成空格。** split_dives 按 `<[^>]+>` 全切，本轮实测那样太细：
#    中文「定语在前」会把词搬过 `<strong>` 边界 —— `Cora Attunement.description` 的英文块
#    「damage equal to 2 times your attunement rank」，中文的「同调阶位」搬到了前一块
#    「你获得等同于同调阶位 2 倍的」。全切的话正向闸报出 42 处，其中成片是这一类；
#    改成只按 p/li/td/h*/br… 切之后掉到 12 处。段落／列表项／表格格仍然比整叶细一个量级。
#
# 2. **富文本增强器连标签一起涂掉**（`@UUID[…]{标签}` 的花括号也涂）。
#    实测 `A Brush With Death` 的英文是**裸** `@UUID[…]`（Foundry 渲染目标名），中文补了
#    `{阿克图里安}`；不涂标签就是「EN 空 / CN 有」的假阳性。标签另有闸看着
#    （R-arcturian-split 的 `\{Arcturians\}` 域 · R-arcturian-actor-card · scan_uuid_swap）。
_BLOCK_TAG = re.compile(
    r"</?(?:p|div|li|ul|ol|tr|td|th|table|thead|tbody|tfoot|caption|h[1-6]|br|hr|"
    r"section|article|aside|header|footer|blockquote|figure|figcaption|dl|dt|dd|pre)\b[^>]*>",
    re.IGNORECASE)
_INLINE_TAG = re.compile(r"<[^>]+>")
_ENRICHER = [re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?"), re.compile(r"\[\[[^\]]*\]\]")]


def split_blocks(s):
    """按块级标签切块；每块内的行内标签与增强器涂成等长空格（保持位置，便于人对照）。"""
    for p in _ENRICHER:
        s = p.sub(lambda m: " " * len(m.group()), s)
    return [_INLINE_TAG.sub(" ", b) for b in _BLOCK_TAG.split(s)]


def _class_re(spec, flags):
    """`[{"re": …, "cls": "X"}, …]` → (合并后的正则, 组名→类名)。

    合并成一条带命名组的交替式，**顺序即优先级**（正则交替是最左优先），
    所以 `Shard Goddess` 必须写在 `Shard Gods` 前面、`阿克图里安人` 写在 `阿克图里安` 前面。
    ⚠ 不用 `m.lastgroup` 取类：条目自身可以带内层括号，内层组一旦参与匹配
    `lastgroup` 就会变成 None。改为按插入顺序找第一个非 None 的命名组。
    """
    parts, names = [], {}
    for i, item in enumerate(spec):
        g = f"c{i}"
        names[g] = item["cls"]
        parts.append(f"(?P<{g}>{item['re']})")
    return re.compile("|".join(parts), flags), names


def _classes(rx, names, text):
    out = []
    for m in rx.finditer(text):
        for g, cls in names.items():
            if m.group(g) is not None:
                out.append(cls)
                break
    return out


def _block_exempt(rule, path, i, en_sig="", cn_sig=""):
    """`except_blocks` 的每一条必须**四项全中**（路径后缀 + 块号 + 两侧类串）。

    故意写这么死：块号会随译文改动漂移，漂了就重新报出来让人看 —— 这个方向是对的。
    宁可将来红一次让人重判，也不要一条谁也不记得为什么存在的死豁免（R-dives-mine 的教训）。

    命中返回该条在 `except_blocks` 里的**下标**，没命中返回 `None`。
    ⚠ 返回下标而不是 True，是因为调用方要数**每条豁免各被用了几次** —— 见 `_unused_exempt`。
    下标 0 是假值，所以调用方必须写 `is not None`，不能写 `if j:`。
    """
    for j, e in enumerate(rule.get("except_blocks", [])):
        if (path.endswith(e["path"]) and e["block"] == i
                and e.get("en", en_sig) == en_sig and e.get("cn", cn_sig) == cn_sig):
            return j
    return None


def _unused_exempt(rule, used, field="except_blocks", label="块"):
    """**死豁免闸**：`except_blocks` 里一次都没命中的条目要当场吵出来。返回 (违规行, 死豁免条数)。

    `field` / `label` 只是为了让第十九轮新增的 `enricher_slot_gate` 复用同一套逻辑
    （它的豁免表叫 `except_slots`、单位是「槽」）—— **不许把这段判据复制第二份**，
    两份判据迟早分叉，那正是本项目反复吃亏的形态（见 `_run_same_en_split` 的注释）。

    这是本文件通则的**第四种空转形态**（前三种写在模块 docstring 里：判据写坏 / 没读库 /
    读的是库的过期快照）。这一种最隐蔽，因为**闸本身照常在跑、照常全绿**：
    豁免不命中只让 detail 里的 `n_exempt` 变小一点，**没有任何断言会因此变红**。

    实测代价（第十八轮收尾，复核单元实跑）：两条块级断言的豁免表里合计躺着 **7 条**
    再也匹配不到的条目 —— `R-rank-sense-blocks` 5 条（Shine On 块 4/14/31 与
    The Old Flame 块 264/268，都是升报后译文已经改对了）、`R-arcturel-arcturian-blocks`
    2 条（Sadri Zhalimorne 块21 与 Constructed Companion 块7，同样已修复）——
    而全套依旧 52 通过 / 0 失败，**只能靠人记**。

    这正是 `R-dives-mine` 的形态：欠账还清了、豁免却留着，于是**遮住了未来的回潮** ——
    下一轮同一个块再错回去，会被一条谁也不记得为什么存在的豁免直接吞掉。

    默认上限 **0**：豁免一旦不再命中就必须由人决定「删掉」还是「块号漂了要重判」，
    不许静静地留着。`max_unused_exempt` 可以调高，但调高就要在 `why` 里说明为什么。
    """
    dead = [(j, e) for j, e in enumerate(rule.get(field, [])) if not used[j]]
    cap = rule.get("max_unused_exempt", 0)
    if len(dead) <= cap:
        return [], len(dead)
    return ([("-", "配置", f"{rule['id']} {field}[{j}]",
              f"这条豁免一次都没命中（{e['path']} {label}{e.get('block', e.get('en', ''))}）—— **死豁免**："
              f"要么它登记的内容欠账已经修完了、该删掉，要么块号／类串漂了、该重新判一次。"
              f"留着它只会遮住同一处将来的回潮（R-dives-mine 形态）"
              + (f"；本条允许 {cap} 条未命中，实有 {len(dead)} 条" if cap else ""))
             for j, e in dead], len(dead))


def _iter_blocks(rule, ctx, leaf_re):
    """产出 (repo, pack, path, 英文块, 中文块)；块数不等的直接算 shape 异常。"""
    exc = _paths_matcher(rule.get("except_paths"))
    n_leaf = n_shape = 0
    shape_bad = []
    rows = []
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if not leaf_re.search(ev):
            continue
        if exc and exc(path):
            continue
        n_leaf += 1
        eb, cb = split_blocks(ev), split_blocks(cv)
        if len(eb) != len(cb):
            n_shape += 1
            shape_bad.append((repo, pack, path,
                              f"块级标签结构两侧不同（{len(eb)} vs {len(cb)}）——本条无法逐块判，"
                              f"该由 scan_markup_drift 先处理"))
            continue
        rows.append((repo, pack, path, eb, cb))
    return rows, n_leaf, n_shape, shape_bad


def a_block_aligned_gate(rule, ctx):
    """按块把英文与中文的**术语类别**对齐。两种 mode，各有各的适用面：

    `sequence` —— 两侧类别**序列必须逐位相等**。最强的一档，能抓「叶内单处串行」
        （城名写成族名）与「漏译一处」。代价是对语序调换敏感，所以只用在中文承载得住
        逐位对应的那些二分上。实测 `Arcturel`/`Arcturian`：1714 块里 1705 块逐位相等。

    `count_ge` —— 中文各类计数**不得少于**英文（可多不可少），另可对指定类加反向存在闸。
        用在中文语法**扛不住**逐位对应的地方。实测 `Shard God`：按单／复数逐位对齐，
        779 块里 46 块不齐，逐条看过**全部是合法中文** —— 中文不标复数，且惯于把
        `the Shard God X` 译成「碎片诸神之一的 X」、把 `three Shard Gods of Fire and four of
        Battle` 拆成「三位火焰之碎片之神和四位战斗之碎片之神」。**是判据不成立，不是译文错**，
        所以单复数不进类。中文真正承载得住的是「女神 vs 神」，那一支用反向闸一处不许错。
        「可多不可少」放过的是**代词还原**（英文 they → 中文点名），实测 13 块残差
        无一例外是中文多；抓得住的是整块漏译一处、把女神并进神、把某一类整支改名。

    反空转：`min_leaves`（闸下多少叶）与 `min_blocks`（有词块数）双护栏，
    detail 里必报「扫了多少叶 / 多少块」——满足本文件通则。
    `max_shape_mismatch` 默认 0：块级标签结构不齐的叶算失败，因为那等于**本条判不了它**，
    而判不了必须吵出来，不能静静地算通过。
    `max_unused_exempt` 默认 0：`except_blocks` 里一次都没命中的条目算失败 —— 理由与实测
    代价见 `_unused_exempt` 的 docstring（那是**闸照常全绿**却已经不设防的第四种形态）。
    """
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    leaf_re = re.compile(rule["leaf_gate"], flags)
    en_rx, en_names = _class_re(rule["en_tokens"], flags)
    cn_rx, cn_names = _class_re(rule["cn_tokens"], 0)
    mode = rule.get("mode", "sequence")

    rows, n_leaf, n_shape, bad = _iter_blocks(rule, ctx, leaf_re)
    n_block = n_ok = n_exempt = 0
    used = [0] * len(rule.get("except_blocks", []))
    for repo, pack, path, eb, cb in rows:
        for i, (e, c) in enumerate(zip(eb, cb)):
            ee = _classes(en_rx, en_names, e)
            cc = _classes(cn_rx, cn_names, c)
            if not ee and not cc:
                continue
            n_block += 1
            en_sig, cn_sig = "".join(ee), "".join(cc)
            why = None
            if mode == "sequence":
                if ee != cc:
                    why = f"块内类别序列不对齐：英文 {en_sig or '∅'} / 中文 {cn_sig or '∅'}"
            elif mode == "count_ge":
                ec, cnt = {}, {}
                for k in ee:
                    ec[k] = ec.get(k, 0) + 1
                for k in cc:
                    cnt[k] = cnt.get(k, 0) + 1
                short = [f"「{k}」类英文 {n} 处、中文只有 {cnt.get(k, 0)} 处" for k, n in ec.items()
                         if cnt.get(k, 0) < n]
                back = [f"中文出现「{k}」类而英文块内没有" for k in rule.get("backward_classes", [])
                        if cnt.get(k, 0) and not ec.get(k, 0)]
                if short or back:
                    why = "；".join(short + back) + f"（英文 {en_sig or '∅'} / 中文 {cn_sig or '∅'}）"
            else:
                why = f"未知 mode {mode!r}"
            if why is None:
                n_ok += 1
                continue
            j = _block_exempt(rule, path, i, en_sig, cn_sig)
            if j is not None:
                used[j] += 1
                n_exempt += 1
            else:
                bad.append((repo, pack, f"{path} 块{i}", why))

    # 判了几条规矩：两侧 token 表各一条一项 + mode + 每条反向类（**有词块 > 0 才算**）。
    if n_block:
        _unit(*(f"en_tok:{g}:{c}" for g, c in en_names.items()))
        _unit(*(f"cn_tok:{g}:{c}" for g, c in cn_names.items()))
        _unit(f"mode:{mode}")
        _unit(*(f"back:{c}" for c in rule.get("backward_classes", [])))

    dead_rows, n_dead = _unused_exempt(rule, used)
    bad.extend(dead_rows)
    detail = (f"{mode} 闸：闸下 {n_leaf} 叶（结构不齐 {n_shape}）· 有词块 {n_block} 块"
              f"（对齐 {n_ok} · 已登记豁免 {len(used)} 条 / 命中 {n_exempt} 块 / 死豁免 {n_dead} 条）")
    if n_shape > rule.get("max_shape_mismatch", 0):
        bad.append(("-", "配置", rule["id"],
                    f"块级标签结构不齐的叶有 {n_shape}（上限 {rule.get('max_shape_mismatch', 0)}）"))
    for field, got, why in (("min_leaves", n_leaf, "闸下叶数"), ("min_blocks", n_block, "有词块数")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{why}只数到 {got}（要求 ≥{want}）—— 这条断言在空转："
                        f"正则被 JSON 转义吃掉了？上游改了措辞？切块规则被改坏了？"))
    return bad, detail


def a_block_sense_gate(rule, ctx):
    """`sense_gated` 的块级版：把义项分类的窗口从**整叶**收到**块内**，于是块内单一义项
    的地方终于能上**正向闸**（叶级版明确写着「故意不做正向闸」，因为纯 GAME 的 230 叶里
    有 13 叶中文正当地没有定译）。

    块级带来两处判据修正，都是实测逼出来的，都是**收紧分类、不是放宽闸**：
      · `strong_game`：块小了之后 COMMON 的 `ranks of` 会咬到 `Ranks of attunement
        progression`（叶级时那一叶别处还有 GAME、落进 MIX 桶所以从没暴露）。同调／魂印／
        `Rank N` 是本系统的机制专名，优先级必须高于 COMMON 的泛化措辞。
      · `exempt`：`rank of exhaustion`（＝层）· `close ranks`（＝并肩结阵）·
        `join their ranks`（＝加入他们）是**第三个义项**，本来就不归「阶位」那条裁决管。
        块内出现即整块不判 —— 这是把它们从分类里摘出去，不是给它们放行。

    ⚠ **判据边界**：块内混合义项（MIX）与无法分类（UNKNOWN）仍然不判 —— 但注意这里的
    「不判」比叶级小得多：叶级是整叶 57 片不判，块级只是那一段落不判，同叶其它段落照判。
    """
    occ = re.compile(rule["occ"], re.IGNORECASE)
    leaf_re = re.compile(rule.get("leaf_gate", rule["occ"]), re.IGNORECASE)
    game = re.compile(rule["sense"]["game"], re.IGNORECASE)
    common = re.compile(rule["sense"]["common"], re.IGNORECASE)
    strong = re.compile(rule["sense"]["strong_game"], re.IGNORECASE)
    exempt = re.compile(rule["sense"]["exempt"], re.IGNORECASE)
    win = rule.get("window", 90)
    need = rule["cn"]

    rows, n_leaf, n_shape, bad = _iter_blocks(rule, ctx, leaf_re)
    k = {"GAME": 0, "COMMON": 0, "MIX": 0, "UNKNOWN": 0, "EXEMPT": 0}
    n_block = n_exempt = 0
    used = [0] * len(rule.get("except_blocks", []))
    for repo, pack, path, eb, cb in rows:
        for i, (e, c) in enumerate(zip(eb, cb)):
            ms = list(occ.finditer(e))
            if not ms:
                continue
            n_block += 1
            seen = set()
            for m in ms:
                w = e[max(0, m.start() - win): m.end() + win]
                if exempt.search(w):
                    seen.add("EXEMPT")
                elif strong.search(w):
                    seen.add("GAME")
                elif common.search(w):
                    seen.add("COMMON")
                elif game.search(w):
                    seen.add("GAME")
                else:
                    seen.add("UNKNOWN")
            bucket = ("EXEMPT" if "EXEMPT" in seen else "UNKNOWN" if "UNKNOWN" in seen
                      else seen.pop() if len(seen) == 1 else "MIX")
            k[bucket] += 1
            if bucket == "GAME" and need not in c and c.strip():
                why = (f"块内 `{rule['occ']}` 全部是机制义，中文这一块却没有「{need}」："
                       f"EN {e.strip()[:60]!r} / CN {c.strip()[:40]!r}")
            elif bucket == "COMMON" and need in c:
                why = (f"块内 `{rule['occ']}` 全部是普通名词义（组织层级／行列／军衔），"
                       f"中文却用了机制义定译「{need}」：EN {e.strip()[:60]!r}")
            else:
                continue
            j = _block_exempt(rule, path, i)
            if j is not None:
                used[j] += 1
                n_exempt += 1
            else:
                bad.append((repo, pack, f"{path} 块{i}", why))

    # 判了几条规矩：四张分类表 + 正反两个方向闸（**含该词的块 > 0 才算**）。
    if n_block:
        _unit(*(f"sense:{k2}" for k2 in rule["sense"]))
        _unit("dir:game→需含定译", "dir:common→不得含定译", f"cn:{need}")

    dead_rows, n_dead = _unused_exempt(rule, used)
    bad.extend(dead_rows)
    detail = (f"闸下 {n_leaf} 叶（结构不齐 {n_shape}）· 含 `{rule['occ']}` 的块 {n_block}"
              f"（机制义 {k['GAME']} / 普通名词义 {k['COMMON']} / 混合 {k['MIX']} / "
              f"无法分类 {k['UNKNOWN']} / 第三义项 {k['EXEMPT']}）· 已登记豁免 {len(used)} 条 / "
              f"命中 {n_exempt} 块 / 死豁免 {n_dead} 条")
    if n_shape > rule.get("max_shape_mismatch", 0):
        bad.append(("-", "配置", rule["id"], f"块级标签结构不齐的叶有 {n_shape}"))
    for field, got, why in (("min_leaves", n_leaf, "闸下叶数"),
                            ("min_blocks", n_block, "含该词的块数"),
                            ("min_game_blocks", k["GAME"], "机制义块数"),
                            ("min_common_blocks", k["COMMON"], "普通名词义块数")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{why}只数到 {got}（要求 ≥{want}）—— 这条断言在空转"))
    return bad, detail


# ============================================ HTML 标签 EN/CN 对等（第三十三轮 B）
#
# 为什么在「§0.1 已收官、原则上不再加判据层」之后还批准这一条
# ------------------------------------------------------------
# 0.10.2 跟版轮的独立复核发现：本轮译文给中文**多加了 2 处 `<strong>`**，
# 而当时**没有任何判据看得见这件事**。两路复核各自报了一个数，都结构上发现不了：
#   · 「标签总数 21732 → 21686」—— 那是**单侧总量**，中文多一个、英文少一个，总数照样对得上；
#   · 「标签种类 85 不变」—— 那是**集合**，多几个同种标签种类一个不多。
# 两个口径都在库级汇总，而对等是**逐叶**性质：汇总量守恒 ≠ 每片叶两侧一致。
#
# 它符合 §0.1 停止条件的例外那一支：**会咬到诚实的维护者**。
# 本轮那 2 处不是攻击、不是投机取巧，就是译者顺手加了个强调 —— 而没有任何东西吭声。
# 与 L6-② 那条「等长乱码替换」（纯攻击场景、判定不修）正好是两类。
# ⚠ **只批了这一条。** 别顺势再加别的层。
#
# 判什么、不判什么（边界写死，免得下一轮扩张）
# --------------------------------------------
# · 判**标签多重集**：把每个标签压成 `p` / `/p` 这样的 (开闭 + 名字) 记号，逐叶比两侧计数。
#   多重集相等 ⇔ 两侧每种标签各出现同样多次。位置/嵌套顺序**不判** —— 那是块对齐那两条
#   （`block_aligned_gate`）的事，本条不重复造第二个判据。
# · **属性不判**（`<span class="x">` 与 `<span>` 视为同一记号）。属性漂移归 scan_markup_drift；
#   把属性也纳进来会让「中文侧给 `<td>` 补了个 colspan」这类合法差异变成噪声。
# · 自闭合写法归一：`<br>` / `<br/>` / `<br />` 是同一个记号 `br`。
# · 两侧都不含标签的叶**不进闸**（纯文本名字/短标签占全库大半），detail 里报进闸叶数。
#
# ⚠ 与硬约束 3 的关系：`@UUID[…]` / `@Condition[…]` / `@Embed[…]` 里不含 `<`，
#   本条压根碰不到它们；本条**只读不写**，任何情况下都不改库。
#
# ⚠ **已知局限，照实写**：`<` 后面紧跟字母就当标签，所以正文里的数学比较
#   （`if damage < weight`）会被误当成 `<weight>`。本轮实测这不是活问题 ——
#   全库 69 种记号逐一看过，**没有一个不是正经 HTML 标签**（a/blockquote/br/code/dd/
#   details/div/dl/dt/em/figcaption/figure/h2-h4/hr/i/img/li/ol/p/s/section/span/strong/
#   sub/sup/summary/table/tbody/td/tfoot/th/thead/tr/ul 及其闭合）。
#   而且就算将来出现，只要两侧写法一致就仍然对等、不会误报；真误报的前提是
#   「英文写 `< weight`、中文把它译没了」—— 那本来也该有人看一眼。
_TAG_TOKEN = re.compile(r"<\s*(/?)\s*([A-Za-z][A-Za-z0-9]*)")

# —— 本条的**死下限**：住在本文件，不住规则文件（§3.7.2「强度参数与判据同层 = 没有强度」）。
#
# 2026-08-17 探针实测（`probe_tag_parity.py` 先落盘再跑，两仓 40676 对中英叶）：
#     进闸叶（两侧任一带标签） 15363 · 单侧标签总数 EN 398486 / CN 398486
#     标签记号种类（含闭合） 69 · **多重集不等的叶 0**
# 历史对照：2026-08-11 那份 21732 → 21686 的数是**只数 crucible 一个仓的英文侧**
# （本轮复算 crucible 侧 21682、ember 侧 376804），所以两个数不可直接比，别当成掉了。
#
# ⚠ 三个死下限取的是实测值的**保守下取整**，不是实测值本身：叶数/标签数会随每一轮
#   译文增删小幅浮动，把实测值原样钉死等于让这条闸每轮都红一次 —— 那会把护栏变成背景音
#   （本项目登记过的代价：护栏一旦常红，人就学会忽略它）。要抓的是「整块塌掉」，不是抖动。
# ⚠ 它**不跟规则文件联动**：规则里那三个 `min_*` 被改小也越不过这三个数（作弊路径 B）。
TAG_PARITY_HARD = {"min_leaves": 15000, "min_tags": 390000, "min_tag_kinds": 60}


def tag_multiset(s):
    """一段文本 → {标签记号: 次数}。开标签记 `p`，闭标签记 `/p`，属性一律不看。

    ⚠ 抽成模块级函数**不是**为了好看：`--selftest` 要能把它单独掏空
      （`return collections.Counter()`）来证明「掏空判定函数必须有用例变红」。
      判定逻辑埋在执行体里的话，那条反例就无从写起（L6-① 的教训）。
    """
    return collections.Counter(f"{m.group(1)}{m.group(2).lower()}"
                               for m in _TAG_TOKEN.finditer(s))


def tag_diff(en_c, cn_c):
    """两个多重集的差异 → 人话。相等返回空串（**判定就在这一行**）。"""
    if en_c == cn_c:
        return ""
    return "；".join(f"<{t}> EN {en_c.get(t, 0)} / CN {cn_c.get(t, 0)}"
                     for t in sorted(set(en_c) | set(cn_c))
                     if en_c.get(t, 0) != cn_c.get(t, 0))


def a_tag_parity(rule, ctx):
    """逐叶比 EN/CN 的 **HTML 标签多重集**，不等就报，并点名到叶路径与差在哪个标签。

    两层地板（本项目现行纪律，缺一不可 —— 见 `_two_layer_floor` 的 docstring）：
      · **现算下限**：规则文件里的 `min_leaves` / `min_tags` / `min_tag_kinds`，
        由本轮实测现算后写进规则集；`PAYLOAD_FLOORS` 把它们钉成 `ge`，只许涨不许降。
      · **历史记录值**：`TAG_PARITY_HARD`，住在**本文件**。规则文件被改松也越不过它。
        规则里缺了 `min_*` ⇒ `_two_layer_floor` 直接报「两层里少了一层」，不是放行。

    ⚠ 「判了几条规矩」数的是**这次真的见到过的标签记号种类**（＋三道地板）。
      它与语料规模无关（同一个 `<p>` 判 18 万次仍只算一条），但**与库里还剩哪些标签种类有关** ——
      所以这条闸红成「规矩数不达地板」时，最可能的解释不是判据空转，而是
      **某一种标签在库里整种消失了**（或执行体被换成了空壳）。两种都得人看一眼。
    """
    exc = _paths_matcher(rule.get("except_paths"))
    bad = []
    n_leaf = n_en = n_cn = 0
    kinds = set()
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        ec, cc = tag_multiset(ev), tag_multiset(cv)
        if not ec and not cc:
            continue                                   # 纯文本叶：本条无从判，也无须判
        if exc and exc(path):
            continue
        n_leaf += 1
        n_en += sum(ec.values())
        n_cn += sum(cc.values())
        for t in set(ec) | set(cc):
            kinds.add(t)
            _unit(f"tag:{t}")                          # 落到实数据上了才记
        why = tag_diff(ec, cc)
        if why:
            bad.append((repo, pack, path,
                        f"标签多重集两侧不等：{why} —— 译文单方面增删了标签。"
                        f"两侧的标签必须一一对应：中文多一个 `<strong>` 不会让"
                        f"「标签总数」或「标签种类数」有任何变化，只有逐叶多重集看得见"))
    _unit("min_leaves", "min_tags", "min_tag_kinds")
    H = TAG_PARITY_HARD
    _two_layer_floor(bad, "进闸叶数（两侧任一带标签）", n_leaf, rule.get("min_leaves"),
                     hard=H["min_leaves"], key="min_leaves",
                     grow_hint="库里带标签的叶变多了（新包/新页），把规则里的 min_leaves 补记上去")
    _two_layer_floor(bad, "英文侧标签总数", n_en, rule.get("min_tags"),
                     hard=H["min_tags"], key="min_tags",
                     grow_hint="正文变多了，把规则里的 min_tags 补记上去")
    _two_layer_floor(bad, "标签记号种类数", len(kinds), rule.get("min_tag_kinds"),
                     hard=H["min_tag_kinds"], key="min_tag_kinds",
                     grow_hint="上游用了新标签，把规则里的 min_tag_kinds 补记上去")
    return bad, (f"逐叶比标签多重集：进闸 {n_leaf} 叶 / 标签 EN {n_en} · CN {n_cn} / "
                 f"记号 {len(kinds)} 种（死下限 叶{H['min_leaves']}·标签{H['min_tags']}·"
                 f"种类{H['min_tag_kinds']}，住在 assert_resolutions.py）")


def _run_same_en_split(ctx, rule):
    """**现场**跑一遍 `scan_same_en_split.py`，返回 (分叉组, 扫到的英文唯一串数, 出错说明)。

    走子进程而不是 import，是因为该扫描器的分组逻辑整个写在 `main()` 里；
    复制一份到这里就等于开了第二个判据，两边迟早分叉 —— 那正是本项目反复吃亏的形态。
    """
    script = os.path.join(HERE, rule.get("scanner", "scan_same_en_split.py"))
    if not os.path.exists(script):
        return None, 0, f"找不到扫描器 {script}"
    fd, tmp = tempfile.mkstemp(suffix=".same_en.json")
    os.close(fd)
    cmd = [sys.executable, script]
    for d in ctx.repos.values():
        cmd += ["--repo", d]
    cmd += ["--show", "0", "--out", tmp]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            return None, 0, f"扫描器退出码 {proc.returncode}：{(proc.stderr or '')[-300:]}"
        groups = json.load(open(tmp, encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001 —— 跑不起来必须判失败，不能判通过
        return None, 0, f"扫描器跑不起来：{exc!r}"
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    m = re.search(r"英文唯一串（有中文的）\s*(\d+)", proc.stdout or "")
    return groups, int(m.group(1)) if m else 0, None


def a_exclusions_closed(rule, ctx):
    """`same_en_split` 的分叉组必须全部已在归档豁免表里 —— **现场重跑扫描器**。

    ⚠ 第十六轮改法，原因是这条断言被实测抓到在**空转**（模块 docstring 的形态 (c)）：
    旧版读的是磁盘上的 `4-临时脚本/2026-08-15-round15/qa2/same_en_final.json`
    ——第十五轮的产物、15 组——而第十六轮库里实跑是 **14 组 / 123 叶**。
    也就是说它**只与「上次谁跑过扫描并写了那个文件」一样新**，库里新分叉出来它不会变红，
    而这条断言存在的唯一理由恰恰是「冒出新组就要有人看」。

    现在的形态：每次都子进程跑一遍 `scan_same_en_split.py`（跑在 `--root` 指定的那棵树上），
    并把「扫到多少条英文唯一串 / 报出多少组多少叶」打进 detail —— 满足本文件的通则
    「任何断言都必须能说出我这次扫了多少」。扫描器跑不起来一律**判失败**，不判通过。

    ⚠ 第十七轮（2026-08-15，A7）**只动了豁免表的位置**，判据形态没变：
    那份 125 条的表原本在 `4-临时脚本/2026-08-13-round12/findings/EXCLUSIONS.json`，
    被 `.gitignore` 的 `4-临时脚本/**/*.json` 挡在仓库外 —— 换台机器 clone 下来它就不存在，
    本函数会走下面「找不到归档豁免表」那条分支报失败。**那是判据环境坏了，不是库坏了**，
    而报出来的样子和真缺陷一模一样。现已挪到 `5-其他内容/EXCLUSIONS.same_en_split.json`
    （路径写在规则的 `exclusions` 字段里，不在本文件里硬编码）。
    ⚠ 它与 `5-其他内容/EXCLUSIONS.json` **是两张不同的表，别合并**：本函数吃的是**裸 list**，
    靠 `` `英文串` `` 这种反引号写法在每条的 what/why 里做子串匹配；那一张是
    `{meta, exclusions:[…]}`，给人每轮读的项目级登记表。合并会当场把本判据打瘸。
    """
    exc_p = os.path.join(ROOT, rule["exclusions"])
    if not os.path.exists(exc_p):
        # 非空 bad ⇒ 本条判**失败**（不是 skipped）。有意如此：表没了就等于没设防，
        # 而「没设防」必须吵出来。detail 里点明这多半是判据环境问题而非库的问题。
        return ([("-", "-", exc_p,
                  "找不到归档豁免表，无法核对 —— 先确认它是不是又被挪回 4-临时脚本/ 被 .gitignore 挡掉了")],
                "归档豁免表缺失（判据环境问题，不是库的问题）")
    exc = json.load(open(exc_p, encoding="utf-8"))
    blob = " ".join(f"{e.get('what','')} {e.get('why','')}" for e in exc)
    _unit("archive")                               # 归档豁免表真的读进来了

    groups, n_en, err = _run_same_en_split(ctx, rule)
    if err:
        return [("-", "-", "scan_same_en_split", err + " —— 判失败而不是判通过：跑不了就等于没设防")], "扫描失败"

    bad = []
    loose = 0
    _unit("scan")                                  # 扫描器现场跑成了
    if groups:
        _unit("closure")                           # 「每一组都得在归档表里」这条真的判到了组
    _unit("min_en_strings")                        # 反空转那道地板
    for g in groups:
        en = g.get("en", "")
        key = _exc_key(en, rule)
        if f"`{key}`" in blob:                     # 豁免表的写法是 `Shield` 护盾术(23) / 盾牌(11)
            continue
        if key in blob:                            # 长正文那两条只能松匹配，记下来让人看得见
            loose += 1
            continue
        bad.append(("-", "-", en[:60],
                    "该分叉组不在已归档豁免表里 —— 要么是新缺陷，要么要补一条豁免"))

    n_leaf = sum(g.get("n_leaf", 0) for g in groups)
    min_en = rule.get("min_en_strings", 1000)
    if n_en < min_en:
        bad.append(("-", "-", rule["id"],
                    f"扫描器只报了 {n_en} 条英文唯一串（要求 ≥{min_en}）—— 它多半没读到库，"
                    f"这条断言在空转"))
    return bad, (f"现场扫描：英文唯一串 {n_en} 条 → 分叉 {len(groups)} 组 / {n_leaf} 叶；"
                 f"归档 {len(exc)} 条（其中 {loose} 组靠松匹配过闸）")


def glossary_value_matches(want, got):
    """词表值判据：产物层允许带双语尾巴（既定书写约定），所以只比**中文头部**。

    ⚠ 头部是**第一个空格前的整词**，不是前缀。这一条看着琐碎，实际是个会立刻把断言打红的坑：
    第十六轮收尾要给 `Cosmology` 登记词表值时，按 §8 的散文裁决「Cosmology＝宇宙」写 want='宇宙'，
    而产物是「宇宙观 Cosmology」、头部取到「宇宙观」—— **'宇宙' ≠ '宇宙观'，当场判失败**。
    正解是给这个**词表键**单列 want='宇宙观'（锚点页名本身就是「宇宙观 Cosmology」），
    与「正文里泛指用法的 cosmology/cosmological＝宇宙」区分开：
    一个是**键的值**，一个是**词根的译法**，两者可以不同。

    ⚠ 另一半：base 里的**多义 list**（`Ordain`＝["奥尔丹","授命"]、`Shield`＝[…]）在这里恒判不过。
    那是有意的 —— 多义词不该用「词表值必须等于某一个中文」来看守，
    该用读库英文闸（见 R-ordain-vs-ordani）。所以它们**不该出现在 entries 里**。
    """
    head = got.split(" ")[0] if isinstance(got, str) else got
    return want in (got, head)


def a_glossary_value(rule, ctx):
    """词表的 base 层与产物层**都**必须是这个值。

    只查产物是不够的：词表是构建产物，`build_glossary.py` 会用 base + harvest 重建，
    只改产物下一次构建就退回去（2026-08-13j 的裁决）。所以两层一起查。
    这一条守的是本项目最隐蔽的机制 —— **词表把错误洗成权威**：
    包里改对了，词表停在旧值，下一轮有人拿词表当依据反向回灌。
    """
    bad = []
    checked = 0
    seen, missing = [], []
    for layer, rel in rule["files"].items():
        path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        if not os.path.exists(path):
            # base 层在项目根之外（fvtt/），跑在别的树上时可能不存在，这一层跳过。
            # ⚠ 但**跳过要记账**：两层全跳过时下面的 `min_checked` 会响 ——
            #   不许「两个文件都不在 ⇒ 0 条待查 ⇒ 通过」（空转形态 (e)：
            #   「0 条问题」有两种含义，判据有义务把「无从查起」和「查过了没问题」分开）。
            missing.append(f"{layer}（{path}）")
            continue
        seen.append(layer)
        d = json.load(open(path, encoding="utf-8"))
        for key, want in rule["entries"].items():
            if key not in d:
                if rule.get("require_present", True):
                    bad.append((layer, os.path.basename(path), key, f"词表里缺这个锚点键（应为「{want}」）"))
                continue
            checked += 1
            _unit(f"entry:{layer}:{key}")          # 每层每个锚点键各算一条规矩
            got = d[key]
            if not glossary_value_matches(want, got):
                bad.append((layer, os.path.basename(path), key, f"是「{got}」，应为「{want}」"))
    floor = rule.get("min_checked", 1)
    if checked < floor:
        bad.append(("-", "配置", "files",
                    f"两层合计只核到 {checked} 条（要求 ≥{floor}）——"
                    f"这些层的文件根本不在：{missing or '（都在）'}。"
                    f"这是**无从查起**，不是「查过了没问题」（空转形态 (e)）"))
    detail = f"两层合计核对 {checked} 条（在册层 {seen or '无'}"
    return bad, detail + (f"；跳过 {missing}" if missing else "") + "）"


def a_version_matrix(rule, ctx):
    """PROJECT.md 抬头与版本矩阵里的版本号，必须与两仓 `module.json` 一致。

    为什么值得一条断言：这一行**历史上漂过两次**（停在 0.9.6/1.1.7 与 0.9.7/1.1.10 各一次）。
    每次都是「发了版、追加了本轮小节、但没回头改抬头」——纯粹靠人记得，就一定会漏。
    新会话第一件事就是读抬头判断现状，读到过期版本会直接判断错「现在到哪一步了」。
    """
    import re as _re
    doc_p = os.path.join(ROOT, rule.get("doc", "PROJECT.md"))
    if not os.path.exists(doc_p):
        return [("-", "-", doc_p, "找不到 PROJECT.md")], "跳过"
    doc = open(doc_p, encoding="utf-8").read()
    bad = []
    detail = []
    n_repo = 0
    for name, repo in ctx.repos.items():
        mj = os.path.join(repo, "module.json")
        if not os.path.exists(mj):
            # ⚠ 不许静默跳过：`module.json` 不在 = 这个仓的版本号**没核成**，不是通过。
            #   （被 `--repo` 限定掉的仓压根不在 `ctx.repos` 里，走不到这里。）
            bad.append((name, "module.json", mj,
                        "找不到 module.json —— 这个仓的版本**没核成**，不是通过（空转形态 (e)）"))
            continue
        manifest = json.load(open(mj, encoding="utf-8"))
        ver = manifest.get("version")
        # ⚠ 原来写的是 `manifest.get("id", name)` —— 取不到就拿**仓的键名**当包名，
        #   那是形态 (g) 的写法（输入不缺、被静默换成了另一份）：包名一旦回落成 `ember`，
        #   下面所有正则就去 PROJECT.md 里找一个根本不存在的名字，报出来的会是
        #   「矩阵里没有 ember 这一行」这种**指向错误方向**的结论。缺就当场说缺。
        if not manifest.get("id"):
            bad.append((name, "module.json", mj,
                        "module.json 里没有 id —— 包名无从取起，这个仓没核成（形态 (g)）"))
            continue
        if not ver:
            bad.append((name, "module.json", mj, "module.json 里没有 version —— 这个仓没核成"))
            continue
        n_repo += 1
        # 判了几条规矩：每个仓三条（抬头 / 声称「当前」的正文行 / 版本矩阵那一行）。
        _unit(f"ver:{name}:head", f"ver:{name}:markers", f"ver:{name}:matrix")
        pkg = manifest["id"]
        prefix = rule["tag_prefix"].get(name, "")
        want = f"{prefix}{ver}"
        detail.append(f"{name}={want}")
        # 抬头段（前 40 行）里必须出现当前版本
        head = "\n".join(doc.splitlines()[:40])
        if want not in head:
            bad.append((name, "PROJECT.md", "抬头段",
                        f"抬头没有写当前版本 {want}（module.json 是 {ver}）—— 抬头又漂了"))

        # ⚑ 只查抬头是不够的：正文里还有「发版状态：…」那一行，它停在 0.9.4 / v1.1.4
        # 漂了四个版本仍然全绿 —— 断言自己的覆盖面就是个盲区。
        #
        # ⚠ 但**不能全文乱扫**：§1 有大量**有意保留**的历史引用（「上一版抬头：… 0.9.5 / v1.1.5」）、
        # §6 年表更是逐版记录。第一版按「§6 之前都算正文」扫，立刻在那些历史行上报了 8 处假阳性。
        # 正确的判据是**只查声称「现在」的那些行** —— 由 current_markers 明确列出。
        markers = rule.get("current_markers", ["当前已发布", "发版状态", "当前版本"])
        for i, line in enumerate(doc.splitlines(), 1):
            if not any(mk in line for mk in markers):
                continue
            for m in _re.finditer(rf"{_re.escape(pkg)}[` ]+v?(\d+\.\d+\.\d+)", line):
                if m.group(1) != ver:
                    bad.append((name, "PROJECT.md", f"第 {i} 行",
                                f"这一行声称的是**当前状态**，却写着 {pkg} {m.group(1)}，"
                                f"而 module.json 是 {ver}"))
        # 版本矩阵那一行
        row = _re.search(rf"^\|\s*{_re.escape(pkg)}\s*\|.*$", doc, _re.MULTILINE)
        if not row:
            bad.append((name, "PROJECT.md", "版本矩阵", f"矩阵里没有 {pkg} 这一行"))
        elif want not in row.group(0):
            bad.append((name, "PROJECT.md", "版本矩阵",
                        f"矩阵写的不是 {want}：{row.group(0)[:90]}"))
    floor = rule.get("min_repos", 1)
    if n_repo < floor:
        bad.append(("-", "配置", "module.json",
                    f"只核成 {n_repo} 个仓（要求 ≥{floor}）—— 这条断言在空转"))
    return bad, " | ".join(detail) + f"（核成 {n_repo} 个仓）"


def a_leaf_literal(rule, ctx):
    """指定叶必须／不得包含某个字面串。

    `pack` 与 `path` 都可以写成**列表** —— 本库大量内容是孪生两包各一份
    （`ember.adventure.json` / `ember.crucible-adventure.json`），只钉一包等于放另一包不管。
    `min_leaves` 是这一类的反空转护栏：路径写错、上游改名、只钉到孪生的一半，
    都会让命中数掉下来，而不是静静地按「0 叶待查 = 通过」放行。
    """
    packs = rule["pack"] if isinstance(rule["pack"], list) else [rule["pack"]]
    paths = rule["path"] if isinstance(rule["path"], list) else [rule["path"]]
    bad = []
    checked = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if pack not in packs or path.removeprefix("entries.") not in paths:
            continue
        checked += 1
        # 判了几条规矩：每条必含串 / 每条禁含串各一条（**命中叶之后**才记）。
        _unit(*(f"must:{m}" for m in rule.get("must_contain", [])))
        _unit(*(f"never:{m}" for m in rule.get("must_not_contain", [])))
        for must in rule.get("must_contain", []):
            if must not in cv:
                bad.append((repo, pack, path, f"必须包含「{must}」但没有：{cv[:40]!r}"))
        for never in rule.get("must_not_contain", []):
            if never in cv:
                bad.append((repo, pack, path, f"不得包含「{never}」但出现了：{cv[:40]!r}"))
    want = rule.get("min_leaves", 1)
    if checked < want:
        bad.append(("-", packs[0], paths[0],
                    f"只命中 {checked} 叶（要求 ≥{want}）—— 这几叶找不到了（上游改名？路径写错？"
                    f"孪生包只钉了一半？），断言失效，需要人看"))
    return bad, f"命中 {checked} 叶（要求 ≥{want}）"


# ====================================================== 增强器槽位（第十九轮 Y6）
#
# 为什么要有这一层：`split_blocks`（见上面它的注释）把 `@X[…]{标签}` **连花括号一起涂空**，
# 于是标签里的译名整条不进任何块级闸。第十八轮把这个洞量化过，第十九轮又逐处复核了一遍：
#
#   全库中文「阿克图瑞尔|阿克图里安」共 **2218 处，其中 278 处（12.5%）落在增强器内**。
#   把那 278 处逐个改错、每处重跑全部读库断言 —— **只有 89 处**会让某条断言变红，
#   且全是**叶级间接覆盖**（只有该叶里这一处是唯一带该词的地方时才响）；**其余 189 处无闸**。
#
# ⚠ 第十八轮那句「标签另有 R-arcturian-split 的 `{Arcturians}` 域 / R-arcturian-actor-card /
# scan_uuid_swap 看着」**逐条落实全不成立**，已在第十八轮末尾按实测改写：
# `R-arcturian-actor-card` 只钉 `actors.Arcturian` 的 name/tokenName 四叶，那四叶里根本没有增强器；
# `scan_uuid_swap` 根本不在断言套里（是独立扫描器），它自己的 docstring 也写明这类不归它管。
# **教训：`why` 里写「另有 X 闸看着」是可证伪断言，写之前必须逐处变异回测。**
#
# 本闸与 `scan_label_vs_name` 的分工（**不许做成它的重复**）
# ------------------------------------------------------
# `scan_label_vs_name` 比的是「`@UUID{标签}` 的中文 ↔ **目标文档的中文 name**」，
# 且只报「英文标签本来就等于目标英文 name」的那些 —— 它管的是**标签与目标是否同名**。
# 本闸管的是另一件事：标签里的**术语**（地名／族名／专名）与既定译名是否一致，
# 而标签本身**可以**与目标 name 不同（作者有意换称呼时 scan_label_vs_name 直接不看）。
# 举例：`@UUID[…]{Arcturel Dives}` 的中文写成「阿克图里安矿渊」——
# 标签与目标 name 的关系没变（都不是目标名），scan_label_vs_name 一声不吭，本闸报红。
#
# 做成断言（新 kind）而不是独立扫描器的理由
# ----------------------------------------
# 判据要用的两侧叶文本 `ctx.pairs` 已经在内存里；独立扫描器要**重读一遍 4 万叶**
# 再走一次子进程。`a_exclusions_closed` 走子进程是因为分组逻辑本来就写在
# `scan_same_en_split.main()` 里、复制一份就等于开第二个判据 —— 本条没有这个前提。
#
# ⚠ 配对不按「出现序号」，按 **(动词, 目标)** 分组后组内取序 —— 这是实测改的
# ------------------------------------------------------------------------
# 第一版按整叶出现序号逐个配对（Y6 任务书里写的做法），实测 **30650 对里有 1388 对
# （4.5%）配歪**：中文「定语在前」经常把整个增强器搬到另一位置
# （`Lake Jinro Lunar Shrine` / `Mythspire Observatory` 成片如此）。配歪的后果不是漏报而是
# **假阳性**：`EN=Arcturel / CN=杰夫赫尔家族`、`EN=Arcturian / CN=奥尔丹`、
# `EN=Arcturian Liquor / CN=烧瓶` 这些「一眼看去像串行」的条目，全部是配对错位造出来的幻影。
# 改成按 (动词, 目标) 分组后，全库 **30650 个增强器 0 个配不上**，那 4 类幻影同时消失。
# 目标相同的多个增强器（同一目标在一叶里出现多次）在组内仍按出现序取，位置精度不丢。
_AT_ENR = re.compile(r"@([A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?")
# ⚠ **必须同时吃带引号与不带引号的值。** 上一版只认双引号 —— 库里
# `@Embed[Actor.LUptsqBgGJVWcg9v label=Squish]`（**无引号**，只在 EN 侧、孪生 2 叶）
# 因此永远不进纯文本，而中文侧写的是 `label="挤压"`（带引号）会进，
# 造成 EN 44 : CN 46 的不对称假象。
# ⚠ 顺带订正「全库参数名只有三种」那句 —— **不全**：还有无引号的
# `count` 166 · `rollable` 20 · `cite` 4 · `label` 2（第二十一轮全库复算）。
# 前三个是机器参数、被白名单排掉是对的，但 `label=Squish` 是**可见文本**。
_ENR_PARAM = re.compile(r'\b([A-Za-z][\w-]*)\s*=\s*(?:"([^"]*)"|([^\s\]"]+))')
# 增强器里**玩家能看见的文本**：花括号标签，以及 `@Embed[… label="…" readaloud="…"]`
# 的这两个参数 —— 实测 `readaloud=` 里塞的是整段朗读正文（**48 处**，
# EN 侧 16966 字符 / 最长 **951** 字，CN 侧 5392 字符 / 最长 311 字），
# ⚠ 上一版写「46 处，最长 300+ 字」两处都不准：漏的 2 处在**小写** `@embed[` 里
# （本文件的 `_AT_ENR` 是 `@([A-Za-z]+)\[`，吃小写，所以那 2 处一样在定义域内）；
# 「最长 300+」只对 CN 侧成立，EN 侧最长是 951。
# 那整段同样被 split_blocks 涂空，是本洞里最大的单块。其余参数（`count=` `classes=`）
# 是机械值，不进槽位。
_ENR_TEXT_PARAMS = ("label", "readaloud")


def _enr_key(m):
    """(动词小写, 目标) —— 目标取方括号内第一个空格前的串（`@Embed` 后面还跟着参数）。"""
    return m.group(1).lower(), m.group(2).split(" ", 1)[0]


def _enr_slots(m, wanted):
    out = {}
    if m.group(3) is not None:
        out["label"] = m.group(3)
    for pm in _ENR_PARAM.finditer(m.group(2)):
        n = pm.group(1).lower()
        if n in wanted:
            # ⚠ `_ENR_PARAM` 有两个候选组：group(2)=带引号的值、group(3)=不带引号的值，
            #   **同一次匹配里必有一个是 None**。第二十一轮加无引号支持时只改了
            #   `scan_content_coverage.py` 的取值处、漏了这里，`R-arcturel-arcturian-labels`
            #   当场 TypeError（NoneType 喂给正则）。两处取值必须一起改。
            out["param:" + n] = pm.group(2) if pm.group(2) is not None else pm.group(3)
    return out


def _enr_pairs(ev, cv, wanted):
    """按 (动词,目标) 分组、组内取序配对。返回 (配对列表, 配不上的个数)。"""
    eg, cg = {}, {}
    for m in _AT_ENR.finditer(ev):
        eg.setdefault(_enr_key(m), []).append(m)
    for m in _AT_ENR.finditer(cv):
        cg.setdefault(_enr_key(m), []).append(m)
    pairs, unpaired = [], 0
    for k in set(eg) | set(cg):
        a, b = eg.get(k, []), cg.get(k, [])
        n = min(len(a), len(b))
        for x, y in zip(a[:n], b[:n]):
            pairs.append((k[1], _enr_slots(x, wanted), _enr_slots(y, wanted)))
        unpaired += abs(len(a) - len(b))
    return pairs, unpaired


def a_enricher_slot_gate(rule, ctx):
    """增强器**槽位级**的术语闸：逐个「英文标签 ↔ 中文标签」比既定译名。

    判据（正向为主，反向可选）
    --------------------------
    * `en_tokens` / `cn_tokens` 与 `block_aligned_gate` 同一套有序交替式（`_class_re`），
      **顺序即优先级**，所以派生词要写在词根前面。
    * 正向：英文槽里出现某一类 → 中文槽必须出现同一类。**这一条就能抓住全部串行**
      （把 `Arcturel` 的中文写成族名，正向闸立刻少一个 E 类）。
    * 反向 `forbid_absent`：中文槽出现了英文槽里没有的类 → 违规。抓的是「凭空多出一个专名」。
      默认**关**：中文标签合法地补上下文的情形不少（`{Trinkets}`→「阿克图里安小饰品」），
      开之前必须先在当前库上实跑一遍看假阳性。

    覆盖不到的地方（照本项目规矩写死，不许含糊）
    -------------------------------------------
    * **中文有标签而英文那一侧是裸增强器**的槽（实测 582 个）没有英文槽可比
      （Foundry 对裸 `@UUID` 渲染目标文档名，中文侧补个标签是**正当做法**）。
      `cn_only_leaf_fallback` 打开后退回**整叶英文**，且**只做反向闸**：
      中文槽里出现的类，整叶英文里必须出现过；反过来**不要求**中文槽含整叶英文的类。
      ⚠ 这个方向是实测逼出来的：第一版写成正向（整叶英文只有一类 → 中文槽必须含该类），
      当场造出 **70 条假阳性** —— 一叶英文里提到 Arcturian，不代表这叶里每个
      中文标签（`{月华花}` `{拉斯克}` `{迷惘}`）都得带族名。反向则站得住：
      中文标签凭空冒出一个整叶英文里根本没有的专名，那要么是串行要么是加戏。
      ⚠ 反向闸的合法英文依据有**两个来源**，缺一个就还会假阳性（也是实测出来的）：
      ① 本叶英文；② **同一目标在全库别处的英文标签**。裸 `@UUID` 由 Foundry 渲染
      目标文档名，中文补个标签正是在写那个名字 —— `@UUID[RollTable.BUurPyycyDIuox5L]`
      在本叶英文里一个 Arctur 字样都没有，而全库别处 28 次写着 `{Arcturian Trinkets}`，
      中文写「阿克图里安小饰品」当然是对的。只认来源 ① 会把这 4 槽误报。
    * 方括号**内部**不判：那是机器参数，既定约定要求照抄英文（`[[/culture Ordani]]`）。
    * **只认 `@Verb[…]` 形态，不认 `[[/verb …]]{标签}`。** 后者实测中文侧 20318 个、
      带花括号标签的 349 个（含中文 344 个），拿七条断言的全部已裁术语去扫这 349 个标签
      **一处都不命中**（伤害类型／灾害名／法术名居多）。所以这不是「漏了」而是**量过、当前为空**；
      哪天已裁术语进了那一格，这里要跟着扩。
    * 增强器配不上对的（实测 0 个）**报出来**，不静默跳过。

    反空转：`min_leaves` / `min_slots`（英文侧有类命中的槽数）双护栏，
    detail 必报「扫了多少叶 / 配了多少对 / 判了多少槽」。
    `max_unused_exempt` 默认 0，与两个块级闸共用 `_unused_exempt`。
    """
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    en_rx, en_names = _class_re(rule["en_tokens"], flags)
    cn_rx, cn_names = _class_re(rule["cn_tokens"], 0)
    wanted = tuple(rule.get("params", _ENR_TEXT_PARAMS))
    slot_filter = set(rule["slots"]) if rule.get("slots") else None
    exc = _paths_matcher(rule.get("except_paths"))
    fallback = rule.get("cn_only_leaf_fallback", False)

    bad = []
    n_leaf = n_pair = n_unpaired = n_slot = n_gated = n_ok = 0
    n_cn_only = n_fb_gated = n_exempt = 0
    used = [0] * len(rule.get("except_slots", []))

    # 预扫：目标 -> 全库英文标签里出现过的类。只有 `cn_only_leaf_fallback` 用得上，
    # 所以不开就不扫（省一遍 4 万叶）。
    tgt_cls = {}
    if fallback:
        for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
            if "@" not in ev:
                continue
            for m in _AT_ENR.finditer(ev):
                for txt in _enr_slots(m, wanted).values():
                    c = _classes(en_rx, en_names, txt)
                    if c:
                        tgt_cls.setdefault(_enr_key(m)[1], set()).update(c)

    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if "@" not in ev and "@" not in cv:
            continue
        if exc and exc(path):
            continue
        n_leaf += 1
        pairs, unpaired = _enr_pairs(ev, cv, wanted)
        n_pair += len(pairs)
        if unpaired:
            n_unpaired += unpaired
            bad.append((repo, pack, path,
                        f"有 {unpaired} 个增强器两侧按 (动词,目标) 配不上对 —— 本条判不了它们，"
                        f"该先由 scan_markup_drift / scan_markup_targets 处理"))
        leaf_en_cls = None
        for tgt, es, cs in pairs:
            for slot in set(es) | set(cs):
                if slot_filter and slot not in slot_filter:
                    continue
                n_slot += 1
                et, ct = es.get(slot), cs.get(slot)
                if ct is None:
                    continue                       # 英文有标签、中文裸 —— 中文侧没字可判
                cc = _classes(cn_rx, cn_names, ct)
                if et is None:
                    n_cn_only += 1
                    if not fallback or not cc:
                        continue
                    if leaf_en_cls is None:
                        leaf_en_cls = set(_classes(en_rx, en_names, ev))
                    allow = leaf_en_cls | tgt_cls.get(tgt, set())
                    ee, src = sorted(allow), "整叶英文+同目标别处英文标签"
                    n_fb_gated += 1
                    miss, extra = [], [c for c in dict.fromkeys(cc) if c not in allow]
                else:
                    ee, src = _classes(en_rx, en_names, et), "英文槽"
                    # ⚠ 英文槽一个类都没命中时**不能直接跳过** —— 反向闸要抓的正是
                    #   「英文槽里没有、中文槽里冒出来」这一形态（`{the Tradeway}`→
                    #   「阿克图里安贸易道」）。第一版写成 `if not ee: continue`，
                    #   于是 forbid_absent 在最该响的那一格永远响不了，自检当场抓出来。
                    if not ee and not (rule.get("forbid_absent") and cc):
                        continue
                    n_gated += 1
                    miss = [c for c in dict.fromkeys(ee) if c not in cc]
                    extra = ([c for c in dict.fromkeys(cc) if c not in ee]
                             if rule.get("forbid_absent") else [])
                if not miss and not extra:
                    n_ok += 1
                    continue
                why = (f"{src}是 {'/'.join(dict.fromkeys(ee))} 类，中文槽是 "
                       f"{'/'.join(dict.fromkeys(cc)) or '∅'} 类"
                       + (f"（缺 {'/'.join(miss)}）" if miss else "")
                       + (f"（多出 {'/'.join(extra)}）" if extra else "")
                       + f"：EN {(et or '（英文侧裸增强器）')[:60]!r} / CN {ct[:40]!r}")
                j = None
                for k2, e2 in enumerate(rule.get("except_slots", [])):
                    if (path.endswith(e2["path"]) and e2.get("slot", slot) == slot
                            and e2.get("en", et) == et and e2.get("cn", ct) == ct):
                        j = k2
                        break
                if j is not None:
                    used[j] += 1
                    n_exempt += 1
                else:
                    bad.append((repo, pack, f"{path} → {tgt[:34]} [{slot}]", why))

    # 判了几条规矩：两侧 token 表逐项 + 打开的那两个可选方向（**真判过槽才算**）。
    if n_gated + n_fb_gated:
        _unit(*(f"en_tok:{g}:{c}" for g, c in en_names.items()))
        _unit(*(f"cn_tok:{g}:{c}" for g, c in cn_names.items()))
        if rule.get("forbid_absent"):
            _unit("dir:forbid_absent")
        if fallback:
            _unit("dir:cn_only_leaf_fallback")

    dead_rows, n_dead = _unused_exempt(rule, used, "except_slots", "槽 ")
    bad.extend(dead_rows)
    detail = (f"闸下 {n_leaf} 叶 · 配对增强器 {n_pair} 个（配不上 {n_unpaired}）· "
              f"可见文本槽 {n_slot}（英文槽有类命中 {n_gated}"
              + (f" + 整叶回退 {n_fb_gated}" if fallback else "")
              + f" · 中文独有槽 {n_cn_only} · 判过 {n_gated + n_fb_gated} 一致 {n_ok}）· "
              f"已登记豁免 {len(used)} 条 / 命中 {n_exempt} 槽 / 死豁免 {n_dead} 条")
    for field, got, txt in (("min_leaves", n_leaf, "闸下叶数"),
                            ("min_slots", n_slot, "可见文本槽数"),
                            ("min_gated", n_gated + n_fb_gated, "真正判过的槽数")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{txt}只数到 {got}（要求 ≥{want}）—— 这条断言在空转："
                        f"正则被 JSON 转义吃掉了？上游改了标签措辞？增强器切法被改坏了？"))
    if n_unpaired > rule.get("max_unpaired", 0):
        bad.append(("-", "配置", rule["id"],
                    f"配不上对的增强器有 {n_unpaired}（上限 {rule.get('max_unpaired', 0)}）"))
    return bad, detail


# --------------------------------------------- 增强器可见正文的覆盖闸（第二十二轮新增）
#
# 为什么要单开一个类型（**先读这段再改判据**）
# --------------------------------------------
# 第二十一轮把 `@Embed[… readaloud="…"]` 的整段朗读正文纳入了 `scan_content_coverage`
# 的正文（EN 48 段 / 16966 字符、CN 48 段 / 5392 字符，落在 30 叶）。**纳入是真的，
# 但纳入之后没有任何判据能判它**，第二十二轮复核逐条实测：
#   · `scan_content_coverage` 的默认判据是**数字多重集**，而这 48 段英文里
#     **一个阿拉伯数字都没有（0/48，本文件现读复算同值）** ⇒ 产出必然 0 命中；
#   · 复核把全库最长的一段中文朗读正文（`ember.adventure ::
#     The Winding Trail / Dusktide Destruction`，311 字）**整段删空**，默认口径**仍报 0 条**；
#   · `--with-terms` 不成立：把该叶单独喂进去，**干净时**就报缺定译 39 条、删空后 43 条 ——
#     两侧都远远非零，**没有判别力**（那是全库开专名闸的固有噪声，模块 docstring 早写过）。
# 也就是说这 16966 字符处在「已经进了统计口径、却没有任何闸」的状态 —— 比全盲更坏，
# 因为口径行会让人以为它被查过了。
#
# 为什么做成断言（本文件），而不是给 `scan_content_coverage` 加作用域开关
# ---------------------------------------------------------------------
# 两条路都试算过，选断言的理由有三条，逐条可证伪：
# 1. **判别力来自「限定在本段范围内」，而不是来自开关。** 专名闸的噪声是**叶级**的：
#    一叶散文里随便就能命中 39 条锚点，中文当然不会条条照搬。把范围收到**这一段
#    朗读正文**之后，本次现读实测：48 段命中锚点 **36 段 / 102 条，缺定译 0 条** ——
#    信噪比从「两侧都 39+」变成「干净侧 0」。这个收缩靠的是**槽位切分**，
#    而槽位切分（按 (动词,目标) 配对、`_enr_slots` 取参数值）已经写在本文件里了。
#    在 `scan_content_coverage` 那边重做一份等于开第二个判据，两份迟早分叉
#    （`_run_same_en_split` 的注释里记着同一个教训）。
# 2. **`scan_content_coverage` 是启发式扫描器，不是闸。** 它自己的 docstring 第一句就写着
#    「优先用 scan_en_drift」，日常没人跑它、跑了也没人看 0 条以外的输出；而本项目的
#    「不许跑红」纪律挂在 `assert_resolutions.py` 上。要让这 30 叶**真的守住**，
#    判据必须在这套里。
# 3. 断言这边已经有反空转的全套护栏（`min_*` / `_unused_exempt` / `--selftest`），
#    重造一遍只会多一处可空转的地方。
# ⇒ 但 `scan_content_coverage` 的 docstring 里那句「想真的判这部分，请加 `--with-terms`」
#    是**已经被证伪的建议**，本轮同时把它改成指向本条断言（那是本轮唯一改动它的地方）。
#
# 判据：四条**互相独立**的信号，每条各报各的
# ------------------------------------------
# （中英字符比的全库中位数是 0.31，这 48 段是 0.318 —— 两者一致，说明比值这个信号在
#  朗读正文上与在普通正文上同分布，可以拿全库经验定阈值。）
# A. **缺失／无汉字**：英文槽有正文而中文槽整个没有、或有而一个汉字都没有。
#    这一条是绝对的，不看阈值。
# B. **段级中英字符比**低于 `min_ratio`。本次现读实测 48 段的比值区间是
#    **[0.263, 0.406]，中位 0.313**（分布很紧）。取 0.20 ⇒ 距最小值仍有 31% 富余。
# C. **本段范围内的定译专名**：英文段命中词表锚点的，中文段必须出现对应定译。
#    实测 36 段命中 / 102 条 / 缺 0 条。**范围限定是这条能用的全部理由**，
#    不要把它放大回叶级（那边噪声 39 条起步）。
# D. **句数**：中文句号级标点数不得低于 `floor(英文句数 × min_sentence_frac)`。
#    实测 48 段的「中文句数 ÷ 英文句数」取值只有 {1.0, 1.17, 1.25, 1.33, 2.0} ——
#    **一段都没有低于 1.0**。取 0.75 ⇒ 每一档都留着余量（英文 4 句只要求中文 ≥3 句），
#    留余量是为了不把「两句英文合成一句中文」这种合法译法判红。
# E/F. **数字**与**段内标记**两支照样实现了，但本次现读：含阿拉伯数字的英文段 **0 段**、
#    含 `<tag>`／`@X[…]` 的英文段 **0 段** ⇒ 这两支现在是**「无从查起」而不是「查过了没问题」**，
#    detail 行会把这个 0 明说出来。写在这里是因为上游随时可能往朗读框里塞一句
#    「DC 18」，那一天它们自动开始有信号，而不必再有人想起来补。
#
# 双向回测（第二十二轮实测，探针落盘在 `4-临时脚本/2026-08-16-round22/`）
# ---------------------------------------------------------------------
# · **特异度**：当前库 48 段，四条信号**合计 0 条违规**（48 段全部非空、比值最低 0.263、
#   36 段锚点 102 条定译一条不缺、句数一段都不低于英文）。
# · **灵敏度（全删）**：逐段把中文朗读正文删空 → **48/48 报**。
# · **灵敏度（删一半）**：逐段把中文朗读正文按字符截掉后一半 → **46/48 报**。
#   跑不掉的那 2 个槽是**同一段孪生两包各一份**：`Gamemaster's Guide / Main Quest Overview`
#   的 EN 219 字 / 1 句、CN 89 字（全库中文最丰的一段，比值 0.406），砍一半后比值 0.201 ——
#   **刚好在 0.20 上面 0.001**，且它英文只有 1 句所以句数闸也够不着。同一段砍 51% 就报。
#   把 `min_ratio` 提到 0.22 能把这 2 个也收进来（实测 48/48），代价是距最小值只剩 20% 富余，
#   **本轮选择留富余**：这条闸守的是「整段没跟上」，不是「少译一句」。这是判据边界，写在这里。
_CN_SENT = re.compile(r"[。！？]")
_EN_SENT = re.compile(r"[.!?](?:\s|$)")
_CJK_RE = re.compile(r"[一-鿿]")
_SEG_NUM = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")
_SEG_MARKUP = re.compile(r"@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]|<[^>]+>")


def _load_anchors(rule):
    """锚点表 = 词表里「多字英文键 → 纯中文多字值」的那些条目。返回 (表 或 None, 说明)。

    与 `scan_content_coverage.py` 取锚点的规则同源同值（`len(k) >= 5` 且中文值 ≥2 字、
    值里不含拉丁字母），本次现读得 **7601 条**。单字中文与含拉丁的值误报率太高，不要放宽。

    ⚠ **表拿不到时必须返回 None 让调用方判失败，不许静静地跑成空表。**
    那正是本文件 docstring 里的空转形态 (e)：`scan_renamed_terms` 的 `--glossary` 默认空串，
    不显式传就 `glossary={}`、每个候选静默 continue、最后打印一份漂亮的「0 条问题」，
    而主控**拿那个 0 当过一次证据**。
    `anchors` 内联字段**只给 `--selftest` 用**（自检不该依赖磁盘上的词表）。
    """
    inline = rule.get("anchors")
    if inline is not None:
        return dict(inline), "内联锚点（自检用）"
    rel = rule.get("glossary")
    if not rel:
        return None, "规则里没有 glossary 字段 —— 定译专名这一支无从查起"
    path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None, f"词表文件不存在：{path}"
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001
        return None, f"词表读不动：{exc!r}"
    min_key = rule.get("min_anchor_key_len", 5)
    min_cn = rule.get("min_anchor_cn_len", 2)
    out = {}
    for k, v in raw.items():
        zh = v.split(" ")[0].strip() if isinstance(v, str) else ""
        if (len(k) >= min_key and len(zh) >= min_cn and _CJK_RE.search(zh)
                and not re.search("[A-Za-z]", zh)):
            out[k] = zh
    return out, f"{os.path.basename(path)} 取锚点 {len(out)} 条"


def _seg_anchor_hits(anchors, text, cache):
    """本段英文里命中的锚点 [(英文键, 中文定译), …]。按段缓存 —— 孪生两包是同一段文本。"""
    got = cache.get(text)
    if got is None:
        got = [(k, v) for k, v in anchors.items()
               if re.search(r"\b" + re.escape(k) + r"\b", text)]
        cache[text] = got
    return got


def a_enricher_text_coverage(rule, ctx):
    """增强器**可见正文**（`readaloud=` 一类整段朗读文本）的覆盖闸。

    与 `enricher_slot_gate` 的分工（**不许做成它的重复**）
    ----------------------------------------------------
    `enricher_slot_gate` 判的是「这个槽里的**术语**有没有串行」——它只在英文槽命中某个
    已裁术语类时才有话说，一段没有任何已裁术语的朗读正文它**完全沉默**（48 段里
    能被那 7 条标签断言碰到的只是零星几段）。
    本条判的是「这段正文的**中文有没有跟上英文**」，与具体术语无关。两者正交。

    判据 A–F、阈值取值与双向回测结论，全部写在上面那段块注释里，**改阈值前先读它**。

    反空转
    ------
    · `min_leaves` / `min_slots` / `min_gated`：闸下叶数、在册槽数、真正判过的段数；
    · `min_anchor_terms` / `min_anchor_hits` / `min_anchor_slots`：锚点表条数、
      本次命中的锚点条数、命中锚点的段数 —— **专门防「词表没读到／锚点规则被改坏」
      导致 C 支静默失效**（形态 (e)）。锚点表拿不到直接判失败，不跑成空表。
    · detail 必报「判了多少段 / 多少字 / 比值区间 / 含数字段 / 含标记段 / 豁免」，
      其中**含数字段与含标记段现在是 0**，这个 0 的含义是「无从查起」不是「查过了」。
    · `max_unused_exempt` 默认 0，与另外三个类型共用 `_unused_exempt`。
    """
    slots_want = rule.get("slots") or ["param:readaloud"]
    params = tuple(s.split(":", 1)[1] for s in slots_want if s.startswith("param:"))
    if not params:
        params = _ENR_TEXT_PARAMS
    min_en = rule.get("min_en_chars", 60)
    min_ratio = rule.get("min_ratio", 0.20)
    sent_frac = rule.get("min_sentence_frac", 0.75)
    exc = _paths_matcher(rule.get("except_paths"))
    anchors, anchor_note = _load_anchors(rule)

    bad = []
    n_leaf = n_pair = n_unpaired = n_slot = n_short = n_cn_only = n_gated = n_ok = 0
    n_anchor_slot = n_anchor_hit = n_num_slot = n_mark_slot = n_exempt = 0
    en_chars = cn_chars = 0
    ratios = []
    used = [0] * len(rule.get("except_slots", []))
    cache = {}

    if anchors is None or not anchors:
        bad.append(("-", "配置", rule["id"],
                    f"定译专名这一支**无从查起**：{anchor_note}。"
                    f"闸的其余三支还能跑，但这条断言不许在锚点表缺失时装作全绿"
                    f"（空转形态 (e)：输入缺失而判据照跑）"))
        anchors = {}

    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if "@" not in ev and "@" not in cv:
            continue
        if exc and exc(path):
            continue
        n_leaf += 1
        pairs, unpaired = _enr_pairs(ev, cv, params)
        n_pair += len(pairs)
        if unpaired:
            n_unpaired += unpaired
            bad.append((repo, pack, path,
                        f"有 {unpaired} 个增强器两侧按 (动词,目标) 配不上对 —— 本条判不了它们"))
        for tgt, es, cs in pairs:
            for slot in slots_want:
                if slot not in es and slot not in cs:
                    continue
                n_slot += 1
                et, ct = es.get(slot), cs.get(slot)
                if et is None:
                    n_cn_only += 1          # 中文侧独有的可见正文：没有英文可比，不是缺陷
                    continue
                if len(et) < min_en:
                    n_short += 1            # 短到不该用比值判（标签、一个词的 label）
                    continue
                n_gated += 1
                en_chars += len(et)
                cn_chars += len(ct or "")
                why = []

                # A 缺失／无汉字
                if not ct or not _CJK_RE.search(ct):
                    why.append(f"中文侧这段可见正文{'整个没有' if not ct else '一个汉字都没有'}"
                               f"（英文 {len(et)} 字）")
                else:
                    ratio = len(ct) / len(et)
                    ratios.append(round(ratio, 3))
                    # B 段级中英字符比
                    if ratio < min_ratio:
                        why.append(f"中英字符比 {ratio:.3f} < {min_ratio}"
                                   f"（EN {len(et)} / CN {len(ct)}；全库这类段实测区间 "
                                   f"[0.263, 0.406]）")
                    # D 句数
                    en_s = len(_EN_SENT.findall(et))
                    cn_s = len(_CN_SENT.findall(ct))
                    need_s = int(en_s * sent_frac)
                    if en_s and cn_s < need_s:
                        why.append(f"中文句数 {cn_s} < 英文 {en_s} 句 × {sent_frac} = {need_s}")
                # C 本段范围内的定译专名
                hits = _seg_anchor_hits(anchors, et, cache) if anchors else []
                if hits:
                    n_anchor_slot += 1
                    n_anchor_hit += len(hits)
                    miss = [f"{k}→{v}" for k, v in hits if v not in (ct or "")]
                    if miss:
                        why.append(f"本段英文命中 {len(hits)} 条词表锚点，中文缺 {len(miss)} 条："
                                   f"{miss[:6]}")
                # E 数字（现读：含数字的英文段 0 段 ⇒ 这一支无从查起，不是查过了）
                nums = list(dict.fromkeys(_SEG_NUM.findall(et)))
                if nums:
                    n_num_slot += 1
                    lost = [n for n in nums if n not in (ct or "")]
                    if lost:
                        why.append(f"英文段里的阿拉伯数字中文没有：{lost[:6]}")
                # F 段内标记（现读：含标记的英文段 0 段，同上）
                marks = _SEG_MARKUP.findall(et)
                if marks:
                    n_mark_slot += 1
                    if len(_SEG_MARKUP.findall(ct or "")) != len(marks):
                        why.append(f"段内标记数两侧不同（EN {len(marks)} / "
                                   f"CN {len(_SEG_MARKUP.findall(ct or ''))}）")
                if not why:
                    n_ok += 1
                    continue
                head = et[:40]
                j = None
                for k2, e2 in enumerate(rule.get("except_slots", [])):
                    if (path.endswith(e2["path"]) and e2.get("slot", slot) == slot
                            and e2.get("en_head", head) == head):
                        j = k2
                        break
                if j is not None:
                    used[j] += 1
                    n_exempt += 1
                else:
                    bad.append((repo, pack, f"{path} → {tgt[:30]} [{slot}]",
                                "；".join(why) + f"：EN {et[:48]!r} / CN {(ct or '')[:32]!r}"))

    dead_rows, n_dead = _unused_exempt(rule, used, "except_slots", "槽 ")
    # 判了几条规矩：每个在册槽名 + 三支覆盖判据（句数比 / 字数比 / 锚点）——**真判过段才算**。
    if n_gated:
        _unit(*(f"slot:{s}" for s in (rule.get("slots") or _ENR_TEXT_PARAMS)))
        _unit("cov:min_sentence_frac", "cov:min_ratio", "cov:anchors")

    bad.extend(dead_rows)
    band = (f"[{min(ratios):.3f}, {max(ratios):.3f}] 中位 {sorted(ratios)[len(ratios) // 2]:.3f}"
            if ratios else "（无）")
    detail = (f"闸下 {n_leaf} 叶 · 配对增强器 {n_pair} 个（配不上 {n_unpaired}）· "
              f"在册可见正文槽 {n_slot}（英文太短跳过 {n_short} · 中文独有 {n_cn_only}）· "
              f"**判过 {n_gated} 段**（一致 {n_ok}）· EN {en_chars} 字 / CN {cn_chars} 字 · "
              f"比值 {band} · 锚点：{anchor_note}，命中 {n_anchor_slot} 段 / {n_anchor_hit} 条 · "
              f"含阿拉伯数字的英文段 {n_num_slot} · 含段内标记的英文段 {n_mark_slot}"
              + ("（这两支 = 0 ⇒ **无从查起**，不是查过了没问题）"
                 if not n_num_slot and not n_mark_slot else "")
              + f" · 已登记豁免 {len(used)} 条 / 命中 {n_exempt} / 死豁免 {n_dead}")
    for field, got, txt in (("min_leaves", n_leaf, "闸下叶数"),
                            ("min_slots", n_slot, "在册可见正文槽数"),
                            ("min_gated", n_gated, "真正判过的段数"),
                            ("min_anchor_terms", len(anchors), "锚点表条数"),
                            ("min_anchor_slots", n_anchor_slot, "命中锚点的段数"),
                            ("min_anchor_hits", n_anchor_hit, "命中的锚点条数")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{txt}只数到 {got}（要求 ≥{want}）—— 这条断言在空转："
                        f"上游把 readaloud 改名了？词表路径变了？`_enr_slots` 被改坏了？"
                        f"（改过判据就必须跑 --selftest，光跑主闸抓不到这一类）"))
    if n_unpaired > rule.get("max_unpaired", 0):
        bad.append(("-", "配置", rule["id"],
                    f"配不上对的增强器有 {n_unpaired}（上限 {rule.get('max_unpaired', 0)}）"))
    return bad, detail



def _twin_repo_dir(name, ctx):
    """把规则里的仓名解析成**本次运行**的实际目录。取不到就当场炸，**不许静默兜底**。

    这个函数是第二十五轮从 `a_twin_files` 里拆出来的，起因是空转形态 (g)：
    旧版写的是 `ctx.repos.get(name, name)`，而规则里 `repo_a` 填的是**目录名**
    `1-Ember汉化插件`、`ctx.repos` 的键却是 `ember` —— 于是 `.get(k, k)`
    **永远走兜底**，退化成裸相对路径按 **cwd** 解析。实测后果：
      · 主闸结论依赖 cwd：项目根 61/0，在 `3-常用脚本/qa` 下跑是 60/1；
      · `--root <副本>` 的灵敏度回测**完全作用不到副本树上** ——
        往副本注入漂移，它照样报「比对 1 对文件、violations = 0」，因为比的是真实树；
      · 而它同时满足了 `min_pairs≥1` 那道专防空转的闸、报告自己比过了、结论是绿的。
    ⇒ 所以这里三种情形分得清清楚楚，一种都不许含混：
    """
    if name in ctx.repos:
        return ctx.repos[name]
    if name in REPOS:
        # 仓名合法，只是被 `--repo` 限定掉了没进 ctx。仍按**本次运行的 ROOT**
        # （`--root` 会改它）解析 —— 这不是「换成另一份输入」，是同一份输入的同一个来源。
        return os.path.join(ROOT, REPOS[name])
    raise KeyError(
        f"规则里的仓名 {name!r} 不在 REPOS（可选：{sorted(REPOS)}）"
        f" —— 这是**规则写错了**，必须当场失败。旧版在这里静默回落到"
        f"「就拿键本身当路径」，于是判据比的根本不是你以为的那棵树（空转形态 (g)）。"
    )


def a_twin_files(rule, ctx):
    """两份**必须逐字节相同**的文件。

    起因：自检面板那份 `.mjs` 在 ember 与 crucible 两个汉化模块里各存一份 —— 复制而不是
    跨模块 import，是因为只装其中一个的用户那边没有对方的文件，跨模块 import 会 404。
    代价是**两份会漂**，而本项目已经因为「两处讲同一件事、改了一边忘了另一边」栽过多次
    （最近一次：`R-moon-hollow` 的 why 与 `scan_renamed_terms.py` 的 docstring 同仓互相打脸）。

    ⚠ 判据本身要能说出「我比了几对文件」——比不到（路径不存在）必须**失败**而不是静默通过，
    否则就是空转形态 (e)「输入缺失但判据照跑」。
    """
    import hashlib
    bad, pairs = [], 0
    for pair in rule["pairs"]:
        # ⚠ 路径按**仓**解析（`ctx.repos` 存的就是各仓在当前 root 下的实际目录），
        #   这样 `--root <副本>` 的灵敏度回测才真的作用在副本树上。
        #   注释与实现曾经正好相反（`.get(k, k)` 永远走兜底 → 按 cwd 解析真实树），
        #   见 `_twin_repo_dir` 的 docstring 与空转形态 (g)。**不许改回 `.get(k, k)`。**
        a = os.path.join(_twin_repo_dir(pair["repo_a"], ctx), pair["path_a"])
        b = os.path.join(_twin_repo_dir(pair["repo_b"], ctx), pair["path_b"])
        if not os.path.isfile(a) or not os.path.isfile(b):
            bad.append(("-", "-", pair["path_a"],
                        f"配对文件缺失（a 在={os.path.isfile(a)} b 在={os.path.isfile(b)}）"
                        f" —— 这一条**没比成**，不是通过"))
            continue
        pairs += 1
        _unit(f"twin:{pair['repo_a']}/{pair['path_a']}")   # 每对逐字节比对各算一条规矩
        ha = hashlib.sha256(open(a, "rb").read()).hexdigest()
        hb = hashlib.sha256(open(b, "rb").read()).hexdigest()
        if ha != hb:
            bad.append(("-", "-", pair["path_a"],
                        f"与 {pair['repo_b']}/{pair['path_b']} **不是逐字节相同**（{ha[:12]} vs {hb[:12]}）"
                        f" —— 改了一边忘了另一边"))
    if pairs < rule.get("min_pairs", 1):
        bad.append(("-", "-", "-",
                    f"只比成 {pairs} 对（要求 ≥{rule.get('min_pairs', 1)}）—— 这条断言在空转"))
    return bad, f"比对 {pairs} 对文件"


def a_source_literal(rule, ctx):
    """指定**源码文件**里必须出现 / 不许出现的字面量（路径按仓解析）。

    为什么第二十七轮要新加这个类型
    ------------------------------
    第二十六轮把自检面板 D 档的档名从「**键活性**」订正成「**上游字面量存在性（仅供人工复核）**」——
    那是那一轮最有价值的一处订正：叫「键活性」时，这一档报的 77 条被当成真缺陷追了整整一轮
    （第二十四轮），而它按设计**根本判不了键活不活**，只判「上游语料里有没有这个字面量」。
    可是**名字漂回去不会有任何判据响**：`cn_absent` 只扫 compendium + lang 两个通道，
    够不到 `.mjs`；`twin_files` 只保证两份面板彼此相同 —— **一起漂回去它照样绿**。
    ⇒ 本类型只做一件事：把「这个名字」钉在源码里。

    ⚠ 文件读不到 = **没判成**，不是通过（空转形态 (e)）；`min_files` 兜住「一个都没读成」。
    ⚠ `forbid_re` 只当第二道保险：真正扛事的是 `require`（把整行档名逐字符钉死），
      因为 `require` **不可能因为正则写坏而静默失效**（形态 (a) 与 (f) 在这条上不成立）。
    """
    bad, n_files, n_checks = [], 0, 0
    for f in rule["files"]:
        p = os.path.join(_twin_repo_dir(f["repo"], ctx), *f["path"].split("/"))
        if not os.path.isfile(p):
            bad.append((f["repo"], "-", f["path"],
                        f"文件不在：{p} —— 这一条**没判成**，不是通过（空转形态 (e)）"))
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                txt = fh.read()
        except Exception as exc:                       # noqa: BLE001 —— 读不动必须判失败
            bad.append((f["repo"], "-", f["path"], f"读不动：{exc!r} —— 没判成，不是通过"))
            continue
        n_files += 1
        for lit in rule.get("require", []):
            n_checks += 1
            _unit(f"require:{f['repo']}:{lit[:40]}")   # 每文件每条字面量各算一条规矩
            if lit not in txt:
                bad.append((f["repo"], "-", f["path"],
                            f"少了必须逐字符存在的字面量 {lit!r} —— 改名/改措辞了？"
                            f"正确处理是「改回来」或「显式推翻那条裁决并同时改断言」"))
        for pat in rule.get("forbid_re", []):
            n_checks += 1
            _unit(f"forbid_re:{f['repo']}:{pat[:40]}")
            m = re.search(pat, txt)
            if m:
                bad.append((f["repo"], "-", f["path"],
                            f"出现了被禁的写法 {m.group(0)[:80]!r}（正则 `{pat}`）"))
    floor = rule.get("min_files", 1)
    if n_files < floor:
        bad.append(("-", "-", "-",
                    f"只读成 {n_files} 个源码文件（要求 ≥{floor}）—— 这条断言在空转"))
    if n_checks < rule.get("min_checks", 1):
        bad.append(("-", "配置", "-",
                    f"只跑了 {n_checks} 条字面量判据（要求 ≥{rule.get('min_checks', 1)}）"
                    f"—— require / forbid_re 都被清空了？"))
    return bad, f"读 {n_files} 个源码文件，跑 {n_checks} 条字面量判据"


# ======================= 自检面板 D 档：**现跑**（第二十八轮 ②）
#
# 为什么非有这一条不可
# --------------------
# `R-selfcheck-d-section-name` 是 `source_literal`，**只钉档名那一个字符串** ——
# 档名漂回「键活性」会红，可 **719 个键的结论漂成什么样都不会红**。
# 复核 grep 整个本文件，`keyLiveness` 出现 **0 次**：没有任何断言跑过那块面板。
# 而全项目唯一能跑出这组数的执行体，是 `4-临时脚本/2026-08-16-round26/probe_panel_report.mjs`
# —— 一次性探针，**git 未跟踪**。于是三路文档都在引用的「719 distinct / 1273 raw ·
# miss 4 键 / 7 报文」，**可再现性挂在一台机器上的一份未入库脚本上**。
#
# ⇒ 这是本项目登记的空转形态 (c) 在**更高一层**复发：不是判据读了旧快照，
#   是**「结论」本身就是一张快照**。修法只有一个：把执行体提升进 `3-常用脚本/qa/`
#   （`selfcheck_panel_runner.mjs`，已入库），每跑一次闸就**现跑一次面板**。
#
# 阈值的方向，以及**为什么这条的强度不放在阈值上**
# ------------------------------------------------
# · 覆盖侧（checkedDistinct / rawChecked / fetchOk …）按「**不得低于**」记：
#   上游长东西、表里加键，这些数只会涨；掉下来只可能是**表被砍了**或**语料没抓着**。
# · miss 侧（missDistinct / rawMiss / fetchFail）按「**不得高于**」记：
#   查无此串的键只许越修越少；涨了就是有键真的对不上上游了，人必须看一眼。
# · ⚠ 但这两侧都是**可调的数**，而本轮堵的正是「调松阈值」这条路。所以这条断言真正扛事的
#   是另外几条**不含阈值、调不松**的恒等式：
#     ① 逐表行的 `checked` 之和 **必须等于**「合计」自报的 rawChecked
#        （面板里两条不同的汇总路径，对不上就是它自己的记账坏了）；
#     ② 模板三路记账（引用推导 + 拼路径展开 + 无引用快照）之和 **必须等于**抓到的份数；
#     ③ 喂进去几张表，报文里就得有几张表的行 —— 少一行 = 有张表被静默吞了；
#     ④ **假阴性对照**：6 个自造的、上游绝无的串必须 **6/6** 被报出来。
#        匹配器要是哪天变成「什么都找得到」（形态：判据在，但恒真），阈值全在也照样绿，
#        只有这一条会当场红。
#     ⑤ 子串型改词那条**已知边界**必须仍然复现（喂 `Increase Ability` 报 0 条）——
#        它是写进报文的边界，不是缺陷；哪天它变了，说明匹配口径被人动过。
#     ⑥ 面板现跑吐出来的**档名**必须与 `R-selfcheck-d-section-name` 钉的那一行一致 ——
#        那条钉的是源码字面量，这条钉的是**运行时真的用了它**。
#   把阈值全调松，①—⑥ 一条都动不了。


def _panel_data_root(rule, ctx, panel_src):
    """Foundry 的 Data 目录（fetch 桩的根），取不到返回 `(None, 说明)`。

    ⚠ **不写死路径**，也不新加一个配置项：从 `meta.lang_sources` 里那个上游安装目录
    （`…/Data/modules/ember`）**倒推**，倒推时用的是**面板源码里现抠的 `EMBER_ROOT`**
    （面板 fetch 的 url 就是 `${EMBER_ROOT}/…`）。倒推完再核一遍尾巴对不对得上 ——
    对不上就当场失败：桩指错地方 ⇒ 语料一份也抓不到 ⇒ 判据照跑，那是形态 (h)。
    （node 侧还会**再核一次**，两边独立核，见 `selfcheck_panel_runner.mjs` 开头。）
    """
    name = rule.get("upstream_repo", "ember")
    srcs = (getattr(ctx, "meta", None) or {}).get("lang_sources") or {}
    if name not in srcs:
        return None, f"meta.lang_sources 里没有 {name!r} —— 上游安装目录无从找起"
    up = os.path.normpath(os.path.expandvars(srcs[name]))
    m = re.search(r'const EMBER_ROOT = "([^"]+)";', panel_src)
    if not m:
        return None, ("面板源码里抠不到 `const EMBER_ROOT = \"…\";` —— "
                      "fetch 桩的上游契约出处没了，必须重新锚定，不许照跑")
    parts = [p for p in m.group(1).split("/") if p]
    root = up
    for want in reversed(parts):
        head, tail = os.path.split(root)
        if tail != want:
            return None, (f"上游安装目录 {up} 的结尾与面板的 EMBER_ROOT={m.group(1)!r} 对不上 —— "
                          f"倒推不出 Data 根目录，桩会指到空地方（形态 (h)）")
        root = head
    return root, root


def a_panel_liveness(rule, ctx):
    """**现跑**自检面板的 D 档（node 子进程），见上面那段注释。"""
    import shutil
    panel = os.path.join(_twin_repo_dir(rule["repo"], ctx), *rule["panel"].split("/"))
    tables = os.path.join(_twin_repo_dir(rule.get("tables_repo", rule["repo"]), ctx),
                          *rule["tables_src"].split("/"))
    for what, p in (("面板", panel), ("硬编码表", tables)):
        if not os.path.isfile(p):
            return ([("-", "-", what, f"文件不在：{p} —— **没跑成**，不是通过（空转形态 (e)）")],
                    "没跑成")
    data_root, note = _panel_data_root(rule, ctx, open(panel, encoding="utf-8").read())
    if data_root is None:
        return [("-", "-", "上游 Data 目录", f"{note} —— **没跑成**，不是通过")], "没跑成"
    node_bin = rule.get("node_bin", "node")
    node = shutil.which(node_bin)
    if not node:
        return ([("-", "-", "node",
                  f"PATH 里找不到 {node_bin!r} —— 面板是 ESM `.mjs`，没有 node 就**跑不了**。"
                  f"这是**没跑成**，不许静默跳过（空转形态 (e)）")], "没跑成")
    runner = os.path.join(HERE, rule.get("runner", "selfcheck_panel_runner.mjs"))
    if not os.path.isfile(runner):
        return ([("-", "-", runner,
                  "找不到 node 侧执行体 —— **没跑成**。⚠ 这个执行体必须是**入库**的："
                  "它一旦只存在于某台机器上，这条断言的结论就又变成一张快照了")], "没跑成")

    tmp = tempfile.mkdtemp(prefix="panel_liveness.")
    try:
        spec = {
            "panel": panel,
            "tables_src": tables,
            "stub_import": rule["stub_import"],
            "data_root": data_root,
            "out": os.path.join(tmp, "result.json"),
            "fakes": rule.get("fakes") or {},
            "substr_probe": rule.get("substr_probe"),
        }
        spec.update(rule.get("mutate") or {})      # 只给 --selftest 的回测用
        spec_p = os.path.join(tmp, "spec.json")
        with open(spec_p, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, ensure_ascii=False)
        proc = subprocess.run([node, runner, spec_p], capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not os.path.exists(spec["out"]):
            msg = (proc.stderr or proc.stdout or "").strip()[-420:]
            return ([("-", "-", "runner",
                      f"node 侧退出码 {proc.returncode}：{msg} —— 这一条**没跑成**，不是通过")],
                    "没跑成")
        res = json.load(open(spec["out"], encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001 —— 跑不起来必须判失败
        return [("-", "-", "runner", f"node 侧跑不起来：{exc!r} —— 没跑成，不是通过")], "没跑成"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    c = res.get("counts") or {}
    bad = []

    # ① 覆盖侧「不得低于」／ miss 侧「不得高于」。⚠ 方向不一样，理由见上面那段。
    for field, floor in (rule.get("min") or {}).items():
        got = c.get(field)
        _unit(f"panel:min:{field}")                 # 每个覆盖侧地板各算一条规矩
        if got is None:
            bad.append(("-", "配置", field, "执行体没吐出这个数 —— 记的和跑的对不上，判失败"))
        elif got < floor:
            bad.append(("-", "面板", field,
                        f"实得 {got}，低于记录值 {floor}（覆盖侧只许涨不许掉）—— "
                        f"表被砍了？语料没抓着？**先分清是哪一种，别直接改小记录值**"))
    for field, ceil in (rule.get("max") or {}).items():
        got = c.get(field)
        _unit(f"panel:max:{field}")                 # 每个 miss 侧上限各算一条规矩
        if got is None:
            bad.append(("-", "配置", field, "执行体没吐出这个数 —— 记的和跑的对不上，判失败"))
        elif got > ceil:
            bad.append(("-", "面板", field,
                        f"实得 {got}，高于记录值 {ceil}（miss 侧只许降不许涨）—— "
                        f"有键对不上上游了，人得去看一眼是真漂了还是语料少抓了："
                        f"{'、'.join(str(x) for x in (res.get('miss') or [])[:6])}"))

    # ② 以下全是**不含阈值**的恒等式：调松阈值动不了它们（本条的强度在这儿）。
    #    每条恒等式各算一条规矩，外加逐个自造串的假阴性对照。
    _unit("panel:id:rowCheckedSum", "panel:id:tplParts", "panel:id:tablesWithoutRow",
          "panel:id:fake")
    # 第三十二轮 V18 补的三条恒等式 + 台账那两条（同样不含阈值，调松阈值动不了它们）
    _unit("panel:id:tableKeysXcheck", "panel:id:ledgerRaw", "panel:id:ledgerDistinct",
          "panel:id:ledgerBalanced", "panel:id:ledgerNoReason")
    _unit(*(f"panel:fake:{k2}" for k2 in (rule.get("fakes") or {})))
    if rule.get("substr_probe") is not None:
        _unit("panel:id:substr")
    if rule.get("section"):
        _unit("panel:id:section")
    if c.get("rowCheckedSum") != c.get("rawChecked"):
        bad.append(("-", "面板", "记账",
                    f"逐表行的 checked 之和 {c.get('rowCheckedSum')} ≠ 合计自报的 rawChecked "
                    f"{c.get('rawChecked')} —— 面板自己的两条汇总路径对不上"))
    if c.get("tplParts") != c.get("tplFiles"):
        bad.append(("-", "面板", "模板记账",
                    f"模板三路记账之和 {c.get('tplParts')} ≠ 抓到的份数 {c.get('tplFiles')}"))
    if c.get("tablesWithoutRow"):
        bad.append(("-", "面板", "表被吞",
                    f"{c.get('tablesWithoutRow')} 张表喂进去了却没有对应的报文行："
                    f"{res.get('tablesWithoutRowNames')} —— 合计的数看不出这种静默丢表"))
    want_fake = c.get("fakeWanted", 0)
    if not want_fake:
        bad.append(("-", "配置", "假阴性对照",
                    "一条自造串都没喂 —— 少了这道对照，「匹配器变成什么都找得到」不会有人发现"))
    elif c.get("fakeMiss") != want_fake or c.get("fakeChecked") != want_fake:
        bad.append(("-", "面板", "假阴性对照",
                    f"自造 {want_fake} 个上游绝无的串，只核了 {c.get('fakeChecked')} 个、"
                    f"报出 {c.get('fakeMiss')} 个 —— 匹配器变成恒真了？"
                    f"这时候所有「通过」都不作数"))
    if rule.get("substr_probe") is not None:
        want_sub = rule.get("substr_expect_miss", 0)
        if c.get("substrMiss") != want_sub:
            bad.append(("-", "面板", "子串边界",
                        f"喂 {rule['substr_probe']!r} 报出 {c.get('substrMiss')} 条（记的是 "
                        f"{want_sub}）—— 这是写进报文的**已知边界**，变了说明匹配口径被动过，"
                        f"报文里那段边界说明也就跟着过期了"))
    # ——— 第三十二轮 V18：台账那一侧的恒等式。
    #     ⚠ 这几条**必须在 Python 侧自己再算一遍**，不能只读面板自报的 `ledgerBalanced`：
    #       面板把那个布尔写死成 true 是个单点动作，只有「拿转发过来的三个数现算」才接得住。
    #     ⚠ 「表侧独立计数 == 面板自报的登记数」是这份计数器**当初宣称的用途**
    #       （见 `selfcheck_panel_runner.mjs` 的 `countTableKeys` 注释）—— 在本轮之前
    #       它算了、印了，**全项目没有一处判它**，而且它算错了整整三轮（721/1304 vs 真值 887/1459）。
    #       注释宣称的用途与实现不符，正是 V14 挂三轮的成因；这一条把「宣称」接到了闸上。
    for a_k, b_k, why in (("tableKeysRaw", "registeredRaw", "raw"),
                          ("tableKeysDistinct", "registeredDistinct", "distinct")):
        if c.get(a_k) != c.get(b_k):
            bad.append(("-", "面板", "登记数对账",
                        f"表侧独立数出来的 {a_k}={c.get(a_k)} ≠ 面板自报的 {b_k}="
                        f"{c.get(b_k)}（{why}）—— 两条独立汇总路径对不上：要么面板悄悄少登记了"
                        f"一批键，要么表侧那份计数器又没穿过 `{{table,…}}` 包装（V14 的老坑）"))
    if None not in (c.get("rawChecked"), c.get("uncheckedRaw"), c.get("registeredRaw")):
        if c["rawChecked"] + c["uncheckedRaw"] != c["registeredRaw"]:
            bad.append(("-", "面板", "台账 raw",
                        f"{c['rawChecked']} 已核 + {c['uncheckedRaw']} 按设计没核 ≠ "
                        f"{c['registeredRaw']} 登记 —— 面板自己的记账坏了，这一档的数全部不可信"))
    if None not in (c.get("checkedDistinct"), c.get("uncheckedDistinct"),
                    c.get("registeredDistinct")):
        if c["checkedDistinct"] + c["uncheckedDistinct"] != c["registeredDistinct"]:
            bad.append(("-", "面板", "台账 distinct",
                        f"{c['checkedDistinct']} + {c['uncheckedDistinct']} ≠ "
                        f"{c['registeredDistinct']} —— 去重侧的账没平"))
    if c.get("ledgerBalanced") is not True:
        bad.append(("-", "面板", "台账自报",
                    f"面板自报 ledgerBalanced={c.get('ledgerBalanced')!r} —— 它必须恒 true"))
    if c.get("ledgerNoReason"):
        bad.append(("-", "面板", "没核却说不出原因",
                    f"{c.get('ledgerNoReason')} 张表没核却登记不出为什么："
                    f"{res.get('ledgerNoReasonNames')} —— 「没核」必须说得出原因，"
                    f"说不出就是被静默吞了，按缺陷处理（这个数必须恒 0）"))
    if rule.get("section") and res.get("section") != rule["section"]:
        bad.append(("-", "面板", "档名",
                    f"现跑吐出的档名是 {res.get('section')!r}，记的是 {rule['section']!r} —— "
                    f"R-selfcheck-d-section-name 钉的是源码里那一行，本条钉的是"
                    f"**运行时真的用了它**"))

    detail = (f"现跑面板 D 档：核 {c.get('checkedDistinct')} 个不同键 / 登记 {c.get('rawChecked')} 条"
              f"（逐行之和 {c.get('rowCheckedSum')}），查无此串 {c.get('missDistinct')} 键 /"
              f" {c.get('rawMiss')} 报文；抓上游语料 {c.get('fetchOk')} 成 / {c.get('fetchFail')} 败"
              f"（模板 {c.get('tplFiles')} 份 = 三路记账 {c.get('tplParts')}）；"
              f"{c.get('tableRows')} 张表全部有报文行；正则表 {c.get('tableRegexEntries')} 条按设计核不了；"
              f"假阴性对照 {c.get('fakeMiss')}/{want_fake}；子串边界复现 {c.get('substrMiss')}；"
              f"台账：登记 raw {c.get('registeredRaw')} / distinct {c.get('registeredDistinct')}"
              f"（表侧独立数 {c.get('tableKeysRaw')} / {c.get('tableKeysDistinct')}，对得上）"
              f"= 已核 {c.get('rawChecked')} / {c.get('checkedDistinct')}"
              f" + 按设计没核 {c.get('uncheckedRaw')} / {c.get('uncheckedDistinct')}"
              f"（包装表 {c.get('wrappedTables')} 张 / 正则表 {c.get('regexTables')} 张；"
              f"没核却说不出原因 {c.get('ledgerNoReason')} 张）")
    return bad, detail


# ============================ 正则译文用例（第二十六轮 ①，第二十七轮 ① 扩面）
#
# 为什么非有这一条不可
# --------------------
# `PATTERNS`（28 条正则）· `PREFIXED`（19 条前缀）· `NOTIFICATION_PATTERNS`（31 条正则）
# 是本项目**能主动改坏译文**的三张表：命中就把**整串**替换掉。
# ⚠ **不要再写「唯一能主动改坏译文的两张表」**（第二十六轮的措辞，第二十七轮复核判为错）：
#   第三张 `NOTIFICATION_PATTERNS` 由 `translateNotification` 消费，而那个函数包的是
#   `ui.notifications.notify` —— **所有模块共用的全局钩子**。PATTERNS 只在 Ember 自己的窗口 /
#   认出归属的框 / 注入子树 / 聊天卡上跑，够不到别的模块；通知这条通道**别的模块发的每一条提示
#   都要过一遍**，爆炸半径更大。第二十六轮那条规则的 `why` 末尾自己写着
#   「NOTIFICATION_PATTERNS 是作用域表、不进本条」，与开头那句「唯一」自相矛盾 —— 本轮一并收了。
#
# 而在第二十六轮之前，这几张表**一道常设闸都没有**：
#   · 自检面板的 D 档「上游字面量存在性」按设计核不了正则表 —— 对 PREFIXED 19 + PATTERNS 28
#     共 47 个键直接报 ⛔（键不是字面量，无从在上游语料里查「有没有这个串」）；
#   · 61 条断言里 `PATTERNS` 出现 **0 次**；
#   · 那一轮验它的探针（`4-临时脚本/2026-08-16-round25/probe_patterns_c.mjs`）是**一次性**的，
#     跑完没人再跑。
# 第二十五轮真正咬人的缺陷就出在这两张表上：`^Result of (.+)$` 会把任意以 "Result of " 开头的
# 英文句子吃掉 →「结果：the investigation was inconclusive」；`^(Music|Environment): (.+?)…$`
# 会把任意 `Music: …` 形状的句子改成「音乐：…」。
# ⇒ 在此之前，**下一轮谁再动一次这两条正则，没有任何常设判据会响。**
#
# 判据形态：**真的 import 发布中的 `translateText` / `translateNotification` 跑用例**
# ----------------------------------------------------------------------------------
# 不在 Python 里重写一遍正则 —— 重写一遍就是又造一个会漂移的副本，那正是本项目反复吃亏的
# 形态（同 `_run_same_en_split` 的注释：判据不许复制第二份实现）。被判的是 ESM `.mjs`，
# 所以走 `node <本目录>/translate_cases_runner.mjs <spec.json>` 子进程。
# `translateNotification` 与三张表都是模块私有的，runner 在 **harness 副本**上追加一行
# `export {…}`（只加导出、函数体一个字节不改），且**每个名字先确认声明存在**，找不到当场失败。
# ⚠ **node 取不到 / import 失败 / 导出名对不上 / 上游抠不到编排名 —— 一律当场失败并说明**，
#   不许静默跳过（空转形态 (e)：「0 条问题」有两种含义，判据有义务把两者分开）。
#
# 四段用例，全部固化在 `RESOLUTIONS.assertions.json` 里
# ----------------------------------------------------
# (A) **反例**：上游产不出、但形状相近的英文串，**一个字都不许被翻动**。
#     13 条来自第二十五轮复核，其中 3 条是旧实现真会误翻的现场
#     （`Result of the investigation was inconclusive` / `Music: my custom playlist` /
#     `Environment: Rain`），另加 2 条钉住另外两条宽正则（`Day, Generic` 曾被旧的
#     `^Day\b(.*)$` 吃掉；`Nosuch Attunement Rank 3` 钉 `^(.+) Rank ([1-5])$` 的查表兜底）。
#     第二十七轮补 13 条专钉 **PREFIXED 那一层**（第二十六轮这一层**反例 0 条**）：
#     `startsWith(en + ": ")` 只锚在**串首**，所以判据要守的是
#     「**别把 Ember 根本不产出的串吃掉**」—— 别的模块产出的 `Name:` / `Type:` / `Source:` 这类
#     通用标签、以及 `Locations:`（复数）/ `location:`（小写）/ `Location:X`（无空格）/
#     `The Location: X`（不在串首）这几种上游产不出的近似写法。
# (B) **正例**：上游**真会产出**的串必须仍然翻得动，且**译文逐字符相符**。
#     只断言「翻动了」是不够的 —— 收紧过头会把译文改成别的东西而断言照样绿。
#     定义域出处逐条写在规则的 `why` 与用例注释里（`enrichCriticalResult` ember.mjs:22905-22913
#     只可能产出 `Result of ${dc-5}-` / `Result of ${dc+5}+`；音景三支 :16255 / :16266 / :16267）。
# (B2) **覆盖归因**（第二十七轮新增，在 runner 的 §④/§⑤）：三张表的**每一条**都必须被至少一条
#     正例触到，漏一条就报违规，并另有表长下限兜住「把表和用例一起删掉换全绿」。
#     ⚠ 这一段是被实测逼出来的：第二十六轮的闸名写着 `PATTERNS / PREFIXED`，79 条用例把
#       PATTERNS 打满 28/28，`PREFIXED` 却只触到 **2/19**（`Attunement`、`Music Mood`）；
#       把 `Knowledge`/`Location`/`Quest`/`Talent`/`Rarity` 五条译名逐个改坏重跑，闸 **5/5 全绿**。
#       **用例数不等于覆盖，人眼数不出来，就让判据自己数。**
#     归因用的是 `translateText` 派发顺序的镜像（EXACT → PREFIXED → PATTERNS），镜像本身有漂移
#     风险，所以每条用例都做**预测输出 vs 实跑输出**的自洽校验 —— 派发顺序被改而镜像没跟上，
#     当场报违规。镜像引用的是**真表对象本身**，不是抄一份表。
# (D) **通知闸**（第二十七轮新增）：31 条 `NOTIFICATION_PATTERNS` 逐条正例 + 反例，
#     反例重点是「**别的模块发的通知不许被吃掉**」（Token Mold / Dice So Nice / core 的权限提示）。
#     `translateNotification` 的**空白折叠回退支**另有专门用例：上游 ember.mjs:36922 与 126652
#     是跨行模板串，源码里的换行 + 缩进原样进了消息文本，用例照那两处的真实空白构造。
# (C) **全量护栏**：`ember.mjs` 里能抠出的**全部编排名** × 2 通道逐条过。
#     ⚠ 编排名**从上游现抠**，不许把 219 条硬编码进断言 —— 硬编码就是形态 (c)「读旧快照」，
#       上游一改断言还在核一份过期名单。抠不出来（上游打包形状变了）**当场失败**。
#     ⚠ 但「**恰好 8 个编排名整串留英**」这个集合**写死当护栏** —— 上游新增编排名时它会响，
#       **那是要的**（人来看一眼这条新名字该不该进表）。7 个属 events/weather 两个音景
#       （type 不是 music/environment，不进播放列表侧栏那两个下拉），
#       `Seven Sails` 是第二十二轮裁定**故意**留英。
#
# 形态 (h)：**喂输入的那一半也必须指得出上游契约的出处**
# ------------------------------------------------------
# (C) 段的编排名如果抄我们自己的 `ARRANGEMENTS` 表，就是拿**被判方**当输入 ——
# 表漏了哪条，用例就正好也不测哪条，永远全绿。所以取的是上游 `soundscapes` 注册表里的
# `arrangements[].label`，那正是 ember.mjs:16266 `${channel.capitalize()}: ${arrangement.label}`
# 里 `arrangement.label` 的来源。抽取逻辑写在 `translate_cases_runner.mjs`。
#
# ⚠ 第二十七轮修的 (h) **残留**：抽取器锚在**精确空白**上（`\n  arrangements: {` 与
#   `\n      label: "…"`），复核把上游 40 条 `label:` 行各加 1 个空格模拟上游换打包格式，
#   编排名 **219 → 196 静默下降，闸仍 0 违规** —— 两处原因，都已改：
#     ① `min_labels` 当时只卡 **150**，比现值低 69，掉 23 条照样在门槛之上
#        ⇒ 提到 **210**（Python 侧默认同步从 100 提到 210），并加一条
#        `labels_recorded`（现记 219）「不得低于上次记录值」；
#     ② 找不到 `arrangements` 块的那一支原先是 `continue`、**不计 unresolved**，是真静默
#        ⇒ 现在与「有但抠不出来」同等计入 `unresolved` 并列出是哪个音景。
#   ⇒ 形态 (h) 的要害不是「没有门槛」，是**门槛松到够不着现值**：判据得贴着现值走。
#
# ⚠ 第二十八轮补的是**上面这一整段的自指的洞**（复核实测，见 `_two_layer_floor`）：
#   这些「贴着现值」的门槛全都是**规则文件里的自报数**。复核把 `coverage.min_prefixed`
#   从 19 改回 2、`min_negative` / `min_positive` 放到 1，**其余一个字节不动 → 违规 0，闸变绿**
#   （对照：删一条用例 4/4 立刻红）。⇒「谁加表项不补用例就当场红」这条纪律，
#   **只要有人顺手把阈值调回去就整条消失，而且没有任何判据会响**。本轮的改法见下。


def _two_layer_floor(bad, what, actual, recorded, live=None, hard=None,
                     grow_hint="", key="-"):
    """可调阈值的**两层**判法：「现算下限」＋「不得低于历史记录值」，缺一不可。

    第二十八轮为什么非做成两层
    --------------------------
    上一轮所有反空转门槛（`min_prefixed` / `min_negative` / `min_positive` / `min_labels` …）
    都是**写在规则文件里的一个数**，而规则文件是可改的 —— 于是整套「表和用例必须同增同减」
    的纪律，被「把那个数改小」一步全部抹掉，且**没有任何判据会响**。
    这不是理论风险：复核只改三个数就把这条闸变绿了。

    两层各管什么（**缺一不可**，别只留一层）
    ----------------------------------------
    · **现算层 `live`**：本次运行**从被判对象身上现算**出来的值（表现在有多少条 /
      规则里现在写着多少条用例 / 上游现抠出多少个编排名）。
      它管的是「**有人把记录值调松了**」——`recorded < live` 直接判违规：
      记录值必须**追着现值走**，落在现值后面只有两种可能，一种是调松，一种是加了东西忘了记，
      **两种都必须当场停下来看**，不许闷着头绿。
    · **历史层 `recorded`**：上一次人工确认过的值，只有「不得低于」这一个方向。
      它管的是「**有人把东西删了**」——`actual < recorded` 判违规。
      现算层管不了删除：现算值是**跟着被判对象一起掉**的，删一条它就跟着降一条，永远自洽。
    · 可选的 **`hard`**：一个不跟任何东西联动的死数（如编排名的 210）。
      当现算层本身可能整体失效时（形态 (h)：喂输入的那一半坏了，现算值集体塌方），
      只有这种**不联动**的数还站得住。

    ⚠ **别指望这能防住蓄意的多处协同改动**：现算值来自被判文件、记录值来自规则文件，
      两份都是可改的 —— 同时改「删表项 + 删用例 + 改两个记录值」仍然能过。
      本函数保证的是：**任何一处单独调松都会当场红**，而要绕过它必须留下一串
      互相印证、写着「我把记录值改小了」的显式改动 —— 那种 diff 是**自证的**，
      而原先那种「悄悄把 19 改成 2」不是。判据能做到的边界就在这里，
      别在 why 里把它写成「不可能被绕过」。
    """
    if recorded is None:
        bad.append(("-", "配置", key,
                    f"{what}：规则里没有登记历史记录值 —— **两层里少了一层**。"
                    f"只剩现算下限的话，删东西时现算值跟着一起降，判据永远自洽 = 空转"))
    floors = [("历史记录值", recorded), ("现算下限", live), ("死下限", hard)]
    floor, floor_src = 0, "-"
    for src, v in floors:
        if v is not None and v > floor:
            floor, floor_src = v, src
    if actual < floor:
        bad.append(("-", "配置", key,
                    f"{what}：实得 {actual}，低于{floor_src} {floor} —— "
                    f"（历史记录值 {recorded} · 现算下限 {live} · 死下限 {hard}）。"
                    f"东西被删了？还是喂输入的那一半坏了？**确认清楚再改记录值，不许直接改小了事**"))
    if recorded is not None and live is not None and recorded < live:
        bad.append(("-", "配置", key,
                    f"{what}：规则里的历史记录值 {recorded} **低于现算值 {live}** —— "
                    f"要么是有人把阈值调松了（第二十八轮堵的就是这条作弊路径），"
                    f"要么是{grow_hint or '涨了但忘了补记录'}。"
                    f"两种都得人来看一眼，把记录值改成 {live} 之前这条闸就是红的"))
    return floor


def _trans_upstream(rule, ctx):
    """上游 `ember.mjs` 的绝对路径，取不到返回 `(None, 说明)`。

    ⚠ **故意不受 `--root` 影响** —— 与 lang 通道同理：上游基准永远取真实安装目录
    （`meta.lang_sources`），副本树里没有上游。灵敏度回测改的是**被判文件**那一侧。
    """
    name = rule.get("upstream_repo", "ember")
    srcs = (getattr(ctx, "meta", None) or {}).get("lang_sources") or {}
    if name not in srcs:
        return None, (f"meta.lang_sources 里没有 {name!r} —— 上游 ember.mjs 无从找起，"
                      f"全量编排名那一段**无从查起**（不是「查过了没问题」）")
    p = os.path.join(os.path.expandvars(srcs[name]),
                     *rule.get("upstream_src", "scripts/ember.mjs").split("/"))
    if not os.path.isfile(p):
        return None, f"上游文件不在：{p}"
    return p, p


def a_translate_cases(rule, ctx):
    """把用例交给**发布中的** `translateText` 跑（node 子进程），见上面那段注释。"""
    import shutil
    # 被判文件按**仓**解析，用的是 twin_files 那条 (g)-安全的解析器：
    # 仓名写错当场 KeyError，`--root <副本>` 真的作用在副本树上（自检就靠这个做灵敏度回测）。
    src = os.path.join(_twin_repo_dir(rule["repo"], ctx), *rule["src"].split("/"))
    if not os.path.isfile(src):
        return ([("-", "-", rule["src"],
                  f"被判文件不在：{src} —— 这一条**没跑成**，不是通过（空转形态 (e)）")],
                "没跑成")
    ember, note = _trans_upstream(rule, ctx)
    if ember is None:
        return [("-", "-", "上游 ember.mjs", f"{note} —— 这一条**没跑成**，不是通过")], "没跑成"
    node_bin = rule.get("node_bin", "node")
    node = shutil.which(node_bin)
    if not node:
        return ([("-", "-", "node",
                  f"PATH 里找不到 {node_bin!r} —— 被判的是 ESM `.mjs`，没有 node 就**跑不了**。"
                  f"这一条是**没跑成**，不许静默跳过（空转形态 (e)）")], "没跑成")
    runner = os.path.join(HERE, rule.get("runner", "translate_cases_runner.mjs"))
    if not os.path.isfile(runner):
        return [("-", "-", runner, "找不到 node 侧执行体 —— 这一条**没跑成**，不是通过")], "没跑成"

    tmp = tempfile.mkdtemp(prefix="translate_cases.")
    try:
        spec = {
            "src": src,
            "ember_mjs": ember,
            "harness": os.path.join(tmp, "_harness.mjs"),
            "out": os.path.join(tmp, "result.json"),
            "stub_import": rule["stub_import"],
            "negative": rule.get("negative", []),
            "positive": rule.get("positive", []),
            "notify_negative": rule.get("notify_negative", []),
            "notify_positive": rule.get("notify_positive", []),
            "coverage": rule.get("coverage") or {},
            "arrangements": rule.get("arrangements"),
        }
        spec_p = os.path.join(tmp, "spec.json")
        with open(spec_p, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, ensure_ascii=False)
        proc = subprocess.run([node, runner, spec_p], capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not os.path.exists(spec["out"]):
            msg = (proc.stderr or proc.stdout or "").strip()[-420:]
            return ([("-", "-", "runner",
                      f"node 侧退出码 {proc.returncode}：{msg} —— 这一条**没跑成**，不是通过")],
                    "没跑成")
        res = json.load(open(spec["out"], encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001 —— 跑不起来必须判失败
        return [("-", "-", "runner", f"node 侧跑不起来：{exc!r} —— 没跑成，不是通过")], "没跑成"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bad = []
    for v in res.get("violations", []):
        bad.append(("-", "translateText", str(v.get("input", ""))[:76],
                    f"[{v.get('group')}] 实得 {v.get('got')!r}，应为 {v.get('want')!r}"
                    + (f"｜{v['note']}" if v.get("note") else "")))

    # 反空转护栏：用例被清空 / 表被砍短 / 上游抠不出编排名，都不许以「0 违规」通过。
    # ⚠ 第二十八轮起，**所有可调的数**都走 `_two_layer_floor`（现算 + 历史记录两层）——
    #   理由见那个函数的文档注释：上一轮这些门槛全是规则文件里的自报数，
    #   复核把三个数调松就把闸变绿了，而且一声不响。
    c = res.get("counts") or {}
    rec = rule.get("recorded")
    if not isinstance(rec, dict):
        bad.append(("-", "配置", "recorded",
                    "规则里没有 `recorded` 块 —— 两层判法的历史层整个不见了。"
                    "**不许把它删掉当作「简化」**：只剩现算层的话，删表项 / 删用例时"
                    "现算值跟着一起降，判据永远自洽（这正是本轮堵的作弊路径 B）"))
        rec = {}

    # ① 三张表的表长：现算层 = 表**现在**有多少条（runner 从发布中的表上现数的），
    #    历史层 = 规则里登记的上次确认值。两层合起来 ⇒ 记录值必须**恰好**等于现值：
    #    删条目 → 低于历史值红；调松记录值 / 加条目忘了记 → 低于现算值红。
    #    ⚠ 表长的记录值另有一份**独立备份**：自检面板那条断言（R-selfcheck-d-liveness）
    #      的 `min_regex_entries` 记的是 PREFIXED + PATTERNS 进面板的条目数（19+28=47），
    #      跑的是另一份执行体、读的是另一条汇总路径。要把表砍短而不红，
    #      得**同时**改两条规则里的两个数 —— 这是有意留的交叉印证。
    for field, label in (("prefixed_size", "PREFIXED"),
                         ("patterns_size", "PATTERNS"),
                         ("np_size", "NOTIFICATION_PATTERNS")):
        n = c.get(field, 0)
        _two_layer_floor(bad, f"{label} 表长", n, rec.get(field), live=n, key=field,
                         grow_hint=f"往 {label} 里加了条目（那就得同时补一条正例，"
                                   f"并把 recorded.{field} 记成新值）")

    # ② 四段用例的条数：现算层 = **规则里现在真写着几条**，历史层 = 登记值。
    #    另外把「规则里写了几条」与「runner 真跑了几条」对一遍 —— 两个数不一样，
    #    说明有一段用例被 runner 静默吞了（那是空转，不是通过）。
    for field, label in (("negative", "反例"), ("positive", "正例"),
                         ("notify_negative", "通知反例"), ("notify_positive", "通知正例")):
        ran = c.get(field, 0)
        in_rule = len(rule.get(field) or [])
        # 判了几条规矩：**runner 真跑了几条用例**就记几条（不是规则里写了几条）——
        # 用例被静默吞掉时这个数跟着掉，与「表有多长」是两回事。
        _unit(*(f"case:{field}:{i}" for i in range(ran)))
        if ran != in_rule:
            bad.append(("-", "配置", field,
                        f"{label}：规则里写着 {in_rule} 条，runner 只跑了 {ran} 条 —— "
                        f"有一段用例被静默吞了，这不是「通过」"))
        _two_layer_floor(bad, f"{label}条数", ran, rec.get(field), live=in_rule, key=field,
                         grow_hint=f"补了{label}（那是好事，把 recorded.{field} 记成新值即可）")

    # ③ 正例条数还有一道**结构下限**：三张表的每一条都得有正例（覆盖归因那一段管漏没漏），
    #    所以正例数**至少**得有「PREFIXED + PATTERNS」这么多条、通知正例至少 NP 这么多条。
    #    这一层是从**表长现算**出来的，规则文件里调不动 —— 就算有人把上面的记录值全改小，
    #    它也拦在这里。
    struct = c.get("prefixed_size", 0) + c.get("patterns_size", 0)
    if c.get("positive", 0) < struct:
        bad.append(("-", "配置", "positive",
                    f"正例只有 {c.get('positive', 0)} 条，而 PREFIXED + PATTERNS 现在共 {struct} 条 ——"
                    f"每条表项都得有正例，正例数**结构上**不可能少于表长之和"))
    if c.get("notify_positive", 0) < c.get("np_size", 0):
        bad.append(("-", "配置", "notify_positive",
                    f"通知正例只有 {c.get('notify_positive', 0)} 条，而 NOTIFICATION_PATTERNS "
                    f"现在有 {c.get('np_size', 0)} 条 —— 每条都得有正例"))

    # ③b 反例侧的结构下限（第二十九轮）。**恒等式，没有可调的旋钮**：
    #    runner 的 §④b 按「一条表项一条专属近似反例」做**完美匹配**，
    #    所以「参与匹配的表项数」必须恰好等于表长之和、「匹配上的条数」必须等于表项数。
    #    这一段在 Python 侧再核一遍，防的是**runner 里那一段被整块删掉**：
    #    删了的话 counts 里这几个键就没了（取到 0），下面当场红 ——
    #    而不是「少判一段还报绿」（那是空转形态 (e) 的又一次现身）。
    #    ⚠ 反例的**条数**下限不写在这里：完美匹配已经蕴含「反例数 ≥ 表项数」，
    #      再补一个数字门槛就是又开一条「调松阈值」的路（作弊路径 B）。
    for got_k, want_v, label in (
            ("neg_entries", c.get("prefixed_size", 0) + c.get("patterns_size", 0),
             "translateText 通道参与反例匹配的表项数（PREFIXED + PATTERNS）"),
            ("np_neg_entries", c.get("np_size", 0),
             "translateNotification 通道参与反例匹配的表项数（NOTIFICATION_PATTERNS）")):
        if c.get(got_k, 0) != want_v:
            bad.append(("-", "配置", got_k,
                        f"{label}：runner 报 {c.get(got_k, 0)}，按表长应为 {want_v} —— "
                        f"runner 里的反例结构护栏（§④b）被删了 / 没跑到 / 只喂了一张表。"
                        f"**这不是「通过」**"))
    for got_k, tot_k, label in (("neg_matched", "neg_entries", "translateText 通道"),
                                ("np_neg_matched", "np_neg_entries", "通知通道")):
        if c.get(got_k, 0) != c.get(tot_k, 0):
            bad.append(("-", "配置", got_k,
                        f"{label}的反例专属归因 {c.get(got_k, 0)}/{c.get(tot_k, 0)} —— "
                        f"有表项没有属于它自己的近似反例（runner 里已逐条点名）"))
    for got_k, tot_k, label in (("neg_probes", "neg_entries", "translateText 通道"),
                                ("np_neg_probes", "np_neg_entries", "通知通道")):
        if c.get(got_k, 0) != c.get(tot_k, 0):
            bad.append(("-", "配置", got_k,
                        f"{label}的机械近似探针只跑了 {c.get(got_k, 0)} 条，表项有 "
                        f"{c.get(tot_k, 0)} 条 —— 有表项抠不出字面骨架（runner 已点名），"
                        f"那种条目宽到没有锚点，人必须看一眼"))

    n_neg, n_pos = c.get("negative", 0), c.get("positive", 0)
    arr = rule.get("arrangements") or {}
    detail = (f"node 跑 translateText {n_neg} 反 / {n_pos} 正"
              f"（覆盖 PREFIXED {c.get('prefixed_covered', 0)}/{c.get('prefixed_size', 0)}"
              f" · 覆盖 PATTERNS {c.get('patterns_covered', 0)}/{c.get('patterns_size', 0)}"
              f" · 反例专属归因 {c.get('neg_matched', 0)}/{c.get('neg_entries', 0)}"
              f" · 机械近似探针 {c.get('neg_probes', 0)} 条）"
              f"；translateNotification {c.get('notify_negative', 0)} 反 /"
              f" {c.get('notify_positive', 0)} 正"
              f"（覆盖 NOTIFICATION_PATTERNS {c.get('np_covered', 0)}/{c.get('np_size', 0)}"
              f"，反例专属归因 {c.get('np_neg_matched', 0)}/{c.get('np_neg_entries', 0)}"
              f"，机械近似探针 {c.get('np_neg_probes', 0)} 条"
              f"，空白折叠回退命中 {c.get('np_flat_fallback', 0)}）")
    if arr:
        # 编排那一段的规矩：每条通道 / 每个前缀各算一条。
        _unit(*(f"arr:ch:{x}" for x in (arr.get("channels") or [])))
        _unit(*(f"arr:pfx:{x}" for x in (arr.get("prefixes") or [])))
        _unit("arr:labels-floor")
        # ⚠ 编排名条数走**三层**：现算（上游这次抠出多少）+ 历史记录（219）+ 死下限（210）。
        #   第三层是给形态 (h) 留的：喂输入的那一半坏掉时，现算值会集体塌方，
        #   跟着现算值走的门槛也就一起塌了，只有**不联动的死数**还站得住。
        #   （第二十七轮实测：上游 label: 行缩进 +1 → 编排名 219→196 静默下降。）
        #   ⚠ 另有一道**不含阈值**的同源检查在 runner 里：被判文件 `ARRANGEMENT_LEAVES`
        #     的每个键都必须能在上游现抠的编排名里找到 —— 那一道调不松，见 runner ⑥ 段。
        _two_layer_floor(bad, "上游现抠的编排名", c.get("labels", 0),
                         arr.get("labels_recorded"), live=c.get("labels", 0),
                         hard=arr.get("min_labels"), key="arrangements",
                         grow_hint="上游新增了编排名（人来看一眼这些新名字该不该进表，"
                                   "再把 labels_recorded 记成新值）")
        detail += (f"；上游 {c.get('soundscapes', 0)} 个音景现抠 {c.get('labels', 0)} 个编排名"
                   f"（解析不出 {c.get('unresolved', 0)} 个）"
                   f" × {len(arr.get('channels', []))} 通道 = {c.get('pairs', 0)} 条"
                   f"（翻得动 {c.get('translated', 0)}，登记整串留英 {c.get('stuck_labels', 0)} 个）")
    return bad, detail


# ============ 「断言把它当运行时输入读、却没入库」整类（第二十八轮 ③）
#
# 为什么非有这一条不可：**同型缺陷这是第四次**
# --------------------------------------------
#   ① `EXCLUSIONS.json` originally 在 `4-临时脚本/…/findings/` 下（未入库的临时目录），
#      被 `exclusions_closed` 当运行时输入读；
#   ② 第十五轮的 findings 同样；
#   ③ `5-其他内容/english-baseline/`（英文基准）在第二十轮之前不在库里；
#   ④ 第二十八轮复核发现：`3-常用脚本/qa/translate_cases_runner.mjs` 至今 `??` ——
#      同目录另外 39 个脚本全部 tracked，**只有它**在外面，而主闸对它是**硬依赖**
#      （找不到判「没跑成」＝失败）。在一份 clean checkout 上，主闸直接判失败。
#
# 四次都是同一个形状：**判据依赖的东西留在工作区里没入库**，
# 在本机跑得好好的，换台机器 / clean checkout 就整条塌掉。
# ⇒ 与其每轮靠人再想起来一次，不如让「断言依赖了未入库文件」**本身变成一条判据**。
#
# 怎么做到「不靠人列清单」
# ------------------------
# 清单**从规则集自身推**：把每条断言里那些「会被当路径打开」的字段（`src` / `files` /
# `glossary` / `exclusions` / `doc` / `scanner` / `panel` / `tables_src`，以及各 kind 的
# **默认 runner 名**）逐条解析成绝对路径 —— 人新加一条断言时，它的输入自动进清单。
# 另有两道兜底：
#   · `sweep`：整个 `3-常用脚本/qa/` 目录（判据执行体的家）一个不落地扫，
#     哪天又有人往里放一个没入库的执行体，不必等它被某条规则引用就会响；
#   · `must_include`：几条**点名**必须出现在推导结果里的路径。推导器要是哪天不认识
#     某个字段名了（改字段名 / 加新 kind），清单会**静默变短**而判据照样报「全部入库」——
#     那正是空转形态 (d)「豁免表已空」的镜像。点名清单一旦对不上，当场红。
#
# ⚠ 判据边界：`meta.lang_sources` 指的是**上游安装目录**（`%LOCALAPPDATA%/FoundryVTT/Data/…`），
#   那是别人的仓、按设计不进我们的库，**不列入本条**（列了就是逼人做不可能的事）。
#   上游那一侧的「输入没了」由各断言自己的「没跑成」分支管。
# ⚠ 「入库」按**拥有这个文件的那个 git 仓**判：两个汉化插件目录各自是**独立仓**，
#   在外层仓里它们是 ignored 的 —— 拿外层仓去问「你跟踪它吗」会得到一个**假的红**。


def _git_repo_of(path):
    """向上找到拥有这个路径的 git 仓（工作区根），找不到返回 None。"""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


# 各 kind 的**默认执行体**：规则里通常不写 `runner`，可它确确实实是运行时输入。
# 第四次同型缺陷（`translate_cases_runner.mjs` 未入库）正好就藏在这个「默认值」里。
_KIND_DEFAULT_RUNNER = {
    "translate_cases": "translate_cases_runner.mjs",
    "panel_liveness": "selfcheck_panel_runner.mjs",
}


def _collect_rule_inputs(rules, root, ctx, here=None):
    """从**规则集自身**推出「断言运行时会去打开的文件」→ {绝对路径: [谁要读它]}。"""
    here = here or HERE
    out = {}

    def add(p, why):
        p = os.path.normpath(p)
        out.setdefault(p, []).append(why)

    def repo_dir(name):
        try:
            return _twin_repo_dir(name, ctx)
        except KeyError:
            return None

    for r in rules.get("assertions", []):
        rid, kind = r.get("id", "?"), r.get("kind", "?")
        repo = r.get("repo")
        for field in ("src", "panel", "tables_src"):
            rel = r.get(field)
            if isinstance(rel, str):
                d = repo_dir(r.get("tables_repo", repo) if field == "tables_src" else repo)
                if d:
                    add(os.path.join(d, *rel.split("/")), f"{rid}.{field}")
        for field in ("glossary", "exclusions", "doc"):
            rel = r.get(field)
            if isinstance(rel, str):
                add(rel if os.path.isabs(rel) else os.path.join(root, rel), f"{rid}.{field}")
        if isinstance(r.get("scanner"), str):
            add(os.path.join(here, r["scanner"]), f"{rid}.scanner")
        # `pack` —— 第二十九轮补。`leaf_literal` 的 `pack` 指的是
        # `<仓>/compendium/{en,cn}/<pack>`（`a_leaf_literal` 里按 `ctx.all_pairs` 过滤的就是它），
        # 而推导器此前**不认识这个字段**：`R-tridents-point-falsepositive` 的
        # `ember.crucible-adventure.json` 与 `R-arcturian-actor-card` 的两份都不在清单里。
        # 今天这 4 份都在库里，是**潜在**缺口不是现网缺口 —— 但「推导器不认识某个字段」
        # 正是 `must_include` 想防的那一类，所以补进来，别等它变成现网缺口。
        packs = r.get("pack")
        if packs:
            packs = packs if isinstance(packs, list) else [packs]
            names = r.get("scope") or ([repo] if repo else list(REPOS))
            for pk in packs:
                if not isinstance(pk, str):
                    continue
                hit = False
                for nm in names:
                    d = repo_dir(nm)
                    if not d:
                        continue
                    for layer in ("en", "cn"):
                        p = os.path.join(d, "compendium", layer, pk)
                        if os.path.exists(p):
                            add(p, f"{rid}.pack（{nm}/compendium/{layer}）")
                            hit = True
                if not hit:
                    # 一个仓都找不到时**照样登记**，让它以「断言要读的文件不在」报出来 ——
                    # 静默跳过就等于「找不到 ⇒ 当作没这回事」，那是空转形态 (e)。
                    d = repo_dir(names[0]) if names else None
                    if d:
                        add(os.path.join(d, "compendium", "cn", pk),
                            f"{rid}.pack（{names[0]}，一个仓都没找到）")
        runner = r.get("runner") or _KIND_DEFAULT_RUNNER.get(kind)
        if runner:
            add(os.path.join(here, runner), f"{rid}.runner（{kind} 的执行体）")
        files = r.get("files")
        if isinstance(files, list):
            for f in files:
                if isinstance(f, dict) and f.get("path"):
                    d = repo_dir(f.get("repo", repo))
                    if d:
                        add(os.path.join(d, *f["path"].split("/")), f"{rid}.files[]")
        elif isinstance(files, dict):
            for layer, rel in files.items():
                if isinstance(rel, str):
                    add(rel if os.path.isabs(rel) else os.path.join(root, rel),
                        f"{rid}.files.{layer}")
    return out


_PATHY_EXT = (".json", ".mjs", ".js", ".py", ".md", ".hbs", ".txt", ".csv")
# 这些键装的是**散文**（why / 标题 / 注记），里面提到的文件名是**引用**不是输入，不参与。
_PROSE_KEYS = ("why", "title", "decision", "note", "rule", "_why")


def _pathish_strings(node, key=None):
    """规则集里所有「看着像一个文件名」的字符串值 —— 用来反过来查推导器漏了什么。"""
    if isinstance(node, str):
        s = node.strip()
        if (s.lower().endswith(_PATHY_EXT) and len(s) <= 160
                and "\n" not in s and " " not in os.path.basename(s)):
            yield s
    elif isinstance(node, dict):
        for k, v in node.items():
            if not isinstance(k, str) or k.startswith("_") or k in _PROSE_KEYS:
                continue
            yield from _pathish_strings(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from _pathish_strings(v, key)


def _pathish_bases(root, ctx, here):
    """一个「像路径的串」可能相对于哪些目录解析。"""
    bases = [root, here, os.path.join(root, "5-其他内容")]
    for d in (getattr(ctx, "repos", None) or {}).values():
        bases.append(d)
        bases.append(os.path.join(d, "compendium", "en"))
        bases.append(os.path.join(d, "compendium", "cn"))
    return bases


def a_tracked_inputs(rule, ctx):
    """断言当运行时输入读的文件，**必须都在 git 里**。见上面那段注释。

    ⚠⚠ **形态 (g) 的现场，第三十轮修**（就在隔壁 `a_ruleset_shape` 的 docstring 明写着
      这个坑之后，本函数却正踩在里面 —— 写的人知道有这一类，仍然漏了一处，
      所以这段话按实情记在这里，别当成「已经想到就等于已经堵住」）：

      本条**从磁盘重读**一份 `rule["rules"]`（写死 `5-其他内容/RESOLUTIONS.assertions.json`）
      来推「运行时输入」，而同一条断言的 `sweep` / `must_include` / `min_checked`
      读的是**传进来的 rule 对象**。半读半不读，比全读错更难看出来。后果两条（都实证过）：
        (a) 任何用 `--rules <副本>` 做的灵敏度回测，对「推导 + 反查 + git 入库」这半边
            **永远是空转** —— 副本里注入的违规根本不会被读到，探针静默变成 no-op 并跑出假绿；
        (b) 真拿 `--rules other.json` 跑时，它报的是**另一份文件**的结论。

      现在：推导的输入 = `ctx.rules`（`main()` 塞进来的、本次真正在跑的那一份），
      与 `a_ruleset_shape` 同一口径。`rule["rules"]` 降级成**登记项** ——
      「这条路径也得在 git 里」，不再是数据来源。
      两者不一致（`--rules` 指了别处）时，detail 里**明写出来**，不许静默。
    """
    import shutil
    root = getattr(ctx, "root", None) or ROOT
    git = shutil.which(rule.get("git_bin", "git"))
    if not git:
        return ([("-", "-", "git",
                  "PATH 里找不到 git —— 本条判的就是「在不在库里」，没有 git 就**没判成**，"
                  "不许静默跳过（空转形态 (e)）")], "没跑成")
    rules = getattr(ctx, "rules", None)
    if not isinstance(rules, dict) or not isinstance(rules.get("assertions"), list):
        return ([("-", "-", "ctx.rules",
                  "拿不到本次运行的规则集 —— 清单无从推起，本条**没跑成**，不是通过"
                  "（空转形态 (e)）。⚠ 不许退回「自己去磁盘上重读一份」：那是形态 (g)")],
                "没跑成")
    rules_rel = rule.get("rules", "5-其他内容/RESOLUTIONS.assertions.json")
    rules_p = os.path.join(root, *rules_rel.split("/"))
    # `rules` 字段现在只是**登记项**：这份文件也必须在 git 里。它不再是数据来源。
    ran_p = getattr(ctx, "rules_path", None)
    same = bool(ran_p) and os.path.normcase(os.path.abspath(ran_p)) == os.path.normcase(
        os.path.abspath(rules_p))
    if not os.path.isfile(rules_p):
        return ([("-", "-", rules_p,
                  f"规则里登记的 `rules` 路径不存在：{rules_p} —— "
                  f"登记项指着一个不在的文件，本条没判成")], "没跑成")

    targets = _collect_rule_inputs(rules, root, ctx, getattr(ctx, "here", None))
    targets.setdefault(os.path.normpath(rules_p), []).append("规则文件自身（登记项 `rules`）")
    # ⚠ 故意**不**把 `--rules` 指到的那份副本也塞进 git 入库清单：
    #   灵敏度回测的副本按设计是临时的、不入库，塞进去等于让每次回测都固定红一条，
    #   而「固定会红的那一条」很快就会被人当噪音略过 —— 那是把护栏变成背景音。
    #   它与登记项不一致这件事**写进 detail**（下面），让人看见，而不是让闸叫。

    bad = []
    n_swept = 0
    ignore = tuple(rule.get("sweep_ignore") or ("__pycache__",))
    for rel in rule.get("sweep") or []:
        d = os.path.join(root, *rel.split("/"))
        if not os.path.isdir(d):
            bad.append(("-", "配置", rel,
                        f"要扫的目录不在：{d} —— 扫了个空目录还报绿就是空转形态 (e)"))
            continue
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if x not in ignore]
            for fn in filenames:
                if fn.endswith(".pyc") or fn in ignore:
                    continue
                n_swept += 1
                targets.setdefault(os.path.normpath(os.path.join(dirpath, fn)), []).append(
                    f"{rel} 目录下的判据执行体（sweep）")

    # ⚠ **让「推导器漏了某类字段」这件事自己变成判据**（第二十九轮）。
    #   `must_include` 点名清单防的是「推导器静默变短」，可它靠**人去点名** ——
    #   人想不到的那个字段（这次是 `pack`）它同样点不出来。
    #   ⇒ 反过来扫：规则集里**所有看着像文件名的字符串**，凡是能在库里解析成一个
    #     真实存在的文件、却**不在推导结果里**的，就地点名。
    #   这一道不靠任何人维护：加一个新字段、字段里写一个真实文件名，它自己会响。
    #   ⚠ 边界：① 只看**非散文键**（why / title / decision 里提到的文件名是引用不是输入）；
    #     ② 只认扩展名像文件的串；③ 只在库内的几个 base 下解析（上游安装目录按设计不列入，
    #        理由与本条 why 里那段一致）。所以它**不是全覆盖**，是一道会自己长大的兜底。
    derived = {os.path.normpath(p) for p in targets}
    bases = _pathish_bases(root, ctx, getattr(ctx, "here", None) or HERE)
    n_pathish = 0
    seen_pathish = set()
    for s in _pathish_strings(rules):
        if s in seen_pathish:
            continue
        seen_pathish.add(s)
        n_pathish += 1
        for b in bases:
            p = os.path.normpath(os.path.join(b, *s.split("/")))
            if os.path.isfile(p) and p not in derived:
                bad.append(("-", "配置", s,
                            f"规则里写着一个像文件名的串，它在库里真的存在（{p}），"
                            f"**却不在推导出来的运行时输入清单里** —— 推导器不认识装它的"
                            f"那个字段？加了新 kind 没登记它的输入？"
                            f"（`pack` 就是这样漏了一轮：4 份 compendium 文件全在库里，"
                            f"是潜在缺口不是现网缺口，但下一个字段就未必了）"))
                break
    # 判了几条规矩：每个 sweep 根 + 每条点名路径 + 反查「像文件名的串」那一道。
    _unit(*(f"sweep:{r}" for r in (rule.get("sweep") or [])))
    _unit(*(f"named:{r}" for r in (rule.get("must_include") or [])))
    if n_pathish:
        _unit("pathish-reverse")
    for rel in rule.get("must_include") or []:
        p = os.path.normpath(os.path.join(root, *rel.split("/")))
        if p not in derived:
            bad.append(("-", "配置", rel,
                        f"点名必须出现在清单里的路径没被推出来：{p} —— "
                        f"推导器不认识某个字段了？加了新 kind 却没登记它的执行体？"
                        f"**清单静默变短**时，本条会一路报「全部入库」，那正是空转"))

    # 按「拥有它的那个仓」分组问 git（两个插件目录各自是独立仓）。
    by_repo, n_checked = {}, 0
    for p, whys in sorted(targets.items()):
        if not os.path.exists(p):
            bad.append(("-", "输入", os.path.relpath(p, root),
                        f"断言要读的文件不在：{p}（{'、'.join(whys[:3])}）"))
            continue
        repo = _git_repo_of(p)
        if repo is None:
            bad.append(("-", "输入", os.path.relpath(p, root),
                        f"不在任何 git 仓里（{'、'.join(whys[:3])}）—— "
                        f"换台机器就没有这份输入了"))
            continue
        n_checked += 1
        by_repo.setdefault(repo, []).append((p, whys))

    for repo, items in by_repo.items():
        rels = [os.path.relpath(p, repo).replace("\\", "/") for p, _ in items]
        proc = subprocess.run([git, "-C", repo, "ls-files", "-z", "--error-unmatch", "--"] + rels,
                              capture_output=True)
        seen = {x.decode("utf-8", "replace") for x in proc.stdout.split(b"\0") if x}
        for (p, whys), rel in zip(items, rels):
            if rel not in seen:
                bad.append(("-", "未入库", os.path.relpath(p, root),
                            f"被 {'、'.join(whys[:3])} 当运行时输入读，却**没进 git**"
                            f"（仓：{os.path.basename(repo)}）—— clean checkout 上这条断言"
                            f"直接判「没跑成」。`git add` 它，别等下一轮再发现一次"))

    floor = rule.get("min_checked", 1)
    if n_checked < floor:
        bad.append(("-", "配置", "-",
                    f"只查了 {n_checked} 个文件（要求 ≥{floor}）—— 清单是空的，这条断言在空转"))
    where = ("推导用的是**本次运行的那份规则集**（ctx.rules）"
             if same or not ran_p else
             f"⚠ 本次跑的是 `--rules {ran_p}`，与规则里登记的 `{rules_rel}` **不是同一份**；"
             f"推导用的是**本次跑的这一份**（形态 (g) 的修法），"
             f"登记的那份只作为「也得在 git 里」的入库项参与检查")
    return bad, (f"从 {len(rules.get('assertions', []))} 条规则推出 {len(targets)} 个运行时输入"
                 f"（含 sweep 扫到的 {n_swept} 个），落在 {len(by_repo)} 个 git 仓里，"
                 f"逐个问过 `git ls-files --error-unmatch`；"
                 f"另反查了规则里 {n_pathish} 个「像文件名」的串有没有漏在清单外。{where}")


# ================================ 「规则集自身的形状」整类（第二十九轮，作弊路径 C）
#
# 三条作弊路径，本项目至今堵过的全部（**名字在这里定义齐，别再让人去别处找 A**）
# ------------------------------------------------------------------------
#   **A ＝ 删／摘用例。** 判据没坏、门槛没动，只是把要判的东西拿走。
#     ▸ 第二十七轮堵：三张表 78 条**逐条改坏**，两种改法各独立跑一遍，**156/156 全红**；
#       正例那道「每条表项都必须有正例」是从表长现算的恒等式，摘用例就掉覆盖。
#   **B ＝ 调松阈值数字。** 判据在跑、也真的在判，但门槛低到判不着。
#     ▸ 第二十八轮堵：`recorded` 块里 7 个数**逐个调松** + 整块删 + 编排名两道，
#       **10 种单点动作全部当场红**（见 `_two_layer_floor`：现算层 + 历史层，缺一不可）。
#       旧的 `coverage.min_*` 通道就此成为惰性字段、没有后门。
#   **C ＝ 删掉装数字的那个容器，或删掉整条断言。** ← **本轮堵的就是这条**
#     ▸ 第二十八轮复核当场证明洞被挪到了容器层：下面每一条都是**单点**改动、
#       改完主闸仍 65/0/0 全绿 ——
#         删 `arrangements` 整块（438 条编排名对拍整段消失）· `channels` 砍成一个（护栏减半）·
#         面板闸 `min`/`max` 整块删掉（覆盖侧与 miss 侧记录值全消失）·
#         `tracked.min_checked` → 0 · `tracked.must_include` 清空 ·
#         `arrangements.min_labels` 单独改小 ·
#         **删掉整条断言（65→64）—— 全项目没有任何东西数一数还剩几条规则**。
#     ⇒ A 和 B 都被堵住之后，洞就往上挪一层：**不改数字，改装数字的容器。**
#
# 本条怎么做到「守卫不与被守卫的东西同生共死」
# --------------------------------------------
# 要害在于**记录值绝不能住在它所守护的那个容器里** —— 住在里面的话，
# 删掉容器就把守卫一起删了，那是把洞原地又挖了一遍（第二十八轮 `recorded` 块被整块删掉
# 那一次之所以能红，靠的正是 `_two_layer_floor` 里「没有 recorded 就报违规」那一支
# 写在**本文件**里，不在规则文件里）。
# ⇒ 记录值放在**本文件的模块级常量 `RULESET_SHAPE`** 里。
#
# ⚠ **绕过它需要改哪几处（照实写，不许写成「不可能被绕过」）：**
#   ① 改小 `RULESET_SHAPE` 里的数 —— 那是**改判据本身**，diff 落在
#      `assert_resolutions.py` 上，写着「我把地板改低了」，是**自证的**；
#   ② 删掉规则文件里那条 `ruleset_shape` 断言 —— `main()` 里有一道
#      **不依赖规则文件**的存在性检查（`_require_ruleset_shape`），
#      规则集里一条都没有就直接判失败，所以只删规则不够；
#   ③ 连 `main()` 里那道存在性检查一起删。
#   **①②③ 同时改，仍然能过。** 本条保证的是：**任何一处单独的容器层删除都会当场红**，
#   而绕过它必须在判据文件上留下一串互相印证的显式改动。
#   （与 `_two_layer_floor` 那段文档注释同一个口径 —— 判据能做到的边界就在这里。）
#
# ⚠ 判据边界：本条只判**形状**（有没有、够不够长、点名的在不在），
#   **不判内容对不对**（`min` 里的数记得准不准、用例写得好不好，那是各条断言自己的事）。
RULESET_SHAPE = {
    # —— 断言条数与 kind 数：**这是最要紧的一条**。在此之前，删掉整条断言
    #    （65 → 64）全项目没有任何东西数一数还剩几条。
    # 第三十三轮 B：66 → 67 / 22 → 23（新增 `tag_parity` 一条，见它的执行体注释；
    # §0.1 收官后唯一批准的新判据层，理由是「它会咬到诚实的维护者」那一支例外）
    "min_assertions": 67,
    "min_kinds": 23,
    # —— 各 kind 必须存在的条数（防「把某一类整类删光」，也防「某一类被削掉大半」）
    #
    # ⚠ 第三十轮：这张表**曾经只有 4 项**（translate_cases / panel_liveness /
    #   tracked_inputs / ruleset_shape），22 种 kind 里剩下的 18 种一条也没点名。
    #   后果有二：
    #     ① 把 `cn_absent` 从 15 条砍到 1 条，`min_assertions` 会响、`min_by_kind` 不会 ——
    #        少的是「哪一类」这条信息，而这正是「整类被掏空」最需要说清楚的一句话；
    #     ② 更要命的是 **H1（`kind` 拼错一个字母）**：`min_kinds` 数的是 distinct kind，
    #        拼错出来的 `cn_absentt` **自己还算一种**，反而帮着满足 22 这个地板 ——
    #        护栏替作弊者数了一票。逐 kind 点名之后，`cn_absent` 少一条当场红。
    #   现在按发布中那份规则集的**实际条数**逐类登记（数字即当时的真值，改小 = 改判据本身）。
    "min_by_kind": {
        "anchor_ids": 1,
        "block_aligned_gate": 2,
        "block_sense_gate": 1,
        "cn_absent": 15,
        "distinct_terms": 9,
        "enricher_slot_gate": 7,
        "enricher_text_coverage": 1,
        "exclusions_closed": 1,
        "glossary_value": 2,
        "lang_parity": 1,
        "leaf_literal": 2,
        "no_bilingual_tail": 1,
        "panel_liveness": 1,
        "ruleset_shape": 1,
        "sense_gated": 1,
        "source_literal": 1,
        "tag_parity": 1,
        "term_domains": 4,
        "term_gated": 11,
        "tracked_inputs": 1,
        "translate_cases": 1,
        "twin_files": 1,
        "version_matrix": 1,
    },
    # —— 容器形状：每条该 kind 的规则必须有这些**非空**的块，块里必须有这些键
    "containers": {
        "translate_cases": {
            "recorded": ["prefixed_size", "patterns_size", "np_size",
                         "negative", "positive", "notify_negative", "notify_positive"],
            "arrangements": ["channels", "prefixes", "min_labels", "labels_recorded",
                             "expect_untranslated"],
        },
        "panel_liveness": {
            # ⚠ 第三十二轮 V18 补进来的四个覆盖侧分母（registered* / wrappedTables /
            #   regexTables）与两个没核侧分母（unchecked*）：它们是第三十一轮为回答 V14
            #   造出来的数，此前 **runner 不转发、min/max 里一个都没有、全项目零引用**——
            #   「算了、印了、没人判」。点名在这里，值住在规则集，死下限在 hard_floors。
            # ⚠ `min` 的键数现在是 11：`_PAY_RECURSE_MAX = 12` 一旦被越过，
            #   `_derive_payload_floors` 就只记总长、不再逐键钉死 —— 再往里加键前先抬那个数。
            "min": ["checkedDistinct", "rawChecked", "fetchOk", "tableRows",
                    "tablesFedIn", "tplFiles", "tableRegexEntries",
                    "registeredRaw", "registeredDistinct", "wrappedTables", "regexTables"],
            "max": ["missDistinct", "rawMiss", "fetchFail",
                    "uncheckedRaw", "uncheckedDistinct"],
            "fakes": [],                      # 只要求非空（假阴性对照，键名是自造串）
        },
        "tracked_inputs": {
            "sweep": [],
            "must_include": [],
        },
    },
    # —— 非空列表的最短长度（`channels` 砍成一个 = 同一道护栏减半）
    "min_len": {
        ("translate_cases", "arrangements", "channels"): 2,
        ("panel_liveness", "fakes"): 6,
        ("tracked_inputs", "must_include"): 4,
        ("tracked_inputs", "sweep"): 1,
    },
    # —— 死下限：这些**强度参数**本身不许被改小（§3.7.2「强度参数与判据同层 = 没有强度」）
    "hard_floors": {
        ("translate_cases", "arrangements", "min_labels"): 210,
        ("translate_cases", "arrangements", "labels_recorded"): 219,
        ("panel_liveness", "min", "checkedDistinct"): 719,
        ("panel_liveness", "min", "rawChecked"): 1273,
        ("panel_liveness", "min", "tableRegexEntries"): 47,
        # 第三十二轮 V18：分母也是强度参数 —— 把 `registeredRaw` 调小 = 把「一共有多少键」
        # 这个分母做小，`已核/登记` 的比例就凭空变好看了，而被核键数一个没涨。
        ("panel_liveness", "min", "registeredRaw"): 1459,
        ("panel_liveness", "min", "registeredDistinct"): 887,
        ("tracked_inputs", "min_checked"): 40,
    },
    # —— miss 侧是「不得高于」，改大就是放水
    "hard_ceilings": {
        ("panel_liveness", "max", "missDistinct"): 4,
        ("panel_liveness", "max", "rawMiss"): 7,
        ("panel_liveness", "max", "fetchFail"): 3,
        # 「按设计没核」的条数同样只许降：调大它 = 允许更多键悄悄退出被核集合
        ("panel_liveness", "max", "uncheckedRaw"): 186,
        ("panel_liveness", "max", "uncheckedDistinct"): 168,
    },
    # —— `must_include` 的点名地板（第二十八轮那次同型缺陷就藏在「默认 runner 名」里）
    "tracked_must_include": [
        "3-常用脚本/qa/translate_cases_runner.mjs",
        "3-常用脚本/qa/selfcheck_panel_runner.mjs",
        "5-其他内容/RESOLUTIONS.assertions.json",
        # 第二十九轮补：`pack` 字段推出来的那一类（推导器此前不认识这个字段）
        "1-Ember汉化插件/compendium/cn/ember.crucible-adventure.json",
    ],
}


# ================================ 登记在册的断言：**id → kind 全集**（第三十轮）
#
# 为什么非有不可 —— 三种**主闸一声不吭**的绕法，复核当场跑通（下称 H1 / H2 / H3）：
#   **H1｜改 1 个字母。** 把某条断言的 `kind` 拼成 `cn_absentt`。
#     旧 `main()` 打一行 `?? R-xxx: 未知断言类型`、`skipped += 1`，而 `return 1 if failed else 0`
#     —— **跳过从来不判失败**。结果 `65/0/跳过 1`、rc=0，那条断言彻底停跑。
#   **H2｜改 2 处、零痕迹。** 删掉一条真断言，再把另一条复制一份换个 id 当填充。
#     `min_assertions` 只数 `len(A)`，66 还是 66；`min_kinds` 数 distinct kind，也还是 22。
#     结果 `66/0/0`、rc=0，**输出里连一行异常都没有**。
#   **H3｜删一条，塞一个 `{"kind": "nope"}` 空壳。** 同 H1，`65/0/跳过 1`、rc=0。
#
# 三道修法各挡一段，缺一不可：
#   ① `_exit_code`：跳过 ≠ 通过；且「认不出 kind」直接判失败（挡 H1/H3 的**停跑**那一半）；
#   ② **本表**：登记过的 id 必须还在、且 kind 必须还是原来那个（挡 H2 的**删一条补一条**，
#      也挡 H1 的**改 kind**——改成任何别的 kind 都会在这里对不上，哪怕改成一个合法 kind）；
#   ③ `min_by_kind` 覆盖全部 22 种（挡「某一类被掏空」，并夺回 H1 里 `min_kinds` 那一票）。
#
# ⚠ **它为什么住在这里**：第二十九轮的教训是「守卫不能和被守护的东西住在同一个容器里」。
#   这张清单守的是**规则文件里还剩哪些断言**，所以它必须住在**规则文件之外** ——
#   就是本文件的模块级常量，和 `RULESET_SHAPE` 同一层。放进规则文件里的话，
#   删断言的同一刀顺手把它那一行也删了，等于没有。
#
# ⚠ **合法的增删怎么办**（这不是「不许动」，是「动了要自证」）：
#   · 新增断言：不必登记也能跑（本表只判「登记过的还在不在」，不判「有没有多」），
#     但 `min_assertions` / `min_by_kind` 会要求你同步上调 —— 那一处 diff 落在本文件上。
#   · 删除 / 改名 / 改 kind：必须同时改本表，diff 写着「我把 R-xxx 摘了」，是**自证的**。
#   这正是与 H2 的区别：H2 的两处改动都在规则文件里、互相抵消、看不出来；
#   合法删除的那处改动在**判据文件**里，一眼可见。
REGISTERED_ASSERTIONS = {
    'R-warlock-sorcerer': 'term_gated',
    'R-electricity-lightning': 'term_gated',
    'R-round-turn': 'distinct_terms',
    'R-tier-rank-level': 'distinct_terms',
    'R-region-area-map': 'distinct_terms',
    'R-tridents-point-falsepositive': 'leaf_literal',
    'R-soulbound-progression': 'cn_absent',
    'R-highgate': 'cn_absent',
    'R-otherhood': 'cn_absent',
    'R-marlstone': 'cn_absent',
    'R-evidence-value': 'cn_absent',
    'R-presence-wisdom': 'cn_absent',
    'R-lang-parity': 'lang_parity',
    'R-anchor-ids': 'anchor_ids',
    'R-bare-cn-fields': 'no_bilingual_tail',
    'R-exclusions-closed': 'exclusions_closed',
    'R-glossary-not-laundering': 'glossary_value',
    'R-catwalk': 'term_gated',
    'R-drakeling': 'term_gated',
    'R-afflicted': 'cn_absent',
    'R-inkaro-pool': 'cn_absent',
    'R-clipper-name': 'cn_absent',
    'R-boardwalk': 'term_gated',
    'R-version-matrix': 'version_matrix',
    'R-junction-wheel': 'term_gated',
    'R-shard-god': 'distinct_terms',
    'R-shard-god-blocks': 'block_aligned_gate',
    'R-rank-sense-compendium': 'sense_gated',
    'R-rank-sense-blocks': 'block_sense_gate',
    'R-dives-mine': 'term_gated',
    'R-arcturel-vs-arcturian': 'distinct_terms',
    'R-arcturel-arcturian-blocks': 'block_aligned_gate',
    'R-kessia-vs-kessian': 'distinct_terms',
    'R-ordain-vs-ordani': 'distinct_terms',
    'R-fortitude-scope': 'glossary_value',
    'R-token-foundry-ui': 'cn_absent',
    'R-aura-three-way': 'term_domains',
    'R-cyclonic': 'distinct_terms',
    'R-hex-tile': 'cn_absent',
    'R-lantyr': 'term_gated',
    'R-corla-cora': 'distinct_terms',
    'R-orb-destroyed': 'term_gated',
    'R-obsidian-antiquary': 'term_gated',
    'R-temple-lunarium': 'term_gated',
    'R-point-cape': 'cn_absent',
    'R-arcturian-split': 'term_domains',
    'R-arcturian-actor-card': 'leaf_literal',
    'R-cosmology-cosmos': 'term_domains',
    'R-hex-homograph': 'term_domains',
    'R-moon-hollow': 'cn_absent',
    'R-moon-charred': 'cn_absent',
    'R-moon-tempest': 'cn_absent',
    'R-arcturel-arcturian-labels': 'enricher_slot_gate',
    'R-dives-labels': 'enricher_slot_gate',
    'R-kessia-labels': 'enricher_slot_gate',
    'R-ordain-labels': 'enricher_slot_gate',
    'R-shard-god-labels': 'enricher_slot_gate',
    'R-moon-epithet-labels': 'enricher_slot_gate',
    'R-token-rank-labels': 'enricher_slot_gate',
    'R-readaloud-coverage': 'enricher_text_coverage',
    'R-selfcheck-twin': 'twin_files',
    'R-patterns-translate-cases': 'translate_cases',
    'R-selfcheck-d-section-name': 'source_literal',
    'R-selfcheck-d-liveness': 'panel_liveness',
    'R-assertion-inputs-tracked': 'tracked_inputs',
    'R-ruleset-shape': 'ruleset_shape',
    'R-html-tag-parity': 'tag_parity',
}


# ============ 载荷地板：**id 和 kind 被钉死了，内容没有**（第三十一轮 L5-①）
#
# 第三十轮之后，一条断言的 `id` 与 `kind` 都进了 `REGISTERED_ASSERTIONS`，
# 条数与每类的条数都进了 `min_assertions` / `min_by_kind`。复核在下面一格找到了第五层：
# **「每条还剩多少内容」没有任何东西数。** 实测（单点动作，全部 66/0/0、rc=0）：
#   · `R-hex-tile` **整条**换成 `{"id":"R-hex-tile","kind":"cn_absent","cn":["绝不可能XYZ"]}`
#     —— id 对、kind 对、条数对、类数对，上面三层**全部满意**，而 `en` 英文闸和
#     `min_hits`（那条断言唯一的反空转护栏）连同禁用词一起没了。
#   · **15 条 `cn_absent` 的禁用词表一次全部清空** —— 全库零容忍的那 16 个词一个不剩。
#     （第三十二轮订正：这里原写 18，实测 16 —— 12 条 `cn` 是单串 × 1 +
#      `R-token-foundry-ui` 2 + `R-hex-tile` 1 + `R-point-cape` 1 = 16。历史结论不动，只订正数。）
#   · `R-bare-cn-fields.fields=[]` —— detail 从「检查 1443 个叶」塌成「检查 0 个叶」。
#   · `R-lang-parity.packages={}` —— detail 变成空字符串。
#   · 4 条 `term_gated` 的 `cn_forbidden=[]` —— detail **一个字节都不变**。
# 根因：`RULESET_SHAPE` 的 `containers`/`min_len`/`hard_floors` 一共只覆盖 **3 个 kind**
# （translate_cases / panel_liveness / tracked_inputs）。第三十轮把 `min_by_kind` 从 4 类扩到
# 22 类，解决的是「每类还剩**几条**」；「每条还剩多少**内容**」这一层停在旧版本。
#
# 本表按 **id** 逐条登记载荷键（不按 kind：同一类里各条的载荷差得很远 ——
# 15 条 `cn_absent` 只有 3 条带 `en`/`min_hits`，按 kind 记要么漏 12 条要么误伤 12 条）。
# 记法 `{键: (类型, 值)}`，`键` 可以带一层 `块.子键`：
#   ("list", n) 列表/字典**至少** n 项 · ("str", n) 字符串**至少** n 字
#   ("ge", v) 数不得小于 v · ("le", v) 数不得大于 v · ("eq", v) 必须原样
# 数字的方向按键名定：`min_*` 是地板、`max_*` 是天花板，其余（`window`/`short_len`/
# `substr_expect_miss`/各种布尔开关）一律 `eq` —— 含义两可的参数不给「往松了调」留缝。
#
# ⚠ **登记表与规则集必须一一对应**：本表的 id 集合与 `REGISTERED_ASSERTIONS` 必须完全相同，
#   这一条自己也判（见 `_payload_floor_check`）。于是「从本表里删掉某条」不是单点动作 ——
#   还得同时从 `REGISTERED_ASSERTIONS` 里摘掉，而那一摘立刻被「登记在册的还在不在」抓住。
#
# ⚠ **地板不是天花板**：往载荷里加词、加域、加字段一律不响（长度只涨不跌），
#   所以日常收紧判据不需要动这张表；只有**删/清空**才要，而那正是要人自证的那一步。
#
# ============ ⚠⚠ 登记的**永久边界**：本表只钉长度，不钉内容（L6-②，第三十二轮）
#
# 复核实测（照抄，别再重新发现一遍）：把 19 条断言里 21 个「禁用写法」**一次全部换成
# 等长互异的乱码** —— 主闸 66/0/0，`--selftest` 当时 285/285，**两道闸一起全绿**。
# 规模：本表 322 道地板里（第三十二轮 V18 前是 316 道），
#   `str` 92 + `list` 102 = **194 道（60%）只钉长度、不钉内容**。
# 变长的替换会被前置 C（「现推 == 登记」）当场抓住，**真正静默的是等长替换**。
#
# ⚠ **项目所有者已裁定：这一条记为永久边界，不修。** 理由照抄，免得下一个接手的人
#   以为是漏掉了、又花一轮去「补」：
#   · 它是**纯攻击场景** —— 没有哪个诚实的维护者会不小心把「海门」写成「海們」；
#     而 L6-①（护栏没有反例）与 L6-③（指纹够不着 node 侧执行体）会真的咬到
#     重构与日常跑法，所以那两条修、这一条不修。
#   · 停止条件不是「再也找不到下一层」，而是「边际一轮不再产出译文缺陷 **且**
#     绕过成本高于诚实的维护者会不小心犯的水平」——这一条落在后半句之外。
# ⚠ 要真去堵它，正确的做法是**给内容配一份独立来源**（比如从词表/术语表现推期望值），
#   而不是把 194 个字面量抄进本表 —— 抄一份就等于测了个副本（本项目登记的空转形态 (c)）。
PAYLOAD_FLOORS = {
    "R-warlock-sorcerer": {"cn_required": ("str", 3), "en": ("str", 11), "min_hits": ("ge", 1)},
    "R-electricity-lightning": {"case_sensitive": ("eq", True), "cn_required": ("str", 2), "en": ("str", 15), "min_hits": ("ge", 1)},
    "R-round-turn": {"min_hits": ("ge", 14), "scan": ("list", 1), "terms": ("list", 2)},
    "R-tier-rank-level": {"min_hits": ("ge", 50), "scan": ("list", 1), "terms": ("list", 3)},
    "R-region-area-map": {"min_hits": ("ge", 7), "scan": ("list", 1), "terms": ("list", 2)},
    "R-tridents-point-falsepositive": {"must_contain": ("list", 1), "must_not_contain": ("list", 1), "pack": ("str", 29), "path": ("str", 71)},
    "R-soulbound-progression": {"cn": ("str", 4)},
    "R-highgate": {"cn": ("str", 2)},
    "R-otherhood": {"cn": ("str", 3)},
    "R-marlstone": {"cn": ("str", 4)},
    "R-evidence-value": {"cn": ("str", 3)},
    "R-presence-wisdom": {"cn": ("str", 3)},
    "R-lang-parity": {"packages": ("list", 2), "packages.crucible": ("str", 47), "packages.ember": ("str", 44)},
    "R-anchor-ids": {"min": ("ge", 900), "scope": ("list", 1)},
    "R-bare-cn-fields": {"fields": ("list", 4), "scope": ("list", 1)},
    "R-exclusions-closed": {"exclusions": ("str", 36), "long_keys": ("list", 2), "min_en_strings": ("ge", 10000), "scanner": ("str", 21), "short_len": ("eq", 40)},
    "R-glossary-not-laundering": {"entries": ("list", 31), "entries.__pins_dropped__": ("eq", 31), "files": ("list", 2), "files.base": ("str", 32), "files.product": ("str", 32)},
    "R-catwalk": {"cn_required": ("str", 2), "en": ("str", 9), "min_hits": ("ge", 1), "scope": ("list", 1)},
    "R-drakeling": {"cn_required": ("str", 3), "en": ("str", 15), "except_paths": ("list", 1), "scope": ("list", 1)},
    "R-afflicted": {"cn": ("str", 5)},
    "R-inkaro-pool": {"cn": ("str", 4)},
    "R-clipper-name": {"cn": ("str", 3)},
    "R-boardwalk": {"cn_required": ("str", 3), "en": ("str", 15), "scope": ("list", 1)},
    "R-version-matrix": {"doc": ("str", 10), "tag_prefix": ("list", 2), "tag_prefix.crucible": ("str", 0), "tag_prefix.ember": ("str", 1)},
    "R-junction-wheel": {"cn_required": ("str", 4), "en": ("str", 18), "scope": ("list", 1)},
    "R-shard-god": {"min_hits": ("ge", 400), "scan": ("list", 1), "terms": ("list", 3)},
    "R-shard-god-blocks": {"backward_classes": ("list", 1), "cn_tokens": ("list", 3), "en_tokens": ("list", 2), "leaf_gate": ("str", 13), "max_shape_mismatch": ("le", 0), "min_blocks": ("ge", 600), "min_leaves": ("ge", 450), "mode": ("str", 8)},
    "R-rank-sense-compendium": {"cn": ("str", 2), "en": ("str", 10), "forbid_when_all_common": ("eq", True), "min_cn_leaves": ("ge", 150), "min_common_leaves": ("ge", 60), "min_en_leaves": ("ge", 300), "require_en_support": ("eq", True), "sense": ("list", 2), "sense.common": ("str", 284), "sense.game": ("str", 160), "window": ("eq", 90)},
    "R-rank-sense-blocks": {"cn": ("str", 2), "except_blocks": ("list", 1), "leaf_gate": ("str", 10), "max_shape_mismatch": ("le", 0), "max_unused_exempt": ("le", 0), "min_blocks": ("ge", 500), "min_common_blocks": ("ge", 80), "min_game_blocks": ("ge", 300), "min_leaves": ("ge", 350), "occ": ("str", 10), "sense": ("list", 4), "sense.common": ("str", 312), "sense.exempt": ("str", 64), "sense.game": ("str", 160), "sense.strong_game": ("str", 49), "window": ("eq", 90)},
    "R-dives-mine": {"case_sensitive": ("eq", True), "cn_required": ("str", 2), "en": ("str", 9), "min_hits": ("ge", 100), "scope": ("list", 1)},
    "R-arcturel-vs-arcturian": {"min_hits": ("ge", 900), "scan": ("list", 1), "terms": ("list", 2)},
    "R-arcturel-arcturian-blocks": {"cn_tokens": ("list", 4), "en_tokens": ("list", 8), "except_blocks": ("list", 4), "except_paths": ("list", 1), "leaf_gate": ("str", 87), "max_shape_mismatch": ("le", 0), "max_unused_exempt": ("le", 0), "min_blocks": ("ge", 1600), "min_leaves": ("ge", 850), "mode": ("str", 8)},
    "R-kessia-vs-kessian": {"min_hits": ("ge", 110), "scan": ("list", 1), "terms": ("list", 2)},
    "R-ordain-vs-ordani": {"min_hits": ("ge", 2000), "scan": ("list", 1), "terms": ("list", 2)},
    "R-fortitude-scope": {"entries": ("list", 3), "entries.Fortitude": ("str", 2), "entries.Toughness": ("str", 2), "entries.Undeathly Fortitude": ("str", 4), "files": ("list", 2), "files.base": ("str", 32), "files.product": ("str", 32), "require_present": ("eq", False)},
    "R-token-foundry-ui": {"cn": ("list", 2), "en": ("str", 11), "except_paths": ("list", 1), "except_paths.代币": ("list", 1), "min_hits": ("ge", 250)},
    "R-aura-three-way": {"domains": ("list", 3), "min_hits": ("ge", 10), "term": ("str", 4)},
    "R-cyclonic": {"min_hits": ("ge", 20), "scan": ("list", 1), "terms": ("list", 2)},
    "R-hex-tile": {"cn": ("list", 1), "en": ("str", 12), "min_hits": ("ge", 120)},
    "R-lantyr": {"cn_forbidden": ("list", 1), "cn_required": ("str", 3), "en": ("str", 8), "min_hits": ("ge", 180)},
    "R-corla-cora": {"min_hits": ("ge", 150), "scan": ("list", 1), "terms": ("list", 2)},
    "R-orb-destroyed": {"cn_forbidden": ("list", 2), "cn_required": ("str", 5), "en": ("str", 17), "min_hits": ("ge", 3)},
    "R-obsidian-antiquary": {"cn_forbidden": ("list", 1), "cn_required": ("str", 3), "en": ("str", 22), "min_hits": ("ge", 4)},
    "R-temple-lunarium": {"cn_forbidden": ("list", 1), "cn_required": ("str", 5), "en": ("str", 19), "min_hits": ("ge", 2)},
    "R-point-cape": {"cn": ("list", 1), "en": ("str", 11), "min_hits": ("ge", 400)},
    "R-arcturian-split": {"domains": ("list", 2), "min_hits": ("ge", 120), "term": ("str", 9)},
    "R-arcturian-actor-card": {"min_leaves": ("ge", 4), "must_contain": ("list", 1), "pack": ("list", 2), "path": ("list", 2), "scope": ("list", 1)},
    "R-cosmology-cosmos": {"domains": ("list", 2), "min_hits": ("ge", 200), "term": ("str", 6)},
    "R-hex-homograph": {"domains": ("list", 2), "min_hits": ("ge", 40), "scope": ("list", 1), "term": ("str", 3)},
    "R-moon-hollow": {"cn": ("str", 4)},
    "R-moon-charred": {"cn": ("str", 4)},
    "R-moon-tempest": {"cn": ("str", 4)},
    "R-arcturel-arcturian-labels": {"case_sensitive": ("eq", False), "cn_only_leaf_fallback": ("eq", True), "cn_tokens": ("list", 2), "en_tokens": ("list", 2), "except_slots": ("list", 2), "forbid_absent": ("eq", True), "max_unpaired": ("le", 0), "min_gated": ("ge", 250), "min_leaves": ("ge", 5000), "min_slots": ("ge", 20000)},
    "R-dives-labels": {"case_sensitive": ("eq", True), "cn_tokens": ("list", 1), "en_tokens": ("list", 1), "max_unpaired": ("le", 0), "min_gated": ("ge", 24), "min_leaves": ("ge", 4500), "min_slots": ("ge", 19000), "scope": ("list", 1)},
    "R-kessia-labels": {"case_sensitive": ("eq", False), "cn_tokens": ("list", 2), "en_tokens": ("list", 2), "except_slots": ("list", 1), "max_unpaired": ("le", 0), "min_gated": ("ge", 45), "min_leaves": ("ge", 5000), "min_slots": ("ge", 20000)},
    "R-ordain-labels": {"case_sensitive": ("eq", False), "cn_tokens": ("list", 2), "en_tokens": ("list", 2), "max_unpaired": ("le", 0), "min_gated": ("ge", 300), "min_leaves": ("ge", 5000), "min_slots": ("ge", 20000)},
    "R-shard-god-labels": {"case_sensitive": ("eq", False), "cn_tokens": ("list", 3), "en_tokens": ("list", 3), "max_unpaired": ("le", 0), "min_gated": ("ge", 100), "min_leaves": ("ge", 5000), "min_slots": ("ge", 20000)},
    "R-moon-epithet-labels": {"case_sensitive": ("eq", False), "cn_tokens": ("list", 3), "en_tokens": ("list", 3), "max_unpaired": ("le", 0), "min_gated": ("ge", 10), "min_leaves": ("ge", 5000), "min_slots": ("ge", 20000)},
    "R-token-rank-labels": {"case_sensitive": ("eq", False), "cn_tokens": ("list", 2), "en_tokens": ("list", 2), "max_unpaired": ("le", 0), "min_gated": ("ge", 6), "min_leaves": ("ge", 5000), "min_slots": ("ge", 20000)},
    "R-readaloud-coverage": {"glossary": ("str", 32), "max_unpaired": ("le", 0), "min_anchor_hits": ("ge", 80), "min_anchor_slots": ("ge", 30), "min_anchor_terms": ("ge", 7000), "min_en_chars": ("ge", 60), "min_gated": ("ge", 40), "min_leaves": ("ge", 4500), "min_ratio": ("eq", 0.2), "min_sentence_frac": ("eq", 0.75), "min_slots": ("ge", 40), "slots": ("list", 1)},
    "R-selfcheck-twin": {"min_pairs": ("ge", 1), "pairs": ("list", 1)},
    "R-patterns-translate-cases": {"arrangements": ("list", 6), "arrangements.channels": ("list", 2), "arrangements.expect_untranslated": ("list", 8), "arrangements.labels_recorded": ("eq", 233), "arrangements.leaves_not_upstream": ("list", 1), "arrangements.min_labels": ("ge", 224), "arrangements.prefixes": ("list", 2), "negative": ("list", 66), "notify_negative": ("list", 45), "notify_positive": ("list", 32), "positive": ("list", 81), "recorded": ("list", 10), "recorded.negative": ("eq", 66), "recorded.notify_negative": ("eq", 45), "recorded.notify_positive": ("eq", 32), "recorded.np_size": ("eq", 27), "recorded.patterns_size": ("eq", 28), "recorded.positive": ("eq", 81), "recorded.prefixed_size": ("eq", 19), "repo": ("str", 5), "src": ("str", 30), "stub_import": ("str", 54), "upstream_repo": ("str", 5), "upstream_src": ("str", 17)},
    "R-selfcheck-d-section-name": {"files": ("list", 2), "forbid_re": ("list", 1), "min_checks": ("ge", 4), "min_files": ("ge", 2), "require": ("list", 1)},
    "R-selfcheck-d-liveness": {"fakes": ("list", 6), "fakes.Gyroscopic Pemmican Requisition": ("str", 3), "fakes.Quaffle Marmalade Dispenser": ("str", 3), "fakes.This String Does Not Exist Upstream At All": ("str", 3), "fakes.Vorpal Blancmange Protocol": ("str", 3), "fakes.Xylophone Requisition Form": ("str", 3), "fakes.Zzq Frobnicated Widget": ("str", 3), "max": ("list", 5), "max.fetchFail": ("eq", 3), "max.missDistinct": ("eq", 4), "max.rawMiss": ("eq", 7), "max.uncheckedDistinct": ("eq", 168), "max.uncheckedRaw": ("eq", 186), "min": ("list", 11), "min.checkedDistinct": ("eq", 719), "min.fetchOk": ("eq", 219), "min.rawChecked": ("eq", 1273), "min.regexTables": ("eq", 2), "min.registeredDistinct": ("eq", 896), "min.registeredRaw": ("eq", 1495), "min.tableRegexEntries": ("eq", 47), "min.tableRows": ("eq", 39), "min.tablesFedIn": ("eq", 39), "min.tplFiles": ("eq", 67), "min.wrappedTables": ("eq", 13), "panel": ("str", 30), "repo": ("str", 5), "section": ("str", 18), "stub_import": ("str", 54), "substr_expect_miss": ("eq", 0), "substr_probe": ("str", 16), "tables_src": ("str", 30), "upstream_repo": ("str", 5)},
    "R-assertion-inputs-tracked": {"min_checked": ("ge", 40), "must_include": ("list", 5), "rules": ("str", 34), "sweep": ("list", 1), "sweep_ignore": ("list", 1)},
    "R-ruleset-shape": {},
    "R-html-tag-parity": {"min_leaves": ("ge", 15000), "min_tag_kinds": ("ge", 60), "min_tags": ("ge", 390000)},
}


# ============ 「这次判了几条规矩」的地板（第三十一轮 L5-①·2）
#
# ⚠ 复核点出的要害，本表按它设计：
#     **反空转的那个数，必须数的是「我判了几条规矩」，不是「我扫了几片叶子」。**
#   `R-bare-cn-fields` 的「检查 0 个叶」其实早就印在屏幕上了，缺的只是一句判定；
#   可**光补那一句不够** —— `R-lantyr` 的 `cn_forbidden` 清空之后，
#   「英文闸命中 244 叶」一个字节都不变。叶数是**语料规模**，规矩数才是**判据规模**。
#   两个数都要，但别把前者当后者用。
#
# 数法见 `_unit()` 上面那段：集合去重，所以与库有多大无关，只与「载荷还剩几项、
# 这次真判到了没有」有关。地板 = 本轮实测值。
JUDGED_UNITS = {
    "R-warlock-sorcerer": 1,
    "R-electricity-lightning": 1,
    "R-round-turn": 3,
    "R-tier-rank-level": 4,
    "R-region-area-map": 3,
    "R-tridents-point-falsepositive": 2,
    "R-soulbound-progression": 1,
    "R-highgate": 1,
    "R-otherhood": 1,
    "R-marlstone": 1,
    "R-evidence-value": 1,
    "R-presence-wisdom": 1,
    "R-lang-parity": 2,
    "R-anchor-ids": 1,
    "R-bare-cn-fields": 4,
    "R-exclusions-closed": 4,
    "R-glossary-not-laundering": 31,
    # 实测 62；地板取**单层**的量 —— 词表 base 层按设计住在项目根之外，执行体明说那一层可以不在（缺了由 min_checked 报）
    "R-catwalk": 1,
    "R-drakeling": 1,
    "R-afflicted": 1,
    "R-inkaro-pool": 1,
    "R-clipper-name": 1,
    "R-boardwalk": 1,
    "R-version-matrix": 6,
    "R-junction-wheel": 1,
    "R-shard-god": 4,
    "R-shard-god-blocks": 7,
    "R-rank-sense-compendium": 5,
    "R-rank-sense-blocks": 7,
    "R-dives-mine": 1,
    "R-arcturel-vs-arcturian": 3,
    "R-arcturel-arcturian-blocks": 13,
    "R-kessia-vs-kessian": 3,
    "R-ordain-vs-ordani": 3,
    "R-fortitude-scope": 3,
    # 实测 6；地板取**单层**的量 —— 词表 base 层按设计住在项目根之外，执行体明说那一层可以不在（缺了由 min_checked 报）
    "R-token-foundry-ui": 3,
    "R-aura-three-way": 11,
    "R-cyclonic": 4,
    "R-hex-tile": 2,
    "R-lantyr": 2,
    "R-corla-cora": 3,
    "R-orb-destroyed": 3,
    "R-obsidian-antiquary": 2,
    "R-temple-lunarium": 2,
    "R-point-cape": 2,
    "R-arcturian-split": 4,
    "R-arcturian-actor-card": 1,
    "R-cosmology-cosmos": 22,
    "R-hex-homograph": 6,
    "R-moon-hollow": 1,
    "R-moon-charred": 1,
    "R-moon-tempest": 1,
    "R-arcturel-arcturian-labels": 6,
    "R-dives-labels": 2,
    "R-kessia-labels": 4,
    "R-ordain-labels": 4,
    "R-shard-god-labels": 6,
    "R-moon-epithet-labels": 6,
    "R-token-rank-labels": 4,
    "R-readaloud-coverage": 4,
    "R-selfcheck-twin": 1,
    "R-patterns-translate-cases": 229,
    "R-selfcheck-d-section-name": 4,
    # 第三十二轮 V18：22 → 33（min 7→11 · max 3→5 · 新增 5 条不含阈值的台账恒等式）
    "R-selfcheck-d-liveness": 33,
    "R-assertion-inputs-tracked": 7,
    "R-ruleset-shape": 32,
    # 第三十三轮 B：69 种标签记号（本轮实测）+ 3 道地板 = 72。
    # ⚠ 地板取 64 而不是 72：这一条的规矩数是**从库里现算的标签种类**，
    #   `<tfoot>` / `<i>` / `<s>` / `<details>` 这几种全库只出现一两次，
    #   删掉一段正文就可能整种消失 —— 钉死 72 会把「译文改了一句」报成「判据空转」，
    #   而那句报错的措辞（「载荷被清空、执行体被换成空壳」）会把下一轮引到错的方向去。
    #   64 仍然远高于「执行体被掏空」的 0，反空转的职能一个字没丢。
    "R-html-tag-parity": 64,
}

# 上表的**合计**。⚠ 单独把某一条的地板改小是个单点动作，本表自己看不出来；
# 记一个合计之后，那一步必须**同时**改这里 —— 两处都在本文件、都是自证的 diff。
# 载荷地板那一侧走的是更强的「现推 == 登记」（见 `_derive_payload_floors`），
# 而规矩数是**运行时量出来的**、推不出来，所以只能到这一档。
# **这是两点门槛，不是墙** —— 写在这里，免得下一轮把它读成「堵死了」。
JUDGED_UNITS_TOTAL = 600          # 第三十三轮 B：536 → 600（tag_parity +64 条）


def _dig(rule, path):
    """按 (块名, 键名) 逐层取值，取不到返回 None。"""
    cur = rule
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


_MISSING = object()

# —— 载荷地板的**推导规则**（`_derive_payload_floors` 用）。
#    ⚠ 它存在的理由见 `run_payload_floor_selftest` 的「前置 C」：
#      光有 PAYLOAD_FLOORS 还不够 —— **把某一道地板改小**（`("list", 15)` → `("list", 0)`）
#      是个单点动作，本表自己看不出来。修法与第三十轮给 `min_by_kind` 用的是同一招：
#      让登记值**可从发布中的规则集现推**，自检里断言「登记 == 现推」。
#      于是「改小一道地板」必然与规则集对不上，而「合法地删内容」要求两处一起改、diff 自证。
_PAY_META = {"id", "kind", "title", "decision", "why", "note", "_why", "rule"}
# 含义两可 / 本来就按等号判的数：一律钉死，不给「往松了调」留缝
_PAY_EQ_KEYS = {"window", "short_len", "substr_expect_miss", "min_ratio", "min_sentence_frac"}
_PAY_RECURSE_MAX = 12          # 子键太多的字典（`entries` 31 条）只记总长，别把表撑爆
# ⚠⚠ 第三十二轮 E2：越过这个上限时，该块的**逐键钉死会整块消失**，而登记表正是从这里现推的。
#     于是存在这样一条**不需要有人存心去绕**的路：往 `panel_liveness.min`（当时正好 11 键）
#     加两个键 → 前置 C「现推 == 登记」当场红 → 照例重新生成一份登记表 → 又绿了，
#     **而那 11 道逐键死钉（min.checkedDistinct ('eq',719) 之类）已经在重生成里被洗掉了**。
#     红过一次、又绿了，没有人会想到去数一数「钉还剩几道」。
#     ⇒ 越限不许静默：记一条显式的 `<块>.__pins_dropped__`，让「钉没了」本身成为登记表里
#       **看得见的一项**。这是形态 (j) 的同型防法 —— 数的是「我判了几条规矩」，
#       而不是「我扫了几片叶子」。
#     ⚠ 边界照实说：它把**无声的退化**变成**有声的 diff**（重生成时 11 行钉塌成 1 行标记，
#       审 diff 的人看得见），**不是**让闸变红。本项目这套自检的职能一直就是前者，不是后者。


def _pay_classify(key, v):
    """一个载荷值 → (类型, 登记值)。数字的方向按**键名**定，见 PAYLOAD_FLOORS 上面那段。"""
    if isinstance(v, bool):
        return ("eq", v)
    if isinstance(v, (int, float)):
        leaf = key.split(".")[-1]
        if leaf in _PAY_EQ_KEYS:
            return ("eq", v)
        if leaf.startswith("max_") or leaf == "max":
            return ("le", v)
        if leaf.startswith("min_") or leaf == "min":
            return ("ge", v)
        return ("eq", v)
    if isinstance(v, str):
        return ("str", len(v))
    if isinstance(v, (list, dict)):
        return ("list", len(v))
    return None


def _derive_payload_floors(A):
    """从规则集**现推**一份载荷地板 —— 自检拿它和 PAYLOAD_FLOORS 逐条对。"""
    out = {}
    for r in A:
        rid = r.get("id")
        if not isinstance(rid, str):
            continue
        spec = {}
        for key, v in r.items():
            if key in _PAY_META:
                continue
            c = _pay_classify(key, v)
            if c:
                spec[key] = c
            if isinstance(v, dict) and len(v) > _PAY_RECURSE_MAX:
                # 越限：逐键钉死在这里整块消失。记一条标记，别让它无声（见上面 E2 那段）
                spec[f"{key}.__pins_dropped__"] = ("eq", len(v))
            elif isinstance(v, dict) and len(v) > 0:
                for k2, v2 in v.items():
                    if k2 in _PAY_META or k2.startswith("_why") or k2.startswith("why"):
                        continue          # 块内的散文注释键不是载荷，别把它的字数钉死
                    c2 = _pay_classify(f"{key}.{k2}", v2)
                    if c2:
                        spec[f"{key}.{k2}"] = c2
        out[rid] = spec
    return out


def _payload_one(rid, key, typ, val, got):
    """判一个载荷键。返回违规说明（str）或 None。抽成函数是为了能被 `--selftest` 直接打。"""
    if got is _MISSING:
        return (f"`{rid}` 少了载荷键 `{key}` —— 登记的是 {typ}{val!r}。"
                f"整条被换成一个只剩 id/kind 的壳，上面三层（条数/类数/登记 id）**全都满意**，"
                f"只有这一层会响")
    if typ == "list":
        if not isinstance(got, (list, dict)):
            return f"`{rid}.{key}` 应该是列表/字典，现在是 {got!r}"
        if len(got) < val:
            return (f"`{rid}.{key}` 只剩 {len(got)} 项（登记 {val} 项）—— "
                    f"载荷被清空/砍短了。要真删，把 PAYLOAD_FLOORS 里这一项一起改小，"
                    f"那处 diff 落在 assert_resolutions.py 上、是自证的")
    elif typ == "str":
        if not isinstance(got, str):
            return f"`{rid}.{key}` 应该是字符串，现在是 {got!r}"
        if len(got) < val:
            return (f"`{rid}.{key}` 只剩 {len(got)} 字（登记 {val} 字）：{got!r} —— "
                    f"判据串被清空/削短了")
    elif typ == "ge":
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            return f"`{rid}.{key}` 应该是数，现在是 {got!r}"
        if got < val:
            return f"`{rid}.{key}` 被改小了：{got} < 登记的地板 {val}（这一侧只许涨）"
    elif typ == "le":
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            return f"`{rid}.{key}` 应该是数，现在是 {got!r}"
        if got > val:
            return f"`{rid}.{key}` 被放大了：{got} > 登记的上限 {val}（这一侧只许降）"
    elif typ == "eq":
        if got != val:
            return (f"`{rid}.{key}` 从 {val!r} 变成了 {got!r} —— 含义两可的强度参数一律钉死，"
                    f"要改就在 PAYLOAD_FLOORS 里一起改")
    else:
        return f"`{rid}.{key}` 登记了未知的地板类型 {typ!r} —— 判据自己写坏了"
    return None


def _payload_floor_check(A):
    """逐条核载荷地板。返回 (违规行, 核过的键数)。`A` 是本次运行的 assertions 列表。"""
    bad = []
    # ⓪ 登记表与登记表之间先对一遍：本表的 id 集合必须与 REGISTERED_ASSERTIONS 相同。
    #    没有这一步，「把某条从本表里删掉」就是单点动作。
    only_p = sorted(set(PAYLOAD_FLOORS) - set(REGISTERED_ASSERTIONS))
    only_r = sorted(set(REGISTERED_ASSERTIONS) - set(PAYLOAD_FLOORS))
    for rid in only_p:
        bad.append(("-", "配置·payload_floors", rid,
                    "在 PAYLOAD_FLOORS 里、却不在 REGISTERED_ASSERTIONS 里 —— 两张登记表必须一一对应"))
    for rid in only_r:
        bad.append(("-", "配置·payload_floors", rid,
                    f"登记在册（kind={REGISTERED_ASSERTIONS[rid]}）却**没有载荷地板** —— "
                    f"新加断言要同时登记它的载荷，否则这一条的内容仍然可以被掏空而没人数"))
    n_keys = 0
    by_id = {r.get("id"): r for r in A if isinstance(r.get("id"), str)}
    for rid, spec in sorted(PAYLOAD_FLOORS.items()):
        r = by_id.get(rid)
        if r is None:
            continue                                # 「不在规则集里了」由 ①b 那一层报，别报两遍
        for key, (typ, val) in sorted(spec.items()):
            n_keys += 1
            path = key.split(".")
            cur, got = r, _MISSING
            if len(path) == 2 and path[1] == "__pins_dropped__":
                # 这不是规则里真有的键，是「本块越过 _PAY_RECURSE_MAX、逐键钉死已整块消失」
                # 这件事的显式标记（见 E2 那段）。它要判的就是那个块现在有多少键 ——
                # 键数一变，标记值就对不上，重生成登记表时那一行会明晃晃地摆在 diff 里。
                blk = r.get(path[0])
                got = len(blk) if isinstance(blk, dict) else _MISSING
            else:
                for i, k in enumerate(path):
                    if not isinstance(cur, dict) or k not in cur:
                        break
                    cur = cur[k]
                    if i == len(path) - 1:
                        got = cur
            why = _payload_one(rid, key, typ, val, got)
            if why:
                bad.append(("-", "配置·payload_floors", f"{rid}.{key}", why))
    return bad, n_keys


def _units_verdict(rid, n):
    """「这条断言这次判了几条规矩」的判定。返回说明（str）或 None。

    ⚠ 与 detail 里那些「扫了多少叶」的数**不是一回事**，见 `_unit()` 上面那段。
    ⚠ 没登记 = 判失败，不是放行：新加一条断言却不登记它的规矩数，
      等于给「新 kind 的执行体是空的」留门。
    """
    floor = JUDGED_UNITS.get(rid)
    if floor is None:
        return (f"`{rid}` 没有登记「判了几条规矩」的地板 —— 新断言必须同时登记 "
                f"JUDGED_UNITS，否则它的执行体空转没人数（本次实测 {n} 条）")
    if n < floor:
        return (f"`{rid}` 这次只判了 **{n} 条规矩**（登记地板 {floor} 条）—— "
                f"载荷被清空、闸没命中、或执行体被换成了空壳。"
                f"⚠ 注意这个数**不是**「扫了几片叶」：叶数是语料规模，清空 `cn_forbidden` "
                f"时叶数一个字节都不变，只有这个数会掉")
    return None


def a_ruleset_shape(rule, ctx):
    """规则集**自身的形状**：条数、kind 数、各 kind 的容器块。见上面那段注释。

    ⚠ 判的是**本次运行真正在跑的那份规则集**（`ctx.rules`，`main()` 里塞进来的），
      不是磁盘上另找一份重读 —— 重读一份就有形态 (g)「输入被静默换成另一份」的风险
      （`--rules` 指向 A 而本条去读 B，两边都绿，判的却不是同一件事）。
      拿不到 `ctx.rules` 一律判「没跑成」，不许静默跳过（形态 (e)）。

    ⚠ 每一条违规的第 2 元（原本一律是「配置」）现在带上**是哪一层响的**：
      `配置·min_assertions` / `配置·registered` / `配置·containers` …
      这不是为了好看 —— `--selftest` 的 ruleset_shape 那一组此前只断言「红了」、
      不断言「哪一层红」，于是把 `RULESET_SHAPE["containers"]` 整块清空，
      14 条用例照样全绿（第三十轮实测）。层名是让那一组能做**分层归因**的钩子。
    """
    rules = getattr(ctx, "rules", None)
    if not isinstance(rules, dict) or not isinstance(rules.get("assertions"), list):
        return ([("-", "没跑成", "ctx.rules",
                  "拿不到本次运行的规则集 —— 本条**没跑成**，不是通过（空转形态 (e)）")],
                "没跑成")
    A = rules["assertions"]
    S = RULESET_SHAPE
    bad = []

    # ① 条数与 kind 数。删掉整条断言在此之前**全项目没有任何东西会数一数**。
    if len(A) < S["min_assertions"]:
        bad.append(("-", "配置·min_assertions", "assertions",
                    f"规则集现在只有 {len(A)} 条断言，登记值是 {S['min_assertions']} 条 —— "
                    f"有断言被删了。要么改回来，要么**显式推翻那条裁决**并把 "
                    f"`RULESET_SHAPE['min_assertions']` 改成新值（那是一处自证的 diff，"
                    f"落在 assert_resolutions.py 上，而不是悄悄少一条）"))
    kinds = {r.get("kind") for r in A}
    if len(kinds) < S["min_kinds"]:
        bad.append(("-", "配置·min_kinds", "kinds",
                    f"断言类型只剩 {len(kinds)} 种，登记值是 {S['min_kinds']} 种 —— "
                    f"整整一类判据被删光了（删一类比删一条更难被人看见）"))
    for kind, n in sorted(S["min_by_kind"].items()):
        got = sum(1 for r in A if r.get("kind") == kind)
        if got < n:
            bad.append(("-", "配置·min_by_kind", kind,
                        f"`{kind}` 这一类只剩 {got} 条（登记值 {n} 条）—— "
                        f"整类被删光、或被削掉一部分。⚠ 这一层同时是 H1（`kind` 拼错一个字母）"
                        f"的接住点：`min_kinds` 数的是 distinct kind，拼错出来的那种"
                        f"**自己也算一种**、反而帮着满足地板；逐类点名之后就藏不住了"))

    # ①b **登记在册的 id 全集**（第三十轮，挡 H2「删一条补一条」）。
    #     `min_assertions` 只数 `len(A)`：删掉一条真的、复制另一条换个 id 补上，
    #     长度不变、distinct kind 不变，**输出里一行异常都没有**。
    #     所以除了「有几条」，还得判「**是哪几条**」。清单住在本文件（见 REGISTERED_ASSERTIONS）。
    seen_ids = {}
    for i, r in enumerate(A):
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            bad.append(("-", "配置·id_unique", f"assertions[{i}]",
                        f"这条断言没有可用的 `id`（{rid!r}）—— 没有 id 就无从登记、"
                        f"也无从核对「登记过的还在不在」，等于给 H2 留门"))
            continue
        if rid in seen_ids:
            bad.append(("-", "配置·id_unique", rid,
                        f"id 重复：assertions[{seen_ids[rid]}] 与 assertions[{i}] 同名 —— "
                        f"「删掉一条真的、把另一条复制一份补上」最省事的做法就是连 id 一起复制；"
                        f"id 必须唯一，否则登记清单是核不动的"))
            continue
        seen_ids[rid] = i
    for rid, want_kind in sorted(REGISTERED_ASSERTIONS.items()):
        if rid not in seen_ids:
            bad.append(("-", "配置·registered", rid,
                        f"登记在册的断言 `{rid}`（kind={want_kind}）**不在规则集里了** —— "
                        f"条数可以靠「补一条填充」凑回去，但**是哪几条**凑不回去。"
                        f"要真删它，就同时把 REGISTERED_ASSERTIONS 里这一行摘掉："
                        f"那处 diff 落在 assert_resolutions.py 上、写着「我把它摘了」，"
                        f"是自证的；而 H2 的两处改动都在规则文件里、互相抵消"))
            continue
        got_kind = A[seen_ids[rid]].get("kind")
        if got_kind != want_kind:
            bad.append(("-", "配置·registered", rid,
                        f"`{rid}` 的 kind 变了：登记的是 `{want_kind}`，现在是 {got_kind!r} —— "
                        f"H1（拼错一个字母让它静默停跑）与「换成一个更松的 kind」"
                        f"都在这一层接住。改 kind 是允许的，前提是同时改登记表"))

    # ①c **逐条的载荷地板**（第三十一轮 L5-①）：id 与 kind 钉死了，内容也得钉。
    #     见 PAYLOAD_FLOORS 上面那段 —— 这一层接住的是「整条换成只剩 id/kind 的壳」
    #     与「把某个载荷键清空」，而上面两层对这两种动作**完全满意**。
    pay_bad, n_payload = _payload_floor_check(A)
    bad.extend(pay_bad)
    _unit("layer:min_assertions", "layer:min_kinds", "layer:registered",
          "layer:id_unique", "layer:payload_floors")
    _unit(*(f"layer:by_kind:{k2}" for k2 in S["min_by_kind"]))

    # ② 容器块：该有的块必须**在、且非空**，块里点名的键必须齐。
    #    这一段专治「删 arrangements 整块 / 删面板闸的 min·max / 清空 must_include」。
    n_containers = n_keys = 0
    for r in A:
        kind = r.get("kind")
        spec = S["containers"].get(kind)
        if not spec:
            continue
        rid = r.get("id", "?")
        for block, keys in spec.items():
            n_containers += 1
            v = r.get(block)
            if not v:
                bad.append(("-", "配置·containers", f"{rid}.{block}",
                            f"`{kind}` 的规则必须有**非空**的 `{block}` 块，现在是 {v!r} —— "
                            f"整块删掉 = 把这一段护栏连同它的记录值一起端走（作弊路径 C）"))
                continue
            for k in keys:
                n_keys += 1
                if not isinstance(v, dict) or k not in v:
                    bad.append(("-", "配置·containers", f"{rid}.{block}.{k}",
                                f"`{block}` 块里少了点名的键 `{k}` —— "
                                f"单独摘掉一个键，跟整块删掉是同一条路"))

    # ③ 非空列表的最短长度（`channels` 砍成一个 = 同一道护栏减半）
    for path, floor in sorted(S["min_len"].items(), key=lambda kv: str(kv[0])):
        kind, sub = path[0], path[1:]
        for r in A:
            if r.get("kind") != kind:
                continue
            v = _dig(r, sub)
            if not isinstance(v, (list, dict)):
                bad.append(("-", "配置·min_len", f"{r.get('id', '?')}.{'.'.join(sub)}",
                            f"应该是个非空列表 / 字典，现在是 {v!r}"))
            elif len(v) < floor:
                bad.append(("-", "配置·min_len", f"{r.get('id', '?')}.{'.'.join(sub)}",
                            f"只剩 {len(v)} 项（登记值 {floor} 项）—— "
                            f"砍短它 = 同一道护栏成比例减半，而条数本身此前没人数"))

    # ④ 强度参数的死下限 / 死上限：**它们与判据不同层**（住在本文件里），
    #    所以「把规则里的强度参数改小」这一步不再是单点动作（§3.7.2）。
    for path, floor in sorted(S["hard_floors"].items(), key=lambda kv: str(kv[0])):
        kind, sub = path[0], path[1:]
        for r in A:
            if r.get("kind") != kind:
                continue
            v = _dig(r, sub)
            if not isinstance(v, (int, float)):
                bad.append(("-", "配置·hard_floors", f"{r.get('id', '?')}.{'.'.join(sub)}",
                            f"强度参数不见了或不是数：{v!r}"))
            elif v < floor:
                bad.append(("-", "配置·hard_floors", f"{r.get('id', '?')}.{'.'.join(sub)}",
                            f"强度参数被改小了：{v} < 登记的死下限 {floor}"))
    for path, ceil in sorted(S["hard_ceilings"].items(), key=lambda kv: str(kv[0])):
        kind, sub = path[0], path[1:]
        for r in A:
            if r.get("kind") != kind:
                continue
            v = _dig(r, sub)
            if not isinstance(v, (int, float)):
                bad.append(("-", "配置·hard_ceilings", f"{r.get('id', '?')}.{'.'.join(sub)}",
                            f"miss 侧的上限不见了或不是数：{v!r}"))
            elif v > ceil:
                bad.append(("-", "配置·hard_ceilings", f"{r.get('id', '?')}.{'.'.join(sub)}",
                            f"miss 侧的上限被放大了：{v} > 登记值 {ceil} —— "
                            f"查无此串的键只许越修越少"))

    # ⑤ `tracked_inputs` 的 `min_checked` 与 `must_include` 走**同一套两层地板**：
    #    历史层住在本文件（`hard_floors` / `tracked_must_include`），
    #    现算层从**规则集自身**现推 —— 每个 kind 的默认执行体都必须被点名，
    #    而 `min_checked` 不许低于点名清单的长度（点了名却不数它，等于没点）。
    live_must = {f"3-常用脚本/qa/{v}" for v in _KIND_DEFAULT_RUNNER.values()}
    for r in A:
        if r.get("kind") != "tracked_inputs":
            continue
        rid = r.get("id", "?")
        named = list(r.get("must_include") or [])
        for want in sorted(set(S["tracked_must_include"]) | live_must):
            if want not in named:
                bad.append(("-", "配置·tracked_must_include", f"{rid}.must_include",
                            f"点名清单里少了 `{want}` —— 历史层登记过它，"
                            f"或它是某个 kind 的**默认执行体**（现算层从规则集自身推出来的："
                            f"{sorted(live_must)}）。点名清单是防「推导器静默变短」的那一层，"
                            f"清空它 / 摘掉一条，那一层就没了"))
        mc = r.get("min_checked")
        if not isinstance(mc, int):
            bad.append(("-", "配置·tracked_min_checked", f"{rid}.min_checked",
                        f"不见了或不是整数：{mc!r}"))
        elif mc < len(named):
            bad.append(("-", "配置·tracked_min_checked", f"{rid}.min_checked",
                        f"`min_checked` = {mc}，比点名清单的 {len(named)} 条还少 —— "
                        f"点了名却不数它们，兜底形同虚设（现算层）"))

    _unit("layer:containers", "layer:min_len", "layer:hard_floors", "layer:hard_ceilings",
          "layer:tracked_must_include")

    return bad, (f"核过 {len(A)} 条断言 / {len(kinds)} 种 kind"
                 f"（登记地板 {S['min_assertions']} 条 / {S['min_kinds']} 种 / "
                 f"逐类点名 {len(S['min_by_kind'])} 类 / 登记在册 id "
                 f"{len(REGISTERED_ASSERTIONS)} 条，"
                 f"地板住在 assert_resolutions.py 的 RULESET_SHAPE 与 REGISTERED_ASSERTIONS，"
                 f"**不在规则文件里**）；"
                 f"逐条核过 {len(PAYLOAD_FLOORS)} 条断言的 {n_payload} 个载荷地板、"
                 f"{n_containers} 个容器块 / {n_keys} 个点名键、"
                 f"{len(S['min_len'])} 道长度地板、"
                 f"{len(S['hard_floors'])} 道强度死下限 + {len(S['hard_ceilings'])} 道死上限")


def _require_ruleset_shape(rules):
    """⚠ **这道检查故意不写成断言** —— 写成断言就又是「住在被守护的容器里」。

    它是 `main()` 的固定动作：规则集里一条 `ruleset_shape` 都没有，主闸直接判失败。
    没有它的话，把那条断言删掉 = 守卫和容器一起消失（本轮堵的正是这种）。
    """
    n = sum(1 for r in rules.get("assertions", []) if r.get("kind") == "ruleset_shape")
    if n:
        return None
    return ("规则集里没有任何 `ruleset_shape` 断言 —— 那是**数一数还剩几条规则**的那一条。"
            "把它删掉，删断言 / 删容器块就再也没有东西会响（作弊路径 C）。"
            "这道检查写在 assert_resolutions.py 的 main() 里、**不在规则文件里**，"
            "就是为了让「连守卫一起删」这一步也得改判据本身。")


def _exit_code(failed, skipped):
    """主闸的**判定式**：跳过 ≠ 通过。抽成函数是为了能被 `--selftest` 直接打（第三十轮）。

    ⚠ 它替代的是 `return 1 if failed else 0`。那一行让 **H1 / H3** 免费通行：
      把某条断言的 `kind` 拼错一个字母、或塞一个 `{"kind": "nope"}` 空壳，
      循环里 `skipped += 1` 就过去了，主闸打出 `65 / 0 / 跳过 1` 并 **rc=0**。
      「有一条断言今天没跑」和「今天全过了」被印成同一个结论，这正是 §3.7 要防的空转。

    ⚠ **`skipped` 现在有没有合法用途？** 查过了：**一个都没有**。
      本文件里 `skipped` 只有一个产出口，就是「KINDS 里没有这个 kind」，
      而那一支现在直接记 `failed`。`--repo` 那种部分仓跑法**不走这个计数器** ——
      它在 `main()` 组装 `repos` 时就把仓过滤掉了，断言照跑，`skipped` 始终是 0；
      各条断言内部的「这条不适用于本仓」也是自己返回 0 违规，不回传跳过。
      所以这里可以一刀切，**不会误伤合法跑法**。
      ⇒ 如果将来真要引入「合法跳过」（比如「这条只在有上游安装目录时跑」），
        **必须新开一个计数器并在这里显式放行**，而不是把它塞回 `skipped` ——
        那等于把这个洞原样挖回来。
    """
    return 1 if (failed or skipped) else 0


KINDS = {
    "term_gated": a_term_gated,
    "ruleset_shape": a_ruleset_shape,
    "translate_cases": a_translate_cases,
    "panel_liveness": a_panel_liveness,
    "tracked_inputs": a_tracked_inputs,
    "cn_absent": a_cn_absent,
    "sense_gated": a_sense_gated,
    "distinct_terms": a_distinct_terms,
    "term_domains": a_term_domains,
    "lang_parity": a_lang_parity,
    "anchor_ids": a_anchor_ids,
    "no_bilingual_tail": a_no_bilingual_tail,
    "exclusions_closed": a_exclusions_closed,
    "leaf_literal": a_leaf_literal,
    "block_aligned_gate": a_block_aligned_gate,
    "block_sense_gate": a_block_sense_gate,
    "tag_parity": a_tag_parity,
    "enricher_slot_gate": a_enricher_slot_gate,
    "enricher_text_coverage": a_enricher_text_coverage,
    "glossary_value": a_glossary_value,
    "version_matrix": a_version_matrix,
    "twin_files": a_twin_files,
    "source_literal": a_source_literal,
}


# —— 给 22 个执行体套上调用计数（第三十一轮 L5-②）。
#
# ⚠ **为什么必须是包装而不是名册**：`--selftest` 有 15 组用例、203 条全绿，可其中 11 个 kind
#   的执行体**一次都没被调用过**（`anchor_ids · cn_absent · distinct_terms · exclusions_closed ·
#   glossary_value · lang_parity · leaf_literal · no_bilingual_tail · term_domains · term_gated ·
#   version_matrix`，覆盖 66 条断言里的 48 条）。把它们中的任何一个整体换成 `return [], ""`，
#   主闸 66/0/0、自检 203/203，**一处改动、两道闸都不响**。
#   名册（「这个 kind 有没有一组用例」）挡不住这个：`glossary_value` 那一组**是有的**，
#   它打的是 `glossary_value_matches` 这个纯函数，跟执行体没关系。所以只能数**真实调用**。
#
# ⚠ 同时把模块级同名函数一起换掉：`run_selftest()` 里是按名字取全局的
#   （`a_cn_absent(...)`），不换的话自检打的仍是**没套计数的那一份**，覆盖表会数出 0。
def _counted(kind, fn):
    def _wrapped(rule, ctx):
        _KIND_CALLS[kind] = _KIND_CALLS.get(kind, 0) + 1
        return fn(rule, ctx)
    _wrapped.__name__ = fn.__name__
    _wrapped.__doc__ = fn.__doc__
    _wrapped.__wrapped__ = fn
    _wrapped._kind = kind
    return _wrapped


for _k, _f in list(KINDS.items()):
    KINDS[_k] = _counted(_k, _f)
    globals()[_f.__name__] = KINDS[_k]
del _k, _f


# ----------------------------------------------------------------- 自检

SELFTEST = [
    # (值, 是否应判为双语尾巴)  —— 正例来自库内真实写法，反例来自本项目既定约定
    ("螯蛛艾斯 Cheliceraeth", True),
    ("血鸟 Gore Bird", True),
    ("赛洛克弓手 Thayloc Courser", True),
    ("圣堂区路人 A", False),        # 单字母编号，英文侧就是 Hallows Passerby A
    ("菌丝旷野底图 B", False),      # 同上
    ("软泥池 2", False),            # 数字编号
    ("环境音效（2）", False),        # 全角括注
    ("斥候", False),                # 纯裸中文
    ("莫伊雷语", False),
    ("瀑布外部东北", False),
]


# 第十六轮补：读库闸本身的正反例。
#
# 光测「双语尾巴」是不够的 —— 上一次事故（`distinct_terms` 空转）恰恰出在**闸的机制**上，
# 而不是出在某个字符串判据上。这里用一棵合成的假库把 `_gate_one` 的四种行为钉住：
#   命中/不命中 · 占位符剥离 · cn_gate 放宽形态 · cn_forbidden。
class _FakeCtx:
    def __init__(self, lang=(), pairs=()):
        self._lang = list(lang)
        self._pairs = list(pairs)

    def all_lang(self, scope=None):
        return iter(self._lang)

    def all_pairs(self, scope=None):
        return iter(self._pairs)


GATE_SELFTEST = [
    # (说明, term, ctx, scan, 期望命中, 期望违规数)
    ("lang 英文闸命中且中文正确",
     {"en": "Region Map", "cn": "地区地图"},
     _FakeCtx(lang=[("ember", "K1", "Region Map", "地区地图")]), ["lang"], 1, 0),
    ("lang 英文闸命中但中文写反 —— 这正是四个版本没人发现的那种",
     {"en": "Region Map", "cn": "地区地图"},
     _FakeCtx(lang=[("ember", "K1", "Region Map", "区域地图")]), ["lang"], 1, 1),
    ("占位符 {rank} 要先剥掉，否则这条会被误判成违规",
     {"en": "Rank", "cn": "阶位"},
     _FakeCtx(lang=[("cruc", "K2", "{rank} vs. {defense}", "{rank} 对抗 {defense}")]),
     ["lang"], 0, 0),
    ("cn_gate 放宽：Level 的定译是「等级」，但「{x} 级」也合法",
     {"en": "level", "cn": "等级", "cn_gate": "级"},
     _FakeCtx(lang=[("ember", "K3", "Level {x}", "{x} 级")]), ["lang"], 1, 0),
    ("cn_forbidden：闸内出现禁用写法要响",
     {"en": "Cyclonic", "cn": "气旋", "cn_forbidden": ["旋风的"]},
     _FakeCtx(pairs=[("ember", "p.json", "a.text", "Cyclonic blast", "气旋冲击（旋风的）")]),
     ["compendium"], 1, 1),
    ("except_paths：登记过的叶不计命中也不报违规",
     {"en": "Cyclonic", "cn": "气旋", "except_paths": ["items.Multiattack"]},
     _FakeCtx(pairs=[("ember", "p.json", "actors.X.items.Multiattack.description",
                      "[[/item Cyclonic]]", "用 [[/item Cyclonic]] 攻击")]),
     ["compendium"], 0, 0),
    ("英文不命中就不该产生任何检查",
     {"en": "Region Map", "cn": "地区地图"},
     _FakeCtx(lang=[("ember", "K4", "Area Map", "区域地图")]), ["lang"], 0, 0),
]


# 第十六轮终段补：`sense_gated` 的正反例。
#
# 这个类型比 `_gate_one` 更容易写坏，因为它有**三条**可以各自失效的判据
# （反向闸 / COMMON 专属闸 / 上下文分类），而且分类是靠一个窗口正则做的 ——
# 窗口取错、忘了剥标签、COMMON 与 GAME 的优先级写反，都会静静地把闸变成摆设。
# 下面六条把这三件事各钉一次，其中「剥标签」那条来自实测：`Bonus | Rank | Scale`
# 在原文里被 `<td>` 拆成三格，不剥标签窗口里根本凑不出 `Bonus`（split_rank.py 第一版栽过）。
_SENSE_RULE = {
    "id": "SELFTEST", "kind": "sense_gated", "en": r"\branks?\b", "cn": "阶位",
    "sense": {"game": r"(Novice|training|skill|Attunement|Soulbound|Rank\s*\d|\bBonus\b|Scale)",
              "common": r"(civic|social|ranks of|rank[- ]and[- ]file|rank as an? )"},
    "window": 90,
}

SENSE_SELFTEST = [
    ("普通名词义专属叶里出现「阶位」→ 要响",
     [("e", "p.json", "a.text", "It denotes their civic rank in the city.", "标示其在城中的阶位")], 1),
    ("同一叶普通名词义、中文写「地位」→ 不响",
     [("e", "p.json", "a.text", "It denotes their civic rank in the city.", "标示其在城中的公民地位")], 0),
    ("机制义叶中文没有「阶位」→ **不响**（正向闸是故意不做的，见 docstring 判据边界）",
     [("e", "p.json", "a.text", "You gain the Novice rank in Arcana.", "你在奥秘上获得新手层级")], 0),
    ("反向闸：中文有「阶位」而英文一个 rank 都没有 → 要响",
     [("e", "p.json", "a.text", "Tier 3 creatures are dangerous.", "3 阶位的生物很危险")], 1),
    ("COMMON 优先级高于 GAME：窗口里同时有 skill 和 ranks of 时判 COMMON",
     [("e", "p.json", "a.text", "grit is required to endure the training to join the ranks of the order",
       "需要毅力才能熬过训练加入该教团的阶位")], 1),
    ("必须先剥标签：`Bonus|Rank|Scale` 被 <td> 拆开，剥了才判得出 GAME（判 GAME 就不报）",
     [("e", "p.json", "a.text", "<tr><td>Bonus</td><td>Rank</td><td>Scale</td></tr>",
       "<tr><td>加值</td><td>阶位</td><td>尺度</td></tr>")], 0),
]


# 第十六轮收尾补：`glossary_value` 的头部判据正反例。
#
# 加这一组的直接理由是它**真的咬过一次**：按 §8 的散文裁决把 `Cosmology` 登成 want='宇宙'，
# 产物「宇宙观 Cosmology」的头部是「宇宙观」，断言当场变红。那不是库错了，是登记值选错了 ——
# 而这种「判据形状决定了该登什么」的知识，只写在 why 里会随人走，钉成正反例才留得下来。
GLOSSARY_SELFTEST = [
    # (说明, want, got, 期望)
    ("base 层裸中文、与登记值相同 → 过", "宇宙观", "宇宙观", True),
    ("产物层双语尾巴，只比中文头部 → 过", "宇宙观", "宇宙观 Cosmology", True),
    ("⚠ 头部是**整词**不是前缀：want='宇宙' 对 got='宇宙观 Cosmology' 必须**不过**",
     "宇宙", "宇宙观 Cosmology", False),
    ("同一个坑的另一半：want='宇宙' 对裸中文「宇宙观」也必须不过", "宇宙", "宇宙观", False),
    ("英文尾巴有多个词时靠 got 全等那一支过（头部只切到第一个空格）",
     "凯西安沙刀", "凯西安沙刀 Kessian Sand Knife", True),
    ("base 里的多义 list（Ordain / Shield）恒判不过 —— 它们不该出现在 entries 里",
     "奥尔丹", ["奥尔丹", "授命"], False),
]


# 第十八轮 Y2 补：两个块对齐类型的正反例。
#
# 这两个类型比前面所有类型都更容易「看起来在跑、其实判不着」，因为它们有**四件**
# 可以各自失效的东西：切块规则 · 类别正则的顺序 · 对齐判据本身 · 豁免匹配。
# 下面把四件各钉一次，其中三条直接来自本轮实测踩到的坑：
#   · 行内标签**不**切块（中文定语在前会把词搬过 `<strong>`，全切会造出成片假阳性）
#   · `count_ge` 而非逐位对齐（中文不标复数，S/P 逐位对齐这个判据本身不成立）
#   · 混合义项叶里的**单一义项块**必须判得动（这正是叶级 sense_gated 整叶放弃的那片）
_ALIGN_RULE = {
    "id": "SELFTEST-align", "kind": "block_aligned_gate", "mode": "sequence",
    "leaf_gate": r"\bArctur", "min_leaves": 1, "min_blocks": 1,
    "en_tokens": [{"re": r"\bArcturians?\b", "cls": "I"}, {"re": r"\bArcturel\b", "cls": "E"}],
    "cn_tokens": [{"re": "阿克图里安人", "cls": "I"}, {"re": "阿克图里安", "cls": "I"},
                  {"re": "阿克图瑞尔", "cls": "E"}],
}
_COUNT_RULE = {
    "id": "SELFTEST-count", "kind": "block_aligned_gate", "mode": "count_ge",
    "leaf_gate": r"\bShards? God", "min_leaves": 1, "min_blocks": 1,
    "backward_classes": ["F"],
    "en_tokens": [{"re": r"\bShards? Goddess(?:es)?\b", "cls": "F"},
                  {"re": r"\bShards? Gods?\b", "cls": "G"}],
    "cn_tokens": [{"re": "碎片女神", "cls": "F"}, {"re": "碎片诸神", "cls": "G"},
                  {"re": "碎片之神", "cls": "G"}],
}


def _P(en, cn):
    return [("e", "p.json", "j.Some Page.text", en, cn)]


BLOCK_ALIGN_SELFTEST = [
    ("逐块对齐、两侧一致 → 不响", _ALIGN_RULE,
     _P("<p>Arcturel is a city.</p><p>Arcturian dwellings.</p>",
        "<p>阿克图瑞尔是一座城。</p><p>阿克图里安住所。</p>"), 0),
    ("⚑ **叶内单处串行** → 要响（这正是叶级判据看不见的：整叶两个中文都在，闸会放行）",
     _ALIGN_RULE,
     _P("<p>Arcturel is a city.</p><p>Arcturian dwellings.</p>",
        "<p>阿克图瑞尔是一座城。</p><p>阿克图瑞尔住所。</p>"), 1),
    ("语序调换会响 —— 这是逐位对齐**已知的代价**，实测 1714 块里只有 1 块，登记在 except_blocks",
     _ALIGN_RULE, _P("<p>Arcturian shops of Arcturel</p>", "<p>阿克图瑞尔的阿克图里安商铺</p>"), 1),
    ("行内标签**不**切块：`<strong>` 两侧的词算同一块（全切的话这条会假阳性）",
     _ALIGN_RULE, _P("<p>the <strong>Arcturel</strong> Dives</p>", "<p>阿克图瑞尔矿渊</p>"), 0),
    # ⚠ 这一条的 leaf_gate 必须真的在**英文原文**里命中，否则整叶根本不进闸、
    #   测到的就只是 min_leaves 在响。实测第一版写成纯 @UUID 的英文，闸下 0 叶、假绿。
    ("增强器标签两侧都涂掉：英文裸 @UUID、中文补了 {标签} → 不响",
     _ALIGN_RULE, _P("<p>the @UUID[Actor.x] bard of Arcturel</p>",
                     "<p>这位@UUID[Actor.x]{阿克图里安}吟游诗人来自阿克图瑞尔</p>"), 0),
    # 结构不齐会**同时**从三个方向吵：该叶一条 + max_shape_mismatch 超限一条 +
    # 该叶被跳过导致 min_blocks 掉到 0 一条。三条都该在 —— 判不了就是不能算通过。
    ("块级标签结构两侧不同 → 要响（判不了必须吵出来，不能当通过）",
     _ALIGN_RULE, _P("<p>Arcturel</p><p>Arcturian</p>", "<p>阿克图瑞尔 阿克图里安</p>"), 3),
    ("英文闸一块都没命中 → min_blocks 把空转抓出来",
     _ALIGN_RULE, _P("<p>Nothing here.</p>", "<p>这里什么都没有。</p>"), 2),
    ("count_ge：单／复数**不进类**，`Shard Gods` 对「碎片之神」不响（中文不标复数）",
     _COUNT_RULE, _P("<p>Shard Gods are mortal ascendants.</p>", "<p>碎片之神是飞升的凡人。</p>"), 0),
    ("count_ge：代词还原让中文多出一处 → 不响（可多不可少）",
     _COUNT_RULE, _P("<p>A Shard God arrived. They blessed it.</p>",
                     "<p>一位碎片之神到来。碎片诸神赐下祝福。</p>"), 0),
    ("count_ge：块内两处英文、中文只译了一处 → 要响",
     _COUNT_RULE, _P("<p>Shard God A fought Shard God B.</p>", "<p>碎片之神 A 与 B 交战。</p>"), 1),
    ("count_ge：把「碎片女神」并进「碎片之神」→ 要响（女神那一支中文扛得住，一处不许错）",
     _COUNT_RULE, _P("<p>the shard goddess Scoris</p>", "<p>碎片之神斯科里斯</p>"), 1),
    ("count_ge 反向闸：英文没有 Goddess 而中文写了「碎片女神」→ 要响",
     _COUNT_RULE, _P("<p>the Shard Gods gathered</p>", "<p>碎片女神们聚集</p>"), 1),
    # ⚑ 死豁免闸（`max_unused_exempt`，默认 0）。这两条钉的是**第四种空转形态**：
    #    闸本身照常跑、照常全绿，只是豁免表里躺着再也匹配不到的条目。实测代价见
    #    `_unused_exempt` 的 docstring（收尾时两条块级断言合计躺着 7 条，全套仍 52/0）。
    ("⚑ 豁免一条都没命中 → 要响（死豁免自己吵出来，不再只能靠人记）",
     dict(_ALIGN_RULE, except_blocks=[
         {"path": "j.Nowhere At All.text", "block": 99, "why": "自检：故意留一条永远匹配不到的"}]),
     _P("<p>Arcturel is a city.</p><p>Arcturian dwellings.</p>",
        "<p>阿克图瑞尔是一座城。</p><p>阿克图里安住所。</p>"), 1),
    ("同一条豁免真的命中 → 不响（证明上一条响的是「没命中」而不是「有豁免」）",
     dict(_ALIGN_RULE, except_blocks=[
         {"path": "j.Some Page.text", "block": 3, "en": "I", "cn": "E", "why": "自检：这条真命中"}]),
     _P("<p>Arcturel is a city.</p><p>Arcturian dwellings.</p>",
        "<p>阿克图瑞尔是一座城。</p><p>阿克图瑞尔住所。</p>"), 0),
]

_SENSE_BLOCK_RULE = {
    "id": "SELFTEST-bsense", "kind": "block_sense_gate", "occ": r"\branks?\b",
    "leaf_gate": r"\branks?\b", "cn": "阶位", "window": 90,
    "min_leaves": 1, "min_blocks": 1,
    "sense": {
        "strong_game": r"(attunement|attuned|soulbound|soulmark|Rank\s*\d)",
        "game": r"(Novice|training|skill|Attunement|Soulbound|exhaustion|Scale|Rank\s*\d|\bBonus\b)",
        "common": r"(ranks? (depending|based) on|civic|social|ranks of|rank[- ]and[- ]file|rank as an? )",
        "exempt": r"(ranks? of[^.]{0,40}?exhaust|close ranks|join(ing)? their ranks)",
    },
}

# ⚠ 第二项是这一条自检往 `_SENSE_BLOCK_RULE` 上打的**规则覆盖**（`None` ＝不覆盖）。
# 原来的写法是「note 里含 'except_blocks' 就塞一张表」，靠**注释文字**驱动判据 ——
# 加第二条豁免用例时它当场就不够用了（几条都含那个词、却要各自不同的表），所以改成显式一列。
BLOCK_SENSE_SELFTEST = [
    ("⚑ **正向闸**：块内全机制义、中文没有「阶位」→ 要响（叶级 sense_gated 故意不做这个方向）",
     None, _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手层级。</p>"), 1),
    ("块内全机制义、中文有「阶位」→ 不响",
     None, _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手阶位。</p>"), 0),
    ("反向闸：块内全普通名词义、中文却用「阶位」→ 要响",
     None, _P("<p>It denotes their civic rank.</p>", "<p>标示其公民阶位。</p>"), 1),
    ("⚑ **混合义项叶里的单一义项块判得动** —— 叶级版对这一叶是 MIX、整叶放弃",
     None, _P("<p>It denotes their civic rank.</p><p>You gain the Novice rank in Arcana.</p>",
              "<p>标示其公民地位。</p><p>你在奥秘上获得新手层级。</p>"), 1),
    ("第三义项 `rank of exhaustion`（＝层）不归这条裁决管 → 不响，哪怕中间夹着增强器",
     None, _P("<p>Each character gains one rank of &amp;Reference[exhaustion] and must save.</p>",
              "<p>每名角色获得一级力竭，并且必须豁免。</p>"), 0),
    ("strong_game 压过 common：`Ranks of attunement progression` 是机制义，不是「行列」",
     None, _P("<p>There are now five full Ranks of attunement progression.</p>",
              "<p>现在同调进阶共有完整的五个阶位。</p>"), 0),
    ("行内标签**不**切块：中文把「同调阶位」搬到了 `<strong>` 前面 → 不响（全切会假阳性）",
     None, _P("<p>You gain resistance to <strong>Acid</strong> damage equal to 2 times "
              "your attunement rank.</p>",
              "<p>你获得等同于同调阶位 2 倍的<strong>强酸</strong>伤害抗性。</p>"), 0),
    ("组织内部层级 `ranks depending on experience and skill` 判 COMMON，中文写「等级」不响",
     None, _P("<p>Within the Guard there are a number of ranks depending on experience and skill.</p>",
              "<p>卫队内部依照经验与技能设有多个等级。</p>"), 0),
    ("登记过的块不再报（登记的是内容欠账，必须同时升报）",
     {"except_blocks": [{"path": "j.Some Page.text", "block": 1, "why": "自检：这条真命中"}]},
     _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手层级。</p>"), 0),
    # ⚑ 死豁免闸在 block_sense_gate 这一侧的正反例。与 block_aligned_gate 那边成对，
    #    因为两个类型各有一份自己的 `used` 计数，只钉一边等于另一边没测。
    ("⚑ 豁免一条都没命中 → 要响（`max_unused_exempt` 默认 0，见 _unused_exempt）",
     {"except_blocks": [{"path": "j.Nowhere.text", "block": 99, "why": "自检：永远匹配不到"}]},
     _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手阶位。</p>"), 1),
    ("把上限显式放宽到 1 → 同一条死豁免不再响（证明响的是「未命中」本身，不是「有豁免」）",
     {"except_blocks": [{"path": "j.Nowhere.text", "block": 99, "why": "自检：永远匹配不到"}],
      "max_unused_exempt": 1},
     _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手阶位。</p>"), 0),
]


# ============================ HTML 标签 EN/CN 对等闸（第三十三轮 B）的正反例
#
# ⚠ 这一组的两条**必答题**（任务点名要的，也是本项目第六层的教训「没有反例的护栏等于没有」）：
#   ① 「给某叶中文多加一个 `<strong>` → 必须变红」——`TAG_PARITY_SELFTEST` 第 2 条；
#   ② 「掏空这条判据的判定函数 → 必须有用例变红」——`run_tag_parity_selftest()` 末尾
#      那两条**元用例**：把 `tag_diff` / `tag_multiset` 换成空壳后重跑第 2 条，
#      **它必须从「响」变成「不响」**（也就是说：它确实是被那个判定函数撑起来的，
#      不是恰好被别的层顺手接住）。只断言「换了空壳之后自检红了」是不够的 ——
#      红也可能是别的层在响，那正是 ruleset_shape 那一组第三十轮踩过的坑（分层归因）。
_TAG_RULE = {
    "id": "SELFTEST-tag", "kind": "tag_parity",
    "min_leaves": 1, "min_tags": 1, "min_tag_kinds": 1,
}
# 自检用的死下限（真身那三个数是按 4 万叶的库定的，合成叶喂进去必然触底）。
# ⚠ 真身的 `TAG_PARITY_HARD` 由下面第 9 条**单独**验它咬不咬人，不能只在放松的副本上测。
_TAG_HARD_TINY = {"min_leaves": 1, "min_tags": 1, "min_tag_kinds": 1}

TAG_PARITY_SELFTEST = [
    ("两侧标签多重集相同 → 不响", None,
     _P("<p>The <strong>Overrun</strong> creature suffers <em>flanked</em>.</p>",
        "<p>处于<strong>围攻</strong>状态的生物承受<em>夹击</em>。</p>"), 0),
    # ⚑ 必答题 ①：本轮真实发生过的那件事 —— 译者顺手给中文加了个 `<strong>`。
    #    此前**没有任何判据看得见**：单侧总数照样守恒、种类数一个不多。
    ("⚑ 中文多一个 `<strong>` → 必须响（这正是本轮漏掉的那 2 处）", None,
     _P("<p>The <strong>Overrun</strong> creature suffers flanked.</p>",
        "<p>处于<strong>围攻</strong>状态的生物承受<strong>夹击</strong>。</p>"), 1),
    ("中文少一个 `<em>` → 必须响（少也是不对等，别只测多的那一侧）", None,
     _P("<p>a <em>straight</em> line, <em>bowling</em> over</p>",
        "<p>一条<em>直线</em>，撞翻挡路者</p>"), 1),
    ("标签换了种类（EN `<em>` ↔ CN `<strong>`）→ 必须响（总数相等，只有多重集看得见）", None,
     _P("<p>a <em>Juggernaut</em> charge</p>", "<p>一次<strong>坚壁</strong>冲锋</p>"), 1),
    ("属性不看：`<span class=\"x\">` 与 `<span>` 是同一个记号 → 不响（属性漂移归 scan_markup_drift）",
     None, _P("<p><span class=\"x\">Stride</span></p>", "<p><span>移动</span></p>"), 0),
    ("自闭合归一：`<br>` / `<br />` 是同一个记号 → 不响", None,
     _P("<p>one<br>two</p>", "<p>一<br />二</p>"), 0),
    ("增强器里没有 `<`，两侧标签本就相同 → 不响（硬约束 3 的东西碰不到本条）", None,
     _P("<p>@UUID[Item.x]{Overrun} knocks them @Condition[prone]</p>",
        "<p>@UUID[Item.x]{冲撞}将其击倒@Condition[prone]</p>"), 0),
    # 三道地板同时响：叶 0 / 标签 0 / 种类 0。三道都要在 —— 「无从查起」不许印成「查过了没问题」。
    ("两侧都不带标签的叶不进闸 → 进闸 0 叶，三道地板一起响（**不是**判成通过）", None,
     _P("Overrun", "冲撞 Overrun"), 3),
    # ⚑ 两层地板各自的反例：规则文件那一层被摘掉 / 被调松，都不许静默过。
    ("规则里少了 `min_leaves` → 报「两层里少了一层」（缺一层不许当没事）",
     {"min_leaves": None}, _P("<p>a</p>", "<p>甲</p>"), 1),
    ("规则里把 `min_tags` 调成 0、而本文件的死下限是 99 → 仍然响（作弊路径 B 越不过 .py）",
     {"min_tags": 0, "_hard": {"min_leaves": 1, "min_tags": 99, "min_tag_kinds": 1}},
     _P("<p>a</p>", "<p>甲</p>"), 1),
]


def run_tag_parity_selftest():
    """返回 (失败条数, 总条数)。含两条**掏空判定函数**的元用例。"""
    global TAG_PARITY_HARD, tag_diff, tag_multiset
    results = []
    saved_hard, saved_diff, saved_ms = TAG_PARITY_HARD, tag_diff, tag_multiset

    def _run(override, pairs):
        hard = (override or {}).get("_hard") or _TAG_HARD_TINY
        rule = dict(_TAG_RULE)
        for k, v in (override or {}).items():
            if k == "_hard":
                continue
            if v is None:
                rule.pop(k, None)                      # None ＝「把这个键摘掉」
            else:
                rule[k] = v
        globals()["TAG_PARITY_HARD"] = hard
        try:
            return a_tag_parity(rule, _FakeCtx(pairs=pairs))
        finally:
            globals()["TAG_PARITY_HARD"] = saved_hard

    for note, override, pairs, want in TAG_PARITY_SELFTEST:
        b, detail = _run(override, pairs)
        ok = len(b) == want
        results.append((note, ok, "" if ok else f"期望违规 {want}，实得 {len(b)}：{b[:2]}｜{detail}"))

    # —— 元用例 A：掏空 `tag_diff`（判定就在那一行）→ 必答题 ② 那条用例必须失守。
    _note, _ov, _pairs, _want = TAG_PARITY_SELFTEST[1]
    try:
        globals()["tag_diff"] = lambda a, b: ""
        b, _d = _run(_ov, _pairs)
        ok = len(b) == 0
    finally:
        globals()["tag_diff"] = saved_diff
    results.append(("⚑ 掏空 `tag_diff` → 上面那条「中文多一个 `<strong>`」必须从响变成不响"
                    "（证明它是被这个判定函数撑起来的，不是被别的层顺手接住）", ok,
                    "" if ok else f"掏空之后仍报了 {len(b)} 处 —— 那条反例测的不是这个函数"))

    # —— 元用例 B：掏空 `tag_multiset` → 一叶都进不了闸，地板层接住。
    try:
        globals()["tag_multiset"] = lambda s: collections.Counter()
        b, d = _run(None, TAG_PARITY_SELFTEST[0][2])
        ok = len(b) >= 1 and "进闸 0 叶" in d
    finally:
        globals()["tag_multiset"] = saved_ms
    results.append(("⚑ 掏空 `tag_multiset` → 进闸 0 叶、地板层当场响（空转不许判成通过）", ok,
                    "" if ok else f"实得违规 {len(b)}｜{d}"))

    # —— 元用例 C：发布中的**真身**死下限确实咬人（上面十条跑的都是放松过的副本，
    #    只测副本等于没测那三个真数 —— 形态 (g)「验的不是产线那一份」）。
    b, d = a_tag_parity(dict(_TAG_RULE), _FakeCtx(pairs=_P("<p>a</p>", "<p>甲</p>")))
    ok = len(b) == 3 and str(TAG_PARITY_HARD["min_leaves"]) in " ".join(x[3] for x in b)
    results.append((f"⚑ 用**真身** TAG_PARITY_HARD（叶{TAG_PARITY_HARD['min_leaves']}·"
                    f"标签{TAG_PARITY_HARD['min_tags']}·种类{TAG_PARITY_HARD['min_tag_kinds']}）"
                    f"跑一片合成叶 → 三道死下限全部响", ok,
                    "" if ok else f"期望 3 处，实得 {len(b)}｜{d}"))

    print("\nHTML 标签 EN/CN 对等闸（tag_parity）正反例：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\ntag_parity：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


_SLOT_RULE = {
    "id": "SELFTEST-slot", "kind": "enricher_slot_gate", "forbid_absent": True,
    "min_leaves": 1, "min_slots": 1, "min_gated": 1,
    "en_tokens": [{"re": r"\bArcturel\w*", "cls": "E"}, {"re": r"\bArcturians?\b", "cls": "I"}],
    "cn_tokens": [{"re": "阿克图瑞尔", "cls": "E"}, {"re": "阿克图里安", "cls": "I"}],
}

# ⚑ 增强器槽位闸（第十九轮 Y6）的正反例。
#
# 这一套的重点不是「术语比得对不对」（那和 block_aligned_gate 同一套 `_class_re`，
# 已经在那边钉过），而是**这个类型特有的三件事**，每一件都是实测踩出来的：
#   ① 配对按 (动词, 目标) 而不是出现序号 —— 全库 4.5% 的增强器被中文语序搬过位；
#   ② 中文独有槽的回退方向是**反向**，且合法英文依据有两个来源（本叶英文 + 同目标别处的英文标签）；
#   ③ 方括号内部不判、`@Embed` 的 label/readaloud 参数要判。
SLOT_SELFTEST = [
    ("标签两侧一致 → 不响", None,
     _P("<p>@UUID[JournalEntry.a]{Arcturel} and @UUID[JournalEntry.b]{Arcturians}</p>",
        "<p>@UUID[JournalEntry.a]{阿克图瑞尔}与@UUID[JournalEntry.b]{阿克图里安人}</p>"), 0),
    ("⚑ **标签里单处串行** → 要响（这正是块级闸看不见的：split_blocks 把标签整条涂空）", None,
     _P("<p>@UUID[JournalEntry.a]{Arcturel} and @UUID[JournalEntry.b]{Arcturians}</p>",
        "<p>@UUID[JournalEntry.a]{阿克图里安}与@UUID[JournalEntry.b]{阿克图里安人}</p>"), 1),
    ("⚑ **中文把两个增强器搬了位** → 不响（按 (动词,目标) 配对；按出现序号配会造出 2 条幻影）", None,
     _P("<p>@UUID[JournalEntry.a]{Arcturel} shops of @UUID[JournalEntry.b]{Arcturians}</p>",
        "<p>@UUID[JournalEntry.b]{阿克图里安}的@UUID[JournalEntry.a]{阿克图瑞尔}商铺</p>"), 0),
    # ⚠ 这两条把 `min_gated` 显式放到 0：它们本来就该「一个类都判不到」，
    #   不放宽的话响的是反空转护栏而不是被测的那件事，测了个寂寞。
    ("方括号**内部**不判：目标串里出现术语也不看（既定约定要求照抄英文）", {"min_gated": 0},
     _P("<p>@UUID[Compendium.ember.x.Arcturel]{the city}</p>",
        "<p>@UUID[Compendium.ember.x.Arcturel]{这座城}</p>"), 0),
    ("`@Embed` 的 readaloud 参数是可见正文，要判 → 中文串行时要响", None,
     _P('<p>@Embed[Actor.z readaloud="These Arcturians are wary."]</p>',
        '<p>@Embed[Actor.z readaloud="这些阿克图瑞尔人心存戒备。"]</p>'), 1),
    ("反向闸 forbid_absent：英文槽没有的类中文槽冒出来 → 要响", None,
     _P("<p>@UUID[JournalEntry.a]{the Tradeway}</p>", "<p>@UUID[JournalEntry.a]{阿克图里安贸易道}</p>"), 1),
    ("英文有标签、中文裸增强器 → 不响（中文侧没字可判，不是缺陷）", {"min_gated": 0},
     _P("<p>@UUID[JournalEntry.a]{Arcturel}</p>", "<p>@UUID[JournalEntry.a]</p>"), 0),
    # ↓ 中文独有槽（英文裸 @UUID、Foundry 渲染目标名）的三条。回退**只做反向**。
    ("⚑ 中文独有槽：回退关着 → 不响", None,
     _P("<p>Arcturel: @UUID[JournalEntry.a] @UUID[JournalEntry.c]{Arcturel}</p>",
        "<p>阿克图瑞尔：@UUID[JournalEntry.a]{阿克图里安} @UUID[JournalEntry.c]{阿克图瑞尔}</p>"), 0),
    ("⚑ 中文独有槽：回退开着、类不在本叶英文里 → 要响", {"cn_only_leaf_fallback": True},
     _P("<p>Arcturel: @UUID[JournalEntry.a] @UUID[JournalEntry.c]{Arcturel}</p>",
        "<p>阿克图瑞尔：@UUID[JournalEntry.a]{阿克图里安} @UUID[JournalEntry.c]{阿克图瑞尔}</p>"), 1),
    ("⚑ 中文独有槽：类不在本叶英文里，但**同一目标在别处的英文标签**是这一类 → 不响",
     {"cn_only_leaf_fallback": True},
     [("e", "p.json", "j.A.text", "<p>Arcturel: @UUID[JournalEntry.a]</p>",
       "<p>阿克图瑞尔：@UUID[JournalEntry.a]{阿克图里安小饰品}</p>"),
      ("e", "p.json", "j.B.text", "<p>@UUID[JournalEntry.a]{Arcturian Trinkets}</p>",
       "<p>@UUID[JournalEntry.a]{阿克图里安小饰品}</p>")], 0),
    ("⚑ 正向回退是错的：本叶英文有 I 类，不代表每个中文标签都得带族名 → 不响",
     {"cn_only_leaf_fallback": True},
     _P("<p>Arcturians live here. @UUID[JournalEntry.a] @UUID[JournalEntry.c]{Arcturel}</p>",
        "<p>阿克图里安人住在这里。@UUID[JournalEntry.a]{月华花} @UUID[JournalEntry.c]{阿克图瑞尔}</p>"), 0),
    ("英文槽一个类都没命中 → min_gated 把空转抓出来", None,
     _P("<p>@UUID[JournalEntry.a]{Nothing}</p>", "<p>@UUID[JournalEntry.a]{什么都没有}</p>"), 1),
    ("登记过的槽不再报（登记的是已裁的合法例外）",
     {"except_slots": [{"path": "j.Some Page.text", "en": "Arcturel", "cn": "阿克图里安",
                        "why": "自检：这条真命中"}]},
     _P("<p>@UUID[JournalEntry.a]{Arcturel} @UUID[JournalEntry.b]{Arcturians}</p>",
        "<p>@UUID[JournalEntry.a]{阿克图里安} @UUID[JournalEntry.b]{阿克图里安人}</p>"), 0),
    ("⚑ 豁免一条都没命中 → 要响（死豁免闸在本类型这一侧也要有自己的用例）",
     {"except_slots": [{"path": "j.Nowhere.text", "en": "x", "cn": "y", "why": "自检：永远匹配不到"}]},
     _P("<p>@UUID[JournalEntry.a]{Arcturel}</p>", "<p>@UUID[JournalEntry.a]{阿克图瑞尔}</p>"), 1),
    ("把上限显式放宽到 1 → 同一条死豁免不再响（证明响的是「未命中」本身）",
     {"except_slots": [{"path": "j.Nowhere.text", "en": "x", "cn": "y", "why": "自检：永远匹配不到"}],
      "max_unused_exempt": 1},
     _P("<p>@UUID[JournalEntry.a]{Arcturel}</p>", "<p>@UUID[JournalEntry.a]{阿克图瑞尔}</p>"), 0),
]


# ⚑ 增强器**可见正文**覆盖闸（第二十二轮）的正反例。
#
# 这一套要钉的不是「术语比得对不对」（那在 enricher_slot_gate 那边钉过），而是
# **「这段朗读正文的中文有没有跟上英文」这件事本身判不判得动** —— 上一轮的教训正是
# 「文本进了口径、却没有任何判据能判它」，所以这里每一条信号都要有自己的正反例。
#
# 样本是照真库的形状造的：英文 3 句 / 99 字，中文 3 句 / 31 字（比值 0.313，
# 正落在真库那 48 段的 [0.263, 0.406] 区间中位附近）。
_RA_EN = ("The smoke clears over Dusktide. Ash falls across the broken road. "
          "Nothing moves in the ruins beyond.")
_RA_CN = "暮潮上空的烟尘散去。灰烬洒落在破碎的道路上。废墟深处一片死寂。"


def _RA(en_val, cn_val):
    """造一对只差 readaloud 值的叶。`None` = 该侧连这个参数都没有（裸增强器）。"""
    def side(v):
        return ("<p>@Embed[Actor.z]</p>" if v is None
                else f'<p>@Embed[Actor.z readaloud="{v}"]</p>')
    return _P(side(en_val), side(cn_val))


_TEXT_RULE = {
    "id": "SELFTEST-ratext", "kind": "enricher_text_coverage",
    "slots": ["param:readaloud"],
    "min_en_chars": 60, "min_ratio": 0.20, "min_sentence_frac": 0.75,
    "min_leaves": 1, "min_slots": 1, "min_gated": 1,
    # ⚠ 自检不读磁盘上的词表：内联两条锚点。真断言走 `glossary` 字段。
    "anchors": {"Dusktide": "暮潮", "Arcturel": "阿克图瑞尔"},
    "min_anchor_terms": 2, "min_anchor_slots": 1, "min_anchor_hits": 1,
}
# ⚠ 这两张「放宽护栏」表是必需的，理由与 SLOT_SELFTEST 里那两条同源：本来就该
#   「一个槽都不在册 / 一段都判不到」的用例，不放宽的话响的是反空转护栏本身，
#   于是被测的那件事测了个寂寞（第一版三条就是这么假红的）。
_NO_ANCHOR_GUARD = {"min_anchor_slots": 0, "min_anchor_hits": 0}
_NO_SLOT_GUARD = dict(_NO_ANCHOR_GUARD, min_slots=0, min_gated=0)

TEXT_COV_SELFTEST = [
    ("中文朗读正文完整 → 不响", None, _RA(_RA_EN, _RA_CN), 0),
    ("⚑ **中文朗读正文删空（整个参数没了）→ 要响**（这正是上一轮默认口径一声不响的那一刀）",
     None, _RA(_RA_EN, None), 1),
    ("⚑ 中文朗读正文是空串 → 要响", None, _RA(_RA_EN, ""), 1),
    ("⚑ 中文只剩没有汉字的占位 → 要响", None, _RA(_RA_EN, "..."), 1),
    ("⚑ **只删一半（中文只剩第一句）→ 要响**（比值 0.101 < 0.20，句数 1 < 2）",
     None, _RA(_RA_EN, "暮潮上空的烟尘散去。"), 1),
    ("⚑ 字数够、**句子塌成一句** → 要响（句数支单独立得住：比值 0.293 是过得了的）",
     None, _RA(_RA_EN, "暮潮上空的烟尘散去，灰烬洒落在破碎的道路上而废墟深处一片死寂"), 1),
    ("⚑ **本段定译专名缺失** → 要响（比值与句数都正常，只有锚点支能看见）",
     None, _RA(_RA_EN, "暮汐上空的烟尘散去。灰烬洒落在破碎的道路上。废墟深处一片死寂。"), 1),
    # ↓ 边界与不该判的东西
    ("英文段短于 min_en_chars → 不判（标签、一个词的 label 不该用比值判）",
     dict(_NO_ANCHOR_GUARD, min_gated=0),
     _RA("Smoke clears.", "烟散了。"), 0),
    ("`label=` 不在 slots 里 → 不判（本条只管整段朗读正文）", _NO_SLOT_GUARD,
     _P('<p>@Embed[Actor.z label="The Dusktide Warden of the Broken Road Beyond"]</p>',
        '<p>@Embed[Actor.z label="断路彼端的暮潮守望"]</p>'), 0),
    ("方括号里的机器参数（count= / classes= / 目标 id）不进本条", _NO_SLOT_GUARD,
     _P('<p>@Embed[Actor.99887766 count="5" classes="pf2e"]</p>',
        '<p>@Embed[Actor.99887766 count="5" classes="pf2e"]</p>'), 0),
    ("中文独有的可见正文（英文侧裸增强器）→ 不响（没有英文可比，不是缺陷）",
     dict(_NO_ANCHOR_GUARD, min_gated=0), _RA(None, _RA_CN), 0),
    # ↓ 反空转护栏本身的正反例
    ("⚑ 一段都没判到 → min_gated 把空转抓出来（只放宽 min_slots，响的必须是 min_gated 那一条）",
     dict(_NO_ANCHOR_GUARD, min_slots=0),
     _P("<p>@UUID[JournalEntry.a]{Dusktide}</p>", "<p>@UUID[JournalEntry.a]{暮潮}</p>"), 1),
    ("⚑ **锚点表拿不到 → 要响**（形态 (e)：输入缺失而判据照跑，不许装作全绿）",
     {"anchors": None, "glossary": "5-其他内容/NO-SUCH-GLOSSARY.json",
      "min_anchor_terms": 0, "min_anchor_slots": 0, "min_anchor_hits": 0},
     _RA(_RA_EN, _RA_CN), 1),
    ("⚑ 锚点一条都没命中 → min_anchor_hits 抓出来（词表在、但锚点规则被改坏时是这个形态）",
     {"anchors": {"Nowhere At All": "根本没有"}, "min_anchor_terms": 1},
     _RA(_RA_EN, _RA_CN), 2),
    ("登记过的槽不再报（登记的是已裁的合法例外）",
     {"except_slots": [{"path": "j.Some Page.text", "en_head": _RA_EN[:40],
                        "why": "自检：这条真命中"}]},
     _RA(_RA_EN, "暮潮上空的烟尘散去。"), 0),
    ("⚑ 豁免一条都没命中 → 要响（死豁免闸在本类型这一侧也要有自己的用例）",
     {"except_slots": [{"path": "j.Nowhere.text", "en_head": "x", "why": "自检：永远匹配不到"}]},
     _RA(_RA_EN, _RA_CN), 1),
    ("把上限显式放宽到 1 → 同一条死豁免不再响（证明响的是「未命中」本身）",
     {"except_slots": [{"path": "j.Nowhere.text", "en_head": "x", "why": "自检：永远匹配不到"}],
      "max_unused_exempt": 1},
     _RA(_RA_EN, _RA_CN), 0),
]


# --------------------------------------------------- twin_files 的副本树灵敏度回测
#
# 第二十五轮补。这一组用例的存在本身就是结论：**改之前 `R-selfcheck-twin` 做不了灵敏度回测**
# —— handler 里的 `ctx.repos.get(k, k)` 让它永远按 cwd 解析**真实树**，
# 往 `--root` 副本注入漂移它一声不响，还报「比对 1 对文件、0 违规」（空转形态 (g)）。
# 所以下面每一条都在**临时副本树**上跑，一条都不碰真实树；用例 2 更是**故意把 cwd 切到项目根**
# ——那里真实树的两份是干净的，旧实现在那儿必然报绿。它现在必须红。

_TWIN_A_REL = os.path.join("scripts", "ember-cn-selfcheck.mjs")
_TWIN_B_REL = os.path.join("selfcheck", "cn-selfcheck.mjs")


class _TwinCtx:
    """只提供 `.repos` 的最小 ctx —— twin_files 用不到叶/键通道。"""

    def __init__(self, repos):
        self.repos = repos


def _twin_make_tree(root, body_a, body_b):
    """按 `REPOS` 的真实目录名造一棵副本树，返回 repos 映射（与 `main()` 同构）。

    `body_* is None` 表示**这一侧根本不建**，用来回测输入缺失（形态 (e)）那一支。
    """
    repos = {name: os.path.join(root, rel) for name, rel in REPOS.items()}
    for key, sub, body in (("ember", _TWIN_A_REL, body_a),
                           ("crucible", _TWIN_B_REL, body_b)):
        if body is None:
            continue
        p = os.path.join(repos[key], sub)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(body)
    return repos


def _twin_rule():
    """取**发布中的那条规则**，而不是在自检里另写一份。

    这样「有人把 `repo_a` 改回目录名」会被自检当场抓住 —— 用例验的是真规则，
    不是自检自己的副本。（自检验一份和产线不同的配置，本身就是形态 (g)。）
    """
    try:
        rules = json.load(open(DEFAULT_RULES, encoding="utf-8"))
    except Exception:                                  # noqa: BLE001
        return None
    for r in rules.get("assertions", []):
        if r.get("kind") == "twin_files":
            return r
    return None


def run_twin_selftest():
    """返回 (失败条数, 总条数)。"""
    results = []                                       # (说明, ok, 备注)
    rule = _twin_rule()
    if rule is None:
        results.append(("前置：规则文件里找得到 twin_files 规则", False,
                        f"{DEFAULT_RULES} 里一条都没有 —— 用例无从跑起"))
    else:
        names = sorted({p["repo_a"] for p in rule["pairs"]}
                       | {p["repo_b"] for p in rule["pairs"]})
        results.append((f"前置：规则里的仓名都是 REPOS 的键（实得 {names}）",
                        all(n in REPOS for n in names),
                        "填目录名 = 空转形态 (g) 复发"))
        rule = dict(rule, min_pairs=1)

        same = b"export const SELFCHECK = 1;\n"
        drift = b"export const SELFCHECK = 2;\n"
        cwd0 = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            # 1 特异度：副本树两份相同 → 绿，且 detail 必须自陈「比对 1 对」（证明真比了）。
            b, d = a_twin_files(rule, _TwinCtx(
                _twin_make_tree(os.path.join(tmp, "clean"), same, same)))
            results.append((f"副本树两份相同 → 0 违规，且真的比成 1 对（{d}）",
                            not b and "比对 1 对文件" in d, ""))

            # 2 **灵敏度**：副本树注入漂移，cwd 切到项目根。必须红 ——
            #   而且**报出来的两个 hash 必须是副本树那两份的**。只断言「红了」不够：
            #   红也可能是因为它去比了真实树、而真实树恰好正在漂（第二十五轮当场就撞上了）。
            #   把 hash 钉死，才是「我读的是你给我的那棵树」的直接证据。
            import hashlib
            want = (hashlib.sha256(same).hexdigest()[:12],
                    hashlib.sha256(drift).hexdigest()[:12])
            ctx2 = _TwinCtx(_twin_make_tree(os.path.join(tmp, "drifted"), same, drift))
            try:
                os.chdir(ROOT)
                b, d = a_twin_files(rule, ctx2)
            finally:
                os.chdir(cwd0)
            msg2 = b[0][3] if len(b) == 1 else ""
            ok2 = "不是逐字节相同" in msg2 and all(h in msg2 for h in want)
            results.append((f"副本树注入漂移 → 必须响，且 hash 必须是**副本树**那两份"
                            f"（cwd 在项目根，不许去比真实树）（{d}）", ok2,
                            "" if ok2 else
                            f"期望 hash 前缀 {want[0]} / {want[1]}；实得「{msg2[-90:]}」"))

            # 3 **历史现场的原样复刻**：规则里填目录名（旧规则那种）+ 副本树注入漂移
            #   + cwd 在项目根。旧实现在这里会静默退化成裸相对路径、按 cwd 比**真实树**，
            #   报一句「比对 1 对文件、0 违规」交差。现在必须当场 KeyError 炸掉。
            bad_rule = dict(rule, pairs=[dict(rule["pairs"][0], repo_a=REPOS["ember"])])
            ctx3 = _TwinCtx(_twin_make_tree(os.path.join(tmp, "badname"), same, drift))
            try:
                os.chdir(ROOT)
                out3 = a_twin_files(bad_rule, ctx3)
                ok3 = False
                note3 = (f"居然没炸 —— `.get(k, k)` 那个静默兜底回来了；它自称「{out3[1]}」，"
                         f"而它比的根本不是副本树")
            except KeyError as exc:
                ok3, note3 = "不在 REPOS" in str(exc), str(exc)[:72]
            finally:
                os.chdir(cwd0)
            results.append(("仓名写成目录名（旧规则那种）+ 副本树有漂移 → 当场 KeyError 硬失败",
                            ok3, note3))

            # 4 形态 (e) 不许回潮：副本树缺一侧 → 报「没比成」并触发 min_pairs 空转闸。
            b, d = a_twin_files(rule, _TwinCtx(
                _twin_make_tree(os.path.join(tmp, "missing"), same, None)))
            results.append((f"副本树缺一侧 → 报「没比成」+ min_pairs 空转闸（{d}）",
                            len(b) == 2 and "比对 0 对文件" in d, ""))

    print("\n配对文件闸（twin_files）副本树回测：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\ntwin_files：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ========================================= translate_cases 的副本树灵敏度回测
#
# **做不到回测的闸不算闸。** 这条断言的驱动方式是「读被判的 .mjs → 交给 node 跑」，
# 所以回测的办法很直接：往**副本树**里放一份**故意改宽了的** `ember-hardcoded-cn.mjs`，
# 断言它必须变红；把那一处改回来，断言它必须变绿。
#
# ⚠ 变异用的搜索串**手写在下面**，不经任何改写脚本传 —— 里头的 `\d` 一转手就会被
#   Python 字符串吃掉，正则当场失效而主闸照样全绿（本项目登记的空转形态 (f)，亲手栽过两次）。
#   两个锚点都**必须存在**，不存在就是有人改了那条正则的写法 —— 这时回测**必须重新锚定**，
#   所以做成硬失败而不是「找不到就跳过」（跳过＝形态 (e)）。
_TRANS_SRC_REL = os.path.join("scripts", "ember-hardcoded-cn.mjs")
_TRANS_TIGHT = r"/^Result of (-?\d+[+-])$/"       # 现在的（收紧后）
_TRANS_WIDE = r"/^Result of (.+)$/"               # 第二十四轮那条会吃散文的
_TRANS_LEAF = "ARRANGEMENT_LEAVES[m[2]]"          # 现在的：查不到就整串不动
_TRANS_LEAF_LOOSE = "(ARRANGEMENT_LEAVES[m[2]] ?? m[2])"   # 旧的：查不到也照翻前缀

# 第二十七轮补的四个变异锚点（对应本轮新加的三段判据）。
# ⚠ 全部**写死成整段字面量**并由 `_trans_mut` 强制「不命中就 AssertionError」——
#   锚点漂了必须重新锚定，回测不到的闸不算闸。
_TRANS_PREFIXED_ONE = '{ en: "Location", cn: "地点", table: {} }'
_TRANS_PREFIXED_BAD = '{ en: "Location", cn: "位置", table: {} }'
_TRANS_PREFIXED_HEAD = "const PREFIXED = [\n"
_TRANS_PREFIXED_PLUS = ("const PREFIXED = [\n"
                        '  { en: "Nosuch Prefix", cn: "无此前缀", table: {} },\n')
# 第二十九轮：**不锚定串首/串尾**的正则（(N2) 机械近似探针那一道的回测锚点）。
# ⚠ 这两行是**手写落盘**的，不经任何改写脚本 —— 正则里的字符一经转手就可能失效
#   而主闸照样全绿（空转形态 (f)，本项目栽过两次）。
_TRANS_PATTERNS_HEAD = "const PATTERNS = [\n"
_TRANS_PATTERNS_UNANCHORED = ("const PATTERNS = [\n"
                              '  { re: /Zzqtest Marker/, cn: () => "自检：不锚定的正则" },\n')
_TRANS_NOTIFY_CN = "`镜子 ${m[1]} 不存在！`"
_TRANS_NOTIFY_CN_BAD = "`镜面 ${m[1]} 不存在！`"
# ⚠ 第二十八轮**重新锚定**：这两条原先是 `/^Mirror (.+) does not exist!$/` →
#   `/^(?:The )?Mirror (.+) does not exist!$/`。本轮被判文件把这条正则**收紧**成了
#   `[ab]\d{1,2}`（上游 BleakArchiveAreaMap.#MIRRORS 的键恰好是 a1…a23 / b1…b23，
#   ember.mjs:97859/98130/98430 三处互相印证），旧锚点不再存在 —— 于是 `_trans_mut`
#   如约当场 AssertionError 把整个 --selftest 炸停（**这正是它该做的**：
#   锚点漂了就必须重新锚定，不许「找不到就跳过」）。
#   重新锚定之后这条回测反而更有价值：放宽方向不再是随便造一个宽正则，
#   而是**改回本轮修掉的那一版**，反例 `Mirror Image does not exist!`（镜影术，标准法术名）
#   会当场被吃成「镜子 Image 不存在！」。
_TRANS_NOTIFY_RE = r"/^Mirror ([ab]\d{1,2}) does not exist!$/"
_TRANS_NOTIFY_RE_WIDE = "/^Mirror (.+) does not exist!$/"

# 第二十八轮：往叶子表里塞一个上游注册表里没有的名字，回测那道**不含阈值**的同源包含检查。
_TRANS_LEAVES_HEAD = "const ARRANGEMENT_LEAVES = {\n  ...ARRANGEMENTS,\n"
_TRANS_LEAVES_PLUS = ("const ARRANGEMENT_LEAVES = {\n  ...ARRANGEMENTS,\n"
                      '  "Nosuch Arrangement Name": "无此编排",\n')

# 形态 (h) 复现：上游 `ember.mjs` 的 `label:` 行缩进 +1（模拟上游换打包格式）。
# 复核实测这会让编排名 **219 → 196 静默下降**，而第二十六轮的 `min_labels=150` 够不着。
_TRANS_LABEL_ANCHOR = '\n      label: "'
_TRANS_LABEL_SHIFTED = '\n       label: "'
_TRANS_LABEL_N = 80                                # 只挪前 80 行 —— 要的是**部分**静默下降


class _TransCtx:
    """只提供 `.repos` 与 `.meta` 的最小 ctx —— translate_cases 用不到叶/键通道。"""

    def __init__(self, repos, meta=None):
        self.repos = repos
        self.meta = meta if meta is not None else {"lang_sources": _TRANS_REAL_SOURCES}


def _trans_mut(find, repl):
    """返回一个「必须命中才肯改」的变异器。"""
    def go(body):
        if find not in body:
            raise AssertionError(
                f"变异锚点不在被判文件里：{find!r} —— 那条正则被改写过，"
                f"灵敏度回测**必须重新锚定**。回测不到的闸不算闸，不许放着不管。")
        return body.replace(find, repl)
    return go


def _trans_make_tree(root, mutate=None, make_src=True):
    """按 `REPOS` 的真实目录名造一棵副本树，只放被判的那一个文件。

    `make_src=False` 表示**这一份根本不建**，用来回测输入缺失（形态 (e)）。
    """
    repos = {name: os.path.join(root, rel) for name, rel in REPOS.items()}
    if make_src:
        body = open(os.path.join(ROOT, REPOS["ember"], _TRANS_SRC_REL), encoding="utf-8").read()
        if mutate:
            body = mutate(body)
        p = os.path.join(repos["ember"], _TRANS_SRC_REL)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(body)
    return repos


def _trans_fake_upstream(root, mutate=None):
    """造一份**上游 ember.mjs 的副本**（可变异），返回可塞进 `meta.lang_sources` 的目录。

    上游基准按设计不受 `--root` 影响（见 `_trans_upstream` 的 docstring），所以要回测
    「上游换了打包格式导致抽取器静默少抠」这类形态 (h)，只能换 `lang_sources` 指向一份副本。
    """
    src = os.path.join(os.path.expandvars(_TRANS_REAL_SOURCES.get("ember", "")),
                       "scripts", "ember.mjs")
    body = open(src, encoding="utf-8").read()
    if mutate:
        body = mutate(body)
    dst = os.path.join(root, "scripts", "ember.mjs")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(body)
    return root


def _trans_rule():
    """取**发布中的那条规则**，而不是在自检里另写一份。

    自检验一份和产线不同的配置，本身就是形态 (g)（见 `_twin_rule` 的同一段理由）。
    """
    try:
        rules = json.load(open(DEFAULT_RULES, encoding="utf-8"))
    except Exception:                                  # noqa: BLE001
        return None
    for r in rules.get("assertions", []):
        if r.get("kind") == "translate_cases":
            return r
    return None


def _trans_real_sources():
    try:
        return (json.load(open(DEFAULT_RULES, encoding="utf-8")).get("meta") or {}
                ).get("lang_sources") or {}
    except Exception:                                  # noqa: BLE001
        return {}


_TRANS_REAL_SOURCES = _trans_real_sources()


def run_translate_selftest():
    """返回 (失败条数, 总条数)。"""
    results = []                                       # (说明, ok, 备注)
    rule = _trans_rule()
    if rule is None:
        results.append(("前置：规则文件里找得到 translate_cases 规则", False,
                        f"{DEFAULT_RULES} 里一条都没有 —— 用例无从跑起"))
    else:
        results.append((f"前置：规则里的仓名是 REPOS 的键（实得 {rule.get('repo')!r}）",
                        rule.get("repo") in REPOS,
                        "填目录名 = 空转形态 (g) 复发（见 `_twin_repo_dir`）"))
        with tempfile.TemporaryDirectory() as tmp:
            # 1 **特异度**：副本树里放的就是原样的被判文件 → 必须 0 违规，
            #   且 detail 必须自陈跑了多少条 / 从上游抠了多少个编排名（证明它真跑了，
            #   而不是「没跑成也返回空」）。
            b, d = a_translate_cases(rule, _TransCtx(
                _trans_make_tree(os.path.join(tmp, "clean"))))
            # ⚠ detail 里必须同时报出**三张表的覆盖分数** —— 第二十六轮闸名写着 PREFIXED
            #   却只覆盖 2/19，正是因为没有任何地方把覆盖率打出来给人看。
            ok1 = (not b and "现抠" in d and "编排名" in d
                   and "覆盖 PREFIXED" in d and "覆盖 PATTERNS" in d
                   and "覆盖 NOTIFICATION_PATTERNS" in d and "空白折叠回退命中" in d)
            results.append((f"副本树放原样的被判文件 → 0 违规，且 detail 自陈跑了什么"
                            f"（含三张表的覆盖分数）（{d}）",
                            ok1, "" if ok1 else f"违规 {len(b)} 处；detail=「{d}」"))

            # 2 **灵敏度①**：把 `^Result of (-?\d+[+-])$` 改回第二十四轮那条宽的 →
            #   必须变红，而且必须点名那条反例（只断言「红了」不够 ——
            #   红也可能是因为它根本没跑成，那是另一回事）。
            b, d = a_translate_cases(rule, _TransCtx(_trans_make_tree(
                os.path.join(tmp, "wide"), _trans_mut(_TRANS_TIGHT, _TRANS_WIDE))))
            hit = [x for x in b if "investigation" in str(x[2])]
            ok2 = bool(hit) and any("negative" in x[3] for x in hit)
            results.append((f"把 `^Result of (.+)$` 改回宽的 → 必须变红并点名那条反例"
                            f"（违规 {len(b)} 处）", ok2,
                            "" if ok2 else f"实得：{[x[2] for x in b][:4]}"))

            # 3 **改回来必须变绿**：在同一份变宽的正文上再改回去（等价于原样），
            #   证明上一条的红确实是那一处改动造成的，而不是副本树本身有问题。
            def _back(body):
                return _trans_mut(_TRANS_WIDE, _TRANS_TIGHT)(
                    _trans_mut(_TRANS_TIGHT, _TRANS_WIDE)(body))
            b, d = a_translate_cases(rule, _TransCtx(
                _trans_make_tree(os.path.join(tmp, "restored"), _back)))
            results.append((f"改宽之后再改回来 → 必须变绿（{d}）", not b,
                            "" if not b else f"仍有 {len(b)} 处：{[x[2] for x in b][:3]}"))

            # 4 **灵敏度②**：把音景那条的查表兜底放宽回旧写法（查不到也照翻前缀）→
            #   必须变红，且**两段都要响**：反例段点名 `Music: my custom playlist`，
            #   全量护栏段报「留英集合对不上」（8 个应留英的会全部被翻动）。
            b, d = a_translate_cases(rule, _TransCtx(_trans_make_tree(
                os.path.join(tmp, "loose"), _trans_mut(_TRANS_LEAF, _TRANS_LEAF_LOOSE))))
            neg_hit = any("my custom playlist" in str(x[2]) for x in b)
            arr_hit = any("留英" in str(x[2]) or "留英" in str(x[3]) for x in b)
            ok4 = neg_hit and arr_hit
            results.append((f"把 ARRANGEMENT_LEAVES 的查表兜底放宽回旧写法 → 反例段与"
                            f"全量护栏段**都**要响（违规 {len(b)} 处）", ok4,
                            "" if ok4 else f"反例段响={neg_hit} 全量段响={arr_hit}；"
                                           f"{[x[2] for x in b][:4]}"))

            # 5 形态 (e)：被判文件在副本树里根本不存在 → 报「没跑成」，不是通过。
            b, d = a_translate_cases(rule, _TransCtx(
                _trans_make_tree(os.path.join(tmp, "missing"), make_src=False)))
            ok5 = bool(b) and d == "没跑成" and "没跑成" in b[0][3]
            results.append((f"副本树里没有被判文件 → 报「没跑成」而不是通过（{d}）", ok5,
                            "" if ok5 else f"实得 detail=「{d}」 bad={b[:1]}"))

            # 6 形态 (e)：node 取不到 → 必须当场失败并说明，不许静默跳过。
            b, d = a_translate_cases(dict(rule, node_bin="node-that-does-not-exist"),
                                     _TransCtx(_trans_make_tree(os.path.join(tmp, "nonode"))))
            ok6 = bool(b) and d == "没跑成" and "找不到" in b[0][3]
            results.append((f"PATH 里没有 node → 报「没跑成」而不是静默跳过（{d}）", ok6,
                            "" if ok6 else f"实得 detail=「{d}」 bad={b[:1]}"))

            # 7 **形态 (h)**：喂输入的那一半坏了 —— 上游 `ember.mjs` 换成一个空文件。
            #   判据本身没坏，坏的是模拟输入的来源。必须当场失败（抠不到 soundscapes 注册表），
            #   **绝不许**因为「编排名一个都没抠到 ⇒ 没有可违反的」而报绿。
            fake_up = os.path.join(tmp, "fakeupstream", "ember")
            os.makedirs(os.path.join(fake_up, "scripts"), exist_ok=True)
            with open(os.path.join(fake_up, "scripts", "ember.mjs"), "w", encoding="utf-8") as fh:
                fh.write("// 空的：上游打包形状变了的模拟\n")
            b, d = a_translate_cases(rule, _TransCtx(
                _trans_make_tree(os.path.join(tmp, "h")),
                meta={"lang_sources": dict(_TRANS_REAL_SOURCES, ember=fake_up)}))
            ok7 = bool(b) and any("soundscapes" in str(x[3]) for x in b)
            results.append((f"上游 ember.mjs 抠不到编排名 → 当场失败（形态 (h)：坏的是"
                            f"喂输入的那一半）（{d}）", ok7,
                            "" if ok7 else f"实得 detail=「{d}」 bad={[x[3][:80] for x in b][:2]}"))

            # 8 形态 (g)：仓名写成目录名 → 当场 KeyError，不许静默按 cwd 解析真实树。
            cwd0 = os.getcwd()
            try:
                os.chdir(ROOT)
                a_translate_cases(dict(rule, repo=REPOS["ember"]), _TransCtx(
                    _trans_make_tree(os.path.join(tmp, "badname"),
                                     _trans_mut(_TRANS_TIGHT, _TRANS_WIDE))))
                ok8, note8 = False, "居然没炸 —— `.get(k, k)` 那个静默兜底回来了"
            except KeyError as exc:
                ok8, note8 = "不在 REPOS" in str(exc), str(exc)[:72]
            finally:
                os.chdir(cwd0)
            results.append(("仓名写成目录名 + 副本树已改宽 → 当场 KeyError 硬失败", ok8, note8))

            # ---------------------------------------------- 第二十七轮新加的五条
            # 9 **灵敏度③（PREFIXED）**：把 `Location` 那条的中文前缀改坏 → 必须变红。
            #   这一条是**本轮存在的理由**：第二十六轮把 Knowledge / Location / Quest /
            #   Talent / Rarity 五条逐个改坏重跑，闸 5/5 全绿（PREFIXED 只覆盖 2/19）。
            b, d = a_translate_cases(rule, _TransCtx(_trans_make_tree(
                os.path.join(tmp, "prefix"),
                _trans_mut(_TRANS_PREFIXED_ONE, _TRANS_PREFIXED_BAD))))
            ok9 = any("Location:" in str(x[2]) for x in b) and any(
                "positive" in str(x[3]) for x in b)
            results.append((f"把 PREFIXED 里 `Location` 的中文前缀改坏 → 必须变红并点名"
                            f"那条正例（违规 {len(b)} 处）", ok9,
                            "" if ok9 else f"实得：{[x[2] for x in b][:4]}"))

            # 10 **灵敏度④（NOTIFICATION_PATTERNS 译文）**：改坏一条译文 → 通知正例段必须响。
            b, d = a_translate_cases(rule, _TransCtx(_trans_make_tree(
                os.path.join(tmp, "notifycn"),
                _trans_mut(_TRANS_NOTIFY_CN, _TRANS_NOTIFY_CN_BAD))))
            ok10 = any("Mirror a1" in str(x[2]) for x in b) and any(
                "notify_positive" in str(x[3]) for x in b)
            results.append((f"把一条 NOTIFICATION_PATTERNS 的译文改坏 → 通知正例段必须响"
                            f"（违规 {len(b)} 处）", ok10,
                            "" if ok10 else f"实得：{[(x[2], x[3][:40]) for x in b][:4]}"))

            # 11 **灵敏度⑤（NOTIFICATION_PATTERNS 定义域）**：把镜子那条放宽回第二十八轮
            #    收紧**之前**的 `(.+)` 版 → 通知反例段必须响，且必须点名
            #    `Mirror Image does not exist!`（Mirror Image = 镜影术，别的模块真会发的句子，
            #    宽正则会把它整句重构成「镜子 Image 不存在！」）。
            #    这一段守的正是那句订正：通知通道挂在**全局** ui.notifications.notify 上。
            b, d = a_translate_cases(rule, _TransCtx(_trans_make_tree(
                os.path.join(tmp, "notifywide"),
                _trans_mut(_TRANS_NOTIFY_RE, _TRANS_NOTIFY_RE_WIDE))))
            ok11 = any("Mirror Image" in str(x[2]) for x in b) and any(
                "notify_negative" in str(x[3]) for x in b)
            results.append((f"把一条 NOTIFICATION_PATTERNS 放宽到能吃别的模块的通知 →"
                            f"通知反例段必须响（违规 {len(b)} 处）", ok11,
                            "" if ok11 else f"实得：{[x[2] for x in b][:4]}"))

            # 12 **覆盖闸本身**：往 PREFIXED 里插一条没有任何用例的新前缀 →
            #    覆盖归因必须报「有条目没被任何正例触到」。没有这一条，覆盖闸自己就是空转的。
            b, d = a_translate_cases(rule, _TransCtx(_trans_make_tree(
                os.path.join(tmp, "cover"),
                _trans_mut(_TRANS_PREFIXED_HEAD, _TRANS_PREFIXED_PLUS))))
            ok12 = any("未被任何正例触到" in str(x[2]) for x in b)
            results.append((f"往 PREFIXED 插一条没有用例的新前缀 → 覆盖归因必须报缺口"
                            f"（违规 {len(b)} 处）", ok12,
                            "" if ok12 else f"实得：{[x[2] for x in b][:4]}"))

            # 13 **形态 (h) 复现**：上游 `label:` 行缩进 +1（模拟上游换打包格式）。
            #    复核实测第二十六轮那版会 **219 → 196 静默下降而闸仍 0 违规**。
            #    现在 min_labels=210 + labels_recorded=219 两道都贴着现值，必须变红。
            def _shift(body):
                if body.count(_TRANS_LABEL_ANCHOR) < _TRANS_LABEL_N:
                    raise AssertionError(
                        f"上游里 {_TRANS_LABEL_ANCHOR!r} 不足 {_TRANS_LABEL_N} 处 —— "
                        f"上游打包形状变了，这条 (h) 回测**必须重新锚定**")
                return body.replace(_TRANS_LABEL_ANCHOR, _TRANS_LABEL_SHIFTED, _TRANS_LABEL_N)
            b, d = a_translate_cases(rule, _TransCtx(
                _trans_make_tree(os.path.join(tmp, "h2")),
                meta={"lang_sources": dict(
                    _TRANS_REAL_SOURCES,
                    ember=_trans_fake_upstream(os.path.join(tmp, "upshift"), _shift))}))
            ok13 = any("编排名" in str(x[3]) for x in b)
            results.append((f"上游 `label:` 行缩进 +1（形态 (h) 复现）→ 编排名静默下降必须"
                            f"变红（违规 {len(b)} 处；detail=「{d[-90:]}」）", ok13,
                            "" if ok13 else f"实得：{[x[3][:90] for x in b][:3]}"))

            # -------------------------------------- 第二十八轮：**作弊路径 B 的回测**
            # 复核实测：把 `coverage.min_prefixed` 从 19 改回 2、`min_negative` /
            # `min_positive` 放到 1，**其余一个字节不动 → 违规 0，闸变绿**。
            # ⇒ 下面五条逐个复现那个作弊动作，每一条都必须**仍然红**。
            # ⚠ 变异的是**规则**（不是被判文件）：这正是那条作弊路径的形状 ——
            #   库和判据都没动，只把规则里的自报数调松。
            clean_tree = _trans_make_tree(os.path.join(tmp, "b_clean"))

            def _rule_mut(**over):
                r = copy.deepcopy(rule)
                for path, val in over.items():
                    node, *rest = path.split("__")
                    if rest:
                        r.setdefault(node, {})[rest[0]] = val
                    else:
                        r[node] = val
                return r

            # 14 把三张表的表长记录值调松（复核用的就是这一招）→ 必须红。
            b, d = a_translate_cases(
                _rule_mut(recorded__prefixed_size=2), _TransCtx(clean_tree))
            ok14 = any("prefixed_size" in str(x[2]) for x in b) and any(
                "低于现算值" in str(x[3]) for x in b)
            results.append((f"把 recorded.prefixed_size 从 19 调到 2（作弊路径 B）→ 必须"
                            f"仍然红（违规 {len(b)} 处）", ok14,
                            "" if ok14 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

            # 15 把用例条数下限调到 1 → 必须红（现算层看的是「规则里现在真写着几条」）。
            b, d = a_translate_cases(
                _rule_mut(recorded__negative=1, recorded__positive=1), _TransCtx(clean_tree))
            ok15 = (any("negative" in str(x[2]) for x in b)
                    and any("positive" in str(x[2]) for x in b))
            results.append((f"把 recorded.negative / positive 调到 1（作弊路径 B）→ 必须"
                            f"仍然红（违规 {len(b)} 处）", ok15,
                            "" if ok15 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

            # 16 干脆把整个 `recorded` 块删掉 → 必须红（两层里少一层就当场判）。
            r16 = copy.deepcopy(rule)
            r16.pop("recorded", None)
            b, d = a_translate_cases(r16, _TransCtx(clean_tree))
            ok16 = any("recorded" in str(x[2]) for x in b) and any(
                "两层" in str(x[3]) for x in b)
            results.append((f"把整个 recorded 块删掉 → 必须红（不许把「历史层」当冗余删掉）"
                            f"（违规 {len(b)} 处）", ok16,
                            "" if ok16 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

            # 17 编排名那两道数（min_labels 与 labels_recorded）**一起**调松 → 仍必须红：
            #    现算层看的是上游这次真抠出多少（219），记录值落在后面就是被人动过。
            b, d = a_translate_cases(
                _rule_mut(arrangements=dict(rule["arrangements"], min_labels=2,
                                            labels_recorded=2)), _TransCtx(clean_tree))
            ok17 = any("编排名" in str(x[3]) for x in b)
            results.append((f"把 min_labels 与 labels_recorded **一起**调到 2 → 必须仍然红"
                            f"（违规 {len(b)} 处）", ok17,
                            "" if ok17 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

            # 18 **不含阈值那一道**的回测：往 ARRANGEMENT_LEAVES 里塞一个上游没有的叶子 →
            #    「叶子表 ⊆ 上游现抠的编排名」必须响。这一道调不松（规则里没有对应的数），
            #    是形态 (h) 的主防线；上面那些数只是第二道。
            b, d = a_translate_cases(rule, _TransCtx(_trans_make_tree(
                os.path.join(tmp, "leaf"),
                _trans_mut(_TRANS_LEAVES_HEAD, _TRANS_LEAVES_PLUS))))
            ok18 = any("查无此名的叶子" in str(x[2]) for x in b)
            results.append((f"往 ARRANGEMENT_LEAVES 塞一个上游没有的叶子 → 同源包含检查必须"
                            f"响（违规 {len(b)} 处）", ok18,
                            "" if ok18 else f"实得：{[x[2] for x in b][:4]}"))

            # ---------------- 第二十九轮：**反例侧结构护栏**的回测（作弊路径 C 的一部分）
            # 复核实测的原始现场：「清空 `negative` + 把 `recorded.negative` 改 0」
            # 两处协同 → **全绿**。下面 19 / 20 逐字复现那两个动作，必须仍然红。
            # ⚠ 变异的又是**规则**，不是被判文件 —— 这条路径的形状就是「库和判据都没动」。

            # 19 清空 negative 并把记录值一起改成 0 → 必须仍然红（完美匹配那一道）。
            b, d = a_translate_cases(
                _rule_mut(negative=[], recorded__negative=0), _TransCtx(clean_tree))
            ok19 = any("没有专属近似反例" in str(x[2]) for x in b)
            results.append((f"清空 negative + recorded.negative 改 0（复核实测能全绿的那两步）"
                            f"→ 必须仍然红（违规 {len(b)} 处）", ok19,
                            "" if ok19 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

            # 20 通知反例同理 —— 这条通道挂在全局 ui.notifications.notify 上，更要紧。
            b, d = a_translate_cases(
                _rule_mut(notify_negative=[], recorded__notify_negative=0),
                _TransCtx(clean_tree))
            ok20 = any("没有专属近似反例" in str(x[2]) for x in b)
            results.append((f"清空 notify_negative + 记录值改 0 → 必须仍然红"
                            f"（违规 {len(b)} 处）", ok20,
                            "" if ok20 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

            # 21 **条数一点没变、内容换成无关串** → 必须仍然红。
            #    这一条证明护栏是**结构的**、不是数出来的：`recorded.negative` 那个数
            #    一个字节都没动，反例条数也分毫不差，只是没有一条挨着表项 ——
            #    「凑够条数」这条路走不通，逐条归因才作数。
            #    ⚠ 反过来也说明「只删一条**冗余**反例」不会红，那是**对的**：
            #      库里本来就有好几条 `Result of …`，删掉其中一条，剩下的照样能认领
            #      那条表项（完美匹配会自动改派）。护栏保的是「每条表项都有人认领」，
            #      不是「一条反例都不许动」。
            b, d = a_translate_cases(
                _rule_mut(negative=[f"Zzq filler {i}" for i in range(len(rule["negative"]))]),
                _TransCtx(clean_tree))
            ok21 = any("没有专属近似反例" in str(x[2]) for x in b)
            results.append((f"反例条数一条不少、内容全换成无关串（记录值一个字节没动）→ "
                            f"必须仍然红（违规 {len(b)} 处）", ok21,
                            "" if ok21 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

            # 22 **(N2) 机械近似探针**：往 PATTERNS 插一条**不锚定串首/串尾**的正则 →
            #    探针 `Zzq <骨架> Zzq` 会被吃掉，必须点名。这一道**不靠人维护**：
            #    表里加一条它就多一条探针，人一条用例都不用写。
            b, d = a_translate_cases(rule, _TransCtx(_trans_make_tree(
                os.path.join(tmp, "unanchored"),
                _trans_mut(_TRANS_PATTERNS_HEAD, _TRANS_PATTERNS_UNANCHORED))))
            ok22 = any("机械近似探针被吃掉" in str(x[3]) for x in b)
            results.append((f"往 PATTERNS 插一条不锚定串首/串尾的正则 → 机械近似探针必须"
                            f"当场响（违规 {len(b)} 处）", ok22,
                            "" if ok22 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

    print("\n正则译文用例闸（translate_cases）副本树回测：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\ntranslate_cases：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ================================== source_literal 的副本树回测（第二十七轮 D）
#
# 与 twin_files / translate_cases 同一套路：**取发布中的那条规则**（不在自检里另写一份，
# 那是形态 (g)），在副本树上逐项注入，确认每一支都真的会响。


class _SrcLitCtx:
    """只提供 `.repos` 的最小 ctx —— source_literal 不走叶/键通道。"""

    def __init__(self, repos):
        self.repos = repos


def _srclit_rule():
    try:
        rules = json.load(open(DEFAULT_RULES, encoding="utf-8"))
    except Exception:                                  # noqa: BLE001
        return None
    for r in rules.get("assertions", []):
        if r.get("kind") == "source_literal":
            return r
    return None


def _srclit_make_tree(root, rule, mutate=None, make=True):
    """按规则里登记的 (repo, path) 造副本树，只放那两份面板文件。"""
    repos = {name: os.path.join(root, rel) for name, rel in REPOS.items()}
    if make:
        for f in rule["files"]:
            src = os.path.join(ROOT, REPOS[f["repo"]], *f["path"].split("/"))
            body = open(src, encoding="utf-8").read()
            if mutate:
                body = mutate(body)
            dst = os.path.join(repos[f["repo"]], *f["path"].split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(body)
    return repos


def run_srclit_selftest():
    """返回 (失败条数, 总条数)。"""
    results = []
    rule = _srclit_rule()
    if rule is None:
        results.append(("前置：规则文件里找得到 source_literal 规则", False,
                        f"{DEFAULT_RULES} 里一条都没有 —— 用例无从跑起"))
    else:
        want = rule["require"][0]
        # 「档名漂回键活性」的两种写法：整段换回旧档名 / 只把档名里塞回那三个字。
        old_name = want.replace("D 上游字面量存在性（仅供人工复核）", "D 键活性")
        other_name = want.replace("D 上游字面量存在性（仅供人工复核）", "D 上游语料对照")
        with tempfile.TemporaryDirectory() as tmp:
            # 1 特异度：原样两份 → 0 违规，且 detail 自陈读了几个文件、跑了几条判据。
            b, d = a_source_literal(rule, _SrcLitCtx(
                _srclit_make_tree(os.path.join(tmp, "clean"), rule)))
            ok1 = (not b and "读 2 个源码文件" in d and "条字面量判据" in d)
            results.append((f"副本树放原样的两份面板 → 0 违规，且 detail 自陈读了什么（{d}）",
                            ok1, "" if ok1 else f"违规 {len(b)} 处；detail=「{d}」"))

            # 2 灵敏度①：档名漂回「键活性」→ require 与 forbid_re **两支都要响**，
            #   而且两份面板**各报一次**（两份一起漂回去，twin_files 是察觉不到的）。
            b, d = a_source_literal(rule, _SrcLitCtx(_srclit_make_tree(
                os.path.join(tmp, "old"), rule, lambda s: s.replace(want, old_name))))
            req_hit = any("必须逐字符存在" in str(x[3]) for x in b)
            fb_hit = any("被禁的写法" in str(x[3]) for x in b)
            ok2 = req_hit and fb_hit and len(b) == 4
            results.append((f"档名漂回「键活性」→ require 与 forbid_re 都要响、两份各报一次"
                            f"（违规 {len(b)} 处）", ok2,
                            "" if ok2 else f"require 响={req_hit} forbid 响={fb_hit}；"
                                           f"{[x[3][:44] for x in b][:4]}"))

            # 3 **两支各管各的**：档名改成别的（不含「键活性」）→ 只有 require 响。
            #   没有这一条，「forbid_re 永远跟着 require 一起响」与「forbid_re 其实写坏了」
            #   分不开（形态 (a)/(f)）。
            b, d = a_source_literal(rule, _SrcLitCtx(_srclit_make_tree(
                os.path.join(tmp, "other"), rule, lambda s: s.replace(want, other_name))))
            ok3 = (len(b) == 2 and all("必须逐字符存在" in str(x[3]) for x in b))
            results.append((f"档名改成别的（不含「键活性」）→ 只有 require 响"
                            f"（违规 {len(b)} 处）", ok3,
                            "" if ok3 else f"实得：{[x[3][:44] for x in b][:4]}"))

            # 4 形态 (e)：两份都不在 → 报「没判成」+ min_files 空转闸。
            b, d = a_source_literal(rule, _SrcLitCtx(
                _srclit_make_tree(os.path.join(tmp, "missing"), rule, make=False)))
            # 实得 4 处：两份各报一次「文件不在」+ min_files + min_checks（一条判据都没跑成）。
            ok4 = (len(b) == 4 and sum("没判成" in str(x[3]) for x in b) == 2
                   and any("这条断言在空转" in str(x[3]) for x in b) and "读 0 个" in d)
            results.append((f"副本树里两份都不在 → 报「没判成」+ min_files 空转闸（{d}）",
                            ok4, "" if ok4 else f"实得 {len(b)} 处：{[x[3][:44] for x in b][:3]}"))

            # 5 形态 (d)：require / forbid_re 都被清空 → min_checks 必须响，
            #   不许以「0 违规」通过（判据在跑，但已经没有任何判据可跑了）。
            b, d = a_source_literal(dict(rule, require=[], forbid_re=[]), _SrcLitCtx(
                _srclit_make_tree(os.path.join(tmp, "empty"), rule)))
            ok5 = bool(b) and any("只跑了 0 条字面量判据" in str(x[3]) for x in b)
            results.append((f"require / forbid_re 都清空 → min_checks 空转闸必须响（{d}）",
                            ok5, "" if ok5 else f"实得：{[x[3][:60] for x in b][:2]}"))

    print("\n源码字面量闸（source_literal）副本树回测：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\nsource_literal：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ================================ 自检面板现跑闸的回测（第二十八轮 ②）
#
# 同 translate_cases 的套路：**取发布中的那条规则**（不在自检里另写一份，那是形态 (g)），
# 在副本树上注入变异，确认每一支真的会响。
# ⚠ 两个变异开关（`drop_corpus` / `keep_table_fraction`）实现在 node 侧执行体里，
#   方向都是**只会让结果更差**（少抓语料 ⇒ miss 涨；摘掉表项 ⇒ 覆盖掉），
#   所以它们进不了「调一下开关就变绿」的作弊路径 —— 这是有意选的方向。


def _panel_rule():
    try:
        rules = json.load(open(DEFAULT_RULES, encoding="utf-8"))
    except Exception:                                  # noqa: BLE001
        return None
    for r in rules.get("assertions", []):
        if r.get("kind") == "panel_liveness":
            return r
    return None


def _panel_make_tree(root, rule, make=True):
    """按 `REPOS` 的真实目录名造副本树，只放面板与硬编码表这两份。"""
    repos = {name: os.path.join(root, rel) for name, rel in REPOS.items()}
    if make:
        for repo, rel in ((rule["repo"], rule["panel"]),
                          (rule.get("tables_repo", rule["repo"]), rule["tables_src"])):
            src = os.path.join(ROOT, REPOS[repo], *rel.split("/"))
            dst = os.path.join(repos[repo], *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(open(src, encoding="utf-8").read())
    return repos


def run_panel_selftest():
    """返回 (失败条数, 总条数)。"""
    results = []
    rule = _panel_rule()
    if rule is None:
        results.append(("前置：规则文件里找得到 panel_liveness 规则", False,
                        f"{DEFAULT_RULES} 里一条都没有 —— 面板无从跑起"))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            clean = _panel_make_tree(os.path.join(tmp, "clean"), rule)

            # 1 特异度：副本树放原样两份 → 0 违规，且 detail 必须自陈**现跑**了什么。
            b, d = a_panel_liveness(rule, _TransCtx(clean))
            ok1 = (not b and "现跑面板" in d and "假阴性对照" in d and "逐行之和" in d)
            results.append((f"副本树放原样的面板 → 0 违规，且 detail 自陈现跑了什么（{d}）",
                            ok1, "" if ok1 else f"违规 {len(b)} 处：{[x[3][:80] for x in b][:3]}"))

            # 2 **让 fetch 少抓一份语料**（任务书点名的回测）：上游脚本抓不着 →
            #   miss 必须暴涨、fetchOk 必须掉，两侧都得响。
            b, d = a_panel_liveness(
                dict(rule, mutate={"drop_corpus": ["ember.mjs"]}), _TransCtx(clean))
            ok2 = (any("missDistinct" in str(x[2]) for x in b)
                   and any("fetchOk" in str(x[2]) for x in b))
            results.append((f"让 fetch 少抓一份语料（ember.mjs）→ miss 侧与覆盖侧**都**要响"
                            f"（违规 {len(b)} 处）", ok2,
                            "" if ok2 else f"实得：{[(x[2], x[3][:50]) for x in b][:4]}"))

            # 3 **把每张表摘掉一半**（任务书点名的回测）：核到的键数必须掉下记录值。
            b, d = a_panel_liveness(
                dict(rule, mutate={"keep_table_fraction": 0.5}), _TransCtx(clean))
            ok3 = (any("checkedDistinct" in str(x[2]) for x in b)
                   and any("rawChecked" in str(x[2]) for x in b))
            results.append((f"把每张表摘掉一半 → 覆盖侧必须响（违规 {len(b)} 处）", ok3,
                            "" if ok3 else f"实得：{[(x[2], x[3][:50]) for x in b][:4]}"))

            # 4 形态 (d)：假阴性对照被清空 → 必须红。
            #   这一道是本条**最扛事**的一条（匹配器变恒真时只有它会响），
            #   所以「它自己被清空」也必须当场判，不许静默少跑一段。
            b, d = a_panel_liveness(dict(rule, fakes={}), _TransCtx(clean))
            ok4 = any("假阴性对照" in str(x[2]) for x in b)
            results.append((f"把假阴性对照清空 → 必须红（违规 {len(b)} 处）", ok4,
                            "" if ok4 else f"实得：{[(x[2], x[3][:50]) for x in b][:3]}"))

            # 5 档名记错 → 必须红（钉的是**运行时真的用了那个档名**）。
            b, d = a_panel_liveness(dict(rule, section="D 键活性"), _TransCtx(clean))
            ok5 = any("档名" in str(x[2]) for x in b)
            results.append((f"把记录的档名改成「D 键活性」→ 必须红（违规 {len(b)} 处）", ok5,
                            "" if ok5 else f"实得：{[(x[2], x[3][:50]) for x in b][:3]}"))

            # 6 形态 (e)：副本树里没有面板 → 报「没跑成」，不是通过。
            b, d = a_panel_liveness(rule, _TransCtx(
                _panel_make_tree(os.path.join(tmp, "missing"), rule, make=False)))
            ok6 = bool(b) and d == "没跑成"
            results.append((f"副本树里没有面板 → 报「没跑成」而不是通过（{d}）", ok6,
                            "" if ok6 else f"实得 detail=「{d}」 bad={b[:1]}"))

            # 7 形态 (e)：node 取不到 → 当场失败并说明。
            b, d = a_panel_liveness(dict(rule, node_bin="node-that-does-not-exist"),
                                    _TransCtx(clean))
            ok7 = bool(b) and d == "没跑成" and "找不到" in b[0][3]
            results.append((f"PATH 里没有 node → 报「没跑成」而不是静默跳过（{d}）", ok7,
                            "" if ok7 else f"实得 detail=「{d}」 bad={b[:1]}"))

            # 8 **形态 (h)**：喂输入的那一半 —— 上游安装目录指到一个尾巴对不上的地方，
            #   必须当场判「没跑成」，绝不许「语料一份没抓到 ⇒ 没有可违反的 ⇒ 绿」。
            b, d = a_panel_liveness(rule, _TransCtx(
                clean, meta={"lang_sources": {"ember": os.path.join(tmp, "nowhere")}}))
            ok8 = bool(b) and d == "没跑成" and "EMBER_ROOT" in b[0][3]
            results.append((f"上游安装目录的尾巴与面板的 EMBER_ROOT 对不上 → 当场判没跑成"
                            f"（形态 (h)）（{d}）", ok8,
                            "" if ok8 else f"实得 detail=「{d}」 bad={[x[3][:70] for x in b][:2]}"))

            # ——— 第三十二轮 V18：**新接上闸的那八个分母**各自的变异回测。
            #     ⚠ 加这五条的理由就是 V18 的另一半：那八个数第三十一轮就算出来了、
            #       也印在报文里了，可 runner 不转发、min/max 里一个没有、全项目零引用 ——
            #       「算了、印了、没人判」。下面每一条都对应一种「它没接上」的失效方式。

            # 9 覆盖侧分母真的在被判：把 `registeredRaw` 的记录值抬高一格 → 必须响。
            #   （它要是没接上闸，抬多少都不会响 —— 这正是本轮之前的状态。）
            b, d = a_panel_liveness(
                dict(rule, min=dict(rule["min"], registeredRaw=rule["min"]["registeredRaw"] + 1)),
                _TransCtx(clean))
            ok9 = any("registeredRaw" in str(x[2]) for x in b)
            results.append((f"把 `min.registeredRaw` 抬高一格 → 覆盖侧分母必须响"
                            f"（违规 {len(b)} 处）", ok9,
                            "" if ok9 else f"实得：{[(x[2], x[3][:50]) for x in b][:3]}"))

            # 10 没核侧分母真的在被判：把 `uncheckedRaw` 的上限压低一格 → 必须响。
            b, d = a_panel_liveness(
                dict(rule, max=dict(rule["max"], uncheckedRaw=rule["max"]["uncheckedRaw"] - 1)),
                _TransCtx(clean))
            ok10 = any("uncheckedRaw" in str(x[2]) for x in b)
            results.append((f"把 `max.uncheckedRaw` 压低一格 → 没核侧分母必须响"
                            f"（违规 {len(b)} 处）", ok10,
                            "" if ok10 else f"实得：{[(x[2], x[3][:50]) for x in b][:3]}"))

            # 11 **V18 那处缺陷本身的回测**：让表侧独立计数退回「不解包」的老写法
            #    （数出 721/1304 而不是 887/1459）→ 对账恒等式必须当场响。
            #    ⚠ 这一条不含阈值：把 min/max 全调松也动不了它。
            b, d = a_panel_liveness(dict(rule, mutate={"count_no_unwrap": True}),
                                    _TransCtx(clean))
            ok11 = any("登记数对账" in str(x[2]) for x in b)
            results.append((f"表侧独立计数退回**不解包**的老写法（V18 那处缺陷）→ "
                            f"「登记数对账」恒等式必须响（违规 {len(b)} 处）", ok11,
                            "" if ok11 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

            # 12 台账恒等式：把「按设计没核」加个正偏移 → 账不平（raw 与 distinct 两条）
            #    且没核侧上限被顶破，三处一起响。
            b, d = a_panel_liveness(dict(rule, mutate={"ledger_offset": 3}), _TransCtx(clean))
            ok12 = (any("台账 raw" in str(x[2]) for x in b)
                    and any("台账 distinct" in str(x[2]) for x in b)
                    and any("uncheckedRaw" in str(x[2]) for x in b))
            results.append((f"把「按设计没核」偏移 +3 → 台账 raw / distinct 两条恒等式与"
                            f"没核侧上限**都**要响（违规 {len(b)} 处）", ok12,
                            "" if ok12 else f"实得：{[(x[2], x[3][:60]) for x in b][:4]}"))

            # 13 面板自报的两个旗标：`ledgerBalanced` 必须恒 true、`ledgerNoReason` 必须恒 0。
            b, d = a_panel_liveness(dict(rule, mutate={"fake_ledger_flags": True}),
                                    _TransCtx(clean))
            ok13 = (any("台账自报" in str(x[2]) for x in b)
                    and any("没核却说不出原因" in str(x[2]) for x in b))
            results.append((f"谎报 `ledgerBalanced=false` / `ledgerNoReason=1` → 两条都必须响"
                            f"（违规 {len(b)} 处）", ok13,
                            "" if ok13 else f"实得：{[(x[2], x[3][:60]) for x in b][:3]}"))

    print("\n自检面板现跑闸（panel_liveness）副本树回测：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\npanel_liveness：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ============================ 「运行时输入必须入库」闸的回测（第二十八轮 ③）
#
# 这一条判的是 git，所以回测**真的建一个 git 仓**（`git init` + `git add`，不 commit ——
# `ls-files` 看的是索引，不需要 commit）。在真实项目树上做不了这种回测：
# 不能为了验判据去把某个文件从库里摘掉。


class _TrackedCtx:
    """`tracked_inputs` 要的最小 ctx：`.root` + `.here` + `.repos` + **`.rules`**。

    ⚠ `.rules` 是第三十轮加的，而且是**这一组自检此前的漏判所在**：本条的推导器原先
      自己去磁盘重读规则文件，于是这个 ctx 里给什么规则都无所谓 ——
      自检以为自己在喂输入，其实喂的是磁盘（形态 (g)，同时也是形态 (h)：
      空转的是喂输入的那个探针）。现在推导吃 `ctx.rules`，这个字段才真的承重。
    """

    def __init__(self, root, repos=None, rules=None, rules_path=None):
        self.root = root
        self.here = os.path.join(root, "3-常用脚本", "qa")
        self.repos = repos or {}
        self.rules = rules
        self.rules_path = rules_path


_TRACKED_RULES_REL = "5-其他内容/RESOLUTIONS.assertions.json"
_TRACKED_DEFAULT_RULES = {"assertions": [
    {"id": "R-x", "kind": "translate_cases", "repo": "ember",
     "src": "scripts/ember-hardcoded-cn.mjs"},
]}


def _tracked_make_repo(root, add=(), rules=None, git=True, disk_rules=None):
    """造一棵最小项目树：规则文件 + 两个执行体；`add` 里的才 `git add`。

    返回一个**配好 `.rules` 的 `_TrackedCtx`** —— 落盘的那份与喂给 ctx 的那份默认相同；
    `disk_rules` 可以让两者**故意不同**，用来回测「推导到底吃的是哪一份」。
    """
    rules = _TRACKED_DEFAULT_RULES if rules is None else rules
    on_disk = rules if disk_rules is None else disk_rules
    files = {
        _TRACKED_RULES_REL: json.dumps(on_disk, ensure_ascii=False),
        "3-常用脚本/qa/translate_cases_runner.mjs": "// 执行体\n",
        "3-常用脚本/qa/selfcheck_panel_runner.mjs": "// 执行体\n",
        "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs": "// 被判文件\n",
    }
    for rel, body in files.items():
        p = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(body)
    if git:
        import shutil
        g = shutil.which("git")
        subprocess.run([g, "init", "-q", root], capture_output=True)
        if add:
            subprocess.run([g, "-C", root, "add", "--"] + [
                os.path.join(*rel.split("/")) for rel in add], capture_output=True)
    repos = {"ember": os.path.join(root, REPOS["ember"]),
             "crucible": os.path.join(root, REPOS["crucible"])}
    return _TrackedCtx(root, repos, rules,
                       os.path.join(root, *_TRACKED_RULES_REL.split("/")))


def run_tracked_selftest():
    """返回 (失败条数, 总条数)。"""
    results = []
    rule = {"kind": "tracked_inputs", "rules": _TRACKED_RULES_REL,
            "sweep": ["3-常用脚本/qa"], "must_include": [], "min_checked": 1}
    all_files = ["5-其他内容/RESOLUTIONS.assertions.json",
                 "3-常用脚本/qa/translate_cases_runner.mjs",
                 "3-常用脚本/qa/selfcheck_panel_runner.mjs",
                 "1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs"]
    with tempfile.TemporaryDirectory() as tmp:
        # 1 全部入库 → 0 违规，且 detail 必须自陈推出了几个输入（证明它真的推了）。
        root = os.path.join(tmp, "ok")
        c = _tracked_make_repo(root, add=all_files)
        b, d = a_tracked_inputs(rule, c)
        ok1 = not b and "运行时输入" in d and "git ls-files" in d
        results.append((f"全部 git add 过 → 0 违规，detail 自陈推了什么（{d}）", ok1,
                        "" if ok1 else f"违规 {len(b)} 处：{[x[3][:70] for x in b][:3]}"))

        # 2 **第 ④ 次缺陷的原样复现**：规则里**没写 runner 字段**（用默认值），
        #   而那个默认执行体没 add → 必须点名它。任何「只看规则里写了什么」的清单
        #   都会恰好漏掉这一个。
        root = os.path.join(tmp, "runner")
        c = _tracked_make_repo(root, add=[f for f in all_files
                                          if "translate_cases_runner" not in f])
        b, d = a_tracked_inputs(rule, c)
        ok2 = any("translate_cases_runner.mjs" in str(x[2]) for x in b) and any(
            "没进 git" in str(x[3]) for x in b)
        results.append((f"默认执行体（规则里没写 runner 字段）没入库 → 必须点名它"
                        f"（违规 {len(b)} 处）", ok2,
                        "" if ok2 else f"实得：{[(x[2], x[3][:50]) for x in b][:3]}"))

        # 3 被判文件本身没入库 → 也要响（`src` 那一路）。
        root = os.path.join(tmp, "src")
        c = _tracked_make_repo(root, add=[f for f in all_files
                                          if "ember-hardcoded" not in f])
        b, d = a_tracked_inputs(rule, c)
        ok3 = any("ember-hardcoded-cn.mjs" in str(x[2]) for x in b)
        results.append((f"被判文件（规则的 src）没入库 → 必须响（违规 {len(b)} 处）", ok3,
                        "" if ok3 else f"实得：{[x[2] for x in b][:3]}"))

        # 4 形态 (d) 的镜像：`must_include` 点名了一个**推不出来**的路径 → 必须响。
        #   这一道防的是「推导器自己静默变短」——清单短了，判据会一路报「全部入库」。
        root = os.path.join(tmp, "must")
        c = _tracked_make_repo(root, add=all_files)
        b, d = a_tracked_inputs(dict(rule, must_include=["5-其他内容/nosuch-input.json"]), c)
        ok4 = any("nosuch-input.json" in str(x[2]) for x in b) and any(
            "清单静默变短" in str(x[3]) for x in b)
        results.append((f"must_include 点名一个推不出来的路径 → 必须响（违规 {len(b)} 处）",
                        ok4, "" if ok4 else f"实得：{[(x[2], x[3][:50]) for x in b][:3]}"))

        # 5 形态 (e)：要扫的目录压根不在 → 必须响，不许「扫了个空目录还报绿」。
        root = os.path.join(tmp, "sweep")
        c = _tracked_make_repo(root, add=all_files)
        b, d = a_tracked_inputs(dict(rule, sweep=["3-常用脚本/nosuch"]), c)
        ok5 = any("要扫的目录不在" in str(x[3]) for x in b)
        results.append((f"sweep 指向不存在的目录 → 必须响（违规 {len(b)} 处）", ok5,
                        "" if ok5 else f"实得：{[x[3][:60] for x in b][:3]}"))

        # 6 **压根不在 git 仓里**（比「没 add」更彻底）→ 必须响。
        root = os.path.join(tmp, "nogit")
        c = _tracked_make_repo(root, git=False)
        b, d = a_tracked_inputs(rule, c)
        ok6 = bool(b) and any("不在任何 git 仓里" in str(x[3]) for x in b)
        results.append((f"整棵树都不在 git 仓里 → 必须响（违规 {len(b)} 处）", ok6,
                        "" if ok6 else f"实得：{[x[3][:60] for x in b][:3]}"))

        # 7 形态 (e)：git 取不到 → 报「没跑成」，不许静默跳过。
        root = os.path.join(tmp, "nogitbin")
        c = _tracked_make_repo(root, add=all_files)
        b, d = a_tracked_inputs(dict(rule, git_bin="git-that-does-not-exist"), c)
        ok7 = bool(b) and d == "没跑成"
        results.append((f"PATH 里没有 git → 报「没跑成」而不是通过（{d}）", ok7,
                        "" if ok7 else f"实得 detail=「{d}」 bad={b[:1]}"))

        # 8 形态 (d)：清单空了（规则里 sweep / 断言都没有）→ min_checked 必须响。
        root = os.path.join(tmp, "empty")
        c = _tracked_make_repo(root, add=all_files, rules={"assertions": []})
        b, d = a_tracked_inputs(dict(rule, sweep=[], min_checked=40), c)
        ok8 = any("这条断言在空转" in str(x[3]) for x in b)
        results.append((f"清单缩到只剩规则文件自己 → min_checked 空转闸必须响"
                        f"（违规 {len(b)} 处）", ok8,
                        "" if ok8 else f"实得：{[x[3][:60] for x in b][:3]}"))

        # ↓↓↓ 第三十轮新增：**形态 (g)「半读半不读 `--rules`」的直接回测** ↓↓↓
        # 这四条此前一条都没有 —— 因为推导器根本不看 ctx，喂什么都一样，
        # 这一组自检自己就是形态 (h)（空转的是喂输入的那个探针）。

        # 9 **推导必须吃 ctx.rules，不许吃磁盘。** 磁盘上放一份「什么都不引用」的空规则，
        #   ctx 里给一份引用了 `src` 的真规则 → 那个 src 必须出现在清单里。
        #   修之前：推导读磁盘 ⇒ 清单里没有 src ⇒ 摘掉 src 的 git add 也不会响（假绿）。
        root = os.path.join(tmp, "g_ctx")
        c = _tracked_make_repo(root, add=[f for f in all_files if "ember-hardcoded" not in f],
                               rules=_TRACKED_DEFAULT_RULES, disk_rules={"assertions": []})
        b, d = a_tracked_inputs(rule, c)
        ok9 = any("ember-hardcoded-cn.mjs" in str(x[2]) for x in b)
        results.append((f"形态 (g)：磁盘上是空规则、ctx.rules 里有 `src` → 必须按 **ctx** 推"
                        f"（违规 {len(b)} 处）", ok9,
                        "" if ok9 else f"实得：{[(x[2], x[3][:50]) for x in b][:3]}"
                                       f"—— 它还在读磁盘"))

        # 10 反向：磁盘上是**真**规则、ctx.rules 是空的 → 清单里**不许**冒出那个 src。
        #    这一条钉的是「别偷偷两份都读」：只有单向的 9 不足以排除「合并磁盘那一份」。
        root = os.path.join(tmp, "g_disk")
        c = _tracked_make_repo(root, add=[f for f in all_files if "ember-hardcoded" not in f],
                               rules={"assertions": []}, disk_rules=_TRACKED_DEFAULT_RULES)
        b, d = a_tracked_inputs(dict(rule, min_checked=1), c)
        ok10 = not any("ember-hardcoded-cn.mjs" in str(x[2]) for x in b)
        results.append((f"形态 (g) 反向：磁盘上有 `src`、ctx.rules 是空的 → 清单里**不许**有它"
                        f"（违规 {len(b)} 处）", ok10,
                        "" if ok10 else f"实得：{[(x[2], x[3][:50]) for x in b][:3]}"
                                        f"—— 它把磁盘那份也读进来了"))

        # 11 形态 (e)：拿不到 ctx.rules → 报「没跑成」，**不许**退回去读磁盘顶上。
        root = os.path.join(tmp, "g_none")
        c = _tracked_make_repo(root, add=all_files)
        c.rules = None
        b, d = a_tracked_inputs(rule, c)
        ok11 = bool(b) and d == "没跑成" and any("ctx.rules" in str(x[2]) for x in b)
        results.append((f"拿不到 ctx.rules → 报「没跑成」而不是回头读磁盘（{d}）", ok11,
                        "" if ok11 else f"实得 detail=「{d}」 bad={b[:1]}"))

        # 12 `--rules` 指到别处时，detail 必须**明说**这件事（不许静默）。
        root = os.path.join(tmp, "g_note")
        c = _tracked_make_repo(root, add=all_files)
        c.rules_path = os.path.join(root, "4-临时脚本", "some-copy.json")
        b, d = a_tracked_inputs(rule, c)
        ok12 = "不是同一份" in d and "some-copy.json" in d
        results.append((f"`--rules` 指到副本 → detail 必须明写「不是同一份」", ok12,
                        "" if ok12 else f"实得 detail=「{d}」"))

    print("\n运行时输入入库闸（tracked_inputs）临时 git 仓回测：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\ntracked_inputs：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ========================= ruleset_shape 的回测：**逐条复现复核实测的七种容器层动作**
#
# 为什么这一组非有不可：本条守的东西就是「有没有被删掉」，
# 而「删掉之后会不会响」只能靠**真的删一次**来证明。
# 下面七条一一对应第二十八轮复核实测的七个**单点**动作 ——
# 那七个动作当时改完主闸仍 65/0/0 全绿，本轮起每一条都必须当场红。
# ⚠ 变异的是**规则集**（不是被判文件、也不是判据）：这正是作弊路径 C 的形状。
# ⚠ 取的是**发布中的那份规则集**，不在自检里另写一份合成的 —— 自检验一份和产线
#   不同的配置，本身就是形态 (g)。


class _RulesCtx:
    """只提供 `.rules` 的最小 ctx —— ruleset_shape 不走叶/键通道。"""

    def __init__(self, rules):
        self.rules = rules


def _shape_pick(rules, kind):
    """规则集里第一条该 kind 的规则在 `assertions` 里的下标。"""
    for i, r in enumerate(rules["assertions"]):
        if r.get("kind") == kind:
            return i
    return None


def _shape_layers(bad):
    """把一串违规按**是哪一层响的**归类 → {层名: [点名的东西]}。

    层名就是违规四元组的第 2 元（`配置·containers` / `配置·registered` …）。
    第三十轮加这个函数的理由写在下面 `run_ruleset_shape_selftest` 的 docstring 里。
    """
    out = {}
    for x in bad:
        out.setdefault(str(x[1]).split("·")[-1], []).append(str(x[2]))
    return out


def run_ruleset_shape_selftest():
    """返回 (失败条数, 总条数)。

    ⚠ **第三十轮：这一组从「只断言红了」升级成「断言是**哪一层**红的」。**
      旧写法每条只查 `want_key in str(x[2])` —— 点名的那个字符串对上就算过。
      于是把 `RULESET_SHAPE["containers"]` 整块清空，14 条用例**照样全绿**：
      那些变异恰好都被 `min_len` / `hard_floors` 顺手接住了，用例根本分不出是谁接的。
      「看着像承重、其实没验过它承不承重」——这跟护栏本身失效只差一步。
      现在每条用例声明一个**精确的层集合**（不是包含，是相等）：
        · 少响一层 → 说明某一层悄悄失效了；
        · 多响一层 → 说明归因变了，也得有人看一眼。

    ⚠ **`containers` 那一层承不承重：验过了，承重。** 复核当时判它是「纯冗余」，
      口径是「清空它 → selftest 仍 14/14 + 主闸仍全红」。那个结论是**用例集选样造成的**：
      当时那 14 条只碰了 `arrangements` / `min` / `max` 这些**同时被 `min_len`
      / `hard_floors` 盖住**的键。逐个单点删一遍才看得出来，有 14 个位置
      **只有 `containers` 一层响**（`translate_cases.recorded` 整块 + 它的 7 个键、
      `arrangements.prefixes`、`arrangements.expect_untranslated`、
      `panel_liveness.min` 的 fetchOk / tableRows / tablesFedIn / tplFiles）。
      其中面板闸那 4 个键更进一步：**主闸层面也只有本条会响** ——
      实测「清空 containers + 删掉 `min.tplFiles`」，主闸 **66/0/0 全绿、rc=0**。
      ⇒ 不是冗余，是此前没被用例碰到的承重段。下面 `SOLE` 组就钉这一点：
        这些位置的违规层集合必须**恰好等于** {containers}。
    """
    results = []
    try:
        base = json.load(open(DEFAULT_RULES, encoding="utf-8"))
    except Exception as exc:                           # noqa: BLE001
        print("\n规则集形状闸（ruleset_shape）回测：")
        print(f"  FAIL 前置：规则文件读不动 {exc!r}")
        print("\nruleset_shape：0 / 1 通过")
        return 1, 1
    rule = next((r for r in base["assertions"] if r.get("kind") == "ruleset_shape"), None)
    if rule is None:
        results.append(("前置：规则集里找得到 ruleset_shape 规则", False,
                        f"{DEFAULT_RULES} 里一条都没有 —— 用例无从跑起"))
        rule = {"id": "R-ruleset-shape", "kind": "ruleset_shape"}

    def mutate(fn):
        r = copy.deepcopy(base)
        fn(r)
        return _RulesCtx(r)

    # 0 特异度：原样规则集 → 0 违规，且 detail 必须自陈「核过多少条 / 多少个容器块」
    #   （说不出数的是自检不是断言 —— 本文件的通则）。
    b, d = a_ruleset_shape(rule, _RulesCtx(copy.deepcopy(base)))
    ok0 = (not b and "核过" in d and "容器块" in d and "RULESET_SHAPE" in d)
    results.append((f"原样规则集 → 0 违规，且 detail 自陈核了什么（{d}）", ok0,
                    "" if ok0 else f"违规 {len(b)} 处：{[(x[2], x[3][:70]) for x in b][:3]}"))

    # —— 前置 A：**登记表不许是个空壳。** REGISTERED_ASSERTIONS 是 H2 的唯一接住点，
    #    可它自己也是本文件里的一个 dict —— 把它删剩一条，H2 就又能过了，而且
    #    这一步是**单点动作**。所以在这里钉死：它必须与发布中那份规则集**逐条对齐**。
    live_ids = {r.get("id") for r in base["assertions"] if isinstance(r.get("id"), str)}
    reg_ids = set(REGISTERED_ASSERTIONS)
    okA = reg_ids == live_ids
    results.append((f"前置 A：登记在册 id（{len(reg_ids)} 条）必须与发布中的规则集"
                    f"（{len(live_ids)} 条）**逐条相等** —— 否则登记表自己就是个空壳", okA,
                    "" if okA else f"登记表多出：{sorted(reg_ids - live_ids)[:5]}；"
                                   f"规则集里多出：{sorted(live_ids - reg_ids)[:5]}"))
    okA2 = all(r.get("kind") == REGISTERED_ASSERTIONS.get(r.get("id"))
               for r in base["assertions"] if r.get("id") in REGISTERED_ASSERTIONS)
    results.append(("前置 A2：登记表里每条的 kind 必须与规则集当前值一致（防登记表写歪 = 恒红）",
                    okA2, "" if okA2 else "登记表与规则集的 kind 对不上"))
    okA3 = set(REGISTERED_ASSERTIONS.values()) <= set(KINDS)
    results.append((f"前置 A3：登记表里出现的 kind 必须都是 KINDS 里认识的"
                    f"（{len(set(REGISTERED_ASSERTIONS.values()))} 种）", okA3,
                    "" if okA3 else f"多出：{sorted(set(REGISTERED_ASSERTIONS.values()) - set(KINDS))}"))

    # —— 前置 B：**`min_by_kind` 不许只覆盖几种。** 同理，它也是单点可缩的。
    #    钉死：覆盖发布中规则集的**全部** kind，且每类的地板 = 当前实际条数。
    live_by_kind = {}
    for r in base["assertions"]:
        live_by_kind[r.get("kind")] = live_by_kind.get(r.get("kind"), 0) + 1
    okB = RULESET_SHAPE["min_by_kind"] == live_by_kind
    results.append((f"前置 B：`min_by_kind` 必须覆盖全部 {len(live_by_kind)} 种 kind、"
                    f"且每类地板 = 当前实际条数（现在登记 "
                    f"{len(RULESET_SHAPE['min_by_kind'])} 类）", okB,
                    "" if okB else f"对不上：{ {k: (RULESET_SHAPE['min_by_kind'].get(k), v) for k, v in live_by_kind.items() if RULESET_SHAPE['min_by_kind'].get(k) != v} }"))
    okB2 = (RULESET_SHAPE["min_assertions"] == len(base["assertions"])
            and RULESET_SHAPE["min_kinds"] == len(live_by_kind))
    results.append((f"前置 B2：`min_assertions` / `min_kinds` 与发布中的规则集对齐"
                    f"（{RULESET_SHAPE['min_assertions']} / {RULESET_SHAPE['min_kinds']}）", okB2,
                    "" if okB2 else f"实际 {len(base['assertions'])} 条 / {len(live_by_kind)} 种"))

    # —— `main()` 的判定式：**跳过 ≠ 通过**（第三十轮 H1/H3 的一半）。
    for note, args, want in [
        ("全过 → rc 0", (0, 0), 0),
        ("**只有跳过、没有失败 → rc 必须非 0**（旧写法在这里返回 0，H1/H3 就是这么过的）",
         (0, 1), 1),
        ("有失败 → rc 非 0", (1, 0), 1),
        ("既有失败又有跳过 → rc 非 0", (2, 3), 1),
    ]:
        got = _exit_code(*args)
        okx = got == want
        results.append((f"_exit_code{args}：{note}", okx,
                        "" if okx else f"实得 {got}，期望 {want}"))

    i_tc = _shape_pick(base, "translate_cases")
    i_pl = _shape_pick(base, "panel_liveness")
    i_ti = _shape_pick(base, "tracked_inputs")

    def _kill(idx, *keys):
        def go(r):
            for k in keys:
                r["assertions"][idx].pop(k, None)
        return go

    def _layer_case(note, fn, want_layers):
        b, _d = a_ruleset_shape(rule, mutate(fn))
        got = _shape_layers(b)
        okc = set(got) == set(want_layers)
        results.append((f"{note} → 必须恰好由 {sorted(want_layers)} 响"
                        f"（违规 {len(b)} 处）", okc,
                        "" if okc else f"实得层 {sorted(got)}："
                                       f"{[(x[1], x[2], x[3][:50]) for x in b][:4]}"))

    # ↓↓↓ 复核实测的七种**单点**容器层动作，逐条复现 —— 现在连「哪一层响」一起断言 ↓↓↓
    _layer_case("C-① 删 `arrangements` 整块（438 条编排名对拍整段消失）",
                _kill(i_tc, "arrangements"),
                ["containers", "hard_floors", "min_len", "payload_floors"])
    _layer_case("C-② `channels` 砍成一个（同一道护栏 438→219）",
                lambda r: r["assertions"][i_tc]["arrangements"].__setitem__(
                    "channels", ["Music"]),
                ["min_len", "payload_floors"])
    _layer_case("C-③ 面板闸的 `min` / `max` 整块删掉（覆盖侧与 miss 侧记录值全消失）",
                _kill(i_pl, "min", "max"),
                ["containers", "hard_floors", "hard_ceilings", "payload_floors"])
    _layer_case("C-④ `tracked.min_checked` → 0（防「清单空转」的兜底消失）",
                lambda r: r["assertions"][i_ti].__setitem__("min_checked", 0),
                ["hard_floors", "tracked_min_checked", "payload_floors"])
    _layer_case("C-⑤ `tracked.must_include` 清空（防「推导器静默变短」的点名消失）",
                lambda r: r["assertions"][i_ti].__setitem__("must_include", []),
                ["containers", "min_len", "tracked_must_include", "payload_floors"])
    _layer_case("C-⑥ `arrangements.min_labels` 单独改小",
                lambda r: r["assertions"][i_tc]["arrangements"].__setitem__("min_labels", 2),
                ["hard_floors", "payload_floors"])
    _layer_case("C-⑦ **删掉整条断言**（此前全项目没有任何东西数一数还剩几条）",
                lambda r: r["assertions"].pop(_shape_pick(r, "cn_absent")),
                ["min_assertions", "min_by_kind", "registered"])

    # ↓↓↓ SOLE 组（第三十轮新增）：第三十轮时**只有 `containers` 一层守着的位置** ↓↓↓
    #   这一组存在的全部意义就是「证明 containers 承重」。层集合必须**恰好**相等。
    #   ⚠ 第三十一轮脚注（**不改上面那句历史结论**）：`payload_floors` 上线之后，
    #     这些位置变成**两层**守着，所以期望集合从 {containers} 改成
    #     {containers, payload_floors}。改的是「现在有几层」，不是「当时只有一层」——
    #     当时那个实测（清空 containers + 删 tplFiles → 主闸 66/0/0 全绿）仍然成立，
    #     只是从第三十一轮起，即使 containers 被整块清空，载荷地板还会独立响一次。
    #     ⇒ 这一组现在**同时**钉两件事：containers 没失效、payload_floors 真的覆盖到了这些键。
    #   ⚠ 前 4 条更狠：主闸层面也只有本条断言会响（实测「清空 containers + 删 tplFiles」
    #     → 主闸 66/0/0 全绿）。它们要是掉了，那 4 个记录值就真的没人看了。
    for blk, key in [("min", "fetchOk"), ("min", "tableRows"),
                     ("min", "tablesFedIn"), ("min", "tplFiles")]:
        _layer_case(f"SOLE 面板闸 `{blk}.{key}` 单独摘掉（**主闸层面也只有本条守着**）",
                    (lambda blk=blk, key=key:
                     lambda r: r["assertions"][i_pl][blk].pop(key, None))(),
                    ["containers", "payload_floors"])
    _layer_case("SOLE 删 `recorded` 整块（三张表的全部记录值一起消失）",
                _kill(i_tc, "recorded"), ["containers", "payload_floors"])
    for key in ("prefixed_size", "patterns_size", "np_size", "negative",
                "positive", "notify_negative", "notify_positive"):
        _layer_case(f"SOLE `recorded.{key}` 单独摘掉",
                    (lambda key=key:
                     lambda r: r["assertions"][i_tc]["recorded"].pop(key, None))(),
                    ["containers", "payload_floors"])
    for key in ("prefixes", "expect_untranslated"):
        _layer_case(f"SOLE `arrangements.{key}` 单独摘掉",
                    (lambda key=key:
                     lambda r: r["assertions"][i_tc]["arrangements"].pop(key, None))(),
                    ["containers", "payload_floors"])

    # 8 **自指那一刀**：把 `ruleset_shape` 断言本身删掉。
    #   这一刀不由断言接（断言已经不在了），由 `main()` 的固定检查接 ——
    #   守卫不能与被守护的容器同生共死，本条就是那个证明。
    r8 = copy.deepcopy(base)
    r8["assertions"] = [x for x in r8["assertions"] if x.get("kind") != "ruleset_shape"]
    msg8 = _require_ruleset_shape(r8)
    ok8 = bool(msg8) and "ruleset_shape" in msg8
    results.append(("C-⑧ 连守卫一起删（删掉 ruleset_shape 断言本身）→ main() 的固定检查"
                    "必须接住（它不在规则文件里）", ok8, "" if ok8 else f"实得 {msg8!r}"))
    # 8b 反向：规则集里有它 → 那道检查必须**不**报（否则它是恒真的，等于没有）。
    ok8b = _require_ruleset_shape(base) is None
    results.append(("C-⑧b 规则集里有 ruleset_shape → 那道固定检查必须不响（防恒真）",
                    ok8b, "" if ok8b else "它对着一份含 ruleset_shape 的规则集也报错 = 恒真"))

    # 9 **强度参数的死下限 / 死上限**（§3.7.2：强度参数与判据同层 = 没有强度）
    _layer_case("把面板闸的 `min.checkedDistinct` 719 改成 1（死下限住在本文件）",
                lambda r: r["assertions"][i_pl]["min"].__setitem__("checkedDistinct", 1),
                ["hard_floors", "payload_floors"])
    _layer_case("把 miss 侧上限 `max.missDistinct` 4 放大到 999",
                lambda r: r["assertions"][i_pl]["max"].__setitem__("missDistinct", 999),
                ["hard_ceilings", "payload_floors"])

    # 11 `must_include` 摘掉一条**默认执行体**（现算层：清单从规则集自身推）
    def _drop_runner(r):
        mi = r["assertions"][i_ti]["must_include"]
        r["assertions"][i_ti]["must_include"] = [
            x for x in mi if not x.endswith("selfcheck_panel_runner.mjs")]
    b, _d = a_ruleset_shape(rule, mutate(_drop_runner))
    got11 = _shape_layers(b)
    ok11 = (set(got11) == {"tracked_must_include", "payload_floors"}
            and any("selfcheck_panel_runner.mjs" in str(x[3]) for x in b))
    results.append((f"SOLE 从 must_include 里摘掉一条 kind 的**默认执行体** → "
                    f"必须恰好由 tracked_must_include 响（现算层）（违规 {len(b)} 处）", ok11,
                    "" if ok11 else f"实得层 {sorted(got11)}："
                                    f"{[(x[2], x[3][:60]) for x in b][:3]}"))
    # 11b `must_include` 摘掉**历史层点名**的那条 `pack`（第二十九轮补的那一条）。
    #     复核实测：把 `tracked_must_include` 清空 + 摘掉这条 → 主闸 66/0/0 全绿。
    #     ⇒ 这一层**不是冗余**，本用例钉的就是它。
    def _drop_pack(r):
        mi = r["assertions"][i_ti]["must_include"]
        r["assertions"][i_ti]["must_include"] = [
            x for x in mi if "crucible-adventure" not in x]
    _layer_case("SOLE `must_include` 摘掉历史层点名的 `pack` 那条（复核实测：把这一层"
                "清空后此动作全绿）", _drop_pack, ["tracked_must_include", "payload_floors"])

    # ↓↓↓ H 组（第三十轮）：复核跑通的三种「主闸一声不吭」的绕法 ↓↓↓
    _layer_case("H1 把一条 `cn_absent` 的 kind 拼错成 `cn_absentt`"
                "（旧闸：65/0/跳过 1、rc=0）",
                lambda r: r["assertions"][_shape_pick(r, "cn_absent")].__setitem__(
                    "kind", "cn_absentt"),
                ["min_by_kind", "registered"])

    def _h2(r):
        r["assertions"].pop(_shape_pick(r, "cn_absent"))
        filler = copy.deepcopy(r["assertions"][_shape_pick(r, "term_gated")])
        filler["id"] = "R-filler-h2"
        r["assertions"].append(filler)
    _layer_case("H2 删一条真断言 + 复制另一条换个 id 当填充（旧闸：66/0/0、rc=0、"
                "输出里一行异常都没有）", _h2, ["min_by_kind", "registered"])

    def _h3(r):
        r["assertions"].pop(_shape_pick(r, "cn_absent"))
        r["assertions"].append({"kind": "nope"})
    _layer_case("H3 删一条 + 塞一个 `{\"kind\": \"nope\"}` 空壳（旧闸：65/0/跳过 1、rc=0）",
                _h3, ["id_unique", "min_by_kind", "registered"])

    def _h2b(r):
        r["assertions"].append(copy.deepcopy(r["assertions"][_shape_pick(r, "cn_absent")]))
        r["assertions"].pop(_shape_pick(r, "term_gated"))
    _layer_case("H2' 连 id 一起复制（最省事的填充法）→ id 唯一性那一层必须接住",
                _h2b, ["id_unique", "min_by_kind", "registered"])
    _layer_case("H2'' 把一条断言的 id 抹掉（没有 id 就无从登记核对）",
                lambda r: r["assertions"][_shape_pick(r, "cn_absent")].pop("id"),
                ["id_unique", "registered"])
    # H1'：不拼错，改成一个**真的存在**的 kind —— 「换成更松的那一类」这条路。
    # 挑 `sense_gated`（全集里只此一条）做样本，所以 `min_kinds` 也会跟着响：
    # 这正好展示分层归因的用处 —— 同一个动作在**独苗 kind** 上比在 15 条的 `cn_absent`
    # 上多响一层，用例把这件事写死，省得下次有人以为「少响一层也无所谓」。
    _layer_case("H1' 把一条断言换成**另一个合法 kind**（不拼错，改成真的存在的那种）",
                lambda r: r["assertions"][_shape_pick(r, "sense_gated")].__setitem__(
                    "kind", "cn_absent"),
                ["min_by_kind", "min_kinds", "registered"])
    _layer_case("H1'' 同一个动作、改的是 15 条那一类里的一条（`cn_absent` → `term_gated`）"
                "→ `min_kinds` 不响，只剩逐类点名与登记表",
                lambda r: r["assertions"][_shape_pick(r, "cn_absent")].__setitem__(
                    "kind", "term_gated"),
                ["min_by_kind", "registered"])

    # 12 形态 (e)：拿不到本次运行的规则集 → 报「没跑成」，不是通过。
    b, d = a_ruleset_shape(rule, _RulesCtx(None))
    ok12 = bool(b) and d == "没跑成"
    results.append((f"拿不到 ctx.rules → 报「没跑成」而不是通过（{d}）", ok12,
                    "" if ok12 else f"实得 detail=「{d}」 bad={b[:1]}"))

    print("\n规则集形状闸（ruleset_shape）回测：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\nruleset_shape：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ===================== 主闸**返回码**本身的回测（第三十轮，`--selftest` 里唯一起子进程跑真身的一组）
#
# 为什么非得起子进程：上面那 4 条 `_exit_code(...)` 用例只证明**那个函数**算得对，
# 证明不了 `main()` 真的用了它。把 `main()` 最后一行改成 `return 0` 是**单点动作**，
# `_exit_code` 的用例照样全绿 —— 那就是「护栏在，但没接在门上」。
# ⇒ 这一组直接问真身要返回码。
#
# ⚠ **不与库的状态耦合**：不假设「库现在是干净的」。库本身要是红的，
#   `rc` 对比那一半就如实说「做不了」，只留下**归因明确**的那两条（FAIL 行 + 跳过计数）。
#   自检验的是判据，不是库 —— 把库的状态混进自检的判定里，下一轮就会有人为了让自检绿去改库。
_VERDICT_TYPO_KIND = "cn_absentt"          # H1 的那一个字母


def run_main_verdict_selftest():
    """返回 (失败条数, 总条数)。起子进程跑 `assert_resolutions.py` 真身。"""
    results = []
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def gate(rules_path):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rules", rules_path],
                           cwd=ROOT, capture_output=True, env=env)
        return p.returncode, p.stdout.decode("utf-8", "replace")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            base = json.load(open(DEFAULT_RULES, encoding="utf-8"))
        except Exception as exc:                       # noqa: BLE001
            results.append(("前置：规则文件读不动", False, repr(exc)))
            base = None
        if base is not None:
            p_base = os.path.join(tmp, "base.json")
            with open(p_base, "w", encoding="utf-8") as fh:
                json.dump(base, fh, ensure_ascii=False)
            rc_base, out_base = gate(p_base)

            mut = copy.deepcopy(base)
            i = _shape_pick(mut, "cn_absent")
            victim = mut["assertions"][i]["id"]
            mut["assertions"][i]["kind"] = _VERDICT_TYPO_KIND
            p_mut = os.path.join(tmp, "typo.json")
            with open(p_mut, "w", encoding="utf-8") as fh:
                json.dump(mut, fh, ensure_ascii=False)
            rc_mut, out_mut = gate(p_mut)

            # ① 认不出的 kind 必须打成 **FAIL 并点名那条规则**，不是一行 `??`。
            ok1 = f"FAIL  {victim}" in out_mut and "未知断言类型" in out_mut
            results.append((f"H1（`{victim}` 的 kind 拼成 `{_VERDICT_TYPO_KIND}`）→ "
                            f"真身必须打 `FAIL {victim} —— 未知断言类型`", ok1,
                            "" if ok1 else "输出里没有这一行 —— 又变回 `??` 那种放行了？"))
            # ② 跳过计数必须是 0：认不出 kind 不许再记成「跳过」。
            ok2 = "/ 跳过 0" in out_mut
            line = next((x for x in out_mut.splitlines() if x.startswith("通过 ")), "?")
            results.append((f"H1 → 计数行的「跳过」必须是 0（`{line}`）", ok2,
                            "" if ok2 else "它又把认不出的 kind 记成跳过了"))
            # ③ **返回码**：这一条才是 `main()` 真的用了 `_exit_code` 的证据。
            if rc_base == 0:
                ok3 = rc_mut != 0
                results.append((f"H1 → 真身返回码必须非 0（原样副本 rc={rc_base}，"
                                f"变异副本 rc={rc_mut}）", ok3,
                                "" if ok3 else "rc=0：`main()` 最后一行还在 `return 1 if failed`？"
                                               "或者它压根没调 `_exit_code`"))
            else:
                results.append((f"H1 → 返回码对比：**这次做不了** —— 原样副本自己就 rc={rc_base}"
                                f"（库当前是红的）。判据没问题，是库要先修；"
                                f"上面两条归因明确的用例照跑", True, ""))

    print("\n主闸返回码（main 的判定式）子进程回测：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\nmain 返回码：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ===================== `--selftest` **自己有多少条用例**（第三十轮，再往上一层）
#
# 本轮把「删一条断言」堵在 `RULESET_SHAPE["min_assertions"]` 上之后，同一个形状
# 在**自检这一侧**原样还在：`--selftest` 打的是「188 / 188 通过」，
# 可全项目**没有任何东西数一数自检该有多少条**。删掉几条用例 —— 比如刚加的
# H1/H2/H3 或 SOLE 组 —— 主闸绿、自检也绿，只是分母悄悄变小，
# 而分母是印在屏幕上、没人逐轮比对的一个数。这就是作弊路径 A 打在自检上。
#
# ⇒ 每组的条数在这里登记。绕过它要**两处**改动（删用例 + 改这里的数），
#   两处都落在 assert_resolutions.py 的 diff 上、都写着「我把用例摘了」，是自证的。
#   ⚠ 只判「不许变少」：加用例不必先改这里（`>` 会提示，不判失败）。
# =============== 第三十一轮 L5-②：`--selftest` 从来不调用 22 个执行体里的 11 个
#
# 复核实测：给全部 `a_*` 套上调用计数器再跑 `--selftest`（那时 203/203 全绿），
# **调用次数为 0 的有 11 个** ——
#   anchor_ids · cn_absent · distinct_terms · exclusions_closed · glossary_value ·
#   lang_parity · leaf_literal · no_bilingual_tail · term_domains · term_gated · version_matrix
# 它们覆盖 **66 条断言里的 48 条（72.7%）**。把任意一个整体换成 `return [], ""`：
# 主闸 66/0/0、`--selftest` 203/203，**一处改动、两道闸都不响**。
# 而它们本来在干活（复核实证）：`a_cn_absent` 注入 `cn:['的']` → 16387 处违规；
# `a_term_gated` 注入不可能的 `cn_required` → 15815 处；`exclusions_closed` 现场扫 14191
# 条唯一英文串；`anchor_ids` 1625 个；`lang_parity` 1842/486 对。
#
# 根因与 L5-① 同型：`SELFTEST_SHAPE` 数的是「**已登记的组**各有几条用例」，
# **没有任何东西要求「`KINDS` 里每个 kind 都得有一组、且那一组真的调到了它**」。
# ⚠ 这正是第三十轮给 `min_kinds` 补 `min_by_kind` 的那个形状，原样长在自检的「组」一级 ——
#   第三十轮的通则「每加一层数一数，都要问自检那一侧的同一个数谁在守」，
#   **当时问的是分母（组里有几条），没问分子（有没有这一组、这一组打到东西没有）**。
#
# ⚠ **名册不够，必须用计数器。** `glossary_value` 那一组是**有的**，它打的是
#   `glossary_value_matches` 这个纯函数；`双语尾巴` 打 `has_bilingual_tail`；
#   `读库闸` 打 `_gate_one` —— 三组都在跑，三个执行体一次没被调用。
#   所以下面这张表配的是 `_KIND_CALLS`（真实调用次数），不是「有没有一组同名用例」。
KIND_SELFTEST_OWNER = {
    # 第三十轮就有的（这些组本来就在调执行体）
    "sense_gated": "sense_gated",
    "block_aligned_gate": "block_aligned_gate",
    "block_sense_gate": "block_sense_gate",
    "tag_parity": "tag_parity",          # 第三十三轮 B
    "enricher_slot_gate": "enricher_slot_gate",
    "enricher_text_coverage": "enricher_text_coverage",
    "twin_files": "twin_files",
    "translate_cases": "translate_cases",
    "source_literal": "source_literal",
    "panel_liveness": "panel_liveness",
    "tracked_inputs": "tracked_inputs",
    "ruleset_shape": "ruleset_shape",
    # 第三十一轮补的 11 个（此前调用次数全是 0）
    "term_gated": "叶级执行体",
    "cn_absent": "叶级执行体",
    "distinct_terms": "叶级执行体",
    "term_domains": "叶级执行体",
    "anchor_ids": "叶级执行体",
    "no_bilingual_tail": "叶级执行体",
    "leaf_literal": "叶级执行体",
    "glossary_value": "读盘执行体",
    "lang_parity": "读盘执行体",
    "version_matrix": "读盘执行体",
    "exclusions_closed": "读盘执行体",
}

# ⚠ **「还差几个」必须是绿闸下的一个数，不能是隐式的。**
#   本轮 11 个 kind 全部补齐了执行体正反例，所以这张待补表是空的；
#   但它**故意留着**：下一轮加了新 kind 却只写了执行体、没写用例时，
#   正确的动作是往这里登记一行（连同理由），而不是让覆盖表悄悄少一格。
#   ⚠ 登记在这里的 kind **仍然会被覆盖表点名**（只是不判失败），
#     `_kind_coverage_check` 会把「已登记 N / 待补 M」两个数一起打出来。
KIND_SELFTEST_TODO = {}

# ⚠ **本护栏自己的单点绕法**：把一个 kind 塞进 KIND_SELFTEST_TODO，覆盖表就只出提示不判失败 ——
#   那等于一处改动就把这一层关掉。所以待补条数**自己也有个地板**（现在是 0）：
#   要真登记一条待补，就得同时把这个上限抬到 1，两处都在本文件、都是自证的 diff。
KIND_SELFTEST_TODO_MAX = 0


# =============== 第三十二轮 L6-①：**这道判定函数自己一条反例都没有**
#
# 独立复核逐个掏空实测（第三十一轮那四道新判定函数）：
#   掏空 `_payload_floor_check`  → rc=1 · 248/285 · FAIL 38   ← 堵住了（有分层归因用例）
#   掏空 `_units_verdict`        → rc=1 · 282/285 · FAIL 3    ← 堵住了（有 7 条直打用例）
#   掏空 `_kind_coverage_check`  → rc=0 · 285/285 · FAIL 0    ← ⚠⚠ 一处改动、两道闸全绿
#   掏空 `_selftest_shape_check` → rc=0 · 285/285 · FAIL 0    ← ⚠⚠ 一处改动、两道闸全绿
#
# 根因与 L5-② **完全同型、只是又高了一层**：`run_selftest()` 里
# `"kind 覆盖表": 1` 是**写死的字面量 1**、`total += len(SELFTEST_SHAPE)` 数的是**表长**，
# 都不是从函数结果推出来的；而这两组**只有正例、一条反例都没有**。
# ⇒ 两处修：① 下面两个函数**参数化**（默认取模块级常量），好让反例喂进合成的名册／计数器；
#          ② 它们各自返回「这次真的比对了几项」，`run_selftest()` 用**这个数**当分母，
#            并断言它等于名册长度 —— 掏空成 `return [], "", 0` 时分母掉到 0，当场判失败。
def _kind_coverage_check(owner=None, todo=None, todo_max=None, calls=None, kinds=None):
    """`KINDS` 里每个 kind 都得有一组用例、且那一组这次**真的调用过它**。

    返回 (违规行, 说明行, 这次比对了几项)。
    ⚠ 判的是 `_KIND_CALLS`（本进程实跑的调用次数），不是名册 —— 见上面那段。
    ⚠ 五个参数只给 `--selftest` 的反例用（默认全部取模块级常量）：没有它们就喂不进
      「摘掉一个 kind 的归属」「某 kind 调用次数为 0」这类反例，而没有反例的判据
      **掏空之后两道闸都不会响** —— 这正是 L6-① 的全部内容。
    """
    tab = KIND_SELFTEST_OWNER if owner is None else owner
    todos = KIND_SELFTEST_TODO if todo is None else todo
    tmax = KIND_SELFTEST_TODO_MAX if todo_max is None else todo_max
    calls = _KIND_CALLS if calls is None else calls
    ks = KINDS if kinds is None else kinds
    bad, notes = [], []
    n_checked = 0                      # 「这次真的比对了几项」—— run_selftest() 拿它当分母
    for kind in sorted(ks):
        n_checked += 1
        who = tab.get(kind)
        is_todo = kind in todos
        n = calls.get(kind, 0)
        if who is None and not is_todo:
            bad.append(f"`{kind}` 在 KIND_SELFTEST_OWNER 里没有归属组 —— "
                       f"加了新 kind 就得同时给它一组用例，或显式登记进 KIND_SELFTEST_TODO "
                       f"（「还差几个」得是个看得见的数）")
            continue
        if is_todo:
            notes.append(f"`{kind}` 显式待补：{todos[kind]}（本次调用 {n} 次）")
            continue
        if n == 0:
            bad.append(f"`{kind}` 的执行体这次**一次都没被调用**（归属组 `{who}`）—— "
                       f"「有一组用例」不等于「那一组打到了执行体」：`glossary_value` 那一组"
                       f"打的是纯函数 `glossary_value_matches`，执行体调用次数是 0。"
                       f"把它整体换成 `return [], ''` 时，主闸与自检都不会响")
    for kind in sorted(set(tab) - set(ks)):
        n_checked += 1
        bad.append(f"覆盖表里的 `{kind}` 不在 KINDS 里 —— 表写歪了（或 kind 改名了）")
    n_checked += 1                     # 待补清单的上限，也算比对了一项
    if len(todos) > tmax:
        bad.append(f"待补清单有 {len(todos)} 条，超过登记的上限 "
                   f"{tmax} —— 「把一个 kind 塞进待补清单」不许是单点动作："
                   f"要登记待补，就同时把 KIND_SELFTEST_TODO_MAX 抬上去，那处 diff 是自证的")
    done = sum(1 for k in ks if k not in todos)
    return bad, (f"kind 执行体覆盖：已登记 {done} / 待补 {len(todos)} / "
                 f"共 {len(ks)} 种；本次实跑调用 "
                 f"{sum(1 for k in ks if calls.get(k, 0))} 种"), n_checked


# ------------------------------------------- L6-① 的反例组（kind 覆盖表 + 自检形状）
#
# ⚠ 这一组存在的唯一理由：上面那两道判定函数**在第三十一轮只有正例**。
#   只有正例时，把函数整体换成 `return [], "", 0` 这种空壳，主闸 66/0/0、自检 285/285,
#   **一处改动、两道闸全绿** —— 复核逐个掏空实测过，见本节顶上那张表。
#   每条都写成「喂一份坏掉的合成名册 / 计数器 → 必须响」，正例也留着（防恒真）。
def _kind_cov_cases():
    """返回 [(说明, 是否通过, 出错时的补充)]。"""
    live = dict(_KIND_CALLS)                       # 本次自检真实跑出来的调用次数
    out = []

    b, _n, ncase = _kind_coverage_check(calls=live)
    ok = (not b) and ncase == len(KINDS) + 1
    out.append((f"正例：原样的名册 + 本次真实调用次数 → 不响，且比对项数 {ncase} "
                f"= {len(KINDS)} 种 kind + 1 道待补上限", ok,
                "" if ok else f"违规 {b[:2]}；比对项数 {ncase}"))

    victim = sorted(KINDS)[0]
    b, _n, _c = _kind_coverage_check(owner={k: v for k, v in KIND_SELFTEST_OWNER.items()
                                            if k != victim}, calls=live)
    ok = any(f"`{victim}` 在 KIND_SELFTEST_OWNER 里没有归属组" in x for x in b)
    out.append((f"反例①：从覆盖表里摘掉 `{victim}` 的归属 → 必须响", ok,
                "" if ok else f"实得 {b[:2]}"))

    b, _n, _c = _kind_coverage_check(calls=dict(live, **{victim: 0}))
    ok = any(f"`{victim}` 的执行体这次**一次都没被调用**" in x for x in b)
    out.append((f"反例②：把 `{victim}` 的实跑调用次数按成 0（= 执行体被换成空壳）→ 必须响",
                ok, "" if ok else f"实得 {b[:2]}"))

    b, _n, _c = _kind_coverage_check(todo={victim: "本轮先不补"}, todo_max=0, calls=live)
    ok = any("超过登记的上限" in x for x in b)
    out.append(("反例③：往待补清单塞一条而**不抬** KIND_SELFTEST_TODO_MAX → 必须响"
                "（「把 kind 塞进待补清单」不许是单点动作）", ok,
                "" if ok else f"实得 {b[:2]}"))

    b, _n, _c = _kind_coverage_check(owner=dict(KIND_SELFTEST_OWNER, 早就没有的kind="X"),
                                     calls=live)
    ok = any("不在 KINDS 里" in x for x in b)
    out.append(("反例④：覆盖表里留着一个 KINDS 里已经没有的 kind（改名/删除的残留）→ 必须响",
                ok, "" if ok else f"实得 {b[:2]}"))
    return out


def _shape_cases():
    """`_selftest_shape_check` 的正反例。返回 [(说明, 是否通过, 补充)]。"""
    full = {k: v for k, v in SELFTEST_SHAPE.items()}
    out = []

    b, notes, n = _selftest_shape_check(full)
    ok = (not b) and n == len(SELFTEST_SHAPE)
    out.append((f"正例：每组条数正好等于登记值 → 不响，且比对项数 {n} = 登记的 "
                f"{len(SELFTEST_SHAPE)} 组", ok, "" if ok else f"违规 {b[:2]}；比对项数 {n}"))

    victim = sorted(SELFTEST_SHAPE)[0]
    b, _t, _n = _selftest_shape_check({k: v for k, v in full.items() if k != victim})
    ok = any(f"登记过的自检组 `{victim}` 这次**没跑**" in x for x in b)
    out.append((f"反例①：**删掉一整组用例**（`{victim}` 这次没跑）→ 必须响 —— "
                f"这就是「分母悄悄变小」那条作弊路径", ok, "" if ok else f"实得 {b[:2]}"))

    b, _t, _n = _selftest_shape_check(dict(full, **{victim: full[victim] - 1}))
    ok = any(f"`{victim}` 只剩" in x for x in b)
    out.append((f"反例②：`{victim}` 少一条用例（登记 {full[victim]} 只剩 {full[victim] - 1}）"
                f"→ 必须响", ok, "" if ok else f"实得 {b[:2]}"))

    b, notes, _n = _selftest_shape_check(dict(full, 新加的一组=3))
    ok = (not b) and any("新组" in x for x in notes)
    out.append(("正例②：**加**一组新用例 → 只出提示、不判失败（加用例不该被罚）", ok,
                "" if ok else f"违规 {b[:2]}；提示 {notes[:2]}"))
    return out


def run_guard_cases_selftest():
    """返回 (失败条数, 总条数)。L6-① 的反例组。"""
    results = _kind_cov_cases() + _shape_cases()
    print("\n两道护栏自己的反例（kind 覆盖表 / 自检形状，第三十二轮 L6-①）：")
    nbad = 0
    for note, ok, extra in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if extra:
            print(f"        {extra}")
    print(f"\n护栏反例：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ------------------------------------------- 叶级执行体的正反例（7 个 kind）
#
# 全部走 `_FakeCtx`（合成的叶／lang 键），不碰真实树 —— 与既有各组同一口径。
# ⚠ 每条都是**成对**的：正例证明它不误报，反例证明它真的会响。
#   只有反例的话，`return [], ""` 那种空壳会在正例上「通过」；
#   只有正例的话，恒真的判据也能全绿。
_LEAF_P = ("ember", "p.json", "entries.A.name")


def _lp(en, cn, path=None):
    return [("ember", "p.json", path or _LEAF_P[2], en, cn)]


LEAF_KIND_SELFTEST = [
    # (说明, 执行体, 规则, ctx, 期望违规数)
    ("term_gated 正例：英文命中且中文含定译 → 不响",
     "term_gated", {"id": "T", "kind": "term_gated", "en": r"\bWarlock\b",
                    "cn_required": "邪术师", "min_hits": 1},
     _FakeCtx(pairs=_lp("The Warlock casts", "邪术师施放")), 0),
    ("term_gated 反例：英文命中而中文没有定译 → 响",
     "term_gated", {"id": "T", "kind": "term_gated", "en": r"\bWarlock\b",
                    "cn_required": "邪术师", "min_hits": 1},
     _FakeCtx(pairs=_lp("The Warlock casts", "术士施放")), 1),
    ("term_gated 反例：中文混进了禁用变体 → 响（**这一半就是清空 cn_forbidden 会丢掉的**）",
     "term_gated", {"id": "T", "kind": "term_gated", "en": r"\bLantyr",
                    "cn_required": "兰提尔", "cn_forbidden": ["兰蒂尔"], "min_hits": 1},
     _FakeCtx(pairs=_lp("Lantyr city", "兰提尔城（旧译兰蒂尔）")), 1),
    ("term_gated 反例：英文闸一叶都没命中 → min_hits 把空转抓出来",
     "term_gated", {"id": "T", "kind": "term_gated", "en": r"\bWarlock\b",
                    "cn_required": "邪术师", "min_hits": 1},
     _FakeCtx(pairs=_lp("nothing here", "无关")), 1),

    ("cn_absent 正例：全库没有这个废写法 → 不响",
     "cn_absent", {"id": "T", "kind": "cn_absent", "cn": "六角格"},
     _FakeCtx(pairs=_lp("hex tile", "六边格")), 0),
    ("cn_absent 反例：compendium 侧出现废写法 → 响",
     "cn_absent", {"id": "T", "kind": "cn_absent", "cn": "六角格"},
     _FakeCtx(pairs=_lp("hex tile", "六角格")), 1),
    ("cn_absent 反例：**lang 侧**出现废写法 → 也得响（第十六轮扩面扩的就是这一侧）",
     "cn_absent", {"id": "T", "kind": "cn_absent", "cn": "六角格"},
     _FakeCtx(lang=[("ember", "K1", "hex", "六角格")]), 1),
    ("cn_absent 反例：一组禁用词里只中了第二个 → 也得响（禁用词是**一组**，不是一个）",
     "cn_absent", {"id": "T", "kind": "cn_absent", "cn": ["令牌", "代币"]},
     _FakeCtx(pairs=_lp("Token", "代币")), 1),
    ("cn_absent 反例：可选英文闸命中 0 → 这条已经不设防，min_hits 抓出来",
     "cn_absent", {"id": "T", "kind": "cn_absent", "cn": "六角格",
                   "en": r"\bhex\b", "min_hits": 1},
     _FakeCtx(pairs=_lp("nothing", "无关")), 1),

    ("distinct_terms 正例：两个术语中文不同且都过了 lang 闸 → 不响",
     "distinct_terms", {"id": "T", "kind": "distinct_terms", "scan": ["lang"], "min_hits": 2,
                        "terms": [{"en": "Round", "cn": "轮"}, {"en": "Turn", "cn": "回合"}]},
     _FakeCtx(lang=[("ember", "K1", "Round", "第一轮"), ("ember", "K2", "Turn", "回合")]), 0),
    ("distinct_terms 反例：两个英文共用同一个中文 → 响（撞名）",
     "distinct_terms", {"id": "T", "kind": "distinct_terms", "scan": ["lang"], "min_hits": 1,
                        "terms": [{"en": "Round", "cn": "回合"}, {"en": "Turn", "cn": "回合"}]},
     _FakeCtx(lang=[("ember", "K1", "Round", "回合"), ("ember", "K2", "Turn", "回合")]), 1),
    ("distinct_terms 反例：**没有 `scan` = 只比配置不读库** → 直接判失败"
     "（R-region-area-map 空转四个版本的那种形态）",
     "distinct_terms", {"id": "T", "kind": "distinct_terms",
                        "terms": [{"en": "Round", "cn": "轮"}, {"en": "Turn", "cn": "回合"}]},
     _FakeCtx(lang=[("ember", "K1", "Round", "轮")]), 1),

    ("term_domains 正例：两个域各自的 lang 键都写对了 → 不响",
     "term_domains", {"id": "T", "kind": "term_domains", "term": "Aura", "min_hits": 2,
                      "domains": [{"name": "机制", "lang": {"ember": {"K1": "光环"}},
                                   "cn_forbidden": ["灵光"]},
                                  {"name": "叙述", "lang": {"ember": {"K2": "灵光"}}}]},
     _FakeCtx(lang=[("ember", "K1", "Aura", "光环"), ("ember", "K2", "aura", "灵光")]), 0),
    ("term_domains 反例：域里混进了别域的写法 → 响（防「顺手统一」）",
     "term_domains", {"id": "T", "kind": "term_domains", "term": "Aura", "min_hits": 1,
                      "domains": [{"name": "机制", "lang": {"ember": {"K1": "光环"}},
                                   "cn_forbidden": ["灵光"]}]},
     _FakeCtx(lang=[("ember", "K1", "Aura", "光环／灵光")]), 1),
    ("term_domains 反例：域点名的 lang 键不见了 → 响（域失去看守，不是通过）",
     "term_domains", {"id": "T", "kind": "term_domains", "term": "Aura", "min_hits": 1,
                      "domains": [{"name": "机制", "lang": {"ember": {"K9": "光环"}}}]},
     _FakeCtx(lang=[("ember", "K1", "Aura", "光环")]), 2),

    ("anchor_ids 正例：显式 id 数达阈值 → 不响",
     "anchor_ids", {"id": "T", "kind": "anchor_ids", "min": 1},
     _FakeCtx(pairs=_lp("x", '<h2 id="a">标题</h2>')), 0),
    ("anchor_ids 反例：一个显式 id 都没有 → 响（锚点链接会整片断掉）",
     "anchor_ids", {"id": "T", "kind": "anchor_ids", "min": 1},
     _FakeCtx(pairs=_lp("x", "<h2>标题</h2>")), 1),

    ("no_bilingual_tail 正例：该字段是裸中文 → 不响",
     "no_bilingual_tail", {"id": "T", "kind": "no_bilingual_tail", "fields": ["tokenName"]},
     _FakeCtx(pairs=_lp("Scout", "斥候", "actors.X.tokenName")), 0),
    ("no_bilingual_tail 反例：该字段带了双语尾巴 → 响",
     "no_bilingual_tail", {"id": "T", "kind": "no_bilingual_tail", "fields": ["tokenName"]},
     _FakeCtx(pairs=_lp("Gore Bird", "血鸟 Gore Bird", "actors.X.tokenName")), 1),
    ("no_bilingual_tail 特异度：**不在 fields 里的字段不看** —— 这一条同时说明"
     "「把 fields 清空 ⇒ 0 违规 ⇒ 绿」为什么骗得过主闸（那一半由规矩数地板接）",
     "no_bilingual_tail", {"id": "T", "kind": "no_bilingual_tail", "fields": ["tokenName"]},
     _FakeCtx(pairs=_lp("Gore Bird", "血鸟 Gore Bird", "actors.X.name")), 0),

    ("leaf_literal 正例：钉住的叶含必含串 → 不响",
     "leaf_literal", {"id": "T", "kind": "leaf_literal", "pack": "p.json", "path": "A.name",
                      "must_contain": ["三叉戟角"], "min_leaves": 1},
     _FakeCtx(pairs=[("ember", "p.json", "entries.A.name", "Trident's Point", "三叉戟角")]), 0),
    ("leaf_literal 反例：必含串没了 → 响",
     "leaf_literal", {"id": "T", "kind": "leaf_literal", "pack": "p.json", "path": "A.name",
                      "must_contain": ["三叉戟角"], "min_leaves": 1},
     _FakeCtx(pairs=[("ember", "p.json", "entries.A.name", "Trident's Point", "三叉戟点")]), 1),
    ("leaf_literal 反例：那几叶找不到了（上游改名／路径写错／孪生包只钉了一半）→ "
     "min_leaves 抓出来，不许按「0 叶待查 = 通过」放行",
     "leaf_literal", {"id": "T", "kind": "leaf_literal", "pack": "p.json", "path": "A.name",
                      "must_contain": ["三叉戟角"], "min_leaves": 1},
     _FakeCtx(pairs=[("ember", "p.json", "entries.B.name", "Other", "别的")]), 1),
]


def run_leaf_kind_selftest():
    """返回 (失败条数, 总条数)。7 个此前**一次都没被自检调用过**的叶级执行体。"""
    print("\n叶级执行体正反例（term_gated / cn_absent / distinct_terms / term_domains /"
          " anchor_ids / no_bilingual_tail / leaf_literal —— 此前调用次数全是 0）：")
    nbad = 0
    for note, kind, rule, ctx, want in LEAF_KIND_SELFTEST:
        try:
            b, detail = KINDS[kind](rule, ctx)
            ok = len(b) == want
        except Exception as exc:                       # noqa: BLE001
            b, detail, ok = [], f"执行出错 {exc!r}", False
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want}，实得 {len(b)}　（{detail}）")
    print(f"\n叶级执行体：{len(LEAF_KIND_SELFTEST) - nbad} / {len(LEAF_KIND_SELFTEST)} 通过")
    return nbad, len(LEAF_KIND_SELFTEST)


# ------------------------------------------- 读盘执行体的正反例（4 个 kind）
#
# 这四个不走 `ctx.pairs`，而是读**磁盘上的文件**（词表 / lang / PROJECT.md / 归档豁免表），
# 所以自检得给它们搭一棵**临时树**并临时改 `ROOT`。
# ⚠ 每个 kind 都必须有一条「输入缺失时必须判失败」的用例 —— 这四个执行体里，
#   「文件不在 ⇒ 0 条待查 ⇒ 通过」正是空转形态 (e) 最容易复发的地方。
class _RootPatch:
    """临时把模块级 `ROOT` 指到一棵合成树上（`--root` 走的也是这个全局）。"""

    def __init__(self, root):
        self.root = root

    def __enter__(self):
        global ROOT
        self._old = ROOT
        ROOT = self.root
        return self.root

    def __exit__(self, *a):
        global ROOT
        ROOT = self._old
        return False


class _DiskCtx:
    """给读盘类执行体用的最小 ctx：只有 `.repos`。"""

    def __init__(self, repos=None):
        self.repos = repos or {}


def _mk(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _mkjson(path, obj):
    _mk(path, json.dumps(obj, ensure_ascii=False))


def run_disk_kind_selftest():
    """返回 (失败条数, 总条数)。glossary_value / lang_parity / version_matrix /
    exclusions_closed —— 四个此前**一次都没被自检调用过**的读盘执行体。"""
    print("\n读盘执行体正反例（glossary_value / lang_parity / version_matrix /"
          " exclusions_closed —— 此前调用次数全是 0）：")
    results = []

    def case(note, kind, rule, ctx, want, root=None):
        try:
            if root:
                with _RootPatch(root):
                    b, detail = KINDS[kind](rule, ctx)
            else:
                b, detail = KINDS[kind](rule, ctx)
            ok = len(b) == want
        except Exception as exc:                       # noqa: BLE001
            b, detail, ok = [], f"执行出错 {exc!r}", False
        results.append((note, ok, want, len(b), detail))

    import shutil
    tmp = tempfile.mkdtemp(prefix="ar_disk_selftest_")
    try:
        # ① glossary_value：两层（base + 产物）都得核
        g = os.path.join(tmp, "g")
        _mkjson(os.path.join(g, "base.json"), {"Cosmology": "宇宙观"})
        _mkjson(os.path.join(g, "built.json"), {"Cosmology": "宇宙观 Cosmology"})
        _mkjson(os.path.join(g, "laundered.json"), {"Cosmology": "宇宙"})
        gr = {"id": "T", "kind": "glossary_value",
              "files": {"base": "g/base.json", "built": "g/built.json"},
              "entries": {"Cosmology": "宇宙观"}, "min_checked": 2}
        case("glossary_value 正例：两层都是登记值（产物层带双语尾巴也算对）→ 不响",
             "glossary_value", gr, _DiskCtx(), 0, tmp)
        case("glossary_value 反例：产物层被洗成了别的值 → 响",
             "glossary_value", dict(gr, files={"base": "g/base.json",
                                               "built": "g/laundered.json"}),
             _DiskCtx(), 1, tmp)
        case("glossary_value 反例：**两层文件都不在** → 必须响（无从查起 ≠ 查过了没问题，"
             "空转形态 (e)）",
             "glossary_value", dict(gr, files={"base": "g/nope1.json", "built": "g/nope2.json"}),
             _DiskCtx(), 1, tmp)
        case("glossary_value 反例：词表里缺这个锚点键 → 响",
             "glossary_value", dict(gr, entries={"NoSuchKey": "无"}, min_checked=1),
             _DiskCtx(), 3, tmp)

        # ② lang_parity：两侧键数必须相等
        lp = os.path.join(tmp, "lp")
        _mkjson(os.path.join(lp, "repo", "lang", "cn.json"), {"A": "甲", "B": "乙"})
        _mkjson(os.path.join(lp, "up", "lang", "en.json"), {"A": "a", "B": "b"})
        _mkjson(os.path.join(lp, "up2", "lang", "en.json"), {"A": "a", "B": "b", "C": "c"})
        lctx = _DiskCtx({"ember": os.path.join(lp, "repo")})
        case("lang_parity 正例：cn 2 键 / en 2 键 → 不响", "lang_parity",
             {"id": "T", "kind": "lang_parity", "packages": {"ember": os.path.join(lp, "up")}},
             lctx, 0)
        case("lang_parity 反例：英文侧多一个键（上游加了词条没跟）→ 响", "lang_parity",
             {"id": "T", "kind": "lang_parity", "packages": {"ember": os.path.join(lp, "up2")}},
             lctx, 1)
        case("lang_parity 反例：lang 文件找不到 → 必须响（不是通过）", "lang_parity",
             {"id": "T", "kind": "lang_parity",
              "packages": {"ember": os.path.join(lp, "nope")}}, lctx, 1)

        # ③ version_matrix：module.json ↔ PROJECT.md
        vm = os.path.join(tmp, "vm")
        _mkjson(os.path.join(vm, "repo", "module.json"), {"id": "ember-cn", "version": "0.9.9"})
        _mk(os.path.join(vm, "PROJECT.md"),
            "# 标题\n当前已发布：`ember-cn` v0.9.9\n\n| 包 | 版本 |\n| ember-cn | v0.9.9 |\n")
        _mk(os.path.join(vm, "STALE.md"),
            "# 标题\n当前已发布：`ember-cn` v0.9.4\n\n| 包 | 版本 |\n| ember-cn | v0.9.4 |\n")
        vctx = _DiskCtx({"ember": os.path.join(vm, "repo")})
        vr = {"id": "T", "kind": "version_matrix", "doc": "PROJECT.md",
              "tag_prefix": {"ember": "v"}, "min_repos": 1}
        case("version_matrix 正例：抬头 / 正文「当前」行 / 版本矩阵三处都是 v0.9.9 → 不响",
             "version_matrix", vr, vctx, 0, vm)
        case("version_matrix 反例：文档整片停在 v0.9.4（历史上漂过两次的那种）→ 响",
             "version_matrix", dict(vr, doc="STALE.md"), vctx, 3, vm)
        case("version_matrix 反例：仓里没有 module.json → 必须响"
             "（这个仓的版本**没核成**，不是通过）",
             "version_matrix", vr, _DiskCtx({"ember": os.path.join(vm, "nope")}), 2, vm)

        # ④ exclusions_closed：现场重跑扫描器 + 归档豁免表闭合
        ex = os.path.join(tmp, "ex")
        rp = os.path.join(ex, "repo")
        _mkjson(os.path.join(rp, "compendium", "en", "x.json"),
                {"entries": {"A": {"name": "Shield"}, "B": {"name": "Shield"}}})
        _mkjson(os.path.join(rp, "compendium", "cn", "x.json"),
                {"entries": {"A": {"name": "护盾术"}, "B": {"name": "盾牌"}}})
        _mkjson(os.path.join(ex, "clean.json"), [{"what": "`Shield`", "why": "合法分叉"}])
        _mkjson(os.path.join(ex, "empty.json"), [])
        ectx = _DiskCtx({"ember": rp})
        er = {"id": "T", "kind": "exclusions_closed", "exclusions": "ex/clean.json",
              "min_en_strings": 1, "short_len": 40, "long_keys": ["Adelyne"]}
        case("exclusions_closed 正例：现场扫出的分叉组已在归档豁免表里 → 不响",
             "exclusions_closed", er, ectx, 0, tmp)
        case("exclusions_closed 反例：**豁免表空了**，同一组分叉就成了未归档 → 响"
             "（第四层的 (d) 形态：闸在跑但豁免表已空）",
             "exclusions_closed", dict(er, exclusions="ex/empty.json"), ectx, 1, tmp)
        case("exclusions_closed 反例：归档豁免表整个不在 → 必须响"
             "（表没了就等于没设防，不许静默通过）",
             "exclusions_closed", dict(er, exclusions="ex/nope.json"), ectx, 1, tmp)
        case("exclusions_closed 反例：扫描器跑不起来 → 判失败而不是判通过",
             "exclusions_closed", dict(er, scanner="no_such_scanner.py"), ectx, 1, tmp)
        case("exclusions_closed 反例：扫描器只报出 0 条英文唯一串 → min_en_strings 抓空转",
             "exclusions_closed", dict(er, min_en_strings=99999), ectx, 1, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    nbad = 0
    for note, ok, want, got, detail in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want}，实得 {got}　（{str(detail)[:150]}）")
    print(f"\n读盘执行体：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


# ------------------------------------------- 载荷地板与规矩数这两道**新护栏自己**的回测
def run_payload_floor_selftest():
    """返回 (失败条数, 总条数)。

    ⚠ 这一组钉的是复核实测的那 12 个「主闸全绿、护栏却真的没了」的单点动作 ——
      每一个在这里都必须**变红**。外加两条自证：
        · 登记表与规则集逐条对齐（否则本表自己就是个空壳，跟 REGISTERED_ASSERTIONS 同理）；
        · 原样规则集必须 0 违规（否则它是恒红的，等于没有）。
    """
    print("\n载荷地板（PAYLOAD_FLOORS）正反例：")
    results = []
    try:
        base = json.load(open(DEFAULT_RULES, encoding="utf-8"))
    except Exception as exc:                           # noqa: BLE001
        print(f"  FAIL 前置：规则文件读不动 {exc!r}")
        print("\n载荷地板：0 / 1 通过")
        return 1, 1
    A0 = base["assertions"]

    # 前置：两张登记表 id 集合必须相等，且都等于发布中的规则集
    live = {r.get("id") for r in A0}
    okP = set(PAYLOAD_FLOORS) == live == set(REGISTERED_ASSERTIONS)
    results.append((f"前置：PAYLOAD_FLOORS（{len(PAYLOAD_FLOORS)} 条）= REGISTERED_ASSERTIONS"
                    f"（{len(REGISTERED_ASSERTIONS)} 条）= 规则集（{len(live)} 条）"
                    f" —— 三者必须逐条相等，否则本表自己就是空壳", okP,
                    "" if okP else f"载荷表多出 {sorted(set(PAYLOAD_FLOORS) - live)[:4]}；"
                                   f"规则集多出 {sorted(live - set(PAYLOAD_FLOORS))[:4]}"))
    # 前置 C：**登记值必须 = 从发布中的规则集现推出来的值。**
    #   ⚠ 这一条堵的是本轮新护栏**自己**的单点绕法：把某一道地板改小
    #     （`("list", 15)` → `("list", 0)`）是一处改动，PAYLOAD_FLOORS 自己看不出来，
    #     上面那条「原样 → 0 违规」也照样过（地板越低越容易过）。
    #     与第三十轮给 `min_by_kind` 用的是同一招：让登记值可现推，然后断言两者相等。
    #     ⇒ 改小一道地板 ⇒ 与现推值对不上 ⇒ 自检红；
    #       合法地删内容 ⇒ 规则集与本表两处一起改 ⇒ diff 自证。
    derived = _derive_payload_floors(A0)
    okD = derived == PAYLOAD_FLOORS
    diff = [] if okD else sorted(
        f"{rid}.{k}: 登记 {PAYLOAD_FLOORS.get(rid, {}).get(k)} vs 现推 {v}"
        for rid, spec in derived.items() for k, v in spec.items()
        if PAYLOAD_FLOORS.get(rid, {}).get(k) != v)
    results.append((f"前置 C：{len(PAYLOAD_FLOORS)} 条 / "
                    f"{sum(len(v) for v in PAYLOAD_FLOORS.values())} 道地板必须**逐道等于**"
                    f"从规则集现推的值 —— 否则「把某道地板改小」就是个单点动作", okD,
                    "" if okD else f"对不上 {len(diff)} 道：{diff[:4]}"))

    # 前置：每条登记的地板必须 = 当前实际值可以满足（原样 → 0 违规）
    b0, n0 = _payload_floor_check(A0)
    ok0 = not b0
    results.append((f"前置：原样规则集过 {n0} 道载荷地板 → 0 违规（防恒红）", ok0,
                    "" if ok0 else f"{[(x[2], x[3][:60]) for x in b0][:4]}"))

    def mut(fn, want_id_key=None):
        r = copy.deepcopy(A0)
        fn(r)
        b, _n = _payload_floor_check(r)
        hit = [x for x in b if want_id_key is None or want_id_key in str(x[2])]
        return b, hit

    def _idx(rules, rid):
        return next(i for i, r in enumerate(rules) if r.get("id") == rid)

    def case(note, fn, key):
        b, hit = mut(fn, key)
        ok = bool(hit)
        results.append((note, ok, "" if ok else f"没响：实得 {len(b)} 处违规 "
                                                f"{[(x[2]) for x in b][:4]}"))

    # ↓↓↓ 复核实测的那 12 个「真丢护栏」动作，逐条复现 ↓↓↓
    def _shell(r):
        i = _idx(r, "R-hex-tile")
        r[i] = {"id": "R-hex-tile", "kind": "cn_absent", "cn": ["绝不可能XYZ"]}
    case("① 把 `R-hex-tile` **整条**换成只剩 id/kind/cn 的壳"
         "（id 对、kind 对、条数对、类数对 —— 上面三层全都满意）",
         _shell, "R-hex-tile")

    def _wipe_cn_absent(r):
        for x in r:
            if x.get("kind") == "cn_absent":
                x["cn"] = [] if isinstance(x.get("cn"), list) else ""
    case("② **15 条 `cn_absent` 的禁用词表一次全部清空**（全库零容忍的那些词一个不剩）",
         _wipe_cn_absent, "R-")
    case("③ `R-bare-cn-fields.fields=[]`（detail 从「检查 1443 个叶」塌成「检查 0 个叶」）",
         lambda r: r[_idx(r, "R-bare-cn-fields")].__setitem__("fields", []), "R-bare-cn-fields")
    case("④ `R-lang-parity.packages={}`（detail 变成空字符串）",
         lambda r: r[_idx(r, "R-lang-parity")].__setitem__("packages", {}), "R-lang-parity")
    for rid in ("R-lantyr", "R-orb-destroyed", "R-obsidian-antiquary", "R-temple-lunarium"):
        case(f"⑤ `{rid}.cn_forbidden=[]`（detail **一个字节都不变**）",
             (lambda rid=rid: lambda r: r[_idx(r, rid)].__setitem__("cn_forbidden", []))(), rid)
    case("⑥ `R-anchor-ids.min` 从 900 改成 0（阈值调松成恒真）",
         lambda r: r[_idx(r, "R-anchor-ids")].__setitem__("min", 0), "R-anchor-ids")
    case("⑦ `R-readaloud-coverage.min_sentence_frac` 从 0.75 调成 0（覆盖判据形同虚设）",
         lambda r: r[_idx(r, "R-readaloud-coverage")].__setitem__("min_sentence_frac", 0),
         "R-readaloud-coverage")
    case("⑧ `R-electricity-lightning.case_sensitive` 翻成 false（判据口径被悄悄改宽）",
         lambda r: r[_idx(r, "R-electricity-lightning")].__setitem__("case_sensitive", False),
         "R-electricity-lightning")
    case("⑨ `R-glossary-not-laundering.entries` 清空（31 个锚点键一个不查）",
         lambda r: r[_idx(r, "R-glossary-not-laundering")].__setitem__("entries", {}),
         "R-glossary-not-laundering")
    case("⑩ `R-aura-three-way.domains` 砍掉一域（三域分裂变两域）",
         lambda r: r[_idx(r, "R-aura-three-way")].__setitem__(
             "domains", r[_idx(r, "R-aura-three-way")]["domains"][:2]), "R-aura-three-way")

    # ⚠ 特异度：**加**东西不许响（地板不是天花板，否则日常收紧判据都会撞墙）
    def _grow(r):
        i = _idx(r, "R-lantyr")
        r[i]["cn_forbidden"] = list(r[i]["cn_forbidden"]) + ["兰泰尔"]
        r[i]["min_hits"] = r[i]["min_hits"] + 50
    b, _hit = mut(_grow)
    okg = not b
    results.append(("特异度：往载荷里**加**禁用词、把 min_hits 调**大** → 一声不吭"
                    "（地板不是天花板，收紧判据不该撞墙）", okg,
                    "" if okg else f"误报 {[(x[2], x[3][:50]) for x in b][:3]}"))

    # ⚠ 单元级：`_payload_one` 的五种类型各要有正反例（防「类型分支写歪了 = 恒真」）
    for note, args, want_hit in [
        ("list 地板：够长 → 不响", ("R", "k", "list", 2, [1, 2]), False),
        ("list 地板：变短 → 响", ("R", "k", "list", 2, [1]), True),
        ("str 地板：够长 → 不响", ("R", "k", "str", 2, "ab"), False),
        ("str 地板：被清空 → 响", ("R", "k", "str", 2, ""), True),
        ("ge 地板：调大 → 不响", ("R", "k", "ge", 10, 11), False),
        ("ge 地板：调小 → 响", ("R", "k", "ge", 10, 9), True),
        ("le 天花板：调小 → 不响", ("R", "k", "le", 4, 3), False),
        ("le 天花板：放大 → 响", ("R", "k", "le", 4, 5), True),
        ("eq 钉死：原值 → 不响", ("R", "k", "eq", True, True), False),
        ("eq 钉死：翻成 false → 响", ("R", "k", "eq", True, False), True),
        ("键整个不见了 → 响（这一支接住的是「整条换成壳」）",
         ("R", "k", "list", 1, _MISSING), True),
        ("⚠ 布尔不许当数用：`ge` 收到 True 必须响（否则 min_hits=True 能绕过地板）",
         ("R", "k", "ge", 1, True), True),
    ]:
        got = _payload_one(*args)
        ok = bool(got) == want_hit
        results.append((f"_payload_one {note}", ok, "" if ok else f"实得 {got!r}"))

    nbad = 0
    for row in results:
        note, ok, why = row[0], row[1], row[-1]
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if not ok:
            print(f"        {why}")
    print(f"\n载荷地板：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


def run_units_selftest():
    """返回 (失败条数, 总条数)。「这次判了几条规矩」那道地板自己的回测。

    ⚠ 这一组要钉住的核心是复核那句话：
      **数的必须是「判了几条规矩」，不是「扫了几片叶子」。**
      所以除了判定式的正反例，还要**实跑执行体**证明：
        · 清空 `cn_forbidden` 时，「英文闸命中 N 叶」一个字节都不变，而规矩数掉一条；
        · 清空 `fields` 时，叶数与规矩数**一起**掉到 0（两个数都要，别把前者当后者用）。
    """
    print("\n判了几条规矩（JUDGED_UNITS）正反例：")
    results = []

    okK = set(JUDGED_UNITS) == set(REGISTERED_ASSERTIONS)
    results.append((f"前置：JUDGED_UNITS（{len(JUDGED_UNITS)} 条）与 REGISTERED_ASSERTIONS "
                    f"逐条相等 —— 少登记一条 = 那条断言的空转没人数", okK,
                    "" if okK else f"多出 {sorted(set(JUDGED_UNITS) ^ set(REGISTERED_ASSERTIONS))[:5]}"))
    okF = all(isinstance(v, int) and v >= 1 for v in JUDGED_UNITS.values())
    results.append(("前置：每条的地板都 ≥ 1（地板写成 0 = 这条不设防）", okF,
                    "" if okF else f"{[k for k, v in JUDGED_UNITS.items() if not (isinstance(v, int) and v >= 1)]}"))
    # ⚠ **本护栏自己的单点绕法**：把某一条的地板改小（`R-lantyr: 2` → `1`）。
    #   与载荷地板不同，规矩数**推不出来**（它是运行时量出来的），所以做不成「现推 == 登记」。
    #   退而求其次记一个**合计**：改小任何一条都会让合计掉下来 ⇒ 得同时改两处（两处都在本文件、
    #   都是自证的 diff）。⚠ 这是**两点门槛，不是墙** —— 明写在这里，别当成堵死了。
    okT = sum(JUDGED_UNITS.values()) >= JUDGED_UNITS_TOTAL
    results.append((f"前置：规矩数地板的**合计**不得低于登记的 {JUDGED_UNITS_TOTAL}"
                    f"（现在 {sum(JUDGED_UNITS.values())}）—— 单独改小一条会在这里掉出来", okT,
                    "" if okT else f"合计只剩 {sum(JUDGED_UNITS.values())}"))

    for note, rid, n, want_hit in [
        ("达标 → 不响", "R-lantyr", JUDGED_UNITS.get("R-lantyr", 2), False),
        ("少一条 → 响（`cn_forbidden` 被清空就是这一步）", "R-lantyr",
         JUDGED_UNITS.get("R-lantyr", 2) - 1, True),
        ("一条都没判 → 响（执行体被换成 `return [], ''` 就是这一步）", "R-lantyr", 0, True),
        ("**没登记地板** → 响（新断言不登记 = 给空转留门）", "R-从来没有过的", 99, True),
    ]:
        got = _units_verdict(rid, n)
        ok = bool(got) == want_hit
        results.append((f"_units_verdict({rid}, {n})：{note}", ok, "" if ok else f"实得 {got!r}"))

    # —— 实跑证明：叶数与规矩数是**两个**数
    pairs = [("ember", "p.json", "entries.A.name", "Lantyr city", "兰提尔城")] * 3
    r_full = {"id": "T", "kind": "term_gated", "en": r"\bLantyr", "cn_required": "兰提尔",
              "cn_forbidden": ["兰蒂尔"], "min_hits": 1}
    r_wiped = dict(r_full, cn_forbidden=[])
    _UNITS.clear()
    _b1, d1 = KINDS["term_gated"](r_full, _FakeCtx(pairs=pairs))
    n1 = len(_UNITS)
    _UNITS.clear()
    _b2, d2 = KINDS["term_gated"](r_wiped, _FakeCtx(pairs=pairs))
    n2 = len(_UNITS)
    okL = (d1 == d2 and n1 == n2 + 1 and n2 >= 1)
    results.append((f"⚑ 清空 `cn_forbidden`：**detail 一模一样**（{d1!r}）而规矩数 "
                    f"{n1} → {n2} —— 这就是「扫了几片叶」为什么当不了反空转的数", okL,
                    "" if okL else f"detail {d1!r} vs {d2!r}；规矩数 {n1} vs {n2}"))

    fr = {"id": "T", "kind": "no_bilingual_tail", "fields": ["tokenName"]}
    fpairs = [("ember", "p.json", "actors.X.tokenName", "Scout", "斥候")]
    _UNITS.clear()
    _b3, d3 = KINDS["no_bilingual_tail"](fr, _FakeCtx(pairs=fpairs))
    n3 = len(_UNITS)
    _UNITS.clear()
    _b4, d4 = KINDS["no_bilingual_tail"](dict(fr, fields=[]), _FakeCtx(pairs=fpairs))
    n4 = len(_UNITS)
    okM = (n3 >= 1 and n4 == 0 and not _b4 and "检查 0 个" in d4)
    results.append((f"⚑ 清空 `fields`：执行体照样返回 0 违规（{d4!r}）—— "
                    f"主闸看不出任何异常，只有规矩数 {n3} → {n4} 会掉", okM,
                    "" if okM else f"规矩数 {n3} → {n4}，违规 {len(_b4)}，detail {d4!r}"))
    _UNITS.clear()

    nbad = 0
    for note, ok, why in results:
        if not ok:
            nbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        if not ok:
            print(f"        {why}")
    print(f"\n判了几条规矩：{len(results) - nbad} / {len(results)} 通过")
    return nbad, len(results)


SELFTEST_SHAPE = {
    "双语尾巴": 10, "读库闸": 7, "sense_gated": 6, "glossary_value": 6,
    "block_aligned_gate": 14, "block_sense_gate": 11, "enricher_slot_gate": 15,
    "enricher_text_coverage": 18, "twin_files": 5, "translate_cases": 23,
    "source_literal": 5, "panel_liveness": 13, "tracked_inputs": 12,
    "ruleset_shape": 45, "main 返回码": 3,
    # 第三十一轮新增的四组（L5-① 两道新护栏各一组、L5-② 11 个 kind 的执行体两组）
    "叶级执行体": 23, "读盘执行体": 15, "载荷地板": 29, "判了几条规矩": 9,
    # 第三十二轮 L6-①：`kind 覆盖表` 从字面量 1 改成**函数自报的比对项数**
    #（22 种 kind + 1 道待补上限 = 23），并新增「护栏反例」组（那两道判定函数自己的反例）
    # 第三十三轮 B：KINDS 从 22 种涨到 23 种 ⇒ 覆盖表的比对项数 23 → 24（23 种 + 1 道待补上限）
    "kind 覆盖表": 24, "护栏反例": 9,
    # 第三十三轮 B：新判据层 tag_parity 的正反例 10 条 + 3 条元用例
    #（掏空 tag_diff / 掏空 tag_multiset / 用真身死下限跑一遍）
    "tag_parity": 13,
}


def _selftest_shape_check(counts, shape=None):
    """返回 (违规行, 说明行, 这次比对了几组)。`counts` 是 {组名: 本次实跑条数}。

    ⚠ 第三十二轮 L6-①：多返回一个「比对了几组」——
      `run_selftest()` 从前是 `total += len(SELFTEST_SHAPE)`（**数的是表长**，
      跟这个函数跑没跑过一点关系），所以把本函数整体换成 `return [], []` 时
      主闸与自检**都不会响**。现在分母从这里取，掏空 ⇒ 分母掉到 0 ⇒ 当场判失败。
    ⚠ `shape` 参数只给反例用（默认取模块级的 SELFTEST_SHAPE）。
    """
    shape = SELFTEST_SHAPE if shape is None else shape
    bad, notes = [], []
    n_checked = 0
    for name, floor in shape.items():
        n_checked += 1
        got = counts.get(name)
        if got is None:
            bad.append(f"登记过的自检组 `{name}` 这次**没跑** —— 整组被摘掉了？")
        elif got < floor:
            bad.append(f"`{name}` 只剩 {got} 条用例（登记 {floor} 条）—— "
                       f"有用例被删了。要真删，把 SELFTEST_SHAPE 里这个数一起改，"
                       f"那是一处自证的 diff")
        elif got > floor:
            notes.append(f"`{name}` 涨到 {got} 条（登记 {floor}）—— 加用例不判失败，"
                         f"顺手把登记值改成 {got} 就行")
    for name in counts:
        if name not in shape:
            notes.append(f"新组 `{name}`（{counts[name]} 条）还没登记进 SELFTEST_SHAPE")
    return bad, notes, n_checked


def run_selftest():
    _KIND_CALLS.clear()               # kind 覆盖表数的是**本次自检**里的真实调用次数
    bad = 0
    for value, want in SELFTEST:
        got = has_bilingual_tail(value)
        flag = "ok  " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"  {flag} {value!r:32s} 期望 {want} 实得 {got}")
    print(f"\n双语尾巴判据：{len(SELFTEST) - bad} / {len(SELFTEST)} 通过")

    print("\n读库闸（_gate_one）正反例：")
    gbad = 0
    for note, term, ctx, scan, want_h, want_b in GATE_SELFTEST:
        h, b = _gate_one(term, ctx, scan, None)
        ok = (h == want_h and len(b) == want_b)
        if not ok:
            gbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望 命中{want_h}/违规{want_b}，实得 命中{h}/违规{len(b)}")
    print(f"\n读库闸：{len(GATE_SELFTEST) - gbad} / {len(GATE_SELFTEST)} 通过")

    print("\n机制义／普通名词义分类闸（sense_gated）正反例：")
    sbad = 0
    for note, pairs, want_b in SENSE_SELFTEST:
        b, detail = a_sense_gated(_SENSE_RULE, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            sbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nsense_gated：{len(SENSE_SELFTEST) - sbad} / {len(SENSE_SELFTEST)} 通过")

    print("\n词表值头部判据（glossary_value）正反例：")
    vbad = 0
    for note, want, got, expect in GLOSSARY_SELFTEST:
        ok = glossary_value_matches(want, got) == expect
        if not ok:
            vbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        want={want!r} got={got!r} 期望 {expect}")
    print(f"\nglossary_value：{len(GLOSSARY_SELFTEST) - vbad} / {len(GLOSSARY_SELFTEST)} 通过")

    print("\n块对齐闸（block_aligned_gate）正反例：")
    abad = 0
    for note, rule, pairs, want_b in BLOCK_ALIGN_SELFTEST:
        b, detail = a_block_aligned_gate(rule, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            abad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nblock_aligned_gate：{len(BLOCK_ALIGN_SELFTEST) - abad} / {len(BLOCK_ALIGN_SELFTEST)} 通过")

    print("\n块级义项闸（block_sense_gate）正反例：")
    bbad = 0
    for note, override, pairs, want_b in BLOCK_SENSE_SELFTEST:
        rule = dict(_SENSE_BLOCK_RULE, **(override or {}))
        b, detail = a_block_sense_gate(rule, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            bbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nblock_sense_gate：{len(BLOCK_SENSE_SELFTEST) - bbad} / {len(BLOCK_SENSE_SELFTEST)} 通过")

    tgbad, _tgn = run_tag_parity_selftest()

    print("\n增强器槽位闸（enricher_slot_gate）正反例：")
    ebad = 0
    for note, override, pairs, want_b in SLOT_SELFTEST:
        rule = dict(_SLOT_RULE, **(override or {}))
        b, detail = a_enricher_slot_gate(rule, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            ebad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nenricher_slot_gate：{len(SLOT_SELFTEST) - ebad} / {len(SLOT_SELFTEST)} 通过")

    # ⚠ 先证明「探针真的在验」：不做这一步，下面十几条正反例全可能是在一个
    #   **根本没取到 readaloud 值**的空判据上比 0==0。与 scan_content_coverage
    #   的 --selftest 头两条同一个用意。
    print("\n增强器可见正文覆盖闸（enricher_text_coverage）正反例：")
    tbad = 0
    _probe, _pd = a_enricher_text_coverage(_TEXT_RULE, _FakeCtx(pairs=_RA(_RA_EN, _RA_CN)))
    _live = ("判过 1 段" in _pd and "命中 1 段 / 1 条" in _pd)
    if not _live:
        tbad += 1
    print(f"  {'ok  ' if _live else 'FAIL'} 前置：readaloud 的值真的被取出来判了"
          f"（detail 必须报「判过 1 段」与锚点「命中 1 段 / 1 条」）")
    print(f"        {_pd}")
    for note, override, pairs, want_b in TEXT_COV_SELFTEST:
        rule = dict(_TEXT_RULE, **(override or {}))
        b, detail = a_enricher_text_coverage(rule, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            tbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nenricher_text_coverage：{len(TEXT_COV_SELFTEST) + 1 - tbad} / "
          f"{len(TEXT_COV_SELFTEST) + 1} 通过")

    wbad, _wn = run_twin_selftest()
    rbad, _rn = run_translate_selftest()
    lbad, _ln = run_srclit_selftest()
    pbad, _pn = run_panel_selftest()
    kbad, _kn = run_tracked_selftest()
    sqbad, _sqn = run_ruleset_shape_selftest()
    mvbad, _mvn = run_main_verdict_selftest()
    # —— 第三十一轮：11 个此前一次都没被调用过的执行体 + 两道新护栏自己的回测
    lkbad, _lkn = run_leaf_kind_selftest()
    dkbad, _dkn = run_disk_kind_selftest()
    pfbad, _pfn = run_payload_floor_selftest()
    jubad, _jun = run_units_selftest()
    # —— 第三十二轮 L6-①：上面两道护栏（kind 覆盖表 / 自检形状）**自己的反例**。
    #    ⚠ 必须在 kind 覆盖表那一步**之前**跑：它会照着本次真实的 _KIND_CALLS 造反例。
    gcbad, _gcn = run_guard_cases_selftest()
    total = (len(SELFTEST) + len(GATE_SELFTEST) + len(SENSE_SELFTEST) + len(GLOSSARY_SELFTEST)
             + len(BLOCK_ALIGN_SELFTEST) + len(BLOCK_SENSE_SELFTEST) + len(SLOT_SELFTEST)
             + len(TEXT_COV_SELFTEST) + 1 + _wn + _rn + _ln + _pn + _kn + _sqn + _mvn
             + _lkn + _dkn + _pfn + _jun + _gcn + _tgn)
    nbad = (bad + gbad + sbad + vbad + abad + bbad + ebad + tbad
            + wbad + rbad + lbad + pbad + kbad + sqbad + mvbad
            + lkbad + dkbad + pfbad + jubad + gcbad + tgbad)

    # —— kind 覆盖表：`KINDS` 里每个 kind 都得有一组、且那一组**真的调到了它**。
    #    ⚠ 必须放在**所有组跑完之后**（数的是本次自检的累计调用次数）。
    # ⚠ 第三十二轮 L6-①：分母 `cov_n` **从函数结果现取**，不再是写死的字面量 1 ——
    #   写死的那个 1 与「这个函数跑没跑过」毫无关系，所以把它掏空成 `return [], ""`
    #   时主闸与自检都不会响（复核实测 rc=0 · 285/285 · FAIL 0）。
    #   现在：掏空 ⇒ `cov_n` 掉到 0 ⇒ 下面那道「比对项数必须等于 len(KINDS)+1」当场红，
    #   同时 `counts["kind 覆盖表"]` 也掉下 SELFTEST_SHAPE 的登记值，两处一起响。
    cov_bad, cov_note, cov_n = _kind_coverage_check()
    print("\nkind 执行体覆盖表（KIND_SELFTEST_OWNER，用 _KIND_CALLS 实测，不是名册）：")
    print(f"  {'ok  ' if not cov_bad else 'FAIL'} {cov_note}")
    for n in cov_bad:
        print(f"  FAIL  {n}")
    print("        逐 kind 调用次数："
          + "、".join(f"{k}={_KIND_CALLS.get(k, 0)}" for k in sorted(KINDS)))
    cov_want = len(KINDS) + 1
    if cov_n != cov_want:
        cov_bad = list(cov_bad) + [
            f"覆盖表这次只比对了 {cov_n} 项（该是 {cov_want} = {len(KINDS)} 种 kind + "
            f"1 道待补上限）—— 判定函数被换成空壳了？分母不是从它算出来的话，"
            f"这一步掏空之后两道闸都不会响"]
        print(f"  FAIL  {cov_bad[-1]}")
    total += cov_n if cov_n else cov_want       # 掏空时分母不许跟着缩水，用登记值兜底
    nbad += len(cov_bad)

    # —— 自检**自己**的条数（见 SELFTEST_SHAPE 上面那段）：分母不许悄悄变小。
    counts = {
        "双语尾巴": len(SELFTEST), "读库闸": len(GATE_SELFTEST),
        "sense_gated": len(SENSE_SELFTEST), "glossary_value": len(GLOSSARY_SELFTEST),
        "block_aligned_gate": len(BLOCK_ALIGN_SELFTEST),
        "block_sense_gate": len(BLOCK_SENSE_SELFTEST),
        "tag_parity": _tgn,
        "enricher_slot_gate": len(SLOT_SELFTEST),
        "enricher_text_coverage": len(TEXT_COV_SELFTEST) + 1,
        "twin_files": _wn, "translate_cases": _rn, "source_literal": _ln,
        "panel_liveness": _pn, "tracked_inputs": _kn, "ruleset_shape": _sqn,
        "main 返回码": _mvn,
        "叶级执行体": _lkn, "读盘执行体": _dkn, "载荷地板": _pfn, "判了几条规矩": _jun,
        "护栏反例": _gcn,
        # ⚠ 第三十二轮 L6-①：这里从前是字面量 `1`。字面量与「覆盖表跑没跑过」无关，
        #   所以掏空 `_kind_coverage_check` 时这个数纹丝不动、shape 检查照样满意。
        #   现在取的是**函数自己报的比对项数**。
        "kind 覆盖表": cov_n,
    }
    shape_bad, shape_notes, shape_n = _selftest_shape_check(counts)
    print("\n自检自身的条数（SELFTEST_SHAPE，住在本文件、不在任何用例表里）：")
    for n in shape_notes:
        print(f"  note  {n}")
    # ⚠ 第三十二轮 L6-①：分母同样**从函数结果现取**。从前是 `total += len(SELFTEST_SHAPE)`
    #   （数的是表长），把本函数换成 `return [], []` 时主闸与自检都不会响。
    if shape_n != len(SELFTEST_SHAPE):
        shape_bad = list(shape_bad) + [
            f"自检形状这次只比对了 {shape_n} 组（登记 {len(SELFTEST_SHAPE)} 组）—— "
            f"判定函数被换成空壳了？分母得是它算出来的，不能是表长"]
    for n in shape_bad:
        print(f"  FAIL  {n}")
    if not shape_bad:
        print(f"  ok    {len(counts)} 组、合计 {sum(counts.values())} 条，一组不少")
    nbad += len(shape_bad)
    total += shape_n if shape_n else len(SELFTEST_SHAPE)
    nbad_disp = nbad
    global _LAST_SELFTEST_SUMMARY
    _LAST_SELFTEST_SUMMARY = f"判据自身回测合计：{total - nbad_disp} / {total} 通过"
    print(f"\n══════ {_LAST_SELFTEST_SUMMARY} ══════")
    return 1 if nbad else 0


# ============ V15：**只跑主闸不算跑**（第三十一轮 C）
#
# 复核实测：针对**判据文件**的 70 个单点掏空，**主闸接住 0 个**，全部只有 `--selftest` 接住。
# 把 `a_cn_absent` 整体换成 `return [], ""` —— 主闸 66/0/0、rc=0，一个字都不会说。
# ⇒ 「下一轮只跑了主闸」这件事本身，就等于第三十／三十一轮加的东西不存在。
# 靠在文档里写「两条缺一不可」是不够的：那又回到「靠人记得」，而本项目反复栽的就是这个。
#
# 修法（选的是**最省事且不添新洞**的一种）：主闸开跑前先比一次**指纹** ——
#   指纹 = 判据文件 + 发布中那份规则集的 sha256。
#   · 指纹与上次「`--selftest` 全绿」时记下的一致 ⇒ 什么都不做（日常跑主闸零额外开销）；
#   · 不一致（或没有记录）⇒ **自动就地跑一遍 `--selftest`**，红了就把主闸一起判失败。
# 于是「改了判据只跑主闸」这条路不再存在：改了判据 ⇒ 指纹变 ⇒ 自检必跑。
#
# ⚠ 为什么记在临时目录而不是仓里：`tracked_inputs` 会 sweep `3-常用脚本/qa/`，
#   往那儿写一个每次都变的戳会让「未入库」那一条固定报红 —— 把护栏变成背景音。
#   记录丢了的后果只是「下次多跑一遍自检」，方向是安全的那一侧。
# ⚠ 递归防护：`run_selftest()` 里会起子进程去打主闸（`run_main_verdict_selftest`）。
#   自检开跑时设一个环境变量，子进程继承到就不再触发自动自检 —— 否则会自检套自检。
# ============ L6-③（第三十二轮）：**指纹只盖 2 个文件，而 node 侧两个执行体在外面**
#
# 复核实测：`_judge_fingerprint()` 从前只哈希 `assert_resolutions.py` + 规则集 json。
# 而 `translate_cases_runner.mjs` 与 `selfcheck_panel_runner.mjs` **同样是判据** ——
# 复核写了个 **3 行的假 runner**（完全不 import `translateText`，照规则里的 `recorded`
# 把 counts 编齐）：那条断言**违规 0 处**、detail 与真身**逐字同形**；
# 而这处改动不碰 `.py` 也不碰 `.json` ⇒ **指纹不变 ⇒ 主闸永远不触发自动自检**。
# ⚠⚠ 这不是理论问题：本轮开工时 `translate_cases_runner.mjs` 正处于已改未提交状态，
#   而主闸一整天都在打「判据指纹与上次 --selftest 全绿时一致，本次不重跑自检」。
#
# 修①：清单**声明式地推**，不写死 —— 判据文件 + 规则集 + 规则里点名的 `runner`
#      + `_KIND_DEFAULT_RUNNER` 里每个 kind 的**默认执行体**。
#      于是下一轮新加一个 kind 配一个 runner，它**自动**进指纹，不必有人记得来改这里。
#
# ⚠ **两份自检面板 `.mjs` 为什么不进指纹**（这是被问到要说得出理由的一处）：
#   它们是**被判对象**，不是判据。① `R-selfcheck-d-liveness` 每跑一次主闸就**离线现跑一遍
#   面板真身**，面板改了当场就在主闸上体现，不需要靠指纹去触发自检；② `R-selfcheck-twin`
#   与 `R-selfcheck-d-section-name` 也都在主闸里逐次读它们。把被判对象塞进判据指纹，
#   只会让「译者动了一行面板文案」触发一次五分钟的自检，把护栏变成背景音 ——
#   而本项目登记过的教训正是「护栏一旦变成背景音就等于没有」。
#   同理 `ember-hardcoded-cn.mjs`（被判文件）也不进。
#
# 修②：戳记从「一行 hex」改成**一张表**（清单 + 逐文件 sha256 + 上次自检的合计行 + 时间）。
# ⚠⚠ **边界照实写，不许写成「堵死了」**：戳记住在 `%TEMP%`，**不可能防伪造** ——
#   任何人都能写一份看起来对的表（极端情况下直接调 `_stamp_selftest_green()`）。
#   它防的是「**忘了跑**」，不是「**故意骗**」。本轮把伪造从「改 1 个字节、零痕迹」
#   抬到「要伪造一张与四个文件逐一对得上的表」，并且**把戳记里记的合计行印在主闸输出上** ——
#   于是伪造的痕迹落在人看得见的地方（主闸会说「上次自检 322/322 于 <时间>」），
#   而不是像从前那样只剩 12 个 hex 字符。
_SELFTEST_ENV = "AR_IN_SELFTEST"
_SELFTEST_STAMP = os.path.join(tempfile.gettempdir(), "assert_resolutions.selftest.stamp")
_LAST_SELFTEST_SUMMARY = ""      # run_selftest() 跑完填上，戳记里存它（好让伪造有痕迹）


def _judge_side_files():
    """**判据侧**文件清单（绝对路径，去重后排序）。声明式推，不写死。

    = 本判据文件 + 发布中的规则集 + 规则里点名的 `runner`
      + `_KIND_DEFAULT_RUNNER` 里每个 kind 的默认执行体。
    ⚠ 被判对象（两份自检面板 `.mjs`、`ember-hardcoded-cn.mjs`）**不在这里**，理由见上面那段。
    """
    out = {os.path.abspath(__file__), os.path.abspath(DEFAULT_RULES)}
    for name in _KIND_DEFAULT_RUNNER.values():
        out.add(os.path.abspath(os.path.join(HERE, name)))
    try:
        rules = json.load(open(DEFAULT_RULES, encoding="utf-8"))
    except Exception:                                  # noqa: BLE001 —— 读不动就只用默认清单
        rules = {}
    for r in rules.get("assertions", []):
        rn = r.get("runner")
        if isinstance(rn, str) and rn:
            out.add(os.path.abspath(os.path.join(HERE, rn)))
    return sorted(out)


def _judge_fingerprint():
    """(联合指纹, {相对路径: 逐文件 sha256})。读不到就返回 (None, None) ⇒ 一律重跑自检。"""
    import hashlib
    per = {}
    h = hashlib.sha256()
    for p in _judge_side_files():
        try:
            with open(p, "rb") as fh:
                one = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            return None, None
        try:
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
        except ValueError:
            rel = p
        per[rel] = one
        h.update((rel + "\n" + one + "\n").encode("utf-8"))
    return h.hexdigest(), per


def _stamp_selftest_green(summary=""):
    """`--selftest` 全绿时把**整张表**记下来。写不动就算了（下次多跑一遍而已）。"""
    fp, per = _judge_fingerprint()
    if not fp:
        return
    try:
        with open(_SELFTEST_STAMP, "w", encoding="utf-8") as fh:
            json.dump({"fingerprint": fp, "files": per, "summary": summary,
                       "when": time.strftime("%Y-%m-%d %H:%M:%S")},
                      fh, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _read_stamp():
    """读戳记。读不到 / 不是本轮那张表的形状 ⇒ 返回 None（⇒ 重跑自检，方向安全）。"""
    try:
        with open(_SELFTEST_STAMP, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:                                  # noqa: BLE001
        return None
    if not isinstance(d, dict) or not isinstance(d.get("files"), dict):
        return None                                    # 旧格式那一行 hex 也走这里
    return d


def _require_selftest_ran():
    """指纹对不上就**就地跑一遍** `--selftest`。返回 (是否失败, 说明行列表)。"""
    if os.environ.get(_SELFTEST_ENV):
        return False, []                              # 自检自己起的子进程，别套娃
    fp, per = _judge_fingerprint()
    st = _read_stamp()
    old = (st or {}).get("fingerprint")
    if fp and st and old == fp and st.get("files") == per:
        return False, [f"判据指纹与上次 `--selftest` 全绿时一致（{fp[:12]}，"
                       f"{len(per)} 个判据侧文件逐一对上），本次不重跑自检",
                       f"  上次自检：{st.get('summary') or '(戳记里没记合计行)'}"
                       f"　于 {st.get('when') or '?'}"]
    # 差在哪一个文件上 —— 说出来，别只给两截 hex（从前那样连「是 runner 变了」都看不出来）
    changed = []
    if st and isinstance(st.get("files"), dict) and per:
        for rel in sorted(set(per) | set(st["files"])):
            a, b = st["files"].get(rel), per.get(rel)
            if a != b:
                changed.append(f"{rel}（{str(a)[:8]} → {str(b)[:8]}）")
    notes = [f"⚠ **判据侧文件**自上次 `--selftest` 全绿以来**变过**"
             f"（记录 {str(old)[:12]} → 现在 {str(fp)[:12]}）—— 就地补跑一遍自检。",
             f"  变的是：{'、'.join(changed) if changed else '（没有可比的旧记录）'}",
             "  理由：针对判据文件的 70 个单点掏空，**主闸一个都接不住**"
             "（`a_cn_absent` 换成 `return [], \"\"` → 主闸 66/0/0、rc=0）。"
             "只跑主闸 = 那些护栏等于不存在。",
             "  ⚠ 指纹**必须盖住 node 侧那两个执行体**：复核用 3 行的假 runner"
             "（不调 `translateText`，照 `recorded` 把 counts 编齐）跑出「违规 0 处」、"
             "detail 与真身逐字同形 —— 那处改动不碰 `.py` 也不碰 `.json`。"]
    env = dict(os.environ, **{_SELFTEST_ENV: "1"})
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--selftest"],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    tail = [x for x in (proc.stdout or "").splitlines() if "FAIL" in x or "回测合计" in x]
    if proc.returncode == 0:
        summary = tail[-1].strip() if tail else "rc=0"
        notes.append(f"  自检结果：{summary}")
        _stamp_selftest_green(summary)
        return False, notes
    notes.append("  ⚠⚠ **自检没过** —— 主闸再绿也不作数（判据自己坏了）：")
    notes.extend(f"    {x}" for x in tail[-8:])
    return True, notes


def main():
    # Windows 控制台默认 gbk，规则里的 ⚠ / ▸ 会直接把脚本炸成 UnicodeEncodeError，
    # 而那看起来像「断言崩了」。与本目录其它扫描器统一：输出一律走 utf-8。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                              # noqa: BLE001 —— 老 Python / 被重定向时无所谓
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--repo", action="append", help="限定仓库（默认两个都跑）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max-show", type=int, default=6)
    ap.add_argument("--selftest", action="store_true", help="只跑判据自身的正反例回测")
    ap.add_argument("--root", help="改用另一棵项目树（灵敏度回测用：往副本里注入违规，确认断言真的会响）")
    a = ap.parse_args()

    if a.selftest:
        os.environ[_SELFTEST_ENV] = "1"             # 子进程继承 → 不会自检套自检
        rc = run_selftest()
        if rc == 0:
            _stamp_selftest_green(_LAST_SELFTEST_SUMMARY)
        return rc

    rules = json.load(open(a.rules, encoding="utf-8"))
    global ROOT
    if a.root:
        ROOT = os.path.abspath(a.root)
    repos = {}
    for name, rel in REPOS.items():
        d = os.path.join(ROOT, rel)
        if a.repo and rel not in a.repo and name not in a.repo:
            continue
        if os.path.isdir(d):
            repos[name] = d
    ctx = Ctx(repos, rules.get("meta"))
    # 本次运行真正在跑的那份规则集 —— `ruleset_shape` 判的就是它本身。
    # ⚠ 故意**不让那条断言自己去磁盘上重读一份**：`--rules` 指向 A 而断言去读 B，
    #   两边都绿、判的却不是同一件事，那是形态 (g)「输入被静默换成另一份」。
    ctx.rules = rules
    ctx.rules_path = a.rules
    total_leaves = sum(len(v) for v in ctx.pairs.values())
    total_keys = sum(len(v) for v in ctx.lang.values())
    print(f"读入 {len(repos)} 个仓库 / {total_leaves} 对中英叶 / {total_keys} 对中英 lang 键\n")
    if not total_keys:
        print("⚠ lang 通道一个键都没读到 —— meta.lang_sources 指向的上游安装目录不在？\n"
              "  依赖 lang 闸的断言会以「空转」形态报失败，那是**判据环境问题**，不是库的问题。\n")

    failed = passed = skipped = 0

    # ⚠ **不是断言，是 main() 的固定动作（第三十一轮 C）：只跑主闸不算跑。**
    #   判据文件动过而自检没跑过 ⇒ 就地补跑；补跑红了，主闸一并判失败。
    _st_bad, _st_notes = _require_selftest_ran()
    for _n in _st_notes:
        print(_n)
    if _st_notes:
        print()
    if _st_bad:
        failed += 1
        print("  FAIL  <main 的固定检查>  —— `--selftest` 没过，判据自身不可信")

    # ⚠ **不是断言，是 main() 的固定动作。** 见 `_require_ruleset_shape` 的 docstring：
    #   数一数还剩几条规则的那条断言，本身也可能被删 —— 守卫不能与被守护的容器同生共死。
    _missing_shape = _require_ruleset_shape(rules)
    if _missing_shape:
        failed += 1
        print(f"  FAIL  <main 的固定检查>  —— 规则集缺少 `ruleset_shape` 断言")
        print(f"        {_missing_shape}")

    for rule in rules["assertions"]:
        rid = rule.get("id", "<无 id>")
        fn = KINDS.get(rule.get("kind"))
        if not fn:
            # ⚠ 第三十轮（H1/H3）：这里**曾经**只打一行 `??` 然后 `skipped += 1`，
            #   而 `main()` 的返回值是 `1 if failed else 0` —— `skipped` 从来不判失败。
            #   后果：把任意一条断言的 `kind` 拼错一个字母（`cn_absent` → `cn_absentt`），
            #   那条断言**彻底停跑**，主闸却是 `66 → 65/0/跳过 1`、rc=0，**一个字节都不用碰本文件**。
            #   现在：认不出 kind 一律**判失败**。不认识 ≠ 没问题。
            failed += 1
            print(f"  FAIL  {rid}  —— 未知断言类型 {rule.get('kind')!r}")
            print(f"        本条**没跑成**（判据里没有这个 kind 的执行体）。"
                  f"拼错一个字母就能让一条断言静默停跑，那是 §3.7 形态 (d) 的变体："
                  f"闸在跑，可这条的执行体是空的。")
            print(f"        已登记的 kind：{sorted(KINDS)}")
            continue
        _UNITS.clear()                                 # 「这次判了几条规矩」按条计
        try:
            bad, detail = fn(rule, ctx)
        except Exception as exc:                       # 断言自己炸了也要说清楚，不能静默
            print(f"  !! {rid}: 断言执行出错 {exc!r}")
            failed += 1
            continue
        n_units = len(_UNITS)
        units_why = _units_verdict(rid, n_units)
        if units_why:
            # ⚠ 不是加进 `bad` 交给下面那段打印 —— 那样它会被 `--max-show` 截掉。
            #   规矩数不达地板是**这条断言没在判**，与「判出了违规」是两件事，分开报。
            failed += 1
            print(f"  FAIL  {rid}  —— 判据空转（规矩数不达地板）")
            print(f"        {units_why}")
            print(f"        本次 detail：{detail}")
            continue
        detail = f"{detail}｜判了 {n_units} 条规矩（地板 {JUDGED_UNITS.get(rid)}）"
        if bad:
            failed += 1
            print(f"  FAIL  {rid}  —— {rule.get('title', '<无标题>')}")
            print(f"        裁决 {rule.get('decision', '?')}：{rule.get('why', '')}")
            print(f"        {detail}，违反 {len(bad)} 处：")
            for repo, pack, path, why in bad[:a.max_show]:
                print(f"          [{repo}/{pack}] {str(path)[:78]}")
                print(f"            {why}")
            if len(bad) > a.max_show:
                print(f"          …另 {len(bad) - a.max_show} 处")
        else:
            passed += 1
            if a.verbose:
                print(f"  ok    {rid}  {rule.get('title', '')}  （{detail}）")

    print(f"\n{'=' * 62}")
    print(f"通过 {passed} / 失败 {failed} / 跳过 {skipped}")
    if skipped:
        print("\n⚠ **跳过不是通过。** 见 `_exit_code` 的 docstring：本项目里"
              "「跳过」没有任何合法的产出口，出现即判失败。")
    if failed:
        print("\n⚠ 失败的每一条都对应第 8 节的一条既定裁决。")
        print("  正确的处理是：要么改回来，要么**显式推翻那条裁决并同时改断言**——")
        print("  不要只改断言让它变绿，那正是这套东西要防的事。")
    return _exit_code(failed, skipped)


if __name__ == "__main__":
    sys.exit(main())
