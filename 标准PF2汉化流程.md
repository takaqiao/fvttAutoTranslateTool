# 标准 PF2 汉化流程

把一个 Pathfinder 2e 的 Foundry VTT 模组/冒险汉化成中文的标准做法。
2026-09-04 定稿，依据是 AV 家族十个包的重制、以及随后用同一套工具跑通的
`pf2e-beginner-box` 与 `shopping-experience`（两个非 AV 模组，验证了流程的通用性）。

**读这份文档的两种人**：几个月后要开新冒险的你；被要求「汉化模组 X」的助手。
两者都不该重新踩一遍下面记的坑。

---

## 0. 三条不可协商的前提

| | 规则 | 为什么 |
|---|---|---|
| **不调 API** | 翻译由 Claude 在会话内完成。**任何脚本都不得调用模型 API**（含 `term_governor.py ai-suggest`）。 | Max 订阅覆盖会话，不覆盖 API 计费。 |
| **术语权威顺序** | 最新 pf2wiki > `pf2_cn` / `pf2e_compendium` > `pf2e_compendium_extra` > 其他 | 由 `工具/翻译流程/scripts/build_3source_tm.py` 的 `PRIORITY` 编码，有单测守着。 |
| **字段口径** | **name 类**（`name`/`tokenName`/`prototypeToken`/`folders.*`/场景注记键）＝`中文 English`，**一个 ASCII 空格，不用括号、不用换行**。**其余一切**＝纯中文。 | name 要能对照原文与搜索；正文双语会让阅读量翻倍。 |

补充口径，同样不商量：

- `@X[...]`、`[[...]]` 方括号内是**机器件**，逐字节复制；只有尾部 `{label}` 是散文，而散文＝纯中文。
- `@Localize[...]` 由 PF2e 系统自行解析，**连键都不能碰**。
- 房间号（`C35`）、骰式（`1d6`）、加值（`+19`）、DC、OGL/版权、署名与第三方工具名保持英文。
- 游戏缩写保持拉丁：XP DC AC HP GM NPC PC Paizo Pathfinder Foundry。
- **HTML 不要规范化**：`<hr>` 与 `<hr />` 的原始差异必须原样保留（下游有逐字节比对）。
- 备份一律 `shutil.copy2` / `Copy-Item`，**绝不 `json.load`+`dump`**——那会重排键序，让 diff 变成噪音。

---

## 1. 先搞清楚你在翻什么

四种形态，做法完全不同。**先判形态，再动手**。

| 形态 | 判据 | 能怎么翻 |
|---|---|---|
| **Adventure 包** | `module.json` 里 `"type": "Adventure"` | Babele。`entries` 顶层只有 1–3 个键，见 §7 的浅展开陷阱 |
| **普通文档包** | `"type": "Actor"` / `"JournalEntry"` / `"Item"` … | Babele，每个文档一个键，最省心 |
| **模组自带 i18n** | `module.json` 有 `languages: [...]` | Babele **够不着**。走 `lang/external/<moduleId>.json` + `inject-lang.js` |
| **系统 i18n** | `systems/pf2e/lang/*.json` | 由 `pf2_cn` 负责，别自己动 |

一个模组常常同时占两种。AV 就是「Adventure 包 + 自带 en.json」——导入对话框那五个选项一直是英文，
因为它们是 i18n 键，Babele 永远看不见。

### 取英文基线：只有实装的包是权威

**磁盘上任何一份「英文文件」都不可信**——不是被污染（Babele 生效时导出的，带中文），
就是版本不符。唯一权威是**实装模组的 LevelDB**。

```bash
# 关闭 Foundry（LevelDB 独占锁；绝不要删 LOCK 文件）
cd 冒险/AV/qa
node dump_pack_keys.mjs   --data-root <Data>/modules --modules a,b --raw-out <proj>/_cache/raw --keys-out <proj>/qa/reports/pack-keys.json
node build_babele_en.mjs  --data-root <Data>/modules --modules a,b --out-dir  <proj>/工作区/en
```

`--data-root` 指到 **`modules` 一级**，不是 `Data`。

---

## 2. 流程

顺序有意义。括号里是「放错位置会怎样」。

```
 0  关闭 Foundry
 1  dump_pack_keys.mjs        LevelDB -> 原始文档 + 按 Babele 身份规则的候选键清单
 2  build_babele_en.mjs       -> 工作区/en/  英文基线（--mapping-from 读各包自带 mapping）
 3  seed_from_existing.py     用旧稿 / 上游 chn 播种（键必须在基线里，否则制造孤儿）
 4  autofill_from_tm.py       TM 能机械回答的：名称 + 有 compendiumSource 的 SRD 描述
 5  autofill_srd_by_name.py   没带 compendiumSource 的 SRD 物品，按名字配 + 三道护栏
 6  emit_units.py             剩余叶子切成可审阅单元（先名后文，按文档边界）
 7  <翻译>                    每单元一个 agent，写 units_out/，自校验 check_unit.py
 8  apply_units.py            六项校验后落盘
 9  repair_split_keys.py      修被点号切开的键（`02. Foo` -> `02` > ` Foo`）
10  normalize_bilingual.py    去双语（标签序列切分）
11  strip_english_suffix.py   切分看不见的双语残留（判据：整叶以英文基线结尾）
11b strip_original_marker.py  剥掉旧工具追加的 `<hr><b>原文:</b>` 块（它只附了部分英文，
                              所以「以英文结尾」判据抓不到）
12  repair_html_prefix.py     补回被丢掉的开标签
13  repair_bracket_bodies.py  方括号内机器件按英文基线重建（--from-baseline）
14  normalize_enricher_labels.py  {label} 中文化
15  normalize_uuid_labels.py  同一 @UUID 目标只留一个中文标签
16  normalize_name_format.py  name 类叶子 中文\nEnglish -> 一个半角空格
17  scan_latin_nouns.py       正文里残留的英文词
18  scan_prose_terms.py       散文里的译名漂移（见 §4）
19  apply_path_patches.py     同形异义（只能靠路径区分的那几条）
20  normalize_terms.py        术语归一（_terms.json）
21  set_pack_labels.py        包 label
22  repair_dead_links.py      失效的 @UUID 目标
23  gate.py                   10 项检查，唯一会非零退出的入口
```

**为什么是这个顺序**

- 先播种再回填再人译：能机械回答的绝不占人工。AV 家族靠 3–5 把 3,185 叶压到 566 叶。
- **先名后文**：正文和 enricher 标签都要引用名字，名字定了正文才有依据。
- 去双语在 enricher 处理**之前**：追加的英文块自带一套 enricher，先剥掉才不会重复计数。
- `normalize_uuid_labels` 在 `normalize_terms` **之前**：术语归一会改标签里的中文，
  先统一标签再归一，两者才幂等。
- `gate.py` 永远最后，且**只有它非零退出**——其余脚本只报告。

---

## 3. 判据文件

人的裁决全部落在文件里，不留在脑子里。每条都要写**理由**。

| 文件 | 内容 |
|---|---|
| `_terms.json` | 术语归一规则。**有序**——具体规则必须排在它要切分的泛规则之前 |
| `_labels.json` | 人工核定的 enricher 标签 |
| `_uuid_labels.json` | 链接标签裁定（术语库对该目标给错答案时） |
| `_link_rulings.json` | 无法从包里恢复的死链目标 |
| `_path_patches.json` | 同形异义的逐路径改写；`_not_patched` 记「故意不改」的 |
| `_pack_labels.json` | 各包 sidebar label |
| `_dead_fields.json` | 目标路径已失效的 mapping 字段 |
| `EXCLUSIONS.binding.json` | 绑定门的归档豁免，每条附证据 |
| `EXCLUSIONS.bilingual.json` | 允许保留长段英文的叶子（署名 / OGL），逐条写理由 |

---

## 4. 五种「看不见的不一致」

覆盖率 100% 的语料里，这五类问题一条都不会报出来。

**① 同一英文名，两个中文译名** — `scan_name_consistency.py`
按双语 name 叶的英文半边分组。AV 家族一轮抓出 484 组、1,132 条叶子。

**② 剥掉房间号后仍冲突** — 英文键不同（`E06 - Bridge` vs `E10 - Bridge`），
分组看不见，但玩家看得见。AV 抓出 10 处。

**③ 散文里的译名漂移** — `scan_prose_terms.py`（这是最难的一类）

散文里没有键告诉你某个中文词对应哪个英文词。**英文基线逐叶对齐提供了这个键**：

> 只在「英文侧确实提到该实体」的中文叶子里，找与规范译名**近似但不相等**的窗口。

规范译名取自语料自己的双语 name 叶——**不要信 TM**，TM 里混着 statblock 垃圾
（`Hit Points → 4生命值`）和整句（`Will → 当你意志豁免成功时…`），拿它当基准只会刷屏。

要收敛得跑几轮：AV 家族是 106 → 67 → 57 → 51，共归一 132 处。
典型产出：`空无之死` vs `空寂之死`(25)、`幽影恶意` vs `暗影恶意`(13)、`加卢杜` vs `加鲁杜`(12)。

两个反例，说明为什么必须过滤：
- 窗口截断会制造假变体（`沙伊坦` 截出 `伊坦`）→ 要求变体与规范**互不包含**。
- `伊德里尼莉` 其实是更长的 `伊德里尼莉丝`，**它本身也是文档名**——旧的名称门看不见它，
  因为英文半边不同（`Cynemi` vs `Cynemi's ...`）。

**⑤ 译者把英文夹注写进了标题和标签** — `strip_baseline_gloss.py`

社区旧译稿常写成 `<h2>遭遇 Encounter</h2>`、`@UUID[…]{延后 Delay}`。按本流程，
只有 name 类叶子能带英文，标题和标签都是散文。

**判据不能是「这段英文在原文里出现过」**——新手包里这么判会命中 198 处，其中
`Pathfinder`/`Foundry`/`Paizo`/`NPC`/`PDF`/`DC 20`/`Ctrl` 全是中文句子正当使用的借词，
错杀近半。夹注与借词的区别在**位置**：夹注是基线里同一位置那条标题（或标签）的全文。

三条判据缺一不可：
- 先按位置对齐，再要求中文以对应英文**结尾**；
- 英文侧带数量词时全等匹配会失效（`6 狗头人战士 Kobold Warriors` vs `6 Kobold Warriors`），
  所以允许**按词边界**的后缀匹配；
- 但去掉那段尾巴后，**英文侧不能还剩下单词**。`Encounter Budget` 去掉 `Budget` 还剩
  `Encounter`，说明 `Budget` 是没译完的半个词，删了就丢词义——`遭遇 Budget` 要改成
  「遭遇预算」，不是删成「遭遇」。

对不齐的叶子（SRD 回填过、enricher 数量与基线不同）改用**链接目标**跨叶子配对；
再对不上的落 `_path_patches.json` 逐条处理。

**④ 术语库把页面标题的消歧后缀带进了正文** — `strip_tm_disambiguators.py`

pf2wiki 给同名页加分类后缀区分：`幽灵（特征）` 是特征页，区别于生物页。那个后缀属于
**标题**，不属于术语。但 `build_3source_tm.py` 抓的就是标题，于是每一次自动填充、
每一个查过词的译者，都把它原样带进了正文：

```
中  典型的幽灵（特征）只能离开它被杀害之处…      英  A typical ghost can stray only…
中  该生物的惊惧（状态）增加1                    英  the creature's frightened value increases by 1
中  （幽灵（特征））束缚之地                      英  (Ghost) Site Bound
```

判据很干脆：**英文侧有没有对应的 `(trait)` / `(condition)`**。AV 家族 722 处，英文侧
零处对应——纯属污染，可以见一个删一个。删完还要收拾它留下的空括号嵌套
（`（幽灵（特征））` → `（幽灵）`，不是 `（幽灵））`）。

这一类值得单独列出来的原因是：它不影响任何覆盖率指标、不影响链接、HTML 也平衡，
`gate.py` 十项全绿——只有读的人会觉得别扭。

---

## 5. 验收：`gate.py`

```bash
python 冒险/AV/qa/gate.py                              # AV 家族
python 冒险/AV/qa/gate.py --cn-dir <工作区>        --keys <proj>/qa/reports/pack-keys.json        --criteria-dir <proj>/qa                       # 换项目
```

十项，任一失败即非零退出：

| 检查 | 判据 |
|---|---|
| `targets` | 每个文件名解析到实装的 `<module>.<pack>`。**零豁免** |
| `binding` | 每个文档能绑上键、每个键能绑上文档。**孤儿零容忍**，未绑定 ≥98% |
| `html` | 无标签失衡 |
| `markup` | 方括号内无中文（`name:` 标签除外——那是给玩家看的） |
| `bilingual` | 散文无追加英文块 |
| `coverage` | **没有「含中文但基本是英文」的叶子** |
| `names` | 一个英文名一个中文译名 |
| `links` | **六种链接形态全查**：跨包 `Compendium.…`、旧式 `@Compendium[…]`、世界域、相对 `.id`、v9 的 `@Actor[id]` 与 `@Actor[名字]`。指向未安装模组的单列一档 |
| `terms` | `_terms.json` 已幂等 |
| `patches` | `_path_patches.json` 全部满足 |

`coverage` 这条是本轮新加的，因为「有中文即已译」是错的：
一条 3,000 拉丁字母里混 2 个汉字的叶子，任何存在性判据都会放行。它在**已经完工的 AV 语料**里
又抓出 2 条（结果是署名页，正当豁免），在新手包里抓出 4 条真漏译。

### 运行时验收（发版前必做）

建独立世界，确认**世界语言是 `cn`**（`DirectoryTranslationSource.supports()` 是严格相等，
值不对两个中文源会一起静默消失）。用浏览器控制台探针逐包断言：包 label、各类文档计数、
`0 英文残留`、`@Localize` 键无中文、随机点开几个链接确认跳对页。

`full` 与 `ondemand` 两种 `loadingMode` **各跑一遍**。

**判据文件按项目走**：`--criteria-dir` 指向本项目的 `qa/`，里面放自己的 `_terms.json`、
`EXCLUSIONS.*.json`、`_path_patches.json`。豁免路径**从数据里生成**，不要手写——
路径里有日志名和页名，猜错了豁免就是静默失效。

---

## 6. 发布

发布是 **CI 干的**：推 `X.Y.Z` tag 触发 `.github/workflows/release.yml`。

```bash
git switch -c release/X.Y.Z
# 拷包文件；改名的包要 git rm 旧文件，否则索引留死键
python scripts/regen-labels-titles.py    # 跑两遍，git diff --exit-code 必须干净
# module.json 改三处：version / download 的 tag 段 / changelog 的 tag 段
git commit   # commit 信息就是 release notes（CI 读 git log -1）
git push origin main && git tag X.Y.Z && git push origin X.Y.Z
```

**zip 白名单**是 `module.json babele.js inject-lang.js compendium homebrew lang scripts .gitignore`。
不在白名单里的目录**静默不进包**——`inject-lang.js` 漏了会 404 让整个模组起不来。

发布后用 `gh release download` 取**真 zip** 解压安装，重跑运行时探针。CI 绿 ≠ 包对。

---

## 7. Babele 机制速查（这些事实决定了上面所有规则）

路径相对 `Data/modules/babele/script/`。

| 事实 | 证据 |
|---|---|
| 翻译文件按**文件名**匹配包：`baseName === encodeURI("<module>.<pack>.json")` | `translation/translation-source-discovery.js:78` |
| 文档匹配顺序 `["_id","name","sourceId"]`，导出键顺序 `["name","_id","id"]` | `identity/document-identity.js:4-5` |
| **两个源合并 `entries` 是浅展开**：`{...旧, ...新}`，后加载的按顶层键整键替换 | `translation/compendium-translation.js:50-53` |
| 包自带 `mapping` 与默认值**合并**，不是替换（`foundry.utils.mergeObject`） | `mapping/document-mappings.js:347-350` |
| `TableResult` 的身份是 `range` 不是 `name` | `mapping/default-mappings.js:189-193` |
| 源排序：**未排序的在前，已排序的按 index 升序 → 数组里靠后者胜**；模组源默认 weight 20 | `translation/translation-source-discovery.js:88-112` + `translation-source-registry.js:59` |

**浅展开是最危险的一条。** Adventure 包的 `entries` 只有一两个键，所以两个汉化模组同时装时，
**后加载的整包替换先加载的，没有部分合并**。如果对方那份覆盖率只有 32%，顺序一翻就是灾难。
可发布的保险形态是在 `babele.js` 里加一段幂等的、仅 GM 的 `Hooks.once('ready')` 断言，
用 `setSourcePriority` 把自己排到末尾。

反过来这条也能用：extra 只提供 `Menace Under Otari` 和 `Pirate King's Plunder` 两个键时，
chn 独有的 `Beginner's Box Credits` 会原样保留——各取所长。

---

## 8. 症状索引

**看到什么 → 是什么 → 怎么修。** 这一节是本文档最值钱的部分。

### 汉化没生效

| 症状 | 原因 | 修法 |
|---|---|---|
| 侧栏包名英文，但点开里面是中文 | 只翻了 `entries`，没设包 `label` | `set_pack_labels.py` |
| 整个包一点没生效 | **文件名不等于 `<module>.<pack>.json`**（模组改过名） | `check_pack_targets.py`；改名要 `git rm` 旧文件 |
| 某个包时好时坏、像随机掉汉化 | 两个汉化模组在同一包上抢，加载顺序不定 | `sourceDiagnostics().translation.overlaps` 看顺序；`babele.js` 里加断言 |
| 轻量模式（ondemand）掉汉化 | 走的是另一条合并路径 | 两种模式各测一遍 |
| 改完存盘刷新还是旧译文 | HTTP 缓存 | 禁用缓存硬刷新；判据用只存在于新文件的字符串 |
| 第三方模组的对话框/提示全英文 | 那是 i18n 键，**Babele 够不着** | `lang/external/<moduleId>.json` + `inject-lang.js` |
| 模组更新后你的中文被覆盖回去 | 你改了实装文件 | 永不编辑实装文件，只走 `lang/external` |
| homebrew 特性名改了没反应也不报错 | 那些写在 `flags.<mod>.pf2e-homebrew`，Babele 看不见 | `scripts/inject-homebrew.js` 在 `setup` 钩子改 `CONFIG.PF2E` |

### 扫描说干净，实际不干净

| 症状 | 原因 | 修法 |
|---|---|---|
| 扫描 0 残留，但界面上还有英文短名 | 残留判据是「英文词数 > 5」 | `scan_short_residue.py`，阈值降到 2 |
| 报 0 残留，某个目录名却长期英文 | 那条键根本没绑上（孤儿） | `scan_pack_binding.py` |
| **一条叶子有中文所以判为已译，其实 3,000 字母里只有 2 个汉字** | 存在性判据 | `gate.py` 的 `coverage`，按**比例** |
| 覆盖率满分但富文本失效 | 方括号内被译成中文 | `repair_bracket_bodies.py --from-baseline` |
| 残留扫描把已译双语条目当漏译 | 判据没认双语约定 | 剥后短于 60 且含汉字即放行 |
| `@Localize[...]` 被当残留报出来、甚至被翻译 | 它是运行时键 | 加白名单；键被翻译过要还原 |
| 译文后面挂着 `<hr><b>原文:</b>` 加一段英文 | 2026-01 那套 Gemini 脚本的调试残留，随译文发了出去 | `strip_original_marker.py`（新手包 133 条） |
| `<figcaption></figcaption>` 空标签导致后面闭合失配 | 同上，旧工具插入的 | 按模式全局剥；`<img ... title />` 同源 |
| 豁免文件写了却不生效 | 路径手写猜错（日志名/页名对不上） | 从数据里生成豁免路径 |
| **整个模组的正文一次都没被检查过，而所有检查都是绿的** | 正文在模组自带的 i18n 文件里，不在 Babele 包里；检查器只 glob 包目录 | `gate.py --also`（`<cn-dir>/lang/*.json` 自动纳入）。AV:E 是 292 条叶、647 条链接 |
| 检查器扫了 0 个文件，输出和「干净」一模一样 | 没有任何一项报告扫了多少 | 让 gate 打印 `scanned N files, M leaves`——空集合通过和真通过必须长得不一样 |
| gate 报出一堆早就修好的问题 | `rglob` 递归进了 `_backup/`，扫的是修复**之前**的副本 | 扫描路径一律排除 `_backup` 段；备份目录放在被扫树里就必须显式排除 |
| 「补丁已满足」但缺陷还在 | in-place 补丁拿「替换文已出现」当判据，而同一片叶子别处正好也有那个词 | 判据应是「待替换文已不存在」 |
| `--criteria-dir` 传了相对路径，三项检查莫名变红 | 子脚本的 CWD 被强制设成 `qa/reports`，相对路径解析到别处，豁免文件根本没加载 | 在 `gate.py` 里 `.resolve()` 一次；目录不存在直接报错，别静默降级 |

### 链接与结构

| 症状 | 原因 | 修法 |
|---|---|---|
| `@UUID` 点开跳错页 / 打不开 | 目标 id 已失效（上游重建时 id 变了） | `repair_dead_links.py`：①叶 id 还在就取它**当前**的父 ②按英文标签唯一解析 ③人工裁定 |
| 一个包一半文档「未绑定」，孤儿却是 0 | **Actor 包的 `items` 是 id 字符串数组**，重组时没替换占位，每个物品被数两遍 | `pack_loader.mjs` 已修：附加子文档时先删同 id 的占位字符串 |
| 某些注记/页永远英文 | 键被按点号切开（`02. Foo` → `02` > ` Foo`） | `repair_split_keys.py`；文档名含点是常态 |
| 段落排版乱了 | 开标签被丢掉 | `repair_html_prefix.py` |
| `<目标>` 之类假标签被浏览器吞掉 | 中文里写了尖括号 | `_markup_patches.json` |
| **`@Actor[Chafkhem]{查夫肯姆}` 在汉化世界里必断，英文世界里却好的** | v9 旧写法**按名字**查世界文档（`foundry.mjs:35782` 的 `collection.getName`），而我们把角色改名成了 `查夫肯姆 Chafkhem` | `repair_named_links.py`：改写成 `@UUID[Actor.<id>]`。冒险导入保留 `_id`，所以语义不变而不再怕改名 |
| 链接检查报「0 条断链」，但方括号里根本没有 id 的那些从没被看过 | `classify()` 对无 id 的写法返回空，调用方 `if not ids: continue` **静默跳过**——71 条被算作「已检查」 | 无 id 不是「没问题」，要当缺陷报出来。`scan_all_links.py` 现在分六种形态计数 |
| `@JournalEntry[E: Arena]` 指向的日志压根不存在 | 上游把分层日志合并成了编号章节 | 别硬编码对照表：`E` 归属哪一章，由**页名房间号**投票决定（`E##` 页最多且几乎只有 `E##` 的那一章）。混编页的日志要排除，否则一章会认领所有字母 |
| 带 `#锚点` 的链接全被判死 | 锚点被当成 id 的一部分去比对（`OmLsmbwPtNMw7csF#Aesephna-menhemes`） | 比对前先 `partition("#")`，重建时再接回去。**这条我自己踩过：十条活链接被误报成死链** |
| 页 id 一个都没变，链接却死了 | 只有父日志 id 过期 | 优先级最高的策略：页 id 还在就查它**当前**的父 |
| 同一个页 id 在多个日志下重复出现 | Adventure 包里子文档 id 只在父内唯一 | 用锚点对应的**标题**在英文基线里定位到底是哪一页 |
| 「这个包没安装」，但它明明装着 | id 索引只 dump 了 `pf2e` 域，家族自己的包不在里面 | `pack_index.py` 把 `pack-keys.json` 合进来；两个检查器共用一份「什么叫已安装」 |
| 目标 id 查不到，名字也查不到，但文档还在 | 换包没换 id（`Hellknight Armiger` 从 npc-gallery 挪到 lost-omens-bestiary） | 全局按 id 找一遍，只改包段 |
| 名字对得上却报找不到 | 上游改了大小写（`Mage For Hire` → `Mage for Hire`） | 兜底做一次 casefold 匹配 |
| **一整段遭遇链接全断，英文侧却是好的** | 上游给怪物类 actor 换了新 id 并改用 `@UUID[Actor.<新>]`，而译文沿用旧译稿的 `@Actor[<旧>]` | 英文基线**同一位置**就是答案：按文档种类序列对齐后逐位还原（`repair_named_links.py`）。种类序列必须相等才允许对齐，否则会把 Item 写到 Actor 的 id 上 |
| 某几条对不齐，剩下的能修 | 那几片叶子的 enricher 数量与基线不同（SRD 回填过） | 「同一个退休 id 全库只指一个文档」——从对齐成功的位置学映射，出现矛盾就整条撤销 |
| 上游自己那条也指着退休 id | 是的，会发生 | 所以「按位置还原」必须先检查基线那一头是活的，否则等于抄一条死链 |
| `Effect: Major Fanged Rune Animal Form` 修不掉，同批的 `(Moderate)` 却能修 | Remaster 合并分级效果时，等级词有的在尾括号、有的**在中间** | 候选变体要同时去尾括号和去中缀等级词 |
| 「这个包没安装」，但装着的是别的项目的包 | pack-ids 只 dump 了 `pf2e` 域 | `pack_index.py` 把 `pack-keys.json` 合进来 |

### 术语

| 症状 | 原因 | 修法 |
|---|---|---|
| 同一英文名两个中文译名 | 单元独立翻译，互不知情 | `scan_name_consistency.py --fix` |
| 同一实体在正文里几种写法 | 见 §4 ③ | `scan_prose_terms.py`，跑到收敛 |
| 同名条目规则描述张冠李戴 | 扁平索引 first-file-wins（`Slither` 专长 vs 法术） | 按 pack 分索引；`compendiumSource` 定位 |
| Remaster 改名后整批译文变「待译」 | 英文键变了 | 术语裁决锚在**英文基线的标签**上，不要锚在语料多数票 |
| 术语库给出语料里没人用过的译名 | `tm-new`，不可全信 | 人审。AV 一轮 19 条里有 2 条是错的（`Invisibility` 的 wiki 页是**符文**不是法术） |
| 按名字取 SRD 描述，取到过期文本 | 上游改过内容 | `autofill_srd_by_name.py` 的三道护栏：唯一候选 + enricher 骨架一致 + 长度带 |
| 把英文词往正文里替换，越修越错 | 包内文档名当词典会被自身误译污染 | 只信 `{wiki, pf2e_compendium}`，禁数字/标点，≤12 字 |

### 工具与环境

| 症状 | 原因 | 修法 |
|---|---|---|
| PowerShell 里中文输出全是 `???` | 编码 | `$env:PYTHONIOENCODING="utf-8"` |
| `release.ps1` 跑一半中断 | PS 5.1 把 git/gh 的正常 stderr 当 NativeCommandError | **绝不用 `*>&1` / `2>&1` 包裹** |
| 无 BOM 的 ps1 里中文串解析失败 | PS 5.1 按 GB2312 读 | 那类脚本**保持纯 ASCII**，路径从 `$PSScriptRoot` 推导 |
| dumper 说某个模组没有 LevelDB | `module.json` 还声明着 v11 前的 `packs/x.db`，实际目录是 `packs/x` | 两个 dumper 已加回退 |
| 备份 diff 满屏噪音 | 用 `json.load`+`dump` 做的备份 | `shutil.copy2` |
| QA 临时文件被提交进模组 | reporter 把 `_tmp_*` 写到 CWD | reporter 的 CWD 固定在 `qa/reports` |
| 离线 wiki「增量重抓」等于空转 | 续跑判据只看 pageid 在不在 done 集合，改过的页永不重抓 | `invalidate_stale_v2.py`：按 `touched` 与各页自带的 `captured_at` 逐页比对 |

---

## 9. 工具索引

house 工具箱在 `冒险/AV/qa/`（名字是历史原因，**它是通用的**——
本轮用同一套跑通了 `pf2e-beginner-box` 与 `shopping-experience`）。

**取基线**：`dump_pack_keys.mjs` · `build_babele_en.mjs` · `pack_loader.mjs`
**播种回填**：`seed_from_existing.py` · `autofill_from_tm.py` · `autofill_srd_by_name.py`
**翻译**：`emit_units.py` · `check_unit.py` · `apply_units.py`
**修复**：`normalize_bilingual.py` · `strip_english_suffix.py` · `repair_html_prefix.py` ·
`repair_bracket_bodies.py` · `repair_split_keys.py` · `repair_dead_links.py` ·
`repair_named_links.py`（v9 `@Type[名字]` 写法）· `normalize_name_format.py` ·
`apply_path_patches.py`
**术语**：`normalize_terms.py` · `normalize_enricher_labels.py` · `normalize_uuid_labels.py` ·
`scan_name_consistency.py` · `scan_prose_terms.py` · `scan_latin_nouns.py` ·
`strip_tm_disambiguators.py`（术语库带出的 `（特征）`/`（状态）` 后缀）·
`strip_baseline_gloss.py`（标题与标签里的英文夹注）
**门禁**：`gate.py` · `scan_pack_binding.py` · `check_pack_targets.py` ·
`scan_all_links.py`（六种链接形态）· `pack_index.py`（两个检查器共用的 id 索引）

跨项目的**共享判据**在 `工具/翻译流程/data/`：`remaster_moves.json` 记录 PF2e
系统自己把文档搬到了哪、改成了什么名（分级效果合并、bestiary→monster-core、
Produce Flame→Ignition）。这是关于系统的事实、不是关于某个译本的决定，所以不放在任何
项目的判据目录里，用 `repair_dead_links.py --moves` 传入；项目自己的 `_link_rulings.json`
可以覆盖它。

跨项目的通用件在 `工具/翻译流程/scripts/`：
`build_3source_tm.py`（翻译记忆，`PRIORITY`/`TIE_BREAK` 冻结，有单测）·
`extract_wiki_terms.py`（离线 wiki 三层抽取）· `apply_tm.py` · `qa_check.py` ·
`audit_translations.py` · `scan_residue.py` · `scan_short_residue.py`

离线 wiki 在 `其他项目/PF2离线百科/`，术语层靠
`pf2wiki-scraper/{cookie_warmup_v2,dump_metadata_v2,invalidate_stale_v2,dump_parsed_v2_concurrent}.py`
刷新后再跑 `extract_wiki_terms.py`。

---

## 10. 一个新模组的最短路径

```bash
# 0 关 Foundry
cd 冒险/AV/qa
NEW=../../新模组
node dump_pack_keys.mjs  --data-root <Data>/modules --modules <mod> --raw-out $NEW/_cache/raw --keys-out $NEW/qa/reports/pack-keys.json
node build_babele_en.mjs --data-root <Data>/modules --modules <mod> --out-dir  $NEW/工作区/en

cd $NEW
python ../AV/qa/seed_from_existing.py   --en-dir 工作区/en --out-dir 工作区 --source "旧稿=<path>" --source "chn=<chn compendium>" --write
python ../AV/qa/autofill_from_tm.py     --en-dir 工作区/en --cn-dir 工作区 --keys qa/reports/pack-keys.json --raw-root _cache/raw --tm <tm> --compendium-dir <chn> --write
python ../AV/qa/autofill_srd_by_name.py --en-dir 工作区/en --cn-dir 工作区 --compendium-dir <chn> --write
python ../AV/qa/emit_units.py           --en-dir 工作区/en --cn-dir 工作区 --out-dir qa/units --max-leaves 60 --max-chars 14000
#   -> 每个单元一个 agent 翻译，写 qa/units_out/，自校验 check_unit.py
python ../AV/qa/apply_units.py          --units-dir qa/units --results-dir qa/units_out --cn-dir 工作区 --en-dir 工作区/en --write
#   -> 依次跑 §2 的 9–22 步
python ../AV/qa/gate.py --cn-dir 工作区
```

模组自带 i18n 文件的（AV:E 那种），正文放 `工作区/lang/<模组id>.json`，英文基线放
`工作区/en/<模组id>.json`，`gate.py` 会自动纳入，`repair_*.py` 用 `--also` 指过去。
发布时它去 `lang/external/`，并在 `inject-lang.js` 的 `EXTERNAL_LANG_SOURCES` 里登记。

实测这条路径把 `pf2e-beginner-box` + `shopping-experience` 的 3,185 叶（104 万字符）
压到 566 叶（20.8 万字符）需要人译，其余全部由播种与回填解决。

---

## 附：这套流程自己的历史

- `冒险/AV/PROJECT.md` —— AV 家族的工程账本，18 步管线与十五条坑的原始记录
- `冒险/EmberCrucible/Ember-Crucible Translation Project/PROJECT.md` —— 去双语两次事故的出处
- `冒险/AlienRPG/Alien-RPG Translation Project/PROJECT.md` —— 非 PF2 系统，证明方法与系统无关
- `冒险/AV/AV-E 翻译/术语治理流程.md` —— extract → 人审 → apply → restore 循环
- `工具/翻译流程/README.md` —— 跨项目管线与残留分类速查表
