# PF2 Plus 十一模组汉化

`clerics-remaster` · `witches-remaster` · `pf2e-summoners-plus` ·
`pf2e-team-plus-{alchemists,barbarians,feats,inventors,magic,oracles-remastered,tian-xia,wizards}`
共 **11 个模组 / 71 个声明包（70 个实装）**。产物并入
`takaqiao/pf2e-compendium-extra-cn`（本地克隆 `C:\Users\Taka\Desktop\fvttpublish\pf2e-compendium-extra`）。

做法依据根目录 `标准PF2汉化流程.md`；工具全部来自 `冒险/AV/qa/`（house 工具箱，通用）。

## 起点（2026-09-04 勘察结论）

上一轮（2026-04/05，`模组/team+2/CN/`）译过 56 个包；与发布仓库 `compendium/` **逐字节相同**，
所以发布仓库是唯一基线，`team+2/CN` 不必再读。缺口有四类：

| 类别 | 规模 |
|---|---|
| 整个模组没译过 | `pf2e-team-plus-alchemists` 8 包 |
| 模组更新后新增的包 | barbarians 2 个 sf2e 包、magic 5 个 sf2e 包 |
| 译过但不合现行标准 | 双语残留 232 条、「有中文但基本是英文」203 条、名称冲突 29 组 |
| 机器件漂移 | 方括号体与基线不一致 40 条 |

## 关键结构事实：sf2e 包是 pf2e 包的镜像

alchemists / barbarians / feats / magic 各有一套 `sf2e-*` 包。把两侧英文的**方括号体全部
掩码**后逐叶比对：**11 对包、2,500+ 叶，散文差异为 0**。差的只有机器件——

```
Compendium.pf2e.conditionitems      -> Compendium.sf2e.conditions
Compendium.pf2e-team-plus-magic.items -> Compendium.pf2e-team-plus-magic.sf2e-items
```

所以 sf2e 侧**不翻译，从 pf2e 侧转写**（`mirror_pack.py`）。
**不能用全局替换**：`Compendium.pf2e.actionspf2e` 在不同条目上分别映射到
`sf2e.actionssf2e` / `sf2e.actions` / `pf2e-anachronism.actions` 三个不同目标，
只有逐叶查表才有唯一解。

## 管线（本项目实际跑的顺序）

```
 0  Foundry 可以开着——这 11 个模组的包没有被任何世界独占
 1  dump_pack_keys.mjs            -> _cache/raw + qa/reports/pack-keys.json   (70/71 包)
 2  build_babele_en.mjs           -> 工作区/en/                                (11,444 叶)
 3  seed_from_existing.py         --source pub=<发布仓库>/compendium          (8,013 叶)
 4  autofill_from_tm.py           名称 + compendiumSource 的 SRD 描述
 5  autofill_srd_by_name.py       0 命中——这批模组全是自制内容，不是 SRD 拷贝
 6  repair_split_keys / strip_original_marker / normalize_bilingual / strip_english_suffix
 7  strip_label_translated_suffix.py   ★新增，见下                            (85 叶)
 8  repair_bracket_bodies.py --from-baseline
 9  repair_brackets_by_identity.py     ★新增，见下                            (8 叶)
10  clear_leaves.py               ★新增：把判死的叶子清空，交回单元流水线    (122 叶)
11  mirror_pack.py                ★新增：sf2e <- pf2e                        (2,552 叶)
12  emit_units.py                 -> qa/units/            61 单元 / 669K 字符
13  build_unit_glossary.py        ★新增 -> qa/units_gloss/  每单元一份术语表
14  <翻译>                        先名后文，两批 Workflow；自校验 check_unit.py
15  apply_units.py
16  mirror_pack.py --overwrite    再镜像一次（pf2e 侧此时才完整）
17  normalize_* / scan_* / repair_dead_links / set_pack_labels
18  gate.py                       唯一非零退出的入口
```

## 本轮新增的五个工具（都在 `冒险/AV/qa/`，通用）

| 脚本 | 解决什么 |
|---|---|
| `strip_label_translated_suffix.py` | 附在译文后面的英文块，其 `{标签}` 也被译过，所以「整叶以英文基线结尾」的判据打不中。掩码掉 `{...}` 再比 |
| `repair_brackets_by_identity.py` | 中文语序会移动 enricher，按位合并因此被拒；改比对**多重集合**，只在两侧各差一个同类 token 时替换 |
| `mirror_pack.py` | pf2e→sf2e 转写。查表按 token 身份而不是位置 |
| `clear_leaves.py` | 把扫描判死的叶子清空，`emit_units.py` 才会重新发出来 |
| `build_unit_glossary.py` | 每个单元配一份术语表：语料自身的双语 name 叶 > TM（wiki / pf2e_compendium） |

## 判据文件（`qa/`）

| 文件 | 内容 |
|---|---|
| `_pack_labels.json` | 71 个包的 sidebar label。同模组内 pf2e/sf2e 双胞胎上游同名，中文半边加（PF2e）/（SF2e）区分 |
| `EXCLUSIONS.bilingual.json` | 目前只有一条：`Feats+` 的 License 页，Paizo 商标名单必须留英文 |
| `_terms.json` | 术语归一规则 |
| `TRANSLATOR_BRIEF.md` | 发给每个单元 agent 的须知 |

## 踩过的坑

1. **`autofill_srd_by_name.py` 的 `--compendium-dir` 要指到 `pf2e_compendium_chn/compendium`**，
   指到模组根目录会静默 0 命中；`--packs` 收的是**裸包名**（`equipment-srd`），不是 `pf2e.equipment-srd`。
2. **按 `compendiumSource` 回填 actor 内嵌物品会取到另一版本的文本**：
   `inventors-iconic` 的 `Revolutionary Innovation`，上游 chn 那版有 15 个 `@UUID` 没有
   `@Check[type:flat|dc:13]`，模组自己那版正好相反——两者是**不同修订**。判据是
   「方括号多重集合与本包基线不一致」，19 条因此被清掉重译。
3. **`[[/r ... #flavor]]` 的 flavor 保持英文**：上游 `pf2e_compendium_chn` 里 930 条 ASCII
   对 10 条中文，压倒性惯例。上一轮译了 4 条（`#尺` `#额外效果` `#伤害类型` `#光之杀手对抗`）。
4. **`Sickened 1/2/3` 三档都译成「恶心」**，档位丢了；`Critical Success`/`Critical Failure`
   一起译成「大成功或大失败」。单元须知里专门点名了这两条。
5. **模组更新会改机器件语法**：`dc:@actor.flags.pf2e.aspectDC` →
   `dc:resolve(@actor.flags.pf2e.aspectDC)`，旧译文的方括号体因此过期。
6. **`pf2e-team-plus-alchemists.sf2e-journals` 声明了但没有 LevelDB**，dumper 报 MISS 是对的，
   不需要为它出文件。
7. **`pf2e-feats-plus.*.json` 是模组改名前的旧文件名**（现为 `pf2e-team-plus-feats`），
   还留在 `team+2/CN/`；发布仓库里没有，不要再捡回去。

## 模组自带 i18n 与 homebrew（Babele 够不着的两处）

- `pf2e-team-plus-oracles-remastered/lang/en.json` **103 个键**（选择提示、规则元素标签），
  走 `lang/external/pf2e-team-plus-oracles-remastered.json` + `inject-lang.js`。
- `pf2e-team-plus-magic/lang/en.json` 1 个键（`PF2E.RuleElement.AspectForm`）。
- `homebrew/`：11 个模组共 34 条 homebrew 特征/基础武器，已译 33 条；
  **`pf2e-team-plus-alchemists` 的 `featTraits.concoction` 是新的**，需要新建
  `homebrew/pf2e-team-plus-alchemists.homebrew.json`。

## 又一个坑：RollTable 的结果键与值形状

旧稿把 `witches-remaster.witches-remaster-roll-tables` 的结果按掷骰区间存成
`{"1": "重投一次…", "14-16": "远程攻击骰…"}`。`seed_from_existing.py` 报
`dead keys per source: {'pub': 9}`——9 个键一个都没落到基线上。查 babele 2.9.1 的源码，
两条独立的原因，各自都足以让这张表在运行时保持英文：

1. **`range` 提取器格式固定是 `${start}-${end}`**（`core/babele.js`
   `registerDefaultIdentityExtractors`，注意它不在 `identity-extractor-registry.js`
   的 `defaultExtractors()` 里，只看后者会误判成「根本没有 range 提取器」）。
   单点区间 `[1,1]` 导出的是 `"1-1"` 而不是 `"1"`，所以 `"1"` `"20"` 这类键匹配不到。
2. **值必须是字段对象，不能是裸字符串**。`FieldMapping.map()` 取的是
   `translations[this.field]`；对 `TableResult`，mapping 声明的字段是 `description`。
   传一个字符串进去，`translations["description"]` 是 `undefined`，**整条静默失效**。

本轮改成 `_id` 键 + `{"description": …}` 的形状。`match: ["_id", "range"]` 里 `_id` 排第一，
所以 id 键既合法又比区间键稳（区间会随表改动而变，id 不会）。

**这不止影响本项目**：发布仓库里另有 7 个文件、共 1,521 条同样是「区间键 + 裸字符串」的
roll table 结果（`xdy-pf2e-workbench` 1,202 条、`battlezoo-eldamon` 211 条、
`season-of-ghosts-tools` 25 条、`secrets-of-grayce` 20 条等），**同样从未生效**。
不在本轮范围内，但值得单独修一轮。

**通用判据**：覆盖率和绑定率都好看、运行时却是英文时，先确认两件事——
导出键的**格式**与提取器一致，以及值是**字段对象**而不是字符串。


## 本轮为工具箱修掉的六个缺陷（都在 `冒险/AV/qa/`，影响所有项目）

1. **`repair_split_keys.py` 只能接一层。** `Eto... Bleh!` 按点号切开是四层
   （`Eto` > `` > `` > ` Bleh!`），旧逻辑只会把 `A` + `B` 拼回 `A.B`，接不上带空段的链。
   已改成沿子树找到「拼起来等于基线里某个键」的整条链再搬。AV 语料复查为 0，无回归。
2. **`gate.py` 只 glob `lang/*.json`。** 本项目按发布仓库的形状把 i18n 放在
   `lang/external/<moduleId>.json`，深一层，于是那 104 个键**一项检查都没过**。改成 rglob。
3. **`gate.py` 写死了工具箱自己的 `reports/pack-ids-all.json`。** 别的项目跑出来，凡是
   那份转储之后才装的模组，链接全被算成「模组未安装」。已加 `--pack-ids`。
4. **手工裁定对 Compendium 链接不生效。** `_link_rulings.json` 的直接目标映射只在世界域
   分支里查，写了也白写——正是「判据文件写了却不生效」那一类。现在**任何链接形态**都先查裁定，
   并支持 `<文件名>::<目标>` 形式把裁定钉在单个文件上（pf2e/sf2e 双胞胎包要指向各自的兄弟包）。
5. **`repair_dead_links.py` 的正则只认 `@UUID`/`@Compendium`。** `@Item[...]`、`@Macro[...]`
   这些 v9 形态它根本看不见，`scan_all_links` 却在查——所以「扫出来 3 条死链，修完还是 3 条」。
6. **两个静默的误判**：
   - `scan_all_links.py` 把 `…<pack>.<父id>.JournalEntryPage.<页id>` 里的字面量
     `JournalEntryPage` 当成了 id——它正好 16 位字母数字，和 Foundry 的 id 长度一模一样，
     于是**每一条日志页链接都被报成死链**（本语料 30 条）。
   - `normalize_uuid_labels.py` 的 rank 正则只认结尾的数字，认不出中文的 `力竭1级`，
     于是把整串当术语、按目标的规范译名重写成 `力竭`——**把状态的档位吃掉了**。
     这与「`Sickened 1/2/3` 全译成恶心」是同一类损失，只是由工具造成。

## 新增工具（本轮）

| 脚本 | 解决什么 |
|---|---|
| `strip_label_translated_suffix.py` | 附加英文块的 `{标签}` 也被译过，「整叶以基线结尾」判据打不中；掩码 `{...}` 再比 |
| `repair_brackets_by_identity.py` | 中文语序会移动 enricher，按位合并被拒；改比多重集合，只差一个同类 token 时替换 |
| `mirror_pack.py` | pf2e→sf2e 转写，按 token 身份查表而不是按位置 |
| `clear_leaves.py` | 把扫描判死的叶子清空，`emit_units.py` 才会重新发出来 |
| `build_unit_glossary.py` | 每单元一份术语表：语料自身的双语 name 叶 > TM（wiki / pf2e_compendium） |
| `scan_name_variants.py` | §4② 那类看不见的冲突：剥掉 `Effect:`/分级后缀再按实体分组（本语料 105 组） |
| `strip_prose_parentheticals.py` | 散文里的 `特技（Acrobatics）` 双语夹注，以及 enricher 被删后留下的空 `（）` |
