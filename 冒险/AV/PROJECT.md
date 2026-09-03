# 憎恨魔窟（Abomination Vaults）家族汉化

六个模组、十个 Babele 合集包的汉化工程。产物发布于
`takaqiao/pf2e-compendium-extra-cn`（本地克隆 `C:\Users\Taka\Desktop\fvttpublish\pf2e-compendium-extra`）。

## 硬约束

- **name 类叶子**（`name` / `tokenName` / `prototypeToken` / `folders.*` / 场景地图注记的键）：
  `中文 English`，**一个 ASCII 空格，不用括号**。
- **其余一切**（`description` / `text` / `publicNotes` / `privateNotes` / `blurb` / `caption` /
  outcome 的 `label`·`summary`，以及 `@UUID[...]{标签}` 里的标签文字）：**纯中文**。
- `@X[...]`、`[[...]]` 方括号内是机器件，**逐字节复制**；只有尾部 `{label}` 是散文。
- `@Localize[...]` 由 PF2e 系统经 pf2_cn 自行解析，**不可翻译**。
- 房间号（`C35`）、骰式（`1d6`）、加值（`+19`）、OGL/版权文本、署名与第三方工具名保持英文。
- 术语权威顺序：**最新 pf2wiki > pf2_cn / pf2e_compendium > pf2e_compendium_extra > 其他**，
  由 `工具/翻译流程/scripts/build_3source_tm.py` 的 `PRIORITY` 编码，有单元测试守着。
- 翻译由 Claude 在会话内完成；**任何脚本都不得调用模型 API**。

## 目录

| 路径 | 作用 |
|---|---|
| `_源/` | 冻结输入，只用 `Copy-Item`/`copy2` 拷入，**永不编辑** |
| `工作区/` | 构建产物，也是 QA 的 target_dir。`工作区/en/` 英文基线，`工作区/lang/` 模组自带 i18n 覆盖件 |
| `_cache/raw/` | LevelDB 原始文档转储（注入 `_key`） |
| `_backup/` | 每次写盘前的 `copy2` 快照，按步骤与时间戳分目录 |
| `qa/` | 全部脚本、判据文件（`_*.json`）、`baseline/`、`reports/` |

`_cache` / `_backup` / `_源` / `工作区` / `qa/reports` 不入库（派生物，可重建）。

## 管线（顺序有意义）

```
关闭 Foundry
 1  dump_pack_keys.mjs      LevelDB -> 原始文档 + 按 babele 身份规则的候选键清单
 2  build_babele_en.mjs     -> 工作区/en/  英文基线（--mapping-from 读各包自带 mapping）
 3  autofill_from_tm.py     TM 能机械回答的叶子：名称 + 有 compendiumSource 的 SRD 物品描述
 4  emit_units.py           把剩余叶子切成可审阅单元（先名后文，按文档边界）
 5  <翻译>                  每单元一个 agent，写 units_out/，自校验 check_unit.py
 6  apply_units.py          六项校验后落盘
 7  normalize_bilingual.py  去双语（标签序列切分为主）
 8  repair_html_prefix.py   补回被丢掉的开标签
 9  normalize_enricher_labels.py   {label} 中文化
10  scan_latin_nouns.py     正文里残留的英文词
11  normalize_terms.py      术语归一（_terms.json）
12  set_pack_labels.py      包 label（_pack_labels.json）
13  scan_pack_binding.py    绑定门（EXCLUSIONS.binding.json）
14  check_pack_targets.py   文件定位门
```

模组自带的 i18n（AV:E）走 `normalize_lang.py`，之后同样过 9/11。

## 判据文件

| 文件 | 内容 |
|---|---|
| `qa/_terms.json` | 术语归一规则（憎恨魔窟 / 奥塔里 / 天泽 …） |
| `qa/_labels.json` | 人工核定的 enricher 标签 |
| `qa/_pack_labels.json` | 各包 sidebar label |
| `qa/_lang_overrides.json` | AV:E i18n 的整叶覆盖与句级 fragment 补丁 |
| `qa/_html_patches.json` | 两处按路径精确修的 HTML 结构 |
| `qa/_markup_patches.json` | `<目标>` 这类会被浏览器吞掉的假标签 |
| `qa/_oversize_patches.json` | 超大叶子（Audio Credits）里只该译的那几句 |
| `qa/_dead_fields.json` | 目标路径已失效的 mapping 字段（senses） |
| `qa/EXCLUSIONS.binding.json` | 绑定门的归档豁免，每条附证据 |

## 踩过的坑（别再踩）

1. **只有实装的包是权威。** 磁盘上每一份「英文」文件不是被污染就是版本不符：
   `冒险/AV/pf2e-abomination-vaults.av.json` 是 Babele 生效时导出的（255/255 actor 带中文），
   `更新merge/` 与 `模组/system/pf2e_compendium/en-US/` 描述的是另一个 AV 版本
   （19 篇字母日志 vs 实装的 13 篇编号日志）。**先读 LevelDB 再下结论。**
2. **Foundry 把嵌入文档存成独立的 LevelDB 键**（`!journal.pages!<父>.<子>`），
   只有 Adventure 包才内联。当成顶层文档读会把 9 个场景读成 653 个，并丢掉所有页与注记。
3. **包自带 `mapping` 是与默认值合并，不是替换**（`DocumentMappings#mergedDefinition`
   用 `foundry.utils.mergeObject`），所以 actor 的 `items`/`tokenName` 与 PF2e 属性字段同时生效。
   而且各包可能有额外字段（凄凉灯塔多了 `speed`/`di`/`languages`/`dr`），必须从包里读。
4. **精确串替换去不掉双语**：追加的英文里 class 属性已漂移，命中率只有 14.5%。
   改用只看标签名的序列切分，98.5%；再加「自身标签序列重复两遍」的自证据策略，100%。
5. **判「已译」不能只看有没有中文**：只有 enricher 标签被译过的叶子仍是英文句子。
   但这条**只对散文成立**——双语 name 本来就以整句英文结尾。
6. **往正文里替换英文词要极保守**：包内文档名做词典会被包自身的误译污染
   （`Shield Block` -> 护盾术格挡），TM 里也混着 statblock 数值（`Hit Points` -> 4生命值）。
7. **PF2e 的 schema 会漂移**：感官已从 `system.traits.senses.value` 移到
   `system.perception.senses`，旧路径在 405 个 actor 上全为 null。
8. **发版是 CI 干的**，推 `X.Y.Z` tag 触发；`RELEASE_PROCESS.md` 旧版 §5–§7 的手工打包
   漏了 `inject-lang.js` 与 `lang/`，已改写。

## 下一轮升级

`qa/baseline/en-2026-09/` 是本轮的英文快照（附 `COVERAGE.txt` 记模组版本）。
上游升版后重跑 1–2，与该快照 diff，即可分出 stale / changed / gone / new，
不必再靠猜哪些中文落后于英文。
