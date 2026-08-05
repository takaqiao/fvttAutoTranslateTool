# 2026-08-06 临时脚本（阶段 0–2）

一次性探针，保留以便复现 PROJECT.md 里的每一个数字。不要在这些脚本上做增量开发——
稳定下来的工具应该提炼进 `3-常用脚本/`。

| 脚本 | 作用 | 对应结论 |
|---|---|---|
| `extract_any.mjs` | 旧抽取器的泛化补丁版（认 module.json） | 阶段 0 的首轮抽取，已被 `3-常用脚本/extract/extract_en.mjs` 取代 |
| `diff_translation.py` | 新英文/旧英文/译文 三方 diff | crucible NEW=132 / STALE=4 / DRIFT=140 |
| `workload.py` | 叶级待译量与覆盖率 | crucible 88% 覆盖；ember 各包覆盖率 |
| `probe_ember_adv.py` | ember 战役顶层键与各区段条目数 | 发现 1.0.0 的 `Ember Beta Two` → 0.6.0 `Ember Early Access` 改名 |
| `probe_lang_and_rename.py` | lang 三方 diff + 模拟改名后的覆盖率 | crucible lang 缺 293 key；ember lang 缺 47 |
| `probe_pages.py` | 枚举 ember 13 种 page 子类型及其字段 | PROJECT.md §3.3 的字段表 |
| `_measure_pages.cjs` | 量化 page.system.* 正文字符数 | 96 万字符「不可达正文」（后经 v1.0.15 修正） |
| `probe_v1015.py` | v1.0.15 实际 mapping / 键形状 / 覆盖率 | 推翻前三条基于陈旧副本的错误结论 |
| `_gap_ember.cjs` | 逐字段测 ember 真实缺口 | 16065 串 / 完成 39% / 待译明细 |
| `probe_source_ids.cjs` | 内嵌物品 `compendiumSource` 覆盖率 | **82.4% 可回源自动翻译**（最关键的发现） |
| `_tm_hit.cjs` | 值级 TM 复用命中率 | crucible→ember 内嵌物品命中 49% |
| `probe_glossaries.py` | 本地 6 份 glossary 谱系与冲突 | 选定 `glossary_crucible_merged.json` 为基底 |
| `crosscheck_vs_crucible_fr.py` | 与 Crucible-FR 独立实现交叉校验抽取器 | 查出多态 `system.description` 漏字段 bug |

运行前提：`node` 能从 `C:/Users/Taka/Desktop/fvtt` 解析到 `classic-level`。
