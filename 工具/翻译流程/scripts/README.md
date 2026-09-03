# scripts/

通用 FVTT 模组汉化流程脚本。所有脚本都接受 CLI 参数，可独立调用。

## 执行顺序（典型 merge 流程）

```bash
# 1. 构建（或重建）3 源 TM
python build_3source_tm.py
# → 翻译流程/tm_cache/tm_3source.json

# 2. 上游推送 NEW 文件后，先 diff 看变化
python diff_structures.py NEW/ OLD/

# 3. Port OLD 翻译进 NEW（公共路径直接复用）
python port_old_to_new.py <OLD.json> <NEW.json> <merged.json>

# 4. (可选) 处理 entry 副本之间的 actor 重叠
python port_entry_to_dup.py <file.json> <main_entry_key> <dup_entry_key>

# 5. 应用 TM（多策略 lookup 自动命中 SRD）
python apply_tm.py <merged.json>

# 6. 扫剩余英文残留
python scan_residue.py <target_dir>
python scan_short_residue.py <target_dir>

# 7. 在会话里翻译剩余 → 写入文件

# 8. 规范化术语（修历史不一致）
python normalize_tradition_terms.py <target_dir>
python normalize_prose_traditions.py <target_dir>

# 9. QA 全套
python audit_translations.py <target_dir>
python qa_check.py <target_dir> [old_dir]
```

所有 scan / audit / qa 脚本现在统一签名：`<target_dir>` 必填位置参数；自动 `os.walk` 发现 `*.json`，跳过 `_backup` / `_tmp` / `_cache` / `_qa_reports` / `_pdf` / `_zh_synthetic` / `NEW` 子目录。无需维护 hardcoded `FILES` 列表。

## 各脚本

| 脚本 | 用途 | 输入 | 输出 |
|---|---|---|---|
| `build_3source_tm.py` | 从 Wiki / pf2e_compendium / pf2_cn 构建优先级合并 TM | `模组/pf2_cn/zh_Hans/`、`模组/pf2e_compendium_chn/compendium/pf2e.*.json`、`pf2wiki-scraper/out/glossary_wiki.json` | `工具/翻译流程/tm_cache/tm_3source.json` |
| `fotrp_update.py` | 以稳定 ID 迁移 FotRP Addon、按来源优先级校准并输出差异审计 | 旧译、旧/新上游 `_source`、Wiki 项目词表与更新配置 | 新 Babele JSON + JSON 报告 |
| `apply_tm.py` | 多策略 TM 应用（直接 / 剥后缀 / 剥括号 / 反序 / 符文剥离 / Lore合成 / spell-trad合成 / natural-attack） | TM + 单 JSON 文件 | 原文件原地覆盖 |
| `diff_structures.py` | 结构化 diff: added/removed/changed paths | NEW/ + OLD/ 两目录 | 控制台报告 |
| `port_old_to_new.py` | 把 OLD 中文 port 进 NEW 公共路径 | OLD.json + NEW.json | merged.json |
| `port_entry_to_dup.py` | entry 副本之间 port 重叠 actor 翻译（处理 hmLe 类） | 单文件 + 两个 entry 键 | 原文件原地覆盖 |
| `scan_residue.py` | 真英文残留扫描（剔除 enricher 内部，识别双语格式） | target dir | `_tmp_residue_report.txt` + `_tmp_residues.json` |
| `scan_short_residue.py` | 短英文（≤5 字符）残留扫描，补 scan_residue 漏过的 | target dir | `_tmp_tight_scan_report.txt` |
| `audit_translations.py` | HTML 平衡 + UUID/enricher 结构 + 双语格式审计 | target dir | `_tmp_audit_report.txt` |
| `normalize_tradition_terms.py` | 把 spell-tradition 名字段统一到 pf2_cn 标准（奥术/神术/异能/原能/内在/聚能/组曲） | target dir | 原地修改 |
| `normalize_prose_traditions.py` | 修复 prose 中残留的 tradition 旧译 | target dir | 原地修改 |
| `qa_check.py` | 综合 QA：JSON validity + UUID/enricher counts；可选 OLD diff | target dir [, old dir] | 控制台报告 |

## 优先级与裁决

来源优先级：**经核对的 PF2 中文 Wiki > pf2e_compendium / pf2_cn > pf2e-compendium-extra-cn > 其他来源**

冲突时：
- Wiki 项必须来自已核对页面或已审阅的离线词表；未经核验的 scraper 结果不得直接覆盖。
- `pf2e_compendium` 精确条目与 `pf2_cn` 同层，精确条目优先于 i18n 键推导。
- extra 旧译仅在稳定 ID 对应、或英文原文完全一致时复用。
- 以上均未命中时，才采用其他经人工审阅的来源。

`build_3source_tm.py` 输出的每条 TM 项会保留 `all_sources`，便于人工裁决：
```json
{
  "Halberd": {
    "name": "戟 Halberd",
    "source": "wiki",
    "all_sources": {
      "pf2_cn": "戟",
      "pf2e_compendium": "戟 Halberd",
      "wiki": "戟"
    }
  }
}
```

## 目录约定

- `翻译流程/tm_cache/` — 缓存的 TM JSON
- `翻译流程/scripts/` — 本目录
- `<工程>/_backup/<timestamp>/` — 修改前的备份

## pf2_cn / compendium 路径细节

| 内容 | 路径 |
|---|---|
| pf2_cn UI 翻译 | `模组/pf2_cn/zh_Hans/*.json` |
| pf2e_compendium 中文 | `模组/pf2e_compendium_chn/compendium/pf2e.*.json`（**仅 pf2e.* 前缀**） |
| Wiki 离线词表 | `pf2wiki-scraper/out/glossary_wiki.json` 或项目级已核对词表 |
| FotRP Wiki 已核对词表 | `工具/翻译流程/data/fotrp_wiki_terms.json` |
