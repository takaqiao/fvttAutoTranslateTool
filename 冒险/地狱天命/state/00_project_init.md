# 00 — 地狱天命 翻译项目初始化记录

_首次会话：2026-05-26_ • _初始化执行者：Claude (Opus 4.7 [1M])_

## 项目身份

- **英文标题**：Hell's Destiny
- **中文标题（wiki 官方）**：《地狱天命》
- **产品代码**：PZO15223 (HC) / PZO15223-HC
- **ISBN**：978-1-64078-799-5
- **PF2e 第 47 部冒险之路**（**重制版** / Remastered 规则集）
- **等级**：10-20 级
- **发售**：2026 年 7 月 1 日
- **页数**：258 页（PDF 物理页数；wiki 列示 256）
- **作者**：Rigby Bendele、Cole Kronewitter、Michael Bramnik、Jason Keeley
- **前传**：《地狱破灭》Hellbreakers (PZO15222E, 1-9 级)
- **续作**：《亵渎堡垒》

## 章节索引（来自 PDF ToC，PDF 物理页）

| # | 章节英文 | 章节中文（计划译名，可调整） | 起始页 | 作者 |
|---|---|---|---|---|
| – | Campaign Overview | 战役概览 | 4 | John Compton |
| 1 | Corentyn | 寇兰廷 | 10 | Rigby Bendele |
| 2 | Operation Broken Key | 破钥行动 | 30 | Rigby Bendele |
| 3 | Hellcoast Heresies | 地狱海岸异端 | 50 | Cole Kronewitter |
| 4 | The Eye of Khari | 卡里之眼 | 70 | Cole Kronewitter |
| 5 | Gathering Allies | 招纳盟友 | 88 | Michael Bramnik |
| 6 | Tip of the Spear | 矛尖 | 106 | Michael Bramnik |
| 7 | War Is Hell | 战争即地狱 | 122 | Michael Bramnik |
| 8 | Tailors and Traitors | 裁缝与叛徒 | 142 | Rigby Bendele & John Compton |
| 9 | Among the Thorns | 荆棘之中 | 158 | Jason Keeley |
| 10 | Midnight Burning | 午夜焚烧 | 176 | Jason Keeley |
| 11 | The Final Damnation | 终极天罚 | 192 | Jason Keeley |
| – | Beyond the Campaign | 战役之外 | 202 | – |
| – | Corentyn 城邦志 | 寇兰廷城邦志 | 210 | Rigby Bendele |
| – | Chelaxian Aristocracy | 切利亚斯贵族 | 218 | John Compton |
| – | Adventure Toolbox | 冒险工具箱 | 224 | – |

## 工作目录布局

```
地狱天命/
├── 地狱天命-英文.pdf              (源, 363MB, 258p)
├── 地狱破灭译文-1.0初校版.pdf      (前传中译, 212p)
├── extract_pdf.py                  (PyMuPDF 抽取脚本, sidebar 自动剥)
├── mine_prequel_terms.py           (前传术语挖矿脚本)
├── build_term_map.py               (合并 SEED + 挖矿 → 02_term_map.json/md)
├── lookup_term.py                  (SRD TM 按需查询)
├── lookup_wiki.py                  (offline wiki grep 查询)
├── extracted/
│   ├── source_pages/page_NNN.md   (258 个英文源文件)
│   ├── pages.json                  (1.0 MB, 结构化)
│   └── text_reference.txt          (1.0 MB, dump)
├── reference/
│   ├── prequel_official_pages/    (1.0 PDF 212 个 MD 页)
│   ├── prequel_term_pairs.json    (307 条 AP-specific 术语对)
│   └── prequel_term_pairs.md      (人读版)
├── state/
│   ├── 00_project_init.md         (本文件)
│   ├── 01_progress.md             (per-page 进度日志)
│   ├── 02_term_map.json           (机读 AP-specific 术语表)
│   ├── 02_term_map.md             (人读版，frozen/tentative)
│   ├── 03_session_logs/           (per-session 收尾记录)
│   ├── 04_uncertain_terms.md      (≥2 源冲突 + 自创待审)
│   └── 99_handoff.md              (最终交接，结尾写)
├── translated/page_NNN.md         (per-page 中文 + <details> 英文)
└── output/
    ├── 地狱天命-中英对照.md         (translated/* 合并)
    ├── 地狱天命-中文.md             (剥 <details>)
    └── 地狱天命.docx                (最终交付，用 python-docx 装配)
```

## 术语优先级（用户指定）

1. **`pf2_cn`**（FVTT 系统 i18n） — 通过 `python lookup_term.py "<term>"` 查询
2. **`pf2e_compendium_chn`** (zh-CN pack) — 同上
3. **`state/02_term_map.md`**（AP-specific：种子 + 前传挖矿 + 翻译中新建）
4. **wiki**：在线 `https://pf2.huijiwiki.com/wiki/<term>` (WebFetch) 优先；离线 `python lookup_wiki.py "<term>"` 兜底
5. 自创（实在找不到才用，记 `04_uncertain_terms.md`）

注：tier 1/2 实际打包在同一份 `翻译流程/tm_cache/tm_3source.json`，由 `lookup_term.py` 统一返回（带 source 字段标注是哪一层命中）。

## 翻译输出格式

每页 `translated/page_NNN.md`：

```markdown
## Page N — [章节名（若适用）]

[完整中文翻译]

<details><summary>English Source</summary>

[原文]

</details>
```

最终装配：
- `output/地狱天命-中英对照.md` = translated/page_*.md 顺序拼接 + 章节边界标题
- `output/地狱天命-中文.md` = 上一份剥 `<details>` 后
- `output/地狱天命.docx` = python-docx 装配（pandoc 不在 PATH）

## 工具可用性

| 工具 | 状态 | 备注 |
|---|---|---|
| PyMuPDF (fitz) | ✓ 1.26.6 | 文本层抽取 OK，sidebar 已自动剥离 |
| python-docx | ✓ 1.2.0 | Phase C 用于 MD→DOCX 装配 |
| pandoc | ✗ 不在 PATH | 不用，改用 python-docx |
| 翻译流程/tm_cache/tm_3source.json | ✓ 存在 (20MB, May 9) | 17 天前缓存，仍可用 |

## Setup 结果摘要

- ✅ A1：258 页抽取完成，sidebar 自动剥 125 页，平均 3761 chars/page
- ✅ A2：307 条 AP-specific 术语对挖出（SEED 70+ + 1.0 PDF 64 + c71c1ec7 192）
- ✅ A3：02_term_map.json + .md 生成（108 frozen / 199 tentative）
- ✅ A4：lookup_wiki.py 验证可用（实测 Twilight Talons → 暮爪 命中）
- ✅ A5：本文件 + 01_progress.md + 04_uncertain_terms.md 初始化

下一步 → Phase B per-page 翻译循环（从 page_001 开始）。
