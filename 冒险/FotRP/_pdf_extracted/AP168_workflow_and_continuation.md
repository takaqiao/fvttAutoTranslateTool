# AP168 PDF 集成 — 工作流文档 + 继续指南

**会话日期**: 2026-05-19
**输入**: 新汉化组 PDF `FotRP/AP168 《赤凰斗士》3-3 巅峰之王 (1).pdf` (65 页, 2.5MB, 2026-05-19 出)
**目标**: 用新 PDF 优化 `FotRP/需要翻译/NEW/第三本/` JSON journal (+ 跨第三本生物的 bestiary)
**累计修改**: 6 轮共 ~145 处, 5 文件全部 JSON 有效 + 0 HTML 不平衡

---

## 一、工作流总览

```
1. 新 PDF 出现
   ↓
2. PDF 文本提取 (PyMuPDF 直接提取失败 → 用 Tesseract OCR chi_sim + 双栏切割)
   ↓
3. 自动术语对照表生成 (regex 抽取双语对照 + PF2 术语)
   ↓
4. 现 JSON 与 PDF 交叉对照 (识别分歧点)
   ↓
5. 按权威层级裁决每个分歧 (pf2cn > AlphaStar > wiki > glossary > 汉化组 PDF)
   ↓
6. 备份 + 应用修改 (Python replace 脚本, 多轮迭代)
   ↓
7. QA (JSON 有效性 + HTML 平衡 + 残留扫描)
   ↓
8. 记入 memory + 总结文档
```

---

## 二、关键脚本与工具

### 2.1 PDF 提取 (OCR)

**字体障碍**: 汉化组 PDF 用 DengXian 字体但无 ToUnicode 映射, PyMuPDF 直接提取得 mojibake.

**解决**: Tesseract OCR (chi_sim 语言包从 tessdata_fast 下载到用户目录, 设 `TESSDATA_PREFIX`), 双栏切割避免读串.

```python
import os, fitz, pytesseract
from PIL import Image
os.environ["TESSDATA_PREFIX"] = r"C:\Users\Taka\Desktop\fvtt\FotRP\_tessdata"

def ocr_two_col(page, dpi=240):
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    w, h = img.size
    mid = w // 2
    margin = int(w * 0.02)
    left = img.crop((0, 0, mid + margin, h))
    right = img.crop((mid - margin, 0, w, h))
    text_l = pytesseract.image_to_string(left, lang="chi_sim+eng", config="--psm 4")
    text_r = pytesseract.image_to_string(right, lang="chi_sim+eng", config="--psm 4")
    return text_l + "\n[COLUMN BREAK]\n" + text_r
```

性能: 65 页 ~6 min @ 240 DPI. 输出 `_pdf_extracted/AP168_book3_zh.ocr.jsonl`.

### 2.2 双语对照抽取

```python
# 中文(English) 模式
patt_zh_en = re.compile(r"([一-鿿·•・\-—]{1,8}[一-鿿·•・\-—\s]{0,5})\s*\(([A-Z][a-zA-Z\s\'\-\d]{2,40})\)")
# English(中文) 模式
patt_en_zh = re.compile(r"([A-Z][a-zA-Z\s\'\-\d]{2,40})\s*\(([一-鿿·•・\-—\s]{1,12})\)")
```

输出 `_pdf_extracted/AP168_book3_zh.english_names.txt` (121 对) + `AP168_pf2_terms.txt` (134 PF2 术语).

### 2.3 跨文件 / 历史 Wave 漂移扫描

```python
historical_pairs = [
    # Wave 6 C2 - Qi/Ki Remaster
    ("内劲爆发", "斗气迸发", "Inner Upheaval"),
    ("真气爆裂", "真气波", "Qi Blast"),
    # Wave 7-8 同义词归一
    ("锦标赛", "武道会", "Tournament canonical"),
    ("维度", "位面", "BK"),  # 注: 需排除 维度叠加 等固定词
    # ...
]
for path in target_files:
    with open(path, encoding="utf-8") as f:
        c = f.read()
    for old, new, src in historical_pairs:
        if c.count(old) > 0:
            print(f"  {old}={c.count(old)} → 应改 {new}, {src}")
```

### 2.4 上下文核查 (避免误改)

```python
# 不要直接 replace_all, 要先看上下文
for term in ["维度", "选手", "真气", "您"]:
    for m in re.finditer(re.escape(term), content):
        i = m.start()
        before = content[max(0,i-15):i]
        after = content[i+len(term):i+len(term)+15]
        print(f"...{before}【{term}】{after}...")
```

### 2.5 应用 + QA

```python
# 1. 备份
shutil.copy2(path, bak_dir/base)
# 2. 替换
content = content.replace(old, new)
# 3. 验证 JSON
json.loads(content)
# 4. HTML 平衡检查 (用 html.parser 子类 TagBalance)
```

---

## 三、权威层级 (用户决策)

**Wave 6 (2026-05-12) 用户确认**:

```
pf2cn (system/sf2_cn)
  ↓
pf2compendium_chn (AlphaStarguide repo)
  ↓
wiki (pf2.huijiwiki.com / 本地 pf2wiki-scraper 缓存)
  ↓
glossary.json (项目本地, 10K+ 条目)
  ↓
汉化组 PDF (最低优先级, 仅当上述全部静默时考虑)
```

**关键原则**:
- 当 pf2cn / AlphaStar / wiki / glossary 静默时, 才考虑接受汉化组 PDF 译法
- 项目内一致性是 secondary 考虑 (不应单方面引入分歧)
- 名词内部分歧 (如 优·苏然 vs 叶索兰) 应跨文件统一

---

## 四、本次会话的 6 轮修改

| 轮次 | 修改类型 | 处数 | 文件范围 | 主要 canonical 出处 |
|---|---|---|---|---|
| 1 | Yoh Souran 译名统一 + Hwanggot 清残留 | 39 | 4 journals | 现 bestiary canonical + 新 PDF |
| 2 | PF2 术语 (Grapple/Stygia/Dis/Radiant Blast) | 8 | back-matter + ch2 | Wave 6 + wiki + glossary |
| 3 | 凤凰血脉法术名 | 8 | back-matter + bestiary + ch2 | glossary + Wave 4/5 |
| 4 | Bestiary 跨第三本生物 (Grapple/Radiant Blast) | 56 | bestiary | Wave 6 + glossary |
| 5 | Wave 历史 canonical 残留 | 13 | bestiary + journals | Wave 5/6/7 |
| 6 | 不知盖 → 天蚀狗 (Bul-Gae) | 29 | bestiary + back-matter | **pf2wiki 独立 page** |

**总计 ~145 处**, **0 回归**.

---

## 五、保留 (不改) 的决策汇总

经权威层级核查后, **明确保留现 JSON** 的术语:

### 5.1 高优先级源裁决保留
| 术语 | JSON | PDF | 理由 |
|---|---|---|---|
| Glass Lighthouse | 琉璃光屋 | 琉璃灯塔 | glossary canonical |
| Hwanggot (主体) | 花国 | 垦郭 | glossary canonical |
| Aldanar Unmar | 阿达纳·乌马尔勋爵 | 阿达娜尔·安马尔领主 | Wave 1 审定 |
| Lady Xhai Zhia | 夏之芽女士 | 载嘉夫人 | Wave-stable, 无更高源 |
| Flying Mountain | 飞行山脉 | 飞来山 | 项目内 17× 稳定, 无更高源 |
| Kaifen Bay | 凯芬湾 | 开芬湾 | 项目内 7× 稳定, 无更高源 |

### 5.2 固定术语 (Wave 已审保留)
| 术语 | 出现处 | 不改理由 |
|---|---|---|
| 维度叠加/吞噬/连击/抓握/暗面之镜 | 26× | 固定能力名 (Syndara), Wave 8 BK 保留 |
| 晋级选手 | 19× | 固定术语 ("Finalists"), Wave 8 BK 保留 |
| 真气 | 15× | pf2cn 法术名 (与 斗气 不同概念) |
| 世界球 / 界球 | 14×+14× | full-name + 简称模式, 内部一致 |

### 5.3 主观风格差异 (都可)
- 章节标题 (奔向峰顶/分形丛林之碎/拆解雕匠：灯塔之诅)
- 16 巅峰专长名 (世间一切时间/仲裁之舞/...)
- 4 Phoenix 血脉法术 (See Invisibility 侦测隐形/Moment of Renewal 新生时刻/Rejuvenating Flames 新生之焰/Cleansing Flames 净化之焰)
- 林冠长老 (Canopy Elder), 渎圣守卫 (Desecrated Guardian), 八岐大蛇 (Orochi), 灵海龟 (Spirit Turtle), 冰尸合体 (Sthira) — wiki/glossary 静默, Wave-stable

---

## 六、如何继续 — 后续可推进的方向

按价值 / 工作量 排序:

### 优先级 A — 高价值 (建议下次推进)

#### A1. 跨书一致性 — 第一本/第二本对照 (类似流程)
**为什么**: Wave 9 Round 1 发现 NEW/bestiary 用「叶索兰」但 journals 用「优·苏然」是因 Wave 5 修改未传播. 第一本/第二本也可能存在类似不一致.

**操作**:
```python
# 跨第一本 + 第二本 + 第三本 journal 扫描相同术语 一致性
files_all = glob.glob(r"...\NEW\第一本\*.json") + \
            glob.glob(r"...\NEW\第二本\*.json") + \
            glob.glob(r"...\NEW\第三本\*.json") + \
            [r"...\NEW\pf2e.fists-of-the-ruby-phoenix-bestiary.json"]
# 检查每个 NPC/地名 在所有文件中的渲染
```

**预期产出**: 若有跨书分歧, 30-100 处修改.

#### A2. Wave 7-8 manual 项推进
Wave 7-8 progress 记录的 **未批量化** prose-level 修改:

- **「您」非引号 GM 旁白 → 「你」** (~76 处全 AP)
  - 需上下文检测: 在 `<p>` 中且不在 `"..."` / `「...」` 引号内的「您」改「你」
  - 工具: 写个上下文判别脚本, 排除对白
- **「对...进行 X」/「执行 X」/「通过 X 来 Y」 prose 用法** (~370 处)
- **长定语链拆句** (~17 处严重)

**预期产出**: 50-150 处, 文本流畅度提升明显.

#### A3. PDF 揭示的具体段落表述 (深度比对)
方法: 抽样 10 个 JSON page, 与 PDF 对应段落做 prose-level 比对, 找翻译错误或漏译.

**已经发现的潜在线索**:
- PDF p11 提到 "Greater Phylactery Faithfulness", "Possibility Tome" 等 Iron Mountain treasures — JSON ch1 是否有完整覆盖待核
- PDF p41-42 Syndara stat block: PDF 称 22 级, JSON 是否 22 级 ✓ (需核)
- PDF 提到 Yoh's wife = 柳景庆 (Kyoung-Yoo Liu), JSON 用 炅裕 — 是否更新留备查

### 优先级 B — 中价值

#### B1. Bestiary 其他生物名 wiki/glossary 核查
本次 Round 6 找到 Bul-Gae 有 wiki canonical (天蚀狗). 其他 9 个第三本怪物名也应批量查 wiki:
- Canopy Elder (林冠长老)
- Desecrated Guardian (渎圣守卫)
- Gumiho (九尾狐)
- Inmyeonjo (人面鸟)
- Lophiithu (深渊鮟鱇)
- Orochi (八岐大蛇)
- Sanzuwu (三足乌)
- Spirit Turtle (灵海龟)
- Sthira (冰尸合体)

**操作**:
```bash
# pf2wiki-scraper/out/titles.json 中搜 "天蚀狗" 类原生 page
grep -l '"title": "九尾狐"' pf2wiki-scraper/out/titles.json
```
若有原生 wiki page, 即为 canonical, 可考虑替换.

#### B2. Stat block 数值核对
本次 Round 4 找到 bestiary 中大量术语漂移. 数值层面 (HP/AC/saves/DCs) 是否与 PDF 一致还未核.

PDF Syndara 数据:
- 雕匠辛达拉 SYNDARA, THE SCULPTOR 22级生物
- 先攻 察觉+39
- 晶化巨兽辛达拉 SPINEL LEVIATHAN SYNDARA 24 级生物
- AC 51; 强韧+46 反射+38 意志+42
- HP 550; 免疫 ... 抗力 电击 25, 心灵 25, 弱点 混乱 25
- 速度 60 尺，飞行 60 尺
- ...

工具: 抽样 5 个 boss 数据卡 (Syndara/Spinel Leviathan/Mogaru/Tino's Toughest 各变体), 写 stat 提取器与 PDF 数字对照.

### 优先级 C — 低价值 (除非用户特别要求)

- **巅峰专长 16 个译名 PDF 化** — 改动量大, JSON 已 Wave 2 翻译稳定, 用户社群可能已熟悉
- **章节标题改 PDF 风格** — 都可, 不必要
- **林冠长老 → 树冠长者** (Canopy Elder) — 同上, Wave-stable

---

## 七、Quick Resume — 下次会话怎么接

### 7.1 必读 (按顺序)
1. `_pdf_extracted/AP168_workflow_and_continuation.md` (本文档)
2. `_pdf_extracted/AP168_total_summary_20260519.md` (6 轮总结)
3. `memory/project_fotrp_progress.md` Wave 9 段
4. `memory/feedback_term_priority.md` + Wave 6 用户权威层级

### 7.2 关键文件路径
```
PDF:
  C:\Users\Taka\Desktop\fvtt\FotRP\AP168 《赤凰斗士》3-3 巅峰之王 (1).pdf

OCR 输出 (供下次复用, 不必重跑):
  C:\Users\Taka\Desktop\fvtt\FotRP\_pdf_extracted\AP168_book3_zh.ocr.jsonl
  C:\Users\Taka\Desktop\fvtt\FotRP\_pdf_extracted\AP168_book3_zh.ocr.full.txt

第三本 NEW canonical (生产目标):
  C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\第三本\*.json (5 文件)
  C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\pf2e.fists-of-the-ruby-phoenix-bestiary.json

Tesseract chi_sim 数据 (供下次 OCR):
  C:\Users\Taka\Desktop\fvtt\FotRP\_tessdata\chi_sim.traineddata

备份 (按需 rollback):
  C:\Users\Taka\Desktop\fvtt\FotRP\_backup\20260519_book3_round{1-6}*/

权威源:
  C:\Users\Taka\Desktop\fvtt\glossary.json
  C:\Users\Taka\Desktop\fvtt\pf2wiki-scraper\out\glossary_wiki.json
  C:\Users\Taka\Desktop\fvtt\system\pf2_cn\zh_Hans\zh_Hans.json
  C:\Users\Taka\Desktop\fvtt\system\sf2_cn\zh_Hans\zh_Hans.json
```

### 7.3 标准操作流程 (SOP) — 复用本工作流

```bash
# 假设需要做 Round 7 / 类似新轮次

# 1. 列出候选修改 (与 PDF 对照, 或扫历史 Wave drift)
python scan_candidates.py

# 2. 核查每个候选的上下文 (避免误改)
python check_contexts.py

# 3. 查权威源 (priority hierarchy)
#    a. pf2cn (system/sf2_cn)
#    b. AlphaStar
#    c. wiki (pf2wiki-scraper)
#    d. glossary
#    e. 汉化组 PDF

# 4. 备份
mkdir _backup/20260519_book3_round7_xxx
copy NEW/第三本/*.json _backup/.../

# 5. 应用替换
python apply_round7.py

# 6. QA (JSON valid + HTML balance + 残留扫描)
python qa.py

# 7. 记入 memory + 写 round 报告
```

### 7.4 决策模板 (针对每个分歧)

```
[术语] PDF=X, JSON=Y

权威源核查:
  pf2cn: ___ (有/无, 译为___)
  AlphaStar: ___
  wiki: ___
  glossary: ___

裁决:
  → 改 (上层源支持改): apply
  → 不改 (上层源支持 JSON, 或都静默): keep JSON
  → 不改 (Wave-stable, 都可): keep JSON
```

### 7.5 不可逆操作前的安全检查

```python
# 所有修改前必备
1. 确认备份目录存在且写入了原文件
2. 用 dry_run / 计数模式先看影响范围
3. 替换后立即跑 JSON 有效性 + HTML 平衡
4. 计数 残留 检验 (避免还有相同 token 漏掉)
5. 抽样读修改后内容确认可读性
```

---

## 八、风险点与教训

### 8.1 字体字体编码陷阱
- 不要假设 PyMuPDF 能直接提取所有 PDF 文本
- 汉化组 PDF 用 subset 字体常无 ToUnicode → 必须 OCR
- OCR 时双栏 PDF 必须先切割

### 8.2 跨文件不一致
- Wave 历史修改 可能只触及部分文件 (working dir vs NEW dir)
- NEW dir 是生产目标 (用户 import 到 Foundry 的)
- working dir 可能停留在更老 Wave 状态
- 修改前先确认目标是 NEW

### 8.3 替换前必看上下文
- 「优」/「维度」/「选手」/「您」 等单字术语 易误伤
- 「擒抱」可能是 prose 动词 (拥抱式包围) 或 PF2 Grapple action — 都要 fix 因为 PF2 zh-CN canonical 是 擒拿
- 「真气」 (Ki 法术) vs 「斗气」 (Qi feat) — 不同概念, 不可互换
- 「位面之门」 (Dimension Door 4 环) vs 「异界之门」 (Plane Gate 10 环) — 不同法术

### 8.4 PDF 也不是绝对权威
- PDF 内部都不完全一致 (如 琉璃灯塔/琉璃光屋 混用)
- OCR 错误也可能让 PDF 看起来 "不一致"
- glossary / wiki / pf2cn 高于 汉化组 PDF

### 8.5 不要单方面引入分歧
- 项目内已稳定的术语 (如 凯芬湾/飞行山脉), 即使 PDF 不同, 也不应单方面换
- 除非 wiki/glossary 提供新 canonical

---

## 九、附: 本次会话产物清单

`_pdf_extracted/` 目录:
```
AP168_book3_zh.ocr.jsonl                    — 65 页 OCR JSONL (~312KB)
AP168_book3_zh.ocr.full.txt                 — 全文可读
AP168_book3_zh.english_names.txt            — 121 双语对照
AP168_pf2_terms.txt                         — 134 PF2 术语
AP168_diff_decisions.md                     — Round 1 决策清单
AP168_optimization_summary_20260519.md      — Round 1 总结
AP168_total_summary_20260519.md             — Round 1-6 总结
AP168_workflow_and_continuation.md          — 本文档 (流程 + 继续指南)
titles_lookup.txt                           — 章节标题 + 关键术语
yoh_audit.txt / yoh_counts.txt              — Yoh Souran 内部不一致取证
capstone_check.txt                          — 16 巅峰专长名状态
spells_audit.txt                            — 凤凰血脉法术对照
pf2_terms_check.txt                         — PF2 通用术语对照
pf2cn_spell_check.txt                       — pf2cn 法术核查
wave_drift_scan.txt                         — 历史 Wave 漂移扫描
borderline_contexts.txt                     — 维度/选手/真气/您 上下文
fix_candidates_contexts.txt                 — 锦标赛/位面之门/阳舰二号 上下文
sunwing_check.txt                           — 阳舰二号 跨文件审计
door_context.txt                            — 次元门 vs 异界之门 核查
round7_term_scan.txt                        — Round 7 候选扫描
round{1-6}_apply.txt                        — 各轮修改报告
final_qa_round{1-6}.txt                     — 各轮 QA 报告
final_round1_6_qa.txt                       — 最终 6 轮综合 QA
```

`_backup/` 目录 (按需 rollback):
```
20260519_book3_pdf_optimization/  — Round 1 (Yoh + Hwanggot)
20260519_book3_round2/             — Round 2 (PF2 术语)
20260519_book3_round3_spells/      — Round 3 (spell names + Contingency)
20260519_book3_round4_bestiary/    — Round 4 (bestiary Grapple/Radiant)
20260519_book3_round5_canonical/   — Round 5 (Wave 历史残留)
20260519_book3_round6_bulgae/      — Round 6 (天蚀狗)
```

`_tessdata/` 目录:
```
chi_sim.traineddata                — Tesseract 简体中文数据 (供后续 OCR)
eng.traineddata
```
