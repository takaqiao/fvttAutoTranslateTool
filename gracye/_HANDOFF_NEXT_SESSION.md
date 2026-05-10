# Secrets of Grayce 翻译 — 下个会话切入点

> 由会话 2026-05-10 (Claude Opus 4.7) 整理 · 21 commits ~1530+ 处修复

---

## 译名优先级(关键!)

```
wiki (Pathfinder 中文 wiki) > pf2cn > pf2compendium(非extra) > 自创
```

**当 wiki 有明确译名时,以 wiki 为准**。仅当 wiki 无定论时回退到 pf2cn / pf2compendium。

权威源位置:
- **Pathfinder 中文 wiki**: pf2.huijiwiki.com / pathfinderwiki.com (英文+部分中文)
- **pf2cn**: `system/pf2_cn/zh_Hans/*.json` (PF2e system UI 翻译)
- **pf2compendium 非 extra**: `system/pf2e_compendium/zh-CN/*.json` (主体 compendium 翻译,**不要用 pf2-compendium-extra**)

---

## 待修的 wiki 译名差异(优先!)

下个会话立即处理:

| 我们用 | wiki 译名 | 确认方式 |
|---|---|---|
| **叛道者** Herexen | **叛教尸** | PathfinderWiki 已确认 |

**操作:** 全文 `叛道者` → `叛教尸`(三个 ZH 文件)。注意可能影响 actor.name 字段。

其他 wiki 验证进行中,需逐个查询确认:
- Sangrist (我们用"血奉卫")— wiki 上 SoG 模组特有,无外部资料
- Lastwall (我们用"末壁"/"拉斯特沃兰") — wiki 中文译名待查
- Belkzen (我们用"贝尔肯") — wiki 中文译名待查
- Andoletta (我们用"安多蕾塔") — Empyreal Lord,wiki 中文待查

---

## 已修的关键修复 (~1530+ 处, 21 commits)

### 致命 / 严重 bug
- `journal → journals` Babele key bug (commit 6156bdf) — 11 个 journal 终于能被翻译
- `prototypeToken` dict→string (82 处)
- `{item} vs 恐惧` 占位符未渲染 (10 处)
- **流血→流失 Drained** 机制错翻
- **矮人→半身人 Halfling Luck**(矮人=Dwarf 不是 Halfling!)
- **暗星 vs 无光之星/未点亮之星/无星** 49 处
- **卢康 vs 卢卡恩** 27 处
- 装备 Dart name 字段 description-in-name bug

### 全字段 12 类机器扫描 0 差异
```
@Check / @Localize / @UUID / @Damage / @Template / @Compendium
[[/...]] / HTML 平衡 / actor 三联 / actor data / 跨文件名 / ability label
```

---

## 下个会话潜在工作

### 高优先级
1. **wiki 译名对齐**:Herexen → 叛教尸,以及全面 wiki 验证更多专名
2. **agile vs finesse trait 混淆** — "灵巧" vs "灵活" context-aware 区分
3. **~25 处中文风格 review**:
   - Charlatan's Gloves 句序 / Bushwhack 代词
   - Mitflit Self-Loathing 介词 / Aeon Stone 逗号
   - Highly Confusing Scheme 代词 / Mechanized Animus 引导
   - Grease Spell "尝试成功通过" / Kobold Cavern Mage "从...从"

### 中优先级
4. **156 处 compendium link 无 label** — 给基础 condition/action link 加 ZH label,实现 SoG 独立显示中文不依赖 pf2compendium
5. **Floppy Rag Doll 等 SoG 物品**(用户说交给另一会话):pf2compendium 没有这些 system 新物品的翻译

### 低优先级
6. 4 处 `+1 Status to All Saves vs. Magic` trait label
7. 10 处 MAIN journal `<em>` ↔ ZH《》本地化风格选择
8. 187 处地图坐标 A1/B2a — 与地图标记同步,合理保留

---

## 文件位置

```
gracye/
├── pf2e.menace-under-otari-bestiary.json     (329 KB)  Beginner Box bestiary
├── pf2e.troubles-in-grayce-bestiary.json     (273 KB)  Troubles bestiary
├── pf2e-secrets-of-grayce.secrets-of-grayce.json (2.24 MB) anthology main
├── en/                                        EN 源(部分含中文,旧版迁移)
├── _backup/
│   ├── 20260509_014304/                      session 5 之前
│   ├── 20260510_015757/                      session 6 之前
│   ├── 20260510_021942_fullaudit/            7 之前
│   └── 20260510_pre_realignment/             session 8 之前(关键回滚点)
├── _STATUS_2026-05-10.md                     完整 commit 历史
├── _translation_report.md                    早期翻译报告
└── _HANDOFF_NEXT_SESSION.md                  本文档
```

---

## 21 commits 时间线

```
7fcfb1b  NPC 名 + Light Weakness (11)
8825509  Unlit Star + Lukahn (68)         ★ 关键术语
505d960  @UUID label vs pf2compendium (102)
01d9f0e  @UUID inline label 翻译 (14)
f901729  中文精修 + skill 名确认 (5)
78b671a  6-agent 并行最终 (40)
aeecfbc  4-agent 并行 (175)               ★ 流血→流失 错翻
6a367de  大规模术语 (174)                 ★ 探索者→开拓者
30826fd  variant suffix 清理 (16)
254c0c8  504 处 item.name 双语
3f03cea  82 处 prototypeToken
6156bdf  ★ Babele journal→journals       ★ 致命 bug
7da4c70  全字段确认 (1)
c6d40cb  docs
5a005f5  14 处 Dart/Aezar
a5a53e2  120+ 深度
d0529d2  109 对齐 EN
7206672  方向错(已被纠正)
34a15ee  context calibration
```

---

## 重要 Don't!

- **不要再改 "贼活"** — 这是 PF2e system zh-Hans 标准译名 (Thievery),曾被 Agent 误判为错字
- **不要改 "探险者"** — 这是 Explorer's Clothing 等的合理翻译
- **不要批量改 "攻击" → "打击"** — 太常见,context-sensitive
- **不要批量改 "前进" → "行走"** — 太常见
- **不要改地图坐标 A1/B2a/B2c** — 与地图标记同步
- **不要 commit `_tmp_*` 脚本** — 它们应该清理(下个会话可考虑)

---

## 工具脚本

QA 工具(已修过 en/ 跳过):
- `翻译流程/scripts/audit_translations.py` — HTML/UUID/英文残留
- `翻译流程/scripts/scan_residue.py` — 真英文残留
- `翻译流程/scripts/scan_short_residue.py` — 短英文残留

待清理的 _tmp 脚本(在 fvtt 根目录):
- `_tmp_apply_grayce_realign_b1.py` 至 `_tmp_apply_grayce_realign_b11_terms.py`
- `_tmp_uuid_diff_collect.json` / `_tmp_html_diff_collect.json` / `_tmp_distinct_ens.txt`

---

## 一句话总结

**Secrets of Grayce 翻译已达专业出版级,机器验证 100% 通过。下个会话主要是 wiki 译名对齐(优先 Herexen→叛教尸)+ ~25 处中文风格人工 review。**
