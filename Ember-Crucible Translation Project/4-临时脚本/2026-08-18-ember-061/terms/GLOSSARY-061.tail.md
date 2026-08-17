
---

## 附表 C：0.6.1 的新词但**不在名称位**、第二阶段翻正文时会撞上的

这些只出现在正文里（`text`/`description`/`exposition`…），本单元不裁，但先点名，免得十个译者各译各的：

| EN | 出处 | 备注 |
|---|---|---|
| `Fin'ian Sphere` | Kadra Zann 正文 | 申特人（`Shent`→申特）造的施法媒介神器 |
| `Ambral` | Proving Your Metal 正文 | 奥肯加德（`Oakengarde`→奥肯加德）特产魔法金属；合金名 `Ambral Bronze` 已在主表 |
| `Casia` | Where Shadows Lie 正文 | 库里已有「卡西娅 Casia」，沿用 |
| `Caryx Savannah` | 只在补丁说明页 | 尚未产生名称叶，待它进包再裁 |
| `Hair roots` / `Hair Highlight` | 只在补丁说明页 | 见 §8 |

---

## 8. 代币颜色命名改动（`Hair1`/`Hair2` → `Hair roots`/`Hair Highlight`）的影响面：**零**

实测三处，全部为空：

1. `grep -ri "Hair1|Hair2|Hair roots|Hair Highlight|hairRoot"` 扫整个项目根 —— **0 命中**。
2. `1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs`（24.9 万字符的硬编码译文表）里 `Hair` —— **0 命中**。
3. 上游 `modules/ember/scripts/ember.mjs` 里这批串确实存在（`"Hair Base"` / `"Hair Roots"` / `"Hair Highlights"` /
   `"Hair Glow"` / `"Hair Sparkle"`，另有一条遗留的 `"Hair 1"`），但它们是**代币制作器的颜色槽名**，
   我们的硬编码表从来没有覆盖这一层。

⇒ **本轮不需要跟改任何东西。** 若将来要把代币制作器也汉化，这六个串是新的入表候选
（发根／发色底／挑染／发光／闪粉），届时再裁。

---

## 9. 本单元的产物与脚本

| 文件 | 是什么 |
|---|---|
| `GLOSSARY-061.md` | **本文件**。第二阶段的唯一术语依据 |
| `collisions.json` | 撞车清单（含逐条处置建议） |
| `candidates.json` / `candidates_extra.json` / `candidates_lang_hardcoded.json` | 裁决表的**机器可读源**（撞车判据的输入） |
| `probe_names.py` | 前置自证 + 抽 1122 名称叶 |
| `build_index.py` | 建库的双向索引（EN→CN / CN核心→EN） |
| `check_collisions.py` | 双向撞车判据（自带正反自证） |
| `gen_glossary.py` | 由候选文件渲染本表的表格段 |
| `names_raw.json` / `names_uniq.json` / `known_216.txt` / `new_ctx.txt` | 中间产物 |
| `lib_en2cn.json` / `lib_cn2en.json` / `lib_prov.json` | 库索引（8375 个 EN 键 / 15177 个 CN 核心键，带出处） |
