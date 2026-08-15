# 主控自留项（本轮结束前必须处理）

## 1. idx 264 —— triage 时被一个分片漏掉的那条

**状态**：triage 拿回 269/270 个 verdict，唯独 `idx 264` 没有。主控已自行核实，**判 OPEN**。

**缺陷**：Ember 三个 RegionBehavior 子类型的整张配置表单是裸英文。
实测 `modules/ember/scripts/ember.mjs`：

| 类 | 行 | 裸串 |
|---|---|---|
| `TrapTriggerRegionBehavior.defineSchema` | 2554-2570 | `Once` / `Locked` / `Discovered` / `Triggered Behaviors` / `Script` / `Trigger Text` / `Pause Game` 七个 label + 七条 hint |
| `AreaEffectRegionBehavior.defineSchema` | 2685-2704 | `Chat Message Description` / `Ability Score` / `Save DC` / `Damage Formula` / `Effect Data` + 四条 hint（`EFFECT.Image` 是真 i18n 键，不算） |
| `EmberFootstepSurfaceRegionBehavior.defineSchema` | 2765-2776 | `Material` + hint + 五个 choices（`Grass`/`Metal`/`Stone`/`Water`/`Wood`） |

**⚠ 不要按 lang 键修。** 这正是第十三轮 `2026-08-14c` 已经**否决**过的做法
（「把英文裸词当 i18n key」）：Foundry 的 i18n 表是全局合并的，
把 `"Once"` / `"Locked"` / `"Script"` / `"Material"` / `"Water"` 这种通用词塞进
`lang/cn.json`，**任何**模块调 `localize("Water")` 都会拿到我们的译文。
这批串里通用词的比例还特别高，风险比第十三轮那 7 个更大。

**建议的修法**：在 `init` 阶段就地改写这三个数据模型的 schema 字段标签 ——
从 `CONFIG.RegionBehavior.dataModels["ember.trapTrigger"]` 之类拿到类，
遍历 `schema.fields` 把 `field.label` / `field.hint` / `field.choices` 换成中文。
作用域精确到 Ember 自己的三个子类型，不碰全局 i18n 表，也不依赖表单的 DOM 选择器。
落点在 `1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs` 或 `register.js` ——
**本轮这两个文件都被修复 agent 独占着，必须等它们交回再动。**

**验证**：改完在世界里开一个 Region 的行为配置；机械侧可以断言
`CONFIG.RegionBehavior.dataModels[<id>].schema.fields.once.label === "仅一次"` 之类。

---

## 2. 落盘与收口（修复轮交回后按顺序做）

1. **三方合并**所有批次 —— 多个 agent 基于同一份 base 产批次，直接按顺序落会静默回滚：
   ```powershell
   python "$P\4-临时脚本\2026-08-12-fix\merge_batches.py" --manifest <manifest.json> --scan
   python "$P\4-临时脚本\2026-08-12-fix\merge_batches.py" --manifest <manifest.json> --merge --out-dir <merged>
   ```
   真冲突逐条人工裁决写进 `resolutions.json`，再逐包 `apply_translations --force`。
2. **改过 `mappings.mjs` / `runtime-converters.js` 就必须重跑**（agent 被禁止自己跑）：
   ```powershell
   node "$P\3-常用脚本\release\generate_runtime.mjs"
   python "$P\4-临时脚本\2026-08-06\crosscheck_vs_crucible_fr.py"
   ```
3. **孪生同步放最后**：`python "$P\3-常用脚本\qa\sync_twin_packs.py"` —— 主线每落一批就会制造新分叉。
4. **5.4 全套复验**（十四项都应为 0）＋ `flatten_lang.py` 三数相等。
5. 更新 `PROJECT.md`：抬头版本号**当前是过期的**（写着 0.9.6 / 1.1.7，
   实际已发 **crucible-cn 0.9.7 / ember_cn_unofficial v1.1.10**），
   并补第十三轮 v1.1.8/1.1.9/1.1.10 与本轮的年表与决议。

---

## 3. 两条最大的内容缺口（idx 34 / 41，严重档）

**encounterTokens 约 894 处 token 覆盖名从未进过英文基线。**
第十三轮已经把 mapping 与转换器补上了（新增 `crucibleTokenName` / `emberEncounterTokenNames`），
但**英文基准没有重抽**，所以 `compendium/cn` 里 0 条译文，运行时 GM 一放怪仍是满地英文名。

要做的（**必须等 `pipeline` agent 交回 `mappings.mjs` 之后**，否则会拿旧 mapping 重抽）：

```powershell
# 1. 先归档当前英文（漏了这步下次就没得比）
Copy-Item -Recurse "<repo>\compendium\en" "5-其他内容\english-baseline\<包>-<旧版本>"
# 2. 重抽 —— ⚠ 只并新字段，别整体覆盖，否则会回退 LOCAL-PATCHES.md 记的三条上游笔误补丁
node "$P\3-常用脚本\extract\extract_en.mjs" --package <foundry包> --out <临时目录>
# 3. 重打 LOCAL-PATCHES.md 的补丁
# 4. TM 预填（大部分 token 名在库里别处已有译文）
python "$P\3-常用脚本\tm\fill_missing.py" --repo <repo> --out-dir <批次目录>
# 5. 剩余的走正常翻译批次
```

参照先例：`Scene.levels` 那次（2026-08-13b）就是「优先完整汉化，补管线」，做法见第 8 节。
