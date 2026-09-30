# RK 局部汉化公开 ZIP 原生验收记录

本记录覆盖计划 Task 3 的 57 个独立显示值或明确动态生产槽。静态对账与受控测试保留；当前原生验收绑定真实公开 ZIP 安装后的固定报告，不把测试数作为字段覆盖率。

| 本次公开 ZIP 分类 | 数量 | 证据边界 |
| --- | ---: | --- |
| 已实命中原生显示槽 | 25 | 普通标题、骰点与表头，目标前缀，六个次序标题，学识提示及空表标题，未受训与专家完整等级、目标专家缩写槽，失败与大失败，七个标准技能名，固定驾轻就熟最终标题 |
| 尚未完整命中 | 30 | 包括 1 个只有原生 breakdown 标签的部分证据；成功、其他等级、tooltip、专长提示、通知、错误、reservation、厄运抵消及 Automatic Knowledge 等仍未在本次报告实触发 |
| 翻译不适用 | 1 | 用户及世界自定义名称保持原样；隐私另核 |
| 遗留名称未核验 | 1 | 军师架势效果没有自身完整英文名称来源，本轮保持旧名 |

主要来源为 [公开 ZIP 双客户端固定报告](C:/Users/Taka/Desktop/fvtt/output/automation-eldamon-ux-20260930/qa/dual-client-report-1790748695576.json)，SHA-256 `a04193d8e6f74f1215e7f2a39adba66fc6000c506e098a634fae55a0a36970fa`。报告已关闭：13 个命令全部 ok，errors 为空；Core 14.368 / PF2e 8.5.1 / Workbench 7.7.5 / 模块 0.9.20 / 语言 cn。发布提交为 `a2dc695bcb8cd6bdda630587c0f3990147b5bf34`。各槽的精确匹配值、卡 ID、命令 ID 与指纹保存在 [字段对账](C:/Users/Taka/.codex/worktrees/automation-native-20260930/fvtt/docs/fortress-automation/rk-localization-2026-09-30.json)。

- `package-rk-cn-normal-no-target`：一个 1d20，点数 15，七个技能候选；当前表头为“技能 / 熟练 / 调整值 / 结果”。
- `package-rk-cn-normal-target`：一个 1d20，点数 2，自动社群一项，总值 15；DC 20/22/25/30 分别显示失败/失败/大失败/大失败。第 5/6 次只命中学识难度提示，空学识技能表只证明表头。
- `package-rk-cn-assurance-no-target`：固定 10 + 原生熟练 9 = 19，零骰；当前最终标题为“回忆知识 — 驾轻就熟”。
- `package-rk-player-actual-secret-privacy`：三卡实际 renderHTML 均 isContentVisible=false；玩家只见 Core 秘骰占位，目标真名、DC、degree、结果及候选技能不可见。privately rolled some dice / ago 是 Core 生产槽，未声称该外部 UI 已汉化。

本次 WB04 明确显示“熟练”：精确 Workbench Prof 表头调用 knowledgeProficiencyLabel，优先当前中文 PF2E.ProficiencyLabel，缺失中文键才回退“熟练”。报告没有分别捕获这两个返回来源，不能从相同文字断言实际走了哪条分支。旧报告的“熟练度”已经归档，不作为本次当前值。WB30“大失败”替换本次未命中的 WB32“成功”；历史与本次合并曾实命中 26 槽，但本次公开 ZIP 仍为 25 槽。

`package-gm-actual-served-151-bytes` 与 `public-release-actual-served-151-bytes-after-zip-install` 实际 fetch 的 151 文件 SHA 全部匹配 Git blob 与 ZIP。公开 ZIP SHA-256 为 `14a7029fe75849471336835d681d1202a073e3ce326f7a81b844a787570f79d7`；[外部字节对账](C:/Users/Taka/Desktop/fvtt/output/automation-eldamon-ux-20260930/release-tools/release/evidence-verification.json) 分别记录工作区 raw SHA 与包内 SHA。

| RK 运行文件 | 工作区与包内关系 |
| --- | --- |
| knowledge-display.mjs | exact-bytes |
| knowledge-workbench.mjs / knowledge-automatic.mjs / knowledge-automation.mjs | CRLF-to-LF-only |
| 既有 knowledge-entrypoints.mjs | CRLF-to-LF-only，单独登记 |

不能把 LF 归一后的相同误写为全部工作区 raw 字节等于公开包。原始 Workbench 命令与 fixture 保持原 SHA，没有修改原宏。开发阶段 fixture ID、选择器与 Combat 夹具错误仍保留在历史记录，不能冒充本次 clean 产品成功。原宏未闭合 rank div、多余引号、Map.length 与未配对链接结束标签作为既有源问题保留；Core 保存后的 HTML 规范化不作为 adapter 修改属性的证据。
