# RK 局部汉化原生验收记录

本记录覆盖计划 Task 3 的 57 个独立显示值或明确动态生产槽。静态对账与受控测试保留；原生验收逐槽绑定成功命令的实际 ChatMessage 内容，测试数不作为字段覆盖率。

| 分类 | 数量 | 证据边界 |
| --- | ---: | --- |
| 已实命中原生显示槽 | 25 | 普通标题、骰点与表头，目标前缀，六个次序标题，学识提示及空表标题，未受训与专家完整等级、目标专家缩写槽，成功与失败，七个标准技能名，固定驾轻就熟最终标题 |
| 尚未完整命中 | 30 | 包括 1 个只有原生 breakdown 标签的部分证据；其他等级、critical、tooltip、专长提示、通知、错误、reservation、厄运抵消和 Automatic Knowledge 等仍待实触发 |
| 翻译不适用 | 1 | 用户及世界自定义名称保持原样；隐私另核 |
| 遗留名称未核验 | 1 | 军师架势效果没有自身完整英文名称来源，本轮保持旧名 |

主要来源为 [原生双客户端报告](C:/Users/Taka/Desktop/fvtt/output/automation-eldamon-ux-20260930/qa/dual-client-report-1790743642388.json)，SHA-256 `79f75911836506d4187f5c036cf627d633bb551f96198a8d0c61c566688e116b`，Core 14.368 / PF2e 8.5.1 / Workbench 7.7.5 / 语言 cn。报告内模块版本仍为开发中的 0.9.19；版本号本身不证明最终 0.9.20 发布字节。各槽的匹配值、卡 ID、命令 ID 与指纹保存在 [字段对账](C:/Users/Taka/.codex/worktrees/automation-native-20260930/fvtt/docs/fortress-automation/rk-localization-2026-09-30.json)。

- `rk-cn-normal-no-target`：一个 1d20，点数 11，七个技能候选。
- `rk-cn-normal-target`：一个 1d20，点数 12，自动社群一项，总值 25；DC 20/22/25/30 显示成功/成功/成功/失败。学识难度表的第 5/6 次只证明提示文字，空学识技能表只证明表头。
- `rk-cn-no-target`：实际 assurance=true；固定 10 + 原生熟练 9 = 19，零骰。不能按命令名归为普通检定。
- `rk-cn-player-privacy`：实际 renderHTML 的三张卡均 isContentVisible=false；玩家只见 Core 秘骰占位，目标真名、DC、degree、结果和候选技能均不可见。占位中的英文 privately rolled some dice / ago 来自 Core，非 RK adapter 生产槽；未宣称该外部 UI 已汉化。

当前四个 RK 脚本与 QA/runtime 物理副本的 raw SHA-256 全部逐字相同；源清单与实际 Workbench 命令 SHA 一致，原宏与 fixture 没有修改。历史报告没有存四脚本的浏览器 sourceSHA，故当前物理副本比较与报告 HTML 证据分开登记。最终发布使用 Git blob 和候选清单绑定，不能声称该历史报告抓取了浏览器源字节。

首次 `rk-cn-setup` 因旧 QA fixture ID 不匹配失败，纠正克隆身份后才执行 RK。这是 harness 设置失败，不计产品成功或产品回归。RK 报告 pageErrors 为空。原宏未闭合 rank div、多余引号、Map.length 与未配对链接结束标签作为既有源问题保留；Core 保存后的 HTML 规范化不作为 adapter 修改属性的证据。
