# 元素化身续接与中文显示 0.9.20 交付

0.9.20 已于 2026-09-30 14:25（Asia/Shanghai）安装到 CN 主实例。GM 与玩家整页刷新后加载新脚本。

发行：[pf2e-third-party-automation-v0.9.20](https://github.com/takaqiao/fvttAutoTranslateTool/releases/tag/pf2e-third-party-automation-v0.9.20)。运行源码与 tag 指向 `a2dc695bcb8cd6bdda630587c0f3990147b5bf34`，基线为 `f45338970afec79e956f3685175758c784f9e098`，前次发行来源为 `f22980350ec176a7b6df69bb68d5062733b7d484`。没有合并其他并行翻译或 Sundry 分支；原脏工作区与其他 QA 实例未改。

## 行为

- 高电压从当前真实命中卡及角色动作区继续同一原活动，不重新 Use 或付款。目标拥有者进行原生反射豁免，保留重掷窗口，再由施法者明确投伤害并经原生应用完成。删除原卡后，续接入口关闭并显示中文来源变化提示。
- 元素操控在真实 Use 时捕获 T 目标并施加带电；展示卡不结算。护盾保留原生 +1 情势 AC，当前未命中卡可声明实际护盾贡献；新旧入口及原生保留重掷沿同一攻击身份防重复。
- 电链从真实伤害回执进入拥有的原生威能 Use，保留目标、分支、准备、反应和支付。只忽略实际证实的 Die 动画字段；公式、骰点、类型、目标、原卡和应用回执仍校验。
- 待用增广/虹吸可在角色动作区明确清除，沿原 activation nonce 核验，处理中不能清除。伤害收尾与恢复查询改用一次初始化、沿文档事件更新的 APPLY nonce 索引。
- 普通 RK 沿 Workbench 同一个 d20 比较知识技能，T 目标匹配适用技能；固定技能与驾轻就熟沿原规则。相关自有显示按 v3 对账，秘骰内容不向玩家公开。不增加移动/距离监听或几何准入判断。

## 验证与边界

完整实际来源测试 **2008/2008**，fail/cancelled/skipped/todo 均为 0；运行脚本语法与 staged diff 检查通过。独立集成审查无剩余 P1/P2，相关 260/260 用例属于完整套件的子集，不相加充当额外覆盖。

实际公开 ZIP 安装在自有隔离实例 `127.0.0.1:30426`：Foundry14.368 / PF2e8.5.1 / Eldamon1.5.1 / Workbench7.7.5 / Toolbelt3.56.5 / HUD2.55.2 / Patreon3.2.29 / Trigger Engine1.35.1，启用四项实际中文依赖。独立 GM/玩家浏览器最终报告有 13 个成功命令、0 失败命令、0 pageerror，SHA256 `a04193d8e6f74f1215e7f2a39adba66fc6000c506e098a634fae55a0a36970fa`。报告为任务输出 `qa/dual-client-report-1790748695576.json`。

- 公开包的普通无 T RK：骰点15、七知识技能；T：骰点2、社群15，DC20/22失败、25/30大失败；固定驾轻就熟19、零骰。三张秘骰在玩家真实 DOM 均保密。当前列名为“熟练”；历史报告的“熟练度”不冒充当前显示。
- 原生电链伤害入口生成27点电击，原生全额应用使 QA 目标 HP146→119；一个实际应用回执 `JkukaHRinZf6GpJK` 经新索引成为 confirmed/lifecycleDone。此为伤害应用与收尾验证，不宣称覆盖电链所有豁免结果。
- 新包复核删原卡后的玩家动作区、原护盾 MISS 重放效果不刷新、外来玩家清除待用状态中文拒绝。较早的高电压 Use→当前命中→玩家豁免→原生保留重掷→GM伤害→玩家全额应用，以及操控/护盾真实 Use 的充分证据继续按来源关系复用。

工作区与 Git blob/ZIP 逐文件对账：151 项中147项原始字节相等；`knowledge-automatic.mjs`、`knowledge-automation.mjs`、`knowledge-entrypoints.mjs`、`knowledge-workbench.mjs` 四项只有 CRLF→LF，已记录双方 SHA。公开 ZIP 与打包 Git blob 原始字节一致，独立 manifest 与 ZIP 内 manifest 一致，真实 latest 安装地址指向同一清单。浏览器实际服务151项及远端 HTTP151项都与公开文件清单相等。

v3 原文/SHA/逐项消费入口与独立审校分别保存在 voltage、electricity、metapower、RK 四份 `*-localization-2026-09-30.json`。字段统计各自保留口径：电击20旧译、5既有中文调整、35新字段；超威能79既有字段、4新可见生产者；高电压及 RK 的当前/历史实际覆盖分别入表。单独错误/通知分支未实触发的仍为未验，2008个断言不替代字段覆盖。自定义世界名称未覆盖，“军师架势：下一次攻击”的真实英文源未核验，未编造英文尾名。其他元素包、全部伤害/IWR分支、长时多用户负载及实际 FPS/延迟没有宣称验证。

早期 QA 的错误 UUID、选择器、错误导出探针、护盾 +2 错误假设及原生 Combat 夹具问题保留在旧报告中并分类，不清除失败来判通过。

## 发行与安装凭据

| 对象 | 结果 / SHA256 |
|---|---|
| ZIP | `14a7029fe75849471336835d681d1202a073e3ce326f7a81b844a787570f79d7` |
| manifest（tag / ZIP 内 / latest） | `c8d28aeabb81ee2acb6541c6331c6c9133251b54540bad69072695769a3168c6` |
| 源码/附件集合 | 151运行文件；对 .19 改19、增3、删0；QA、测试、备份、许可证、第三方 raw 包不进入附件。 |
| CI | 仓库没有 `.github/workflows`，沿已有手工发行方式；不声称 CI 通过。 |
| 完整测试日志 | `bd37616a4a99597de7a24a9041c01ded8a332b757ce6aef592d78ae0e99129f6` |
| 公开下载证明 | `45d22ecfc2f3e61de414937044af38422583c9b7de93ac69f8dec118dac8958a` |
| frozen plan | `db36fb427fea54265f88ed37fcbe1aa980a57ad4cedc63bbe4b35a97a233880f` |
| 安装回执 | installed；`186073b32cf6bfb58820641717ec6a6375f4c3f9a6a2aa3bdb92dbbe928c8b4b` |
| 安装后核验 | `7fd469bd85fba31633ed5d1ea4f4c5885db51087c20cbc48b1e217139360bf3e` |

新基线确认主实例 Setup、world=null、ready=false、零连接用户、模块未锁；inspect 通过后执行同步原生安装事务，再逐文件检查实际 HTTP 字节。原模块 `.19` 在 `/root/fvtt-patch-backups/automation-release-0920-20260930/backup-pf2e-third-party-automation-0.9.19` 原样备份。

主 PID1621366、PM2 id2/id4 的进程/启动定义、options、170个完整世界文件和八项受保护模块均保持原样，没有重启生产服务或覆盖世界 Actor/Item。自己的30426浏览器和 PID55004已关闭，其他实例未触碰。

完整外部凭据保存在 `C:/Users/Taka/Desktop/fvtt/output/automation-eldamon-ux-20260930/release-tools/release/`，原始 QA/输入和私有运行副本留在任务自有输出，未加入 Git。隔离工作树及已推送 `codex/automation-native-20260930` 分支保留供继续维护。
