# 0.9.19 原生入口与回忆知识补全

以 CN 正式安装的 0.9.18.6 为基线，保留所有已部署热修。用户确认不需要移动、路线、距离、视线、触碰事实检测及额外完成确认；本轮据此直接衔接真实原生 Use/检定/效果回执。

## 行为

- 角色卡、热栏、HUD、原生回忆知识动作、Workbench 宏及现有能力附带的回忆知识使用同一桥接。一次原始骰点供所有候选技能比较；唯一 T 目标按特征匹配技能，最高适用修正用于唯一机械收益。Lore 及无/多目标的信息裁定由 GM 沿用原骰，不要求玩家猜技能。固定技能及 Assurance 保留能力要求。
- 组合活动的攻击与武器/法术伤害在原操作者客户端调用原生 API，采用其窗口偏好。首次攻击取消不支付；接受时先付款再投骰。已付款法术保存唯一原生卡，未命中及后续取消也能继续处理。实际保存的文档可解除 socket 回复丢失后的等待，不以短时 UI 超时重试。
- 原生英雄点重掷保留的新检定会在各客户端立即使旧组合伤害失效，GM 再保存来源接续和待处理状态。已应用伤害沿原生撤销；GM 依据保留结果核对法术倍伤、持续伤害及精确伤害，再合并应用一次。本版不自动重投复杂合伤。
- 激发武器只在指定武器完成真实攻击后消费，命中、未命中及无 AC 检定均覆盖。延后伤害使用该次快照，不借用/删除后来新施加的效果。重准备的原生 Strike 通过对象身份保证包装幂等，避免无窗口付款后等待自身队列。
- 公共名称依据实际接收者权限显示；秘密检定的 DC、目标信息和结果留在 GM 秘骰卡。关闭或重绘角色卡释放旧监听。使用次数凭据绑定真实操作和原卡，不因五秒延迟失效。
- 精神伤痕维护改用相关效果与伤害回执索引；普通聊天不读取全世界库存。相关维护按角色与字段合并，真实行动逐次执行；普通 Token 移动和外观变更不再触发全队维护。

Assurance 的固定结果和 Fortune／Misfortune 抵消依据[现行专长规则](https://2e.aonprd.com/Feats.aspx?ID=5121&Redirected=1)。本版使用实际准备的熟练值，保留无等级熟练变体；无明确 DC 的 Lore 和不适用目标特征的固定普通技能交给 GM 沿原骰裁定。

## 验收范围

最终自动测试、实机双客户端覆盖、发行哈希及部署回执记录于[执行记录](native-followups-2026-09-30.md)。测试世界使用独立数据目录、普通玩家与 GM，Foundry 14.368 / PF2e 8.5.1、Workbench 7.7.5、HUD 2.55.2、Toolbelt 3.56.5、Patreon 3.2.29、Trigger Engine 1.35.1 与正式环境相同；隔离世界的 Dailies 为 4.20.0，正式环境为 4.20.1，不能据此宣称 Dailies 的所有现场分支均已验证。测试不接触正式世界 HP、物品、宏或活动设置。

最终完整回归为 1,921/1,921，零失败、零跳过，运行脚本语法和差异检查通过。实际双客户端证明一次检定共用保留骰点、隐藏目标自动匹配 Society 且玩家看不到真名/DC、原生幸运和替代骰各消费一次。Assurance 实机固定为 10+9、零骰，未参与的 if-enabled 加值保留；厄运抵消后为一次 1d20+16，完整保留 INT 4、熟练 9 与加值 3。原操作者付款、Surge 未命中与延迟伤害、玩家英雄点重掷后的两端旧伤害拦截均有单独证据。

以下保留此前接受的手工边界：通用外部合伤的逐来源 IWR、法袭箭存储/豁免/持续伤害、一结果型复杂反应、召唤师跨模块共享状态。详见 [独立评估](bounded-followup-scope-2026-09-30.md)。此前停止的威能指环与天际雷球项目保持停止。

运行计数与索引回归证明工作量下降，不代表实测 FPS、网络延迟或 LiveKit 长时负载改善。浏览器 QA 的范围应按实际结果阅读，不视为所有能力的全流程 HP/IWR 验收。

## 更新与恢复

所有 GM 与玩家整页刷新后加载新脚本。发行包只含模块运行文件；已有世界数据不随包覆盖。部署前保存完整旧模块备份并校验所有文件，失败时恢复旧目录及原生包缓存；不自动关闭世界、不重启正式服务。

2026-09-30 11:46（Asia/Shanghai）已在 CN Setup 状态部署，原生缓存为 0.9.19，服务器 HTTP 提供的 148 个文件全部匹配发行清单；170 个世界文件、八个受保护模块、服务进程与启动配置保持一致。完整旧模块备份位于 CN 的 `/root/fvtt-patch-backups/automation-release-0919-20260930/backup-pf2e-third-party-automation-0.9.18.6`。

[正式发行](https://github.com/takaqiao/fvttAutoTranslateTool/releases/tag/pf2e-third-party-automation-v0.9.19)的源提交为 `f22980350ec176a7b6df69bb68d5062733b7d484`。ZIP SHA256 为 `50b6bfd1b9ee842d973e7c838a5156d848eb6fd481338ba8fb1f8ed02f533b80`；公开下载包和 latest manifest 已重新下载并验证。最终隔离双客户端与发行包 148/148 文件 SHA 相符，随后只停止本轮自有测试服务。

本地证据：[公开下载校验](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/release-tools/release/public-download-verification.json)、[发行包实机报告](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/qa/dual-client-report-1790739641161.json)、[部署回执](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/release-tools/deployment-evidence/deployment-receipt.json)、[部署后核对](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/release-tools/deployment-evidence/postcheck.json)。这些只证明本轮列出的范围；测试世界已保留以便后续复查。
