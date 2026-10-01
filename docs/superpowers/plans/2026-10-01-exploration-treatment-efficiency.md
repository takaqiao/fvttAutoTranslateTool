# 治疗效率选择：可证明的 prepared 情境

沿已批准设计收尾 Risky Surgery、Assurance 和情境修正的 DC 选择。原生治疗仍负责所有骰点、割伤和 HP 应用；估算不得调用 Roll、Check.roll、beforeRoll 或原生行动。本项不改 Stage1 的执行权和时间提交协议。

## 接口与范围

capabilities 由 root 单点修改。每个 treatmentEstimate[skill] 保留既有普通无条件字段，同时新增 selections 数组。每项由本次真实 prepared 原生技能上下文准备：

```js
{
  riskySurgery: false, assurance: false, ready: true,
  sourceVersion: '8.5.1', modifier: 0,
  outcomesByRank: [
    {rank: 'trained', dc: 15, cases: [{weight: 0.05, outcome: 2}]}
  ],
  source: {actorUUID, skill, ruleSources: []},
  reason: null
}
```

outcome 为 0 大失败、1 失败、2 成功、3 大成功。普通检定恰好二十个 weight=1/20 情形；Assurance 恰好一个 weight=1。DC 仅为 15/20/30/40，档位始终受所选技能 rank 限制。来源和未知原因保留；不能把缺失或 malformed 表格当成可估算。

root 按固定 PF2e 8.5.1 的公开 CheckModifier 和静态核验成功度逻辑准备结果，不依赖运行时私有类名。每个 selection 对原 modifier 做副本，按真实 predicate 和 stacking 得到 totalModifier。Assurance 只使用实际技能的已选择 substitution 及 prepared proficiency；Risky 成功提升和环境加值按原规则仅应用一次。任何需要有副作用 beforeRoll、未知 degree/predicate、fortune/required substitution 冲突、Magic Hands/Mortal Healing 等情境保留具体 unavailable 原因，回落合法固定档，不冒称最优。

efficiency/policy 由 P 作者独占。估算只消费以上 outcomes 与 patient 的 hp、healingExpectationReady、damageExpectationReady。普通无条件旧上下文保持兼容。Risky 要求 Medicine、真实 feat 资格、可证明的普通伤害模型（无 temp/IWR）、患者当前 HP 至少 17 且施治者不共享其 HP 池；未证明自疗后行动能力和复杂患者损伤先回落。不以平均 4.5 推断失能。

## 有效收益

逐结果枚举真实 1d8 割伤、2d8/4d8 治疗和大失败 1d8 的完整离散分布。先割伤，再本次治疗/大失败，HP 截至 maxHP。收益为 min(targetHP, afterHP)-min(targetHP, beforeHP)，目标由当前 deficit 与 hp 推出。割伤增加治疗空间，不能用 min(originalDeficit, heal)-4.5。每个不同 patient/pool 各付一次割伤；相同 pool 不重复计收益。Medic 仅在既有无 receiving 修正资格下附加原生档位加值。

返回 expectedNetHealing、expectedHPDamage 和 estimate/source 摘要。policy 只按净收益排序，不再扣一次损伤；expectedDamage 用已计算实际损伤作平局和说明。固定用户 DC 与不完整模型仍可执行，但标未估算。精确估算只是当前可执行候选的期望收益，不称未来随机疗程全局最优。

## 验证与集成

P 先写失败测试，再实现纯效率与 proposal 摘要；root 独立实现 prepared 投影并写真实源语义对应测试。覆盖合法档位、四结果、二十面与 Assurance 单项、环境 stacking、技能匹配 Assurance、不加等级熟练、Risky 被压过仍付代价、溢出与新增缺口、每患者代价、相同池、复杂模型/自疗 fallback、畸形 outcomes、零随机调用。用固定原生纯成功度源码做离线 parity；独立审查后才生成下一冻结候选并进入实际玩家矩阵。实际治疗输出永远优先，不重骰或补 HP。
