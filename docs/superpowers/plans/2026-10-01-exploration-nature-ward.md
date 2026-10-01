# 探索收尾：自然医疗群体提案

已批准设计要求替代检定技能不扩大看护师容量。本项复用现有治疗引擎，只补policy遗漏的Nature群体提案。效率模型、特殊聚能、有限资源和手动检查点分别实施。

## 行为契约

- 延用当前Natural Medicine来源／Nature受训资格、活体患者和提供者begin检查。没有自然医疗或对应技能资格不产生Nature提案。
- 同一候选的所有患者在当前时刻可开始，按当前目标缺口稳定排序；每个当前ready HP池最多选择一名患者。看护师容量读取prepared Medicine的wardCapacity，不由Nature熟练度推导。
- Nature rank继续限制本次8.5.1原生适配可用DC；对应技能Assurance单独选择，Risky保持关闭。Continual Recovery和已有免疫来源沿既有规则；后来持续恢复者不抹去旧免疫。
- 复用selectTreatmentRank、600秒duration、既有patient／pool互斥和唯一原生完成路径。保留单体候选；群体净收益按原估算边界计算，不宣称复杂情境已优化。
- 荒野环境加值、房规容量、Magic Hands／Mortal Healing和Occultism普通治疗不因新增Nature组自动开放。新增QA Natural Medicine构筑必须明确标记，不能称原名册角色已有。

## 唯一文件归属与步骤

实施人只拥有scripts/exploration/policy.mjs及对应policy测试；必要增加nature-ward-policy.test.mjs。不得修改capabilities、treatment、native-treatment、hp-pool、runtime、coordinator、owner协议或UI。

1. 保存相关源，先写RED：Nature传奇／Medicine受训只有单体；Medicine专家／大师／传奇分别2／4／8；无Ward／无Natural／Nature未受训；同pool去重；旧冷却；对应技能Assurance与Risky关闭；DC及600秒字段。
2. 窄补群体提案，复用Medicine组的当前资格、容量、pool去重和排序。不要复制一套治疗／应用实现。
3. 定向policy和既有capabilities／treatment回归、语法与diff检查，保存源SHA及准确结果。独立审查后由root统一集成；本项不单独提交或安装。

源码测试证明提案形状与规则约束，不能替代最终实际Nature／Ward原生多患者检定、HP／免疫及时间证据。完整Stage2矩阵仍须从公开入口验证本项。
