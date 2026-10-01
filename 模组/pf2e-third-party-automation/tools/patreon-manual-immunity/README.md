# Patreon 原生手动治疗免疫观察

仅支持 PF2e 8.5.1、Patreon 3.2.29 的固定本地源码。记录器观察原 `Ma → p` 的免疫创建，不重新掷骰、应用 HP、创建免疫或修改原有 start、duration、wounded 处理。

接受的 Patreon 输入 SHA-256：

- 原包：`89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9`
- 已验证的 v5 时间补丁：`d29a3878f9521cc185eb295288fe660b4c41a86a0a07fcc966423e887b15de22`
- PF2e bundle：`d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157`

`patch.mjs` 接受 v5 输入，或接受原包加既有 v5 patcher。原包必须先得到完全相同的 v5 字节。本补丁保留时间 handler 和接口字节。完整输入、补丁输出和 manifest 写到新建的私有输出目录；整个 fvtt checkout 均禁止作为付费源码输出位置，付费源码不得进入 Git 或发行 ZIP。

```powershell
node 模组/pf2e-third-party-automation/tools/patreon-manual-immunity/patch.mjs <v5-input.js> <private-new-output-directory>
node 模组/pf2e-third-party-automation/tools/patreon-manual-immunity/patch.mjs <original-input.js> <private-new-output-directory> <existing-v5-patcher.mjs>
```

原免疫创建客户端的 `createEmbeddedDocuments` Promise 才提供 terminal。玩家拥有患者 OWNER 时，原创建仍在玩家客户端执行：本地 observer 收到 terminal 后沿现有认证 owner 传输送 active GM，GM 核实 caller 是实际 creator、双方角色权限、同次 check/useId、当前 recording 会话、唯一原生 Item、来源和原始时间，再保存 proof。玩家不能读取 GM 私有账本。ACK 未知只读取同一已存 proof，不重新提交创建或重放治疗。

玩家没有患者写权时，保留原 Patreon socket 代建分支。GM 在原 `p` 分支等待真实创建 Promise，记录自己作为 creator；原 emit 返回不是免疫完成。两路都保存原 `MessageForHandling` 的唯一目标快照。有真实 check target 时核对它；空 target 时仅接受该原生调用的唯一患者关联，GM 不读取自己的 targets。

公共 Patreon API 只提供 `explorationManualImmunity.descriptor/subscribe`。普通 Item/check flags 不会触发 terminal。跨客户端接受的是认证 owner 的本地终态陈述，与已有原生 owner 回执采用相同信任边界；它不证明任意恶意客户端 JS 的调用历史，也不是密码学证明。

首批限单患者、非共享 HP 池、普通手动治疗。Risky Surgery、多张或被删除的来源卡、失权、未知版本、外部时间变化、缺少真实 terminal 或原 Patreon 没有创建免疫时保持缺失证据。元数据失败保留原治疗行为。

离线 seam 测试读取私有固定源码，不把源码复制到测试仓库：

```powershell
$env:PATREON_MANUAL_SOURCE='<verified-v5-input.js>'
$env:PF2E_MANUAL_SOURCE='<fixed-pf2e-8.5.1-bundle.js>'
node --test 模组/pf2e-third-party-automation/tools/patreon-manual-immunity/seam.test.mjs
```
