# Toolbelt 共享 HP 手动完成接口

此工具为 Toolbelt 3.56.5 的原主人 HP 更新添加完成观察接口，供探索手动记录器使用。它捕获本地 OWNER 或原 active GM 转发调用返回的更新 Promise，再与发起端的原生应用回执关联。患者更新返回、Socket emit 和两角色 HP 相等均不能作为完成证明。

输入是 Toolbelt 的 `scripts/main.js`，SHA256 必须为：

```
2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f
```

在本模组目录执行：

```sh
node tools/toolbelt-manual-pool/patch.mjs /path/to/pf2e-toolbelt/scripts/main.js /private/output/toolbelt-manual-pool
```

输出目录须位于本仓库之外且尚不存在。目录中保存原文件、候选文件和 SHA 清单；安装时使用已审查的 `patched-main.js`。完整 Toolbelt 文件保存在私有输出目录，公共源码包只包含补丁工具与观察器。

候选文件在 `ready` 后提供 `game.modules.get('pf2e-toolbelt').api.explorationManualPool`，公开接口为固定 descriptor 和 `subscribe`。探索运行时负责真实来源登记、认证授权、血池互斥和账本确认。接口本身只观察原调用；需要配套运行时接线后才能用于手动探索治疗。

原 `shareData` 关系必须可确认。配套运行时要求 descriptor 的 `hpBaselineGuardVersion:1`，在原主人更新前核对本次治疗的血量输入；旧观察器不具备此检查，不能继续应用。来源、操作者、HP 字段或最后完成边界不匹配时，该次应用保留未知状态。结果查证只读取已保存证据，未知请求不重新执行主人更新。此客户端检查不提供服务器并发写入的比较交换保证。

Core 的原更新 Promise 在没有字段差异或取消更新等情况下可能返回 `undefined`。仅当请求字段在调用前已分别等于原始与准备后 HP、完整原始 HP 和准备后的 value/max/temp/sp（含字段是否存在）在完成后保持不变，且来源、操作者及权限仍有效时，观察器才返回带 `updateOutcome:'unchanged'` 的 fulfilled 终态。这表示原调用已完成且没有待保存的 HP 差异，不证明服务器发生了写入，不替换原生应用回执，也不把应用结果改为 `noChange`；其他空返回或未知结果仍拒绝。

离线接缝测试使用同一固定 Toolbelt 源文件：

```sh
TOOLBELT_MANUAL_SOURCE=/path/to/pf2e-toolbelt/scripts/main.js node --test tools/toolbelt-manual-pool/seam.test.mjs
```

这些测试提取原 Toolbelt 调用链；真实 GM 与玩家客户端验收另行执行。
