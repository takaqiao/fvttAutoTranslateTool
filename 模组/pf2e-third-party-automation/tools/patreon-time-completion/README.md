# Patreon 时间完成回执补丁

适配 Patreon 3.2.29 的探索快速治疗与效果计数。补丁等待原生骰点及文档更新完成，保留原生治疗算法。这里只提供补丁工具；不包含付费模组源码。

只接受以下准确来源：

| 文件 | SHA256 |
| --- | --- |
| Patreon `src/index.js` | `89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9` |
| Patreon `module.json` | `1751242ef5858ee55bb33a79194ddbce5b2109f35e89d4d64df0bf4cb6fc012a` |
| 补丁后的 `src/index.js` | `d29a3878f9521cc185eb295288fe660b4c41a86a0a07fcc966423e887b15de22` |

先将 Foundry 返回 Setup 并关闭世界客户端。准备已安装的 Patreon 路径和本工具目录，使用 Node.js 执行：

```text
node <工具目录>/patch-patreon-v5.mjs <Patreon目录>/src/index.js <Patreon目录>/module.json <工具目录>/prepared-3.2.29
```

输出目录必须是工具目录下尚不存在的直接子目录。工具保存原文件、补丁副本、差异及哈希清单，不修改已安装文件。确认输出 `patched-index.js` 的哈希与上表一致后，将它复制为 Patreon 的 `src/index.js`；保留原文件和 `original-module.json` 备份。不要替换 Patreon 的 manifest。

启用 Patreon 的时间快速治疗设置；再聚能还需要启用 PF2e Workbench。GM 与玩家整页刷新后加载。若更新 Patreon 或恢复原文件，接口会消失，探索恢复会暂停；为新来源重新核验后才能继续。已有未知结果保持原样，不通过重放时间或治疗补回。

回退时在 Setup 将备份的 `original-index.js` 复制回原 `src/index.js`，再整页刷新客户端。此工具不能对未知来源、不同版本或已打补丁文件重复打补丁。
