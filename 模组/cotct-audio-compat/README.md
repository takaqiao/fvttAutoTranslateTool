# CoTCT 定向音频停止兼容

为已有 CoTCT 播放器补充 Soundscape Adventure 6.1 的定向音效停止支持。旧播放器 STOP 和 Full Moods STOP 会停止旧受管音源向指定玩家发送的声音；普通切换音乐继续保留这些声音。

适配器记录 Soundscape 发出的播放 key、原始目标玩家和精确音源路径，只使用现有 guard 目录中的路径。它使用上游的停止消息，不遍历或停止全局声音。实际接收玩家以自己的身份回报结束，使 Soundscape 的发送按钮恢复可用。Full Moods 自有音乐与音景仍由原控制器管理。

该包是代码补丁工具，不是可直接安装的 Foundry 模组。它不包含音源、素材目录、私有合集或世界数据。需要已有 `cotct-music-guard 0.1.1`、`cotct-full-moods 0.3.1` 和匹配的旧播放器运行时；构建后分别为 `0.1.2`、`0.3.2`。补丁制作时校验源码接缝，运行时不使用 Soundscape 版本或整函数哈希白名单。

## 制作补丁

使用 Node.js，输入已有 Foundry Data 的离线副本和一个尚不存在的输出目录：

```sh
node tools/build-candidate.mjs /path/to/Data /path/to/new-output
```

输出 `Data/` 中仅包含六个变更文件，`manifest.json` 记录各文件的修改前后 SHA-256。先核对清单、备份原文件，再按清单更新；不能用这个稀疏目录替换整个模块。新 `targeted-audio.mjs` 应先于引用它的 `bootstrap.mjs` 安装，两个 `module.json` 最后更新。构建命令本身不连接或修改服务器。

所有 GM 和目标玩家需要刷新后再开始新的定向播放。观察器不会恢复刷新前遗失的播放 key；这类旧播放仍用 Soundscape 原停止操作处理。断线后的历史也不通过全局强停补救。

## 验证

```sh
npm test
```

测试覆盖多客户端消息范围、真实发送者校验、原始目标列表、自然结束、停止回报、发送失败和会话失效。真实音频还需在隔离 Foundry 中用 GM 与两名玩家验证：一次和循环播放、两条 STOP 入口、按钮恢复、加载中停止，以及不属于此范围的声音继续播放。Node 测试不等同于多人音频验收。

接口依据：[Soundscape PlayerAudio](https://github.com/piecesofcodecom/soundscape-adventure/blob/v6.1.0/scripts/utils/player-audio.mjs)、[Socket.IO 出站监听](https://socket.io/docs/v4/client-api/#socketonanyoutgoingcallback)。
