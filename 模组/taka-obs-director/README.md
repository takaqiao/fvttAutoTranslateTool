# Taka OBS Director

为指定录制账号提供 PF2e 地图、角色、战斗焦点和收到的剧情图片展示。其他账号保留原生界面。角色展示沿用本人 Avatar、已批准立绘和既有肖像校准。

在 Foundry 模组安装器中填写：

```text
https://github.com/takaqiao/fvttAutoTranslateTool/releases/download/taka-obs-director-v1.0.3/module.json
```

Foundry VTT 最低版本为 14；需要 PF2e、OBS Utils、LiveKit AVClient 和 libWrapper。启用后，在模组设置中指定录制账号。只在该账号的 Director 生效时暂停 OBS Utils 的摄像机广播（`canvasPan` 上名为 `socketCanvas` 的监听）；其余 pan 回调、接收跟随和停止后的恢复继续工作。找不到这个监听或有多个时保持原样。OBS Utils 5.3.1 起，录制账号进场景时和 OBS 端刷新时各会单独广播一次视角，这两次不经过 `canvasPan`，不在暂停范围内。

安装包只含模组运行文件和现有美术。OBS 场景、配置、录制账号设置和世界数据由各自环境维护。

源码测试：`npm ci` 后运行 `npm test`。旧交付工具的测试需要本地 OBS 配置；这些配置不进入公开源码或安装包。

[更新记录](CHANGELOG.md)
