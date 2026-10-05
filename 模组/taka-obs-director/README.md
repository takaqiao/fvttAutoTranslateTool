# Taka OBS Director

为指定录制账号提供 PF2e 地图、角色、战斗焦点和收到的剧情图片展示。其他账号保留原生界面。角色展示沿用本人 Avatar、已批准立绘和既有肖像校准。

在 Foundry 模组安装器中填写：

```text
https://github.com/takaqiao/fvttAutoTranslateTool/releases/download/taka-obs-director-v1.0.1/module.json
```

需要 Foundry VTT 14、PF2e、OBS Utils、LiveKit AVClient 和 libWrapper。启用后，在模组设置中指定录制账号。只在该账号的 Director 生效时暂停已核验的 OBS Utils 摄像机广播；其余 pan 回调、接收跟随和停止后的恢复继续工作。未知 sender 或多个匹配 listener 保持原样。

安装包只含模组运行文件和现有美术。OBS 场景、配置、录制账号设置和世界数据由各自环境维护。

源码测试：`npm ci` 后运行 `npm test`。旧交付工具的测试需要本地 OBS 配置；这些配置不进入公开源码或安装包。

[更新记录](CHANGELOG.md)
