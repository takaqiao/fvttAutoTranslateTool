# 本机双实例端到端测试

`harness.py` 在本机起两个真实的 Foundry 14：主服（端口 30181，团世界 `e2e-camp`）和建卡服（端口 30182，世界 `pf2creation`），由 Foundry 自己写出库，再在上面跑引擎和浏览器端场景。数据目录建在系统临时目录下（`charsync-e2e-*`），测试结束时停掉两个实例并整个删除，测试失败也一样。

## 运行条件

- Windows，本机装有 Foundry 14。程序目录默认 `C:/Program Files/Foundry Virtual Tabletop/resources/app`，装在别处时设 `CHARSYNC_FOUNDRY_APP`。
- 许可证从 `%LOCALAPPDATA%\FoundryVTT\Config\license.json` 拷进两个数据目录，两个实例共用它；本机的 Foundry 数据目录不会被改动。
- `node` 在 PATH 上（Foundry 和引擎都用它）。
- Python 3.12 以上，装好 `playwright` 和它的 Chromium（`python -m playwright install chromium`）。
- 端口 30181、30182 空着。
- 两个环境变量。不设时从 git 推出主检出，取其中的 `output/charsync-20261016/e2e-cache/systems/pf2e` 和 `output/charsync-20261016/fixtures`；推不出来或目录里缺文件时报错，并点名要设哪个变量：
  - `CHARSYNC_PF2E_DIR`：pf2e 系统目录（含 `system.json`）。以目录联接挂进两边的 `Data/systems/pf2e`，删数据目录时先拆联接，不会动到它。
  - `CHARSYNC_FIXTURES`：夹具目录，里面的 `cards.json` 形如 `{exportedAt, actors: {<id>: {world, doc}}}`，`doc` 是导出形状（`items`、`effects` 和条目的 `effects` 都是文档数组）。夹具是真实的卡，不进仓库。

## 运行

```sh
python -X utf8 test_engine_real_db.py
# 提交、主服套用、卡面交互；测试类内部有顺序依赖，运行完整类
python -B -X utf8 -m unittest -v test_submit test_apply test_sync_ux
```

要排查时设 `CHARSYNC_E2E_KEEP=1`：实例照样停掉，数据目录保留并把路径打印到 stderr（两个实例的日志是其中的 `main.log`、`char.log`）。pf2e 联接照样拆掉，保留下来的目录可以直接手删。

测试台只终止本次 `Env` 启动并持有的进程。端口 30181、30182 上有其他进程时直接报错，不终止未知进程。各套测试共用固定端口，必须串行运行。

`test_sync_ux.py` 验证同级修改经 GM 套用、折叠明细、只改魔宠、新主人与魔宠首次导入，以及撤回后拒绝旧操作。截图默认保存在测试数据目录的 `screenshots/`，可通过 `CHARSYNC_E2E_ARTIFACTS` 指定外部目录；需人工检查截图后才能认定布局验收完成。

`python -B -X utf8 -m unittest -v test_harness_lifecycle` 不启动实例，用模拟进程检查测试台只终止自身进程。

## 测试台

`with Env(root, fixtures) as env:` 退出时关浏览器、停实例、删 `root`，某一步出错也会接着做完后面的。`build()` 只在不存在或空的 `root` 里建，没建过的 `root` 不删。

| 方法 | 做什么 |
|---|---|
| `build()` | 建 `root/main`、`root/char` 两个数据目录（许可证、pf2e 联接、从工作树拷一份模组、手写 `world.json`），写 `root/engine.json` 和 `root/seed.json` |
| `start(which)` | `which` 为 `main` 或 `char`。起实例并直接进世界，等到 `/api/status` 报告 `active` |
| `stop(which)` | 停止本次测试持有的实例进程。主服随后以不开世界的状态重新起来、状态接口报 `active: false`，和线上主服核心常驻一样；引擎连不上主服状态接口时不读团世界 |
| `seed()` | 见下；结束时改写 `root/seed.json` 的 `bootstrapSince` |
| `engine_once()` | 跑一轮 `engine/main.mjs --once`，返回解析后的 TickReport |
| `login(which, user)` | 新开一个浏览器上下文登录，等到 `game.ready`，视口 1600×900，没有指定角色的玩家先关掉用户配置窗。返回 `(page, logs)` |
| `inject_invalid_item(which, actor_id)` | 世界关着时直接往 `actors` 库加一条 `system.level.value` 为 `"abc"` 的专长，并把它的 id 追加进 actor 行的 `items`，模拟过不了校验的旧数据。返回条目 id |

`seed()` 在两边以 `Gamemaster` 登录：

- 建玩家 `P1`、`P2`，主服另建 GM 账号 `GM2`。种子表不写账号对照，两边同名账号由引擎按名字配对。
- 建卡服先创建归属 `e2e-camp` 的测试角色文件夹，将其 ID 写入 `campaignFolders`；新建测试角色也放入该文件夹。
- 按夹具原 id 建全部卡，所有者只有 `P1`；`P1` 的指定角色是第一张主卡（`env.main_cards[0]`）。
- 启用模组，世界设置 `taka-charsync.publishDir` 两边都写 `charsync-e2e-0000000000`。
- `e2e-camp`：自由变体开、专长分节为族裔典范、`pf2e.campaignFeats` 关、`pf2e.staminaVariant` 开。
- `pf2creation`：自由变体关、渐进属性开、`pf2e.campaignFeats` 开、专长分节为 Team+ 无限制+包的形状；`pf2e.staminaVariant` 不设。另建「已提交」「已撤回」「草稿备份」三个带 `role` 记号的文件夹。

种子表的卡是夹具里除同伴外的每张卡，都归 `e2e-camp`：幻灵按 `pf2e-eidolon-helper` 的 `summoner` 记号认出主人、写进主人的 `companions`；魔宠不进种子，由引擎按 `system.master.id` 认。`bootstrapSince` 在 `seed()` 结束时改写成那一刻，对应线上一次性同步写完卡的时刻：之后只在主服动过的卡，起步基线是 `char-unchanged`，没动过的是 `main-unchanged`。
