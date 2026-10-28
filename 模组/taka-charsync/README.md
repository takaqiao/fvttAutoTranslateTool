# 角色卡同步（taka-charsync）

在主服与建卡服之间同步 PF2e 角色卡的 Foundry VTT 14 模组，加一个常驻的同步引擎。

引擎（Node，pm2 常驻）只读两边 LevelDB 的临时副本，把主服版本、冻结副本、历史版本和映射表发布成静态 JSON，放进两台 Data 下一个不公开的目录（发布目录），并把各团规则表写进建卡服的 `generated/rules.mjs`。模组在建卡服世界里按每张卡的所属团计算规则、跟上主服、提交冻结副本；在团世界里由 activeGM 自动套用，给玩家弹框、面板和撤回。

## 玩家与 GM 的常用操作

1. **同级修改**：玩家在建卡服工作卡上修改构筑或外观，点「提交本级修改」。卡面显示「待 GM 确认」；主服 GM 在同步面板查看变动后点「套用」，更新原卡。同级提交不会自动套用，保留主服经验；当前血量以主服为准，最大血量改变时按差额调整。
2. **升级**：工作卡等级高于主服时，按钮显示「提交 N 级升级」。升级仍按经验、在线玩家和冲突等条件决定自动套用或交给 GM。
3. **查看改动与提交**：卡面默认只显示变更数量，展开「当前草稿」看完整明细；「提交记录」可打开冻结副本或撤回。提交后仍能编辑工作卡，状态条会提示是否又有修改；要把新修改交给主服，需要重新提交。
4. **主人与同伴**：魔宠卡的「主人」下拉选择同团的工作卡；玩家需要同时拥有两张卡的所有者权限。提交副本、草稿备份和归档卡不作为主人候选。卡面列出已识别的主人／同伴，可直接打开对应卡；只改魔宠也可在主人卡点普通提交，新主人首次提交也会带上魔宠。幻灵、追随者等角色卡仍由 GM 在种子表配置关联，见「配置与种子表」。
5. **其他操作**：整卡同步和放弃草稿收在「其他操作」。整卡同步要 GM 确认，会覆盖血量、物品等完整卡片内容，保留主服经验。放弃草稿会先备份工作卡，再恢复成主服版；已经提交的副本仍在。

「刷新状态」只重读状态，不会把主服内容写进当前打开的工作卡。撤回需要等同步引擎发布后主服才会看到，已经套用的结果不会自动撤销。

普通本级修改／升级同步构筑，包括职业、专长、技能、语言、法术与施法列表，以及头像和默认 Token 外观。库存、金币、物品数量、装备状态、条件和已消耗的资源以主服为准。新卡首次导入和明确的整卡同步才会导入来源完整内容；车卡上的测试装备不会随普通升级进入主服。工作卡登录或重新打开时会跟上主服的完整最新内容，主服库存与资源覆盖车卡上的本地改动，同时保留未提交的构筑和外观草稿；打开期间点「刷新状态」仍只读。

头像与默认 Token 分别连同各自的 Tokenizer 图层配方同步；只换 Token 不会自动替换头像。两边分别改头像和 Token 时可以同时保留；同一组两边都改且不一致时，交给 GM 确认。普通提交只更新默认 Token 的外观，保留主服的占格、视野、灯光等设置；已经放到场景中的 Token 保持其现有设置，新拖出的 Token 使用更新后的默认外观。

通过 Tokenizer 2 编辑器保存时，两服都会为头像和 Token 使用独立文件名，保留之前的图片。提交前会按实际内容固定整组角色的本地图片，包括图层、遮罩和物品图片；图片未齐或校验失败时不生成提交。引擎等主人、同伴及其素材全部就绪后才公布该次提交，缺图会自动重试。提交后继续编辑，不会改变已冻结的本地素材。主服发布、套用和工作卡跟随时，也按来源端的实际内容核对图片；两服同名但内容不同的旧图不会被当成同一张图。旧基线无法确定内容、且两边图片不同时，保留冲突供 GM 确认。

GM 的「查看变动」按主人和同伴列出实际套用结果，包含角色字段、物品和效果。套用时重新核对最新已发布的副本；副本换版、已撤回、已处理或任一同伴缺失／版本不符时，整组停止并说明原因。

新卡在面板逐组确认新增或替换，不参与「全部套用」。确认框列出将新建的主人和同伴；替换还会列出将共同归档的旧主人和旧同伴。已有卡「恢复到套用前」会先说明恢复范围，以及需要删除的本次新建同伴。

## 建卡服团规

每张角色卡和魔宠卡按自己的所属团准备数据、显示卡面，GM 和玩家查看同一张卡时使用同一套规则。卡面状态条显示「按某团团规」，不需要手动切团，也不使用浏览器里旧的选团记录或用户的指定角色推断归属。

- 已有工作卡使用引擎发布的卡片映射；冻结副本使用提交时的目标团，草稿备份追溯原卡，魔宠追溯主人。
- 新卡要放进明确配置了所属团的角色文件夹，子文件夹也继承该归属。GM 在引擎配置的 `campaignFolders` 中填写建卡服文件夹 ID 与团世界 ID 的对应关系，文件夹名称不参与判断。已有卡的映射与文件夹归属冲突时也会报错。
- 没有归属、归属冲突或循环、读不到完整规则、规则适配失败时，状态条说明原因，并禁止构筑修改和提交。仍可移动文件夹以修复新卡归属；修复后核对卡面再继续。

规则只在对应 Actor 的同步计算和卡面适配中生效，不写建卡服的公共规则设置。设置窗口、Team+ 初始化和其他没有 Actor 上下文的读取仍使用建卡服原值。当前适配面向 Foundry 14 / PF2e 8.6.1，升级系统或会改动卡面、角色准备流程的模组后，需要重新验证这些适配。

引擎发布完整的 v2 规则快照：`meta.json` 的 `ruleSnapshot` 与 `generated/rules.mjs` 的 `SNAPSHOT` 包含各团规则、内容哈希、卡片和文件夹归属。主服没有保存的规则使用该版本注册的默认值，不借用建卡服的保存值。已打开的客户端在 ready 时和之后每 30 秒读取发布数据；规则或归属变化会重算受影响的卡并刷新已打开的卡面。读取失败时不能继续按旧规则构筑或提交。

提交前重新核对所属团和规则哈希，冻结副本保存 `rulesHash`。规则刚更新时要先检查重新计算后的工作卡，再次提交。新副本缺少哈希、哈希过期或与目标团不符时，引擎不会把它发布为可套用版本，主服也会拒绝套用。启用 v2 之前的无哈希冻结副本保留原有历史和撤回路径，不批量重写；启用时间记录在引擎状态中，不随重启后移。

`scripts/char/rules-override.mjs` 提供 `actorRuleState(actor)`、`ruleValue(actor, namespace, key)` 和 `selfCheck(actor)`，调用方必须传入对应 Actor。`withActorRules(actor, callback)` 只接受同步回调，不能跨 `await` 保持规则作用域；更新完整快照使用 `updateRuleSnapshot(snapshot)`。

## 目录

| 路径 | 内容 |
|---|---|
| `module.json` | 模组清单 |
| `scripts/main.mjs` | 入口，注册世界设置 `taka-charsync.publishDir`（发布目录名，不在设置界面显示），按世界分建卡服模式和主服模式 |
| `scripts/char/` | 建卡服模式：规则覆盖、跟上主服、提交副本、状态栏 |
| `scripts/main/` | 主服模式：自动套用与撤回、面板 |
| `scripts/foundry/` | 两种模式共用的 Foundry 侧读写：取发布文件、导入卡 |
| `scripts/lib/` | 纯函数（规范化与哈希、合并、副本链、规则、时间），引擎、模组和单元测试共用 |
| `generated/rules.mjs` | 各团的 v2 规则快照 `SNAPSHOT`；仓库里是空表，建卡服上由引擎重写 |
| `styles/charsync.css` | 样式 |
| `engine/` | 同步引擎，入口 `engine/main.mjs` |
| `tools/` | 部署工具：`install.sh`、`enable-worlds.mjs`、`reset-xp.mjs`，后两个共用 `offline-db.mjs` |
| `test/unit/`、`test/engine/` | 单元测试 |
| `test/e2e/` | 本机双实例端到端测试，见 `test/e2e/README.md` |

## 角色涉及的模组

提交会随主人和同伴冻结模组使用证据。主服「查看变动」按实际套用范围显示模组检查；套用后的 GM 悄悄话和同步面板也会保留提示。普通同步保留主服库存与状态，因此建卡服测试装备上的引用不会作为本次套用的需求。

- 「规则引用的模组」检查已识别的 PF2e UUID 字段：GrantItem、ChoiceSet、Aura、EphemeralEffect、SpecialResource，以及族裔、背景和职业的授予列表。这些引用可能在规则准备、重新选择或授予时需要模组，不阻断同步。
- 「模组数据，需核对」表示角色保留了该模组的 flags，可能只是编辑记录；「内容来源」只表示从该合集导入过条目。两者都不等于必须开启。
- 当前世界可见但未启用的模组标为「未启用」；当前世界找不到的标为「未安装或不兼容」，需 GM 检查安装与兼容性。

「打开模组管理」进入 Foundry 原生窗口，由 GM 勾选、确认依赖并按提示重载世界。同步器不自动改模组开关。模组启用后重新打开同步面板，相应提示会消失。

ChoiceSet 目前只识别带 `.Item.` 文档类型的 UUID，旧格式需要额外的合集类型信息。识别使用实际字段与模块 ID，不按名称猜测无来源手工条目；动态 UUID 和第三方脚本注册的自定义规则不在自动识别范围内。旧提交没有冻结的模块信息时，仍可识别明确的 UUID 引用；无法识别的命名空间不会当成确定依赖。

## 测试

在本目录下运行：

```sh
node --test "test/unit/*.test.mjs" "test/engine/*.test.mjs"
```

端到端测试见 `test/e2e/README.md`，部署工具的端到端测试是 `test/e2e/test_tools.py`。

## 部署

CN 上的位置：

| 路径 | 用途 |
|---|---|
| `/root/foundryvtt` | Foundry 程序目录，工具和引擎从这里加载 `classic-level` 和 `common/server.mjs` |
| `/root/fvtt14-data/Data` | 主服 Data，状态接口 `http://127.0.0.1:30001/api/status` |
| `/root/fvtt14-char-data/Data` | 建卡服 Data，世界 `pf2creation`；核心 pm2 名 `fvtt-v14-char`，按需启停的控制器 `fvtt-v14-char-control`（端口 8793） |
| `/root/fvtt-charsync/app/` | 整个模组目录，引擎和工具从这里跑 |
| `/root/fvtt-charsync/releases/<时间>/` | `install.sh` 换下来的旧版 |
| `/root/fvtt-charsync/config.json`、`seed.json` | 引擎配置和种子表，不进仓库 |
| `/root/fvtt-charsync/state/` | 引擎状态：`rules.json`、`rules-enforcement.json`、`bootstrap.json`、`gc.json` |

首次上线按这个顺序：

1. 在仓库里按提交打包，传到 CN 解包，`bash <解包目录>/tools/install.sh --dry <解包目录>` 看一遍要做的拷贝，再去掉 `--dry` 跑：

   ```sh
   git -c core.autocrlf=false archive --format=tar.gz -o <文件>.tgz <提交>:模组/taka-charsync
   ```

   `install.sh` 要以 LF 换行到 CN，bash 不认 CRLF。模组目录的 `.gitattributes` 已经把 `*.sh` 定成 LF；`-c core.autocrlf=false` 让别的文件也按仓库里的原样打包。不要从 Windows 检出的工作区直接拷。
2. 等建卡服休眠：`pm2 jlist` 里 `fvtt-v14-char` 是 `stopped`，`curl http://127.0.0.1:8793/control/status` 报告 offline；有人在线就等它自己休眠，不踢人。然后 `pm2 stop fvtt-v14-char-control`，跑 `enable-worlds.mjs`（`WORLDS=pf2creation`），跑完 `pm2 restart fvtt-v14-char-control`。工具自己还会核对：pm2 里核心和控制器两个进程都是 `stopped`，30002 和 8793 两个端口都连不上，不满足就不写。
3. 主服的团世界逐个在没开时跑 `enable-worlds.mjs`；正开着的那个跳过，等它关了再补。
4. 需要时跑 `reset-xp.mjs`。
5. `config.json`、`seed.json` 放进 `/root/fvtt-charsync/`，先 `--dry` 跑一轮（见下）。
6. 起引擎：

   ```sh
   pm2 start /usr/bin/node --name fvtt-charsync-engine --kill-timeout 30000 -- /root/fvtt-charsync/app/engine/main.mjs --config /root/fvtt-charsync/config.json
   ```

   这里让 pm2 起 node 本身、把 `main.mjs` 当参数，和线上一致。直接 `pm2 start main.mjs` 时 pm2 6 用自己的容器脚本加载模块，`process.argv[1]` 是容器脚本；0.1.1 起引擎也看 `pm_exec_path`，这样起同样能跑，0.1.0 只看 `argv[1]`，这样起会 online 却一轮都不跑、日志为空。`--kill-timeout` 留时间给引擎跑完这一轮再退出，pm2 默认只等 1.6 秒。

   等到一行 `errors=0` 的 tick 日志，再按下面的条件 `pm2 save`。

升级：`pm2 stop fvtt-charsync-engine`，做第 1 步，再 `pm2 restart fvtt-charsync-engine`。先停引擎：它每轮可能重写建卡服模组目录里的 `generated/rules.mjs`，在换目录的空档里写进去会和安装打架。客户端文件换了以后，已经开着的世界要玩家刷新才会加载新版；之后单独更新团规由客户端定时读取。升级到 v2 时引擎与两端模组要一起更新，保留 `rules-enforcement.json`，并确认引擎已重新读取全部团世界、发布完整快照。

### `pm2 save` 的条件

只在建卡服休眠态做：`pm2 jlist` 里 `fvtt-v14-char` 是 `stopped`、`fvtt-v14-char-control` 是 `online`。CN 重启后 pm2 按存下来的进程表恢复，建卡服核心的启停要留给控制器管。

### install.sh

```sh
bash tools/install.sh [--dry] <解包后的模组目录>
```

- 两台 `Data/modules/taka-charsync/` 只放 `module.json`、`scripts/`、`styles/`、`generated/rules.mjs`。Data 整个目录都能被静态下载，`engine/`、`test/`、`tools/` 不放进去。
- `generated/rules.mjs` 只有建卡服那份由引擎重写：建卡服沿用现有的（没有才用包里的空表），主服每次都换成包里的空表。规则表的格式改了要换新的时，装完删掉建卡服那份，引擎下一轮会重写。
- 整个模组目录放到 `$CHARSYNC_HOME/app/`。两台的模组目录和 `app/` 已存在时，先整份挪到 `$CHARSYNC_HOME/releases/<YYYYmmdd-HHMMSS>/` 下的 `main-module/`、`char-module/`、`app/`。
- 主服有 `pf2e-eidolon-helper`、建卡服没有时整份拷过去；建卡服已有就不动。
- 新内容都先拷进同一父目录下的暂存目录（`.taka-charsync.installing` 等），拷完再换上。中途失败时删掉暂存目录、退出码非 0，还没换的旧版留在原处，已经挪走的在 `releases/<时间>/`。
- 传进来的目录没有 `module.json` 或 id 不对时直接报错退出。
- 不碰 pm2，不重启任何进程，不改文件属主。

环境变量：`MAIN_DATA`（默认 `/root/fvtt14-data/Data`）、`CHAR_DATA`（默认 `/root/fvtt14-char-data/Data`）、`CHARSYNC_HOME`（默认 `/root/fvtt-charsync`）。

### enable-worlds.mjs 与 reset-xp.mjs

两个工具都是离线写世界库，都分两种模式：

- `MODE=dry`：把库拷到临时目录，在副本上走和 apply 相同的路径（校验、备份、写入、读回）；运行检查不通过只记成警告。生产库不打开，一个字节都不碰。
- `MODE=apply`：先在副本上定好要写什么、过数据模型校验；没什么要写就直接结束，不建备份。要写时先做运行检查：写团世界时主服状态接口必须读得到（连不上、返回的不是 JSON、`active` 不是布尔值都算读不到）、正开着的不是它；写建卡服世界时 pm2 里核心和控制器都要是 `stopped`、两个端口都拒连。再打开生产库（打不开说明有程序开着这个世界，拒绝），确认内容和副本一致，整份备份到 `BACKUP` 并核对哈希（备份是工具占住库之后拷的，逻辑内容等于写前），写前再查一次运行状态，逐库写入，关库后从新副本读回逐行核对。一次写多个库时，中途失败的报告里 `written` 列着已经写了的库。

每次运行在 stdout 打一份 JSON 报告，`ok` 为 `false` 时退出码是 1，原因在 `problems` 里；`warnings` 不影响结果。

```sh
MODE=dry WORLDS=<世界1>,<世界2> PUBLISH_DIR=<发布目录名> CHAR_EXTRA_MODULES=pf2e-eidolon-helper \
  node /root/fvtt-charsync/app/tools/enable-worlds.mjs
MODE=apply WORLDS=<世界1> PUBLISH_DIR=<发布目录名> CHAR_EXTRA_MODULES=pf2e-eidolon-helper BACKUP=/root/backups/<新目录> \
  node /root/fvtt-charsync/app/tools/enable-worlds.mjs
MODE=dry WORLD=<世界> ACTORS=<actor id>,<actor id> node /root/fvtt-charsync/app/tools/reset-xp.mjs
```

两个工具都按路径运行，要和同目录的 `offline-db.mjs` 放在一起。

`enable-worlds.mjs` 对每个世界：

- `core.moduleConfiguration` 加上 `taka-charsync: true`，建卡服世界另加 `CHAR_EXTRA_MODULES` 里的模组；已有的其他条目原样保留。
- 写 `taka-charsync.publishDir`。已经是另一个非空值时报错，不覆盖。
- 建卡服世界另建「已提交」「已撤回」「草稿备份」三个带 `flags.taka-charsync.role` 的角色文件夹；已有同 `role` 的就不建。
- 新行和改动的行都过 Foundry 的 `BaseSetting`、`BaseFolder` 严格校验，`_stats.lastModifiedBy` 用该世界 `_id` 最小的 GM。

`reset-xp.mjs` 只把 `ACTORS` 里各行的 `system.details.xp.value` 改成 0，行里其余字节不动；经验已是 0 的跳过。行重新序列化后和原文不一致的不改，报错。

| 环境变量 | 用于 | 含义 |
|---|---|---|
| `MODE` | 都要 | `dry` 或 `apply` |
| `BACKUP` | apply 必填 | 备份目录，必须还不存在 |
| `WORLDS` | enable-worlds | 逗号分隔的世界 id；和 `CHAR_WORLD` 相同的按建卡服处理 |
| `PUBLISH_DIR` | enable-worlds | 发布目录名，`^[A-Za-z0-9_-]{12,}$` |
| `CHAR_EXTRA_MODULES` | enable-worlds | 建卡服世界另外启用的模组，逗号分隔，可空 |
| `WORLD`、`ACTORS` | reset-xp | 主服世界 id；逗号分隔的 actor id |
| `FOUNDRY_APP` | 都用 | 默认 `/root/foundryvtt` |
| `MAIN_DATA`、`CHAR_DATA`、`CHAR_WORLD` | 都用 | 默认 `/root/fvtt14-data/Data`、`/root/fvtt14-char-data/Data`、`pf2creation` |
| `MAIN_STATUS_URL`、`CHAR_STATUS_URL`、`CONTROL_URL` | 都用 | 默认主服 30001、建卡服 30002 的 `/api/status`，控制器 `http://127.0.0.1:8793/control/status` |
| `PM2_BIN`、`CHAR_PM2_NAME`、`CONTROL_PM2_NAME` | 都用 | 默认 `pm2`、`fvtt-v14-char`、`fvtt-v14-char-control` |
| `TMP_ROOT` | 都用 | 放临时副本的目录，默认系统临时目录；结束时删掉自己建的子目录 |

### 配置与种子表

`config.json`（路径字段按配置文件所在目录解析）：

```json
{
  "foundryAppDir": "/root/foundryvtt",
  "main": {"dataDir": "/root/fvtt14-data/Data", "statusUrl": "http://127.0.0.1:30001/api/status"},
  "char": {"dataDir": "/root/fvtt14-char-data/Data", "world": "pf2creation", "statusUrl": "http://127.0.0.1:30002/api/status"},
  "publishDir": "<发布目录名>",
  "worlds": ["<团世界 id>"],
  "titles": {"<团世界 id>": "<显示名>"},
  "campaignFolders": {"<建卡服角色文件夹 id>": "<团世界 id>"},
  "intervalSeconds": 60,
  "historyDays": 60,
  "stateDir": "/root/fvtt-charsync/state",
  "tmpRoot": "/tmp",
  "seedPath": "/root/fvtt-charsync/seed.json"
}
```

`campaignFolders` 可省略，默认为 `{}`；没有卡片映射的新卡必须通过这里取得归属。键是建卡服角色文件夹的 16 位字母数字 ID，值必须在 `worlds` 中。只配置工作卡的团文件夹，不把「已提交」「已撤回」「草稿备份」这些跨团文件夹映射到某个团。

`publishDir` 当作口令用：知道它就能下载里面所有卡，所以取足够长的随机串，不写进仓库。试跑时另加 `"dryOutDir"`：`--dry` 的发布文件、规则表和引擎状态都写到它下面的 `main/`、`char/`、`state/`，不用 `stateDir`；库和图片照样从两台 Data 读。`tmpRoot` 必须是已经存在的目录。

```sh
node /root/fvtt-charsync/app/engine/main.mjs --config <试跑配置> --once --dry
```

`seed.json`：

```json
{
  "cards": [{"charId": "<建卡服 actor id>", "world": "<团世界 id>", "mainId": "<主服 actor id>", "companions": ["<建卡服同伴 id>"], "bootstrap": "auto"}],
  "users": [{"world": "<团世界 id>", "worldUser": "<主服用户 id>", "charUser": "<建卡服用户 id>"}],
  "bootstrapSince": "<一次性同步写完的时刻，ISO 8601>"
}
```

- 魔宠不用写进 `companions`，引擎按 `system.master.id` 认；幻灵、追随者这类要写。
- 两边同名的账号自动配对，`users` 只写名字不同的。
- `bootstrap` 决定起步基线：`auto` 按 `bootstrapSince` 之后哪边改过判断，`main` 一律以主服为准（两边同 id、一直没同步过的卡用它）。
- 配置和种子表只在启动时读，改了要 `pm2 restart fvtt-charsync-engine`。

## 日志

`pm2 logs fvtt-charsync-engine`。每轮一行：

```
[charsync] tick main=<正开着的主服世界或 none> read=<这一轮读了的库数> copies=<新发布的副本数> skipped=<条目不全、沿用上次版本的卡数> errors=<错误数>
```

每个错误另起一行 `[charsync] error <范围>: <说明>`，范围是世界 id，或 `main-status`、`publish`、`images`、`rules`、`undo`、`char-history`、`bootstrap`、`gc`。单个库出错只影响它自己，其他照常发布。整轮抛错是 `[charsync] tick failed: …`，下一轮照常跑；停进程时最后一行是 `[charsync] stopped`。

模组在浏览器控制台的日志前缀是 `taka-charsync |`。

## 发布目录

两台 Data 下各有一份 `<publishDir>/`，内容相同，引擎每轮只写有变化的文件，每个文件先写临时文件再改名。

| 路径 | 内容 |
|---|---|
| `meta.json` | 卡和账号的映射、各团信息、v2 团规快照 `ruleSnapshot`、起步基线、已发布的副本、建卡服各卡的版本摘要；最后写，客户端从它开始读 |
| `main/<世界>/<主服 id>.json` | 主服当前版本 |
| `copies/<卡 id>.json` | 这张卡的冻结副本列表 |
| `working/<卡 id>.json` | 建卡服上玩家卡的当前版本 |
| `history/main-<哈希>.json`、`history/copy-<副本 id>-<版本>.json`、`history/char-<哈希>.json` | 历史版本。超过 `historyDays` 天又没被引用的每天清一次 |
| `undo/<世界>/` | 只在主服：GM 客户端套用前上传的撤回快照 |

图片保存在 Data 的素材目录，不放在发布目录里。引擎每轮会确认两端的冻结素材目录已建立；玩家使用文件上传权限保存，无需目录创建或管理权限。提交冻结的本地素材使用 `assets/taka-charsync-media/<SHA-256>.<扩展名>`，引擎核对两端实际字节，不覆盖哈希不符的文件；普通旧路径只补缺。Tokenizer 的图层、遮罩和画笔遮罩也随卡同步。生成式圆环遮罩保留原有图层引用，只冻结配方中实际使用的图片。头像和 Token 的图片与配方分别作为一组选择，冻结路径变化本身不计作玩家修改。

该保证覆盖通过编辑器 hook 的保存和提交时可读取的本地素材。未经过该 hook 的批量/API 保存、外站 HTTP/S3 内容及过去已经被同名覆盖的图片，不具有完整历史像素保证。核心、系统和模组资源沿用各端安装；工作卡直接引用的世界目录素材仍不镜像，提交时会先冻结到独立素材目录。

## 回退

- 只想先停掉自动套用：在团世界的模组管理里停用 `taka-charsync`，或停引擎 `pm2 stop fvtt-charsync-engine`（发布目录停在最后一轮，客户端还能读）。
- 换回旧版代码：`pm2 stop fvtt-charsync-engine`，把 `releases/<时间>/` 下的 `main-module/`、`char-module/`、`app/` 挪回原位（当前版先挪开），删掉建卡服 `modules/taka-charsync/generated/rules.mjs`，再 `pm2 restart fvtt-charsync-engine`，引擎下一轮按换回的版本重写规则表。
- 撤销离线写库：世界关着时（建卡服还要先停控制器），把那个世界 `data/` 下对应的库目录（`settings`、`folders` 或 `actors`）整个换成 `BACKUP` 里同名的那份。备份是工具占住库之后、写入之前拷的，内容等于写前（打开库时 LevelDB 可能整理过日志文件，文件不一定和打开前逐字节相同）。换回去会连同之后别的改动一起丢掉。
- 单张卡被套错：主服面板上「恢复到套用前」用的是 `undo/<世界>/` 里的快照。
- 彻底撤掉：停引擎并 `pm2 delete fvtt-charsync-engine`，按上面的条件 `pm2 save`；各世界停用模组；删两台的 `modules/taka-charsync/` 和 `<publishDir>/`。拷到另一边的图片不会跟着删。
