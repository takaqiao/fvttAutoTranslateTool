# 升报给主控的精确补丁：`R-selfcheck-d-liveness` 的记录值随本轮新表上移

## 为什么需要

本轮往 `1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs` 加了一张 `TOKEN_MAKER_UI`（149 键）
并把它登记进 `SELFCHECK_TABLES`。自检面板 D 档的**覆盖侧分母**因此上移：

| 项 | 记录值 | 本轮实测 | 方向 |
|---|---|---|---|
| `min.checkedDistinct` | 719 | **891** | 只许涨 |
| `min.rawChecked` | 1273 | **1473** | 只许涨 |
| `min.tableRows` | 39 | **40** | 只许涨 |
| `min.tablesFedIn` | 39 | **40** | 只许涨 |
| `min.registeredRaw` | 1495 | **1659** | 只许涨 |
| `min.registeredDistinct` | 896 | **1056** | 只许涨 |
| `max.uncheckedDistinct` | 168 | **165** | 只许降 |
| `max.missDistinct` / `max.rawMiss` | 4 / 7 | **4 / 7 不变** | 149 键全部在上游有字面量 |

D 档那些 min/max 本身都满足（min 只许涨、max 只许降）。
真正掉的是 `--selftest`：**357 → 356**，主闸随之红在 `<main 的固定检查> —— --selftest 没过`。
掉的那一条是

```
FAIL 把 `min.registeredRaw` 抬高一格 → 覆盖侧分母必须响（违规 0 处）
```

`assert_resolutions.py:5739` 的灵敏度用例做的是 `registeredRaw + 1` 再要求报违规。
记录值 1495 + 1 = 1496，而实测已是 1659 —— **1659 ≥ 1496，闸不响，这条灵敏度用例空转了**。
这正是第三十二轮 V18 建它时要防的失效方式，所以不能靠「反正是 min，涨了不红」放着不管。

### 「那就别把新表登记进面板」——实测行不通，别再试

我先试过这条退路：把 `TOKEN_MAKER_UI` 从 `SELFCHECK_TABLES` 里摘掉，只留 `DIALOG_UI` 那 15 键。
实测 `registeredRaw` 仍从 1495 涨到 **1510**（就是那 15 条），1510 ≥ 1496，
**同一条灵敏度用例照样 FAIL，`--selftest` 仍是 356/357**。

⇒ 结论：**往任何一张已登记的表加键，都必须连着抬记录值**。
这与「往 PATTERNS / PREFIXED / NOTIFICATION_PATTERNS 加条目必须同时补用例」是同一条纪律，
不是这张新表的特例，也没有绕过去的写法。既然代价一样，本轮就把 `TOKEN_MAKER_UI`
**正常登记**（149 键全部纳入 D 档看守），而不是为了躲一条红把新表放在闸外。

## 补丁必须同时改两份文件

只改规则 JSON 会让主闸从 67/1 掉到 **65/4**（实测），因为
`assert_resolutions.py` 的 `PAYLOAD_FLOORS` 里这几个键是 `("eq", N)` **精确钉死**的：

```
[-/配置·payload_floors] R-selfcheck-d-liveness.min.registeredRaw
  `…min.registeredRaw` 从 1495 变成了 1659 —— 含义两可的强度参数一律钉死，
  要改就在 PAYLOAD_FLOORS 里一起改
```

### ① `5-其他内容/RESOLUTIONS.assertions.json`

`R-selfcheck-d-liveness` 的 `min` / `max` 两块，改这 7 个值（见上表）。
已备好整份补好的文件：`RESOLUTIONS.assertions.FULL.json`（与本文件同目录），
差异只有这 7 个数字，见 `RESOLUTIONS.min_max.diff`。

### ② `3-常用脚本/qa/assert_resolutions.py`

`PAYLOAD_FLOORS` 里 `"R-selfcheck-d-liveness"` 那一行（约 :3656），7 处 `("eq", N)`。
每个左侧串在全文件**各只出现 1 次**（已实测计数），逐条替换即可：

```
"min.checkedDistinct": ("eq", 719)      ->  "min.checkedDistinct": ("eq", 891)
"min.rawChecked": ("eq", 1273)          ->  "min.rawChecked": ("eq", 1473)
"min.tableRows": ("eq", 39)             ->  "min.tableRows": ("eq", 40)
"min.tablesFedIn": ("eq", 39)           ->  "min.tablesFedIn": ("eq", 40)
"min.registeredRaw": ("eq", 1495)       ->  "min.registeredRaw": ("eq", 1659)
"min.registeredDistinct": ("eq", 896)   ->  "min.registeredDistinct": ("eq", 1056)
"max.uncheckedDistinct": ("eq", 168)    ->  "max.uncheckedDistinct": ("eq", 165)
```

`:3417` 的强度死下限 `("panel_liveness","min","registeredRaw"): 1459` **不用动**
（1659 ≥ 1459，它是「不得低于」）。

## 补丁验到底的过程（不是推断，是实跑）

⚠ **`--rules <副本>` 对这一条无效**，必须说清楚：`_panel_rule()`（`assert_resolutions.py:5636`）
读的是模块级常量 `DEFAULT_RULES`（:272，由 `__file__` 推出的项目根），
**不接 `--rules`**。拿副本跑实测复现：打了补丁的副本照样报 356/357。
所以改用**整棵镜像树**（`ROOT` 由 `__file__` 推出 ⇒ 镜像里的脚本读镜像里的规则）：

```
scratchpad/mirror/
  3-常用脚本/qa/            <- 真实拷贝
  5-其他内容/               <- 真实拷贝（规则文件在这里打补丁）
  1-Ember汉化插件/           <- 目录联接（junction）指向真实树
  2-Crucible汉化插件/        <- 同上
  PROJECT.md / PARALLEL-RUNBOOK.md
```

| 镜像树状态 | `--selftest` | 主闸 |
|---|---|---|
| 未打补丁（＝当前真实树） | **356 / 357**（`registeredRaw` 那条 FAIL） | 67 通过 / 1 失败 |
| 只打 ① 规则 JSON | 345 / 357 | **65 / 4**（payload_floors 报 4 处） |
| ① + ② 两份都打 | **357 / 357** · `panel_liveness：13 / 13` | 66 / 2 |

最后一行主闸的 2 处失败逐条说明，都不是本补丁造成的：
* `R-version-matrix` —— PROJECT.md 抬头/矩阵还停在 v1.1.25 而 module.json 已是 1.1.26
  （**本轮开工前就红**，见下节）；
* `R-assertion-inputs-tracked` —— 镜像树的 `3-常用脚本` 不是 git 仓，是镜像固有产物；
  真实树上这条实测 `ok`。

## 顺带升报：主闸开工前就红着的一条

`R-version-matrix` 在**我动手之前**就是失败的（基线实测 67 / **1** / 0，不是任务书写的 67/0/0）：
`ember_cn_unofficial` 的 `module.json` 已经是 `1.1.26`（急修 `EMBER.*` 命名空间那次发的版），
而 `PROJECT.md` 的抬头段、第 7 行、版本矩阵三处仍写着 v1.1.25。
`PROJECT.md` 不归我，未动。
