# Terminal 模组简体中文汉化 — 工作副本

上游：[CodaBool/terminal](https://github.com/CodaBool/terminal) **v4.0.11**（本地基线）
部署目标：`%LOCALAPPDATA%\FoundryVTT\Data\modules\terminal`
原版备份：`Desktop\fvtt\terminal-4.0.11-original-20260903-221133.zip`

## 设计约定（改动必须同时满足）

1. `cn` 客户端：所有人读的文字都是中文。
2. 非中文客户端：看到的英文与上游 4.0.11 **逐字一致**（含上游自带的拼写错误，如 `sever`、`Then chose`）。英文用户不应察觉模组被改过。
3. 本地化在**显示端**完成。Foundry 的语言是每客户端设置，所以跨 `game.socket` 的载荷只能带 key + data，不能带整句。
4. 刻意不翻译：shell 命令名、路径、虚构的机器输出（`ping`/`ps`/`df`/`top`/`curl`、`arch btw`、`dug`）。
5. 只改表现，不改行为。

## 语言代码必须是 `cn`，不是 `zh-CN`

Foundry 的 `#filterLanguagePaths` 是严格相等比较（`client/helpers/localization.mjs`）。本机
`Config\options.json` 是 `"language": "cn.foundry_chn"`，且已装的每一个中文包
（`foundry_chn`/`pf2_cn`/`crucible-cn`/`5e_chn`/`quick-insert`…）都注册 `cn`，繁体用 `zh-tw`。

早期版本注册成 `zh-CN`，结果 `lang/zh-CN.json` **从未被加载**，
`isSimplifiedChinese()` 恒为 false，**整套汉化完全没有生效**。
`tests/release-gates.test.mjs` 第一个用例就守着这条。

## 目录

| 路径 | 说明 |
| --- | --- |
| `lang/en.json`、`lang/cn.json` | 322 条，key 集合与占位符必须完全一致 |
| `templates/*.hbs` | 14 个英文模板，与上游**逐字节相同**，不要动 |
| `templates/cn/*.hbs` | 14 个中文对应模板，只允许可见文字不同 |
| `scripts/i18n.js` | `t` / `templatePath` / `actionLabel` / `docTypeLabel` / `configureLocalizedApplications` |
| `scripts/util.js` | 翻译助手在此文件里叫 **`tr`**（该文件原本已有局部变量 `t`） |
| `macro.js` | 合集宏的**源**，运行时不加载 |
| `packs/terminal-macros/` | 用户真正拿到的宏；改完 `macro.js` **必须**重建 |
| `tools/` | `deploy.ps1`、`localize-pack.mjs`、`verify-pack.mjs`、`macro-commands.mjs` |
| `tests/` | 16 个用例，见下 |

`tests/`、`tools/`、`package.json` 不随部署下发。

**入库范围：** 只有源文本与代码进 git。`audio/`、`background/` 是未改动的上游素材（19MB），
`packs/` 是 LevelDB 二进制。全新检出后要先从原版备份 zip 解出 `packs/terminal-macros`
作为种子，再跑 `npm run pack:build` —— `localize-pack.mjs` 只会改写已存在的记录，不会凭空建库。

## 工作流

```
npm test          # 16 个用例
npm run pack:build   # macro.js -> LevelDB（改了 macro.js 就要跑）
npm run pack:verify  # 逐字节比对 pack 与 macro.js
npm run deploy       # 复制运行时文件并逐个 SHA256 校验（Foundry 必须关闭）
```

## 已知的坑（都栽过）

- **`zh-CN` 语言代码让整套汉化静默失效。** 见上。
- **pack 与 `macro.js` 会不同步。** 曾经发布过一版 pack，宏里传的是 `{users}` 而目录要的是
  `{count}`/`{names}`。Foundry 的替换是 `data[k.slice(1,-1)]`，缺 key 得到的是字符串
  `undefined`，不是原样的花括号。GM 看到「已为 undefined 名用户打开终端（undefined）」。
  旧的 `verify-pack.mjs` 对这个坏 pack 是**通过**的。
- **模块加载早于 `Localization.initialize()`。** 任何在模块顶层算出来的
  `PARTS` / `DEFAULT_OPTIONS` / 模板路径都会冻结成错的语言。所有绑定都放在 `i18nInit` 里做。
- **发送端本地化 = 跨客户端串语言。** 中文玩家发出的整句会原样显示给英文 GM。
  技能检定描述改成 `descriptionKey`/`descriptionData`，由 `hooks.js` 的 `skillDescription()`
  在 GM 自己的客户端渲染。
- **`[^a-zA-Z0-9]` 会把中文名字清成空串。** 脚本/区域/角色的假路径原本用它做 slug，
  中文名全部塌缩成同一个 `/usr/local/bin/`。现在用 `[^\p{L}\p{N}]`。
- **`padEnd` 按 UTF-16 码元计数。** CJK 一个字占两格，CLI 表格全是歪的。改为按显示宽度补齐。
- **`data.title` 同时是身份和显示文本。** `hooks.js` 里有十处 `===` 比较依赖它的英文原值，
  所以 wire 上保持英文，只在显示时套 `actionLabel()`。

## 有意保留 / 未做

- **聊天栏消息仍由创建者的客户端渲染**（`util.js` 的 `Chat.ControlTimer` /
  `Chat.MotionTimer` / `Chat.AccessRevoked`）。ChatMessage 是持久化并广播给所有人的，
  要按读者语言渲染得加一层 `renderChatMessage` 重绘，会碰到正在跑的倒计时逻辑，风险大于收益。
  混合语言的桌子上，这几条计时消息会是创建者的语言。
- **shadow 视图的标题由来源客户端渲染。** 这是「旁观某人的终端」，显示成对方看到的样子是对的。
- **`ui.js` 新建样式名 `custom style N` 保持英文。** 它是持久化的世界数据，且
  `.includes("custom ")` 靠它计数，翻译会改行为。
- **两处上游 bug 没有照抄**：地图下载通知原本会打印字面量 `undefined`/`false`；
  未知系统的「Access granted, 」原本留着一个悬空逗号。这两处保留了修正版，是对约定 2 的有意偏离。
- **未升级到上游 4.0.13。** 4.0.12 加了 Levels 支持、4.0.13 修了其中的 bug。
  上游仓库只放 `module.json`，代码走 Foundry 包服务器，且新版是 `protected: true`，
  无法在此处取得源码重做基线。

## ⚠️ 更新会抹掉汉化

`module.json` 的 `manifest` 仍指向上游。Foundry 会提示可更新到 4.0.13，
**点下去整套汉化就没了**（而且 4.0.13 是 `protected`，可能还要过授权）。
升级流程只能是：装官方新版 → 重新做 diff → 重跑 `npm test` 与 `npm run deploy`。
