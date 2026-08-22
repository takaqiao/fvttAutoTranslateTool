# crucible 硬编码汉化 · 真浏览器作用域验（2026-08-22）

页面 `scope.html`，按上游模板逐行搭；被测代码是**发布中那个文件的原文**（只去掉 `export ` 以便内联）。
真 DOM、真 CSS 选择器引擎、真身 `translateCrucibleRoot()`。

| 断言 | 结果 |
|---|---|
| chat 6 条加值/减值来源名全译（已是中文的那条不动） | PASS |
| chat 5 条 context tooltip 全译，`ACTION.TAGS.Target`（i18n 键）不动 | PASS |
| 5 个 placeholder 全译，ember 的 `Character Name` 不动 | PASS |
| 创建页加减按钮只译前缀，物品名 `Steel Longsword` 原样保留 | PASS |
| **负例区（别的模块的窗口）一处未动** | PASS |
| 负例区计数 `{text:0, attr:0, aria:0}` | PASS |
| 幂等：第二遍全零 | PASS |
| 首遍共改 18 处 | PASS |

**8 / 8 通过。**

负例区放的是同名串：`input[placeholder="Item Name"]`、`input[placeholder="Actor Name"]`、
`span.label>Special`、`label.tag-icon[data-tooltip="Reload"]`、`button[aria-label="Add one Widget"]`，
都摆在一个 `class="application some-other-module"` 的窗口里。
一条都没被吃 —— 这就是「分两档作用域」那条设计的实测依据：
结构选择器档（boon/bane、context-tags）不要求 `.crucible` 祖先（因为掷骰聊天卡根是动态 `cssClass`），
通用词档（placeholder、aria 前缀）必须落在 `.crucible` 内。
