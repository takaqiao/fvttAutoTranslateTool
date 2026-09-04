# 单元翻译须知（PF2 Plus 十一模组）

你负责**一个单元**。读它、翻它、写结果、自校验通过为止。别改别的文件。

## 你要读的三个文件

| 文件 | 内容 |
|---|---|
| `qa/units/<pack>/<NN>.json` | 待译叶子。每条有 `path`（数组）、`style`、`field`、`en` |
| `qa/units_gloss/<pack>/<NN>.json` | 本单元可能用到的术语，`from` 标了来源 |
| `工作区/en/<pack>.json` | 整包英文基线，需要上下文时查（同一文档的别的字段） |

## 你要写的一个文件

`qa/units_out/<pack>/<NN>.json`：

```json
{"pack": "<照抄单元里的 pack>", "unit": "<照抄 unit>",
 "translations": [{"path": ["entries", "...", "description"], "zh": "……"}]}
```

`path` **逐元素照抄**（文档名里常带点号，不能拼成点分字符串）。单元里的每条叶子都要有一条对应的翻译，不许漏。

## 两种文体，不可混用

**`style: "bilingual"`**（`name` / `tokenName` / `prototypeToken` / 文件夹名）
→ `中文 English`，**一个半角空格**，不加括号、不换行。
例：`Acid Cannon` → `强酸炮 Acid Cannon`

**`style: "prose"`**（`description` / `text` / `caption` / 各类正文）
→ **纯中文**。正文里不留英文句子，也不要在中文后面附一段英文原文。

## 不能碰的东西（碰了就是坏掉，而且覆盖率指标照样满分）

1. **方括号里全是机器件，逐字节照抄**：`@UUID[...]`、`@Check[...]`、`@Damage[...]`、
   `@Template[...]`、`[[/r ...]]`、`[[/act ...]]`。
   里面的 `fire` / `reflex` / `dc:25` / `#Light Killer Counteract` / `Compendium.pf2e.x.Item.y`
   **一个字符都不改**——它们是拿去和英文键匹配的，译了就解析失败。
   *唯一例外*：`@Check[...|name:某某]` 里 `name:` 后面那段是给玩家看的标签，要中文。
2. **方括号后面紧跟的 `{标签}` 是散文**，要译成中文：
   `@UUID[Compendium.pf2e.conditionitems.Item.xxx]{Sickened}` → `…]{恶心}`
3. **`@Localize[...]` 连键都不能碰**，整条照抄。
4. **HTML 标签序列必须一模一样**：`<p>` `<hr />` `<ul>` `<li>` `<strong>` `<em>` `<h3>` `<section class="...">`
   的**数量、种类、顺序**都不能变。`<hr />` 写成 `<hr>` 也算改动，照抄原样。
5. 骰式（`1d6`）、加值（`+19`）、DC、房间号、署名/版权/商标名单保持英文。
   `XP DC AC HP GM NPC PC Paizo Pathfinder Starfinder Foundry` 保持拉丁。

## 术语

优先级：**本单元 glossary 里 `from: "corpus"` 的 > `from: "wiki"` 的 > 你自己的判断**。
glossary 是参考不是命令：明显不适用于上下文的条目（`Guard` 在句子里是动词时）别硬套。
glossary 没有、你也拿不准的专有名词，按 PF2e 中文社群惯例音译，并在返回里列出来。

规则关键词按 PF2e 中文惯例：
`Strike 打击` `Interact 交互` `Craft 制作` `Activate 激发` `Frequency 频率`
`Trigger 触发` `Requirements 需求` `Effect 效果` `Special 特殊` `Cost 成本`
`Critical Success 大成功` `Success 成功` `Failure 失败` `Critical Failure 大失败`
`Saving Throw 豁免` `Fortitude 强韧` `Reflex 反射` `Will 意志`
`status bonus 状态加值` `circumstance bonus 环境加值` `item bonus 物品加值`
`penalty 减值` `resistance 抗力` `weakness 弱点` `temporary Hit Points 临时生命值`

**同一个英文名在本单元里只能有一个中文译名。** `Sickened 1` / `Sickened 2` 这种带数字的，
数字要留：`恶心 1` / `恶心 2`——上一轮就是把三档都译成「恶心」丢了档位。

## 写完自校验（必做）

```bash
cd C:/Users/Taka/Desktop/fvtt/模组/PF2Plus汉化
PYTHONIOENCODING=utf-8 python ../../冒险/AV/qa/check_unit.py \
  --unit qa/units/<pack>/<NN>.json --result qa/units_out/<pack>/<NN>.json
```

退出码 0 才算完。它查六项：有中文 / HTML 标签序列一致 / 方括号体逐条一致 /
文体正确 / 不是照抄英文 / 一条不漏。报错就按报的那条改，改到过。

## 返回

一句话：单元名、叶子数、check_unit 是否通过、以及你拿不准的术语（若有）。
