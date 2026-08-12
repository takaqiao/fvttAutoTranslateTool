# 对上游英文基准打的本地补丁

> **每次用 `extract_en.mjs` 重抽英文之后，必须回来照这张表重打一遍**，
> 否则会静默回退，相关条目的译文会重新被闸门拒收。

这里只收录**上游英文自身写坏、且坏到会卡住翻译流程**的地方。
上游把它修好之后，对应行可以删掉（重抽后发现不再匹配即说明已修）。

---

## 1. `Toothbreaker Hideout / Prison` 的 `@Condition[exhaustion` 缺右方括号

- **日期**：2026-08-09
- **影响包**：`ember.crucible-adventure.json`、`ember.adventure.json`（两包英文逐字节相同）
- **路径**：`Ember Early Access.journals.Toothbreaker Hideout.pages.Prison.text`
- **上游原文**：

  ```html
  <sub data-system="crucible">@Condition[exhaustion</sub></sup>
  ```

- **补成**：

  ```html
  <sub data-system="crucible">@Condition[exhaustion]</sub></sup>
  ```

**为什么非修不可**：`markup_signature` 的 `MARKUP = @[A-Za-z]+\[[^\]]*\]` 会从
`@Condition[` 一直吞到**下一个** `]`。英文与中文后续的文字不同，被吞进标记的范围就不同，
于是签名永远对不上 —— 即使中文一字不差地照抄了这个坏标记也一样被拒。
这是全库最后一条过不了闸的条目（34859 分之 1）。

补上之后：英文侧标记变成干净的 `@Condition[exhaustion]`，中文照抄同一个标记即可匹配；
渲染出来玩家还能得到一个可点的「力竭」状态链接，比上游原样更好。

**同时改的**：`compendium/cn/ember.crucible-adventure.json` 同一路径的中文里
那个照抄来的坏标记也补了 `]`。

---

## 2. `Unfinished Business / Answers From On High` 把 `{Persuasion}` 裸写在正文里

- **日期**：2026-08-12
- **影响包**：`ember.crucible-adventure.json`、`ember.adventure.json`
- **路径**：`Ember Early Access.journals.Unfinished Business.pages.Answers From On High.text`
- **上游原文**：`Agraband automatically succeeds on the Charisma {Persuasion} check.`
- **补成**：`… on the Charisma (Persuasion) check.`

**为什么要修**：这里的 `{Persuasion}` 前面**没有** `@UUID[…]`，也不在任何 enricher 里 ——
作者多半是想写个链接、只留下了标签。Foundry 会把花括号原样渲染给玩家看。

而 `scan_markup_drift` 的 PLACEHOLDER 判据（`\{[A-Za-z_][A-Za-z0-9_.\-]*\}`）分不清
「真占位符」和「作者手滑留下的花括号」，于是把这条永远报成缺失。中文写的是
「魅力（游说）」——比上游还正确，不该为了迁就一条正则改成「魅力{游说}」。
把英文基准补成圆括号，两边就都对了。
