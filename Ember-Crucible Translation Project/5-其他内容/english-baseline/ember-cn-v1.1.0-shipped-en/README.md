# `ember-cn-v1.1.0-shipped-en` —— ember_cn_unofficial v1.1.0 发版当时的英文

## 这份是从哪来的、为什么必须有它

**从 `1-Ember汉化插件` 自己的 git 历史里取的**：

```bash
cd 1-Ember汉化插件
for f in $(git ls-tree -r --name-only v1.1.0 | grep '^compendium/en/.*\.json$'); do
  git show "v1.1.0:$f" > "<本目录>/$(basename "$f")"
done
```

⚠⚠ **「那 6 个包的历史 drift 永久不可答」这个结论是错的，2026-08-15 第二十轮推翻。**
此前两轮都以为 `ember-cn-v1.0.15-shipped-en/` 只有 3 个包 ⇒ 另 6 包
（`ember.adventure` 等，约 15147 条中文叶）的「上游从那时到现在改了什么」无法回答。
**真相是**：`v1.0.15` 那会儿模块**根本只有 3 个 crucible 侧的包**，
`ember.adventure.json` 等是 **`v1.1.0` 才加进来的**（`git ls-tree v1.0.15` / `v1.1.0` 可复现）。
所以它们的正确基准不是 v1.0.15，而是 **v1.1.0**，而 v1.1.0 的 `compendium/en/` 一直躺在插件仓里。

**教训：找历史英文之前，先去插件仓的 git 历史里看一眼 ——
`compendium/en/` 从 v1.1.0 起就是被跟踪的，每个 tag 都是一份现成的历史基准。**

## 拿它 diff 出来的结论（2026-08-15 实测）

v1.1.0 → 当前，**英文变过的叶只有 3 条**（× 孪生两包），而且**三条全是我们自己打的
本地补丁**（补 `]]` · `C0nsortium`→`Consortium` · `{Persuasion}`→`(Persuasion)`，见 `../LOCAL-PATCHES.md`）：
**上游英文一个字都没改。**

⚠ **别把这个「3」和 `LOCAL-PATCHES.md` 正表的「5」当成矛盾 —— 两边都对，只是起算点不同。**
正表另外 2 条（第 1 条与第 5 条）打于 `dd54bdd`（2026-08-09），**早于 `v1.1.0` tag**
（`c426d45`，2026-08-11），所以它们**已经烘进 v1.1.0 的英文里**，从 v1.1.0 起算不显示为差异。
换算关系：**从上游 LevelDB 重抽起算 = 5 条 · 从 v1.1.0 起算 = 3 条**。
（`git merge-base --is-ancestor dd54bdd v1.1.0` 成立，第二十轮验过。） 另有新增 792 叶 / 删除 352 叶（每包），
新增部分按 `validate_translations` 是 100% 已译。

⇒ **那个 15147 叶的盲区，判据覆盖是真缺，但内容敞口是零。** 现在这句话有据可查。

## 怎么用

三条 drift 闸（`scan_dropped_terms` / `scan_number_drift` / `scan_marker_followup`）
以及 `scan_en_drift` / `scan_renamed_terms`，对 **ember 侧的这 10 个包**都可以拿它当 `--baseline`。
⚠ `ember-cn-v1.0.15-shipped-en/` **仍然有用且不可替代** —— 它回答的是更早那一段
（v1.0.15 → 今天）对那 3 个 crucible 侧包的 drift，本目录回答不了。**两份并存，各答各的区间。**
