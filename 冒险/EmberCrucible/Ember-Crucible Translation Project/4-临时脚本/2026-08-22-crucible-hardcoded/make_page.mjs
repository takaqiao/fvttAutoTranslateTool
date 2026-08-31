/**
 * 造一张**真浏览器**里跑的验收页：真 DOM、真 CSS 选择器引擎、真身 `translateCrucibleRoot()`。
 *
 * ⚠ 被测代码是**发布中那个文件的原文**，只把 `export ` 去掉好内联，别的一个字节不改。
 *   在 node 里跑不了它：要 `querySelectorAll`，而自己写一个假的
 *   等于把「选择器选得对不对」测成同义反复 —— 那正是本轮要避免的空转形态。
 *
 * 页面按**上游模板逐行**搭：
 *   · `standard-check-chat.hbs:1` 的根是 `<div class="{{cssClass}} line-item">`（**动态、不含 crucible**）
 *   · `standard-check-details.hbs:2-6/14-18` 的 boon/bane 结构
 *   · `action-use-header.hbs:21-23` 的 context-tags
 *   · `item-header.hbs:5` / `hero-header.hbs:3` / `group.hbs:6` / `affix-header.hbs:5` / `action/header.hbs:5`
 *   · `creation/equipment.hbs:47/51` 的加减按钮
 * 另外摆一批**别的模块**的近似串（同名 placeholder、同名 tooltip、同名 .label），一条都不许被吃。
 */
import fs from "node:fs";
const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/2-Crucible汉化插件/crucible-hardcoded-cn.mjs";
const code = fs.readFileSync(SRC, "utf8").replace(/\r\n/g, "\n").replace(/^export /gm, "");

const page = `<!doctype html><meta charset="utf-8"><title>crucible hardcoded scope</title>
<body>

<!-- ① 掷骰聊天卡：根是动态 cssClass，**不含 crucible** —— 结构规则必须照样命中 -->
<div id="chat" class="expanded line-item">
  <div class="boon-details flexcol">
    <div class="boon flexrow"><span class="label">Special</span><span class="type">Boon</span><span class="value">1</span></div>
    <div class="boon flexrow"><span class="label">Reserved Action</span><span class="value">2</span></div>
    <div class="boon flexrow"><span class="label">Elite</span><span class="value">2</span></div>
    <div class="boon flexrow"><span class="label">Boss</span><span class="value">4</span></div>
    <div class="boon flexrow"><span class="label">已经是中文</span><span class="value">1</span></div>
  </div>
  <div class="bane-details flexcol">
    <div class="bane flexrow"><span class="label">Slow Weaponry</span><span class="value">1</span></div>
    <div class="bane flexrow"><span class="label">Bulky Armor</span><span class="value">2</span></div>
  </div>
  <div class="context-tags full-tags">
    <label class="tag-icon" data-tooltip="Strikes" data-tooltip-direction="LEFT"><i></i></label>
  </div>
  <div class="context-tags full-tags"><label class="tag-icon" data-tooltip="Weapon Tags"><i></i></label></div>
  <div class="context-tags full-tags"><label class="tag-icon" data-tooltip="Spell Tags"><i></i></label></div>
  <div class="context-tags full-tags"><label class="tag-icon" data-tooltip="Skill Tags"><i></i></label></div>
  <div class="context-tags full-tags"><label class="tag-icon" data-tooltip="Reload"><i></i></label></div>
  <!-- 负例：ACTION.TAGS.Target 是 i18n 键，Foundry 自己会本地化，我们不许碰 -->
  <div class="target-tags full-tags"><label class="tag-icon" data-tooltip="ACTION.TAGS.Target"><i></i></label></div>
</div>

<!-- ② crucible 自家的卡：根**自己**带 crucible 类（querySelectorAll 找不到自己，要单独算） -->
<div id="sheet" class="application crucible item standard-form">
  <input name="name" type="text" placeholder="Item Name">
  <input name="x" type="text" placeholder="Character Name">   <!-- 负例：ember 的串，不归本表 -->
</div>
<div id="sheet2" class="application crucible actor standard-form">
  <input class="charname" name="name" type="text" placeholder="Actor Name">
</div>
<div id="sheet3" class="application"><div class="crucible group"><input placeholder="Group Name"></div></div>
<div id="sheet4" class="application crucible item affix"><input placeholder="Affix Name"></div>
<div id="sheet5" class="application crucible action"><input placeholder="Action Name"></div>

<!-- ③ 创建页加减按钮：物品名那一段必须原样保留（它由 Babele 翻） -->
<div id="creation" class="application crucible crucible-fullscreen">
  <button aria-label="Remove one Steel Longsword"></button>
  <button aria-label="Add one Steel Longsword"></button>
</div>

<!-- ④ 负例区：别的模块的窗口，同名串一条都不许被吃 -->
<div id="other" class="application some-other-module">
  <input placeholder="Item Name">
  <input placeholder="Actor Name">
  <button aria-label="Add one Widget"></button>
  <span class="label">Special</span>
  <label class="tag-icon" data-tooltip="Reload"></label>
</div>

<script type="module">
${code}

const snap = (sel, what) => Array.from(document.querySelectorAll(sel)).map(el =>
  what === "text" ? el.textContent.trim() : el.getAttribute(what));

const before = {
  chatLabels: snap("#chat .label", "text"),
  chatTips: snap("#chat .tag-icon", "data-tooltip"),
  sheets: snap("#sheet input, #sheet2 input, #sheet3 input, #sheet4 input, #sheet5 input", "placeholder"),
  creation: snap("#creation button", "aria-label"),
  other: [...snap("#other input", "placeholder"), ...snap("#other .label", "text"),
          ...snap("#other .tag-icon", "data-tooltip"), ...snap("#other button", "aria-label")],
};

// 真身：像 renderChatMessageHTML / renderApplicationV2 那样，逐个根调用
const counts = [];
for (const id of ["chat", "sheet", "sheet2", "sheet3", "sheet4", "sheet5", "creation", "other"]) {
  counts.push({id, n: translateCrucibleRoot(document.getElementById(id))});
}
// 幂等：再跑一遍，应当一处都不改
const again = [];
for (const id of ["chat", "sheet", "sheet2", "sheet3", "sheet4", "sheet5", "creation", "other"]) {
  again.push({id, n: translateCrucibleRoot(document.getElementById(id))});
}

const after = {
  chatLabels: snap("#chat .label", "text"),
  chatTips: snap("#chat .tag-icon", "data-tooltip"),
  sheets: snap("#sheet input, #sheet2 input, #sheet3 input, #sheet4 input, #sheet5 input", "placeholder"),
  creation: snap("#creation button", "aria-label"),
  other: [...snap("#other input", "placeholder"), ...snap("#other .label", "text"),
          ...snap("#other .tag-icon", "data-tooltip"), ...snap("#other button", "aria-label")],
};
const result = {before, after, counts, again, stats: hardcodedStats()};
document.title = "DONE";
window.__RESULT__ = result;
console.log("CRUCIBLE_SCOPE_RESULT " + JSON.stringify(result));
</script>
</body>`;
fs.writeFileSync("scope.html", page, "utf8");
console.log("→ scope.html  (" + page.length + " 字符)");
