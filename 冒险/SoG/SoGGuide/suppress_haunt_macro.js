/*
压制作祟 Suppress Haunt

用法：在画布上选中作祟的 token 后运行本宏；没选中（或选中的不是危害）时，
会先弹出当前场景的危害清单让你挑。确认 DC / 动作数 / 技能后点「确定」，
宏会把一条内联检定发到聊天栏，玩家点它掷骰。

检定串带 action:suppress-haunt 骰点选项，物品「压制作祟」上的 4 条 Note 规则
据此把大成功／成功／失败／大失败的判定文字挂到掷骰结果下方 ——
所以那件物品要放在掷骰玩家的角色卡上，否则只会得到一次普通技能检定。
*/

// PF2e 等级基准 DC 表（-1 ~ 25 级）
const DCS = new Map([
  [-1, 13], [0, 14], [1, 15], [2, 16], [3, 18], [4, 19],
  [5, 20], [6, 22], [7, 23], [8, 24], [9, 26],
  [10, 27], [11, 28], [12, 30], [13, 31],
  [14, 32], [15, 34], [16, 35], [17, 36],
  [18, 38], [19, 39], [20, 40], [21, 42],
  [22, 44], [23, 46], [24, 48], [25, 50]
]);

// 动作数决定 DC 档位：1 动作极难、2 动作标准、3 动作简单
const TIERS = [
  { actions: 1, adjustment: 5, label: "极难" },
  { actions: 2, adjustment: 0, label: "标准" },
  { actions: 3, adjustment: -2, label: "简单" }
];

const DEFAULT_SKILL = "religion";
const isHazard = (a) => a?.type === "hazard";

/*
只认你亲手选中的那个危害。Foundry 在没有选中任何 token 时，会把宏作用域里的
token / actor 回退到你绑定的角色（见 macro.mjs 的 #executeScript），
沿用那个回退值会静默按 PC 的等级算出 DC。
*/
const controlled = canvas.tokens?.controlled ?? [];
const selected = controlled.length === 1 && isHazard(controlled[0].actor)
  ? controlled[0].actor
  : null;

if (selected) promptSuppress(selected);
else chooseHazard();

/* 列出当前场景上的全部危害。用 token id 回查，因为同名作祟经常不止一个。 */
function chooseHazard() {
  const tokens = (canvas.scene?.tokens.filter((t) => isHazard(t.actor)) ?? [])
    .sort((a, b) => a.name.localeCompare(b.name, "zh-Hans"));

  if (!tokens.length) {
    ui.notifications.warn("当前场景上没有危害类 token。");
    return;
  }

  const options = tokens.map((t) =>
    `<option value="${t.id}">${foundry.utils.escapeHTML(t.name)}（等级 ${t.actor.level}）</option>`
  ).join("");

  new foundry.applications.api.DialogV2({
    window: { title: "压制作祟" },
    content: `<p>选择危害</p>
  <select name="hazard" id="hazard" style="width:100%">${options}</select>`,
    buttons: [{
      action: "hazard",
      label: "确认选择",
      default: true,
      callback: (event, button) => button.form.elements.hazard.value
    }],
    submit: (id) => {
      const hazard = tokens.find((t) => t.id === id)?.actor;
      if (hazard) promptSuppress(hazard);
    }
  }).render(true);
}

function promptSuppress(hazard) {
  const level = Math.clamp(hazard.level, -1, 25);
  const dc = DCS.get(level);

  const tiers = TIERS.map((t) => `
  <label style="display:block;line-height:1.7">
    <input type="radio" name="actions" value="${t.actions}"${t.actions === 2 ? " checked" : ""}>
    ${t.actions} 动作 · ${t.label} DC（${t.adjustment >= 0 ? "+" : ""}${t.adjustment}）
  </label>`).join("");

  // 技能名直接取 pf2e 的本地化条目，由 pf2_cn 翻译；下拉框也杜绝了手打出无效 slug
  const skills = Object.entries(CONFIG.PF2E.skills)
    .map(([slug, s]) => [slug, game.i18n.localize(s.label)])
    .concat([["perception", game.i18n.localize("PF2E.PerceptionLabel")]])
    .sort((a, b) => a[1].localeCompare(b[1], "zh-Hans"))
    .map(([slug, label]) =>
      `<option value="${slug}"${slug === DEFAULT_SKILL ? " selected" : ""}>${label}</option>`)
    .join("");

  new foundry.applications.api.DialogV2({
    window: { title: `压制作祟：${hazard.name}` },
    content: `
  <p>基于等级的 DC（等级 ${level}）：</p>
  <input type="number" id="dc" name="dc" value="${dc}" style="width:100%">
  <p>动作数：</p>${tiers}
  <p>使用的技能：</p>
  <select id="skill" name="skill" style="width:100%">${skills}</select>`,
    buttons: [{
      action: "details",
      label: "确定",
      default: true,
      callback: (event, button) => ({
        skill: button.form.elements.skill.value,
        dc: button.form.elements.dc.value,
        actions: Number(button.form.elements.actions.value)
      })
    }],
    submit: (result) => postCheck(hazard, result)
  }).render(true);
}

// 同时把检定串打到控制台，同一场要反复发很多次时可以复制出来改参数直接贴聊天栏
function postCheck(hazard, { skill, dc, actions }) {
  const { adjustment } = TIERS.find((t) => t.actions === actions) ?? TIERS[1];
  const message = `@Check[${skill}|dc:${dc}|adjustment:${adjustment}|name:压制作祟|traits:haunt,hazard|options:action:suppress-haunt]`;
  console.log(message);
  // 只给 alias 不给 actor：内联检定会把 message.actor 当作掷骰者的回退，
  // 绑上危害的话，玩家没选中自己的 token 时点击就变成危害自己掷了。
  ChatMessage.create({
    content: message,
    speaker: { alias: hazard.name }
  });
}
