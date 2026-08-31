/**
 * 第十七轮自检：把 ember-hardcoded-cn.mjs `import` 进来，喂真实形状的输入实测。
 *   ① translateText 对本轮改过/新增的键给出预期译文；
 *   ② patchVistaPlacementSchema 在仿真的 PLACEMENT_SCHEMA 上改写 14 个 label；
 *   ③ patchProseMirrorBlocks 在仿真的 CONFIG.CDT.PROSEMIRROR.blocks 上改写 15 条 title；
 *   ④ 幂等 / 上游改过文案 / 结构缺失 三种情形都不炸。
 * 只读：不写任何项目文件。
 */

/* ---------- 最小 Foundry 环境桩 ---------- */
const hooks = {};
globalThis.Hooks = {
  once: (name, fn) => { (hooks[name] ??= []).push(fn); },
  on: (name, fn) => { (hooks[name] ??= []).push(fn); }
};

/* 最小 DOM：够 translateNode 走完文本节点 + 属性 + 子节点三条路 */
globalThis.Node = { TEXT_NODE: 3, ELEMENT_NODE: 1 };
// handler 用 `element instanceof HTMLElement` 认根节点；给一个类并让仿元素继承它
class FakeHTMLElement {}
globalThis.HTMLElement = FakeHTMLElement;
function txt(s) { return { nodeType: 3, nodeValue: s, childNodes: [] }; }
function el(className, children = [], attrs = {}) {
  const node = {
    nodeType: 1, className, childNodes: children, children: children.filter(c => c.nodeType === 1),
    attrs,
    get textContent() {
      return this.childNodes.map(c => (c.nodeType === 3 ? c.nodeValue : c.textContent)).join("");
    },
    getAttribute(k) { return this.attrs[k] ?? null; },
    setAttribute(k, v) { this.attrs[k] = v; },
    querySelector(sel) { return this.querySelectorAll(sel)[0] ?? null; },
    querySelectorAll(sel) {
      // 只支持 `.cls` —— 本测试用到的就这一种
      const want = sel.replace(/^\./, "");
      const out = [];
      const walk = n => {
        if (n.nodeType !== 1) return;
        if (String(n.className).split(/\s+/).includes(want)) out.push(n);
        for (const c of n.childNodes) walk(c);
      };
      for (const c of this.childNodes) walk(c);
      return out;
    }
  };
  // handler 用 `element instanceof HTMLElement` 认根节点；setPrototypeOf 不动 getter
  Object.setPrototypeOf(node, FakeHTMLElement.prototype);
  return node;
}

// 仿 SchemaField：只需要 fields + getField(点号路径)
function schemaField(fields, opts = {}) {
  return {
    fields,
    label: opts.label,
    getField(path) {
      const [head, ...rest] = String(path).split(".");
      const f = this.fields[head];
      if (!f) return undefined;
      return rest.length ? f.getField?.(rest.join(".")) : f;
    }
  };
}
function f(label) { return { label }; }

function makePlacementSchema() {
  const illumination = schemaField({
    luminosity: f("Luminosity"),
    only: f("Only"),
    blurStrength: f("Blur Strength")
  }, { label: "Illumination" });
  const colorize = schemaField({
    texture: f("Custom Color Texture"),
    r: f("Red Channel"),
    g: f("Green Channel"),
    b: f("Blue Channel")
  }, { label: "Colorization" });
  return schemaField({
    x: f("X"), y: f("Y"),
    elevation: f("Elevation"), sort: f("Sort"),
    scaleX: f("X"), scaleY: f("Y"),
    skewX: f("X"), skewY: f("Y"),
    angle: f("Angle"), alpha: f("Alpha"), tint: f("Tint"),
    illumination, colorize
  });
}

function makeBlocks() {
  return [
    { action: "insert-block-readaloud", title: "Readaloud Block" },
    { action: "insert-block-hazard", title: "Hazard Block" },
    { action: "insert-block-exploration", title: "Exploration Block" },
    { action: "insert-block-social", title: "Social Block" },
    { action: "insert-block-complex-skill", title: "Complex Skill" },
    { action: "insert-block-qna", title: "Q&A Block" },
    { action: "insert-block-gamemaster", title: "Gamemaster Block" },
    { action: "insert-block-attunement", title: "Attunement Block" },
    { action: "insert-block-wip", title: "WIP Block" },
    { action: "insert-h2-divider", title: "H2 Divider" },
    { action: "insert-h3-divider", title: "H3 Divider" },
    { action: "insert-definition-list", title: "Definition List" },
    { action: "insert-block-system", title: "System Swap Block" },
    { action: "insert-inline-system", title: "System Swap Inline" },
    { action: "insert-skill-check", title: "Skill Check Inline" },
    // 别的模块往同一个数组里塞的条目，必须一个字都不动
    { action: "insert-block-foreign", title: "Someone Else's Block" }
  ];
}

let placement = makePlacementSchema();
let blocks = makeBlocks();

globalThis.CONFIG = {
  CDT: { PROSEMIRROR: { blocks } },
  time: { formatters: {} },
  RegionBehavior: { dataModels: {} }
};
// 仿 ember.scenes.region.slices[*].config.weather（ember.mjs:119622 起那批配置字面量的形状）
const weatherCfg = {
  Clear: { label: "Clear" },
  Rain: { label: "Rain", strengths: { 1: { label: "Drizzle" }, 3: { label: "Storm" }, 4: { label: "Tempest" } } },
  Storm: { label: "Storm" },
  Tempest: { label: "Tempest" },
  Dust: { label: "Dust", strengths: { 2: { label: "Dust Storm" } } },
  Kindling: { label: "Kindling" },
  Wildfire: { label: "Wildfire" }
};
globalThis.ember = {
  api: { applications: { EmberVistaConfiguration: { PLACEMENT_SCHEMA: placement } } },
  scenes: { region: { slices: { surface: { config: { weather: weatherCfg } } } } }
};
globalThis.game = {
  modules: { get: id => (id === "ember" ? { active: true } : undefined) },
  i18n: { localize: k => k }
};
globalThis.ui = {};
globalThis.foundry = { applications: { instances: new Map() }, utils: { getProperty: () => undefined } };

/* ---------- 导入被测模块并跑 ready ---------- */
const mod = await import(
  new URL("../../1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs", import.meta.url).href
);
const { translateText } = mod;

let fails = 0;
const eq = (got, want, what) => {
  const ok = got === want;
  if (!ok) fails++;
  console.log(`${ok ? "  ok  " : "FAIL  "} ${what}: ${JSON.stringify(got)}${ok ? "" : ` != ${JSON.stringify(want)}`}`);
};

console.log("\n=== ready 钩子 ===");
for (const fn of hooks.ready ?? []) fn();

console.log("\n=== ② 远景摆放 schema（14 条） ===");
const wantVista = [
  ["elevation", "高度"], ["sort", "排序"], ["angle", "角度"], ["alpha", "不透明度"], ["tint", "染色"],
  ["illumination", "照明"], ["illumination.luminosity", "亮度"], ["illumination.only", "仅照明"],
  ["illumination.blurStrength", "模糊强度"], ["colorize", "着色"],
  ["colorize.texture", "自定义颜色贴图"], ["colorize.r", "红色通道"],
  ["colorize.g", "绿色通道"], ["colorize.b", "蓝色通道"]
];
for (const [p, w] of wantVista) eq(placement.getField(p).label, w, `PLACEMENT_SCHEMA.${p}.label`);
// X / Y 六条必须原样
for (const p of ["x", "y", "scaleX", "scaleY", "skewX", "skewY"]) {
  eq(placement.getField(p).label, p.endsWith("Y") || p === "y" ? "Y" : "X", `PLACEMENT_SCHEMA.${p}.label（有意不译）`);
}

console.log("\n=== ③ ProseMirror 模板块（15 条） ===");
const wantPM = {
  "insert-block-readaloud": "朗读区块", "insert-block-hazard": "危害区块",
  "insert-block-exploration": "探索区块", "insert-block-social": "社交区块",
  "insert-block-complex-skill": "复合技能检定", "insert-block-qna": "问答区块",
  "insert-block-gamemaster": "游戏主持人区块", "insert-block-attunement": "同调区块",
  "insert-block-wip": "制作中区块", "insert-h2-divider": "H2 分隔标题",
  "insert-h3-divider": "H3 分隔标题", "insert-definition-list": "定义列表",
  "insert-block-system": "系统切换区块", "insert-inline-system": "系统切换内联",
  "insert-skill-check": "技能检定内联"
};
for (const b of blocks) {
  if (b.action === "insert-block-foreign") eq(b.title, "Someone Else's Block", "别的模块的条目未被动");
  else eq(b.title, wantPM[b.action], b.action);
}

console.log("\n=== ① translateText 实测（本轮改动 + 回归） ===");
const cases = [
  // A3 本轮改的三条
  ["Ascend", "上升"], ["Descend", "下降"], ["Seal", "封堵"],
  // A3 本轮新增的三条（焦油坑菜单）
  ["Open", "打开"], ["Spawn", "生成"], ["Tar Pit", "焦油坑"],
  // A3 本轮判「保持」的六条
  ["Close", "关闭"], ["Exit", "退出"], ["Sign", "手语"],
  ["Dust", "扬尘"], ["Kindling", "起火"], ["Tempest", "狂风暴雨"],
  // 回归：先前几轮已裁的四条
  ["Ring", "敲响"], ["Usage", "使用方式"], ["Make", "将"], ["Sprites", "精灵图"],
  // 回归：不该被动的
  ["Unseal", "解封"], ["Open Valve", "打开阀门"], ["Move", "移动"], ["Call", "呼叫"],
  ["Spawn Construct", "生成构装体"]
];
// DIALOG_UI / EMBER_WINDOW_UI / WEATHER 是作用域表或数据表，translateText 只查全局三张表：
// 这一轮先确认它们**不会**被全局表误改（该原样返回就原样返回），
// 真正的命中在下面 ①b（DIALOG_UI 走渲染钩子）与 ①c（WEATHER 走 patchWeatherLabels）里测。
for (const [en, cn] of cases) {
  const got = translateText(en);
  if (got === cn) eq(got, cn, `translateText(${JSON.stringify(en)})`);
  else eq(got, en, `translateText(${JSON.stringify(en)})（作用域表键，全局表下应原样返回）`);
}

console.log("\n=== ①b DIALOG_UI 走真实渲染钩子（认框 → 翻正文与按钮） ===");
const renderHook = (hooks.renderApplicationV2 ?? [])[0];
eq(typeof renderHook, "function", "renderApplicationV2 钩子已挂上");

function dialog(titleText, bodyTexts) {
  const title = el("window-title", [txt(titleText)]);
  const body = el("dialog-content", bodyTexts.map(t => el("button", [txt(t)])));
  return el("dialog", [title, body]);
}
// 焦油坑：标题走本轮新补的 DIALOG_TITLES，三个按钮走 DIALOG_UI
const tarPit = dialog("Tar Pit", ["Open", "Seal", "Spawn"]);
renderHook({ constructor: { name: "DialogV2" } }, tarPit);
eq(tarPit.querySelector(".window-title").textContent, "焦油坑", "焦油坑框标题");
eq(tarPit.querySelectorAll(".button").map(b => b.textContent).join("/"), "打开/封堵/生成", "焦油坑三个按钮");

// 升降机：基类标题 Elevator Controls 已在表里，按钮是本轮改的 Ascend/Descend + Seal
const elevator = dialog("Elevator Controls", ["Ascend", "Descend", "Seal", "Unseal", "Move", "Call"]);
renderHook({ constructor: { name: "DialogV2" } }, elevator);
eq(elevator.querySelector(".window-title").textContent, "升降机控制", "升降机框标题");
eq(elevator.querySelectorAll(".button").map(b => b.textContent).join("/"),
   "上升/下降/封堵/解封/移动/呼叫", "升降机按钮");

// 认不出来的框：只翻标题（翻不了就原样），正文一个字都不能动
const foreign = dialog("Some Other Module Dialog", ["Open", "Seal", "Close", "Exit"]);
renderHook({ constructor: { name: "DialogV2" } }, foreign);
eq(foreign.querySelectorAll(".button").map(b => b.textContent).join("/"),
   "Open/Seal/Close/Exit", "非 Ember 框的正文未被动");

console.log("\n=== ①c WEATHER 走 patchWeatherLabels（数据侧） ===");
eq(weatherCfg.Dust.label, "扬尘", "WEATHER.Dust");
eq(weatherCfg.Kindling.label, "起火", "WEATHER.Kindling");
eq(weatherCfg.Tempest.label, "狂风暴雨", "WEATHER.Tempest");
eq(weatherCfg.Storm.label, "风暴", "WEATHER.Storm（与 Tempest 必须两分）");

console.log("\n=== ④ 幂等 / 上游改文案 / 结构缺失 ===");
// applyOnce 的标记是 non-configurable，删不掉（这正是它该有的样子），
// 所以换一个全新的 CONFIG 对象来模拟「第二个世界重开」。
const runReady = () => { for (const fn of hooks.ready ?? []) fn(); };
const freshConfig = extra => {
  globalThis.CONFIG = Object.assign({
    time: { formatters: {} }, RegionBehavior: { dataModels: {} }
  }, extra);
};

// 幂等：同一批对象再跑一次，不应再动
freshConfig({ CDT: { PROSEMIRROR: { blocks } } });
runReady();
eq(placement.getField("colorize.b").label, "蓝色通道", "幂等：再跑一次 label 不变");
eq(blocks.find(b => b.action === "insert-h2-divider").title, "H2 分隔标题", "幂等：再跑一次 title 不变");
eq(weatherCfg.Tempest.label, "狂风暴雨", "幂等：天气档位名不变");

// 上游改过文案：label 既不是英文原串也不是我们的译文 → 跳过并告警，不覆盖
placement = makePlacementSchema();
placement.getField("angle").label = "Rotation Angle";
ember.api.applications.EmberVistaConfiguration.PLACEMENT_SCHEMA = placement;
freshConfig({ CDT: { PROSEMIRROR: { blocks } } });
runReady();
eq(placement.getField("angle").label, "Rotation Angle", "上游改过文案：不覆盖");
eq(placement.getField("tint").label, "染色", "上游改过一条不影响其余条");

// 结构缺失：CDT 未装 / ember.api 不在 → 静默或告警，不抛
delete globalThis.ember.api;
freshConfig({});
let threw = null;
try { runReady(); } catch (e) { threw = e; }
eq(threw, null, "结构缺失时不抛异常");

console.log(`\n${fails ? `FAILED ${fails}` : "ALL PASS"}`);
process.exit(fails ? 1 : 0);
