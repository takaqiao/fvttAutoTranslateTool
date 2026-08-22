
/**
 * 腿部四姿势的基名。上游 `makeLegPoseParts()`（ember.mjs:59435）把 `${baseId}${suffix}` 拼成
 * 部件 id（19 个调用点 × 4 姿势），hulgrun 的腿另有 `id.replace("Bare", "Chiseled"|"Marbled")`
 * （:57251-57252）—— 拼出来的整串在上游文本里**没有字面量**，所以不进登记表，
 * 由下面 `deriveParts()` 用**同一条拼法**现拼。
 * ⚠ 这 25 个基名与 4 个姿势名**本身**都是上游字面量（`makeLegPoseParts("Smooth1", …)` /
 *   `LEG_POSES = {backward:"Backward", …}` :59421-59425），都在上面那张表里。
 * ⚠ 拼串**只**在这两族上做。对全表做笛卡尔积会造出几千个上游根本不产出的键 ——
 *   那是「建一张永远命不中的表」，比不建更糟（面板报绿而实际零效果）。
 */
const LEG_POSE_BASES = ["AedirHeavy", "AedirLight", "Armored", "BrokenAedirHeavy", "BrokenAedirLight",
  "BrokenArmored", "BrokenHeavy", "BrokenMetal1", "BrokenPointed", "BrokenSmooth1", "BrokenSmooth2",
  "Chiseled", "Hairy", "Heavy", "Marbled", "Metal1", "Pointed", "Smooth1", "Smooth2", "Spiky",
  "Straw", "Swirled", "Wood", "Wood1", "Woody"];
const LEG_POSE_SUFFIXES = ["Backward", "Forward", "Neutral", "Sitting"];
/** hulgrun 手部：`id.replace(/([^/]+)$/, "Marbled$1")`（ember.mjs:57239），前缀式 */
const MARBLED_HAND_BASES = ["Casual", "Down", "Fist", "Spell", "Up", "Vertical", "Weapon"];

/**
 * 把 id 拆成上屏显示名 —— **逐字符照抄** ember.mjs:64457 那一行，别改。
 * ⚠ 字符类是 `[A-Z1-9]`，**不含 `0`**：`Beard1`→`Beard 1` 而 `Beard10`→`Beard 10`、
 *   `Beard11`→`Beard 1 1`。这条怪癖是实测出来的，不是推的。
 */
const splitPartId = (id) => id.replace(/(?<!^)([A-Z1-9])/g, " $1");

/** 拼中文：基名译文以 ASCII 字母/数字收尾时补一个空格（「光滑 1」+「后撤」→「光滑 1 后撤」） */
const joinCn = (a, b) => (/[0-9A-Za-z]$/.test(a) ? `${a} ${b}` : `${a}${b}`);

/**
 * id 表 → 显示名表。表里 671 个 id 各算一次，再补上两族拼串（共 107 条）。
 * 拼不出来（基名不在表里）就跳过并 warn —— 静默跳过等于给自己留一个查不出的洞。
 */
const TOKEN_MAKER_PARTS = (() => {
  const out = {};
  for (const [id, cn] of Object.entries(TOKEN_MAKER_PART_IDS)) out[splitPartId(id)] = cn;
  let composed = 0, missing = 0;
  const add = (id, cn) => { const d = splitPartId(id); if (!(d in out)) { out[d] = cn; composed++; } };
  for (const base of LEG_POSE_BASES) {
    const b = TOKEN_MAKER_PART_IDS[base];
    if (!b) { missing++; continue; }
    for (const pose of LEG_POSE_SUFFIXES) {
      const p = TOKEN_MAKER_PART_IDS[pose];
      if (!p) { missing++; continue; }
      add(base + pose, joinCn(b, p));
    }
  }
  for (const base of MARBLED_HAND_BASES) {
    const b = TOKEN_MAKER_PART_IDS[base], m = TOKEN_MAKER_PART_IDS.Marbled;
    if (!b || !m) { missing++; continue; }
    add("Marbled" + base, joinCn(m, b));
  }
  if (missing) warn(`部件显示名拼串有 ${missing} 条缺基名，没拼出来。`);
  if (composed !== 107) warn(`部件显示名拼串实得 ${composed} 条，登记的是 107 条 —— 上游的部件族变了，去核一遍。`);
  return out;
})();

/**
 * 指示物制作器的**部件名行**与**计数行**。只对 `.choice.layer` 这一族选择器动手：
 * `.choice.build` / `.choice.stance` 是体格与姿态，那两行的 `Heavy` / `Lithe` 归 TOKEN_MAKER_UI
 * （壮硕 / 纤瘦），不能被本表的「粗壮 / 柔韧」改掉 —— 这正是本轮要修的那处错译。
 * 幂等：已经是中文的节点查不到键，原样返回。
 * @param {HTMLElement} root  指示物制作器窗口的根元素
 */
function translateTokenMakerParts(root) {
  let n = 0;
  for (const el of root.querySelectorAll?.(".token-maker-layers .choice.layer .part") ?? []) {
    if (el.children.length) continue;
    const raw = el.textContent.trim();
    const cn = TOKEN_MAKER_PARTS[raw];
    if (cn && cn !== raw) { el.textContent = cn; n++; }
  }
  // layers.hbs:35 `{{layer.chosenLabel}} of {{layer.choices.length}}` —— 整行一个文本节点，
  // `of` 是拼在模板里的，查表接不住。只认 `数字 of 数字` 这一种形状。
  for (const el of root.querySelectorAll?.(".token-maker-layers .choice.layer .count") ?? []) {
    if (el.children.length) continue;
    const m = /^(\d+) of (\d+)$/.exec(el.textContent.trim());
    if (m) { el.textContent = `${m[1]} / ${m[2]}`; n++; }
  }
  return n;
}
