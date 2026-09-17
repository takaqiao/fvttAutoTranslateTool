import { MODULE_ID, SOURCES, findFeature, hasSource, buildLegendEffect, buildAttunementEffect } from "./rules.mjs";

const DAILIES = "pf2e-dailies";
const registeredApis = new WeakSet();
const ENERGY_OPTIONS = Object.freeze([
  ["air", "气（挥砍）"], ["cold", "寒冷"], ["earth", "土（钝击）"],
  ["electricity", "电击"], ["fire", "火焰"], ["metal", "金（挥砍）"],
  ["poison", "毒素"], ["sonic", "音波"], ["vitality", "命能"],
  ["void", "虚能"], ["water", "水（钝击）"], ["wood", "木（穿刺）"],
]);
const values = actor => Array.isArray(actor?.items) ? actor.items : [...(actor?.items?.values?.() ?? [])];
const itemId = item => item.id ?? item._id;
const escape = text => String(text ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const keyFor = kind => `${MODULE_ID}-${kind}`;
const ownsFeature = (actor, key) => {
  const feature = actor?.type === "character" && findFeature(actor, key);
  return !!feature && feature.isSuppressed !== true && feature.system?.suppressed !== true;
};

/** Fixed mutable inline checks are the PF2e DC surface our AdjustModifier can affect. */
export function isDailyLegendItem(item) {
  if (!["weapon", "armor", "shield", "equipment", "backpack"].includes(item?.type)
    || item.isMagical !== true || (item.quantity ?? item.system?.quantity ?? 0) <= 0) return false;
  const description = item.system?.description?.value ?? "";
  return [...description.matchAll(/@Check\[([^\]]+)\]/g)].some(([, parameters]) => {
    const tokens = parameters.split("|").map(value => value.trim());
    return tokens.some(token => /^dc:\d+$/.test(token))
      && !tokens.some(token => token === "immutable" || /^immutable:(?:true|1)$/i.test(token));
  });
}

const existingEffects = (actor, kind) => values(actor).filter(item => item.type === "effect" && item.flags?.[MODULE_ID]?.kind === kind);

/** Queue through Dailies' batch helpers; never race its final actor update with a socket write. */
function stageEffect(options, kind, source) {
  const existing = existingEffects(options.actor, kind);
  if (!source) {
    for (const item of existing) options.deleteItem(item);
    return;
  }
  if (existing.length) {
    const [first, ...duplicates] = existing;
    options.updateItem({
      ...source,
      _id: itemId(first),
      flags: {
        ...first.flags,
        ...source.flags,
        [DAILIES]: { ...first.flags?.[DAILIES], temporary: true, daily: `module.${keyFor(kind)}` },
      },
    });
    for (const item of duplicates) options.deleteItem(item);
  } else options.addItem(source, true);
}

function rest(kind, { actor, removeItem }) {
  for (const item of existingEffects(actor, kind)) {
    // Dailies cleanup already deletes every temporary item, even after feat removal.
    // Only add legacy panel-created effects here to avoid duplicate deletion IDs.
    if (item.flags?.[DAILIES]?.temporary !== true) removeItem(itemId(item));
  }
}

function requireFeature(actor, key) {
  if (!ownsFeature(actor, key)) throw new Error("角色没有可用的对应专长。");
}

export function createThirdPartyDailies() {
  return [
    {
      key: keyFor("legend"),
      label: "魔法物品传奇",
      condition: actor => ownsFeature(actor, "legend"),
      rows: actor => {
        const options = [
          { value: "", label: "不选择", skipUnique: true },
          ...values(actor).filter(isDailyLegendItem).map(item => ({ value: itemId(item), label: item.name })),
        ];
        return ["item1", "item2"].map((slug, index) => ({
          type: "select", slug, label: `第${index + 1}件物品`, save: true, empty: true,
          unique: "legend-items", options: options.map(option => ({ ...option })),
        }));
      },
      process: options => {
        const { actor, rows, messages } = options;
        requireFeature(actor, "legend");
        const selected = [rows.item1, rows.item2].filter(value => value !== "" && value != null);
        if (selected.some(id => typeof id !== "string")) throw new Error("请选择有效的魔法物品。");
        if (new Set(selected).size !== selected.length) throw new Error("两次选择不能重复同一件物品。");
        const items = selected.map(id => values(actor).find(item => itemId(item) === id));
        if (items.some(item => !isDailyLegendItem(item))) throw new Error("所选物品已不可用，或不含可调整的固定启动DC。");
        // Validate before any queued delete/update so a stale UI cannot erase a valid preparation.
        const source = selected.length ? buildLegendEffect(selected) : null;
        stageEffect(options, "legend", source);
        if (selected.length) messages.add("third-party", { label: "魔法物品传奇", selected: items.map(item => escape(item.name)).join("、") });
        else messages.add("third-party", { label: "魔法物品传奇：本日不选择物品" });
      },
      rest: options => rest("legend", options),
    },
    {
      key: keyFor("attunement"),
      label: "能量调谐",
      condition: actor => ownsFeature(actor, "attunement"),
      rows: () => [{
        type: "select", slug: "trait", label: "调谐特征", save: true,
        options: ENERGY_OPTIONS.map(([value, label]) => ({ value, label })),
      }],
      process: options => {
        const { actor, rows, messages } = options;
        requireFeature(actor, "attunement");
        const spells = values(actor).filter(item => item.type === "spell" && hasSource(item, SOURCES.swordQiSpell));
        const source = buildAttunementEffect(rows.trait, spells.map(itemId));
        stageEffect(options, "attunement", source);
        messages.add("third-party", { label: "能量调谐", selected: ENERGY_OPTIONS.find(([value]) => value === rows.trait)[1] });
      },
      rest: options => rest("attunement", options),
    },
  ];
}

/** Call after init (setup or ready). No world settings or upstream code are modified. */
export function registerThirdPartyDailies(game) {
  if (!game?.modules?.get(DAILIES)?.active) return { status: "inactive" };
  const api = game.dailies?.api;
  // Verified in the 4.20.0 bundle and its source map. There is no registerDailies alias.
  if (typeof api?.registerCustomDailies !== "function") return { status: "unavailable" };
  if (registeredApis.has(api)) return { status: "already-registered" };
  const dailies = createThirdPartyDailies();
  api.registerCustomDailies(dailies);
  registeredApis.add(api);
  return { status: "registered", method: "registerCustomDailies", keys: dailies.map(daily => daily.key) };
}
