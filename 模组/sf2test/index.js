var j = Object.defineProperty;
var l = (o, e) => j(o, "name", { value: e, configurable: !0 });
const f = { id: "sf2e-murder-in-metal-city", title: "Murder in Metal City", cssClass: "metalcity", version: "13.2.0" }, $ = { id: "sf2e-murder-in-metal-city", shortTitle: "Murder in Metal City" };
Promise.resolve();
Hooks.once("init", () => {
});
const { StringField: N } = foundry.data.fields;
Hooks.once("init", () => {
  game.settings.register(f.id, "migrationVersion", {
    type: new N({
      initial: "",
      blank: !0,
      label: "Migration Version",
      hint: "Current migration version of this module."
    }),
    scope: "world",
    config: !1
  });
  const o = new L();
  foundry.utils.setProperty(CONFIG, `METAMORPHIC.${f.id}.migration`, o), Hooks.once("ready", () => {
    o.performMigration();
  });
});
class L {
  static {
    l(this, "Migration");
  }
  /**
   * The version of the module when migration was last performed.
   * @type {string}
   */
  get migrationVersion() {
    return game.settings.get(f.id, "migrationVersion");
  }
  /* -------------------------------------------------- */
  /**
   * The current migration version in the module. If the stored setting
   * is lower than this, a migration is needed. If this value is `null`,
   * this module has no relevant migrations.
   * @type {string|null}
   */
  get moduleMigrationVersion() {
    const e = f.id, t = game.modules.get(e).flags?.[e]?.migration?.version;
    return t || null;
  }
  /* -------------------------------------------------- */
  /**
   * A module requires migration if the module has a migration version
   * flag higher than the stored setting. If the module stores a method
   * for additional checking, this must also return anything but `false`.
   * @type {boolean}
   */
  get needsMigration() {
    const e = this.moduleMigrationVersion;
    if (!e)
      return !1;
    const t = this.migrationVersion;
    return !t && this.#t instanceof Function && this.#t.call(this) === !1 ? !1 : !t || foundry.utils.isNewerVersion(e, t);
  }
  /* -------------------------------------------------- */
  /**
   * Set the current migration version in this world.
   */
  #n() {
    const e = game.modules.get(f.id).version;
    game.settings.set(f.id, "migrationVersion", e);
  }
  /* -------------------------------------------------- */
  /**
   * Perform the actual migration.
   */
  async performMigration() {
    if (game.user.isActiveGM) {
      if (!this.needsMigration) {
        this.#n();
        return;
      }
      if (this.#e instanceof Function) {
        const e = { permanent: !0 }, t = ui.notifications.info(`Performing migration for ${f.title}. Please wait...`, e);
        await this.#e.call(this), ui.notifications.info(`Migration for ${f.title} completed!`, e), t.remove();
      }
      this.#n();
    }
  }
  /* -------------------------------------------------- */
  /**
   * The migration script. A module will assign its method here, which should accept
   * no parameters, and return a promise.
   * @type {Function|null}
   */
  #e = null;
  get script() {
    return this.#e;
  }
  set script(e) {
    this.#e = e;
  }
  /* -------------------------------------------------- */
  /**
   * A method used to conditionally negate the execution of a migration script. Modules should
   * use this to determine if a migration should be run despite any stored setting, or lack
   * thereof. Useful for migrations that are only recently introduced, if this explicitly
   * returns `false`, migration will not be performed. The method will receive no parameters.
   * @type {Function|null}
   */
  #t = null;
  get condition() {
    return this.#t;
  }
  set condition(e) {
    this.#t = e;
  }
}
const R = { "sf2e-murder-in-metal-city": { label: "Murder in Metal City Ring", spritesheet: "modules/sf2e-murder-in-metal-city/assets/ring/rings.json", effects: { RING_PULSE: "TOKEN.RING.EFFECTS.RING_PULSE", RING_GRADIENT: "TOKEN.RING.EFFECTS.RING_GRADIENT", BKG_WAVE: "TOKEN.RING.EFFECTS.BKG_WAVE", INVISIBILITY: "TOKEN.RING.EFFECTS.INVISIBILITY" } } };
Hooks.once("initializeDynamicTokenRingConfig", (o) => {
  for (const [e, t] of Object.entries(R)) {
    const n = new foundry.canvas.placeables.tokens.DynamicRingData(t);
    o.addConfig(e, n);
  }
});
const { JournalEntrySheet: O } = foundry.applications.sheets.journal;
class D extends O {
  static {
    l(this, "MetaMorphicJournalSheet");
  }
  /** @inheritdoc */
  _initializeApplicationOptions(e) {
    const t = super._initializeApplicationOptions(e);
    t.classes ??= [];
    const n = [
      "metamorphic",
      "themed",
      `${f.cssClass}-wrapper`
    ].filter((i) => i);
    return n.push("theme-light"), M.properties.distractionFree && game.settings.get(f.id, "distraction-free") && n.push("distraction-free"), t.classes.push(...n), t;
  }
  /* -------------------------------------------------- */
  /** @inheritdoc */
  _preparePageData() {
    const e = super._preparePageData();
    let t = 1;
    for (const [n, i] of Object.entries(e)) {
      const a = this.document.pages.get(n), { pageNumber: s, pageNumberClass: r, editable: c, additionalCssClasses: d } = a.flags?.metamorphic ?? {};
      s ? (i.number = s, Number.isNumeric(s) && (t = Number(i.number) + 1)) : i.number = t++, r && (i.tocClass = [i.tocClass, r].filterJoin(" ")), i.editable = i.editable && !!c, i.viewClass = Array.from(new Set(i.viewClass.split(" ").concat(d).filter((u) => u))).join(" ");
    }
    return e;
  }
  /* -------------------------------------------------- */
  /** @inheritdoc */
  async _renderHeadings(e, t) {
    return Object.entries(t || {}).forEach(([n, i]) => {
      i.element?.classList.contains("no-toc") && delete t[n];
      const a = i.element?.querySelectorAll("span") ?? [];
      a.length > 0 && (i.text = a[0].textContent);
    }), super._renderHeadings(e, t);
  }
  /* -------------------------------------------------- */
  /** @inheritdoc */
  async _renderFrame(e) {
    const t = await super._renderFrame(e), n = [
      "decoration",
      M.properties?.frame ? "frame" : null
    ].filter((i) => i);
    for (const i of n)
      for (const a of ["top-left", "top-right", "bottom-left", "bottom-right"]) {
        const s = [i, a].join(" ");
        t.insertAdjacentHTML("afterbegin", `<div class="${s}"></div>`);
      }
    return M.properties?.accessories && (t.insertAdjacentHTML("afterbegin", "<div class='accessory accessory-1'></div>"), t.insertAdjacentHTML("afterbegin", "<div class='accessory accessory-2'></div>")), t;
  }
  /* -------------------------------------------------- */
  /** @inheritdoc */
  async _onRender(e, t) {
    await super._onRender(e, t);
    const n = this.document.flags?.metamorphic?.additionalCssClasses ?? [];
    for (const i of n)
      this.element.classList.add(i);
    if ("scrollTag" in t) {
      const i = this.element.querySelector(`[data-scroll="${t.scrollTag}"]`);
      i && i.scrollIntoView();
    }
    this.#e();
  }
  /* -------------------------------------------------- */
  /** @inheritdoc */
  async _renderHTML(e, t) {
    const n = await super._renderHTML(e, t);
    return "pages" in n && await this.#n(n.pages), n;
  }
  /* -------------------------------------------------- */
  /**
   * Inject ToC navigation if given flags are provided on the journal entry.
   * @param {HTMLElement} element   The `pages` part of the sheet.
   * @returns {Promise<void>}       A promise that resolves once an element has been injected.
   */
  async #n(e) {
    const { backward: t, forward: n } = this.document.getFlag(f.id, "navigation") ?? {};
    if (!t && !n)
      return;
    const i = e.querySelector(".journal-entry-content .journal-header"), [a, s] = await Promise.all([t, n].map((c) => fromUuid(c))), r = e.ownerDocument.createElement("NAV");
    if (r.classList.add("toc-navigation"), a) {
      const c = e.ownerDocument.createElement("A");
      c.innerHTML = `<i class="fa-solid fa-arrow-left"></i> ${a.name}`, c.classList.add("backward"), c.dataset.tooltipDirection = "LEFT", c.dataset.tooltip = "Previous", c.dataset.uuid = a.uuid, r.insertAdjacentElement("afterbegin", c);
    }
    if (s) {
      const c = e.ownerDocument.createElement("A");
      c.innerHTML = `${s.name} <i class="fa-solid fa-arrow-right"></i>`, c.classList.add("forward"), c.dataset.tooltipDirection = "RIGHT", c.dataset.tooltip = "Next", c.dataset.uuid = s.uuid, r.insertAdjacentElement("beforeend", c);
    }
    for (const c of r.querySelectorAll(".backward, .forward"))
      c.addEventListener("click", async (d) => {
        const u = d.currentTarget.dataset.uuid;
        (await fromUuid(u))?.sheet.render(!0);
      });
    i.insertAdjacentElement("afterend", r);
  }
  /* -------------------------------------------------- */
  /**
   * Apply additional click event listeners and modifications.
   */
  #e() {
    for (const t of this.element.querySelectorAll(".read-aloud"))
      t.addEventListener("click", this.#i.bind(this));
    for (const t of this.element.querySelectorAll("a.content-link[data-type=Scene]"))
      t.addEventListener("click", this.#t.bind(this));
    const e = this.document.flags.metamorphic?.variations ?? [];
    if (e.length)
      for (const t of this.element.querySelectorAll("[data-option][data-variation]")) {
        const { option: n, variation: i } = t.dataset, a = e.find((s) => s.name === i)?.option;
        a && n !== a && (t.style.display = "none");
      }
  }
  /* -------------------------------------------------- */
  /*   Event Handlers                                   */
  /* -------------------------------------------------- */
  /**
   * When clicking a content link for a Scene document, if the user is
   * a GM, view the scene and render its journal (if any), instead of
   * rendering the scene's sheet.
   * @param {PointerEvent} event    Initiating click event.
   */
  #t(e) {
    e.preventDefault();
    const t = e.currentTarget;
    if (!t.dataset.uuid.startsWith("Scene"))
      return;
    const n = game.scenes.get(t.dataset.id);
    n && (e.stopPropagation(), n.view(), n.journal?.sheet.render(!0, { pageId: n.journalEntryPage }));
  }
  /* -------------------------------------------------- */
  /**
   * @param {PointerEvent} event    Initiating click event.
   */
  #i(e) {
    if (e.preventDefault(), ["IMG", "A"].includes(e.target.tagName))
      return;
    const t = this.document, { speakerId: n, speakerClasses: i } = e.currentTarget.dataset, a = `<div data-metamorphic-chatable class="${i || ""}">
      ${e.currentTarget.innerHTML}
    </div>`;
    this.#s.call(t, { content: a, speakerId: n });
  }
  /* -------------------------------------------------- */
  /**
   * Initiate a dialog prompt to configure which users to whisper a message to.
   * @this {JournalEntry}
   * @param {SpeakAloudMessageOptions} options    Speak-aloud options.
   */
  async #s(e) {
    if (!this.isOwner) {
      ui.notifications.error("JOURNAL.ShowBadPermissions", { localize: !0 });
      return;
    }
    const { BooleanField: t, SetField: n, StringField: i } = foundry.data.fields, a = new t().toFormGroup({
      label: game.i18n.localize("OWNERSHIP.AllPlayers"),
      hint: "Whisper to all players rather than a select few.",
      rootId: foundry.utils.randomID()
    }, { name: "all", value: !0 }).outerHTML, s = new n(new i()).toFormGroup({ label: "Players", classes: ["stacked"] }, {
      name: "players",
      type: "checkboxes",
      options: game.users.reduce((p, h) => (h === game.user || p.push({ value: h.id, label: h.name, selected: h.active }), p), []),
      disabled: !0
    }).outerHTML, r = `
    <fieldset>
      <legend>${game.i18n.localize("JOURNAL.ShowTo")}</legend>
      ${a}
      ${s}
    </fieldset>`, c = /* @__PURE__ */ l((p, h) => {
      const y = h.element, w = y.querySelector("[name=all]"), b = y.querySelector("[name=players]");
      w.addEventListener("change", (A) => b.disabled = w.checked);
    }, "render"), d = await foundry.applications.api.Dialog.input({
      render: c,
      ok: { label: "Whisper", icon: "fa-solid fa-check" },
      content: r,
      position: { width: 400, height: "auto" },
      window: { title: "Speak Aloud Message", icon: "fa-solid fa-comment" }
    });
    if (!d)
      return;
    const u = d.all || !d.players.length ? [] : d.players, g = game.actors.get(e.speakerId) ?? void 0;
    ChatMessage.implementation.create({
      whisper: u,
      content: e.content,
      speaker: g ? ChatMessage.implementation.getSpeaker({ actor: g }) : void 0
    });
  }
}
const M = {
  label: "Murder in Metal City Journal",
  themes: void 0,
  properties: {}
};
Hooks.once("init", () => {
  Object.defineProperty(D, "name", { value: "MetaMorphicJournalEntrySheet" });
  const { JournalEntry: o } = foundry.documents, { DocumentSheetConfig: e } = foundry.applications.apps, t = {
    makeDefault: !1,
    canBeDefault: !1,
    canConfigure: !0,
    label: M.label,
    themes: M.themes
  };
  e.registerSheet(o, f.id, D, t), game.settings.register(f.id, "distraction-free", {
    type: new foundry.data.fields.BooleanField(),
    name: "Distraction Free Mode",
    hint: "Disables animations and other distracting styling on Journal Entries of this module.",
    config: !0,
    scope: "client",
    requiresReload: !0,
    default: !1
  });
});
async function _(o) {
  if (![0, 0.4, 1].includes(o)) {
    const e = {
      default: { alpha: 1, label: "Default Roofs" },
      transparent: { alpha: 0.4, label: "Transparent" },
      disabled: { alpha: 0, label: "Disabled" }
    }, t = Object.entries(e).map(([i, { label: a }]) => ({ action: i, label: a })), n = await foundry.applications.api.Dialog.wait({
      buttons: t,
      window: { title: "Change Roof Behavior" },
      content: "<p>Change the default opacity of roofs across all relevant scenes.</p>"
    });
    if (!n)
      return;
    o = e[n].alpha;
  }
  for (const e of game.scenes) {
    const t = e.tiles.filter((n) => n.getFlag("world", "isRoof")).map((n) => ({ _id: n.id, alpha: o }));
    await e.updateEmbeddedDocuments("Tile", t);
  }
  ui.notifications.info("Updated transparency of all scenes' roof tiles!");
}
l(_, "changeRoofOpacity");
async function F(o = [], e = {}) {
  for (const t of o)
    t.options ??= {}, foundry.utils.mergeObject(t.options, e, { overwrite: !1 }), await H(t);
}
l(F, "changeTokens");
async function H({ sceneId: o, tokenId: e, updates: t = {}, options: n = {} }) {
  const i = game.scenes.get(o);
  if (!i) {
    ui.notifications.error(`The scene [${o}] does not exist!`);
    return;
  }
  const a = i.tokens.get(e);
  if (!a) {
    ui.notifications.error(`The token [${e}] does not exist!`);
    return;
  }
  const { actor: s, token: r } = t;
  !foundry.utils.isEmpty(s) && !a.actor && ui.notifications.error(`Unable to perform an update to the actor of token [${e}].`), await a.actor?.update(s, n), await a.update(r, n), t.waypoints?.length && await a.move(t.waypoints);
}
l(H, "_changeToken");
function I(o, e, { fallback: t = 0 } = {}) {
  const n = { _id: o.id }, i = o.toObject();
  for (const a in e) {
    const s = e[a];
    if (!Array.isArray(s) || ![1, 2].includes(s.length))
      continue;
    const r = foundry.utils.getProperty(i, a);
    let c;
    s.length === 1 ? c = s[0] : s[0] === r ? c = s[1] : s[1] === r ? c = s[0] : Number.isInteger(t) && t in s && (c = s[t]), c !== void 0 && (n[a] = c);
  }
  return n;
}
l(I, "createUpdateFromModification");
async function U(o, e = {}, { fallback: t = 0 } = {}) {
  const n = game.scenes.get(o);
  if (!n) {
    ui.notifications.error(`The scene [${o}] does not exist.`);
    return;
  }
  const i = I(n, e, { fallback: t });
  return "playlistSound" in i && n.playlistSound && await n.playlistSound.parent.stopSound(n.playlistSound), n.update(i);
}
l(U, "modifyScene");
async function G(o = [], e = {}) {
  const t = {
    window: {
      title: "Execute Macro"
    },
    content: "",
    position: {
      height: "auto",
      width: 420
    },
    ok: { label: "Close", icon: "fa-solid fa-times" },
    actions: {
      executeMacro: /* @__PURE__ */ l((a, s) => {
        const r = s.dataset.id;
        game.macros.get(r).execute();
      }, "executeMacro")
    }
  };
  e = foundry.utils.mergeObject(t, e);
  const n = /* @__PURE__ */ new Map();
  n.set("", []);
  for (const a of o) {
    const s = game.macros.get(a.id);
    if (!s) {
      console.warn(`Unable to find macro with id [${a.id}].`);
      continue;
    }
    const r = a.group ?? "";
    n.get(r) || n.set(r, []), n.get(r).push({
      macro: s,
      label: a.label ? a.label : s.name
    });
  }
  const i = document.createElement("DIV");
  e.content && i.insertAdjacentHTML("afterbegin", `<p class="hint">${e.content}</p>`);
  for (const [a, s] of n.entries())
    if (a !== "") {
      const r = document.createElement("FIELDSET");
      r.classList.add("flexcol"), r.insertAdjacentHTML("afterbegin", `<legend>${a}</legend>`);
      for (const { macro: c, label: d } of s) {
        const u = document.createElement("BUTTON");
        u.type = "button", u.textContent = d, u.dataset.action = "executeMacro", u.dataset.id = c.id, r.insertAdjacentElement("beforeend", u);
      }
      i.insertAdjacentElement("beforeend", r);
    } else
      for (const { macro: r, label: c } of s) {
        const d = document.createElement("BUTTON");
        d.type = "button", d.textContent = c, d.dataset.action = "executeMacro", d.dataset.id = r.id, i.insertAdjacentElement("beforeend", d);
      }
  return e.content = i, foundry.applications.api.Dialog.prompt(e);
}
l(G, "pickMacro");
async function z(o, e, t, n = {}) {
  const i = game.scenes.get(o);
  if (!i) {
    ui.notifications.error(`The scene [${o}] does not exist.`);
    return;
  }
  const a = i.tiles.get(e);
  if (!a) {
    ui.notifications.error(`The tile [${e}] does not exist.`);
    return;
  }
  const s = document.createElement("DIV");
  n.content && s.insertAdjacentHTML("afterbegin", `<p class="hint">${n.content}</p>`), n.content = s, n.window?.title || foundry.utils.setProperty(n, "window.title", "Change Tile Image");
  const r = /* @__PURE__ */ new Map();
  s.insertAdjacentHTML("beforeend", '<div class="form-footer metamorphic-custom-buttons"></div>');
  for (const { filepath: p, thumbnail: h, label: y } of t) {
    if (!p || !await foundry.utils.srcExists(p))
      continue;
    const w = foundry.utils.randomID(), b = document.createElement("BUTTON");
    b.type = "button", b.dataset.action = "updateTileImage", b.dataset.filepath = w, r.set(w, p), h && await foundry.utils.srcExists(h) && b.insertAdjacentHTML("afterbegin", `<img src="${h}">`), y && b.insertAdjacentHTML("beforeend", `<span>${y}</span>`), s.querySelector(".metamorphic-custom-buttons").insertAdjacentElement("beforeend", b);
  }
  const c = 250, u = Math.ceil(r.size / 10), g = document.createElement("STYLE");
  return g.textContent = `
  .metamorphic-custom-buttons {
    display: grid;
    grid-template-columns: repeat(${u}, 1fr);

    button {
      display: flex;
      flex-direction: column;
      padding: 0;
      max-height: 100px;
      width: ${c}px;
      overflow: hidden;

      &:has(img) {
        height: 100px;
      }

      &:has(span) {
        padding-bottom: 3px;
      }

      img {
        object-fit: cover;
        overflow: hidden;
        width: 100%;
      }
    }
  }
  `, s.insertAdjacentElement("afterbegin", g), n.actions ??= {}, n.actions.updateTileImage = function(p, h) {
    const y = h.dataset.filepath, w = r.get(y);
    a.update({ "texture.src": w });
  }, foundry.utils.mergeObject(n, {
    ok: { label: "Close", icon: "fa-solid fa-times" },
    position: { width: "auto" }
  }, { overwrite: !1 }), foundry.applications.api.Dialog.prompt(n);
}
l(z, "pickTileImage");
async function J(o, { state: e, stopAll: t = !1 } = {}) {
  if (t)
    for (const i of game.playlists.playing)
      await i.stopAll();
  const n = await fromUuid(o);
  if (n) {
    if (e ??= !n.playing, n.documentName === "PlaylistSound")
      return e ? n.parent.playSound(n) : n.parent.stopSound(n);
    if (n.documentName === "Playlist")
      return e ? n.playAll() : n.stopAll();
  }
}
l(J, "playSound");
async function B(o, e, t, { state: n } = {}) {
  const i = game.scenes.get(o);
  if (!i) {
    ui.notifications.error(`The scene [${o}] does not exist.`);
    return;
  }
  if (!["Tile", "Token", "AmbientLight", "AmbientSound"].includes(t))
    throw new Error(`Invalid documentName '${t}' for helper macro.`);
  const a = e.map((c) => i.getEmbeddedDocument(t, c)).filter((c) => c);
  if (!a.length)
    return;
  const s = {
    hidden: [!0, !1].includes(n) ? [n] : [!0, !1]
  }, r = a.map((c) => I(c, s));
  return i.updateEmbeddedDocuments(t, r);
}
l(B, "toggleHiddenState");
async function V(o, e, { state: t } = {}) {
  const n = game.scenes.get(o);
  if (!n) {
    ui.notifications.error(`The scene [${o}] does not exist.`);
    return;
  }
  const i = e.map((s) => n.getEmbeddedDocument("Region", s)).filter((s) => s);
  if (!i.length)
    return;
  const a = { disabled: [!0, !1].includes(t) ? [t] : [!0, !1] };
  for (const s of i) {
    const r = s.behaviors.map((c) => I(c, a));
    s.updateEmbeddedDocuments("RegionBehavior", r);
  }
}
l(V, "toggleRegionBehaviors");
async function W(o, e, t, n, { fallback: i = 0, animate: a } = {}) {
  const s = game.scenes.get(o);
  if (!s) {
    ui.notifications.error(`The scene [${o}] does not exist.`);
    return;
  }
  const r = [], c = e.map((d) => s.getEmbeddedDocument(t, d)).filter((d) => d);
  if (c.length) {
    for (const d of c) {
      const u = I(d, n, { fallback: i });
      r.push(u);
    }
    await s.updateEmbeddedDocuments(t, r, { animate: a });
  }
}
l(W, "toggleSceneEmbedded");
async function q(o, e, { state: t } = {}) {
  const n = game.scenes.get(o);
  if (!n)
    throw new Error(`The scene by id '${o}' does not exist!`);
  const i = e.map((s) => n.walls.get(s)).filter((s) => s);
  if (!i.length)
    throw new Error("No valid ids for wall documents were provided!");
  const a = i.map((s) => Y(s, t));
  n.updateEmbeddedDocuments("Wall", a);
}
l(q, "toggleTemporaryWalls");
function Y(o, e) {
  const t = CONST.WALL_RESTRICTION_TYPES, n = { _id: o.id }, i = o.flags.metamorphic;
  if (!(![!0, !1].includes(e) || i && e || !i && !e))
    return n;
  if (i) {
    for (const s of t)
      n[s] = i[s];
    n["flags.-=metamorphic"] = null;
  } else {
    const s = o.toObject();
    for (const r of t)
      n[r] = CONST.WALL_SENSE_TYPES.NONE, n[`flags.metamorphic.${r}`] = s[r];
  }
  return foundry.utils.expandObject(n);
}
l(Y, "_craftWallUpdate");
class K {
  static {
    l(this, "TokenPlacement");
  }
  constructor(e, t = []) {
    this.#n = e, this.#e = t;
  }
  /* -------------------------------------------------- */
  /**
   * The center point around which to place tokens.
   * @type {{ x: number, y: number }}
   */
  #n;
  /* -------------------------------------------------- */
  /**
   * The actors whose tokens to place.
   * @type {foundry.documents.Actor[]}
   */
  #e;
  /* -------------------------------------------------- */
  /**
   * Current multiplying value, often increased and reset for each actor's token.
   * @type {number}
   */
  #t = 1;
  /* -------------------------------------------------- */
  /**
   * Stored locations at where the tokens will be placed.
   * @type {Map<string, { x: number, y: number }|null>}
   */
  #i = /* @__PURE__ */ new Map();
  /* -------------------------------------------------- */
  /**
   * Occupied center points as hashes.
   * @type {Set<string>}
   */
  #s = /* @__PURE__ */ new Set();
  /* -------------------------------------------------- */
  /**
   * Place actors around an origin.
   * @param {{ x: number, y: number }} origin               The origin around which to place tokens.
   * @param {foundry.documents.Actor[]} [actors=[]]         The actors whose tokens to place.
   * @returns {Promise<foundry.documents.TokenDocument[]}   A promise that resolves to the created tokens.
   */
  static async place(e, t = []) {
    return new this(e, t).place();
  }
  /* -------------------------------------------------- */
  /**
   * Place the actors' tokens on the canvas.
   * @returns {Promise<foundry.documents.TokenDocument[]}   A promise that resolves to the created tokens.
   */
  async place() {
    this.#i.clear();
    for (const n of this.#e) {
      const i = this.#o(n);
      this.#i.set(n.uuid, i);
    }
    const { tokenData: e, invalid: t } = await Array.from(this.#i.entries()).reduce(async (n, [i, a]) => {
      if (n = await n, !a)
        n.invalid++;
      else {
        const s = await fromUuid(i).then((r) => r.getTokenDocument(a));
        n.tokenData.push(s.toObject());
      }
      return n;
    }, { tokenData: [], invalid: 0 });
    return e.length ? (t && ui.notifications.warn(`Skipped ${t} tokens as no suitable location was found.`), canvas.scene.createEmbeddedDocuments("Token", e)) : (ui.notifications.error("Unable to find any suitable locations to place down tokens."), []);
  }
  /* -------------------------------------------------- */
  /**
   * Find a suitable location to place down a token for an actor.
   * @param {foundry.documents.Actor} actor     The actor whose token to place.
   * @returns {{ x: number, y: number }|null}   A suitable location.
   */
  #o(e) {
    const t = {
      width: e.prototypeToken.width,
      height: e.prototypeToken.height
    }, n = {
      width: Math.max(1, t.width),
      height: Math.max(1, t.height)
    }, i = n.height * n.width, a = new PIXI.Rectangle(0, 0, n.width * canvas.grid.size, n.height * canvas.grid.size);
    for (this.#t = 0; this.#t < 50; ) {
      this.#t++;
      const s = this.#a();
      for (const r of s) {
        const { x: c, y: d } = canvas.grid.getTopLeftPoint(r);
        a.x = c, a.y = d;
        const u = s.filter((g) => a.contains(g.x, g.y));
        if (u.length === i) {
          for (const g of u)
            this.#s.add(`${g.x}:${g.y}`);
          return {
            x: t.width < 1 ? a.x + canvas.grid.sizeX / 4 : a.x,
            y: t.height < 1 ? a.y + canvas.grid.sizeY / 4 : a.y
          };
        }
      }
    }
    return null;
  }
  /* -------------------------------------------------- */
  /**
   * Find available points within the sweep.
   * @returns {{ x: number, y: number }[]}    The unoccupied center points within the sweep.
   */
  #a() {
    const e = CONFIG.Canvas.polygonBackends.move.create(this.#n, {
      angle: 360,
      hasLimitedAngle: !0,
      hasLimitedRadius: !0,
      radius: canvas.grid.size * this.#t,
      rotation: 0,
      type: "move",
      useThreshold: !1
    }), [t, n, i, a] = canvas.grid.getOffsetRange(e.bounds), s = [], r = canvas.grid.isGridless ? (d) => d + canvas.grid.sizeX : (d) => d + 1, c = canvas.grid.isGridless ? (d) => d + canvas.grid.sizeY : (d) => d + 1;
    for (let d = t; d < i; d = r(d))
      for (let u = n; u < a; u = c(u)) {
        const g = { i: d, j: u }, p = canvas.grid.getCenterPoint(g);
        !e.contains(p.x, p.y) || this.#s.has(`${p.x}:${p.y}`) || canvas.tokens.quadtree.getObjects(new PIXI.Rectangle(p.x, p.y, 0, 0), { collisionTest: /* @__PURE__ */ l((y) => y.t.hitArea.contains(p.x - y.t.x, p.y - y.t.y), "collisionTest") }).size || s.push(p);
      }
    return s;
  }
}
Hooks.once("init", () => {
  const o = game.modules.get(f.id);
  o.api ??= {}, o.api.changeRoofOpacity = _, o.api.changeTokens = F, o.api.createUpdateFromModifications = I, o.api.modifyScene = U, o.api.pickMacro = G, o.api.pickTileImage = z, o.api.placeTokens = (e, t) => K.place(e, t), o.api.playSound = J, o.api.toggleHiddenState = B, o.api.toggleRegionBehaviors = V, o.api.toggleSceneEmbedded = W, o.api.toggleTemporaryWalls = q;
});
const { SchemaField: X, BooleanField: T } = foundry.data.fields, { AdventureImporter: Q } = foundry.applications.sheets;
class E extends Q {
  static {
    l(this, "CoreAdventureImporter");
  }
  /** @inheritdoc */
  static DEFAULT_OPTIONS = {
    classes: [f.cssClass, "themed", "theme-light"],
    actions: {
      changelog: E._changelog
    }
  };
  /* -------------------------------------------------- */
  /**
   * Import options and handlers. Subclasses can extend this to add or change handlers.
   * @type {Record<string, MetaMorphicAdventureImportConfig>}
   */
  get importConfigurations() {
    const e = this.document.flags.metamorphic ?? {}, t = this.document.uuid, n = this.adventureModule.description, i = {
      type: "post",
      field: new T({
        label: "Don't show again",
        initial: /* @__PURE__ */ l(() => {
          const p = game.settings.get("core", "adventureImports")?.[t], h = game.settings.get(f.id, "firstStartup");
          return !!p || !!h;
        }, "initial")
      }),
      ignored: /* @__PURE__ */ l(() => !game.settings.settings.has(`${f.id}.firstStartup`), "ignored"),
      handler: /* @__PURE__ */ l(() => {
        game.settings.set(f.id, "firstStartup", !1);
      }, "handler")
    }, a = {
      type: "post",
      field: new T({
        label: "Activate Scene",
        initial: !0
      }),
      ignored: /* @__PURE__ */ l(() => !e.initialSceneId, "ignored"),
      handler: /* @__PURE__ */ l(() => {
        game.scenes.get(e.initialSceneId)?.activate();
      }, "handler")
    }, s = {
      type: "post",
      field: new T({
        label: "Display Introduction",
        initial: !0
      }),
      ignored: /* @__PURE__ */ l(() => !e.initialJournalEntryId || !e.initialJournalPageId, "ignored"),
      handler: /* @__PURE__ */ l(() => {
        game.journal.get(e.initialJournalEntryId)?.sheet.render(!0, { pageId: e.initialJournalPageId });
      }, "handler")
    }, r = {
      type: "post",
      field: new T({
        label: "Style Login Screen",
        initial: !1
      }),
      ignored: /* @__PURE__ */ l(() => !e.initialLoginScreenBackground, "ignored"),
      handler: /* @__PURE__ */ l(async () => {
        const p = {
          id: game.world.id,
          action: "editWorld",
          description: n,
          background: `modules/${f.id}/${e.initialLoginScreenBackground}`
        }, h = await foundry.utils.fetchJsonWithTimeout(foundry.utils.getRoute("setup"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(p)
        });
        game.world.updateSource(h);
      }, "handler")
    }, c = {
      type: "post",
      field: new T({
        label: e.chatMessage?.label,
        initial: !0
      }),
      ignored: /* @__PURE__ */ l(() => !e.chatMessage?.label || !e.chatMessage.content, "ignored"),
      handler: /* @__PURE__ */ l(() => {
        ChatMessage.implementation.create({
          content: e.chatMessage.content,
          whisper: ChatMessage.implementation.getWhisperRecipients("GM")
        });
      }, "handler")
    }, d = {
      type: "post",
      field: new T({
        label: "Use Party Token",
        initial: !1
      }),
      ignored: /* @__PURE__ */ l(() => game.system.id !== "pf2e" || !e.partyToken, "ignored"),
      handler: /* @__PURE__ */ l(() => {
        game.actors.party?.update({ img: e.partyToken });
      }, "handler")
    }, u = {
      type: "post",
      field: new T({
        label: "Apply Dynamic Token Ring",
        initial: !0
      }),
      ignored: /* @__PURE__ */ l(() => e.tokenRing in CONFIG.Token.ring.configLabels ? game.settings.get("core", "dynamicTokenRing") === e.tokenRing : !0, "ignored"),
      handler: /* @__PURE__ */ l(async () => {
        await game.settings.set("core", "dynamicTokenRing", e.tokenRing), foundry.applications.settings.SettingsConfig.reloadConfirm();
      }, "handler")
    }, g = {
      type: "post",
      field: new T({
        label: "Apply Turn Marker",
        initial: !0
      }),
      ignored: /* @__PURE__ */ l(() => !e.turnMarker, "ignored"),
      handler: /* @__PURE__ */ l(() => {
        const { animation: p, src: h } = e.turnMarker, y = "core", w = "combatTrackerConfig", b = foundry.utils.deepClone(game.settings.get(y, w));
        CONFIG.Combat.settings.turnMarkerAnimations.find((A) => A.value === p) && (b.turnMarker.animation = p), b.turnMarker.src = h, b.turnMarker.enabled = !0, game.settings.set(y, w, b);
      }, "handler")
    };
    return {
      chatMessage: c,
      customizeJoin: r,
      displayJournal: s,
      dontShowAgain: i,
      initialScene: a,
      partyToken: d,
      tokenRing: u,
      turnMarker: g
    };
  }
  /* -------------------------------------------------- */
  /**
   * The adventure module.
   * @type {Module}
   */
  get adventureModule() {
    return game.modules.get(f.id);
  }
  /* -------------------------------------------------- */
  /** @inheritdoc */
  _initializeApplicationOptions(e) {
    const t = super._initializeApplicationOptions(e);
    if (t.classes ??= [], e.document.flags.metamorphic?.cssClasses)
      for (const n of e.document.flags.metamorphic.cssClasses) {
        const i = n.split(" ").map((a) => a.trim()).filter((a) => a);
        t.classes.push(...i);
      }
    return t;
  }
  /* -------------------------------------------------- */
  /** @inheritdoc */
  async _prepareContext(e) {
    const t = await super._prepareContext(e);
    return this.adventureModule.changelog && t.buttons.push({
      type: "button",
      label: "SETUP.ViewPackageChangelog",
      action: "changelog",
      icon: "fa-solid fa-file-lines"
    }), t;
  }
  /* -------------------------------------------------- */
  /**
   * Prepare import options schema.
   * Options are rendered using the DataField#toInput method.
   * @param {AdventureImportOptions} options
   * @returns {SchemaField|undefined}
   * @protected
   */
  _prepareImportOptionsSchema(e) {
    const t = {};
    for (const [n, i] of Object.entries(this.importConfigurations))
      i.ignored?.() || typeof i.handler != "function" || (t[n] = i.field);
    return foundry.utils.isEmpty(t) ? void 0 : new X(t);
  }
  /* -------------------------------------------------- */
  /**
   * Open the changelog page.
   * @this {CoreAdventureImporter}
   * @param {PointerEvent} event    Initiating click event.
   * @param {HTMLElement} target    The capturing element that defined the [data-action].
   */
  static async _changelog(e, t) {
    window.open(this.adventureModule.changelog, "_blank").focus();
  }
  /* -------------------------------------------------- */
  /** @inheritdoc */
  _onChangeForm(e, t) {
    t.target.name === "dontShowAgain" && game.settings.set(f.id, "firstStartup", t.target.checked), super._onChangeForm(e, t);
  }
  /* -------------------------------------------------- */
  /**
   * Configure how adventures that use this sheet class are imported.
   * This can be implemented by subclasses to implement custom import workflows.
   * @param {AdventureImportOptions} importOptions
   * @returns {Promise<void>}
   * @internal
   */
  async _configureImport(e) {
    const t = this.importConfigurations, n = this._prepareImportOptionsSchema() ?? [];
    for (const i of n) {
      if (!e[i.name])
        continue;
      const { type: a = "pre", handler: s } = t[i.name];
      switch (a) {
        case "post":
          e.postImport.push(s);
          break;
        case "pre":
          e.preImport.push(s);
          break;
      }
    }
  }
  /* -------------------------------------------------- */
  /**
   * Configure how adventures that use this sheet class are imported.
   * This can be implemented by subclasses to implement custom import workflows.
   * @param {AdventureImportData} importData
   * @param {AdventureImportOptions} importOptions
   * @returns {Promise<void>}
   * @internal
   */
  async _preImport(e, t) {
    const { toCreate: n, toUpdate: i } = e;
    for (const a of [n, i]) {
      const s = ["Actor", "Item", "Scene", "JournalEntry", "Macro"];
      for (const r of s) {
        const c = a[r] ?? [];
        if (!c.length)
          continue;
        const d = this[`applyImport${r}Changes`];
        d && await d.call(this, c);
      }
    }
  }
  /* -------------------------------------------------- */
  /**
   * Perform modificatons to actor data prior to importing.
   * @param {object[]} data   The actor data.
   */
  async applyImportActorChanges(e) {
  }
  /* -------------------------------------------------- */
  /**
   * Perform modifications to item data prior to importing.
   * @param {object[]} data   The item data.
   */
  async applyImportItemChanges(e) {
  }
  /* -------------------------------------------------- */
  /**
   * Perform modifications to journal entry data prior to importing.
   * @param {object[]} data   The journal entry data.
   */
  async applyImportJournalEntryChanges(e) {
  }
  /* -------------------------------------------------- */
  /**
   * Perform modifications to scene data prior to importing.
   * @param {object[]} data   The scene data.
   */
  async applyImportSceneChanges(e) {
  }
  /* -------------------------------------------------- */
  /**
   * Perform modifications to macro data prior to importing.
   * @param {object[]} data   The macro data.
   */
  async applyImportMacroChanges(e) {
  }
}
class x extends E {
  static {
    l(this, "PF2EAdventureImporter");
  }
  /** @inheritdoc */
  async applyImportActorChanges(e) {
    if (!e?.length)
      return;
    const t = this.document.flags.metamorphic ?? {};
    for (const [n, i] of e.entries()) {
      const a = i._stats?.compendiumSource, s = await fromUuid(a);
      if (!s) {
        a?.startsWith("Compendium.") && console.warn(`Compendium source data for "${i.name}" [${a}] not found.`);
        continue;
      }
      const r = s.toObject(), c = (r.items ?? []).filter((g) => !x.#n(t, i._id, g));
      await x.#e(t, i._id, c);
      const d = [
        "_id",
        "folder",
        "img",
        "name",
        "prototypeToken.ring",
        "prototypeToken.texture",
        "prototypeToken.name"
      ];
      switch (s.type) {
        case "npc":
          d.push("prototypeToken.flags.pf2e", "system.attributes.adjustment", "system.attributes.hp.value", "system.details.blurb", "system.details.languages.value", "system.traits.size", "system.traits.value");
          break;
        case "hazard":
          d.push("prototypeToken.height", "prototypeToken.width", "system.traits.value");
          break;
        case "vehicle":
          d.push("prototypeToken.height", "prototypeToken.width");
          break;
        default:
          continue;
      }
      const u = {};
      u.items = c;
      for (const g of d)
        foundry.utils.hasProperty(i, g) && (u[g] = foundry.utils.getProperty(i, g));
      e[n] = foundry.utils.mergeObject(r, u);
    }
  }
  /* -------------------------------------------------- */
  /**
   * Determine whether an item should be removed from an imported actor.
   * @param {object} flags      Flag data retrieved from the importer.
   * @param {string} actorId    The id of the actor.
   * @param {object} itemData   The data for the item.
   * @returns {boolean}         Whether the item should be removed.
   */
  static #n(e, t, n) {
    const i = e._removeItems?.[t] ?? [];
    for (const a of i)
      if (a.id === n._id || a.name === n.name)
        return !0;
    return !1;
  }
  /* -------------------------------------------------- */
  /**
   * Mutate the array of item data that is to be created on the actor.
   * @param {object} flags            Flag data retrieved from the importer.
   * @param {string} actorId          The id of the actor.
   * @param {object[]} sourceItems    The item data. **will be mutated**
   * @returns {Promise<void>}         A promise that resolves once the item data array has been mutated.
   */
  static async #e(e, t, n) {
    const i = e.additionalItems?.[t] ?? [], a = n.map((s) => s._id);
    for (let s of i) {
      let r = await fromUuid(s);
      for (s = r.uuid, r = r.toObject(); a.includes(r._id); )
        r._id = foundry.utils.randomID();
      foundry.utils.setProperty(r, "_stats.compendiumSource", s), a.push(r._id), n.push(r);
    }
  }
}
const P = [];
P.push({
  sheet: x,
  system: "sf2e",
  importerName: "MetaMorphicAdventureImporter"
});
Hooks.once("init", () => {
  for (const { sheet: o, system: e, importerName: t } of P)
    game.system.id === e && (Object.defineProperty(o, "name", {
      value: t
    }), foundry.applications.apps.DocumentSheetConfig.registerSheet(Adventure, $.id, o, {
      label: `${$.shortTitle} Adventure Importer Sheet`,
      makeDefault: !1
    }));
  game.settings.register(f.id, "autoOpenAdventures", {
    name: "One-Time Startup Prompt",
    scope: "world",
    config: !1,
    type: new foundry.data.fields.BooleanField(),
    default: !0
  }), Hooks.on("updateSetting", (o) => {
    if (o.key === "core.moduleConfiguration") {
      const e = o.value;
      game.settings.set(f.id, "autoOpenAdventures", !e[f.id]);
    }
  });
});
Hooks.once("ready", async () => {
  const o = game.packs.filter((t) => {
    if (t.metadata.type !== "Adventure")
      return !1;
    const n = t.metadata.packageName;
    return !!game.settings.get(n, "autoOpenAdventures");
  }), e = await Promise.all((await Promise.all(o.map((t) => t.getIndex({ fields: ["flags.metamorphic"] })))).map(({ contents: t }) => t).flat().filter(({ flags: t }) => t?.metamorphic?.parentPackageId === $.id && t?.metamorphic?.autoOpen).map(({ uuid: t }) => fromUuid(t)));
  for (const t of e)
    t.sheet.render(!0);
  for (const t of o)
    game.settings.set(t.metadata.packageName, "autoOpenAdventures", !1);
});
Hooks.on("activateNote", (o, e) => {
  if (!o.entry)
    return;
  const n = o.document.flags.metamorphic?.scroll;
  n && (e.scrollTag = n);
});
const v = "sf2e-murder-in-metal-city", S = {
  d4: "sf2e-metal-custom-d4",
  d6: "sf2e-metal-custom-d6",
  d8: "sf2e-metal-custom-d8",
  d10: "sf2e-metal-custom-d10",
  d12: "sf2e-metal-custom-d12",
  d20: "sf2e-metal-custom-d20",
  d100: "sf2e-metal-custom-d100"
}, C = "sf2e-metal-custom-system", m = `modules/${v}/assets/dice`, k = {
  d2: {
    id: `${v}-d2-texture`,
    source: `${m}/d4/d4-texture.webp`,
    bump: `${m}/d4/d4-bump.png`,
    emissive: `${m}/d4/d4-emissive.webp`
  },
  d3: {
    id: `${v}-d3-texture`,
    source: `${m}/d4/d4-texture.webp`,
    bump: `${m}/d4/d4-bump.png`,
    emissive: `${m}/d4/d4-emissive.webp`
  },
  d4: {
    id: `${v}-d4-texture`,
    source: `${m}/d4/d4-texture.webp`,
    bump: `${m}/d4/d4-bump.png`,
    emissive: `${m}/d4/d4-emissive.webp`
  },
  d6: {
    id: `${v}-d6-texture`,
    source: `${m}/d6/d6-texture.webp`,
    bump: `${m}/d6/d6-bump.png`,
    emissive: `${m}/d6/d6-emissive.webp`
  },
  d8: {
    id: `${v}-d8-texture`,
    source: `${m}/d8/d8-texture.webp`,
    bump: `${m}/d8/d8-bump.png`,
    emissive: `${m}/d8/d8-emissive.webp`
  },
  d10: {
    id: `${v}-d10-texture`,
    source: `${m}/d10/d10-texture.webp`,
    bump: `${m}/d10/d10-bump.png`,
    emissive: `${m}/d10/d10-emissive.webp`
  },
  d12: {
    id: `${v}-d12-texture`,
    source: `${m}/d12/d12-texture.webp`,
    bump: `${m}/d12/d12-bump.png`,
    emissive: `${m}/d12/d12-emissive.webp`
  },
  d20: {
    id: `${v}-d20-texture`,
    source: `${m}/d20/d20-texture.webp`,
    bump: `${m}/d20/d20-bump.png`,
    emissive: `${m}/d20/d20-emissive.webp`
  },
  d100: {
    id: `${v}-d100-texture`,
    source: `${m}/d100/d100-texture.webp`,
    bump: `${m}/d100/d100-bump.png`,
    emissive: `${m}/d100/d100-emissive.webp`
  }
};
Hooks.once("diceSoNiceInit", (o) => {
  o && Object.entries(k).forEach(([e, t]) => {
    o.addTexture(t.id, {
      name: `Murder in Metal City ${e.toUpperCase()} Texture`,
      composite: "source-over",
      source: t.source,
      bump: t.bump,
      emissive: t.emissive,
      emissiveColor: "#5ae5ff",
      emissiveIntensity: 6,
      bumpScale: 1.5,
      scale: 0.75
    });
  });
});
Hooks.once("diceSoNiceReady", (o) => {
  if (!o)
    return;
  const e = {
    description: "Murder in Metal City Plain Dice",
    category: "Pathfinder",
    edge: "#5ae5ff",
    foreground: "#cdeff4",
    background: "#282828",
    visibility: "hidden",
    material: "chrome",
    font: "TiltNeon",
    outline: !0,
    outlineColor: "#282828"
  };
  Object.entries(S).forEach(([t, n]) => {
    o.addColorset({
      ...e,
      name: n,
      description: `${e.description} (${t.toUpperCase()})`,
      texture: k[t].id
    });
  }), o.addSystem({ id: C, name: "Murder in Metal City" }, !1), o.addDicePreset({
    type: "d4",
    colorset: S.d4,
    labels: ["1", "2", "3", "4"],
    system: C,
    texture: k.d4.id
  }), o.addDicePreset({
    type: "d6",
    colorset: S.d6,
    labels: ["1", "2", "3", "4", "5", "6"],
    system: C,
    texture: k.d6.id
  }), o.addDicePreset({
    type: "d8",
    colorset: S.d8,
    labels: ["1", "2", "3", "4", "5", "6", "7", "8"],
    system: C,
    texture: k.d8.id
  }), o.addDicePreset({
    type: "d10",
    colorset: S.d10,
    labels: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    system: C,
    texture: k.d10.id
  }), o.addDicePreset({
    type: "d12",
    colorset: S.d12,
    labels: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    system: C,
    texture: k.d12.id
  }), o.addDicePreset({
    type: "d20",
    colorset: S.d20,
    labels: [
      "1",
      "2",
      "3",
      "4",
      "5",
      "6",
      "7",
      "8",
      "9",
      "10",
      "11",
      "12",
      "13",
      "14",
      "15",
      "16",
      "17",
      "18",
      "19",
      "20"
    ],
    system: C,
    texture: k.d20.id
  }), o.addDicePreset({
    type: "d100",
    colorset: S.d100,
    labels: ["10", "20", "30", "40", "50", "60", "70", "80", "90", "00"],
    system: C,
    texture: k.d100.id
  });
});
console.log(`[${f.id}@${f.version}...] successfully loaded!`);
