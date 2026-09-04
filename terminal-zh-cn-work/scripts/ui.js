/* @license 2026 CodaBool all rights reserved */

import { setFlagAndTab, buildFontOptions } from './util.js'
import { ASCII, defaultStyles } from "./presets.js"
import { configureLocalizedApplications, t, templatePath } from "./i18n.js"
const { HandlebarsApplicationMixin, ApplicationV2 } = foundry.applications.api
export const ID = 'terminal'

export class Rename extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    classes: ["terminal-scroll"],
    tag: "form",
    window: {
      title: "Rename Buttons",
      icon: "fa-solid fa-input-text",
    },
    form: {
      handler: this.submit,
      submitOnChange: true,
    },
    actions: {
      show: this.show,
    },
    position: { width: 450 },
  }
  static PARTS = {
    form: { template: "modules/terminal/templates/rename.hbs" },
  }

  constructor(tileDoc) {
    super()
    this.tileDoc = tileDoc
  }

  async _prepareContext() {
    let macroName = this.tileDoc.getFlag("terminal", "macroName")
    if (!macroName) {
      macroName = game.macros.get(this.tileDoc.getFlag("terminal", "macro"))?.name
    }

    const unlockIds = this.tileDoc.getFlag("terminal", "unlockIds")
    const doors = []
    if (unlockIds?.trim()) {
      const ids = unlockIds.trim().split(",")
      for (const uuid of ids) {
        const door = await fromUuid(uuid)
        if (!door) continue
        const name = door.getFlag("terminal", "name")
        doors.push({
          name,
          uuid,
        })
      }
    }

    const rawSSH = this.tileDoc.getFlag("terminal", "ssh")
    const ssh = []
    if (rawSSH?.trim()) {
      const ids = rawSSH.trim().split(",")
      for (const uuid of ids) {
        const tile = await fromUuid(uuid)
        if (!tile) continue
        const name = tile.getFlag("terminal", "name")
        ssh.push({
          name,
          uuid,
        })
      }
    }

    return {
      ...this.tileDoc.flags.terminal,
      doors,
      doorsLength: doors.length,
      sshLength: ssh.length,
      ssh,
      macroName,
      activeMonk: game.modules.get("monks-active-tiles")?.active,
      activeTriggers: game.modules.get("monks-active-tiles")?.active,
    }
  }
  static async show(e) {
    const doc = fromUuidSync(e.target.attributes.uuid.value)
    if (!doc.parent.isView) {
      ui.notifications.info(t("TERMINAL.Action.DocumentOnScene", {
        type: doc.documentName === "Wall" ? t("TERMINAL.Document.Wall") : t("TERMINAL.Document.Tile"),
        scene: doc.parent.name,
      }))
      return
    }
    if (doc.documentName === "Tiles") {
      canvas.tiles.activate()
    } else if (doc.documentName === "Wall") {
      canvas.walls.activate()
    }
    canvas.ping(doc.object.center)
    canvas.animatePan({ ...doc.object.center, duration: 500 })
  }
  static async submit(e, form, data) {
    for (const [name, value] of Object.entries(data.object)) {
      if (name === "macroName") {
        if (
          value !== game.macros.get(this.tileDoc.getFlag("terminal", "macro"))?.name &&
          value !== this.tileDoc.getFlag("terminal", "macroName")
        ) {
          await setFlagAndTab(this.tileDoc, "macroName", value)
        }
      } else if (name.includes("Scene.")) {
        const doc = fromUuidSync(name)
        await doc.setFlag("terminal", "name", value)
      } else {

        // default
        await setFlagAndTab(this.tileDoc, name, value)
      }
    }

    this.render()
  }
}

export class Regions extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    tag: "form",
    classes: ["terminal-scroll"],
    window: {
      title: "Trigger Regions",
      icon: "fa-solid fa-map",
    },
    form: {
      handler: this.submit,
      submitOnChange: true,
    },
    position: { width: 450 },
    actions: {
      addRegion: this.addRegion,
      removeRegion: this.removeRegion,
      showTips: this.showTips,
    },
  }
  static PARTS = {
    form: { template: "modules/terminal/templates/regions.hbs" },
  }

  constructor(tileDoc) {
    super()
    this.tileDoc = tileDoc
    this.controller = null
  }

  async _prepareContext() {
    const regionsRaw = this.tileDoc.getFlag("terminal", "regions") || []
    const regions = regionsRaw.reduce((acc, curr) => {
      const region = fromUuidSync(curr.uuid)
      if (region) {
        acc.push({
          ...curr,
          prettyContext: t("TERMINAL.Region.Context", { scene: region.parent.name, region: region.name }),
        })
      }
      return acc
    }, [])
    // could do an operation to remove deleted regions, but it shouldn't cause bugs
    // so I'll just let it crust, https://www.tiktok.com/@claxmcb/video/7309329487315881246
    return {
      noRegion: regions.length === 0,
      regions,
      // behaviorStatus triggers whenever someone views the scene, it's just not useful
      events: Object.values(CONST.REGION_EVENTS).filter(e => e !== "behaviorStatus"),
    }
  }

  static async removeRegion(e) {
    const regions = this.tileDoc.getFlag(ID, "regions") || []
    const index = regions.findIndex(region => region.uuid === e.target.id)
    if (index === -1) {
      console.error("could not find region to remove")
      return
    }
    // do a workaround by manually removing the element
    document.querySelector(`div[uuid="${regions[index].uuid}"]`)?.remove()
    regions.splice(index, 1)
    setFlagAndTab(this.tileDoc, "regions", regions)
  }
  static async showTips() {
    const hint = document.querySelector(`.terminal-region-tips`)
    if (hint.style.display === "block") {
      hint.style.display = "none"
      hint.previousElementSibling.textContent = t("TERMINAL.Common.ShowTips")
    } else {
      hint.style.display = "block"
      hint.previousElementSibling.textContent = t("TERMINAL.Common.HideTips")
    }
  }
  static async addRegion() {
    ui.notifications.info(t("TERMINAL.Action.ClickRegion"))
    Object.values(ui.windows).forEach(app => app.minimize())
    foundry.applications.instances.forEach(a => a.minimize())

    if (game.release.build > 353) {
      // 14.354+, has placeables tab and removed regions manager UI
      ui.sidebar.expand()
      ui.sidebar.changeTab("placeables", "primary")
      document.querySelector('button[data-tab="regions"]')?.click();
    }


    canvas.regions.activate()
    const controller = new AbortController()
    this.controller = controller
    const { signal } = controller
    window.addEventListener("click", e => {
      let id
      if (game.release.build > 353) {
        // 14.354+, has placeables tab and removed regions manager UI
        if (e.target.nodeName === "LI") {
          id = e.target.getAttribute("data-entry-id")
        } else if (e.target.nodeName === "SPAN") {
          id = e.target.parentNode.parentNode.getAttribute("data-entry-id")
        }
      } else if (e.target.nodeName === "A" && e.target.classList?.contains("region-name")) {
        // 14.353-
        id = e.target.parentNode?.getAttribute("data-region-id")
      }

      if (id) {
        const regions = this.tileDoc.getFlag(ID, "regions") || []
        const uuid = `Scene.${canvas.scene.id}.Region.${id}`
        const rand = Math.random().toString(36).substring(2, 8)
        regions.push({
          uuid,
          id: rand,
          event: "tokenEnter",
          btnName: `script_${rand}`,
        })
        setFlagAndTab(this.tileDoc, "regions", regions)
        Object.values(ui.windows).forEach(app => app.maximize())
        foundry.applications.instances.forEach(a => a.maximize())
        // when adding the first region trigger, a delay is needed
        setTimeout(() => this.render(), 500)
        controller.abort()
        this.controller = null
      }
    }, { signal, capture: true })
  }
  static async submit(e, form, data) {
    const regionsRaw = this.tileDoc.getFlag(ID, "regions") || []
    const regions = regionsRaw.filter(r => {
      const region = fromUuidSync(r.uuid)
      return region !== null
    })
    for (const [name, value] of Object.entries(data.object)) {
      if (name === "btnName") {
        const arr = typeof value === "string" ? [value] : value
        for (const [index, btnName] of Object.entries(arr)) {
          regions[index].btnName = btnName
        }
      } else if (name === "event") {
        const arr = typeof value === "string" ? [value] : value
        for (const [index, event] of Object.entries(arr)) {
          regions[index].event = event
        }
      }
    }
    setFlagAndTab(this.tileDoc, "regions", regions)
  }
  close() {
    if (this.controller) {
      this.controller.abort()
      ui.notifications.info(t("TERMINAL.Action.RegionCanceled"))
      Object.values(ui.windows).forEach(app => app.maximize())
      foundry.applications.instances.forEach(a => a.maximize())
    }
    return super.close()
  }
}

export class OpenForPlayers extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    classes: ["terminal-scroll"],
    tag: "form",
    window: {
      title: "Open terminal for select players",
      icon: "fa-solid fa-users",
    },
    form: {
      handler: this.submit,
      closeOnSubmit: true,
    },
  };

  static PARTS = {
    form: { template: "modules/terminal/templates/open_for_players.hbs" },
  }

  constructor(tileUuid) {
    super()
    this.tileUuid = tileUuid
  }

  _prepareContext() {
    const users = game.users.filter(u => u.active)
    return {
      users,
      gmId: game.user.id,
      tileUuid: this.tileUuid,
    };
  }

  static submit(event, form, {object}) {
    const i = typeof object.users === "string" ? [object.users] : object.users?.filter(u => u)
    if (i.length === 0) return
    const users = i.map(id => game.users.get(id))
    const tileDoc = fromUuidSync(this.tileUuid)
    const skilled = tileDoc.getFlag(ID, "skilled")
    if (!window.validateTerminalTile(tileDoc, true)) return
    if (skilled) ui.notifications.info(t("TERMINAL.Action.SkillIgnored"))
    ui.notifications.info(t("TERMINAL.Action.OpenedFor", { count: users.length, names: users.map(u => u.name).join(", ") }))
    for (const user of users) {
      if (user.id === game.user.id) new Terminal(this.tileUuid).render(true)
      game.socket.emit('module.terminal', { action: "render", tuid: this.tileUuid, uid: user.id })
    }
  }
}


export class StyleForm extends HandlebarsApplicationMixin(ApplicationV2) {
  constructor(id, readOnly) {
    super()
    this.uuid = id
    this.readOnly = readOnly || false
  }
  static DEFAULT_OPTIONS = {
    classes: ["terminal-scroll"],
    tag: "form",
    window: {
      title: "Style Details - the look and feel of your Terminal",
      icon: "fa-solid fa-palette",
    },
    form: {
      handler: this.submit,
      submitOnChange: true,
    },
    position: { width: 700 },
    actions: {
      playSound: this.playSound,
      pickFile: this.pickFile,
      refresh: this.refresh,
    },
  }
  static PARTS = {
    form: { template: "modules/terminal/templates/form.hbs" },
  }
  async _prepareContext() {
    const styles = game.settings.get(ID, "styles")
    let video = false
    if (styles[this.uuid].background) {
      video = Object.keys(CONST.VIDEO_FILE_EXTENSIONS).some(ext => styles[this.uuid].background.endsWith('.' + ext))
    }
    const fonts = buildFontOptions(styles[this.uuid].font || "inherit")
    return { ...styles[this.uuid], readOnly: this.readOnly, video, fonts }
  }
  static playSound(e) {
    const audio = e.target.attributes.audio.value
    if (!audio) return
    new foundry.audio.Sound(audio).load().then(sound => {
      sound.play({ volume: game.settings.get("core", "globalInterfaceVolume") })
    })
  }
  static refresh() {
    this.render()
  }
  static async pickFile(e) {
    if (this.readOnly) return
    const name = e.target.attributes.name?.value || e.target.attributes.field.value
    const type = e.target.attributes.file.value
    if (!$(`input[name="${name}"]`).val() || name === "borderImage" || name === "background") {
      new foundry.applications.apps.FilePicker({
        type,
        callback: async path => {
          // TODO: should scope to the correct window
          $(`input[name="${name}"]`).val(path)
          if ($(`#terminal-refresh-form`).length) $(`#terminal-refresh-form`).click()
        }
      }).browse()
      return
    }
    new foundry.applications.apps.FilePicker({
      type,
      callback: async path => {
        $(`input[name="${name}"]`).val(path)
        if ($(`#terminal-refresh-form`).length) $(`#terminal-refresh-form`).click()
      }
    }).browse()
  }
  _onClose(options) {
    super._onClose(options)
    if (this.readOnly) return
    // refresh styles dropdown on all tile configs
    // need to set to a random value to force an update
    for (const config of Object.values(ui.windows)) {
      if (config.options.id !== "tile-config") continue
      setFlagAndTab(config.object, "refreshConfig", Math.random())
    }
  }
  static async submit(e, form, data) {
    const styles = game.settings.get(ID, "styles")
    styles[this.uuid] = { ...data.object, uuid: this.uuid }
    await game.settings.set(ID, "styles", styles)
    // refresh parent Form
    document.querySelector("#terminal-refresh")?.click()
    this.render()
  }
}

export class QuickStart extends HandlebarsApplicationMixin(ApplicationV2) {
  constructor(doc, id) {
    super()
    this.doc = doc
    this.uuid = id
  }
  static DEFAULT_OPTIONS = {
    classes: ["terminal-scroll"],
    tag: "form",
    window: {
      title: "Tile Style Quick-start",
      icon: "fa-solid fa-wand-magic-sparkles",
    },
    form: {
      handler: this.submit,
      closeOnSubmit: true,

    },
    position: { width: 400 },
  }
  static PARTS = {
    form: { template: "modules/terminal/templates/quick_start.hbs" },
  }
  static async submit(e, form, { object }) {
    const { image, color, addLight } = object
    this.doc.update({
      "texture.src": `/modules/terminal/background/${image}.webp`,
      "texture.tint": color,
    })

    // create light
    if (addLight) {
      const scene = this.doc.parent
      const size = scene.grid.size
      const bright = (this.doc.width / size + this.doc.height / size) / 2 * scene.grid.distance
      scene.createEmbeddedDocuments("AmbientLight", [{
        x: this.doc._object.center.x,
        y: this.doc._object.center.y,
        config: {
          animation: {
            type: "grid",
            intensity: 4,
          },
          color,
          alpha: .3,
          bright,
        },
      }])
    }
  }
}

export class Style extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    classes: ["terminal-scroll"],
    tag: "form",
    window: {
      title: "Collection",
      icon: "fa-solid fa-paintbrush",
    },
    form: {
      handler: this.submit,
    },
    actions: {
      view: this.view,
      delete: this.delete,
      create: this.create,
      update: this.update,
      refresh: this.refresh,
      togglePresets: this.togglePresets,
    },
    position: { width: 500 },
  }

  static PARTS = {
    form: { template: "modules/terminal/templates/styles.hbs" },
  }
  // get data
  async _prepareContext(options) {
    const styles = game.settings.get(ID, "styles")
    const custom = Object.values(styles).filter(s => !defaultStyles[s.uuid])
    const preset = Object.values(styles).filter(s => defaultStyles[s.uuid])
    return { custom, preset }
  }
  static view(e) {
    if (typeof e.target.attributes.readOnly === 'undefined') {
      new StyleForm(e.target.attributes.uuid.value).render(true)
    } else {
      new StyleForm(e.target.attributes.uuid.value, true).render(true)
    }
  }
  static async togglePresets() {
    const hint = document.querySelector(`.terminal-style-presets`)
    if (hint.style.display === "block") {
      hint.style.display = "none"
      document.querySelector('a[data-action="togglePresets"]').textContent = t("TERMINAL.Common.ShowPresets")
    } else {
      hint.style.display = "block"
      document.querySelector('a[data-action="togglePresets"]').textContent = t("TERMINAL.Common.HidePresets")
    }
  }
  static async delete(e) {
    const uuid = e.target.attributes.uuid.value
    const styles = game.settings.get(ID, "styles")

    let content = `<p style="margin-top: 1em; font-size:1.3em">${t("TERMINAL.Dialog.DeleteConfirm", { name: styles[uuid].name })}<p>`
    for (const tile of canvas.tiles.objects.children) {
      if (uuid === tile.document.flags.terminal?.style) {
        content = `
          <p style="margin: 1em; font-size:1.3em">${t("TERMINAL.Dialog.DeleteConfirm", { name: styles[uuid].name })}<p>
          <p style="margin-bottom: 1em; font-size:1.2em; text-shadow: 0 0 red;">${t("TERMINAL.Dialog.DeleteWarning")}</p>
        `
      }
    }

    const replace = await foundry.applications.api.DialogV2.confirm({
      modal: true,
      rejectClose: false,
      window: { title: t("TERMINAL.Dialog.DeleteStyle", { name: styles[uuid].name }) },
      position: { width: 400 },
      content,
    })
    if (!replace) return
    delete styles[uuid]
    await game.settings.set(ID, "styles", styles)
    this.render()
  }
  static async create() {
    let uuid
    // could use foundry.utils.randomID() instead
    if (crypto.randomUUID) {
      uuid = crypto.randomUUID()
    } else { // fallback to getRandomValues, since randomUUID is not available over HTTP
      uuid = [...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, "0")).join("")
    }

    const styles = game.settings.get(ID, "styles")
    const len = Object.values(styles).filter(k => k.name.includes("custom ")).length
    styles[uuid] = {
      name: "custom style " + (len + 1),
      click: `modules/${ID}/audio/blade_runner_click.mp3`,
      close: `modules/${ID}/audio/cyberpunk_close.mp3`,
      startup: `modules/${ID}/audio/generic_startup.mp3`,
      borderImage: `modules/${ID}/background/metal_square_1.webp`,
      shadow: "#19572e",
      base: "#23a84f",
      highlight: "#b6fab6",
      ascii: ASCII.STARWARS,
      opacity: .5,
      uuid,
      effectScan: true,
      effectScramble: true,
      effectGlitch: true,
      showASCIILoading: true,
    }
    await game.settings.set(ID, "styles", styles)
    new StyleForm(uuid).render(true)
    this.render()
  }
  static refresh() {
    this.render()
  }
}

export class Feedback extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    classes: ["terminal-scroll"],
    tag: "form",
    window: {
      title: "Feedback",
      icon: "fa-solid fa-bug",
    },
    form: {
      handler: this.submit,
      closeOnSubmit: true,
    },
  }
  static PARTS = {
    form: { template: "modules/terminal/templates/feedback.hbs" },
  }
  static async submit(e, form, data) {
    if (!data.object.feedback) return
    const version = game.data.release.generation + "." + game.data.release.build
    const module = "terminal-" + game.modules.get("terminal").version
    const active = game.modules.filter(m => m.active).map(m => m.id).join(";")
    const system = game.system.id
    fetch("https://d3erver.codabool.workers.dev/email" + encodeURI(`?version=${version}&module=${module}&active=${active}&system=${system}`), {
      method: "post",
      body: data.object.feedback
    })
    ui.notifications.info(t("TERMINAL.Action.FeedbackReceived"))
  }
}

export class Check extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    classes: ["terminal-scroll"],
    tag: "form",
    window: {
      title: "Skill Checks",
      icon: "fa-solid fa-dice-d20",
    },
    form: {
      handler: this.submit,
      submitOnChange: true,
    },
    position: { width: 600, height: "auto" },
  }
  static PARTS = {
    form: { template: "modules/terminal/templates/check.hbs" },
  }
  constructor(tileDoc) {
    super()
    this.tileDoc = tileDoc
  }
  async _prepareContext() {
    return {
      ...this.tileDoc.flags.terminal,
      hasMonk: game.modules.get("monks-active-tiles")?.active,
    }
  }
  static async submit(e, form, data) {
    for (const [flag, value] of Object.entries(data.object)) {
      setFlagAndTab(this.tileDoc, flag, value)
    }
  }
}

export class Shadow extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    tag: "form",
    classes: ["terminal-scroll", "terminal-shadow-window"],
    window: {
      title: "Shadow",
      icon: "fa-solid fa-eye",
    },
    form: {
      handler: this.submit,
      submitOnChange: true,
      // closeOnSubmit: true,
    },
    actions: {
      shadowSelect: this.shadowSelect,
      sourceSelect: this.sourceSelect,
      shadowSubmit: this.shadowSubmit,
      resetWindow: this.resetWindow,
      swap: this.swap,
      showLimits: this.showLimits,
    },
    position: { width: 500 },
  }
  static PARTS = {
    form: { template: "modules/terminal/templates/shadow.hbs" },
  }
  constructor() {
    super()
    this.interval = null
    this.shadow = null
    this.source = null
    this.selected = new Set()
  }
  static async shadowSelect(e) {
    const shadow = game.users.get(e.target.attributes.pid.value)
    if (this.selected.has(shadow.id)) {
      this.selected.delete(shadow.id);
    } else {
      this.selected.add(shadow.id);
    }
    this.render()
  }
  static async shadowSubmit(e) {
    if (this.selected.size === 0) {
      ui.notifications.error(t("TERMINAL.Error.NoShadows"))
      return;
    }
    document.querySelector(".terminal-shadow-select").style.display = "none";
    document.querySelector(".terminal-source-select").style.display = "block"
  }
  static async showLimits(e) {
    const hint = document.querySelector(`.terminal-shadow-tips`)
    if (hint.style.display === "block") {
      hint.style.display = "none"
      hint.previousElementSibling.innerHTML = `<i class="fa-solid fa-traffic-cone"></i> ${t("TERMINAL.Common.Limitations")}`
    } else {
      hint.style.display = "block"
      hint.previousElementSibling.innerHTML = `<i class="fa-solid fa-eye-slash"></i> ${t("TERMINAL.Common.HideLimits")}`
    }
  }
  static async swap() {
    const s = game.settings.get("terminal", "shadow")
    await game.settings.set("terminal", "shadow", { ...s, shadow: s.source, source: s.shadow })
    this.resetShared()
    this.render()
  }
  static async sourceSelect(e) {
    const source = game.users.get(e.target.attributes.pid.value)
    if (this.selected.has(source.id)) {
      ui.notifications.error(t("TERMINAL.Error.SelfShadow"))
      this.resetShared()
      return
    }
    const sourceElements = document.querySelectorAll(`.terminal-shadow-${source.id}`);
    let tuid = null
    for (const sourceElement of sourceElements) {
      if (sourceElement.style.display === "block") {
        if (sourceElement.classList.contains("fa-computer")) {
          tuid = sourceElement.getAttribute("tuid")
        }
      }
    }
    if (!tuid) {
      ui.notifications.error(t("TERMINAL.Error.NotUsingTerminal", { name: source.name }))
      this.resetShared()
      return
    }
    Array.from(this.selected).forEach(id => {
      if (id === game.user.id) {
        if (!document.querySelector(`.${tuid.replace(/\./g, '-')}`)) {
          new Terminal(tuid, { shadowView: true }).render(true)
        }
      } else {
        game.socket.emit('module.terminal', {
          action: "render",
          uid: id,
          arg: { shadowView: true },
          tuid,
        })
      }
    })
    const names = Array.from(this.selected).map(id => game.users.get(id).name).join(", ")
    ui.notifications.info(t("TERMINAL.Action.Shadowing", { count: this.selected.size, source: source.name, names }))
    this.resetShared()
  }
  static async resetWindow() { this.resetShared() }
  resetShared() {
    document.querySelector(".terminal-shadow-select").style.display = "block";
    document.querySelector(".terminal-source-select").style.display = "none"
    this.selected.clear()
    clearInterval(this.interval)
    this.createInterval()
    this.render()
  }
  async _prepareContext() {

    const activePlayers = game.users.contents.filter(p => p.active)
    const allPlayers = game.users.contents
    const shadow = game.settings.get("terminal", "shadow")
    // const browserTheme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? "dark" : "light"
    // const theme = game.settings.get("core", "colorScheme") || browserTheme

    for (const p of activePlayers) {
      if (this.selected.has(p.id)) {
        p.terminalSelected = true;
      } else {
        p.terminalSelected = false;
      }
    }
    return {
      activePlayers,
      allPlayers,
      selected: Array.from(this.selected),
      ...shadow,
      // darkTheme: theme === "dark",
      inTerminal: $(".terminal-window").length > 0,
      uid: game.user.id,
      hasMinUsers: game.users.contents.length > 1,
      // hasMonk: game.modules.get("monks-active-tiles")?.active,
    }
  }
  createInterval() {
    this.interval = setInterval(() => {
      const otherActiveUserIds = game.users.contents.filter(p => p.active && !p.isSelf).map(p => p.id)
      game.socket.emit('module.terminal', {
        action: "shadowPing",
        gid: game.user.id,
        otherActiveUserIds,
      });
    }, 1_000);

    setTimeout(() => {
      clearInterval(this.interval);
      const refreshElement = document.getElementById("terminal-shadow-refresh")
      if (refreshElement) {
        refreshElement.textContent = t("TERMINAL.Common.TimedOut")
      }
    }, 300_000);
  }
  async _onRender(context) {
    if (this.interval) return
    this.createInterval()
  }
  close() {
    clearInterval(this.interval);
    return super.close()
  }

  static async submit(e, form, { object }) {
    if (!object.shadow) {
      object.shadow = game.users.contents[0].id;
      object.source = game.users.contents[1].id;
    }
    if (object.source === object.shadow) {
      ui.notifications.error(t("TERMINAL.Error.SelfShadow"))
    } else {
      await game.settings.set("terminal", "shadow", object)
    }
    this.resetShared()
    this.render()
  }
}

export class Skilled extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    classes: ["terminal-skilled"],
    position: { width: $(window).height() * .6, height: $(window).height() * .6 },
  }
  static PARTS = {
    form: { template: "modules/terminal/templates/skilled.hbs" },
  }
  constructor(message) {
    super()
    this.message = message
  }
  async _prepareContext() {
    const size = $(window).height() * .6
    const performance = game.settings.get("core", "performanceMode")
    let simple = !game.settings.get("terminal", "screensaver")
    // COMPATABILITY: Firefox seems to be buggy, just keep it simple for now
    if (!simple) {
      simple = navigator.userAgent.toLowerCase().indexOf('firefox') > -1 ? true : simple
    }
    return { size, performance, simple, message: this.message }
  }
  _onRender(context, options) {
    const a = document.querySelector('.terminal-skilled canvas')

    // remove padding from section
    document.querySelector('.terminal-skilled section').style.padding = 0

    const ctx = a.getContext('2d')
    const w = a.width
    const h = a.height
    let mx = w / 2
    let my = h / 2
    let p = new Array();
    let radius = 1;	// radius
    let coef = 0;		// coef
    let angle, dx, dy, speed, pr, l


    a.addEventListener('mousemove', function (evt) {
      var r = a.getBoundingClientRect();
      mx = evt.clientX - r.left;
      my = evt.clientY - r.top;

      // 0,0 is the center of canvas.
      dx = w / 2 - mx;
      dy = h / 2 - my;

      coef = Math.sqrt(dx * dx + dy * dy) / ((w + h) / 14);
    }, false)

    // loop function that recycles "for"
    myLoop(0)

    // Animate
    animate()

    function Pa(ox, oy, c, s) {
      this.x = 0;		  // center x coordinate
      this.y = 0;		  // center y coordinate
      this.ox = ox;	  // offset x
      this.oy = oy;	  // offset y
      this.c = c;		  // colour
      this.speed = s;	// speed
      this.sx = 0;		// shake x
      this.sy = 0;		// shake y
    }

    // Clear canvas, update positions.
    function animate() {

      ctx.clearRect(0, 0, w, h)

      if (!context.simple) {
        myLoop(1)
      }

      // Canvas rectangle
      ctx.beginPath();
      ctx.lineWidth = 1;
      ctx.rect(0, 0, w, h);
      ctx.stroke();

      // add text
      ctx.font = "24px serif"
      ctx.fillStyle = 'grey'
      ctx.textAlign = "center"
      ctx.fillText(t("TERMINAL.Skilled.Found"), (w / 2), (h / 2) - 40)
      ctx.fillText(context.message, (w / 2), (h / 2))

      if (!context.simple) {
        window.setTimeout(animate, 24)
      }
    }

    //Perform initialization or update positions. Hack - recycle the function.
    //functionality=0 then function update positions.
    //functionality=1 then function does initialization.
    function myLoop(functionality) {

      // create number of points based on performance mode number
      // 0 = low
      // 1 = medium
      // 2 = high
      // 3 = max

      for (var i = 100 * (context.performance + 2); i >= 0; i--) {
        if (functionality) {
          // Reduce code- there are many places that use p[i].
          l = p[i];

          // Update position using custom Xenon theorem (half path between two points)
          // in this case a proportional distance between source and target given by .t
          l.x += (mx - l.x) / l.speed;
          l.y += (my - l.y) / l.speed;

          l.sx = Math.random();
          l.sy = Math.random();
          ctx.beginPath();

          // Use "arc" to draw circle, but 6.28318-coef for special effect.
          // 3 is the radius of the points
          ctx.arc(l.x + l.ox + l.sx, l.y + l.oy + l.sy, 3, 0, 6.28318 - coef);
          ctx.fillStyle = l.c;
          ctx.fill();
          ctx.strokeStyle = l.c;
          ctx.stroke();
        } else {
          // Convert to radian.(will effect length/width of system
          angle = (137.5077 * Math.PI / 180 * i);

          // Each 5 loops, perform radius variation. Higher the mod #, narrower the partical tunnel
          if ((i % 6) == 0)
            radius += 2;

          // Hack: use speed to generate color and make it depend on radius.
          speed = radius + 1; // 4 default

          // Create and initialize particle item.
          pr = new Pa(
            Math.cos(angle) * radius,
            Math.sin(angle) * radius,
            'rgba(' + speed * 4 + ',' + speed + ',' + speed * 12 + ',' + (0.8) + ')',
            speed);
          p.push(pr);
        }
      }
    }
  }
}

export function configureUILocalization() {
  configureLocalizedApplications([
    { application: Rename, template: "rename.hbs", title: "TERMINAL.Window.Rename" },
    { application: Regions, template: "regions.hbs", title: "TERMINAL.Window.Regions" },
    { application: OpenForPlayers, template: "open_for_players.hbs", title: "TERMINAL.Window.OpenPlayers" },
    { application: StyleForm, template: "form.hbs", title: "TERMINAL.Window.StyleDetails" },
    { application: QuickStart, template: "quick_start.hbs", title: "TERMINAL.Window.QuickStart" },
    { application: Style, template: "styles.hbs", title: "TERMINAL.Window.Collection" },
    { application: Feedback, template: "feedback.hbs", title: "TERMINAL.Window.Feedback" },
    { application: Check, template: "check.hbs", title: "TERMINAL.Window.SkillChecks" },
    { application: Shadow, template: "shadow.hbs", title: "TERMINAL.Window.Shadow" },
    { application: Skilled, template: "skilled.hbs" },
  ])
}
