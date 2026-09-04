/* @license 2026 CodaBool all rights reserved */

const {
  HandlebarsApplicationMixin,
  ApplicationV2
} = foundry.applications.api
import {
  toggleDoorLock,
  exploreMap,
  generatePseudoText,
  validateDoor,
  detectMotion,
  startObservation,
  toggleLights,
  runMacro,
  warmCache,
  triggerRegion,
  autoShadowTerminal,
} from "./util.js"
import { audio } from "./hooks.js"
import { ASCII } from "./presets.js"
import { actionLabel, t, templatePath } from "./i18n.js"
import { gsap } from "/scripts/greensock/esm/gsap-core.js"
import TextPlugin from "/scripts/greensock/esm/TextPlugin.js"
import ScrambleTextPlugin from "/scripts/greensock/esm/ScrambleTextPlugin.js"
gsap.registerPlugin(TextPlugin, ScrambleTextPlugin)

const ID = 'terminal'

export default class Terminal extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    classes: ["terminal-window"],
    position: {
      width: Math.min(window.innerHeight * .8, 1200),
      height: Math.min(window.innerHeight * .8, 1200),
    },
    actions: {
      closeWindow: this.closeWindow,
      minimize: this.minimize,
      password: this.password,
    },
  }
  static PARTS = {
    form: {
      template: "modules/terminal/templates/terminal.hbs"
    },
  }

  _awaitTransition(html) {
    if (this.context.error) return
    const minimizeBtn = html.querySelector("button[data-action='minimize']")
    if (!minimizeBtn) return
    const app = foundry.applications.instances.get(foundry.applications.instances.entries().find(f =>
      foundry.applications.instances.get(f[0]).uuid === this.uuid)[0]
    )
    // handle race condition of resizing.
    // can set padding-left on button to fix styling issue but I'm just ignoring
    const interval = setInterval(() => {
      if (app.minimized) {
        html.style.borderWidth = "5px"
        minimizeBtn.classList.remove("fa-dash")
        minimizeBtn.classList.add("fa-square")
      } else {
        html.style.borderWidth = "30px"
        minimizeBtn.classList.add("fa-dash")
        minimizeBtn.classList.remove("fa-square")
        // manually set height since Foundry maximize seems bugged
        foundry.applications.instances.forEach(a => {
          if (a.uuid === this.uuid) {
            a.setPosition({ height: Math.min(window.innerHeight * .8, 1200) })
          }
        })
      }
    }, 10)
    setTimeout(() => {
      clearInterval(interval);
    }, 2_500)
  }

  static minimize(e) {
    // COMPATABILITY: use Firefox supported .forEach instead of .values().find()
    // const app = foundry.applications.instances.values().find(a => a.uuid === this.uuid)
    foundry.applications.instances.forEach(a => {
      if (a.uuid === this.uuid) {
        if (e.target.classList.contains('fa-square')) {
          a.maximize()
        } else {
          a.minimize()
        }
      }
    })
    if (this.context.click) {
      audio.click = new foundry.audio.Sound(this.context.click)
      audio.click.load().then(s => s.play({ volume: game.settings.get("core", "globalInterfaceVolume") }))
    }
  }

  static async closeWindow(e) {
    const { close, click, journal, disableJournalSharing } = this.context
    if (click) {
      audio.click = new foundry.audio.Sound(click)
      audio.click.load().then(s => s.play({ volume: game.settings.get("core", "globalInterfaceVolume") }))
    }
    if (!disableJournalSharing) {
      const j = game.journal.get(journal)
      const hasAccess = j?.permission >= CONST.DOCUMENT_OWNERSHIP_LEVELS.OBSERVER
      if (j?.ownership?.default < 2 && !game.user.isGM && hasAccess && !this.arg.shadowView) {
        const share = await foundry.applications.api.DialogV2.confirm({
          modal: true,
          rejectClose: false,
      window: { title: t("TERMINAL.Window.ShareData") },
      content: `<p style="font-size:1.4em">${t("TERMINAL.Dialog.SharePrompt")}</p>`,
        })
        if (share) {
          const gms = game.users?.filter(u => u.active && u.isGM)
          if (!gms.length) {
      ui.notifications.error(t("TERMINAL.Error.NoGMShare"))
            return
          }
          game.socket.emit('module.terminal', {
            action: "publish",
            jid: this.context.journal,
            gid: gms[0]._id,
          })
        }
      } else if (game.user.isGM && !localStorage.getItem("terminal-skip-close-prompt") && !this.arg.shadowView) {
        await foundry.applications.api.DialogV2.confirm({
          rejectClose: false,
      window: { title: t("TERMINAL.Window.SharePreview") },
          classes: ["terminal-prompt"],
          content: `<p>${t("TERMINAL.Dialog.SharePreviewText")}</p><hr/>
          <p style="font-size:1.4em">${t("TERMINAL.Dialog.SharePrompt")}</p>
          <div class="form-group">
            <label style="flex: 5">${t("TERMINAL.Dialog.HideDemo")}</label>
            <div class="form-fields"><input type="checkbox" name="show" /></div>
          </div>`,
        })
        const check = document.querySelector('.terminal-prompt input[name="show"]')?.checked
        if (check) {
          localStorage.setItem("terminal-skip-close-prompt", "yes")
        }
      }
    }
    if (close) {
      audio.close = new foundry.audio.Sound(close)
      audio.close.load().then(s => s.play({ volume: game.settings.get("core", "globalInterfaceVolume") }))
    }
    this.lain.querySelector(`button[data-action="close"]`).click()
  }

  static password() {
    const password = this.lain.querySelector(".terminal-password")
    if (password.value === this.context.password) {
      this.lain.querySelector(`.terminal-loading`).style.display = "block"
      this.lain.querySelector(`.terminal-password-container`).style.display = "none"
      // TODO: could use this. instead of looping over apps here
      // COMPATABILITY: use Firefox supported .forEach instead of .values().find()
      foundry.applications.instances.forEach(a => {
        if (a.uuid === this.uuid) {
          a.createHTML()
          a.initCLI()
        }
      })
    } else {
      ui.notifications.error(t("TERMINAL.Error.IncorrectLogin"))
      password.value = ""
      password.focus()
    }
  }

  constructor(uuid, arg) {
    super()
    this.uuid = uuid
    this.lain = null
    this.context = null
    this.createHTML = null
    this.replaceHTML = null
    this.injectHTML = null
    this.successHTML = null
    this.initCLI = null
    this.arg = arg || {}
  }

  async _onRender(ctx) {
    const tuidCSS = this.uuid.replace(/\./g, '-')

    // add class with tile UUID to window
    for (const l of document.querySelectorAll('.terminal-window')) {
      if (l.classList.length === 2) {
        l.classList.add(tuidCSS)
        if (this.arg.isSSH) l.classList.add("terminal-ssh")
        if (this.arg.shadowView) l.classList.add("terminal-shadow")
        this.lain = l
        break
      }
    }

    const lain = this.lain
    const arg = this.arg
    this.context = ctx
    this.createHTML = createHTML
    this.replaceHTML = replaceHTML
    this.injectHTML = injectHTML
    this.successHTML = showContent
    this.initCLI = initCLI
    if (ctx.error) return

    generateBackground(ctx)
    if (ctx.password) {
      takePassword()
    } else {
      createHTML()
      initCLI()
    }

    async function styleClick(index) {
      if (ctx.effectGlitch) {
        if (Math.random() < 1 / 7) {
          lain.querySelector(".terminal-main").style.filter = "url(#noise)"
          const turbVal = { val: 0.01 }
          const turb = document.querySelectorAll("feTurbulence")[0]
          const timeline = gsap.timeline({
            onUpdate: () => turb.setAttribute("baseFrequency", "0 " + turbVal.val)
          })
          timeline.to(turbVal, .1, { val: .1 }).to(turbVal, .05, { val: 0 })
          setTimeout(() => {
            lain.querySelector(".terminal-main").style.filter = "none"
          }, 100)
        }
      }
      const buttons = lain.querySelectorAll(".terminal-button")

      // Remove 'selected' class from all buttons
      buttons.forEach(button => {
        if (button.classList.contains("terminal-selected")) {
          button.classList.remove('terminal-selected')
        }
      })

      if (buttons[index]) buttons[index].classList.add('terminal-selected')
      if (ctx.click) {
        audio.click = new foundry.audio.Sound(ctx.click)
        audio.click.load().then(s => s.play({ volume: game.settings.get("core", "globalInterfaceVolume") }))
      }
    }

    function showContent(index, content, title, func) {
      styleClick(index)
      const injectObj = {
        type: "text",
        text: {
          content
        },
        func: func,
        name: title
      }
      injectHTML(injectObj)
      game.socket.emit('module.terminal', {
        tuid: ctx.tuid,
        action: "shadowBtn",
        injectObj,
      })
    }

    // A skill-check description travels over game.socket to the GM, whose Foundry language may
    // differ from this player's. Send a catalog key + data so the GM's client renders it, and fall
    // back to a literal only for GM-authored world data (custom button names, journal page titles).
    function describe(data, literal, key, keyData) {
      delete data.description
      delete data.descriptionKey
      delete data.descriptionData
      if (literal) data.description = literal
      else {
        data.descriptionKey = key
        if (keyData) data.descriptionData = keyData
      }
    }

    function skillCheck(data) {
      data.action = "gmApprove"
      styleClick(data.index)
      lain.querySelectorAll(".terminal-skill-check").forEach(b => b.remove())
      let content = `
        <h1 style="font-size: 2em">${t("TERMINAL.Dialog.PermissionDenied")}</h1>
        <p style="font-size: 1.3em">${t("TERMINAL.Dialog.AccessProhibited")}</p>
        <button style="margin-top:3em" class="terminal-skill-check">${t("TERMINAL.Dialog.PerformSkilled")}</button>`

      if (data.title === "decrypt a file") {
        content = `
        <p style="font-size: 1.1em">${generatePseudoText(100)}</p>
        <button style="margin-top:3em" class="terminal-skill-check terminal-distortion">${t("TERMINAL.Dialog.DecryptionKey")}</button>`
      }
      injectHTML({
        type: "text",
        callback: () => {
          const app = foundry.applications.instances.values().find(a => a.uuid === ctx.tuid)
          app.injectHTML({
            type: "loading",
            name: actionLabel(data.title),
            data: `${JSON.stringify(data)}`,
          })
        },
        text: { content },
        name: actionLabel(data.title)
      })
    }

    function stopLoading() {
      lain.querySelector(`.terminal-splash`).style.display = "none"
      lain.querySelector(`.terminal-main`).style.display = "block"
      if (ctx.background) {
        lain.querySelector(`.terminal-noise`)?.remove()
      }
      setTimeout(() => lain.querySelector('.terminal-input')?.focus(), 0)
      if (!ctx.ambient) return
      // start ambient audio
      audio.ambient = new foundry.audio.Sound(ctx.ambient, { loop: true })
      audio.ambient.load().then(s => s.play({
        volume: game.settings.get("core", "globalInterfaceVolume") * .5,
        loop: true,
      }))
    }

    async function createButtons() {
      const gms = game.users?.filter(u => u.active && u.isGM)
      if (!gms.length) {
        ui.notifications.error(t("TERMINAL.Error.GMRequired"))
        return
      }

      const tile = fromUuidSync(ctx.tuid)
      const data = {
        sid: canvas.scene.id,
        name: game.user.name,
        gid: gms[0]._id,
        uid: game.user.id,
        tuid: ctx.tuid,
      }

      let size = 0
      lain.querySelector(`.terminal-folders`).innerHTML = '';
      const sibling = lain.querySelector(`.terminal-folders`)
      const pagesMap = game.journal.get(ctx.journal).pages
      const sortedPages = [...pagesMap.values()].sort((a, b) => a.sort - b.sort)
      for (const p of sortedPages) {
        // limited ownership pages are always hidden
        if (p.ownership.default === 1) continue
        if (!ctx.encryption && p.ownership.default === 0) continue
        const div = document.createElement("div")
        div.className = "terminal-button terminal-journal-page"
        div.dataset.pid = p._id
        div.dataset.type = p.type
        div.textContent = p.name
        const newPointer = size
        div.onclick = () => {

          // encrypted file, keep hidden until approved
          if (ctx.encryption && p.ownership.default === 0 && !game.user.isGM) {
            lain.querySelector(`.terminal-title`).textContent = data.title
            data.title = "decrypt a file"
            data.pid = p._id
            describe(data, p.name)
            data.jid = ctx.journal
            data.index = newPointer
            data.content = div.textContent
            delete data.ASCII
            skillCheck(data)

            game.socket.emit('module.terminal', {
              tuid: ctx.tuid,
              action: "shadowBtn",
              injectObj: {
                type: "text",
                text: {
                  content: textbox.textContent
                },
                name: actionLabel(data.title)
              },
            })
            return
          }
          styleClick(newPointer)

          if (ctx.encryption && p.ownership.default === 0 && game.user.isGM) {
      ui.notifications.info(t("TERMINAL.Action.GMSkipEncryption"))
          }
          injectHTML(p)
          game.socket.emit('module.terminal', {
            tuid: ctx.tuid,
            action: "shadowBtn",
            injectObj: p,
          })
        }
        sibling.appendChild(div)
        size++
      }

      // remove spaces
      const unsplit = ctx.unlockIds?.replace(/\s/g, '')
      let wallIteration = 0
      for (const uuid of unsplit?.split(",") || []) {
        if (!uuid) continue
        const wall = await validateDoor(uuid)
        if (!wall) return
        const div = document.createElement("div")
        div.dataset.wid = uuid
        const name = wall.getFlag("terminal", "name")
        div.textContent = name ? name : t("TERMINAL.Button.AlterDoor", { number: wallIteration + 1 })
        div.className = "terminal-button terminal-wall-btn"
        div.dataset.wallName = name
        div.dataset.wallId = wall.id
        const newPointer = size
        div.onclick = async () => {
          const w = fromUuidSync(uuid)
          const ascii = w.ds === CONST.WALL_DOOR_STATES.LOCKED ? ASCII.DOOR_OPEN : ASCII.DOOR_LOCK
          data.wid = uuid
          if (game.user.isGM) {
            showContent(newPointer, ascii, div.textContent)
            toggleDoorLock(w, "GM")
            return
          }
          if (ctx.doorCheck) {
            data.title = "lock or unlock door"
            describe(data, name, "TERMINAL.Button.AlterDoor", { number: wallIteration + 1 })
            data.index = newPointer
            data.ASCII = ascii
            data.content = div.textContent
            skillCheck(data)
            return
          }
          showContent(newPointer, ascii, div.textContent)
          game.socket.emit('module.terminal', {
            ...data,
            action: "toggleDoorLock",
          })
        }
        size++
        sibling.appendChild(div)
        wallIteration++
      }

      if (ctx.macro && game.macros.get(ctx.macro)) {
        const macro = game.macros.get(ctx.macro)
        const div = document.createElement("div")
        div.className = "terminal-button terminal-macro-btn"
        div.textContent = ctx.macroName || macro.name
        div.dataset.id = ctx.macro
        const newPointer = size
        div.onclick = async () => {
          const verifyLock = await fromUuid(ctx.tuid)
          let ascii = ASCII.MACRO
          if (verifyLock.getFlag(ID, "lockableMacro") && verifyLock.getFlag(ID, "macroLocked")) {
            ascii = ASCII.MACRO_DENY
          }
          data.macroName = ctx.macroName || macro.name
          data.macroNoProxy = ctx.macroNoProxy
          data.triggeredBy = data.uid
          if (ctx.macroCheck && !game.user.isGM) {
            data.title = "run script"
            describe(data, div.textContent)
            data.index = newPointer
            data.ASCII = ascii
            data.content = div.textContent
            skillCheck(data)
          } else if (game.user.isGM) {
            showContent(newPointer, ascii, div.textContent)
    if (ctx.macroCheck && game.user.isGM) ui.notifications.info(t("TERMINAL.Action.GMSkipCheck"))
            runMacro(ctx.tuid, data.uid, data.uid)
          } else {
            showContent(newPointer, ascii, div.textContent)
            if (ctx.macroNoProxy) {
              runMacro(ctx.tuid, data.uid, data.uid)
            }
            game.socket.emit('module.terminal', {
              ...data,
              action: "macro",
            })
          }
        }
        size++
        sibling.appendChild(div)
      }

      if (ctx.observeActor && game.actors.get(ctx.observeActor)) {
        const actorName = game.actors.get(ctx.observeActor)?.name
        const div = document.createElement("div")
        div.className = "terminal-button terminal-observe-btn"
        div.textContent = ctx.observeActorName || t("TERMINAL.Button.ControlActor", { name: actorName })
        div.dataset.actor = ctx.observeActor
        div.dataset.actorName = ctx.observeActorName || actorName
        const newPointer = size
        div.onclick = () => {
          if (ctx.observeCheck && !game.user.isGM) {
            data.title = "observe Actor"
            data.observeActor = ctx.observeActor
            data.observeTimer = ctx.observeTimer
            describe(data, ctx.observeActorName, "TERMINAL.Button.ControlActor", { name: actorName })
            data.index = newPointer
            data.ASCII = ASCII.EYE
            data.content = div.textContent
            skillCheck(data)
          } else {
            if (game.user.isGM) {
    if (ctx.observeCheck) ui.notifications.info(t("TERMINAL.Action.GMSkipCheck"))
              startObservation(ctx.observeActor, ctx.observeTimer, null, ctx.tuid)
            }
            showContent(newPointer, ASCII.EYE, div.textContent)
            game.socket.emit('module.terminal', {
              ...data,
              action: "startObservation",
              observeActor: ctx.observeActor,
              observeTimer: ctx.observeTimer,
            })
          }
        }
        size++
        sibling.appendChild(div)
      }

      if (ctx.alienCharge) {
        const div = document.createElement("div")
        div.className = "terminal-button terminal-charge-btn"
        div.textContent = t("TERMINAL.Button.Charge")
        const newPointer = size
        div.onclick = () => {
          showContent(newPointer, ASCII.BATTERY, t("TERMINAL.Action.AllBatteriesCharged"))
          if (game.user.isGM) {
            // GM owns all tokens, this is far from intended
            ui.notifications.error(t("TERMINAL.Error.GMChargeUnsupported"))
            return
          }
          const actorIds = game.actors.filter(actor => actor.hasPlayerOwner)
          game.socket.emit('module.terminal', {
            ...data,
            action: "alienCharge",
            actorIds,
          })
        }
        size++
        sibling.appendChild(div)
      }

      for (const regionData of ctx.regions || []) {
        const region = fromUuidSync(regionData.uuid)
        if (!region) continue
        const div = document.createElement("div")
        div.textContent = regionData.btnName ? regionData.btnName : t("TERMINAL.Button.RunScript")
        div.className = "terminal-button terminal-region-btn"
        div.dataset.region = regionData.uuid
        div.dataset.regionId = "r-" + String(Math.random()).slice(2, 10) + regionData.id
        div.dataset.regionName = regionData.btnName || "script-" + String(Math.random()).slice(2, 6)
        const newPointer = size
        div.onclick = async () => {
          if (ctx.regionCheck && !game.user.isGM) {
            data.title = "execute script"
            data.regionUUID = regionData.uuid
            data.event = regionData.event
            describe(data, regionData.btnName, "TERMINAL.Button.RunScript")
            data.index = newPointer
            data.ASCII = ASCII.MACRO
            data.content = div.textContent
            skillCheck(data)
          } else {
    if (ctx.regionCheck && game.user.isGM) ui.notifications.info(t("TERMINAL.Action.GMSkipCheck"))
            showContent(newPointer, ASCII.MACRO, div.textContent)
            triggerRegion(regionData.uuid, regionData.event, ctx.tuid)
          }
        }
        size++
        sibling.appendChild(div)
      }

      if (ctx.ping) {
        const div = document.createElement("div")
        div.className = "terminal-button terminal-ping-btn"
        div.textContent = t("TERMINAL.Button.DetectMotion")
        const newPointer = size
        div.onclick = () => {
          if (ctx.pingCheck && !game.user.isGM) {
            data.title = "detect motion"
            data.limit = ctx.pingRange
            data.index = newPointer
            data.ASCII = ASCII.MOTION
            data.content = div.textContent
            describe(
              data,
              null,
              ctx.pingRange ? "TERMINAL.Skill.MotionDescription" : "TERMINAL.Skill.MotionDescriptionNoLimit",
              ctx.pingRange ? { scene: canvas.scene.name, limit: ctx.pingRange } : { scene: canvas.scene.name },
            )
            skillCheck(data)
          } else {
    if (ctx.pingCheck && game.user.isGM) ui.notifications.info(t("TERMINAL.Action.GMSkipCheck"))
            showContent(newPointer, ASCII.MOTION, div.textContent)
            detectMotion(null, ctx.tuid, ctx.pingRange) // can be ran from any client
          }
        }
        size++
        sibling.appendChild(div)
      }

      if (ctx.monk && game.modules.get("monks-active-tiles")?.active) {
        const div = document.createElement("div")
        div.className = "terminal-button terminal-monk-btn"
        div.textContent = ctx.monkName || t("TERMINAL.Button.RunScript")
        div.dataset.monkName = ctx.monkName || "script-" + String(Math.random()).slice(2, 6)
        div.dataset.fakeId = "m-" + String(Math.random()).slice(2, 16)
        const newPointer = size
        div.onclick = () => {
          if (ctx.macroCheck && !game.user.isGM) {
            data.title = "run script"
            data.monk = true
            data.macroName = "Monks Active Tile Triggers"
            describe(data, null, "TERMINAL.Skill.MATTDescription")
            data.index = newPointer
            data.ASCII = ASCII.MACRO
            data.content = div.textContent
            skillCheck(data)
          } else {
    if (ctx.macroCheck && game.user.isGM) ui.notifications.info(t("TERMINAL.Action.GMSkipCheck"))
            showContent(newPointer, ASCII.MACRO, div.textContent)
            tile.trigger({}) // ran as either GM or player
          }
        }
        size++
        sibling.appendChild(div)
      }

      if (ctx.lights) {
        const div = document.createElement("div")
        div.className = "terminal-button terminal-lights-btn"
        div.textContent = t("TERMINAL.Button.TogglePower")
        const newPointer = size
        div.onclick = () => {
          const ascii = tile.parent.darkness < 0.5 ? ASCII.POWER_OFF : ASCII.POWER_ON
          if (ctx.lightCheck && !game.user.isGM) {
            data.title = "toggle power"
            data.sid = tile.parent.id
            describe(data, null, "TERMINAL.Skill.SceneDescription", { scene: canvas.scene.name })
            data.index = newPointer
            data.ASCII = ascii
            data.content = div.textContent
            skillCheck(data)
          } else {
            if (game.user.isGM) {
    if (ctx.lightCheck) ui.notifications.info(t("TERMINAL.Action.GMSkipCheck"))
              toggleLights(tile.parent.id)
              return
            }
            game.socket.emit('module.terminal', {
              ...data,
              sid: tile.parent.id,
              action: "lights"
            })
            showContent(newPointer, ascii, div.textContent)
          }
        }
        size++
        sibling.appendChild(div)
      }

      let tilesIteration = 0
      for (const uuid of ctx.ssh?.split(",") || []) {
        if (!uuid) continue
        const tile = fromUuidSync(uuid)
        if (!tile) return
        const div = document.createElement("div")
        const name = tile.getFlag("terminal", "name")
        div.textContent = name ? name : t("TERMINAL.Button.SecureShell", { number: tilesIteration + 1 })
        div.className = "terminal-button terminal-ssh-btn"
        div.dataset.targetTile = name || uuid.split(".")[3]
        const newPointer = size
        div.onclick = async () => {

          // could also use foundry.applications
          if (document.querySelector(`.${uuid.replace(/\./g, '-')}`)) {
            ui.notifications.error(t("TERMINAL.Error.SSHRunning"))
            return
          }

          const bypass = tile.getFlag(ID, "keycard") || tile.getFlag(ID, "skilled")
          if (ctx.sshCheck && !game.user.isGM) {
            data.title = "start a secure shell"
            data.ssh = uuid
            data.bypass = bypass
            describe(data, name, "TERMINAL.Button.SecureShell", { number: tilesIteration + 1 })
            data.index = newPointer
            data.ASCII = ASCII.SSH
            data.content = div.textContent
            skillCheck(data)
          } else {
            if (ctx.sshCheck && game.user.isGM) {
    if (ctx.sshCheck) ui.notifications.info(t("TERMINAL.Action.GMSkipCheck"))
            }
            // warn about bypass
            if (game.settings.get('terminal', "notice") && bypass) {
              if (game.user.isGM) {
                ui.notifications.info(t("TERMINAL.Action.SSHBypass"))
              } else {
                game.socket.emit('module.terminal', {
                  action: "notify",
                  messageKey: "TERMINAL.Action.UserSSHBypass",
                  messageData: { name: game.user.name },
                })
              }
            }
            showContent(newPointer, ASCII.SSH, div.textContent)
            autoShadowTerminal(uuid, { isSSH: true })
            new Terminal(uuid, { isSSH: true }).render(true)
          }
        }
        if (tile) {
          size++
          sibling.appendChild(div)
          tilesIteration++
        }
      }

      if (ctx.explore) {
        const div = document.createElement("div")
        div.className = "terminal-button terminal-explore-btn"
        div.textContent = t("TERMINAL.Button.DownloadMap")
        const newPointer = size
        div.onclick = () => {
          if (ctx.exploreCheck && !game.user.isGM) {
            data.title = "download map"
            data.exploreAll = ctx.exploreAll
            describe(data, null, "TERMINAL.Skill.MapDescription", { scene: canvas.scene.name })
            data.index = newPointer
            data.ASCII = ASCII.ATLAS
            data.content = div.textContent
            skillCheck(data)
          } else {
    if (ctx.exploreCheck && game.user.isGM) ui.notifications.info(t("TERMINAL.Action.GMSkipCheck"))
            exploreMap(ctx.exploreAll)
            showContent(newPointer, ASCII.ATLAS, div.textContent)
          }
        }
        sibling.appendChild(div)
      }

      if (arg.shadowView) {
        sibling.querySelectorAll("div").forEach(d => d.onclick = null)
      }

      // Apply styles
      const textbox = lain.querySelector(`.terminal-textbox`)
      textbox.style.border = `3px solid ${ctx.base}`;
      textbox.style.borderTop = `20px solid ${ctx.base}`
      const title = lain.querySelector(`.terminal-title`)
      title.style.backgroundColor = ctx.shadowView
      title.style.color = ctx.highlight
      const allBtns = lain.querySelectorAll(`.terminal-button`)
      allBtns.forEach(btn => {
        btn.style.borderTop = `2px solid ${ctx.base}`
        btn.style.borderBottom = `2px solid ${ctx.base}`
        btn.style.borderLeft = `10px solid ${ctx.base}`
        btn.style.borderRight = `10px solid ${ctx.base}`
      })
      lain.querySelector(`.terminal-buttons`).style.color = ctx.highlight

      if (ctx.effectScan) {
        const hex = ctx.base.replace(/^#/, '')
        var r = parseInt(hex.slice(0, 2), 16)
        var g = parseInt(hex.slice(2, 4), 16)
        var b = parseInt(hex.slice(4, 6), 16)
        lain.querySelector('.terminal-effects-scan').style.backgroundImage = `linear-gradient(
          0deg, transparent 0%,
          rgba(${r}, ${g}, ${b}, 0.05) 2%,
          rgba(${r}, ${g}, ${b}, .9) 3%,
          rgba(${r}, ${g}, ${b}, 0.2) 4%,
          transparent 20%
        )`
      }
    }

    async function buildPageFragment(p, { ctx, gsap, target = "textbox" } = {}) {
      const fragment = document.createDocumentFragment()
      let plainText = `[${p.type}] ${p.name}`
      let rootEl = null
      const title = p.name ?? ''

      if (p.type === "video") {
        const video = document.createElement("video")
        video.style.width = "100%"
        if (target === "history") video.style.maxHeight = "40vh"
        video.controls = p.video?.controls ?? true
        video.loop = !!p.video?.loop
        video.autoplay = !!p.video?.autoplay
        if (typeof p.video?.volume === 'number') video.volume = p.video.volume
        const source = document.createElement("source")
        source.src = p.src
        video.appendChild(source)
        fragment.appendChild(video)
        rootEl = video
        plainText = `[video] ${title} <${p.src || ""}>`

      } else if (p.type === "image") {
        const wrap = document.createElement("div")
        const img = document.createElement("img")
        img.src = p.src
        img.style.maxWidth = "100%"
        if (target === "history") img.style.maxHeight = "40vh"
        const caption = document.createElement("p")
        caption.textContent = p.image?.caption || ""
        caption.style.textAlign = "center"
        wrap.appendChild(img)
        wrap.appendChild(caption)
        fragment.appendChild(wrap)
        rootEl = wrap
        plainText = `[image] ${title} <${p.src || ""}> ${p.image?.caption ? `— ${p.image.caption}` : ""}`

      } else if (p.type === "pdf") {
        const frame = document.createElement("iframe")
        const params = new URLSearchParams()
        if (p.src) {
          const src = URL.parseSafe(p.src) ? p.src : foundry.utils.getRoute(p.src)
          params.append("file", src)
        }
        frame.src = `scripts/pdfjs/web/viewer.html?${params}`
        frame.setAttribute("loading", "lazy")
        frame.style.display = "block"
        frame.style.border = "0"

        if (target === "history") {
          // narrower + margins so scrolling past is easier
          frame.style.width = "90%"
          frame.style.height = "50vh"
          frame.style.margin = "8px auto"

          const wrap = document.createElement("div")
          wrap.style.margin = "8px 0"          // extra vertical spacing
          wrap.appendChild(frame)
          fragment.appendChild(wrap)
          rootEl = wrap
        } else {
          frame.style.width = "100%"
          frame.style.height = "99%"
          fragment.appendChild(frame)
          rootEl = frame
        }

        plainText = `[pdf] ${title} <${p.src || ""}>`
      } else {
        const div = document.createElement("div")
        div.id = "terminal-text"
        let richHTML = await foundry.applications.ux.TextEditor.implementation.enrichHTML(p.text?.content ?? "")

        const parser = new DOMParser()
        if (p.type === "loading") {
          richHTML = `
          <h1 style="font-size: 2.2em" class="terminal-rm">${t("TERMINAL.Dialog.Authenticating")}</h1>
            <div style="display: flex; justify-content: center; " class="terminal-rm">
              <i style="font-size: 2em; margin: 1em" class="fa-duotone fa-spinner-third fa-spin"></i>
            </div>
          `
          game.socket.emit("module.terminal", JSON.parse(p.data))
        }
        const htmlDoc = parser.parseFromString(richHTML, "text/html")
        htmlDoc.querySelector("button")?.addEventListener("click", p.callback)

        // puzzle locks
        htmlDoc.querySelectorAll("[data-uuid]").forEach(e => {
          if (!game.modules.get("puzzle-locks")?.active) return
          fromUuid(e.getAttribute("data-uuid")).then(doc => {
            if (doc?.flags["puzzle-locks"]?.general?.unlocked) return
            e.onclick = () => foundry.applications.instances.forEach(a => a.minimize())
          })
        })

        const df = document.createDocumentFragment()
        for (const node of Array.from(htmlDoc.body.childNodes)) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            const tag = node.tagName
            if ((tag === "H1" || tag === "H2" || tag === "H3") && node.children.length === 0) {
              node.classList.add("terminal-typewriter")
            }
          }
          df.appendChild(node)
        }
        div.appendChild(df)
        fragment.appendChild(div)
        rootEl = div
        plainText = (htmlDoc.body.textContent || "").trim()
      }

      const postMount = () => {
        if (!ctx?.effectScramble || !gsap) return
        const words = rootEl?.querySelectorAll?.(".terminal-typewriter") || []
        const tl = gsap.timeline()
        const shuffle = str => [...str].sort(() => Math.random() - 0.5).join('')
        words.forEach(el => {
          if (!el.textContent?.length) return
          const original = el.innerHTML
          gsap.to(el, { duration: 0, text: { value: shuffle(el.textContent) } })
          tl.to(el, {
            duration: Math.max(0.2, original.length * 0.1),
            scrambleText: { text: original, chars: "abcdefghijklmnopqrstuvwxyz ", ease: "none" }
          })
        })
      }
      return { title, kind: p.type, fragment, plainText, postMount }
    }

    async function injectHTML(p) {
      if (typeof p.text?.content === "undefined" && p.type === "text") return
      const { title, fragment, postMount } = await buildPageFragment(p, { ctx, gsap, target: ctx.cliMode ? "history" : "textbox" })

      const titleEl = lain.querySelector('.terminal-title')
      const textbox = lain.querySelector('.terminal-textbox')
      if (ctx.cliMode) {
        const history = lain.querySelector('.terminal-history')
        const block = document.createElement('div');
        block.className = 'terminal-line terminal-block';
        block.style.whiteSpace = 'normal';
        block.style.fontFamily = 'inherit';
        block.style.padding = '4px 0';
        block.appendChild(fragment);
        history.appendChild(block);
        setTimeout(() => {
          history.scrollTop = history.scrollHeight;
          history.scrollTo({ top: history.scrollHeight, behavior: 'smooth' });
        }, 20);
      } else {
        titleEl.textContent = title
        textbox.innerHTML = ''
        textbox.appendChild(fragment)
      }
      postMount()
    }

    function createHTML() {
      if (!checkAccess()) return
      createButtons()
      fakeLoading()

      // have shadows bypass password
      if (ctx.password) {
        game.socket.emit('module.terminal', {
          tuid: ctx.tuid,
          action: "shadowBtn",
          password: ctx.password,
        })
      }

      // play startup audio
      if (ctx.startup) {
        audio.startup = new foundry.audio.Sound(ctx.startup)
        audio.startup.load().then(s => s.play({ volume: game.settings.get("core", "globalInterfaceVolume") }))
      }
    }

    function replaceHTML(pid, manualObj) {
      const page = game.journal.get(ctx.journal).pages.get(pid) || manualObj
      injectHTML(page)
    }

    function checkAccess() {
      const j = game.journal.get(ctx.journal)
      const gms = game.users?.filter(u => u.active && u.isGM)
      if (!gms.length) {
        ui.notifications.error(t("TERMINAL.Error.GMRequired"))
        return
      }
      if (j.permission < CONST.DOCUMENT_OWNERSHIP_LEVELS.OBSERVER) {
        game.socket.emit('module.terminal', {
          action: "updateJournal",
          uid: game.user.id,
          jid: ctx.journal,
          tuid: ctx.tuid,
          gid: gms[0]._id
        })
        return false
      }
      return true
    }

    function takePassword() {
      lain.querySelector(".terminal-loading").style.display = "none"
      lain.querySelector(".terminal-ascii").style.display = "grid"
      const pword = lain.querySelector(".terminal-password")
      pword.focus()
      pword.addEventListener("keyup", e => {
        if (e.key === "Enter") {
          if (e.target.value === ctx.password) {
            lain.querySelector(`.terminal-loading`).style.display = "block"
            lain.querySelector(`.terminal-password-container`).style.display = "none"
            createHTML()
            initCLI()
          } else {
      ui.notifications.error(t("TERMINAL.Error.IncorrectLogin"))
            pword.value = ""
            pword.focus()
          }
        }
      })
    }

    function fakeLoading() {
      const loadingBar = lain.querySelector(".terminal-bar")
      let width = 0
      lain.querySelector(".terminal-splash").addEventListener("click", () => {
        stopLoading()
      })
      const interval = setInterval(() => {
        if (width >= 100) {
          clearInterval(interval)
          stopLoading()
        } else {
          width += 2
          if (loadingBar) {
            loadingBar.style.width = width + '%'
          }
        }
      }, 50)
    }

    function initCLI(opts = {}) {
      const cliRoot = lain.querySelector('.terminal-cli')
      if (!cliRoot) return
      let username = cliRoot.dataset.username || opts.username || (game?.user?.name ?? 'user')
      username = slug(username)
      // illegal username
      if (username.includes("help")) {
        username = "user"
      }
      const host = cliRoot.dataset.host || opts.host || 'host'
      const history = cliRoot.querySelector('.terminal-history')
      const input = cliRoot.querySelector('.terminal-input')
      const prompt = cliRoot.querySelector('.terminal-prompt')

      // Config
      const cps = Number(opts.cps ?? 80)                    // chars per second
      const scanlineEnabled = opts.scanlineEnabled !== false
      const scanlineSize = Number(opts.scanlineSize ?? 140) // px
      const scanlineDuration = Number(opts.scanlineDuration ?? 180) // ms
      let currentRun = 0;
      const active = {
        tweens: new Set(),    // gsap tweens
        anims: new Set(),     // Web Animations
        timeouts: new Set(),  // setTimeout ids
        observers: new Set(), // ResizeObservers etc.
      }

      prompt.textContent = `${username}@${host}: $ `

      function cancelActive() {
        active.tweens.forEach(t => { try { t.kill?.(); } catch { } });
        active.anims.forEach(a => { try { a.cancel?.(); } catch { } });
        active.timeouts.forEach(id => clearTimeout(id));
        active.observers.forEach(o => { try { o.disconnect?.(); } catch { } });
        active.tweens.clear(); active.anims.clear(); active.timeouts.clear(); active.observers.clear();
      }
      const addTimeout = (fn, ms) => {
        const id = setTimeout(() => { active.timeouts.delete(id); fn(); }, ms);
        active.timeouts.add(id);
        return id;
      };

      const getLineHeightPx = (el) => {
        const s = getComputedStyle(el)
        const lh = s.lineHeight
        if (!lh || lh === 'normal') {
          const fs = parseFloat(s.fontSize) || 16
          return Math.round(fs * 1.3)
        }
        return Math.round(parseFloat(lh))
      }

      const swoosh = (lineEl, runId) => {
        if (!scanlineEnabled) return Promise.resolve();
        if (runId !== currentRun) return Promise.resolve();
        return new Promise((resolve) => {
          const r = parseInt(ctx.highlight.slice(1, 3), 16);
          const g = parseInt(ctx.highlight.slice(3, 5), 16);
          const b = parseInt(ctx.highlight.slice(5, 7), 16);
          const h = getLineHeightPx(lineEl);

          const scan = document.createElement('div');
          scan.style.cssText = `
            position: absolute;
            width: ${scanlineSize}px;
            height: ${h}px;
            background: radial-gradient(circle, ${ctx.highlight} 50%, rgba(${r}, ${g}, ${b}, 0.7) 70%, transparent 90%);
            left: 100%;
            top: 0;
            filter: blur(2px) brightness(0.95);
            opacity: 1;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 0 10px ${ctx.highlight}, 0 0 20px ${ctx.highlight};
          `;
          lineEl.style.position = lineEl.style.position || 'relative';
          lineEl.appendChild(scan);

          const anim = scan.animate(
            [
              { left: '100%', filter: 'blur(2px) brightness(0.9)' },
              { left: `-${scanlineSize}px`, filter: 'blur(3px) brightness(1.0)' }
            ],
            { duration: scanlineDuration, easing: 'linear' }
          );
          active.anims.add(anim);

          const done = () => { active.anims.delete(anim); scan.remove(); resolve(); };

          anim.onfinish = done;
          // If this run gets preempted mid-animation:
          addTimeout(() => {
            if (runId !== currentRun) { try { anim.cancel(); } catch { } done(); }
          }, 0);
        });
      };

      function typeOutCancelable(el, text, cps, runId) {
        return new Promise((resolve) => {
          // GSAP path
          if (window.gsap && gsap.plugins?.TextPlugin) {
            const tween = gsap.to(el, {
              duration: Math.max(0.05, text.length / cps),
              text: { value: text },
              ease: 'none',
              onComplete: resolve,
            });
            // resolve if killed
            tween.eventCallback?.('onInterrupt', resolve);
            active.tweens.add(tween);
            // immediate preemption
            if (runId !== currentRun) { tween.kill(); return resolve(); }
            return;
          }
          // Fallback manual typer
          el.textContent = '';
          let i = 0;
          const tick = () => {
            if (runId !== currentRun) return resolve();
            el.textContent += text[i++];
            if (i < text.length) {
              const t = addTimeout(tick, 1000 / cps);
              // addTimeout already tracks
            } else resolve();
          };
          tick();
        });
      }

      // Create a new line and type text into it, with a pre-swoosh
      async function addTypedLine(text, runId) {
        if (runId !== currentRun) return;
        const line = document.createElement('div');
        line.className = `terminal-line`
        line.style.minHeight = '1.4em';
        line.style.position = 'relative';
        history.appendChild(line);
        history.scrollTop = history.scrollHeight;

        await swoosh(line, runId);
        if (runId !== currentRun) return;

        await typeOutCancelable(line, text, cps, runId);
        if (runId !== currentRun) return;

        history.scrollTop = history.scrollHeight;
      }


      // help
      // Builds the fake filesystem path shown for a script/actor button. Both the displayed path and
      // the path the user types are slugged through here, so they always agree. Unicode letters are
      // kept: `[^a-zA-Z0-9]` erased CJK names entirely, collapsing every script onto one empty path.
      function slug(value) {
        return (value ?? "").replace(/[^\p{L}\p{N}]/gu, "").toLowerCase()
      }

      function generateTable(headers, rows) {
        const tableLines = [];
        // A CJK glyph is one code unit but two terminal cells, so String.length would leave every
        // Chinese column short. Measure in cells and pad to that width instead.
        const displayWidth = value => [...value].reduce((total, char) => total + (/[ᄀ-ᅟ⺀-〾ぁ-㏿㐀-䶿一-鿿ꀀ-꓏가-힣豈-﫿︰-﹯＀-｠￠-￦]/.test(char) ? 2 : 1), 0);
        const pad = (value, width, padChar) => value + padChar.repeat(Math.max(0, width - displayWidth(value)));

        const columnWidths = headers.map((header, i) => {
          const columnValues = rows.map(row => row[i].toString());
          return Math.max(displayWidth(header), ...columnValues.map(value => displayWidth(value)));
        });

        const generateLine = (row, padChar = ' ') => '| ' + row.map((cell, index) => pad(cell.toString(), columnWidths[index], padChar)).join(' | ') + ' |';

        const divider = generateLine(headers.map((_, index) => '-'.repeat(columnWidths[index])), '-');

        tableLines.push(divider);
        tableLines.push(generateLine(headers));
        tableLines.push(divider);
        for (const row of rows) {
          tableLines.push(generateLine(row));
        }
        tableLines.push(divider);

        return tableLines;
      }
      const availableCommands = [["ls", t("TERMINAL.CLI.ListContents")], ["cat [file]", t("TERMINAL.CLI.DisplayFile")]]
      if (ctx.ssh) {
        availableCommands.push(["ssh", t("TERMINAL.CLI.ConnectTerminal")]);
      }
      if (ctx.macro || ctx.monk || ctx.regions?.length) {
        availableCommands.push(["sh", t("TERMINAL.CLI.RunShell")]);
      }
      if (ctx.lights) {
        availableCommands.push(["power", t("TERMINAL.CLI.TogglePower")]);
      }
      if (ctx.explore) {
        availableCommands.push(["map", t("TERMINAL.CLI.DownloadMap")]);
      }
      if (ctx.ping) {
        availableCommands.push(["ping", t("TERMINAL.CLI.PingHosts")]);
      }
      if (ctx.observeActor) {
        availableCommands.push(["control", t("TERMINAL.CLI.ControlActor")]);
      }
      if (ctx.alienCharge) {
        availableCommands.push(["charge", t("TERMINAL.CLI.ChargeBatteries")]);
      }
      if (ctx.unlockIds) {
        availableCommands.push(["door", t("TERMINAL.CLI.DoorControl")]);
      }
      const helpLines = generateTable([t("TERMINAL.CLI.Command"), t("TERMINAL.CLI.Description")], availableCommands)
      helpLines.forEach(line => {
        const div = document.createElement('div');
        div.className = 'terminal-line';
        div.style.minHeight = '1.4em';
        div.style.position = 'relative';
        div.innerHTML = line; // innerHTML so &lt;file&gt; is rendered as <file>
        history.appendChild(div);
      })
      lain.querySelector(`button[data-action="help"]`).addEventListener("click", async () => {
        const runId = ++currentRun
        await addTypedOutput(helpLines.join('\n'), runId)
      })

      // Supports multi-line outputs; each line gets its own swoosh + type
      async function addTypedOutput(textOrLines, runId) {
        const lines = Array.isArray(textOrLines) ? textOrLines : String(textOrLines).split('\n');
        for (const ln of lines) {
          if (runId !== currentRun) break;
          await addTypedLine(ln, runId);
        }
      }

      input.addEventListener('keydown', async (e) => {
        if (e.key === 'ArrowUp') {
          const nodes = document.querySelectorAll(".terminal-echo");
          const last = nodes[nodes.length - 1]
          if (!last) return
          input.value = last.textContent.replace(`${username}@${host}: $ `, '').trim()
          requestAnimationFrame(() => {
            input.setSelectionRange(input.value.length, input.value.length);
          })
          return
        }
        if (e.key !== 'Enter') return;
        const cmd = input.value.trim();
        if (!cmd) return;

        currentRun += 1;
        cancelActive();
        const runId = currentRun;

        // Echo
        const cmdLine = document.createElement('div');
        cmdLine.className = 'terminal-line terminal-echo';
        cmdLine.textContent = `${username}@${host}: $ ${cmd}`;
        cmdLine.style.minHeight = '1.4em';
        history.appendChild(cmdLine);
        history.scrollTop = history.scrollHeight;

        input.value = '';

        // commands
        if (cmd.startsWith('cat ') || cmd === "cat") {
          if (cmd === "cat") {
            await addTypedOutput(t("TERMINAL.CLI.UsageCat"), runId);
            return;
          } else if (cmd === "cat /etc/os-release") {
            await addTypedOutput(`arch btw`, runId);
            return;
          }
          const arg = cmd.slice(4).trim();
          const buttons = Array.from(lain.querySelectorAll('.terminal-journal-page'));
          let page, btn

          if (/^\d+$/.test(arg)) {
            const idx = parseInt(arg, 10) - 1;
            btn = buttons[idx];
            page = game.journal.get(ctx.journal)?.pages?.get(btn?.dataset?.pid) ?? null;
          } else {
            const norm = s => s.trim().toLowerCase().replace(/^"(.*)"$/, '$1');
            btn = buttons.find(b => norm(b.textContent) === norm(arg));
            page = game.journal.get(ctx.journal)?.pages?.get(btn?.dataset?.pid) ?? null;
          }

          if (!page) {
            await addTypedOutput(t("TERMINAL.CLI.FileNotFound", { path: arg }), runId);
            return;
          }

          btn?.click();

        } else if (cmd === 'clear') {
          history.innerHTML = '';
        } else if (cmd === 'help' || cmd.startsWith('help ')) {
          await addTypedOutput(helpLines.join('\n'), runId);
        } else if (cmd === 'ssh' || cmd.startsWith('ssh ')) {
          const arg = cmd.slice(4).trim()
          const btns = lain.querySelectorAll('.terminal-ssh-btn');
          if (arg && !arg.includes("help")) {
            const index = arg.split(".")[3]
            const btn = btns[Number(index) - 10];
            if (btn) {
              btn.click();
            } else {
              await addTypedOutput(t("TERMINAL.CLI.SSHTimeout", { host: arg }), runId);
            }
          } else {
            const table = generateTable([t("TERMINAL.CLI.Header.IP"), t("TERMINAL.CLI.Header.Hostname")], Array.from(btns).map((el, i) => [`192.168.0.${Number(i) + 10}`, el.dataset.targetTile]))
            await addTypedOutput(btns.length ? t("TERMINAL.CLI.UsageSSH", { user: username }) : t("TERMINAL.CLI.NoConnections"), runId)
            if (!btns.length) return
            await addTypedOutput(table.join('\n'), runId);
          }
        } else if (cmd === 'sh' || cmd.startsWith('sh ')) {
          const arg = cmd.slice(3).trim()
          const btn = lain.querySelector('.terminal-macro-btn');
          const monkBtn = lain.querySelector('.terminal-monk-btn');
          const regionBtns = lain.querySelectorAll('.terminal-region-btn');
          if (!btn && !monkBtn && !regionBtns.length) {
            await addTypedOutput(t("TERMINAL.CLI.NoScripts"), runId);
            return
          }
          const path = `/usr/local/bin/${slug(btn?.textContent)}`
          const monkPath = `/usr/local/bin/${slug(monkBtn?.dataset.monkName)}`
          // const monkPath = `/usr/local/bin/${slug(monkBtn?.dataset.monkName)}`
          if (arg && !arg.includes("help")) {
            const name = arg.split("/")[4] || ""
            let found = false
            Array.from(regionBtns).forEach(btn => {
              const regionPath = `/usr/local/bin/${slug(btn.dataset.regionName)}`
              if (regionPath === `/usr/local/bin/${name}`) {
                btn.click()
                found = true
                return
              }
            })
            if (found) return
            if (path === `/usr/local/bin/${name}`) {
              btn.click();
            } else if (monkPath === `/usr/local/bin/${name}`) {
              monkBtn.click();
            } else {
              await addTypedOutput(t("TERMINAL.CLI.ShellFileNotFound", { path: arg }), runId);
            }
          } else {
            let subtable = []
            if (btn) subtable.push([path, btn.dataset.id])
            if (monkBtn) subtable.push([monkPath, monkBtn.dataset.fakeId])
            Array.from(regionBtns).forEach(btn => {
              const regionPath = `/usr/local/bin/${slug(btn.dataset.regionName)}`
              subtable.push([regionPath, btn.dataset.regionId])
            })
            const table = generateTable([t("TERMINAL.CLI.Header.Path"), t("TERMINAL.CLI.Header.ID")], subtable)
            await addTypedOutput(t("TERMINAL.CLI.UsageShell"), runId)
            await addTypedOutput(table.join('\n'), runId);
          }
        } else if (cmd === 'pwd' || cmd === 'cwd') {
          await addTypedOutput(`/home/${username}`, runId);
        } else if (cmd === 'dig' || cmd.startsWith('dig')) {
          await addTypedOutput(`dug`, runId);
        } else if (cmd.startsWith('sudo')) {
          await addTypedOutput(t("TERMINAL.CLI.SudoDenied"), runId);
        } else if (cmd === 'exit' || cmd === 'close' || cmd === 'quit' || cmd === 'shutdown') {
          await addTypedOutput(t("TERMINAL.CLI.Goodbye"), runId);
          setTimeout(() => lain.querySelector(`button[data-action="closeWindow"]`).click(), 1_000)
        } else if (cmd === 'minimize' || cmd === 'min') {
          lain.querySelector(`button[data-action="minimize"]`).click()
        } else if (cmd.startsWith('cd ') || cmd === "cd") {
          const arg = cmd.slice(3).trim();
          await addTypedOutput(t("TERMINAL.CLI.CdDenied", { path: arg }), runId);
        } else if (cmd.startsWith('mkdir ') || cmd === "mkdir") {
          const arg = cmd.slice(6).trim();
          await addTypedOutput(t("TERMINAL.CLI.MkdirDenied", { path: arg }), runId);
        } else if (cmd.startsWith('touch ') || cmd === "touch") {
          const arg = cmd.slice(6).trim();
          await addTypedOutput(t("TERMINAL.CLI.TouchDenied", { path: arg }), runId);
        } else if (cmd.startsWith('power')) {
          const btn = lain.querySelector('.terminal-lights-btn');
          if (!btn) {
            await addTypedOutput(t("TERMINAL.CLI.ExecuteDenied", { command: "power" }), runId);
            return
          }
          btn.click();
        } else if (cmd.startsWith('map')) {
          const btn = lain.querySelector('.terminal-explore-btn');
          if (!btn) {
            await addTypedOutput(t("TERMINAL.CLI.ExecuteDenied", { command: "map" }), runId);
            return
          }
          btn.click();
        } else if (cmd.startsWith('ping')) {
          const btn = lain.querySelector('.terminal-ping-btn');
          if (!btn) {
            let arg = cmd.slice(5).trim();
            if (arg === "") arg = "192.168.0.1";
            const out = [
              `PING ${arg} (${arg}) 56(84) bytes of data.`,
              `64 bytes from ${arg}: icmp_seq=1 ttl=64 time=0.126 ms`,
              `64 bytes from ${arg}: icmp_seq=2 ttl=64 time=0.130 ms`,
              `64 bytes from ${arg}: icmp_seq=3 ttl=64 time=0.171 ms`,
              `64 bytes from ${arg}: icmp_seq=4 ttl=64 time=0.155 ms`,
              `64 bytes from ${arg}: icmp_seq=5 ttl=64 time=0.139 ms`,
              `--- ${arg} ping statistics ---`,
              `5 packets transmitted, 5 received, 0% packet loss, time 4083ms`,
              `rtt min/avg/max/mdev = 0.126/0.144/0.171/0.016 ms`,
            ].join('\n');
            await addTypedOutput(out, runId);
            return
          }
          btn.click();
        } else if (cmd.startsWith('cp ') || cmd.startsWith('mv ') || cmd === "cp" || cmd === "mv") {
          const arg = cmd.slice(3).trim();
          await addTypedOutput(t("TERMINAL.CLI.CannotStat", { path: arg }), runId);
        } else if (cmd.startsWith('rm ') || cmd === "rm") {
          const arg = cmd.slice(3).trim();
          await addTypedOutput(t("TERMINAL.CLI.RemoveDenied", { path: arg }), runId);
        } else if (cmd.startsWith('control ') || cmd === "control") {
          const arg = cmd.slice(8).trim();
          const btn = lain.querySelector('.terminal-observe-btn');
          if (!btn) {
            await addTypedOutput(t("TERMINAL.CLI.NoControlDaemons"), runId);
            return
          }
          const path = `/usr/local/bin/${slug(btn.dataset.actorName)}`
          if (arg && !arg.includes("help")) {

            const actorName = arg.split("/")[4]
            if (path === `/usr/local/bin/${actorName}`) {
              btn.click();
            } else {
              await addTypedOutput(t("TERMINAL.CLI.ControlNotFound", { path: arg }), runId);
            }
          } else {
            const table = generateTable([t("TERMINAL.CLI.Header.Path"), t("TERMINAL.CLI.Header.ID")], [[path, btn.dataset.actor]])
            await addTypedOutput(t("TERMINAL.CLI.UsageControl"), runId)
            await addTypedOutput(table.join('\n'), runId);
          }
        } else if (cmd.startsWith('charge ') || cmd === "charge") {
          // TODO: test this works, needs Alien system
          const btn = lain.querySelector('.terminal-charge-btn');
          if (!btn) {
            await addTypedOutput(t("TERMINAL.CLI.ChargeDenied"), runId);
            return
          }
          btn.click();
        } else if (cmd.startsWith('door ') || cmd === "door") {
          const arg = cmd.slice(5).trim();
          const btns = lain.querySelectorAll('.terminal-wall-btn');
          if (arg && !arg.includes("help")) {
            const commands = arg.split(" ") || []
            if (commands.length !== 2 || (commands[0] !== "lock" && commands[0] !== "unlock")) {
              await addTypedOutput(t("TERMINAL.CLI.UsageDoor"), runId)
              return
            }
            const btn = Array.from(btns).find(el => el.dataset.wallId === commands[1])
            if (!btn) {
              await addTypedOutput(t("TERMINAL.CLI.DoorNotFound", { id: commands[1] }), runId)
              return
            }
            const wall = fromUuidSync(btn.dataset.wid)
            const locked = wall.ds === CONST.WALL_DOOR_STATES.LOCKED
            if (locked && commands[0] === "lock") {
              await addTypedOutput(t("TERMINAL.CLI.DoorAlreadyLocked", { id: commands[1] }), runId)
              return
            } else if (!locked && commands[0] === "unlock") {
              await addTypedOutput(t("TERMINAL.CLI.DoorAlreadyUnlocked", { id: commands[1] }), runId)
              return
            }
            btn.click();
          } else {
            if (!btns.length) {
              await addTypedOutput(t("TERMINAL.CLI.NoConnections"), runId);
              return
            }
            const table = generateTable([t("TERMINAL.CLI.Header.Name"), t("TERMINAL.CLI.Header.State"), t("TERMINAL.CLI.Header.ID")], Array.from(btns).map((el, i) => {
              const w = fromUuidSync(el.dataset.wid)
              const state = w.ds === CONST.WALL_DOOR_STATES.LOCKED ? t("TERMINAL.CLI.State.Locked") : t("TERMINAL.CLI.State.Unlocked")
              return [el.dataset.wallName, state, el.dataset.wallId]
            }))
            await addTypedOutput(t("TERMINAL.CLI.UsageDoor"), runId)
            await addTypedOutput(table.join('\n'), runId)
          }
        } else if (cmd.startsWith('chmod ') || cmd.startsWith('chown ') || cmd === "chmod" || cmd === "chown") {
          const arg = cmd.slice(6).trim();
          await addTypedOutput(t("TERMINAL.CLI.ChangePermissionsDenied", { path: arg }), runId);
        } else if (cmd.startsWith('echo ') || cmd === "echo") {
          const arg = cmd.slice(5).trim();
          await addTypedOutput(arg, runId);
        } else if (cmd === 'ls') {
          const btns = lain.querySelectorAll('.terminal-journal-page');
          if (!btns.length) {
            await addTypedOutput(t("TERMINAL.CLI.NoFiles"), runId)
            return
          }
          const table = generateTable([t("TERMINAL.CLI.Header.ID"), t("TERMINAL.CLI.Header.Filename"), t("TERMINAL.CLI.Header.FileType")], Array.from(btns).map((el, i) => {
            return [`${String(i + 1)}`, el.textContent.trim(), el.dataset.type]
          }))
          await addTypedOutput(t("TERMINAL.CLI.ReadUsage"), runId)
          await addTypedOutput(table.join('\n'), runId)

        } else if (cmd === 'ps' || cmd.startsWith('ps ')) {
          // PS (generic)
          const out = [
            `    PID TTY          TIME CMD`,
            `   1042 pts/1    00:00:01 sh`,
            `   2140 pts/1    00:00:00 ps`,
            `    930 pts/1    00:02:44 legacy-daemon`,
          ].join('\n');
          await addTypedOutput(out, runId);
        } else if (cmd === 'whoami' || cmd.startsWith('whoami ')) {
          await addTypedOutput(`${username}`, runId);
        } else if (cmd === 'df' || cmd.startsWith('df ')) {
          // DF (generic, sci-fi flavored devices)
          const out = [
            `Filesystem           1K-blocks       Used Available Use% Mounted on`,
            `/dev/blk0p1          524288000  131072000 393216000 25%  /`,
            `devtmpfs               8388608          0   8388608  0%  /dev`,
            `tmpfs                 16777216        128  16777088  1%  /dev/shm`,
            `/dev/blk0p2          524288000  104857600 419430400 20%  /home`,
            `tmpfs                  4194304       2048   4192256  1%  /run`,
            `/dev/blk0p0             262144      65536    196608 25%  /boot`,
            `tmpfs                 16777216      69632  16707584  1%  /tmp`,
          ].join('\n');
          await addTypedOutput(out, runId);
        } else if (cmd === 'top' || cmd.startsWith('top ')) {
          // TOP (generic snapshot)
          const out = [
            `top - 12:03:11 up 3 days,  4 users,  load average: 0.44, 0.48, 0.55`,
            `Tasks: 217 total,   2 running, 213 sleeping,   0 stopped,   2 zombie`,
            `%Cpu(s):  3.1 us,  1.0 sy,  0.0 ni, 95.5 id,  0.1 wa,  0.1 hi,  0.2 si,  0.0 st`,
            `MiB Mem :   8192.0 total,   3072.4 free,   2048.5 used,   3071.1 buff/cache`,
            `MiB Swap:   2048.0 total,   2048.0 free,      0.0 used.   5631.7 avail Mem`,
            `    PID USER       PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND`,
            `   0411 root       20   0  512.0m  38.2m   9.2m S   5.6   0.5   5:09.12 legacy-daemon`,
            `   0930 ops        20   0  128.0m  12.3m   6.1m S   2.3   0.1   2:44.18 comms-daemon`,
            `   1577 nav        20   0  896.0m 104.7m  21.0m R   1.7   1.3   0:49.03 nav-solution`,
            `   2210 sensor     20   0  256.0m  18.9m   7.8m S   0.9   0.2   0:12.70 sensor-arrayd`,
            `   1042 ${username.padEnd(9)} 20   0   16.0m   1.3m   0.8m S   0.3   0.0   0:01.27 sh`,
          ].join('\n');
          await addTypedOutput(out, runId);
        } else if (cmd.startsWith('curl') || cmd.startsWith('wget') || cmd === "curl" || cmd === "wget") {
          // CURL / WGET (shared verbose-ish HTTP output)
          let arg = cmd.slice(5).trim();
          if (arg === "") arg = "192.168.0.1";
          const out = [
            `*   Trying ${arg}:443...`,
            `* Connected to ${arg} port 443 (#0)`,
            `> GET /status 1.1`,
            `> User-Agent: 7.88.1`,
            `> Accept: */*`,
            `< 200 OK`,
            `< Server: 1.21.6 (Orion)`,
            `< Date: Sun, 10 Aug 2325 03:40:12 GMT`,
            `< Content-Type: application/json`,
            `< Content-Length: 128`,
            `< Connection: keep-alive`,
            `{"node":"MU/TH/UR","uptime":"21d 04h","integrity":"97%","support":"nominal","system":"online"}`,
            `* Connection #0 to host ${arg} left intact`,
          ].join('\n');
          await addTypedOutput(out, runId);
        } else {
          await addTypedOutput(t("TERMINAL.CLI.CommandNotFound", { command: cmd }), runId);
        }
      })
    }

    function generateBackground({ base, splashFile, password, background, opacity, borderImage, borderSlice, titlebarName }) {

      // border
      if (borderImage) {
        lain.style.border = "30px solid"
        lain.style.borderImage = `url(${borderImage}) ${borderSlice || 35} / 1 / 0 round`
      }

      //header
      lain.querySelector(`button[data-action="close"]`).style.display = "none"
      const header = lain.querySelector(`.window-header`)
      header.style.backgroundColor = base
      const headerTitle = lain.querySelector(`.window-title`)
      headerTitle.textContent = titlebarName || t("TERMINAL.CLI.PersonalTerminal")
      headerTitle.textContent += `${arg.shadowView ? ` | ${t("TERMINAL.CLI.Shadow")}` : ""}${arg.isSSH ? " | SSH" : ""}`
      header.insertAdjacentHTML('beforeend', `
        <button class="icon fa-regular fa-dash header-control" data-action="minimize" style="font-size: 1.5em; display: block" />
        <button class="icon fa-solid fa-xmark header-control" data-action="closeWindow" style="font-size: 1.5em; display: block" />
      `)

      // remove # character
      const hex = base.replace(/^#/, '')
      const r = parseInt(hex.slice(0, 2), 16)
      const g = parseInt(hex.slice(2, 4), 16)
      const b = parseInt(hex.slice(4, 6), 16)

      // find luminance
      const relativeLuminance = 0.2126 * Math.pow(r / 255, 2.2) +
        0.7152 * Math.pow(g / 255, 2.2) +
        0.0722 * Math.pow(b / 255, 2.2)
      const bright = relativeLuminance > 0.5
      if (bright) {
        header.style.color = 'black'
        headerTitle.style.color = "black"
      } else {
        header.style.color = 'white'
        headerTitle.style.color = "white"
      }

      // I use a black background until splash loads,, this is a great workaround
      // but this does need to get replaced in the case of a splash image (not video)
      if (splashFile) {
        const ext = splashFile.split('.').pop()
        if (!Object.keys(CONST.VIDEO_FILE_EXTENSIONS).includes(ext)) {
          lain.querySelector(`.terminal-splash`).style.backgroundImage = `url('${splashFile}')`
          lain.querySelector(`.terminal-splash`).style.backgroundSize = "cover"
          lain.querySelector(`.terminal-splash`).style.backgroundPosition = "center"
        }
      }

      // have a placeholder background
      if (!background || !splashFile) {
        const cv = lain.querySelector(".terminal-noise")
        cv.width = 800
        cv.height = 800
        const maxRadius = Math.max((cv.width / 2), (cv.height / 2))
        const ctx = cv.getContext("2d")
        const imageData = ctx.createImageData(cv.width, cv.height)

        for (let y = 0; y < cv.height; y++) {
          for (let x = 0; x < cv.width; x++) {
            const distanceToCenter = Math.sqrt((x - (cv.width / 2)) ** 2 + (y - (cv.height / 2)) ** 2)
            const opacityValue = 1 - (distanceToCenter / maxRadius) + .2 // Opacity decreases with distance from center

            const pixelIndex = (y * cv.width + x) * 4
            imageData.data[pixelIndex] = Math.floor(Math.random() * r) // Red channel
            imageData.data[pixelIndex + 1] = Math.floor(Math.random() * g) // Green channel
            imageData.data[pixelIndex + 2] = Math.floor(Math.random() * b) // Blue channel
            imageData.data[pixelIndex + 3] = Math.floor(opacityValue * 120) // Alpha channel (opacity)
          }
        }
        ctx.putImageData(imageData, 0, 0)
        if (!splashFile) {
          cv.style.zIndex = 2
          lain.querySelector(`.terminal-splash`).style.zIndex = 2
        }
      }
    }
  }

  async _prepareContext() {
    const tile = await fromUuid(this.uuid)
    if (!tile) {
      return { error: t("TERMINAL.Error.TerminalMissing") }
    }

    const flags = tile.flags.terminal
    const style = game.settings.get(ID, "styles")[flags?.style]
    if (!style) {
      return {
        error: t("TERMINAL.Error.StyleMissing")
      }
    }

    const journal = game.journal.get(flags?.journal)
    if (!journal) {
      return {
        error: t("TERMINAL.Error.JournalMissing")
      }
    }

    if (typeof ForgeVTT !== 'undefined') {
      let count = 0
      for (const asset of ["background", "borderImage", "click", "close", "startup", "splashFile"]) {
        if (!style[asset]) continue
        if (!style[asset].includes("http") && style[asset].includes("modules/terminal")) {

          // slow, bad request
          // /modules/terminal

          // assets, better but his is not a global asset
          // https://assets.forge-vtt.com/65386216bc2d33002bda378a/modules/terminal

          // global, ideal really
          // https://assets.forge-vtt.com/bazaar/modules/terminal-b65eff5013953bbe/assets
          count++
          // console.log("found relative path, convert to assets URL, forge exclusive", style[asset])
          style[asset] = ForgeVTT.ASSETS_LIBRARY_URL_PREFIX + "bazaar/modules/terminal-b65eff5013953bbe/assets" + style[asset].split("modules/terminal")[1]
        }
      }
      if (count) console.log(`Terminal | replaced ${count} relative path assets with assets.forge-vtt.com for faster loads`)
    }

    if (typeof ForgeVTT !== 'undefined' || game.settings.get(ID, "warmCache")) {
      warmCache(0)
    }

    const backgroundVideo = Object.keys(CONST.VIDEO_FILE_EXTENSIONS).includes(style.background?.split('.').pop())

    return {
      tuid: this.uuid,
      backgroundOpacity: 1 - Number(style?.opacity),
      height: Math.min(window.innerHeight * .8, 1200),
      backgroundVideo,
      ...style,
      ...flags,
      ...this.arg,
    }
  }

  close() {
    for (const sound of Object.values(audio)) {
      sound.fade(0).then(() => sound.stop())
    }
    return super.close()
  }
}

export function configureTerminalLocalization() {
  Terminal.PARTS.form.template = templatePath("terminal.hbs")
}
