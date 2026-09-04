
/* @license 2026 CodaBool all rights reserved */

import { Skilled, ID, Style, Check, Feedback, Rename, Regions, QuickStart, Shadow, OpenForPlayers, configureUILocalization } from "./ui.js"
import Terminal, { configureTerminalLocalization } from "./terminal.js"
import {
  moveToken,
  updateJournal,
  alienCharge,
  toggleDoorLock,
  exploreMap,
  updatePage,
  setFlagAndTab,
  toggleLights,
  startObservation,
  detectMotion,
  showObservers,
  validateTile,
  migrateToUuid,
  openDialog,
  runMacro,
  warmCache,
  injectListener,
  updateStyleDefaults,
  triggerRegion,
  autoShadowTerminal,
  learnAPIDialog,
} from "./util.js"
import { configurePresetLocalization, defaultStyles, successLines, stylePresets } from "./presets.js"
import { actionLabel, docTypeLabel, t, templatePath } from "./i18n.js"
export const audio = {}
/**
 * Renders the optional description fragment of a skill-check request on the GM's own client.
 * The requesting player sends `descriptionKey`/`descriptionData` for module-owned text and a bare
 * `description` only for world data (a custom button name, a journal page title), so the GM never
 * sees a sentence already rendered in the player's language. Each catalog entry supplies its own
 * The wrapper key supplies the separator, so spacing and brackets suit the language.
 */
function skillDescription(data) {
  const description = data.descriptionKey ? t(data.descriptionKey, data.descriptionData) : data.description
  return description ? t("TERMINAL.Dialog.SkillDescriptionWrap", { description }) : ""
}

let controller

Hooks.once("i18nInit", () => {
  configureUILocalization()
  configureTerminalLocalization()
  configurePresetLocalization()
})

Hooks.once("ready", () => {
  migrateToUuid()
  updateStyleDefaults()

  if (!window.warmCacheInterval && game.settings.get(ID, "warmCache"))
    warmCache(360_000, true)

  // allows for a single Terminal class for both hbs files
  loadTemplates([templatePath("terminal_cli.hbs")])

  setTimeout(() => {
    const u = game.users?.filter(u => u.active && u.isGM)
    if (!u?.length) {
      ui.notifications.error(
        t("TERMINAL.Error.NoGMOnline"),
      )
    }

    // GMs only past this point
    if (!game.user.isGM) return

    // movement is bugged in 13.344
    if (game.version === "13.344") {
      ui.notifications.error(t("TERMINAL.Error.IncompatibleVersion"))
    }

    // reset all timer locks in case a reload happened while a timer was going
    // TODO: this could be intensive, maybe just do this on the current scene
    for (const s of game.scenes) {
      for (const t of s.tiles) {
        if (t.getFlag(ID, "observeTimerRunning")) {
          t.setFlag(ID, "observeTimerRunning", false)
        }
        if (t.getFlag(ID, "detectTimerRunning")) {
          t.setFlag(ID, "detectTimerRunning", false)
        }
      }
    }

    // check for old macros
    const usingDeprecated = game.macros.some(m =>
      m.name.includes("Open Terminal for") &&
      (m.name.includes("V2") || m.name.includes("V3") || m.name.includes("V4") || m.name.includes("V5"))
    )
    if (usingDeprecated) {
      ui.notifications.info(
        t("TERMINAL.Info.NewMacros"),
      )
    }
  }, 15_000)
})

Hooks.once("init", async () => {
  // allows opening through macro
  window.Terminal = Terminal
  window.Skilled = Skilled
  window.validateTerminalTile = validateTile

  // inject into directory events for Terminal tile config ID helper buttons
  injectListener("journal", "journal", "JournalDirectory")
  injectListener("observeActor", "actors", "ActorDirectory")

  game.settings.register(ID, "pointer", {
    scope: "world",
    restricted: true,
    type: String,
    default: "",
  })
  game.settings.register(ID, "styles", {
    scope: "world",
    restricted: true,
    type: Object,
    default: defaultStyles,
  })
  // deprecated but still need it to exist for migrating off ID, to UUID
  game.settings.register(ID, "tiles", {
    scope: "world",
    restricted: true,
    type: Object,
    default: {},
  })
  game.settings.register(ID, "shadow", {
    scope: "world",
    restricted: true,
    type: Object,
    default: {},
  })
  game.settings.register(ID, "screensaver", {
    scope: "world",
    name: "TERMINAL.Setting.Screensaver.Name",
    hint: "TERMINAL.Setting.Screensaver.Hint",
    type: Boolean,
    default: true,
    config: true,
    restricted: true,
  })
  game.settings.register(ID, "notice", {
    scope: "world",
    name: "TERMINAL.Setting.Notice.Name",
    hint: "TERMINAL.Setting.Notice.Hint",
    type: Boolean,
    default: false,
    config: true,
    restricted: true,
  })
  game.settings.register(ID, "warmCache", {
    scope: "world",
    name: "TERMINAL.Setting.Cache.Name",
    hint: "TERMINAL.Setting.Cache.Hint",
    type: Boolean,
    default: false,
    config: true,
    restricted: true,
    onChange: value => {
      if (value) {
        warmCache(360_000)
      } else {
        clearInterval(window.warmCacheInterval)
      }
    }
  })
  game.settings.registerMenu(ID, "styleMenu", {
    name: "TERMINAL.Setting.Styles.Name",
    label: "TERMINAL.Setting.Styles.Label",
    hint: "TERMINAL.Setting.Styles.Hint",
    type: Style,
    icon: "fa-solid fa-paintbrush",
    restricted: true,
  })
  Handlebars.registerHelper("readonlyIf", readOnly => {
    return readOnly ? "readonly disabled" : ""
  })
  Handlebars.registerHelper("styleSelect", uuid => {
    const preset = Object.values(game.settings.get(ID, "styles")).filter(s => defaultStyles[s.uuid])
    return preset.some(s => s.uuid === uuid) ? "47, 224, 44,.1" : "4,45,107,.1"
  })
  game.socket.on("module.terminal", async data => {
    const gm = game.user.isGM && game.user.id === data.gid
    if (data.action === "alienCharge") {
      if (!gm) return
      alienCharge(data.actorIds)
    } else if (data.action === "resetVision") {
      canvas.perception.update({ initializeVision: true, refreshLighting: true, refreshSounds: true })
      canvas.tokens.objects.children.forEach(t => t.control({ releaseOthers: false }))
      canvas.tokens.releaseAll()
      canvas.perception.initialize()
      if (!game.user.isGM)
        ui.notifications.info(t("TERMINAL.Action.AccessRevoked"))
    } else if (data.action === "puzzle") {
      if (game.user.id !== data.uid) return
      const journal = await fromUuid(data.uuid)
      // COMPATABILITY: use Firefox supported .forEach instead of .values().find()
      // const app = foundry.applications.instances.values().find(a => a.context.journal === journal.parent.id)
      foundry.applications.instances.forEach(a => {
        if (a.context.journal === journal.parent.id) {
          a.createHTML()
        }
        a.maximize()
      })
    } else if (data.action === "lights") {
      if (!gm) return
      if (game.settings.get(ID, "notice")) {
        ui.notifications.info(t("TERMINAL.Action.LightsToggled", { name: data.name, scene: game.scenes.get(data.sid)?.name }))
      }
      toggleLights(data.sid)
    } else if (data.action === "updateJournal") {
      if (!gm) return
      await updateJournal(data.jid, data.uid)
      // allow for slow networks to process journal update, possibly buggy
      setTimeout(() => {
        game.socket.emit("module.terminal", { ...data, action: "createHTML" })
      }, 1_000)
    } else if (data.action === "createHTML") {
      if (game.user.id !== data.uid) return
      // COMPATABILITY: use Firefox supported .forEach instead of .values().find()
      foundry.applications.instances.forEach(a => {
        if (a.uuid === data.tuid) {
          a.createHTML()
        }
      })
    } else if (data.action === "startObservation") {
      if (!gm) return
      startObservation(data.observeActor, data.observeTimer, data.uid, data.tuid)
    } else if (data.action === "setFlag") {
      if (!gm) return
      const tile = await fromUuid(data.tuid)
      if (!tile) {
        ui.notifications.error(t("TERMINAL.Error.TileGone", { tile: data.tuid, flag: data.flag, value: data.value }))
        return
      }
      tile.setFlag(ID, data.flag, data.value)
    } else if (data.action === "userRunMacro") {
      if (game.user.id !== data.uid) return
      game.macros.get(data.macroId).execute({
        args: data.args,
      })
    } else if (data.action === "showObservers") {
      if (game.user.id !== data.uid) return
      showObservers(data.actorId, data.observeTimer)
    } else if (data.action === "shadowPing") {
      if (!data.otherActiveUserIds.includes(game.user.id)) return
      const el = Array.from(document.querySelectorAll(".terminal-window")).filter(e => {
        return !e.classList.contains("terminal-ssh") && !e.classList.contains("terminal-shadow")
      })
      const tuidCSS = el.length > 0 ? el[0].classList.value.split(' ').find(c => c.startsWith('Scene-')) : ""
      game.socket.emit('module.terminal', {
        action: "shadowPong",
        gid: data.gid,
        uid: game.user.id,
        inTerminal: el.length > 0,
        tuid: tuidCSS.replaceAll("-", ".")
      })
    } else if (data.action === "shadowBtn") {
      foundry.applications.instances.forEach(a => {
        if (a.uuid === data.tuid && a.arg.shadowView) {
          if (data.password) {
            a.element.querySelector(".terminal-password").value = data.password
            a.element.querySelector('button[data-action="password"]')?.click()
          } else {
            a.replaceHTML(null, data.injectObj)
          }
        }
      })
    } else if (data.action === "shadowPong") {
      if (!gm) return
      if (data.inTerminal) {
        const loading = document.querySelector(`div[uid="${data.uid}"]`);
        if (loading) {
          loading.style.display = "none";
        }
        const check = document.querySelector(`i[uid="true.${data.uid}"]`);
        if (check) {
          check.style.display = "block"
          check.setAttribute("tuid", data.tuid)
        }
        const x = document.querySelector(`i[uid="false.${data.uid}"]`);
        if (x) {
          x.style.display = "none";
        }
      } else {
        const loading = document.querySelector(`div[uid="${data.uid}"]`);
        if (loading) {
          loading.style.display = "none";
        }
        const x = document.querySelector(`i[uid="false.${data.uid}"]`);
        if (x) {
          x.style.display = "block";
        }
        const check = document.querySelector(`i[uid="true.${data.uid}"]`);
        if (check) {
          check.style.display = "none";
        }
      }
    } else if (data.action === "changeScene") {
      if (game.user.id !== data.uid) return
      await game.scenes.get(data.sid).view()
      if (data.showObservers) {
        showObservers(data.actorId, data.observeTimer)
      }
    } else if (data.action === "toggleDoorLock") {
      if (!gm) return
      toggleDoorLock(data.wid, data.name, data.uid)
    } else if (data.action === "macro") {
      if (!gm) return
      runMacro(data.tuid, data.uid, data.macroNoProxy ? data.triggeredBy : null)
    } else if (data.action === "close") {
      if (game.user.id !== data.uid) return
      if (document.querySelector(".terminal-skilled")) {
        ui.notifications.error(t("TERMINAL.Error.AccessDenied"))
        document.querySelector(`.terminal-skilled button[data-action="close"]`).click()
      }
    } else if (data.action === "panToPoint") {
      if (game.user.id !== data.uid) return
      if (data.min) {
        Object.values(ui.windows).forEach(app => app.minimize())
        foundry.applications.instances.forEach(a => a.minimize())
      }
      canvas.animatePan({
        x: data.x,
        y: data.y,
        duration: 1700,
      })
      if (!data.min) return
      setTimeout(() => {
        Object.values(ui.windows).forEach(app => app.maximize())
        foundry.applications.instances.forEach(a => a.maximize())
      }, 2100)
    } else if (data.action === "resetCheck") {
      if (game.user.id !== data.uid) return
      const app = foundry.applications.instances.values().find(a => a.uuid === data.tuid)
      const win = document.querySelector(`.${data.tuid.replace(/\./g, '-')}`)
      win.querySelectorAll('.terminal-rm').forEach(e => e.remove())
      if (data.failed) {
        app.injectHTML({
          type: "text",
          text: {
            content: `<h1>${t("TERMINAL.Dialog.PermissionDeniedAction", { title: actionLabel(data.title) })}</h1>`
          },
          name: actionLabel(data.title)
        })
        return
      }
      app.successHTML(data.index, data.ASCII, data.content)
    } else if (data.action === "notify") {
      const error = data.errorKey ? t(data.errorKey, data.errorData) : data.error
      const message = data.messageKey ? t(data.messageKey, data.messageData) : data.message
      if (data.uid) {
        if (game.user.id !== data.uid) return
        if (error) {
          ui.notifications.error(error, { permanent: data.perm || false })
        } else {
          ui.notifications.info(message, {
            permanent: data.perm || false,
          })
        }
        return
      }
      if (!game.user.isGM) return
      if (error) {
        ui.notifications.error(error, { permanent: data.perm || false })
      } else if (game.settings.get(ID, "notice") && data.notice) {
        ui.notifications.info(message, { permanent: data.perm || false })
      } else {
        ui.notifications.info(message, { permanent: data.perm || false })
      }
    } else if (data.action === "gmApprove") {
      if (!gm) return
      foundry.applications.api.DialogV2.wait({
        window: { title: t("TERMINAL.Window.SkillChecks") },
        content: `<p style="font-size:1.4em; max-width: 400px">${t("TERMINAL.Dialog.SkillAction", { name: data.name, title: actionLabel(data.title), description: skillDescription(data) })}</p>`, buttons: [{
          action: "1",
          label: t("TERMINAL.Common.Allow", { name: data.name }),
          icon: "fas fa-check",
          callback: async () => {
            if (data.title === "lock or unlock door") {
              toggleDoorLock(data.wid, data.name, data.uid)
            } else if (data.title === "download map") {
              game.socket.emit('module.terminal', { ...data, action: "exploreMap" })
            } else if (data.title === "toggle power") {
              toggleLights(data.sid)
            } else if (data.title === "detect motion") {
              detectMotion(data.uid, data.tuid, data.limit)
            } else if (data.title === "observe Actor") {
              startObservation(data.observeActor, data.observeTimer, data.uid, data.tuid)
            } else if (data.title === "start a secure shell") {
              if (game.settings.get("terminal", "notice") && data.bypass) {
                ui.notifications.info(
                  t("TERMINAL.Action.UserSSHBypass", { name: data.name }),
                )
              }
              game.socket.emit("module.terminal", {
                ...data,
                tuid: data.ssh,
                arg: { isSSH: true },
                action: "render",
              })
              autoShadowTerminal(data.ssh, { isSSH: true })
            } else if (data.title === "opening a skilled Terminal") {
              game.socket.emit("module.terminal", {
                ...data,
                action: "render",
                skilled: true,
              })
              return
            } else if (data.title === "decrypt a file") {
              await updatePage(data.jid, data.pid)
              game.socket.emit("module.terminal", {
                ...data,
                action: "decryptApprove",
              })
            } else if (data.title === "run script") {
              if (data.monk) {
                const tile = await fromUuid(data.tuid)
                tile.trigger({})
              } else {
                runMacro(data.tuid, data.uid, data.macroNoProxy ? data.triggeredBy : null)
              }
            } else if (data.title === "execute script") {
              triggerRegion(data.regionUUID, data.event, data.tuid)
            }
            game.socket.emit("module.terminal", {
              ...data,
              action: "resetCheck",
            })
          },
        },
        {
          action: "2",
          label: t("TERMINAL.Common.Deny", { name: data.name }),
          icon: "fa-solid fa-ban",
          callback: () => {
            if (data.title === "opening a skilled Terminal") {
              game.socket.emit("module.terminal", {
                action: "close",
                uid: data.uid,
              })
              return
            }
            game.socket.emit("module.terminal", {
              action: "resetCheck",
              uid: data.uid,
              tuid: data.tuid,
              title: data.title,
              failed: true,
            })
          },
        }],
      })
    } else if (data.action === "exploreMap") {
      if (game.user.id !== data.uid) return
      exploreMap(data.exploreAll, data.skipNotification)
    } else if (data.action === "teleport") {
      if (!gm) return
      const user = game.users.get(data.uid)
      const originScene = game.scenes.get(data.sid)
      const token = originScene?.tokens.find(t => t.id === data.token)
      const region = fromUuidSync(data.rid)
      if (!user || !originScene || !token || !region) {
        console.error("Terminal |", user, originScene, token, region)
        ui.notifications.error(t("TERMINAL.Error.TeleportFailed", { scene: originScene?.name }))
      }
      console.log("handle", {
        name: data.event,
        data: {
          token,
          movement: token.movement
        },
        region,
        user,
      })
      region._handleEvent({
        name: data.event,
        data: {
          token,
          movement: token.movement
        },
        region,
        user,
      })
    } else if (data.action === "render") {
      if ((game.user.id !== data.uid) && (game.user.uuid !== data.uid)) return
      if (data.skilled) {
        document.querySelector(`.terminal-skilled button[data-action="close"]`)?.click()
        ui.notifications.info(
          t("TERMINAL.Action.AccessGranted", {
            suffix: successLines.get(game.system.id)
              ? t("TERMINAL.Action.AccessGrantedSuffix", { flavor: t(successLines.get(game.system.id)) })
              : "",
          })
        )
      }
      if (!document.querySelector(`.${data.tuid.replace(/\./g, '-')}`)) {
        new Terminal(data.tuid, data.arg).render(true)
      }
    } else if (data.action === "decryptApprove") {
      if (game.user.id !== data.uid) return
      // COMPATABILITY: use Firefox supported .forEach instead of .values().find()
      foundry.applications.instances.forEach(a => {
        if (a.uuid === data.tuid) {
          a.replaceHTML(data.pid)
          game.socket.emit('module.terminal', {
            tuid: data.tuid,
            action: "shadowBtn",
            injectObj: game.journal.get(a.context.journal).pages.get(data.pid),
          })
        }
      })
    } else if (data.action === "publish") {
      if (!gm) return
      const j = game.journal.get(data.jid)
      if (game.settings.get(ID, "notice")) {
        ui.notifications.info(
          t("TERMINAL.Action.EveryoneObserver", { journal: j.name }),
        )
      }
      j.update({ ownership: { default: 2 } }) // observer
      ChatMessage.create({
        content: `@UUID[JournalEntry.${data.jid}]{${j.name}} ${t("TERMINAL.Chat.HasBeenDiscovered")}`,
      })
    }
  })

  Hooks.on("moveToken", (token, update) => {
    if (game.release.generation === 13) return
    moveToken(token, update)
  })

  // Hooks.on("updateToken", (token, update) => {
  // })
  Hooks.on("preMoveToken", (token, update) => {
    if (game.release.generation >= 14) return
    moveToken(token, update)
  })

  Hooks.on("renderSceneControls", async (app, html, data) => {
    if (!game.user.isGM) return
    if (html.querySelector('.terminal-shadow-control')) return

    const li = document.createElement('li')
    const button = document.createElement('button')
    button.className = 'control ui-control layer icon fa-solid fa-computer terminal-shadow-control'
    button.dataset.tooltip = t("TERMINAL.Tooltip.ShadowTerminal")
    li.appendChild(button)
    html.querySelector('menu[id="scene-controls-layers"]').appendChild(li)

    button.addEventListener("click", () => {
      if (document.querySelector(".terminal-shadow-window")) {
        ui.notifications.error(t("TERMINAL.Error.WindowOpen"))
        return
      }
      new Shadow().render(true)
    })
  })

  // Hooks.on("renderSettingsConfig", (_, html) => { })
  // Hooks.on("preUpdateTile", tile => {})

  // cache warming
  Hooks.on("canvasReady", canvas => {
    foundry.applications.instances.forEach(a => {
      if (a.uuid && !a.context?.keepOpenOnSceneChange) {
        a.close()
      }
    })
    if (typeof ForgeVTT !== 'undefined' || game.settings.get(ID, "warmCache")) {
      warmCache(0)
    }
  })

  // validation on tile settings
  Hooks.on("updateTile", tile => {
    validateTile(tile)
  })

  // safety in case someone closes while a pointer exists
  Hooks.on('closeTileConfig', () => {
    controller?.abort()
    if (game.settings.get(ID, "pointer")) {
      game.settings.set("terminal", "pointer", "")
      ui.notifications.info(t("TERMINAL.Action.ClickCanceled"))
    }
  })

  Hooks.on("controlWall", async (wall, control) => {
    console.log("wall", wall, control)
    if (!control || !wall.isDoor || !game.settings.get(ID, "pointer")) return

    const tileUuid = game.settings.get(ID, "pointer")
    game.settings.set("terminal", "pointer", "")
    const tileDoc = await fromUuid(tileUuid)
    Object.values(ui.windows).forEach(async app => await app.maximize())
    foundry.applications.instances.forEach(async a => await a.maximize())
    const unlockIds = tileDoc.flags.terminal.unlockIds
    let newVal = ""
    if (unlockIds?.length) {
      const ids = unlockIds.replace(/\s/g, "").split(",")
      if (ids.includes(wall.document.uuid)) {
        ui.notifications.error(t("TERMINAL.Error.DoorAdded"))
      } else {
        newVal = `${unlockIds},${wall.document.uuid}`
      }
    } else {
      newVal = wall.document.uuid
    }
    if (newVal) {
      setFlagAndTab(tileDoc, "unlockIds", newVal)
    }
    canvas.tiles.activate()
  })

  Hooks.on("controlTile", async (tile, control) => {
    if (!control || !game.settings.get(ID, "pointer")) return
    const tileUuid = game.settings.get(ID, "pointer")
    if (tileUuid === tile.document.uuid) {
      ui.notifications.error(t("TERMINAL.Error.SelfReference"))
      return
    }
    game.settings.set(ID, "pointer", "")
    Object.values(ui.windows).forEach(async app => await app.maximize())
    foundry.applications.instances.forEach(async a => await a.maximize())
    const tileDoc = await fromUuid(tileUuid)

    const ssh = tileDoc.flags.terminal.ssh
    let newVal = ""
    if (ssh?.length) {
      const ids = ssh.split(",")
      if (ids.includes(tile.document.uuid)) {
        ui.notifications.error(t("TERMINAL.Error.TileAdded"))
      } else {
        newVal = `${ssh},${tile.document.uuid}`
      }
    } else {
      newVal = tile.document.uuid
    }
    if (newVal) {
      setFlagAndTab(tileDoc, "ssh", newVal)
    }
  })

  Hooks.on("renderTileConfig", async (app, html) => {
    const doc = app.document
    const styleExists = Object.keys(game.settings.get(ID, "styles")).includes(
      doc.getFlag(ID, "style"),
    )
    const hasPlaceablesTab = game.release.build > 353
    const itemPrettyName = game.items.get(doc.getFlag(ID, "keycard")?.split("@")[0])?.name
    let macroPrettyName = game.macros.get(doc.getFlag(ID, "macro"))?.name
    let actorPrettyName = game.actors.get(doc.getFlag(ID, "observeActor"))?.name
    let observeScenePrettyName = game.scenes.get(doc.getFlag(ID, "observeScene"))?.name
    let journalPrettyName = game.journal.get(doc.getFlag(ID, "journal"))?.name

    if (doc.getFlag(ID, "journal") && !journalPrettyName)
      journalPrettyName = t("TERMINAL.Error.DeletedJournal")
    if (doc.getFlag(ID, "observeActor") && !actorPrettyName)
      actorPrettyName = t("TERMINAL.Error.DeletedActor")
    if (doc.getFlag(ID, "observeScene") && !observeScenePrettyName)
      observeScenePrettyName = t("TERMINAL.Error.DeletedScene")
    if (doc.getFlag(ID, "macro") && !macroPrettyName)
      macroPrettyName = t("TERMINAL.Error.DeletedMacro")

    if (!styleExists && doc._id) {
      const preset = stylePresets.get(game.system.id)
      doc.setFlag(ID, "style", preset || "generic-green")
    }
    const stylesUnsorted = game.settings.get(ID, "styles")
    const styles = Object.values(stylesUnsorted).sort((a, b) => {
      return (defaultStyles[a.uuid] ? 1 : 0) - (defaultStyles[b.uuid] ? 1 : 0)
    })
    const terminalActive = !html.querySelector(".active") || html.querySelector('div[data-tab="terminal"]')?.classList?.contains("active")
    let content = await foundry.applications.handlebars.renderTemplate(templatePath("config.hbs"), {
      styles,
      ...doc.flags.terminal,
      system: game.system.id,
      hasMonk: game.modules.get("monks-active-tiles")?.active,
      hasPuzzleLocks: game.modules.get("puzzle-locks")?.active,
      release: game.data.release.generation,
      journalPrettyName,
      actorPrettyName,
      macroPrettyName,
      observeScenePrettyName,
      itemPrettyName,
      terminalActive,
      styleExists,
      TileInit: !doc._id,
      doorLength: doc.getFlag(ID, "unlockIds")?.length && doc.getFlag(ID, "unlockIds")?.split(",")?.length || 0,
      sshLength: doc.getFlag(ID, "ssh")?.length && doc.getFlag(ID, "ssh")?.split(",")?.length || 0,
      ssh: doc.getFlag(ID, "ssh")?.length ? doc.getFlag(ID, "ssh")?.split(",") : [],
      doorIds: doc.getFlag(ID, "unlockIds")?.length ? doc.getFlag(ID, "unlockIds")?.split(",") : [],
    })

    // add nav tab
    const a = document.createElement("a")
    const tabs = document.getElementById(app.id).querySelector(".tabs a:nth-child(3)")
    tabs.insertAdjacentElement("afterend", a)
    a.outerHTML = `<a class="${terminalActive ? 'active' : ''}" data-action="tab" data-group="sheet" data-tab="terminal"><i class="fa-solid fa-computer"></i><label> ${t("TERMINAL.Tab.Title")}</label></a>`

    // add content
    const tabContent = html.querySelector('div[data-tab="terminal"]')
    if (tabContent) {

      // full replace existing content
      tabContent.outerHTML = content
    } else {

      // add in a new div with content
      const div = document.createElement("div")
      html.querySelector(".window-content").insertBefore(div, html.querySelector("footer"))
      div.outerHTML = content
    }

    // return early if listeners have already been added
    if (!doc._id) return

    // add listeners
    html.querySelector("#terminal-style")?.addEventListener("click", () => new Style().render(true))
    html.querySelector("#terminal-autostyle")?.addEventListener("click", () => new QuickStart(doc, app.id).render(true))
    html.querySelector("#terminal-check-btn")?.addEventListener("click", () => new Check(doc).render(true))
    html.querySelector("#terminal-rename-btn")?.addEventListener("click", () => new Rename(doc).render(true))
    html.querySelector("#terminal-feedback-btn")?.addEventListener("click", () => new Feedback().render(true))
    html.querySelector("#terminal-regions-btn")?.addEventListener("click", () => new Regions(doc).render(true))
    html.querySelector("#terminal-open-gui-btn")?.addEventListener("click", () => new OpenForPlayers(doc.uuid).render(true))
    html.querySelector("#terminal-learn-api-btn")?.addEventListener("click", learnAPIDialog)
    // sub tab switching
    html.querySelectorAll(".terminal-submenu")?.forEach(el => el.addEventListener("click", e => {
      html.querySelectorAll(".terminal-submenu").forEach(el => el.classList.remove("active"))
      e.target.classList.add("active")
      html.querySelectorAll(".terminal-subtab").forEach(el => el.classList.remove("active"))
      html.querySelector(`div[data-tab="${e.target.getAttribute("data-tab")}"]`).classList.add("active")
    }))
    html.querySelectorAll(".terminal-rm-door-btn")?.forEach(el => el.addEventListener("click", e => {
      const ids = doc.getFlag(ID, "unlockIds")?.split(",")?.filter(d =>
        d !== e.target.attributes.uuid.value
      )
      setFlagAndTab(doc, "unlockIds", ids.join(","))
    }))

    html.querySelectorAll(".terminal-rm-tile-btn").forEach(el => el.addEventListener("click", e => {
      const uuids = doc.getFlag(ID, "ssh")?.split(",")?.filter(d =>
        d !== e.target.attributes.uuid.value
      )
      setFlagAndTab(doc, "ssh", uuids.join(","))
    }))

    // WORK AROUND: revert to controller for items
    html.querySelector('button[link="items"]')?.addEventListener("click", async () => {
      controller = new AbortController()
      const { signal } = controller
      window.addEventListener(
        "click",
        async e => {
          let id = null
          if (e.target.nodeName === "A" || e.target.nodeName === "IMG") {
            if (e.target.parentNode?.classList.contains("directory-item")) {
              id = e.target.parentNode?.getAttribute("data-entry-id")
            }
          }
          if (id) {
            e.stopPropagation() // prevent the event from reaching bubbling phase and opening item window
            const sourceId = game.items.get(id)?.flags?.core?.sourceId
            Object.values(ui.windows).forEach(app => app.maximize())
            foundry.applications.instances.forEach(a => a.maximize())
            setFlagAndTab(doc, "keycard", `${id}@${sourceId || ""}`)
            game.settings.set("terminal", "pointer", "")
            controller.abort()
          }
        },
        { signal, capture: true },
      )
    })

    // WORK AROUND: just to support hotbar macros
    html.querySelector('button[link="macros"]')?.addEventListener("click", async () => {
      controller = new AbortController()
      const { signal } = controller
      window.addEventListener("click", async e => {
        let id = null

        if (e.target.nodeName === "LI") {
          if (e.target.getAttribute("data-slot")) {
            id = game.user.hotbar[Number(e.target.getAttribute("data-slot"))]
          }
        }
        if (e.target.nodeName === "A" || e.target.nodeName === "IMG") {
          if (e.target.parentNode?.classList.contains("directory-item")) {
            id = e.target.parentNode?.getAttribute("data-entry-id")
          }
        }
        if (id) {
          e.stopPropagation() // prevent the event from reaching bubbling phase and opening item window
          Object.values(ui.windows).forEach(app => app.maximize())
          foundry.applications.instances.forEach(a => a.maximize())
          setFlagAndTab(doc, "macro", id)
          game.settings.set("terminal", "pointer", "")
          controller.abort()
        }
      }, { signal, capture: true })
    })

    // WORK AROUND: revert to controller for scenes
    html.querySelector('button[link="scenes"]')?.addEventListener("click", async () => {
      controller = new AbortController()
      const { signal } = controller
      window.addEventListener(
        "click",
        async e => {
          let id = null
          if (e.target.nodeName === "A") {
            if (e.target.parentNode?.classList.contains("directory-item")) {
              id = e.target.parentNode?.getAttribute("data-entry-id")
            }
          }
          if (id) {
            e.stopPropagation() // prevent the event from reaching bubbling phase and opening item window
            Object.values(ui.windows).forEach(app => app.maximize())
            foundry.applications.instances.forEach(a => a.maximize())
            setFlagAndTab(doc, "observeScene", id)
            game.settings.set("terminal", "pointer", "")
            controller.abort()
          }
        },
        { signal, capture: true },
      )
    })

    // link buttons
    html.querySelectorAll(".terminal-link").forEach(el => el.addEventListener("click", e => {


      const type = e.target.attributes.link.value
      canvas.tiles.releaseAll()
      const name = docTypeLabel(e.target.attributes.pretty?.value || type.toUpperCase())
      ui.notifications.info(t("TERMINAL.Action.ClickToConnect", {
        name,
        cancel: type === "items" ? "" : t("TERMINAL.Action.CloseConfigToCancel"),
      }))
      Object.values(ui.windows).forEach(app => app.minimize())
      foundry.applications.instances.forEach(a => a.minimize())
      if (type === "journal" || type === "actors" || type === "items" || type === "macros" || type === "scenes") {
        ui.sidebar.expand()
        ui.sidebar.changeTab(type, "primary")
      } else if ((type === "tiles" || type === "walls") && hasPlaceablesTab) {
        ui.sidebar.expand()
        ui.sidebar.changeTab("placeables", "primary")
        document.querySelector(`button[data-tab="${type}"]`)?.click()
      } else { // tiles, walls and no placeables tab
        canvas[type].activate()
      }
      game.settings.set(ID, "pointer", doc.uuid)
    }))

    // delete buttons
    html.querySelectorAll(".terminal-delete").forEach(el => el.addEventListener("click", e => {
      setFlagAndTab(doc, e.target.id.split(".")[2], "")
    }))

    // view buttons
    html.querySelectorAll(".terminal-view").forEach(el => el.addEventListener("click", async e => {
      const doc = await fromUuid(e.target.attributes.uuid.value)
      if (!doc) return
      if (!doc.parent.isView) {
        ui.notifications.info(t("TERMINAL.Error.CannotViewScene", { collection: doc.parentCollection, scene: doc.parent.name }))
        return
      }
      canvas.ping(doc.object.center)
      canvas[doc.parentCollection].activate()
      canvas.animatePan({ ...doc.object.center, duration: 500 })
    }))

    // make data save instant for checkboxes
    html.querySelectorAll("div[data-tab='terminal'] input[type='checkbox']").forEach(el =>
      el.addEventListener("click", e => setFlagAndTab(doc, e.target.name.split(".")[2], e.target.checked))
    )

    // use a dialog, to never leave unsaved changes
    html.querySelectorAll(".terminal-use-dialog").forEach(el => el.addEventListener("click", e => {
      openDialog(
        doc,
        e.target.name.split(".")[2],
        e.target.attributes.readable.value,
        e.target.type,
      )
    }))

    // make data save instant for selects
    html.querySelector("div[data-tab='terminal'] select")?.addEventListener("input", e => {
      setFlagAndTab(doc, "style", e.target.value)
    })

    // fix issue of there being a scrollbar
    if (html.length) {
      html.querySelector('div[data-tab="terminal"]').style.overflow = "auto"
      html[0].style.height = "auto"
    }
  })
})
