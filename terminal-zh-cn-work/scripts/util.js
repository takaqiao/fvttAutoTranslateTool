/* @license 2026 CodaBool all rights reserved */

import { ID, Skilled } from "./ui.js"
import Terminal from "./terminal.js"
import { defaultStyles } from "./presets.js"
import { t as tr } from "./i18n.js"

export function validateTile(tile, macroMode) {
  const migrating = game.settings.settings.get("terminal.migrating") ? game.settings.get(ID, "migrating") : false
  if (migrating) return
  if (!game.user.isGM && !macroMode) return
  if (!tile.getFlag(ID, "enabled") && !macroMode) return
  // just skip validation if changing values with the skill check window open
  if (document.querySelector(".terminal-check") && !macroMode) return

  // style
  if (!tile.getFlag(ID, "style")) {
    ui.notifications.error(tr("TERMINAL.Error.StyleUnassigned"))
    if (macroMode) return
  }
  const styleExists = Object.keys(game.settings.get(ID, "styles")).includes(
    tile.getFlag(ID, "style"),
  )
  if (tile.getFlag(ID, "style") && !styleExists) {
    ui.notifications.error(
      tr("TERMINAL.Error.StyleDeleted"),
    )
    if (macroMode) return
  }

  // unlock
  if (tile.getFlag(ID, "unlockIds")) {
    const wallIds = tile.getFlag(ID, "unlockIds")?.replace(/\s/g, "")?.split(",")
    if (wallIds) {
      for (const id of wallIds) {
        if (id === "") continue
        const wall = fromUuidSync(id)
        if (!wall) {
          ui.notifications.error(tr("TERMINAL.Error.InvalidWall", { id }))
          if (macroMode) return
          break
        }
      }
    }
  }

  // run macro
  const runMacroId = tile.getFlag(ID, "macro")
  const macro = game.macros.get(runMacroId)
  if (!macro && runMacroId !== "" && runMacroId !== undefined) {
    ui.notifications.error(
      tr("TERMINAL.Error.MacroGone", { id: runMacroId }),
    )
    if (macroMode) return
  }

  // journal
  const journalId = tile.getFlag(ID, "journal")
  if (!game.journal.get(journalId)) {
    ui.notifications.error(
      tr("TERMINAL.Error.JournalRequired"),
    )
    if (macroMode) return
  }

  // observer token exists
  const observeActor = tile.getFlag(ID, "observeActor")
  const observeScene = tile.getFlag(ID, "observeScene")

  let found = false
  if (observeActor) {
    if (!game.actors.get(observeActor)) {
      found = null
      ui.notifications.error(
        tr("TERMINAL.Error.ActorGone", { id: observeActor }),
      )
      if (macroMode) return
    }
    // TODO: this may cause issues in cross scene use
    for (const t of tile.parent.tokens) {
      if (t.actorId === observeActor) {
        found = true
        if (!t.sight.enabled && t.object) {
          ui.notifications.error(
            tr("TERMINAL.Error.SightDisabled", { name: t.name }),
          )
          canvas.ping(t.object.center)
          // if (macroMode) return
        } else if (t.sight.range < 1 && t.object) {
          ui.notifications.error(
            tr("TERMINAL.Error.ZeroVision", { name: t.name }),
          )
          canvas.ping(t.object.center)
          // if (macroMode) return
        }
      }
    }
  }

  // no observer token
  // TODO: this may cause issues in cross scene use
  if (!found && observeActor && found !== null && !observeScene) {
    ui.notifications.error(
      tr("TERMINAL.Error.NoActorToken", { actor: game.actors.get(observeActor)?.name, scene: tile.parent.name }),
    )
    // if (macroMode) return
  }

  // vision
  // TODO: this may cause issues in cross scene use
  if (!tile.parent.tokenVision && observeActor) {
    ui.notifications.error(
      tr("TERMINAL.Error.TokenVisionObserve", { scene: tile.parent.name }),
    )
  }

  // item
  const itemURI = tile.getFlag(ID, "keycard")
  if (itemURI) {
    if (macroMode) {
      ui.notifications.info(tr("TERMINAL.Info.MacroItemSkipped"))
    } else {
      const [itemId, sourceId] = itemURI?.split("@")
      if (!game.items.get(itemId)) {
        ui.notifications.error(
          tr("TERMINAL.Error.ItemGone", { id: itemId }),
        )
      }
    }
  }

  // map
  // TODO: this may cause issues in cross scene use
  if (!tile.parent.tokenVision && tile.getFlag(ID, "explore")) {
    ui.notifications.error(
      tr("TERMINAL.Error.TokenVisionMap", { scene: tile.parent.name }),
    )
  } else if (!tile.parent.fog.exploration && tile.getFlag(ID, "explore")) {
    ui.notifications.error(
      tr("TERMINAL.Error.FogRequired", { scene: tile.parent.name }),
    )
  }

  // ssh terminal on scene
  const ssh = tile.getFlag("terminal", "ssh")
  for (const uuid of ssh?.split(",") || []) {
    const sshTile = fromUuidSync(uuid)
    if (ssh && !sshTile) {
      ui.notifications.error(tr("TERMINAL.Error.SSHTileGone", { uuid }))
      if (macroMode) return
    }
  }

  // observe time is number
  const observeTimer = tile.getFlag(ID, "observeTimer")
  if (!/^[0-9]+$/.test(observeTimer) && observeTimer) {
    ui.notifications.error(
      tr("TERMINAL.Error.TimerNumber", { value: observeTimer }),
    )
    if (macroMode) return
  }

  // observe time is under 10
  if (Number(observeTimer) < 10 && observeTimer) {
    ui.notifications.error(
      tr("TERMINAL.Error.TimerMinimum", { value: observeTimer }),
    )
    if (macroMode) return
  }
  if (macroMode) return true
}

export function learnAPIDialog() {
  foundry.applications.api.DialogV2.wait({
    window: { title: tr("TERMINAL.Window.Programmatic"), width: 400 },
    content: `
      <div style="font-size:1.1em; line-height:1.4;" class="terminal-dialog-add-overflow">
        <blockquote style="padding:0">
          <p style="text-align: center;margin:0">${tr("TERMINAL.API.OpenWays")}</p>
        </blockquote>

        <section>
          <h2>${tr("TERMINAL.API.Macros")}</h2>
          <p>
            ${tr("TERMINAL.API.IncludedBefore")}<span class="terminal-macros-link" style="cursor:pointer; text-shadow: 0 0 red;color:#ee9b3a;text-decoration:underline;">${tr("TERMINAL.API.IncludedLink")}</span>${tr("TERMINAL.API.IncludedAfter")}
          </p>
          <ul>
            <li>
              <strong>${tr("TERMINAL.API.OpenAll")}</strong>
              <ul>
                <li>${tr("TERMINAL.API.OpenAllHint")}</li>
              </ul>
            </li>
            <li>
              <strong>${tr("TERMINAL.API.OpenOne")}</strong>
              <ul>
                <li>${tr("TERMINAL.API.OpenOneHint")}</li>
              </ul>
            </li>
          </ul>
        </section>
        <section>
          <h2>Monk's Active Tiles (MATT)</h2>
          <p>
            ${tr("TERMINAL.API.MattBefore")}<a href="https://github.com/CodaBool/terminal/wiki/Terminal-Click-to-Open">${tr("TERMINAL.API.MattLink")}</a>${tr("TERMINAL.API.MattAfter")}
          </p>
        </section>
        <section>
          <h2>API</h2>
          <h4 style="margin:0">${tr("TERMINAL.API.Client")}</h3>
          <pre><code class="language-javascript" style="padding-left: 1em; white-space: pre-wrap; background: black; color: white; user-select: text;">
new window.Terminal(tileUuid).render(true)
          </code></pre>
          <h4 style="margin:0">${tr("TERMINAL.API.OtherUser")}</h3>
          <pre><code class="language-javascript" style="padding-left: 1em;white-space: pre-wrap; background: black; color: white; user-select: text;">
game.socket.emit("module.terminal", {
  action: "render",
  tuid: tileUuid, // e.g. "Scene.u9dyY4mXfvAVzi3J.Tile.lKESQm6N5g4yrxBx"
  uid:  userId, // id or uuid e.g. "User.PR8dJFqczgK3X1ns"
})
          </code></pre>
        </section>
      </div>`,
    render: (event, dialog) => {
      const l = dialog.element.querySelector('.terminal-macros-link')
      if (l) {
        l.addEventListener('click', () => {
          game.packs.get('terminal.terminal-macros').render(true);
        });
      }
      dialog.element.querySelector("form").style.overflow = "auto"
    },
    buttons: [{
      action: "1",
      label: tr("TERMINAL.Common.Done"),
      icon: "fas fa-check",
    }],
  })
}

function genericListener(e, flag, name) {
  let id = e.target.parentElement.attributes["data-entry-id"]?.value
  if (!id) {
    id = e.target.parentElement.parentElement.attributes["data-entry-id"]?.value
  }
  const generic = game[name].get(id)
  const tileUuid = game.settings.get(ID, "pointer")
  if (!id || !generic || !tileUuid) return
  game.settings.set("terminal", "pointer", "")
  Object.values(ui.windows).forEach(app => app.maximize())
  foundry.applications.instances.forEach(a => a.maximize())
  const tileDoc = fromUuidSync(tileUuid)
  if (flag === "keycard") {
    const sourceId = generic?.flags?.core?.sourceId
    id = `${id}@${sourceId || ""}`
  }
  setFlagAndTab(tileDoc, flag, id)
  return true
}

export function injectListener(flag, name, funcName) {
  if (typeof libWrapper !== 'undefined') {
    // sometimes I see event not spread (libwrapper docs) but item-piles takes issue with it not being spread
    libWrapper.register(ID, `foundry.applications.sidebar.tabs.${funcName}.prototype._onClickEntry`, (wrapped, ...event) => {
      if (typeof wrapped !== "function") return
      let preventDefault = false
      try {
        preventDefault = genericListener(event[0], flag, name)
      } catch (error) {
        console.error(error)
      }
      if (preventDefault) {
        return
      }
      return wrapped(...event)
    }, "MIXED")
  } else {
    extendFunction(funcName + ".prototype._onClickEntryName", async e => {
      return genericListener(e, flag, name)
    })
  }
}

// inject into directory events for Terminal tile config ID helper buttons
function extendFunction(functionName, injectedFunc) {
  const oldFunc = eval(functionName)
  const wrapper = (w, ...arg) => w(...arg)
  eval(`${functionName} = async function (event) {
    let preventDefault = false
    try {
      preventDefault = await injectedFunc(event)
    } catch(error) {
      console.error(error)
    }
    if (preventDefault) return
    return wrapper.call(this, oldFunc.bind(this), ...arguments);
  }`)
}

export function updateStyleDefaults() {
  if (!game.user.isGM) return
  const styles = game.settings.get(ID, "styles")
  try {
    for (const [k, v] of Object.entries(defaultStyles)) {
      if (!styles[k]) {
        console.log(`Terminal | adding a new preset style of ${v.name}`)
        styles[k] = v
        continue
      }
      for (const [key, value] of Object.entries(v)) {
        if (styles[k][key] !== value) {
          console.log(`Terminal | performing update to preset style '${v.name}' and its '${key}' value`)
          styles[k][key] = defaultStyles[k][key]
        }
      }
    }
    // temporarily add the showASCIILoading prop and default to true
    for (const style of Object.values(styles)) {
      if (typeof style.showASCIILoading === 'undefined') {
        console.log(`Terminal | adding new property to style '${style.name}' to enforce a new default behavior of showing ASCII & Loading`)
        style.showASCIILoading = true
      }
    }
    game.settings.set("terminal", "styles", styles)
  } catch (e) {
    console.error("Terminal |", e)
  }
}

export function migrateToUuid() {
  let content = '<div style="overflow: auto; max-height: 600px">'
  setTimeout(() => {
    if (!game.user.isGM) return
    const tiles = game.settings.get("terminal", "tiles")
    const deprecated = []
    for (const s of game.scenes) {
      content += `<p style="font-size: 1.5em; user-select: text">${tr("TERMINAL.Migration.Log.Scene", { scene: s.name })}</p>`
      for (const t of s.tiles) {
        const temp = {}
        content += `<p style="font-size: 1.3em; user-select: text">${tr("TERMINAL.Migration.Log.Tile", { tile: t.id })}</p>`
        if (t.flags.terminal?.ssh) {
          if (t.flags.terminal.ssh?.includes(".")) {
            content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.SSHSkip")}</p>`
          } else {
            content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.SSHFound", { value: t.flags.terminal.ssh })}</p>`
            temp.ssh = t.flags.terminal.ssh
          }
        }
        if (t.flags.terminal?.unlockIds) {
          if (t.flags.terminal.unlockIds?.includes(".")) {
            content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.DoorSkip")}</p>`
          } else {
            content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.DoorFound", { value: t.flags.terminal.unlockIds })}</p>`
            temp.unlockIds = t.flags.terminal.unlockIds
          }
        }

        let foundJournal = false
        for (const [tid, jid] of Object.entries(tiles)) {
          if (tid === t.id) {
            foundJournal = true
            if (t.flags.terminal?.journal) {
              content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.FlagSkip")}</p>`
              continue
            }
            const journal = game.journal.get(jid)
            if (!journal) {
              // give error that a tile is misconfigured but still do a best effort attempt to migrate it
              content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.JournalMissing", { journal: jid })}</p>`
            }
            content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.JournalFound", { value: journal?.name })}</p>`
            temp.journal = jid
          }
        }

        if (!foundJournal) {
          content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.NoJournal")}</p>`
        }
        if (Object.keys(temp).length !== 0) {
          deprecated.push({ ...temp, tid: t.id, sceneName: s.name, tuid: t.uuid, sid: s.id })
        }
      }
    }
    if (!deprecated.length) return

    foundry.applications.api.DialogV2.wait({
      window: { title: tr("TERMINAL.Window.Migration") },
      content: `<p style="font-size:1.3em; max-width: 500px">${tr("TERMINAL.Migration.Intro")}</p><iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/MdTXuEB0pwQ?si=fu1n-jWNGq5gW21t" frameborder="0" allow="autoplay;encrypted-media" allowfullscreen></iframe><p style="display: none; font-size:1.3em; max-width: 500px" id="terminal-migrate-script">${tr("TERMINAL.Migration.WhyText")}</p><div id="terminal-migrate-container" style="overflow: auto; max-height: 200px"></div><hr/><div id="terminal-migrate-results" style="overflow: auto; max-height: 200px"></div>`,
      render: (event, dialog) => {
        const whyBtn = document.createElement("button");
        whyBtn.innerHTML = `<i class='fa-solid fa-question'></i> ${tr("TERMINAL.Migration.Why")}`
        whyBtn.type = "button"
        whyBtn.addEventListener("click", () => {
          dialog.querySelector("#terminal-migrate-script").style.display = "block"
        })
        const migrateBtn = document.createElement("button");
        migrateBtn.innerHTML = `<i class='fa-solid fa-file-arrow-up'></i> ${tr("TERMINAL.Migration.Begin")}`
        migrateBtn.type = "button"

        const deprecatedList = document.createElement("ul")
        deprecated.forEach(tile => {
          const listItem = document.createElement("li")
          listItem.textContent = `${game.scenes.get(tile.sid)?.name} - ${tile.tid}`
          deprecatedList.appendChild(listItem)

          const subList = document.createElement("ul")
          subList.style.color = "tomato"

          if (tile.journal) {
            const sub = document.createElement("li")
            sub.textContent = `JOURNAL - ${tile.journal}`
            subList.appendChild(sub)
          }
          if (tile.ssh) {
            const sub = document.createElement("li")
            sub.textContent = `SSH - ${tile.ssh}`
            subList.appendChild(sub)
          }
          if (tile.unlockIds) {
            const sub = document.createElement("li")
            sub.textContent = `DOOR - ${tile.unlockIds}`
            subList.appendChild(sub)
          }
          listItem.appendChild(subList)
        })
        const div = dialog.querySelector("#terminal-migrate-container")
        div.appendChild(deprecatedList)

        // migration script
        migrateBtn.addEventListener("click", async () => {
          game.settings.register(ID, "migrating", { type: Boolean, default: true })
          game.settings.set(ID, "migrating", true)

          const upgradeList = document.createElement("ul")
          content += `<br/><p style="font-size: 1.5em; user-select: text">${tr("TERMINAL.Migration.Starting")}</p>`

          for (const tile of deprecated) {
            const doc = await fromUuid(tile.tuid)
            if (!doc) {
              content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.TileMissing", { tile: tile.tuid, scene: tile.sceneName })}</p>`
              console.error(`Terminal | could not get tile document for ${tile.tuid}`, tile)
              continue
            }
            const listItem = document.createElement("li")
            listItem.textContent = `${tile.sceneName} - ${tile.tid}`
            upgradeList.appendChild(listItem)
            const subList = document.createElement("ul")
            subList.style.color = "cadetblue"
            if (tile.journal) {
              const sub = document.createElement("li")
              content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.AddFlag", { journal: tile.journal })}</p>`
              doc.setFlag(ID, "journal", tile.journal)
              sub.textContent = `JOURNAL - same value, just converted from global flag to tile flag`
              subList.appendChild(sub)
            }
            if (tile.ssh) {
              const sub = document.createElement("li")
              content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.ConvertSSH", { from: tile.ssh, to: `Scene.${tile.sid}.Tile.${tile.ssh}` })}</p>`
              doc.setFlag(ID, "ssh", `Scene.${tile.sid}.Tile.${tile.ssh}`)
              sub.textContent = `SSH - Scene.${tile.sid}.Tile.${tile.ssh}`
              subList.appendChild(sub)
            }
            if (tile.unlockIds) {
              const sub = document.createElement("li")

              const uuidList = tile.unlockIds.split(",").map(id => `Scene.${tile.sid}.Wall.${id}`).join(",")
              content += `<p style="user-select: text">${tr("TERMINAL.Migration.Log.ConvertDoor", { from: tile.unlockIds, to: uuidList })}</p>`

              doc.setFlag(ID, "unlockIds", uuidList);
              sub.textContent = `DOOR - ${uuidList}`
              subList.appendChild(sub)
            }
            listItem.appendChild(subList)
          }

          dialog.querySelector("#terminal-migrate-results").appendChild(upgradeList)

          migrateBtn.disabled = true;
          migrateBtn.style.display = "inline"
          migrateBtn.style.margin = "0 0 2em 0"
          migrateBtn.innerHTML = `${tr("TERMINAL.Migration.Complete", { count: deprecated.length })} <i class='fa-solid fa-check'></i>`

          content += `</div>`

          const logsBtn = document.createElement("button")
          logsBtn.innerHTML = `<i class='fa-solid fa-clipboard'></i> ${tr("TERMINAL.Migration.Logs")}`
          logsBtn.type = "button"
          logsBtn.addEventListener("click", () => {
            foundry.applications.api.DialogV2.prompt({
              window: { title: tr("TERMINAL.Window.Logs") },
              content,
            })
          })
          dialog.querySelector("#terminal-migrate-results").after(logsBtn)
          // open logs by default
          foundry.applications.api.DialogV2.prompt({
            window: { title: tr("TERMINAL.Window.Logs") },
            content,
          })
          setTimeout(() => { game.settings.set(ID, "migrating", false) }, 10_000)
        })
        dialog.querySelector("#terminal-migrate-results").after(migrateBtn)
        dialog.querySelector("#terminal-migrate-results").after(whyBtn)
        dialog.querySelector("footer").remove()
      },
      buttons: [{}]
    })
  }, 3_000)
}

// GM
export function alienCharge(actorIds) {
  for (const actorId of actorIds) {
    const actor = game.actors.get(actorId)
    for (const i of actor.items) {
      let item = game.items.filter(
        j => j.flags.core?.sourceId === i.flags.core?.sourceId,
      )
      if (item.length === 0) {
        console.warn(
          "Terminal | failed to find item",
          i.flags.core?.sourceId,
          i.name,
          "falling back to match by name",
        )
        item = game.items.filter(j => j.name === i.name)
        if (item.length === 0) {
          console.error(
            "Terminal | failed to find item when using match by name for",
            i.name,
          )
          continue
        }
      }
      console.log(
        "Terminal | power",
        item[0].system.attributes.power.value,
        "on",
        i.name,
        "which has a max of",
        i.system.attributes.power.value,
      )
      if (
        item[0].system.attributes.power.value !==
        i.system.attributes.power.value
      ) {
        i.update({
          "system.attributes.power.value":
            item[0].system.attributes.power.value,
        })
      }
    }
  }
  if (game.settings.get(ID, "notice")) {
    const names = actorIds.map(a => game.actors.get(a).name).toString()
    ui.notifications.info(tr("TERMINAL.Action.ItemsCharged", { names }))
  }
}

export async function replaceImageHelper(tileDoc, appId) {
  const replace = await foundry.applications.api.DialogV2.confirm({
    rejectClose: false,
    window: { title: tr("TERMINAL.Window.StylingQuickStart") },
    classes: ["terminal-image-helper"],
    position: { width: 400 },
    content: `<p style="font-size:1.2em">${tr("TERMINAL.Dialog.QuickStartIntro")}</p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px">
          <label>
            <input type="radio" name="image" value="terminal" checked>
            <img src="/modules/terminal/background/terminal.webp" style="width: 80%; height: auto; background: grey; cursor: pointer">
          </label>
          <label>
            <input type="radio" name="image" value="terminal_icon_1">
            <img src="/modules/terminal/background/terminal_icon_1.webp" style="width: 80%; height: auto; background: grey; cursor: pointer">
          </label>
        </div>
        <div class="form-group">
            <label>${tr("TERMINAL.Dialog.Tint")}</label>
            <div class="form-fields">
                <color-picker name="color" value="#ffffff">
                    <input type="text" placeholder="#FFFFFF">
                    <input type="color">
                </color-picker>
            </div>
            <p class="hint">${tr("TERMINAL.Dialog.TintHint")}</p>
        </div>
        <div class="form-group">
            <label>${tr("TERMINAL.Dialog.CreateLight")}</label>
            <div class="form-fields">
                <input type="checkbox" name="addLight" checked/>
            </div>
            <p class="hint">${tr("TERMINAL.Dialog.CreateLightHint")}</p>
        </div>
        `,
  })

  if (!replace) return

  // replace image
  const image = document.querySelector('.terminal-image-helper input[name="image"]:checked').value
  const color = document.querySelector('.terminal-image-helper color-picker').value
  tileDoc.update({
    "texture.src": `/modules/terminal/background/${image}.webp`,
    "texture.tint": color,
  })

  // create light
  const addLight = document.querySelector('.terminal-image-helper input[name="addLight"]').checked
  if (addLight) {
    const scene = tileDoc.parent
    const size = scene.grid.size
    const bright = (tileDoc.width / size + tileDoc.height / size) / 2 * scene.grid.distance
    scene.createEmbeddedDocuments("AmbientLight", [{
      x: tileDoc._object.center.x,
      y: tileDoc._object.center.y,
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

export async function openDialog(app, flag, readableName, type) {
  new foundry.applications.api.DialogV2({
    // modal: true,
    rejectClose: false,
    window: { title: tr("TERMINAL.Window.EditValue", { name: readableName }), },
    buttons: [{
      action: "1",
      label: tr("TERMINAL.Common.Save"),
      default: true,
      icon: "fas fa-check",
      callback: (event, target, el) => {
        setFlagAndTab(app, flag, el.element.querySelector("input").value)
      }
    },
    {
      action: "2",
      label: tr("TERMINAL.Common.Cancel"),
      icon: "fa-solid fa-ban",
    }],
    content: `<p style="font-size:1.3em">${tr("TERMINAL.Dialog.EnterValue", { name: readableName })}</p><input type="${type}" autofocus onfocus="var temp_value=this.value; this.value=''; this.value=temp_value" style="margin-bottom: 1em; width: 100%" value="${app.getFlag(ID, flag) || ''}" />`,
  }).render(true)
}

export function moveToken(token, update) {
  // only on current scene
  if (token.parent.id !== canvas.scene.id) return
  // only if you own them
  if (!token.isOwner) return
  // only if you have them selected
  if (!canvas.tokens.controlled.find(t => t.id === token.id)) return

  token.parent.tiles.forEach(tile => {
    if (tile.flags[ID]?.enabled) {
      if (game.combat?.active && tile.flags[ID]?.ignoreIfEncounter) return
      if (game.modules.get("levels")?.active) {
        if (CONFIG.Levels.API.inRange(tile, token.elevation)) {
          checkCollision(token, tile, update)
        }
      } else {
        if (token.elevation !== tile.elevation) return
        checkCollision(token, tile, update)
      }
    }
  })
}

// can look at Foundry #preUpdateMovement function for reference
export function checkCollision(token, tile, update) {
  const radiusWidth = (token.width * token.parent.dimensions.size) / 2
  const radiusHeight = (token.height * token.parent.dimensions.size) / 2
  const destination = { x: (update.destination?.x || token.x) + radiusWidth, y: (update.destination?.y || token.y) + radiusHeight }
  let origin = {}
  let wasInside = false
  let destinationInside = false

  if (game.release.generation >= 14) {
    origin = { x: token?._movement.origin?.x + radiusWidth, y: token?._movement.origin?.y + radiusHeight }
    if (
      origin.x >= tile.x - tile.width / 2 &&
      origin.x <= tile.x + tile.width / 2 &&
      origin.y >= tile.y - tile.height / 2 &&
      origin.y <= tile.y + tile.height / 2
    ) {
      wasInside = true
    }
    if (
    destination.x >= tile.x - tile.width / 2 &&
    destination.x <= tile.x + tile.width / 2 &&
    destination.y >= tile.y - tile.height / 2 &&
    destination.y <= tile.y + tile.height / 2
    ) {
      destinationInside = true
    }
  } else {
    origin = { x: token.x + radiusWidth, y: token.y + radiusHeight }
    if (
      origin.x >= tile.x &&
      origin.x <= tile.x + tile.width &&
      origin.y >= tile.y &&
      origin.y <= tile.y + tile.height
    ) {
      wasInside = true
    }
    if (
    destination.x >= tile.x &&
    destination.x <= tile.x + tile.width &&
    destination.y >= tile.y &&
    destination.y <= tile.y + tile.height
    ) {
      destinationInside = true
    }
  }

  // Check if the token's new position is within the tile's boundaries
  if (destinationInside) {
    // it is inside
    if (!wasInside) {
      renderTerminal(tile, token)
      return
    }
  } else {
    // outside
    if (wasInside) {
      if (document.querySelector(".terminal-window")) {
        document.querySelectorAll(`.terminal-window button[data-action="close"]`).forEach(e => e.click())
      }
      if (document.querySelector(".terminal-skilled")) {
        document.querySelectorAll(`.terminal-skilled button[data-action="close"]`).forEach(e => e.click())
      }
      return
    }
  }

  // Calculate the Euclidean distance between the token center and the tile center
  const distance = Math.sqrt(
    Math.pow(destination.x - tile.object.bounds.center.x, 2) +
    Math.pow(destination.y - tile.object.bounds.center.y, 2)
  )

  if (Math.abs(distance - ((tile.width + tile.height) / 4)) < canvas.scene.dimensions.size * 2) {
    // too close to an edge, ignore ray checks
    return
  }

  if (
    (foundry.utils.lineSegmentIntersection(token.object.center, destination, { x: tile.object.bounds.bottomEdge.A.x, y: tile.object.bounds.bottomEdge.A.y }, { x: tile.object.bounds.bottomEdge.B.x, y: tile.object.bounds.bottomEdge.B.y }) === null) &&
    (foundry.utils.lineSegmentIntersection(token.object.center, destination, { x: tile.object.bounds.leftEdge.A.x, y: tile.object.bounds.leftEdge.A.y }, { x: tile.object.bounds.leftEdge.B.x, y: tile.object.bounds.leftEdge.B.y }) === null) &&
    (foundry.utils.lineSegmentIntersection(token.object.center, destination, { x: tile.object.bounds.topEdge.A.x, y: tile.object.bounds.topEdge.A.y }, { x: tile.object.bounds.topEdge.B.x, y: tile.object.bounds.topEdge.B.y }) === null) &&
    (foundry.utils.lineSegmentIntersection(token.object.center, destination, { x: tile.object.bounds.rightEdge.A.x, y: tile.object.bounds.rightEdge.A.y }, { x: tile.object.bounds.rightEdge.B.x, y: tile.object.bounds.rightEdge.B.y }) === null)
  ) {
    return
  }
  ui.notifications.info(tr("TERMINAL.Action.PassedTerminal"))
}

function renderTerminal(tile, token) {
  if ($(".terminal-window").length) return

  const flags = tile.flags[ID]
  if (game.user.isGM) {
    if (flags.skilled || flags.keycard) {
      ui.notifications.info(tr("TERMINAL.Action.GMSkipRequirement"))
    }
    autoShadowTerminal(tile.uuid)
    new Terminal(tile.uuid).render(true)
    return
  }

  // pick a random GM
  const gms = game.users?.filter(u => u.active && u.isGM)
  if (!gms.length) {
    ui.notifications.error(tr("TERMINAL.Error.NoGM"))
    return
  }

  if (flags.skilled) {
    new Skilled(tr("TERMINAL.Skilled.Difficult")).render(true)
    game.socket.emit("module.terminal", {
      action: "gmApprove",
      uid: game.user.id,
      tuid: tile.uuid,
      gid: gms[0]._id,
      name: game.user.name,
      title: "opening a skilled Terminal",
    })
    return
  }

  if (flags.keycard) {
    const sourceId = flags.keycard.split("@")[1]
    let item = []
    if (!sourceId) {
      const itemName = game.items.get(flags.keycard.split("@")[0])?.name
      console.log("Terminal | using fallback method of matching by item name of", itemName)
      item = token.actor.items.filter(i => itemName === i.name)
    } else {
      // use percision method
      // this seems to only be present if the item has been added at to an inventory before
      item = token.actor.items.filter(i => i.flags?.core?.sourceId === flags.keycard.split("@")[1])
    }
    if (!item?.length) {
      new Skilled(tr("TERMINAL.Skilled.ItemRequired")).render(true)
      return
    }
    ui.notifications.info(tr("TERMINAL.Action.AuthItemAccepted"))
  }
  autoShadowTerminal(tile.uuid)
  new Terminal(tile.uuid).render(true)
}

export function autoShadowTerminal(tuid, arg = {}) {
  const { auto, source, shadow } = game.settings.get(ID, "shadow")
  // console.log("walk", source, shadow, game.user.id)
  if (!auto) return

  // see if necessary players are online
  const shadowOn = game.users.filter(p => p.active).some(u => u.id === shadow)
  const sourceOn = game.users.filter(p => p.active).some(u => u.id === source)
  if (!shadowOn || !sourceOn) {
    const shadowName = game.users.find(u => u.id === shadow)?.name || shadow
    const sourceName = game.users.find(u => u.id === source)?.name || source
    if (game.user.isGM) {
      ui.notifications.error(tr("TERMINAL.Error.ShadowOffline", { user: game.user.name, shadow: shadowName, source: sourceName }))
    } else {
      game.socket.emit("module.terminal", {
        action: "notify",
        errorKey: "TERMINAL.Error.ShadowOffline",
        errorData: { user: game.user.name, shadow: shadowName, source: sourceName },
      })
    }
  }

  console.log("sending", {
    action: "render",
    uid: shadow,
    myself: game.user.id === source,
    tuid,
    arg: { shadowView: true, ...arg },
  })

  game.socket.emit('module.terminal', {
    action: "render",
    uid: shadow,
    myself: game.user.id === source,
    tuid,
    arg: { shadowView: true, ...arg },
  })
}

export async function setFlagAndTab(tileDoc, name, value) {
  const id = `TileConfig-${tileDoc.uuid.replace(/\./g, '-')}`
  await tileDoc.setFlag(ID, name, value)

  // keep focus on Terminal tab
  let config = document.querySelector("#" + id)
  if (!config) {
    config = document.querySelector("#Active" + id)
  }

  config?.querySelectorAll('.active')?.forEach(el => {
    if (el.getAttribute("data-tab") && el.classList.contains("terminal-submenu")) {
      const tab = el.getAttribute("data-tab")
      if (tab !== "general") {
        setTimeout(() => {
          config?.querySelectorAll(".terminal-submenu").forEach(el => el.classList.remove("active"))
          config?.querySelectorAll(".terminal-subtab").forEach(el => el.classList.remove("active"))
          config?.querySelector(`div[data-tab="${tab}"]`).classList.add("active")
          config?.querySelector(`a[data-tab="${tab}"]`).classList.add("active")
        }, 0)
      }
    }
  })

  // deprecated since V12
  // config?.querySelector(`a[data-tab='terminal']`)?.click()

  // setTimeout(() => {
  //   // config?.querySelector(`a[data-tab='terminal']`)?.click()

  //   // possibly unecessary, but just in case
  //   config?.querySelectorAll('.active')?.forEach(el => {
  //     if (el.getAttribute("data-tab") && el.classList.contains("terminal-submenu")) {
  //       const tab = el.getAttribute("data-tab")
  //       if (tab !== "general") {
  //         config?.querySelectorAll(".terminal-submenu").forEach(el => el.classList.remove("active"))
  //         config?.querySelectorAll(".terminal-subtab").forEach(el => el.classList.remove("active"))
  //         config?.querySelector(`div[data-tab="${tab}"]`).classList.add("active")
  //         config?.querySelector(`a[data-tab="${tab}"]`).classList.add("active")
  //       }
  //     }
  //   })

  // }, 50)
}

export async function updatePage(journalId, pageId) {
  const journal = game.journal.get(journalId)
  for (const page of journal.pages) {
    if (page._id !== pageId) continue
    page.update({ "ownership.default": -1 })
  }
}

export async function updateJournal(journalId, userId) {
  const j = game.journal.get(journalId)
  if (!j) {
    ui.notifications.error(
      tr("TERMINAL.Error.NoJournalOnEnabled", { name: game.users.get(userId)?.name }),
      { permanent: true },
    )
    return
  }
  const perm = j.ownership.default
  if (game.settings.get(ID, "notice")) {
    ui.notifications.info(
      tr("TERMINAL.Info.ObserverGranted", { name: game.users.get(userId)?.name, journal: j.name }),
    )
  }
  if (perm < 2) {
    await j.update({ ownership: { [userId]: 2 } }) // observer
  }
}

export async function validateDoor(uuid) {
  if (uuid === "") return false
  const wall = await fromUuid(uuid)
  if (!wall || wall?.door === CONST.WALL_DOOR_TYPES.NONE) {
    if (game.user.isGM) {
      ui.notifications.error(
        tr("TERMINAL.Error.InvalidDoorUUID", { uuid }),
        { permanent: true },
      )
    } else {
      game.socket.emit("module.terminal", {
        action: "notify",
        errorKey: "TERMINAL.Error.InvalidDoorUUIDRemote",
        errorData: { name: game.user.name, uuid, scene: canvas.scene.name },
        perm: true,
      })
    }
    return false
  }
  return wall
}

// only ran by GM
export async function toggleDoorLock(wallId, userName, uid) {
  // support also just providing the wall document so save API call
  let wall = typeof wallId === "string" ? await fromUuid(wallId) : wallId
  if (!wall || wall?.door === CONST.WALL_DOOR_TYPES.NONE) {
    ui.notifications.error(
      tr("TERMINAL.Error.ToggleDoor", { uuid: wallId }),
    )
    return
  }
  const newWall = await wall.update({
    ds: wall.ds === CONST.WALL_DOOR_STATES.LOCKED
      ? CONST.WALL_DOOR_STATES.OPEN
      : CONST.WALL_DOOR_STATES.LOCKED,
  })
  if (!newWall) {
    ui.notifications.error(
      tr("TERMINAL.Error.ToggleDoor", { uuid: wallId }),
    )
    return
  }
  // send notifications when ran for another user
  if (game.settings.get(ID, "notice") && uid) {
    const doorName = newWall.getFlag(ID, "name") || tr("TERMINAL.DocType.Door")
    if (newWall.ds === CONST.WALL_DOOR_STATES.OPEN) {
      ui.notifications.info(
        tr("TERMINAL.Action.DoorUnlocked", { door: doorName, scene: newWall.parent.name, user: userName }),
      )
    } else {
      ui.notifications.info(
        tr("TERMINAL.Action.DoorLockedBy", { door: doorName, scene: newWall.parent.name, user: userName }),
      )
    }
  }
}

export function generatePseudoText(lines) {
  let randomText = ""
  for (let i = 0; i < lines; i++) {
    randomText += (Math.random() + 1).toString(36).substring(2) + "\n"
  }
  return randomText
}

export async function toggleLights(sceneId) {
  const scene = game.scenes.get(sceneId)
  let key = "environment.darknessLevel"
  if (game.release.generation === 13) {
    key = "darkness"
  }

  if (Math.abs(scene.environment.darknessLevel - 0) < Math.abs(scene.environment.darknessLevel - 1)) {
    scene.update({ [key]: 1.0 }, { animateDarkness: 1000 })
  } else {
    scene.update({ [key]: 0.0 }, { animateDarkness: 1000 })
  }
}


// if ran by user they need observer on the macro
export function triggerRegion(uuid, event, tuid) {
  const region = fromUuidSync(uuid)
  const tile = fromUuidSync(tuid)
  if (!region || !tile) {
    ui.notifications.error(tr("TERMINAL.Error.RegionBroken"))
    return
  }
  const obj = {
    name: event,
    data: {},
    region,
    user: game.user,
  }
  if (event.includes("token")) {
    let tokens = canvas.tokens.objects.children

    // catch no token issue
    if (!tokens.length) {
      if (game.user.isGM) {
        ui.notifications.error(tr("TERMINAL.Error.TokenRequired"))
      } else {
        ui.notifications.error(tr("TERMINAL.Error.SentToGM"))
        game.socket.emit("module.terminal", {
          action: "notify",
          errorKey: "TERMINAL.Error.RegionNoTokens",
          errorData: { user: game.user.name, event, region: region.name, scene: canvas.scene.name },
          perm: true
        })
      }
      return
    }

    // teleport
    const teleport = region.behaviors?.find(b => b.type === "teleportToken")
    if (event === "tokenMoveIn" && !teleport?.disabled && teleport) {
      tokens = tokens.filter(t => t.isOwner)
      if (tokens.length !== 1) {
        ui.notifications.error(tr("TERMINAL.Error.OneToken", {
          count: tokens.length,
          gmHint: game.user.isGM ? tr("TERMINAL.Error.GMPlayerHint") : "",
        }))
        return
      }
      if (!game.user.isGM) {
          game.socket.emit("module.terminal", {
            action: "notify",
            messageKey: "TERMINAL.Action.Teleported",
            messageData: { user: game.user.name, scene: canvas.scene.name },
          })
          game.socket.emit("module.terminal", {
            action: "teleport",
            sid: canvas.scene.id,
            gid: game.users?.filter(u => u.active && u.isGM)[0].id,
            rid: uuid,
            event,
            uid: game.user.id,
            token: tokens[0].id,
          })
        return
      }
    }
    obj.data.token = tokens[0].document
    obj.data.movement = tokens[0].document.movement
  }
  console.log(`Terminal | Simulating '${event}' event for Region '${region.name}' ${obj.data.token ? "using token '" + obj.data.token.name + "'" : ''}`)

  console.log("handle", obj)
  region._handleEvent(obj)
}

export async function startObservation(actorId, observeTimer, uid, tuid) {
  // return if there is already a timer running
  const tile = await fromUuid(tuid)
  const observeTimerRunning = tile.getFlag(ID, "observeTimerRunning")
  if (observeTimerRunning) {
    if (uid) {
      game.socket.emit("module.terminal", {
        action: "notify",
        uid,
        errorKey: "TERMINAL.Error.ControlRunning",
      })
    } else {
      ui.notifications.error(tr("TERMINAL.Error.ControlRunning"))
    }
    return
  }

  const name = game.actors.get(actorId)?.name

  // check that the observeScene and current scene match
  const tileOffScene = canvas.scene.id !== tile.parent.id
  const observeScene = tile.getFlag(ID, "observeScene")

  if (tileOffScene && !observeScene) {
    if (uid) {
      game.socket.emit("module.terminal", {
        action: "notify",
        uid,
        perm: true,
        messageKey: "TERMINAL.Info.OffSceneControl",
        messageData: { terminalScene: tile.parent.name, name, currentScene: canvas.scene.name },
      })
    } else {
      ui.notifications.info(tr("TERMINAL.Info.OffSceneControl", {
        terminalScene: tile.parent.name,
        name,
        currentScene: canvas.scene.name,
      }), { permanent: true })
    }
  }

  await updatePerm(2, actorId)

  if (observeScene) {
    const scene = game.scenes.get(observeScene);
    if (scene) {
      if (uid) {
        if (game.users.get(uid).viewedScene !== observeScene) {
          game.socket.emit("module.terminal", {
            action: "changeScene",
            uid,
            sid: observeScene,
            showObservers: true,
            actorId,
            observeTimer,
          })
        }
      } else {
        await scene.view()
      }
    } else {
      ui.notifications.error(tr("TERMINAL.Error.SceneGone", { scene: observeScene }))
    }
  }


  // Foundry V12+ initialize function no longer updates vision.
  // Instead use workaround of altering a vision related item slightly
  for (const t of canvas.tokens.objects.children) {
    if (t.document.actorId !== actorId) continue
    if (Math.random() < 0.5) {
      t.document.update({ "sight.contrast": t.document.sight.contrast + 0.00000001 })
    } else {
      t.document.update({ "sight.contrast": t.document.sight.contrast - 0.00000001 })
    }
  }

  // start a chat message
  if (observeTimer) {
    const chat = await ChatMessage.create({
      content: tr("TERMINAL.Chat.ControlTimer", { name, seconds: observeTimer }),
    })
    let timer = observeTimer
    const timerId = setInterval(() => {
      if (timer > 0) {
        chat.update({
          content: tr("TERMINAL.Chat.ControlTimer", { name, seconds: timer }),
        })
        timer--
      } else {
        clearInterval(timerId)
        chat.update({ content: tr("TERMINAL.Chat.AccessRevoked", { name }) })
        game.actors.get(actorId).update({ ownership: { default: 0 } }).then(() => {
          game.socket.emit("module.terminal", { action: "resetVision" })
        })
        tile.setFlag(ID, "observeTimerRunning", false)
      }
    }, 1000)
    // only allow one timer to run
    tile.setFlag(ID, "observeTimerRunning", true)
  }

  if (!observeScene && tileOffScene) return
  if (observeScene && (canvas.scene.id !== observeScene)) return
  if (uid) {
    // send to client
    game.socket.emit("module.terminal", {
      action: "showObservers",
      actorId,
      uid,
      observeTimer,
    })
    return
  }

  // running as GM
  showObservers(actorId, observeTimer)
}
export async function runMacro(tuid, uid, proxyTarget) {
  const tileDoc = await fromUuid(tuid)
  if (tileDoc.getFlag(ID, "macroLocked") && tileDoc.getFlag(ID, "lockableMacro")) {
    if (uid) {
      game.socket.emit("module.terminal", {
        action: "notify",
        uid,
        errorKey: "TERMINAL.Error.OnlyOnce",
      })
    } else {
      ui.notifications.error(tr("TERMINAL.Error.OnlyOnce"))
    }
    return
  }
  const macro = game.macros.get(tileDoc.getFlag(ID, "macro"))
  const user = game.users.get(proxyTarget)

  if (user) {
    // proxy this back to user
    const userCanExec = macro.ownership.default > 0 || macro.ownership[proxyTarget] > 0
    if (userCanExec) {
      // give notice
      if (game.settings.get(ID, "notice") && uid && (proxyTarget !== game.user.id)) {
        ui.notifications.info(tr("TERMINAL.Action.MacroRan", { name: game.users.get(uid).name, macro: macro.name }))
      }
      // lock
      if (tileDoc.getFlag(ID, "lockableMacro")) {
        tileDoc.setFlag(ID, "macroLocked", true)
      }
      game.socket.emit("module.terminal", {
        action: "userRunMacro",
        uid,
        args: tileDoc.getFlag(ID, "macroArgs").split(","),
        macroId: macro.id,
      })
    } else {
      ui.notifications.error(tr("TERMINAL.Error.MacroPermission", { name: user.name, macro: macro.name }))
    }
    return
  }

  macro.execute({
    args: tileDoc.getFlag(ID, "macroArgs").split(",")
  })

  // give notice
  if (game.settings.get(ID, "notice") && uid) {
    ui.notifications.info(tr("TERMINAL.Action.MacroRan", { name: game.users.get(uid).name, macro: macro.name }))
  }

  // lock
  if (tileDoc.getFlag(ID, "lockableMacro")) {
    tileDoc.setFlag(ID, "macroLocked", true)
  }
}

function pingTokens(shouldPan, center, limit, uid) {
  let hasPanned = false
  for (const t of canvas.tokens.objects.children) {

    // only detect within the limit
    const distance = canvas.grid.measurePath([center, t.center]).distance
    if (Number(limit) < distance && Number(limit) !== 0) continue

    let dead = t.actor.statuses.has("dead")

    // starfinder doesn't add statuses correctly
    // https://github.com/foundryvtt-starfinder/foundryvtt-starfinder/issues/1002#issuecomment-2000620732
    if (game.system.id === "sfrpg" && !dead) {
      dead = t.actor.system.conditions.dead
    }

    if (t.document.hidden) {
      dead = true
    }

    // skip over item piles tokens
    if (t.document.flags["item-piles"]) {
      dead = true
    }

    if (!dead) {
      canvas.ping(t.center)

      // only pan once, and when told that a pan should be done
      // this basically picks a the first token it lists
      // in canvas.tokens.objects.children that's dead
      if (!hasPanned && shouldPan) {
        hasPanned = true
        if (uid) {
          game.socket.emit('module.terminal', { action: "panToPoint", uid, x: t.x, y: t.y, min: true })
        } else {
          Object.values(ui.windows).forEach(app => app.minimize())
          foundry.applications.instances.forEach(a => a.minimize())
          canvas.animatePan({
            x: t.x,
            y: t.y,
            duration: 1700,
          })
          setTimeout(() => {
            Object.values(ui.windows).forEach(app => app.maximize())
            foundry.applications.instances.forEach(a => a.maximize())
          }, 2100)
        }
      }
    }
  }
}

// can be ran by GM & non-GM users
// also if uid is defined, then it's skilled and uses a GM proxy
// meaning there are 3 conditions this will run under
export async function detectMotion(uid, tuid, limit) {
  const tile = await fromUuid(tuid)
  const detectTimerRunning = tile.getFlag(ID, "detectTimerRunning")
  if (detectTimerRunning) {
    if (uid) {
      game.socket.emit("module.terminal", {
        action: "notify",
        uid,
        errorKey: "TERMINAL.Error.MotionRunning",
      })
    } else {
      ui.notifications.error(tr("TERMINAL.Error.MotionRunning"))
    }
    return
  }

  if (canvas.scene.id !== tile.parent.id && limit) {
    if (uid) {
      game.socket.emit("module.terminal", {
        action: "notify",
        errorKey: "TERMINAL.Error.MotionOtherScene",
        errorData: { current: canvas.scene.name, terminal: tile.parent.name },
      })
    } else {
      ui.notifications.error(tr("TERMINAL.Error.MotionOtherScene", { current: canvas.scene.name, terminal: tile.parent.name }))
    }
    return
  }

  let timer = 60
  const chat = await ChatMessage.create({
    content: tr("TERMINAL.Chat.MotionTimer", { seconds: timer }),
  })

  let center = { x: tile.x, y: tile.y }
  if (game.release.generation < 14) {
    center = { x: tile.x - tile.width / 2, y: tile.y - tile.height / 2 }
  }
  pingTokens(true, center, limit, uid)
  const timerId = setInterval(() => {
    if (timer > 0) {
      timer--
      chat.update({
        content: tr("TERMINAL.Chat.MotionTimer", { seconds: timer }),
      })
      if (timer % 5 !== 0) return
      pingTokens(false, center, limit, uid)
    } else {
      clearInterval(timerId)
      chat.update({
        content: tr("TERMINAL.Chat.MotionRevoked"),
      })
      // proxy turning off timer if not GM
      if (game.user.isGM) {
        tile.setFlag(ID, "detectTimerRunning", false)
      } else {
        game.socket.emit("module.terminal", {
          action: "setFlag",
          flag: "detectTimerRunning",
          value: false,
          tuid,
          gid: game.users?.filter(u => u.active && u.isGM)[0].id,
        })
      }
    }
  }, 1000)
  // only allow one timer to run and proxy if not GM
  if (game.user.isGM) {
    tile.setFlag(ID, "detectTimerRunning", true)
  } else {
    game.socket.emit("module.terminal", {
      action: "setFlag",
      flag: "detectTimerRunning",
      value: true,
      tuid,
      gid: game.users?.filter(u => u.active && u.isGM)[0].id,
    })
  }

  // notify
  if (uid) {
    // if a uid is specified then it used a GM proxy for the ping and should notify the client
    game.socket.emit("module.terminal", {
      action: "notify",
      uid,
      messageKey: "TERMINAL.Action.DetectingMinute",
    })
  } else {
    ui.notifications.info(tr("TERMINAL.Action.DetectingMinute"))
  }
}
export async function showObservers(actorId, observeTimer) {
  if (observeTimer) {
    ui.notifications.info(
      tr("TERMINAL.Action.AccessExpires", { seconds: observeTimer }),
    )
  }
  canvas.perception.update({ initializeVision: true, refreshLighting: true, refreshSounds: true });
  canvas.tokens.objects.children.forEach(t => t.control({ releaseOthers: false }));
  canvas.tokens.releaseAll()
  canvas.perception.initialize()
  Object.values(ui.windows).forEach(app => app.minimize())
  foundry.applications.instances.forEach(a => a.minimize())

  // TODO: find a better method, this gives some time for the canvas to load
  await new Promise(resolve => setTimeout(resolve, 1_000));

  let time = 0
  for (const t of canvas.tokens.objects.children) {
    if (t.document.actorId === actorId) {
      setTimeout(() => {
        canvas.animatePan({
          x: t.transform.position._x,
          y: t.transform.position._y,
          duration: 1700,
        })
      }, time)
      time += 2000
    }
  }
  setTimeout(() => {
    Object.values(ui.windows).forEach(app => app.maximize())
    foundry.applications.instances.forEach(a => a.maximize())
  }, time)
}

export async function updatePerm(level, actorId) {
  const actor = game.actors.get(actorId)
  if (!actor) {
    ui.notifications.error(tr("TERMINAL.Error.ActorMissing", { id: actorId }))
    return
  }
  await actor.update({ ownership: { default: level } })
}

// ran by either GM or player
export async function exploreMap(exploreForAll, skipNotification) {
  await canvas.tokens.releaseAll()
  await canvas.perception.initialize()
  canvas.fog.exploration.updateSource({
    explored:
      "data:image/webp;base64,UklGRjwAAABXRUJQVlA4IDAAAADQAQCdASoBAAEAAgA0JaACdLoB+AADsAD+8MQL/yC5YXXI1/8gP+QH/ID/+PIAAAA=",
    timestamp: Date.now(),
  })
  canvas.fog.exploration.constructor.create(canvas.fog.exploration.toJSON(), {
    loadFog: true,
  })

  if (exploreForAll) {
    for (const u of game.users) {
      if (u.isSelf || u.isGM) continue
      game.socket.emit("module.terminal", {
        action: "exploreMap",
        uid: u.id,
        skipNotification: true
      })
    }
  }

  if (skipNotification) return

  // notifications
  if (game.user.isGM) {
    if (game.settings.get(ID, "notice")) {
      ui.notifications.info(tr("TERMINAL.Action.MapRevealed", {
        scene: canvas.scene.name,
        shared: exploreForAll ? tr("TERMINAL.Action.SharedSuffix") : "",
      }))
    }
  } else {
    game.socket.emit("module.terminal", {
      action: "notify",
      messageKey: exploreForAll ? "TERMINAL.Action.MapRevealedShared" : "TERMINAL.Action.MapRevealedByUser",
      messageData: { user: game.user.name, scene: canvas.scene.name },
      notice: true,
    })
  }
}

export function buildFontOptions(selected) {
  const foundryOptions = Object.entries(CONFIG?.fontDefinitions || {}).map(([value, font]) => ({
    value,
    label: `Foundry: ${value}`,
  }))
  const commonFonts = [
    { value: "inherit", label: tr("TERMINAL.Font.ThemeDefault") },
    { value: "'Open Sans', sans-serif", label: "Open Sans" },
    { value: "Montserrat, sans-serif", label: "Montserrat" },
    { value: "'Noto Sans', sans-serif", label: "Noto Sans" },
    { value: "Orbitron, sans-serif", label: "Orbitron" },
    { value: "monospace, sans-serif", label: "Monospace" },
  ]

  const unique = [...commonFonts, ...foundryOptions].filter(
    (font, index, arr) => arr.findIndex(f => f.value === font.value) === index
  )
  return unique.map(font => ({
    ...font,
    selected: font.value === selected || (!selected && font.value === "inherit"),
  }))
}

export async function warmCache(timer, ranFromInit) {
  // if forge implements some auth for bazaar assets then use the user route mentioned here
  // https://forums.forge-vtt.com/t/using-the-filepicker-to-interact-with-the-assets-library/98008
  const warm = notify => {
    const urls = new Set()
    for (const t of canvas.tiles.objects.children) {
      if (!t.document.getFlag(ID, "enabled")) continue
      const assets = game.settings.get("terminal", "styles")[t.document.getFlag(ID, "style")]
      if (!assets) continue
      for (const k of ["background", "borderImage", "click", "close", "startup", "splashFile"]) {
        if (!assets[k]) continue
        let url = assets[k]
        if (typeof ForgeVTT !== 'undefined' && !assets[k].includes("http") && assets[k].includes("modules/terminal")) {
          // console.log("found relative path, convert to assets URL, forge exclusive", assets[k])
          url = ForgeVTT.ASSETS_LIBRARY_URL_PREFIX + "bazaar/modules/terminal-b65eff5013953bbe/assets" + assets[k].split("modules/terminal")[1]
        }
        if (urls.has(url)) continue
        urls.add(url)
        const ext = assets[k].split('.')?.pop()
        if (Object.keys(CONST.AUDIO_FILE_EXTENSIONS).includes(ext)) {
          // XML requests are the only way to have CF cache audio
          const xhr = new XMLHttpRequest()
          xhr.open('GET', url, true)
          xhr.send()
        } else {
          fetch(url, { mode: "no-cors" })
        }
      }
    }

    if (game.settings.get(ID, "notice") && notify && urls.size && game.user.isGM) {
      ui.notifications.info(tr("TERMINAL.Action.CacheWarmed", { count: urls.size }))
    } else if (urls.size) {
      console.log(`Terminal | warmed ${urls.size} assets`)
    }
  }

  warm(ranFromInit)
  if (timer === 0) return
  window.warmCacheInterval = setInterval(warm, timer)
}
