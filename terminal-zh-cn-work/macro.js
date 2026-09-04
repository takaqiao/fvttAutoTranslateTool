const terminalT = (key, data) => data ? game.i18n.format(key, data) : game.i18n.localize(key)

async function openForAll() {
    /*
      This macro opens a Terminal for all users regardless of their position in the scene
      An argument of the tile UUID for the specific Terminal you wish to use is required
  
      I recommend using the Monk's Active Tiles module to add clickable tiles which can
      run marcros with arguments. Alternatively you can use Foundry's built-in Regions
    */
  
    // validate existance of arguments
    if (typeof args === 'undefined') {
      ui.notifications.error(terminalT("TERMINAL.Macro.TileRequired"))
      return
    }
  
    // validate length of arguments
    if (args.length !== 1) {
      ui.notifications.error(terminalT("TERMINAL.Macro.OneArgument", { args }))
      return
    }
  
    // validate the tile's Terminal settings
    const t = await fromUuid(args[0])
    const valid = await window.validateTerminalTile(t, true)
  
    // if the validation is getting in your way feel free to comment the next line out
    if (!valid) return
  
    const users = game.users?.filter(u => u.active && !u.isGM)
  
    // prompt for skill check roll if skilled
    if (t.getFlag("terminal", "skilled")) {
      for (const user of users) {
        foundry.applications.api.DialogV2.wait({
          window: { title: terminalT("TERMINAL.Window.TerminalAccess") },
          content: `<p style="font-size:1.4em; max-width: 400px">${terminalT("TERMINAL.Dialog.SkillRequest", { name: user.name })}</p>`,
          buttons: [{
            action: "1",
            label: terminalT("TERMINAL.Common.Allow", { name: user.name }),
            icon: "fas fa-check",
            callback: () => {
              game.socket.emit('module.terminal', {
                uid: user.id, tuid: args[0], action: "render"
              })
            }
          },
          {
            action: "2",
            label: terminalT("TERMINAL.Common.Deny", { name: user.name }),
            icon: "fa-solid fa-ban",
          }],
        })
      }
    } else {
  
      // open terminal for each user
      ui.notifications.info(terminalT("TERMINAL.Action.OpenedFor", { count: users.length, names: users.map(u => u.name).join(", ") }))
      for (const user of users) {
        game.socket.emit('module.terminal', { action: "render", tuid: args[0], uid: user.id })
      }
    }
  }
  
  async function openForOne() {
    /*
      This macro opens a Terminal for a specific user regardless of their position
      in the scene. Arguments of the tile UUID for the specific Terminal you wish
      to use and optionally the user UUID
  
      If only the Tile UUID is given as an argument, and no user UUID
      this macro will assume you are running this with Monk's Active Tiles
      and that you have specified it to run "as triggering player".
  
      If you supply both the Tile ID and a user ID as arguments then it will
      assume its running as GM
  
      I recommend using the Monk's Active Tiles module to add clickable tiles which can
      run marcros with arguments. Alternatively you can use Foundry's built-in Regions
    */
  
  
    // validate existance of arguments
    if (typeof args === 'undefined') {
      ui.notifications.error(terminalT("TERMINAL.Macro.TileRequired"))
      return
    }
  
    // validate length of arguments
    if (args.length !== 2 && args.length !== 1) {
      ui.notifications.error(terminalT("TERMINAL.Macro.OneOrTwoArguments", { args }))
      return
    }
  
    // validate the tile's Terminal settings
    const t = await fromUuid(args[0])
    // TODO: should not await this and start using fromUuidSync
    const valid = await window.validateTerminalTile(t, true)
  
    // if the validation is getting in your way feel free to comment the next line out
    if (!valid) return
  
    // get user
    let user
    if (args.length !== 2) {
      user = game.user
    } else {
      user = await fromUuid(args[1])
    }
  
    // validate that the specific user exists
    if (!user) {
      ui.notifications.error(terminalT("TERMINAL.Macro.UserMissing", { user: args[1] }))
      return
    }
  
    // validate that the specific user is connected
    const userExists = game.users.some(u => u.id === user.id)
    if (!userExists) {
      ui.notifications.error(terminalT("TERMINAL.Macro.UserInactive", { user: user.name }))
      return
    }
  
    // if there are 2 args then open for specified user
    const skilled = t.getFlag("terminal", "skilled")
    if (args.length === 2) {
      if (skilled) {
        foundry.applications.api.DialogV2.wait({
          window: { title: terminalT("TERMINAL.Window.TerminalAccess") },
          content: `<p style="font-size:1.4em; max-width: 400px">${terminalT("TERMINAL.Dialog.SkillRequest", { name: user.name })}</p>`,
          buttons: [{
            action: "1",
            label: terminalT("TERMINAL.Common.Allow", { name: user.name }),
            icon: "fas fa-check",
            callback: () => {
              game.socket.emit('module.terminal', {
                uid: user.id, tuid: args[0], action: "render"
              })
            }
          },
          {
            action: "2",
            label: terminalT("TERMINAL.Common.Deny", { name: user.name }),
            icon: "fa-solid fa-ban",
          }],
        })
      } else {
        ui.notifications.info(terminalT("TERMINAL.Action.OpenedForOne", { name: user.name }))
        game.socket.emit('module.terminal', { action: "render", tuid: args[0], uid: user.id })
      }
    } else {
      // this block could be reached by either a GM or a user
      if (!game.user.isGM) {
        // find a connected GM
        const gms = game.users?.filter(u => u.active && u.isGM)
        if (!gms.length) {
          ui.notifications.error(terminalT("TERMINAL.Error.NoGM"))
          return
        }
        if (skilled && !game.user.isGM) {
          new Skilled(terminalT("TERMINAL.Skilled.Difficult")).render(true)
          game.socket.emit("module.terminal", {
            action: "gmApprove",
            uid: game.user.id,
            tuid: args[0],
            gid: gms[0]._id,
            name: game.user.name,
            title: "opening a skilled Terminal",
          })
          return
        }
        game.socket.emit('module.terminal', {
          action: "notify",
          messageKey: "TERMINAL.Macro.Opened",
          messageData: { user: user.name },
        })
      } else {
        ui.notifications.info(terminalT("TERMINAL.Macro.GMHint"))
      }
      // don't open if a Terminal is already open
      if (!document.querySelector(`.${args[0].replace(/\./g, '-')}`)) {
        new Terminal(args[0]).render(true)
      }
    }
  }
  
