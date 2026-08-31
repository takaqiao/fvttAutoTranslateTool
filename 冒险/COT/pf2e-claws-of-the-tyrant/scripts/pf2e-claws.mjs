/* Macro code lives here instead of inside the macro document so that it can be updated & maintained more easily,
    and without requiring reimports from users.
   As a result, macros only need to reference these methods, eg: game.modules.get("pf2e-claws-of-the-tyrant").macros.<macroname>()
*/

// Reusable function that pauses for a duration in milliseconds
function sleep(milliseconds) {
  return new Promise(resolve => {
      setTimeout(() => { resolve(''); }, milliseconds);
  })
}

// Reusable function that toggles the visibility of an array of placeables, if they can be found
async function togglePlaceables(placeables) {
  placeables.forEach(async (placeable) => {
    console.debug(placeable);
    if (placeable != undefined) { await placeable.update({hidden: !placeable.hidden}); }
    else { console.debug(`Placeable was not found on this scene.`); }
  });
}

// Reusable function that toggles the open status of a door
async function toggleDoor(door) {
  console.debug(door);
  if (door != undefined) {
    if (door.ds === 1) {
      if (door) await door.update({ds: 0});
    }
    else {
      if (door) await door.update({ds: 1});
    }
  } else {
    console.debug(`Door was not found on this scene.`);
  }
}

const MACROS = {
    // 1.1
    async taliaRises() {
        const tile = canvas.scene.tiles.get("cHRjnWxHaHBQa470");
        const token = canvas.scene.tokens.get("34cPQHM31yLfH4xl");
        await togglePlaceables([tile, token]);
    },
    async A09BreweryNoCollapse() {
        const tile = canvas.scene.tiles.get("C2g43CNGvTZvHyTP");
        await togglePlaceables([tile]);
    },
    async A09BreweryCollapse() {
        const tile = canvas.scene.tiles.get("X1eu2Yio3LHO137Y");
        await togglePlaceables([tile]);
    },
    async HBridgeCollapse() {
        const tile = canvas.scene.tiles.get("YvvZOLAJxxIgnbgY");
        await togglePlaceables([tile]);
    },
    // 1.2
    // 2.1
    async OzemAwareness() {
        if (canvas.scene.id !== "XasLoepXduwBdrtx") { return }
        const playlist = await game.playlists.get("WmRvOvVxgxRkrDiQ");
        const lights = {
            off: "",
            white: "CT0xgfvmf9rZnJOr",
            yellow: "ZVh2YqFih4DPV87R",
            blue: "aVTzVuqDf1Q6fSpb",
            red: "znZ5IlFmiwPvbK7U"
        };
        const tiles = {
            off: "",
            white: "V84NJEkKdfO8tHqi",
            yellow: "qW18S8E1EBWob700",
            blue: "y88f7N9sLHkaUoLP",
            red: "551AyGfFGt7RgI1B"
        };
        const awarenessLevels = {
            // off
            off: {
                flame: "off",
            },
            // All Clear
            level0: {
                flame: "white",
                bell: "DhISqYBgDHv5mr5K",
            },
            // Pair Up
            level1: {
                flame: "white",
                bell: "OqZLIjP2Ks7mcO2r",
            },
            // Increase Patrols
            level2: {
                flame: "yellow",
                bell: "bPY3SMfdaj75kEoC",
            },
            // Fortify
            level3: {
                flame: "blue",
            },
            // Search
            level4: {
                flame: "red",
                bell: "AWABWD4IRDuBeYWC",
            },
            // Evacuate
            level5: {
                flame: "red",
                bell: "7awl4TXtocGJXaOF",
            }
        };
        const state = await foundry.applications.api.DialogV2.prompt({
            window: { title: "Awareness States" },
            classes: ["pf2e-claws"],
            content: `<h3>Alarm Levels</h3>
            <p>There are several states of awareness that the keep can be in, depending on a combination of values. The current state is indicated by the colour of <em>Iomedae's Flame</em>. The castle's defenders use bells to coordinate.</p>
            <p>If the keep's defenders (or your players) change their tactics, you can use this macro to play the appropriate signal through the bells, as well as to adjust the colour of <em>Iomedae's Flame</em> accordingly.</p>
            <table>
            <thead>
            <tr>
            <th></th><th>Awareness</th><th>Behaviour</th><th>Flame</th><th>Bells</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td><input type="radio" id="off" name="alarm" value="off"/></td><td></td><td>Flame Disabled.</td><td>None</td><td></td>
            </tr>
            <tr>
            <td><input type="radio" id="level0" name="alarm" value="level0" default="true"/></td><td>0-3</td><td>All Clear.</td><td>White</td><td>2x Small, 2x Large, 2x Small</td>
            </tr>
            <tr>
            <td><input type="radio" id="level1" name="alarm" value="level1"/></td><td>4-7</td><td>Possible Intruders. Travel in pairs.</td><td>White</td><td>4x Small</td>
            </tr>
            <tr>
            <td><input type="radio" id="level2" name="alarm" value="level2"/></td><td>8-11</td><td>Intruders Suspected. Increase patrols.</td><td>Yellow</td><td>1x Large, 3x Small</td>
            </tr>
            <tr>
            <td><input type="radio" id="level3" name="alarm" value="level3"/></td><td>12-15</td><td>All defenders relocate to the stronghold.</td><td>Blue</td><td>N/A</td>
            </tr>
            <tr>
            <td><input type="radio" id="level4" name="alarm" value="level4"/></td><td>16-20</td><td>Chalice at Risk. Initiate room-to-room search.</td><td>Red</td><td>2x Large, 2x Small</td>
            </tr>
            <tr>
            <td><input type="radio" id="level5" name="alarm" value="level5"/></td><td>21</td><td>Intruders Confirmed. Evacuate Fort.</td><td>Red</td><td>All 6x</td>
            </tr>
            <tbody>
            </table>
            `,
            ok: {
                action: "confirm",
                label: "Confirm",
                default: true,
                callback: (event, button, dialog) => button.form.elements.alarm.value
            }
            /*
            <p>Since the players might have disabled the bells or the flames, you can also adjust either of these alone using the alternate buttons provided.</p>
            buttons: [
                {
                    actions: "flame-only",
                    label: "Flame Only"
                },
                {
                    actions: "bell-only",
                    label: "Bells Only",
                }
            ]
            */
        });
        // Play sound
        if (awarenessLevels[state].bell) {
            const sound = await playlist.sounds.get(awarenessLevels[state].bell);
            if (sound != undefined) { await playlist.playSound(sound); }
            sleep(1000);
        }
        // Disable all the lights and tiles
        for ( const [key, id] of Object.entries(lights) ) {
            const light = canvas.scene.lights.get(id);
            if (light != undefined) { await light.update({ hidden: (key !== awarenessLevels[state].flame) });}        }
        for ( const [key, id] of Object.entries(tiles) ) {
            const tile = canvas.scene.tiles.get(id);
            if (tile != undefined) { await tile.update({ hidden: (key !== awarenessLevels[state].flame) });}        }
    },
    async A05TallusianAmbush() {
      const tile = canvas.scene.tiles.get("vtiW9heQSNyMqSDm");
      const token1 = canvas.scene.tokens.get("KK1BYRaoabgNDqSO");
      const token2 = canvas.scene.tokens.get("4f4GZPlm2PH5Sahm");
      const token3 = canvas.scene.tokens.get("QVxgEFLYTns8AxDc");
      await togglePlaceables([tile, token1, token2, token3]);
    },
    // 3.1
    // 3.2
    async A03BridgeCollapse() {
      const tileIds = [
        "HpeCW0AJooJhyQY6",
        "cNnkiatmYiwiNEY2",
        "16D37V3EivBOq4zb",
        "WEQClp100xGM2q3t",
        "Zd84MckDuldEXMi4"
      ];
      const tiles = tileIds.map(id => canvas.scene.tiles.get(id));
      let currentIndex = tiles.findIndex(tile => !tile.hidden);
      let nextIndex = (currentIndex + 1) % tiles.length;
      for (let tile of tiles) {
        await tile.update({ hidden: true, sort: 0 });
      }
      await tiles[nextIndex].update({ hidden: false, sort: 5 });
    },
    async A04HardenedEntry() {
      const door = canvas.scene.walls.get("NsPhg0mIJ8rXu68f");
      const tile = canvas.scene.tiles.get("Ep7bqwtYjnqUnsGl");
      await togglePlaceables([tile]);
      await toggleDoor(door);
    },
    async A15IomedaeWarden() {
      const tile = canvas.scene.tiles.get("gmLzSIs4Gx6mrZpm");
      const token = canvas.scene.tokens.get("a4cxmfgayA5txYQj");
      await togglePlaceables([tile, token]);
    },
    async A17ArazniWardens() {
      const tile = canvas.scene.tiles.get("WjVPPehDXAgJaMBo");
      const token1 = canvas.scene.tokens.get("6jkNamrUBrboUxqJ");
      const token2 = canvas.scene.tokens.get("uyo0Mn5DwF5gkWgh");
      const token3 = canvas.scene.tokens.get("OeB9wMBe7ny262cb");
      const token4 = canvas.scene.tokens.get("6FUdaC85pwnNAYhI");
      await togglePlaceables([tile, token1, token2, token3, token4]);
    },
    async A18CleansingSuite() {
      const light = canvas.lighting.get("Rv1kXZuKZdKgVDwY");
      const doorIds = [
        "2Ia32UXfijcDW5kd",
        "IEhKPcjRNiGLUCL5",
        "YxTBXmMQR6z8xOJa"
      ];
      const doors = doorIds.map(id => canvas.walls.get(id));
      const turningOn = light.document.hidden;
      // Toggle light visibility
      await light.document.update({ hidden: !turningOn });
      // Toggle doors
      for (let wall of doors) {
        await wall.document.update({
          door: 1,                   // Make sure it's a *regular* door
          ds: turningOn ? 2 : 0      // 2 = locked/closed, 0 = open/unlocked
        });
      }
    },
    // 3.3
    async D02FleshWallWest() {
      const door = canvas.scene.walls.get("FhczYz2581eoxWTn");
      const tile = canvas.scene.tiles.get("fh7x9yA6r7EFU5Zs");
      // Sound & Sleep?
      await togglePlaceables([tile]);
      await toggleDoor(door);
    },
    async D02FleshWallEast() {
      const door = canvas.scene.walls.get("w6pVxln7wGluLaaS");
      const tile = canvas.scene.tiles.get("JgGWCGyPac5hmLqr");
      // Sound & Sleep?
      await togglePlaceables([tile]);
      await toggleDoor(door);
    },
    async D05RitualProgression() {
      const tileIds = [
        "lvyQEnhYu0iyHEWW",
        "OWLMsUIJNBp1fIGN",
        "UDDkK2FjUuoVooIF",
        "lDxqmlH29Be3PuqC",
        "t4vyTcCyiJ5ZVf3t",
        "aNLH4sV4BsS6ZXIw",
        "xun8m2emqcxPJ99J",
        "0ENL0eVZBed4e5eO",
        "s6R0deOPQBD3jaZ1",
        "UjDnV1ANtmd3bDbw",
        "2b9I4eehTDppd9yH"
      ];
      const lightIds = [
        "P39Xqlk67yvDZRlt",
        "uZjgNcrySGWfO09h",
        "lTtOyiNUXfOCX83I",
        "zGhoXZb8HaAoj0ns",
        "08dLpcjNWPJVpGoJ",
        "tvWSnJi2HzIb67Fg",
        "I4vELpG3zsOvfEYt",
        "fcbien93AyfTewit",
        "IDO47MN36wa2UbBK",
        "69rtxHJAhOHVHZCZ",
        "ocZznTn0KG4UfMIV",
      ];
      const tiles = tileIds.map(id => canvas.scene.tiles.get(id));
      const lights = lightIds.map(id => canvas.lighting.get(id));
      // Count how many tiles are currently visible
      let visibleCount = tiles.filter(tile => !tile.hidden).length;
      console.debug(visibleCount);
      if (visibleCount >= tiles.length) {
        // Reset everything if all are already visible
        for (let i = 0; i < tiles.length; i++) {
          tiles[i].update({ hidden: true });
          lights[i].document.update({ hidden: true });
        }
      } else {
        // Reveal the next tile and light in sequence
        tiles[visibleCount].update({ hidden: false });
        console.debug(lights[visibleCount]);
        lights[visibleCount].document.update({ hidden: false });
      }
    }
};const MODULE_ID = "pf2e-claws-of-the-tyrant";
const ADVENTURE_UUID = "Compendium.pf2e-claws-of-the-tyrant.claws-of-the-tyrant.Adventure.5oLiLaXFkwdn9428";

/**
 * A CSS added to core applications displaying module content
 */
const CSS_CLASS = "pf2e-claws";/**
 * @typedef {Object} LocalizationData
 * @property {Set<string>} html       HTML files which provide Journal Entry page translations
 * @property {object} i18n            An object of localization keys and translation strings
 */

/**
 * A subclass of the core AdventureImporter which performs some special functions for the Claws of the Tyrant module.
 */
class ClawsAdventureImporter extends foundry.applications.sheets.AdventureImporter {
  constructor(doc, options) {
    super(doc, options);
    this.options.classes.push(CSS_CLASS);
  }

  /** @override */
  async _preImport({toCreate, toUpdate}) {
    // Merge Compendium Actor data
    if ("Actor" in toCreate) await this.#readyActors(toCreate.Actor);
    if ("Actor" in toUpdate) await this.#readyActors(toUpdate.Actor);
  }

  /* -------------------------------------------- */
  /*  Pre-Import Customizations                   */
  /* -------------------------------------------- */

  /**
   * Merge Actor data with authoritative source data from system compendium packs, make followup adjustments as needed
   * for the adventure.
   * @param {ActorData[]} actors Actor data from the Adventure document
   * @returns {Promise<void>}
   */
  async #readyActors(actors) {
    for (const actor of actors) {
      const pack = game.packs.get(actor._stats.compendiumSource);
      const fromPack = await pack?.getDocument(actor._id);
      if ( fromPack ) {
        const sourceData = fromPack.toObject();
        foundry.utils.mergeObject(actor, {
          system: sourceData.system,
          items: sourceData.items,
          effects: sourceData.effects,
          rules: sourceData.rules
        });
      }
      try {
        this.#modifyActor(actor);
      }
      catch (error) {
        console.warn(`Failed making modifications to ${actor.name} (${actor._id}): ${error.message}`);
      }
    }
  }

  /**
   * Modify certain actors for the adventure.
   * @param {ActorData} source
   */
  #modifyActor(source) {
    switch ( source._id ) {
      case "c3AvFxqVXyTH4Ux7": // Yeast Ooze
        source.system.attributes.adjustment = "elite";
        source.system.attributes.hp.value = 75;
        break;
      case "MF8m47hlaBWYBH84":
        source.name = "Goblin Skeleton";
        source.system.traits.size.value = "sm";
        break;
      case "gw9NY9aw1Q9k8b6r": // Zombie Hulk
        source.system.attributes.adjustment = "weak";
        source.system.attributes.hp.value = 140;
        break;
      case "mXNXQHNJpYyYCMzl": // Shevarna
        source.system.skills.medicine.mod = 18;
        break;
    }
  }
}/**
 * The custom Journal Sheet used for Claws content.
 */
class ClawsJournalSheet extends foundry.appv1.sheets.JournalSheet {
  constructor(doc, options) {
    super(doc, options);
    this.options.classes.push(CSS_CLASS);
  }
}Hooks.once("init", () => {
    // Create global reference to module
    globalThis.claws = game.modules.get(MODULE_ID);
    claws.macros = MACROS;

    // Register sheets
    foundry.applications.apps.DocumentSheetConfig.registerSheet(JournalEntry, MODULE_ID, ClawsJournalSheet, {
        types: ["base"],
        label: "Claws of the Tyrant",
        makeDefault: false,
        canBeDefault: false,
        canConfigure: true
    });
    DocumentSheetConfig.registerSheet(Adventure, MODULE_ID, ClawsAdventureImporter, {
      label: "Claws of the Tyrant Importer",
      makeDefault: false,
      canBeDefault: false,
      canConfigure: true
    });

    // Allow disabling of the "send to chat" button
    game.settings.register(MODULE_ID, "descriptiveTextButton", {
        name: "\"Send To Chat\" button",
        hint: "Adds a button to the Claws of the Tyrant journals that can be used to send the \"readaloud\" blocks of descriptive text as a chat message.",
        scope: "client",
        config: true,
        type: Boolean,
        default: true
    });
});

Hooks.once("ready", async () => {

    // Launch the Adventure importer if first startup
    const imported = !!game.settings.get("core", "adventureImports")?.[ADVENTURE_UUID];
    if ( !imported && game.user.isGM ) {
        const adventure = await fromUuid(ADVENTURE_UUID);
        await adventure.sheet.render({force: true});
    }
});

/* -------------------------------------------- */
/*  Journals Rendering                          */
/* -------------------------------------------- */

// The listener that adds that functionality
Hooks.on("renderJournalEntryPageSheet", (app, html, doc, event) => {
  if ( !game.settings.get(MODULE_ID, "descriptiveTextButton") ) return;
  // Check if a button is needed and create if so
  const descriptions = app.element.querySelectorAll("section.description:not(.readout)");
  for ( const description of descriptions ) {
    description.classList.add("readout");
    const readoutButton = document.createElement("button");
    readoutButton.className = "icon plain fa-regular fa-comment-alt readout";
    readoutButton.dataset.tooltip = "";
    readoutButton.ariaLabel = "Send To Chat"; //Should localise this
    readoutButton.addEventListener("click", () => {
      ChatMessage.create({
          flavor: `<div class="page" data-visibility="gm"><p>@UUID[${doc.uuid}]</p></div>`,
          content: description.outerHTML,
          speaker: {alias: "Description"},
      });
    });
    description.prepend(readoutButton);
  }
});