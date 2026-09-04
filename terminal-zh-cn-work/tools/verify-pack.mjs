import assert from "node:assert/strict"
import { ClassicLevel } from "file:///C:/Program%20Files/Foundry%20Virtual%20Tabletop/resources/app/node_modules/classic-level/index.js"
import { buildMacroCommands, commandFor } from "./macro-commands.mjs"

const databasePath = process.argv[2] ?? new URL("../packs/terminal-macros", import.meta.url).pathname.slice(1)
const database = new ClassicLevel(databasePath, { keyEncoding: "utf8", valueEncoding: "json" })
await database.open()

const macros = []
for await (const [, value] of database.iterator()) macros.push(value)
await database.close()

const commands = await buildMacroCommands()

assert.equal(macros.length, 2, "expected exactly two macros in the pack")
for (const macro of macros) {
  const expected = commandFor(macro.name, commands)
  assert.ok(expected, `pack holds an unrecognised macro: ${macro.name}`)

  // Byte equality against macro.js. This is the assertion that matters: any edit to macro.js
  // that has not been written back to the pack fails here, rather than shipping silently.
  assert.equal(macro.command, expected, `pack macro "${macro.name}" has drifted from macro.js — run tools/localize-pack.mjs`)

  // Cross-client rule: a socket payload must carry a key, never a sentence rendered by the sender.
  assert.doesNotMatch(
    macro.command,
    /message:\s*terminalT\(/,
    `pack macro "${macro.name}" pre-renders a socket notification instead of sending messageKey/messageData`,
  )
}

console.log(`Verified ${macros.length} compendium macros byte-match macro.js.`)
