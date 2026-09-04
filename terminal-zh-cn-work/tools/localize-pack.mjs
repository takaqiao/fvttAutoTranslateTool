import fs from "node:fs/promises"
import { buildMacroCommands } from "./macro-commands.mjs"
import { ClassicLevel } from "file:///C:/Program%20Files/Foundry%20Virtual%20Tabletop/resources/app/node_modules/classic-level/index.js"

const root = new URL("../", import.meta.url)
const commands = await buildMacroCommands(root)

const databasePath = new URL("packs/terminal-macros", root).pathname.slice(1)
const database = new ClassicLevel(databasePath, { keyEncoding: "utf8", valueEncoding: "json" })
await database.open()

let updated = 0
for await (const [key, value] of database.iterator()) {
  if (value.name.includes("all users")) value.command = commands.all
  else if (value.name.includes("specific user")) value.command = commands.one
  else continue
  await database.put(key, value)
  updated += 1
}

await database.close()
if (updated !== 2) {
  // This tool rewrites the commands of records that already exist; it cannot create the pack.
  // A fresh checkout has no packs/ directory (it is git-ignored), so seed it from the pristine
  // module before running this: unzip packs/terminal-macros from the 4.0.11 backup archive.
  throw new Error(
    `Expected to update 2 macros, updated ${updated}. ` +
    "Seed packs/terminal-macros from the original module archive first (see PROJECT.md).",
  )
}
console.log(`Localized ${updated} compendium macros.`)
