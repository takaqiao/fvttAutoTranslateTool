import fs from "node:fs/promises"

/**
 * Derives the two compendium macro commands from macro.js.
 *
 * localize-pack.mjs writes these into the pack and verify-pack.mjs asserts the pack still
 * matches them, so both sides share this one derivation. If they used separate copies, an
 * edit to macro.js that was never written to the pack would pass verification — which is
 * exactly how the pack came to ship a macro that rendered "undefined" to the GM.
 */
export async function buildMacroCommands(root = new URL("../", import.meta.url)) {
  const source = await fs.readFile(new URL("macro.js", root), "utf8")
  const helperEnd = source.indexOf("async function openForAll()")
  if (helperEnd < 0) throw new Error("macro.js no longer defines openForAll()")
  const helper = source.slice(0, helperEnd).trim()

  const functionBody = (name, endMarker) => {
    const marker = `async function ${name}() {`
    const start = source.indexOf(marker)
    if (start < 0) throw new Error(`Could not find ${name} in macro.js`)
    const end = endMarker ? source.indexOf(endMarker, start) : source.length
    if (end < 0) throw new Error(`Could not find the end of ${name} in macro.js`)
    const block = source.slice(start + marker.length, end).trimEnd()
    const finalBrace = block.lastIndexOf("}")
    if (finalBrace < 0) throw new Error(`Could not find the closing brace for ${name}`)
    return `${helper}\n\n${block.slice(0, finalBrace).trim()}\n`
  }

  return {
    all: functionBody("openForAll", "async function openForOne()"),
    one: functionBody("openForOne"),
  }
}

/** Maps a stored macro's name to the command it should hold. */
export function commandFor(macroName, commands) {
  if (macroName.includes("all users")) return commands.all
  if (macroName.includes("specific user")) return commands.one
  return null
}
