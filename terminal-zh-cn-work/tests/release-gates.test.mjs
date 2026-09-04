// Gates added after the pre-release audit. Each one closes a class of defect that actually
// shipped once, so a green run means something a reviewer could not check by eye.
import assert from "node:assert/strict"
import fs from "node:fs"
import path from "node:path"
import test from "node:test"
import { fileURLToPath } from "node:url"

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")

const sourceFiles = () => [
  "macro.js",
  ...fs.readdirSync(path.join(root, "scripts")).filter(file => file.endsWith(".js")).map(file => path.join("scripts", file)),
]

const readCatalog = name => JSON.parse(fs.readFileSync(path.join(root, "lang", name), "utf8"))

const FOUNDRY = "file:///C:/Program%20Files/Foundry%20Virtual%20Tabletop/resources/app/node_modules"

test("the Chinese catalog is registered under the language code Foundry's zh ecosystem uses", () => {
  // Foundry matches the declared lang against the active one exactly. Every Simplified Chinese
  // package in the wild (foundry_chn, pf2_cn, crucible-cn, ...) registers "cn", so a catalog
  // declared as "zh-CN" is never loaded and the whole localization stays silently inert.
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "module.json"), "utf8"))
  const codes = manifest.languages.map(entry => entry.lang)

  assert.ok(codes.includes("cn"), `expected a "cn" catalog, got ${JSON.stringify(codes)}`)
  assert.ok(!codes.includes("zh-CN"), '"zh-CN" is not the code Foundry Chinese packages register')

  const i18n = fs.readFileSync(path.join(root, "scripts", "i18n.js"), "utf8")
  assert.match(i18n, /lang === "cn"/, "isSimplifiedChinese must test the code module.json declares")
})

// Reads the argument list of the call whose name starts at `callStart`, with quote and bracket
// awareness so multi-line calls, template literals and nested calls are all handled.
const callArguments = (source, callStart) => {
  const open = source.indexOf("(", callStart)
  if (open < 0) return null
  let depth = 0
  let quote = null
  for (let i = open; i < source.length; i++) {
    const char = source[i]
    if (quote) {
      if (char === "\\") i++
      else if (char === quote) quote = null
      continue
    }
    if (char === '"' || char === "'" || char === "`") { quote = char; continue }
    if ("({[".includes(char)) depth++
    else if (")}]".includes(char)) {
      depth--
      if (depth === 0) return source.slice(open + 1, i)
    }
  }
  return null
}

// Splits an object literal body on its TOP-LEVEL commas and returns the property names. Splitting
// rather than pattern-matching matters: a regex that consumes the separating comma silently skips
// every other property.
const objectKeys = body => {
  const inner = body.slice(body.indexOf("{") + 1, body.lastIndexOf("}"))
  const parts = []
  let depth = 0
  let quote = null
  let current = ""
  for (let i = 0; i < inner.length; i++) {
    const char = inner[i]
    if (quote) {
      current += char
      if (char === "\\") { current += inner[++i] ?? "" }
      else if (char === quote) quote = null
      continue
    }
    if (char === '"' || char === "'" || char === "`") { quote = char; current += char; continue }
    if ("({[".includes(char)) depth++
    else if (")}]".includes(char)) depth--
    if (char === "," && depth === 0) { parts.push(current); current = ""; continue }
    current += char
  }
  parts.push(current)

  const keys = new Set()
  for (const part of parts) {
    const match = part.match(/^\s*(?:\.\.\.)?([A-Za-z_$][\w$]*)\s*(?::|$)/)
    if (match) keys.add(match[1])
  }
  return keys
}

test("every translate call supplies the placeholders its catalog entry declares", () => {
  // A call that omits a placeholder renders the literal string "undefined" in that slot, because
  // Foundry substitutes data[name] without checking. Nothing warns about it at runtime.
  const english = readCatalog("en.json")
  const problems = []

  for (const file of sourceFiles()) {
    const source = fs.readFileSync(path.join(root, file), "utf8")
    for (const match of source.matchAll(/\b(?:t|tr|terminalT)\(\s*["'](TERMINAL\.[\w.]+)["']\s*(,?)/g)) {
      const key = match[1]
      const value = english[key]
      if (value === undefined) continue
      const declared = [...value.matchAll(/\{([^}]+)\}/g)].map(m => m[1])
      if (!declared.length) continue
      if (!match[2]) {
        problems.push(`${file}: ${key} declares {${declared}} but is called with no data`)
        continue
      }
      const args = callArguments(source, match.index)
      if (args === null || !args.includes("{")) continue
      const supplied = objectKeys(args)
      for (const name of declared) {
        if (!supplied.has(name)) problems.push(`${file}: ${key} declares {${name}} but the call does not supply it`)
      }
    }
  }
  assert.deepEqual(problems, [])
})

test("socket payload keys resolve against both catalogs", () => {
  // Foundry's language setting is per client, so what crosses game.socket must be a key the
  // RECEIVER renders. A key that exists on the sender but not in the catalog reaches the
  // recipient as a raw TERMINAL.* string.
  const english = readCatalog("en.json")
  const chinese = readCatalog("cn.json")
  const missing = []
  const pattern = /(?:message|error|description)Key:\s*(?:[\w.?]+\s*\?\s*)?["'](TERMINAL\.[\w.]+)["'](?:\s*:\s*["'](TERMINAL\.[\w.]+)["'])?/g

  for (const file of sourceFiles()) {
    const source = fs.readFileSync(path.join(root, file), "utf8")
    for (const match of source.matchAll(pattern)) {
      for (const key of [match[1], match[2]].filter(Boolean)) {
        if (!(key in english)) missing.push(`${file}: ${key} missing from en.json`)
        if (!(key in chinese)) missing.push(`${file}: ${key} missing from cn.json`)
      }
    }
  }
  assert.deepEqual(missing, [])
})

test("no socket payload carries a sentence rendered by the sender", () => {
  // Foundry's language is a per-client setting, so a sentence rendered by the sender arrives in
  // the sender's language. Only payloads that actually cross game.socket are in scope here.
  const offenders = []
  for (const file of sourceFiles()) {
    const source = fs.readFileSync(path.join(root, file), "utf8")
    for (const match of source.matchAll(/game\.socket\.emit\(/g)) {
      const payload = callArguments(source, match.index + "game.socket".length)
      if (payload === null) continue
      for (const hit of payload.matchAll(/\b(message|error|description):\s*(?:tr?|terminalT)\(/g)) {
        const line = source.slice(0, match.index).split("\n").length
        offenders.push(`${file}:${line} ${hit[1]} is rendered on the sending client`)
      }
    }
  }
  assert.deepEqual(offenders, [])
})

test("every Handlebars template compiles, in both languages", async () => {
  const { default: Handlebars } = await import(`${FOUNDRY}/handlebars/lib/index.js`)
  let compiled = 0
  for (const dir of [path.join(root, "templates"), path.join(root, "templates", "cn")]) {
    for (const file of fs.readdirSync(dir).filter(name => name.endsWith(".hbs"))) {
      const source = fs.readFileSync(path.join(dir, file), "utf8")
      assert.doesNotThrow(() => Handlebars.precompile(source), path.relative(root, path.join(dir, file)))
      compiled++
    }
  }
  assert.equal(compiled, 28)
})

test("each template pair is structurally identical", () => {
  // Only visible text may differ. A drifted form field name or Handlebars expression breaks the
  // dialog's submit handler for Chinese clients alone, which is the kind of bug nobody notices.
  const shape = source => ({
    expressions: [...source.matchAll(/\{\{[^}]*\}\}/g)]
      .map(match => match[0])
      .filter(expression => !expression.includes("templates/")),
    tags: [...source.matchAll(/<([a-zA-Z][\w]*)/g)].map(match => match[1]).sort(),
    fields: [...source.matchAll(/\bname="([^"]+)"/g)].map(match => match[1]).sort(),
    actions: [...source.matchAll(/\bdata-action="([^"]+)"/g)].map(match => match[1]).sort(),
  })

  for (const file of fs.readdirSync(path.join(root, "templates")).filter(name => name.endsWith(".hbs"))) {
    const english = shape(fs.readFileSync(path.join(root, "templates", file), "utf8"))
    const chinese = shape(fs.readFileSync(path.join(root, "templates", "cn", file), "utf8"))
    assert.deepEqual(chinese.expressions, english.expressions, `${file}: Handlebars expressions differ`)
    assert.deepEqual(chinese.tags, english.tags, `${file}: HTML tags differ`)
    assert.deepEqual(chinese.fields, english.fields, `${file}: form field names differ`)
    assert.deepEqual(chinese.actions, english.actions, `${file}: data-action values differ`)
  }
})

test("a non-Chinese client gets the English templates and the English catalog", async () => {
  const { isSimplifiedChinese, templatePath, t } = await import("../scripts/i18n.js")
  const english = readCatalog("en.json")

  globalThis.game = {
    i18n: { lang: "en", localize: key => english[key] ?? key, format: key => english[key] ?? key },
  }
  try {
    assert.equal(isSimplifiedChinese(), false)
    assert.equal(templatePath("config.hbs"), "modules/terminal/templates/config.hbs")
    assert.equal(t("TERMINAL.Title"), "Terminal")
  } finally {
    delete globalThis.game
  }

  // And with no game object at all — the state during ES module evaluation — nothing may throw.
  assert.equal(isSimplifiedChinese(), false)
  assert.equal(templatePath("config.hbs"), "modules/terminal/templates/config.hbs")
})

test("the shipped compendium pack matches macro.js", async () => {
  // The pack is what users import; macro.js is only its source. They drifted apart once and the
  // pack shipped a macro that printed "undefined" to the GM.
  const { ClassicLevel } = await import(`${FOUNDRY}/classic-level/index.js`)
  const { buildMacroCommands, commandFor } = await import("../tools/macro-commands.mjs")

  const database = new ClassicLevel(path.join(root, "packs", "terminal-macros"), {
    keyEncoding: "utf8",
    valueEncoding: "json",
  })
  await database.open()
  const macros = []
  try {
    for await (const [, value] of database.iterator()) macros.push(value)
  } finally {
    await database.close()
  }

  const commands = await buildMacroCommands(new URL("../", import.meta.url))
  assert.equal(macros.length, 2)
  for (const macro of macros) {
    const expected = commandFor(macro.name, commands)
    assert.ok(expected, `unrecognised macro in pack: ${macro.name}`)
    assert.equal(macro.command, expected, `"${macro.name}" has drifted — run tools/localize-pack.mjs`)
  }
})
