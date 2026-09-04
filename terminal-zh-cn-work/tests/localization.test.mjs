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

test("Foundry can discover the Simplified Chinese catalog", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "module.json"), "utf8"))
  const language = manifest.languages?.find(entry => entry.lang === "cn")

  assert.deepEqual(language, {
    lang: "cn",
    name: "简体中文",
    path: "lang/cn.json"
  })

  const catalog = JSON.parse(fs.readFileSync(path.join(root, language.path), "utf8"))
  assert.equal(catalog["TERMINAL.Title"], "终端")
  assert.equal(catalog["TERMINAL.Common.Save"], "保存")
  assert.equal(catalog["TERMINAL.Action.MapDownloaded"], "地图已下载")
  assert.match(catalog["TERMINAL.Error.NoGM"], /GM/)
})

test("the Chinese catalog has no blank translations", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(root, "lang", "cn.json"), "utf8"))
  const blanks = Object.entries(catalog).filter(([, value]) => typeof value !== "string" || value.trim() === "")
  assert.deepEqual(blanks, [])
})

test("English remains the fallback for localized runtime strings", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "module.json"), "utf8"))
  const language = manifest.languages?.find(entry => entry.lang === "en")

  assert.deepEqual(language, {
    lang: "en",
    name: "English",
    path: "lang/en.json"
  })

  const catalog = JSON.parse(fs.readFileSync(path.join(root, language.path), "utf8"))
  const chinese = JSON.parse(fs.readFileSync(path.join(root, "lang", "cn.json"), "utf8"))
  assert.equal(catalog["TERMINAL.Title"], "Terminal")
  assert.equal(catalog["TERMINAL.Common.Save"], "Save")
  assert.equal(catalog["TERMINAL.Action.MapDownloaded"], "Map Downloaded")
  assert.equal(catalog["TERMINAL.Error.NoGM"], "Terminal | No GM is connected, this module cannot work without one.")
  assert.deepEqual(Object.keys(catalog).sort(), Object.keys(chinese).sort())
})

test("application templates and titles can be localized after i18n initialization", async () => {
  const { configureLocalizedApplications } = await import("../scripts/i18n.js")
  const chinese = JSON.parse(fs.readFileSync(path.join(root, "lang", "cn.json"), "utf8"))

  globalThis.game = {
    i18n: {
      lang: "cn",
      localize: key => chinese[key] ?? key,
      format: key => chinese[key] ?? key,
    },
  }

  class FakeApplication {
    static DEFAULT_OPTIONS = { window: { title: "Rename Buttons" } }
    static PARTS = { form: { template: "modules/terminal/templates/rename.hbs" } }
  }

  configureLocalizedApplications([{
    application: FakeApplication,
    template: "rename.hbs",
    title: "TERMINAL.Window.Rename",
  }])

  try {
    assert.equal(FakeApplication.DEFAULT_OPTIONS.window.title, chinese["TERMINAL.Window.Rename"])
    assert.equal(FakeApplication.PARTS.form.template, "modules/terminal/templates/cn/rename.hbs")
  } finally {
    delete globalThis.game
  }
})

test("English and Chinese catalogs use the same interpolation placeholders", () => {
  const english = JSON.parse(fs.readFileSync(path.join(root, "lang", "en.json"), "utf8"))
  const chinese = JSON.parse(fs.readFileSync(path.join(root, "lang", "cn.json"), "utf8"))
  const placeholders = value => [...value.matchAll(/\{([^}]+)\}/g)].map(match => match[1]).sort()

  for (const key of Object.keys(english)) {
    assert.deepEqual(placeholders(chinese[key]), placeholders(english[key]), key)
  }
})

test("all referenced localization keys exist in both catalogs", () => {
  const english = JSON.parse(fs.readFileSync(path.join(root, "lang", "en.json"), "utf8"))
  const chinese = JSON.parse(fs.readFileSync(path.join(root, "lang", "cn.json"), "utf8"))
  const referenced = new Set()

  for (const file of sourceFiles()) {
    const source = fs.readFileSync(path.join(root, file), "utf8")
    for (const match of source.matchAll(/TERMINAL\.[A-Za-z0-9_.]+/g)) referenced.add(match[0])
  }

  assert.deepEqual([...referenced].filter(key => !(key in english)), [])
  assert.deepEqual([...referenced].filter(key => !(key in chinese)), [])
})

test("every English template has a Simplified Chinese counterpart", () => {
  const templates = fs.readdirSync(path.join(root, "templates")).filter(file => file.endsWith(".hbs")).sort()
  const chineseTemplates = fs.readdirSync(path.join(root, "templates", "cn")).filter(file => file.endsWith(".hbs")).sort()

  assert.deepEqual(chineseTemplates, templates)
  assert.match(
    fs.readFileSync(path.join(root, "templates", "cn", "terminal.hbs"), "utf8"),
    /templates\/cn\/terminal_cli\.hbs/,
  )
})

test("common interactive CLI messages are routed through localization", () => {
  const untranslated = [
    "Usage: cat <file>",
    "No such file or directory",
    "permission denied",
    "Operation not permitted",
    "All Batteries Charged",
    "no control daemons found",
    "no files found",
    "is already locked",
    "is already unlocked",
  ]

  for (const file of sourceFiles()) {
    const source = fs.readFileSync(path.join(root, file), "utf8")
    for (const text of untranslated) assert.doesNotMatch(source, new RegExp(text, "i"), `${file}: ${text}`)
  }
})
