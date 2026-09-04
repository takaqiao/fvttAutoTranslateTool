/* Simplified Chinese localization helpers */

export function isSimplifiedChinese() {
  return globalThis.game?.i18n?.lang === "cn"
}

export function templatePath(file) {
  const locale = isSimplifiedChinese() ? "cn/" : ""
  return `modules/terminal/templates/${locale}${file}`
}

export function t(key, data) {
  const i18n = globalThis.game?.i18n
  if (!i18n) return key
  return data ? i18n.format(key, data) : i18n.localize(key)
}

export function configureLocalizedApplications(definitions) {
  for (const { application, template, title } of definitions) {
    if (template && application.PARTS?.form) {
      application.PARTS.form.template = templatePath(template)
    }
    if (title && application.DEFAULT_OPTIONS?.window) {
      application.DEFAULT_OPTIONS.window.title = t(title)
    }
  }
}

const actionKeys = {
  "lock or unlock door": "TERMINAL.ActionName.Door",
  "download map": "TERMINAL.ActionName.Map",
  "toggle power": "TERMINAL.ActionName.Power",
  "detect motion": "TERMINAL.ActionName.Motion",
  "observe Actor": "TERMINAL.ActionName.Observe",
  "start a secure shell": "TERMINAL.ActionName.SSH",
  "run a Macro": "TERMINAL.ActionName.Macro",
  "trigger a Region": "TERMINAL.ActionName.Region",
  "opening a skilled Terminal": "TERMINAL.ActionName.OpenTerminal",
  "decrypt a file": "TERMINAL.ActionName.Decrypt",
  "run script": "TERMINAL.ActionName.RunScript",
  "execute script": "TERMINAL.ActionName.ExecuteScript"
}

export function actionLabel(value) {
  const key = actionKeys[value]
  return key ? t(key) : value
}

// The tile config's link buttons label themselves with a `pretty` attribute, or fall back to the
// collection name upper-cased. Both forms were dropped verbatim into a translated sentence, so a
// Chinese GM read "点击一个ACTOR". The English values reproduce what upstream displayed.
const docTypeKeys = {
  ACTOR: "TERMINAL.DocType.Actor",
  ACTORS: "TERMINAL.DocType.Actors",
  DOOR: "TERMINAL.DocType.Door",
  ITEM: "TERMINAL.DocType.Item",
  ITEMS: "TERMINAL.DocType.Items",
  JOURNAL: "TERMINAL.DocType.Journal",
  MACROS: "TERMINAL.DocType.Macros",
  SCENE: "TERMINAL.DocType.Scene",
  SCENES: "TERMINAL.DocType.Scenes",
  TILE: "TERMINAL.DocType.Tile",
  TILES: "TERMINAL.DocType.Tiles",
  WALLS: "TERMINAL.DocType.Walls",
}

export function docTypeLabel(value) {
  const key = docTypeKeys[value]
  return key ? t(key) : value
}
