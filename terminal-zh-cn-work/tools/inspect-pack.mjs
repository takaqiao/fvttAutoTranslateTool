import { ClassicLevel } from "file:///C:/Program%20Files/Foundry%20Virtual%20Tabletop/resources/app/node_modules/classic-level/index.js"

const database = new ClassicLevel(new URL("../packs/terminal-macros", import.meta.url).pathname.slice(1), {
  keyEncoding: "utf8",
  valueEncoding: "json",
})

await database.open()
for await (const [key, value] of database.iterator()) {
  console.log(JSON.stringify({ key, value }, null, 2))
}
await database.close()
