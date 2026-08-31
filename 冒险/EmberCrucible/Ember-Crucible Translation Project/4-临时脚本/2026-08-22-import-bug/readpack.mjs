/**
 * 直读上游 LevelDB 包，看 `Potion of Climbing` 那条 item 的 `system.description` 到底是什么形状。
 * 这是判断「导入崩溃是不是我们造成的」唯一的硬证据 —— 英文基线里没有它，说明抽取器当时没抽到，
 * 但抽不到有两种可能（形状不对 / 通道没覆盖），只有直读能分清。
 */
import { pathToFileURL } from "node:url";
const { ClassicLevel } = await import(pathToFileURL("C:/Program Files/Foundry Virtual Tabletop/resources/app/node_modules/classic-level/index.js").href);
const PACK = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs/crucible-adventure";
const db = new ClassicLevel(PACK, { valueEncoding: "json" });
await db.open();
let found = 0;
for await (const [key, val] of db.iterator()) {
  const scan = (o, path) => {
    if (!o || typeof o !== "object") return;
    if (o.name === "Potion of Climbing" || o.name === "Bramble Bundle") {
      found++;
      const d = o.system?.description;
      console.log(`\n=== ${o.name}  (key=${key})  path=${path}`);
      console.log("  item.type      =", o.type);
      console.log("  description 形状 =", d === undefined ? "(无)" : (typeof d === "object" ? "object{" + Object.keys(d).join(",") + "}" : typeof d));
      if (d && typeof d === "object") for (const [k, v] of Object.entries(d))
        console.log(`     .${k} = ${typeof v === "string" ? JSON.stringify(v.slice(0, 60)) : typeof v}`);
      else if (typeof d === "string") console.log("     值 =", JSON.stringify(d.slice(0, 80)));
    }
    for (const [k, v] of Object.entries(o)) {
      if (Array.isArray(v)) v.forEach((x, i) => scan(x, `${path}.${k}[${i}]`));
      else if (v && typeof v === "object") scan(v, `${path}.${k}`);
    }
  };
  scan(val, key);
}
await db.close();
console.log(`\n命中 ${found} 处`);
