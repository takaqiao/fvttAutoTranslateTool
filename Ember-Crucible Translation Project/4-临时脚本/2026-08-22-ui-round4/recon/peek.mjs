import { pathToFileURL } from "node:url";
const { TEMPLATES } = await import(pathToFileURL("recon/_slice.mjs").href);
const partsOf = (l) => Array.isArray(l.parts) ? l.parts.map(p => typeof p === "string" ? p : p.id) : Object.keys(l.parts ?? {});
console.log("human 的图层:", Object.keys(TEMPLATES.human.layers).join(" "));
console.log("\nhuman.helm (前 12):", partsOf(TEMPLATES.human.layers.helm).slice(0, 12).join(" · "));
console.log("human.hair (前 12):", partsOf(TEMPLATES.human.layers.hair).slice(0, 12).join(" · "));
console.log("human.handItemR? ", Object.keys(TEMPLATES.human.layers).filter(k => /item|hand/i.test(k)).join(" "));
