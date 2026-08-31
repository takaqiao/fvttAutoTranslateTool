import { ClassicLevel } from "file:///C:/Program%20Files/Foundry%20Virtual%20Tabletop/resources/app/node_modules/classic-level/index.js";
import path from "node:path";
import fs from "node:fs";

const roots = [
  "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs",
  "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs"
];
const TARGET = new Set(["talent","spell","ancestry","archetype","background","taxonomy"]);
const PHYS = new Set(["accessory","armor","consumable","loot","schematic","tool","weapon"]);

for ( const root of roots ) {
  if ( !fs.existsSync(root) ) continue;
  for ( const p of fs.readdirSync(root) ) {
    const dir = path.join(root, p);
    if ( !fs.statSync(dir).isDirectory() ) continue;
    let db;
    try { db = new ClassicLevel(dir, {keyEncoding:"utf8", valueEncoding:"json"}); await db.open(); }
    catch(e) { console.log(`${p}: OPEN FAIL ${e.message}`); continue; }
    const counts = {};
    let strDesc = 0, objDesc = 0, embeddedTarget = 0, embeddedStr = 0;
    const bump = (o,k)=>o[k]=(o[k]??0)+1;
    const scanItem = (it, embedded) => {
      if ( !it || typeof it !== "object" ) return;
      const t = it.type;
      if ( !TARGET.has(t) ) return;
      if ( embedded ) embeddedTarget++;
      else bump(counts, t);
      const d = it.system?.description;
      if ( typeof d === "string" ) { strDesc++; if (embedded) embeddedStr++; }
      else if ( d && typeof d === "object" ) objDesc++;
    };
    const walkActor = (a) => { for ( const it of (a.items ?? []) ) scanItem(it, true); };
    for await ( const [k,v] of db.iterator() ) {
      if ( k.startsWith("!items") && !k.includes(".") ) scanItem(v,false);
      else if ( k.startsWith("!actors") && k.split("!")[1] === "actors" ) walkActor(v);
      else if ( k.startsWith("!adventures") ) {
        for ( const a of (v.actors ?? []) ) walkActor(a);
        for ( const it of (v.items ?? []) ) scanItem(it,false);
      }
      // embedded items stored separately
      if ( k.startsWith("!actors.items!") ) scanItem(v,true);
    }
    await db.close();
    const tot = Object.values(counts).reduce((a,b)=>a+b,0);
    if ( tot || embeddedTarget ) console.log(`${root.includes("ember")?"ember":"crucible"}/${p}: top-level=${JSON.stringify(counts)} (${tot}) embedded6=${embeddedTarget} strDesc=${strDesc} objDesc=${objDesc}`);
  }
}
