import { ClassicLevel } from "classic-level";
const db = new ClassicLevel("C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs/crucible-character", {keyEncoding:"utf8", valueEncoding:"json"});
await db.open();
const want = new Set(["tyraphicTransformation","formidableStamina","implacableHunter","crystalizeWounds","inscrutableVisage"]);
for await (const [, doc] of db.iterator()) {
  const acts = doc?.system?.actions; if (!acts) continue;
  for ( const a of (Array.isArray(acts)?acts:Object.values(acts)) ) {
    if ( !want.has(a?.id) ) continue;
    for ( const e of (a.effects ?? []) )
      console.log(`${doc.name} | ${a.id} | duration = ${JSON.stringify(e.duration)}`);
  }
}
await db.close();
