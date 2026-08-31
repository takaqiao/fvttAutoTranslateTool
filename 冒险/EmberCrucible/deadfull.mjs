import { ClassicLevel } from "classic-level";
import { readFileSync } from "node:fs";
const NAMES=new Set(["Harrier","Opportunist","Opportunistic Spellcraft","Perspicacity","Safecracker","Versatile Translator","Aster"]);
const strip=h=>String(h??"").replace(/<[^>]*>/g," ").replace(/\s+/g," ").trim();
for ( const [root,pk] of [["C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs","talent"],
                          ["C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs","crucible-character"]] ) {
  const db=new ClassicLevel(`${root}/${pk}`,{keyEncoding:"utf8",valueEncoding:"json"}); await db.open();
  for await ( const [,doc] of db.iterator() ) {
    if ( doc?.type!=="talent" || !NAMES.has(doc.name) ) continue;
    const s=doc.system??{};
    console.log(`\n══ ${doc.name}  [${pk}]  _id=${doc._id}`);
    console.log(`   描述: ${strip(s.description)}`);
    console.log(`   system 的非空字段: ${Object.entries(s).filter(([k,v])=>v!=null&&!(Array.isArray(v)&&!v.length)&&!(typeof v==="object"&&v&&!Array.isArray(v)&&!Object.values(v).some(x=>x!=null&&x!==""&&!(Array.isArray(x)&&!x.length)))).map(([k])=>k).join(", ")}`);
    for ( const k of ["node","rune","gesture","inflection","training","actorHooks","advancement","requirements"] )
      if ( s[k]!=null && JSON.stringify(s[k])!=="{}" && JSON.stringify(s[k])!=="[]" )
        console.log(`     ${k} = ${JSON.stringify(s[k]).slice(0,140)}`);
  }
  await db.close();
}
