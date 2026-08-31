import { ClassicLevel } from "classic-level";
import { readFileSync, writeFileSync } from "node:fs";

// ── 谁在代码里注册了钩子（判断「有没有自动化」的唯一合法依据，见 P4 教训）
const hookIds = new Set();
for ( const f of ["C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/crucible-compiled.mjs",
                  "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs"] ) {
  const s = readFileSync(f, "utf8");
  for ( const m of s.matchAll(/HOOKS\$?\d*\.(\w+)\s*=\s*\{/g) ) hookIds.add(m[1]);
  for ( const m of s.matchAll(/HOOKS\[(?:id|`[^`]*`)\]/g) ) {}          // 动态注册，另计
}
console.log(`代码里注册了钩子的 id：${hookIds.size} 个`);

// ── 标签闭包（不求闭包会误报：natural→melee→strike 也提供 roll）
const PROP = { unarmed:["melee"], melee:["strike"], ranged:["strike"], mainhand:["strike"],
               twohand:["strike"], offhand:["strike"], thrown:["melee"], natural:["melee"], dualwield:["strike"] };
const ROLLERS = new Set(["generic","strike","spell","hazard","summon","skill","heal","restoration","rallying"]);
const SKILLS = new Set(["acrobatics","awareness","athletics","deception","diplomacy","intimidation","medicine",
  "perception","performance","stealth","survival","arcana","craftsmanship","investigation","lore","nature",
  "society","spellcraft","wilderness","bartering","beastmastery","navigation","piracy","politics","stagecraft","tradecraft"]);
const DEFENSES = new Set(["reflex","fortitude","willpower","physical"]);
const close = tags => { const o=new Set(tags); let g=true;
  while(g){ g=false; for(const t of [...o]) for(const p of (PROP[t]??[])) if(!o.has(p)){o.add(p);g=true;} } return o; };

// ── 只看玩家可及的来源
const P="C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs";
const E="C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs";
const PLAYER=[[P,"talent"],[P,"spell"],[P,"archetype"],[P,"ancestry"],[P,"background"],
              [P,"taxonomy"],[P,"summons"],[P,"equipment"],[P,"affixes"],[P,"pregens"],
              [E,"crucible-character"],[E,"crucible-items"],[E,"crucible-affixes"]];

const noRoll=[], deadTalent=[];
let acts=0, talents=0;
const seen=new Set();
for ( const [root,pk] of PLAYER ) {
  let db; try{db=new ClassicLevel(`${root}/${pk}`,{keyEncoding:"utf8",valueEncoding:"json"});await db.open();}catch{continue;}
  for await ( const [,doc] of db.iterator() ) {
    const visit=(it,owner)=>{
      const list=Array.isArray(it?.system?.actions)?it.system.actions:Object.values(it?.system?.actions??{});
      if(it?.type==="talent"){
        talents++;
        // 「死天赋」启发式：0 动作 + 0 效果 + 自己没注册钩子
        const noHook=!hookIds.has(it._id) && !hookIds.has(it.name?.replace(/\W/g,""));
        if(!list.length && noHook) deadTalent.push({pk,name:it.name,id:it._id,owner});
      }
      for(const a of list){
        acts++;
        const key=`${a.id}`; if(seen.has(key)) continue;
        const t=close(a.tags??[]);
        const declares=[...t].some(x=>DEFENSES.has(x));
        const hasRoll=[...t].some(x=>ROLLERS.has(x)||SKILLS.has(x));
        const hooked=hookIds.has(a.id);
        if(declares && !hasRoll && !hooked){ seen.add(key);
          noRoll.push({pk,owner,item:it.name,id:a.id,name:a.name,tags:a.tags,target:a.target?.type}); }
      }
    };
    visit(doc,"(顶层)");
    for(const it of (doc?.items??[])) visit(it,doc.name);
    for(const ac of (doc?.actors??[])) for(const it of (ac?.items??[])) visit(it,`${ac.type}:${ac.name}`);
  }
  await db.close();
}
writeFileSync("silent.json",JSON.stringify({noRoll,deadTalent},null,1),"utf8");
console.log(`玩家侧扫过 ${talents} 个天赋 / ${acts} 个动作`);
console.log(`\nA. 声明了攻击意图却没有任何掷骰实现（且代码里也没钩子）：${noRoll.length} 条`);
for(const r of noRoll) console.log(`   ${r.pk} | ${r.item} > ${r.name} (${r.id}) | tags=${JSON.stringify(r.tags)}`);
console.log(`\nB. 0 动作 0 钩子的天赋（多数是设计上的被动，仅供人工过目）：${deadTalent.length} 条`);
