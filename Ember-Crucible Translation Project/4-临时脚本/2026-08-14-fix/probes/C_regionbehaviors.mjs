import { createRequire } from 'module';
const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');
const base='C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs/';
const cbase='C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs/';
async function scan(dir,label){
  let db; try{ db=new ClassicLevel(dir,{createIfMissing:false}); }catch(e){ console.log('skip',dir); return; }
  const byType={};
  const soundNames=new Set(); let soundLeaves=0;
  for await (const [k,v] of db.iterator()){
    const key=k.toString();
    let doc; try{doc=JSON.parse(v.toString());}catch{continue;}
    const scenes = key.startsWith('!adventures!') ? (doc.scenes??[]) : (key.startsWith('!scenes!') ? [doc] : []);
    for(const s of scenes){
      for(const snd of s.sounds??[]){ if(typeof snd?.name==='string'&&snd.name.trim()){soundLeaves++;soundNames.add(snd.name);} }
      for(const r of s.regions??[]) for(const b of r.behaviors??[]){
        const t=b.type||'?';
        const e=(byType[t]??={count:0,keys:{},samples:{}});
        e.count++;
        const sys=b.system??{};
        for(const [kk,vv] of Object.entries(sys)){
          const kind = typeof vv;
          if(kind==='string'&&vv.trim()){ e.keys[kk]=(e.keys[kk]||0)+1; (e.samples[kk]??=new Set()).add(vv.slice(0,120)); }
          else if(Array.isArray(vv)&&vv.length&&vv.some(x=>x&&typeof x==='object'&&typeof x.name==='string')){
            e.keys[kk+'[].name']=(e.keys[kk+'[].name']||0)+ vv.filter(x=>typeof x?.name==='string').length;
            for(const x of vv) if(typeof x?.name==='string')(e.samples[kk+'[].name']??=new Set()).add(x.name);
          }
          else if(vv&&typeof vv==='object'){
            for(const [k2,v2] of Object.entries(vv)) if(typeof v2==='string'&&v2.trim()){ e.keys[kk+'.'+k2]=(e.keys[kk+'.'+k2]||0)+1; (e.samples[kk+'.'+k2]??=new Set()).add(v2.slice(0,120)); }
          }
        }
      }
    }
  }
  await db.close();
  console.log('=== '+label+' ===');
  console.log(' sounds[].name leaves='+soundLeaves+' uniq='+soundNames.size);
  for(const [t,e] of Object.entries(byType)){
    const interesting=Object.entries(e.keys);
    console.log(` type=${t} n=${e.count}`, interesting.length?JSON.stringify(Object.fromEntries(interesting)):'(no string system fields)');
    for(const [kk,ss] of Object.entries(e.samples)) console.log(`    ${kk}: ${[...ss].slice(0,4).map(x=>JSON.stringify(x)).join(' | ')}`);
  }
}
for(const p of ['adventure','crucible-adventure']) await scan(base+p, 'ember/'+p);
import fs from 'fs';
for(const p of fs.readdirSync(cbase)) await scan(cbase+p, 'crucible/'+p);
