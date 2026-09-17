import test from 'node:test';
import assert from 'node:assert/strict';
import {createGlimpseReactionCache} from '../scripts/glimpse-reaction-cache.mjs';
const SHA='4a81322796ce1c6ed545edc09e1aa3a96a9c8a96dfd034403bf657068ed7036c',SLUG='glimpse-of-redemption';
function fixture({version='1.4.3',active=true,sha=SHA}={}){
 const installed={active,version},registrations=[],flags={'pf2e-reaction':{availableReactions:['shield-block'],unrelated:42}};
 const combat={id:'battle',started:true,turns:[{actor:{type:'npc',items:[{type:'action',slug:SLUG}]},flags:{'pf2e-reaction':{state:false}}}],flags,update:()=>{throw Error('compat must never write a combat')}};
 const game={user:{id:'gm'},users:{activeGM:{id:'gm'}},modules:new Map([['pf2e-reaction',installed]]),combats:new Map([[combat.id,combat]]),settings:{get:()=>[SLUG,'shield-block']}};
 const libWrapper={register:(module,path,wrapper,type)=>registrations.push({module,path,wrapper,type}),unregister:(module,path)=>{registrations.splice(registrations.findIndex(r=>r.module===module&&r.path===path),1)}};
 const adapter=createGlimpseReactionCache({game,fetchSource:async()=>'verified installed source',hashSource:async()=>sha});
 const get=(ns='pf2e-reaction',key='availableReactions',...extra)=>{assert.equal(registrations.length,1,'one getFlag wrapper must be registered');return registrations[0].wrapper.call(combat,(...args)=>{assert.deepEqual(args,[ns,key,...extra]);return flags[ns]?.[key];},ns,key,...extra)};
 return{game,combat,flags,installed,registrations,libWrapper,adapter,get};
}
test('verified initialization installs exactly one wrapper; repeated initialize and unregister are safe',async()=>{
 const f=fixture();assert.equal(f.adapter.ready(),false);assert.equal(await f.adapter.initialize({libWrapper:f.libWrapper}),true);assert.equal(f.adapter.ready(),true);
 await f.adapter.initialize({libWrapper:f.libWrapper});assert.equal(f.registrations.length,1);assert.equal(f.registrations[0].path,'CONFIG.Combat.documentClass.prototype.getFlag');assert.equal(f.registrations[0].type,'WRAPPER');
 f.adapter.unregister();assert.equal(f.registrations.length,0);assert.equal(f.adapter.ready(),false);
});
test('unsupported or mismatched modules and absent wrapper never become ready or register',async()=>{
 for(const config of [{active:false},{version:'1.4.4'},{sha:'wrong'}]){const f=fixture(config);assert.equal(await f.adapter.initialize({libWrapper:f.libWrapper}),false);assert.equal(f.adapter.ready(),false);assert.equal(f.registrations.length,0);}
 const f=fixture();assert.equal(await f.adapter.initialize({libWrapper:null}),false);assert.equal(f.adapter.ready(),false);
});
test('eligible reads add only a virtual member without mutating native cache or reaction resources',async()=>{
 const f=fixture();await f.adapter.initialize({libWrapper:f.libWrapper});const native=f.flags['pf2e-reaction'].availableReactions,result=f.get();
 assert.deepEqual(result,['shield-block',SLUG]);assert.notEqual(result,native);assert.deepEqual(native,['shield-block']);
 result.push('caller-only');assert.deepEqual(f.get(),['shield-block',SLUG]);assert.equal(f.combat.turns[0].flags['pf2e-reaction'].state,false);
});
test('concurrent native cache additions survive every effective read; adapter writes nothing',async()=>{
 const f=fixture();await f.adapter.initialize({libWrapper:f.libWrapper});
 const old=f.get();f.flags['pf2e-reaction'].availableReactions.push('reactive-strike');
 assert.deepEqual(old,['shield-block',SLUG]);assert.deepEqual(f.get(),['shield-block','reactive-strike',SLUG]);
 const next=f.get();next.push('reactive-shield');f.flags['pf2e-reaction'].availableReactions=next;
 assert.equal(f.get(),next);assert.deepEqual(next,['shield-block','reactive-strike',SLUG,'reactive-shield']);
});
test('unknown namespace, keys, extra arguments and unknown return values preserve the native contract',async()=>{
 const f=fixture();await f.adapter.initialize({libWrapper:f.libWrapper});
 f.flags.other={availableReactions:{untouched:true}};assert.equal(f.get('other','availableReactions','default'),f.flags.other.availableReactions);assert.equal(f.get('pf2e-reaction','unrelated'),42);
 for(const native of [undefined,null,{},'unknown']){f.flags['pf2e-reaction'].availableReactions=native;assert.equal(f.get(),native);}
});
test('disabled setting, inactive combat, missing holder and changed module return native array unchanged',async()=>{
 for(const configure of [f=>f.game.settings.get=()=>[],f=>f.combat.started=false,f=>f.combat.turns=[],f=>f.combat.turns[0].actor.items=[{type:'feat',slug:SLUG}],f=>f.game.combats.delete('battle'),f=>f.installed.active=false,f=>f.installed.version='1.4.4',f=>f.game.modules.set('pf2e-reaction',{active:true,version:'1.4.3'})]){
  const f=fixture();await f.adapter.initialize({libWrapper:f.libWrapper});configure(f);assert.equal(f.get(),f.flags['pf2e-reaction'].availableReactions);
 }
});
test('uses actual Combat this and supports player-client native reads and synthetic holders',async()=>{
 const f=fixture();await f.adapter.initialize({libWrapper:f.libWrapper});f.game.user.id='player';
 f.combat.turns=[{actor:{isToken:true,itemTypes:{action:[{type:'action',system:{slug:SLUG}}]}}}];assert.deepEqual(f.get(),['shield-block',SLUG]);
 const foreign={...f.combat};const result=[];assert.equal(f.registrations[0].wrapper.call(foreign,()=>result,'pf2e-reaction','availableReactions'),result);
});
test('source identity changing while hashing cannot install a wrapper',async()=>{
 const f=fixture();const a=createGlimpseReactionCache({game:f.game,fetchSource:async()=>'',hashSource:async()=>{f.installed.version='1.4.4';return SHA;}});
 assert.equal(await a.initialize({libWrapper:f.libWrapper}),false);assert.equal(a.ready(),false);assert.equal(f.registrations.length,0);
});
test('concurrent initialization installs once, while cancellation before verification prevents registration',async()=>{
 const f=fixture();await Promise.all([f.adapter.initialize({libWrapper:f.libWrapper}),f.adapter.initialize({libWrapper:f.libWrapper})]);assert.equal(f.registrations.length,1);
 const next=fixture();let release;const a=createGlimpseReactionCache({game:next.game,fetchSource:()=>new Promise(resolve=>release=resolve),hashSource:async()=>SHA});
 const pending=a.initialize({libWrapper:next.libWrapper});a.unregister();release('source');assert.equal(await pending,false);assert.equal(next.registrations.length,0);
});
