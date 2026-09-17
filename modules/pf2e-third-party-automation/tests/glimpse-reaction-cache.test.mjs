import test from 'node:test';
import assert from 'node:assert/strict';
const module=await import('../scripts/glimpse-reaction-cache.mjs').catch(error=>{if(error.code==='ERR_MODULE_NOT_FOUND')return {};throw error;});
const SHA='4a81322796ce1c6ed545edc09e1aa3a96a9c8a96dfd034403bf657068ed7036c',SLUG='glimpse-of-redemption';
function fixture({version='1.4.3',active=true,sha=SHA}={}){
 assert.equal(typeof module.createGlimpseReactionCache,'function','verified cache adapter must exist');
 const installed={active,version},writes=[],flags={'pf2e-reaction':{availableReactions:['shield-block'],unrelated:42}};
 const combat={id:'battle',started:true,turns:[{actor:{type:'npc',items:[{type:'action',slug:SLUG}]},flags:{'pf2e-reaction':{state:false}}}],flags,getFlag:(m,k)=>flags[m]?.[k],update:async changes=>{writes.push(structuredClone(changes));flags['pf2e-reaction'].availableReactions=changes['flags.pf2e-reaction.availableReactions'];}};
 const game={user:{id:'gm'},users:{activeGM:{id:'gm'}},modules:new Map([['pf2e-reaction',installed]]),combats:new Map([[combat.id,combat]]),settings:{get:()=>[SLUG,'shield-block']}};
 let reads=0;const adapter=module.createGlimpseReactionCache({game,fetchSource:async()=>{reads++;return 'verified installed source'},hashSource:async()=>sha});
 return{game,combat,flags,installed,writes,adapter,reads:()=>reads};
}
test('only verified bundle identity becomes ready; disabled, changed version, replaced module fail closed',async()=>{
 for(const config of [{active:false},{version:'1.4.4'},{sha:'wrong'}]){const f=fixture(config);assert.equal(await f.adapter.initialize(),false);assert.equal(f.adapter.ready(),false);await f.adapter.restore();assert.equal(f.writes.length,0);}
 const f=fixture();assert.equal(f.adapter.ready(),false);assert.equal(await f.adapter.initialize(),true);assert.equal(f.adapter.ready(),true);
 f.game.modules.set('pf2e-reaction',{active:true,version:'1.4.3'});assert.equal(f.adapter.ready(),false);await f.adapter.restore();assert.equal(f.writes.length,0);
});
test('restores one missing cached slug, preserving other cache and reaction resource flags',async()=>{
 const f=fixture();await f.adapter.initialize();await f.adapter.restore();await f.adapter.restore();
 assert.deepEqual(f.writes,[{'flags.pf2e-reaction.availableReactions':['shield-block',SLUG]}]);
 assert.equal(f.flags['pf2e-reaction'].unrelated,42);assert.equal(f.combat.turns[0].flags['pf2e-reaction'].state,false);
});
test('inactive combats, non-holder combatants, unknown caches and disabled builtin remain untouched',async()=>{
 for(const configure of [f=>f.combat.started=false,f=>f.combat.turns=[],f=>f.combat.turns[0].actor.items=[{type:'feat',slug:SLUG}],f=>f.flags['pf2e-reaction'].availableReactions=undefined,f=>f.flags['pf2e-reaction'].availableReactions={},f=>f.game.settings.get=()=>[]]){
  const f=fixture();await f.adapter.initialize();configure(f);await f.adapter.restore();assert.equal(f.writes.length,0);
 }
});
test('all started combats are considered, including synthetic actor holders',async()=>{
 const f=fixture();const second={...f.combat,id:'second',turns:[{actor:{isToken:true,itemTypes:{action:[{type:'action',system:{slug:SLUG}}]}}}],getFlag:()=>[],update:async changes=>f.writes.push(changes)};
 f.game.combats.set(second.id,second);await f.adapter.initialize();await f.adapter.restore();assert.equal(f.writes.length,2);
});
test('non-primary GM never repairs and handoff during source verification cannot authorize a write',async()=>{
 const f=fixture();f.game.user.id='player';await f.adapter.initialize();await f.adapter.restore();assert.equal(f.writes.length,0);
 f.game.users.activeGM.id='player';await f.adapter.restore();assert.equal(f.writes.length,1);
});
test('identity and cache changes detected immediately before update are not overwritten',async()=>{
 for(const mutate of [f=>f.flags['pf2e-reaction'].availableReactions.push('reactive-shield'),f=>f.game.users.activeGM.id='other',f=>f.game.settings.get=()=>[],f=>f.game.combats.delete('battle'),f=>f.installed.active=false]){
  const f=fixture();await f.adapter.initialize();let calls=0;const native=f.combat.getFlag;
  f.combat.getFlag=(...args)=>{const value=native(...args);if(++calls===2)mutate(f);return value;};
  await f.adapter.restore();assert.equal(f.writes.length,0);
 }
});
test('source identity changed while hashing cannot become ready',async()=>{
 const f=fixture();const adapter=module.createGlimpseReactionCache({game:f.game,fetchSource:async()=>'',hashSource:async()=>{f.installed.version='1.4.4';return SHA;}});
 assert.equal(await adapter.initialize(),false);assert.equal(adapter.ready(),false);
});
test('simultaneous repairs serialize and retain native additions from an earlier write',async()=>{
 const f=fixture();await f.adapter.initialize();let release;
 f.combat.update=async changes=>{f.writes.push(changes);await new Promise(resolve=>release=resolve);f.flags['pf2e-reaction'].availableReactions=[...changes['flags.pf2e-reaction.availableReactions'],'reactive-shield'];};
 const first=f.adapter.restore(),second=f.adapter.restore();await new Promise(resolve=>setImmediate(resolve));release();await Promise.all([first,second]);
 assert.equal(f.writes.length,1);assert.deepEqual(f.flags['pf2e-reaction'].availableReactions,['shield-block',SLUG,'reactive-shield']);
});
