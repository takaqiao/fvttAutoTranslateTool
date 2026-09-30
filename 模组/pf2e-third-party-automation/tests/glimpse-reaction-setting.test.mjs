import test from 'node:test';
import assert from 'node:assert/strict';
import * as setting from '../scripts/glimpse-reaction-setting.mjs';
const {glimpseReactionSetting,GLIMPSE_REACTION_REASON:reason}=setting;
const before=['shield-block','glimpse-of-redemption','reactive-strike'],after=['shield-block','reactive-strike'];
const game={settings:{get:()=>[{setting:'pf2e-reaction.builtinReactionsEnabled',changes:[{path:'builtinReactionsEnabled',before,after,reason}]}]}};
test('verified provider removes only its builtin reminder; an unavailable provider restores its exact prior change',()=>{
 assert.deepEqual(glimpseReactionSetting(before,game,true),after);assert.deepEqual(glimpseReactionSetting(after,game,false),before);assert.deepEqual(before,['shield-block','glimpse-of-redemption','reactive-strike']);
});
test('provider fallback preserves a reminder disabled before module ownership',()=>{
 assert.deepEqual(glimpseReactionSetting(after,{settings:{get:()=>[]}},false),after);
});
test('fallback restores only Glimpse while preserving unrelated user additions and removals',()=>{
 const current=['reactive-shield','shield-block'];
 assert.deepEqual(glimpseReactionSetting(current,game,false),['reactive-shield','glimpse-of-redemption','shield-block']);
 assert.deepEqual(current,['reactive-shield','shield-block']);
});
const record=(before,after,why=reason)=>({setting:'pf2e-reaction.builtinReactionsEnabled',changes:[{path:'builtinReactionsEnabled',before,after,reason:why}]});
const historyGame=history=>({settings:{get:()=>history}});
test('another module maintenance of the same array does not strand the owned removal',()=>{
 const first=['glimpse-of-redemption','disarming-block','shield-block'],removed=['disarming-block','shield-block'];
 const history=[record(first,removed),record(removed,['shield-block'],'disarming-block integration')];
 assert.deepEqual(glimpseReactionSetting(['shield-block'],historyGame(history),false),['glimpse-of-redemption','shield-block']);
});
test('restoration settles ownership so a later user disable is not resurrected',()=>{
 const history=[record(before,after),record(after,before)];
 assert.deepEqual(glimpseReactionSetting(after,historyGame(history),false),after);
});
test('a later explicit membership change by another integration supersedes ownership',()=>{
 const history=[record(before,after),record(after,before,'manual restore'),record(before,after,'manual disable')];
 assert.deepEqual(glimpseReactionSetting(after,historyGame(history),false),after);
});
test('new suppression after a prior restore establishes fresh ownership',()=>{
 const history=[record(before,after),record(after,before),record(before,after)];
 assert.deepEqual(glimpseReactionSetting(['shield-block'],historyGame(history),false),['shield-block','glimpse-of-redemption']);
});
test('foreign or malformed records cannot establish ownership; present reminder is not duplicated',()=>{
 for(const history of [null,{},[record(before,after,'other integration')],[record(null,after)],[{...record(before,after),setting:'other.setting'}]]){
  assert.deepEqual(glimpseReactionSetting(after,historyGame(history),false),after);
 }
 assert.deepEqual(glimpseReactionSetting(before,game,false),before);
 assert.equal(glimpseReactionSetting(null,game,false),null);
});
const action={type:'action',slug:'glimpse-of-redemption'};
const covered={type:'character',level:5,items:[action]};
const provider={handlesActor:actor=>actor===covered};
function canSuppress(actors,p=provider){
 assert.equal(typeof setting.canSuppressGlimpseReminder,'function','coverage helper must exist');
 return setting.canSuppressGlimpseReminder(actors,p);
}
test('suppress only when every actual holder is covered; unrelated actors do not veto',()=>{
 assert.equal(canSuppress([covered,{type:'npc',items:[]}]),true);
 assert.equal(canSuppress([covered,{type:'character',level:6,items:[action]}]),false);
});
test('NPC and familiar action holders veto world suppression just as Reaction builtin permits them',()=>{
 for(const type of ['npc','familiar'])assert.equal(canSuppress([covered,{type,itemTypes:{action:[action]}}]),false);
});
test('source-matched or raw-slug action holders conservatively veto even when renamed or unprepared',()=>{
 for(const item of [{type:'action',system:{slug:'glimpse-of-redemption'}},{type:'action',sourceId:'Compendium.pf2e.actionspf2e.Item.tuZnRWHixLArvaIf'},{type:'action',_stats:{compendiumSource:'Compendium.pf2e.actionspf2e.Item.tuZnRWHixLArvaIf'}}]){
  assert.equal(canSuppress([covered,{items:[item]}]),false);
 }
});
test('empty holder set never disables the builtin, and feats with matching names are not actions',()=>{
 assert.equal(canSuppress([]),false);
 assert.equal(canSuppress([{items:[{type:'feat',slug:'glimpse-of-redemption'}]}]),false);
});
test('Foundry-style collections work and absent or throwing provider cannot claim coverage',()=>{
 assert.equal(canSuppress(new Map([['a',{...covered,items:new Map([['i',action]])}]]),{handlesActor:()=>true}),true);
 assert.equal(canSuppress([covered],{}),false);
 assert.equal(canSuppress([covered],{handlesActor:()=>{throw Error('not ready')}}),false);
});
