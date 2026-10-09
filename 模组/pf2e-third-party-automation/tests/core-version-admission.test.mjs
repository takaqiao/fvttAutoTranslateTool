import test from 'node:test';
import assert from 'node:assert/strict';
import {loadSalubriousWorkbench} from '../scripts/salubrious-message-privacy.mjs';
import {installPatreonTreatmentCompatibility} from '../scripts/patreon-treatment-compat.mjs';
import {registerPatreonInitiativeCompatibility} from '../scripts/patreon-initiative-compat.mjs';

const cores=['14.369','15.1','25.999',undefined];
function gameFor(version){
 return {version,...version?{release:{generation:Number(version.split('.')[0])}}:{},system:{id:'pf2e',version:'8.6.0'},modules:new Map([
  ['xdy-pf2e-workbench',{active:true,version:'installed'}],['patreon-v3',{active:true,version:'installed'}],
 ]),PF2eWorkbench:{refocus(){}},pf2e:{Check:{}}};
}
function treatmentFixture(version){
 const game=gameFor(version),scope={acquirePatreonPublicScope:()=>({mode:'public',forcedMode:'blind',revalidate:()=>true})};
 const original=function checkCall(next,check,context){context.messageMode='blind';return next(check,context)};
 const entry={fn:original,package_info:{id:'patreon-v3'},target:'game.pf2e.Check.roll',setter:false,type:{name:'WRAPPER'},chain:true,bind:null};
 const holder={name:'game.pf2e.Check.roll',is_property:false,active:true,_outstanding_wrappers:0,getter_data:[entry],get_fn_data(){return this.getter_data},clear_static_dispatch_chain_cache(){},get_static_dispatch_chain(){},call_wrapper(){}};
 const get=()=>entry.fn;get._lib_wrapper=holder;Object.defineProperty(game.pf2e.Check,'roll',{get});
 return {game,scope,entry,holder,original,install:()=>installPatreonTreatmentCompatibility({game,scope})};
}
function initiativeFixture(version){
 const game=gameFor(version),gm={id:'gm',active:true,isGM:true,viewedScene:'scene'},owner={id:'owner'},battleCry={sourceId:'Compendium.pf2e.feats-srd.Item.ePObIpaJDgDb9CQj'},other={sourceId:'other'};
 const actor={id:'actor',type:'character',skills:{intimidation:{rank:3}},items:[battleCry,other],testUserPermission:(user,level)=>user===owner&&level==='OWNER'};
 const scene={id:'scene',tokens:new Map()},token={id:'token',parent:scene,actor};token.object={document:token};scene.tokens.set(token.id,token);
 const message={id:'message',actor,author:owner,isAuthor:true,isCheckRoll:true,rolls:[{}],speaker:{actor:actor.id,scene:scene.id,token:token.id},flags:{pf2e:{context:{type:'initiative'}}}};
 game.user=owner;game.users=new Map([[gm.id,gm],[owner.id,owner]]);game.users.activeGM=gm;game.scenes=new Map([[scene.id,scene]]);game.messages=new Map([[message.id,message]]);
 const original=function processMessage(message){return message.actor.items.filter(()=>true)};
 const chat={id:1,hook:'createChatMessage',once:false,fn:original},custom={id:2,hook:'patreon-v3.processMessage',once:false,fn:original},Hooks={events:{createChatMessage:[chat],'patreon-v3.processMessage':[custom]}};
 return {game,actor,message,owner,battleCry,other,chat,custom,Hooks,original,install:()=>registerPatreonInitiativeCompatibility({game,Hooks,isProviderReady:()=>true})};
}

for(const core of cores){
 test(`native Refocus interface is admitted on core ${core??'without version metadata'}`,async()=>{
  const game=gameFor(core);assert.equal((await loadSalubriousWorkbench({game})).ready,true);
  delete game.PF2eWorkbench.refocus;assert.equal((await loadSalubriousWorkbench({game})).ready,false);
 });
 test(`Patreon treatment preserves the authorized audience on core ${core??'without version metadata'}`,async()=>{
  const f=treatmentFixture(core),result=await f.install();assert.equal(result.installed,true);
  assert.equal(f.entry.fn((_check,context)=>context.messageMode,{}, {messageMode:'public'}),'public');
  f.scope.acquirePatreonPublicScope=()=>({mode:'public',revalidate:()=>false});
  assert.throws(()=>f.entry.fn(()=>assert.fail('lost authorization must not roll'),{}, {messageMode:'public'}),/authorization changed/);
  assert.equal(result.dispose(),true);assert.equal(f.entry.fn,f.original);
 });
 test(`Patreon initiative preserves paired hooks and owner routing on core ${core??'without version metadata'}`,async()=>{
  const f=initiativeFixture(core),result=await f.install();assert.equal(result.status,'installed');
  for(const entry of [f.chat,f.custom])assert.deepEqual(entry.fn(f.message),[f.other]);
  assert.deepEqual(f.actor.items,[f.battleCry,f.other]);assert.equal(f.chat.id,1);assert.equal(f.custom.id,2);
  f.actor.testUserPermission=()=>false;assert.deepEqual(f.chat.fn(f.message),[f.battleCry,f.other]);
  result.dispose();assert.equal(f.chat.fn,f.original);assert.equal(f.custom.fn,f.original);
 });
}

for(const [name,change]of Object.entries({inactive:f=>f.game.modules.get('patreon-v3').active=false,system:f=>f.game.system.id='other',busy:f=>f.holder._outstanding_wrappers=1,duplicate:f=>f.holder.getter_data.push({...f.entry}),entry:f=>f.entry.chain=false,method:f=>delete f.holder.call_wrapper}))test(`future core retains Patreon treatment ${name} guard`,async()=>{
 const f=treatmentFixture('25.999');change(f);assert.equal((await f.install()).installed,false);assert.equal(f.entry.fn,f.original);
});
for(const [name,change]of Object.entries({inactive:f=>f.game.modules.get('patreon-v3').active=false,duplicate:f=>f.Hooks.events['patreon-v3.processMessage'].push({...f.custom}),identity:f=>f.custom.fn=function processMessage(){},readonly:f=>Object.defineProperty(f.custom,'fn',{writable:false}),once:f=>f.chat.once=true}))test(`future core retains Patreon initiative ${name} guard`,async()=>{
 const f=initiativeFixture('25.999');change(f);assert.equal((await f.install()).status,'unsupported');assert.equal(f.chat.fn,f.original);
});
