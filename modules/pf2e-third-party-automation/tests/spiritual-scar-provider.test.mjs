import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture as base} from './glimpse-fixture.mjs';
import {SPIRITUAL_SCAR_SOURCE} from '../scripts/spiritual-scar-native.mjs';
import {createReactionBudget} from '../scripts/reaction-budget.mjs';
let api={};try{api=await import('../scripts/spiritual-scar.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const M='pf2e-third-party-automation';
function apply(target,changes){for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let at=target;for(const p of parts.slice(0,-1))at=at[p]??={};at[parts.at(-1)]=structuredClone(value)}}
async function fixture(){
 assert.equal(typeof api.createSpiritualScarProvider,'function');const f=base(),actor=f.ally,token=f.allyToken,combatant=f.combat.turns[2],calls={uses:0,native:0,followups:[],errors:[],manual:[]};let sequence=0;
 const events=new Map(),Hooks={on(name,fn){const id=++sequence;events.set(id,{name,fn});return id},off(_name,id){events.delete(id)},call(name,...args){for(const e of events.values())if(e.name===name&&e.fn(...args)===false)return false;return true}};
 f.game.world={id:'ujx5r8oipw7ercdr'};f.game.system={id:'pf2e',version:'8.5.1'};f.game.time={worldTime:100};f.game.modules=new Map();f.game.pf2e={settings:{iwr:true}};actor.rollOptions={all:{}};actor.attributes={resistances:[]};actor.classDC={dc:21};actor.flags={};actor.system.resources={reactions:{max:1}};
 f.enemy.system.traits.value=['fiend'];f.roll.instances=[{type:'spirit',persistent:false,total:10}];f.params.rollOptions.add('origin:trait:fiend');
 const ability={id:'scar',uuid:actor.uuid+'.Item.scar',actor,type:'action',sourceId:SPIRITUAL_SCAR_SOURCE,flags:{},system:{slug:'spiritual-scar',actionType:{value:'reaction'},frequency:{max:1,per:'day',value:1}},
  async update(changes,options={}){if(Hooks.call('preUpdateItem',this,changes,options,f.user.id)===false)return;if(ability.mode==='veto')return;apply(this,changes);Hooks.call('updateItem',this,changes,options,f.user.id);return this}};
 actor.items.set(ability.id,ability);f.docs.set(ability.uuid,ability);combatant.uuid='Combat.combat.Combatant.ally';combatant.flags['pf2e-reaction']={state:true};
 combatant.update=async changes=>{if(combatant.mode==='veto')return;apply(combatant,changes);Hooks.call('updateCombatant',combatant,changes,{},f.user.id);return combatant};
 const resources={async snapshot(c){return {c,state:c.flags['pf2e-reaction'].state}},available:s=>s.state,reserve:s=>({proof:{before:true,after:false},changes:{'flags.pf2e-reaction.state':false}}),async release(c,p){return c.flags['pf2e-reaction'].state===p.after?{'flags.pf2e-reaction.state':p.before}:{}}};
 const observers=[],nativeAdapter={nativeBridgeDiagnostic:()=>({ready:true}),addNativeObserver(callback,options){const o={callback,...options};observers.push(o);return()=>observers.splice(observers.indexOf(o),1)}};
 const followup={ready:()=>true,async apply(context){calls.followups.push(context);return {status:'done'}}};
 let decision='use',restriction={status:'clear'},provider;const budget=createReactionBudget({game:f.game,fromUuid:f.fromUuid});
 const message=(data)=>{const id='new-message-'+(++sequence),m={id,uuid:'ChatMessage.'+id,author:f.user,user:f.user,speaker:{actor:actor.id,scene:f.scene.id,token:token.id},rolls:[],flags:{},blind:false,whisper:[],updateSource(ch){apply(this,ch)},async update(ch){apply(this,ch);return this},...data};if(Hooks.call('preCreateChatMessage',m,{}, {},f.user.id)===false)return null;f.game.messages.set(id,m);f.docs.set(m.uuid,m);Hooks.call('createChatMessage',m,{},f.user.id);return m};
 async function originalUse(i){
  calls.uses++;provider.beforeUse(i);const options={[M]:{frequencyReceipt:{id:'receipt-'+(++sequence),itemUuid:i.uuid,userId:f.user.id,before:1,after:0,createdAt:100}}};
  await i.update({'system.frequency.value':0},options);
  const captured=provider.captureUsage(i),m=message({flags:{pf2e:{origin:{actor:actor.uuid,uuid:i.uuid,type:'action'}},[M]:{...captured,usageInput:{actualUse:true,frequencyReceiptId:options[M].frequencyReceipt.id}}}});
  if(!m)return null;
  await budget.record(m,f.user.id);await provider.executeUsage({actor,item:i,message:m,user:f.user,frequencyReceipt:options[M].frequencyReceipt});return m;
 }
 provider=api.createSpiritualScarProvider({game:f.game,fromUuid:f.fromUuid,getRollContext:()=>f.source,reactionResources:resources,reactionRestriction:()=>restriction,nativeAdapter,followup,
  choose:async()=>decision,originalUse,compileResistance:()=>({value:14,applicationLabel:'Spirit from Fiends',test:()=>true}),withResistance:async(a,r,native)=>{a.attributes.resistances.push(r);try{return await native()}finally{a.attributes.resistances.splice(a.attributes.resistances.indexOf(r),1)}},onError:e=>calls.errors.push(e),onManual:context=>calls.manual.push(context),onUnsupported:context=>calls.manual.push(context)});
 provider.register({Hooks});
 async function run({damage=0,applications=[{category:'resistance',type:'Spirit from Fiends',adjustment:-10,ignored:false}],persistent=[],card=true,emit=true}={}){
  const prepared=await provider.beforeDamage(actor,f.params);let applied=false;
  try{const result=await provider.wrapNativeDamage(actor,prepared.params,async params=>{calls.native++;if(emit)for(const o of observers)if(o.matches(actor,params))o.callback({actorUuid:actor.uuid,tokenUuid:token.uuid,itemUuid:f.item.uuid,total:10,rollOptions:[...params.rollOptions],iwr:{finalDamage:damage,applications,persistent},nativeAmounts:{actorDamage:damage,shieldDamage:0}});
   if(card)message({flags:{pf2e:{context:{type:'damage-taken',options:[...params.rollOptions]},origin:{actor:f.enemy.uuid,uuid:f.item.uuid}}}});return 'native-return';});applied=true;return result;
  }finally{await provider.afterDamage(prepared.receipt,{applied})}
 }
 return {...f,actor,token,ability,combatant,provider,calls,run,Hooks,events,createMessage:message,followup,decision:value=>decision=value,restriction:value=>restriction=value};
}
test('one private damage scope pays original daily Use and one reaction, then follows actual Scar prevention',async()=>{
 const f=await fixture();assert.equal(await f.run(),'native-return');assert.equal(f.calls.uses,1);assert.equal(f.calls.native,1);assert.equal(f.calls.followups.length,1);assert.equal(f.ability.system.frequency.value,0);assert.equal(f.combatant.flags['pf2e-reaction'].state,false);assert.equal(f.combatant.flags[M].reactionBudget.entries.length,1);assert.equal(f.actor.attributes.resistances.length,0);assert.equal(f.actor.rollOptions.all['spiritual-scar'],undefined);
});
test('declining the choice does not pay or change ordinary native damage',async()=>{const f=await fixture();f.decision('decline');await f.run();assert.equal(f.calls.uses,0);assert.equal(f.calls.native,1);assert.equal(f.calls.followups.length,0);assert.equal(f.ability.system.frequency.value,1);assert.equal(f.combatant.flags['pf2e-reaction'].state,true)});
test('Toolbelt merged spirit damage leaves Scar daily use and reaction untouched while native damage proceeds',async()=>{
 const f=await fixture();f.message.flags['pf2e-toolbelt']={betterChat:{mergeDamage:{merged:true,data:[{source:{_id:'strike-one'}},{source:{_id:'strike-two'}}]}}};
 const prepared=await f.provider.beforeDamage(f.actor,f.params);assert.equal(prepared.params,f.params);assert.equal(prepared.receipt,undefined);
 let nativeCalls=0;assert.equal(await f.provider.wrapNativeDamage(f.actor,prepared.params,async params=>{nativeCalls++;assert.equal(params,f.params);return 'native-return'}),'native-return');
 assert.equal(nativeCalls,1);assert.equal(f.calls.uses,0);assert.equal(f.calls.followups.length,0);assert.equal(f.calls.manual.length,1);assert.equal(f.ability.system.frequency.value,1);assert.equal(f.combatant.flags['pf2e-reaction'].state,true);assert.deepEqual(f.calls.errors,[]);
});
test('the actual target encounter is used even while a different combat is viewed',async()=>{const f=await fixture();assert.notEqual(f.game.combat,f.combat);await f.run();assert.equal(f.combatant.flags[M].spiritualScarClaims[0].epoch,'combat:1')});
test('existing spent reaction or active reaction restriction cannot spend another daily use',async()=>{
 for(const change of [f=>f.combatant.flags['pf2e-reaction'].state=false,f=>f.restriction({status:'restricted'}),f=>f.combatant.flags[M]={reactionBudget:{epoch:'combat:1',entries:[{type:'reaction',cost:1}]}}]){const f=await fixture();change(f);await f.run();assert.equal(f.calls.uses,0);assert.equal(f.ability.system.frequency.value,1);}
});
test('manual Scar toggle stays manual and is never silently disabled or double charged',async()=>{const f=await fixture();f.actor.rollOptions.all['spiritual-scar']=true;await f.run();assert.equal(f.calls.uses,0);assert.equal(f.actor.rollOptions.all['spiritual-scar'],true)});
test('an invalid original damage source fails before payment or native damage',async()=>{const f=await fixture();f.message.author={id:'unknown'};await assert.rejects(f.run());assert.equal(f.calls.uses,0);assert.equal(f.calls.native,0);assert.equal(f.ability.system.frequency.value,1)});
test('native damage leaving HP damage does not trigger Will',async()=>{const f=await fixture();await f.run({damage:3,applications:[{category:'resistance',type:'Spirit from Fiends',adjustment:-7,ignored:false}]});assert.equal(f.calls.followups.length,0);assert.equal(f.calls.uses,1)});
for(const[name,options]of[
 ['immunity',{applications:[{category:'immunity',type:'Spirit',adjustment:-10}]}],['another resistance',{applications:[{category:'resistance',type:'All Damage',adjustment:-10,ignored:false}]}],['redirected resistance',{applications:[{category:'resistance',type:'Spirit from Fiends',adjustment:-10,ignored:false,redirect:'Fire'}]}],['future persistent damage',{persistent:[{type:'spirit',formula:'1d6'}]}]
])test(`zero damage from ${name} is not an automatic Scar follow-up`,async()=>{const f=await fixture();await f.run(options);assert.equal(f.calls.followups.length,0);assert.equal(f.calls.uses,1)});
test('missing native IWR observation or unique damage receipt leaves paid use uncertain and cannot replay',async()=>{
 for(const options of [{emit:false},{card:false}]){const f=await fixture();await assert.rejects(f.run(options));assert.equal(f.calls.followups.length,0);assert.equal(f.ability.system.frequency.value,0);assert.equal(f.combatant.flags['pf2e-reaction'].state,false);assert.equal(f.actor.attributes.resistances.length,0);}
});
test('a no-op reaction resource write cannot authorize original daily Use merely by saving the claim',async()=>{
 const f=await fixture(),update=f.combatant.update;f.combatant.update=async changes=>{const result=await update(changes);if(changes['flags.pf2e-reaction.state']===false)f.combatant.flags['pf2e-reaction'].state=true;return result};
 await assert.rejects(f.run());assert.equal(f.calls.uses,0);assert.equal(f.calls.native,0);assert.equal(f.ability.system.frequency.value,1);
});
test('a failed follow-up result is not marked as fully completed',async()=>{
 const f=await fixture();f.followup.apply=async()=>undefined;await assert.rejects(f.run());assert.equal(f.combatant.flags[M].spiritualScarClaims[0].status,'uncertain');assert.equal(f.calls.uses,1);
});
test('same-label and mixed-prevention ambiguity retain the valid resistance use and explicitly defer only the follow-up',async()=>{
 const f=await fixture(),other={applicationLabel:'Spirit from Fiends',value:30};f.actor.attributes.resistances.push(other);await f.run();assert.equal(f.calls.uses,1);assert.equal(f.calls.followups.length,0);assert.equal(f.calls.manual[0].reason,'same-label-resistance');assert.deepEqual(f.actor.attributes.resistances,[other]);
 const g=await fixture();await g.run({applications:[{category:'immunity',type:'Fire',adjustment:-3},{category:'resistance',type:'Spirit from Fiends',adjustment:-7,ignored:false}]});assert.equal(g.calls.manual[0].reason,'combined-damage-prevention');assert.equal(g.calls.followups.length,0);assert.equal(g.calls.uses,1);
});
test('an altered automatic Use marker cannot publish a default-public card before failing payment binding',async()=>{
 const f=await fixture();f.message.blind=true;f.message.whisper=['u'];f.provider.captureUsage=()=>({spiritualScarInput:{nonce:'forged',paymentNonce:'forged'}});
 await assert.rejects(f.run());assert.equal(f.calls.native,0);assert.equal(f.ability.system.frequency.value,0);assert.equal([...f.game.messages.values()].filter(m=>m.speaker?.actor===f.actor.id).length,0);
});
test('failure before original Use begins releases only the proven unattempted reservation',async()=>{
 const f=await fixture(),update=f.combatant.update;f.combatant.update=async changes=>{const result=await update(changes);if(changes['flags.pf2e-reaction.state']===false)f.actor.canAct=false;return result};
 await assert.rejects(f.run());assert.equal(f.calls.uses,0);assert.equal(f.ability.system.frequency.value,1);assert.equal(f.combatant.flags['pf2e-reaction'].state,true);assert.equal(f.combatant.flags[M].spiritualScarClaims[0].status,'refunded');assert.equal(f.combatant.flags[M].reactionBudget.entries.length,0);
});
test('pre-Use cancellation does not restore a reaction now spent by another exact entry',async()=>{
 const f=await fixture(),update=f.combatant.update;f.combatant.update=async changes=>{const result=await update(changes);if(changes['flags.pf2e-reaction.state']===false){f.actor.canAct=false;f.combatant.flags[M].reactionBudget.entries.push({type:'reaction',cost:1,slug:'other',msgId:'other-use'})}return result};
 await assert.rejects(f.run());assert.equal(f.calls.uses,0);assert.equal(f.combatant.flags['pf2e-reaction'].state,false);assert.equal(f.combatant.flags[M].reactionBudget.entries.length,1);assert.equal(f.combatant.flags[M].reactionBudget.entries[0].msgId,'other-use');
});
test('a turn refresh during daily-claim cancellation never overwrites the new reaction ledger',async()=>{
 const f=await fixture(),update=f.combatant.update,itemUpdate=f.ability.update;
 f.combatant.update=async changes=>{const result=await update(changes);if(changes['flags.pf2e-reaction.state']===false)f.actor.canAct=false;return result};
 f.ability.update=async(...args)=>{const result=await itemUpdate.apply(f.ability,args);const s=f.ability.flags[M]?.spiritualScarUse;if(s?.operations?.[s.currentNonce]?.status==='cancelled'){f.combat.round=3;f.combat.turn=2;f.combatant.flags[M].reactionBudget={epoch:'combat:3',entries:[{type:'reaction',cost:1,msgId:'new-turn-use'}]};f.combatant.flags['pf2e-reaction'].state=false}return result};
 await assert.rejects(f.run());assert.equal(f.calls.uses,0);assert.equal(f.combatant.flags['pf2e-reaction'].state,false);assert.deepEqual(f.combatant.flags[M].reactionBudget,{epoch:'combat:3',entries:[{type:'reaction',cost:1,msgId:'new-turn-use'}]});
});
test('unregister cancels pending synchronization and removes its temporary hooks',async()=>{
 const f=await fixture();f.provider.executeUsage=async()=>undefined;const running=f.run();running.catch(()=>{});
 for(let n=0;n<100&&[...f.events.values()].filter(e=>e.name==='updateItem').length<2;n++)await new Promise(setImmediate);
 assert.equal([...f.events.values()].filter(e=>e.name==='updateItem').length,2);f.provider.unregister();
 assert.equal(f.events.size,0);assert.equal(f.provider.ready(),false);await assert.rejects(running);assert.equal(f.calls.native,0);
});
