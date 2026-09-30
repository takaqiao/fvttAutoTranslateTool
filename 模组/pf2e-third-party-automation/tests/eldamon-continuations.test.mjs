import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './eldamon-electricity-fixture.mjs';
import {createEldamonBasicSettlement} from '../scripts/eldamon-basic-settlement.mjs';
import {createEldamonElectricityProvider} from '../scripts/eldamon-electricity-provider.mjs';
import {ELECTRICITY_BASIC_SOURCES as B,ELECTRICITY_SOURCES as S,electricityEffects,electricityState} from '../scripts/eldamon-electricity.mjs';
import {registerUsageEvents} from '../scripts/usage-events.mjs';
const ID='pf2e-third-party-automation';
function basic(kind='manipulation'){
 const f=fixture();f.game.world={id:'ujx5r8oipw7ercdr'};f.game.system={id:'pf2e',version:'8.5.1'};f.game.modules=new Map([['battlezoo-eldamon-pf2e',{active:true}]]);
 f.game.scenes=new Map([['s',f.scene]]);f.item(f.caster,'element',B.element,{type:'feat'});f.power.sourceId=B[kind];f.power.type=kind==='shield'?'action':'feat';
 f.card.flags.pf2e.origin={uuid:f.power.uuid,type:f.power.type,actor:f.caster.uuid};f.card.flags[ID].usageInput={actualUse:true,targetUuids:[f.tokens[1].uuid]};
 f.card.timestamp=100;
 if(kind==='shield'){f.power.system.selfEffect={uuid:'Compendium.battlezoo-eldamon-pf2e.effects.Item.OJMStIdZzBU4L4N6'};f.item(f.caster,'nativeShield',f.power.system.selfEffect.uuid,{system:{context:{origin:{actor:f.caster.uuid,item:f.power.uuid}}}});}
 let sequence=0;const provider=createEldamonBasicSettlement({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),random:()=>`declaration-${++sequence}`,notify:()=>{},basicUse:(payload,user)=>f.ledger().basicUse(payload,user),apply:async payload=>{const caller=f.game.user;f.game.user=f.gm;try{return await f.ledger().confirmedAction(payload,caller)}finally{f.game.user=caller}}});
 const use=()=>provider.executeUsage({actor:f.caster,item:f.power,message:f.card,user:f.owner,action:provider.resolveAction(f.power)});
 return {...f,provider,use};
}
function attack(f,{outcome='failure',melee=true}={}){
 const weapon=f.item(f.target,'weapon','Compendium.pf2e.equipment-srd.Item.example',{type:'weapon',isMelee:melee});
 const m={id:'attack',uuid:'ChatMessage.attack',timestamp:200,author:f.owner,isCheckRoll:true,rolls:[{_evaluated:true,total:20}],speaker:{actor:f.target.id,scene:'s',token:f.target.id},item:weapon,async update(patch){if(patch['flags.pf2e.context.options'])this.flags.pf2e.context.options=structuredClone(patch['flags.pf2e.context.options'])},
 flags:{pf2e:{origin:{uuid:weapon.uuid},context:{type:'attack-roll',outcome,options:[melee?'item:melee':'item:ranged'],target:{actor:f.caster.uuid,token:f.tokens[0].uuid}}}}};
 f.game.messages.set(m.id,m);f.docs.set(m.uuid,m);return m;
}
test('real manipulation Use settles its captured target, not the executing GM current targets',async()=>{
 const f=basic();f.gm.targets=new Set([{document:f.tokens[2]}]);assert.equal(f.provider.requiresActualUse(f.power),true);await f.use();
 assert.equal(electricityEffects(f.target,S.shocked).length,1);assert.equal(electricityEffects(f.other,S.shocked).length,0);
});
test('concurrent original Use deliveries create one Shocked effect and self-target does not deadlock the actor queue',async()=>{
 const f=basic(),ledger=f.ledger(),create=f.target.createEmbeddedDocuments;let creations=0,release;
 f.target.createEmbeddedDocuments=async function(...args){creations++;await new Promise(r=>release=r);return create.apply(this,args)};
 const payload={actorUuid:f.caster.uuid,itemUuid:f.power.uuid,messageUuid:f.card.uuid},first=ledger.basicUse(payload,f.owner),second=ledger.basicUse(payload,f.owner);
 await new Promise(r=>setImmediate(r));release();await Promise.all([first,second]);assert.equal(creations,1);assert.equal(electricityEffects(f.target,S.shocked).length,1);
 const g=basic();g.card.flags[ID].usageInput.targetUuids=[g.tokens[0].uuid];await g.use();assert.equal(electricityEffects(g.caster,S.shocked).length,1);
});
test('display-only manipulation cannot settle even through the direct public executor',async()=>{
 const f=basic();f.card.flags[ID].usageInput.actualUse=false;await assert.rejects(f.use());assert.equal(electricityEffects(f.target,S.shocked).length,0);
});
test('the original manipulation activity cannot refresh or switch targets through a second entrance/nonce',async()=>{
 const f=basic();await f.use();await f.ledger().interact({actorUuid:f.target.uuid,nonce:'discharge',confirmed:true},f.gm);
 await f.ledger().confirmedAction({actorUuid:f.caster.uuid,itemUuid:f.power.uuid,messageUuid:f.card.uuid,targetUuid:f.tokens[2].uuid,nonce:'different-nonce',confirmed:true},f.owner);
 assert.equal(electricityEffects(f.target,S.shocked).length,0);assert.equal(electricityEffects(f.other,S.shocked).length,0);
});
test('a lost target journal cannot let an older manipulation overwrite a newer activity',async()=>{
 const f=basic(),nativeUpdate=f.target.update;let fail=true;
 f.target.update=async function(data,options){if(fail&&electricityEffects(this,S.shocked).length){fail=false;throw Error('journal lost')}return nativeUpdate.call(this,data,options)};
 await assert.rejects(f.use(),/journal lost/);const original=f.card;
 f.card={...original,id:'second-use',uuid:'ChatMessage.second-use',flags:structuredClone(original.flags)};f.card.timestamp=300;f.game.messages.set(f.card.id,f.card);f.docs.set(f.card.uuid,f.card);
 await f.provider.executeUsage({actor:f.caster,item:f.power,message:f.card,user:f.owner});const effect=electricityEffects(f.target,S.shocked)[0],key=effect.flags[ID].electricityShock.key;
 await assert.rejects(f.provider.executeUsage({actor:f.caster,item:f.power,message:original,user:f.owner}));assert.equal(effect.flags[ID].electricityShock.key,key);
});
test('interrupted manipulation retains its target actor identity when the original token is relinked',async()=>{
 const f=basic();f.target.createEmbeddedDocuments=async()=>{throw Error('native interrupted')};await assert.rejects(f.use(),/native interrupted/);
 f.tokens[1].actor=f.other;await assert.rejects(f.use());assert.equal(electricityEffects(f.other,S.shocked).length,0);
});
test('shield Use initializes only the shield activity; a bound real miss declaration shocks its original attacker once',async()=>{
 const f=basic('shield');await f.use();assert.equal(electricityEffects(f.target,S.shocked).length,0);
 const m=attack(f);f.game.user=f.owner;await f.provider.declareShieldTrigger(f.caster,{message:m});
 assert.equal(electricityEffects(f.target,S.shocked).length,1);
 f.game.user=f.gm;await f.ledger().interact({actorUuid:f.target.uuid,nonce:'discharge',confirmed:true},f.gm);f.game.user=f.owner;
 await f.provider.declareShieldTrigger(f.caster,{message:m});assert.equal(electricityEffects(f.target,S.shocked).length,0);
});
test('the old shield card and current attack shortcut share the same recorded native trigger',async()=>{
 const f=basic('shield');await f.use();const m=attack(f);f.game.user=f.owner;f.owner.targets=new Set([{document:f.tokens[1]}]);await f.provider.declareShieldTrigger(f.caster,{message:m});
 f.game.user=f.gm;await f.ledger().interact({actorUuid:f.target.uuid,nonce:'discharged',confirmed:true},f.gm);f.game.user=f.owner;
 await f.provider.settleFromCard(f.card);assert.equal(electricityEffects(f.target,S.shocked).length,0);
 f.game.messages.values=()=>assert.fail('second full chat scan');await f.provider.settleFromCard(f.card);assert.equal(electricityEffects(f.target,S.shocked).length,0);
});
test('a kept native hero reroll carries the original consumed shield trigger instead of becoming a second trigger',async()=>{
 const f=basic('shield');await f.use();const original=attack(f);f.game.user=f.owner;await f.provider.declareShieldTrigger(f.caster,{message:original});
 f.game.user=f.gm;await f.ledger().interact({actorUuid:f.target.uuid,nonce:'discharge-original',confirmed:true},f.gm);f.game.user=f.owner;f.game.messages.delete(original.id);
 const kept={...original,id:'kept',uuid:'ChatMessage.kept',flags:structuredClone(original.flags)};kept.flags.pf2e.context.isReroll=true;kept.flags.pf2e.context.options.push('check:reroll');f.game.messages.set(kept.id,kept);f.docs.set(kept.uuid,kept);
 await f.provider.declareShieldTrigger(f.caster,{message:kept});assert.equal(electricityEffects(f.target,S.shocked).length,0);assert.equal(Object.keys(electricityState(f.caster).basicActions['use-channel'].triggers).length,1);
});
test('the first observed kept native miss can declare one shield trigger, while a copy of a live original cannot',async()=>{
 const f=basic('shield');await f.use();const kept=attack(f);kept.flags.pf2e.context.isReroll=true;kept.flags.pf2e.context.options.push('check:reroll');f.game.user=f.owner;await f.provider.declareShieldTrigger(f.caster,{message:kept});assert.equal(electricityEffects(f.target,S.shocked).length,1);
 const copy={...kept,id:'copy',uuid:'ChatMessage.copy',flags:structuredClone(kept.flags)};f.game.messages.set(copy.id,copy);f.docs.set(copy.uuid,copy);await assert.rejects(f.provider.declareShieldTrigger(f.caster,{message:copy}));
});
test('shield declaration refuses native hit, ranged attack, expired shield and wrong target binding',async()=>{
 for(const change of [f=>attack(f,{outcome:'success'}),f=>attack(f,{melee:false}),f=>{const m=attack(f);m.flags.pf2e.context.target.actor=f.other.uuid;return m},f=>{const m=attack(f);f.combat.round=2;return m}]){
  const f=basic('shield');await f.use();const m=change(f);f.game.user=f.owner;await assert.rejects(f.provider.declareShieldTrigger(f.caster,{message:m}));assert.equal(electricityEffects(f.target,S.shocked).length,0);
 }
});
test('shield actor fallback declares an unrecorded real trigger and requires an active original shield Use',async()=>{
 const f=basic('shield');f.owner.targets=new Set([{document:f.tokens[1]}]);f.game.user=f.owner;await assert.rejects(f.provider.declareShieldTrigger(f.caster));
 f.game.user=f.gm;await f.use();f.game.user=f.owner;await f.provider.declareShieldTrigger(f.caster);assert.equal(electricityEffects(f.target,S.shocked).length,1);
});
test('GM can continue the original player owned shield',async()=>{
 const f=basic('shield');await f.use();const m=attack(f);await f.provider.declareShieldTrigger(f.caster,{message:m});assert.equal(electricityEffects(f.target,S.shocked).length,1);
});
test('an earlier attack cannot trigger a later shield',async()=>{
 const g=basic('shield');g.card.timestamp=300;await g.use();const old=attack(g);g.game.user=g.owner;await assert.rejects(g.provider.declareShieldTrigger(g.caster,{message:old}));assert.equal(electricityEffects(g.target,S.shocked).length,0);
});
test('deleting or expiring the native shield effect disables its still-recorded continuation',async()=>{
 for(const remove of [f=>f.caster.items.delete('nativeShield'),f=>f.caster.items.get('nativeShield').isExpired=true]){const f=basic('shield');await f.use();remove(f);f.game.user=f.owner;await assert.rejects(f.provider.declareShieldTrigger(f.caster,{message:attack(f)}));assert.equal(electricityEffects(f.target,S.shocked).length,0);}
});
test('shield attack provenance cannot substitute a different actor weapon or relinked source token',async()=>{
 for(const alter of [(f,m)=>m.item.actor=f.other,(f,m)=>f.tokens[0].actor=f.other]){const f=basic('shield');await f.use();const m=attack(f);alter(f,m);f.game.user=f.owner;await assert.rejects(f.provider.declareShieldTrigger(f.caster,{message:m}));assert.equal(electricityEffects(f.target,S.shocked).length,0);}
});
test('chain eligibility never reads geometry and normal discharge still differs from siphoning',async()=>{
 const f=fixture();for(const token of f.tokens)token.object={distanceTo(){assert.fail('geometry gate called')}};await(await f.damage()).finish();
 const payload={actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:true},kind:'normal'};
 assert.equal((await f.ledger().candidates(payload,f.owner)).length,1);payload.kind='siphoning';assert.equal((await f.ledger().candidates(payload,f.owner)).length,0);
});
test('GM-coordinated chain candidates and validation keep private receipt/source amounts from an ordinary owner',async()=>{
 for(const hide of [(source,receipt)=>{receipt.blind=true;receipt.whisper=['gm']},(source,receipt)=>{source.whisper=['gm']},(source,receipt)=>{receipt.whisper=['gm']}]){
  const f=fixture();f.item(f.other,'shock',S.shocked);const d=await f.damage();await d.finish();const source=f.docs.get(d.payload.sourceMessageUuid);
  const payload={actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'};
  const [visible]=await f.ledger().candidates(payload,f.owner);assert.equal(visible.amount,13);hide(source,d.receipt);assert.deepEqual(await f.ledger().candidates(payload,f.owner),[]);assert.equal((await f.ledger().candidates(payload,f.gm)).length,1);
  f.power.sourceId=S.chain;await assert.rejects(f.ledger().validateSelection({actor:f.caster,item:f.power,user:f.owner,kind:'normal',selection:{...payload.selection,triggerDamage:13,electricityEvidence:visible.evidence}}));
 }
});
test('current receipt shortcut enters one genuine owned Use and fixes its exact source through beforeChannel',async()=>{
 const f=fixture();f.game.modules=new Map();f.item(f.other,'shock',S.shocked);const first=await f.damage();await first.finish();const second=await f.damage();await second.finish();
 f.power.sourceId=S.chain;f.caster.getActiveTokens=()=>[f.tokens[0]];let uses=0,observed;
 const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),useOwnedAction:async item=>{uses++;assert.equal(item,f.power);observed=await provider.beforeChannel({item,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'});return 'native-card'},selectChoice:()=>assert.fail('bound source must not choose unrelated receipt')});
 assert.equal(await provider.continueChain({message:first.receipt,actor:f.caster}),'native-card');assert.equal(uses,1);assert.equal(observed.electricityEvidence.receiptUuid,first.receipt.uuid);assert.notEqual(observed.electricityEvidence.nonce,second.payload.nonce);
});
test('bound chain source changes/cancels/duplicate clicks cannot fall through into an unrelated source',async()=>{
 const f=fixture();f.game.modules=new Map();f.item(f.other,'shock',S.shocked);const d=await f.damage();await d.finish();f.power.sourceId=S.chain;f.caster.getActiveTokens=()=>[f.tokens[0]];
 let release,uses=0;const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),useOwnedAction:async()=>{uses++;await new Promise(r=>release=r);return null;}});
 const pending=provider.continueChain({message:d.receipt,actor:f.caster});await new Promise(r=>setImmediate(r));await provider.continueChain({message:d.receipt,actor:f.caster});assert.equal(uses,1);release();assert.equal(await pending,null);
 d.receipt.flags[ID].electricityApplied.amount=0;
 const rejecting=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),useOwnedAction:async item=>rejecting.beforeChannel({item,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'})});
 await assert.rejects(rejecting.continueChain({message:d.receipt,actor:f.caster}));assert.equal(electricityState(f.caster).damage?.length,undefined);
});
function dom(){const node=tag=>({tag,dataset:{},children:[],listeners:{},append(c){this.children.push(c)},addEventListener(k,f){this.listeners[k]=f},querySelector(s){return this.querySelectorAll(s)[0]??null},querySelectorAll(s){const all=this.children.flatMap(c=>[c,...c.querySelectorAll('*')]);if(s==='*')return all;const field=s.match(/^\[data-([\w-]+)\]$/)?.[1]?.replace(/-([a-z])/g,(_m,c)=>c.toUpperCase());return field?all.filter(c=>Object.hasOwn(c.dataset,field)):[]}});const root=node('root');root.ownerDocument={createElement:node};return root;}
function hooks(){const rows=new Map();return {rows,on(name,fn){const list=rows.get(name)??[];list.push(fn);rows.set(name,list);return fn},off(name,fn){rows.set(name,(rows.get(name)??[]).filter(f=>f!==fn))},async fire(name,...args){for(const fn of rows.get(name)??[])await fn(...args)}};}
test('the real usage observer ignores display and executes original two-action manipulation exactly once',async()=>{
 const f=basic(),Hooks=hooks(),errors=[];f.card.update=async patch=>{for(const [key,value]of Object.entries(patch))if(key===`flags.${ID}.usage`)f.card.flags[ID].usage=structuredClone(value)};
 const unregister=registerUsageEvents({game:f.game,Hooks,libWrapper:null,fromUuid:async uuid=>f.docs.get(uuid),resolveAction:f.provider.resolveAction,requiresActualUse:f.provider.requiresActualUse,executeUsage:f.provider.executeUsage,onError:error=>errors.push(error)});
 try{f.card.flags[ID].usageInput.actualUse=false;await Hooks.fire('createChatMessage',f.card,{},f.owner.id);assert.equal(f.card.flags[ID].usage,undefined);assert.equal(electricityEffects(f.target,S.shocked).length,0);
  f.card.flags[ID].usageInput.actualUse=true;await Hooks.fire('createChatMessage',f.card,{},f.owner.id);await Hooks.fire('createChatMessage',f.card,{},f.owner.id);assert.equal(f.card.flags[ID].usage.status,'done');assert.equal(electricityEffects(f.target,S.shocked).length,1);assert.deepEqual(errors,[]);
 }finally{unregister()}
});
test('current failed attack renders one local shield declaration and private attacks expose no controls',async()=>{
 const f=basic('shield');await f.use();f.game.user=f.owner;const Hooks=hooks();f.provider.register({Hooks});const m=attack(f),root=dom();
 await Hooks.fire('renderChatMessageHTML',m,root);await Hooks.fire('renderChatMessageHTML',m,root);assert.equal(root.querySelectorAll('[data-eldamon-shield-trigger]').length,1);
 const hidden=dom();m.isContentVisible=false;await Hooks.fire('renderChatMessageHTML',m,hidden);assert.equal(hidden.children.length,0);
});
test('private damage receipts do not expose chain source controls or mechanical attribution',async()=>{
 const f=fixture(),d=await f.damage();await d.finish();f.game.modules=new Map();f.power.sourceId=S.chain;f.game.user.character=f.caster;const Hooks=hooks(),provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid)});provider.register({Hooks});
 const saved=globalThis.document,root=dom();globalThis.document=root.ownerDocument;try{d.receipt.isContentVisible=false;await Hooks.fire('renderChatMessageHTML',d.receipt,root);assert.equal(root.children.length,0);}finally{globalThis.document=saved;}
});
