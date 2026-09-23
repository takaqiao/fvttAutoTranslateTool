import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createEldamonElectricityProvider,preserveElectricityOnAlter} from '../scripts/eldamon-electricity-provider.mjs';
import {fixture} from './eldamon-electricity-fixture.mjs';
import {destructiveBlockAmounts} from '../scripts/shield-damage-adapter.mjs';
import {ELECTRICITY_SOURCES as S,electricityEffects,electricityState} from '../scripts/eldamon-electricity.mjs';
const ID='pf2e-third-party-automation';
test('electricity application indexes source cards once and releases deleted chat references',async()=>{
 const f=fixture(),hooks=new Map(),provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid)});
 provider.register({Hooks:{on:(name,fn)=>hooks.set(name,fn)}});const first=f.source();
 assert.ok((await provider.beforeDamage(f.target,{damage:first.rolls[0],token:f.tokens[1],item:f.power})).receipt);
 f.game.messages.values=()=>{throw Error('chat history copied again')};
 const second=f.source();hooks.get('createChatMessage')(second,{},f.gm.id);
 assert.ok((await provider.beforeDamage(f.target,{damage:second.rolls[0],token:f.tokens[1],item:f.power})).receipt);
 f.game.messages.delete(first.id);hooks.get('deleteChatMessage')?.(first);
 assert.equal(await provider.beforeDamage(f.target,{damage:first.rolls[0],token:f.tokens[1],item:f.power}),null);
});
test('area damage binds native targets after template placement instead of the pre-use selection',async()=>{
 for(const explicit of [false,true]){
  const f=fixture();f.receipt.snapshot.area={type:'cone',distance:60};f.game.user.targets=new Set([{document:f.tokens[0]}]);f.owner.active=true;f.owner.targets=new Set([{document:f.tokens[2]}]);
  const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid)});
  const data={flags:{pf2e:{origin:{uuid:f.power.uuid},context:{type:'damage-roll',options:[`${ID}:metapower:channel:channel`]}},...(explicit?{'pf2e-toolbelt':{targetHelper:{targets:[f.tokens[3].uuid]}}}:{})}};
  let result;await provider.interceptDamageMessage({instances:[{type:'electricity',total:13}],options:{}},data,{},async d=>{result=d});
  const expected=[f.tokens[explicit?3:2].uuid];assert.deepEqual(result.flags[ID].electricitySource.targetUuids,expected);
  assert.deepEqual(result.flags['pf2e-toolbelt'].targetHelper.targets,expected);
  assert.deepEqual(f.receipt.selection.targetUuids,[f.tokens[1].uuid],'original admission is unchanged');
 }
});
test('a GM area damage click never substitutes GM targets when the source owner is offline',async()=>{
 const f=fixture();f.receipt.snapshot.area={type:'cone',distance:60};f.game.user.targets=new Set([{document:f.tokens[0]}]);f.owner.active=false;
 const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid)});let result;
 await provider.interceptDamageMessage({instances:[{type:'electricity',total:13}],options:{}},{flags:{pf2e:{origin:{uuid:f.power.uuid},context:{options:[`${ID}:metapower:channel:channel`]}}}},{},async d=>{result=d});
 assert.deepEqual(result.flags['pf2e-toolbelt'].targetHelper.targets,[]);
});
test('native damage draft survives delayed creation and keeps template recipients or fills the original empty manifest',async()=>{
 for(const keepNative of [false,true])for(const draftFirst of [false,true]){
  const f=fixture();f.owner.active=true;f.owner.targets=new Set([{document:f.tokens[2]}]);f.receipt.snapshot.area={type:'cone'};
  const hooks=new Map();let sequence=0;const Hooks={on(name,fn){const id=++sequence;const list=hooks.get(name)??new Map();list.set(id,fn);hooks.set(name,list);return id},off(name,id){hooks.get(name)?.delete(id)}};
  const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid)});provider.register({Hooks});
  // Toolbelt registers its one-shot source-card handoff before opening native damage.
  Hooks.on('preCreateChatMessage',doc=>{doc.flags['pf2e-toolbelt']={targetHelper:{targets:keepNative?[f.tokens[1].uuid,f.tokens[3].uuid]:[]}}});
  // Toolbelt's registerUpstreamHook moves its original-card handoff to the front.
  const listeners=hooks.get('preCreateChatMessage'),last=[...listeners].at(-1);listeners.delete(last[0]);hooks.set('preCreateChatMessage',new Map([last,...listeners]));
  const baseline=hooks.get('preCreateChatMessage').size;let saved;
  function create(data){
   const document={...structuredClone(data),updateSource(changes){for(const [key,value]of Object.entries(changes)){const parts=key.split('.');let at=this;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=structuredClone(value)}}};
   for(const fn of hooks.get('preCreateChatMessage').values())fn(document,data,{},f.gm.id);return document;
  }
  const result=await provider.interceptDamageMessage({instances:[{type:'electricity',total:13}],options:{}},{flags:{pf2e:{origin:{uuid:f.power.uuid},context:{type:'damage-roll',options:[`${ID}:metapower:channel:channel`]}}}},{create:!draftFirst},async data=>{
   return draftFirst?structuredClone(data):create(data);
  });
  // PF2e DamagePF2e.roll uses toMessage({create:false}) then ChatMessage.create.
  saved=draftFirst?create(result):result;
  const expected=keepNative?[f.tokens[1].uuid,f.tokens[3].uuid]:[f.tokens[2].uuid];
  assert.deepEqual(saved.flags['pf2e-toolbelt'].targetHelper.targets,expected);assert.deepEqual(saved.flags[ID].electricitySource.targetUuids,expected);
  assert.equal(hooks.get('preCreateChatMessage').size,baseline,'temporary publication observer is removed');
 }
});
test('single-target channel damage retains the admitted target after the caster retargets',async()=>{
 const f=fixture();f.game.user.targets=new Set([{document:f.tokens[0]}]);
 const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid)});let result;
 await provider.interceptDamageMessage({instances:[{type:'electricity',total:13}],options:{}},{flags:{pf2e:{origin:{uuid:f.power.uuid},context:{options:[`${ID}:metapower:channel:channel`]}}}},{},async d=>{result=d});
 assert.deepEqual(result.flags['pf2e-toolbelt'].targetHelper.targets,[f.tokens[1].uuid]);
});

for(const status of ['restricted','manual'])test(`provider forwards ${status} restriction to its real Reactive Chain ledger`,async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);await(await f.damage()).finish();f.power.sourceId=S.chain;f.caster.getActiveTokens=()=>[f.tokens[0]];
 let current=status;const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),reactionRestriction:actor=>{assert.equal(actor,f.caster);return {status:current}}});
 const context={item:f.power,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'};
 await assert.rejects(provider.beforeChannel(context),/反应/);current='clear';
 const result=await provider.beforeChannel(context);assert.equal(result.triggerDamage,13);assert.equal(result.electricityEvidence.targetUuid,f.tokens[2].uuid);
 current=status;await assert.rejects(provider.validateSelection({actor:f.caster,item:f.power,selection:result,kind:'normal',user:f.owner}),/Reactive Chain/);
});

test('Destructive Block final native proof overrides earlier IWR amount; uncertain or mismatched proof cannot trigger lifecycle',async()=>{
 for(const scenario of ['absorbed','partial','uncertain','wrong-nonce','missing-proof']){
  const f=fixture();f.target.hitPoints={max:20,value:20};f.item(f.target,'charge',S.charged,{system:{badge:{value:2}}});f.item(f.other,'shock',S.shocked);
  const hooks={},provider=createEldamonElectricityProvider({game:f.game,fromUuid:async id=>f.docs.get(id)});provider.register({Hooks:{on:(name,fn)=>{hooks[name]=fn;}}});
  const source=f.source(),prepared=await provider.beforeDamage(f.target,{damage:source.rolls[0],token:f.tokens[1],item:f.power});
  provider.observeNativeIWR(f.target,prepared.params,{},prepared.params.rollOptions,{actorDamage:10,shieldDamage:0},false);
  const nonce='destructive123',calculation=destructiveBlockAmounts({incoming:10,shieldHardness:scenario==='partial'?4:5,shieldHP:100});
  const block={kind:'destructive-block',nonce:scenario==='wrong-nonce'?'othernonce':nonce,shieldId:'shield',...calculation,blocked:true,...(scenario==='uncertain'?{uncertain:true}:{})};
  const data={id:'blocked',uuid:'ChatMessage.blocked',author:f.gm,speaker:{actor:f.target.id,scene:'s',token:f.target.id},flags:{[ID]:scenario==='missing-proof'?{}:{shieldBlock:block},pf2e:{origin:{uuid:f.power.uuid},context:{type:'damage-taken',options:[...prepared.params.rollOptions,`${ID}:destructive-block:${nonce}`]},appliedDamage:{uuid:f.target.uuid,isHealing:false,shield:{id:'shield',damage:calculation.shieldDamage},updates:[]}}}};
  const message={...data,updateSource(patch){this.flags[ID].electricityApplied=patch[`flags.${ID}.electricityApplied`];}};
  hooks.preCreateChatMessage(message,data,{},f.gm.id);f.game.messages.set(message.id,message);f.docs.set(message.uuid,message);hooks.createChatMessage(message,{},f.gm.id);
  await provider.afterDamage(prepared.receipt,{applied:true,uncertain:false});
  const expected=scenario==='absorbed'?0:scenario==='partial'?2:null,record=electricityState(f.target).damage[prepared.receipt.nonce];
  assert.equal(record.electricityAmount,expected,scenario);assert.equal(electricityEffects(f.target,S.charged)[0].system.badge.value,expected>0?1:2,scenario);
  const candidates=await f.ledger().candidates({actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'},f.owner);
  assert.equal(candidates.length,expected>0?1:0,scenario);if(expected>0){
   assert.equal(candidates[0].amount,expected);message.flags[ID].shieldBlock.actorDamage=0;
   assert.equal((await f.ledger().candidates({actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'},f.owner)).length,0,'changed final native proof');
  }
 }
});

test('native damage publication stamps durable source identity shared by altered target rolls',async()=>{
 const gm={id:'gm',targets:new Set([{document:{uuid:'Scene.s.Token.a'}}])},game={user:gm,users:{activeGM:gm},messages:new Map()};
 const provider=createEldamonElectricityProvider({game,fromUuid:async()=>null});
 assert.equal(typeof provider.interceptDamageMessage,'function');
 const roll={instances:[{type:'electricity',total:13}],options:{}},data={flags:{pf2e:{context:{type:'damage-roll',options:[]}}}};
 let seen;
 await provider.interceptDamageMessage(roll,data,{},async(d)=>{seen=d;return {id:'m'}});
 const source=seen.flags[ID].electricitySource;
 assert.ok(source.nonce);assert.deepEqual(source.targetUuids,['Scene.s.Token.a']);
 assert.equal(roll.options[ID].electricitySource.nonce,source.nonce);
 const result=preserveElectricityOnAlter(roll,{options:{}});
 assert.deepEqual(result.options[ID].electricitySource,roll.options[ID].electricitySource);
 assert.notEqual(result.options[ID].electricitySource,roll.options[ID].electricitySource);
});

test('electricity trait on an attack is not damage-type evidence for Shocked off-guard',async()=>{
 const target={items:new Map([['shock',{sourceId:'Compendium.battlezoo-eldamon-pf2e.conditions.Item.1fZbuJEbVmE3J4XL'}]]),attributes:{ac:{modifiers:[]}}};
 const provider=createEldamonElectricityProvider({game:{user:{id:'gm'},users:{activeGM:{id:'gm'}},messages:new Map()},fromUuid:async()=>null});
 assert.equal(typeof provider.interceptCheck,'function');
 const native=(_check,c)=>c,base={type:'attack-roll',target:{actor:target},dc:{value:20},options:new Set(['electricity']),item:{type:'weapon',system:{damage:{damageType:'slashing'},traits:{value:['electricity']}}}};
 assert.equal((await provider.interceptCheck(native,{},base)).dc.value,20);
 const actual={...base,item:{type:'weapon',system:{damage:{damageType:'electricity'}}}};
 const adjusted=await provider.interceptCheck(native,{},actual);
 assert.equal(adjusted.dc.value,18);assert.ok(adjusted.options.has('target:condition:off-guard'));
 target.attributes.ac.modifiers=[{type:'circumstance',modifier:-2,enabled:true}];
 assert.equal((await provider.interceptCheck(native,{},{...actual,dc:{value:18}})).dc.value,18);
});
test('actual application scope carries IWR net amount into one matching native receipt and restores after reload',async()=>{
 const f=fixture();f.target.hitPoints={max:20,value:4};const hooks=new Map(),Hooks={on:(name,fn)=>{const entries=hooks.get(name)??[];entries.push(fn);hooks.set(name,entries);}},provider=createEldamonElectricityProvider({game:f.game,fromUuid:async id=>f.docs.get(id)});
 provider.register({Hooks});const source=f.source(),damage=source.rolls[0];
 const prepared=await provider.beforeDamage(f.target,{damage,token:f.tokens[1],item:f.power,rollOptions:new Set()});assert.ok(prepared.receipt);
 provider.observeNativeIWR(f.target,prepared.params,{},prepared.params.rollOptions,{actorDamage:13,shieldDamage:0},false);
 const data={id:'actualreceipt',uuid:'ChatMessage.actualreceipt',author:f.gm,speaker:{actor:f.target.id,scene:'s',token:f.target.id},flags:{pf2e:{origin:{uuid:f.power.uuid},context:{type:'damage-taken',options:[...prepared.params.rollOptions]},appliedDamage:{uuid:f.target.uuid,isHealing:false,updates:[{path:'system.attributes.hp.value',value:4}]}}}};
 const message={...data,updateSource(patch){this.flags[ID]={electricityApplied:patch[`flags.${ID}.electricityApplied`]};}};
 for(const hook of hooks.get('preCreateChatMessage'))hook(message,data,{},f.gm.id);
 assert.equal(message.flags[ID].electricityApplied.amount,13);
 f.game.messages.set(message.id,message);f.docs.set(message.uuid,message);
 for(const hook of hooks.get('createChatMessage'))hook(message,{},f.gm.id);
 await provider.afterDamage(prepared.receipt,{applied:true,uncertain:false});
 const record=f.target.flags[ID].electricity.damage[prepared.receipt.nonce];assert.equal(record.electricityAmount,13);assert.equal(record.status,'confirmed');
 await createEldamonElectricityProvider({game:f.game,fromUuid:async id=>f.docs.get(id)}).maintain(f.target);assert.equal(f.target.flags[ID].electricity.damage[prepared.receipt.nonce].electricityAmount,13);
});
test('suppressed native IWR reports zero; persistent and untyped applications never enter the electricity provider',async()=>{
 const f=fixture(),provider=createEldamonElectricityProvider({game:f.game,fromUuid:async id=>f.docs.get(id)});
 for(const damage of [{instances:[{type:'electricity',persistent:true,total:10}]},{instances:[{type:'untyped',total:10}]}])assert.equal(await provider.beforeDamage(f.target,{damage,token:f.tokens[1],item:f.power}),null);
 const source=f.source();f.target.hitPoints={max:20};const hooks={},Hooks={on:(key,fn)=>{hooks[key]=fn;}};provider.register({Hooks});
 const prepared=await provider.beforeDamage(f.target,{damage:source.rolls[0],token:f.tokens[1],item:f.power});
 provider.observeNativeIWR(f.target,prepared.params,{},prepared.params.rollOptions,{actorDamage:13,shieldDamage:0},true);
 let proof;hooks.preCreateChatMessage({updateSource:p=>{proof=p;}},{speaker:{actor:f.target.id,scene:'s',token:f.target.id},flags:{pf2e:{context:{type:'damage-taken',options:[...prepared.params.rollOptions]}}}},{},f.gm.id);
 assert.equal(proof[`flags.${ID}.electricityApplied`].amount,0);
});
test('active-GM channel delivery retains the original card author even after that player disconnects',async()=>{
 const f=fixture();f.owner.active=false;
 const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async id=>f.docs.get(id)});
 await provider.onCommittedChannel({receipt:f.receipt,message:f.card,user:f.owner});
 assert.equal([...f.caster.items.values()].filter(i=>i.sourceId==='Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR').length,1);
 await provider.onCommittedChannel({receipt:f.receipt,message:f.card});
 assert.equal([...f.caster.items.values()].filter(i=>i.sourceId==='Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR').length,1);
});
