import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createEldamonElectricityProvider,preserveElectricityOnAlter} from '../scripts/eldamon-electricity-provider.mjs';
import {fixture} from './eldamon-electricity-fixture.mjs';
const ID='pf2e-third-party-automation';

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
