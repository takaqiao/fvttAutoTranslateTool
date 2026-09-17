import test from 'node:test';
import assert from 'node:assert/strict';
import {METAPOWER_SOURCES} from '../scripts/metapower/rules.mjs';
let api={};try{api=await import('../scripts/metapower/provider.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const ID='pf2e-third-party-automation';
test('Widen metadata repair is exact-source, owned and idempotent',async()=>{
 assert.equal(typeof api.createMetapowerProvider,'function');const gm={id:'gm'},game={user:gm,users:{activeGM:gm}},changes=[];
 const item={id:'w',sourceId:METAPOWER_SOURCES.widen,system:{actionType:{value:'passive'},actions:{value:null}},async update(data){changes.push(data);this.system.actionType.value=data['system.actionType.value'];this.system.actions.value=data['system.actions.value']}};
 const actor={type:'character',items:new Map([['w',item],['fake',{sourceId:'fake',name:'Widen Element'}]]),testUserPermission:()=>true};
 const p=api.createMetapowerProvider({game,fromUuid:()=>{}});await p.maintain(actor);await p.maintain(actor);assert.equal(changes.length,1);assert.equal(item.system.actions.value,1);
});
test('target coefficient uses bound immutable source card and creature traits only before native IWR',async()=>{
 const snapshot={kind:'siphoning',siphon:{applies:true},disruptive:true,associatedTraits:['electricity'],actorUuid:'Actor.a',itemUuid:'Actor.a.Item.i'},receipt={nonce:'n',status:'committed',messageUuid:'ChatMessage.c',snapshot};
 const actor={uuid:'Actor.a',flags:{[ID]:{metapower:{receipts:{n:receipt}}}}};
 const card={id:'c',uuid:'ChatMessage.c',flags:{[ID]:{metapowerUse:{nonce:'n',actorUuid:actor.uuid,itemUuid:snapshot.itemUuid}},pf2e:{origin:{uuid:snapshot.itemUuid}}}};
 const game={messages:new Map([['c',card]])},p=api.createMetapowerProvider({game,fromUuid:async()=>actor});
 const multipliers=[],proof={actorUuid:actor.uuid,cardId:'c',nonce:'n'},damage={options:{[ID]:{metapowerDamage:proof}},alter(m){multipliers.push(m);return {scaled:m}}};
 const full=await p.beforeDamage({traits:new Set(['electricity'])},{damage});assert.equal(full.params.damage,damage);
 const half=await p.beforeDamage({traits:new Set(['humanoid']),system:{attributes:{immunities:[{type:'electricity'}]}}},{damage});assert.equal(half.params.damage.scaled,.5);assert.deepEqual(multipliers,[.5]);
 card.flags.pf2e.origin.uuid='forged';await assert.rejects(p.beforeDamage({traits:new Set()},{damage}),/source|binding/i);
});
test('retained discharge degree downgrade augments native save arithmetic and preserves other adjustments',()=>{
 const context={type:'saving-throw',dosAdjustments:[{adjustments:{success:{label:'native',amount:1}}}]};
 const result=api.adjustMetapowerCheckContext({saveDowngrade:1},context);
 assert.equal(result.dosAdjustments[0],context.dosAdjustments[0]);assert.deepEqual(result.dosAdjustments[1].adjustments.all,{label:'Retributive Shock · Discharge',amount:-1});
 assert.equal(context.dosAdjustments.length,1);assert.equal(api.adjustMetapowerCheckContext({saveDowngrade:1},{type:'attack-roll'}).dosAdjustments,undefined);
});
test('Electric Shot half-failure application is confined to its bound recipient through native alter',async()=>{
 const snapshot={powerId:'electric-shot',itemUuid:'Actor.a.Item.i'},actor={uuid:'Actor.a',flags:{[ID]:{metapower:{receipts:{n:{status:'committed',messageUuid:'ChatMessage.c',snapshot}}}}}},card={uuid:'ChatMessage.c',flags:{[ID]:{metapowerUse:{nonce:'n'}},pf2e:{origin:{uuid:snapshot.itemUuid}}}};
 const proof={actorUuid:actor.uuid,cardId:'c',nonce:'n',targetActorUuid:'Actor.target'},damage={options:{[ID]:{metapowerShotFailure:proof}}},p=api.createMetapowerProvider({game:{messages:new Map([['c',card]])},fromUuid:async()=>actor});
 assert.equal(await p.beforeDamage({uuid:'Actor.target'},{damage}),null);await assert.rejects(p.beforeDamage({uuid:'Actor.other'},{damage}),/recipient/i);
 const altered=api.preserveMetapowerOnAlter(damage,{options:{}});assert.deepEqual(altered.options[ID].metapowerShotFailure,proof);assert.notEqual(altered.options[ID].metapowerShotFailure,proof);
});
