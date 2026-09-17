import test from 'node:test';
import assert from 'node:assert/strict';
import {METAPOWER_SOURCES,POWER_PROFILES} from '../scripts/metapower/rules.mjs';
let api={};try{api=await import('../scripts/metapower/lifecycle.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const ID='pf2e-third-party-automation';
function fixture(){
 const user={id:'owner'},gm={id:'gm'},game={user:gm,users:{activeGM:gm},combat:{id:'combat',round:1,turn:0,combatant:{id:'turn',actor:{uuid:'Actor.a'}}}};
 const actor={uuid:'Actor.a',id:'a',type:'character',level:5,flags:{},items:new Map(),testUserPermission:u=>u===user||u===gm,getRollOptions:()=>['active-power-one:electric-surge'],async update(data){for(const[key,value]of Object.entries(data))if(key===`flags.${ID}.metapower`)this.flags[ID]={...this.flags[ID],metapower:structuredClone(value)}}};
 const item=(id,source)=>{const i={id,uuid:`Actor.a.Item.${id}`,sourceId:source,type:'feat',actor,system:{traits:{value:[]},frequency:{value:1}}};actor.items.set(id,i);return i};
 const siphon=item('s',METAPOWER_SOURCES.siphoning),widen=item('w',METAPOWER_SOURCES.widen),power=item('p',Object.values(POWER_PROFILES).find(p=>p.id==='electric-surge').sourceUuid);
 const documents=new Map([[actor.uuid,actor],...[...actor.items.values()].map(i=>[i.uuid,i])]);
 return {game,actor,user,siphon,widen,power,documents,service:()=>api.createMetapowerLedger({game,fromUuid:async uuid=>documents.get(uuid)})};
}
const begin=(service,f,item,nonce,extra={})=>service.begin({actorUuid:f.actor.uuid,itemUuid:item?.uuid??null,nonce,...extra},f.user);
async function finish(service,f,receipt,item){const message={id:'m'+receipt.nonce,uuid:'ChatMessage.m'+receipt.nonce,speaker:{actor:f.actor.id},author:f.user,flags:{pf2e:{origin:{uuid:item.uuid}},[ID]:{metapowerUse:{nonce:receipt.nonce,actorUuid:f.actor.uuid,itemUuid:item.uuid}}}};f.documents.set(message.uuid,message);return service.finish({actorUuid:f.actor.uuid,nonce:receipt.nonce,messageUuid:message.uuid,status:'committed'},f.user)}
test('durable original-card activation, replay binding and replacement',async()=>{
 assert.equal(typeof api.createMetapowerLedger,'function');const f=fixture(),s=f.service();const r=await begin(s,f,f.siphon,'one');await finish(s,f,r,f.siphon);
 assert.equal(f.actor.flags[ID].metapower.armed.nonce,'one');
 assert.equal((await finish(f.service(),f,r,f.siphon)).status,'committed');
 await assert.rejects(s.finish({actorUuid:f.actor.uuid,nonce:'one',messageUuid:'ChatMessage.other',status:'committed'},f.user),/binding|card/i);
 const next=await begin(s,f,f.widen,'two');await finish(s,f,next,f.widen);assert.equal(f.actor.flags[ID].metapower.armed.kind,'widen');
});
test('one outstanding lease rejects another client; pre-native cancellation preserves activation',async()=>{
 const f=fixture(),s=f.service();await finish(s,f,await begin(s,f,f.siphon,'one'),f.siphon);
 await begin(s,f,f.power,'two',{selection:{discharge:false,baseDistance:30}});
 await assert.rejects(begin(f.service(),f,f.widen,'three'),/progress|pending/i);
 await s.finish({actorUuid:f.actor.uuid,nonce:'two',status:'cancelled'},f.user);
 assert.equal(f.actor.flags[ID].metapower.armed.nonce,'one');
});
test('started uncertain actions cannot refund and explicit cancel clears only captured nonce',async()=>{
 const f=fixture(),s=f.service();await finish(s,f,await begin(s,f,f.siphon,'one'),f.siphon);
 await begin(s,f,null,'two');await s.start({actorUuid:f.actor.uuid,nonce:'two'},f.user);
 await assert.rejects(s.finish({actorUuid:f.actor.uuid,nonce:'two',status:'cancelled'},f.user),/started|cancellation/i);
 await s.finish({actorUuid:f.actor.uuid,nonce:'two',status:'uncertain'},f.user);assert.equal(f.actor.flags[ID].metapower.armed,null);
 const r=await begin(s,f,f.widen,'three');await finish(s,f,r,f.widen);
 await s.clear({actorUuid:f.actor.uuid,activationNonce:'one'},f.user);assert.equal(f.actor.flags[ID].metapower.armed.nonce,'three');
});
test('turn admission rejects stale activation, ownership and inactive GM',async()=>{
 const f=fixture(),s=f.service();await finish(s,f,await begin(s,f,f.siphon,'one'),f.siphon);f.game.combat.turn=1;
 const r=await begin(s,f,f.power,'two',{selection:{discharge:false}});assert.equal(r.snapshot,null);
 await assert.rejects(s.clear({actorUuid:f.actor.uuid}, {id:'stranger'}),/owner|permission/i);
 f.game.users.activeGM={id:'other'};await assert.rejects(s.clear({actorUuid:f.actor.uuid},f.user),/GM/i);
});
test('prepared options and frequency remain native gates; High Voltage has explicit no-refresh snapshot',async()=>{
 const f=fixture(),s=f.service();f.actor.getRollOptions=()=>[];await assert.rejects(begin(s,f,f.power,'a'),/prepared/i);
 f.actor.getRollOptions=()=>['active-power-one:electric-surge'];f.power.system.frequency.value=0;await assert.rejects(begin(s,f,f.power,'b'),/frequency|depleted/i);
 f.power.system.frequency.value=1;await finish(s,f,await begin(s,f,f.siphon,'one'),f.siphon);
 f.power.sourceId=Object.values(POWER_PROFILES).find(p=>p.id==='high-voltage').sourceUuid;f.actor.getRollOptions=()=>['active-power-refresh:high-voltage'];
 const r=await begin(s,f,f.power,'c');assert.equal(r.snapshot.siphon.applies,true);assert.equal(r.snapshot.policy.highVoltage,'convert');assert.deepEqual(r.snapshot.suppressEffects,['refresh']);
});
test('selected discharge requires Charged and pays exactly once only on original card completion',async()=>{
 const f=fixture(),s=f.service();await finish(s,f,await begin(s,f,f.siphon,'one'),f.siphon);
 await assert.rejects(begin(s,f,f.power,'bad',{selection:{discharge:true,baseDistance:60}}),/Charged/i);
 const charge={id:'charge',uuid:'Actor.a.Item.charge',sourceId:'Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR',system:{badge:{value:2}},flags:{},async update(p){this.system.badge.value=p['system.badge.value'];this.flags[ID]={payment:p[`flags.${ID}.payment`]}}};f.actor.items.set(charge.id,charge);f.documents.set(charge.uuid,charge);
 const r=await begin(s,f,f.power,'two',{selection:{discharge:true,baseDistance:60}});assert.equal(charge.system.badge.value,2);
 await finish(s,f,r,f.power);assert.equal(charge.system.badge.value,1);await finish(f.service(),f,r,f.power);assert.equal(charge.system.badge.value,1);
});
test('immutable channel receipt binds activation, source, branch and original card',async()=>{
 const f=fixture(),s=f.service();await finish(s,f,await begin(s,f,f.widen,'one'),f.widen);
 const selection={discharge:false,baseDistance:30};const r=await begin(s,f,f.power,'two',{selection});selection.baseDistance=20;
 assert.equal(r.snapshot.area.distance,40);await finish(s,f,r,f.power);assert.equal(f.actor.flags[ID].metapower.armed,null);
 assert.equal((await begin(f.service(),f,f.power,'two',{selection:{baseDistance:20}})).snapshot.area.distance,40);
});
