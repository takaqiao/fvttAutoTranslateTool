import test from 'node:test';
import assert from 'node:assert/strict';
import {createDeflectionWeapons} from '../scripts/transcendent-deflection-weapons.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';

function patch(document,changes){
 for(const [path,value]of Object.entries(changes)){
  const keys=path.split('.');let at=document;
  for(const key of keys.slice(0,-1))at=at[key]??={};
  at[keys.at(-1)]=structuredClone(value);
 }
}
function fixture({quantity=2,max=20,alternate=false}={}){
 const gm={id:'gm',isGM:true},game={user:gm,users:{activeGM:gm}},nonce='original-paid-reaction';
 const claim={nonce,claimKey:'deflect:'+nonce,actorUuid:'Actor.deflector',weaponUuid:'Actor.deflector.Item.weapon',state:'claimed'};
 const actor={uuid:claim.actorUuid,flags:{[M]:{transcendentDeflection:{reactions:[structuredClone(claim)]}}},items:new Map(),system:{actions:[]}};
 const source={_id:'weapon',name:'Sword',type:'weapon',system:{category:'martial',quantity,hp:{max,value:max},usage:{hands:1},equipped:{carryType:'held',handsHeld:1},traits:{value:[]},containerId:null},flags:{},_stats:{compendiumSource:'Compendium.test.Item.sword'}};
 let writes=0,onClaim=()=>{};const created=[];
 const weapon={id:source._id,uuid:claim.weaponUuid,actor,type:'weapon',_source:source,system:source.system,flags:source.flags,isMelee:!alternate,isRanged:alternate,isBroken:false,isDestroyed:false,toObject(){return structuredClone(this._source)},async update(changes){writes++;patch(source,changes);this.system=source.system;this.flags=source.flags;}};
 actor.items.set(weapon.id,weapon);
 if(alternate)actor.system.actions=[{item:weapon,altUsages:[{item:{...weapon,isMelee:true,isRanged:false,system:{...weapon.system,traits:{value:['finesse']}}}}]}];
 actor.update=async changes=>{patch(actor,changes);if(actor.flags[M].transcendentDeflection.reactions[0].weaponState==='breaking')await onClaim()};
 actor.createEmbeddedDocuments=async(_kind,items)=>{created.push(...structuredClone(items));return items.map(data=>({...data,id:data._id}))};
 const service=createDeflectionWeapons({game});
 return {game,actor,weapon,source,claim,created,service,onClaim:fn=>onClaim=fn,writes:()=>writes,current:()=>actor.flags[M].transcendentDeflection.reactions[0],run:()=>service.breakWeapon({actor,weapon,claim})};
}
for(const change of ['quantity','hp','max','carry','name','flags','source','eligibility','alternate','claim-plan','claim-state'])test(`changed ${change} during awaited breaking claim is preserved without physical mutation`,async()=>{
 const f=fixture({alternate:change==='alternate'});f.onClaim(()=>{
  if(change==='quantity')f.weapon.system.quantity=3;
  if(change==='hp')f.weapon.system.hp.value=19;
  if(change==='max')f.weapon.system.hp.max=22;
  if(change==='carry')f.weapon.system.equipped.carryType='worn';
  if(change==='name')f.source.name='Renamed sword';
  if(change==='flags')f.source.flags.another={newValue:true};
  if(change==='source')f.source._stats.compendiumSource='Compendium.test.Item.changed';
  if(change==='eligibility')f.weapon.isBroken=true;
  if(change==='alternate')f.actor.system.actions=[];
  if(change==='claim-plan')f.current().weaponPlan.quantity=3;
  if(change==='claim-state')f.current().state='done';
 });
 await assert.rejects(f.run(),/改变|不确定|合格|认领/);
 assert.equal(f.writes(),0);assert.equal(f.created.length,0);assert.equal(f.current().weaponState,'breaking');
 if(change==='quantity')assert.equal(f.weapon.system.quantity,3);
 if(change==='hp')assert.equal(f.weapon.system.hp.value,19);
 if(change==='name')assert.equal(f.source.name,'Renamed sword');
 await assert.rejects(f.run(),/不确定/);assert.equal(f.writes(),0);assert.equal(f.created.length,0);
});
for(const [quantity,max,alternate]of [[1,20,false],[2,20,false],[3,0,false],[2,20,true]])test(`unchanged ${quantity}-item ${max}-HP ${alternate?'alternate':'ordinary'} weapon completes exactly once`,async()=>{
 const f=fixture({quantity,max,alternate});assert.equal(await f.run(),f.weapon);assert.equal(f.current().weaponState,'broken');assert.equal(f.writes(),1);
 assert.equal(f.weapon.system.quantity,1);assert.equal(f.weapon.system.hp.value,max?Math.floor(max/2):0);
 assert.equal(f.weapon.flags[M].transcendentDeflection.broken.virtualHP,max===0);
 assert.equal(f.created.length,quantity>1?1:0);
 if(quantity>1){const spare=f.created[0];assert.equal(spare.system.quantity,quantity-1);assert.equal(spare.system.hp.value,max);assert.equal(spare.system.equipped.carryType,'worn');assert.equal(spare.system.equipped.handsHeld,0);assert.equal(spare.system.containerId,null);assert.equal(spare.flags[M]?.transcendentDeflection,undefined);}
 assert.equal(await f.run(),f.weapon);assert.equal(f.writes(),1);assert.equal(f.created.length,quantity>1?1:0);
});
test('a physical write of uncertain outcome never permits another split',async()=>{
 const f=fixture();f.weapon.update=async()=>{throw Error('unknown write')};
 await assert.rejects(f.run(),/unknown write/);assert.equal(f.current().weaponState,'breaking');
 await assert.rejects(f.run(),/不确定/);assert.equal(f.created.length,0);
});
test('normal repair removing the broken marker does not replay an old paid split',async()=>{
 const f=fixture();await f.run();delete f.weapon.flags[M].transcendentDeflection.broken;f.weapon.system.hp.value=20;
 await assert.rejects(f.run(),/不确定/);assert.equal(f.writes(),1);assert.equal(f.created.length,1);assert.equal(f.weapon.system.hp.value,20);
});
test('exact item replacement during breaking leaves the original source untouched',async()=>{
 const f=fixture();f.onClaim(()=>f.actor.items.set(f.weapon.id,{...f.weapon}));
 await assert.rejects(f.run(),/认领/);assert.equal(f.writes(),0);assert.equal(f.created.length,0);assert.equal(f.weapon.system.quantity,2);assert.equal(f.current().weaponState,'breaking');
});
test('GM change during the awaited claim cannot mutate the weapon',async()=>{
 const f=fixture();f.onClaim(()=>f.game.users.activeGM={id:'new-gm'});
 await assert.rejects(f.run(),/主GM/);assert.equal(f.writes(),0);assert.equal(f.created.length,0);assert.equal(f.current().weaponState,'breaking');
});
