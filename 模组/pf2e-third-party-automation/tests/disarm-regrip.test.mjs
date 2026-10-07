import test from 'node:test';
import assert from 'node:assert/strict';
import {createDisarmRegrip,INTERACT_SOURCE} from '../scripts/disarm-regrip.mjs';
import {createDisarmingBlock,getWeakenedGrasp} from '../scripts/disarming-block.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';

function assign(document,changes){for(const[path,value]of Object.entries(changes)){const keys=path.split('.');let parent=document;for(const key of keys.slice(0,-1))parent=parent[key]??={};parent[keys.at(-1)]=structuredClone(value);}return document;}
function fixture({chooseOwner}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',isGM:false,active:true},users=new Map([[gm.id,gm],[player.id,player]]);users.activeGM=gm;
 let owned=true;const writes=[],choices=[],actor={id:'actor',uuid:'Actor.actor',type:'character',items:new Map(),flags:{},testUserPermission:(user,permission)=>permission==='OWNER'&&(user===gm||user===player&&owned)};
 const game={user:player,users,actors:new Map([[actor.id,actor]]),messages:new Map(),time:{worldTime:100}},docs=new Map([[actor.uuid,actor]]);
 const item={id:'interact',uuid:actor.uuid+'.Item.interact',type:'action',sourceId:INTERACT_SOURCE,actor,system:{actionType:{value:'action'},actions:{value:1},traits:{value:['manipulate']}}};
 const weapon={id:'weapon',uuid:actor.uuid+'.Item.weapon',name:'Sword',type:'weapon',actor,flags:{},system:{equipped:{carryType:'held',handsHeld:1},category:'martial',traits:{value:[]}}};
 weapon.update=async changes=>{writes.push({type:'weapon',changes:structuredClone(changes)});return assign(weapon,changes)};
 actor.deleteEmbeddedDocuments=async(type,ids)=>{writes.push({type:'effect-delete',documentType:type,ids:[...ids],clearedChecks:structuredClone(weapon.flags[M]?.disarmingBlock?.clearedChecks)});for(const id of ids){const effect=actor.items.get(id);actor.items.delete(id);docs.delete(effect?.uuid)}return []};
 for(const document of [item,weapon]){actor.items.set(document.id,document);docs.set(document.uuid,document)}
 const addGrasp=checkId=>{
  const effect={id:'grasp',uuid:actor.uuid+'.Item.grasp',type:'effect',actor,isExpired:false,system:{duration:{value:-1,unit:'unlimited',expiry:null,sustained:false},context:{origin:{actor:'Actor.attacker',item:null,token:null}},rules:[{key:'FlatModifier',selector:['weapon-attack'],type:'circumstance',value:-2}]},flags:{[M]:{nativeEffectKey:'disarming-block:grasp:'+weapon.uuid,disarmingBlock:{kind:'weakened-grasp',weaponUuid:weapon.uuid,nonce:'ordinary-disarm',checkId,attackIds:[weapon.id],handsHeld:1}}}};
  actor.items.set(effect.id,effect);docs.set(effect.uuid,effect);return effect;
 };
 const grasp=addGrasp('original-check'),fromUuid=async uuid=>docs.get(uuid),processor=createDisarmingBlock({game,fromUuid,canvas:{tokens:{placeables:[]}}});
 const provider=createDisarmRegrip({game,fromUuid,processor,chooseOwner:async request=>{choices.push(request);return chooseOwner?chooseOwner(request):weapon.uuid}});
 const captured=provider.captureUsage(item),card={id:'interact-card',uuid:'ChatMessage.interact-card',author:player,actor,speaker:{actor:actor.id},rolls:[],isRoll:false,flags:{pf2e:{origin:{uuid:item.uuid,type:'action',actor:actor.uuid}},[M]:captured}};
 card.update=async changes=>{assign(card,changes);writes.push({type:'card',state:structuredClone(card.flags[M].disarmRegrip)});return card};
 game.messages.set(card.id,card);game.user=gm;
 const execute=()=>provider.executeUsage({actor,item,message:card,user:player});
 return {game,gm,player,actor,item,weapon,grasp,card,captured,provider,processor,writes,choices,addGrasp,execute,setOwned:value=>owned=value};
}

test('Disarm Interact captured by its Player clears the original grasp once through the real processor',async()=>{
 const f=fixture(),input=f.captured.disarmRegripInput;
 assert.equal(input.mode,'item');assert.equal(input.userId,f.player.id);assert.equal(input.source,INTERACT_SOURCE);assert.equal(input.itemUuid,f.item.uuid);
 assert.deepEqual(input.candidates,[{weaponUuid:f.weapon.uuid,graspCheckId:'original-check'}]);assert.equal(getWeakenedGrasp(f.actor,f.weapon.uuid),f.grasp);
 const results=await Promise.all([f.execute(),f.execute()]);assert.ok(results.every(result=>/已调整握持/.test(result)));
 assert.deepEqual(f.writes.filter(w=>w.type==='card').map(w=>[w.state.status,w.state.result]),[['offered',undefined],['claimed',undefined],['done','cleared']]);
 assert.deepEqual(f.weapon.flags[M].disarmingBlock.clearedChecks,['original-check']);assert.equal(getWeakenedGrasp(f.actor,f.weapon.uuid),null);assert.equal(f.choices.length,1);
 assert.deepEqual(f.writes.map(w=>w.type),['card','card','weapon','effect-delete','card']);assert.deepEqual(f.writes.find(w=>w.type==='effect-delete').clearedChecks,['original-check']);
 f.addGrasp('later-check');await f.execute();assert.equal(getWeakenedGrasp(f.actor,f.weapon.uuid).flags[M].disarmingBlock.checkId,'later-check');assert.deepEqual(f.weapon.flags[M].disarmingBlock.clearedChecks,['original-check']);assert.equal(f.writes.filter(w=>w.type==='effect-delete').length,1);
});

test('Disarm Interact cancellation saves done and cancelled without clearing its grasp',async()=>{
 const f=fixture({chooseOwner:async()=>null});assert.match(await f.execute(),/未选择/);
 assert.equal(f.card.flags[M].disarmRegrip.status,'done');assert.equal(f.card.flags[M].disarmRegrip.result,'cancelled');assert.equal(getWeakenedGrasp(f.actor,f.weapon.uuid),f.grasp);
 await f.execute();assert.equal(f.choices.length,1);assert.deepEqual(f.writes.map(w=>w.type),['card','card']);assert.equal(f.weapon.flags[M],undefined);
});

test('Disarm Interact chooser await cannot clear a later grasp check',async()=>{
 const f=fixture({chooseOwner:async()=>{f.addGrasp('later-check');return f.weapon.uuid}});assert.match(await f.execute(),/已经结束/);
 assert.equal(f.card.flags[M].disarmRegrip.status,'done');assert.equal(f.card.flags[M].disarmRegrip.result,'superseded');assert.equal(getWeakenedGrasp(f.actor,f.weapon.uuid).flags[M].disarmingBlock.checkId,'later-check');
 assert.deepEqual(f.writes.map(w=>w.type),['card','card']);assert.equal(f.weapon.flags[M],undefined);
});

test('Disarm Interact chooser await losing OWNER rejects before claiming or clearing',async()=>{
 const f=fixture({chooseOwner:async()=>{f.setOwned(false);return f.weapon.uuid}});await assert.rejects(f.execute(),/所有者权限/);
 assert.equal(f.card.flags[M].disarmRegrip.status,'offered');assert.equal(getWeakenedGrasp(f.actor,f.weapon.uuid),f.grasp);assert.deepEqual(f.writes.map(w=>w.type),['card']);assert.equal(f.weapon.flags[M],undefined);
});

test('Disarm processor rejects a claimed proof changed during the card update',async()=>{
 const f=fixture(),update=f.card.update;f.card.update=async changes=>{const card=await update(changes);if(card.flags[M].disarmRegrip.status==='claimed')card.flags[M].disarmRegrip.graspCheckId='other-check';return card};
 await assert.rejects(f.execute(),/准确认领/);assert.equal(getWeakenedGrasp(f.actor,f.weapon.uuid),f.grasp);assert.equal(f.weapon.flags[M],undefined);assert.deepEqual(f.writes.map(w=>w.type),['card','card']);
});

for(const [label,change]of [['a roll',f=>f.card.rolls.push({})],['a different origin',f=>f.card.flags.pf2e.origin.uuid='Actor.actor.Item.other'],['a replaced card',f=>f.game.messages.set(f.card.id,{...f.card})]])test(`Disarm Interact with ${label} rejects before offering or clearing`,async()=>{
 const f=fixture();change(f);await assert.rejects(f.execute());assert.deepEqual(f.writes,[]);assert.equal(f.choices.length,0);assert.equal(getWeakenedGrasp(f.actor,f.weapon.uuid),f.grasp);
});
