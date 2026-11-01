import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/volatile-spark-reload.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const NS='pf2e-third-party-automation',reload='Compendium.pf2e.actionspf2e.Item.tmbf6xouEnRXCnKR',cascade='Compendium.pf2e.actionspf2e.Item.HbejhIywqIufrmVM',effect='Compendium.pf2e.feat-effects.Item.fsjO5oTKttsbpaKl';
function fixture({cancel=false}={}){
 const calls=[],user={id:'gm',isGM:true,active:true};
 const actor={id:'pc',uuid:'Actor.pc',items:new Map(),system:{actions:[]},testUserPermission:()=>true,async createEmbeddedDocuments(type,data){calls.push(['effect',data]);return data}};
 const gun={id:'gun',actor,type:'weapon',isRanged:true,system:{group:'firearm',equipped:{carryType:'held',handsHeld:1}}},strike={type:'strike',item:gun};actor.items.set(gun.id,gun);actor.system.actions.push(strike);
 const item={id:'reload',uuid:'Actor.pc.Item.reload',actor,type:'action',sourceId:reload,system:{}};actor.items.set(item.id,item);actor.items.set('study',{type:'feat',sourceId:'Compendium.pf2e.classfeatures.Item.aRDNtB0mh6MMRSKf',system:{}});
 const message={id:'use',author:user,speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:item.uuid}},[NS]:{usageInput:{actualUse:true},spellCombinationUse:{state:'reloading',kind:'reload',userId:user.id,weaponId:gun.id}}}};
 const game={user,users:Object.assign(new Map([[user.id,user]]),{activeGM:user}),actors:new Map([[actor.id,actor]]),messages:new Map([[message.id,message]]),modules:new Map([['pf2e-ranged-combat',{active:true}]])};
 const fromUuid=async uuid=>uuid===item.uuid?item:uuid===cascade?{system:{selfEffect:{uuid:effect}}}:uuid===effect?{toObject:()=>({_id:'effect',name:'Arcane Cascade',type:'effect',_stats:{compendiumSource:effect},system:{rules:[{key:'ChoiceSet'}]}})}:null;
 const modules={WeaponSystem:{transformWeapon(a,w){assert.equal(a,actor);assert.equal(w,gun);return {actor,pf2eWeapon:w,reloadActions:1,isRepeating:false}}},Reload:{async reload(weapon,updates,options){assert.equal(weapon.pf2eWeapon,gun);assert.equal(options.fullyReload,false);calls.push(['reload']);return !cancel}},Updates:class{constructor(a){assert.equal(a,actor)}async commit(){await Promise.resolve();calls.push(['commit'])}}};
 assert.equal(typeof api.createVolatileSparkReload,'function');const helper=api.createVolatileSparkReload({game,fromUuid,loadRanged:async()=>modules});
 const run=()=>helper.run({actor,item,message,user,weaponId:gun.id,assertLive(){}});
 return {actor,item,gun,strike,message,game,calls,helper,run,modules};
}
test('Reloading Cascade reloads only its selected firearm, awaits writes, and applies the native selfEffect',async()=>{
 const f=fixture();assert.deepEqual(await f.run(),{status:'done'});assert.deepEqual(f.calls.map(c=>c[0]),['reload','commit','effect']);assert.equal(f.calls[2][1][0]._id,undefined);assert.deepEqual(f.calls[2][1][0].system.rules,[{key:'ChoiceSet'}]);
});
test('cancelled reload does not create a stance or alter Spellstrike charge',async()=>{
 const f=fixture({cancel:true});f.actor.flags={[NS]:{spellstrike:{charged:false}}};assert.deepEqual(await f.run(),{status:'cancelled'});assert.deepEqual(f.calls.map(c=>c[0]),['reload']);assert.equal(f.actor.flags[NS].spellstrike.charged,false);
});
test('existing Arcane Cascade is retained while one Interact reload still occurs',async()=>{
 const f=fixture();f.actor.items.set('stance',{type:'effect',sourceId:effect});await f.run();assert.deepEqual(f.calls.map(c=>c[0]),['reload','commit']);
});
test('the same reload request cannot consume ammunition twice',async()=>{
 const f=fixture();await f.run();await assert.rejects(f.run(),/重复|已经/);assert.equal(f.calls.filter(c=>c[0]==='reload').length,1);
});
test('concurrent copies of one owner request cannot both reach Reload.reload',async()=>{
 const f=fixture();const results=await Promise.allSettled([f.run(),f.run()]);assert.equal(results.filter(r=>r.status==='fulfilled').length,1);assert.equal(f.calls.filter(c=>c[0]==='reload').length,1);
});
test('unowned or changed original Use cannot invoke a reload',async()=>{
 const f=fixture();f.actor.testUserPermission=()=>false;await assert.rejects(f.run(),/权限|操作者/);assert.equal(f.calls.length,0);
});
test('one Interact cannot silently complete a multi-action or magazine reload',async()=>{
 const f=fixture();f.modules.WeaponSystem.transformWeapon=()=>({actor:f.actor,reloadActions:2,isRepeating:false});
 await assert.rejects(f.run(),/一次|Interact/);assert.equal(f.calls.length,0);
});
