import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createShieldBlockEvents} from '../scripts/shield-block-events.mjs';
import {registerReactionShieldWallEmptyCompatibility} from '../scripts/reaction-shield-wall-empty-compat.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';

const fortress='ujx5r8oipw7ercdr';
const bundleHash='4a81322796ce1c6ed545edc09e1aa3a96a9c8a96dfd034403bf657068ed7036c';
const callbackHash='6cb70e38abbdc59e441e56f085144eaa54f69763e8649e7ea942e32326ad4841';

function blockFixture(world=fortress){
 const user={id:'gm',isGM:true},hooks=new Map(),delivered=[];
 const actor={id:'defender',uuid:'Actor.defender',type:'character',flags:{},items:[{sourceId:'Compendium.pf2e.feats-srd.Item.dSSwRyuhKTq1VubX'}],hitPoints:{value:50},attributes:{shield:{itemId:'shield',raised:true,broken:false,destroyed:false}},testUserPermission:u=>u===user,
  async update(changes){for(const [path,value]of Object.entries(changes)){const keys=path.split('.');let parent=this;for(const key of keys.slice(0,-1))parent=parent[key]??={};parent[keys.at(-1)]=structuredClone(value);}}
 };
 const token={id:'defender',uuid:'Scene.scene.Token.defender',documentName:'Token',actor,parent:{id:'scene'}};
 const source={verified:true,actorUuid:actor.uuid,tokenUuid:token.uuid,attackerActorUuid:'Actor.enemy',attackerTokenUuid:'Scene.scene.Token.enemy',attackItemUuid:'Actor.enemy.Item.weapon',weaponUuid:'Actor.enemy.Item.weapon',damageMessageId:'damage',rollIndex:0,sourceSnapshot:{proof:'exact-source'}};
 const game={world:{id:world},user,users:{activeGM:user,get:()=>user},messages:new Map(),combat:null};
 const observer=createShieldBlockEvents({game,fromUuid:async uuid=>uuid===actor.uuid?actor:uuid===token.uuid?token:null,resolveSource:async()=>source,validateSource:async({snapshot})=>snapshot?.proof==='exact-source'?source:{verified:false},onConfirmed:event=>delivered.push(event)});
 observer.register({Hooks:{on:(name,fn)=>hooks.set(name,fn),off(){}},socket:{register(){}}});
 const params={shieldBlockRequest:true,damage:12,token,rollOptions:new Set()};
 const native=async p=>{
  const message={id:'native-block',author:user,speaker:{actor:actor.id,scene:'scene',token:token.id},content:'native block',flags:{pf2e:{context:{type:'damage-taken',options:[...p.rollOptions]},appliedDamage:{shield:{id:'shield',damage:4}}}}};
  game.messages.set(message.id,message);hooks.get('createChatMessage')(message,{},user.id);return actor;
 };
 return {actor,observer,params,native,delivered};
}

test('fortress native block dispatches one exact-weapon event after its real result card',async()=>{
 const f=blockFixture(),prepared=await f.observer.beforeDamage(f.actor,f.params);
 assert.ok(prepared?.receipt,'fortress must arm its existing block observer');
 assert.equal(f.delivered.length,0);
 await f.observer.wrapNativeDamage(f.actor,prepared.params,f.native);
 assert.equal(f.delivered.length,0);
 await f.observer.afterDamage(prepared.receipt,{applied:true,uncertain:false});
 assert.equal(f.delivered.length,1);assert.equal(f.delivered[0].weaponUuid,'Actor.enemy.Item.weapon');
 assert.equal(f.actor.flags[M].shieldBlockEvents.records[0].status,'confirmed');
 await f.observer.afterDamage(prepared.receipt,{applied:true,uncertain:false});assert.equal(f.delivered.length,1);
});

test('the fortress addition keeps unrelated worlds and same-name feats outside the observer',async()=>{
 const other=blockFixture('unreviewed-world');assert.equal(await other.observer.beforeDamage(other.actor,other.params),null);assert.deepEqual(other.actor.flags,{});
 const f=blockFixture();f.actor.items=[{name:'Disarming Block',sourceId:'Compendium.other.feats.Item.other'}];assert.equal(await f.observer.beforeDamage(f.actor,f.params),null);assert.deepEqual(f.actor.flags,{});
});

function compatibilityFixture(world=fortress){
 const actor={id:'defender',type:'character',alliance:'party',itemTypes:{feat:[]}},ally={id:'ally',type:'character',alliance:'party',itemTypes:{feat:[]}};
 const game={world:{id:world},release:{generation:14},system:{version:'8.5.1'},modules:new Map([['pf2e-reaction',{active:true,version:'1.4.3'}]]),userId:'gm',combat:{turns:[{id:'defender',actorId:actor.id,actor},{id:'ally',actorId:ally.id,actor:ally}]}};
 const calls=[];
 const original=async function(item,...args){calls.push([item,...args]);return item.slug==="effect-raise-a-shield"?"shield-wall":'other';};
 const entry={fn:original,id:12,hook:'createItem',once:false},Hooks={events:{createItem:[entry]}};
 const item={type:'effect',slug:'effect-raise-a-shield',actor};
 const install=overrides=>registerReactionShieldWallEmptyCompatibility({game,Hooks,fetchSource:async()=>'fixture bundle',hashSource:async source=>source==='fixture bundle'?bundleHash:callbackHash,...overrides});
 return {game,actor,ally,item,entry,Hooks,original,calls,install};
}

test('fortress protects only empty Shield Wall candidates and preserves upstream identity on disposal',async()=>{
 const f=compatibilityFixture(),result=await f.install();assert.equal(result.status,'installed');
 assert.equal(await f.entry.fn(f.item,{},'gm'),undefined);assert.equal(f.calls.length,0);
 f.ally.itemTypes.feat.push({slug:'shield-wall'});
 assert.equal(await f.entry.fn(f.item,{},'gm'),'shield-wall');assert.equal(f.calls.length,1);
 f.ally.itemTypes.feat=[];assert.equal(await f.entry.fn(f.item,{},'other-user'),'shield-wall');assert.equal(f.calls.length,2);
 result.dispose();assert.equal(f.entry.fn,f.original);
});

test('fortress compatibility still rejects unknown worlds, dependency versions and source hashes',async()=>{
 for(const mutate of [f=>{f.game.world.id='unreviewed-world';},f=>{f.game.modules.get('pf2e-reaction').version='unknown';},f=>{f.game.system.version='unknown';}]){
  const f=compatibilityFixture();mutate(f);const result=await f.install();assert.equal(result.status,'unsupported');assert.equal(f.entry.fn,f.original);
 }
 const f=compatibilityFixture(),result=await f.install({hashSource:async()=> 'not-the-audited-source'});
 assert.equal(result.reason,'unknown-reaction-bundle');assert.equal(f.entry.fn,f.original);
});

// Optional local integration: execute the installed upstream callback, never a
// copied bundle or a user-specific path in the public test suite.
const nativePath=process.env.FVTT_REACTION_BUNDLE??'';
test('audited Reaction callback no longer throws for fortress empty candidates but remains untouched for real Shield Wall', {skip:!nativePath},async()=>{
 const source=readFileSync(nativePath,'utf8'),start=source.indexOf('Hooks.on("createItem",')+'Hooks.on("createItem",'.length,end=source.indexOf('),Hooks.on("preUpdateToken"',start);
 assert.ok(start>'Hooks.on("createItem",'.length&&end>start,'expected audited native callback');
 const f=compatibilityFixture();
 const original=Function('game','d','H','R','i','m','N',`return (${source.slice(start,end)})`)(f.game,()=>true,(actor,slug)=>actor.itemTypes.feat.find(feat=>feat.slug===slug),()=>true,()=>true,()=>true,()=>assert.fail('no real Shield Wall offer should run in this fixture'));
 f.entry.fn=original;
 await assert.rejects(original(f.item,{},'gm'),/a is not defined/);
 const result=await registerReactionShieldWallEmptyCompatibility({game:f.game,Hooks:f.Hooks,fetchSource:async()=>source});
 assert.equal(result.status,'installed');assert.equal(result.sourceSHA256,bundleHash);assert.equal(result.callbackSHA256,callbackHash);
 await assert.doesNotReject(f.entry.fn(f.item,{},'gm'));
 f.ally.itemTypes.feat.push({slug:'shield-wall'});
 await assert.rejects(f.entry.fn(f.item,{},'gm'),/a is not defined/);
 result.dispose();assert.equal(f.entry.fn,original);
});
