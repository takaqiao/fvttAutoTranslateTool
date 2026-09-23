import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID} from '../scripts/rules.mjs';
import {DEFENSIVE_ADVANCE_SOURCE} from '../scripts/defensive-advance-compat.mjs';
import {USE_ACTION_OPTION} from '../scripts/usage-events.mjs';
import {createDefensiveAdvance} from '../scripts/defensive-advance.mjs';

const update=function(changes){for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let at=this;for(const key of parts.slice(0,-1))at=at[key]??={};at[parts.at(-1)]=structuredClone(value);}return Promise.resolve(this)};
const firstChoice=async({choices})=>choices[0].value;
function fixture({startup='ready',choose=firstChoice}={}){
 const user={id:'gm',active:true,isGM:true,settings:{showCheckDialogs:false}},game={user,users:new Map([['gm',user]]),world:{id:'ujx5r8oipw7ercdr'},modules:new Map([['patreon-v3',{active:true,version:'3.2.28'}]]),messages:new Map(),actors:new Map(),scenes:new Map(),combats:new Map()};game.users.activeGM=user;
 const actor={id:'a',uuid:'Actor.a',type:'character',flags:{},items:new Map(),canAct:true,alliance:'party',attributes:{shield:{itemId:'shield',raised:true,broken:false,destroyed:false}},system:{movement:{speeds:{land:{value:20}}},actions:[]},testUserPermission:u=>u===user,update};game.actors.set('a',actor);
 const item={id:'feat',uuid:'Actor.a.Item.feat',actor,type:'feat',sourceId:DEFENSIVE_ADVANCE_SOURCE,system:{actionType:{value:'action'},actions:{value:2},traits:{value:['flourish']}}};actor.items.set('feat',item);
 const scene={id:'s',tokens:new Map()},token={id:'t',uuid:'Scene.s.Token.t',documentName:'Token',parent:scene,actor,x:0,y:0,elevation:0};scene.tokens.set('t',token);game.scenes.set('s',scene);actor.getActiveTokens=()=>[token];
 const combat={id:'real',started:true,round:2,turn:0,turns:[{id:'c',actor,token}]};game.combats.set('real',combat);game.combat={id:'different-view'};
 const hooks=new Map(),Hooks={on(n,fn){if(!hooks.has(n))hooks.set(n,new Set());hooks.get(n).add(fn);return fn},off(n,fn){hooks.get(n)?.delete(fn)},async emit(n,...args){for(const fn of hooks.get(n)??[])await fn(...args)}};
 const documents=new Map([[actor.uuid,actor],[item.uuid,item],[token.uuid,token]]),rpc=new Map(),requests=[],movementCalls=[];
 token.object={document:token,planMovement:async()=>{movementCalls.push('plan');throw Error('Native map movement belongs to the operator.')}};
 token.startMovement=async()=>{movementCalls.push('start');throw Error('No automated movement may start.')};
 token.update=async()=>{movementCalls.push('coordinates');throw Error('No automatic coordinate changes.')};
 const provider=createDefensiveAdvance({game,fromUuid:async uuid=>documents.get(uuid),choose:async request=>{requests.push(request);return choose(request)},startupCompatibility:{status:startup},onError:()=>{}});
 const socket={register:(name,fn)=>rpc.set(name,fn)},dispose=provider.register({Hooks,socket});
 function message(id='original'){const m={id,uuid:`ChatMessage.${id}`,author:user,actor,item,speaker:{actor:'a',scene:'s',token:'t'},flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid,type:'feat',rollOptions:[USE_ACTION_OPTION,'origin:item:trait:flourish']}},[MODULE_ID]:{usageInput:{actualUse:true,targetUuids:Array.from(user.targets??[]).map(t=>t.document.uuid)},...provider.captureUsage(item)}},update,updateSource(changes){return update.call(this,changes)}};game.messages.set(id,m);return m;}
 const run=message=>provider.executeUsage({actor,item,message,user,action:'defensive-advance'});
 return {game,user,actor,item,token,combat,Hooks,hooks,rpc,socket,provider,message,scene,documents,requests,movementCalls,run,dispose};
}
function armEnemy(f){
 const enemy={id:'e',uuid:'Actor.e',alliance:'opposition'},target={id:'enemy',uuid:'Scene.s.Token.enemy',documentName:'Token',parent:f.scene,actor:enemy,x:200,y:0,elevation:0};target.object={document:target};f.scene.tokens.set(target.id,target);f.documents.set(target.uuid,target);f.user.targets=new Set([target.object]);
 const weapon={id:'w',uuid:'Actor.a.Item.w',actor:f.actor,name:'Sword',isMelee:true,isRanged:false},calls=[];
 const strike={type:'strike',ready:true,item:weapon,variants:[0,1,2].map(map=>({async roll(params){
  const nativeTarget=params.target.document;calls.push({map,target:params.target,options:[...params.options]});
  const card={id:'attack',author:f.nativeUser??f.user,speaker:{actor:'a',scene:'s',token:'t'},flavor:'<h4 class="action"><span class="action-glyph">A</span>Strike</h4>',flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:weapon.uuid},context:{type:'attack-roll',action:'strike',mapIncreases:map,origin:{actor:f.actor.uuid,token:f.token.uuid},target:{actor:nativeTarget.actor.uuid,token:nativeTarget.uuid},outcome:'success',options:[...params.options]}}},rolls:[{total:21}],isCheckRoll:true,updateSource:changes=>update.call(card,changes)};
  await f.Hooks.emit('preCreateChatMessage',card);f.game.messages.set(card.id,card);await params.callback(card.rolls[0],'success',card);return card.rolls[0];
 }}))};f.actor.system.actions.push(strike);return {target,strike,calls};
}
async function original(f){const m=f.message();await f.Hooks.emit('preCreateChatMessage',m);return m}
const receipt=m=>m.flags[MODULE_ID].defensiveAdvance;

test('only this provider exact original feature requires actual Use',()=>{
 const f=fixture();assert.equal(f.provider.requiresActualUse(f.item),true);assert.equal(f.provider.requiresActualUse({type:'feat',sourceId:'Compendium.other.Item.feature'}),false);
});

test('one explicit continuation produces one native included Strike without planning or observing movement',async()=>{
 const f=fixture({choose:async request=>request.title.includes('MAP')?'1':request.choices[0].value}),enemy=armEnemy(f),m=await original(f);
 await f.run(m);
 assert.equal(f.hooks.has('moveToken'),false);assert.equal(f.hooks.has('preUpdateToken'),false);assert.deepEqual(f.movementCalls,[]);
 assert.deepEqual([...f.rpc.keys()],['defensive-advance:strike']);
 assert.deepEqual(f.requests[0].choices,[{value:'continue',label:'地图移动后继续内含打击'},{value:'decline',label:'结束此活动'}]);
 assert.equal(receipt(m).status,'done');assert.equal(receipt(m).cost,2);assert.equal(receipt(m).map,1);assert.equal(receipt(m).checkId,'attack');
 assert.equal(enemy.calls.length,1);assert.equal(enemy.calls[0].target,enemy.target.object);assert.equal(enemy.calls[0].map,1);
 const card=f.game.messages.get('attack');assert.ok(card.flags.pf2e.context.options.includes('action:free'));assert.equal(card.flags.pf2e.context.target.token,'Scene.s.Token.enemy');assert.match(card.flavor,/>F</);
 await f.run(m);assert.equal(enemy.calls.length,1);assert.equal(f.actor.flags[MODULE_ID].defensiveAdvanceUses.length,1);
 const second=f.message('second');await f.Hooks.emit('preCreateChatMessage',second);await assert.rejects(f.run(second),/华丽/);assert.equal(enemy.calls.length,1);
});

test('native map movement during the pending continuation writes no activity receipt and only its explicit answer permits Strike',async()=>{
 const answer=Promise.withResolvers(),offered=Promise.withResolvers();let f;
 f=fixture({choose:async request=>{if(request.choices.some(c=>c.value==='continue')){offered.resolve();return answer.promise}return request.choices[0].value}});
 const enemy=armEnemy(f),m=await original(f),flow=f.run(m);await Promise.race([offered.promise,flow.then(()=>assert.fail('activity ended before continuation'),error=>{throw error})]);
 assert.equal(enemy.calls.length,0);assert.equal(receipt(m).status,'awaiting-movement');const before=structuredClone(receipt(m));
 f.token.x=10000;f.token.elevation=20;await f.Hooks.emit('moveToken',f.token,{passed:{cost:1000,waypoints:[{action:'teleport'}]}},{},f.user);
 assert.deepEqual(receipt(m),before);await f.run(m);assert.equal(f.requests.length,1);answer.resolve('continue');await flow;assert.equal(enemy.calls.length,1);
});

for(const choice of ['decline',null])test(`ending continuation with ${choice} preserves original cost and never starts a Strike or charges again`,async()=>{
 const f=fixture({choose:async()=>choice}),enemy=armEnemy(f),m=await original(f);
 await f.run(m);assert.equal(receipt(m).status,'cancelled');assert.equal(receipt(m).cost,2);assert.equal(enemy.calls.length,0);
 await f.run(m);assert.equal(f.requests.length,1);assert.equal(enemy.calls.length,0);assert.equal(f.actor.flags[MODULE_ID].defensiveAdvanceUses.length,1);assert.deepEqual(f.movementCalls,[]);
});

test('the continuation gives the native shield executor time to finish without polling',async t=>{
 let f;f=fixture({choose:async request=>{if(request.choices.some(c=>c.value==='continue'))f.actor.attributes.shield.raised=true;return request.choices[0].value}});
 f.actor.attributes.shield.raised=false;const enemy=armEnemy(f),m=await original(f);t.mock.method(globalThis,'setTimeout',()=>assert.fail('shield readiness must not poll'));
 await f.run(m);assert.equal(enemy.calls.length,1);
});

test('missing native raised-shield result stops after continuation without replacement effect or replay',async t=>{
 const f=fixture(),enemy=armEnemy(f),m=await original(f);f.actor.attributes.shield.raised=false;
 t.mock.method(globalThis,'setTimeout',()=>assert.fail('shield readiness must not poll'));f.actor.createEmbeddedDocuments=()=>assert.fail('native shield effect must not be replaced');
 await assert.rejects(f.run(m),/原生举盾/);assert.equal(receipt(m).status,'uncertain');assert.equal(f.requests.length,1);assert.equal(enemy.calls.length,0);
 await f.run(m);assert.equal(f.requests.length,1);assert.equal(enemy.calls.length,0);
});

test('distance, walls, missing land speed and movement after Use do not change the chosen native target or MAP',async()=>{
 let f,enemy;f=fixture({choose:async request=>{
  if(request.choices.some(c=>c.value==='continue')){f.actor.system.movement.speeds.land.value=0;f.token.x=10000;f.token.elevation=40}
  if(request.title.includes('MAP')){enemy.target.x=90000;f.game.user.targets=new Set([{document:{uuid:'Scene.s.Token.other'}}]);return '2'}
  return request.choices[0].value;
 }});
 enemy=armEnemy(f);f.token.object.checkCollision=()=>assert.fail('wall adjudication belongs to GM');f.token.object.distanceTo=()=>assert.fail('distance adjudication belongs to GM');f.actor.getReach=()=>assert.fail('reach adjudication belongs to GM');
 const m=await original(f);f.token.x=500;f.actor.hasCondition=name=>['immobilized','restrained'].includes(name);await f.run(m);
 assert.equal(enemy.calls.length,1);assert.equal(enemy.calls[0].target,enemy.target.object);assert.equal(enemy.calls[0].map,2);assert.deepEqual(f.game.messages.get('attack').flags.pf2e.context.target,{actor:'Actor.e',token:'Scene.s.Token.enemy'});
});

test('relinked target actor during weapon or MAP selection cannot inherit the chosen target binding',async()=>{
 for(const moment of ['weapon','map']){let enemy;const f=fixture({choose:async request=>{
  if(moment==='weapon'&&request.choices.some(c=>c.value==='Actor.a.Item.w#base')||moment==='map'&&request.title.includes('MAP'))enemy.target.actor={id:'replacement',uuid:'Actor.replacement',alliance:'opposition'};
  return request.choices[0].value;
 }});enemy=armEnemy(f);const m=await original(f);await assert.rejects(f.run(m),/目标|近战Strike/);assert.equal(enemy.calls.length,0);}
});

test('only the original Use target is resolved; later GM targets and scene-wide candidates are never read',async()=>{
 const f=fixture(),enemy=armEnemy(f),m=await original(f);Object.defineProperty(f.user,'targets',{get:()=>assert.fail('do not read GM targets after Use')});f.scene.tokens.values=()=>assert.fail('do not enumerate scene targets');
 await f.run(m);assert.equal(enemy.calls.length,1);assert.equal(receipt(m).targetUuid,'Scene.s.Token.enemy');assert.equal(receipt(m).targetActorUuid,'Actor.e');
});

test('missing, multiple, malformed or stale original targets end the follow-up without borrowing GM targets',async()=>{
 for(const targets of [undefined,[],['Scene.s.Token.enemy','Scene.s.Token.other'],['invalid-target'],['Scene.s.Token.deleted']]){
  const f=fixture(),enemy=armEnemy(f),m=await original(f);m.flags[MODULE_ID].usageInput.targetUuids=targets;
  const result=await f.run(m);assert.match(result,/原始Use.*T/);assert.equal(receipt(m).status,'done');assert.equal(receipt(m).cost,2);assert.equal(enemy.calls.length,0);assert.equal(f.requests.length,0);
  await f.run(m);assert.equal(f.actor.flags[MODULE_ID].defensiveAdvanceUses.length,1);
 }
});

test('a target resolver cannot substitute another live Token for the original Use UUID',async()=>{
 const f=fixture(),enemy=armEnemy(f),m=await original(f);m.flags[MODULE_ID].usageInput.targetUuids=['Scene.s.Token.different'];f.documents.set('Scene.s.Token.different',enemy.target);
 await f.run(m);assert.equal(enemy.calls.length,0);assert.equal(f.requests.length,0);assert.equal(receipt(m).targetUuid,null);
});

test('GM migration while resolving the original target cannot write a new activity charge',async()=>{
 const f=fixture(),enemy=armEnemy(f),m=await original(f),get=f.documents.get;f.documents.get=function(uuid){const result=get.call(this,uuid);if(uuid===enemy.target.uuid)f.game.users.activeGM={id:'replacement',isGM:true};return result};
 await assert.rejects(f.run(m),/主GM/);assert.equal(f.actor.flags[MODULE_ID]?.defensiveAdvanceUses,undefined);assert.equal(receipt(m),undefined);assert.equal(enemy.calls.length,0);
});

test('the original player executes the bound native Strike and a duplicate owner RPC cannot roll twice',async()=>{
 const f=fixture({choose:async request=>{assert.equal(request.user.id,'player');return request.choices[0].value}}),enemy=armEnemy(f),player={id:'player',active:true,isGM:false};f.game.users.set(player.id,player);f.actor.testUserPermission=user=>user===f.user||user===player;f.nativeUser=player;
 const ownerGame={...f.game,user:player},ownerRpc=new Map(),ownerProvider=createDefensiveAdvance({game:ownerGame,fromUuid:async uuid=>f.documents.get(uuid),choose:()=>assert.fail('owner native Strike must not repeat continuation'),startupCompatibility:{status:'ready'}});
 ownerProvider.register({Hooks:f.Hooks,socket:{register:(name,fn)=>ownerRpc.set(name,fn)}});
 const m=await original(f);m.author=player;let requests=0;f.socket.executeAsUser=async(name,userId,payload)=>{
  requests++;assert.equal(name,'defensive-advance:strike');assert.equal(userId,'player');const handler=ownerRpc.get(name),sender={socketdata:{userId:f.user.id}};
  const result=await handler.call(sender,payload);assert.equal(result.ok,true);const duplicate=await handler.call(sender,payload);assert.equal(duplicate.ok,false);assert.match(duplicate.error,/已经进入/);return result;
 };
 await f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:player,action:'defensive-advance'});assert.equal(requests,1);assert.equal(enemy.calls.length,1);assert.equal(f.game.messages.get('attack').author,player);assert.equal(receipt(m).userId,'player');assert.equal(receipt(m).status,'done');
});

test('saved repair without startup cache, display and previous flourish cannot enter continuation',async()=>{
 for(const mode of ['reload','display','flourish']){const f=fixture({startup:mode==='reload'?'requires-reload':'ready'}),m=await original(f);
  if(mode==='display'){m.flags[MODULE_ID].usageInput.actualUse=false;m.flags.pf2e.origin.rollOptions=[]}
  if(mode==='flourish')f.game.messages.set('earlier',{id:'earlier',speaker:{actor:'a'},flags:{pf2e:{origin:{rollOptions:[USE_ACTION_OPTION,'origin:item:trait:flourish']}},[MODULE_ID]:{defensiveAdvanceObservedTurn:'real:2:0'}}});
  await assert.rejects(f.run(m));assert.equal(f.requests.length,0);assert.deepEqual(f.movementCalls,[]);
 }
});

test('source, owner, original Use, turn or GM changes during continuation cannot start the committed Strike',async()=>{
 for(const mode of ['source','owner','message','turn','gm']){let f;f=fixture({choose:async request=>{
  if(request.choices.some(c=>c.value==='continue')){
   if(mode==='source')f.item.sourceId='Compendium.other.Item.changed';
   if(mode==='owner')f.actor.testUserPermission=()=>false;
   if(mode==='message')f.game.messages.set('original',{id:'original'});
   if(mode==='turn')f.combat.turn=1;
   if(mode==='gm')f.game.users.activeGM={id:'replacement',isGM:true};
  }return request.choices[0].value;
 }});
 const enemy=armEnemy(f),m=await original(f);await assert.rejects(f.run(m));assert.equal(enemy.calls.length,0);assert.equal(f.actor.flags[MODULE_ID].defensiveAdvanceUses.length,1);assert.equal(f.actor.flags[MODULE_ID].defensiveAdvanceUses[0].cost,2);
 }
});
