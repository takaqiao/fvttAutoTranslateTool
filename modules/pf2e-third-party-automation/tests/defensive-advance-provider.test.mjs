import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID} from '../scripts/rules.mjs';
import {DEFENSIVE_ADVANCE_SOURCE} from '../scripts/defensive-advance-compat.mjs';
import {USE_ACTION_OPTION} from '../scripts/usage-events.mjs';
const api=await import('../scripts/defensive-advance.mjs').catch(()=>({}));
const update=function(changes){for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let at=this;for(const key of parts.slice(0,-1))at=at[key]??={};at[parts.at(-1)]=structuredClone(value);}return Promise.resolve(this)};
function fixture({startup='ready',cancel=false,breakMovement=false,choose=async()=> 'decline'}={}){
 const user={id:'gm',active:true,isGM:true,settings:{showCheckDialogs:false}},game={user,users:new Map([['gm',user]]),world:{id:'ujx5r8oipw7ercdr'},modules:new Map([['patreon-v3',{active:true,version:'3.2.28'}]]),messages:new Map(),actors:new Map(),scenes:new Map(),combats:new Map()};game.users.activeGM=user;
 const actor={id:'a',uuid:'Actor.a',type:'character',flags:{},items:new Map(),canAct:true,alliance:'party',attributes:{shield:{itemId:'shield',raised:true,broken:false,destroyed:false}},system:{movement:{speeds:{land:{value:20}}},actions:[]},testUserPermission:u=>u===user,update};game.actors.set('a',actor);
 const item={id:'feat',uuid:'Actor.a.Item.feat',actor,type:'feat',sourceId:DEFENSIVE_ADVANCE_SOURCE,system:{actionType:{value:'action'},actions:{value:2},traits:{value:['flourish']}}};actor.items.set('feat',item);
 const scene={id:'s',tokens:new Map()},token={id:'t',uuid:'Scene.s.Token.t',documentName:'Token',parent:scene,actor,x:0,y:0,elevation:0};scene.tokens.set('t',token);game.scenes.set('s',scene);actor.getActiveTokens=()=>[token];
 const combat={id:'real',started:true,round:2,turn:0,turns:[{id:'c',actor,token}]};game.combats.set('real',combat);game.combat={id:'different-view'};
 const hooks=new Map(),Hooks={on(n,fn){if(!hooks.has(n))hooks.set(n,new Set());hooks.get(n).add(fn);return fn},off(n,fn){hooks.get(n)?.delete(fn)},async emit(n,...args){for(const fn of hooks.get(n)??[])await fn(...args)}};
 let planned=0,started=0;const documents=new Map([[actor.uuid,actor],[item.uuid,item],[token.uuid,token]]);
 const provider=api.createDefensiveAdvance({game,fromUuid:async uuid=>documents.get(uuid),choose,startupCompatibility:{status:startup},onError:()=>{}});provider.register({Hooks});
 token.object={document:token,async planMovement(options){planned++;assert.deepEqual(options,{allowedActions:['walk'],maxCost:20,preventDrop:true});if(cancel)return null;token.movement={id:'plan',state:'planned',user,pending:{cost:10,waypoints:[{action:'walk'}]},origin:{x:0,y:0,elevation:0},finished:Promise.resolve(true),animation:{ended:Promise.resolve()}};return {id:'plan',origin:token.movement.origin,destination:{x:100,y:0,elevation:0}}}};
 token.startMovement=async id=>{assert.equal(id,'plan');started++;token.x=100;const movement={id:'plan',chain:[],origin:{x:0,y:0,elevation:0},destination:{x:100,y:0,elevation:0},passed:{cost:10,waypoints:[{action:'walk'}]},pending:{waypoints:[]},constrained:false,finished:Promise.resolve(true),animation:{ended:Promise.resolve()}};token.movement={...movement,state:'completed',user};if(!breakMovement)await Hooks.emit('moveToken',token,movement,{_movement:{t:movement}},user);return true};
 token.stopMovement=()=>{token.movement.state='stopped'};
 function message(id='original') {const m={id,uuid:`ChatMessage.${id}`,author:user,actor,item,speaker:{actor:'a',scene:'s',token:'t'},flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid,type:'feat',rollOptions:[USE_ACTION_OPTION,'origin:item:trait:flourish']}},[MODULE_ID]:{usageInput:{actualUse:true},...provider.captureUsage(item)}},update,updateSource(changes){return update.call(this,changes)}};game.messages.set(id,m);return m;}
 return {game,user,actor,item,token,combat,Hooks,provider,message,scene,documents,counts:()=>({planned,started})};
}

test('only this provider own exact original feature requires actualUse; unrelated legacy routes stay unchanged',()=>{
 const f=fixture();assert.equal(f.provider.requiresActualUse(f.item),true);
 assert.equal(f.provider.requiresActualUse({type:'feat',sourceId:'Compendium.other.Item.feature'}),false);
});

test('original Use completes one bound native Stride, commits two-action flourish once, and replay never moves',async()=>{
 assert.equal(typeof api.createDefensiveAdvance,'function');const f=fixture(),m=f.message();await f.Hooks.emit('preCreateChatMessage',m);
 await f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'});
 assert.deepEqual(f.counts(),{planned:1,started:1});assert.equal(m.flags[MODULE_ID].defensiveAdvance.status,'done');assert.equal(m.flags[MODULE_ID].defensiveAdvance.cost,2);assert.equal(m.flags[MODULE_ID].defensiveAdvance.movementCost,10);
 await f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'});assert.equal(f.counts().planned,1);
 const second=f.message('second');await f.Hooks.emit('preCreateChatMessage',second);await assert.rejects(f.provider.executeUsage({actor:f.actor,item:f.item,message:second,user:f.user,action:'defensive-advance'}),/华丽/);assert.equal(f.counts().planned,1);
});

test('saved repair without startup cache, ordinary display, and previous flourish cannot enter movement',async()=>{
 assert.equal(typeof api.createDefensiveAdvance,'function');
 for(const mode of ['reload','display','flourish']){const f=fixture({startup:mode==='reload'?'requires-reload':'ready'}),m=f.message();await f.Hooks.emit('preCreateChatMessage',m);
  if(mode==='display'){m.flags[MODULE_ID].usageInput.actualUse=false;m.flags.pf2e.origin.rollOptions=[];}
  if(mode==='flourish')f.game.messages.set('earlier',{id:'earlier',speaker:{actor:'a'},flags:{pf2e:{origin:{rollOptions:[USE_ACTION_OPTION,'origin:item:trait:flourish']}},[MODULE_ID]:{defensiveAdvanceObservedTurn:'real:2:0'}}});
  await assert.rejects(f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'}));assert.equal(f.counts().planned,0);
 }
});

test('movement cancellation preserves committed flourish and arbitrary position changes never grant the Strike stage',async()=>{
 assert.equal(typeof api.createDefensiveAdvance,'function');
 for(const cancel of [true,false]){const f=fixture({cancel,breakMovement:!cancel}),m=f.message();await f.Hooks.emit('preCreateChatMessage',m);
  await f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'}).catch(()=>{});
  const receipt=m.flags[MODULE_ID].defensiveAdvance;assert.ok(['cancelled','uncertain'].includes(receipt.status));assert.equal(receipt.cost,2);assert.notEqual(receipt.status,'done');assert.ok(f.actor.flags[MODULE_ID].defensiveAdvanceUses.some(u=>u.messageId===m.id));
 }
});

function armEnemy(f,{blocked=()=>false}={}){
 const enemy={id:'e',uuid:'Actor.e',alliance:'opposition'},target={id:'enemy',uuid:'Scene.s.Token.enemy',documentName:'Token',parent:f.scene,actor:enemy,x:200,y:0,elevation:0,getCenterPoint:()=>({x:250,y:50})};target.object={document:target};f.scene.tokens.set(target.id,target);
 f.token.object.checkCollision=blocked;f.token.object.distanceTo=()=>5;f.actor.getReach=()=>5;
 const weapon={id:'w',uuid:'Actor.a.Item.w',actor:f.actor,name:'Sword',isMelee:true,isRanged:false};let rolls=0;
 const strike={type:'strike',ready:true,item:weapon,variants:[0,1,2].map(map=>({async roll(params){rolls++;const card={id:'attack',author:f.user,speaker:{actor:'a',scene:'s',token:'t'},flavor:'<h4 class="action"><span class="action-glyph">A</span>Strike</h4>',flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:weapon.uuid},context:{type:'attack-roll',action:'strike',mapIncreases:map,origin:{actor:f.actor.uuid,token:f.token.uuid},target:{actor:enemy.uuid,token:target.uuid},outcome:'success',options:[...params.options]}}},rolls:[{total:21}],isCheckRoll:true,updateSource:changes=>update.call(card,changes)};await f.Hooks.emit('preCreateChatMessage',card);f.game.messages.set(card.id,card);await params.callback(card.rolls[0],'success',card);return card.rolls[0]}}))};f.actor.system.actions.push(strike);return {target,strike,rolls:()=>rolls};
}

test('complete movement then chosen native Strike preserves MAP and ends at one included attack',async()=>{
 const f=fixture({choose:async({title,choices})=>title.includes('MAP')?'1':choices[0].value}),enemy=armEnemy(f),m=f.message();await f.Hooks.emit('preCreateChatMessage',m);
 await f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'});
 const r=m.flags[MODULE_ID].defensiveAdvance;assert.equal(r.status,'done');assert.equal(r.checkId,'attack');assert.equal(r.map,1);assert.equal(enemy.rolls(),1);assert.equal(f.game.messages.get('attack').flags.pf2e.context.action,'strike');assert.match(f.game.messages.get('attack').flavor,/>F</);
 await f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'});assert.equal(enemy.rolls(),1);assert.equal(f.counts().started,1);
});

test('walls changing during MAP selection prevent the native Strike after completed movement',async()=>{
 let blocked=false;const f=fixture({choose:async({title,choices})=>{if(title.includes('MAP')){blocked=true;return '0'}return choices[0].value}}),enemy=armEnemy(f,{blocked:()=>blocked}),m=f.message();await f.Hooks.emit('preCreateChatMessage',m);
 await assert.rejects(f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'}),/近战Strike/);assert.equal(enemy.rolls(),0);assert.equal(m.flags[MODULE_ID].defensiveAdvance.status,'uncertain');assert.equal(f.counts().started,1);
});

test('a genuine movement receipt cannot grant Strike if finished coordinates differ from its server endpoint',async()=>{
 const f=fixture(),enemy=armEnemy(f),m=f.message();await f.Hooks.emit('preCreateChatMessage',m);
 const start=f.token.startMovement;f.token.startMovement=async id=>{const done=await start(id);f.token.x=50;return done};
 await assert.rejects(f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'}),/完成后Token位置/);
 assert.equal(enemy.rolls(),0);assert.equal(m.flags[MODULE_ID].defensiveAdvance.status,'uncertain');
 assert.equal(m.flags[MODULE_ID].defensiveAdvance.lastPosition.x,100);
});

test('a turn change or authority migration during native path selection cannot start the committed movement',async()=>{
 for(const change of ['turn','gm']){const f=fixture(),m=f.message();await f.Hooks.emit('preCreateChatMessage',m);const plan=f.token.object.planMovement;f.token.object.planMovement=async options=>{const result=await plan(options);if(change==='turn')f.combat.turn=1;else f.game.users.activeGM={id:'replacement',isGM:true};return result};
  await assert.rejects(f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'}));assert.equal(f.counts().started,0);assert.equal(f.actor.flags[MODULE_ID].defensiveAdvanceUses[0].cost,2);
 }
});

test('moving away after original Use or losing speed during selection cannot borrow a different origin or old allowance',async()=>{
 for(const mode of ['origin','speed']){const f=fixture(),m=f.message();await f.Hooks.emit('preCreateChatMessage',m);
  if(mode==='origin')f.token.x=50;
  else{const plan=f.token.object.planMovement;f.token.object.planMovement=async options=>{const result=await plan(options);f.actor.system.movement.speeds.land.value=5;return result};}
  await assert.rejects(f.provider.executeUsage({actor:f.actor,item:f.item,message:m,user:f.user,action:'defensive-advance'}));assert.equal(f.counts().started,0);if(mode==='origin')assert.equal(f.counts().planned,0);
 }
});
