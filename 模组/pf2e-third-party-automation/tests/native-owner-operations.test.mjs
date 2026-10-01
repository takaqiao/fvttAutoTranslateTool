import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
let api={};try{api=await import('../scripts/native-owner-operations.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
const ID='pf2e-third-party-automation';
const set=(object,path,value)=>{const bits=path.split('.');let at=object;for(const bit of bits.slice(0,-1))at=at[bit]??={};at[bits.at(-1)]=value;};
function fixture({cancel=false,offline=false}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:!offline,settings:{showCheckDialogs:false,showDamageDialogs:false}},users=new Map([[gm.id,gm],[player.id,player]]);users.activeGM=gm;
 const messages=new Map(),calls=[],hookEntries=new Map();let nextHook=0;
 const hooks={on(event,fn){const id=++nextHook;hookEntries.set(id,{event,fn});return id;},off(_event,id){hookEntries.delete(id);},call(event,...args){for(const entry of hookEntries.values())if(entry.event===event)entry.fn(...args);}};
 const actor={uuid:'Actor.pc',id:'pc',testUserPermission:u=>u.id==='player',items:new Map(),system:{actions:[]}},target={uuid:'Scene.s.Token.t',actor:{uuid:'Actor.t'},object:{}};
 const item={id:'weapon',uuid:'Actor.pc.Item.weapon',type:'weapon',actor};actor.items.set(item.id,item);
 let current;
 class Message{constructor(data){Object.assign(this,data)}toObject(){return {...this}}static async create(data){const message=new Message({...data,id:'check'});messages.set(message.id,message);return message;}}
 actor.system.actions=[{type:'strike',item,variants:[{async roll(options){calls.push({user:current.user.id,options});if(cancel)return null;await options.callback({},'success',new Message({speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid},context:{type:'attack-roll',outcome:'success',target:{token:target.uuid,actor:target.actor.uuid}}}},rolls:[{total:20}]}));return {};}}]}];
 const card={id:'activity',author:player,flags:{pf2e:{origin:{actor:actor.uuid}}},async update(changes){for(const[path,value]of Object.entries(changes))set(this,path,value);hooks.call('updateChatMessage',this);}};messages.set(card.id,card);
 const gmGame={user:gm,users,messages},ownerGame={user:player,users,messages},docs=new Map([[actor.uuid,actor],[target.uuid,target]]),handlers=new Map(),gmHandlers=new Map();
 const create=api.createNativeOwnerOperations;assert.equal(typeof create,'function');
 const owner=create({game:ownerGame,fromUuid:async uuid=>docs.get(uuid),scope:'test'}),root=create({game:gmGame,fromUuid:async uuid=>docs.get(uuid),scope:'test'});
 const ownerSocket={register(name,handler){handlers.set(name,handler)},async executeAsUser(name,userId,payload){assert.equal(userId,'gm');const prior=current;current=gmGame;try{return await gmHandlers.get(name).call({socketdata:{userId:'player'}},payload);}finally{current=prior;}}};
 owner.register({Hooks:hooks,socket:ownerSocket});
 const socket={register(name,handler){gmHandlers.set(name,handler);},async executeAsUser(name,userId,payload){assert.equal(userId,'player');const prior=current;current=ownerGame;try{return await handlers.get(name).call({socketdata:{userId:'gm'}},payload);}finally{current=prior;}}};
 root.register({Hooks:hooks,socket});
 return {root,owner,gmGame,ownerGame,actor,target,card,item,calls,socket,ownerSocket,handlers,Message,gm,player,hookEntries,hooks,getCurrent:()=>current};
}
async function setup(options,fn){const old=globalThis.CONFIG;try{const f=fixture(options);globalThis.CONFIG={ChatMessage:{documentClass:f.Message}};await fn(f);}finally{globalThis.CONFIG=old;}}
test('native attack runs on original owner awaiting the manual window and canonical check',()=>setup({},async f=>{
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[],flags:{test:true}});
 assert.equal(result.status,'rolled');assert.equal(result.messageId,'check');assert.equal(f.calls.length,1);assert.equal(f.calls[0].user,'player');assert.equal(f.calls[0].options.event.shiftKey,true);
 assert.equal(f.gmGame.messages.get('check').author,'player');
}));
test('owner dialog preference is not inverted by a synthetic Shift key',()=>setup({},async f=>{
 f.player.settings.showCheckDialogs=true;
 await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]});
 assert.equal(f.calls[0].options.event.shiftKey,false);
 const path=process.env.PF2E_NATIVE_BUNDLE;
 if(path){const source=fs.readFileSync(path,'utf8'),start=source.indexOf('function isRelevantEvent('),end=source.indexOf('function eventToMessageMode(',start);assert.ok(start>=0&&end>start);
  const native=new Function('game',`${source.slice(start,end)};return eventToRollParams;`)(f.ownerGame);
  assert.equal(native(f.calls[0].options.event,{type:'check'}).skipDialog,false);
 }
}));
test('owner cancellation is explicit and does not fabricate an attack result',()=>setup({cancel:true},async f=>{
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]});
 assert.equal(result.status,'cancelled');assert.equal(f.gmGame.messages.has('check'),false);
}));
test('disconnected original owner fails before native interaction',()=>setup({offline:true},async f=>{
 await assert.rejects(f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]}),/连接|离线/);assert.equal(f.calls.length,0);
}));
test('untrusted requester cannot initiate an owner operation',()=>setup({},async f=>{
 const handler=f.handlers.get('native-owner:test');await assert.rejects(handler.call({socketdata:{userId:'player'}},{messageId:'activity',nonce:'fake'}),/主GM/);assert.equal(f.calls.length,0);
}));
test('a target Token relink during the original native attack cannot settle against its replacement actor',()=>setup({},async f=>{
 const original=f.Message.create;f.Message.create=async data=>{const result=await original(data);f.target.actor={uuid:'Actor.replacement'};return result;};
 await assert.rejects(f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]}),/目标|确认/);
 assert.equal(f.calls.length,1);assert.equal(f.gmGame.messages.has('check'),true,'an already published native result is not replayed');
}));
test('a durable native result completes the GM call even if its owner RPC never replies',()=>setup({},async f=>{
 const original=f.socket.executeAsUser;f.socket.executeAsUser=async(...args)=>{await original(...args);return new Promise(()=>{});};
 let timer;try{
  const result=await Promise.race([f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]}),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('saved operation still blocked by lost RPC reply')),100);})]);
  assert.equal(result.messageId,'check');assert.equal(f.calls.length,1);assert.equal(f.hookEntries.size,0);
 }finally{clearTimeout(timer);}
}));
test('a committed payment continues the real native roll even if the commit RPC reply is lost',()=>setup({},async f=>{
 const original=f.ownerSocket.executeAsUser;f.ownerSocket.executeAsUser=async(...args)=>{await original(...args);return new Promise(()=>{});};let payments=0,timer;
 try{const result=await Promise.race([f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]},null,async()=>{payments++;}),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('committed payment still blocked by lost reply')),100);})]);
  assert.equal(result.status,'rolled');assert.equal(payments,1);assert.equal(f.calls.length,1);assert.equal(f.hookEntries.size,0);
 }finally{clearTimeout(timer);}
}));
test('owner disconnect ends an unresolved GM operation and releases its hooks without replaying payment',()=>setup({},async f=>{
 let started,finishPreparation;const entered=new Promise(resolve=>{started=resolve;}),preparing=new Promise(resolve=>finishPreparation=resolve);let payments=0;
 f.actor.system.actions[0].variants[0].roll=async options=>{started();await preparing;let resolve;const closed=new Promise(done=>resolve=done);const app={context:{options:options.options},resolve};f.hooks.call('renderCheckModifiersDialog',app);await closed;return null;};
 const operation=f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]},null,async()=>{payments++;});
 await entered;f.player.active=false;f.hooks.call('userConnected',f.player,false);let timer;
 try{await assert.rejects(Promise.race([operation,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('still pending after owner disconnect')),100);})]),/原操作者|离线|连接/);
  finishPreparation();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(payments,0);assert.equal(f.hookEntries.size,0);assert.notEqual(Object.values(f.card.flags[ID].nativeOwnerOperations)[0].committed,true);
 }finally{clearTimeout(timer);}
}));
test('a native window rendered after aborted preparation closes without dice and then releases its hook',async()=>{
 const entries=new Map();let seq=0,alive=true,finishPreparation,dice=0,closed=0;const preparing=new Promise(resolve=>finishPreparation=resolve);
 const Hooks={on(event,fn){const id=++seq;entries.set(id,{event,fn});return id},off(_event,id){entries.delete(id)},call(event,...args){for(const e of [...entries.values()])if(e.event===event)e.fn(...args)}};
 const operation=api.beforeNativeRoll({Hooks,marker:'late',showDialog:true,assertLive:()=>{if(!alive)throw Error('owner disconnected')},commit:async()=>{},native:async()=>{
  await preparing;let resolve;const acceptance=new Promise(done=>resolve=done);const app={context:{options:new Set(['late'])},resolve,close:()=>closed++};Hooks.call('renderCheckModifiersDialog',app);if(await acceptance)dice++;
 }});
 await new Promise(resolve=>setImmediate(resolve));alive=false;Hooks.call('userConnected');await assert.rejects(operation,/owner disconnected/);
 alive=true;finishPreparation();await new Promise(resolve=>setImmediate(resolve));assert.equal(dice,0);assert.equal(closed,1);assert.equal(entries.size,0);
});
test('a saved completed operation remains authoritative when its owner disconnects before the reply',()=>setup({},async f=>{
 const update=f.card.update;f.card.update=async changes=>{if(Object.values(changes).some(value=>value?.status==='done'))f.player.active=false;return update.call(f.card,changes);};
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]});
 assert.equal(result.status,'rolled');assert.equal(result.messageId,'check');assert.equal(f.calls.length,1);assert.equal(f.hookEntries.size,0);
}));
for(const accepted of [false,true])test(`native dialog ${accepted?'acceptance':'cancellation'} commits resources only before an accepted roll`,async()=>{
 assert.equal(typeof api.beforeNativeRoll,'function');let hook,commits=0,resolved;
 const Hooks={on(event,callback){assert.equal(event,'renderCheckModifiersDialog');hook=callback;return 1;},off(){hook=null;}};
 await api.beforeNativeRoll({Hooks,marker:'operation',showDialog:true,commit:async()=>{commits++;},native:async()=>{
  const app={context:{options:new Set(['operation'])},resolve:value=>{resolved=value;}};hook(app);await app.resolve(accepted);
 }});
 assert.equal(commits,accepted?1:0);assert.equal(resolved,accepted);assert.equal(hook,null);
});
test('skipped native dialog commits once before any die is rolled',async()=>{
 assert.equal(typeof api.beforeNativeRoll,'function');const order=[];
 await api.beforeNativeRoll({showDialog:false,commit:async()=>order.push('payment'),native:async()=>order.push('roll')});assert.deepEqual(order,['payment','roll']);
});
test('a stale native acceptance rechecks authority even before its disconnect hook arrives',async()=>{
 let alive=true,commits=0,resolved;const entries=new Map();let seq=0;
 const Hooks={on(event,fn){const id=++seq;entries.set(id,{event,fn});return id},off(_event,id){entries.delete(id)}};
 await assert.rejects(api.beforeNativeRoll({Hooks,marker:'operation',showDialog:true,assertLive:()=>{if(!alive)throw Error('owner disconnected')},commit:async()=>{commits++},native:async()=>{
  const app={context:{options:new Set(['operation'])},resolve:value=>{resolved=value}};
  for(const e of entries.values())if(e.event==='renderCheckModifiersDialog')e.fn(app);
  alive=false;await app.resolve(true);
 }}),/owner disconnected/);
 assert.equal(commits,0);assert.equal(resolved,false);assert.equal(entries.size,0);
});
test('the same submission guard waits on the actual damage dialog hook',async()=>{
 let hook,commits=0,resolved;const Hooks={on(event,fn){assert.equal(event,'renderDamageModifierDialog');hook=fn;return 1},off(){hook=null}};
 await api.beforeNativeRoll({Hooks,marker:'damage',dialogKind:'damage',showDialog:true,commit:async()=>{commits++},native:async()=>{const app={context:{options:new Set(['damage'])},resolve:value=>resolved=value};hook(app);await app.resolve(true)}});
 assert.equal(commits,1);assert.equal(resolved,true);assert.equal(hook,null);
});

function actualEventParams(game,event,type){
 const path=process.env.PF2E_NATIVE_BUNDLE;assert.ok(path,'the actual PF2e source is required');
 const source=fs.readFileSync(path,'utf8'),start=source.indexOf('function isRelevantEvent('),end=source.indexOf('function eventToMessageMode(',start);
 assert.ok(start>=0&&end>start);return new Function('game',`${source.slice(start,end)};return eventToRollParams;`)(game)(event,{type});
}
test('automation attack still opens the actual native window when the owner disabled quick-roll dialogs',()=>setup({},async f=>{
 await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]});
 assert.equal(actualEventParams(f.ownerGame,f.calls[0].options.event,'check').skipDialog,false);
}));
test('automation weapon damage still opens the actual native window when its owner disabled dialogs',()=>setup({},async f=>{
 let request;f.actor.system.actions[0].damage=async options=>{request=options;return {toJSON:()=>({formula:'1d6',total:3})};};
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'damage',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]});
 assert.equal(result.status,'rolled');assert.equal(actualEventParams(f.ownerGame,request.event,'damage').skipDialog,false);
}));
test('closing the forced owner window does not prepay even when the owner disabled dialogs',()=>setup({},async f=>{
 let payments=0;
 f.actor.system.actions[0].variants[0].roll=async options=>{
  const app={context:{options:options.options},resolve:accepted=>{assert.equal(accepted,false);}};
  f.hooks.call('renderCheckModifiersDialog',app);await app.resolve(false);return null;
 };
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]},null,async()=>{payments++;});
 assert.equal(result.status,'cancelled');assert.equal(payments,0);
 assert.notEqual(Object.values(f.card.flags[ID].nativeOwnerOperations)[0].committed,true);assert.equal(f.gmGame.messages.has('check'),false);
}));
for(const mode of ['blind','self'])test(`owner weapon damage retains the ${mode} audience selected in its native window`,()=>setup({},async f=>{
 f.Message.applyMode=(data,value)=>({...data,blind:value==='blind',whisper:value==='self'?['player']:['gm']});
 f.actor.system.actions[0].damage=async options=>{
  f.hooks.call('renderDamageModifierDialog',{context:{options:options.options,messageMode:mode}});
  return {toJSON:()=>({formula:'1d6',total:3})};
 };
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'damage',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]});
 assert.deepEqual(result.privacy,{messageMode:mode==='self'?'gm':mode,blind:mode==='blind',whisper:mode==='self'?['player']:['gm']});
 const stored=Object.values(f.card.flags[ID].nativeOwnerOperations)[0].result;
 assert.deepEqual(stored,{status:'rolled',messageId:result.messageId});
 assert.equal(f.gmGame.messages.get(result.messageId).flags[ID].nativeOwnerDamageResult.privacy.messageMode,mode,'the private native selection remains unchanged');
 assert.equal(f.hookEntries.size,0);
}));
test('owner weapon damage closes on lost ownership before any actual damage evaluation',()=>setup({},async f=>{
 let dice=0,accept;const opened=new Promise(resolve=>accept=resolve);
 f.actor.system.actions[0].damage=async options=>{let resolve;const pending=new Promise(done=>resolve=done);const app={context:{options:options.options,messageMode:'public'},resolve,close(){}};f.hooks.call('renderDamageModifierDialog',app);accept();if(await pending){dice++;return {toJSON:()=>({formula:'1d6',total:3})}}return null;};
 const operation=f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'damage',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]});await opened;
 f.player.active=false;f.hooks.call('userConnected',f.player,false);await assert.rejects(operation,/离线|权限|身份/);await new Promise(resolve=>setImmediate(resolve));assert.equal(dice,0);assert.equal(f.hookEntries.size,0);
}));
test('owner damage retains its native modifier context and cannot widen the blind source audience',()=>setup({},async f=>{
 f.Message.applyMode=(data,value)=>({...data,blind:value==='blind',whisper:value==='public'?[]:['gm']});
 f.actor.system.actions[0].damage=async options=>{f.hooks.call('renderDamageModifierDialog',{context:{options:new Set([...options.options,'item:trait:magical']),domains:['damage','strike-damage'],traits:['magical'],messageMode:'public'}});return {toJSON:()=>({formula:'1d6',total:3})};};
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'damage',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[],minimumPrivacy:{blind:true,whisper:['gm']}});
 assert.deepEqual(result.privacy,{messageMode:'blind',blind:true,whisper:['gm']});assert.ok(result.context,'native damage context must survive the owner boundary');assert.ok(result.context.options.includes('item:trait:magical'));assert.deepEqual(result.context.domains,['damage','strike-damage']);assert.deepEqual(result.context.traits,['magical']);assert.equal(f.hookEntries.size,0);
}));

function privateDamageFixture(f,type,{cancel=false}={}){
 const rollJSON={class:'DamageRoll',formula:'1d6[fire]',total:3,evaluated:true,options:{secret:'hidden-result'},terms:[]},context={options:new Set(['hidden-damage-option']),domains:['damage'],traits:['fire'],messageMode:'blind'};
 f.Message.applyMode=(data,mode)=>({...data,blind:mode==='blind',whisper:mode==='public'?[]:['gm']});
 const checkContext={dc:{value:51,visible:false},roll:{total:37},options:['hidden-check-option']},request={type,targetUuid:f.target.uuid,options:['hidden-request-option'],minimumPrivacy:{blind:true,whisper:['gm']}};
 let rolls=0;
 if(type==='damage'){
  Object.assign(request,{weaponId:'weapon',map:0,checkContext});
  f.actor.system.actions[0].damage=async options=>{assert.equal(f.getCurrent().user.id,'player');assert.deepEqual(options.checkContext,checkContext);f.hooks.call('renderDamageModifierDialog',{context:{...context,options:new Set([...options.options,...context.options])}});if(cancel)return null;rolls++;return {toJSON:()=>rollJSON};};
 }else{
  Object.assign(request,{spellId:'spell',rank:2,overlayIds:['overlay'],checkContext});
  f.actor.items.set('spell',{id:'spell',type:'spell',loadVariant:({castRank,overlayIds})=>{assert.equal(castRank,2);assert.deepEqual(overlayIds,['overlay']);return {getDamage:async parameters=>{assert.equal(f.getCurrent().user.id,'player');assert.equal(parameters.skipDialog,false);assert.equal(parameters.target,f.target);if(cancel)return null;return {context,template:{damage:{roll:{evaluate:async()=>{rolls++;return {toJSON:()=>rollJSON}}}}}};}}}});
 }
 return {request,rollJSON,context,rolls:()=>rolls};
}
for(const type of ['damage','spell-damage']){
 test(`${type} keeps its secret request and result out of public activity flags`,()=>setup({},async f=>{
  const native=privateDamageFixture(f,type),send=f.socket.executeAsUser;let payload,reply;
  f.socket.executeAsUser=async(...args)=>{payload=args[2];return reply=await send(...args)};
  const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},native.request);
  const operation=Object.values(f.card.flags[ID].nativeOwnerOperations)[0];
  assert.equal(Object.hasOwn(operation.request,'checkContext'),false,'hidden check DC and total must not be on the public activity');
  assert.equal(Object.hasOwn(operation.request,'options'),false);assert.equal(Object.hasOwn(operation.request,'minimumPrivacy'),false);
  assert.deepEqual(operation.result,{status:'rolled',messageId:result.messageId});assert.deepEqual(reply,operation.result);
  assert.deepEqual(payload.privateRequest.checkContext,native.request.checkContext);assert.deepEqual(payload.privateRequest.options,native.request.options);
  assert.equal(JSON.stringify(operation).includes('hidden-'),false);assert.equal(JSON.stringify(operation).includes('"total"'),false);
  const receipt=f.gmGame.messages.get(result.messageId),proof=receipt.flags[ID].nativeOwnerDamageResult;
  assert.equal(receipt.author,'player');assert.equal(receipt.blind,true);assert.deepEqual(receipt.whisper,['gm']);assert.deepEqual(receipt.rolls,[]);assert.equal(receipt.flags.pf2e,undefined);
  assert.equal(proof.type,type);assert.equal(proof.actorUuid,f.actor.uuid);assert.equal(proof.nonce,operation.nonce);
  assert.deepEqual(proof.roll,native.rollJSON);assert.deepEqual(result.roll,native.rollJSON,'legacy consumers still receive roll JSON, not an evaluated instance');
  assert.deepEqual(result.privacy,{messageMode:'blind',blind:true,whisper:['gm']});assert.deepEqual(result.context.domains,['damage']);assert.deepEqual(result.context.traits,['fire']);assert.ok(result.context.options.includes('hidden-damage-option'));
  assert.equal(native.rolls(),1);assert.equal(f.hookEntries.size,0);
 }));
 test(`${type} recovers its private saved result after a lost owner RPC reply`,()=>setup({},async f=>{
  const native=privateDamageFixture(f,type),send=f.socket.executeAsUser;f.socket.executeAsUser=async(...args)=>{await send(...args);return new Promise(()=>{})};let timer;
  try{const result=await Promise.race([f.root.run({actor:f.actor,message:f.card,user:f.player},native.request),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('private damage result blocked by lost reply')),100)})]);
   assert.ok(result.messageId,'recovery must use a persisted private receipt');assert.deepEqual(result.roll,native.rollJSON);assert.equal(native.rolls(),1);assert.equal(f.hookEntries.size,0);
  }finally{clearTimeout(timer)}
 }));
 test(`${type} cancellation saves no receipt and does not copy the private request into flags`,()=>setup({},async f=>{
  const native=privateDamageFixture(f,type,{cancel:true}),result=await f.root.run({actor:f.actor,message:f.card,user:f.player},native.request);
  assert.deepEqual(result,{status:'cancelled'});assert.equal(f.gmGame.messages.size,1);assert.equal(native.rolls(),0);
  const operation=Object.values(f.card.flags[ID].nativeOwnerOperations)[0];assert.equal(Object.hasOwn(operation.request,'checkContext'),false);assert.equal(f.hookEntries.size,0);
 }));
 test(`${type} rejects a private request whose source identity differs from the public operation`,()=>setup({},async f=>{
  const native=privateDamageFixture(f,type),send=f.socket.executeAsUser;
  f.socket.executeAsUser=(name,userId,payload)=>send(name,userId,{...payload,privateRequest:{...payload.privateRequest,...type==='damage'?{weaponId:'another-weapon'}:{spellId:'another-spell'}}});
  await assert.rejects(f.root.run({actor:f.actor,message:f.card,user:f.player},native.request),/认证请求/);assert.equal(native.rolls(),0);assert.equal(f.gmGame.messages.size,1);assert.equal(f.hookEntries.size,0);
 }));
 for(const changed of ['audience','nonce','type'])test(`${type} rejects a private receipt with changed ${changed} without repeating native dice`,()=>setup({},async f=>{
  const native=privateDamageFixture(f,type),create=f.Message.create;
  f.Message.create=async data=>{const receipt=await create(data);if(changed==='audience')receipt.whisper.push('player');else receipt.flags[ID].nativeOwnerDamageResult[changed]=changed==='nonce'?'another-operation':type==='damage'?'spell-damage':'damage';return receipt;};
  await assert.rejects(f.root.run({actor:f.actor,message:f.card,user:f.player},native.request),/伤害凭据/);assert.equal(native.rolls(),1,'a rejected saved proof cannot replay its original native dice');assert.equal(f.hookEntries.size,0);
  const result=Object.values(f.card.flags[ID].nativeOwnerOperations)[0].result;assert.deepEqual(Object.keys(result).sort(),['messageId','status']);
 }));
}
