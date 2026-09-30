import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID} from '../scripts/rules.mjs';
import {SCARE_SOURCE} from '../scripts/scare-to-death-rules.mjs';
import {createScareExecutor} from '../scripts/scare-to-death-executor.mjs';

const tick=()=>new Promise(resolve=>setImmediate(resolve));
function fixture({lostReply=false,stage='intimidation'}={}){
 const listeners=new Map();let seq=0,requests=0,rolls=0,ownerTask,dialog;
 const Hooks={on(event,fn){const id=++seq;listeners.set(id,{event,fn});return id},off(_event,id){listeners.delete(id)},call(event,...args){for(const {event:kind,fn}of [...listeners.values()])if(kind===event)fn(...args)}};
 const gm={id:'gm',isGM:true,active:true},user={id:'player',active:true},saver={id:'save-player',active:true},users=new Map([gm,user,saver].map(u=>[u.id,u]));users.activeGM=gm;
 const actor={id:'a',uuid:'Actor.a',flags:{},items:new Map(),testUserPermission:u=>u===user},patient={id:'b',uuid:'Actor.b',flags:{},items:new Map(),testUserPermission:u=>u===saver};
 const scene={id:'s',tokens:new Map()},origin={id:'o',uuid:'Scene.s.Token.o',parent:scene,actor,object:{}},target={id:'t',uuid:'Scene.s.Token.t',parent:scene,actor:patient,object:{}};scene.tokens.set(origin.id,origin);scene.tokens.set(target.id,target);actor.getActiveTokens=()=>[origin];patient.getActiveTokens=()=>[target];
 const item={id:'feat',uuid:'Actor.a.Item.feat',actor,_stats:{compendiumSource:SCARE_SOURCE},system:{traits:{value:[]}}};actor.items.set(item.id,item);
 const claim={usageId:'usage',actorUuid:actor.uuid,originUuid:origin.uuid,targetUuid:target.uuid,targetActorUuid:patient.uuid,itemUuid:item.uuid,userId:user.id,saveUserId:saver.id,penalty:0,nonce:'nonce',state:stage+'-ready',...stage==='fortitude'?{checkId:'intimidation'}:{}};
 const usage={id:'usage',flags:{[MODULE_ID]:{scare:{claim}}}},messages=new Map([[usage.id,usage]]),objects=new Map([actor,patient,origin,target,item].map(d=>[d.uuid,d]));
 const game={user:gm,users,actors:new Map([actor,patient].map(a=>[a.id,a])),messages,scenes:new Map([[scene.id,scene]])},ownerGame={...game,user:stage==='fortitude'?saver:user},fromUuid=async uuid=>objects.get(uuid);
 const fingerprint=JSON.stringify(['usage','Actor.a','Scene.s.Token.o','Scene.s.Token.t','Actor.b','Actor.a.Item.feat','player','save-player',0,'nonce']);
 for(const a of [actor,patient])a.update=async changes=>{for(const [path,value]of Object.entries(changes)){assert.equal(path,`flags.${MODULE_ID}.scareExecutions`);a.flags[MODULE_ID]={...a.flags[MODULE_ID],scareExecutions:structuredClone(value)}}Hooks.call('updateActor',a,changes);return a};
 function card(kind,id=kind){const save=kind==='fortitude',degree=save?2:3,roller=save?patient:actor,token=save?target:origin,rollerUser=save?saver:user;const roll={_evaluated:true,total:save?24:35,options:{degreeOfSuccess:degree},toJSON(){return {evaluated:true,total:this.total,options:this.options}}};return {id,isCheckRoll:true,author:rollerUser,actor:roller,speaker:{actor:roller.id,scene:'s',token:token.id},rolls:[roll],flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid},context:{origin:{actor:actor.uuid,token:origin.uuid},target:{actor:patient.uuid,token:target.uuid},type:save?'saving-throw':'skill-check',action:'scare-to-death',dc:{slug:save?'intimidation':'will',value:save?32:25},options:[`${MODULE_ID}:scare:nonce:${kind}`,'item:trait:incapacitation'],outcome:save?'success':'criticalSuccess'}}}}}
 if(stage==='fortitude')messages.set('intimidation',card('intimidation'));
 const roller=stage==='fortitude'?patient:actor;
 roller.getStatistic=name=>({roll:async options=>{assert.equal(name,stage);assert.equal(options.skipDialog,false);assert.equal(options.event,null);let complete;const accepted=new Promise(resolve=>complete=resolve);dialog={context:{options:new Set(options.extraRollOptions)},resolve(value){complete(value)},close(){return Promise.resolve()}};Hooks.call('renderCheckModifiersDialog',dialog);if(!await accepted)return null;rolls++;const message=card(stage);messages.set(message.id,message);Hooks.call('createChatMessage',message);await options.callback(message.rolls[0],message.flags.pf2e.context.outcome,message);return message.rolls[0]}});
 const owner=createScareExecutor({game:ownerGame,fromUuid,Hooks,timeoutMs:10}),executor=createScareExecutor({game,fromUuid,Hooks,timeoutMs:10});
 executor.register({socket:{register(){},executeAsUser(_name,id,payload){requests++;assert.equal(id,ownerGame.user.id);ownerTask=owner.ownerRoll(payload,gm);ownerTask.catch(()=>{});return lostReply?new Promise(()=>{}):ownerTask.then(value=>({ok:true,value}),error=>({ok:false,error:error.message}))}}});
 return {game,ownerGame,gm,user,saver,actor,patient,roller,claim,usage,messages,Hooks,listeners,fingerprint,owner,executor,get requests(){return requests},get rolls(){return rolls},get ownerTask(){return ownerTask},get dialog(){return dialog},card};
}

test('a native Scare window may remain open beyond the old RPC deadline before one acceptance',async()=>{
 const f=fixture();let settled=false;const pending=f.executor.roll(f.claim,'intimidation').then(value=>{settled=true;return {value}},error=>{settled=true;return {error}});await new Promise(resolve=>setTimeout(resolve,30));try{assert.equal(settled,false);assert.equal(f.rolls,0);assert.equal(f.messages.size,1);assert.equal(f.requests,1)}finally{f.dialog.resolve(true);await f.ownerTask}const result=await pending;if(result.error)throw result.error;assert.equal(result.value.id,'intimidation');assert.equal(f.rolls,1);assert.equal(f.listeners.size,0);
});

test('the exact saved Scare completion wakes a permanently lost socket reply',async()=>{
 const f=fixture({lostReply:true}),pending=f.executor.roll(f.claim,'intimidation').then(value=>({value}),error=>({error}));await tick();f.dialog.resolve(true);await f.ownerTask;const result=await Promise.race([pending,new Promise(resolve=>setTimeout(()=>resolve({error:Error('saved completion did not wake the caller')}),40))]);if(result.error)throw result.error;assert.equal(result.value.id,'intimidation');assert.equal(f.requests,1);assert.equal(f.rolls,1);assert.equal(f.listeners.size,0);
});

for(const stage of ['intimidation','fortitude'])test(`${stage} owner disconnection releases the wait and cancels the exact native window without rolling`,async()=>{
 const f=fixture({stage,lostReply:true});let result;const pending=f.executor.roll(f.claim,stage).then(value=>result={value},error=>result={error});await tick();f.ownerGame.user.active=false;f.Hooks.call('userConnected',f.ownerGame.user,false);await tick();try{assert.match(result?.error?.message??'',/离线|拥有者/);assert.equal(f.rolls,0)}finally{f.dialog.resolve(true);await f.ownerTask.catch(()=>{});await Promise.race([pending,new Promise(resolve=>setTimeout(resolve,40))])}assert.equal(f.listeners.size,0);assert.equal(f.messages.size,stage==='fortitude'?2:1);await assert.rejects(f.owner.ownerRoll({usageId:'usage',nonce:'nonce',stage},f.gm));assert.equal(f.rolls,0);
});

test('GM handoff releases the wait and a stale native confirmation cannot roll',async()=>{
 const f=fixture({lostReply:true});let result;const pending=f.executor.roll(f.claim,'intimidation').then(value=>result={value},error=>result={error});await tick();f.game.users.activeGM={id:'new-gm',active:true,isGM:true};f.Hooks.call('userConnected',f.gm,true);await tick();try{assert.match(result?.error?.message??'',/主GM/);assert.equal(f.rolls,0)}finally{f.dialog.resolve(true);await f.ownerTask.catch(()=>{});await Promise.race([pending,new Promise(resolve=>setTimeout(resolve,40))])}assert.equal(f.listeners.size,0);assert.equal(f.messages.size,1);
});

test('native acceptance rechecks owner permission even if no disconnect hook arrived',async()=>{
 const f=fixture(),pending=f.executor.roll(f.claim,'intimidation');pending.catch(()=>{});await tick();f.actor.testUserPermission=()=>false;f.dialog.resolve(true);await assert.rejects(pending,/权限|拥有者/);assert.equal(f.rolls,0);assert.equal(f.messages.size,1);assert.equal(f.listeners.size,0);
});

test('native acceptance cannot roll a source actor removed while the native window was open',async()=>{
 const f=fixture(),pending=f.executor.roll(f.claim,'intimidation');pending.catch(()=>{});await tick();f.game.actors.delete(f.actor.id);f.dialog.resolve(true);await assert.rejects(pending);await assert.rejects(f.ownerTask,/来源|角色/);assert.equal(f.rolls,0);assert.equal(f.messages.size,1);assert.equal(f.listeners.size,0);
});

test('closing the native Scare window emits no dice or card and cannot replay a started execution',async()=>{
 const f=fixture(),pending=f.executor.roll(f.claim,'intimidation');pending.catch(()=>{});await tick();f.dialog.resolve(false);await assert.rejects(pending);assert.equal(f.rolls,0);assert.equal(f.messages.size,1);assert.equal(f.listeners.size,0);assert.equal(f.actor.flags[MODULE_ID].scareExecutions[0].status,'uncertain');await assert.rejects(f.owner.ownerRoll({usageId:'usage',nonce:'nonce',stage:'intimidation'},f.gm),/不能重掷/);
});

test('an uncertain saved Scare execution releases a lost reply without retrying',async()=>{
 const f=fixture({lostReply:true});let result;const pending=f.executor.roll(f.claim,'intimidation').then(value=>result={value},error=>result={error});await tick();f.dialog.resolve(false);await f.ownerTask.catch(()=>{});await tick();try{assert.match(result?.error?.message??'',/未确认|不能重掷/);assert.equal(f.requests,1);assert.equal(f.rolls,0)}finally{await Promise.race([pending,new Promise(resolve=>setTimeout(resolve,40))])}assert.equal(f.listeners.size,0);
});

test('a saved done row with a different fingerprint never substitutes for this native Scare request',async()=>{
 const f=fixture({lostReply:true});let result;const pending=f.executor.roll(f.claim,'intimidation').then(value=>result={value},error=>result={error});await tick();f.actor.flags[MODULE_ID].scareExecutions=[{usageId:'usage',stage:'intimidation',fingerprint:'other',status:'done',messageId:'forged'}];f.Hooks.call('updateActor',f.actor);await tick();try{assert.match(result?.error?.message??'',/认领|记录|身份/)}finally{f.dialog.resolve(false);await f.ownerTask.catch(()=>{});await Promise.race([pending,new Promise(resolve=>setTimeout(resolve,40))])}assert.equal(f.rolls,0);assert.equal(f.listeners.size,0);
});

test('a completed saved native Scare remains authoritative when its owner disconnects as it is persisted',async()=>{
 const f=fixture({lostReply:true}),update=f.actor.update;f.actor.update=async changes=>{if(changes[`flags.${MODULE_ID}.scareExecutions`]?.[0]?.status==='done')f.user.active=false;return update(changes)};
 const pending=f.executor.roll(f.claim,'intimidation');await tick();f.dialog.resolve(true);const card=await pending;await f.ownerTask;assert.equal(card.id,'intimidation');assert.equal(f.rolls,1);assert.equal(f.listeners.size,0);
});

test('saved native completion still requires its exact message ID to synchronize within the bounded document wait',async()=>{
 const f=fixture({lostReply:true}),pending=f.executor.roll(f.claim,'intimidation');pending.catch(()=>{});await tick();
 f.actor.flags[MODULE_ID].scareExecutions=[{usageId:'usage',stage:'intimidation',fingerprint:f.fingerprint,status:'done',messageId:'missing-native-card'}];f.Hooks.call('updateActor',f.actor);
 try{await assert.rejects(pending,/原生消息同步超时/);assert.equal(f.rolls,0)}finally{f.dialog.resolve(false);await f.ownerTask.catch(()=>{})}assert.equal(f.listeners.size,0);
});
