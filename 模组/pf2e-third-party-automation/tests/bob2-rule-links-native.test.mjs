import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createNativeOwnerOperations} from '../scripts/native-owner-operations.mjs';
import {Predicate} from './knowledge-pf2e-861-fixture.mjs';
let api={};try{api=await import('../scripts/bob2-rule-links-native.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
test('Uplifting delegates to native Aid with Performance and publishes its actual result once',async()=>{
 const actor={},target={},roll={total:23},message={},calls=[],game={pf2e:{actions:new Map([['aid',{async use(options){calls.push(options);return [{actor,roll,message,outcome:'success'}]}}]])}};
 let published;const result=await api.rollNativeUpliftingAid?.({game,actor,target,dc:{value:15},options:['marker'],event:{shiftKey:true},callback:async(...args)=>{published=args}});
 assert.equal(result,roll);assert.equal(calls.length,1);assert.equal(calls[0].statistic,'performance');assert.equal(calls[0].actors,actor);assert.equal(calls[0].target,target);assert.deepEqual(calls[0].message,{create:false});assert.deepEqual(calls[0].rollOptions,['marker']);assert.deepEqual(published,[roll,'success',message]);assert.equal(calls[0].modifiers,undefined);
});
test('cancelled native Aid does not publish or manufacture a result',async()=>{
 let published=0;const game={pf2e:{actions:new Map([['aid',{use:async()=>[]}]])}};
 assert.equal(await api.rollNativeUpliftingAid?.({game,actor:{},dc:{value:15},options:[],callback:()=>published++}),null);assert.equal(published,0);
});

const bundle=process.env.PF2E_NATIVE_861_BUNDLE;
test('native circumstance stacking and Aid consumption retain one-use semantics even under a larger bonus',{skip:!bundle},async()=>{
 const source=fs.readFileSync(bundle,'utf8'),start=source.indexOf('var HIGHER_BONUS ='),end=source.indexOf('var StatisticModifier =',start);
 assert.ok(start>0&&end>start);const stack=Function(`${source.slice(start,end)};return applyStackingRules`)();
 assert.equal(stack([{type:'circumstance',modifier:2,ignored:false},{type:'circumstance',modifier:3,ignored:false}]),3);
 assert.equal(stack([{type:'circumstance',modifier:4,ignored:false},{type:'circumstance',modifier:2,ignored:false}]),4);
 const first=source.indexOf('async afterRoll({ check: e, rollOptions: t })'),last=source.indexOf('\n}, ',first);assert.ok(first>0&&last>first);
 const afterRoll=Function(`return ({${source.slice(first,last)}}).afterRoll`)();let consumed=0;
 const rule={ignored:false,removeAfterRoll:new Predicate(['check:statistic:acrobatics']),item:{isOfType:type=>type==='effect',delete:async()=>consumed++}};
 await afterRoll.call(rule,{check:{modifiers:[]},rollOptions:new Set(['check:statistic:stealth'])});assert.equal(consumed,0);
 await afterRoll.call(rule,{check:{modifiers:[{rule,enabled:false}]},rollOptions:new Set(['check:statistic:acrobatics'])});assert.equal(consumed,1);
});

function ownerFixture(accepted){
 const id='pf2e-third-party-automation',user={id:'gm',isGM:true,active:true,settings:{showCheckDialogs:false}},users=Object.assign(new Map([['gm',user]]),{activeGM:user}),messages=new Map(),docs=new Map(),listeners=new Map(),order=[];let seq=0;
 const Hooks={on(name,fn){const key=++seq;listeners.set(key,{name,fn});return key},off(_name,key){listeners.delete(key)},call(name,...args){for(const e of [...listeners.values()])if(e.name===name)e.fn(...args)}};
 const actor={id:'bard',uuid:'Actor.bard',items:new Map(),getStatistic:()=>({check:{roll(){throw Error('expected native Aid, not a replacement statistic call')}}}),testUserPermission:()=>true};
 const item={id:'aid',uuid:actor.uuid+'.Item.aid',actor,sourceId:'Compendium.pf2e.actionspf2e.Item.HCl3pzVefiv9ZKQW'},token={uuid:'Scene.s.Token.bard',actor},target={uuid:'Scene.s.Token.ally',actor:{uuid:'Actor.ally'},object:{}};actor.items.set(item.id,item);for(const doc of [actor,item,token,target])docs.set(doc.uuid,doc);
 function assign(o,data){for(const [path,value]of Object.entries(data)){const parts=path.split('.');let at=o;for(const key of parts.slice(0,-1))at=at[key]??={};at[parts.at(-1)]=value}}
 class Message{constructor(data){Object.assign(this,data)}toObject(){return {...this}}static async create(data){const m=new Message({...data,id:'roll'});messages.set(m.id,m);order.push('publish');Hooks.call('createChatMessage',m);return m}}
 const card={id:'use',author:user,flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid}}},async update(data){assign(this,data);Hooks.call('updateChatMessage',this)}};messages.set(card.id,card);
 const game={user,users,messages,pf2e:{actions:new Map([['aid',{async use(options){
  assert.equal(options.statistic,'performance');assert.equal(options.actors,actor);assert.equal(options.target,target.object);assert.equal(options.modifiers,undefined);order.push('dialog');
  let resolve;const decided=new Promise(done=>resolve=done),app={context:{options:new Set(options.rollOptions)},resolve};Hooks.call('renderCheckModifiersDialog',app);await app.resolve(accepted);if(!await decided)return [];
  order.push('roll');const roll={_evaluated:true,total:23,options:{degreeOfSuccess:2}},message=new Message({speaker:{actor:actor.id},rolls:[roll],flags:{pf2e:{context:{type:'skill-check',dc:options.difficultyClass,outcome:'success',options:[...options.rollOptions,'action:aid','action:aid:performance','check:statistic:performance']}}}});
  return [{actor,roll,message,outcome:'success'}];
 }}]])}};
 const provider=createNativeOwnerOperations({game,fromUuid:async uuid=>docs.get(uuid),scope:'bob2-rules'});provider.register({Hooks});
 return {provider,Message,order,listeners,context:{actor,message:card,user},request:{type:'check',nativeAction:'aid',action:'aid',statistic:'performance',itemUuid:item.uuid,sourceUuid:item.sourceId,tokenUuid:token.uuid,targetUuid:target.uuid,targetActorUuid:target.actor.uuid,dc:{value:15},options:['action:aid']}};
}
for(const accepted of [false,true])test(`owner operation routes real Aid ${accepted?'acceptance':'cancellation'} through one dialog and commit`,async()=>{
 const f=ownerFixture(accepted),prior=globalThis.CONFIG;globalThis.CONFIG={ChatMessage:{documentClass:f.Message}};
 try{const result=await f.provider.run(f.context,f.request,null,async()=>f.order.push('commit'));assert.equal(result.status,accepted?'rolled':'cancelled');assert.deepEqual(f.order,accepted?['dialog','commit','roll','publish']:['dialog']);assert.equal(f.listeners.size,0);}finally{globalThis.CONFIG=prior}
});
