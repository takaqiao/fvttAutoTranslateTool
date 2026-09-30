import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHalflingLuckProvider} from '../scripts/halfling-luck.mjs';
import {assessHalflingLuck} from '../scripts/halfling-luck-rules.mjs';
import {createHalflingLuckLedger} from '../scripts/halfling-luck-ledger.mjs';

const ID='pf2e-third-party-automation',SOURCE='Compendium.pf2e.feats-srd.Item.ZbRVqf14RTJJIZXG';
const copy=v=>structuredClone(v);
class Draft{
 constructor(data){this.data=copy(data);}
 get flags(){return this.data.flags;}
 toObject(){return copy(this.data);}
 updateSource(data){this.data={...this.data,...copy(data)};}
}
function fixture({answer='use',eligible=true,createMessage=false,callback=true}={}){
 const user={id:'gm',isGM:true,active:true},actor={id:'a',uuid:'Actor.a',type:'character',canAct:true,isDead:false,flags:{'pf2e-reaction':{reaction:false}},testUserPermission:u=>u===user,items:new Map()};
 const item={id:'luck',uuid:'Actor.a.Item.luck',type:'feat',sourceId:SOURCE,actor,system:{actionType:{value:'free'},frequency:{value:1,max:1,per:'day'}},flags:{}};actor.items.set(item.id,item);
 const users=new Map([[user.id,user]]);users.activeGM=user;
 const game={user,users,actors:new Map([[actor.id,actor]]),messages:new Map(),pf2e:{Check:{renderReroll:async r=>String(r.total)}}};
 const hooks=new Map(),rpc=new Map(),Hooks={on(name,fn){const list=hooks.get(name)??[];list.push(fn);hooks.set(name,list);return fn;},off(name,fn){hooks.set(name,(hooks.get(name)??[]).filter(v=>v!==fn));}};
 const emit=(name,...args)=>Promise.all((hooks.get(name)??[]).map(fn=>fn(...args)));
 const calls=[],nativeCalls=[],callbacks=[],published=[],prompts=[],errors=[];let record=null,authorized=null,provider,unregister,seq=0;
 const ledger={
  current:()=>copy(record),
  async claim(p){calls.push(['claim',p]);assert.equal(record,null);record={nonce:'claim-nonce',paymentNonce:'payment',status:'claimed',userId:p.user.id,gmId:user.id,invocationId:p.invocationId,fingerprint:p.fingerprint};return copy(record);},
  authorizePayment(i,n,u){calls.push(['authorize',n]);assert.equal(n,record.nonce);authorized=n;},
  clearAuthorization(i,n){calls.push(['clear',n]);authorized=null;},
  async cancelClaim(p){calls.push(['cancelClaim',p]);record.status='cancelled';return copy(record);},
  async bindUsage(p){calls.push(['bindUsage',p]);assert.equal(record.status,'paid');assert.equal(p.message.flags[ID].halflingLuckInput.nonce,record.nonce);record.status='ready';return copy(record);},
  async startRolling(p){calls.push(['startRolling',p]);assert.equal(record.status,'ready');record.status='rolling';return copy(record);},
  async recordResult(p){calls.push(['recordResult',p]);assert.equal(record.status,'rolling');record.status='result-ready';record.rollJSON=copy(p.rollJSON);return copy(record);},
  async beginDelivery(p){calls.push(['beginDelivery',p]);assert.equal(record.status,'result-ready');record.status='delivering';return copy(record);},
  async finishDelivery(p){calls.push(['finishDelivery',p]);assert.equal(record.status,'delivering');record.status='callback-returned';return copy(record);},
  async uncertain(p){calls.push(['uncertain',p]);if(record)record.status='uncertain';return copy(record);},
  preparePayment(){},observePayment(){},
 };
 const rolls=[{total:8,degree:1},{total:2,degree:0}].map(({total,degree})=>({total,options:{degreeOfSuccess:degree},toJSON(){return {total:this.total,options:copy(this.options)};}}));
 const check={slug:'will',modifiers:[{slug:'native',modifier:5}]},context={actor,type:'saving-throw',domains:['will'],options:new Set(),dc:{value:18},createMessage};
 const native=async(c,ctx,event,cb)=>{const i=nativeCalls.length,r=rolls[i];nativeCalls.push({check:c,context:ctx,event});const outcome=i?'criticalFailure':'failure';
  const draft=new Draft({author:user.id,blind:false,whisper:[],speaker:{actor:actor.id},flags:{pf2e:{context:{type:ctx.type,outcome,options:[...ctx.options],isReroll:!!ctx.isReroll}}},rolls:[r.toJSON()]});
  await cb(r,outcome,draft,event);return r;};
 const options={game,ledger,assess:()=>({eligible,reason:eligible?null:'manual-boundary'}),fromUuid:async uuid=>uuid===item.uuid?item:uuid===actor.uuid?actor:null,
  choose:async p=>{prompts.push(p);return answer;},randomId:()=>`nonce-${++seq}`,onError:e=>errors.push(e),
  publish:async data=>{published.push(copy(data));return new Draft(data);},
  originalUse:async i=>{calls.push(['originalUse',i.uuid]);assert.equal(authorized,record.nonce);item.system.frequency.value=0;record.status='paid';const input=provider.captureUsage(item);
   const message={id:'use',uuid:'ChatMessage.use',flags:{[ID]:input},author:user,speaker:{actor:actor.id}};game.messages.set(message.id,message);
   await provider.executeUsage({actor,item,user,message,frequencyReceipt:{id:'frequency',itemUuid:item.uuid,userId:user.id,before:1,after:0}});},
 };
 function make(){unregister?.();provider=createHalflingLuckProvider(options);unregister=provider.register?.({Hooks,socket:{register:(name,fn)=>rpc.set(name,fn)}});return provider;}
 make();
 const f={game,user,actor,item,ledger,options,check,context,native,calls,nativeCalls,callbacks,published,prompts,errors,rolls,rpc,emit,
  get provider(){return provider;},get record(){return record;},get authorized(){return authorized;},make,
  run:()=>provider.interceptCheck(f.native,check,context,{original:true},callback?async(...args)=>{callbacks.push(args);}:undefined)};return f;
}

test('eligible original failure uses original feat once, keeps worse new native roll, no bonus/reaction and delivers once',async()=>{
 const f=fixture(),reaction=copy(f.actor.flags);const result=await f.run();
 assert.equal(result,f.rolls[1]);assert.equal(f.nativeCalls.length,2);assert.equal(f.prompts.length,1);assert.equal(f.callbacks.length,1);assert.equal(f.callbacks[0][0],result);assert.equal(f.callbacks[0][1],'criticalFailure');
 assert.equal(f.nativeCalls[1].check,f.check);assert.equal(f.nativeCalls[1].context.isReroll,true);assert.equal(f.nativeCalls[1].context.skipDialog,true);assert.deepEqual(f.check.modifiers,[{slug:'native',modifier:5}]);
 assert.deepEqual(f.actor.flags,reaction);assert.equal(f.item.system.frequency.value,0);assert.equal(f.calls.filter(c=>c[0]==='originalUse').length,1);assert.equal(f.record.status,'callback-returned');assert.equal(f.authorized,null);
 assert.deepEqual(f.calls.filter(c=>['startRolling','recordResult','beginDelivery','finishDelivery'].includes(c[0])).map(c=>c[0]),['startRolling','recordResult','beginDelivery','finishDelivery']);
 assert.equal(f.published.length,0);assert.ok(f.callbacks[0][2] instanceof Draft);assert.equal(f.rolls[1].options.halflingLuckNonce,f.record.nonce);assert.equal(f.callbacks[0][2].flags.pf2e.context.halflingLuckNonce,f.record.nonce);
});

test('decline and unsupported eligibility retain the original native result without payment',async()=>{
 for(const opts of [{answer:'decline'},{eligible:false}]){const f=fixture(opts);assert.equal(await f.run(),f.rolls[0]);assert.equal(f.nativeCalls.length,1);assert.equal(f.callbacks.length,1);assert.equal(f.item.system.frequency.value,1);assert.equal(f.calls.length,0);assert.equal(f.prompts.length,opts.eligible===false?0:1);}
});

test('native cancellation never prompts, claims, pays or invokes caller',async()=>{
 const f=fixture();f.native=async()=>null;assert.equal(await f.run(),null);assert.equal(f.prompts.length,0);assert.equal(f.callbacks.length,0);assert.equal(f.calls.length,0);
});

test('published mode emits only one final result; no original callback still completes return delivery',async()=>{
 for(const callback of [true,false]){const f=fixture({createMessage:true,callback});assert.equal(await f.run(),f.rolls[1]);assert.equal(f.published.length,1);assert.equal(f.callbacks.length,callback?1:0);assert.equal(f.record.status,'callback-returned');}
});

test('a card-return without exact ready payment cannot authorize native reroll',async t=>{
 t.mock.timers.enable({apis:['setTimeout']});const f=fixture();let used;const entered=new Promise(r=>used=r);
 f.options.originalUse=async()=>{used();return {id:'unproven-card'};};f.make();const task=f.run();await entered;t.mock.timers.tick(15001);
 await assert.rejects(task,/回执|付款|等待|payment/i);assert.equal(f.nativeCalls.length,1);assert.equal(f.callbacks.length,0);assert.equal(f.item.system.frequency.value,1);assert.equal(f.record.status,'uncertain');assert.equal(f.authorized,null);
});

test('native reroll or callback error stays uncertain with no repeated Use, die or callback',async()=>{
 for(const phase of ['reroll','callback']){const f=fixture();let deliveries=0;const native=f.native;
  f.native=async(...args)=>{if(phase==='reroll'&&f.nativeCalls.length===1)throw Error('native disconnected');return native(...args);};
  const cb=async()=>{deliveries++;throw Error('caller failed');};
  await assert.rejects(f.provider.interceptCheck(f.native,f.check,f.context,null,cb),phase==='reroll'?/native disconnected/:/caller failed/);
  assert.equal(f.record.status,'uncertain');assert.equal(f.calls.filter(c=>c[0]==='originalUse').length,1);assert.equal(deliveries,phase==='callback'?1:0);assert.equal(f.authorized,null);
 }
});

test('manual original Use is not blocked and cannot claim an old or recent check',async()=>{
 const f=fixture();assert.equal(f.provider.beforeUse(f.item),true);assert.equal(f.provider.captureUsage(f.item),null);assert.equal(f.provider.tracksFrequency(f.item),true);
 const result=await f.provider.executeUsage({actor:f.actor,item:f.item,user:f.user,message:{flags:{}}});assert.match(result,/未绑定|手工|原生/);assert.equal(f.calls.length,0);
});

test('only the current GM can prove the exact still-live local invocation; completed scope is erased',async()=>{
 const f=fixture(),claim=f.ledger.claim;let payload;
 f.ledger.claim=async p=>{
  payload={invocationId:p.invocationId,proofNonce:p.invocationId,actorUuid:f.actor.uuid,itemUuid:f.item.uuid,userId:f.user.id,gmId:f.user.id,fingerprint:p.fingerprint};
  const prove=f.rpc.get('halfling-luck:proof');
  assert.equal((await prove.call({socketdata:{userId:'untrusted'}},payload)).ok,false);
  for(const patch of [{proofNonce:'different'},{actorUuid:'Actor.other'},{itemUuid:'Actor.a.Item.other'},{fingerprint:'0'.repeat(64)},{userId:'forged'}])assert.equal((await prove.call({socketdata:{userId:f.user.id}},{...payload,...patch})).ok,false);
  const valid=await prove.call({socketdata:{userId:f.user.id}},payload);assert.equal(valid.ok,true);assert.deepEqual(valid.value,payload);
  return claim(p);
 };
 await f.run();assert.equal((await f.rpc.get('halfling-luck:proof').call({socketdata:{userId:f.user.id}},payload)).ok,false);
});

test('socket ledger endpoint takes transport sender and rejects arbitrary operations',async()=>{
 const f=fixture(),rpc=f.rpc.get('halfling-luck:ledger');f.game.users.set('other',{id:'other',active:true});
 for(const operation of ['authorizePayment','constructor','__proto__','startRolling']){
  const response=await rpc.call({socketdata:{userId:'other'}},{operation,actorUuid:f.actor.uuid,itemUuid:f.item.uuid,userId:f.user.id,nonce:'forged'});
  assert.equal(response.ok,false);
 }
 assert.equal(f.calls.length,0);
});

test('GM handoff after acceptance prevents Use, and handoff after payment prevents any reroll',async()=>{
 for(const phase of ['choice','payment']){
  const f=fixture(),other={id:'next-gm',isGM:true,active:true};f.game.users.set(other.id,other);
  if(phase==='choice')f.options.choose=async()=>{f.game.users.activeGM=other;return 'use';};
  else {const use=f.options.originalUse;f.options.originalUse=async item=>{await use(item);f.game.users.activeGM=other;};}
  f.make();await assert.rejects(f.run(),/主GM/);assert.equal(f.nativeCalls.length,1);assert.equal(f.callbacks.length,0);
  assert.equal(f.calls.filter(c=>c[0]==='originalUse').length,phase==='payment'?1:0);assert.equal(f.authorized,null);
 }
});

test('a changed original context after the choice is not silently admitted under its old fingerprint',async()=>{
 const f=fixture();f.options.choose=async()=>{f.context.dc.value++;return 'use';};f.make();
 assert.equal(await f.run(),f.rolls[0]);assert.equal(f.calls.length,0);assert.equal(f.callbacks.length,1);
});

test('already-rerolled input can pass through unchanged without being mistaken for this providers second die',async()=>{
 const f=fixture({eligible:false});f.context.isReroll=true;assert.equal(await f.run(),f.rolls[0]);assert.equal(f.calls.length,0);assert.equal(f.callbacks.length,1);
});

test('duplicate native second callbacks cannot persist or deliver another result',async()=>{
 const f=fixture(),native=f.native;
 f.native=async(c,ctx,event,callback)=>native(c,ctx,event,async(...args)=>{await callback(...args);if(ctx.isReroll)await callback(...args);});
 await assert.rejects(f.run(),/一次性许可/);assert.equal(f.calls.filter(c=>c[0]==='recordResult').length,1);assert.equal(f.callbacks.length,0);assert.equal(f.record.status,'uncertain');
});

test('original caller publication mode is frozen for assessment rather than inferred from the internal draft',async()=>{
 for(const createMessage of [true,false]){const f=fixture({createMessage});const seen=[];f.options.assess=p=>{seen.push(p.requestedCreateMessage);assert.equal(p.context.createMessage,false);return {eligible:false};};f.make();await f.run();assert.deepEqual(seen,[createMessage]);assert.equal(f.calls.length,0);}
});

test('a real ledger observes the native frequency update, binds original card and grants each stage only once',async()=>{
 const f=fixture({createMessage:true}),docs=new Map([[f.actor.uuid,f.actor],[f.item.uuid,f.item]]),updates=[];
 const fromUuid=async uuid=>docs.get(uuid);let n=0;
 const ledger=createHalflingLuckLedger({game:f.game,fromUuid,randomId:()=>`real-${++n}`});
 const apply=(target,changes)=>{for(const[key,value]of Object.entries(changes)){const parts=key.split('.');let at=target;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=copy(value);}};
 f.item.update=async(changes,options={})=>{updates.push(copy(changes));apply(f.item,changes);await f.emit('updateItem',f.item,changes,options,f.user.id);return f.item;};
 f.options.ledger=ledger;f.options.fromUuid=fromUuid;
 f.options.originalUse=async item=>{
  const changes={'system.frequency.value':0},options={[ID]:{frequencyReceipt:{id:'real-frequency',itemUuid:item.uuid,userId:f.user.id,before:1,after:0,createdAt:100}}};
  assert.ok((await f.emit('preUpdateItem',item,changes,options,f.user.id)).every(v=>v!==false));await item.update(changes,options);
  const message={id:'original-use',uuid:'ChatMessage.original-use',author:f.user,speaker:{actor:f.actor.id},rolls:[],flags:{pf2e:{origin:{uuid:item.uuid,actor:f.actor.uuid,type:'feat'}},[ID]:{...f.provider.captureUsage(item),usageInput:{actualUse:true,frequencyReceiptId:'real-frequency'}}}};
  docs.set(message.uuid,message);f.game.messages.set(message.id,message);
  await f.provider.executeUsage({actor:f.actor,item,user:f.user,message,frequencyReceipt:options[ID].frequencyReceipt});
 };
 f.make();assert.equal(await f.run(),f.rolls[1]);const record=ledger.current(f.item);
 assert.equal(record.status,'callback-returned');assert.equal(record.rollJSON.options.halflingLuckNonce,record.nonce);
 assert.equal(updates.filter(c=>c['system.frequency.value']===0).length,1);assert.equal(f.callbacks.length,1);assert.equal(f.published.length,1);
 await assert.rejects(ledger.startRolling({actor:f.actor,item:f.item,user:f.user,nonce:record.nonce}));
 await assert.rejects(ledger.beginDelivery({actor:f.actor,item:f.item,user:f.user,nonce:record.nonce}));
});

function nativeRuleFixture(t,{createMessage=true,messageMode='public'}={}){
 const f=fixture({createMessage}),before={CONFIG:globalThis.CONFIG,foundry:globalThis.foundry};
 class Die{number=1;faces=20;modifiers=[];results=[{result:3,active:true}];_evaluated=true;}
 class CheckRoll{get isRerollable(){return !this.options.isReroll;}get isReroll(){return this.options.isReroll;}}
 class CheckModifier{}
 class CheckDraft extends Draft{
  constructor(data,r){super(data);this.nativeRoll=r;this.author=f.user;}
  get isCheckRoll(){return true;}
  get speaker(){return this.data.speaker;}
  get blind(){return this.data.blind;}
  get whisper(){return this.data.whisper;}
  get rolls(){return [this.nativeRoll];}
 }
 globalThis.CONFIG={Dice:{rolls:[CheckRoll]},ChatMessage:{documentClass:CheckDraft}};globalThis.foundry={dice:{terms:{Die}}};
 t.after(()=>{for(const[k,v]of Object.entries(before))if(v===undefined)delete globalThis[k];else globalThis[k]=v;});
 f.game.system={id:'pf2e',version:'8.5.1'};f.game.pf2e.CheckModifier=CheckModifier;f.game.pf2e.settings={metagame:{results:true}};
 f.item.system.traits={value:['fortune','halfling']};f.actor.synthetics={rollTwice:{}};Object.setPrototypeOf(f.check,CheckModifier.prototype);f.check.totalModifier=5;
 Object.assign(f.context,{origin:{actor:f.actor,self:true,token:null},target:null,token:null,messageMode,rollTwice:false,substitutions:[],traits:[]});
 f.native=async(check,ctx,event,cb)=>{
  const reroll=!!ctx.isReroll,r=f.rolls[reroll?1:0],outcome=reroll?'criticalFailure':'failure';Object.setPrototypeOf(r,CheckRoll.prototype);r.dice=[new Die()];r._evaluated=true;
  Object.assign(r.options,{type:ctx.type,dice:'1d20',totalModifier:5,domains:[...ctx.domains],rollerId:f.user.id,isReroll:reroll});
  ctx.outcome=outcome;ctx.unadjustedOutcome=outcome;ctx.isReroll=reroll;f.nativeCalls.push({check,context:ctx,event});
  const card=new CheckDraft({author:f.user.id,speaker:{actor:f.actor.id},blind:messageMode==='blind',whisper:messageMode==='public'?[]:['gm'],flags:{pf2e:{modifierName:check.slug,modifiers:[],context:{actor:f.actor.id,token:null,origin:{actor:f.actor.uuid},type:ctx.type,domains:[...ctx.domains],options:[...ctx.options],dc:copy(ctx.dc),isReroll:reroll,outcome,unadjustedOutcome:outcome,rollTwice:false,substitutions:[],messageMode,traits:[]}}},rolls:[r.toJSON()]},r);
  await cb(r,outcome,card,event);return r;
 };
 const assessments=[];f.options.assess=p=>{const result=assessHalflingLuck(p);assessments.push(result);return result;};f.make();return {...f,run:()=>f.provider.interceptCheck(f.native,f.check,f.context,null,async(...args)=>f.callbacks.push(args)),assessments};
}

test('real rules allow a public native contract through the provider despite its internal draft mode',async t=>{
 const f=nativeRuleFixture(t);assert.equal(await f.run(),f.rolls[1]);assert.ok(f.assessments.length>=3);assert.ok(f.assessments.every(r=>r.eligible));assert.equal(f.callbacks.length,1);assert.equal(f.published.length,1);
});

test('real rules keep both public and private unbound no-message callers manual, with no offer or fee',async t=>{
 for(const messageMode of ['public','blind']){const f=nativeRuleFixture(t,{createMessage:false,messageMode});assert.equal(await f.run(),f.rolls[0]);assert.equal(f.prompts.length,0);assert.equal(f.calls.length,0);assert.equal(f.callbacks.length,1);assert.equal(f.published.length,0);assert.equal(f.assessments[0].reason,'manual-unproven-draft-privacy');}
});

test('no online active GM leaves the native invocation untouched',async()=>{
 const f=fixture();f.game.users.activeGM=null;assert.equal(await f.run(),f.rolls[0]);assert.equal(f.calls.length,0);assert.equal(f.prompts.length,0);assert.equal(f.nativeCalls[0].context,f.context);
});

test('automatic frequency hook vetoes lost ability to act before native payment',async()=>{
 const f=fixture();let prepared=0;f.ledger.preparePayment=()=>{prepared++;return true;};
 f.options.originalUse=async item=>{f.actor.canAct=false;const results=await f.emit('preUpdateItem',item,{'system.frequency.value':0},{},f.user.id);assert.ok(results.includes(false));throw Error('native payment vetoed');};f.make();
 await assert.rejects(f.run(),/native payment vetoed/);assert.equal(prepared,0);assert.equal(f.item.system.frequency.value,1);assert.equal(f.nativeCalls.length,1);
});

test('separate roller and GM clients use authenticated proof RPC and cannot replay rolling or delivery',async()=>{
 const f=fixture({createMessage:true}),owner={id:'owner',active:true,isGM:false};f.game.users.set(owner.id,owner);f.actor.testUserPermission=u=>u===f.user||u===owner;
 const handlers=new Map(),wire=[];
 const socket=user=>({register(name,fn){handlers.set(`${user.id}:${name}`,fn);},async executeAsUser(name,target,payload){wire.push({sender:user.id,target,name,payload:copy(payload)});return handlers.get(`${target}:${name}`).call({socketdata:{userId:user.id}},payload);}});
 const Hooks={on:()=>1,off:()=>{}},gm=createHalflingLuckProvider(f.options);gm.register({Hooks,socket:socket(f.user)});
 const clientGame={...f.game,user:owner};let roller;
 roller=createHalflingLuckProvider({...f.options,game:clientGame,originalUse:async item=>{
  f.calls.push(['originalUse',item.uuid]);item.system.frequency.value=0;f.record.status='paid';
  const message={id:'owner-use',uuid:'ChatMessage.owner-use',author:owner,speaker:{actor:f.actor.id},flags:{[ID]:roller.captureUsage(item)}};
  await gm.executeUsage({actor:f.actor,item,user:owner,message,frequencyReceipt:{id:'owner-frequency'}});
 }});roller.register({Hooks,socket:socket(owner)});
 const result=await roller.interceptCheck(f.native,f.check,f.context,null,async(...args)=>f.callbacks.push(args));
 assert.equal(result,f.rolls[1]);assert.equal(f.record.userId,owner.id);assert.equal(f.record.status,'callback-returned');assert.equal(f.callbacks.length,1);assert.equal(f.calls.filter(c=>c[0]==='originalUse').length,1);
 const proof=wire.filter(p=>p.name==='halfling-luck:proof');assert.equal(proof.length,1);assert.equal(proof[0].sender,f.user.id);assert.equal(proof[0].target,owner.id);
 assert.match(proof[0].payload.fingerprint,/^[a-f0-9]{64}$/);
 for(const operation of ['startRolling','beginDelivery']){
  const response=await socket(owner).executeAsUser('halfling-luck:ledger',f.user.id,{operation,actorUuid:f.actor.uuid,itemUuid:f.item.uuid,nonce:f.record.nonce,userId:f.user.id});
  assert.equal(response.ok,false);
 }
 assert.equal(f.callbacks.length,1);assert.equal(f.nativeCalls.length,2);
});

test('loss of ability to act after proven payment records uncertainty without granting a reroll',async()=>{
 const f=fixture(),use=f.options.originalUse;f.options.originalUse=async item=>{await use(item);f.actor.canAct=false;};f.make();
 await assert.rejects(f.run(),/改变/);assert.equal(f.item.system.frequency.value,0);assert.equal(f.record.status,'uncertain');assert.equal(f.nativeCalls.length,1);assert.equal(f.callbacks.length,0);
});

test('unhashable native context keeps original result without prompting or debiting',async()=>{
 const f=fixture();f.context.dc.circular=f.context.dc;assert.equal(await f.run(),f.rolls[0]);assert.equal(f.calls.length,0);assert.equal(f.prompts.length,0);assert.equal(f.callbacks.length,1);
});

test('GM handoff while rendering a paid reroll cannot publish its final card',async()=>{
 const f=fixture({createMessage:true}),next={id:'next-gm',active:true,isGM:true};f.game.users.set(next.id,next);
 f.game.pf2e.Check.renderReroll=async roll=>{f.game.users.activeGM=next;return String(roll.total);};
 await assert.rejects(f.run(),/主GM/);assert.equal(f.nativeCalls.length,2);assert.equal(f.callbacks.length,0);assert.equal(f.published.length,0);assert.equal(f.record.status,'result-ready');
});

test('a returned final callback can make its actor unable to act without corrupting completed delivery',async()=>{
 const f=fixture({createMessage:true});let delivered=0;
 assert.equal(await f.provider.interceptCheck(f.native,f.check,f.context,null,async()=>{delivered++;f.actor.canAct=false;}),f.rolls[1]);
 assert.equal(delivered,1);assert.equal(f.published.length,1);assert.equal(f.record.status,'callback-returned');assert.equal(f.calls.filter(c=>c[0]==='uncertain').length,0);
});

test('an evaluated rerolls result can still be recorded and delivered after its actor loses ability to act',async()=>{
 const f=fixture({createMessage:true}),native=f.native;
 f.native=async(c,ctx,event,callback)=>native(c,ctx,event,async(...args)=>{if(ctx.isReroll)f.actor.canAct=false;return callback(...args);});
 assert.equal(await f.run(),f.rolls[1]);assert.equal(f.callbacks.length,1);assert.equal(f.record.status,'callback-returned');
});

test('unrelated item changes bypass automatic-payment preflight even when the actor becomes unable to act',async()=>{
 const f=fixture();let prepared=0;f.ledger.preparePayment=()=>{prepared++;return undefined;};
 f.options.originalUse=async item=>{f.actor.canAct=false;
  for(const changes of [{name:'unrelated'}, {[`flags.${ID}.halflingLuck.operations.pending.status`]:'ready'},{system:{description:{value:'updated'}}}])assert.deepEqual(await f.emit('preUpdateItem',item,changes,{},f.user.id),[undefined]);
  throw Error('end probe');
 };f.make();await assert.rejects(f.run(),/end probe/);assert.equal(prepared,0);assert.equal(f.item.system.frequency.value,1);
});
