import test from 'node:test';
import assert from 'node:assert/strict';
import {createMetapowerProvider} from '../scripts/metapower/provider.mjs';
import {METAPOWER_SOURCES} from '../scripts/metapower/rules.mjs';
import {MODULE_ID as ID,ledgerState} from '../scripts/metapower/lifecycle.mjs';

async function fixture(t,options={}){
 const previousConfig=globalThis.CONFIG,previousFoundry=globalThis.foundry;
 globalThis.CONFIG={Actor:{sheetClasses:{character:{}}},Dice:{rolls:[class DamageRoll{}]}};
 t.after(()=>{if(previousConfig===undefined)delete globalThis.CONFIG;else globalThis.CONFIG=previousConfig;if(previousFoundry===undefined)delete globalThis.foundry;else globalThis.foundry=previousFoundry});
 const clones=[],nativeClone=globalThis.structuredClone;
 t.mock.method(globalThis,'structuredClone',value=>{clones.push(value);return nativeClone(value)});
 const hooks=new Map(),wrappers=new Map(),endpoints=new Map(),requests=[],errors=[],buttons=[],counts={reads:0,writes:0,nativeCalls:0};
 const gm={id:'gm',isGM:true,active:true,targets:new Set()},owner={id:'owner',active:true},users=new Map([[gm.id,gm],[owner.id,owner]]);users.activeGM=gm;
 const actor={id:'a',uuid:'Actor.a',type:'character',level:3,isOwner:true,flags:{},items:new Map(),allow:true,testUserPermission(){return this.allow},async update(changes){counts.writes++;this.flags[ID]={metapower:nativeClone(changes[`flags.${ID}.metapower`])};return this}};
 const widen={id:'w',uuid:'Actor.a.Item.w',sourceId:METAPOWER_SOURCES.widen,actor,system:{actionType:{value:'action'},actions:{value:1}}};actor.items.set(widen.id,widen);
 const actions=new Map();actions.balance=function(input){counts.nativeCalls++;return input};
 const game={user:gm,users,actors:new Map([[actor.id,actor]]),scenes:new Map(),messages:new Map(),combats:new Map(),time:{worldTime:0},modules:new Map(),pf2e:{actions}};
 const documents=new Map([[actor.uuid,actor]]);
 let route=false;
 const socket={register(name,fn){endpoints.set(name,fn)},async executeAsUser(name,gmId,payload){requests.push({name,gmId,payload});if(route)return endpoints.get(name).call({socketdata:{userId:game.user.id}},payload);return {ok:true,value:{...payload,status:'started',nativeStartAuthorized:true}}}};
 globalThis.foundry={applications:{api:{DialogV2:{wait:async()=> 'retry',confirm:async()=>true}}}};
 const provider=createMetapowerProvider({game,fromUuid:async uuid=>documents.get(uuid),onError:error=>errors.push(error),...options});
 provider.register({Hooks:{on(name,fn){const callbacks=hooks.get(name)??[];callbacks.push(fn);hooks.set(name,callbacks);return callbacks.length},off(){}},libWrapper:{register(_id,path,fn){wrappers.set(path,fn)}},socket});
 await Promise.resolve();await Promise.resolve();
 const root={querySelectorAll:()=>[],querySelector(selector){if(selector.startsWith('.metapower-'))return buttons.find(button=>button.className===selector.slice(1))??null;return this},append(button){buttons.push(button)},addEventListener(){},ownerDocument:{createElement(){const callbacks=new Map();const button={addEventListener(name,fn){callbacks.set(name,fn)},remove(){const index=buttons.indexOf(this);if(index>=0)buttons.splice(index,1)},click(){return callbacks.get('click')({preventDefault(){},stopPropagation(){}})}};return button}}};
 function populate(size,extra={}){
  const receipts={};for(let index=0;index<size;index++){const receipt={nonce:`old${index}`,status:'committed',snapshot:{powerId:'ordinary'},delivery:{status:'done'}};Object.defineProperty(receipts,receipt.nonce,{enumerable:true,configurable:true,get(){counts.reads++;return receipt}})}
  const state={version:1,sequence:size,armed:null,pending:null,receipts,...extra};actor.flags[ID]={metapower:state};return state;
 }
 function reset(){clones.length=0;counts.reads=counts.writes=counts.nativeCalls=0;requests.length=0;errors.length=0}
 reset();
 return {actor,game,gm,owner,documents,provider,buttons,counts,clones,errors,requests,wrappers,populate,reset,setRoute(value){route=value},render(){for(const callback of hooks.get('renderActorSheetPF2e'))callback({actor,isEditable:true},root)},button(name){return buttons.find(button=>button.className===name)}};
}

test('registered sheet render reads receipts once without cloning the ledger',async t=>{
 const f=await fixture(t),state=f.populate(1024,{pending:'pending'});state.receipts.first={nonce:'first',status:'committed',delivery:{status:'pending'}};state.receipts.second={nonce:'second',status:'committed',delivery:{status:'pending'}};
 f.render();assert.ok(f.button('metapower-reconcile'));assert.ok(f.button('metapower-delivery-recovery'));
 assert.equal(f.clones.filter(value=>value===state).length,0);assert.equal(f.counts.reads,1024);assert.equal(f.counts.writes,0);assert.equal(f.requests.length,0);
 await f.button('metapower-delivery-recovery').click();assert.equal(f.requests[0].payload.nonce,'first');
});

for(const malformed of [false,true])test(`recovery dialog captures its render nonce before live receipt changes (${malformed?'JSON':'string'})`,async t=>{
 const f=await fixture(t),nonce=malformed?{nested:{value:'old'}}:'old',state=f.populate(8),receipt={nonce,status:'committed',delivery:{status:'pending'}};state.receipts.old=receipt;
 let release;globalThis.foundry.applications.api.DialogV2.wait=()=>new Promise(resolve=>{release=resolve});
 f.render();const click=f.button('metapower-delivery-recovery').click();
 if(malformed)nonce.nested.value='changed';else receipt.nonce='changed';release('retry');await click;
 assert.deepEqual(f.requests[0].payload.nonce,malformed?{nested:{value:'old'}}:'old');
 if(malformed)assert.notEqual(f.requests[0].payload.nonce,nonce);
});

test('malformed pending JSON remains an isolated render snapshot and missing recovery nonce still shows a control',async t=>{
 const f=await fixture(t),pending={nested:{value:'old'}},state=f.populate(8,{pending});state.receipts.missing={status:'committed',delivery:{status:'pending'}};
 let release;globalThis.foundry.applications.api.DialogV2.confirm=()=>new Promise(resolve=>{release=resolve});
 f.render();assert.ok(f.button('metapower-delivery-recovery'));
 const click=f.button('metapower-reconcile').click();pending.nested.value='changed';release(true);await click;
 assert.deepEqual(f.requests[0].payload.nonce,{nested:{value:'old'}});assert.notEqual(f.requests[0].payload.nonce,pending);
});

for(const mode of ['idle','armed','pending'])test(`registered legacy alias checks ${mode} flags without cloning any ledger`,async t=>{
 const f=await fixture(t),state=f.populate(1024,{armed:mode==='armed'?{nonce:'armed'}:null,pending:mode==='pending'?'pending':null}),input={actors:[f.actor],callback(){}};
 if(mode==='idle')assert.equal(f.game.pf2e.actions.balance(input),input);else assert.throws(()=>f.game.pf2e.actions.balance(input),/旧式|原生动作/);
 assert.equal(f.clones.filter(value=>value===state).length,0);assert.equal(f.counts.reads,0);assert.equal(f.counts.nativeCalls,mode==='idle'?1:0);assert.equal(f.counts.writes,0);
});

test('sheet pending control keeps ledger authority after OWNER changes during its confirmation',async t=>{
 const f=await fixture(t),state=f.populate(0,{pending:'old'});state.receipts.old={nonce:'old',status:'started'};f.setRoute(true);
 let release;globalThis.foundry.applications.api.DialogV2.confirm=()=>new Promise(resolve=>{release=resolve});
 f.render();const click=f.button('metapower-reconcile').click();f.actor.allow=false;release(true);await click;
 assert.equal(f.counts.writes,0);assert.equal(state.pending,'old');assert.match(f.errors[0].message,/权限/);
});

for(const malformed of [false,true])test(`native selection freezes only armed kind across choice await (${malformed?'JSON':'string'})`,async t=>{
 let release,choicesSeen,kindSeen;
 const f=await fixture(t,{selectChoice:question=>{choicesSeen=question.choices;return new Promise(resolve=>{release=resolve})},beforeChannel:async({kind,selection})=>{kindSeen=kind;return selection}}),kind=malformed?{nested:{value:'old'}}:'siphoning',state=f.populate(1024,{armed:{kind}});
 const item={id:'power',uuid:'Actor.a.Item.power',actor:f.actor,type:'action',name:'Electric Surge',sourceId:'Compendium.battlezoo-eldamon-pf2e.powers.Item.hQOa1yaP9C6wajNn',system:{traits:{value:['electricity']}}};f.actor.items.set(item.id,item);
 const observed=f.provider.observe({actor:f.actor,item},async()=> 'native');assert.ok(choicesSeen);
 if(malformed)kind.nested.value='changed';else state.armed.kind='widen';release(JSON.stringify({discharge:false}));assert.equal(await observed,'native');
 assert.deepEqual(kindSeen,malformed?{nested:{value:'old'}}:'siphoning');if(malformed)assert.notEqual(kindSeen,kind);
 assert.equal(f.clones.filter(value=>value===state).length,0);assert.equal(f.counts.reads,0);assert.equal(f.counts.writes,0);
});

test('ledgerState remains a full independent snapshot for mutations',()=>{
 const actor={flags:{[ID]:{metapower:{armed:{kind:'widen'},receipts:{n:{selection:{targetUuids:['old']}}}}}}},copy=ledgerState(actor);
 copy.armed.kind='siphoning';copy.receipts.n.selection.targetUuids[0]='new';assert.equal(actor.flags[ID].metapower.armed.kind,'widen');assert.deepEqual(actor.flags[ID].metapower.receipts.n.selection.targetUuids,['old']);
});

for(const mode of ['done','no-delivery','uncertain','missing'])test(`deliverCommitted reads only the exact ${mode} receipt and returns an isolated value`,async t=>{
 const f=await fixture(t),state=f.populate(1024),receipt={nonce:'selected',status:mode==='uncertain'?'uncertain':'committed',snapshot:{nested:{value:'old'}},selection:{targetUuids:['old']},delivery:{status:'done'}};
 if(mode==='no-delivery')delete receipt.delivery;if(mode!=='missing')state.receipts.selected=receipt;
 const result=await f.provider.deliverCommitted({actorUuid:f.actor.uuid,nonce:'selected'});
 if(mode==='missing')assert.equal(result,undefined);else{assert.deepEqual(result,receipt);assert.notEqual(result,receipt);result.snapshot.nested.value='changed';result.selection.targetUuids[0]='changed';assert.equal(receipt.snapshot.nested.value,'old');assert.deepEqual(receipt.selection.targetUuids,['old'])}
 assert.equal(f.clones.filter(value=>value===state).length,0);assert.equal(f.counts.reads,0);assert.equal(f.counts.writes,0);assert.equal(f.requests.length,0);assert.deepEqual(f.errors,[]);
});

test('failed delivery authorization keeps an isolated exact fallback receipt without scanning other records',async t=>{
 const f=await fixture(t),state=f.populate(1024),receipt={nonce:'selected',status:'committed',messageUuid:'ChatMessage.original',snapshot:{nested:{value:'old'}},delivery:{status:'pending',attempts:0}};state.receipts.selected=receipt;f.actor.allow=false;
 const result=await f.provider.deliverCommitted({actorUuid:f.actor.uuid,nonce:'selected'});assert.deepEqual(result,receipt);assert.notEqual(result,receipt);result.snapshot.nested.value='changed';assert.equal(receipt.snapshot.nested.value,'old');
 assert.equal(f.clones.filter(value=>value===state).length,0);assert.equal(f.counts.reads,0);assert.equal(f.counts.writes,0);assert.equal(f.errors.length,1);assert.match(f.errors[0].message,/权限/);
});

function damageFixture(f,state){
 const target={uuid:'Actor.target',items:new Map([['shocked',{sourceId:'Compendium.battlezoo-eldamon-pf2e.conditions.Item.1fZbuJEbVmE3J4XL'}]])},token={uuid:'Scene.s.Token.t',actor:target},itemUuid='Actor.a.Item.power';
 const receipt={nonce:'selected',status:'committed',messageUuid:'ChatMessage.c',snapshot:{powerId:'electric-shot',itemUuid},selection:{targetUuids:[token.uuid]},delivery:{status:'done'}},card={id:'c',uuid:'ChatMessage.c',flags:{[ID]:{metapowerUse:{nonce:receipt.nonce,actorUuid:f.actor.uuid}},pf2e:{origin:{uuid:itemUuid}}}};
 const proof={actorUuid:f.actor.uuid,cardId:card.id,nonce:receipt.nonce,targetActorUuid:target.uuid,targetTokenUuid:token.uuid};state.receipts.selected=receipt;f.game.messages.set(card.id,card);f.documents.set(token.uuid,token);
 return {target,token,receipt,proof,damage:{options:{[ID]:{metapowerShotFailure:proof}}},data:{flags:{pf2e:{origin:{uuid:itemUuid},context:{options:[`${ID}:metapower:c:selected`,`${ID}:electric-shot-failure-half`]}}}}};
}

test('beforeDamage freezes the selected target array before Token await and ignores unrelated receipts',async t=>{
 let f,requestTarget,releaseTarget;const requested=new Promise(resolve=>{requestTarget=resolve}),waiting=new Promise(resolve=>{releaseTarget=resolve});
 f=await fixture(t,{fromUuid:async uuid=>{if(uuid==='Scene.s.Token.t'){requestTarget();return waiting}return f.documents.get(uuid)}});
 const state=f.populate(1024),d=damageFixture(f,state),checking=f.provider.beforeDamage(d.target,{damage:d.damage,token:d.token});await requested;
 d.receipt.selection.targetUuids[0]='Scene.s.Token.changed';releaseTarget(d.token);assert.equal(await checking,null);
 assert.equal(f.clones.filter(value=>value===state).length,0);assert.equal(f.counts.reads,0);assert.equal(f.counts.writes,0);
 await assert.rejects(f.provider.beforeDamage(d.target,{damage:d.damage,token:d.token}),/目标/);
});

test('registered DamageRoll wrapper reads only original target and preserves one native half conversion',async t=>{
 const f=await fixture(t),state=f.populate(1024),d=damageFixture(f,state);f.game.user.targets.add({document:d.token,actor:d.target});
 let conversions=0,nativeCalls=0;const roll={_evaluated:true,options:{},alter(multiplier){conversions++;assert.equal(multiplier,.5);return {_evaluated:true,options:{},terms:[],_formula:'4',_total:4,_dice:[]}}};
 const wrapper=f.wrappers.get('CONFIG.Dice.rolls.0.prototype.toMessage'),native=async()=>{nativeCalls++;return 'original-card'};
 assert.equal(await wrapper.call(roll,native,d.data,{}),'original-card');assert.equal(await wrapper.call(roll,native,d.data,{}),'original-card');assert.equal(conversions,1);assert.equal(nativeCalls,2);assert.equal(roll.options[ID].metapowerShotFailure.nonce,'selected');
 assert.equal(f.clones.filter(value=>value===state).length,0);assert.equal(f.counts.reads,0);assert.equal(f.counts.writes,0);
 d.receipt.selection.targetUuids[0]='Scene.s.Token.changed';await assert.rejects(wrapper.call(roll,native,d.data,{}),/目标/);assert.equal(conversions,1);assert.equal(nativeCalls,2);
});
