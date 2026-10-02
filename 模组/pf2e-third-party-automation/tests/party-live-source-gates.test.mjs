import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createPartyAutomation,PARTY_SOURCES} from '../scripts/party-automation.mjs';
import {registerUsageEvents} from '../scripts/usage-events.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

const nativeSources={
 token:{path:'C:/Program Files/Foundry Virtual Tabletop/resources/app/client/documents/token.mjs',sha:'44b8f03c161d8f077991167d652924228fe2f88b2b47ef4354b98df1a83feba7'},
 actor:{path:'C:/Program Files/Foundry Virtual Tabletop/resources/app/client/documents/actor.mjs',sha:'e82580bf9cef39d934c972dee859a3b9ba7ab5f3ebdc7502319dfed1bc214bb3'},
};
for(const source of Object.values(nativeSources)){const bytes=readFileSync(source.path);assert.equal(createHash('sha256').update(bytes).digest('hex'),source.sha);source.text=bytes.toString('utf8');}
function nativeGetter(source,name,dependencies){
 const match=new RegExp(`  get ${name}\\(\\) \\{\\r?\\n([\\s\\S]*?)\\r?\\n  \\}`).exec(source.text);assert.ok(match,`native ${name} getter`);
 return new Function(...Object.keys(dependencies),`return function(){${match[1]}\n}`)(...Object.values(dependencies));
}
function nativeBindings(f,{syntheticSource=false,syntheticTarget=false}={}){
 class NativeToken{}
 class NativeActor{}
 const foundry={documents:{TokenDocument:NativeToken}};
 for(const name of ['actor','baseActor','isLinked','isLazyDelta'])Object.defineProperty(NativeToken.prototype,name,{get:nativeGetter(nativeSources.token,name,{game:f.game,TokenDocument:NativeToken})});
 for(const name of ['isToken','token'])Object.defineProperty(NativeActor.prototype,name,{get:nativeGetter(nativeSources.actor,name,{foundry})});
 f.baseActors=new Map();
 f.bindNative=(actor,token,synthetic)=>{
  Object.setPrototypeOf(actor,NativeActor.prototype);Object.setPrototypeOf(token,NativeToken.prototype);delete token.actor;
  token.actorId=actor.id;token.actorLink=!synthetic;
  if(synthetic){
   const base=Object.assign(new NativeActor(),{id:actor.id,uuid:actor.uuid,type:actor.type,parent:null});f.game.actors.set(base.id,base);f.docs.set(base.uuid,base);f.baseActors.set(actor,base);
   actor.parent=token;actor.uuid=`${token.uuid}.Actor.${actor.id}`;token.delta={syntheticActor:actor};
  }else actor.parent=null;
  assert.equal(token.actor,actor);assert.equal(token.baseActor,f.game.actors.get(actor.id));assert.equal(actor.isToken,synthetic);assert.equal(actor.token,synthetic?token:null);
 };
 for(const binding of [[f.actor,f.source,syntheticSource],[f.recipient,f.target,syntheticTarget]])f.bindNative(...binding);
 f.item.uuid=`${f.actor.uuid}.Item.${f.item.id}`;f.message.flags.pf2e.origin.actor=f.actor.uuid;f.message.flags.pf2e.origin.uuid=f.item.uuid;f.marked.flags[ID].party.sourceActor=f.actor.uuid;f.raiseShield();
 return f;
}

function patch(doc,changes){for(const[path,value]of Object.entries(changes)){const keys=path.split('.');let at=doc;for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=structuredClone(value);}}
function fixture(t,{kind='clue',change,paid=false,nativeError=false}={}){
 const gm={id:'gm',isGM:true,active:true},nextGM={id:'next',isGM:true,active:true},user={id:'owner',active:true},users=Object.assign(new Map([gm,nextGM,user].map(u=>[u.id,u])),{activeGM:gm});
 const game={user:gm,users,actors:new Map(),messages:new Map(),scenes:new Map(),time:{worldTime:100}},docs=new Map(),counts={effects:0,updates:0,deletes:0,frequency:0,cooldown:0,paid:0};let owned=true,selectedRecipient;
 const makeActor=id=>({id,uuid:`Actor.${id}`,type:'character',name:id,flags:{},items:new Map(),level:5,attributes:{shield:{raised:true,broken:false,destroyed:false,itemId:'shield'}},testUserPermission:()=>owned,isAllyOf:()=>true,getRollOptions:()=>['blood-magic:imperial'],async update(changes){patch(this,changes);counts.cooldown++;await change?.('cooldown',f);return this;},async createEmbeddedDocuments(_kind,rows){counts.effects++;const result=rows.map((row,index)=>({...structuredClone(row),id:`effect-${index}`,actor:this,async update(changes){counts.updates++;patch(this,changes);await change?.('effect-update',f);return this;}}));for(const row of result)this.items.set(row.id,row);await change?.('effect',f);if(nativeError)throw Error('native effect response lost');return result;},async deleteEmbeddedDocuments(_kind,ids){counts.deletes++;for(const id of ids)this.items.delete(id);await change?.('effect-delete',f);return [];}});
 const actor=makeActor('source'),recipient=makeActor('recipient'),replacement=makeActor('replacement'),scene={id:'scene',tokens:new Map()};for(const doc of [actor,recipient,replacement]){game.actors.set(doc.id,doc);docs.set(doc.uuid,doc);}
 const priorCanvas=globalThis.canvas;globalThis.canvas={scene};t.after(()=>globalThis.canvas=priorCanvas);
 const source={id:'source',uuid:'Scene.scene.Token.source',documentName:'Token',actor,parent:scene,object:{}},target={id:'target',uuid:'Scene.scene.Token.target',documentName:'Token',actor:recipient,parent:scene,object:{}};for(const doc of [source,target]){scene.tokens.set(doc.id,doc);docs.set(doc.uuid,doc);}game.scenes.set(scene.id,scene);actor.getActiveTokens=()=>[source];recipient.getActiveTokens=()=>[target];
 const item={id:kind,uuid:`Actor.source.Item.${kind}`,type:kind==='imperial'?'spell':kind==='clue'?'action':'feat',actor,parent:actor,sourceId:kind==='imperial'?'Compendium.pf2e.spells-srd.Item.original':PARTY_SOURCES[kind],system:{frequency:{max:1,value:paid?0:1,per:'PT10M'},traits:{otherTags:['blood-magic-spell']}},getOriginData:()=>({rollOptions:[]}),async update(changes){counts.frequency++;patch(this,changes);await change?.('frequency',f);return this;}};actor.items.set(item.id,item);docs.set(item.uuid,item);
 const shield={id:'shield',type:'shield',actor,parent:actor,baseType:'shield'};actor.items.set(shield.id,shield);actor.items.set('imperial-feature',{id:'imperial-feature',type:'feat',actor,parent:actor,sourceId:PARTY_SOURCES.imperial});
 const receipt=paid?{id:'original-payment',itemUuid:item.uuid,userId:user.id,before:1,after:0}:null;
 const message={id:'use',author:user,speaker:{actor:actor.id,scene:scene.id,token:source.id},flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid,type:item.type}},[ID]:{usageInput:{actualUse:true,targetUuids:[target.uuid],frequencyReceiptId:receipt?.id??null},...paid?{usage:{status:'pending',frequencyReceipt:receipt}}:{}}},async update(changes){patch(this,changes);return this;}};game.messages.set(message.id,message);
 const marked={id:'marked',type:'effect',actor:recipient,flags:{[ID]:{party:{kind:'anoint',sourceActor:actor.uuid,sourceToken:source.uuid,targetToken:target.uuid,expiresAt:160}}},isExpired:false};if(kind==='imperial')recipient.items.set(marked.id,marked);
 const handlers=new Map(),Hooks={on(name,fn){const list=handlers.get(name)??[];list.push(fn);handlers.set(name,list);return fn;},off(name,fn){handlers.set(name,(handlers.get(name)??[]).filter(v=>v!==fn));}};
 const provider=createPartyAutomation({game,fromUuid:async uuid=>{await change?.(`resolve:${uuid}`,f);if(docs.has(uuid))return docs.get(uuid);await change?.('template',f);return {toObject:()=>({type:'effect',name:'Native effect',system:{rules:[],duration:{}},flags:{}})};},choose:async({title,choices})=>{await change?.(title.includes('受益者')?'recipient-choice':'defense-choice',f);return title.includes('受益者')?selectedRecipient??recipient.uuid:choices[0].value;},castEvents:{addMatcher(){},async ensurePaid(){counts.paid++;await change?.('paid',f);}}});t.after(provider.register({Hooks}));
 const f={game,gm,nextGM,user,actor,recipient,replacement,source,target,scene,item,message,receipt,marked,counts,provider,docs,Hooks,handlers,revokeOwner(){owned=false},selectRecipient(uuid){selectedRecipient=uuid},raiseShield(){for(const fn of handlers.get('createChatMessage')??[])fn({actor,item:{slug:'raise-a-shield'},flags:{}});},run:()=>provider.executeUsage({actor,item,message,user,action:`party:${kind}`,frequencyReceipt:receipt})};
 f.raiseShield();
 return f;
}

test('Clue In retains the original effect, frequency and cooldown path',async t=>{
 const f=fixture(t);await f.run();assert.deepEqual(f.counts,{effects:1,updates:0,deletes:0,frequency:1,cooldown:1,paid:0});assert.equal(f.item.system.frequency.value,0);assert.equal(f.actor.flags[ID].party.clueUntil,700);
});
for(const name of ['GM handoff','target relink'])test(`Clue In stops ${name} during the original template await`,async t=>{
 const f=fixture(t,{change(phase,f){if(phase==='template'){if(name==='GM handoff')f.game.users.activeGM=f.nextGM;else f.target.actor=f.replacement;}}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);assert.equal(f.replacement.items.size,0);
});
const changes={
 gm:f=>f.game.users.activeGM=f.nextGM,
 'same-ID GM':f=>{const next={...f.gm};f.game.users.set(next.id,next);f.game.users.activeGM=next;},
 'current GM user':f=>f.game.user={...f.gm},
 'GM registry':f=>f.game.users.set(f.gm.id,{...f.gm}),
 owner:f=>f.revokeOwner(),
 user:f=>f.game.users.set(f.user.id,{...f.user}),
 actor:f=>f.game.actors.set(f.actor.id,{...f.actor}),
 item:f=>f.actor.items.set(f.item.id,{...f.item}),
 card:f=>f.game.messages.set(f.message.id,{...f.message}),
 'card origin':f=>f.message.flags.pf2e.origin.actor='Actor.other',
 'source relink':f=>f.source.actor=f.replacement,
 'target relink':f=>f.target.actor=f.replacement,
 scene:f=>f.game.scenes.set(f.scene.id,{...f.scene}),
};
for(const kind of ['clue','anoint','guardian','imperial']){
 test(`${kind} keeps its valid native effect path`,async t=>{const f=fixture(t,{kind});await f.run();assert.equal(f.counts.effects,1);assert.equal(f.counts.paid,kind==='imperial'?1:0);});
 test(`${kind} retains valid synthetic source and recipient bindings`,async t=>{
  const f=nativeBindings(fixture(t,{kind}),{syntheticSource:true,syntheticTarget:true});await f.run();assert.equal(f.counts.effects,1);
 });
 for(const [name,mutate]of Object.entries(changes))test(`${kind} stops ${name} at its native preparation await`,async t=>{
  const f=fixture(t,{kind,change(phase,f){if(phase===(kind==='guardian'?`resolve:${f.target.uuid}`:'template'))mutate(f);}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);
 });
 test(`${kind} stops GM handoff after a completed effect without further writes`,async t=>{
  const f=fixture(t,{kind,change(phase,f){if(phase==='effect')f.game.users.activeGM=f.nextGM;}});await assert.rejects(f.run());assert.equal(f.counts.effects,1);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);
 });
}
for(const kind of ['clue','anoint','guardian'])test(`${kind} retains the original target actor across its first lookup`,async t=>{
 const f=fixture(t,{kind,change(phase,f){if(phase===`resolve:${f.target.uuid}`)f.target.actor=f.replacement;}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);
});
for(const field of ['suppressed','isSuppressed','system.suppressed'])test(`Clue In stops a source ${field} change before writing`,async t=>{
 const f=fixture(t,{change(phase,f){if(phase==='template')patch(f.item,{[field]:true});}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);
});
test('Guardian rechecks its original usable shield after target lookup',async t=>{
 const f=fixture(t,{kind:'guardian',change(phase,f){if(phase===`resolve:${f.source.uuid}`)f.actor.attributes.shield.broken=true;}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);
});
for(const phase of ['recipient-choice','defense-choice','paid'])test(`Imperial stops the old GM after ${phase} and retains native payment`,async t=>{
 const f=fixture(t,{kind:'imperial',change(at,f){if(at===phase)f.game.users.activeGM=f.nextGM;}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.paid,phase==='paid'?1:0);
});
test('Imperial rechecks the exact original anoint mark after native payment',async t=>{
 const f=fixture(t,{kind:'imperial',change(phase,f){if(phase==='paid')f.recipient.items.set(f.marked.id,{...f.marked});}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.paid,1);
});
for(const field of ['sourceActor','kind','expiresAt'])test(`Imperial rejects an original mark ${field} change during its first owner choice`,async t=>{
 const f=fixture(t,{kind:'imperial',change(phase,f){if(phase==='recipient-choice')f.marked.flags[ID].party[field]=field==='expiresAt'?170:'changed';}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.paid,0);
});
test('Clue In treats an empty native effect result as unknown without charging or refunding',async t=>{
 const f=fixture(t,{paid:true});const create=f.recipient.createEmbeddedDocuments;f.recipient.createEmbeddedDocuments=async function(...args){await create.apply(this,args);return [];};await assert.rejects(f.run());assert.equal(f.counts.effects,1);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);assert.equal(f.item.system.frequency.value,0);
});
for(const kind of ['clue','anoint','guardian','imperial'])test(`${kind} stops after an existing native effect update before duplicate cleanup`,async t=>{
 const f=fixture(t,{kind,change(phase,f){if(phase==='effect-update')f.game.users.activeGM=f.nextGM;}});
 for(const id of ['first','duplicate'])f.recipient.items.set(id,{id,type:'effect',actor:f.recipient,flags:{[ID]:{nativeEffectKey:`${kind}:${f.actor.uuid}`}},async update(changes){f.counts.updates++;patch(this,changes);await changesFn();return this;}});
 const changesFn=async()=>{f.game.users.activeGM=f.nextGM;};await assert.rejects(f.run());assert.equal(f.counts.updates,1);assert.equal(f.counts.deletes,0);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);
});
test('paid Clue In refunds only its original undelivered known refusal',async t=>{
 const f=fixture(t,{paid:true});f.message.flags[ID].usageInput.targetUuids=[];await assert.rejects(f.run(),/其他生物/);assert.equal(f.counts.effects,0);assert.equal(f.counts.frequency,1);assert.equal(f.item.system.frequency.value,1);
});
for(const name of ['gm','same-ID GM','item','card','source relink'])test(`paid Clue In cannot refund through ${name} loss`,async t=>{
 const f=fixture(t,{paid:true,change(phase,f){if(phase==='template')changes[name](f);}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.frequency,0);assert.equal(f.item.system.frequency.value,0);
});
test('an entered native Clue effect with a lost response keeps its original payment occupied',async t=>{
 const f=fixture(t,{paid:true,nativeError:true});await assert.rejects(f.run(),/response lost/);assert.equal(f.counts.effects,1);assert.equal(f.counts.frequency,0);assert.equal(f.item.system.frequency.value,0);
});
test('paid Clue In refuses a mismatched persisted original receipt without refunding',async t=>{
 const f=fixture(t,{paid:true});f.message.flags[ID].usage.frequencyReceipt={...f.receipt,id:'another-payment'};await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.frequency,0);
});
test('the actual usage claim prevents replay after an unknown native Clue effect',async t=>{
 const f=fixture(t,{nativeError:true}),errors=[];
 t.after(registerUsageEvents({game:f.game,Hooks:f.Hooks,fromUuid:async uuid=>f.docs.get(uuid),libWrapper:null,canvas:null,resolveAction:f.provider.resolveAction,requiresActualUse:f.provider.requiresActualUse,tracksFrequency:()=>false,executeUsage:context=>f.provider.executeUsage(context),onError:error=>errors.push(error)}));
 const dispatch=async()=>{for(const fn of f.handlers.get('createChatMessage')??[])await fn(f.message,{},f.user.id);};await dispatch();await dispatch();assert.equal(errors.length,1);assert.equal(f.message.flags[ID].usage.status,'error');assert.equal(f.counts.effects,1);
});
for(const kind of ['clue','anoint','guardian','imperial'])for(const changed of [false,true])test(`${kind} ${changed?'refuses replaced':'keeps original'} duplicate effect documents after its update`,async t=>{
 const f=fixture(t,{kind}),key=`${kind}:${f.actor.uuid}`;
 const duplicate={id:'duplicate',type:'effect',flags:{[ID]:{nativeEffectKey:key}}},first={id:'first',type:'effect',flags:{[ID]:{nativeEffectKey:key}},async update(changes){f.counts.updates++;patch(this,changes);if(changed)f.recipient.items.set(duplicate.id,{...duplicate});return this;}};f.recipient.items.set(first.id,first);f.recipient.items.set(duplicate.id,duplicate);
 if(changed)await assert.rejects(f.run());else await f.run();assert.equal(f.counts.updates,1);assert.equal(f.counts.deletes,changed?0:1);assert.equal(f.counts.effects,0);
});
for(const kind of ['anoint','guardian'])test(`${kind} stops after its original old-effect deletion when the GM changes`,async t=>{
 const f=fixture(t,{kind,change(phase,f){if(phase==='effect-delete')f.game.users.activeGM=f.nextGM;}});f.replacement.items.set('old',{id:'old',type:'effect',flags:{[ID]:{party:{kind,sourceActor:f.actor.uuid}}}});await assert.rejects(f.run());assert.equal(f.counts.deletes,1);assert.equal(f.counts.effects,0);
});
for(const response of ['undefined','detached'])test(`paid Clue In keeps an unknown ${response} native effect receipt occupied`,async t=>{
 const f=fixture(t,{paid:true}),create=f.recipient.createEmbeddedDocuments;f.recipient.createEmbeddedDocuments=async function(...args){const result=await create.apply(this,args);return response==='undefined'?undefined:[{...result[0]}];};await assert.rejects(f.run());assert.equal(f.counts.effects,1);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);assert.equal(f.item.system.frequency.value,0);
});
for(const changed of ['frequency','cooldown'])test(`Clue In refuses a changed original ${changed} after template lookup`,async t=>{
 const f=fixture(t,{change(phase,f){if(phase==='template'){if(changed==='frequency')f.item.system.frequency.value=0;else f.actor.flags[ID]={party:{clueUntil:1000}};}}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);
});
test('Clue In stops the old GM after its native frequency write before cooldown',async t=>{
 const f=fixture(t,{change(phase,f){if(phase==='frequency')f.game.users.activeGM=f.nextGM;}});await assert.rejects(f.run());assert.equal(f.counts.effects,1);assert.equal(f.counts.frequency,1);assert.equal(f.counts.cooldown,0);
});
test('Clue In refuses a changed original paid receipt after its template lookup',async t=>{
 const f=fixture(t,{paid:true,change(phase,f){if(phase==='template')f.message.flags[ID].usage.frequencyReceipt.id='changed';}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.frequency,0);
});
test('Imperial keeps its original self recipient and native payment without an anoint mark',async t=>{
 const f=fixture(t,{kind:'imperial'});f.selectRecipient(f.actor.uuid);f.recipient.items.delete(f.marked.id);await f.run();assert.equal(f.counts.effects,1);assert.equal(f.counts.paid,1);assert.equal(f.actor.items.get('effect-0').flags[ID].party.kind,'imperial');
});
for(const unknown of [false,true])test(`the actual native paid Clue claim ${unknown?'keeps an unknown effect occupied':'delivers once without a second deduction'}`,async t=>{
 const f=fixture(t,{nativeError:unknown}),errors=[],receipt={id:'native-payment',itemUuid:f.item.uuid,userId:f.user.id,before:1,after:0,createdAt:100};
 t.after(registerUsageEvents({game:f.game,Hooks:f.Hooks,fromUuid:async uuid=>f.docs.get(uuid),libWrapper:null,canvas:null,resolveAction:f.provider.resolveAction,requiresActualUse:f.provider.requiresActualUse,tracksFrequency:item=>item===f.item,executeUsage:context=>f.provider.executeUsage(context),onError:error=>errors.push(error)}));
 f.item.system.frequency.value=0;for(const fn of f.handlers.get('updateItem')??[])await fn(f.item,{'system.frequency.value':0},{[ID]:{frequencyReceipt:receipt}},f.user.id);f.message.flags[ID].usageInput.frequencyReceiptId=receipt.id;
 const dispatch=async()=>{for(const fn of f.handlers.get('createChatMessage')??[])await fn(f.message,{},f.user.id);};await dispatch();await dispatch();assert.equal(f.counts.effects,1);assert.equal(f.counts.frequency,0);assert.equal(f.item.system.frequency.value,0);assert.equal(f.message.flags[ID].usage.frequencyReceipt.id,receipt.id);assert.equal(f.message.flags[ID].usage.status,unknown?'error':'done');assert.equal(errors.length,unknown?1:0);assert.equal(f.counts.cooldown,unknown?0:1);
});

for(const kind of ['clue','anoint','guardian','imperial']){
 test(`${kind} retains actual native linked world Token getters`,async t=>{
  const f=nativeBindings(fixture(t,{kind}));await f.run();assert.equal(f.counts.effects,1);
 });
 for(const synthetic of ['source','target'])test(`${kind} retains mixed actual native ${synthetic} synthetic and linked Token bindings`,async t=>{
  const f=nativeBindings(fixture(t,{kind}),{syntheticSource:synthetic==='source',syntheticTarget:synthetic==='target'});await f.run();assert.equal(f.counts.effects,1);
 });
 for(const side of ['source','target'])test(`${kind} refuses an actual native synthetic ${side} without an original actorId`,async t=>{
  const f=nativeBindings(fixture(t,{kind}),{syntheticSource:true,syntheticTarget:true});delete f[side].actorId;
  await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);
 });
 for(const side of ['source','target'])for(const boundary of ['base registry','actorId','actor parent','actorLink','Token documentName'])test(`${kind} rejects native synthetic ${side} ${boundary} change during preparation`,async t=>{
  const f=nativeBindings(fixture(t,{kind,change(phase,f){
   if(phase!==(kind==='guardian'?`resolve:${f.target.uuid}`:'template'))return;
   const token=f[side],actor=side==='source'?f.actor:f.recipient;
   if(boundary==='base registry')f.game.actors.set(token.actorId,{...token.baseActor});
   else if(boundary==='actorId')token.actorId=f.replacement.id;
   else if(boundary==='actor parent')actor.parent=f.source===token?f.target:f.source;
   else if(boundary==='actorLink')token.actorLink=true;
   else token.documentName='Actor';
  }}),{syntheticSource:true,syntheticTarget:true});
  await assert.rejects(f.run());assert.equal(f.counts.effects,0);assert.equal(f.counts.deletes,0);assert.equal(f.counts.frequency,0);assert.equal(f.counts.cooldown,0);
 });
}
for(const kind of ['anoint','guardian'])for(const boundary of ['unchanged','base registry','actor parent'])test(`${kind} ${boundary==='unchanged'?'removes':'rejects'} its captured native synthetic old-effect actor ${boundary}`,async t=>{
 const f=nativeBindings(fixture(t,{kind,change(phase,f){
  if(phase!==(kind==='guardian'?`resolve:${f.target.uuid}`:'template')||boundary==='unchanged')return;
  if(boundary==='base registry')f.game.actors.set(f.oldToken.actorId,{...f.oldToken.baseActor});
  else f.replacement.parent=f.source;
 }}));
 f.oldToken={id:'old',uuid:'Scene.scene.Token.old',documentName:'Token',parent:f.scene,object:{}};f.scene.tokens.set(f.oldToken.id,f.oldToken);f.docs.set(f.oldToken.uuid,f.oldToken);f.bindNative(f.replacement,f.oldToken,true);
 f.replacement.items.set('old',{id:'old',type:'effect',flags:{[ID]:{party:{kind,sourceActor:f.actor.uuid}}}});
 if(boundary==='unchanged')await f.run();else await assert.rejects(f.run());assert.equal(f.counts.deletes,boundary==='unchanged'?1:0);assert.equal(f.counts.effects,boundary==='unchanged'?1:0);
});
