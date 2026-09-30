import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync,readFileSync} from 'node:fs';
import vm from 'node:vm';
import {runDamagePipeline} from '../scripts/native-context.mjs';
let api={};try{api=await import('../scripts/eldamon-voltage.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const ID='pf2e-third-party-automation',HV='Compendium.battlezoo-eldamon-pf2e.powers.Item.9bElF2uVf5FCJtb9';
function fixture(){
 const gm={id:'gm',isGM:true,active:true},user={id:'owner',active:true},users=new Map([[gm.id,gm],[user.id,user]]);users.activeGM=gm;
 const game={user:gm,users,actors:new Map(),messages:new Map(),combat:{id:'fight',round:1,turn:0,started:true,turns:[]}};
 const docs=new Map(),set=(obj,key,value)=>{const parts=key.split('.');let cursor=obj;for(const part of parts.slice(0,-1))cursor=cursor[part]??={};cursor[parts.at(-1)]=structuredClone(value)};
 const actor={id:'a',uuid:'Actor.a',level:5,type:'character',flags:{},items:new Map(),getRollOptions:()=>['active-power-refresh:high-voltage','active-power-one:electric-surge','active-power-reactive:reactive-chain'],testUserPermission:u=>u===gm||u===user,async update(data){for(const [k,v]of Object.entries(data))set(this,k,v)}};docs.set(actor.uuid,actor);game.actors.set(actor.id,actor);
 const batches=[];actor.updateEmbeddedDocuments=async(type,updates)=>{assert.equal(type,'Item');batches.push(structuredClone(updates));const changed=[];for(const {_id,...patch}of updates){const item=actor.items.get(_id);assert.ok(item);await item.update(patch);changed.push(item)}return changed};
 const addItem=(id,source,slug,frequency)=>{const item={id,uuid:actor.uuid+'.Item.'+id,sourceId:source,actor,type:'feat',name:slug,flags:{},system:{slug,frequency,traits:{value:['electricity'],otherTags:['eldamon-power']}},async update(data){for(const[k,v]of Object.entries(data))set(this,k,v)}};actor.items.set(id,item);docs.set(item.uuid,item);return item};
 const item=addItem('h',HV,'high-voltage'),spent=addItem('s','Compendium.battlezoo-eldamon-pf2e.powers.Item.veFrnrxYjlqca13w','electric-surge',{max:1,per:'PT10M',value:0}),reaction=addItem('r','Compendium.battlezoo-eldamon-pf2e.powers.Item.fzV5Ly3a9nEsfcAJ','reactive-chain',{max:1,per:'PT10M',value:0});
 const scene={id:'scene',tokens:new Map()},token=(id,a)=>{const t={id,uuid:'Scene.scene.Token.'+id,parent:scene,actor:a,object:{distanceTo:()=>5}};scene.tokens.set(id,t);docs.set(t.uuid,t);return t};
 const origin=token('origin',actor),targetActor={id:'b',uuid:'Actor.b',type:'npc'},target=token('target',targetActor);docs.set(targetActor.uuid,targetActor);
 game.scenes=new Map([[scene.id,scene]]);game.actors.set(targetActor.id,targetActor);
 game.combat.turns=[{id:'ca',actor,token:origin},{id:'cb',actor:targetActor,token:target}];game.combat.combatant=game.combat.turns[0];
 const receipt={nonce:'channel',userId:user.id,actorUuid:actor.uuid,itemUuid:item.uuid,sourceUuid:HV,status:'committed',messageUuid:'ChatMessage.card',snapshot:null,turn:'fight:1:0:ca'};
 const message={id:'card',uuid:'ChatMessage.card',timestamp:100,speaker:{actor:'a',scene:'scene',token:'origin'},author:user,flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid}},[ID]:{metapowerUse:{nonce:'channel',actorUuid:actor.uuid,itemUuid:item.uuid}}}};
 actor.flags[ID]={metapower:{receipts:{channel:receipt}}};docs.set(message.uuid,message);game.messages.set(message.id,message);
 const payload={actorUuid:actor.uuid,nonce:'channel',messageUuid:message.uuid};
 const service=()=>api.createVoltageLedger({game,fromUuid:async uuid=>docs.get(uuid)});
 return{game,gm,user,actor,item,spent,reaction,addItem,origin,target,docs,message,receipt,payload,service,batches};
}
test('normal original High Voltage channel refreshes actual spent prepared powers immediately and only once',async()=>{
 assert.equal(typeof api.createVoltageLedger,'function');const f=fixture(),s=f.service();
 const unrelated=f.addItem('other','Compendium.pf2e.feats-srd.Item.fake','electric-surge',{max:2,per:'PT10M',value:0});
 const daily=f.addItem('daily','Compendium.battlezoo-eldamon-pf2e.powers.Item.daily','electric-surge',{max:1,per:'day',value:0});
 const unprepared=f.addItem('u','Compendium.battlezoo-eldamon-pf2e.powers.Item.unprepared','electric-shot',{max:1,per:'PT10M',value:0});
 const result=await s.channel(f.payload,f.user);assert.equal(result.status,'armed');assert.equal(f.spent.system.frequency.value,1);assert.equal(f.reaction.system.frequency.value,1);assert.equal(unrelated.system.frequency.value,0);assert.equal(daily.system.frequency.value,0);assert.equal(unprepared.system.frequency.value,0);
 assert.equal(f.batches.length,1);assert.deepEqual(f.batches[0].map(update=>update._id),[f.spent.id,f.reaction.id]);
 f.spent.system.frequency.value=0;await f.service().channel(f.payload,f.user);assert.equal(f.spent.system.frequency.value,0);assert.equal(f.actor.getRollOptions()[0],'active-power-refresh:high-voltage');
});

test('automatic outside-encounter Refresh uses the same idempotent resource writer and requires GM/source/no active encounter',async()=>{
 const f=fixture(),s=f.service(),p={actorUuid:f.actor.uuid,nonce:'end:fight'};
 assert.equal(typeof s.refreshOutsideEncounter,'function');
 f.addItem('feature',api.ELEMENTAL_POWERS_SOURCE,'elemental-powers');
 await assert.rejects(s.refreshOutsideEncounter(p,f.gm),/遭遇/i);
 f.game.combat=null;await assert.rejects(s.refreshOutsideEncounter(p,f.user),/主持人/i);
 await s.refreshOutsideEncounter(p,f.gm);assert.equal(f.spent.system.frequency.value,1);assert.equal(f.reaction.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.service().refreshOutsideEncounter(p,f.gm);assert.equal(f.spent.system.frequency.value,0);
 f.game.combats=new Map([['other',{started:true,combatants:[{actor:f.actor}]}]]);await assert.rejects(s.refreshOutsideEncounter({...p,nonce:'new'},f.gm),/遭遇/i);
 f.game.combats.clear();f.actor.items.delete('feature');await assert.rejects(s.refreshOutsideEncounter({...p,nonce:'new'},f.gm),/元素威能/i);
});
test('committed Refresh notifies lifecycle once and retries failed cleanup without refilling powers',async()=>{
 const f=fixture();let calls=0,fail=true;
 const service=()=>api.createVoltageLedger({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),onRefresh:async context=>{
  assert.equal(context.actor,f.actor);assert.equal(context.nonce,'channel');assert.equal(f.spent.system.frequency.value,calls===0?1:0);calls++;
  if(fail){fail=false;throw Error('cleanup interrupted');}
 }});
 await assert.rejects(service().channel(f.payload,f.user),/cleanup interrupted/);
 assert.equal(f.actor.flags[ID].voltage.refreshes.channel.status,'done');
 f.spent.system.frequency.value=0;await service().channel(f.payload,f.user);await service().channel(f.payload,f.user);
 assert.equal(calls,2);assert.equal(f.spent.system.frequency.value,0);assert.equal(f.actor.flags[ID].voltage.refreshes.channel.effectsDone,true);
});
test('Siphoning High Voltage never emits a Refresh lifecycle callback',async()=>{
 const f=fixture();let calls=0;f.receipt.snapshot={kind:'siphoning',siphon:{applies:true},level:5,itemUuid:f.item.uuid,actorUuid:f.actor.uuid,powerSourceUuid:HV};
 const s=api.createVoltageLedger({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),onRefresh:async()=>calls++});
 await s.channel(f.payload,f.user);assert.equal(calls,0);assert.equal(f.spent.system.frequency.value,0);
});
test('Siphoning snapshot suppresses this channel Refresh without affecting the next normal channel',async()=>{
 const f=fixture();f.receipt.snapshot={kind:'siphoning',siphon:{applies:true},suppressEffects:['refresh'],level:5,itemUuid:f.item.uuid,actorUuid:f.actor.uuid,powerSourceUuid:HV};
 const r=await f.service().channel(f.payload,f.user);assert.equal(r.refreshSuppressed,true);assert.equal(f.spent.system.frequency.value,0);
 f.receipt.snapshot.siphon.applies=false;assert.equal(f.actor.flags[ID].voltage.activations.channel.snapshot.siphon.applies,true);
 const next={...f.receipt,nonce:'next',snapshot:null,messageUuid:'ChatMessage.next'};f.actor.flags[ID].metapower.receipts.next=next;
 const message=structuredClone(f.message);message.id='next';message.uuid=next.messageUuid;message.flags[ID].metapowerUse.nonce='next';message.author=f.user;f.docs.set(message.uuid,message);f.game.messages.set(message.id,message);
 await f.service().channel({...f.payload,nonce:'next',messageUuid:message.uuid},f.user);assert.equal(f.spent.system.frequency.value,1);
});
test('forged original use, unprepared power, wrong owner and inactive GM cannot arm',async()=>{
 const f=fixture(),s=f.service();await assert.rejects(s.channel({...f.payload,messageUuid:'ChatMessage.other'},f.user),/原|绑定|卡/i);
 await assert.rejects(s.channel(f.payload,{id:'stranger'}),/拥有|权限/i);
 f.actor.getRollOptions=()=>[];await assert.rejects(s.channel(f.payload,f.user),/准备/i);
 f.game.user=f.user;await assert.rejects(s.channel(f.payload,f.user),/主持人/i);
});
test('one durable claim wins concurrent touches and remains consumed after zero or cancelled child roll',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 const p={...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true};
 const claims=await Promise.all([s.claim(p,f.user),s.claim(p,f.user)]);assert.equal(claims.filter(Boolean).length,1);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'claimed');
 await s.settle({...f.payload,status:'cancelled'},f.gm);assert.equal(await f.service().claim(p,f.user),null);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'cancelled');
});
test('touch requires explicit fact confirmation; caster touching another creature does not trigger itself',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 await assert.rejects(s.claim({...f.payload,targetUuid:f.target.uuid,kind:'touch'},f.user),/声明|确认/i);
 await assert.rejects(s.claim({...f.payload,targetUuid:f.origin.uuid,kind:'touch',confirmed:true},f.user),/其他|生物|来源/i);
 assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'armed');
});
test('next source turn expires lazily across reload, while other creature turn does not',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);f.game.combat.turn=1;await s.expire(f.payload,f.gm);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'armed');
 f.game.combat.round=2;f.game.combat.turn=0;assert.equal(await f.service().claim({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user),null);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'expired');
});
test('High Voltage follows its actor encounter instead of the GM viewed encounter',async()=>{
 const f=fixture(),bound=f.game.combat;
 const other={id:'other',started:true,round:7,turn:0,turns:[]};
 f.game.combats=new Map([[bound.id,bound],[other.id,other]]);f.game.combat=other;
 const s=f.service(),a=await s.channel(f.payload,f.user);assert.equal(a.expires.combatId,bound.id);assert.equal(a.status,'armed');
 await s.expire(f.payload,f.gm);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'armed');
 bound.round=2;bound.turn=0;await s.expire(f.payload,f.gm);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'expired');
});
test('deleted original source closes a window without damage',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);f.actor.items.delete(f.item.id);
 assert.equal(await s.claim({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user),null);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'expired');
});
test('attack eligibility reads real melee outcomes without requiring geometry facts',()=>{
 assert.equal(typeof api.voltageAttackEligibility,'function');const facts={outcome:'success',melee:true,adjacent:true,unarmed:false,metal:false};
 assert.equal(api.voltageAttackEligibility(facts),true);assert.equal(api.voltageAttackEligibility({...facts,adjacent:false,unarmed:true}),true);assert.equal(api.voltageAttackEligibility({...facts,adjacent:false,metal:true}),true);
 assert.equal(api.voltageAttackEligibility({...facts,adjacent:false,metal:undefined}),true);assert.equal(api.voltageAttackEligibility({...facts,outcome:'failure'}),false);assert.equal(api.voltageAttackEligibility({...facts,melee:false}),false);
});
test('native attack card proves hit, direction and token identities before claim',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 const weapon={actor:f.target.actor,isMelee:true,type:'weapon',system:{category:'martial',material:{type:null}}};
 const attack={isCheckRoll:true,rolls:[{_evaluated:true}],id:'attack',uuid:'ChatMessage.attack',timestamp:200,author:f.gm,speaker:{scene:'scene',token:'target',actor:'b'},item:weapon,flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:['item:melee']}}}};
 f.game.messages.set(attack.id,attack);f.docs.set(attack.uuid,attack);
 assert.ok(await s.claim({...f.payload,kind:'attack',attackUuid:attack.uuid},f.gm));assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:attack.uuid},f.gm),null);
});
test('copied and pre-channel attack messages cannot consume the window',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 const attack={isCheckRoll:true,rolls:[{_evaluated:true}],id:'attack',uuid:'ChatMessage.attack',timestamp:50,author:f.gm,speaker:{scene:'scene',token:'target',actor:'b'},item:{actor:f.target.actor,isMelee:true,type:'weapon',system:{}},flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:[]}}}};f.docs.set(attack.uuid,attack);f.game.messages.set(attack.id,attack);
 assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:attack.uuid},f.gm),null);attack.timestamp=200;attack.flags.pf2e.context.isReroll=true;assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:attack.uuid},f.gm),null);
});
export {fixture};

test('standalone two-action Refresh requires its own original committed activity and granted feature',async()=>{
 const f=fixture(),feature=f.addItem('feature','Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.naawsnBug9EOpzfN','elemental-powers');
 f.receipt.sourceUuid=feature.sourceId;f.receipt.itemUuid=feature.uuid;f.message.flags.pf2e.origin.uuid=feature.uuid;f.message.flags[ID].metapowerUse.itemUuid=feature.uuid;
 await assert.rejects(f.service().refreshActivity(f.payload,f.user),/活动/i);
 f.message.flags[ID].voltageRefreshActivity={actions:2};await f.service().refreshActivity(f.payload,f.user);assert.equal(f.spent.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.service().refreshActivity(f.payload,f.user);assert.equal(f.spent.system.frequency.value,0);assert.equal(f.actor.flags[ID].voltage.activeNonce,null);
});
test('interrupted native Refresh batch resumes only uncommitted items without refilling a subsequently used power',async()=>{
 const f=fixture(),native=f.actor.updateEmbeddedDocuments.bind(f.actor);let fail=true;f.actor.updateEmbeddedDocuments=async(type,updates)=>{if(fail){fail=false;await native(type,updates.slice(0,1));throw Error('network failure')}return native(type,updates)};
 await assert.rejects(f.service().channel(f.payload,f.user),/network/);assert.equal(f.spent.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.service().channel(f.payload,f.user);assert.equal(f.spent.system.frequency.value,0);assert.equal(f.reaction.system.frequency.value,1);
 assert.deepEqual(f.batches.map(updates=>updates.map(update=>update._id)),[[f.spent.id],[f.reaction.id]]);
});

let executorApi={};try{executorApi=await import('../scripts/eldamon-voltage-executor.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
function executorFixture(outcome='failure',{siphon=false,traits=[],selectChoice,damageDialog,damagePrivacy,saveDialog}={}){
 const f=fixture(),messages=[],saves=[],applications=[],damageWindows=[],evaluations=[],publications=[],hooks=new Map(),routes=new Map(),rollContexts=new WeakMap(),dialogs=[];let hookSequence=0;
 const targetUser={id:'target-owner',active:true};f.game.users.set(targetUser.id,targetUser);f.actor.getStatistic=()=>({dc:{value:22}});
 f.target.actor.testUserPermission=u=>u===f.gm||u===targetUser;f.target.actor.traits=new Set(traits);
 f.target.actor.getSelfRollOptions=()=>['self:level:5'];f.target.actor.getContextualClone=()=>({...f.target.actor,async applyDamage(params){applications.push(params);await provider.beforeDamage(this,params);return this}});
 f.item.getOriginData=()=>({actor:f.actor.uuid,uuid:f.item.uuid,type:'feat',rollOptions:[]});
 const fire=(name,...args)=>Promise.all([...(hooks.get(name)??[])].map(fn=>fn(...args)));
 const Hooks={on(name,fn){const wrapped=(...args)=>fn(...args);wrapped.hookId=++hookSequence;hooks.set(name,[...hooks.get(name)??[],wrapped]);return wrapped.hookId},off(name,id){hooks.set(name,(hooks.get(name)??[]).filter(fn=>fn.hookId!==id))}};
 const publish=async data=>{const m={...data,author:f.game.user,timestamp:300,isCheckRoll:data.flags?.pf2e?.context?.type==='saving-throw',isDamageRoll:data.flags?.pf2e?.context?.type==='damage-roll',item:f.item,actor:f.actor};for(const fn of [...hooks.get('preCreateChatMessage')??[]])if(fn(m)===false)return null;f.game.messages.set(m.id,m);f.docs.set(m.uuid,m);await fire('createChatMessage',m,{},f.game.user.id);return m};
 f.message.update=async data=>{for(const[k,v]of Object.entries(data)){if(k.startsWith('flags.' ))f.message.flags[ID][k.slice(('flags.'+ID+'.').length)]=v}};
 f.target.actor.getStatistic=()=>({check:{async roll(params){saves.push({user:f.game.user,params});if(outcome===null)return null;if(saveDialog){let resolve;const accepted=new Promise(r=>resolve=r),app={context:{options:new Set(params.extraRollOptions)},resolve,close:async()=>{}};dialogs.push(app);await fire('renderCheckModifiersDialog',app);saveDialog(app);if(!await accepted)return null}const card=await publish({id:'save',uuid:'ChatMessage.save',speaker:{actor:'b',scene:'scene',token:'target'},rolls:[{_evaluated:true,total:18}],flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:f.item.uuid},context:{type:'saving-throw',outcome,dc:params.dc,options:params.extraRollOptions}}}});params.callback(card.rolls[0],outcome,card);return card.rolls[0]}}});
 class DamageRoll {
  constructor(formula,_data={},options={}){this.formula=formula;this.options=options;this.total=21;this._evaluated=false;this.type='electricity'}
  async evaluate(){evaluations.push(this);this._evaluated=true;return this}
  alter(n,addend=0){const r=new DamageRoll(this.formula,{},{});r.total=Math.floor(this.total*n)+addend;r._evaluated=true;r.type=this.type;if(rollContexts.has(this))rollContexts.set(r,rollContexts.get(this));return r}
  async toMessage(data,options={}){publications.push({data,options});const m=await publish({...data,id:'damage',uuid:'ChatMessage.damage',rolls:[this]});messages.push(m);rollContexts.set(this,{messageId:m.id,rollIndex:0});return m}
 }
 if(siphon)f.receipt.snapshot={kind:'siphoning',siphon:{applies:true},level:5,itemUuid:f.item.uuid,actorUuid:f.actor.uuid,powerSourceUuid:HV,disruptive:true,associatedTraits:['electricity']};
 const provider=executorApi.createEldamonVoltageProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),DamageRoll,getRollContext:roll=>rollContexts.get(roll),selectChoice,
  manualDamageRoll:async({game,roll,assertLive})=>{damageWindows.push({user:game.user,roll});const accepted=typeof damageDialog==='function'?await damageDialog(roll):damageDialog!==false;if(!accepted)return null;assertLive?.();return roll.evaluate()},
  manualDamagePrivacy:roll=>{assert.ok(damageWindows.some(w=>w.roll===roll),'read the native audience before conversion replaces its Roll');return damagePrivacy??null},
  convertRoll:(roll,options)=>{assert.equal(options.rejectMixedPartitions,true);roll.type='untyped';return roll},onError:()=>{}});
 const socket={register:(name,fn)=>routes.set(name,fn),async executeAsUser(name,_gm,payload){const caller=f.game.user;f.game.user=f.gm;try{return await routes.get(name).call({socketdata:{userId:caller.id}},payload)}finally{f.game.user=caller}}};
 provider.register({socket,Hooks});
 const activation=()=>f.actor.flags[ID].voltage.activations.channel;
 const channel=()=>provider.onCommittedChannel({receipt:f.receipt,message:f.message,user:f.user});
 const trigger=()=>provider.trigger({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user);
 const save=async()=>{f.game.user=targetUser;return provider.rollSave(activation())};
 const damage=async()=>{f.game.user=f.user;return provider.rollDamage(activation())};
 const apply=async({token=f.target,multiplier=1,nativeReceipt=true,nativeError=false}={})=>{
  f.game.user=targetUser;const card=messages[0],params={damage:card.rolls[0].alter(multiplier,0),item:f.item,token,skipIWR:false,rollOptions:new Set(card.flags.pf2e.context.options),outcome:card.flags.pf2e.context.outcome};
  const prepared=await provider.beforeDamage(token.actor,params),actual=prepared?.params??params;applications.push(actual);
  if(nativeReceipt)await publish({id:'taken',uuid:'ChatMessage.taken',speaker:{actor:token.actor.id,scene:'scene',token:token.id},flags:{pf2e:{origin:f.item.getOriginData(),context:{type:'damage-taken',options:[...actual.rollOptions]},appliedDamage:actual.damage.total?{uuid:token.actor.uuid,isHealing:false,isReverted:false}:null}}});
  await provider.afterDamage(prepared.receipt,{applied:!nativeError,uncertain:nativeError});return actual;
 };
 return {...f,provider,messages,saves,applications,damageWindows,evaluations,publications,hooks,routes,fire,activation,channel,trigger,save,damage,apply,targetUser,publish,DamageRoll,dialogs};
}

test('an explicit owner trigger claims a response without opening saves, rolling damage or applying HP',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();
 assert.equal(f.saves.length,0);assert.equal(f.messages.length,0);assert.equal(f.applications.length,0);
 assert.equal(f.activation().status,'claimed');assert.equal(f.activation().native.phase,'awaiting-save');
});

test('ordinary native attack cards and token movement never trigger Voltage or rescan actor collections',async()=>{
 const f=executorFixture();await f.channel();let reads=0;f.game.actors.values=()=>{reads++;throw Error('unexpected actor scan')};
 await f.fire('createChatMessage',{flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid}}}}});
 await f.fire('updateToken',f.origin,{x:10,y:20});await f.fire('updateCombat',f.game.combat);
 assert.equal(reads,0);assert.equal(f.activation().status,'armed');assert.equal(f.saves.length,0);
});

test('target owner rolls the native save; source owner separately publishes bound native damage before manual application',async()=>{
 for(const[outcome,total]of [['criticalSuccess',0],['success',10],['failure',21],['criticalFailure',42]]){
  const f=executorFixture(outcome);await f.channel();await f.trigger();await f.save();
  assert.equal(f.saves[0].user,f.targetUser);assert.equal(f.messages.length,0);assert.equal(f.activation().native.phase,'awaiting-damage');
  await f.damage();assert.equal(f.applications.length,0);assert.equal(f.activation().status,'claimed');assert.equal(f.activation().native.phase,'awaiting-application');
  const card=f.messages[0];assert.equal(card.rolls[0].total,total);assert.equal(card.speaker.actor,'a');assert.equal(card.flags.pf2e.origin.uuid,f.item.uuid);
  assert.deepEqual(card.flags['pf2e-toolbelt'].targetHelper.targets,[f.target.uuid]);assert.equal(card.flags['pf2e-toolbelt'].targetHelper.saveVariants,undefined);
  assert.match(card.flavor,/全额/);const params=await f.apply();assert.equal(params.damage.total,total);assert.equal(params.skipIWR,false);
  assert.equal(f.activation().status,'done');assert.equal(f.activation().result.receiptUuid,'ChatMessage.taken');
  await assert.rejects(f.apply(),/消耗|结算|等待|认领/i);
 }
});

test('Siphon converts native Voltage damage and applies its target multiplier once without refreshing',async()=>{
 for(const[traits,total]of [[[],10],[['electricity'],21]]){
  const f=executorFixture('failure',{siphon:true,traits});await f.channel();await f.trigger();await f.save();await f.damage();
  assert.equal(f.messages[0].rolls[0].type,'untyped');assert.equal(f.messages[0].rolls[0].total,21);assert.equal(f.spent.system.frequency.value,0);
  const params=await f.apply();assert.equal(params.damage.total,total);assert.equal(params.damage.options[ID]?.metapowerDamage,undefined);assert.equal(f.activation().status,'done');
 }
});

test('owner and fixed target checks run before native actions or application claims',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();f.game.user=f.user;
 await assert.rejects(f.provider.rollSave(f.activation()),/拥有|权限/i);assert.equal(f.saves.length,0);
 await f.save();f.game.user=f.targetUser;await assert.rejects(f.provider.rollDamage(f.activation()),/拥有|权限/i);
 await f.damage();await assert.rejects(f.apply({token:f.origin}),/目标|绑定/i);assert.equal(f.activation().native.phase,'awaiting-application');
 await f.apply();
});

test('cancelled native save consumes the trigger and cannot be retried',async()=>{
 const f=executorFixture(null);await f.channel();await f.trigger();await f.save();
 assert.equal(f.activation().status,'cancelled');assert.equal(f.messages.length,0);f.game.user=f.gm;assert.equal(await f.trigger(),null);
 await assert.rejects(f.save(),/消耗|认领|等待/i);
});

test('Voltage save always requests its owner native confirmation even when the owner disables check dialogs',async()=>{
 const f=executorFixture(null);await f.channel();await f.trigger();
 f.targetUser.settings={showCheckDialogs:false};await f.save();
 assert.equal(f.saves[0].user,f.targetUser);
 assert.equal(f.saves[0].params.skipDialog,false,'the explicit continuation must wait for the native check window');
 assert.equal(f.saves[0].params.event,null,'a triggering click must not toggle the native dialog with shift');
 assert.equal(f.activation().nonce,'channel');assert.equal(f.activation().messageUuid,f.message.uuid);
 assert.equal(f.activation().status,'cancelled');assert.equal(f.messages.length,0);assert.equal(f.applications.length,0);
 assert.equal(f.actor.flags[ID].voltage.activeNonce,null);assert.equal(f.spent.system.frequency.value,1);
 assert.equal(f.receipt.status,'committed','closing a child check does not repeat or refund original Use');
});

test('Voltage damage waits for source owner native confirmation before rolling or publishing the bound kept-save result',async()=>{
 let release;const accepted=new Promise(resolve=>{release=resolve});
 const f=executorFixture('success',{damageDialog:()=>accepted});await f.channel();await f.trigger();await f.save();
 f.user.settings={showDamageDialogs:false};const pending=f.damage();
 try{
  await new Promise(setImmediate);
  assert.equal(f.damageWindows.length,1,'damage must enter the native window before evaluation');
  assert.equal(f.damageWindows[0].user,f.user);assert.equal(f.damageWindows[0].roll.formula,'6d6[electricity]');
  assert.equal(f.damageWindows[0].roll._evaluated,false);assert.equal(f.evaluations.length,0);
  assert.equal(f.messages.length,0);assert.equal(f.applications.length,0);assert.equal(f.activation().native.phase,'rolling-damage');
  await assert.rejects(f.provider.continueActivity({...f.payload,action:'damage'}),/结束|处理中|核对/);
  assert.equal(f.damageWindows.length,1,'another entrance cannot open another roll while the owner is choosing');
 }finally{release(true);await pending;}
 assert.equal(f.evaluations.length,1);assert.equal(f.messages.length,1);assert.equal(f.messages[0].rolls[0].total,10);
 assert.equal(f.messages[0].flags[ID].voltageDamage.nonce,'channel');
 assert.equal(f.messages[0].flags[ID].voltageDamage.targetUuid,f.target.uuid);
 assert.equal(f.activation().native.phase,'awaiting-application');assert.equal(f.applications.length,0);
});

test('closing the native Voltage damage window consumes only this same trigger without dice, output or new Use',async()=>{
 const f=executorFixture('failure',{damageDialog:false});await f.channel();await f.trigger();await f.save();
 const originalSave=f.activation().native.save.messageUuid;assert.equal(await f.damage(),null);
 assert.equal(f.damageWindows.length,1);assert.equal(f.evaluations.length,0);assert.equal(f.messages.length,0);assert.equal(f.applications.length,0);
 assert.equal(f.activation().nonce,'channel');assert.equal(f.activation().messageUuid,f.message.uuid);
 assert.equal(f.activation().native.save.messageUuid,originalSave);assert.equal(f.activation().status,'cancelled');assert.equal(f.activation().native.phase,'cancelled');
 assert.equal(f.actor.flags[ID].voltage.activeNonce,null);assert.equal(f.spent.system.frequency.value,1);assert.equal(f.receipt.status,'committed');
 await assert.rejects(f.damage(),/消耗|等待|认领/);assert.equal(f.damageWindows.length,1);
});

test('an uncertain native Voltage damage window never publishes or refunds the original activity',async()=>{
 const f=executorFixture('failure',{damageDialog:()=>{throw Error('native window failed')}});await f.channel();await f.trigger();await f.save();
 await assert.rejects(f.damage(),/native window failed/);
 assert.equal(f.evaluations.length,0);assert.equal(f.messages.length,0);assert.equal(f.applications.length,0);
 assert.equal(f.activation().native.phase,'uncertain');assert.equal(f.receipt.status,'committed');assert.equal(f.spent.system.frequency.value,1);
 await assert.rejects(f.damage(),/消耗|等待|认领/);assert.equal(f.damageWindows.length,1);
});

for(const mutation of ['original-card','source-item','source-actor','origin-token','target-relink','owner-permission','gm-handoff','operation','original-save'])test(`a Voltage damage window cannot publish a stale ${mutation} after waiting for its owner`,async()=>{
 let accept;const gate=new Promise(resolve=>accept=resolve),f=executorFixture('failure',{damageDialog:()=>gate});await f.channel();await f.trigger();await f.save();
 const pending=f.damage();pending.catch(()=>{});await new Promise(setImmediate);assert.equal(f.damageWindows.length,1);assert.equal(f.messages.length,0);
 if(mutation==='original-card'){f.game.messages.delete(f.message.id);f.docs.delete(f.message.uuid)}
 if(mutation==='source-item')f.actor.items.delete(f.item.id);
 if(mutation==='source-actor')f.game.actors.delete(f.actor.id);
 if(mutation==='origin-token')f.origin.parent.tokens.delete(f.origin.id);
 if(mutation==='target-relink')f.target.actor={...f.target.actor,uuid:'Actor.replacement',id:'replacement'};
 if(mutation==='owner-permission')f.actor.testUserPermission=()=>false;
 if(mutation==='gm-handoff'){const gm={id:'new-gm',isGM:true,active:true};f.game.users.set(gm.id,gm);f.game.users.activeGM=gm}
 if(mutation==='operation')f.activation().native.damage.id='another-operation';
 if(mutation==='original-save')f.game.messages.delete('save');
 accept(true);await assert.rejects(pending);assert.equal(f.evaluations.length,0,'the same submit guard reaches the native damage window before dice');assert.equal(f.messages.length,0,'an invalidated native window must not publish an orphan damage card');assert.equal(f.publications.length,0);assert.equal(f.applications.length,0);assert.equal(f.receipt.status,'committed');assert.equal(f.spent.system.frequency.value,1);
});

for(const mutation of ['owner-permission','gm-handoff','original-card','target-relink'])test(`the exact native Voltage save rejects ${mutation} at acceptance without publishing a stale check`,async()=>{
 const f=executorFixture('failure',{saveDialog:()=>{}});await f.channel();await f.trigger();const before=[...f.hooks.values()].reduce((n,list)=>n+list.length,0),pending=f.save();pending.catch(()=>{});await new Promise(setImmediate);assert.equal(f.dialogs.length,1);
 if(mutation==='owner-permission')f.target.actor.testUserPermission=()=>false;
 if(mutation==='gm-handoff'){const gm={id:'new-gm',isGM:true,active:true};f.game.users.set(gm.id,gm);f.game.users.activeGM=gm}
 if(mutation==='original-card')f.game.messages.delete(f.message.id);
 if(mutation==='target-relink')f.target.actor={...f.target.actor,uuid:'Actor.replacement',id:'replacement'};
 f.dialogs[0].resolve(true);await assert.rejects(pending);assert.equal(f.game.messages.has('save'),false,'native submit guard must reject before any check card is published');assert.equal(f.messages.length,0);assert.equal(f.applications.length,0);assert.equal([...f.hooks.values()].reduce((n,list)=>n+list.length,0),before);
});

test('Voltage rechecks its exact scope after asynchronous final rendering and does not block unrelated native cards',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();let entered,release;const entry=new Promise(resolve=>entered=resolve),gate=new Promise(resolve=>release=resolve),native=f.DamageRoll.prototype.toMessage;
 f.DamageRoll.prototype.toMessage=async function(...args){entered();await gate;return native.apply(this,args)};
 const before=[...f.hooks.values()].reduce((n,list)=>n+list.length,0),pending=f.damage();pending.catch(()=>{});await entry;
 const unrelated=await f.publish({id:'ordinary',uuid:'ChatMessage.ordinary',rolls:[],flags:{pf2e:{context:{type:'damage-roll',options:['ordinary-native']}}}});assert.equal(f.game.messages.get('ordinary'),unrelated);
 f.game.messages.delete(f.message.id);release();await assert.rejects(pending);assert.equal(f.game.messages.has('damage'),false);assert.equal(f.messages.filter(Boolean).length,0);assert.equal(f.applications.length,0);assert.equal([...f.hooks.values()].reduce((n,list)=>n+list.length,0),before);
});

for(const [messageMode,blind,whisper]of [['blind',true,['gm']],['gm',false,['gm']],['self',false,['owner']],['public',false,[]]])test(`Voltage preserves the native ${messageMode} damage audience through Siphon and kept-save scaling`,async()=>{
 const f=executorFixture('success',{siphon:true,damagePrivacy:{messageMode,blind,whisper}});await f.channel();await f.trigger();await f.save();await f.damage();
 assert.equal(f.messages.length,1);assert.equal(f.messages[0].blind,blind);assert.deepEqual(f.messages[0].whisper,whisper);
 assert.equal(f.publications[0].options.messageMode,messageMode,'Foundry toMessage must receive the native mode, not default public');
 assert.equal(f.messages[0].flags.pf2e.context.messageMode,messageMode);assert.equal(f.messages[0].rolls[0].type,'untyped');assert.equal(f.messages[0].rolls[0].total,10);
 assert.equal(f.messages[0].flags[ID].voltageDamage.nonce,'channel');assert.equal(f.activation().native.phase,'awaiting-application');assert.equal(f.applications.length,0);
});

test('application without its native receipt stays uncertain and never replays HP',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();await f.apply({nativeReceipt:false,nativeError:true});
 assert.equal(f.activation().status,'uncertain');await assert.rejects(f.apply(),/消耗|认领|等待/i);assert.equal(f.applications.length,1);
});

test('changed original save or damage source cannot authorize a native application',async()=>{
 for(const mutate of [f=>f.docs.get('ChatMessage.save').flags.pf2e.context.outcome='criticalFailure',f=>f.messages[0].rolls[0].total=999,f=>f.actor.items.delete(f.item.id)]){
  const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();mutate(f);
  await assert.rejects(f.apply(),/改变|无效|来源|绑定/i);assert.equal(f.applications.length,0);
 }
});

test('legacy claimed and consumed Voltage records never acquire resumable native actions',async()=>{
 for(const status of ['claimed','done','uncertain','cancelled']){
  const f=executorFixture();await f.channel();await f.provider.ledger.claim({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user);
  f.actor.flags[ID].voltage.activations.channel.status=status;await assert.rejects(f.save(),/旧|核对|消耗|认领/i);
  assert.equal(f.saves.length,0);assert.equal(f.activation().status,status);
 }
});

test('foreign rolls and voltage options without a real tracked damage card are rejected',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();f.game.user=f.targetUser;
 await assert.rejects(f.provider.beforeDamage(f.target.actor,{damage:{options:{},total:21},item:f.item,token:f.target,rollOptions:new Set(f.messages[0].flags.pf2e.context.options)}),/原生|记录|来源|授权/i);
 assert.equal(f.activation().native.phase,'awaiting-application');
});

test('registration authenticates the socket requester while committed channel preserves original author and refresh once',async()=>{
 const f=executorFixture();await f.channel();assert.equal(f.activation().userId,f.user.id);assert.equal(f.spent.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.channel();assert.equal(f.spent.system.frequency.value,0);
 const denied=await f.routes.get('voltage:trigger').call({socketdata:{userId:'stranger'}},{...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true});
 assert.equal(denied.ok,false);assert.equal(f.activation().status,'armed');
});

test('the source owner declares an authentic native unarmed hit without another confirmation',async()=>{
 const f=executorFixture();await f.channel();
 const attack=await f.publish({id:'attack',uuid:'ChatMessage.attack',isCheckRoll:true,rolls:[{_evaluated:true}],timestamp:200,speaker:{scene:'scene',token:'target',actor:'b'},flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:['item:melee']}}}});
 attack.isCheckRoll=true;attack.item={actor:f.target.actor,isMelee:true,system:{category:'unarmed'}};
 await f.provider.trigger({...f.payload,kind:'attack',attackUuid:attack.uuid},f.user);
 assert.equal(f.activation().native.phase,'awaiting-save');assert.equal(f.saves.length,0);
});

test('a changed original High Voltage card cannot claim a new native response',async()=>{
 const f=executorFixture();await f.channel();f.message.flags[ID].metapowerUse.itemUuid='Actor.other.Item.copy';
 await assert.rejects(f.trigger(),/原|绑定|卡/i);assert.equal(f.activation().status,'armed');
});

test('only the bound target owner sees the native save control on the original card',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();f.game.scenes=new Map([[f.origin.parent.id,f.origin.parent]]);
 const render=user=>{f.game.user=user;const root=voltageDOM();f.provider.renderCard(f.message,root);return root.querySelectorAll('*').filter(node=>node.tag==='button').map(node=>node.textContent)};
 assert.deepEqual(render(f.user),[]);assert.deepEqual(render(f.targetUser),['绑定目标：原生反射豁免']);assert.deepEqual(render(f.gm),['绑定目标：原生反射豁免']);
});

const nativePath=process.env.PF2E_NATIVE_BUNDLE??'';
test('actual PF2e damage-card application keeps baked save scaling and settles the bound native receipt',{skip:!existsSync(nativePath)},async()=>{
 const source=readFileSync(nativePath,'utf8'),start=source.indexOf('async function applyDamageFromMessage('),end=source.indexOf('\nasync function shiftAdjustDamage',start);
 assert.ok(start>=0&&end>start,'Native damage-card boundary changed');
 for(const[siphon,total]of [[false,10],[true,5]]){
  const f=executorFixture('success',{siphon});await f.channel();await f.trigger();await f.save();await f.damage();f.game.user=f.targetUser;
  f.game.user.getActiveTokens=()=>[f.target];f.item.isOfType=()=>false;
  f.target.actor.getContextualClone=()=>({...f.target.actor,applyDamage:params=>runDamagePipeline({actor:f.target.actor,params,providers:[f.provider],apply:async actual=>{
   f.applications.push(actual);await f.publish({id:'taken',uuid:'ChatMessage.taken',speaker:{actor:'b',scene:'scene',token:'target'},flags:{pf2e:{origin:f.item.getOriginData(),context:{type:'damage-taken',options:[...actual.rollOptions]},appliedDamage:{uuid:f.target.actor.uuid,isHealing:false,isReverted:false}}}});return f.target.actor;
  }})});
  const native=vm.runInNewContext('('+source.slice(start,end)+')',{game:f.game,ui:{chat:{element:{}}},htmlQuery:()=>null,cn:f.DamageRoll,CONFIG:{PF2E:{chatDamageButtonShieldToggle:false}},gt:tokens=>tokens,extractEphemeralEffects:async()=>[],toggleOffShieldBlock(){},ErrorPF2e:Error});
  await native({message:f.messages[0],multiplier:1});
  assert.equal(f.applications.length,1);assert.equal(f.applications[0].damage.total,total);assert.equal(f.applications[0].skipIWR,false);assert.equal(f.applications[0].item,f.item);assert.equal(f.activation().status,'done');
  await assert.rejects(native({message:f.messages[0],multiplier:1}),/消耗|认领|等待/i);assert.equal(f.applications.length,1);
 }
});

test('an error after the authentic native receipt settles once without replaying damage',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();await f.apply({nativeError:true});
 assert.equal(f.activation().status,'done');await assert.rejects(f.apply(),/消耗|认领|等待/i);assert.equal(f.applications.length,1);
});

test('a native application with a wrong-user or reverted receipt cannot settle the Voltage claim',async()=>{
 for(const mutate of [m=>m.author={id:'stranger'},m=>m.flags.pf2e.appliedDamage.isReverted=true,m=>m.speaker.token='other']){
  const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();f.game.user=f.gm;
  const application=await f.provider.ledger.beginDamage({...f.payload,operationId:'native',damageUuid:'ChatMessage.damage',targetUuid:f.target.uuid,targetActorUuid:f.target.actor.uuid,rollIndex:0},f.targetUser);
  const receipt={id:'taken',uuid:'ChatMessage.taken',author:f.targetUser,speaker:{actor:'b',scene:'scene',token:'target'},flags:{pf2e:{origin:f.item.getOriginData(),context:{type:'damage-taken',options:[api.voltageRollOption(application),api.VOLTAGE_APPLY_PREFIX+'native']},appliedDamage:{uuid:f.target.actor.uuid,isHealing:false,isReverted:false}}}};
  mutate(receipt);f.game.messages.set(receipt.id,receipt);f.docs.set(receipt.uuid,receipt);
  await assert.rejects(f.provider.ledger.finishDamage({...f.payload,operationId:'native',receiptUuid:receipt.uuid},f.targetUser),/真实|回执/i);assert.equal(f.activation().native.phase,'applying');
  await assert.rejects(f.provider.ledger.beginDamage({...f.payload,operationId:'again',damageUuid:'ChatMessage.damage',targetUuid:f.target.uuid,targetActorUuid:f.target.actor.uuid,rollIndex:0},f.targetUser),/消耗|等待/i);
 }
});

test('an owner can confirm the current kept native attack reroll without an automatic attack observer',async()=>{
 const f=executorFixture();await f.channel();
 const attack=await f.publish({id:'kept',uuid:'ChatMessage.kept',rolls:[{_evaluated:true}],speaker:{scene:'scene',token:'target',actor:'b'},flags:{pf2e:{context:{type:'attack-roll',isReroll:true,outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:['item:melee','check:reroll']}}}});
 attack.isCheckRoll=true;attack.item={uuid:'Actor.b.Item.fist',actor:f.target.actor,isMelee:true,system:{category:'unarmed'}};
 const claimed=await f.provider.trigger({...f.payload,kind:'attack',attackUuid:attack.uuid,confirmed:true},f.user);
 assert.equal(claimed?.native.phase,'awaiting-save');assert.equal(f.saves.length,0);
});

test('the damage click binds the unique kept native save reroll after PF2e deletes the original save',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();const prior=f.docs.get('ChatMessage.save');
 f.game.messages.delete(prior.id);f.docs.delete(prior.uuid);
 const kept=await f.publish({...prior,id:'save-kept',uuid:'ChatMessage.save-kept',flags:structuredClone(prior.flags),rolls:[{_evaluated:true,total:24}]});
 kept.flags.pf2e.context.isReroll=true;kept.flags.pf2e.context.outcome='success';kept.flags.pf2e.context.options.push('check:reroll');
 await f.damage();assert.equal(f.messages[0].rolls[0].total,10);assert.equal(f.activation().native.save.messageUuid,kept.uuid);
 await f.apply();assert.equal(f.activation().result.saveUuid,kept.uuid);
});

test('the legacy manual settlement endpoint cannot mark new native claims done without a damage receipt',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();
 await assert.rejects(f.provider.ledger.settle({...f.payload,status:'done'},f.gm),/原生|回执/i);assert.equal(f.activation().status,'claimed');
});

test('concurrent native save requests reserve only one owner interaction',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();
 const results=await Promise.allSettled(['first','second'].map(operationId=>f.provider.ledger.beginSave({...f.payload,operationId},f.targetUser)));
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);assert.equal(f.activation().native.phase,'rolling-save');
 assert.equal(f.activation().native.save.id,'first');assert.equal(f.saves.length,0);
});

test('ambiguous copied save rerolls cannot determine a new damage amount',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();const original=f.docs.get('ChatMessage.save');f.game.messages.delete(original.id);f.docs.delete(original.uuid);
 for(const id of ['kept-one','kept-two']){
  const message=await f.publish({...original,id,uuid:'ChatMessage.'+id,flags:structuredClone(original.flags)});message.flags.pf2e.context.isReroll=true;message.flags.pf2e.context.options.push('check:reroll');
 }
 await assert.rejects(f.damage(),/改变|核对/i);assert.equal(f.messages.length,0);assert.equal(f.activation().native.phase,'awaiting-damage');
});

test('a native failed attack rerolled to a hit keeps its original activation provenance and triggers once',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 const attack={isCheckRoll:true,rolls:[{_evaluated:true}],id:'miss',uuid:'ChatMessage.miss',timestamp:200,author:f.gm,speaker:{scene:'scene',token:'target',actor:'b'},item:{uuid:'Actor.b.Item.weapon',actor:f.target.actor,isMelee:true,type:'weapon',system:{}},flags:{pf2e:{context:{type:'attack-roll',outcome:'failure',target:{actor:f.actor.uuid,token:f.origin.uuid},options:[]}}},async update(data){this.flags.pf2e.context.options=data['flags.pf2e.context.options']}};
 f.docs.set(attack.uuid,attack);f.game.messages.set(attack.id,attack);assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:attack.uuid},f.gm),null);
 const reroll={...attack,id:'reroll',uuid:'ChatMessage.reroll',timestamp:210,flags:structuredClone(attack.flags)};reroll.flags.pf2e.context.isReroll=true;reroll.flags.pf2e.context.outcome='success';reroll.flags.pf2e.context.options.push('check:reroll');
 f.game.messages.delete(attack.id);f.docs.delete(attack.uuid);f.docs.set(reroll.uuid,reroll);f.game.messages.set(reroll.id,reroll);
 assert.ok(await f.service().claim({...f.payload,kind:'attack',attackUuid:reroll.uuid},f.gm));assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:reroll.uuid},f.gm),null);
});
test('the optional unknown-metal declaration is bound to one native hit without geometry',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);f.origin.object.distanceTo=()=>{throw Error('geometry accessed')};
 const attack={isCheckRoll:true,rolls:[{_evaluated:true}],id:'reach',uuid:'ChatMessage.reach',timestamp:200,author:f.gm,speaker:{scene:'scene',token:'target',actor:'b'},item:{uuid:'Actor.b.Item.weapon',actor:f.target.actor,isMelee:true,type:'weapon',system:{}},flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:[]}}},async update(data){this.flags.pf2e.context.options=data['flags.pf2e.context.options']}};f.docs.set(attack.uuid,attack);f.game.messages.set(attack.id,attack);
 await assert.rejects(s.claim({...f.payload,kind:'metal-hit',attackUuid:attack.uuid},f.user),/声明|确认/i);
 assert.ok(await s.claim({...f.payload,kind:'metal-hit',attackUuid:attack.uuid,confirmed:true},f.user));
});
test('outside an encounter normal High Voltage still Refreshes but identifies the unavailable turn window',async()=>{
 const f=fixture();f.game.combat=null;const r=await f.service().channel(f.payload,f.user);assert.equal(f.spent.system.frequency.value,1);assert.equal(r.expiryReason,'no-encounter');assert.equal(r.status,'expired');
});
test('a chat card with attack-shaped flags but no evaluated native check does not trigger High Voltage',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);const message={id:'fake',uuid:'ChatMessage.fake',timestamp:200,speaker:{scene:'scene',token:'target',actor:'b'},item:{actor:f.target.actor,isMelee:true,system:{}},flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:[]}}}};f.docs.set(message.uuid,message);f.game.messages.set(message.id,message);
 assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:message.uuid},f.gm),null);
});
test('source turn advancing while native attack evidence is persisted expires before a claim',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 const message={isCheckRoll:true,rolls:[{_evaluated:true}],id:'late',uuid:'ChatMessage.late',timestamp:200,speaker:{scene:'scene',token:'target',actor:'b'},item:{actor:f.target.actor,isMelee:true,system:{}},flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:[]}}},async update(){f.game.combat.round=2;f.game.combat.turn=0}};f.docs.set(message.uuid,message);f.game.messages.set(message.id,message);
 assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:message.uuid},f.gm),null);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'expired');
});
test('standalone Refresh UI entry uses the owned feature through the native observer and emits a two-action original card',async()=>{
 const f=fixture(),feature=f.addItem('feature','Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.naawsnBug9EOpzfN','elemental-powers'),created=[],savedConfig=globalThis.CONFIG;
 feature.toMessage=async(_event,options)=>{assert.equal(options.create,false);return {toObject:()=>({speaker:{actor:'a'},flags:{pf2e:{origin:{uuid:feature.uuid}}},content:'feature description'})}};
 globalThis.CONFIG={ChatMessage:{documentClass:{async create(data){created.push(data);return data}}}};
 try{
  const provider=executorApi.createEldamonVoltageProvider({game:f.game,fromUuid:async u=>f.docs.get(u),observe:async(context,native)=>{assert.equal(context.actor,f.actor);assert.equal(context.item,feature);return native()}});
  await provider.useRefresh(f.actor,{});assert.equal(created.length,1);assert.equal(created[0].flags[ID].voltageRefreshActivity.actions,2);assert.equal(created[0].flags.pf2e.origin.uuid,feature.uuid);assert.match(created[0].content,/刷新威能 · 2动作/);assert.doesNotMatch(created[0].content,/Refresh/);assert.equal(feature.system.actionType,undefined);
 }finally{globalThis.CONFIG=savedConfig;}
});

function voltageDOM(){
 const node=tag=>({tag,dataset:{},style:{},children:[],listeners:{},isConnected:true,append(child){child.parent=this;this.children.push(child)},remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this)},setAttribute(){},removeAttribute(){},addEventListener(name,fn){this.listeners[name]=fn},querySelector(selector){return this.querySelectorAll(selector)[0]??null},querySelectorAll(selector){const all=this.children.flatMap(c=>[c,...c.querySelectorAll('*')]);if(selector==='*')return all;const field=selector.match(/^\[data-([\w-]+)\]$/)?.[1]?.replace(/-([a-z])/g,(_m,c)=>c.toUpperCase());return field?all.filter(c=>Object.hasOwn(c.dataset,field)):[]}});
 const root=node('root');root.ownerDocument={createElement:node};return root;
}
async function nativeVoltageHit(f,{id='local-hit',metal=null}={}){
 const attack=await f.publish({id,uuid:'ChatMessage.'+id,timestamp:200,isCheckRoll:true,rolls:[{_evaluated:true}],speaker:{scene:'scene',token:'target',actor:'b'},flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:['item:melee']}}}});
 attack.isCheckRoll=true;attack.item={actor:f.target.actor,uuid:'Actor.b.Item.native-weapon',isMelee:true,type:'weapon',system:{category:'martial',material:{type:metal}}};await f.fire('updateChatMessage',attack,{});return attack;
}
test('a current real melee hit declares the trigger once without confirmation or geometry access',async()=>{
 const f=executorFixture();await f.channel();f.origin.object.distanceTo=()=>{throw Error('geometry accessed')};
 const attack=await nativeVoltageHit(f);await f.provider.trigger({...f.payload,kind:'attack',attackUuid:attack.uuid},f.user);
 assert.equal(f.activation().native.phase,'awaiting-save');assert.equal(f.saves.length,0);assert.equal(f.messages.length,0);
});
test('the current hit card offers the original owner one bound continuation without replaying Use',async()=>{
 const f=executorFixture();await f.channel();const attack=await nativeVoltageHit(f);f.game.user=f.user;
 const root=voltageDOM();f.provider.renderCard(attack,root);const hit=root.querySelectorAll('*').find(n=>n.dataset.voltageAction==='hit');
 assert.ok(hit,'the current native hit card needs its local response');await hit.listeners.click({preventDefault(){},stopPropagation(){}});
 assert.equal(f.activation().trigger.attackUuid,attack.uuid);assert.equal(f.activation().messageUuid,f.message.uuid);assert.equal(f.activation().nonce,'channel');assert.equal(f.saves.length,0);
});
test('a first-seen kept native attack reroll continues from its current card without another confirmation',async()=>{
 const f=executorFixture();await f.channel();const attack=await nativeVoltageHit(f,{id:'first-kept-hit'});
 attack.flags.pf2e.context.isReroll=true;attack.flags.pf2e.context.options.push('check:reroll');await f.fire('updateChatMessage',attack,{});f.game.user=f.user;
 const root=voltageDOM();f.provider.renderCard(attack,root);const hit=root.querySelectorAll('*').find(n=>n.dataset.voltageAction==='hit');
 assert.ok(hit);await hit.listeners.click({preventDefault(){},stopPropagation(){}});
 assert.equal(f.activation().status,'claimed');assert.equal(f.activation().trigger.attackUuid,attack.uuid);assert.equal(f.activation().native.phase,'awaiting-save');assert.equal(f.saves.length,0);
});
test('claimed activities remain discoverable on the target actor after activeNonce clears',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();f.game.user=f.targetUser;
 assert.equal(typeof f.provider.getContinuations,'function');assert.equal(f.actor.flags[ID].voltage.activeNonce,null);
 const entries=f.provider.getContinuations(f.target.actor.uuid);assert.equal(entries.length,1);assert.equal(entries[0].actorUuid,f.actor.uuid);assert.ok(entries[0].actions.some(a=>a.action==='save'));
 const root=voltageDOM();f.provider.renderActorContinuations(f.target.actor,root);assert.ok(root.querySelectorAll('*').some(n=>n.dataset.voltageAction==='save'));
 f.game.user={id:'stranger'};assert.deepEqual(f.provider.getContinuations(f.target.actor.uuid),[]);
});
test('a native save result offers caster damage and preserves the hero-reroll window',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();f.game.user=f.user;
 const root=voltageDOM();f.provider.renderCard(f.docs.get('ChatMessage.save'),root);
 assert.ok(root.querySelectorAll('*').some(n=>n.dataset.voltageAction==='damage'));assert.equal(f.messages.length,0,'a save never auto-rolls damage');
 f.game.user=f.targetUser;const targetRoot=voltageDOM();f.provider.renderCard(f.docs.get('ChatMessage.save'),targetRoot);assert.ok(!targetRoot.querySelectorAll('*').some(n=>n.dataset.voltageAction==='damage'));
});
test('two entrances share the native phase claim and cannot roll a save twice',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();f.game.user=f.gm;
 assert.equal(typeof f.provider.continueActivity,'function');const command={...f.payload,action:'save'};
 await Promise.allSettled([f.provider.continueActivity(command),f.provider.continueActivity(command)]);
 assert.equal(f.saves.length,1);assert.equal(f.activation().native.phase,'awaiting-damage');assert.equal(f.messages.length,0);
});
test('uncertain activities expose no runnable continuation and are not replayed',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();f.game.user=f.user;
 f.activation().status='uncertain';f.activation().native.phase='uncertain';await f.fire('updateActor',f.actor,{});
 assert.equal(typeof f.provider.continueActivity,'function');assert.ok(f.provider.getContinuations(f.actor.uuid).every(a=>a.actions.length===0));
 await assert.rejects(f.provider.continueActivity({...f.payload,action:'damage'}),/不确定|核对|阶段|继续/);assert.equal(f.messages.length,0);
});
test('local continuations recover chat once and do not scan it during rendering or selection',async()=>{
 const f=executorFixture();await f.channel();const attack=await nativeVoltageHit(f);f.game.user=f.user;
 assert.equal(typeof f.provider.getContinuations,'function');f.provider.getContinuations(f.actor.uuid);
 f.game.messages.values=()=>{throw Error('whole chat scan')};const root=voltageDOM();f.provider.renderCard(attack,root);f.provider.renderActorContinuations(f.actor,voltageDOM());
 await f.provider.continueActivity({...f.payload,action:'hit',attackUuid:attack.uuid});assert.equal(f.activation().status,'claimed');
});
test('touch branch choices use recipient-safe names and declare the fact only once',async()=>{
 const choices=[];const f=executorFixture('failure',{selectChoice:async input=>{choices.push(input);return f.target.uuid}});await f.channel();f.game.user=f.user;
 f.game.pf2e={settings:{tokens:{nameVisibility:true}}};f.target.name='SECRET TARGET';f.target.actor.name='SECRET ACTOR';
 assert.equal(typeof f.provider.continueActivity,'function');await f.provider.continueActivity({...f.payload,action:'touch'});
 assert.equal(choices.length,1);assert.ok(choices[0].choices.every(c=>!c.label.includes('SECRET')));assert.equal(f.activation().status,'claimed');assert.equal(f.saves.length,0);
});
test('a source or target relink leaves no runnable projected step',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();f.game.user=f.user;
 f.target.actor={id:'replacement',uuid:'Actor.replacement',testUserPermission:()=>true};
 assert.ok(f.provider.getContinuations(f.actor.uuid).every(entry=>entry.actions.length===0));
 await assert.rejects(f.provider.continueActivity({...f.payload,action:'damage'}),/目标|阶段|核对|结束/);assert.equal(f.messages.length,0);
});
for(const phase of ['save','damage'])test(`deleting the original channel card removes an already rendered ${phase} continuation without native side effects`,async()=>{
 const f=executorFixture();await f.channel();await f.trigger();if(phase==='damage')await f.save();
 f.game.user=phase==='save'?f.targetUser:f.user;const participant=phase==='save'?f.target.actor:f.actor,root=voltageDOM();
 f.provider.renderActorContinuations(participant,root);assert.ok(root.querySelectorAll('*').some(n=>n.dataset.voltageAction===phase));
 const savesBefore=f.saves.length;f.game.messages.delete(f.message.id);f.docs.delete(f.message.uuid);await f.fire('deleteChatMessage',f.message);
 assert.ok(!root.querySelectorAll('*').some(n=>n.dataset.voltageAction),'the existing actor view must refresh when its source card is deleted');
 const entries=f.provider.getContinuations(participant.uuid);assert.equal(entries.length,1);assert.deepEqual(entries[0].actions,[]);assert.match(entries[0].label,/原来源.*已改变/);
 await assert.rejects(f.provider.continueActivity({...f.payload,action:phase}),/核对|结束|改变/);
 assert.equal(f.saves.length,savesBefore);assert.equal(f.messages.length,0);assert.equal(f.applications.length,0);
});

test('a newly rendered detached save card receives the completed phase before insertion',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();f.game.user=f.user;
 const a=f.activation();a.native.phase='rolling-save';const card=await f.publish({id:'pending-save-view',uuid:'ChatMessage.pending-save-view',speaker:{actor:'b',scene:'scene',token:'target'},rolls:[{_evaluated:true}],flags:{pf2e:{context:{type:'saving-throw',options:[api.voltageRollOption(a)]}}}});
 const root=voltageDOM();root.isConnected=false;f.provider.renderCard(card,root);a.native.phase='awaiting-damage';await f.fire('updateActor',f.actor,{});
 assert.ok(root.querySelectorAll('*').some(n=>n.dataset.voltageAction==='damage'),'Foundry renders HTML before attaching it to the chat log');
});
test('a private blind native hit is not revealed by card or actor continuation choices',async()=>{
 const f=executorFixture();await f.channel();const attack=await nativeVoltageHit(f);attack.blind=true;f.game.user=f.user;
 const root=voltageDOM();f.provider.renderCard(attack,root);assert.ok(!root.querySelectorAll('*').some(n=>n.dataset.voltageAction==='hit'));
 assert.ok(f.provider.getContinuations(f.actor.uuid).every(e=>e.actions.every(a=>a.action!=='hit')));
 f.game.user=f.gm;assert.equal(await f.provider.trigger({...f.payload,kind:'attack',attackUuid:attack.uuid},f.user),null);assert.equal(f.activation().status,'armed');
});
test('fresh registration recovers claimed target-owner continuation without replaying a native roll',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();f.game.user=f.targetUser;
 const restored=executorApi.createEldamonVoltageProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),DamageRoll:f.DamageRoll});
 restored.register({socket:{register(){}},Hooks:{on(){}}});const entries=restored.getContinuations(f.target.actor.uuid);
 assert.equal(entries.length,1);assert.equal(entries[0].nonce,'channel');assert.ok(entries[0].actions.some(a=>a.action==='save'));assert.equal(f.saves.length,0);assert.equal(f.messages.length,0);
});
test('the registered native target sheet shows continuation without owning an elemental feature',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();f.game.user=f.targetUser;const app={actor:f.target.actor},root=voltageDOM();
 await f.fire('renderActorSheetPF2e',app,root);assert.ok(root.querySelectorAll('*').some(n=>n.dataset.voltageAction==='save'));
 await f.fire('closeActorSheetPF2e',app);await f.save();await f.fire('updateActor',f.actor,{});
 assert.ok(!root.querySelectorAll('*').some(n=>n.dataset.voltageAction==='damage'),'a closed sheet does not retain a live UI observer');
});
test('kept save reroll is adopted through its incremental source index at the explicit damage click',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();const old=f.docs.get('ChatMessage.save');
 f.game.messages.delete(old.id);f.docs.delete(old.uuid);await f.fire('deleteChatMessage',old);
 const flags=structuredClone(old.flags);flags.pf2e.context.isReroll=true;flags.pf2e.context.outcome='success';flags.pf2e.context.options.push('check:reroll');
 const kept=await f.publish({id:'indexed-kept-save',uuid:'ChatMessage.indexed-kept-save',speaker:old.speaker,rolls:[{_evaluated:true,total:23}],flags});
 f.game.messages.values=()=>{throw Error('whole chat scan')};f.game.user=f.user;
 const root=voltageDOM();f.provider.renderCard(kept,root);assert.ok(root.querySelectorAll('*').some(n=>n.dataset.voltageAction==='damage'));assert.equal(f.messages.length,0);
 await f.provider.continueActivity({...f.payload,action:'damage'});assert.equal(f.messages.length,1);assert.equal(f.activation().native.save.messageUuid,kept.uuid);assert.equal(f.messages[0].rolls[0].total,10);
});
