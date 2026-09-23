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
 const addItem=(id,source,slug,frequency)=>{const item={id,uuid:actor.uuid+'.Item.'+id,sourceId:source,actor,type:'feat',name:slug,flags:{},system:{slug,frequency,traits:{value:['electricity'],otherTags:['eldamon-power']}},async update(data){for(const[k,v]of Object.entries(data))set(this,k,v)}};actor.items.set(id,item);docs.set(item.uuid,item);return item};
 const item=addItem('h',HV,'high-voltage'),spent=addItem('s','Compendium.battlezoo-eldamon-pf2e.powers.Item.veFrnrxYjlqca13w','electric-surge',{max:1,per:'PT10M',value:0}),reaction=addItem('r','Compendium.battlezoo-eldamon-pf2e.powers.Item.fzV5Ly3a9nEsfcAJ','reactive-chain',{max:1,per:'PT10M',value:0});
 const scene={id:'scene',tokens:new Map()},token=(id,a)=>{const t={id,uuid:'Scene.scene.Token.'+id,parent:scene,actor:a,object:{distanceTo:()=>5}};scene.tokens.set(id,t);docs.set(t.uuid,t);return t};
 const origin=token('origin',actor),targetActor={id:'b',uuid:'Actor.b',type:'npc'},target=token('target',targetActor);docs.set(targetActor.uuid,targetActor);
 game.combat.turns=[{id:'ca',actor,token:origin},{id:'cb',actor:targetActor,token:target}];game.combat.combatant=game.combat.turns[0];
 const receipt={nonce:'channel',userId:user.id,actorUuid:actor.uuid,itemUuid:item.uuid,sourceUuid:HV,status:'committed',messageUuid:'ChatMessage.card',snapshot:null,turn:'fight:1:0:ca'};
 const message={id:'card',uuid:'ChatMessage.card',timestamp:100,speaker:{actor:'a',scene:'scene',token:'origin'},author:user,flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid}},[ID]:{metapowerUse:{nonce:'channel',actorUuid:actor.uuid,itemUuid:item.uuid}}}};
 actor.flags[ID]={metapower:{receipts:{channel:receipt}}};docs.set(message.uuid,message);game.messages.set(message.id,message);
 const payload={actorUuid:actor.uuid,nonce:'channel',messageUuid:message.uuid};
 const service=()=>api.createVoltageLedger({game,fromUuid:async uuid=>docs.get(uuid)});
 return{game,gm,user,actor,item,spent,reaction,addItem,origin,target,docs,message,receipt,payload,service};
}
test('normal original High Voltage channel refreshes actual spent prepared powers immediately and only once',async()=>{
 assert.equal(typeof api.createVoltageLedger,'function');const f=fixture(),s=f.service();
 const unrelated=f.addItem('other','Compendium.pf2e.feats-srd.Item.fake','electric-surge',{max:2,per:'PT10M',value:0});
 const daily=f.addItem('daily','Compendium.battlezoo-eldamon-pf2e.powers.Item.daily','electric-surge',{max:1,per:'day',value:0});
 const unprepared=f.addItem('u','Compendium.battlezoo-eldamon-pf2e.powers.Item.unprepared','electric-shot',{max:1,per:'PT10M',value:0});
 const result=await s.channel(f.payload,f.user);assert.equal(result.status,'armed');assert.equal(f.spent.system.frequency.value,1);assert.equal(f.reaction.system.frequency.value,1);assert.equal(unrelated.system.frequency.value,0);assert.equal(daily.system.frequency.value,0);assert.equal(unprepared.system.frequency.value,0);
 f.spent.system.frequency.value=0;await f.service().channel(f.payload,f.user);assert.equal(f.spent.system.frequency.value,0);assert.equal(f.actor.getRollOptions()[0],'active-power-refresh:high-voltage');
});

test('automatic outside-encounter Refresh uses the same idempotent resource writer and requires GM/source/no active encounter',async()=>{
 const f=fixture(),s=f.service(),p={actorUuid:f.actor.uuid,nonce:'end:fight'};
 assert.equal(typeof s.refreshOutsideEncounter,'function');
 f.addItem('feature',api.ELEMENTAL_POWERS_SOURCE,'elemental-powers');
 await assert.rejects(s.refreshOutsideEncounter(p,f.gm),/encounter/i);
 f.game.combat=null;await assert.rejects(s.refreshOutsideEncounter(p,f.user),/GM/i);
 await s.refreshOutsideEncounter(p,f.gm);assert.equal(f.spent.system.frequency.value,1);assert.equal(f.reaction.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.service().refreshOutsideEncounter(p,f.gm);assert.equal(f.spent.system.frequency.value,0);
 f.game.combats=new Map([['other',{started:true,combatants:[{actor:f.actor}]}]]);await assert.rejects(s.refreshOutsideEncounter({...p,nonce:'new'},f.gm),/encounter/i);
 f.game.combats.clear();f.actor.items.delete('feature');await assert.rejects(s.refreshOutsideEncounter({...p,nonce:'new'},f.gm),/Elemental Powers/i);
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
 const f=fixture(),s=f.service();await assert.rejects(s.channel({...f.payload,messageUuid:'ChatMessage.other'},f.user),/original|binding|card/i);
 await assert.rejects(s.channel(f.payload,{id:'stranger'}),/owner|permission/i);
 f.actor.getRollOptions=()=>[];await assert.rejects(s.channel(f.payload,f.user),/prepared/i);
 f.game.user=f.user;await assert.rejects(s.channel(f.payload,f.user),/GM/i);
});
test('one durable claim wins concurrent touches and remains consumed after zero or cancelled child roll',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 const p={...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true};
 const claims=await Promise.all([s.claim(p,f.user),s.claim(p,f.user)]);assert.equal(claims.filter(Boolean).length,1);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'claimed');
 await s.settle({...f.payload,status:'cancelled'},f.gm);assert.equal(await f.service().claim(p,f.user),null);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'cancelled');
});
test('touch requires explicit fact confirmation; caster touching another creature does not trigger itself',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 await assert.rejects(s.claim({...f.payload,targetUuid:f.target.uuid,kind:'touch'},f.user),/confirm/i);
 await assert.rejects(s.claim({...f.payload,targetUuid:f.origin.uuid,kind:'touch',confirmed:true},f.user),/other|creature|source/i);
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
test('attack eligibility has independent adjacent, unarmed and known-metal branches and never infers a hit from damage',()=>{
 assert.equal(typeof api.voltageAttackEligibility,'function');const facts={outcome:'success',melee:true,adjacent:true,unarmed:false,metal:false};
 assert.equal(api.voltageAttackEligibility(facts),true);assert.equal(api.voltageAttackEligibility({...facts,adjacent:false,unarmed:true}),true);assert.equal(api.voltageAttackEligibility({...facts,adjacent:false,metal:true}),true);
 assert.equal(api.voltageAttackEligibility({...facts,adjacent:false,metal:undefined}),false);assert.equal(api.voltageAttackEligibility({...facts,outcome:'failure'}),false);assert.equal(api.voltageAttackEligibility({...facts,melee:false}),false);
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
 await assert.rejects(f.service().refreshActivity(f.payload,f.user),/activity/i);
 f.message.flags[ID].voltageRefreshActivity={actions:2};await f.service().refreshActivity(f.payload,f.user);assert.equal(f.spent.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.service().refreshActivity(f.payload,f.user);assert.equal(f.spent.system.frequency.value,0);assert.equal(f.actor.flags[ID].voltage.activeNonce,null);
});
test('interrupted Refresh resumes original per-item writes without refilling a subsequently used power',async()=>{
 const f=fixture(),native=f.reaction.update.bind(f.reaction);let fail=true;f.reaction.update=async data=>{if(fail){fail=false;throw Error('network failure')}return native(data)};
 await assert.rejects(f.service().channel(f.payload,f.user),/network/);assert.equal(f.spent.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.service().channel(f.payload,f.user);assert.equal(f.spent.system.frequency.value,0);assert.equal(f.reaction.system.frequency.value,1);
});

let executorApi={};try{executorApi=await import('../scripts/eldamon-voltage-executor.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
function executorFixture(outcome='failure',{siphon=false,traits=[]}={}){
 const f=fixture(),messages=[],saves=[],applications=[],hooks=new Map(),routes=new Map(),rollContexts=new WeakMap();
 const targetUser={id:'target-owner',active:true};f.game.users.set(targetUser.id,targetUser);f.actor.getStatistic=()=>({dc:{value:22}});
 f.target.actor.testUserPermission=u=>u===f.gm||u===targetUser;f.target.actor.traits=new Set(traits);
 f.target.actor.getSelfRollOptions=()=>['self:level:5'];f.target.actor.getContextualClone=()=>({...f.target.actor,async applyDamage(params){applications.push(params);await provider.beforeDamage(this,params);return this}});
 f.item.getOriginData=()=>({actor:f.actor.uuid,uuid:f.item.uuid,type:'feat',rollOptions:[]});
 const fire=(name,...args)=>Promise.all((hooks.get(name)??[]).map(fn=>fn(...args)));
 const publish=async data=>{const m={...data,author:f.game.user,timestamp:300,isCheckRoll:data.flags?.pf2e?.context?.type==='saving-throw',isDamageRoll:data.flags?.pf2e?.context?.type==='damage-roll',item:f.item,actor:f.actor};f.game.messages.set(m.id,m);f.docs.set(m.uuid,m);await fire('createChatMessage',m,{},f.game.user.id);return m};
 f.message.update=async data=>{for(const[k,v]of Object.entries(data)){if(k.startsWith('flags.' ))f.message.flags[ID][k.slice(('flags.'+ID+'.').length)]=v}};
 f.target.actor.getStatistic=()=>({check:{async roll(params){saves.push({user:f.game.user,params});if(outcome===null)return null;const card=await publish({id:'save',uuid:'ChatMessage.save',speaker:{actor:'b',scene:'scene',token:'target'},rolls:[{_evaluated:true,total:18}],flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:f.item.uuid},context:{type:'saving-throw',outcome,dc:params.dc,options:params.extraRollOptions}}}});params.callback(card.rolls[0],outcome,card);return card.rolls[0]}}});
 class DamageRoll {
  constructor(formula,_data={},options={}){this.formula=formula;this.options=options;this.total=21;this._evaluated=false;this.type='electricity'}
  async evaluate(){this._evaluated=true;return this}
  alter(n,addend=0){const r=new DamageRoll(this.formula,{},{});r.total=Math.floor(this.total*n)+addend;r._evaluated=true;r.type=this.type;if(rollContexts.has(this))rollContexts.set(r,rollContexts.get(this));return r}
  async toMessage(data){const m=await publish({...data,id:'damage',uuid:'ChatMessage.damage',rolls:[this]});messages.push(m);rollContexts.set(this,{messageId:m.id,rollIndex:0});return m}
 }
 if(siphon)f.receipt.snapshot={kind:'siphoning',siphon:{applies:true},level:5,itemUuid:f.item.uuid,actorUuid:f.actor.uuid,powerSourceUuid:HV,disruptive:true,associatedTraits:['electricity']};
 const provider=executorApi.createEldamonVoltageProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),DamageRoll,getRollContext:roll=>rollContexts.get(roll),convertRoll:(roll,options)=>{assert.equal(options.rejectMixedPartitions,true);roll.type='untyped';return roll}});
 const socket={register:(name,fn)=>routes.set(name,fn),async executeAsUser(name,_gm,payload){const caller=f.game.user;f.game.user=f.gm;try{return await routes.get(name).call({socketdata:{userId:caller.id}},payload)}finally{f.game.user=caller}}};
 provider.register({socket,Hooks:{on:(name,fn)=>hooks.set(name,[...hooks.get(name)??[],fn])}});
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
 return {...f,provider,messages,saves,applications,hooks,routes,fire,activation,channel,trigger,save,damage,apply,targetUser,publish,DamageRoll};
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
  await assert.rejects(f.apply(),/consumed|settled|awaiting|claim/i);
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
 await assert.rejects(f.provider.rollSave(f.activation()),/owner|permission/i);assert.equal(f.saves.length,0);
 await f.save();f.game.user=f.targetUser;await assert.rejects(f.provider.rollDamage(f.activation()),/owner|permission/i);
 await f.damage();await assert.rejects(f.apply({token:f.origin}),/target|bound|recipient/i);assert.equal(f.activation().native.phase,'awaiting-application');
 await f.apply();
});

test('cancelled native save consumes the trigger and cannot be retried',async()=>{
 const f=executorFixture(null);await f.channel();await f.trigger();await f.save();
 assert.equal(f.activation().status,'cancelled');assert.equal(f.messages.length,0);f.game.user=f.gm;assert.equal(await f.trigger(),null);
 await assert.rejects(f.save(),/consumed|claim|awaiting/i);
});

test('application without its native receipt stays uncertain and never replays HP',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();await f.apply({nativeReceipt:false,nativeError:true});
 assert.equal(f.activation().status,'uncertain');await assert.rejects(f.apply(),/consumed|claim|awaiting/i);assert.equal(f.applications.length,1);
});

test('changed original save or damage source cannot authorize a native application',async()=>{
 for(const mutate of [f=>f.docs.get('ChatMessage.save').flags.pf2e.context.outcome='criticalFailure',f=>f.messages[0].rolls[0].total=999,f=>f.actor.items.delete(f.item.id)]){
  const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();mutate(f);
  await assert.rejects(f.apply(),/changed|invalid|source|binding/i);assert.equal(f.applications.length,0);
 }
});

test('legacy claimed and consumed Voltage records never acquire resumable native actions',async()=>{
 for(const status of ['claimed','done','uncertain','cancelled']){
  const f=executorFixture();await f.channel();await f.provider.ledger.claim({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user);
  f.actor.flags[ID].voltage.activations.channel.status=status;await assert.rejects(f.save(),/legacy|reconcil|consumed|claim/i);
  assert.equal(f.saves.length,0);assert.equal(f.activation().status,status);
 }
});

test('foreign rolls and voltage options without a real tracked damage card are rejected',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();f.game.user=f.targetUser;
 await assert.rejects(f.provider.beforeDamage(f.target.actor,{damage:{options:{},total:21},item:f.item,token:f.target,rollOptions:new Set(f.messages[0].flags.pf2e.context.options)}),/native|tracked|source|authorized/i);
 assert.equal(f.activation().native.phase,'awaiting-application');
});

test('registration authenticates the socket requester while committed channel preserves original author and refresh once',async()=>{
 const f=executorFixture();await f.channel();assert.equal(f.activation().userId,f.user.id);assert.equal(f.spent.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.channel();assert.equal(f.spent.system.frequency.value,0);
 const denied=await f.routes.get('voltage:trigger').call({socketdata:{userId:'stranger'}},{...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true});
 assert.equal(denied.ok,false);assert.equal(f.activation().status,'armed');
});

test('the source owner can explicitly confirm an authentic native unarmed hit on the original card',async()=>{
 const f=executorFixture();await f.channel();
 const attack=await f.publish({id:'attack',uuid:'ChatMessage.attack',isCheckRoll:true,rolls:[{_evaluated:true}],timestamp:200,speaker:{scene:'scene',token:'target',actor:'b'},flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:['item:melee']}}}});
 attack.isCheckRoll=true;attack.item={actor:f.target.actor,isMelee:true,system:{category:'unarmed'}};
 await assert.rejects(f.provider.trigger({...f.payload,kind:'attack',attackUuid:attack.uuid},f.user),/confirm/i);
 await f.provider.trigger({...f.payload,kind:'attack',attackUuid:attack.uuid,confirmed:true},f.user);
 assert.equal(f.activation().native.phase,'awaiting-save');assert.equal(f.saves.length,0);
});

test('a changed original High Voltage card cannot claim a new native response',async()=>{
 const f=executorFixture();await f.channel();f.message.flags[ID].metapowerUse.itemUuid='Actor.other.Item.copy';
 await assert.rejects(f.trigger(),/original|binding|card/i);assert.equal(f.activation().status,'armed');
});

test('only the bound target owner sees the native save control on the original card',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();f.game.scenes=new Map([[f.origin.parent.id,f.origin.parent]]);
 const render=user=>{f.game.user=user;const buttons=[],root={querySelector:()=>null,querySelectorAll:()=>[],append(){},ownerDocument:{createElement:tag=>({dataset:{},set textContent(text){this.text=text},append(node){if(tag==='div')buttons.push(node.text)},addEventListener(){}})}};f.provider.renderCard(f.message,root);return buttons};
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
  await assert.rejects(native({message:f.messages[0],multiplier:1}),/consumed|claim|awaiting/i);assert.equal(f.applications.length,1);
 }
});

test('an error after the authentic native receipt settles once without replaying damage',async()=>{
 const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();await f.apply({nativeError:true});
 assert.equal(f.activation().status,'done');await assert.rejects(f.apply(),/consumed|claim|awaiting/i);assert.equal(f.applications.length,1);
});

test('a native application with a wrong-user or reverted receipt cannot settle the Voltage claim',async()=>{
 for(const mutate of [m=>m.author={id:'stranger'},m=>m.flags.pf2e.appliedDamage.isReverted=true,m=>m.speaker.token='other']){
  const f=executorFixture();await f.channel();await f.trigger();await f.save();await f.damage();f.game.user=f.gm;
  const application=await f.provider.ledger.beginDamage({...f.payload,operationId:'native',damageUuid:'ChatMessage.damage',targetUuid:f.target.uuid,targetActorUuid:f.target.actor.uuid,rollIndex:0},f.targetUser);
  const receipt={id:'taken',uuid:'ChatMessage.taken',author:f.targetUser,speaker:{actor:'b',scene:'scene',token:'target'},flags:{pf2e:{origin:f.item.getOriginData(),context:{type:'damage-taken',options:[api.voltageRollOption(application),api.VOLTAGE_APPLY_PREFIX+'native']},appliedDamage:{uuid:f.target.actor.uuid,isHealing:false,isReverted:false}}}};
  mutate(receipt);f.game.messages.set(receipt.id,receipt);f.docs.set(receipt.uuid,receipt);
  await assert.rejects(f.provider.ledger.finishDamage({...f.payload,operationId:'native',receiptUuid:receipt.uuid},f.targetUser),/authentic|receipt/i);assert.equal(f.activation().native.phase,'applying');
  await assert.rejects(f.provider.ledger.beginDamage({...f.payload,operationId:'again',damageUuid:'ChatMessage.damage',targetUuid:f.target.uuid,targetActorUuid:f.target.actor.uuid,rollIndex:0},f.targetUser),/consumed|awaiting/i);
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
 await assert.rejects(f.provider.ledger.settle({...f.payload,status:'done'},f.gm),/native|receipt/i);assert.equal(f.activation().status,'claimed');
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
 await assert.rejects(f.damage(),/changed|reconcile/i);assert.equal(f.messages.length,0);assert.equal(f.activation().native.phase,'awaiting-damage');
});

test('a native failed attack rerolled to a hit keeps its original activation provenance and triggers once',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);
 const attack={isCheckRoll:true,rolls:[{_evaluated:true}],id:'miss',uuid:'ChatMessage.miss',timestamp:200,author:f.gm,speaker:{scene:'scene',token:'target',actor:'b'},item:{uuid:'Actor.b.Item.weapon',actor:f.target.actor,isMelee:true,type:'weapon',system:{}},flags:{pf2e:{context:{type:'attack-roll',outcome:'failure',target:{actor:f.actor.uuid,token:f.origin.uuid},options:[]}}},async update(data){this.flags.pf2e.context.options=data['flags.pf2e.context.options']}};
 f.docs.set(attack.uuid,attack);f.game.messages.set(attack.id,attack);assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:attack.uuid},f.gm),null);
 const reroll={...attack,id:'reroll',uuid:'ChatMessage.reroll',timestamp:210,flags:structuredClone(attack.flags)};reroll.flags.pf2e.context.isReroll=true;reroll.flags.pf2e.context.outcome='success';reroll.flags.pf2e.context.options.push('check:reroll');
 f.game.messages.delete(attack.id);f.docs.delete(attack.uuid);f.docs.set(reroll.uuid,reroll);f.game.messages.set(reroll.id,reroll);
 assert.ok(await f.service().claim({...f.payload,kind:'attack',attackUuid:reroll.uuid},f.gm));assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:reroll.uuid},f.gm),null);
});
test('unknown metal at reach requires explicit confirmation bound to a real native melee hit',async()=>{
 const f=fixture(),s=f.service();await s.channel(f.payload,f.user);f.origin.object.distanceTo=()=>10;
 const attack={isCheckRoll:true,rolls:[{_evaluated:true}],id:'reach',uuid:'ChatMessage.reach',timestamp:200,author:f.gm,speaker:{scene:'scene',token:'target',actor:'b'},item:{uuid:'Actor.b.Item.weapon',actor:f.target.actor,isMelee:true,type:'weapon',system:{}},flags:{pf2e:{context:{type:'attack-roll',outcome:'success',target:{actor:f.actor.uuid,token:f.origin.uuid},options:[]}}},async update(data){this.flags.pf2e.context.options=data['flags.pf2e.context.options']}};f.docs.set(attack.uuid,attack);f.game.messages.set(attack.id,attack);
 assert.equal(await s.claim({...f.payload,kind:'attack',attackUuid:attack.uuid},f.gm),null);
 await assert.rejects(s.claim({...f.payload,kind:'metal-hit',attackUuid:attack.uuid},f.user),/confirm/i);
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
  await provider.useRefresh(f.actor,{});assert.equal(created.length,1);assert.equal(created[0].flags[ID].voltageRefreshActivity.actions,2);assert.equal(created[0].flags.pf2e.origin.uuid,feature.uuid);assert.match(created[0].content,/Refresh/);assert.equal(feature.system.actionType,undefined);
 }finally{globalThis.CONFIG=savedConfig;}
});
