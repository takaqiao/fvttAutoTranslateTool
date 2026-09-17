import test from 'node:test';
import assert from 'node:assert/strict';
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
function executorFixture(outcome,{siphon=false,disruptive=false,traits=[]}={}){
 const f=fixture(),applications=[],messages=[];f.actor.getStatistic=()=>({dc:{value:22}});
 f.item.getOriginData=()=>({actor:f.actor.uuid,uuid:f.item.uuid,type:'feat',rollOptions:[]});
 f.target.actor.getSelfRollOptions=()=>['self:level:5'];f.target.actor.traits=new Set(traits);
 f.target.actor.getContextualClone=(options,effects)=>({...f.target.actor,async applyDamage(params){applications.push({params,options,effects,claimed:f.actor.flags[ID].voltage.activations.channel.status});await provider.beforeDamage(this,params);return this}});
 f.target.actor.getStatistic=()=>({check:{async roll(params){if(outcome===null)return null;const card={id:'save',uuid:'ChatMessage.save',speaker:{actor:'b',scene:'scene',token:'target'},flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:f.item.uuid},context:{type:'saving-throw',outcome,options:params.extraRollOptions}}}};f.game.messages.set(card.id,card);f.docs.set(card.uuid,card);params.callback({_evaluated:true},outcome,card);return {};}}});
 class DamageRoll {
  constructor(formula,_data={},options={}){this.formula=formula;this.options=options;this.total=21;this._evaluated=false;this.type='electricity'}
  async evaluate(){this._evaluated=true;return this}
  alter(n){const r=new DamageRoll(this.formula,{},structuredClone(this.options));r.total=Math.floor(this.total*n);r._evaluated=true;r.type=this.type;return r}
  async toMessage(data){const m={...data,id:'damage',uuid:'ChatMessage.damage',rolls:[this]};messages.push(m);f.game.messages.set(m.id,m);f.docs.set(m.uuid,m);return m}
 }
 if(siphon)f.receipt.snapshot={kind:'siphoning',siphon:{applies:true},level:5,itemUuid:f.item.uuid,actorUuid:f.actor.uuid,powerSourceUuid:HV,disruptive,associatedTraits:['electricity']};
 const provider=executorApi.createEldamonVoltageProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),DamageRoll,convertRoll:(roll,options)=>{assert.equal(options.rejectMixedPartitions,true);roll.type='untyped';return roll}});
 return {...f,provider,applications,messages};
}
test('basic Reflex executor preserves native damage API source, target and contextual options for every outcome',async()=>{
 assert.equal(typeof executorApi.createEldamonVoltageProvider,'function');
 for(const [outcome,total]of [['criticalSuccess',0],['success',10],['failure',21],['criticalFailure',42]]){
  const f=executorFixture(outcome);await f.provider.ledger.channel(f.payload,f.user);
  await f.provider.trigger({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user);
  assert.equal(f.applications.length,1);const {params,claimed}=f.applications[0];assert.equal(claimed,'claimed');assert.equal(params.damage.total,total);assert.equal(params.item,f.item);assert.equal(params.token,f.target);assert.equal(params.skipIWR,false);assert.equal(params.outcome,outcome);assert.equal(f.messages[0].speaker.actor,'a');assert.equal(f.messages[0].flags.pf2e.origin.uuid,f.item.uuid);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'done');
  await assert.rejects(f.provider.beforeDamage(f.target.actor,params),/already|authorized|grant/i);
 }
});
test('Siphon delayed damage uses target traits for full versus half without duplicate metapower marker',async()=>{
 for(const [traits,want]of [[[],10],[['electricity'],21]]){
  const f=executorFixture('failure',{siphon:true,disruptive:true,traits});await f.provider.ledger.channel(f.payload,f.user);await f.provider.trigger({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user);
  assert.equal(f.applications[0].params.damage.total,want);assert.equal(f.applications[0].params.damage.type,'untyped');assert.equal(f.applications[0].params.damage.options[ID].metapowerDamage,undefined);assert.equal(f.spent.system.frequency.value,0);
 }
});
test('cancelled native child save consumes the claim without damage or a second trigger',async()=>{
 const f=executorFixture(null);await f.provider.ledger.channel(f.payload,f.user);await f.provider.trigger({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user);
 assert.equal(f.applications.length,0);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'cancelled');assert.equal(await f.provider.trigger({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user),null);
});
test('registration binds active-GM socket calls to their real requester and refreshes the original card after arming',async()=>{
 const f=executorFixture('success'),routes=new Map(),hooks=new Map(),updates=[];f.message.update=async data=>updates.push(data);
 const socket={register:(name,fn)=>routes.set(name,fn),async executeAsUser(name,_gm,payload){return routes.get(name).call({socketdata:{userId:f.user.id}},payload)}};
 f.provider.register({socket,Hooks:{on:(name,fn)=>hooks.set(name,fn)}});
 await f.provider.onCommittedChannel({receipt:f.receipt,message:f.message});assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'armed');assert.ok(updates.some(u=>u[`flags.${ID}.voltageStatus`]==='armed'));
 const denied=await routes.get('voltage:trigger').call({socketdata:{userId:'stranger'}},{...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true});assert.equal(denied.ok,false);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'armed');
 assert.equal(typeof hooks.get('createChatMessage'),'function');assert.equal(typeof hooks.get('renderChatMessageHTML'),'function');assert.equal(typeof hooks.get('pf2e.startTurn'),'function');
});
test('durable GM channel delivery preserves original player identity without an initiating client',async()=>{
 const f=executorFixture('success');
 await f.provider.onCommittedChannel({receipt:f.receipt,message:f.message,user:f.user});
 assert.equal(f.actor.flags[ID].voltage.activations.channel.userId,f.user.id);assert.equal(f.spent.system.frequency.value,1);
 f.spent.system.frequency.value=0;await f.provider.onCommittedChannel({receipt:f.receipt,message:f.message,user:f.user});assert.equal(f.spent.system.frequency.value,0);
});
test('damage card replay with only native context option is blocked even after native roll alteration loses private metadata',async()=>{
 const f=executorFixture('failure');await assert.rejects(f.provider.beforeDamage(f.target.actor,{damage:{options:{}},rollOptions:new Set([ID+':voltage:channel'])}),/authorized|grant/i);
});
test('source deletion during the native save consumes the trigger without applying damage',async()=>{
 const f=executorFixture('failure'),stat=f.target.actor.getStatistic();f.target.actor.getStatistic=()=>({check:{async roll(params){await stat.check.roll(params);f.actor.items.delete(f.item.id)}}});
 await f.provider.ledger.channel(f.payload,f.user);await assert.rejects(f.provider.trigger({...f.payload,targetUuid:f.target.uuid,kind:'touch',confirmed:true},f.user),/changed/i);assert.equal(f.applications.length,0);assert.equal(f.actor.flags[ID].voltage.activations.channel.status,'uncertain');
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
