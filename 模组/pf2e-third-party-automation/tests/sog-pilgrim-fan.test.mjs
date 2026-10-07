import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createPilgrimFan} from '../scripts/sog-pilgrim-fan.mjs';

const M='pf2e-third-party-automation';
const clone=value=>structuredClone(value);
function apply(document,changes){
 for(const [path,value] of Object.entries(changes)){
  const parts=path.split('.');let at=document;
  for(const key of parts.slice(0,-1))at=at[key]??={};
  at[parts.at(-1)]=clone(value);
 }
 return document;
}
function fixture(){
 const user={id:'gm',active:true,isGM:true},users=new Map([['gm',user]]);users.activeGM=user;
 const actor={id:'self',uuid:'Actor.self',type:'character',isDead:false,canSee:true,items:new Map(),isOfType(...types){return types.includes(this.type)},getActiveTokens(){return [origin.object]},getCondition(){return null}};
 const item={id:'fan',uuid:'Actor.self.Item.fan',type:'weapon',actor,flags:{world:{sogWontonNativeAutomation:{key:'spirit-fan'},sogB2Ch2Resources:{originalSlug:'spirit-fan',fileSha256:'a'.repeat(64)}},[M]:{sogPilgrim:{use:{nonce:'provider'}}}},system:{equipped:{carryType:'held',handsHeld:1}},async update(changes){apply(this,changes);return this}};actor.items.set(item.id,item);
 const scene={id:'s',uuid:'Scene.s',tokens:new Map()},origin={id:'o',uuid:'Scene.s.Token.o',documentName:'Token',parent:scene,actor,hidden:false},targetActor={id:'other',uuid:'Actor.other',type:'npc',isDead:false,traits:new Set(),isOfType(...types){return types.includes(this.type)}},target={id:'t',uuid:'Scene.s.Token.t',documentName:'Token',parent:scene,actor:targetActor,hidden:false};scene.tokens.set('o',origin);scene.tokens.set('t',target);
 origin.object={document:origin,center:{x:0,y:0},distanceTo:()=>20,checkCollision:()=>false};target.object={document:target,center:{x:100,y:0},isVisible:true};
 const game={user,users,actors:new Map([['self',actor],['other',targetActor]]),scenes:new Map([['s',scene]]),messages:new Map(),time:{worldTime:100},combat:null};
 const templates=new Map(['leaves','light'].map((kind,i)=>[`Item.${i?'53ad48c5ad472dcf':'e3f4b1e0c69e0cba'}`,{type:'effect',name:kind,system:{duration:{value:kind==='leaves'?-1:1,unit:kind==='leaves'?'unlimited':'minutes',expiry:kind==='leaves'?null:'turn-start',sustained:false},start:{value:0,initiative:null},badge:kind==='leaves'?{type:'counter',value:0,min:0,max:3}:null,rules:kind==='light'?[{key:'TokenLight',value:{bright:20,dim:40}}]:[],description:{value:'',gm:''},context:null},flags:{},toObject(){const {toObject,...data}=this;return clone(data)}}]));
 const docs=new Map([actor,item,origin,target,targetActor,...templates.values()].filter(d=>d.uuid).map(d=>[d.uuid,d]));
 for(const [uuid,template]of templates)docs.set(uuid,template);
 const mediaCalls=[],errors=[],writes=[];let seq=0;
 actor.createEmbeddedDocuments=async(_type,data)=>data.map(source=>{
  const effect={...clone(source),id:`e${++seq}`,uuid:`Actor.self.Item.e${seq}`,actor,async update(changes){writes.push(changes);apply(this,changes);return this}};actor.items.set(effect.id,effect);return effect;
 });
 actor.deleteEmbeddedDocuments=async(_type,ids)=>{for(const id of ids)actor.items.delete(id);return []};
 const nativeRemaining=(data,parent)=>{
  assert.equal(parent,actor,'Expiry is bound to the fan wielder');
  const remaining=data.system.start.value+60-game.time.worldTime,current=game.combat?.combatant;
  return {remaining,expired:remaining<0||remaining===0&&(!current?.actor||current.actor===parent&&(data.system.start.initiative??0)===(current.initiative??0))};
 };
 const options={game,fromUuid:async uuid=>docs.get(uuid),nativeRemaining,media:{async leaves(_item,count){mediaCalls.push(['leaves',count])},async release(){mediaCalls.push(['release'])}},onError:error=>errors.push(error)};
 const fan=createPilgrimFan(options),state=()=>item.flags[M].sogPilgrim.fan,effects=kind=>[...actor.items.values()].filter(e=>e.flags?.[M]?.sogPilgrim?.kind===kind);
 function message(id='hit',overrides={}){
  const message={id,uuid:`ChatMessage.${id}`,actor,item,isCheckRoll:true,flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid},context:{type:'attack-roll',outcome:'success',target:{actor:targetActor.uuid,token:target.uuid},dc:{value:20}}}},rolls:[{_evaluated:true,total:22,options:{degreeOfSuccess:2}}],...overrides};game.messages.set(id,message);return message;
 }
 async function charge(){targetActor.traits.add('undead');await fan.handleStrike(message());targetActor.traits.clear();}
 const card=async data=>{const saved={id:'release',uuid:'ChatMessage.release',...data};game.messages.set(saved.id,saved);return saved};
 return {game,user,actor,item,scene,origin,target,targetActor,fan,options,message,state,effects,mediaCalls,errors,writes,charge,card};
}

test('a saved native success adds one timed leaf and preserves provider state',async()=>{
 const f=fixture();await f.fan.handleStrike(f.message());
 assert.deepEqual(f.state(),{leaves:[{messageId:'hit',start:{value:100,initiative:null}}],hitIds:['hit']});
 assert.deepEqual(f.item.flags[M].sogPilgrim.use,{nonce:'provider'});
 assert.equal(f.effects('leaves')[0].system.badge.value,1);
 assert.equal(f.effects('light')[0].system.rules[0].value.bright,20);
 assert.deepEqual(f.effects('leaves')[0].system.duration,{value:1,unit:'minutes',expiry:'turn-start',sustained:false});
});

test('ordinary later hits never refresh existing leaves and the count stops at three',async()=>{
 const f=fixture();await f.fan.handleStrike(f.message('one'));f.game.time.worldTime=115;await f.fan.handleStrike(f.message('two'));f.game.time.worldTime=120;await f.fan.handleStrike(f.message('three'));f.game.time.worldTime=130;await f.fan.handleStrike(f.message('full'));
 assert.deepEqual(f.state().leaves.map(l=>l.start.value),[100,115,120]);assert.equal(f.state().hitIds.includes('full'),true);
 for(const kind of ['leaves','light'])assert.deepEqual(f.effects(kind)[0].system.start,{value:120,initiative:null});
});

test('an undead hit fills empty leaves without refreshing already lit leaves',async()=>{
 const f=fixture();await f.fan.handleStrike(f.message('one'));f.game.time.worldTime=110;f.targetActor.traits.add('undead');await f.fan.handleStrike(f.message('undead'));
 assert.deepEqual(f.state().leaves.map(l=>l.start.value),[100,110,110]);
 f.game.time.worldTime=120;await f.fan.handleStrike(f.message('full'));assert.deepEqual(f.state().leaves.map(l=>l.start.value),[100,110,110]);
});

test('negative-healing alone does not count as the undead trait',async()=>{
 const f=fixture();f.targetActor.traits.add('negative-healing');await f.fan.handleStrike(f.message());assert.equal(f.state().leaves.length,1);
});

test('a destroyed undead corpse cannot charge the fan',async()=>{
 const f=fixture();f.targetActor.traits.add('undead');f.targetActor.isDead=true;
 await f.fan.handleStrike(f.message());assert.equal(f.state(),undefined);
});

test('zero HP alone does not exclude a creature from charging the fan',async()=>{
 const f=fixture();f.targetActor.system={attributes:{hp:{value:0}}};f.targetActor.isDead=false;
 await f.fan.handleStrike(f.message());assert.equal(f.state().leaves.length,1);
});

test('concurrent redraw and a reconnected service never count the same hit again',async()=>{
 const f=fixture(),message=f.message();await Promise.all([f.fan.handleStrike(message),f.fan.handleStrike(message)]);const reconnected=createPilgrimFan(f.options);await reconnected.handleStrike(message);
 assert.equal(f.state().leaves.length,1);f.game.time.worldTime=170;await reconnected.reconcile(f.actor);await reconnected.handleStrike(message);assert.equal(f.state().leaves.length,0);
});

test('a replaced GM cannot count a queued strike',async()=>{
 const f=fixture();f.game.users.activeGM={id:'replacement',active:true,isGM:true};await f.fan.handleStrike(f.message());assert.equal(f.state(),undefined);
});

test('shared-rune eidolon and detached fan sources do not charge the real wielder',async()=>{
 for(const kind of ['eidolon','detached','non-character']){
  const f=fixture(),message=f.message();
  if(kind==='eidolon')message.actor={id:'eidolon',uuid:'Actor.eidolon',type:'character',items:new Map()};
  if(kind==='detached')message.item={...f.item};
  if(kind==='non-character')f.actor.type='npc';
  await f.fan.handleStrike(message);assert.equal(f.state(),undefined,kind);
 }
});

test('a character actor bearing the eidolon trait cannot charge copied fan data',async()=>{
 const f=fixture();f.actor.traits=new Set(['eidolon']);await f.fan.handleStrike(f.message());assert.equal(f.state(),undefined);
});

test('unheld, unproven, rerolled, failed and malformed attacks never add leaves',async()=>{
 const changes=[
  f=>f.item.system.equipped.handsHeld=0,
  f=>f.game.messages.clear(),
  (_f,m)=>m.isCheckRoll=false,
  (_f,m)=>m.flags.pf2e.context.type='damage-roll',
  (_f,m)=>m.flags.pf2e.context.outcome='failure',
  (_f,m)=>m.flags.pf2e.context.isReroll=true,
  (_f,m)=>m.rolls[0].options.degreeOfSuccess=1,
  (_f,m)=>m.rolls[0].total=-1,
  (_f,m)=>m.flags.pf2e.context.target.actor='Actor.unrelated',
  (_f,m)=>m.flags.pf2e.origin.uuid='Actor.self.Item.sword',
  f=>f.item.flags.world.sogB2Ch2Resources.fileSha256='invalid',
 ];
 for(const change of changes){const f=fixture(),message=f.message();change(f,message);await f.fan.handleStrike(message);assert.equal(f.state(),undefined);}
});

test('each leaf expires independently while lighting retains the latest valid original start',async()=>{
 const f=fixture();await f.fan.handleStrike(f.message('first'));f.game.time.worldTime=110;await f.fan.handleStrike(f.message('second'));f.game.time.worldTime=161;await f.fan.reconcile(f.actor);
 assert.deepEqual(f.state().leaves.map(l=>l.messageId),['second']);assert.equal(f.effects('leaves')[0].system.badge.value,1);assert.equal(f.effects('light')[0].system.start.value,110);
 f.game.time.worldTime=171;await f.fan.reconcile(f.actor);assert.equal(f.effects('light').length,0);assert.equal(f.effects('leaves').length,0);assert.equal(f.state().leaves.length,0);assert.deepEqual(f.mediaCalls.at(-1),['leaves',0]);
});

test('zero seconds honours native initiative and owning-actor turn-start semantics',async()=>{
 const f=fixture();f.game.combat={combatant:{initiative:22,actor:f.actor}};await f.fan.handleStrike(f.message());assert.equal(f.state().leaves[0].start.initiative,22);
 f.game.time.worldTime=160;f.game.combat.combatant={initiative:22,actor:f.targetActor};await f.fan.reconcile(f.actor);assert.equal(f.state().leaves.length,1);
 f.game.combat.combatant={initiative:19,actor:f.actor};await f.fan.reconcile(f.actor);assert.equal(f.state().leaves.length,1);
 f.game.combat.combatant.initiative=22;await f.fan.reconcile(f.actor);assert.equal(f.state().leaves.length,0);
});

test('only effects owned by this physical fan are removed when it is deleted',async()=>{
 const f=fixture();await f.charge();f.actor.items.set('other',{id:'other',type:'effect',flags:{[M]:{sogPilgrim:{kind:'light',source:'Actor.self.Item.other-fan'}}}});f.actor.items.delete(f.item.id);await f.fan.reconcile(f.actor);
 assert.equal(f.effects('leaves').length,0);assert.equal(f.effects('light').length,1);assert.equal(f.actor.items.has('other'),true);assert.deepEqual(f.mediaCalls.at(-1),['leaves',0]);
});

test('deleting a fan and its effects together still clears the known leaf display',async()=>{
 const f=fixture();await f.charge();for(const effect of [...f.effects('leaves'),...f.effects('light')])f.actor.items.delete(effect.id);f.actor.items.delete(f.item.id);await f.fan.reconcile(f.actor);assert.deepEqual(f.mediaCalls.at(-1),['leaves',0]);
});

test('cancelled native damage selection preserves all leaves and lighting',async()=>{
 const f=fixture();await f.charge();const before=clone(f.state());assert.equal(await f.fan.release({item:f.item,target:f.target,nonce:'cancel',card:async()=>null}),null);assert.deepEqual(f.state(),before);assert.equal(f.effects('light').length,1);
});

test('living release publishes native healing and only then clears leaves without applying HP',async()=>{
 const f=fixture();await f.charge();let formula,label;
 const result=await f.fan.release({item:f.item,target:f.target,nonce:'use',card:async data=>{formula=data.formula;label=data.label;assert.equal(f.state().leaves.length,3);return f.card(data)}});
 assert.deepEqual(result,{messageId:'release'});assert.equal(formula,'(3d8+8)[healing]');assert.match(label,/灵魂连携/);assert.equal(f.state().leaves.length,0);assert.equal(f.effects('light').length,0);assert.deepEqual(f.item.flags[M].sogPilgrim.use,{nonce:'provider'});
});

test('undead release keeps vitality damage and its basic Fortitude save on the native card',async()=>{
 const f=fixture();await f.charge();f.targetActor.traits.add('undead');await f.fan.release({item:f.item,target:f.target,nonce:'use',card:f.card});const message=f.game.messages.get('release');assert.equal(message.formula,'(3d8+8)[vitality]');assert.deepEqual(message.save,{type:'fortitude',dc:23,basic:true});
});

test('no release occurs with construct, dead, noncreature or out-of-range targets',async()=>{
 for(const kind of ['construct','dead','hazard','range','sight','hidden','blind','detached']){
  const f=fixture();await f.charge();let calls=0;
  if(kind==='construct')f.targetActor.traits.add('construct');
  if(kind==='dead')f.targetActor.isDead=true;
  if(kind==='hazard')f.targetActor.type='hazard';
  if(kind==='range')f.origin.object.distanceTo=()=>35;
  if(kind==='sight')f.origin.object.checkCollision=()=>true;
  if(kind==='hidden')f.target.hidden=true;
  if(kind==='blind')f.actor.canSee=false;
  if(kind==='detached')f.scene.tokens.delete(f.target.id);
  await assert.rejects(f.fan.release({item:f.item,target:f.target,nonce:'use',card:async()=>{calls++;return f.card({})}}),undefined,kind);
  assert.equal(calls,0,kind);assert.equal(f.state().leaves.length,3,kind);
 }
});

test('expired leaves are checked before a release card is requested',async()=>{
 const f=fixture();await f.charge();f.game.time.worldTime=161;let called=0;await assert.rejects(f.fan.release({item:f.item,target:f.target,nonce:'expired',card:async()=>{called++;return f.card({})}}));assert.equal(called,0);assert.equal(f.state().leaves.length,3);
});

test('restoring a native effect never gives old leaves a new start',async()=>{
 const f=fixture();await f.charge();f.game.time.worldTime=120;await f.actor.deleteEmbeddedDocuments('Item',f.effects('light').map(e=>e.id));const create=f.actor.createEmbeddedDocuments;
 f.actor.createEmbeddedDocuments=async(type,data)=>{const effects=await create(type,data);for(const effect of effects)effect.system.start={value:f.game.time.worldTime,initiative:99};return effects;};
 await f.fan.reconcile(f.actor);assert.deepEqual(f.effects('light')[0].system.start,{value:100,initiative:null});
});

test('validation immediately before publication refuses leaves expiring during a roll',async()=>{
 const f=fixture();await f.charge();await assert.rejects(f.fan.release({item:f.item,target:f.target,nonce:'late',card:async data=>{f.game.time.worldTime=161;await data.validate();return f.card(data)}}));
 assert.equal(f.game.messages.has('release'),false);assert.equal(f.state().leaves.length,3);
});

test('a saved release clears leaves even when time advances before its response arrives',async()=>{
 const f=fixture();await f.charge();await f.fan.release({item:f.item,target:f.target,nonce:'late-response',card:async data=>{const message=await f.card(data);f.game.time.worldTime=161;return message}});assert.equal(f.state().leaves.length,0);
});

test('recovering a persisted release clears the fan without changing its hit receipts',async()=>{
 const f=fixture();await f.charge();await f.fan.clearAfterRelease(f.item);assert.deepEqual(f.state(),{leaves:[],hitIds:['hit']});assert.equal(f.effects('light').length,0);
});

test('release uses the selected current source rather than another token of the same actor',async()=>{
 const f=fixture();await f.charge();const remote={id:'remote',uuid:'Scene.s.Token.remote',documentName:'Token',parent:f.scene,actor:f.actor,object:{...f.origin.object,distanceTo:()=>50}};f.scene.tokens.set(remote.id,remote);f.actor.getActiveTokens=()=>[remote,f.origin];
 await f.fan.release({item:f.item,source:f.origin,target:f.target,nonce:'selected',card:f.card});assert.equal(f.state().leaves.length,0);
});

test('release refuses an explicitly selected source from another actor',async()=>{
 const f=fixture();await f.charge();await assert.rejects(f.fan.release({item:f.item,source:f.target,target:f.target,nonce:'wrong-source',card:f.card}));assert.equal(f.game.messages.has('release'),false);
});

test('the native remainingDuration getter determines expiry on a temporary owned effect',async t=>{
 const prior=globalThis.CONFIG;t.after(()=>globalThis.CONFIG=prior);const f=fixture();
 globalThis.CONFIG={Item:{documentClass:class {
  constructor(data,{parent}){this.system=data.system;this.parent=parent}
  get remainingDuration(){assert.equal(this.parent,f.actor);assert.equal(this.system.context.origin.actor,f.actor.uuid);return {remaining:0,expired:false}}
 }}};
 const fan=createPilgrimFan({...f.options,nativeRemaining:undefined});await fan.handleStrike(f.message());f.game.time.worldTime=160;await fan.reconcile(f.actor);assert.equal(f.state().leaves.length,1);
});

test('a lost hit-write response cannot add the already persisted leaf twice',async()=>{
 const f=fixture(),update=f.item.update;let first=true;
 f.item.update=async changes=>{await update.call(f.item,changes);if(first){first=false;throw Error('response lost')}return f.item};
 const message=f.message();await assert.rejects(f.fan.handleStrike(message));await createPilgrimFan(f.options).handleStrike(message);assert.equal(f.state().leaves.length,1);assert.deepEqual(f.state().hitIds,['hit']);
});

test('an unpersisted release card cannot consume leaves',async()=>{
 const f=fixture();await f.charge();await assert.rejects(f.fan.release({item:f.item,target:f.target,nonce:'lost',card:async()=>({id:'unsaved'})}));assert.equal(f.state().leaves.length,3);
});

test('media failures are reported and never block counting or a successful release',async()=>{
 const f=fixture(),failure=Error('media unavailable');f.options.media={leaves:async()=>{throw failure},release:async()=>{throw failure}};const fan=createPilgrimFan(f.options);f.targetActor.traits.add('undead');await fan.handleStrike(f.message());f.targetActor.traits.clear();await fan.release({item:f.item,target:f.target,nonce:'use',card:f.card});assert.equal(f.state().leaves.length,0);assert.equal(f.errors.includes(failure),true);
});
