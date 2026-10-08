import test from 'node:test';
import assert from 'node:assert/strict';
import {createPilgrimMedia,PILGRIM_MEDIA} from '../scripts/sog-pilgrim-media.mjs';

const M='pf2e-third-party-automation';
const club='modules/jb2a_patreon/Library/Generic/Weapon_Attacks/Melee/Group02/MeleeAttack02_Club01_01_800x600.webm';
function fixture(t,{matchTrigger,playError,lookup,endAction}={}) {
 const played=[],ended=[],effects=[],stored=[],playOptions=[],errors=[],writes=[],queries=[];
 class Sequence {
  sections=[];
  section(kind){
   const data={kind};this.sections.push(data);
   const section={};
   for(const method of ['file','name','attachTo','atLocation','rotateTowards','scaleToObject','spriteOffset','opacity','persist','temporary','tieToDocuments','mask','fadeIn','fadeOut','shape','loopProperty','size','volume','duration','fadeOutAudio'])section[method]=(...args)=>{data[method]=args;return section;};
   return section;
  }
  effect(){return this.section('effect');}
  sound(){return this.section('sound');}
  async play(options){
   if(playError)throw playError;
   played.push(this.sections);playOptions.push(options);
   for(const section of this.sections)if(section.persist){
    const effect={...section,id:`visual${played.length}-${effects.length}`,sceneId:canvas.scene.id};effects.push(effect);
    if(!section.temporary)stored.push(effect);
   }
  }
 }
 // Sequencer 4.2.3 filters visible effects only, and ignores sceneId in _filterEffects.
 const matching=filter=>effects.filter(effect=>(!filter.name||effect.name?.[0]===filter.name)&&(!filter.effects||filter.effects.includes(effect.id)));
 const globals={Sequence,PIXI:{Circle:class Circle{constructor(x,y,radius){Object.assign(this,{x,y,radius});}}},Sequencer:{EffectManager:{
  getEffects(filter){return matching(filter).map(effect=>({id:effect.id,data:{name:effect.name?.[0],sceneId:effect.sceneId}}));},
  async endEffects(filter,push){
   ended.push({filter,push});
   for(const effect of matching(filter)){effects.splice(effects.indexOf(effect),1);const index=stored.indexOf(effect);if(index>=0)stored.splice(index,1);}
   await endAction?.();
  }
 }},triggerAnimations:matchTrigger?{api:{async matchTrigger(names){queries.push(names);return matchTrigger(names);}}}:undefined};
 for(const [key,value]of Object.entries(globals)){
  const descriptor=Object.getOwnPropertyDescriptor(globalThis,key);
  Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
  t.after(()=>{if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];});
 }
 const user={id:'gm'},users=new Map([['gm',user]]);users.activeGM=user;
 const actor={id:'hero',uuid:'Actor.hero',type:'character',items:new Map(),rollOptions:{all:{}}};
 const item={id:'branch',uuid:'Actor.hero.Item.branch',slug:'branch-of-the-great-sugi',type:'weapon',actor,system:{baseItem:'whip',group:'flail',equipped:{carryType:'held',handsHeld:1}},flags:{world:{sogWontonNativeAutomation:{key:'branch-of-the-great-sugi'},sogB2Ch2Resources:{originalSlug:'branch-of-the-great-sugi',fileSha256:'a'.repeat(64)}}}};
 actor.items.set(item.id,item);
 const targetActor={id:'foe',uuid:'Actor.foe',type:'npc'},scene={id:'battle',tokens:new Map(),grid:{size:100,distance:5}};
 const source={id:'hero',uuid:'Scene.battle.Token.hero',documentName:'Token',actor,parent:scene},target={id:'foe',uuid:'Scene.battle.Token.foe',documentName:'Token',actor:targetActor,parent:scene};
 for(const token of [source,target]){token.object={document:token};scene.tokens.set(token.id,token);}
 const canvas={ready:true,scene},game={user,users,modules:new Map([['sequencer',{active:true}]]),actors:new Map([[actor.id,actor],[targetActor.id,targetActor]]),messages:new Map(),scenes:new Map([[scene.id,scene]])};
 const documents=new Map([actor,item,source,target].map(doc=>[doc.uuid,doc]));
 const options={game,canvas,fromUuid:async uuid=>lookup?lookup(uuid,documents):documents.get(uuid),onError:error=>errors.push(error)};
 const media=createPilgrimMedia(options);
 function message(id='hit'){
  const message={id,uuid:`ChatMessage.${id}`,item,actor,isCheckRoll:true,speaker:{scene:scene.id,token:source.id},rolls:[{_evaluated:true,total:25,options:{degreeOfSuccess:2}}],flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid},context:{type:'attack-roll',outcome:'success',token:source.uuid,target:{token:target.uuid,actor:targetActor.uuid}}}},async update(change){
   writes.push(change);
   for(const [path,value]of Object.entries(change)){const parts=path.split('.');let data=this;for(const part of parts.slice(0,-1))data=data[part]??={};data[parts.at(-1)]=value;}
   return this;
  }};
  game.messages.set(id,message);return message;
 }
 const sections=kind=>played.flat().filter(section=>section.kind===kind);
 function enterScene(next){
  canvas.scene=next;effects.length=0;
  for(let index=stored.length-1;index>=0;index--){
   const effect=stored[index],tied=[effect.tieToDocuments?.[0]??[]].flat();
   if(tied.some(uuid=>!documents.has(uuid)))stored.splice(index,1);
   else if(effect.sceneId===next.id)effects.push(effect);
  }
 }
 function areaDocuments(nonce='storm'){
  const effect={id:nonce,uuid:`Actor.hero.Item.${nonce}`,type:'effect',actor};
  const template={id:nonce,uuid:`Scene.battle.MeasuredTemplate.${nonce}`,parent:scene};
  documents.set(effect.uuid,effect);documents.set(template.uuid,template);actor.items.set(effect.id,effect);
  return {kind:'storm',scene,position:{x:100,y:100},nonce,effect,template};
 }
 function fanEffect(count=2){
  item.flags.world.sogWontonNativeAutomation.key='spirit-fan';item.flags.world.sogB2Ch2Resources.originalSlug='spirit-fan';
  item.flags[M]={sogPilgrim:{fan:{leaves:Array.from({length:3},()=>({}))}}};
  const effect={id:'leaves',uuid:'Actor.hero.Item.leaves',type:'effect',actor,system:{badge:{type:'counter',value:count}},flags:{[M]:{sogPilgrim:{kind:'leaves',source:item.uuid}}},isExpired:false,remainingDuration:{expired:false}};
  actor.items.set(effect.id,effect);documents.set(effect.uuid,effect);return effect;
 }
 return {media,options,game,canvas,actor,item,source,target,scene,message,played,ended,effects,stored,playOptions,documents,enterScene,areaDocuments,fanEffect,errors,writes,queries,sections};
}

test('an active Trigger Animations attack leaves only the hit decoration',async t=>{
 const f=fixture(t,{matchTrigger:()=>({id:'configured-attack'})});f.item.system={...f.item.system,baseItem:'longsword',group:'sword'};
 await f.media.strike(f.message());
 assert.deepEqual(f.queries,['attack:Actor.hero.Item.branch,attack:branch-of-the-great-sugi,attack:longsword,attack:sword']);
 assert.deepEqual(f.sections('effect').map(section=>section.file[0]),[PILGRIM_MEDIA.leaves]);
 assert.equal(f.sections('sound').length,0);
});

test('the original branch adds a club attack when no provider matches',async t=>{
 const f=fixture(t,{matchTrigger:()=>undefined});await f.media.strike(f.message());
 assert.deepEqual(f.sections('effect').map(section=>section.file[0]),[club,PILGRIM_MEDIA.leaves]);
 assert.deepEqual(f.sections('sound').map(section=>section.file[0]),[PILGRIM_MEDIA.natureSound]);
});

test('only a real sword form receives the sword attack and sword sound',async t=>{
 const f=fixture(t);f.item.system.baseItem='longsword';f.item.system.group='sword';await f.media.strike(f.message());
 assert.deepEqual(f.sections('effect').map(section=>section.file[0]),[PILGRIM_MEDIA.sword,PILGRIM_MEDIA.leaves]);
 assert.deepEqual(f.sections('sound').map(section=>section.file[0]),[PILGRIM_MEDIA.swordSound]);
});

test('other branch forms receive leaves and a nature sound without a sword attack',async t=>{
 const f=fixture(t);f.item.system.baseItem='spear';f.item.system.group='spear';await f.media.strike(f.message());
 assert.deepEqual(f.sections('effect').map(section=>section.file[0]),[PILGRIM_MEDIA.leaves]);
 assert.deepEqual(f.sections('sound').map(section=>section.file[0]),[PILGRIM_MEDIA.natureSound]);
});

test('provider lookup errors suppress duplicate attacks but retain leaves',async t=>{
 const error=Error('trigger lookup unavailable'),f=fixture(t,{matchTrigger:()=>{throw error;}});await f.media.strike(f.message());
 assert.deepEqual(f.sections('effect').map(section=>section.file[0]),[PILGRIM_MEDIA.leaves]);
 assert.equal(f.sections('sound').length,0);assert.deepEqual(f.errors,[error]);
});

test('concurrent message hooks and a reconnected service play a hit once',async t=>{
 const f=fixture(t),message=f.message();
 await Promise.all([f.media.strike(message),f.media.strike(message)]);
 await createPilgrimMedia(f.options).strike(message);
 assert.equal(f.played.length,1);assert.equal(f.writes.length,1);
 assert.deepEqual(f.writes[0],{[`flags.${M}.pilgrimStrike`]:true});
});

test('only a saved native successful attack from the owned branch can play',async t=>{
 const f=fixture(t);
 const cases=[
  message=>{message.flags.pf2e.context.outcome='failure';},
  message=>{message.flags.pf2e.context.type='damage-roll';},
  message=>{message.isCheckRoll=false;},
  message=>{message.rolls[0]._evaluated=false;},
  message=>{message.rolls[0].options.degreeOfSuccess=1;},
  message=>{f.game.messages.delete(message.id);},
  message=>{message.actor={...f.actor,uuid:'Actor.other'};},
  message=>{message.item={...f.item};},
  message=>{message.flags.pf2e.origin.uuid='Actor.hero.Item.unrelated';},
  message=>{message.flags.pf2e.context.token=f.target.uuid;},
  message=>{message.flags.pf2e.context.target.actor='Actor.unrelated';},
 ];
 for(const [index,change]of cases.entries()){const message=f.message(`invalid${index}`);change(message);await f.media.strike(message);}
 assert.equal(f.played.length,0);assert.equal(f.writes.length,0);
});

test('a critical success and a speaker token fallback remain valid',async t=>{
 const f=fixture(t),message=f.message();delete message.flags.pf2e.context.token;
 message.flags.pf2e.context.outcome='criticalSuccess';message.rolls[0].options.degreeOfSuccess=3;
 await f.media.strike(message);assert.equal(f.played.length,1);
});

test('native PF2e bare token IDs use the full context origin token UUID',async t=>{
 const lookedUp=[],f=fixture(t,{lookup:async(uuid,documents)=>{lookedUp.push(uuid);return documents.get(uuid);}}),message=f.message();
 const context=message.flags.pf2e.context;
 context.token=f.source.id;context.actor=f.actor.id;context.origin={actor:f.actor.uuid,token:f.source.uuid};
 await f.media.strike(message);
 assert.deepEqual(lookedUp,[f.source.uuid,f.target.uuid]);assert.equal(f.played.length,1);assert.equal(f.writes.length,1);
});

test('a bare context token ID without an origin UUID uses the speaker scene',async t=>{
 const f=fixture(t),message=f.message();message.flags.pf2e.context.token=f.source.id;
 await f.media.strike(message);assert.equal(f.played.length,1);assert.equal(f.writes.length,1);
});

test('malformed source token references never reach UUID lookup',async t=>{
 const lookedUp=[],f=fixture(t,{lookup:async(uuid,documents)=>{lookedUp.push(uuid);return documents.get(uuid);}});
 for(const [index,token]of ['Actor.hero','Scene.battle.Token.hero.Item.branch','../hero','hero,foe',''].entries()){
  const message=f.message(`malformed${index}`);message.flags.pf2e.context.token=token;await f.media.strike(message);
 }
 const origin=f.message('malformed-origin');origin.flags.pf2e.context.origin={token:'Actor.hero'};await f.media.strike(origin);
 assert.deepEqual(lookedUp,[]);assert.equal(f.played.length,0);assert.equal(f.writes.length,0);
});

test('a native alternate usage retains the real owned branch binding',async t=>{
 const f=fixture(t),message=f.message();
 const alternate={...f.item,system:{...f.item.system,baseItem:'longsword',group:'sword'}};
 f.actor.system={actions:[{type:'strike',item:f.item,altUsages:[{type:'strike',item:alternate}]}]};
 message.item=alternate;await f.media.strike(message);
 assert.deepEqual(f.sections('effect').map(section=>section.file[0]),[PILGRIM_MEDIA.sword,PILGRIM_MEDIA.leaves]);
 f.actor.items.delete(f.item.id);const deleted=f.message('deleted-source');deleted.item=alternate;await f.media.strike(deleted);
 assert.equal(f.played.length,1);
});

test('an authority or scene change during lookup cannot play or claim the message',async t=>{
 let f,change;
 f=fixture(t,{lookup:async(uuid,documents)=>{change();return documents.get(uuid);}});
 change=()=>{f.canvas.scene={id:'elsewhere',tokens:new Map()};};await f.media.strike(f.message('scene-change'));
 f.canvas.scene=f.scene;change=()=>{f.game.users.activeGM={id:'new-gm'};};await f.media.strike(f.message('gm-change'));
 assert.equal(f.played.length,0);assert.equal(f.writes.length,0);
});

test('media failures are reported without touching actor or item rule data',async t=>{
 const error=Error('asset load failed'),f=fixture(t,{playError:error}),message=f.message();
 const before=structuredClone({system:f.item.system,flags:f.item.flags,roll:message.rolls});
 await assert.doesNotReject(f.media.strike(message));
 assert.deepEqual({system:f.item.system,flags:f.item.flags,roll:message.rolls},before);
 assert.deepEqual(f.errors,[error]);assert.equal(f.writes.length,1);
 await f.media.strike(message);assert.equal(f.errors.length,1);
});

test('attached decoration respects hidden token alpha and visibility',async t=>{
 const f=fixture(t);f.source.hidden=true;f.target.hidden=true;
 await f.media.leaves(f.item,2);await f.media.scarf(f.item,true);await f.media.strike(f.message());
 for(const effect of f.sections('effect')){
  assert.ok(effect.attachTo,'token visuals must remain attached');
  assert.deepEqual(effect.attachTo[1],{bindAlpha:true,bindVisibility:true});
 }
 assert.equal(f.sections('effect').filter(effect=>effect.persist).length,3);
});

test('reconciliation and expiry clear only the matching item resource in the viewed scene',async t=>{
 const f=fixture(t),own=`${M}.pilgrim.${f.item.uuid}`;
 const keep=[{name:[`${own}.areaother-use`],sceneId:f.scene.id},{name:[`${M}.pilgrim.Actor.other.Item.branch.areaexpired`],sceneId:f.scene.id},{name:[`${own}.areaexpired`],sceneId:'other-scene'},{name:['other-module.effect'],sceneId:f.scene.id}].map((effect,index)=>({...effect,id:`keep${index}`}));
 f.effects.push(...keep,{id:'expired',name:[`${own}.areaexpired`],sceneId:f.scene.id});
 await f.media.clearArea(f.item,'expired');assert.deepEqual(f.effects,keep);
 await f.media.leaves(f.item,1);await f.media.leaves(f.item,0);
 assert.deepEqual(f.effects,keep);
 f.canvas.scene={id:'other-scene',tokens:new Map()};await f.media.clearArea(f.item,'expired');
 assert.deepEqual(f.effects,keep.filter(effect=>effect.sceneId!=='other-scene'));
});

test('unlit scarf reconciliation removes its own glow without affecting other effects',async t=>{
 const f=fixture(t);f.item.type='equipment';f.item.flags.world.sogWontonNativeAutomation.key='ghost-scarf';f.item.flags.world.sogB2Ch2Resources.originalSlug='ghost-scarf';
 f.actor.rollOptions.all['wonton-ghost-scarf:lit']=true;await f.media.reconcile(f.actor);
 assert.equal(f.effects.length,1);
 f.actor.rollOptions.all['wonton-ghost-scarf:lit']=false;await f.media.reconcile(f.actor);
 assert.equal(f.effects.length,0);
});

for(const {distance,sizes,radius}of [{distance:5,sizes:[7.02,7.2],radius:300},{distance:10,sizes:[3.51,3.6],radius:150}])test(`both storm layers keep a 15-foot mask on a ${distance}-foot grid`,async t=>{
 const f=fixture(t),area=f.areaDocuments();f.scene.grid.distance=distance;
 area.position={x:450,y:375};await f.media.area(f.item,area);
 const layers=f.sections('effect');assert.equal(layers.length,2);
 assert.deepEqual(layers.map(layer=>layer.file[0]),[
  'modules/jb2a_patreon/Library/Generic/Nature/SwirlingLeavesLoop02_01_Regular_Pink_400x400.webm',
  'modules/eskie-effects/assets/Nature/Flower/Particle/Flower_Particle_01_Pink.webm',
 ]);
 for(const [index,layer]of layers.entries()){
  assert.ok(Math.abs(layer.size[0]-sizes[index])<1e-12);assert.deepEqual(layer.size[1],{gridUnits:true});
  assert.deepEqual(layer.atLocation,[area.position]);assert.deepEqual(layer.opacity,[index===0?0.65:0.8]);
  assert.deepEqual(layer.name,[`${M}.pilgrim.${f.item.uuid}.areastorm`]);
  assert.deepEqual(layer.tieToDocuments,[[f.item.uuid,area.effect.uuid,area.template.uuid]]);
  const mask=layer.mask[0];assert.ok(mask instanceof PIXI.Circle);
  assert.deepEqual({x:mask.x,y:mask.y,radius:mask.radius},{x:450,y:375,radius});
  assert.ok(layer.persist);assert.equal(layer.temporary,undefined);
 }
 assert.equal(f.stored.length,2);assert.deepEqual(f.errors,[]);
});

test('storm cleanup removes both layers of one activation and preserves another',async t=>{
 const f=fixture(t),first=f.areaDocuments('first'),second=f.areaDocuments('second');
 await f.media.area(f.item,first);await f.media.area(f.item,second);
 assert.equal(f.stored.length,4);
 const keep=f.effects.filter(effect=>effect.name[0].endsWith('areasecond'));
 await f.media.clearArea(f.item,first.nonce);
 assert.equal(keep.length,2);assert.deepEqual(f.effects,keep);assert.deepEqual(f.stored,keep);
});

test('both storm layers survive a scene change while their lifecycle documents exist',async t=>{
 const f=fixture(t),area=f.areaDocuments(),other={id:'elsewhere',tokens:new Map()};
 await f.media.area(f.item,area);const saved=[...f.stored];assert.equal(saved.length,2);
 f.enterScene(other);assert.equal(f.effects.length,0);
 f.enterScene(f.scene);assert.deepEqual(new Set(f.effects),new Set(saved));assert.equal(f.stored.length,2);
 await f.media.clearArea(f.item,area.nonce);assert.equal(f.effects.length,0);assert.equal(f.stored.length,0);
});

test('release, tree, and storm activations use distinct sounds at calibrated volumes',async t=>{
 const f=fixture(t);
 await f.media.release(f.item,f.target);
 await f.media.area(f.item,{kind:'tree',scene:f.scene,position:{x:100,y:100},nonce:'tree'});
 await f.media.area(f.item,f.areaDocuments());
 assert.deepEqual(f.sections('sound').map(sound=>({file:sound.file[0],volume:sound.volume[0]})),[
  {file:'modules/psfx-patreon/library/1st-level-spells/cure-wounds/v1/cure-wounds-00.ogg',volume:0.22},
  {file:'modules/psfx-patreon/library/1st-level-spells/entangle/vines/v1/entangle-intro.ogg',volume:0.5},
  {file:'modules/psfx-patreon/library/cantrips/gust/v1/gust-001.ogg',volume:0.46},
 ]);
 assert.deepEqual(f.errors,[]);
});

test('activation and attack sounds keep their full natural duration with a short fade out',async t=>{
 const f=fixture(t);
 await f.media.release(f.item,f.target);
 await f.media.area(f.item,{kind:'tree',scene:f.scene,position:{x:100,y:100},nonce:'tree'});
 await f.media.area(f.item,f.areaDocuments());
 await f.media.strike(f.message('nature'));
 f.item.system.baseItem='longsword';f.item.system.group='sword';await f.media.strike(f.message('sword'));
 const sounds=f.sections('sound');assert.equal(sounds.length,5);
 for(const sound of sounds){assert.equal(sound.duration,undefined);assert.deepEqual(sound.fadeOutAudio,[200]);}
 assert.deepEqual(sounds.slice(-2).map(sound=>({file:sound.file[0],volume:sound.volume[0]})),[
  {file:PILGRIM_MEDIA.natureSound,volume:0.25},{file:PILGRIM_MEDIA.swordSound,volume:0.25},
 ]);
});

test('a storm cannot return after its effect, template, or source was deleted off scene',async t=>{
 const f=fixture(t),other={id:'elsewhere',tokens:new Map()};
 for(const key of ['effect','template','source']){
  const area=f.areaDocuments(`storm${key}`);f.documents.set(f.item.uuid,f.item);
  await f.media.area(f.item,area);assert.equal(f.stored.length,2);
  f.enterScene(other);f.documents.delete((key==='source'?f.item:area[key]).uuid);
  await f.media.clearArea(f.item,area.nonce);f.enterScene(f.scene);
  assert.equal(f.effects.length,0,`${key} deletion must prevent replay`);
  assert.equal(f.stored.length,0,`${key} deletion must invalidate the saved visual`);
 }
});

test('a storm without its own lifecycle documents never creates an orphan visual',async t=>{
 const f=fixture(t);await f.media.area(f.item,{kind:'storm',scene:f.scene,position:{x:100,y:100},nonce:'missing'});
 assert.equal(f.stored.length,0);assert.equal(f.played.length,0);assert.equal(f.errors.length,1);
});

test('leaf counts and scarf light are rebuilt after scene changes without saved stale visuals',async t=>{
 const f=fixture(t),other={id:'elsewhere',tokens:new Map()};
 await f.media.leaves(f.item,3);await f.media.scarf(f.item,true);
 assert.equal(f.effects.length,4);assert.equal(f.stored.length,0);
 f.enterScene(other);await f.media.leaves(f.item,1);await f.media.scarf(f.item,false);
 f.enterScene(f.scene);assert.equal(f.effects.length,0);
 await f.media.leaves(f.item,1);await f.media.scarf(f.item,false);
 assert.equal(f.effects.length,1);assert.equal(f.stored.length,0);
});

test('players can restore local token decorations without broadcasting duplicate media',async t=>{
 const f=fixture(t);f.game.user={id:'player'};
 await f.media.leaves(f.item,2);await f.media.scarf(f.item,true);
 assert.equal(f.effects.length,3);assert.equal(f.stored.length,0);
 assert.ok(f.playOptions.every(options=>options?.local===true));
 await f.media.leaves(f.item,0);await f.media.scarf(f.item,false);
 assert.equal(f.effects.length,0);assert.ok(f.ended.every(({push})=>push===false));
});

test('overlapping leaf and scarf refreshes keep only the latest state',async t=>{
 const f=fixture(t);
 await Promise.all([f.media.leaves(f.item,3),f.media.leaves(f.item,1)]);
 assert.equal(f.effects.length,1);
 await Promise.all([f.media.scarf(f.item,true),f.media.scarf(f.item,false)]);
 assert.equal(f.effects.length,1);
});

test('a scene change during decoration cleanup cannot render in the replacement scene',async t=>{
 let f;f=fixture(t,{endAction:()=>{f.canvas.scene={id:'elsewhere',tokens:new Map([[f.source.id,f.source]])};}});
 await f.media.leaves(f.item,2);await f.media.scarf(f.item,true);
 assert.equal(f.effects.length,0);
});

test('leaf restoration reads its native counter rather than stale saved leaves',async t=>{
 const f=fixture(t),effect=f.fanEffect(2);
 await f.media.reconcile(f.actor);assert.equal(f.effects.length,2);
 effect.system.badge.value=1;await f.media.reconcile(f.actor);assert.equal(f.effects.length,1);
 effect.isExpired=true;await f.media.reconcile(f.actor);assert.equal(f.effects.length,0);
 effect.isExpired=false;effect.remainingDuration.expired=true;await f.media.reconcile(f.actor);assert.equal(f.effects.length,0);
 f.actor.items.delete(effect.id);await f.media.reconcile(f.actor);assert.equal(f.effects.length,0);
});

test('leaf restoration ignores another source and binds the matching native effect',async t=>{
 const f=fixture(t),effect=f.fanEffect(2);effect.flags[M].sogPilgrim.source='Actor.hero.Item.anotherfan';
 await f.media.reconcile(f.actor);assert.equal(f.effects.length,0);
 effect.flags[M].sogPilgrim.source=f.item.uuid;await f.media.reconcile(f.actor);assert.equal(f.effects.length,2);
 assert.deepEqual(f.effects[0].tieToDocuments[0],[f.item.uuid,effect.uuid]);
});

test('deleting a fan during queued local refreshes leaves unrelated media alone',async t=>{
 const f=fixture(t);f.fanEffect(2);f.game.user={id:'player'};
 const keep={id:'other',name:['other-module.leaves'],sceneId:f.scene.id};f.effects.push(keep);
 const pending=f.media.leaves(f.item,2);f.actor.items.delete(f.item.id);f.documents.delete(f.item.uuid);
 await Promise.all([pending,f.media.deleted(f.item)]);
 assert.deepEqual(f.effects,[keep]);assert.equal(f.stored.length,0);assert.equal(f.errors.length,0);
});

test('an animation error releases the local refresh queue and leaves native rule data intact',async t=>{
 const error=Error('missing image'),f=fixture(t,{playError:error}),effect=f.fanEffect(2);
 const before=structuredClone({system:f.item.system,flags:f.item.flags,effect:effect.system});
 await f.media.leaves(f.item,2);await f.media.leaves(f.item,0);
 assert.deepEqual(f.errors,[error]);assert.equal(f.effects.length,0);
 assert.deepEqual({system:f.item.system,flags:f.item.flags,effect:effect.system},before);
});
