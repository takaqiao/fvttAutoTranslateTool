import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {createCommunalHealing} from '../scripts/communal-healing.mjs';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
import {registerAvDamageSnapshot} from '../scripts/av-damage-snapshot.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

const HEAL='Compendium.pf2e.spells-srd.Item.rfZpqmj0AIIdkVIs',FEAT='Compendium.pf2e.feats-srd.Item.yExxOkHN1PN37hUa';
const ROLL='CONFIG.PF2E.Item.documentClasses.spell.prototype.rollDamage',DAMAGE='CONFIG.PF2E.Item.documentClasses.spell.prototype.getDamage';
const TAG=`${ID}:communal-healing:cast:`;
const set=(doc,changes)=>{for(const[path,value]of Object.entries(changes)){let obj=doc;const keys=path.split('.');for(const k of keys.slice(0,-1))obj=obj[k]??={};obj[keys.at(-1)]=structuredClone(value);}};

function fixture({choice='Scene.scene.Token.secondary',full=false,world='bob-2',avSnapshot,libWrapperFactory}={}){
 const wrappers=new Map(),hooks=new Map(),docs=new Map(),choices=[],applied=[],errors=[],matchers=[];
 const gm={id:'gm',isGM:true,active:true,targets:new Set()},player={id:'player',active:true};
 const game={world:{id:world},user:gm,users:new Map([['gm',gm],['player',player]]),actors:new Map(),messages:new Map(),scenes:new Map(),settings:{get:()=> 'public'},time:{worldTime:1}};game.users.activeGM=gm;
 const Hooks={on:(n,fn)=>{const rows=hooks.get(n)??[];rows.push(fn);hooks.set(n,rows);return fn;},off:(n,fn)=>hooks.set(n,(hooks.get(n)??[]).filter(x=>x!==fn))};
 const emit=async(n,...args)=>{for(const fn of hooks.get(n)??[])await fn(...args);};
 const registrations=[],libWrapper=libWrapperFactory?.({wrappers,registrations})??{register:(id,path,fn)=>{if(wrappers.has(path))throw Error(`A wrapper for '${path}' has already been registered by module ${id}.`);registrations.push({id,path});wrappers.set(path,fn);},unregister:(_id,path)=>wrappers.delete(path)};
 const actor=id=>{const a={id,uuid:`Actor.${id}`,name:id,type:'character',modeOfBeing:'living',isDead:false,flags:{},items:new Map(),system:{attributes:{hp:{value:full?30:10,max:30,negativeHealing:false}},traits:{value:[]}},testUserPermission:u=>u===gm||u===player,getActiveTokens:()=>[]};a.update=async changes=>{set(a,changes);return a;};game.actors.set(id,a);docs.set(a.uuid,a);return a;};
 const caster=actor('caster'),primary=actor('primary'),secondary=actor('secondary');
 const scene={id:'scene',tokens:new Map()};game.scenes.set(scene.id,scene);game.scenes.current=scene;
 const token=a=>{const t={id:a.id,uuid:`Scene.scene.Token.${a.id}`,documentName:'Token',name:a.name,actor:a,parent:scene,hidden:false};t.object={document:t,isVisible:true};scene.tokens.set(t.id,t);docs.set(t.uuid,t);a.getActiveTokens=()=>[t.object];return t;};
 const tokens=[caster,primary,secondary].map(token);
 for(const a of [caster,primary,secondary])a.applyDamage=async params=>{applied.push({actor:a,params});a.system.attributes.hp.value=Math.min(a.system.attributes.hp.max,a.system.attributes.hp.value-params.damage);return a;};
 const entry={id:'entry',uuid:'Actor.caster.Item.entry',type:'spellcastingEntry',actor:caster,system:{prepared:{value:'spontaneous'},slots:{slot3:{value:4}}}};
 const feat={id:'feat',uuid:'Actor.caster.Item.feat',type:'feat',sourceId:FEAT,actor:caster};
 const spell={id:'heal',uuid:'Actor.caster.Item.heal',type:'spell',sourceId:HEAL,actor:caster,rank:3,spellcasting:entry,system:{location:{value:'entry'},cast:{focusPoints:0},time:{value:'2'},area:null},appliedOverlays:new Map([['override','two-actions']])};
 spell.loadVariant=()=>spell;
 for(const d of [entry,feat,spell]){caster.items.set(d.id,d);docs.set(d.uuid,d);}
 const casts=createNativeCastEvents({game,fromUuid:async uuid=>docs.get(uuid),messageTimeoutMs:50});
 const originalMatch=casts.addMatcher;casts.addMatcher=fn=>{matchers.push(fn);return originalMatch(fn);};
 const rpc=new Map(),socket={register:(name,fn)=>rpc.set(name,fn),executeAsUser:async(name,userId,payload)=>{const previous=game.user;game.user=game.users.get(userId);try{return await rpc.get(name).call({socketdata:{userId:previous.id}},payload);}finally{game.user=previous;}}};casts.register({Hooks,libWrapper,socket});
 let consumes=0,sequence=0,lastCard;
 entry.consume=(item,rank,slot,cap)=>wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.consume').call(entry,async()=>{consumes++;entry.system.slots.slot3.value--;return true;},item,rank,slot,cap);
 const card=async(item,options)=>{
  const opts={actualCast:options.consume!==false,data:{castRank:options.rank??item.rank}},own=casts.captureUsage(item,{options:opts});
  const m={id:`cast${++sequence}`,uuid:`ChatMessage.cast${sequence}`,author:player,actor:caster,item,rolls:[],speaker:{actor:caster.id,token:caster.id,scene:'scene'},flags:{[ID]:own??{},pf2e:{origin:{actor:caster.uuid,uuid:item.uuid,castRank:3}}},update:async changes=>{set(m,changes);return m;}};
  game.messages.set(m.id,m);docs.set(m.uuid,m);casts.captureMessageOutcome(item,opts,{message:m,castNonce:own?.nativeCast?.id});lastCard=m;return m;
 };
 const cast=async(options={rank:3})=>{const previous=game.user;game.user=player;player.targets=new Set([tokens[1].object]);try{await wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast').call(entry,async(item,opts)=>{if(opts.consume===false||await entry.consume(item,opts.rank??item.rank,opts.slotId)){if(opts.message!==false)await card(item,opts);}},spell,options);return lastCard;}finally{game.user=previous;}};
 const choose=async args=>{choices.push(args);return typeof choice==='function'?choice(args):choice;};
 const provider=createCommunalHealing({game,fromUuid:async uuid=>docs.get(uuid),castEvents:casts,choose,onError:error=>errors.push(error)});
 let cleanupAv;if(avSnapshot==='before')cleanupAv=registerAvDamageSnapshot({game,libWrapper});
 const cleanup=provider.register({Hooks,libWrapper,socket});
 if(avSnapshot==='after')cleanupAv=registerAvDamageSnapshot({game,libWrapper});
 const result=(source,changes={})=>{const m={id:`result${++sequence}`,uuid:`ChatMessage.result${sequence}`,author:player,speaker:{actor:primary.id,token:primary.id,scene:'scene'},token:tokens[1],actor:primary,item:spell,flags:{pf2e:{origin:{uuid:spell.uuid,actor:caster.uuid,castRank:3},context:{type:'damage-taken',domains:['healing-received'],options:[TAG+source.id]},appliedDamage:full?null:{uuid:primary.uuid,isHealing:true,isReverted:false,updates:[{path:'system.attributes.hp.value',value:-12}]}}}};set(m,changes);game.messages.set(m.id,m);docs.set(m.uuid,m);return m;};
 const run=async()=>{const source=await cast(),message=result(source);await emit('createChatMessage',message,{},player.id);return {source,message};};
 return {game,gm,player,caster,primary,secondary,entry,feat,spell,tokens,scene,casts,provider,cast,card,result,run,emit,Hooks,wrappers,docs,choices,applied,errors,matchers,cleanup,cleanupAv,registrations,consumes:()=>consumes};
}

test('native Heal spends one slot and prompts only after applying its primary healing',async()=>{
 const f=fixture(),source=await f.cast();assert.equal(f.choices.length,0);assert.equal(f.applied.length,0);
 await f.emit('createChatMessage',f.result(source),{},'player');
 assert.equal(f.consumes(),1);assert.equal(f.entry.system.slots.slot3.value,3);assert.equal(f.secondary.system.attributes.hp.value,13);
 assert.equal(f.choices[0].user,f.player);assert.equal(f.applied[0].params.damage,-3);assert.equal(f.applied[0].params.final,true);assert.equal(f.errors.length,0);
});
test('a full-health primary recipient still grants the secondary healing',async()=>{const f=fixture({full:true});await f.run();assert.equal(f.applied.length,1);});
test('the caster is an available recipient but the primary recipient is excluded',async()=>{
 const f=fixture({choice:'Scene.scene.Token.caster'});await f.run();
 assert.equal(f.caster.system.attributes.hp.value,13);assert.equal(f.choices[0].choices.some(c=>c.value===f.tokens[1].uuid),false);
});
test('a player character with the construct trait remains eligible when modeOfBeing is living',async()=>{const f=fixture();f.secondary.system.traits.value.push('construct');await f.run();assert.equal(f.secondary.system.attributes.hp.value,13);});
for(const change of [a=>a.modeOfBeing='undead',a=>a.modeOfBeing='construct',a=>a.system.attributes.hp.negativeHealing=true,a=>a.isDead=true])test('an ineligible secondary recipient cannot receive vitality healing',async()=>{const f=fixture();change(f.secondary);await f.run();assert.equal(f.applied.length,0);});
test('cancel leaves native primary healing and paid resources alone',async()=>{const f=fixture({choice:null});const {message}=await f.run();await f.emit('createChatMessage',message,{},'player');assert.equal(f.applied.length,0);assert.equal(f.consumes(),1);assert.equal(f.choices.length,1);});
test('duplicate result hooks and separate target results of one cast cannot heal twice',async()=>{
 const f=fixture(),source=await f.cast(),one=f.result(source),two=f.result(source,{'speaker.actor':'secondary','speaker.token':'secondary','flags.pf2e.appliedDamage.uuid':f.secondary.uuid});
 await Promise.all([f.emit('createChatMessage',one,{},'player'),f.emit('createChatMessage',two,{},'player'),f.emit('createChatMessage',one,{},'player')]);assert.equal(f.applied.length,1);assert.equal(f.choices.length,1);
});
test('two actual casts of the same Heal each grant one benefit',async()=>{const f=fixture();await f.run();await f.run();assert.equal(f.applied.length,2);assert.equal(f.consumes(),2);});
test('a second primary target cannot be selected from a multi-target result',async()=>{const f=fixture({choice:'Scene.scene.Token.primary'});await f.run();assert.equal(f.applied.length,0);});
for(const [name,alter]of [
 ['another world',f=>f.game.world.id='old-world'],['missing feat',f=>f.caster.items.delete('feat')],['three-action area Heal',f=>{f.spell.system.time.value='3';f.spell.system.area={type:'emanation',value:30};}],
 ['damage instead of healing',(f,m)=>{m.flags.pf2e.context.domains=['damage-received'];m.flags.pf2e.appliedDamage.isHealing=false;}],
 ['no cast source',(f,m)=>m.flags.pf2e.context.options=[]],['display-only card',(f,m,s)=>delete s.flags[ID].nativeCast],
 ['wrong spell origin',(f,m)=>m.flags.pf2e.origin.uuid='Actor.caster.Item.other'],['wrong rank',(f,m)=>m.flags.pf2e.origin.castRank=2],['a non-GM client',f=>f.game.user=f.player],
])test(`${name} does not grant communal healing`,async()=>{const f=fixture(),source=await f.cast(),message=f.result(source);alter(f,message,source);await f.emit('createChatMessage',message,{},'player');assert.equal(f.applied.length,0);assert.equal(f.choices.length,0);});
test('a claimed application is not repeated after a lost applyDamage reply',async()=>{const f=fixture();f.secondary.applyDamage=async params=>{f.applied.push(params);throw Error('reply lost');};const {message}=await f.run();await f.emit('createChatMessage',message,{},'player');assert.equal(f.applied.length,1);assert.equal(f.errors.length,1);});
test('changing the chosen recipient to undead while choosing stops the application',async()=>{const f=fixture({choice:()=>{f.secondary.modeOfBeing='undead';return f.tokens[2].uuid;}});await f.run();assert.equal(f.applied.length,0);});
test('a detached or display-only healing roll cannot acquire a cast marker',async()=>{
 const f=fixture(),source=await f.cast({consume:false});
 let data;const damage=async()=>({context:{options:new Set()},template:{}});f.spell.getDamage=(...args)=>f.wrappers.get(DAMAGE).call(f.spell,damage,...args);
 await f.wrappers.get(ROLL).call(f.spell,async()=>{data=await f.spell.getDamage();},{target:{closest:()=>({dataset:{messageId:source.id}})}});
 assert.deepEqual([...data.context.options],[]);
});
for(const [name,messageId]of [['no chat-card DOM',null],['a missing chat message','missing-message']])test(`Heal with ${name} preserves the native roll`,async()=>{
 const f=fixture(),event={target:{closest:()=>messageId?{dataset:{messageId}}:null}},expected={total:17};let calls=0;
 const result=await f.wrappers.get(ROLL).call(f.spell,async(nativeEvent,map)=>{calls++;assert.equal(nativeEvent,event);assert.equal(map,2);return expected;},event,2);
 assert.equal(result,expected);assert.equal(calls,1);assert.equal(f.choices.length,0);assert.equal(f.applied.length,0);
});
test('a non-Heal roll only uses the existing AV source lookup and calls the native roll once',async()=>{
 const f=fixture(),other={...f.spell,sourceId:'Compendium.pf2e.spells-srd.Item.other'},expected={total:9};let lookups=0,calls=0;
 const event={target:{closest:()=>{lookups++;return null;}}};
 const result=await f.wrappers.get(ROLL).call(other,async()=>{calls++;return expected;},event);
 assert.equal(result,expected);assert.equal(calls,1);assert.equal(lookups,1);assert.equal(f.applied.length,0);
});
test('a healing result referring to a missing Cast card is ignored without an error',async()=>{
 const f=fixture(),source=await f.cast(),message=f.result(source,{'flags.pf2e.context.options':[TAG+'missing-message']});
 await f.emit('createChatMessage',message,{},'player');assert.equal(f.errors.length,0);assert.equal(f.choices.length,0);assert.equal(f.applied.length,0);
});
for(const avSnapshot of ['before','after'])test(`AV ${avSnapshot} communal initialization shares one package registration and keeps Heal tagging`,async()=>{
 const f=fixture({avSnapshot}),cast=await f.cast();let data,nativeCalls=0;
 f.spell.getDamage=()=>f.wrappers.get(DAMAGE).call(f.spell,async()=>({context:{options:new Set()},template:{}}));
 const result=await f.wrappers.get(ROLL).call(f.spell,async()=>{nativeCalls++;return data=await f.spell.getDamage();},{target:{closest:()=>({dataset:{messageId:cast.id}})}});
 assert.equal(result,data);assert.equal(nativeCalls,1);assert.deepEqual([...data.context.options],[TAG+cast.id]);
 assert.deepEqual(f.registrations.filter(r=>r.path===ROLL),[{id:ID,path:ROLL}]);
 f.cleanup();assert.equal(typeof f.wrappers.get(ROLL),'function');f.cleanupAv();assert.equal(f.wrappers.has(ROLL),false);
});
test('removing the AV registration keeps the communal subscriber until its own cleanup',async()=>{
 const f=fixture({avSnapshot:'before'});f.cleanupAv();assert.equal(typeof f.wrappers.get(ROLL),'function');f.cleanup();assert.equal(f.wrappers.has(ROLL),false);
});
test('the shared roll wrapper preserves AV amp snapshots and recursive native calls',async()=>{
 const f=fixture({avSnapshot:'before'}),clones=[],nativeActors=[];
 const spell={...f.spell,sourceId:'Compendium.test.psi'},actor=f.caster;
 actor._source={items:[{_id:'amp',system:{rules:[{key:'RollOption',option:'amp-spell',value:false}]}}]};actor.rollOptions={all:{'amp-spell':false}};
 const card={id:'psi-cast',speaker:{actor:actor.id},flags:{[ID]:{avCast:{psiCantrip:true,itemUuid:spell.uuid,sourceId:spell.sourceId,amped:true,alternateAmp:null}},pf2e:{origin:{uuid:spell.uuid,actor:actor.uuid,castRank:3}}}};f.game.messages.set(card.id,card);
 actor.clone=(changes,options)=>{clones.push({changes,options});const copy={...actor,rollOptions:changes.flags.pf2e.rollOptions},variant={...spell,actor:copy};copy.items=new Map([[spell.id,variant]]);variant.loadVariant=()=>variant;variant.rollDamage=(event,map)=>f.wrappers.get(ROLL).call(variant,async()=>{nativeActors.push(copy);return {total:23};},event,map);return copy;};
 const event={target:{closest:()=>({dataset:{messageId:card.id}})}};
 const result=await f.wrappers.get(ROLL).call(spell,async()=>{throw Error('live actor damage must use the captured amp clone');},event,1);
 assert.equal(result.total,23);assert.equal(clones.length,1);assert.equal(nativeActors.length,1);assert.equal(nativeActors[0].rollOptions.all['amp-spell'],true);assert.equal(actor.rollOptions.all['amp-spell'],false);assert.equal(clones[0].options.keepId,true);assert.equal(f.choices.length,0);
 f.cleanup();assert.equal(typeof f.wrappers.get(ROLL),'function');
 const after=await f.wrappers.get(ROLL).call(spell,async()=>{throw Error('AV snapshot must remain installed');},event,1);assert.equal(after.total,23);assert.equal(clones.length,2);
});
test('another world adds no communal wrappers',()=>{const f=fixture({world:'old-world'});assert.equal(f.wrappers.has(ROLL),false);assert.equal(f.wrappers.has(DAMAGE),false);});
test('PF2e 8.6.1 native rollDamage carries the exact cast ID through getDamage',async()=>{
 const path=process.env.PF2E_NATIVE_861_BUNDLE;assert.ok(path,'run through run-tests.mjs for the pinned native source');
 const source=readFileSync(path,'utf8'),start=source.indexOf('\tasync rollDamage(e, t) {',source.indexOf('async getDamage(e = { skipDialog: !0 })'));
 assert.ok(start>0);const end=source.indexOf('\n\tasync rollCounteract(',start),method=source.slice(start,end);
 const f=fixture(),cast=await f.cast();f.spell.isVariant=true;let rolled;
 const native=vm.runInNewContext(`({${method}}).rollDamage`,{htmlClosest:(node,selector)=>node.closest(selector),game:f.game,eventToRollParams:()=>({}),DamagePF2e:{roll:async(template,context)=>{rolled={template,context};return rolled;}}});
 f.spell.getDamage=(...args)=>f.wrappers.get(DAMAGE).call(f.spell,async()=>({template:{formula:'3d10+24'},context:{options:new Set(['healing'])}}),...args);
 const event={target:{closest:selector=>({dataset:selector==='[data-cast-rank]'?{castRank:'3'}:{messageId:cast.id}})}};
 const result=await f.wrappers.get(ROLL).call(f.spell,native.bind(f.spell),event);
 assert.equal(result,rolled);assert.equal(rolled.template.formula,'3d10+24');assert.deepEqual([...rolled.context.options],['healing',TAG+cast.id]);assert.equal(f.applied.length,0);
});
