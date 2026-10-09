import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';

const fixture = JSON.parse(await readFile(new URL('./fixtures/turn-lifecycle-native.json', import.meta.url), 'utf8'));
const patchURL = new URL('../scripts/patches/turn-lifecycle.mjs', import.meta.url);
const install = existsSync(patchURL) ? (await import(patchURL.href)).installTurnLifecyclePatch : () => ({status:'skipped', reason:'missing-patch'});
const versions = {'pf2e-reaction':'1.4.3', 'pf2e-sustain-reminder':'1.1.0', 'pf2e-summons-assistant':'2.20.2', 'pf2e-toolbelt':'3.56.4'};

function environment({useChat=true, autoExpire=true, coreVersion='14.368', systemVersion='8.6.0', systemId='pf2e'} = {}) {
  const writes=[], messages=[], renders=[], deleted=[], timers=[], reports=[], prompts=[];
  const idError = new Error('You must provide an _id for every object in the update data array');
  const g = {console, CONFIG:{debug:{hooks:false}}, CONST:{vtt:'Foundry', DOCUMENT_OWNERSHIP_LEVELS:{OWNER:3}},
    foundry:{documents:{}, canvas:{placeables:{Token:class Token {}}}}, canvas:{ready:false, scene:null},
    game:{version:coreVersion, release:{generation:Number.parseInt(coreVersion,10)}, system:{id:systemId, version:systemVersion},
      modules:new Map(Object.entries(versions).map(([id,version])=>[id,{active:true,version}])), actors:new Map(),
      user:{id:'gm', isGM:true}, users:new Map([['gm',{id:'gm'}],['player',{id:'player'}]]), time:{worldTime:42},
      settings:{get(module,key){if(module==='pf2e-sustain-reminder'&&key==='useChat')return useChat; return false;}}},
    renderTemplate:async (path,data)=>{renders.push({path,data});return `${data.actor.id}:${data.effects.map(effect=>effect.id).join(',')}`;},
    setTimeout(fn,delay){timers.push({fn,delay});return timers.length;},
    reactionOptions:{allReactionEffect:false},
    A:()=>[], N:async (...args)=>prompts.push(args), h:item=>item.uuid, u:()=>0, ja:async()=>prompts.push(['round-one']),
    fromUuid:async()=>({toObject:()=>({type:'effect',system:{badge:{value:0}}})})};
  g.foundry.applications={handlebars:{renderTemplate:g.renderTemplate}};
  class Scene {constructor(){this.id='scene';this.tokens=new Map();}}
  class TokenDocument {
    constructor(actor,scene,id='token') {Object.assign(this,{actor,parent:scene,id,name:'Thrall token',actorLink:false});scene.tokens.set(id,this);}
    async delete() {if(this.failure)throw this.failure; deleted.push(this);this.parent.tokens.delete(this.id);return this;}
  }
  class Item {constructor(parent,{id='effect',name='Sustaining: Spell',type='effect',slug='thrall-expiration-date'}={}){Object.assign(this,{parent,id,name,type,rollOptionSlug:slug});}}
  Object.assign(g.foundry.documents,{Scene,TokenDocument,Item});
  vm.createContext(g);
  const Actor = vm.runInContext(`(class Actor {${fixture.core.isToken}\n${fixture.core.token}})`,g);
  g.foundry.documents.Actor=Actor;
  class Combatant {
    constructor(data,{parent}){Object.assign(this,data);this.parent=parent;this.flags={};}
    get id(){return this._id??null;}
    get token(){return this.parent.scene.tokens.get(this.tokenId)??null;}
    get actor(){return this.token?.actor??g.game.actors.get(this.actorId);}
    getFlag(scope,key){return this.flags[scope]?.[key];}
    async setFlag(scope,key,value){writes.push({combatant:this,scope,key,value});if(!this._id)throw idError;(this.flags[scope]??={})[key]=value;return this;}
  }
  g.foundry.documents.Combatant=Combatant;
  g.CONFIG.Actor={documentClass:Actor};g.CONFIG.Token={documentClass:TokenDocument};g.CONFIG.Combatant={documentClass:Combatant};
  const scene=new Scene();g.canvas.scene=scene;
  const encounter={id:'encounter',scene,round:2,combatants:[],turns:[]};g.game.combat=g.combat=encounter;
  g.Hooks=vm.runInContext(`${fixture.core.findSplice};Object.defineProperty(Array.prototype,'findSplice',{value:findSplice});${fixture.core.Hooks};Hooks$1`,g);
  const speakerMethods=['getSpeaker','getSpeakerFromToken','getSpeakerFromActor','getSpeakerFromUser'].map(key=>fixture.core[key]).join('\n');
  g.ChatMessage=vm.runInContext(`(class ChatMessage {${speakerMethods}})`,g);
  g.ChatMessage.create=async data=>{messages.push(data);return data;};
  vm.runInContext(fixture.sustain.script,g);
  const helpers=['reset','afterReset','count','feat','action','effect','npc','party','sourceEffect'].map(key=>fixture.reaction[key]).join('\n');
  vm.runInContext(`${helpers}\nconst S='pf2e-reaction',Aa='Compendium.pf2e-reaction.reaction-effects.Item.Bq05rfSWsBjNzjwq',w=reactionOptions;Hooks.on('pf2e.startTurn',${fixture.reaction.startTurn});`,g);
  if(autoExpire)vm.runInContext(`${fixture.summons.deleteItem}\n${fixture.summons.setup}\nsetupAutoDeleteThrallHook();`,g);
  const original={sustain:g.Hooks.events['pf2e.startTurn'][0].fn,reaction:g.Hooks.events['pf2e.startTurn'][1].fn,
    summons:g.Hooks.events.deleteItem?.[0]?.fn};
  const callback=part=>part==='summons'?g.Hooks.events.deleteItem[0].fn:g.Hooks.events['pf2e.startTurn'][part==='sustain'?0:1].fn;
  function actor(id='actor',type='npc') {
    const doc=new Actor();Object.assign(doc,{id,name:'Thrall actor',parent:null,type,alliance:'opposition',ownership:{gm:3,player:3,observer:2},items:[],
      itemTypes:{effect:[],action:[],feat:[],condition:[]},rules:[],conditions:{active:[]},isDead:false,inCombat:false,timeEvents:true,
      getActiveTokens:()=>[],isOfType:(...types)=>types.includes(type),update:async()=>{},recharge:async()=>{},
      createEmbeddedDocuments:async (kind,data)=>{prompts.push(['effect',kind,data]);return data;}});
    g.game.actors.set(id,doc);return doc;
  }
  function combatant({id='combatant',token=true,actor:doc=actor()}={}) {
    let tok;if(token)tok=new TokenDocument(doc,scene);
    const data={actorId:doc.id,tokenId:tok?.id,sceneId:scene.id,hidden:false};if(id!==null)data._id=id;
    return new Combatant(data,{parent:encounter});
  }
  function thrall({world=false,id='thrall-token'}={}) {
    const doc=actor('thrall-actor'),tok=new TokenDocument(doc,scene,id);if(!world)doc.parent=tok;
    const effect=new Item(doc);doc.items.push(effect);return {actor:doc,token:tok,effect,info:{parent:doc}};
  }
  const apply=()=>install({g,report:report=>reports.push(report)});
  return {g,writes,messages,renders,deleted,timers,reports,prompts,idError,original,callback,actor,combatant,thrall,encounter,scene,apply};
}

test('installed upstream callbacks reproduce null-token, missing-id and world-Actor expiry failures',async()=>{
  const f=environment(),temporary=f.combatant({id:null,token:false});
  await assert.rejects(f.original.sustain(temporary,f.encounter,'gm'),/null.*actor/);
  await assert.rejects(f.original.reaction(temporary),error=>error===f.idError);
  assert.equal(f.writes.length,1);assert.equal(f.writes[0].key,'state');
  const thrall=f.thrall({world:true});await assert.rejects(f.original.summons(thrall.effect,thrall.info),/null.*delete/);
});

test('null-token shared turn keeps the Actor sustain reminder, ownership and native speaker selection',async()=>{
  const f=environment(),doc=f.actor();
  doc.items=[{id:'sustain',type:'effect',name:'Sustaining: Summon'},{id:'ordinary',type:'effect',name:'Ordinary effect'},
    {id:'spell',type:'spell',name:'Sustaining: Not an effect'}];
  const temporary=f.combatant({id:null,token:false,actor:doc});f.apply();
  await f.callback('sustain')(temporary,f.encounter,'player');
  assert.equal(f.messages.length,1);assert.equal(f.messages[0].content,'actor:sustain');
  assert.deepEqual(Array.from(f.messages[0].whisper),['gm','player']);
  assert.deepEqual({...f.messages[0].speaker},{scene:'scene',actor:'actor',token:null,alias:'Thrall actor'});
  assert.deepEqual({...f.messages[0].flags},{'pf2e-sustain-reminder':true});
  assert.equal(f.renders[0].data.actor,doc);assert.equal(temporary.token,null);assert.equal(temporary.id,null);
});

test('Reaction Checker alone ignores temporary Combatants without generating a persistent identity',async()=>{
  for(const token of [false,true]){
    const f=environment(),temporary=f.combatant({id:null,token});f.apply();
    await f.callback('reaction')(temporary,f.encounter,'gm');
    assert.equal(f.writes.length,0);assert.equal(temporary._id,undefined);assert.equal(temporary.id,null);
    assert.deepEqual(temporary.flags,{});assert.equal(f.prompts.length,0);
  }
});

test('persisted combatants keep native reset, bonus reactions, other-combatant updates and reaction effects',async()=>{
  const f=environment(),doc=f.actor();
  doc.itemTypes.action.push({slug:'triple-opportunity'});doc.itemTypes.effect.push({slug:'effect-hydra-heads',system:{badge:{value:3}}});
  const real=f.combatant({actor:doc}),ally=f.combatant({id:'ally',token:false,actor:f.actor('ally','character')});
  ally.actor.alliance='party';ally.actor.itemTypes.feat.push({slug:'inexhaustible-countermoves'});f.encounter.combatants.push(real,ally);
  f.g.reactionOptions.allReactionEffect=true;f.apply();await f.callback('reaction')(real);
  assert.deepEqual(real.flags,{'pf2e-reaction':{'triple-opportunity':1,'hydra-heads':2,state:true}});
  assert.equal(ally.getFlag('pf2e-reaction','inexhaustible-countermoves'),1);
  assert.equal(f.timers.length,1);assert.equal(f.timers[0].delay,300);await f.timers[0].fn();
  assert.equal(f.prompts[0][0],'effect');assert.equal(f.prompts[0][2][0].system.badge.value,4);
});

test('normal token-based sustain uses the original selection and returns the original asynchronous result',async()=>{
  const left=environment(),right=environment();
  for(const f of [left,right]){
    const real=f.combatant();real.actor.items.push({id:'sustain',type:'effect',name:'Sustaining: Summon'});
    f.real=real;
  }
  right.apply();const a=await left.original.sustain(left.real,left.encounter,'gm'),b=await right.callback('sustain')(right.real,right.encounter,'gm');
  assert.equal(b,a);assert.equal(right.messages.length,1);
  assert.deepEqual(JSON.parse(JSON.stringify(right.messages)),JSON.parse(JSON.stringify(left.messages)));
});

test('missing-token sustain respects useChat and does not change effects or duration',async()=>{
  for(const useChat of [false,true]){
    const f=environment({useChat}),doc=f.actor();f.apply();
    await f.callback('sustain')(f.combatant({id:null,token:false,actor:doc}));assert.equal(f.messages.length,0);
    doc.items.push({id:'sustain',type:'effect',name:'Sustaining: Spell',duration:{remaining:6}});
    await f.callback('sustain')(f.combatant({id:null,token:false,actor:doc}));
    assert.equal(f.messages.length,useChat?1:0);assert.equal(doc.items[0].duration.remaining,6);
  }
});

test('Toolbelt native turn-start/end still updates actors and effects and dispatches to unrelated consumers',async()=>{
  const f=environment(),events=[],hooks=[],doc=f.actor('slave','character');f.apply();
  doc.rules=[{onUpdateEncounter:async ({event,actorUpdates})=>{events.push(['rule',event]);actorUpdates.ready=true;}}];
  doc.update=async data=>events.push(['update',{...data}]);doc.recharge=async data=>events.push(['recharge',{...data}]);
  doc.itemTypes.effect=[{onEncounterEvent:async event=>events.push(['effect',event])}];
  doc.familiar={itemTypes:{effect:[{onEncounterEvent:async event=>events.push(['familiar',event])}]}};
  doc.conditions.active=[{onEndTurn:async data=>events.push(['condition',data.token])}];
  f.g.Hooks.on('pf2e.startTurn',(combatant,encounter,user)=>hooks.push(['start',combatant,encounter,user]));
  f.g.Hooks.on('pf2e.endTurn',(combatant,encounter,user)=>hooks.push(['end',combatant,encounter,user]));
  const calls=[];
  const SharedTurn=vm.runInNewContext(`(class SharedTurn {${Object.values(fixture.toolbelt).join('\n')} #v(actor,key){return actor[key];} run(options){return this.#m(options);}})`,
    {game:f.g.game,Hooks:{callAll:(...args)=>calls.push(args)},cr:()=>null,getDocumentClass:name=>{assert.equal(name,'Combatant');return f.g.foundry.documents.Combatant;}});
  const shared=new SharedTurn();
  for(const eventType of ['turn-start','turn-end']){
    await shared.run({encounter:f.encounter,eventType,slave:doc});
    const [hook,...args]=calls.at(-1);for(const entry of f.g.Hooks.events[hook])await entry.fn(...args);
  }
  assert.deepEqual(events,[['rule','turn-start'],['update',{ready:true}],['recharge',{duration:'round'}],['effect','turn-start'],['familiar','turn-start'],
    ['condition',null],['effect','turn-end'],['familiar','turn-end']]);
  assert.deepEqual(hooks.map(([kind,c,enc,user])=>[kind,c.id,c.actor.id,enc.id,user]),[['start',null,'slave','encounter','gm'],['end',null,'slave','encounter','gm']]);
  assert.equal(f.writes.length,0);assert.equal(doc.timeEvents,true);assert.equal(f.g.Hooks.events['pf2e.startTurn'].length,3);
});

test('expiry of a synthetic thrall deletes only its exact Token and preserves the native result',async()=>{
  const f=environment(),thrall=f.thrall();
  const other=new f.g.foundry.documents.TokenDocument(thrall.actor,f.scene,'other-token');f.apply();
  assert.equal(await f.callback('summons')(thrall.effect,thrall.info,'gm'),thrall.token);
  assert.deepEqual(f.deleted,[thrall.token]);assert.equal(f.scene.tokens.get('other-token'),other);
});

test('world-Actor expiry safely skips Token deletion even when matching active Tokens exist',async()=>{
  const f=environment(),thrall=f.thrall({world:true});f.apply();
  thrall.actor.getActiveTokens=()=>[{document:thrall.token}];
  await f.callback('summons')(thrall.effect,thrall.info,'gm');
  assert.deepEqual(f.deleted,[]);assert.equal(f.scene.tokens.get(thrall.token.id),thrall.token);
});

test('expiry rejects foreign parents, linked tokens, non-Token parents and non-effect items',async()=>{
  for(const kind of ['foreign-actor','linked','not-token','not-effect','no-info']){
    const f=environment(),thrall=f.thrall();f.apply();
    if(kind==='foreign-actor')thrall.info.parent=f.thrall({id:'foreign-token'}).actor;
    if(kind==='linked')thrall.token.actorLink=true;
    if(kind==='not-token')thrall.actor.parent={delete:async()=>{throw Error('not a Token');}};
    if(kind==='not-effect')thrall.effect.type='spell';
    await f.callback('summons')(thrall.effect,kind==='no-info'?undefined:thrall.info,'gm');
    assert.deepEqual(f.deleted,[],kind);
  }
});

test('other effects and non-GM callbacks preserve native early returns',async()=>{
  const f=environment(),thrall=f.thrall({world:true});f.apply();
  thrall.effect.rollOptionSlug='other-effect';assert.equal(await f.callback('summons')(thrall.effect,thrall.info),undefined);
  f.g.game.user.isGM=false;assert.equal(await f.callback('summons')(undefined,undefined),undefined);
  assert.deepEqual(f.deleted,[]);
});

test('unrelated original errors retain their identity for durable turns, reminders and legal deletion',async()=>{
  for(const part of ['reaction','sustain','summons']){
    const f=environment(),error=new Error('original-'+part);f.apply();
    let invoke;
    if(part==='reaction'){const real=f.combatant();real.setFlag=async()=>{throw error;};invoke=()=>f.callback(part)(real);}
    if(part==='sustain'){const real=f.combatant();Object.defineProperty(real.actor,'items',{get(){throw error;}});invoke=()=>f.callback(part)(real);}
    if(part==='summons'){const thrall=f.thrall();thrall.token.failure=error;invoke=()=>f.callback(part)(thrall.effect,thrall.info);}
    await assert.rejects(invoke,errorValue=>errorValue===error);
  }
});

test('null-token reminder errors are propagated unchanged instead of hiding failed chat creation',async()=>{
  for(const operation of ['template','create']){
    const f=environment(),doc=f.actor(),error=new Error(operation+' failed');
    doc.items.push({id:'sustain',type:'effect',name:'Sustaining: Spell'});f.apply();
    if(operation==='template')f.g.foundry.applications.handlebars.renderTemplate=async()=>{throw error;};
    else f.g.ChatMessage.create=async()=>{throw error;};
    await assert.rejects(()=>f.callback('sustain')(f.combatant({id:null,token:false,actor:doc})),value=>value===error);
  }
});

test('objects that only resemble a temporary Combatant retain the original callback behavior',async()=>{
  const f=environment(),error=new Error('foreign flag operation'),doc=f.actor();f.apply();
  const foreign={id:null,_id:null,token:null,actor:doc,getFlag:()=>undefined,setFlag:async()=>{throw error;}};
  await assert.rejects(()=>f.callback('reaction')(foreign),value=>value===error);
  await assert.rejects(()=>f.callback('sustain')(foreign),/null.*actor/);
});

test('other systems leave every native callback untouched',()=>{
  for(const args of [{systemId:'sf2e'}]){
    const f=environment(args),result=f.apply();assert.equal(result.status,'skipped');
    for(const part of ['reaction','sustain','summons'])assert.equal(f.callback(part),f.original[part]);
  }
});

test('audited callbacks retain their protections across core and PF2e version labels',async()=>{
 for(const systemVersion of ['8.5.2','8.6.0','9.0.0']){
 const f=environment({coreVersion:'15.1',systemVersion}),result=f.apply();
 assert.equal(result.parts.summons.status,'installed');assert.equal(result.parts.sustain.status,'installed');
 assert.equal(result.parts.reaction.status,'installed');
 const world=f.thrall({world:true});assert.equal(await f.callback('summons')(world.effect,world.info),undefined);
 assert.equal(f.deleted.length,0);
 }
});

test('unknown or inactive consumer versions skip independently while known consumers install',()=>{
  for(const [part,id] of [['reaction','pf2e-reaction'],['sustain','pf2e-sustain-reminder'],['summons','pf2e-summons-assistant']])for(const kind of ['version','inactive']){
    const f=environment();if(kind==='version')f.g.game.modules.get(id).version='999';else f.g.game.modules.get(id).active=false;
    const result=f.apply();assert.equal(result.status,'installed');assert.equal(result.parts[part].status,'skipped');
    assert.equal(f.callback(part),f.original[part]);
    for(const other of ['reaction','sustain','summons'].filter(key=>key!==part))assert.equal(result.parts[other].status,'installed');
  }
});

test('unknown callback sources, duplicate listeners and foreign Hook implementations are not overwritten',()=>{
  for(const part of ['reaction','sustain','summons'])for(const kind of ['source','duplicate','core-wrapper']){
    const f=environment();
    const hook=part==='summons'?'deleteItem':'pf2e.startTurn',index=part==='reaction'?1:0;
    if(kind==='source')f.g.Hooks.events[hook][index].fn=async()=> 'foreign';
    if(kind==='duplicate')f.g.Hooks.on(hook,f.original[part]);
    if(kind==='core-wrapper')f.g.Hooks.callAll=function(){};
    const before=f.callback(part),result=f.apply();assert.equal(f.callback(part),before);
    assert.equal(result.parts[part].status,'skipped');
    if(kind==='core-wrapper')assert.equal(result.status,'skipped');else assert.equal(result.status,'installed');
  }
});

test('diagnostics distinguish absent expiry hook, preserve native records and support removal and restore',()=>{
  const absent=environment({autoExpire:false}),partial=absent.apply();assert.equal(partial.parts.summons.status,'skipped');
  const f=environment(),Hooks=f.g.Hooks,off=Hooks.off,list=Hooks.events['pf2e.startTurn'],records=[...list],result=f.apply();
  assert.equal(result.status,'installed');assert.equal(f.apply(),result);assert.equal(f.reports[0].feature,'turn-lifecycle');
  assert.equal(Hooks.events['pf2e.startTurn'],list);assert.deepEqual([...list],records);
  Hooks.off('pf2e.startTurn',f.original.reaction);assert.equal(list.length,1);
  result.restore();result.restore();assert.equal(list.length,1);assert.equal(list[0].fn,f.original.sustain);assert.equal(Hooks.off,off);
  assert.equal(Hooks.events.deleteItem[0].fn,f.original.summons);
});

test('null-token reminders use the namespaced template API without a global alias',async()=>{
  const f=environment({coreVersion:'14.369'}),doc=f.actor();
  delete f.g.renderTemplate;
  doc.items.push({id:'sustain',type:'effect',name:'Sustaining: Spell'});
  assert.equal(f.apply().parts.sustain.status,'installed');
  await f.callback('sustain')(f.combatant({id:null,token:false,actor:doc}),f.encounter,'player');
  assert.equal(f.messages.length,1);assert.equal(f.messages[0].content,'actor:sustain');
  assert.equal(f.renders[0].path,'modules/pf2e-sustain-reminder/templates/sustain-reminder.hbs');
});

test('a global template alias cannot enable sustain when the namespaced API is unavailable',()=>{
  const f=environment();delete f.g.foundry.applications.handlebars.renderTemplate;
  const result=f.apply();
  assert.equal(result.parts.sustain.status,'skipped');
  assert.equal(result.parts.sustain.reason,'document-api-unavailable');
  assert.equal(f.callback('sustain'),f.original.sustain);
  assert.equal(result.parts.reaction.status,'installed');assert.equal(result.parts.summons.status,'installed');
});
