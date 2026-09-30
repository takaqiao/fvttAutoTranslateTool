import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {MODULE_ID} from '../scripts/rules.mjs';
let api;
try { api = await import('../scripts/knowledge-workbench.mjs'); } catch {}
const command=fs.readFileSync(process.env.FVTT_WORKBENCH_RECALL_MACRO??new URL('./fixtures/workbench-7.7.5-recall.txt',import.meta.url),'utf8');
const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
function fixture(){
 const die={count:0};
 class NativeRoll {constructor(formula){assert.equal(formula,'1d20');this.total=12;this.options={};}async roll(){die.count++;return this;}}
 const skills=Object.fromEntries(['arcana','crafting','medicine','nature','occultism','religion','society'].map((slug,index)=>[slug,{slug,label:slug,rank:2,totalModifier:slug==='society'?13:index,modifiers:[],async roll(options){assert.equal(options.createMessage,false);const roll={total:7+this.totalModifier,options:{totalModifier:this.totalModifier}},message={flags:{pf2e:{context:{options:options.extraRollOptions,domains:['skill-check',slug],type:'skill-check'},modifiers:[]}},getFlag(ns,key){return key.split('.').reduce((a,k)=>a?.[k],this.flags[ns]);},flavor:''};options.callback(roll,undefined,message);return roll;}}]));
 const user={id:'player',isGM:false}; const gm={id:'gm',isGM:true};
 const actor={id:'a',uuid:'Actor.a',name:'Hero',skills,itemTypes:{feat:[],lore:[]},synthetics:{degreeOfSuccessAdjustments:{}},testUserPermission:u=>u===user||u===gm};
 const targetActor={uuid:'Actor.enemy',name:'Hidden enemy',level:5,traits:new Set(['humanoid']),rarity:'common',getSelfRollOptions:()=>['target:trait:humanoid'],isOfType:()=>true};
 const target={id:'enemy',uuid:'Scene.s.Token.enemy',documentName:'Token',actor:targetActor,name:'Hidden enemy',parent:{id:'s'}};target.object={document:target,actor:targetActor};
 const token={id:'hero',uuid:'Scene.s.Token.hero',documentName:'Token',actor,parent:target.parent};token.object={document:token,actor};
 const messages=new Map();
 class Messages {static getSpeaker(){return {actor:actor.id,token:token.id,scene:'s'};}static getWhisperRecipients(){return [gm];}static async create(data){const message={id:'rk1',uuid:'ChatMessage.rk1',actor,...data,author:user,async update(changes){for(const[k,v]of Object.entries(changes)){let o=this;const ps=k.split('.');for(const p of ps.slice(0,-1))o=o[p]??={};o[ps.at(-1)]=v;}return this;}};messages.set(message.id,message);return message;}}
 const macro={type:'script',command,async execute(scope){return new AsyncFunction(...Object.keys(scope),this.command)(...Object.values(scope));}};
 const game={user,userId:user.id,users:{get:id=>id===user.id?user:id===gm.id?gm:null,activeGM:gm},system:{id:'pf2e'},messages,settings:{get:()=> 'none'},modules:new Map([['xdy-pf2e-workbench',{active:true}]]),packs:new Map(),time:{worldTime:0}};
 game.user.targets=new Set([target.object]);game.user.targets.first=()=>target.object;
 const fromUuid=async uuid=>uuid.endsWith('xcFr7PWwG5OVALNJ')?macro:uuid===actor.uuid?actor:uuid===token.uuid?token:uuid===target.uuid?target:null;
 const globals={Roll:NativeRoll,ChatMessage:Messages,CONST:{DICE_ROLL_MODES:{BLIND:'blindroll'},CHAT_MESSAGE_STYLES:{OTHER:0}},CONFIG:{PF2E:{abilities:{}}},ui:{notifications:{info(){}}},document:{createElement(){throw Error('none breakdown must not create DOM');}}};
 return {game,actor,user,gm,target,token,fromUuid,globals,die};
}
test('installed Workbench macro produces one secret same-die target comparison with native captured modifiers',async()=>{
 assert.ok(api?.captureWorkbenchRecall,'Workbench bridge is missing');const f=fixture();
 const capture=await api.captureWorkbenchRecall({...f,requestId:'request1',targetUuids:[f.target.uuid]});
 assert.equal(f.die.count,1);assert.equal(capture.die,12);assert.equal(capture.candidates.length,1);assert.equal(capture.candidates[0].statistic,'society');assert.equal(capture.candidates[0].total,25);assert.equal(capture.candidates[0].dc,20);assert.equal(capture.candidates[0].degree,2);
 assert.equal(capture.message.blind,true);assert.deepEqual(capture.message.whisper,['gm']);assert.equal(capture.message.flags[MODULE_ID].workbenchRecall.requestId,'request1');
});
test('fixed skill is retained instead of being replaced by the target highest skill',async()=>{
 assert.ok(api?.captureWorkbenchRecall,'Workbench bridge is missing');const f=fixture();f.actor.skills.society.totalModifier=2;
 const capture=await api.captureWorkbenchRecall({...f,requestId:'request2',targetUuids:[f.target.uuid],statistic:'occultism',dc:20});
 assert.equal(capture.candidates.length,1);assert.equal(capture.candidates[0].statistic,'occultism');assert.equal(capture.candidates[0].total,16);assert.equal(f.die.count,1);
});
test('Assurance uses 10 plus proficiency without rolling or adding ability/item bonuses',async()=>{
 assert.ok(api?.captureWorkbenchRecall,'Workbench bridge is missing');const f=fixture();f.actor.skills.occultism.modifiers=[{type:'proficiency',modifier:9},{type:'ability',modifier:4},{type:'item',modifier:1}];
 f.actor.items=[{_stats:{compendiumSource:'Compendium.pf2e.feats-srd.Item.W6Gl9ePmItfDHji0'},system:{rules:[{key:'ChoiceSet',flag:'assurance',selection:'occultism'}]}}];
 const capture=await api.captureWorkbenchRecall({...f,requestId:'assurance1',targetUuids:[f.target.uuid],statistic:'occultism',assurance:true,dc:20});
 assert.equal(f.die.count,0);assert.equal(capture.die,null);assert.equal(capture.candidates[0].total,19);assert.equal(capture.candidates[0].degree,1);assert.deepEqual(capture.message.whisper,['gm']);
});
test('final result can only be selected by an actual GM from the captured native candidate',async()=>{
 assert.ok(api?.finalizeWorkbenchRecall,'Workbench bridge is missing');const f=fixture();const capture=await api.captureWorkbenchRecall({...f,requestId:'selection1',targetUuids:[f.target.uuid]});
 await assert.rejects(()=>api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:f.game,message:capture.message,user:f.user,statistic:'society'}),/GM/);
 const result=await api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:{...f.game,user:f.gm},message:capture.message,user:f.gm,statistic:'society'});assert.equal(result.degree,2);assert.equal(capture.message.flags.pf2e.context.outcome,'success');assert.equal(capture.message.flags.pf2e.context.type,'skill-check');
 await assert.rejects(()=>api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:{...f.game,user:f.gm},message:capture.message,user:f.gm,statistic:'religion',total:99}),/候选|candidate/);
});
test('native natural 20/1 and the most favorable adjustment apply once, never stack domains',()=>{
 assert.ok(api?.recallDegree);assert.equal(api.recallDegree({total:19,die:20,dc:20}),2);assert.equal(api.recallDegree({total:20,die:1,dc:20}),1);
 const adjustment={predicate:{test:()=>true},adjustments:{all:[{amount:1}]}};
 const actor={synthetics:{degreeOfSuccessAdjustments:{society:[adjustment],'skill-check':[adjustment]}}};
 assert.equal(api.recallDegree({total:19,die:12,dc:20,actor,domains:['society','skill-check']}),2);
});
test('automatic primary is highest applicable captured modifier and later GM changes cannot replay it',async()=>{
 const f=fixture();f.target.actor.traits=new Set(['construct']);f.actor.skills.arcana.totalModifier=7;f.actor.skills.crafting.totalModifier=8;
 const capture=await api.captureWorkbenchRecall({...f,requestId:'primary',targetUuids:[f.target.uuid]});
 const result=await api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:{...f.game,user:f.gm},message:capture.message,user:f.gm});assert.equal(result.statistic,'crafting');assert.equal(result.degree,2);
 await assert.rejects(()=>api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:{...f.game,user:f.gm},message:capture.message,user:f.gm,statistic:'arcana',dc:18}),/锁定/);assert.equal(f.die.count,1);
 assert.deepEqual(await api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:{...f.game,user:f.gm},message:capture.message,user:f.gm}),result);
});
test('no target keeps a blind comparison without inventing a creature result',async()=>{
 const f=fixture();const capture=await api.captureWorkbenchRecall({...f,requestId:'none',targetUuids:[]});
 assert.equal(capture.candidates.length,7);assert.equal(await api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:{...f.game,user:f.gm},message:capture.message,user:f.gm}),null);assert.equal(f.die.count,1);
});
test('a persisted die or candidate total mismatch rejects external result substitution',async()=>{
 const f=fixture();const capture=await api.captureWorkbenchRecall({...f,requestId:'integrity',targetUuids:[f.target.uuid]});capture.message.flags[MODULE_ID].workbenchRecall.candidates[0].total=99;
 await assert.rejects(()=>api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:{...f.game,user:f.gm},message:capture.message,user:f.gm}),/不可验证|原始/);
});
test('scoped macro supports native private fields and locked actor skills without mutating native objects',async()=>{
 const f=fixture();class NativeTarget{#kind='creature';isOfType(){return this.#kind==='creature';}getSelfRollOptions(){return ['target:trait:humanoid'];}}
 const targetActor=Object.assign(new NativeTarget(),f.target.actor);delete targetActor.isOfType;delete targetActor.getSelfRollOptions;f.target.actor=targetActor;f.target.object.actor=targetActor;
 Object.defineProperty(f.actor,'skills',{value:f.actor.skills,writable:false,configurable:false,enumerable:true});
 const capture=await api.captureWorkbenchRecall({...f,requestId:'private',targetUuids:[f.target.uuid]});assert.equal(capture.candidates[0].statistic,'society');assert.equal(f.die.count,1);
});
import {createWorkbenchRecallController} from '../scripts/knowledge-entrypoints.mjs';
import {automaticKnowledgeChoices,automaticKnowledgeRound,AUTOMATIC_KNOWLEDGE_SOURCE,ASSURANCE_SOURCE} from '../scripts/knowledge-automatic.mjs';
function controllerFixture(){
 const f=fixture();f.user.active=true;f.gm.active=true;f.actor.getActiveTokens=()=>[f.token.object];f.game.user.character=f.actor;
 const gmGame={...f.game,user:f.gm};const handlers=new Map(),wrappers=new Map(),resolved=[];
 const socketFor=user=>({register(name,fn){handlers.set(`${user.id}:${name}`,fn);},executeAsUser(name,id,payload){return handlers.get(`${id}:${name}`).call({socketdata:{userId:user.id}},payload);}});
 const native={slug:'recall-knowledge',async use(){throw Error('unwrapped native skill chooser');}};class Variant{get slug(){return 'recall-knowledge';}async use(){throw Error('unwrapped native variant');}};native.getDefaultVariant=()=>new Variant();
 f.game.pf2e={actions:new Map([['recall-knowledge',native]])};gmGame.pf2e={actions:new Map()};
 const hooks={on(){return 1;},off(){}};
 const owner=createWorkbenchRecallController({...f,onError:e=>{throw e;}}),gmController=createWorkbenchRecallController({...f,game:gmGame,onResolved:message=>resolved.push(message.id)});
 gmController.register({Hooks:hooks,socket:socketFor(f.gm)});const cleanup=owner.register({Hooks:hooks,socket:socketFor(f.user),libWrapper:{register(_id,path,fn){wrappers.set(path,fn);},unregister(_id,path){wrappers.delete(path);}}});
 return {...f,gmGame,owner,gmController,handlers,wrappers,resolved,cleanup,native,socketFor};
}
test('ordinary native RK and explicit variants automatically choose primary on owner and expose no secret RPC totals',async()=>{
 const f=controllerFixture();const result=await f.native.getDefaultVariant().use({actors:[f.actor],statistic:'arcana'});assert.equal(result[0].message.flags[MODULE_ID].workbenchRecall.result.statistic,'society');assert.equal(f.die.count,1);assert.deepEqual(f.resolved,['rk1']);
 const response=await f.handlers.get('gm:knowledge-rk-finalize').call({socketdata:{userId:'player'}},{messageId:'rk1',total:999,dc:1});assert.deepEqual(response,{ok:true,value:{messageId:'rk1'}});assert.equal(f.die.count,1);f.cleanup();
});
test('public Workbench macro hotbar interception routes before its detached native wrapper starts another macro',async()=>{
 const f=controllerFixture();let oldCalls=0;const wrapper=f.wrappers.get('CONFIG.Macro.documentClass.prototype.execute');
 await wrapper.call({uuid:'Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros.Macro.es70r3Bq0bxZSCuk'},()=>{oldCalls++;},{});assert.equal(oldCalls,0);assert.equal(f.die.count,1);f.cleanup();
});
test('GM dispatches incidental RK to its original owner and repeated request never rolls twice',async()=>{
 const f=controllerFixture();const original={id:'source',actor:f.actor,author:f.user,speaker:{actor:f.actor.id,scene:'s',token:f.token.id},flags:{[MODULE_ID]:{knowledge:{recall:{actorUuid:f.actor.uuid,targetUuid:f.target.uuid,userId:f.user.id}}}},async update(changes){for(const[k,v]of Object.entries(changes)){let o=this;const ps=k.split('.');for(const p of ps.slice(0,-1))o=o[p]??={};o[ps.at(-1)]=v;}}};f.game.messages.set(original.id,original);
 const input={actor:f.actor,token:f.token,user:f.user,targetUuids:[f.target.uuid],requestId:'incidental',origin:{messageId:original.id,rollOptions:[`${MODULE_ID}:knowledge:recall:source`]}};
 await f.gmController.run(input);await f.gmController.run(input);assert.equal(f.die.count,1);assert.deepEqual(f.resolved,['rk1']);f.cleanup();
});
test('a target Token relinked while GM settlement is delayed cannot grant the old actor result to its replacement',async()=>{
 const f=controllerFixture(),capture=await api.captureWorkbenchRecall({...f,requestId:'target-relink',targetUuids:[f.target.uuid]});
 f.target.actor={...f.target.actor,uuid:'Actor.replacement'};f.target.object.actor=f.target.actor;
 await assert.rejects(()=>f.gmController.settle(capture.message.id,f.user),/目标.*角色|目标.*改变/);
 assert.equal(f.die.count,1);assert.deepEqual(f.resolved,[]);assert.equal(capture.message.flags[MODULE_ID].workbenchRecall.status,'pending');f.cleanup();
});
test('Automatic Knowledge uses the fixed eligible Assurance skill and only actual combat rounds',()=>{
 const f=fixture();const feat={actor:f.actor,_stats:{compendiumSource:AUTOMATIC_KNOWLEDGE_SOURCE},flags:{},system:{rules:[]}},assurance={_stats:{compendiumSource:ASSURANCE_SOURCE},system:{rules:[{key:'ChoiceSet',flag:'assurance',selection:'occultism'}]}};f.actor.items=[feat,assurance];
 assert.equal(automaticKnowledgeChoices(f.actor,feat).statistic,'occultism');assert.throws(()=>automaticKnowledgeRound(f.game),/遭遇/);f.game.combat={id:'c',round:2,started:true};assert.equal(automaticKnowledgeRound(f.game),'c:2');f.actor.skills.occultism.rank=1;assert.deepEqual(automaticKnowledgeChoices(f.actor,feat).choices,[]);
});
import {createKnowledgeAutomation,KNOWLEDGE_SOURCES} from '../scripts/knowledge-automation.mjs';
test('knowledge provider exposes normal and Automatic Knowledge entry points with strict actual-use requirement',()=>{
 const f=fixture(),provider=createKnowledgeAutomation({...f,choose:async()=>{throw Error('ordinary RK must not ask skill');}});
 const rk={actor:f.actor,type:'action',_stats:{compendiumSource:'Compendium.pf2e.actionspf2e.Item.1OagaWtBpVXExToo'}};const automatic={actor:f.actor,type:'feat',_stats:{compendiumSource:AUTOMATIC_KNOWLEDGE_SOURCE}};
 assert.equal(provider.resolveAction(rk),'knowledge:recall');assert.equal(provider.resolveAction(automatic),'knowledge:automatic');assert.equal(provider.requiresActualUse(rk,'knowledge:recall'),true);assert.equal(provider.requiresActualUse(automatic,'knowledge:automatic'),true);
});
test('a target relink after native result publication cannot enter knowledge benefit processing',async()=>{
 const f=fixture(),monster={id:'monster',_stats:{compendiumSource:KNOWLEDGE_SOURCES.monster},type:'feat'};f.actor.items=new Map([[monster.id,monster]]);
 const capture=await api.captureWorkbenchRecall({...f,requestId:'published-relink',targetUuids:[f.target.uuid]}),gmGame={...f.game,user:f.gm};
 await api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:gmGame,message:capture.message});f.target.actor={...f.target.actor,uuid:'Actor.replacement'};f.target.object.actor=f.target.actor;
 const provider=createKnowledgeAutomation({...f,game:gmGame});await provider.processRecall(capture.message);
 assert.equal(capture.message.flags[MODULE_ID].knowledge?.processed,undefined);assert.equal(f.die.count,1);
});
test('HUD render capture takes the real earliest RK click and releases detached application roots',async()=>{
 const f=fixture();f.user.active=true;f.gm.active=true;f.actor.getActiveTokens=()=>[f.token];f.game.user.character=f.actor;
 const callbacks=new Map();const Hooks={on(name,fn){callbacks.set(name,fn);return fn;},off(name){callbacks.delete(name);}};
 const controller=createWorkbenchRecallController({...f});const cleanup=controller.register({Hooks,socket:{register(){},executeAsUser:async()=>({ok:true,value:{messageId:'rk1'}})}});
 const makeRoot=()=>({handlers:new Map(),contains:()=>true,addEventListener(type,fn){this.handlers.set(type,fn);},removeEventListener(type,fn){if(this.handlers.get(type)===fn)this.handlers.delete(type);}});
 const app={actor:f.actor},old=makeRoot(),next=makeRoot();callbacks.get('renderApplicationV2')(app,old);assert.ok(old.handlers.has('click'));callbacks.get('renderApplicationV2')(app,next);assert.equal(old.handlers.size,0);
 let stopped=0;next.handlers.get('click')({target:{closest:()=>({dataset:{action:'roll-statistic-action',key:'recall-knowledge'}})},preventDefault(){},stopImmediatePropagation(){stopped++;}});await new Promise(r=>setImmediate(r));assert.equal(stopped,1);assert.equal(f.die.count,1);
 callbacks.get('closeApplicationV2')(app);assert.equal(next.handlers.size,0);cleanup();
});
test('installed HUD frozen actions API registers safely and real RK click still uses one Workbench die',async()=>{
 const f=fixture();f.user.active=true;f.gm.active=true;f.actor.getActiveTokens=()=>[f.token];f.game.user.character=f.actor;
 // PF2e HUD CustomModule.apiExpose freezes the exposed object and defines its
 // parent property as non-writable/non-configurable. Public actions cannot be patched.
 let nativeCalls=0;const actions=Object.freeze({rollRecallKnowledge:()=>{nativeCalls++;}}),hudApi={};
 Object.defineProperty(hudApi,'actions',{value:actions,configurable:false,enumerable:false,writable:false});f.game.modules.set('pf2e-hud',{active:true,api:hudApi});
 const callbacks=new Map(),Hooks={on(name,fn){callbacks.set(name,fn);return fn;},off(name){callbacks.delete(name);}};
 const controller=createWorkbenchRecallController({...f}),cleanup=controller.register({Hooks,socket:{register(){},executeAsUser:async()=>({ok:true,value:{messageId:'rk1'}})}});
 const root={handlers:new Map(),contains:()=>true,addEventListener(type,fn){this.handlers.set(type,fn);},removeEventListener(type,fn){if(this.handlers.get(type)===fn)this.handlers.delete(type);}};
 callbacks.get('renderApplicationV2')({actor:f.actor},root);let stopped=0;
 root.handlers.get('click')({target:{closest:()=>({dataset:{action:'roll-statistic-action',key:'recall-knowledge'}})},preventDefault(){},stopImmediatePropagation(){stopped++;}});
 await new Promise(resolve=>setImmediate(resolve));assert.equal(stopped,1);assert.equal(nativeCalls,0);assert.equal(f.die.count,1);assert.equal(hudApi.actions,actions);cleanup();assert.equal(root.handlers.size,0);assert.equal(hudApi.actions,actions);
});
test('a no-GM result still saves one secret die and an owner request rejects a forged original card',async()=>{
 const f=controllerFixture();f.gm.active=false;await assert.rejects(()=>f.owner.run({actor:f.actor,token:f.token,targetUuids:[f.target.uuid],requestId:'offlineGM'}),/秘骰已保存/);assert.equal(f.die.count,1);assert.equal(f.game.messages.get('rk1').blind,true);
 const reply=await f.handlers.get('player:knowledge-rk-run').call({socketdata:{userId:'player'}},{requestId:'fake',origin:{messageId:'other'}});assert.equal(reply.ok,false);assert.match(reply.error,/主 GM/);assert.equal(f.die.count,1);f.cleanup();
});
test('invalid native probe receipt fails instead of hanging Workbench callback or silently replaying a die',async()=>{
 const f=fixture();f.actor.skills.society.roll=async options=>{options.callback({options:{totalModifier:13}},null,{flags:{pf2e:{context:{type:'other'}}}});};
 const timeout=new Promise((_,reject)=>setTimeout(()=>reject(Error('native callback hung')),30));
 await assert.rejects(Promise.race([api.captureWorkbenchRecall({...f,requestId:'bad-probe',targetUuids:[f.target.uuid]}),timeout]),/回执/);assert.equal(f.die.count,1);
});
test('incidental native roll options affect captured modifiers before primary selection',async()=>{
 const f=fixture();const original=f.actor.skills.society.roll;f.actor.skills.society.roll=async function(options){this.totalModifier=options.extraRollOptions.includes('origin:item:known-weaknesses')?17:13;return original.call(this,options);};
 const capture=await api.captureWorkbenchRecall({...f,requestId:'origin',targetUuids:[f.target.uuid],origin:{rollOptions:['origin:item:known-weaknesses']}});assert.equal(capture.candidates[0].total,29);
});
const nativeBundle=process.env.FVTT_PF2E_BUNDLE??process.env.FVTT_PF2E_RUNTIME??'';
test('installed PF2e RecallKnowledgeActionVariant bypasses its native player-skill prerequisite through the real prototype', {skip:!fs.existsSync(nativeBundle)},async()=>{
 const bundle=fs.readFileSync(nativeBundle,'utf8'),begin=bundle.indexOf('RecallKnowledgeActionVariant = class extends SingleCheckActionVariant {'),end=bundle.indexOf('}, RecallKnowledgeAction =',begin);assert.ok(begin>=0&&end>begin,'installed RK variant source unavailable');
 const source=bundle.slice(begin+'RecallKnowledgeActionVariant = '.length,end+1);class BaseVariant{get slug(){return 'recall-knowledge';}}
 const Variant=new Function('SingleCheckActionVariant',`return (${source});`)(BaseVariant);const f=fixture();f.user.active=true;f.gm.active=true;f.actor.getActiveTokens=()=>[f.token];f.game.user.character=f.actor;const native={use:()=>{throw Error('native should not require selected skill');},getDefaultVariant:()=>new Variant()};f.game.pf2e={actions:new Map([['recall-knowledge',native]])};
 const controller=createWorkbenchRecallController({...f}),cleanup=controller.register({Hooks:{on(){return 1;},off(){}},socket:{register(){},executeAsUser:async()=>({ok:true,value:{messageId:'rk1'}})}});
 await native.getDefaultVariant().use({actors:[f.actor]});assert.equal(f.die.count,1);assert.equal(f.game.messages.get('rk1').flags[MODULE_ID].workbenchRecall.candidates[0].statistic,'society');cleanup();
});
test('only the persisted automatic GM result settles Monster Hunter once; replay and GM information changes add no benefit',async()=>{
 const f=fixture();f.actor.type='character';f.actor.flags={};f.actor.getActiveTokens=()=>[f.token];
 const monster={id:'monster',_stats:{compendiumSource:KNOWLEDGE_SOURCES.monster},type:'feat'},prey={id:'prey',_stats:{compendiumSource:KNOWLEDGE_SOURCES.prey},type:'effect',system:{rules:[{key:'TokenMark',slug:'hunted-prey',uuid:f.target.uuid}]}};f.actor.items=new Map([[monster.id,monster],[prey.id,prey]]);
 let effects=0;f.actor.update=async changes=>{for(const[k,v]of Object.entries(changes)){let o=f.actor;const ps=k.split('.');for(const p of ps.slice(0,-1))o=o[p]??={};o[ps.at(-1)]=v;}};
 f.actor.createEmbeddedDocuments=async(_type,data)=>{effects+=data.length;return data.map((d,i)=>({...d,id:`effect${i}`}));};
 const native=f.fromUuid;f.fromUuid=async uuid=>uuid===KNOWLEDGE_SOURCES.monsterEffect?{toObject:()=>({type:'effect',system:{rules:[{key:'TokenMark',slug:'monster-hunter'},{key:'FlatModifier',predicate:[]}],start:{},duration:{}},flags:{}})}:native(uuid);
 const OriginalRoll=f.globals.Roll;f.globals.Roll=class extends OriginalRoll{constructor(...args){super(...args);this.total=20;}};
 const capture=await api.captureWorkbenchRecall({...f,requestId:'benefit',targetUuids:[f.target.uuid]});assert.equal(effects,0);
 const gmGame={...f.game,user:f.gm},provider=createKnowledgeAutomation({...f,game:gmGame,choose:async()=>{throw Error('no redundant information recipient choice');}});
 await api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:gmGame,message:capture.message});await provider.processRecall(capture.message);assert.equal(effects,1);
 await provider.processRecall(capture.message);assert.equal(effects,1);await assert.rejects(()=>api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:gmGame,message:capture.message,dc:15}),/锁定/);assert.equal(effects,1);assert.equal(f.die.count,1);
});
test('Automatic Knowledge executes fixed Assurance on original owner and shares one per-round pool across feat copies',async()=>{
 const f=controllerFixture();f.actor.type='character';f.actor.flags={};f.gmGame.combat={id:'c',round:1,started:true};f.actor.skills.occultism.modifiers=[{type:'proficiency',modifier:9},{type:'ability',modifier:4}];
 const update=function(changes){for(const[k,v]of Object.entries(changes)){let o=this;const ps=k.split('.');for(const p of ps.slice(0,-1))o=o[p]??={};o[ps.at(-1)]=v;}return Promise.resolve(this);};f.actor.update=update;
 const assurance={id:'assurance',_stats:{compendiumSource:ASSURANCE_SOURCE},system:{rules:[{key:'ChoiceSet',flag:'assurance',selection:'occultism'}]}};
 const automatic=id=>({id,uuid:`Actor.a.Item.${id}`,actor:f.actor,type:'feat',_stats:{compendiumSource:AUTOMATIC_KNOWLEDGE_SOURCE},system:{rules:[]},flags:{},update});const first=automatic('auto'),second=automatic('auto2');f.actor.items=new Map([[assurance.id,assurance],[first.id,first],[second.id,second]]);
 const card=(id,item)=>({id,author:f.user,actor:f.actor,speaker:{actor:f.actor.id,scene:'s',token:f.token.id},flags:{[MODULE_ID]:{usageInput:{actualUse:true,targetUuids:[f.target.uuid]}}},item,update});const source=card('source',first),source2=card('source2',second);f.game.messages.set(source.id,source);f.game.messages.set(source2.id,source2);
 const provider=createKnowledgeAutomation({...f,game:f.gmGame,choose:async()=>{throw Error('no skill guessing');}}),cleanup=provider.register({Hooks:{on(){return 1;},off(){}},socket:f.socketFor(f.gm)});
 await provider.executeUsage({actor:f.actor,item:first,message:source,user:f.user,action:'knowledge:automatic'});assert.equal(f.die.count,0);assert.equal(f.game.messages.get('rk1').flags[MODULE_ID].workbenchRecall.result.total,19);assert.equal(first.flags[MODULE_ID].knowledge.automaticSkill,'occultism');
 await assert.rejects(()=>provider.executeUsage({actor:f.actor,item:second,message:source2,user:f.user,action:'knowledge:automatic'}),/本轮/);assert.equal(f.die.count,0);
 f.gmGame.combat.round=2;await provider.executeUsage({actor:f.actor,item:second,message:source2,user:f.user,action:'knowledge:automatic'});assert.equal(f.actor.flags[MODULE_ID].knowledge.automaticRound.epoch,'c:2');assert.equal(f.die.count,0);cleanup();f.cleanup();
});
test('a fixed Lore ability preserves its statistic and DC instead of being excluded by ordinary primary policy',async()=>{
 const f=fixture();f.actor.skills['warfare-lore']={...f.actor.skills.occultism,slug:'warfare-lore',label:'Warfare Lore',lore:true};
 const capture=await api.captureWorkbenchRecall({...f,requestId:'fixed-lore',targetUuids:[f.target.uuid],statistic:'warfare-lore',dc:18});
 const result=await api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:{...f.game,user:f.gm},message:capture.message,user:f.gm});assert.equal(result?.statistic,'warfare-lore');assert.equal(result.dc,18);
});
test('a player cannot impersonate the GM by supplying the actual GM User object to finalization',async()=>{
 const f=fixture();const capture=await api.captureWorkbenchRecall({...f,requestId:'gm-spoof',targetUuids:[f.target.uuid]});
 await assert.rejects(()=>api.finalizeWorkbenchRecall({fromUuid:f.fromUuid,game:f.game,message:capture.message,user:f.gm}),/GM/);
});
