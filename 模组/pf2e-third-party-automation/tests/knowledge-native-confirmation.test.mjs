import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as knowledge from '../scripts/knowledge-automation.mjs';
import {createUsageExecutor} from '../scripts/runtime.mjs';
import {createAvAutomation,AV_SOURCES} from '../scripts/av-automation.mjs';
import {createElementalMedicine} from '../scripts/elemental-medicine.mjs';
import {ELEMENTAL_MEDICINE_SOURCE,ELEMENTAL_MEDICINE_DAILY} from '../scripts/elemental-medicine-rules.mjs';
import {MODULE_ID,SOURCES} from '../scripts/rules.mjs';
function updateFlags(document,changes){for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let current=document;for(const key of parts.slice(0,-1))current=current[key]??={};current[parts.at(-1)]=value;}return document;}
function context(){const gm={id:'gm',isGM:true,active:true,settings:{showCheckDialogs:false}},player={id:'player',active:true};const users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm});const game={user:gm,users,time:{worldTime:100},messages:new Map(),actors:new Map(),scenes:new Map(),modules:new Map()};return {gm,player,game};}
test('Devise confirms a zero-modifier native check but preserves its selected d20 instead of a dialog bonus',async()=>{
 assert.equal(typeof knowledge.rollKnowledgeD20,'function');const {game}=context(),actor={uuid:'Actor.hero'},token={uuid:'Scene.scene.Token.hero'},item={name:'出谋划策'};let rolls=0,published=0;
 class CheckModifier{constructor(slug,{modifiers}){this.slug=slug;this.modifiers=modifiers;}}
 game.pf2e={CheckModifier,Check:{async roll(check,parameters,event){assert.deepEqual(check.modifiers,[]);assert.equal(parameters.skipDialog,false);assert.equal(parameters.type,'check');assert.equal(parameters.createMessage,false);assert.equal(parameters.dc,undefined);assert.equal(event,null);rolls++;return {total:17,options:{totalModifier:5},dice:[{faces:20,total:12,results:[{result:12,active:true}]}]};}}};
 const globals={Roll:{fromTerms(terms){return {total:terms[0].total,dice:terms,toMessage(){published++;}};}}};
 const die=await knowledge.rollKnowledgeD20({game,actor,token,item,globals});assert.equal(rolls,1);assert.equal(die.total,12);assert.equal(published,0);
});
test('Devise native cancellation returns no die or card for its downstream strategy and frequency steps',async()=>{
 assert.equal(typeof knowledge.rollKnowledgeD20,'function');const {game}=context();let rawRolls=0;game.pf2e={CheckModifier:class{},Check:{async roll(){return null;}}};
 assert.equal(await knowledge.rollKnowledgeD20({game,actor:{uuid:'Actor.hero'},globals:{Roll:{fromTerms(){rawRolls++;}}}}),null);assert.equal(rawRolls,0);
});
const nativeBundle=process.env.FVTT_PF2E_BUNDLE??'';
test('installed native Check builds the complete self context and opens Devise confirmation despite disabled dialog preferences',{skip:!fs.existsSync(nativeBundle)},async()=>{
 const bundle=fs.readFileSync(nativeBundle,'utf8'),start=bundle.indexOf('static async roll(e, t = {}, n = null, r) {',bundle.indexOf('Sa = class Check {')),end=bundle.indexOf('let s = [], c = t.isReroll',start);assert.ok(start>=0&&end>start);
 const {game}=context();game.settings={get:()=> 'public'};game.pf2e={settings:{metagame:{secretChecks:false}}};const actor={id:'hero',uuid:'Actor.hero'},item={name:'出谋划策'};let dialogs=0;
 class CheckModifiersDialog{constructor(_check,resolve,context){this.resolve=resolve;this.context=context;}render(){dialogs++;assert.equal(this.context.origin.self,true);assert.equal(this.context.origin.item,item);this.resolve(true);}}
 const prefix=bundle.slice(start+'static async roll(e, t = {}, n = null, r) {'.length,end);
 const enter=new Function('game','foundry','CONFIG','CheckModifiersDialog','objectHasKey',`return async function(e,t,n){${prefix};return true;};`)(game,{utils:{mergeObject:Object.assign}},{ChatMessage:{modes:{public:0,blind:1}}},CheckModifiersDialog,(object,key)=>Object.hasOwn(object,key));
 game.pf2e.CheckModifier=class{constructor(){this.modifiers=[];}calculateTotal(){}};game.pf2e.Check={async roll(check,parameters,event){assert.equal(await enter(check,parameters,event),true);return {dice:[{total:12}]};}};
 const result=await knowledge.rollKnowledgeD20({game,actor,item,globals:{Roll:{fromTerms:terms=>({total:terms[0].total})}}});assert.equal(result.total,12);assert.equal(dialogs,1);
});
test('Strategist Stance native cancellation has no cooldown or effect even when roll dialogs were disabled',async()=>{
 const {game,player}=context();game.combat={started:true};let writes=0;
 const statistic={check:{async roll(parameters){assert.equal(parameters.skipDialog,false);assert.equal(parameters.event,null);return null;}}};
 const actor={id:'hero',uuid:'Actor.hero',type:'character',level:5,flags:{},skills:{society:{label:'社群'}},itemTypes:{lore:[]},testUserPermission:()=>true,getStatistic:()=>statistic,async update(){writes++;},createEmbeddedDocuments(){writes++;}};
 const item={id:'stance',actor,sourceId:knowledge.KNOWLEDGE_SOURCES.stance};const provider=knowledge.createKnowledgeAutomation({game});
 const result=await provider.executeUsage({actor,item,message:{id:'source'},user:player,action:'knowledge:stance'});assert.match(result,/未进入/);assert.equal(writes,0);
});
test('Circadian rest cancellation leaves recovery state and health untouched after requiring its native window',async t=>{
 const {game,player}=context(),oldGame=globalThis.game;t.after(()=>{globalThis.game=oldGame});globalThis.game=game;let writes=0;
 const actor={uuid:'Actor.hero',type:'character',level:3,flags:{},items:[{sourceId:SOURCES.circadian}],testUserPermission:()=>true,skills:{survival:{async roll(parameters){assert.equal(parameters.skipDialog,false);assert.equal(parameters.event,null);return null;}}},async update(){writes++;}};
 assert.match(await createUsageExecutor()({actor,item:actor.items[0],message:{id:'source'},user:player,action:'rest'}),/未完成/);assert.equal(writes,0);
});
test('Shake It Off native cancellation preserves the source sickened condition while retaining its earlier frightened reduction',async()=>{
 const {game,player}=context();const sickened={id:'sickened',flags:{'patreon-v3':{dc:20}}},frightened={id:'frightened'},decreases=[];
 const actor={uuid:'Actor.hero',type:'character',flags:{},items:new Map([['rage',{type:'effect',slug:'rage'}],[sickened.id,sickened]]),testUserPermission:()=>true,getCondition:slug=>slug==='frightened'?frightened:slug==='sickened'?sickened:null,async decreaseCondition(value){decreases.push(value);},async update(changes){return updateFlags(this,changes);},saves:{fortitude:{async roll(parameters){assert.equal(parameters.skipDialog,false);assert.equal(parameters.event,null);return null;}}}};
 const item={id:'shake',uuid:'Actor.hero.Item.shake',type:'feat',sourceId:AV_SOURCES.shake,actor,system:{}};const castEvents={addMatcher(){},addCapture(){}};
 const provider=createAvAutomation({game,castEvents});const result=await provider.executeUsage({actor,item,message:{id:'shake-card',flags:{}},user:player,action:'av:shake'});assert.match(result,/未完成/);assert.deepEqual(decreases,['frightened']);
});
test('Elemental Medicine does not open a GM window when the original player connection is missing',async()=>{
 const {game,gm,player}=context();let diagnoses=0,medicines=0,checks=0;
 const patient={uuid:'Actor.patient',type:'character',level:1,items:new Map(),testUserPermission:()=>true};
 const doctor={id:'doctor',uuid:'Actor.doctor',type:'character',items:new Map(),flags:{},testUserPermission:()=>true,async update(changes){return updateFlags(this,changes);},getStatistic:slug=>slug==='medicine'?{check:{async roll(parameters){checks++;assert.equal(parameters.skipDialog,false);assert.equal(parameters.event,null);assert.equal(parameters.dc.visible,false);assert.equal(parameters.messageMode,'blind');return null;}}}:null};
 doctor.items.set('feat',{id:'feat',type:'feat',sourceId:ELEMENTAL_MEDICINE_SOURCE});
 const request={id:'request',uuid:'Actor.doctor.Item.request',actor:doctor,flags:{'pf2e-dailies':{daily:`module.${ELEMENTAL_MEDICINE_DAILY}`},[MODULE_ID]:{elementalMedicine:{kind:'preparation',actorUuid:doctor.uuid,userId:player.id,status:'diagnosing',factsId:'facts',patients:[{patientUuid:patient.uuid,skill:'medicine',state:'pending'}]}}},async update(changes){return updateFlags(this,changes);}};doctor.items.set(request.id,request);
 const facts={id:'facts',author:gm,blind:true,whisper:[gm.id],flags:{[MODULE_ID]:{elementalMedicineFacts:{requestUuid:request.uuid,facts:[{patientUuid:patient.uuid,skill:'medicine',dc:15,binding:{itemUuid:'Actor.patient.Item.affliction'},correctElement:'wood',wrongElement:'earth'}]}}}};game.messages.set(facts.id,facts);game.actors.set(doctor.id,doctor);
 patient.createEmbeddedDocuments=async()=>{medicines++;};const provider=createElementalMedicine({game,fromUuid:async uuid=>uuid===request.uuid?request:uuid===patient.uuid?patient:null,publishDiagnosis:async()=>{diagnoses++;}});provider.register({Hooks:{on(){return 1;},off(){}}});
 await assert.rejects(()=>provider.prepare(request.uuid,player),/原日备操作者连接/);assert.equal(checks,0);assert.equal(diagnoses,0);assert.equal(medicines,0);assert.equal(request.flags[MODULE_ID].elementalMedicine.status,'uncertain');
});
