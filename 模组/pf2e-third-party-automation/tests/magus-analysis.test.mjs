import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {MODULE_ID} from '../scripts/rules.mjs';
import {Predicate,degree as nativeDegree} from './knowledge-pf2e-861-fixture.mjs';
import {createWorkbenchRecallController} from '../scripts/knowledge-entrypoints.mjs';
import {interceptKnowledgeProbe} from '../scripts/knowledge-probes.mjs';

let api={};
try{api=await import('../scripts/magus-analysis.mjs');}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
const source='Compendium.pf2e.feats-srd.Item.0yPbPVEESwB6Bdfw';
function document(data){return {...data,writes:[],async update(changes){const serialized=JSON.parse(JSON.stringify(changes));this.writes.push(serialized);for(const[path,value]of Object.entries(serialized)){const parts=path.split('.');let object=this;for(const part of parts.slice(0,-1))object=object[part]??={};object[parts.at(-1)]=value;}return this;}};}
function fixture({hit='yes',statistic='arcana',degree=1,ownerIsGM=false}={}){
 const gm={id:'gm',isGM:true,active:true},player=ownerIsGM?gm:{id:'player',active:true},users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm});
 const actor=document({id:'hero',uuid:'Actor.hero',type:'character',flags:{[MODULE_ID]:{spellstrike:{charged:false,messageId:'strike'}}},skills:{arcana:{label:'奥法'},nature:{label:'自然'},'warfare-lore':{label:'战争学识',lore:true}},testUserPermission:user=>users.has(user.id)});
 const item={id:'analysis',uuid:actor.uuid+'.Item.analysis',type:'feat',actor,sourceId:source,system:{rules:[{key:'FlatModifier',predicate:['maguss-analysis'],selector:'skill-check',type:'circumstance',value:1}]},getOriginData:()=>({rollOptions:['origin:item:maguss-analysis']})};
 actor.items=new Map([[item.id,item],['spellstrike',{id:'spellstrike',type:'action',actor,sourceId:'Compendium.pf2e.actionspf2e.Item.QDW9H8XLIjuW2fE4'}]]);
 const scene={id:'scene',tokens:new Map()},targetActor=document({id:'enemy',uuid:'Actor.enemy',type:'npc',flags:{}}),token={id:'hero',uuid:'Scene.scene.Token.hero',documentName:'Token',actor,parent:scene},target={id:'enemy',uuid:'Scene.scene.Token.enemy',documentName:'Token',actor:targetActor,parent:scene};
 scene.tokens.set(token.id,token);scene.tokens.set(target.id,target);actor.getActiveTokens=()=>[token];
 const game={world:{id:'bob-2'},user:gm,users,time:{worldTime:1000},messages:new Map(),actors:new Map([[actor.id,actor],[targetActor.id,targetActor]]),scenes:new Map([[scene.id,scene]]),i18n:{lang:'cn'}};
 const f={game,actor,item,token,target,targetActor,player,gm,choices:[],rolls:0,nativeInputs:[]};
 f.card=id=>{const message=document({id,actor,author:player,speaker:{actor:actor.id,token:token.id,scene:scene.id},flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid,type:'feat'}},[MODULE_ID]:{usageInput:{actualUse:true,targetUuids:[target.uuid]}}}});game.messages.set(id,message);return message;};
 f.message=f.card('use1');f.context=()=>({actor,item,message:f.message,user:player,action:'magus-analysis'});
 f.fromUuid=async uuid=>[actor,item,token,target,targetActor].find(doc=>doc.uuid===uuid);
 f.choose=async request=>{f.choices.push(request);return request.choices.some(choice=>choice.value==='yes')?hit:statistic;};
 f.runRecallKnowledge=async input=>{
  f.rolls++;f.nativeInputs.push(input);
  const check=document({id:'rk'+f.rolls,actor,author:player,blind:true,whisper:[gm.id],rolls:[{total:degree>=2?17:3}],speaker:{actor:actor.id,scene:scene.id,token:token.id},flags:{pf2e:{context:{type:'skill-check',options:['action:recall-knowledge','secret',...input.origin.rollOptions],outcome:['criticalFailure','failure','success','criticalSuccess'][degree]}},[MODULE_ID]:{workbenchRecall:{schema:1,status:'done',resolved:true,requestId:input.requestId,actorUuid:actor.uuid,tokenUuid:token.uuid,userId:player.id,targetUuids:input.targetUuids,targetActors:[{tokenUuid:target.uuid,actorUuid:target.actor.uuid}],origin:input.origin,statistic:input.statistic,die:degree>=2?17:3,probeUse:{status:'done'}}}}});
  game.messages.set(check.id,check);return {messageId:check.id};
 };
 f.provider=()=>{assert.equal(typeof api.createMagusAnalysis,'function');return api.createMagusAnalysis({game,fromUuid:f.fromUuid,choose:f.choose,runRecallKnowledge:(...args)=>f.runRecallKnowledge(...args)});};
 return f;
}

for(const degree of [0,1,2,3])test(`a completed native Recall Knowledge degree ${degree} recharges and grants one day of immunity`,async()=>{
 const f=fixture({degree}),provider=f.provider();await provider.executeUsage(f.context());
 assert.deepEqual(f.actor.flags[MODULE_ID].spellstrike,{charged:true,messageId:'use1'});
 assert.equal(f.targetActor.flags[MODULE_ID].magusAnalysisImmunity.until,87400);
 assert.equal(f.rolls,1);assert.equal(f.nativeInputs[0].statistic,'arcana');
 assert.deepEqual(f.nativeInputs[0].targetUuids,[f.target.uuid]);
 assert.equal(f.message.flags[MODULE_ID].knowledge.recall.userId,f.player.id);
});

test('only explicit confirmation of the earlier hit enables the existing circumstance modifier',async()=>{
 for(const [hit,expected]of [['yes',true],['no',false]]){
  const f=fixture({hit,statistic:'warfare-lore'});await f.provider().executeUsage(f.context());
  const options=f.nativeInputs[0].origin.rollOptions;
  assert.equal(new Predicate(...f.item.system.rules[0].predicate).test(new Set(options)),expected);
  assert.equal(options.filter(option=>option==='maguss-analysis').length,expected?1:0);
  assert.equal(f.nativeInputs[0].statistic,'warfare-lore');
 }
});

test('closing either necessary choice leaves charging and immunity untouched',async()=>{
 for(const selection of [{hit:null},{statistic:null}]){
  const f=fixture(selection);const result=await f.provider().executeUsage(f.context());
  assert.equal(result.status,'cancelled');assert.equal(f.rolls,0);
  assert.equal(f.actor.flags[MODULE_ID].spellstrike.charged,false);assert.equal(f.targetActor.writes.length,0);
 }
});

test('native cancellation never recharges and the same activity cannot roll again',async()=>{
 const f=fixture();f.runRecallKnowledge=async()=>{f.rolls++;throw Error('原生回忆知识技能检定已取消。');};
 await assert.rejects(()=>f.provider().executeUsage(f.context()),/取消/);
 await assert.rejects(()=>f.provider().executeUsage(f.context()),/已开始|重复|未完成/);
 assert.equal(f.rolls,1);assert.equal(f.actor.flags[MODULE_ID].spellstrike.charged,false);assert.equal(f.targetActor.writes.length,0);
});

test('concurrent and reloaded processing of one Use card cannot recharge a later spent Spellstrike',async()=>{
 const f=fixture(),provider=f.provider();await Promise.all([provider.executeUsage(f.context()),provider.executeUsage(f.context())]);
 await f.actor.update({[`flags.${MODULE_ID}.spellstrike`]:{charged:false,messageId:'later-strike'}});
 await f.provider().executeUsage(f.context());
 assert.equal(f.rolls,1);assert.equal(f.actor.flags[MODULE_ID].spellstrike.charged,false);
 assert.equal(f.targetActor.writes.length,1);
});

test('immunity rejects a new activity through 24 hours and expires on world time',async()=>{
 const f=fixture();await f.provider().executeUsage(f.context());
 f.message=f.card('use2');f.game.time.worldTime=87399;
 await assert.rejects(()=>f.provider().executeUsage(f.context()),/免疫/);assert.equal(f.rolls,1);
 f.game.time.worldTime=87400;await f.provider().executeUsage(f.context());
 assert.equal(f.rolls,2);assert.equal(f.targetActor.flags[MODULE_ID].magusAnalysisImmunity.until,173800);
});

test('showing a feat card, unrelated sources, and other worlds do not activate Analysis',async()=>{
 const f=fixture(),provider=f.provider();assert.equal(provider.resolveAction(f.item),'magus-analysis');assert.equal(provider.requiresActualUse(f.item,'magus-analysis'),true);
 f.message.flags[MODULE_ID].usageInput.actualUse=false;await assert.rejects(()=>provider.executeUsage(f.context()),/使用/);
 assert.equal(provider.resolveAction({...f.item,sourceId:'other'}),null);
 f.game.world.id='other';assert.equal(provider.resolveAction(f.item),null);
 assert.equal(f.rolls,0);assert.equal(f.actor.writes.length,0);
});

test('a foreign or public Recall Knowledge receipt cannot charge the source actor',async()=>{
 for(const mutate of [check=>check.author={id:'other'},check=>check.blind=false,check=>check.flags[MODULE_ID].workbenchRecall.targetActors[0].actorUuid='Actor.other']){
  const f=fixture(),native=f.runRecallKnowledge;f.runRecallKnowledge=async input=>{const result=await native(input);mutate(f.game.messages.get(result.messageId));return result;};
  await assert.rejects(()=>f.provider().executeUsage(f.context()),/回执|匹配|暗骰/);
  assert.equal(f.actor.flags[MODULE_ID].spellstrike.charged,false);assert.equal(f.targetActor.writes.length,0);
 }
});

function attachWorkbench(f,{cancel=false}={}){
 const command=fs.readFileSync(process.env.FVTT_WORKBENCH_RECALL_MACRO??new URL('./fixtures/workbench-7.7.5-recall.txt',import.meta.url),'utf8');
 const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
 const macro={uuid:'Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros-internal.Macro.xcFr7PWwG5OVALNJ',type:'script',command,execute:scope=>new AsyncFunction(...Object.keys(scope),command)(...Object.values(scope))};
 if(process.env.FVTT_MAGUS_ANALYSIS_CURRENT_ACTOR){
  const raw=JSON.parse(fs.readFileSync(process.env.FVTT_MAGUS_ANALYSIS_CURRENT_ACTOR,'utf8'));
  const current=raw.items.find(item=>item._stats?.compendiumSource===source);
  assert.ok(current);f.item.system=structuredClone(current.system);
 }
 const fromUuid=f.fromUuid;f.fromUuid=async uuid=>uuid===macro.uuid?macro:fromUuid(uuid);
 Object.assign(f.game,{system:{id:'pf2e'},settings:{get:()=> 'none'},modules:new Map([['xdy-pf2e-workbench',{active:true,version:'7.7.5'}]])});
 Object.assign(f.target.actor,{name:'Enemy',level:5,traits:new Set(['dragon']),rarity:'common',getSelfRollOptions:()=>['target:trait:dragon'],isOfType:()=>true});
 f.actor.name='Magus';f.actor.itemTypes={feat:[f.item],lore:[]};f.actor.rules=[];f.actor.synthetics={degreeOfSuccessAdjustments:{}};
 f.token.object={document:f.token,actor:f.actor};f.target.object={document:f.target,actor:f.target.actor};
 class Messages{
  constructor(data){Object.assign(this,data);}
  getFlag(ns,path){return path.split('.').reduce((value,key)=>value?.[key],this.flags[ns]);}
  static getWhisperRecipients(){return [f.gm];}
  static async create(data){const message=document({...data,id:'native-rk',actor:f.actor,author:f.player});message.delete=async()=>f.game.messages.delete(message.id);f.game.messages.set(message.id,message);return message;}
 }
 class Roll{
  constructor(formula){this.total=formula==='1d20'?3:Number(formula);this.dice=formula==='1d20'?[{faces:20,total:3,results:[{result:3,active:true}]}]:[];this.terms=this.dice;this.options={};}
  async roll(){f.rolls++;return this;}
  async evaluate(){return this;}
  static fromTerms(dice){const roll=new Roll('1d20');roll.dice=dice;roll.terms=dice;roll.total=dice.reduce((sum,die)=>sum+die.total,0);return roll;}
 }
 const globals={ChatMessage:Messages,Roll,CONST:{DICE_ROLL_MODES:{BLIND:'blindroll'},CHAT_MESSAGE_STYLES:{OTHER:0}},CONFIG:{PF2E:{abilities:{}}},ui:{notifications:{info(){}}},document:{createElement(){throw Error('unexpected DOM');}}};
 const native=async(check,context,_event,callback)=>{
  if(cancel)return null;
  assert.equal(context.messageMode,'blind');assert.equal(context.createMessage,false);
  check.calculateTotal(context.options);const roll=await new Roll('1d20').roll();roll.total+=check.totalModifier;roll.options.totalModifier=check.totalModifier;
  roll.options.degreeOfSuccess=nativeDegree({total:roll.total,die:3,dc:context.dc.value,rollOptions:[...context.options],dosAdjustments:[]});
  await callback(roll,undefined,new Messages({flags:{pf2e:{context:{type:'skill-check',actor:f.actor.id,token:f.token.id,domains:context.domains,options:[...context.options],traits:context.traits,dc:context.dc,outcome:['criticalFailure','failure','success','criticalSuccess'][roll.options.degreeOfSuccess]}}}}));
  return roll;
 };
 f.game.pf2e={Predicate,Check:{roll:(...args)=>interceptKnowledgeProbe(native,...args)}};
 f.actor.skills=Object.fromEntries(['arcana','crafting','medicine','nature','occultism','religion','society'].map(slug=>[slug,{slug,label:slug,rank:1,modifiers:[],async roll(parameters){
  const modifier=f.item.system.rules.find(rule=>rule.key==='FlatModifier');
  const check={slug,modifiers:[],calculateTotal(options){this.totalModifier=5+(new Predicate(...modifier.predicate).test(options)?modifier.value:0);}};
  const context={actor:f.actor,origin:{actor:f.actor,token:f.token},token:f.token,type:'skill-check',domains:['skill-check',slug],dc:parameters.dc,options:new Set(parameters.extraRollOptions),dosAdjustments:[],createMessage:false};
  return interceptKnowledgeProbe(native,check,context,null,parameters.callback);
 }}]));
 const controller=createWorkbenchRecallController({game:f.game,fromUuid:f.fromUuid,globals});
 f.runRecallKnowledge=controller.run;
 return controller;
}

test('the real Workbench controller and macro preserve a failed secret attempt and settle Analysis once',async()=>{
 const f=fixture({ownerIsGM:true});attachWorkbench(f);await f.provider().executeUsage(f.context());
 const check=f.game.messages.get('native-rk'),recall=check.flags[MODULE_ID].workbenchRecall;
 assert.equal(f.rolls,1);assert.equal(check.blind,true);assert.deepEqual(check.whisper,['gm']);
 assert.equal(recall.result.degree,0);assert.equal(recall.result.total,9);assert.equal(recall.resolved,true);
 assert.equal(f.actor.flags[MODULE_ID].spellstrike.charged,true);assert.equal(f.targetActor.flags[MODULE_ID].magusAnalysisImmunity.until,87400);
 assert.equal(f.message.flags[MODULE_ID].knowledge.recall.workbenchOperation.messageId,'native-rk');
});

test('the real Workbench controller null roll removes its reservation without charging or immunity',async()=>{
 const f=fixture({ownerIsGM:true});attachWorkbench(f,{cancel:true});
 await assert.rejects(()=>f.provider().executeUsage(f.context()),/取消/);
 assert.equal(f.rolls,0);assert.equal(f.game.messages.has('native-rk'),false);
 assert.equal(f.actor.flags[MODULE_ID].spellstrike.charged,false);assert.equal(f.targetActor.writes.length,0);
});
