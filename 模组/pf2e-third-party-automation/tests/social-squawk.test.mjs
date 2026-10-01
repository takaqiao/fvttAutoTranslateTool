import test from 'node:test';
import assert from 'node:assert/strict';
import {createSocialAutomation} from '../scripts/social-automation.mjs';
import {createReactionChecks,runCheckReactionPipeline,REACTION_CHECK_SOURCES} from '../scripts/reaction-checks.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

function patch(doc,changes){for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let at=doc;for(const key of parts.slice(0,-1))at=at[key]??={};at[parts.at(-1)]=structuredClone(value);}}
function fixture({reaction='squawk',cancel=false,settled=false,change}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:true},users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm}),messages=new Map();
 const game={user:gm,users,messages,actors:new Map(),scenes:new Map(),modules:new Map(),time:{worldTime:100}},counts={dice:0,callbacks:0,finalCards:0,pipelinePublications:0},summaries=[];
 const actor={id:'hero',uuid:'Actor.hero',type:'character',items:new Map(),flags:{},system:{details:{languages:{value:['common']}}},testUserPermission:()=>true,hasCondition:()=>false,getCondition:()=>null,getStatistic:()=>({roll(){throw Error('GM rolled')},check:{domains:['diplomacy']}}),async update(changes){patch(this,changes)}};
 game.actors.set(actor.id,actor);
 const scene={id:'scene',tokens:new Map(),levels:new Map([['level',{}]])},origin={id:'origin',uuid:'Scene.scene.Token.origin',documentName:'Token',parent:scene,actor,_source:{level:'level'},object:{center:{x:0,y:0},distanceTo:()=>5,checkCollision:()=>false}};scene.tokens.set(origin.id,origin);game.scenes.set(scene.id,scene);
 const targets=[40,22,20,10].map((dc,index)=>{
  const recipient={id:`patient-${index}`,uuid:`Actor.patient-${index}`,type:'npc',items:new Map(),flags:{},fear:3,system:{details:{languages:{value:['common']}}},getCondition(name){return name==='frightened'?{value:this.fear}:null},hasCondition:()=>false,getStatistic:()=>({dc:{value:dc}}),async decreaseCondition(){this.fear--},async createEmbeddedDocuments(_kind,rows){return rows.map(row=>{const item={...structuredClone(row),id:`effect-${this.items.size}`,actor:this};this.items.set(item.id,item);return item})}};
  game.actors.set(recipient.id,recipient);const token={id:recipient.id,uuid:`Scene.scene.Token.${recipient.id}`,documentName:'Token',parent:scene,actor:recipient,object:{center:{x:5,y:0},checkCollision:()=>false}};scene.tokens.set(token.id,token);return token;
 });
 const item={id:'alarm',uuid:'Actor.hero.Item.alarm',type:'feat',actor,sourceId:'Compendium.pf2e.feats-srd.Item.6ON8DjFXSMITZleX'},squawk={id:'squawk',uuid:'Actor.hero.Item.squawk',type:'feat',actor,sourceId:REACTION_CHECK_SOURCES.squawk,async toMessage(){const card={id:'paid-squawk',author:gm,speaker:{actor:actor.id},item:this,flags:{pf2e:{origin:{actor:actor.uuid,uuid:this.uuid}}},async update(changes){patch(this,changes)}};messages.set(card.id,card);return card}};
 actor.items.set(item.id,item);actor.items.set(squawk.id,squawk);
 const source={id:'activity',author:player,speaker:{actor:actor.id,scene:scene.id,token:origin.id},flags:{}};messages.set(source.id,source);
 const documents=new Map([actor,origin,...targets,...targets.map(t=>t.actor)].map(doc=>[doc.uuid,doc]));
 const reactions=createReactionChecks({game,fromUuid:async uuid=>documents.get(uuid),choose:async request=>request.choices.some(c=>c.value==='squawk')?reaction:'visible'});
 const roll={_evaluated:true,total:21,dice:[{faces:20,total:12}],options:{degreeOfSuccess:0,totalModifier:9},termOptions:{},toJSON(){return {class:'CheckRoll',formula:'1d20 + 9',total:this.total,evaluated:true,terms:[{class:'Die',faces:20,options:{...this.termOptions,...this.display?{dsnRole:'d20',dsnRoleManaged:true}:{}},results:[{result:this.dice[0].total,active:true,...this.display?{indexThrow:0}:{}}]}],options:structuredClone(this.options)}},async render(){return '<p>12 + 9 = 21</p>'}};
 const provider=createSocialAutomation({game,runNative:async(ctx,request)=>{
  assert.equal(ctx.user,player);assert.equal(request.targetUuid,undefined);assert.equal(request.dc.value,40);assert.equal(Object.hasOwn(request.dc,'statistic'),false);assert.equal(Object.hasOwn(request.dc,'slug'),false);
  const context={actor,token:origin,type:'skill-check',domains:['diplomacy'],dc:request.dc,options:new Set(request.options),createMessage:false};
  const data={author:player.id,speaker:{actor:actor.id,scene:scene.id,token:origin.id},rolls:[],flags:{pf2e:{context:{type:'skill-check',dc:request.dc,options:[...context.options],domains:['diplomacy'],dosAdjustments:{},outcome:'criticalFailure',unadjustedOutcome:'criticalFailure'}}}};
  const raw={toObject:()=>structuredClone(data),updateSource:changes=>Object.assign(data,changes)};let finalCard;
  await runCheckReactionPipeline({game,check:{},context,native:async(_check,_context,_event,callback)=>{if(cancel)return null;counts.dice++;await callback(roll,'criticalFailure',raw);return roll},decide:async()=>{
   const nonce='squawk-test-nonce',selected=await reactions.decideCheckReaction({actorUuid:actor.uuid,tokenUuid:origin.uuid,targetUuid:targets[0].uuid,nonce,type:'skill-check',degree:0,domains:['diplomacy']},player);return selected?{reaction:selected,nonce,actorUuid:actor.uuid}:null;
  },publish:async()=>{counts.pipelinePublications++;throw Error('staged pipeline must not publish')},callback:async(_roll,_outcome,draft)=>{counts.callbacks++;counts.finalCards++;finalCard={...draft.toObject(),id:'native-check',rolls:[roll]};messages.set(finalCard.id,finalCard)}});
  if(!finalCard)return {status:'cancelled'};
  if(settled)await reactions.settleReaction(finalCard);
  change?.({actor,roll,card:finalCard,payment:messages.get('paid-squawk')});
  return {status:'rolled',check:finalCard};
 }});
 return {game,actor,targets,source,item,counts,summaries,run:()=>provider.executeUsage({actor,item,message:source,user:player,action:'social:no-cause-for-alarm'})};
}
async function setup(options,callback){const config=globalThis.CONFIG,Chat=globalThis.ChatMessage;try{globalThis.CONFIG={Canvas:{polygonBackends:{sound:{testCollision:()=>false}}}};const f=fixture(options);globalThis.ChatMessage={getSpeaker:()=>({actor:f.actor.id}),create:async data=>{f.summaries.push(data);return {id:'summary',...data}}};await callback(f)}finally{globalThis.CONFIG=config;globalThis.ChatMessage=Chat}}
const socialEffects=actor=>[...actor.items.values()].filter(item=>item.flags?.[ID]?.kind==='social-alarm-immunity');

test('real GM-paid Squawk pipeline settles shared Alarm once and converts only raw critical failures',()=>setup({},async f=>{
 assert.match(await f.run(),/已完成/);assert.deepEqual(f.counts,{dice:1,callbacks:1,finalCards:1,pipelinePublications:0});
 assert.deepEqual(f.summaries[0].flags[ID].socialAlarm.outcomes.map(o=>o.degree),['failure','failure','success','criticalSuccess']);assert.deepEqual(f.targets.map(t=>t.actor.fear),[3,3,2,1]);assert.deepEqual(f.targets.map(t=>socialEffects(t.actor).length),[1,1,1,1]);
 assert.match(await f.run(),/已结算/);assert.equal(f.counts.dice,1);assert.equal(f.summaries.length,1);assert.deepEqual(f.targets.map(t=>socialEffects(t.actor).length),[1,1,1,1]);
}));
test('a real native publication that already marked Squawk used keeps its same result binding',()=>setup({settled:true},async f=>{
 assert.match(await f.run(),/已完成/);const record=f.actor.flags[ID].reactionChecks.reactions[0];assert.equal(record.state,'used');assert.equal(record.resultMessageId,'native-check');assert.equal(f.counts.dice,1);assert.equal(f.counts.callbacks,1);assert.deepEqual(f.targets.map(t=>socialEffects(t.actor).length),[1,1,1,1]);
}));
test('native Dice So Nice throw cosmetics cannot invalidate an otherwise identical paid original die',()=>setup({change({roll}){roll.display=true}},async f=>{
 assert.match(await f.run(),/已完成/);assert.equal(f.counts.dice,1);assert.equal(f.counts.callbacks,1);assert.deepEqual(f.targets.map(t=>socialEffects(t.actor).length),[1,1,1,1]);
}));
test('owner window cancellation creates no Alarm immunity or final callback',()=>setup({cancel:true},async f=>{
 assert.match(await f.run(),/取消/);assert.deepEqual(f.counts,{dice:0,callbacks:0,finalCards:0,pipelinePublications:0});assert.equal(f.summaries.length,0);assert.deepEqual(f.targets.map(t=>socialEffects(t.actor).length),[0,0,0,0]);
}));
test('declining Squawk preserves the original native critical failure and ordinary Alarm settlement',()=>setup({reaction:'decline'},async f=>{
 assert.match(await f.run(),/已完成/);assert.deepEqual(f.summaries[0].flags[ID].socialAlarm.outcomes.map(o=>o.degree),['criticalFailure','failure','success','criticalSuccess']);assert.equal(f.game.messages.has('paid-squawk'),false);assert.equal(f.counts.dice,1);
}));
for(const changed of ['record','record-user','record-state','used-result','nonce','actor','author','payment-author','payment-source','previous-total','previous-degree','previous-die','die-type','roll-option','dos-adjustment','unadjusted-outcome'])test(`a forged Squawk ${changed} cannot bypass the native Alarm degree proof`,()=>setup({change({actor,roll,card,payment}){
 const proof=card.flags[ID].reactionChecks,record=actor.flags[ID].reactionChecks.reactions[0];
 if(changed==='record')actor.flags[ID].reactionChecks.reactions=[];if(changed==='record-user')record.userId='gm';if(changed==='record-state')record.state='uncertain';if(changed==='used-result'){record.state='used';record.resultMessageId='another-check'}if(changed==='nonce')proof.nonce='forged-nonce';if(changed==='actor')proof.actorUuid='Actor.other';if(changed==='author')card.author='gm';if(changed==='payment-author')payment.author={id:'player'};if(changed==='payment-source')payment.flags.pf2e.origin.uuid='Actor.hero.Item.other';if(changed==='previous-total')proof.previousRoll.total=99;if(changed==='previous-degree')proof.previousRoll.options.degreeOfSuccess=2;if(changed==='previous-die')roll.dice[0].total=11;if(changed==='die-type')roll.termOptions.type='poison';if(changed==='roll-option')roll.options.dsnRole='mechanical-metadata';if(changed==='dos-adjustment')card.flags.pf2e.context.dosAdjustments={all:{amount:1,label:'forged'}};if(changed==='unadjusted-outcome')card.flags.pf2e.context.unadjustedOutcome='failure';
}},async f=>{await assert.rejects(f.run(),/成功度|喀咯|反应/);assert.deepEqual(f.targets.map(t=>socialEffects(t.actor).length),[0,0,0,0]);assert.deepEqual(f.targets.map(t=>t.actor.fear),[3,3,3,3]);assert.equal(f.counts.dice,1);assert.equal(f.counts.callbacks,1);assert.equal(f.summaries.length,0)}));
