import test from 'node:test';
import assert from 'node:assert/strict';
import {createKnowledgeAutomation,KNOWLEDGE_SOURCES} from '../scripts/knowledge-automation.mjs';
import {createSocialAutomation} from '../scripts/social-automation.mjs';
import {createReactionChecks,REACTION_CHECK_SOURCES} from '../scripts/reaction-checks.mjs';
import {createUsageExecutor} from '../scripts/runtime.mjs';
import {SOURCES} from '../scripts/rules.mjs';

function fixture(){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:true};
 const users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm});
 const game={user:gm,users,time:{worldTime:100},messages:new Map(),scenes:new Map(),modules:new Map(),combat:{started:true}};
 const actor={id:'hero',uuid:'Actor.hero',type:'character',level:5,items:new Map(),flags:{},itemTypes:{lore:[]},skills:{society:{label:'社群'}},system:{details:{languages:{value:['common']}}},testUserPermission:()=>true,getCondition:()=>null,hasCondition:()=>false,getStatistic:()=>({roll(){throw Error('GM rolled');},check:{roll(){throw Error('GM rolled');},domains:[]}}),async update(){throw Error('Unexpected settlement');}};
 const scene={id:'scene',tokens:new Map(),levels:new Map([['level',{}]])},origin={id:'origin',uuid:'Scene.scene.Token.origin',parent:scene,actor,_source:{level:'level'},object:{center:{x:0,y:0},distanceTo:()=>5,checkCollision:()=>false}};
 const recipient={id:'target',uuid:'Actor.target',type:'npc',items:new Map(),system:{details:{languages:{value:['common']}}},getCondition:()=>({value:1}),hasCondition:()=>false,isAllyOf:()=>false,getStatistic:()=>({dc:{value:20}})};
 const target={id:'target',uuid:'Scene.scene.Token.target',documentName:'Token',parent:scene,actor:recipient,object:{center:{x:5,y:0}}};
 scene.tokens.set(origin.id,origin);scene.tokens.set(target.id,target);game.scenes.set(scene.id,scene);
 const message={id:'source',author:player,speaker:{actor:actor.id,scene:scene.id,token:origin.id},flags:{}};game.messages.set(message.id,message);
 return {game,gm,player,actor,scene,origin,target,message};
}

test('Strategist Stance sends its untargeted native check to the source player and cancellation does not settle',async()=>{
 const f=fixture(),item={id:'stance',uuid:'Actor.hero.Item.stance',actor:f.actor,sourceId:KNOWLEDGE_SOURCES.stance};let calls=0;
 const provider=createKnowledgeAutomation({game:f.game,runNative:async(ctx,request)=>{calls++;assert.equal(ctx.user,f.player);assert.equal(ctx.message,f.message);assert.equal(request.type,'check');assert.equal(request.statistic,'society');assert.equal(request.targetUuid,undefined);return {status:'cancelled'};}});
 assert.match(await provider.executeUsage({...f,item,user:f.player,action:'knowledge:stance'}),/未进入|取消/);assert.equal(calls,1);
});

test('No Cause for Alarm sends shared Diplomacy to its player without borrowing the GM target',async t=>{
 const f=fixture(),old=globalThis.CONFIG;t.after(()=>globalThis.CONFIG=old);globalThis.CONFIG={Canvas:{polygonBackends:{sound:{testCollision:()=>false}}}};
 f.actor.update=async()=>{};const item={id:'alarm',uuid:'Actor.hero.Item.alarm',type:'feat',actor:f.actor,sourceId:'Compendium.pf2e.feats-srd.Item.6ON8DjFXSMITZleX'};let calls=0;
 const provider=createSocialAutomation({game:f.game,runNative:async(ctx,request)=>{calls++;assert.equal(ctx.user,f.player);assert.equal(request.statistic,'diplomacy');assert.equal(request.targetUuid,undefined);assert.equal(request.dc.visible,false);return {status:'cancelled'};}});
 const result=await provider.executeUsage({...f,item,user:f.player,action:'social:no-cause-for-alarm'});assert.match(result,/取消|未完成/);assert.equal(calls,1);
});

test('Pointed Question routes its native check with the original source and target',async()=>{
 const f=fixture();f.actor.update=async()=>{};f.message.flags.pf2e={context:{target:{token:f.target.uuid}}};let calls=0;
 const item={id:'pointed',uuid:'Actor.hero.Item.pointed',type:'action',actor:f.actor,sourceId:REACTION_CHECK_SOURCES.pointed};
 const provider=createReactionChecks({game:f.game,fromUuid:async uuid=>uuid===f.target.uuid?f.target:null,runNative:async(ctx,request)=>{calls++;assert.equal(ctx.user,f.player);assert.equal(request.statistic,'diplomacy');assert.equal(request.targetUuid,f.target.uuid);assert.equal(request.tokenUuid,f.origin.uuid);return {status:'cancelled'};}});
 const result=await provider.executeUsage({...f,item,user:f.player,action:'reaction-checks:pointed-question'});assert.match(result,/取消|未完成/);assert.equal(calls,1);
});

test('Partial rest sends Survival to the source player and cancellation keeps recovery untouched',async t=>{
 const f=fixture(),old=globalThis.game;t.after(()=>globalThis.game=old);globalThis.game=f.game;const item={sourceId:SOURCES.circadian};f.actor.items=[item];f.actor.skills.survival={roll(){throw Error('GM rolled');}};let calls=0;
 const execute=createUsageExecutor({runNative:async(ctx,request)=>{calls++;assert.equal(ctx.user,f.player);assert.equal(request.statistic,'survival');return {status:'cancelled'};}});
 assert.match(await execute({...f,item,user:f.player,action:'rest'}),/未完成|取消/);assert.equal(calls,1);
});

for(const trusted of [true,false])test(`staged owner check ${trusted?'retains':'does not grant an unauthenticated draft'} native failure reactions`,async()=>{
 const f=fixture();f.game.user=f.player;f.game.actors=new Map([[f.actor.id,f.actor]]);f.actor.items.set('clock',{id:'clock',uuid:`${f.actor.uuid}.Item.clock`,actor:f.actor,type:'feat',sourceId:REACTION_CHECK_SOURCES.clock});
 let wrapper,rpcs=0,callbacks=0,updates=0,natives=0;
 const provider=createReactionChecks({game:f.game,nativeInvocation:()=>trusted?{actorUuid:f.actor.uuid,tokenUuid:f.origin.uuid,user:f.player,targetUuid:f.target.uuid}:null});
 provider.register({Hooks:{on:()=>1,off(){}},libWrapper:{register(_id,path,fn){if(path==='game.pf2e.Check.roll')wrapper=fn;}},socket:{register(){},async executeAsUser(){rpcs++;return {ok:true,value:null};}}});
 const roll={options:{degreeOfSuccess:1},toJSON:()=>({evaluated:true,total:10})};
 const raw={flags:{pf2e:{context:{type:'skill-check',outcome:'failure',options:[]}}},toObject(){return {flags:structuredClone(this.flags),rolls:[]};},updateSource(){updates++;}};
 const context={actor:f.actor,token:f.origin,type:'skill-check',domains:['diplomacy'],options:new Set(['native-owner-marker']),createMessage:false};
 await wrapper(async(_check,_context,_event,callback)=>{natives++;await callback(roll,'failure',raw);return roll;},{modifiers:[]},context,null,()=>{callbacks++;});
 assert.equal(natives,1);assert.equal(callbacks,1);assert.equal(rpcs,trusted?1:0);assert.equal(updates,trusted?1:0);
});
