import test from 'node:test';
import assert from 'node:assert/strict';
import {createKnowledgeAutomation,KNOWLEDGE_SOURCES} from '../scripts/knowledge-automation.mjs';

function fixture({prepared=true,embedded=false,owner=true,selection='warfare-lore'}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:true};
 const users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm});
 const game={user:gm,users,time:{worldTime:100},messages:new Map(),scenes:new Map(),modules:new Map(),combat:{started:true}};
 const check={roll(){throw Error('GM must not roll for the player')}};
 const skills={society:{label:'社群',check},...(prepared?{'warfare-lore':{slug:'warfare-lore',label:'战争学识',lore:true,check}}:{})};
 let writes=0;
 const actor={id:'hero',uuid:'Actor.hero',type:'character',level:5,items:new Map(),flags:{},skills,
  itemTypes:{lore:embedded?[{slug:'warfare-lore',name:'战争学识'}]:[]},
  testUserPermission:user=>owner&&user===player,getStatistic:slug=>skills[slug],async update(){writes++;}};
 const item={id:'stance',uuid:'Actor.hero.Item.stance',actor,sourceId:KNOWLEDGE_SOURCES.stance};
 const message={id:'source',author:player,flags:{}},requests=[],prompts=[];
 const provider=createKnowledgeAutomation({game,choose:async prompt=>{prompts.push(prompt);return selection},runNative:async(ctx,request)=>{requests.push({ctx,request});return {status:'cancelled'}}});
 return {game,actor,item,message,player,requests,prompts,writes:()=>writes,run:()=>provider.executeUsage({actor,item,message,user:player,action:'knowledge:stance'})};
}

for(const embedded of [false,true])test(`Strategist Stance offers prepared Warfare Lore with embedded item ${embedded}`,async()=>{
 const f=fixture({embedded});await f.run();
 assert.deepEqual(f.prompts[0]?.choices,[{value:'society',label:'社群'},{value:'warfare-lore',label:'战争学识'}]);
 assert.equal(f.requests.length,1);assert.equal(f.requests[0].request.statistic,'warfare-lore');
 assert.equal(f.requests[0].ctx.user,f.player);assert.equal(f.requests[0].ctx.message,f.message);assert.equal(f.writes(),0);
});

test('an embedded Warfare Lore item without a prepared statistic does not authorize a check',async()=>{
 const f=fixture({prepared:false,embedded:true});await f.run();
 assert.equal(f.prompts.length,0);assert.equal(f.requests[0].request.statistic,'society');assert.equal(f.writes(),0);
});

test('another prepared Lore cannot substitute for Warfare Lore',async()=>{
 const f=fixture({prepared:false});f.actor.skills['sailing-lore']={slug:'sailing-lore',label:'Sailing Lore',lore:true,check:{roll(){}}};await f.run();
 assert.equal(f.prompts.length,0);assert.equal(f.requests[0].request.statistic,'society');
});

test('cancelled prepared Lore selection does not roll or settle',async()=>{
 const f=fixture({selection:null});await f.run();assert.equal(f.prompts.length,1);assert.equal(f.requests.length,0);assert.equal(f.writes(),0);
});

test('prepared Warfare Lore preserves the source actor ownership guard',async()=>{
 const f=fixture({owner:false});await assert.rejects(f.run());assert.equal(f.prompts.length,0);assert.equal(f.requests.length,0);assert.equal(f.writes(),0);
});
