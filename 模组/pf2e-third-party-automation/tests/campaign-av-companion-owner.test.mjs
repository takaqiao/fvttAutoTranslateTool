import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createCampaignFeats,CAMPAIGN_SOURCES} from '../scripts/campaign-feats.mjs';
import {createAvAutomation,AV_SOURCES} from '../scripts/av-automation.mjs';
import {withDamageMessageTarget} from '../scripts/damage-message-targets.mjs';
const ID='pf2e-third-party-automation';
function fixture(t,{cancel=null,blind=false,confirmation}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:true};
 const game={user:gm,users:Object.assign(new Map([gm,player].map(u=>[u.id,u])),{activeGM:gm}),actors:new Map(),messages:new Map(),scenes:new Map(),modules:new Map(),time:{worldTime:10}},docs=new Map(),windows=[],writes=[],cards=[];
 function update(changes){writes.push(game.user.id);for(const[path,value]of Object.entries(changes)){const keys=path.split('.');let at=this;for(const k of keys.slice(0,-1))at=at[k]??={};at[keys.at(-1)]=value;}return this;}
 function actor(id,type='character'){const a={id,uuid:'Actor.'+id,type,level:5,name:id,flags:{},items:new Map(),conditions:[],system:{actions:[]},signature:id+'-signature',testUserPermission:u=>[gm,player].includes(u),update,async createEmbeddedDocuments(_type,data){writes.push(game.user.id);return data.map((d,i)=>{const doc={...d,id:'effect'+i,actor:a,update};a.items.set(doc.id,doc);return doc;});},async deleteEmbeddedDocuments(_type,ids){writes.push(game.user.id);for(const id of ids)a.items.delete(id);},async applyDamage(params){writes.push(game.user.id);a.applied=params;},getReach:()=>5,hasCondition:()=>false};game.actors.set(id,a);docs.set(a.uuid,a);return a;}
 function item(a,id,source,type='feat'){const i={id,uuid:a.uuid+'.Item.'+id,actor:a,type,sourceId:source,system:{traits:{value:[],otherTags:[]},range:null},getOriginData:()=>({actor:a.uuid,uuid:i.uuid,type}),update};a.items.set(id,i);docs.set(i.uuid,i);return i;}
 const scene={id:'s',tokens:new Map()};game.scenes.set(scene.id,scene);
 function token(a,id){const token={id,uuid:'Scene.s.Token.'+id,actor:a,parent:scene,object:{center:{},distanceTo:()=>5,checkCollision:()=>false}};token.object.document=token;scene.tokens.set(id,token);docs.set(token.uuid,token);a.getActiveTokens=()=>[token];return token;}
 const pc=actor('pc'),enemy=actor('enemy','npc'),sourceToken=token(pc,'pc'),target=token(enemy,'enemy');
 const use={id:'use',uuid:'ChatMessage.use',author:player,speaker:{actor:pc.id,scene:'s',token:sourceToken.id},flags:{[ID]:{usageInput:{targetUuids:[target.uuid]}}},update};game.messages.set(use.id,use);
 const roll={total:22,_evaluated:true,options:{degreeOfSuccess:2},toJSON:()=>({formula:'1d20+7',total:22,evaluated:true,options:{degreeOfSuccess:2}})};
 function check(kind,parameters,weapon){const id='check-'+game.messages.size,card={id,uuid:'ChatMessage.'+id,author:game.user,isCheckRoll:true,rolls:[roll],blind,whisper:blind?[gm.id]:[],speaker:{actor:pc.id},flags:{pf2e:{origin:weapon?.getOriginData?.(),context:{type:kind==='attack'?'attack-roll':'saving-throw',action:kind==='attack'?'strike':parameters.action,outcome:'success',options:[...parameters.options??parameters.extraRollOptions??[]],target:{actor:enemy.uuid,token:target.uuid}}}},update};game.messages.set(id,card);return card;}
 class DamageRoll{constructor(formula){this.formula=formula;this.options={};}async evaluate(){windows.push({kind:'formula-damage',user:game.user.id});this._evaluated=true;this.total=8;return this;}toJSON(){return {formula:this.formula,total:this.total,evaluated:this._evaluated};}static fromJSON(data){const o=typeof data==='string'?JSON.parse(data):data;return Object.assign(new DamageRoll(o.formula),o,{_evaluated:o.evaluated});}async toMessage(data,options){cards.push({data,options,user:game.user.id});return {id:'damage',uuid:'ChatMessage.damage',flags:data.flags,rolls:[this]};}}
 game.pf2e={DamageRoll};const previous={CONFIG:globalThis.CONFIG,ChatMessage:globalThis.ChatMessage};globalThis.CONFIG={Dice:{rolls:[DamageRoll]}};globalThis.ChatMessage={getSpeaker:({actor,token})=>({actor:actor.id,scene:'s',token:token?.id})};t.after(()=>Object.assign(globalThis,previous));
 const weapon=item(pc,'weapon','weapon','weapon');weapon.isMelee=true;const strike={type:'strike',ready:true,item:weapon,label:'Sword',variants:Array.from({length:3},()=>({async roll(parameters){windows.push({kind:'attack',user:game.user.id});if(cancel==='attack')return null;const card=check('attack',parameters,weapon);await parameters.callback?.(roll,'success',card);return roll;}})),async damage(parameters){if(cancel==='damage')return null;return new DamageRoll('1d8[slashing]').evaluate();}};pc.system.actions=[strike];
 const stat={rank:1,check:{async roll(parameters){windows.push({kind:'check',user:game.user.id});if(confirmation)await confirmation;if(cancel==='check')return null;const card=check('check',parameters);await parameters.callback?.(roll,'success',card);return roll;}}};pc.skills={medicine:stat};pc.saves={fortitude:stat.check};pc.getStatistic=slug=>slug==='medicine'?stat:slug==='fortitude'?{check:stat.check,roll:stat.check.roll}:null;
 async function runNative(context,request){assert.equal(game.user,gm);assert.equal(context.message.author.id,context.user.id);const before=game.user;game.user=context.user;try{
  if(request.type==='formula-damage'){const damage=await new DamageRoll(request.formula).evaluate();return {status:'rolled',roll:damage,privacy:request.minimumPrivacy?.blind?{messageMode:'blind',blind:true,whisper:[gm.id]}:{messageMode:'public',blind:false,whisper:[]}};}
  if(request.type==='attack'){const result=await strike.variants[request.map].roll({options:request.options});return result?{status:'rolled',messageId:[...game.messages.values()].at(-1).id}:{status:'cancelled'};}
  if(request.type==='damage'){const damage=await strike.damage({options:request.options});return damage?{status:'rolled',roll:damage.toJSON(),context:{options:[...request.options,'damage:manual-adjustment'],domains:['damage','strike-damage'],traits:['attack']},privacy:request.minimumPrivacy?.blind?{messageMode:'blind',blind:true,whisper:[gm.id]}:{messageMode:'public',blind:false,whisper:[]}}:{status:'cancelled'};}
  assert.equal(request.type,'check');const statistic=context.actor.getStatistic(request.statistic),result=await statistic.check.roll({skipDialog:false,event:null,dc:request.dc,extraRollOptions:request.options});const card=[...game.messages.values()].at(-1);return result?{status:'rolled',messageId:card.id,check:card}:{status:'cancelled'};
 }finally{game.user=before;}}
 const fromUuid=async uuid=>docs.get(uuid)??{toObject:()=>({type:'effect',system:{},flags:{}})};
 const manualDamageRoll=async({roll})=>roll.evaluate();
 return {game,gm,player,pc,enemy,target,use,docs,windows,writes,cards,item,runNative,fromUuid,DamageRoll,manualDamageRoll};
}
test('campaign composite attack and damage use the activity author while GM writes the final card',async t=>{
 const f=fixture(t,{blind:true}),item=f.item(f.pc,'slam',CAMPAIGN_SOURCES.slam);f.item(f.pc,'crashing',CAMPAIGN_SOURCES.crashing);
 const provider=createCampaignFeats({game:f.game,fromUuid:f.fromUuid,choose:async()=> '0:0',runNative:f.runNative,manualDamageRoll:f.manualDamageRoll});await provider.executeUsage({actor:f.pc,item,message:f.use,user:f.player,action:'campaign:crashing-slam'});
 assert.deepEqual(f.windows.map(w=>w.user),['player','player']);assert.equal(f.cards.length,1);assert.equal(f.cards[0].user,'gm');assert.equal(f.cards[0].data.blind,true);assert.deepEqual(f.cards[0].data.whisper,['gm']);assert.equal(f.cards[0].options.messageMode,'blind');assert.ok(f.cards[0].data.flags.pf2e.context.options.includes('damage:manual-adjustment'));assert.ok(f.writes.every(u=>u==='gm'));
});
test('campaign attack cancellation does not request damage or alter its original paid usage',async t=>{
 const f=fixture(t,{cancel:'attack'}),item=f.item(f.pc,'slam',CAMPAIGN_SOURCES.slam);f.item(f.pc,'crashing',CAMPAIGN_SOURCES.crashing);f.use.flags[ID].usage={status:'done',paid:true};
 await assert.rejects(createCampaignFeats({game:f.game,fromUuid:f.fromUuid,choose:async()=> '0:0',runNative:f.runNative,manualDamageRoll:f.manualDamageRoll}).executeUsage({actor:f.pc,item,message:f.use,user:f.player,action:'campaign:crashing-slam'}),/取消/);
 assert.deepEqual(f.windows,[{kind:'attack',user:'player'}]);assert.equal(f.cards.length,0);assert.deepEqual(f.use.flags[ID].usage,{status:'done',paid:true});
});
test('campaign damage cancellation retains the confirmed attack and existing payment without a damage card',async t=>{
 const f=fixture(t,{cancel:'damage'}),item=f.item(f.pc,'slam',CAMPAIGN_SOURCES.slam);f.item(f.pc,'crashing',CAMPAIGN_SOURCES.crashing);f.use.flags[ID].usage={status:'done',paid:true};
 await assert.rejects(createCampaignFeats({game:f.game,fromUuid:f.fromUuid,choose:async()=> '0:0',runNative:f.runNative}).executeUsage({actor:f.pc,item,message:f.use,user:f.player,action:'campaign:crashing-slam'}),/伤害/);
 assert.equal(f.windows[0].user,'player');assert.equal(f.cards.length,0);assert.equal(f.use.flags[ID].campaignStrike.status,'awaiting-damage');assert.ok(f.game.messages.get(f.use.flags[ID].campaignStrike.checkId));assert.deepEqual(f.use.flags[ID].usage,{status:'done',paid:true});
});
test('campaign owner damage cannot publish after the original activity card disappears while its reply settles',async t=>{
 const f=fixture(t),item=f.item(f.pc,'slam',CAMPAIGN_SOURCES.slam);f.item(f.pc,'crashing',CAMPAIGN_SOURCES.crashing);
 const runNative=async(context,request)=>{const result=await f.runNative(context,request);if(request.type==='damage')f.game.messages.delete(f.use.id);return result;};
 await assert.rejects(createCampaignFeats({game:f.game,fromUuid:f.fromUuid,choose:async()=> '0:0',runNative}).executeUsage({actor:f.pc,item,message:f.use,user:f.player,action:'campaign:crashing-slam'}),/来源/);assert.equal(f.cards.length,0);
});
test('campaign keeps the player self audience when the GM publishes their saved native damage',async t=>{
 const f=fixture(t),item=f.item(f.pc,'slam',CAMPAIGN_SOURCES.slam);f.item(f.pc,'crashing',CAMPAIGN_SOURCES.crashing);
 const runNative=async(context,request)=>{const result=await f.runNative(context,request);if(request.type==='damage')result.privacy={messageMode:'self',blind:false,whisper:[f.player.id]};return result;};
 await createCampaignFeats({game:f.game,fromUuid:f.fromUuid,choose:async()=> '0:0',runNative}).executeUsage({actor:f.pc,item,message:f.use,user:f.player,action:'campaign:crashing-slam'});
 assert.deepEqual(f.cards[0].data.whisper,[f.player.id]);assert.equal(f.cards[0].options.messageMode,'gm');
});
test('campaign extra damage authenticates the original Slam activity while preserving its Crashing Slam damage origin',async t=>{
 const f=fixture(t),activityItem=f.item(f.pc,'slam',CAMPAIGN_SOURCES.slam,'action'),damageItem=f.item(f.pc,'crashing',CAMPAIGN_SOURCES.crashing);f.use.flags.pf2e={origin:{uuid:activityItem.uuid}};
 const source=fs.readFileSync(new URL('../scripts/campaign-feats.mjs',import.meta.url),'utf8'),start=source.indexOf('async function postDamage('),end=source.indexOf('async function maintain(',start);
 const postDamage=vm.runInNewContext('('+source.slice(start,end).trim()+')',{game:f.game,MODULE_ID:ID,fromUuid:f.fromUuid,gm(){assert.equal(f.game.user,f.gm);},userFor:message=>message.author,messageItem:async message=>f.fromUuid(message.flags.pf2e.origin.uuid),damageRollClass:()=>f.DamageRoll,actorTokens:()=>[],withDamageMessageTarget,runNative:async(context,request)=>{assert.equal(request.itemUuid,context.message.flags.pf2e.origin.uuid,'native owner source must match the actual originating activity');return f.runNative(context,request);}});
 await postDamage({actor:f.pc,item:damageItem,target:f.target,formula:'1d6[bludgeoning]',usageId:f.use.id});assert.equal(f.windows[0].user,f.player.id);assert.equal(f.cards[0].data.flags.pf2e.origin.uuid,damageItem.uuid);
});
test('Battle Medicine waits for the player confirmation before GM immunity and healing',async t=>{
 let accept;const confirmation=new Promise(resolve=>{accept=resolve;}),f=fixture(t,{confirmation}),item=f.item(f.pc,'bm',CAMPAIGN_SOURCES.battleMedicine,'action');f.item(f.pc,'paragon',CAMPAIGN_SOURCES.paragon);
 const pending=createCampaignFeats({game:f.game,fromUuid:f.fromUuid,runNative:f.runNative}).executeUsage({actor:f.pc,item,message:f.use,user:f.player,action:'campaign:battle-medicine'});
 await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(f.windows,[{kind:'check',user:'player'}]);assert.equal(f.enemy.items.size,0);assert.equal(f.enemy.applied,undefined);assert.equal(f.cards.length,0);
 accept();await pending;assert.equal(f.enemy.applied.damage,-8);assert.equal(f.enemy.items.size,1);assert.ok(f.writes.every(user=>user==='gm'));
});
for(const cancel of [null,'check'])test(`Battle Medicine ${cancel?'cancellation':'success'} keeps owner dice and GM immunity/healing`,async t=>{
 const f=fixture(t,{cancel}),item=f.item(f.pc,'bm',CAMPAIGN_SOURCES.battleMedicine,'action');f.item(f.pc,'paragon',CAMPAIGN_SOURCES.paragon);
 const pending=createCampaignFeats({game:f.game,fromUuid:f.fromUuid,runNative:f.runNative,manualDamageRoll:f.manualDamageRoll}).executeUsage({actor:f.pc,item,message:f.use,user:f.player,action:'campaign:battle-medicine'});
 if(cancel){await assert.rejects(pending,/取消/);assert.equal(f.enemy.items.size,0);assert.equal(f.enemy.applied,undefined);}else{await pending;assert.equal(f.enemy.applied.damage,-8);assert.equal(f.enemy.items.size,1);}
 assert.ok(f.windows.length);assert.ok(f.windows.every(w=>w.user==='player'));assert.ok(f.writes.every(u=>u==='gm'));
});
for(const cancel of [null,'check'])test(`Shake It Off ${cancel?'cancellation':'success'} routes only Fortitude to its activity author`,async t=>{
 const f=fixture(t,{cancel}),sickened={id:'sickened',flags:{'patreon-v3':{dc:20}}},reductions=[];f.pc.items.set('rage',{type:'effect',slug:'rage'});f.pc.items.set(sickened.id,sickened);f.pc.getCondition=slug=>slug==='frightened'?{id:'frightened'}:sickened;f.pc.decreaseCondition=async c=>{reductions.push({condition:typeof c==='string'?c:c.id,user:f.game.user.id});};
 const item=f.item(f.pc,'shake',AV_SOURCES.shake),provider=createAvAutomation({game:f.game,fromUuid:f.fromUuid,castEvents:{addMatcher(){},addCapture(){}},runNative:f.runNative,manualDamageRoll:f.manualDamageRoll});await provider.executeUsage({actor:f.pc,item,message:f.use,user:f.player,action:'av:shake'});
 assert.deepEqual(f.windows,[{kind:'check',user:'player'}]);assert.equal(reductions.filter(r=>r.condition==='sickened').length,cancel?0:2);assert.ok(reductions.every(r=>r.user==='gm'));assert.equal(reductions[0].condition,'frightened');
});
