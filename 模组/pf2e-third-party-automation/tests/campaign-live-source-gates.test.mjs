import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID as ID} from '../scripts/rules.mjs';
import {createThrallAutomation,CONSUME_THRALL_SOURCE} from '../scripts/thrall-automation.mjs';
import {createCampaignDailies,ARCANE_EVOLUTION_SOURCE} from '../scripts/daily-feats.mjs';
import {createCampaignFeats,CAMPAIGN_SOURCES} from '../scripts/campaign-feats.mjs';
import {createSocialAutomation} from '../scripts/social-automation.mjs';

function patch(doc,changes){for(const[path,value]of Object.entries(changes)){const keys=path.split('.');let at=doc;for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=structuredClone(value);}}
function bindSynthetic(actor,token,game){
 const base={...actor,uuid:`Actor.${actor.id}`,isToken:false,parent:null,token:null};game.actors.set(base.id,base);
 actor.isToken=true;actor.parent=token;actor.token=token;actor.uuid=`${token.uuid}.Actor.${actor.id}`;
 token.actorLink=false;token.actorId=base.id;token.delta={syntheticActor:actor};Object.defineProperty(token,'baseActor',{get:()=>game.actors.get(token.actorId)??null});
}
function thrallFixture({change,paid=false,deleteError=false}={}){
 const gm={id:'gm',isGM:true},nextGM={id:'next-gm',isGM:true},user={id:'owner'},users=Object.assign(new Map([gm,nextGM,user].map(u=>[u.id,u])),{activeGM:gm});
 const game={user:gm,users,actors:new Map(),messages:new Map(),scenes:new Map(),time:{worldTime:10}},docs=new Map(),counts={deletes:0,grants:0,itemUpdates:0},scene={id:'scene',tokens:new Map()};game.scenes.set(scene.id,scene);
 let owned=true,distance=5;
 const actor={id:'summoner',uuid:'Actor.summoner',type:'character',name:'Summoner',canAct:true,items:new Map(),flags:{},system:{resources:{focus:{value:0,max:1}}},testUserPermission:()=>owned,async update(changes){patch(this,changes);if(changes['system.resources.focus.value']===1)counts.grants++;await change?.(changes[`flags.${ID}.thrallUse`]?.status??'actor-update',f);}};game.actors.set(actor.id,actor);docs.set(actor.uuid,actor);
 const item={id:'consume',uuid:'Actor.summoner.Item.consume',type:'action',actor,parent:actor,sourceId:CONSUME_THRALL_SOURCE,system:{frequency:{value:paid?0:1,max:1,per:'day'}},async update(changes){counts.itemUpdates++;patch(this,changes);await change?.('frequency',f);}};actor.items.set(item.id,item);docs.set(item.uuid,item);
 const source={id:'source',uuid:'Scene.scene.Token.source',documentName:'Token',actor,parent:scene,object:{distanceTo:()=>distance}};
 const thrall={id:'thrall-actor',uuid:'Actor.thrall',flags:{'pf2e-summons-assistant':{summoner:{uuid:actor.uuid}}},rollOptions:{all:{'self:trait:thrall':true}},system:{attributes:{hp:{value:1}}}};
 const target={id:'thrall',uuid:'Scene.scene.Token.thrall',name:'Thrall',documentName:'Token',actor:thrall,parent:scene,object:{},async delete(){counts.deletes++;docs.delete(this.uuid);scene.tokens.delete(this.id);if(deleteError)throw Error('native deletion reply lost');return this;}};
 for(const doc of [source,target]){docs.set(doc.uuid,doc);scene.tokens.set(doc.id,doc);}
 const receipt=paid?{id:'paid-use',itemUuid:item.uuid,userId:user.id,before:1,after:0}:null;
 const message={id:'use',author:user,speaker:{actor:actor.id,scene:scene.id,token:source.id},flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid}},[ID]:{usageInput:{actualUse:true,targetUuids:[target.uuid],frequencyReceiptId:receipt?.id??null},usage:{frequencyReceipt:receipt}}}};game.messages.set(message.id,message);
 const provider=createThrallAutomation({game,fromUuid:async uuid=>{await change?.(`resolve:${uuid}`,f);return docs.get(uuid)},castEvents:{addActorMatcher(){},addConsumePolicy(){}}});
 const f={game,gm,nextGM,user,actor,item,source,target,thrall,scene,docs,message,counts,receipt,provider,revokeOwner(){owned=false},move(value){distance=value},run:()=>provider.executeUsage({actor,item,message,user,action:'thrall:consume',frequencyReceipt:receipt})};return f;
}

test('Consume Thrall stops before deletion when summoner changes during the deleting save',async()=>{
 const f=thrallFixture({change(phase,f){if(phase==='deleting')f.thrall.flags['pf2e-summons-assistant'].summoner.uuid='Actor.other'}});
 await assert.rejects(f.run());assert.equal(f.counts.deletes,0);assert.equal(f.counts.grants,0);assert.equal(f.item.system.frequency.value,0);
});
test('Consume Thrall retains the native one-delete and one-grant path for unchanged documents',async()=>{
 const f=thrallFixture();await f.run();assert.equal(f.counts.deletes,1);assert.equal(f.counts.grants,1);assert.equal(f.item.system.frequency.value,0);
});
test('Consume Thrall keeps exact synthetic source and target actors bound to their scene tokens',async()=>{
 const f=thrallFixture();bindSynthetic(f.actor,f.source,f.game);f.item.uuid=`${f.actor.uuid}.Item.${f.item.id}`;f.message.flags.pf2e.origin={actor:f.actor.uuid,uuid:f.item.uuid};bindSynthetic(f.thrall,f.target,f.game);f.thrall.flags['pf2e-summons-assistant'].summoner.uuid=f.actor.uuid;
 await f.run();assert.equal(f.counts.deletes,1);assert.equal(f.counts.grants,1);
});

const thrallChanges={
 gm:f=>f.game.users.activeGM=f.nextGM,
 'same-ID GM document':f=>{const replacement={...f.gm};f.game.users.set(f.gm.id,replacement);f.game.users.activeGM=replacement;},
 'same-ID GM user':f=>f.game.user={...f.gm},
 'same-ID GM registry':f=>f.game.users.set(f.gm.id,{...f.gm}),
 owner:f=>f.revokeOwner(),
 user:f=>f.game.users.set(f.user.id,{...f.user}),
 sourceItem:f=>f.actor.items.set(f.item.id,{...f.item}),
 card:f=>f.game.messages.set(f.message.id,{...f.message}),
 sourceToken:f=>f.source.actor={...f.actor},
 targetToken:f=>f.scene.tokens.set(f.target.id,{...f.target}),
 targetActor:f=>f.target.actor={...f.thrall},
 summoner:f=>f.thrall.flags['pf2e-summons-assistant'].summoner.uuid='Actor.other',
 range:f=>f.move(31),
 living:f=>f.thrall.system.attributes.hp.value=0,
 focus:f=>f.actor.system.resources.focus.value=1,
 frequency:f=>f.item.system.frequency.value=2,
};
for(const phase of ['claimed','frequency','deleting'])for(const [name,mutate]of Object.entries(thrallChanges))test(`Consume Thrall stops ${name} changed at ${phase}`,async()=>{
 const f=thrallFixture({change(at,f){if(at===phase)mutate(f)}});
 await assert.rejects(f.run());assert.equal(f.counts.deletes,0);assert.equal(f.counts.grants,0);
});
test('original paid Consume Thrall card cannot refund after a completed use is replayed',async()=>{
 const f=thrallFixture({paid:true});await f.run();await assert.rejects(f.run());assert.equal(f.counts.deletes,1);assert.equal(f.counts.grants,1);assert.equal(f.item.system.frequency.value,0);
});
test('lost native deletion reply stays occupied without a second delete or payment refund',async()=>{
 const f=thrallFixture({paid:true,deleteError:true});await assert.rejects(f.run(),/reply lost/);await assert.rejects(f.run());assert.equal(f.counts.deletes,1);assert.equal(f.counts.grants,0);assert.equal(f.item.system.frequency.value,0);assert.equal(f.actor.flags[ID].thrallUse.status,'uncertain');
});
for(const field of ['sourceTokenUuid','frequencyReceiptId','focusMax'])test(`Consume Thrall stops changed saved ${field} before native deletion`,async()=>{
 const f=thrallFixture({change(phase,f){if(phase==='deleting')f.actor.flags[ID].thrallUse[field]='changed'}});await assert.rejects(f.run());assert.equal(f.counts.deletes,0);
});
test('Consume Thrall stops a changed original target list after saving its claim',async()=>{
 const f=thrallFixture({change(phase,f){if(phase==='claimed')f.message.flags[ID].usageInput.targetUuids=['Scene.scene.Token.other']}});await assert.rejects(f.run());assert.equal(f.counts.deletes,0);
});
test('Consume Thrall reconciliation cannot grant after source changes during lookup',async()=>{
 const f=thrallFixture({paid:true,deleteError:true,change(phase,f){if(phase===`resolve:${f.target.uuid}`&&f.counts.deletes===1&&f.actor.flags[ID]?.thrallUse?.status==='uncertain')f.actor.items.delete(f.item.id)}});await assert.rejects(f.run());await f.provider.maintain(f.actor).catch(()=>{});assert.equal(f.counts.grants,0);assert.equal(f.counts.deletes,1);assert.equal(f.item.system.frequency.value,0);
});
test('Consume Thrall can reconcile its own deleted target without deleting or granting twice',async()=>{
 const f=thrallFixture({paid:true,deleteError:true});await assert.rejects(f.run());await f.provider.maintain(f.actor);await f.provider.maintain(f.actor);assert.equal(f.counts.deletes,1);assert.equal(f.counts.grants,1);assert.equal(f.item.system.frequency.value,0);
});
for(const name of ['same-ID GM document','same-ID GM user','same-ID GM registry'])test(`Consume Thrall reconciliation stops ${name} replaced during target lookup`,async()=>{
 const f=thrallFixture({paid:true,deleteError:true,change(phase,f){if(phase===`resolve:${f.target.uuid}`&&f.counts.deletes===1&&f.actor.flags[ID]?.thrallUse?.status==='uncertain')thrallChanges[name](f)}});
 await assert.rejects(f.run());await f.provider.maintain(f.actor).catch(()=>{});assert.equal(f.counts.deletes,1);assert.equal(f.counts.grants,0);assert.equal(f.item.system.frequency.value,0);assert.equal(f.actor.flags[ID].thrallUse.status,'uncertain');
});
test('Consume Thrall preserves the original target actor across its first resolver await',async()=>{
 let changed=false;const f=thrallFixture({change(phase,f){if(phase===`resolve:${f.target.uuid}`&&!changed){changed=true;f.target.actor={...f.thrall,id:'replacement',uuid:'Actor.replacement'};f.game.actors.set('replacement',f.target.actor);}}});await assert.rejects(f.run());assert.equal(f.counts.deletes,0);
});
test('a known refusal before native deletion cannot later reconcile an external target deletion as a grant',async()=>{
 const f=thrallFixture({change(phase,f){if(phase==='deleting')f.thrall.flags['pf2e-summons-assistant'].summoner.uuid='Actor.other'}});await assert.rejects(f.run());assert.equal(f.actor.flags[ID].thrallUse.status,'rejected');f.docs.delete(f.target.uuid);f.scene.tokens.delete(f.target.id);await f.provider.maintain(f.actor);assert.equal(f.counts.deletes,0);assert.equal(f.counts.grants,0);assert.equal(f.item.system.frequency.value,0);
});

function dailyFixture({mode='signature',change}={}){
 const staged=[],actor={type:'character',items:new Map(),flags:{[ID]:{arcaneEvolutionLearned:[{uuid:'Compendium.pf2e.spells-srd.Item.learned',learned:true}]}}};
 const feat={id:'feat',type:'feat',actor,sourceId:ARCANE_EVOLUTION_SOURCE,system:{}},entry={id:'entry',type:'spellcastingEntry',actor,system:{prepared:{value:'spontaneous'},tradition:{value:'arcane'},slots:{slot1:{max:1}}}},spell={id:'spell',name:'Original spell',type:'spell',actor,system:{location:{value:'entry',signature:false},level:{value:1},traits:{value:[]}}};
 for(const doc of [feat,entry,spell])actor.items.set(doc.id,doc);
 const learned={uuid:'Compendium.pf2e.spells-srd.Item.learned',type:'spell',baseRank:1,system:{level:{value:1},traits:{value:[],traditions:['arcane']}},toObject(){return {type:this.type,name:'Learned',system:structuredClone(this.system)}}};
 const [daily]=createCampaignDailies({fromUuid:async()=>{change?.(f);return learned}});
 const rows={mode,signature:spell.id,learned:learned.uuid,rank:'1'};
 const f={actor,feat,entry,spell,learned,staged,daily,rows,run:()=>daily.process({actor,rows,updateItem:row=>staged.push({kind:'update',row}),deleteItem:row=>staged.push({kind:'delete',row}),addItem:row=>staged.push({kind:'add',row}),messages:{add(){}}})};return f;
}
for(const field of ['suppressed','isSuppressed','system.suppressed'])test(`Arcane Evolution rejects native or legacy ${field}`,async()=>{
 const f=dailyFixture();patch(f.feat,{[field]:true});assert.equal(f.daily.condition(f.actor),false);await assert.rejects(f.run());assert.equal(f.staged.length,0);
});
for(const mode of ['signature','learned'])test(`Arcane Evolution preserves valid ${mode} staging`,async()=>{
 const f=dailyFixture({mode});await f.run();assert.equal(f.staged.length,1);assert.equal(f.staged[0].kind,mode==='learned'?'add':'update');
});
const dailyChanges={
 feat:f=>f.actor.items.delete(f.feat.id),
 suppressed:f=>f.feat.suppressed=true,
 entry:f=>f.actor.items.set(f.entry.id,{...f.entry}),
 tradition:f=>f.entry.system.tradition.value='divine',
 slot:f=>f.entry.system.slots.slot1.max=0,
 learned:f=>f.actor.flags[ID].arcaneEvolutionLearned=[],
 repertoire:f=>f.actor.items.set('new-known',{id:'new-known',type:'spell',sourceId:f.learned.uuid,system:{location:{value:f.entry.id},traits:{value:[]}}}),
};
for(const [name,change]of Object.entries(dailyChanges))test(`Arcane Evolution rejects ${name} changed during learned lookup before staging`,async()=>{
 const f=dailyFixture({mode:'learned',change});await assert.rejects(f.run());assert.equal(f.staged.length,0);
});
test('Arcane Evolution keeps native rest cleanup after the feat is gone',()=>{
 const f=dailyFixture();f.actor.items.delete(f.feat.id);f.spell.system.location.signature=true;f.spell.flags={[ID]:{arcaneEvolutionSignature:{existed:true,before:false}}};f.daily.rest({actor:f.actor,updateItem:row=>f.staged.push(row),removeItem(){}});assert.equal(f.staged[0]['system.location.signature'],false);
});
test('Arcane Evolution keeps the captured selected rank while the learned lookup awaits',async()=>{
 const f=dailyFixture({mode:'learned',change:f=>f.rows.rank='2'});f.entry.system.slots.slot2={max:1};await f.run();assert.equal(f.staged[0].row.system.location.heightenedLevel,1);
});

function effectFixture(t,{providerName,change}={}){
 const old={CONFIG:globalThis.CONFIG,ChatMessage:globalThis.ChatMessage};t.after(()=>Object.assign(globalThis,old));globalThis.CONFIG={Canvas:{polygonBackends:{sound:{testCollision:()=>false}}}};
 const gm={id:'gm',isGM:true},nextGM={id:'next-gm',isGM:true},user={id:'owner'},users=Object.assign(new Map([gm,nextGM,user].map(u=>[u.id,u])),{activeGM:gm});
 const game={user:gm,users,actors:new Map(),messages:new Map(),scenes:new Map(),modules:new Map(),time:{worldTime:10}},docs=new Map(),counts={effects:0,conditions:0,checks:0,summaries:0};let owned=true,distance=5;
 const makeActor=(id,type)=>({id,uuid:`Actor.${id}`,name:id,type,items:new Map(),flags:{},canAct:true,system:{details:{languages:{value:['common']}}},testUserPermission:()=>owned,hasCondition:()=>false,isAllyOf:()=>true,getCondition(name){return name==='frightened'&&this.fear>0?{value:this.fear}:null},getStatistic(slug){return slug==='will'?{dc:{value:15}}:{roll(){},check:{domains:['diplomacy']}}},async update(changes){patch(this,changes)},async createEmbeddedDocuments(kind,rows){counts.effects++;const result=rows.map((row,index)=>({...structuredClone(row),id:`effect-${this.id}-${index}`,actor:this,async update(changes){patch(this,changes)}}));for(const row of result)this.items.set(row.id,row);await change?.('effect',f);return result},async decreaseCondition(){counts.conditions++;this.fear--;await change?.('condition',f)}});
 const actor=makeActor('hero','character'),recipient=makeActor('recipient','character');recipient.fear=3;for(const doc of [actor,recipient]){game.actors.set(doc.id,doc);docs.set(doc.uuid,doc);}
 const scene={id:'scene',tokens:new Map(),levels:new Map([['level',{}]])},origin={id:'origin',uuid:'Scene.scene.Token.origin',documentName:'Token',actor,parent:scene,_source:{level:'level'},object:{center:{x:0,y:0},distanceTo:()=>distance,checkCollision:()=>false}},target={id:'target',uuid:'Scene.scene.Token.target',documentName:'Token',actor:recipient,parent:scene,object:{center:{x:5,y:0}}};
 for(const doc of [origin,target]){scene.tokens.set(doc.id,doc);docs.set(doc.uuid,doc);}game.scenes.set(scene.id,scene);actor.getActiveTokens=()=>[origin];
 const item={id:'feature',uuid:'Actor.hero.Item.feature',type:providerName==='social'?'feat':'action',actor,parent:actor,sourceId:providerName==='social'?'Compendium.pf2e.feats-srd.Item.6ON8DjFXSMITZleX':CAMPAIGN_SOURCES.walls};actor.items.set(item.id,item);docs.set(item.uuid,item);
 const message={id:'use',author:user,speaker:{actor:actor.id,scene:scene.id,token:origin.id},flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid}},[ID]:{usageInput:{actualUse:true,targetUuids:[target.uuid]}}},async update(changes){patch(this,changes);await change?.('receipt',f)}};game.messages.set(message.id,message);
 const roll={total:25,dice:[{faces:20,total:10}]},check={id:'check',author:user,speaker:{actor:actor.id},rolls:[roll],flags:{pf2e:{context:{options:['action:no-cause-for-alarm'],dosAdjustments:{},outcome:'criticalSuccess',unadjustedOutcome:'criticalSuccess'}}}};game.messages.set(check.id,check);
 globalThis.ChatMessage={getSpeaker:()=>({actor:actor.id}),create:async data=>{counts.summaries++;return data}};
 const fromUuid=async uuid=>{await change?.(`resolve:${uuid}`,f);if(docs.has(uuid))return docs.get(uuid);await change?.('lookup',f);return {uuid,toObject:()=>({type:'effect',system:{},flags:{}})}};
 const provider=providerName==='social'?createSocialAutomation({game,fromUuid,choose:async()=>{await change?.('choice',f);return 'common'},runNative:async()=>{counts.checks++;await change?.('check',f);return {status:'rolled',check}}}):createCampaignFeats({game,fromUuid});
 const f={game,gm,nextGM,user,actor,recipient,item,origin,target,scene,message,check,counts,provider,revokeOwner(){owned=false},move(value){distance=value},run:()=>provider.executeUsage({actor,item,message,user,action:providerName==='social'?'social:no-cause-for-alarm':'campaign:raise-walls'})};return f;
}
const effectChanges={
 gm:f=>f.game.users.activeGM=f.nextGM,
 'same-ID GM document':f=>{const replacement={...f.gm};f.game.users.set(f.gm.id,replacement);f.game.users.activeGM=replacement;},
 'same-ID GM user':f=>f.game.user={...f.gm},
 'same-ID GM registry':f=>f.game.users.set(f.gm.id,{...f.gm}),
 owner:f=>f.revokeOwner(),
 item:f=>f.actor.items.set(f.item.id,{...f.item}),
 card:f=>f.game.messages.set(f.message.id,{...f.message}),
 sourceToken:f=>f.origin.actor={...f.actor},
 targetToken:f=>f.target.actor={...f.recipient},
 range:f=>f.move(100),
};
test('Raise Walls preserves exactly two effect writes for live native documents',async t=>{const f=effectFixture(t,{providerName:'walls'});await f.run();assert.equal(f.counts.effects,2)});
for(const phase of ['lookup','effect'])for(const [name,mutate]of Object.entries(effectChanges))test(`Raise Walls stops ${name} changed at ${phase} without replaying partial effects`,async t=>{
 const f=effectFixture(t,{providerName:'walls',change(at,f){if(at===phase)mutate(f)}});await assert.rejects(f.run());assert.equal(f.counts.effects,phase==='lookup'?0:1);await assert.rejects(f.run());assert.equal(f.counts.effects,phase==='lookup'?0:1);
});
test('No Cause for Alarm keeps one original check and native effects and reductions',async t=>{const f=effectFixture(t,{providerName:'social'});await f.run();assert.deepEqual(f.counts,{effects:1,conditions:2,checks:1,summaries:1});await f.run();assert.equal(f.counts.checks,1)});
for(const providerName of ['walls','social'])test(`${providerName} keeps native synthetic actor identities with their original world base actors`,async t=>{
 const f=effectFixture(t,{providerName});for(const [actor,token]of [[f.actor,f.origin],[f.recipient,f.target]])bindSynthetic(actor,token,f.game);f.item.uuid=`${f.actor.uuid}.Item.${f.item.id}`;f.message.flags.pf2e.origin={actor:f.actor.uuid,uuid:f.item.uuid};await f.run();assert.equal(f.counts.effects,providerName==='walls'?2:1);assert.equal(f.counts.checks,providerName==='social'?1:0);
});
for(const phase of ['check','effect','condition'])for(const [name,mutate]of Object.entries(effectChanges))test(`No Cause for Alarm stops ${name} changed at ${phase} and retains its original check`,async t=>{
 const f=effectFixture(t,{providerName:'social',change(at,f){if(at===phase)mutate(f)}});await assert.rejects(f.run());assert.equal(f.counts.checks,1);assert.equal(f.counts.effects,phase==='check'?0:1);assert.equal(f.counts.conditions,phase==='condition'?1:0);assert.equal(f.counts.summaries,0);await f.run().catch(()=>{});assert.equal(f.counts.checks,1);
});
test('No Cause for Alarm stops changed original native result proof after an effect await',async t=>{
 const f=effectFixture(t,{providerName:'social',change(phase,f){if(phase==='effect')f.check.flags.pf2e.context.outcome='failure'}});await assert.rejects(f.run());assert.equal(f.counts.effects,1);assert.equal(f.counts.conditions,0);assert.equal(f.counts.checks,1);assert.equal(f.counts.summaries,0);
});
test('Raise Walls stops a changed saved pending receipt after the first effect without replay',async t=>{
 const f=effectFixture(t,{providerName:'walls',change(phase,f){if(phase==='effect')f.message.flags[ID].campaignWalls.targetUuid='Scene.scene.Token.other'}});await assert.rejects(f.run());assert.equal(f.counts.effects,1);await assert.rejects(f.run());assert.equal(f.counts.effects,1);
});
test('Raise Walls preserves the original target actor across its first resolver await',async t=>{
 let changed=false;const f=effectFixture(t,{providerName:'walls',change(phase,f){if(phase===`resolve:${f.target.uuid}`&&!changed){changed=true;f.target.actor={...f.recipient,id:'replacement',uuid:'Actor.replacement'};f.game.actors.set('replacement',f.target.actor);}}});await assert.rejects(f.run());assert.equal(f.counts.effects,0);
});
test('No Cause for Alarm preserves recipient actor identity while the owner chooses language',async t=>{
 const f=effectFixture(t,{providerName:'social',change(phase,f){if(phase==='choice'){f.target.actor={...f.recipient,id:'replacement',uuid:'Actor.replacement'};f.game.actors.set('replacement',f.target.actor);}}});f.actor.system.details.languages.value.push('elven');await assert.rejects(f.run());assert.equal(f.counts.checks,0);assert.equal(f.counts.effects,0);
});
