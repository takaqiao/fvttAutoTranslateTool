import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createDualStrikeAutomation} from '../scripts/dual-strike-automation.mjs';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
import {createBravery} from '../scripts/bravery.mjs';
let api={};try{api=await import('../scripts/spell-combination.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const NS='pf2e-third-party-automation',S={strike:'Compendium.pf2e.actionspf2e.Item.QDW9H8XLIjuW2fE4',swipe:'Compendium.pf2e.feats-srd.Item.Fs88vjez9px2mmrC',combination:'Compendium.pf2e.actionspf2e.Item.zUWj4zmBNWOTzeFJ'};
const set=(o,p,v)=>{const bits=p.split('.');let at=o;for(const k of bits.slice(0,-1))at=at[k]??={};at[bits.at(-1)]=v;};
function roll(total=10){return {total,instances:[],options:{},toJSON(){return {total:this.total,instances:this.instances,options:structuredClone(this.options)}},async evaluate(){return this},alter(multiplier,addend){assert.equal(addend,0);return roll(total*multiplier)},async toMessage(data){return {...data,rolls:[this]}}};}
function fixture({kind='strike',map=0,outcomes=['success','success'],save='failure',spellSave=false,basic=true,attack=!spellSave,targets=kind==='swipe'?2:1,multi=false,charged=true,overlays=false,consumeFails=false}={}){
 const calls=[],messages=[],merges=[];
 const actor={id:'pc',uuid:'Actor.pc',type:'character',flags:{[NS]:{spellstrike:{charged}}},level:15,testUserPermission:()=>true,items:[],system:{resources:{focus:{value:3}}},async update(changes){for(const[k,v]of Object.entries(changes))set(this,k,v)}};
 const feat={id:'feature',uuid:'Actor.pc.Item.feature',sourceId:S[kind],type:kind==='swipe'?'feat':'action',actor,system:{rules:[]},name:kind};
 const scene={id:'s',tokens:new Map()};
 const targetDocs=Array.from({length:targets},(_,i)=>({id:`target${i}`,uuid:`Scene.s.Token.target${i}`,documentName:'Token',parent:scene,actor:{id:`target${i}`,uuid:`Actor.target${i}`,type:'npc'},object:{distanceTo:()=>5}}));
 for(const target of targetDocs){target.object.document=target;target.actor.getStatistic=()=>({check:{roll:async opts=>{calls.push({kind:'save',opts,target});const raw=new Message({speaker:{actor:target.actor.uuid},flags:{pf2e:{context:{type:'saving-throw',outcome:save,options:opts.extraRollOptions}}},rolls:[roll(20)]});await opts.callback?.(roll(20),save,raw);return roll(20);}}});}
 const origin={id:'pcToken',uuid:'Scene.s.Token.pcToken',documentName:'Token',parent:scene,actor,object:{distanceTo:()=>5,checkCollision:()=>false}};actor.getActiveTokens=()=>[origin];actor.getReach=()=>5;
 for(const token of [origin,...targetDocs])scene.tokens.set(token.id,token);
 const weapons=[{id:'weapon',slug:'sword',system:{equipped:{carryType:'held',handsHeld:1},traits:{value:[]}},isUnarmed:false},{id:'fist',slug:'fist',system:{equipped:{carryType:'worn',handsHeld:0},traits:{value:['unarmed']}},isUnarmed:true}].map(w=>({...w,uuid:`Actor.pc.Item.${w.id}`,name:w.id,actor,type:'weapon',category:w.isUnarmed?'unarmed':'martial',isMelee:true,hands:'1',isEquipped:true,get isHeld(){return this.system.equipped.carryType==='held'},get handsHeld(){return this.system.equipped.handsHeld}}));
 actor.system.actions=weapons.map((item,index)=>({type:'strike',ready:true,item,variants:[0,1,2].map(tier=>({roll:async opts=>{const i=calls.filter(c=>c.kind==='attack').length;calls.push({kind:'attack',index,tier,opts});const raw=new Message({speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:item.uuid,type:'weapon',actor:actor.uuid},context:{type:'attack-roll',outcome:outcomes[i],options:[...(opts.options??[])],target:{actor:opts.target.document.actor.uuid,token:opts.target.document.uuid}}}},rolls:[roll(25)]});await opts.callback(roll(25),outcomes[i],raw);return roll(25)}})),damage:async opts=>{calls.push({kind:'weapon-damage',index,opts});return roll(10)},critical:async opts=>{calls.push({kind:'weapon-damage',index,opts});return roll(20)}}));
 const spell={id:'spell',uuid:'Actor.pc.Item.spell',sourceId:'Compendium.pf2e.spells-srd.Item.TestSpell0000000',type:'spell',name:'test spell',actor,isAttack:attack,isCantrip:true,isFocusSpell:false,rank:8,system:{time:{value:'2'},traits:{value:attack?['attack']:[]},location:{value:'entry'},cast:{focusPoints:0},damage:{damage:{formula:'2d6'}},defense:spellSave?{save:{statistic:'reflex',basic}}:null,target:{value:multi?'2 creatures':'1 creature'},area:null},overlays:overlays?new Map([['variant1',{overlayType:'override',name:'Variant 1'}],['variant2',{overlayType:'override',name:'Variant 2'}]]):new Map(),appliedOverlays:new Map(),async getDamage(opts){calls.push({kind:'spell-damage',opts,overlays:[...this.appliedOverlays.values()]});return {template:{damage:{roll:roll(16)}},context:{options:new Set(),domains:['damage','spell-damage'],traits:['arcane']}}},loadVariant({overlayIds=[],castRank=this.rank}={}){return {...this,rank:castRank,isVariant:true,appliedOverlays:new Map(overlayIds.map(id=>['override',id]))}},async toMessage(event,{create,data}={}){assert.equal(create,false);calls.push({kind:'spell-card'});return new Message({speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:this.uuid,type:'spell',actor:actor.uuid,castRank:data?.castRank??this.rank}}},content:'spell card'})}};
 const entry={id:'entry',uuid:'Actor.pc.Item.entry',type:'spellcastingEntry',actor,isPrepared:true,isFlexible:false,statistic:{getChatData:()=>({dc:{value:35}})},async getSheetData(){return {groups:[{id:'cantrips',maxRank:8,active:[{spell,castRank:8,expended:false}]}]}}};spell.spellcasting=entry;
 actor.items=[feat,...weapons,spell,entry];actor.items.get=id=>actor.items.find(i=>i.id===id);actor.spellcasting={contents:[entry],collections:[]};
 actor._source={items:[]};actor.clone=changes=>{calls.push({kind:'infusion-clone',changes});return actor};
 class Message{constructor(data){Object.assign(this,data);this.sourceRolls=this.rolls?.map(r=>({...r,options:structuredClone(r.options??{})}));this.isCheckRoll=!!this.rolls?.length&&this.flags?.pf2e?.context?.type!=='damage-roll';}toObject(){return {...this,rolls:this.sourceRolls,flags:structuredClone(this.flags)}}async update(changes){for(const[k,v]of Object.entries(changes))set(this,k,v);return this}static async create(data){const msg=new Message(data);msg.id=`m${messages.length}`;messages.push(msg);game.messages.set(msg.id,msg);return msg}}
 const gm={id:'gm',settings:{},isGM:true,active:true},player={id:'player',active:true},users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm});
 const game={user:gm,users,actors:new Map([[actor.id,actor],...targetDocs.map(t=>[t.actor.id,t.actor])]),messages:new Map(),modules:new Map([['pf2e-toolbelt',{active:true}]]),time:{worldTime:100},toolbelt:{api:{betterChat:{mergeDamageMessages:async(a,b,opts)=>{merges.push({a,b,opts});return new Message({flags:{pf2e:{context:{options:[]}}},rolls:[roll(a.rolls[0].total+b.rolls[0].total)]})}}}}};
 game.scenes=new Map([[scene.id,scene]]);
 const casts={addMatcher(){},captureUsage(){},register(){return()=>{}},async payForActivity(ctx){calls.push({kind:'payment',ctx});if(consumeFails)throw Error('法术资源不足');return {id:'paid'}},async ensurePaid(ctx){calls.push({kind:'bind-payment',ctx});return{id:'paid'}},async finishActivityWithoutSpell(ctx){calls.push({kind:'finish-payment',ctx})}};
 const chooser=async({title,choices})=>title.includes('多重')?String(map):choices[0].value;
 const nativeOperations={run:async(_ctx,_request,execute,beforeRoll)=>{await beforeRoll?.();return execute();},register(){}};
 assert.equal(typeof api.createSpellCombination,'function');const provider=api.createSpellCombination({game,choose:chooser,fromUuid:async uuid=>[actor,origin,...targetDocs,...actor.items].find(d=>d.uuid===uuid),nativeCasts:casts,nativeOperations});
 const message=new Message({id:'use1',author:player,speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:feat.uuid,type:feat.type,actor:actor.uuid}},[NS]:{usageInput:{actualUse:true,targetUuids:targetDocs.map(t=>t.uuid)}}}});game.messages.set(message.id,message);
 const use=()=>provider.executeUsage({actor,item:feat,message,user:game.users.get('player'),action:provider.resolveAction(feat)});
 return {provider,actor,feat,spell,entry,origin,targetDocs,calls,messages,merges,game,casts,use,message,Message,weapons,nativeOperations};
}
async function setup(options,fn){const previous=globalThis.CONFIG;try{const f=fixture(options);globalThis.CONFIG={ChatMessage:{documentClass:f.Message},Dice:{rolls:[]},PF2E:{}};await fn(f)}finally{globalThis.CONFIG=previous}}
function runWith(f,options={}){const p=api.createSpellCombination({game:f.game,choose:async({choices})=>choices[0].value,fromUuid:async uuid=>[f.actor,f.origin,...f.targetDocs,...f.actor.items].find(d=>d.uuid===uuid),nativeCasts:f.casts,nativeOperations:f.nativeOperations,...options});return()=>p.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:f.game.users.get('player'),action:p.resolveAction(f.feat)});}
const handoff=f=>{f.game.users.activeGM={id:'new-gm'}};
test('a PC target save is dispatched to its owner and closing it never rolls on GM',()=>setup({spellSave:true},async f=>{
 f.targetDocs[0].actor.type='character';let requested=0;
 const use=runWith(f,{runTargetSave:async(context,request)=>{requested++;assert.equal(context.sourceItem.uuid,f.spell.uuid);assert.equal(context.target,f.targetDocs[0]);assert.equal(request.statistic,'reflex');return {status:'cancelled'};}});
 await assert.rejects(use(),/豁免|取消/);assert.equal(requested,1);assert.equal(f.calls.filter(c=>c.kind==='save').length,0);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);
}));
test('a PC target native degree continues the already paid spell with its selected overlay and no GM save',()=>setup({spellSave:true,overlays:true},async f=>{
 f.targetDocs[0].actor.type='character';let requested=0;
 await runWith(f,{runTargetSave:async(context,request)=>{
  requested++;assert.equal(context.sourceMessage.flags[NS].usageGenerated,true);assert.equal(context.sourceItem.actor,f.actor);assert.equal(request.rank,8);assert.deepEqual(request.overlayIds,['variant1']);assert.deepEqual(request.dc,{value:35});
  const check=await f.Message.create({author:{id:'target-player'},speaker:{actor:'target0'},flags:{pf2e:{context:{type:'saving-throw',outcome:'failure'}},[NS]:{nativeTargetSave:{saveUserId:'target-player'}}},rolls:[roll(15)]});return {status:'rolled',messageId:check.id,check};
 }})();
 assert.equal(requested,1);assert.equal(f.calls.filter(c=>c.kind==='save').length,0);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='spell-damage').length,1);assert.equal(f.merges.length,1);
 const card=f.messages.find(m=>m.flags?.[NS]?.spellCombinationSave);assert.equal(card.author.id,'target-player');assert.equal(card.flags[NS].usageGenerated,true);
}));
function dualUse(f,{takedown=false}={}){
 f.feat.type='feat';f.feat.sourceId=takedown?'Compendium.pf2e.feats-srd.Item.Gw0wGXikhAhiGoud':'Compendium.pf2e.feats-srd.Item.onde0SxLoxLBTnvm';
 f.feat.system.rules=[{key:'FlatModifier',predicate:[{or:['double-slice-second',NS+':double-slice-second']}]}];
 f.weapons[1].system.equipped={carryType:'held',handsHeld:1};
 const provider=createDualStrikeAutomation({game:f.game,nativeOperations:f.nativeOperations,fromUuid:async uuid=>[f.origin,...f.targetDocs].find(d=>d.uuid===uuid),choose:async({choices})=>choices[0].value});
 return ()=>provider.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:{id:'player'},action:takedown?'dual:twin-takedown':'dual:double-slice'});
}
function twinPair(f){
 for(const item of f.weapons){item.system.traits.value=['twin'];item.system.baseItem='sawtooth-saber';}
 // The native contextual Strike owns all damage arithmetic. This boundary stub
 // distinguishes its result from the original Strike; real PF2e QA covers dice,
 // circumstance stacking and critical multiplication.
 f.actor.clone=changes=>{f.calls.push({kind:'twin-clone',changes});return {system:{actions:f.actor.system.actions.map(s=>({...s,damage:async opts=>{f.calls.push({kind:'twin-damage',opts,item:s.item});return roll(11)},critical:async opts=>{f.calls.push({kind:'twin-damage',opts,item:s.item});return roll(22)}}))}}};
}
test('combination native attack and damage are dispatched to the original owner',()=>setup({kind:'combination'},async f=>{
 const requests=[];
 await runWith(f,{nativeOperations:{async run(ctx,request,execute){assert.equal(ctx.user.id,'player');assert.equal(ctx.message,f.message);requests.push(request);return execute();},register(){}}})();
 assert.deepEqual(requests.map(request=>request.type),['attack','attack','damage','damage']);
 assert(requests.every(request=>request.targetUuid===f.targetDocs[0].uuid));
}));
test('dual Strikes and damage are dispatched to the original owner',()=>setup({},async f=>{
 const requests=[];f.nativeOperations.run=async(ctx,request,execute)=>{assert.equal(ctx.user.id,'player');requests.push(request);return execute();};
 await dualUse(f)();assert.deepEqual(requests.map(request=>request.type),['attack','damage','attack','damage']);
}));
test('cancelled first Spellstrike native window preserves its spell resource and charge',()=>setup({},async f=>{
 const result=await runWith(f,{nativeOperations:{run:async()=>({status:'cancelled'}),register(){}}})();
 assert.equal(result.status,'cancelled');assert.equal(f.calls.filter(call=>call.kind==='payment').length,0);
 assert.equal(f.actor.flags[NS].spellstrike.charged,true);assert.equal(f.message.flags[NS].spellCombinationUse.state,'cancelled');
}));
test('a missed paid Spellstrike keeps one native spell card for a later kept reroll',()=>setup({outcomes:['failure']},async f=>{
 await f.use();
 const cards=f.messages.filter(message=>message.flags?.[NS]?.spellCombination);
 assert.equal(cards.length,1);assert.equal(cards[0].flags[NS].usageGenerated,true);
 assert.equal(f.calls.filter(call=>call.kind==='payment').length,1);
 assert.equal(f.calls.filter(call=>call.kind==='finish-payment').length,0);
 assert.equal(f.calls.filter(call=>call.kind==='spell-damage').length,0);
}));
test('a cancelled second Spell Swipe keeps the first paid spell available on its native card',()=>setup({kind:'swipe'},async f=>{
 let attacks=0;const original=f.nativeOperations.run;
 f.nativeOperations.run=async(ctx,request,...args)=>request.type==='attack'&&++attacks===2?{status:'cancelled'}:original(ctx,request,...args);
 const result=await f.use();assert.equal(result.status,'cancelled');
 assert.equal(f.messages.filter(message=>message.flags?.[NS]?.spellCombination).length,1);
 assert.equal(f.calls.filter(call=>call.kind==='payment').length,1);
}));
for(const takedown of [false,true])for(const outcomes of [['success','success'],['failure','success'],['success','criticalSuccess']])test(`paired twin weapons use native bonus on the second Strike only (${takedown?'Takedown':'Double Slice'}, ${outcomes})`,()=>setup({outcomes},async f=>{
 twinPair(f);f.actor.signature='hunter';f.targetDocs[0].actor.getRollOptions=()=>['self:prey:hunter'];
 const use=dualUse(f,{takedown});await use();await use();
 assert.equal(f.calls.filter(c=>c.kind==='twin-clone').length,1);
 const rule=f.calls.find(c=>c.kind==='twin-clone').changes.items.at(-1).system.rules[0];
 assert.equal(rule.key,'FlatModifier');assert.equal(rule.type,'circumstance');assert.equal(rule.value,'@weapon.system.damage.dice');assert.equal(rule.selector,`${f.weapons[1].id}-damage`);
 const damage=f.calls.filter(c=>c.kind==='twin-damage');assert.equal(damage.length,1);assert.equal(damage[0].item.id,f.weapons[1].id);
 assert(rule.predicate.includes(`item:id:${f.weapons[1].id}`));assert(rule.predicate.includes('item:trait:twin'));assert(rule.predicate.some(p=>damage[0].opts.options.has(p)));
 assert.equal(damage[0].opts.target.document.uuid,f.targetDocs[0].uuid);assert.equal(damage[0].opts.checkContext.outcome,outcomes[1]);
 assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,outcomes[0]==='success'?1:0);
 assert.equal(f.messages.at(-1).rolls[0].total,(outcomes[0]==='success'?10:0)+(outcomes[1]==='criticalSuccess'?22:11));
 assert.deepEqual(f.actor._source.items,[],'no persistent effect or turn history is written');
}));
for(const mismatch of ['different-type','first-not-twin','second-not-twin','unknown-type'])test(`unconfirmed twin pairing leaves native manual damage unchanged (${mismatch})`,()=>setup({},async f=>{
 twinPair(f);
 if(mismatch==='different-type')f.weapons[1].system.baseItem='khopesh';
 if(mismatch==='first-not-twin')f.weapons[0].system.traits.value=[];
 if(mismatch==='second-not-twin')f.weapons[1].system.traits.value=[];
 if(mismatch==='unknown-type')for(const w of f.weapons)delete w.system.baseItem;
 await dualUse(f)();assert.equal(f.calls.filter(c=>c.kind==='twin-clone').length,0);assert.equal(f.messages.at(-1).rolls[0].total,20);
}));
test('a missed second Strike does not prepare any twin damage clone',()=>setup({outcomes:['success','failure']},async f=>{twinPair(f);await dualUse(f)();assert.equal(f.calls.filter(c=>c.kind==='twin-clone').length,0);assert.equal(f.messages.at(-1).rolls[0].total,10)}));
for(const same of [false,true])test(`unknown weapon base uses exact source identity (${same?'matching':'different'})`,()=>setup({},async f=>{
 twinPair(f);for(const w of f.weapons)delete w.system.baseItem;
 f.weapons[0].sourceId='Compendium.pf2e.equipment-srd.Item.nativeTwin';f.weapons[1].sourceId=same?f.weapons[0].sourceId:'Compendium.other.Item.nativeTwin';
 await dualUse(f)();assert.equal(f.calls.filter(c=>c.kind==='twin-clone').length,same?1:0);assert.equal(f.messages.at(-1).rolls[0].total,same?21:20);
}));
for(const kind of ['dual','swipe','combination'])for(const deleted of ['source','target','scene'])test(`${kind} cannot continue after its ${deleted} is deleted between Strikes`,()=>setup({kind:kind==='dual'?'strike':kind},async f=>{
 const use=kind==='dual'?dualUse(f):f.use;
 const strike=f.actor.system.actions[0].variants[0],original=strike.roll;
 strike.roll=async opts=>{const result=await original(opts);if(deleted==='scene')f.game.scenes.clear();else{const token=deleted==='source'?f.origin:f.targetDocs.at(-1);token.parent.tokens.delete(token.id);token.object=null;}return result};
 await assert.rejects(use,/来源|目标|场景/);
 if(kind==='dual')await use();else await assert.rejects(use,/未完成|重试|确认/);
 assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);
 assert.equal(f.calls.filter(c=>c.kind==='payment').length,kind==='swipe'?1:0);
 assert.equal(f.messages.filter(m=>m.flags[NS]?.dualStrike||m.flags[NS]?.spellCombinationDamage).length,0);
}));
for(const outcomes of [['success','success'],['failure','success']])test('Dual Strike publishes the fixed recipient after one or two hits',()=>setup({outcomes},async f=>{
 f.game.user.targets=new Set([{document:{uuid:'Scene.s.Token.gm-current'}}]);
 f.feat.type='feat';f.feat.sourceId='Compendium.pf2e.feats-srd.Item.onde0SxLoxLBTnvm';
 f.feat.system.rules=[{key:'FlatModifier',predicate:[{or:['double-slice-second',NS+':double-slice-second']}]}];
 f.weapons[1].system.equipped={carryType:'held',handsHeld:1};
 const forbidden=()=>{throw Error('spatial probe must not run')};
 f.actor.getReach=forbidden;f.origin.object.distanceTo=forbidden;f.origin.object.checkCollision=forbidden;
 const provider=createDualStrikeAutomation({game:f.game,nativeOperations:f.nativeOperations,fromUuid:async uuid=>[f.origin,...f.targetDocs].find(d=>d.uuid===uuid),choose:async({choices})=>choices[0].value});
 await provider.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:{id:'player'},action:'dual:double-slice'});
 await provider.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:{id:'player'},action:'dual:double-slice'});
 assert.equal(f.calls.filter(c=>c.kind==='attack').length,2,'The original activity still cannot repeat either Strike');
 const cards=f.messages.filter(m=>m.flags[NS]?.dualStrike);
 assert.equal(cards.length,1);
 assert.deepEqual(cards[0].flags['pf2e-toolbelt']?.targetHelper?.targets,['Scene.s.Token.target0']);
 assert.equal(cards[0].flags.pf2e.context.target.token,'Scene.s.Token.target0');
}));
for(const kind of ['strike','combination','swipe'])test(`${kind} GM damage cards retain each recipient for Toolbelt instead of the GMs current target`,()=>setup({kind},async f=>{
 f.game.user.targets=new Set([{document:{uuid:'Scene.s.Token.gm-current'}}]);
 await f.use();
 const cards=f.messages.filter(m=>m.flags[NS]?.spellCombinationDamage);
 assert.deepEqual(cards.map(m=>m.flags['pf2e-toolbelt']?.targetHelper?.targets),kind==='swipe'?[['Scene.s.Token.target0'],['Scene.s.Token.target1']]:[['Scene.s.Token.target0']]);
}));
test('queued GM rest stops on handoff instead of overwriting the new GMs discharged state',()=>setup({},async f=>{
 let enter,release,onRest;const entered=new Promise(resolve=>{enter=resolve}),pending=new Promise(resolve=>{release=resolve}),errors=[];
 const provider=api.createSpellCombination({game:f.game,nativeCasts:f.casts,fromUuid:async uuid=>[f.actor,f.origin,...f.targetDocs,...f.actor.items].find(d=>d.uuid===uuid),choose:async({choices})=>{enter();await pending;return choices[0].value},onError:error=>errors.push(error)});
 provider.register({Hooks:{on:(event,callback)=>{assert.equal(event,'pf2e.restForTheNight');onRest=callback;return 1},off(){}}});
 let writes=0;const update=f.actor.update;f.actor.update=async changes=>{writes++;return update.call(f.actor,changes)};
 const use=provider.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:f.game.users.get('player'),action:provider.resolveAction(f.feat)});await entered;
 onRest(f.actor);await new Promise(resolve=>setImmediate(resolve));assert.equal(writes,0);
 handoff(f);f.actor.flags[NS].spellstrike={charged:false,messageId:'new-gm-cast'};release();
 await assert.rejects(use,/GM/);await new Promise(resolve=>setImmediate(resolve));
 assert.equal(writes,0);assert.deepEqual(f.actor.flags[NS].spellstrike,{charged:false,messageId:'new-gm-cast'});assert.equal(errors.length,1);assert.match(errors[0].message,/GM/);
}));
test('ordinary Player OWNER rest still restores Spellstrike without becoming the active GM',()=>setup({charged:false},async f=>{
 f.game.user={id:'player',isGM:false};f.actor.testUserPermission=(user,permission)=>user.id==='player'&&permission==='OWNER';let onRest,writes=0;const errors=[],update=f.actor.update;
 f.actor.update=async changes=>{writes++;return update.call(f.actor,changes)};
 const provider=api.createSpellCombination({game:f.game,nativeCasts:f.casts,onError:error=>errors.push(error)});
 provider.register({Hooks:{on:(event,callback)=>{assert.equal(event,'pf2e.restForTheNight');onRest=callback;return 1},off(){}}});
 onRest(f.actor);await new Promise(resolve=>setImmediate(resolve));
 assert.equal(writes,1);assert.equal(f.actor.flags[NS].spellstrike.charged,true);assert.equal(errors.length,0);assert.equal(f.game.users.activeGM.id,'gm');
}));
for(const kind of ['strike','combination'])test(`${kind} stops after GM handoff during a player choice before payment, record or attack`,()=>setup({kind},async f=>{
 const use=runWith(f,{choose:async({choices})=>{handoff(f);return choices[0].value}});await assert.rejects(use,/GM/);assert.equal(f.calls.filter(c=>['attack','payment'].includes(c.kind)).length,0);assert.equal(f.message.flags[NS].spellCombinationUse,undefined);assert.equal(f.actor.flags[NS].spellstrike.charged,true);
}));
test('a queued usage rechecks GM authority before its first choice or mutation',()=>setup({kind:'combination'},async f=>{
 const promise=f.use();handoff(f);await assert.rejects(()=>promise,/GM|无权/);assert.equal(f.calls.length,0);assert.equal(f.message.flags[NS].spellCombinationUse,undefined);
}));
test('handoff after the first Combination hit stops the second hit and preserves the started receipt',()=>setup({kind:'combination'},async f=>{
 await assert.rejects(runWith(f,{afterAttack:async()=>handoff(f)}),/GM/);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);assert.equal(f.messages.length,1);assert.equal(f.message.flags[NS].spellCombinationUse.state,'started');assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,0);
}));
test('native attack callback cannot publish a card after GM handoff during the roll',()=>setup({kind:'combination'},async f=>{
 const variant=f.actor.system.actions[0].variants[0],native=variant.roll;variant.roll=async opts=>{handoff(f);return native(opts)};
 await assert.rejects(f.use,/GM/);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);assert.equal(f.messages.length,0);assert.equal(f.message.flags[NS].spellCombinationUse.state,'started');
}));
test('GM handoff after an already-paid spell keeps the started receipt and blocks discharge writes',()=>setup({},async f=>{
 const pay=f.casts.payForActivity;f.casts.payForActivity=async ctx=>{const result=await pay(ctx);handoff(f);return result};
 await assert.rejects(f.use,/GM/);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='attack').length,0);assert.equal(f.message.flags[NS].spellCombinationUse.state,'started');assert.equal(f.actor.flags[NS].spellstrike.charged,true);
}));
test('a spell-card draft returning after GM handoff cannot publish or bind the canonical card',()=>setup({},async f=>{
 const toMessage=f.spell.toMessage;f.spell.toMessage=async(...args)=>{const result=await toMessage.apply(f.spell,args);handoff(f);return result};
 await assert.rejects(f.use,/GM/);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);assert.equal(f.calls.filter(c=>c.kind==='bind-payment').length,0);assert.equal(f.messages.length,1);assert.equal(f.message.flags[NS].spellCombinationUse.state,'started');assert.equal(f.actor.flags[NS].spellstrike.charged,false);
}));
test('a completed Toolbelt merge after GM handoff cannot publish damage or mark the activity done',()=>setup({kind:'combination'},async f=>{
 const merge=f.game.toolbelt.api.betterChat.mergeDamageMessages;f.game.toolbelt.api.betterChat.mergeDamageMessages=async(...args)=>{const result=await merge(...args);handoff(f);return result};
 await assert.rejects(f.use,/GM/);assert.equal(f.calls.filter(c=>c.kind==='attack').length,2);assert.equal(f.messages.length,2);assert.equal(f.message.flags[NS].spellCombinationUse.state,'started');
}));
test('exact source routing rejects same-name and same-slug items',()=>setup({},async f=>{assert.equal(f.provider.resolveAction(f.feat),'spell-combination:strike');assert.equal(f.provider.resolveAction({type:'action',name:'Spellstrike',system:{slug:'spellstrike'}}),null)}));
test('Overwhelming Combination uses weapon and fist, successive MAP, and one merged IWR card',()=>setup({kind:'combination',map:1},async f=>{await f.use();const attacks=f.calls.filter(c=>c.kind==='attack');assert.deepEqual(attacks.map(c=>c.index),[0,1]);assert.deepEqual(attacks.map(c=>c.tier),[1,2]);assert.equal(f.merges.length,1);assert.equal(f.messages.at(-1).rolls[0].total,20);assert.equal(f.calls.filter(c=>c.kind==='payment').length,0)}));

test('Overwhelming Combination accepts a one-handed base weapon wielded in two hands',()=>setup({kind:'combination'},async f=>{
 const weapon=f.weapons[0];weapon.hands='1';weapon.system.equipped.handsHeld=2;await f.use();assert.equal(f.calls.filter(c=>c.kind==='attack').length,2);
}));
test('a two-handed weapon held in only one hand is not a wielded Combination weapon',()=>setup({kind:'combination'},async f=>{
 const weapon=f.weapons[0];weapon.hands='2';weapon.system.equipped.handsHeld=1;weapon.system.traits.value=['finesse'];await assert.rejects(f.use());assert.equal(f.calls.filter(c=>c.kind==='attack').length,0);
}));
test('Spellstrike consumes once before attacking and combines attack spell damage at its Strike degree',()=>setup({outcomes:['criticalSuccess']},async f=>{await f.use();assert.deepEqual(f.calls.filter(c=>['payment','attack','spell-damage'].includes(c.kind)).map(c=>c.kind),['payment','attack','spell-damage']);assert.equal(f.messages.at(-1).rolls[0].total,52);assert.equal(f.actor.flags[NS].spellstrike.charged,false);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1)}));
for(const kind of ['strike','swipe','combination'])test(`${kind} carries only each attack's witnessed off-guard into deferred native weapon damage`,()=>setup({kind},async f=>{
 const affected=kind==='strike'?0:1;
 for(const strike of f.actor.system.actions)for(const variant of strike.variants){const native=variant.roll;variant.roll=opts=>native({...opts,callback:async(roll,outcome,raw)=>{
  if(f.calls.filter(c=>c.kind==='attack').length-1===affected)raw.flags.pf2e.context.options.push('target:condition:off-guard','attack-only:must-not-carry');
  return opts.callback(roll,outcome,raw);
 }})}
 await runWith(f,{afterAttack:async message=>{message.flags.pf2e.context.options=message.flags.pf2e.context.options.filter(o=>o!=='target:condition:off-guard')}})();
 const damage=f.calls.filter(c=>c.kind==='weapon-damage');assert.equal(damage.length,kind==='strike'?1:2);
 for(const [index,call]of damage.entries()){
  assert.equal(call.opts.options.has('target:condition:off-guard'),index===affected,'later checks and consumed temporary effects must not change the original attack fact');
  assert.equal(call.opts.options.has('attack-only:must-not-carry'),false);
  assert.equal(call.opts.target.document.uuid,f.targetDocs[kind==='swipe'?index:0].uuid);
 }
}));
for(const target of [null,{token:'Scene.s.Token.someone-else'}])test(`off-guard from a missing or different attack target is not copied (${target?.token??'missing'})`,()=>setup({},async f=>{
 const variant=f.actor.system.actions[0].variants[0],native=variant.roll;variant.roll=opts=>native({...opts,callback:async(roll,outcome,raw)=>{raw.flags.pf2e.context.target=target;raw.flags.pf2e.context.options.push('target:condition:off-guard');return opts.callback(roll,outcome,raw)}});
 await f.use();assert.equal(f.calls.find(c=>c.kind==='weapon-damage').opts.options.has('target:condition:off-guard'),false);
}));

function tumbleBehind(f,{genuineOffGuard=false}={}){
 const effect={id:'tumble',type:'effect',sourceId:'Compendium.patreon-v3.effects.Item.Vh5E1Qgp34sTKfVs',actor:f.actor,system:{rules:[{key:'EphemeralEffect',selectors:['attack-roll','damage'],predicate:['self:mark:enemy-tumble-behind'],uuid:'Compendium.pf2e.conditionitems.Item.AJh5ex99aV6VTggg'},{key:'TokenMark',slug:'enemy-tumble-behind',uuid:f.targetDocs[0].uuid}],start:{value:100,initiative:15}}};
 f.actor.items.push(effect);
 const condition={id:'real-offguard',type:'condition',slug:'off-guard'};
 f.targetDocs[0].actor.items=genuineOffGuard?[condition]:[];
 f.actor.deleteEmbeddedDocuments=async(type,ids)=>{assert.equal(type,'Item');f.calls.push({kind:'consume-tumble',ids});for(const id of ids){const index=f.actor.items.findIndex(i=>i.id===id);if(index>=0)f.actor.items.splice(index,1)}};
 for(const strike of f.actor.system.actions)for(const variant of strike.variants){const native=variant.roll;variant.roll=opts=>{
  const active=f.actor.items.includes(effect),offGuard=genuineOffGuard||active&&opts.target.document.uuid===effect.system.rules[1].uuid;
  f.calls.push({kind:'tumble-at-attack',active,offGuard});
  return native({...opts,callback:async(result,outcome,raw)=>{if(active)raw.flags.pf2e.context.options.push('self:effect:off-guard-tumble-behind');if(offGuard)raw.flags.pf2e.context.options.push('target:condition:off-guard');return opts.callback(result,outcome,raw)}});
 }}
 return {effect,condition};
}
for(const kind of ['combination','swipe','dual'])for(const first of ['success','criticalSuccess','failure'])test(`${kind} consumes Tumble Behind before its next attack while preserving the first delayed damage (${first})`,()=>setup({kind:kind==='dual'?'strike':kind,outcomes:[first,'success']},async f=>{
 const {effect}=tumbleBehind(f);
 await (kind==='dual'?dualUse(f):f.use)();
 assert.deepEqual(f.calls.filter(c=>c.kind==='tumble-at-attack').map(c=>c.active),[true,false]);
 const damage=f.calls.filter(c=>c.kind==='weapon-damage');
 assert.deepEqual(damage.map(c=>c.opts.options.has('target:condition:off-guard')),first==='failure'?[false]:[true,false]);
 assert.equal(f.actor.items.includes(effect),false);
 assert.equal(f.calls.filter(c=>c.kind==='consume-tumble').length,1);
 assert.equal(f.messages.filter(m=>m.flags[NS]?.spellCombinationDamage||m.flags[NS]?.dualStrike).length,kind==='swipe'&&first!=='failure'?2:1);
 assert(f.messages.filter(m=>m.flags.pf2e?.context?.type==='damage-roll').every(m=>!m.flags.pf2e.context.options.includes('self:effect:off-guard-tumble-behind')),'delayed cards cannot delete a newly gained use through the Patreon marker');
}));
test('the next Combination attack waits for effect consumption to complete',()=>setup({kind:'combination'},async f=>{
 tumbleBehind(f);let enter,release;const entered=new Promise(resolve=>{enter=resolve}),pending=new Promise(resolve=>{release=resolve});
 const nativeDelete=f.actor.deleteEmbeddedDocuments;f.actor.deleteEmbeddedDocuments=async(...args)=>{enter();await pending;return nativeDelete(...args)};
 const use=f.use();await entered;
 assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);
 assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,0);
 release();await use;assert.deepEqual(f.calls.filter(c=>c.kind==='tumble-at-attack').map(c=>c.active),[true,false]);
}));
for(const kind of ['combination','dual'])test(`${kind} stops before its second Strike when effect settlement fails and cannot replay the first Strike`,()=>setup({kind:kind==='dual'?'strike':kind},async f=>{
 tumbleBehind(f);f.actor.deleteEmbeddedDocuments=async()=>{throw Error('database unavailable')};const use=kind==='dual'?dualUse(f):f.use;
 await assert.rejects(use,/database unavailable/);
 if(kind==='dual')await use();else await assert.rejects(use,/未完成|重试|确认/);
 assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);
 assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,0);
}));
test('consuming Tumble Behind does not remove genuine target off-guard or limit Combination precision to one hit',()=>setup({kind:'combination'},async f=>{
 const {condition}=tumbleBehind(f,{genuineOffGuard:true});
 for(const strike of f.actor.system.actions)strike.damage=async opts=>{
  f.calls.push({kind:'weapon-damage',opts});
  // A native result with precision on each independently off-guard hit. The
  // activity must preserve both; only Double Slice has a once-only restriction.
  const result=roll(13);result.instances=[{total:13,componentTotal:type=>type==='precision'?3:0,head:{options:{}}}];return result;
 };
 await f.use();
 assert.deepEqual(f.calls.filter(c=>c.kind==='tumble-at-attack').map(c=>c.active),[true,false]);
 assert.deepEqual(f.calls.filter(c=>c.kind==='weapon-damage').map(c=>c.opts.options.has('target:condition:off-guard')),[true,true]);
 assert.deepEqual(f.targetDocs[0].actor.items,[condition]);
 assert.equal(f.messages.at(-1).rolls[0].total,26);
}));
test('Spellstrike with save spell still rolls save and damage when the Strike misses',()=>setup({spellSave:true,outcomes:['failure']},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='save').length,1);assert.equal(f.messages.at(-1).rolls[0].total,16);assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,0)}));
test('Spellstrike critical failure spends the spell but disrupts saves and spell damage',()=>setup({spellSave:true,outcomes:['criticalFailure']},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='save'||c.kind==='spell-damage').length,0);assert.equal(f.actor.flags[NS].spellstrike.charged,false)}));
test('a successful basic save halves only the spell component before merging weapon damage',()=>setup({spellSave:true,save:'success'},async f=>{await f.use();assert.equal(f.messages.at(-1).rolls[0].total,18)}));
test('save failure on a critical Strike does not double the save spell',()=>setup({spellSave:true,outcomes:['criticalSuccess']},async f=>{await f.use();assert.equal(f.messages.at(-1).rolls[0].total,36)}));
test('Spell Swipe rolls the same weapon against adjacent targets at the same MAP and pays once',()=>setup({kind:'swipe',map:1,multi:true},async f=>{await f.use();assert.deepEqual(f.calls.filter(c=>c.kind==='attack').map(c=>[c.index,c.tier]),[[0,1],[0,1]]);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='spell-damage').length,2);assert.equal(f.merges.length,2)}));
for(const first of ['failure','criticalFailure','success','criticalSuccess'])test(`Spell Swipe derives Backswing only from its own preceding miss (${first})`,()=>setup({kind:'swipe',outcomes:[first,'success']},async f=>{
 f.weapons[0].system.traits.value=['sweep','backswing'];await f.use();
 const attacks=f.calls.filter(c=>c.kind==='attack');
 assert(attacks.every(c=>c.opts.options.has('sweep-bonus')),'Spell Swipe explicitly grants Sweep on both Strikes');
 assert.equal(attacks[0].opts.options.has('backswing-bonus'),false);
 assert.equal(attacks[1].opts.options.has('backswing-bonus'),['failure','criticalFailure'].includes(first));
}));
for(const first of ['success','failure'])test(`deferred Spell Swipe damage grants Forceful only to its second Strike (${first})`,()=>setup({kind:'swipe',map:2,outcomes:[first,'criticalSuccess']},async f=>{
 f.weapons[0].system.traits.value=['forceful'];
 const nativeClone=f.actor.clone;
 f.actor.clone=changes=>{
  const rules=changes.items.at(-1).system.rules;
  if(!rules.some(r=>r.key==='FlatModifier'))return nativeClone(changes);
  f.calls.push({kind:'sequence-damage-clone',rules});
  return {system:{actions:f.actor.system.actions.map(s=>({...s,critical:async opts=>{f.calls.push({kind:'sequence-critical',opts});return roll(22)}}))}};
 };
 await f.use();
 assert.equal(f.calls.filter(c=>c.kind==='sequence-damage-clone').length,1,'only a completed prior attack proves the bonus, independently of MAP');
 const damage=f.calls.filter(c=>c.kind==='sequence-critical');assert.equal(damage.length,1);
 assert.equal(damage[0].opts.target.document.uuid,f.targetDocs[1].uuid);
 assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,first==='success'?1:0);
 assert.deepEqual(f.actor._source.items,[]);
}));
test('Overwhelming Combination does not transfer same-weapon history from sword to fist',()=>setup({kind:'combination',outcomes:['failure','success']},async f=>{
 for(const w of f.weapons)w.system.traits.value.push('forceful','backswing','sweep');
 await f.use();const attacks=f.calls.filter(c=>c.kind==='attack');
 assert(attacks.every(c=>!c.opts.options.has('backswing-bonus')&&!c.opts.options.has('sweep-bonus')));
 assert.equal(f.calls.filter(c=>c.kind==='infusion-clone').length,0);
}));
test('single-target Spell Swipe affects only the selected target with the spell',()=>setup({kind:'swipe'},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='spell-damage').length,1);assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,2);assert.equal(f.merges.length,1)}));
test('discharged Spellstrike rejects use before payment or attack',()=>setup({charged:false},async f=>{await assert.rejects(f.use,/充能/);assert.equal(f.calls.length,0)}));
test('unavailable spell resources cause no attack or discharge',()=>setup({consumeFails:true},async f=>{await assert.rejects(f.use,/资源/);assert.equal(f.calls.filter(c=>c.kind==='attack').length,0);assert.equal(f.actor.flags[NS].spellstrike.charged,true)}));
test('replaying the original activity cannot spend or attack twice',()=>setup({},async f=>{await f.use();await f.use();assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1)}));
test('a partial native attack failure cannot replay an already paid activity',()=>setup({},async f=>{f.actor.system.actions[0].variants[0].roll=async()=>{f.calls.push({kind:'attack'});throw Error('native failure')};await assert.rejects(f.use,/native failure/);await assert.rejects(f.use,/未完成|重试|确认/);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1)}));
test('Spell Swipe leaves adjacency to the table and keeps both original targets with one payment',()=>setup({kind:'swipe'},async f=>{f.targetDocs[0].object.distanceTo=()=>{throw Error('adjacency probe must not run')};await f.use();assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.deepEqual(f.calls.filter(c=>c.kind==='attack').map(c=>c.opts.target.document.uuid),f.targetDocs.map(t=>t.uuid))}));
test('Spell Swipe cannot substitute a token from another scene',()=>setup({kind:'swipe'},async f=>{f.targetDocs[1].parent={id:'other'};await assert.rejects(f.use,/场景|目标/);assert.equal(f.calls.filter(c=>c.kind==='payment'||c.kind==='attack').length,0)}));
test('normal player variant selection is carried into payment and damage',()=>setup({overlays:true},async f=>{await f.use();assert.deepEqual(f.calls.find(c=>c.kind==='payment').ctx.item.appliedOverlays,new Map([['override','variant1']]));assert.deepEqual(f.calls.find(c=>c.kind==='spell-damage').overlays,['variant1'])}));
test('same-name action variants remain distinguishable and retain the selected heightened slot',()=>setup({},async f=>{
 f.spell.name='Blazing Bolt';f.spell.rank=3;f.spell.isCantrip=false;
 f.spell.system.time.value='1 to 3';
 f.spell.overlays=new Map([['two-actions',{overlayType:'override',sort:2,time:'2 or 3'}],['one-action',{overlayType:'override',sort:1,time:'1'}]]);
 const load=f.spell.loadVariant;f.spell.loadVariant=function(options={}){const variant=load.call(this,options),overlay=this.overlays.get(options.overlayIds?.[0]);return {...variant,system:{...variant.system,time:{value:overlay?.time??this.system.time.value}}}};
 f.entry.getSheetData=async()=>({groups:[{id:7,active:[null,{spell:f.spell,castRank:7,expended:false}]}]});
 let shown;
 await runWith(f,{choose:async({title,choices})=>{if(title==='选择法术变体'){shown=choices;assert.notEqual(choices[0].label,choices[1].label,'action count is necessary to choose the right native damage');assert.deepEqual(choices.map(c=>c.value),['one-action','two-actions']);assert.match(choices[0].label,/1.*动作/);assert.match(choices[1].label,/2.*动作/);return choices[1].value}return choices[0].value}})();
 assert(shown);const payment=f.calls.find(c=>c.kind==='payment').ctx;
 assert.equal(payment.rank,7);assert.equal(payment.slotId,1);assert.equal(payment.item.rank,7);
 assert.deepEqual([...payment.item.appliedOverlays.values()],['two-actions']);
 assert.deepEqual(f.calls.find(c=>c.kind==='spell-damage').overlays,['two-actions']);
 assert.equal(f.messages.find(m=>m.flags[NS]?.spellCombination)?.flags.pf2e.origin.castRank,7);
}));
for(const kind of ['strike','combination','swipe'])test(`${kind} settles native attacks without measuring range or collision`,()=>setup({kind},async f=>{const forbidden=()=>{throw Error('spatial probe must not run')};f.actor.getReach=forbidden;f.origin.object.distanceTo=forbidden;f.origin.object.checkCollision=forbidden;await f.use();assert.equal(f.calls.filter(c=>c.kind==='attack').length,kind==='strike'?1:2);assert.equal(f.message.flags[NS].spellCombinationUse.state,'done')}));
test('native generated attack cards suppress duplicate Workbench auto damage',()=>setup({},async f=>{await f.use();const attacks=f.messages.filter(m=>m.flags[NS]?.spellCombinationAttack);assert.equal(attacks.length,1);assert.equal(attacks[0].flags['xdy-pf2e-workbench'].noAutoDamageRoll,true)}));
test('unprepared and expended spells are never choices even if in the spellbook',()=>setup({},async f=>{f.entry.getSheetData=async()=>({groups:[{id:8,active:[{spell:f.spell,castRank:8,expended:true}]}]});await assert.rejects(f.use,/法术/);assert.equal(f.calls.length,0)}));
test('Spell Swipe enables the native sweep-bonus predicate against both targets',()=>setup({kind:'swipe'},async f=>{f.weapons[0].system.traits.value=['sweep'];await f.use();assert(f.calls.filter(c=>c.kind==='attack').every(c=>c.opts.options.has('sweep-bonus')))}));
test('Overwhelming Combination allows fist first and waits for first-hit providers before the next strike',async()=>{
 const prior=globalThis.CONFIG;try{const f=fixture({kind:'combination'});globalThis.CONFIG={ChatMessage:{documentClass:f.Message}};let after=0;
 const provider=api.createSpellCombination({game:f.game,nativeCasts:f.casts,nativeOperations:f.nativeOperations,fromUuid:async id=>f.targetDocs.find(t=>t.uuid===id),choose:async({title,choices})=>title.includes('攻击顺序')?'fist':title.includes('多重')?'0':choices[0].value,afterAttack:async()=>{after++;f.calls.push({kind:'after-attack'})}});
 await provider.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:f.game.users.get('player'),action:provider.resolveAction(f.feat)});
 assert.deepEqual(f.calls.filter(c=>['attack','after-attack'].includes(c.kind)).map(c=>c.kind),['attack','after-attack','attack','after-attack']);assert.deepEqual(f.calls.filter(c=>c.kind==='attack').map(c=>c.index),[1,0]);assert.equal(after,2);
 }finally{globalThis.CONFIG=prior}
});
test('Disintegrate critical lowers the save through a transient native RuleElement, not unsupported check options',()=>setup({spellSave:true,outcomes:['criticalSuccess']},async f=>{
 f.spell.sourceId=S.disintegrate??'Compendium.pf2e.spells-srd.Item.r7ihOgKv19eJQnik';let cloned;
 const target=f.targetDocs[0].actor;target._source={items:[]};target.clone=(changes)=>{cloned=changes;return target};await f.use();assert(cloned.items.some(i=>i.system.rules.some(r=>r.key==='AdjustDegreeOfSuccess'&&r.adjustment.all==='one-degree-worse')));assert.equal(f.calls.find(c=>c.kind==='save').opts.dosAdjustments,undefined);
}));
test('Spellstrike gives the selected native Strike arcane and magical traits in an ephemeral actor only',()=>setup({},async f=>{await f.use();const data=f.calls.find(c=>c.kind==='infusion-clone')?.changes;assert(data);const rules=data.items.at(-1).system.rules;assert(rules.some(r=>r.key==='AdjustStrike'&&r.property==='traits'&&r.value==='arcane'));assert(rules.some(r=>r.property==='weapon-traits'&&r.value==='magical'));assert(rules.every(r=>r.definition.includes('item:id:weapon')));assert.equal(f.actor._source.items.length,0)}));
test('Ignition and Needle Darts critical persistent components use actual heightened rank and selected variant',()=>{
 assert.equal(api.criticalSpellPersistentFormula({sourceId:'Compendium.pf2e.spells-srd.Item.6DfLZBl8wKIV03Iq',rank:8,system:{range:{value:'touch'}}}),'8d6[persistent,fire]');
 assert.equal(api.criticalSpellPersistentFormula({sourceId:'Compendium.pf2e.spells-srd.Item.6DfLZBl8wKIV03Iq',rank:8,system:{range:{value:'30 feet'}}}),'8d4[persistent,fire]');
 assert.equal(api.criticalSpellPersistentFormula({sourceId:'Compendium.pf2e.spells-srd.Item.iYRDFxeVpJ5KIjmr',rank:8}),'8[persistent,bleed]');
 assert.equal(api.criticalSpellPersistentFormula({name:'Ignition',rank:8}),null);
});
test('multi-target Spell Swipe save spell extends only to enemies hit, per its specific legacy wording',()=>setup({kind:'swipe',multi:true,spellSave:true,outcomes:['failure','success']},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='save').length,1);assert.equal(f.calls.find(c=>c.kind==='save').target.uuid,f.targetDocs[1].uuid)}));
test('single-target Spell Swipe save keeps normal Spellstrike miss semantics for the chosen target',()=>setup({kind:'swipe',spellSave:true,outcomes:['failure','success']},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='save').length,1);assert.equal(f.calls.find(c=>c.kind==='save').target.uuid,f.targetDocs[0].uuid)}));
test('critical failure keeps one paid native spell card without rolling its save or damage',()=>setup({spellSave:true,outcomes:['criticalFailure']},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='finish-payment').length,0);assert.equal(f.calls.filter(c=>c.kind==='spell-card').length,1);assert.equal(f.calls.filter(c=>c.kind==='save'||c.kind==='spell-damage').length,0)}));
test('merged damage preserves native spell and weapon roll options needed by IWR',()=>setup({},async f=>{
 f.spell.getDamage=async()=>({template:{damage:{roll:roll(16)}},context:{options:new Set(['item:trait:arcane','item:type:spell','self:trait:elf']),domains:['damage','spell-damage']}});
 f.weapons[0].getRollOptions=()=>['item:type:weapon','item:trait:magical'];await f.use();const options=f.messages.at(-1).flags.pf2e.context.options;
 for(const key of ['item:trait:arcane','item:type:spell','self:trait:elf','item:type:weapon','item:trait:magical'])assert(options.includes(key),key);
}));

test('spell combination keeps unique weapon fire bypass in the final combined card',()=>setup({},async f=>{
 globalThis.CONFIG.PF2E={damageTypes:{fire:'Fire',slashing:'Slashing'}};
 const strike=f.actor.system.actions[0],original=strike.damage;strike.damage=async opts=>{const r=await original(opts);r.instances=[{type:'fire',total:10,persistent:false}];r.options.bypass={resistance:{ignore:[{type:'fire',max:Infinity}]}};return r};
 const nativeSpell=f.spell.getDamage;f.spell.getDamage=async opts=>{const data=await nativeSpell.call(f.spell,opts);data.template.damage.roll.instances=[{type:'slashing',total:16,persistent:false}];return data};
 await f.use();const message=f.messages.find(m=>m.flags?.[NS]?.spellCombinationDamage);assert.deepEqual(message.rolls[0].options.bypass?.resistance.ignore,[{type:'fire',max:Infinity}]);
}));
test('merged critical Strike damage retains native degreeOfSuccess and outcome predicate for critical-hit immunity',()=>setup({outcomes:['criticalSuccess']},async f=>{await f.use();assert.equal(f.messages.at(-1).rolls[0].options.degreeOfSuccess,3);assert(f.messages.at(-1).flags.pf2e.context.options.includes('check:outcome:critical-success'))}));
test('multi-target basic-save Spell Swipe reuses one native spell damage roll for both targets',()=>setup({kind:'swipe',multi:true,spellSave:true},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='save').length,2);assert.equal(f.calls.filter(c=>c.kind==='spell-damage').length,1);assert.equal(f.messages.filter(m=>m.flags[NS]?.spellCombinationDamage).length,2)}));
test('position changes after the first Spell Swipe Strike do not interrupt its paid settlement',()=>setup({kind:'swipe',multi:true},async f=>{const original=f.actor.system.actions[0].variants[0].roll;f.actor.system.actions[0].variants[0].roll=async opts=>{const r=await original(opts);f.origin.object.distanceTo=()=>20;return r};await f.use();await f.use();assert.equal(f.calls.filter(c=>c.kind==='attack').length,2);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.message.flags[NS].spellCombinationUse.state,'done');assert.deepEqual(f.messages.filter(m=>m.flags[NS]?.spellCombinationDamage).map(m=>m.flags['pf2e-toolbelt'].targetHelper.targets),f.targetDocs.map(t=>[t.uuid]))}));
test('native PF2e unarmed category identifies Fist without a nonexistent isUnarmed getter',()=>setup({kind:'combination'},async f=>{for(const w of f.weapons){w.category=w.isUnarmed?'unarmed':'martial';delete w.isUnarmed}await f.use();assert.deepEqual(f.calls.filter(c=>c.kind==='attack').map(c=>c.index),[0,1])}));
test('native merge boundaries prevent deterministic simplification without replacing precision or splash flavor',()=>{assert.equal(typeof api.preserveDamagePartForMerge,'function');const roll={instances:[{head:{options:{}}},{head:{options:{flavor:'precision'}}}]};assert.equal(api.preserveDamagePartForMerge(roll),roll);assert.equal(roll.instances[0].head.options.flavor,'damage');assert.equal(roll.instances[1].head.options.flavor,'precision')});
test('Spell Swipe treats native one-or-two creature target ranges as multi-target',()=>setup({kind:'swipe',spellSave:true},async f=>{f.spell.system.target.value='1 or 2 creatures';await f.use();assert.equal(f.calls.filter(c=>c.kind==='save').length,2)}));

// Only the newly opted-in activity changes its legacy display-card behavior.
test('Combination declares exact-source actual-Use policy without changing other activities',()=>setup({kind:'combination'},async f=>{
 assert.equal(typeof f.provider.requiresActualUse,'function');
 assert.equal(f.provider.requiresActualUse(f.feat,'spell-combination:combination'),true);
 assert.equal(f.provider.requiresActualUse({...f.feat,sourceId:S.strike},'spell-combination:strike'),false);
 assert.equal(f.provider.requiresActualUse({...f.feat,sourceId:'Compendium.other.Item.fake'},'spell-combination:combination'),false);
 assert.equal(f.provider.requiresActualUse(f.feat,'other:combination'),false);
}));
for(const actualUse of [false,undefined,'true'])test(`Combination rejects display-only direct execution (${actualUse}) before choices or mutations`,()=>setup({kind:'combination'},async f=>{
 f.message.flags[NS].usageInput.actualUse=actualUse;
 let choices=0;const use=runWith(f,{choose:async()=>{choices++;return '0'}});
 await assert.rejects(use,/实际使用|actual use/i);
 assert.equal(choices,0);assert.equal(f.calls.length,0);assert.equal(f.messages.length,0);assert.equal(f.message.flags[NS].spellCombinationUse,undefined);
}));
test('Combination accepts Toolbelt native original use-action marker without the local flag',()=>setup({kind:'combination'},async f=>{
 delete f.message.flags[NS].usageInput.actualUse;f.message.flags.pf2e.origin.rollOptions=['origin:action:slug:use-action'];
 await f.use();assert.equal(f.calls.filter(c=>c.kind==='attack').length,2);assert.equal(f.message.flags[NS].spellCombinationUse.state,'done');
}));
for(const kind of ['strike','swipe'])test(`${kind} preserves existing original-card behavior without actualUse`,()=>setup({kind},async f=>{
 delete f.message.flags[NS].usageInput.actualUse;await f.use();assert.ok(f.calls.some(c=>c.kind==='attack'));assert.equal(f.message.flags[NS].spellCombinationUse.state,'done');
}));

const volatileSource='Compendium.pf2e.classfeatures.Item.aRDNtB0mh6MMRSKf';
const boomingSource='Compendium.pf2e.spells-srd.Item.DNvfKfDSs9DstDfT';
function firearm(f,{study=true,concussive=true}={}){
 f.game.modules.set('pf2e-ranged-combat',{active:true});
 if(study)f.actor.items.push({id:'study',actor:f.actor,type:'feat',sourceId:volatileSource,system:{}});
 const weapon=f.weapons[0];weapon.isMelee=false;weapon.isRanged=true;weapon.system.group='firearm';weapon.system.traits.value=concussive?['concussive']:[];
 return weapon;
}
for(const slug of ['double-barreled-pistol','mace-multipistol'])test(`Volatile Spark uses the held ${slug} native ranged Strike and pays once`,()=>setup({},async f=>{
 firearm(f).slug=slug;let confirmed=0;
 const use=runWith(f,{volatileSparkEnabled:true,choose:async({title,choices})=>{if(title.includes('第一射程增量'))confirmed++;return choices[0].value;}});
 await use();await use();assert.equal(confirmed,1);assert.equal(f.calls.filter(c=>c.kind==='attack')[0].index,0);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.actor.flags[NS].spellstrike.charged,false);
 assert.equal(f.calls.find(c=>c.kind==='attack').opts.consumeAmmo,undefined,'Ranged Combat owns ammunition');
}));
for(const mode of ['no-study','gate-off','crossbow'])test(`ranged Spellstrike remains unavailable for ${mode}`,()=>setup({},async f=>{
 const gun=firearm(f,{study:mode!=='no-study'});if(mode==='crossbow')gun.system.group='bow';
 const seen=[];await runWith(f,{volatileSparkEnabled:mode!=='gate-off',choose:async({title,choices})=>{if(title.includes('攻击'))seen.push(...choices);return choices[0].value;}})();
 assert.equal(f.calls.find(c=>c.kind==='attack').index,1);
}));
test('a Volatile Spark combination firearm retains its melee mode without a range prompt',()=>setup({},async f=>{
 firearm(f);const melee={...f.actor.system.actions[0],item:{...f.weapons[0],isMelee:true,isRanged:false,altUsageType:'melee'}};f.actor.system.actions[0].altUsages=[melee];
 await runWith(f,{volatileSparkEnabled:true,choose:async({title,choices})=>{assert(!title.includes('第一射程增量'));return title.includes('武器')?choices.find(c=>c.value.endsWith(':melee')).value:choices[0].value;}})();
 assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);
}));
test('declining the first range increment leaves resources and charge intact',()=>setup({},async f=>{
 firearm(f);const result=await runWith(f,{volatileSparkEnabled:true,choose:async({title,choices})=>title.includes('第一射程增量')?'cancel':choices[0].value})();
 assert.equal(result.status,'cancelled');assert.equal(f.calls.filter(c=>c.kind==='attack'||c.kind==='payment').length,0);assert.equal(f.actor.flags[NS].spellstrike.charged,true);
}));
test('a native empty-firearm refusal does not pay for Spellstrike',()=>setup({},async f=>{
 firearm(f);const result=await runWith(f,{volatileSparkEnabled:true,nativeOperations:{run:async()=>({status:'cancelled'}),register(){}}})();
 assert.equal(result.status,'cancelled');assert.equal(f.calls.filter(c=>c.kind==='payment').length,0);assert.equal(f.actor.flags[NS].spellstrike.charged,true);
}));
test('concussive Spellstrike keeps one native damage application with a targeted spell provenance',()=>setup({},async f=>{
 firearm(f);await runWith(f,{volatileSparkEnabled:true})();
 const damage=f.messages.filter(m=>m.flags[NS]?.spellCombinationDamage);assert.equal(damage.length,1);assert.equal(f.merges.length,1);
 assert.equal(damage[0].flags[NS].volatileSparkSpellDamage.spellUuid,f.spell.uuid);assert.equal(damage[0].flags[NS].volatileSparkSpellDamage.targetUuid,f.targetDocs[0].uuid);
 assert.equal(damage[0].rolls[0].total,26);
}));
for(const outcome of ['failure','criticalFailure','success'])test(`Booming Blast ${outcome} still rolls Will and charges exactly once`,()=>setup({outcomes:[outcome],save:'criticalFailure',charged:false},async f=>{
 firearm(f);const spellstrike={...f.feat,id:'spellstrike',uuid:'Actor.pc.Item.spellstrike'};f.actor.items.push(spellstrike);
 Object.assign(f.feat,{type:'spell',sourceId:boomingSource,spellcasting:f.entry,rank:3});f.feat.system={traits:{value:['focus','magus','mental','emotion']},time:{value:'1'}};
 const condition=nativeFrightened(f,{current:3});
 const use=runWith(f,{volatileSparkEnabled:true});await use();await use();
 assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);assert.equal(f.calls.filter(c=>c.kind==='save').length,1);assert.equal(f.calls.find(c=>c.kind==='save').opts.dc.value,35);
 assert.equal(condition.changes,0);assert.equal(condition.value,3);assert.equal(f.calls.filter(c=>c.kind==='bind-payment').length,1);assert.equal(f.actor.flags[NS].spellstrike.charged,true);
}));
test('Reloading Cascade owns one reload and never recharges Spellstrike',()=>setup({charged:false},async f=>{
 firearm(f);f.feat.sourceId='Compendium.pf2e.actionspf2e.Item.tmbf6xouEnRXCnKR';let reloads=0;
 const use=runWith(f,{volatileSparkEnabled:true,volatileSparkReload:{async run(ctx){reloads++;assert.equal(ctx.weaponId,'weapon');ctx.assertLive();return {status:'done'}},register(){}}});
 await use();await use();assert.equal(reloads,1);assert.equal(f.actor.flags[NS].spellstrike.charged,false);assert.equal(f.calls.filter(c=>c.kind==='payment'||c.kind==='bind-payment'||c.kind==='attack').length,0);
}));
test('Reloading Cascade display card cannot reload or create a stance',()=>setup({},async f=>{
 firearm(f);f.feat.sourceId='Compendium.pf2e.actionspf2e.Item.tmbf6xouEnRXCnKR';delete f.message.flags[NS].usageInput.actualUse;
 await assert.rejects(runWith(f,{volatileSparkEnabled:true,volatileSparkReload:{run(){throw Error('unexpected reload')},register(){}}})(),/实际使用/);
}));
test('maintenance grants the original Reloading Cascade once even when Recharge already exists',()=>setup({},async f=>{
 firearm(f);f.actor.items.push({type:'action',flags:{[NS]:{spellstrikeRecharge:true}}});const created=[];
 f.actor.createEmbeddedDocuments=async(_type,items)=>{created.push(...items);f.actor.items.push(...items.map(i=>({...i,actor:f.actor,sourceId:i._stats.compendiumSource})));};
 const source='Compendium.pf2e.actionspf2e.Item.tmbf6xouEnRXCnKR',provider=api.createSpellCombination({game:f.game,volatileSparkEnabled:true,nativeCasts:f.casts,nativeOperations:f.nativeOperations,fromUuid:async uuid=>uuid===source?{toObject:()=>({_id:'native',type:'action',name:'Reloading Cascade',system:{actions:{value:1}}})}:null});
 await provider.maintain(f.actor);await provider.maintain(f.actor);assert.equal(created.length,1);assert.equal(created[0]._stats.compendiumSource,source);assert.equal(created[0]._id,undefined);
}));
test('mixed weapon and spell fire is a single card with an explicit manual resistance notice',()=>setup({},async f=>{
 firearm(f);f.actor.system.actions[0].damage=async()=>{const r=roll(10);r.instances=[{type:'fire',total:10}];return r};
 f.spell.getDamage=async()=>{const r=roll(16);r.instances=[{type:'fire',total:16}];return {template:{damage:{roll:r}},context:{options:[]}}};
 await runWith(f,{volatileSparkEnabled:true})();const cards=f.messages.filter(m=>m.flags[NS]?.spellCombinationDamage);assert.equal(cards.length,1);
 assert.deepEqual(cards[0].flags[NS].volatileSparkSpellDamage.types,[]);assert.deepEqual(cards[0].flags[NS].volatileSparkSpellDamage.mixedTypes,['fire']);assert.match(cards[0].flavor,/GM.*调整栏/);
}));
for(const [save,value] of [['failure',1],['criticalFailure',2],['success',0],['criticalSuccess',0]])test(`Booming Blast ${save} applies frightened ${value} without replacing the native Strike`,()=>setup({outcomes:['failure'],save,charged:false},async f=>{
 firearm(f);f.actor.items.push({...f.feat,id:'spellstrike',uuid:'Actor.pc.Item.spellstrike'});
 Object.assign(f.feat,{type:'spell',sourceId:boomingSource,spellcasting:f.entry,rank:3});f.feat.system={traits:{value:['focus','magus','mental','emotion']},time:{value:'1'}};
 const condition=nativeFrightened(f);
 await runWith(f,{volatileSparkEnabled:true})();assert.equal(condition.value,value);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);
 const opts=f.calls.find(c=>c.kind==='attack').opts.options;assert(opts.has('action:booming-blast'));assert(!opts.has('action:spellstrike'));
 assert(!opts.has('item:trait:magical'),'Booming Blast does not infuse its firearm Strike');
}));
test('cancelled Booming Blast native Strike does not spend focus or recharge',()=>setup({charged:false},async f=>{
 firearm(f);f.actor.items.push({...f.feat,id:'spellstrike',uuid:'Actor.pc.Item.spellstrike'});
 Object.assign(f.feat,{type:'spell',sourceId:boomingSource,spellcasting:f.entry});f.feat.system={traits:{value:['focus','magus']},time:{value:'1'}};
 const result=await runWith(f,{volatileSparkEnabled:true,nativeOperations:{run:async()=>({status:'cancelled'}),register(){}}})();
 assert.equal(result.status,'cancelled');assert.equal(f.calls.filter(c=>c.kind==='bind-payment'||c.kind==='save').length,0);assert.equal(f.actor.flags[NS].spellstrike.charged,false);
}));
test('ranged Spellstrike requires its current ammunition provider before committing resources',()=>setup({},async f=>{
 firearm(f);f.game.modules.delete('pf2e-ranged-combat');await assert.rejects(runWith(f,{volatileSparkEnabled:true})(),/Ranged Combat/);assert.equal(f.calls.filter(c=>c.kind==='payment'||c.kind==='attack').length,0);
}));
function paidBooming(f){
 firearm(f);f.actor.items.push({...f.feat,id:'spellstrike',uuid:'Actor.pc.Item.spellstrike'});
 Object.assign(f.feat,{type:'spell',sourceId:boomingSource,spellcasting:f.entry,rank:3});f.feat.system={traits:{value:['focus','magus']},time:{value:'1'},location:{value:'entry'},cast:{focusPoints:1}};
 const input={actorUuid:f.actor.uuid,itemUuid:f.feat.uuid,sourceId:boomingSource,entryUuid:f.entry.uuid,rank:3,slotId:null,focusPoints:1,overlayIds:[]};
 f.actor.flags[NS].nativeCasts=[{...input,id:'native-cast',userId:'player',state:'paid'}];f.actor.system.resources.focus.value=2;
 f.message.flags[NS].nativeCast={id:'native-cast',actorUuid:f.actor.uuid,itemUuid:f.feat.uuid,userId:'player'};f.message.flags[NS].nativeCastInput=input;
 f.message.flags.pf2e.context={type:'spell-cast'};delete f.message.flags[NS].usageInput.actualUse;
 Object.assign(f.targetDocs[0].actor,{getCondition:()=>null,increaseCondition:async()=>{}});
 return createNativeCastEvents({game:f.game,fromUuid:async uuid=>[f.actor,...f.actor.items].find(d=>d.uuid===uuid)});
}
test('the real native cast coordinator accepts a paid actualCast card without an action actualUse marker',()=>setup({outcomes:['failure'],charged:false},async f=>{
 const casts=paidBooming(f),use=runWith(f,{volatileSparkEnabled:true,nativeCasts:casts});await use();await use();
 assert.equal(f.actor.system.resources.focus.value,2);assert.equal(f.actor.flags[NS].nativeCasts[0].state,'used');assert.equal(f.actor.flags[NS].nativeCasts[0].messageId,f.message.id);
 assert.equal(f.actor.flags[NS].spellstrike.charged,true);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1);
}));
test('a paid native Cast keeps its focus payment and conflux recharge when its subsequent Strike is cancelled',()=>setup({charged:false},async f=>{
 const casts=paidBooming(f),result=await runWith(f,{volatileSparkEnabled:true,nativeCasts:casts,nativeOperations:{run:async()=>({status:'cancelled'}),register(){}}})();
 assert.equal(result.status,'cancelled');assert.match(result.result,/已支付|已经支付/);assert.equal(f.actor.system.resources.focus.value,2);assert.equal(f.actor.flags[NS].nativeCasts[0].state,'used');assert.equal(f.actor.flags[NS].spellstrike.charged,true);
}));
test('a paid native Cast cancelling at the firearm choice remains a single recorded conflux',()=>setup({charged:false},async f=>{
 const casts=paidBooming(f);f.actor.system.actions.push({...f.actor.system.actions[0],item:{...f.weapons[0],id:'otherGun'}});
 const use=runWith(f,{volatileSparkEnabled:true,nativeCasts:casts,choose:async()=>null});const result=await use();await assert.rejects(use(),/重试|未完成/);
 assert.equal(result.status,'cancelled');assert.equal(f.actor.system.resources.focus.value,2);assert.equal(f.actor.flags[NS].spellstrike.charged,true);assert.equal(f.calls.filter(c=>c.kind==='attack').length,0);
}));
test('Booming Blast preserves native immunity to its mental or emotion spell',()=>setup({outcomes:['failure'],save:'criticalFailure'},async f=>{
 const casts=paidBooming(f);let conditions=0;
 f.targetDocs[0].actor.isImmuneTo=item=>item===f.feat;f.targetDocs[0].actor.increaseCondition=async()=>{conditions++};
 await runWith(f,{volatileSparkEnabled:true,nativeCasts:casts})();assert.equal(conditions,0);
}));

function nativeFrightened(f,{current=0,bravery=false}={}){
 const source=readFileSync(process.env.PF2E_NATIVE_861_BUNDLE,'utf8');
 const section=(start,end)=>{const begin=source.indexOf(start),finish=source.indexOf(end,begin+start.length);assert(begin>=0&&finish>begin);return source.slice(begin,finish);};
 const increase=section('\n\tasync increaseCondition(','\n\tasync toggleCondition('),update=section('\n\tstatic async updateConditionValue(','\n}, zl =').replace('static ','');
 const actor=f.targetDocs[0].actor,items=new Map(),hooks=new Map(),wrappers=new Map();let changes=0;
 Object.assign(actor,{items,prototypeToken:{},isOfType:type=>type==='creature',getCondition:slug=>[...items.values()].find(item=>item.slug===slug)??null});
 Object.defineProperty(actor,'itemTypes',{get:()=>({condition:[...items.values()].filter(item=>item.type==='condition')})});
 const document=value=>({id:'frightened',type:'condition',actor,flags:{},_source:{system:{slug:'frightened',value:{value}}},get system(){return this._source.system},get slug(){return this.system.slug},get value(){return this.system.value.value},active:true,isLocked:false,isOfType:type=>type==='condition',updateSource(data){for(const[key,value]of Object.entries(data))set(this._source,key,value)},async update(data){changes++;hooks.get('preUpdateItem')?.(this,data,{},f.game.user.id);this.updateSource(data);return this},async delete(){items.delete(this.id)}});
 if(current)items.set('frightened',document(current));
 actor.createEmbeddedDocuments=async(_type,data)=>{
  changes++;const documents=data.map(data=>document(data.system.value.value));
  await wrappers.get('CONFIG.Item.documentClass._preCreateOperation')?.(async()=>{},documents,{},f.game.user);
  for(const item of documents)items.set(item.id,item);return documents;
 };
 const game={pf2e:{ConditionManager:{getCondition:()=>({toObject:()=>({type:'condition',system:{slug:'frightened',value:{value:1}}})}),...Function('tupleHasValue',`return {${update}}`)((list,item)=>list.includes(item))}}};
 Object.assign(actor,Function('game','Math',`return {${increase}}`)(game,{clamp:(value,min,max)=>Math.min(Math.max(value,min),max)}));
 if(bravery){
  actor.type='character';items.set('bravery',{type:'feat',sourceId:'Compendium.pf2e.classfeatures.Item.GJKJafDGuX4BeAeN'});
  f.game.world={id:'bob-2'};f.game.system={id:'pf2e'};
  createBravery({game:f.game}).register({Hooks:{on:(name,callback)=>hooks.set(name,callback)},libWrapper:{register:(_module,key,callback)=>wrappers.set(key,callback)}});
 }
 return {get value(){return actor.getCondition('frightened')?.value??0},get changes(){return changes}};
}
for(const [current,save,expected] of [[1,'criticalFailure',2],[3,'criticalFailure',3],[0,'failure',1],[0,'criticalFailure',2]])test(`Booming Blast uses native condition semantics: frightened ${current}, ${save} -> ${expected}`,()=>setup({outcomes:['failure'],save},async f=>{
 const casts=paidBooming(f),condition=nativeFrightened(f,{current}),use=runWith(f,{volatileSparkEnabled:true,nativeCasts:casts});
 await use();await use();assert.equal(condition.value,expected);assert.equal(condition.changes,current===3?0:1);
}));
for(const [current,save,expected] of [[0,'failure',0],[0,'criticalFailure',1],[1,'criticalFailure',1],[3,'criticalFailure',3]])test(`Booming Blast lets Bravery reduce native frightened once: ${current}, ${save} -> ${expected}`,()=>setup({outcomes:['failure'],save},async f=>{
 const casts=paidBooming(f),condition=nativeFrightened(f,{current,bravery:true});
 const use=runWith(f,{volatileSparkEnabled:true,nativeCasts:casts,runTargetSave:async()=>({status:'rolled',check:await f.Message.create({flags:{pf2e:{context:{type:'saving-throw',outcome:save}}}})})});
 await use();await use();assert.equal(condition.value,expected);assert.equal(condition.changes,current===3?0:1);
}));
