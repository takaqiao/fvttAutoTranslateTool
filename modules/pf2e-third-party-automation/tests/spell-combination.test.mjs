import test from 'node:test';
import assert from 'node:assert/strict';
import {createDualStrikeAutomation} from '../scripts/dual-strike-automation.mjs';
let api={};try{api=await import('../scripts/spell-combination.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const NS='pf2e-third-party-automation',S={strike:'Compendium.pf2e.actionspf2e.Item.QDW9H8XLIjuW2fE4',swipe:'Compendium.pf2e.feats-srd.Item.Fs88vjez9px2mmrC',combination:'Compendium.pf2e.actionspf2e.Item.zUWj4zmBNWOTzeFJ'};
const set=(o,p,v)=>{const bits=p.split('.');let at=o;for(const k of bits.slice(0,-1))at=at[k]??={};at[bits.at(-1)]=v;};
function roll(total=10){return {total,instances:[],options:{},toJSON(){return {total:this.total,instances:this.instances,options:structuredClone(this.options)}},async evaluate(){return this},alter(multiplier,addend){assert.equal(addend,0);return roll(total*multiplier)},async toMessage(data){return {...data,rolls:[this]}}};}
function fixture({kind='strike',map=0,outcomes=['success','success'],save='failure',spellSave=false,basic=true,attack=!spellSave,targets=kind==='swipe'?2:1,multi=false,charged=true,overlays=false,consumeFails=false}={}){
 const calls=[],messages=[],merges=[];
 const actor={id:'pc',uuid:'Actor.pc',type:'character',flags:{[NS]:{spellstrike:{charged}}},level:15,testUserPermission:()=>true,items:[],system:{resources:{focus:{value:3}}},async update(changes){for(const[k,v]of Object.entries(changes))set(this,k,v)}};
 const feat={id:'feature',uuid:'Actor.pc.Item.feature',sourceId:S[kind],type:kind==='swipe'?'feat':'action',actor,system:{rules:[]},name:kind};
 const scene={id:'s',tokens:new Map()};
 const targetDocs=Array.from({length:targets},(_,i)=>({id:`target${i}`,uuid:`Scene.s.Token.target${i}`,documentName:'Token',parent:scene,actor:{uuid:`Actor.target${i}`,type:'npc'},object:{distanceTo:()=>5}}));
 for(const target of targetDocs){target.object.document=target;target.actor.getStatistic=()=>({check:{roll:async opts=>{calls.push({kind:'save',opts,target});const raw=new Message({speaker:{actor:target.actor.uuid},flags:{pf2e:{context:{type:'saving-throw',outcome:save,options:opts.extraRollOptions}}},rolls:[roll(20)]});await opts.callback?.(roll(20),save,raw);return roll(20);}}});}
 const origin={id:'pcToken',uuid:'Scene.s.Token.pcToken',parent:scene,actor,object:{distanceTo:()=>5,checkCollision:()=>false}};actor.getActiveTokens=()=>[origin];actor.getReach=()=>5;
 for(const token of [origin,...targetDocs])scene.tokens.set(token.id,token);
 const weapons=[{id:'weapon',slug:'sword',system:{equipped:{carryType:'held',handsHeld:1},traits:{value:[]}},isUnarmed:false},{id:'fist',slug:'fist',system:{equipped:{carryType:'worn',handsHeld:0},traits:{value:['unarmed']}},isUnarmed:true}].map(w=>({...w,uuid:`Actor.pc.Item.${w.id}`,name:w.id,actor,type:'weapon',category:w.isUnarmed?'unarmed':'martial',isMelee:true,hands:'1',isEquipped:true,get isHeld(){return this.system.equipped.carryType==='held'},get handsHeld(){return this.system.equipped.handsHeld}}));
 actor.system.actions=weapons.map((item,index)=>({type:'strike',ready:true,item,variants:[0,1,2].map(tier=>({roll:async opts=>{const i=calls.filter(c=>c.kind==='attack').length;calls.push({kind:'attack',index,tier,opts});const raw=new Message({speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:item.uuid,type:'weapon',actor:actor.uuid},context:{type:'attack-roll',outcome:outcomes[i],options:[...(opts.options??[])],target:{actor:opts.target.document.actor.uuid,token:opts.target.document.uuid}}}},rolls:[roll(25)]});await opts.callback(roll(25),outcomes[i],raw);return roll(25)}})),damage:async opts=>{calls.push({kind:'weapon-damage',index,opts});return roll(10)},critical:async opts=>{calls.push({kind:'weapon-damage',index,opts});return roll(20)}}));
 const spell={id:'spell',uuid:'Actor.pc.Item.spell',sourceId:'Compendium.pf2e.spells-srd.Item.TestSpell0000000',type:'spell',name:'test spell',actor,isAttack:attack,isCantrip:true,isFocusSpell:false,rank:8,system:{time:{value:'2'},traits:{value:attack?['attack']:[]},location:{value:'entry'},cast:{focusPoints:0},damage:{damage:{formula:'2d6'}},defense:spellSave?{save:{statistic:'reflex',basic}}:null,target:{value:multi?'2 creatures':'1 creature'},area:null},overlays:overlays?new Map([['variant1',{overlayType:'override',name:'Variant 1'}],['variant2',{overlayType:'override',name:'Variant 2'}]]):new Map(),appliedOverlays:new Map(),async getDamage(opts){calls.push({kind:'spell-damage',opts,overlays:[...this.appliedOverlays.values()]});return {template:{damage:{roll:roll(16)}},context:{options:new Set(),domains:['damage','spell-damage'],traits:['arcane']}}},loadVariant({overlayIds=[],castRank=this.rank}={}){return {...this,rank:castRank,isVariant:true,appliedOverlays:new Map(overlayIds.map(id=>['override',id]))}},async toMessage(event,{create,data}={}){assert.equal(create,false);calls.push({kind:'spell-card'});return new Message({speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:this.uuid,type:'spell',actor:actor.uuid,castRank:data?.castRank??this.rank}}},content:'spell card'})}};
 const entry={id:'entry',uuid:'Actor.pc.Item.entry',type:'spellcastingEntry',actor,isPrepared:true,isFlexible:false,statistic:{getChatData:()=>({dc:{value:35}})},async getSheetData(){return {groups:[{id:'cantrips',maxRank:8,active:[{spell,castRank:8,expended:false}]}]}}};spell.spellcasting=entry;
 actor.items=[feat,...weapons,spell,entry];actor.items.get=id=>actor.items.find(i=>i.id===id);actor.spellcasting={contents:[entry],collections:[]};
 actor._source={items:[]};actor.clone=changes=>{calls.push({kind:'infusion-clone',changes});return actor};
 class Message{constructor(data){Object.assign(this,data);this.sourceRolls=this.rolls?.map(r=>({...r,options:structuredClone(r.options??{})}));this.isCheckRoll=!!this.rolls?.length&&this.flags?.pf2e?.context?.type!=='damage-roll';}toObject(){return {...this,rolls:this.sourceRolls,flags:structuredClone(this.flags)}}async update(changes){for(const[k,v]of Object.entries(changes))set(this,k,v);return this}static async create(data){const msg=new Message(data);msg.id=`m${messages.length}`;messages.push(msg);game.messages.set(msg.id,msg);return msg}}
 const game={user:{id:'gm',settings:{}},users:{activeGM:{id:'gm'}},messages:new Map(),modules:new Map([['pf2e-toolbelt',{active:true}]]),time:{worldTime:100},toolbelt:{api:{betterChat:{mergeDamageMessages:async(a,b,opts)=>{merges.push({a,b,opts});return new Message({flags:{pf2e:{context:{options:[]}}},rolls:[roll(a.rolls[0].total+b.rolls[0].total)]})}}}}};
 game.scenes=new Map([[scene.id,scene]]);
 const casts={addMatcher(){},captureUsage(){},register(){return()=>{}},async payForActivity(ctx){calls.push({kind:'payment',ctx});if(consumeFails)throw Error('法术资源不足');return {id:'paid'}},async ensurePaid(ctx){calls.push({kind:'bind-payment',ctx});return{id:'paid'}},async finishActivityWithoutSpell(ctx){calls.push({kind:'finish-payment',ctx})}};
 const chooser=async({title,choices})=>title.includes('多重')?String(map):choices[0].value;
 assert.equal(typeof api.createSpellCombination,'function');const provider=api.createSpellCombination({game,choose:chooser,fromUuid:async uuid=>[actor,origin,...targetDocs,...actor.items].find(d=>d.uuid===uuid),nativeCasts:casts});
 const message=new Message({id:'use1',author:{id:'player'},speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:feat.uuid,type:feat.type,actor:actor.uuid}},[NS]:{usageInput:{actualUse:true,targetUuids:targetDocs.map(t=>t.uuid)}}}});game.messages.set(message.id,message);
 const use=()=>provider.executeUsage({actor,item:feat,message,user:{id:'player'},action:provider.resolveAction(feat)});
 return {provider,actor,feat,spell,entry,origin,targetDocs,calls,messages,merges,game,casts,use,message,Message,weapons};
}
async function setup(options,fn){const previous=globalThis.CONFIG;try{const f=fixture(options);globalThis.CONFIG={ChatMessage:{documentClass:f.Message},Dice:{rolls:[]},PF2E:{}};await fn(f)}finally{globalThis.CONFIG=previous}}
function runWith(f,options={}){const p=api.createSpellCombination({game:f.game,choose:async({choices})=>choices[0].value,fromUuid:async uuid=>[f.actor,f.origin,...f.targetDocs,...f.actor.items].find(d=>d.uuid===uuid),nativeCasts:f.casts,...options});return()=>p.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:{id:'player'},action:p.resolveAction(f.feat)});}
const handoff=f=>{f.game.users.activeGM={id:'new-gm'}};
function dualUse(f){
 f.feat.type='feat';f.feat.sourceId='Compendium.pf2e.feats-srd.Item.onde0SxLoxLBTnvm';
 f.feat.system.rules=[{key:'FlatModifier',predicate:[{or:['double-slice-second',NS+':double-slice-second']}]}];
 f.weapons[1].system.equipped={carryType:'held',handsHeld:1};
 const provider=createDualStrikeAutomation({game:f.game,fromUuid:async uuid=>[f.origin,...f.targetDocs].find(d=>d.uuid===uuid),choose:async({choices})=>choices[0].value});
 return ()=>provider.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:{id:'player'},action:'dual:double-slice'});
}
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
 const provider=createDualStrikeAutomation({game:f.game,fromUuid:async uuid=>[f.origin,...f.targetDocs].find(d=>d.uuid===uuid),choose:async({choices})=>choices[0].value});
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
 const use=provider.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:{id:'player'},action:provider.resolveAction(f.feat)});await entered;
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
test('GM handoff after an already-paid spell keeps payment and blocks subsequent receipt and discharge writes',()=>setup({},async f=>{
 const pay=f.casts.payForActivity;f.casts.payForActivity=async ctx=>{const result=await pay(ctx);handoff(f);return result};
 await assert.rejects(f.use,/GM/);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='attack').length,0);assert.equal(f.message.flags[NS].spellCombinationUse,undefined);assert.equal(f.actor.flags[NS].spellstrike.charged,true);
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
test('Spellstrike with save spell still rolls save and damage when the Strike misses',()=>setup({spellSave:true,outcomes:['failure']},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='save').length,1);assert.equal(f.messages.at(-1).rolls[0].total,16);assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,0)}));
test('Spellstrike critical failure spends the spell but disrupts saves and spell damage',()=>setup({spellSave:true,outcomes:['criticalFailure']},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='save'||c.kind==='spell-damage').length,0);assert.equal(f.actor.flags[NS].spellstrike.charged,false)}));
test('a successful basic save halves only the spell component before merging weapon damage',()=>setup({spellSave:true,save:'success'},async f=>{await f.use();assert.equal(f.messages.at(-1).rolls[0].total,18)}));
test('save failure on a critical Strike does not double the save spell',()=>setup({spellSave:true,outcomes:['criticalSuccess']},async f=>{await f.use();assert.equal(f.messages.at(-1).rolls[0].total,36)}));
test('Spell Swipe rolls the same weapon against adjacent targets at the same MAP and pays once',()=>setup({kind:'swipe',map:1,multi:true},async f=>{await f.use();assert.deepEqual(f.calls.filter(c=>c.kind==='attack').map(c=>[c.index,c.tier]),[[0,1],[0,1]]);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='spell-damage').length,2);assert.equal(f.merges.length,2)}));
test('single-target Spell Swipe affects only the selected target with the spell',()=>setup({kind:'swipe'},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='spell-damage').length,1);assert.equal(f.calls.filter(c=>c.kind==='weapon-damage').length,2);assert.equal(f.merges.length,1)}));
test('discharged Spellstrike rejects use before payment or attack',()=>setup({charged:false},async f=>{await assert.rejects(f.use,/充能/);assert.equal(f.calls.length,0)}));
test('unavailable spell resources cause no attack or discharge',()=>setup({consumeFails:true},async f=>{await assert.rejects(f.use,/资源/);assert.equal(f.calls.filter(c=>c.kind==='attack').length,0);assert.equal(f.actor.flags[NS].spellstrike.charged,true)}));
test('replaying the original activity cannot spend or attack twice',()=>setup({},async f=>{await f.use();await f.use();assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1)}));
test('a partial native attack failure cannot replay an already paid activity',()=>setup({},async f=>{f.actor.system.actions[0].variants[0].roll=async()=>{f.calls.push({kind:'attack'});throw Error('native failure')};await assert.rejects(f.use,/native failure/);await assert.rejects(f.use,/未完成|重试|确认/);assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='attack').length,1)}));
test('Spell Swipe leaves adjacency to the table and keeps both original targets with one payment',()=>setup({kind:'swipe'},async f=>{f.targetDocs[0].object.distanceTo=()=>{throw Error('adjacency probe must not run')};await f.use();assert.equal(f.calls.filter(c=>c.kind==='payment').length,1);assert.deepEqual(f.calls.filter(c=>c.kind==='attack').map(c=>c.opts.target.document.uuid),f.targetDocs.map(t=>t.uuid))}));
test('Spell Swipe cannot substitute a token from another scene',()=>setup({kind:'swipe'},async f=>{f.targetDocs[1].parent={id:'other'};await assert.rejects(f.use,/场景|目标/);assert.equal(f.calls.filter(c=>c.kind==='payment'||c.kind==='attack').length,0)}));
test('normal player variant selection is carried into payment and damage',()=>setup({overlays:true},async f=>{await f.use();assert.deepEqual(f.calls.find(c=>c.kind==='payment').ctx.item.appliedOverlays,new Map([['override','variant1']]));assert.deepEqual(f.calls.find(c=>c.kind==='spell-damage').overlays,['variant1'])}));
for(const kind of ['strike','combination','swipe'])test(`${kind} settles native attacks without measuring range or collision`,()=>setup({kind},async f=>{const forbidden=()=>{throw Error('spatial probe must not run')};f.actor.getReach=forbidden;f.origin.object.distanceTo=forbidden;f.origin.object.checkCollision=forbidden;await f.use();assert.equal(f.calls.filter(c=>c.kind==='attack').length,kind==='strike'?1:2);assert.equal(f.message.flags[NS].spellCombinationUse.state,'done')}));
test('native generated attack cards suppress duplicate Workbench auto damage',()=>setup({},async f=>{await f.use();const attacks=f.messages.filter(m=>m.flags[NS]?.spellCombinationAttack);assert.equal(attacks.length,1);assert.equal(attacks[0].flags['xdy-pf2e-workbench'].noAutoDamageRoll,true)}));
test('unprepared and expended spells are never choices even if in the spellbook',()=>setup({},async f=>{f.entry.getSheetData=async()=>({groups:[{id:8,active:[{spell:f.spell,castRank:8,expended:true}]}]});await assert.rejects(f.use,/法术/);assert.equal(f.calls.length,0)}));
test('Spell Swipe enables the native sweep-bonus predicate against both targets',()=>setup({kind:'swipe'},async f=>{f.weapons[0].system.traits.value=['sweep'];await f.use();assert(f.calls.filter(c=>c.kind==='attack').every(c=>c.opts.options.has('sweep-bonus')))}));
test('Overwhelming Combination allows fist first and waits for first-hit providers before the next strike',async()=>{
 const prior=globalThis.CONFIG;try{const f=fixture({kind:'combination'});globalThis.CONFIG={ChatMessage:{documentClass:f.Message}};let after=0;
 const provider=api.createSpellCombination({game:f.game,nativeCasts:f.casts,fromUuid:async id=>f.targetDocs.find(t=>t.uuid===id),choose:async({title,choices})=>title.includes('攻击顺序')?'fist':title.includes('多重')?'0':choices[0].value,afterAttack:async()=>{after++;f.calls.push({kind:'after-attack'})}});
 await provider.executeUsage({actor:f.actor,item:f.feat,message:f.message,user:{id:'player'},action:provider.resolveAction(f.feat)});
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
test('fully disrupted Spellstrike closes its paid receipt without publishing a second payable spell card',()=>setup({spellSave:true,outcomes:['criticalFailure']},async f=>{await f.use();assert.equal(f.calls.filter(c=>c.kind==='finish-payment').length,1);assert.equal(f.calls.filter(c=>c.kind==='spell-card').length,0)}));
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
