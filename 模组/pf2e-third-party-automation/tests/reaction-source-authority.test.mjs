import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const {SOURCES,MODULE_ID:ID}=await import('../scripts/rules.mjs');
const {getCycleTrigger,resolveCycleDamageMessage}=await import('../scripts/cycle-automation.mjs');
const {createCycleCoordinator}=await import('../scripts/cycle-coordinator.mjs');
const {createReactionChecks,REACTION_CHECK_SOURCES}=await import('../scripts/reaction-checks.mjs');
const {createRoaringEffects}=await import('../scripts/roaring-effects.mjs');
const {createRoaringSource}=await import('../scripts/roaring-lifecycle.mjs');
const {createCycleAutomation}=await import('../scripts/cycle-automation.mjs');
assert.ok(process.env.PF2E_NATIVE_BUNDLE,'PF2E_NATIVE_BUNDLE must identify the pinned primary source');
const primary=await readFile(process.env.PF2E_NATIVE_BUNDLE,'utf8');assert.equal(createHash('sha256').update(primary).digest('hex'),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');
const definition=primary.match(/function suppressFeats\(e\) \{[\s\S]*?\n\}/)?.[0];assert.ok(definition);const suppressFeats=Function(definition+';return suppressFeats')();
function fixture(){
 const gm={id:'gm',isGM:true,active:true},next={id:'next',isGM:true,active:true},owner={id:'owner',active:true},users=new Map([[gm.id,gm],[next.id,next],[owner.id,owner]]);users.activeGM=gm;
 const game={user:gm,users,actors:new Map(),scenes:new Map(),messages:new Map(),modules:new Map(),pf2e:{},settings:{get:()=>false},time:{worldTime:100},combat:null},writes=[];
 let owned=true;const actor={id:'a',uuid:'Actor.a',type:'character',level:5,canAct:true,isDead:false,flags:{pf2e:{cultivator:{energy:'fire'}}},items:new Map(),hasCondition:()=>false,testUserPermission:u=>u===gm||owned&&u===owner};game.actors.set(actor.id,actor);
 const feat=(id,sourceId)=>{const item={id,uuid:actor.uuid+'.Item.'+id,actor,parent:actor,type:'feat',sourceId,suppressed:false,flags:{pf2e:{itemGrants:{}}},system:{frequency:{value:1,max:1}}};actor.items.set(id,item);return item;};
 const cycle=feat('cycle',SOURCES.cycle),attunement=feat('attunement',SOURCES.attunement),clock=feat('clock',REACTION_CHECK_SOURCES.clock);
 actor.update=async changes=>{writes.push({kind:'actor',activeGM:game.users.activeGM.id,owned});actor.flags[ID]??={};for(const [key,value]of Object.entries(changes)){if(key===`flags.${ID}.cyclePending`)actor.flags[ID].cyclePending=structuredClone(value);else if(key===`flags.${ID}.reactionChecks.reactions`){actor.flags[ID].reactionChecks??={};actor.flags[ID].reactionChecks.reactions=structuredClone(value);}}return actor;};
 clock.update=async changes=>{writes.push({kind:'clock',activeGM:game.users.activeGM.id,owned,current:actor.items.get(clock.id)===clock});clock.system.frequency.value=changes['system.frequency.value'];return clock;};
 clock.toMessage=async()=>{const card={id:'clock-card',uuid:'ChatMessage.clock-card',item:clock,flags:{pf2e:{origin:{uuid:clock.uuid}}},async update(changes){writes.push({kind:'card'});for(const [path,value]of Object.entries(changes)){let at=this;const keys=path.split('.');for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=structuredClone(value);}return this;}};game.messages.set(card.id,card);return card;};
 const damage={total:8,instances:[]},item={traits:new Set(['fire'])},message={id:'damage',uuid:'ChatMessage.damage',timestamp:Date.now(),isDamageRoll:true,item,rolls:[damage],flags:{pf2e:{context:{target:{actor:actor.uuid},options:[]}}}};game.messages.set(message.id,message);
 const context={actorUuid:actor.uuid,tokenUuid:null,messageId:message.id,rollIndex:0,trait:'fire',damageType:'fire',level:5};
 const combatant={id:'turn-a',actor,initiative:12},combat={id:'combat',started:true,round:1,turn:0,turns:[combatant],combatants:[combatant]};
 return {game,gm,next,owner,actor,clock,cycle,attunement,damage,item,message,context,combat,writes,setOwned:value=>owned=value};
}
test('active exact Cycle and Attunement admit the matching damage trait',()=>{const f=fixture();assert.deepEqual(getCycleTrigger(f.actor,{damage:f.damage,item:f.item}),{trait:'fire',damageType:'fire',level:5});});
for(const key of ['cycle','attunement'])test(`native suppressed ${key} cannot admit Cycle resistance`,()=>{const f=fixture();suppressFeats([f[key]]);const trigger=getCycleTrigger(f.actor,{damage:f.damage,item:f.item});assert.equal(trigger,null);});
function pending(f){f.game.combat=f.combat;f.actor.flags[ID]={cyclePending:{...f.context,status:'armed',reactionId:'reaction-original',nonce:'0123456789abcdef0123456789abcdef',timing:{combatId:'combat',combatantId:'turn-a',endRound:2}}};}
test('unchanged Cycle claim consumes its one armed nonce and a repeated claim cannot claim again',async()=>{const f=fixture();pending(f);const c=createCycleCoordinator({game:f.game,fromUuid:async()=>f.actor});assert.equal((await c.claim(f.context,f.owner)).status,'claimed');assert.equal(await c.claim(f.context,f.owner),null);assert.equal(f.writes.length,1);});
for(const [label,change]of [['GM',f=>f.game.users.activeGM=f.next],['OWNER',f=>f.setOwned(false)],['Actor',f=>f.game.actors.set(f.actor.id,{...f.actor})]])test(`Cycle actor lookup await loses current ${label}: must not claim or write`,async()=>{
 const f=fixture();pending(f);const c=createCycleCoordinator({game:f.game,fromUuid:async()=>{change(f);return f.actor;}});let error,result;try{result=await c.claim(f.context,f.owner);}catch(e){error=e.message;}assert.equal(f.writes.length,0);assert.equal(result,undefined);
});
async function decide({suppressed=false,change}={}){
 const f=fixture();if(suppressed)suppressFeats([f.clock]);const api=createReactionChecks({game:f.game,fromUuid:async uuid=>uuid===f.actor.uuid?f.actor:null,choose:async()=>{change?.(f);return 'clock';}});
 let result,error;try{result=await api.decideCheckReaction({nonce:'reaction-once',actorUuid:f.actor.uuid,type:'saving-throw',degree:1,isReroll:false,fortune:false},f.owner);}catch(e){error=e.message;}return {f,result,error};
}
test('active Clock consumes one original use and durable claim',async()=>{const r=await decide();assert.equal(r.result,'clock');assert.equal(r.f.clock.system.frequency.value,0);assert.equal(r.f.writes.filter(w=>w.kind==='clock').length,1);});
test('native suppressed Clock must not consume its native daily use',async()=>{const r=await decide({suppressed:true});assert.equal(r.f.clock.system.frequency.value,1);assert.equal(r.result,null);});
for(const [label,change]of [['OWNER',f=>f.setOwned(false)],['current Item',f=>f.actor.items.set(f.clock.id,{...f.clock})]])test(`Clock owner choice await loses ${label}: must not pay or publish`,async()=>{const r=await decide({change});assert.equal(r.f.clock.system.frequency.value,1);assert.equal(r.f.writes.length,0);});
test('Cycle token lookup await loses OWNER: must not claim or write',async()=>{
 const f=fixture();f.context.tokenUuid='Scene.scene.Token.a';f.message.flags.pf2e.context.target.token=f.context.tokenUuid;pending(f);const scene={id:'scene',tokens:new Map()},token={id:'a',uuid:f.context.tokenUuid,documentName:'Token',parent:scene,actor:f.actor,actorLink:true};scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);
 const c=createCycleCoordinator({game:f.game,fromUuid:async uuid=>{if(uuid===f.actor.uuid)return f.actor;f.setOwned(false);return token;}});let result,error;try{result=await c.claim(f.context,f.owner);}catch(e){error=e.message;}
 assert.equal(f.writes.length,0);assert.equal(result,undefined);
});
test('Roaring pre-persist resolver await replaces current Actor: must not write the stale Actor',async()=>{
 const f=fixture();f.actor.id='target';f.actor.uuid='Actor.target';f.game.actors=new Map([['target',f.actor]]);const writes=[];f.actor.flags={};
 f.actor.update=async changes=>{writes.push({current:f.game.actors.get(f.actor.id)===f.actor});for(const [path,value]of Object.entries(changes)){const keys=path.split('.');let object=f.actor;for(const key of keys.slice(0,-1))object=object[key]??={};object[keys.at(-1)]=structuredClone(value);}return f.actor;};
 const turn={combatId:'combat',combatantId:'caster',actorUuid:'Actor.caster',tokenUuid:'Scene.scene.Token.caster',started:true,round:4,turn:0,lastTurnEnd:3,order:[{id:'caster',initiative:20,overridePriority:null}]};
 const state=createRoaringSource({sourceNonce:'source-one',castNonce:'cast-one',sourceId:'Compendium.pf2e.spells-srd.Item.czO0wbT1i320gcu9',itemUuid:'Actor.caster.Item.spell',entryUuid:'Actor.caster.Item.entry',rank:3,casterActorUuid:'Actor.caster',casterTokenUuid:'Scene.scene.Token.caster',targetActorUuid:f.actor.uuid,targetTokenUuid:'Scene.scene.Token.target',originalMessageUuid:'ChatMessage.original',completedWorldTime:10,turn,finiteEnvelope:{start:{value:10,initiative:20},duration:{value:1,unit:'rounds',expiry:'turn-end',sustained:false}}});
 const context={userId:'owner',gmId:'gm',dc:21,paymentId:'cast-one',immunity:{checked:true,spell:false,slowed:false,fascinated:false,systemVersion:'8.5.1'}};
 let reads=0;const effects=createRoaringEffects({game:f.game,fromUuid:async()=>{const resolved=f.game.actors.get('target');if(++reads===2)f.game.actors.set('target',{...f.actor,flags:{}});return resolved;}});let result,error;try{result=await effects.claim({actor:f.actor,state,context});}catch(e){error=e.message;}
 assert.equal(writes.length,0);
});
for(const alias of ['suppressed','isSuppressed','system.suppressed'])for(const key of ['cycle','attunement'])test(`Cycle excludes ${key} with ${alias} while retaining another active exact copy`,()=>{
 const f=fixture();if(alias==='system.suppressed')f[key].system.suppressed=true;else f[key][alias]=true;
 assert.equal(getCycleTrigger(f.actor,{damage:f.damage,item:f.item}),null);
 const copy={...f[key],id:key+'-active',uuid:f.actor.uuid+'.Item.'+key+'-active',suppressed:false,isSuppressed:false,system:{...f[key].system,suppressed:false}};f.actor.items.set(copy.id,copy);
 assert.deepEqual(getCycleTrigger(f.actor,{damage:f.damage,item:f.item}),{trait:'fire',damageType:'fire',level:5});
});
test('Cycle rejects a different current source despite matching legacy provenance',()=>{const f=fixture();f.cycle.sourceId='Other.source';f.cycle.flags.core={sourceId:SOURCES.cycle};assert.equal(getCycleTrigger(f.actor,{damage:f.damage,item:f.item}),null);});
test('suppressed Cycle follows the ordinary native damage path once without a claim or temporary resistance',async()=>{
 const f=fixture();suppressFeats([f.cycle]);let claims=0,native=0;const automation=createCycleAutomation({onClaim:async()=>{claims++;assert.fail('suppressed source cannot claim')}});automation.recordDamageMessage(f.message);
 assert.equal(await automation.applyDamage(f.actor,async params=>{native++;assert.ok(params.rollOptions.has(ID+':source:damage:0'));return 'native-result'}, {damage:f.damage,item:f.item}),'native-result');assert.equal(claims,0);assert.equal(native,1);
});
for(const alias of ['suppressed','isSuppressed','system.suppressed'])test(`Clock ignores ${alias} and can select another current active copy`,async()=>{
 const f=fixture();if(alias==='system.suppressed')f.clock.system.suppressed=true;else f.clock[alias]=true;let prompts=0;const api=createReactionChecks({game:f.game,fromUuid:async()=>f.actor,choose:async()=>{prompts++;return 'clock'}}),payload={nonce:'active-copy-clock',actorUuid:f.actor.uuid,type:'saving-throw',degree:1};
 assert.equal(await api.decideCheckReaction(payload,f.owner),null);assert.equal(prompts,0);assert.equal(f.clock.system.frequency.value,1);
 const active={...f.clock,id:'clock-active',uuid:f.actor.uuid+'.Item.clock-active',suppressed:false,isSuppressed:false,system:{frequency:{value:1,max:1},suppressed:false}};active.update=async changes=>{f.writes.push({kind:'active-clock'});active.system.frequency.value=changes['system.frequency.value'];return active};active.toMessage=async()=>{const card={id:'active-card',uuid:'ChatMessage.active-card',async update(changes){for(const [path,value]of Object.entries(changes)){let at=this;const keys=path.split('.');for(const k of keys.slice(0,-1))at=at[k]??={};at[keys.at(-1)]=structuredClone(value);}return this;}};f.game.messages.set(card.id,card);return card;};f.actor.items.set(active.id,active);
 assert.equal(await api.decideCheckReaction(payload,f.owner),'clock');assert.equal(active.system.frequency.value,0);assert.equal(f.clock.system.frequency.value,1);
});
const authorityChanges=[['GM object',f=>{f.game.users.activeGM={...f.gm};f.game.users.set(f.gm.id,f.game.users.activeGM)}],['OWNER user object',f=>f.game.users.set(f.owner.id,{...f.owner})],['Actor',f=>f.game.actors.set(f.actor.id,{...f.actor})],['OWNER permission',f=>f.setOwned(false)],['native source suppression',f=>suppressFeats([f.clock])],['current Item',f=>f.actor.items.set(f.clock.id,{...f.clock})]];
for(const [name,change]of authorityChanges)for(const phase of ['claim','payment'])test(`Clock loses ${name} after ${phase}: retained claim cannot pay or publish again`,async()=>{
 const f=fixture(),originalActorUpdate=f.actor.update,originalClockUpdate=f.clock.update;
 const target=phase==='claim'?f.actor:f.clock,original=target.update;target.update=async(...args)=>{const result=await original(...args);change(f);return result};
 const api=createReactionChecks({game:f.game,fromUuid:async()=>f.actor,choose:async()=> 'clock'}),payload={nonce:'clock-uncertain-once',actorUuid:f.actor.uuid,type:'saving-throw',degree:1};await assert.rejects(api.decideCheckReaction(payload,f.owner));
 assert.equal(f.actor.flags[ID].reactionChecks.reactions[0].state,'claimed');assert.equal(f.clock.system.frequency.value,phase==='claim'?1:0);assert.equal(f.writes.filter(w=>w.kind==='card').length,0);
 f.game.users.set(f.gm.id,f.gm);f.game.users.activeGM=f.gm;f.game.users.set(f.owner.id,f.owner);f.game.actors.set(f.actor.id,f.actor);f.setOwned(true);f.clock.suppressed=false;f.actor.items.set(f.clock.id,f.clock);f.actor.update=originalActorUpdate;f.clock.update=originalClockUpdate;
 assert.equal(await api.decideCheckReaction(payload,f.owner),null);assert.equal(f.clock.system.frequency.value,phase==='claim'?1:0);
});
for(const [name,change]of [['source card',f=>f.game.messages.set(f.message.id,{...f.message})],['source roll',f=>f.message.rolls=[{...f.damage}]],['pending nonce',f=>f.actor.flags[ID].cyclePending.nonce='replaced-nonce']])test(`Cycle token await changes original ${name}: no claim write`,async()=>{
 const f=fixture(),scene={id:'scene',tokens:new Map()},token={id:'a',uuid:'Scene.scene.Token.a',documentName:'Token',actorLink:true,actor:f.actor,parent:scene};scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);f.context.tokenUuid=token.uuid;f.message.flags.pf2e.context.target.token=token.uuid;pending(f);
 const c=createCycleCoordinator({game:f.game,fromUuid:async uuid=>{if(uuid===f.actor.uuid)return f.actor;change(f);return token;}});await assert.rejects(c.claim(f.context,f.owner));assert.equal(f.writes.length,0);
});
test('Cycle loses GM after durable claim: repeat cannot rearm or claim again',async()=>{
 const f=fixture();pending(f);const original=f.actor.update;f.actor.update=async(...args)=>{const result=await original(...args);f.game.users.activeGM=f.next;return result};const c=createCycleCoordinator({game:f.game,fromUuid:async()=>f.actor});await assert.rejects(c.claim(f.context,f.owner));assert.equal(f.actor.flags[ID].cyclePending.status,'claimed');
 f.game.users.activeGM=f.gm;f.actor.update=original;assert.equal(await c.claim(f.context,f.owner),null);assert.equal(f.writes.length,1);
});

test('active Cycle supplies one native contextual resistance and removes it from both prepared arrays after damage',async()=>{
 const f=fixture(),original=[],replacement=[],completed=[];f.actor.attributes={resistances:original};let clones=0,native=0,claims=0;
 const resistance={type:'custom',test:()=>true,getDoubledValue:()=>10};
 f.actor.getContextualClone=(options,effects)=>{clones++;assert.equal(effects.length,1);resistance.definition=effects[0].system.rules[0].definition;assert.ok(options.includes(resistance.definition[0]));assert.equal(effects[0].system.rules[0].value,5);return {attributes:{resistances:[resistance]}};};
 const automation=createCycleAutomation({iwrEnabled:()=>true,onClaim:async context=>{claims++;return {...context,nonce:'0123456789abcdef0123456789abcdef',reactionId:'once'}},onComplete:async result=>completed.push(result)});automation.recordDamageMessage(f.message);
 assert.equal(await automation.applyDamage(f.actor,async params=>{native++;assert.equal(params.damage,f.damage);assert.equal(original[0],resistance);replacement.push(resistance);f.actor.attributes.resistances=replacement;return 'native-applied'}, {damage:f.damage,item:f.item}),'native-applied');
 assert.deepEqual([claims,clones,native,completed.length],[1,1,1,1]);assert.equal(completed[0].applied,true);assert.deepEqual(original,[]);assert.deepEqual(replacement,[]);
});

test('a failed native Cycle damage call remains uncertain and never calls the native handler a second time',async()=>{
 const f=fixture(),completed=[];f.actor.attributes={resistances:[]};let native=0;
 const automation=createCycleAutomation({iwrEnabled:()=>true,onClaim:async context=>({...context,nonce:'0123456789abcdef0123456789abcdef',reactionId:'once'}),createResistance:()=>({}),onComplete:async result=>completed.push(result)});automation.recordDamageMessage(f.message);
 await assert.rejects(automation.applyDamage(f.actor,async()=>{native++;throw Error('native response lost')},{damage:f.damage,item:f.item}),/native response lost/);
 assert.equal(native,1);assert.equal(completed.length,1);assert.equal(completed[0].uncertain,true);assert.equal(completed[0].applied,false);assert.deepEqual(f.actor.attributes.resistances,[]);
});

test('the latest settled eligible damage card does not fall back to an older eligible attack',()=>{
 const f=fixture(),now=Date.now(),old={...f.message,id:'old',timestamp:now-20},latest={...f.message,id:'latest',timestamp:now-10,flags:{...f.message.flags,[ID]:{cycleApplications:[{...f.context,messageId:'latest',applied:true}]}}};
 const result=resolveCycleDamageMessage(f.actor,[old,latest],{now});assert.equal(result.messageId,'latest');assert.equal(result.status,'already-applied');
});

function nativeToken(f,{synthetic=false}={}){
 const scene={id:'scene',tokens:new Map()},baseActor=synthetic?{id:'base',uuid:'Actor.base'}:f.actor;
 if(synthetic){f.game.actors.delete(f.actor.id);f.game.actors.set(baseActor.id,baseActor);f.actor.isToken=true;f.actor.uuid='Scene.scene.Token.a.Actor.a';f.context.actorUuid=f.actor.uuid;f.message.flags.pf2e.context.target.actor=f.actor.uuid;}
 const token={id:'a',uuid:'Scene.scene.Token.a',documentName:'Token',parent:scene,actorLink:!synthetic,actorId:baseActor.id,baseActor,actor:f.actor};scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);if(synthetic)f.actor.token=token;f.context.tokenUuid=token.uuid;f.message.flags.pf2e.context.target.token=token.uuid;return token;
}
for(const synthetic of [false,true])test(`Cycle can claim and complete the exact current ${synthetic?'synthetic':'world'} Token once`,async()=>{
 const f=fixture(),token=nativeToken(f,{synthetic});pending(f);let effects=0;
 f.message.update=async changes=>{f.writes.push({kind:'damage-card'});f.message.flags[ID]??={};f.message.flags[ID].cycleApplications=structuredClone(changes[`flags.${ID}.cycleApplications`]);return f.message;};
 const c=createCycleCoordinator({game:f.game,fromUuid:async uuid=>uuid===f.actor.uuid?f.actor:token,onEffect:async actor=>{assert.equal(actor,f.actor);effects++;}}),claim=await c.claim(f.context,f.owner);
 await c.complete({context:f.context,claim,applied:true},f.owner);await c.complete({context:f.context,claim,applied:true},f.owner);assert.equal(effects,1);assert.equal(f.actor.flags[ID].cyclePending.status,'done');assert.equal(f.writes.filter(w=>w.kind==='damage-card').length,1);
});

for(const boundary of ['created','annotated'])test(`Clock original native card replaced after it is ${boundary}: retains its paid claim and cannot publish again`,async()=>{
 const f=fixture(),original=f.clock.toMessage;let cards=0;f.clock.toMessage=async()=>{cards++;const card=await original();if(boundary==='created')f.game.messages.set(card.id,{...card});else{const update=card.update;card.update=async changes=>{const result=await update.call(card,changes);f.game.messages.set(card.id,{...card});return result};}return card;};
 const api=createReactionChecks({game:f.game,fromUuid:async()=>f.actor,choose:async()=> 'clock'}),payload={nonce:'clock-card-uncertain',actorUuid:f.actor.uuid,type:'saving-throw',degree:1};await assert.rejects(api.decideCheckReaction(payload,f.owner));
 assert.equal(f.clock.system.frequency.value,0);assert.equal(f.actor.flags[ID].reactionChecks.reactions[0].state,'claimed');assert.equal(await api.decideCheckReaction(payload,f.owner),null);assert.equal(cards,1);
});

for(const synthetic of [false,true])test(`Clock accepts a current ${synthetic?'synthetic':'world'} original Token without borrowing a different Actor`,async()=>{
 const f=fixture(),token=nativeToken(f,{synthetic});f.clock.uuid=f.actor.uuid+'.Item.clock';const api=createReactionChecks({game:f.game,fromUuid:async uuid=>uuid===f.actor.uuid?f.actor:token,choose:async()=> 'clock'});
 assert.equal(await api.decideCheckReaction({nonce:'clock-original-token',actorUuid:f.actor.uuid,tokenUuid:token.uuid,type:'saving-throw',degree:1},f.owner),'clock');assert.equal(f.clock.system.frequency.value,0);
});

for(const label of ['origin Token','target Actor'])test(`Clock choice changes the original ${label}: no claim or payment`,async()=>{
 const f=fixture(),token=nativeToken(f),targetActor={id:'target',uuid:'Actor.target'},target={id:'target',uuid:'Scene.scene.Token.target',documentName:'Token',actor:targetActor,parent:token.parent};f.game.actors.set(targetActor.id,targetActor);token.parent.tokens.set(target.id,target);
 const api=createReactionChecks({game:f.game,fromUuid:async uuid=>uuid===f.actor.uuid?f.actor:uuid===token.uuid?token:target,choose:async()=>{if(label==='origin Token')token.parent.tokens.set(token.id,{...token});else f.game.actors.set(targetActor.id,{...targetActor});return 'clock'}});
 await assert.rejects(api.decideCheckReaction({nonce:'clock-original-docs',actorUuid:f.actor.uuid,tokenUuid:token.uuid,targetUuid:target.uuid,type:'saving-throw',degree:1},f.owner));assert.equal(f.clock.system.frequency.value,1);assert.deepEqual(f.writes,[]);
});

test('a synthetic Clock Actor without a native Token document cannot claim or pay',async()=>{
 const f=fixture(),token=nativeToken(f,{synthetic:true});delete token.documentName;const api=createReactionChecks({game:f.game,fromUuid:async()=>f.actor,choose:async()=> 'clock'});
 await assert.rejects(api.decideCheckReaction({nonce:'clock-false-synthetic',actorUuid:f.actor.uuid,type:'saving-throw',degree:1},f.owner));assert.equal(f.clock.system.frequency.value,1);assert.deepEqual(f.writes,[]);
});

function squawkFixture(){
 const f=fixture(),origin=nativeToken(f);origin.object={center:{x:0,y:0}};
 const recipient={id:'witness',uuid:'Actor.witness',items:new Map()},target={id:'witness',uuid:'Scene.scene.Token.witness',documentName:'Token',actor:recipient,parent:origin.parent,object:{center:{x:5,y:0},checkCollision:()=>false}};f.game.actors.set(recipient.id,recipient);origin.parent.tokens.set(target.id,target);
 recipient.createEmbeddedDocuments=async(_kind,rows)=>rows.map(data=>{f.writes.push({kind:'immunity'});const item={...structuredClone(data),id:'immunity',actor:recipient};recipient.items.set(item.id,item);return item;});
 const squawk={...f.clock,id:'squawk',uuid:f.actor.uuid+'.Item.squawk',sourceId:REACTION_CHECK_SOURCES.squawk,system:{},async toMessage(){const card=await f.clock.toMessage();card.item=this;card.flags.pf2e.origin.uuid=this.uuid;return card;}};f.actor.items.set(squawk.id,squawk);
 const payload={nonce:'squawk-original-docs',actorUuid:f.actor.uuid,tokenUuid:origin.uuid,targetUuid:target.uuid,type:'skill-check',degree:0,domains:['diplomacy'],fortune:true};
 const resolve=async uuid=>uuid===f.actor.uuid?f.actor:uuid===origin.uuid?origin:target;return {...f,origin,target,squawk,payload,resolve};
}
for(const alias of ['suppressed','isSuppressed','system.suppressed'])test(`Squawk excludes ${alias} but retains another active owned exact copy`,async()=>{
 const f=squawkFixture();if(alias==='system.suppressed')f.squawk.system.suppressed=true;else f.squawk[alias]=true;
 const api=createReactionChecks({game:f.game,fromUuid:f.resolve,choose:async()=> 'squawk'});assert.equal(await api.decideCheckReaction(f.payload,f.owner),null);assert.deepEqual(f.writes,[]);
 const active={...f.squawk,id:'squawk-active',uuid:f.actor.uuid+'.Item.squawk-active',suppressed:false,isSuppressed:false,system:{}};f.actor.items.set(active.id,active);
 assert.equal(await api.decideCheckReaction(f.payload,f.owner),'squawk');assert.equal(f.writes.filter(w=>w.kind==='immunity').length,1);assert.equal(f.clock.system.frequency.value,1);assert.equal(await api.decideCheckReaction(f.payload,f.owner),null);
});
test('Squawk loses its originally selected feature after choice: no claim or immunity write',async()=>{
 const f=squawkFixture(),api=createReactionChecks({game:f.game,fromUuid:f.resolve,choose:async()=>{suppressFeats([f.squawk]);return 'squawk'}});await assert.rejects(api.decideCheckReaction(f.payload,f.owner));assert.deepEqual(f.writes,[]);
});
test('Squawk witness update loses its original Token: claimed reaction cannot write or consume again',async()=>{
 const f=squawkFixture(),original=f.target.actor.createEmbeddedDocuments;f.target.actor.createEmbeddedDocuments=async(...args)=>{const result=await original(...args);f.target.parent.tokens.set(f.target.id,{...f.target});return result};
 const api=createReactionChecks({game:f.game,fromUuid:f.resolve,choose:async()=> 'squawk'});await assert.rejects(api.decideCheckReaction(f.payload,f.owner));assert.equal(f.actor.flags[ID].reactionChecks.reactions[0].state,'claimed');assert.equal(f.writes.filter(w=>w.kind==='immunity').length,1);assert.equal(f.writes.filter(w=>w.kind==='card').length,0);
 f.target.parent.tokens.set(f.target.id,f.target);assert.equal(await api.decideCheckReaction(f.payload,f.owner),null);assert.equal(f.writes.filter(w=>w.kind==='immunity').length,1);
});
