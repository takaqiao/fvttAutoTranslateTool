import test from 'node:test';
import assert from 'node:assert/strict';
import {createReactionChecks,REACTION_CHECK_SOURCES} from '../scripts/reaction-checks.mjs';
import {EAT_FORTUNE_SOURCES} from '../scripts/eat-fortune.mjs';
import {createFearAutomation,battleCryReactionAvailable,FEAR_SOURCES} from '../scripts/fear-automation.mjs';
import {createTranscendentDeflection} from '../scripts/transcendent-deflection.mjs';
import {TRANSCENDENT_DEFLECTION_SOURCE} from '../scripts/transcendent-deflection-rules.mjs';
import {createDisruptPrey} from '../scripts/disrupt-prey.mjs';
import {DISRUPT_PREY_SOURCE} from '../scripts/disrupt-prey-rules.mjs';
import {fixture as damageFixture} from './glimpse-fixture.mjs';
const M='pf2e-third-party-automation';
function patch(doc,changes){for(const [path,value]of Object.entries(changes)){let at=doc;const keys=path.split('.');for(const k of keys.slice(0,-1))at=at[k]??={};at[keys.at(-1)]=structuredClone(value)}}
function basic(){
 const user={id:'gm',isGM:true,active:true},users=new Map([[user.id,user]]);users.activeGM=user;
 const actor={id:'pc',uuid:'Actor.pc',type:'character',items:new Map(),flags:{},canAct:true,isDead:false,testUserPermission:u=>u===user,async update(p){patch(this,p)}};
 const game={user,users,actors:new Map([[actor.id,actor]]),scenes:new Map(),messages:new Map(),modules:new Map(),time:{worldTime:100}};
 const item={id:'clock',uuid:`${actor.uuid}.Item.clock`,type:'feat',actor,sourceId:REACTION_CHECK_SOURCES.clock,system:{frequency:{value:1,max:1}},async update(p){patch(this,p)},async toMessage(){return {id:'paid',flags:{},async update(p){patch(this,p)}}}};actor.items.set(item.id,item);
 return {actor,item,game,user};
}
for(const status of ['restricted','manual'])test(`Clock ${status} blocks its no-encounter branch without paying or offering`,async()=>{
 const f=basic();let prompts=0;const provider=createReactionChecks({game:f.game,fromUuid:async()=>f.actor,reactionRestriction:()=>({status}),choose:async()=>{prompts++;return 'clock'}});
 assert.equal(await provider.decideCheckReaction({actorUuid:f.actor.uuid,nonce:'clock-test',type:'saving-throw',degree:1},f.user),null);
 assert.equal(prompts,0);assert.equal(f.item.system.frequency.value,1);assert.deepEqual(f.actor.flags,{});
});
test('Clock rechecks restriction after selection and preserves a later real ordinary use',async()=>{
 const f=basic();let status='clear',duringChoice=true;const provider=createReactionChecks({game:f.game,fromUuid:async()=>f.actor,reactionRestriction:()=>({status}),choose:async()=>{if(duringChoice)status='manual';return 'clock'}});
 const payload={actorUuid:f.actor.uuid,nonce:'clock-test',type:'saving-throw',degree:1};
 assert.equal(await provider.decideCheckReaction(payload,f.user),null);assert.equal(f.item.system.frequency.value,1);
 status='clear';duringChoice=false;assert.equal(await provider.decideCheckReaction(payload,f.user),'clock');assert.equal(f.item.system.frequency.value,0);
});
for(const status of ['restricted','manual'])test(`Battle Cry ${status} blocks before no-epoch fallback`,()=>{
 const f=basic();assert.equal(battleCryReactionAvailable(f.actor,f.game,{reactionRestriction:()=>({status})}),false);
 assert.equal(battleCryReactionAvailable(f.actor,f.game),true);assert.deepEqual(f.actor.flags,{});
});

function eatSetup(query,changeOnChoice){
 const f=basic(),source={...f.actor,id:'roller',uuid:'Actor.roller',items:new Map(),flags:{[M]:{reactionChecks:{reactions:[{kind:'clock',nonce:'paid-clock',state:'claimed',checkId:'source-card'}]}}}},item={...f.item,id:'eat',uuid:`${f.actor.uuid}.Item.eat`,sourceId:EAT_FORTUNE_SOURCES.eat};f.actor.items=new Map([[item.id,item]]);
 const clock={...f.item,id:'source-clock',uuid:`${source.uuid}.Item.clock`,actor:source};source.items.set(clock.id,clock);f.game.actors.set(source.id,source);
 const scene={id:'scene',tokens:new Map()},token=a=>{const t={documentName:'Token',id:a.id,uuid:`Scene.scene.Token.${a.id}`,actor:a,parent:scene,object:{distanceTo:()=>5}};scene.tokens.set(t.id,t);return t},roller=token(source),reactor=token(f.actor);f.game.scenes.set(scene.id,scene);
 const card={id:'source-card',item:clock,flags:{[M]:{reactionChecks:{kind:'reaction-use',nonce:'paid-clock'}}}};f.game.messages.set(card.id,card);
 const docs=new Map([source,clock,roller,reactor,f.actor,item].map(d=>[d.uuid,d]));let prompts=0;
 const provider=createReactionChecks({game:f.game,fromUuid:async uuid=>docs.get(uuid),reactionRestriction:query,choose:async()=>{prompts++;changeOnChoice?.();return 'eat'}});
 provider.eatFortune.register({Hooks:{on:()=>1,off(){}}});
 const payload={nonce:'eat-test',kind:'clock',clockNonce:'paid-clock',sourceItemUuid:clock.uuid,sourceActorUuid:source.uuid,sourceTokenUuid:roller.uuid,rollerActorUuid:source.uuid,rollerTokenUuid:roller.uuid,effectType:'fortune',type:'saving-throw',options:[]};
 return {...f,item,run:()=>provider.eatFortune.decide(payload,f.user),prompts:()=>prompts};
}
for(const status of ['restricted','manual'])test(`Eat Fortune inherits ${status} through ReactionChecks with no encounter`,async()=>{
 const f=eatSetup(()=>({status}));assert.equal(await f.run(),null);assert.equal(f.prompts(),0);assert.equal(f.item.system.frequency.value,1);assert.deepEqual(f.actor.flags,{});
});
test('Eat Fortune rechecks new restriction after its awaited choice',async()=>{
 let status='clear';const f=eatSetup(()=>({status}),()=>status='restricted');assert.equal(await f.run(),null);assert.equal(f.prompts(),1);assert.equal(f.item.system.frequency.value,1);assert.deepEqual(f.actor.flags,{});
});

function meleeFixture(source){
 const f=damageFixture();f.game.combat=null;f.game.modules=new Map();f.game.time={worldTime:100};
 for(const actor of f.game.actors.values()){actor.flags={};actor.update=async p=>patch(actor,p)}
 const actor=f.champion,token=f.championToken,feat={id:'feat',uuid:`${actor.uuid}.Item.feat`,type:'feat',sourceId:source,actor,system:{frequency:{value:1}}};actor.items.set(feat.id,feat);
 const weapon={id:'weapon',uuid:`${actor.uuid}.Item.weapon`,actor,type:'weapon',isMelee:true,isRanged:false,system:{usage:{hands:1},equipped:{carryType:'held',handsHeld:1},traits:{value:[]}}};actor.items.set(weapon.id,weapon);f.docs.set(weapon.uuid,weapon);
 actor.getReach=()=>5;actor.system.actions=[{type:'strike',ready:true,item:weapon,variants:[{roll:async()=>{}}]}];token.object.distanceTo=()=>5;token.object.checkCollision=()=>false;f.enemyToken.getCenterPoint=()=>({x:1,y:1});
 return {...f,actor,token,feat,weapon};
}
for(const status of ['clear','restricted','manual'])test(`Deflection no-encounter candidate uses ${status} restriction query`,async()=>{
 const f=meleeFixture(TRANSCENDENT_DEFLECTION_SOURCE);let prompts=0;
 const provider=createTranscendentDeflection({game:f.game,fromUuid:f.fromUuid,getRollContext:()=>f.source,nativeBridgeAvailable:()=>true,reactionRestriction:()=>({status}),choose:async()=>{prompts++;return 'decline'},weapons:{}});
 const prepared=await provider.beforeDamage(f.ally,f.params);assert.equal(provider.hasNativePlan(f.ally,prepared.params),true);
 await provider.interceptNative({actor:f.ally,params:prepared.params,incoming:10,persistent:[],shield:null,prevent:()=>assert.fail('declined or restricted')});
 assert.equal(prompts,status==='clear'?1:0);assert.equal(f.actor.flags[M]?.transcendentDeflection?.reactions?.length??0,0);
});
test('Deflection rechecks restriction after the asynchronous weapon choice before paying or breaking a weapon',async()=>{
 const f=meleeFixture(TRANSCENDENT_DEFLECTION_SOURCE);let status='clear',broken=0;
 const provider=createTranscendentDeflection({game:f.game,fromUuid:f.fromUuid,getRollContext:()=>f.source,nativeBridgeAvailable:()=>true,reactionRestriction:()=>({status}),choose:async c=>{status='manual';return c.choices[0].value},weapons:{breakWeapon:async()=>{broken++}}});
 const prepared=await provider.beforeDamage(f.ally,f.params);
 await assert.rejects(provider.interceptNative({actor:f.ally,params:prepared.params,incoming:10,persistent:[],shield:null,prevent:()=>assert.fail('must not prevent')}),/改变/);
 assert.equal(broken,0);assert.equal(f.actor.flags[M]?.transcendentDeflection?.reactions?.length??0,0);assert.equal(f.feat.system.frequency.value,1);
});
for(const moment of ['initial','after-choice'])test(`Battle Cry factory forwards restriction ${moment} with no encounter`,async()=>{
 const f=meleeFixture(FEAR_SOURCES.battle);f.actor.skills={intimidation:{rank:4}};f.actor.name='reactor';let status=moment==='initial'?'manual':'clear',prompts=0,nativeCalls=0;
 const message={id:'critical',actor:f.actor,author:f.user,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},rolls:[{}],flags:{pf2e:{context:{type:'attack-roll',outcome:'criticalSuccess',target:{token:f.enemyToken.uuid}}}},async update(p){patch(this,p)}};f.game.messages.set(message.id,message);
 f.game.pf2e={actions:new Map([['demoralize',{toActionVariant:()=>({use:async()=>{nativeCalls++;return [{message:{id:'check'}}]}})}]])};
 const provider=createFearAutomation({game:f.game,fromUuid:f.fromUuid,reactionRestriction:()=>({status}),choose:async()=>{prompts++;status='restricted';return f.enemyToken.uuid}});
 if(moment==='initial')await provider.battleCry(message,f.user.id);else await assert.rejects(provider.battleCry(message,f.user.id),/改变/);
 assert.equal(prompts,moment==='initial'?0:1);assert.equal(nativeCalls,0);assert.equal(f.actor.flags[M]?.fear?.reactions?.length??0,0);
});
for(const moment of ['initial','after-source-proof'])test(`Disrupt Prey refuses a restriction ${moment}`,async()=>{
 const f=meleeFixture(DISRUPT_PREY_SOURCE);f.game.combat=f.combat;f.actor.synthetics={tokenMarks:new Map([[f.enemyToken.uuid,['hunted-prey']]])};let status=moment==='initial'?'restricted':'clear',prompts=0,executions=0;
 const provider=createDisruptPrey({game:f.game,fromUuid:f.fromUuid,reactionRestriction:()=>({status}),choose:async c=>{prompts++;return c.choices[0].value},validateSource:async(_e,c)=>{if(c.stage==='claim')status='manual';return true},performStrike:async()=>{executions++}});
 const event={eventId:'event',nonce:'disrupt-test',actorUuid:f.actor.uuid,tokenUuid:f.token.uuid,sourceActorUuid:f.enemy.uuid,sourceTokenUuid:f.enemyToken.uuid,sourceUserId:f.user.id,kind:'stand',phase:'after-action'};
 const result=await provider.handleConfirmed(event);assert.equal(result.status,'ineligible');assert.equal(prompts,moment==='initial'?0:1);assert.equal(executions,0);assert.equal(f.actor.flags[M]?.disruptPrey?.reactions?.length??0,0);
});
