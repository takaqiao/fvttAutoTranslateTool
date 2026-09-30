import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture as base} from './glimpse-fixture.mjs';
let api={};try{api=await import('../scripts/spiritual-scar-source.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const resolve=f=>{assert.equal(typeof api.resolveSpiritualScarSource,'function');return api.resolveSpiritualScarSource({...f,actor:f.ally})};
const validate=(f,snapshot)=>api.validateSpiritualScarSource({...f,snapshot});
function fixture(){const f=base();f.enemy.system.traits.value=['fiend','devil'];f.roll.instances=[{type:'spirit',persistent:false,total:10}];f.params.rollOptions.add('origin:trait:fiend');f.message.blind=false;f.message.whisper=[];return f;}
test('binds actual target, original fiend and native card with privacy, independent of viewed combat',async()=>{
 const f=fixture(),r=await resolve(f);assert.equal(r.verified,true);assert.equal(r.sourceActor,f.enemy);assert.equal(r.sourceToken,f.enemyToken);assert.equal(r.actor,f.ally);assert.equal(r.token,f.allyToken);assert.equal(r.item,f.item);assert.equal(r.snapshot.sourceActorUuid,f.enemy.uuid);assert.deepEqual(r.privacy,{blind:false,whisper:[]});assert.equal((await validate(f,r.snapshot)).verified,true);
});
test('ability damage is eligible without native Strike metadata, attack sourceType, or enemy relationship',async()=>{
 const f=fixture();f.item.type='action';f.message.flags.pf2e.origin.type='action';delete f.message.flags.pf2e.strike;delete f.message.flags.pf2e.context.sourceType;f.enemy.isEnemyOf=()=>false;f.enemy.isAllyOf=()=>true;
 assert.equal((await resolve(f)).verified,true);
});
test('a known off-canvas origin actor does not need an attacker token or same-scene aura',async()=>{
 const f=fixture();f.message.speaker={actor:f.enemy.id};delete f.enemyToken.object;const r=await resolve(f);assert.equal(r.verified,true);assert.equal(r.sourceToken,null);assert.equal(r.sourceActor,f.enemy);assert.equal((await validate(f,r.snapshot)).verified,true);
});
test('rendered origin token is unnecessary and originating scene may differ',async()=>{
 const f=fixture(),other={id:'other',tokens:new Map()};f.scene.tokens.delete(f.enemyToken.id);f.enemyToken.parent=other;f.enemyToken.uuid='Scene.other.Token.enemy';delete f.enemyToken.object;other.tokens.set(f.enemyToken.id,f.enemyToken);f.game.scenes.set(other.id,other);f.docs.set(f.enemyToken.uuid,f.enemyToken);f.message.speaker.scene=other.id;
 const r=await resolve(f);assert.equal(r.verified,true);assert.equal(r.sourceToken,f.enemyToken);
});
test('itemless native fiend damage requires explicit actor provenance, not a guessed item',async()=>{
 const f=fixture();f.params.item=null;f.message.flags.pf2e.origin={actor:f.enemy.uuid};delete f.message.flags.pf2e.strike;delete f.message.flags.pf2e.context.sourceType;const r=await resolve(f);assert.equal(r.verified,true);assert.equal(r.item,null);assert.equal(r.snapshot.itemUuid,null);
});
test('the actual native recipient can be selected at damage application without a pre-recorded attack target',async()=>{
 const f=fixture();delete f.message.flags.pf2e.context.target;const r=await resolve(f);assert.equal(r.verified,true);assert.equal(r.snapshot.tokenUuid,f.allyToken.uuid);
});
test('Toolbelt multiple recipients bind the actual damage invocation without borrowing attack-only target restrictions',async()=>{
 const f=fixture();f.message.flags.pf2e.context.target={actor:f.champion.uuid,token:f.championToken.uuid};f.message.flags['pf2e-toolbelt']={targetHelper:{targets:[f.championToken.uuid,f.allyToken.uuid]}};assert.equal((await resolve(f)).verified,true);
});
test('a positive evaluated persistent spirit tick can qualify when the fiend origin is actually retained',async()=>{
 const f=fixture();f.roll.options.evaluatePersistent=true;f.roll.instances[0].persistent=true;assert.equal((await resolve(f)).verified,true);
});
test('a later native damage roll on the same original card keeps its own index',async()=>{
 const f=fixture();f.message.rolls.unshift({...f.roll,toJSON:()=>({class:'DamageRoll',evaluated:true,total:2})});f.source.rollIndex=1;const r=await resolve(f);assert.equal(r.verified,true);assert.equal(r.snapshot.rollIndex,1);
});
for(const[name,change]of[
 ['missing private roll context',f=>f.source=null],['numeric damage',f=>f.params.damage=10],['final damage',f=>f.params.final=true],['skipped IWR',f=>f.params.skipIWR=true],['no positive spirit',f=>f.roll.instances[0].type='fire'],['future persistent creation only',f=>f.roll.instances[0].persistent=true],['zero incoming damage',f=>f.roll.total=0],['non-fiend origin',f=>f.enemy.system.traits.value=['humanoid']],['missing native fiend option',f=>f.params.rollOptions.clear()],['foreign origin actor',f=>f.message.flags.pf2e.origin.actor=f.champion.uuid],['unknown author',f=>f.message.author={id:'unknown'}],['actor-shaped fake target token',f=>f.params.token={...f.allyToken}],['origin item on another actor',f=>f.item.actor=f.champion],['wrong application item',f=>f.params.item={...f.item,uuid:'Actor.enemy.Item.other'}],['merged ambiguous sources',f=>f.message.flags['pf2e-toolbelt']={betterChat:{mergeDamage:true}}],['unresolved source token',f=>f.message.speaker.token='missing'],['unresolved origin item',f=>f.docs.delete(f.item.uuid)],['non-native card',f=>f.message.isDamageRoll=false],['unevaluated original roll',f=>f.roll.toJSON=()=>({class:'DamageRoll',evaluated:false,total:10})]
])test(`does not authorize Scar evidence for ${name}`,async()=>{const f=fixture();change(f);assert.equal((await resolve(f)).verified,false);});
test('changing damage, owner, private recipients, or fiend identity invalidates prior evidence',async()=>{
 for(const change of [f=>f.roll.total=11,f=>f.message.author={id:'unknown'},f=>f.message.whisper=['gm'],f=>f.message.blind=true,f=>f.enemy.system.traits.value=[]]){const f=fixture(),r=await resolve(f);assert.equal(r.verified,true);change(f);assert.equal((await validate(f,r.snapshot)).verified,false);}
});
test('normal application receipts do not invalidate immutable damage evidence',async()=>{
 const f=fixture();f.message.flags['pf2e-toolbelt']={targetHelper:{targets:[f.allyToken.uuid]}};const r=await resolve(f);f.message.flags['pf2e-toolbelt'].targetHelper.applied={ally:{0:true}};assert.equal((await validate(f,r.snapshot)).verified,true);
});
test('private card classification is explicit and cannot be edited through the returned evidence',async()=>{
 const f=fixture();f.message.blind=true;f.message.whisper=['u'];const r=await resolve(f);assert.equal(r.verified,true);assert.deepEqual(r.privacy,{blind:true,whisper:['u']});assert.throws(()=>r.privacy.whisper.push('other'));assert.throws(()=>r.snapshot.itemUuid='forged');assert.deepEqual(f.message.whisper,['u']);
});
test('author ownership changing during an awaited item lookup is rechecked',async()=>{
 const f=fixture();f.user.isGM=false;const original=f.fromUuid;f.fromUuid=async uuid=>{const d=await original(uuid);if(uuid===f.item.uuid)f.enemy.testUserPermission=()=>false;return d;};assert.equal((await resolve(f)).verified,false);
});
test('an off-canvas source actor removed during resolution cannot remain authoritative',async()=>{
 const f=fixture();f.message.speaker={actor:f.enemy.id};const original=f.fromUuid;f.fromUuid=async uuid=>{const d=await original(uuid);if(uuid===f.item.uuid)f.game.actors.delete(f.enemy.id);return d;};assert.equal((await resolve(f)).verified,false);
});
test('an item removed while UUID resolution is pending cannot be used as a live origin',async()=>{
 const f=fixture(),original=f.fromUuid;f.fromUuid=async uuid=>{const d=await original(uuid);if(uuid===f.item.uuid)f.enemy.items.delete(f.item.id);return d;};assert.equal((await resolve(f)).verified,false);
});

test('native prepared unarmed and alternate-usage items remain valid without an embedded document',async()=>{
 for(const alternate of [false,true]){const f=fixture();f.enemy.items.delete(f.item.id);f.docs.delete(f.item.uuid);f.enemy.system.actions=alternate?[{altUsages:[{item:f.item}]}]:[{item:f.item}];assert.equal((await resolve(f)).verified,true);}
});

test('an actual off-canvas synthetic token actor has its own live document authority',async()=>{
 const f=fixture();f.game.actors.delete(f.enemy.id);f.enemy.isToken=true;f.enemy.token=f.enemyToken;f.enemy.uuid=f.enemyToken.uuid+'.Actor.enemy';f.docs.set(f.enemy.uuid,f.enemy);f.message.flags.pf2e.origin.actor=f.enemy.uuid;delete f.enemyToken.object;
 const r=await resolve(f);assert.equal(r.verified,true);f.scene.tokens.delete(f.enemyToken.id);assert.equal((await validate(f,r.snapshot)).verified,false);
});
