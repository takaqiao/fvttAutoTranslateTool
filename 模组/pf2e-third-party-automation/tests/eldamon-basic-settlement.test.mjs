import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './eldamon-electricity-fixture.mjs';
import {ELECTRICITY_SOURCES as E,electricityEffects} from '../scripts/eldamon-electricity.mjs';
import {createEldamonBasicSettlement} from '../scripts/eldamon-basic-settlement.mjs';
import {createEldamonElectricityProvider} from '../scripts/eldamon-electricity-provider.mjs';

const NS='pf2e-third-party-automation';
const sources={element:'Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.9KtNlRXeuxZSoVaI',shield:'Compendium.battlezoo-eldamon-pf2e.actions.Item.g8lH9enxTY6Cpx9V',manipulation:'Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.h4D0hXhSHqmyBEQN'};
function setup(kind='manipulation'){
 const f=fixture();
 f.game.world={id:'ujx5r8oipw7ercdr'};f.game.system={id:'pf2e',version:'8.5.1'};
 f.game.modules=new Map([['battlezoo-eldamon-pf2e',{active:true}]]);
 f.item(f.caster,'element',sources.element,{type:'feat'});
 f.power.sourceId=sources[kind];
 f.card.flags.pf2e.origin={uuid:f.power.uuid,type:'feat',actor:f.caster.uuid};
 const payload={actorUuid:f.caster.uuid,itemUuid:f.power.uuid,messageUuid:f.card.uuid,targetUuid:f.tokens[1].uuid,nonce:'confirmed-basic-1',confirmed:true};
 const run=(patch={},user=f.owner)=>f.ledger().confirmedAction?.({...payload,...patch},user);
 return {...f,payload,run};
}

test('confirmed manipulation applies native Shocked to the explicit T target for two source turns',async()=>{
 const f=setup();await f.run();
 const effects=electricityEffects(f.target,E.shocked);assert.equal(effects.length,1);
 assert.equal(electricityEffects(f.caster,E.shocked).length,0);
 assert.equal(effects[0].system.context.origin.actor,f.caster.uuid);
 assert.deepEqual(effects[0].flags[NS].electricityShock.expires,{combatId:'c',combatantId:'casterturn',round:3,phase:'start'});
 await f.ledger().expire({combat:{...f.combat,round:2},combatant:f.combat.turns[0],phase:'end',actors:[f.target]});
 assert.equal(electricityEffects(f.target,E.shocked).length,1);
 await f.ledger().expire({combat:{...f.combat,round:3},combatant:f.combat.turns[1],phase:'start',actors:[f.target]});
 assert.equal(electricityEffects(f.target,E.shocked).length,1);
 await f.ledger().expire({combat:{...f.combat,round:3},combatant:f.combat.turns[0],phase:'start',actors:[f.target]});
 assert.equal(electricityEffects(f.target,E.shocked).length,0);
});
test('confirmed shield trigger lasts through next source turn and does not add Charged',async()=>{
 const f=setup('shield');f.combat.turn=1;await f.run();
 const effect=electricityEffects(f.target,E.shocked)[0];assert.ok(effect);
 assert.deepEqual(effect.flags[NS].electricityShock.expires,{combatId:'c',combatantId:'casterturn',round:2,phase:'end'});
 assert.equal(electricityEffects(f.caster,E.charged).length,0);
 assert.equal(electricityEffects(f.target,E.charged).length,0);
});
test('replaying the same confirmation cannot reapply consumed Shocked; a new trigger can',async()=>{
 const f=setup('shield');await f.run();assert.equal(electricityEffects(f.target,E.shocked).length,1);
 await f.ledger().interact({actorUuid:f.target.uuid,nonce:'discharge',confirmed:true},f.gm);
 await f.run();assert.equal(electricityEffects(f.target,E.shocked).length,0);
 await f.run({nonce:'confirmed-basic-2'});assert.equal(electricityEffects(f.target,E.shocked).length,1);
});
test('repeat uses refresh the same owned source effect without changing another source effect',async()=>{
 const f=setup();const unrelated=f.item(f.target,'unrelated',E.shocked,{system:{duration:{unit:'unlimited'}},flags:{[NS]:{electricityShock:{key:'unrelated',sourceActorUuid:f.other.uuid}}}});
 await f.run();f.combat.round=2;await f.run({nonce:'confirmed-basic-2'});
 assert.equal(electricityEffects(f.target,E.shocked).length,2);
 assert.equal(f.target.items.get(unrelated.id),unrelated);
 assert.equal(electricityEffects(f.target,E.shocked).find(e=>e!==unrelated).flags[NS].electricityShock.expires.round,4);
});
test('confirmation must name the actual owned elemental card and cannot be forged by another user',async()=>{
 const f=setup();assert.equal(typeof f.ledger().confirmedAction,'function');
 await assert.rejects(f.run({confirmed:false}));
 await assert.rejects(f.run({messageUuid:'ChatMessage.absent'}));
 const unauthorized={id:'intruder'};f.game.users.set(unauthorized.id,unauthorized);
 await assert.rejects(f.run({},unauthorized));
 f.caster.items.delete('element');await assert.rejects(f.run());
 assert.equal(electricityEffects(f.target,E.shocked).length,0);
});
test('a movement or LOS helper is never needed after the user confirms the trigger',async()=>{
 const f=setup();for(const t of f.tokens)t.object={distanceTo(){throw Error('No geometry needed')},checkCollision(){throw Error('No LOS needed')}};
 await f.run();assert.equal(electricityEffects(f.target,E.shocked).length,1);
});

test('the active GM ledger rejects unsupported worlds, versions or disabled Eldamon even for a valid owner',async()=>{
 for(const change of [f=>f.game.world.id='another',f=>f.game.system.version='8.6.0',f=>f.game.system.id='sf2e',f=>f.game.modules.clear()]){
  const f=setup();change(f);await assert.rejects(f.run());assert.equal(electricityEffects(f.target,E.shocked).length,0);
 }
});

test('origin-matching roll and damage-receipt cards cannot invoke the GM basic-action settlement',async()=>{
 for(const type of ['attack-roll','damage-roll','saving-throw','skill-check','damage-taken']){
  const f=setup();f.card.flags.pf2e.context={type};await assert.rejects(f.run());assert.equal(electricityEffects(f.target,E.shocked).length,0);
 }
 const f=setup();f.card.isRoll=true;await assert.rejects(f.run());
});

test('a lost GM reply retries the original confirmation even after Shocked is consumed and the player retargets',async()=>{
 const f=setup('shield');f.owner.targets=new Set([{document:f.tokens[1]}]);f.game.user=f.owner;
 const deliveries=[],prompts=[];let attempts=0;
 const provider=createEldamonBasicSettlement({game:f.game,random:()=>`ui-confirmation-${++attempts}`,notify:()=>{},confirm:async context=>{prompts.push(context);return true},apply:async payload=>{
  deliveries.push(payload);f.game.user=f.gm;
  try{const value=await f.ledger().confirmedAction(payload,f.owner);if(deliveries.length===1)throw Error('GM reply lost');return value}finally{f.game.user=f.owner}
 }});
 await assert.rejects(provider.settleFromCard(f.card),/GM reply lost/);
 assert.equal(electricityEffects(f.target,E.shocked).length,1);
 f.game.user=f.gm;await f.ledger().interact({actorUuid:f.target.uuid,nonce:'release-first',confirmed:true},f.gm);f.game.user=f.owner;
 f.owner.targets=new Set([{document:f.tokens[2]}]);await provider.settleFromCard(f.card);
 assert.equal(electricityEffects(f.target,E.shocked).length,0);assert.equal(electricityEffects(f.other,E.shocked).length,0);
 assert.deepEqual(deliveries[1],deliveries[0]);assert.equal(prompts[1].retry,true);
 // After acknowledged completion, a deliberate new shield trigger is a new use.
 f.owner.targets=new Set([{document:f.tokens[1]}]);await provider.settleFromCard(f.card);
 assert.notEqual(deliveries[2].nonce,deliveries[0].nonce);assert.equal(electricityEffects(f.target,E.shocked).length,1);
});

test('a definite pre-mutation GM rejection releases the card for a later valid target through the socket adapter',async()=>{
 const f=setup('shield'),handlers=new Map(),Hooks={on:()=>{}};
 const gmProvider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid)});
 gmProvider.register({Hooks,socket:{register:(name,fn)=>handlers.set(name,fn)}});
 const playerGame={...f.game,user:f.owner};f.owner.targets=new Set([{document:f.tokens[1]}]);
 const client=createEldamonElectricityProvider({game:playerGame,fromUuid:async uuid=>f.docs.get(uuid)});
 client.register({Hooks,socket:{register:()=>{},executeAsUser:async(name,gmId,payload)=>{assert.equal(gmId,f.gm.id);return handlers.get(name).call({socketdata:{userId:f.owner.id}},payload)}}});
 let clicks=0;
 const provider=createEldamonBasicSettlement({game:playerGame,random:()=>`rejected-attempt-${clicks+1}`,notify:()=>{},confirm:async()=>{if(++clicks===1)f.scene.tokens.delete(f.tokens[1].id);return true},apply:payload=>client.confirmedAction(payload)});
 await assert.rejects(provider.settleFromCard(f.card),error=>error.electricityNotApplied===true);
 assert.equal(electricityEffects(f.target,E.shocked).length,0);
 f.owner.targets=new Set([{document:f.tokens[2]}]);await provider.settleFromCard(f.card);
 assert.equal(electricityEffects(f.other,E.shocked).length,1);
});

test('a failure after native effect creation is never described as safely unapplied',async()=>{
 const f=setup();f.target.update=async()=>{throw Error('state write lost')};
 await assert.rejects(f.run(),error=>error.message==='state write lost'&&error.electricityNotApplied!==true);
 assert.equal(electricityEffects(f.target,E.shocked).length,1);
});
test('outside combat manipulation uses native two-round duration; shield requests manual expiry',async()=>{
 for(const kind of ['manipulation','shield']){
  const f=setup(kind);f.game.combats.clear();f.game.combat=null;
  const result=await f.run();const effect=electricityEffects(f.target,E.shocked)[0];assert.ok(effect);
  assert.equal(effect.flags[NS].electricityShock.expires,null);
  assert.equal(effect.system.duration.unit,kind==='manipulation'?'rounds':'unlimited');
  assert.equal(effect.system.duration.expiry,kind==='manipulation'?'turn-start':null);
  assert.equal(result.manualExpiry,kind==='shield');
 }
});
