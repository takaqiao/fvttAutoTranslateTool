import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createElectricityLedger,ELECTRICITY_SOURCES as S,electricityState,electricityEffects} from '../scripts/eldamon-electricity.mjs';
const ID='pf2e-third-party-automation';
import {fixture} from './eldamon-electricity-fixture.mjs';
import {createReactionBudget,genericReactionAvailable} from '../scripts/reaction-budget.mjs';

// Foundry recursively expands dot keys, then merges updates without deleting
// omitted properties. Model that persistence boundary, including explicit -=.
function useFoundryFlagUpdates(actor){
 const plain=v=>v&&typeof v==='object'&&!Array.isArray(v);
 const expand=v=>{if(Array.isArray(v))return v.map(expand);if(!plain(v))return v;const out={};for(const [key,value]of Object.entries(v)){const parts=key.split('.');let at=out;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=expand(value);}return out;};
 const merge=(target,change)=>{for(const [key,value]of Object.entries(change)){if(key.startsWith('-=')){delete target[key.slice(2)];continue;}if(plain(value)&&plain(target[key]))merge(target[key],value);else target[key]=structuredClone(value);}};
 actor.update=async function(patch){merge(this,expand(patch));};
}
function anvilSave(f){
 f.power.sourceId=f.receipt.sourceUuid=S.anvil;
 const save={id:'save',uuid:'ChatMessage.save',isCheckRoll:true,rolls:[{_evaluated:true,total:10}],speaker:{actor:f.target.id,scene:'s',token:f.target.id},flags:{pf2e:{origin:{uuid:f.power.uuid},context:{type:'saving-throw',outcome:'failure',options:[`${ID}:metapower:channel:channel`]}}}};
 f.game.messages.set(save.id,save);f.docs.set(save.uuid,save);return save;
}
test('Anvil settles a bound native save target chosen after the original power card',async()=>{
 const f=fixture(),save=anvilSave(f);f.receipt.selection.targetUuids=[];
 await f.ledger().check(save);
 assert.equal(Object.values(electricityState(f.target).pendingShocks).length,1);
 await (await f.damage()).finish();assert.equal(electricityEffects(f.target,S.shocked).length,1);
});
test('pending Anvil survives Foundry dot expansion and is consumed once after native damage',async()=>{
 const f=fixture();useFoundryFlagUpdates(f.target);f.target.flags.unrelated={keep:true};const save=anvilSave(f);
 await f.ledger().check(save);
 assert.equal(Object.values(electricityState(f.target).pendingShocks)[0]?.effectKey,'channel:Actor.caster:channel');
 const damage=await f.damage();await damage.finish();await damage.finish();
 assert.equal(electricityEffects(f.target,S.shocked).length,1);assert.deepEqual(electricityState(f.target).pendingShocks,{});
 assert.deepEqual(f.target.flags.unrelated,{keep:true});
});
test('pending Anvil expiry persists deletion under Foundry recursive merge and cannot apply after expiry',async()=>{
 const f=fixture();useFoundryFlagUpdates(f.target);const save=anvilSave(f);await f.ledger().check(save);
 f.combat.round=2;await f.ledger().expire({combat:f.combat,combatant:f.combat.turns[0],phase:'end',actors:[f.target]});
 assert.deepEqual(electricityState(f.target).pendingShocks,{});
 await(await f.damage()).finish();assert.equal(electricityEffects(f.target,S.shocked).length,0);
});

for(const status of ['restricted','manual'])test(`Reactive Chain ${status} survives a bounded encounter game and injected availability`,async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);await(await f.damage()).finish();
 const payload={actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'siphoning'};
 let current=status;const ledger=createElectricityLedger({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),reactionAvailable:()=>true,reactionRestriction:actor=>{assert.equal(actor,f.caster);return {status:current}}});
 f.game.combat={id:'other-viewed',started:true,round:9,turn:0,turns:[]};
 assert.equal((await ledger.candidates(payload,f.owner)).length,0);current='clear';assert.equal((await ledger.candidates(payload,f.owner)).length,1);
});

test('authentic confirmed zero releases an original area target, while unresolved and changed receipts remain excluded',async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);const m=f.source('pure',[f.tokens[1].uuid,f.tokens[2].uuid]);await (await f.damage(f.target,{m})).finish();
 const payload={actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'siphoning'};
 const candidates=()=>f.ledger().candidates(payload,f.owner);assert.equal((await candidates()).length,0);
 const zero=await f.damage(f.other,{m,amount:0});assert.equal((await candidates()).length,0);await zero.finish();assert.equal((await candidates()).length,1);
 zero.receipt.flags[ID].electricityApplied.amount=1;assert.equal((await candidates()).length,0);zero.receipt.flags[ID].electricityApplied.amount=0;
 assert.equal((await candidates()).length,1);const sibling=await f.damage(f.other,{m,amount:0});assert.equal((await candidates()).length,0);await sibling.finish();assert.equal((await candidates()).length,1);
 const positive=await f.damage(f.other,{m,amount:1});await positive.finish();f.item(f.other,'restoredShock',S.shocked);assert.equal((await candidates()).length,0);
});
test('Reactive Chain reserves final native Toolbelt area recipients, even when pre-create fallback targets differ',async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);const m=f.source('pure',[f.tokens[0].uuid]);
 m.flags['pf2e-toolbelt']={targetHelper:{targets:[f.tokens[1].uuid,f.tokens[2].uuid]}};
 await (await f.damage(f.target,{m})).finish();
 const payload={actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'siphoning'};
 assert.equal((await f.ledger().candidates(payload,f.owner)).length,0,'unapplied actual AOE recipient is reserved');
 await (await f.damage(f.other,{m,amount:0})).finish();assert.equal((await f.ledger().candidates(payload,f.owner)).length,1);
 m.flags['pf2e-toolbelt'].targetHelper.targets=[f.tokens[1].uuid];
 assert.equal((await f.ledger().candidates(payload,f.owner)).length,0,'changed saved target manifest invalidates the source proof');
});
test('legacy damage source fingerprints can finish their existing receipt after target proof is added',async()=>{
 const f=fixture(),m=f.source();m.flags['pf2e-toolbelt']={targetHelper:{targets:[f.tokens[1].uuid]}};const applied=await f.damage(f.target,{m});
 const record=f.target.flags[ID].electricity.damage[applied.payload.nonce];delete record.sourceTargets;
 record.sourceFingerprint=JSON.stringify({pf:m.flags.pf2e,source:m.flags[ID].electricitySource,rolls:m.rolls.map(r=>({options:r.options,instances:r.instances}))});
 await applied.finish();assert.equal(electricityState(f.target).damage[applied.payload.nonce].status,'confirmed');
});

test('hostile independent Shocked retains its electricity save penalty through Resistant Shell without changing the original template',async()=>{
 const f=fixture();f.power.sourceId=f.receipt.sourceUuid=S.anvil;f.item(f.target,'shell',S.shell);
 const original={type:'effect',flags:{},system:{rules:[{key:'FlatModifier',selector:['fortitude','reflex'],value:-2,predicate:[{and:['electricity',{nor:['resistant-shell']}]}]},{key:'RollOption',domain:'all',option:'shocked'}]}};
 f.docs.set(S.shocked,{type:'effect',toObject:()=>structuredClone(original)});
 const save={id:'save',uuid:'ChatMessage.save',isCheckRoll:true,rolls:[{_evaluated:true,total:10}],speaker:{actor:f.target.id,scene:'s',token:f.target.id},flags:{pf2e:{origin:{uuid:f.power.uuid},context:{type:'saving-throw',outcome:'failure',options:[`${ID}:metapower:channel:channel`]}}}};
 f.game.messages.set(save.id,save);f.docs.set(save.uuid,save);await f.ledger().check(save);await (await f.damage()).finish();
 const [shock]=electricityEffects(f.target,S.shocked),predicate=shock.system.rules[0].predicate;
 // Exercise the relevant PF2e predicate operators against real roll options.
 const matches=(p,options)=>p.every(term=>typeof term==='string'?options.has(term):term.and?matches(term.and,options):term.nor?!term.nor.some(t=>matches([t],options)):false);
 assert.equal(matches(predicate,new Set(['electricity','resistant-shell'])),true);assert.equal(matches(predicate,new Set(['fire','resistant-shell'])),false);
 assert.deepEqual(shock.system.rules[1],original.system.rules[1]);assert.deepEqual(f.docs.get(S.shocked).toObject(),original);
 await f.ledger().channel(f.channel,f.owner);assert.deepEqual(electricityEffects(f.caster,S.charged)[0].system.rules,[{key:'GrantItem',uuid:S.shocked}]);
});

test('native Chain reaction card pays its actual encounter once when another encounter is viewed',async()=>{
 const f=fixture();f.game.modules=new Map();f.power.sourceId=S.chain;f.power.system.actionType={value:'reaction'};f.card.actor=f.caster;f.card.item=f.power;
 const update=async function(changes){for(const [key,value]of Object.entries(changes)){const parts=key.split('.');let object=this;for(const part of parts.slice(0,-1))object=object[part]??={};object[parts.at(-1)]=structuredClone(value);}};
 f.card.update=update;f.combat.turns[0].update=update;f.combat.turns[0].flags={};
 f.game.combat={id:'unrelated',started:true,round:9,turn:0,turns:[]};f.game.combats.set(f.game.combat.id,f.game.combat);
 const budget=createReactionBudget({game:f.game,fromUuid:async id=>f.docs.get(id)}),bounded={...f.game,combat:f.combat};
 assert.equal(genericReactionAvailable(f.caster,bounded),true);assert.equal(await budget.record(f.card,f.owner.id),true);assert.equal(genericReactionAvailable(f.caster,bounded),false);
 assert.equal(await budget.record(f.card,f.owner.id),false);assert.equal(f.combat.turns[0].flags[ID].reactionBudget.entries.length,1);assert.equal(f.card.flags[ID].reactionBudget.epoch,'c:1');
 const duplicate={...f.combat,id:'duplicate'};f.game.combats.set(duplicate.id,duplicate);assert.equal(await budget.record(f.card,f.owner.id),false);
});

test('committed normal use grants native Charged once, preserving GrantItem; Siphon and discharge never gain',async()=>{
 const f=fixture();await f.ledger().channel(f.channel,f.owner);await f.ledger().channel(f.channel,f.owner);
 const [charged]=electricityEffects(f.caster,S.charged);assert.equal(charged.system.badge.value,1);assert.equal(charged.system.rules[0].key,'GrantItem');
 assert.equal(charged.flags[ID].electricityCharge.ownPower,true);
 for(const variant of ['siphon','discharge']){const g=fixture();if(variant==='siphon')g.receipt.snapshot={siphon:{applies:true}};else g.receipt.selection.discharge=true;await g.ledger().channel(g.channel,g.owner);assert.equal(electricityEffects(g.caster,S.charged).length,0);}
 await assert.rejects(f.ledger().channel(f.channel,{id:'impostor'}),/owner/i);
});
test('charge cap stays three and actual electricity reduces once across ledger reload, except own-power shell',async()=>{
 const f=fixture();f.item(f.caster,'charged',S.charged,{flags:{[ID]:{electricityCharge:{ownPower:true}}},system:{badge:{value:3}}});
 await f.ledger().channel(f.channel,f.owner);assert.equal(electricityEffects(f.caster,S.charged)[0].system.badge.value,3);
 const d=await f.damage(f.caster);await d.finish();await d.finish();assert.equal(electricityEffects(f.caster,S.charged)[0].system.badge.value,2);
 f.item(f.caster,'shell',S.shell);const e=await f.damage(f.caster);await e.finish();assert.equal(electricityEffects(f.caster,S.charged)[0].system.badge.value,2);
});
test('immunity zero leaves Shocked; real positive receipt clears it; copied or wrong-author receipts are rejected',async()=>{
 const f=fixture();f.item(f.target,'shock',S.shocked);const zero=await f.damage(f.target,{amount:0});await zero.finish();assert.equal(electricityEffects(f.target,S.shocked).length,1);
 const d=await f.damage();d.receipt.author=f.owner;await assert.rejects(d.finish(),/authentic/i);assert.equal(electricityEffects(f.target,S.shocked).length,1);
 d.receipt.author=f.gm;await d.finish();assert.equal(electricityEffects(f.target,S.shocked).length,0);
});
test('mixed receipt remains unproven until GM attributes exact electric amount on that same receipt',async()=>{
 const f=fixture();f.item(f.target,'shock',S.shocked);const d=await f.damage(f.target,{kind:'mixed',amount:21});const first=await d.finish();assert.equal(first.status,'needs-attribution');assert.equal(first.electricityAmount,null);assert.equal(electricityEffects(f.target,S.shocked).length,1);
 await assert.rejects(f.ledger().confirmMixed({actorUuid:f.target.uuid,nonce:d.payload.nonce,receiptUuid:d.receipt.uuid,amount:8,confirmed:true},f.owner),/GM/);
 await assert.rejects(f.ledger().confirmMixed({actorUuid:f.target.uuid,nonce:d.payload.nonce,receiptUuid:d.receipt.uuid,amount:22,confirmed:true},f.gm),/amount|total/i);
 await f.ledger().confirmMixed({actorUuid:f.target.uuid,nonce:d.payload.nonce,receiptUuid:d.receipt.uuid,amount:0,confirmed:true},f.gm);assert.equal(electricityEffects(f.target,S.shocked).length,1);
});
test('expiry removes only that source effect; ending an encounter never clears outsiders charge',async()=>{
 const f=fixture();f.item(f.target,'soon',S.shocked,{flags:{[ID]:{electricityShock:{expires:{combatId:'c',combatantId:'casterturn',round:2,phase:'end'}}}}});
 f.item(f.target,'later',S.shocked,{flags:{[ID]:{electricityShock:{expires:{combatId:'c',combatantId:'casterturn',round:3,phase:'end'}}}}});
 f.combat.round=2;await f.ledger().expire({combat:f.combat,combatant:f.combat.turns[0],phase:'end',actors:[f.target]});assert.equal(f.target.items.has('later'),true);assert.equal(f.target.items.has('soon'),false);
 f.item(f.outsider,'charge',S.charged,{system:{badge:{value:2}}});f.item(f.caster,'charge',S.charged,{system:{badge:{value:2}}});
 await f.ledger().expire({combat:f.combat,ended:true,actors:[f.caster,f.outsider]});assert.equal(electricityEffects(f.caster,S.charged).length,0);assert.equal(electricityEffects(f.outsider,S.charged)[0].system.badge.value,2);
});

test('start, end and deleted-encounter expiry never writes actors without electricity work',async()=>{
 for(const event of [{phase:'start'},{phase:'end'},{ended:true}]){
  const f=fixture(),actors=[f.caster,f.target,f.outsider],before=actors.map(a=>structuredClone(a.flags));let updates=0;
  for(const actor of actors){const update=actor.update;actor.update=async function(...args){updates++;return update.apply(this,args)};}
  await f.ledger().expire({combat:f.combat,combatant:f.combat.turns[0],actors,...event});
  assert.equal(updates,0,JSON.stringify(event));assert.deepEqual(actors.map(a=>a.flags),before);
 }
});

test('historical electricity records and another encounter pending Shock do not cause expiry writes',async()=>{
 const f=fixture(),actor=f.target;actor.flags[ID]={electricity:{version:1,damage:{old:{status:'confirmed'}},operations:{'encounter:c:charge':{status:'done'},'encounter:other:charge':{status:'started',before:0,after:0,gain:false,itemId:null}},pendingShocks:{later:{expires:{combatId:'other',combatantId:'someone',round:2,phase:'end'}}}}};
 const before=structuredClone(actor.flags);let updates=0;const update=actor.update;actor.update=async function(...args){updates++;return update.apply(this,args)};
 for(const event of [{phase:'start'},{phase:'end'},{ended:true}])await f.ledger().expire({combat:f.combat,combatant:f.combat.turns[0],actors:[actor],...event});
 assert.equal(updates,0);assert.deepEqual(actor.flags,before);
});

test('encounter expiry still removes non-Eldamon participant and outsider Shocks without empty charge operations',async()=>{
 for(const participant of [true,false]){
 const f=fixture(),actor=participant?f.target:f.outsider,expires={combatId:f.combat.id,combatantId:'casterturn',round:2,phase:'end'};
 f.item(actor,'hostileShock',S.shocked,{flags:{[ID]:{electricityShock:{expires}}}});
 actor.flags[ID]={electricity:{version:1,damage:{},operations:{},pendingShocks:{pending:{expires}}}};useFoundryFlagUpdates(actor);
 await f.ledger().expire({combat:f.combat,ended:true,actors:[actor]});
 assert.equal(actor.items.has('hostileShock'),false);assert.deepEqual(electricityState(actor).pendingShocks,{});
 assert.equal(electricityState(actor).operations[`encounter:${f.combat.id}:charge`],undefined);
 }
});

test('encounter charge cleanup resumes its exact unfinished operation without inventing work for others',async()=>{
 const f=fixture(),actor=f.target,item=f.item(actor,'charge',S.charged,{system:{badge:{value:2}}}),update=item.update;let fail=true;
 item.update=async function(...args){if(fail){fail=false;throw Error('interrupted cleanup');}return update.apply(this,args)};
 const payload={combat:f.combat,ended:true,actors:[actor]},key=`encounter:${f.combat.id}:charge`;
 await assert.rejects(f.ledger().expire(payload),/interrupted cleanup/);assert.equal(electricityState(actor).operations[key].status,'started');
 await f.ledger().expire(payload);assert.equal(electricityEffects(actor,S.charged).length,0);assert.equal(electricityState(actor).operations[key].status,'done');
 let writes=0;actor.update=async()=>{writes++};await f.ledger().expire(payload);assert.equal(writes,0);
 // A historical empty cleanup interrupted before its final receipt still has
 // exact, safe work to finish, even though no Charged item remains.
 actor.flags[ID].electricity.operations[key]={status:'started',itemId:null,before:0,after:0,gain:false};useFoundryFlagUpdates(actor);
 await f.ledger().expire(payload);assert.equal(electricityState(actor).operations[key].status,'done');
 actor.flags[ID].electricity.operations[key]={status:'started',itemId:'charge',before:2,after:0,gain:false};
 await assert.rejects(f.ledger().expire(payload),/Interrupted Charged mutation/);
});

test('expiry consuming the final charge does not start an empty encounter cleanup afterwards',async()=>{
 const f=fixture(),actor=f.target;f.item(actor,'charge',S.charged,{system:{badge:{value:1}}});
 f.item(actor,'shock',S.shocked,{flags:{[ID]:{electricityShock:{expires:{combatId:f.combat.id,combatantId:'casterturn',round:2,phase:'end'}}}}});
 await f.ledger().expire({combat:f.combat,ended:true,actors:[actor]});
 assert.equal(electricityEffects(actor,S.charged).length,0);assert.equal(electricityEffects(actor,S.shocked).length,0);
 assert.equal(electricityState(actor).operations[`encounter:${f.combat.id}:charge`],undefined);
});
test('same effect two targets cannot chain to one another even between application receipts',async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);const m=f.source('pure',[f.tokens[1].uuid,f.tokens[2].uuid]);const d=await f.damage(f.target,{m});await d.finish();
 const candidates=await f.ledger().candidates({actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'},f.owner);assert.deepEqual(candidates,[]);
 m.flags[ID].electricitySource.targetUuids=[f.tokens[1].uuid];
 const changed=await f.ledger().candidates({actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'},f.owner);assert.deepEqual(changed,[]);
});
test('Anvil failure adds original Shocked after damage, expires at source next end, and does not affect allies or Siphon',async()=>{
 const f=fixture();f.power.sourceId=f.receipt.sourceUuid=S.anvil;
 const save={id:'save',uuid:'ChatMessage.save',isCheckRoll:true,rolls:[{_evaluated:true,total:10}],speaker:{actor:f.target.id,scene:'s',token:f.target.id},flags:{pf2e:{origin:{uuid:f.power.uuid},context:{type:'saving-throw',outcome:'failure',options:[`${ID}:metapower:channel:channel`]}}}};
 f.game.messages.set(save.id,save);f.docs.set(save.uuid,save);
 await f.ledger().check(save);assert.equal(electricityEffects(f.target,S.shocked).length,0);
 const d=await f.damage();await d.finish();let [shock]=electricityEffects(f.target,S.shocked);assert.ok(shock);assert.equal(shock.system.rules[0].selector[0],'fortitude');
 f.combat.round=2;await f.ledger().expire({combat:f.combat,combatant:f.combat.turns[1],phase:'end',actors:[f.target]});assert.equal(electricityEffects(f.target,S.shocked).length,1);
 await f.ledger().expire({combat:f.combat,combatant:f.combat.turns[0],phase:'start',actors:[f.target]});assert.equal(electricityEffects(f.target,S.shocked).length,1);
 await f.ledger().expire({combat:f.combat,combatant:f.combat.turns[0],phase:'end',actors:[f.target]});assert.equal(electricityEffects(f.target,S.shocked).length,0);
 f.receipt.snapshot={siphon:{applies:true}};await f.ledger().check(save);assert.deepEqual(electricityState(f.target).pendingShocks,{});
 f.receipt.snapshot={kind:'normal'};f.target.alliance=f.caster.alliance;await f.ledger().check(save);assert.deepEqual(electricityState(f.target).pendingShocks,{});
});
test('Static Shock failure gets timed Shocked but critical failure does not, and source selection cannot be replaced',async()=>{
 const f=fixture();f.power.sourceId=f.receipt.sourceUuid=S.static;
 const attack={id:'attack',uuid:'ChatMessage.attack',isCheckRoll:true,rolls:[{_evaluated:true,total:10}],speaker:{actor:f.caster.id,scene:'s',token:f.caster.id},flags:{pf2e:{origin:{uuid:f.power.uuid},context:{type:'attack-roll',outcome:'failure',target:{token:f.tokens[1].uuid},options:[`${ID}:metapower:channel:channel`]}}}};
 f.game.messages.set(attack.id,attack);f.docs.set(attack.uuid,attack);await f.ledger().check(attack);assert.equal(Object.keys(electricityState(f.target).pendingShocks).length,1);
 attack.flags.pf2e.context.outcome='criticalFailure';await f.ledger().check(attack);assert.equal(Object.keys(electricityState(f.target).pendingShocks).length,0);
 attack.flags.pf2e.context.outcome='success';attack.flags.pf2e.context.target.token=f.tokens[2].uuid;await f.ledger().check(attack);assert.equal(Object.keys(electricityState(f.other).pendingShocks).length,0);
});
test('legal chain selection uses exact actual receipt amount and rejects stale frame, forged amount, unavailable targets',async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);const d=await f.damage();await d.finish();const chain=f.item(f.caster,'chain',S.chain,{type:'feat'});
 const [candidate]=await f.ledger().candidates({actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'},f.owner);assert.equal(candidate.amount,13);
 const context={actor:f.caster,item:chain,kind:'normal',user:f.owner,selection:{discharge:false,targetUuids:[f.tokens[2].uuid],triggerDamage:13,electricityEvidence:candidate.evidence}};
 await f.ledger().validateSelection(context);
 await assert.rejects(f.ledger().validateSelection({...context,selection:{...context.selection,triggerDamage:21}}),/source-bound/i);
 f.combat.turn=1;await assert.rejects(f.ledger().validateSelection(context),/source-bound/i);
});
test('out-of-encounter Refresh removes charge but in-encounter Refresh preserves it',async()=>{
 const f=fixture();f.item(f.caster,'charge',S.charged,{system:{badge:{value:2}}});await f.ledger().refresh({actor:f.caster,nonce:'refresh1'});assert.equal(electricityEffects(f.caster,S.charged)[0].system.badge.value,2);
 f.combat.started=false;await f.ledger().refresh({actor:f.caster,nonce:'refresh2'});assert.equal(electricityEffects(f.caster,S.charged).length,0);
});
test('a post-damage lifecycle interruption is recoverable without applying damage or reducing charge twice',async()=>{
 const f=fixture();f.item(f.target,'charge',S.charged,{system:{badge:{value:2}}});const item=f.target.items.get('charge'),native=item.update;let fail=true;
 item.update=async function(data){if(fail){fail=false;throw Error('disconnected before item mutation');}return native.call(this,data);};
 const d=await f.damage();await assert.rejects(d.finish(),/disconnected/);assert.equal(item.system.badge.value,2);
 await d.finish();assert.equal(item.system.badge.value,1);await d.finish();assert.equal(item.system.badge.value,1);
});
test('damage evidence and chain eligibility follow the actual encounter when another encounter is viewed',async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);const unrelated={id:'unrelated',started:true,round:9,turn:0,turns:[{id:'outside-turn',actor:f.outsider,token:f.tokens[3]}]};unrelated.combatants=unrelated.turns;f.game.combats.set(unrelated.id,unrelated);f.game.combat=unrelated;
 const d=await f.damage();await d.finish();assert.equal(electricityState(f.target).damage[d.payload.nonce].frame,'c:1:0');
 const payload={actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'};
 assert.equal((await f.ledger().candidates(payload,f.owner)).length,1);f.game.combat=f.combat;assert.equal((await f.ledger().candidates(payload,f.owner)).length,1);
 f.combat.turn=1;assert.equal((await f.ledger().candidates(payload,f.owner)).length,0);
});
test('Anvil expiry binds the exact source encounter and does not borrow an unrelated viewed encounter',async()=>{
 const f=fixture();f.power.sourceId=f.receipt.sourceUuid=S.anvil;const unrelated={id:'unrelated',started:true,round:9,turn:0,turns:[]};unrelated.combatants=[];f.game.combats.set(unrelated.id,unrelated);f.game.combat=unrelated;
 const save={id:'save',uuid:'ChatMessage.save',isCheckRoll:true,rolls:[{_evaluated:true,total:10}],speaker:{actor:f.target.id,scene:'s',token:f.target.id},flags:{pf2e:{origin:{uuid:f.power.uuid},context:{type:'saving-throw',outcome:'failure',options:[`${ID}:metapower:channel:channel`]}}}};
 f.game.messages.set(save.id,save);f.docs.set(save.uuid,save);await f.ledger().check(save);
 const pending=Object.values(electricityState(f.target).pendingShocks);assert.equal(pending.length,1);assert.deepEqual(pending[0].expires,{combatId:'c',combatantId:'casterturn',round:2,phase:'end'});
});
test('outside or ambiguous encounter membership cannot borrow the viewed encounter for a chain trigger',async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);const d=await f.damage(f.outsider);await d.finish();assert.equal(electricityState(f.outsider).damage[d.payload.nonce].frame,null);
 const payload={actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'};
 assert.equal((await f.ledger().candidates(payload,f.owner)).length,0);
 const duplicate={...f.combat,id:'duplicate'};f.game.combats.set(duplicate.id,duplicate);const e=await f.damage();await e.finish();assert.equal((await f.ledger().candidates(payload,f.owner)).length,0);
});
test('native reaction availability is queried against the actual source encounter facade',async()=>{
 const f=fixture();f.item(f.other,'shock',S.shocked);const d=await f.damage();await d.finish();f.game.combat={id:'viewed-other',started:true,round:99,turn:0,turns:[]};let observed;
 const ledger=createElectricityLedger({game:f.game,fromUuid:async id=>f.docs.get(id),reactionAvailable:(actor,bounded)=>{observed=bounded.combat;assert.equal(actor,f.caster);return false;}});
 assert.deepEqual(await ledger.candidates({actorUuid:f.caster.uuid,sourceTokenUuid:f.tokens[0].uuid,selection:{targetUuids:[f.tokens[2].uuid],discharge:false},kind:'normal'},f.owner),[]);assert.equal(observed,f.combat);
});
export {fixture};
