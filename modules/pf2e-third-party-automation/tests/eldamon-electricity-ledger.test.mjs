import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createElectricityLedger,ELECTRICITY_SOURCES as S,electricityState,electricityEffects} from '../scripts/eldamon-electricity.mjs';
const ID='pf2e-third-party-automation';
import {fixture} from './eldamon-electricity-fixture.mjs';

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
