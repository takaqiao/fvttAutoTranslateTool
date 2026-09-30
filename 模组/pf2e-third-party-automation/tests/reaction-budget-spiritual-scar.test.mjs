import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture as base} from './glimpse-fixture.mjs';
import {SPIRITUAL_SCAR_SOURCE} from '../scripts/spiritual-scar-native.mjs';
import {createReactionBudget,genericReactionAvailable} from '../scripts/reaction-budget.mjs';
const M='pf2e-third-party-automation';
const put=(target,changes)=>{for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let at=target;for(const p of parts.slice(0,-1))at=at[p]??={};at[parts.at(-1)]=structuredClone(value)}};
function fixture(){
 const f=base(),actor=f.champion,item=f.ability;item.sourceId=SPIRITUAL_SCAR_SOURCE;item.system.slug='spiritual-scar';actor.system.resources={reactions:{max:2}};
 const nonce='scar-one',paymentNonce='payment-one',combatant=f.combat.turns[1],epoch='combat:1',privacy={blind:true,whisper:['u']};
 combatant.uuid='Combat.combat.Combatant.champion';combatant.update=async ch=>{put(combatant,ch);return combatant};
 const claim={nonce,claimKey:'scar:'+nonce,status:'reserved',actorUuid:actor.uuid,tokenUuid:f.championToken.uuid,combatId:f.combat.id,combatantId:combatant.id,itemUuid:item.uuid,userId:f.user.id,epoch};
 combatant.flags[M]={spiritualScarClaims:[claim],reactionBudget:{epoch,entries:[{type:'reaction',cost:1,slug:'spiritual-scar',claimKey:claim.claimKey}]}};
 item.flags={[M]:{spiritualScarUse:{currentNonce:nonce,operations:{[nonce]:{nonce,status:'paid',actorUuid:actor.uuid,itemUuid:item.uuid,userId:f.user.id,paymentNonce,privacy}}}}};
 const message={id:'scar-card',uuid:'ChatMessage.scar-card',author:f.user,actor,speaker:{actor:actor.id,scene:f.scene.id,token:f.championToken.id},rolls:[],blind:true,whisper:['u'],flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid,type:'action'}},[M]:{usageInput:{actualUse:true,frequencyReceiptId:'receipt-one'},spiritualScarInput:{nonce,paymentNonce}}},async update(ch){put(this,ch);return this}};
 f.game.messages.set(message.id,message);f.docs.set(message.uuid,message);const budget=createReactionBudget({game:f.game,fromUuid:f.fromUuid}),bounded={...f.game,combat:f.combat};
 return {...f,actor,item,nonce,paymentNonce,combatant,claim,message,budget,bounded,entries:()=>combatant.flags[M].reactionBudget.entries};
}
test('Scar reservation and exact original paid Use card are one generic reaction before messageId binding',async()=>{
 const f=fixture();assert.equal(genericReactionAvailable(f.actor,f.bounded),true);assert.equal(await f.budget.record(f.message,f.user.id),true);assert.equal(f.entries().length,1);assert.equal(f.entries()[0].msgId,f.message.id);assert.equal(f.entries()[0].claimKey,f.claim.claimKey);assert.equal(genericReactionAvailable(f.actor,f.bounded),true);assert.equal(await f.budget.record(f.message,f.user.id),false);
});
test('durable Scar claim consumes its slot even if another module replaced the generic log',()=>{
 const f=fixture();f.actor.system.resources.reactions.max=1;f.combatant.flags[M].reactionBudget.entries=[];assert.equal(genericReactionAvailable(f.actor,f.bounded),false);
});
test('an old paid card delivered next turn cannot spend the newly refreshed reaction',async()=>{
 const f=fixture();f.actor.system.resources.reactions.max=1;f.combat.round=3;f.combat.turn=1;f.combatant.flags[M].reactionBudget={epoch:'combat:3',entries:[]};assert.equal(await f.budget.record(f.message,f.user.id),true);assert.equal(f.entries().length,0);assert.equal(f.message.flags[M].reactionBudget.epoch,'combat:1');assert.equal(genericReactionAvailable(f.actor,f.bounded),true);assert.equal(await f.budget.record(f.message,f.user.id),false);
});
for(const[name,change]of[
 ['manual unbound card',f=>delete f.message.flags[M].spiritualScarInput],['wrong payment',f=>f.message.flags[M].spiritualScarInput.paymentNonce='forged'],['unpaid daily claim',f=>f.item.flags[M].spiritualScarUse.operations[f.nonce].status='paying'],['different reaction action',f=>f.item.sourceId='other'],['different token',f=>f.claim.tokenUuid=f.allyToken.uuid],['different combatant',f=>f.claim.combatantId='other'],['unbound invocation key',f=>f.claim.claimKey='other'],['unrelated private recipients',f=>f.message.whisper=[]],['already-bound other card',f=>f.claim.messageId='earlier'],['different daily owner',f=>f.item.flags[M].spiritualScarUse.operations[f.nonce].userId='other']
])test(`Scar merging does not hide an ordinary reaction for ${name}`,async()=>{const f=fixture();change(f);await f.budget.record(f.message,f.user.id);assert.equal(f.entries().length,2);assert.equal(f.entries()[1].claimKey,undefined)});
test('a refunded claim is not an additional paid reaction',()=>{const f=fixture();f.actor.system.resources.reactions.max=1;f.combatant.flags[M].reactionBudget.entries=[];f.claim.status='refunded';assert.equal(genericReactionAvailable(f.actor,f.bounded),true)});
