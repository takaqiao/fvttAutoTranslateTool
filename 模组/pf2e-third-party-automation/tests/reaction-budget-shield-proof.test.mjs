import test from 'node:test';
import assert from 'node:assert/strict';
import {createReactionBudget,classifyNativeShieldBlock,genericReactionAvailable} from '../scripts/reaction-budget.mjs';
import {markUnappliedDamageError} from '../scripts/native-context.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';

function patch(document,changes){
 for(const [path,value]of Object.entries(changes)){
  const keys=path.split('.');let at=document;
  for(const key of keys.slice(0,-1))at=at[key]??={};
  at[keys.at(-1)]=structuredClone(value);
 }
}
function fixture(){
 const gm={id:'gm',isGM:true},owner={id:'owner',isGM:false},users=new Map([[gm.id,gm],[owner.id,owner]]);users.activeGM=gm;
 const actor={id:'defender',uuid:'Actor.defender',flags:{},items:[],hitPoints:{value:50},attributes:{shield:{itemId:'shield',raised:true,broken:false,destroyed:false}},system:{resources:{reactions:{max:1}}},testUserPermission:u=>u===gm||u===owner};
 const token={id:'defender',uuid:'Scene.scene.Token.defender',documentName:'Token',name:'Defender',actor};
 const combatant={id:'combatant',actor,token,flags:{},async update(changes){patch(this,changes)}};
 const combat={id:'combat',started:true,round:2,turn:0,turns:[combatant]};
 const game={user:gm,users,modules:new Map(),messages:new Map(),combat,combats:new Map([[combat.id,combat]]),i18n:{localize:key=>key.endsWith('TakesNoDamage')?'{actor} takes no damage.':key}};
 const rpc=new Map(),hooks=new Map(),errors=[];let releases=0,onRelease=()=>{};
 const resources={snapshot:async()=>({}),available:()=>true,reserve:()=>({proof:{kind:'fixture'},changes:{'flags.testResource.spent':true}}),release:async()=>{releases++;await onRelease();return {'flags.testResource.spent':false}}};
 const fromUuid=async uuid=>uuid===actor.uuid?actor:uuid===token.uuid?token:null;
 const budget=createReactionBudget({game,fromUuid,reactionResources:resources,onError:e=>errors.push(e)});
 budget.register({Hooks:{on:(name,fn)=>hooks.set(name,fn),off(){}},socket:{register:(name,fn)=>rpc.set(name,fn)}});
 const payload={nonce:'original-block-nonce',actorUuid:actor.uuid,tokenUuid:token.uuid,shieldId:'shield',combatId:combat.id,combatantId:combatant.id,epoch:'combat:2'};
 const send=(method,body=payload,user=owner)=>rpc.get('reaction-budget:shield-'+method).call({socketdata:{userId:user.id}},body);
 const entries=()=>combatant.flags[M]?.reactionBudget?.entries??[];
 function card(kind='block',user=owner,nonce=payload.nonce){
  const source={_id:'native-result',author:user.id,speaker:{actor:actor.id,scene:'scene',token:token.id},content:kind==='nonblock'?'<section class="damage-taken"><span class="statements">Defender takes no damage.</span></section>':'native persisted result',flags:{pf2e:{context:{type:'damage-taken',options:[M+':native-shield:'+nonce]},appliedDamage:{uuid:actor.uuid,isHealing:false,shield:kind==='block'?{id:'shield',damage:4}:null,persistent:[],updates:[]}}}};
  const message={id:source._id,actor,author:user,speaker:source.speaker,content:source.content,flags:source.flags,toObject:()=>structuredClone({...source,content:message.content,flags:message.flags,speaker:message.speaker})};
  game.messages.set(message.id,message);return message;
 }
 const finish=(message,blocked,extra={})=>send('finish',{...payload,messageId:message.id,content:message.content,blocked,...extra});
 return {gm,owner,game,actor,token,combatant,budget,hooks,errors,payload,send,entries,card,finish,releaseCount:()=>releases,onRelease:fn=>onRelease=fn,fromUuid,resources,rpc};
}
function pending(f){
 assert.equal(f.entries().length,1);assert.equal(f.entries()[0].shield.state,'pending');
 assert.equal(f.combatant.flags.testResource.spent,true);assert.equal(genericReactionAvailable(f.actor,f.game),false);
}

for(const [kind,reported]of [['block',false],['nonblock',true],['unknown',false],['unknown',true]])test(`GM rejects owner ${reported} for persisted ${kind} classification`,async()=>{
 const f=fixture();assert.equal((await f.send('begin')).ok,true);const card=f.card(kind);
 assert.equal(classifyNativeShieldBlock(card,{token:f.token,shieldId:'shield',nativeBlockNonce:f.payload.nonce,game:f.game}),kind==='unknown'?null:kind==='block');
 const result=await f.finish(card,reported);assert.equal(result.ok,false);pending(f);assert.equal(f.releaseCount(),0);
});
for(const kind of ['block','nonblock'])test(`GM accepts exact saved native ${kind} and matching owner result`,async()=>{
 const f=fixture();await f.send('begin');const result=await f.finish(f.card(kind),kind==='block');assert.equal(result.ok,true);
 assert.equal(f.releaseCount(),kind==='block'?0:1);assert.equal(f.entries().length,kind==='block'?1:0);
 if(kind==='block'){assert.equal(f.entries()[0].shield.state,'used');assert.equal(f.combatant.flags.testResource.spent,true)}
 else{assert.equal(f.combatant.flags.testResource.spent,false);assert.equal(genericReactionAvailable(f.actor,f.game),true)}
});
test('used native block accepts the same proof again but rejects a contradictory refund',async()=>{
 const f=fixture();await f.send('begin');const card=f.card();assert.equal((await f.finish(card,true)).ok,true);
 assert.equal((await f.finish(card,true)).ok,true);assert.equal((await f.finish(card,false)).ok,false);
 assert.equal(f.entries()[0].shield.state,'used');assert.equal(f.releaseCount(),0);
});
test('socket enteredNative=false cannot release a reservation without a card',async()=>{
 const f=fixture();await f.send('begin');assert.equal((await f.send('finish',{...f.payload,enteredNative:false})).ok,false);pending(f);assert.equal(f.releaseCount(),0);
});
test('socket enteredNative=false cannot override an exact native block',async()=>{
 const f=fixture();await f.send('begin');assert.equal((await f.finish(f.card(),false,{enteredNative:false})).ok,false);pending(f);assert.equal(f.releaseCount(),0);
});
for(const changed of ['author','speaker','marker','content','shield','shieldId'])test(`changed ${changed} cannot prove a native refund`,async()=>{
 const f=fixture();await f.send('begin');const card=f.card('nonblock'),body={...f.payload,messageId:card.id,content:card.content,blocked:false};
 if(changed==='author')card.author=f.gm;
 if(changed==='speaker')card.speaker.token='other';
 if(changed==='marker')card.flags.pf2e.context.options=[];
 if(changed==='content')body.content='different content';
 if(changed==='shield')card.flags.pf2e.appliedDamage.shield={id:'different-shield',damage:4};
 if(changed==='shieldId')body.shieldId='different-shield';
 assert.equal((await f.send('finish',body)).ok,false);pending(f);assert.equal(f.releaseCount(),0);
});
for(const changed of ['source','unrelated-source','instance'])test(`native card ${changed} change during awaited release keeps pending payment`,async()=>{
 const f=fixture();await f.send('begin');const card=f.card('nonblock');f.onRelease(()=>{
  if(changed==='source')card.flags.pf2e.appliedDamage.shield={id:'shield',damage:4};
  else if(changed==='unrelated-source')card.flags.another={changed:true};
  else f.game.messages.set(card.id,{...card});
 });
 assert.equal((await f.finish(card,false)).ok,false);pending(f);
});
test('local branded pre-native failure releases only the original private scope',async()=>{
 const f=fixture();await assert.rejects(f.budget.applyDamage(f.actor,{damage:12,shieldBlockRequest:true,token:f.token},async()=>{throw markUnappliedDamageError(Error('proved before native'))}),/proved before native/);
 assert.equal(f.entries().length,0);assert.equal(f.releaseCount(),1);assert.equal(f.combatant.flags.testResource.spent,false);assert.deepEqual(f.errors,[]);
});
test('local unbranded uncertainty keeps the original payment',async()=>{
 const f=fixture();await assert.rejects(f.budget.applyDamage(f.actor,{damage:12,shieldBlockRequest:true,token:f.token},async()=>{throw Error('unknown native result')}),/unknown native result/);
 pending(f);assert.equal(f.releaseCount(),0);
});
test('local branded failure cannot refund a changed original token binding',async()=>{
 const f=fixture();await assert.rejects(f.budget.applyDamage(f.actor,{damage:12,shieldBlockRequest:true,token:f.token},async()=>{f.token.actor={uuid:'Actor.other'};throw markUnappliedDamageError(Error('before native, token changed'))}),/before native, token changed/);
 pending(f);assert.equal(f.releaseCount(),0);assert.equal(f.errors.length,1);
});
test('remote branded no-card failure remains pending because the GM has no private proof',async()=>{
 const f=fixture(),clientGame={...f.game,user:f.owner},client=createReactionBudget({game:clientGame,fromUuid:f.fromUuid,reactionResources:f.resources,onError:e=>f.errors.push(e)});
 client.register({Hooks:{on:()=>1,off(){}},socket:{register(){},executeAsUser:async(name,_gm,payload)=>f.rpc.get(name).call({socketdata:{userId:f.owner.id}},payload)}});
 await assert.rejects(client.applyDamage(f.actor,{damage:12,shieldBlockRequest:true,token:f.token},async()=>{throw markUnappliedDamageError(Error('remote before native'))}),/remote before native/);
 pending(f);assert.equal(f.releaseCount(),0);assert.equal(f.errors.length,1);
});
