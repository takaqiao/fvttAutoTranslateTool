import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createReactionBudget,genericReactionAvailable} from '../scripts/reaction-budget.mjs';
import {markUnappliedDamageError} from '../scripts/native-context.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';
const resources=await import('../scripts/shield-reaction-resources.mjs').catch(()=>({}));
const reactionHash='4a81322796ce1c6ed545edc09e1aa3a96a9c8a96dfd034403bf657068ed7036c';

function fixture({prior=null,reaction=false,state=true,viewOther=false,ambiguous=false,version='1.4.3'}={}){
 const user={id:'gm',isGM:true},hooks=new Map(),errors=[];
 const actor={id:'defender',uuid:'Actor.defender',flags:{},items:[],system:{resources:{reactions:{max:1}}},hitPoints:{value:50},attributes:{shield:{itemId:'shield',raised:true,broken:false,destroyed:false}},testUserPermission:u=>u===user};
 const token={id:'defender',uuid:'Scene.scene.Token.defender',documentName:'Token',actor};
 const combatant={id:'defender',actor,token,flags:{'pf2e-reaction':{state},[M]:{reactionBudget:{epoch:'encounter:2',entries:prior?[{type:'reaction',cost:1,slug:prior,msgId:'prepaid'}]:[]}}},getFlag(m,k){return this.flags[m]?.[k]},
  async update(changes){for(const [path,value]of Object.entries(changes)){let parent=this;const keys=path.split('.');for(const key of keys.slice(0,-1))parent=parent[key]??={};parent[keys.at(-1)]=structuredClone(value);}}
 };
 const encounter={id:'encounter',started:true,round:2,turn:0,turns:[combatant]},other={id:'other',started:true,round:7,turn:0,turns:ambiguous?[{...combatant,id:'duplicate'}]:[]};
 const game={user,users:{activeGM:user,get:()=>user},world:{id:'ujx5r8oipw7ercdr'},modules:new Map([['pf2e-reaction',{active:reaction,version}],['pf2e-auto-action-tracker',{active:false}]]),messages:new Map(),combats:new Map([[encounter.id,encounter],[other.id,other]]),combat:viewOther?other:encounter};
 const item={id:'block',uuid:'Actor.defender.Item.block',actor,type:'feat',sourceId:'Compendium.pf2e.feats-srd.Item.jM72TjJ965jocBV8',system:{actionType:{value:'reaction'},slug:'shield-block'}};
 const prepaid={id:'prepaid',actor,item,author:user,speaker:{actor:actor.id,scene:'scene',token:token.id},rolls:[],flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid,type:'feat'}},[M]:{reactionBudget:{epoch:'encounter:2',actorUuid:actor.uuid,combatantId:combatant.id}}}};if(prior==='shield-block')game.messages.set(prepaid.id,prepaid);
 let reactionResources;
 if(reaction&&version==='1.4.3'){
  assert.equal(typeof resources.createShieldReactionResources,'function');
  reactionResources=resources.createShieldReactionResources({game,fetchSource:async()=>'audited fixture',hashSource:async()=>reactionHash});
 }
 const budget=createReactionBudget({game,fromUuid:async uuid=>uuid===actor.uuid?actor:uuid===token.uuid?token:uuid===item.uuid?item:null,reactionResources,onError:e=>errors.push(e)});
 budget.register({Hooks:{on:(name,fn)=>hooks.set(name,fn),off(){}},socket:{register(){}}});
 const params={damage:12,shieldBlockRequest:true,token,rollOptions:new Set()};let nativeCalls=0;
 const native=async(p,{blocked=true}={})=>{
  nativeCalls++;const message={id:'native-block',actor,author:user,speaker:{actor:actor.id,scene:'scene',token:token.id},content:blocked?'native block':'<section class="damage-taken"><span class="statements">Defender takes no damage.</span></section>',flags:{pf2e:{context:{type:'damage-taken',options:[...p.rollOptions??[]]},...(blocked?{appliedDamage:{shield:{id:'shield',damage:4}}}:{})}}};
  game.messages.set(message.id,message);hooks.get('createChatMessage')(message,{},user.id);return actor;
 };
 token.name='Defender';game.i18n={localize:key=>key.endsWith('TakesNoDamage')?'{actor} takes no damage.':key};
 return {game,actor,token,combatant,encounter,other,item,prepaid,budget,params,native,errors,count:()=>nativeCalls,entries:()=>combatant.flags[M].reactionBudget.entries};
}

test('a spent ordinary reaction prevents another native Shield Block before damage',async()=>{
 const f=fixture({prior:'glimpse-of-redemption'});assert.equal(genericReactionAvailable(f.actor,f.game),false);
 await assert.rejects(f.budget.applyDamage(f.actor,f.params,f.native),/反应/);
 assert.equal(f.count(),0);assert.equal(f.entries().length,1);
});

test('native Shield Block binds the actor and token encounter even while another is viewed',async()=>{
 const f=fixture({viewOther:true});await f.budget.applyDamage(f.actor,f.params,f.native);
 assert.equal(f.count(),1);assert.equal(f.entries().length,1);assert.equal(f.entries()[0].shield.state,'used');assert.equal(f.entries()[0].shield.combatId,'encounter');
 assert.equal(f.errors.length,0);assert.equal(f.game.combat,f.other);
});

test('ambiguous actual encounters stop before native damage instead of guessing the viewed one',async()=>{
 const f=fixture({ambiguous:true});await assert.rejects(f.budget.applyDamage(f.actor,f.params,f.native),/遭遇/);assert.equal(f.count(),0);assert.equal(f.entries().length,0);
});

test('direct native block reserves the audited Reaction resource and pays once',async()=>{
 const f=fixture({reaction:true});await f.budget.applyDamage(f.actor,f.params,async p=>{assert.equal(f.combatant.flags['pf2e-reaction'].state,false);return f.native(p)});
 assert.equal(f.count(),1);assert.equal(f.entries().length,1);assert.equal(f.entries()[0].shield.state,'used');assert.equal(f.errors.length,0);
 await assert.rejects(f.budget.applyDamage(f.actor,f.params,f.native),/反应/);assert.equal(f.count(),1);
});

test('Reaction spent state prevents a block even when its native reaction card has not arrived',async()=>{
 const f=fixture({reaction:true,state:false});await assert.rejects(f.budget.applyDamage(f.actor,f.params,f.native),/反应/);assert.equal(f.count(),0);assert.equal(f.entries().length,0);
});

test('an exact same-epoch Shield Block card is reused without charging a second reaction',async()=>{
 const f=fixture({reaction:true,state:false,prior:'shield-block'});await f.budget.applyDamage(f.actor,f.params,f.native);
 assert.equal(f.count(),1);assert.equal(f.entries().length,1);assert.equal(f.entries()[0].msgId,'prepaid');assert.equal(f.entries()[0].shield.state,'used');assert.equal(f.combatant.flags['pf2e-reaction'].state,false);
});

test('a same-name foreign card is not treated as Shield Block prepayment',async()=>{
 const f=fixture({prior:'shield-block'});f.item.sourceId='Compendium.other.feats.Item.fake';
 await assert.rejects(f.budget.applyDamage(f.actor,f.params,f.native),/反应|格挡/);assert.equal(f.count(),0);
});

test('Reaction disabled uses the existing ledger; an active unknown version fails explicitly',async()=>{
 const off=fixture();await off.budget.applyDamage(off.actor,off.params,off.native);assert.equal(off.entries()[0].shield.state,'used');assert.equal(off.combatant.flags['pf2e-reaction'].state,true);
 const unknown=fixture({reaction:true,version:'future'});await assert.rejects(unknown.budget.applyDamage(unknown.actor,unknown.params,unknown.native),/Reaction|版本|手工/);assert.equal(unknown.count(),0);assert.equal(unknown.entries().length,0);
});

test('a proven non-block or pre-native failure returns only its own Reaction reservation',async()=>{
 for(const kind of ['not-blocked','never-entered']){
  const f=fixture({reaction:true});const execute=()=>f.budget.applyDamage(f.actor,f.params,p=>kind==='not-blocked'?f.native(p,{blocked:false}):Promise.reject(markUnappliedDamageError(Error('never entered'))));
  if(kind==='never-entered')await assert.rejects(execute(),/never entered/);else await execute();
  assert.equal(f.combatant.flags['pf2e-reaction'].state,true);assert.equal(f.entries().length,0);assert.equal(f.errors.length,0);
 }
});

test('uncertain native failure keeps its resource reservation spent',async()=>{
 const f=fixture({reaction:true});await assert.rejects(f.budget.applyDamage(f.actor,f.params,async()=>{throw Error('native result unknown')}),/native result unknown/);
 assert.equal(f.combatant.flags['pf2e-reaction'].state,false);assert.equal(f.entries().length,1);assert.equal(f.entries()[0].shield.state,'pending');
});

test('Quick Shield Block uses its dedicated slot after the ordinary reaction is spent',async()=>{
 const f=fixture({reaction:true,state:false,prior:'glimpse-of-redemption'});
 f.actor.items.push({sourceId:'Compendium.pf2e.feats-srd.Item.pRqcm5P2ZFihSpVI',system:{slug:'quick-shield-block'}});f.combatant.flags['pf2e-reaction']['quick-shield-block']=1;
 await f.budget.applyDamage(f.actor,f.params,f.native);
 assert.equal(f.combatant.flags['pf2e-reaction'].state,false);assert.equal(f.combatant.flags['pf2e-reaction']['quick-shield-block'],0);assert.equal(f.entries()[1].shield.resourceSlot,'quick-shield-block');
 await assert.rejects(f.budget.applyDamage(f.actor,f.params,f.native),/反应/);assert.equal(f.count(),1);
});

test('a changed prepaid card is revalidated after asynchronous Reaction source verification',async()=>{
 const f=fixture({prior:'shield-block',reaction:true,state:false});
 const adapter=resources.createShieldReactionResources({game:f.game,fetchSource:async()=>{f.prepaid.flags[M].reactionBudget.epoch='old:1';return 'bundle'},hashSource:async()=>reactionHash});
 const budget=createReactionBudget({game:f.game,fromUuid:async uuid=>uuid===f.actor.uuid?f.actor:uuid===f.token.uuid?f.token:null,reactionResources:adapter});
 await assert.rejects(budget.applyDamage(f.actor,f.params,f.native),/预付|格挡|反应/);assert.equal(f.count(),0);
});

test('unknown active Reaction bundle and uninitialized resources fail before native damage',async()=>{
 for(const kind of ['hash','state']){
  const f=fixture({reaction:true});if(kind==='state')delete f.combatant.flags['pf2e-reaction'].state;
  const adapter=resources.createShieldReactionResources({game:f.game,fetchSource:async()=> 'bundle',hashSource:async()=>kind==='hash'?'unknown':reactionHash});
  const budget=createReactionBudget({game:f.game,fromUuid:async uuid=>uuid===f.actor.uuid?f.actor:uuid===f.token.uuid?f.token:null,reactionResources:adapter});
  await assert.rejects(budget.applyDamage(f.actor,f.params,f.native),/Reaction Checker/);assert.equal(f.count(),0);assert.equal(f.entries().length,0);
 }
});

test('no-block recovery preserves a newer manual Reaction resource value',async()=>{
 const f=fixture({reaction:true});await f.budget.applyDamage(f.actor,f.params,async p=>{f.combatant.flags['pf2e-reaction'].state=true;return f.native(p,{blocked:false})});
 assert.equal(f.combatant.flags['pf2e-reaction'].state,true);assert.equal(f.entries().length,0);
});

const nativePath=process.env.FVTT_REACTION_BUNDLE??'';
test('the real Reaction availability function sees the adapter payment', {skip:!nativePath},async()=>{
 const f=fixture({reaction:true}),source=readFileSync(nativePath,'utf8'),start=source.indexOf('function Ma('),end=source.indexOf('function x(',start);
 assert.ok(start>0&&end>start);
 const available=Function('S','I',source.slice(start,end)+';return R;')('pf2e-reaction',()=>false);
 const adapter=resources.createShieldReactionResources({game:f.game,fetchSource:async()=>source});
 const budget=createReactionBudget({game:f.game,fromUuid:async uuid=>uuid===f.actor.uuid?f.actor:uuid===f.token.uuid?f.token:null,reactionResources:adapter,onError:e=>{throw e}});
 // Use the adapter with its actual hash verifier; no hook replacement is needed
 // for the pre-native proof. A branded failure rolls back this isolated claim.
 assert.equal(available(f.combatant,'shield-block'),true);
 await assert.rejects(budget.applyDamage(f.actor,f.params,async()=>{assert.equal(available(f.combatant,'shield-block'),false);throw markUnappliedDamageError(Error('verified before native'))}),/verified before native/);
 assert.equal(available(f.combatant,'shield-block'),true);
});
