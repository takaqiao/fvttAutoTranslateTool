import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const M='C:/Users/Taka/.codex/worktrees/exploration-core-release/fvtt/模组/pf2e-third-party-automation';
const ID='pf2e-third-party-automation';
const {createHalflingLuckLedger,HALFLING_LUCK_SOURCE}=await import('../scripts/halfling-luck-ledger.mjs');
const {createEatFortune,EAT_FORTUNE_SOURCES:E}=await import('../scripts/eat-fortune.mjs');
const {createRuneTransfer,selectedRuneWeaponId,runeTransferStatus,registerRuneTransferRuleElement,RUNE_TRANSFER_SOURCE,RUNE_TRANSFER_KEY}=await import('../scripts/rune-transfer.mjs');
const {createSpellCombination,SPELL_COMBINATION_SOURCES:S}=await import('../scripts/spell-combination.mjs');
assert.ok(process.env.PF2E_NATIVE_BUNDLE,'PF2E_NATIVE_BUNDLE must identify the pinned primary source');const primary=await readFile(process.env.PF2E_NATIVE_BUNDLE,'utf8');assert.equal(createHash('sha256').update(primary).digest('hex'),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');
const suppressFeats=Function(primary.match(/function suppressFeats\(e\) \{[\s\S]*?\n\}/)[0]+';return suppressFeats')();
function apply(target,changes){for(const[path,value]of Object.entries(changes)){let object=target;const keys=path.split('.');for(const key of keys.slice(0,-1))object=object[key]??={};object[keys.at(-1)]=structuredClone(value);}return target;}
function base(){const gm={id:'gm',isGM:true,active:true},owner={id:'owner',active:true},users=new Map([[gm.id,gm],[owner.id,owner]]);users.activeGM=gm;let owned=true;const writes=[],actor={id:'actor',uuid:'Actor.actor',type:'character',isToken:false,canAct:true,isDead:false,items:new Map(),flags:{},system:{actions:[]},testUserPermission:user=>user===gm||user===owner&&owned};const game={user:gm,users,actors:new Map([[actor.id,actor]]),scenes:new Map(),messages:new Map(),time:{worldTime:100},world:{id:'-'},pf2e:{}};actor.update=async changes=>{writes.push({type:'actor',current:game.actors.get(actor.id)===actor,owned});return apply(actor,changes)};return {gm,owner,actor,game,writes,setOwned:value=>owned=value};}
function feat(f,id,sourceId,type='feat'){const item={id,uuid:f.actor.uuid+'.Item.'+id,type,sourceId,suppressed:false,actor:f.actor,flags:{pf2e:{itemGrants:{}}},system:{}};f.actor.items.set(id,item);return item;}
function luck(){const f=base(),item=feat(f,'luck',HALFLING_LUCK_SOURCE);item.system={actionType:{value:'free'},frequency:{value:1,max:1,per:'day'}};item.update=async changes=>{f.writes.push({type:'luck'});return apply(item,changes)};const fromUuid=async uuid=>uuid===f.actor.uuid?f.actor:uuid===item.uuid?item:null;const options={fromUuid,randomId:()=> 'luck-nonce'};const ledger=createHalflingLuckLedger({game:f.game,...options}),client=createHalflingLuckLedger({game:{...f.game,user:f.owner},...options});return {...f,item,ledger,client,scope:{actor:f.actor,item,user:f.owner,invocationId:'original-check',fingerprint:'original-native-proof'}};}
test('active exact Halfling Luck claims once with prepared frequency retained',async()=>{const f=luck();const r=await f.ledger.claim(f.scope);assert.equal(r.status,'claimed');assert.equal(f.item.system.frequency.value,1);assert.equal(f.writes.length,1);});
test('native suppressed Halfling Luck must not claim a new automatic operation',async()=>{const f=luck();suppressFeats([f.item]);let result,error;try{result=await f.ledger.claim(f.scope)}catch(e){error=e.message}assert.equal(f.writes.length,0);});
test('native suppression after claim must block original automatic payment preparation',async()=>{const f=luck(),r=await f.ledger.claim(f.scope);f.client.authorizePayment(f.item,r.nonce,f.owner);suppressFeats([f.item]);const changes={'system.frequency.value':0},options={};const accepted=f.client.preparePayment(f.item,changes,options,f.owner.id);assert.equal(accepted,false);});
async function eat({suppressed=false,change,setup}={}){const f=base();f.actor.id='reactor';f.actor.uuid='Actor.reactor';f.game.actors=new Map([[f.actor.id,f.actor]]);f.owner.character=f.actor;const item=feat(f,'eat',E.eat);item.system.frequency={value:1,max:1};item.update=async changes=>{f.writes.push({type:'eat',owned:f.actor.testUserPermission(f.owner,'OWNER'),current:f.actor.items.get(item.id)===item});return apply(item,changes)};item.toMessage=async()=>{const card={id:'eat-card',uuid:'ChatMessage.eat-card',item,flags:{},async update(changes){f.writes.push({type:'card'});return apply(this,changes)}};f.game.messages.set(card.id,card);return card};if(suppressed)suppressFeats([item]);
 const roller={id:'roller',uuid:'Actor.roller',items:new Map(),testUserPermission:user=>user===f.gm,flags:{[ID]:{reactionChecks:{reactions:[{kind:'clock',nonce:'paid-clock',state:'claimed',checkId:'clock-card'}]}}}},clock={id:'clock',uuid:'Actor.roller.Item.clock',actor:roller,type:'feat',sourceId:E.clock};roller.items.set(clock.id,clock);f.game.actors.set(roller.id,roller);
 const scene={id:'scene',tokens:new Map()},token=actor=>({id:actor.id,uuid:'Scene.scene.Token.'+actor.id,documentName:'Token',actor,parent:scene,name:'Visible target',playersCanSeeName:true});const origin=token(roller),reactor=token(f.actor);scene.tokens.set(origin.id,origin);scene.tokens.set(reactor.id,reactor);f.game.scenes.set(scene.id,scene);f.game.messages.set('clock-card',{id:'clock-card',item:clock,flags:{[ID]:{reactionChecks:{kind:'reaction-use',nonce:'paid-clock'}}}});const docs=new Map([f.actor,item,roller,clock,origin,reactor].map(doc=>[doc.uuid,doc]));
 const payload={nonce:'original-reaction',kind:'clock',clockNonce:'paid-clock',sourceItemUuid:clock.uuid,sourceActorUuid:roller.uuid,sourceTokenUuid:origin.uuid,rollerActorUuid:roller.uuid,rollerTokenUuid:origin.uuid,effectType:'fortune',type:'saving-throw',options:[]};Object.assign(f,{item,roller,clock,scene,origin,reactor,docs,payload});setup?.(f);const provider=createEatFortune({game:f.game,fromUuid:async uuid=>docs.get(uuid),choose:async()=>{change?.({...f,item});return 'eat'}});provider.register({Hooks:{on(){return 1},off(){}}});let result,error;try{result=await provider.decide(payload,f.gm)}catch(e){error=e.message}return {...f,item,result,error};}
test('active Eat Fortune consumes one exact use and durable claim',async()=>{const f=await eat();assert.equal(f.result.reactorActorUuid,f.actor.uuid);assert.equal(f.item.system.frequency.value,0);assert.equal(f.writes.filter(w=>w.type==='eat').length,1);});
test('native suppressed Eat Fortune must not pay or disrupt',async()=>{const f=await eat({suppressed:true});assert.equal(f.item.system.frequency.value,1);});
for(const [label,change]of [['OWNER',f=>f.setOwned(false)],['current Item',f=>f.actor.items.set(f.item.id,{...f.item})]])test(`Eat Fortune decision await loses ${label}: must not pay or disrupt`,async()=>{const f=await eat({change});assert.equal(f.item.system.frequency.value,1);});
function runes(suppressed=false,setup){const f=base(),item=feat(f,'cutting',RUNE_TRANSFER_SOURCE);if(suppressed)suppressFeats([item]);const effect={id:'state',uuid:'Actor.actor.Item.state',type:'effect',actor:f.actor,flags:{[ID]:{kind:'rune-transfer',runeTransfer:{version:1,source:RUNE_TRANSFER_SOURCE,featId:item.id,selectedWeaponId:'weapon',revision:1}}}};const weapon={id:'weapon',type:'weapon',actor:f.actor,category:'martial',isMelee:true,isEquipped:true,isHeld:true,isStowed:false,hands:'1',handsHeld:1,traits:new Set(),system:{runes:{potency:1,striking:0,property:[]},material:{},traits:{value:[]}}},wraps={id:'wraps',type:'weapon',category:'unarmed',isEquipped:true,isInvested:true,system:{traits:{otherTags:['handwraps-of-mighty-blows']},runes:{potency:2,striking:1,property:[]}}};f.actor.items.set(effect.id,effect);f.actor.items.set(weapon.id,weapon);f.actor.items.set(wraps.id,wraps);f.actor.synthetics={strikeAdjustments:[],itemAlterations:[]};f.actor.rollOptions={all:{}};f.game.pf2e={variantRules:{AutomaticBonusProgression:{isEnabled:()=>false}}};
 class Base{constructor(data,{parent}){Object.assign(this,data);this.item=parent;this.actor=parent.actor;this.ignored=false;}}class ItemAlteration{constructor(data){this.data=data;}applyAlteration(){f.writes.push({type:'native-ItemAlteration',property:this.data.property,value:this.data.value});}}class AdjustStrike{};f.game.pf2e.RuleElement=Base;f.game.pf2e.RuleElements={custom:{},builtin:{ItemAlteration,AdjustStrike}};setup?.({...f,item,effect,weapon,wraps});const Rule=registerRuneTransferRuleElement(f.game);new Rule({key:RUNE_TRANSFER_KEY},{parent:effect}).beforePrepareData();return {...f,item,selected:selectedRuneWeaponId(f.actor)};}
test('active exact Cutting Heaven prepares native higher handwrap runes',()=>{const f=runes();assert.equal(f.selected,'weapon');assert.deepEqual(f.writes.map(w=>w.property),['runes-potency','runes-striking']);});
test('native suppressed Cutting Heaven must not apply transferred weapon runes',()=>{const f=runes(true);assert.equal(f.writes.length,0);});
function combination(){const f=base(),spellstrike=feat(f,'spellstrike',S.strike,'action'),recharge=feat(f,'recharge','local-generated','action');recharge.flags[ID]={spellstrikeRecharge:true};const message={id:'usage',uuid:'ChatMessage.usage',author:f.owner,flags:{pf2e:{origin:{uuid:recharge.uuid}}},update:async changes=>{f.writes.push({type:'message'});return apply(message,changes)}};f.game.messages.set(message.id,message);const nativeCasts={addMatcher(){},addActorMatcher(){},addActivityMatcher(){},captureUsage(){},async ensurePaid(){f.writes.push({type:'native-payment'})}};const provider=createSpellCombination({game:f.game,nativeCasts,nativeOperations:{}});const execute=()=>provider.executeUsage({actor:f.actor,item:recharge,message,user:f.owner,action:'spell-combination:recharge'});return {...f,spellstrike,recharge,message,nativeCasts,provider,execute};}
test('active original Spellstrike recharge charges once',async()=>{const f=combination();assert.equal(await f.execute(),'法术打击已充能。');assert.equal(f.actor.flags[ID].spellstrike.charged,true);assert.equal(f.writes.filter(w=>w.type==='actor').length,1);});
test('native suppressed original Spellstrike must not authorize its derived recharge',async()=>{const f=combination();suppressFeats([f.spellstrike]);let result,error;try{result=await f.execute()}catch(e){error=e.message}assert.equal(f.writes.length,0);});
for(const [label,change]of [['Actor',f=>f.game.actors.set(f.actor.id,{...f.actor})],['Item',f=>f.actor.items.set(f.recharge.id,{...f.recharge})]])test(`Spellstrike recharge queue await replaces current ${label}: must not write stale source`,async()=>{const f=combination(),pending=f.execute();change(f);let result,error;try{result=await pending}catch(e){error=e.message}assert.equal(f.writes.length,0);});
test('Spellstrike recharge queue OWNER loss is rejected before writing',async()=>{const f=combination(),pending=f.execute();f.setOwned(false);await assert.rejects(pending);assert.equal(f.writes.length,0);});

const {isHalflingLuckItem}=await import('../scripts/halfling-luck-rules.mjs');
const {createHalflingLuckProvider}=await import('../scripts/halfling-luck.mjs');
for(const alias of ['suppressed','isSuppressed','system.suppressed'])test(`Halfling Luck ${alias} blocks admission and exact original payment even with a second active copy`,async()=>{
 const f=luck();if(alias==='system.suppressed')f.item.system.suppressed=true;else f.item[alias]=true;
 assert.equal(isHalflingLuckItem(f.item),false);await assert.rejects(f.ledger.claim(f.scope));assert.deepEqual(f.writes,[]);
 const active={...f.item,id:'active-luck',uuid:f.actor.uuid+'.Item.active-luck',suppressed:false,isSuppressed:false,system:{...f.item.system,suppressed:false}};f.actor.items.set(active.id,active);const provider=createHalflingLuckProvider({game:f.game,ledger:f.ledger});assert.equal(provider.handlesActor(f.actor),true);assert.equal(provider.resolveAction(f.item),undefined);assert.equal(provider.resolveAction(active),'halfling-luck:use');
});
test('Halfling Luck registered native preUpdate still rejects a suppressed originally authorized payment',async()=>{
 const f=luck(),record=await f.ledger.claim(f.scope);f.client.authorizePayment(f.item,record.nonce,f.owner);const hooks=new Map(),provider=createHalflingLuckProvider({game:{...f.game,user:f.owner},ledger:f.client});provider.register({Hooks:{on:(name,fn)=>{hooks.set(name,fn);return name},off(){}}});
 suppressFeats([f.item]);const changes={'system.frequency.value':0},options={};assert.equal(hooks.get('preUpdateItem')(f.item,changes,options,f.owner.id),false);assert.equal(f.item.system.frequency.value,1);assert.equal(changes[`flags.${ID}.halflingLuck`],undefined);
});
for(const alias of ['isSuppressed','system.suppressed'])test(`Cutting Heaven ${alias} prevents native rune adjustments`,()=>{const f=runes(false,({item})=>{if(alias==='system.suppressed')item.system.suppressed=true;else item[alias]=true});assert.equal(f.selected,null);assert.deepEqual(f.writes,[]);});
test('Cutting Heaven rejects conflicting current source while accepting a separate active owned source',()=>{
 const f=runes(false,({actor,item,effect})=>{item.sourceId='Other.source';item.flags.core={sourceId:RUNE_TRANSFER_SOURCE};const copy={...item,id:'active-cutting',sourceId:RUNE_TRANSFER_SOURCE,uuid:actor.uuid+'.Item.active-cutting'};actor.items.set(copy.id,copy);effect.flags[ID].runeTransfer.featId=copy.id});assert.equal(f.selected,'weapon');assert.equal(f.writes.length,2);
});
for(const alias of ['suppressed','isSuppressed','system.suppressed'])test(`Spellstrike ${alias} cannot authorize derived recharge; a separate active owned action can`,async()=>{
 const f=combination();if(alias==='system.suppressed')f.spellstrike.system.suppressed=true;else f.spellstrike[alias]=true;await assert.rejects(f.execute());assert.deepEqual(f.writes,[]);
 const active={...f.spellstrike,id:'active-strike',uuid:f.actor.uuid+'.Item.active-strike',suppressed:false,isSuppressed:false,system:{}};f.actor.items.set(active.id,active);assert.equal(await f.execute(),'法术打击已充能。');assert.equal(f.writes.filter(w=>w.type==='actor').length,1);
});
const sourceChanges=[['current GM document',f=>{const next={...f.gm};f.game.user=next;f.game.users.set(next.id,next);f.game.users.activeGM=next}],['current OWNER User',f=>f.game.users.set(f.owner.id,{...f.owner})],['Actor',f=>f.game.actors.set(f.actor.id,{...f.actor})],['Item',f=>f.actor.items.set(f.recharge.id,{...f.recharge})],['source card',f=>f.game.messages.set(f.message.id,{...f.message})],['native source suppression',f=>suppressFeats([f.spellstrike])]];
for(const [label,change]of sourceChanges)test(`Spellstrike conflux payment await loses ${label}: durable started card prevents charge or second payment`,async()=>{
 const f=combination(),conflux=feat(f,'conflux','Native.conflux','spell');conflux.system={traits:{value:['focus','magus']},time:{value:'1'}};f.message.flags.pf2e.origin.uuid=conflux.uuid;
 f.nativeCasts.ensurePaid=async()=>{f.writes.push({type:'native-payment'});change({...f,recharge:conflux})};const run=()=>f.provider.executeUsage({actor:f.actor,item:conflux,message:f.message,user:f.owner,action:'spell-combination:conflux'});await assert.rejects(run());
 assert.equal(f.message.flags[ID]?.spellCombinationUse?.state,'started');assert.equal(f.writes.filter(w=>w.type==='actor').length,0);assert.equal(f.writes.filter(w=>w.type==='native-payment').length,1);
 f.game.user=f.gm;f.game.users.set(f.gm.id,f.gm);f.game.users.activeGM=f.gm;f.game.users.set(f.owner.id,f.owner);f.game.actors.set(f.actor.id,f.actor);f.actor.items.set(conflux.id,conflux);f.game.messages.set(f.message.id,f.message);f.spellstrike.suppressed=false;await assert.rejects(run());assert.equal(f.writes.filter(w=>w.type==='native-payment').length,1);
});
test('Spellstrike started card write loses original OWNER: no subsequent charge and no automatic retry',async()=>{
 const f=combination(),original=f.message.update;f.message.update=async changes=>{const result=await original(changes);if(changes[`flags.${ID}.spellCombinationUse`]?.state==='started')f.setOwned(false);return result};await assert.rejects(f.execute());assert.equal(f.message.flags[ID].spellCombinationUse.state,'started');assert.equal(f.writes.filter(w=>w.type==='actor').length,0);
 f.setOwned(true);await assert.rejects(f.execute());assert.equal(f.writes.filter(w=>w.type==='actor').length,0);
});
for(const [label,change]of sourceChanges.slice(0,5))test(`Spellstrike charge await loses ${label}: retains committed charge and started occupancy`,async()=>{
 const f=combination(),original=f.actor.update;f.actor.update=async changes=>{const result=await original(changes);change(f);return result};await assert.rejects(f.execute());assert.equal(f.actor.flags[ID]?.spellstrike?.charged,true);assert.equal(f.message.flags[ID]?.spellCombinationUse?.state,'started');assert.equal(f.writes.filter(w=>w.type==='actor').length,1);
});
for(const alias of ['isSuppressed','system.suppressed'])test(`Eat Fortune excludes ${alias} before any claim or payment`,async()=>{const f=await eat({setup:f=>{if(alias==='system.suppressed')f.item.system.suppressed=true;else f.item[alias]=true}});assert.equal(f.result,null);assert.equal(f.item.system.frequency.value,1);assert.deepEqual(f.writes,[]);});
const eatChanges=[['GM document',f=>{const next={...f.gm};f.game.user=next;f.game.users.set(next.id,next);f.game.users.activeGM=next}],['chosen OWNER User',f=>f.game.users.set(f.owner.id,{...f.owner})],['chosen OWNER permission',f=>f.setOwned(false)],['reactor Actor',f=>f.game.actors.set(f.actor.id,{...f.actor})],['reactor Item',f=>f.actor.items.set(f.item.id,{...f.item})],['reactor Token',f=>f.scene.tokens.set(f.reactor.id,{...f.reactor})],['original Clock card',f=>f.game.messages.set('clock-card',{...f.game.messages.get('clock-card')})],['original Clock Item',f=>f.roller.items.set(f.clock.id,{...f.clock})]];
for(const [label,change]of eatChanges)for(const phase of ['choice','claim','payment'])test(`Eat Fortune loses ${label} after ${phase}: stops with any durable claim and paid use retained`,async()=>{
 const f=await eat({change:phase==='choice'?change:undefined,setup:f=>{if(phase==='choice')return;const target=phase==='claim'?f.actor:f.item,original=target.update;target.update=async changes=>{const result=await original(changes);change(f);return result};}});
 assert.ok(f.error);assert.equal(f.item.system.frequency.value,phase==='payment'?0:1);assert.equal(f.result,undefined);assert.equal(f.writes.filter(w=>w.type==='card').length,0);if(phase!=='choice')assert.equal(f.actor.flags[ID].reactionChecks.reactions[0].state,'claimed');
});

function substitution(f,kind='assurance'){
 const item={id:'substitute',uuid:f.roller.uuid+'.Item.substitute',type:kind==='devise'?'effect':'feat',sourceId:E[kind],actor:f.roller,flags:{pf2e:{itemGrants:{}}},system:{context:{origin:{actor:f.roller.uuid,token:f.origin.uuid}}}};f.roller.items.set(item.id,item);f.docs.set(item.uuid,item);
 const rule={item,key:'SubstituteRoll',selector:kind==='devise'?'strike-attack-roll':'saving-throw',slug:'native-substitution',value:12,effectType:'fortune',required:true,label:'Native substitution',ignored:false,test:()=>true,resolveInjectedProperties:value=>value,resolveValue:value=>value};f.roller.rules=[rule];
 Object.assign(f.payload,{kind,sourceItemUuid:item.uuid,slug:rule.slug,value:12,required:true,label:rule.label,type:kind==='devise'?'attack-roll':'saving-throw',action:kind==='devise'?'strike':null,domains:[rule.selector],substitutions:[{slug:rule.slug,value:12,effectType:'fortune',required:true,selected:true,label:rule.label}],options:kind==='devise'?['devise-a-stratagem:attack','target:mark:devise-a-stratagem']:[]});delete f.payload.clockNonce;return {item,rule};
}
test('Eat Fortune current exact SubstituteRoll source consumes only its original reaction use',async()=>{
 const f=await eat({setup:f=>substitution(f)});assert.equal(f.error,undefined);assert.equal(f.result.sourceKind,'assurance');assert.equal(f.item.system.frequency.value,0);assert.equal(f.writes.filter(w=>w.type==='eat').length,1);
});
test('Eat Fortune choice cannot change its current prepared SubstituteRoll value away from the original native DTO',async()=>{
 const f=await eat({setup:f=>substitution(f),change:f=>f.roller.rules=[{...f.roller.rules[0],value:13}]});assert.ok(f.error);assert.equal(f.item.system.frequency.value,1);assert.deepEqual(f.writes,[]);
});

test('Eat Fortune accepts native reprepare of an equivalent SubstituteRoll on the same original Item Document',async()=>{
 const f=await eat({setup:f=>{substitution(f);const original=f.actor.update;f.actor.update=async changes=>{const result=await original(changes);f.roller.rules=[{...f.roller.rules[0]}];return result};},change:f=>f.roller.rules=[{...f.roller.rules[0]}]});assert.equal(f.error,undefined);assert.equal(f.result.sourceKind,'assurance');assert.equal(f.item.system.frequency.value,0);assert.equal(f.writes.filter(w=>w.type==='eat').length,1);
});
test('Eat Fortune loses its exact substitute feat after claim: held claim cannot proceed to payment',async()=>{
 const f=await eat({setup:f=>{const {item}=substitution(f),original=f.actor.update;f.actor.update=async changes=>{const result=await original(changes);suppressFeats([item]);return result};}});assert.ok(f.error);assert.equal(f.item.system.frequency.value,1);assert.equal(f.actor.flags[ID].reactionChecks.reactions[0].state,'claimed');assert.equal(f.writes.filter(w=>w.type==='eat').length,0);
});
test('Eat Fortune exact Devise native deletion is a successful consumed-source proof and does not demand the deleted item remain live',async()=>{
 const f=await eat({setup:f=>{const {item}=substitution(f,'devise');f.roller.deleteEmbeddedDocuments=async(_kind,ids)=>{assert.deepEqual(ids,[item.id]);f.writes.push({type:'native-delete'});f.roller.items.delete(item.id);f.roller.rules=[];return [item]};}});
 assert.equal(f.error,undefined);assert.equal(f.result.sourceKind,'devise');assert.equal(f.writes.filter(w=>w.type==='native-delete').length,1);assert.equal(f.actor.flags[ID].reactionChecks.reactions[0].sourceConsumed,true);assert.equal(f.item.system.frequency.value,0);
});
test('Eat Fortune lost Devise deletion response retains claim and payment without publishing a replacement action',async()=>{
 const f=await eat({setup:f=>{const {item}=substitution(f,'devise');f.roller.deleteEmbeddedDocuments=async()=>{f.roller.items.delete(item.id);f.roller.rules=[];f.writes.push({type:'native-delete'});throw Error('native delete response lost')};}});
 assert.match(f.error,/response lost/);assert.equal(f.item.system.frequency.value,0);assert.equal(f.actor.flags[ID].reactionChecks.reactions[0].state,'claimed');assert.equal(f.writes.filter(w=>w.type==='card').length,0);assert.equal(f.writes.filter(w=>w.type==='native-delete').length,1);
});
test('a generated Spellstrike recharge can complete on the exact current unlinked synthetic Actor',async()=>{
 const f=combination(),baseActor={id:'base',uuid:'Actor.base'},scene={id:'scene',tokens:new Map()},token={id:'synthetic',uuid:'Scene.scene.Token.synthetic',documentName:'Token',actorLink:false,actorId:baseActor.id,baseActor,actor:f.actor,parent:scene};f.game.actors.delete(f.actor.id);f.game.actors.set(baseActor.id,baseActor);f.game.scenes.set(scene.id,scene);scene.tokens.set(token.id,token);f.actor.isToken=true;f.actor.token=token;f.actor.uuid='Scene.scene.Token.synthetic.Actor.actor';f.recharge.uuid=f.actor.uuid+'.Item.recharge';f.spellstrike.uuid=f.actor.uuid+'.Item.spellstrike';f.message.flags.pf2e.origin.uuid=f.recharge.uuid;
 assert.equal(await f.execute(),'法术打击已充能。');assert.equal(await f.execute(),'本次组合活动已经结算。');assert.equal(f.writes.filter(w=>w.type==='actor').length,1);
});
test('Spellstrike native payment with a lost response leaves started occupancy and cannot pay or charge again',async()=>{
 const f=combination(),item=feat(f,'conflux','Native.conflux','spell');item.system={traits:{value:['focus','magus']},time:{value:'1'}};f.message.flags.pf2e.origin.uuid=item.uuid;
 f.nativeCasts.ensurePaid=async()=>{f.writes.push({type:'native-payment'});throw Error('payment response lost')};const run=()=>f.provider.executeUsage({actor:f.actor,item,message:f.message,user:f.owner,action:'spell-combination:conflux'});
 await assert.rejects(run(),/response lost/);await assert.rejects(run(),/不能|不会/);assert.equal(f.message.flags[ID].spellCombinationUse.state,'started');assert.equal(f.writes.filter(w=>w.type==='native-payment').length,1);assert.equal(f.writes.filter(w=>w.type==='actor').length,0);
});
test('Spellstrike actor update veto cannot record a completed charge and does not retry its started card',async()=>{
 const f=combination();f.actor.update=async()=>undefined;await assert.rejects(f.execute());await assert.rejects(f.execute());assert.equal(f.actor.flags[ID]?.spellstrike,undefined);assert.equal(f.message.flags[ID].spellCombinationUse.state,'started');
});

test('Cutting Heaven capacity conflict preserves existing property runes while retaining native fundamental upgrades',()=>{
 const f=runes(false,({wraps})=>wraps.system.runes.property=['flaming','frost','shock']);const status=runeTransferStatus(f.actor,f.game);assert.equal(status.status,'capacity-conflict');assert.equal(status.capacity,2);assert.deepEqual(status.transfer,[]);assert.deepEqual(status.properties,[]);assert.deepEqual(f.writes.map(w=>w.property),['runes-potency','runes-striking']);
});
test('Cutting Heaven uses the native ABP capacity function without replacing native ItemAlteration upgrades',()=>{
 const f=runes(false,({actor,game})=>{actor.level=10;game.pf2e.variantRules.AutomaticBonusProgression={isEnabled:a=>a===actor,getAttackPotency:level=>{assert.equal(level,10);return 3}}});assert.equal(runeTransferStatus(f.actor,f.game).capacity,3);assert.equal(f.writes.length,2);
});
test('Cutting Heaven late preparation handles only the current selected melee usage and stops if its source becomes suppressed',()=>{
 const f=runes(),original=f.actor.items.get('weapon'),late=f.actor.synthetics.itemAlterations[0];late.applyAlteration({singleItem:{...original,isMelee:false}});assert.equal(f.writes.length,2);late.applyAlteration({singleItem:{...original,altUsageType:'melee'}});assert.equal(f.writes.length,4);suppressFeats([f.item]);late.applyAlteration({singleItem:{...original,altUsageType:'melee'}});assert.equal(f.writes.length,4);
});

function runeUsage({selected=true,create=false,choose}={}){
 const choices=[],f=runes(false,({actor,effect,weapon,writes})=>{
  effect.system={rules:[{key:RUNE_TRANSFER_KEY}]};
  if(!selected){effect.flags[ID].runeTransfer.selectedWeaponId=null;effect.flags[ID].runeTransfer.revision=0}
  const attachUpdate=document=>{document.update=async changes=>{writes.push({type:'rune-update',id:document.id,changes:structuredClone(changes)});return apply(document,changes)};return document};attachUpdate(effect);
  weapon.name='First sword';weapon.uuid=actor.uuid+'.Item.'+weapon.id;
  const second={...weapon,id:'second',uuid:actor.uuid+'.Item.second',name:'Second sword',system:structuredClone(weapon.system),traits:new Set(weapon.traits)};actor.items.set(second.id,second);
  actor.createEmbeddedDocuments=async(documentType,data)=>data.map((source,index)=>{const document=attachUpdate({...structuredClone(source),id:'created-state-'+index,uuid:actor.uuid+'.Item.created-state-'+index,actor});writes.push({type:'rune-create',documentType,data:structuredClone(source),id:document.id});actor.items.set(document.id,document);return document});
  if(create)actor.items.delete(effect.id);
 });
 const provider=createRuneTransfer({game:f.game,choose:async request=>{choices.push(request);return choose?choose(request):'second'}}),execute=()=>provider.executeUsage({actor:f.actor,item:f.item,user:f.owner,action:'rune-transfer:select'});
 return {...f,provider,execute,choices};
}
const runeDocumentWrites=f=>f.writes.filter(w=>w.type==='rune-update'||w.type==='rune-create');

test('Cutting Heaven executeUsage selects the second held weapon and non-force readiness preserves it without writes',async()=>{
 const f=runeUsage();assert.match(await f.execute(),/已保存/);
 const effect=f.actor.items.get('state'),state=effect.flags[ID].runeTransfer;
 assert.equal(selectedRuneWeaponId(f.actor),'second');assert.equal(state.revision,2);assert.equal(state.declined,null);assert.deepEqual(effect.system.rules,[{key:RUNE_TRANSFER_KEY}]);
 assert.deepEqual(f.choices[0].choices.map(c=>c.value),['weapon','second']);assert.equal(f.choices[0].user,f.owner);assert.equal(runeDocumentWrites(f).length,1);
 assert.deepEqual(runeTransferStatus(f.actor,f.game),{status:'ready',capacity:2,properties:[],transfer:[],skipped:[],selectedWeaponId:'second'});
 await f.provider.ensureReady(f.actor,f.owner);await f.provider.maintain(f.actor);
 assert.equal(selectedRuneWeaponId(f.actor),'second');assert.equal(effect.flags[ID].runeTransfer.revision,2);assert.equal(f.choices.length,1);assert.equal(runeDocumentWrites(f).length,1);
});

test('Cutting Heaven explicit executeUsage reopens selection and advances the saved revision',async()=>{
 const f=runeUsage();await f.execute();await f.execute();
 assert.equal(f.choices.length,2);assert.equal(selectedRuneWeaponId(f.actor),'second');assert.equal(f.actor.items.get('state').flags[ID].runeTransfer.revision,3);assert.equal(runeDocumentWrites(f).length,2);
});

test('Cutting Heaven cancelled explicit selection records decline without erasing an existing choice or revision',async()=>{
 const f=runeUsage({choose:async()=>null});await f.execute();const state=f.actor.items.get('state').flags[ID].runeTransfer;
 assert.equal(selectedRuneWeaponId(f.actor),'weapon');assert.equal(state.revision,1);assert.equal(state.declined,'second,weapon');
 assert.deepEqual(runeDocumentWrites(f).map(w=>w.changes),[{[`flags.${ID}.runeTransfer.declined`]:'second,weapon'}]);
 await f.provider.ensureReady(f.actor,f.owner);await f.provider.maintain(f.actor);assert.equal(f.choices.length,1);assert.equal(runeDocumentWrites(f).length,1);assert.equal(selectedRuneWeaponId(f.actor),'weapon');assert.equal(state.revision,1);
});

test('Cutting Heaven cancelled first choice suppresses repeated readiness prompts for the same eligible weapons',async()=>{
 const f=runeUsage({selected:false,choose:async()=>null});assert.match(await f.execute(),/没有更改/);
 const effect=f.actor.items.get('state');assert.equal(selectedRuneWeaponId(f.actor),null);assert.equal(effect.flags[ID].runeTransfer.revision,0);assert.equal(effect.flags[ID].runeTransfer.declined,'second,weapon');
 assert.deepEqual(await f.provider.ensureReady(f.actor,f.owner),{status:'inactive',selectedWeaponId:null});await f.provider.maintain(f.actor);
 assert.equal(f.choices.length,1);assert.equal(runeDocumentWrites(f).length,1);assert.equal(effect.flags[ID].runeTransfer.revision,0);
});

for(const [label,change]of [['OWNER',f=>f.setOwned(false)],['weapon eligibility',f=>{f.actor.items.get('second').isHeld=false}],['active native feature',f=>suppressFeats([f.item])]])test(`Cutting Heaven executeUsage chooser await loses ${label} and cannot save a new selection`,async()=>{
 const f=runeUsage({choose:async()=>{change(f);return 'second'}});await assert.rejects(f.execute());
 const state=f.actor.items.get('state').flags[ID].runeTransfer;assert.equal(state.selectedWeaponId,'weapon');assert.equal(state.revision,1);assert.deepEqual(runeDocumentWrites(f),[]);assert.equal(f.choices.length,1);
});

for(const [label,change]of [['allowed world',f=>{f.game.world.id='other-world'}],['character type',f=>{f.actor.type='npc'}],['current Actor',f=>f.game.actors.set(f.actor.id,{...f.actor})],['OWNER',f=>f.setOwned(false)],['exact feat source',f=>{f.item.sourceId='Other.source'}],['active native feature',f=>suppressFeats([f.item])]])test(`Cutting Heaven executeUsage without ${label} rejects before selecting or writing`,async()=>{
 const f=runeUsage();change(f);await assert.rejects(f.execute());assert.equal(f.choices.length,0);assert.deepEqual(runeDocumentWrites(f),[]);
});

test('Cutting Heaven executeUsage creates its canonical effect before saving the first selected weapon',async()=>{
 const f=runeUsage({create:true});assert.match(await f.execute(),/已保存/);
 const effect=f.actor.items.get('created-state-0'),state=effect.flags[ID].runeTransfer,writes=runeDocumentWrites(f);
 assert.equal(selectedRuneWeaponId(f.actor),'second');assert.equal(state.revision,1);assert.equal(state.featId,f.item.id);assert.equal(state.source,RUNE_TRANSFER_SOURCE);assert.deepEqual(effect.system.rules,[{key:RUNE_TRANSFER_KEY}]);
 assert.deepEqual(writes.map(w=>w.type),['rune-create','rune-update']);assert.equal(writes[0].documentType,'Item');assert.equal(writes[0].data.flags[ID].runeTransfer.selectedWeaponId,null);assert.equal(writes[0].data.flags[ID].runeTransfer.revision,0);
 await f.provider.ensureReady(f.actor,f.owner);await f.provider.maintain(f.actor);assert.equal(f.choices.length,1);assert.equal(runeDocumentWrites(f).length,2);assert.equal(effect.flags[ID].runeTransfer.revision,1);
});
