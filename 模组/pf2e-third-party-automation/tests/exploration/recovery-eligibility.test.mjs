import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createCapabilities} from '../../scripts/exploration/capabilities.mjs';
import {recoveryProposals} from '../../scripts/exploration/policy.mjs';
import {createFocusHealingProvider,createRefocusAdapter,createRefocusProvider} from '../../scripts/exploration/refocus.mjs';
const M='pf2e-third-party-automation',LAY='Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS';
const nativeData='C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/qa/runtime/Data';

function focusFixture({beforeConsume=()=>{},beforeCommit=()=>{},afterPaid=()=>{}}={}){
 const actor={id:'H',uuid:'Actor.H',flags:{},isDead:false,hasCondition:()=>false,modeOfBeing:'living',items:new Map(),getStatistic:()=>({rank:1}),system:{attributes:{hp:{value:30,max:30,negativeHealing:false}},resources:{focus:{value:1,max:1}}}};
 const patient={id:'P',uuid:'Actor.P',isDead:false,modeOfBeing:'living',items:[],getStatistic:()=>({rank:0}),attributes:{immunities:[]},system:{attributes:{hp:{value:1,max:30,negativeHealing:false}},resources:{focus:{value:0,max:0}}}};
 const hooks=new Map(),messages=new Map();let seq=0,consume,capture,applies=0,payments=0,rolls=0;
 const Hooks={on:(n,fn)=>{hooks.set(++seq,{n,fn});return seq},off:(_n,id)=>hooks.delete(id)};
 const fire=(n,m)=>{for(const h of hooks.values())if(h.n===n)h.fn(m,m)};
 const dice={total:18,_evaluated:true,toJSON:()=>({total:18,formula:'{18[healing]}',evaluated:true})};
 const variant={id:'L',uuid:'Actor.H.Item.L',actor,sourceId:LAY,rank:3,system:{cast:{focusPoints:1}},damageKinds:new Set(['healing']),rollDamage:async()=>{rolls++;const m={id:'D',flags:{pf2e:{origin:{uuid:variant.uuid},context:{options:[]}}},rolls:[dice]};fire('preCreateChatMessage',m);messages.set('D',m);fire('createChatMessage',m);return dice}};
 const item={id:'L',uuid:variant.uuid,type:'spell',actor,sourceId:LAY,rank:3,system:{overlays:{HEAL:{}}},loadVariant:()=>variant};actor.items.set('L',item);
 const ctx={validate(){}};const game={user:{id:'GM',settings:{showCheckDialogs:false}},time:{worldTime:6},messages};
 const objects=new Map([actor,patient,item].map(x=>[x.uuid,x]));
 const fromUuid=async uuid=>objects.get(uuid);
 const capabilities=createCapabilities({game,fromUuid,hpPools:{discover:a=>({ready:true,poolUUID:a.uuid})}});
 const activity={id:'A',actorUUID:actor.uuid,patientUUIDs:[patient.uuid],startedAt:0,endsAt:6,options:{itemUUID:item.uuid}};
 const provider=createFocusHealingProvider({game,Hooks,fromUuid,ownerOperations:{isActivityContext:c=>c===ctx},castEvents:{addMatcher(){},addConsumePolicy:fn=>consume=fn,addCapture:(_id,fn)=>capture=fn,ensurePaid:async()=>({id:'N',state:'used',messageId:'CAST'})},nativeTreatment:{applySavedResult:async()=>{applies++;return {receiptId:'R'}}}});provider.register();
 variant.spellcasting={cast:async spell=>{
  await beforeConsume({actor,patient,item});let commit;
  await consume({actor,item:spell,payload:{focusPoints:1},castNonce:'N',expectFocusCommit:spec=>commit=spec},async()=>{
   await beforeCommit({actor,patient,item});
   const proof={castNonce:'N',before:1,after:0,cost:1};
   const fields=commit.changes(proof);payments++;actor.system.resources.focus.value=0;
   actor.flags[M]={explorationFocusCommits:{A:fields[`flags.${M}.explorationFocusCommits.A`]}};return true;
  });
  const card={id:'CAST',flags:{pf2e:{origin:{uuid:spell.uuid}},[M]:{explorationFocus:capture(spell),nativeCast:{id:'N'}}},rolls:[]};messages.set(card.id,card);fire('createChatMessage',card);await afterPaid({actor,patient,item});
 }};
 return {actor,patient,item,variant,capabilities,provider,activity,ctx,get payments(){return payments},get applies(){return applies},get rolls(){return rolls}};
}

for(const [name,alter] of [
 ['negative healing',p=>p.system.attributes.hp.negativeHealing=true],
 ['vitality immunity',p=>p.attributes.immunities=[{type:'vitality'}]],
 ['healing immunity',p=>p.attributes.immunities=[{type:'healing'}]],
 ['unknown vitality eligibility',p=>delete p.system.attributes.hp.negativeHealing]
])test(`automatic focus healing excludes ${name} during planning and begin, retaining mundane treatment`,async()=>{
 const f=focusFixture();alter(f.patient);
 const snapshots=await f.capabilities.snapshot([f.actor.uuid,f.patient.uuid]);
 const proposals=recoveryProposals({actors:snapshots,activities:[],session:{goalsByPool:[{poolUUID:f.patient.uuid,targetHP:30}]},now:0,providerIds:['treat-wounds','focus-healing']});
 assert.equal(proposals.some(p=>p.providerId==='treat-wounds'),true);
 assert.equal(proposals.some(p=>p.providerId==='focus-healing'),false);
 assert.equal((await f.provider.begin(f.activity)).status,'blocked');
});

test('ordinary living patient retains the single native focus payment and healing',async()=>{
 const f=focusFixture();assert.equal((await f.provider.begin(f.activity)).status,'started');
 const actors=await f.capabilities.snapshot([f.actor.uuid,f.patient.uuid]);
 assert.equal(recoveryProposals({actors,activities:[],session:{goalsByPool:[{poolUUID:f.patient.uuid,targetHP:30}]},now:0,providerIds:['focus-healing']}).length,1);
 assert.equal((await f.provider.complete(f.activity,f.ctx)).status,'confirmed');
 assert.equal(f.payments,1);assert.equal(f.applies,1);assert.equal(f.actor.system.resources.focus.value,0);
});
for(const boundary of ['beforeConsume','beforeCommit'])for(const [name,alter] of [
 ['negative healing',({patient})=>patient.system.attributes.hp.negativeHealing=true],
 ['vitality immunity',({patient})=>patient.attributes.immunities=[{type:'vitality'}]],
 ['healing immunity',({patient})=>patient.attributes.immunities=[{type:'healing'}]],
 ['unconscious caster',({actor})=>actor.hasCondition=c=>c==='unconscious']
])test(`${name} at ${boundary} stops the original focus debit`,async()=>{
 const f=focusFixture({[boundary]:alter});assert.equal((await f.provider.begin(f.activity)).status,'started');
 await assert.rejects(f.provider.complete(f.activity,f.ctx),/unqualified|cannot|patient|source/i);
 assert.equal(f.payments,0);assert.equal(f.applies,0);assert.equal(f.actor.system.resources.focus.value,1);
});
test('post-payment loss of vitality eligibility stops HP application and cannot repay the activity',async()=>{
 const f=focusFixture({afterPaid:({patient})=>patient.system.attributes.hp.negativeHealing=true});
 await assert.rejects(f.provider.complete(f.activity,f.ctx),/unqualified|patient/i);
 assert.equal(f.payments,1);assert.equal(f.applies,0);assert.equal(f.actor.system.resources.focus.value,0);
 await assert.rejects(f.provider.complete(f.activity,f.ctx),/already/);assert.equal(f.payments,1);
});

async function refocusFixture(){
 const actor={uuid:'Actor.H',items:[],isDead:false,hasCondition:()=>false,system:{resources:{focus:{value:0,max:1}}}};
 const canvas={tokens:{controlled:[{actor}]}};const ctx={validate(){}};
 const activity={id:'A',actorUUID:actor.uuid,startedAt:0,endsAt:600,options:{}};
 const game={time:{worldTime:600},system:{id:'pf2e'},i18n:{format:()=>''}};let adapter;
 const source=await readFile(process.env.WORKBENCH_NATIVE_BUNDLE??`${nativeData}/modules/xdy-pf2e-workbench/xdy-pf2e-workbench.js`,'utf8');
 const start=source.indexOf('async function Tc(t, n)'),end=source.indexOf('//#endregion',start);assert.ok(start>0&&end>start);
 const native=Function('canvas','game','ChatMessage','CONST','e','ui',`${source.slice(start,end)};return Ec`)(canvas,game,{getSpeaker:()=>({}),create:async()=>{}},{CHAT_MESSAGE_STYLES:{EMOTE:1}},'xdy-pf2e-workbench',{});
 actor.update=async changes=>{const before=actor.system.resources.focus.value,after=adapter.commitValue(activity,actor,changes['system.resources.focus.value']);actor.system.resources.focus.value=after;adapter.capture({actor,proof:{nonce:'A',startedAt:0,before,after}});return actor};
 game.PF2eWorkbench={refocus:native};
 const ownerOperations={isActivityContext:c=>c===ctx};
 adapter=createRefocusAdapter({game,canvas,fromUuid:async()=>actor,ownerOperations,timeoutMs:30});
 const provider=createRefocusProvider({game,ledger:{getActivity:async()=>({state:'completing'})},capabilities:{discover:async()=>({isDead:actor.isDead,unconscious:actor.hasCondition('unconscious'),focus:actor.system.resources.focus})},refocusEvents:adapter,salubriousKiss:{},ownerOperations});
 return {actor,activity,ctx,canvas,game,adapter,provider};
}
test('qualified Refocus still executes the actual Workbench resource calculation once',async()=>{
 const f=await refocusFixture();assert.equal((await f.provider.begin(f.activity,f.ctx)).status,'started');
 const result=await f.provider.complete(f.activity,f.ctx);assert.equal(result.status,'confirmed');assert.equal(result.focusBefore,0);assert.equal(result.focusAfter,1);
 assert.equal((await f.provider.complete(f.activity,f.ctx)).focusAfter,1);assert.equal(f.actor.system.resources.focus.value,1);
});
for(const [name,alter] of [['unconscious',a=>a.hasCondition=c=>c==='unconscious'],['dead',a=>a.isDead=true]])test(`${name} actor after begin cannot complete native Workbench Refocus`,async()=>{
 const f=await refocusFixture();assert.equal((await f.provider.begin(f.activity,f.ctx)).status,'started');alter(f.actor);
 await assert.rejects(f.provider.complete(f.activity,f.ctx),/cannot-refocus/);assert.equal(f.actor.system.resources.focus.value,0);
});
for(const [name,alter] of [
 ['unconscious',f=>f.actor.hasCondition=c=>c==='unconscious'],
 ['unsupported feat',f=>f.actor.items.push({type:'feat',slug:'meditative-focus'})],
 ['changed control',f=>f.canvas.tokens.controlled=[]]
])test(`Refocus atomic commit rejects ${name} after native entry`,async()=>{
 const f=await refocusFixture();let rejected;
 f.game.PF2eWorkbench.refocus=async()=>{alter(f);try{f.adapter.commitValue(f.activity,f.actor,1)}catch(error){rejected=error}throw Error('stop-native-after-probe')};
 await assert.rejects(f.adapter.complete(f.activity,f.ctx),/stop-native/);
 assert.ok(rejected,'qualification must be checked at the final resource boundary');assert.equal(f.actor.system.resources.focus.value,0);
});
