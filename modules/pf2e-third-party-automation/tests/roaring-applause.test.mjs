import test from 'node:test';
import assert from 'node:assert/strict';
import {createRoaringEffects} from '../scripts/roaring-effects.mjs';
import {createRoaringApplause} from '../scripts/roaring-applause.mjs';
const ID='pf2e-third-party-automation',SOURCE='Compendium.pf2e.spells-srd.Item.czO0wbT1i320gcu9';
const copy=x=>structuredClone(x);
async function activeReactionSource(options={},outcome='failure'){
 const f=fixture(options);await f.run();const r=f.effects.list(f.targetActor)[0];
 await f.clients.gm.provider.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:outcome,revision:1,proof:{invocationId:'save1'}});
 return f;
}
for(const outcome of ['success','failure','criticalFailure'])test(`reaction query restricts accepted ${outcome} on both clients without writes`,async()=>{
 const f=await activeReactionSource({},outcome),before=copy([...f.records.values()]),operations=[...f.operations];
 for(const c of Object.values(f.clients))for(let n=0;n<3;n++){const q=c.provider.reactionRestriction(f.targetActor);assert.equal(q.status,'restricted');assert.equal(q.sources.length,1);assert.equal(q.sources[0].status,'restricted');}
 assert.deepEqual([...f.records.values()],before);assert.deepEqual(f.operations,operations);assert.equal(f.entry.system.slots.slot3.value,1);assert.equal(f.counts(),1);
});
test('reaction query clears absent, ended and whole-spell immune sources',async()=>{
 const empty=fixture();assert.equal(empty.clients.gm.provider.reactionRestriction(empty.targetActor).status,'clear');
 const ended=await activeReactionSource({},'criticalSuccess');assert.equal(ended.clients.gm.provider.reactionRestriction(ended.targetActor).status,'clear');
 const immune=fixture({immunity:{spell:true,slowed:false,fascinated:false}});await immune.run();assert.equal(immune.clients.gm.provider.reactionRestriction(immune.targetActor).status,'clear');
});
test('awaiting save and inconsistent active result are manual',async()=>{
 const f=fixture();await f.run();const q=()=>f.clients.gm.provider.reactionRestriction(f.targetActor);assert.equal(q().status,'manual');
 const r=[...f.records.values()][0];r.state.status='active';r.state.result={revision:1,receiptId:'save1',outcome:'criticalSuccess'};assert.equal(q().status,'manual');
 r.state.result={revision:2,receiptId:'save1',outcome:'failure'};assert.equal(q().status,'manual');
});
test('actual caster end and hard cap clear before asynchronous effect cleanup',async()=>{
 const f=await activeReactionSource(),q=()=>f.clients.gm.provider.reactionRestriction(f.targetActor),before=[...f.operations];
 f.game.combat={id:'viewed-other'};f.combat.turn=1;f.combatant.flags.pf2e.roundOfLastTurnEnd=4;assert.equal(q().status,'restricted');
 f.combat.round=5;f.combatant.flags.pf2e.roundOfLastTurnEnd=5;assert.equal(q().status,'clear');assert.deepEqual(f.operations,before);
 const capped=await activeReactionSource();capped.game.time.worldTime=700;assert.equal(capped.clients.gm.provider.reactionRestriction(capped.targetActor).status,'clear');
});
test('finite fallback preview retains strict six second boundary and never persists a new deadline',async()=>{
 const f=await activeReactionSource(),before=copy([...f.records.values()]),q=()=>f.clients.gm.provider.reactionRestriction(f.targetActor);
 f.game.combats.clear();f.game.time.worldTime=106;assert.equal(q().status,'manual');f.game.time.worldTime=107;assert.equal(q().status,'clear');assert.deepEqual([...f.records.values()],before);
});
test('reordered encounter and clock rewind are manual without query writes',async()=>{
 const f=await activeReactionSource();f.combat.turns.reverse();assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'manual');
 const g=await activeReactionSource();g.game.time.worldTime=99;assert.equal(g.clients.gm.provider.reactionRestriction(g.targetActor).status,'manual');
});
test('caster death, disconnected original caster and deleted spell do not repay or end an accepted source',async()=>{
 const f=await activeReactionSource();f.caster.isDead=true;f.caster.canAct=false;f.users.get('player').active=false;f.caster.items.delete(f.item.id);f.caster.items.delete(f.entry.id);f.caster.flags[ID].nativeCasts=[];
 assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'restricted');assert.equal(f.counts(),1);
});
test('actual owned parent proof restricts despite condition immunity and deliberate child removal',async()=>{
 const f=await activeReactionSource({realEffects:true,immunity:{spell:false,slowed:true,fascinated:false}},'criticalFailure');
 const r=f.effects.list(f.targetActor)[0];f.targetActor.items.delete(r.effects.children.fascinated.id);f.targetActor.items.set('other',{id:'other',type:'condition',system:{slug:'slowed',value:{value:2}}});
 assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'restricted');
 f.targetActor.items.delete(r.effects.parentId);assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'clear');
});
for(const lost of ['result','GM'])test(`confirmed parent deletion clears before the ${lost} continuity barrier and delete hook`,async()=>{
 const f=await activeReactionSource({realEffects:true}),r=f.effects.list(f.targetActor)[0],c=f.clients.player;
 if(lost==='result')await f.clients.gm.provider.onManual({sourceNonce:r.state.sourceNonce,reason:'reroll'});
 else {f.users.activeGM=null;await c.hookMap.get('userConnected')(f.users.get('gm'),false);}
 assert.equal(c.provider.reactionRestriction(f.targetActor).status,'manual');f.targetActor.items.delete(r.effects.parentId);
 const before=f.effects.get(f.targetActor,r.state.sourceNonce);assert.equal(c.provider.reactionRestriction(f.targetActor).status,'clear');assert.deepEqual(f.effects.get(f.targetActor,r.state.sourceNonce),before);
});
test('parent absence without confirmed operation stays manual; removed source cannot clear another restriction',async()=>{
 const f=await activeReactionSource(),r=[...f.records.values()][0],other=copy(r);other.state.sourceNonce='other';other.state.manualReview={reason:'reroll'};f.records.set('other',other);
 f.effects.inspectReactionParent=({nonce})=>nonce==='other'?{status:'removed'}:{status:'present'};
 assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'restricted');
 const g=await activeReactionSource({realEffects:true}),actual=g.effects.list(g.targetActor)[0];g.targetActor.items.delete(actual.effects.parentId);
 g.targetActor.flags[ID].roaringApplause.sources[actual.state.sourceNonce].effects.status='uncertain';assert.equal(g.clients.gm.provider.reactionRestriction(g.targetActor).status,'manual');
});
for(const seam of ['parent','save'])test(`unproven ${seam} evidence is manual and cannot mutate a source`,async()=>{
 const f=await activeReactionSource(),before=copy([...f.records.values()]);
 if(seam==='parent')f.effects.inspectReactionParent=()=>({status:'unproven',reason:'parent-rules-changed'});
 else f.clients.gm.saveEvidence.inspectResultContinuity=()=>({status:'unproven',reason:'row-changed'});
 assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'manual');assert.deepEqual([...f.records.values()],before);
});
test('missing or throwing read seams stay manual',async()=>{
 const f=await activeReactionSource();delete f.clients.gm.saveEvidence.inspectResultContinuity;assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'manual');
 f.clients.gm.saveEvidence.inspectResultContinuity=()=>{throw Error('missing source')};assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'manual');
});
test('sticky manual and reload continuity remain manual even with present parent and current row',async()=>{
 const f=await activeReactionSource();[...f.records.values()][0].state.manualReview={reason:'reroll'};assert.equal(f.clients.gm.provider.reactionRestriction(f.targetActor).status,'manual');
 const g=await activeReactionSource(),c=g.clients.player;c.provider.cleanup();c.provider.register({Hooks:c.Hooks,socket:c.socket});assert.equal(c.provider.reactionRestriction(g.targetActor).status,'manual');
});
for(const hook of ['userConnected','updateUser'])test(`${hook} synchronously preserves GM loss-and-return continuity barrier`,async()=>{
 const f=await activeReactionSource(),c=f.clients.player;assert.equal(c.provider.reactionRestriction(f.targetActor).status,'restricted');
 f.users.activeGM=null;const gone=c.hookMap.get(hook)(f.users.get('gm'),false);assert.equal(c.provider.reactionRestriction(f.targetActor).status,'manual');
 f.users.activeGM=f.users.get('gm');const returned=c.hookMap.get(hook)(f.users.get('gm'),true);assert.equal(c.provider.reactionRestriction(f.targetActor).status,'manual');await Promise.all([gone,returned]);assert.equal(c.provider.reactionRestriction(f.targetActor).status,'manual');
});
test('an older awaited lifecycle write cannot consume a newly raised GM continuity barrier',async()=>{
 const f=await activeReactionSource(),c=f.clients.gm,r=[...f.records.values()][0],save=f.effects.saveState;
 let enterFirst,enterSecond,releaseFirst,releaseSecond,calls=0;
 const first=new Promise(resolve=>enterFirst=resolve),second=new Promise(resolve=>enterSecond=resolve),firstReply=new Promise(resolve=>releaseFirst=resolve),secondWrite=new Promise(resolve=>releaseSecond=resolve);
 f.effects.saveState=async args=>{if(++calls===1){const saved=await save(args);enterFirst();await firstReply;return saved}if(calls===2){enterSecond();await secondWrite}return save(args)};
 f.game.time.worldTime=101;const older=c.provider.applyLifecycleEvent({actor:f.targetActor,nonce:r.state.sourceNonce,event:{type:'reconcile'}});await first;
 f.users.activeGM=null;await c.hookMap.get('userConnected')(f.users.get('gm'),false);f.users.activeGM=f.users.get('gm');const newer=c.hookMap.get('userConnected')(f.users.get('gm'),true);
 releaseFirst();await second;
 try{assert.equal(c.provider.reactionRestriction(f.targetActor).status,'manual')}finally{releaseSecond();await Promise.all([older,newer])}
 assert.equal(c.provider.reactionRestriction(f.targetActor).status,'manual');
});
test('reaction query aggregates restricted above manual above clear per independent source',async()=>{
 const f=await activeReactionSource(),r=[...f.records.values()][0],other=copy(r);other.state.sourceNonce='other';other.state.manualReview={reason:'unverified'};f.records.set('other',other);
 const q=()=>f.clients.gm.provider.reactionRestriction(f.targetActor);assert.equal(q().status,'restricted');assert.equal(q().sources.length,2);
 r.state.status='ended';assert.equal(q().status,'manual');other.state.status='ended';assert.equal(q().status,'clear');
});
test('wrong live actor, unsupported provider and malformed candidate cannot imply clear',async()=>{
 const f=await activeReactionSource(),q=a=>f.clients.gm.provider.reactionRestriction(a);assert.equal(q({...f.targetActor}).status,'manual');
 const r=[...f.records.values()][0];r.context.immunity.checked=false;assert.equal(q(f.targetActor).status,'manual');r.context.immunity.checked=true;
 f.targetActor.flags[ID]={roaringApplause:{sources:{broken:{schema:1}}}};assert.equal(q(f.targetActor).sources.find(s=>s.sourceNonce==='broken').status,'manual');
 f.clients.gm.provider.cleanup();assert.equal(q(f.targetActor).status,'manual');
});
function fixture({choose=async()=>({perception:'sees',lineOfEffectConfirmed:true}),immunity={spell:false,slowed:false,fascinated:false},iwr=true,mode='normal',realEffects=false}={}){
 const users=new Map([['gm',{id:'gm',active:true,isGM:true}],['player',{id:'player',active:true,isGM:false,targets:new Set()}]]);users.activeGM=users.get('gm');
 const caster={id:'caster',uuid:'Actor.caster',type:'character',isToken:false,canAct:true,isDead:false,flags:{[ID]:{nativeCasts:[]}},items:new Map(),testUserPermission:u=>['gm','player'].includes(u?.id)};
 const targetActor={id:'target',uuid:'Actor.target',type:'npc',isDead:false,flags:{},items:new Map(),isImmuneTo:item=>immunity[item.type==='spell'?'spell':item.system.slug],testUserPermission:u=>u?.id==='gm'};
 const entry={id:'entry',uuid:'Actor.caster.Item.entry',actor:caster,type:'spellcastingEntry',isSpontaneous:true,statistic:{dc:{value:22},withRollOptions:()=>({dc:{value:22}})},system:{prepared:{value:'spontaneous'},tradition:{value:'occult'},slots:{slot3:{value:2,max:2}}}};
 const item={id:'spell',uuid:'Actor.caster.Item.spell',sourceId:SOURCE,actor:caster,type:'spell',rank:3,spellcasting:entry,system:{location:{value:'entry',signature:true},level:{value:3},time:{value:'2'},range:{value:'60 feet'},duration:{sustained:true},rules:[],damage:{},defense:{save:{statistic:'will',basic:false}},traits:{value:['concentrate','emotion','manipulate','mental']}}};caster.items.set(item.id,item);caster.items.set(entry.id,entry);
 const scene={id:'scene',grid:{type:1,units:'ft'},tokens:new Map()};
 const token={id:'casterToken',uuid:'Scene.scene.Token.casterToken',documentName:'Token',parent:scene,actor:caster,elevation:0,level:0,object:{distanceTo:()=>30}},target={id:'targetToken',uuid:'Scene.scene.Token.targetToken',documentName:'Token',parent:scene,actor:targetActor,elevation:0,level:0,object:{}};
 scene.tokens.set(token.id,token);scene.tokens.set(target.id,target);caster.getActiveTokens=()=>[token];users.get('player').targets.add({document:target});users.get('gm').targets=new Set([{document:target}]);
 const combatant={id:'casterCombatant',actor:caster,token,initiative:20,flags:{pf2e:{roundOfLastTurnEnd:3}}},enemy={id:'enemyCombatant',actor:targetActor,token:target,initiative:20,flags:{pf2e:{roundOfLastTurnEnd:3}}};
 const combat={id:'combat',started:true,scene,round:4,turn:0,turns:[combatant,enemy]};
 const docs=new Map([caster,targetActor,entry,item,token,target].map(d=>[d.uuid,d])),messages=new Map(),clients={},operations=[],manual=[],errors=[],tracked=[],claps=[],records=new Map();let serial=0,castCount=0;
 const merge=(doc,changes)=>{for(const [path,value] of Object.entries(changes)){const parts=path.split('.');let o=doc;for(const k of parts.slice(0,-1))o=o[k]??={};const k=parts.at(-1);if(value&&typeof value==='object'&&!Array.isArray(value))merge(o[k]??={},value);else o[k]=copy(value);}};
 targetActor.update=async changes=>{merge(targetActor,changes);return targetActor};
 const makeItem=(data,id)=>{const d={...copy(data),id,uuid:`Actor.target.Item.${id}`,actor:targetActor,sourceId:data._stats?.compendiumSource};d.update=async changes=>{merge(d,changes);return d};targetActor.items.set(id,d);docs.set(d.uuid,d);return d};
 targetActor.createEmbeddedDocuments=async(_type,data)=>{const result=[];for(const source of data){const parent=makeItem(source,`parent${++serial}`);parent.flags.pf2e={itemGrants:{}};result.push(parent);for(const rule of parent.system.rules){const child=makeItem({type:'condition',system:{slug:rule.uuid.endsWith('xYTAsEpcJE1Ccni3')?'slowed':'fascinated'},flags:{pf2e:{grantedBy:{id:parent.id,onDelete:'cascade'}}},_stats:{compendiumSource:rule.uuid}},`child${++serial}`);parent.flags.pf2e.itemGrants[rule.flag]={id:child.id,onDelete:'detach'};result.push(child);}}if(mode==='lost-materialization')throw Error('native creation reply lost');return result};
 targetActor.deleteEmbeddedDocuments=async(_type,ids)=>{const removed=[];for(const id of ids){const parent=targetActor.items.get(id);if(!parent)continue;for(const grant of Object.values(parent.flags?.pf2e?.itemGrants??{})){const child=targetActor.items.get(grant.id);if(child?.flags?.pf2e?.grantedBy?.id===id){targetActor.items.delete(child.id);docs.delete(child.uuid);removed.push(child);}}targetActor.items.delete(id);docs.delete(parent.uuid);removed.push(parent);}return removed};
 const effects={inspectReactionParent:()=>({status:'present',reason:'parent-confirmed'}),list:a=>[...records.values()].filter(r=>r.state.source.targetActorUuid===a.uuid).map(copy),get:(a,n)=>records.has(n)&&records.get(n).state.source.targetActorUuid===a.uuid?copy(records.get(n)):null,
  claim:async({actor,state,context})=>{operations.push('claim');if(mode==='claim-veto')return null;const r={schema:1,revision:0,state:copy(state),context:copy(context),effects:{status:'not-started',parentId:null,children:{slowed:null,fascinated:null}}};records.set(state.sourceNonce,r);return copy(r)},
  saveState:async({actor,nonce,state,expectedRevision})=>{const r=records.get(nonce);assert.equal(r.revision,expectedRevision);r.state=copy(state);r.revision++;operations.push('save');return copy(r)},
  materialize:async({nonce})=>{operations.push('materialize');records.get(nonce).effects.status=records.get(nonce).context.immunity.spell?'immune':'created';},end:async({nonce})=>{operations.push('end');records.get(nonce).effects.status='ended';},endFascination:async()=>operations.push('end-fascination'),renew:async()=>operations.push('renew'),restoreFinite:async()=>operations.push('finite')};
 for(const uid of ['gm','player']){
  const game={world:{id:'ujx5r8oipw7ercdr'},system:{id:'pf2e',version:'8.5.1'},user:users.get(uid),users,actors:new Map([[caster.id,caster],[targetActor.id,targetActor]]),scenes:new Map([[scene.id,scene]]),combats:new Map([[combat.id,combat]]),messages,time:{worldTime:100},settings:{get:()=> 'public'},pf2e:{settings:{iwr},ConditionManager:{conditions:new Map(['slowed','fascinated'].map(slug=>[slug,{type:'condition',uuid:slug==='slowed'?'Compendium.pf2e.conditionitems.Item.xYTAsEpcJE1Ccni3':'Compendium.pf2e.conditionitems.Item.AdPVz7rbaVSRxHFg',system:{slug}}]))}}};game.scenes.current=scene;
  if(realEffects&&uid==='gm')Object.assign(effects,createRoaringEffects({game,fromUuid:async u=>docs.get(u),randomId:()=>`op${++serial}`}));
  const handlers=new Map(),hookMap=new Map(),adapters=new Map();
  const socket={register:(n,fn)=>handlers.set(n,fn),executeAsUser:async(n,to,p)=>clients[to].handlers.get(n).call({socketdata:{userId:uid}},copy(p)),executeForEveryone:async(n,p)=>Promise.all(Object.values(clients).map(c=>c.handlers.get(n).call({socketdata:{userId:uid}},copy(p))))};
  const Hooks={on:(n,fn)=>{hookMap.set(n,fn);return n},off:n=>hookMap.delete(n)};
  const nativeCasts={addInvocationAdapter:(k,v)=>adapters.set(k,v)};
  const saveEvidence={inspectResultContinuity:()=>({status:'current',reason:null}),register:()=>{},track:m=>{tracked.push([uid,m.uuid]);return true},cleanup:()=>{}};
  const provider=createRoaringApplause({game,fromUuid:async u=>docs.get(u),nativeCasts,effects,choose,saveEvidence,onError:e=>errors.push(e),onManual:e=>manual.push(e),onClap:e=>claps.push(e),randomId:()=>`nonce${++serial}`});provider.register({Hooks,socket});clients[uid]={provider,game,handlers,hookMap,adapters,Hooks,socket,saveEvidence};
 }
 const next=async()=>{castCount++;return 'passthrough'};
 next.withOutcome=async invocation=>{
  castCount++;const data=invocation.data,input={actorUuid:caster.uuid,itemUuid:item.uuid,entryUuid:entry.uuid,sourceId:SOURCE,rank:3,slotId:null,focusPoints:0,overlayIds:[],messageMode:'public'},castNonce=`cast${serial}`,inv={...invocation,gmId:'gm'};
  await clients.gm.adapters.get('roaring-applause').validate({actor:caster,item,entry,user:users.get('player'),castNonce,payload:{...input,id:castNonce},invocation:inv});
  if(mode==='veto')throw Error('native slot veto');
  const before=entry.system.slots.slot3.value;entry.system.slots.slot3.value--;
  if(mode==='disrupted')return {status:'disrupted',nativeResult:false};
  const m={id:`message${serial}`,uuid:`ChatMessage.message${serial}`,documentName:'ChatMessage',author:users.get('player'),speaker:{actor:caster.id,scene:scene.id,token:token.id},rolls:[],blind:false,whisper:[],flags:{pf2e:{origin:{uuid:item.uuid,actor:caster.uuid,castRank:3}},[ID]:{nativeCast:{id:castNonce,actorUuid:caster.uuid,itemUuid:item.uuid,userId:'player'},nativeCastInput:copy(input)},'pf2e-toolbelt':{targetHelper:{targets:[target.uuid]}}}};
  m.update=async changes=>{if(mode==='card-veto')return m;for(const [path,v]of Object.entries(changes)){if(path===`flags.${ID}.roaringSource`)m.flags[ID].roaringSource=copy(v);}return m};messages.set(m.id,m);docs.set(m.uuid,m);
  const receipt={...input,id:castNonce,state:'used',userId:'player',messageId:m.id,completedWorldTime:100,invocation:copy(inv),nativeCastScope:{castNonce,tokenUuid:token.uuid},slotCommit:{castNonce,actorUuid:caster.uuid,itemUuid:item.uuid,entryUuid:entry.uuid,userId:'player',gmId:'gm',rank:3,before,after:before-1,cost:1}};caster.flags[ID].nativeCasts.push(receipt);
  if(mode==='missing-time')delete receipt.completedWorldTime;
  if(mode==='wrong-payment')receipt.slotCommit.after=before;
  if(mode==='turn-changed')combat.turn=1;
  if(mode==='gm-changed')users.activeGM=users.get('player');
  return {status:'completed',castNonce,input,receipt:copy(receipt),message:m,nativeResult:undefined,completedWorldTime:100};
 };
 const run=()=>clients.player.provider.interceptCast({item,entry,options:{rank:3,slotId:NaN}},next);
 return {run,next,clients,game:clients.gm.game,users,caster,targetActor,item,entry,token,target,combat,combatant,enemy,records,effects,operations,tracked,errors,manual,claps,messages,counts:()=>castCount};
}
test('message source queries skip unrelated actors after indexing and retain ended records',async()=>{
 const f=await activeReactionSource(),p=f.clients.gm.provider,m=[...f.messages.values()][0];
 assert.equal(p.listSources({message:m}).length,1);let reads=0;const list=f.effects.list;f.effects.list=a=>{reads++;return list(a)};
 for(let n=0;n<1000;n++)assert.deepEqual(p.listSources({message:{uuid:`ChatMessage.other${n}`}}),[]);
 assert.equal(reads,0);[...f.records.values()][0].state.status='ended';assert.equal(p.listSources({message:m})[0].record.state.status,'ended');assert.equal(reads,1);
});
test('actor source queries do not inspect other world or scene actors',async()=>{
 const f=await activeReactionSource(),p=f.clients.gm.provider;const seen=[],list=f.effects.list;f.effects.list=a=>{seen.push(a.uuid);return list(a)};
 assert.deepEqual(p.listSources({actor:f.caster}),[]);assert.deepEqual(seen,[f.caster.uuid]);
});
test('actor updates remove old message index entries and enroll new ones, including unmarked original cards',async()=>{
 const f=await activeReactionSource(),c=f.clients.gm,m=[...f.messages.values()][0];c.provider.listSources({message:m});
 const row=[...f.records.values()][0];row.state.source.originalMessageUuid='ChatMessage.rebound';await c.hookMap.get('updateActor')(f.targetActor);
 assert.deepEqual(c.provider.listSources({message:m}),[]);assert.equal(c.provider.listSources({message:{uuid:'ChatMessage.rebound'}}).length,1);
});
test('message index preserves duplicate-source ambiguity and revalidates current actor identity',async()=>{
 const f=await activeReactionSource(),p=f.clients.gm.provider,m=[...f.messages.values()][0];p.listSources({message:m});
 const second=copy([...f.records.values()][0]);second.state.sourceNonce='second';f.records.set('second',second);assert.equal(p.listSources({message:m}).length,2);
 const replacement={...f.targetActor};f.game.actors.set(replacement.id,replacement);f.target.actor=replacement;
 assert.equal(p.listSources({message:m})[0].actor,replacement);f.game.actors.delete(replacement.id);f.target.actor=null;assert.deepEqual(p.listSources({message:m}),[]);
});
test('structural token changes invalidate message membership, pure movement does not scan sources',async()=>{
 const f=await activeReactionSource(),c=f.clients.gm,m=[...f.messages.values()][0];c.provider.listSources({message:m});let reads=0;const list=f.effects.list;f.effects.list=a=>{reads++;return list(a)};
 await c.hookMap.get('updateToken')(f.target,{x:5,_movementHistory:[]});assert.deepEqual(c.provider.listSources({message:{uuid:'ChatMessage.other'}}),[]);assert.equal(reads,0);
 const r=[...f.records.values()][0];r.state.source.originalMessageUuid='ChatMessage.new';await c.hookMap.get('updateToken')(f.target,{actorLink:false});assert.equal(c.provider.listSources({message:{uuid:'ChatMessage.new'}}).length,1);
});
test('base actor source updates also refresh inherited unlinked synthetic memberships',async()=>{
 const f=fixture(),c=f.clients.gm,m={uuid:'ChatMessage.inherited'};assert.deepEqual(c.provider.listSources({message:m}),[]);
 const synthetic={...f.targetActor,uuid:'Scene.scene.Token.unlinked.Actor.target',isToken:true},token={id:'unlinked',actor:synthetic},preview={...synthetic};f.target.parent.tokens.set(token.id,token);
 // Core registers unpersisted TokenConfig previews as dependents too. A stale
 // preview can have the same UUID as the real actor but different source rows.
 f.targetActor.getDependentTokens=({concreteOnly}={})=>concreteOnly?[token]:[token,{id:token.id,actor:preview}];
 const list=f.effects.list;f.effects.list=a=>a===preview?[]:list(a===synthetic?f.targetActor:a);
 await f.run();[...f.records.values()][0].state.source.originalMessageUuid=m.uuid;await c.hookMap.get('updateActor')(f.targetActor);
 assert.deepEqual(c.provider.listSources({message:m}).map(x=>x.actor.uuid).sort(),[f.targetActor.uuid,synthetic.uuid].sort());
 f.records.clear();await c.hookMap.get('updateActor')(f.targetActor);assert.deepEqual(c.provider.listSources({message:m}),[]);
});
test('one real enrollment creates paid source and tracks card on both clients',async()=>{const f=fixture();await f.run();assert.equal(f.counts(),1);assert.equal(f.entry.system.slots.slot3.value,1);assert.equal(f.records.size,1);assert.equal(f.tracked.length,2);const r=[...f.records.values()][0];assert.equal(r.state.completedWorldTime,100);assert.equal(r.state.hardStopAt,700);assert.equal(r.context.userId,'player');assert.equal(f.clients.gm.provider.lookupSource([...f.messages.values()][0]).castNonce,r.state.source.castNonce);});
test('cancel performs no native cast or claim',async()=>{const f=fixture({choose:async()=>null});await f.run();assert.equal(f.counts(),0);assert.equal(f.records.size,0);});
test('unrelated spell keeps original continuation and no enrollment',async()=>{const f=fixture();f.item.sourceId='other';assert.equal(await f.run(),'passthrough');assert.equal(f.records.size,0);});
for(const mode of ['veto','missing-time','wrong-payment','turn-changed','gm-changed'])test(`${mode} never creates a managed source or retries`,async()=>{const f=fixture({mode});await assert.rejects(f.run());assert.equal(f.counts(),1);assert.equal(f.records.size,0);});
test('disrupted native cast creates no source',async()=>{const f=fixture({mode:'disrupted'});await f.run();assert.equal(f.counts(),1);assert.equal(f.records.size,0);});
test('unknown IWR does not mistake native false for no immunity',async()=>{const f=fixture({iwr:false});await assert.rejects(f.run());assert.equal(f.records.size,0);assert.equal(f.entry.system.slots.slot3.value,2);});
test('whole spell immunity is a proven fact, not a fabricated save',async()=>{const f=fixture({immunity:{spell:true,slowed:false,fascinated:false}});await f.run();assert.equal([...f.records.values()][0].context.immunity.spell,true);assert.equal([...f.records.values()][0].state.result,null);});
test('claim veto stops without card marker/retry',async()=>{const f=fixture({mode:'claim-veto'});await assert.rejects(f.run());assert.equal(f.counts(),1);assert.equal(f.records.size,0);});
test('card marker write veto never announces or tracks',async()=>{const f=fixture({mode:'card-veto'});await assert.rejects(f.run());assert.equal(f.tracked.length,0);assert.equal(f.counts(),1);assert.ok([...f.records.values()][0].state.manualReview);});
test('verified save materializes once; unverified revision stays manual',async()=>{const f=fixture();await f.run();const r=[...f.records.values()][0],p=f.clients.gm.provider;const event={sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'failure',revision:1,proof:{invocationId:'save1'}};await p.onVerified(event);await p.onVerified(event);assert.equal(f.operations.filter(o=>o==='materialize').length,1);await p.onManual({...event,reason:'reroll'});assert.ok([...f.records.values()][0].state.manualReview);});
test('caster next end expires when another combat is viewed',async()=>{const f=fixture();await f.run();f.game.combat={id:'unrelated'};f.combat.turn=1;f.combatant.flags.pf2e.roundOfLastTurnEnd=4;await f.clients.gm.provider.reconcile();assert.equal(f.operations.filter(o=>o==='end').length,0);f.combat.round=5;f.combatant.flags.pf2e.roundOfLastTurnEnd=5;await f.clients.gm.provider.reconcile();await f.clients.gm.provider.reconcile();assert.equal(f.operations.filter(o=>o==='end').length,1);});
test('caster death preserves known deadline while clock cap still ends',async()=>{const f=fixture();await f.run();f.caster.canAct=false;f.caster.isDead=true;await f.clients.gm.provider.reconcile();assert.equal(f.operations.includes('end'),false);f.game.time.worldTime=700;await f.clients.gm.provider.reconcile();assert.equal(f.operations.filter(o=>o==='end').length,1);});

test('second same-target source cannot spend a slot',async()=>{const f=fixture();await f.run();await assert.rejects(f.run());assert.equal(f.entry.system.slots.slot3.value,1);assert.equal(f.records.size,1);});
test('after-native source incapacitation does not invalidate already completed casting',async()=>{const f=fixture();const original=f.next.withOutcome;f.next.withOutcome=async data=>{const out=await original(data);f.caster.canAct=false;return out};await f.run();assert.equal(f.records.size,1);});
test('forged source notice from player cannot enroll old card',async()=>{const f=fixture();await f.run();const fn=f.clients.gm.handlers.get('roaring-applause:notice');const before=f.tracked.length;const response=await fn.call({socketdata:{userId:'player'}},{messageUuid:[...f.messages.values()][0].uuid});assert.equal(response.ok,false);assert.equal(f.tracked.length,before);});
test('source receipt rollback before claiming prevents creation',async()=>{const f=fixture();const original=f.next.withOutcome;f.next.withOutcome=async data=>{const out=await original(data);f.caster.flags[ID].nativeCasts=[];return out};await assert.rejects(f.run());assert.equal(f.records.size,0);assert.equal(f.counts(),1);});
test('source deletion uses frozen finite fallback, never now plus one round',async()=>{const f=fixture();await f.run();f.game.combats.clear();f.game.time.worldTime=102;await f.clients.gm.provider.reconcile();assert.equal([...f.records.values()][0].state.timing.finiteEnvelope.start.value,100);assert.equal(f.operations.filter(o=>o==='finite').length,1);f.game.time.worldTime=107;await f.clients.gm.provider.reconcile();assert.equal(f.operations.filter(o=>o==='end').length,1);});
test('changed target immunity before save becomes manual without materialization',async()=>{const im={spell:false,slowed:false,fascinated:false},f=fixture({immunity:im});await f.run();im.spell=true;const r=[...f.records.values()][0];await f.clients.gm.provider.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'failure',revision:1,proof:{invocationId:'save1'}});assert.ok([...f.records.values()][0].state.manualReview);assert.equal(f.operations.includes('materialize'),false);});

test('logical ended source observes unfinished cleanup without replaying Cast',async()=>{const f=fixture();await f.run();let calls=0;const original=f.effects.end;f.effects.end=async args=>{if(++calls===1)throw Error('delete response lost');return original(args)};f.game.time.worldTime=700;await assert.rejects(f.clients.gm.provider.reconcile());assert.equal([...f.records.values()][0].state.status,'ended');await f.clients.gm.provider.reconcile();assert.equal([...f.records.values()][0].effects.status,'ended');assert.equal(f.counts(),1);assert.equal(calls,2);});
test('uncertain materialization without parent stays manual and never creates again',async()=>{const f=fixture();await f.run();const r=[...f.records.values()][0];r.state.status='active';r.state.result={revision:1,receiptId:'save1',outcome:'failure'};r.effects.status='uncertain';f.effects.materialize=async()=>{throw Error('no unique original operation')};await f.clients.gm.provider.reconcile();await f.clients.gm.provider.reconcile();assert.ok(r.state.manualReview);assert.equal(f.operations.includes('materialize'),false);assert.equal(f.manual.length,1);});

test('actual effects store and reducer integrate paid source, own grants and next-end cleanup',async()=>{const f=fixture({realEffects:true});f.targetActor.items.set('other',{id:'other',type:'condition',system:{slug:'slowed',value:{value:2}}});await f.run();let r=f.effects.list(f.targetActor)[0];assert.equal(r.state.status,'awaiting-save');await f.clients.gm.provider.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'criticalFailure',revision:1,proof:{invocationId:'nativeSave'}});r=f.effects.get(f.targetActor,r.state.sourceNonce);assert.equal(r.effects.status,'created');assert.ok(r.effects.children.slowed);assert.ok(r.effects.children.fascinated);assert.equal(f.targetActor.items.size,4);f.combat.round=5;f.combatant.flags.pf2e.roundOfLastTurnEnd=5;await f.clients.gm.provider.reconcile();r=f.effects.get(f.targetActor,r.state.sourceNonce);assert.equal(r.state.status,'ended');assert.equal(r.effects.status,'ended');assert.deepEqual([...f.targetActor.items.keys()],['other']);assert.equal(f.counts(),1);});

test('one uncertain cleanup cannot block another source deadline',async()=>{const f=fixture();await f.run();const first=[...f.records.values()][0],second=copy(first);second.state.sourceNonce='secondSource';f.records.set('secondSource',second);const firstNonce=first.state.sourceNonce,original=f.effects.end;f.effects.end=async args=>{if(args.nonce===firstNonce)throw Error('uncertain old delete');return original(args)};f.game.time.worldTime=700;await assert.rejects(f.clients.gm.provider.reconcile());assert.equal(f.records.get('secondSource').state.status,'ended');assert.equal(f.records.get('secondSource').effects.status,'ended');});

test('same-slug foreign condition cannot attest native immunity',async()=>{const f=fixture();f.game.pf2e.ConditionManager.conditions.get('slowed').uuid='Compendium.foreign.Item.slowed';await assert.rejects(f.run());assert.equal(f.entry.system.slots.slot3.value,2);assert.equal(f.records.size,0);});

test('late source actor broadcast enrolls after the card marker broadcast',async()=>{const f=fixture();await f.run();f.tracked.length=0;const get=f.effects.get;f.effects.get=()=>null;f.clients.player.hookMap.get('updateChatMessage')([...f.messages.values()][0]);assert.equal(f.tracked.length,0);f.effects.get=get;f.clients.player.hookMap.get('updateActor')(f.targetActor);assert.deepEqual(f.tracked,[['player',[...f.messages.values()][0].uuid]]);});

test('actual lost-create receipt with null parentId observes the unique native operation',async()=>{const f=fixture({realEffects:true,mode:'lost-materialization'});await f.run();let r=f.effects.list(f.targetActor)[0];await assert.rejects(f.clients.gm.provider.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'failure',revision:1,proof:{invocationId:'save1'}}));r=f.effects.get(f.targetActor,r.state.sourceNonce);assert.equal(r.effects.parentId,null);assert.equal(f.targetActor.items.size,2);await f.clients.gm.provider.reconcile();r=f.effects.get(f.targetActor,r.state.sourceNonce);assert.equal(r.effects.status,'created');assert.ok(r.effects.parentId);assert.equal(f.targetActor.items.size,2);assert.equal(f.counts(),1);});

test('Sustain source enumeration exposes live actors and isolated state snapshots',async()=>{const f=fixture();await f.run();const listed=f.clients.gm.provider.listSources();assert.equal(listed[0].actor,f.targetActor);listed[0].record.state.status='ended';assert.equal([...f.records.values()][0].state.status,'awaiting-save');});

test('a second live encounter for the source token falls back to frozen finite timing',async()=>{
 const f=fixture();await f.run();f.game.combats.set('second',{...f.combat,id:'second'});
 await f.clients.gm.provider.reconcile();assert.equal([...f.records.values()][0].state.timing.mode,'manual-finite');assert.equal(f.operations.filter(o=>o==='finite').length,1);
});
test('native priority maps remain exact during subsequent reconciliation',async()=>{
 const f=fixture();f.combatant.flags.pf2e.overridePriority={20:0};f.enemy.flags.pf2e.overridePriority={20:1};await f.run();
 await f.clients.gm.provider.reconcile();assert.equal([...f.records.values()][0].state.timing.mode,'exact');assert.equal(f.operations.includes('finite'),false);
});
test('a reloaded source cannot assume an uninterrupted turn epoch',async()=>{
 const f=fixture();await f.run();const client=f.clients.gm;client.provider.cleanup();client.provider.register({Hooks:client.Hooks,socket:client.socket});
 await client.provider.reconcile();const state=[...f.records.values()][0].state;assert.equal(state.timing.mode,'manual-finite');assert.equal(state.timing.finiteEnvelope.start.value,100);
});
test('a save arriving during readiness cannot materialize a reloaded unverified epoch',async()=>{
 const f=fixture();await f.run();const client=f.clients.gm,r=[...f.records.values()][0];client.provider.cleanup();client.provider.register({Hooks:client.Hooks,socket:client.socket});
 await client.provider.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'failure',revision:1,proof:{invocationId:'save1'}});
 assert.equal(f.operations.includes('materialize'),false);assert.equal([...f.records.values()][0].state.timing.mode,'manual-finite');
});
test('a persisted actual target start prompts once after failure, with a receipt saved first',async()=>{
 const f=fixture();await f.run();const r=[...f.records.values()][0],p=f.clients.gm.provider;
 await p.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'failure',revision:1,proof:{invocationId:'save1'}});
 f.combat.turn=1;f.enemy.flags.pf2e.roundOfLastTurn=4;
 await f.clients.gm.hookMap.get('updateCombatant')(f.enemy,{'flags.pf2e.roundOfLastTurn':4},{},'gm');
 assert.equal(f.claps.length,1);assert.equal(Object.keys([...f.records.values()][0].state.clapReceipts).length,1);
 await f.clients.gm.hookMap.get('updateCombatant')(f.enemy,{'flags.pf2e.roundOfLastTurn':4},{},'gm');assert.equal(f.claps.length,1);
});
test('a start hook or a change without a persisted native start flag cannot prompt clap',async()=>{
 const f=fixture();await f.run();const r=[...f.records.values()][0],p=f.clients.gm.provider;
 await p.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'failure',revision:1,proof:{invocationId:'save1'}});
 f.combat.turn=1;await f.clients.gm.hookMap.get('pf2e.startTurn')(f.enemy,f.combat,'gm');
 await f.clients.gm.hookMap.get('updateCombatant')(f.enemy,{'flags.pf2e.roundOfLastTurn':4},{},'gm');assert.equal(f.claps.length,0);
});
test('whole-spell immunity suppresses the target-start clap despite a failed save',async()=>{
 const f=fixture({immunity:{spell:true,slowed:false,fascinated:false}});await f.run();const r=[...f.records.values()][0];
 await f.clients.gm.provider.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'failure',revision:1,proof:{invocationId:'save1'}});
 f.combat.turn=1;f.enemy.flags.pf2e.roundOfLastTurn=4;
 await f.clients.gm.hookMap.get('updateCombatant')(f.enemy,{'flags.pf2e.roundOfLastTurn':4},{},'gm');assert.equal(f.claps.length,0);
});
test('an unrelated combatant update cannot consume a start while reconciliation awaits cleanup',async()=>{
 const f=fixture();await f.run();const r=[...f.records.values()][0],p=f.clients.gm.provider;
 await p.onVerified({sourceNonce:r.state.sourceNonce,originalMessageUuid:r.state.source.originalMessageUuid,targetUuid:f.target.uuid,castNonce:r.state.source.castNonce,adjustedOutcome:'failure',revision:1,proof:{invocationId:'save1'}});
 const old=copy(r);old.state.sourceNonce='oldSource';old.state.status='ended';old.effects.status='created';f.records.set('oldSource',old);
 let signal,release;const entered=new Promise(resolve=>signal=resolve),reply=new Promise(resolve=>release=resolve),end=f.effects.end;
 f.effects.end=async args=>{const value=await end(args);if(args.nonce==='oldSource'){signal();await reply;}return value;};
 f.combat.turn=1;f.enemy.flags.pf2e.roundOfLastTurn=4;const hook=f.clients.gm.hookMap.get('updateCombatant');
 const pending=hook(f.enemy,{'flags.pf2e.roundOfLastTurn':4},{},'gm');await entered;
 await hook(f.enemy,{resource:1},{},'gm');release();await pending;
 assert.equal(f.claps.length,1);assert.equal(Object.keys(f.records.get(r.state.sourceNonce).state.clapReceipts).length,1);
});
