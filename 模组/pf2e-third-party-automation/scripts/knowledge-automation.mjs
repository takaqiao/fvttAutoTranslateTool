import {MODULE_ID,hasSource,levelDC} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {isActiveGM,resolveMessageTargets,upsertOwnedEffect} from './native-context.mjs';
import {withEatStrikeFrame,isEatFortuneProbe} from './eat-fortune.mjs';

export const KNOWLEDGE_SOURCES=Object.freeze({
 monster:'Compendium.pf2e.feats-srd.Item.YTTJqRKH8QZl6al2',known:'Compendium.pf2e.feats-srd.Item.iWvpq3uDZcXvBJj8',stance:'Compendium.pf2e.feats-srd.Item.fJwsZM6WXwP8EStV',
 hunt:'Compendium.pf2e.actionspf2e.Item.JYi4MnsdFu618hPm',devise:'Compendium.pf2e.actionspf2e.Item.m0f2B7G9eaaTmhFL',
 prey:'Compendium.pf2e-ranged-combat.effects.Item.rdLADYwOByj8AZ7r',monsterEffect:'Compendium.pf2e.feat-effects.Item.W2tWq0gdAcnoz2MO',knownEffect:'Compendium.pf2e.feat-effects.Item.DvyyA11a63FBwV7x',
 stanceEffect:'Compendium.pf2e.feat-effects.Item.z6oLNlBs724PCcR6',deviseEffect:'Compendium.pf2e.feat-effects.Item.XQpTyjXFYYNexyOk',
});
const values=x=>Array.from(x?.values?.()??x?.contents??x??[]),own=x=>x?.flags?.[MODULE_ID]?.knowledge??{};
const has=(a,k)=>values(a?.items).some(i=>hasSource(i,KNOWLEDGE_SOURCES[k])),feature=(a,k)=>values(a.items).find(i=>hasSource(i,KNOWLEDGE_SOURCES[k]));
const option=`${MODULE_ID}:knowledge`,clone=structuredClone,tokenDoc=t=>t?.document??t;
const degree=m=>m?.rolls?.[0]?.options?.degreeOfSuccess??({criticalFailure:0,failure:1,success:2,criticalSuccess:3}[m?.flags?.pf2e?.context?.outcome]);
const options=m=>values(m?.flags?.pf2e?.context?.options);
const random=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID().replaceAll('-','').slice(0,16);

/** Preserve all unrelated/custom rules. Negate the exact already-required predicate. */
export function buildKnowledgePatreonRepairs(original){
 const rules=clone(original),changes=[];
 for(const[id,group]of Object.entries(rules??{}))for(const list of ['baseRules','handlerRules'])for(const[index,rule]of(group[list]??[]).entries()){
  if(!Array.isArray(rule.predicate))continue;let gate;
  if(group.source?.length===0&&rule.triggerType==='self-effect'&&rule.value==='applySelfEffect'){
   const exclusion={not:{and:[`origin:item:sourceId:${KNOWLEDGE_SOURCES.devise}`,'feat:known-weaknesses']}};
   if(!rule.predicate.some(p=>JSON.stringify(p)===JSON.stringify(exclusion))){const before=clone(rule.predicate);rule.predicate.push(exclusion);changes.push({path:`${id}.${list}.${index}.predicate`,before,after:clone(rule.predicate),reason:'精确Devise统一结算后不再自动建立第二份自效与d20'});}
   continue;
  }
  if(group.source?.includes(KNOWLEDGE_SOURCES.monster)&&rule.triggerType==='skill-check'&&rule.value==='monsterHunter'&&rule.predicate.includes('feat:monster-hunter'))gate='feat:monster-hunter';
  if(group.source?.includes(KNOWLEDGE_SOURCES.known)&&rule.triggerType==='skill-check'&&rule.value===KNOWLEDGE_SOURCES.knownEffect&&rule.predicate.includes('feat:known-weaknesses'))gate='feat:known-weaknesses';
  if(group.source?.includes(KNOWLEDGE_SOURCES.known)&&rule.triggerType==='postInfo'&&rule.value==='knownWeaknesses'&&rule.predicate.includes('origin:item:known-weaknesses'))gate='origin:item:known-weaknesses';
  if(!gate||rule.predicate.some(p=>p?.not===gate))continue;const before=clone(rule.predicate);rule.predicate.push({not:gate});
  changes.push({path:`${id}.${list}.${index}.predicate`,before,after:clone(rule.predicate),reason:'以精确检定来源、目标及受益者替代重复知识联动'});
 }
 return {rules,changes};
}

export function buildStrategistAttackEffect({targetUuid,claim,sourceActor}){
 if(!/^Scene\.[^.]+\.Token\.[^.]+$/.test(targetUuid)||!claim)throw Error('军师架势需要明确Token及攻击回执。');
 return {type:'effect',name:'军师架势：下一次攻击',img:'icons/sundries/gaming/chess-knight-white-glass.webp',system:{description:{value:''},level:{value:1},traits:{value:[]},duration:{unit:'unlimited',value:-1,expiry:null,sustained:false},tokenIcon:{show:false},rules:[
  {key:'TokenMark',slug:`strategist-${claim.toLowerCase()}`,uuid:targetUuid},
  // PF2e evaluates effects on the target clone with that token's marks as self:mark.
  {key:'EphemeralEffect',selectors:['strike-attack-roll','spell-attack-roll'],uuid:'Compendium.pf2e.conditionitems.Item.AJh5ex99aV6VTggg',predicate:[`self:mark:strategist-${claim.toLowerCase()}`,`${option}:claim:${claim}`]},
 ]},flags:{[MODULE_ID]:{knowledge:{kind:'strategist-claim',claim,sourceActor,targetUuid}}}};
}

export function createKnowledgeAutomation({game,fromUuid=globalThis.fromUuid,choose,strikeMiddleware=null,spellAttackMiddleware=null,onError=console.error}={}){
 const queue=new SerialActions();let socket,registered=false;
 const gm=()=>{if(!isActiveGM(game))throw Error('知识联动必须由当前主GM结算。')};
 const actors=()=>[...new Map([...values(game.actors),...values(game.scenes).flatMap(s=>values(s.tokens).map(t=>t.actor))].filter(Boolean).map(a=>[a.uuid,a])).values()];
 const save=(doc,key,data)=>doc.update({[`flags.${MODULE_ID}.knowledge.${key}`]:data});
 const now=()=>game.time.worldTime??0,turn=()=>game.combat?.started?`${game.combat.id}:${game.combat.round}:${game.combat.turn}`:null;
 const states=actor=>own(actor).strategistStates??(own(actor).strategist?[own(actor).strategist]:[]);
 const saveStates=(actor,list)=>{gm();return actor.update({[`flags.${MODULE_ID}.knowledge.strategistStates`]:list,[`flags.${MODULE_ID}.knowledge.strategist`]:list.at(-1)??null});};
 const saveState=(actor,state)=>saveStates(actor,states(actor).map(s=>s.id===state.id?state:s));
 const pick=async(actor,user,title,choices)=>{const selected=choices.length===1?choices[0].value:await choose({actor,user,title,choices});if(selected!=null&&!choices.some(c=>c.value===selected))throw Error('无效的规则选择。');return selected};
 const userFor=m=>m.author??game.users.get(m.user?.id??m.user);
 const sourceToken=(actor,message)=>{
  const ts=values(actor.getActiveTokens?.(true,true)).map(tokenDoc),id=message?.speaker?.token,scene=message?.speaker?.scene;
  return ts.find(t=>t.id===id&&t.parent?.id===scene)??(ts.length===1?ts[0]:null);
 };
 const load=async uuid=>{const item=await fromUuid(uuid);if(!item?.toObject)throw Error('知识联动的原生效果不可用。');const data=item.toObject();delete data._id;return data};
 const resolveAction=item=>hasSource(item,KNOWLEDGE_SOURCES.stance)?'knowledge:stance':hasSource(item,KNOWLEDGE_SOURCES.hunt)&&has(item.actor,'monster')?'knowledge:hunt':hasSource(item,KNOWLEDGE_SOURCES.devise)&&has(item.actor,'known')?'knowledge:devise':null;
 const prey=(actor,target)=>values(actor.items).some(i=>hasSource(i,KNOWLEDGE_SOURCES.prey)&&!i.isExpired&&i.system?.rules?.some(r=>r.key==='TokenMark'&&r.slug==='hunted-prey'&&r.uuid===target.uuid))
  ||target.actor.getRollOptions?.(['all'])?.includes(`self:prey:${actor.signature}`);
 function timing(actor){const c=game.combat,index=c?.turns?.findIndex(t=>t.actor?.uuid===actor.uuid)??-1;if(!c?.started||index<0)return null;return {combatId:c.id,combatantId:c.turns[index].id,round:c.round+(index<=c.turn?1:0),initiative:c.turns[index].initiative,rounds:index<=c.turn?1:0};}
 function expired(t){if(!t)return false;const c=game.combat,i=c?.turns?.findIndex(x=>x.id===t.combatantId)??-1;return !c?.started||c.id!==t.combatId||i<0||c.round>t.round||c.round===t.round&&c.turn>=i;}
 function origin(data,actor,item,target,t=null){
  data.system.context={origin:{actor:actor.uuid,item:item?.uuid??null,token:sourceToken(actor)?.uuid??null,rollOptions:item?.getOriginData?.().rollOptions??[]},target:{actor:target.actor.uuid,token:target.uuid},roll:null};
  data.system.start={value:now(),initiative:t?.initiative??null};if(t)data.system.duration={value:t.rounds,unit:'rounds',expiry:'turn-start',sustained:false};return data;
 }
 async function recipients(actor,user,message){
  const source=sourceToken(actor,message),result=[actor];if(!source)return result;
  const candidates=values(source.parent?.tokens).filter(t=>t.actor&&t.actor.uuid!==actor.uuid&&t.actor.isAllyOf?.(actor)&&(!t.hidden||user.isGM));
  const available=[...new Map(candidates.map(t=>[t.actor.uuid,t])).values()];
  while(available.length){const selected=await pick(actor,user,'选择已告知信息的盟友',[...available.map(t=>({value:t.uuid,label:t.name??t.actor.name})),{value:'done',label:'完成，不再添加'}]);if(!selected||selected==='done')break;const index=available.findIndex(t=>t.uuid===selected);result.push(available.splice(index,1)[0].actor);}
  return result;
 }
 async function benefit(kind,actor,recipient,target,message){
  const t=kind==='known'?timing(actor):null;if(kind==='known'&&!t)return;
  const days=own(recipient).days??[],creature=target.actor.uuid;
  if(kind==='monster'&&days.some(r=>r.target===creature))return;
  const data=origin(await load(KNOWLEDGE_SOURCES[`${kind}Effect`]),actor,feature(actor,kind),target,t),slug=kind==='known'?'known-weaknesses':'monster-hunter';
  for(const rule of data.system.rules){if(rule.key==='TokenMark'){rule.slug=slug;rule.uuid=target.uuid;}if(rule.key==='FlatModifier'){rule.predicate=[`target:mark:${slug}`];rule.selector='attack-roll';}}
  data.flags??={};data.flags[MODULE_ID]={...data.flags[MODULE_ID],knowledge:{kind,sourceActor:actor.uuid,targetUuid:target.uuid,sourceMessageId:message.id,timing:t}};
  if(kind==='monster')await save(recipient,'days',[...days,{target:creature,day:own(recipient).dailyEpoch??0}]);
  await upsertOwnedEffect(recipient,`knowledge:${kind}:${actor.uuid}:${target.uuid}`,data);
 }
 async function preciseOrigin(message,actor,target){
  const prefix=`${option}:recall:`,ids=options(message).filter(o=>typeof o==='string'&&o.startsWith(prefix)).map(o=>o.slice(prefix.length));if(ids.length!==1)return null;
  const source=game.messages.get(ids[0]),state=own(source).recall;if(!state||state.actorUuid!==actor.uuid||state.targetUuid!==target.uuid||state.usedBy&&state.usedBy!==message.id)return null;
  const item=actor.items.get?.(state.itemId)??values(actor.items).find(i=>i.id===state.itemId);
  if(resolveAction(item)!==state.action||!['knowledge:hunt','knowledge:devise'].includes(state.action))return null;
  await save(source,'recall',{...state,usedBy:message.id});return state;
 }
 async function recallTargets(message,actor){
  const native=await resolveMessageTargets(message,{fromUuid});if(native.length)return native;
  const prefix=`${option}:recall:`,markers=options(message).filter(o=>typeof o==='string'&&o.startsWith(prefix));
  const state=markers.length===1?own(game.messages.get(markers[0].slice(prefix.length))).recall:null;
  const uuids=state?.actorUuid===actor.uuid?[state.targetUuid]:own(message).targetUuids??[];
  if(uuids.length!==1||!/^Scene\.[^.]+\.Token\.[^.]+$/.test(uuids[0]))return [];
  const token=await fromUuid(uuids[0]);return token?.documentName==='Token'&&token.actor?[token]:[];
 }
 async function processRecall(message){
  if(!isActiveGM(game)||message.flags?.pf2e?.context?.type!=='skill-check'||!options(message).includes('action:recall-knowledge')||!Number.isInteger(degree(message)))return;
  return queue.run('recall',async()=>{
   if(own(message).processed)return;
   const actor=message.actor,author=userFor(message);if(!actor||!author||!actor.testUserPermission?.(author,'OWNER')||!['monster','known','stance'].some(k=>has(actor,k)))return;
   const targets=await recallTargets(message,actor);if(targets.length!==1||targets[0].actor.uuid===actor.uuid)return;const target=targets[0];
   const source=await preciseOrigin(message,actor,target),dos=degree(message),user=source?.userId?game.users.get(source.userId):author;
   if(!user||!actor.testUserPermission(user,'OWNER'))return;
   await save(message,'processed',true);
   const kinds=[];if(dos===3&&has(actor,'monster')&&prey(actor,target))kinds.push('monster');
   if(dos===3&&has(actor,'known')&&source?.action==='knowledge:devise')kinds.push('known');
   if(kinds.length){const allies=await recipients(actor,user,message);for(const kind of kinds)for(const recipient of allies)await benefit(kind,actor,recipient,target,message);}
   if(dos>=2&&has(actor,'stance')&&turn()&&stance(actor)&&target.actor.isEnemyOf?.(actor)!==false){
    await queue.run('strategist-claims',async()=>{
     if(own(actor).strategistTurn===turn())return;
     const token=sourceToken(actor,message);if(!token)return;
     const pending=states(actor).filter(s=>['pending','claimed'].includes(s.status)&&s.combatId===game.combat.id);
     await save(actor,'strategistTurn',turn());
     if(!pending.some(s=>s.targetUuid===target.uuid))pending.push({id:message.id,status:'pending',targetUuid:target.uuid,sourceTokenUuid:token.uuid,combatId:game.combat.id});
     await saveStates(actor,pending);
    });
   }
   // No public footer or changes to content, blind, whisper, DC or actual knowledge.
  });
 }
 function stance(actor){return values(actor.items).find(i=>hasSource(i,KNOWLEDGE_SOURCES.stanceEffect)&&!i.isExpired);}
 async function claimAttack(payload,user){
  gm();const actor=await fromUuid(payload?.actorUuid);requireOwner(actor,user);
  return queue.run('strategist-claims',async()=>{
   const attacker=await fromUuid(payload.sourceTokenUuid),target=await fromUuid(payload.targetUuid),claims=[],id=random();
   if(attacker?.documentName!=='Token'||target?.documentName!=='Token'||attacker.actor?.uuid!==actor.uuid||attacker.parent?.id!==target.parent?.id)return {id,actorUuid:actor.uuid,claims};
   const eligible=[];
   for(const source of actors())for(const state of states(source)){
    if(!['pending','claimed'].includes(state?.status)||state.targetUuid!==target.uuid||state.combatId!==game.combat?.id||!game.combat?.started||!stance(source)||!has(source,'stance'))continue;
    const token=await fromUuid(state.sourceTokenUuid),slug=stance(source).system?.rules?.find(r=>r.key==='Aura')?.slug,aura=slug?token?.auras?.get?.(slug):null;
    // Native coverage includes walls, active aura squares, elevation and hidden tokens.
    if(token?.parent?.id!==attacker.parent?.id||(source.uuid!==actor.uuid&&(!actor.isAllyOf?.(source)||!aura?.containsToken?.(attacker))))continue;
    if(state.status==='claimed')return {id,actorUuid:actor.uuid,claims:[],busy:true};
    eligible.push([source,state]);
   }
   for(const[source,state]of eligible){
    await saveState(source,{...state,status:'claimed',claim:id,actorUuid:actor.uuid,userId:user.id,attackerTokenUuid:attacker.uuid});
    claims.push({sourceActor:source.uuid,stateId:state.id});
   }
   try{if(claims.length)await upsertOwnedEffect(actor,`strategist:${id}`,buildStrategistAttackEffect({targetUuid:target.uuid,claim:id,sourceActor:claims.map(c=>c.sourceActor).join(',')}));}
   catch(error){for(const c of claims){const source=await fromUuid(c.sourceActor),state=states(source).find(s=>s.id===c.stateId);if(state?.claim===id)await saveState(source,{...state,status:'pending',claim:null,actorUuid:null,userId:null});}throw error;}
   return {id,actorUuid:actor.uuid,claims};
  });
 }
 async function completeAttack(receipt,{rolled},user){
  gm();const actor=await fromUuid(receipt?.actorUuid);requireOwner(actor,user);
  return queue.run('strategist-claims',async()=>{
   for(const claim of receipt.claims??[]){const source=await fromUuid(claim.sourceActor),state=states(source).find(s=>s.id===claim.stateId);if(state?.claim!==receipt.id||state.actorUuid!==actor.uuid||state.userId!==user.id||state.id!==claim.stateId||state.status!=='claimed')continue;
    await saveState(source,{...state,status:rolled?'consumed':'pending',settledClaim:rolled?{id:receipt.id,actorUuid:actor.uuid}:null,cleanupDone:false,claim:null,actorUuid:null,userId:null});
   }
   gm();const ids=values(actor.items).filter(i=>own(i).kind==='strategist-claim'&&own(i).claim===receipt.id).map(i=>i.id);if(ids.length)await actor.deleteEmbeddedDocuments('Item',ids);
   for(const claim of receipt.claims??[]){const source=await fromUuid(claim.sourceActor),state=states(source).find(s=>s.id===claim.stateId);if(state?.status==='consumed'&&state.settledClaim?.id===receipt.id)await saveState(source,{...state,cleanupDone:true});}
  });
 }
 function claimProof(message){
  if(game.messages.get(message?.id)!==message||!message.isCheckRoll||message.flags?.pf2e?.context?.type!=='attack-roll'||![0,1,2,3].includes(degree(message))||!Number.isFinite(message.rolls?.[0]?.total))return null;
  const prefix=`${option}:claim:`,ids=options(message).filter(o=>typeof o==='string'&&o.startsWith(prefix)).map(o=>o.slice(prefix.length));
  const actor=message.actor,user=userFor(message),target=message.flags.pf2e.context.target?.token;
  if(ids.length!==1||!actor||!user||!actor.testUserPermission?.(user,'OWNER')||typeof target!=='string')return null;
  return {claim:ids[0],actor,user,target};
 }
 async function settleClaimProof(message,proof){
  gm();const matched=[];
  for(const source of actors())for(const state of states(source)){
   if(state.status==='consumed'&&state.settledClaim?.id===proof.claim&&state.settledClaim.actorUuid===proof.actor.uuid&&state.attackMessageId===message.id){if(!state.cleanupDone)matched.push([source,state.id]);continue;}
   if(state.status!=='claimed'||state.claim!==proof.claim||state.actorUuid!==proof.actor.uuid||state.targetUuid!==proof.target)continue;
   // Native composite activities may be rolled by the GM on behalf of the
   // original owner. Either author must still own the exact attacking actor.
   if(state.userId!==proof.user.id&&!proof.user.isGM&&!game.users.get(state.userId)?.isGM)continue;
   if(state.attackerTokenUuid&&message.speaker?.token&&state.attackerTokenUuid!==`Scene.${message.speaker.scene}.Token.${message.speaker.token}`)continue;
   await saveState(source,{...state,status:'consumed',attackMessageId:message.id,settledClaim:{id:proof.claim,actorUuid:proof.actor.uuid},cleanupDone:false,claim:null,actorUuid:null,userId:null});matched.push([source,state.id]);
  }
  if(matched.length){
   gm();const ids=values(proof.actor.items).filter(i=>own(i).kind==='strategist-claim'&&own(i).claim===proof.claim).map(i=>i.id);if(ids.length)await proof.actor.deleteEmbeddedDocuments('Item',ids);
   for(const [source,id]of matched){const state=states(source).find(s=>s.id===id);if(state?.settledClaim?.id===proof.claim)await saveState(source,{...state,cleanupDone:true});}
  }
 }
 async function processClaimCheck(message){
  if(!isActiveGM(game))return;const proof=claimProof(message);if(!proof)return;
  return queue.run('strategist-claims',()=>settleClaimProof(message,proof));
 }
 const rpc=async(method,payload)=>{if(isActiveGM(game))return method==='claim'?claimAttack(payload,game.user):completeAttack(payload.receipt,payload.result,game.user);if(!socket||!game.users.activeGM)throw Error('需要在线GM处理军师架势的共享攻击机会。');const r=await socket.executeAsUser(`knowledge-${method}`,game.users.activeGM.id,payload);if(!r.ok)throw Error(r.error);return r.value;};
 async function withAttack(actor,params,roll,beforeNative,spellAttack=false){
  const invoke=next=>{beforeNative?.(next);return roll(next);};
  if(isEatFortuneProbe(params))return invoke(params);
  // Spell.rollAttack forwards a TokenDocument; StatisticCheck.roll falls back
  // to the first active token. Strike uses a different native origin selection.
  const spellSource=next=>next.token??values(actor.getActiveTokens?.(true,true))[0]??null;
  const target=tokenDoc(params.target?.getActiveTokens?.(true,true)?.[0]??params.target)??tokenDoc(values(game.user.targets)[0]),token=spellAttack?spellSource(params):sourceToken(actor);if(!target?.uuid||!token)return invoke(params);
  // Do not add an RPC to actors/scenes with no armed marshal.
  if(!actors().some(a=>states(a).some(s=>['pending','claimed'].includes(s.status)&&s.targetUuid===target.uuid)))return invoke(params);
  const sourceUuid=token.uuid,sourceScene=token.parent,user=game.user;
  const assertSource=(next=params)=>{
   if(!spellAttack)return;
   if(game.user!==user||game.users.get(user?.id)!==user||!actor.testUserPermission?.(user,'OWNER')||token.documentName!=='Token'||token.actor!==actor||token.uuid!==sourceUuid||token.parent!==sourceScene||values(game.scenes).find(scene=>scene.id===sourceScene?.id)!==sourceScene||values(sourceScene?.tokens).find(current=>current.id===token.id)!==token||spellSource(params)!==token||spellSource(next)!==token)throw Error('军师架势法术攻击的原始来源 Token 或角色身份已经变化。');
  };
  const confirmSource=async(next=params)=>{
   if(!spellAttack)return;
   assertSource(next);const [liveActor,liveToken]=await Promise.all([fromUuid(actor.uuid),fromUuid(sourceUuid)]);assertSource(next);
   if(liveActor!==actor||liveToken!==token)throw Error('军师架势法术攻击的来源 Token 文档已经替换。');
  };
  if(spellAttack)await confirmSource();
  const input={actorUuid:actor.uuid,targetUuid:target.uuid,sourceTokenUuid:token.uuid},start=Date.now();let receipt;
  do{receipt=await rpc('claim',input);if(receipt.busy){assertSource();if(Date.now()-start>60000)throw Error('另一次攻击尚未结束，本次攻击未投骰。');await new Promise(r=>setTimeout(r,100));assertSource();}}while(receipt.busy);
  if(!receipt.claims.length){if(spellAttack)await confirmSource();return invoke(params);}
  let rolled=false,enteredNative=false,returned=false,guardRejected=false;try{
   const started=Date.now();while(!values(actor.items).some(i=>own(i).claim===receipt.id)){if(Date.now()-started>5000)throw Error('军师架势效果尚未同步；本次攻击未投骰。');await new Promise(r=>setTimeout(r,20));}
   const next={...params,options:new Set([...params.options??[],`${option}:claim:${receipt.id}`]),extraRollOptions:[...params.extraRollOptions??[],`${option}:claim:${receipt.id}`],callback:(...args)=>{rolled=!!args[0];return params.callback?.(...args)}};
   // The owner guard runs after claim/effect synchronization, before either
   // automation records native entry. A veto therefore returns the unrolled claim.
   try{if(spellAttack)await confirmSource(next);beforeNative?.(next);assertSource(next);}catch(error){guardRejected=true;throw error;}
   enteredNative=true;const result=await roll(next);returned=true;rolled||=!!result;return result;
  }finally{if(!enteredNative||returned||rolled)try{await rpc('complete',{receipt,result:{rolled}});}catch(error){if(!guardRejected)throw error;try{Promise.resolve(onError(error)).catch(()=>{});}catch{}}}
 }
 function wrapStrike(strike,actor){for(const variant of strike?.variants??[]){const native=variant.roll;if(typeof native!=='function'||native.knowledgeWrapped)continue;const wrapped=async(params={})=>{
   const target=tokenDoc(params.target)??tokenDoc(values(game.user.targets)[0]);
   if(values(actor.items).some(i=>own(i).kind==='devise'&&own(i).branch==='skill'&&!i.isExpired&&!expired(own(i).timing)&&own(i).targetUuid===target?.uuid))throw Error('技能策略期间不能打击该生物；可选择其他目标。');
   const invoke=next=>withAttack(actor,next,claimed=>withEatStrikeFrame({actor,strike,variant,params:claimed},framed=>native.call(variant,framed)));
   return strikeMiddleware?strikeMiddleware({actor,strike,variant,params,probe:isEatFortuneProbe(params)},invoke):invoke(params);
  };wrapped.knowledgeWrapped=true;variant.roll=wrapped;}return strike;}
 async function recall({actor,item,message,user,action},target){
  const answer=await pick(actor,user,'作为本次动作的一部分回忆知识',[{value:'roll',label:'进行回忆知识检定'},{value:'skip',label:'跳过'}]);if(answer!=='roll')return;
  const skillSlugs=new Set(['arcana','crafting','medicine','nature','occultism','religion','society',...values(actor.itemTypes?.lore).map(i=>i.slug)]);
  const choices=[...skillSlugs].map(k=>actor.getStatistic?.(k)??actor.skills?.[k]).filter(Boolean).map(s=>({value:s.slug,label:s.label??s.slug}));
  const skill=await pick(actor,user,'回忆知识 · 选择使用的技能',choices);if(!skill)return;
  const native=game.pf2e?.actions?.get?.('recall-knowledge');if(!native?.use)throw Error('当前系统缺少原生回忆知识动作。');
  await save(message,'recall',{actorUuid:actor.uuid,targetUuid:target.uuid,itemId:item.id,action,userId:user.id});
  const results=await native.use({actors:[actor],statistic:skill,target:target.object,rollOptions:[`${option}:recall:${message.id}`],event:{ctrlKey:false,metaKey:false,shiftKey:!!game.user.settings?.showCheckDialogs}});
  for(const r of results??[])if(r.message)await processRecall(r.message);
 }
 async function executeUsage(ctx){
  gm();const {actor,item,message,user,action,frequencyReceipt}=ctx;requireOwner(actor,user);if(item.actor?.uuid!==actor.uuid||resolveAction(item)!==action)throw Error('知识能力来源不匹配。');
  return queue.run(`usage:${actor.uuid}`,async()=>{
   if(action==='knowledge:stance'){
    if(!game.combat?.started)throw Error('军师架势只能在遭遇中使用。');const cooldown=own(actor).stanceCooldown;if(Number.isFinite(cooldown)&&cooldown>now())throw Error('军师架势仍处于1分钟冷却。');
    const choices=[{value:'society',label:'社会 Society'},...values(actor.itemTypes?.lore).filter(i=>i.slug==='warfare-lore').map(i=>({value:i.slug,label:i.name}))];
    const selected=await pick(actor,user,'军师架势 · 选择检定',choices);if(!selected)return '已取消进入架势。';
    const stat=actor.getStatistic?.(selected)??actor.skills?.[selected];if(!stat?.check?.roll)throw Error('找不到所选技能的原生检定。');
    let result='未进入军师架势。';await stat.check.roll({dc:{value:levelDC(actor.level),visible:true},skipDialog:true,extraRollOptions:['action:strategist-stance'],callback:async(roll,outcome)=>{
     const dos=roll.options?.degreeOfSuccess??({criticalFailure:0,failure:1,success:2,criticalSuccess:3}[outcome]);
     if(dos===0){await save(actor,'stanceCooldown',now()+60);result='未进入架势，1分钟后可再次尝试。';}
     if(dos>=2){const data=await load(KNOWLEDGE_SOURCES.stanceEffect),r=dos===3?20:15;for(const rule of data.system.rules)if(rule.key==='ChoiceSet'&&rule.flag==='auraRadius')rule.selection=r;
      data.flags??={};data.flags.system={...data.flags.system,rulesSelections:{...data.flags.system?.rulesSelections,auraRadius:r}};data.flags[MODULE_ID]={knowledge:{kind:'stance'}};data._stats={...data._stats,compendiumSource:KNOWLEDGE_SOURCES.stanceEffect};await upsertOwnedEffect(actor,'knowledge:stance',data);result=`已进入军师架势，灵光${r}尺。`;
     }
    }});return result;
   }
   let spent=false;try{
   const targets=await resolveMessageTargets(message,{fromUuid});if(targets.length!==1||!targets[0].object||targets[0].actor.uuid===actor.uuid)throw Error('请为本次动作选中一个可见的生物Token。');const target=targets[0];
   if(action==='knowledge:hunt'){
    if(game.modules?.get('pf2e-ranged-combat')?.active){
     const before=own(message).huntBefore;if(!Array.isArray(before))throw Error('缺少本次猎杀目标的原始记录；请重新使用猎杀目标。');
     const started=Date.now();while(!values(actor.items).some(i=>hasSource(i,KNOWLEDGE_SOURCES.prey)&&!before.includes(i.id)&&i.system.rules?.some(r=>r.key==='TokenMark'&&r.slug==='hunted-prey'&&r.uuid===target.uuid))){
      if(Date.now()-started>5000)throw Error('猎杀目标尚未建立本次猎物记录，未进行回忆知识。');await new Promise(r=>setTimeout(r,20));
     }
    }
    await recall(ctx,target);return '本次猎杀目标的知识步骤已处理。';
   }
   if(!game.combat?.started||!timing(actor))throw Error('出谋划策需要角色已加入遭遇。');
    const uses=item.system.frequency?.value??item.system.frequency?.max??1;if(!frequencyReceipt&&uses<1)throw Error('出谋划策本轮次数已用尽。');
    await recall(ctx,target);
    const Roll=globalThis.Roll;if(!Roll)throw Error('原生掷骰API不可用。');const roll=await new Roll('1d20').evaluate();
    await roll.toMessage({speaker:globalThis.ChatMessage.getSpeaker({actor,token:sourceToken(actor,message)}),flavor:'出谋划策',flags:{[MODULE_ID]:{usageGenerated:true}}});
    spent=true;if(!frequencyReceipt&&item.system.frequency)await item.update({'system.frequency.value':uses-1},{[MODULE_ID]:{usageInternal:true}});
    const branch=await pick(actor,user,`出谋划策 · d20=${roll.total}`,[{value:'attack',label:'攻击策略'},{value:'skill',label:'技能策略'}]);if(!branch)return '本次出谋划策已掷骰；未选择策略。';
    if(branch==='skill'){
     const rk=own(message).recall?.usedBy,ids=values(actor.items).filter(i=>own(i).kind==='known'&&own(i).sourceActor===actor.uuid&&own(i).sourceMessageId===rk).map(i=>i.id);
     if(ids.length)await actor.deleteEmbeddedDocuments('Item',ids);
    }
    const data=origin(await load(KNOWLEDGE_SOURCES.deviseEffect),actor,item,target,timing(actor));data.system.badge={type:'counter',value:roll.total,min:1,max:20};
    data.system.rules=data.system.rules.flatMap(r=>r.key==='TokenMark'?[{...r,uuid:target.uuid}]:r.key==='RollOption'&&r.option==='devise-a-stratagem'?[{key:'RollOption',domain:'all',option:'devise-a-stratagem'},{key:'RollOption',domain:'all',option:`devise-a-stratagem:${branch}`}]:[r]);
    if(branch==='skill')for(const rule of data.system.rules)if(['FlatModifier','AdjustModifier'].includes(rule.key)&&rule.predicate?.includes('devise-a-stratagem:skill'))rule.predicate.push('target:mark:devise-a-stratagem');
    data.flags??={};data.flags[MODULE_ID]={knowledge:{kind:'devise',branch,targetUuid:target.uuid,timing:timing(actor)}};
    await upsertOwnedEffect(actor,'knowledge:devise',data);return `已采用${branch==='attack'?'攻击':'技能'}策略，至下回合开始。`;
   }catch(error){if(action==='knowledge:devise'&&frequencyReceipt&&!spent)await item.update({'system.frequency.value':Math.min(item.system.frequency.max,(item.system.frequency.value??0)+1)},{[MODULE_ID]:{usageInternal:true}});throw error;}
  });
 }
 async function maintain(actor){if(!isActiveGM(game)||!actor?.items)return;const remove=values(actor.items).filter(i=>own(i).timing&&expired(own(i).timing)).map(i=>i.id);if(remove.length)await actor.deleteEmbeddedDocuments('Item',remove);await queue.run('strategist-claims',async()=>{
 const interrupted=new Set(states(actor).flatMap(s=>s.status==='claimed'?[s.claim]:s.settledClaim&&!s.cleanupDone?[s.settledClaim.id]:[]));
  if(interrupted.size)for(const message of values(game.messages)){const proof=claimProof(message);if(proof&&interrupted.has(proof.claim))await settleClaimProof(message,proof);}
  for(const state of states(actor).filter(s=>s.status==='consumed'&&s.settledClaim&&!s.cleanupDone)){
   const recipient=await fromUuid(state.settledClaim.actorUuid);if(!recipient?.items)continue;
   gm();const ids=values(recipient.items).filter(i=>own(i).kind==='strategist-claim'&&own(i).claim===state.settledClaim.id).map(i=>i.id);if(ids.length)await recipient.deleteEmbeddedDocuments('Item',ids);
   await saveState(actor,{...state,cleanupDone:true});
  }
  const existing=states(actor),keep=existing.filter(s=>has(actor,'stance')&&stance(actor)&&game.combat?.started&&s.combatId===game.combat.id);if(keep.length!==existing.length)await saveStates(actor,keep);
 });}
 function register({Hooks,libWrapper,socket:socketApi,onError:report=onError}={}){
  if(registered)return()=>{};registered=true;socket=socketApi;const registrations=[],on=(name,fn)=>registrations.push([name,Hooks.on(name,(...args)=>Promise.resolve().then(()=>fn(...args)).catch(report))]);
  if(socket)for(const method of ['claim','complete'])socket.register(`knowledge-${method}`,async function(payload){try{const user=game.users.get(this.socketdata.userId);return {ok:true,value:method==='claim'?await claimAttack(payload,user):await completeAttack(payload.receipt,payload.result,user)}}catch(e){return {ok:false,error:e.message}}});
  // Numeric-DC skill checks omit context.target in PF2e. Snapshot on the creator,
  // never from the executing GM's selection, and never replace native context.
  const pre=Hooks.on('preCreateChatMessage',message=>{
   if(message.flags?.pf2e?.context?.type==='skill-check'&&options(message).includes('action:recall-knowledge'))message.updateSource({[`flags.${MODULE_ID}.knowledge.targetUuids`]:values(game.user.targets).map(t=>tokenDoc(t).uuid)});
   if(resolveAction(message.item)==='knowledge:hunt')message.updateSource({[`flags.${MODULE_ID}.knowledge.huntBefore`]:values(message.item.actor.items).filter(i=>hasSource(i,KNOWLEDGE_SOURCES.prey)).map(i=>i.id)});
  });registrations.push(['preCreateChatMessage',pre]);
  on('createChatMessage',processRecall);on('updateChatMessage',processRecall);
  on('createChatMessage',processClaimCheck);on('updateChatMessage',processClaimCheck);
  on('updateCombat',async()=>{for(const actor of actors())await maintain(actor)});on('deleteCombat',async()=>{for(const actor of actors())await maintain(actor)});on('deleteItem',i=>i.actor&&maintain(i.actor));
  // PF2e emits this completed-rest hook only on its initiating client.
  on('pf2e.restForTheNight',async actor=>{if(actor?.testUserPermission?.(game.user,'OWNER')&&(has(actor,'monster')||own(actor).days?.length))await actor.update({[`flags.${MODULE_ID}.knowledge.days`]:[],[`flags.${MODULE_ID}.knowledge.dailyEpoch`]:(own(actor).dailyEpoch??0)+1})});
  const path='CONFIG.PF2E.Item.documentClasses.spell.prototype.rollAttack';
  if(libWrapper)libWrapper.register(MODULE_ID,path,function(wrapped,event,attackNumber=1,params={}){
   const spell=this,invoke=(next,beforeNative)=>withAttack(spell.actor,next,claimed=>wrapped(event,attackNumber,claimed),beforeNative,true);
   return spellAttackMiddleware?spellAttackMiddleware({spell,event,attackNumber,params,nativeBoundary:true},invoke):invoke(params);
  },'WRAPPER');
  const effectPath='CONFIG.PF2E.Item.documentClasses.effect.prototype._preCreate';
  if(libWrapper)libWrapper.register(MODULE_ID,effectPath,function(wrapped,...args){
   const source=values(this.actor?.items).find(i=>i.uuid===this.system?.context?.origin?.item);
   // HUD/Toolbelt apply their self-effect before posting the original Use card.
   // Stop only this redundant native effect before its badge rolls another d20.
   if(own(this).kind!=='devise'&&hasSource(this,KNOWLEDGE_SOURCES.deviseEffect)&&source&&resolveAction(source)==='knowledge:devise')return false;
   return wrapped(...args);
  },'MIXED');
  return()=>{for(const[name,id]of registrations)Hooks.off(name,id);if(libWrapper){libWrapper.unregister(MODULE_ID,path);libWrapper.unregister(MODULE_ID,effectPath);}registered=false;};
 }
 return {resolveAction,executeUsage,register,maintain,processRecall,claimAttack,completeAttack,processClaimCheck,wrapStrike};
}
