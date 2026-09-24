import {ELECTRICITY_MODULE_ID as ID,ELECTRICITY_SOURCES as S,ELECTRICITY_APPLY_PREFIX as APPLY,ELECTRICITY_SOURCE_PREFIX as SOURCE,
 classifyElectricityDamage,createElectricityLedger,electricityState,electricityEffects,receiptMatches} from './eldamon-electricity.mjs';
import {sourceUuid} from './metapower/rules.mjs';
import {showNativeChoice} from './native-context.mjs';
import {createActorStateIndex} from './actor-state-index.mjs';
const values=c=>Array.from(c?.values?.()??c??[]),random=()=>globalThis.foundry?.utils?.randomID?.(24)??globalThis.crypto.randomUUID();
const tokenUuid=s=>s?.scene&&s?.token?`Scene.${s.scene}.Token.${s.token}`:null;
const currentToken=t=>t?.actor&&t.parent?.tokens?.get(t.id)===t;
export function preserveElectricityOnAlter(original,result){
 const proof=original?.options?.[ID]?.electricitySource;
 if(proof&&result?.options)result.options[ID]={...result.options[ID],electricitySource:structuredClone(proof)};
 return result;
}

/** Small adapter around native publication and application. It never rolls or
 * applies replacement damage, and never treats a rolled total as damage taken. */
export function createEldamonElectricityProvider({game,reactionRestriction,fromUuid,onError=console.error,selectChoice=showNativeChoice,refreshOutsideEncounter=async()=>{}}={}){
 const ledger=createElectricityLedger({game,reactionRestriction,fromUuid}),scopes=new Map(),sourceKeys=new WeakMap();let socket,sourceCards;
 function forgetSource(message){
  const key=sourceKeys.get(message),bucket=sourceCards?.get(key);if(!bucket)return;
  bucket.delete(message);if(!bucket.size)sourceCards.delete(key);sourceKeys.delete(message);
 }
 function rememberSource(message){
  if(!sourceCards)return;forgetSource(message);const nonce=message.flags?.[ID]?.electricitySource?.nonce;
  if(typeof nonce!=='string'||!nonce)return;const bucket=sourceCards.get(nonce)??new Set();bucket.add(message);sourceCards.set(nonce,bucket);sourceKeys.set(message,nonce);
 }
 function sourceMessages(nonce){
  if(!sourceCards){sourceCards=new Map();for(const message of values(game.messages))rememberSource(message)}
  return [...sourceCards.get(nonce)??[]].filter(message=>game.messages.get(message.id)===message&&message.flags?.[ID]?.electricitySource?.nonce===nonce);
 }
 const activeActors=createActorStateIndex({game,matches:actor=>{
  const state=actor.flags?.[ID]?.electricity;
  return Object.keys(state?.pendingShocks??{}).length>0||Object.values(state?.operations??{}).some(r=>r.status==='started')||
   electricityEffects(actor,S.charged).length>0||electricityEffects(actor,S.shocked).some(item=>item.flags?.[ID]?.electricityShock?.expires);
 }});
 const active=()=>game.user?.id===game.users.activeGM?.id;
 async function rpc(method,payload){
  if(active())return ledger[method](payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('Electricity lifecycle requires an online active GM.');
  const result=await socket.executeAsUser(`electricity:${method}`,game.users.activeGM.id,payload);
  if(!result?.ok){const error=Error(result?.error??'Electricity coordinator did not respond.');if(method==='confirmedAction'&&result?.electricityNotApplied===true)error.electricityNotApplied=true;throw error}return result.value;
 }
 async function interceptDamageMessage(roll,data={},options={},native){
  if(classifyElectricityDamage(roll)==='none')return native(data,options);
  const opts=data.flags?.pf2e?.context?.options??[],markers=opts.filter(o=>o.startsWith(`${ID}:metapower:`));
  const nonce=random(),nativeTargets=data.flags?.['pf2e-toolbelt']?.targetHelper?.targets;
  let effectKey=`damage:${nonce}`,targetUuids=Array.isArray(nativeTargets)?nativeTargets:values(game.user.targets).map(t=>(t.document??t).uuid),channel=false,area=false;
  if(markers.length===1){
   const [cardId,channelNonce]=markers[0].slice(`${ID}:metapower:`.length).split(':'),card=game.messages.get(cardId),actor=await fromUuid(card?.flags?.[ID]?.metapowerUse?.actorUuid);
   const receipt=actor?.flags?.[ID]?.metapower?.receipts?.[channelNonce];
   if(receipt?.status==='committed'&&receipt.messageUuid===card?.uuid&&receipt.itemUuid===data.flags?.pf2e?.origin?.uuid){
    effectKey=`channel:${actor.uuid}:${channelNonce}`;channel=true;area=!!receipt.snapshot?.area;
    // Area recipients are selected by the native template after Use. Single
    // target powers retain their admitted recipient across the damage click.
    if(!receipt.snapshot?.area)targetUuids=receipt.selection?.targetUuids??targetUuids;
    else if(!Array.isArray(nativeTargets)){
     const owner=game.users.get(receipt.userId);
     targetUuids=owner?.active&&actor.testUserPermission?.(owner,'OWNER')?values(owner.targets).map(t=>(t.document??t).uuid).filter(uuid=>uuid?.startsWith(`Scene.${card.speaker?.scene}.Token.`)):[];
    }
   }
  }
  const source={nonce,effectKey,targetUuids:[...new Set(targetUuids)],...(channel?{targetMode:area?'area':'single'}:{})};
  roll.options??={};roll.options[ID]={...roll.options[ID],electricitySource:{nonce}};
  const next={...data,flags:{...data.flags,[ID]:{...data.flags?.[ID],electricitySource:source},pf2e:{...data.flags?.pf2e,context:{...data.flags?.pf2e?.context,options:[...opts.filter(o=>!o.startsWith(SOURCE)),SOURCE+nonce]}}}};
  if(channel)next.flags['pf2e-toolbelt']={...data.flags?.['pf2e-toolbelt'],targetHelper:{...data.flags?.['pf2e-toolbelt']?.targetHelper,targets:source.targetUuids}};
  return native(next,options);
 }
 async function beforeDamage(actor,params){
  const kind=classifyElectricityDamage(params.damage);if(kind==='none'||params.final)return null;
  const target=params.token?.document??params.token;if(!currentToken(target)||target.actor.uuid!==actor.uuid||!target.actor.testUserPermission?.(game.user,'OWNER'))return null;
  const sourceNonce=params.damage?.options?.[ID]?.electricitySource?.nonce;if(!sourceNonce)return null;
  const matches=sourceMessages(sourceNonce);if(matches.length!==1)return null;
  const message=matches[0],rollIndex=message.rolls?.findIndex(r=>r.options?.[ID]?.electricitySource?.nonce===sourceNonce)??-1;
  if(rollIndex<0||classifyElectricityDamage(message.rolls[rollIndex])!==kind)return null;
  const source=message.flags[ID].electricitySource,record=await rpc('beginDamage',{nonce:random(),actorUuid:actor.uuid,tokenUuid:target.uuid,sourceMessageUuid:message.uuid,sourceNonce,rollIndex,
   sourceItemUuid:params.item?.uuid??null,effectKey:source.effectKey,kind});
  scopes.set(record.nonce,{record,actor,amount:null,messageId:null});
  return {params:{...params,rollOptions:new Set([...params.rollOptions??[],APPLY+record.nonce])},receipt:{actorUuid:actor.uuid,nonce:record.nonce}};
 }
 // The patched PF2e 8.5 IWR seam runs after immunity/resistance/weakness,
 // shield hardness and actor hardness. The existing adapter's decision is
 // awaited by main before observing; a suppressed application is exactly zero.
 function observeNativeIWR(actor,params,_iwr,options,context,handled){
  const tags=values(options).filter(o=>typeof o==='string'&&o.startsWith(APPLY));if(tags.length!==1)return;
  const scope=scopes.get(tags[0].slice(APPLY.length));
  if(!scope||actor.uuid!==scope.record.actorUuid||params.token?.uuid!==scope.record.tokenUuid||classifyElectricityDamage(params.damage)!==scope.record.kind)return;
  const amount=handled===true?0:context?.actorDamage;
  if(Number.isFinite(amount)&&amount>=0&&actor.hitPoints?.max>0)scope.amount=amount;
 }
 function decorateReceipt(document,data,_options,userId){
  if(userId!==game.user.id)return;
  const source=document.flags?.[ID]?.electricitySource;
  // PF2e returns a toMessage(create:false) draft before its later native create.
  // Toolbelt's upstream preCreate handoff runs first. Finalize the two target
  // manifests here, in that same creation, with no draft-lifetime listener.
  if(data.flags?.pf2e?.context?.type==='damage-roll'&&['area','single'].includes(source?.targetMode)&&Array.isArray(source.targetUuids)){
   const current=document.flags?.['pf2e-toolbelt']?.targetHelper?.targets;
   const targets=source.targetMode==='area'&&Array.isArray(current)&&current.length?current:source.targetUuids;
   document.updateSource({[`flags.${ID}.electricitySource.targetUuids`]:targets,'flags.pf2e-toolbelt.targetHelper.targets':targets});
   return;
  }
  if(data.flags?.pf2e?.context?.type!=='damage-taken')return;
  const options=data.flags.pf2e.context.options??[],tags=options.filter(o=>o.startsWith(APPLY));if(tags.length!==1)return;
  const scope=scopes.get(tags[0].slice(APPLY.length));if(!scope||scope.amount===null||tokenUuid(data.speaker)!==scope.record.tokenUuid)return;
  let amount=scope.amount;
  const prefix=`${ID}:destructive-block:`,blocks=options.filter(o=>typeof o==='string'&&o.startsWith(prefix)),proof=data.flags?.[ID]?.shieldBlock;
  if(blocks.length||proof?.kind==='destructive-block'){
   // Destructive Block's health-delta adapter runs after the ordinary IWR seam.
   // Its exact native receipt is the final amount; an uncertain block cannot
   // fall back to the earlier amount, even if that amount was positive.
   const applied=data.flags.pf2e.appliedDamage;
   if(blocks.length!==1||proof?.kind!=='destructive-block'||proof.uncertain||typeof proof.nonce!=='string'||!proof.nonce||blocks[0]!==prefix+proof.nonce||
    data.speaker?.actor!==scope.actor.id||(data.flags.pf2e.origin?.uuid??null)!==scope.record.sourceItemUuid||applied&&(applied.uuid!==scope.record.actorUuid||applied.isHealing||applied.isReverted)||
    typeof proof.shieldId!=='string'||!proof.shieldId||applied?.shield&&applied.shield.id!==proof.shieldId||!Number.isFinite(proof.incoming)||proof.incoming<0||
    !Number.isFinite(proof.actorDamage)||proof.actorDamage<0||proof.actorDamage>proof.incoming||proof.actorDamage>scope.amount)return;
   amount=proof.actorDamage;
  }
  document.updateSource({[`flags.${ID}.electricityApplied`]:{nonce:scope.record.nonce,amount}});
 }
 function capture(message,_options,creator){
  if(creator!==game.user.id)return;
  const tags=(message.flags?.pf2e?.context?.options??[]).filter(o=>o.startsWith(APPLY));if(tags.length!==1)return;
  const scope=scopes.get(tags[0].slice(APPLY.length));if(scope&&receiptMatches(message,scope.record)&&!scope.messageId)scope.messageId=message.id;
 }
 async function afterDamage(receipt,{applied,uncertain}){
  const scope=scopes.get(receipt?.nonce);if(!scope)return;
  try{
   if((applied||uncertain)&&scope.messageId){const message=game.messages.get(scope.messageId);await rpc('finishDamage',{...receipt,receiptUuid:message.uuid});}
  }finally{scopes.delete(receipt.nonce);}
 }
 async function beforeChannel({item,selection,kind}){
  if(sourceUuid(item)!==S.chain)return selection;
  if(selection.targetUuids?.length!==1)throw Error('先选定一个反应电链目标，再使用威能。');
  const tokens=values(globalThis.canvas?.tokens?.controlled).map(t=>t.document).filter(t=>t.actor?.uuid===item.actor.uuid);
  const source=tokens.length===1?tokens[0]:item.actor.getActiveTokens?.(true,true)?.[0];
  if(!currentToken(source))throw Error('反应电链需要场景中的施法者Token。');
  const candidates=await rpc('candidates',{actorUuid:item.actor.uuid,sourceTokenUuid:source.uuid,selection,kind});
  if(!candidates.length)throw Error('没有当前、可核验的电伤来源满足这两个30尺距离、目标与反应条件。混合伤害需GM先在原伤害回执确认电击分量。');
  const choice=candidates.length===1?'0':await selectChoice({title:'选择反应电链的实际电伤来源',choices:candidates.map((c,i)=>({value:String(i),label:c.label}))});
  if(choice===null||choice===undefined)return null;
  const chosen=candidates[Number(choice)];
  if(!chosen)return null;
  return {...selection,triggerConfirmed:true,eligibleTargetConfirmed:true,triggerDamage:chosen.amount,electricityEvidence:chosen.evidence};
 }
 async function interceptCheck(native,check,context={},...args){
  if(context.type!=='attack-roll'||!Number.isFinite(context.dc?.value))return native(check,context,...args);
  const target=context.target?.actor,item=context.item;
  if(!target||!electricityEffects(target,S.shocked).length)return native(check,context,...args);
  let electric=item?.type==='weapon'&&item.system?.damage?.damageType==='electricity';
  if(item?.type==='weapon'&&item.system?.runes?.property?.some(r=>['shock','greaterShock'].includes(r)))electric=true;
  const markers=values(context.options).filter(o=>o.startsWith(`${ID}:metapower:`));
  if(markers.length===1){
   const [id,nonce]=markers[0].slice(`${ID}:metapower:`.length).split(':'),card=game.messages.get(id),actor=await fromUuid(card?.flags?.[ID]?.metapowerUse?.actorUuid),r=actor?.flags?.[ID]?.metapower?.receipts?.[nonce];
   if(r?.status==='committed'&&r.messageUuid===card?.uuid&&r.itemUuid===item?.uuid&&[S.static,S.shot].includes(r.sourceUuid))electric=!r.snapshot?.siphon?.applies;
  }
  if(!electric)return native(check,context,...args);
  // Use the contextual AC modifiers that native already evaluated; circumstance
  // penalties do not stack. The status option follows this exact attack only.
  const modifiers=context.target?.statistic?.modifiers??target.attributes?.ac?.modifiers??[];
  const current=Math.min(0,...modifiers.filter(m=>m.enabled!==false&&!m.ignored&&m.type==='circumstance').map(m=>Number(m.modifier)||0));
  const already=values(context.options).includes('target:condition:off-guard')||target.hasCondition?.('off-guard');
  const delta=already?0:Math.min(0,-2-current),options=new Set([...values(context.options),'target:condition:off-guard']);
  return native(check,{...context,options,dc:{...context.dc,value:context.dc.value+delta}},...args);
 }
 async function onCommittedChannel({receipt,message,user=game.users.get(receipt?.userId)}){
  if(!receipt?.powerId)return;
  const payload={actorUuid:receipt.actorUuid,nonce:receipt.nonce,messageUuid:message.uuid};
  if(active())return ledger.channel(payload,user);
  return rpc('channel',payload);
 }
 async function onRefresh({actor,nonce}){if(active())return ledger.refresh({actor,nonce});}
 async function encounterEnded(combat){
  if(!active())return;
  const participants=[...new Map(values(combat.combatants).map(c=>c.actor).filter(Boolean).map(a=>[a.uuid,a])).values()];
  await ledger.expire({combat,ended:true,actors:activeActors.values()});
  for(const actor of participants){
   if(!values(actor.items).some(i=>sourceUuid(i)==='Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.naawsnBug9EOpzfN'))continue;
   await refreshOutsideEncounter({actor,nonce:`encounter:${combat.id}`});
  }
 }
 function renderReceipt(message,html){
  const root=html?.[0]??html;if(!root?.querySelector||root.querySelector('[data-electricity-receipt]'))return;
  const proof=message.flags?.[ID]?.electricityApplied,tags=(message.flags?.pf2e?.context?.options??[]).filter(o=>o.startsWith(APPLY));if(tags.length!==1)return;
  const block=document.createElement('div');block.dataset.electricityReceipt=tags[0].slice(APPLY.length);
  block.textContent=proof?'已记录原生伤害应用；反应电链将核验电击分量、两段距离与目标。':'缺少原生IWR应用回执；不会用掷骰总数触发电链。';
  if(game.user.isGM){
   const button=document.createElement('button');button.type='button';button.textContent='确认混合伤害中的实际电击分量';
   button.addEventListener('click',async()=>{try{
    const target=await fromUuid(tokenUuid(message.speaker)),record=electricityState(target?.actor).damage[block.dataset.electricityReceipt];
    if(record?.kind!=='mixed'||record.status!=='needs-attribution')throw Error('此回执不需要混合伤害分量确认。');
    const amount=await globalThis.foundry.applications.api.DialogV2.wait({window:{title:'原伤害回执：实际电击分量'},content:'<p>输入已应用的非持续电击伤害；须按本次免疫、抗力、弱点与硬度后的实际结果确认。</p><input name="amount" type="number" min="0" step="1">',buttons:[{action:'confirm',label:'确认此回执的电击分量',callback:(_event,b)=>Number(new FormData(b.form).get('amount'))},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
    if(amount!==null)await rpc('confirmMixed',{actorUuid:target.actor.uuid,nonce:record.nonce,receiptUuid:message.uuid,amount,confirmed:true});
   }catch(error){onError(error)}});block.append(button);
  }
  (root.querySelector('.message-content')??root).append(block);
 }
 function register({Hooks,socket:api}={}){
  socket=api;activeActors.register(Hooks);
  for(const method of ['channel','beginDamage','finishDamage','confirmMixed','candidates','interact','confirmedAction'])socket?.register(`electricity:${method}`,async function(payload){try{return {ok:true,value:await ledger[method](payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message,...(method==='confirmedAction'&&error.electricityNotApplied===true?{electricityNotApplied:true}:{})}}});
  Hooks.on('preCreateChatMessage',decorateReceipt);
  Hooks.on('createChatMessage',(message,options,userId)=>{rememberSource(message);capture(message,options,userId);if(active())ledger.check(message).catch(onError)});
  Hooks.on('updateChatMessage',rememberSource);Hooks.on('deleteChatMessage',forgetSource);
  Hooks.on('renderChatMessageHTML',renderReceipt);
  for(const phase of ['start','end'])Hooks.on(`pf2e.${phase}Turn`,(combatant,combat)=>{if(active())return ledger.expire({combat,combatant,phase,actors:activeActors.values()}).catch(onError)});
  Hooks.on('deleteCombat',combat=>encounterEnded(combat).catch(onError));
  Hooks.on('updateCombat',(combat,changes)=>{if((changes.round===0||changes.round===null)&&!combat.started)encounterEnded(combat).catch(onError)});
 }
 // Startup recovery can settle only the unique current native receipt. There is
 // no retry of applyDamage and no reconstruction from a generic HP update.
 async function maintain(actor){
  if(!active()||!actor)return;
  for(const record of Object.values(electricityState(actor).damage).filter(r=>r.status==='pending'||r.status==='confirmed'&&!r.lifecycleDone)){
   const matches=values(game.messages).filter(m=>receiptMatches(m,record));if(matches.length!==1)continue;
   const user=game.users.get(record.userId);if(user&&actor.testUserPermission?.(user,'OWNER'))await ledger.finishDamage({actorUuid:actor.uuid,nonce:record.nonce,receiptUuid:matches[0].uuid},user);
  }
 }
 return {register,maintain,beforeDamage,afterDamage,observeNativeIWR,interceptDamageMessage,interceptCheck,beforeChannel,onCommittedChannel,onRefresh,
  validateSelection:context=>ledger.validateSelection(context),confirmMixed:payload=>rpc('confirmMixed',payload),interact:payload=>rpc('interact',payload),confirmedAction:payload=>rpc('confirmedAction',payload),
  diagnostic:{nativeElectricity:'source-bound native IWR and damage-taken receipt',mixedDamage:'active-GM exact attribution',unsupported:['preexisting untagged damage cards','arbitrary custom attack damage components','unrecorded touches/actions']}};
}
