import {MODULE_ID} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {getSourceId,isActiveGM,resolveMessageTargets} from './native-context.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';
import {createDirtyMaintenance,isUnrelatedMaintenanceUpdate,COSMETIC_UPDATE_FIELDS} from './maintenance-events.mjs';
import {isActualUseMessage} from './usage-events.mjs';

export const PARTY_SOURCES=Object.freeze({
 clue:'Compendium.pf2e.actionspf2e.Item.25WDi1cVUrW92sUj',clueEffect:'Compendium.pf2e.feat-effects.Item.vhSYlQiAQMLuXqoc',
 anoint:'Compendium.pf2e.feats-srd.Item.bRftzbFvSF1pilIo',anointEffect:'Compendium.pf2e.feat-effects.Item.nnF7RSVlC6swbSw8',
 guardian:'Compendium.pf2e.feats-srd.Item.YwVdaszwpDJd6kf9',
 imperial:'Compendium.pf2e.classfeatures.Item.ZEtJJ5UOlV5oTWWp',imperialEffect:'Compendium.pf2e.feat-effects.Item.vguxP8ukwVTWWWaA',
 knownEffect:'Compendium.pf2e.feat-effects.Item.DvyyA11a63FBwV7x',
 raiseShieldEffect:'Compendium.pf2e.equipment-effects.Item.2YgXoHvJfrDHucMr',
});
const values=x=>Array.from(x?.values?.()??x??[]),own=x=>x?.flags?.[MODULE_ID]?.party;
const has=(actor,key)=>values(actor?.items).some(i=>getSourceId(i)===PARTY_SOURCES[key]);
const clone=structuredClone;
const tokenDoc=t=>t?.document??t;
export function buildPartyPatreonRepairs(original){
 const rules=clone(original),changes=[];
 for(const[id,group]of Object.entries(rules??{}))for(const list of ['baseRules','handlerRules'])for(const[index,rule]of(group[list]??[]).entries()){
  if(!Array.isArray(rule.predicate))continue;
  let exclusion;
  if(group.source?.includes(PARTY_SOURCES.clue)&&rule.triggerType==='postInfo'&&rule.target==='SelfEffect'&&rule.value===PARTY_SOURCES.clueEffect&&rule.predicate.includes('origin:item:clue-in'))exclusion={not:'origin:item:sourceId:'+PARTY_SOURCES.clue};
  if(group.source?.includes(PARTY_SOURCES.anoint)&&rule.triggerType==='postInfo'&&rule.value===PARTY_SOURCES.anointEffect&&rule.predicate.includes('origin:item:anoint-ally'))exclusion={not:'origin:item:sourceId:'+PARTY_SOURCES.anoint};
  if(group.source?.includes(PARTY_SOURCES.guardian)&&rule.triggerType==='postInfo'&&rule.value==='devotedGuardian')exclusion={not:'origin:item:sourceId:'+PARTY_SOURCES.guardian};
  if(rule.triggerType==='spell-cast'&&rule.value==='bloodlines'&&rule.predicate.includes('feature:bloodline-spells'))exclusion={not:MODULE_ID+':usage:party:imperial'};
  if(!exclusion||rule.predicate.some(p=>JSON.stringify(p)===JSON.stringify(exclusion)))continue;
  const before=clone(rule.predicate);rule.predicate.push(exclusion);changes.push({path:`${id}.${list}.${index}.predicate`,before,after:clone(rule.predicate),reason:'由原始使用者与目标明确关联的协作能力统一结算'});
 }
 return {rules,changes};
}
const needsKnownWeaknessCheck=r=>r==null||(r.key==='TokenMark'&&r.slug==='known-weakness');
export function buildKnownWeaknessRepair(item){
 // Most inventory has no legacy TokenMark. Check current rules before the
 // PF2e sourceId getter; retain every maintenance hook and avoid a stale cache.
 const currentRules=item?.system?.rules;
 if(Array.isArray(currentRules)&&!currentRules.some(needsKnownWeaknessCheck))return null;
 if(getSourceId(item)!==PARTY_SOURCES.knownEffect)return null;
 const rules=clone(item.system.rules??[]),mark=rules.find(r=>r.key==='TokenMark'&&r.slug==='known-weakness');
 if(!mark||!rules.some(r=>r.key==='FlatModifier'&&r.predicate?.includes('target:mark:known-weaknesses')))return null;
 mark.slug='known-weaknesses';return rules;
}
export function isImperialBloodMagic(item){
 if(item?.type!=='spell'||!has(item.actor,'imperial')||!item.system?.traits?.otherTags?.includes('blood-magic-spell'))return false;
 const options=values(item.actor.getRollOptions?.());
 return options.includes('blood-magic:imperial')&&!options.some(o=>o.startsWith('second-blood-magic:'));
}
export function guardianActive(source,target){
 const shield=source?.actor?.attributes?.shield;
 return !!target?.actor&&!!shield?.raised&&!shield.broken&&!shield.destroyed;
}
export function createPartyAutomation({game,fromUuid=globalThis.fromUuid,choose,onError=console.error,castEvents=getNativeCastEvents({game,fromUuid})}={}){
 castEvents.addMatcher(isImperialBloodMagic);
 const queue=new SerialActions(),lastActions=new Map();
 const now=()=>game.time.worldTime;
 const gm=()=>{if(!isActiveGM(game))throw Error('协作能力必须由当前主GM结算。');};
 const resolveAction=item=>{const key=['clue','anoint','guardian'].find(k=>getSourceId(item)===PARTY_SOURCES[k]);return key?'party:'+key:isImperialBloodMagic(item)?'party:imperial':undefined};
 const requiresActualUse=(_item,action)=>['party:clue','party:anoint','party:guardian'].includes(action);
 const load=async uuid=>{const i=await fromUuid(uuid);if(!i?.toObject)throw Error('无法读取协作能力的原生效果。');const d=i.toObject();delete d._id;return d;};
 const actors=()=>{
  const seen=new Set();
  return values(game.actors).concat(values(globalThis.canvas?.scene?.tokens).map(t=>t.actor)).filter(actor=>{
   const uuid=actor?.uuid;
   // Strict equality in the old findIndex never matched NaN, even to itself.
   if(uuid!==uuid)return false;
   const first=!seen.has(uuid);seen.add(uuid);
   // Null entries still reserve undefined, matching the original first-index rule.
   return !!actor&&first;
  });
 };
 const sourceToken=async(actor,message)=>{
  const scene=message.speaker?.scene,token=message.speaker?.token;
  if(scene&&token){const t=await fromUuid(`Scene.${scene}.Token.${token}`);if(t?.actor?.uuid===actor.uuid)return t;}
  const active=values(actor.getActiveTokens?.(true,true)).map(tokenDoc);return active.length===1?active[0]:null;
 };
 async function targetFor(context,guard){
  const targets=await resolveMessageTargets(context.message,{fromUuid});guard();
  const source=await sourceToken(context.actor,context.message);guard();
  if(targets.length!==1||targets[0].actor.uuid===context.actor.uuid)throw Error(`使用此能力时请选中一个其他生物作为目标。能力：${context.item.name}`);
  return {source,target:targets[0]};
 }
 function origin(data,actor,item,token){
  data.system.context={origin:{actor:actor.uuid,item:item.uuid,token:token?.uuid??null,rollOptions:item.getOriginData?.().rollOptions??[]},target:null,roll:null};
  data.system.start={value:now(),initiative:game.combat?.combatants?.find?.(c=>c.actor?.uuid===actor.uuid)?.initiative??null};
  return data;
 }
 async function executeUsage(context){
  const {actor,item,message,user,action,frequencyReceipt}=context;gm();requireOwner(actor,user);
  if(resolveAction(item)!==action)throw Error('能力来源与使用事件不匹配。');
  const originalGM=game.user,actorUUID=actor.uuid,itemUUID=item.uuid,sourceOrigin=clone(message.flags?.pf2e?.origin??{}),speaker=clone(message.speaker??{}),input=clone(message.flags?.[MODULE_ID]?.usageInput??{}),usage=JSON.stringify(message.flags?.[MODULE_ID]?.usage??null);
  const itemSources=[item.sourceId,item._stats?.compendiumSource,item.flags?.core?.sourceId],receipt=frequencyReceipt?clone(frequencyReceipt):null;
  const lookupToken=uuid=>{const match=/^Scene\.([^.]+)\.Token\.([^.]+)$/.exec(uuid??'');return match?game.scenes.get(match[1])?.tokens.get(match[2]):null;};
  const capture=token=>{if(!token)return null;const a=token.actor;return {token,actor:a,scene:token.parent,uuid:token.uuid,synthetic:a?.isToken===true,baseActor:token.baseActor,actorId:token.actorId,actorLink:token.actorLink};};
  const sources=(speaker.scene&&speaker.token?[game.scenes.get(speaker.scene)?.tokens.get(speaker.token)]:values(actor.getActiveTokens?.(true,true)).map(tokenDoc)).map(capture).filter(Boolean);
  const nativeTarget=message.flags?.pf2e?.context?.target,recordedTarget=typeof nativeTarget?.token==='string'?nativeTarget.token:nativeTarget?.token?.uuid;
  const targetUUIDs=recordedTarget?[recordedTarget]:Array.isArray(input.targetUuids)?input.targetUuids:message.flags?.['pf2e-toolbelt']?.targetHelper?.targets??[];
  const targets=new Map(targetUUIDs.map(uuid=>{const proof=capture(lookupToken(uuid));return [proof?.token,proof];}).filter(([token])=>token));
  const tokenProofs=['party:anoint','party:guardian'].includes(action)?new Map(values(globalThis.canvas?.scene?.tokens).map(token=>[token.actor,capture(token)])):null;
  const currentActor=(a,proof)=>proof?.synthetic?a.isToken===true&&a.token===proof.token&&proof.actorLink===false&&proof.token.actorLink===false&&proof.token.actorId===proof.actorId&&proof.baseActor?.id===proof.actorId&&proof.token.baseActor===proof.baseActor&&game.actors.get(proof.actorId)===proof.baseActor:!a.isToken&&game.actors.get(a.id)===a;
  const assertToken=proof=>{
   if(!proof||proof.token.documentName!=='Token'||proof.token.uuid!==proof.uuid||proof.token.actor!==proof.actor||proof.token.parent!==proof.scene||game.scenes.get(proof.scene?.id)!==proof.scene||proof.scene.tokens.get(proof.token.id)!==proof.token||!currentActor(proof.actor,proof))throw Error('协作能力的原始Token、角色或场景已改变。');
  };
  const assertSource=()=>{
   gm();requireOwner(actor,user);
   if(game.user!==originalGM||game.users.activeGM!==originalGM||game.users.get(originalGM.id)!==originalGM||game.users.get(user.id)!==user||actor.uuid!==actorUUID||item.uuid!==itemUUID||actor.items.get(item.id)!==item||item.actor!==actor||resolveAction(item)!==action||item.suppressed===true||item.isSuppressed||item.system?.suppressed||game.messages.get(message.id)!==message||(message.author?.id??message.author??message.user?.id??message.user)!==user.id||message.speaker?.actor!==actor.id||message.flags?.pf2e?.origin?.actor!==actorUUID||message.flags.pf2e.origin.uuid!==itemUUID||JSON.stringify(message.flags.pf2e.origin)!==JSON.stringify(sourceOrigin)||JSON.stringify(message.speaker)!==JSON.stringify(speaker)||JSON.stringify(message.flags?.[MODULE_ID]?.usageInput??{})!==JSON.stringify(input)||JSON.stringify(message.flags?.[MODULE_ID]?.usage??null)!==usage||itemSources.some((source,index)=>source!==[item.sourceId,item._stats?.compendiumSource,item.flags?.core?.sourceId][index]))throw Error('协作能力的原始来源、消息或执行权限已改变；不会重放。');
   if(!sources.length||sources.some(proof=>proof.actor!==actor))throw Error('协作能力需要原始使用者Token。');
   for(const proof of sources)assertToken(proof);
   if(receipt&&(!receipt.id||receipt.itemUuid!==itemUUID||receipt.userId!==user.id||receipt.before!==1||receipt.after!==0||input.frequencyReceiptId!==receipt.id||JSON.stringify(message.flags?.[MODULE_ID]?.usage?.frequencyReceipt)!==JSON.stringify(receipt)))throw Error('协作能力的原始次数回执已改变。');
  };
  assertSource();
  return queue.run(actor.uuid,async()=>{
   assertSource();let delivered=false,nativeEntered=false;
   const write=async(fn,guard)=>{guard();nativeEntered=true;const result=await fn();guard();return result;};
   const upsert=async(recipient,key,data,guard)=>{
    if(typeof key!=='string'||!key||data?.type!=='effect')throw Error('需要有效的自动化效果及唯一标识。');
    const existing=values(recipient.items).filter(i=>i.type==='effect'&&i.flags?.[MODULE_ID]?.nativeEffectKey===key),next=clone(data);delete next._id;
    next.flags={...next.flags,[MODULE_ID]:{...next.flags?.[MODULE_ID],nativeEffectKey:key}};
    if(existing.length){
     const live=()=>{guard();if(recipient.items.get(existing[0].id)!==existing[0]||existing[0].flags?.[MODULE_ID]?.nativeEffectKey!==key)throw Error('协作能力的原始效果文档已改变；不会重放。');};
     await write(()=>existing[0].update(next),live);
     if(existing.length>1)await write(()=>{if(existing.slice(1).some(i=>recipient.items.get(i.id)!==i||i.flags?.[MODULE_ID]?.nativeEffectKey!==key))throw Error('协作能力的重复效果文档已改变。');return recipient.deleteEmbeddedDocuments('Item',existing.slice(1).map(i=>i.id));},live);
     return existing[0];
    }
    const result=await write(()=>recipient.createEmbeddedDocuments('Item',[next]),guard),saved=result?.[0];
    if(!Array.isArray(result)||result.length!==1||!saved?.id||recipient.items.get(saved.id)!==saved||saved.type!=='effect'||saved.flags?.[MODULE_ID]?.nativeEffectKey!==key)throw Error('协作能力的原生效果回执尚未确认；不会退款或重放。');
    return saved;
   };
   const removeForSource=async(kind,guard)=>{
    for(const a of actors()){
    const effects=values(a.items).filter(i=>own(i)?.kind===kind&&own(i).sourceActor===actorUUID);
    const live=()=>{guard();if(a.isToken)assertToken(tokenProofs.get(a));else if(!currentActor(a))throw Error('协作能力的原始效果所属角色已改变。');};
    if(effects.length)await write(()=>{if(effects.some(i=>a.items.get(i.id)!==i||own(i)?.kind!==kind||own(i).sourceActor!==actorUUID))throw Error('协作能力的原始效果文档已改变。');return a.deleteEmbeddedDocuments('Item',effects.map(i=>i.id));},live);
   }};
   const targetGuard=(source,target)=>{
    const sourceProof=sources.find(proof=>proof.token===source),targetProof=targets.get(target);
    return ()=>{assertSource();assertToken(sourceProof);assertToken(targetProof);if(sourceProof.actor!==actor||targetProof.actor===actor)throw Error('协作能力的原始目标已改变。');};
   };
   try{
   if(action==='party:clue'){
    const cooldown=actor.flags?.[MODULE_ID]?.party?.clueUntil,uses=item.system.frequency?.value??item.system.frequency?.max??0,max=item.system.frequency?.max;
    let expectedUses=uses,expectedCooldown=cooldown;
    const {source,target}=await targetFor(context,assertSource),bound=targetGuard(source,target);bound();const recipient=targets.get(target).actor;
    const guard=()=>{bound();if((item.system.frequency?.value??item.system.frequency?.max??0)!==expectedUses||item.system.frequency?.max!==max||actor.flags?.[MODULE_ID]?.party?.clueUntil!==expectedCooldown)throw Error('线索指引的原始次数或冷却已改变。');};guard();
    // World time can be negative; zero is a valid absolute expiry.
    if(cooldown!=null&&cooldown>now())throw Error('线索指引尚未结束10分钟冷却。');
    if(!frequencyReceipt&&uses<1)throw Error('线索指引没有可用次数。');
    const data=origin(await load(PARTY_SOURCES.clueEffect),actor,item,source);guard();
    await upsert(recipient,'clue:'+actor.uuid,data,guard);
    delivered=true;
    if(!frequencyReceipt)await write(async()=>{const result=await item.update({'system.frequency.value':uses-1},{[MODULE_ID]:{usageInternal:true}});expectedUses=uses-1;return result;},guard);
    const until=now()+600;await write(async()=>{const result=await actor.update({[`flags.${MODULE_ID}.party.clueUntil`]:until});expectedCooldown=until;return result;},guard);return `已向${recipient.name}提供下一次检定加值。`;
   }
   if(action==='party:anoint'){
    const {source,target}=await targetFor(context,assertSource),bound=targetGuard(source,target);bound();const recipient=targets.get(target).actor;
    const guard=()=>{bound();if(recipient.isAllyOf&&!recipient.isAllyOf(actor))throw Error('符血点化的目标必须是盟友。');};guard();
    const data=origin(await load(PARTY_SOURCES.anointEffect),actor,item,source);guard();
    data.flags??={};data.flags[MODULE_ID]={party:{kind:'anoint',sourceActor:actor.uuid,sourceToken:source?.uuid??null,targetToken:target.uuid,expiresAt:now()+60}};
    await removeForSource('anoint',guard);await upsert(recipient,'anoint:'+actor.uuid,data,guard);return `已点化${recipient.name}，持续1分钟。`;
   }
   if(action==='party:guardian'){
    const last=lastActions.get(actor.uuid),shield=actor.items.get(actor.attributes?.shield?.itemId),shieldType=shield?.baseType,shieldBase=shield?.system?.baseItem;
    const {source,target}=await targetFor(context,assertSource),bound=targetGuard(source,target);bound();const recipient=targets.get(target).actor;
    const guard=()=>{bound();if(!guardianActive(source,target)||!shield||actor.items.get(shield.id)!==shield||actor.attributes.shield.itemId!==shield.id||shield.baseType!==shieldType||shield.system?.baseItem!==shieldBase)throw Error('需要举起可用的盾牌。');if(last?.kind!=='raise-shield'||lastActions.get(actor.uuid)!==last)throw Error('忠诚卫士要求上一个动作是举盾；请通过角色卡或HUD举盾后使用。');};guard();
    const tower=shield?.baseType==='tower-shield'||shield?.system?.baseItem==='tower-shield';
    const data=origin({name:'忠诚卫士',type:'effect',img:item.img,system:{description:{value:''},rules:[{key:'FlatModifier',selector:'ac',type:'circumstance',value:tower?2:1,predicate:['parent:origin:shield:raised']}],duration:{value:-1,unit:'unlimited',expiry:null,sustained:false},level:{value:actor.level},traits:{value:[]},tokenIcon:{show:true}},flags:{[MODULE_ID]:{party:{kind:'guardian',sourceActor:actor.uuid,sourceToken:source.uuid,targetToken:target.uuid,shieldId:shield.id}}}},actor,item,source);
    await removeForSource('guardian',guard);await upsert(recipient,'guardian:'+actor.uuid,data,guard);lastActions.set(actor.uuid,{kind:'guardian'});return `已守护${recipient.name}；盾牌放下或不再可用时自动结束。`;
   }
   if(action==='party:imperial'){
    const marks=actors().flatMap(a=>values(a.items).filter(i=>own(i)?.kind==='anoint'&&own(i).sourceActor===actor.uuid&&!i.isExpired&&own(i).expiresAt>now()).map(i=>({actor:a,item:i,actorUUID:a.uuid,state:clone(own(i)),target:capture(lookupToken(own(i).targetToken))})));
    const choices=[{value:actor.uuid,label:`自己 · ${actor.name}`},...marks.map(m=>({value:m.actor.uuid,label:`已点化盟友 · ${m.actor.name}`}))];
    const selected=await choose({actor,user,title:'帝皇血统奥秘 · 受益者',choices});assertSource();if(!selected)return '已取消血统奥秘。';
    const mark=marks.find(m=>m.actorUUID===selected),recipient=selected===actor.uuid?actor:mark?.actor;if(!recipient)throw Error('符血点化已失效。');
    const markProof=mark?.state,recipientUUID=mark?.actorUUID??actorUUID,targetProof=mark?.target;
    const guard=()=>{assertSource();if(recipient.uuid!==recipientUUID||recipient!==actor&&(!mark||recipient.items.get(mark.item.id)!==mark.item||mark.item.isExpired||JSON.stringify(own(mark.item))!==JSON.stringify(markProof)||own(mark.item)?.expiresAt<=now()))throw Error('选择期间符血点化已失效，未转交血统奥秘。');if(recipient!==actor){assertToken(targetProof);if(targetProof.actor!==recipient)throw Error('血统奥秘的原始受益者已改变。');}};guard();
    const defense=await choose({actor,user,title:'帝皇血统奥秘 · 防护',choices:[{value:'ac',label:'AC +1状态加值'},{value:'saving-throw',label:'豁免 +1状态加值'}]});guard();if(!defense)return '已取消血统奥秘。';
    await castEvents.ensurePaid(context);
    guard();const data=await load(PARTY_SOURCES.imperialEffect);guard();const source=await sourceToken(actor,message);guard();origin(data,actor,item,source);
    const combat=game.combat,index=combat?.turns?.findIndex(c=>c.actor?.uuid===actor.uuid);
    if(combat?.started&&index>=0)data.system.duration.value=index>combat.turn?0:1;
    for(const rule of data.system.rules)if(rule.key==='ChoiceSet'&&rule.flag==='defense')rule.selection=defense;
    data.flags??={};data.flags.system={...data.flags.system,rulesSelections:{...data.flags.system?.rulesSelections,defense}};
    data.flags[MODULE_ID]={party:{kind:'imperial',sourceActor:actor.uuid}};
    await upsert(recipient,'imperial:'+actor.uuid,data,guard);return `已为${recipient.name}提供${defense==='ac'?'AC':'豁免'}加值，至施法者下回合开始。`;
   }
   }catch(error){
    if(action==='party:clue'&&receipt&&!delivered&&!nativeEntered){
     try{assertSource();}catch{throw error;}
     if(item.system.frequency?.value===receipt.after&&item.system.frequency.max===receipt.before)await item.update({'system.frequency.value':receipt.before},{[MODULE_ID]:{usageInternal:true}});
    }
    throw error;
   }
  });
 }
 async function maintain(actor){
  if(!isActiveGM(game)||!actor?.items)return;
  return queue.run(actor.uuid,async()=>{
   if(!isActiveGM(game))return;
   for(const item of values(actor.items)){
    if(!isActiveGM(game))return;
    const repaired=buildKnownWeaknessRepair(item);if(repaired){await item.update({[`flags.${MODULE_ID}.knownWeaknessBefore`]:clone(item.system.rules),'system.rules':repaired});continue;}
    const state=own(item);if(!state)continue;
    let expired=state.expiresAt&&state.expiresAt<=now();
    if(state.kind==='guardian'){
     const s=await fromUuid(state.sourceToken),t=await fromUuid(state.targetToken);
     expired=!guardianActive(s,t)||s?.actor?.attributes?.shield?.itemId!==state.shieldId||!has(s?.actor,'guardian');
    }
    if(expired&&isActiveGM(game))await item.delete();
   }
   if(!isActiveGM(game))return;
   const until=actor.flags?.[MODULE_ID]?.party?.clueUntil;if(until!=null&&until<=now()){
    const clue=values(actor.items).find(i=>getSourceId(i)===PARTY_SOURCES.clue);
    if(clue?.system.frequency)await clue.update({'system.frequency.value':clue.system.frequency.max},{[MODULE_ID]:{usageInternal:true}});
    await actor.update({[`flags.${MODULE_ID}.party.-=clueUntil`]:null});
   }
  });
 }
 function register({Hooks}={}){
  const registrations=[],on=(name,fn)=>registrations.push([name,Hooks.on(name,fn)]);
  const maintenance=createDirtyMaintenance({enabled:()=>isActiveGM(game),run:scope=>Promise.all((scope===null?actors():[...new Map([...scope].map(a=>[a.uuid,a])).values()]).map(maintain)),onError}),all=()=>maintenance.request();
  on('createItem',item=>{
   if(isActiveGM(game)&&item.actor&&getSourceId(item)===PARTY_SOURCES.raiseShieldEffect)lastActions.set(item.actor.uuid,{kind:'raise-shield'});
   if(item.type==='effect'&&item.actor)return maintenance.request(item.actor);
  });
  on('createChatMessage',m=>{
   if(!isActiveGM(game)||m.flags?.[MODULE_ID]?.usageGenerated)return;
   if(requiresActualUse(m.item,resolveAction(m.item))&&!isActualUseMessage(m))return;
   const actor=m.actor;if(!actor)return;
   const source=m.item?.sourceId,context=m.flags?.pf2e?.context;
   if(m.item?.slug==='raise-a-shield'){lastActions.set(actor.uuid,{kind:'raise-shield'});return;}
   if(source===PARTY_SOURCES.guardian)return;
   if(['attack-roll','skill-check'].includes(context?.type)||m.item?.type==='spell'||['feat','action'].includes(m.item?.type))lastActions.set(actor.uuid,{kind:'other'});
  });
  let electedGM=game.users.activeGM?.id;
  const authorityChanged=()=>{const current=game.users.activeGM?.id;if(current!==electedGM){lastActions.clear();electedGM=current;}return all();};
  on('userConnected',authorityChanged);on('updateUser',authorityChanged);on('canvasReady',all);
  on('deleteToken',all);on('deleteItem',all);on('updateItem',(item,changes={})=>{if(!isUnrelatedMaintenanceUpdate(changes,COSMETIC_UPDATE_FIELDS)&&(item.type==='shield'||getSourceId(item)===PARTY_SOURCES.raiseShieldEffect))return all()});on('updateWorldTime',all);on('updateCombat',(_combat,changes={})=>{if(isUnrelatedMaintenanceUpdate(changes,COSMETIC_UPDATE_FIELDS))return;if('round'in changes||'turn'in changes)lastActions.clear();return all()});on('deleteCombat',()=>{lastActions.clear();return all()});
  return()=>{maintenance.dispose();for(const[n,id]of registrations)Hooks.off(n,id);};
 }
 return {resolveAction,requiresActualUse,executeUsage,maintain,register,captureUsage:(item,context)=>isImperialBloodMagic(item)?castEvents.captureUsage(item,context):null};
}
