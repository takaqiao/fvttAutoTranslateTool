import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';
import {registerAvDamageSnapshot} from './av-damage-snapshot.mjs';
import {registerAvRefocusEvents} from './av-refocus-events.mjs';
import {buildAvWaveRepair,WAVE_SPELL_SLUGS,WAVE_SPELL_SOURCES,isAvWaveSpell,getAvWaveFeature,hasAvWaveEnergy} from './av-wave-repair.mjs';

export const AV_SOURCES=Object.freeze({
 shield:'Compendium.pf2e-team-plus-magic.items.Item.u8BuGKcF9VYk9TZi',
 warp:'Compendium.pf2e-team-plus-magic.items.Item.g2xM25qb607keJxw',
 coreWarp:'Compendium.pf2e.spells-srd.Item.mAMEt4FFbdqoRnkN',
 balm:'Compendium.pf2e.feats-srd.Item.pEFcWRYiWLSjxvkW',
 balmEffect:'Compendium.pf2e.feat-effects.Item.tcdIBJTHuexSKJ6S',
 restore:'Compendium.pf2e.actionspf2e.Item.XkrN7gxdRXTYYBkX',
 shake:'Compendium.pf2e.feats-srd.Item.auv1lss6LxM0q3gz',
 wheel:'Compendium.pf2e.spells-srd.Item.X4T5RlQBrdpmA35n',
 wheelEffect:'Compendium.pf2e.spell-effects.Item.znwjWUvGOFQ6VYaE',
 shieldImmunity:'Compendium.pf2e.spell-effects.Item.QF6RDlCoTvkVHRo4',
 enfeebled:'Compendium.pf2e.conditionitems.Item.MIRkyAjyBeXivMa7',
});
const SHIELDS=['6wSXeRgBrodYOOyR','eS7pZNCRasZ4DsQy','P73ulk5252swrJEB'].map(id=>'Compendium.pf2e-team-plus-magic.misc.Item.'+id);
const WARP_VARIANTS={Vjb7qHfjiCoqypoj:1,np6AuFc56zRKjwBM:2,g2KuA9pYf6agLoe6:3};
const OUTCOMES=['criticalFailure','failure','success','criticalSuccess'];
const values=c=>Array.from(c?.values?.()??c??[]);
const source=i=>i?.sourceId??i?._stats?.compendiumSource??i?.flags?.core?.sourceId??null;
const own=i=>i?.flags?.[MODULE_ID]?.av;
const clone=x=>globalThis.foundry?.utils?.deepClone?.(x)??structuredClone(x);
const slug=i=>i?.slug??i?.system?.slug;
const isGM=game=>!!game.user?.id&&game.user.id===game.users.activeGM?.id;
const effectSlug=i=>(slug(i)??'').replace(/^(?:spell-)?effect-/,'');
const hasEffect=(actor,name)=>values(actor.items).some(i=>i.type==='effect'&&(slug(i)===name||effectSlug(i)===name)&&!i.isExpired);
const degreeOf=m=>Number.isInteger(m?.flags?.pf2e?.context?.degreeOfSuccess)?m.flags.pf2e.context.degreeOfSuccess:OUTCOMES.indexOf(m?.flags?.pf2e?.context?.outcome);
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const hasBalm=actor=>values(actor?.items).some(i=>source(i)===AV_SOURCES.balm);
const psi=item=>item?.type==='spell'&&item.system?.traits?.otherTags?.includes('psi-cantrip');
const LEGACY_WAVE=['breathe-fire','heat-metal','fireball','fire-shield','cone-of-cold','flame-vortex','fiery-body','polar-ray','falling-stars','produce-flame','ray-of-frost','ignition','frostbite'].map(s=>'item:slug:'+s);
const EARLY_WAVE=['arctic-rift','blazing-bolt','breathe-fire','falling-star','fireball','fire-shield','frostbite','frozen-fog','ice-storm','ignition','volcanic-eruption'].map(s=>'item:slug:'+s);
const CURRENT_WAVE=WAVE_SPELL_SLUGS.map(s=>'item:slug:'+s);

// Native Item.getOriginData only copies actor self:* and item:* options. The
// alternate-amp toggle is neither; preserve it on the creating client.
export function buildAvCastSnapshot(item,game=globalThis.game){
 const options=item?.actor?.getRollOptions?.(['all'])??[];
 const alternates=options.filter(o=>o.startsWith('alternate-amp:')).map(o=>o.slice('alternate-amp:'.length));
 const current=game?.combat,combatId=current?.started&&current.turns?.some(c=>c.actor?.uuid===item.actor?.uuid)?current.id:null;
 return {itemUuid:item.uuid,sourceId:source(item),combatId,psiCantrip:!!psi(item),amped:item.system.traits?.otherTags?.includes('amped')??false,
  alternateAmp:alternates.length===1?alternates[0]:options.includes('alternate-amp')||alternates.length?'unknown':null};
}
function castState(item,message){
 const saved=message.flags?.[MODULE_ID]?.avCast;
 if(saved?.itemUuid===item.uuid&&saved.sourceId===source(item))return saved;
 const options=message.flags?.pf2e?.origin?.rollOptions;
 return {psiCantrip:Array.isArray(options)?options.includes('origin:item:tag:psi-cantrip'):!!psi(item),
  amped:Array.isArray(options)?options.includes('origin:item:tag:amped'):item.system.traits?.otherTags?.includes('amped'),
  alternateAmp:options?.find(o=>o.startsWith('alternate-amp:'))?.slice('alternate-amp:'.length)??(options?.includes('alternate-amp')?'unknown':null)};
}

/** Only narrow, known handler predicates change. Root owns the settings write/backup queue. */
export function buildAvPatreonRepairs(original){
 const rules=clone(original),changes=[];
 for(const[id,group]of Object.entries(rules??{}))for(const[index,rule]of (group.handlerRules??[]).entries()){
  if(group.source?.includes('Compendium.pf2e.spells-srd.Item.IxhGEKl63R4QBvkj')&&rule.value==='frostbiteAmped'&&rule.triggerType==='damage-taken'&&Array.isArray(rule.predicate)&&['item:slug:frostbite','item:tag:amped','feature:the-oscillating-wave'].every(p=>rule.predicate.includes(p))){
   if(!rule.predicate.some(p=>p?.not==='alternate-amp')){const before=clone(rule.predicate);rule.predicate.push({not:'alternate-amp'});changes.push({path:`${id}.handlerRules.${index}.predicate`,before,after:clone(rule.predicate),reason:'替代增幅不能同时获得冻伤术原增幅临时生命'});}
   continue;
  }
  if(rule.value==='theOscillatingWave'&&rule.triggerType==='spell-cast'&&Array.isArray(rule.predicate)&&rule.predicate.includes('feature:the-oscillating-wave')&&rule.predicate.includes('conservation-of-energy')){
   const before=clone(rule.predicate);
   const at=rule.predicate.findIndex(p=>Array.isArray(p?.or)&&[LEGACY_WAVE,EARLY_WAVE].some(list=>p.or.length===list.length&&list.every(s=>p.or.includes(s))));
   if(at>=0)rule.predicate[at].or=[...CURRENT_WAVE];
   const recognized=rule.predicate.some(p=>p?.or?.length===CURRENT_WAVE.length&&CURRENT_WAVE.every(s=>p.or.includes(s)));
   const exclusion={not:{or:WAVE_SPELL_SOURCES.map(s=>'origin:item:sourceId:'+s)}};
   if(recognized&&!rule.predicate.some(p=>JSON.stringify(p)===JSON.stringify(exclusion)))rule.predicate.push(exclusion);
   if(JSON.stringify(before)!==JSON.stringify(rule.predicate))changes.push({path:`${id}.handlerRules.${index}.predicate`,before,after:clone(rule.predicate),reason:'正确授予名单；精确来源交AV单次切换并冻结本次能量'});
   continue;
  }
  if(rule.value==='deleteShieldEffect'&&Array.isArray(rule.predicate)&&rule.predicate.includes('shield:block')){
   const exclusion={not:MODULE_ID+':av-dynamic-shield'};
   if(!rule.predicate.some(p=>JSON.stringify(p)===JSON.stringify(exclusion))){const before=clone(rule.predicate);rule.predicate.push(exclusion);changes.push({path:`${id}.handlerRules.${index}.predicate`,before,after:clone(rule.predicate),reason:'动态护盾格挡免疫由已认领的伤害结果统一处理'});}
   continue;
  }
  const entry=rule.value==='addShieldEffect'?['shield','origin:item:shield']:rule.value==='entropicWheel'?['wheel','origin:item:entropic-wheel']:null;
  if(!entry||!Array.isArray(rule.predicate)||!rule.predicate.includes(entry[1]))continue;
  const exclusion={not:'origin:item:sourceId:'+AV_SOURCES[entry[0]]};
  if(rule.predicate.some(p=>JSON.stringify(p)===JSON.stringify(exclusion)))continue;
  const before=clone(rule.predicate);rule.predicate.push(exclusion);
  changes.push({path:`${id}.handlerRules.${index}.predicate`,before,after:clone(rule.predicate),reason:entry[0]==='shield'?'动态护盾使用其原生动作数效果':'熵能轮由AV适配统一计数和到期'});
 }
 return {rules,changes};
}
export function getShieldPlan(actions,rank){
 if(![1,2,3].includes(actions)||!Number.isInteger(rank)||rank<1||rank>10)throw Error('动态护盾动作数或施法环级无效。');
 return {actions,rank,ac:[1,2,4][actions-1],hardness:[3,7,9][actions-1]*Math.ceil(rank/2),source:SHIELDS[actions-1]};
}
export function getWarpOutcome(actions,degree){
 if(![1,2,3].includes(actions)||!Number.isInteger(degree)||degree<0||degree>3)throw Error('虚能噬需要确定的动作版本和真实豁免结果。');
 return {enfeebled:actions===1?(degree<2?1:0):actions===2?(degree===0?1:0):degree===0?2:degree===1?1:0,immunity:actions===1};
}
function timedEffect(name,kind,now,{seconds=60,rules=[],...metadata}={}){
 return {name,type:'effect',img:'icons/svg/aura.svg',system:{slug:'third-party-av-'+kind,description:{value:''},level:{value:1},traits:{value:[]},rules,
  duration:{value:seconds/60,unit:'minutes',expiry:null,sustained:false},start:{value:now,initiative:null},tokenIcon:{show:true}},flags:{[MODULE_ID]:{av:{kind,expiresAt:now+seconds,...metadata}}}};
}
function nextTurn(game,actor){
 const combat=game.combat,index=combat?.turns?.findIndex(c=>c.actor?.uuid===actor.uuid);
 if(!combat?.started||index==null||index<0)throw Error('此效果需要已开始的遭遇及施法者先攻，才能准确在其下回合开始结束。');
 return {combatId:combat.id,combatantId:combat.turns[index].id,round:combat.round+(index>combat.turn?0:1)};
}
function timingExpired(game,timing){
 if(!timing)return false;const c=game.combats?.get(timing.combatId);
 if(!c?.started)return true;
 const index=c.turns.findIndex(t=>t.id===timing.combatantId);
 return index<0||c.round>timing.round||(c.round===timing.round&&c.turn>=index);
}
const turnKey=game=>game.combat?.started?`${game.combat.id}:${game.combat.round}:${game.combat.turn}`:null;

export function createAvAutomation({game,fromUuid=globalThis.fromUuid,choose,onError=console.error,castEvents=getNativeCastEvents({game,fromUuid}),refocusSubscribers=[],refocusPrivacy}={}){
 const queue=new SerialActions(),targetQueue=new SerialActions();let socket;
 const now=()=>game.time.worldTime;
 const gm=()=>{if(!isGM(game))throw Error('AV能力必须由当前主GM统一结算。');};
 const requireOwner=(actor,user)=>{gm();if(!actor||!user||!actor.testUserPermission(user,'OWNER'))throw Error('没有此能力所属角色的所有者权限。');};
 const saveState=(actor,key,value)=>actor.update({[`flags.${MODULE_ID}.av.${key}`]:value});
 const findKind=(actor,kind)=>values(actor.items).find(i=>own(i)?.kind===kind&&!i.isExpired&&(!own(i).expiresAt||own(i).expiresAt>now()));
 const loadEffect=async uuid=>{const document=await fromUuid(uuid);if(!document?.toObject)throw Error('未找到此能力的原生效果。');const data=document.toObject();delete data._id;return data;};
 async function replace(actor,key,data){
  const old=values(actor.items).filter(i=>own(i)?.key===key);
  data.flags??={};data.flags[MODULE_ID]??={};data.flags[MODULE_ID].av={...data.flags[MODULE_ID].av,key};
  if(old.length)await actor.deleteEmbeddedDocuments('Item',old.map(i=>i.id));
  return (await actor.createEmbeddedDocuments('Item',[data]))[0];
 }
 async function targetFor(actor,message){
  const pf=message.flags?.pf2e??{},ids=message.flags?.[MODULE_ID]?.usageInput?.targetUuids??[pf.context?.target?.token].filter(Boolean);
  if(ids.length!==1)throw Error('请在使用能力时指定一个目标。');
  const token=await fromUuid(ids[0]),target=token?.actor;
  if(!target||target.uuid===actor.uuid||!actor.isAllyOf?.(target))throw Error('此能力需要一个其他盟友目标。');
  const origin=(actor.getActiveTokens?.(true,true)??[]).find(t=>(t.document??t).parent?.id===token.parent?.id);const originToken=origin?.document??origin;
  const distance=originToken?.object?.distanceTo?.(token.object);
  if(!Number.isFinite(distance)||distance>30)throw Error('盟友必须在30尺内，并有可测量的同场景Token。');
  if(actor.canSee===false||originToken.object?.checkCollision?.(token.object.center,{type:'sight',mode:'any'}))throw Error('必须能看见目标盟友。');
  return token;
 }
 async function choice(actor,user,title,choices){
  const answer=choices.length===1?choices[0].value:await choose?.({actor,user,title,choices});
  if(!choices.some(c=>c.value===answer))throw Error('已取消能力选择。');return answer;
 }
 async function once(actor,message,operation){
  const key=message.id;if(!key)throw Error('缺少原始能力消息。');
  const prior=actor.flags?.[MODULE_ID]?.av?.receipts??[];
  if(prior.some(r=>r.id===key))return '此能力消息已结算或已认领，不会重复应用。';
  // Claim persists before any HP/condition write; uncertain failures are never automatically replayed.
  await saveState(actor,'receipts',[...prior.slice(-99),{id:key,state:'claimed',at:now()}]);
  const result=await operation();
  const current=actor.flags?.[MODULE_ID]?.av?.receipts??[];
  await saveState(actor,'receipts',current.map(r=>r.id===key?{...r,state:'done'}:r));return result;
 }
 const resolveAction=item=>{
  for(const key of ['shield','restore','shake','wheel'])if(source(item)===AV_SOURCES[key]&&(['shield','wheel'].includes(key)?item.type==='spell':['feat','action'].includes(item.type)))return 'av:'+key;
  if(isAvWaveSpell(item))return 'av:wave';
  if(psi(item)&&hasBalm(item.actor))return 'av:balm';
  return null;
 };
 castEvents.addMatcher(item=>item?.type==='spell'&&(source(item)===AV_SOURCES.wheel||isAvWaveSpell(item)||psi(item)&&hasBalm(item.actor)));
 castEvents.addCapture('avCast',item=>item?.type==='spell'&&resolveAction(item)?buildAvCastSnapshot(item,game):undefined);
 const captureUsage=(item,context)=>item?.type==='spell'&&resolveAction(item)?{avCast:buildAvCastSnapshot(item,game),...castEvents.captureUsage(item,context)}:null;
 async function prepareBalm(actor,item,message,user){
  const cast=castState(item,message);
  if(!hasBalm(actor)||!cast.psiCantrip||!cast.amped||cast.alternateAmp!=='mental-balm')return null;
  // A selected Token is not proof that no enemy is inside an arbitrary area.
  // Current self-only Wheel and explicit non-area ally casts have complete data.
  if(item.system.area)throw Error('心灵疗愈需要确认法术区域实际未影响敌人；当前区域缺少完整受影响目标，未附加效果。');
  const uuids=message.flags?.[MODULE_ID]?.usageInput?.targetUuids??[];
  const targets=await Promise.all(uuids.map(uuid=>fromUuid(uuid)));
  if(source(item)!==AV_SOURCES.wheel){
   if(!targets.length)throw Error('心灵疗愈需要本次心能戏法的实际目标。');
   if(targets.some(t=>!t?.actor||t.actor.uuid!==actor.uuid&&!actor.isAllyOf?.(t.actor)))throw Error('心灵疗愈只能用于影响自身或盟友、且不影响敌人的心能戏法。');
  }
  const active=values(actor.getActiveTokens?.(true,true)).map(t=>t.document??t);
  const origin=active.find(t=>t.id===message.speaker?.token&&t.parent?.id===message.speaker?.scene)??(active.length===1?active[0]:null);
  const available=new Map([[actor.uuid,actor]]);
  for(const token of values(origin?.parent?.tokens)){
   const target=token.actor;if(!target||token.hidden&&!user.isGM||target.uuid===actor.uuid||!actor.isAllyOf?.(target))continue;
   const distance=origin?.object?.distanceTo?.(token.object);if(Number.isFinite(distance)&&distance<=30)available.set(target.uuid,target);
  }
  const selected=await choice(actor,user,'心灵疗愈：选择自己或30尺内盟友',[...available.values()].map(a=>({value:a.uuid,label:a.name})));
  const target=available.get(selected);
  if(target.uuid!==actor.uuid&&!values(origin?.parent?.tokens).some(t=>t.actor?.uuid===target.uuid&&(!t.hidden||user.isGM)&&actor.isAllyOf?.(t.actor)&&Number.isFinite(origin?.object?.distanceTo?.(t.object))&&origin.object.distanceTo(t.object)<=30))throw Error('所选盟友已离开30尺范围，未支付聊天施法资源。');
  if(targets.some(t=>!t?.actor||t.actor.uuid!==actor.uuid&&!actor.isAllyOf?.(t.actor)))throw Error('心灵疗愈原法术目标已不符合盟友条件。');
  const data=await loadEffect(AV_SOURCES.balmEffect);
  return ()=>targetQueue.run(target.uuid,async()=>{
   data.system.level={value:message.flags?.pf2e?.origin?.castRank??item.rank??1};
   data.system.duration={value:1,unit:'minutes',expiry:null,sustained:false};data.system.start={value:now(),initiative:null};
   data.system.context={origin:{actor:actor.uuid,item:item.uuid,token:origin?.uuid??null,rollOptions:message.flags?.pf2e?.origin?.rollOptions??[]},target:null,roll:null};
   data.flags??={};data.flags[MODULE_ID]={av:{kind:'balm',casterUuid:actor.uuid,castMessageId:message.id,expiresAt:now()+60}};
   await replace(target,'balm:'+actor.uuid,data);
   return '已应用心灵疗愈：对情绪效果的意志豁免+2状态加值，持续1分钟。';
  });
 }
 async function executeUsage({actor,item,message,user,action}){
  requireOwner(actor,user);if(resolveAction(item)!==action)throw Error('能力来源与自动化入口不匹配。');
  return queue.run(actor.uuid,()=>once(actor,message,async()=>{
   const applyBalm=await prepareBalm(actor,item,message,user);
   let waveEnergy=null,waveEncounterId=null;
   if(isAvWaveSpell(item)){
    const captured=game.combats?.get(castState(item,message).combatId);
    waveEncounterId=captured?.started&&captured.turns?.some(c=>c.actor?.uuid===actor.uuid)?captured.id:null;
    const first=waveEncounterId&&actor.flags?.[MODULE_ID]?.av?.waveEncounterId!==waveEncounterId;
    const options=actor.getRollOptions?.(['all'])??[],last=first?null:options.includes('conservation-of-energy:fire')?'fire':options.includes('conservation-of-energy:cold')?'cold':null;
    waveEnergy=last?(last==='fire'?'cold':'fire'):await choice(actor,user,'热流谐摆：选择本次能量',[{value:'fire',label:'添加能量：火焰'},{value:'cold',label:'移除能量：寒冷'}]);
   }
   if(action==='av:wheel'){
    if(!turnKey(game))throw Error('熵能轮需要遭遇回合来限制每回合微子增长。');
    const rank=message.flags?.pf2e?.origin?.castRank??item.rank??item.system.level?.value;if(!Number.isInteger(rank)||rank<1||rank>10)throw Error('缺少熵能轮实际施法环级。');
   }
   if(item.type==='spell'&&(action==='av:wheel'||waveEnergy||castState(item,message).amped))await castEvents.ensurePaid({actor,item,message,user});
   if(waveEnergy){
    await actor.toggleRollOption('all','conservation-of-energy',getAvWaveFeature(actor).id,true,waveEnergy);
    if(!hasAvWaveEnergy(actor,waveEnergy))throw Error('未能保存本次热流谐摆能量，未重复应用能力。');
    await message.update({[`flags.${MODULE_ID}.avWave`]:{actorUuid:actor.uuid,itemUuid:item.uuid,sourceId:source(item),energy:waveEnergy}});
    if(waveEncounterId)await saveState(actor,'waveEncounterId',waveEncounterId);
   }
   const result=await (async()=>{
   if(action==='av:shield'){
    if(hasEffect(actor,'effect-shield-immunity')||values(actor.items).some(i=>source(i)===AV_SOURCES.shieldImmunity&&!i.isExpired&&(!own(i)?.expiresAt||own(i).expiresAt>now())))throw Error('护盾术仍在10分钟免疫期内。');
    const current=message.item??item;let actions=Number(current.system?.time?.value);
    if(![1,2,3].includes(actions))actions=Number(await choice(actor,user,'动态护盾：施放动作数',[1,2,3].map(n=>({value:String(n),label:`${n}动作`}))));
    const rank=message.flags?.pf2e?.origin?.castRank??current.rank??item.rank??item.system.level?.value,plan=getShieldPlan(actions,rank),timing=nextTurn(game,actor);
    const effect=await loadEffect(plan.source);effect.system.level={value:rank};effect.system.start={value:now(),initiative:actor.combatant?.initiative??null};
    effect.flags??={};effect.flags[MODULE_ID]={av:{kind:'shield',casterUuid:actor.uuid,actions,rank,timing,castMessageId:message.id}};
    await replace(actor,'shield',effect);return `动态护盾已生效：AC +${plan.ac}，硬度${plan.hardness}。`;
   }
   if(action==='av:restore'){
    if(!hasEffect(actor,'unleash-psyche'))throw Error('心灵再造需要心力解放正在生效。');
    const token=await targetFor(actor,message),target=token.actor;
    return targetQueue.run(target.uuid,async()=>{
     if(findKind(target,'restore-immunity'))throw Error('该盟友对心灵再造仍处于10分钟暂时免疫。');
     const selected=await choice(actor,user,'心灵再造：选择给予盟友的效果',[{value:'healing',label:`恢复${2+2*actor.level}生命值`},{value:'save',label:'对心灵效果豁免+1状态加值'}]);
     await replace(target,'restore-immunity',timedEffect('心灵再造：暂时免疫','restore-immunity',now(),{seconds:600,casterUuid:actor.uuid}));
     if(selected==='healing')await target.applyDamage({damage:-(2+2*actor.level),token,item,rollOptions:new Set(item.getRollOptions?.('item')??[])});
     else await replace(target,'restore-save:'+actor.uuid,timedEffect('心灵再造：心灵豁免','restore-save',now(),{casterUuid:actor.uuid,seconds:600,rules:[{key:'FlatModifier',selector:'saving-throw',type:'status',value:1,predicate:['item:trait:mental']}]}));
     return selected==='healing'?'已按原生治疗结算并登记10分钟免疫。':'盟友已获得心灵豁免加值，随你的心力解放结束。';
    });
   }
   if(action==='av:shake'){
    if(!hasEffect(actor,'rage'))throw Error('摆脱困境只能在狂暴中使用。');
    if(actor.getCondition('frightened'))await actor.decreaseCondition('frightened');
    const sickened=actor.getCondition('sickened');if(!sickened)return '惊惧已降低1；没有恶心状态。';
    const dc=sickened.flags?.['patreon-v3']?.dc??sickened.system?.context?.roll?.dc?.value;
    if(!Number.isFinite(dc)||dc<=0)return '惊惧已降低1；恶心未记录来源DC，请在现有状态DC字段补入后正常干呕，未擅自减少恶心。';
    let reduction=null;await actor.saves.fortitude.roll({dc:{value:dc},skipDialog:true,item,extraRollOptions:['action:shake-it-off'],callback:async(_roll,outcome,rollMessage)=>{
     const degree=OUTCOMES.indexOf(outcome)>=0?OUTCOMES.indexOf(outcome):degreeOf(rollMessage);
     if(degree<0||degree>3)return;reduction=degree;
     // Operate on the same source condition, never walk down unrelated independent sources.
     for(let k=0;k<degree&&actor.items.has(sickened.id);k++)await actor.decreaseCondition(sickened);
    }});
    return reduction==null?'惊惧已降低1；恶心豁免尚未完成。':`惊惧已降低1；恶心降低${reduction}。`;
   }
   if(action==='av:wheel'){
    const turn=turnKey(game);if(!turn)throw Error('熵能轮需要遭遇回合来限制每回合微子增长。');
    const rank=message.flags?.pf2e?.origin?.castRank??item.rank??item.system.level?.value;if(!Number.isInteger(rank)||rank<1||rank>10)throw Error('缺少熵能轮实际施法环级。');
    const cast=castState(item,message),amped=cast.amped&&!cast.alternateAmp;
    const effect=await loadEffect(AV_SOURCES.wheelEffect);effect.system.level={value:rank};effect.system.badge={...effect.system.badge,value:Math.min(rank,amped?2:1),max:rank};
    effect.system.duration={value:1,unit:'minutes',expiry:null,sustained:false};effect.system.start={value:now(),initiative:null};
    effect.flags??={};effect.flags[MODULE_ID]={av:{kind:'wheel',casterUuid:actor.uuid,expiresAt:now()+60,lastTurn:turn,step:amped?2:1,castMessageId:message.id}};
    const oldNative=values(actor.items).filter(i=>source(i)===AV_SOURCES.wheelEffect&&!own(i));
    if(oldNative.length)await actor.deleteEmbeddedDocuments('Item',oldNative.map(i=>i.id));
    await replace(actor,'wheel',effect);return '熵能轮已开始：微子按回合自动累积，一分钟后结束。';
   }
   })();
   const balm=applyBalm?await applyBalm():'';
   return [result,balm,waveEnergy?`本次热流谐摆使用${waveEnergy==='fire'?'火焰':'寒冷'}能量，原卡伤害已固定。`:''].filter(Boolean).join(' ')||'本次施法未采用心灵疗愈增幅。';
  }));
 }
 async function processMessage(message){
  if(!isGM(game)||message.flags?.[MODULE_ID]?.avProcessed||message.flags?.pf2e?.appliedDamage?.isReverted)return;
  const pf=message.flags?.pf2e??{},item=message.item??(pf.origin?.uuid?await fromUuid(pf.origin.uuid):null);
  if(!item)return;
  const roller=message.actor??game.actors.get?.(message.speaker?.actor),user=game.users.get(author(message));
  if(!user||!roller?.testUserPermission?.(user,'OWNER'))return;
  if(message.isCheckRoll&&pf.context?.type==='saving-throw'&&[AV_SOURCES.warp,AV_SOURCES.coreWarp].includes(source(item))){
   if(roller.modeOfBeing&&roller.modeOfBeing!=='living')return;
   const core=source(item)===AV_SOURCES.coreWarp,degree=degreeOf(message),overlay=pf.origin?.variant?.overlays?.find(id=>WARP_VARIANTS[id]),actions=core?2:WARP_VARIANTS[overlay]??Number(item.system?.time?.value);
   if(degree<0||degree>3)return;
   const outcome=core?{enfeebled:degree===0?1:0,immunity:false}:getWarpOutcome(actions,degree),caster=await fromUuid(pf.origin?.actor??item.actor?.uuid);if(!caster||!roller)return;
   return queue.run(roller.uuid,async()=>{
    if(message.flags?.[MODULE_ID]?.avProcessed)return;
    const previousId=pf.context.isReroll?pf.context.options?.find(o=>o.startsWith(MODULE_ID+':av-reroll:'))?.slice((MODULE_ID+':av-reroll:').length):null;
    const previousSources=previousId?values(roller.items).filter(i=>own(i)?.saveMessageId===previousId&&own(i).casterUuid===caster.uuid&&own(i).sourceItemUuid===item.uuid):[];
    const previous=previousSources.filter(i=>['warp','core-warp'].includes(own(i).kind));
    if(actions===1&&findKind(roller,'warp-immunity')&&!previousSources.length)return message.update({[`flags.${MODULE_ID}.avProcessed`]:{status:'immune'}});
    const timing=outcome.enfeebled?(previous.find(i=>own(i).timing)?own(previous.find(i=>own(i).timing)).timing:nextTurn(game,caster)):null;
    await message.update({[`flags.${MODULE_ID}.avProcessed`]:{status:'claimed'}});
    if(previous.length)await roller.deleteEmbeddedDocuments('Item',previous.map(i=>i.id));
    if(outcome.enfeebled)await replace(roller,previousId?'warp-reroll:'+message.id:(core?'core-warp:':'warp:')+caster.uuid,timedEffect(core?'虚能噬：虚弱':'动态虚能噬：虚弱',core?'core-warp':'warp',now(),{seconds:3600,casterUuid:caster.uuid,timing,sourceItemUuid:item.uuid,saveMessageId:message.id,
     rules:[{key:'GrantItem',uuid:AV_SOURCES.enfeebled,allowDuplicate:true,onDeleteActions:{grantee:'cascade'},alterations:[{mode:'override',property:'badge-value',value:outcome.enfeebled}]}]}));
    if(outcome.immunity&&!previousId)await replace(roller,'warp-immunity',timedEffect('动态虚能噬：一动作版暂时免疫','warp-immunity',now(),{seconds:60,sourceItemUuid:item.uuid,saveMessageId:message.id,casterUuid:caster.uuid}));
    await message.update({[`flags.${MODULE_ID}.avProcessed`]:{status:'done'}});
   });
  }
  const caster=item.actor;
  if(!caster||source(item)===AV_SOURCES.wheel)return;
  const type=pf.context?.type,options=[...pf.origin?.rollOptions??[],...pf.context?.options??[]];
  const coldFire=(item.system?.traits?.value??[]).some(t=>t==='fire'||t==='cold')||options.some(o=>/^(?:origin:)?item:(?:trait|damage:type):(cold|fire)$/.test(o));
  const used=!message.isRoll&&!message.isCheckRoll&&!message.rolls?.length&&
   (item.type==='spell'&&options.includes('origin:action:slug:cast-a-spell')||['action','feat'].includes(item.type)&&options.includes('origin:action:slug:use-action'));
  // A condition/effect belongs to the damaged actor, not necessarily its creator.
  // Native persistent conditions do not retain a reliable damage-dealer UUID.
  const dealt=type==='damage-taken'&&['weapon','spell','feat','action'].includes(item.type)&&(pf.appliedDamage?.updates??[]).some(u=>u.value>0&&u.path.startsWith('system.attributes.hp.'));
  let damageQualifies=false,ambiguous=false;
  if(dealt){
   const marker=options.find(o=>o.startsWith(MODULE_ID+':source:'));
   const [messageId,index]=marker?.slice((MODULE_ID+':source:').length).split(':')??[];
   const original=game.messages.get(messageId),roll=original?.isDamageRoll&&original.rolls?.[Number(index)];
   const instances=values(roll?.instances).filter(i=>i.total>0&&(!i.persistent||i.options?.evaluatePersistent));
   // HP/temp-HP changes prove net damage, but not which component of a mixed
   // roll survived IWR and hardness. Never infer that from the weapon's tags.
   const energy=instances.filter(i=>['cold','fire'].includes(i.type));
   damageQualifies=energy.length>0&&energy.length===instances.length;
   ambiguous=energy.length>0&&!damageQualifies;
  }
  if(!(used&&coldFire)&&!damageQualifies&&!ambiguous)return;
  return queue.run(caster.uuid,async()=>{
   if(message.flags?.[MODULE_ID]?.avProcessed)return;
   const effect=findKind(caster,'wheel'),turn=turnKey(game);if(!effect||!turn)return;
   if(ambiguous&&!(used&&coldFire))return message.update({[`flags.${MODULE_ID}.avProcessed`]:{status:'wheel-uncertain',reason:'混合伤害仅记录总生命变化，无法确定寒冷/火焰分量穿过IWR；未擅自增加微子。'}});
   await message.update({[`flags.${MODULE_ID}.avProcessed`]:{status:'wheel'}});
   if(own(effect).lastTurn===turn)return;
   const maximum=effect.system.level.value,next=Math.min(maximum,effect.system.badge.value+own(effect).step);
   await effect.update({'system.badge.value':next,[`flags.${MODULE_ID}.av.lastTurn`]:turn});
  });
 }
 async function maintain(actor){
  if(!isGM(game)||!actor)return;
  return queue.run(actor.uuid,async()=>{
   const remove=[];
   for(const item of values(actor.items)){
    const waveRepair=buildAvWaveRepair(item);if(waveRepair)await item.update(waveRepair);
    const data=own(item);if(!data)continue;
    let expired=Number.isFinite(data.expiresAt)&&data.expiresAt<=now()||timingExpired(game,data.timing);
    if(data.kind==='restore-save'){const caster=await fromUuid(data.casterUuid);expired||=!caster||!hasEffect(caster,'unleash-psyche');}
    if(expired)remove.push(item.id);
   }
   if(remove.length)await actor.deleteEmbeddedDocuments('Item',remove);
  });
 }
 async function claimShield(payload,user){
  gm();const actor=await fromUuid(payload?.actorUuid);requireOwner(actor,user);
  return queue.run(actor.uuid,async()=>{
   const token=await fromUuid(payload.tokenUuid),message=game.messages.get(payload.messageId),effect=actor.items.get(payload.effectId);
   const target=message?.flags?.pf2e?.context?.target,helper=message?.flags?.['pf2e-toolbelt']?.targetHelper;
   if(token?.actor?.uuid!==actor.uuid||!message?.isDamageRoll||!Number.isInteger(payload.rollIndex)||!message.rolls?.[payload.rollIndex]||message.rolls[payload.rollIndex].total<=0)throw Error('动态护盾伤害来源或Token不匹配。');
   if(target?.actor!==actor.uuid&&target?.token!==token.uuid&&!helper?.targets?.includes(token.uuid))throw Error('此伤害卡没有以护盾持有者为目标。');
   if(own(effect)?.kind!=='shield'||effect.uuid!==payload.effectUuid||actor.attributes?.shield?.itemId!==effect.id||!actor.attributes.shield.raised||actor.attributes.shield.broken||actor.attributes.shield.destroyed||effect.isExpired||timingExpired(game,own(effect).timing))return null;
   if(own(effect).claim)return null;
   const receipt={actorUuid:actor.uuid,tokenUuid:token.uuid,effectUuid:effect.uuid,effectId:effect.id,messageId:message.id,rollIndex:payload.rollIndex,nonce:globalThis.foundry?.utils?.randomID?.(32)??globalThis.crypto.randomUUID()};
   await effect.update({[`flags.${MODULE_ID}.av.claim`]:{...receipt,state:'claimed',userId:user.id}});return receipt;
  });
 }
 async function completeShield({receipt,result},user){
  gm();if(!receipt)return;const actor=await fromUuid(receipt.actorUuid);requireOwner(actor,user);
  return queue.run(actor.uuid,async()=>{
   const effect=actor.items.get(receipt.effectId),claim=own(effect)?.claim;
   if(!claim||claim.nonce!==receipt.nonce||claim.state!=='claimed'||claim.effectUuid!==receipt.effectUuid||claim.messageId!==receipt.messageId||claim.tokenUuid!==receipt.tokenUuid||claim.rollIndex!==receipt.rollIndex||claim.userId!==user.id)return;
   if(!result.applied){
    await effect.update(result.uncertain?{[`flags.${MODULE_ID}.av.claim`]:{...claim,state:'uncertain'}}:{[`flags.${MODULE_ID}.av.-=claim`]:null});return;
   }
   const immunity=await loadEffect(AV_SOURCES.shieldImmunity);immunity.system.duration={value:10,unit:'minutes',expiry:null,sustained:false};immunity.system.start={value:now(),initiative:null};
   immunity.flags??={};immunity.flags[MODULE_ID]={av:{kind:'shield-immunity',expiresAt:now()+600}};
   await effect.update({[`flags.${MODULE_ID}.av.claim`]:{...claim,state:'done'}});
   await replace(actor,'shield-immunity',immunity);await actor.deleteEmbeddedDocuments('Item',[effect.id]);
  });
 }
 async function rpc(method,payload){
  if(isGM(game))return method==='av-shield-claim'?claimShield(payload,game.user):completeShield(payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('动态护盾格挡需要在线主GM。');
  const response=await socket.executeAsUser(method,game.users.activeGM.id,payload);if(!response?.ok)throw Error(response?.error??'动态护盾认领失败。');return response.value;
 }
 async function beforeDamage(contextualActor,params){
  const effect=contextualActor.items?.get(contextualActor.attributes?.shield?.itemId);
  if(own(effect)?.kind!=='shield'||!params.shieldBlockRequest||params.final)return {params};
  const damage=params.damage,options=Array.from(params.rollOptions??[]),sourceOption=options.find(o=>o.startsWith(MODULE_ID+':source:'));
  if(!sourceOption||!params.token?.uuid||!damage||typeof damage==='number'||damage.total<=0)throw Error('动态护盾需要原始伤害卡与目标Token，未应用伤害。');
  const physical=damage.instances?.every(i=>['bludgeoning','piercing','slashing','bleed'].includes(i.type)),magical=params.item?.isMagical||options.some(o=>/^(?:(?:origin:)?item:)?(?:trait:)?(?:magical|arcane|divine|occult|primal)$/.test(o));
  if(!physical&&!magical)throw Error('动态护盾不能格挡非物理且非魔法效果的伤害。');
  const [messageId,index]=sourceOption.slice((MODULE_ID+':source:').length).split(':');
  const receipt=await rpc('av-shield-claim',{actorUuid:contextualActor.uuid,tokenUuid:params.token.uuid,effectUuid:effect.uuid,effectId:effect.id,messageId,rollIndex:Number(index)});
  return receipt?{params:{...params,rollOptions:new Set([...options,MODULE_ID+':av-dynamic-shield'])},receipt}:{params:{...params,shieldBlockRequest:false}};
 }
 async function afterDamage(receipt,result){if(receipt)await rpc('av-shield-complete',{receipt,result});}
 function register({Hooks,libWrapper,socket:socketApi,onError:report=onError}={}){
  socket=socketApi;const subscriptions=[];
  const unregisterCast=castEvents.register({libWrapper,socket:socketApi});
  const unregisterDamageSnapshot=registerAvDamageSnapshot({game,libWrapper});
  const unregisterRefocus=registerAvRefocusEvents({game,Hooks,libWrapper,refocusPrivacy,onError:report,runExclusive:(actor,fn)=>queue.run(actor.uuid,fn),
   actorMatchers:refocusSubscribers.map(subscriber=>subscriber.matchesActor),
   onRefocus:refocusSubscribers.length?async event=>{for(const subscriber of refocusSubscribers)if(subscriber.matchesActor(event.actor))await subscriber.onRefocus(event)}:undefined});
  if(socket)for(const[method,fn]of [['av-shield-claim',claimShield],['av-shield-complete',completeShield]])socket.register(method,async function(payload){
   try{return {ok:true,value:await fn(payload,game.users.get(this.socketdata.userId))};}catch(error){return {ok:false,error:error.message};}
  });
  const on=(name,fn)=>subscriptions.push([name,Hooks.on(name,fn)]);
  on('preCreateChatMessage',message=>{
   const item=message.item;if(item?.type==='spell'&&resolveAction(item)&&!message.flags?.[MODULE_ID]?.avCast){
    // Legacy drafts may have no creator snapshot. Origin tags take precedence
    // over a live reconstructed item; missing alternate choice is not invented.
    const saved=message.flags?.pf2e?.origin?.rollOptions?{itemUuid:item.uuid,sourceId:source(item),...castState(item,message)}:buildAvCastSnapshot(item);
    message.updateSource({[`flags.${MODULE_ID}.avCast`]:saved});
   }
  });
  const rerollPath='game.pf2e.Check.rerollFromMessage';
  if(libWrapper)libWrapper.register(MODULE_ID,rerollPath,async function(wrapped,message,options={}){
   const pf=message?.flags?.pf2e;
   if(!message?.isCheckRoll||pf?.context?.type!=='saving-throw'||![AV_SOURCES.warp,AV_SOURCES.coreWarp].includes(source(message.item)))return wrapped(message,options);
   const original=[...pf.context.options??[]],marker=MODULE_ID+':av-reroll:';
   // PF2e reroll clones only core/pf2e flags. A scoped roll option survives that
   // reconstruction and binds the replacement save to this exact original.
   message.updateSource({'flags.pf2e.context.options':[...original.filter(o=>!o.startsWith(marker)),marker+message.id]});
   try{return await wrapped(message,options);}finally{message.updateSource({'flags.pf2e.context.options':original});}
  },'WRAPPER');
  on('createChatMessage',message=>processMessage(message).catch(report));
  on('updateChatMessage',message=>processMessage(message).catch(report));
  const all=()=>{if(isGM(game)){
   const actors=new Map(values(game.actors).map(a=>[a.uuid,a]));
   for(const scene of values(game.scenes))for(const token of values(scene.tokens))if(token.actor)actors.set(token.actor.uuid,token.actor);
   for(const combatant of values(game.combat?.combatants))if(combatant.actor)actors.set(combatant.actor.uuid,combatant.actor);
   for(const actor of actors.values())maintain(actor).catch(report);
  }};
  on('updateCombat',all);on('deleteCombat',all);on('updateWorldTime',all);on('deleteItem',item=>{if(effectSlug(item)==='unleash-psyche')all();});
  return ()=>{for(const[name,id]of subscriptions)Hooks.off(name,id);unregisterCast();unregisterDamageSnapshot();unregisterRefocus();if(libWrapper)libWrapper.unregister(MODULE_ID,rerollPath);};
 }
 return {resolveAction,captureUsage,executeUsage,processMessage,maintain,register,beforeDamage,afterDamage};
}
