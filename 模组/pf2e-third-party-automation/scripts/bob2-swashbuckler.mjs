import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';

export const BOB2_SWASHBUCKLER_SOURCES=Object.freeze({
 panache:'Compendium.pf2e.feat-effects.Item.uBJsxCzNhje8m8jj',
 temporary:'Compendium.pf2e.feat-effects.Item.paKKHPds77fq5VTH',
 feature:'Compendium.pf2e.classfeatures.Item.LzYi0OuOoypNb6jd',
 flying:'Compendium.pf2e.feats-srd.Item.qgNc5XwjsaWET0Op',
 riposte:'Compendium.pf2e.actionspf2e.Item.EfjoIuDmtUn4yiow',
 charmed:'Compendium.pf2e.feats-srd.Item.DkoxNw9tsFFXrfJY',
});
const S=BOB2_SWASHBUCKLER_SOURCES,PARRY='WJjRqP7GYCWlSnXw',PANACHE='wmvqDMawtx5dcYHd',MARKER=ID+':swashbuckler:';
const values=c=>Array.from(c?.values?.()??c??[]),copy=v=>structuredClone(v),equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const enabled=game=>BOB2_AUTOMATION_WORLDS.has(game?.world?.id)&&game.system?.id==='pf2e';
const feature=(actor,source)=>values(actor?.items).find(i=>getSourceId(i)===source);
const outcomes=new Set(['criticalSuccess','success','failure','criticalFailure']);
const successful=outcome=>outcome==='success'||outcome==='criticalSuccess';
const optionsOf=m=>new Set(m.flags?.pf2e?.context?.options??[]);
const author=m=>m.author?.id??m.user?.id??m.user;
const own=m=>m.flags?.[ID]?.swashbuckler;

/** Toolbelt owns the native ChoiceSet selfEffect. Keep all other saved rules. */
export function buildBob2SwashbucklerPatreonRepairs(original,{game}={}){
 const rules=copy(original),changes=[];
 if(!enabled(game)||game.modules?.get('patreon-v3')?.active!==true)return {rules,changes};
 const parry=rules?.[PARRY];
 if(parry?.uuid===PARRY&&equal(parry.source,['Compendium.pf2e.feats-srd.Item.WcY7H7mRiK2VDquV'])&&Array.isArray(parry.baseRules)){
  const obsolete=r=>r.type==='base'&&r.triggerType==='postInfo'&&r.target==='SelfEffect'&&r.value==='Compendium.pf2e.feat-effects.Item.1nCwQErK6hpkNvfw'&&equal(r.predicate,['origin:item:extravagant-parry'])&&r.range==null&&r.extend==null&&equal(r.values,[]);
  if(parry.baseRules.filter(obsolete).length===1){
   const before=copy(parry.baseRules);parry.baseRules=parry.baseRules.filter(r=>!obsolete(r));
   changes.push({path:PARRY+'.baseRules',before,after:copy(parry.baseRules),reason:'奢华拨挡由原生自效果选择 +1/+2，移除旧固定 +2 的 Dueling Parry 映射。'});
  }
 }
 const panache=rules?.[PANACHE];
 if(panache?.uuid===PANACHE&&equal(panache.source,[S.feature])&&Array.isArray(panache.handlerRules)){
  const melee=['item:melee','finisher','self:effect:panache',{or:['finisher:confident','agile','finesse']}];
  const thrown=['feat:flying-blade','item:ranged','item:thrown','target:range-increment:1',{or:['item:trait:agile','item:trait:finesse']},'finisher','self:effect:panache'];
  for(const [index,rule]of panache.handlerRules.entries()){
   const outcome=rule.triggerType==='attack-roll'?[{or:['outcome:failure','outcome:criticalFailure']}]:[];
   if(rule.type!=='handler'||rule.target!=='none'||rule.value!=='deletePanache'||!['attack-roll','damage-roll'].includes(rule.triggerType)||!equal(rule.predicate,[...melee,...outcome]))continue;
   const before=copy(rule.predicate);rule.predicate=[{or:[{and:copy(melee)},{and:copy(thrown)}]},...outcome];
   changes.push({path:`${PANACHE}.handlerRules.${index}.predicate`,before,after:copy(rule.predicate),reason:'派头消费继续由 Patreon 处理；仅补飞刀手在第一射程增量内使用灵巧或娴熟投掷武器的终结。'});
  }
 }
 return {rules,changes};
}

/** Patreon owns skill bravado and all finisher consumption; Trove owns Parry
 * misses and Unbalancing Finisher. Only explicit reaction checks grant here. */
export function createBob2SwashbucklerProvider({game,fromUuid=globalThis.fromUuid,choose,onError=()=>{},randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID()}={}){
 const scopes=new Map(),queue=new SerialActions();
 const managed=actor=>enabled(game)&&actor?.type==='character'&&!!feature(actor,S.feature);
 const panaches=actor=>values(actor.items).filter(i=>i.type==='effect'&&[S.panache,S.temporary].includes(getSourceId(i)));
 const temporary=(item,actor)=>item.system.duration?.unit!=='encounter'||getSourceId(actor.items.get(item.flags?.pf2e?.grantedBy?.id))===S.temporary;
 async function clearToggle(actor,item,option){
  const rule=item.system.rules?.find(r=>r.key==='RollOption'&&r.option===option);
  if(rule?.value===true)await actor.toggleRollOption(rule.domain??'all',option,item.id,false);
 }
 async function interceptCheck(native,check,context={},event=null,callback){
  const candidate=context.actor??(context.origin?.self?context.origin.actor:context.target?.self?context.target.actor:null),worldActor=game.actors?.get(candidate?.id);
  const actor=worldActor?.uuid===candidate?.uuid?worldActor:candidate,options=new Set(context.options??[]);
  if(!managed(actor)||actor.testUserPermission?.(game.user,'OWNER')!==true||context.isReroll)return native(check,context,event,callback);
  const kind=context.type==='saving-throw'&&options.has('charmed-life')?'charmed':context.type==='attack-roll'&&context.domains?.includes('strike-attack-roll')&&options.has('item:melee')&&options.has('opportune-riposte')?'riposte':null;
  const item=kind&&feature(actor,S[kind]);if(!item)return native(check,context,event,callback);
  const option=kind==='charmed'?'charmed-life':'opportune-riposte';let marker;
  try{
   const answer=await choose({actor,user:game.user,title:kind==='charmed'?'快乐人生：本次豁免':'适时反击：本次近战打击',choices:[{value:'use',label:kind==='charmed'?'使用快乐人生反应（本次豁免 +2 环境加值）':'确认本次打击使用适时反击反应'},{value:'decline',label:'正常掷骰，不使用本次反应'}]});
   if(answer==='use'){
    marker=MARKER+randomId();scopes.set(marker,{actorUuid:actor.uuid,itemUuid:item.uuid,kind});options.add(marker);
   }else options.delete(option);
   // PF2e Check.roll recalculates predicates using these options, then runs its
   // normal modifiers dialog, roll, message and caller callback exactly once.
   context.options=options;
   return await native(check,context,event,callback);
  }finally{
   if(marker)scopes.delete(marker);
   try{await clearToggle(actor,item,option)}catch(error){onError(error)}
  }
 }
 function captureCheck(message,_data,_options,userId){
  const actor=message.actor,context=message.flags?.pf2e?.context,options=optionsOf(message);
  if(userId!==game.user.id||!managed(actor)||!message.isCheckRoll||!outcomes.has(context?.outcome)||context.isReroll||!Number.isFinite(message.rolls?.[0]?.total))return;
  const marker=[...options].find(o=>scopes.has(o)),scope=scopes.get(marker);
  const kind=scope?.actorUuid===actor.uuid?scope.kind:null;
  if(!kind)return;
  message.updateSource({[`flags.${ID}.swashbuckler`]:{nonce:marker.slice(MARKER.length),kind,actorUuid:actor.uuid,itemUuid:scope.itemUuid,status:'pending'}});
 }
 async function grant(actor,temporaryResult){
  const existing=panaches(actor);
  if(existing.some(i=>getSourceId(i)===S.panache&&!temporary(i,actor)&&!i.isExpired))return;
  const source=await fromUuid(S.panache);if(!source)throw Error('找不到原生派头效果。');
  const effect=source.toObject();delete effect._id;
  // Use the same Panache effect as Patreon, with native duration metadata. A
  // second GrantItem wrapper would cascade-delete an upgraded encounter effect.
  if(temporaryResult){
   const current=game.combat?.combatant?.initiative,owner=actor.combatant?.initiative;
   const rounds=Number.isFinite(current)&&Number.isFinite(owner)&&current>owner?0:1;
   effect.system.duration={unit:'rounds',value:rounds,expiry:'turn-end',sustained:false};
  }
  effect.system.start={value:game.time.worldTime,initiative:actor.combatant?.initiative??null};
  effect.flags={...effect.flags,[ID]:{...effect.flags?.[ID],swashbucklerPanache:{combatId:game.combat.id}}};
  if(existing.length)await actor.deleteEmbeddedDocuments('Item',existing.map(i=>i.id));
  await actor.createEmbeddedDocuments('Item',[effect]);
 }
 async function processCheck(message){
  const actor=message.actor;
  if(!isActiveGM(game)||!managed(actor)||!own(message)||game.messages.get(message.id)!==message)return;
  return queue.run(actor.uuid,async()=>{
   const receipt=own(message),context=message.flags?.pf2e?.context,options=optionsOf(message),user=game.users.get(author(message));
   if(!isActiveGM(game)||receipt.status!=='pending'||receipt.actorUuid!==actor.uuid||typeof receipt.nonce!=='string'||!receipt.nonce||!user?.active||actor.testUserPermission?.(user,'OWNER')!==true||!message.isCheckRoll||!Number.isFinite(message.rolls?.[0]?.total)||!outcomes.has(context?.outcome)||context.isReroll)return;
   const history=actor.flags?.[ID]?.swashbucklerChecks??[];
   if(history.includes(receipt.nonce))return;
   const kind=receipt.kind;
   if(kind==='riposte'||kind==='charmed'){
    if(!options.has(MARKER+receipt.nonce)||feature(actor,S[kind])?.uuid!==receipt.itemUuid)return;
    if(kind==='riposte'&&(context.type!=='attack-roll'||!context.domains?.includes('strike-attack-roll')||!options.has('item:melee')||!options.has('opportune-riposte')))return;
    if(kind==='charmed'&&(context.type!=='saving-throw'||!options.has('charmed-life')))return;
   }else return;
   // Claim before document mutation: duplicate hooks cannot restore spent panache.
   await actor.update({[`flags.${ID}.swashbucklerChecks`]:[...history.slice(-63),receipt.nonce]});
   await message.update({[`flags.${ID}.swashbuckler.status`]:'settling'});
   if(game.combat?.started){
    if(successful(context.outcome))await grant(actor,kind==='charmed');
    else if(kind==='riposte'&&context.outcome==='failure')await grant(actor,true);
   }
   await message.update({[`flags.${ID}.swashbuckler.status`]:'settled'});
  });
 }
 async function endEncounter(combat){
  if(!isActiveGM(game)||!enabled(game))return;
  for(const actor of new Set(values(combat.combatants).map(c=>c.actor).filter(Boolean))){
   await queue.run(actor.uuid,async()=>{
    const effects=panaches(actor).filter(i=>i.system.duration.unit==='rounds'&&i.flags?.[ID]?.swashbucklerPanache?.combatId===combat.id);
    if(effects.length)await actor.deleteEmbeddedDocuments('Item',effects.map(i=>i.id));
   });
  }
 }
 function register({Hooks}){
  if(!enabled(game))return()=>{};
  const handles=[['preCreateChatMessage',Hooks.on('preCreateChatMessage',captureCheck)],['createChatMessage',Hooks.on('createChatMessage',message=>processCheck(message).catch(onError))],['deleteCombat',Hooks.on('deleteCombat',combat=>endEncounter(combat).catch(onError))]];
  return()=>{for(const[name,id]of handles)Hooks.off(name,id);scopes.clear()};
 }
 return {interceptCheck,processCheck,register};
}
