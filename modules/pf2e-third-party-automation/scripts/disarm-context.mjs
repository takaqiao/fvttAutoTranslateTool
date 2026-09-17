import {MODULE_ID} from './rules.mjs';
import {showNativeChoice} from './native-context.mjs';
import {disarmWeaponOption,getWeakenedGrasp} from './disarming-block.mjs';

const values=c=>Array.from(c?.values?.()??c??[]);
const tokenDocument=t=>t?.document??(t?.documentName==='Token'?t:null);
const creature=a=>a?.isOfType?.('creature')??['character','npc'].includes(a?.type);
const weapons=a=>values(a?.items).filter(w=>w.type==='weapon'&&w.system?.equipped?.carryType==='held'&&w.system.equipped.handsHeld>0&&w.system.category!=='unarmed'&&!w.system.traits?.value?.includes('free-hand'));
const needsContext=a=>weapons(a).some(w=>getWeakenedGrasp(a,w.uuid));
const alreadyBound=params=>(params.rollOptions??[]).some(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:disarm-weapon:`));
function contextualDC(dc){
 // A numeric DC otherwise discards the native target and its EphemeralEffects.
 // PF2e still prioritizes the supplied value over the contextual Reflex DC.
 if(typeof dc==='number')return {value:dc,slug:'reflex'};
 if(dc&&typeof dc==='object'&&typeof dc.value==='number'&&!dc.slug)return {...dc,slug:'reflex'};
 return dc;
}

/** Bind ordinary Disarm before StatisticCheck resolves target EphemeralEffects.
 * Only the dedicated Disarm variant prototype is changed; Check and other skill
 * actions remain untouched. Choices run on the actual initiating client.
 */
export function createDisarmContext({game,processor,choose=showNativeChoice}={}){
 let installed=null;
 const actors=params=>{
  if(params.actors)return Array.isArray(params.actors)?params.actors:[params.actors];
  const selected=[...new Set(values(game.user.getActiveTokens?.()).map(t=>t.actor).filter(a=>a&&!a.isOfType?.('loot','party')))];
  return selected.length?selected:game.user.character?[game.user.character]:[];
 };
 const selectedTarget=params=>{
  const explicit=params.target;
  if(explicit){const doc=tokenDocument(explicit);return doc??tokenDocument(explicit.getActiveTokens?.(true,true)?.find(t=>creature(t.actor)))}
  return tokenDocument(values(game.user.targets).find(t=>creature(t.actor)));
 };
 const assertOwner=(actor,userId,leader)=>{
  if(game.user.id!==userId||!actor.testUserPermission?.(game.user,'OWNER')||userId===leader&&leader!==game.users.activeGM?.id)throw Error('本次缴械操作者已失去所有者权限或主GM身份。');
 };
 const assertTokens=(actor,target)=>{
  const origin=tokenDocument(actor.getActiveTokens?.(false,true)?.[0]);
  const actualTarget=tokenDocument(target.actor?.getActiveTokens?.(true,true)?.find(t=>creature(t.actor)));
  if(!origin||origin.actor?.uuid!==actor.uuid||!actualTarget||actualTarget.uuid!==target.uuid||origin.parent?.id!==target.parent?.id)throw Error('原生缴械的来源或目标Token不唯一匹配；请使用对应的当前场景Token。');
  return origin;
 };
 async function use(native,self,params={}){
  const target=selectedTarget(params);
  if(alreadyBound(params)||!target||!needsContext(target.actor))return native.call(self,params);
  const selectedActors=actors(params);if(!selectedActors.length)return native.call(self,params);
  const userId=game.user.id,leader=game.users.activeGM?.id,results=[];
  for(const actor of selectedActors){
   assertOwner(actor,userId,leader);assertTokens(actor,target);
   const disclose=target.actor.testUserPermission?.(game.user,'OWNER');
   const choices=weapons(target.actor).map((w,index)=>({value:w.uuid,label:disclose?(w.name??w.id):`持握武器 ${index+1}`}));
   const selected=await choose({actor,user:game.user,title:'缴械：选择本次要缴下的武器',choices});
   assertOwner(actor,userId,leader);if(selected==null)continue;
   const weapon=weapons(target.actor).find(w=>w.uuid===selected);
   if(!weapon||!choices.some(c=>c.value===selected))throw Error('本次选择的武器已不存在或不再持握。');
   const origin=assertTokens(actor,target);
   const claim=await processor.prepareOrdinaryDisarm({actorUuid:actor.uuid,tokenUuid:origin.uuid,attackerActorUuid:target.actor.uuid,attackerTokenUuid:target.uuid,weaponUuid:weapon.uuid});
   assertOwner(actor,userId,leader);assertTokens(actor,target);
   if(!weapons(target.actor).some(w=>w.uuid===weapon.uuid))throw Error('认领期间目标武器已不再持握。');
   if(!claim?.nonce||!claim.rollOptions?.includes(disarmWeaponOption(weapon))||!claim.rollOptions.includes(`${MODULE_ID}:disarming-block:${claim.nonce}`)||!claim.rollOptions.includes('skip-handling-message'))throw Error('普通缴械的原生结果认领标记无效。');
   const options={...params,actors:[actor],target:target.object??target.actor,rollOptions:[...new Set([...(params.rollOptions??[]),...claim.rollOptions])]};
   if(params.difficultyClass!==undefined)options.difficultyClass=contextualDC(params.difficultyClass);
   const rolled=await native.call(self,options);
   for(const result of rolled??[]){
    if(result.actor?.uuid!==actor.uuid||!result.message||!result.message.id&&params.message?.create!==false)throw Error('原生缴械未返回本次角色的真实检定卡。');
    // An explicitly unpublished native draft can be returned to its caller.
    // Its nonce remains recoverable if that same marked card is later published.
    if(params.message?.create!==false)await processor.settleOrdinaryDisarm({actorUuid:actor.uuid,nonce:claim.nonce,checkId:result.message.id});
    await params.callback?.(result);results.push(result);
   }
  }
  return results;
 }
 function register(){
  if(installed)return unregister;
  const action=game.pf2e.actions.get('disarm'),variant=action?.toActionVariant?.();
  if(!variant||typeof variant.use!=='function')throw Error('缺少原生Disarm动作变体。');
  const prototype=Object.getPrototypeOf(variant),descriptor=Object.getOwnPropertyDescriptor(prototype,'use'),native=variant.use;
  const wrapped=function(params={}){return use(native,this,params)};
  Object.defineProperty(prototype,'use',{configurable:true,writable:true,value:wrapped});
  const legacy=game.pf2e.actions.disarm;
  const legacyWrapper=function(params={}){
   const target=selectedTarget(params);
   if(alreadyBound(params)||!target||!needsContext(target.actor))return legacy.call(this,params);
   const glyph=params.glyph??'A',cost={A:1,B:2,C:3,F:'free',R:'reaction'}[glyph];
   if(cost===undefined)throw Error('无法保留旧缴械宏的动作符号。');
   return action.toActionVariant({cost}).use({...params,statistic:params.statistic??params.skill});
  };
  if(typeof legacy==='function')game.pf2e.actions.disarm=legacyWrapper;
  installed={prototype,descriptor,wrapped,legacy,legacyWrapper};return unregister;
 }
 function unregister(){
  if(!installed)return;const {prototype,descriptor,wrapped,legacy,legacyWrapper}=installed;
  if(prototype.use===wrapped){if(descriptor)Object.defineProperty(prototype,'use',descriptor);else delete prototype.use}
  if(game.pf2e.actions.disarm===legacyWrapper)game.pf2e.actions.disarm=legacy;installed=null;
 }
 return {register,unregister};
}
