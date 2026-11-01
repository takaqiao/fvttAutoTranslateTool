import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId,isActiveGM,publicTargetName} from './native-context.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';
import {registerAvDamageSnapshot} from './av-damage-snapshot.mjs';
import {SerialActions} from './runtime.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';

const HEAL='Compendium.pf2e.spells-srd.Item.rfZpqmj0AIIdkVIs',FEAT='Compendium.pf2e.feats-srd.Item.yExxOkHN1PN37hUa';
const TAG=`${ID}:communal-healing:cast:`,PATH=`flags.${ID}.communalHealing`;
const values=c=>Array.from(c?.values?.()??c??[]),author=m=>m?.author?.id??m?.user?.id??m?.user;
const feature=a=>values(a?.items).find(i=>i.type==='feat'&&getSourceId(i)===FEAT);
const living=a=>a?.modeOfBeing==='living'&&a.isDead!==true&&a.system?.attributes?.hp?.negativeHealing!==true;
const single=i=>['1','2'].includes(String(i?.system?.time?.value))&&!i.system.area;

export function createCommunalHealing({game,fromUuid=globalThis.fromUuid,choose,onError=error=>console.error(ID,error),castEvents=getNativeCastEvents({game,fromUuid})}={}){
 const queue=new SerialActions(),rolling=new WeakMap();let installed=false;
 const enabled=()=>installed&&BOB2_AUTOMATION_WORLDS.has(game.world?.id);
 const matches=item=>enabled()&&item?.type==='spell'&&getSourceId(item)===HEAL&&!!feature(item.actor);
 function sourceContext(message){
  if(!enabled()||!message||game.messages.get(message.id)!==message||message.rolls?.length)return null;
  const item=message.item,actor=item?.actor,proof=message.flags?.[ID]?.nativeCast,input=message.flags?.[ID]?.nativeCastInput,origin=message.flags?.pf2e?.origin;
  if(!matches(item)||!proof?.id||proof.actorUuid!==actor.uuid||proof.itemUuid!==item.uuid||proof.userId!==author(message)||origin?.uuid!==item.uuid)return null;
  const user=game.users.get(proof.userId),receipt=actor.flags?.[ID]?.nativeCasts?.find(r=>r.id===proof.id);
  if(!user||!actor.testUserPermission(user,'OWNER')||!receipt||!['paid','used'].includes(receipt.state)||receipt.userId!==user.id||receipt.itemUuid!==item.uuid||receipt.messageId&&receipt.messageId!==message.id)return null;
  const rank=input?.rank;
  if(!Number.isInteger(rank)||rank<1||rank>10||receipt.rank!==rank||origin.castRank!==rank)return null;
  return {message,item,actor,user,rank,castId:proof.id};
 }
 function rollSource(item,event){
  if(!matches(item))return null;
  const node=event?.target?.closest?.('[data-message-id]')??event?.currentTarget?.closest?.('[data-message-id]');
  const context=sourceContext(game.messages.get(node?.dataset?.messageId));
  return context&&item.uuid===context.item.uuid&&item.rank===context.rank?context:null;
 }
 function healingSource(message){
  if(!enabled()||game.messages.get(message?.id)!==message)return null;
  const pf=message.flags?.pf2e,context=pf?.context,applied=pf?.appliedDamage;
  if(context?.type!=='damage-taken'||!context.domains?.includes('healing-received')||applied?.isReverted||applied&&applied.isHealing!==true)return null;
  const tags=context.options?.filter(o=>typeof o==='string'&&o.startsWith(TAG))??[];
  if(tags.length!==1)return null;
  const source=sourceContext(game.messages.get(tags[0].slice(TAG.length)));
  if(!source||!single(message.item)||getSourceId(message.item)!==HEAL||pf.origin?.uuid!==source.item.uuid||pf.origin.castRank!==source.rank)return null;
  const scene=game.scenes.get(message.speaker?.scene),token=scene?.tokens.get(message.speaker?.token),primary=token?.actor;
  if(!primary||message.speaker.actor!==primary.id||!living(primary)||applied&&applied.uuid!==primary.uuid)return null;
  return {...source,result:message,scene,primary};
 }
 function candidates(context){
  const seen=new Set([context.primary.uuid]),result=[];
  for(const token of values(context.scene.tokens)){
   const actor=token.actor;
   if(!living(actor)||seen.has(actor.uuid)||token.hidden&&!context.user.isGM&&!actor.testUserPermission(context.user,'OWNER'))continue;
   seen.add(actor.uuid);result.push(token);
  }
  return result;
 }
 async function settle(message){
  if(!isActiveGM(game))return;
  const initial=healingSource(message);if(!initial)return;
  return queue.run(`${initial.actor.uuid}:${initial.castId}`,async()=>{
   const context=healingSource(message);if(!context||!isActiveGM(game)||context.message.flags?.[ID]?.communalHealing)return;
   const gmId=game.user.id;
   // A nativeCast receipt is required before ensurePaid: a display card must
   // never enter the helper's separate chat-payment path.
   await castEvents.ensurePaid({actor:context.actor,item:context.item,message:context.message,user:context.user});
   const state={castId:context.castId,resultMessageId:message.id,primaryUuid:context.primary.uuid,rank:context.rank,status:'choosing'};
   await context.message.update({[PATH]:state});
   try{
    const targets=candidates(context),choices=targets.map(token=>({value:token.uuid,label:`${publicTargetName(token,{game,user:context.user})}：恢复 ${context.rank} HP`}));
    choices.push({value:'decline',label:'不使用共享治疗'});
    const selected=await choose({actor:context.actor,user:context.user,title:'共享治疗：选择另一名范围内的自愿活体（选择即确认）',choices});
    if(!enabled()||!isActiveGM(game)||game.user.id!==gmId)throw Error('共享治疗结算期间主GM或功能状态已改变。');
    if(!selected||selected==='decline'){await context.message.update({[PATH]:{...state,status:'declined'}});return;}
    const target=targets.find(t=>t.uuid===selected),current=healingSource(message);
    if(!current||!target||current.scene.tokens.get(target.id)!==target||!candidates(current).includes(target))throw Error('共享治疗所选受益人无效或已改变。');
    const application={...state,targetUuid:target.uuid,status:'applying'};
    await context.message.update({[PATH]:application});
    // final keeps this fixed-rank benefit from rolling receiver bonus dice.
    await target.actor.applyDamage({damage:-context.rank,token:target,item:feature(context.actor),final:true,skipIWR:true,rollOptions:new Set(['item:slug:communal-healing','item:trait:healing','item:trait:vitality'])});
    await context.message.update({[PATH]:{...application,status:'applied'}});
   }catch(error){
    const current=context.message.flags?.[ID]?.communalHealing;
    await context.message.update({[PATH]:{...current,status:current?.status==='applying'?'uncertain':'cancelled'}});
    throw error;
   }
  });
 }
 function register({Hooks=globalThis.Hooks,libWrapper=globalThis.libWrapper}={}){
  if(installed||!BOB2_AUTOMATION_WORLDS.has(game.world?.id))return()=>{};installed=true;
  castEvents.addMatcher(matches);
  const damagePath='CONFIG.PF2E.Item.documentClasses.spell.prototype.getDamage';
  const unregisterRoll=registerAvDamageSnapshot({game,libWrapper,interceptRollDamage:async function(wrapped,event,...args){
   const context=rollSource(this,event);if(!context)return wrapped(event,...args);
   rolling.set(this,context);
   try{return await wrapped(event,...args);}finally{if(rolling.get(this)===context)rolling.delete(this);}
  }});
  libWrapper.register(ID,damagePath,async function(wrapped,...args){
   const context=rolling.get(this),result=await wrapped(...args);
   if(context&&single(this)&&sourceContext(context.message)&&result?.context?.options instanceof Set)result.context.options.add(TAG+context.message.id);
   return result;
  },'WRAPPER');
  const hook=Hooks.on('createChatMessage',message=>settle(message).catch(onError));
  return()=>{installed=false;Hooks.off('createChatMessage',hook);unregisterRoll();libWrapper.unregister(ID,damagePath);};
 }
 return {register};
}
