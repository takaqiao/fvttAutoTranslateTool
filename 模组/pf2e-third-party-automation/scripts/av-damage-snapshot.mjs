import {MODULE_ID} from './rules.mjs';
import {WAVE_SOURCE,isAvWaveSpell} from './av-wave-repair.mjs';
const clone=value=>globalThis.foundry?.utils?.deepClone?.(value)??structuredClone(value);
const source=item=>item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId??null;
const ampOption=option=>option==='amp-spell'||option==='alternate-amp'||option.startsWith('alternate-amp:');

/** PF2e rebuilds RuleElements from _source on every contextual clone. Restore
 * only the native amp toggles in an ephemeral actor, never the live actor. */
export function buildAvDamageCloneChanges(actor,cast,wave){
 const items=clone(actor._source.items),all={...actor.rollOptions?.all};
 if(cast){
 for(const option of Object.keys(all))if(ampOption(option))all[option]=false;
 all['amp-spell']=cast.amped;all['alternate-amp']=!!cast.alternateAmp;
 if(cast.alternateAmp)all['alternate-amp:'+cast.alternateAmp]=true;
 for(const item of items)for(const rule of item.system?.rules??[]){
  if(rule.key!=='RollOption'||(rule.domain??'all')!=='all')continue;
  if(rule.option==='amp-spell')rule.value=cast.amped;
  if(rule.option==='alternate-amp'){rule.value=!!cast.alternateAmp;if(cast.alternateAmp)rule.selection=cast.alternateAmp;}
 }
 }
 if(wave){
  for(const option of Object.keys(all))if(option==='conservation-of-energy'||option.startsWith('conservation-of-energy:'))all[option]=false;
  all['conservation-of-energy']=true;all['conservation-of-energy:'+wave.energy]=true;
  for(const item of items)if(source(item)===WAVE_SOURCE){
   for(const rule of item.system?.rules??[])if(rule.key==='RollOption'&&rule.option==='conservation-of-energy'){rule.selection=wave.energy;rule.value=true;}
   item.flags??={};item.flags.pf2e??={};item.flags.pf2e.rulesSelections??={};item.flags.pf2e.rulesSelections.conservationOfEnergy=wave.energy;
  }
  for(const item of items)if(item._id===wave.itemUuid?.split('.').at(-1)&&source(item)===wave.sourceId&&item.type==='spell'){
   const types=new Set(Object.values(item.system.damage??{}).map(d=>d.type));
   item.system.traits.value=[...item.system.traits.value.filter(t=>!types.has(t)&&!['fire','cold'].includes(t)),wave.energy];
  }
 }
 return {items,flags:{pf2e:{rollOptions:{all}}}};
}

export function registerAvDamageSnapshot({game,libWrapper}={}){
 if(!libWrapper)return ()=>{};
 const contextual=new WeakSet(),path='CONFIG.PF2E.Item.documentClasses.spell.prototype.rollDamage';
 libWrapper.register(MODULE_ID,path,async function(wrapped,event,mapIncreases){
  if(contextual.has(this.actor))return wrapped(event,mapIncreases);
  const messageId=event?.target?.closest?.('[data-message-id]')?.dataset.messageId;
  const message=game.messages.get(messageId),cast=message?.flags?.[MODULE_ID]?.avCast,origin=message?.flags?.pf2e?.origin;
  const isWave=isAvWaveSpell(this);
  if((cast?.psiCantrip!==true&&!isWave)||cast?.itemUuid!==this.uuid||cast.sourceId!==source(this)||origin?.uuid!==this.uuid||origin.actor!==this.actor?.uuid||message.speaker?.actor!==this.actor?.id)return wrapped(event,mapIncreases);
  if(typeof cast.amped!=='boolean'||cast.alternateAmp==='unknown')throw Error('原施法卡缺少确定的增幅选择，无法重建此次伤害。');
  let wave;
  if(isWave){
   // The card can render before the GM finishes its exact usage transaction.
   // Wait only for this card; never infer energy from another cast or live state.
   for(let n=0;n<300;n++){
    const flags=game.messages.get(messageId)?.flags?.[MODULE_ID];wave=flags?.avWave;
    if(wave||flags?.usage?.status==='error')break;
    await new Promise(resolve=>setTimeout(resolve,100));
   }
   if(wave?.actorUuid!==this.actor.uuid||wave?.itemUuid!==this.uuid||wave?.sourceId!==source(this)||!['fire','cold'].includes(wave?.energy))throw Error('此次震荡波施法尚未成功结算，无法确定原卡伤害能量。');
  }
  const actor=this.actor.clone(buildAvDamageCloneChanges(this.actor,cast.psiCantrip?cast:null,wave),{keepId:true});contextual.add(actor);
  let spell=actor.items.get(this.id);if(!spell)throw Error('无法从原生临时角色恢复此心能戏法。');
  spell=spell.loadVariant?.({castRank:origin.castRank,overlayIds:origin.variant?.overlays??[]})??spell;
  // Call the public method on the temporary document so all other wrappers and
  // the native DamageContext still execute. The WeakSet stops our recursion.
  return spell.rollDamage(event,mapIncreases);
 },'MIXED');
 return ()=>libWrapper.unregister(MODULE_ID,path);
}
