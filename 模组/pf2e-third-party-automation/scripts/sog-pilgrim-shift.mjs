import {MODULE_ID as M} from './rules.mjs';
import {PILGRIM_FLAG,values,rewardKey,usableReward} from './sog-pilgrim-rules.mjs';

const PACK='pf2e.equipment-srd';
const formPattern=/^Compendium\.pf2e\.equipment-srd\.Item\.[A-Za-z0-9]+$/;
const magicalTraits=new Set(['magical','arcane','divine','occult','primal','artifact','cursed','tech','analog']);
const indexFields=['slug','baseItem','category','group','level.value','damage','traits.value','usage.value','range','runes','material','specific'].map(key=>`system.${key}`);
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function baseType(data) {
 const s=data?.system;
 if(typeof s?.slug!=='string'||!s.slug)return null;
 return s.baseItem===s.slug||s.baseItem==null&&Object.hasOwn(globalThis.CONFIG?.PF2E?.baseWeaponTypes??{},s.slug)?s.slug:null;
}
function baseWeapon(data) {
 const s=data?.system,runes=s?.runes;
 return data?.type==='weapon'&&['simple','martial','advanced'].includes(s?.category)&&baseType(data)!==null&&s.level?.value===0&&
  s.usage?.value==='held-in-one-hand'&&!s.range&&!s.specific&&!s.material?.type&&!s.material?.grade&&
  !runes?.potency&&!runes?.striking&&!runes?.property?.length&&Array.isArray(s.traits?.value)&&!s.traits.value.some(trait=>magicalTraits.has(trait))&&
  s.damage?.dice===1&&['d4','d6','d8','d10','d12'].includes(s.damage.die)&&typeof s.damage.damageType==='string';
}
// Foundry merges object updates; remove fields that belonged only to the old form.
function replaceObject(patch,path,before,after) {
 patch[path]=structuredClone(after);
 function removed(previous,next,at) {
  for(const [key,value]of Object.entries(previous??{})){
   if(!Object.hasOwn(next,key))patch[`${at}.-=${key}`]=null;
   else if(value&&typeof value==='object'&&!Array.isArray(value)&&next[key]&&typeof next[key]==='object'&&!Array.isArray(next[key]))removed(value,next[key],`${at}.${key}`);
  }
 }
 removed(before,after,path);
}

/** The reward provider owns permissions, serialization, and the action receipt. */
export function createPilgrimShift({game,fromUuid=globalThis.fromUuid}={}) {
 let choices;
 const currentActor=actor=>actor?.isToken?
  actor.token?.actor===actor&&game.scenes?.get(actor.token.parent?.id)===actor.token.parent&&actor.token.parent?.tokens?.get(actor.token.id)===actor.token:
  !!actor&&game.actors?.get(actor.id)===actor;
 function source(item,actor=item?.actor) {
  if(rewardKey(item)!=='branch'||item.actor!==actor||!currentActor(actor)||actor.items?.get(item.id)!==item||!usableReward(item))throw Error('请先持握大杉枝。');
  if(values(actor.items).some(effect=>effect.type==='effect'&&effect.flags?.[M]?.[PILGRIM_FLAG]?.kind==='tree'&&effect.flags[M][PILGRIM_FLAG].source===item.uuid))throw Error('请先将大杉枝恢复为武器。');
 }
 async function list() {
  if(!choices){
   const pack=game.packs?.get(PACK);
   if(!pack)throw Error('无法读取武器目录，请联系主持人。');
   choices=pack.getIndex({fields:indexFields}).then(index=>values(index).filter(baseWeapon).map(entry=>({value:`Compendium.${PACK}.Item.${entry._id}`,label:entry.name})).filter(choice=>formPattern.test(choice.value)).sort((a,b)=>a.label.localeCompare(b.label)));
   choices.catch(()=>{choices=undefined});
  }
  return choices;
 }
 async function select(item) {
  const actor=item?.actor;source(item,actor);
  const options=await list();source(item,actor);
  if(!options.length)throw Error('没有可变成的单手近战武器。');
  const selected=await globalThis.foundry.applications.api.DialogV2.wait({
   window:{title:'大杉枝变型'},content:`<div class="form-group"><label for="sog-pilgrim-form">武器形态</label><select id="sog-pilgrim-form" name="formUuid">${options.map(option=>`<option value="${escapeHTML(option.value)}">${escapeHTML(option.label)}</option>`).join('')}</select></div>`,
   buttons:[{action:'shift',label:'变型',default:true,callback:(_event,button)=>button.form.elements.formUuid.value},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false,
  });
  if(selected===null||selected===undefined||selected===false)return null;
  source(item,actor);
  if(!options.some(option=>option.value===selected))throw Error('请选择目录中的单手近战武器。');
  return selected;
 }
 async function apply({item,formUuid}) {
  const actor=item?.actor;source(item,actor);
  if(typeof formUuid!=='string'||!formPattern.test(formUuid))throw Error('请选择目录中的单手近战武器。');
  const target=await fromUuid(formUuid);source(item,actor);
  const data=target?.toObject?.();
  if(target?.uuid!==formUuid||!baseWeapon(data)||target.isMelee!==true||target.hands!=='1'||target.category==='unarmed')throw Error('只能变成普通的单手近战武器。');
  const before=item.toObject().system,s=data.system,base=baseType(data),patch={
   'system.baseItem':base,'system.category':s.category,'system.group':s.group??null,
   'system.traits.value':[...new Set([...s.traits.value,...(item.system.traits.value.includes('magical')?['magical']:[])])],
   'system.usage.value':s.usage.value,'system.usage.canBeAmmo':s.usage.canBeAmmo??false,
   'system.range':s.range??null,'system.maxRange':s.maxRange??null,'system.reload.value':s.reload?.value??null,
   'system.meleeUsage':s.meleeUsage??null,'system.ammo':s.ammo??null,'system.expend':s.expend??null,'system.selectedAmmoId':null,'system.attribute':s.attribute??null,
   'system.bonus.value':s.bonus?.value??0,'system.splashDamage.value':s.splashDamage?.value??0,
  };
  // Prepared damage already includes striking; the new source must keep its one die.
  replaceObject(patch,'system.damage',before.damage,s.damage);
  replaceObject(patch,'system.traits.toggles',before.traits?.toggles,s.traits.toggles??{});
  replaceObject(patch,'system.traits.config',before.traits?.config,s.traits.config??{});
  await item.update(patch);source(item,actor);
  const saved=item.toObject().system;
  if(saved.baseItem!==base||saved.category!==s.category||saved.group!==(s.group??null)||saved.damage.dice!==s.damage.dice||saved.damage.die!==s.damage.die||saved.damage.damageType!==s.damage.damageType)throw Error('大杉枝的形态未能保存，请联系主持人核对。');
  return item;
 }
 return {select,apply};
}
