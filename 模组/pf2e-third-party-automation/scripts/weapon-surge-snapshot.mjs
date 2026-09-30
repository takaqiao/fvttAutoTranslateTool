import {MODULE_ID} from './rules.mjs';
import {getSourceId} from './native-context.mjs';

export const WEAPON_SURGE_SOURCE='Compendium.pf2e.spell-effects.Item.qlz0sJIvqc0FdUdr';
export const WEAPON_SURGE_OPTION=`${MODULE_ID}:weapon-surge:`;
const selection=item=>item?.flags?.system?.rulesSelections?.spellEffectWeaponSurge;
const own=item=>item?.flags?.[MODULE_ID]??{};
const copy=data=>structuredClone(data);
export const weaponSurgeSourceData=item=>copy(item.toObject?.()??item._source??item);
export const isWeaponSurgeFor=(item,weaponId)=>item?.type==='effect'&&!item.isExpired&&!own(item).weaponSurgeFrozen&&getSourceId(item)===WEAPON_SURGE_SOURCE&&selection(item)===weaponId;
const uid=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID().replaceAll('-','').slice(0,16);

export function createWeaponSurgeSnapshot(actor,weapon,effects=[]){
 return {schema:1,actorUuid:actor.uuid,weaponId:weapon.id,weaponUuid:weapon.uuid,nonce:uid(),effects:effects.map(weaponSurgeSourceData)};
}
export function validWeaponSurgeSnapshot(record,actor,weapon){
 return record?.schema===1&&record.actorUuid===actor?.uuid&&typeof record.weaponId==='string'&&!!record.weaponId&&record.weaponUuid===`${actor.uuid}.Item.${record.weaponId}`&&(!weapon||record.weaponId===weapon.id&&record.weaponUuid===weapon.uuid)&&typeof record.nonce==='string'&&!!record.nonce&&Array.isArray(record.effects)&&record.effects.length<=20&&record.effects.every(item=>typeof item?._id==='string'&&isWeaponSurgeFor(item,record.weaponId)&&Array.isArray(item.system?.rules));
}
export function weaponSurgeTransientItems(record){
 return [{_id:uid(),name:'本次攻击：激发武器快照',type:'effect',system:{slug:'tpa-weapon-surge-snapshot',duration:{value:-1,unit:'unlimited'},rules:[]},flags:{[MODULE_ID]:{weaponSurgeSnapshot:copy(record)}}}];
}
function frozenEffect(data,record,index){
 const item=copy(data);item._id=`${record.nonce.replace(/[^a-z0-9]/gi,'').slice(0,12).padEnd(12,'0')}${String(index).padStart(4,'0')}`;
 // The original native rules and ChoiceSet selection prepare on this temporary
 // actor. A distinct slug/source prevents third-party next-use cleanup from
 // confusing the frozen old damage with a live new casting.
 delete item.sourceId;delete item._stats?.compendiumSource;delete item.flags?.core?.sourceId;
 item.system.slug='tpa-weapon-surge-frozen';item.system.duration={value:-1,unit:'unlimited'};
 item.flags={...item.flags,[MODULE_ID]:{...item.flags?.[MODULE_ID],weaponSurgeFrozen:{nonce:record.nonce,sourceId:WEAPON_SURGE_SOURCE,weaponId:record.weaponId}}};
 return item;
}
/** Only transient differences cross the owner socket. Expand the exact original
 * record once, exclude a later live use of the same weapon, retain all unrelated
 * documents and activity modifiers. This never writes the real actor. */
export function prepareWeaponSurgeDamageSnapshotItems(actor,transientItems=[]){
 const records=transientItems.map(item=>own(item).weaponSurgeSnapshot).filter(Boolean);
 if(records.length>1||records.some(record=>!validWeaponSurgeSnapshot(record,actor)))throw Error('本次激发武器快照无效。');
 const record=records[0],base=copy(actor._source.items),extras=copy(transientItems.filter(item=>!own(item).weaponSurgeSnapshot&&!own(item).weaponSurgeFrozen));
 if(!record)return [...base,...extras.filter(item=>!base.some(existing=>existing._id===item._id))];
 const items=base.filter(item=>!isWeaponSurgeFor(item,record.weaponId)&&own(item).weaponSurgeFrozen?.weaponId!==record.weaponId&&!own(item).weaponSurgeSnapshot);
 // Keep the no-rule marker in the temporary source so the owner RPC can send
 // the original record, rather than the full inventory or expanded dice twice.
 const marker=copy(transientItems.find(item=>own(item).weaponSurgeSnapshot));
 return [...items,...extras.filter(item=>!items.some(existing=>existing._id===item._id)),marker,...record.effects.map((effect,index)=>frozenEffect(effect,record,index))];
}
