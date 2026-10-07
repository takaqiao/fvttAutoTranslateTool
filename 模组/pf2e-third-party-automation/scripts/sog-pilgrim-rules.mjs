export const PILGRIM_FLAG = 'sogPilgrim';
export class PilgrimError extends Error {}
export const REWARDS = Object.freeze({
 fan: {id:'c575e25b61e2bb47',key:'spirit-fan',type:'weapon'},
 branch: {id:'8c07e2ce6002e44d',key:'branch-of-the-great-sugi',type:'weapon'},
 scarf: {id:'a8f3970a174e8ea2',key:'ghost-scarf',type:'equipment'},
 hairpin: {id:'d534c054ee1c5176',key:'hairpin-of-blooming-flowers',type:'equipment'},
});
export const EFFECTS = Object.freeze({leaves:'e3f4b1e0c69e0cba',light:'53ad48c5ad472dcf',tree:'1ade2944acadc195',ghost:'ffebdbe91ceb569e',astral:'6d00b8dddd4d6a71',ward:'2d43538383b1d58b',storm:'9570410bfd571f7d'});
export const values = collection => Array.from(collection?.values?.() ?? collection ?? []);
const pending = new Map();
export function serializeReward(uuid, operation) {
 const next=(pending.get(uuid)??Promise.resolve()).catch(()=>{}).then(operation);
 pending.set(uuid,next);
 return next.finally(()=>{if(pending.get(uuid)===next)pending.delete(uuid);});
}
export function rewardKey(item) {
 const native=item?.flags?.world?.sogWontonNativeAutomation?.key;
 const source=item?.flags?.world?.sogB2Ch2Resources;
 return Object.entries(REWARDS).find(([,r])=>item.type===r.type && native===r.key && source?.originalSlug===r.key && typeof source.fileSha256==='string' && source.fileSha256.length===64)?.[0] ?? null;
}
export function usableReward(item) {
 const key=rewardKey(item),e=item?.system?.equipped;
 if(!key||!item.actor||item.isEquipped===false)return false;
 if(item.type==='weapon')return e?.carryType==='held'&&e.handsHeld>=1;
 return e?.carryType==='worn'&&e.invested===true&&item.isInvested!==false;
}
export function scarfBranch(weapon) {
 const runes=weapon?.system?.runes?.property??[];
 return runes.includes('astral')?'ward':runes.includes('ghostTouch')?'astral':'ghost';
}
