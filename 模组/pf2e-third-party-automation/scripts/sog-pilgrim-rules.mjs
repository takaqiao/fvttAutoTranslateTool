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
 if(!item)return null;
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
/** Operation option on the effects and actions TPA mirrors, so hooks can tell them from manual edits. */
export const PILGRIM_SYNC='tpaPilgrimSync';
const units={rounds:'轮',minutes:'分钟',hours:'小时',days:'天'};
export const durationText=duration=>units[duration?.unit]&&Number.isFinite(duration.value)?`持续 ${duration.value} ${units[duration.unit]}`:'持续至效果结束';
export const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
/** The reward and operation behind an action that pf2e-item-activations generated from a reward's description. */
export function activationOf(action) {
 const id=action?.type==='action'?action.flags?.['pf2e-item-activations']?.grantedBy?._id:null;
 const reward=typeof id==='string'?action.actor?.items?.get(id):null,key=rewardKey(reward);
 if(!key)return null;
 // The branch describes two activations: Ember Rebirth costs three actions, returning to weapon form is free.
 const operation=key==='branch'?action.system?.actionType?.value==='free'?'restore':action.system?.actions?.value===3?'tree':null:{fan:'release',scarf:'scarf',hairpin:'storm'}[key];
 return operation?{reward,operation}:null;
}
export function tokenCenter(token) {
 if(typeof token.getCenterPoint==='function'){const {x,y}=token.getCenterPoint();return {x,y};}
 const size=token.parent.grid.size;
 return {x:token.x+token.width*size/2,y:token.y+token.height*size/2};
}
/** PF2e's token distanceTo (edge to edge, alternating diagonals) from scene documents, so no canvas is needed. */
export function tokenDistance(source,target) {
 const grid=source.parent.grid,size=grid.size;
 if(!grid.isSquare)return Math.round(grid.measurePath([tokenCenter(source),tokenCenter(target)]).distance);
 const bounds=token=>{
  const width=token.width*size,height=token.height*size;
  if(token.width>=1&&token.height>=1)return {left:token.x,top:token.y,right:token.x+width,bottom:token.y+height};
  const left=Math.floor((token.x+width/2)/size)*size,top=Math.floor((token.y+height/2)/size)*size;
  return {left,top,right:left+Math.max(size,width),bottom:top+Math.max(size,height)};
 };
 const snap=(box,toward)=>{
  const left=(box.left<toward.left?Math.ceil:Math.floor)(box.left/size)*size,top=(box.top<toward.top?Math.ceil:Math.floor)(box.top/size)*size;
  return {left,top,right:left+Math.ceil((box.right-box.left)/size)*size,bottom:top+Math.ceil((box.bottom-box.top)/size)*size};
 };
 const a=bounds(source),b=bounds(target),overlap=b.right>a.left&&b.left<a.right&&b.bottom>a.top&&b.top<a.bottom;
 const near=snap(a,b),far=snap(b,a);
 const dx=overlap?0:Math.max(near.left-far.right,far.left-near.right,0)+size,dy=overlap?0:Math.max(near.top-far.bottom,far.top-near.bottom,0)+size;
 const dz=Math.abs((source.elevation??0)-(target.elevation??0))/grid.distance*size;
 const [low,mid,high]=[dx,dy,dz].map(delta=>Math.ceil(delta/size)).sort((x,y)=>x-y);
 return Math.floor(low*1.75+(mid-low)*1.5+high-mid)*grid.distance;
}
/** Walls are known only to a client that has drawn the token's scene; null means this client cannot judge. */
export function sightBlocked(canvas,source,target) {
 const from=source?.object,to=target?.object;
 if(!canvas||canvas.scene!==source?.parent||typeof from?.checkCollision!=='function'||!to?.center)return null;
 return !!from.checkCollision(to.center,{origin:from.center,type:'sight',mode:'any'});
}
