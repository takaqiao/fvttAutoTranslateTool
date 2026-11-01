import {MODULE_ID} from './rules.mjs';
import {getSourceId} from './native-context.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';

const BRAVERY='Compendium.pf2e.classfeatures.Item.GJKJafDGuX4BeAeN';
const FRIGHTENED='Compendium.pf2e.conditionitems.Item.TBSHQspnbcqxsmjL';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const enabled=game=>game.system?.id==='pf2e'&&BOB2_AUTOMATION_WORLDS.has(game.world?.id);
const hasBravery=actor=>actor?.type==='character'&&values(actor.items).some(item=>item.type==='feat'&&!item.suppressed&&getSourceId(item)===BRAVERY);
const frightened=item=>item?.type==='condition'&&item.system.slug==='frightened'&&hasBravery(item.actor);

function reduceGrantedFrightened(item,onError){
 if(item.type!=='effect'||!hasBravery(item.actor))return;
 const rules=item._source.system.rules;if(!Array.isArray(rules))return;
 let changed=false;
 const next=rules.flatMap(rule=>{
  if(rule.key!=='GrantItem'||rule.inMemoryOnly!==true||rule.uuid!==FRIGHTENED)return [rule];
  const badges=(rule.alterations??[]).filter(alteration=>alteration.property==='badge-value');
  const value=badges.length?badges[0].value:1;
  if(badges.length>1||badges.length&&badges[0].mode!=='override'||!Number.isInteger(value)||value<1){
   onError(Error('英勇：此效果的惊惧授予值不是固定数字，请手动减 1。'));return [rule];
  }
  changed=true;if(value===1)return [];
  const next=structuredClone(rule);next.alterations.find(alteration=>alteration.property==='badge-value').value=value-1;return [next];
 });
 // In-memory grants never create a Condition document. Persist their adjusted
 // value on the new effect once, so later actor preparation cannot reduce again.
 if(changed)item.updateSource({'system.rules':next});
}

export function createBravery({game,onError=error=>globalThis.ui?.notifications?.warn(error.message)}){
 return {register({Hooks,libWrapper}){
  if(!enabled(game))return()=>{};
  const path='CONFIG.Item.documentClass._preCreateOperation';
  libWrapper.register(MODULE_ID,path,async function(wrapped,documents,operation,user){
   if(user.id===game.user.id){
    const removed=[];
    for(const item of documents){
     reduceGrantedFrightened(item,onError);
     const value=item._source.system?.value?.value;
     if(!frightened(item)||!Number.isInteger(value)||value<1)continue;
     if(value===1)removed.push(item);else item.updateSource({'system.value.value':value-1});
    }
    for(const item of removed){
     documents.splice(documents.indexOf(item),1);
     // GrantItem has already linked its parent before this native batch hook.
     const parentId=item.flags.pf2e?.grantedBy?.id;
     const pending=documents.find(doc=>doc.id===parentId),parent=pending??item.actor.items.get(parentId);
     for(const [key,grant]of Object.entries(parent?.flags.pf2e?.itemGrants??{}))if(grant.id===item.id){
      const change={[`flags.pf2e.itemGrants.-=${key}`]:null};
      if(pending)pending.updateSource(change);else await parent.update(change,{render:false});
     }
    }
   }
   return wrapped(documents,operation,user);
  },'WRAPPER');
  const hook=Hooks.on('preUpdateItem',(item,changes,_options,userId)=>{
   if(userId!==game.user.id||!frightened(item))return;
   const flattened=Object.hasOwn(changes,'system.value.value');
   const incoming=flattened?changes['system.value.value']:changes.system?.value?.value,current=item._source.system.value.value;
   if(!Number.isInteger(incoming)||!Number.isInteger(current)||incoming<=current)return;
   const value=Math.max(current,incoming-1);
   if(flattened)changes['system.value.value']=value;else changes.system.value.value=value;
  });
  return()=>{Hooks.off('preUpdateItem',hook);libWrapper.unregister(MODULE_ID,path);};
 }};
}
