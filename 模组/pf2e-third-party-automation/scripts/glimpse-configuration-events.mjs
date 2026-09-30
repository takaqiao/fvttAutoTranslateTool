import {isActiveGM} from './native-context.mjs';
import {createDirtyMaintenance,isUnrelatedMaintenanceUpdate,COSMETIC_UPDATE_FIELDS} from './maintenance-events.mjs';

// Only actor rebinding/imports can change a Token's action/source coverage.
const changesTokenActor=changes=>Object.keys(changes??{}).some(key=>['actorId','actorLink','delta'].includes(key)||key.startsWith('delta.'));

/** Reconcile only the owned reminder as its dependency and actor coverage change. */
export function registerGlimpseConfigurationEvents({game,Hooks,reconcile,onError=console.error}){
 const maintenance=createDirtyMaintenance({enabled:()=>isActiveGM(game),run:reconcile,onError}),reconcileNow=maintenance.request;
 const subscriptions=[];
 const on=(name,fn=reconcileNow)=>subscriptions.push([name,Hooks.on(name,fn)]);
 const keys=new Set(['trigger-engine.pf2e-trigger-triggers','pf2e-reaction.builtinReactionsEnabled','core.moduleConfiguration']);
 on('updateSetting',setting=>keys.has(setting.key)?reconcileNow():undefined);
 on('updateToken',(_token,changed)=>{
  if(changesTokenActor(changed))return reconcileNow();
 });
 on('updateActor',(_actor,changes)=>isUnrelatedMaintenanceUpdate(changes,[...COSMETIC_UPDATE_FIELDS,'system.attributes.hp','system.resources.focus'])?undefined:reconcileNow());
 on('updateItem',(_item,changes)=>isUnrelatedMaintenanceUpdate(changes,[...COSMETIC_UPDATE_FIELDS,'system.frequency','system.description'])?undefined:reconcileNow());
 for(const name of ['createActor','deleteActor','createItem','deleteItem','createToken','deleteToken','createScene','deleteScene','combatStart','updateCombat','createCombatant','updateCombatant','deleteCombatant','updateUser','userConnected'])on(name);
 return {reconcileNow,dispose:()=>{maintenance.dispose();for(const [name,id]of subscriptions)Hooks.off(name,id)}};
}
