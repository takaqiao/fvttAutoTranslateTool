import {isActiveGM} from './native-context.mjs';

// Position and native movement bookkeeping cannot change action/source coverage.
// Keep every unknown or mixed actor-data change on the conservative refresh path.
const MOVEMENT_FIELDS=new Set(['x','y','elevation','rotation','level','movementAction','_movementHistory','_regions','_id','_stats']);

/** Reconcile only the owned reminder as its dependency and actor coverage change. */
export function registerGlimpseConfigurationEvents({game,Hooks,reconcile,onError=console.error}){
 let pending=null,dirty=false,disposed=false;
 const reconcileNow=()=>{
  if(disposed||!isActiveGM(game))return Promise.resolve();
  dirty=true;
  if(!pending)pending=Promise.resolve().then(async()=>{
   while(dirty&&!disposed){dirty=false;if(isActiveGM(game))await reconcile();}
  }).catch(onError).finally(()=>{pending=null});
  return pending;
 };
 const subscriptions=[];
 const on=(name,fn=reconcileNow)=>subscriptions.push([name,Hooks.on(name,fn)]);
 const keys=new Set(['trigger-engine.pf2e-trigger-triggers','pf2e-reaction.builtinReactionsEnabled','core.moduleConfiguration']);
 on('updateSetting',setting=>keys.has(setting.key)?reconcileNow():undefined);
 on('updateToken',(_token,changed)=>{
  const fields=Object.keys(changed??{});
  if(fields.length&&fields.every(key=>MOVEMENT_FIELDS.has(key)))return;
  return reconcileNow();
 });
 for(const name of ['createActor','updateActor','deleteActor','createItem','updateItem','deleteItem','createToken','deleteToken','createScene','deleteScene','combatStart','updateCombat','createCombatant','updateCombatant','deleteCombatant','updateUser','userConnected'])on(name);
 return {reconcileNow,dispose:()=>{disposed=true;for(const [name,id]of subscriptions)Hooks.off(name,id)}};
}
