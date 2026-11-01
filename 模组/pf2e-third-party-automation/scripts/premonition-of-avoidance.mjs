import {getSourceId} from './native-context.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';

const SOURCE='Compendium.pf2e.feats-srd.Item.KYl1rWFOHe0e6VqJ',OPTION='premonition-of-avoidance';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const toggle=item=>item.system.rules?.find(rule=>rule.key==='RollOption'&&rule.option===OPTION&&rule.domain==='saving-throw'&&rule.toggleable===true);

export function createPremonitionOfAvoidance({game,choose,onError=()=>{}}={}){
 const pending=new Set();
 const managed=actor=>BOB2_AUTOMATION_WORLDS.has(game?.world?.id)&&game.system?.id==='pf2e'&&actor?.type==='character'&&actor.testUserPermission?.(game.user,'OWNER')===true;
 const matches=item=>item?.type==='feat'&&getSourceId(item)===SOURCE;
 async function interceptCheck(native,check,context={},event=null,callback){
  const candidate=context.actor??(context.origin?.self?context.origin.actor:context.target?.self?context.target.actor:null),worldActor=game.actors?.get(candidate?.id);
  const actor=worldActor?.uuid===candidate?.uuid?worldActor:candidate,options=new Set(context.options??[]);
  if(!managed(actor)||context.isReroll||context.type!=='saving-throw'||!options.has('origin:type:hazard')||!options.has(OPTION))return native(check,context,event,callback);
  const item=values(actor.items).find(matches);if(!item)return native(check,context,event,callback);
  // A second prepared check can still carry the saved option while the first
  // confirmation is open, or after its original actor has been disarmed.
  if(pending.has(actor.uuid)||toggle(item)?.value!==true){
   options.delete(OPTION);context.options=options;
   return native(check,context,event,callback);
  }
  const current=()=>managed(actor)&&actor.items.get(item.id)===item&&matches(item);
  pending.add(actor.uuid);
  try{
   const answer=await choose({actor,user:game.user,title:'预感回避：本次危害豁免',choices:[
    {value:'use',label:'使用预感回避反应（本次危害豁免 +2 环境加值）'},
    {value:'decline',label:'正常掷骰，不使用本次反应'},
   ]});
   if(answer!=='use'||!current()||toggle(item)?.value!==true)options.delete(OPTION);
   // Native Check.roll retests the prepared modifiers with these options and
   // applies normal circumstance stacking before displaying its roll dialog.
   context.options=options;
   return await native(check,context,event,callback);
  }finally{
   try{
    if(current()&&toggle(item)?.value===true)await actor.toggleRollOption('saving-throw',OPTION,item.id,false);
   }catch(error){onError(error)}
   finally{pending.delete(actor.uuid)}
  }
 }
 return {interceptCheck};
}
