const values=c=>Array.from(c?.values?.()??c??[]);
export function wrapSheetHandlers(sheet,handlers,observe,eligible){
 const native=handlers?.['use-action'];if(typeof native!=='function'||!eligible(sheet.actor))return handlers;
 handlers['use-action']=function(event,target,...rest){
  if(target?.closest?.('[data-action-slug]'))return native.call(this,event,target,...rest);
  const item=sheet.actor.items.get(target?.closest?.('[data-item-id]')?.dataset.itemId);
  return item?observe({actor:sheet.actor,item},()=>native.call(this,event,target,...rest)):native.call(this,event,target,...rest);
 };return handlers;
}
/** Wrap the complete resolved variant, including subclass side effects after
 * super.use, plus cached variants and future factory products. Never split a
 * multi-actor native call: acquire sorted independent leases around one call. */
export function installActionEntrances({game,eligible,observe}){
 const patched=new WeakSet(),restores=[];
 const patch=variant=>{
  if(!variant||patched.has(variant)||typeof variant.use!=='function')return variant;patched.add(variant);
  const native=variant.use,descriptor=Object.getOwnPropertyDescriptor(variant,'use');
  const use=function(options={}){
   const selected=options.actors??game.user.getActiveTokens?.().map(t=>t.actor)??[];
   const actors=values(Array.isArray(selected)?selected:[selected]).filter(a=>a&&eligible(a)).sort((a,b)=>a.uuid.localeCompare(b.uuid));
   const invoke=index=>index===actors.length?native.call(this,options):observe({actor:actors[index],entry:'statistic'in variant?'native-check':'native-action'},()=>invoke(index+1));
   return invoke(0);
  };
  Object.defineProperty(variant,'use',{configurable:true,writable:true,value:use});
  restores.push(()=>{if(variant.use===use){if(descriptor)Object.defineProperty(variant,'use',descriptor);else delete variant.use}});return variant;
 };
 for(const action of values(game.pf2e.actions)){
  if(typeof action.toActionVariant!=='function')continue;
  for(const variant of values(action.variants))patch(variant);
  const native=action.toActionVariant,descriptor=Object.getOwnPropertyDescriptor(action,'toActionVariant');
  const factory=function(...args){return patch(native.apply(this,args))};
  Object.defineProperty(action,'toActionVariant',{configurable:true,writable:true,value:factory});
  restores.push(()=>{if(action.toActionVariant===factory){if(descriptor)Object.defineProperty(action,'toActionVariant',descriptor);else delete action.toActionVariant}});
 }
 return ()=>{for(const restore of restores.reverse())restore()};
}
