const values=c=>Array.from(c?.values?.()??c??[]);
/** Supply the native delegated entrance only for explicitly supported sources.
 * PF2e omits Use for activities with no frequency/self-effect. */
export function ensureNativeUseControls(root,actor,supported){
 for(const row of root.querySelectorAll?.('[data-item-id]')??[]){
  const item=actor.items.get(row.dataset.itemId);if(!item||!supported(item)||row.querySelector('[data-action="use-action"],button.use-action'))continue;
  const host=row.querySelector('.item-controls, .button-group');if(!host)continue;
  const button=root.ownerDocument.createElement('button');button.type='button';button.dataset.action='use-action';button.className='metapower-native-use';button.textContent='使用';button.title=item.name??'Use';host.append(button);
 }
}
const patchedHud=new WeakMap();
const digestMethod=async source=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(source))),b=>b.toString(16).padStart(2,'0')).join('');
/** These method hashes were checked against installed HUD 2.55.2. Its helper
 * has the same frequency/self-effect/toMessage contract as Toolbelt 3.56.2. */
export async function patchHudController(controller,{kind,eligible,useToolbelt,digest=digestMethod,expectedHash=kind==='sidebar'?'6298be15832cd498563798053f88d93de41746ac0b3220c9224b341f858d7cae':'f35c7a798a03b2590f9113d30fc8c10a23a54da8274138af4ff8f1858211d1dd'}){
 const prototype=Object.getPrototypeOf(controller);if(patchedHud.has(prototype))return patchedHud.get(prototype);
 const descriptor=Object.getOwnPropertyDescriptor(prototype,'use');
 if(!descriptor?.configurable||typeof descriptor.value!=='function'||await digest(Function.prototype.toString.call(descriptor.value))!==expectedHash)throw Error('HUD use entry differs from the reviewed version; metapower use requires the original actor sheet.');
 if(patchedHud.has(prototype))return patchedHud.get(prototype);
 const native=descriptor.value;
 const use=function(event,...args){
  if(!this.item||!eligible(this.item.actor)||this.isExploration)return native.call(this,event,...args);
  if(this.virtualData)throw Error('Virtual HUD actions need manual metapower resolution.');
  return useToolbelt(event,this.item);
 };
 Object.defineProperty(prototype,'use',{...descriptor,value:use});patchedHud.set(prototype,use);return use;
}
export function createToolbeltEntrance({native,eligible,observe}){
 return (event,item,virtual)=>{
  if(!eligible(item?.actor))return native(event,item,virtual);
  if(virtual||item.flags?.['pf2e-toolbelt']?.actionable?.linked)throw Error('Virtual/linked-macro action needs manual metapower resolution; use the owned original item.');
  return observe({actor:item.actor,item},()=>native(event,item,virtual));
 };
}
/** Deprecated callback-only aliases have no awaitable cancellation receipt.
 * Refuse that unsupported entrance while a metapower is live rather than
 * silently carrying it across an unobserved action. Modern collection variants
 * remain available and the original alias is unchanged when no state is live. */
export function installLegacyActionBoundary({game,blocked,onError}){
 for(const key of Object.keys(game.pf2e.actions)){
  const descriptor=Object.getOwnPropertyDescriptor(game.pf2e.actions,key),native=descriptor?.value;
  if(typeof native!=='function'||!descriptor.configurable)continue;
  Object.defineProperty(game.pf2e.actions,key,{...descriptor,value:function(options={},...args){
   const selected=options.actors??game.user.getActiveTokens?.().map(t=>t.actor)??[];
   if(values(Array.isArray(selected)?selected:[selected]).some(blocked)){
    const error=Error('此 legacy 动作宏无法验证执行／取消；请用原生动作列表入口，或先在超威能原卡明确清除待用状态。');onError(error);throw error;
   }
   return native.call(this,options,...args);
  }});
 }
}
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
export function installActionEntrances({game,eligible,observe,continuation=()=>false}){
 const patched=new WeakSet(),restores=[];
 const patch=variant=>{
  if(!variant||patched.has(variant)||typeof variant.use!=='function')return variant;patched.add(variant);
  const native=variant.use,descriptor=Object.getOwnPropertyDescriptor(variant,'use');
  const use=function(options={}){
   const selected=options.actors??game.user.getActiveTokens?.().map(t=>t.actor)??[];
   const actors=values(Array.isArray(selected)?selected:[selected]).filter(a=>a&&eligible(a)).sort((a,b)=>a.uuid.localeCompare(b.uuid));
   const parent=options['pf2e-third-party-automation']?.metapowerContinuation;
   if(parent){if(!actors.every(actor=>continuation(actor,parent)))throw Error('Original action continuation binding is invalid.');return native.call(this,options)}
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
