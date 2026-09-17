import {isActiveGM} from './native-context.mjs';
import {MODULE_ID} from './rules.mjs';
import {GLIMPSE_SOURCES} from './glimpse-source.mjs';
import {validGlimpseTemplate} from './glimpse-native.mjs';
import {createGlimpseExpiry} from './glimpse-expiry.mjs';
export const GLIMPSE_TRIGGER_ID='TPAGlimpseFlow01',GLIMPSE_EVENT='tpa-glimpse-resist-event';
export const glimpseWorld=game=>['ujx5r8oipw7ercdr','team-automation-qa2'].includes(game.world?.id);
const ENGINE='trigger-engine',SETTING='pf2e-trigger-triggers',HASH='4f62f4c45a1a36d19a39f6b0da17ef5c311ced57af92265e38fd0930f3ae53de';
const random=()=>globalThis.foundry?.utils?.randomID?.(24)??crypto.randomUUID();
const slugFor=nonce=>`tpa-glimpse-${nonce.toLowerCase()}`;
/** Rule facts only: the installed engine remains the sole condition writer. */
export function glimpseGraph(){return {id:GLIMPSE_TRIGGER_ID,name:'救赎瞥视：已验证的抗拒后续',priority:0,nodes:[
 {id:'TPAGlimpseEvnt01',type:GLIMPSE_EVENT,position:{x:0,y:0},outs:{out:{connection:'TPAGlimpseCond01:ins:in'}}},
 {id:'TPAGlimpseCond01',type:'create-condition',position:{x:300,y:0},state:'timed',inputs:{condition:{value:'enfeebled'},value:{value:2},duration:{value:1},unit:{value:'rounds'},expiry:{value:'turn-end'},name:{value:'救赎瞥视：衰弱 2'},target:{connection:'TPAGlimpseEvnt01:outputs:target'},origin:{connection:'TPAGlimpseEvnt01:outputs:target'},slug:{connection:'TPAGlimpseEvnt01:outputs:slug'}}},
 ]}}
async function verifyInstalledEngine(){const url=globalThis.foundry?.utils?.getRoute?.('modules/trigger-engine/scripts/main.js')??'/modules/trigger-engine/scripts/main.js',response=await fetch(url,{cache:'no-store'});if(!response.ok)return false;const bytes=await response.arrayBuffer(),actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');return actual===HASH}
export function createGlimpseCompat({game,fromUuid=globalThis.fromUuid,api=()=>globalThis.triggerEngine,verifyEngine=verifyInstalledEngine,query=data=>globalThis.CONFIG?.queries?.['trigger-engine.user-query']?.(data)}={}){
 const scopes=new Map();let registered=false,engineReady=false,initialized=false,template,probed=false,wake,lifecycle,hooksApi;
 const readyPromise=new Promise(resolve=>wake=resolve);
 const setting=()=>game.settings.get(ENGINE,SETTING)??{};
 const versions=()=>game.system?.id==='pf2e'&&game.system.version==='8.5.1'&&game.modules.get(ENGINE)?.active&&game.modules.get(ENGINE).version==='1.35.0'&&game.modules.get('pf2e-trigger-trove')?.active&&game.modules.get('pf2e-trigger-trove').version==='2.3.5';
 const safeSetting=()=>{const s=setting();if(s.sources?.some(g=>g.id===GLIMPSE_TRIGGER_ID))throw Error('救赎瞥视模块图已被世界配置覆盖。');return s};
 const ready=()=>glimpseWorld(game)&&initialized&&versions()&&setting().enabled?.includes(GLIMPSE_TRIGGER_ID)&&!setting().disabled?.includes(GLIMPSE_TRIGGER_ID)&&!setting().sources?.some(g=>g.id===GLIMPSE_TRIGGER_ID)&&(!isActiveGM(game)||probed);
 function register({Hooks}){
  if(registered)return;registered=true;hooksApi=Hooks;
  Hooks.once('triggerEngine.registerNodes',registerNodes=>{
   const Base=api()?.TriggerNode;if(!Base)return;
   class GlimpseResistEvent extends Base{
    static get type(){return GLIMPSE_EVENT}static get isEvent(){return true}
    static get defineOutputs(){return [{key:'target',type:'target'},{key:'slug',type:'text'}]}
    get title(){return '救赎瞥视：已验证后续'}
    async _execute(args){
     const scope=scopes.get(args?.key);if(!scope||scope.used||!isActiveGM(game))return true;scope.used=true;
     if(scope.probe){scope.done=true;return true}
     if(!await scope.authorize())return true;
     this.setOutputValue('target',{actor:scope.enemy.actor,token:scope.enemy});this.setOutputValue('slug',scope.slug);
     await this.executeNext('out');scope.done=true;return true;
    }
   }
   registerNodes(ENGINE,'pf2e-trigger',[GlimpseResistEvent]);
  });
  Hooks.once('triggerEngine.registerTriggers',registerTriggers=>registerTriggers(ENGINE,'pf2e-trigger',[glimpseGraph()]));
  Hooks.once('triggerEngine.ready',()=>{engineReady=true;wake()});
 }
 async function dispatch(scope){const key=random();scopes.set(key,scope);try{await query({_type:'execute-trigger',triggerPath:`${ENGINE}:pf2e-trigger:${GLIMPSE_TRIGGER_ID}`,eventName:GLIMPSE_EVENT,args:{key},userId:game.user.id});if(!scope.done)throw Error('救赎瞥视原生后续未得到可验证完成。')}finally{scopes.delete(key)}}
 async function initialize({game:runtimeGame}={}){
  // Foundry replaces the bootstrap `game` after loading module scripts. Node
  // registration must happen early; all world authorization uses the ready Game.
  if(runtimeGame)game=runtimeGame;
  if(!glimpseWorld(game))return false;
  if(!registered)throw Error('救赎瞥视节点须在 Trigger Engine init 前注册。');
  // Cleanup of previously verified owned effects must remain available even
  // when a dependency upgrade prevents creating any new Glimpse automation.
  lifecycle??=createGlimpseExpiry({game});lifecycle.register({Hooks:hooksApi});await lifecycle.reconcile();
  if(!versions()||!await verifyEngine())throw Error('救赎瞥视依赖版本或引擎源码不匹配。');
  if(!engineReady){let timer;try{await Promise.race([readyPromise,new Promise((_,reject)=>timer=setTimeout(()=>reject(Error('Trigger Engine 尚未就绪。')),10000))])}finally{clearTimeout(timer)}}
  const doc=await fromUuid(GLIMPSE_SOURCES.resistance);template=doc?.toObject?.();if(!validGlimpseTemplate(template))throw Error('救赎瞥视原生抗力模板已改变。');
  const s=safeSetting();if(s.disabled?.includes(GLIMPSE_TRIGGER_ID))throw Error('救赎瞥视后续图已明确禁用。');
  if(isActiveGM(game)){
   if(!s.enabled?.includes(GLIMPSE_TRIGGER_ID)){
    const enabled=[...s.enabled??[],GLIMPSE_TRIGGER_ID],history=game.settings.get(MODULE_ID,'configurationBackups')??[];
    await game.settings.set(MODULE_ID,'configurationBackups',[...history,{version:game.modules.get(MODULE_ID)?.version??'glimpse',time:new Date().toISOString(),setting:`${ENGINE}.${SETTING}`,changes:[{path:'enabled',before:structuredClone(s.enabled??[]),after:enabled,reason:'仅启用本模块来源受控的救赎瞥视后续图。'}]}]);
    if(!isActiveGM(game)||JSON.stringify(safeSetting())!==JSON.stringify(s))throw Error('图配置或主GM在备份期间已改变。');
    await game.settings.set(ENGINE,SETTING,{...structuredClone(s),enabled});
   }
   await dispatch({probe:true});probed=true;
  }
  initialized=true;return ready();
 }
 function exactEffects(enemy,slug){const uuid=game.pf2e?.ConditionManager?.conditions?.get('enfeebled')?.uuid;return Array.from(enemy.actor.items.values()).filter(i=>i.type==='effect'&&i.system?.slug===slug&&i.system.context?.origin?.actor===enemy.actor.uuid&&i.system.context?.origin?.token===enemy.uuid&&i.system.duration?.unit==='rounds'&&i.system.duration.value===1&&i.system.duration.expiry==='turn-end'&&i.system.rules?.length===1&&i.system.rules[0].key==='GrantItem'&&i.system.rules[0].uuid===uuid&&i.system.rules[0].inMemoryOnly===true&&i.system.rules[0].alterations?.some(a=>a.mode==='override'&&a.property==='badge-value'&&a.value===2))}
 async function apply({nonce,enemy,expiry,authorize}){
  if(!ready()||!isActiveGM(game)||!/^[A-Za-z0-9-]{1,80}$/.test(nonce))throw Error('救赎瞥视后续未就绪。');
  const slug=slugFor(nonce);if(Array.from(enemy.actor.items.values()).some(i=>i.system?.slug===slug))throw Error('本次救赎瞥视效果已存在，不能重复执行。');
  await dispatch({enemy,slug,authorize});
  if(!isActiveGM(game))throw Error('后续期间主GM已改变。');
  const effects=exactEffects(enemy,slug);if(!effects.length)throw Error('没有本次原生衰弱效果的确切回执；不能重试。');
  if(effects.length>1)await enemy.actor.deleteEmbeddedDocuments('Item',effects.slice(1).map(i=>i.id));
  await lifecycle.arm({effect:effects[0],expiry,nonce});
  return {effectId:effects[0].id,slug};
 }
 return {register,initialize,ready,apply,template:()=>template&&structuredClone(template)};
}
