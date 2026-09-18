import {MODULE_ID} from './rules.mjs';

const REACTION='pf2e-reaction',WORLD='ujx5r8oipw7ercdr',PATH='CONFIG.Combatant.documentClass.prototype.getFlag';
const BUNDLE_SHA256='4a81322796ce1c6ed545edc09e1aa3a96a9c8a96dfd034403bf657068ed7036c';
const CALLBACK_SHA256='5d2a14c36862e10b963978b68ed7b5d5741993ef83169202658220eef18319b1';
const KEYS=new Set(['state','hydra-heads','triple-opportunity','combat-reflexes','tactical-reflexes','inexhaustible-countermoves','reflexive-riposte','quick-shield-block']);
const installations=new WeakMap();
const values=c=>Array.from(c?.values?.()??c??[]);
const sourceOf=fn=>typeof fn==='function'?Function.prototype.toString.call(fn):'';
const defaultHash=async text=>Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),b=>b.toString(16).padStart(2,'0')).join('');
const defaultFetch=async()=>{const path=globalThis.foundry?.utils?.getRoute?.('modules/pf2e-reaction/pf2e-reaction.js')??'/modules/pf2e-reaction/pf2e-reaction.js';const r=await fetch(path,{cache:'no-store'});if(!r.ok)throw Error('reaction-source-unavailable');return r.text()};
const unsupported=reason=>Object.freeze({status:'unsupported',reason,ready:()=>false,dispose(){}});

/** Audited availability reads and old delegated Reaction cards only. Raw flags,
 * native resets/refunds and already-selected third-party modifiers are not
 * rewritten. Every participating client must install its own verified guard. */
export function registerRoaringReactionCompatibility({game,query,libWrapper=globalThis.libWrapper,document=globalThis.document,jQuery=globalThis.jQuery,fetchSource=defaultFetch,hashSource=defaultHash,notify=text=>globalThis.ui?.notifications?.warn(text)}={}){
 if(!document||!['object','function'].includes(typeof document))return Promise.resolve(unsupported('missing-document'));
 const previous=installations.get(document);if(previous)return previous.promise;
 const world=game?.world,module=game?.modules?.get(REACTION);
 const profileCurrent=()=>game?.world===world&&world?.id===WORLD&&game?.system?.id==='pf2e'&&game.system.version==='8.5.1'&&game.release?.generation===14&&game.modules.get(REACTION)===module&&module?.active===true&&module.version==='1.4.3';
 if(!profileCurrent())return Promise.resolve(unsupported('unknown-world-or-dependency-profile'));
 if(typeof query!=='function'||typeof jQuery!=='function'||typeof jQuery._data!=='function'||typeof document.addEventListener!=='function'||typeof document.removeEventListener!=='function'||typeof libWrapper?.register!=='function'||typeof libWrapper?.unregister!=='function')return Promise.resolve(unsupported('missing-native-adapter-interface'));
 const entries=jQuery._data(document,'events')?.click;
 const matches=Array.isArray(entries)?entries.filter(e=>e?.selector==='.reaction-check'):[];
 if(matches.length!==1)return Promise.resolve(unsupported('ambiguous-reaction-click-handler'));
 const entry=matches[0],handler=entry.handler,guid=entry.guid;
 const handlerCurrent=()=>jQuery._data(document,'events')?.click===entries&&entries.includes(entry)&&entries.filter(e=>e?.selector==='.reaction-check').length===1&&entry.handler===handler&&entry.guid===guid&&Number.isSafeInteger(guid)&&guid>0&&entry.type==='click'&&entry.origType==='click'&&entry.namespace==='';
 if(!handlerCurrent()||!sourceOf(handler))return Promise.resolve(unsupported('unknown-reaction-click-profile'));
 const state={promise:null};installations.set(document,state);
 state.promise=(async()=>{
  let active=false,registration,listener;
  const ready=()=>active&&profileCurrent()&&handlerCurrent();
  const liveCombatant=c=>!!c?.actor&&game.combats?.get(c.parent?.id)===c.parent&&c.parent?.combatants?.get(c.id)===c;
  const assess=actor=>{try{const r=query(actor);return r&&!r.then&&['restricted','clear','manual'].includes(r.status)?r:{status:'manual',reason:'reaction-query-unavailable'}}catch{return {status:'manual',reason:'reaction-query-failed'}}};
  const dispose=()=>{active=false;if(listener){document.removeEventListener('click',listener,true);listener=null}if(registration!==undefined){libWrapper.unregister(MODULE_ID,registration);registration=undefined}if(installations.get(document)===state)installations.delete(document)};
  try{
   if(await hashSource(await fetchSource())!==BUNDLE_SHA256)return unsupported('unknown-reaction-bundle');
   if(await hashSource(sourceOf(handler))!==CALLBACK_SHA256)return unsupported('unknown-reaction-click-callback');
   if(!profileCurrent()||!handlerCurrent())return unsupported('identity-changed-during-verification');
   const read=function(wrapped,...args){
    const native=wrapped(...args);
    if(!ready()||args[0]!==REACTION||!KEYS.has(args[1])||!liveCombatant(this))return native;
    return assess(this.actor).status==='restricted'?(args[1]==='state'?false:0):native;
   };
   listener=event=>{
    if(!ready())return;
    const button=event.target?.closest?.('.reaction-check');if(!button||!document.contains(button))return;
    let verdict={status:'manual',reason:'unproven-native-reaction-card'};
    try{
     // Match the audited original handler's exact jQuery traversal/data cache.
     // Do not redirect its cId to a guessed encounter or resolve a new item.
     const id=jQuery(button).parent().parent().parent().data('message-id'),message=typeof id==='string'?game.messages?.get(id):null;
     if(message?.id===id){
      const cId=message.getFlag(REACTION,'cId');
      const candidates=values(game.combats).flatMap(c=>Array.isArray(c.turns)?c.turns.filter(row=>row?._id===cId&&row.parent===c&&liveCombatant(row)):[]);
      const selected=game.combat?.turns?.find(c=>c._id===cId);
      if(typeof cId==='string'&&cId&&candidates.length===1&&candidates[0]===selected&&selected.parent===game.combat&&game.combats.get(game.combat.id)===game.combat)verdict=assess(selected.actor);
      else verdict={status:'manual',reason:'ambiguous-or-different-viewed-combatant'};
     }
    }catch{verdict={status:'manual',reason:'unreadable-native-reaction-card'}}
    if(verdict.status==='clear')return;
    event.preventDefault();event.stopImmediatePropagation();
    notify(verdict.status==='restricted'?'轰然喝彩：此角色当前不能使用反应。未执行旧提示卡，请由 GM 核对来源。':'轰然喝彩：未执行这张旧反应提示。请 GM 核对它对应的角色及当前反应资格。');
   };
   // libWrapper registration IDs allow disposal of this registration only,
   // preserving other packages' wrappers on the inherited Combatant method.
   registration=libWrapper.register(MODULE_ID,PATH,read,'WRAPPER');
   if(!Number.isSafeInteger(registration)){if(registration===undefined)registration=PATH;throw Error('unknown-libWrapper-registration-id')}
   document.addEventListener('click',listener,true);active=true;
   return Object.freeze({status:'installed',scope:'roaring-reaction-availability-and-old-card',sourceSHA256:BUNDLE_SHA256,callbackSHA256:CALLBACK_SHA256,ready,dispose});
  }catch(error){dispose();return unsupported(String(error.message??error))}
 })().then(result=>{if(result.status==='unsupported'&&installations.get(document)===state)installations.delete(document);return result});
 return state.promise;
}
