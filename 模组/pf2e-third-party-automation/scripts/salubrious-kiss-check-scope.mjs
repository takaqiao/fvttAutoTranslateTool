import {validateSalubriousPrivacy} from './salubrious-privacy.mjs';
import {MODULE_ID} from './rules.mjs';
import {assertSource} from './salubrious-kiss-context.mjs';
import {beforeNativeRoll} from './native-owner-operations.mjs';
const marker=nonce=>`${MODULE_ID}:salubrious-check:${nonce}`;
const options=c=>new Set(c.options??[]);

/** One private exact-context mode capability in the existing Check entry.
 * The verified Patreon callback may temporarily force gm/blind. Restore only
 * this source's captured native mode, after rechecking hidden/secret floors.
 * The legacy public-named methods remain aliases for old integration callers. */
export function createSalubriousCheckScope({game,Hooks=globalThis.Hooks,isExplorationContext=()=>false}){
 const scopes=new Map(),contexts=new WeakMap(),exploration=new Map();
 async function runExploration(input,operation){
  if(!isExplorationContext(input.ctx,input.activity.id)||exploration.has(input.activity.id))throw Error('Invalid exploration check context');
  const scope={...input,entered:false};exploration.set(input.activity.id,scope);
  try{const result=await operation();input.ctx.validate();if(!scope.entered)throw Error('Native exploration check not observed');return result}finally{exploration.delete(input.activity.id)}
 }
 function valid(scope,c,{provenBlind=false}={}){
  const {actor,item,token,claim}=scope;
  assertSource({game,actor,item,token,user:game.user,privacy:claim.privacy});
  if(game.user.id!==claim.userId||actor.uuid!==claim.actorUuid||item.uuid!==claim.itemUuid||token.uuid!==claim.tokenUuid)throw Error('Invalid native treatment owner scope');
  if(c&&(c.actor!==actor||c.item!==item||c.token!==token||c.type!=='skill-check'||c.action!=='treat-wounds'||!c.domains?.includes('occultism')||c.dc?.value!==claim.dc||c.createMessage!==false||!options(c).has('action:treat-wounds')||!options(c).has(marker(claim.nonce))))throw Error('Native treatment scope identity mismatch');
  const mode=validateSalubriousPrivacy({game,user:game.user,token,item,privacy:claim.privacy,context:c});
  if(c&&c.messageMode!==mode&&!(provenBlind&&c.messageMode===(game.user.isGM?'gm':'blind')))throw Error('Actual private treatment mode differs from source authorization');
 }
 async function run({actor,item,token,claim},operation){
  if(!claim?.nonce)throw Error('Invalid treatment scope');const key=marker(claim.nonce);
  if(scopes.has(key))throw Error('Treatment scope already running');
  const scope={actor,item,token,claim:{...claim},entered:false};valid(scope);scopes.set(key,scope);
  try{const result=await operation();if(!scope.entered)throw Error('Native treatment check boundary was not observed');valid(scope,scope.context);return result;}
  finally{if(scope.context)contexts.delete(scope.context);scopes.delete(key)}
 }
 function allowPatreonBlanketBlind(context){
  const scope=contexts.get(context);if(!scope||scopes.get(marker(scope.claim.nonce))!==scope)return false;
  try{valid(scope,context);return true}catch{return false}
 }
 function matching(context){const matches=[...options(context)].filter(o=>scopes.has(o));if(matches.length>1)throw Error('Conflicting treatment scope');return matches.length?scopes.get(matches[0]):null;}
 function bind(scope,check,context){
  valid(scope,context);if(check?.slug!=='occultism'||scope.context&&(scope.context!==context||scope.check!==check))throw Error('Native treatment check/context identity changed');
  scope.context=context;scope.check=check;contexts.set(context,scope);
 }
 // Patreon is the first WRAPPER, before our existing MIXED Check boundary.
 // Only a run's private nonce can claim that first genuine check/context. The
 // opaque closure remains tied to them; it cannot authorize a copied context.
 function acquirePatreonPublicScope(check,context){
  try{
   const scope=matching(context);if(!scope||scope.entered||scope.acquired)return null;
   bind(scope,check,context);scope.acquired=true;let used=false;
   return Object.freeze({mode:scope.claim.privacy?.mode??'public',forcedMode:game.user.isGM?'gm':'blind',revalidate(){if(used)return false;used=true;
    if(scopes.get(marker(scope.claim.nonce))!==scope||contexts.get(context)!==scope||scope.context!==context||scope.check!==check)return false;
    try{valid(scope,context,{provenBlind:true});return true}catch{return false}
   }});
  }catch{return null}
 }
 async function interceptCheck(wrapped,check,context={},event,callback){
  const explorationMarkers=[...options(context)].filter(o=>o.startsWith('exploration-activity:'));
  if(explorationMarkers.length){
   const scope=exploration.get(explorationMarkers[0].slice('exploration-activity:'.length));
   if(scope){
    const expectedDC={trained:15,expert:20,master:30,legendary:40}[scope.activity.options.rank??'trained'];
    if(explorationMarkers.length!==1||scope.entered||context.actor!==scope.healer||context.type!=='skill-check'||context.dc?.value!==expectedDC||!options(context).has('action:treat-wounds')||!context.domains?.includes(scope.activity.options.skill??'medicine')||!isExplorationContext(scope.ctx,scope.activity.id))throw Error('Exploration native context mismatch');
    scope.ctx.validate();scope.assertQualification?.();scope.entered=true;
    // Assurance is a native substitution, never a roll option that claims a roll.
    if(scope.activity.options.assurance){const substitution=context.substitutions?.find(s=>s.slug==='assurance'&&!s.ignored);if(!substitution||context.substitutions.some(s=>s!==substitution&&s.required))throw Error('Native Assurance substitution unavailable');for(const sub of context.substitutions)sub.selected=sub===substitution;context.options.add('substitute:assurance');check.calculateTotal(context.options)}
    if(scope.ctx.nativeDialogMode!=='owner-preference'){
     context.skipDialog=true;
     return wrapped(check,context,event,callback);
    }
    return beforeNativeRoll({Hooks,marker:'exploration-activity:'+scope.activity.id,
     showDialog:true,signal:scope.ctx.executionSignal,commit:async()=>{},
     assertLive:()=>{scope.ctx.validate();scope.assertQualification?.()},native:()=>wrapped(check,context,event,callback)});
   }
  }
  const scope=matching(context);if(!scope)return wrapped(check,context,event,callback);
  if(scope.entered)throw Error('A treatment scope may enter native dice only once');
  bind(scope,check,context);
  scope.entered=true;scope.context=context;contexts.set(context,scope);
  try{const result=await wrapped(check,context,event,callback);valid(scope,context);return result}
  finally{contexts.delete(context)}
 }
 return {run,runExploration,interceptCheck,allowPatreonBlanketBlind,acquirePatreonPublicScope,acquirePatreonModeScope:acquirePatreonPublicScope};
}
