// Only protects the empty Shield Wall candidate branch in the audited
// campaigns. It does not implement the Shield Wall reaction.
const MODULE='pf2e-reaction',EVENT='createItem';
const TARGETS=new Set(['-','sog','pnvfcgjbf2cjp7gz','ujx5r8oipw7ercdr','team-automation-qa2']);
const PROFILES=Object.freeze({
 '8.5.0':Object.freeze({coreGeneration:14,system:'8.5.0'}),
 '8.5.1':Object.freeze({coreGeneration:14,system:'8.5.1'}),
});
const installations=new WeakMap();
const sourceOf=fn=>typeof fn==='function'?Function.prototype.toString.call(fn):'';
const unsupported=reason=>Object.freeze({status:'unsupported',reason,dispose(){}});

function emptyCandidateRaise(game,item,userId){
 if(userId!==game.userId||item?.type!=='effect'||item.slug!=='effect-raise-a-shield')return false;
 const actor=item.actor,turns=game.combat?.turns;
 if(!actor||!['character','npc','familiar'].includes(actor.type)||actor.alliance!=='party'||!Array.isArray(turns))return false;
 const source=turns.find(row=>row?.actorId===actor.id);
 if(!source||source.actor!==actor)return false;
 // Do not invent candidates from names, linked action UUIDs or cached party
 // membership. Any actual Shield Wall feat leaves the upstream path intact.
 return turns.every(row=>!row?.actor||Array.isArray(row.actor.itemTypes?.feat)&&!row.actor.itemTypes.feat.some(feat=>feat.slug==='shield-wall'));
}

export function registerReactionShieldWallEmptyCompatibility({game,Hooks}={}){
 if(!Hooks||!['object','function'].includes(typeof Hooks))return Promise.resolve(unsupported('missing-hooks'));
 const existing=installations.get(Hooks);if(existing)return existing.promise;
 const module=game?.modules?.get(MODULE),world=game?.world,worldId=world?.id,profile=PROFILES[game?.system?.version];
 const profileCurrent=()=>!!profile&&game?.modules?.get(MODULE)===module&&module?.active===true&&game.release?.generation===profile.coreGeneration&&game.system?.version===profile.system&&game.world===world&&world?.id===worldId&&TARGETS.has(worldId);
 if(!profileCurrent())return Promise.resolve(unsupported('unknown-dependency-or-world-profile'));
 const entries=Hooks.events?.[EVENT];
 if(!Array.isArray(entries))return Promise.resolve(unsupported('missing-create-item-hooks'));
 const matching=entries.filter(entry=>{const source=sourceOf(entry?.fn);return source.includes('shield-wall')&&source.includes('effect-raise-a-shield');});
 if(matching.length!==1)return Promise.resolve(unsupported('ambiguous-create-item-handler'));
 const entry=matching[0],original=entry.fn,id=entry.id,index=entries.indexOf(entry);
 const entryCurrent=()=>Hooks.events?.[EVENT]===entries&&entries[index]===entry&&entry.id===id&&entry.hook===EVENT&&entry.once===false&&Number.isSafeInteger(id)&&id>0&&entry.fn===original&&Object.getOwnPropertyDescriptor(entry,'fn')?.writable===true;
 if(!entryCurrent())return Promise.resolve(unsupported('unknown-hook-entry-profile'));
 const state={promise:null};
 // Repeated initializers share one registration.
 installations.set(Hooks,state);
 state.promise=Promise.resolve().then(async()=>{
  try{
   if(!profileCurrent()||!entryCurrent())return unsupported('identity-changed-during-verification');
   let active=true;
   const wrapped=function(item,options,userId,...rest){
    if(active&&profileCurrent()&&Hooks.events?.[EVENT]?.includes(entry)&&entry.id===id&&emptyCandidateRaise(game,item,userId))return Promise.resolve();
    return Reflect.apply(original,this,[item,options,userId,...rest]);
   };
   entry.fn=wrapped;
   return Object.freeze({status:'installed',scope:'empty-shield-wall-candidates-only',sourceSHA256:null,callbackSHA256:null,hook:{event:EVENT,id,index},dispose(){
    active=false;
    if(Hooks.events?.[EVENT]?.includes(entry)&&entry.id===id&&entry.fn===wrapped&&Object.getOwnPropertyDescriptor(entry,'fn')?.writable===true)entry.fn=original;
    if(installations.get(Hooks)===state)installations.delete(Hooks);
   }});
  }catch(error){return unsupported(String(error.message??error));}
 }).then(result=>{
  if(result.status==='unsupported'&&installations.get(Hooks)===state)installations.delete(Hooks);
  return result;
 });
 return state.promise;
}
