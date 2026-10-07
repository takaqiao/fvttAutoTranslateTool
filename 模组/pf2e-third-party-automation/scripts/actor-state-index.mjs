const values=collection=>Array.from(collection?.values?.()??collection??[]);
/** Recover once, then track actors at document mutations. Pure token movement
 * does no work. The caller's predicate must read current state without cloning it. */
export function createActorStateIndex({game,matches,observedRecovery=false}){
 const entries=new Map();let seeded=false,installed=false;
 const remember=(actor,evaluate=matches)=>{if(!actor?.uuid)return;if(evaluate(actor))entries.set(actor.uuid,actor);else entries.delete(actor.uuid)};
 const refresh=actor=>{remember(actor);if(!actor?.isToken)for(const token of actor?.getDependentTokens?.({concreteOnly:true})??[])if(token.actor&&token.actor!==actor)remember(token.actor)};
 const scene=(scene,evaluate=matches)=>{for(const token of values(scene?.tokens))remember(token.actor,evaluate)};
 function seed(evaluate=matches){if(seeded)return false;seeded=true;for(const actor of values(game.actors))remember(actor,evaluate);for(const document of values(game.scenes))scene(document,evaluate);return true}
 const recoverIndex=observedRecovery?()=>{if(!seeded)throw Error('Observed actor index requires initialization');return false}:seed;
 function removeToken(token){
  if(token?.uuid)for(const key of entries.keys())if(key.startsWith(`${token.uuid}.Actor.`))entries.delete(key);
  const actor=token?.actor;if(actor&&game.actors?.get?.(actor.id)!==actor)entries.delete(actor.uuid);
 }
 function register(Hooks,{recover=true}={}){
  if(installed)return;installed=true;if(recover)recoverIndex();
  for(const name of ['createActor','updateActor'])Hooks.on(name,refresh);
  Hooks.on('deleteActor',actor=>entries.delete(actor.uuid));
  for(const name of ['createItem','updateItem','deleteItem'])Hooks.on(name,item=>refresh(item.actor??item.parent));
  Hooks.on('createToken',token=>refresh(token.actor));Hooks.on('deleteToken',removeToken);
  Hooks.on('updateToken',(token,changes={})=>{if(Object.hasOwn(changes,'actorId')||Object.hasOwn(changes,'actorLink')){removeToken(token);refresh(token.actor)}});
  Hooks.on('createScene',document=>scene(document));
  Hooks.on('deleteScene',document=>{for(const key of entries.keys())if(key.startsWith(`Scene.${document.id}.Token.`))entries.delete(key)});
 }
 const index={refresh,register,recover:recoverIndex,invalidate(){entries.clear();seeded=false},values(){recoverIndex();return [...entries.values()]}};
 if(observedRecovery){
  index.initialize=cohort=>{
   if(seeded)return false;
   try{entries.clear();for(const actor of cohort){const uuid=actor?.uuid;if(uuid)entries.set(uuid,actor)}seeded=true;return true}
   catch(error){entries.clear();seeded=false;throw error}
  };
  index.observe=(actor,related)=>{
   if(related!==false||!seeded)return false;
   const uuid=actor?.uuid;return entries.get(uuid)===actor&&entries.delete(uuid);
  };
 }
 return index;
}
