const values=collection=>Array.from(collection?.values?.()??collection??[]);
/** Recover once, then track actors at document mutations. Pure token movement
 * does no work. The caller's predicate must read current state without cloning it. */
export function createActorStateIndex({game,matches}){
 const entries=new Map();let seeded=false,installed=false;
 const remember=actor=>{if(!actor?.uuid)return;if(matches(actor))entries.set(actor.uuid,actor);else entries.delete(actor.uuid)};
 const refresh=actor=>{remember(actor);if(!actor?.isToken)for(const token of actor?.getDependentTokens?.({concreteOnly:true})??[])if(token.actor&&token.actor!==actor)remember(token.actor)};
 const scene=scene=>{for(const token of values(scene?.tokens))remember(token.actor)};
 function seed(){if(seeded)return;seeded=true;for(const actor of values(game.actors))remember(actor);for(const document of values(game.scenes))scene(document)}
 function removeToken(token){
  if(token?.uuid)for(const key of entries.keys())if(key.startsWith(`${token.uuid}.Actor.`))entries.delete(key);
  const actor=token?.actor;if(actor&&game.actors?.get?.(actor.id)!==actor)entries.delete(actor.uuid);
 }
 function register(Hooks){
  if(installed)return;installed=true;seed();
  for(const name of ['createActor','updateActor'])Hooks.on(name,refresh);
  Hooks.on('deleteActor',actor=>entries.delete(actor.uuid));
  for(const name of ['createItem','updateItem','deleteItem'])Hooks.on(name,item=>refresh(item.actor??item.parent));
  Hooks.on('createToken',token=>refresh(token.actor));Hooks.on('deleteToken',removeToken);
  Hooks.on('updateToken',(token,changes)=>{if(Object.hasOwn(changes,'actorId')||Object.hasOwn(changes,'actorLink')){removeToken(token);refresh(token.actor)}});
  Hooks.on('createScene',scene);
  Hooks.on('deleteScene',document=>{for(const key of entries.keys())if(key.startsWith(`Scene.${document.id}.Token.`))entries.delete(key)});
 }
 return {refresh,register,values(){seed();return [...entries.values()]}};
}
