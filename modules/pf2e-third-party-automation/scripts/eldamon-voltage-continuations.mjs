import {VOLTAGE_MODULE_ID as ID,voltageRollOption} from './eldamon-voltage.mjs';
const values=c=>Array.from(c?.values?.()??c??[]);
const put=(map,key,value)=>{if(!key)return;let bucket=map.get(key);if(!bucket)map.set(key,bucket=new Set());bucket.add(value)};
const drop=(map,key,value)=>{const bucket=map.get(key);bucket?.delete(value);if(bucket&&!bucket.size)map.delete(key)};
export const canReadVoltageMessage=(message,user)=>!!user&&(user.isGM===true||!message?.blind&&message?.isContentVisible!==false&&(!message?.whisper?.length||message.whisper.includes(user.id)||(message.author?.id??message.user?.id)===user.id));
/** These are lookup projections, never activity authority. Recover documents once;
 * subsequent state and message mutations update only their affected entries. */
export function createVoltageContinuationIndex({game}){
 const entries=new Map(),sources=new Map(),participants=new Map(),tokens=new Map(),origins=new Map(),cards=new Map(),saveOptions=new Map();
 const attacks=new Map(),saves=new Map(),messages=new Map();let seeded=false,registered=false;
 function removeEntry(key){
  const entry=entries.get(key);if(!entry)return;const a=entry.activation;
  for(const uuid of new Set([a.actorUuid,a.trigger?.targetActorUuid]))drop(participants,uuid,key);
  for(const uuid of new Set([a.originUuid,a.trigger?.targetUuid]))drop(tokens,uuid,key);
  drop(origins,a.originUuid,key);drop(saveOptions,voltageRollOption(a),key);
  for(const uuid of [a.messageUuid,a.trigger?.attackUuid,a.native?.save?.messageUuid,a.native?.damage?.messageUuid])drop(cards,uuid,key);
  entries.delete(key);
 }
 function refresh(actor){
  if(!actor?.uuid)return;
  for(const key of sources.get(actor.uuid)??[])removeEntry(key);sources.delete(actor.uuid);
  for(const a of Object.values(actor.flags?.[ID]?.voltage?.activations??{})){
   if(!['armed','claimed','uncertain'].includes(a.status)||a.actorUuid!==actor.uuid||typeof a.messageUuid!=='string')continue;
   const key=actor.uuid+':'+a.nonce;entries.set(key,{actor,activation:a});put(sources,actor.uuid,key);
   for(const uuid of new Set([actor.uuid,a.trigger?.targetActorUuid]))put(participants,uuid,key);
   for(const uuid of new Set([a.originUuid,a.trigger?.targetUuid]))put(tokens,uuid,key);
   if(a.status==='armed')put(origins,a.originUuid,key);
   put(saveOptions,voltageRollOption(a),key);
   for(const uuid of [a.messageUuid,a.trigger?.attackUuid,a.native?.save?.messageUuid,a.native?.damage?.messageUuid])put(cards,uuid,key);
  }
 }
 function forget(message){
  const previous=messages.get(message?.id);if(!previous)return;
  drop(attacks,previous.target,previous.message);
  for(const option of previous.options)drop(saves,option,previous.message);messages.delete(message.id);
 }
 function remember(message){
  forget(message);const c=message?.flags?.pf2e?.context;
  if(!message?.id||!message.isCheckRoll||!message.rolls?.[0]?._evaluated)return;
  const target=c?.type==='attack-roll'?c.target?.token:null;
  const options=c?.type==='saving-throw'?(c.options??[]).filter(o=>typeof o==='string'&&o.startsWith(ID+':voltage:')):[];
  if(!target&&!options.length)return;
  if(target)put(attacks,target,message);for(const option of options)put(saves,option,message);
  messages.set(message.id,{message,target,options});
 }
 function seed(){
  if(seeded)return;seeded=true;
  for(const actor of values(game.actors))refresh(actor);
  for(const scene of values(game.scenes))for(const token of values(scene.tokens))refresh(token.actor);
  for(const message of values(game.messages))remember(message);
 }
 const selected=keys=>[...keys??[]].map(key=>entries.get(key)).filter(Boolean);
 const current=items=>[...items??[]].filter(m=>game.messages.get(m.id)===m);
 function forCard(message){
  seed();const keys=new Set(cards.get(message?.uuid)??[]),c=message?.flags?.pf2e?.context;
  if(message?.isCheckRoll&&message.rolls?.[0]?._evaluated&&c?.type==='attack-roll'&&['success','criticalSuccess'].includes(c.outcome)){
   for(const key of origins.get(c.target?.token)??[])if(message.timestamp>=entries.get(key).activation.channelTimestamp)keys.add(key);
  }
  if(c?.type==='saving-throw')for(const option of c.options??[])for(const key of saveOptions.get(option)??[])keys.add(key);
  return selected(keys);
 }
 function register(Hooks){
  if(registered)return;registered=true;seed();
  for(const name of ['createActor','updateActor'])Hooks.on(name,refresh);
  Hooks.on('deleteActor',actor=>{for(const key of sources.get(actor.uuid)??[])removeEntry(key);sources.delete(actor.uuid)});
  for(const name of ['createItem','updateItem','deleteItem'])Hooks.on(name,item=>refresh(item.actor??item.parent));
  Hooks.on('createToken',token=>refresh(token.actor));
  Hooks.on('createChatMessage',remember);Hooks.on('updateChatMessage',remember);Hooks.on('deleteChatMessage',forget);
 }
 return {register,refresh,remember,forCard,forActor(uuid){seed();return selected(participants.get(uuid))},forToken(uuid){seed();return selected(tokens.get(uuid))},
  actors(){seed();return [...new Set([...entries.values()].map(e=>e.actor))]},
  attacks(a){seed();return current(attacks.get(a.originUuid)).filter(m=>m.timestamp>=a.channelTimestamp&&['success','criticalSuccess'].includes(m.flags.pf2e.context.outcome))},
  saves(a){seed();return current(saves.get(voltageRollOption(a)))} };
}
