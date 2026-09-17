import {getSourceId} from './native-context.mjs';

export const FORCE_BARRAGE_WORKBENCH_SOURCE='Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros-internal.Macro.784iD1y6DBFSB5d2';
const SPELL_SOURCE='Compendium.pf2e.spells-srd.Item.gKKqvLohtrSJj3BM';
const COMMAND_HASH='464728041aab5a3d230b5d15b32fcad270fed7372c1bfd94a4ad3ee54fe670cb';
function reject(reason){const error=Error(`Force Barrage requires manual handling: ${reason}`);error.code=`force-barrage-${reason}`;throw error}
function requireBuild(game){
 if(game?.version!=='14.368'||game.system?.id!=='pf2e'||game.system?.version!=='8.5.1')reject('unsupported-system-build');
 for(const [id,version]of [['xdy-pf2e-workbench','7.7.5'],['pf2e-toolbelt','3.56.2']]){
  const module=game.modules?.get(id);if(module?.active!==true||module.version!==version)reject(`unsupported-${id}`);
 }
}
function requireNumbers(rank,actions){if(!Number.isInteger(rank)||rank<1||rank>3||!Number.isInteger(actions)||actions<1||actions>3)reject('unsupported-rank-or-actions')}
function uniqueSlice(source,start,end,{includeEnd=false}={}){
 const from=source.indexOf(start),to=source.indexOf(end,from+start.length);
 if(from<0||to<from||source.indexOf(start,from+start.length)!==-1||source.indexOf(end,to+end.length)!==-1)reject('unrecognized-source-boundary');
 return source.slice(from,to+(includeEnd?end.length:0));
}
function line(source,prefix){const lines=source.split('\n').filter(value=>value.startsWith(prefix));if(lines.length!==1)reject('unrecognized-source-line');return lines[0]}
const tokenDocument=token=>token?.document??token;
function liveToken(game,token){
 const doc=tokenDocument(token),scene=doc?.parent;
 if(!doc?.actor||typeof doc.uuid!=='string'||game.scenes?.get(scene?.id)!==scene||scene?.tokens?.get(doc.id)!==doc)reject('nonlive-token');
 return doc;
}

/** Load only the audited internal Macro document. This compiles four unchanged
 * upstream fragments; it does not execute the macro's UI, late cast or animation
 * control flow. Unknown bytes never reach Function. The provider owns native
 * cast authorization, payment evidence, roll evaluation and durable delivery. */
export async function loadForceBarrageWorkbench({game=globalThis.game,fromUuid=globalThis.fromUuid}={}){
 requireBuild(game);
 if(typeof fromUuid!=='function')reject('missing-source-resolver');
 const macro=await fromUuid(FORCE_BARRAGE_WORKBENCH_SOURCE);
 if(macro?.uuid!==FORCE_BARRAGE_WORKBENCH_SOURCE||macro.documentName!=='Macro'||macro.type!=='script'||typeof macro.command!=='string')reject('wrong-macro-source');
 const command=macro.command;
 if(!globalThis.crypto?.subtle)reject('unavailable-source-hash');
 const bytes=await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(command));
 const hash=Array.from(new Uint8Array(bytes),byte=>byte.toString(16).padStart(2,'0')).join('');
 if(hash!==COMMAND_HASH)reject('unreviewed-macro-command');
 // No formula or missile-count implementation is maintained here. Retain the
 // complete audited constructor block, including its original bonus branches.
 const count=new Function('mmdiag','mmch',line(command,'const multi = ')+'\nreturn multi;');
 const damage=new Function('token','mmch','a','DamageRoll',uniqueSlice(command,'\tlet dam;','\tconst droll = new DamageRoll(dam);',{includeEnd:true})+'\nreturn droll;');
 const name=new Function('targets','i','c',line(command,'\tconst name = ')+'\nreturn name;');
 const template=uniqueSlice(command,'\tdroll.toMessage(\n','\n\t);').slice('\tdroll.toMessage(\n'.length);
 const message=new Function('a','mmch','ChatMessage',`return (${template});`);
 function current(){requireBuild(game);if(macro.command!==command||macro.uuid!==FORCE_BARRAGE_WORKBENCH_SOURCE||macro.documentName!=='Macro'||macro.type!=='script')reject('changed-macro-source')}
 function getMissileCount({rank,actions}={}){current();requireNumbers(rank,actions);return count([null,actions],{rank})}
 async function run({actor,token,item,entry,rank,actions,allocations,bridge}={}){
  const missiles=getMissileCount({rank,actions}),original=item?.original??item;
  if(!actor||game.actors?.get(actor.id)!==actor||actor.type!=='character'||actor.isOwner!==true||actor.canAct!==true||actor.isDead===true)reject('ineligible-caster');
  if(original?.type!=='spell'||getSourceId(original)!==SPELL_SOURCE||actor.items?.get(original.id)!==original||original.actor!==actor||item.actor!==actor||item.uuid!==original.uuid)reject('wrong-owned-spell');
  if(entry?.actor!==actor||actor.items?.get(entry.id)!==entry||entry.type!=='spellcastingEntry'||entry.system?.prepared?.value!=='spontaneous'||original.system?.location?.value!==entry.id||original.system.location.signature!==true)reject('unsupported-spell-entry');
  const slots=entry.system?.slots?.[`slot${rank}`]?.value;
  if(!Number.isInteger(slots)||slots<1)reject('unavailable-slot');
  const source=liveToken(game,token);if(source.actor!==actor)reject('wrong-caster-token');
  const feats=actor.itemTypes?.feat,effects=actor.itemTypes?.effect;
  if(!Array.isArray(feats)||!Array.isArray(effects))reject('unknown-caster-features');
  if(feats.some(f=>f.slug==='sorcerous-potency')||effects.some(e=>e.slug==='effect-unleash-psyche'))reject('unverified-damage-bonus');
  if(!Array.isArray(allocations)||allocations.length===0)reject('missing-allocations');
  const seen=new Set(),snapshot=[];let total=0;
  for(const [index,allocation]of allocations.entries()){
   if(!Number.isSafeInteger(allocation?.count)||allocation.count<0||allocation.count>missiles)reject('invalid-allocation-count');
   const target=liveToken(game,allocation.targetToken);
   if(target.parent!==source.parent||allocation.targetUuid!==target.uuid||seen.has(target.uuid))reject('invalid-allocation-target');
   seen.add(target.uuid);total+=allocation.count;
   // Preserve upstream displayName/ownership privacy and original target index,
   // including zero allocations. Freeze data, never the native documents.
   const targetName=name([{document:target,actor:target.actor,name:target.name}],0,index+1);
   snapshot.push(Object.freeze({name:targetName,num:allocation.count,uuid:target.uuid}));
  }
  if(total!==missiles)reject('incorrect-allocation-total');
  Object.freeze(snapshot);
  if(typeof bridge?.payAndBindOriginalCast!=='function'||typeof bridge?.publishTarget!=='function')reject('missing-original-cast-bridge');
  const DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(Roll=>Roll.name==='DamageRoll');
  if(typeof DamageRoll!=='function'||typeof globalThis.ChatMessage?.getSpeaker!=='function'||typeof item.link!=='string')reject('missing-native-damage-api');
  const speaker=structuredClone(globalThis.ChatMessage.getSpeaker({actor,token:source}));
  const nativeMessage={getSpeaker:()=>structuredClone(speaker)},mmch=Object.freeze({rank,link:item.link});
  const damageSource={actor:{itemTypes:{feat:feats.map(f=>({slug:f.slug})),effect:effects.map(e=>({slug:e.slug}))}}};
  const pay=bridge.payAndBindOriginalCast.bind(bridge),publish=bridge.publishTarget.bind(bridge);
  const payment=await pay();
  // Native SpellcastingEntry.cast returns void. That is not payment proof.
  if(!payment||typeof payment!=='object'||Array.isArray(payment))reject('unproven-original-cast-payment');
  const targets=[];
  for(const allocation of snapshot){
   if(allocation.num===0)continue;
   const roll=damage(damageSource,mmch,allocation,DamageRoll);
   const messageData=message(allocation,mmch,nativeMessage);
   const result=await publish({roll,messageData,targetUuid:allocation.uuid});
   targets.push({targetUuid:allocation.uuid,count:allocation.num,result});
  }
  // The independent Workbench macro retains its own animation. This adapter
  // makes no Sequencer/media call and never repeats mechanics for display.
  return {status:'completed',missiles,targets,display:{status:'manual'}};
 }
 return Object.freeze({getMissileCount,run});
}
