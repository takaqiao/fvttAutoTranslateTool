import {MODULE_ID as ID} from './rules.mjs';

const HELPER='pf2e-flatcheck-helper',HIDDEN='tpa-quiet-message';
const own=message=>message?.flags?.[ID]??{};
const author=message=>message.author?.id??message.author;
const string=value=>typeof value==='string'&&value.length>0;
const total=value=>Number.isInteger(value)&&value>=1&&value<=20;

/** Presentation only: the original documents and native visibility remain intact. */
export function createQuietChat({game,refresh=message=>globalThis.ui?.chat?.updateMessage(message,{notify:false})}={}){
 const outgoing=new Map(),incoming=new Map();let indexed=false;
 const readable=message=>message?.isContentVisible===true;
 const gmWhisper=message=>message?.whisper?.length>0&&message.whisper.every(id=>game.users.get(id)?.isGM);
 const receiptProof=message=>{
  const proof=own(message).dualStrikeFlatCheck;
  return proof&&message.blind===true&&gmWhisper(message)&&proof.scope==='dual-strike'&&string(proof.activityMessageId)&&string(proof.nonce)&&string(proof.actorUuid)&&proof.userId===author(message)&&[0,1].includes(proof.index)?proof:null;
 };
 const summaryParent=message=>own(message).socialAlarm?.checkMessageId??(own(message).reactionChecks?.kind==='pointed-result'?own(message).reactionChecks.checkId:null);
 function index(message,deleted=false){
  const flags=own(message),proof=flags.dualStrikeFlatCheck;
  const links=new Set(deleted?[]:[proof?.activityMessageId,proof?.nativeMessageId,flags.dualStrikeFlatCheckRoll?.receiptId,flags.dualStrikeFlatCheckId,summaryParent(message)].filter(id=>string(id)&&id!==message.id));
  const previous=outgoing.get(message.id);
  if((previous?.size??0)===links.size&&[...links].every(id=>previous.has(id)))return;
  for(const target of outgoing.get(message.id)??[]){const sources=incoming.get(target);sources?.delete(message.id);if(!sources?.size)incoming.delete(target);}
  outgoing.delete(message.id);if(deleted)return;
  if(!links.size)return;outgoing.set(message.id,links);
  for(const target of links){if(!incoming.has(target))incoming.set(target,new Set());incoming.get(target).add(message.id);}
 }
 function ensureIndex(){
  if(indexed)return;indexed=true;
  for(const message of game.messages.values())index(message);
 }
 const neighbors=id=>new Set([...outgoing.get(id)??[],...incoming.get(id)??[]]);
 function connected(id){
  const ids=new Set([id]);for(const current of ids)for(const neighbor of neighbors(current))ids.add(neighbor);return ids;
 }
 function changed(message,deleted=false){
  ensureIndex();const related=connected(message.id);index(message,deleted);for(const id of connected(message.id))related.add(id);
  for(const id of related){const current=game.messages.get(id);if(id!==message.id&&current?.logged)refresh(current);}
 }
 function flatSource(receipt){
  const proof=receiptProof(receipt);if(!proof)return null;
  const candidates=proof.nativeMessageId?[game.messages.get(proof.nativeMessageId)]:proof.status==='pending'?[...incoming.get(receipt.id)??[]].map(id=>game.messages.get(id)):[];
  const matches=candidates.filter(message=>{
   const link=own(message).dualStrikeFlatCheckRoll,context=message?.flags?.pf2e?.context;
   return link?.receiptId===receipt.id&&link.nonce===proof.nonce&&author(message)===proof.userId&&message.speaker?.actor===receipt.speaker?.actor&&context?.type==='flat-check'&&Number.isFinite(context.dc?.value)&&message.rolls?.length===1&&total(message.rolls[0].total);
  });
  return matches.length===1?matches[0]:null;
 }
 function attackFor(receipt){
  const proof=receiptProof(receipt),checks=receipt.flags?.[HELPER]?.flatchecks;if(proof?.status!=='passed'||!checks)return null;
  for(const id of incoming.get(receipt.id)??[]){
   const attack=game.messages.get(id),flags=own(attack),operation=flags.nativeOwnerOperation,utility=attack?.flags?.[HELPER];
   if(flags.dualStrikeFlatCheckId===receipt.id&&operation?.activityMessageId===proof.activityMessageId&&operation.nonce===proof.nonce&&operation.scope===proof.scope&&author(attack)===proof.userId&&attack.speaker?.actor===receipt.speaker?.actor&&attack.flags.pf2e?.context?.type==='attack-roll'&&utility?.revealed===true&&JSON.stringify(utility.flatchecks)===JSON.stringify(checks)&&readable(attack))return attack;
  }
  return null;
 }
 const sameSpeaker=(parent,source)=>string(parent?.speaker?.actor)&&parent.speaker.actor===source?.speaker?.actor;
 function notesFor(parent){
  const notes=new Map();if(game.messages.get(parent?.id)!==parent||!readable(parent))return notes;
  for(const id of incoming.get(parent.id)??[]){
   const message=game.messages.get(id);if(!message)continue;
   const proof=receiptProof(message);
   if(proof?.activityMessageId===parent.id){
    if(parent.rolls?.length||!sameSpeaker(parent,message)||!['pending','passed','failed'].includes(proof.status))continue;
    const source=flatSource(message);if(!source||!readable(source)||source.flags.pf2e.context.dc.visible===false&&!game.user.isGM||attackFor(message))continue;
    const status={pending:'待结算',passed:'通过',failed:'未通过'}[proof.status];
    const key=`flat-${proof.index}`,previous=notes.get(key)?.source,time=source.timestamp??0,previousTime=previous?.timestamp??0;
    if(!previous||time>previousTime||time===previousTime&&source.id>previous.id)notes.set(key,{source,text:`第${proof.index+1}击：平检 ${source.rolls[0].total} / DC ${source.flags.pf2e.context.dc.value}，${status}。`});
   }else if(summaryParent(message)===parent.id&&parent.flags?.pf2e?.context?.type==='skill-check'&&sameSpeaker(parent,message)&&gmWhisper(message)&&game.users.get(author(message))?.isGM&&readable(message)&&string(message.content)){
    notes.set(message.id,{source:message,html:message.content});
   }
  }
  return notes;
 }
 function hidden(message){
  const flags=own(message),damage=flags.nativeOwnerDamageResult;
  if(damage&&message.blind===true&&gmWhisper(message)&&string(damage.activityMessageId)&&string(damage.nonce)&&string(damage.actorUuid)&&['damage','spell-damage','formula-damage'].includes(damage.type))return true;
  const proof=receiptProof(message);
  if(proof)return proof.status!=='failed'||!!flatSource(message);
  const receipt=game.messages.get(flags.dualStrikeFlatCheckRoll?.receiptId),flat=receipt&&flatSource(receipt);
  if(flat===message&&readable(message)){
   if(attackFor(receipt))return true;
   const parent=game.messages.get(receiptProof(receipt).activityMessageId);
   return [...notesFor(parent).values()].some(note=>note.source===message);
  }
  const parent=game.messages.get(summaryParent(message));
  return [...notesFor(parent).values()].some(note=>note.source===message);
 }
 function render(message,html){
  if(!html?.matches?.('.message[data-message-id]')||html.dataset.messageId!==message?.id)return;
  ensureIndex();index(message);
  html.classList.toggle(HIDDEN,hidden(message));
  for(const note of html.querySelectorAll('[data-tpa-quiet-note]'))note.remove();
  const content=html.querySelector('.message-content')??html;
  for(const [key,note]of notesFor(message)){
   const section=html.ownerDocument.createElement('section');section.dataset.tpaQuietNote=key;section.className='tpa-quiet-note';
   if(note.html)section.insertAdjacentHTML('beforeend',note.html);else section.textContent=note.text;
   content.append(section);
  }
 }
 function register({Hooks}){
  // Register before ready so the first historical render and Utility's later
  // asynchronous button insertion both see the root presentation class.
  Hooks.on('renderChatMessageHTML',render);
  Hooks.on('createChatMessage',message=>changed(message));
  Hooks.on('updateChatMessage',message=>changed(message));
  Hooks.on('deleteChatMessage',message=>changed(message,true));
 }
 return {render,register};
}

export function registerQuietChat({Hooks}){
 // Module imports precede Foundry's replacement of the temporary game object.
 Hooks.once('init',()=>createQuietChat({game:globalThis.game}).register({Hooks}));
}
