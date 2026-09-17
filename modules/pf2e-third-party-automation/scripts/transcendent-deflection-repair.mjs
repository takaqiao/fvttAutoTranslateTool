import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM} from './native-context.mjs';
import {deflectionBroken} from './transcendent-deflection-rules.mjs';
const own=m=>m?.flags?.[MODULE_ID]??{},author=m=>m?.author?.id??m?.user?.id??m?.user,values=c=>Array.from(c?.values?.()??c??[]);
const random=()=>globalThis.foundry?.utils?.randomID?.(24)??crypto.randomUUID();
// Foundry HTMLField is sanitized on the server after client preCreate: our
// hidden/disabled booleans can become absent/disabled="". Normalize only these
// presentation attributes on the exact restore button, retaining the entire
// remaining flavor including native item UUID and restore amount.
const repairFlavor=flavor=>(flavor??'').replace(/<button\b[^>]*\bdata-repair=["']restore["'][^>]*>/g,tag=>tag.replace(/\s(?:hidden|disabled)(?:=(?:"[^"]*"|'[^']*'|[^\s>]+))?(?=\s|>)/g,''));
const evidence=card=>JSON.stringify({actor:card.actor?.uuid,author:author(card),speaker:card.speaker,flavor:repairFlavor(card.flavor),context:card.flags?.pf2e?.context,rolls:card.rolls?.map(r=>r.toJSON())});
const hash=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),n=>n.toString(16).padStart(2,'0')).join('');
const attr=(tag,name)=>new RegExp(`\\b${name}=["']([^"']*)["']`).exec(tag)?.[1];
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const physical=item=>item?.isOfType?.('physical')===true;
// SelectItemDialog is not exported by PF2e 8.5. Preserve its ordinary item
// choice, including owned physical-item drops, before the untouched Repair.
async function selectRepairItem({items,fromUuid}){
 const choices=new Map(items.map(i=>[i.uuid,i]));let selected=items[0]??null,dropTask;
 const result=await globalThis.foundry.applications.api.DialogV2.wait({
  window:{title:'修理：选择物品'},content:`<label>修理物品<select name="repair-item">${items.map(i=>`<option value="${escape(i.uuid)}">${escape(i.actor?.name??'')}：${escape(i.name)}</option>`).join('')}</select></label><p data-repair-drop>也可以将拥有权限的实体物品拖到此处。</p>`,
  buttons:[{action:'repair',label:'修理',callback:async()=>{await dropTask;return selected}},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false,
  render:(_event,dialog)=>{
   const element=dialog.element,select=element.querySelector('[name="repair-item"]');
   select?.addEventListener('change',()=>{selected=choices.get(select.value)??null});
   element.addEventListener('dragover',event=>event.preventDefault());
   element.addEventListener('drop',event=>{
    event.preventDefault();selected=null;dropTask=(async()=>{
     let data;try{data=JSON.parse(event.dataTransfer?.getData('text/plain')??'')}catch{return}
     const item=data?.type==='Item'&&typeof data.uuid==='string'?await fromUuid(data.uuid):null;
     if(!physical(item)||!item.isEmbedded||!item.isOwner)return;
     choices.set(item.uuid,item);selected=item;
     if(select){if(!Array.from(select.options).some(o=>o.value===item.uuid)){const option=select.ownerDocument.createElement('option');option.value=item.uuid;option.textContent=`${item.actor?.name??''}：${item.name}`;select.append(option)}select.value=item.uuid;}
    })();dropTask.catch(()=>{});
   });
  },
 });
 // DialogV2 substitutes the button action when its callback returns null.
 return result&&typeof result==='object'?result:null;
}
function repairTarget(message){
 const html=message.flavor??'',cards=[...html.matchAll(/<div\b[^>]*\bdata-item-uuid=["'][^"']+["'][^>]*>/g)];
 if(cards.length!==1||!/(?:^|\s)pf2e(?:\s|$)/.test(attr(cards[0][0],'class')??'')||!/(?:^|\s)chat-card(?:\s|$)/.test(attr(cards[0][0],'class')??''))return null;
 const buttons=[...html.matchAll(/<button\b[^>]*\bdata-repair=["']restore["'][^>]*>/g)];
 const amount=buttons.length===1?Number(attr(buttons[0][0],'data-repair-value')):null;
 return {itemUuid:attr(cards[0][0],'data-item-uuid'),amount};
}
function nativeCheck(card){const c=card?.flags?.pf2e?.context,r=card?.rolls?.[0],raw=r?.toJSON?.();return !!(card.isCheckRoll&&card.rolls.length===1&&c?.type==='skill-check'&&c.options?.includes('action:repair')&&!c.isReroll&&r?._evaluated===true&&Number.isFinite(r.total)&&raw?.evaluated===true&&raw.total===r.total)}

/** Reuse the ordinary PF2e Repair macro and its final evaluated card. The source
 * scope is captured before native use; success is never inferred from HP/grip.
 * Native Repair's exact restore amount replaces its manual restore button for
 * this marked broken item. Other Repair cards keep the native behavior. */
export function createDeflectionRepair({game,fromUuid=globalThis.fromUuid,onError=console.error,selectItem=selectRepairItem}={}){
 const scopes=new Map(),queue=new SerialActions();let installation,socket,selecting=false;
 const gm=()=>{if(!isActiveGM(game))throw Error('修理主GM已经改变。')};
 const owner=(actor,user)=>!!user&&actor?.testUserPermission?.(user,'OWNER');
 const report=e=>{try{onError(e)}catch{/* Never repeat a paid Repair because error reporting failed. */}};
 async function scopeProof(cardId,user){
  const matches=[...scopes.values()].filter(s=>s.cardId===cardId&&s.userId===user?.id);
  if(matches.length!==1)throw Error('没有这张卡对应的真实原生修理调用。');
  const s=matches[0];return {id:s.id,itemUuid:s.itemUuid,actorUuid:s.actorUuid,brokenNonce:s.brokenNonce,userId:s.userId,cardId:s.cardId,rank:s.rank,evidence:await hash(s.evidence)};
 }
 async function nativeUse(native,self,params={}){
  let item=params.item??(params.uuid?await fromUuid(params.uuid):null);
  const actors=params.actors?(Array.isArray(params.actors)?params.actors:[params.actors]):[...new Set(values(globalThis.canvas?.tokens?.controlled??game.user.getActiveTokens?.()).map(t=>t.actor).filter(Boolean))];
  if(!actors.length&&game.user.character)actors.push(game.user.character);
  if(!item&&!params.uuid){
   const accessible=[...new Set([...actors,...values(game.actors)])].filter(a=>owner(a,game.user)),items=accessible.flatMap(a=>values(a.items));
   if(!items.some(deflectionBroken))return native.call(self,params);
   if(selecting)throw Error('修理物品选择尚未结束，不会重复开始修理。');
   const userId=game.user.id;selecting=true;
   try{item=await selectItem({actors,items:items.filter(i=>physical(i)),fromUuid})}finally{selecting=false}
   if(!item)return null;
   if(game.user.id!==userId||item.actor?.items?.get(item.id)!==item||await fromUuid(item.uuid)!==item||!owner(item.actor,game.user))throw Error('选择的修理物品或权限已经改变。');
   params={...params,item,actors};
  }
  const broken=deflectionBroken(item);if(!broken)return native.call(self,params);
  if(!actors.length||actors.some(a=>!owner(a,game.user)))throw Error('需要实际修理角色的所有者。');
  if(actors.some(a=>[...scopes.values()].some(s=>s.actorUuid===a.uuid&&!s.cardId)))throw Error('该角色的原生修理尚未返回结果，不会重复掷骰。');
  for(const actor of actors){const id=random(),scope={id,actorUuid:actor.uuid,itemUuid:item.uuid,brokenNonce:broken.nonce,userId:game.user.id,rank:actor.skills?.crafting?.rank??0};scope.timer=setTimeout(()=>scopes.delete(id),600000);scope.timer.unref?.();scopes.set(id,scope);}
  return native.call(self,{...params,item,actors});
 }
 function preCreate(card,_data,_options,userId){
  if(userId!==game.user.id||author(card)!==game.user.id||!nativeCheck(card))return;
  const target=repairTarget(card),actorUuid=card.actor?.uuid,failed=card.flags.pf2e.context.outcome==='failure',candidates=[...scopes.values()].filter(s=>!s.draft&&!s.cardId&&s.actorUuid===actorUuid&&s.userId===userId&&(s.itemUuid===target?.itemUuid||failed&&!target));
  if(candidates.length!==1)return;
  const scope=candidates[0];scope.cardId=card.id; // Native pre-create may not yet assign an ID.
  scope.draft=card;
  // Strip the native action before the first render. ChatCards.listen only
  // binds [data-action], so even a delayed GM reply leaves no restore race.
  const flavor=(card.flavor??'').replace(/<button\b[^>]*\bdata-repair=["']restore["'][^>]*>/g,tag=>tag.replace(/\sdata-action=["'][^"']*["']/g,'').replace(/>$/,' hidden disabled>'));
  card.updateSource({flavor,[`flags.${MODULE_ID}.deflectionRepairInput`]:{id:scope.id,actorUuid:scope.actorUuid,itemUuid:scope.itemUuid,brokenNonce:scope.brokenNonce,userId,rank:scope.rank}});
  scope.evidence=evidence(card);
 }
 async function processCard(card,user){
  gm();return queue.run(card?.id,async()=>{
   gm();const input=own(card).deflectionRepairInput;
   if(!card?.id||game.messages.get(card.id)!==card||author(card)!==user?.id||!nativeCheck(card)||input?.userId!==user.id)throw Error('需要实际拥有者的原生修理卡。');
   const actor=await fromUuid(input.actorUuid),weapon=await fromUuid(input.itemUuid);gm();
   if(!owner(actor,user)||card.actor?.uuid!==actor.uuid||card.speaker?.actor!==actor.id||weapon?.type!=='weapon')throw Error('修理卡来源或角色权限不符。');
   const previous=own(card).deflectionRepair;
   if(previous){if(previous.id!==input.id||previous.cardId!==card.id||previous.itemUuid!==weapon.uuid||previous.brokenNonce!==input.brokenNonce)throw Error('修理回执已改变。');if(previous.state==='done')return previous.result;throw Error('本次修理结果不确定，不会重复恢复武器。');}
   let source;if(user.id===game.user.id)source=await scopeProof(card.id,user);else{const result=await socket?.executeAsUser('deflection-repair:scope',user.id,{cardId:card.id});gm();if(result?.ok)source=result.value;}
   if(!source||source.id!==input.id||source.itemUuid!==weapon.uuid||source.actorUuid!==actor.uuid||source.brokenNonce!==input.brokenNonce||source.userId!==user.id||source.rank!==input.rank)throw Error('修理卡没有确切的本次原生调用证明。');
   const captured=evidence(card);if(source.evidence!==await hash(captured)||captured!==evidence(card))throw Error('原生修理最终结果已经改变。');gm();
   const target=repairTarget(card),outcome=card.flags.pf2e.context.outcome,success=['success','criticalSuccess'].includes(outcome),expected=(outcome==='criticalSuccess'?10:5)*(1+input.rank);
   if(success&&(target?.itemUuid!==weapon.uuid||target.amount!==expected||!Number.isInteger(expected)||expected<=0))throw Error('原生修理回复数值或目标不一致。');
   const base={id:input.id,cardId:card.id,itemUuid:weapon.uuid,brokenNonce:input.brokenNonce};
   const finish=async result=>{gm();await card.update({[`flags.${MODULE_ID}.deflectionRepair`]:{...base,state:'done',result}});gm();return result};
   if(!success)return finish('failure');
   if(deflectionBroken(weapon)?.nonce!==input.brokenNonce)return finish('superseded');
   await card.update({[`flags.${MODULE_ID}.deflectionRepair`]:{...base,state:'applying'}});gm();
   if(deflectionBroken(weapon)?.nonce!==input.brokenNonce)return finish('superseded');
   const hp=weapon.system.hp,after=hp.max>0?Math.min(hp.max,hp.value+target.amount):hp.value,cleared=hp.max===0||after>Math.floor(hp.max/2);
   await weapon.update({'system.hp.value':after,...(cleared?{[`flags.${MODULE_ID}.transcendentDeflection.broken`]:null}:{})});gm();
   return finish(cleared?'repaired':'still-broken');
  });
 }
 async function created(card,_options,userId){
  const input=own(card).deflectionRepairInput;if(!input||userId!==game.user.id||author(card)!==userId)return;
  const scope=scopes.get(input.id);if(!scope||!scope.draft||scope.cardId&&scope.cardId!==card.id||scope.actorUuid!==input.actorUuid||scope.itemUuid!==input.itemUuid||scope.brokenNonce!==input.brokenNonce||scope.userId!==userId)return;
  scope.cardId=card.id;delete scope.draft;
  try{if(isActiveGM(game))return await processCard(card,game.user);const result=await socket?.executeAsUser('deflection-repair:process',game.users.activeGM?.id,{cardId:card.id});if(!result?.ok)throw Error(result?.error??'原生修理回执未能结算。');return result.value}catch(error){report(error)}
 }
 function register({Hooks,socket:api}={}){
  if(installation)return unregister;socket=api;const actions=game.pf2e.actions,native=actions.repair;
  if(typeof native!=='function')throw Error('缺少原生Repair入口。');
  const wrapped=function(params){return nativeUse(native,this,params)};actions.repair=wrapped;
  const ids=[['preCreateChatMessage',Hooks.on('preCreateChatMessage',preCreate)],['createChatMessage',Hooks.on('createChatMessage',created)],['renderChatMessageHTML',Hooks.on('renderChatMessageHTML',(card,html)=>{if(own(card).deflectionRepairInput||own(card).deflectionRepair?.state)for(const b of (html?.[0]??html)?.querySelectorAll?.('[data-repair="restore"]')??[]){b.hidden=true;b.disabled=true;b.removeAttribute?.('data-action');b.addEventListener?.('click',event=>{event.preventDefault();event.stopImmediatePropagation()},true)}})]];
  socket?.register('deflection-repair:scope',async function(payload){try{if(this.socketdata.userId!==game.users.activeGM?.id)throw Error('只有当前主GM可验证修理来源。');return {ok:true,value:await scopeProof(payload?.cardId,game.user)}}catch(error){return {ok:false,error:error.message}}});
  socket?.register('deflection-repair:process',async function(payload){try{return {ok:true,value:await processCard(game.messages.get(payload?.cardId),game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  installation={actions,native,wrapped,Hooks,ids};return unregister;
 }
 function unregister(){if(installation){const {actions,native,wrapped,Hooks,ids}=installation;if(actions.repair===wrapped)actions.repair=native;for(const[name,id]of ids)Hooks.off(name,id)}installation=null;for(const scope of scopes.values())clearTimeout(scope.timer);scopes.clear()}
 return {register,unregister,processCard};
}
