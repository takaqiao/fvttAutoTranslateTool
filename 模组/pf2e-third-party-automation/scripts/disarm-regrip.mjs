import {MODULE_ID,hasSource} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM,showNativeChoice} from './native-context.mjs';
import {getWeakenedGrasp} from './disarming-block.mjs';

export const INTERACT_SOURCE='Compendium.pf2e.actionspf2e.Item.pvQ5rY2zrtPI614F';
const values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]??{},author=c=>c?.author?.id??c?.user?.id??c?.user;
const held=w=>w?.type==='weapon'&&w.system?.equipped?.carryType==='held'&&w.system.equipped.handsHeld>0;
const oneAction=i=>i?.type==='action'&&i.system?.actionType?.value==='action'&&i.system.actions?.value===1&&i.system.traits?.value?.includes('manipulate');
const exactItem=i=>hasSource(i,INTERACT_SOURCE)&&oneAction(i);
const random=()=>globalThis.foundry?.utils?.randomID?.(24)??globalThis.crypto.randomUUID();
function affected(actor){
 return values(actor?.items).filter(held).flatMap(weapon=>{
  const grasp=getWeakenedGrasp(actor,weapon.uuid),checkId=own(grasp).disarmingBlock?.checkId;
  return typeof checkId==='string'&&checkId?[{weaponUuid:weapon.uuid,graspCheckId:checkId}]:[];
 });
}
const sameGrasp=(actor,candidate)=>affected(actor).some(c=>c.weaponUuid===candidate.weaponUuid&&c.graspCheckId===candidate.graspCheckId);
// PF2e 8.5 SimpleActionVariant.glyph calls getActionGlyph(1), which emits "1".
// Legacy macro glyph letters are not evidence for this native-action route.
const nativeCard=card=>/<h4\b[^>]*class=["'][^"']*\baction\b[^"']*["'][^>]*>[\s\S]*?<span\b[^>]*class=["'][^"']*\baction-glyph\b[^"']*["'][^>]*>\s*1\s*<\/span>[\s\S]*?<\/h4>/.test(card.flavor??'')&&/data-slug=["']manipulate["']/.test(card.flavor??'');

/** One real Interact card is the payment and recovery boundary. A snapshot of
 * the affected check IDs prevents replay from clearing a later Disarm result.
 */
export function createDisarmRegrip({game,fromUuid=globalThis.fromUuid,processor,show=showNativeChoice,chooseOwner,publish=data=>globalThis.ChatMessage.create(data),onError=console.error}={}){
 const queue=new SerialActions(),pending=new Map();let socket,installation;
 const report=error=>{try{onError(error)}catch{/* A report must not strand other paid actions. */}};
 const gm=()=>{if(!isActiveGM(game))throw Error('改握必须由当前主GM结算。')};
 const owner=(actor,user)=>{if(!actor||!user||!actor.testUserPermission?.(user,'OWNER'))throw Error('没有本次改握角色的所有者权限。')};
 const input=(actor,mode,item=null)=>({nonce:random(),mode,source:INTERACT_SOURCE,actorUuid:actor.uuid,userId:game.user.id,candidates:affected(actor),...(item?{itemUuid:item.uuid}:{})});
 const resolveAction=item=>exactItem(item)&&affected(item.actor).length?'disarm-regrip:interact':null;
 const captureUsage=item=>resolveAction(item)?{disarmRegripInput:input(item.actor,'item',item)}:null;
 const choices=(actor,candidates)=>[...candidates.map(c=>({value:c.weaponUuid,label:`调整握持：${values(actor.items).find(i=>i.uuid===c.weaponUuid)?.name??'武器'}`})),{value:'other',label:'进行其他交互'}];
 const currentInput=card=>own(card).disarmRegripInput;
 async function validate(card,user){
  gm();if(!card?.id||game.messages.get(card.id)!==card||author(card)!==user?.id||card.rolls?.length||card.isRoll)throw Error('改握需要本次操作者的真实Interact动作卡。');
  const data=currentInput(card);
  if(!data||data.source!==INTERACT_SOURCE||!/^[A-Za-z0-9-]{8,80}$/.test(data.nonce??'')||data.userId!==user.id||!['native-action','item'].includes(data.mode)||!Array.isArray(data.candidates)||!data.candidates.length||data.candidates.length>100||data.candidates.some(c=>typeof c.weaponUuid!=='string'||typeof c.graspCheckId!=='string'||!c.graspCheckId)||new Set(data.candidates.map(c=>c.weaponUuid)).size!==data.candidates.length)throw Error('改握动作的来源和握持快照无效。');
  const inputKey=JSON.stringify(data),actor=await fromUuid(data.actorUuid);gm();owner(actor,user);
  if(card.speaker?.actor!==actor.id||card.actor?.uuid&&card.actor.uuid!==actor.uuid)throw Error('Interact动作卡与改握角色不一致。');
  if(data.mode==='item'){
   const item=await fromUuid(data.itemUuid);gm();const origin=card.flags?.pf2e?.origin;
   if(!exactItem(item)||item.actor?.uuid!==actor.uuid||origin?.uuid!==item.uuid||origin.actor&&origin.actor!==actor.uuid||origin.type!=='action')throw Error('Interact物品卡不是确切的原生来源。');
  }else{
   const source=await fromUuid(INTERACT_SOURCE);gm();if(!oneAction(source)||!nativeCard(card))throw Error('改握没有真实的一动作Interact卡。');
  }
  if(game.messages.get(card.id)!==card||author(card)!==user.id||JSON.stringify(currentInput(card))!==inputKey)throw Error('验证期间Interact卡已改变。');
  return {actor,data:structuredClone(data)};
 }
 async function save(card,state){gm();await card.update({[`flags.${MODULE_ID}.disarmRegrip`]:state});gm();}
 async function processCard(card,user){
  gm();pending.set(card?.id,currentInput(card)?.actorUuid);
  return queue.run(card?.id,async()=>{
   let {actor,data}=await validate(card,user);gm();let state=own(card).disarmRegrip;
   if(state&&([state.nonce,state.actorUuid,state.userId].some((value,index)=>value!==[data.nonce,actor.uuid,user.id][index])||state.inputKey!==JSON.stringify(data)))throw Error('Interact认领与动作卡来源不一致。');
   if(state?.status==='done'){pending.delete(card.id);return state.result;}
   if(!state){state={nonce:data.nonce,status:'offered',actorUuid:actor.uuid,userId:user.id,inputKey:JSON.stringify(data)};await save(card,state)}
   if(state.status==='offered'){
    const candidates=data.candidates.filter(c=>sameGrasp(actor,c));
    let selected=data.selected;
    if(!candidates.length)selected='superseded';
    else if(selected===undefined){
     const request={actor,user,title:'交互：是否调整武器握持？',choices:choices(actor,candidates)};
     // The shared remote chooser is character-only; a GM's own NPC action is local.
     if(user.id===game.user.id&&actor.type!=='character')selected=await show(request);
     else if(chooseOwner)selected=await chooseOwner(request);
     else if(user.id===game.user.id)selected=await show(request);
     else throw Error('缺少改握操作者的原生选择通讯。');
    }
    ({actor,data}=await validate(card,user));gm();
    if(state.inputKey!==JSON.stringify(data))throw Error('选择期间Interact握持快照已改变。');
    if(selected==null||selected==='other'||selected==='superseded'){
     const result=selected==null?'cancelled':selected;await save(card,{...state,status:'done',result});pending.delete(card.id);return result;
    }
    const candidate=data.candidates.find(c=>c.weaponUuid===selected);
    if(!candidate||!candidates.some(c=>c.weaponUuid===selected))throw Error('本次Interact选择的改握武器无效。');
    if(!sameGrasp(actor,candidate)){await save(card,{...state,status:'done',result:'superseded'});pending.delete(card.id);return 'superseded'}
    state={...state,status:'claimed',...candidate};await save(card,state);
   }
   if(state.status!=='claimed'||!data.candidates.some(c=>c.weaponUuid===state.weaponUuid&&c.graspCheckId===state.graspCheckId))throw Error('改握认领缺少确切的原始缴械检定。');
   ({actor,data}=await validate(card,user));gm();
   if(state.inputKey!==JSON.stringify(data))throw Error('结算期间Interact握持快照已改变。');
   const result=await processor.clearRegripFromCard({actor,weaponUuid:state.weaponUuid,user,graspCheckId:state.graspCheckId,cardId:card.id});gm();
   await save(card,{...state,status:'done',result:result?.status??'cleared'});pending.delete(card.id);return result?.status??'cleared';
  });
 }
 async function processRemote(card){
  if(isActiveGM(game))return processCard(card,game.user);
  if(!socket||!game.users.activeGM?.id)throw Error('自动改握需要在线主GM。');
  const result=await socket.executeAsUser('disarm-regrip:process',game.users.activeGM.id,{cardId:card.id});
  if(!result?.ok)throw Error(result?.error??'Interact改握验证失败。');return result.value;
 }
 const selectedActors=params=>{
  if(params.actors)return Array.isArray(params.actors)?params.actors:[params.actors];
  const selected=[...new Set(values(game.user.getActiveTokens?.()).map(t=>t.actor).filter(a=>a&&!a.isOfType?.('loot','party')))];
  return selected.length?selected:game.user.character?[game.user.character]:[];
 };
 async function nativeUse(native,variant,params={}){
  const actors=selectedActors(params);
  if(variant.cost!==1||!variant.traits.includes('manipulate')||!actors.some(a=>affected(a).length))return native.call(variant,params);
  const user=game.user,leader=game.users.activeGM?.id,results=[];
  const localOwner=actor=>{owner(actor,user);if(game.user!==user||user.id===leader&&game.users.activeGM?.id!==leader)throw Error('Interact操作者或主GM已改变。')};
  for(const actor of actors){
   localOwner(actor);const data=input(actor,'native-action');
   if(!data.candidates.length){results.push(...await native.call(variant,{...params,actors:[actor]}));continue;}
   const selected=await show({actor,user,title:'交互：是否调整武器握持？',choices:choices(actor,data.candidates)});localOwner(actor);
   if(selected==null)continue;
   if(selected==='other'){results.push(...await native.call(variant,{...params,actors:[actor]}));continue;}
   const candidate=data.candidates.find(c=>c.weaponUuid===selected);if(!candidate||!sameGrasp(actor,candidate))throw Error('选择的武器握持已经改变，本次Interact尚未发出。');
   const rolled=await native.call(variant,{...params,actors:[actor],message:{...params.message,create:false}});localOwner(actor);
   for(const result of rolled??[]){
    if(result.actor?.uuid!==actor.uuid||!result.message)throw Error('原生Interact未返回本次角色的动作卡。');
    result.message.updateSource({[`flags.${MODULE_ID}.disarmRegripInput`]:{...data,selected}});
    if(params.message?.create===false){results.push(result);continue;}
    const source=result.message.toObject();delete source._id;const card=await publish(source);if(!card?.id)throw Error('Interact动作卡未能发布。');
    await processRemote(card);results.push({...result,message:card});
   }
  }
  return results;
 }
 async function executeUsage({actor,item,message,user}){
  gm();if(!exactItem(item)||item.actor?.uuid!==actor?.uuid)throw Error('没有确切的Interact技能来源。');const result=await processCard(message,user);
  return {cleared:'已调整握持，解除这把武器的握持削弱。',other:'本次进行其他交互。',cancelled:'未选择改握武器。',superseded:'本次动作对应的握持削弱已经结束。'}[result]??result;
 }
 async function maintain(actor){
  if(!isActiveGM(game)||!actor)return;
  for(const [id,actorUuid]of [...pending]){
   if(actorUuid!==actor.uuid)continue;gm();const card=game.messages.get(id);
   if(!card){pending.delete(id);continue;}
   const user=game.users.get(author(card));
   // Keep the original card and owner; account changes never authorize a substitute.
   if(!user||!actor.testUserPermission?.(user,'OWNER'))continue;
   try{await processCard(card,user)}catch(error){if(!isActiveGM(game))return;report(error)}
  }
 }
 async function recoverPending(){
  if(!isActiveGM(game))return;
  for(const uuid of new Set(pending.values())){
   if(!isActiveGM(game))return;if(typeof uuid!=='string')continue;
   try{
    const actor=await fromUuid(uuid);gm();
    if(actor)await maintain(actor);else for(const [id,target]of [...pending])if(target===uuid)pending.delete(id);
   }catch(error){if(!isActiveGM(game))return;report(error)}
  }
 }
 function register({Hooks,socket:api}={}){
  if(installation)return unregister;socket=api;
  const track=card=>{const data=currentInput(card);if(data?.actorUuid&&own(card).disarmRegrip?.status!=='done')pending.set(card.id,data.actorUuid);else pending.delete(card.id)};
  for(const card of values(game.messages))track(card);
  const onCreate=(card,_options,creator)=>{
   track(card);if(currentInput(card)?.mode==='native-action'&&creator===author(card)&&isActiveGM(game))void processCard(card,game.users.get(creator)).catch(onError);
  };
  const ids=[['createChatMessage',Hooks.on('createChatMessage',onCreate)],['updateChatMessage',Hooks.on('updateChatMessage',track)],['updateUser',Hooks.on('updateUser',()=>recoverPending().catch(onError))]];
  socket?.register('disarm-regrip:process',async function(payload){try{return {ok:true,value:await processCard(game.messages.get(payload?.cardId),game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  const action=game.pf2e.actions.get('interact'),descriptor=Object.getOwnPropertyDescriptor(action,'toActionVariant'),original=action.toActionVariant;
  const wrapped=function(...args){const variant=original.apply(this,args),native=variant.use;variant.use=function(params={}){return nativeUse(native,this,params)};return variant};
  Object.defineProperty(action,'toActionVariant',{configurable:true,writable:true,value:wrapped});installation={action,descriptor,original,wrapped,Hooks,ids};void recoverPending().catch(onError);return unregister;
 }
 function unregister(){
  if(!installation)return;const {action,descriptor,wrapped,Hooks,ids}=installation;
  if(action.toActionVariant===wrapped){if(descriptor)Object.defineProperty(action,'toActionVariant',descriptor);else delete action.toActionVariant}
  for(const [name,id]of ids)Hooks.off(name,id);installation=null;
 }
 return {resolveAction,captureUsage,executeUsage,processCard,maintain,register,unregister};
}
