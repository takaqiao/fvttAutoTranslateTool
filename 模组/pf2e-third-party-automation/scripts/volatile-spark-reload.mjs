import {MODULE_ID} from './rules.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';
import {VOLATILE_SPARK_SOURCES as S,hasVolatileSpark,heldFirearm} from './volatile-spark.mjs';

const values=c=>Array.from(c?.values?.()??c??[]);
const author=m=>m?.author?.id??m?.user?.id??m?.user;
async function loadRanged(){
 const [{WeaponSystem},{Reload},{Updates}]=await Promise.all([
  import('/modules/pf2e-ranged-combat/scripts/weapons/system.js'),
  import('/modules/pf2e-ranged-combat/scripts/ammunition-system/actions/reload.js'),
  import('/modules/pf2e-ranged-combat/scripts/utils/updates.js'),
 ]);
 return {WeaponSystem,Reload,Updates};
}

/** Ranged Combat 8.0.7's public reload() uses the controlled token and does not
 * await its write. Use its actor-bound operation and await the same Updates. */
export function createVolatileSparkReload({game,fromUuid=globalThis.fromUuid,loadRanged:load=loadRanged}={}){
 const entered=new Set();let socket;
 async function context(payload,sender){
  if(!sender?.active||!sender.isGM||game.users.activeGM!==sender)throw Error('装填请求必须来自当前主 GM。');
  const message=game.messages.get(payload.messageId),receipt=message?.flags?.[MODULE_ID]?.spellCombinationUse;
  const item=await fromUuid(message?.flags?.pf2e?.origin?.uuid),actor=item?.actor,user=game.user;
  const validate=()=>{
   if(game.users.activeGM!==sender||!user?.active||game.user!==user||game.users.get(user.id)!==user||!actor?.testUserPermission(user,'OWNER')||author(message)!==user.id)throw Error('装填的原操作者或权限已改变。');
   if(game.messages.get(message.id)!==message||actor.items.get(item.id)!==item||getSourceId(item)!==S.reload||!hasVolatileSpark(actor)||!isActualUseMessage(message)||receipt!==message.flags[MODULE_ID].spellCombinationUse||receipt?.state!=='reloading'||receipt.userId!==user.id||receipt.weaponId!==payload.weaponId)throw Error('装填的原始 Use 或能力已改变。');
   const strike=values(actor.system.actions).find(s=>s.item?.id===payload.weaponId&&heldFirearm(s));
   if(!strike||!game.modules.get('pf2e-ranged-combat')?.active)throw Error('原生火器装填入口已不可用。');
   return strike;
  };
  validate();return {actor,message,validate};
 }
 async function execute(payload,sender){
  const {actor,message,validate}=await context(payload,sender);
  if(entered.has(message.id))throw Error('本次装填已经进入，不能重复执行。');
  const {WeaponSystem,Reload,Updates}=await load();const strike=validate();
  const cascade=await fromUuid(S.cascade);validate();
  const effectUuid=cascade?.system.selfEffect?.uuid,effect=effectUuid?await fromUuid(effectUuid):null;validate();
  if(!effect)throw Error('原生奥术奔涌 selfEffect 不可用。');
  const weapon=WeaponSystem.transformWeapon(actor,strike.item);
  if(!weapon)throw Error('Ranged Combat 无法识别所选火器。');
  if(weapon.isRepeating||weapon.reloadActions!==1)throw Error('填弹奔涌仅自动处理一次 Interact 完成的装填；此枪械请沿原生入口分步装填。');
  if(entered.has(message.id))throw Error('本次装填已经进入，不能重复执行。');
  entered.add(message.id);
  const updates=new Updates(actor),reloaded=await Reload.reload(weapon,updates,{fullyReload:false});validate();
  if(!reloaded)return {status:'cancelled'};
  await updates.commit();validate();
  if(!values(actor.items).some(item=>item.type==='effect'&&getSourceId(item)===effectUuid)){
   const data=effect.toObject();delete data._id;data._stats??={};data._stats.compendiumSource=effectUuid;
   await actor.createEmbeddedDocuments('Item',[data]);validate();
  }
  return {status:'done'};
 }
 async function run({actor,item,message,user,weaponId,assertLive}){
  assertLive();if(!isActiveGM(game)||item.actor!==actor)throw Error('装填活动的主 GM 已改变。');
  const payload={messageId:message.id,weaponId};
  if(user.id===game.user.id)return execute(payload,game.user);
  if(!socket)throw Error('缺少装填的原操作者连接。');
  const response=await socket.executeAsUser('volatile-spark:reload',user.id,payload);assertLive();
  if(!response?.ok)throw Error(response?.error??'装填结果未确认，不会自动重试。');return response.result;
 }
 function register({socket:api}={}){
  socket=api;socket?.register('volatile-spark:reload',async function(payload){
   try{return {ok:true,result:await execute(payload,game.users.get(this.socketdata.userId))};}
   catch(error){return {ok:false,error:error.message};}
  });
 }
 return {run,register};
}
