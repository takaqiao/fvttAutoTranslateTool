import {hasSource} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {reactionEpoch} from './reaction-budget.mjs';
import {DESTRUCTIVE_BLOCK_SOURCE} from './shield-damage-adapter.mjs';

const worlds=new Set(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2']);
const INDESTRUCTIBLE='Compendium.pf2e.equipment-srd.Item.SUbYk6B1iPoGyyjh';
const values=c=>Array.from(c?.values?.()??c??[]);
export function createDestructiveBlock({game,fromUuid=globalThis.fromUuid,choose}={}){
 const planKey=Symbol('destructive-block-plan');let socket;
 const feat=actor=>values(actor?.items).find(i=>hasSource(i,DESTRUCTIVE_BLOCK_SOURCE));
 const supported=()=>worlds.has(game.world?.id)&&!game.modules?.get('pf2e-auto-action-tracker')?.active;
 const eligible=actor=>{const shield=actor?.heldShield,s=actor?.attributes?.shield;return supported()&&actor?.type==='character'&&feat(actor)&&shield&&shield.id===s?.itemId&&s.raised&&!s.broken&&!s.destroyed&&shield._source?.system?.hp?.value>0&&!hasSource(shield,INDESTRUCTIBLE)};
 function validate(actor,token,payload,user){
  if(!isActiveGM(game)||!user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('破坏性格挡需要当前主GM验证本次操作者。');
  if(token?.documentName!=='Token'||token.actor?.uuid!==actor.uuid||!eligible(actor)||feat(actor).id!==payload.featId||actor.heldShield.id!==payload.shieldId||actor.heldShield._source.system.hp.value!==payload.shieldHP||actor.attributes.shield.hardness!==payload.shieldHardness||reactionEpoch(actor,game)!==payload.epoch)throw Error('选择期间角色、盾牌或回合已经改变，本次伤害尚未结算。');
 }
 async function select(payload,user){
  const actor=await fromUuid(payload.actorUuid),token=await fromUuid(payload.tokenUuid);validate(actor,token,payload,user);
  const selected=await choose({actor,user,title:'盾牌格挡：是否使用破坏性格挡？',choices:[{value:'normal',label:'普通格挡'},{value:'destructive',label:'破坏性格挡（双倍防护，盾牌受双倍来袭伤害）'}]});
  validate(actor,token,payload,user);if(selected===null||selected===undefined||selected==='normal')return null;
  if(selected!=='destructive')throw Error('破坏性格挡选择无效。');
  return {nonce:globalThis.foundry?.utils?.randomID?.(24)??globalThis.crypto.randomUUID(),shieldId:payload.shieldId,featId:payload.featId,epoch:payload.epoch};
 }
 async function beforeDamage(actor,params){
  const damage=typeof params.damage==='number'?params.damage:params.damage?.total;
  if(!params.shieldBlockRequest||params.final||!Number.isFinite(damage)||damage<=0||!eligible(actor))return null;
  const token=params.token?.document??params.token;
  if(!actor.testUserPermission?.(game.user,'OWNER')||token?.actor?.uuid!==actor.uuid)throw Error('无权使用这个角色的破坏性格挡。');
  const payload={actorUuid:actor.uuid,tokenUuid:token.uuid,featId:feat(actor).id,shieldId:actor.heldShield.id,shieldHP:actor.heldShield._source.system.hp.value,shieldHardness:actor.attributes.shield.hardness,epoch:reactionEpoch(actor,game)};
  let plan;
  if(isActiveGM(game))plan=await select(payload,game.user);
  else{
   if(!socket||!game.users.activeGM)throw Error('破坏性格挡需要在线主GM。');
   const result=await socket.executeAsUser('destructive-block:choose',game.users.activeGM.id,payload);
   if(!result?.ok)throw Error(result?.error??'破坏性格挡规则选择失败。');plan=result.value;
  }
  if(!plan)return null;
  if(!eligible(actor)||reactionEpoch(actor,game)!==payload.epoch||actor.heldShield.id!==plan.shieldId)throw Error('选择结果到达时盾牌或回合已经改变，本次伤害尚未结算。');
  return {params:{...params,[planKey]:plan}};
 }
 function register({socket:api}={}){
  socket=api;socket?.register('destructive-block:choose',async function(payload){
   try{if(typeof payload?.actorUuid!=='string'||typeof payload.tokenUuid!=='string')throw Error('破坏性格挡来源无效。');return {ok:true,value:await select(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}
  });
 }
 return {beforeDamage,planFor:params=>params[planKey]??null,register};
}
