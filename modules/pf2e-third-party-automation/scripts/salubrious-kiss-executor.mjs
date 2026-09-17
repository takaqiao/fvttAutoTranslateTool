import {assertSalubriousCardPrivacy,salubriousPrivacyData,assertSalubriousReceiptPrivacy} from './salubrious-privacy.mjs';
import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {treatmentOutcome,kissState} from './salubrious-kiss-rules.mjs';
import {identityKeys,sameClaim,claimOf,marker,outcomes,contextFor,assertSource,assertPatient,assertClaimPrivacy,fail} from './salubrious-kiss-context.mjs';
const proof=(claim,kind)=>({kind,...Object.fromEntries(identityKeys.map(k=>[k,claim[k]])),...claim.privacy?{privacy:structuredClone(claim.privacy)}:{}});
const author=m=>m.author?.id??m.author??m.user?.id??m.user;
const dataRoll=r=>typeof r==='string'?JSON.parse(r):r;
const evaluated=r=>Number.isFinite(r?.total)&&(r._evaluated===true||r.evaluated===true);
const checkMarker=claim=>marker('check',claim);
const skip='skip-handling-message';

export function validateSalubriousCard({game,message,claim,damage=false,stored=true,decorated=true}){
 const pf=message?.flags?.pf2e,c=pf?.context,r=dataRoll(message?.rolls?.[0]),degree=outcomes.indexOf(c?.outcome),p=message?.flags?.[MODULE_ID]?.salubriousKiss;
 if(stored&&(game.messages.get(message?.id)!==message||!(damage?message.isDamageRoll:message.isCheckRoll)))throw fail('没有确切原生消息文档');
 if(author(message)!==claim.userId||message.speaker?.actor!==claim.actorUuid.split('.').at(-1)||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==claim.tokenUuid||pf?.origin?.actor!==claim.actorUuid||pf.origin.uuid!==claim.itemUuid||c?.origin?.actor!==claim.actorUuid||c.origin.token!==claim.tokenUuid||c.type!=='skill-check'||c.action!=='treat-wounds'||c.dc?.value!==claim.dc||!c.domains?.includes('occultism')||!c.options?.includes(checkMarker(claim))||degree<0||message.rolls.length!==1||!evaluated(r))throw fail('原生医疗卡的拥有者、来源、技能、成功度或公开模式不符');
 assertSalubriousCardPrivacy({message,claim});
 if(!damage&&r.options?.degreeOfSuccess!==degree)throw fail('原生最终成功度不一致');
 if(stored&&(r._evaluated!==true||dataRoll(r.toJSON?.())?.evaluated!==true))throw fail('真实原生骰子尚未完成');
 if(decorated&&(!sameClaim(p,claim)||p.kind!==(damage?'damage':'check')||!c.options.includes(skip)||pf.suppressDamageButtons!==true))throw fail('本次医疗卡缺少精确来源和唯一结算保护');
 if(damage&&(!c.options.includes(marker('damage',claim))||!pf.origin.messageId))throw fail('医疗伤害卡没有原检定关联');
 return degree;
}

/** Use the prepared Occultism Statistic and native DamageRoll. Numeric-DC
 * Statistic legitimately keeps context.target null; exact patient is separately
 * persisted in this module's proof, never invented as a native target context. */
export async function rollSalubriousTreatment({game,actor,item,token,target,claim,checkScope,createMessage=data=>globalThis.ChatMessage.create(data),DamageRoll}={}){
 assertSource({game,actor,item,token,user:game.user,privacy:claim.privacy});assertPatient({game,actor,token,target,user:game.users.get(claim.userId)});
 assertClaimPrivacy({game,claim,token,item,target,user:game.user});
 if(game.user.id!==claim.userId||actor.uuid!==claim.actorUuid||item.uuid!==claim.itemUuid||token.uuid!==claim.tokenUuid||target.uuid!==claim.targetUuid||target.actor.uuid!==claim.targetActorUuid)throw fail('本客户端不是原医疗来源和拥有者');
 const native=actor.skills?.occultism?.check;if(!native?.roll||!checkScope?.run)throw fail('原生神秘Statistic或私有检定scope缺失');
 let draft,fingerprint,callbackError;
 const callback=async(roll,outcome,message)=>{
  try{
   if(typeof message?.toObject!=='function'||message.id&&game.messages.get(message.id)===message)throw fail('回调不是未发布的原生草稿');
   const data=message.toObject();delete data._id;const degree=validateSalubriousCard({game,message:data,claim,stored:false,decorated:false});
   if(outcomes[degree]!==outcome||roll.total!==dataRoll(data.rolls[0]).total)throw fail('原生回调结果不一致');
   const key=JSON.stringify(data);if(fingerprint&&fingerprint!==key)throw fail('同一次检定出现多个不同最终草稿');fingerprint=key;draft=data;
  }catch(error){callbackError=error;throw error}
 };
 const roll=await checkScope.run({actor,item,token,claim},()=>native.roll({token,item,action:'treat-wounds',dc:{value:claim.dc,visible:true},traits:['exploration','healing','manipulate','vitality'],skipDialog:true,messageMode:claim.privacy?.mode??'public',createMessage:false,
  extraRollOptions:['action:treat-wounds',checkMarker(claim),'item:trait:healing','item:trait:vitality'],callback}));
 if(callbackError)throw callbackError;if(!roll||!draft)throw fail('原生医疗取消或没有最终草稿');
 assertSource({game,actor,item,token,user:game.user,privacy:claim.privacy});assertPatient({game,actor,token,target,user:game.users.get(claim.userId)});
 assertClaimPrivacy({game,claim,token,item,target,user:game.user});
 draft.flags.pf2e.context.options=[...new Set([...draft.flags.pf2e.context.options,skip])];draft.flags.pf2e.suppressDamageButtons=true;
 draft.flags[MODULE_ID]={...draft.flags[MODULE_ID],salubriousKiss:proof(claim,'check')};
 const check=await createMessage(draft),degree=validateSalubriousCard({game,message:check,claim});
 const outcome=treatmentOutcome({degree,tier:claim.tier});if(!outcome.formula)return {check,damage:null,degree};
 DamageRoll??=game.pf2e?.DamageRoll??globalThis.CONFIG?.Dice?.rolls?.find(cls=>cls.name==='DamageRoll');if(typeof DamageRoll!=='function')throw fail('原生DamageRoll类不可用');
 const damageRoll=await new DamageRoll(outcome.formula).evaluate({allowInteractive:(claim.privacy?.mode??'public')!=='blind'});assertSource({game,actor,item,token,user:game.user,privacy:claim.privacy});
 assertClaimPrivacy({game,claim,token,item,target,user:game.user});
 const flags=structuredClone(check.flags);flags.pf2e.origin={...flags.pf2e.origin,messageId:check.id};flags.pf2e.context.options.push(marker('damage',claim));flags[MODULE_ID].salubriousKiss=proof(claim,'damage');
 const damage=await createMessage({author:claim.userId,speaker:structuredClone(check.speaker),flavor:outcome.kind==='healing'?'仙露三吻：医疗恢复':'仙露三吻：医疗大失败',...salubriousPrivacyData(claim),flags,rolls:[damageRoll.toJSON()]});
 validateSalubriousCard({game,message:damage,claim,damage:true});return {check,damage,degree};
}

export function createSalubriousExecutor({game,fromUuid=globalThis.fromUuid,Hooks=globalThis.Hooks,checkScope,createMessage,DamageRoll,authorizeDamage,timeoutMs=60000}={}){
 let socket;const tasks=new Map(),writes=new SerialActions();
 const requireGM=user=>{if(!user?.active||!user.isGM||game.users.get(user.id)!==user||game.users.activeGM!==user)throw fail('请求者不是当前主GM');};
 function waitFor(read,event,match){const first=read();if(first)return Promise.resolve(first);if(!Hooks?.on)throw fail('来源文档尚未同步');return new Promise((resolve,reject)=>{let id,timer,done=false;const finish=(error,value)=>{if(done)return;done=true;clearTimeout(timer);Hooks.off(event,id);error?reject(error):resolve(value)};const check=()=>{try{const v=read();if(v)finish(null,v)}catch(e){finish(e)}};id=Hooks.on(event,(...args)=>{if(match(...args))check()});timer=setTimeout(()=>finish(fail('准确文档同步超时')),timeoutMs);check()})}
 const card=id=>{if(typeof id!=='string'||!id)throw fail('没有准确原生卡ID');return waitFor(()=>game.messages.get(id),'createChatMessage',m=>m.id===id)};
 async function deadline(promise){let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(fail('原拥有者执行结果超时，不能重掷')),timeoutMs)})])}finally{clearTimeout(timer)}}
 async function ownerRoll(payload,requester){
  requireGM(requester);const actor=await fromUuid(payload?.actorUuid);if(!actor)throw fail('原角色不存在');
  const read=()=>{requireGM(requester);return claimOf(actor,payload.nonce)},claim=await waitFor(()=>{const c=read();return c&&c.state!=='choosing'?c:null},'updateActor',a=>a===actor);
  const context=await contextFor({game,fromUuid,claim});if(game.user!==context.user)throw fail('本客户端不是原医疗拥有者');
  if(claim.state!=='rolling')throw fail('医疗已开始或状态不允许重复');
  const key=actor.uuid+':'+claim.nonce;if(tasks.has(key))return tasks.get(key);
  const promise=(async()=>{
   const write=(state,result)=>writes.run(actor.uuid,async()=>{requireGM(requester);const records=structuredClone(actor.flags?.[MODULE_ID]?.salubriousKissExecutions??[]),existing=records.find(r=>r.nonce===claim.nonce);
    if(state==='started'&&existing||state!=='started'&&existing?.state!=='started')throw fail('已有执行或未确认记录，不能重复');
    const next={nonce:claim.nonce,state,...result?{result}:{}},list=records.filter(r=>r.nonce!==claim.nonce);list.push(next);await actor.update({[`flags.${MODULE_ID}.salubriousKissExecutions`]:list});
    if(JSON.stringify(actor.flags?.[MODULE_ID]?.salubriousKissExecutions?.find(r=>r.nonce===claim.nonce))!==JSON.stringify(next))throw fail('原拥有者执行记录没有持久保存');
   });
   await write('started');
   try{requireGM(requester);if(!sameClaim(read(),claim)||read().state!=='rolling')throw fail('投骰前认领已改变');
    const output=await rollSalubriousTreatment({game,...context,claim,checkScope,createMessage,DamageRoll});requireGM(requester);if(!sameClaim(read(),claim)||read().state!=='rolling')throw fail('投骰后认领已改变');
    const result={checkId:output.check.id,damageId:output.damage?.id??null};await write('done',result);return result;
   }catch(error){await write('uncertain').catch(()=>{});throw error}
  })();tasks.set(key,promise);promise.finally(()=>{if(tasks.get(key)===promise)tasks.delete(key)}).catch(()=>{});return promise;
 }
 async function roll(claim){
  requireGM(game.user);const {actor,user}=await contextFor({game,fromUuid,claim});if(!sameClaim(claimOf(actor,claim.nonce),claim)||claimOf(actor,claim.nonce).state!=='rolling')throw fail('GM医疗认领未持久保存');
  const payload={actorUuid:actor.uuid,nonce:claim.nonce};let result;
  if(game.user===user)result=await ownerRoll(payload,game.user);else{if(!socket?.executeAsUser)throw fail('缺少原拥有者通讯');const response=await deadline(socket.executeAsUser('salubrious-kiss:roll',user.id,payload));if(!response?.ok)throw fail(response?.error??'原拥有者未确认');result=response.value;}
  requireGM(game.user);const check=await card(result.checkId),degree=validateSalubriousCard({game,message:check,claim});
  if(result.degree!==undefined&&degree!==result.degree)throw fail('GM收到的成功度不一致');if(degree!==1){const damage=await card(result.damageId);validateSalubriousCard({game,message:damage,claim,damage:true});if(damage.flags.pf2e.origin.messageId!==check.id)throw fail('医疗伤害未关联原检定')}else if(result.damageId)throw fail('医疗失败不应有伤害卡');
  requireGM(game.user);await contextFor({game,fromUuid,claim});if(!sameClaim(claimOf(actor,claim.nonce),claim)||claimOf(actor,claim.nonce).state!=='rolling')throw fail('拥有者执行期间GM医疗认领已改变');return {...result,degree};
 }
 async function apply(claim,result){
  if(typeof authorizeDamage!=='function')throw fail('缺少私有原生应用权限接口');
  requireGM(game.user);const {actor,item,token,target}=await contextFor({game,fromUuid,claim,allowImmune:true}),saved=claimOf(actor,claim.nonce),check=await card(result.checkId),damage=await card(result.damageId);
  if(!sameClaim(saved,claim)||saved.state!=='applying'||['checkId','damageId'].some(k=>saved.result?.[k]!==result?.[k])||kissState(target.actor).pending?.nonce!==claim.nonce||kissState(target.actor).pending.actorUuid!==actor.uuid)throw fail('没有本次准确应用认领');
  const degree=validateSalubriousCard({game,message:check,claim});validateSalubriousCard({game,message:damage,claim,damage:true});
  if(result.degree!==undefined&&degree!==result.degree||degree===1||damage.flags.pf2e.origin.messageId!==check.id)throw fail('应用来源不是本次原生医疗结果');
  DamageRoll??=game.pf2e?.DamageRoll??globalThis.CONFIG?.Dice?.rolls?.find(cls=>cls.name==='DamageRoll');const dice=damage.rolls[0],expected=treatmentOutcome({degree,tier:claim.tier});
  // DamageRoll.formula is a display expression without braces/healing flavor.
  // The evaluated roll's serialization retains the actual formula source.
  if(typeof DamageRoll!=='function'||!(dice instanceof DamageRoll)||dataRoll(dice.toJSON())?.formula?.replace(/\s/g,'')!==expected.formula.replace(/\s/g,''))throw fail('伤害不是本次原生医疗DamageRoll');
  const opts=damage.flags.pf2e.context.options.filter(o=>o!==skip),originOptions=opts.filter(o=>o.startsWith('self:')).map(o=>o.replace(/^self\b/,'origin'));
  if(target.actor.alliance)opts.push(`origin:${target.actor.alliance===actor.alliance?'ally':'enemy'}`);opts.push(...target.actor.getSelfRollOptions('target'));
  const ephemeral=degree===0?await Promise.all((actor.synthetics?.ephemeralEffects?.['damage-received']?.target??[]).map(fn=>fn({test:[...opts,...actor.getRollOptions(['damage-received']),...target.actor.getSelfRollOptions('target')],resolvables:{}}))):[];
  await contextFor({game,fromUuid,claim,allowImmune:true});validateSalubriousCard({game,message:check,claim});validateSalubriousCard({game,message:damage,claim,damage:true});
  if(damage.rolls[0]!==dice||damage.flags.pf2e.origin.messageId!==check.id)throw fail('准备期间原生伤害来源已改变');
  const effects=ephemeral.filter(Boolean).map(effect=>{const copy=structuredClone(effect);if(copy.type==='effect'){copy.system.context={origin:{actor:actor.uuid,token:null,item:null,spellcasting:null,rollOptions:[]},target:{actor:target.actor.uuid,token:null},roll:null};copy.system.duration={value:-1,unit:'unlimited',expiry:null,sustained:false};}return copy});
  const recipient=target.actor.getContextualClone(originOptions,effects),application=marker('apply',claim),source=`${MODULE_ID}:source:${damage.id}:0`;
  const params={damage:degree===0?dice.alter(1,0):-dice.total,token:target,item,skipIWR:degree!==0,final:false,shieldBlockRequest:false,outcome:outcomes[degree],rollOptions:new Set([...opts.filter(o=>!/^(?:self|target)(?::|$)/.test(o)),...originOptions,...recipient.getSelfRollOptions(),source,application])};
  await writes.run(actor.uuid,async()=>{requireGM(game.user);const current=claimOf(actor,claim.nonce);if(!sameClaim(current,claim)||current.state!=='applying'||current.application)throw fail('该次治疗应用已经开始，不能重复');const claims=structuredClone(kissState(actor).claims),application={userId:game.user.id,damageId:damage.id,targetUuid:target.uuid,state:'started'};claims.find(c=>c.nonce===claim.nonce).application=application;await actor.update({[`flags.${MODULE_ID}.salubriousKiss.claims`]:claims});const saved=claimOf(actor,claim.nonce);if(!sameClaim(saved,claim)||saved.state!=='applying'||JSON.stringify(saved.application)!==JSON.stringify(application))throw fail('治疗应用记录没有持久保存');});
  if(!Hooks?.on||!Hooks.off)throw fail('无法观察原生应用回执');const captured=new Set();let error,hook;
  const revoke=await authorizeDamage({reactor:actor,actor:recipient,item,token,target,check,message:damage,claim,params});
  if(typeof revoke!=='function')throw fail('私有应用权限没有返回释放接口');
  try{
  hook=Hooks.on('createChatMessage',message=>{const c=message.flags?.pf2e?.context;if(!c?.options?.includes(application))return;captured.add(message);
   if(claim.privacy){try{assertSalubriousReceiptPrivacy({message,claim});const p=message.flags?.[MODULE_ID]?.salubriousKiss;if(p?.kind!=='receipt'||p.nonce!==claim.nonce||p.damageId!==damage.id||p.targetUuid!==target.uuid)throw fail('原生私密回执的标记不符')}catch(e){error=e}}
   if(c.type!=='damage-taken'||!c.options.includes(source)||c.options.includes(skip)||author(message)!==game.user.id||message.speaker?.actor!==target.actor.id||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==target.uuid||message.flags.pf2e.appliedDamage&&message.flags.pf2e.appliedDamage.uuid!==target.actor.uuid)error=fail('原生医疗应用回执来源不符');
  });
  requireGM(game.user);await recipient.applyDamage(params);requireGM(game.user);if(error)throw error;if(captured.size!==1)throw fail('原生应用回执不唯一或未出现');const [receipt]=captured;if(game.messages.get(receipt.id)!==receipt)throw fail('原生回执未持久保存');return {messageId:receipt.id,targetUuid:target.uuid,kind:expected.kind};}
  finally{if(hook!==undefined)Hooks.off('createChatMessage',hook);revoke();}
 }
 function register({socket:api}={}){if(socket&&socket!==api)throw fail('通讯重复注册');socket=api;socket?.register('salubrious-kiss:roll',async function(payload){try{return {ok:true,value:await ownerRoll(payload,game.users.get(this.socketdata?.userId))}}catch(error){return {ok:false,error:String(error.message??error)}}})}
 return {roll,apply,ownerRoll,register};
}
