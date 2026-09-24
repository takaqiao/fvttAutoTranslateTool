import {MODULE_ID} from './rules.mjs';
import {withDamageMessageTarget} from './damage-message-targets.mjs';
import {getSourceId,isActiveGM,resolveMessageTargets} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {preserveDamagePartForMerge,preserveMergedDamageBypass} from './native-damage-components.mjs';
import {createAttackSequence} from './activity-attack-sequence.mjs';

const SOURCES={
 twin:'Compendium.pf2e.feats-srd.Item.Gw0wGXikhAhiGoud',
 double:'Compendium.pf2e.feats-srd.Item.onde0SxLoxLBTnvm',
};
const values=c=>Array.from(c?.values?.()??c??[]);
const own=d=>d?.flags?.[MODULE_ID]??{};
const SECOND_ATTACK=MODULE_ID+':double-slice-second';
const messageClass=()=>globalThis.CONFIG?.ChatMessage?.documentClass??globalThis.ChatMessage;
const precisionTotal=roll=>roll.instances.reduce((total,instance)=>total+instance.componentTotal('precision'),0);
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

/** Remove only precision subterms, retaining native crit/splash/die-result structure. */
export function removePrecisionDamage(roll){
 if(precisionTotal(roll)<=0)return roll;
 const data=structuredClone(roll.toJSON()),pool=data.terms?.[0];
 if(pool?.class!=='InstancePool'||pool.rolls?.length!==roll.instances.length)throw Error('当前PF2e伤害格式无法安全移除重复精确伤害。');
 for(const[index,instance]of roll.instances.entries()){
  const precision=instance.componentTotal('precision');if(precision<=0)continue;
  const remaining=instance.total-precision,entry=pool.rolls[index];
  if(!Number.isFinite(remaining)||remaining<0)throw Error('精确伤害分量无效。');
  let removed=0;
  const strip=term=>{
   if(String(term?.options?.flavor??'').split(',').includes('precision')){removed++;return {class:'NumericTerm',number:0,options:{},evaluated:true};}
   if(term?.operands)term.operands=term.operands.map(strip);
   if(term?.term)term.term=strip(term.term);
   if(term?.terms)term.terms=term.terms.map(strip);
   return term;
  };
  entry.terms=entry.terms.map(strip);if(!removed)throw Error('精确伤害未找到可识别的原生分量，不能安全移除。');
  entry.total=remaining;entry.dice=[];pool.results[index].result=remaining;
 }
 data.total=pool.rolls.reduce((n,r)=>n+r.total,0);
 // Serialized formula remains the original parse seed; native fromData replaces
 // its terms with this evaluated tree, and renders from those actual terms.
 data.dice=[];
 const DamageRoll=globalThis.CONFIG.Dice.rolls.find(c=>c.name==='DamageRoll');
 return DamageRoll.fromData(data);
}

export function createDualStrikeAutomation({game,fromUuid=globalThis.fromUuid,choose,onError=()=>{}}={}){
 const queue=new SerialActions();
 const resolveAction=item=>item?.type==='feat'?getSourceId(item)===SOURCES.twin?'dual:twin-takedown':getSourceId(item)===SOURCES.double?'dual:double-slice':null:null;
 async function select(ctx,title,choices){
  if(choices.length===1)return choices[0].value;
  const result=await choose({...ctx,title,choices});
  if(result==null)return null;
  if(!choices.some(c=>c.value===result))throw Error('无效的双武器活动选择。');
  return result;
 }
 async function sourceToken(actor,message,target){
  const {scene,token}=message.speaker??{};
  if(scene&&token){const source=await fromUuid(`Scene.${scene}.Token.${token}`);if(source?.actor?.uuid!==actor.uuid||source.parent?.id!==target.parent?.id)throw Error('双武器活动来源token不匹配。');return source;}
  const candidates=values(actor.getActiveTokens?.(true,true)).map(t=>t.document??t).filter(t=>t.actor?.uuid===actor.uuid&&t.parent?.id===target.parent?.id);
  if(candidates.length!==1)throw Error('无法唯一确定双武器活动的来源token；请从场景角色使用。');return candidates[0];
 }
 function requireSceneTarget(actor,origin,target){
  // Keep native Strike identity; reach and intervening walls belong to GM adjudication.
  const scene=origin.parent;
  if(origin.actor?.uuid!==actor.uuid||!scene||game.scenes?.get(scene.id)!==scene||scene.tokens?.get(origin.id)!==origin||target.parent!==scene||scene.tokens.get(target.id)!==target||!target.object||!target.actor)throw Error('双武器活动的来源或场景目标已改变。');
 }
 async function maintain(actor){
  if(!isActiveGM(game))return {status:'not-authority'};
  const feat=values(actor.items).find(i=>getSourceId(i)===SOURCES.double);
  const rules=feat?.system.rules;
  if(!Array.isArray(rules))return {status:'unchanged'};
  const next=structuredClone(rules);
  for(const rule of next){
   if(rule.key==='RollOption'&&rule.option==='double-slice-second'&&rule.toggleable===true&&rule.value===true)rule.value=false;
   // Native RollOption.beforeRoll removes its own option when the saved toggle
   // is false. Preserve that manual toggle and add an independent per-use key.
   if(rule.key==='FlatModifier'&&rule.value===-2&&rule.predicate?.includes('double-slice-second'))rule.predicate=rule.predicate.map(p=>p==='double-slice-second'?{or:[p,SECOND_ATTACK]}:p);
  }
  if(JSON.stringify(next)===JSON.stringify(rules))return {status:'unchanged'};
  if(!own(actor).dualStrikeRepair)await actor.update({[`flags.${MODULE_ID}.dualStrikeRepair`]:{itemId:feat.id,rules:structuredClone(rules),worldTime:game.time.worldTime}});
  await feat.update({'system.rules':next});return {status:'repaired'};
 }
 async function executeUsage({actor,item,message,user,action}){
  if(!isActiveGM(game)||!actor?.testUserPermission(user,'OWNER')||item?.actor?.uuid!==actor.uuid||resolveAction(item)!==action)throw Error('无权执行此双武器活动。');
  return queue.run(actor.uuid,async()=>{
   if(own(actor).dualStrikeUses?.includes(message.id))return '本次双武器活动已经结算。';
   // Toolbelt 3.56 exposes tool APIs under game.toolbelt.api (CustomModule.apiExpose).
   const merger=game.toolbelt?.api?.betterChat?.mergeDamageMessages;
   if(!game.modules.get('pf2e-toolbelt')?.active||typeof merger!=='function')throw Error('需要PF2e Toolbelt原生伤害合并API，尚未进行攻击。');
   const twin=action==='dual:twin-takedown';
   const targets=await resolveMessageTargets(message,{game,fromUuid});
   if(targets.length!==1||!targets[0].object)throw Error('请为此活动选定一个场景目标。');
   const target=targets[0];
   if(twin&&!target.actor.getRollOptions?.(['all']).includes(`self:prey:${actor.signature}`))throw Error('双重攻击只能针对本角色已确认的猎杀目标。');
   const wielded=()=>values(actor.system.actions).filter(s=>s.type==='strike'&&s.ready!==false&&s.item?.type==='weapon'&&s.item.isMelee===true&&s.item.system.equipped?.carryType==='held'&&s.item.system.equipped.handsHeld===1);
   const strikes=wielded();
   if(new Set(strikes.map(s=>s.item.id)).size<2)throw Error('此活动需要两把分别单手持用的近战武器。');
   const choices=Array.from(new Map(strikes.map(s=>[s.item.id,{value:s.item.id,label:s.item.name}])).values());
   const first=await select({actor,user},'选择首先攻击的武器',choices);if(first===null)return '已取消。';
   const second=await select({actor,user},'选择第二把武器',choices.filter(c=>c.value!==first));if(second===null)return '已取消。';
   const selectedMap=await select({actor,user},'当前多重攻击惩罚档位',[
    {value:'0',label:'本回合尚未攻击（MAP 0）'},
    {value:'1',label:'已攻击一次（MAP 1）'},
    {value:'2',label:'已攻击两次或更多（MAP 2）'},
   ]);if(selectedMap===null)return '已取消。';
   const map=Number(selectedMap);
   await maintain(actor);
   const selected=[first,second].map(id=>wielded().find(s=>s.item.id===id));
   if(selected.some(s=>!s))throw Error('武器持用状态已改变，尚未进行攻击。');
   const origin=await sourceToken(actor,message,target);requireSceneTarget(actor,origin,target);
   if(!twin&&!item.system.rules?.some(r=>r.key==='FlatModifier'&&r.predicate?.some(p=>p?.or?.includes(SECOND_ATTACK))))throw Error('双重切割的原生第二击修正规则缺失，尚未攻击。');
   await actor.update({[`flags.${MODULE_ID}.dualStrikeUses`]:[...(own(actor).dualStrikeUses??[]).slice(-127),message.id]});
   const hits=[],sequence=createAttackSequence({actor});
   for(const[index,strike]of selected.entries()){
    requireSceneTarget(actor,origin,target);
    const frame=sequence.begin(strike,target);
    const options=new Set([`${MODULE_ID}:dual-strike:${message.id}`,`action:${twin?'twin-takedown':'double-slice'}`,...frame.attackOptions]);
    if(twin)options.add('hunted-prey');else if(index===1){options.add('double-slice-second');options.add(SECOND_ATTACK);}
    const tier=twin?Math.min(map+index,2):map;
    let attackMessage=null;
    const check=await strike.variants[tier].roll({target:target.object,options,event:{ctrlKey:false,metaKey:false,shiftKey:game.user.settings?.showCheckDialogs??true},createMessage:false,callback:async(_roll,_outcome,raw)=>{
     const data=raw.toObject();delete data._id;
     data.flags={...data.flags,'xdy-pf2e-workbench':{...data.flags?.['xdy-pf2e-workbench'],noAutoDamageRoll:true},[MODULE_ID]:{...data.flags?.[MODULE_ID],dualStrikeAttack:{usageMessageId:message.id,index}}};
     attackMessage=await messageClass().create(data);
    }});
    if(!check||!attackMessage)throw Error('双武器活动已中止；已发生的攻击不会自动重试。');
    frame.capture(attackMessage);
    const outcome=attackMessage.flags.pf2e.context.outcome;
    sequence.record(frame,outcome);
    if(!isActiveGM(game))throw Error('主 GM 已交接；旧客户端停止双武器活动。');
    await frame.consume();
    if(!isActiveGM(game))throw Error('主 GM 已交接；旧客户端停止双武器活动。');
    if(!['success','criticalSuccess'].includes(outcome))continue;
    const damageOptions=new Set([`${MODULE_ID}:bear-attack:${attackMessage.id}`]);
    if(twin)damageOptions.add('hunted-prey');
    const {strike:damageStrike,options:sequenceOptions}=frame.damage(strike);
    for(const option of sequenceOptions)damageOptions.add(option);
    const roll=await damageStrike[outcome==='criticalSuccess'?'critical':'damage']({target:target.object,checkContext:attackMessage.flags.pf2e.context,mapIncreases:tier,options:damageOptions,event:{ctrlKey:false,metaKey:false,shiftKey:game.user.settings?.showDamageDialogs??true},createMessage:false});
    if(!roll)throw Error('攻击已发生，但原生伤害尚未完成。');
    hits.push({roll,strike,attackMessage,outcome});
   }
   if(!hits.length)return '两次攻击均未命中。';
   if(!twin){
    const precise=hits.filter(h=>precisionTotal(h.roll)>0);
    if(precise.length>1){
     const keep=await select({actor,user},'双重切割：哪次命中保留精确伤害',precise.map(h=>({value:h.attackMessage.id,label:`${h.strike.item.name}（${h.outcome==='criticalSuccess'?'大成功':'成功'}）`})));
     if(keep===null)throw Error('攻击已经完成；需要选择保留精确伤害的命中才能生成合并伤害。');
     for(const hit of precise)if(hit.attackMessage.id!==keep)hit.roll=removePrecisionDamage(hit.roll);
    }
   }
   const docs=[];
   for(const hit of hits){
    preserveDamagePartForMerge(hit.roll);
    const flags={pf2e:{origin:{uuid:hit.strike.item.uuid,type:'weapon',actor:actor.uuid},context:{type:'damage-roll',sourceType:'attack',outcome:hit.outcome,domains:['damage','strike-damage'],options:[`${MODULE_ID}:bear-attack:${hit.attackMessage.id}`],target:{actor:target.actor.uuid,token:target.uuid}}}};
    const data=await hit.roll.toMessage({speaker:hit.attackMessage.speaker,flags,flavor:escape(hit.strike.item.name)},{create:false});
    docs.push(new (messageClass())(typeof data?.toObject==='function'?data.toObject():data));
   }
   const combined=docs.length===1?docs[0]:await merger(docs[0],docs[1],{updateMessages:false});
   if(!combined)throw Error('原生合并伤害未完成。');
   if(docs.length>1)preserveMergedDamageBypass(combined.rolls[0],hits.map(hit=>hit.roll));
   // Toolbelt drops this option during merge. PF2e checks it before applying
   // critical-hit immunity; the evaluated terms still identify each crit part.
   if(hits.some(hit=>hit.outcome==='criticalSuccess')){
    combined.rolls[0].options??={};combined.rolls[0].options.degreeOfSuccess=3;
   }
   const data=combined.toObject();delete data._id;
   // toObject() reads _source, not the derived rolls just corrected above.
   data.rolls=combined.rolls.map(roll=>roll.toJSON());
   data.flags??={};data.flags.pf2e??={};
   data.flags.pf2e.origin={uuid:hits[0].strike.item.uuid,type:'weapon',actor:actor.uuid};
   data.flags.pf2e.context={...data.flags.pf2e.context,type:'damage-roll',sourceType:'attack',domains:['damage','strike-damage'],options:[...new Set([...(data.flags.pf2e.context?.options??[]),...hits.some(h=>h.outcome==='criticalSuccess')?['check:outcome:critical-success']:[],...hits.map(h=>`${MODULE_ID}:bear-attack:${h.attackMessage.id}`)])],target:{actor:target.actor.uuid,token:target.uuid}};
   data.flags[MODULE_ID]={...data.flags[MODULE_ID],usageGenerated:true,dualStrike:{usageMessageId:message.id,kind:action,map,attacks:hits.map(h=>({messageId:h.attackMessage.id,weaponUuid:h.strike.item.uuid,actorUuid:actor.uuid}))}};
   data.flavor=`<h4 class="action">${twin?'双重攻击':'双重切割'}：合并伤害</h4>${data.flavor??''}`;
   await messageClass().create(withDamageMessageTarget(data,target.uuid));
   return `已完成两次攻击，${hits.length}次命中；合并伤害只需应用一次，熊支援会随之自动结算。`;
  });
 }
 return {resolveAction,executeUsage,maintain,register:()=>()=>{}};
}
