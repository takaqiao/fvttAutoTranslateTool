import {MODULE_ID} from './rules.mjs';
const scoped=(original,overrides)=>new Proxy(original,{get(target,key){if(Object.hasOwn(overrides,key))return overrides[key];const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}});
export const pinnedMedicTarget=token=>scoped(token.actor,{getActiveTokens:()=>[token]});
/** Delegate rules to the installed provider while pinning all selection reads in the macro's lexical scope. */
export function createMedicNative({game,choose,canvas=globalThis.canvas,Dialog=globalThis.Dialog,ChatMessage=globalThis.ChatMessage,Hooks=globalThis.Hooks,CONFIG=globalThis.CONFIG}={}){
 return async function delegate({actor,healer,target,user,branch,continuation,validate}){
  validate();
  if(branch!=='battle-medicine'){
   const action=game.pf2e?.actions?.get?.(branch);if(!action?.use)throw Error('缺少原生医疗动作接口。');
   let variant;
   if(branch==='administer-first-aid'){
    variant=await choose?.({actor,user,title:'急救：选择方式',choices:[{value:'stabilize',label:'稳定濒死'},{value:'stop-bleeding',label:'止血'}]});
    if(!variant)return {status:'cancelled'};if(!['stabilize','stop-bleeding'].includes(variant))throw Error('无效急救方式。');
   }
   validate();const results=await action.use({actors:[actor],target:target.actor,variant,[MODULE_ID]:{metapowerContinuation:continuation}});
   return {status:results?.length?'delegated':'cancelled',text:'已调用原生医疗检定；后续效果由原生/已安装医疗提供者处理。'};
  }
  if(!game.modules.get('xdy-pf2e-workbench')?.active)throw Error('医师探访的战地医疗分支需要启用Workbench。');
  const pack=game.packs.get('xdy-pf2e-workbench.asymonous-benefactor-macros-internal');
  const macros=await pack?.getDocuments({name:'XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine'});
  if(macros?.length!==1||!Dialog||!healer.object||!target.object)throw Error('缺少Workbench原生战地医疗宏或场景Token。');
  const nativeMacro=macros[0];let resolve,reject,opened=false,submitting=false,settled=false,checkStarted=false,checkFinished=false,checkMessage=null,resultMessage=null,pendingAssuranceMessage=null;
  const nonce=continuation?.nonce,marker=`${MODULE_ID}:medic-workbench:${nonce}`,listeners=[],hookIds=[],assuranceRolls=new Set();
  const completed=new Promise((yes,no)=>{resolve=yes;reject=no;});
  const clean=()=>{for(const [event,id]of hookIds)Hooks.off(event,id);for(const [element,fn]of listeners)element.removeEventListener('submit',fn,true);};
  const settle=value=>{if(!settled){settled=true;clean();resolve(value);}};
  const fail=error=>{if(!settled){settled=true;clean();reject(error);}};
  const finish=()=>{if(checkFinished&&checkMessage&&resultMessage)settle({status:'delegated',text:'Workbench战地医疗检定及结果卡已完成；按其原生卡应用治疗与免疫。'});};
  const proof=()=>({nonce,cardId:continuation.cardId,actorUuid:actor.uuid,targetUuid:target.uuid,checkId:checkMessage?.id});
  const bindResult=async(data,create)=>{
   const result=data?.flags?.treat_wounds_battle_medicine;
   if(!result)return create(data);
   try{
    if(settled)return null;validate();if(pendingAssuranceMessage)await pendingAssuranceMessage;if(settled)return null;
    if(!checkMessage||result.id!==target.id||result.healerId!==actor.id)throw Error('Workbench医疗结果不属于原医疗检定。');
    const message=await create({...data,flags:{...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],medicWorkbench:proof()}}});
    if(!message?.id)throw Error('Workbench医疗结果消息未保存。');
    resultMessage=message;finish();return message;
   }catch(error){fail(error);return null;}
  };
  // Native Statistic.roll resolves only after its native check and callback; Workbench's outer callback does not.
  const skillScope=Object.fromEntries(Object.entries(actor.skills??{}).map(([slug,stat])=>[slug,scoped(stat,{roll:async args=>{
   try{
    if(settled)return null;validate();if(!nonce||checkStarted)throw Error('Workbench医疗检定回执已开始或缺失。');checkStarted=true;
    const rolled=await stat.roll({...args,token:healer,target:pinnedMedicTarget(target),dc:{...args.dc,slug:'medicine'},extraRollOptions:[...(args.extraRollOptions??[]),marker],callback:async(roll,outcome,message,...rest)=>{
     if(settled)return;validate();const context=message?.flags?.pf2e?.context;
     if(!message?.id||game.messages.get(message.id)!==message||message.speaker?.actor!==actor.id||context?.isReroll||context?.type!=='skill-check'||!context.options?.includes(marker)||context.target?.actor!==target.actor.uuid||context.target?.token!==target.uuid)throw Error('Workbench原生检定回执不匹配。');
     checkMessage=message;await message.update({[`flags.${MODULE_ID}.medicWorkbench`]:proof()});
     await args.callback?.(roll,outcome,message,...rest);
    }});
    if(settled)return rolled;if(!rolled){settle({status:'cancelled'});return null;}
    if(!checkMessage)throw Error('Workbench没有返回原生医疗检定消息。');checkFinished=true;finish();return rolled;
   }catch(error){fail(error);return null;}
  }})]));
  const scopedActor=scoped(actor,{skills:skillScope}),scopedHealer=scoped(healer.object,{actor:scopedActor});
  if(Hooks?.on)hookIds.push(['renderCheckModifiersDialog',Hooks.on('renderCheckModifiersDialog',(dialog,html)=>{
   const options=dialog.context?.options;if(!(options?.has?.(marker)||options?.includes?.(marker)))return;
   const element=html?.[0]??html;if(!element?.addEventListener)return;
   const check=event=>{try{if(settled)throw Error('此医疗检定已结束。');validate();}catch(error){event.preventDefault();event.stopImmediatePropagation();dialog.resolve?.(false);void dialog.close?.();fail(error);}};
   element.addEventListener('submit',check,true);listeners.push([element,check]);
  })]);
  const scopedMessages=ChatMessage?scoped(ChatMessage,{getSpeaker:()=>ChatMessage.getSpeaker({actor,token:healer.object}),create:async(data,...args)=>{
   const rolls=data?.rolls??data?.roll;
   if(Array.isArray(rolls)&&rolls.some(roll=>assuranceRolls.has(roll))){
    pendingAssuranceMessage=(async()=>{try{if(settled)return null;validate();const message=await ChatMessage.create({...data,flags:{...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],medicWorkbench:proof()}}},...args);if(!message?.id)throw Error('Workbench Assurance消息未保存。');checkMessage=message;checkFinished=true;finish();return message;}catch(error){fail(error);return null;}})();return pendingAssuranceMessage;
   }
   return bindResult(data,actual=>ChatMessage.create(actual,...args));
  }}):ChatMessage;
  const rollClasses=(CONFIG?.Dice?.rolls??[]).map(Base=>{
   if(Base.name==='DamageRoll')return class DamageRoll extends Base{async toMessage(data,...args){return bindResult(data,actual=>super.toMessage(actual,...args));}};
   if(Base.name==='CheckRoll')return class CheckRoll extends Base{async roll(...args){try{validate();if(settled||checkStarted||!nonce)throw Error('Workbench Assurance回执无效。');checkStarted=true;const roll=await super.roll(...args);assuranceRolls.add(roll);return roll;}catch(error){fail(error);return null;}}};
   return Base;
  });
  const scopedConfig=CONFIG?scoped(CONFIG,{Dice:scoped(CONFIG.Dice,{rolls:rollClasses})}):CONFIG;
  class TreatmentDialog extends Dialog{
   constructor(data,options){
    opened=true;const original=data.buttons?.yes?.callback;
    if(typeof original!=='function')throw Error('Workbench医疗对话框接口已改变。');
    super({...data,buttons:{...data.buttons,yes:{...data.buttons.yes,callback:async html=>{
     if(submitting||settled)return;submitting=true;
     try{validate();const control=html.find('[name="useBattleMedicine"]');control.val('1');await original(html);if(!checkStarted&&!settled)throw Error('Workbench未启动医疗检定，未重复调用。');}catch(error){fail(error);}
    }},no:{...data.buttons.no,callback:()=>settle({status:'cancelled'})}},render:html=>{data.render?.(html);html.find('[name="useBattleMedicine"]').val('1').trigger('change').prop('disabled',true);},close:html=>{data.close?.(html);if(!submitting)settle({status:'cancelled'});}},options);
   }
  }
  const scopedUser=scoped(game.user,{targets:new Set([target.object])});
  const scopedGame=scoped(game,{user:scopedUser});
  const scopedCanvas=scoped(canvas,{tokens:scoped(canvas.tokens,{controlled:[scopedHealer]})});
  try{await nativeMacro.execute({actor:scopedActor,token:scopedHealer,game:scopedGame,canvas:scopedCanvas,Dialog:TreatmentDialog,ChatMessage:scopedMessages,CONFIG:scopedConfig});if(!opened)throw Error('Workbench未打开医疗对话框，未自动重复调用。');}catch(error){fail(error);}
  return completed;
 };
}
