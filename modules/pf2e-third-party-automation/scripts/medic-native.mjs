import {MODULE_ID} from './rules.mjs';
const scoped=(original,overrides)=>new Proxy(original,{get(target,key){if(Object.hasOwn(overrides,key))return overrides[key];const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}});
/** Delegate rules to the installed provider while pinning all selection reads in the macro's lexical scope. */
export function createMedicNative({game,choose,canvas=globalThis.canvas,Dialog=globalThis.Dialog,ChatMessage=globalThis.ChatMessage}={}){
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
  const nativeMacro=macros[0];let resolve,reject,opened=false,submitting=false,settled=false;
  const completed=new Promise((yes,no)=>{resolve=yes;reject=no;});
  const settle=value=>{if(!settled){settled=true;resolve(value);}};
  class TreatmentDialog extends Dialog{
   constructor(data,options){
    opened=true;const original=data.buttons?.yes?.callback;
    if(typeof original!=='function')throw Error('Workbench医疗对话框接口已改变。');
    super({...data,buttons:{...data.buttons,yes:{...data.buttons.yes,callback:async html=>{
     if(submitting||settled)return;submitting=true;
     try{validate();const control=html.find('[name="useBattleMedicine"]');control.val('1');await original(html);settle({status:'delegated',text:'已交给Workbench战地医疗；按其原生结果卡应用治疗与免疫。'});}catch(error){settled=true;reject(error);}
    }},no:{...data.buttons.no,callback:()=>settle({status:'cancelled'})}},render:html=>{data.render?.(html);html.find('[name="useBattleMedicine"]').val('1').trigger('change').prop('disabled',true);},close:html=>{data.close?.(html);if(!submitting)settle({status:'cancelled'});}},options);
   }
  }
  const scopedUser=scoped(game.user,{targets:new Set([target.object])});
  const scopedGame=scoped(game,{user:scopedUser});
  const scopedCanvas=scoped(canvas,{tokens:scoped(canvas.tokens,{controlled:[healer.object]})});
  const scopedMessages=ChatMessage?scoped(ChatMessage,{getSpeaker:()=>ChatMessage.getSpeaker({actor,token:healer.object})}):ChatMessage;
  await nativeMacro.execute({actor,token:healer.object,game:scopedGame,canvas:scopedCanvas,Dialog:TreatmentDialog,ChatMessage:scopedMessages});
  if(!opened)throw Error('Workbench未打开医疗对话框，未自动重复调用。');
  return completed;
 };
}
