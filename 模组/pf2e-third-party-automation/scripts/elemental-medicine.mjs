import {MODULE_ID} from './rules.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {ELEMENTAL_MEDICINE_DAILY,ELEMENTAL_MEDICINE_EFFECT,ELEMENTAL_MEDICINE_ELEMENTS,elementalMedicineFeat,hasElementalMedicine,elementalMedicineSkills,validateElementalMedicinePatients,elementalMedicineBinding,elementalMedicineDC,buildElementalMedicineEffect} from './elemental-medicine-rules.mjs';

const DAILIES='pf2e-dailies',values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]?.elementalMedicine;
const registered=new WeakSet(),sessions=new WeakMap();
const uid=()=>globalThis.foundry?.utils?.randomID?.(16)??globalThis.crypto.randomUUID().replaceAll('-','').slice(0,16);
const copy=value=>structuredClone(value),escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const terminal=s=>['done','cancelled'].includes(s);
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const digest=async value=>Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),v=>v.toString(16).padStart(2,'0')).join('');
const roster=(game,user)=>{
 const candidates=[...values(game.actors).filter(a=>a.type==='character'),...values(game.scenes?.current?.tokens).filter(t=>t.object?.visible).map(t=>t.actor)];
 return [...new Map(candidates.filter(a=>a&&['character','npc','familiar'].includes(a.type)&&a.testUserPermission?.(user,'LIMITED')).map(a=>[a.uuid,a])).values()];
};

/** GM fact entry is not an outcome picker. All checks and medicine consequences follow automatically. */
async function nativeFacts({actor,patients}){
 const Dialog=globalThis.foundry?.applications?.api?.DialogV2;if(!Dialog)throw Error('缺少原生GM诊疗事实对话框。');
 const elementOptions=Object.keys(ELEMENTAL_MEDICINE_ELEMENTS).map(e=>`<option value="${e}">${{earth:'土',fire:'火',metal:'金',water:'水',wood:'木'}[e]}</option>`).join('');
 const content=patients.map((p,i)=>`<fieldset><legend>${escape(p.actor.name)} · ${escape(p.skill)}</legend>
  <label>已有病症豁免来源 UUID <input name="source${i}" placeholder="Actor.…Item.…"></label>
  <p>无现成来源时，填写以下事实，建立原生保存入口；不自动推断苦难阶段。</p>
  <label>显示给患者的症状名 <input name="name${i}"></label>
  <label>豁免 <select name="save${i}"><option value="fortitude">强韧</option><option value="reflex">反射</option><option value="will">意志</option></select></label>
  <label>实际病症豁免 DC <input name="saveDC${i}" type="number" min="1" max="100"></label>
  <label>病症特征 <select name="trait${i}"><option value="disease">疾病</option><option value="poison">毒素</option><option value="curse">诅咒</option></select></label>
  <label>诊断等级 DC 调整 <input name="adjustment${i}" type="number" min="-50" max="50" value="0"></label>
  <label><input name="materials${i}" type="checkbox" checked>本次所需材料可用</label>
  <label>正确用药元素 <select name="correctElement${i}">${elementOptions}</select></label>
  <label>误诊用药元素 <select name="wrongElement${i}">${elementOptions}</select></label>
  <label>成功时告知的诊断 <input name="correct${i}" placeholder="可留空，仍自动结算药效"></label>
  <label>误诊时告知的叙事 <input name="wrong${i}" placeholder="不要向玩家注明误诊"></label></fieldset>`).join('');
 const result=await Dialog.wait({window:{title:`五气养生 · GM事实 · ${actor.name}`},content:`<p>以下真实事实仅供GM。秘密检定自动决定结果；此表单不让GM选择成功度。</p>${content}`,buttons:[{action:'confirm',label:'确认事实并秘密诊断',default:true,callback:(_event,button)=>{
  const data=new FormData(button.form);return patients.map((p,i)=>({patientUuid:p.actor.uuid,sourceUuid:String(data.get(`source${i}`)??'').trim(),createSource:{name:String(data.get(`name${i}`)??'').trim(),save:String(data.get(`save${i}`)),dc:Number(data.get(`saveDC${i}`)),trait:String(data.get(`trait${i}`))},adjustment:Number(data.get(`adjustment${i}`)),materials:data.has(`materials${i}`),correctElement:String(data.get(`correctElement${i}`)),wrongElement:String(data.get(`wrongElement${i}`)),correctDiagnosis:String(data.get(`correct${i}`)??'').trim(),wrongDiagnosis:String(data.get(`wrong${i}`)??'').trim()}));
 }},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
 return result==='cancel'?null:result;
}

/** Uses Dailies' own item batch, then routes cross-actor outcomes to the active GM. */
export function createElementalMedicine({game,fromUuid=globalThis.fromUuid,requestFacts=nativeFacts,createMessage=(data,options)=>globalThis.ChatMessage.create(data,options),rollCheck,publishDiagnosis,onError=console.error}={}){
 if(!sessions.has(game))sessions.set(game,{doctors:new SerialActions(),patients:new SerialActions(),running:new Map()});
 const session=sessions.get(game),pendingChecks=new Map(),hooks=[],tracked=new Map();let socket,hookApi,installed=false,indexed=false;
 const gm=()=>{if(!isActiveGM(game)||game.users.get(game.user.id)!==game.user||!game.user.isGM)throw Error('五气养生主GM已改变，未继续处理。');};
 const allowed=(actor,user)=>!!(user?.active&&game.users.get(user.id)===user&&actor.testUserPermission?.(user,'OWNER'));
 const gmIds=()=>values(game.users).filter(u=>u.isGM).map(u=>u.id);
 const report=e=>{try{onError(e);}catch{/* Reporting never repeats a diagnosis. */}};
 const save=async(request,change)=>{gm();if(request.actor.items.get(request.id)!==request)throw Error('每日准备凭据已不存在。');const data=copy(own(request));change(data);gm();await request.update({[`flags.${MODULE_ID}.elementalMedicine`]:data});gm();return own(request);};
 function live(request,user,{closed=false}={}){
  gm();const state=own(request),actor=request?.actor;
  if(!actor||actor.items.get(request.id)!==request||state?.kind!=='preparation'||state.actorUuid!==actor.uuid||state.userId!==user?.id||!allowed(actor,user)||!hasElementalMedicine(actor)||request.flags?.[DAILIES]?.daily!==`module.${ELEMENTAL_MEDICINE_DAILY}`||!closed&&state.closed)throw Error('五气养生的真实日备凭据、操作者或专长已失效。');
  if(!Array.isArray(state.patients)||!state.patients.length||state.patients.length>6||new Set(state.patients.map(p=>p.patientUuid)).size!==state.patients.length)throw Error('每日准备的患者数量或身份无效。');return state;
 }
 async function patientsFor(request,user,{closed=false}={}){
  const state=live(request,user,{closed}),patients=[];
  for(const p of state.patients){const actor=await fromUuid(p.patientUuid);gm();if(actor?.uuid!==p.patientUuid||!['character','npc','familiar'].includes(actor.type)||!actor.testUserPermission?.(user,'LIMITED')||!elementalMedicineSkills(request.actor).some(s=>s.slug===p.skill))throw Error('本次患者或技能已不可用。');patients.push({...p,actor});}return patients;
 }
 function secret(message){return message&&game.messages.get(message.id)===message&&game.users.get(author(message))?.isGM&&message.blind===true&&Array.isArray(message.whisper)&&message.whisper.length>0&&message.whisper.every(id=>game.users.get(id)?.isGM);}
 async function collectFacts(request,user,patients){
  await save(request,s=>{s.status='choosing';});const proposed=await requestFacts({actor:request.actor,patients,user});gm();live(request,user);
  if(proposed==null){await save(request,s=>{s.status='cancelled';});return null;}
  if(!Array.isArray(proposed)||proposed.length!==patients.length||new Set(proposed.map(f=>f.patientUuid)).size!==patients.length)throw Error('GM诊疗事实必须逐一对应准确患者。');
  const facts=[];
  // Validate all user-entered facts before creating any source document or rolling.
  for(const p of patients){const f=proposed.find(f=>f.patientUuid===p.patientUuid);
   if(!f||f.materials!==true||!Object.hasOwn(ELEMENTAL_MEDICINE_ELEMENTS,f.correctElement)||!Object.hasOwn(ELEMENTAL_MEDICINE_ELEMENTS,f.wrongElement)||[f.correctDiagnosis,f.wrongDiagnosis].some(v=>v!=null&&(typeof v!=='string'||v.length>2000)))throw Error('请确认材料、用药元素及给玩家的诊断叙事。');
   const dc=elementalMedicineDC(p.actor.level,{pwol:game.pf2e?.settings?.variants?.pwol?.enabled===true,adjustment:f.adjustment??0});let binding;
   if(f.sourceUuid){const source=await fromUuid(f.sourceUuid);gm();binding=elementalMedicineBinding(source);}
   else{const input=f.createSource;if(!input?.name||typeof input.name!=='string'||input.name.length>160||!['fortitude','reflex','will'].includes(input.save)||!['disease','poison','curse'].includes(input.trait)||!Number.isInteger(input.dc)||input.dc<1||input.dc>100)throw Error('没有现成来源时，需要GM提供症状名、准确豁免/DC和病症特征。');}
   facts.push({patientUuid:p.patientUuid,skill:p.skill,level:p.actor.level,dc,binding,createSource:binding?null:copy(f.createSource),correctElement:f.correctElement,wrongElement:f.wrongElement,correctDiagnosis:f.correctDiagnosis??'',wrongDiagnosis:f.wrongDiagnosis??''});
  }
  for(const f of facts)if(!f.binding)f.sourceId=uid();
  const factsId=uid();await save(request,s=>{s.factsId=factsId;s.status='facts-writing';});gm();live(request,user);
  const message=await createMessage({_id:factsId,user:game.user.id,blind:true,whisper:gmIds(),speaker:{actor:request.actor.id},content:'<p>五气养生：本次真实诊疗事实（仅GM）。</p>',flags:{[MODULE_ID]:{elementalMedicineFacts:{requestUuid:request.uuid,facts}}}},{keepId:true});gm();
  if(message?.id!==factsId||!secret(message))throw Error('真实诊疗事实没有保存为GM秘密记录。');await save(request,s=>{s.status='diagnosing';});return facts;
 }
 function readFacts(request){const message=game.messages.get(own(request).factsId),proof=message?.flags?.[MODULE_ID]?.elementalMedicineFacts;if(!secret(message)||proof?.requestUuid!==request.uuid||!Array.isArray(proof.facts))throw Error('本次GM秘密事实记录不可核实。');return proof.facts;}
 async function materializeSources(request,user,patients,patientUuid){
  let facts=copy(readFacts(request));
  for(const f of facts)if(f.patientUuid===patientUuid&&!f.binding){
   if(own(request).closed)throw Error('旧日备已结束，不能补建未完成的病症来源。');
   const patient=patients.find(p=>p.patientUuid===f.patientUuid)?.actor,input=f.createSource,message=game.messages.get(own(request).factsId);gm();live(request,user);
   if(!patient||!input||!f.sourceId)throw Error('缺少GM确认的病症保存入口。');
   let source=patient.items.get(f.sourceId);
   if(!source){
    if(f.sourceStarted)throw Error('病症入口已开始创建但回执不确定，不重复创建。');
    f.sourceStarted=true;gm();await message.update({[`flags.${MODULE_ID}.elementalMedicineFacts.facts`]:facts});gm();live(request,user);
    const data={_id:f.sourceId,name:input.name,type:'action',system:{actionType:{value:'passive'},actions:{value:null},traits:{value:[input.trait]},description:{value:`<p>@Check[${input.save}|dc:${input.dc}|traits:${input.trait}]{病症豁免}</p>`,gm:''},rules:[]},flags:{[MODULE_ID]:{elementalMedicine:{kind:'affliction-source',requestUuid:request.uuid,patientUuid:patient.uuid}}}};
    await patient.createEmbeddedDocuments('Item',[data],{keepId:true});gm();source=patient.items.get(f.sourceId);
   }
   if(own(source)?.kind!=='affliction-source'||own(source).requestUuid!==request.uuid||own(source).patientUuid!==patient.uuid)throw Error('病症入口的准确文档回执不匹配。');
   f.binding=elementalMedicineBinding(source);delete f.createSource;gm();await message.update({[`flags.${MODULE_ID}.elementalMedicineFacts.facts`]:facts});gm();
  }
  return facts;
 }
 const checkProof=(request,p,message,fact)=>{
  const context=message?.flags?.pf2e?.context,proof=message?.flags?.[MODULE_ID]?.elementalMedicineCheck,options=context?.options??[],roll=message?.rolls?.[0],degree=['criticalFailure','failure','success','criticalSuccess'].indexOf(context?.outcome);
  if(!secret(message)||author(message)!==p.rollUserId||message.actor?.uuid!==request.actor.uuid||message.speaker?.actor!==request.actor.id||context?.type!=='skill-check'||context.dc?.value!==fact.dc||!options.includes(`${MODULE_ID}:elemental-medicine:${p.nonce}`)||!options.includes(`check:statistic:${p.skill}`)||proof?.requestUuid!==request.uuid||proof.patientUuid!==p.patientUuid||proof.nonce!==p.nonce||proof.skill!==p.skill||!roll?._evaluated||!Number.isFinite(roll.total)||degree<0||roll.options?.degreeOfSuccess!==degree)throw Error('缺少准确的原生秘密诊断检定回执。');return {degree,message};
 };
 function findCheck(request,p,fact){const matches=values(game.messages).filter(m=>{try{checkProof(request,p,m,fact);return true;}catch{return false;}});return matches.length===1?matches[0]:null;}
 async function nativeRoll(args){
  const statistic=elementalMedicineSkills(args.actor).find(s=>s.slug===args.skill)?.statistic;
  if(typeof statistic?.check?.roll!=='function'||!hookApi)throw Error('缺少原生秘密技能检定接口。');
  pendingChecks.set(args.nonce,args);
  try{return await statistic.check.roll({item:elementalMedicineFeat(args.actor),dc:{value:args.dc,visible:false},skipDialog:true,messageMode:'blind',extraRollOptions:[`action:prepare-elemental-medicine`,`${MODULE_ID}:elemental-medicine:${args.nonce}`],traits:['exploration','manipulate','secret'],label:'五气养生',action:'prepare-elemental-medicine'});}
  finally{pendingChecks.delete(args.nonce);}
 }
 async function tell(request,p,fact,degree){
  if(p.noticeId||p.noticeStarted)return;
  const text=degree===1?'未能诊断这次病症，未配制新的五行药物。':degree===0?(fact.wrongDiagnosis||'已按照诊断配制并施用了五行药物。'):(fact.correctDiagnosis||'已按照诊断配制并施用了五行药物。');
  await save(request,s=>{s.patients.find(r=>r.nonce===p.nonce).noticeStarted=true;});gm();
  const recipients=[...new Set([...gmIds(),own(request).userId,...values(game.users).filter(u=>u.active&&game.actors.get(p.patientUuid.split('.').at(-1))?.testUserPermission?.(u,'OWNER')).map(u=>u.id)])];
  const context={actor:request.actor,patientUuid:p.patientUuid,nonce:p.nonce,text,recipients};
  const message=publishDiagnosis?await publishDiagnosis(context):await createMessage({user:game.user.id,whisper:recipients,speaker:{actor:request.actor.id},content:`<p><strong>五气养生</strong></p><p>${escape(text)}</p>`,flags:{[MODULE_ID]:{elementalMedicineNotice:{nonce:p.nonce,requestUuid:request.uuid}}}});gm();
  if(message?.id)await save(request,s=>{s.patients.find(r=>r.nonce===p.nonce).noticeId=message.id;});
 }
 async function processPatient(request,user,patient,fact,{recoverOnly=false}={}){
  let p=own(request).patients.find(p=>p.patientUuid===patient.uuid);if(p.state==='done')return;
  if(!p.nonce){if(own(request).closed)return;await save(request,s=>{Object.assign(s.patients.find(r=>r.patientUuid===patient.uuid),{nonce:uid(),rollUserId:game.user.id,state:'rolling'});});p=own(request).patients.find(p=>p.patientUuid===patient.uuid);
   const proof={requestUuid:request.uuid,patientUuid:p.patientUuid,nonce:p.nonce,skill:p.skill};gm();live(request,user);
   await (rollCheck??nativeRoll)({actor:request.actor,patient,skill:p.skill,dc:fact.dc,nonce:p.nonce,proof});gm();
  }
  let check=p.checkId?game.messages.get(p.checkId):findCheck(request,p,fact);if(!check)throw Error('诊断已开始但没有可核实回执，保留不确定状态，不重掷。');
  const {degree}=checkProof(request,p,check,fact);
  if(!p.checkId){await save(request,s=>{const row=s.patients.find(r=>r.nonce===p.nonce);row.checkId=check.id;row.state='checked';});p=own(request).patients.find(r=>r.nonce===p.nonce);}
  if(degree!==1){
   await session.patients.run(patient.uuid,async()=>{
    gm();live(request,user,{closed:true});p=own(request).patients.find(r=>r.nonce===p.nonce);
    if(p.state==='applying'){
     const applied=patient.items.get(p.effectId);if(own(applied)?.kind!=='medicine'||own(applied).nonce!==p.nonce||own(applied).checkId!==check.id)throw Error('施药已经开始但没有准确药效回执，不重复施药。');
    }else if(p.state!=='applied'){
     if(own(request).closed)throw Error('上一日备周期已经结束，不延后补做未执行的施药。');
     const source=await fromUuid(fact.binding.itemUuid);gm();if(JSON.stringify(elementalMedicineBinding(source))!==JSON.stringify(fact.binding))throw Error('病症保存来源发生变化，未施药。');
     const previous=values(patient.items).filter(i=>i.type==='effect'&&getSourceId(i)===ELEMENTAL_MEDICINE_EFFECT),id=previous[0]?.id??uid(),effectSource=(await fromUuid(ELEMENTAL_MEDICINE_EFFECT))?.toObject();gm();live(request,user);
     const data=buildElementalMedicineEffect({source:effectSource,degree,skill:p.skill,element:degree===0?fact.wrongElement:fact.correctElement,binding:fact.binding,actor:request.actor,item:elementalMedicineFeat(request.actor),patientUuid:patient.uuid,nonce:p.nonce,checkId:check.id,worldTime:game.time.worldTime,effectId:id});
     const duplicates=await Promise.all(previous.slice(1).map(async i=>({id:i.id,fingerprint:await digest(JSON.stringify(i.toObject()))})));gm();live(request,user);
     await save(request,s=>{Object.assign(s.patients.find(r=>r.nonce===p.nonce),{state:'applying',effectId:id,applicationUserId:game.user.id,duplicates});});gm();live(request,user);
     if(previous.length)await patient.updateEmbeddedDocuments('Item',[data]);else await patient.createEmbeddedDocuments('Item',[data],{keepId:true});gm();
     const actual=patient.items.get(id);if(own(actual)?.nonce!==p.nonce||own(actual)?.checkId!==check.id)throw Error('原生药效写入未取得准确回执。');
    }
    p=own(request).patients.find(r=>r.nonce===p.nonce);const remove=[];
    for(const proof of p.duplicates??[]){const existing=patient.items.get(proof.id);if(!existing)continue;const source=JSON.stringify(existing.toObject()),hash=await digest(source);gm();if(patient.items.get(proof.id)!==existing||getSourceId(existing)!==ELEMENTAL_MEDICINE_EFFECT||hash!==proof.fingerprint||JSON.stringify(existing.toObject())!==source)throw Error('原有重复药效已被其他操作改变，保留现场，不删除新的药效。');remove.push(proof.id);}
    if(remove.length){gm();await patient.deleteEmbeddedDocuments('Item',remove);gm();}
    await save(request,s=>{const row=s.patients.find(r=>r.nonce===p.nonce);row.state='applied';delete row.duplicates;});
   });
  }
  p=own(request).patients.find(r=>r.patientUuid===patient.uuid);await tell(request,p,fact,degree);await save(request,s=>{s.patients.find(r=>r.nonce===p.nonce).state='done';});
 }
 async function execute(request,user,{recoverOnly=false}={}){
  gm();const state=live(request,user,{closed:recoverOnly});if(terminal(state.status))return {status:state.status};
  await session.doctors.run(request.actor.uuid,async()=>{gm();if(recoverOnly&&own(request).closed)return;const prior=request.actor.flags?.[MODULE_ID]?.elementalMedicineDaily?.requestUuid,existing=prior?await fromUuid(prior):null;gm();
   if(existing&&existing!==request&&!own(existing)?.closed)throw Error('本次每日准备已经存在诊疗记录，不能重复诊断。');
   if(prior!==request.uuid){if(recoverOnly&&(own(request).status!=='pending'||own(request).patients.some(p=>p.nonce)))throw Error('已开始的旧日备不能在恢复时重新认领。');gm();await request.actor.update({[`flags.${MODULE_ID}.elementalMedicineDaily`]:{requestUuid:request.uuid}});gm();}
  });
  try{
   const patients=await patientsFor(request,user,{closed:recoverOnly});let facts=own(request).factsId?readFacts(request):own(request).closed?null:await collectFacts(request,user,patients);if(!facts)return {status:own(request).status};
   const failures=[];
   for(const patient of patients)try{facts=await materializeSources(request,user,patients,patient.patientUuid);const fact=facts.find(f=>f.patientUuid===patient.patientUuid&&f.skill===patient.skill);if(!fact)throw Error('秘密事实与本次患者身份不符。');await processPatient(request,user,patient.actor,fact,{recoverOnly});}catch(error){gm();live(request,user,{closed:recoverOnly});failures.push(error);}
   const finished=own(request).patients.every(p=>p.state==='done');await save(request,s=>{s.status=finished?'done':'uncertain';});if(failures.length)throw failures[0];return {status:own(request).status};
  }catch(error){if(isActiveGM(game)&&request.actor.items.get(request.id)===request)await save(request,s=>{s.status='uncertain';}).catch(report);throw error;}
 }
 async function prepare(uuid,user,{recoverOnly=false}={}){
  gm();const request=await fromUuid(uuid);gm();if(!request)throw Error('找不到真实每日准备凭据。');live(request,user,{closed:recoverOnly});tracked.set(request.uuid,request);
  if(session.running.has(uuid))return session.running.get(uuid);
  const task=execute(request,user,{recoverOnly});session.running.set(uuid,task);try{return await task;}finally{if(session.running.get(uuid)===task)session.running.delete(uuid);}
 }
 const daily={key:ELEMENTAL_MEDICINE_DAILY,label:'五气养生',condition:actor=>hasElementalMedicine(actor)||values(actor.items).some(i=>own(i)?.kind==='preparation'),
  rows:actor=>{const patients=roster(game,game.user),skills=elementalMedicineSkills(actor),rows=[];for(let i=1;i<=6;i++){rows.push({type:'select',slug:`patient${i}`,label:`患者 ${i}`,save:false,empty:true,unique:'elemental-medicine-patients',options:[{value:'',label:'不选择',skipUnique:true},...patients.map(a=>({value:a.uuid,label:a.name}))]});rows.push({type:'select',slug:`skill${i}`,label:`患者 ${i} · 诊断技能`,save:false,options:skills.map(s=>({value:s.slug,label:s.label}))});}return rows;},
  process:({actor,rows,addItem,messages})=>{
   if(!allowed(actor,game.user))throw Error('只有操作者本人可提交每日准备。');const patients=validateElementalMedicinePatients(actor,rows,roster(game,game.user));if(!patients.length)return;
   if(values(actor.items).some(i=>own(i)?.kind==='preparation'&&!own(i).closed))throw Error('本次每日准备已经有诊疗记录；再次休息后的新日备才重新诊断。');
   const data={name:'五气养生 · 日备记录',type:'effect',system:{description:{value:'<p>本次每日准备的诊疗凭据。真实病症与秘密诊断由GM处理。</p>',gm:''},duration:{value:-1,unit:'unlimited'},rules:[],tokenIcon:{show:false}},flags:{[DAILIES]:{daily:`module.${ELEMENTAL_MEDICINE_DAILY}`},[MODULE_ID]:{elementalMedicine:{kind:'preparation',actorUuid:actor.uuid,userId:game.user.id,status:'pending',patients:patients.map(p=>({...p,state:'pending'}))}}}};
   addItem(data,false);messages?.add?.('third-party',{label:'五气养生：本次诊疗由GM秘密处理'});
  },
  afterItemAdded:async({actor,addedItems})=>{const requests=addedItems.filter(i=>i.actor===actor&&own(i)?.kind==='preparation'&&own(i).userId===game.user.id);for(const request of requests){tracked.set(request.uuid,request);try{if(isActiveGM(game))await prepare(request.uuid,game.user);else{if(!socket||!game.users.activeGM)throw Error('五气养生需要在线主GM处理秘密诊断。');const response=await socket.executeAsUser('elemental-medicine:prepare',game.users.activeGM.id,{requestUuid:request.uuid});if(!response?.ok)throw Error(response?.error??'秘密诊断尚未确认。');}}catch(error){report(error);}}},
  rest:({actor,removeItem,updateItem})=>{for(const request of values(actor.items).filter(i=>own(i)?.kind==='preparation')){if(terminal(own(request).status))removeItem(request.id);else updateItem({_id:request.id,[`flags.${MODULE_ID}.elementalMedicine.closed`]:true});}},
 };
 function registerDailies(){if(!game.modules.get(DAILIES)?.active)return {status:'inactive'};const api=game.dailies?.api;if(typeof api?.registerCustomDailies!=='function')return {status:'unavailable'};if(registered.has(api))return {status:'already-registered'};api.registerCustomDailies([daily]);registered.add(api);return {status:'registered',keys:[daily.key]};}
 function index(){if(indexed)return;indexed=true;for(const a of values(game.actors).filter(a=>a.type==='character'))for(const request of values(a.items).filter(i=>own(i)?.kind==='preparation'))tracked.set(request.uuid,request);}
 async function maintain(){if(!isActiveGM(game))return;index();
  for(const [uuid,request]of tracked){if(request.actor?.items.get(request.id)!==request){tracked.delete(uuid);continue;}const state=own(request),user=game.users.get(state.userId);if(terminal(state.status)||!allowed(request.actor,user)||session.running.has(uuid))continue;try{await prepare(uuid,user,{recoverOnly:true});}catch(error){report(error);}}
 }
 function register({Hooks=globalThis.Hooks,socket:api}={}){
  if(installed)return;installed=true;socket=api;hookApi=Hooks;index();
  socket?.register('elemental-medicine:prepare',async function(payload){try{const user=game.users.get(this.socketdata?.userId);return {ok:true,value:await prepare(payload?.requestUuid,user)};}catch(error){report(error);return {ok:false,error:'五气养生未取得完整诊疗回执，请由GM查看秘密诊疗记录。'};}});
  // Synchronous preCreate: attach proof before native publication, not afterward.
  hooks.push(['preCreateChatMessage',Hooks.on('preCreateChatMessage',(message,_data,_options,userId)=>{
   if(!isActiveGM(game)||userId!==game.user.id||author(message)!==game.user.id)return;const c=message.flags?.pf2e?.context;if(c?.type!=='skill-check'||message.actor?.uuid==null)return;
   // PF2e 8.5 converts a GM's secret check to "gm" mode. Preserve the strict
   // blind receipt contract on this exact pending check before it is published.
   for(const [nonce,args]of pendingChecks)if(message.actor.uuid===args.actor.uuid&&c.options?.includes(`${MODULE_ID}:elemental-medicine:${nonce}`)&&c.options.includes(`check:statistic:${args.skill}`)&&c.dc?.value===args.dc)message.updateSource({blind:true,whisper:gmIds(),'flags.pf2e.context.messageMode':'blind',[`flags.${MODULE_ID}.elementalMedicineCheck`]:args.proof});
  })]);
  hooks.push(['updateUser',Hooks.on('updateUser',()=>{if(isActiveGM(game))maintain().catch(report);})]);
  hooks.push(['createItem',Hooks.on('createItem',item=>{if(own(item)?.kind==='preparation')tracked.set(item.uuid,item);})]);
  hooks.push(['deleteItem',Hooks.on('deleteItem',item=>{tracked.delete(item.uuid);})]);
  hooks.push(['deleteActor',Hooks.on('deleteActor',actor=>{for(const [uuid,item]of tracked)if(item.actor?.uuid===actor.uuid)tracked.delete(uuid);})]);
  return()=>{for(const [name,id]of hooks)Hooks.off(name,id);installed=false;};
 }
 return {daily,prepare,registerDailies,register,maintain};
}
