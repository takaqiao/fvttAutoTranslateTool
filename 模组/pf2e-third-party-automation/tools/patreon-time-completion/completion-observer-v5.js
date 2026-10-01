const __patreonTimeCompletion=(()=>{
 const descriptor=Object.freeze({version:2,markedCommitOwnership:"private-prepare.v1",providerId:"patreon-v3",providerVersion:"3.2.29",baseSourceSHA256:"89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9",pf2eSourceSHA256:"d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157"});
 const observers=new Map();let owned=false;
 function json(value){
  function check(v){if(v===null||typeof v==="string"||typeof v==="boolean"||v===undefined)return;if(typeof v==="number"&&Number.isFinite(v))return;if(typeof v!=="object"||(!Array.isArray(v)&&v.constructor?.name!=="Object"))throw Error("time-completion-json-unavailable");for(const x of Object.values(v))check(x);}
  check(value);return value===undefined?null:JSON.parse(JSON.stringify(value));
 }
 function frozen(value){if(value&&typeof value==="object"){for(const x of Object.values(value))frozen(x);Object.freeze(value);}return value;}
 function snapshot(inv,value){try{return json(value);}catch(error){inv.errors.push(String(error));return null;}}
 function begin(worldTime,delta,options,userId){
  const inv={invocation:{invocationId:crypto.randomUUID(),worldTime,delta,userId,handlerUserId:game.user.id,activeGMId:game.users.activeGM?.id},members:[],rules:[],rolls:[],writes:[],errors:[],ruleIndex:new Map(),sequence:0};
  inv.invocation.options=snapshot(inv,options);snapshot(inv,inv.invocation);return inv;
 }
 function rule(inv,value){
  if(!inv.ruleIndex.has(value)){const entry={sequence:++inv.sequence,actorUUID:value.actor?.uuid??value.item?.actor?.uuid??null,itemUUID:value.item?.uuid??null,sourceId:value.item?.sourceId??null,sourceIndex:value.sourceIndex??null,resolved:[]};inv.rules.push(entry);inv.ruleIndex.set(value,entry);}return inv.ruleIndex.get(value);
 }
 function predicate(inv,value,passing){if(inv===undefined)return passing;rule(inv,value).passing=snapshot(inv,passing);return passing;}
 function resolved(inv,value,result,phase){if(inv===undefined)return result;rule(inv,value).resolved.push({sequence:++inv.sequence,phase,value:snapshot(inv,result)});return result;}
 function member(inv,id,actor){if(inv===undefined)return;inv.members.push({sequence:++inv.sequence,actorId:id,actorUUID:actor?.uuid??null});}
 function roll(inv,value,result){if(inv===undefined)return result;let source=null;try{source=snapshot(inv,result.toJSON());}catch(error){inv.errors.push(String(error));}inv.rolls.push({sequence:++inv.sequence,actorUUID:value.actor?.uuid??value.item?.actor?.uuid??null,itemUUID:value.item?.uuid??null,sourceIndex:value.sourceIndex??null,roll:source});return result;}
 function track(inv,kind,document,data,promise){
  const entry={sequence:++inv.sequence,kind,documentUUID:document.uuid??null,data:snapshot(inv,data),status:"pending"};inv.writes.push(entry);
  if(typeof entry.documentUUID!=="string"||!entry.documentUUID)inv.errors.push("native-document-source-unavailable");
  if(!promise||typeof promise.then!=="function"){entry.status="unproven";entry.error="native-promise-unavailable";inv.errors.push(entry.error);entry.settlement=Promise.resolve();return promise;}
  entry.settlement=Promise.resolve(promise).then(()=>{entry.status="fulfilled";},error=>{entry.status="rejected";entry.error=String(error);});return promise;
 }
 function update(inv,document,data,kind){if(inv===undefined)return document.update(data);return track(inv,kind,document,data,document.update(data));}
 function decrease(inv,document){if(inv===undefined)return document.decrease();return track(inv,"effect-decrease",document,null,document.decrease());}
 function branches(tasks,inv){if(inv===undefined)return;return(async()=>{const results=await Promise.allSettled(tasks);const failed=results.find(r=>r.status==="rejected");if(failed)throw failed.reason;})();}
 async function finish(inv,work){
  const results=await Promise.allSettled(work);await Promise.all(inv.writes.map(w=>w.settlement));
  const proof=frozen({descriptor,invocation:snapshot(inv,inv.invocation),members:snapshot(inv,inv.members),rules:snapshot(inv,inv.rules),rolls:snapshot(inv,inv.rolls),writes:snapshot(inv,inv.writes.map(({settlement,...entry})=>entry)),errors:[...inv.errors]});
  const failed=results.find(r=>r.status==="rejected");if(failed||inv.errors.length||inv.writes.some(w=>w.status!=="fulfilled")){const error=Error(inv.errors[0]??String(failed?.reason??"native-write-rejected"));error.proof=proof;throw error;}return proof;
 }
 function authorize(inv){
  const options=inv.invocation.options,namespace=options?.pf2eThirdPartyAutomation;
  if(!namespace||!Object.prototype.hasOwnProperty.call(namespace,"exploration"))return true;
  if(inv.errors.length)return false;
  const invocation=frozen(snapshot(inv,inv.invocation)),approved=[];
  let failed=false;
  for(const entry of [...observers.values()])if(entry.gate)try{if(entry.gate(invocation)===true)approved.push(entry);}catch{failed=true;}
  return !failed&&approved.length===1&&observers.get(approved[0].observer)===approved[0];
 }
 function publish(inv,work){
  const terminalPromise=finish(inv,work);terminalPromise.catch(()=>{});
  const event=Object.freeze({descriptor,invocation:frozen(snapshot(inv,inv.invocation)),terminalPromise});
  for(const observer of [...observers.keys()])try{observer(event);}catch{}
 }
 function install(){
  const module=game.modules.get("patreon-v3");if(!module)throw Error("time-completion-module-unavailable");const api=module.api===undefined?{}:module.api;
  if(api===null||typeof api!=="object"||Array.isArray(api)||"explorationTimeCompletion"in api)throw Error("time-completion-api-collision");
  const observerAPI=Object.freeze({descriptor,subscribe(observer,{authorizeMarkedCommit}={}){if(typeof observer!=="function")throw TypeError("time-completion-observer-required");if(authorizeMarkedCommit!==undefined&&typeof authorizeMarkedCommit!=="function")throw TypeError("time-completion-gate-required");let entry=observers.get(observer);if(entry&&entry.gate!==authorizeMarkedCommit)throw Error("time-completion-subscription-conflict");if(!entry){entry={observer,gate:authorizeMarkedCommit};observers.set(observer,entry);}return()=>{if(observers.get(observer)===entry)observers.delete(observer);};}});
  Object.defineProperty(api,"explorationTimeCompletion",{value:observerAPI,enumerable:true});if(module.api===undefined)module.api=api;
  owned=module.api.explorationTimeCompletion===observerAPI;
 }
 return {begin,predicate,resolved,member,roll,update,decrease,branches,authorize,publish,install,owned:()=>owned};
})();
Hooks.once("init",()=>__patreonTimeCompletion.install());
