import {MODULE_ID} from './lifecycle.mjs';
/** Local capabilities exist only while an awaited actual-use entry is running.
 * toMessage/drafts do not create them. The GM separately verifies persisted
 * nonce, owner, actor, source and original card before changing armed state. */
export function createMetapowerObserver({request,select=async()=>({}),id=()=>globalThis.crypto.randomUUID(),onError=console.error}){
 const scopes=new Map();
 return {
  active:item=>scopes.get(item?.uuid)??null,
  async observe({actor,item=null,entry='item'},native){
   const selection=item?await select(item):{};if(selection===null)return null;
   const payload={actorUuid:actor.uuid,itemUuid:item?.uuid??null,nonce:id(),selection,entry};
   const receipt=await request('begin',payload);
   if(receipt.status!=='reserved')throw Error('This invocation already ran; native execution will not be replayed.');
   const scope={receipt,messages:[],item,actor};
   let started=false,finished=false;
   try{
    await request('start',payload);started=true;if(item)scopes.set(item.uuid,scope);
    const result=await native();
    const message=scope.messages.length===1?scope.messages[0]:null;
    const noCheck=entry==='native-check'&&(result===null||Array.isArray(result)&&result.length===0);
    const status=noCheck?'cancelled':item?(message?'committed':'uncertain'):'committed';
    await request('finish',{...payload,status,messageUuid:message?.uuid??null,...(noCheck?{confirmation:'native-check-no-result'}:{})});finished=true;
    return result;
   }catch(error){
    if(!finished)await request('finish',{...payload,status:started?'uncertain':'cancelled'}).catch(onError);
    throw error;
   }finally{if(item&&scopes.get(item.uuid)===scope)scopes.delete(item.uuid)}
  },
  decorate(data){
   const itemUuid=data.flags?.pf2e?.origin?.uuid,scope=scopes.get(itemUuid);
   if(!scope||data.speaker?.actor!==scope.actor.id||data.rolls?.length||data.flags?.pf2e?.context?.type==='damage-roll')return data;
   const prior=data.flags?.[MODULE_ID]?.metapowerUse;
   if(prior&&prior.nonce!==scope.receipt.nonce)throw Error('A copied card cannot become a new native metapower use.');
   data.flags??={};data.flags[MODULE_ID]={...data.flags[MODULE_ID],metapowerUse:{nonce:scope.receipt.nonce,actorUuid:scope.actor.uuid,itemUuid, snapshot:scope.receipt.snapshot??null,kind:scope.receipt.kind??null}};
   return data;
  },
  record(messages){for(const message of messages??[]){const proof=message.flags?.[MODULE_ID]?.metapowerUse,scope=scopes.get(proof?.itemUuid);if(scope&&message.id&&proof.nonce===scope.receipt.nonce&&!scope.messages.some(m=>m.id===message.id))scope.messages.push(message)}},
 };
}
