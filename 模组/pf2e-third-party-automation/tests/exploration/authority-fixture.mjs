import {createHash} from 'node:crypto';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createRevisionStore} from '../../scripts/exploration/revision-store.mjs';
import {canonicalJSON} from '../../scripts/exploration/revision-codec.mjs';
import {MODULE_ID,emptyLedger} from '../../scripts/exploration/schema.mjs';

export async function authorityFixture(seed=emptyLedger()){
 const rootUUID='JournalEntry.ROOT000000000001',raw={_id:'ROOT000000000001',ownership:{default:0},pages:[],flags:{[MODULE_ID]:{explorationLedger:structuredClone(seed)}}};
 let serial=0,acknowledgement=ack=>ack;
 function storage(clientNonce,userId='G'){
  return createRevisionStore({getRootUUID:()=>rootUUID,readRoot:async()=>structuredClone(raw),canUseRoot:()=>true,isAuthority:()=>true,writerUserId:()=>userId,writerClientId:clientNonce,nonce:()=>`nonce-${++serial}`,createPage:async({rootUUID,page})=>{
   const envelope={type:'JournalEntryPage',action:'create',broadcast:false,userId,operation:{parentUuid:rootUUID}};
   if(raw.pages.some(p=>p._id===page._id))return {...envelope,error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}};
   raw.pages.push(structuredClone(page));return acknowledgement({...envelope,result:[structuredClone(page)]});
  }});
 }
 const initial=storage('setup');await initial.initialize({epoch:'epoch',expectedSourceDigest:createHash('sha256').update(canonicalJSON(seed)).digest('hex')});
 return {raw,rootUUID,read:initial.read,setAcknowledgement:fn=>{acknowledgement=fn},storage,client:(clientNonce,userId='G')=>createLedger({...storage(clientNonce,userId),isAuthority:()=>true,identity:()=>({userId,clientNonce})})};
}
