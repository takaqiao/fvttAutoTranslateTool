import {createManualEvents,WORKBENCH_SOURCE_SHA,IMMUNITY_SOURCES} from '../../scripts/exploration/manual-events.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
export const M='pf2e-third-party-automation';
export const flush=async()=>{for(let i=0;i<12;i++)await new Promise(r=>setImmediate(r))};
export function manualEvidenceFixture(){
 let data={sessions:{S:{id:'S',status:'recording',actorUUIDs:['Actor.H','Actor.P'],activityIds:[]}},activities:{},clocks:{}};
 const users=new Map(['G','HUSER','PUSER','OUTSIDER'].map(id=>[id,{id,isGM:id==='G',active:true}]));users.activeGM=users.get('G');
 const forbidden=()=>{throw Error('read-only evidence must not execute native operations')};
 const actor=(id,owner)=>({id,uuid:`Actor.${id}`,items:new Map(),testUserPermission:u=>u?.isGM===true||u?.id===owner,createEmbeddedDocuments:forbidden,applyDamage:forbidden});
 const healer=actor('H','HUSER'),patient=actor('P','PUSER'),actors=new Map([[healer.uuid,healer],[patient.uuid,patient]]),messages=new Map(),handlers=new Map(),errors=[];
 const scenes=new Map();scenes.active={tokens:new Map([['T',{id:'T',actor:patient}]])};
 const game={user:users.get('G'),users,actors,time:{worldTime:0,advance:forbidden},messages,scenes};
 const Hooks={on:(n,f)=>{handlers.set(n,f);return n},off:n=>handlers.delete(n)};
 const ledger=createLedger({read:async()=>structuredClone(data),write:async s=>{data=structuredClone(s)},isAuthority:()=>true});
 const options={game,Hooks,ledger,fromUuid:async uuid=>actors.get(uuid),isAuthority:()=>true,sessionId:()=> 'S',onChange:e=>{if(e.error)errors.push(e.error)}};
 const check={id:'C',author:users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true}],flags:{pf2e:{context:{options:['exploration-manual-use:U']}}}};
 const result={id:'W',author:users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true,formula:'{2d8[healing]}'}],flags:{[M]:{explorationManual:{lexicalSource:true,sourceSHA:WORKBENCH_SOURCE_SHA,useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P',kind:'treatment',checkIds:['C'],stageIds:[],riskySurgery:false}},treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:2,healing:9}}};
 messages.set('C',check);messages.set('W',result);
 const event={id:'W',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],kind:'treatment',durationSeconds:600,useId:'U',checkIds:['C'],resultIds:['W'],source:{type:'workbench'},missing:['native-application-receipt','native-immunity-receipt']};
 const receipt=(id='R',author='PUSER')=>({id,author:users.get(author),speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[`${M}:source:W:0`]},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}});
 const immunity=(id='I')=>({id,uuid:`Actor.P.Item.${id}`,actor:patient,parent:patient,type:'effect',sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5',flags:{[M]:{explorationManualImmunity:{messageId:'W',useId:'U',patientUUID:'Actor.P',kind:'treatment',sourceSHA:IMMUNITY_SOURCES['XDY DO_NOT_IMPORT TW Immunity CD'].sha,creatorId:'PUSER'}}}});
 return {game,ledger,users,healer,patient,actors,messages,handlers,errors,options,event,check,result,receipt,immunity,createRecorder:()=>createManualEvents(options),fire:async message=>{messages.set(message.id,message);handlers.get('createChatMessage')?.(message);await flush()}};
}
