import {authorityFixture} from './authority-fixture.mjs';
import {seamFixture} from '../../tools/toolbelt-manual-pool/seam.test.mjs';
import {createHpPools} from '../../scripts/exploration/hp-pool.mjs';
import {createManualPoolCompletion} from '../../scripts/exploration/manual-pool-completion.mjs';

export async function fixture(remote,options={}){
 const seam=seamFixture({remote}),storage=await authorityFixture(),ledger=storage.client('G1');
 await ledger.createSession({id:'S',manual:true,status:'recording',actorUUIDs:['Actor.H','Actor.P'],startedAt:0,budgetEndsAt:600});
 const activityId=options.activityId??'A',sourceType=options.sourceType??'native-action';
 await ledger.insertActivity({id:activityId,sessionId:'S',providerId:'manual',kind:'treatment',durationSeconds:600,treatmentImmunitySeconds:3600,actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.M'],startedAt:0,endsAt:600,state:'awaiting-evidence',source:{manual:true,type:sourceType,messageId:sourceType==='workbench'?'R':'C',tag:'tagU'},options:{effectiveOutcome:'success',missing:['native-application-receipt','native-immunity-receipt','shared-hp-completion-unavailable']},proof:{useId:'U',checkIds:['C'],resultIds:['R']}});
 const clients=[seam.gm,seam.owner],messages=new Map(),packets=[],errors=[];let valid=true,dropTerminal=false;
 for(const id of ['C','R'])messages.set(id,{id,author:'O',speaker:{actor:'H'},rolls:[{_evaluated:true,total:20}],toObject(){return {id,author:this.author,speaker:this.speaker,rolls:this.rolls}}});
 const request={sessionId:'S',activityId,actorUUID:'Actor.H',sourceType,useId:'U',checkId:'C',resultId:'R',rollIndex:0,stage:'healing',poolUUID:'Actor.M',patientUUIDs:['Actor.P'],batchId:'batch',ownerClientNonce:'O1',attemptNonce:'attempt'};
 const binding={...request,effectId:'R',selectedPatientUUID:'Actor.P',worldTime:0};delete binding.ownerClientNonce;delete binding.attemptNonce;
 for(const c of clients){
  const {game,master,patient}=c;c.listeners=new Set();game.time={worldTime:0};game.messages=messages;game.settings={get:()=>true};game.modules.get('pf2e-toolbelt').active=true;
  game.actors.set(patient.id,patient);
  const healer={id:'H',uuid:'Actor.H',testUserPermission:()=>true};game.actors.set(healer.id,healer);patient.testUserPermission=()=>true;master.testUserPermission=user=>user.id==='G'||!remote;
  master.system={attributes:{hp:{value:1,max:30,temp:0}}};patient.modules={'pf2e-toolbelt':{shareData:{data:{health:true}}}};
  game.toolbelt={api:{shareData:{getMasterInMemory:()=>master,getSlavesInMemory:()=>[patient]}}};
  const actors=new Map([['Actor.H',healer],['Actor.M',master],['Actor.P',patient]]);c.actors=actors;
  game.socket={on:(_,fn)=>c.listeners.add(fn),off:(_,fn)=>c.listeners.delete(fn),emit(channel,packet,options,ack){packets.push({sender:game.user.id,packet});ack?.({relayed:true});if(dropTerminal&&packet.kind==='completion-ack')return;for(const target of clients.filter(x=>options.recipients.includes(x.game.user.id)))for(const listener of target.listeners)listener(structuredClone(packet),game.user.id)}};
  c.pools=createHpPools({game,actorUpdateEvents:{addActorUpdateMiddleware:fn=>{c.update=fn}}});patient._preUpdate=(changes,options)=>c.tool.pre(patient,changes,options);
  c.broker=createManualPoolCompletion({game,ledger:game.user.id==='G'?ledger:undefined,clientNonce:game.user.id+'1',isIssuer:()=>game.user.id==='G',hpPools:c.pools,fromUuid:async uuid=>actors.get(uuid),
   resolveSource:async input=>{const b={...input,effectId:'R',selectedPatientUUID:'Actor.P',worldTime:0};delete b.ownerClientNonce;delete b.attemptNonce;return {binding:b,isCurrent:()=>valid}},getProvider:()=>c.api,timeoutMs:200,onError:e=>errors.push(e)});c.broker.start();
 }
 const receipt={id:'receipt',author:'O',speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:['pf2e-third-party-automation:source:R:0']},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}};
 let applications=0;
 const operation=async()=>{applications++;await seam.owner.update.call(seam.owner.patient,async(changes,options)=>{await Promise.resolve();await seam.owner.patient._preUpdate(changes,options);return seam.owner.patient},{'system.attributes.hp.value':20},{});messages.set(receipt.id,receipt);return {receipt}};
 return {...seam,...storage,ledger,request,receipt,messages,packets,errors,operation,applications:()=>applications,invalid:()=>{valid=false},loseTerminalAck:()=>{dropTerminal=true},close(){for(const c of clients){c.broker.stop();c.pools.dispose()}}};
}

