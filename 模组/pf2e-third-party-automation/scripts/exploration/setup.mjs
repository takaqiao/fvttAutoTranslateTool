import {isActiveGM} from './document-store.mjs';
import {createGenesisRevision} from './revision-codec.mjs';
import {clockBlocksRecovery} from './current-time-recovery.mjs';

export function createLedgerSetup({game,store,ledger,writerClientId}){
 const authority=()=>{if(!isActiveGM(game))throw Error('active-gm-required')};
 function controlled(options){authority();if(['issuersStopped','clientsReloaded','recoveryDisabled'].some(key=>options?.[key]!==true))throw Error('setup-confirmations-required')}
 const requiresQuarantine=seed=>Object.values(seed.sessions).some(session=>session.manual!==true&&session.status==='running'&&!session.protocol&&!session.driver);
 async function describe(value){return {...value,state:!value.initialized?'legacy':requiresQuarantine(await store.read())?'migration-required':'ready'}}
 async function status(){try{return await describe(await store.status())}catch(error){return error.message==='root-not-configured'?{state:'unconfigured'}:{state:'blocked',reason:error.message}}}
 async function inspect(){try{const {status:value,state}=await store.inspect();return {ledgerStatus:{...value,state:!value.initialized?'legacy':requiresQuarantine(state)?'migration-required':'ready'},ledgerState:state}}catch(error){return {ledgerStatus:error.message==='root-not-configured'?{state:'unconfigured'}:{state:'blocked',reason:error.message},ledgerState:null}}}
 async function quarantine(value){
  const seed=await store.read();authority();
  if(!requiresQuarantine(seed))return {...value,state:'ready'};
  if(typeof ledger?.quarantineLegacySessions!=='function')throw Error('legacy-quarantine-unavailable');
  await ledger.quarantineLegacySessions();authority();
  const result=await describe(await store.status());authority();return result;
 }
 async function initialize(options){
  controlled(options);const current=await store.status();authority();
  if(current.initialized)return quarantine(current);
  const seed=await store.read();authority();
  const epoch=crypto.randomUUID(),userId=game.user.id,clientId=typeof writerClientId==='function'?writerClientId():writerClientId;
  const genesis=await createGenesisRevision({rootUUID:current.rootUUID,epoch,nonce:crypto.randomUUID(),writerUserId:userId,writerClientId:clientId,seed});
  authority();if(game.user.id!==userId)throw Error('setup-user-changed');
  const initialized=await store.initialize({...options,epoch,expectedSourceDigest:genesis.sourceDigest,allowStoppedLegacy:true});
  authority();return quarantine(initialized);
 }
 // Replacement abandons whatever the ledger still holds open, so it needs a ready ledger that nothing is running or recording in.
 async function replaceable(){
  authority();const {status:value,state}=await store.inspect();authority();
  if(!value.initialized||requiresQuarantine(state))throw Error('replace-ledger-ready-required');
  if(Object.values(state.sessions).some(session=>['running','recording'].includes(session.status)))throw Error('replace-ledger-session-active');
  return state;
 }
 // What stays behind in the archive: sessions that never reached complete or closed, and clocks that would block a new recovery.
 async function replacePreview(){
  const state=await replaceable();
  return {unfinishedSessions:Object.values(state.sessions).filter(session=>!['complete','closed'].includes(session.status)).length,unconfirmedClocks:Object.values(state.clocks).filter(clock=>clockBlocksRecovery(state,clock)).length};
 }
 async function replace(options){controlled(options);await replaceable();return store.replace(options)}
 return {status,inspect,initialize,provision:async options=>{controlled(options);return store.provision(options)},select:async options=>{controlled(options);return store.select(options)},replacePreview,replace};
}
