import {isActiveGM} from './document-store.mjs';
import {createGenesisRevision} from './revision-codec.mjs';

export function createLedgerSetup({game,store,ledger,writerClientId}){
 const authority=()=>{if(!isActiveGM(game))throw Error('active-gm-required')};
 function controlled(options){authority();if(['issuersStopped','clientsReloaded','recoveryDisabled'].some(key=>options?.[key]!==true))throw Error('setup-confirmations-required')}
 const requiresQuarantine=seed=>Object.values(seed.sessions).some(session=>session.manual!==true&&session.status==='running'&&!session.protocol&&!session.driver);
 async function describe(value){return {...value,state:!value.initialized?'legacy':requiresQuarantine(await store.read())?'migration-required':'ready'}}
 async function status(){try{return await describe(await store.status())}catch(error){return error.message==='root-not-configured'?{state:'unconfigured'}:{state:'blocked',reason:error.message}}}
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
 return {status,initialize,provision:async options=>{controlled(options);return store.provision(options)},select:async options=>{controlled(options);return store.select(options)}};
}
