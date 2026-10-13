import {MODULE_ID,emptyLedger} from './schema.mjs';
import {canonicalJSON,normalizeLedger,projectRevisionPages,createGenesisRevision} from './revision-codec.mjs';
import {createRevisionStore} from './revision-store.mjs';
export const isActiveGM=game=>!!game.user?.isGM&&game.users?.activeGM?.id===game.user.id;
export const requireGMRead=game=>{if(game.user?.isGM!==true)throw Error('gm-read-required')};

export function createDocumentStore({game,writerClientId=globalThis.crypto.randomUUID(),nonce=()=>globalThis.crypto.randomUUID(),timeoutMs=15000}) {
  if(!Number.isSafeInteger(timeoutMs)||timeoutMs<=0)throw Error('invalid-document-timeout');
  let pinnedRoot=null,requiresReload=false;
  const setting=()=>game.settings.get(MODULE_ID,'explorationLedgerUUID');
  const authority=()=>{if(!isActiveGM(game))throw Error('active-gm-required')};
  async function readAsGM(operation){
    requireGMRead(game);const user=game.user,userId=user.id;
    try{return await operation()}
    finally{requireGMRead(game);if(game.user!==user||game.user.id!==userId)throw Error('document-user-changed')}
  }
  const validUUID=uuid=>{if(typeof uuid!=='string'||!/^JournalEntry\.[A-Za-z0-9]{16}$/.test(uuid))throw Error('invalid-revision-root')};
  function getRootUUID(){
    const root=setting();
    if(requiresReload||pinnedRoot&&root!==pinnedRoot)throw Error('revision-root-changed');
    if(root){validUUID(root);pinnedRoot??=root}
    return root;
  }
  function approvedRoot(raw){
    if(raw?.ownership?.default!==0)return false;
    return Object.entries(raw.ownership).every(([id,level])=>Number.isInteger(level)&&level>=-1&&level<=3&&(id==='default'||level<=0||game.users?.get(id)?.isGM===true));
  }
  function validateRoot(raw,uuid){
    if(!raw||`JournalEntry.${raw._id}`!==uuid)throw Error('root-document-mismatch');
    if(raw.ownership?.default!==0)throw Error('root-not-private');
    if(!approvedRoot(raw))throw Error('root-not-approved');
    if(!Array.isArray(raw.pages))throw Error('invalid-revision-pages');
  }
  function request(data,{write=false}={}){
    if(write)authority();else requireGMRead(game);
    const socket=game.socket,socketId=socket?.id,userId=game.user?.id;
    if(!socket?.connected||!socketId||!userId)throw Error('document-socket-disconnected');
    const wire=structuredClone(data);
    return new Promise((resolve,reject)=>{
      let finished=false;
      const complete=(error,response)=>{
        if(finished)return;
        finished=true;clearTimeout(timer);socket.off('disconnect',disconnected);
        if(error){reject(error);return}
        try{
          if(game.socket!==socket||!socket.connected||socket.id!==socketId)throw Error('document-socket-changed');
          if(game.user?.id!==userId)throw Error('document-user-changed');
          if(write)authority();else requireGMRead(game);
          resolve(structuredClone(response));
        }catch(error){reject(error)}
      };
      const disconnected=()=>complete(Error('document-socket-disconnected'));
      const timer=setTimeout(()=>complete(Error('document-acknowledgement-timeout')),timeoutMs);
      socket.on('disconnect',disconnected);
      try{socket.emit('modifyDocument',wire,response=>complete(null,response))}catch(error){complete(error)}
    });
  }
  function envelope(ack,type,action,userId){return ack?.type===type&&ack.action===action&&ack.userId===userId&&ack.broadcast===false&&!ack.results}
  function sameJSON(left,right){try{return canonicalJSON(left)===canonicalJSON(right)}catch{return false}}
  async function readRoot(rootUUID){
    requireGMRead(game);validUUID(rootUUID);const id=rootUUID.slice('JournalEntry.'.length),userId=game.user?.id;
    const ack=await request({type:'JournalEntry',action:'get',operation:{query:{_id:id},broadcast:false}});
    requireGMRead(game);
    if(!envelope(ack,'JournalEntry','get',userId)||ack.error||!Array.isArray(ack.result)||!sameJSON(ack.operation?.query,{_id:id}))throw Error('root-read-acknowledgement-unknown');
    if(ack.result.length===0)throw Error('root-not-found');
    if(ack.result.length!==1)throw Error('root-read-acknowledgement-unknown');
    const raw=ack.result[0];validateRoot(raw,rootUUID);return raw;
  }
  const revisionsFor=rootSource=>createRevisionStore({getRootUUID:rootSource,readRoot,
    createPage:({rootUUID,page})=>request({type:'JournalEntryPage',action:'create',operation:{parentUuid:rootUUID,keepId:true,render:false,renderSheet:false,data:[page],broadcast:false}},{write:true}),
    isAuthority:()=>isActiveGM(game),canUseRoot:approvedRoot,writerUserId:()=>game.user?.id,writerClientId,nonce});
  const revisions=revisionsFor(getRootUUID);

  function setup(options){
    authority();
    if(['issuersStopped','clientsReloaded','recoveryDisabled'].some(key=>options?.[key]!==true))throw Error('setup-confirmations-required');
    if(requiresReload)throw Error('root-runtime-reload-required');
  }
  async function configure(rootUUID,before){
    authority();
    if(setting()!==before)throw Error('root-config-conflict');
    if(before!==rootUUID)await game.settings.set(MODULE_ID,'explorationLedgerUUID',rootUUID);
    authority();
    if(setting()!==rootUUID)throw Error('root-config-conflict');
    // Explicit replacement invalidates this runtime; it does not retarget its existing permits.
    requiresReload=!!(before&&before!==rootUUID)||!!(pinnedRoot&&pinnedRoot!==rootUUID);
    if(!requiresReload)pinnedRoot=rootUUID;
    return {rootUUID,requiresReload};
  }
  async function createRoot(){
    const userId=game.user.id,marker={schemaVersion:1,setupNonce:nonce()};
    const data={name:'探索活动账本',ownership:{default:0},flags:{[MODULE_ID]:{explorationLedger:emptyLedger(),explorationLedgerRoot:marker}}};
    const ack=await request({type:'JournalEntry',action:'create',operation:{render:false,renderSheet:false,data:[data],broadcast:false}},{write:true});
    if(!envelope(ack,'JournalEntry','create',userId)||ack.error||!Array.isArray(ack.result)||ack.result.length!==1)throw Error('root-create-acknowledgement-unknown');
    const raw=ack.result[0],rootUUID=`JournalEntry.${raw?._id}`;validUUID(rootUUID);validateRoot(raw,rootUUID);
    if(raw.name!==data.name||raw.pages.length||!sameJSON(raw.flags?.[MODULE_ID],data.flags[MODULE_ID]))throw Error('root-create-acknowledgement-mismatch');
    return rootUUID;
  }
  async function provision(options){
    setup(options);const before=setting();
    if(before)throw Error('root-already-configured');
    if(pinnedRoot)throw Error('revision-root-changed');
    const rootUUID=await createRoot();
    try{return {...await configure(rootUUID,before),initialized:false}}
    catch(error){error.rootUUID=rootUUID;throw error}
  }
  // Starts an empty ledger beside the configured one, which is never written and stays as the archive.
  // Genesis goes into the new root before the setting moves, so the next runtime opens it initialized instead of as legacy data.
  async function replace(options){
    setup(options);const before=setting();
    if(!before)throw Error('root-not-configured');
    const rootUUID=await createRoot();
    try{
      const epoch=nonce(),genesis=await createGenesisRevision({rootUUID,epoch,nonce:nonce(),writerUserId:game.user.id,writerClientId:typeof writerClientId==='function'?writerClientId():writerClientId,seed:emptyLedger()});
      await revisionsFor(()=>rootUUID).initialize({epoch,expectedSourceDigest:genesis.sourceDigest,allowStoppedLegacy:true});
      return {...await configure(rootUUID,before),previousRootUUID:before,initialized:true};
    }catch(error){error.rootUUID=rootUUID;throw error}
  }
  async function select(options){
    setup(options);const rootUUID=options.rootUUID,before=setting();
    const raw=await readRoot(rootUUID);
    const protocol=raw.pages.some(page=>(typeof page?._id==='string'&&page._id.startsWith('er'))||page?.flags?.[MODULE_ID]?.explorationRevision!==undefined);
    if(protocol)await projectRevisionPages({rootUUID,pages:raw.pages});
    else normalizeLedger(raw.flags?.[MODULE_ID]?.explorationLedger??emptyLedger());
    authority();
    return configure(rootUUID,before);
  }
  return {
    read:()=>readAsGM(()=>getRootUUID()?revisions.read():emptyLedger()),
    transact:revisions.transact,status:()=>readAsGM(revisions.status),inspect:()=>readAsGM(revisions.inspect),
    initialize:async options=>{setup(options);return revisions.initialize({...options,allowStoppedLegacy:options.allowStoppedLegacy===true})},
    provision,select,replace
  };
}
