import {MODULE_ID,emptyLedger} from './schema.mjs';
import {canonicalJSON,normalizeLedger,revisionId,createGenesisRevision,createSuccessorRevision,projectRevisionPages,encodeRevision,decodeRevision} from './revision-codec.mjs';

export function createRevisionStore({getRootUUID,readRoot,createPage,isAuthority,canUseRoot,writerUserId,writerClientId,nonce,maxConflicts=8}) {
  let pinnedRoot,knownHead=null;const knownDigests=new Map();
  if(!Number.isSafeInteger(maxConflicts)||maxConflicts<0)throw Error('invalid-conflict-limit');

  function rootUUID(){
    const root=getRootUUID();
    if(pinnedRoot&&root!==pinnedRoot)throw Error('revision-root-changed');
    if(!root)throw Error('root-not-configured');
    if(typeof root!=='string'||!/^JournalEntry\.[A-Za-z0-9]{16}$/.test(root))throw Error('invalid-revision-root');
    pinnedRoot??=root;return root;
  }
  function authority(){if(isAuthority()!==true)throw Error('active-gm-required');rootUUID()}
  const clientId=()=>typeof writerClientId==='function'?writerClientId():writerClientId;
  function identity(){return {writerUserId:writerUserId(),writerClientId:clientId(),nonce:nonce()}}
  function checkWriter(writer){authority();if(writerUserId()!==writer.writerUserId||clientId()!==writer.writerClientId)throw Error('revision-writer-changed')}
  function validateRoot(raw,root){
    if(!raw||`JournalEntry.${raw._id}`!==root)throw Error('root-document-mismatch');
    if(raw.ownership?.default!==0)throw Error('root-not-private');
    if(canUseRoot?.(raw)!==true)throw Error('root-not-approved');
    if(!Array.isArray(raw.pages))throw Error('invalid-revision-pages');
  }
  function remember(head,chain=[head]){
    if(knownHead){
      if(head.rootUUID!==knownHead.rootUUID||head.epoch!==knownHead.epoch)throw Error('known-head-provenance-mismatch');
      const common=Math.min(head.revision,knownHead.revision);
      const digest=chain.find(revision=>revision.revision===common)?.digest;
      const nextAcknowledgement=head.revision===knownHead.revision+1&&head.previousDigest===knownHead.digest;
      if(digest!==knownDigests.get(common)&&!nextAcknowledgement)throw Error('known-head-replaced');
    }
    for(const revision of chain)if(knownDigests.has(revision.revision)&&knownDigests.get(revision.revision)!==revision.digest)throw Error('known-head-replaced');
    for(const revision of chain)knownDigests.set(revision.revision,revision.digest);
    // Overlapping reads and exact older ACKs may finish later without lowering the high-water mark.
    if(!knownHead||head.revision>knownHead.revision)knownHead=structuredClone(head);
  }
  async function load(){
    const root=rootUUID(),baseline=knownHead,raw=structuredClone(await readRoot(root));rootUUID();validateRoot(raw,root);
    const protocol=raw.pages.some(page=>page?._id?.startsWith('er')||page?.flags?.[MODULE_ID]?.explorationRevision!==undefined);
    if(!protocol){
      if(baseline)throw Error('known-head-regression');
      return {rootUUID:root,raw,state:normalizeLedger(raw.flags?.[MODULE_ID]?.explorationLedger??emptyLedger()),head:null,genesis:null};
    }
    const {state,head}=await projectRevisionPages({rootUUID:root,pages:raw.pages,knownHead:baseline});
    rootUUID();remember(head,raw.pages.map(page=>page.flags?.[MODULE_ID]?.explorationRevision).filter(revision=>revision!==undefined).map(decodeRevision));
    const genesis=decodeRevision(raw.pages.find(page=>page._id===revisionId(0)).flags[MODULE_ID].explorationRevision);
    return {rootUUID:root,raw,state,head,genesis};
  }
  function describe(loaded){return {initialized:!!loaded.head,rootUUID:loaded.rootUUID,epoch:loaded.head?.epoch??null,revision:loaded.head?.revision??null,sourceDigest:loaded.genesis?.sourceDigest??null}}
  function pageFor(metadata){return {_id:revisionId(metadata.revision),name:`Exploration revision ${metadata.revision}`,type:'text',ownership:{default:0},flags:{[MODULE_ID]:{explorationRevision:encodeRevision(metadata)}}}}
  function sameEnvelope(ack,writer,root){return ack?.type==='JournalEntryPage'&&ack.action==='create'&&ack.broadcast===false&&ack.operation?.parentUuid===root&&ack.userId===writer.writerUserId}
  function duplicate(ack,writer,root,page){
    return sameEnvelope(ack,writer,root)&&ack.error?.class==='ServerError'&&(!ack.result||Array.isArray(ack.result)&&ack.result.length===0)&&ack.error.message===`The _id [${page._id}] already exists within the parent collection: JournalEntry [${root.slice('JournalEntry.'.length)}] pages`;
  }
  function validateAcknowledgement(ack,writer,root,page){
    if(!sameEnvelope(ack,writer,root)||ack.error||!Array.isArray(ack.result)||ack.result.length!==1)throw Error('revision-acknowledgement-unknown');
    const saved=ack.result[0];
    // Foundry adds native defaults and stats. The signed protocol content must be unchanged.
    if(saved?._id!==page._id||saved.type!==page.type||saved.name!==page.name||saved.ownership?.default!==0||canonicalJSON(saved.flags?.[MODULE_ID])!==canonicalJSON(page.flags[MODULE_ID]))throw Error('revision-acknowledgement-mismatch');
  }
  async function submit(loaded,metadata,writer){
    checkWriter(writer);validateRoot(loaded.raw,loaded.rootUUID);
    const page=pageFor(metadata);
    // The injected transport performs one keepId:true, broadcast:false create, and rejects unknown timeouts.
    const ack=await createPage({rootUUID:loaded.rootUUID,page:structuredClone(page)});
    checkWriter(writer);validateRoot(loaded.raw,loaded.rootUUID);
    if(duplicate(ack,writer,loaded.rootUUID,page))return false;
    validateAcknowledgement(ack,writer,loaded.rootUUID,page);remember(metadata);return true;
  }
  function matchGenesis(loaded,{epoch,expectedSourceDigest}){
    if(loaded.head.epoch!==epoch)throw Error('genesis-epoch-conflict');
    if(loaded.genesis.sourceDigest!==expectedSourceDigest)throw Error('genesis-source-conflict');
    return describe(loaded);
  }
  async function initialize({epoch,expectedSourceDigest,allowStoppedLegacy=false}){
    authority();const loaded=await load();authority();
    if(loaded.head)return matchGenesis(loaded,{epoch,expectedSourceDigest});
    if(allowStoppedLegacy!==true&&Object.values(loaded.state.sessions).some(session=>session.status==='running'&&session.manual!==true))throw Error('legacy-automatic-session-running');
    const writer=identity(),metadata=await createGenesisRevision({rootUUID:loaded.rootUUID,epoch,...writer,seed:loaded.state});
    if(metadata.sourceDigest!==expectedSourceDigest)throw Error('genesis-source-conflict');
    if(await submit(loaded,metadata,writer))return describe({...loaded,head:metadata,genesis:metadata});
    // Explicit setup may recognize another initializer's same seed. This never returns an executable grant.
    const current=await load();authority();if(!current.head)throw Error('protocol-not-initialized');return matchGenesis(current,{epoch,expectedSourceDigest});
  }
  async function transact(fn){
    if(typeof fn!=='function'||fn.constructor?.name==='AsyncFunction')throw Error('synchronous-mutation-required');
    for(let conflicts=0;;conflicts++){
      authority();const loaded=await load();authority();if(!loaded.head)throw Error('protocol-not-initialized');
      const context=Object.freeze({rootUUID:loaded.rootUUID,epoch:loaded.head.epoch,revision:loaded.head.revision+1,previousDigest:loaded.head.digest});
      const nextState=structuredClone(loaded.state),value=fn(nextState,context);
      if(value&&typeof value.then==='function')throw Error('synchronous-mutation-required');
      const result=structuredClone(value),writer=identity();
      const metadata=await createSuccessorRevision({previous:loaded.head,state:loaded.state,nextState,...writer});
      if(await submit(loaded,metadata,writer))return result;
      if(conflicts>=maxConflicts)throw Error('revision-conflict-limit');
    }
  }
  return {read:async()=>structuredClone((await load()).state),status:async()=>describe(await load()),initialize,transact};
}
