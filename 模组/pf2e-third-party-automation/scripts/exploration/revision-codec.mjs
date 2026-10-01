import {MODULE_ID} from './schema.mjs';

const collections=['sessions','activities','clocks'];
const forbiddenKeys=new Set(['__proto__','constructor','prototype']);
const metadataKeys=['schemaVersion','rootUUID','epoch','revision','previousDigest','nonce','writerUserId','writerClientId','digest'];
const digestPattern=/^[0-9a-f]{64}$/;

function objectKeys(value) {
  if(!value||typeof value!=='object'||![Object.prototype,null].includes(Object.getPrototypeOf(value)))throw Error('invalid-json-record');
  const keys=Reflect.ownKeys(value);
  for(const key of keys){
    if(typeof key!=='string'||forbiddenKeys.has(key))throw Error('invalid-json-key');
    const descriptor=Object.getOwnPropertyDescriptor(value,key);
    if(!descriptor.enumerable||!Object.hasOwn(descriptor,'value'))throw Error('unsupported-json-property');
  }
  return keys;
}

export function canonicalJSON(value) {
  const ancestors=new Set();
  function encode(input){
    if(input===null||typeof input==='string'||typeof input==='boolean')return JSON.stringify(input);
    if(typeof input==='number'){
      if(!Number.isFinite(input))throw Error('invalid-json-number');
      return JSON.stringify(input);
    }
    if(typeof input!=='object')throw Error('unsupported-json-value');
    if(ancestors.has(input))throw Error('cyclic-json-value');
    ancestors.add(input);
    try{
      if(Array.isArray(input)){
        if(Object.getPrototypeOf(input)!==Array.prototype)throw Error('unsupported-json-array');
        const keys=Reflect.ownKeys(input);
        if(keys.length!==input.length+1)throw Error('invalid-dense-json-array');
        const entries=[];
        for(let index=0;index<input.length;index++){
          const descriptor=Object.getOwnPropertyDescriptor(input,String(index));
          if(!descriptor?.enumerable||!Object.hasOwn(descriptor,'value'))throw Error('invalid-dense-json-array');
          entries.push(encode(descriptor.value));
        }
        return '['+entries.join(',')+']';
      }
      const entries=[];
      for(const key of objectKeys(input).sort())if(input[key]!==undefined)entries.push(JSON.stringify(key)+':'+encode(input[key]));
      return '{'+entries.join(',')+'}';
    }finally{ancestors.delete(input)}
  }
  return encode(value);
}

const copy=value=>JSON.parse(canonicalJSON(value));

export function normalizeLedger(state) {
  for(const key of objectKeys(state))if(!collections.includes(key))throw Error('unsupported-ledger-collection');
  const normalized={};
  for(const name of collections){
    const records=Object.hasOwn(state,name)?state[name]:{};
    normalized[name]={};
    for(const key of objectKeys(records)){
      if(!key.trim())throw Error('invalid-ledger-record-id');
      objectKeys(records[key]);
      normalized[name][key]=copy(records[key]);
    }
  }
  return normalized;
}

export function revisionId(revision) {
  if(!Number.isSafeInteger(revision)||revision<0)throw Error('invalid-revision');
  return 'er'+revision.toString(36).padStart(14,'0');
}

function validateRoot(rootUUID){
  if(typeof rootUUID!=='string'||!/^JournalEntry\.[A-Za-z0-9]{16}$/.test(rootUUID))throw Error('invalid-revision-root');
}

function validateMetadata(metadata,{signed=true}={}){
  const keys=objectKeys(metadata);
  const payloadKeys=metadata.revision===0?['seed','sourceDigest']:['changes'];
  const required=[...metadataKeys.filter(key=>signed||key!=='digest'),...payloadKeys];
  if(keys.length!==required.length||required.some(key=>!Object.hasOwn(metadata,key)))throw Error('invalid-revision-metadata');
  if(metadata.schemaVersion!==1)throw Error('unsupported-revision-schema');
  validateRoot(metadata.rootUUID);
  revisionId(metadata.revision);
  for(const key of ['epoch','nonce','writerUserId','writerClientId'])if(typeof metadata[key]!=='string'||!metadata[key].trim())throw Error('invalid-revision-'+key);
  if(signed&&!digestPattern.test(metadata.digest))throw Error('invalid-revision-digest');
  if(metadata.revision===0){
    if(metadata.previousDigest!==null||!digestPattern.test(metadata.sourceDigest))throw Error('invalid-genesis-provenance');
    normalizeLedger(metadata.seed);
  }else{
    if(!digestPattern.test(metadata.previousDigest))throw Error('invalid-previous-digest');
    normalizeLedger(metadata.changes);
  }
}

// Foundry expands dotted keys and interprets operator keys inside object flags.
// A canonical string preserves the signed JSON without changing its logical schema.
export function encodeRevision(metadata) {
  validateMetadata(metadata);
  return canonicalJSON(metadata);
}

export function decodeRevision(value) {
  let metadata;
  if(typeof value==='string'){
    try{metadata=JSON.parse(value)}catch{throw Error('invalid-revision-json')}
    if(canonicalJSON(metadata)!==value)throw Error('noncanonical-revision-json');
  }else metadata=copy(value);
  validateMetadata(metadata);
  return metadata;
}

async function digest(value){
  const bytes=new TextEncoder().encode(canonicalJSON(value));
  const result=await globalThis.crypto.subtle.digest('SHA-256',bytes);
  return Array.from(new Uint8Array(result),byte=>byte.toString(16).padStart(2,'0')).join('');
}

async function verifyMetadata(metadata){
  validateMetadata(metadata);
  const {digest:expected,...body}=metadata;
  if(await digest(body)!==expected)throw Error('revision-digest-mismatch');
  if(metadata.revision===0&&await digest(normalizeLedger(metadata.seed))!==metadata.sourceDigest)throw Error('genesis-source-digest-mismatch');
}

export async function createGenesisRevision({rootUUID,epoch,nonce,writerUserId,writerClientId,seed}) {
  const normalized=normalizeLedger(seed);
  const body=copy({schemaVersion:1,rootUUID,epoch,revision:0,previousDigest:null,nonce,writerUserId,writerClientId,seed:normalized,sourceDigest:'0'.repeat(64)});
  validateMetadata(body,{signed:false});
  body.sourceDigest=await digest(normalized);
  return {...body,digest:await digest(body)};
}

export async function createSuccessorRevision({previous,state,nextState,nonce,writerUserId,writerClientId}) {
  const prior=copy(previous),current=normalizeLedger(state),next=normalizeLedger(nextState);
  const changes={sessions:{},activities:{},clocks:{}};
  for(const name of collections){
    for(const key of Object.keys(current[name]))if(!Object.hasOwn(next[name],key))throw Error('ledger-record-deletion-forbidden');
    for(const key of Object.keys(next[name]))if(!Object.hasOwn(current[name],key)||canonicalJSON(current[name][key])!==canonicalJSON(next[name][key]))changes[name][key]=next[name][key];
  }
  const body=copy({schemaVersion:1,rootUUID:prior.rootUUID,epoch:prior.epoch,revision:prior.revision+1,previousDigest:prior.digest,nonce,writerUserId,writerClientId,changes});
  validateMetadata(body,{signed:false});
  await verifyMetadata(prior);
  return {...body,digest:await digest(body)};
}

export async function projectRevisionPages({rootUUID,pages,knownHead=null}) {
  validateRoot(rootUUID);
  if(!Array.isArray(pages))throw Error('invalid-revision-pages');
  // Take a detached snapshot before the first asynchronous digest operation.
  const snapshot=copy(pages),known=knownHead===null?null:copy(knownHead),revisions=new Map();
  for(const page of snapshot){
    const stored=page?.flags?.[MODULE_ID]?.explorationRevision;
    const reserved=typeof page?._id==='string'&&page._id.startsWith('er');
    if(!reserved&&stored===undefined)continue;
    if(!reserved||!/^er[0-9a-z]{14}$/.test(page._id)||!stored)throw Error('invalid-revision-page');
    const metadata=decodeRevision(stored);
    if(metadata.rootUUID!==rootUUID)throw Error('revision-root-mismatch');
    if(page._id!==revisionId(metadata.revision))throw Error('revision-page-id-mismatch');
    if(revisions.has(metadata.revision))throw Error('duplicate-revision-page');
    revisions.set(metadata.revision,metadata);
  }
  if(!revisions.has(0))throw Error('protocol-not-initialized');
  const ordered=[...revisions.values()].sort((a,b)=>a.revision-b.revision),genesis=ordered[0];
  let state,head;
  for(let index=0;index<ordered.length;index++){
    const metadata=ordered[index];
    if(metadata.revision!==index)throw Error('revision-chain-gap');
    if(metadata.epoch!==genesis.epoch)throw Error('revision-epoch-mismatch');
    if(index>0&&metadata.previousDigest!==head.digest)throw Error('revision-predecessor-mismatch');
    await verifyMetadata(metadata);
    if(index===0)state=normalizeLedger(metadata.seed);
    else{
      const changes=normalizeLedger(metadata.changes);
      for(const name of collections)Object.assign(state[name],changes[name]);
    }
    head=metadata;
  }
  if(known){
    await verifyMetadata(known);
    if(known.rootUUID!==rootUUID||known.epoch!==genesis.epoch)throw Error('known-head-provenance-mismatch');
    if(known.revision>head.revision)throw Error('known-head-regression');
    if(revisions.get(known.revision)?.digest!==known.digest)throw Error('known-head-replaced');
  }
  return {state,head};
}
