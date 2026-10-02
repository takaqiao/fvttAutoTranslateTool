import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';

const MODULE='pf2e-third-party-automation';
const nativeRoot=process.env.FOUNDRY_NATIVE_APP_ROOT??'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const pf2eFile=process.env.PF2E_NATIVE_BUNDLE??'C:/Users/Taka/Desktop/fvtt/tmp/fortress-gap-audit-20260925/code/systems/pf2e/pf2e.mjs';
const poolsFile=process.env.EXPLORATION_HP_POOLS_UNDER_TEST;
const {createHpPools}=await import(poolsFile?pathToFileURL(poolsFile).href:'../../scripts/exploration/hp-pool.mjs');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const boundedTest=(name,run)=>test(name,{timeout:5000},run);
const foundryPins=[
 ['common/abstract/document.mjs','303e6bc84fbaa7d564dab144b1f59fd0a1f6584a14e872b293e74358ed9b49dc'],
 ['common/abstract/data.mjs','c1756403ea82839a82e0cb1358f48c4a5d3dc4ed991cc6bc0dd6aa82df4508e9'],
 ['common/abstract/backend.mjs','75fd7fc2907f3521eb2ff1751c517fd3f275a2547b944c376d9f1d66b6bdb5d4'],
 ['client/data/client-backend.mjs','8ffe114bca601980bf2f4582ce1f6e27c555d76586470f1a16ac23f6f021b9c5'],
 ['common/data/fields.mjs','b8a7eafb6063e201cc34137da26b081385ac731a4e52c4a7bf80b7c5ed6fb6c2'],
 ['common/utils/helpers.mjs','25e75a70ab539c6878108ab4369416bde6edb7aef8a629f91675cd6a2d11b006'],
 ['common/data/operators.mjs','0ee26ded57750615b6cb4033c8aa90c444224b41df940caba508dd036d5f3121'],
 ['common/primitives/array.mjs','42ee5e3775afe5bde2efb359b799970f91176da0f075d0d7005845084ef55460'],
 ['common/primitives/math.mjs','d8fc810f456e07c10bed3638010147dfcfe413fdedbe23f4b107f022d8acc663']
];
function section(source,start,end) {
 const first=source.indexOf(start),last=source.indexOf(end,first+start.length);
 assert.ok(first>=0&&last>first,`Native method boundaries changed: ${start}`);
 return source.slice(first,last);
}
let sources;
async function nativeSources() {
 sources??=(async()=>{
  const foundry={};
  for(const [file,pin] of foundryPins){
   const bytes=await readFile(join(nativeRoot,file));assert.equal(sha(bytes),pin,`Foundry 14.368 source changed: ${file}`);foundry[file]=bytes.toString('utf8');
  }
  const bytes=await readFile(pf2eFile);assert.equal(sha(bytes),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157','PF2e 8.5.1 source changed');
  const pf=bytes.toString('utf8'),methods=[
   section(pf,'\n\tasync applyDamage({','\n\tasync undoDamage('),
   section(pf,'\n\tcalculateHealthDelta(e) {','\n\tgetRollOptions('),
   section(pf,'\n\tgetContextualClone(e, t = []) {','\n\tasync applyAreaEffects(')
  ];
  for(const [method,pin] of methods.map((method,index)=>[method,[
   '81281413a285c54b53b6cce41b90e5526dfc74f3299c518eabdf2631d6967f50',
   '97a78efcdae37408faab1c2c38ee32c1b961b296e2d470abbbe99c14c3019bfa',
   'ff71de85c0603366bb5e15fafeca6e78ba3f3f449576b0ba9056c6ddc5daa3ac'
  ][index]]))assert.equal(sha(method),pin);
  const stack=section(pf,'var HIGHER_BONUS =','var StatisticModifier =');
  const importNative=file=>import(pathToFileURL(join(nativeRoot,file)).href);
  await importNative('common/primitives/array.mjs');await importNative('common/primitives/math.mjs');
  return {methods,foundry,stack,
   Document:(await importNative('common/abstract/document.mjs')).default,
   DataModel:(await importNative('common/abstract/data.mjs')).default,
   DatabaseBackend:(await importNative('common/abstract/backend.mjs')).default,
   fields:await importNative('common/data/fields.mjs'),utils:await importNative('common/utils/helpers.mjs')};
 })();
 return sources;
}

async function fixture(t,{rawMax=36,preparedMax=36,value=36,preUpdate}={}) {
 const native=await nativeSources(),{Document,DataModel,DatabaseBackend,fields,utils}=native;
 const globals=['foundry','CONFIG','game'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]);
 t.after(()=>{for(const [key,descriptor] of globals)descriptor?Object.defineProperty(globalThis,key,descriptor):delete globalThis[key]});
 const calls={updates:[],updateResults:[],diffs:[],receipts:[],clones:[]};
 const game={user:{id:'GM'},messages:new Map(),collections:new Map(),modules:new Map(),
  settings:{get:()=>false},pf2e:{settings:{variants:{stamina:false}}},i18n:{getListFormatter:options=>new Intl.ListFormat('en',options)}};
 const foundry={abstract:{Document,DataModel},data:{fields},utils:{...utils,buildUuid:({documentName,id})=>`${documentName}.${id}`},
  documents:{collections:{CompendiumCollection:class{}}},applications:{handlebars:{renderTemplate:async()=>'<p>Native damage receipt</p>'}}};
 globalThis.foundry=foundry;globalThis.game=game;
 const client=native.foundry['client/data/client-backend.mjs'];
 const method=start=>section(client,start,'\n  /* -------------------------------------------- */');
 // Keep the native update and empty-diff branches byte-for-byte. Any server
 // dispatch is a test failure: this fixture only exercises a local no-op.
 const ClientDatabaseBackend=Function('DatabaseBackend','game','foundry','Hooks','CONST','ui',`return class ClientDatabaseBackend extends DatabaseBackend {
  ${method('  async _updateDocuments(documentClass, operation, user) {')}
  ${method('  static async #preUpdateDocumentArray(documentClass, operation, user) {')}
  ${method('  static #getCollection(documentClass, {parent, pack}) {')}
  ${method('  static async #loadCompendiumDocuments(collection, documents) {')}
  static #buildRequest(){throw Error('unexpected-native-server-write')}
  static #dispatchRequest(){throw Error('unexpected-native-server-write')}
  async #handleResponse(){throw Error('unexpected-native-server-write')}
 }`)(DatabaseBackend,game,foundry,{call:()=>true,onError:(_name,error)=>{throw error}},{vtt:'Foundry'},
  {notifications:{error:message=>{throw Error(message)}}});
 let middleware,patient;
 class PatientDocument extends Document {
  static metadata={...Document.metadata,name:'Actor',collection:'actors'};
  static get baseDocument(){return this}
  static defineSchema(){return {
   _id:new fields.StringField({required:true,nullable:false}),system:new fields.ObjectField({required:true,nullable:false}),
   flags:new fields.ObjectField({required:true,nullable:false}),items:new fields.ArrayField(new fields.ObjectField())
  }}
  get isOwner(){return true}
  get hasPlayerOwner(){return true}
  get hitPoints(){return this.system.attributes.hp}
  get attributes(){return this.system.attributes}
  get heldShield(){return null}
  get hardness(){return 0}
  get synthetics(){return {damageDice:{},rollNotes:{}}}
  isOfType(...types){return types.includes('character')}
  isImmuneTo(){return false}
  clone(changes,options){
   calls.clones.push({changes:structuredClone(changes),options:structuredClone(options)});
   const clone=super.clone(changes,options);clone.system.attributes.hp=structuredClone(this.hitPoints);return clone;
  }
  async _preUpdate(changes,options,user){return preUpdate?.({patient,changes,options,user})}
  _updateDiff(...args){const diff=super._updateDiff(...args);calls.diffs.push(structuredClone(diff));return diff}
  update(changes,options){
   calls.updates.push({actor:this,changes:structuredClone(changes),options:structuredClone(options)});
   return middleware.call(this,(nextChanges,nextOptions)=>{
    const original=Document.prototype.update.call(this,nextChanges,nextOptions);
    original.then(saved=>calls.updateResults.push(saved),()=>{});return original;
   },changes,options);
  }
 }
 globalThis.CONFIG={Actor:{documentClass:PatientDocument},DatabaseBackend:new ClientDatabaseBackend()};
 patient=new PatientDocument({_id:'MH0N4niWCJIwAAGo',system:{attributes:{hp:{value,max:rawMax,temp:0}}},flags:{},items:[]});
 patient.system.attributes.hp.max=preparedMax;
 game.collections.set('Actor',new Map([[patient.id,patient]]));
 const ChatMessagePF2e={getSpeaker:({token})=>({actor:token.actor.id,token:token.id}),create:async data=>{
  const receipt={id:'NativeReceipt',...structuredClone(data)};
  Object.defineProperty(receipt,'author',{enumerable:true,get:()=>game.user});
  calls.receipts.push(receipt);game.messages.set(receipt.id,receipt);return receipt;
 }};
 const helpers={extractDamageDice:()=>[],extractModifiers:()=>[],extractNotes:()=>[],
  applyStackingRules:Function(`${native.stack};return applyStackingRules`)(),
  applyIWR:()=>{throw Error('unexpected-iwr')},_loc:key=>key,signedInteger:String,g:Boolean,S:entry=>typeof entry==='string',
  K:{convertXMLNode:()=>{}},Wi:{notesToHTML:()=>null},sluggify:String,
  Roll:class{constructor(){throw Error('unexpected-new-roll')}},ChatMessage:ChatMessagePF2e,CONFIG:{PF2E:{}},
  ui:{notifications:{warn:()=>{throw Error('unexpected-native-warning')}}},createDisintegrateEffect:()=>{throw Error('unexpected-effect')}};
 const nativeMethods=Function('game','foundry','ChatMessagePF2e','document','CONST','helpers',`
  const {extractDamageDice,extractModifiers,extractNotes,applyStackingRules,applyIWR,_loc,signedInteger,g,S,K,Wi,sluggify,Roll,ChatMessage,CONFIG,ui,createDisintegrateEffect}=helpers;
  return {${native.methods.join(',')}};
 `)(game,foundry,ChatMessagePF2e,{createElement:()=>({innerHTML:''})},{CHAT_MESSAGE_STYLES:{OTHER:0}},helpers);
 Object.assign(PatientDocument.prototype,nativeMethods);
 const pools=createHpPools({game,actorUpdateEvents:{addActorUpdateMiddleware:fn=>{middleware=fn;return()=>{}}}});t.after(()=>pools.dispose());
 const token={id:'Token',name:'Patient',actor:patient};
 const source=`${MODULE}:source:OriginalResult:0`,application=`${MODULE}:exploration-apply:Activity:OriginalResult:${patient.uuid}`;
 const run=()=>pools.withNativeApplication({id:'Activity',hpPoolUUIDs:[patient.uuid]},patient,async()=>{
  const recipient=patient.getContextualClone(['origin:actor:healer'],[]);
  assert.notEqual(recipient,patient);assert.equal(recipient.uuid,patient.uuid);
  assert.equal(recipient.flags.pf2e.rollOptions.all['origin:actor:healer'],true);
  const nativeResult=await recipient.applyDamage({damage:-18,token,skipIWR:true,final:false,outcome:'criticalSuccess',rollOptions:new Set([source,application])});
  assert.equal(nativeResult,recipient);
  return {nativeResult,receipt:calls.receipts.at(-1)};
 });
 return {nativeMethods,patient,calls,game,pools,run,source,application};
}

function assertNativeNoop(f) {
 assert.equal(f.calls.clones.length,1);assert.equal(f.calls.clones[0].options.keepId,true);
 assert.equal(f.calls.updates.length,1);assert.notEqual(f.calls.updates[0].actor,f.patient);
 assert.deepEqual(f.calls.updates[0].changes,{'system.attributes.hp.value':36});assert.equal(f.calls.updates[0].options.damageTaken,-18);
 assert.deepEqual(f.calls.diffs,[{}]);assert.deepEqual(f.calls.updateResults,[undefined]);assert.equal(f.calls.receipts.length,1);
 const receipt=f.calls.receipts[0];assert.equal(receipt.author,f.game.user);assert.equal(receipt.speaker.actor,f.patient.id);
 assert.deepEqual(receipt.flags.pf2e.appliedDamage,{uuid:f.patient.uuid,isHealing:true,shield:null,persistent:[],updates:[]});
 assert.equal(receipt.flags.pf2e.context.type,'damage-taken');assert.deepEqual(receipt.flags.pf2e.context.domains,['healing-received']);
 assert.deepEqual(receipt.flags.pf2e.context.options,[f.source,f.application]);assert.equal(f.game.messages.get(receipt.id),receipt);
}

boundedTest('pinned PF2e full-HP healing persists non-null empty undo proof after native Foundry update returns undefined',async t=>{
 const f=await fixture(t);assert.deepEqual(f.nativeMethods.calculateHealthDelta({hp:f.patient.hitPoints,sp:null,delta:-18}),{updates:{'system.attributes.hp.value':36},totalApplied:-18});
 let result;try{result=await f.run()}finally{assertNativeNoop(f)}
 assert.deepEqual(result.poolReceipt,{activityId:'Activity',actorUUID:f.patient.uuid,noChange:true,receiptId:'NativeReceipt'});
 assert.deepEqual(f.patient._source.system.attributes.hp,{value:36,max:36,temp:0});assert.equal(f.patient.flags.pf2e,undefined);
});

boundedTest('native prepared HP maximum may differ from source maximum while both values prove the same empty diff',async t=>{
 const f=await fixture(t,{rawMax:0});let result;try{result=await f.run()}finally{assertNativeNoop(f)}
 assert.equal(result.poolReceipt.noChange,true);assert.equal(f.patient._source.system.attributes.hp.max,0);assert.equal(f.patient.hitPoints.max,36);
});

boundedTest('the primary native application awaits the actual Foundry update before publishing its no-op receipt',async t=>{
 let enter,release;const entered=new Promise(resolve=>enter=resolve),gate=new Promise(resolve=>release=resolve);
 t.after(()=>release());
 const f=await fixture(t,{preUpdate:async()=>{enter();await gate}});let completed=false;
 const pending=f.run().then(result=>{completed=true;return result});pending.catch(()=>{});
 await Promise.race([entered,pending.then(()=>{throw Error('native-update-gate-not-reached')})]);
 assert.equal(completed,false);assert.equal(f.calls.receipts.length,0);assert.deepEqual(f.calls.updateResults,[]);
 release();let result;try{result=await pending}finally{assertNativeNoop(f)}assert.equal(result.poolReceipt.noChange,true);
});

boundedTest('a raw HP change during native pre-update leaves the original native application unconfirmed',async t=>{
 const f=await fixture(t,{preUpdate:async({patient})=>{await Promise.resolve();patient._source.system.attributes.hp.temp=1;return false}});
 await assert.rejects(f.run(),/native-hp-forward-unconfirmed/);
 assert.equal(f.calls.updates.length,1);assert.deepEqual(f.calls.updateResults,[undefined]);assert.equal(f.calls.receipts.length,1);
 assert.equal(f.patient.hitPoints.temp,0);assert.equal(f.patient._source.system.attributes.hp.temp,1);
 assert.deepEqual(f.calls.receipts[0].flags.pf2e.appliedDamage.updates,[]);
});

boundedTest('the primary native zero-maximum path still confirms its null receipt without an HP update',async t=>{
 const f=await fixture(t,{rawMax:0,preparedMax:0,value:0}),result=await f.run();
 assert.equal(result.poolReceipt.noChange,true);assert.equal(f.calls.updates.length,0);assert.equal(f.calls.receipts.length,1);
 assert.equal(f.calls.receipts[0].flags.pf2e.appliedDamage,null);
});
