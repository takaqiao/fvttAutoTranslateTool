import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {createHpPools} from '../../scripts/exploration/hp-pool.mjs';

const sourceDirectory='C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/shared-manual-task4-core-source';
const toolbeltSource='C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/shared-manual-toolbelt-seam/original-main.js';
const sha=value=>createHash('sha256').update(value).digest('hex');
const sourceBytes=fs.readFileSync(`${sourceDirectory}/update-route.fragments.json`);
assert.equal(sha(sourceBytes),'75527e8d586b5e311192c4581284acdad52249cf6f706cb085f8b89b4aece216');
const fragments=JSON.parse(sourceBytes);
const metadata=JSON.parse(fs.readFileSync(`${sourceDirectory}/metadata.json`));
assert.equal(metadata.coreVersion,'14.368');
for(const source of metadata.sources){
 assert.equal(sha(fs.readFileSync(source.path)),source.sha256);
 for(const excerpt of source.excerpts)assert.equal(sha(fragments[excerpt.key]),excerpt.sha256);
}
assert.equal(sha(fs.readFileSync(toolbeltSource)),'2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f');
const previousToolbeltSource=process.env.TOOLBELT_MANUAL_SOURCE;
process.env.TOOLBELT_MANUAL_SOURCE=toolbeltSource;
const {seamFixture}=await import('../../tools/toolbelt-manual-pool/seam.test.mjs');
if(previousToolbeltSource===undefined)delete process.env.TOOLBELT_MANUAL_SOURCE;
else process.env.TOOLBELT_MANUAL_SOURCE=previousToolbeltSource;

const permit={permitNonce:'permit',applicationNonce:'application',ownerUserId:'O',selectedPatientUUID:'Actor.P',poolUUID:'Actor.M'};
const method=(file,signature)=>fragments[`${file}:${signature}`];

function coreRoute(game){
 const preUpdates=[],dispatches=[],lookups=[];
 class ActorCollection extends Map{
  get(id,options){lookups.push({id,options});return super.get(id)}
 }
 const actors=new ActorCollection(game.actors);
 game.actors=actors;game.collections=new Map([['Actor',actors]]);
 const context=vm.createContext({game,console,Date,Promise,Object,Array,Error,structuredClone,
  Hooks:{call:()=>true,onError:(_where,error)=>{throw error}},CONST:{vtt:'Foundry'},
  ui:{notifications:{error:message=>{throw Error(message)}}},
  foundry:{utils:{isEmpty:value=>Object.keys(value).length===0},
   documents:{collections:{CompendiumCollection:class{}}},
   abstract:{DocumentSocketResponse:class{constructor(value){Object.assign(this,value)}}}},
  SocketInterface:{async dispatch(channel,request){
   assert.equal(channel,'modifyDocument');assert.equal(request.action,'update');
   dispatches.push(request);return {...request,result:request.operation.updates};
  }},
  persist(response){return response.result.map(change=>{
   const doc=actors.get(change._id);doc.updateSource(change,{dryRun:false});return doc;
  })}
 });
 // These are unchanged installed Core methods. The server response boundary
 // saves the accepted delta locally; no socket, world, or database is opened.
 vm.runInContext(`
  class Document {
   ${method('document','async update(data={}, operation={})')}
   ${method('document','static async updateDocuments(updates=[], operation={})')}
  }
  class ClientDatabaseBackend {
   ${method('backend','async update(documentClass, operation, user)')}
   ${method('backend','async #configureUpdate(documentClass, operation)')}
   ${method('backend','async #configureOperation(documentClass, operation)')}
   ${method('backend','async _getParent(operation)')}
   ${method('backend','#assertCompendiumUnlocked(operation)')}
   ${method('client-backend','async _updateDocuments(documentClass, operation, user)')}
   ${method('client-backend','static async #preUpdateDocumentArray(documentClass, operation, user)')}
   ${method('client-backend','static #getCollection(documentClass, {parent, pack})')}
   ${method('client-backend','static #buildRequest(documentClass, action, operation)')}
   ${method('client-backend','static async #dispatchRequest(request)')}
   ${method('client-backend','static async #loadCompendiumDocuments(collection, documents)')}
   static #adjustActorDeltaRequest(){throw Error('embedded documents are outside this fixture')}
   async #handleResponse(response){return persist(response)}
  }
  globalThis.Actor=class Actor extends Document {
   static documentName='Actor';
   static implementation=this;
   static database=new ClientDatabaseBackend();
   static cleanData(update){return structuredClone(update)}
   static async _preUpdateOperation(){return true}
   constructor(source){super();Object.assign(this,source);this.parent=null;this.pack=null;this._source={_id:this.id}}
   updateSource(changes,{dryRun}={}){
    if(!dryRun&&Object.hasOwn(changes,'system.attributes.hp.value'))this.system.attributes.hp.value=changes['system.attributes.hp.value'];
    return structuredClone(changes);
   }
  };
 `,context);
 return {Actor:context.Actor,actors,preUpdates,dispatches,lookups};
}

function fixture(){
 const f=seamFixture(),c=f.owner,route=coreRoute(c.game);let middleware,receiptCount=0;
 const patient=new route.Actor({...c.patient,system:{attributes:{hp:{value:1,max:30,temp:0}}},
  modules:{'pf2e-toolbelt':{shareData:{data:{health:true}}}}});
 patient._preUpdate=function(changes,options){route.preUpdates.push(this);return c.tool.pre(this,changes,options)};
 c.patient=patient;route.actors.set(patient.id,patient);
 const contextualActor=new route.Actor(patient),otherClone=new route.Actor(patient);
 c.master.system={attributes:{hp:{value:1,max:30,temp:0}}};
 c.game.settings={get:()=>true};c.game.modules.get('pf2e-toolbelt').active=true;
 c.game.toolbelt={api:{shareData:{getMasterInMemory:()=>c.master,getSlavesInMemory:()=>[patient]}}};
 c.game.messages=new Map();
 const pools=createHpPools({game:c.game,actorUpdateEvents:{addActorUpdateMiddleware:fn=>{middleware=fn}}});
 const nativeUpdate=route.Actor.prototype.update;
 route.Actor.prototype.update=function(changes,options){return middleware.call(this,nativeUpdate.bind(this),changes,options)};
 const options={permit,provider:c.api,validate:()=>true,request:{resultId:'R',rollIndex:0},updateActor:contextualActor};
 async function originalOperation(actor=contextualActor){
  const saved=await actor.update({'system.attributes.hp.value':20},{});
  assert.equal(saved,patient,'Core must return the canonical patient, rather than the contextual clone');
  const receipt={id:'receipt',speaker:{actor:patient.id},flags:{pf2e:{
   context:{type:'damage-taken',options:['pf2e-third-party-automation:source:R:0']},
   appliedDamage:{uuid:patient.uuid,isHealing:true,isReverted:false}}}};
  c.game.messages.set(receipt.id,receipt);receiptCount++;return {receipt};
 }
 return {...f,route,patient,contextualActor,otherClone,pools,options,originalOperation,receiptCount:()=>receiptCount};
}

test('contextual update uses the Core canonical pre-update route and awaits the original master',async()=>{
 const f=fixture();let done=false;
 assert.notEqual(f.contextualActor,f.patient);assert.equal(f.contextualActor.uuid,f.patient.uuid);
 const pending=f.pools.withManualApplication(f.options,f.patient,f.originalOperation).then(value=>{done=true;return value});
 pending.catch(()=>{});await f.turn();
 assert.equal(done,false);assert.equal(f.writes.length,1);assert.equal(f.receiptCount(),1);
 assert.equal(f.route.preUpdates.length,1);assert.equal(f.route.preUpdates[0],f.patient);
 assert.equal(f.route.dispatches.length,1);
 assert.ok(f.route.lookups.some(call=>call.id==='P'&&call.options?.strict===true));
 f.finish();const result=await pending;
 assert.equal(result.result.receipt,f.owner.game.messages.get('receipt'));
 assert.equal(result.poolReceipt.receiptId,'receipt');assert.equal(result.poolReceipt.actorUUID,'Actor.M');
 assert.equal(result.poolReceipt.master.terminal,'fulfilled');assert.equal(f.writes.length,1);assert.equal(f.receiptCount(),1);
});

test('another clone with the same UUID cannot borrow the permitted contextual update frame',async()=>{
 const f=fixture();assert.notEqual(f.otherClone,f.contextualActor);assert.equal(f.otherClone.uuid,f.contextualActor.uuid);
 const pending=f.pools.withManualApplication(f.options,f.patient,()=>f.originalOperation(f.otherClone));pending.catch(()=>{});
 await f.turn();f.finish();
 await assert.rejects(pending,/manual-pool/);
});

test('the canonical patient cannot be replaced while its original master Promise is pending',async()=>{
 const f=fixture();let done=false;
 const pending=f.pools.withManualApplication(f.options,f.patient,f.originalOperation).then(value=>{done=true;return value});pending.catch(()=>{});
 await f.turn();assert.equal(done,false);assert.equal(f.writes.length,1);assert.equal(f.receiptCount(),1);
 const replacement=new f.route.Actor(f.patient);assert.equal(replacement.uuid,f.patient.uuid);
 f.route.actors.set('P',replacement);f.finish();
 await assert.rejects(pending,/manual-pool/);
});
