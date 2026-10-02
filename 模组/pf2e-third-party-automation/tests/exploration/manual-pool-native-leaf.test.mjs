import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {fixture} from './manual-pool-provider-fixture.mjs';
import {createManualPoolApplication} from '../../scripts/exploration/manual-pool-application.mjs';

const source=fs.readFileSync(process.env.PF2E_MANUAL_SOURCE??'C:/Users/Taka/Desktop/fvtt/tmp/fortress-gap-audit-20260925/code/systems/pf2e/pf2e.mjs');
assert.equal(createHash('sha256').update(source).digest('hex'),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');
const text=source.toString(),start=text.indexOf('return await ChatMessagePF2e.create({',text.indexOf('async applyDamage(')),end=text.indexOf('}), this;',start)+'}), this;'.length;
assert.ok(start>0&&end>start);const originalTail=text.slice(start,end);
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject}};
async function leafFixture(remote,{coreCreate=false}={}){
 const f=await fixture(remote),{game,patient}=f.owner,grant=await f.owner.broker.claim(f.request),roll=f.messages.get('R').rolls[0],token={actor:patient};
 const originalParams={damage:-9,token,item:null,skipIWR:true,rollOptions:new Set(),outcome:undefined,shieldBlockRequest:undefined};
 const params={...originalParams,rollOptions:new Set(['pf2e-third-party-automation:source:R:0'])};
 const frame={contextualActor:patient,patient,token,message:f.messages.get('R'),roll,rollIndex:0,item:null,grant,isCurrent:()=>true,paramsSnapshot:{damage:-9,skipIWR:true,rollOptions:[],outcome:undefined,shieldBlockRequest:undefined}};
 const provider={descriptor:{version:1,providerId:'pf2e',providerVersion:'8.5.1',protocol:'pf2e-third-party-automation:manual-pool-batch:1',model:'numeric-empty-reception.v1',baseSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157'},currentCall:(actor,input)=>actor===patient&&input===originalParams?frame:null};
 const application=createManualPoolApplication({game,completion:f.owner.broker,getProvider:()=>provider}),createGate=deferred(),createStarted=deferred();let nativeCalls=0,createCalls=0;
 const originalCreate=async original=>{createCalls++;createStarted.resolve();await createGate.promise;
  const receipt={id:'original-receipt',author:'O',speaker:original.speaker,flags:original.flags,toObject(){return {id:this.id,author:this.author,speaker:this.speaker,flags:this.flags}}};f.messages.set(receipt.id,receipt);return receipt;
 };
 let create=originalCreate;
 if(coreCreate){
  const base='C:/Program Files/Foundry Virtual Tabletop/resources/app',document=fs.readFileSync(`${base}/common/abstract/document.mjs`,'utf8'),backend=fs.readFileSync(`${base}/client/data/client-backend.mjs`,'utf8');
  assert.equal(createHash('sha256').update(document).digest('hex'),'303e6bc84fbaa7d564dab144b1f59fd0a1f6584a14e872b293e74358ed9b49dc');assert.equal(createHash('sha256').update(backend).digest('hex'),'8ffe114bca601980bf2f4582ce1f6e27c555d76586470f1a16ac23f6f021b9c5');
  const excerpt=(text,needle)=>{const at=text.indexOf(needle);return text.slice(at,text.indexOf('\n  /* -------------------------------------------- */',at)).trim()};
  const pre=excerpt(backend,'static async #preCreateDocumentArray(').replace('#preCreateDocumentArray','preCreateDocumentArray'),single=excerpt(document,'static async create(data='),multiple=excerpt(document,'static async createDocuments(data=');
  const core=vm.createContext({game:{user:game.user},Hooks:{call:()=>true,onError:error=>{throw error}},foundry:{utils:{deepClone:structuredClone}},originalCreate});
  vm.runInContext(`class Backend{${pre}}class Message{constructor(data){this.data=data}static get implementation(){return this}static documentName='ChatMessage';static cleanData(data,options){if(options.copy!==false)throw Error('real in-place cleaning required');data.style=0;data.speaker.scene??=null;return data}async _preCreate(){return true}static async _preCreateOperation(){return true}static database={async create(type,operation){await Backend.preCreateDocumentArray(type,operation,game.user);return [await originalCreate(operation.data[0].data)]}};${single}${multiple}}globalThis.create=data=>Message.create(data);`,core);create=core.create;
 }
 const context=vm.createContext({game:{settings:{get:()=>false}},CONST:{CHAT_MESSAGE_STYLES:{OTHER:0}},ChatMessagePF2e:{getSpeaker:()=>({actor:'P'}),getWhisperRecipients:()=>[],create:data=>application.observeCreate(create,structuredClone(data))}});
 vm.runInContext(`globalThis.originalReceiptTail=async function({token:t,item:n,rollOptions:r}){const xe={uuid:'Actor.P',isHealing:true,isReverted:false},p='healing-received',be='',ye='';${originalTail}}`,context);
 const native=async next=>{nativeCalls++;await f.owner.update.call(patient,async(changes,options)=>{await patient._preUpdate(changes,options);return patient},{'system.attributes.hp.value':20},{});return context.originalReceiptTail.call(patient,next)};
 const captured=application.captureFrame(patient,originalParams);
 return {...f,application,params,frame,captured,native,createGate,createStarted,nativeCalls:()=>nativeCalls,createCalls:()=>createCalls,run:()=>application.applyNativeDamage(patient,params,native,captured)};
}
for(const remote of [false,true])test(`fixed native receipt create Promise and ${remote?'GM socket':'local'} master Promise must both finish`,async t=>{
 const f=await leafFixture(remote);t.after(()=>{f.application.stop();f.close()});let done=false;
 const pending=f.run().then(value=>{done=true;return value});pending.catch(()=>{});await Promise.race([f.createStarted.promise,pending]);
 await Promise.race([f.writeStarted,pending]);
 assert.equal(f.nativeCalls(),1);assert.equal(f.createCalls(),1);assert.equal(done,false);assert.equal(f.writes.length,1);
 f.finish();await f.turn();assert.equal(done,false);assert.equal((await f.owner.broker.lookup(f.request)).status,'reserved');
 f.createGate.resolve();assert.equal(await pending,f.owner.patient);assert.equal((await f.owner.broker.lookup(f.request)).status,'settled');
 assert.equal(f.writes.length,1);assert.equal(f.messages.get('original-receipt').flags.pf2e.context.type,'damage-taken');
});
test('an original receipt create rejection leaves the one native write unknown and cannot be replayed',async t=>{
 const f=await leafFixture(false);t.after(()=>{f.application.stop();f.close()});const pending=f.run();pending.catch(()=>{});await Promise.race([f.createStarted.promise,pending]);f.finish();f.createGate.reject(Error('original-create-rejected'));
 await assert.rejects(pending,/original-create-rejected/);assert.equal((await f.owner.broker.lookup(f.request)).status,'reserved');
 await assert.rejects(f.run(),/original-frame/);assert.equal(f.nativeCalls(),1);assert.equal(f.writes.length,1);
});
test('reading a matching forged receipt never substitutes for the original create Promise',async t=>{
 const f=await leafFixture(false);t.after(()=>{f.application.stop();f.close()});f.messages.set(f.receipt.id,f.receipt);
 const pending=f.application.applyNativeDamage(f.owner.patient,f.params,async()=>f.owner.patient,f.captured);pending.catch(()=>{});
 await assert.rejects(pending,/original-receipt-unavailable/);assert.equal(f.createCalls(),0);assert.equal(f.writes.length,0);assert.equal((await f.owner.broker.lookup(f.request)).status,'reserved');
});
test('Core 14.368 in-place creation cleaning preserves the exact original receipt Promise',async t=>{
 const f=await leafFixture(false,{coreCreate:true});t.after(()=>{f.application.stop();f.close()});const pending=f.run();pending.catch(()=>{});await Promise.race([f.createStarted.promise,pending]);f.finish();f.createGate.resolve();assert.equal(await pending,f.owner.patient);assert.equal((await f.owner.broker.lookup(f.request)).status,'settled');assert.equal(f.nativeCalls(),1);
});
