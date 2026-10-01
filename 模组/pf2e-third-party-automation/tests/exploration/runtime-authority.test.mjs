import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createExplorationRuntime} from '../../scripts/exploration/runtime.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';

function fixture(){
 const gm={id:'G',isGM:true,active:true,getFlag:()=>null},calls=[];
 const game={user:gm,users:{activeGM:gm,get:id=>id==='G'?gm:null},settings:{get:()=>''},modules:new Map(),packs:new Map(),messages:new Map(),actors:{},time:{worldTime:0},system:{version:'8.5.1'},socket:{id:'socket',connected:true,on(){},off(){},emit(){calls.push('emit');throw Error('unexpected-socket-write')}},pf2e:{actions:new Map()}};
 const nativeCasts={addMatcher(){},addCapture(){},addActorUpdateMiddleware(){},addConsumePolicy(){},addObserver(){}};
 return {game,calls,nativeCasts,Hooks:{on(){},off(){}},fromUuid:async()=>null};
}
function documentFixture(){
 const f=fixture(),hooks=new Map(),listeners=new Map();let configured='',raw;
 f.Hooks.on=(event,handler)=>{const list=hooks.get(event)??[];list.push(handler);hooks.set(event,list);return handler};
 f.game.settings={get:()=>configured,set:async(namespace,key,value)=>{configured=value;for(const handler of hooks.get('updateSetting')??[])handler({key:`${namespace}.${key}`})}};
 f.game.user.setFlag=async()=>{};
 f.game.socket.on=(event,handler)=>{const list=listeners.get(event)??new Set();list.add(handler);listeners.set(event,list)};
 f.game.socket.off=(event,handler)=>listeners.get(event)?.delete(handler);
 f.game.socket.emit=(event,request,...args)=>{
  if(event!== 'modifyDocument')return;
  const respond=args[0],envelope={type:request.type,action:request.action,operation:request.operation,broadcast:false,userId:'G'};
  if(request.action==='get')return respond({...envelope,result:raw?[structuredClone(raw)]:[]});
  if(request.type==='JournalEntry'){
   raw={...structuredClone(request.operation.data[0]),_id:'ROOT000000000001',pages:[]};return respond({...envelope,result:[structuredClone(raw)]});
  }
  const page=request.operation.data[0];
  if(raw.pages.some(p=>p._id===page._id))return respond({...envelope,error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}});
  raw.pages.push(structuredClone(page));respond({...envelope,result:[structuredClone(page)]});
 };
 return {...f,seed:state=>{raw={_id:'ROOT000000000001',ownership:{default:0},pages:[],flags:{[MODULE_ID]:{explorationLedger:state}}};configured='JournalEntry.ROOT000000000001'},emitHook:(event,...args)=>hooks.get(event)?.forEach(handler=>handler(...args))};
}

test('a new runtime binds its transaction identity and exposes controlled setup',async()=>{
 const f=fixture(),runtime=createExplorationRuntime(f);await runtime.bind({});
 assert.equal(typeof runtime.api.storage.status,'function');assert.equal(typeof runtime.api.storage.provision,'function');assert.equal(typeof runtime.api.storage.initialize,'function');assert.equal(typeof runtime.api.storage.select,'function');assert.equal(typeof runtime.api.takeover,'function');
 assert.equal(runtime.api.diagnostic().status,'setup-required');assert.equal((await runtime.api.storage.status()).state,'unconfigured');assert.equal(f.calls.length,0);
});

test('binding and restoring an unconfigured runtime do not create storage or issue native work',async()=>{
 const f=fixture(),runtime=createExplorationRuntime(f);await runtime.bind({});
 await assert.rejects(runtime.api.start({actorUUIDs:[],manual:true}),/exploration-ledger-setup-required/);
 await assert.rejects(runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true}),/root-not-configured/);assert.equal(f.calls.length,0);
});

test('first root provisioning keeps the new runtime available for owner evidence',async()=>{
 const f=documentFixture(),runtime=createExplorationRuntime(f);await runtime.bind({});await runtime.register({socket:null});
 const result=await runtime.api.storage.provision({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});assert.equal(result.requiresReload,false);
 await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});assert.equal(runtime.api.diagnostic().status,'ready');
 await assert.doesNotReject(runtime.ownerOperations.reconcile({id:'Absent',actorUUID:'Actor.None',providerId:'treat-wounds',options:{}}));
});

test('the panel resumes through the runtime encounter and PF2e version guards',async t=>{
 const before=globalThis.foundry;t.after(()=>{globalThis.foundry=before});globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 for(const mode of ['system','encounter']){
  const f=documentFixture();f.seed({sessions:{S:{id:'S',status:'paused',actorUUIDs:[],startedAt:0,cursorAt:0,budgetEndsAt:600,activityIds:[],goalsByPool:[]}},activities:{},clocks:{}});f.game.user.getFlag=()=> 'S';
  const runtime=createExplorationRuntime(f);await runtime.bind({});await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});
  if(mode==='system')f.game.system.version='8.5.2';else f.game.combat={started:true};
  const panel=await runtime.api.open([]);await assert.rejects(panel.act('resume',{}),/encounter-or-system-unavailable/);
 }
});

test('runtime initialization quarantines legacy automatic sessions without adopting their driver',async()=>{
 const f=documentFixture();f.seed({sessions:{S:{id:'S',status:'running',manual:false,actorUUIDs:[],startedAt:0,cursorAt:0,budgetEndsAt:600,activityIds:[],goalsByPool:[]}},activities:{},clocks:{}});f.game.user.getFlag=()=> 'S';
 const runtime=createExplorationRuntime(f);await runtime.bind({});assert.equal((await runtime.api.storage.status()).state,'legacy');
 const prepared=await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});assert.equal(prepared.state,'ready');
 const data=await runtime.api.snapshot('S');assert.equal(data.session.status,'paused');assert.equal(data.session.stopReason,'legacy-migration-quarantine');assert.equal(data.session.driver,undefined);
});
