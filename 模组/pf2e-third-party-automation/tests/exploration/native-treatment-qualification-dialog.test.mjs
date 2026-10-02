import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createSalubriousCheckScope} from '../../scripts/salubrious-kiss-check-scope.mjs';
const flush=async()=>{for(let i=0;i<25;i++)await Promise.resolve()};
function fixture({remote=false,withQualification=true}={}){
  let qualified=true,rolls=0,app,qualificationCalls=0;const listeners=new Map();let serial=0;
  const Hooks={on:(name,fn)=>{listeners.set(++serial,{name,fn});return serial},off:(_name,id)=>listeners.delete(id),
    fire:(name,...args)=>{for(const row of [...listeners.values()])if(row.name===name)row.fn(...args)}};
  const ctx=Object.freeze({nativeDialogMode:remote?'owner-preference':'automatic',validate(){}}),healer={},
    activity={id:'A',options:{skill:'medicine',rank:'trained'}};
  const scope=createSalubriousCheckScope({game:{user:{settings:{showCheckDialogs:true}}},Hooks,isExplorationContext:c=>c===ctx});
  const context={actor:healer,type:'skill-check',domains:['medicine'],dc:{value:15},options:new Set(['action:treat-wounds','exploration-activity:A'])};
  const wrapped=async(_check,actual)=>{if(actual.skipDialog!==true){const accepted=await new Promise(resolve=>{app={context:actual,resolve,close:async()=>{}};Hooks.fire('renderCheckModifiersDialog',app)});if(!accepted)return null}rolls++;return 'native-roll-boundary'};
  const input={ctx,healer,activity,...withQualification?{assertQualification:()=>{qualificationCalls++;if(!qualified)throw Error('source-qualification-lost')}}:{}};
  const run=()=>scope.runExploration(input,()=>scope.interceptCheck(wrapped,{},context));
  return {scope,ctx,context,input,run,lose:()=>{qualified=false},get rolls(){return rolls},get app(){return app},get qualificationCalls(){return qualificationCalls}};
}
test('private qualification callback blocks the exact automatic check before dice',async()=>{
  const f=fixture();f.lose();await assert.rejects(f.run(),/source-qualification-lost/);assert.equal(f.rolls,0);assert.ok(f.qualificationCalls>0);
});
test('owner dialog final acceptance rereads qualification lost without an update event',async()=>{
  const f=fixture({remote:true}),pending=f.run().then(value=>({value}),error=>({error}));await flush();assert.ok(f.app);assert.equal(f.rolls,0);
  f.lose();await f.app.resolve(true);const result=await pending;assert.match(result.error?.message??'',/source-qualification-lost/);assert.equal(f.rolls,0);
});
test('a valid owner qualification crosses the same dialog boundary exactly once',async()=>{
  const f=fixture({remote:true}),pending=f.run();await flush();assert.ok(f.app);await f.app.resolve(true);await f.app.resolve(true);
  assert.equal(await pending,'native-roll-boundary');assert.equal(f.rolls,1);assert.ok(f.qualificationCalls>1);
});
test('legacy exploration callers without an optional callback keep their native check boundary',async()=>{
  const f=fixture({withQualification:false});assert.equal(await f.run(),'native-roll-boundary');assert.equal(f.rolls,1);
});
test('an unrelated check cannot consume the private qualification callback',async()=>{
  const f=fixture();await f.scope.runExploration(f.input,async()=>{
    const unrelated={options:new Set(),skipDialog:false};assert.equal(await f.scope.interceptCheck(async()=> 'unrelated',{},unrelated),'unrelated');
    assert.equal(f.qualificationCalls,0);
    return f.scope.interceptCheck(async()=> 'bound',{},f.context);
  });assert.ok(f.qualificationCalls>0);
});
test('copied context cannot install a qualification callback',async()=>{
  const f=fixture();await assert.rejects(f.scope.runExploration({...f.input,ctx:{...f.ctx}},async()=>assert.fail('private operation entered')),/context/);
  assert.equal(f.qualificationCalls,0);assert.equal(f.rolls,0);
});
