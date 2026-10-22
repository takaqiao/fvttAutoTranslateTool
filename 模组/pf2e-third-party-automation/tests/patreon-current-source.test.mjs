import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {buildPatreonSource} from '../tools/automatic-source-patches/patreon.mjs';
import {createPatreonTimeCompletion} from '../scripts/exploration/patreon-time-completion.mjs';
import {createTimeEffects} from '../scripts/exploration/time-effects.mjs';
import {createClock} from '../scripts/exploration/clock.mjs';

const input=process.env.PATREON_CURRENT_SOURCE;
assert.ok(input,'PATREON_CURRENT_SOURCE must name the external Patreon 3.2.30 source');
const source=readFileSync(input);
assert.equal(createHash('sha256').update(source).digest('hex'),
 '3373d8608064158ab3aaec5ad593c7a2e1aed4a78f7e4f6397b4ecee5cff8d5e','current source fixture SHA256');
const build=(bytes=source)=>buildPatreonSource({source:bytes,version:'3.2.30',pf2eVersion:'8.6.1'});
const original=source.toString('utf8');
function change(text,before,after){
 assert.equal(text.split(before).length,2,'unique mutation fixture');
 return text.replace(before,after);
}

test('current Patreon builds both bridges and leaves the second startup byte-identical',()=>{
 const first=build();
 assert.equal(first.status,'patch');
 assert.doesNotThrow(()=>new vm.Script(first.buffer.toString('utf8')));
 const second=build(first.buffer);
 assert.equal(second.status,'unchanged');
 assert.deepEqual(second.buffer,first.buffer);
});

for(const [name,before,after]of [
 ['a changed predicate','.filter(l=>l.test())','.filter(l=>!l.test())'],
 ['a changed healing operation','s+=Math.ceil(m.total*e)','s+=Math.floor(m.total*e)'],
 ['an altered manual immunity branch','async function Ea(a,e){','async function Ea(a,e){return;'],
 ['an empty item creation','let i=await a.createEmbeddedDocuments("Item",[e]);','let i=await a.createEmbeddedDocuments("Item",[]);'],
 ['a wrong active-GM helper','k()&&(h("fastHealingTime")','b()&&(h("fastHealingTime")'],
 ['a wrong setting helper','h("fastHealingTime")&&En(e)','b("fastHealingTime")&&En(e)'],
 ['a wrong dice namespace','new y.DamageRoll(`{(${f.resolveValue(f.value)})[healing]}`)','new z.DamageRoll(`{(${f.resolveValue(f.value)})[healing]}`)'],
 ['a wrong dice lookup variable','y.DamageRoll=CONFIG.Dice.rolls.find(e=>e.name==="DamageRoll")','y.DamageRoll=CONFIG.Dice.rolls.find(e=>z.name==="DamageRoll")'],
 ['a duplicate helper registration','r(k,"isActiveGM");','r(k,"isActiveGM");r(zz,"isActiveGM");'],
 ['a duplicate time registration','r(En,"handleFastHealingTime");','r(En,"handleFastHealingTime");r(En,"handleFastHealingTime");']
])test('current source refuses '+name,()=>{
 assert.throws(()=>build(change(original,before,after)),/patreon-/);
});

test('another consistent minifier rename builds and recovers without a version registration',()=>{
 assert.equal(/\bZq\b/.test(original),false,'fresh fixture name');
 const renamed=original.replace(/\bEn\b/g,'Zq');
 const first=buildPatreonSource({source:renamed,version:'future',pf2eVersion:'future'});
 assert.equal(first.status,'patch');assert.doesNotThrow(()=>new vm.Script(first.buffer.toString('utf8')));
 assert.deepEqual(build(first.buffer).buffer,first.buffer);
});

test('a renamed local at the start of a replacement anchor builds and recovers',()=>{
 const from=original.indexOf('Hooks.on("updateWorldTime",'),
  to=original.indexOf('r(En,"handleFastHealingTime");',from)+'r(En,"handleFastHealingTime");'.length;
 assert.ok(from>=0&&to>from);
 const renamed=original.slice(0,from)+original.slice(from,to).replace(/\bl\b/g,'Lq')+original.slice(to);
 const first=build(renamed);
 assert.equal(first.status,'patch');assert.doesNotThrow(()=>new vm.Script(first.buffer.toString('utf8')));
 assert.equal(build(first.buffer).status,'unchanged');assert.deepEqual(build(first.buffer).buffer,first.buffer);
});

test('current source refuses a partial observation or changed generated predicate',()=>{
 assert.throws(()=>build('const __patreonTimeCompletion={};'+original),/patreon-/);
 const text=build().buffer.toString('utf8');
 assert.throws(()=>build(change(text,'const __patreonManualImmunity=','const __patreonManualImmunityChanged=')),/patreon-/);
 assert.throws(()=>build(change(text,'__patreonTimeCompletion.predicate(x,l,l.test())',
  '__patreonTimeCompletion.predicate(x,l,!l.test())')),/patreon-/);
});

test('shape-qualified output retains provenance and ignores unrelated outer bytes and diagnostic versions',()=>{
 const first=build(),outer=Buffer.concat([Buffer.from('// outside\n'),first.buffer]);
 assert.equal(first.descriptor.version,4);
 assert.equal(first.descriptor.qualification,'patreon-source-shapes.v1');
 assert.equal(first.descriptor.sourceSHA256,'3373d8608064158ab3aaec5ad593c7a2e1aed4a78f7e4f6397b4ecee5cff8d5e');
 const next=buildPatreonSource({source:outer,version:'future',pf2eVersion:'future'});
 assert.equal(next.status,'unchanged');assert.deepEqual(next.buffer,outer);
});

const checkpoint={id:'C',sessionId:'S',from:100,to:106,gmId:'G'};
const clone=value=>JSON.parse(JSON.stringify(value));
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject}}

// Execute the generated current-source handler. Only Foundry documents, dice,
// settings, hooks and ledger storage cross into this isolated environment.
function runtime({healing=true,rules=true,duplicate=false,timeoutMs=1000}={}){
 const callbacks=new Set(),initializers=[],hp=deferred(),decrement=deferred();
 const calls={advance:0,hp:0,start:0,decrease:0,rolls:0};let lease={leaseNonce:'lease'},invocation=0;
 const effect={uuid:'Actor.A.Item.E',system:{start:{value:100}},getFlag:(module,key)=>{
  assert.equal(module,'patreon-v3');assert.equal(key,'decrementPeriod');return 6;
 },update:async data=>{calls.start++;assert.deepEqual(clone(data),{'system.start.value':106});effect.system.start.value=106},
 decrease:()=>{calls.decrease++;return decrement.promise}};
 const actor={id:'A',uuid:'Actor.A',system:{attributes:{hp:{value:10,max:30}}},itemTypes:{effect:[effect]},
  isOfType:()=>false,items:{contents:[]},update:data=>{
   calls.hp++;assert.equal(data['system.attributes.hp.value'],13);
   return hp.promise.then(()=>{actor.system.attributes.hp.value=data['system.attributes.hp.value']});
  }};
 class FastHealingRuleElement{
  constructor(type,value,index){this.type=type;this.value=value;this.sourceIndex=index;this.actor=actor;this.item={uuid:'Actor.A.Item.R'+index,sourceId:'Rule.'+index,actor}}
  test(){return true}
  resolveValue(value){return value}
 }
 // PF2e 8.6.1 represents both variants with this class and a type field.
 actor.items.contents=[{rules:rules?[new FastHealingRuleElement('fast-healing',1,0),new FastHealingRuleElement('regeneration',2,1)]:[]}];
 const actors=[actor,{isOfType:type=>type==='party',members:[actor],itemTypes:{effect:[]}}];
 actors.get=id=>actors.find(actor=>actor.id===id);
 const module={active:true,version:'3.2.30',api:{sibling:true}};
 const game={user:{id:'G'},users:{activeGM:{id:'G'}},actors,system:{id:'pf2e',version:'8.6.1'},
  modules:new Map([['patreon-v3',module]]),time:{worldTime:100,advance:async(delta,options)=>{
   calls.advance++;assert.equal(delta,6);game.time.worldTime+=delta;
   for(const callback of [...callbacks]){callback(game.time.worldTime,delta,options,'G');if(duplicate)callback(game.time.worldTime,delta,options,'G')}
  }}};
 const Hooks={once:(name,fn)=>{assert.equal(name,'init');initializers.push(fn)},
  on:(name,fn)=>{assert.equal(name,'updateWorldTime');callbacks.add(fn);return fn},off:(_name,fn)=>callbacks.delete(fn)};
 class DamageRoll{
  constructor(formula){assert.ok(['{(1)[healing]}','{(2)[healing]}'].includes(formula));this.formula=formula;this.total=Number(formula[2])}
  async evaluate(){calls.rolls++;return this}
  toJSON(){return {class:'DamageRoll',formula:this.formula,total:this.total}}
 }
 const context=vm.createContext({game,Hooks,crypto:{randomUUID:()=>String(++invocation)},structuredClone,console,
  r:()=>{},u:'patreon-v3',k:()=>game.user.id===game.users.activeGM.id,
  h:key=>{assert.equal(key,'fastHealingTime');return healing},Mn:values=>[...new Set(values)],y:{DamageRoll}});
 const patched=build().buffer.toString('utf8'),start=patched.indexOf('const __patreonTimeCompletion='),
  end=patched.indexOf('r(En,"handleFastHealingTime");',start)+'r(En,"handleFastHealingTime");'.length;
 assert.ok(start>=0&&end>start);vm.runInContext(patched.slice(start,end),context);
 for(const initialize of initializers)initialize();
 assert.equal(module.api.sibling,true);
 const native=module.api.explorationTimeCompletion;
 // Adapt only the VM/host realm boundary; keep the generated observer and its
 // authorization and terminal promise in the path exercised by the real clock.
 const provider={descriptor:clone(native.descriptor),subscribe:(observe,{authorizeMarkedCommit})=>native.subscribe(
  event=>observe({descriptor:clone(event.descriptor),invocation:clone(event.invocation),terminalPromise:event.terminalPromise.then(clone)}),
  {authorizeMarkedCommit:invocation=>authorizeMarkedCommit(clone(invocation))})};
 const hostGame={...game,modules:new Map([['patreon-v3',{...module,api:{explorationTimeCompletion:provider}}]])};
 const adapter=createPatreonTimeCompletion({game:hostGame,runtimeIdentity:()=>({userId:'G',clientNonce:'client'}),
  getDriverScope:()=>lease,timeoutMs});
 const timeEffects=createTimeEffects({capabilities:{activePassiveRules:async()=>rules?[{providerId:'pf2e-patreon',passing:true}]:[]},completionAdapters:[adapter]});
 const records=new Map(),ledger={getClockCommit:async id=>records.get(id),
  upsertClockCommit:async commit=>records.set(commit.id,clone(commit)),
  transitionClockCommit:async(id,{expected,patch})=>{
   const prior=records.get(id);assert.ok(expected.includes(prior.state));const saved={...prior,...patch};records.set(id,saved);return saved;
  }};
 const clock=createClock({game:hostGame,Hooks,ledger,timeEffects,isAuthority:()=>true});
 return {clock,adapter,provider,actor,game,calls,hp,decrement,loseLease:()=>{lease=null}};
}

test('current bridge advances once and waits fast healing, regeneration and decrement writes before confirming',async t=>{
 const f=runtime({duplicate:true});t.after(()=>f.adapter.invalidate('test-end'));
 let settled=false;const result=f.clock.advanceTo(checkpoint).then(value=>{settled=true;return value});
 await tick();assert.deepEqual(f.calls,{advance:1,hp:1,start:1,decrease:1,rolls:2});assert.equal(settled,false);
 f.hp.resolve();await tick();assert.equal(f.actor.system.attributes.hp.value,13);assert.equal(settled,false);
 f.decrement.resolve();const completed=await result;assert.equal(completed.status,'confirmed');
 assert.deepEqual(completed.commit.evidence[1].writes.map(write=>write.kind).sort(),['actor-hp','effect-decrease','effect-start']);
 assert.equal((await f.clock.advanceTo(checkpoint)).status,'confirmed');assert.equal(f.calls.advance,1);assert.equal(f.calls.hp,1);
});

for(const options of [{healing:false},{rules:false}])test('current bridge still waits timed effects with '+JSON.stringify(options),async t=>{
 const f=runtime(options);t.after(()=>f.adapter.invalidate('test-end'));
 let settled=false;const result=f.clock.advanceTo(checkpoint).then(value=>{settled=true;return value});
 await tick();assert.deepEqual(f.calls,{advance:1,hp:0,start:1,decrease:1,rolls:0});assert.equal(settled,false);
 f.decrement.resolve();assert.equal((await result).status,'confirmed');assert.equal(f.actor.system.attributes.hp.value,10);
});

for(const failure of ['rejected','timeout','lease'])test('current bridge leaves '+failure+' writes uncertain and never replays time',async t=>{
 const f=runtime({timeoutMs:failure==='timeout'?20:1000});t.after(()=>f.adapter.invalidate('test-end'));
 const result=f.clock.advanceTo(checkpoint);await tick();
 if(failure==='rejected')f.hp.reject(Error('native-write-failed'));
 if(failure==='lease')f.loseLease();
 if(failure!=='timeout'){f.hp.resolve();f.decrement.resolve()}
 assert.equal((await result).status,'uncertain');
 f.hp.resolve();f.decrement.resolve();await tick();
 assert.equal((await f.clock.advanceTo(checkpoint)).status,'uncertain');
 assert.deepEqual(f.calls,{advance:1,hp:1,start:1,decrease:1,rolls:2});
});

test('changed shape descriptor is blocked before time or document writes',async t=>{
 const f=runtime();t.after(()=>f.adapter.invalidate('test-end'));
 f.provider.descriptor.seams.time='0'.repeat(64);
 assert.equal((await f.clock.advanceTo(checkpoint)).status,'blocked');
 assert.deepEqual(f.calls,{advance:0,hp:0,start:0,decrease:0,rolls:0});
});
