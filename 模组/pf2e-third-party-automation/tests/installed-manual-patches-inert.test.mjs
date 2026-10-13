import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {buildPatreonSource} from '../tools/automatic-source-patches/patreon.mjs';
import {batchRegion,flatRegion,observerRegion,toolSeams} from '../scripts/native-source-shapes.mjs';
import {native861,native861Source,hash} from './native-iwr-861-fixture.mjs';
import {receiverContext} from './native-iwr-861-receiver-fixture.mjs';
import {referenceToolbelt,edited} from './toolbelt-shape-fixture.mjs';

// The manual-pool patches earlier releases wrote into PF2e and Toolbelt stay on
// disk until those packages update: maintenance neither writes nor removes them
// now. The Patreon patch keeps its manual-immunity half. This module no longer
// subscribes to any of the three and no new message carries its manual flag; in
// that state every patched call site has to do what the original source does.
//
// The patched forms run here are the ones maintenance installed, composed from
// the observer.js files under tools/, whose bytes native.mjs and patreon.mjs pin.
// One case per package also loads its observer.js directly.

// Upstream files are read from the environment and never stored here. A case
// whose file is not supplied is skipped.
function input(name,sha256){
 if(!process.env[name])return;
 const bytes=readFileSync(process.env[name]);assert.equal(hash(bytes),sha256,name);return bytes;
}
const component=file=>readFileSync(new URL('../tools/'+file,import.meta.url),'utf8');
const M='pf2e-third-party-automation';
const turn=()=>new Promise(resolve=>setImmediate(resolve));
// Values built inside a vm context carry that context's prototypes; a clone compares by content.
const sameData=(actual,expected)=>assert.deepEqual(structuredClone(actual),structuredClone(expected));

// PF2e: the 8.6.1 release, and the same release as maintenance installed it, with
// the batch observer and the static receiver composed from their observer.js files.
const pf2eOriginal=native861Source().toString();
const installed=buildSharedManualPair({pf2eSource:buildNativeBridge({source:native861Source(),version:'8.6.1'}).buffer,toolbeltSource:Buffer.from(referenceToolbelt),pf2eVersion:'8.6.1',toolbeltVersion:'3.59.0'});
const pf2ePatched=installed.pf2e.toString();
const pf2eCapture=input('PF2E_NATIVE_861_PATCHED','a720ae12536008dc6a20ab14fcc25b267b9697d6d1430153feb17ee97e7321af')?.toString();

const uniqueBy=(values,key)=>{const seen=new Set();return values.filter(value=>{const id=key(value);if(seen.has(id))return false;seen.add(id);return true})};
class DamageRoll{constructor(total){this.total=total;this._evaluated=true}alter(multiplier,addend){return {altered:this.total*multiplier+addend}}toJSON(){return {total:this.total}}}
/** One client's chat damage buttons, run from the batch region of `source`. */
function damageButtons(source,{shieldToggle=false,actorIsTarget=false}={}){
 const patched=source.includes('__nativeManualPoolBatch'),trace=[],frames=[];let selected=[],batches=0;
 const context=patched?receiverContext(source):vm.createContext({game:{pf2e:{}},CONFIG:{PF2E:{}},ErrorPF2e:message=>Error(message)}),game=context.game;
 Object.assign(game,{time:{worldTime:100},user:{id:'U',active:true,getActiveTokens:()=>selected},messages:new Map()});
 context.CONFIG.PF2E.chatDamageButtonShieldToggle=shieldToggle;
 // admit() draws a batch id before it looks for a subscriber, so the patched
 // button needs crypto.randomUUID, which a page has only in a secure context.
 Object.assign(context,{crypto:{randomUUID:()=>`BATCH-${++batches}`},un:DamageRoll,_t:uniqueBy,Fm:{onInit(){}},
  ui:{chat:{element:{}},notifications:{error:(...args)=>trace.push(['notify',...args])}},htmlQuery:()=>({dataset:{actorIsTarget:actorIsTarget?'true':''}}),
  extractEphemeralEffects:async({origin,target,item,domains,options})=>{trace.push(['effects',origin?.id??null,target.id,item?.uuid??null,domains,[...options]]);return [`effect:${target.id}`]},
  toggleOffShieldBlock:id=>trace.push(['shield-off',id]),shiftAdjustDamage:(message,multiplier,rollIndex)=>{trace.push(['prompt',message.id,multiplier,rollIndex]);return 'prompted'}});
 vm.runInContext(batchRegion(source),context);
 // What PF2e runs when it initializes; the patch appends the observer installation.
 const init=patched?native861.init+'__nativeManualPoolBatch.install();':native861.init;
 assert.equal(source.split(init).length,2);vm.runInContext(init,context);
 function token(id,{actor=true,troop}={}){
  const patient=actor?{id,uuid:`Actor.${id}`,alliance:'party',getSelfRollOptions:prefix=>[`${prefix}:actor:${id}`],
   getContextualClone(options,effects){
    const clone={uuid:`Actor.${id}`,getSelfRollOptions:()=>[`self:actor:${id}`],
     applyDamage(params){
      frames.push(game.pf2e.thirdPartyManualPoolBatch?.currentCall(this,params)??null);
      trace.push(['apply',id,this===clone,Object.keys(params),{...params,token:params.token.id,item:params.item?.uuid??null,rollOptions:[...params.rollOptions]}]);
      return Promise.resolve(this);
     }};
    trace.push(['clone',id,[...options],[...effects]]);return clone;
   }}:null;
  return {id,uuid:`Scene.S.Token.${id}`,actor:patient,flags:{pf2e:troop?{troop:{id:troop}}:{}}};
 }
 function message(id,{total=10,actor=null,item=null,token=null,outcome,roll=new DamageRoll(total)}={}){
  const result={id,uuid:`ChatMessage.${id}`,rolls:[roll],actor,item,token,flags:{pf2e:{context:{options:['self:class:cleric','item:trait:healing'],...(outcome?{outcome}:{})}}}};
  game.messages.set(id,result);return result;
 }
 return {trace,frames,token,message,api:game.pf2e.thirdPartyManualPoolBatch,select:tokens=>{selected=tokens},press:request=>context.applyDamageFromMessage(request),
  admit:candidate=>{context.candidate=candidate;return vm.runInContext('__nativeManualPoolBatch.admit(candidate)',context)}};
}
/** Press the buttons a table presses: healing, damage, half damage, a modifier prompt, no selection, a result without a damage roll. */
async function presses(buttons){
 const outcome=async work=>{try{return {returned:await work()}}catch(error){return {threw:error.message}}},results=[];
 const healer={id:'H',alliance:'party'},effect={uuid:'Item.E',isOfType:(...types)=>types.includes('effect'),getRollOptions:prefix=>[`${prefix}:slug:blessing`]};
 const a=buttons.token('A'),b=buttons.token('B'),troop=[buttons.token('T1',{troop:'TROOP'}),buttons.token('T2',{troop:'TROOP'})];
 const heal=buttons.message('HEAL',{outcome:'success'}),hit=buttons.message('HIT',{total:12,actor:healer,item:effect,token:buttons.token('SPEAKER')});
 buttons.select([a,b]);
 results.push(await outcome(()=>buttons.press({message:heal,multiplier:-1})));
 // The same result pressed again: an unsubscribed observer keeps no record of it.
 results.push(await outcome(()=>buttons.press({message:heal,multiplier:-1,addend:-2})));
 buttons.select([a,buttons.token('GONE',{actor:false}),...troop,b]);
 results.push(await outcome(()=>buttons.press({message:hit})));
 results.push(await outcome(()=>buttons.press({message:hit,multiplier:0.5,addend:1})));
 results.push(await outcome(()=>buttons.press({message:hit,multiplier:2,promptModifier:true})));
 buttons.select([]);
 results.push(await outcome(()=>buttons.press({message:hit})));
 buttons.select([a]);
 results.push(await outcome(()=>buttons.press({message:buttons.message('TEXT',{roll:{total:3}})})));
 return results;
}

test('PF2e: admit() returns null while nobody subscribes, from observer.js and from the installed observer',()=>{
 const game={pf2e:{}},context=vm.createContext({game,crypto:{randomUUID:()=>'BATCH'}});
 vm.runInContext(component('native-manual-pool-batch/observer.js'),context);
 const admit=candidate=>{context.candidate=candidate;return vm.runInContext('__nativeManualPoolBatch.admit(candidate)',context)};
 const buttons=damageButtons(pf2ePatched),message=buttons.message('HEAL'),token=buttons.token('A');
 const source={message,roll:message.rolls[0],rollIndex:0,multiplier:-1,addend:0,item:null,damage:-10,targets:[token]};
 assert.equal(admit(source),null);
 vm.runInContext('__nativeManualPoolBatch.install()',context);
 for(const api of [game.pf2e.thirdPartyManualPoolBatch,buttons.api])assert.deepEqual(Object.keys(api).sort(),['currentCall','descriptor','subscribe']);
 for(const unsubscribed of [admit,buttons.admit]){assert.equal(unsubscribed(source),null);assert.equal(unsubscribed(source),null)}
});

for(const [options,targets]of [[{},['A','B','A','B','A','T1','B','A','T1','B']],[{shieldToggle:true,actorIsTarget:true},['A','B','A','B','SPEAKER','SPEAKER','SPEAKER']]])
test(`PF2e: the patched damage buttons make the original calls while nobody subscribes (${JSON.stringify(options)})`,async()=>{
 const original=damageButtons(pf2eOriginal,options),patched=damageButtons(pf2ePatched,options);
 assert.equal(original.api,undefined);assert.ok(patched.api);
 const expected=await presses(original),actual=await presses(patched);
 assert.deepEqual(original.trace.filter(([kind])=>kind==='apply').map(([,id])=>id),targets);
 assert.deepEqual(expected.map(result=>Object.keys(result)[0]),[...Array(6).fill('returned'),'threw']);assert.equal(expected[4].returned,'prompted');
 sameData(actual,expected);sameData(patched.trace,original.trace);
 // No native call was given a manual frame.
 assert.equal(patched.frames.length,targets.length);assert.ok(patched.frames.every(frame=>frame===null));
});

test('PF2e: the patched FlatModifier preparation queues what the original queues',()=>{
 const register='__nativeManualPoolStaticReceiver.register(';
 assert.ok(flatRegion(pf2ePatched).includes(register)&&!flatRegion(pf2eOriginal).includes(register));
 const context=receiverContext(pf2ePatched);
 vm.runInContext(`const OriginalFlatModifier=class extends ${native861.receiverBindings.ruleElement}{${native861.receiverRegions.flatConstructor}get selectors(){return this.selector}${flatRegion(pf2eOriginal)}};`,context);
 // Prepare one rule on a fresh actor, then build every modifier it queued.
 const prepare=vm.runInContext(`(Rule,input)=>{
  const {raw,options}=JSON.parse(input),actor={uuid:'Actor.patient',items:new Map(),rules:[],flags:{pf2e:{}},synthetics:{modifiers:{},damageDice:{},modifierAdjustments:{}},isOfType:()=>true,getRollData:()=>({})};
  const item={id:'bonus',uuid:'Actor.patient.Item.bonus',name:'Bonus',actor,_source:{system:{rules:[raw]}},isOfType:type=>type==='effect',getRollOptions:()=>['item:slug:bonus']};
  actor.items.set(item.id,item);
  const rule=new Rule(raw,{parent:item,sourceIndex:0});actor.rules.push(rule);
  const returned=rule.beforePrepareData();
  const queued=Object.entries(actor.synthetics.modifiers).map(([selector,list])=>[selector,list.map(construct=>{try{return {modifier:construct({test:options})}}catch(error){return {threw:error.message}}})]);
  return {state:JSON.stringify({returned:returned===undefined,queued,rule:{...rule,parent:undefined},source:item._source}),
   // What the receiver holds for this actor: its flat healing total, or its refusal.
   received(){try{return __nativeManualPoolStaticReceiver.model(actor,{damage:-10,rollOptions:new Set(options),outcome:'success'}).flatTotal}catch(error){return error.message}}};
 }`,context);
 const [Patched,Original]=['FlatModifierRuleElement','OriginalFlatModifier'].map(name=>vm.runInContext(name,context)),unavailable='native-manual-batch-static-receiver-unavailable';
 for(const [raw,selector,built,received]of [
  // A constant healing bonus: the receiver records it.
  [{key:'FlatModifier',selector:['healing-received'],value:5,type:'status'},'healing-received','modifier',5],
  // Healing rules its model does not cover: it records nothing and refuses the actor.
  [{key:'FlatModifier',selector:['healing-received'],value:5,type:'status',predicate:['self:condition:wounded']},'healing-received','modifier',unavailable],
  [{key:'FlatModifier',selector:['healing-received'],value:'@actor.level'},'healing-received','threw',unavailable],
  // Any other selector: it returns at once.
  [{key:'FlatModifier',selector:['ac'],value:1,type:'circumstance'},'ac','modifier',0],
 ])for(const options of [[],['self:condition:wounded']]){
  const input=JSON.stringify({raw,options}),expected=prepare(Original,input),actual=prepare(Patched,input);
  assert.deepEqual(JSON.parse(expected.state).queued.map(([name,list])=>[name,list.map(entry=>Object.keys(entry)[0])]),[[selector,[built]]]);
  assert.equal(actual.state,expected.state);assert.equal(actual.received(),received);
 }
});

test('PF2e: the server capture holds the regions exercised here',{skip:!pf2eCapture},()=>{
 assert.equal(observerRegion(pf2eCapture,'pf2e').normalized,observerRegion(pf2ePatched,'pf2e').normalized);
 assert.equal(batchRegion(pf2eCapture),batchRegion(pf2ePatched));assert.equal(flatRegion(pf2eCapture),flatRegion(pf2ePatched));
 assert.equal(pf2eCapture.split(native861.init+'__nativeManualPoolBatch.install();').length,2);
});

// Toolbelt: the reference bundle holds the 3.59.0 seam regions verbatim, and
// maintenance installed the manual pool into it as it did into the release.
const toolbeltPatched=installed.toolbelt.toString();
const toolbeltCapture=input('TOOLBELT_3584_PATCHED','ffc9ab831ef313b246e51cfdf1939365f338083152bb4b3a863c1ba5bb9c96c7')?.toString();
// The seam tests match the reference bundle without running it. Two additions
// outside the seam regions let a case construct the tool and open its socket.
const runnable=bundle=>edited(edited(bundle,'init(){','listen(){this.#t.activate()}init(){'),'a(this,"NextTool")','a(this,"NextTool");globalThis.ShareData=Tool');
/** One client's shared-data tool, run from `bundle`. `owner` says whether its user owns the master. */
function shareData(bundle,{gm=false,owner=false}={}){
 const trace=[],listeners=[],wrappers={},ready=[],module={};
 class Document{}
 class Actor extends Document{
  constructor(id,isOwner){super();Object.assign(this,{id,uuid:`Actor.${id}`,pack:null,isOwner})}
  isOfType(){return true}
  update(fields){trace.push(['update',this.id,{...fields}]);return Promise.resolve(this)}
 }
 const master=new Actor('M',owner),patient=new Actor('P',true),userId=gm?'G':'O';
 const game={user:{id:userId,isActiveGM:gm},userId,users:{activeGM:{id:'G'}},actors:new Map([['M',master],['P',patient]]),modules:new Map([['pf2e-toolbelt',module]]),
  socket:{on:(channel,listener)=>listeners.push(listener),off(){},emit:(channel,packet)=>trace.push(['emit',channel,structuredClone(packet)])}};
 const context=vm.createContext({game,Actor,A:{id:'pf2e-toolbelt'},ne:{shared:key=>key},Hooks:{once:(event,fn)=>{if(event==='ready')ready.push(fn)}},
  ui:{notifications:{error:reason=>trace.push(['error',reason])}},fromUuid:async uuid=>game.actors.get(uuid.slice('Actor.'.length)),
  foundry:{abstract:{Document},utils:{getProperty:(object,key)=>object[key],deleteProperty:(object,key)=>delete object[key]}},
  u:{pipe:(value,...steps)=>steps.reduce((result,step)=>step(result),value),map:fn=>items=>items.map(fn),filter:fn=>items=>items.filter(fn),isDefined:value=>value!==undefined,
   fromEntries:Object.fromEntries,entries:Object.entries,mapValues:(object,fn)=>Object.fromEntries(Object.entries(object).map(([key,value])=>[key,fn(value,key)])),isArray:Array.isArray},
  // Stands in for the registration that hands Toolbelt's _preUpdate method to its wrapper.
  r:(event,method)=>{wrappers[event]=method}});
 vm.runInContext(runnable(bundle),context);
 Object.assign(context.ShareData.prototype,{path:name=>name,getMasterInMemory:()=>undefined,getMasterId:actor=>actor===patient?'M':undefined});
 const tool=new context.ShareData();tool.init();for(const hook of ready)hook();
 return {trace,api:module.api?.explorationManualPool,
  // The patient is saved with an HP change that Toolbelt moves to the shared master.
  preUpdate(){
   const changes={flags:{'pf2e-toolbelt':{shareData:{data:{master:'M',health:true}}}},'system.attributes.hp.value':20,'system.attributes.hp.temp':3,name:'kept'};
   return wrappers._preUpdate.call(tool,patient,(...args)=>{trace.push(['next',structuredClone(args)]);return 'saved'},changes,{diff:true},userId);
  },
  // The active GM's client receives what a player's client emitted.
  deliver(packet,senderId){tool.listen();assert.equal(listeners.length,1);return listeners[0](structuredClone(packet),senderId)}};
}
const hp={'system.attributes.hp.value':20,'system.attributes.hp.temp':3};
const next=['next',[{flags:{'pf2e-toolbelt':{shareData:{data:{master:'M',health:true}}}},name:'kept'},{diff:true},'O']];

test('Toolbelt: forward() and receive() in observer.js fall through to the original calls while nobody subscribes',()=>{
 // observer.js as written installs on Toolbelt 3.56.5; maintenance replaces that check with the module's presence.
 const hooks=[],module={version:'3.56.5'},context=vm.createContext({game:{modules:new Map([['pf2e-toolbelt',module]]),user:{id:'O'}},Hooks:{once:(event,fn)=>hooks.push(fn)}});
 vm.runInContext(component('toolbelt-manual-pool/observer.js'),context);for(const hook of hooks)hook();
 assert.deepEqual(Object.keys(module.api.explorationManualPool).sort(),['descriptor','subscribe']);
 const pool=vm.runInContext('__toolbeltManualPool',context);
 for(const isOwner of [true,false]){
  const calls=[],master={uuid:'Actor.M',isOwner};
  assert.equal(pool.forward({uuid:'Actor.P'},master,hp,{},(...args)=>calls.push(['local',...args]),(...args)=>calls.push(['remote',...args])),null);
  if(isOwner)assert.deepEqual(calls,[['local']]);
  else{assert.deepEqual(calls.map(call=>[call[0],call.length]),[['remote',2]]);assert.equal(calls[0][1].master,master);assert.deepEqual({...calls[0][1]},{master,...hp})}
 }
 for(const result of [Promise.resolve('written'),false]){
  let calls=0;
  assert.equal(pool.receive({uuid:'Actor.M'},hp,undefined,'O',()=>{calls++;return result}),result);assert.equal(calls,1);
 }
});

for(const owner of [true,false])test(`Toolbelt: the patched seam ${owner?'updates its own master':'sends to the GM'} as the original does while nobody subscribes`,async()=>{
 const forms=[shareData(referenceToolbelt,{owner}),shareData(toolbeltPatched,{owner})],results=[];
 assert.equal(forms[0].api,undefined);assert.ok(forms[1].api);
 for(const client of forms){
  const saving=client.preUpdate();
  // The write and the wrapped update are both made before the method first yields.
  results.push({made:structuredClone(client.trace),returned:await saving});
 }
 const write=owner?['update','M',hp]:['emit','module.pf2e-toolbelt',{master:'Actor.M',...hp,__converter__:{master:'document'},__source__:'object',__type__:'master'}];
 assert.deepEqual(results[0],{made:[write,next],returned:'saved'});
 sameData(results[1],results[0]);
 if(owner)return;
 for(const [index,bundle]of [referenceToolbelt,toolbeltPatched].entries()){
  const gm=shareData(bundle,{gm:true,owner:true});
  assert.equal(await gm.deliver(results[index].made[0][2],'O'),undefined);await turn();
  assert.deepEqual(gm.trace,[['update','M',hp]]);
 }
});

test('Toolbelt: the server capture holds the observer and seam exercised here',{skip:!toolbeltCapture},()=>{
 assert.equal(observerRegion(toolbeltCapture,'toolbelt').normalized,observerRegion(toolbeltPatched,'toolbelt').normalized);
 // toolSeams accepts only the reference regions, under whatever names the build gave them.
 assert.equal(toolSeams(toolbeltCapture).installed,true);assert.equal(toolSeams(toolbeltPatched).installed,true);
});

// Patreon: built with the versions the server's file records, the composite is that file byte for byte.
const patreonOriginal=input('PATREON_AUTOMATIC_SOURCE','89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9');
const patreonPatched=patreonOriginal&&buildPatreonSource({source:patreonOriginal,version:'3.2.29',pf2eVersion:'8.6.0'}).buffer;
/** One ordinary Treat Wounds check, without the manual flag, handled by the treatment regions of `source`. */
function treatment(source,{direct=true}={}){
 const writes=[],sent=[],updates=[],hooks=new Map(),module={api:{}},users=new Map(['HUSER','GM'].map(id=>[id,{id,isGM:id==='GM'}]));users.activeGM=users.get('GM');
 const healer={id:'H',uuid:'Actor.H',testUserPermission:user=>user?.isGM||user?.id==='HUSER'};
 const patient={id:'P',uuid:'Actor.P',name:'Patient',items:new Map(),conditions:{wounded:{async decrease(){writes.push(['wounded'])}}},
  testUserPermission:user=>user?.isGM||direct&&user?.id==='HUSER',canUserModify:user=>user?.isGM||direct,
  createEmbeddedDocuments(type,data){writes.push(['create',type,structuredClone(data)]);return Promise.resolve([{...structuredClone(data[0]),id:`I${writes.length}`}])}};
 const check={id:'C',isCheckRoll:true,isReroll:false,author:users.get('HUSER'),speaker:{actor:'H'},actor:healer,rolls:[{_evaluated:true}],
  flags:{pf2e:{modifiers:[],context:{type:'skill-check',origin:{actor:healer.uuid},target:{actor:patient.uuid},options:['action:treat-wounds'],outcome:'success'}}},
  async update(changes){updates.push(changes);return this}};
 const token={actor:patient,document:{uuid:'Scene.S.Token.T'}},targets=new Set([token]);targets.first=()=>token;users.get('HUSER').targets=targets;
 const game={user:users.get('HUSER'),users,system:{version:'8.6.1'},time:{worldTime:100},messages:new Map([['C',check]]),actors:new Map([['H',healer],['P',patient]]),modules:new Map([['patreon-v3',module]]),
  socket:{emit:(channel,payload)=>sent.push(structuredClone(payload)),on:(channel,listener)=>hooks.set(channel,listener)}};
 const context=vm.createContext({game,Hooks:{once:(event,fn)=>hooks.set(event,fn)},u:'patreon-v3',Aa:'module.patreon-v3',$t:'addItemToActor',r(){},k:()=>game.user===users.activeGM,console,
  fromUuid:async uuid=>uuid==='Actor.P'?patient:{toObject:()=>({type:'effect',system:{rules:[],duration:{unit:'hours',value:1,expiry:'turn-start',sustained:false},context:{}},flags:{}})},
  foundry:{utils:{mergeObject:(left,right)=>({...left,...right})}},ui:{notifications:{info(){}}}});
 const region=(start,end)=>{const from=source.indexOf(start),to=source.indexOf(end,from)+end.length;assert.ok(from>=0&&to>from,start);return source.slice(from,to)};
 const observerAt=source.indexOf('const __patreonManualImmunity=');
 if(observerAt>=0)vm.runInContext(source.slice(observerAt,source.indexOf('async function Ma(',observerAt)),context);
 for(const [start,end]of [['async function p(','r(p,"addItemToActor");'],['function x(','r(x,"executeAsGM");'],['function Ia(','r(Ia,"socketListener");'],['async function b(','r(b,"createItemObjectUuid");'],['async function Ma(','r(Ma,"treatWounds");'],['B=class{','this.content=e?.content}};']])vm.runInContext(region(start,end),context);
 hooks.get('init')?.();
 return {writes,sent,updates,api:module.api.explorationManualImmunity,
  treat:()=>context.Ma(new context.B(check,null,new Set(check.flags.pf2e.context.options)),patient),
  // Patreon adds every item through this function, not only the treatment immunity.
  add:data=>context.p(patient,data),
  receive(){game.user=users.get('GM');context.Ia();return Promise.all(sent.map(payload=>hooks.get('module.patreon-v3')(payload)))}};
}

test('Patreon: bind(), mark() and create() in observer.js leave a check without the manual flag to the original calls',async()=>{
 const hooks=new Map(),module={},updates=[],healer={id:'H',uuid:'Actor.H'},patient={id:'P',uuid:'Actor.P'},token={actor:patient,document:{uuid:'Scene.S.Token.T'}};
 const game={modules:new Map([['patreon-v3',module]]),messages:new Map(),user:{id:'HUSER'}};
 const context=vm.createContext({game,Hooks:{once:(event,fn)=>hooks.set(event,fn)},
  crypto:{randomUUID(){throw Error('an unflagged check opens no invocation')}},fromUuid(){throw Error('an unflagged treatment resolves no patient')}});
 vm.runInContext(component('patreon-manual-immunity/observer.js'),context);hooks.get('init')();
 assert.deepEqual(Object.keys(module.api.explorationManualImmunity).sort(),['descriptor','subscribe']);
 const immunity=vm.runInContext('__patreonManualImmunity',context),pf2e={modifiers:[],context:{type:'skill-check',options:['action:treat-wounds']}};
 // A check this module never marked, and one carrying only its other flags.
 for(const flags of [{pf2e},{pf2e,[M]:{other:true}}]){
  game.messages.set('C',{id:'C',actor:healer,author:{id:'HUSER'},flags,update:async changes=>{updates.push(changes)}});
  assert.equal(immunity.bind({messageId:'C',mainActor:healer,messageFlags:flags,tokenTargets:new Set([token])},patient),null);
 }
 assert.equal(immunity.bind({messageId:'DELETED',mainActor:healer,tokenTargets:new Set([token])},patient),null);
 for(const data of [{type:'effect',system:{duration:{unit:'hours',value:1}}},{type:'effect',flags:{[M]:{other:true}}}]){
  const before=structuredClone(data),created=Promise.resolve([{id:'I'}]),refused=Error('create-refused');let calls=0;
  assert.equal(await immunity.mark(data,null),data);
  assert.equal(immunity.create(patient,data,()=>{calls++;return created}),created);
  assert.throws(()=>immunity.create(patient,data,()=>{calls++;throw refused}),error=>error===refused);
  assert.equal(calls,2);assert.deepEqual(data,before);
 }
 assert.deepEqual(updates,[]);
});

test('Patreon: the composite built here is the file captured from the server',{skip:!patreonPatched},()=>{
 assert.equal(hash(patreonPatched),'08f20b2a5b862d931597377d229dab2d00918e71e2f2c71d58fc14953b0efa56');
});

for(const direct of [true,false])test(`Patreon: the patched treatment ${direct?'creates its immunity':'asks the GM for its immunity'} as the original does for a check without the manual flag`,{skip:!patreonPatched},async()=>{
 const forms=[treatment(patreonOriginal.toString(),{direct}),treatment(patreonPatched.toString(),{direct})],results=[];
 assert.equal(forms[0].api,undefined);assert.ok(forms[1].api);
 for(const client of forms){
  const returned=[await client.treat()];
  // Any other item Patreon adds goes through the same patched create.
  for(const data of [{type:'effect',name:'Other',system:{}},{type:'condition',name:'Flagged',flags:{[M]:{other:true}}}])returned.push(await client.add(data));
  if(!direct)returned.push(await client.receive());
  await turn();results.push({returned,writes:client.writes,sent:client.sent,updates:client.updates});
 }
 const [original]=results;
 assert.deepEqual(original.writes.map(([kind])=>kind),direct?['create','wounded','create','create']:['wounded','create','create','create']);
 assert.equal(original.sent.length,direct?0:3);assert.deepEqual(original.updates,[]);
 sameData(results[1],original);
});
