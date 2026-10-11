import test, {describe} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {installDsnChatRecovery} from '../scripts/patches/dsn-chat.mjs';
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/dsn-chat-native.json',import.meta.url)));
for(const name of ['dsn-queue-native.json','dsn-queue-6.4.3-native.json','dsn-queue-6.4.4-native.json'])describe(name,()=>{
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/'+(name.includes('6.4.4')?'dsn-chat-6.4.4-native.json':'dsn-chat-native.json'),import.meta.url)));
const queueFixture=JSON.parse(fs.readFileSync(new URL('./fixtures/'+name,import.meta.url)));
const settle=()=>new Promise(resolve=>setImmediate(resolve));
function setup({failure=null,hidden=false,pending=false,secret=false}={}){
 const errors=[],events=[],queued=[],classes=new Set(['dsn-hide']);
 const node={classList:{remove:v=>classes.delete(v)},querySelectorAll:()=>[]};
 const user={id:'player'},message={id:'message',author:user,getFlag:()=>false,speaker:{actor:'actor'},whisper:secret?['gm']:[],isContentVisible:!secret,content:secret?'???':'12',_dice3dMessageHidden:true,_dice3dPendingRenders:1,_dice3danimating:true};
 const settings=new Map([['forceCharacterOwnerAppearance','0'],['hide3dDiceOnSecretRolls',true]]);
 const game={version:'14.368',release:{generation:14},view:'game',user,settings:{get:(_id,key)=>settings.get(key)},messages:new Map([[message.id,message]]),modules:new Map([['dice-so-nice',{active:true,version:queueFixture.provenance.version}]]),actors:new Map(),users:[]};
 const ui={chat:{element:{querySelector:()=>node},_shouldShowNotifications:()=>false,scrollBottom(){}},sidebar:{popouts:{}}};
 const context=vm.createContext({game,window:{ui,document:{hidden}},ui,document:{querySelector:()=>null},Hooks:{callAll:(...args)=>events.push(args)},InitiativeMask:{release(){}},CompanionLink:{release:()=>[]},ChatMessage:{getSpeakerActor:()=>null},DsnSettings:{CONFIG:()=>({visibility:'all'}),ALL_CONFIG:()=>({}),ALL_CUSTOMIZATION:()=>({}),isEnabled:()=>true},DiceNotation:class{constructor(roll){if(failure==='notation')throw Error('notation failed');this.throws=[roll];}},setTimeout,CONST:{DOCUMENT_OWNERSHIP_LEVELS:{OWNER:3}}});
 if(fixture.rollRules)vm.runInContext(fixture.rollRules,context);
 const Native=vm.runInContext('(class Native {'+Object.values(fixture.methods).join('\n')+'})',context);
 const pipeline=new Native();
 Object.assign(pipeline,{_assignDependentRollOrder(){},_stampRole(){},_stampAppearance(){},_buildRollList:rolls=>rolls,_showNestedParts(){},pendingThrows:{isPending:()=>pending},messageUpdateHideSelector:'.dice-roll'});
 const queue={enqueue(data){assert.equal(this,queue);queued.push(data);return failure==='queue'?Promise.reject(Error('queue failed')):Promise.resolve(true);},deferHidden(data){assert.equal(this,queue);return this.enqueue(data);}};
 pipeline.queue=queue;
 game.dice3d={pipeline};
 const g={game,console:{error:(...args)=>errors.push(args)}};
 return {g,pipeline,message,errors,events,queued,classes,settings,Native,context};
}
const rolls=()=>[{dice:[{}],total:12},{dice:[{}],total:4}];

test('chat animation setup failure releases the existing message through the native pipeline',async()=>{
 const f=setup({failure:'notation'});installDsnChatRecovery({g:f.g});
 f.pipeline.renderRolls(f.message,rolls());await settle();
 assert.equal(f.message._dice3dPendingRenders,0);assert.equal(f.classes.has('dsn-hide'),false);
 assert.equal(f.events.filter(e=>e[0]==='diceSoNiceRollComplete').length,1);
 assert.equal(f.errors.length,2);
});
for(const hidden of [false,true])test(`rejected ${hidden?'deferred':'visible'} queue cannot strand a chat message`,async()=>{
 const f=setup({failure:'queue',hidden});installDsnChatRecovery({g:f.g});
 f.pipeline.renderRolls(f.message,rolls());await settle();
 assert.equal(f.message._dice3dPendingRenders,0);assert.equal(f.classes.has('dsn-hide'),false);assert.equal(f.queued.length,2);
});
test('successful ordered throws retain native metadata and reveal exactly once',async()=>{
 const f=setup();installDsnChatRecovery({g:f.g});
 f.pipeline.renderRolls(f.message,rolls());await settle();
 assert.deepEqual(f.queued.map(d=>[d.rollTotal,d.messageId]),[[16,'message'],[16,'message']]);
 assert.equal(f.message._dice3dPendingRenders,0);assert.equal(f.errors.length,0);
 assert.equal(f.events.filter(e=>e[0]==='diceSoNiceRollComplete').length,1);
});

test('supported DsN and later Foundry labels use function contracts and preserve render return',async()=>{
 const f=setup();f.g.game.version='15.1';f.g.game.release.generation=15;f.g.game.modules.get('dice-so-nice').version='7.0.0';
 const result=installDsnChatRecovery({g:f.g});assert.equal(result.status,'installed');
 assert.equal(f.pipeline.renderRolls(f.message,rolls()),undefined);await settle();
 assert.equal(f.message._dice3dPendingRenders,0);assert.equal(f.errors.length,0);
});

test('public show rejects invalid notation with the native error after recovery installs',async()=>{
 const f=setup(),native=f.pipeline.show;assert.equal(installDsnChatRecovery({g:f.g}).status,'installed');
 assert.equal(f.pipeline.show,native);await assert.rejects(f.pipeline.show({}),/Roll data should be not null/);
});
test('recovery preserves secret content and a pending interactive throw',async()=>{
 const f=setup({failure:'queue',secret:true,pending:true});installDsnChatRecovery({g:f.g});
 f.pipeline.renderRolls(f.message,rolls());await settle();
 assert.equal(f.message.content,'???');assert.equal(f.message.isContentVisible,false);assert.deepEqual(f.message.whisper,['gm']);
 assert.equal(f.events.filter(e=>e[0]==='diceSoNiceRollComplete').length,0);
 assert(f.queued.every(d=>d.throws[0].ghost===true&&d.throws[0].secret===true));
});
test('public showForRoll API keeps its own exception behavior',()=>{
 const f=setup({failure:'notation'});const original=f.pipeline.showForRoll;
 installDsnChatRecovery({g:f.g});assert.equal(f.pipeline.showForRoll,original);
 assert.throws(()=>f.pipeline.showForRoll(rolls()[0]),/notation failed/);
});
test('inactive DsN and foreign pipeline methods remain untouched',()=>{
 for(const kind of ['inactive','source']){
  const f=setup();if(kind==='inactive')f.g.game.modules.get('dice-so-nice').active=false;else f.pipeline.show=function foreign(){};
  const original=f.pipeline.renderRolls,result=installDsnChatRecovery({g:f.g});
  assert.equal(f.pipeline.renderRolls,original);assert.notEqual(result.status,'installed');
 }
});
if(fixture.rollRules){
 test('actor groups retain owner appearance and ordered rolls after one notation failure',async()=>{
  const f=setup();f.settings.set('forceCharacterOwnerAppearance','2');
  const owners=[{id:'owner-a',character:{id:'a'}},{id:'owner-b',character:{id:'b'}}];
  f.g.game.users.push(...owners);
  for(const id of ['a','b'])f.g.game.actors.set(id,{id,hasPlayerOwner:true});
  const NativeNotation=f.context.DiceNotation;
  f.context.DiceNotation=class extends NativeNotation{constructor(roll,...args){if(roll.total===4)throw Error('one bad model');super(roll,...args);}};
  const entries=[{dice:[{}],total:12,data:{actorId:'a'}},{dice:[{}],total:4,data:{actorId:'a'}},
    {dice:[{}],total:3,data:{actorId:'a'}},{dice:[{}],total:8,data:{actorId:'b'}}];
  assert.equal(installDsnChatRecovery({g:f.g}).status,'installed');
  f.pipeline.renderRolls(f.message,entries);await settle();
  assert.deepEqual(f.queued.map(d=>d.throws[0].total).sort((a,b)=>a-b),[3,8,12]);
  assert.deepEqual(f.queued.filter(d=>d.throws[0].data.actorId==='a').map(d=>d.throws[0].total),[12,3]);
  assert.deepEqual(f.queued.filter(d=>d.throws[0].data.actorId==='a').map(d=>d.rollTotal),[19,19]);
  assert.deepEqual(f.events.filter(e=>e[0]==='diceSoNiceRollStart').map(e=>e[2].user.id).sort(),['owner-a','owner-a','owner-a','owner-b']);
  assert.equal(f.events.filter(e=>e[0]==='diceSoNiceRollComplete').length,1);
  assert.equal(f.message._dice3dPendingRenders,0);assert.equal(f.message.content,'12');
 });
 for(const method of ['animateRolls','_showRollList'])test('unknown '+method+' retains the complete native pipeline',()=>{
  const f=setup();f.Native.prototype[method]=function foreign(){};
  const render=f.pipeline.renderRolls;
  assert.equal(installDsnChatRecovery({g:f.g}).status,'unsupported-source');
  assert.equal(f.pipeline.renderRolls,render);
 });
 test('direct animateRolls callers retain native rejection behavior',async()=>{
  const f=setup({failure:'notation'}),native=f.pipeline.animateRolls;
  assert.equal(installDsnChatRecovery({g:f.g}).status,'installed');
  assert.equal(f.pipeline.animateRolls,native);
  await assert.rejects(f.pipeline.animateRolls(rolls(),{author:f.g.game.user}),/notation failed/);
 });
}

test('restore removes only the owned render wrapper and repeated install is idempotent',()=>{
 const f=setup(),original=f.pipeline.renderRolls;
 const installed=installDsnChatRecovery({g:f.g}),wrapper=f.pipeline.renderRolls;
 installDsnChatRecovery({g:f.g});assert.equal(f.pipeline.renderRolls,wrapper);
 installed.restore();assert.equal(f.pipeline.renderRolls,original);assert.equal(Object.hasOwn(f.pipeline,'renderRolls'),false);
});


function realQueue(f,{persistent=false}={}){
 const {context}=f;let failures=1,simulations=0,hidden=0,landed=[];
 Object.assign(context,{setTimeout,clearTimeout,Utils:{removeTicker(){}},canvas:{app:{ticker:{add(){}}}}});
 context.DsnSettings.ALL_CUSTOMIZATION=()=>({});
 context.DiceNotation.mergeQueuedRollCommands=items=>items.map(i=>[{dice:[],dsnConfig:{}}]);
 const classes=vm.runInContext(`(()=>{${queueFixture.accumulator};${queueFixture.queue};return {AnimationQueue,Box:class{${queueFixture.boxStart}},Engine:class{${queueFixture.engineStart}}}})()`,context);
 const engine=new classes.Engine();
 Object.assign(engine,{rolling:false,running:false,diceList:[],deadDiceList:[],persistentDiceList:[],clearDice(){},diceScene:{display:{innerWidth:1000,innerHeight:800}},getVectors(){},checkForAnimatedDice:async()=>false,soundManager:{generateCollisionSounds:()=>[]},physicsWorker:{async exec(name){if(name==='simulateThrow'){simulations++;if(failures-->0)throw Error('native worker failed');return {ids:[],quaternionsBuffers:[],positionsBuffers:[],detectedCollides:[],deads:[],iterationsNeeded:0,faceValues:{},finalQuaternions:{}};}return true;}}});
 const box=new classes.Box();Object.assign(box,{throwEngine:engine,physicsWorker:engine.physicsWorker,inputHandler:{clearPendingThrowDice(){}},animateThrow(){}});
 const queue=new classes.AnimationQueue({canvasVisibility:{show(){},hide(){hidden++;}},pendingThrows:{noteBindsLanded:v=>landed.push(...v)}});queue.attach(box);
 f.pipeline.queue=queue;f.g.game.dice3d.box=box;f.settings.set('maxDiceNumber',20);
 return {queue,box,engine,hidden:()=>hidden,simulations:()=>simulations,failNext:()=>failures++,landed};
}

// _buildDiceBox starts initialize() and attaches immediately. Its ready promise
// resolves only after async scene/worker setup has created the throw engine.
function initializingBox(q){
 const box=Object.assign(Object.create(Object.getPrototypeOf(q.box)),{throwEngine:null,physicsWorker:q.box.physicsWorker,inputHandler:null,initialized:false,animateThrow(){}});
 let initialize;
 box.ready=new Promise(resolve=>{initialize=()=>{
  box.throwEngine=Object.assign(new q.engine.constructor(),q.engine);
  box.inputHandler=q.box.inputHandler;box.initialized=true;resolve();
 };});
 return {box,initialize};
}

test('attached native DiceBox installs recovery when its engine becomes ready after diceSoNiceReady',async()=>{
 const f=setup(),q=realQueue(f),boot=initializingBox(q);q.queue.attach(boot.box);
 const result=installDsnChatRecovery({g:f.g});assert.equal(result.queueStatus,'waiting-dsn');
 boot.initialize();await settle();assert.equal(result.queueStatus,'installed');
 f.pipeline.renderRolls(f.message,[rolls()[0]]);await settle();await settle();
 assert.equal(f.message._dice3dPendingRenders,0);assert.equal(f.classes.has('dsn-hide'),false);
 assert.equal(boot.box.throwEngine.rolling,false);assert.equal(q.simulations(),1);
});

test('superseded box readiness cannot replace the current queue recovery',async()=>{
 const f=setup(),q=realQueue(f),old=initializingBox(q),current=initializingBox(q);q.queue.attach(old.box);
 const reports=[],result=installDsnChatRecovery({g:f.g,report:value=>reports.push(value.queueStatus)});
 assert.equal(result.queueStatus,'waiting-dsn');q.queue.attach(current.box);
 current.initialize();await settle();assert.equal(result.queueStatus,'installed');
 const callback=q.queue.nextAnimation._onEnd,wrapper=current.box.startUnifiedBatch,reported=reports.length;
 old.initialize();await settle();
 assert.equal(q.queue.nextAnimation._onEnd,callback);assert.equal(current.box.startUnifiedBatch,wrapper);
 assert.equal(Object.hasOwn(old.box,'startUnifiedBatch'),false);assert.equal(reports.length,reported);
});

test('restore cancels pending box readiness without installing late wrappers',async()=>{
 const f=setup(),q=realQueue(f),boot=initializingBox(q);q.queue.attach(boot.box);
 const callback=q.queue.nextAnimation._onEnd,result=installDsnChatRecovery({g:f.g});
 assert.equal(result.queueStatus,'waiting-dsn');result.restore();boot.initialize();await settle();
 assert.equal(q.queue.nextAnimation._onEnd,callback);assert.equal(Object.hasOwn(boot.box,'startUnifiedBatch'),false);
 assert.equal(Object.hasOwn(q.queue,'attach'),false);assert.equal(Object.hasOwn(f.pipeline,'renderRolls'),false);
});

test('a restored installation cannot cancel a later installation while the same box initializes',async()=>{
 const f=setup(),q=realQueue(f),boot=initializingBox(q);q.queue.attach(boot.box);
 const old=installDsnChatRecovery({g:f.g});assert.equal(old.queueStatus,'waiting-dsn');old.restore();
 const current=installDsnChatRecovery({g:f.g});old.restore();boot.initialize();await settle();
 assert.equal(current.queueStatus,'installed');assert.equal(installDsnChatRecovery({g:f.g}),current);
 assert.equal(Object.hasOwn(boot.box,'startUnifiedBatch'),true);assert.equal(Object.hasOwn(q.queue,'attach'),true);
});

for(const changed of ['consumer','box','engine','attach','attach-prototype'])test(`box readiness preserves a foreign ${changed} replacement`,async()=>{
 const f=setup(),q=realQueue(f),boot=initializingBox(q);q.queue.attach(boot.box);
 const result=installDsnChatRecovery({g:f.g});assert.equal(result.queueStatus,'waiting-dsn');
 const foreign=function foreign(){};boot.initialize();
 if(changed==='consumer')q.queue.nextAnimation._onEnd=foreign;
 if(changed==='box')boot.box.startUnifiedBatch=foreign;
 if(changed==='engine')boot.box.throwEngine.startUnifiedBatch=foreign;
 if(changed==='attach')q.queue.attach=foreign;
 if(changed==='attach-prototype')Object.getPrototypeOf(q.queue).attach=foreign;
 await settle();assert.equal(result.queueStatus,'unsupported-queue');result.restore();
 assert.equal(Object.hasOwn(boot.box,'startUnifiedBatch'),changed==='box');
 assert.equal(Object.hasOwn(q.queue,'attach'),changed==='attach');
 if(changed==='consumer')assert.equal(q.queue.nextAnimation._onEnd,foreign);
 if(changed==='box')assert.equal(boot.box.startUnifiedBatch,foreign);
 if(changed==='engine')assert.equal(boot.box.throwEngine.startUnifiedBatch,foreign);
 if(changed==='attach')assert.equal(q.queue.attach,foreign);
 if(changed==='attach-prototype')assert.equal(Object.getPrototypeOf(q.queue).attach,foreign);
});
test('actual AnimationQueue recovers a worker rejection and accepts the following batch',async()=>{
 const f=setup(),q=realQueue(f);installDsnChatRecovery({g:f.g});
 f.pipeline.renderRolls(f.message,[rolls()[0]]);await settle();await settle();
 assert.equal(f.message._dice3dPendingRenders,0);assert.equal(f.classes.has('dsn-hide'),false);
 assert.equal(q.engine.rolling,false);assert.equal(q.box._preparingThrow,false);
 assert.equal(q.hidden(),1);
 const next=q.queue.enqueue({throws:[{}]},{});await settle();
 assert.equal(q.simulations(),2);assert.equal(typeof q.engine.callback,'function');
 q.engine.callback();assert.equal(await next,true);await q.queue.idle();
 assert.equal(q.queue.length,0);assert.equal(q.hidden(),2);
});
test('queue recovery installs after the real DiceBox is attached at diceSoNiceReady',async()=>{
 const f=setup(),q=realQueue(f);q.queue.attach(null);let ready;
 f.g.Hooks={once:(name,fn)=>{assert.equal(name,'diceSoNiceReady');ready=fn;return 1;},off(){}};
 const result=installDsnChatRecovery({g:f.g});assert.equal(result.queueStatus,'waiting-dsn');
 q.queue.attach(q.box);ready();assert.equal(result.queueStatus,'installed');
 f.pipeline.renderRolls(f.message,[rolls()[0]]);await settle();await settle();
 assert.equal(f.message._dice3dPendingRenders,0);assert.equal(q.engine.rolling,false);
});
test('rebuilding the DiceBox replaces only the owned queue recovery wrapper',async()=>{
 const f=setup(),q=realQueue(f),result=installDsnChatRecovery({g:f.g});
 const newBox=Object.assign(Object.create(Object.getPrototypeOf(q.box)),{throwEngine:q.engine,physicsWorker:q.box.physicsWorker,inputHandler:q.box.inputHandler,animateThrow(){}});
 q.queue.attach(newBox);
 assert.equal(Object.hasOwn(q.box,'startUnifiedBatch'),false);assert.equal(Object.hasOwn(newBox,'startUnifiedBatch'),true);
 f.pipeline.renderRolls(f.message,[rolls()[0]]);await settle();await settle();
 assert.equal(f.message._dice3dPendingRenders,0);assert.equal(newBox._preparingThrow,false);
 result.restore();assert.equal(Object.hasOwn(newBox,'startUnifiedBatch'),false);assert.equal(Object.hasOwn(q.queue,'attach'),false);
});
test('failed persistent batches settle their actual pending binds and retain unrelated dice',async()=>{
 const f=setup(),q=realQueue(f);installDsnChatRecovery({g:f.g});
 const bind={id:'pending'},held={id:'held',userData:{pendingBind:bind},quaternion:{set(){}}},unrelated={id:'other'};
 q.engine.persistentDiceList.push(unrelated);q.engine.persistentDiceManager={buildImpulseMap:()=>({})};
 const result=q.queue.enqueuePersistent({heldDice:[held],forcedByMesh:new Map(),velocity:{},queuedAt:Date.now()});
 await settle();assert.equal(await result,false);assert.deepEqual(q.landed,[bind]);
 assert.equal(held.userData.pendingBind,undefined);assert.equal(q.engine.persistentDiceList[0],unrelated);
 assert.equal(q.engine.rolling,false);
});
test('changed native queue consumers are not overwritten',()=>{
 const f=setup(),q=realQueue(f),original=()=>{};q.queue.nextAnimation._onEnd=original;
 const result=installDsnChatRecovery({g:f.g});
 assert.equal(result.queueStatus,'unsupported-queue');assert.equal(q.queue.nextAnimation._onEnd,original);
 assert.equal(Object.hasOwn(q.box,'startUnifiedBatch'),false);
});
test('native resize in queue.idle continuation stays protected before accumulator finally',async()=>{
 const f=setup(),q=realQueue(f),result=installDsnChatRecovery({g:f.g});
 f.pipeline.renderRolls(f.message,[rolls()[0]]);
 let newBox,processing;
 await q.queue.idle().then(()=>{
   processing=q.queue.nextAnimation._isProcessing;
   newBox=Object.assign(Object.create(Object.getPrototypeOf(q.box)),{throwEngine:q.engine,physicsWorker:q.box.physicsWorker,inputHandler:q.box.inputHandler,animateThrow(){}});
   q.queue.attach(newBox);
 });
 await settle();
 assert.equal(processing,true);assert.equal(result.queueStatus,'installed');assert.equal(Object.hasOwn(newBox,'startUnifiedBatch'),true);
 assert.equal(Object.hasOwn(q.box,'startUnifiedBatch'),false);
});

test('native resize waits for the replacement engine created asynchronously after queue.idle',async()=>{
 const f=setup(),q=realQueue(f),result=installDsnChatRecovery({g:f.g});
 f.pipeline.renderRolls(f.message,[rolls()[0]]);
 let resized,processing;
 await q.queue.idle().then(()=>{
  processing=q.queue.nextAnimation._isProcessing;resized=initializingBox(q);q.queue.attach(resized.box);
 });
 assert.equal(processing,true);assert.equal(result.queueStatus,'waiting-dsn');
 assert.equal(Object.hasOwn(q.box,'startUnifiedBatch'),false);
 resized.initialize();await settle();assert.equal(result.queueStatus,'installed');
 q.failNext();const next=q.queue.enqueue({throws:[{}]},{});await settle();
 assert.equal(await next,false);await q.queue.idle();assert.equal(q.simulations(),2);
 assert.equal(resized.box.throwEngine.rolling,false);assert.equal(resized.box._preparingThrow,false);
 assert.equal(q.queue.length,0);assert.equal(q.hidden(),2);
 result.restore();assert.equal(Object.hasOwn(resized.box,'startUnifiedBatch'),false);
});

});
