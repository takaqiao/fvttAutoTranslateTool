import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {spawnSync} from 'node:child_process';

// Tests run exact v14.367 source fragments; see fixtures/chat-provenance.json.
const methods=JSON.parse(fs.readFileSync(new URL('./fixtures/chat-methods.json',import.meta.url),'utf8'));
const semaphore=fs.readFileSync(new URL('./fixtures/chat-semaphore.js',import.meta.url),'utf8');
function method(name){assert.equal(typeof methods[name],'string',name);return methods[name];}
const {createChatDeleteWrapper}=process.env.BASELINE ? {createChatDeleteWrapper:()=>function(wrapped,...args){return wrapped(...args);}} : await import('../scripts/patches/chat.mjs');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}

function fixture(ids,{popout=false,child=null,version='14.368',callbackMutation=null,schedule=setTimeout,sharedWrapper=null}={}){
  const counts={layouts:0,anchors:0,loads:0,refits:0,removed:0,notifications:0,overflow:0},nodes=new Map(),docs=new Map(),animations=[],rejections=[];
  let scrollTop=0;const scroll={get scrollTop(){return scrollTop;},set scrollTop(n){scrollTop=n;},get scrollHeight(){counts.layouts++;return 500;},clientHeight:100,
    classList:{toggle(){}},querySelector(){return {get offsetTop(){counts.anchors++;return 13;}};}};
  const idOf=s=>/data-message-id="([^"]+)"/.exec(s)?.[1];
  function add(id){
    docs.set(id,{id,logged:true});const node={dataset:{messageId:id},nextElementSibling:null,classList:{add(){}},getBoundingClientRect(){return {height:24};},
      remove(){counts.removed++;nodes.delete(id);},animate(frames,options){
        assert.deepEqual([...frames.height],['24px','0']);assert.equal(options.duration,100);
        const d=deferred();const originalThen=d.promise.then;
        // Observe the core's deliberately unawaited animation rejection without changing its returned child promise.
        d.promise.then=function(...args){const child=Reflect.apply(originalThen,this,args);Promise.prototype.then.call(child,undefined,error=>rejections.push(error));return child;};
        const animation={finished:d.promise};animations.push({node,...d,animation,originalThenDescriptor:Object.getOwnPropertyDescriptor(d.promise,'then')});return animation;
      }};
    nodes.set(id,node);return node;
  }
  ids.forEach(add);for(let i=0;i<ids.length-1;i++)nodes.get(ids[i]).nextElementSibling=nodes.get(ids[i+1]);
  const element={querySelector:s=>s==='.chat-scroll'?scroll:nodes.get(idOf(s))};
  const c=vm.createContext({game:{messages:docs},foundry:{utils:{}},CONFIG:{ChatMessage:{batchSize:100}},Promise,setTimeout,clearTimeout});
  vm.runInContext(`${semaphore};foundry.utils.Semaphore=Semaphore;`,c);
  let deletion=method('#deleteMessage');if(callbackMutation)deletion=callbackMutation(deletion);
  const Class=vm.runInContext(`(class {
    #renderingQueue=new foundry.utils.Semaphore(1);#lastId;#isAtBottom=true;
    #notificationsElement;#jumpToBottomElement={toggleAttribute(){}};#overflowingDebounce=()=>{this.counts.overflow++;};
    constructor(){this.element=element;this.rendered=true;this.isPopout=popout;this.popout=child;this.counts=counts;this.#lastId=first;
      this.#notificationsElement={querySelector:()=>({remove:()=>{counts.notifications++;}})};}
    renderBatch(){this.counts.loads++;return this.#renderingQueue.add(()=>{this.historyLoaded=true;});}
    _refit(){this.counts.refits++;}
    hold(promise){return this.#renderingQueue.add(()=>promise);}
    get debug(){return {lastId:this.#lastId,isAtBottom:this.#isAtBottom};}
    ${method('deleteMessage')}${deletion}${method('#onScrollLog')}
  })`,Object.assign(c,{element,popout,child,counts,first:ids[0]??null}));
  const chat=new Class();const native=chat.deleteMessage.bind(chat),wrapper=sharedWrapper??createChatDeleteWrapper({getVersion:()=>version,schedule});
  let lastNativeReturn;
  chat.deleteMessage=function(...args){return wrapper.call(this,(...passed)=>{lastNativeReturn=native(...passed);return lastNativeReturn;},...args);};
  return {chat,counts,nodes,docs,animations,rejections,add,lastNative:()=>lastNativeReturn};
}
function finishAll(f){for(const a of f.animations)a.resolve(a.animation);}

test('native queue result, lastId/logged/notifications and older-message loading remain intact',async()=>{
  const f=fixture(['oldest','next']);const node=f.nodes.get('oldest'),animate=node.animate;
  const returned=f.chat.deleteMessage('oldest');assert.notEqual(returned,f.lastNative());assert.equal(await returned,await f.lastNative());
  assert.equal(node.animate,animate);assert.equal(f.docs.get('oldest').logged,false);assert.equal(f.chat.debug.lastId,'next');
  assert.equal(f.counts.notifications,1);assert.equal(f.counts.overflow,1);assert.equal(f.nodes.size,2);
  finishAll(f);await sleep(15);assert.equal(f.nodes.size,1);assert.equal(f.counts.layouts,1);assert.equal(f.counts.loads,1);assert.equal(f.counts.anchors,1);assert.equal(f.chat.historyLoaded,true);
});
test('870 completed animations coalesce, while all native message actions execute',async()=>{
  const ids=Array.from({length:870},(_,i)=>`m${i}`),f=fixture(ids);
  await Promise.all(ids.map(id=>f.chat.deleteMessage(id,{deleteAll:true})));assert.equal(f.chat.debug.lastId,null);assert.equal(f.animations.length,870);
  finishAll(f);await sleep(20);assert.equal(f.counts.removed,870);assert.equal(f.counts.layouts,1);assert.equal(f.counts.loads,1);assert.equal(f.counts.anchors,1);
  assert.equal(f.counts.notifications,870);assert.equal(f.counts.overflow,870);assert.ok([...f.docs.values()].every(d=>!d.logged));
});
test('sidebar and popout use independent batches and preserve automatic forwarding',async()=>{
  const sharedWrapper=createChatDeleteWrapper({getVersion:()=> '14.368'});
  const child=fixture(['a','b'],{popout:true,sharedWrapper}),main=fixture(['a','b'],{child:child.chat,sharedWrapper});
  await Promise.all(['a','b'].map(id=>main.chat.deleteMessage(id)));await sleep(0);finishAll(main);finishAll(child);await sleep(15);
  assert.equal(main.counts.layouts,1);assert.equal(child.counts.layouts,1);assert.equal(main.counts.refits,0);assert.equal(child.counts.refits,1);
  assert.equal(main.counts.notifications,2);assert.equal(child.counts.notifications,0);
});
test('an asynchronous queue retains its own order and restores the temporary node method afterward',async()=>{
  const f=fixture(['a','b']),gate=deferred(),node=f.nodes.get('a'),animate=node.animate;f.chat.hold(gate.promise);
  const deletion=f.chat.deleteMessage('a');await sleep(0);assert.equal(f.animations.length,0);assert.equal(f.docs.get('a').logged,true);
  gate.resolve();await deletion;assert.equal(node.animate,animate);assert.equal(f.animations.length,1);finishAll(f);await sleep(15);assert.equal(f.counts.loads,1);
});
test('a card appearing only after an asynchronous render remains fully handled by native fallback',async()=>{
  const f=fixture([]),gate=deferred();f.chat.hold(gate.promise);const deletion=f.chat.deleteMessage('late');f.add('late');gate.resolve();await deletion;
  finishAll(f);await sleep(15);assert.equal(f.nodes.size,0);assert.equal(f.counts.layouts,1);assert.equal(f.docs.get('late').logged,false);
});
test('simultaneous deletes of the same ID do not leak adapters or duplicate final layout',async()=>{
  const f=fixture(['a']),node=f.nodes.get('a'),animate=node.animate;await Promise.all([f.chat.deleteMessage('a'),f.chat.deleteMessage('a')]);
  assert.equal(node.animate,animate);assert.equal(f.animations.length,2);finishAll(f);await sleep(15);assert.equal(f.nodes.size,0);assert.equal(f.counts.layouts,1);
});
test('cancelled animation preserves rejection and does not enter the successful-removal batch',async()=>{
  const f=fixture(['a','b']),node=f.nodes.get('a'),animate=node.animate;await f.chat.deleteMessage('a');
  f.animations[0].reject(Error('cancelled'));await sleep(15);assert.equal(node.animate,animate);assert.equal(f.nodes.has('a'),true);assert.equal(f.counts.layouts,0);assert.equal(f.rejections.length,1);
  await f.chat.deleteMessage('b');f.animations[1].resolve();await sleep(15);assert.equal(f.counts.layouts,1);assert.equal(f.nodes.has('b'),false);
});
test('native animation identity and finished promise method descriptor are preserved',async()=>{
  const f=fixture(['a']);await f.chat.deleteMessage('a');const a=f.animations[0];
  assert.deepEqual(Object.getOwnPropertyDescriptor(a.promise,'then'),a.originalThenDescriptor);assert.equal(a.animation.finished,a.promise);
  a.resolve(a.animation);await sleep(15);assert.deepEqual(Object.getOwnPropertyDescriptor(a.promise,'then'),a.originalThenDescriptor);
});
test('unknown callback semantics fall through instead of skipping any new behavior',async()=>{
  const f=fixture(['a','b'],{callbackMutation:s=>s.replace('li.remove();','li.remove(); this.extra=(this.extra??0)+1;')});
  await Promise.all(['a','b'].map(id=>f.chat.deleteMessage(id)));finishAll(f);await sleep(15);assert.equal(f.chat.extra,2);assert.equal(f.counts.layouts,2);
});
test('other Foundry versions use only native behavior',async()=>{
  const f=fixture(['a','b'],{version:'15.0'});await Promise.all(['a','b'].map(id=>f.chat.deleteMessage(id)));finishAll(f);await sleep(15);assert.equal(f.counts.layouts,2);
});
test('closed popout and detached nodes complete without refitting a closed application',async()=>{
  const f=fixture(['a','b'],{popout:true});await Promise.all(['a','b'].map(id=>f.chat.deleteMessage(id)));f.chat.rendered=false;f.nodes.clear();finishAll(f);await sleep(15);
  assert.equal(f.counts.layouts,0);assert.equal(f.counts.refits,0);assert.equal(f.rejections.length,0);
});
test('a scheduler failure runs the original completion rather than leaving a card stuck',async()=>{
  const f=fixture(['a'],{schedule(){throw Error('timer unavailable');}});await f.chat.deleteMessage('a');finishAll(f);await sleep(15);assert.equal(f.nodes.size,0);assert.equal(f.counts.layouts,1);
});

test('a native animation failure rejects the native queue promise and restores the node method',async()=>{
  const f=fixture(['a']),node=f.nodes.get('a');const original=node.animate=function(){throw Error('animation failed');};
  const returned=f.chat.deleteMessage('a');assert.notEqual(returned,f.lastNative());await assert.rejects(returned,/animation failed/);
  assert.equal(node.animate,original);assert.equal(f.counts.layouts,0);assert.equal(f.nodes.has('a'),true);
});

test('an inherited animate property is restored without leaving an own property',async()=>{
  const f=fixture(['a']),node=f.nodes.get('a'),animate=node.animate;delete node.animate;Object.setPrototypeOf(node,{animate});
  await f.chat.deleteMessage('a');assert.equal(Object.hasOwn(node,'animate'),false);assert.equal(node.animate,animate);
  finishAll(f);await sleep(15);assert.equal(f.counts.layouts,1);
});

test('an intervening module change is not overwritten when the queue completes',async()=>{
  const f=fixture(['a']),node=f.nodes.get('a'),gate=deferred();f.chat.hold(gate.promise);const deletion=f.chat.deleteMessage('a');
  const original=f.nodes.get('a').animate;const intervening=node.animate=function(...args){return Reflect.apply(original,this,args);};
  gate.resolve();await deletion;assert.equal(node.animate,intervening);finishAll(f);await sleep(15);assert.equal(f.counts.layouts,1);
});

test('a failing native popout refit is not replayed after its callback has already run',async()=>{
  const f=fixture(['a','b'],{popout:true});let calls=0;f.chat._refit=()=>{calls++;throw Error('closed during refit');};
  await Promise.all(['a','b'].map(id=>f.chat.deleteMessage(id)));finishAll(f);await sleep(15);
  assert.equal(calls,1);assert.equal(f.nodes.size,0);assert.equal(f.rejections.length,2);assert.equal(f.counts.layouts,1);
});

test('settlement continuation preserves the exact value and error with one additional microtask',async()=>{
  const node={animate(){}},app={rendered:true,element:{querySelector:()=>node}};
  const wrapper=createChatDeleteWrapper({getVersion:()=> '14.368'}),value={},native=Promise.resolve(value),order=[];
  const returned=wrapper.call(app,()=>native,'a');
  assert.notEqual(returned,native);
  native.then(()=>order.push('native'));
  returned.then(()=>order.push('returned'));
  await native;assert.deepEqual(order,['native']);
  assert.equal(await returned,value);assert.deepEqual(order,['native','returned']);
  const error=Error('native rejection'),rejected=wrapper.call(app,()=>Promise.reject(error),'a');
  await assert.rejects(rejected,actual=>actual===error);
});

test('unobserved native queue failure remains an unhandled rejection on the returned continuation',()=>{
  const moduleURL=new URL('../scripts/patches/chat.mjs',import.meta.url).href;
  const code=`import {createChatDeleteWrapper} from ${JSON.stringify(moduleURL)}; import vm from 'node:vm';
    const Semaphore=vm.runInNewContext(${JSON.stringify(semaphore)}+';Semaphore',{Promise}),queue=new Semaphore(1);
    const seen=[],node={animate(){}},original=node.animate,app={rendered:true,element:{querySelector:()=>node}};
    const error=Error('native queue failed');let native;
    const returned=createChatDeleteWrapper({getVersion:()=> '14.368'}).call(app,()=>native=queue.add(()=>{throw error;}),'a');
    process.on('unhandledRejection',(reason,promise)=>seen.push({sameError:reason===error,samePromise:promise===returned}));
    setTimeout(()=>console.log(JSON.stringify({seen,restored:node.animate===original,distinct:returned!==native})),25);`;
  const result=spawnSync(process.execPath,['--input-type=module','--eval',code],{encoding:'utf8',timeout:5000});
  assert.equal(result.status,0,result.stderr);
  assert.deepEqual(JSON.parse(result.stdout),{seen:[{sameError:true,samePromise:true}],restored:true,distinct:true});
});

test('overlapping same-node deletes behind a pending queue release every adapter and preserve failures',async()=>{
  const f=fixture(['a']),gate=deferred(),node=f.nodes.get('a'),original=node.animate;
  f.chat.hold(gate.promise);
  const first=f.chat.deleteMessage('a'),second=f.chat.deleteMessage('a');
  assert.notEqual(node.animate,original);
  gate.resolve();await Promise.all([first,second]);
  assert.equal(node.animate,original);assert.equal(f.animations.length,2);
  finishAll(f);await sleep(15);assert.equal(f.counts.layouts,1);
});
