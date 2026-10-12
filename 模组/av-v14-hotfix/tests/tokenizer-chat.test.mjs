import test from 'node:test';
import assert from 'node:assert/strict';
import {createTokenizerChatPortraitWrapper,installTokenizerChatPortraitPatch} from '../scripts/patches/tokenizer-chat.mjs';
import {environment,native} from './tokenizer-native-harness.mjs';

const selector=':scope > header.message-header > .portrait > img';
function fixture({tokenizer=true,visible=true,portrait=true}={}){
  const image={style:{transform:'scale(2.5199980754894695)',maskImage:'radial-gradient(circle at center, black 28%, rgba(0, 0, 0, 0.2) 46%)'},src:'tokenizer/pc-avatars/Avatar.webp?v=1'};
  const contentImage={style:{transform:'scale(2)',maskImage:'unrelated'}};
  const html={querySelector:s=>s===selector&&portrait?image:null,contentImage};
  const actor={flags:tokenizer?{'tokenizer-2':{layerStack:[{}],isOversized:true}}:{},prototypeToken:{texture:{scaleX:2,scaleY:2},ring:{enabled:true,subject:{scale:2}}}};
  return {image,html,message:{speakerActor:actor,isContentVisible:visible},actor};
}

test('Tokenizer2 chat avatar stays unscaled and unmasked without changing the token or card content',async()=>{
  const {image,html,message,actor}=fixture(),before=structuredClone(actor);
  const result=await createTokenizerChatPortraitWrapper().call(message,async()=>html);
  assert.equal(result,html);
  assert.equal(image.style.transform,'none');
  assert.equal(image.style.maskImage,'none');
  assert.equal(image.style.webkitMaskImage,'none');
  assert.equal(image.src,'tokenizer/pc-avatars/Avatar.webp?v=1');
  assert.deepEqual(actor,before);
  assert.equal(html.contentImage.style.transform,'scale(2)');
  assert.equal(html.contentImage.style.maskImage,'unrelated');
});

test('ordinary PF2e tokens retain the native oversize rendering',async()=>{
  const {image,html,message}=fixture({tokenizer:false}),before=structuredClone(image);
  await createTokenizerChatPortraitWrapper().call(message,async()=>html);
  assert.deepEqual(image,before);
});

test('does not add portraits to OOC or hidden messages',async()=>{
  for(const options of [{visible:false},{portrait:false}]){
    const {image,html,message}=fixture(options),before=structuredClone(image);
    await createTokenizerChatPortraitWrapper().call(message,async()=>html);
    assert.deepEqual(image,before);
  }
});

test('uses the message speaker actor, including scene-token synthetic actors',async()=>{
  const {image,html,message}=fixture();
  message.actor={flags:{}};
  await createTokenizerChatPortraitWrapper().call(message,async()=>html);
  assert.equal(image.style.transform,'none');
});

test('a dynamic-ring subject with transparent map padding uses the actor avatar in chat',async()=>{
  const {image,html,message,actor}=fixture();
  actor.img='tokenizer/pc-avatars/Avatar.webp?v=2';
  image.src='tokenizer/pc-images/Token.webp?v=2';
  image.getAttribute=name=>name==='src'?image.src:null;
  message.token={texture:{src:image.src},ring:{enabled:true,subject:{texture:image.src,scale:2}}};
  const before=structuredClone(message.token);
  await createTokenizerChatPortraitWrapper().call(message,async()=>html);
  assert.equal(image.src,'tokenizer/pc-avatars/Avatar.webp?v=2');
  assert.equal(image.style.transform,'none');
  assert.deepEqual(message.token,before);
});

test('keeps a distinct custom token image even when a dynamic ring is enabled',async()=>{
  const {image,html,message,actor}=fixture();
  actor.img='actor-portrait.webp';
  image.src='custom-token.webp';
  image.getAttribute=()=>image.src;
  message.token={texture:{src:image.src},ring:{enabled:true,subject:{texture:'separate-ring-subject.webp'}}};
  await createTokenizerChatPortraitWrapper().call(message,async()=>html);
  assert.equal(image.src,'custom-token.webp');
});

test('preserves upstream arguments, null results and rendering errors',async()=>{
  const wrapper=createTokenizerChatPortraitWrapper(),options={popout:true};
  assert.equal(await wrapper.call({},async value=>{assert.equal(value,options);return null;},options),null);
  const error=new Error('native rendering failed');
  await assert.rejects(wrapper.call({},async()=>{throw error;}),e=>e===error);
});

test('8.6.1 render method differs from 8.6.0 only in two compiler closure names and four mangled private names',()=>{
 const current=native['8.6.1'].renderHTML,prior=native['8.6.0'].renderHTML;
 assert.notEqual(current,prior);
 assert.equal(current.replace('Ds.listen(','Ts.listen(').replace('instanceof un ','instanceof ln ')
  .replace('this.#e(','this.#highlightDoS(').replace('this.#t(','this.#appendSetAsInitiative(')
  .replace('this.#n(','this.#onHoverIn(').replace('this.#r(','this.#onHoverOut('),prior);
});

test('registers once for each native render method regardless of the PF2e version label',()=>{
 for(const sourceVersion of ['8.5.1','8.6.0','8.6.1'])for(const version of [sourceVersion,'9.0.0']){
  const {runtime,calls}=environment({sourceVersion,version});
  assert.equal(installTokenizerChatPortraitPatch({runtime}).status,'installed');
  assert.equal(installTokenizerChatPortraitPatch({runtime}).status,'already-installed');
  assert.equal(calls.length,1);
  assert.equal(calls[0][1],'CONFIG.ChatMessage.documentClass.prototype.renderHTML');
  assert.equal(calls[0][3],'WRAPPER');
 }
});

test('other systems remain untouched',()=>{
  const {runtime,calls}=environment({systemId:'sf2e'});
  assert.equal(installTokenizerChatPortraitPatch({runtime}).status,'unsupported-system');
  assert.equal(calls.length,0);
});

test('a render result without querySelector is returned untouched instead of throwing',async()=>{
  const wrapper=createTokenizerChatPortraitWrapper(),{message}=fixture();
  for(const value of ['<li>text</li>',42,{outerHTML:'<li></li>'},{querySelector:null}])assert.equal(await wrapper.call(message,async()=>value),value);
});

test('any render implementation is wrapped, and HTML without the portrait is returned as it came',async()=>{
 const f=environment(),queried=[],bare={querySelector(value){queried.push(value);return null;}};
 f.runtime.CONFIG.ChatMessage.documentClass.prototype.renderHTML=async function(){return bare;};
 assert.equal(installTokenizerChatPortraitPatch({runtime:f.runtime}).status,'installed');
 assert.equal(f.calls.length,1);
 assert.equal(await f.calls[0][2].call(f.message,f.message.renderHTML.bind(f.message)),bare);
 assert.deepEqual(queried,[selector]);assert.deepEqual(Object.keys(bare),['querySelector']);
});

for(const sourceVersion of ['8.5.1','8.6.0','8.6.1']){
 test(`native ${sourceVersion} still applies map scaling and masking to Tokenizer portraits`,async()=>{
  const f=environment({sourceVersion});await f.message.renderHTML();
  assert.match(f.image().style.transform,/^scale\(2\.5/);
  assert.match(f.image().style.maskImage,/^radial-gradient/);assert.equal(f.image().src,'subject.webp');
 });
 test(`registered wrapper corrects the native ${sourceVersion} Tokenizer portrait`,async()=>{
  const f=environment({sourceVersion}),options={popout:true};
  assert.equal(installTokenizerChatPortraitPatch({runtime:f.runtime}).status,'installed');
  const wrapper=f.calls[0][2],before=structuredClone(f.actor);
  assert.equal(await wrapper.call(f.message,f.message.renderHTML.bind(f.message),options),f.html);
  assert.equal(f.image().style.transform,'none');assert.equal(f.image().style.maskImage,'none');
  assert.equal(f.image().src,'avatar.webp');assert.deepEqual(f.actor,before);assert.equal(f.message.nativeOptions,options);
 });
 test(`native ${sourceVersion} ordinary portraits, hidden messages and OOC keep their behavior`,async()=>{
  for(const kind of ['ordinary','hidden','ooc']){
   const f=environment({sourceVersion});
   if(kind==='ordinary')f.actor.flags={};
   if(kind==='hidden')f.message.isContentVisible=false;
   if(kind==='ooc')f.message.style=1;
   await createTokenizerChatPortraitWrapper().call(f.message,f.message.renderHTML.bind(f.message));
   if(kind==='ordinary')assert.match(f.image().style.transform,/^scale\(2\.5/);
   else assert.equal(f.image(),null);
  }
 });
}

for(const sourceVersion of ['8.6.0','8.6.1']){
 test(`${sourceVersion} mirrored token scale still needs the Tokenizer chat correction`,async()=>{
  const f=environment({sourceVersion});f.actor.prototypeToken.texture.scaleX=-2;
  await f.message.renderHTML();assert.match(f.image().style.transform,/^scale\(2\.5/);
  await createTokenizerChatPortraitWrapper().call(f.message,async()=>f.html);
  assert.equal(f.image().style.transform,'none');assert.equal(f.image().style.maskImage,'none');
 });

 test(`changed native ${sourceVersion} portrait code is still wrapped, and only a portrait it emits is corrected`,async()=>{
  for(const [before,after] of [['header.message-header','header.changed-header'],['Math.abs(e.texture.scaleX ?? 1)','1'],
    ['a.style.maskImage =','a.style.background ='],['&& this.isContentVisible','&& true']]){
   const renderSource=native[sourceVersion].renderHTML.replace(before,after);
   assert.notEqual(renderSource,native[sourceVersion].renderHTML,before);
   const f=environment({renderSource});
   assert.equal(installTokenizerChatPortraitPatch({runtime:f.runtime}).status,'installed',before);
   assert.equal(await f.calls[0][2].call(f.message,f.message.renderHTML.bind(f.message)),f.html);
   // The harness only knows the native header selector, so that variant emits no portrait.
   if(after==='header.changed-header')assert.equal(f.image(),null);
   else{assert.equal(f.image().style.transform,'none',before);assert.equal(f.image().style.maskImage,'none',before);}
  }
 });
}

// Small synchronous wrapper dispatcher for installation-order checks. Browser
// libWrapper internals and its priority configuration are outside this fixture.
function enableWrapperChain(f){
 f.runtime.libWrapper.register=(...args)=>{
  const [,target,wrapper,type]=args;
  assert.equal(target,'CONFIG.ChatMessage.documentClass.prototype.renderHTML');assert.equal(type,'WRAPPER');
  const prototype=f.runtime.CONFIG.ChatMessage.documentClass.prototype,original=prototype.renderHTML;
  prototype.renderHTML=function(...args){return wrapper.call(this,original.bind(this),...args);};
  f.calls.push(args);
 };
}

test('the correction installs on top of a pre-existing render wrapper and both run',async()=>{
 const f=environment(),reports=[];enableWrapperChain(f);
 let calls=0;
 f.runtime.libWrapper.register('portrait-module','CONFIG.ChatMessage.documentClass.prototype.renderHTML',async function(wrapped,...args){calls++;return wrapped(...args);},'WRAPPER');
 assert.equal(installTokenizerChatPortraitPatch({runtime:f.runtime,report:value=>reports.push(value)}).status,'installed');
 assert.equal(f.calls.length,2);assert.deepEqual(reports,[{feature:'tokenizerChat',status:'installed'}]);
 await f.message.renderHTML();assert.equal(calls,1);assert.equal(f.image().style.transform,'none');assert.equal(f.image().src,'avatar.webp');
});

test('a later render wrapper keeps the installed correction and repeated install adds no wrapper',async()=>{
 const f=environment();enableWrapperChain(f);
 assert.equal(installTokenizerChatPortraitPatch({runtime:f.runtime}).status,'installed');
 assert.equal(installTokenizerChatPortraitPatch({runtime:f.runtime}).status,'already-installed');
 let calls=0;
 f.runtime.libWrapper.register('portrait-module','CONFIG.ChatMessage.documentClass.prototype.renderHTML',async function(wrapped,...args){calls++;return wrapped(...args);},'WRAPPER');
 assert.equal(installTokenizerChatPortraitPatch({runtime:f.runtime}).status,'already-installed');
 assert.equal(f.calls.length,2);
 await f.message.renderHTML();assert.equal(calls,1);assert.equal(f.image().style.transform,'none');assert.equal(f.image().src,'avatar.webp');
});

test('matching native portrait renderer is corrected on later core generations',async()=>{
 const f=environment({generation:15});assert.equal(installTokenizerChatPortraitPatch({runtime:f.runtime}).status,'installed');
 await f.calls[0][2].call(f.message,f.message.renderHTML.bind(f.message));
 assert.equal(f.image().style.transform,'none');assert.equal(f.image().style.maskImage,'none');
});
