import test from 'node:test';
import assert from 'node:assert/strict';
import {createTokenizerChatPortraitWrapper,installTokenizerChatPortraitPatch} from '../scripts/patches/tokenizer-chat.mjs';

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

test('registers once for the verified system and declines unsupported versions',()=>{
  const make=(generation=14,version='8.5.1')=>{
    const calls=[];
    return {calls,runtime:{game:{release:{generation},system:{id:'pf2e',version}},CONFIG:{ChatMessage:{documentClass:class {renderHTML(){}}}},libWrapper:{register:(...args)=>calls.push(args)}}};
  };
  const {runtime,calls}=make();
  assert.equal(installTokenizerChatPortraitPatch({runtime}).status,'installed');
  assert.equal(installTokenizerChatPortraitPatch({runtime}).status,'already-installed');
  assert.equal(calls.length,1);
  assert.equal(calls[0][1],'CONFIG.ChatMessage.documentClass.prototype.renderHTML');
  assert.equal(calls[0][3],'WRAPPER');
  for(const args of [[13,'8.5.1'],[14,'8.5.2']]){
    const {runtime,calls}=make(...args);
    assert.match(installTokenizerChatPortraitPatch({runtime}).status,/^unsupported-/);
    assert.equal(calls.length,0);
  }
});
