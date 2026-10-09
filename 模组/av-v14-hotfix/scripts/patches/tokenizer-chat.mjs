import {sha256Fallback} from '../source-hash.mjs';

const RENDER_HTML_HASHES = new Set([
  '2002b840988a5456d11b20cbd235a2d067d046678af3e1781090c2990849338c', // PF2e 8.5.1
  '8a7d64e3a8c487efbf6f0ac3ff670af5f9d4719f2e86e18ffcdf9fe0c3075c94' // PF2e 8.6.0
]);

export function createTokenizerChatPortraitWrapper(){
  return async function(wrapped,...args){
    const html=await wrapped(...args);
    const flags=this.speakerActor?.flags?.['tokenizer-2'];
    if(!this.isContentVisible||!flags||!Object.keys(flags).length)return html;
    const image=html?.querySelector(':scope > header.message-header > .portrait > img');
    if(image){
      const actor=this.speakerActor,token=this.token??actor.prototypeToken;
      const subject=token?.ring?.enabled?token.ring.subject?.texture:null;
      const avatar=actor.img;
      // A ring subject is padded for map compositing, so its visible subject
      // can occupy only a small fraction of a chat thumbnail. Use the saved
      // avatar only when the native renderer selected that exact subject.
      if(subject&&image.getAttribute('src')===subject&&typeof avatar==='string'&&avatar.trim()
        &&avatar!==(globalThis.CONST?.DEFAULT_TOKEN??'icons/svg/mystery-man.svg'))image.src=avatar;
      // PF2e applies map texture scale plus ring padding compensation to this
      // 36px avatar, then adds a radial mask. Tokenizer's oversized map scale
      // is not a chat-image scale: keep native sizing.
      image.style.transform='none';
      image.style.maskImage='none';
      image.style.webkitMaskImage='none';
    }
    return html;
  };
}

const installed=new WeakSet();
export function installTokenizerChatPortraitPatch({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const finish=status=>{const result={feature:'tokenizerChat',status};report(result);return result;};
  if(runtime.game?.system?.id!=='pf2e')return finish('unsupported-system');
  const prototype=runtime.CONFIG?.ChatMessage?.documentClass?.prototype;
  if(typeof prototype?.renderHTML!=='function'||typeof runtime.libWrapper?.register!=='function')return finish('unsupported-runtime');
  if(installed.has(prototype))return finish('already-installed');
  // Keep setup registration synchronous, before the first chat history render.
  if(!RENDER_HTML_HASHES.has(sha256Fallback(Function.prototype.toString.call(prototype.renderHTML))))return finish('unsupported-source');
  runtime.libWrapper.register(moduleId,'CONFIG.ChatMessage.documentClass.prototype.renderHTML',createTokenizerChatPortraitWrapper(),'WRAPPER');
  installed.add(prototype);
  return finish('installed');
}
