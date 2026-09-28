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
  if((runtime.game?.release?.generation??Number.parseInt(runtime.game?.version,10))!==14)return finish('unsupported-core');
  if(runtime.game?.system?.id!=='pf2e'||runtime.game.system.version!=='8.5.1')return finish('unsupported-system');
  const prototype=runtime.CONFIG?.ChatMessage?.documentClass?.prototype;
  if(typeof prototype?.renderHTML!=='function'||typeof runtime.libWrapper?.register!=='function')return finish('unsupported-runtime');
  if(installed.has(prototype))return finish('already-installed');
  runtime.libWrapper.register(moduleId,'CONFIG.ChatMessage.documentClass.prototype.renderHTML',createTokenizerChatPortraitWrapper(),'WRAPPER');
  installed.add(prototype);
  return finish('installed');
}
