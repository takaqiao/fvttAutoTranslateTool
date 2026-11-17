// PF2e Toolbelt's betterInventory "split item" option (3.61) turns every stacked quantity on an actor
// sheet into a split link, on each renderActorSheet. Per row it reads the quantity span's innerText and
// then replaces the span's children, so every row after the first forces another synchronous layout of
// the whole page. On the 2026-11-17 bob-1 copy that hook cost 8-9 ms per party sheet render, about
// 0.6 ms more per stacked row, and 0.2-0.4 ms with the option off.
//
// The span holds only {{item.quantity}}, so its text content is the same value without layout. A
// renderActorSheet hook registered in setup, ahead of Toolbelt's (registered in ready), gives each such
// span its own innerText that reads textContent. The render hooks run synchronously, so a microtask
// queued in the first one removes the overrides after the whole chain, Toolbelt's included, has run.

const TOOLBELT='pf2e-toolbelt';
const SETTING='betterInventory.splitItem';
const QUANTITIES='.items [data-item-id] .quantity span';
const installed=new WeakSet();

// The quantity is plain text, which innerText only differs from by collapsed outer whitespace.
function readText(){return this.textContent.trim();}
// A write ends the override for this element and goes to the native setter.
function writeText(value){delete this.innerText;this.innerText=value;}

export function splitItemEnabled(runtime=globalThis){
  try{return runtime.game.settings.get(TOOLBELT,SETTING)===true;}catch{return false;}
}

export function installToolbeltSplitLayoutPatch({runtime=globalThis,report=()=>{}}={}){
  const stats={renders:0,spans:0};
  const finish=(status,reason)=>{const result={feature:'toolbeltSplitLayout',status,...(reason?{reason}:{}),stats};report(result);return result;};
  if(!runtime.game?.modules?.get?.(TOOLBELT)?.active)return finish('skipped','module-inactive');
  if(installed.has(runtime.Hooks))return finish('already-installed');

  const pending=[];
  const restore=()=>{
    for(const span of pending.splice(0))if(Object.getOwnPropertyDescriptor(span,'innerText')?.get===readText)delete span.innerText;
  };
  runtime.Hooks.on('renderActorSheet',(app,html)=>{
    // Toolbelt leaves sheets the user doesn't own alone.
    if(!app?.actor?.isOwner||!splitItemEnabled(runtime))return;
    const root=html?.[0]??html;
    const spans=root?.querySelector?.('.tab.inventory')?.querySelectorAll?.(QUANTITIES);
    if(!spans?.length)return;
    if(!pending.length)runtime.queueMicrotask(restore);
    for(const span of spans){
      Object.defineProperty(span,'innerText',{get:readText,set:writeText,configurable:true});
      pending.push(span);
    }
    stats.renders++;stats.spans+=spans.length;
  });
  installed.add(runtime.Hooks);
  return finish('installed');
}
