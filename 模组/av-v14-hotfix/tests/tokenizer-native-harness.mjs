import vm from 'node:vm';
import {readFile} from 'node:fs/promises';

export const native = JSON.parse(await readFile(new URL('./fixtures/tokenizer-chat-native.json', import.meta.url), 'utf8'));
const selector = ':scope > header.message-header > .portrait > img';

export function environment({sourceVersion='8.6.0', version=sourceVersion, generation=14, systemId='pf2e', renderSource=native[sourceVersion].renderHTML}={}) {
  const calls = [], header = {children:[], classList:{toggle(){}}, prepend(node){this.children.unshift(node);}, append(node){this.children.push(node);}};
  const html = {dataset:{}, addEventListener(){}, querySelector(value){
    if (value === 'header.message-header') return header;
    if (value === selector) return header.children.find(node => node.classes?.includes('portrait'))?.children[0] ?? null;
    return null;
  }};
  class Base {async renderHTML(options){this.nativeOptions=options;return html;}}
  class DamageRoll {}
  const listeners = {listen:async()=>{}};
  const runtime = {Base, console,
    game:{release:{generation}, system:{id:systemId,version}, settings:{get:()=>true}},
    K:{enrichHTML:async text=>text}, ws:listeners, Ts:listeners, cn:DamageRoll, ln:DamageRoll,
    CONST:{CHAT_MESSAGE_STYLES:{OOC:1}}, canvas:{ready:false},
    foundry:{helpers:{media:{ImageHelper:{hasImageExtension:value=>/\.webp(?:\?|$)/.test(value)}}}},
    isDefaultTokenImage:()=>false, createHTMLElement:(tag,options)=>({tag,...options}),
    document:{createElement:tag=>({tag,style:{},getAttribute(name){return this[name];}})},
    htmlQuery:(node,value)=>node.querySelector(value), htmlQueryAll:()=>[],
    CriticalHitAndFumbleCards:{appendButtons(){}}, ChatCards:listeners,
    UserVisibilityPF2e:{processMessageSender(){},process(){}},
    libWrapper:{register:(...args)=>calls.push(args)}
  };
  // The complete native render method runs unchanged. Other chat-card listeners
  // and private hover/initiative methods are outside this portrait fixture.
  const ChatMessage = vm.runInNewContext(`(class ChatMessage extends Base {
    ${renderSource}
    #highlightDoS(){} #appendSetAsInitiative(){} #onHoverIn(){} #onHoverOut(){}
  })`,runtime);
  runtime.CONFIG = {ChatMessage:{documentClass:ChatMessage}};
  const actor = {name:'Speaker',img:'avatar.webp',flags:{'tokenizer-2':{layerStack:[{}]}},
    prototypeToken:{texture:{src:'subject.webp',scaleX:2},ring:{enabled:true,subject:{texture:'subject.webp'}}}};
  const message = Object.assign(new ChatMessage(),{speakerActor:actor,token:null,isContentVisible:true,style:0,
    flavor:'',getRollData:()=>({}),flags:{pf2e:{}},rolls:[]});
  return {runtime,calls,message,actor,html,image:()=>html.querySelector(selector)};
}
