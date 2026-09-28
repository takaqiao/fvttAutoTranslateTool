import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {nativeHooks} from './hook-harness.mjs';
export const fixture=JSON.parse(await readFile(new URL('./fixtures/timestamps-core-14.367.json',import.meta.url),'utf8'));

// Deliberately bounded DOM boundary doubles. Proxies give these simulated host
// accessors native-looking Function#toString, without replacing that method.
// Chrome verifies actual DOM/MutationObserver semantics separately.
const host=fn=>new Proxy(fn,{});
export function timestampEnvironment() {
  class Node {
    constructor(type){this._type=type;this._children=[];this._parent=null;this.writes=0;}
    append(...nodes){for(const node of nodes){node.remove();this._children.push(node);node._parent=this;}}
    remove(){if(this._parent)this._parent._children.splice(this._parent._children.indexOf(this),1);this._parent=null;}
  }
  class CharacterData extends Node {constructor(data){super(3);this._data=data;}}
  class Text extends CharacterData {}
  class Element extends Node {constructor(classes=[]){super(1);this._classes=new Set(classes);this.dataset={};}}
  class Document extends Node {constructor(){super(9);}}
  const descriptor=(target,key,get,set)=>Object.defineProperty(target,key,{get:host(get),...(set?{set:host(set)}:{}),enumerable:true,configurable:true});
  const text=node=>node._type===3?node._data:node._children.map(text).join('');
  descriptor(Node.prototype,'textContent',function(){if(!(this instanceof Node))throw TypeError('Illegal invocation');return text(this);},function(value){
    if(!(this instanceof Node))throw TypeError('Illegal invocation');if(typeof value==='symbol')throw TypeError('Symbol cannot convert');
    value=value===null?'':String(value);this.writes++;for(const child of this._children)child._parent=null;this._children=[];
    if(value)this.append(new Text(value));
  });
  descriptor(Node.prototype,'firstChild',function(){return this._children[0]??null;});
  descriptor(Node.prototype,'nextSibling',function(){return this._parent?this._parent._children[this._parent._children.indexOf(this)+1]??null:null;});
  descriptor(Node.prototype,'nodeType',function(){return this._type;});
  descriptor(CharacterData.prototype,'data',function(){return this._data;},function(value){this._data=String(value);});
  const matches=(node,selector)=>{
    if(!(node instanceof Element))return false;
    const message=n=>n._classes.has('chat-message')&&Object.hasOwn(n.dataset,'messageId');
    if(selector==='.chat-message[data-message-id]')return message(node);
    if(selector==='.message-timestamp')return node._classes.has('message-timestamp');
    if(selector!=='.chat-message[data-message-id] .message-timestamp')throw Error('Unmodeled selector '+selector);
    if(!node._classes.has('message-timestamp'))return false;
    for(let p=node._parent;p;p=p._parent)if(p instanceof Element&&message(p))return true;
    return false;
  };
  const query=function(selector){const out=[];function visit(n){for(const child of n._children){if(matches(child,selector))out.push(child);visit(child);}}visit(this);return out;};
  Object.defineProperty(Element.prototype,'matches',{value:host(function(selector){return matches(this,selector);}),writable:true,configurable:true});
  for(const type of [Element,Document])Object.defineProperty(type.prototype,'querySelectorAll',{value:host(query),writable:true,configurable:true});
  const runtime={game:{version:'14.368',user:{isGM:true}},Node:host(Node),Element:host(Element),Document:host(Document),CharacterData:host(CharacterData),Text:host(Text),WeakRef,FinalizationRegistry,queueMicrotask,document:new Document(),Hooks:nativeHooks()};
  const f=fixture.fragments;
  const context=vm.createContext({Hooks$1:runtime.Hooks,CONFIG:{debug:{applications:false}},game:runtime.game});
  runtime.foundry=vm.runInContext(`class ApplicationV2 {
    static BASE_APPLICATION; static emittedEvents=[];
    static ${f.inheritanceChain.source}
    ${f.applicationEvent.source}
    ${f.dispatchEvent.source}
    ${f.callHooks.source}
    dispatchEvent(){}
  } ApplicationV2.BASE_APPLICATION=ApplicationV2;
  class ChatMessage {${f.messageHTML.source} async #renderRollContent(){}}
  class ChatLog extends ApplicationV2 {static ${f.renderMessage.source}}
  class ChatPopout extends ApplicationV2 {${f.popoutHTML.source}}
  ({documents:{ChatMessage},applications:{api:{ApplicationV2},sidebar:{tabs:{ChatLog},apps:{ChatPopout}}}})`,context);
  const stamp=(value='1 min ago')=>{const message=new Element(['chat-message']);message.dataset.messageId='message';const element=new Element(['message-timestamp']);message.append(element);element.append(new Text(value));return {message,element};};
  return {runtime,Node,Element,Text,Document,CharacterData,stamp};
}
