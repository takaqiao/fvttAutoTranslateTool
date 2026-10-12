// Just enough DOM for core's timestamp refresh and the patch. A write is counted
// each time the inherited textContent setter replaces an element's children;
// real childList mutation records are a browser matter.
export function timestampEnvironment(){
  class Node{
    constructor(nodeType){this.nodeType=nodeType;this.childNodes=[];this.writes=0;}
    get firstChild(){return this.childNodes[0]??null;}
    get lastChild(){return this.childNodes.at(-1)??null;}
    append(...nodes){this.childNodes.push(...nodes);return this;}
  }
  class Text extends Node{constructor(data){super(3);this.data=data;}}
  const isStamp=node=>node.className==='message-timestamp';
  class Element extends Node{
    constructor(className=''){super(1);this.className=className;this.dataset={};}
    querySelector(selector){
      if(selector!=='.message-timestamp')throw Error('Unmodeled selector '+selector);
      return this.childNodes.find(isStamp)??null;
    }
  }
  const text=node=>node.nodeType===3?node.data:node.nodeType===1?node.childNodes.map(text).join(''):'';
  Object.defineProperty(Node.prototype,'textContent',{enumerable:true,configurable:true,
    get(){return text(this);},
    set(value){
      // As in WebIDL, the conversion comes first: a value that cannot become a string writes nothing.
      value=value==null?'':`${value}`;
      this.writes++;this.childNodes=value?[new Text(value)]:[];
    }});

  // Every entry of `body` stands for a rendered `.chat-message[data-message-id]`.
  const document={body:[],querySelectorAll(selector){
    if(selector==='.chat-message[data-message-id]')return [...this.body];
    if(selector==='.chat-message[data-message-id] .message-timestamp')return this.body.flatMap(li=>li.childNodes.filter(isStamp));
    throw Error('Unmodeled selector '+selector);
  }};
  const records=new Map(),labels=new Map();
  const game={messages:{get:id=>records.get(id)}};
  const foundry={utils:{timeSince:timestamp=>labels.get(timestamp)}};
  // Same statements as core's ChatLog#updateTimestamps.
  class ChatLog{
    updateTimestamps(){
      for(const li of document.querySelectorAll('.chat-message[data-message-id]')){
        const message=game.messages.get(li.dataset.messageId);
        if(!message?.timestamp)return;
        const stamp=li.querySelector('.message-timestamp');
        if(stamp)stamp.textContent=foundry.utils.timeSince(message.timestamp);
      }
    }
  }
  foundry.applications={sidebar:{tabs:{ChatLog}}};
  const registrations=[];
  const libWrapper={register(moduleId,target,fn,type){registrations.push({moduleId,target,fn,type});return registrations.length;}};

  /** Add a displayed message; `age(label)` sets what the next refresh will show for it. */
  function message(label='1 min ago'){
    const timestamp=records.size+1,id=`message-${timestamp}`,li=new Element('chat-message'),stamp=new Element('message-timestamp');
    li.dataset.messageId=id;li.append(stamp.append(new Text(label)));
    records.set(id,{timestamp});labels.set(timestamp,label);document.body.push(li);
    return {li,stamp,age:next=>labels.set(timestamp,next)};
  }
  const log=new ChatLog(),core=()=>log.updateTimestamps();
  // libWrapper calls a WRAPPER with the instance as `this` and a bound continuation.
  const refresh=(wrapped=core,...args)=>Reflect.apply(registrations.at(-1).fn,log,[wrapped,...args]);
  return {runtime:{Node,document,foundry,libWrapper},Node,Element,Text,ChatLog,registrations,message,core,refresh};
}
