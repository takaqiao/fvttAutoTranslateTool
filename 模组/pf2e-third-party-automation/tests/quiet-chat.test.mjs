import test from 'node:test';
import assert from 'node:assert/strict';
import * as quietChat from '../scripts/quiet-chat.mjs';
const {createQuietChat}=quietChat;
const ID='pf2e-third-party-automation',HELPER='pf2e-flatcheck-helper',HIDDEN='tpa-quiet-message';

function element(messageId){
 const node=()=>({dataset:{},children:[],textContent:'',innerHTML:'',append(child){child.parent=this;this.children.push(child)},remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this)},insertAdjacentHTML(_position,html){this.innerHTML+=html},querySelectorAll(){return this.children.filter(child=>child.dataset.tpaQuietNote)},querySelector(){return null}});
 const root=node(),classes=new Set();root.dataset.messageId=messageId;root.matches=selector=>selector==='.message[data-message-id]';root.classList={toggle(name,enabled){enabled?classes.add(name):classes.delete(name)},contains:name=>classes.has(name)};root.ownerDocument={createElement:node};return root;
}
function fixture({status='failed',native=true,attack=false,mode='public',index=0}={}){
 const gm={id:'gm',isGM:true},player={id:'player',isGM:false},other={id:'other',isGM:false},users=new Map([gm,player,other].map(user=>[user.id,user])),messages=new Map(),hooks=new Map(),refreshed=[];
 const game={user:gm,users,messages};let reads=0;
 const values=messages.values.bind(messages);messages.values=()=>{reads++;return values()};
 const make=(id,data={})=>{
  const message={id,author:player,speaker:{actor:'pc'},flags:{},rolls:[],blind:false,whisper:[],logged:true,...data};
  Object.defineProperty(message,'isContentVisible',{configurable:true,get(){return (!this.blind||game.user.isGM)&&(!this.whisper.length||this.whisper.includes(game.user.id)||this.author.id===game.user.id&&!this.blind)}});
  messages.set(id,message);return message;
 };
 const parent=make('activity',{flags:{pf2e:{origin:{actor:'Actor.pc',type:'feat'}}}});
 const proof={activityMessageId:parent.id,nonce:'attempt',scope:'dual-strike',actorUuid:'Actor.pc',userId:'player',index,status,nativeMessageId:native?'flat':null};
 const checks={target:{type:'concealed',finalDc:5,roll:status==='passed'?15:4}};
 const receipt=make('receipt',{blind:true,whisper:['gm'],flags:{[ID]:{dualStrikeFlatCheck:proof},[HELPER]:{flatchecks:checks,revealed:true}}});
 const source=native?make('flat',{blind:mode==='blind',whisper:mode==='public'?[]:mode==='self'?['player']:['gm'],rolls:[{total:status==='passed'?15:4}],flags:{[ID]:{dualStrikeFlatCheckRoll:{receiptId:receipt.id,nonce:'attempt'}},pf2e:{context:{type:'flat-check',dc:{value:5}}}}}):null;
 const strike=attack?make('attack',{blind:source?.blind??false,whisper:source?.whisper??[],rolls:[{total:25}],flags:{pf2e:{context:{type:'attack-roll'}},[HELPER]:{flatchecks:structuredClone(checks),revealed:true},[ID]:{dualStrikeFlatCheckId:receipt.id,nativeOwnerOperation:{activityMessageId:parent.id,nonce:'attempt',scope:'dual-strike'}}}}):null;
 const quiet=createQuietChat({game,refresh:message=>refreshed.push(message.id)});
 quiet.register({Hooks:{on(event,fn){hooks.set(event,fn)},once(event,fn){hooks.set(event,fn)}}});
 const render=(message,root=element(message.id))=>{quiet.render(message,root);return root};
 const fire=(event,message)=>hooks.get(event)?.(message);
 const summary=kind=>make(kind,{author:gm,whisper:['gm'],content:'<p>秘密目标：success，惊惧 2 → 1。</p>',flags:{[ID]:{usageGenerated:true,...kind==='alarm'?{socialAlarm:{checkMessageId:'check',usageMessageId:parent.id}}:{reactionChecks:{kind:'pointed-result',checkId:'check',usageId:parent.id}}}}});
 return {game,gm,player,other,messages,make,parent,receipt,proof,source,strike,quiet,render,fire,summary,refreshed,reads:()=>reads};
}
const notes=root=>root.children.filter(child=>child.dataset.tpaQuietNote);
const hidden=root=>root.classList.contains(HIDDEN);

test('verified successful flat dice live on the attack Utility result without another chat card',()=>{
 const f=fixture({status:'passed',attack:true}),saved=JSON.stringify(f.strike.flags),count=f.messages.size;
 assert.equal(hidden(f.render(f.receipt)),true);assert.equal(hidden(f.render(f.source)),true);assert.equal(hidden(f.render(f.strike)),false);
 assert.equal(notes(f.render(f.parent)).length,0);assert.equal(JSON.stringify(f.strike.flags),saved);assert.equal(f.messages.size,count);
});

test('failed flat dice appear once on their activity and keep the saved native roll',()=>{
 const f=fixture(),saved=JSON.stringify(f.parent),root=f.render(f.parent);f.render(f.parent,root);
 assert.equal(notes(root).length,1);assert.match(notes(root)[0].textContent,/第1击.*4.*DC 5.*未通过/);
 assert.equal(hidden(f.render(f.source)),true);assert.equal(hidden(f.render(f.receipt)),true);assert.equal(f.source.rolls[0].total,4);assert.equal(JSON.stringify(f.parent),saved);
});

test('pending native dice are attached before settlement so the independent result never flashes',()=>{
 const f=fixture({status:'pending'});f.proof.nativeMessageId=null;
 assert.equal(hidden(f.render(f.source)),true);const root=f.render(f.parent);assert.equal(notes(root).length,1);assert.match(notes(root)[0].textContent,/4.*待结算/);
});

test('a passed flat die without an attack remains readable on the original activity',()=>{
 const f=fixture({status:'passed'});assert.equal(hidden(f.render(f.source)),true);assert.match(notes(f.render(f.parent))[0].textContent,/15.*通过/);
});

for(const cause of ['missing parent','different actor','roll parent','unreadable parent','missing receipt','changed nonce','missing total'])test(`flat result stays visible with ${cause}`,()=>{
 const f=fixture();
 if(cause==='missing parent')f.messages.delete(f.parent.id);
 if(cause==='different actor')f.parent.speaker.actor='another';
 if(cause==='roll parent')f.parent.rolls=[{total:1}];
 if(cause==='unreadable parent')Object.defineProperty(f.parent,'isContentVisible',{value:false});
 if(cause==='missing receipt')f.messages.delete(f.receipt.id);
 if(cause==='changed nonce')f.source.flags[ID].dualStrikeFlatCheckRoll.nonce='different';
 if(cause==='missing total')f.source.rolls=[];
 assert.equal(hidden(f.render(f.source)),false);assert.equal(notes(f.render(f.parent)).length,0);
});

test('a failed receipt remains a visible fallback when its native result is unavailable',()=>{
 for(const native of [false,true]){const f=fixture({native});if(native)f.messages.delete(f.source.id);assert.equal(hidden(f.render(f.receipt)),false);assert.equal(notes(f.render(f.parent)).length,0);}
});

for(const mode of ['blind','gm','self'])test(`${mode} flat notes follow the source audience even on a public activity`,()=>{
 const f=fixture({mode});
 for(const user of [f.gm,f.player,f.other]){f.game.user=user;const readable=mode==='self'?user===f.player:mode==='blind'?user===f.gm:user!==f.other;assert.equal(notes(f.render(f.parent)).length,readable?1:0,`${mode} ${user.id}`);assert.equal(hidden(f.render(f.source)),readable);}
});

test('native content visibility remains authoritative for notes',()=>{
 const f=fixture();Object.defineProperty(f.source,'isContentVisible',{value:false});assert.equal(notes(f.render(f.parent)).length,0);assert.equal(hidden(f.render(f.source)),false);
});

test('a hidden native DC cannot become a visible player note',()=>{
 const f=fixture({mode:'gm'});f.source.flags.pf2e.context.dc.visible=false;f.game.user=f.player;
 assert.equal(notes(f.render(f.parent)).length,0);assert.equal(hidden(f.render(f.source)),false);
});

test('two attempts for the same attack index cannot duplicate a note or hide the unselected evidence',()=>{
 const f=fixture(),second=f.make('second-receipt',{...f.receipt,id:'second-receipt',flags:{[ID]:{dualStrikeFlatCheck:{...f.proof,nonce:'second',nativeMessageId:'second-flat'}}}}),source=f.make('second-flat',{...f.source,id:'second-flat',flags:{...f.source.flags,[ID]:{dualStrikeFlatCheckRoll:{receiptId:second.id,nonce:'second'}}}});
 const root=f.render(f.parent);assert.equal(notes(root).length,1);assert.equal([f.source,source].filter(message=>hidden(f.render(message))).length,1);
});

for(const order of [['receipt','flat','second-receipt','second-flat'],['second-flat','receipt','flat','second-receipt'],['second-receipt','receipt','second-flat','flat']])test(`same-index evidence stays consistent when rendered in order ${order.join(', ')}`,()=>{
 const f=fixture();f.source.timestamp=100;
 f.make('second-receipt',{...f.receipt,id:'second-receipt',flags:{[ID]:{dualStrikeFlatCheck:{...f.proof,nonce:'second',nativeMessageId:'second-flat'}}}});
 const second=f.make('second-flat',{...f.source,id:'second-flat',timestamp:200,rolls:[{total:3}],flags:{...f.source.flags,[ID]:{dualStrikeFlatCheckRoll:{receiptId:'second-receipt',nonce:'second'}}}});
 const parent=f.render(f.parent);assert.match(notes(parent)[0].textContent,/平检 3 /);
 for(const id of order)f.render(f.messages.get(id));
 assert.equal(hidden(f.render(f.source)),false,'the result omitted from the parent must stay independently visible');
 assert.equal(hidden(f.render(second)),true);assert.match(notes(parent)[0].textContent,/平检 3 /);
});

for(const kind of ['alarm','pointed'])test(`${kind} GM summary is attached to the skill card without public secret data`,()=>{
 const f=fixture(),check=f.make('check',{rolls:[{total:21}],flags:{pf2e:{context:{type:'skill-check'}}}}),source=f.summary(kind),saved=JSON.stringify(check);
 assert.equal(hidden(f.render(source)),true);const root=f.render(check);assert.equal(notes(root).length,1);assert.match(notes(root)[0].innerHTML,/秘密目标/);assert.equal(JSON.stringify(check),saved);
 f.game.user=f.player;assert.equal(notes(f.render(check)).length,0);assert.equal(hidden(f.render(source)),false);
 f.game.user=f.gm;f.messages.delete(check.id);f.fire('deleteChatMessage',check);assert.equal(hidden(f.render(source)),false);
});

test('unrelated automation, paid originals and Workbench blind cards keep their native presentation',()=>{
 const f=fixture();
 for(const flags of [{[ID]:{usageGenerated:true}},{[ID]:{reference:{kind:'paid-reroll-original'}}},{'xdy-pf2e-workbench':{hidden:true}},{[ID]:{reactionChecks:{kind:'reaction-use'}}}]){
  const message=f.make('ordinary',{blind:true,whisper:['gm'],flags});assert.equal(hidden(f.render(message)),false);assert.equal(notes(f.render(message)).length,0);
 }
});

test('damage result receipts hide without altering their privacy or saved data',()=>{
 const f=fixture(),message=f.make('damage',{blind:true,whisper:['gm'],flags:{[ID]:{nativeOwnerDamageResult:{activityMessageId:f.parent.id,nonce:'damage-attempt',scope:'test',actorUuid:'Actor.pc',type:'damage',roll:{total:9}}}}}),saved=JSON.stringify(message);
 assert.equal(hidden(f.render(message)),true);assert.equal(JSON.stringify(message),saved);
});

test('historical relations are indexed once and parent deletion immediately refreshes the native fallback',()=>{
 const f=fixture();f.render(f.parent);const count=f.reads();for(let i=0;i<10;i++)f.render(f.source);assert.equal(f.reads(),count);
 f.messages.delete(f.parent.id);f.fire('deleteChatMessage',f.parent);assert.ok(f.refreshed.includes(f.source.id));assert.equal(hidden(f.render(f.source)),false);
});

test('late source loading and source deletion refresh the original card without scanning all messages',()=>{
 const f=fixture(),source=f.source;f.messages.delete(source.id);assert.equal(notes(f.render(f.parent)).length,0);const reads=f.reads();
 f.messages.set(source.id,source);f.fire('createChatMessage',source);assert.ok(f.refreshed.includes(f.parent.id));assert.equal(notes(f.render(f.parent)).length,1);
 f.messages.delete(source.id);f.fire('deleteChatMessage',source);assert.equal(notes(f.render(f.parent)).length,0);assert.equal(hidden(f.render(f.receipt)),false);assert.equal(f.reads(),reads);
});

test('rendering before the creation hook still attaches a newly loaded native result',()=>{
 const f=fixture({status:'pending'});f.proof.nativeMessageId=null;f.messages.delete(f.source.id);f.render(f.parent);const reads=f.reads();
 f.messages.set(f.source.id,f.source);assert.equal(hidden(f.render(f.source)),true);assert.equal(notes(f.render(f.parent)).length,1);assert.equal(f.reads(),reads);
});

test('chat registration captures the real game at init before historical messages render',()=>{
 assert.equal(typeof quietChat.registerQuietChat,'function');
 const f=fixture(),original=globalThis.game,once=new Map(),on=new Map(),Hooks={once:(event,fn)=>once.set(event,fn),on:(event,fn)=>on.set(event,fn)};
 try{
  globalThis.game={};quietChat.registerQuietChat({Hooks});assert.equal(on.has('renderChatMessageHTML'),false);
  globalThis.game=f.game;once.get('init')();
  const root=element(f.receipt.id);on.get('renderChatMessageHTML')(f.receipt,root);assert.equal(hidden(root),true);
 }finally{globalThis.game=original;}
});
