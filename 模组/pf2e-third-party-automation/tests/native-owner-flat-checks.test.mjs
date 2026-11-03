import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createNativeOwnerOperations} from '../scripts/native-owner-operations.mjs';

const ID='pf2e-third-party-automation',HELPER='pf2e-flatcheck-helper';
const source=fs.readFileSync(process.env.PF2E_UTILITY_BUNDLE,'utf8');
const {mergeObject}=await import(pathToFileURL(join(process.env.FVTT_NATIVE_APP,'common/utils/helpers.mjs')).href);
const section=(start,end)=>{const a=source.indexOf(start),b=source.indexOf(end,a);assert.ok(a>=0&&b>a,`Missing Utility Buttons source: ${start}`);return source.slice(a,b);};
const utilitySource=[section('function Ze(','var I ='),section('var ft =','//#endregion'),section('async function pt(','//#endregion'),section('async function bt(','async function xt('),section('function wt(','async function Et('),section('function Dt(','async function Ot(')].join('\n');
const set=(object,path,value)=>{const parts=path.split('.');let at=object;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=value;};
const turn=()=>new Promise(resolve=>setImmediate(resolve));
const sureStrike='Compendium.pf2e.spell-effects.Item.fpGDAz2v5PG0zUSl',blindFight='Compendium.pf2e.feats-srd.Item.y2XeMe1F18lIyo59';

async function fixture(options,run){
 const {condition='concealed',total=15,additional=[],adjustments=[],autoResults,scope='dual-strike',coreMode='public',cancel=false,flatCancel=false,missing=false,mutateReceipt,effects=[]}=options;
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:true,settings:{showCheckDialogs:true},targets:new Set()},users=new Map([[gm.id,gm],[player.id,player]]);users.activeGM=gm;
 const messages=new Map(),entries=new Map(),ownerHandlers=new Map(),gmHandlers=new Map(),published=[],order=[],flatContexts=[],providerOptions=[];let seq=0,hookSeq=0,attacks=0,flatDice=0;
 const Hooks={on(event,fn){const id=++hookSeq;entries.set(id,{event,fn});return id;},off(_event,id){entries.delete(id);},call(event,...args){for(const entry of [...entries.values()])if(entry.event===event)entry.fn(...args);}};
 const conditions=list=>({stored:list.map(slug=>({slug})),bySlug:slug=>list.includes(slug)?[{slug}]:[]});
 const actor={id:'pc',uuid:'Actor.pc',type:'character',testUserPermission:user=>user.id===player.id,items:new Map(),system:{actions:[]},conditions:conditions([]),getRollOptions:domains=>['self:level:5',...(domains??[]).map(domain=>`domain:${domain}`)],getSelfRollOptions:()=>['self:level:5']};
 for(const [index,effect]of effects.entries())actor.items.set(`effect-${index}`,{id:`effect-${index}`,type:'effect',isExpired:false,rules:effect.sourceId===sureStrike?[{key:'RollTwice',keep:'higher',selector:['attack-roll'],test:()=>true}]:[],...effect});
 const targetActor={uuid:'Actor.target',conditions:conditions(condition?[condition]:[]),getSelfRollOptions:()=>['target:condition:'+condition]};
 const scene={id:'s',tokens:new Map()},origin={documentName:'Token',id:'pc-token',uuid:'Scene.s.Token.pc-token',actor,parent:scene},target={documentName:'Token',id:'target',uuid:'Scene.s.Token.target',actor:targetActor,parent:scene};origin.object={document:origin,distanceTo:()=>10};target.object={document:target};scene.tokens.set(origin.id,origin);scene.tokens.set(target.id,target);actor.getActiveTokens=()=>[origin];
 const item={id:'weapon',uuid:'Actor.pc.Item.weapon',type:'weapon',actor,name:'Sword',system:{traits:{value:[]}},getRollOptions:()=>['item:melee','item:trait:agile'],isOfType:(...types)=>types.includes('weapon'),async toMessage(_event,{create}){assert.equal(create,false);order.push('intent');return new Message({author:player.id,speaker:{actor:actor.id,scene:scene.id,token:origin.id},content:'<p>weapon description</p>',rolls:[],flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid,type:'weapon'}}},...privacy(coreMode)});}};actor.items.set(item.id,item);
 const privacy=mode=>({blind:mode==='blind',whisper:mode==='public'?[]:mode==='self'?[player.id]:[gm.id]});
 const gmGame={user:gm,users,messages,scenes:new Map([[scene.id,scene]]),modules:new Map([[HELPER,{active:true}]])},ownerGame={user:player,users,messages,scenes:gmGame.scenes,modules:gmGame.modules,settings:{get:(module,key)=>module==='core'&&key==='messageMode'?coreMode:undefined,set(){throw Error('World settings must not change');}}};
 const docs=new Map([[actor.uuid,actor],[target.uuid,target],[origin.uuid,origin]]);
 let utility;
 class Message{
  constructor(data){this.data=structuredClone(data);this.id=data._id;}
  get flags(){return this.data.flags??={};}get speaker(){return this.data.speaker;}get author(){return users.get(this.data.author);}get actor(){return this.speaker?.actor===actor.id?actor:null;}get token(){return scene.tokens.get(this.speaker?.token);}get target(){const uuid=this.flags.pf2e?.context?.target?.token;return uuid?{token:docs.get(uuid)}:null;}get item(){return this.flags.pf2e?.origin?.uuid===item.uuid?item:null;}
  get rolls(){return this.data.rolls??=[];}get isRoll(){return this.rolls.length>0;}get isCheckRoll(){return this.flags.pf2e?.context?.type==='attack-roll';}get flavor(){return this.data.flavor??'';}get blind(){return this.data.blind===true;}get whisper(){return this.data.whisper??[];}
  toObject(){return structuredClone(this.data);}toJSON(){return this.toObject();}updateSource(changes){for(const [path,value]of Object.entries(changes))set(this.data,path,value);}
  async update(changes){mergeObject(this.data,changes,{applyOperators:true});mutateReceipt?.(this);Hooks.call('updateChatMessage',this);return this;}
  static applyMode(data,mode){return {...data,...privacy(mode)};}
  static getSpeaker({actor,token}){return {actor:actor.id,scene:token.parent.id,token:token.id};}
  static async create(data,options){
   if(missing&&data.flags?.[ID]?.dualStrikeFlatCheck)return undefined;
   const message=new Message(data);if(await utility.bt.call(message,async()=>undefined)===false)return undefined;
   message.id=`message-${++seq}`;message.data._id=message.id;messages.set(message.id,message);message.creationOptions=options;published.push(message);Hooks.call('createChatMessage',message);return message;
  }
 }
 class Adjustments{
  getAdditionalSources(){return structuredClone(additional);}getTreatAsAdjustments(){return null;}treatObservedAs(){return null;}
  getDcAdjustment(_type,options,base){providerOptions.push([...options]);return {finalDc:adjustments.reduce((dc,entry)=>dc+Number(entry.value),base),adjustments:structuredClone(adjustments)};}
 }
 const settings=Object.freeze({flatAutoRoll:!!autoResults,flatHideRoll:true,flatAutoReveal:false});
 utility=new Function('e','d','game','$','ke','ot','M','C','Oe','lt','ut','ct','Me','Ke','O','Et','l','XMLSerializer','getDocumentClass','CONFIG',`const I={forCheck:Ze,forRollMessage:Qe,forMixed:$e,lightLevelOptions:()=>[]};${utilitySource};return {bt};`)(HELPER,{id:'pf2e'},ownerGame,{settings},['character'],Adjustments,{toSets:()=>({ignored:new Set(),experimental:new Set()})},{concealed:5,hidden:11,undetected:11,unnoticed:11},{blinded:'hidden',dazzled:'concealed'},()=>null,()=>null,async()=>null,(value,...fns)=>fns.reduce((value,fn)=>fn(value),value),fn=>values=>values.map(fn),([fn])=>values=>[...values].sort((a,b)=>fn(b)-fn(a))[0],async message=>{for(const [key,value]of Object.entries(autoResults)){message.flags[HELPER].flatchecks[key].roll=value;message.rolls.push({total:value,evaluated:true});}return {};},()=>({querySelector:()=>null}),class{serializeToString(){return '<div>Strike</div>';}},type=>type==='ChatMessage'?Message:class Token{}, {sounds:{dice:'dice',lock:'lock'}});
 const roll=value=>({_evaluated:true,total:value,options:{degreeOfSuccess:value===1?0:value===20?3:2},toJSON:()=>({class:'CheckRoll',formula:'1d20',total:value,evaluated:true,options:{}})});
 ownerGame.pf2e={CheckModifier:class{constructor(_label,{modifiers}){this.modifiers=modifiers;}},Check:{async roll(check,context,_event,callback){
  assert.deepEqual(check.modifiers,[]);assert.equal(context.type,'flat-check');assert.equal(context.skipDialog,true);assert.equal(context.rollTwice,false);assert.deepEqual(context.substitutions,[]);
  flatContexts.push(context);order.push('flat');if(flatCancel)return null;flatDice++;
  const evaluated=roll(total),raw=new Message({author:player.id,speaker:{actor:actor.id,scene:scene.id,token:origin.id},rolls:[evaluated.toJSON()],flags:{pf2e:{context:{type:'flat-check',options:[...context.options],dc:context.dc,outcome:'success',target:{token:target.uuid,actor:targetActor.uuid}}}},...privacy(context.messageMode??coreMode)});
  await callback?.(evaluated,'success',raw);return evaluated;
 }}};
 actor.system.actions=[{type:'strike',item,options:['action:strike','self:action:slug:strike'],domains:['attack-roll','strike-attack-roll'],variants:[{async roll(parameters){
  order.push('attack-window');if(cancel)return null;attacks++;order.push('attack');
  await parameters.callback({},'success',new Message({author:player.id,speaker:{actor:actor.id,scene:scene.id,token:origin.id},...privacy(coreMode),flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid},context:{type:'attack-roll',outcome:'success',options:[...parameters.options],dc:{value:20},target:{token:target.uuid,actor:targetActor.uuid}}}},rolls:[{total:25,evaluated:true}]}));
  order.push('afterRoll');return {};
 }}]}];
 const card={id:'activity',author:player,flags:{},async update(changes){for(const [path,value]of Object.entries(changes))set(this,path,value);Hooks.call('updateChatMessage',this);}};messages.set(card.id,card);
 const config={fromUuid:async uuid=>docs.get(uuid),scope,syncTimeoutMs:30},owner=createNativeOwnerOperations({...config,game:ownerGame}),root=createNativeOwnerOperations({...config,game:gmGame});
 owner.register({Hooks,socket:{register(name,fn){ownerHandlers.set(name,fn);},async executeAsUser(name,_user,payload){return gmHandlers.get(name).call({socketdata:{userId:player.id}},payload);}}});
 root.register({Hooks,socket:{register(name,fn){gmHandlers.set(name,fn);},async executeAsUser(name,_user,payload){return ownerHandlers.get(name).call({socketdata:{userId:gm.id}},payload);}}});
 const request={type:'attack',automaticFlatChecks:true,tokenUuid:origin.uuid,weaponId:'weapon',map:0,targetUuid:target.uuid,options:['action:double-slice'],flags:{dualStrikeAttack:{usageMessageId:card.id,index:0}}};
 const attack=(extra={})=>root.run({actor,message:card,user:player},{...request,...extra});
 const prior=globalThis.CONFIG,priorConst=globalThis.CONST;globalThis.CONFIG={ChatMessage:{documentClass:Message},compatibility:{mode:0}};globalThis.CONST={COMPATIBILITY_MODES:{SILENT:0}};
 try{await run({attack,root,owner,card,actor,target,origin,gm,player,users,messages,Message,Hooks,entries,published,order,flatContexts,providerOptions,settings,item,request,gmGame,ownerGame,attacks:()=>attacks,flatDice:()=>flatDice});}finally{await turn();globalThis.CONFIG=prior;globalThis.CONST=priorConst;}
}

for(const [condition,total,passed]of [['concealed',5,true],['concealed',4,false],['hidden',11,true],['hidden',10,false]])test(`${condition} flat check ${total} is resolved before the owner's strike`,()=>fixture({condition,total},async f=>{
 const result=await f.attack();assert.equal(result.status,passed?'rolled':'flat-check-failed');assert.equal(f.attacks(),passed?1:0);assert.equal(f.flatDice(),1);assert.ok(f.order.indexOf('flat')<f.order.indexOf('attack')||!passed);
 const message=f.messages.get(result.messageId);assert.ok(message);assert.equal(message.flags[HELPER].flatchecks.target.roll,total);
 if(passed){assert.equal(message.flags[HELPER].revealed,true);assert.deepEqual(f.order.slice(-2),['attack','afterRoll']);}else assert.notEqual(message.flags.pf2e?.context?.type,'attack-roll');
 assert.equal(f.published.filter(message=>message.flags[HELPER]?.hiddenMsg).length,0);assert.equal(f.entries.size,0);
}));

test('preflight retains the real item, source, target, strike and activity options',()=>fixture({},async f=>{
 f.player.targets.add({document:{uuid:'Scene.s.Token.unrelated'}});await f.attack();
 const options=f.providerOptions.flat();for(const expected of ['item:melee','self:level:5','domain:strike-attack-roll','target:condition:concealed','target:distance:10','action:strike','action:double-slice'])assert.ok(options.includes(expected),expected);
 assert.equal(f.flatContexts[0].token,f.origin);assert.equal(f.flatContexts[0].actor,f.actor);
}));

test('multiple required checks share one flat die against the highest DC',()=>fixture({total:12,additional:[{slot:'curse',type:'curse',baseDc:13}]},async f=>{
 const result=await f.attack();assert.equal(result.status,'flat-check-failed');assert.equal(f.flatDice(),1);assert.equal(f.flatContexts[0].dc.value,13);assert.equal(f.attacks(),0);
 const checks=f.messages.get(result.messageId).flags[HELPER].flatchecks;assert.equal(checks.target.roll,12);assert.equal(checks.curse.roll,12);
}));

for(const total of [3,17])test(`an existing Utility auto result ${total} is reused without another die`,()=>fixture({autoResults:{target:total}},async f=>{
 const result=await f.attack();assert.equal(result.status,total>=5?'rolled':'flat-check-failed');assert.equal(f.flatDice(),0);assert.equal(f.attacks(),total>=5?1:0);assert.equal(f.messages.get(result.messageId).flags[HELPER].flatchecks.target.roll,total);
}));

test('unknown required DC does not count as a passed check',()=>fixture({condition:null,additional:[{slot:'unclear',type:'unknown',baseDc:null}]},async f=>{
 await assert.rejects(f.attack(),/未确认|裁定|DC/);assert.equal(f.flatDice(),0);assert.equal(f.attacks(),0);
}));

for(const [dc,total,passed,dice]of [[1,1,true,0],[20,20,true,1],[20,19,false,1],[21,20,false,0]])test(`flat DC${dc} uses the native total without natural-degree adjustments`,()=>fixture({condition:null,total,additional:[{slot:'custom',type:'custom',baseDc:dc}]},async f=>{
 const result=await f.attack();assert.equal(result.status,passed?'rolled':'flat-check-failed');assert.equal(f.flatDice(),dice);assert.equal(f.attacks(),passed?1:0);
}));

for(const mode of ['blind','self'])test(`the ${mode} core audience survives the native preflight and attack`,()=>fixture({coreMode:mode},async f=>{
 const result=await f.attack(),message=f.messages.get(result.messageId),flat=f.published.find(message=>message.flags.pf2e?.context?.type==='flat-check');
 assert.equal(flat.blind,mode==='blind');assert.deepEqual(flat.whisper,mode==='self'?['player']:['gm']);assert.equal(message.blind,mode==='blind');assert.deepEqual(message.whisper,flat.whisper);
}));

test('a secret check stays private even when core defaults to public',()=>fixture({condition:'undetected',total:15},async f=>{
 await f.attack();const flat=f.published.find(message=>message.flags.pf2e?.context?.type==='flat-check');assert.equal(flat.blind,true);assert.deepEqual(flat.whisper,['gm']);
}));

for(const condition of ['concealed','hidden'])test(`active Sure Strike bypasses only its ${condition} target check`,()=>fixture({condition,effects:[{sourceId:sureStrike}],additional:[{slot:'curse',type:'curse',baseDc:7}]},async f=>{
 const result=await f.attack();assert.equal(result.status,'rolled');assert.equal(f.flatDice(),1);assert.equal(f.flatContexts[0].dc.value,7);assert.equal(f.messages.get(result.messageId).flags[HELPER].flatchecks.target,undefined);assert.ok(f.actor.items.get('effect-0'),'native strike owns fortune consumption');
}));

test('expired Sure Strike does not bypass a concealed target',()=>fixture({total:1,effects:[{sourceId:sureStrike,isExpired:true}]},async f=>{
 assert.equal((await f.attack()).status,'flat-check-failed');assert.equal(f.flatDice(),1);assert.equal(f.attacks(),0);
}));

test('Blind-Fight ignores concealed while leaving other flat checks in place',()=>fixture({effects:[{type:'feat',sourceId:blindFight}],additional:[{slot:'curse',type:'curse',baseDc:7}]},async f=>{
 const result=await f.attack();assert.equal(result.status,'rolled');assert.equal(f.flatContexts[0].dc.value,7);
 const attack=f.messages.get(result.messageId),receipt=f.messages.get(attack.flags[ID].dualStrikeFlatCheckId);
 assert.equal(receipt.flags[HELPER].flatchecks.target,undefined);assert.equal(receipt.flags[HELPER].flatchecks.curse.roll,15);assert.equal(receipt.flags[ID].dualStrikeFlatCheck.status,'passed');assert.equal(receipt.blind,true);assert.deepEqual(receipt.whisper,['gm']);
}));

for(const [name,effect]of [['Sure Strike',{sourceId:sureStrike}],['Blind-Fight',{type:'feat',sourceId:blindFight}]])test(`${name} removes the only concealed check through Foundry's recursive flag update`,()=>fixture({effects:[effect],coreMode:'self'},async f=>{
 const result=await f.attack(),attack=f.messages.get(result.messageId),receipt=f.messages.get(attack.flags[ID].dualStrikeFlatCheckId);
 assert.equal(result.status,'rolled');assert.equal(f.flatDice(),0);assert.equal(f.attacks(),1);assert.deepEqual(receipt.flags[HELPER].flatchecks,{});assert.deepEqual(attack.flags[HELPER].flatchecks,{});
 assert.deepEqual(receipt.flags[ID].dualStrikeFlatCheck.privacy,{messageMode:'self',blind:false,whisper:['player']});assert.equal(receipt.blind,true);assert.deepEqual(receipt.whisper,['gm']);assert.deepEqual(attack.whisper,['player']);
}));

test('Blind-Fight lowers hidden base DC before retaining additive adjustments',()=>fixture({condition:'hidden',total:7,effects:[{type:'feat',sourceId:blindFight}],adjustments:[{label:'curse',value:'+2'}]},async f=>{
 const result=await f.attack(),check=f.messages.get(result.messageId).flags[HELPER].flatchecks.target;assert.equal(result.status,'rolled');assert.equal(check.baseDc,5);assert.equal(check.finalDc,7);assert.deepEqual(check.dcAdjustments,[{label:'curse',value:'+2'}]);
}));

test('ambiguous Blind-Fight DC overrides wait for GM adjudication',()=>fixture({condition:'hidden',effects:[{type:'feat',sourceId:blindFight}],adjustments:[{label:'override',value:'7'}]},async f=>{
 await assert.rejects(f.attack(),/未确认|裁定|DC/);assert.equal(f.flatDice(),0);assert.equal(f.attacks(),0);
}));

test('cancelling the original attack after a passed flat check does not roll an attack',()=>fixture({cancel:true},async f=>{
 assert.deepEqual(await f.attack(),{status:'cancelled'});assert.equal(f.flatDice(),1);assert.equal(f.attacks(),0);
}));

test('a cancelled native preflight cannot be treated as passing',()=>fixture({flatCancel:true},async f=>{
 assert.deepEqual(await f.attack(),{status:'cancelled'});assert.equal(f.attacks(),0);
}));

test('a missing preflight receipt stops before either die',()=>fixture({missing:true},async f=>{
 await assert.rejects(f.attack(),/未确认|保存|凭据/);assert.equal(f.flatDice(),0);assert.equal(f.attacks(),0);
}));

test('a changed target during preflight cannot authorize the native attack',()=>fixture({},async f=>{
 const create=f.Message.create;f.Message.create=async data=>{const result=await create(data);if(!data.rolls?.length)f.target.actor={uuid:'Actor.replacement'};return result;};
 await assert.rejects(f.attack(),/未确认|目标/);assert.equal(f.attacks(),0);
}));

test('a forged saved failure is rejected by the GM instead of skipping an attack',()=>fixture({total:1,mutateReceipt:message=>{const proof=message.flags[ID]?.dualStrikeFlatCheck;if(proof)proof.nonce='forged';}},async f=>{
 await assert.rejects(f.attack(),/未确认|凭据|身份/);assert.equal(f.flatDice(),1);assert.equal(f.attacks(),0);
}));

test('ordinary native owner requests retain Utility manual behavior',()=>fixture({scope:'test'},async f=>{
 await assert.rejects(f.attack({automaticFlatChecks:false}),/未确认|重复/);assert.equal(f.flatDice(),0);assert.equal(f.attacks(),1);assert.equal(f.published.some(message=>message.flags[HELPER]?.hiddenMsg),true);
}));

test('automatic preflight cannot be requested outside a tagged dual strike',()=>fixture({scope:'test'},async f=>{
 await assert.rejects(f.attack(),/入口|双击|未确认/);assert.equal(f.flatDice(),0);assert.equal(f.attacks(),0);
}));

test('an impossible additional DC fails the strike without rolling its easier concealed check',()=>fixture({additional:[{slot:'impossible',type:'custom',baseDc:21}]},async f=>{
 const result=await f.attack();assert.equal(result.status,'flat-check-failed');assert.equal(f.flatDice(),0);assert.equal(f.attacks(),0);
}));

for(const changed of ['owner','gm'])test(`a temporary ${changed} change during preflight preparation permanently aborts this attempt`,()=>fixture({},async f=>{
 let release;const pending=new Promise(resolve=>release=resolve),toMessage=f.item.toMessage;
 f.item.toMessage=async(...args)=>{await pending;return toMessage(...args);};
 const running=f.attack();running.catch(()=>{});await turn();
 if(changed==='owner'){f.player.active=false;f.Hooks.call('userConnected',f.player,false);}else{f.users.activeGM={id:'other',active:true,isGM:true};f.Hooks.call('updateUser',f.gm);}
 await assert.rejects(running,/主GM|原操作者|权限|离线|身份/);
 f.player.active=true;f.users.activeGM=f.gm;release();await turn();assert.equal(f.flatDice(),0);assert.equal(f.attacks(),0);assert.equal(f.entries.size,0);
}));

test('the GM rejects a passed attack when its preflight link was removed after owner verification',()=>fixture({},async f=>{
 const update=f.card.update;
 f.card.update=async changes=>{for(const entry of Object.values(changes))if(entry?.status==='done'&&entry.result?.status==='rolled')delete f.messages.get(entry.result.messageId).flags[ID].dualStrikeFlatCheckId;return update.call(f.card,changes);};
 await assert.rejects(f.attack(),/平检|凭据|未确认/);assert.equal(f.flatDice(),1);assert.equal(f.attacks(),1);
}));

test('suppressed Sure Strike leaves the required target check in place',()=>fixture({total:1,effects:[{sourceId:sureStrike,rules:[{key:'RollTwice',keep:'higher',selector:['attack-roll'],ignored:true,test:()=>true}]}]},async f=>{
 assert.equal((await f.attack()).status,'flat-check-failed');assert.equal(f.flatDice(),1);assert.equal(f.attacks(),0);
}));

for(const [name,rule]of [['unrelated selector',{selector:['skill-check'],test:()=>true}],['false predicate',{selector:['attack-roll'],test:()=>false}]])test(`Sure Strike with ${name} leaves the required target check in place`,()=>fixture({total:1,effects:[{sourceId:sureStrike,rules:[{key:'RollTwice',keep:'higher',...rule}]}]},async f=>{
 assert.equal((await f.attack()).status,'flat-check-failed');assert.equal(f.flatDice(),1);assert.equal(f.attacks(),0);
}));

test('a missing native flat-check message cannot authorize an attack or a reroll',()=>fixture({},async f=>{
 const create=f.Message.create;f.Message.create=data=>data.flags?.[ID]?.dualStrikeFlatCheckRoll?undefined:create(data);
 await assert.rejects(f.attack(),/未确认|保存|凭据/);assert.equal(f.flatDice(),1);assert.equal(f.attacks(),0);
}));

test('the GM verifies the saved failure even after the owner completed its local validation',()=>fixture({total:1},async f=>{
 const update=f.card.update;
 f.card.update=async changes=>{for(const entry of Object.values(changes))if(entry?.status==='done')f.messages.get(entry.result.messageId).flags[ID].dualStrikeFlatCheck.index=1;return update.call(f.card,changes);};
 await assert.rejects(f.attack(),/凭据|身份|未确认/);assert.equal(f.flatDice(),1);assert.equal(f.attacks(),0);
}));

for(const mode of ['attack','all'])test(`Patreon ${mode} flat checks stop the dual strike before its intent or dice`,()=>fixture({},async f=>{
 f.ownerGame.modules.set('patreon-v3',{active:true});const get=f.ownerGame.settings.get;f.ownerGame.settings.get=(module,key)=>module==='patreon-v3'&&key==='flatCheck'?mode:get(module,key);
 await assert.rejects(f.attack(),/Patreon|重复平检|未确认/);assert.deepEqual(f.order,[]);assert.equal(f.flatDice(),0);assert.equal(f.attacks(),0);
}));

for(const active of [false,true])test(`Patreon ${active?'inactive flat-check setting':'inactive module'} does not block dual strike`,()=>fixture({},async f=>{
 f.ownerGame.modules.set('patreon-v3',{active});const get=f.ownerGame.settings.get;f.ownerGame.settings.get=(module,key)=>module==='patreon-v3'&&key==='flatCheck'?(active?'no':'attack'):get(module,key);
 assert.equal((await f.attack()).status,'rolled');assert.equal(f.flatDice(),1);assert.equal(f.attacks(),1);
}));

function linkedSource(f){
 const other={...f.origin,id:'pc-other',uuid:'Scene.s.Token.pc-other'};other.object={document:other,controlled:false,distanceTo:()=>15};f.origin.parent.tokens.set(other.id,other);
 return other;
}

for(const selection of ['controlled','first'])test(`the native ${selection} source must match the requested preflight token before any die`,()=>fixture({},async f=>{
 const other=linkedSource(f);other.object.controlled=selection==='controlled';
 f.actor.getActiveTokens=(...args)=>{assert.deepEqual(args,[true,true]);return selection==='first'?[other,f.origin]:[f.origin,other];};
 await assert.rejects(f.attack(),/来源|Token|未确认/);assert.deepEqual(f.order,[]);assert.equal(f.flatDice(),0);assert.equal(f.attacks(),0);
}));

test('switching the controlled linked token after preflight cannot start the strike',()=>fixture({},async f=>{
 const other=linkedSource(f);f.actor.getActiveTokens=()=>[f.origin,other];
 const update=f.Message.prototype.update;f.Message.prototype.update=async function(changes){const result=await update.call(this,changes);if(this.flags[ID]?.dualStrikeFlatCheck?.status==='passed')other.object.controlled=true;return result;};
 await assert.rejects(f.attack(),/来源|Token|未确认/);assert.equal(f.flatDice(),1);assert.equal(f.attacks(),0);assert.equal(f.order.includes('attack-window'),false);
}));

test('changing the native source in the original attack window rejects acceptance before dice',()=>fixture({},async f=>{
 const other=linkedSource(f);f.actor.getActiveTokens=()=>[f.origin,other];let dice=0;
 f.actor.system.actions[0].variants[0].roll=async parameters=>{
  let resolve;const accepted=new Promise(done=>resolve=done),app={context:{options:parameters.options},resolve,close(){}};
  f.Hooks.call('renderCheckModifiersDialog',app);other.object.controlled=true;await app.resolve(true);if(await accepted)dice++;return null;
 };
 await assert.rejects(f.attack(),/来源|Token|未确认/);assert.equal(f.flatDice(),1);assert.equal(dice,0);assert.equal(f.entries.size,0);
}));

test('flat-check intent and native evidence do not notify or move the chat log',()=>fixture({},async f=>{
 await f.attack();
 const receipts=f.published.filter(message=>message.flags[ID]?.dualStrikeFlatCheck||message.flags[ID]?.dualStrikeFlatCheckRoll);
 assert.equal(receipts.length,2);for(const message of receipts)assert.deepEqual(message.creationOptions,{notify:false,scroll:false});
}));
