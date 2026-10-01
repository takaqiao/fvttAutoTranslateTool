import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createNativeOwnerOperations,getNativeOwnerInvocation} from '../scripts/native-owner-operations.mjs';

const ID='pf2e-third-party-automation';
const set=(object,path,value)=>{const parts=path.split('.');let at=object;for(const key of parts.slice(0,-1))at=at[key]??={};at[parts.at(-1)]=value;};
function fixture({accepted=true,offline=false}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',isGM:false,active:!offline,settings:{showCheckDialogs:false,showDamageDialogs:false}},users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm});
 const messages=new Map(),docs=new Map(),hooks=new Map(),calls=[],ownerHandlers=new Map(),gmHandlers=new Map();let seq=0,current;
 const Hooks={on(event,fn){const id=++seq;hooks.set(id,{event,fn});return id;},off(_event,id){hooks.delete(id);},call(event,...args){for(const entry of [...hooks.values()])if(entry.event===event)entry.fn(...args);}};
 class DamageRoll{constructor(formula='1d6'){this.formula=formula;this._evaluated=false;this.total=4;this.options={};}toJSON(){return {class:'DamageRoll',formula:this.formula,evaluated:this._evaluated,total:this.total,options:this.options};}static fromJSON(json){const data=JSON.parse(json),roll=new this(data.formula);roll._evaluated=data.evaluated;return roll;}}
 class CheckRoll{constructor(){this._evaluated=true;this.total=19;this.options={degreeOfSuccess:2};this.dice=[{faces:20,total:12,results:[{result:12,active:true}]}];}toJSON(){return {class:'CheckRoll',formula:'1d20+7',total:this.total,evaluated:true,options:this.options};}}
 const actor={id:'pc',uuid:'Actor.pc',items:new Map(),system:{actions:[]},testUserPermission:user=>user===player},item={id:'ability',uuid:'Actor.pc.Item.ability',sourceId:'Compendium.test.abilities.Item.ability',actor},token={uuid:'Scene.s.Token.pc',actor,object:{}};
 actor.items.set(item.id,item);for(const document of [actor,item,token])docs.set(document.uuid,document);
 class Message{constructor(data){Object.assign(this,data);this.actor=data.speaker?.actor===actor.id?actor:null;}toObject(){return {...this};}static applyMode(data,mode){return {...data,blind:mode==='blind',whisper:mode==='self'?[player.id]:['gm']};}static async create(data){const message=new Message({...data,id:`check-${messages.size}`});messages.set(message.id,message);Hooks.call('createChatMessage',message);return message;}}
 const source={id:'activity',uuid:'ChatMessage.activity',author:player,speaker:{actor:actor.id},flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid}}},async update(changes){for(const [path,value]of Object.entries(changes))set(this,path,value);Hooks.call('updateChatMessage',this);}};messages.set(source.id,source);
 const gmGame={user:gm,users,messages},ownerGame={user:player,users,messages};
 const native=async(parameters,kind='skill-check',statistic)=>{
  calls.push({user:current.user.id,kind,parameters});
  let resolve;const decision=new Promise(done=>resolve=done),options=[...parameters.extraRollOptions??parameters.options,...statistic?[`check:statistic:${statistic}`]:[]];
  const app={context:{options:new Set(options),messageMode:parameters.messageMode},resolve};Hooks.call('renderCheckModifiersDialog',app);await app.resolve(accepted);
  if(!await decision)return null;
  const roll=new CheckRoll(),raw=new Message({author:player.id,speaker:{actor:actor.id},blind:parameters.messageMode==='blind',whisper:parameters.messageMode==='blind'?[gm.id]:[],rolls:[roll],flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid},context:{type:kind,options:[...options],dc:parameters.dc,outcome:'success',...parameters.target?{target:{actor:parameters.target.uuid,token:parameters.target.getActiveTokens()[0].uuid}}:{}}}}});
  await parameters.callback?.(roll,'success',raw);return roll;
 };
 actor.getStatistic=slug=>slug==='medicine'||slug==='fortitude'?{check:{roll:parameters=>native(parameters,slug==='fortitude'?'saving-throw':'skill-check',slug)}}:null;
 class CheckModifier{constructor(slug,{modifiers}){this.slug=slug;this.modifiers=modifiers;}}
 ownerGame.pf2e={CheckModifier,Check:{roll:async(_check,parameters,event,callback)=>{assert.equal(event,null);return native({...parameters,callback},parameters.type);}}};gmGame.pf2e={Check:{roll(){throw Error('GM must not roll');}}};
 const manualDamageRoll=async({game,roll,assertLive})=>{calls.push({user:game.user.id,kind:'formula-damage'});assertLive();if(!accepted)return null;roll._evaluated=true;return roll;},manualDamagePrivacy=(_roll,minimum)=>minimum?{messageMode:'blind',blind:true,whisper:[gm.id]}:{messageMode:'self',blind:false,whisper:[player.id]};
 const options={fromUuid:async uuid=>docs.get(uuid),scope:'typed',manualDamageRoll,manualDamagePrivacy,confirmFlatCheck:async()=>{calls.push({user:current.user.id,kind:'flat-confirm'});return accepted;}};
 const owner=createNativeOwnerOperations({game:ownerGame,...options}),root=createNativeOwnerOperations({game:gmGame,...options});
 const ownerSocket={register:(name,fn)=>ownerHandlers.set(name,fn),async executeAsUser(name,userId,payload){assert.equal(userId,gm.id);const previous=current;current=gmGame;try{return await gmHandlers.get(name).call({socketdata:{userId:player.id}},payload);}finally{current=previous;}}};
 const socket={register:(name,fn)=>gmHandlers.set(name,fn),async executeAsUser(name,userId,payload){assert.equal(userId,player.id);const previous=current;current=ownerGame;try{return await ownerHandlers.get(name).call({socketdata:{userId:gm.id}},payload);}finally{current=previous;}}};
 owner.register({Hooks,socket:ownerSocket});root.register({Hooks,socket});
 return {root,owner,gmGame,ownerGame,gm,player,actor,item,token,source,messages,docs,calls,hooks,Hooks,Message,DamageRoll,ownerHandlers,socket,native};
}
async function setup(options,run){const previous=globalThis.CONFIG;try{const f=fixture(options);globalThis.CONFIG={ChatMessage:{documentClass:f.Message},Dice:{rolls:[f.DamageRoll]}};await run(f);}finally{globalThis.CONFIG=previous;}}
const request=f=>({type:'check',statistic:'medicine',itemUuid:f.item.uuid,sourceUuid:f.item.sourceId,tokenUuid:f.token.uuid,dc:{value:31,visible:false},messageMode:'blind',options:['action:diagnosis']});
const context=f=>({actor:f.actor,message:f.source,user:f.player});

test('typed no-target skill runs only on original owner and keeps secret DC and roll out of public operation',()=>setup({},async f=>{
 const result=await f.root.run(context(f),request(f));
 assert.equal(result.status,'rolled');assert.equal(result.check,f.messages.get(result.messageId));assert.equal(result.check.author,'player');assert.equal(result.check.blind,true);
 assert.equal(f.calls.length,1);assert.equal(f.calls[0].user,'player');assert.equal(f.calls[0].parameters.skipDialog,false);assert.equal(f.calls[0].parameters.event,null);
 const operation=Object.values(f.source.flags[ID].nativeOwnerOperations)[0];assert.equal(operation.request.dc,undefined);assert.equal(operation.request.messageMode,undefined);assert.deepEqual(operation.result,{status:'rolled',messageId:result.messageId});assert.equal(f.hooks.size,0);
}));
test('typed saving throw retains its native type and bound target actor',()=>setup({},async f=>{
 const result=await f.root.run(context(f),{...request(f),statistic:'fortitude',checkKind:'saving-throw'});
 assert.equal(result.check.flags.pf2e.context.type,'saving-throw');assert.equal(f.calls[0].user,'player');
}));
test('closing owner skill window creates neither result nor committed payment',()=>setup({accepted:false},async f=>{
 let payments=0;const result=await f.root.run(context(f),request(f),null,async()=>payments++);
 assert.deepEqual(result,{status:'cancelled'});assert.equal(payments,0);assert.equal(f.messages.size,1);assert.equal(f.hooks.size,0);
}));
test('accepted owner check commits existing pre-roll payment once before publishing its real native card',()=>setup({},async f=>{
 let payments=0;const result=await f.root.run(context(f),request(f),null,async()=>payments++);
 assert.equal(payments,1);assert.equal(result.status,'rolled');assert.equal(f.calls.length,1);
}));
test('flat check confirmation and zero-modifier d20 both execute only on owner',()=>setup({},async f=>{
 for(const type of ['flat','d20']){const result=await f.root.run(context(f),{type,tokenUuid:f.token.uuid,options:['action:test'],...type==='flat'?{dc:{value:5}}:{}});assert.equal(result.status,'rolled');assert.equal(result.check.flags.pf2e.context.type,type==='flat'?'flat-check':'check');}
 assert.ok(f.calls.every(call=>call.user==='player'));assert.equal(f.calls.filter(call=>call.kind==='flat-confirm').length,1);
}));
test('formula damage stores only a private non-native proof and restores owner self audience for GM publication',()=>setup({},async f=>{
 const result=await f.root.run(context(f),{type:'formula-damage',formula:'1d6[electricity]',itemUuid:f.item.uuid,tokenUuid:f.token.uuid});
 assert.equal(result.status,'rolled');assert.ok(result.roll instanceof f.DamageRoll);assert.equal(result.roll._evaluated,true);assert.deepEqual(result.privacy,{messageMode:'gm',blind:false,whisper:['player']});assert.equal(f.calls[0].user,'player');
 const proof=f.messages.get(result.messageId);assert.equal(proof.blind,true);assert.deepEqual(proof.whisper,['gm']);assert.deepEqual(proof.rolls,[]);assert.equal(proof.flags.pf2e,undefined);
 const operation=Object.values(f.source.flags[ID].nativeOwnerOperations)[0];assert.equal(operation.request.formula,undefined);assert.deepEqual(operation.result,{status:'rolled',messageId:proof.id});assert.equal(f.hooks.size,0);
}));
test('cancelled formula window leaves no receipt or roll',()=>setup({accepted:false},async f=>{
 const result=await f.root.run(context(f),{type:'formula-damage',formula:'1d6'});assert.deepEqual(result,{status:'cancelled'});assert.equal(f.messages.size,1);
}));
test('typed operation fails closed if the original player is offline',()=>setup({offline:true},async f=>{
 await assert.rejects(f.root.run(context(f),request(f)),/离线|连接/);assert.equal(f.calls.length,0);assert.equal(f.messages.size,1);
}));
test('a player cannot dispatch the GM-authorized typed operation to itself',()=>setup({},async f=>{
 await assert.rejects(f.ownerHandlers.get('native-owner:typed').call({socketdata:{userId:'player'}},{messageId:f.source.id,nonce:'invented',privateRequest:request(f)}),/主GM/);assert.equal(f.calls.length,0);
}));
test('durable private result releases GM wait when owner reply is lost without another roll',()=>setup({},async f=>{
 const execute=f.socket.executeAsUser;f.socket.executeAsUser=async(...args)=>{await execute(...args);return new Promise(()=>{});};let timer;
 try{const result=await Promise.race([f.root.run(context(f),request(f)),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('saved owner check still waiting')),150);})]);assert.equal(result.check.author,'player');assert.equal(f.calls.length,1);assert.equal(f.hooks.size,0);}finally{clearTimeout(timer);}
}));
for(const field of ['dc','type','origin','degree'])test(`GM rejects a native card with changed private ${field} proof`,()=>setup({},async f=>{
 const create=f.Message.create;f.Message.create=async data=>{
  if(field==='dc')data.flags.pf2e.context.dc={value:99};
  if(field==='type')data.flags.pf2e.context.type='saving-throw';
  if(field==='origin')data.flags.pf2e.origin.uuid='Actor.pc.Item.other';
  if(field==='degree')data.flags.pf2e.context.outcome='criticalFailure';
  return create(data);
 };
 await assert.rejects(f.root.run(context(f),request(f)),/检定卡|来源|结果/);assert.equal(f.calls.length,1);
}));
test('invocation exists only during authenticated local native check and does not trust public flags',()=>setup({},async f=>{
 const get=f.actor.getStatistic;let observed;
 f.actor.getStatistic=slug=>{const statistic=get(slug),roll=statistic.check.roll;statistic.check.roll=async parameters=>{observed=getNativeOwnerInvocation(f.ownerGame,parameters.extraRollOptions);assert.equal(getNativeOwnerInvocation(f.gmGame,parameters.extraRollOptions),null);return roll(parameters);};return statistic;};
 await f.root.run(context(f),request(f));assert.equal(observed.user,f.player);assert.equal(observed.usageId,f.source.id);assert.equal(observed.actorUuid,f.actor.uuid);
 const nonce=Object.keys(f.source.flags[ID].nativeOwnerOperations)[0];assert.equal(getNativeOwnerInvocation(f.ownerGame,[`${ID}:native-operation:${nonce}`]),null);
}));
test('a source deletion closes pending owner check without dice or payment and releases local invocation',()=>setup({},async f=>{
 let opened,payments=0,dice=0;const entered=new Promise(resolve=>opened=resolve);
 f.actor.getStatistic=()=>({check:{roll:async parameters=>{let resolve;const accepted=new Promise(done=>resolve=done),app={context:{options:parameters.extraRollOptions},resolve,close(){}};f.Hooks.call('renderCheckModifiersDialog',app);opened();if(await accepted)dice++;return null;}}});
 const pending=f.root.run(context(f),request(f),null,async()=>payments++);await entered;f.actor.items.delete(f.item.id);f.Hooks.call('deleteItem',f.item);let timer;
 try{await assert.rejects(Promise.race([pending,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('window remained open after source deletion')),100);})]),/来源|尚未确认/);await new Promise(resolve=>setImmediate(resolve));assert.equal(payments,0);assert.equal(dice,0);assert.equal(f.hooks.size,0);}finally{clearTimeout(timer);f.Hooks.call('userConnected');await pending.catch(()=>{});}
}));
test('bound target facade agrees with actual PF StatisticCheck token lookup and preserves private actor receivers',()=>setup({},async f=>{
 const path=process.env.PF2E_NATIVE_BUNDLE;assert.ok(path,'actual PF2e source required');const source=fs.readFileSync(path,'utf8'),start=source.indexOf('l = s ? a : e.target?.getActiveTokens'),end=source.indexOf(', u = ',start);assert.ok(start>=0&&end>start);
 const lookup=new Function('game','e','s','a',`return ${source.slice(start+4,end)};`);
 class Target{#id='target';get uuid(){return `Actor.${this.#id}`;}isOfType(...types){assert.equal(this.#id,'target');return types.includes('creature');}}
 const target={uuid:'Scene.s.Token.target',actor:new Target(),object:{}};f.docs.set(target.uuid,target);
 await f.root.run(context(f),{...request(f),targetUuid:target.uuid});
 const facade=f.calls[0].parameters.target;assert.equal(facade.uuid,target.actor.uuid);assert.equal(lookup({user:{targets:[{actor:{isOfType:()=>true},document:{uuid:'wrong'}}]}},{target:facade},false,null),target);
}));
test('no-target numeric DC remains non-opposed in actual StatisticCheck even with another owner target selected',()=>setup({},async f=>{
 const path=process.env.PF2E_NATIVE_BUNDLE;assert.ok(path,'actual PF2e source required');const source=fs.readFileSync(path,'utf8'),start=source.indexOf('u = !s && !!l && ('),end=source.indexOf(';\n',start);assert.ok(start>=0&&end>start);
 const opposed=new Function('e','l','s','c',`return ${source.slice(start+4,end)};`);
 await f.root.run(context(f),request(f));const parameters=f.calls[0].parameters;
 assert.equal(opposed.call({type:'skill-check',domains:['skill-check']},parameters,{actor:{}},false,null),false);
 assert.equal('statistic' in parameters.dc,false);assert.equal('slug' in parameters.dc,false);
}));
test('numeric-DC native skill may omit target context while authenticated proof retains exact selected token',()=>setup({},async f=>{
 const target={uuid:'Scene.s.Token.target',actor:{uuid:'Actor.target'},object:{}};f.docs.set(target.uuid,target);
 const create=f.Message.create;f.Message.create=async data=>{delete data.flags.pf2e.context.target;return create(data);};
 const result=await f.root.run(context(f),{...request(f),targetUuid:target.uuid});
 assert.equal(result.check.flags.pf2e.context.target,undefined);assert.equal(result.check.flags[ID].nativeOwnerOperation.targetUuid,target.uuid);assert.equal(result.check.flags[ID].nativeOwnerOperation.targetActorUuid,target.actor.uuid);
}));
test('a changed native target context cannot be excused by numeric-DC metadata binding',()=>setup({},async f=>{
 const target={uuid:'Scene.s.Token.target',actor:{uuid:'Actor.target'},object:{}};f.docs.set(target.uuid,target);
 const create=f.Message.create;f.Message.create=async data=>{data.flags.pf2e.context.target={actor:'Actor.other',token:'Scene.s.Token.other'};return create(data);};
 await assert.rejects(f.root.run(context(f),{...request(f),targetUuid:target.uuid}),/检定卡|目标/);
}));
