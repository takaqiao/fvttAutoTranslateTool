import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/native-owner-operations.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
const ID='pf2e-third-party-automation';
const set=(object,path,value)=>{const bits=path.split('.');let at=object;for(const bit of bits.slice(0,-1))at=at[bit]??={};at[bits.at(-1)]=value;};
function fixture({cancel=false,offline=false}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:!offline,settings:{showCheckDialogs:false,showDamageDialogs:false}},users=new Map([[gm.id,gm],[player.id,player]]);users.activeGM=gm;
 const messages=new Map(),calls=[],hooks={on(){return 1},off(){}};
 const actor={uuid:'Actor.pc',id:'pc',testUserPermission:u=>u.id==='player',items:new Map(),system:{actions:[]}},target={uuid:'Scene.s.Token.t',actor:{uuid:'Actor.t'},object:{}};
 const item={id:'weapon',uuid:'Actor.pc.Item.weapon',type:'weapon',actor};actor.items.set(item.id,item);
 let current;
 class Message{constructor(data){Object.assign(this,data)}toObject(){return {...this}}static async create(data){const message=new Message({...data,id:'check'});messages.set(message.id,message);return message;}}
 actor.system.actions=[{type:'strike',item,variants:[{async roll(options){calls.push({user:current.user.id,options});if(cancel)return null;await options.callback({},'success',new Message({speaker:{actor:'pc'},flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid},context:{type:'attack-roll',outcome:'success',target:{token:target.uuid}}}},rolls:[{total:20}]}));return {};}}]}];
 const card={id:'activity',author:player,flags:{pf2e:{origin:{actor:actor.uuid}}},async update(changes){for(const[path,value]of Object.entries(changes))set(this,path,value);}};messages.set(card.id,card);
 const gmGame={user:gm,users,messages},ownerGame={user:player,users,messages},docs=new Map([[actor.uuid,actor],[target.uuid,target]]),handlers=new Map();
 const create=api.createNativeOwnerOperations;assert.equal(typeof create,'function');
 const owner=create({game:ownerGame,fromUuid:async uuid=>docs.get(uuid),scope:'test'}),root=create({game:gmGame,fromUuid:async uuid=>docs.get(uuid),scope:'test'});
 owner.register({Hooks:hooks,socket:{register(name,handler){handlers.set(name,handler)}}});
 const socket={register(){},async executeAsUser(name,userId,payload){assert.equal(userId,'player');current=ownerGame;return handlers.get(name).call({socketdata:{userId:'gm'}},payload);}};
 root.register({Hooks:hooks,socket});
 return {root,owner,gmGame,ownerGame,actor,target,card,item,calls,socket,handlers,Message,gm,player};
}
async function setup(options,fn){const old=globalThis.CONFIG;try{const f=fixture(options);globalThis.CONFIG={ChatMessage:{documentClass:f.Message}};await fn(f);}finally{globalThis.CONFIG=old;}}
test('native attack runs on original owner using that user dialog preferences and canonical check',()=>setup({},async f=>{
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[],flags:{test:true}});
 assert.equal(result.status,'rolled');assert.equal(result.messageId,'check');assert.equal(f.calls.length,1);assert.equal(f.calls[0].user,'player');assert.equal(f.calls[0].options.event.shiftKey,false);
 assert.equal(f.gmGame.messages.get('check').author,'player');
}));
test('owner cancellation is explicit and does not fabricate an attack result',()=>setup({cancel:true},async f=>{
 const result=await f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]});
 assert.equal(result.status,'cancelled');assert.equal(f.gmGame.messages.has('check'),false);
}));
test('disconnected original owner fails before native interaction',()=>setup({offline:true},async f=>{
 await assert.rejects(f.root.run({actor:f.actor,message:f.card,user:f.player},{type:'attack',weaponId:'weapon',map:0,targetUuid:f.target.uuid,options:[]}),/连接|离线/);assert.equal(f.calls.length,0);
}));
test('untrusted requester cannot initiate an owner operation',()=>setup({},async f=>{
 const handler=f.handlers.get('native-owner:test');await assert.rejects(handler.call({socketdata:{userId:'player'}},{messageId:'activity',nonce:'fake'}),/主GM/);assert.equal(f.calls.length,0);
}));
for(const accepted of [false,true])test(`native dialog ${accepted?'acceptance':'cancellation'} commits resources only before an accepted roll`,async()=>{
 assert.equal(typeof api.beforeNativeRoll,'function');let hook,commits=0,resolved;
 const Hooks={on(event,callback){assert.equal(event,'renderCheckModifiersDialog');hook=callback;return 1;},off(){hook=null;}};
 await api.beforeNativeRoll({Hooks,marker:'operation',showDialog:true,commit:async()=>{commits++;},native:async()=>{
  const app={context:{options:new Set(['operation'])},resolve:value=>{resolved=value;}};hook(app);await app.resolve(accepted);
 }});
 assert.equal(commits,accepted?1:0);assert.equal(resolved,accepted);assert.equal(hook,null);
});
test('skipped native dialog commits once before any die is rolled',async()=>{
 assert.equal(typeof api.beforeNativeRoll,'function');const order=[];
 await api.beforeNativeRoll({showDialog:false,commit:async()=>order.push('payment'),native:async()=>order.push('roll')});assert.deepEqual(order,['payment','roll']);
});
