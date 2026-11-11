import test from 'node:test';
import assert from 'node:assert/strict';
import {createPilgrimRewards} from '../scripts/sog-pilgrim-rewards.mjs';

const M='pf2e-third-party-automation',IA='pf2e-item-activations',HAUNT='wonton-ghost-scarf:haunt-context';
const clone=value=>structuredClone(value);
function set(document,changes){
 for(const [path,value]of Object.entries(changes)){
  const parts=path.split('.');let at=document;
  for(const key of parts.slice(0,-1))at=at[key]??={};
  at[parts.at(-1)]=clone(value);
 }
 return document;
}
class Element {
 constructor(tag){Object.assign(this,{tagName:tag,children:[],dataset:{},listeners:new Map(),className:'',textContent:'',disabled:false,parent:null,root:false});}
 get isConnected(){return this.root||!!this.parent?.isConnected;}
 append(...nodes){for(const node of nodes){node.parent=this;this.children.push(node);}}
 replaceChildren(...nodes){for(const node of this.children)node.parent=null;this.children=[];this.append(...nodes);}
 addEventListener(type,listener){this.listeners.set(type,listener);}
 querySelector(selector){
  for(const child of this.children){if(child.className.split(' ').includes(selector.slice(1)))return child;const found=child.querySelector(selector);if(found)return found;}
  return null;
 }
}
const html=content=>{const root=new Element('div');root.root=true;const body=new Element('div');body.className=content;root.append(body);return root;};
function buttons(root){
 const found=[];
 const walk=node=>{for(const child of node.children){if(child.tagName==='button')found.push({label:child.textContent,disabled:child.disabled,reason:node.dataset.tooltip,click:()=>child.listeners.get('click')({preventDefault(){},stopPropagation(){}})});walk(child);}};
 walk(root);return found;
}
function globals(t,values){
 for(const [key,value]of Object.entries(values)){
  const descriptor=Object.getOwnPropertyDescriptor(globalThis,key);
  Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
  t.after(()=>{if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];});
 }
}

function world(t) {
 let seq=0;const next=prefix=>`${prefix}${String(++seq).padStart(8,'0')}`;
 const gm={id:'gm',name:'GM',isGM:true,active:true},player={id:'player',name:'玩家',isGM:false,active:true};
 const users=new Map([[gm.id,gm],[player.id,player]]);users.activeGM=gm;
 const scene={id:'scene',uuid:'Scene.scene',grid:{size:100,distance:5,isSquare:true,measurePath:([a,b])=>({distance:Math.hypot(b.x-a.x,b.y-a.y)/20})},tokens:new Map(),templates:new Map(),regions:new Map(),tiles:new Map()};
 const elsewhere={id:'elsewhere',uuid:'Scene.elsewhere',tokens:new Map()};
 const shared={system:{id:'pf2e'},world:{id:'sog'},users,actors:new Map(),scenes:new Map([[scene.id,scene]]),messages:new Map(),time:{worldTime:10000},combat:null,settings:{get:()=>'publicroll'},modules:new Map(),items:new Map()};
 const documents=new Map(),sent=[],warnings=[],errors=[],dialogs=[],writes=[];
 const collections={MeasuredTemplate:'templates',Region:'regions',Tile:'tiles'};
 const sceneDocument=(type,row)=>{
  const document={...clone(row),id:next(type),parent:scene};document.uuid=`${scene.uuid}.${type}.${document.id}`;
  document.delete=async()=>{scene[collections[type]].delete(document.id);documents.delete(document.uuid);};
  scene[collections[type]].set(document.id,document);documents.set(document.uuid,document);return document;
 };
 scene.createEmbeddedDocuments=async(type,rows)=>rows.map(row=>sceneDocument(type,row));
 scene.deleteEmbeddedDocuments=async(type,ids)=>{for(const id of ids)await scene[collections[type]].get(id)?.delete();};
 const actor={id:'hero',uuid:'Actor.hero',name:'罗格斯',type:'character',isOwner:true,hasPlayerOwner:true,canSee:true,isDead:false,items:new Map(),rollOptions:{all:{}},traits:new Set(),
  testUserPermission:()=>true,hasCondition:()=>false,getReach:()=>5,isOfType(...types){return types.includes(this.type);},
  async toggleRollOption(domain,option,itemId,value){writes.push({toggle:[domain,option,itemId,value]});this.rollOptions[domain][option]=value;}};
 const embed=data=>{
  const document={...clone(data),actor,parent:actor};document.uuid=`${actor.uuid}.Item.${document.id}`;
  document.update=async(changes,options={})=>{writes.push({id:document.id,changes:clone(changes),options});set(document,changes);return document;};
  actor.items.set(document.id,document);documents.set(document.uuid,document);return document;
 };
 actor.createEmbeddedDocuments=async(_type,rows)=>rows.map(row=>embed({...row,id:next('item'),isExpired:false}));
 actor.deleteEmbeddedDocuments=async(_type,ids)=>{for(const id of ids){documents.delete(actor.items.get(id)?.uuid);actor.items.delete(id);}};
 shared.actors.set(actor.id,actor);documents.set(actor.uuid,actor);
 const token=(id,owner,x)=>{const document={id,uuid:`${scene.uuid}.Token.${id}`,documentName:'Token',parent:scene,actor:owner,x,y:0,width:1,height:1,elevation:0,hidden:false};document.object={document,center:{x:x+50,y:50},checkCollision:()=>false,isVisible:true};scene.tokens.set(id,document);documents.set(document.uuid,document);return document;};
 const hero=token('heroToken',actor,0);
 const foe={id:'foe',uuid:'Actor.foe',name:'怨灵',type:'npc',isDead:false,hasPlayerOwner:false,traits:new Set(),isOfType(...types){return types.includes(this.type);}};
 const foeToken=token('foeToken',foe,300);
 const slugs={fan:'spirit-fan',branch:'branch-of-the-great-sugi',scarf:'ghost-scarf',hairpin:'hairpin-of-blooming-flowers'},names={fan:'灵扇',branch:'大杉枝',scarf:'幽魂围巾',hairpin:'繁花发簪'};
 const reward=key=>{
  const weapon=['fan','branch'].includes(key);
  return embed({id:key,name:names[key],type:weapon?'weapon':'equipment',img:`${key}.webp`,system:{equipped:weapon?{carryType:'held',handsHeld:1}:{carryType:'worn',invested:true},traits:{value:[]},runes:{property:[]}},
   flags:{world:{sogWontonNativeAutomation:{key:slugs[key]},sogB2Ch2Resources:{originalSlug:slugs[key],fileSha256:'a'.repeat(64)}}}});
 };
 const activation=(source,{id=`${source.id}Use`,actions=2,actionType='action',frequency={max:1,per:'day',value:1}}={})=>embed({id,name:`启动项: ${source.name}`,type:'action',system:{actions:{value:actions},actionType:{value:actionType},frequency:clone(frequency),traits:{value:['activation']}},flags:{[IA]:{enabled:true,grantedBy:{_id:source.id}}}});
 const weapon=(id,name,carryType='held',runes=[])=>embed({id,name,type:'weapon',system:{equipped:{carryType,handsHeld:carryType==='held'?1:0},runes:{property:runes}}});
 const effect=(name,duration,extra={})=>({type:'effect',toObject:()=>clone({type:'effect',name,system:{duration,start:{value:0,initiative:null},rules:[],badge:null,context:null,...extra},flags:{}})});
 const choice={rules:[{key:'ChoiceSet',flag:'weapon'}]};
 for(const [id,template]of Object.entries({e3f4b1e0c69e0cba:effect('金叶',{value:-1,unit:'unlimited'}),'53ad48c5ad472dcf':effect('光芒',{value:1,unit:'minutes'}),'1ade2944acadc195':effect('杉树形态',{value:1,unit:'hours'}),ffebdbe91ceb569e:effect('幽冥',{value:5,unit:'minutes'},choice),'6d00b8dddd4d6a71':effect('星魂',{value:5,unit:'minutes'},choice),'2d43538383b1d58b':effect('护佑',{value:5,unit:'minutes'},choice),'9570410bfd571f7d':effect('花瓣风暴',{value:1,unit:'minutes'})}))documents.set(`Item.${id}`,template);
 const fromUuid=async uuid=>documents.get(uuid);
 const message=data=>{const saved={id:next('msg'),rolls:[],...data};shared.messages.set(saved.id,saved);return saved;};
 class DamageRoll {
  constructor(formula){this.formula=formula;}
  async evaluate(){return this;}
  async toMessage(data,options){return message({...data,options,formula:this.formula,rolls:[this],isRoll:true,isDamageRoll:true});}
 }
 globals(t,{
  ChatMessage:{getSpeaker:({actor:speaker})=>({actor:speaker?.id,alias:speaker?.name}),getWhisperRecipients:()=>[gm],create:async data=>message(clone(data))},
  CONFIG:{Dice:{rolls:[DamageRoll]},Item:{documentClass:class {constructor(data){this.data=data;}get remainingDuration(){const remaining=this.data.system.start.value+60-shared.time.worldTime;return {remaining,expired:remaining<=0};}}}},
  foundry:{applications:{ux:{TextEditor:{enrichHTML:async text=>text}},api:{DialogV2:{wait:async options=>{dialogs.push(options);return options.buttons[0].callback();}}}},utils:{randomID:()=>next('use')}},
  ui:{notifications:{warn:text=>warnings.push(text),error:text=>warnings.push(text)}},
  document:{createElement:tag=>new Element(tag)},
 });
 const bus=new Map();
 function client(userId,{canvas=null}={}) {
  const game=Object.create(shared);game.user=users.get(userId);
  const hooks=new Map(),Hooks={on:(name,handler)=>{if(!hooks.has(name))hooks.set(name,[]);hooks.get(name).push(handler);}};
  const socket={register:(name,handler)=>{if(!bus.has(userId))bus.set(userId,new Map());bus.get(userId).set(name,handler);},
   executeAsUser:async(name,target,payload)=>{sent.push({name,from:userId,to:target,payload:clone(payload)});return bus.get(target).get(name).call({socketdata:{userId}},clone(payload));}};
  return createPilgrimRewards({game,fromUuid,canvas,onError:error=>errors.push(error)}).then(api=>{
   api.register({Hooks,socket});
   return {api,game,canvas,emit:async(name,...args)=>{for(const handler of hooks.get(name)??[])await handler(...args);}};
  });
 }
 const view=(on=scene)=>({ready:true,scene:on,tokens:{controlled:[]}});
 // Fan leaves are stored on the fan with their native starts, as the GM writes them.
 const light=(fan,count,start=shared.time.worldTime)=>set(fan,{[`flags.${M}.sogPilgrim.fan`]:{leaves:Array.from({length:count},(_,index)=>({messageId:`hit${index}`,start:{value:start,initiative:null}})),hitIds:Array.from({length:count},(_,index)=>`hit${index}`)}});
 return {shared,scene,elsewhere,actor,hero,foe,foeToken,gm,player,documents,sent,warnings,errors,dialogs,writes,reward,activation,weapon,client,view,light,message,
  state:item=>item.flags?.[M]?.sogPilgrim??{},
  card:(action,{author='player',content='<div class="pf2e chat-card action-card" data-item-id="x"></div>'}={})=>({id:next('msg'),author:users.get(author),content,item:action,rolls:[],flags:{pf2e:{origin:{uuid:action.uuid,actor:actor.uuid,type:'action'}}}})};
}

test('using an activation action starts its TPA activation once, from the client that posted the card',async t=>{
 const w=world(t),scarf=w.reward('scarf'),action=w.activation(scarf,{actions:1});w.weapon('sword','长剑');
 const gm=await w.client('gm'),player=await w.client('player',{canvas:w.view()});
 const card=w.card(action);
 await gm.emit('createChatMessage',card,{},'player');assert.equal(w.sent.length,0);
 await player.emit('createChatMessage',card,{},'player');
 assert.deepEqual(w.sent.map(({name,payload})=>[name,payload.operation,payload.itemUuid,payload.weaponId]),[['pilgrim-use','scarf',scarf.uuid,'sword']]);
 assert.equal(w.state(scarf).spent,true);assert.equal(w.state(scarf).use.status,'done');
 const repost=w.card(action,{content:'<span data-visibility="all">@Check[perception]</span>'});
 await player.emit('createChatMessage',repost,{},'player');
 await player.emit('createChatMessage',w.card(w.activation(w.reward('fan'))),{},'player');
 assert.equal(w.sent.length,1);assert.deepEqual(w.warnings,['请先选中一个目标。']);
});

test('an activation card carries the bound TPA button with its state and a retry after a failed start',async t=>{
 const w=world(t),scarf=w.reward('scarf'),action=w.activation(scarf,{actions:1});w.weapon('sword','长剑');
 set(scarf,{[`flags.${M}.sogPilgrim.spent`]:true});
 const gm=await w.client('gm'),player=await w.client('player',{canvas:w.view()});
 await player.emit('createChatMessage',w.card(action),{},'player');
 assert.deepEqual(w.warnings,['今日已经启动；完成每日休息后恢复。']);assert.equal(w.sent.length,0);
 const root=html('message-content');await player.emit('renderChatMessageHTML',{item:action,rolls:[]},root);
 assert.deepEqual(buttons(root).map(({label,disabled,reason})=>({label,disabled,reason})),[{label:'启动围巾 ◆ · 今日已用',disabled:true,reason:'今日已经启动；完成每日休息后恢复。'}]);
 set(scarf,{[`flags.${M}.sogPilgrim.spent`]:false});await player.emit('updateItem',scarf,{flags:{}},{},'gm');
 const [retry]=buttons(root);assert.deepEqual([retry.label,retry.disabled],['启动围巾 ◆',false]);
 await retry.click();
 assert.equal(w.sent.length,1);assert.equal(w.state(scarf).spent,true);
 assert.deepEqual(buttons(root).map(button=>button.label),['启动围巾 ◆ · 今日已用']);
 await gm.emit('renderChatMessageHTML',{item:null,rolls:[]},html('message-content'));
});

test('activation counts mirror the TPA state and undo a native use that never started',async t=>{
 const w=world(t),fan=w.reward('fan'),scarf=w.reward('scarf'),branch=w.reward('branch');
 const release=w.activation(fan),wrap=w.activation(scarf,{actions:1}),tree=w.activation(branch,{actions:3}),restore=w.activation(branch,{id:'restoreUse',actions:null,actionType:'free',frequency:null});
 const book=w.actor.items.set('book',{id:'book',type:'equipment',actor:w.actor,flags:{}})&&w.activation({id:'book',name:'祷告之书'},{id:'bookUse'});
 const gm=await w.client('gm');await gm.api.reconcile(w.actor);
 assert.deepEqual(release.system.frequency,{max:1,per:'PT1H',value:1});
 assert.deepEqual(wrap.system.frequency,{max:1,per:'day',value:1});
 assert.deepEqual(tree.system.frequency,{max:1,per:'day',value:1});
 assert.equal(restore.system.frequency,null);assert.deepEqual(book.system.frequency,{max:1,per:'day',value:1});
 assert.ok(w.writes.filter(write=>write.changes?.['system.frequency']).every(write=>write.options.tpaPilgrimSync===true));
 await wrap.update({'system.frequency.value':0});await gm.emit('updateItem',wrap,{system:{frequency:{value:0}}},{},'player');
 assert.equal(wrap.system.frequency.value,1);
 set(fan,{[`flags.${M}.sogPilgrim.lastUse.release`]:w.shared.time.worldTime});await gm.emit('updateItem',fan,{flags:{}},{},'gm');
 assert.equal(release.system.frequency.value,0);
 w.shared.time.worldTime+=3600;await gm.emit('updateWorldTime',w.shared.time.worldTime,3600);
 assert.equal(release.system.frequency.value,1);
 set(scarf,{[`flags.${M}.sogPilgrim.spent`]:true});await gm.emit('updateItem',scarf,{flags:{}},{},'gm');
 assert.equal(wrap.system.frequency.value,0);
 book.system.frequency.value=0;await gm.emit('updateItem',book,{system:{frequency:{value:0}}},{},'player');assert.equal(book.system.frequency.value,0);
});

test('placing the hairpin area from its description starts the storm there and removes the native area',async t=>{
 const w=world(t),hairpin=w.reward('hairpin'),action=w.activation(hairpin);
 const gm=await w.client('gm',{canvas:w.view(w.elsewhere)}),player=await w.client('player',{canvas:w.view()});
 const place=(origin,{user='player',areaShape='burst',radius=300,x=600}={})=>{
  const region=w.scene.createEmbeddedDocuments('Region',[{name:'花瓣风暴',shapes:[{type:'circle',x,y:50,radius}],flags:{pf2e:{areaShape,origin:{uuid:origin.uuid,actor:w.actor.uuid}}}}]).then(([document])=>document);
  return region.then(async document=>{await player.emit('createRegion',document,{},user);return document;});
 };
 const other=await place(hairpin,{user:'gm'}),emanation=await place(hairpin,{areaShape:'emanation'}),wide=await place(hairpin,{radius:600});
 assert.equal(w.sent.length,0);
 const native=await place(hairpin);
 assert.deepEqual(w.sent.map(({payload})=>[payload.operation,payload.position]),[['storm',{x:600,y:50}]]);
 assert.equal(w.scene.regions.has(native.id),false);
 for(const kept of [other,emanation,wide])assert.equal(w.scene.regions.has(kept.id),true);
 const storm=[...w.scene.regions.values()].find(region=>region.flags?.[M]?.sogPilgrim?.kind==='storm');
 assert.deepEqual(storm.shapes[0],{type:'circle',x:600,y:50,radius:300,gridBased:true});
 await player.emit('createRegion',storm,{},'player');assert.equal(w.sent.length,1);
 const again=await place(action);
 assert.equal(w.sent.length,1);assert.equal(w.scene.regions.has(again.id),true);
 assert.deepEqual(w.warnings,['今日已经启动；完成每日休息后恢复。']);
 const notices=[...w.shared.messages.values()].filter(message=>message.flags?.[M]?.sogPilgrim?.announce);
 assert.equal(notices.length,1);assert.equal(notices[0].whisper,undefined);assert.match(notices[0].content,/罗格斯/);assert.match(notices[0].content,/花瓣风暴/);
 assert.equal([...w.shared.messages.values()].some(message=>message.isDamageRoll),false);
});

test('release validates range and lit leaves on the token scene while the GM views another scene',async t=>{
 const w=world(t),fan=w.reward('fan');w.light(fan,3);
 await w.client('gm',{canvas:w.view(w.elsewhere)});const player=await w.client('player',{canvas:w.view()});
 player.game.user={...w.player,targets:new Set([w.foeToken])};
 const result=await player.api.request(fan,'release');
 assert.equal(result.status,'done');assert.equal(result.shown,false);
 const card=[...w.shared.messages.values()].find(message=>message.isDamageRoll);
 assert.equal(card.formula,'(3d8+8)[healing]');assert.equal(card.whisper,undefined);assert.equal(card.options.messageMode,'public');
 assert.deepEqual(w.state(fan).fan.leaves,[]);
 w.shared.time.worldTime+=3600;w.light(fan,3,w.shared.time.worldTime-61);
 await assert.rejects(player.api.request(fan,'release'),/金叶/);
 w.light(fan,3);w.foeToken.x=800;
 await assert.rejects(player.api.request(fan,'release'),/范围/);
 w.foeToken.x=300;w.hero.object.checkCollision=()=>true;const before=w.sent.length;
 await assert.rejects(player.api.request(fan,'release'),/看见/);assert.equal(w.sent.length,before);
});

test('a failure after the pending mark fails the use unless its result was already published',async t=>{
 const w=world(t),scarf=w.reward('scarf');w.weapon('sword','长剑');
 const gm=await w.client('gm'),create=w.actor.createEmbeddedDocuments;
 w.actor.createEmbeddedDocuments=async()=>{throw Error('database unavailable');};
 const payload={itemUuid:scarf.uuid,operation:'scarf',nonce:'firstuse0001',sourceUuid:w.hero.uuid,weaponId:'sword'};
 await assert.rejects(gm.api.execute(payload,w.player),/database unavailable/);
 assert.equal(w.state(scarf).use.status,'failed');assert.notEqual(w.state(scarf).spent,true);
 w.actor.createEmbeddedDocuments=create;
 const update=scarf.update;let fail=true;
 scarf.update=async(changes,options)=>{if(fail&&changes[`flags.${M}.sogPilgrim.use`]?.status==='done'){fail=false;throw Error('response lost');}return update(changes,options);};
 const result=await gm.api.execute({...payload,nonce:'seconduse001'},w.player);
 assert.equal(result.status,'done');assert.equal(w.state(scarf).use.status,'done');assert.equal(w.state(scarf).use.reconciled,true);assert.equal(w.state(scarf).spent,true);
 assert.equal(w.errors.at(-1).message,'response lost');
});

test('the GM can clear a stuck pending activation from the reward sheet',async t=>{
 const w=world(t),hairpin=w.reward('hairpin');
 set(hairpin,{[`flags.${M}.sogPilgrim.use`]:{nonce:'stuck0000001',operation:'storm',status:'pending',startedAt:1}});
 const gm=await w.client('gm'),player=await w.client('player');
 const playerSheet=html('window-content');await player.emit('renderItemSheetPF2e',{item:hairpin},playerSheet);
 assert.deepEqual(buttons(playerSheet).map(button=>[button.label,button.disabled]),[['花瓣风暴 ◆◆ · 待核对',true]]);
 const gmSheet=html('window-content');await gm.emit('renderItemSheetPF2e',{item:hairpin},gmSheet);
 const clear=buttons(gmSheet).find(button=>button.label==='清除待核对');assert.ok(clear);
 await clear.click();
 assert.equal(w.state(hairpin).use.status,'cleared');
 assert.deepEqual(buttons(gmSheet).map(button=>[button.label,button.disabled]),[['花瓣风暴 ◆◆',false]]);
 await player.emit('updateItem',hairpin,{flags:{}},{},'gm');
 assert.deepEqual(buttons(playerSheet).map(button=>[button.label,button.disabled]),[['花瓣风暴 ◆◆',false]]);
});

test('rendered buttons follow leaf and cooldown changes as time passes',async t=>{
 const w=world(t),fan=w.reward('fan');w.light(fan,2);
 const player=await w.client('player');
 const sheet=html('window-content');await player.emit('renderItemSheetV2',{document:fan},sheet);
 assert.deepEqual(buttons(sheet).map(({label,reason})=>[label,reason]),[['灵魂连携 ◆◆ · 金叶 2/3','需要三片亮起的金叶。']]);
 w.light(fan,3);await player.emit('updateItem',fan,{flags:{}},{},'gm');
 assert.deepEqual(buttons(sheet).map(({label,disabled})=>[label,disabled]),[['灵魂连携 ◆◆ · 金叶 3/3',false]]);
 set(fan,{[`flags.${M}.sogPilgrim.lastUse.release`]:w.shared.time.worldTime});w.light(fan,0);w.shared.time.worldTime+=600;
 await player.emit('updateWorldTime',w.shared.time.worldTime,600);
 assert.deepEqual(buttons(sheet).map(button=>button.label),['灵魂连携 ◆◆ · 冷却 50 分钟']);
 sheet.children[0].parent=null;sheet.root=false;
 w.shared.time.worldTime+=3000;await player.emit('updateWorldTime',w.shared.time.worldTime,3000);
 assert.deepEqual(buttons(sheet).map(button=>button.label),['灵魂连携 ◆◆ · 冷却 50 分钟']);
});

test('the scarf asks which weapon to wrap and posts a public notice of the result',async t=>{
 const w=world(t),scarf=w.reward('scarf');w.weapon('sword','长剑','held',['ghostTouch']);w.weapon('bow','长弓','worn');
 const player=await w.client('player',{canvas:w.view()});await w.client('gm');
 await player.api.request(scarf,'scarf');
 assert.equal(w.dialogs.length,1);assert.equal(w.dialogs[0].window.title,'选择围巾缠绕的武器');assert.match(w.dialogs[0].content,/选择围巾缠绕的武器/);
 assert.deepEqual(w.dialogs[0].buttons.map(button=>button.label),['长剑（手持） · 星魂','长弓 · 幽冥']);
 const [notice]=[...w.shared.messages.values()].filter(message=>message.flags?.[M]?.sogPilgrim?.announce);
 assert.equal(notice.whisper,undefined);assert.match(notice.content,/罗格斯/);assert.match(notice.content,/长剑/);assert.match(notice.content,/星魂/);assert.match(notice.content,/5 分钟/);
 assert.equal(notice.rolls.length,0);
});

test('a branch shift posts a public notice naming its new form',async t=>{
 const w=world(t),branch=w.reward('branch'),uuid='Compendium.pf2e.equipment-srd.Item.longsword';
 Object.assign(branch.system,{baseItem:'whip',category:'martial',group:'flail',damage:{dice:1,die:'d4',damageType:'slashing'},traits:{value:['magical']},usage:{value:'held-in-one-hand'}});
 branch.toObject=()=>({system:clone(branch.system)});
 w.documents.set(uuid,{uuid,name:'长剑',isMelee:true,hands:'1',category:'martial',toObject:()=>({type:'weapon',system:{slug:'longsword',baseItem:'longsword',category:'martial',group:'sword',level:{value:0},damage:{dice:1,die:'d8',damageType:'slashing'},traits:{value:[]},usage:{value:'held-in-one-hand'},range:null,runes:{potency:0,striking:0,property:[]},material:{type:null,grade:null},specific:null}})});
 const gm=await w.client('gm');
 assert.equal((await gm.api.execute({itemUuid:branch.uuid,operation:'shift',formUuid:uuid,nonce:'shiftuse0001'},w.player)).status,'done');
 assert.equal(branch.system.baseItem,'longsword');
 const [notice]=[...w.shared.messages.values()].filter(message=>message.flags?.[M]?.sogPilgrim?.announce);
 assert.equal(notice.whisper,undefined);assert.match(notice.content,/罗格斯/);assert.match(notice.content,/大杉枝/);assert.match(notice.content,/长剑/);
});

test('a haunt in the encounter turns the scarf haunt context on at the start and off at the end',async t=>{
 const w=world(t),scarf=w.reward('scarf'),gm=await w.client('gm'),player=await w.client('player');
 const haunt={id:'haunt',uuid:'Actor.haunt',type:'hazard',traits:new Set(['haunt']),items:new Map()};
 const combat=(actors,round=0)=>({id:'combat',round,get started(){return this.round>0;},combatants:new Map(actors.map((owner,index)=>[`c${index}`,{actor:owner}]))});
 const quiet=combat([w.actor,w.foe]);quiet.round=1;await gm.emit('updateCombat',quiet,{round:1},{},'gm');
 assert.equal(w.writes.some(write=>write.toggle),false);
 const haunted=combat([w.actor,haunt]);haunted.round=1;
 await player.emit('updateCombat',haunted,{round:1},{},'gm');assert.equal(w.writes.some(write=>write.toggle),false);
 await gm.emit('updateCombat',haunted,{round:1},{},'gm');
 assert.deepEqual(w.writes.filter(write=>write.toggle).map(write=>write.toggle),[['all',HAUNT,scarf.id,true]]);
 await gm.emit('updateCombat',haunted,{round:2},{},'gm');assert.equal(w.writes.filter(write=>write.toggle).length,1);
 await gm.emit('deleteCombat',haunted,{},'gm');
 assert.deepEqual(w.writes.filter(write=>write.toggle).at(-1).toggle,['all',HAUNT,scarf.id,false]);
 const joined=combat([w.actor]);joined.round=3;joined.combatants.set('late',{actor:haunt,parent:joined});
 await gm.emit('createCombatant',{actor:haunt,parent:joined},{},'gm');
 assert.deepEqual(w.writes.filter(write=>write.toggle).at(-1).toggle,['all',HAUNT,scarf.id,true]);
});

test('a GM edit of the leaf counter is adopted while TPA and player writes are not',async t=>{
 const w=world(t),fan=w.reward('fan');w.light(fan,1);
 const gm=await w.client('gm');await gm.api.reconcile(w.actor);
 const counter=[...w.actor.items.values()].find(item=>item.flags?.[M]?.sogPilgrim?.kind==='leaves');assert.equal(counter.system.badge.value,1);
 counter.system.badge.value=3;await gm.emit('updateItem',counter,{system:{badge:{value:3}}},{tpaPilgrimSync:true},'gm');
 assert.equal(w.state(fan).fan.leaves.length,1);
 await gm.emit('updateItem',counter,{system:{badge:{value:3}}},{},'player');assert.equal(w.state(fan).fan.leaves.length,1);
 await gm.emit('updateItem',counter,{system:{badge:{value:3}}},{},'gm');
 assert.equal(w.state(fan).fan.leaves.length,3);assert.equal(counter.system.badge.value,3);
 await w.actor.deleteEmbeddedDocuments('Item',[counter.id]);await gm.emit('deleteItem',counter,{},'gm');
 assert.equal(w.state(fan).fan.leaves.length,0);
});
