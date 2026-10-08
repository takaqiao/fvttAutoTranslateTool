import test from 'node:test';
import assert from 'node:assert/strict';
import {rewardKey,usableReward,scarfBranch} from '../scripts/sog-pilgrim-rules.mjs';
import {activationOperations,assertAvailable,createPilgrimRewards,hasPublishedUse,playerError,prepareScarfChoices} from '../scripts/sog-pilgrim-rewards.mjs';
const item=(key,type='weapon')=>({type,actor:{},system:{equipped:{carryType:type==='weapon'?'held':'worn',handsHeld:1,invested:true}},flags:{world:{sogWontonNativeAutomation:{key},sogB2Ch2Resources:{originalSlug:key,fileSha256:'a'.repeat(64)}}}});
test('a copied reward retains its identity without trusting its Chinese name',()=>{
 const fan=item('spirit-fan');fan.name='铁扇';assert.equal(rewardKey(fan),'fan');
 fan.flags.world.sogB2Ch2Resources.originalSlug='another-fan';assert.equal(rewardKey(fan),null);
});
test('wearable rewards require investment, weapons require a held hand',()=>{
 const scarf=item('ghost-scarf','equipment');assert.equal(usableReward(scarf),true);
 scarf.system.equipped.invested=false;assert.equal(usableReward(scarf),false);
 const branch=item('branch-of-the-great-sugi');branch.system.equipped.handsHeld=0;assert.equal(usableReward(branch),false);
});

test('native equipped and invested state overrides raw worn flags',()=>{
 const scarf=item('ghost-scarf','equipment');
 scarf.isEquipped=false;scarf.isInvested=false;
 assert.equal(usableReward(scarf),false);
 scarf.isEquipped=true;scarf.isInvested=false;
 assert.equal(usableReward(scarf),false);
 scarf.isInvested=true;
 assert.equal(usableReward(scarf),true);
 const branch=item('branch-of-the-great-sugi');branch.isEquipped=false;
 assert.equal(usableReward(branch),false);
});
test('scarf chooses exactly one branch from the selected weapon',()=>{
 assert.equal(scarfBranch({system:{runes:{property:[]}}}),'ghost');
 assert.equal(scarfBranch({system:{runes:{property:['ghostTouch']}}}),'astral');
 assert.equal(scarfBranch({system:{runes:{property:['astral','ghostTouch']}}}),'ward');
});
test('each reward offers only its own activation',()=>{
 assert.deepEqual(activationOperations('fan'),['release']);
 assert.deepEqual(activationOperations('scarf'),['scarf']);
 assert.deepEqual(activationOperations('branch'),['shift','tree','restore']);
 assert.deepEqual(activationOperations('hairpin'),['storm']);
});
test('hourly release and daily preparation have different recovery boundaries',()=>{
 assert.throws(()=>assertAvailable({lastUse:{release:400}},'release',3999),/每小时/);
 assert.doesNotThrow(()=>assertAvailable({lastUse:{release:400}},'release',4000));
 assert.throws(()=>assertAvailable({spent:true},'storm',100000000),/每日/);
 assert.doesNotThrow(()=>assertAvailable({spent:false},'storm',0));
});
test('uncertain previous side effects prevent a second activation',()=>{
 assert.throws(()=>assertAvailable({use:{status:'pending'}},'scarf',0),/核对/);
 assert.doesNotThrow(()=>assertAvailable({use:{status:'done'},spent:false},'scarf',0));
});

test('reward cards use the registered PF2e DamageRoll when it is not exported on game.pf2e',async()=>{
 const previous={CONFIG:globalThis.CONFIG,ChatMessage:globalThis.ChatMessage,foundry:globalThis.foundry};
 const published=[];
 class DamageRoll {
  constructor(formula){this.formula=formula;}
  async evaluate(){return this;}
  async toMessage(data,options){published.push({formula:this.formula,data,options});return {id:'nativeCard'};}
 }
 globalThis.CONFIG={Dice:{rolls:[DamageRoll]}};
 globalThis.ChatMessage={getSpeaker:()=>({actor:'actor'}),getWhisperRecipients:()=>[{id:'gm'}]};
 globalThis.foundry={applications:{ux:{TextEditor:{enrichHTML:async text=>text}}}};
 try{
  const game={system:{id:'pf2e'},world:{id:'sog-pilgrim-qa'},pf2e:{},settings:{get:()=> 'publicroll'}};
  const api=await createPilgrimRewards({game,fromUuid:async()=>null});
  await api.card({item:{uuid:'Actor.actor.Item.reward',type:'equipment',actor:{uuid:'Actor.actor'}},formula:'{1d10[slashing],1d10[vitality]}',label:'花瓣风暴',nonce:'native-card',gm:true,save:{type:'reflex',dc:23}});
  assert.equal(published.length,1);assert.equal(published[0].formula,'{1d10[slashing],1d10[vitality]}');
  assert.deepEqual(published[0].data.whisper,['gm']);
  assert.equal(published[0].data.flags.pf2e.origin.uuid,'Actor.actor.Item.reward');
 }finally{for(const [key,value]of Object.entries(previous))if(value===undefined)delete globalThis[key];else globalThis[key]=value;}
});

test('partial placement and unrelated cards cannot complete an uncertain activation',()=>{
 const item={uuid:'Actor.actor.Item.reward'};
 const use={nonce:'same-use',operation:'storm'};
 const mark=(source,nonce)=>({flags:{'pf2e-third-party-automation':{sogPilgrim:{generated:true,source,nonce}}}});
 const game={messages:new Map(),actors:new Map()};
 game.messages.set('wrong-use',{...mark(item.uuid,'another-use'),isDamageRoll:true});
 assert.equal(hasPublishedUse(item,use,game),false);
 game.messages.set('plain-card',mark(item.uuid,use.nonce));
 assert.equal(hasPublishedUse(item,use,game),false);
 game.messages.set('committed-card',{...mark(item.uuid,use.nonce),isDamageRoll:true});
 assert.equal(hasPublishedUse(item,use,game),true);
});

test('internal errors are kept out of player notices',()=>{
 assert.equal(playerError(Error('Item validation: source UUID and module path')), '启动未完成，请联系GM核对。');
});

test('manual scarf effects include carried and stowed weapons while excluding dropped and wrong-rune weapons',()=>{
 const mark={key:'effect-ghost-scarf-ghost-touch'};
 const template={flags:{world:{sogWontonNativeAutomation:mark}}};
 const data={type:'effect',flags:{world:{sogWontonNativeAutomation:{...mark}}},system:{rules:[{key:'ChoiceSet',flag:'weapon',choices:{ownedItems:true}}]}};
 const weapon=(id,carryType,runes=[])=>({id,name:id,type:'weapon',system:{equipped:{carryType},runes:{property:runes}}});
 const actor={items:[weapon('held','held'),weapon('stowed','stowed'),weapon('dropped','dropped'),weapon('astral','held',['astral'])]};
 prepareScarfChoices(data,actor,new Map([['ffebdbe91ceb569e',template]]));
 assert.deepEqual(data.system.rules[0].choices.map(choice=>choice.value),['held','stowed']);
});

test('native rest by a player requests serialized daily recovery from the active GM',async()=>{
 const actor={uuid:'Actor.owner',items:new Map()};const registered=new Map(),sent=[];
 const game={system:{id:'pf2e'},world:{id:'sog'},user:{id:'player'},users:{activeGM:{id:'gm'}},actors:[],scenes:[]};
 const api=await createPilgrimRewards({game,fromUuid:async()=>null});
 api.register({Hooks:{on:(name,handler)=>registered.set(name,handler)},socket:{register(){},executeAsUser:async(...args)=>{sent.push(args);return {ok:true};}}});
 await registered.get('pf2e.restForTheNight')(actor);
 assert.deepEqual(sent,[['pilgrim-rest','gm',{actorUuid:'Actor.owner'}]]);
});

test('player cosmetics restore after Sequencer is ready and follow native leaf changes',async()=>{
 const previous={Sequence:globalThis.Sequence,Sequencer:globalThis.Sequencer};
 const played=[],ended=[],registered=new Map();
 class Sequence {
  constructor(){this.count=0;}
  effect(){this.count++;return this;}
  async play(options){played.push({count:this.count,options});}
 }
 for(const method of ['file','name','attachTo','tieToDocuments','scaleToObject','spriteOffset','opacity','persist','temporary','fadeIn','fadeOut'])Sequence.prototype[method]=function(){return this;};
 globalThis.Sequence=Sequence;
 globalThis.Sequencer={EffectManager:{getEffects:()=>[],endEffects:async(...args)=>ended.push(args)}};
 try{
  const actor={uuid:'Actor.owner',items:new Map()},fan={...item('spirit-fan'),id:'fan',uuid:'Actor.owner.Item.fan',actor};
  const effect={id:'leaves',uuid:'Actor.owner.Item.leaves',type:'effect',actor,system:{badge:{value:2}},flags:{'pf2e-third-party-automation':{sogPilgrim:{kind:'leaves',source:fan.uuid}}}};
  actor.items.set(fan.id,fan);actor.items.set(effect.id,effect);
  const scene={id:'scene',tokens:new Map()},token={parent:scene,actor,object:{}};scene.tokens.set('token',token);
  const game={system:{id:'pf2e'},world:{id:'sog'},user:{id:'player'},users:{activeGM:{id:'gm'}},actors:[actor],scenes:new Map([['scene',scene]]),modules:new Map([['sequencer',{active:true}]])};
  const canvas={ready:false,scene};
  const api=await createPilgrimRewards({game,canvas,fromUuid:async()=>null});
  api.register({Hooks:{on:(name,handler)=>registered.set(name,handler)},socket:{register(){}}});
  canvas.ready=true;
  await registered.get('canvasReady')();assert.equal(played.length,0);
  await registered.get('sequencerEffectManagerReady')();assert.deepEqual(played,[{count:2,options:{local:true}}]);
  effect.system.badge.value=3;await registered.get('updateItem')(effect,{'system.badge.value':3});assert.equal(played.at(-1).count,3);
  const before=ended.length;await registered.get('updateItem')({actor,type:'weapon',flags:{}},{});assert.equal(ended.length,before);
  actor.items.delete(effect.id);await registered.get('deleteItem')(effect);assert.equal(played.length,2);assert.equal(ended.length,before+1);assert.equal(ended.at(-1)[1],false);
  actor.items.set(effect.id,effect);
  const late=await createPilgrimRewards({game,canvas,fromUuid:async()=>null});
  late.register({Hooks:{on(){}},socket:{register(){}}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(played.length,3);assert.equal(played.at(-1).count,3);
 }finally{for(const [key,value]of Object.entries(previous))if(value===undefined)delete globalThis[key];else globalThis[key]=value;}
});
