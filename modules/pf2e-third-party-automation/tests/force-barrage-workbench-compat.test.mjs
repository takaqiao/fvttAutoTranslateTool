import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
let api={};try{api=await import('../scripts/force-barrage-workbench-compat.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const SOURCE='Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros-internal.Macro.784iD1y6DBFSB5d2';
const SPELL='Compendium.pf2e.spells-srd.Item.gKKqvLohtrSJj3BM';
const HASH='464728041aab5a3d230b5d15b32fcad270fed7372c1bfd94a4ad3ee54fe670cb';
// The exact installed command is an optional local dependency, never a copied
// public fixture. These tests execute its gated fragments with document doubles.
const command=process.env.FVTT_FORCE_BARRAGE_MACRO?await readFile(process.env.FVTT_FORCE_BARRAGE_MACRO,'utf8'):null;
const nativeOptions={skip:!command&&'Set FVTT_FORCE_BARRAGE_MACRO to the audited Workbench internal command'};
function fixture(commandOverride=command??'unreviewed command'){
 const calls=[],actor={id:'actor',uuid:'Actor.actor',type:'character',canAct:true,isDead:false,isOwner:true,items:new Map(),itemTypes:{feat:[],effect:[]}};
 const entry={id:'entry',uuid:'Actor.actor.Item.entry',actor,type:'spellcastingEntry',system:{prepared:{value:'spontaneous'},slots:{slot1:{value:3},slot2:{value:3},slot3:{value:2}}}};
 const item={id:'spell',uuid:'Actor.actor.Item.spell',actor,type:'spell',_stats:{compendiumSource:SPELL},system:{location:{value:'entry',signature:true}},link:'@UUID[Actor.actor.Item.spell]{Force Barrage}'};
 actor.items.set(entry.id,entry);actor.items.set(item.id,item);
 const scene={id:'scene',uuid:'Scene.scene',tokens:new Map()};
 function token(id,owner=true){const document={id,uuid:`Scene.scene.Token.${id}`,parent:scene,name:'Creature',displayName:50,texture:{src:'portrait.webp'},actor:{name:'Creature',isOwner:owner}};scene.tokens.set(id,document);return document}
 const sourceToken=token('source');sourceToken.actor=actor;
 const targetA=token('targetA'),targetB=token('targetB');
 const game={version:'14.368',system:{id:'pf2e',version:'8.5.1'},user:{id:'owner',active:true},actors:new Map([[actor.id,actor]]),scenes:new Map([[scene.id,scene]]),modules:new Map([['xdy-pf2e-workbench',{active:true,version:'7.7.5'}],['pf2e-toolbelt',{active:true,version:'3.56.2'}]])};
 const macro={uuid:SOURCE,documentName:'Macro',type:'script',command:commandOverride};
 const fromUuid=async uuid=>{calls.push(['load',uuid]);return macro};
 class DamageRoll{constructor(formula){this.formula=formula;calls.push(['construct',this])}evaluate(){throw Error('Only bridge may evaluate')}toMessage(){throw Error('Only bridge may publish')}}
 globalThis.CONFIG={Dice:{rolls:[DamageRoll]}};
 globalThis.ChatMessage={getSpeaker:({actor,token}={})=>{assert.equal(actor?.id,'actor');assert.equal(token?.id,'source');return {actor:actor.id,scene:scene.id,token:token.id}}};
 const bridge={payAndBindOriginalCast:async()=>{calls.push(['pay']);return {nonce:'verified-original-cast'}},publishTarget:async args=>{calls.push(['publish',args]);return {messageUuid:'ChatMessage.'+args.targetUuid}}};
 const input={actor,token:sourceToken,item,entry,rank:3,actions:3,allocations:[{targetUuid:targetA.uuid,count:4,targetToken:targetA},{targetUuid:targetB.uuid,count:2,targetToken:targetB}],bridge};
 return {game,macro,fromUuid,input,calls,actor,item,entry,scene,sourceToken,targetA,targetB,DamageRoll};
}
const publications=f=>f.calls.filter(c=>c[0]==='publish').map(c=>c[1]);
async function load(f){return api.loadForceBarrageWorkbench({game:f.game,fromUuid:f.fromUuid})}
test('exports the exact internal macro source and loader',()=>{assert.equal(api.FORCE_BARRAGE_WORKBENCH_SOURCE,SOURCE);assert.equal(typeof api.loadForceBarrageWorkbench,'function')});
for(const [name,change]of [
 ['disabled Workbench',f=>f.game.modules.get('xdy-pf2e-workbench').active=false],
 ['unknown Workbench version',f=>f.game.modules.get('xdy-pf2e-workbench').version='7.7.6'],
 ['unknown PF build',f=>f.game.system.version='8.5.2'],
 ['unknown core build',f=>f.game.version='14.369'],
 ['unknown Toolbelt version',f=>f.game.modules.get('pf2e-toolbelt').version='3.57.0'],
 ['public macro UUID',f=>f.macro.uuid=SOURCE.replace('-internal','')],
 ['wrong document',f=>f.macro.documentName='Item'],
 ['unknown complete command hash',()=>{}],
])test(`rejects ${name} without payment or roll`,async()=>{const f=fixture('unreviewed command');change(f);await assert.rejects(load(f),error=>error.code?.startsWith('force-barrage-'));assert.equal(f.calls.some(c=>c[0]==='pay'||c[0]==='construct'),false)});
test('audited command bytes and nine counts remain those of the original count statement',nativeOptions,async()=>{
 assert.equal(createHash('sha256').update(command).digest('hex'),HASH);
 const statement=command.split('\n').find(line=>line.startsWith('const multi = '));
 const upstream=new Function('mmdiag','mmch',statement+'\nreturn multi;');
 const f=fixture(),adapter=await load(f);assert.ok(Object.isFrozen(adapter));
 for(let rank=1;rank<=3;rank++)for(let actions=1;actions<=3;actions++)assert.equal(adapter.getMissileCount({rank,actions}),upstream([null,actions],{rank}));
 for(const rank of [0,4,NaN,Infinity,1.5,'3'])assert.throws(()=>adapter.getMissileCount({rank,actions:1}));
 for(const actions of [0,4,NaN,-1,1.5,'3'])assert.throws(()=>adapter.getMissileCount({rank:1,actions}));
});
test('pays once first, preserves native rolls and original message template, publishes sequentially',nativeOptions,async()=>{
 const f=fixture(),adapter=await load(f);const result=await adapter.run(f.input);
 assert.deepEqual(f.calls.map(c=>c[0]),['load','pay','construct','publish','construct','publish']);
 const sent=publications(f);assert.deepEqual(sent.map(s=>s.roll.formula),['(4d4 + 4)[force]','(2d4 + 2)[force]']);
 for(const [i,p]of sent.entries()){assert.ok(p.roll instanceof f.DamageRoll);assert.equal(p.roll,f.calls.filter(c=>c[0]==='construct')[i][1]);assert.deepEqual(p.messageData.flags,{'pf2e-toolbelt.targetHelper.targets':[p.targetUuid]});assert.deepEqual(p.messageData.speaker,{actor:'actor',scene:'scene',token:'source'});assert.ok(p.messageData.flavor.endsWith('<br>'+f.item.link))}
 assert.deepEqual(result,{status:'completed',missiles:6,targets:sent.map((p,i)=>({targetUuid:p.targetUuid,count:[4,2][i],result:{messageUuid:'ChatMessage.'+p.targetUuid}})),display:{status:'manual'}});
});
for(const [name,change]of [
 ['NaN',f=>f.input.allocations[0].count=NaN],['infinity',f=>f.input.allocations[0].count=Infinity],['negative',f=>f.input.allocations[0].count=-1],['fraction',f=>f.input.allocations[0].count=4.5],['string',f=>f.input.allocations[0].count='4'],
 ['too few',f=>f.input.allocations[0].count=3],['too many',f=>f.input.allocations[0].count=5],['no allocation',f=>f.input.allocations=null],['empty',f=>f.input.allocations=[]],
 ['duplicate target',f=>f.input.allocations[1]={...f.input.allocations[0],count:2}],['mismatched UUID',f=>f.input.allocations[0].targetUuid=f.targetB.uuid],['deleted target',f=>f.scene.tokens.delete(f.targetA.id)],
 ['wrong actor',f=>f.input.actor={...f.actor}],['nonowned source',f=>f.actor.isOwner=false],['wrong spell source',f=>f.item._stats.compendiumSource='unknown'],['prepared entry',f=>f.entry.system.prepared.value='prepared'],['not signature',f=>f.item.system.location.signature=false],['empty slot',f=>f.entry.system.slots.slot3.value=0],
 ['future damage bonus',f=>f.actor.itemTypes.feat.push({slug:'sorcerous-potency'})],['cannot act',f=>f.actor.canAct=false],['missing bridge',f=>f.input.bridge=null],
])test(`rejects ${name} before payment`,nativeOptions,async()=>{const f=fixture(),adapter=await load(f);change(f);await assert.rejects(adapter.run(f.input),error=>error.code?.startsWith('force-barrage-'));assert.deepEqual(f.calls.map(c=>c[0]),['load'])});
test('a legitimate heightened variant keeps its live original spell identity',nativeOptions,async()=>{const f=fixture(),adapter=await load(f);f.input.item={...f.item,original:f.item};await adapter.run(f.input);assert.equal(publications(f).length,2)});
test('unknown or changed dependencies cannot use a previously loaded adapter',nativeOptions,async()=>{for(const change of [f=>f.macro.command+=' ',f=>f.game.modules.get('xdy-pf2e-workbench').active=false]){const f=fixture(),adapter=await load(f);change(f);await assert.rejects(adapter.run(f.input));assert.deepEqual(f.calls.map(c=>c[0]),['load'])}});
test('false, absent and failed payment never construct or publish a roll',nativeOptions,async()=>{for(const value of [false,null,undefined,true]){const f=fixture(),adapter=await load(f);f.input.bridge.payAndBindOriginalCast=async()=>{f.calls.push(['pay']);return value};await assert.rejects(adapter.run(f.input));assert.deepEqual(f.calls.map(c=>c[0]),['load','pay'])}const f=fixture(),adapter=await load(f);f.input.bridge.payAndBindOriginalCast=async()=>{throw Error('uncertain payment')};await assert.rejects(adapter.run(f.input),/uncertain payment/);assert.equal(publications(f).length,0)});
test('freezes allocation and native display privacy before awaiting payment; skips zero allocation accurately',nativeOptions,async()=>{
 const f=fixture(),adapter=await load(f);f.input.allocations[0].count=0;f.input.allocations[1].count=6;f.targetB.actor.isOwner=false;f.targetB.displayName=20;
 Object.defineProperty(f.game.user,'targets',{get(){throw Error('Never reread current targets')}});globalThis.canvas={get tokens(){throw Error('Never reread controlled tokens')}};
 f.input.bridge.payAndBindOriginalCast=async()=>{f.calls.push(['pay']);f.input.allocations[1].targetUuid='changed';f.input.allocations[1].count=99;f.targetB.name='Late name';return {nonce:'paid'}};
 await adapter.run(f.input);const [p]=publications(f);assert.equal(p.targetUuid,f.targetB.uuid);assert.equal(p.roll.formula,'(6d4 + 6)[force]');assert.match(p.messageData.flavor,/<figcaption>Target #2<\/figcaption>/);assert.equal(p.messageData.flavor.includes('Creature'),false);assert.equal(p.messageData.flavor.includes('Late name'),false);assert.equal(Object.isFrozen(f.targetB),false);
});
test('a delivery error stops remaining targets and never retries payment or roll',nativeOptions,async()=>{const f=fixture(),adapter=await load(f);f.input.bridge.publishTarget=async args=>{f.calls.push(['publish',args]);throw Error('uncertain delivery')};await assert.rejects(adapter.run(f.input),/uncertain delivery/);assert.deepEqual(f.calls.map(c=>c[0]),['load','pay','construct','publish'])});
test('each delivered formula and message data match the original audited construction and template bytes',nativeOptions,async()=>{
 const lines=command.split('\n');
 const originalDamage=new Function('token','mmch','a','DamageRoll',lines.slice(155,172).join('\n')+'\nreturn droll;');
 const originalMessage=new Function('a','mmch','ChatMessage','return ('+lines.slice(173,178).join('\n')+');');
 const f=fixture(),adapter=await load(f);await adapter.run(f.input);
 for(const [index,p]of publications(f).entries()){
  const a={num:[4,2][index],name:'Creature',uuid:p.targetUuid},mmch={rank:3,link:f.item.link};
  const original=originalDamage({actor:f.actor},mmch,a,f.DamageRoll);
  assert.equal(p.roll.formula,original.formula);
  assert.deepEqual(p.messageData,originalMessage(a,mmch,{getSpeaker:()=>({actor:'actor',scene:'scene',token:'source'})}));
 }
});
