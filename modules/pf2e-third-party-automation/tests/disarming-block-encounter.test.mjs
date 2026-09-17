import test from 'node:test';
import assert from 'node:assert/strict';
import {createDisarmingBlock,DISARMING_BLOCK_SOURCE} from '../scripts/disarming-block.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';

function assign(document,changes){for(const[path,value]of Object.entries(changes)){const keys=path.split('.');let parent=document;for(const key of keys.slice(0,-1))parent=parent[key]??={};parent[keys.at(-1)]=structuredClone(value);}}
function fixture({epoch='actual:3',ownTurn=true}={}){
 const user={id:'gm',isGM:true,active:true,settings:{showCheckDialogs:true}},hooks=new Map(),choices=[],native=[];
 const actor={id:'defender',uuid:'Actor.defender',type:'character',size:'med',flags:{},items:[{sourceId:DISARMING_BLOCK_SOURCE}],getStatistic:()=>({rank:1}),getReach:()=>5,testUserPermission:()=>true,async update(c){assign(this,c)}};
 const target={id:'enemy',uuid:'Actor.enemy',type:'character',size:'med',items:[]};
 const token={id:'defender',uuid:'Scene.scene.Token.defender',actor,parent:{id:'scene'},object:{distanceTo:()=>5}},targetToken={id:'enemy',uuid:'Scene.scene.Token.enemy',actor:target,parent:{id:'scene'},object:{}};
 actor.getActiveTokens=()=>[token];target.getActiveTokens=()=>[targetToken];
 const weapon={id:'weapon',uuid:'Actor.enemy.Item.weapon',type:'weapon',actor:target,system:{equipped:{carryType:'held',handsHeld:1},category:'martial',traits:{value:[]}}};target.items.push(weapon);
 const combatant={id:'defender-combatant',actor,token,initiative:20},enemy={id:'enemy-combatant',actor:target,token:targetToken,initiative:10};
 const actual={id:'actual',started:true,round:3,turn:ownTurn?0:1,turns:[combatant,enemy]},viewed={id:'viewed-other',started:true,round:10,turn:0,turns:[]};
 const game={user,users:Object.assign(new Map([[user.id,user]]),{activeGM:user}),messages:new Map(),actors:[actor,target],combat:viewed,combats:new Map([[actual.id,actual],[viewed.id,viewed]]),time:{worldTime:100},pf2e:{actions:new Map()}};
 const event={nonce:'native-block-nonce',actorUuid:actor.uuid,tokenUuid:token.uuid,attackerActorUuid:target.uuid,attackerTokenUuid:targetToken.uuid,attackItemUuid:weapon.uuid,weaponUuid:weapon.uuid,userId:user.id,epoch};
 const docs=new Map([actor,target,token,targetToken,weapon].map(d=>[d.uuid,d]));
 game.pf2e.actions.set('disarm',{toActionVariant:({cost})=>({async use(options){
  native.push({cost,options});const card={id:'disarm-check',actor,author:user,rolls:[{}],flags:{pf2e:{context:{type:'skill-check',outcome:'failure',options:['action:disarm',...options.rollOptions],target:{actor:target.uuid,token:targetToken.uuid}}}},updateSource(c){assign(this,c)}};
  hooks.get('preCreateChatMessage')(card);game.messages.set(card.id,card);return [{message:card}];
 }})});
 const provider=createDisarmingBlock({game,canvas:{tokens:{placeables:[]}},fromUuid:async u=>docs.get(u),validateConfirmed:async()=>true,choose:async args=>{choices.push(args);return args.choices.find(c=>c.value==='use:1')?.value??'use:0'}});
 provider.register({Hooks:{on:(n,f)=>{hooks.set(n,f);return n},off(){}}});
 return {game,actor,actual,combatant,token,provider,event,choices,native};
}

test('free Disarm asks and applies current MAP on the actual own turn despite a different viewed encounter',async()=>{
 const f=fixture(),result=await f.provider.handleConfirmed(f.event);
 assert.equal(result.status,'done');assert.deepEqual(f.choices[0].choices.map(c=>c.value),['use:0','use:1','use:2','decline']);
 assert.equal(f.native[0].cost,'free');assert.equal(f.native[0].options.multipleAttackPenalty,1);
 const receipt=f.actor.flags[M].disarmingBlock.uses[0];assert.equal(receipt.turnKey,'actual:3:0');
 const timing=f.game.messages.get(result.checkId).flags[M].disarmingBlock.timing;
 assert.deepEqual(timing.combat,{id:'actual',round:3,turn:0,combatantId:f.combatant.id});assert.equal(timing.initiative,20);
});

test('free Disarm outside the actual own turn remains MAP zero despite a different viewed encounter',async()=>{
 const f=fixture({ownTurn:false}),result=await f.provider.handleConfirmed(f.event);
 assert.equal(result.status,'done');assert.deepEqual(f.choices[0].choices.map(c=>c.value),['use:0','decline']);assert.equal(f.native[0].options.multipleAttackPenalty,0);
});

test('expired actual encounter and out-of-combat receipts cannot become fresh offers when another encounter is viewed',async()=>{
 for(const epoch of ['actual:2',null]){const f=fixture({epoch}),result=await f.provider.handleConfirmed(f.event);assert.equal(result.status,'ineligible');assert.equal(f.choices.length,0);assert.equal(f.native.length,0);}
});

test('ambiguous actual encounters cannot offer or roll a free Disarm',async()=>{
 const f=fixture();f.game.combats.set('duplicate',{...f.actual,id:'duplicate'});
 const result=await f.provider.handleConfirmed(f.event);assert.equal(result.status,'ineligible');assert.equal(f.choices.length,0);assert.equal(f.native.length,0);
});

test('moving to the next actual own turn while the offer is open expires the old trigger',async()=>{
 const f=fixture();f.game.users.get('gm').active=true;
 // The choice dialog is asynchronous; advance the authoritative encounter in it.
 const originalFind=f.choices.push.bind(f.choices);f.choices.push=(...args)=>{const r=originalFind(...args);f.actual.round++;return r};
 const result=await f.provider.handleConfirmed(f.event);assert.equal(result.status,'ineligible');assert.equal(f.native.length,0);
});
