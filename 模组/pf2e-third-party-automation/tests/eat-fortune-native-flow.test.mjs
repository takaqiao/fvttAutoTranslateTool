import test from 'node:test';
import assert from 'node:assert/strict';
import {createEatFortune,EAT_FORTUNE_SOURCES as S} from '../scripts/eat-fortune.mjs';
const M='pf2e-third-party-automation';
const patch=async function(changes){for(const[path,value]of Object.entries(changes)){let at=this;const keys=path.split('.');for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=value;}return this;};
test('Eat Fortune offers its real reaction without geometry and conceals the source name from its owner',async()=>{
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:true},users=new Map([[gm.id,gm],[player.id,player]]);users.activeGM=gm;
 const actor={id:'reactor',uuid:'Actor.reactor',items:new Map(),flags:{},canAct:true,testUserPermission:u=>u===player||u===gm,update:patch};
 const source={id:'source',uuid:'Actor.source',name:'Secret actor',items:new Map(),testUserPermission:u=>u===gm,flags:{[M]:{reactionChecks:{reactions:[{kind:'clock',nonce:'paid-clock',state:'claimed',checkId:'clock-card'}]}}}};
 const item={id:'eat',uuid:`${actor.uuid}.Item.eat`,actor,type:'feat',sourceId:S.eat,system:{frequency:{value:1}},update:patch,toMessage:async()=>({id:'eat-card',flags:{},update:patch})};actor.items.set(item.id,item);
 const clock={id:'clock',uuid:`${source.uuid}.Item.clock`,actor:source,type:'feat',sourceId:S.clock};source.items.set(clock.id,clock);
 const scene={id:'scene',tokens:new Map()},make=a=>({id:a.id,uuid:`Scene.scene.Token.${a.id}`,documentName:'Token',actor:a,parent:scene,name:'Secret monster',playersCanSeeName:false,object:{distanceTo(){assert.fail('no distance checks')}}});
 const origin=make(source),reactor=make(actor);scene.tokens.set(origin.id,origin);scene.tokens.set(reactor.id,reactor);
 const game={user:gm,users,actors:new Map([[actor.id,actor],[source.id,source]]),scenes:new Map([[scene.id,scene]]),messages:new Map([['clock-card',{id:'clock-card',item:clock,flags:{[M]:{reactionChecks:{kind:'reaction-use',nonce:'paid-clock'}}}}]]),time:{worldTime:100},pf2e:{settings:{tokens:{nameVisibility:true}}}};
 const docs=new Map([actor,source,item,clock,origin,reactor].map(d=>[d.uuid,d])),prompts=[];
 const provider=createEatFortune({game,fromUuid:async uuid=>docs.get(uuid),choose:async request=>{prompts.push(request);return 'eat'}});provider.register({Hooks:{on:()=>1,off(){}}});
 const result=await provider.decide({nonce:'reaction-one',kind:'clock',clockNonce:'paid-clock',sourceItemUuid:clock.uuid,sourceActorUuid:source.uuid,sourceTokenUuid:origin.uuid,rollerActorUuid:source.uuid,rollerTokenUuid:origin.uuid,effectType:'fortune',type:'saving-throw',options:[]},gm);
 assert.equal(result.reactorActorUuid,actor.uuid);assert.equal(item.system.frequency.value,0);assert.equal(prompts.length,1);assert.equal(prompts[0].user,player);assert.doesNotMatch(prompts[0].title,/Secret/);assert.equal(prompts[0].choices.at(-1).value,'decline');
});
