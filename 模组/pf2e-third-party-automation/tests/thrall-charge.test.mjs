import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID} from '../scripts/rules.mjs';
import {createNativeChooser} from '../scripts/native-context.mjs';
import {createThrallCharge,THRALL_CHARGE_SPELL,THRALL_CHARGE_EFFECT,SUMMONS_THRALL_CHARGE_EFFECT} from '../scripts/thrall-charge.mjs';

const patch=(doc,changes)=>{for(const[key,value]of Object.entries(changes)){const parts=key.split('.');let at=doc;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=structuredClone(value);}};
// Rules as published in Compendium.pf2e.spell-effects (Spell Effect: Thrall Charge).
const effectSource=()=>({_id:'Wd1fYphfUNpZlgcH',name:'Spell Effect: Thrall Charge',type:'effect',img:'icons/thrall-charge.webp',_stats:{compendiumSource:null},flags:{},system:{slug:'spell-effect-thrall-charge',level:{value:1},duration:{value:1,unit:'rounds',expiry:'turn-end',sustained:false},rules:[
 {key:'DamageDice',selector:'strike-damage',diceNumber:'ternary(gte(@item.level,10),4,ternary(gte(@item.level,6),3,ternary(gte(@item.level,2),2,1)))',dieSize:'d6'},
 {key:'ChoiceSet',choices:[{label:'PF2E.Yes',value:'yes'},{label:'PF2E.No',value:'no'}],prompt:'PF2E.SpecificRule.SpellEffectThrallCharge.Prompt.DestroyThrall',rollOption:'thrall-charge:destroy-thrall'},
 {key:'FlatModifier',selector:'strike-damage',type:'status',value:'@item.level',predicate:['thrall-charge:destroy-thrall:yes']},
]}});

function fixture({castRank=2,casting=true,actualCast=true,spellSource=THRALL_CHARGE_SPELL,answers=[]}={}){
 const listeners=new Map(),errors=[],prompts=[],deleted=[];let nextId=0;
 const Hooks={on(name,fn){listeners.set(name,fn);return name;},off(name){listeners.delete(name);}};
 const gmUser={id:'gm',isGM:true},player={id:'player',isGM:false,targets:new Set()};
 const users=Object.assign(new Map([[gmUser.id,gmUser],[player.id,player]]),{activeGM:gmUser});
 const scene={id:'scene',tokens:new Map()};
 const game={user:gmUser,users,scenes:new Map([[scene.id,scene]]),messages:new Map()};
 const makeItem=(actor,source)=>{const item={...source,id:`item${++nextId}`,parent:actor,actor,flags:source.flags??{},system:source.system??{},async update(changes){patch(this,changes);return this;},async delete(){actor.items.delete(this.id);return this;}};delete item._id;return item;};
 const makeActor=props=>{const actor={items:new Map(),...props,
  async createEmbeddedDocuments(_type,data){const created=[];for(const source of data){const item=makeItem(actor,structuredClone(source));if(listeners.get('preCreateItem')?.(item,source,{},game.user.id)===false)continue;actor.items.set(item.id,item);created.push(item);}return created;},
  async deleteEmbeddedDocuments(_type,ids){for(const id of ids)actor.items.delete(id);return ids;}};return actor;};
 const caster=makeActor({id:'necro',uuid:'Actor.necro',type:'character',name:'Necromancer',testUserPermission:(user,level)=>level==='OWNER'&&(user.isGM||user.id===player.id)});
 const spell={id:'spell',uuid:'Actor.necro.Item.spell',type:'spell',actor:caster,_stats:{compendiumSource:spellSource}};caster.items.set(spell.id,spell);
 const casterToken={id:'caster',uuid:'Scene.scene.Token.caster',documentName:'Token',name:'Necromancer',parent:scene,actor:caster,object:{distanceTo:target=>target.distance}};scene.tokens.set(casterToken.id,casterToken);
 function addThrall(id,{distance=10,owner=caster.uuid,hp=1}={}){
  const actor=makeActor({id:'thrall-base',uuid:`Scene.scene.Token.${id}.Actor.thrall-base`,type:'npc',name:'Thrall',flags:{'pf2e-summons-assistant':{summoner:{uuid:owner}}},rollOptions:{all:{'self:trait:thrall':true}},system:{attributes:{hp:{value:hp}}}});
  const token={id,uuid:`Scene.scene.Token.${id}`,documentName:'Token',name:`Thrall ${id}`,parent:scene,actor,object:{distance},async delete(){deleted.push(id);scene.tokens.delete(id);return this;}};
  scene.tokens.set(id,token);return token;
 }
 const effectDoc={toObject:effectSource};
 const fromUuid=async uuid=>uuid===spell.uuid?spell:uuid===THRALL_CHARGE_EFFECT?effectDoc:scene.tokens.get(uuid.split('.')[3])??null;
 const choose=createNativeChooser({game,send:async(userId,payload)=>{prompts.push({userId,...payload});const answer=answers.length?answers.shift():payload.choices[0].value;return typeof answer==='function'?answer(payload):answer;}});
 const castMessage=(id='cast',props={})=>{const message={id,author:player,speaker:{scene:scene.id,token:casterToken.id,actor:caster.id},flags:{pf2e:{origin:{actor:caster.uuid,uuid:spell.uuid,type:'spell',castRank,rollOptions:actualCast?['origin:item:slug:thrall-charge','origin:action:slug:cast-a-spell']:['origin:item:slug:thrall-charge']},...casting?{casting:{id:'entry',tradition:'occult'}}:{}}},async update(changes){patch(this,changes);return this;},...props};game.messages.set(id,message);return message;};
 const strikeMessage=(token,kind,outcome)=>({id:`${kind}${++nextId}`,author:player,speaker:{scene:scene.id,token:token.id,actor:token.actor.id},flags:{pf2e:{context:kind==='attack'?{type:'attack-roll',action:'strike',domains:['strike-attack-roll','attack-roll'],outcome}:{type:'damage-roll',sourceType:'attack',domains:['strike-damage','damage'],outcome}}}});
 const provider=createThrallCharge({game,fromUuid,choose,onError:error=>errors.push(error.message)});
 const unregister=provider.register({Hooks});
 return {game,gmUser,player,scene,caster,casterToken,addThrall,castMessage,strikeMessage,prompts,errors,deleted,listeners,unregister,
  emit:message=>listeners.get('createChatMessage')(message,{},message.author?.id),
  effects:token=>[...token.actor.items.values()].filter(item=>item.type==='effect'),
  state:id=>game.messages.get(id).flags[MODULE_ID]?.thrallCharge};
}

test("the author's one targeted thrall is used without asking which thrall",async()=>{
 const f=fixture(),a=f.addThrall('a'),b=f.addThrall('b');f.player.targets.add({document:b});
 await f.emit(f.castMessage());
 assert.deepEqual(f.prompts.map(p=>[p.userId,p.actorUuid,p.title]),[['player','Actor.necro','奴仆冲锋：是否在这次打击中摧毁奴仆']]);
 assert.equal(f.effects(a).length,0);assert.equal(f.effects(b).length,1);assert.deepEqual(f.errors,[]);
});

test('a single candidate is used without asking which thrall',async()=>{
 const f=fixture(),a=f.addThrall('a');
 await f.emit(f.castMessage());
 assert.equal(f.prompts.length,1);assert.equal(f.effects(a).length,1);
 assert.equal(f.effects(a)[0].system.rules.find(rule=>rule.key==='ChoiceSet').selection,'no');
 assert.equal(f.state('cast').status,'applied');
});

test('several candidates without one targeted thrall ask the author',async()=>{
 const f=fixture({answers:[payload=>payload.choices[1].value]}),a=f.addThrall('a'),b=f.addThrall('b',{distance:25});
 f.player.targets.add({document:a});f.player.targets.add({document:b});
 await f.emit(f.castMessage());
 assert.equal(f.prompts.length,2);assert.equal(f.prompts[0].userId,'player');
 assert.deepEqual(f.prompts[0].choices,[{value:a.uuid,label:'Thrall a · 10尺'},{value:b.uuid,label:'Thrall b · 25尺'}]);
 assert.equal(f.effects(a).length,0);assert.equal(f.effects(b).length,1);
});

test('no thrall within 30 feet reports a Chinese error and records the failure',async()=>{
 const f=fixture();
 await f.emit(f.castMessage());
 assert.equal(f.errors.length,1);assert.match(f.errors[0],/^奴仆冲锋：施法者30尺内没有/);
 assert.equal(f.state('cast').status,'failed');assert.equal(f.prompts.length,0);
});

test('thralls beyond 30 feet, owned by another caster or destroyed are not candidates',async()=>{
 const f=fixture(),far=f.addThrall('far',{distance:35}),foreign=f.addThrall('foreign',{owner:'Actor.other'}),dead=f.addThrall('dead',{hp:0}),near=f.addThrall('near',{distance:30});
 f.player.targets.add({document:far});
 await f.emit(f.castMessage());
 assert.equal(f.prompts.length,1);assert.equal(f.effects(near).length,1);
 for(const token of [far,foreign,dead])assert.equal(f.effects(token).length,0);
});

test('the effect carries the cast rank, a preset ChoiceSet and its compendium source',async()=>{
 const f=fixture({castRank:6,answers:['destroy']}),token=f.addThrall('a');
 await f.emit(f.castMessage());
 const labels=f.prompts[0].choices.map(choice=>choice.label);
 assert.deepEqual(f.prompts[0].choices.map(choice=>choice.value),['keep','destroy']);
 assert.match(labels[0],/3d6/);assert.match(labels[1],/3d6/);assert.match(labels[1],/\+6 状态加值/);
 const [effect]=f.effects(token);
 assert.equal(effect.system.level.value,6);
 assert.equal(effect.system.rules.find(rule=>rule.key==='ChoiceSet').selection,'yes');
 assert.equal(effect._stats.compendiumSource,THRALL_CHARGE_EFFECT);
 assert.deepEqual(effect.flags[MODULE_ID],{thrallCharge:{messageId:'cast',casterUuid:'Actor.necro',destroy:true,rank:6},nativeEffectKey:'thrall-charge'});
 assert.deepEqual(f.state('cast'),{status:'applied',rank:6,destroy:true,thrallUuid:token.uuid,effectId:effect.id});
});

test('extra dice follow the heightened ranks',async()=>{
 for(const [castRank,dice] of [[1,'1d6'],[2,'2d6'],[5,'2d6'],[6,'3d6'],[9,'3d6'],[10,'4d6']]){
  const f=fixture({castRank});f.addThrall('a');
  await f.emit(f.castMessage());
  assert.match(f.prompts[0].choices[0].label,new RegExp(`额外 ${dice} `));
 }
});

test('an invalid cast rank is reported instead of guessed',async()=>{
 const f=fixture({castRank:0}),token=f.addThrall('a');
 await f.emit(f.castMessage());
 assert.equal(f.errors.length,1);assert.match(f.errors[0],/环级/);assert.equal(f.effects(token).length,0);
});

for(const [step,thralls] of [['thrall',2],['destroy',1]])test(`cancelling the ${step} choice adds no effect`,async()=>{
 const f=fixture({answers:[null]}),tokens=Array.from({length:thralls},(_,index)=>f.addThrall(`t${index}`));
 await f.emit(f.castMessage());
 assert.equal(f.prompts.length,1);assert.equal(f.state('cast').status,'cancelled');assert.deepEqual(f.errors,[]);
 for(const token of tokens)assert.equal(f.effects(token).length,0);
});

test('thrall charge effects from PF2e, Summons Assistant or an earlier cast are replaced; other effects stay',async()=>{
 const f=fixture(),token=f.addThrall('a');
 await token.actor.createEmbeddedDocuments('Item',[
  {type:'effect',name:'Summons Charge',_stats:{compendiumSource:SUMMONS_THRALL_CHARGE_EFFECT},system:{slug:'effect-thrall-charge'}},
  {type:'effect',name:'Dragged Charge',system:{slug:'spell-effect-thrall-charge'}},
  {type:'effect',name:'Thrall Expiration',system:{slug:'effect-thrall-expiration-date'}},
 ]);
 await f.emit(f.castMessage('first'));
 await f.emit(f.castMessage('second'));
 const effects=f.effects(token);
 assert.deepEqual(effects.map(effect=>effect.name).sort(),['Spell Effect: Thrall Charge','Thrall Expiration']);
 assert.equal(effects.find(effect=>effect.name==='Spell Effect: Thrall Charge').flags[MODULE_ID].thrallCharge.messageId,'second');
});

test('an unmarked thrall charge effect cannot be added beside ours, on any client',async()=>{
 const f=fixture(),token=f.addThrall('a'),other=f.addThrall('b');f.player.targets.add({document:token});
 await f.emit(f.castMessage());
 f.game.user=f.player;
 const created=await token.actor.createEmbeddedDocuments('Item',[
  {type:'effect',name:'Summons Charge',_stats:{compendiumSource:SUMMONS_THRALL_CHARGE_EFFECT},system:{slug:'effect-thrall-charge'}},
  {type:'effect',name:'Native Charge',_stats:{compendiumSource:THRALL_CHARGE_EFFECT},system:{slug:'spell-effect-thrall-charge'}},
  {type:'effect',name:'Unrelated',system:{slug:'unrelated'}},
 ]);
 assert.deepEqual(created.map(effect=>effect.name),['Unrelated']);
 assert.equal((await other.actor.createEmbeddedDocuments('Item',[{type:'effect',name:'Summons Charge',system:{slug:'effect-thrall-charge'}}])).length,1);
 f.unregister();assert.equal(f.listeners.size,0);
});

for(const [answer,destroyed] of [['keep',[]],['destroy',['a']]])for(const outcome of ['failure','criticalFailure'])test(`a ${outcome} strike ends the effect at once (${answer})`,async()=>{
 const f=fixture({answers:[answer]}),token=f.addThrall('a');
 await f.emit(f.castMessage());
 await f.emit(f.strikeMessage(token,'attack',outcome));
 assert.equal(f.effects(token).length,0);assert.deepEqual(f.deleted,destroyed);assert.deepEqual(f.errors,[]);
});

for(const [answer,destroyed] of [['keep',[]],['destroy',['a']]])for(const outcome of ['success','criticalSuccess'])test(`a ${outcome} strike waits for its damage roll (${answer})`,async()=>{
 const f=fixture({answers:[answer]}),token=f.addThrall('a');
 await f.emit(f.castMessage());
 const attack=f.strikeMessage(token,'attack',outcome);
 await f.emit(attack);
 const [effect]=f.effects(token);
 assert.deepEqual(effect.flags[MODULE_ID].thrallCharge.attack,{messageId:attack.id,outcome});assert.deepEqual(f.deleted,[]);
 await f.emit(f.strikeMessage(token,'damage',outcome));
 assert.equal(f.effects(token).length,0);assert.deepEqual(f.deleted,destroyed);assert.deepEqual(f.errors,[]);
});

test('other tokens, non-strike rolls and thralls without the effect leave it alone',async()=>{
 const f=fixture(),token=f.addThrall('a'),other=f.addThrall('b');f.player.targets.add({document:token});
 await f.emit(f.castMessage());
 await f.emit(f.strikeMessage(other,'attack','failure'));
 const spellDamage=f.strikeMessage(token,'damage','success');spellDamage.flags.pf2e.context={type:'damage-roll',sourceType:'save',domains:['spell-damage','damage']};
 await f.emit(spellDamage);
 await f.emit(f.strikeMessage(f.casterToken,'attack','failure'));
 assert.equal(f.effects(token).length,1);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.errors,[]);
});

test('a cast message and an effect are each settled once',async()=>{
 const f=fixture({answers:['destroy']}),token=f.addThrall('a'),message=f.castMessage();
 await Promise.all([f.emit(message),f.emit(message)]);
 await f.emit(message);
 assert.equal(f.prompts.length,1);assert.equal(f.effects(token).length,1);
 await Promise.all([f.emit(f.strikeMessage(token,'damage','success')),f.emit(f.strikeMessage(token,'damage','success')),f.emit(f.strikeMessage(token,'attack','failure'))]);
 assert.deepEqual(f.deleted,['a']);assert.deepEqual(f.errors,[]);
});

test('a message already carrying a result is not processed again',async()=>{
 const f=fixture(),token=f.addThrall('a'),message=f.castMessage();message.flags[MODULE_ID]={thrallCharge:{status:'cancelled',rank:2}};
 await f.emit(message);
 assert.equal(f.prompts.length,0);assert.equal(f.effects(token).length,0);
});

test('only the active GM handles casts and strikes',async()=>{
 const f=fixture(),token=f.addThrall('a');
 f.game.user=f.player;await f.emit(f.castMessage('player-client'));
 f.game.user={id:'gm2',isGM:true};await f.emit(f.castMessage('second-gm'));
 assert.equal(f.prompts.length,0);assert.equal(f.effects(token).length,0);
 f.game.user=f.gmUser;await f.emit(f.castMessage('active-gm'));assert.equal(f.effects(token).length,1);
 f.game.user=f.player;await f.emit(f.strikeMessage(token,'attack','failure'));assert.equal(f.effects(token).length,1);
});

for(const options of [{casting:false},{actualCast:false},{spellSource:'Compendium.pf2e.spells-srd.Item.other'}])test(`a spell card that is not an actual Thrall Charge cast does nothing: ${JSON.stringify(options)}`,async()=>{
 const f=fixture(options),token=f.addThrall('a');
 await f.emit(f.castMessage());
 assert.equal(f.prompts.length,0);assert.equal(f.effects(token).length,0);assert.deepEqual(f.errors,[]);assert.equal(f.state('cast'),undefined);
});

test('a cast without the caster token on the GM canvas reports why',async()=>{
 const f=fixture(),token=f.addThrall('a');
 await f.emit(f.castMessage('no-token',{speaker:{actor:'necro'}}));
 f.casterToken.object=null;
 await f.emit(f.castMessage('other-scene'));
 assert.equal(f.errors.length,2);assert.match(f.errors[0],/Token/);assert.match(f.errors[1],/场景/);
 assert.equal(f.effects(token).length,0);assert.equal(f.state('no-token').status,'failed');
});
