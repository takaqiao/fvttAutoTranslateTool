import {test} from 'node:test';
import assert from 'node:assert/strict';

import {assessRoaringCast,validateRoaringTarget,roaringOwnTurn} from '../scripts/roaring-applause-rules.mjs';

function fixture(){
 const user={id:'user',active:true},actor={id:'caster',uuid:'Actor.caster',type:'character',canAct:true,isDead:false,items:new Map(),testUserPermission:u=>u===user};
 const entry={id:'entry',uuid:'Actor.caster.Item.entry',type:'spellcastingEntry',actor,isSpontaneous:true,system:{prepared:{value:'spontaneous'},tradition:{value:'occult'},slots:{slot3:{value:1,max:2}}}};
 const item={id:'spell',uuid:'Actor.caster.Item.spell',type:'spell',rank:3,sourceId:'Compendium.pf2e.spells-srd.Item.czO0wbT1i320gcu9',actor,spellcasting:entry,flags:{},system:{location:{value:'entry',signature:true},level:{value:3},rules:[],overlays:{},traits:{value:['concentrate','emotion','manipulate','mental']},time:{value:'2'},range:{value:'60 feet'},area:null,duration:{value:'本地化文本',sustained:true},damage:{},defense:{save:{basic:false,statistic:'will'}},heightening:{type:'fixed',levels:{6:{target:{value:'10 creatures'}}}}}};
 actor.items.set(item.id,item);actor.items.set(entry.id,entry);
 const users=new Map([[user.id,user]]);users.activeGM={id:'gm',active:true};
 const scene={id:'scene',uuid:'Scene.scene',grid:{type:1,units:'ft',distance:5},tokens:new Map()};
 const token={id:'caster',uuid:'Scene.scene.Token.caster',documentName:'Token',parent:scene,actor,hidden:false,elevation:0,object:{center:{x:0,y:0},distanceTo:()=>60}};
 const target={id:'target',uuid:'Scene.scene.Token.target',documentName:'Token',parent:scene,hidden:false,elevation:0,actor:{id:'target',uuid:'Actor.target',type:'npc',isDead:false,hasCondition:()=>false},object:{center:{x:500,y:0}}};
 scene.tokens.set(token.id,token);scene.tokens.set(target.id,target);
 const sourceCombatant={id:'source-turn',actor,token,initiative:20,flags:{pf2e:{roundOfLastTurnEnd:3}}};
 const enemyCombatant={id:'enemy-turn',actor:target.actor,token:target,initiative:20,flags:{}};
 const combat={id:'actual',started:true,scene,round:4,turn:0,turns:[sourceCombatant,enemyCombatant]};
 const game={world:{id:'ujx5r8oipw7ercdr'},system:{version:'8.5.1'},user,users,actors:new Map([[actor.id,actor]]),scenes:new Map([[scene.id,scene]]),combats:new Map([[combat.id,combat]]),settings:{get:()=> 'public'}};
 return {game,user,actor,item,entry,options:{rank:3},token,targets:[target],perception:'sees',lineOfEffectConfirmed:true,combat,sourceCombatant};
}

 test('current original and native same-rank variant pass without English duration/target labels',()=>{
  const f=fixture();assert.equal(assessRoaringCast(f).eligible,true);
  const variant={...f.item,original:f.item,system:{...f.item.system,location:{...f.item.system.location,heightenedLevel:3}}};
  assert.equal(assessRoaringCast({...f,item:variant}).eligible,true);
  assert.equal(validateRoaringTarget(f),f.targets[0]);
 });
 test('unrelated worlds/items, NPC and preview calls stay outside enrollment',()=>{
  for(const mutate of [f=>f.game.world.id='other',f=>f.item.sourceId='same-name-copy',f=>f.actor.type='npc',f=>f.actor.isToken=true,f=>f.options.consume=false,f=>f.options.message=false]){
   const f=fixture();mutate(f);assert.equal(assessRoaringCast(f).handled,false);
  }
 });
 test('changed source, owner, native slot, private/overlaid and future-rank entry cannot be admitted',()=>{
  for(const mutate of [f=>f.actor.items.set('spell',{...f.item}),f=>f.item.spellcasting={},f=>f.actor.canAct=false,f=>f.actor.isDead=true,f=>f.user.active=false,f=>f.game.users.activeGM=null,f=>f.entry.system.tradition.value='arcane',f=>f.entry.system.slots.slot3.value=0,f=>f.options.rank=6,f=>f.item.system.location.signature=false,f=>f.item.system.rules.push({key:'ActiveEffectLike'}),f=>f.item.system.defense.save.basic=true,f=>f.item.system.duration.sustained=false,f=>f.item.system.traits.value.push('auditory'),f=>f.item.appliedOverlays=new Set(['other']),f=>f.options.messageMode='blind',f=>f.item.flags['pf2e-toolbelt']={actionable:{linked:'Macro.custom'}}]){
   const f=fixture();mutate(f);assert.equal(assessRoaringCast(f).eligible,false);
  }
 });
 test('understanding is see OR hear OR otherwise understand, independent of caster blindness',()=>{
  for(const perception of ['sees','hears','understands']){
   const f=fixture();f.perception=perception;f.actor.canSee=false;f.actor.hasCondition=()=>true;
   assert.equal(validateRoaringTarget(f),f.targets[0]);
  }
  const f=fixture();f.perception='hears';f.targets[0].actor.hasCondition=c=>c==='blinded';assert.equal(validateRoaringTarget(f),f.targets[0]);
  f.perception='sees';assert.throws(()=>validateRoaringTarget(f));
  f.perception='understands';assert.equal(validateRoaringTarget(f),f.targets[0]);
 });
 test('target validation preserves single live creature, range, geometry and explicit line-of-effect boundaries',()=>{
  for(const mutate of [f=>f.targets=[],f=>f.targets.push(f.targets[0]),f=>f.token.object.distanceTo=()=>61,f=>f.token.object.distanceTo=()=>NaN,f=>f.targets[0].hidden=true,f=>f.targets[0].actor.type='loot',f=>f.targets[0].elevation=5,f=>f.targets[0].parent={...f.token.parent},f=>f.token.parent.tokens.delete('target'),f=>f.token.parent.grid.type=2,f=>f.lineOfEffectConfirmed=false,f=>f.perception=null,f=>{f.perception='hears';f.targets[0].actor.hasCondition=c=>c==='deafened';}]){
   const f=fixture();mutate(f);assert.throws(()=>validateRoaringTarget(f));
  }
 });
 test('actual own turn is frozen from token encounter, never viewed combat or initiative equality',()=>{
  const f=fixture();Object.defineProperty(f.game,'combat',{get(){throw Error('viewed combat must not be read');}});
  const frame=roaringOwnTurn(f);
  assert.deepEqual(frame,{combatId:'actual',combatantId:'source-turn',actorUuid:f.actor.uuid,tokenUuid:f.token.uuid,started:true,round:4,turn:0,order:[{id:'source-turn',initiative:20,overridePriority:null},{id:'enemy-turn',initiative:20,overridePriority:null}],lastTurnEnd:3});
  f.combat.turn=1;assert.throws(()=>roaringOwnTurn(f));
 });
 test('ambiguous encounters and already-ended or malformed own turns require manual timing',()=>{
  for(const mutate of [f=>f.game.combats.set('other',{...f.combat,id:'other'}),f=>f.combat.started=false,f=>f.sourceCombatant.flags.pf2e.roundOfLastTurnEnd=4,f=>f.sourceCombatant.initiative=null,f=>f.combat.turn=1,f=>f.combat.round=0,f=>f.combat.turns.push({...f.sourceCombatant,id:'duplicate'})]){
   const f=fixture();mutate(f);assert.throws(()=>roaringOwnTurn(f));
  }
 });
