import {test} from 'node:test';
import assert from 'node:assert/strict';
import {FORCE_BARRAGE_SOURCE,assessForceBarrageCast,validateForceBarrageTargets,validateForceBarrageAllocation} from '../scripts/force-barrage-rules.mjs';

function fixture(){
 const user={id:'u',active:true},actor={id:'a',uuid:'Actor.a',type:'character',canAct:true,isDead:false,items:new Map(),testUserPermission:u=>u===user};
 const entry={id:'e',uuid:'Actor.a.Item.e',type:'spellcastingEntry',actor,isSpontaneous:true,system:{prepared:{value:'spontaneous'},tradition:{value:'occult'},slots:{slot1:{value:2,max:3},slot2:{value:2,max:3},slot3:{value:1,max:2}}}};
 const item={id:'s',uuid:'Actor.a.Item.s',type:'spell',sourceId:FORCE_BARRAGE_SOURCE,actor,spellcasting:entry,flags:{},system:{location:{value:'e',signature:true},level:{value:1},rules:[],overlays:{},traits:{value:['concentrate','force','manipulate']},time:{value:'1 to 3'},range:{value:'120 feet'},area:null,duration:{value:'',sustained:false},damage:{0:{applyMod:false,category:null,formula:'1d4+1',kinds:['damage'],materials:[],type:'force'}}}};
 actor.items.set('s',item);actor.items.set('e',entry);
 const users=new Map([['u',user]]);users.activeGM={id:'gm',active:true};
 const scene={id:'sc',uuid:'Scene.sc',grid:{type:1,units:'ft',distance:5},tokens:new Map()};
 const token={id:'src',uuid:'Scene.sc.Token.src',documentName:'Token',parent:scene,actor,hidden:false,object:{center:{x:0,y:0},distanceTo:()=>120,checkCollision:()=>false}};scene.tokens.set(token.id,token);
 const target={id:'tar',uuid:'Scene.sc.Token.tar',documentName:'Token',parent:scene,actor:{type:'npc',isDead:false,hasCondition:()=>false},hidden:false,object:{center:{x:500,y:0}}};scene.tokens.set(target.id,target);
 const game={world:{id:'ujx5r8oipw7ercdr'},system:{version:'8.5.1'},users,user,actors:new Map([['a',actor]]),scenes:new Map([['sc',scene]])};
 game.settings={get:()=> 'public'};token.elevation=0;target.elevation=0;
 return {game,actor,item,entry,user,options:{rank:3,slotId:NaN},token,targets:[target],visibilityConfirmed:true};
}
test('current spontaneous original Cast and actual native heightened variant admitted without English target name',()=>{
 const f=fixture();assert.equal(assessForceBarrageCast(f).eligible,true);
 const variant=Object.assign(Object.create(Object.getPrototypeOf(f.item)),f.item,{original:f.item,system:{...f.item.system,location:{...f.item.system.location,heightenedLevel:3}}});
 assert.equal(assessForceBarrageCast({...f,item:variant}).eligible,true);
 assert.equal(validateForceBarrageTargets(f).length,1);
});
test('unrelated or preview calls stay outside the exact invocation adapter',()=>{
 for(const mutate of [f=>f.item.sourceId='same-name-copy',f=>f.options.message=false,f=>f.options.consume=false,f=>f.game.world.id='cotct']){
  const f=fixture();mutate(f);assert.equal(assessForceBarrageCast(f).handled,false);
 }
});
test('owned/source/slot/custom/overlay/private uncertainty rejects before payment',()=>{
 const changes=[f=>f.actor.items.set('s',{...f.item}),f=>f.actor.canAct=false,f=>f.user.active=false,f=>f.game.users.activeGM=null,f=>f.entry.isSpontaneous=false,f=>f.entry.system.slots.slot3.value=0,f=>f.options.rank=4,f=>f.item.system.location.signature=false,f=>f.item.system.damage[0].formula='2d4+1',f=>f.item.system.rules.push({key:'DamageDice'}),f=>f.item.appliedOverlays=new Set(['x']),f=>f.item.flags['pf2e-toolbelt']={actionable:{linked:'Macro.x'}},f=>f.options.messageMode='gm'];
 for(const mutate of changes){const f=fixture();mutate(f);assert.equal(assessForceBarrageCast(f).eligible,false);}
});
test('range/scene/creature/vision and original visible source are checked independently of GM sight',()=>{
 for(const mutate of [f=>f.token.object.distanceTo=()=>121,f=>f.token.object.distanceTo=()=>NaN,f=>f.token.object.checkCollision=()=>true,f=>f.targets[0].hidden=true,f=>f.targets[0].actor.type='loot',f=>f.targets[0].actor.isDead=true,f=>f.targets[0].actor.hasCondition=()=>true,f=>f.targets[0].parent={...f.token.parent},f=>f.token.parent.grid.type=2,f=>f.token.parent.grid.units='m',f=>f.token.parent.tokens.delete('tar'),f=>f.visibilityConfirmed=false,f=>f.actor.hasCondition=condition=>condition==='blinded']){
  const f=fixture();mutate(f);assert.throws(()=>validateForceBarrageTargets(f));
 }
});

test('confirmed sight of a locally lit target is not vetoed by scene-wide canSee',()=>{
 const f=fixture();f.actor.canSee=false;f.actor.hasCondition=()=>false;
 assert.deepEqual(validateForceBarrageTargets(f),f.targets);
 f.visibilityConfirmed=false;assert.throws(()=>validateForceBarrageTargets(f));
 f.visibilityConfirmed=true;f.actor.hasCondition=condition=>condition==='blinded';assert.throws(()=>validateForceBarrageTargets(f));
});
test('allocation takes upstream missile count and refuses any unsafe numeric or target input',()=>{
 const targets=[{uuid:'Scene.s.Token.a'},{uuid:'Scene.s.Token.b'}];
 const good=[{targetUuid:targets[0].uuid,count:0},{targetUuid:targets[1].uuid,count:6}];
 assert.deepEqual(validateForceBarrageAllocation({targets,allocations:good,missiles:6}),good);
 for(const count of [NaN,Infinity,-1,1.5,'6',Number.MAX_SAFE_INTEGER+1])assert.throws(()=>validateForceBarrageAllocation({targets,allocations:[good[0],{...good[1],count}],missiles:6}));
 for(const allocations of [[good[1]],[good[0],good[0]],[good[0],{targetUuid:'Scene.s.Token.c',count:6}],[{...good[0],count:0},{...good[1],count:0}]])assert.throws(()=>validateForceBarrageAllocation({targets,allocations,missiles:6}));
});
test('Core14 implicit private defaults and differing elevations stop before payment',()=>{
 for(const mode of ['gm','blind','self',undefined]){const f=fixture();f.game.settings.get=()=>mode;assert.equal(assessForceBarrageCast(f).eligible,false);}
 const f=fixture();f.targets[0].elevation=5;assert.throws(()=>validateForceBarrageTargets(f),/高度/);
});
test('NPC and synthetic-token Cast routes are not enrolled; current bridge requires the audited occult entry',()=>{
 for(const change of [f=>f.actor.type='npc',f=>f.actor.isToken=true]){const f=fixture();change(f);assert.equal(assessForceBarrageCast(f).handled,false);}
 const f=fixture();f.entry.system.tradition={value:'arcane'};assert.equal(assessForceBarrageCast(f).eligible,false);
});
