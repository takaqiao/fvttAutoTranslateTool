import test from 'node:test';
import assert from 'node:assert/strict';
import {defensiveAdvanceContext,defensiveAdvanceMeleeChoices} from '../scripts/defensive-advance-rules.mjs';

function fixture(){
 const actor={uuid:'Actor.a',id:'a',type:'character',canAct:true,alliance:'party',system:{actions:[]}};
 const enemy={uuid:'Actor.e',id:'e',type:'npc',alliance:'opposition'};
 const scene={id:'s',tokens:new Map()},token={id:'t',uuid:'Scene.s.Token.t',documentName:'Token',parent:scene,actor,x:0,y:0,elevation:0},target={id:'e',uuid:'Scene.s.Token.e',documentName:'Token',parent:scene,actor:enemy,x:90000,y:0,elevation:100};
 token.object={document:token};target.object={document:target};scene.tokens.set('t',token);scene.tokens.set('e',target);
 const combatant={id:'c',actor,token},combat={id:'real',started:true,round:3,turn:0,turns:[combatant]},game={scenes:new Map([['s',scene]]),combats:new Map([['real',combat]]),combat:{id:'viewed-other'}};
 const item={id:'w',uuid:'Actor.a.Item.w',actor,name:'Sword',isMelee:true,isRanged:false},strike={type:'strike',ready:true,item,variants:[0,1,2].map(()=>({roll:async()=>{}}))};actor.system.actions.push(strike);
 return {actor,token,target,game,combat,scene,strike};
}

test('Defensive Advance binds the actual own turn without requiring prepared land speed',()=>{
 const f=fixture();f.actor.hasCondition=name=>['immobilized','restrained'].includes(name);
 assert.deepEqual(defensiveAdvanceContext(f),{turn:'real:3:0'});
 f.combat.turn=1;assert.throws(()=>defensiveAdvanceContext(f),/自己的回合/);
 f.combat.turn=0;f.game.combats.set('duplicate',{...f.combat,id:'duplicate'});assert.throws(()=>defensiveAdvanceContext(f),/多个/);
});

test('native melee choices retain ready weapon identity while geometry remains GM-adjudicated',()=>{
 const f=fixture();f.token.object.checkCollision=()=>assert.fail('walls must not filter choices');f.token.object.distanceTo=()=>assert.fail('distance must not filter choices');f.actor.getReach=()=>assert.fail('reach must not filter choices');
 assert.equal(defensiveAdvanceMeleeChoices(f)[0]?.key,'Actor.a.Item.w#base');
 f.strike.item.isRanged=true;assert.deepEqual(defensiveAdvanceMeleeChoices(f),[]);
 f.strike.item.isRanged=false;f.strike.ready=false;assert.deepEqual(defensiveAdvanceMeleeChoices(f),[]);
});

test('melee choices cannot borrow a different actor weapon or stale target Token',()=>{
 const f=fixture();f.strike.item.actor={uuid:'Actor.other'};assert.deepEqual(defensiveAdvanceMeleeChoices(f),[]);
 f.strike.item.actor=f.actor;f.scene.tokens.set('e',{...f.target});assert.deepEqual(defensiveAdvanceMeleeChoices(f),[]);
});

test('alternate native melee usage stays selectable without its ranged base usage',()=>{
 const f=fixture(),alternate={...f.strike,item:{...f.strike.item,altUsageType:'melee'}};f.strike.item.isMelee=false;f.strike.item.isRanged=true;f.strike.altUsages=[alternate];
 assert.deepEqual(defensiveAdvanceMeleeChoices(f).map(o=>o.key),['Actor.a.Item.w#melee']);
});
