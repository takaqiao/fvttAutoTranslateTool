import test from 'node:test';
import assert from 'node:assert/strict';
const rules=await import('../scripts/defensive-advance-rules.mjs').catch(()=>({}));

export function fixture(){
 const actor={uuid:'Actor.a',id:'a',type:'character',canAct:true,alliance:'party',system:{movement:{speeds:{land:{value:20}}},actions:[]},getReach:()=>10};
 const enemy={uuid:'Actor.e',id:'e',type:'npc',alliance:'opposition',getReach:()=>0};
 const scene={id:'s',tokens:new Map()},token={id:'t',uuid:'Scene.s.Token.t',documentName:'Token',parent:scene,actor,x:0,y:0,elevation:0},target={id:'e',uuid:'Scene.s.Token.e',documentName:'Token',parent:scene,actor:enemy,x:100,y:0,elevation:0,getCenterPoint:()=>({x:150,y:50})};
 token.object={document:token,distanceTo:()=>10,checkCollision:()=>false};target.object={document:target};scene.tokens.set('t',token);scene.tokens.set('e',target);
 const combatant={id:'c',actor,token},combat={id:'real',started:true,round:3,turn:0,turns:[combatant]},game={scenes:new Map([['s',scene]]),combats:new Map([['real',combat]]),combat:{id:'viewed-other'}};
 const item={id:'w',uuid:'Actor.a.Item.w',actor,name:'Reach sword',isMelee:true,isRanged:false},strike={type:'strike',ready:true,item,variants:[0,1,2].map(()=>({roll:async()=>{}}))};actor.system.actions.push(strike);
 return {actor,token,target,game,combat,scene,strike};
}

test('Defensive Advance uses the actual own turn and prepared land speed despite another viewed encounter',()=>{
 assert.equal(typeof rules.defensiveAdvanceContext,'function');const f=fixture();
 assert.deepEqual(rules.defensiveAdvanceContext(f),{turn:'real:3:0',speed:20});
 f.combat.turn=1;assert.throws(()=>rules.defensiveAdvanceContext(f),/自己的回合/);
 f.combat.turn=0;f.game.combats.set('duplicate',{...f.combat,id:'duplicate'});assert.throws(()=>rules.defensiveAdvanceContext(f),/多个/);
});

test('a ready native reach weapon may attack an enemy with no reverse reach; walls or ranged usages exclude it',()=>{
 assert.equal(typeof rules.defensiveAdvanceMeleeChoices,'function');const f=fixture();
 assert.equal(rules.defensiveAdvanceMeleeChoices(f)[0]?.key,'Actor.a.Item.w#base');
 f.token.object.checkCollision=()=>true;assert.deepEqual(rules.defensiveAdvanceMeleeChoices(f),[]);
 f.token.object.checkCollision=()=>false;f.strike.item.isRanged=true;assert.deepEqual(rules.defensiveAdvanceMeleeChoices(f),[]);
});

test('only the original server movement chain and user within Stride cost can become a completion receipt',()=>{
 assert.equal(typeof rules.defensiveAdvanceMovementProof,'function');const f=fixture(),user={id:'owner'},receipt={planId:'plan',userId:'owner',origin:{x:0,y:0,elevation:0},lastPosition:{x:0,y:0,elevation:0},movementIds:[],movementCost:0,speed:20};
 f.token.x=100;
 const movement={id:'segment',chain:['plan'],origin:{x:0,y:0,elevation:0},destination:{x:100,y:0,elevation:0},passed:{cost:10,waypoints:[{action:'walk'}]},pending:{waypoints:[]},constrained:false},operation={_movement:{t:movement}};
 const args={...f,user,receipt,movement,operation},proof=rules.defensiveAdvanceMovementProof(args);
 assert.deepEqual(proof,{movementIds:['segment'],movementCost:10,lastPosition:{x:100,y:0,elevation:0}});
 assert.equal(rules.defensiveAdvanceMovementProof({...args,operation:{}}),null);
 assert.equal(rules.defensiveAdvanceMovementProof({...args,user:{id:'other'}}),null);
 assert.equal(rules.defensiveAdvanceMovementProof({...args,movement:{...movement,chain:['unrelated']}}),null);
 movement.passed.cost=25;assert.throws(()=>rules.defensiveAdvanceMovementProof(args),/步行/);
 movement.passed.cost=10;movement.passed.waypoints[0].action='teleport';assert.throws(()=>rules.defensiveAdvanceMovementProof(args),/步行/);
});

test('native moveToken precedes animated document coordinates; record its server endpoint for finished verification',()=>{
 const f=fixture(),user={id:'owner'},receipt={planId:'plan',userId:'owner',lastPosition:{x:0,y:0,elevation:0},movementIds:[],movementCost:0,speed:20};
 const movement={id:'plan',chain:[],origin:{x:0,y:0,elevation:0},destination:{x:100,y:0,elevation:0},passed:{cost:10,waypoints:[{action:'walk'}]},constrained:false},operation={_movement:{t:movement}};
 assert.equal(f.token.x,0);
 assert.deepEqual(rules.defensiveAdvanceMovementProof({...f,user,receipt,movement,operation}),{movementIds:['plan'],movementCost:10,lastPosition:{x:100,y:0,elevation:0}});
 movement.destination.x=NaN;
 assert.throws(()=>rules.defensiveAdvanceMovementProof({...f,user,receipt,movement,operation}),/步行/);
});
