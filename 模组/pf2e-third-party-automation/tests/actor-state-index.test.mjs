import test from 'node:test';
import assert from 'node:assert/strict';
let createActorStateIndex;try{({createActorStateIndex}=await import('../scripts/actor-state-index.mjs'))}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
function fixture(){
 const active={id:'a',uuid:'Actor.a',active:true},idle={id:'b',uuid:'Actor.b'},synthetic={id:'c',uuid:'Scene.s.Token.c.Actor.c',active:true};
 const tokens=new Map([['linked',{actor:active}],['c',{actor:synthetic}]]),scene={id:'s',tokens},game={actors:new Map([['a',active],['b',idle]]),scenes:new Map([['s',scene]])};
 const hooks=new Map();return {game,active,idle,synthetic,scene,hooks,Hooks:{on:(key,fn)=>hooks.set(key,fn)}};
}
test('active actor index seeds once including synthetic actors and never rescans on repeated reads',()=>{
 assert.equal(typeof createActorStateIndex,'function');const f=fixture(),index=createActorStateIndex({game:f.game,matches:a=>a.active});index.register(f.Hooks);
 assert.deepEqual(index.values(),[f.active,f.synthetic]);
 f.game.actors.values=()=>{throw Error('world traversal')};f.game.scenes.values=()=>{throw Error('scene traversal')};
 assert.deepEqual(index.values(),[f.active,f.synthetic]);f.idle.active=true;f.hooks.get('updateActor')(f.idle,{});
 assert.deepEqual(index.values(),[f.active,f.synthetic,f.idle]);f.active.active=false;f.hooks.get('updateItem')({actor:f.active},{});
 assert.deepEqual(index.values(),[f.synthetic,f.idle]);
});
test('token movement does not inspect actor state, while deletion and actor reassignment update membership',()=>{
 assert.equal(typeof createActorStateIndex,'function');const f=fixture();let reads=0;const index=createActorStateIndex({game:f.game,matches:a=>{reads++;return a.active}});index.register(f.Hooks);reads=0;
 f.hooks.get('updateToken')({actor:f.synthetic},{x:100,y:100});assert.equal(reads,0);
 f.hooks.get('deleteToken')({actor:f.synthetic});assert.deepEqual(index.values(),[f.active]);
 f.hooks.get('deleteToken')({actor:f.active});assert.deepEqual(index.values(),[f.active],'linked world actor remains');
 f.hooks.get('updateToken')({actor:f.synthetic},{actorId:'c'});assert.deepEqual(index.values(),[f.active,f.synthetic]);
});
test('base actor document changes refresh its existing unlinked synthetic dependents',()=>{
 const f=fixture();f.active.active=false;f.synthetic.active=false;f.active.getDependentTokens=options=>{assert.equal(options.concreteOnly,true);return [{actor:f.active},{actor:f.synthetic}]};
 const index=createActorStateIndex({game:f.game,matches:a=>a.active});index.register(f.Hooks);assert.deepEqual(index.values(),[]);
 f.game.scenes.values=()=>{throw Error('world token traversal')};f.active.active=f.synthetic.active=true;f.hooks.get('createItem')({actor:f.active});
 assert.deepEqual(index.values(),[f.active,f.synthetic]);f.active.active=f.synthetic.active=false;f.hooks.get('updateActor')(f.active,{});assert.deepEqual(index.values(),[]);
});
