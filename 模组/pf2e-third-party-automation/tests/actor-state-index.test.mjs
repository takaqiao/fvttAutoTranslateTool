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
 assert.equal(Object.hasOwn(index,'initialize'),false);assert.equal(Object.hasOwn(index,'observe'),false);
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
test('token update without changed fields leaves membership intact without inspecting actors',()=>{
 const f=fixture();let reads=0;const index=createActorStateIndex({game:f.game,matches:actor=>{reads++;return actor.active}});index.register(f.Hooks);reads=0;
 assert.doesNotThrow(()=>f.hooks.get('updateToken')({actor:f.synthetic}));
 assert.equal(reads,0);assert.deepEqual(index.values(),[f.active,f.synthetic]);
});

test('invalidating membership recovers only current world and synthetic actors once',()=>{
 const f=fixture(),index=createActorStateIndex({game:f.game,matches:actor=>actor.active});index.register(f.Hooks);assert.deepEqual(index.values(),[f.active,f.synthetic]);
 assert.equal(typeof index.invalidate,'function');
 const replacement={id:'a',uuid:'Actor.a',active:true},synthetic={id:'d',uuid:'Scene.s.Token.d.Actor.d',active:true};f.game.actors.set('a',replacement);f.scene.tokens.clear();f.scene.tokens.set('d',{actor:synthetic});f.idle.active=true;
 let scans=0;for(const collection of [f.game.actors,f.game.scenes]){const original=collection.values.bind(collection);collection.values=()=>{scans++;return original()}}
 index.invalidate();assert.equal(scans,0);assert.deepEqual(index.values(),[replacement,f.idle,synthetic]);assert.equal(scans,2);
 assert.deepEqual(index.values(),[replacement,f.idle,synthetic]);assert.equal(scans,2);f.idle.active=false;f.hooks.get('updateActor')(f.idle,{});assert.deepEqual(index.values(),[replacement,synthetic]);
});

test('explicit recovery uses its evaluator once and reports whether it seeded membership',()=>{
 const f=fixture(),evaluated=[],index=createActorStateIndex({game:f.game,matches:actor=>actor.active});index.register(f.Hooks,{recover:false});
 f.game.actors.set('missing',{active:true});
 assert.equal(typeof index.recover,'function');
 assert.equal(index.recover(actor=>{evaluated.push(actor);return actor!==f.active}),true);
 assert.deepEqual(evaluated,[f.active,f.idle,f.active,f.synthetic]);assert.deepEqual(index.values(),[f.idle,f.synthetic]);
 assert.equal(index.recover(()=>assert.fail('hot recovery must not evaluate')),false);
 index.invalidate();assert.equal(index.recover(),true);assert.deepEqual(index.values(),[f.active,f.synthetic]);
});

test('createScene with full Hook options uses the default membership predicate',()=>{
 const f=fixture(),index=createActorStateIndex({game:f.game,matches:actor=>actor.active});index.register(f.Hooks,{recover:false});
 const active={uuid:'Scene.imported.Token.active.Actor.active',active:true},idle={uuid:'Scene.imported.Token.idle.Actor.idle'},document={id:'imported',tokens:new Map([['active',{actor:active}],['idle',{actor:idle}]])};
 f.hooks.get('createScene')(document,{render:true,keepId:true},'gm');
 assert.deepEqual(index.values(),[active,f.active,f.synthetic]);
});

test('observed recovery initializes identities without reading membership',()=>{
 const f=fixture();let reads=0;
 const index=createActorStateIndex({game:f.game,matches:()=>{reads++;throw Error('unexpected membership')},observedRecovery:true});
 index.register(f.Hooks,{recover:false});
 assert.equal(typeof index.initialize,'function');assert.equal(typeof index.observe,'function');
 assert.throws(()=>index.values(),/Observed actor index requires initialization/);
 assert.throws(()=>index.recover(),/Observed actor index requires initialization/);
 const replacement={...f.active};
 assert.equal(index.initialize([f.active,f.idle,replacement,f.synthetic]),true);
 assert.deepEqual(index.values(),[replacement,f.idle,f.synthetic]);
 assert.equal(index.recover(),false);assert.equal(reads,0);
 assert.equal(index.initialize([]),false);
});
test('observed recovery removes only the initialized current object on false',()=>{
 const f=fixture(),index=createActorStateIndex({game:f.game,matches:a=>a.active,observedRecovery:true});
 assert.equal(typeof index.initialize,'function');assert.equal(typeof index.observe,'function');
 index.initialize([f.active,f.idle]);
 const replacement={...f.active};index.refresh(replacement);
 assert.equal(index.observe(f.active,false),false);
 for(const value of [true,null,undefined,0])assert.equal(index.observe(replacement,value),false);
 assert.deepEqual(index.values(),[replacement,f.idle]);
 assert.equal(index.observe(replacement,false),true);
 assert.deepEqual(index.values(),[f.idle]);
 index.invalidate();assert.equal(index.observe(f.idle,false),false);
 assert.throws(()=>index.values(),/Observed actor index requires initialization/);
});
test('observed initialization failure leaves no partial cohort',()=>{
 const f=fixture(),error=Error('identity failed'),failing={get uuid(){throw error}};
 const index=createActorStateIndex({game:f.game,matches:a=>a.active,observedRecovery:true});
 assert.equal(typeof index.initialize,'function');
 assert.throws(()=>index.initialize([f.active,failing]),e=>e===error);
 assert.throws(()=>index.values(),/Observed actor index requires initialization/);
 assert.equal(index.initialize([f.idle]),true);
 assert.deepEqual(index.values(),[f.idle]);
});
