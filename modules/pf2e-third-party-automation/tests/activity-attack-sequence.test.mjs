import test from 'node:test';
import assert from 'node:assert/strict';
let createAttackSequence;
try{({createAttackSequence}=await import('../scripts/activity-attack-sequence.mjs'))}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}

function fixture(){
 const clones=[],actor={uuid:'Actor.hero',_source:{items:[]},system:{actions:[]},clone(changes){clones.push(changes);return this}};
 const weapon=(id,traits=['forceful','backswing','sweep'],baseItem='sword')=>{
  const strike={item:{id,uuid:`${actor.uuid}.Item.${id}`,actor,type:'weapon',isMelee:true,system:{traits:{value:traits},baseItem,damage:{dice:2}}}};
  actor.system.actions.push(strike);return strike;
 };
 const a=weapon('a'),b=weapon('b'),target=uuid=>({uuid});
 assert.equal(typeof createAttackSequence,'function');
 return {actor,clones,weapon,a,b,target,sequence:createAttackSequence({actor})};
}
const rules=f=>f.clones.at(-1)?.items.at(-1).system.rules??[];
test('unresolved checks and a new activity leave native options and damage alone',()=>{
 const f=fixture(),frame=f.sequence.begin(f.a,f.target('Token.one'));
 assert.deepEqual([...frame.attackOptions],[]);assert.equal(frame.damage(f.a).strike,f.a);
 for(const outcome of [undefined,null,'unknown','cancelled'])assert.equal(f.sequence.record(frame,outcome),false);
 const next=f.sequence.begin(f.a,f.target('Token.two'));assert.deepEqual([...next.attackOptions],[]);assert.equal(f.clones.length,0);
 f.sequence.record(frame,'failure');
 assert.deepEqual([...createAttackSequence({actor:f.actor}).begin(f.a,f.target('Token.two')).attackOptions],[]);
});
for(const outcome of ['failure','criticalFailure','success','criticalSuccess'])test(`only a completed prior miss grants Backswing (${outcome})`,()=>{
 const f=fixture();f.sequence.record(f.sequence.begin(f.a,f.target('Token.one')),outcome);
 const next=f.sequence.begin(f.a,f.target('Token.two'));
 assert.equal(next.attackOptions.has('backswing-bonus'),outcome.endsWith('Failure')||outcome==='failure');
 assert(next.attackOptions.has('sweep-bonus'));
});
test('Sweep compares Token identity, including two Tokens backed by the same Actor',()=>{
 const f=fixture(),one={uuid:'Scene.s.Token.one',actor:{uuid:'Actor.same'}},two={uuid:'Scene.s.Token.two',actor:one.actor};
 f.sequence.record(f.sequence.begin(f.a,one),'success');
 assert.equal(f.sequence.begin(f.a,one).attackOptions.has('sweep-bonus'),false);
 assert.equal(f.sequence.begin(f.a,two).attackOptions.has('sweep-bonus'),true);
 assert.equal(f.sequence.begin(f.a,{}).attackOptions.has('sweep-bonus'),false);
});
test('same-weapon history survives an intervening other weapon without leaking to that weapon',()=>{
 const f=fixture();f.sequence.record(f.sequence.begin(f.a,f.target('Token.one')),'failure');
 const other=f.sequence.begin(f.b,f.target('Token.two'));assert.deepEqual([...other.attackOptions],[]);f.sequence.record(other,'success');
 assert(f.sequence.begin(f.a,f.target('Token.one')).attackOptions.has('backswing-bonus'));
});
test('the latest same-weapon result replaces an older miss for Backswing',()=>{
 const f=fixture();for(const outcome of ['failure','success'])f.sequence.record(f.sequence.begin(f.a,f.target('Token.one')),outcome);
 assert.equal(f.sequence.begin(f.a,f.target('Token.two')).attackOptions.has('backswing-bonus'),false);
});
test('deferred damage uses each frames preceding facts and duplicate records do not increase Forceful',()=>{
 const f=fixture(),first=f.sequence.begin(f.a,f.target('Token.one'));
 assert(f.sequence.record(first,'failure'));assert.equal(f.sequence.record(first,'success'),false);
 const second=f.sequence.begin(f.a,f.target('Token.two'));f.sequence.record(second,'success');
 const third=f.sequence.begin(f.a,f.target('Token.three'));f.sequence.record(third,'criticalSuccess');
 first.damage(f.a);assert.equal(f.clones.length,0);
 second.damage(f.a);assert.equal(rules(f)[0].value,'@weapon.system.damage.dice');
 third.damage(f.a);assert.equal(rules(f)[0].value,'2 * @weapon.system.damage.dice');
 assert.equal(rules(f)[0].type,'circumstance');assert.equal(rules(f)[0].selector,'a-damage');
 assert.deepEqual(f.actor._source.items,[]);
});
test('a foreign actors weapon with an equal item ID cannot become sequence history',()=>{
 const f=fixture(),foreign={item:{...f.a.item,actor:{uuid:'Actor.other'}}};
 f.sequence.record(f.sequence.begin(foreign,f.target('Token.one')),'failure');
 const next=f.sequence.begin(f.a,f.target('Token.two'));assert.deepEqual([...next.attackOptions],[]);f.sequence.record(next,'success');next.damage(f.a);assert.equal(f.clones.length,0);
});
test('a frame cannot be recorded into another activity',()=>{
 const f=fixture(),frame=f.sequence.begin(f.a,f.target('Token.one')),other=createAttackSequence({actor:f.actor});
 assert.equal(other.record(frame,'failure'),false);assert.deepEqual([...other.begin(f.a,f.target('Token.two')).attackOptions],[]);
});
test('Twin uses frozen weapon facts even if the first item is later re-prepared',()=>{
 const f=fixture();for(const s of [f.a,f.b])s.item.system.traits.value=['twin'];
 const first=f.sequence.begin(f.a,f.target('Token.one'));f.sequence.record(first,'failure');f.a.item.system.baseItem='axe';
 const second=f.sequence.begin(f.b,f.target('Token.one'));f.sequence.record(second,'success');second.damage(f.b);
 assert.equal(rules(f).length,1);assert(rules(f)[0].predicate.includes('item:trait:twin'));
});
test('damage retains alternate usage and the infused actors item data',()=>{
 const f=fixture();f.sequence.record(f.sequence.begin(f.a,f.target('Token.one')),'failure');
 const infused={...f.actor,_source:{items:[{name:'existing magical infusion'}]},clone(changes){f.clones.push(changes);return {system:{actions:[{item:{id:'a'},altUsages:[alternate]}]}}}},alternate={item:{...f.a.item,actor:infused,altUsageType:'melee'}};
 const frame=f.sequence.begin(alternate,f.target('Token.two'));f.sequence.record(frame,'success');
 assert.equal(frame.damage(alternate).strike,alternate);assert.equal(f.clones[0].items[0].name,'existing magical infusion');
 const wrong={item:{...f.b.item,actor:infused}};assert.equal(frame.damage(wrong).strike,wrong);assert.equal(f.clones.length,1);
});
