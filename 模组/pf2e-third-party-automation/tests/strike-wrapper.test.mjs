import test from 'node:test';
import assert from 'node:assert/strict';
import {registerStrikeWrapper} from '../scripts/strike-wrapper.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

const strikePath='CONFIG.PF2E.Actor.documentClasses.character.prototype.prepareStrike';
function fixture({wrapStrike}={}){
 const calls=[],context={type:'attack-roll'},messages=new Map([['other',{id:'other',isCheckRoll:true,flags:{pf2e:{context:{...context}}}}],['damage',{id:'damage',isCheckRoll:false,flags:{pf2e:{context}}}],['attack',{id:'attack',isCheckRoll:true,flags:{pf2e:{context}}}]]);
 let handler;const unregistered=[],libWrapper={register(id,path,fn,type){assert.deepEqual([id,path,type],[ID,strikePath,'WRAPPER']);handler=fn},unregister(id,path){unregistered.push([id,path])}};
 const dispose=registerStrikeWrapper({game:{messages},libWrapper,wrapStrike}),actor={id:'actor'};
 const prepare=strike=>handler.call(actor,()=>strike);
 const native=method=>async function(params){calls.push({method,self:this,options:[...params.options],params});return method};
 return {prepare,native,calls,context,actor,dispose,unregistered};
}

test('native Strike damage carries the attack card that owns the exact check context',async()=>{
 const f=fixture(),strike={damage:f.native('damage'),critical:f.native('critical')};assert.equal(f.prepare(strike),strike);
 assert.equal(await strike.damage({checkContext:f.context,options:['kept'],mapIncreases:1}),'damage');assert.equal(await strike.critical({checkContext:f.context}),'critical');
 assert.deepEqual(f.calls.map(call=>call.options),[['kept',`${ID}:bear-attack:attack`],[`${ID}:bear-attack:attack`]]);assert.equal(f.calls[0].self,strike);assert.equal(f.calls[0].params.mapIncreases,1);
 // An equal copy of the context, or no context at all, identifies no attack card.
 await strike.damage({checkContext:{...f.context},options:new Set(['kept'])});await strike.damage();assert.deepEqual(f.calls.slice(2).map(call=>call.options),[['kept'],[]]);
});

test('provider rewrites follow the tagging and the attack aliases track the first variant',()=>{
 const seen=[],first=()=>'native',replaced=()=>'provider',unrelated=()=>'other';
 const f=fixture({wrapStrike:(strike,actor)=>{seen.push([typeof strike.damage,actor]);strike.variants[0].roll=replaced;return strike}});
 const strike={variants:[{roll:first},{roll:unrelated}],attack:first,roll:unrelated,damage:f.native('damage')},native=strike.damage,result=f.prepare(strike);
 assert.deepEqual(seen,[['function',f.actor]]);assert.notEqual(strike.damage,native);assert.equal(result.attack,replaced);assert.equal(result.roll,unrelated);
 // A Strike without damage methods or variants passes through untouched.
 const bare={};assert.equal(fixture().prepare(bare),bare);assert.deepEqual(bare,{});
});

test('the wrapper is registered only through libWrapper and released on the same path',()=>{
 assert.doesNotThrow(()=>registerStrikeWrapper({game:{messages:new Map()}})());
 const f=fixture();f.dispose();assert.deepEqual(f.unregistered,[[ID,strikePath]]);
});
