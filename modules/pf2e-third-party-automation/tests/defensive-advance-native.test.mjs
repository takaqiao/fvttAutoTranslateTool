import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID} from '../scripts/rules.mjs';
const native=await import('../scripts/defensive-advance-native.mjs').catch(()=>({}));

test('native Stride binds the plan before starting and waits for its original finished promise',async()=>{
 assert.equal(typeof native.runDefensiveAdvanceMovement,'function');
 const user={id:'owner'},game={user},calls=[],plan={id:'p'},token={id:'t',uuid:'Scene.s.Token.t',x:0,y:0,elevation:0,parent:{tokens:new Map()},object:{async planMovement(options){calls.push(['plan',options]);return plan}},movement:{id:'p',state:'planned',user,finished:Promise.resolve(true),animation:{ended:Promise.resolve()}},async startMovement(id){calls.push(['start',id]);this.movement.state='completed';return true}};token.parent.tokens.set('t',token);
 const result=await native.runDefensiveAdvanceMovement({game,token,receipt:{nonce:'n',speed:20,origin:{x:0,y:0,elevation:0}},validate:()=>{},bindPlan:async id=>calls.push(['bind',id]),confirm:async()=>{calls.push(['confirm']);return 'moved'}});
 assert.equal(result,'moved');assert.deepEqual(calls.map(c=>c[0]),['plan','bind','start','confirm']);
 assert.deepEqual(calls[0][1],{allowedActions:['walk'],maxCost:20,preventDrop:true});
});

test('cancelled or incomplete native movement cannot grant a Strike and never moves by coordinate update',async()=>{
 assert.equal(typeof native.runDefensiveAdvanceMovement,'function');
 for(const cancel of [true,false]){let binds=0,confirms=0;const user={id:'owner'},game={user},token={id:'t',uuid:'Scene.s.Token.t',x:0,y:0,elevation:0,parent:{tokens:new Map()},object:{planMovement:async()=>cancel?null:{id:'p'}},movement:{id:'p',state:'completed',user,finished:Promise.resolve(false)},startMovement:async()=>true};token.parent.tokens.set('t',token);
 const result=await native.runDefensiveAdvanceMovement({game,token,receipt:{nonce:'n',speed:20,origin:{x:0,y:0,elevation:0}},validate:()=>{},bindPlan:async()=>binds++,confirm:async()=>confirms++});assert.equal(result,null);assert.equal(confirms,0);assert.equal(binds,cancel?0:1);}
});

test('native movement finished can precede visual coordinates; await the bound animation before confirmation',async()=>{
 const user={id:'owner'},game={user},animation=Promise.withResolvers();let confirms=0;
 const token={id:'t',x:0,y:0,elevation:0,movement:{id:'p',state:'planned',user,finished:Promise.resolve(true),animation:{ended:animation.promise}},object:{planMovement:async()=>({id:'p'})},startMovement:async()=>true};
 const pending=native.runDefensiveAdvanceMovement({game,token,receipt:{speed:20,origin:{x:0,y:0,elevation:0}},validate:()=>{},bindPlan:async()=>{},confirm:async()=>++confirms});
 await new Promise(setImmediate);assert.equal(confirms,0);
 token.movement.state='completed';animation.resolve();assert.equal(await pending,1);
});

test('included native Strike keeps exact MAP and target, publishes natively, and scopes free glyph to its own card',async()=>{
 assert.equal(typeof native.rollDefensiveAdvanceStrike,'function');
 const handlers=new Map(),Hooks={on:(n,fn)=>(handlers.set(n,fn),n),off:n=>handlers.delete(n)},game={user:{id:'owner',settings:{showCheckDialogs:false}},messages:new Map()},actor={id:'a',uuid:'Actor.a'},token={id:'t',uuid:'Scene.s.Token.t',parent:{id:'s'}},target={uuid:'Scene.s.Token.e',actor:{uuid:'Actor.e'},object:{}},receipt={nonce:'n',messageId:'original',userId:'owner',actorUuid:actor.uuid,tokenUuid:token.uuid,map:1,weaponKey:'Actor.a.Item.w#base'};
 let options;
 // PF2e 8.5.1 native Strike has no action glyph in its heading. A later note
 // may contain a glyph and must not be relabelled as the included Strike.
 const card={id:'check',author:game.user,speaker:{actor:'a',scene:'s',token:'t'},flavor:'<h4 class="action"><strong>Melee Strike: Sword</strong></h4><p>Other note <span class="action-glyph">A</span></p>',isCheckRoll:true,rolls:[{total:24}],flags:{pf2e:{origin:{actor:actor.uuid,uuid:'Actor.a.Item.w'},context:{type:'attack-roll',action:'strike',mapIncreases:1,target:{actor:'Actor.e',token:target.uuid},origin:{actor:actor.uuid,token:token.uuid},outcome:'success',options:[]}}},updateSource(changes){for(const[k,v]of Object.entries(changes)){if(k==='flavor')this.flavor=v;else this.flags[MODULE_ID]={...this.flags[MODULE_ID],defensiveAdvanceStrike:v}}}};
 const option={key:receipt.weaponKey,itemUuid:'Actor.a.Item.w',usage:null,strike:{variants:[{}, {async roll(params){options=params;card.flags.pf2e.context.options=[...params.options];handlers.get('preCreateChatMessage')(card);game.messages.set('check',card);await params.callback(card.rolls[0],'success',card);return card.rolls[0]}}]}};
 actor.getActiveTokens=()=>[token];
 const result=await native.rollDefensiveAdvanceStrike({game,Hooks,actor,token,target,receipt,option,validate:()=>option});
 assert.equal(result,'check');assert.equal(options.target,target.object);assert.equal(options.createMessage,true);assert.ok(options.options.has('action:free'));assert.equal(card.flags.pf2e.context.action,'strike');assert.equal(card.flags.pf2e.context.mapIncreases,1);assert.match(card.flavor,/<h4 class="action">.*>F<.*<\/h4>/);assert.match(card.flavor,/<p>Other note <span class="action-glyph">A<\/span><\/p>/);assert.equal(card.flags[MODULE_ID].defensiveAdvanceStrike.messageId,'original');assert.equal(handlers.size,0);
});
