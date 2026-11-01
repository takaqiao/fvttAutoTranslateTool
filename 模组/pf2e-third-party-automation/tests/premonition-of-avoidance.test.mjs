import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/premonition-of-avoidance.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}

const SOURCE='Compendium.pf2e.feats-srd.Item.KYl1rWFOHe0e6VqJ',OPTION='premonition-of-avoidance';
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve}};
function fixture({choice='use',choose,onError}={}){
 const user={id:'player'},game={world:{id:'bob-2'},system:{id:'pf2e'},user,actors:new Map()};
 const toggles=[],errors=[],prompts=[],calls=[];
 const actor={id:'hero',uuid:'Actor.hero',type:'character',items:new Map(),owner:true,testUserPermission:()=>actor.owner,
  toggleRollOption:async(domain,option,itemId,value)=>{toggles.push({domain,option,itemId,value});actor.items.get(itemId).system.rules.find(r=>r.option===option).value=value}};
 const item={id:'feat',uuid:'Actor.hero.Item.feat',type:'feat',name:'Premonition of Avoidance',_stats:{compendiumSource:SOURCE},system:{slug:OPTION,rules:[
  {key:'RollOption',domain:'saving-throw',option:OPTION,toggleable:true,value:true},
  {key:'FlatModifier',selector:'saving-throw',type:'circumstance',value:2,predicate:['origin:type:hazard',OPTION]},
  {key:'RollOption',domain:'all',option:'another-option',toggleable:true,value:true},
 ]}};
 actor.items.set(item.id,item);game.actors.set(actor.id,actor);
 const provider=api.createPremonitionOfAvoidance?.({game,choose:async request=>{prompts.push(request);return choose?choose(request):choice},onError:onError??(error=>errors.push(error))})??{interceptCheck:(native,...args)=>native(...args)};
 const context=()=>({actor,type:'saving-throw',domains:['all','saving-throw','reflex'],options:new Set(['origin:type:hazard',OPTION,'another-option'])});
 const native=async(check,ctx,event,callback)=>{calls.push({check,context:ctx,event,callback});const result={option:ctx.options.has(OPTION)};await callback?.(result);return result};
 const run=(ctx=context(),next=native,callback)=>provider.interceptCheck(next,{modifiers:[]},ctx,null,callback);
 return {provider,game,actor,item,toggles,errors,prompts,calls,context,native,run};
}

test('one confirmed hazard save keeps its native option and clears only its saved toggle',async()=>{
 const f=fixture(),ctx=f.context(),result=await f.run(ctx);
 assert.equal(result.option,true);assert.equal(f.prompts.length,1);assert.equal(f.calls.length,1);
 assert.equal(f.item.system.rules[0].value,false);assert.equal(f.item.system.rules[2].value,true);
 assert.deepEqual([...ctx.options],['origin:type:hazard',OPTION,'another-option']);
 assert.deepEqual(f.toggles,[{domain:'saving-throw',option:OPTION,itemId:'feat',value:false}]);
});

for(const choice of ['decline',null,false,undefined])test('declined or closed choice removes only this roll option: '+String(choice),async()=>{
 const f=fixture({choose:async()=>choice}),ctx=f.context(),result=await f.run(ctx);
 assert.equal(result.option,false);assert.deepEqual([...ctx.options],['origin:type:hazard','another-option']);
 assert.equal(f.item.system.rules[0].value,false);assert.equal(f.calls.length,1);
});

test('normal spell saves neither prompt nor consume the armed hazard reaction',async()=>{
 const f=fixture(),ctx=f.context();ctx.options.delete('origin:type:hazard');ctx.options.add('origin:type:npc');
 await f.run(ctx);assert.equal(f.prompts.length,0);assert.equal(f.item.system.rules[0].value,true);assert.equal(f.toggles.length,0);
});

for(const [name,change]of [
 ['world',f=>f.game.world.id='other-world'],['system',f=>f.game.system.id='other-system'],
 ['actor type',f=>f.actor.type='npc'],['ownership',f=>f.actor.owner=false],
 ['source',f=>f.item._stats.compendiumSource='Compendium.homebrew.feats.Item.KYl1rWFOHe0e6VqJ'],
 ['item type',f=>f.item.type='effect'],['missing feat',f=>f.actor.items.clear()],
 ['reroll',(_f,ctx)=>ctx.isReroll=true],['non-save',(_f,ctx)=>ctx.type='skill-check'],
 ['unarmed option',(_f,ctx)=>ctx.options.delete(OPTION)],
])test('unmanaged '+name+' preserves the native invocation',async()=>{
 const f=fixture(),ctx=f.context();change(f,ctx);const options=ctx.options;let callbackCalls=0;
 await f.run(ctx,f.native,()=>callbackCalls++);
 assert.equal(f.prompts.length,0);assert.equal(f.toggles.length,0);assert.equal(f.calls.length,1);assert.equal(callbackCalls,1);assert.equal(ctx.options,options);
});

test('a stale prepared second save cannot reuse the first confirmation',async()=>{
 const f=fixture(),second=f.context();assert.equal((await f.run()).option,true);
 assert.equal((await f.run(second)).option,false);assert.equal(f.prompts.length,1);assert.equal(f.toggles.length,1);
 f.item.system.rules[0].value=true;assert.equal((await f.run()).option,true);assert.equal(f.prompts.length,2);
});

test('a false saved toggle does not become armed through stale check options',async()=>{
 const f=fixture();f.item.system.rules[0].value=false;
 assert.equal((await f.run()).option,false);assert.equal(f.prompts.length,0);assert.equal(f.toggles.length,0);
});

test('concurrent prepared saves use one armed toggle without blocking the other native roll',async()=>{
 const ready=deferred(),answer=deferred(),f=fixture({choose:async()=>{ready.resolve();return answer.promise}});
 const first=f.run();await Promise.race([ready.promise,first.then(()=>assert.fail('the first save never asked for confirmation'))]);
 const second=await f.run();assert.equal(second.option,false);assert.equal(f.prompts.length,1);
 answer.resolve('use');assert.equal((await first).option,true);assert.equal(f.calls.length,2);assert.equal(f.toggles.length,1);
});

test('a nested check from the native callback cannot reuse the pending toggle',async()=>{
 const f=fixture();let nested;
 assert.equal((await f.run(f.context(),f.native,async()=>{nested=await f.run()})).option,true);
 assert.equal(nested.option,false);assert.equal(f.prompts.length,1);assert.equal(f.calls.length,2);
});

for(const location of ['actor','origin','target'])test('contextual '+location+' clone clears the matching world actor',async()=>{
 const f=fixture(),ctx=f.context(),clone={...f.actor,items:new Map(),toggleRollOption:()=>assert.fail('updated a contextual clone')};
 delete ctx.actor;if(location==='actor')ctx.actor=clone;else ctx[location]={actor:clone,self:true};
 await f.run(ctx);assert.equal(f.prompts[0]?.actor,f.actor);assert.equal(f.item.system.rules[0].value,false);
});

test('a synthetic token actor is not replaced by a world actor sharing its id',async()=>{
 const f=fixture(),ctx=f.context(),worldActor={...f.actor,uuid:'Actor.other',items:new Map()};f.game.actors.set(f.actor.id,worldActor);
 await f.run(ctx);assert.equal(f.prompts[0]?.actor,f.actor);assert.equal(f.item.system.rules[0].value,false);
});

for(const [name,change]of [
 ['source',f=>f.item._stats.compendiumSource='Compendium.homebrew.feats.Item.other'],
 ['removed feat',f=>f.actor.items.clear()],['permission',f=>f.actor.owner=false],
 ['world',f=>f.game.world.id='other-world'],
 ['replacement feat',f=>f.actor.items.set('feat',{...f.item,uuid:'Actor.hero.Item.replacement'})],
])test('a '+name+' change during confirmation cannot authorize the prepared bonus',async()=>{
 let f;f=fixture({choose:async()=>{change(f);return 'use'}});
 assert.equal((await f.run()).option,false);assert.equal(f.toggles.length,0);assert.equal(f.calls.length,1);
});

test('turning off the saved toggle during confirmation removes the prepared bonus',async()=>{
 let f;f=fixture({choose:async()=>{f.item.system.rules[0].value=false;return 'use'}});
 assert.equal((await f.run()).option,false);assert.equal(f.toggles.length,0);
});

test('native cancellation clears the toggle without inventing a callback',async()=>{
 const f=fixture();let callbacks=0,calls=0;
 const result=await f.run(f.context(),async()=>{calls++;return null},()=>callbacks++);
 assert.equal(result,null);assert.equal(calls,1);assert.equal(callbacks,0);assert.equal(f.item.system.rules[0].value,false);
});

for(const stage of ['choose','native','callback'])test(stage+' failure keeps the original error and clears the saved toggle',async()=>{
 const failure=new Error(stage),f=fixture(stage==='choose'?{choose:()=>{throw failure}}:{});
 await assert.rejects(()=>f.run(f.context(),stage==='native'?async()=>{throw failure}:f.native,stage==='callback'?()=>{throw failure}:undefined),error=>error===failure);
 assert.equal(f.item.system.rules[0].value,false);assert.equal(f.toggles.length,1);
 f.item.system.rules[0].value=true;
 if(stage!=='choose')assert.equal((await f.run()).option,true);
});

test('toggle cleanup reports its error without replacing the native result',async()=>{
 const f=fixture(),failure=new Error('toggle update');f.actor.toggleRollOption=async()=>{throw failure};
 const result=await f.run();assert.equal(result.option,true);assert.deepEqual(f.errors,[failure]);assert.equal(f.calls.length,1);
});

test('native receives the original check, event and callback exactly once',async()=>{
 const f=fixture(),check={modifiers:[]},ctx=f.context(),event={shiftKey:true},callback=()=>{};
 await f.provider.interceptCheck(f.native,check,ctx,event,callback);
 assert.equal(f.calls.length,1);assert.equal(f.calls[0].check,check);assert.equal(f.calls[0].event,event);assert.equal(f.calls[0].callback,callback);
});
