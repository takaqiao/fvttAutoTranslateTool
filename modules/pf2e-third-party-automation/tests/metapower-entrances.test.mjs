import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/metapower/entrances.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
test('cached, factory and default variants wait for complete override and execute one native use for many actors',async()=>{
 assert.equal(typeof api.installActionEntrances,'function');const events=[],a={uuid:'a'},b={uuid:'b'};
 class Variant{async use(options){events.push('effect');await Promise.resolve();events.push('condition');return options.actors}}
 const cached=new Variant(),action={cost:'free',variants:new Map([['v',cached]]),toActionVariant:()=>new Variant(),async use(options){return this.toActionVariant().use(options)}};
 const game={pf2e:{actions:new Map([['action',action]])},user:{getActiveTokens:()=>[]}};
 api.installActionEntrances({game,eligible:()=>true,observe:async({actor},native)=>{events.push('begin:'+actor.uuid);const r=await native();events.push('finish:'+actor.uuid);return r}});
 assert.deepEqual(await action.use({actors:[b,a],message:{create:false}}),[b,a]);assert.deepEqual(events,['begin:a','begin:b','effect','condition','finish:b','finish:a']);
 events.length=0;await cached.use({actors:[a]});assert.deepEqual(events,['begin:a','effect','condition','finish:a']);
});
test('sheet handler wrapper preserves description sends and awaits the original Use handler',async()=>{
 const calls=[],item={id:'i'},actor={items:new Map([['i',item]])};const handlers={'item-to-chat':()=>calls.push('description'),'use-action':async()=>{calls.push('native');return 8}};
 api.wrapSheetHandlers({actor},handlers,async(_context,native)=>{calls.push('begin');const r=await native();calls.push('finish');return r},()=>true);
 handlers['item-to-chat']();const r=await handlers['use-action']({}, {closest:selector=>selector==='[data-item-id]'?{dataset:{itemId:'i'}}:null});
 assert.equal(r,8);assert.deepEqual(calls,['description','begin','native','finish']);
});
test('frozen Toolbelt helper stays callable through awaited DOM adapter without replacing its descriptor',async()=>{
 const events=[],item={actor:{uuid:'Actor.a'},system:{frequency:{value:1}}},native=async(_event,i)=>{events.push('native');i.system.frequency.value--;return 9};
 const toolbelt=Object.freeze({useAction:native}),use=api.createToolbeltEntrance({native:toolbelt.useAction,eligible:()=>true,observe:async(_ctx,next)=>{events.push('begin');const result=await next();events.push('finish');return result}});
 assert.equal(await use({},item),9);assert.equal(toolbelt.useAction,native);assert.equal(item.system.frequency.value,0);assert.deepEqual(events,['begin','native','finish']);
});
test('verified HUD persistent closure is replaced with awaited same native helper once',async()=>{
 const item={actor:{uuid:'a'}},events=[];
 class Controller{constructor(){this.item=item}use(){events.push('unawaited-original')}}
 const controller=new Controller();
 await api.patchHudController(controller,{kind:'persistent',eligible:()=>true,digest:async()=> 'verified',expectedHash:'verified',useToolbelt:async(_event,i)=>{assert.equal(i,item);events.push('helper');await Promise.resolve();return 7}});
 assert.equal(await controller.use({}),7);assert.deepEqual(events,['helper']);
});
test('validated original activity continuation is not admitted as a new action',async()=>{
 const actor={uuid:'a'},marker={actorUuid:'a',cardId:'c',nonce:'n'},variant={use:async()=> 'native'},action={variants:new Map([['v',variant]]),toActionVariant:()=>variant};
 const game={pf2e:{actions:new Map([['x',action]])},user:{}};let admissions=0;
 api.installActionEntrances({game,eligible:()=>true,continuation:(_actor,value)=>value===marker,observe:()=>{admissions++;throw Error('not a new action')}});
 assert.equal(await variant.use({actors:[actor],'pf2e-third-party-automation':{metapowerContinuation:marker}}),'native');assert.equal(admissions,0);
});
test('legacy callback-only aliases cannot silently cross an armed activation',()=>{
 const actor={uuid:'a'},callback=()=>{},nativeCalls=[],actions={balance:options=>{nativeCalls.push(options);return undefined}},game={pf2e:{actions},user:{getActiveTokens:()=>[{actor}]}};let armed=true;
 api.installLegacyActionBoundary({game,blocked:a=>a===actor&&armed,onError:()=>{}});
 assert.throws(()=>actions.balance({actors:[actor],callback}),/legacy|原生动作/i);assert.equal(nativeCalls.length,0);
 armed=false;const options={actors:[actor],callback};assert.equal(actions.balance(options),undefined);assert.equal(nativeCalls[0],options);
});
test('supported source rows receive an idempotent native Use control without changing item resources',()=>{
 const added=[],items=[{id:'w'},{id:'medic'},{id:'unrelated'}],rows=items.map(item=>({dataset:{itemId:item.id},querySelector:selector=>selector.includes('use-action')?(added.find(b=>b.item===item.id)??null):{append:button=>added.push({...button,item:item.id})}}));
 const root={ownerDocument:{createElement:()=>({dataset:{}})},querySelectorAll:()=>rows},actor={items:new Map(items.map(i=>[i.id,i]))};
 api.ensureNativeUseControls(root,actor,item=>item.id!=='unrelated');api.ensureNativeUseControls(root,actor,item=>item.id!=='unrelated');
 assert.deepEqual(added.map(b=>[b.item,b.dataset.action]),[['w','use-action'],['medic','use-action']]);assert.equal(added[0].type,'button');assert.equal(items[0].system,undefined);
});
