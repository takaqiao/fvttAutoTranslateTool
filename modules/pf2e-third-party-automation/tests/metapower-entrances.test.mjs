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
