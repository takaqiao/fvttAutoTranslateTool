import test from 'node:test';
import assert from 'node:assert/strict';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
test('existing native cast wrapper awaits actual-use middleware once, preserving native arguments and result',async()=>{
 const callbacks=new Map(),events=[],casts=createNativeCastEvents({game:{},fromUuid:()=>{}}),item={actor:{items:[]}},options={rank:3};
 assert.equal(typeof casts.addCastMiddleware,'function');
 casts.addCastMiddleware(async(context,next)=>{assert.equal(context.item,item);assert.equal(context.options,options);events.push('begin');const result=await next();events.push('finish');return result});
 casts.register({libWrapper:{register:(_id,path,fn)=>callbacks.set(path,fn)}});
 const wrapper=callbacks.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast');
 assert.equal(await wrapper.call({},async(i,o)=>{assert.equal(i,item);assert.equal(o,options);events.push('native');return 7},item,options),7);assert.deepEqual(events,['begin','native','finish']);
});
