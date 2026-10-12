import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {buildNativeBridge,buildSharedManualPair} from '../../tools/automatic-source-patches/native.mjs';
import {verifyManualPoolProviders,isManualPoolProvider} from '../../scripts/exploration/manual-pool-provider.mjs';
import {manualPoolBatchModel} from '../../scripts/exploration/manual-pool-model.mjs';
import {batchRegion,flatRegion,observerRegion} from '../../scripts/native-source-shapes.mjs';
import {native860} from '../native-iwr-860-fixture.mjs';
import {native861,native861Source,hash} from '../native-iwr-861-fixture.mjs';
import {receiverContext} from '../native-iwr-861-receiver-fixture.mjs';

assert.ok(process.env.TOOLBELT_CURRENT_SOURCE,'TOOLBELT_CURRENT_SOURCE is required');
const toolbelt=readFileSync(process.env.TOOLBELT_CURRENT_SOURCE);
assert.equal(hash(toolbelt),'6acd53e7a5198ef8b9f7a35caae943b4582cabe61ff57bb76572038f61e549da');

function fixture(){
 const module={version:'3.58.4',active:true},bridge=buildNativeBridge({source:native861Source(),version:'8.6.1'});
 const sources=buildSharedManualPair({pf2eSource:bridge.buffer,toolbeltSource:toolbelt,pf2eVersion:'8.6.1',toolbeltVersion:module.version});
 const hooks=[],context=receiverContext(sources.pf2e.toString()),game=context.game;
 Object.assign(game,{system:{id:'pf2e',version:'8.6.1'},modules:new Map([['pf2e-toolbelt',module]])});context.Hooks={once:(_event,fn)=>hooks.push(fn)};
 vm.runInContext('__nativeManualPoolBatch.install();',context);
 vm.runInContext(sources.toolbelt.toString().split('/* end toolbelt manual pool */')[0],context);for(const hook of hooks)hook();
 return {game,module,sources,qualify:()=>verifyManualPoolProviders({game,pf2eSource:sources.pf2e,toolbeltSource:sources.toolbelt,hash})};
}

test('actual 8.6.1 shared source seams qualify the retained native and Toolbelt providers',async()=>{
 const f=fixture(),proof=await f.qualify();assert.equal(proof.ready,true,proof.reason);
 assert.equal(proof.pf2eSHA256,hash(f.sources.pf2e));assert.equal(proof.toolbeltSHA256,hash(f.sources.toolbelt));
 assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),true);assert.equal(isManualPoolProvider(f.module.api.explorationManualPool),true);
 f.game.system.version='8.6.2';assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
});
for(const [name,region,before,after]of [
 ['initializer',text=>text,'Fm.onInit();__nativeManualPoolBatch.install();','Nm.onInit();__nativeManualPoolBatch.install();'],
 ['batch roll',batchRegion,'instanceof un','instanceof ln'],
 ['receiver modifier',flatRegion,'new Y(','new J('],
 ['native bridge',text=>text,'actorDamage: x - D - ae','actorDamage: x - D - ie'],
])test(`an 8.6.1 shared source with another ${name} cannot reuse its proof`,async()=>{
 const f=fixture();assert.equal((await f.qualify()).ready,true);const text=f.sources.pf2e.toString(),part=region(text);assert.ok(part.includes(before));f.sources.pf2e=Buffer.from(text.replace(part,part.replace(before,after)));
 assert.equal((await f.qualify()).ready,false);assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
});
test('an 8.6.1 served source carrying the 8.6.0 stacking loop is not qualified',async()=>{
 const f=fixture();assert.equal((await f.qualify()).ready,true);
 // The observer keeps its own copy of the stacking rules, so only the native
 // region outside it is exchanged.
 const text=f.sources.pf2e.toString(),stacking=native861.regions.stackingRegion,observer=observerRegion(text,'pf2e').region,from=text.indexOf(observer);
 const at=[text.indexOf(stacking),text.lastIndexOf(stacking)].find(index=>index<from||index>=from+observer.length);
 assert.ok(at>=0);
 f.sources.pf2e=Buffer.from(text.slice(0,at)+native860.regions.stackingRegion+text.slice(at+stacking.length));
 const refused=await f.qualify();assert.equal(refused.ready,false);assert.equal(refused.reason,'manual-pool-native-receiver-seam-mismatch');
});
