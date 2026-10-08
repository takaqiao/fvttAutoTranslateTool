import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {buildNativeBridge,buildSharedManualPair} from '../../tools/automatic-source-patches/native.mjs';
import {verifyManualPoolProviders,isManualPoolProvider} from '../../scripts/exploration/manual-pool-provider.mjs';
import {manualPoolBatchModel} from '../../scripts/exploration/manual-pool-model.mjs';
import {batchRegion,flatRegion} from '../../scripts/native-source-shapes.mjs';
import {native860Source,hash} from '../native-iwr-860-fixture.mjs';
import {receiverContext} from '../native-iwr-860-receiver-fixture.mjs';

function fixture(){
 const module={version:'3.56.5',active:true},bridge=buildNativeBridge({source:native860Source(),version:'8.6.0'});
 const sources=buildSharedManualPair({pf2eSource:bridge.buffer,toolbeltSource:readFileSync(process.env.TOOLBELT_MANUAL_SOURCE),pf2eVersion:'8.6.0',toolbeltVersion:module.version});
 const hooks=[],context=receiverContext(sources.pf2e.toString()),game=context.game;
 Object.assign(game,{system:{id:'pf2e',version:'8.6.0'},modules:new Map([['pf2e-toolbelt',module]])});context.Hooks={once:(_event,fn)=>hooks.push(fn)};
 vm.runInContext('__nativeManualPoolBatch.install();',context);
 vm.runInContext(sources.toolbelt.toString().split('/* end toolbelt manual pool */')[0],context);for(const hook of hooks)hook();
 return {game,module,sources,qualify:()=>verifyManualPoolProviders({game,pf2eSource:sources.pf2e,toolbeltSource:sources.toolbelt,hash})};
}

test('actual 8.6 shared source seams qualify the retained native and Toolbelt providers',async()=>{
 const f=fixture(),proof=await f.qualify();assert.equal(proof.ready,true,proof.reason);
 assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),true);assert.equal(isManualPoolProvider(f.module.api.explorationManualPool),true);
 f.game.system.version='8.6.1';assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
});
for(const [name,region,before,after]of [['initializer',text=>text,'Nm.onInit();__nativeManualPoolBatch.install();','jm.onInit();__nativeManualPoolBatch.install();'],['batch roll',batchRegion,'instanceof ln','instanceof cn'],['receiver modifier',flatRegion,'new Y(','new J(']])test(`an 8.6 shared source with another ${name} cannot reuse its proof`,async()=>{
 const f=fixture();assert.equal((await f.qualify()).ready,true);const text=f.sources.pf2e.toString(),part=region(text);assert.ok(part.includes(before));f.sources.pf2e=Buffer.from(text.replace(part,part.replace(before,after)));
 assert.equal((await f.qualify()).ready,false);assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
});
