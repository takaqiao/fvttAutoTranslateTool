import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {verifyManualPoolProviders,isManualPoolProvider} from '../scripts/exploration/manual-pool-provider.mjs';
import {manualPoolBatchModel} from '../scripts/exploration/manual-pool-model.mjs';
import {native860Source,hash} from './native-iwr-860-fixture.mjs';
import {receiverContext} from './native-iwr-860-receiver-fixture.mjs';

const toolbelt=readFileSync(process.env.TOOLBELT_CURRENT_SOURCE);
assert.equal(hash(toolbelt),'6acd53e7a5198ef8b9f7a35caae943b4582cabe61ff57bb76572038f61e549da');
const bridge=buildNativeBridge({source:native860Source(),version:'8.6.0'}).buffer;
const pair=(pf2eSource=bridge,toolbeltSource=toolbelt,toolbeltVersion='3.58.4')=>buildSharedManualPair({pf2eSource,toolbeltSource,pf2eVersion:'8.6.0',toolbeltVersion});

function fixture(version='3.58.4'){
 const module={version,active:true},sources=pair(bridge,toolbelt,version);
 const hooks=[],context=receiverContext(sources.pf2e.toString()),game=context.game;
 Object.assign(game,{system:{id:'pf2e',version:'8.6.0'},modules:new Map([['pf2e-toolbelt',module]])});
 context.Hooks={once:(_event,fn)=>hooks.push(fn)};
 vm.runInContext('__nativeManualPoolBatch.install();',context);
 vm.runInContext(sources.toolbelt.toString().split('/* end toolbelt manual pool */')[0],context);
 for(const hook of hooks)hook();
 return {game,module,sources,qualify:()=>verifyManualPoolProviders({game,pf2eSource:sources.pf2e,toolbeltSource:sources.toolbelt,hash})};
}

test('3.58.4 shared HP seams build once and preserve the full current bundle',()=>{
 const first=pair(),again=pair(first.pf2e,first.toolbelt);
 assert.equal(first.alreadyPatched,false);assert.equal(again.alreadyPatched,true);
 assert.deepEqual(again.pf2e,first.pf2e);assert.deepEqual(again.toolbelt,first.toolbelt);
 assert.equal(first.descriptors.toolbelt.sourceSHA256,hash(toolbelt));
 const refreshed=pair(first.pf2e,toolbelt);assert.deepEqual(refreshed.pf2e,first.pf2e);assert.deepEqual(refreshed.toolbelt,first.toolbelt);
});

for(const version of ['3.58.4','unlisted-release'])test(`current shared HP source qualifies independently of version ${version}`,async()=>{
 const f=fixture(version),proof=await f.qualify();assert.equal(proof.ready,true,proof.reason);
 assert.equal(proof.toolbeltSHA256,hash(f.sources.toolbelt));
 assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),true);
 assert.equal(isManualPoolProvider(f.module.api.explorationManualPool),true);
 f.module.version='changed-after-proof';
 assert.equal(isManualPoolProvider(f.module.api.explorationManualPool),false);
});

for(const [name,before,after]of [
 ['sender','e(c,s)','e(c)'],
 ['GM authority','!game.user.isActiveGM','false'],
 ['packet type','o.__type__!==t','false'],
 ['awaited document decoder','await IA(o)','IA(o)'],
 ['document resolution','case"document":return fromUuid(t)','case"document":return t'],
 ['master Promise','this.isValidMaster(e)&&e.update(n)','this.isValidMaster(e)&&void e.update(n)'],
 ['owner route','m.isOwner?m.update(g)','true?m.update(g)'],
 ['actor identity','e instanceof Actor&&!e.pack','true&&!e.pack'],
 ['master callback','this.path("master"),this.#g.bind(this)','this.path("master"),this.#f.bind(this)'],
 ['socket registration','game.socket.on(`module.${M.id}`,t)','game.socket.off(`module.${M.id}`,t)'],
])test(`changed 3.58.4 ${name} refuses both source outputs`,()=>{
 const changed=Buffer.from(toolbelt.toString().replace(before,after));assert.notDeepEqual(changed,toolbelt);
 const originals=[Buffer.from(bridge),Buffer.from(changed)];assert.throws(()=>pair(bridge,changed),/toolbelt-source-seam/);
 assert.deepEqual(bridge,originals[0]);assert.deepEqual(changed,originals[1]);
});

test('a changed served socket invalidates previously qualified 3.58.4 providers',async()=>{
 const f=fixture();assert.equal((await f.qualify()).ready,true);
 f.sources.toolbelt=Buffer.from(f.sources.toolbelt.toString().replace('e(c,s)','e(c)'));
 assert.equal((await f.qualify()).ready,false);assert.equal(isManualPoolProvider(f.module.api.explorationManualPool),false);
 assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
});

test('same-version module replacement cannot reuse 3.58.4 provider proof',async()=>{
 const f=fixture();assert.equal((await f.qualify()).ready,true);
 f.game.modules.set('pf2e-toolbelt',{...f.module});
 assert.equal(isManualPoolProvider(f.module.api.explorationManualPool),false);
});
