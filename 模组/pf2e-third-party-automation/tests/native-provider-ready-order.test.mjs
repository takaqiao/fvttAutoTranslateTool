import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

function readySource() {
 const main = fs.readFileSync(new URL('../scripts/main.mjs', import.meta.url), 'utf8');
 const first = main.indexOf('let nativeBridgeVerification='), end = main.indexOf(' const shieldAdapter=', first);
 assert(first >= 0 && end > first, 'Native ready initialization source must be present');
 return new Function('fetch', 'game', 'verifyNativeIWRBridge', `return (async()=>{${main.slice(first, end)} return nativeBridgeVerification;})()`);
}
test('actual ready initialization verifies the fetched system source and requests no Toolbelt source', async () => {
 const events = [], pf = Uint8Array.of(1, 2, 3), game = {system: {id: 'pf2e'}, modules: new Map([['pf2e-toolbelt', {active: true}]])};
 const bridgeProof = {ready: true};
 const fetch = async (url, options) => { events.push(url); assert.equal(options.cache, 'no-cache'); return {ok: true, arrayBuffer: async () => pf.buffer}; };
 const result = await readySource()(fetch, game, async input => { assert.equal(input.game, game); assert.deepEqual(input.source, pf); return bridgeProof; });
 assert.equal(result, bridgeProof);
 assert.deepEqual(events, ['systems/pf2e/pf2e.mjs']);
});
