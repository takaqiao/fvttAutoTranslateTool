import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

function readySource() {
 const main = fs.readFileSync(new URL('../scripts/main.mjs', import.meta.url), 'utf8');
 const first = main.indexOf('let nativeBridgeVerification='), end = main.indexOf(' const shieldAdapter=', first);
 assert(first >= 0 && end > first, 'Native ready initialization source must be present');
 return new Function('fetch', 'game', 'verifyNativeIWRBridge', 'verifyManualPoolProviders', `return (async()=>{${main.slice(first, end)} return nativeBridgeVerification;})()`);
}
test('actual ready initialization qualifies the fetched pair before exploration creates its readers', async () => {
 const events = [], pf = Uint8Array.of(1, 2, 3), tb = Uint8Array.of(4, 5), game = {system: {id: 'pf2e'}, modules: new Map([['pf2e-toolbelt', {active: true}]])};
 const bridgeProof = {ready: true};
 const fetch = async (url, options) => { events.push(url); assert.equal(options.cache, 'no-store'); return {ok: true, arrayBuffer: async () => url.startsWith('systems/') ? pf.buffer : tb.buffer}; };
 const result = await readySource()(fetch, game, async input => { assert.deepEqual(input.source, pf); return bridgeProof; }, async input => { assert.equal(input.game, game); assert.deepEqual(input.pf2eSource, pf); assert.deepEqual(input.toolbeltSource, tb); events.push('pair-qualified'); return {ready: true}; });
 assert.equal(result, bridgeProof);
 assert.deepEqual(events, ['systems/pf2e/pf2e.mjs', 'modules/pf2e-toolbelt/scripts/main.js', 'pair-qualified']);
});
test('a missing Toolbelt source preserves IWR initialization and never issues a pair proof', async () => {
 const game = {system: {id: 'pf2e'}, modules: new Map([['pf2e-toolbelt', {active: true}]])}, bridgeProof = {ready: true};
 let issued = false;
 const fetch = async url => ({ok: url.startsWith('systems/'), arrayBuffer: async () => Uint8Array.of(1).buffer});
 const result = await readySource()(fetch, game, async () => bridgeProof, async () => { issued = true; return {ready: true}; });
 assert.equal(result, bridgeProof); assert.equal(issued, false);
});
