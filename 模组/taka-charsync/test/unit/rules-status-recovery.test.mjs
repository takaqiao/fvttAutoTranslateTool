import test from 'node:test';
import assert from 'node:assert/strict';
import {makeCampaignEnvironment, ruleSnapshot} from '../helpers/campaign-environment.mjs';

test('an already open sheet refreshes its failed status after rules recover', async () => {
  const {api, a} = await makeCampaignEnvironment({isolated: false});
  a.name = 'A';
  a.isOwner = true;
  const get = game.settings.get.bind(game.settings);
  game.settings.get = (namespace, key, options) => namespace === 'taka-charsync'
    ? 'test-published' : get(namespace, key, options);
  foundry.utils.getRoute = value => value;
  let unavailable = true;
  let requests = 0;
  globalThis.fetch = async () => {
    requests++;
    if (unavailable) throw new Error('offline');
    return {ok: true, status: 200, json: async () => ({ruleSnapshot: ruleSnapshot(), cards: {}})};
  };
  const {renderStatusBar, endStatusSession} = await import('../../scripts/char/status-bar.mjs');
  const app = {actor: a, options: {classes: []}, element: [], rendered: true};
  const settle = () => new Promise(resolve => setImmediate(resolve));
  const log = console.error;
  try {
    console.error = () => {};
    renderStatusBar(app);
    await settle();
    assert.equal(requests, 1);
    assert.equal(api.actorRuleState(a).ok, false);
    unavailable = false;
    api.setRuleRefreshError(null);
    renderStatusBar(app);
    renderStatusBar(app);
    await settle();
    assert.equal(requests, 2, 'two renders share the pending recovery read');
    assert.equal(api.actorRuleState(a).ok, true);
    renderStatusBar(app);
    await settle();
    assert.equal(requests, 2, 'successful recovery no longer retains an error session');
  } finally {
    console.error = log;
    endStatusSession(app);
  }
});
