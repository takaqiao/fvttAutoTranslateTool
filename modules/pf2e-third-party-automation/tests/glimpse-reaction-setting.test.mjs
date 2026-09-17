import test from 'node:test';
import assert from 'node:assert/strict';
import {glimpseReactionSetting,GLIMPSE_REACTION_REASON as reason} from '../scripts/glimpse-reaction-setting.mjs';
const before=['shield-block','glimpse-of-redemption','reactive-strike'],after=['shield-block','reactive-strike'];
const game={settings:{get:()=>[{setting:'pf2e-reaction.builtinReactionsEnabled',changes:[{path:'builtinReactionsEnabled',before,after,reason}]}]}};
test('verified provider removes only its builtin reminder; an unavailable provider restores its exact prior change',()=>{
 assert.deepEqual(glimpseReactionSetting(before,game,true),after);assert.deepEqual(glimpseReactionSetting(after,game,false),before);assert.deepEqual(before,['shield-block','glimpse-of-redemption','reactive-strike']);
});
test('provider fallback preserves pre-existing disabled reminders and subsequent custom settings',()=>{
 assert.deepEqual(glimpseReactionSetting(after,{settings:{get:()=>[]}},false),after);
 assert.deepEqual(glimpseReactionSetting(['shield-block'],game,false),['shield-block']);
});
