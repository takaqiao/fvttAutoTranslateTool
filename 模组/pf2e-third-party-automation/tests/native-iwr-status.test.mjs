import test from 'node:test';
import assert from 'node:assert/strict';
import {notifyNativeIWRStatus} from '../scripts/native-iwr-status.mjs';
const source='Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.naawsnBug9EOpzfN';
test('new Eldamon worlds report unavailable actual-damage accounting when the native bridge is missing',()=>{
 const user={id:'gm',isGM:true},game={world:{id:'new-eldamon-world'},user,users:{activeGM:user},actors:[{items:[{sourceId:source}]}]},messages=[];
 assert.equal(notifyNativeIWRStatus({game,diagnostic:{ready:false,reason:'missing-native-iwr-bridge'},warn:m=>messages.push(m)}),true);
 assert.match(messages[0],/电击实伤/);assert.match(messages[0],/普通伤害仍照常/);
 assert.equal(notifyNativeIWRStatus({game,diagnostic:{ready:true},warn:m=>messages.push(m)}),false);assert.equal(messages.length,1);
 game.user={id:'player',isGM:false};assert.equal(notifyNativeIWRStatus({game,diagnostic:{ready:false},warn:m=>messages.push(m)}),false);
});
test('unrelated new worlds do not receive Eldamon bridge notices',()=>{
 const user={id:'gm',isGM:true},game={world:{id:'unrelated'},user,users:{activeGM:user},actors:[{items:[{name:'Elemental Powers',sourceId:'wrong'}]}]};
 assert.equal(notifyNativeIWRStatus({game,diagnostic:{ready:false},warn:()=>assert.fail('irrelevant notice')}),false);
});
