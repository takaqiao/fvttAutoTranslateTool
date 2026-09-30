import test from 'node:test';
import assert from 'node:assert/strict';
import * as native from '../scripts/native-context.mjs';

test('target names follow the choice recipient rather than the GM executor',()=>{
 const gm={id:'gm',isGM:true},player={id:'player'},owner={id:'owner'};
 const game={user:gm,pf2e:{settings:{tokens:{nameVisibility:true}}}};
 const target={name:'Secret monster',playersCanSeeName:false,actor:{name:'Secret actor',testUserPermission:user=>user===owner}};
 assert.equal(typeof native.publicTargetName,'function');
 assert.equal(native.publicTargetName(target,{game,user:player}),'目标');
 assert.equal(native.publicTargetName(target,{game,user:owner}),'Secret monster');
 assert.equal(native.publicTargetName(target,{game,user:gm}),'Secret monster');
 target.playersCanSeeName=true;
 assert.equal(native.publicTargetName(target,{game,user:player}),'Secret monster');
 game.pf2e.settings.tokens.nameVisibility=false;target.playersCanSeeName=false;
 assert.equal(native.publicTargetName(target,{game,user:player}),'Secret monster');
});
