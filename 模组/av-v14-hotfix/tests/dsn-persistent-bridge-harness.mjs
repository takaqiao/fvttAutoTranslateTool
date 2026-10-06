import fs from 'node:fs';
import vm from 'node:vm';
export const bridgeFixture=JSON.parse(fs.readFileSync(new URL('./fixtures/persistent-bridge-0.5.4-native.json',import.meta.url)));

export function prepareBridge(f){
  const noop=()=>{},yes=()=>true;
  Object.assign(f.game,{user:{id:'player'},modules:new Map([['pf2e-dsn-persistent-bridge',{active:true}]])});
  Object.assign(f.box,{ready:Promise.resolve(),allowInteractivity:true});
  f.box.diceScene.scene={add:noop,remove:noop};
  Object.assign(f.box.inputHandler,{_beginPersistentGrab:noop,_activatePreRoll:noop,_resetPreRollState:noop,
    _computeThrowVelocity:noop,mouse:{heldPersistentDice:[]}});
  Object.assign(f.box.persistentDiceManager,{onQueueThrow:yes,matchSFX:noop,throwPersistentDice:noop,_getPersistentTextureCache:noop});
  Object.assign(f.game.dice3d,{box:f.box,exports:{Utils:f.context.Utils},DiceFactory:{getAppearanceForDice:noop},
    persistent:{spawn:noop,remove:noop,_emitPersistentEvent:noop},_buildDiceBox:noop,_fadeOutCanvas:noop});
  Object.assign(f.game.dice3d.pendingThrows,{claimThrow:noop,shouldStampInteractive:yes});
  f.context.selectThrowDirection=noop;f.context.applyThrowDirection=(original,...args)=>original(...args);
  const factory=vm.runInContext('(()=>{'+bridgeFixture.source.replace(/^import[^\r\n]+[\r\n]+/,'').replace(/^export /gm,'')+';return createDsnAdapter;})()',f.context);
  return factory({dice3d:f.game.dice3d,user:f.game.user,utils:{},hooks:{on:()=>1,off:noop},
    nativePersistentEnabled:()=>false,onSettled:noop});
}
