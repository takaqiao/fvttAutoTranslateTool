import fs from 'node:fs';
import vm from 'node:vm';
export const bridgeFixture=JSON.parse(fs.readFileSync(new URL('./fixtures/persistent-bridge-0.5.4-native.json',import.meta.url)));

export function prepareBridge(f,callbacks={}){
  const noop=()=>{},yes=()=>true;
  Object.assign(f.game,{user:{id:'player'},modules:new Map([['pf2e-dsn-persistent-bridge',{active:true}]])});
  Object.assign(f.box,{ready:Promise.resolve(),allowInteractivity:true});
  f.box.diceScene.scene={children:[],add:noop,remove:noop};
  Object.assign(f.box.inputHandler,{_beginPersistentGrab:noop,_activatePreRoll:noop,_resetPreRollState:noop,
    _computeThrowVelocity:noop,mouse:{heldPersistentDice:[]}});
  Object.assign(f.box.persistentDiceManager,{onQueueThrow:yes,matchSFX:noop,throwPersistentDice:noop,_getPersistentTextureCache:noop,
    async removePersistentDie(id){
      const index=f.engine.persistentDiceList.findIndex(mesh=>mesh.userData?.persistentId===id);
      if(index>=0)f.engine.persistentDiceList.splice(index,1);
      return true;
    }});
  Object.assign(f.game.dice3d,{box:f.box,exports:{Utils:f.context.Utils},DiceFactory:{getAppearanceForDice:noop},
    persistent:{spawn:noop,remove:noop,_emitPersistentEvent:noop},_buildDiceBox:noop,_fadeOutCanvas:noop});
  Object.assign(f.game.dice3d.pendingThrows,{claimThrow:noop,shouldStampInteractive:yes});
  f.context.selectThrowDirection=noop;f.context.applyThrowDirection=(original,...args)=>original(...args);
  const factory=vm.runInContext('(()=>{'+bridgeFixture.source.replace(/^import[^\r\n]+[\r\n]+/,'').replace(/^export /gm,'')+';return createDsnAdapter;})()',f.context);
  return factory({dice3d:f.game.dice3d,user:f.game.user,utils:{},hooks:{on:()=>1,off:noop},
    nativePersistentEnabled:()=>false,onSettled:noop,...callbacks});
}

export function prepareRegisteredSettings(f){
  const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/persistent-bridge-settings-0.5.4-native.json',import.meta.url)));
  const native=vm.runInContext('(()=>{'+fixture.declarations.replace(/^export /gm,'')+';'+fixture.createBridge+';'+fixture.registerSettings+
    ';return {createBridge,registerSettings,SETTINGS,MOD_ID};})()',f.context);
  const registered=new Map(),values=new Map(),read=f.game.settings.get,noop=()=>{};
  let bridge,changes=Promise.resolve();
  f.game.settings.register=(_id,key,definition)=>{registered.set(key,definition);values.set(key,definition.default);};
  f.game.settings.get=(id,key)=>id===native.MOD_ID?values.get(key):read(id,key);
  native.registerSettings(()=>changes=changes.then(async()=>{
    if(values.get(native.SETTINGS.enabled)===false){await bridge?.disable();return;}
    bridge=native.createBridge({getSetting:key=>values.get(key),userId:'player',
      dice3d:callbacks=>prepareBridge(f,callbacks),pf2e:()=>noop,
      view:async()=>({element:{},mount:noop,setSize:noop,layout:noop,show:noop,clear:noop,setState:noop,dispose:noop}),
      gestures:()=>Object.assign(noop,{cancel:noop})});
    await bridge.enable();
  }),f.game);
  f.game.settings.set=async(id,key,value)=>{
    values.set(key,value);registered.get(key).onChange();await changes;
  };
  return {registered,get bridge(){return bridge;},setEnabled:value=>f.game.settings.set(native.MOD_ID,native.SETTINGS.enabled,value)};
}
