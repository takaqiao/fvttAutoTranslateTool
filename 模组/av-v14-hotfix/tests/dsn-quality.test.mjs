import test from 'node:test';
import assert from 'node:assert/strict';
import {harness,optionalPatch,quality,high,plain,apply,operators} from './bbmm-harness.mjs';
const patch=await optionalPatch('dsn-quality');
function setup(opts,before){const e=harness(opts);before?.(e);e.register=()=>patch.registerDsnQualitySettings?.({runtime:e.runtime});e.register();e.install=()=>patch.installDsnQualityLocks?.({runtime:e.runtime});e.result=e.install();return e;}
const checkQuality=value=>{for(const[k,v]of Object.entries(quality))assert.equal(value[k],v,k);};
test('five existing BBMM rule IDs reconnect as client settings',()=>{const e=setup();for(const k of Object.keys(quality)){const cfg=e.registry.get('bbmm.dsnQuality.'+k);assert.equal(cfg?.scope,'client');assert.equal(cfg.config,true);assert.equal(cfg.requiresReload,true);}});

test('patch labels do not disable verified BBMM and DsN contracts',()=>{
 const e=setup({bbmmVersion:'1.5.0',dsnVersion:'6.5.0'});
 assert.equal(e.result.status,'installed');checkQuality(e.context.DsnSettings.CONFIG());
});

test('adapter registration can precede BBMM init, but setup requires its actual registry',()=>{
 const e=harness();e.registry.delete('bbmm.userSettingSync');e.registry.delete('bbmm.enableUserSettingSync');
 assert.equal(patch.registerDsnQualitySettings({runtime:e.runtime}).status,'registered');
 assert.equal(patch.installDsnQualityLocks({runtime:e.runtime}).status,'unsupported-bbmm-registry');
 assert.equal(e.wrappers.size,0);assert.equal(e.callbacks.size,0);
});

test('current native medium shadow quality reaches preview, saved flags and immediate renderer',async()=>{
 const e=setup();e.rules['bbmm.dsnQuality.shadowQuality'].value='medium';
 assert.equal(e.context.DsnSettings.CONFIG().shadowQuality,'medium');
 assert.equal(e.registry.get('bbmm.dsnQuality.shadowQuality').choices.medium,'中');
 const app=e.makeApp();app.onApply();await e.flush();
 assert.equal(e.factory.shadowQuality,'medium');assert.equal(e.factory.shadows,true);
 assert.equal(app.showcaseView.last.shadowQuality,'medium');assert.equal(e.controls.shadowQuality.value,'medium');
 await app._updateObject(null,{object:{...high,'appearance[global][colorset]':'custom'}});
 assert.equal(e.game.user.flags['dice-so-nice'].settings.shadowQuality,'medium');
 assert.equal(e.rendered.at(-1).shadowQuality,'medium');
});

test('medium shadow rules keep GM, soft-lock and disabled-sync native renderer choices',async()=>{
 for(const options of [{gm:true},{soft:true},{sync:false}]){
  const e=setup(options);e.rules['bbmm.dsnQuality.shadowQuality'].value='medium';
  assert.equal(e.context.DsnSettings.CONFIG().shadowQuality,'high');
  const app=e.makeApp();app.onApply();await e.flush();assert.equal(e.factory.shadowQuality,'high');
  await app._updateObject(null,{object:{...high,'appearance[global][colorset]':'custom'}});
  assert.equal(e.game.user.flags['dice-so-nice'].settings.shadowQuality,'high');
  assert.equal(e.rendered.at(-1).shadowQuality,'high');
 }
});

test('persisted quality survives a new config read after save and reset',async()=>{
 const e=setup();await e.makeApp()._updateObject(null,{object:{...high,'appearance[global][colorset]':'custom'}});
 checkQuality(e.context.DsnSettings.CONFIG());await e.makeApp()._clearUserRecord();
 const saved=structuredClone(e.game.user.flags);e.result.restore();
 const fresh=setup();fresh.game.user.flags=saved;checkQuality(fresh.context.DsnSettings.CONFIG());
 assert.equal(fresh.updates.length,0);
});
test('native CONFIG enforces new-player quality without mutating saved flags',()=>{const e=setup();delete e.game.user.flags['dice-so-nice'].settings;const before=structuredClone(e.game.user.flags);checkQuality(e.context.DsnSettings.CONFIG());assert.deepEqual(e.game.user.flags,before);});
test('reads preserve volume, rolling area, skin and all original arguments',()=>{const e=setup();const before=structuredClone(e.game.user.flags);const got=e.context.DsnSettings.CONFIG();checkQuality(got);assert.equal(got.soundsVolume,.7);assert.deepEqual(plain(got.rollingArea),{left:80});assert.deepEqual(e.game.user.flags,before);assert.equal(e.game.user.getFlag('world','keep'),1);});
test('native form parser clamps only selected quality fields',()=>{const e=setup();const got=e.makeApp().parseInputs({...high,soundsVolume:.2,'appearance[global][colorset]':'custom'});checkQuality(got);assert.equal(got.soundsVolume,.2);assert.equal(got.appearance.global.colorset,'custom');});
test('native 6.4.2 save keeps quality in both persisted flags and immediate renderer config',async()=>{const e=setup();await e.makeApp()._updateObject(null,{object:{...high,soundsVolume:.2,'appearance[global][colorset]':'custom'}});checkQuality(e.game.user.flags['dice-so-nice'].settings);assert.equal(e.game.user.flags['dice-so-nice'].settings.soundsVolume,.2);assert.deepEqual(plain(e.game.user.flags['dice-so-nice'].saved),{appearance:true,sfx:true});checkQuality(e.rendered.at(-1));assert.equal(e.rendered.at(-1).soundsVolume,.2);assert.deepEqual(plain(e.rendered.at(-1).rollingArea),{left:80});});
test('native 6.4.2 reset deletes original preferences but retains locks and native protected area',async()=>{const e=setup();await e.makeApp()._clearUserRecord();const flags=e.game.user.flags['dice-so-nice'];assert.deepEqual(plain(flags.settings),{...quality,rollingArea:{left:80},protectPersistent:true});for(const key of ['saved','sfxList','appearance','roleAppearance'])assert.equal(flags[key],undefined,key);checkQuality(e.rendered.at(-1));assert.equal(e.rendered.at(-1).soundsVolume,.5);assert.equal(e.game.user.flags.world.keep,1);});
test('the reset action in native save reaches the new reset method',async()=>{const e=setup(),app=e.makeApp();app.reset=true;await app._updateObject(null,{object:{...high,'appearance[global][colorset]':'custom'}});checkQuality(e.game.user.flags['dice-so-nice'].settings);assert.equal(e.game.user.flags['dice-so-nice'].appearance,undefined);});
test('partial flag updates preserve unrelated preferences and fields',async()=>{const e=setup();await e.game.user.update({flags:{'dice-so-nice':{settings:{glow:true,soundsVolume:.2}}}});checkQuality(e.game.user.flags['dice-so-nice'].settings);assert.equal(e.game.user.flags['dice-so-nice'].settings.soundsVolume,.2);assert.equal(e.game.user.flags.world.keep,1);assert.ok(e.game.user.flags['dice-so-nice'].appearance);});
for(const path of ['settings','namespace','flags'])test('native deletion of '+path+' preserves deletion semantics',async()=>{const e=setup(),del=new operators.ForcedDeletion();const change={flags:path==='flags'?del:{'dice-so-nice':path==='namespace'?del:{settings:del}}};await e.game.user.update(change);assert.deepEqual(plain(e.game.user.flags['dice-so-nice'].settings),quality);assert.equal(e.game.user.flags.world?.keep,path==='flags'?undefined:1);assert.equal(Boolean(e.game.user.flags['dice-so-nice'].appearance),path==='settings');});
for(const path of ['settings','namespace','flags'])test('native replacement of '+path+' preserves replacement semantics',async()=>{const e=setup(),value={soundsVolume:.4,glow:true};const change={flags:path==='flags'?new operators.ForcedReplacement({'dice-so-nice':{settings:value}}):{'dice-so-nice':path==='namespace'?new operators.ForcedReplacement({settings:value}):{settings:new operators.ForcedReplacement(value)}}};await e.game.user.update(change);assert.deepEqual(plain(e.game.user.flags['dice-so-nice'].settings),{...quality,soundsVolume:.4});assert.equal(e.game.user.flags.world?.keep,path==='flags'?undefined:1);});
test('legacy deletion preserves only locked fields under the deleted path',async()=>{const e=setup();await e.game.user.update({flags:{'dice-so-nice':{'-=settings':null}}});assert.deepEqual(plain(e.game.user.flags['dice-so-nice'].settings),quality);});
test('ready only persists differing locked fields and never rewrites rules',async()=>{const e=setup(),before=structuredClone(e.rules);await e.call('ready');checkQuality(e.game.user.flags['dice-so-nice'].settings);assert.equal(e.game.user.flags['dice-so-nice'].settings.soundsVolume,.7);assert.deepEqual(e.rules,before);assert.equal(e.updates.length,1);});
test('already matching flags and unrelated user edits need no quality write',async()=>{const e=setup();Object.assign(e.game.user.flags['dice-so-nice'].settings,quality);await e.call('ready');assert.equal(e.updates.length,0);const change={name:'new'};await e.call('preUpdateUser',e.game.user,change);assert.deepEqual(change,{name:'new'});});
for(const[name,opts]of Object.entries({GM:{gm:true},soft:{soft:true},disabled:{sync:false},inactiveBBMM:{bbmmActive:false},inactiveDsN:{dsnActive:false},unknownBBMM:{bbmmVersion:'2.0.0'},unknownDsN:{dsnVersion:'7.0.0'},unknownCore:{generation:15}}))test(name+' retains native flags and parsed input',async()=>{const e=setup(opts);assert.equal(e.game.user.getFlag('dice-so-nice','settings').glow,true);assert.equal(e.makeApp().parseInputs({glow:true}).glow,true);const change={flags:{'dice-so-nice':{settings:{glow:true}}}},before=structuredClone(change);await e.call('preUpdateUser',e.game.user,change);assert.deepEqual(change,before);});
test('other users and Actor/world-role dialogs remain unaffected',()=>{const e=setup();assert.equal(new e.User('other',false).getFlag('dice-so-nice','settings').glow,true);const app=e.makeApp();app.isUser=false;assert.equal(app.parseInputs({glow:true}).glow,true);});
test('unlocking and invalid rule values immediately stop enforcing those fields',()=>{const e=setup();delete e.rules['bbmm.dsnQuality.glow'];e.rules['bbmm.dsnQuality.shadowQuality'].value='invalid';e.rules['bbmm.dsnQuality.useHighDPI'].value='false';const got=e.context.DsnSettings.CONFIG();assert.equal(got.glow,true);assert.equal(got.shadowQuality,'high');assert.equal(got.useHighDPI,true);assert.equal(got.advancedGlass,false);});
test('the form disables only locked quality controls',async()=>{const e=setup(),fields=Object.fromEntries([...Object.keys(quality),'soundsVolume'].map(k=>[k,{disabled:false}]));const root={querySelectorAll:s=>[fields[s.match(/name="([^"]+)"/)[1]]].filter(Boolean)};await e.call('renderDiceConfig',e.makeApp(),root);for(const k of Object.keys(quality))assert.equal(fields[k].disabled,true);assert.equal(fields.soundsVolume.disabled,false);});
for(const key of ['parseInputs','_updateObject','_clearUserRecord','_prepareContext','getShowcaseAppearance','onApply','onReset'])test('unknown native '+key+' fails closed without partial wrappers',()=>{const e=setup({},e=>{e.context.DiceConfig.prototype[key]=function changed(){};});assert.equal(e.result?.status,'unsupported-source');assert.equal(e.wrappers.size,0);assert.equal(e.game.user.getFlag('dice-so-nice','settings').glow,true);});
test('an existing adapter setting is left intact and prevents double installation',()=>{const e=setup({},e=>e.registry.set('bbmm.dsnQuality.glow',{scope:'client',type:Boolean,default:true}));assert.equal(e.result?.status,'settings-conflict');assert.equal(e.registry.get('bbmm.dsnQuality.glow').default,true);assert.equal(e.wrappers.size,0);});
test('repeated install and registration do not layer wrappers; restore retains a later foreign method',()=>{const e=setup();assert.equal(e.register()?.status,'already-registered');assert.equal(e.install()?.status,'already-installed');const foreign=function(){return{glow:true};};e.context.DiceConfig.prototype.parseInputs=foreign;e.result.restore();assert.equal(e.context.DiceConfig.prototype.parseInputs,foreign);assert.equal(e.game.user.getFlag('dice-so-nice','settings').glow,true);assert.equal(e.wrappers.size,0);});
test('later source replacement retires the bridge before another read or write',async()=>{const e=setup();e.context.DiceConfig.prototype._updateObject=function changed(){};assert.equal(e.game.user.getFlag('dice-so-nice','settings').glow,true);const change={flags:{'dice-so-nice':{settings:{glow:true}}}};await e.call('preUpdateUser',e.game.user,change);assert.deepEqual(change,{flags:{'dice-so-nice':{settings:{glow:true}}}});});
const checkFactory=f=>checkQuality({...f,antialiasing:f.aa});
test('native preset preview clamps factory, showcase input and overwritten disabled controls',async()=>{const e=setup(),app=e.makeApp();app.onApply();await e.flush();checkFactory(e.factory);checkQuality(app.showcaseView.last);assert.equal(e.factory.realisticLighting,true);assert.equal(e.controls.glow.checked,false);assert.equal(e.updates.length,0);});
test('native reset and immediate render keep shared factory locked before any save',async()=>{const e=setup(),app=e.makeApp();app.onReset();const data=await app.renderPromise;checkFactory(e.factory);checkQuality(data);assert.equal(e.updates.length,0);});
test('preview is covered before the asynchronous diceSoNiceReady notification',async()=>{const e=setup({},e=>{delete e.game.dice3d.DiceFactory;delete e.game.dice3d.box;});e.game.dice3d.DiceFactory=e.factory;e.game.dice3d.box={dicefactory:e.factory};const app=e.makeApp();app.reset=true;await app._prepareContext({});checkFactory(e.factory);});
test('factory clamp preserves receiver, return, additional args and original argument',async()=>{const e=setup();await e.call('diceSoNiceReady',e.game.dice3d);const options={...high,bumpMapping:true,ambiance:'foyer_1k'},before=structuredClone(options);assert.equal(e.factory.setQualitySettings(options,'extra'),undefined);checkFactory(e.factory);assert.deepEqual(options,before);assert.equal(e.factory.realisticLighting,true);const fresh=new e.context.DiceFactory();fresh.setQualitySettings(options);checkFactory(fresh);});
test('unlocked and GM factory calls keep native quality',async()=>{for(const opts of [{gm:true},{sync:false},{soft:true}]){const e=setup(opts);await e.call('diceSoNiceReady',e.game.dice3d);e.factory.setQualitySettings(high);assert.equal(e.factory.glow,true);}});
test('factory replacement retires all enforcement and restore preserves foreign method',async()=>{const e=setup();await e.call('diceSoNiceReady',e.game.dice3d);const foreign=function(){};Object.getPrototypeOf(e.factory).setQualitySettings=foreign;assert.equal(e.game.user.getFlag('dice-so-nice','settings').glow,true);e.result.restore();assert.equal(Object.getPrototypeOf(e.factory).setQualitySettings,foreign);});
test('an unknown initial factory leaves every lock path inactive',()=>{const e=setup({},e=>{Object.getPrototypeOf(e.factory).setQualitySettings=function foreign(){};});assert.equal(e.result.status,'unsupported-factory-source');assert.equal(e.game.user.getFlag('dice-so-nice','settings').glow,true);assert.equal(e.makeApp().parseInputs(high).glow,true);});
test('a later own factory override retires the bridge without overwriting that override',()=>{const e=setup(),foreign=function(){};e.factory.setQualitySettings=foreign;assert.equal(e.game.user.getFlag('dice-so-nice','settings').glow,true);e.result.restore();assert.equal(e.factory.setQualitySettings,foreign);});
