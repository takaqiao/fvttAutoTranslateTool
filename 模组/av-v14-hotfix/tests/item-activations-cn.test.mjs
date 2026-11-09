import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {correctActivation,installItemActivationsCnPatch,traitLabels} from '../scripts/patches/item-activations-cn.mjs';

const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/item-activations-14.3.3.json',import.meta.url),'utf8'));
const MODULE='pf2e-item-activations';
// CONFIG.PF2E.actionTraits once PF2e has registered the module's homebrew feat traits.
const actionTraits={...fixture.actionTraits,
  ...Object.fromEntries(Object.entries(fixture.homebrew).map(([slug,value])=>[slug,typeof value==='string'?value:value.label]))};

const setProperty=(object,path,value)=>{
  const keys=path.split('.'),last=keys.pop();
  let target=object;
  for(const key of keys)target=target[key]??={};
  target[last]=value;
};
// Objects merge, everything else is replaced, as Document#updateSource does for these fields.
const merge=(target,changes)=>{
  for(const [key,value] of Object.entries(changes)){
    if(value&&typeof value==='object'&&!Array.isArray(value)&&target[key]&&typeof target[key]==='object')merge(target[key],value);
    else target[key]=value;
  }
  return target;
};
const lookup=(lang,key)=>{
  if(key.startsWith(`${MODULE}.`))return key.slice(MODULE.length+1).split('.').reduce((node,part)=>node?.[part],fixture.languages[lang])??key;
  return fixture.labels[lang][key]??key;
};

// The module's own generateActivations, augmentAction and deactivateAction (14.3.3), with its ready
// hook run for the given language, and PF2e 8.7.0's sluggify.
function upstream(lang){
  const ready=[];
  const context=vm.createContext({
    foundry:{utils:{setProperty}},
    Hooks:{on:(name,fn)=>{if(name==='ready')ready.push(fn);}},
    game:{i18n:{format:key=>lookup(lang,key)},settings:{get:()=>false}},
    ErrorPF2e:message=>new Error(message)
  });
  const strip=source=>source.replace(/^import[\s\S]*?from\s+"[^"]+";\s*$/gm,'').replace(/^export /gm,'');
  vm.runInContext(`${fixture.sluggify}
const MODULE_ID=${JSON.stringify(MODULE)};const TEXT={};
${strip(fixture.sources.specific)}
${strip(fixture.sources.misc)}
${strip(fixture.sources.activate)}
${strip(fixture.sources.onCreate)}
${strip(fixture.sources.generateActivation)}
game.pf2e={system:{sluggify}};
globalThis.api={generateActivations,augmentAction,deactivateAction,sluggify};`,context);
  for(const fn of ready)fn();
  return context.api;
}
const modules={cn:upstream('cn'),en:upstream('en')};

function grantingItem(sample,{invested=true}={}){
  const [name]=sample.item.split(/ (?=[A-Z])/);
  const data={_id:'grantingItem0001',name,type:'equipment',img:'icons/item.webp',
    system:{slug:sample.slug,description:{value:sample.description},traits:{value:invested?['invested','magical']:['magical']}}};
  return {...data,slug:sample.slug,link:`@UUID[Actor.a.Item.${data._id}]{${name}}`,toObject:()=>structuredClone(data)};
}

// What the module passes to createEmbeddedDocuments, as the Item the preCreateItem hook receives: PF2e's
// LaxArrayField has already dropped trait values that are not action traits, and ItemPF2e#_preCreate
// has sorted the rest.
function created(sample,{inactive=false,invested}={}){
  const {generateActivations,augmentAction,deactivateAction}=modules[sample.lang];
  const item=grantingItem(sample,{invested});
  return generateActivations(item).map(action=>augmentAction(action,item)).map(action=>inactive?deactivateAction(action):action).map(data=>{
    const source=structuredClone(data);
    source.system.traits.value=source.system.traits.value.filter(trait=>Object.hasOwn(actionTraits,trait)).sort();
    return {_source:source,updates:[],updateSource(changes){this.updates.push(changes);merge(this._source,structuredClone(changes));}};
  });
}

function runtime({lang='cn',active=true}={}){
  const hooks=[],reports=[];
  const value={
    game:{system:{id:'pf2e'},modules:{get:id=>id===MODULE?{active}:undefined},i18n:{localize:key=>lookup(lang,key)}},
    CONFIG:{PF2E:{actionTraits}},
    Hooks:{on:(name,fn)=>{hooks.push({name,fn});return hooks.length;}}
  };
  return {runtime:value,hooks,reports,report:entry=>reports.push(entry),
    create(docs){for(const doc of docs)for(const {name,fn} of hooks)if(name==='preCreateItem')fn(doc,{},{},'user');return docs;}};
}
const sample=slug=>fixture.samples.find(entry=>entry.slug===slug&&entry.lang==='cn');
const english=slug=>fixture.samples.find(entry=>entry.slug===slug&&entry.lang==='en');
function install(options){
  const env=runtime(options);
  env.result=installItemActivationsCnPatch({runtime:env.runtime,report:env.report});
  return env;
}

test('the module itself leaves the extra dash and drops the Chinese trait labels',()=>{
  const [first]=created(sample('empyreal-dragonbreath-greatsword'));
  assert.equal(first._source.name,'启动项: —释放龙息');
  assert.deepEqual(first._source.system.traits.value,['activation','invested']);
  const [thurible]=created(sample('thurible-of-revelation-lesser'));
  assert.equal(thurible._source.name,'启动项: —焚香');
  assert.deepEqual(thurible._source.system.traits.value,['activation','invested']);
});

test('a double dash is cut to the name and full-width trait labels become traits',()=>{
  const env=install();
  assert.equal(env.result.status,'installed');
  const [breath,surge]=env.create(created(sample('empyreal-dragonbreath-greatsword')));
  assert.equal(breath._source.name,'启动项: 释放龙息');
  assert.equal(surge._source.name,'启动项: 喷涌');
  assert.deepEqual(breath._source.system.traits.value,['activation','divine','fire','invested','spirit']);
  const [light]=env.create(created(sample('wayfinder')));
  assert.equal(light._source.name,'启动项: 照亮');
  assert.deepEqual(light._source.system.traits.value,['activation','concentrate','invested']);
  const [glow]=env.create(created(sample('shining-symbol')));
  assert.equal(glow._source.name,'启动项: 精魄辉光');
  assert.deepEqual(glow._source.system.traits.value,['activation','concentrate','invested','light','revelation']);
  assert.deepEqual(env.result.stats,{renamed:4,traitsFilled:4,labelsDropped:0});
});

test('a single dash keeps its name; traits are still filled, including after a name gloss',()=>{
  const env=install();
  const [sigil]=env.create(created(sample('ring-of-sigils')));
  assert.equal(sigil._source.name,'启动项: 魔印追迹');
  assert.deepEqual(sigil._source.system.traits.value,['activation','concentrate','detection','invested']);
  const [stoke]=env.create(created(sample('smoking-sword')));
  assert.equal(stoke._source.name,'启动项: 燃火（Stoke Flames）');
  assert.deepEqual(stoke._source.system.traits.value,['activation','concentrate','invested']);
  const [scry,clairvoyance]=env.create(created(sample('crystal-ball-clear-quartz')));
  assert.deepEqual(scry._source.system.traits.value,['activation','concentrate','invested','manipulate']);
  assert.deepEqual(clairvoyance._source.system.traits.value,['activation','concentrate','invested','manipulate']);
  for(const doc of [sigil,stoke,scry,clairvoyance])assert.equal(doc.updates.some(change=>'name' in change),false);
  assert.deepEqual(env.result.stats,{renamed:0,traitsFilled:4,labelsDropped:0});
});

test('a half-width group of Chinese labels is read; no group leaves the traits alone',()=>{
  const env=install();
  const [incense]=env.create(created(sample('thurible-of-revelation-lesser')));
  assert.equal(incense._source.name,'启动项: 焚香');
  assert.deepEqual(incense._source.system.traits.value,['activation','invested','manipulate']);
  const [alert]=env.create(created(sample('name-pendant'),{invested:false}));
  assert.equal(alert._source.name,'启动项: 警示上级 Alert Superior Officer');
  assert.deepEqual(alert._source.system.traits.value,['activation']);
  assert.deepEqual(alert.updates,[]);
});

test('a label without a trait of that name is dropped and counted',()=>{
  const env=install();
  const [link]=env.create(created(sample('wonton-spirit-fan')));
  assert.deepEqual(link._source.system.traits.value,['activation','invested']);
  assert.deepEqual(link.updates,[]);
  assert.deepEqual(env.result.stats,{renamed:0,traitsFilled:0,labelsDropped:1});
  const mixed=correctActivation({name:'启动项: —试',system:{description:{value:'<p><strong>启动——试</strong>（想象，专注）</p>'},traits:{value:['activation']}}},
    {activateText:'启动',activationText:'启动项',actionTraits,localize:key=>lookup('cn',key)});
  assert.deepEqual(mixed.changes,{name:'启动项: 试',system:{traits:{value:['activation','concentrate']}}});
  assert.equal(mixed.dropped,1);
});

test('the module marker on an inactive action stays in front of the name',()=>{
  const env=install();
  const [breath]=env.create(created(sample('empyreal-dragonbreath-greatsword'),{inactive:true}));
  assert.equal(breath._source.name,'[X] 启动项: 释放龙息');
  assert.equal(breath._source.flags[MODULE].enabled,false);
});

test('slug and flags stay as the module set them, so earlier and later actions still match',()=>{
  const env=install(),{sluggify}=modules.cn;
  for(const entry of fixture.samples.filter(entry=>entry.lang==='cn')){
    const before=created(entry),after=env.create(created(entry));
    after.forEach((doc,index)=>{
      // The module skips creating an action whose slug the actor already has, and finds its actions
      // by grantedBy._id: both must be the same whether or not the action was corrected.
      assert.equal(doc._source.system.slug,before[index]._source.system.slug);
      assert.equal(sluggify(doc._source.name),doc._source.system.slug);
      assert.deepEqual(doc._source.flags,before[index]._source.flags);
      assert.equal(doc._source.flags[MODULE].grantedBy._id,'grantingItem0001');
      for(const change of doc.updates)assert.deepEqual(Object.keys(change).filter(key=>key!=='name'&&key!=='system'),[]);
      for(const change of doc.updates)if(change.system)assert.deepEqual(Object.keys(change.system),['traits']);
    });
  }
});

test('English descriptions are left as the module parses them',()=>{
  const env=install({lang:'en'});
  const [light]=env.create(created(english('wayfinder')));
  assert.equal(light._source.name,'Activation: Light');
  assert.deepEqual(light._source.system.traits.value,['activation','concentrate','invested']);
  const [track]=env.create(created(english('ring-of-sigils')));
  assert.equal(track._source.name,'Activation: Track Sigil');
  // 14.3.3 keeps the space after the comma, so PF2e drops " detection".
  assert.deepEqual(track._source.system.traits.value,['activation','concentrate','invested']);
  assert.deepEqual([light.updates,track.updates],[[],[]]);
  assert.deepEqual(traitLabels('<p><strong>Activate—Light</strong> (concentrate)'),[]);
});

test('actions the module did not generate are not touched',()=>{
  const env=install();
  const [generated]=created(sample('empyreal-dragonbreath-greatsword'));
  const own=structuredClone(generated._source);
  delete own.flags;
  const foreign={_source:own,updates:[],updateSource(changes){this.updates.push(changes);}};
  const effect={_source:{...structuredClone(generated._source),type:'effect'},updates:[],updateSource(changes){this.updates.push(changes);}};
  const named={_source:{...structuredClone(generated._source),flags:{[MODULE]:{grantedBy:'grantingItem0001'}}},updates:[],updateSource(changes){this.updates.push(changes);}};
  env.create([foreign,effect,named]);
  assert.deepEqual([foreign.updates,effect.updates,named.updates],[[],[],[]]);
  // An action of the module's own list arrives with its own traits; those are kept as they are.
  const [listed]=created(sample('wayfinder'));
  listed._source.system.traits.value=['activation','light'];
  env.create([listed]);
  assert.equal(listed._source.name,'启动项: 照亮');
  assert.deepEqual(listed._source.system.traits.value,['activation','light']);
});

test('install is skipped without the module or PF2e, and a failure keeps the action as built',()=>{
  assert.deepEqual(install({active:false}).result,{feature:'itemActivationsCn',status:'skipped',reason:'module-inactive'});
  const other=runtime();
  other.runtime.game.system.id='dnd5e';
  assert.equal(installItemActivationsCnPatch({runtime:other.runtime}).status,'unsupported-system');
  const env=install();
  assert.equal(installItemActivationsCnPatch({runtime:env.runtime}).status,'already-installed');
  const [breath]=created(sample('empyreal-dragonbreath-greatsword'));
  breath.updateSource=()=>{throw new Error('read-only');};
  env.create([breath,breath]);
  assert.equal(breath._source.name,'启动项: —释放龙息');
  const failures=env.reports.filter(entry=>entry.status==='failed');
  assert.equal(failures.length,1);
  assert.match(failures[0].detail,/read-only/);
});
