import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {createDurationFastPath,installDurationPatch} from '../scripts/patches/duration.mjs';
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/duration-core-14.367.json',import.meta.url),'utf8'));
const native=Intl.DurationFormat;
function setup(){
 let constructions=0;
 const game={i18n:{lang:'en'}},localIntl={DurationFormat:class extends native{constructor(...args){super(...args);constructions++;}}};
 function *objectEntries(obj){for(const key in obj){if(Object.hasOwn(obj,key))yield [key,obj[key]];}}
 function iterateEntries(obj){if(obj==null)throw new TypeError('Cannot convert undefined or null to object');return Iterator.from(objectEntries(obj));}
 const CalendarData={_DURATION_FORMAT_UNITS:new Set(['year','month','day','hour','minute','second'])};
 const original=Function('CalendarData','iterateEntries','game','Intl',`return ({${fixture.source}}).formatDuration`)(CalendarData,iterateEntries,game,localIntl);
 const wrapped=createDurationFastPath(original,{game,Intl:localIntl});
 return {game,localIntl,original,wrapped,get constructions(){return constructions;},call(fn,components={minute:5,second:3},options={maxTerms:2,style:'narrow'}){return fn.call(CalendarData,null,components,options);}};
}
test('fixture is the unmodified 14.367 method',()=>assert.equal(createHash('sha256').update(fixture.source).digest('hex'),fixture.sha256));
test('reuses formatter across real changing durations without caching text',()=>{
 const s=setup();const before=[];for(let i=0;i<20;i++)before.push(s.call(s.original,{minute:i,second:i%7,leapYear:false}));
 const start=s.constructions;const after=[];for(let i=0;i<20;i++)after.push(s.call(s.wrapped,{minute:i,second:i%7,leapYear:false}));
 assert.deepEqual(after,before);assert.equal(s.constructions-start,2);
});
test('language change is immediate and uses only one current cache entry',()=>{
 const s=setup();for(const lang of ['en','zh-CN','fr','en']){s.game.i18n.lang=lang;const expected=s.call(s.original);const n=s.constructions;assert.equal(s.call(s.wrapped),expected);assert.equal(s.call(s.wrapped),expected);assert.equal(s.constructions-n,2);}
});
test('explicit formatter and custom styles/options retain original path',()=>{
 const s=setup();s.call(s.wrapped);let calls=0;const formatter={format:()=>{calls++;return 'custom';}};
 assert.equal(s.call(s.wrapped,{}, {maxTerms:2,style:'narrow',formatter}),'custom');assert.equal(calls,1);
 for(const opts of [{style:'long'}, {maxTerms:3,style:'narrow'},{maxTerms:2,style:'narrow',secondsDisplay:'always'},Object.create({formatter})])assert.equal(s.call(s.wrapped,{minute:3},opts),s.call(s.original,{minute:3},opts));
});
test('getters execute only in original order and may change locale',()=>{
 const s=setup();s.call(s.wrapped);let count=0;
 const component={get minute(){count++;s.game.i18n.lang='fr';return 5;}};
 const result=s.call(s.wrapped,component);assert.equal(count,2);assert.equal(result,s.call(s.original,{minute:5}));
 let reads=0;const options={maxTerms:2,get style(){reads++;return 'narrow';}};
 assert.equal(s.call(s.wrapped,{minute:1},options),s.call(s.original,{minute:1}));assert.equal(reads,1);
});
test('nonfinite early return and invalid locale preserve constructor/error behavior',()=>{
 const s=setup();s.game.i18n.lang='bad_locale';assert.equal(s.call(s.wrapped,{day:Infinity}),'∞');assert.equal(s.constructions,0);
 assert.throws(()=>s.call(s.wrapped),RangeError);assert.equal(s.constructions,0);
 s.game.i18n.lang='en';s.call(s.wrapped);s.game.i18n.lang='bad_locale';assert.throws(()=>s.call(s.wrapped),RangeError);
});
test('changed constructor or format method immediately bypasses old formatter',()=>{
 const s=setup();s.call(s.wrapped);let calls=0;s.localIntl.DurationFormat=class{format(){calls++;return 'replacement';}};
 assert.equal(s.call(s.wrapped),'replacement');assert.equal(calls,1);
 const t=setup();t.call(t.wrapped);t.localIntl.DurationFormat.prototype.format=()=> 'changed';assert.equal(t.call(t.wrapped),'changed');
});
test('frozen/null-prototype input and options are never mutated',()=>{
 const s=setup();const c=Object.freeze(Object.assign(Object.create(null),{minute:9,second:1})),o=Object.freeze(Object.assign(Object.create(null),{maxTerms:2,style:'narrow'}));
 assert.equal(s.call(s.wrapped,c,o),s.call(s.original,c,o));assert.equal(s.call(s.wrapped,c,o),s.call(s.original,c,o));assert.deepEqual(Reflect.ownKeys(o),['maxTerms','style']);
});
test('symbol/accessor options and inherited components retain original behavior',()=>{
 const s=setup();s.call(s.wrapped);const sym=Symbol('x');const o={maxTerms:2,style:'narrow',[sym]:1};
 assert.equal(s.call(s.wrapped,{},o),s.call(s.original,{},o));const c=Object.assign(Object.create({second:50}),{minute:2});assert.equal(s.call(s.wrapped,c),s.call(s.original,c));
});
test('live format accessor is neither called by guards nor hidden by a stale cache',()=>{
 const s=setup();s.call(s.wrapped);let gets=0;Object.defineProperty(s.localIntl.DurationFormat.prototype,'format',{get(){gets++;return ()=> 'getter';},configurable:true});
 assert.equal(s.call(s.wrapped),'getter');assert.equal(gets,1);
});
test('warm-up failure cannot change a completed original call',()=>{
 const s=setup();let n=0;const original=s.original;const Constructor=s.localIntl.DurationFormat;
 s.localIntl.DurationFormat=class extends Constructor{constructor(...args){if(++n===2)throw Error('warm only');super(...args);}};
 const wrapper=createDurationFastPath(original,{game:s.game,Intl:s.localIntl});assert.equal(s.call(wrapper),'5m 3s');assert.equal(n,2);assert.equal(s.call(wrapper),'5m 3s');assert.equal(n,4);assert.equal(s.call(wrapper),'5m 3s');assert.equal(n,4);
});
test('pre-existing stateful format overrides keep fresh instances',()=>{
 const s=setup(),Constructor=s.localIntl.DurationFormat;
 s.localIntl.DurationFormat=class extends Constructor{format(){return String(this.calls=(this.calls??0)+1);}};
 const wrapped=createDurationFastPath(s.original,{game:s.game,Intl:s.localIntl});
 assert.equal(s.call(wrapped),'1');assert.equal(s.call(wrapped),'1');assert.equal(s.call(wrapped),'1');
});
test('a native-code marker in an ordinary function does not enable formatter reuse',()=>{
 const s=setup(),Constructor=s.localIntl.DurationFormat;
 s.localIntl.DurationFormat=class extends Constructor{format(){/* [native code] */return String(this.calls=(this.calls??0)+1);}};
 const wrapped=createDurationFastPath(s.original,{game:s.game,Intl:s.localIntl});
 assert.deepEqual([s.call(wrapped),s.call(wrapped),s.call(wrapped)],['1','1','1']);
});
test('restore leaves later descriptor edits and frozen targets intact',async()=>{
 for(const change of [target=>Object.freeze(target),target=>Object.defineProperty(target,'formatDuration',{enumerable:false})]){
  const s=setup(),target={formatDuration:s.original},runtime={game:{...s.game,version:'14.368'},Intl,foundry:{data:{CalendarData:target}}};
  const patch=await installDurationPatch({runtime});assert.equal(patch.status,'installed');change(target);const descriptor=Object.getOwnPropertyDescriptor(target,'formatDuration');
  assert.doesNotThrow(()=>patch.restore());assert.deepEqual(Object.getOwnPropertyDescriptor(target,'formatDuration'),descriptor);
 }
});
test('installer verifies source, retains descriptors, and restores only its own method',async()=>{
 const s=setup(),target={formatDuration:s.original};
 const runtime={game:{...s.game,version:'14.368'},Intl,foundry:{data:{CalendarData:target}}};
 const descriptor=Object.getOwnPropertyDescriptor(target,'formatDuration');const installed=await installDurationPatch({runtime});assert.equal(installed.status,'installed');assert.notEqual(target.formatDuration,s.original);installed.restore();assert.deepEqual(Object.getOwnPropertyDescriptor(target,'formatDuration'),descriptor);
 const again=await installDurationPatch({runtime});const later=()=> 'other';target.formatDuration=later;again.restore();assert.equal(target.formatDuration,later);
 assert.equal((await installDurationPatch({runtime})).status,'unsupported-source');
});
test('installer refuses unsupported core, runtime and changes during asynchronous verification',async()=>{
 const s=setup(),target={formatDuration:s.original};const runtime={game:{...s.game,version:'14.368'},Intl,foundry:{data:{CalendarData:target}}};
 runtime.game.version='15.0';assert.equal((await installDurationPatch({runtime})).status,'unsupported-core');runtime.game.version='14.368';
 const other=()=>{};assert.equal((await installDurationPatch({runtime,hash:async()=>{target.formatDuration=other;return fixture.sha256;}})).status,'source-changed-during-validation');assert.equal(target.formatDuration,other);
});
