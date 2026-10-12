import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {SALUBRIOUS_SOURCE,salubriousFeat,treatmentTiers} from '../scripts/salubrious-kiss-rules.mjs';

const primary=process.env.PF2E_MANUAL_POOL_BATCH_SOURCE??process.env.FVTT_PF2E_BUNDLE;
assert.ok(primary,'PF2E_MANUAL_POOL_BATCH_SOURCE or FVTT_PF2E_BUNDLE is required');
const bytes=fs.readFileSync(primary);
assert.equal(createHash('sha256').update(bytes).digest('hex'),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');
const text=bytes.toString(),start=text.indexOf('function suppressFeats(e) {'),end=text.indexOf('\n}',start)+2;
assert.ok(start>=0&&end>start);
const nativeSuppress=vm.runInNewContext(text.slice(start,end)+';suppressFeats');
function fixture(){
 const actor={items:new Map(),skills:{occultism:{rank:3,proficient:true}}};
 const item={id:'feat',type:'feat',sourceId:SALUBRIOUS_SOURCE,actor,flags:{pf2e:{itemGrants:{}}},isOfType:kind=>kind==='feat'};
 actor.items.set(item.id,item);return {actor,item};
}
const eligible=actor=>!!salubriousFeat(actor);
test('Salubrious respects the actual native suppressed field',()=>{const f=fixture();assert.equal(eligible(f.actor),true);nativeSuppress([f.item]);assert.equal(f.item.suppressed,true);assert.equal(eligible(f.actor),false);assert.throws(()=>treatmentTiers(f.actor));});
for(const alias of ['isSuppressed','system'])test(`Salubrious retains ${alias} suppression compatibility`,()=>{const f=fixture();if(alias==='system')f.item.system={suppressed:true};else f.item.isSuppressed=true;assert.equal(eligible(f.actor),false);});
test('Salubrious may use a separate active copy of the exact source',()=>{const f=fixture();nativeSuppress([f.item]);const active={...f.item,id:'active',suppressed:false,flags:{pf2e:{itemGrants:{}}}};f.actor.items.set(active.id,active);assert.equal(eligible(f.actor),true);assert.equal(salubriousFeat(f.actor),active);assert.equal(treatmentTiers(f.actor).length,3);});
test('Salubrious does not accept an active unrelated source',()=>{const f=fixture();nativeSuppress([f.item]);f.actor.items.set('other',{...f.item,id:'other',sourceId:'Compendium.other.Item.other',suppressed:false});assert.equal(eligible(f.actor),false);});
