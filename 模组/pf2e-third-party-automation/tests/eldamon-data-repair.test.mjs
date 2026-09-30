import test from 'node:test';
import assert from 'node:assert/strict';
import {buildEldamonDataRepairs,createEldamonDataRepair} from '../scripts/eldamon-data-repair.mjs';

const bz='Compendium.battlezoo-eldamon-pf2e.';
const sources={powers:bz+'eldamon-features.Item.naawsnBug9EOpzfN',electricity:bz+'eldamon-features.Item.9KtNlRXeuxZSoVaI',resistance:bz+'feats.Item.5pbln7VjWjAIxLZT',surge:bz+'powers.Item.veFrnrxYjlqca13w',voltage:bz+'powers.Item.9bElF2uVf5FCJtb9',siphon:bz+'actions.Item.4w72ljp4eLBqeZB2',widen:bz+'feats.Item.3YasBiZw3N96rdUW',wraps:'Compendium.pf2e.equipment-srd.Item.FNDq4NFSN0g2HKWO'};
function item(id,key,system={}){return {id,_id:id,name:'任意名称',_stats:{compendiumSource:sources[key]},system:{rules:[],traits:{value:[]},...system},flags:{}};}
function fixture(){return {uuid:'Actor.test',type:'character',items:[item('p','powers'),item('e','electricity'),item('r','resistance'),item('w','wraps',{runes:{potency:2},equipped:{carryType:'worn',invested:true}}),item('s','surge',{description:{value:'<p>@Damage[(1+@actor.level)d4[electricity]]{normal}; @Damage[(1+@actor.level)d4[electricity]]{d8s}</p>'}}),item('v','voltage',{traits:{value:['concentrate','electricity','magical']}}),item('x','siphon',{traits:{value:['concentrate']}}),item('y','widen',{traits:{value:['concentrate']}})]};}
function apply(actor,updates){for(const update of updates){const i=actor.items.find(i=>i.id===update._id);for(const [path,value] of Object.entries(update)){if(path==='_id')continue;const keys=path.split('.');let obj=i;for(const key of keys.slice(0,-1))obj=obj[key]??=( {} );obj[keys.at(-1)]=structuredClone(value);}}}

test('repairs actual native rules, the broken discharge formula and source traits without changing the input',()=>{
 const a=fixture(),before=structuredClone(a),updates=buildEldamonDataRepairs(a);assert.deepEqual(a,before);apply(a,updates);
 assert.deepEqual(a.items.find(i=>i.id==='w').system.rules,[{key:'FlatModifier',selector:'eldamon-power-attack',value:'@item.system.runes.potency',type:'item'}]);
 assert.deepEqual(a.items.find(i=>i.id==='r').system.rules,[{key:'Resistance',type:'electricity',value:'floor(@actor.level / 2)',predicate:['feature:electricity-element']}]);
 assert.match(a.items.find(i=>i.id==='s').system.description.value,/d4\[electricity\]\]\{normal\}/);assert.match(a.items.find(i=>i.id==='s').system.description.value,/d8\[electricity\]\]\{d8s\}/);
 assert.deepEqual(a.items.find(i=>i.id==='v').system.traits.value,['concentrate','electricity','magical','refresh']);
 for(const id of ['x','y'])assert.deepEqual(a.items.find(i=>i.id===id).system.traits.value,['concentrate','elemental-avatar','metapower']);
 assert.equal(a.items.find(i=>i.id==='w').system.runes.potency,2);
 assert.equal(buildEldamonDataRepairs(a).length,0);
});

test('unrelated names/sources and non-character documents do not acquire fixes',()=>{
 const a=fixture();for(const i of a.items)i._stats.compendiumSource='Compendium.other.items.Item.'+i.id;
 assert.deepEqual(buildEldamonDataRepairs(a),[]);assert.deepEqual(buildEldamonDataRepairs({...fixture(),type:'npc'}),[]);
});

test('missing electricity or elemental powers never grants their corresponding passive rules',()=>{
 const a=fixture();a.items=a.items.filter(i=>!['e','p'].includes(i.id));apply(a,buildEldamonDataRepairs(a));
 assert.deepEqual(a.items.find(i=>i.id==='r').system.rules,[]);assert.deepEqual(a.items.find(i=>i.id==='w').system.rules,[]);
});

test('existing rules, traits, unrelated damage and custom formula variants are preserved',()=>{
 const a=fixture(),w=a.items.find(i=>i.id==='w'),r=a.items.find(i=>i.id==='r'),s=a.items.find(i=>i.id==='s');
 w.system.rules=[{key:'FlatModifier',selector:['eldamon-power-attack'],type:'item',value:'@item.system.runes.potency'}, {key:'RollOption',domain:'all',option:'custom'}];
 r.system.rules=[{key:'Resistance',type:['electricity'],value:7}];
 s.system.description.value='@Damage[10d4[electricity]]{d8} @Damage[(1+@actor.level)d4[fire]]{d8} @Damage[(1+@actor.level)d8[electricity]]{d8}';
 const before=structuredClone({w:w.system.rules,r:r.system.rules,s:s.system.description.value});apply(a,buildEldamonDataRepairs(a));
 assert.deepEqual(w.system.rules,before.w);assert.deepEqual(r.system.rules,before.r);assert.equal(s.system.description.value,before.s);
});

test('stores only changed fields once for local recovery',()=>{
 const a=fixture();apply(a,buildEldamonDataRepairs(a));const flags=a.items.find(i=>i.id==='s').flags['pf2e-third-party-automation'].eldamonDataRepair;
 assert.match(flags.before['system.description.value'],/d4\[electricity\]\]\{d8s\}/);assert.equal(flags.version,1);
 assert.equal(buildEldamonDataRepairs(a).length,0);
});

test('maintenance runs only on active GM, re-reads inside a queue, and does not repeat successful updates',async()=>{
 const a=fixture(),gm={id:'gm'},other={id:'other'},game={user:other,users:{activeGM:gm}};let calls=0;
 a.updateEmbeddedDocuments=async(type,updates)=>{assert.equal(type,'Item');calls++;await new Promise(r=>setTimeout(r,5));apply(a,updates);};
 const provider=createEldamonDataRepair({game});await provider.maintain(a);assert.equal(calls,0);game.user=gm;
 await Promise.all([provider.maintain(a),provider.maintain(a)]);assert.equal(calls,1);assert.deepEqual(buildEldamonDataRepairs(a),[]);
});
