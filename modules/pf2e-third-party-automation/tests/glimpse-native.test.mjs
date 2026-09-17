import test from 'node:test';
import assert from 'node:assert/strict';
import {compileGlimpseResistance,withGlimpseResistance,repentParams,glimpseMarker} from '../scripts/glimpse-native.mjs';
const template=()=>({type:'effect',name:'Native',system:{rules:[{key:'Resistance',type:'all-damage',value:'@item.origin.level+2'}],context:null}});
test('native resistance keeps all-damage semantics and is restricted to its actual call nonce',async()=>{
 const native={type:'all-damage',value:7,exceptions:[],doubleVs:[],test:o=>!o.has('bypass'),getDoubledValue:()=>7};let compiled;
 const actor={attributes:{resistances:[]},getContextualClone(options,effects){compiled=effects[0];return {attributes:{resistances:[native]}}}};
 const r=compileGlimpseResistance({actor,template:template(),champion:{uuid:'Actor.champ',level:5},ability:{uuid:'Actor.champ.Item.g'},nonce:'one'});
 assert.equal(compiled.system.context.origin.actor,'Actor.champ');assert.equal(r.type,'all-damage');assert.equal(r.test(new Set([glimpseMarker('one')])),true);assert.equal(r.test(new Set([glimpseMarker('two')])),false);assert.equal(r.test(new Set([glimpseMarker('one'),'bypass'])),false);
 await assert.rejects(withGlimpseResistance(actor,r,async()=>{assert.deepEqual(actor.attributes.resistances,[r]);throw Error('native failed')}),/native failed/);assert.deepEqual(actor.attributes.resistances,[]);
});
test('exact temporary object is removed from replaced arrays without removing existing resistance',async()=>{
 const old={value:12},r={value:7},actor={attributes:{resistances:[old]}};const original=actor.attributes.resistances;
 await withGlimpseResistance(actor,r,async()=>{actor.attributes.resistances=[old,r]});assert.deepEqual(original,[old]);assert.deepEqual(actor.attributes.resistances,[old]);
});
test('unknown template / failed original scaling stops before native and preserves original template',()=>{
 const original=template(),actor={getContextualClone:()=>({attributes:{resistances:[{type:'all-damage',value:6,test(){},getDoubledValue(){}}]}})};
 assert.throws(()=>compileGlimpseResistance({actor,template:original,champion:{level:5},ability:{},nonce:'x'}),/抗力/);assert.equal(original.system.context,null);original.system.rules.push({key:'FlatModifier'});assert.throws(()=>compileGlimpseResistance({actor,template:original,champion:{level:5},ability:{},nonce:'x'}),/原生/);
});
test('Repent uses exactly native final zero semantics with source token/item retained',()=>{const token={},item={},damage={instances:[{persistent:true}]},p=repentParams({damage,token,item,shieldBlockRequest:true,rollOptions:new Set(['original'])},'once');assert.equal(p.damage,0);assert.equal(p.final,true);assert.equal(p.shieldBlockRequest,false);assert.equal(p.token,token);assert.equal(p.item,item);assert.deepEqual([...p.rollOptions],['original',glimpseMarker('once')]);assert.equal(damage.instances.length,1)});
