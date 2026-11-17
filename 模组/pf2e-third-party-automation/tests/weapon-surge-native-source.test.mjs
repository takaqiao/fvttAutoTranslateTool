import test from 'node:test';
import assert from 'node:assert/strict';
import {createNextStrikeEffectFrame} from '../scripts/next-strike-effects.mjs';
import {validWeaponSurgeSnapshot,weaponSurgeTransientItems} from '../scripts/weapon-surge-snapshot.mjs';
import {createWeaponSurgeAutomation,prepareWeaponSurgeDamageSnapshotItems} from '../scripts/weapon-surge.mjs';

// PF2e persists a ChoiceSet choice only on its rule source. flags.pf2e.rulesSelections
// is filled in while the document prepares and flags.system is a getter for it, so
// toObject() and actor._source never carry the selection flag (live world 2026-11-17).
const ID='pf2e-third-party-automation',SOURCE='Compendium.pf2e.spell-effects.Item.qlz0sJIvqc0FdUdr';
const clone=structuredClone;
const dice=items=>items.filter(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice'));
function source(id='surge',rank=3,weapon='sword'){
 return {_id:id,name:'法术效果：激发武器 Weapon Surge (长剑 Longsword)',type:'effect',_stats:{compendiumSource:SOURCE,duplicateSource:null},
  flags:{babele:{translated:true,originalName:'Spell Effect: Weapon Surge'}},
  system:{slug:'spell-effect-weapon-surge',level:{value:rank},start:{value:329268,initiative:18},duration:{expiry:'turn-start',sustained:false,unit:'rounds',value:1},rules:[
   {key:'ChoiceSet',flag:'spellEffectWeaponSurge',choices:{ownedItems:true,predicate:['item:equipped'],types:['weapon']},prompt:'PF2E.SpecificRule.Prompt.Weapon',selection:weapon},
   {key:'FlatModifier',selector:'{item|flags.system.rulesSelections.spellEffectWeaponSurge}-attack',type:'status',value:1},
   {key:'DamageDice',selector:'{item|flags.system.rulesSelections.spellEffectWeaponSurge}-damage',damageType:'spirit',diceNumber:'ternary(gte(@item.level,9),3,ternary(gte(@item.level,5),2,1))',dieSize:'d6'},
   {key:'AdjustStrike',definition:['item:id:{item|flags.system.rulesSelections.spellEffectWeaponSurge}'],mode:'add',property:'traits',value:'sanctified'}]}};
}
/** A prepared document the way PF2e builds it from source. */
function prepare(data,actor){
 const flags=clone(data.flags??{});flags.pf2e={...flags.pf2e,rulesSelections:{...flags.pf2e?.rulesSelections}};
 for(const rule of data.system?.rules??[])if(rule.key==='ChoiceSet'&&rule.selection!==undefined)flags.pf2e.rulesSelections[rule.flag]=rule.selection;
 Object.defineProperty(flags,'system',{get:()=>flags.pf2e,enumerable:false});
 const item={...clone(data),flags,id:data._id,uuid:`${actor.uuid}.Item.${data._id}`,actor,_source:clone(data),isExpired:false,
  get sourceId(){return this._source._stats?.compendiumSource??null},toObject(){return clone(this._source)}};
 return item;
}
function fixture(){
 const writes=[],actor={uuid:'Actor.sAJeWzzdMROned3V',items:new Map(),_source:{items:[]},system:{actions:[]},
  async deleteEmbeddedDocuments(_type,ids){writes.push(ids);for(const id of ids)this.items.delete(id);this._source.items=this._source.items.filter(item=>!ids.includes(item._id));}};
 const add=data=>{const item=prepare(data,actor);actor.items.set(item.id,item);actor._source.items.push(clone(data));return item};
 add({_id:'sword',type:'weapon',system:{traits:{value:[]}}});add(source());
 const makeStrike=owner=>({type:'strike',item:{id:'sword',uuid:`${owner.uuid}.Item.sword`,actor:owner,type:'weapon',system:{traits:{value:[]}}},variants:[{roll:async()=>null}],
  damage:async params=>({items:owner._source.items,options:params.options}),critical:async params=>({items:owner._source.items,options:params.options})});
 const strike=makeStrike(actor),target={uuid:'Scene.s.Token.target'};actor.system.actions.push(strike);
 const message={flags:{pf2e:{origin:{actor:actor.uuid,uuid:strike.item.uuid},context:{type:'attack-roll',outcome:'success',options:[],target:{token:target.uuid}}}}};
 actor.clone=function(changes){const copy={...actor,_source:clone(changes),items:new Map()};for(const data of changes.items)copy.items.set(data._id,prepare(data,copy));copy.system={actions:[makeStrike(copy)]};return copy};
 const begin=()=>createNextStrikeEffectFrame({actor,strike,target,consumeTumble:false});
 return {actor,strike,target,message,writes,add,begin};
}

test('a snapshot taken from a native-shaped Weapon Surge validates against its actor and weapon',()=>{
 const f=fixture(),record=f.begin().snapshot();
 assert.equal(record.effects.length,1);assert.equal(record.effects[0].flags.pf2e,undefined);
 assert.equal(validWeaponSurgeSnapshot(record,f.actor,f.strike.item),true);assert.equal(validWeaponSurgeSnapshot(record,f.actor),true);
});
test('the attack card holding a native-shaped snapshot consumes exactly that Weapon Surge',async()=>{
 const f=fixture(),frame=f.begin();f.message.flags.pf2e.context.weaponSurgeSnapshot=frame.snapshot();
 assert.equal(frame.capture(f.message),true);await frame.consume();assert.deepEqual(f.writes,[['surge']]);
});
for(const method of ['damage','critical'])test(`${method} from the saved card opens with one frozen Surge after it was consumed`,async()=>{
 const f=fixture(),frame=f.begin();f.message.flags.pf2e.context.weaponSurgeSnapshot=frame.snapshot();frame.capture(f.message);await frame.consume();
 const provider=createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});provider.wrapStrike(f.strike,f.actor);
 const result=await f.strike[method]({checkContext:f.message.flags.pf2e.context,options:['other']}),surges=dice(result.items);
 assert.equal(surges.length,1);assert.equal(surges[0].system.slug,'tpa-weapon-surge-frozen');assert.equal(surges[0].system.level.value,3);assert.equal(result.options.has('other'),true);
});
test('a lingering live Surge for the same weapon is replaced, not doubled, by the frozen copy',async()=>{
 const f=fixture(),record=f.begin().snapshot();f.message.flags.pf2e.context.weaponSurgeSnapshot=record;
 const provider=createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});provider.wrapStrike(f.strike,f.actor);
 const surges=dice((await f.strike.damage({checkContext:f.message.flags.pf2e.context})).items);
 assert.equal(surges.length,1);assert.equal(surges[0].system.slug,'tpa-weapon-surge-frozen');
 const expanded=dice(prepareWeaponSurgeDamageSnapshotItems(f.actor,weaponSurgeTransientItems(record)));
 assert.equal(expanded.length,1);assert.equal(expanded[0].flags[ID].weaponSurgeFrozen.weaponId,'sword');
});
test('a Surge bound to another weapon stays in the snapshot roller and out of the record',()=>{
 const f=fixture();f.add(source('bow-surge',5,'bow'));const record=f.begin().snapshot();
 assert.deepEqual(record.effects.map(effect=>effect._id),['surge']);
 const items=prepareWeaponSurgeDamageSnapshotItems(f.actor,weaponSurgeTransientItems(record));
 assert.ok(items.some(item=>item._id==='bow-surge'));assert.equal(items.some(item=>item._id==='surge'),false);
});
