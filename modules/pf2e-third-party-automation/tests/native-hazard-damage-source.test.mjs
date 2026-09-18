import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './glimpse-fixture.mjs';
import {resolveGlimpseSource,validateGlimpseSource,glimpseCandidates} from '../scripts/glimpse-source.mjs';
import {resolveDeflectionSource,validateDeflectionSource} from '../scripts/transcendent-deflection-source.mjs';

function hazard(){
 const f=fixture();f.enemy.type='hazard';f.enemy.alliance=null;
 f.item.system={action:'strike'};
 f.enemy.system.actions=[{type:'strike',item:f.item,ready:true}];
 f.message.flags.pf2e.strike=null;
 f.roll.total=25;f.roll.instances[0].total=25;
 f.roll.toJSON=function(){return {class:'DamageRoll',evaluated:true,formula:'{25[slashing]}',total:this.total};};
 f.champion.isEnemyOf=a=>a.alliance!==null&&a.id==='enemy';
 return f;
}
for(const [name,resolve,validate] of [['glimpse',resolveGlimpseSource,validateGlimpseSource],['deflection',resolveDeflectionSource,validateDeflectionSource]]){
 test(`${name} proves a native 25-damage hazard Strike without creature-only strike flags`,async()=>{
  const f=hazard(),result=await resolve({...f,actor:f.ally});
  assert.equal(result.verified,true);assert.equal(result.item,f.item);
  assert.equal((await validate({...f,snapshot:result.snapshot})).verified,true);
  if(name==='glimpse')assert.deepEqual(glimpseCandidates(result,f.game),[]);
 });
 for(const [reason,mutate] of [
  ['creature missing flags',f=>f.enemy.type='npc'],
  ['contradictory strike flags',f=>f.message.flags.pf2e.strike={damaging:false,actor:f.enemy.uuid}],
  ['unprepared owned item',f=>f.enemy.system.actions=[]],
  ['same-name foreign prepared item',f=>f.enemy.system.actions[0].item={...f.item}],
  ['item removed from inventory',f=>f.enemy.items.delete(f.item.id)],
  ['non-strike action',f=>f.item.system.action='area-fire'],
  ['wrong prepared action type',f=>f.enemy.system.actions[0].type='area-fire'],
  ['ambiguous prepared strike',f=>f.enemy.system.actions.push({...f.enemy.system.actions[0]})],
  ['save source on melee item',f=>f.message.flags.pf2e.context.sourceType='save'],
  ['weapon-shaped hazard origin',f=>{f.item.type='weapon';f.message.flags.pf2e.origin.type='weapon';}],
 ])test(`${name} rejects hazard fallback with ${reason}`,async()=>{
  const f=hazard();mutate(f);assert.equal((await resolve({...f,actor:f.ally})).verified,false);
 });
}
