import test from 'node:test';
import assert from 'node:assert/strict';
import {electricityBasicSettlementEnabled} from '../scripts/eldamon-electricity.mjs';
import {fortressRuleCompatibilityEnabled} from '../scripts/fortress-rule-compat.mjs';
import {loadSalubriousWorkbench} from '../scripts/salubrious-message-privacy.mjs';
import {installPatreonTreatmentCompatibility} from '../scripts/patreon-treatment-compat.mjs';

function gameFor(version){
 return {world:{id:'ujx5r8oipw7ercdr'},system:{id:'pf2e',version},release:{generation:14},modules:new Map([
  ['battlezoo-eldamon-pf2e',{active:true}],['xdy-pf2e-workbench',{active:true,version:'installed'}],['patreon-v3',{active:true,version:'installed'}],
 ]),PF2eWorkbench:{refocus(){}},pf2e:{Check:{}}};
}

for(const version of ['8.5.1','8.6.0','9.0.0'])test(`world repair and native Refocus interface remain available on PF2e ${version}`,async()=>{
 const game=gameFor(version);
 assert.equal(electricityBasicSettlementEnabled(game),true);
 assert.equal(fortressRuleCompatibilityEnabled(game),true);
 assert.equal((await loadSalubriousWorkbench({game})).ready,true);
 game.system.id='other';
 assert.equal(electricityBasicSettlementEnabled(game),false);
 assert.equal(fortressRuleCompatibilityEnabled(game),false);
 assert.equal((await loadSalubriousWorkbench({game})).ready,false);
});

test('Refocus still requires the installed callable interface and supported core',async()=>{
 for(const change of [game=>delete game.PF2eWorkbench.refocus,game=>game.release.generation=15,game=>game.modules.get('xdy-pf2e-workbench').active=false]){
  const game=gameFor('8.6.0');change(game);assert.equal((await loadSalubriousWorkbench({game})).ready,false);
 }
});

for(const version of ['8.6.0','9.0.0'])test(`treatment compatibility retains its native wrapper contract on PF2e ${version}`,async()=>{
 const game=gameFor(version),scope={acquirePatreonPublicScope:()=>null};
 const original=function checkCall(next,...args){return next(...args)};
 const entry={fn:original,package_info:{id:'patreon-v3'},target:'game.pf2e.Check.roll',setter:false,type:{name:'WRAPPER'},chain:true,bind:null};
 const holder={name:'game.pf2e.Check.roll',is_property:false,active:true,_outstanding_wrappers:0,getter_data:[entry],get_fn_data(){return this.getter_data},clear_static_dispatch_chain_cache(){},get_static_dispatch_chain(){},call_wrapper(){}};
 const get=()=>entry.fn;get._lib_wrapper=holder;Object.defineProperty(game.pf2e.Check,'roll',{get});
 const result=await installPatreonTreatmentCompatibility({game,scope});assert.equal(result.installed,true);
 assert.equal(entry.fn((_check,context)=>context.result,{}, {result:42}),42);
 assert.equal(result.dispose(),true);assert.equal(entry.fn,original);
 holder.getter_data.push({...entry});assert.equal((await installPatreonTreatmentCompatibility({game,scope})).installed,false);
});
