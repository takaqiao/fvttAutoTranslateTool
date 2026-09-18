import test from 'node:test';
import assert from 'node:assert/strict';
import {createConfigurationMaintenance} from '../scripts/config-maintenance.mjs';
import {MODULE_ID} from '../scripts/rules.mjs';

function fixture(){
 const rules={group:{predicate:['old']}},reads=[],writes=[];
 const data=new Map([[`${MODULE_ID}.configurationBackups`,[]],['patreon-v3.rulesV3',rules],['pf2e-reaction.builtinReactionsEnabled',['glimpse-of-redemption','shield-block']]]);
 const game={user:{id:'gm'},users:{activeGM:{id:'gm'}},modules:new Map([['pf2e-reaction',{active:true}],['patreon-v3',{active:true}]]),settings:{get:(module,key)=>{reads.push(`${module}.${key}`);return data.get(`${module}.${key}`)},set:async(module,key,value)=>{writes.push(`${module}.${key}`);data.set(`${module}.${key}`,value)}}};
 return {game,data,rules,reads,writes};
}

test('settings-only maintenance performs its settings write without reading Patreon rules',async()=>{
 const f=fixture();
 const reconcile=createConfigurationMaintenance({game:f.game,settings:[{module:'pf2e-reaction',key:'builtinReactionsEnabled',when:()=>true,reason:'verified coverage',transform:value=>value.filter(slug=>slug!=='glimpse-of-redemption')}]});
 await reconcile();
 assert.deepEqual(f.data.get('pf2e-reaction.builtinReactionsEnabled'),['shield-block']);
 assert.equal(f.reads.includes('patreon-v3.rulesV3'),false);
 assert.equal(f.data.get(`${MODULE_ID}.configurationBackups`).length,1);
 assert.deepEqual(f.writes,[`${MODULE_ID}.configurationBackups`,'pf2e-reaction.builtinReactionsEnabled']);
});

test('configured Patreon repairs still clone, back up and write the repaired rules',async()=>{
 const f=fixture();
 const reconcile=createConfigurationMaintenance({game:f.game,repairs:[rules=>{rules.group.predicate=['fixed'];return {rules,changes:[{path:'group.predicate',before:['old'],after:['fixed']}]}}]});
 await reconcile();
 assert.deepEqual(f.rules,{group:{predicate:['old']}},'repair must not mutate the live setting before writing');
 assert.deepEqual(f.data.get('patreon-v3.rulesV3'),{group:{predicate:['fixed']}});
 assert.deepEqual(f.data.get(`${MODULE_ID}.configurationBackups`)[0].changes,[{path:'group.predicate',before:['old'],after:['fixed']}]);
 assert.deepEqual(f.writes,[`${MODULE_ID}.configurationBackups`,'patreon-v3.rulesV3']);
});
