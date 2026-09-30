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

const repairKnownRules=rules=>{
 if(!rules.group.predicate.includes('old'))return {rules,changes:[]};
 rules.group.predicate=['fixed'];
 return {rules,changes:[{path:'group.predicate',before:['old'],after:['fixed'],reason:'verified native rule correction'}]};
};
test('Patreon change notification runs once after the successful write resolves and receives the saved change summary',async()=>{
 const f=fixture(),notices=[];let entered,release;
 const writing=new Promise(resolve=>{entered=resolve}),pending=new Promise(resolve=>{release=resolve}),set=f.game.settings.set;
 f.game.settings.set=async(module,key,value)=>{if(module==='patreon-v3'&&key==='rulesV3'){entered();await pending}return set(module,key,value)};
 const reconcile=createConfigurationMaintenance({game:f.game,repairs:[repairKnownRules],onRulesChanged:changes=>{
  assert.deepEqual(f.data.get('patreon-v3.rulesV3'),{group:{predicate:['fixed']}},'notification cannot claim a saved repair before its write');
  notices.push(changes);
 }});
 const run=reconcile();await writing;assert.deepEqual(notices,[]);release();await run;await reconcile();
 assert.deepEqual(notices,[[{path:'group.predicate',before:['old'],after:['fixed'],reason:'verified native rule correction'}]]);
 assert.deepEqual(f.writes,[`${MODULE_ID}.configurationBackups`,'patreon-v3.rulesV3']);
 notices[0][0].after.push('notification-only');
 assert.deepEqual(f.data.get(`${MODULE_ID}.configurationBackups`)[0].changes[0].after,['fixed'],'notification consumers must not alter the saved backup');
});
for(const situation of ['unchanged','not-gm','inactive-patreon'])test(`Patreon change notification is silent when no repair write is authorized or needed (${situation})`,async()=>{
 const f=fixture(),notices=[];
 if(situation==='unchanged')f.rules.group.predicate=['fixed'];
 if(situation==='not-gm')f.game.user.id='player';
 if(situation==='inactive-patreon')f.game.modules.get('patreon-v3').active=false;
 await createConfigurationMaintenance({game:f.game,repairs:[repairKnownRules],onRulesChanged:changes=>notices.push(changes)})();
 assert.deepEqual(notices,[]);assert.deepEqual(f.writes,[]);
});
test('an unrelated setting write never sends the Patreon rules notification',async()=>{
 const f=fixture(),notices=[];
 await createConfigurationMaintenance({game:f.game,onRulesChanged:changes=>notices.push(changes),settings:[{module:'pf2e-reaction',key:'builtinReactionsEnabled',when:()=>true,value:['shield-block'],reason:'verified coverage'}]})();
 assert.deepEqual(notices,[]);assert(f.writes.includes('pf2e-reaction.builtinReactionsEnabled'));
});
for(const failedKey of ['configurationBackups','rulesV3'])test(`a failed ${failedKey} write cannot announce that Patreon rules were saved`,async()=>{
 const f=fixture(),notices=[],set=f.game.settings.set;
 f.game.settings.set=async(module,key,value)=>{if(key===failedKey)throw Error('write failed');return set(module,key,value)};
 const reconcile=createConfigurationMaintenance({game:f.game,repairs:[repairKnownRules],onRulesChanged:changes=>notices.push(changes)});
 await assert.rejects(reconcile,/write failed/);assert.deepEqual(notices,[]);assert.deepEqual(f.data.get('patreon-v3.rulesV3'),{group:{predicate:['old']}});
});
test('GM handoff while saving the backup does not write rules or send a reload notice',async()=>{
 const f=fixture(),notices=[],set=f.game.settings.set;
 f.game.settings.set=async(...args)=>{const result=await set(...args);f.game.users.activeGM={id:'new-gm'};return result};
 const reconcile=createConfigurationMaintenance({game:f.game,repairs:[repairKnownRules],onRulesChanged:changes=>notices.push(changes)});
 await assert.rejects(reconcile,/主GM/);assert.deepEqual(notices,[]);assert.deepEqual(f.writes,[`${MODULE_ID}.configurationBackups`]);
});
