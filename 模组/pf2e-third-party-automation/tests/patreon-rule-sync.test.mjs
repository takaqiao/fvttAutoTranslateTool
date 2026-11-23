import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createConfigurationMaintenance} from '../scripts/config-maintenance.mjs';
import {buildKnowledgePatreonRepairs} from '../scripts/knowledge-automation.mjs';
import {buildPartyPatreonRepairs} from '../scripts/party-automation.mjs';
import {buildBob2SwashbucklerPatreonRepairs} from '../scripts/bob2-swashbuckler.mjs';
import {MODULE_ID} from '../scripts/rules.mjs';

// Patreon 3.2.32 as installed on 11-23, and the 3.2.31 rules it replaces (version 211).
const work='C:/Users/Taka/Desktop/fvtt/work';
const patreonRoot=process.env.PATREON_3232_ROOT??`${work}/main-compat-20261123/live/modules/patreon-v3`;
const previousRules=process.env.PATREON_211_RULES??`${work}/main-compat-20261108/live/modules/patreon-v3/rules/config.json`;
const pinned=(file,sha256)=>{const bytes=readFileSync(file);assert.equal(createHash('sha256').update(bytes).digest('hex'),sha256,file);return bytes.toString('utf8');};
const patreonSource=pinned(`${patreonRoot}/src/index.js`,'74961f3f5fa2e75b7a84367fbe9be4634c9f861f4c5ffccef58da68ea1b97523');
const config212=JSON.parse(pinned(`${patreonRoot}/rules/config.json`,'8e6bfd8f36e34b6849719d7547733168d3c805362b12e6138558eb845ebf4c94'));
const config211=JSON.parse(readFileSync(previousRules,'utf8'));
assert.equal(config211.version,211);
const main=readFileSync(new URL('../scripts/main.mjs',import.meta.url),'utf8');
const mainLine=(match,label)=>{const line=main.split(/\r?\n/).find(match);assert.ok(line,label);return line;};

/** Patreon's own syncRulesFn. Its Fe() also rebuilds rule-group class instances;
 * a deep clone keeps the same saved data for this merge. */
function patreonSync(game,{onRebuild=()=>{}}={}){
 const start=patreonSource.indexOf('async function Sa(){'),end=patreonSource.indexOf('r(Sa,"syncRulesFn")');
 assert.ok(start>=0&&end>start,'Patreon syncRulesFn');
 const fetch=async url=>{assert.equal(url,'modules/patreon-v3/rules/config.json');return {ok:true,json:async()=>structuredClone(config212)}};
 const errors=[],ui={notifications:{info:()=>{}}},log={log:()=>{},error:(...args)=>errors.push(args)};
 const Sa=Function('fetch','Fe','xt','u','B','ye','Ft','ui','game','console',`${patreonSource.slice(start,end)};return Sa;`)(
  fetch,()=>structuredClone(game.settings.get('patreon-v3','rulesV3')),()=>structuredClone(game.settings.get('patreon-v3','ruleVersion3')),
  'patreon-v3','rulesV3','ruleVersion3',onRebuild,ui,game,log);
 return {run:Sa,errors};
}

function fixture({world='bob-2',patreon=true}={}){
 const handlers=new Map(),writes=[],data=new Map([[`${MODULE_ID}.configurationBackups`,[]],['patreon-v3.rulesV3',Object.fromEntries(config211.groups.map(group=>[group.uuid,structuredClone(group)]))],['patreon-v3.ruleVersion3',211]]);
 const Hooks={on:(name,fn)=>{handlers.set(name,[...handlers.get(name)??[],fn]);}};
 const emit=(name,document)=>{for(const fn of handlers.get(name)??[])fn(document);};
 // Foundry calls create/updateSetting before the awaited set() resolves.
 const game={world:{id:world},system:{id:'pf2e'},user:{id:'gm'},users:{activeGM:{id:'gm'}},modules:new Map([['patreon-v3',{active:patreon}]]),
  settings:{get:(module,key)=>data.get(`${module}.${key}`),set:async(module,key,value)=>{const id=`${module}.${key}`,existed=data.has(id);data.set(id,structuredClone(value));writes.push(id);emit(existed?'updateSetting':'createSetting',{key:id,value});return value;}}};
 const warnings=[],errors=[],runs=[];
 const onRulesChanged=Function('game','ui',`${mainLine(line=>line.trimStart().startsWith('const onRulesChanged='),'onRulesChanged')}\nreturn onRulesChanged;`)(game,{notifications:{warn:message=>warnings.push(message)}});
 const reconcile=createConfigurationMaintenance({game,onRulesChanged,repairs:[buildPartyPatreonRepairs,buildKnowledgePatreonRepairs,rules=>buildBob2SwashbucklerPatreonRepairs(rules,{game})]});
 const install=Function('Hooks','report',`let configuration;${mainLine(line=>line.includes("'patreon-v3.ruleVersion3'"),'Patreon rule-sync hook')}\nreturn value=>{configuration=value};`);
 install(Hooks,error=>errors.push(error))(()=>{const run=reconcile();runs.push(run);return run;});
 const settle=async()=>{let seen;do{seen=runs.length;await Promise.allSettled(runs);}while(seen!==runs.length);};
 return {game,data,writes,warnings,errors,runs,emit,reconcile,settle,onRulesChanged};
}
const knowledgeExclusion={not:{and:['origin:item:sourceId:Compendium.pf2e.actionspf2e.Item.m0f2B7G9eaaTmhFL','feat:known-weaknesses']}};
const devise=rules=>rules.PyqEyRJewo3W4z0w.handlerRules[0].predicate.some(p=>JSON.stringify(p)===JSON.stringify(knowledgeExclusion));
const flyingBlade=(rules,index)=>JSON.stringify(rules.wmvqDMawtx5dcYHd.handlerRules[index].predicate).includes('feat:flying-blade');
const count=(list,value)=>list.filter(entry=>entry===value).length;

test('Patreon 212 sync overwrites the repaired force groups; its setting writes run the repairs again',async()=>{
 const f=fixture();
 await f.reconcile();await f.settle();
 const repaired=f.data.get('patreon-v3.rulesV3');
 assert.ok(devise(repaired));assert.ok(flyingBlade(repaired,0));
 const backups=f.data.get(`${MODULE_ID}.configurationBackups`).length,snapshots=[];
 f.writes.length=0;f.warnings.length=0;f.runs.length=0;
 const sync=patreonSync(f.game,{onRebuild:()=>snapshots.push(structuredClone(f.data.get('patreon-v3.rulesV3')))});
 await sync.run();await f.settle();
 assert.deepEqual(sync.errors,[]);assert.deepEqual(f.errors,[]);
 assert.equal(snapshots.length,1,'Patreon rebuilds its rule cache once, at the end of the sync');
 const rules=f.data.get('patreon-v3.rulesV3');
 assert.equal(f.data.get('patreon-v3.ruleVersion3'),212);
 assert.ok(rules['0FtUFVHrdyK7xwZT'],'groups added in 212 are present');
 for(const id of config212.removeIds)assert.equal(rules[id],undefined);
 assert.deepEqual(rules.cb4a9xC8VCzVtZ9Y,config212.groups.find(group=>group.uuid==='cb4a9xC8VCzVtZ9Y'));
 assert.ok(devise(rules));assert.ok(flyingBlade(rules,0));assert.ok(flyingBlade(rules,1));
 // Groups outside forceIds keep the world copy, so their earlier repairs survive untouched.
 assert.deepEqual(rules.CRPyKS9y4hzRpupT,repaired.CRPyKS9y4hzRpupT);
 assert.equal(count(f.writes,'patreon-v3.rulesV3'),2,'one Patreon sync write and one repair write');
 assert.equal(count(f.writes,'patreon-v3.ruleVersion3'),1);
 const added=f.data.get(`${MODULE_ID}.configurationBackups`).slice(backups);
 assert.equal(added.length,1);
 assert.deepEqual(added[0].changes.map(change=>change.path),['PyqEyRJewo3W4z0w.handlerRules.0.predicate','wmvqDMawtx5dcYHd.handlerRules.0.predicate','wmvqDMawtx5dcYHd.handlerRules.1.predicate']);
 assert.equal(f.warnings.length,1,'the repairing GM is told to refresh every client');
 assert.equal(f.runs.length,3,'rulesV3, ruleVersion3 and the repair write each queue one pass');
});

test('a Patreon sync landing while the ready pass saves its backup is repaired by the next pass',async()=>{
 const f=fixture(),set=f.game.settings.set;let hold=true,entered,release;
 const started=new Promise(resolve=>{entered=resolve}),gate=new Promise(resolve=>{release=resolve});
 f.game.settings.set=async(module,key,value)=>{if(hold&&key==='configurationBackups'){hold=false;entered();await gate;}return set(module,key,value);};
 const ready=f.reconcile();await started;
 await patreonSync(f.game).run();release();
 await assert.rejects(ready,/发生变化/);await f.settle();
 const rules=f.data.get('patreon-v3.rulesV3');
 assert.equal(f.data.get('patreon-v3.ruleVersion3'),212);
 assert.ok(devise(rules));assert.ok(flyingBlade(rules,0));assert.ok(flyingBlade(rules,1));
 assert.deepEqual(f.errors,[]);
});

for(const hook of ['createSetting','updateSetting'])for(const key of ['patreon-v3.rulesV3','patreon-v3.ruleVersion3'])test(`${hook} of ${key} queues one configuration pass`,async()=>{
 const f=fixture();f.emit(hook,{key});await f.settle();assert.equal(f.runs.length>=1,true);
 assert.equal(f.runs.length,1+count(f.writes,'patreon-v3.rulesV3'),'only the pass and its own repair write');
});

for(const key of ['pf2e.activeParty','patreon-v3.fastHealingTime','pf2e-third-party-automation.configurationBackups'])test(`an update of ${key} does not run Patreon configuration`,async()=>{
 const f=fixture();f.emit('updateSetting',{key});await f.settle();assert.equal(f.runs.length,0);
});

test('every world running Patreon is told to refresh after a rules repair',()=>{
 for(const world of ['cotct','sog','avvvvvv','bob-2','ujx5r8oipw7ercdr']){const f=fixture({world});f.onRulesChanged([]);assert.equal(f.warnings.length,1,world);}
 const f=fixture({patreon:false});f.onRulesChanged([]);assert.equal(f.warnings.length,0);
});
