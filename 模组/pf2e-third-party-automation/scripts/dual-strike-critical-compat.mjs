import {MODULE_ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';
import {createConfigurationMaintenance} from './config-maintenance.mjs';

export const DUAL_STRIKE_CRITICAL_MARKER=`${MODULE_ID}:dual-strike-critical-specialization`;
// Trigger Trove 2.5.0 rebuilt "Critical Specialization" as NzgcT7S1lbrIAVRb (2.4.1:
// 6JdDvJY9YKdxoi2o). The predicate node and its options connection kept their ids.
const ENGINE='trigger-engine',SETTING='pf2e-trigger-triggers',GRAPHS=['NzgcT7S1lbrIAVRb','6JdDvJY9YKdxoi2o'],NODE='oYYR4W7IaJpI1OpI';
const basePredicate=['check:outcome:critical-success','critical-specialization'];
const excludedPredicate=[...basePredicate,{not:DUAL_STRIKE_CRITICAL_MARKER}];
const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function patreonSword(game){
 if(game.system?.id!=='pf2e'||!BOB2_AUTOMATION_WORLDS.has(game.world?.id)||game.modules.get('patreon-v3')?.active!==true)return false;
 const group=game.settings.get('patreon-v3','rulesV3')?.uYv7S8tdeXXpmd8a;
 return group?.isActive===true&&group.handlerRules?.some(rule=>rule.triggerType==='damage-roll'&&rule.value==='criticalSpecializationSword');
}
const troveActive=game=>game.modules.get(ENGINE)?.active===true&&game.modules.get('pf2e-trigger-trove')?.active===true;
// trigger-engine runs a module graph listed in `enabled` and a world copy not listed in `disabled`.
const graphEnabled=(setting,id)=>!setting?.disabled?.includes(id)&&(setting?.enabled?.includes(id)||setting?.sources?.some(graph=>graph.id===id));
function predicateNode(graph){
 const node=graph?.nodes?.find(node=>node.id===NODE);
 return node?.type==='match-predicate'&&node.inputs?.options?.connection==='sj9X74RFqaVJPJpl:outputs:options'?node:null;
}
function readPredicate(graph){try{return JSON.parse(predicateNode(graph)?.inputs.predicate.value);}catch{return null;}}
const excluded=graph=>equal(readPredicate(graph),excludedPredicate);

/** The origin on a merged damage card cannot identify two weapons. Let the
 * existing Patreon sword handler consume the separately attributed crit keys. */
export function dualStrikeSwordAutomationReady(game){
 if(!patreonSword(game))return false;
 if(!troveActive(game))return true;
 const setting=game.settings.get(ENGINE,SETTING);
 return GRAPHS.every(id=>!graphEnabled(setting,id)||excluded(setting.sources?.find(graph=>graph.id===id)));
}

/** Pure: returns the trigger-engine.pf2e-trigger-triggers value with a world copy
 * of `graph` that excludes TPA-marked Double Slice crits. `graph` is the Critical
 * Specialization entry of the installed modules/pf2e-trigger-trove/triggers/base-triggers.json,
 * and its id is the current one. A TPA copy under the other known id is dropped and
 * that id in `enabled` is replaced in place. A copy TPA did not make throws. */
export function buildDualStrikeCriticalCompatibility(setting,{graph}){
 if(!GRAPHS.includes(graph?.id))throw Error('Trigger Trove 的重击专精图不是已知版本，未修改设置。');
 const current=graph.id,stale=GRAPHS.find(id=>id!==current),result=structuredClone(setting),sources=result.sources??[];
 if(sources.some(source=>source.id===stale&&!excluded(source)))throw Error('旧版重击专精图的世界副本不是 TPA 建的，未覆盖自定义配置。');
 const source=structuredClone(sources.find(source=>source.id===current)??graph);
 if(!excluded(source)){
  if(!equal(readPredicate(source),basePredicate))throw Error('双斩剑专精的 Trigger Trove predicate 已改变，未覆盖自定义配置。');
  predicateNode(source).inputs.predicate.value=JSON.stringify(excludedPredicate);
 }
 // The copy takes the slot of the first Critical Specialization copy already saved.
 let placed=false;
 result.sources=sources.flatMap(entry=>![current,stale].includes(entry.id)?[entry]:placed?[]:(placed=true,[source]));
 if(!placed)result.sources.push(source);
 if(Array.isArray(result.enabled)&&result.enabled.includes(stale))result.enabled=result.enabled.includes(current)?result.enabled.filter(id=>id!==stale):result.enabled.map(id=>id===stale?current:id);
 return result;
}

export function createDualStrikeCriticalCompatibility({game,loadGraphs=async()=>{
 const response=await fetch('modules/pf2e-trigger-trove/triggers/base-triggers.json');
 if(!response.ok)throw Error('无法读取 Trigger Trove 的剑专精图。');
 return response.json();
}}){
 let graph;
 const eligible=()=>patreonSword(game)&&troveActive(game)&&GRAPHS.some(id=>graphEnabled(game.settings.get(ENGINE,SETTING),id));
 const maintain=createConfigurationMaintenance({game,settings:[{module:ENGINE,key:SETTING,when:eligible,
  transform:setting=>buildDualStrikeCriticalCompatibility(setting,{graph}),
  reason:'仅由 Patreon 处理 TPA 双斩已确认的剑重击专精，避免 Trigger Trove 重复应用；其它攻击保留原图。',
 }]});
 return {async initialize(){
  if(!isActiveGM(game)||!eligible())return;
  if(!graph){const graphs=await loadGraphs();graph=GRAPHS.map(id=>graphs.find(source=>source.id===id)).find(Boolean);}
  await maintain();
 }};
}
