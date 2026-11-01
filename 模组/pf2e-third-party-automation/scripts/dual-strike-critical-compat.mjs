import {MODULE_ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';
import {createConfigurationMaintenance} from './config-maintenance.mjs';

export const DUAL_STRIKE_CRITICAL_MARKER=`${MODULE_ID}:dual-strike-critical-specialization`;
const ENGINE='trigger-engine',SETTING='pf2e-trigger-triggers',GRAPH='6JdDvJY9YKdxoi2o',NODE='oYYR4W7IaJpI1OpI';
const basePredicate=['check:outcome:critical-success','critical-specialization'];
const exclusion={not:DUAL_STRIKE_CRITICAL_MARKER};
const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function patreonSword(game){
 if(game.system?.id!=='pf2e'||!BOB2_AUTOMATION_WORLDS.has(game.world?.id)||game.modules.get('patreon-v3')?.active!==true)return false;
 const group=game.settings.get('patreon-v3','rulesV3')?.uYv7S8tdeXXpmd8a;
 return group?.isActive===true&&group.handlerRules?.some(rule=>rule.triggerType==='damage-roll'&&rule.value==='criticalSpecializationSword');
}
const troveActive=game=>game.modules.get(ENGINE)?.active===true&&game.modules.get('pf2e-trigger-trove')?.active===true;
const graphEnabled=setting=>!setting?.disabled?.includes(GRAPH)&&(setting?.enabled?.includes(GRAPH)||setting?.sources?.some(graph=>graph.id===GRAPH));
function predicateNode(graph){
 const node=graph?.nodes?.find(node=>node.id===NODE);
 return node?.type==='match-predicate'&&node.inputs?.options?.connection==='sj9X74RFqaVJPJpl:outputs:options'?node:null;
}
function readPredicate(graph){try{return JSON.parse(predicateNode(graph)?.inputs.predicate.value);}catch{return null;}}

/** The origin on a merged damage card cannot identify two weapons. Let the
 * existing Patreon sword handler consume the separately attributed crit keys. */
export function dualStrikeSwordAutomationReady(game){
 if(!patreonSword(game))return false;
 if(!troveActive(game))return true;
 const setting=game.settings.get(ENGINE,SETTING);
 return !graphEnabled(setting)||equal(readPredicate(setting.sources?.find(graph=>graph.id===GRAPH)),[...basePredicate,exclusion]);
}

export function buildDualStrikeCriticalCompatibility(setting,{graph}){
 const result=structuredClone(setting),sources=result.sources??=[],existing=sources.find(source=>source.id===GRAPH),source=structuredClone(existing??graph);
 const predicate=readPredicate(source);
 if(equal(predicate,[...basePredicate,exclusion]))return result;
 if(source?.id!==GRAPH||!equal(predicate,basePredicate))throw Error('双斩剑专精的 Trigger Trove predicate 已改变，未覆盖自定义配置。');
 predicateNode(source).inputs.predicate.value=JSON.stringify([...basePredicate,exclusion]);
 result.sources=existing?sources.map(graph=>graph.id===GRAPH?source:graph):[...sources,source];
 return result;
}

export function createDualStrikeCriticalCompatibility({game,loadGraphs=async()=>{
 const response=await fetch('modules/pf2e-trigger-trove/triggers/base-triggers.json');
 if(!response.ok)throw Error('无法读取 Trigger Trove 的剑专精图。');
 return response.json();
}}){
 let graph;
 const eligible=()=>patreonSword(game)&&troveActive(game)&&graphEnabled(game.settings.get(ENGINE,SETTING));
 const maintain=createConfigurationMaintenance({game,settings:[{module:ENGINE,key:SETTING,when:eligible,
  transform:setting=>buildDualStrikeCriticalCompatibility(setting,{graph}),
  reason:'仅由 Patreon 处理 TPA 双斩已确认的剑重击专精，避免 Trigger Trove 重复应用；其它攻击保留原图。',
 }]});
 return {async initialize(){
  if(!isActiveGM(game)||!eligible())return;
  const existing=game.settings.get(ENGINE,SETTING).sources?.find(source=>source.id===GRAPH);
  if(!existing&&!graph)graph=(await loadGraphs()).find(source=>source.id===GRAPH);
  await maintain();
 }};
}
