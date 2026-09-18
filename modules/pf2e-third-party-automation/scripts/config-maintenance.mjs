import {MODULE_ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';

/** One serialized settings write; every changed predicate retains its original value. */
export function createConfigurationMaintenance({game,repairs=[],settings=[]}){
 const queue=new SerialActions();
 const write=async(module,key,original,value,changes)=>{
  const requireGM=()=>{if(!isActiveGM(game))throw Error('主GM已改变，配置修复将由当前主GM重试。')};
  requireGM();
  const history=game.settings.get(MODULE_ID,'configurationBackups')??[];
  await game.settings.set(MODULE_ID,'configurationBackups',[...history,{version:'0.3.0',time:new Date().toISOString(),setting:`${module}.${key}`,changes}]);
  requireGM();
  if(JSON.stringify(game.settings.get(module,key))!==JSON.stringify(original))throw Error('自动化配置在修复期间发生变化，已保留备份，稍后重试。');
  await game.settings.set(module,key,value);
 };
 return ()=>queue.run('configuration',async()=>{
  if(!isActiveGM(game))return;
  for(const setting of settings){
   if(!game.modules.get(setting.module)?.active||!setting.when(game))continue;
   const before=game.settings.get(setting.module,setting.key);
   const value=setting.transform?setting.transform(structuredClone(before),game):setting.value;
   if(JSON.stringify(before)===JSON.stringify(value))continue;
   await write(setting.module,setting.key,before,value,[{path:setting.key,before,after:value,reason:setting.reason}]);
  }
  if(!repairs.length||!game.modules.get('patreon-v3')?.active)return;
  const original=game.settings.get('patreon-v3','rulesV3');
  if(!original||typeof original!=='object'||Array.isArray(original))return;
  let rules=structuredClone(original);const changes=[];
  for(const repair of repairs){const result=repair(rules);rules=result.rules;changes.push(...result.changes);}
  if(!changes.length)return;
  await write('patreon-v3','rulesV3',original,rules,changes);
 });
}
