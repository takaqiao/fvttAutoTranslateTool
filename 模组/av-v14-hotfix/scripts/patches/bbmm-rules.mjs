const BBMM='bbmm';

export function bbmmCompatibility(runtime){
  const {game}=runtime;
  const module=game?.modules?.get(BBMM);
  if(!module?.active)return 'inactive-bbmm';
  if(Number.parseInt(module.version,10)!==1)return 'unsupported-bbmm';
  return null;
}

// BBMM 1.4.11 stores a world Object of {namespace,key,value,soft?} rules.
// Check the registry at read/setup time; BBMM may register after our init hook.
export function bbmmRegistryCompatibility(runtime){
  const settings=runtime.game?.settings,registry=settings?.settings;
  if(typeof registry?.get!=='function'||typeof settings?.get!=='function')return 'unsupported-bbmm-registry';
  for(const[key,type]of [['userSettingSync',Object],['enableUserSettingSync',Boolean]]){
    const cfg=registry.get(`${BBMM}.${key}`);
    if(cfg?.scope!=='world'||cfg.type!==type)return 'unsupported-bbmm-registry';
  }
  return null;
}

const record=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);

export function readBbmmHardRules(runtime,user=runtime.game?.user){
  const {game}=runtime;
  if(bbmmCompatibility(runtime)||bbmmRegistryCompatibility(runtime)
    ||!user?.id||user.id!==game.user?.id||game.user.isGM)return {};
  try{
    if(game.settings.get(BBMM,'enableUserSettingSync')!==true)return {};
    const rules=game.settings.get(BBMM,'userSettingSync');
    if(!record(rules))return {};
    return Object.fromEntries(Object.entries(rules).filter(([id,row])=>{
      const cfg=game.settings.settings.get(id),dot=id.indexOf('.');
      return ['client','user'].includes(cfg?.scope)&&dot>0&&record(row)
        &&row.namespace===id.slice(0,dot)&&row.key===id.slice(dot+1)
        &&Object.hasOwn(row,'value')&&(!Object.hasOwn(row,'soft')||row.soft===false);
    }));
  }catch{return {};}
}
