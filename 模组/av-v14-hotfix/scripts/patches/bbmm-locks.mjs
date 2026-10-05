const BBMM='bbmm';
const installed=new WeakMap();

export function bbmmCompatibility(runtime){
  const {game}=runtime;
  if((game?.release?.generation??Number.parseInt(game?.version,10))!==14)return 'unsupported-core';
  const module=game.modules.get(BBMM);
  if(!module?.active)return 'inactive-bbmm';
  if(module.version!=='1.4.9')return 'unsupported-bbmm';
  return null;
}

// BBMM hard locks apply to the current non-GM account. Never interpret an
// unavailable registry as permission to enforce saved, possibly stale rules.
export function readBbmmHardRules(runtime,user=runtime.game?.user){
  const {game}=runtime;
  if(bbmmCompatibility(runtime)||!user?.id||user.id!==game.user?.id||game.user.isGM)return {};
  try{
    if(!game.settings.get(BBMM,'enableUserSettingSync'))return {};
    return game.settings.get(BBMM,'userSettingSync')??{};
  }catch{return {};}
}

export function installBbmmHardLocks({runtime=globalThis,report=()=>{}}={}){
  const finish=status=>{const result={feature:'bbmmLocks',status};report(result);return result;};
  const reason=bbmmCompatibility(runtime);if(reason)return finish(reason);
  const {game,Hooks}=runtime,settings=game.settings;
  if(installed.has(settings))return finish('already-installed');
  if(typeof Hooks?.on!=='function'||typeof Hooks?.off!=='function'||typeof runtime.setTimeout!=='function'
    ||!settings.settings.has('bbmm.userSettingSync')||!settings.settings.has('bbmm.enableUserSettingSync'))return finish('unsupported-runtime');
  const equals=runtime.foundry.utils.equals??runtime.foundry.utils.objectsEqual??((a,b)=>JSON.stringify(a)===JSON.stringify(b));
  const pending=new Map(),hooks=[];let active=true;
  const error=err=>runtime.console?.warn?.('av-v14-hotfix | BBMM hard-lock repair failed',err);
  const target=id=>{
    if(!active)return null;
    const cfg=settings.settings.get(id);
    if(!cfg||!['client','user'].includes(cfg.scope))return null;
    const row=readBbmmHardRules(runtime)[id];
    if(!row||row.soft===true||!Object.hasOwn(row,'value'))return null;
    const dot=id.indexOf('.');if(dot<=0)return null;
    const namespace=id.slice(0,dot),key=id.slice(dot+1);
    if(equals(settings.get(namespace,key),row.value))return null;
    return{namespace,key,value:row.value};
  };
  const enforce=id=>{
    try{
      if(typeof id!=='string'||pending.has(id)||!target(id))return;
      pending.set(id,runtime.setTimeout(async()=>{
        pending.delete(id);
        try{
          // The GM can change/unlock the rule while this repair is queued.
          const next=target(id);if(!next)return;
          await settings.set(next.namespace,next.key,runtime.foundry.utils.duplicate(next.value));
          runtime.ui?.notifications?.warn?.('此设置已由 GM 通过 BBMM 锁定');
        }catch(err){error(err);}
      },0));
    }catch(err){error(err);}
  };
  const on=(name,fn)=>hooks.push([name,Hooks.on(name,fn)]);
  on('clientSettingChanged',id=>enforce(id));
  const userChange=doc=>{
    if(doc?.user===game.user?.id&&typeof doc.key==='string'&&settings.settings.get(doc.key)?.scope==='user')enforce(doc.key);
  };
  on('createSetting',userChange);on('updateSetting',userChange);
  const restore=()=>{
    active=false;
    for(const timer of pending.values())runtime.clearTimeout(timer);
    pending.clear();for(const[name,id]of hooks)Hooks.off(name,id);
    installed.delete(settings);
  };
  const result={feature:'bbmmLocks',status:'installed',restore};installed.set(settings,result);report(result);return result;
}
