import {registerLegacyCompat} from './scripts/compat/legacy.mjs';
import {createItemNameFastPath,ITEM_NAME_HASHES} from './scripts/patches/item-name.mjs';
import {installSundryPatch,prepareSundryPatch} from './scripts/patches/sundry.mjs';
import {installDurationPatch} from './scripts/patches/duration.mjs';
import {installTimestampPatch} from './scripts/patches/timestamps.mjs';
import {installChatDeleteCoalescing} from './scripts/patches/chat.mjs';
import {registerBabeleIndex,verifyBabeleIndexSources} from './scripts/patches/babele.mjs';
import {hashSource} from './scripts/source-hash.mjs';
import {installTokenizerChatPortraitPatch} from './scripts/patches/tokenizer-chat.mjs';
import {installSoundStopPatch} from './scripts/patches/sound-stop.mjs';
import {registerDsnQualitySettings,installDsnQualityLocks} from './scripts/patches/dsn-quality.mjs';
import {installTurnLifecyclePatch,prepareTurnLifecyclePatch} from './scripts/patches/turn-lifecycle.mjs';
import {installDsnChatRecovery} from './scripts/patches/dsn-chat.mjs';

const ID='av-v14-hotfix';
const state={version:'0.6.26',patches:{patreon:{status:'retired',detail:'Upstream 3.2.29 includes the relationship refresh guards.'},wayfinderFog:{status:'retired',detail:'Wayfinder 14.1.1 replaced the old fog implementation; the 14.0.1 adapter is retired.'},grid:{status:'retired',detail:'Use Grid 2.3.1 distance and undrawn-token aura handling.'},bbmmLocks:{status:'retired',detail:'Use BBMM 1.4.11 submenu hard locks and notifications.'}}};
let babele;
const report=(feature,status,detail)=>{
  if(typeof feature==='object'){const {restore,...data}=feature;state.patches[feature.feature]=data;return;}
  state.patches[feature]={status,...(detail!==undefined?{detail}: {})};
  console.info(`${ID} | ${feature}: ${status}`,detail??'');
};
const hash=hashSource;
const enabled=key=>game.settings.get(ID,key);
const run=async(key,fn)=>{try{if(!enabled(key))return report(key,'disabled');await fn();}catch(error){report(key,'failed',String(error));console.error(`${ID} | ${key}`,error);}};

Hooks.once('init',()=>{
  for(const [key,name] of Object.entries({legacy:'旧导入器数据兼容',itemNames:'PF2e 物品名称性能优化',sundry:'Sundry 效果图标刷新优化',duration:'聊天时间戳格式化性能优化',timestamps:'聊天时间戳 DOM 更新优化',babele:'汉化索引增量更新',chat:'批量删除聊天性能优化',tokenizerChat:'Tokenizer2 聊天头像缩放与遮罩修正',soundStop:'原生缓冲音频停止修正',dsnQualityLocks:'BBMM / DsN 画质硬锁适配',turnLifecycle:'共享回合与奴仆到期兼容修正',dsnChat:'DsN 动画失败后恢复聊天与队列'}))
    game.settings.register(ID,key,{name,scope:'world',config:true,type:Boolean,default:true,requiresReload:true});
  game.modules.get(ID).api={status:()=>({...structuredClone(state),babele:babele?.status()})};
  void run('dsnQualityLocks',()=>registerDsnQualitySettings({report}));
});

Hooks.once('setup',async()=>{
  // Install the quality bridge before upstream ready synchronization.
  void run('dsnQualityLocks',()=>installDsnQualityLocks({moduleId:ID,report}));
  // Register synchronously in setup so the first chat history render is covered.
  void run('tokenizerChat',()=>installTokenizerChatPortraitPatch({moduleId:ID,report}));
  void run('soundStop',()=>installSoundStopPatch({moduleId:ID,report}));
  // Register before DsN starts its ready-time model preloads.
  void run('dsnChat',()=>installDsnChatRecovery({report}));
  await run('legacy',()=>registerLegacyCompat({moduleId:ID,registerWrapper:(target,fn,type)=>libWrapper.register(ID,target,fn,type),report}));
});

Hooks.once('ready',()=>queueMicrotask(async()=>{
  report('runtime','initializing');
  await run('itemNames',async()=>{
    const system=game.system,version=system.version;
    if(system.id!=='pf2e')return report('itemNames','unsupported-system');
    const target=game.pf2e?.system,original=target?.generateItemName;
    if(typeof original!=='function')return report('itemNames','unsupported-runtime');
    if(!Object.values(ITEM_NAME_HASHES).includes(await hash(Function.prototype.toString.call(original))))return report('itemNames','unsupported-source');
    if(game.system!==system||system.id!=='pf2e'||system.version!==version||game.pf2e.system!==target||target.generateItemName!==original)return report('itemNames','source-changed-during-validation');
    target.generateItemName=createItemNameFastPath(original,globalThis);
    report('itemNames','installed');
  });
  await run('sundry',async()=>{await prepareSundryPatch();return installSundryPatch({report});});
  await run('turnLifecycle',async()=>{await prepareTurnLifecyclePatch();return installTurnLifecyclePatch({report});});
  await run('dsnChat',()=>installDsnChatRecovery({report}));
  await run('chat',()=>report('chat',installChatDeleteCoalescing({moduleId:ID})?'installed':'unsupported'));
  await run('duration',()=>installDurationPatch({report}));
  await run('timestamps',()=>installTimestampPatch({report}));
  await run('babele',async()=>{
    let preserveIndexFlags,sourceContract;
    const owners=['pf2e_compendium_chn','babele','lib-wrapper'].map(id=>({id,module:game.modules.get(id),version:game.modules.get(id)?.version}));
    const unchanged=()=>owners.every(({id,module,version})=>module?.active&&game.modules.get(id)===module&&module.version===version);
    if(unchanged())try{
      const [translation,wrapper,libWrapper]=await Promise.all([
        '../pf2e_compendium_chn/babele-ondemand-patch.js','../babele/script/foundry/wrapper.js','../lib-wrapper/lib-wrapper.js'
      ].map(async path=>{
        const response=await fetch(new URL(path,import.meta.url));
        if(!response.ok)throw Error('Unable to read Babele index contract: '+response.status);
        return response.text();
      }));
      sourceContract=verifyBabeleIndexSources({translation,wrapper,libWrapper});
      if(sourceContract.supported)({preserveIndexFlags}=await import('../babele/script/foundry/wrapper.js'));
      if(!unchanged())sourceContract={supported:false,reason:'Babele source owner changed during validation.'};
    }catch(error){sourceContract={supported:false,reason:String(error)};}
    babele=registerBabeleIndex({moduleId:ID,preserveIndexFlags,sourceContract});report('babele',babele.status().state);
  });
  report('runtime','ready');
}));
