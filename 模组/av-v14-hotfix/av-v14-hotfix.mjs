import {registerLegacyCompat} from './scripts/compat/legacy.mjs';
import {captureGridNative,installGrid} from './scripts/patches/grid.mjs';
import {createItemNameFastPath,ITEM_NAME_HASHES} from './scripts/patches/item-name.mjs';
import {installSundryPatch} from './scripts/patches/sundry.mjs';
import {installDurationPatch} from './scripts/patches/duration.mjs';
import {installTimestampPatch} from './scripts/patches/timestamps.mjs';
import {installChatDeleteCoalescing} from './scripts/patches/chat.mjs';
import {registerBabeleIndex} from './scripts/patches/babele.mjs';
import {hashSource} from './scripts/source-hash.mjs';
import {installTokenizerChatPortraitPatch} from './scripts/patches/tokenizer-chat.mjs';
import {installSoundStopPatch} from './scripts/patches/sound-stop.mjs';
import {installBbmmHardLocks} from './scripts/patches/bbmm-locks.mjs';
import {registerDsnQualitySettings,installDsnQualityLocks} from './scripts/patches/dsn-quality.mjs';
import {installTurnLifecyclePatch} from './scripts/patches/turn-lifecycle.mjs';
import {installDsnChatRecovery} from './scripts/patches/dsn-chat.mjs';

const ID='av-v14-hotfix';
const nativeGrid=captureGridNative();
const state={version:'0.6.18',patches:{patreon:{status:'retired',detail:'Upstream 3.2.29 includes the relationship refresh guards.'},wayfinderFog:{status:'retired',detail:'Wayfinder 14.1.1 replaced the old fog implementation; the 14.0.1 adapter is retired.'}}};
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
  for(const [key,name] of Object.entries({legacy:'旧导入器数据兼容',grid:'Grid 距离与灵光兼容修正',itemNames:'PF2e 物品名称性能优化',sundry:'Sundry 效果图标刷新优化',duration:'聊天时间戳格式化性能优化',timestamps:'聊天时间戳 DOM 更新优化',babele:'汉化索引增量更新',chat:'批量删除聊天性能优化',tokenizerChat:'Tokenizer2 聊天头像缩放与遮罩修正',soundStop:'原生缓冲音频停止修正',bbmmLocks:'BBMM v14 玩家硬锁修正',dsnQualityLocks:'BBMM / DsN 画质硬锁适配',turnLifecycle:'共享回合与奴仆到期兼容修正',dsnChat:'DsN 动画失败后恢复聊天与队列'}))
    game.settings.register(ID,key,{name,scope:'world',config:true,type:Boolean,default:true,requiresReload:true});
  game.modules.get(ID).api={status:()=>({...structuredClone(state),babele:babele?.status()})};
  void run('dsnQualityLocks',()=>registerDsnQualitySettings({report}));
});

Hooks.once('setup',async()=>{
  if((game.release?.generation??Number.parseInt(game.version,10))!==14){report('runtime','unsupported-core',game.version);return;}
  // Both adapters must be installed before upstream ready synchronization.
  void run('bbmmLocks',()=>installBbmmHardLocks({report}));
  void run('dsnQualityLocks',()=>installDsnQualityLocks({moduleId:ID,report}));
  // Register synchronously in setup so the first chat history render is covered.
  void run('tokenizerChat',()=>installTokenizerChatPortraitPatch({moduleId:ID,report}));
  void run('soundStop',()=>installSoundStopPatch({moduleId:ID,report}));
  // Register before DsN starts its ready-time model preloads.
  void run('dsnChat',()=>installDsnChatRecovery({report}));
  await run('legacy',()=>registerLegacyCompat({moduleId:ID,registerWrapper:(target,fn,type)=>libWrapper.register(ID,target,fn,type),report}));
  await run('grid',()=>installGrid({native:nativeGrid,hash,report}));
});

Hooks.once('ready',()=>queueMicrotask(async()=>{
  if((game.release?.generation??Number.parseInt(game.version,10))!==14)return;
  report('runtime','initializing');
  await run('itemNames',async()=>{
    const system=game.system,version=system.version;
    if(system.id!=='pf2e'||!Object.hasOwn(ITEM_NAME_HASHES,version))return report('itemNames','unsupported-system');
    const expected=ITEM_NAME_HASHES[version];
    const target=game.pf2e.system,original=target.generateItemName;
    if(await hash(original.toString())!==expected)return report('itemNames','unsupported-source');
    if((game.release?.generation??Number.parseInt(game.version,10))!==14||game.system!==system||system.id!=='pf2e'||system.version!==version||game.pf2e.system!==target||target.generateItemName!==original)return report('itemNames','source-changed-during-validation');
    target.generateItemName=createItemNameFastPath(original,globalThis);
    report('itemNames','installed');
  });
  await run('sundry',()=>installSundryPatch({report}));
  await run('turnLifecycle',()=>installTurnLifecyclePatch({report}));
  await run('dsnChat',()=>installDsnChatRecovery({report}));
  await run('chat',()=>report('chat',installChatDeleteCoalescing({moduleId:ID})?'installed':'unsupported'));
  await run('duration',()=>installDurationPatch({report}));
  await run('timestamps',()=>installTimestampPatch({report}));
  await run('babele',async()=>{
    let preserveIndexFlags;
    if(game.modules.get('babele')?.active&&game.modules.get('babele').version==='2.9.1')
      ({preserveIndexFlags}=await import('../babele/script/foundry/wrapper.js'));
    babele=registerBabeleIndex({moduleId:ID,preserveIndexFlags});report('babele',babele.status().state);
  });
  report('runtime','ready');
}));
