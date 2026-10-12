import {registerLegacyCompat} from './scripts/compat/legacy.mjs';
import {installItemNameFastPath} from './scripts/patches/item-name.mjs';
import {installSundryPatch,prepareSundryPatch} from './scripts/patches/sundry.mjs';
import {installTimestampPatch} from './scripts/patches/timestamps.mjs';
import {installChatDeleteCoalescing} from './scripts/patches/chat.mjs';
import {installTokenizerChatPortraitPatch} from './scripts/patches/tokenizer-chat.mjs';
import {installSoundStopPatch} from './scripts/patches/sound-stop.mjs';
import {registerDsnQualitySettings,installDsnQualityLocks} from './scripts/patches/dsn-quality.mjs';
import {installTurnLifecyclePatch,prepareTurnLifecyclePatch} from './scripts/patches/turn-lifecycle.mjs';
import {installDsnChatRecovery} from './scripts/patches/dsn-chat.mjs';

const ID='av-v14-hotfix';
const state={version:'0.6.28',patches:{patreon:{status:'retired',detail:'Upstream 3.2.29 includes the relationship refresh guards.'},wayfinderFog:{status:'retired',detail:'Wayfinder 14.1.1 replaced the old fog implementation; the 14.0.1 adapter is retired.'},grid:{status:'retired',detail:'Use Grid 2.3.1 distance and undrawn-token aura handling.'},bbmmLocks:{status:'retired',detail:'Use BBMM 1.4.11 submenu hard locks and notifications.'},babele:{status:'retired',detail:'Translation 3.2.3 replaces changed document index entries itself.'},duration:{status:'retired',detail:'The timestamp formatter cache saved a few milliseconds per refresh and no longer matched core 14.369.'},legacyAdventureSettings:{status:'retired',detail:'The Murder in Metal City importer setting pre-registration is retired; legacy keeps the chat type and scene field migrations.'}}};
// Patches report under their own names; status keys follow the world switches.
const ALIASES={'legacy-compat':'legacy','turn-lifecycle':'turnLifecycle'};
// Everything else means a repair that should be running is not, and is warned about.
const QUIET=new Set(['installed','registered','retired','disabled','ready','initializing','already-installed','already-registered','waiting-dsn','inactive','inactive-dsn','inactive-bbmm','unsupported-system']);
const QUIET_REASONS=new Set(['module-inactive','feature-disabled','unsupported-system','no-generated-name-sample']);
const SUBSTATUS=['chatStatus','sceneStatus','queueStatus','queueCompletionStatus','modelStatus'];
const signatures=new Map();
function gaps(data){
  const found=[],check=(label,status,reason)=>{
    if(status===undefined||QUIET.has(status)||(status==='skipped'&&QUIET_REASONS.has(reason)))return;
    found.push(label+status+(reason?` (${reason})`:''));
  };
  // A feature with independent parts is judged part by part.
  if(data.parts)for(const [key,part] of Object.entries(data.parts))check(key+' ',part?.status,part?.reason);
  else check('',data.status,data.reason);
  for(const key of SUBSTATUS)check(key+' ',data[key]);
  return found;
}
const report=(feature,status,detail)=>{
  const {restore,...data}=typeof feature==='object'?feature:{feature,status,...(detail!==undefined?{detail}: {})};
  const key=data.feature=ALIASES[data.feature]??data.feature;
  state.patches[key]=data;
  // Counters and hook ids change without the outcome changing; log each outcome once.
  const signature=JSON.stringify([data.status,data.reason,data.detail,...SUBSTATUS.map(name=>data[name]),data.parts&&Object.entries(data.parts).map(([name,part])=>[name,part?.status,part?.reason])]);
  if(signatures.get(key)===signature)return;
  signatures.set(key,signature);
  const missing=gaps(data);
  if(missing.length)return console.warn(`${ID} | ${key} is not active: ${missing.join('; ')}`,data.detail??'');
  // Say why something expected is idle, and which parts of an installed repair are still waiting.
  const notes=[data.reason,...SUBSTATUS.filter(name=>data[name]!==undefined&&data[name]!=='installed').map(name=>`${name} ${data[name]}`),
    ...Object.entries(data.parts??{}).filter(([,part])=>part?.status!=='installed').map(([name,part])=>`${name} ${part?.reason??part?.status}`)].filter(Boolean);
  console.info(`${ID} | ${key}: ${data.status}${notes.length?` (${notes.join(', ')})`:''}`,data.detail??'');
};
const enabled=key=>game.settings.get(ID,key);
const run=async(key,fn)=>{try{if(!enabled(key))return report(key,'disabled');await fn();}catch(error){report(key,'failed',String(error));console.error(`${ID} | ${key}`,error);}};
const status=()=>{
  const copy=structuredClone(state),runtime=copy.patches.runtime;
  if(runtime?.status==='initializing')runtime.waitedMs=Date.now()-runtime.since;
  return copy;
};

Hooks.once('init',()=>{
  for(const [key,name] of Object.entries({legacy:'旧导入器数据兼容',itemNames:'PF2e 物品名称性能优化',sundry:'Sundry 效果图标刷新优化',timestamps:'聊天时间戳 DOM 更新优化',chat:'批量删除聊天性能优化',tokenizerChat:'Tokenizer2 聊天头像缩放与遮罩修正',soundStop:'原生缓冲音频停止修正',dsnQualityLocks:'BBMM / DsN 画质硬锁适配',turnLifecycle:'共享回合与奴仆到期兼容修正',dsnChat:'DsN 动画失败后恢复聊天与队列'}))
    game.settings.register(ID,key,{name,scope:'world',config:true,type:Boolean,default:true,requiresReload:true});
  game.modules.get(ID).api={status};
  void run('dsnQualityLocks',()=>registerDsnQualitySettings({report}));
});

Hooks.once('setup',async()=>{
  // Install the quality bridge before upstream ready synchronization.
  void run('dsnQualityLocks',()=>installDsnQualityLocks({moduleId:ID,report}));
  // Register synchronously in setup so the first chat history render is covered.
  void run('tokenizerChat',()=>installTokenizerChatPortraitPatch({moduleId:ID,report}));
  void run('soundStop',()=>installSoundStopPatch({moduleId:ID,report}));
  // The chat log binds its periodic refresh on first render, right after setup.
  void run('timestamps',()=>installTimestampPatch({moduleId:ID,report}));
  // Register before DsN starts its ready-time model preloads.
  void run('dsnChat',()=>installDsnChatRecovery({report}));
  await run('legacy',()=>registerLegacyCompat({moduleId:ID,registerWrapper:(target,fn,type)=>libWrapper.register(ID,target,fn,type),report}));
});

Hooks.once('ready',()=>queueMicrotask(async()=>{
  const started=Date.now();
  // Name the step in progress so a stalled chain is distinguishable from a slow one.
  const step=async(key,fn)=>{state.patches.runtime={status:'initializing',step:key,since:Date.now()};await run(key,fn);};
  await step('itemNames',()=>installItemNameFastPath({report}));
  await step('sundry',async()=>{await prepareSundryPatch();return installSundryPatch({report});});
  await step('turnLifecycle',async()=>{await prepareTurnLifecyclePatch();return installTurnLifecyclePatch({report});});
  await step('dsnChat',()=>installDsnChatRecovery({report}));
  await step('chat',()=>installChatDeleteCoalescing({moduleId:ID,report}));
  report({feature:'runtime',status:'ready',elapsedMs:Date.now()-started});
  const missing=Object.entries(state.patches).flatMap(([key,data])=>gaps(data).map(text=>`${key}: ${text}`));
  if(missing.length)console.warn(`${ID} | ${missing.length} repair(s) not active after startup: ${missing.join(' | ')}`);
}));
