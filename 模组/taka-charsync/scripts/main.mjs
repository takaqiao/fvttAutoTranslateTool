import * as generatedRules from '../generated/rules.mjs';
import { catchUpOnLogin } from './char/catch-up.mjs';
import { CHAR_WORLD, actorRuleState, installActorRuleHooks, installOverride } from './char/rules-override.mjs';
import { installRuleEditGuards, startRuleRefresh } from './char/rules-session.mjs';
import { endStatusSession, renderStatusBar } from './char/status-bar.mjs';
import { addCleanupButton, installCopyLock } from './char/submit.mjs';
import { installTokenizerSaveAdapter } from './foundry/tokenizer-save.mjs';
import { lastCheck } from './main/apply.mjs';
import { addPanelButton, checkNow, openPanel, startChecks } from './main/panel.mjs';

// 安装读取入口；具体团规只在 Actor 计算范围使用，不改变 PF2e 初始化读到的世界值。
try {
  installOverride(generatedRules.SNAPSHOT);
} catch (error) {
  console.error('taka-charsync | 团规则适配未安装，建卡服暂不可编辑构筑', error);
}

// PF2e 8.6 的这两种卡面都是 AppV1，钩子按类名派发
const SHEET_CLASSES = ['CharacterSheetPF2e', 'FamiliarSheetPF2e'];

/** 'char'：建卡服世界；'main'：团世界。`init` 时定下。 */
let mode = null;

Hooks.once('init', () => {
  game.settings.register('taka-charsync', 'publishDir', {
    scope: 'world',
    config: false,
    type: String,
    default: '',
  });
  mode = game.world.id === CHAR_WORLD ? 'char' : 'main';
  installTokenizerSaveAdapter();
  if (mode === 'main') {
    game.modules.get('taka-charsync').api = {runApplyCheck: ({trigger}) => checkNow(trigger), openPanel, lastCheck};
    Hooks.on('renderActorDirectory', addPanelButton);
    return;
  }
  // 赶在侧栏第一次渲染、有人改卡之前装上
  installActorRuleHooks();
  installRuleEditGuards();
  installCopyLock();
  Hooks.on('renderActorDirectory', addCleanupButton);
});

Hooks.once('ready', () => {
  if (mode === 'main') {
    // 主服卡只由 GM 客户端写，其中又只有 activeGM 真正动手（runApplyCheck 自己判断）
    if (game.user.isGM) startChecks();
    return;
  }
  for (const sheet of SHEET_CLASSES) {
    Hooks.on(`render${sheet}`, renderStatusBar);
    Hooks.on(`close${sheet}`, endStatusSession);
  }
  startRuleRefresh().then(meta => meta && catchUpOnLogin({meta, canPrepare: actor => actorRuleState(actor).ok}))
    .catch(error => console.error('taka-charsync | 登录时跟上主服失败', error));
});
