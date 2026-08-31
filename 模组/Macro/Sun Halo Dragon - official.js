/**
 * Sun Halo Dragon —— 官方版调用 (eskie-macro-pack / eskie-macros v1.2.12)
 *
 * 用法：
 *   1. 选中施法者 token
 *   2. 按 T 依次锁定目标（任意数量，6 个也行）
 *   3. 跑本宏
 *   4. 弹出十字准星时，点冲刺的终点位置
 *
 * 跟 Tagger 版的区别：不用打任何标签；火龙从起点拉伸到终点（不是贴在 drag 图块上）；
 * 配色是官方的 orange/red，不是你那版的 redyellow/redorange。
 */

const caster  = canvas.tokens.controlled[0];
const targets = Array.from(game.user.targets);
const api     = globalThis.eskie?.showcase?.sunHaloDragon;

if (!api)             return ui.notifications.error("Eskie Macro Pack 没启用（globalThis.eskie 不存在）");
if (!caster)          return ui.notifications.warn("先选中施法者 token");
if (!targets.length)  return ui.notifications.warn("先按 T 锁定至少一个目标");

// 施法者的显隐由主序列自己收尾，但报错时不会——单独留一个恢复句柄
const restoreCaster = () => new Sequence().animation().on(caster).opacity(1).play();

// 上一次跑崩留下的残影/隐形，先清干净
await api.clean(caster, targets);
await restoreCaster();

try {
  await api.play(caster, targets, {
    impact: false,                          // true = 换成白闪定格帧
    screen: true,                           // 横向速度线
    sound:  { enabled: true, volume: 0.5 },
  });
} catch (err) {
  await api.clean(caster, targets);
  await restoreCaster();
  throw err;
}

// 目标的 alpha 官方版不自动恢复，12 秒后收尾（不想自动清就注释掉这段，改成手动跑 clean）
setTimeout(async () => {
  await api.clean(caster, targets);
  await restoreCaster();
}, 12000);
