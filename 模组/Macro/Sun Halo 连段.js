/**
 * Sun Halo 连段
 *   段1 相机聚焦 → 段2 跳跃冲刺进场(准星①=落点) → 段3 Sun Halo Dragon(准星②=冲刺终点)
 *   → 段4 目标腰斩(主动画内置) → 段5 半身燃烧消散，token 隐形留场
 *
 * 依赖：sequencer / eskie-effects / eskie-macros / jb2a_patreon / psfx（都已装）
 * 用法：选中施法者 token → 按 T 依次锁定目标（数量不限）→ 跑本宏
 *       会弹两次十字准星，别按 Esc 取消
 *
 * 宏类型必须是 Script，不能是 Chat。
 * 跑砸了用配套的「Sun Halo 复原全场」宏收拾。
 */

// ─────────── 配置 ───────────
const CFG = {
  // 相机：'bbox' 推到全体目标的包围盒中心并自动算缩放（多目标推荐）
  //       'first' 第一个 T 锁定的目标 / 'pick' 跑宏时用准星点一下 / 'off' 不动镜头
  camera:         "bbox",
  cameraScale:    0.5,         // 'first' / 'pick' 用的固定倍率
  cameraPadding:  3,           // 'bbox' 模式下包围盒外留几格
  cameraScaleMin: 0.2,
  cameraScaleMax: 1.0,

  cinemaBars:     true,        // 电影黑边挂满全程
  speedLineColor: "redyellow", // 官方要的是 orange，库里没有会静默退成 black（全屏发黑），所以自己补一条
  impact:         false,       // true = 主动画换成白闪定格帧
  sound:          { enabled: true, volume: 0.5 },

  corpseAppearAt: 3000,        // 主序列开始后多久让「尸体半身」淡入（官方半身 2500~3500 淡出，正好交接）
                               // 看到上半身出现重影 = 调大；看到中间断了一拍 = 调小
  corpseFadeIn:   700,
  corpseFadeOut:  1200,
  burnDelay:      800,         // 主序列跑完后多久开始烧
  burnHold:       900,         // 烧起来多久后让半身淡掉
};

const FLAG_SCOPE = "world";
const FLAG_KEY   = "sunHaloAlpha";
const CORPSE     = "SunHaloCorpse";

// ─────────── 工具 ───────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 上下半身用的三角形遮罩（对角线切过 token 中心，所以任何尺寸都对）
const maskTop = () => ({
  lineSize: 1, lineColor: "#FF0000", fillColor: "#FF0000", fillAlpha: 1,
  points: [{ x: -1, y: -1 }, { x: 1, y: 1 }, { x: 1, y: -1 }],
  gridUnits: true, isMask: true, name: "corpseTop",
});
const maskBottom = () => ({
  lineSize: 1, lineColor: "#FF0000", fillColor: "#FF0000", fillAlpha: 1,
  points: [{ x: -1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }],
  gridUnits: true, isMask: true, name: "corpseBottom",
});

// 上半身相对 token 中心的位移，跟官方 gs*0.25 对齐，交接才不跳
const OFF_TOP = { offset: { x: 0.25, y: -0.25 }, gridUnits: true };

// ─────────── 相机 ───────────
function targetsBBox(list) {
  const gs = canvas.grid.size;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const t of list) {
    const hw = (t.document.width  * gs) / 2;
    const hh = (t.document.height * gs) / 2;
    minX = Math.min(minX, t.center.x - hw); maxX = Math.max(maxX, t.center.x + hw);
    minY = Math.min(minY, t.center.y - hh); maxY = Math.max(maxY, t.center.y + hh);
  }
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, w: maxX - minX, h: maxY - minY };
}

async function focusCamera(list) {
  if (CFG.camera === "off") return;

  let x, y, scale = CFG.cameraScale;

  if (CFG.camera === "pick") {
    const p = await Sequencer.Crosshair.show({ label: "相机聚焦点", drawIcon: true, drawOutline: true });
    if (!p || p.cancelled) return;
    x = p.x; y = p.y;
  } else if (CFG.camera === "first") {
    const t = list[0];
    x = t.center.x; y = t.center.y;
  } else {
    const b = targetsBBox(list);
    x = b.cx; y = b.cy;
    const pad = CFG.cameraPadding * canvas.grid.size * 2;
    const vw = canvas.app?.renderer?.screen?.width  ?? window.innerWidth;
    const vh = canvas.app?.renderer?.screen?.height ?? window.innerHeight;
    scale = Math.min(vw / (b.w + pad), vh / (b.h + pad));
    scale = Math.max(CFG.cameraScaleMin, Math.min(CFG.cameraScaleMax, scale));
  }

  await new Sequence().canvasPan({ duration: 400, x, y, scale }).play();
}

// ─────────── 保险丝 ───────────
async function restoreAll() {
  await Sequencer.EffectManager.endEffects({ name: `${CORPSE}|*` });
  for (const t of canvas.tokens.placeables) {
    const orig = t.document.getFlag(FLAG_SCOPE, FLAG_KEY);
    if (orig === undefined) continue;
    await t.document.update({ alpha: orig });
    await t.document.unsetFlag(FLAG_SCOPE, FLAG_KEY);
  }
}

async function armFuse(tokens) {
  for (const t of tokens) {
    if (t.document.getFlag(FLAG_SCOPE, FLAG_KEY) === undefined) {
      await t.document.setFlag(FLAG_SCOPE, FLAG_KEY, t.document.alpha ?? 1);
    }
  }
}

// ─────────── 特效 ───────────
// 补一条颜色正确的速度线（官方那条被 screen:false 关掉了）
function speedLines() {
  return new Sequence()
    .effect()
      .file(`eskie.screen_overlay.speed_lines.horizontal.02.${CFG.speedLineColor}`)
      .screenSpace()
      .screenSpaceScale({ fitX: true, fitY: true })
      .mirrorX()
      .delay(200)
      .duration(2500)
      .fadeOut(500);
}

/* 「尸体半身」：官方的半身 3.5 秒后就淡没了，这两片接管，一直躺到最后被烧掉。
   注意效果从 t=0 就建好（此时 token 还没被设成 alpha 0），靠 fadeIn 的 delay 延后显形——
   否则开了动态 token 环(ring)的 token 会因为 copySprite 复制到透明的 mesh 而整片看不见。*/
function corpses(list) {
  const seq = new Sequence();
  for (const t of list) {
    seq.effect()
      .name(`${CORPSE}|${t.id}|Top`)
      .copySprite(t)
      .spriteRotation(-t.document.rotation)
      .scaleToObject(1, { considerTokenScale: true })
      .atLocation(t, OFF_TOP)
      .shape("polygon", maskTop())
      .fadeIn(CFG.corpseFadeIn, { delay: CFG.corpseAppearAt })
      .persist()
      .extraEndDuration(CFG.corpseFadeOut)
      .fadeOut(CFG.corpseFadeOut)
      .zIndex(0.15);

    seq.effect()
      .name(`${CORPSE}|${t.id}|Bottom`)
      .copySprite(t)
      .spriteRotation(-t.document.rotation)
      .scaleToObject(1, { considerTokenScale: true })
      .atLocation(t)
      .shape("polygon", maskBottom())
      .fadeIn(CFG.corpseFadeIn, { delay: CFG.corpseAppearAt })
      .persist()
      .extraEndDuration(CFG.corpseFadeOut)
      .fadeOut(CFG.corpseFadeOut)
      .zIndex(0.1);
  }
  return seq;
}

// 段5：把切开状态的半身烧掉
function burnAway(list) {
  const seq = new Sequence();
  for (const t of list) {
    for (const [off, mask, z] of [[OFF_TOP, maskTop, 1.1], [{}, maskBottom, 1.0]]) {
      seq.effect()
        .file("eskie.burn.token_mask.orange.no_base.fast.01")
        .atLocation(t, off)
        .scaleToObject(1.1, { considerTokenScale: true })
        .shape("polygon", mask())
        .zIndex(z);
    }

    seq.effect()
      .file("eskie.burn.embers.orange")
      .atLocation(t, OFF_TOP)
      .scaleToObject(1.5, { considerTokenScale: true })
      .mirrorX()
      .fadeIn(400)
      .zIndex(2);

    seq.effect()
      .file("eskie.burn.embers.orange")
      .atLocation(t)
      .scaleToObject(1.5, { considerTokenScale: true })
      .mirrorX()
      .spriteRotation(-45)
      .fadeIn(400)
      .zIndex(2);

    seq.effect()
      .delay(200)
      .file("eskie.particle.04.orange")
      .atLocation(t, { randomOffset: 0.4, gridUnits: true })
      .scaleToObject(2, { considerTokenScale: true })
      .randomRotation()
      .zIndex(3);
  }

  seq.wait(CFG.burnHold);
  seq.thenDo(async () => {
    await Sequencer.EffectManager.endEffects({ name: `${CORPSE}|*` });
    for (const t of list) await t.document.update({ alpha: 0 }); // 落定：隐形留场，不删除
  });
  return seq;
}

// ─────────── 主流程 ───────────
const caster  = canvas.tokens.controlled[0];
const targets = Array.from(game.user.targets);
const api     = globalThis.eskie;

if (!api?.showcase?.sunHaloDragon) return ui.notifications.error("Eskie Macro Pack 没启用（globalThis.eskie 不存在）");
if (!api?.effect?.leap)            return ui.notifications.error("找不到 eskie.effect.leap");
if (!caster)         return ui.notifications.warn("先选中施法者 token");
if (!targets.length) return ui.notifications.warn("先按 T 锁定至少一个目标");

await restoreAll();                       // 清上一次的残留
await armFuse([caster, ...targets]);      // 记下原始 alpha

try {
  if (CFG.cinemaBars) await api.overlay.cinemaBars.play({ dim: true });

  // 段1 ── 相机聚焦（多目标就推到包围盒中心，保证都在画面里）
  await focusCamera(targets);

  // 段2 ── 跳跃冲刺进场，准星① = 落点
  await api.effect.leap.play(caster);

  // 段3/4 ── 主动画。先 create()（弹准星②并把序列建好），再让三条同时起跑
  const main = await api.showcase.sunHaloDragon.create(caster, targets, {
    impact: CFG.impact,
    screen: false,                                    // 关掉官方那条会变黑的速度线
    sound:  CFG.sound,
  });

  speedLines().play();          // 不 await：跟主序列同一刻起跑
  corpses(targets).play();      // 不 await
  await main.play();

  // 段5 ── 烧掉
  await sleep(CFG.burnDelay);
  await burnAway(targets).play();

  if (CFG.cinemaBars) await api.overlay.cinemaBars.stop();
} catch (err) {
  await Sequencer.EffectManager.endEffects({ name: `${CORPSE}|*` });
  await api.overlay?.cinemaBars?.stop?.();
  await restoreAll();
  ui.notifications.error("演出中断，已复原。详见 F12 控制台");
  throw err;
}
