# -*- coding: utf-8 -*-
"""往自检面板 C 档加「命名空间被顶 · 抢回」一节，并给 I18N_PROBES 补两条被顶过的键。
两份孪生文件（ember / crucible）逐字节相同，改完必须都写。"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')

TARGETS = [
    '1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs',
    '2-Crucible汉化插件/selfcheck/cn-selfcheck.mjs',
]

PROBE_OLD = '''  ["SPELL.INFLECTIONS.Aura", "奥拉", "Ember 的月亮/同调/屈折 Aura＝奥拉（Crucible 的手势/目标类型才是灵气）"],
];'''
PROBE_NEW = '''  ["SPELL.INFLECTIONS.Aura", "奥拉", "Ember 的月亮/同调/屈折 Aura＝奥拉（Crucible 的手势/目标类型才是灵气）"],
  // 下面两条来自**被第三方核心汉化包整块顶掉过**的那 42 条（详见本档「命名空间被顶 · 抢回」）。
  // 放进探针不是为了再查一遍那一节 —— 那一节读的是抢回器自己的账本，这两条走的是
  // `game.i18n.localize()` 这条**玩家真正用的**通道，两边同时绿才说明「账上抢回了」
  // 和「屏幕上真是中文」是同一件事。crucible-cn 没装时它们会落进「取不到、没查」，不是通过。
  ["TOKEN.MOVEMENT.ACTIONS.walk.label", "行走", "TOKEN 命名空间被顶掉过；这条是移动方式下拉与移动标尺上的字"],
  ["WARNING.NoParty", "主要队伍", "WARNING 命名空间被顶掉过；这条是没配主要队伍时的报错提示"],
];'''

CALL_OLD = '''      : Check.ok(S, "裁决键抽验", checked,
          `查 ${checked}/${I18N_PROBES.length} 条，全部符合既定裁决。` + absentNote, absent));
  }
  return out;
}'''
CALL_NEW = '''      : Check.ok(S, "裁决键抽验", checked,
          `查 ${checked}/${I18N_PROBES.length} 条，全部符合既定裁决。` + absentNote, absent));
  }
  out.push(...checkLangSquat(S, table));
  return out;
}

/**
 * 「自家的命名空间被别的语言包整块顶掉了没有，顶了的话抢回来了没有」。
 *
 * ── 为什么这一节非有不可 ──────────────────────────────────────────────────
 * 抢回是**静默**的，而且三种状态在屏幕上长得一模一样或差不多：
 *   ① 没人顶 ⇒ 正常中文；② 顶了、抢回成功 ⇒ 正常中文；③ 抢回器根本没跑 ⇒ 英文，**且不报错**。
 * ①②肉眼分不出，②③只有装了 `foundry_chn` 的人才碰得到 —— 维护者自己的机器上很可能永远
 * 复现不了。⇒ 只能由抢回器如实记账、面板照实报。这与本文件开头那条
 * 「0 个问题 和 没查到东西 必须分开显示」是同一条原则的第 N 次应用。
 *
 * ── 不信账本，当场复验 ────────────────────────────────────────────────────
 * 账本说「抢回了 42 条」只是说**写入动作做了**。这一节还会把这 42 条逐条丢回
 * `game.i18n.localize()` 再看一眼 —— 那是玩家真正走的通道。账本说抢回了、
 * 复验却仍是英文，就是判据空转的第 (h) 种形态（探针自己捏输入），必须报失败。
 *
 * @param {string} S 档名
 * @param {Record<string, unknown>} table `game.i18n.translations`
 * @returns {Check[]}
 */
function checkLangSquat(S, table) {
  const NAME = "命名空间被顶 · 抢回";
  const pkg = mod("crucible-cn");
  if (!pkg?.active) {
    return [Check.skip(S, NAME, "`crucible-cn` 没装或没启用 —— 抢回器随它一起来，本档无从查起。")];
  }
  const getState = pkg.api?.getReclaimState;
  if (typeof getState !== "function") {
    return [Check.fail(S, NAME, 0,
      "`crucible-cn`（" + (pkg.version ?? "版本取不到") + "）装着，但**没挂上抢回账本**。" +
      "要么这一版早于 `0.9.16`（那时抢回器还不记账），要么 `i18nInit` 那一步整个没跑起来。" +
      "⚠ **这不是「通过」，是查不出成败** —— 真被顶掉的话，屏幕上那 42 条会静静显示英文原文。")];
  }
  let st;
  try { st = getState(); }
  catch (err) { return [Check.fail(S, NAME, 0, "读抢回账本时抛了：`" + (err?.message ?? err) + "`")]; }

  if (st.phase === "error") {
    return [Check.fail(S, NAME, 0,
      "抢回器跑了但**抛了异常**：`" + (st.error ?? "（没记下消息）") + "`。" +
      "⇒ 若自家命名空间此刻正被顶着，受害的键会显示英文。")];
  }
  if (st.phase === "no-table") {
    return [Check.fail(S, NAME, 0,
      "抢回器跑了，但在 `module.json` 的 `languages` 里**找不到自家的 `cn` 声明** ⇒ 无表可抢。" +
      "上游改了 `languages` 的形状，或打包漏了 `lang/cn.json`。")];
  }
  if (st.phase !== "done") {
    return [Check.fail(S, NAME, 0,
      "抢回账本停在 `" + st.phase + "` —— `i18nInit` 那一步没跑到底。")];
  }

  const out = [];
  const rec = st.report?.reclaimed ?? [];
  const scanned = st.tableSize || 0;

  if (!rec.length) {
    out.push(Check.ok(S, NAME, scanned,
      "扫了自家 `lang/cn.json` 的 **" + scanned + " 个键**，**一个都没被顶掉**，本次写入 0 次。" +
      "⚠ 这**不等于**抢回器没用 —— 它确实跑完了整表；只是这台机器上没装会顶掉命名空间的语言包" +
      "（典型肇事者是 `foundry_chn`，A 档里能看到它装没装）。"));
    return out;
  }

  // ① 被顶掉的**命名空间根**逐个点名，连同它现在被换成了什么。
  const roots = [...new Set(rec.map((k) => k.split(".")[0]))];
  const rootLines = roots.map((r) => {
    const v = table[r];
    const shown = (typeof v === "string") ? "字符串「" + v + "」"
      : (v === undefined ? "**根本不存在**（不是被顶，是压根没人建这个命名空间）" : "`" + typeof v + "`");
    const n = rec.filter((k) => k.split(".")[0] === r).length;
    return "`" + r + "` —— 顶层现在是 " + shown + "，本项目在它下面的 **" + n + " 条**被整块盖掉";
  });

  // ② **不信账本，当场复验**：走玩家真正用的 localize 通道再看一眼。
  const stillEnglish = rec.filter((k) => !hasCJK(game.i18n.localize(k)));
  if (stillEnglish.length) {
    out.push(Check.fail(S, NAME, rec.length,
      "账本说抢回了 **" + rec.length + "** 条，但**复验时仍有 " + stillEnglish.length +
      " 条取不到中文** —— 写入动作做了、玩家通道上却没生效。逐条点名：",
      rootLines.concat(["（复验仍是英文的键）"], cappedList(stillEnglish))));
  } else {
    out.push(Check.ok(S, NAME, rec.length,
      "扫了自家 `lang/cn.json` 的 **" + scanned + " 个键**，其中 **" + rec.length +
      " 条**被别的语言包整块顶掉，已全部抢回；**逐条回 `game.i18n.localize()` 复验，" +
      rec.length + "/" + rec.length + " 条现在都取得到中文**。被顶掉的命名空间：",
      rootLines));
  }

  // ③ 跳过的两类要单独说 —— 它们既不是「抢回了」也不是「没被顶」。
  const so = st.report?.skippedObject ?? [];
  const sb = st.report?.skippedBadValue ?? [];
  if (so.length || sb.length) {
    out.push(Check.warn(S, NAME + " · 跳过项", so.length + sb.length,
      "抢回时**刻意跳过** " + (so.length + sb.length) + " 条：" +
      so.length + " 条的位置被**别人的命名空间对象**占着（写扁平键会把人家整个遮住），" +
      sb.length + " 条自家的值**不是字符串**（那正是 v1.1.25 的死法，不许再往 `translations` 里塞对象）。",
      cappedList(so.map((k) => "`" + k + "` —— 位置被命名空间对象占用")
        .concat(sb.map((k) => "`" + k + "` —— 自家值不是字符串 ⚠ 应由 `R-lang-flat-keys` 挡住，它漏了")))));
  }
  return out;
}'''

for rel in TARGETS:
    p = rel
    s = io.open(p, encoding='utf-8', newline='').read()
    nl = "\r\n" if "\r\n" in s else "\n"
    for old, new in ((PROBE_OLD, PROBE_NEW), (CALL_OLD, CALL_NEW)):
        o = old.replace("\n", nl); n = new.replace("\n", nl)
        assert s.count(o) == 1, (p, s.count(o), old[:40])
        s = s.replace(o, n)
    io.open(p, 'w', encoding='utf-8', newline='').write(s)
    print("已改", p)
