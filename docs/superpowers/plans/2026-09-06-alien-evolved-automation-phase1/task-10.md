> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 10 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 10: 掷骰卡上的合并成功数行（`roll-record-and-success-line`）

**背景（实现者必读，不需要任何 Alien RPG 或 Foundry 前置知识）**

Foundry VTT 是一个网页版桌游平台，游戏规则由「系统」（system）提供，第三方「模组」（module）在旁边加功能。这里的系统是 `alienrpg` 4.1.13，本任务写的是模组 `alien-evolved-automation` 里的一条特性。

`alienrpg` 的掷骰卡今天把成功数拆成两处打印：黑色基础骰的六点数在 `systems/alienrpg/module/helpers/YZEDiceRoller.mjs:456-460`（`ALIENRPG.Sixes` 后面跟 `${R6.length}`），黄色压力骰的一点数与六点数在 `:547-553`（`ALIENRPG.Ones` 与 `ALIENRPG.Sixes`）。规则上「任一颜色的每个 6 都是一个成功」，而 `:287-333` 那段「计算总成功数」的 `switch` 只有辐射两个 case 会打印合计，`:332-333` 的 `default:` 是空的 `break` —— 也就是说普通掷骰卡上根本没有合计数，玩家每掷一次都要自己把两个数字加起来。本任务在卡上补这一行。

四条硬约束：

1. 成功数**只能从消息 flag 里的 RollRecord 读**。绝不解析渲染后的 HTML 文本（"Sixes: " 是 `ALIENRPG.Sixes`，会被 Babele 汉化模组改写），也绝不读 `Roll#total`（`module/helpers/alienRPGBaseDice.mjs:13` 与 `:43` 两个骰子类的 `get total()` 都是 `return this.results.length`，返回的是池子大小，不是点数和）。
2. 卡上已经有合并总数的两种情况不要重复打印：`YZEDiceRoller.mjs:289-331` 的辐射两个分支（`ALIENRPG.Radiation` / `ALIENRPG.RadiationReduced` 已经打印「你受到 N 点伤害」），以及 `:287` 的 `if (actortype !== "supply")` —— 补给掷骰整段都不进这个块（它在 `:244-249` 改打「补给下降」），补给骰数的是 1 不是 6，算成功没有意义。
3. **只加一行，不删系统那两行 "Sixes"**。这是一条被记录在案的显示决定，理由有二：删它们必须靠文本或畸形 HTML 去定位（`:457` 生成的是 `<span <span class=...>` 这种没闭合的标签），那正是第 1 条禁止的做法；而且黑黄两排骰面图本身有用。结果是卡上会有三个数字：黑池 6 数、黄池 6 数、以及我们这行合计。手工验收清单里要把「基础骰单掷 / 带压力骰 / 推骰 / 补给 / 辐射」五种卡都过一遍，确认数字互相对得上。
4. 「本次新增」而不是「累计」：`RollRecord.push.parentRollId` 一期确实由内核的 `pushRoll` 包装器产出，理论上可以顺着它回溯算累计。**本任务刻意不这么做** —— 系统自己在 `:340-370` 已经打印了一个累计数（`oldRoll + multiPush + r1Six + r2Six`），我们再打印第二个累计数，只要两者口径有一丝差异，同一张卡上就会出现两个互相矛盾的总数。所以本特性只消费 `push.count`（用于切换文案），**不消费 `parentRollId`**（该字段一期只产出不消费，消费者在二期的推骰历史特性）。

**名词解释**

- **Hook**：Foundry 的全局事件总线。`Hooks.on("renderChatMessageHTML", fn)` 表示「每当一条聊天消息被渲染成 DOM 时调用 fn」。**本特性不挂任何钩子** —— 按契约 §0.2，`renderChatMessageHTML` 归 `kernel/cards.mjs` 独占，特性只能被 cards 回调，通路是契约 §4 K8 的 `cards.onRender(name, handler)`。
- **flag**：挂在 Foundry 文档（这里是 ChatMessage）上的、按模组 id 分命名空间的任意 JSON 数据。本模组把每次掷骰的结构化结果（RollRecord）写进 `message.flags["alien-evolved-automation"].roll`。本特性**不自己去读 flag**：`cards.onRender` 的回调参数里已经带了 `record`，那就是 `rollBus.recordOf(message)` 的结果，非掷骰卡上为 `null`。
- **labelKey**：`yzeRoll` 收到的 `label` 是**已经本地化过的文本**（`YZEDiceRoller.mjs:79/90/289` 全都拿 `game.i18n.localize("ALIENRPG.Radiation")` 去比对它）。内核在 `i18nInit` 阶段由 `main.mjs` 注入一张「译文 → i18n 键」的反查表，组装记录时把 `label` 还原成键写进 `RollRecord.labelKey`（同一段译文对应多个键时该文本映射为 `null`）。所以本特性判断「这是不是辐射卡」查的是 `labelKey` 这个键，不是译文；**这两件事都不属于本任务，本任务只消费 `labelKey`。**
- **挂点（mount）与分区（section）**：`kernel/cards.mjs` 在每张卡上准备一个模组自有的 `<div class="aea-mount">`，一期有多条特性往同一个挂点里写东西。按契约 §4 K8 [v3.1]，`cards.render(target, ...)` **只替换 `target` 自身的 `innerHTML`**，所以谁把挂点本身当渲染目标，谁就会清掉别人写的内容。因此**每条特性必须先在挂点下建一个属于自己的子元素**，类名逐字为 `aea-<feature-id>`（本特性即 `aea-roll-record-and-success-line`），用 `mount.querySelector` 复用、没有才创建，再对那个子元素调 `render()`。
- **三档开关**：契约 §4 K5 的每条特性有 `"full" | "prompt" | "off"` 三档，`features.enabled(id)` 对前两档都返回 `true`。本特性是**只读展示**，卡上多打一行数字，没有任何可供玩家或 GM 确认的动作 —— 拿一张确认卡去问「要不要打印这个数字」是荒谬的。因此本特性只查 `features.enabled(id)`，`"prompt"` 与 `"full"` 行为完全相同。**这是刻意的取舍，不是漏实现**；一期的「提示并确认」交互卡由二期的执行器负责，本特性届时也不会有可确认的动作。

**Files:**
- Create: `scripts/features/roll-record-and-success-line.pure.mjs`
- Create: `scripts/features/roll-record-and-success-line.mjs`
- Create: `templates/success-line.hbs`
- Modify: `lang/en.json`（顶层 `AEA` 对象内）
- Modify: `lang/cn.json`（顶层 `AEA` 对象内）
- Modify: `scripts/main.mjs`（**只在两个锚点各插一行**：`/* AEA-ANCHOR: imports */` 之后插一行 import、`/* AEA-ANCHOR: features */` 之后插一个数组成员。其余八个锚点 —— `repairs`、`init`、`i18nInit`、`diceSoNiceReady`、`ready.registry`、`ready.patches`、`ready.rollbus`、`ready.cards` —— 一行都不加；`export const api` 那个八键字面量也一个字不改，它只属于内核任务）
- Test: `test/roll-record-and-success-line.pure.test.mjs`
- Test: `test/roll-record-and-success-line.wiring.test.mjs`

路径均相对模组仓库根目录 `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation`。

**Interfaces:**

- Consumes:
  - `scripts/const.mjs` → `MID = "alien-evolved-automation"`
  - `scripts/kernel/features.mjs` → `features.register(def)`（`def = {id, default:"full"|"prompt"|"off", gmOnly:false, requires:[], hint:""}`）、`features.enabled(id) -> boolean`、`features.all() -> def[]`
  - `scripts/kernel/cards.mjs` → `cards.onRender(name, handler)`：契约 §4 K8 的渲染扇出点，也是特性拿到挂点的**唯一**通路。cards 在自己那个唯一的渲染监听器里按注册顺序**同步**调用各 handler，参数是一个对象 `{message, element, mount, record}`：`message` 是 ChatMessage 文档，`element` 是这条消息渲染出的 DOM 根，`record` 是 `rollBus.recordOf(message)`（非掷骰卡上为 `null`），`mount` 是模组自有的 `<div class="aea-mount">`。三条契约 [v3.1] 消歧，逐条影响本任务的写法：①`mount` **永不为 `null`**（cards 保证每张进入渲染钩子的卡都有可插入位，消息根元素本身即最后回退位），**因此不要写判空**；②`mount` 是**惰性创建**的 —— 只有 handler 第一次读它才真正插进 DOM，所以早退的 handler 必须在读 `mount` 之前返回，否则会在每张非掷骰卡上留下一个空挂点；③handler **必须同步返回**，cards **不 await** 它的返回值，只捕获它**同步段**抛出的异常（记 `console.error` 后继续调用下一个），**异步段抛出的它捕不到**，所以异步尾巴必须自己 `.catch`。
  - `scripts/kernel/cards.mjs` → `cards.render(target, templatePath, data) -> Promise<void>`：`await renderTemplate(templatePath, data)` 之后**只把结果写进 `target` 自身的 `innerHTML`**，绝不触碰它的父节点、兄弟节点或挂点上的其它内容，并**返回 Promise**（调用方 `await` 它，「渲染完成」因此可观测）。本任务传进去的 `target` 是自己在挂点下建的 `aea-roll-record-and-success-line` 分区元素，**绝不是 `cards.mount()` 那个共用挂点**。
  - `scripts/kernel/dice-barrier.mjs` → `diceBarrier.awaitDice(message) -> Promise<void>`：装了 Dice So Nice（3D 骰子动画模组）时，渲染时刻骰子还在空中，这个 Promise 等到动画落地才 resolve；没装 DsN 或本客户端不是收件人时立即 resolve。
  - `scripts/kernel/rollbus.mjs` → `rollBus.recordOf(message) -> RollRecord|null`（**仅自检条目用**：自检要遍历世界里已有的消息；渲染路径上的 record 由 `cards.onRender` 直接给）
  - `scripts/kernel/selftest.mjs` → `selftest.register({id, label, run})`，`run()` 返回 `{ok:boolean, detail:string}`，可 async
  - RollRecord v1 字段（契约 §3）：`v`、`kind`（一期取值只有 `"attribute"|"skill"|"weapon"|"armor"|"supply"|"other"`）、`labelKey`、`successes`、`banes`、`results.{baseSixes,baseOnes,stressSixes,stressOnes}`、`push.count`
  - `test/stubs/foundry.mjs` → `installFoundryStub(options) -> ctx`、`uninstallFoundryStub()`：契约 §0.3 的共享桩，由别的任务独占实现并以 `test/stub-fidelity.test.mjs` 守卫。本任务**只读不改**，**禁止**在自己的测试文件里就地造 `globalThis.game`、也禁止用私有 Map 顶替 `game.settings`。
- Produces:
  - `pureSuccessLineId() -> "roll-record-and-success-line"`
  - `pureSuccessLineKeys() -> string[]`（本特性用到的 7 个 i18n 键，语言包测试、运行期自检共用同一份）
  - `pureSuccessLineClasses() -> {section, main, banes}`（模板、渲染层、自检共用同一份类名，改名不会静默失效；`section` 的值逐字等于 `aea-` 加特性 id，即契约 §4 K8 [v3.1] 要求的分区类名）
  - `pureSuccessLineModel(record) -> {show:boolean, successes:number, banes:number, pushCount:number}`
  - `pureSuccessLineText(model, t) -> {tone:string, main:string, banes:string}|null`，`t(key, data?) -> string` 为注入的本地化函数
  - `successLineFeature = { id, register(), install() }`
  - `templates/success-line.hbs`
  - `test/roll-record-and-success-line.pure.test.mjs`（17 例）、`test/roll-record-and-success-line.wiring.test.mjs`（10 例）
  - `selftest` 条目 `roll-record-and-success-line.cards-agree-with-records`
  - `scripts/main.mjs` 两个锚点各一行（import 与 `FEATURES` 数组成员）

  **无法写成单测的断言 → 替代覆盖（一一对应，删掉替代物时这张表会变成孤行）**：

  | 想断言的事 | 为什么 vitest 做不到 | 替代覆盖 |
  |---|---|---|
  | 同一张卡重渲染后只有一行成功数 | 需要真实 ChatLog 重渲染时序，桩无法忠实模拟 | 接线测「分区复用」两例（同一挂点渲染两次只 `appendChild` 一次、两次目标是同一个元素）+ selftest `…cards-agree-with-records`（每张卡恰好 1 行）+ MANUAL VERIFICATION 第 4 步 |
  | 渲染目标是本特性自己的分区，不是共用挂点 | 需要真实 DOM 才能看出互相覆盖 | 接线测「分区复用」第 1 例（`target !== mount` 且 `target.className === "aea-roll-record-and-success-line"`）+ MANUAL VERIFICATION 第 4 步后半 |
  | 卡上打印的数字与该卡自己的记录一致 | 需要真实渲染出的 DOM 文本 | selftest 同上（`textContent` 含 `model.successes`）+ MANUAL VERIFICATION 第 2 步 |
  | 抑制卡（补给／辐射）上不出现成功数行 | 抑制判定本身是纯函数，但「卡上没有」要看 DOM | 纯测 `pureSuccessLineModel` 三例 + 接线测「早退不读 mount」两例 + selftest 的 `!model.show` 分支 + MANUAL VERIFICATION 第 5、6 步 |
  | DsN 动画落地前不往挂点写字 | 桩无法模拟 DsN 的 3D 动画时序 | **仅** MANUAL VERIFICATION 第 3 步（唯一覆盖，删不得） |
  | 关成 off 后新卡上没有这一行 | 依赖设置面板与真实渲染 | 接线测「off 时早退且不读 mount」+ MANUAL VERIFICATION 第 9 步 |

---

- [ ] **Step 1: 写下会失败的纯函数测试**

新建 `test/roll-record-and-success-line.pure.test.mjs`。这一层完全不碰 Foundry 全局，是真单测。

```js
import { describe, it, expect } from "vitest";
import {
  pureSuccessLineClasses,
  pureSuccessLineId,
  pureSuccessLineKeys,
  pureSuccessLineModel,
  pureSuccessLineText,
} from "../scripts/features/roll-record-and-success-line.pure.mjs";

/** 测试用的假 i18n：把键和参数原样拼出来，方便断言用了哪个键 */
const t = (key, data) => (data ? `${key}|${JSON.stringify(data)}` : key);

/** 一条完整的 v1 RollRecord，按 CONTRACT §3 的 schema 逐字段写全 */
function makeRecord(over = {}) {
  return {
    v: 1,
    id: "roll-1",
    actorUuid: "Actor.abc",
    tokenUuid: null,
    userId: "user-1",
    kind: "skill",
    label: "Rolling Ranged Combat",
    labelKey: null,
    attr: null,
    itemUuid: null,
    pools: { base: 5, stress: 2 },
    results: { baseSixes: 1, baseOnes: 0, stressSixes: 1, stressOnes: 1 },
    successes: 2,
    banes: 1,
    push: { count: 0, pushable: true, parentRollId: null },
    targets: [],
    consumed: { ammo: null },
    at: { worldTime: 0, real: 0 },
    ...over,
  };
}

describe("pureSuccessLineModel", () => {
  it("把黑池六与黄池六合并成一个成功数", () => {
    const m = pureSuccessLineModel(makeRecord());
    expect(m.show).toBe(true);
    expect(m.successes).toBe(2);
    expect(m.banes).toBe(1);
    expect(m.pushCount).toBe(0);
  });

  it("successes 字段缺失时从 results 现算，不读任何别的来源", () => {
    const m = pureSuccessLineModel(makeRecord({ successes: undefined, banes: undefined }));
    expect(m.successes).toBe(2);
    expect(m.banes).toBe(1);
  });

  it("补给掷骰不显示成功行", () => {
    expect(pureSuccessLineModel(makeRecord({ kind: "supply" })).show).toBe(false);
  });

  it("两种辐射掷骰不显示成功行（系统自己已打印伤害总数）", () => {
    expect(pureSuccessLineModel(makeRecord({ labelKey: "ALIENRPG.Radiation" })).show).toBe(false);
    expect(pureSuccessLineModel(makeRecord({ labelKey: "ALIENRPG.RadiationReduced" })).show).toBe(false);
  });

  it("labelKey 反查失败（null）时照常显示 —— 不允许改用译文匹配来兜底", () => {
    expect(pureSuccessLineModel(makeRecord({ labelKey: null })).show).toBe(true);
  });

  it("护甲掷骰照常显示（护甲成功数就是挡下的伤害）", () => {
    expect(pureSuccessLineModel(makeRecord({ kind: "armor" })).show).toBe(true);
  });

  it("非 v1 记录与空记录一律不显示", () => {
    expect(pureSuccessLineModel(null).show).toBe(false);
    expect(pureSuccessLineModel(undefined).show).toBe(false);
    expect(pureSuccessLineModel(makeRecord({ v: 2 })).show).toBe(false);
  });

  it("推骰卡带出 push.count", () => {
    const r = makeRecord({ push: { count: 1, pushable: false, parentRollId: "roll-0" } });
    expect(pureSuccessLineModel(r).pushCount).toBe(1);
  });
});

describe("pureSuccessLineText", () => {
  it("show 为 false 时返回 null，渲染层据此整段跳过", () => {
    expect(pureSuccessLineText({ show: false, successes: 3, banes: 0, pushCount: 0 }, t)).toBe(null);
    expect(pureSuccessLineText(null, t)).toBe(null);
  });

  it("零成功用 none 键，色调用系统真正的蓝（alienchatblue）", () => {
    const d = pureSuccessLineText({ show: true, successes: 0, banes: 0, pushCount: 0 }, t);
    expect(d.main).toBe("AEA.successLine.none");
    expect(d.tone).toBe("alienchatblue");
    expect(d.banes).toBe("");
  });

  it("单个成功用单数键、绿色", () => {
    const d = pureSuccessLineText({ show: true, successes: 1, banes: 0, pushCount: 0 }, t);
    expect(d.main).toBe("AEA.successLine.one");
    expect(d.tone).toBe("alienchatlightgreen");
  });

  it("多个成功用复数键并带上数量", () => {
    const d = pureSuccessLineText({ show: true, successes: 3, banes: 0, pushCount: 0 }, t);
    expect(d.main).toBe('AEA.successLine.many|{"n":3}');
  });

  it("推骰卡用「本次新增」文案，避免和系统 :340-370 的累计行冲突", () => {
    const d = pureSuccessLineText({ show: true, successes: 2, banes: 0, pushCount: 1 }, t);
    expect(d.main).toBe('AEA.successLine.pushDelta|{"n":2}');
  });

  it("有压力骰 1 时给出 banes 文案", () => {
    const d = pureSuccessLineText({ show: true, successes: 1, banes: 2, pushCount: 0 }, t);
    expect(d.banes).toBe('AEA.successLine.banes|{"n":2}');
  });
});

describe("纯层的标识、键表与类名访问器", () => {
  it("特性 id 是逐字的 kebab-case 串", () => {
    expect(pureSuccessLineId()).toBe("roll-record-and-success-line");
  });

  it("键表覆盖 5 个文案键与 2 个特性描述键，顺序固定", () => {
    expect(pureSuccessLineKeys()).toEqual([
      "AEA.successLine.none",
      "AEA.successLine.one",
      "AEA.successLine.many",
      "AEA.successLine.pushDelta",
      "AEA.successLine.banes",
      "AEA.feature.roll-record-and-success-line.name",
      "AEA.feature.roll-record-and-success-line.hint",
    ]);
  });

  it("分区类名逐字是 aea-<feature-id>，且类名表每次返回新对象", () => {
    const a = pureSuccessLineClasses();
    expect(a).toEqual({
      section: "aea-roll-record-and-success-line",
      main: "aea-success-line-main",
      banes: "aea-success-line-banes",
    });
    expect(a.section).toBe(`aea-${pureSuccessLineId()}`);
    a.main = "tampered";
    expect(pureSuccessLineClasses().main).toBe("aea-success-line-main");
  });
});
```

- [ ] **Step 2: 跑它，看它失败**

Run: `npx vitest run test/roll-record-and-success-line.pure.test.mjs`

Expected: FAIL —— vitest 在收集阶段就报 `Error: Failed to load url ../scripts/features/roll-record-and-success-line.pure.mjs (resolved id: .../scripts/features/roll-record-and-success-line.pure.mjs). Does the file exist?`，3 个 describe、17 个用例一个都没执行。

- [ ] **Step 3: 写纯函数层**

新建 `scripts/features/roll-record-and-success-line.pure.mjs`。按契约 §6，纯函数文件与副作用文件 stem 逐字相同，且**只导出 `pure*` 函数**（常量表是文件私有的，通过访问器暴露）。这个文件**不 import 任何东西**，也不引用 `game`/`ui`/`CONFIG`/`document` —— 它必须能被 vitest 直接 import。

```js
/**
 * Pure layer for the `roll-record-and-success-line` feature.
 * No imports, no Foundry globals: vitest imports this file as-is.
 */

const FEATURE_ID = "roll-record-and-success-line";

// One home for every string that is written down in more than one place
// (template, renderer, tests, self-test). A rename here moves all of them.
const TEXT_KEYS = {
  none: "AEA.successLine.none",
  one: "AEA.successLine.one",
  many: "AEA.successLine.many",
  pushDelta: "AEA.successLine.pushDelta",
  banes: "AEA.successLine.banes",
};

// `section` is the private container this feature creates inside the shared
// `.aea-mount`. CONTRACT §4 K8 [v3.1] fixes its name at `aea-<feature-id>`:
// cards.render() replaces the innerHTML of whatever element it is handed, so
// two features sharing one mount must hand it two differently named children.
const CLASSES = {
  section: `aea-${FEATURE_ID}`,
  main: "aea-success-line-main",
  banes: "aea-success-line-banes",
};

// Radiation cards already print their own combined total at
// systems/alienrpg/module/helpers/YZEDiceRoller.mjs:289-331.
const SUPPRESSED_LABEL_KEYS = ["ALIENRPG.Radiation", "ALIENRPG.RadiationReduced"];

// Supply rolls count ones, not sixes; the system's whole success-total block is
// skipped for them at :287 and it prints "supply decreases" at :244-249 instead.
const SUPPRESSED_KINDS = ["supply"];

// Both tones are the system's own chat classes so the line matches the card.
// NOT `alienchatlightblue`: css/alienrpg.css:67-68 defines --alienchatlightblue
// and --alienchatlightgreen as the SAME hsl(120,97%,41%) green in 4.1.13, so it
// would look identical to a hit. `.alienchatblue` (css/alienrpg.css:666-668) is
// the real blue, and the system itself uses it for neutral notices at :245.
const TONE_HIT = "alienchatlightgreen";
const TONE_MISS = "alienchatblue";

const HIDDEN = { show: false, successes: 0, banes: 0, pushCount: 0 };

/** @returns {string} the feature id, verbatim, for every consumer */
export function pureSuccessLineId() {
  return FEATURE_ID;
}

/** @returns {string[]} every i18n key this feature needs at runtime */
export function pureSuccessLineKeys() {
  return [...Object.values(TEXT_KEYS), `AEA.feature.${FEATURE_ID}.name`, `AEA.feature.${FEATURE_ID}.hint`];
}

/** @returns {{section:string, main:string, banes:string}} a fresh copy each call */
export function pureSuccessLineClasses() {
  return { ...CLASSES };
}

/**
 * @param {object|null|undefined} record a v1 RollRecord
 * @returns {{show:boolean, successes:number, banes:number, pushCount:number}}
 */
export function pureSuccessLineModel(record) {
  if (!record || record.v !== 1) return { ...HIDDEN };
  if (SUPPRESSED_KINDS.includes(record.kind)) return { ...HIDDEN };
  if (record.labelKey && SUPPRESSED_LABEL_KEYS.includes(record.labelKey)) return { ...HIDDEN };

  const res = record.results || {};
  const successes = Number.isFinite(record.successes)
    ? record.successes
    : (Number(res.baseSixes) || 0) + (Number(res.stressSixes) || 0);
  // Only yellow (stress) ones are banes; black ones mean nothing.
  const banes = Number.isFinite(record.banes) ? record.banes : Number(res.stressOnes) || 0;

  return { show: true, successes, banes, pushCount: Number(record.push?.count) || 0 };
}

/**
 * Turn the model into the exact strings the Handlebars template prints.
 * The localizer is injected so this stays testable and Foundry-free.
 *
 * @param {{show:boolean, successes:number, banes:number, pushCount:number}|null} model
 * @param {(key:string, data?:object) => string} t
 * @returns {{tone:string, main:string, banes:string}|null} null = print nothing
 */
export function pureSuccessLineText(model, t) {
  if (!model || !model.show) return null;

  let main;
  if (model.pushCount > 0) main = t(TEXT_KEYS.pushDelta, { n: model.successes });
  else if (model.successes === 0) main = t(TEXT_KEYS.none);
  else if (model.successes === 1) main = t(TEXT_KEYS.one);
  else main = t(TEXT_KEYS.many, { n: model.successes });

  return {
    tone: model.successes > 0 ? TONE_HIT : TONE_MISS,
    main,
    banes: model.banes > 0 ? t(TEXT_KEYS.banes, { n: model.banes }) : "",
  };
}
```

- [ ] **Step 4: 跑它，看它通过**

Run: `npx vitest run test/roll-record-and-success-line.pure.test.mjs`
Expected: PASS —— 3 个 describe、17 个用例全绿。

- [ ] **Step 5: 提交纯函数层**

```bash
git add scripts/features/roll-record-and-success-line.pure.mjs test/roll-record-and-success-line.pure.test.mjs && git commit -m "$(cat <<'EOF'
feat(success-line): 合并成功数的纯模型与文案选择

把黑池六点与压力池六点合成一个数字；补给掷骰与两种辐射掷骰不显示
（系统在 YZEDiceRoller.mjs:287-331 已自行打印总数，:332-333 的 default 是
空的，普通卡上确实没有合计数）。辐射的判定走 RollRecord.labelKey 这个
i18n 键，绝不比对已本地化的 label 文本。推骰卡走「本次新增」文案，不与
系统 :340-370 的累计行冲突。

零成功的色调用 .alienchatblue 而不是 .alienchatlightblue：后者在
css/alienrpg.css:67-68 与 --alienchatlightgreen 是同一个绿，用它做区分
等于什么都没做。

id、i18n 键表、类名表都由纯层的访问器出，模板、渲染层、自检与测试共用
同一份，改名不会有一处静默漏改。分区类名逐字取 aea-<feature-id>，这是
契约 §4 K8 给共用挂点定下的隔离约定。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 加 i18n 键**

契约 §4 K5 规定特性的显示名与说明固定走 `AEA.feature.<id>.name` / `AEA.feature.<id>.hint`，`<id>` 逐字等于 `features.register({id})` 的 id（这里是带连字符的 `roll-record-and-success-line`）。模组自有键一律 `AEA.` 前缀、嵌套结构（Foundry 会把嵌套 JSON 自动拍平成点号键）。

在 `lang/en.json` 顶层的 `"AEA"` 对象里并入下列内容（若已有 `"feature"` 子对象则只添它的子键，不要覆盖兄弟键）：

```json
"successLine": {
  "none": "No successes",
  "one": "1 success",
  "many": "{n} successes",
  "pushDelta": "+{n} new successes from this push",
  "banes": "{n} stress dice rolled 1"
},
"feature": {
  "roll-record-and-success-line": {
    "name": "Combined success line",
    "hint": "Print one combined success line on every roll card, read from the roll record."
  }
}
```

在 `lang/cn.json` 顶层的 `"AEA"` 对象里并入：

```json
"successLine": {
  "none": "0 个成功",
  "one": "1 个成功",
  "many": "{n} 个成功",
  "pushDelta": "本次推骰新增 {n} 个成功",
  "banes": "{n} 颗压力骰掷出 1"
},
"feature": {
  "roll-record-and-success-line": {
    "name": "合并成功数行",
    "hint": "在每张掷骰卡上打印一行合并后的成功数，数据取自掷骰记录。"
  }
}
```

- [ ] **Step 7: 建模板**

新建 `templates/success-line.hbs`。文案已在 JS 里格式化完毕（`{n}` 需要 `game.i18n.format`，模板里的 `{{localize}}` 只能处理无参键），所以这里只放结构。`{{tone}}`／`{{main}}`／`{{banes}}` 就是 `pureSuccessLineText` 返回的三个字段；Handlebars 的 `{{ }}` 会做 HTML 转义，这正是我们要的（文案里不含标记）。

模板里**不写分区类名** —— 分区元素由渲染层建、`cards.render` 只往它的 `innerHTML` 里写，模板产出的是分区的内容。加粗与放大写成 inline style，因为我们用的两种色调类都只设颜色：`.alienchatblue`（css/alienrpg.css:666-668）与 `.alienchatlightgreen`（`:676-678`）里都没有 `font-weight`/`font-size`（有的是 `.alienchatred` `:657-665` 与 `.alienchatlightblue` `:670-674`），不写死就会和系统那两行大小不一。

```hbs
<hr />
<div class="aea-success-line-main {{tone}}" style="font-weight:bold;font-size:larger">{{main}}</div>
{{#if banes}}<div class="aea-success-line-banes alienchatred">{{banes}}</div>{{/if}}
```

- [ ] **Step 8: 写下会失败的接线测试**

新建 `test/roll-record-and-success-line.wiring.test.mjs`。它验五件靠肉眼最容易漏、又确实能脱离真实 Foundry 判定的事：id 三处一致、`install()` 真的把回调注册进了 `cards.onRender`（而不是留了个空壳）、两份语言文件都有本特性用到的每一个键、**早退分支绝不读 `mount`**（读了就会在每张非掷骰卡上留下空挂点）、以及**渲染目标是本特性自己的分区元素、重复渲染复用同一个分区**（写错就会覆盖同挂点上另一条特性的内容，或者在卡上叠出第二行）。

按契约 §0.3，Foundry 全局一律用共享桩 `installFoundryStub()`，**不得就地自造** `globalThis.game` 之类的东西。测试里对 `features.enabled` / `cards.render` / `diceBarrier.awaitDice` 三个**内核导出对象的方法**做临时替换是另一回事：它们不是 Foundry 全局，而且这三个方法的返回值形状分别属于 K5/K8/K6 三个内核任务 —— 本文件只关心「渲染回调走了哪条分支、把什么交给了 render」，替换掉它们才能不依赖别人的内部实现。每处替换都在 `afterAll` 里原样还原。

```js
import { describe, it, expect, beforeAll, beforeEach, afterAll } from "vitest";
import { readFileSync } from "node:fs";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { MID } from "../scripts/const.mjs";
import {
  pureSuccessLineClasses,
  pureSuccessLineId,
  pureSuccessLineKeys,
} from "../scripts/features/roll-record-and-success-line.pure.mjs";

// The shared stub must be on globalThis before the feature module (and the
// kernels it imports) are evaluated, hence top-level install + dynamic import.
installFoundryStub();
const { successLineFeature } = await import("../scripts/features/roll-record-and-success-line.mjs");
const { features } = await import("../scripts/kernel/features.mjs");
const { cards } = await import("../scripts/kernel/cards.mjs");
const { diceBarrier } = await import("../scripts/kernel/dice-barrier.mjs");

const ID = pureSuccessLineId();
const CLASSES = pureSuccessLineClasses();
const registered = [];
let handler = null;
let enabled = true;
const realEnabled = features.enabled;
const realOnRender = cards.onRender;

function readText(relative) {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}
function loadLang(file) {
  return JSON.parse(readText(`../lang/${file}`));
}
function hasKey(root, dotted) {
  return dotted.split(".").reduce((node, seg) => (node == null ? undefined : node[seg]), root) !== undefined;
}

/** A minimal v1 RollRecord — only the fields pureSuccessLineModel reads. */
function record(over = {}) {
  return {
    v: 1,
    kind: "skill",
    labelKey: null,
    successes: 2,
    banes: 0,
    results: { baseSixes: 2, baseOnes: 0, stressSixes: 0, stressOnes: 0 },
    push: { count: 0, pushable: true, parentRollId: null },
    ...over,
  };
}

/**
 * The argument object cards hands to a render handler, for the early-return
 * cases. `mount` is a lazy getter on the real thing too (CONTRACT §4 K8: the
 * div is only inserted into the DOM when a handler first reads it), so a getter
 * that records the read is a faithful stand-in — and reading it too early is
 * exactly the mistake we are hunting.
 */
function renderArgs(rec) {
  const args = { message: { id: "msg-1" }, element: {}, record: rec, mountRead: false };
  Object.defineProperty(args, "mount", {
    get() {
      args.mountRead = true;
      return {};
    },
  });
  return args;
}

/**
 * A DOM-free stand-in for the `.aea-mount` div. The renderer only ever calls
 * `querySelector`, `appendChild`, `ownerDocument.createElement` and reads
 * `isConnected` on it, so this covers its whole contact surface and lets the
 * happy path run under plain node without jsdom.
 */
function fakeMount() {
  const children = [];
  const mount = {
    isConnected: true,
    appendCount: 0,
    children,
    ownerDocument: {
      createElement: (tag) => ({ tagName: tag.toUpperCase(), className: "", innerHTML: "" }),
    },
    querySelector: (selector) => children.find((c) => c.className && selector.includes(c.className)) ?? null,
    appendChild(child) {
      mount.appendCount += 1;
      children.push(child);
      return child;
    },
  };
  return mount;
}

beforeAll(() => {
  successLineFeature.register();
  cards.onRender = (name, fn) => registered.push({ name, fn });
  successLineFeature.install();
  cards.onRender = realOnRender;
  handler = registered[0]?.fn ?? null;
  // features.enabled reads the SETTING_FEATURES payload, whose shape belongs to
  // kernel/features.mjs; this file only cares which branch the handler takes.
  features.enabled = (id) => (id === ID ? enabled : realEnabled.call(features, id));
});

afterAll(() => {
  features.enabled = realEnabled;
  uninstallFoundryStub();
});

describe("successLineFeature 的注册与接线", () => {
  it("特性 id 在纯层、特性对象、features 表三处逐字一致", () => {
    expect(ID).toBe("roll-record-and-success-line");
    expect(successLineFeature.id).toBe(ID);
    const def = features.all().find((d) => d.id === ID);
    expect(def).toBeDefined();
    expect(def.default).toBe("full");
    expect(def.gmOnly).toBe(false);
    expect(def.requires).toEqual([]);
  });

  it("install() 以特性 id 为名注册了一个渲染回调，且模块里没有任何自挂钩子", () => {
    expect(registered).toHaveLength(1);
    expect(registered[0].name).toBe(ID);
    expect(typeof registered[0].fn).toBe("function");
    expect(readText("../scripts/features/roll-record-and-success-line.mjs")).not.toContain("Hooks.on");
  });

  it.each(["en.json", "cn.json"])("%s 含有本特性用到的全部 i18n 键", (file) => {
    const lang = loadLang(file);
    expect(pureSuccessLineKeys().filter((k) => !hasKey(lang, k))).toEqual([]);
  });

  it("模板里的类名与纯层导出的一致（改名不会静默失效）", () => {
    const hbs = readText("../templates/success-line.hbs");
    expect(hbs).toContain(CLASSES.main);
    expect(hbs).toContain(CLASSES.banes);
  });
});

describe("渲染回调的早退分支不创建挂点", () => {
  it("非掷骰卡（record 为 null）直接返回", async () => {
    const args = renderArgs(null);
    await handler(args);
    expect(args.mountRead).toBe(false);
  });

  it("特性被关成 off 时直接返回", async () => {
    enabled = false;
    const args = renderArgs(record());
    try {
      await handler(args);
    } finally {
      enabled = true;
    }
    expect(args.mountRead).toBe(false);
  });

  it("补给卡（kind=supply）直接返回", async () => {
    const args = renderArgs(record({ kind: "supply" }));
    await handler(args);
    expect(args.mountRead).toBe(false);
  });
});

describe("渲染目标是自己的分区，不是共用挂点", () => {
  const painted = [];
  const realRender = cards.render;
  const realAwaitDice = diceBarrier.awaitDice;

  beforeAll(() => {
    cards.render = async (target, template, data) => {
      painted.push({ target, template, data });
    };
    diceBarrier.awaitDice = async () => {};
  });
  beforeEach(() => {
    painted.length = 0;
  });
  afterAll(() => {
    cards.render = realRender;
    diceBarrier.awaitDice = realAwaitDice;
  });

  it("在挂点下建一个 aea-<feature-id> 分区并渲染进它，绝不把挂点本身当目标", async () => {
    const mount = fakeMount();
    await handler({ message: { id: "msg-2" }, element: {}, record: record(), mount });
    expect(painted).toHaveLength(1);
    expect(mount.appendCount).toBe(1);
    expect(painted[0].target).not.toBe(mount);
    expect(painted[0].target.className).toBe(CLASSES.section);
    expect(painted[0].target.className).toBe(`aea-${ID}`);
    expect(painted[0].template).toBe(`modules/${MID}/templates/success-line.hbs`);
    // tone comes from the pure layer, not from i18n, so it is safe to assert
    // without depending on how the stub's i18n formats an unknown key.
    expect(painted[0].data.tone).toBe("alienchatlightgreen");
  });

  it("同一个挂点再渲染一次时复用已有分区，不追加第二个", async () => {
    const mount = fakeMount();
    const args = () => ({ message: { id: "msg-3" }, element: {}, record: record(), mount });
    await handler(args());
    await handler(args());
    expect(painted).toHaveLength(2);
    expect(mount.appendCount).toBe(1);
    expect(mount.children).toHaveLength(1);
    expect(painted[0].target).toBe(painted[1].target);
  });
});
```

- [ ] **Step 9: 跑它，看它失败**

Run: `npx vitest run test/roll-record-and-success-line.wiring.test.mjs`
Expected: FAIL —— 顶层 `await import` 解析不到文件，vitest 报 `Error: Failed to load url ../scripts/features/roll-record-and-success-line.mjs (resolved id: .../scripts/features/roll-record-and-success-line.mjs). Does the file exist?`，3 个 describe、10 个用例一个都没执行（纯层与模板、语言键此时已存在，所以缺的只有特性模块这一个文件）。

- [ ] **Step 10: 写特性模块**

新建 `scripts/features/roll-record-and-success-line.mjs`。这一层碰 Foundry。它**不自己挂任何钩子**：契约 §0.2 把 `renderChatMessageHTML` 判给 `kernel/cards.mjs` 独占，特性经 `cards.onRender` 被回调。

```js
import { MID } from "../const.mjs";
import { features } from "../kernel/features.mjs";
import { selftest } from "../kernel/selftest.mjs";
import { cards } from "../kernel/cards.mjs";
import { rollBus } from "../kernel/rollbus.mjs";
import { diceBarrier } from "../kernel/dice-barrier.mjs";
import {
  pureSuccessLineClasses,
  pureSuccessLineId,
  pureSuccessLineKeys,
  pureSuccessLineModel,
  pureSuccessLineText,
} from "./roll-record-and-success-line.pure.mjs";

const ID = pureSuccessLineId();
const CLASSES = pureSuccessLineClasses();
const TEMPLATE = `modules/${MID}/templates/success-line.hbs`;

/** Bridge Foundry's localizer into the pure layer's injected `t`. */
function t(key, data) {
  return data ? game.i18n.format(key, data) : game.i18n.localize(key);
}

/**
 * Get (or create) this feature's own container inside the shared mount.
 * Several features render into one `.aea-mount`, and cards.render() replaces
 * the innerHTML of whatever element it is handed (CONTRACT §4 K8 [v3.1]), so
 * each feature hands it a private `aea-<feature-id>` section instead of the
 * mount itself — otherwise whoever renders last wipes out the others, with
 * symptoms that depend on cards' onRender registration order.
 * `ownerDocument` rather than the `document` global: same element in Foundry,
 * and it keeps this function callable from a test without a DOM.
 */
function sectionOf(mount) {
  const existing = mount.querySelector(`:scope > .${CLASSES.section}`);
  if (existing) return existing;
  const section = mount.ownerDocument.createElement("div");
  section.className = CLASSES.section;
  mount.appendChild(section);
  return section;
}

/**
 * @param {{message: ChatMessage, element: HTMLElement, mount: HTMLElement,
 *          record: object|null}} args handed over by cards' render fan-out
 */
async function paint(args) {
  if (!features.enabled(ID)) return;

  // The count comes from the record only — never from the rendered text, never
  // from Roll#total. cards already resolved it via rollBus.recordOf(message);
  // it is null on every chat card that is not one of our rolls.
  if (!args.record) return;

  const data = pureSuccessLineText(pureSuccessLineModel(args.record), t);
  if (!data) return; // supply / radiation cards: the system prints its own total

  // Read `mount` only now: cards creates the div on first access, so touching it
  // any earlier would litter every non-roll card with an empty mount point. It
  // is never null (CONTRACT §4 K8 [v3.1]: the message root is the last-resort
  // insertion point), so there is nothing to null-check here.
  const mount = args.mount;

  // With Dice So Nice active the 3D dice are still in the air at render time;
  // printing the result now spoils the animation. The barrier resolves at once
  // when DsN is absent or this client is not a recipient of the roll.
  await diceBarrier.awaitDice(args.message);
  if (!mount.isConnected) return; // the chat log re-rendered while we waited

  await cards.render(sectionOf(mount), TEMPLATE, data);
}

/**
 * cards calls render handlers synchronously and only catches what they throw
 * synchronously (CONTRACT §4 K8); it does not await the returned promise. So
 * the async tail must swallow its own rejection, or a single bad card turns
 * into an unhandled rejection in every client's console. The promise is still
 * returned so tests can await one pass deterministically.
 */
function onRenderSuccessLine(args) {
  return paint(args).catch((err) => console.error(`${MID} | ${ID} | render failed`, err));
}

/**
 * Live-world assertion, runnable on demand from the self-test panel.
 * Checks the two things vitest cannot: that the i18n keys resolve in the
 * language the world actually runs in, and that every rendered roll card
 * carries exactly one success line whose number agrees with its own record.
 */
async function selftestSuccessLine() {
  const missing = pureSuccessLineKeys().filter((k) => !game.i18n.has(k));
  if (missing.length) return { ok: false, detail: `missing i18n keys: ${missing.join(", ")}` };

  const bad = [];
  let checked = 0;
  for (const message of game.messages) {
    const record = rollBus.recordOf(message);
    if (!record) continue;
    const el = document.querySelector(`[data-message-id="${message.id}"]`);
    if (!el) continue; // scrolled out of the rendered window
    checked += 1;
    const lines = el.querySelectorAll(`.${CLASSES.section} .${CLASSES.main}`);
    const model = pureSuccessLineModel(record);
    if (!model.show) {
      if (lines.length) bad.push(`${message.id}: suppressed card still shows a success line`);
      continue;
    }
    if (lines.length !== 1) {
      bad.push(`${message.id}: ${lines.length} success lines, expected exactly 1`);
      continue;
    }
    if (!lines[0].textContent.includes(String(model.successes))) {
      bad.push(`${message.id}: line does not carry ${model.successes}`);
    }
  }
  return bad.length
    ? { ok: false, detail: bad.join("; ") }
    : { ok: true, detail: `${checked} rendered roll cards agree with their records` };
}

export const successLineFeature = {
  id: ID,

  /** init phase: called by main.mjs's `for (const f of FEATURES) f.register()`. */
  register() {
    features.register({
      id: ID,
      default: "full",
      gmOnly: false,
      requires: [],
      // Display name and hint are resolved by features.mjs from
      // AEA.feature.<id>.name / .hint, so nothing user-facing is hardcoded here.
      hint: "",
    });
    selftest.register({
      id: `${ID}.cards-agree-with-records`,
      // register() runs at init, before i18nInit: game.i18n.localize() would
      // echo a raw key back at this point, so this label stays a fixed English
      // identifier. It survives either self-test runner policy — a runner that
      // localizes labels gets this string echoed back unchanged.
      label: "Success line matches every card's own roll record",
      run: selftestSuccessLine,
    });
  },

  /** ready phase: called by main.mjs's `for (const f of FEATURES) f.install()`. */
  install() {
    cards.onRender(ID, onRenderSuccessLine);
  },
};
```

- [ ] **Step 11: 跑接线测试，看它通过**

Run: `npx vitest run test/roll-record-and-success-line.wiring.test.mjs`
Expected: PASS —— 3 个 describe、10 个用例全绿（id 一致 1、install 注册回调 1、两份语言文件各 1、模板类名 1、早退分支 3、分区渲染 2）。

- [ ] **Step 12: 在 main.mjs 的两个锚点各插一行**

`scripts/main.mjs` 是全模组唯一挂生命周期钩子的地方。它由骨架任务一次写成，里面有十个 `/* AEA-ANCHOR: ... */` 注释锚点、一个八键的 `export const api` 字面量，以及两个显式数组 `FEATURES` 与 `REPAIRS`：`init` 阶段跑 `for (const f of FEATURES) f.register()`，`ready` 阶段跑 `for (const f of FEATURES) f.install()`。

**本任务只在 `imports` 与 `features` 这两个锚点各插一行，别的一个字不改。** 逐字声明三条：四个生命周期钩子（`init` / `i18nInit` / `diceSoNiceReady` / `ready`）**一行未动**；`successLineFeature.register()` 与 `.install()` **由上面那两个 for..of 循环统一调用，禁止在任何锚点处重复直调**（重复调用会让 `features.register` 二次登记同一个 id，并让 `cards.onRender` 注册两个同名回调，卡上叠出两行）；`export const api` 那个八键字面量**不增删也不替换**，它只属于内核模块的属主任务。

先定位（按锚点文本，不用行号）：

```bash
grep -n 'AEA-ANCHOR: imports' scripts/main.mjs && grep -n 'AEA-ANCHOR: features' scripts/main.mjs
```

Expected: 两条命令各恰好一行命中（形如 `2:/* AEA-ANCHOR: imports */` 与 `12:  /* AEA-ANCHOR: features */`）。若任一条命中零行或多行，停下来 —— 骨架被改过，不要凭猜测插入。

然后做两处插入：

1. 在**含 `/* AEA-ANCHOR: imports */` 的那一行之后**插入这一行（模组全部 import 集中在文件顶部这一段，不在别处另起 import 区）：

```js
import { successLineFeature } from "./features/roll-record-and-success-line.mjs";
```

2. 在**含 `/* AEA-ANCHOR: features */` 的那一行之后**插入这一行，作为 `FEATURES` 数组的一项（缩进两个空格，与锚点注释和数组里已有的项对齐）：

```js
  successLineFeature,
```

**不要**在 `main.mjs` 里加 `Hooks.on("renderChatMessageHTML", ...)`（那个钩子归 `kernel/cards.mjs` 独占），也不要往 `ready.cards` 等任何 ready 子锚点插东西 —— 那四个子锚点各有属主，本任务不是其中任何一个。

- [ ] **Step 13: 跑全套，确认接线没打破别的东西**

Run: `node --check scripts/main.mjs && npm test`

Expected: `node --check` 无输出（退出码 0，证明新加的 import 行与数组项没让 `main.mjs` 语法失效）；随后 vitest 打印全部测试文件通过，其中包含本任务新增的 `test/roll-record-and-success-line.pure.test.mjs`（17 例）与 `test/roll-record-and-success-line.wiring.test.mjs`（10 例），且既有的骨架、桩保真度与内核测试一条都没变红。

- [ ] **Step 14: MANUAL VERIFICATION —— 在 Foundry 里验**

前置：世界已启用 `alien-evolved-automation` 与 `lib-wrapper`，系统 `alienrpg` 4.1.13，且内核的 RollBus 已在往消息 flag 里写记录（否则 `record` 恒为 `null`，这一行永远不出现 —— 那是内核的问题，不是本特性的）。

1. 打开任意一个 **character** 类型角色卡，点一个技能（例如 RANGED COMBAT），在弹出的对话框里把压力骰留成 0 直接确认（**基础骰单掷卡**）。
   - **期望**：卡上除了原有的 `Black N Dice / Sixes: a` 一段，多出一条水平线加一行 **`a 个成功`**；`a` 为 0 时是淡紫蓝色的「0 个成功」（系统的 `.alienchatblue`，`css/alienrpg.css:666-668`），大于 0 时是绿色（`.alienchatlightgreen`，`:676-678`）；两种都加粗放大。卡上此时共两个数字，且 `a` 与我们这行相等。
2. 让角色带着压力值再掷一次同一技能（**带压力骰卡**）。
   - **期望**：卡上出现三个数字：黑池 `Sixes: a`、黄池 `Sixes: c`、模组这行 `a+c 个成功`。逐个核对 `a + c` 确实等于模组打印的数。
   - **期望**：黄池 `Ones: b` 且 `b > 0` 时，模组这行下面紧跟一行红色的「b 颗压力骰掷出 1」。
3. 若世界装了 **Dice So Nice**：再掷一次，盯住 3D 骰子。
   - **期望**：模组这行在**骰子停下之后**才出现，不会提前剧透。若它在骰子还在滚时就出现，说明 `diceBarrier.awaitDice` 没被等到 —— 回头检查 `paint` 是不是 async 且真的 `await` 了。这一条是 DsN 时序的**唯一**覆盖，vitest 里没有对应断言。
4. 按 F5 重载客户端，滚回第 2 步那张卡；然后在 F12 控制台执行
   `document.querySelectorAll('.aea-roll-record-and-success-line .aea-success-line-main').length`。
   - **期望**：成功数行仍在；上面那句返回的数字等于当前聊天栏里可见的掷骰卡张数（每张卡恰好一行，不叠加）。
   - **期望**：再执行 `document.querySelector('.aea-mount').children.length` 看挂点里的分区数 —— 若模组还有别的特性也往挂点里写东西，它们的兄弟分区（各自 `aea-<别的特性 id>`）都还在。谁把整个挂点当渲染目标，谁就会在这一步把对方的分区抹掉。
5. 在角色卡上点 **SUPPLY** 类掷骰（消耗品格子上的补给按钮，**补给卡**）。
   - **期望**：卡上**没有**新增的成功数行（补给掷骰不算成功），系统自己的「补给下降」照常。
6. 在角色卡上点辐射掷骰（Radiation）。
   - **期望**：卡上仍只有系统自己的「你受到 N 点伤害」，**没有**模组的第二行合并成功数。若这里出现了重复，说明 `RollRecord.labelKey` 没被填成 `ALIENRPG.Radiation` —— 病根在内核 i18nInit 阶段注入的那张反查表（译文歧义会让它把该文本映射成 `null`）。**记录下来反馈给内核 K1 的属主，绝不在本特性里改用文本匹配兜底。**
7. 在第 2 步那张技能卡上点 **Push**（**推骰卡**）。
   - **期望**：新的推骰卡上模组这行显示「本次推骰新增 N 个成功」，且系统自己那行「Following the push … total of …」（`YZEDiceRoller.mjs:340-370`）照常存在，两行数字含义不同但不打架。
8. 控制台执行 `game.modules.get("alien-evolved-automation").api.selftest.runAll()`。
   - **期望**：返回数组里 `id` 为 `roll-record-and-success-line.cards-agree-with-records` 的那条 `ok: true`，`detail` 形如 `"5 rendered roll cards agree with their records"`，条数与你刚掷出的卡数吻合。
9. 打开模组设置，把「合并成功数行」关成 `off`，重新掷一次骰。
   - **期望**：新卡上没有模组的成功数行，系统原有两行 "Sixes" 一切照旧；旧卡上已经画好的那行不受影响（它不重渲染就不重画）。
   - **期望**：新卡上也**没有**空的 `.aea-mount` 挂点残留 —— 控制台执行 `document.querySelectorAll('.aea-mount').length`，数字不应随着新掷的骰增长（关掉时早退发生在读 `mount` 之前，挂点是惰性创建的，压根不会被创建）。

- [ ] **Step 15: 提交特性层**

```bash
git add scripts/features/roll-record-and-success-line.mjs templates/success-line.hbs test/roll-record-and-success-line.wiring.test.mjs scripts/main.mjs lang/en.json lang/cn.json && git commit -m "$(cat <<'EOF'
feat(success-line): 在掷骰卡上渲染合并后的成功数行

数据只从 cards 回调带来的 RollRecord 读，不解析渲染文本、不读 Roll#total。
渲染经 cards.onRender 注册，本特性不挂任何钩子（renderChatMessageHTML 归
K8 独占）；写进挂点前先 await diceBarrier.awaitDice，避免 DsN 的 3D 骰还在
滚就把结果剧透出来。

挂点是惰性创建的，所以三条早退分支（特性关闭、非掷骰卡、补给与辐射卡）
一律在读 mount 之前返回，否则每张聊天卡上都会多出一个空挂点；接线测试
用一个会记录读取的 getter 把这三条都钉住了。

cards.render 只替换传进去那个元素的 innerHTML，而多条特性共用一个
.aea-mount，所以渲染目标是本特性自己的 aea-<feature-id> 分区，复用已有的、
没有才建；两例接线测试断住了「目标不是挂点」与「重复渲染不追加第二个
分区」。cards 只捕获同步异常、不 await 回调，所以异步部分自己 catch。

main.mjs 只在 imports 与 features 两个锚点各插一行：一行 import、一个
FEATURES 数组成员。register()/install() 由骨架的两个 for..of 调用，不重复
直调；四个生命周期钩子与 api 字面量一个字未动。

同一批断言登记进 selftest，可在世界里按需重跑。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```
