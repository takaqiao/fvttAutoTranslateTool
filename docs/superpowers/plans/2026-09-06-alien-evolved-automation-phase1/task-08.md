> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 8 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 8: K6 DiceBarrier —— 骰子落地再结算

**背景（实现者必读，假设你完全不了解 Foundry 与 Alien RPG）**

- Foundry VTT 是跑在浏览器里的桌游平台。**Hook** 是它的全局事件总线：`Hooks.on(name, fn)` 订阅、`Hooks.once(name, fn)` 订阅一次、`Hooks.callAll(name, ...args)` 广播。没人 `callAll` 的事件，`on` 上去的回调永远不会被调用 —— 这是本任务全部设计的出发点：**await 一个不会被广播的事件 = 永久挂起**。
- **Dice So Nice!（DsN）** 是第三方模组，把掷骰渲染成 3D 骰子在屏幕上滚动。动画结束时它广播 `diceSoNiceRollComplete`。「栅栏」就是等骰子落地再让后续自动化继续，免得玩家先看到「成功 2 次」、几秒后才看到骰子停下。
- 本模组叫 `alien-evolved-automation`（下文的 `MID`），装在 `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/`。被自动化的游戏系统叫 `alienrpg`（4.1.13），装在 `C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/`。
- **钩子归属（契约 §0.2）**：`diceSoNiceReady` 是**生命周期钩子**，只准在 `main.mjs` 里挂，它在里面调 `diceBarrier.init()`；`diceSoNiceRollComplete` 是**领域钩子**，**归本文件所有** —— 在 `dice-barrier.mjs` 自己的 `init()` 里挂，全模组只此一处、只一个监听器。不要挂到 `main.mjs`。
- **本任务同时负责三处 `main.mjs` 接线**（契约 §5「api 的装配规则」：每个内核模块的属主任务做三件事、一件不多 —— 在 `/* AEA-ANCHOR: imports */` 后加自己的 import、把 `api` 里自己那一个 `null` 换成该对象、插自己的调用）。上一版把这三件事都当成了「别人已经做好」，结果：`main.mjs` 从未 import 本模块 → 启动即 `ReferenceError`；`api.diceBarrier` 恒为 `null` → 手工验证第一行就炸；`diceBarrier.init()` 无人认领 → 栅栏静默退化成空操作（代码全在、测试全绿、线上一次都不等）。Step 22-29 就是堵这三个洞。

**九条已亲手核对的源码事实（都已重新打开原文件确认，不要重新猜）**

1. `systems/alienrpg/module/alienrpg.mjs:381` 是 `Hooks.on("diceSoNiceRollComplete", (chatMessageID) => {});` —— **函数体是空的**。系统占了坑但什么都没做，指望不上它。
2. 同文件 `:383` 是 `Hooks.once("diceSoNiceReady", (dice3d) => { dice3d.addColorset({...}) ... })`。系统自己也挂了一个 `diceSoNiceReady`。Foundry 允许同一钩子多个监听器，我们在 `main.mjs` 挂的那个是**第二个**，两个都会跑，互不影响。
3. `systems/alienrpg/module/helpers/YZEDiceRoller.mjs:408-417`（逐字，制表符已还原为空格）：
   ```js
   if (["gmroll", "blindroll"].includes(chatData.rollMode)) {
     chatData.whisper = ChatMessage.getWhisperRecipients("GM")   // :409
   } else if (chatData.rollMode === "selfroll") {
     chatData.whisper = [game.user]                              // :411  ← User 文档，不是 id 字符串
   } else if (blind) {
     chatData.whisper = ChatMessage.getWhisperRecipients("GM")   // :413
     chatData.blind = true                                       // :414
   }
   await ChatMessage.create(chatData)                            // :416
   return                                                        // :417
   ```
   密语／盲骰消息在**非接收者**的客户端上根本不播动画，那个客户端**永远收不到**完成事件；在那里 await 它就是永久挂起。注意 `whisper` 有两种形态：`:411` 塞的是 `[game.user]`（User 文档），`getWhisperRecipients()` 返回的也是 User 文档数组；消息落库后 `message.whisper` 通常已被规范化成 id 字符串数组。**两种形态都要接**，不要假设其一。
4. 本机装的 DsN 是 **6.2.9**（`modules/dice-so-nice/module.json` 的 `version` 字段，已重新 grep 确认）。它的 `compatibility.verified` 是 `"14.365"`，那是 **Foundry** 的版本号，不是 DsN 的。
5. DsN 在自己的 `Hooks.on("createChatMessage", ...)` 监听器里**同步**给消息文档打 `message._dice3danimating = true`，动画跑完时删掉它并 `Hooks.callAll("diceSoNiceRollComplete", message.id, companionRelease)`。所以**消息上没有这个标记 = 本客户端这次不播动画**。这是避免「每张卡白等 4 秒」的关键闸门，也是 `awaitDice` 必须在**渲染阶段**调用的原因：`createChatMessage` 阶段标记还没打上。
6. `Dice3D#isEnabled()` 的实现是：`return "none" !== Dice3D.CONFIG().visibility && (没在战斗中 || 没开 disabledDuringCombat)`。
7. `Dice3D.CONFIG(user = game.user)` 把 `user.getFlag("dice-so-nice", "settings")` 合并到 `Dice3D.DEFAULT_OPTIONS` 上；`DEFAULT_OPTIONS` 的 `visibility` 默认值是 **`"all"`**。这个字段只有三个取值：`"all"` | `"mine"` | `"none"`。
8. DsN 的 `showForRoll(roll, user = game.user, ...)` 里有两条**打完 `_dice3danimating` 标记之后**才生效的抑制分支，都直接 `return Promise.resolve(false)`，**都不广播完成事件**：
   - `hideNpcRolls` 打开且 `ChatMessage.getSpeakerActor(speaker)` 存在而 `!hasPlayerOwner`；
   - `Dice3D.CONFIG().visibility === "mine"` 且掷骰的 user 不是 `game.user`。

   所以这两种情况下标记在、事件永远不来 —— 硬超时必须保留，同时我们主动复刻这两条判据，否则每张 NPC 卡都要白卡满一个超时。
9. **DsN 6.2.9 自带 `game.dice3d.waitFor3DAnimationByMessageID(id)`**，做的事和我们很像。我们**不用**它，理由有三条，都写在这里免得后来者以为是重复造轮子：(a) 它**没有超时**；(b) 它的闸只有三道，缺 `hideNpcRolls`、`"mine"`、密语／盲骰这三种「标记在但事件不来」的情况，正好撞上事实 8；(c) 它的内部 `buildHook` 在 id 不匹配时用 `Hooks.once` 无限自我重挂，事件一旦丢失就永久泄漏一个监听器。K6 是它的严格超集，且在 DsN 缺席时也能安全返回。

**DsN 的生命周期时机（影响接线位置，已核实）**

`game.dice3d = new Dice3D` + `game.dice3d.init()` 发生在 DsN 自己的 `Hooks.once("ready", ...)` 里、且在一次 await 过的设置迁移之后；`diceSoNiceReady` 是从 `Dice3D#init()` 内部广播的。因此：

- `diceSoNiceReady` 触发时刻 **晚于** Foundry 的 `ready`，也就是晚于本模组 ready 段里的其它安装动作。这没问题：栅栏只在渲染时被调用，而那时它早已 armed。
- 即使用户把 3D 骰子可见度设成 `"none"`，DsN 仍然会广播 `diceSoNiceReady`（源码里 `"none"` 分支照样 `Hooks.call("diceSoNiceReady", this)`），所以我们的 `init()` 照常跑。
- 在 `/stream` 视图或开了 `core.noCanvas` 时，DsN **根本不构造** `Dice3D`：`game.dice3d` 是 `undefined`，`diceSoNiceReady` 永不触发。此时 `init()` 不跑、`armed` 为假、`awaitDice` 立即放行 —— 正确行为。

**四种必须正确处理的情况（设计文档 §3.2 K6）**

| # | 情况 | 正确行为 |
|---|---|---|
| 1 | DsN 没装／没就绪，或本用户关了 3D 骰 | 立即 resolve，绝不 await 一个永不触发的钩子 |
| 2 | 消息是密语／盲骰且本客户端不是接收者 | 立即 resolve |
| 3 | 开了 `immediatelyDisplayChatMessages`；或 `visibility === "mine"` 而我不是掷骰人；或 `hideNpcRolls` 且这是 NPC 掷骰；或消息上没有 `_dice3danimating` | 立即 resolve |
| 4 | 以上都不成立 | 等钩子，但**必须有硬超时** |

情况 1 与 3 全部归并成布尔量 `dsnActive`（效果层算），情况 2 归并成 `isRecipient`（纯函数算）。契约 §4 K6 把 `pureBarrierPlan` 的入参定死成三个，这个归并是唯一能在不改签名的前提下覆盖情况 3 的做法。

**Files:**
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/kernel/dice-barrier.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/dice-barrier.test.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/dice-barrier-wiring.test.mjs`
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/main.mjs`（**三处，一处不多**：`/* AEA-ANCHOR: imports */` 后加一行 import；`api` 字面量里把 `diceBarrier: null` 换成 `diceBarrier`；`/* AEA-ANCHOR: diceSoNiceReady */` 后加一行 `diceBarrier.init();`）
- Modify: `.../lang/en.json`、`.../lang/cn.json`（加设置项与自检条目的 i18n 键）

**Interfaces:**

- **Consumes:**
  - `scripts/const.mjs` 的 `MID`（值为 `"alien-evolved-automation"`）与 `SETTING_DICE_TIMEOUT`（值为 `"diceTimeoutMs"`，语义：默认 4000ms）。
  - `scripts/kernel/selftest.mjs` 的 `selftest.register({id, label, run})`；`run()` 返回 `{ok:boolean, detail:string}`，可以是 async。`selftest.runAll()` 返回 `[{id, label, ok, detail}]`。
  - `test/stubs/foundry.mjs` 的 `installFoundryStub(options)` / `uninstallFoundryStub()`（契约 §0.3，**唯一允许的 Foundry 替身**）。本任务依赖它的四条已写进契约的行为：① 幂等，可重复 `installFoundryStub()`；② `game.settings.register/get/set` 由 `ctx.settings` 真后端支撑，`register` 把 `default` 播种进去，`get` 读**未注册**键**抛错**（本任务的 `readSetting()` 正是靠这条失败模式来兜底）；③ `Hooks.callAll(name, ...args)` 真正派发给 `Hooks.on` 注册的回调；④ 装好 `globalThis.game`（含对象型 `game.user`、`game.modules`）与 `globalThis.ChatMessage`。这四条由桩的属主任务用 `test/stub-fidelity.test.mjs` 逐条守卫，**本任务只读不改**：不得在自己的测试里造 `globalThis.game`，也不得用私有 `Map` 顶替 `game.settings`。
  - `scripts/main.mjs` 里由骨架任务逐字写好的三个锚点注释 `/* AEA-ANCHOR: imports */`、`/* AEA-ANCHOR: diceSoNiceReady */`，以及那个**八个键、值全为 `null`** 的 `api` 字面量（`{features, patches, resolver, registry, rollBus, diceBarrier, cards, selftest}`）。

- **Produces:**
  - `export function pureBarrierPlan({dsnActive, isRecipient, timeoutMs}) -> {wait:boolean, timeoutMs:number}`
  - `export function pureIsRecipient({whisper, blind, userId, isGM}) -> boolean`
  - `export const diceBarrier = { init(), awaitDice(message) /* -> Promise<void> */ }` —— 导出面**恰好两个成员**，见下方「关于超时设置注册时机」。
  - 客户端设置 `alien-evolved-automation.diceTimeoutMs`（`scope: "client"`，默认 4000，由 `init()` 注册）。
  - 自检条目 `k6.barrier-armed`、`k6.barrier-nonblocking`（`label` 存的是 **i18n 键**，不是已本地化文本）。
  - `main.mjs`：多一行 `import { diceBarrier } from "./kernel/dice-barrier.mjs";`；`api.diceBarrier` 从 `null` 变成真对象；`diceSoNiceReady` 段多出一行 `diceBarrier.init();`。

- **关于超时设置注册时机（终检对这一条有异议，这里给出结论、理由与未收口的部分）**：终检的意见是「§4 K6 应追加第三个成员 `registerSettings()`，并在 `main.mjs` 的 init 段加一行调用，否则没装 DsN 的世界看不到这个设置项」。**本任务不这么做**，因为契约 v3.1 §4 的段首写着「各内核模块的导出（签名不得改动）」，K6 那一行逐字是 `export const diceBarrier = { init(), awaitDice(message) };`，§5 的 init 段清单里也没有 `diceBarrier.registerSettings()` 这一行，而契约开头明写「任何任务若需要一个此处未定义的符号，说明任务划分错了 —— 在你的产出里明确指出，不要自行发明」，§5 又把内核任务的 `main.mjs` 改动数量钉死成「三件事、一件不多」。这条修法要求的是**改契约**，而契约「不得改动」。因此本任务按契约字面执行，并把它作为待裁决项上报。
  实质影响仅此：没装 DsN 的世界里，`game.dice3d` 为 `undefined` → `dsnWillAnimateHere()` 恒假 → `plan.wait` 恒假 → 这个超时值**没有任何消费者**，此时它不出现在设置面板里不会造成任何行为差异；装上 DsN 并重载后，`diceSoNiceReady` 触发 → `init()` → 设置项出现，默认 4000 本来就是安全值。Step 30 的第 6 步会亲眼确认「装了 DsN 时它确实出现在 Configure Settings 的本客户端分区」。另外，「Foundry 惯例上设置只在 init 注册」这条不构成阻塞：设置面板是在被打开时才枚举已注册项的，`ready` 之后注册照样显示得出来；而未注册的键被 `game.settings.get` 读到会抛异常这一点，由 `readSetting()` 的 try/catch 兜住，并由 Step 11 的第一条用例真实覆盖。

- **调用约定（写给下游特性，这是 K6 存在的全部意义）**：
  - `awaitDice(message)` 必须在**渲染阶段**调用 —— 契约 §4 K8 规定特性拿到聊天卡挂点的唯一通路是 `cards.onRender(name, handler)`，`handler({message, element, mount, record})`。只有到这一步，DsN 的 `_dice3danimating` 标记才一定已经打好（事实 5）；在 `createChatMessage` 阶段调用会因为标记还没打上而立即放行，等于没等。
  - **任何往挂点里写「判定结果」的 handler，都必须先 `await diceBarrier.awaitDice(message)`**，否则开着 DsN 时成功数会先于 3D 骰子出现。只画按钮、不画判定结果的 handler 不用等。
  - 契约 §4 K8 规定 cards 的 handler 是**同步调用**、异常由 cards 逐个捕获。这意味着 handler 一旦 `await`，cards 就**不再**为它兜底了。正确写法是 handler 同步返回、自己起一段异步续程并自己 catch；另外契约 §4 K8 规定每条特性必须先在 `mount` 下建自己的 `aea-<feature-id>` 子元素再对**那个子元素**调 `render()`（`render` 只替换目标元素自身的 innerHTML 并返回 Promise，直接对 `mount` 调会清掉别的特性的内容）：
    ```js
    cards.onRender("my-feature", ({ message, mount }) => {
      void (async () => {
        try {
          await diceBarrier.awaitDice(message);
          if (!mount.isConnected) return;      // 等待期间聊天栏可能已经重渲染
          let slot = mount.querySelector(".aea-my-feature");
          if (!slot) { slot = document.createElement("div"); slot.className = "aea-my-feature"; mount.appendChild(slot); }
          await cards.render(slot, "modules/alien-evolved-automation/templates/xxx.hbs", data);
        } catch (err) { console.error("alien-evolved-automation | my-feature", err); }
      })();
    });
    ```
  - 注意 `mount` 是 cards **惰性创建**的（第一次访问才插进 DOM），所以只在确实要写内容时才碰它，否则会在不需要的卡上留下空挂点。

- **覆盖面对照表**（哪条断言在哪里跑；少一行就说明有断言无声地死了）：

  | 断言的内容 | 跑在哪里 |
  |---|---|
  | `wait` 真值表、超时钳制与缺省、接收者判定 | vitest 真单测，无桩（Step 1-10） |
  | `armed` 闸、九道放行闸、事件释放、并发等待者、超时读设置 | vitest + `installFoundryStub()`（Step 11-15） |
  | 两条自检条目被登记、`init()` 幂等、`initCalls` 计数 | vitest + `installFoundryStub()`（Step 17-21） |
  | `main.mjs` 三处接线（import／api 槽／`diceSoNiceReady` 调用） | `test/dice-barrier-wiring.test.mjs` 读源码断言（Step 22-29） |
  | 真 DsN 真的带 message id 广播完成事件、等待时长肉眼对得上骰子停下 | selftest `k6.barrier-armed` + MANUAL 第 2 步 |
  | 不播动画的消息 250ms 内放行 | selftest `k6.barrier-nonblocking` + MANUAL 第 3/4/5 步 |
  | 设置项在 Configure Settings 里可见、是 client 作用域、改了生效 | MANUAL 第 6 步 |

---

- [ ] **Step 1: Write the failing test（`pureBarrierPlan`）**

新建 `test/dice-barrier.test.mjs`：

```js
import { describe, it, expect } from "vitest";
import { pureBarrierPlan } from "../scripts/kernel/dice-barrier.mjs";

describe("pureBarrierPlan", () => {
  it("waits only when DsN will animate here AND this client is a recipient", () => {
    expect(pureBarrierPlan({ dsnActive: true, isRecipient: true, timeoutMs: 4000 }))
      .toEqual({ wait: true, timeoutMs: 4000 });
    for (const [dsnActive, isRecipient] of [[false, true], [true, false], [false, false]]) {
      expect(pureBarrierPlan({ dsnActive, isRecipient, timeoutMs: 4000 }))
        .toEqual({ wait: false, timeoutMs: 0 });
    }
  });

  it("falls back to 4000ms for a missing, zero, negative or NaN timeout", () => {
    for (const bad of [undefined, null, 0, -1, Number.NaN, "abc"]) {
      expect(pureBarrierPlan({ dsnActive: true, isRecipient: true, timeoutMs: bad }))
        .toEqual({ wait: true, timeoutMs: 4000 });
    }
  });

  it("clamps into [250, 30000] and rounds", () => {
    const plan = (ms) => pureBarrierPlan({ dsnActive: true, isRecipient: true, timeoutMs: ms }).timeoutMs;
    expect(plan(10)).toBe(250);
    expect(plan(999999)).toBe(30000);
    expect(plan(1234.6)).toBe(1235);
  });

  it("tolerates being called with no argument at all", () => {
    expect(pureBarrierPlan()).toEqual({ wait: false, timeoutMs: 0 });
  });

  it("never touches a Foundry global", () => {
    const saved = { game: globalThis.game, Hooks: globalThis.Hooks, foundry: globalThis.foundry };
    delete globalThis.game; delete globalThis.Hooks; delete globalThis.foundry;
    try {
      expect(pureBarrierPlan({ dsnActive: true, isRecipient: true, timeoutMs: 4000 }))
        .toEqual({ wait: true, timeoutMs: 4000 });
    } finally { Object.assign(globalThis, saved); }
  });
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `npx vitest run test/dice-barrier.test.mjs -t "pureBarrierPlan"`
Expected: FAIL —— `Error: Failed to load url ../scripts/kernel/dice-barrier.mjs`（文件还不存在，5 条用例全部因加载失败报错）。

- [ ] **Step 3: 只写纯函数 `pureBarrierPlan`**

新建 `scripts/kernel/dice-barrier.mjs`（此刻**不写任何 import**，纯函数区块因此天然满足契约 §0.1 的分层铁律）：

```js
/** K6 · DiceBarrier —— 等 3D 骰子落地再结算。pure* 区块不得引用任何 Foundry 全局。 */

const MIN_TIMEOUT_MS = 250;
const MAX_TIMEOUT_MS = 30000;
const FALLBACK_TIMEOUT_MS = 4000;

/**
 * dsnActive   —— DsN 这次真的会在本客户端播动画（效果层算，见 dsnWillAnimateHere）。
 * isRecipient —— 本客户端是这条消息的接收者。
 * 两者同时为真才等；否则动画不会播，等一个永不触发的钩子会永久挂起。
 */
export function pureBarrierPlan(input = {}) {
  const { dsnActive, isRecipient, timeoutMs } = input ?? {};
  if (!dsnActive || !isRecipient) return { wait: false, timeoutMs: 0 };
  const raw = Number(timeoutMs);
  if (!Number.isFinite(raw) || raw <= 0) return { wait: true, timeoutMs: FALLBACK_TIMEOUT_MS };
  return { wait: true, timeoutMs: Math.min(MAX_TIMEOUT_MS, Math.max(MIN_TIMEOUT_MS, Math.round(raw))) };
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `npx vitest run test/dice-barrier.test.mjs -t "pureBarrierPlan"`
Expected: PASS —— 5 passed。

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/dice-barrier.mjs test/dice-barrier.test.mjs \
 && git commit -m "feat(kernel): K6 骰子栅栏的纯决策函数

pureBarrierPlan 用 dsnActive × isRecipient 的真值表决定等不等：
只有 DsN 确实会在本客户端播动画、且本客户端是这条消息的接收者时才等。
超时钳到 [250, 30000]，缺省 4000ms。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 6: Write the failing test（`pureIsRecipient`）**

追加到 `test/dice-barrier.test.mjs` 末尾：

```js
import { pureIsRecipient } from "../scripts/kernel/dice-barrier.mjs";

describe("pureIsRecipient", () => {
  const me = { userId: "u1", isGM: false };

  it("treats a public roll (empty or missing whisper) as visible to everyone", () => {
    expect(pureIsRecipient({ whisper: [], ...me })).toBe(true);
    expect(pureIsRecipient({ ...me })).toBe(true);
  });

  it("honours a whisper list of plain ids (the shape a persisted message carries)", () => {
    expect(pureIsRecipient({ whisper: ["gm1"], ...me })).toBe(false);
    expect(pureIsRecipient({ whisper: ["gm1"], userId: "gm1", isGM: true })).toBe(true);
  });

  it("accepts User documents in whisper too (YZEDiceRoller.mjs:411 passes [game.user])", () => {
    expect(pureIsRecipient({ whisper: [{ id: "u1" }], ...me })).toBe(true);
    expect(pureIsRecipient({ whisper: [{ id: "u2" }], ...me })).toBe(false);
  });

  it("locks a blind roll to GMs (YZEDiceRoller.mjs:413-414)", () => {
    expect(pureIsRecipient({ whisper: ["gm1"], blind: true, ...me })).toBe(false);
    expect(pureIsRecipient({ whisper: ["gm1"], blind: true, userId: "gm1", isGM: true })).toBe(true);
  });

  it("returns false when there is no current user id", () => {
    expect(pureIsRecipient({ whisper: [], userId: null, isGM: false })).toBe(false);
    expect(pureIsRecipient()).toBe(false);
  });

  it("never touches a Foundry global", () => {
    const saved = { game: globalThis.game, Hooks: globalThis.Hooks };
    delete globalThis.game; delete globalThis.Hooks;
    try { expect(pureIsRecipient({ whisper: [], ...me })).toBe(true); }
    finally { Object.assign(globalThis, saved); }
  });
});
```

- [ ] **Step 7: Run it and watch it fail**

Run: `npx vitest run test/dice-barrier.test.mjs -t "pureIsRecipient"`
Expected: FAIL —— `SyntaxError: The requested module '../scripts/kernel/dice-barrier.mjs' does not provide an export named 'pureIsRecipient'`（整个文件加载失败，两个 describe 都报错）。

- [ ] **Step 8: 实现 `pureIsRecipient`**

追加到 `scripts/kernel/dice-barrier.mjs` 的纯函数区块末尾：

```js
/**
 * 本客户端是不是这条消息的接收者。数据来源是 YZEDiceRoller.mjs:408-415 那四条
 * 设置 whisper / blind 的路径；:411 塞的是 [game.user]（User 文档），
 * getWhisperRecipients() 返回的也是 User 文档，而落库后的 message.whisper
 * 通常已被规范化成 id 字符串数组——两种形态都要接。
 */
export function pureIsRecipient(input = {}) {
  const { whisper, blind, userId, isGM } = input ?? {};
  if (!userId) return false;
  if (blind === true && !isGM) return false;
  const ids = Array.from(whisper ?? [], (w) => (typeof w === "string" ? w : w?.id)).filter(Boolean);
  if (!ids.length) return true;   // 不是密语，全场可见
  return ids.includes(userId);
}
```

- [ ] **Step 9: Run it and watch it pass**

Run: `npx vitest run test/dice-barrier.test.mjs`
Expected: PASS —— 11 passed（5 + 6）。

- [ ] **Step 10: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/dice-barrier.mjs test/dice-barrier.test.mjs \
 && git commit -m "feat(kernel): K6 接收者判定提成纯函数

契约 §4 K6 把 pureIsRecipient({whisper, blind, userId, isGM}) 写进导出面，
密语/盲骰这段分支因此能进真单测，而不是藏在模块私有函数里只能间接测。
whisper 同时接 id 字符串与 User 文档——YZEDiceRoller.mjs:411 传的是后者，
落库后的消息给的是前者。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 11: Write the failing test（副作用层，全部跑在共享桩上）**

追加到 `test/dice-barrier.test.mjs` 末尾。四点手法说明：

1. 契约 §0.3 规定 `test/stubs/foundry.mjs` 是**唯一**允许的 Foundry 替身，它的行为（settings 真后端且未注册键会抛、`Hooks.callAll` 真派发、幂等安装）由桩的属主任务在 `test/stub-fidelity.test.mjs` 里逐条守卫。本文件**只用不改**：不造 `globalThis.game`，不用私有 `Map` 顶替 `game.settings`。若下面这批用例集体红且报错指向 `game.settings` / `Hooks`，先去看 `test/stub-fidelity.test.mjs` 是不是也红了 —— 那是桩的问题，不是本模块的问题。
2. `game.dice3d`（第三方模组 DsN 的对象）、`game.user` 的具体字段值、`ChatMessage.getSpeakerActor` 的返回值 —— 这三样是**往桩上喂夹具数据**，不是另造替身，是正当的。
3. 触发完成事件一律走 `Hooks.callAll`，不去翻桩的内部记录形状。
4. `dice-barrier.mjs` 有 `armed` / `waiters` 这样的模块级状态，必须 `vi.resetModules()` + 动态 `import()`，每条用例才能拿到干净实例。

```js
import { beforeEach, afterEach, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { MID, SETTING_DICE_TIMEOUT } from "../scripts/const.mjs";

const DSN = "dice-so-nice";

describe("diceBarrier (effect layer, on installFoundryStub)", () => {
  let mod;

  const fire = (id) => globalThis.Hooks.callAll("diceSoNiceRollComplete", id);

  /** 让 DsN 看起来「会在本客户端播动画」，并返回一条对应的假消息。 */
  async function enableDsn({ animating = true, immediate = false, visibility = "all",
                             hideNpcRolls = false, enabled = true, author = "u1", npc = false } = {}) {
    globalThis.game.dice3d = { isEnabled: () => enabled };
    await globalThis.game.settings.set(DSN, "immediatelyDisplayChatMessages", immediate);
    await globalThis.game.settings.set(DSN, "hideNpcRolls", hideNpcRolls);
    globalThis.game.user.getFlag = (scope, key) =>
      (scope === DSN && key === "settings" ? { visibility } : undefined);
    globalThis.ChatMessage.getSpeakerActor = () => ({ hasPlayerOwner: !npc });
    const msg = { id: "m1", whisper: [], author: { id: author }, speaker: {} };
    if (animating) msg._dice3danimating = true;   // DsN 在 createChatMessage 里同步打的标记
    return msg;
  }

  beforeEach(async () => {
    installFoundryStub();
    Object.assign(globalThis.game.user, { id: "u1", isGM: false, getFlag: () => undefined });
    globalThis.game.settings.register(DSN, "immediatelyDisplayChatMessages",
      { scope: "world", config: false, type: Boolean, default: false });
    globalThis.game.settings.register(DSN, "hideNpcRolls",
      { scope: "world", config: false, type: Boolean, default: false });
    globalThis.game.dice3d = undefined;
    vi.resetModules();
    mod = await import("../scripts/kernel/dice-barrier.mjs");
  });

  afterEach(() => { vi.useRealTimers(); uninstallFoundryStub(); });

  it("does not wait until init() has armed the listener", async () => {
    const msg = await enableDsn();    // DsN 全开，唯一缺的就是 init()
    vi.useFakeTimers();               // 若真去等，假定时器不前进 => 这里会挂到用例超时
    await mod.diceBarrier.awaitDice(msg);
    // 顺带覆盖 readSetting 的 try/catch：此刻 diceTimeoutMs 还没注册，
    // 桩与真 Foundry 的 game.settings.get 对未注册键都会抛异常。
  });

  it("waits once armed, and resolves when the event carries this message id", async () => {
    mod.diceBarrier.init();
    const msg = await enableDsn();
    let settled = false;
    const p = mod.diceBarrier.awaitDice(msg).then(() => { settled = true; });
    await Promise.resolve();
    expect(settled).toBe(false);
    fire("m1");
    await p;
    expect(settled).toBe(true);
  });

  it("registers the client timeout setting with a 4000ms default", () => {
    mod.diceBarrier.init();
    expect(globalThis.game.settings.get(MID, SETTING_DICE_TIMEOUT)).toBe(4000);
  });

  it("releases two concurrent waiters on the same message", async () => {
    mod.diceBarrier.init();
    const msg = await enableDsn();
    const flags = [];
    const ps = [0, 1].map((i) => mod.diceBarrier.awaitDice(msg).then(() => flags.push(i)));
    await Promise.resolve();
    fire("m1");
    await Promise.all(ps);
    expect(flags.sort()).toEqual([0, 1]);
  });

  it("ignores a completion event for another message and gives up at the configured timeout", async () => {
    mod.diceBarrier.init();
    const msg = await enableDsn();
    await globalThis.game.settings.set(MID, SETTING_DICE_TIMEOUT, 1500);
    vi.useFakeTimers();
    let settled = false;
    const p = mod.diceBarrier.awaitDice(msg).then(() => { settled = true; });
    fire("SOME-OTHER-MESSAGE");
    await vi.advanceTimersByTimeAsync(1499);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await p;
    expect(settled).toBe(true);
  });

  it("gives up after the registered 4000ms default when the event is lost", async () => {
    mod.diceBarrier.init();
    const msg = await enableDsn();
    vi.useFakeTimers();
    let settled = false;
    const p = mod.diceBarrier.awaitDice(msg).then(() => { settled = true; });
    await vi.advanceTimersByTimeAsync(3999);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await p;
    expect(settled).toBe(true);
  });

  // 每一行都是一条「绝不能等」的现实路径；等错了就是每张卡白卡满一个超时。
  const noWait = [
    ["no message id", async () => ({ ...(await enableDsn()), id: undefined })],
    ["DsN not installed", async () => { const m = await enableDsn(); globalThis.game.dice3d = undefined; return m; }],
    ["DsN isEnabled() false (visibility none / disabledDuringCombat)", async () => enableDsn({ enabled: false })],
    ["immediatelyDisplayChatMessages on", async () => enableDsn({ immediate: true })],
    ["visibility 'mine' and I am not the roller", async () => enableDsn({ visibility: "mine", author: "u2" })],
    ["hideNpcRolls on and the speaker is an NPC", async () => enableDsn({ hideNpcRolls: true, npc: true })],
    ["message is not animating (no _dice3danimating)", async () => enableDsn({ animating: false })],
    ["GM whisper this client is not on", async () => ({ ...(await enableDsn()), whisper: ["gm1"] })],
    ["blind roll and this client is not a GM", async () => ({ ...(await enableDsn()), whisper: ["gm1"], blind: true })],
  ];
  for (const [name, build] of noWait) {
    it(`does not wait: ${name}`, async () => {
      mod.diceBarrier.init();
      const msg = await build();
      vi.useFakeTimers();             // 真去等的话这里会挂到用例超时
      await mod.diceBarrier.awaitDice(msg);
    });
  }
});
```

- [ ] **Step 12: Run it and watch it fail**

Run: `npx vitest run test/dice-barrier.test.mjs -t "effect layer"`
Expected: FAIL —— 15 条用例全部报 `TypeError: Cannot read properties of undefined (reading 'init')` 或 `reading 'awaitDice'`。文件在、纯函数在，但还没有 `export const diceBarrier`。

- [ ] **Step 13: 实现副作用层**

编辑 `scripts/kernel/dice-barrier.mjs`：把 `import { MID, SETTING_DICE_TIMEOUT } from "../const.mjs";` 加到**文件第一行**（`const.mjs` 只是常量文件，不是 Foundry 全局，纯函数区块因此仍满足契约 §0.1），然后把下面这段追加到文件末尾（纯函数区块保持不动、且仍不引用 `MID`）。

```js
const HOOK_DSN_COMPLETE = "diceSoNiceRollComplete";
const DSN_ID = "dice-so-nice";

/** messageId -> Set<() => void>。同一条消息可能有多个等待者。 */
const waiters = new Map();

/** armed 不是「DsN 装了没」，而是「我们的监听器已经装好了」。没装就去等 = 空转到超时。 */
let armed = false;
/** init() 被调用过几次（含幂等空转的那几次）。自检据此报告 diceSoNiceReady 接线是否活着。 */
let initCalls = 0;
/** 设置只注册一次；不依赖桩或 Foundry 暴露「某键是否已注册」的查询面。 */
let settingRegistered = false;

function releaseWaiters(messageId) {
  const set = waiters.get(messageId);
  if (!set) return;
  waiters.delete(messageId);
  for (const fn of set) {
    try { fn(); } catch (err) { console.error(`${MID} | dice barrier waiter failed`, err); }
  }
}

/** 读设置永不抛：未注册的键在真 Foundry 与共享桩里 game.settings.get 都会抛异常。 */
function readSetting(scope, key, fallback) {
  try {
    const v = globalThis.game?.settings?.get?.(scope, key);
    return v === undefined ? fallback : v;
  } catch { return fallback; }
}

function timeoutMs() {
  return Number(readSetting(MID, SETTING_DICE_TIMEOUT, FALLBACK_TIMEOUT_MS));
}

function messageAuthorId(message) {
  return message?.author?.id ?? message?.author ?? message?.user?.id ?? message?.user ?? null;
}

/**
 * DsN 这次到底会不会在本客户端播动画（上表的情况 1 与情况 3）。
 * 最后一道 _dice3danimating 是决定性的：DsN 在自己的 createChatMessage 监听器里同步打上它，
 * 所以渲染阶段读一定准；没有它就说明本客户端这次不播，等下去必然空转满超时。
 * 它前面那两条（hideNpcRolls / visibility "mine"）复刻的是 DsN showForRoll 里
 * 「打完标记之后才 return Promise.resolve(false)」的抑制分支——那两种情况标记在、
 * 事件却永远不来，只靠硬超时的话每张 NPC 卡都要白卡满一个超时。
 * visibility 的三个取值是 "all" | "mine" | "none"，缺省 "all"（Dice3D.DEFAULT_OPTIONS）。
 */
function dsnWillAnimateHere(message) {
  const d3d = globalThis.game?.dice3d;
  if (!d3d) return false;
  if (!armed) return false;
  if (typeof d3d.isEnabled === "function" && !d3d.isEnabled()) return false;
  if (readSetting(DSN_ID, "immediatelyDisplayChatMessages", false) === true) return false;
  let visibility = "all";
  try {
    visibility = globalThis.game?.user?.getFlag?.(DSN_ID, "settings")?.visibility ?? "all";
  } catch { visibility = "all"; }
  if (visibility === "none") return false;
  if (visibility === "mine" && messageAuthorId(message) !== (globalThis.game?.user?.id ?? null)) return false;
  if (readSetting(DSN_ID, "hideNpcRolls", false) === true) {
    const actor = globalThis.ChatMessage?.getSpeakerActor?.(message?.speaker) ?? null;
    if (actor && !actor.hasPlayerOwner) return false;
  }
  return message?._dice3danimating === true;
}

function registerTimeoutSetting() {
  if (settingRegistered) return;
  const settings = globalThis.game?.settings;
  if (typeof settings?.register !== "function") return;
  try {
    settings.register(MID, SETTING_DICE_TIMEOUT, {
      name: "AEA.setting.diceTimeoutMs.name",
      hint: "AEA.setting.diceTimeoutMs.hint",
      scope: "client",          // 动画时长取决于本机与本用户的 DsN 设置，不是世界级的事
      config: true,
      type: Number,
      default: FALLBACK_TIMEOUT_MS,
    });
    settingRegistered = true;
  } catch (err) {
    console.warn(`${MID} | dice timeout setting could not be registered`, err);
  }
}

export const diceBarrier = {
  /** 由 main.mjs 在生命周期钩子 diceSoNiceReady 里调用。幂等。 */
  init() {
    initCalls += 1;
    if (armed) return;
    if (typeof globalThis.Hooks?.on !== "function") return;   // 没有 Hooks 就不算装好，留待重试
    // 系统在 alienrpg.mjs:381 也订阅了这个钩子，但函数体是空的，指望不上。
    globalThis.Hooks.on(HOOK_DSN_COMPLETE, (messageId) => releaseWaiters(messageId));
    armed = true;
    registerTimeoutSetting();
  },

  /**
   * 等这条消息的 3D 骰子落地。永不因为「事件不会来」而挂起。
   * 必须在渲染阶段调用（cards.onRender 的 handler 里）：DsN 的 _dice3danimating
   * 是在它自己的 createChatMessage 监听器里打的，更早读不到。
   * @param {{id?: string, whisper?: unknown[], blind?: boolean, speaker?: object}} message
   * @returns {Promise<void>}
   */
  async awaitDice(message) {
    const messageId = message?.id ?? null;
    if (!messageId) return;
    const plan = pureBarrierPlan({
      dsnActive: dsnWillAnimateHere(message),
      isRecipient: pureIsRecipient({
        whisper: message?.whisper,
        blind: message?.blind === true,
        userId: globalThis.game?.user?.id ?? null,
        isGM: globalThis.game?.user?.isGM === true,
      }),
      timeoutMs: timeoutMs(),
    });
    if (!plan.wait) return;

    await new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        const set = waiters.get(messageId);
        if (set) {
          set.delete(finish);
          if (!set.size) waiters.delete(messageId);
        }
        resolve();
      };
      const timer = setTimeout(finish, plan.timeoutMs);
      let set = waiters.get(messageId);
      if (!set) { set = new Set(); waiters.set(messageId, set); }
      set.add(finish);
    });
  },
};
```

- [ ] **Step 14: Run it and watch it pass**

Run: `npx vitest run test/dice-barrier.test.mjs`
Expected: PASS —— 26 passed（5 + 6 + 15）。

- [ ] **Step 15: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/dice-barrier.mjs test/dice-barrier.test.mjs \
 && git commit -m "feat(kernel): K6 骰子栅栏接管 diceSoNiceRollComplete

契约 §0.2：这个领域钩子归 dice-barrier 自己挂，main.mjs 只在 diceSoNiceReady 里调 init()。
系统在 alienrpg.mjs:381 占了同一个钩子但函数体是空的。

九道放行闸，任何一道成立就立即 resolve，绝不 await 一个不会触发的事件：
没有 messageId／DsN 缺席／监听器没装／isEnabled() 为假／immediatelyDisplayChatMessages／
visibility=mine 而我不是掷骰人／hideNpcRolls 且发言方是 NPC／消息没有 _dice3danimating／
密语盲骰而本客户端不在接收者名单里。
后三条复刻的是 DsN 6.2.9 showForRoll 里「标记已打、事件却不来」的抑制分支，
没有它们每张 NPC 卡都要白卡满一个超时。

不复用 DsN 自带的 waitFor3DAnimationByMessageID：它没有超时，闸只有三道，
且 id 不匹配时用 Hooks.once 无限自我重挂，事件丢失就永久泄漏监听器。

测试全部跑在契约 §0.3 的共享桩上，不就地另造 game/settings/Hooks。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 16: 补 i18n 键**

契约 §7：面向用户的字符串一律走 `game.i18n`，键前缀 `AEA.`、**嵌套结构**（顶层只有 `AEA` 一个键，骨架任务有护栏测试盯着），两个语言文件同步。把下面的键**合并**进 `lang/en.json` 已有的 `AEA` 对象（只加不覆盖别人的键）：

```json
{
  "AEA": {
    "setting": {
      "diceTimeoutMs": {
        "name": "3D dice barrier timeout (ms)",
        "hint": "How long automation waits for Dice So Nice to finish animating before continuing anyway. 4000 is a safe default."
      }
    },
    "selftest": {
      "k6": {
        "armed": "K6 dice barrier is armed",
        "nonblocking": "K6 dice barrier does not stall on a non-animating message"
      }
    }
  }
}
```

同样的键合并进 `lang/cn.json`：

```json
{
  "AEA": {
    "setting": {
      "diceTimeoutMs": {
        "name": "3D 骰子栅栏超时（毫秒）",
        "hint": "自动化最多等 Dice So Nice 的动画多久，超过就照常继续。默认 4000 毫秒。"
      }
    },
    "selftest": {
      "k6": {
        "armed": "K6 骰子栅栏已装好监听",
        "nonblocking": "K6 骰子栅栏不会在不播动画的消息上空转"
      }
    }
  }
}
```

- [ ] **Step 17: Write the failing test（`init()` 登记自检条目、幂等、`initCalls` 计数）**

追加到 `test/dice-barrier.test.mjs` 末尾。自检套件（`kernel/selftest.mjs`，契约 §4）是「需要真 Foundry 的那一半」的可执行验证入口；这里验证三件事：条目确实被登记了、在无 DsN 环境下判定为通过、`init()` 的幂等闸真的挡住了重复登记而 `initCalls` 仍然如实计数。

```js
describe("diceBarrier selftest registration", () => {
  it("registers exactly two k6 entries however many times init() runs", async () => {
    installFoundryStub();
    Object.assign(globalThis.game.user, { id: "u1", isGM: false, getFlag: () => undefined });
    globalThis.game.dice3d = undefined;
    vi.resetModules();
    const mod = await import("../scripts/kernel/dice-barrier.mjs");
    const { selftest } = await import("../scripts/kernel/selftest.mjs");

    mod.diceBarrier.init();
    mod.diceBarrier.init();
    mod.diceBarrier.init();

    const results = await selftest.runAll();
    const k6 = results.filter((r) => r.id.startsWith("k6."));
    expect(k6.map((r) => r.id).sort()).toEqual(["k6.barrier-armed", "k6.barrier-nonblocking"]);
    expect(k6.every((r) => r.ok)).toBe(true);
    expect(k6.find((r) => r.id === "k6.barrier-armed").detail).toContain("initCalls=3");
    uninstallFoundryStub();
  });
});
```

- [ ] **Step 18: Run it and watch it fail**

Run: `npx vitest run test/dice-barrier.test.mjs -t "selftest registration"`
Expected: FAIL —— `AssertionError: expected [] to deeply equal [ 'k6.barrier-armed', 'k6.barrier-nonblocking' ]`（`init()` 目前一条自检都不登记）。

- [ ] **Step 19: 实现自检登记**

编辑 `scripts/kernel/dice-barrier.mjs`：在第一行的 `import { MID, ... }` 下面加 `import { selftest } from "./selftest.mjs";`，把下面这段插在 `registerTimeoutSetting` 之后，并在 `init()` 的 `registerTimeoutSetting();` 后面加一行 `registerSelftests();`。

**`label` 存的是 i18n 键、不是已本地化文本**：登记发生在 Foundry 的 `init` / `diceSoNiceReady` 时机，`runAll()` 才是展示时刻；而且 Foundry 的 `init` 早于 `i18nInit`，那时语言包还没加载，`game.i18n.localize()` 只会把键名原样回声回来 —— 存键永远不比存文本差，运行器负责本地化。

```js
let selftestsRegistered = false;

function registerSelftests() {
  if (selftestsRegistered) return;
  selftestsRegistered = true;

  selftest.register({
    id: "k6.barrier-armed",
    label: "AEA.selftest.k6.armed",      // i18n 键，由 selftest 的运行器本地化
    run() {
      const dsn = globalThis.game?.modules?.get?.(DSN_ID)?.active === true;
      return {
        ok: armed === true,
        detail: `armed=${armed} initCalls=${initCalls} dsnModuleActive=${dsn} timeoutMs=${timeoutMs()}`,
      };
    },
  });

  // 不播动画的消息必须立刻放行。这条一旦失败，症状就是每张卡都白卡满一个超时。
  selftest.register({
    id: "k6.barrier-nonblocking",
    label: "AEA.selftest.k6.nonblocking",
    async run() {
      const t0 = Date.now();
      await diceBarrier.awaitDice({ id: "aea-selftest-not-a-real-message", whisper: [], speaker: {} });
      const ms = Date.now() - t0;
      return { ok: ms < 250, detail: `returned in ${ms}ms (must be < 250)` };
    },
  });
}
```

- [ ] **Step 20: Run it and watch it pass**

Run: `npx vitest run test/dice-barrier.test.mjs`
Expected: PASS —— 27 passed。

- [ ] **Step 21: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/dice-barrier.mjs test/dice-barrier.test.mjs lang/en.json lang/cn.json \
 && git commit -m "feat(kernel): K6 登记两条自检并把超时做成客户端设置

契约 §4 的 kernel/selftest.mjs：需要真 Foundry 的验证一律登记成可执行条目，
MANUAL VERIFICATION 只留给需要人眼判断的部分。
k6.barrier-armed 报告 armed / initCalls / DsN 模块状态 / 当前超时值；
initCalls 是 diceSoNiceReady 接线活着的运行期证据（装了 DsN 应为 2）。
k6.barrier-nonblocking 实测「不播动画的消息必须 250ms 内放行」——
它一旦红，症状就是每张卡白卡满一个超时。
init() 的幂等闸由「三次 init 后仍只有两条 k6 条目」这条用例守住。

label 存 i18n 键而非已本地化文本：登记时刻语言包可能尚未加载，localize 只会回声键名。
设置键取自 const.mjs 的 SETTING_DICE_TIMEOUT，i18n 走嵌套的 AEA.setting.diceTimeoutMs.*。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 22: Write the failing test（`main.mjs` 的三处接线）**

`main.mjs` 是模组的唯一入口文件，由骨架任务写好、里面只有生命周期钩子和一串 `/* AEA-ANCHOR: xxx */` 注释锚点；每个内核模块的属主任务按**锚点文本**（不是行号）往里插自己的东西。本任务要插的三样，每一样漏掉都是静默失效而不是报错失效：

- 漏 import → `main.mjs` 里 `diceBarrier` 是未声明标识符 → 世界启动时 `ReferenceError`，整个模组死在开机（这条是报错失效，但发生在所有人身上）。
- 漏 `api` 槽 → `api.diceBarrier` 恒为 `null` → 自检与手工验证第一行就炸。
- 漏 `diceBarrier.init()` → `armed` 恒假 → 每次 `awaitDice` 立即放行 → 成功数照旧抢在 3D 骰子前面画出来，而且全套单测仍然是绿的。

把 `main.mjs` import 进 vitest 会牵连整棵内核依赖树、还会真去挂 Foundry 钩子，不是这三条断言该付的代价。所以这里**直接对源码文本断言**。新建 `test/dice-barrier-wiring.test.mjs`：

```js
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const MAIN_PATH = fileURLToPath(new URL("../scripts/main.mjs", import.meta.url));
const MAIN = readFileSync(MAIN_PATH, "utf8");

const API_KEYS = ["features", "patches", "resolver", "registry",
                  "rollBus", "diceBarrier", "cards", "selftest"];

/** 从 openIndex 处的 "{" 起，靠配对花括号切出整段（含两端花括号）。 */
function balanced(source, openIndex, what) {
  let depth = 0;
  for (let i = openIndex; i < source.length; i++) {
    if (source[i] === "{") depth++;
    else if (source[i] === "}" && --depth === 0) return source.slice(openIndex, i + 1);
  }
  throw new Error(`unbalanced braces in ${what}`);
}

/** 取出 Hooks.once("<name>", ...) 那个回调的函数体。 */
function hookBody(source, hookName) {
  const m = new RegExp(`Hooks\\.once\\(\\s*["']${hookName}["']`).exec(source);
  if (!m) throw new Error(`Hooks.once("${hookName}", ...) not found in scripts/main.mjs`);
  const open = source.indexOf("{", m.index);
  if (open < 0) throw new Error(`no callback body after Hooks.once("${hookName}"`);
  return balanced(source, open, `the Hooks.once("${hookName}") callback`);
}

/** 取出 export const api = { ... } 那个对象字面量。 */
function apiLiteral(source) {
  const m = /export\s+const\s+api\s*=\s*\{/.exec(source);
  if (!m) throw new Error("export const api = { ... } not found in scripts/main.mjs");
  return balanced(source, m.index + m[0].length - 1, "the api object literal");
}

describe("main.mjs wiring for K6 dice barrier", () => {
  it("imports the module that owns the barrier", () => {
    expect(MAIN).toMatch(/import\s*\{\s*diceBarrier\s*\}\s*from\s+["']\.\/kernel\/dice-barrier\.mjs["']/);
  });

  it("keeps the anchor comments verbatim for the other tasks to locate", () => {
    expect(MAIN).toContain("/* AEA-ANCHOR: imports */");
    expect(MAIN).toContain("/* AEA-ANCHOR: diceSoNiceReady */");
  });

  it("fills in the diceBarrier api slot without adding or removing a key", () => {
    const literal = apiLiteral(MAIN);
    for (const key of API_KEYS) expect(literal).toMatch(new RegExp(`\\b${key}\\b`));
    expect(literal).not.toMatch(/diceBarrier\s*:\s*null/);
    expect(literal).toMatch(/[{,\s]diceBarrier\s*[,}]/);
  });

  it("calls diceBarrier.init() inside the diceSoNiceReady lifecycle hook", () => {
    expect(hookBody(MAIN, "diceSoNiceReady")).toContain("diceBarrier.init()");
  });

  it("does not hook diceSoNiceRollComplete in main.mjs (contract §0.2: that domain hook belongs to dice-barrier.mjs)", () => {
    expect(MAIN).not.toContain("diceSoNiceRollComplete");
  });
});
```

- [ ] **Step 23: Run it and watch it fail**

Run: `npx vitest run test/dice-barrier-wiring.test.mjs`
Expected: FAIL —— `2 failed | 3 passed`：第一条报 `AssertionError: expected '…' to match /import\s*\{\s*diceBarrier…/`，第三条报 `expected '{ features: null, … diceBarrier: null, … }' not to match /diceBarrier\s*:\s*null/`；「anchor comments」「不挂 diceSoNiceRollComplete」两条本来就该绿；第四条此刻也红（`expected '{ /* AEA-ANCHOR: diceSoNiceReady */ }' to contain 'diceBarrier.init()'`）——即 `3 failed | 2 passed`，以实际输出为准，关键是「imports / api 槽 / init 调用」这三条必须是红的。

若报 `Error: ENOENT ... scripts/main.mjs`，说明骨架任务的 `main.mjs` 还没落地；若报 `Hooks.once("diceSoNiceReady", ...) not found`，说明骨架缺了这个生命周期钩子 —— 两种情况都先把骨架补齐再回来。

- [ ] **Step 24: 插入 import 行**

打开 `scripts/main.mjs`，找到这一行（逐字）：

```
/* AEA-ANCHOR: imports */
```

在**它的下一行**插入：

```js
import { diceBarrier } from "./kernel/dice-barrier.mjs";
```

**不要改动锚点注释本身的一个字符** —— 契约 §5 规定所有任务全靠锚点文本定位，改了就把别人的插入步骤打断了。这一行与别的任务插在同一个锚点后，彼此顺序无关（ESM 的 import 会被提升，谁前谁后都一样）。

- [ ] **Step 25: 填 `api` 的 `diceBarrier` 槽**

同一个文件里找到骨架写好的那个八键字面量：

```js
export const api = {
  features: null, patches: null, resolver: null, registry: null,
  rollBus: null, diceBarrier: null, cards: null, selftest: null,
};
```

把 `diceBarrier: null` 改成简写形式 `diceBarrier`（引用 Step 24 那行 import 进来的对象）：

```js
export const api = {
  features: null, patches: null, resolver: null, registry: null,
  rollBus: null, diceBarrier, cards: null, selftest: null,
};
```

契约 §5 的硬规矩，逐条对照着做：**只换自己那一个 `null`**；**不得**替换 `api` 这个对象本身；**不得**增删任何键；别人的 `null` 一个都不许动（那是他们各自任务的活儿，此刻还没干完是正常的）。

- [ ] **Step 26: Run it and watch two of three go green**

Run: `npx vitest run test/dice-barrier-wiring.test.mjs`
Expected: `1 failed | 4 passed` —— 只剩 `calls diceBarrier.init() inside the diceSoNiceReady lifecycle hook` 是红的，报 `AssertionError: expected '{ /* AEA-ANCHOR: diceSoNiceReady */ }' to contain 'diceBarrier.init()'`。

- [ ] **Step 27: 把 `diceBarrier.init()` 插进 `diceSoNiceReady` 钩子**

骨架里这个钩子是**一行写完**的，锚点在花括号中间：

```js
Hooks.once("diceSoNiceReady", () => { /* AEA-ANCHOR: diceSoNiceReady */ });
```

把它拆成三行，锚点注释**原样保留**、调用写在锚点的下一行：

```js
Hooks.once("diceSoNiceReady", () => {
  /* AEA-ANCHOR: diceSoNiceReady */
  diceBarrier.init();
});
```

时序说明（免得有人担心）：`diceSoNiceReady` 由 DsN 在它自己的 `ready` 钩子里、且在一次 await 之后广播，所以 `diceBarrier.init()` 跑在本模组 ready 段的各项安装**之后**。这没有问题：栅栏只在渲染时被 `awaitDice` 调用，那时它早已 armed。DsN 缺席时这个钩子永不触发、`init()` 永不执行、`armed` 保持假、`awaitDice` 一律立即放行 —— 正是我们要的降级行为。

- [ ] **Step 28: Run both files and watch them pass**

Run: `npx vitest run test/dice-barrier.test.mjs test/dice-barrier-wiring.test.mjs`
Expected: PASS —— 32 passed（27 + 5）。

- [ ] **Step 29: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/main.mjs test/dice-barrier-wiring.test.mjs \
 && git commit -m "feat(main): K6 的三处接线——import、api 槽、diceSoNiceReady 调用

契约 §5「api 的装配规则」：内核模块的属主任务做三件事、一件不多。
上一版这三件事全被当成别人已经做好了，实际结果是：
main.mjs 从未 import 本模块（启动即 ReferenceError）、
api.diceBarrier 恒为 null（自检与手工验证第一行就炸）、
diceBarrier.init() 无人认领（armed 恒假、awaitDice 一律立即放行，
单测照样全绿，线上表现就是成功数抢在 3D 骰子前面画出来）。

api 只把 diceBarrier 那一个 null 换成 import 进来的对象，不增删键、不替换对象本身。
接线一律按锚点文本定位，不用行号；diceSoNiceReady 那行由一行拆成三行，锚点注释原样保留。

test/dice-barrier-wiring.test.mjs 直接读 main.mjs 源码断言，不 import 它——
那会牵连整棵内核依赖树、还会真去挂 Foundry 钩子。
顺带守住契约 §0.2：diceSoNiceRollComplete 这个领域钩子不许出现在 main.mjs 里。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 30: MANUAL VERIFICATION（vitest 测的是桩，这一步测真 DsN）**

上面的用例跑在共享 Foundry 桩上，它证明不了「真的 DsN 真的会带着那个 message id 广播完成事件」，也证明不了等待时长和骰子停下的时刻对得上。

前置条件：世界启用 `alienrpg` 系统（4.1.13）与 `dice-so-nice` 模组（6.2.9），本模组已正常启动，场上有一个 character actor 和一个 NPC（`hasPlayerOwner` 为假的）actor。

在浏览器控制台（F12）里执行：

```js
const api = game.modules.get("alien-evolved-automation").api;
console.log("api.diceBarrier =", api.diceBarrier);   // 必须是对象，不是 null
await api.selftest.runAll();                          // 再看两条 k6 是不是绿的
// 用角色卡掷一次普通属性骰，趁 3D 骰子还在滚的时候立刻跑：
const msg = game.messages.contents.at(-1);
const t0 = performance.now();
await api.diceBarrier.awaitDice(msg);
console.log("barrier ms =", Math.round(performance.now() - t0), "animating =", !!msg._dice3danimating);
```

逐条确认：

1. **三处接线都活着**：`api.diceBarrier` 打印出的是对象而不是 `null`（Step 25 的 api 槽）；`k6.barrier-armed` 是 `ok: true`，且它的 `detail` 里 `armed=true`、**`initCalls=1`**、`dsnModuleActive=true`、`timeoutMs=4000`（`initCalls` 为 1 正说明 Step 27 那唯一一处调用点跑到了；若是 `initCalls=0` 或 `armed=false`，回去 grep `main.mjs` 里的 `diceBarrier.init()`）。同时 `k6.barrier-nonblocking` 也是 `ok: true`。
2. **公开掷骰 + DsN 开着**：`barrier ms` 落在 500–4000 之间，且 `await` 是在骰子停下那一刻返回的（肉眼可对上）。
3. **把 3D 骰子可见度设成 “Hide all”**（DsN 设置里 `visibility === "none"`）后重复第 2 步：`barrier ms` 应为 **0 或 1**。
4. **打开 DsN 的 “Hide NPC rolls”，用 NPC 掷一次**后重复：`barrier ms` 应为 **0 或 1**，**绝不能是 4000** —— 是 4000 就说明 `hideNpcRolls` 那道闸没生效，回去查 `ChatMessage.getSpeakerActor(msg.speaker)?.hasPlayerOwner` 到底返回了什么。
5. **以非 GM 玩家身份登录，让 GM 掷一次盲骰**（敌对 NPC 的掷骰走 `YZEDiceRoller.mjs:412-414`），在玩家客户端重复：`barrier ms` 应为 **0 或 1**。若是 4000，打印 `console.log(msg.whisper, msg.blind)` 看实际形态，对照 `pureIsRecipient` 的两种 whisper 形态处理。
6. **设置项**：在 Configure Settings 里能看到「3D 骰子栅栏超时（毫秒）」这一项、且它在**本客户端**分区（不是世界分区）；把它改成 1000 后重复第 2 步，`barrier ms` 的上限随之变成 1000。
7. **已知且有意为之的边界（不是缺陷，记录在案以免被当成 bug 重报）**：在**没装 DsN** 的世界里，`diceSoNiceReady` 永不触发 → `init()` 不跑 → 这条设置不会出现在 Configure Settings 里。此时它也没有任何消费者（`awaitDice` 恒立即放行），行为无差别。要验证的话：停用 `dice-so-nice` 并重载世界，确认设置面板里没有这一项、且掷骰后 `await api.diceBarrier.awaitDice(game.messages.contents.at(-1))` 仍然是 0–1ms 返回。（终检曾要求把这条设置改到 `init` 阶段注册，需要给 K6 加第三个导出成员 `registerSettings()`；契约 §4 把 K6 的导出面钉死为两个成员且「签名不得改动」，故未采纳，已上报待裁决。）
