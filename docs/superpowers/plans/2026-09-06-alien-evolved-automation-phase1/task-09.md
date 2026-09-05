> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 9 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 9: K8 Cards —— 模组自有的聊天卡契约与渲染扇出

**Files:**
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/kernel/cards.mjs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/templates/aea-card-actions.hbs`
- Create: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/test/cards.test.mjs`
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/scripts/main.mjs`（三处、且只有三处：加一行 import、把 `api` 里 `cards: null` 换成 `cards,`、在 `ready.cards` 子锚点后插一行 `cards.init();`）
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/styles/alien-evolved-automation.css`
- Modify: `C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation/lang/en.json`、`.../lang/cn.json`

**Interfaces:**

- **Consumes**
  - `scripts/const.mjs` 的 `MID`（值为字符串 `"alien-evolved-automation"`）：日志前缀与模板 URL 拼接。
  - `scripts/kernel/rollbus.mjs` 的 `rollBus.recordOf(message)` -> `RollRecord|null`（读 `message.flags["alien-evolved-automation"].roll`；不要求 `rollBus.install()` 已经跑过）。渲染扇出把它的结果作为 `record` 交给各特性。该文件由别的任务建立，本任务只 import，**不得重建、不得改它**。
  - `scripts/kernel/selftest.mjs` 的 `selftest.register({id, label, run})`；`run()` 返回 `{ok:boolean, detail:string}`，可 async。`label` 传的是 **i18n 键**（不是已本地化文本），运行器负责本地化。该文件由别的任务建立，本任务只 import 后登记条目。
  - `scripts/main.mjs`：由骨架任务建立，交付时逐字带十个锚点注释，且 `export const api` 是一个八个键、值全为 `null` 的对象字面量：
    ```js
    export const api = {
      features: null, patches: null, resolver: null, registry: null,
      rollBus: null, diceBarrier: null, cards: null, selftest: null,
    };
    ```
    **`cards` 这一个槽的属主就是本任务**：本任务把 `cards: null` 换成 `cards,`，**不替换 `api` 这个对象本身，不增删任何键**（其余七个槽各有属主，动它们会把别人的装配覆盖掉）。
  - `test/stubs/foundry.mjs` 的 `installFoundryStub(options)` / `uninstallFoundryStub()`。桩由骨架任务独占实现，其行为是契约定死的：`installFoundryStub()` 幂等；`Hooks.callAll(name, ...args)` **真正同步派发**给经 `Hooks.on` / `Hooks.once` 注册的回调；`game.release.generation` 由 `installFoundryStub({ generation })` 指定（默认 14）；`game.system.version` 默认 `"4.1.13"`。本任务**只读不改**这个文件，**禁止**在测试里自建 `globalThis.game` / `globalThis.Hooks`，也**禁止**用私有 Map 顶替 `game.settings`。唯一的例外见下一条。
  - `foundry.applications.handlebars.{loadTemplates, renderTemplate}`：这是 Foundry V13 起模板预加载器与模板渲染器的真实位置。桩只保证 `globalThis.foundry.applications` 这个对象存在（它铺的是 `foundry.applications.api.*`），**没有**铺 handlebars 子对象。所以测试里对 `globalThis.foundry.applications.handlebars` 赋一个 `{ loadTemplates: vi.fn(), renderTemplate: vi.fn() }` —— 这是给被测代码要调的那两个函数装探针，不是另造一套 `game`/`settings` 后端，两者别混为一谈。

- **Produces**
  - ```js
    export const cards = {
      init(),
      mount(element, message),            // 幂等；返回 <div class="aea-mount">|null
      render(target, templatePath, data), // 只替换 target 自身的 innerHTML；返回 Promise<void>
      registerAction(name, handler),      // handler(event, {action, message, messageId, element, mount})
      onRender(name, handler),            // handler({message, element, mount, record}) —— 唯一的渲染扇出点
    };
    ```
    这五个成员就是全部，**不许多导出第六个成员**（契约 §4 K8 的签名不得改动）。
  - `export const TEMPLATE_CARD_ACTIONS = "modules/alien-evolved-automation/templates/aea-card-actions.hbs"`，渲染上下文 `{cardId, title, actions: [{name, label, icon, value, disabled}]}`；`title` 与 `label` 传的是 i18n **键**，模板里走 `{{localize}}`。
  - 纯决策函数 `pureCardDispatch({action, insideMount, registered})`、`pureCardMessageId({mountMessageId, cardMessageId})`（供真单测；不碰任何 Foundry 全局）。
  - 自检条目五条：`k8.mount-idempotent`、`k8.delegated-click`、`k8.render-target`、`k8.render-fanout`、`k8.lazy-mount`。
  - i18n 键：`AEA.selftest.k8.mountIdempotent` / `.delegatedClick` / `.renderTarget` / `.renderFanout` / `.lazyMount`（en + cn 各一份）。
  - CSS 类：`.aea-mount` / `.aea-actions` / `.aea-actions-title` / `.aea-actions-row` / `.aea-action`。
  - **`render()` 的边界（这是契约 §4 K8 的硬规定，实现与文档都必须照此）**：`render(target, templatePath, data)` **只**把渲染结果写进 `target.innerHTML`，**绝不**触碰 `target` 的父节点、兄弟节点，也不在挂点里替调用方创建任何容器。一期有两条特性共用同一个 `.aea-mount`，**所以每条特性必须自己先在挂点下建一个类名 `aea-<feature-id>` 的子元素**（`mount.querySelector` 复用、没有才创建），再对那个子元素调 `render()`。直接对 `mount` 调 `render()` 会清掉另一条特性刚写进去的内容，而且症状取决于 `onRender` 的注册顺序（时灵时不灵）。本实现对「target 就是 `.aea-mount` 本身」这种调用会 `console.warn` 一句，把这种静默互相覆盖变成有声的。
  - **`onRender` 的五条硬约定（下游特性必须照做；写在这里，是因为特性作者只看得到他自己那份任务）**：
    1. handler **同步执行**，cards **不 await 它的返回值**。要等骰子就在自己的异步续段里等，别把 `onRender` 写成 `async` 后指望 cards 排队。
    2. `ctx.mount` 是**惰性**的：只有第一次读它才真的往卡里注入 `.aea-mount`。所以**不要在形参上解构 `{mount}`**（解构即读取，会给每张卡都造出一个空挂点），写成 `(ctx) => {...}` 再按需 `ctx.mount`。它返回 `null` 表示这张卡注入不了，调用方必须判空。
    3. 先建自己的 `aea-<feature-id>` 子元素，再对**那个子元素**调 `cards.render()`；永远不要把 `ctx.mount` 直接当 `render()` 的第一个参数。
    4. 任何**写判定结果**的渲染，必须先 `await diceBarrier.awaitDice(ctx.message)`，等待返回后**复查 `isConnected`** 再写；只画控件、不画结果的可以不等。
    5. handler 同步抛出的异常由 cards 捕获并 `console.error` 后继续调用下一个；**异步**拒绝由 cards 兜一层 `.then(null, …)` 记日志，但特性自己也该 `.catch`。

    典型写法：
    ```js
    cards.onRender("my-feature", (ctx) => {
      if (ctx.record?.kind !== "skill") return;          // 不关心 → 直接返回，挂点根本不会被创建
      const mount = ctx.mount;                           // 第一次读取才注入 .aea-mount
      if (!mount) return;
      let section = mount.querySelector(".aea-my-feature");
      if (!section) {
        section = (mount.ownerDocument ?? document).createElement("div");
        section.className = "aea-my-feature";
        mount.appendChild(section);                      // 自己的地盘，别人的 section 不受影响
      }
      void (async () => {
        await diceBarrier.awaitDice(ctx.message);
        if (!section.isConnected) return;                // 等待期间聊天日志可能已重渲染
        await cards.render(section, TEMPLATE_X, data);   // 只替换 section 的 innerHTML
      })().catch((err) => console.error(err));
    });
    ```
  - **删掉的假 DOM 断言 → 替代物的逐条映射**（vitest 默认 node 环境没有 `document`，这些断言只能在真 Foundry 里成立；少一条自检就等于少一条断言，这张表让它显形）：

    | 要证明的性质 | 替代它的可执行检查 |
    |---|---|
    | 挂点幂等、只 append 到末尾 | selftest `k8.mount-idempotent` + 手工验收第 3 条 |
    | 不打断系统 `multiPush` 与推骰按钮的相邻关系 | selftest `k8.mount-idempotent`（`multiPushStillPrecedesPush`）+ 手工验收第 6 条 |
    | 委托点击能送达已注册动作 | selftest `k8.delegated-click` + 手工验收第 4 条 |
    | `render()` 只改 target 自身、邻居的内容原样保留 | selftest `k8.render-target` |
    | 渲染扇出把**真实**元素与记录交给特性、钩子名选对 | selftest `k8.render-fanout` + 手工验收第 2 条 |
    | 没有特性关心的卡不产生挂点 | selftest `k8.lazy-mount` + 手工验收第 5 条 |
    | 扇出顺序、异常隔离、惰性挂点只创建一次、钩子只绑一次 | `test/cards.test.mjs` 的真单测（经桩的 `Hooks.callAll` 真派发触发，不伪造 DOM） |

---

**背景（实现者必读，动手前读完）**

- Foundry VTT 是一个跑在浏览器里的桌面 RPG 平台。聊天日志里每条消息是一个 `<li class="chat-message" data-message-id="...">`，里面有 `.message-header` 与 `.message-content`。
- 消息渲染时 Foundry 广播渲染钩子。**V13 起是 `renderChatMessageHTML(message, html, context)`，`html` 是原生 `HTMLElement`**；旧名 `renderChatMessage(message, html, data)` 的 `html` 是 jQuery 对象（数组式，`html[0]` 才是元素），现在只剩一个带弃用警告的兼容层。已核实：`systems/alienrpg/system.json:22-26` 的 `compatibility` 是 `{minimum:"13", verified:"14", maximum:"14"}`、`:21` 版本 `4.1.13`，而系统自己在 `module/alienrpg.mjs:464` 绑的是**旧名**（全系统源码里 grep 不到一处 `renderChatMessageHTML`）。
- **钩子归属（契约 §0.2）**：`renderChatMessageHTML`（V13+）与 `renderChatMessage`（旧版回退）**归本文件独占** —— 在 `cards.mjs` 自己的 `init()` 里挂，全模组只此一处，特性不得自挂。因此**特性拿到挂点的唯一通路是 `cards.onRender(name, handler)`**（契约 §4 K8）：cards 在自己那个唯一的渲染监听器里，按注册顺序把 `{message, element, mount, record}` 回调给每条特性，逐条捕获异常，绝不让一条特性打断整张卡的渲染。

**必须亲眼读过的反面教材：`systems/alienrpg/module/alienrpg.mjs:464-497`（已逐行核对）**

```js
Hooks.on("renderChatMessage", (message, html, data) => {          // :464
  html.find("button.alien-Push-button").each((i, li) => {         // :465
    li.addEventListener("click", (ev) => {                        // :467
      const tarG = ev.target.previousElementSibling.checked;      // :468
      const actor = game.actors.get(message.speaker.actor);       // :472
      switch (actor.type) {                                       // :487
        case "character":                                         // :488
          actor.pushRoll(actor, reRoll, hostile, blind, message); // :489
          break;                                                  // :490
        default:                                                  // :491
          return;                                                 // :492
      }
    });
  });
});
```

它一次踩中五个坑，本任务每一个都不许重犯：

1. 绑的是**已废弃的旧名**（:464）。兼容层一消失，整个推骰按钮静默失效。
2. **每次重渲染都重新绑一遍**（:465、:467）：聊天日志一重渲染就再叠一层监听器。
3. `game.actors.get(message.speaker.actor)`（:472）丢掉 token 信息：场上三只同名 Drone 时全部落到共享的基础 actor。
4. `:491-492` 对非 `character` 类型直接 `return`：NPC、怪物、飞船卡上按了推骰什么都不发生，也没有任何提示。
5. `ev.target.previousElementSibling.checked`（:468）靠 DOM 相邻关系读「多重推骰」复选框。**任何插到按钮前面的节点都会打断它**（读出 `undefined`，多重推骰静默失效）。这条直接决定本任务的挂载约束：**只能 append 到 `.message-content` 末尾，绝不能插到 `input.multiPush` 与 `button.alien-Push-button` 之间。**

**为什么必须自己注入挂点（已 grep 核实）**

整份 `alienrpg` 4.1.13 里 `dmgBtn-container` 只出现两次，`module/helpers/YZEDiceRoller.mjs:390` 与 `:391`，而这两行位于 `:378` 开始、`:392` 结束的分支之内：

```js
if (!reRoll || reRoll === "mPush") {                                              // :378
  if (reRoll !== "mPush") {                                                       // :379
    chatMessage += `<span style="font-size:larger">` + game.i18n.localize("ALIENRPG.MultiPush")
      + " " + `</span> <input class="multiPush" name="multiPush" type="checkbox" ... /> `  // :380-384
  }
  chatMessage += `<button class="alien-Push-button" title="PUSH Roll?">` + ... + "</button>"  // :388-389
  chatMessage += `<span class="dmgBtn-container" ...></span>`                     // :390
  chatMessage += `<span class="dmgBtn-container" ...></span>`                     // :391
}                                                                                 // :392
```

`reRoll === "push"` 时（推骰产生的那张卡）条件为假，两个 span **根本不会被写进 HTML**；NPC 卡、怪物卡同理。把它当锚点，会在恰恰最需要自动化的那些卡上悄悄失效。所以模组注入自己的 `.aea-mount`。

**委托根的选择（契约 §4 K8 已明文认可）**

委托监听挂在 `document` 上、以 `.aea-mount` 作用域守卫，**不**字面挂 `#chat-log`：ChatLog 重渲染会整体替换 `#chat-log` 元素（绑在旧元素上的监听器随之丢失，就得重绑，那正是 :465 的反模式），且聊天弹出窗口里有第二个同 id 元素。行为是 `#chat-log` 委托的严格超集，仍然只有一个监听器、仍然按 `data-action` 分发。

**测试策略（认真读，别绕开）**

仓库只有 vitest 一个 devDependency，没有 jsdom，vitest 默认 node 环境**没有 `document`**。手搓一个假 `document` 只能证明「作者写的假 `querySelector` 和作者写的代码互相同意」。所以本任务分三层：

- **真单测**（vitest）：两个纯决策函数；`registerAction` / `onRender` 的参数校验；`init()` 选哪个钩子名、只绑一次、缺 `document` 不炸、预加载模板；**渲染扇出的顺序 / 异常隔离 / 惰性挂点**；`render()` 的保护分支与「只写 target 自身」；模板文件与语言包的内容契约；`main.mjs` 的三处接线（读文件断言）。**钩子名与只绑一次这两件事一律用行为断言**：经桩的 `Hooks.callAll(name, ...)`（契约 §0.3 保证它真派发）看回调有没有被叫到、被叫了几次 —— 不去翻 `ctx.hooks` 的条目形状，桩内部换形状也不会波及本文件。
- **真 DOM 检查**：挂载幂等、事件委托、`render()` 的作用域、扇出拿到真元素、无人关心的卡不产生挂点 —— 五条 `selftest` 条目，在真 Foundry 里跑真 `document`、真 `MouseEvent`、真 Handlebars。它们是**可执行**的，不是散文；在 node 里跑到时统一返回 `ok:false, detail:"requires Foundry ..."`，不会假装通过。
- **MANUAL VERIFICATION**：只留给需要人眼判断的部分（真实卡片种类、重渲染、系统多重推骰是否还灵）。

---

- [ ] **Step 1: Write the failing test（两个纯决策函数）**

新建 `test/cards.test.mjs`：

```js
import { describe, it, expect } from "vitest";
import { pureCardDispatch, pureCardMessageId } from "../scripts/kernel/cards.mjs";

describe("pureCardDispatch", () => {
  it("dispatches only for a registered action inside our own mount", () => {
    expect(pureCardDispatch({ action: "aea-push", insideMount: true, registered: true }))
      .toEqual({ dispatch: true, reason: "ok" });
  });

  it("refuses a click that hit no data-action element", () => {
    expect(pureCardDispatch({ action: null, insideMount: true, registered: true }))
      .toEqual({ dispatch: false, reason: "no-action" });
    expect(pureCardDispatch({ action: "", insideMount: true, registered: true }).reason).toBe("no-action");
    expect(pureCardDispatch()).toEqual({ dispatch: false, reason: "no-action" });
  });

  it("refuses a data-action element outside our mount (system and other modules' buttons)", () => {
    expect(pureCardDispatch({ action: "aea-push", insideMount: false, registered: true }))
      .toEqual({ dispatch: false, reason: "outside-mount" });
  });

  it("refuses an unregistered action name", () => {
    expect(pureCardDispatch({ action: "aea-unknown", insideMount: true, registered: false }))
      .toEqual({ dispatch: false, reason: "unregistered" });
  });

  it("never touches a Foundry global", () => {
    const saved = { game: globalThis.game, Hooks: globalThis.Hooks, document: globalThis.document };
    delete globalThis.game; delete globalThis.Hooks; delete globalThis.document;
    try {
      expect(pureCardDispatch({ action: "a", insideMount: true, registered: true }).dispatch).toBe(true);
    } finally { Object.assign(globalThis, saved); }
  });
});

describe("pureCardMessageId", () => {
  it("prefers the id stamped on our own mount", () => {
    expect(pureCardMessageId({ mountMessageId: "m1", cardMessageId: "m2" })).toBe("m1");
  });

  it("falls back to the enclosing chat message's id", () => {
    expect(pureCardMessageId({ mountMessageId: "", cardMessageId: "m2" })).toBe("m2");
    expect(pureCardMessageId({ cardMessageId: "m2" })).toBe("m2");
  });

  it("trims whitespace and ignores non-strings", () => {
    expect(pureCardMessageId({ mountMessageId: "  m1  " })).toBe("m1");
    expect(pureCardMessageId({ mountMessageId: 42, cardMessageId: "m2" })).toBe("m2");
  });

  it("returns null when nothing is known", () => {
    expect(pureCardMessageId({ mountMessageId: "", cardMessageId: "  " })).toBeNull();
    expect(pureCardMessageId()).toBeNull();
  });
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs`
Expected: FAIL —— 整个文件加载失败，`Error: Failed to load url ../scripts/kernel/cards.mjs`（文件还不存在）。

- [ ] **Step 3: 建 `cards.mjs`，只写两个纯函数**

新建 `scripts/kernel/cards.mjs`：

```js
import { MID } from "../const.mjs";

/**
 * K8 · Cards —— 模组自有的聊天卡契约。
 *
 * 不复用系统的 dmgBtn-container 挂点：整份 alienrpg 4.1.13 里它只出现两次，
 * helpers/YZEDiceRoller.mjs:390 与 :391，两行都在 :378 的
 *   if (!reRoll || reRoll === "mPush")
 * 分支内部（到 :392 结束）。reRoll === "push" 的推骰卡、以及 NPC 卡 / 怪物卡上
 * 这两个 span 根本不会被写进 HTML。
 */

const MOUNT_CLASS = "aea-mount";
const ACTION_ATTR = "data-action";

/** 模组自有模板。Foundry 按 Data 根目录下的 URL 解析模板路径。 */
export const TEMPLATE_CARD_ACTIONS = `modules/${MID}/templates/aea-card-actions.hbs`;

/** 委托点击到底该不该派发。纯决策，不碰 DOM。 */
export function pureCardDispatch(input = {}) {
  const { action, insideMount, registered } = input ?? {};
  if (!action) return { dispatch: false, reason: "no-action" };
  if (!insideMount) return { dispatch: false, reason: "outside-mount" };
  if (!registered) return { dispatch: false, reason: "unregistered" };
  return { dispatch: true, reason: "ok" };
}

/** 挂点上的 id 优先，其次退到外层 li[data-message-id]。 */
export function pureCardMessageId(input = {}) {
  const { mountMessageId, cardMessageId } = input ?? {};
  const own = typeof mountMessageId === "string" ? mountMessageId.trim() : "";
  if (own) return own;
  const card = typeof cardMessageId === "string" ? cardMessageId.trim() : "";
  return card || null;
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `npx vitest run test/cards.test.mjs`
Expected: PASS —— 9 passed。

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/cards.mjs test/cards.test.mjs \
 && git commit -m "feat(kernel): K8 委托分发的两个纯决策函数

把「该不该派发」和「这条消息的 id 是什么」从 DOM 里抠出来做成纯函数，
它们能进真单测；剩下的 querySelector/closest 部分靠 selftest 在真 Foundry 里验。
node 环境没有 document，手搓假 DOM 只能证明作者和自己一致。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 6: Write the failing test（`registerAction` 参数校验）**

追加到 `test/cards.test.mjs` 末尾。ESM 的 `import` 会被提升到模块顶部，所以把它写在追加段落的开头是合法的；这样每一段自带它需要的符号，读起来不用来回翻。

```js
import { cards } from "../scripts/kernel/cards.mjs";

describe("cards.registerAction", () => {
  it("rejects an empty name loudly", () => {
    expect(() => cards.registerAction("", () => {})).toThrow(/non-empty name/);
    expect(() => cards.registerAction(undefined, () => {})).toThrow(/non-empty name/);
  });

  it("rejects a non-function handler loudly", () => {
    expect(() => cards.registerAction("aea-x", "not a function")).toThrow(/handler function/);
  });

  it("accepts a valid pair", () => {
    expect(() => cards.registerAction("aea-ok", () => {})).not.toThrow();
  });
});
```

- [ ] **Step 7: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs`
Expected: FAIL —— 整个文件加载失败（前 9 条也一并变红），`SyntaxError: The requested module '../scripts/kernel/cards.mjs' does not provide an export named 'cards'`。

- [ ] **Step 8: 实现 `actions` 表与 `registerAction`**

追加到 `scripts/kernel/cards.mjs` 末尾：

```js
/** 已注册的动作：name -> handler。 */
const actions = new Map();

export const cards = {
  /**
   * 注册一个 data-action 处理器。
   * @param {string} name 按钮上的 data-action 值，约定以 "aea-" 开头
   * @param {(event: Event, ctx: {action: string, message: object|null, messageId: string|null,
   *          element: HTMLElement, mount: HTMLElement}) => void} handler
   */
  registerAction(name, handler) {
    if (typeof name !== "string" || !name) throw new Error(`${MID} | registerAction requires a non-empty name`);
    if (typeof handler !== "function") throw new Error(`${MID} | registerAction requires a handler function`);
    actions.set(name, handler);
  },
};
```

- [ ] **Step 9: Run it and watch it pass**

Run: `npx vitest run test/cards.test.mjs`
Expected: PASS —— 12 passed。

- [ ] **Step 10: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/cards.mjs test/cards.test.mjs \
 && git commit -m "feat(kernel): K8 registerAction 与动作表

handler 形参按契约 §4 K8 定死为 (event, {action, message, messageId, element, mount})。
参数不合法立刻抛，不静默吞——静默吞的后果是按钮永远没反应且没人知道为什么。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 11: Write the failing test（`onRender` 参数校验）**

追加到 `test/cards.test.mjs` 末尾：

```js
describe("cards.onRender", () => {
  it("rejects an empty name loudly", () => {
    expect(() => cards.onRender("", () => {})).toThrow(/non-empty name/);
    expect(() => cards.onRender(undefined, () => {})).toThrow(/non-empty name/);
  });

  it("rejects a non-function handler loudly", () => {
    expect(() => cards.onRender("aea-x", null)).toThrow(/handler function/);
  });

  it("accepts a valid pair and tolerates re-registering the same name", () => {
    expect(() => cards.onRender("aea-ok", () => {})).not.toThrow();
    expect(() => cards.onRender("aea-ok", () => {})).not.toThrow();
  });
});
```

- [ ] **Step 12: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs -t "cards.onRender"`
Expected: FAIL —— 三条全红；第三条直接报 `TypeError: cards.onRender is not a function`。

- [ ] **Step 13: 实现 `renderers` 表与 `onRender`**

在 `scripts/kernel/cards.mjs` 里，紧跟 `const actions = new Map();` 之后插入：

```js
/**
 * 已注册的渲染回调：name -> handler。
 * Map 保插入序 = 调用序；同名重复注册只替换实现、保留原有位置
 * （重复 init / 热重载不会让同一条特性排两遍）。
 */
const renderers = new Map();
```

并把这个成员加进 `export const cards`（放在 `registerAction` 之后）：

```js
  /**
   * 注册一个渲染回调 —— 特性拿到挂点的唯一通路（契约 §4 K8）。
   * 渲染钩子由本文件独占，特性不得自挂。
   *
   * handler 同步执行，cards 不 await 它的返回值。ctx.mount 是惰性的：
   * 只有第一次读取才真的注入 .aea-mount，所以不要在形参上解构 {mount}，
   * 否则每张卡都会多出一个空挂点。
   * 拿到 mount 之后要先建自己的 aea-<feature-id> 子元素，再对那个子元素调
   * cards.render()：render 只替换 target 自身的 innerHTML，直接冲着 mount 渲染
   * 会把同一张卡上别的特性刚写的内容抹掉。
   *
   * @param {string} name 特性 id，用于日志与去重
   * @param {(ctx: {message: object, element: HTMLElement,
   *          mount: HTMLElement|null, record: object|null}) => void} handler
   */
  onRender(name, handler) {
    if (typeof name !== "string" || !name) throw new Error(`${MID} | onRender requires a non-empty name`);
    if (typeof handler !== "function") throw new Error(`${MID} | onRender requires a handler function`);
    renderers.set(name, handler);
  },
```

- [ ] **Step 14: Run it and watch it pass**

Run: `npx vitest run test/cards.test.mjs`
Expected: PASS —— 15 passed。

- [ ] **Step 15: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/cards.mjs test/cards.test.mjs \
 && git commit -m "feat(kernel): K8 onRender 的注册面

契约 §4 K8：onRender(name, handler) 是特性拿到挂点的唯一扇出点。
按 name 存进 Map：插入序即调用序，同名重注册只换实现、不改位置，
所以重复 init 或热重载不会让同一条特性排两遍。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 16: Write the failing test（`init()` 的钩子选择 + 渲染扇出的行为）**

这两组测试写在同一步，因为它们咬合：扇出回调只能靠触发 `init()` 挂上去的那个钩子来驱动，没法先测其中一个。

两组都**不去翻桩的内部记录**：契约 §0.3 保证 `Hooks.callAll(name, ...args)` 会真正同步派发给经 `Hooks.on` 注册的回调，所以「绑没绑对名字」「绑了几次」直接用行为断言 —— 发一次事件，看回调被叫了几次。桩以后换内部形状也波及不到本文件。

追加到 `test/cards.test.mjs` 末尾：

```js
import { beforeEach, afterEach, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";

// element 只是个占位对象：unwrap() 认元素的判据就是「有 querySelector」。
// 真正的挂载与委托由 selftest 在真 Foundry 的真 DOM 里验，这里只看扇出逻辑。
const fakeElement = { querySelector: () => null };

describe("cards.init", () => {
  let mod, rollbus;

  /**
   * 桩装的是契约 §0.3 定死的那套全局；generation 走 options，不手改 game.release。
   * foundry.applications.handlebars 是 V13 起 loadTemplates/renderTemplate 的真实位置，
   * §0.3 的桩没铺这个子对象（它铺的是 foundry.applications.api.*），
   * 所以在这里给被测代码要调的那两个函数装探针。
   */
  async function boot(generation, { withLoader = true } = {}) {
    installFoundryStub({ generation });
    if (withLoader) globalThis.foundry.applications.handlebars = { loadTemplates: vi.fn() };
    else { delete globalThis.foundry.applications.handlebars; delete globalThis.loadTemplates; }
    vi.resetModules();
    mod = await import("../scripts/kernel/cards.mjs");
    rollbus = await import("../scripts/kernel/rollbus.mjs");
    rollbus.rollBus.recordOf = vi.fn(() => null);
    return mod;
  }

  afterEach(() => uninstallFoundryStub());

  it("binds the V13+ hook name and ignores the legacy one", async () => {
    const m = await boot(14);
    m.cards.init();
    const seen = [];
    m.cards.onRender("probe", (c) => seen.push(c.message.id));
    globalThis.Hooks.callAll("renderChatMessageHTML", { id: "new" }, fakeElement);
    globalThis.Hooks.callAll("renderChatMessage", { id: "old" }, fakeElement);
    expect(seen).toEqual(["new"]);
  });

  it("falls back to the legacy hook name below V13", async () => {
    const m = await boot(12);
    m.cards.init();
    const seen = [];
    m.cards.onRender("probe", (c) => seen.push(c.message.id));
    globalThis.Hooks.callAll("renderChatMessageHTML", { id: "new" }, fakeElement);
    globalThis.Hooks.callAll("renderChatMessage", { id: "old" }, fakeElement);
    expect(seen).toEqual(["old"]);
  });

  it("binds the render hook only once across repeated init() calls", async () => {
    const m = await boot(14);
    m.cards.init(); m.cards.init(); m.cards.init();
    const seen = [];
    m.cards.onRender("probe", (c) => seen.push(c.message.id));
    globalThis.Hooks.callAll("renderChatMessageHTML", { id: "m1" }, fakeElement);
    expect(seen).toEqual(["m1"]);            // 绑了三次的话这里会是三条
  });

  it("does not throw in an environment without document (vitest runs on node)", async () => {
    const m = await boot(14);
    expect(globalThis.document).toBeUndefined();
    expect(() => m.cards.init()).not.toThrow();
  });

  it("preloads the module's own Handlebars template", async () => {
    (await boot(14)).cards.init();
    expect(globalThis.foundry.applications.handlebars.loadTemplates)
      .toHaveBeenCalledWith(["modules/alien-evolved-automation/templates/aea-card-actions.hbs"]);
  });

  it("does not throw when no template loader exists", async () => {
    const m = await boot(14, { withLoader: false });
    expect(() => m.cards.init()).not.toThrow();
  });
});

describe("cards render fan-out", () => {
  let mod, rollbus, order, mountNode;

  function fire(message) {
    globalThis.Hooks.callAll("renderChatMessageHTML", message, fakeElement);
  }

  beforeEach(async () => {
    installFoundryStub({ generation: 14 });
    globalThis.foundry.applications.handlebars = { loadTemplates: vi.fn() };
    vi.resetModules();
    mod = await import("../scripts/kernel/cards.mjs");
    rollbus = await import("../scripts/kernel/rollbus.mjs");
    rollbus.rollBus.recordOf = vi.fn(() => null);
    mod.cards.init();
    mountNode = { id: "mount" };
    mod.cards.mount = vi.fn(() => mountNode);   // 探针：只为观察「谁在什么时候要了挂点」
    order = [];
  });

  afterEach(() => uninstallFoundryStub());

  it("calls every registered handler in registration order", () => {
    mod.cards.onRender("a", () => order.push("a"));
    mod.cards.onRender("b", () => order.push("b"));
    fire({ id: "m1" });
    expect(order).toEqual(["a", "b"]);
  });

  it("lets a re-registered name replace its handler without changing its position", () => {
    mod.cards.onRender("a", () => order.push("a1"));
    mod.cards.onRender("b", () => order.push("b"));
    mod.cards.onRender("a", () => order.push("a2"));
    fire({ id: "m1" });
    expect(order).toEqual(["a2", "b"]);
  });

  it("hands each handler the message, the unwrapped element and the RollRecord", () => {
    const record = { v: 1, id: "r1", kind: "skill" };
    rollbus.rollBus.recordOf = vi.fn(() => record);
    const message = { id: "m1" };
    let seen = null;
    mod.cards.onRender("a", (c) => { seen = c; });
    fire(message);
    expect(seen.message).toBe(message);
    expect(seen.element).toBe(fakeElement);
    expect(seen.record).toBe(record);
    expect(rollbus.rollBus.recordOf).toHaveBeenCalledWith(message);
  });

  it("keeps one handler's exception from stopping the ones after it", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    mod.cards.onRender("boom", () => { throw new Error("boom"); });
    mod.cards.onRender("after", () => order.push("after"));
    expect(() => fire({ id: "m1" })).not.toThrow();
    expect(order).toEqual(["after"]);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("logs a rejected async handler instead of leaving the rejection unhandled", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    mod.cards.onRender("late", async () => { throw new Error("late"); });
    fire({ id: "m1" });
    await Promise.resolve(); await Promise.resolve();
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("never creates a mount for handlers that ignore ctx.mount", () => {
    mod.cards.onRender("a", (c) => order.push(c.message.id));
    fire({ id: "m1" });
    expect(order).toEqual(["m1"]);
    expect(mod.cards.mount).not.toHaveBeenCalled();
  });

  it("creates the mount at most once and hands every reader the same node", () => {
    const got = [];
    mod.cards.onRender("a", (c) => got.push(c.mount));
    mod.cards.onRender("b", (c) => { got.push(c.mount); got.push(c.mount); });
    fire({ id: "m1" });
    expect(mod.cards.mount).toHaveBeenCalledTimes(1);
    expect(got).toEqual([mountNode, mountNode, mountNode]);
  });

  it("does nothing at all when no handler is registered", () => {
    fire({ id: "m1" });
    expect(rollbus.rollBus.recordOf).not.toHaveBeenCalled();
    expect(mod.cards.mount).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 17: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs -t "cards.init"`
Expected: FAIL —— 六条全红，第一条报 `TypeError: m.cards.init is not a function`。

- [ ] **Step 18: 实现 `unwrap` / `mount` / 委托处理器**

在 `scripts/kernel/cards.mjs` 里、`export const cards` **之前**插入：

```js
/** 既接原生 HTMLElement，也接旧渲染钩子传来的 jQuery 对象。 */
function unwrap(element) {
  if (!element) return null;
  if (typeof element.querySelector === "function") return element;
  const first = element[0];
  if (first && typeof first.querySelector === "function") return first;
  return null;
}

/**
 * 唯一的委托处理器。委托根用 document 而不是 #chat-log：
 * ChatLog 重渲染会整体替换 #chat-log 元素，绑它就得每次重绑（alienrpg.mjs:465 的反模式），
 * 而且聊天弹出窗口里还有第二个同 id 元素。作用域靠 .aea-mount 守卫收窄。
 */
function onDelegatedClick(event) {
  const target = event?.target?.closest?.(`[${ACTION_ATTR}]`) ?? null;
  const action = target?.getAttribute?.(ACTION_ATTR) ?? null;
  const mount = target?.closest?.(`.${MOUNT_CLASS}`) ?? null;
  const plan = pureCardDispatch({ action, insideMount: !!mount, registered: actions.has(action) });
  if (!plan.dispatch) return;
  event.preventDefault?.();
  event.stopPropagation?.();
  const messageId = pureCardMessageId({
    mountMessageId: mount.dataset?.aeaMessage,
    cardMessageId: target.closest?.("[data-message-id]")?.dataset?.messageId,
  });
  const message = messageId ? (globalThis.game?.messages?.get?.(messageId) ?? null) : null;
  try {
    actions.get(action)(event, { action, message, messageId, element: target, mount });
  } catch (err) {
    console.error(`${MID} | chat action "${action}" failed`, err);
  }
}
```

并把 `mount` 加进 `export const cards`（放在 `onRender` 之后）：

```js
  /**
   * 幂等地在一条聊天消息里注入模组自有挂点。
   * 只 append 到末尾——alienrpg.mjs:468 用 ev.target.previousElementSibling.checked
   * 读多重推骰复选框，往 input.multiPush 与 button.alien-Push-button 之间插任何节点
   * 都会让系统的多重推骰静默失效。
   * @returns {HTMLElement|null} 入参不是元素、或环境里没有 document 时返回 null，调用方必须判空
   */
  mount(element, message) {
    const root = unwrap(element);
    if (!root) return null;
    const doc = root.ownerDocument ?? globalThis.document;
    if (typeof doc?.createElement !== "function") return null;
    const existing = root.querySelector(`.${MOUNT_CLASS}`);
    if (existing) {
      if (message?.id) existing.dataset.aeaMessage = message.id;
      return existing;
    }
    const node = doc.createElement("div");
    node.className = MOUNT_CLASS;
    node.dataset.aeaMessage = message?.id ?? "";
    const host = root.querySelector(".message-content") ?? root;
    host.appendChild(node);
    return node;
  },
```

- [ ] **Step 19: 实现渲染扇出 `onCardRender` 与 `init()`**

在 `scripts/kernel/cards.mjs` 顶部把 import 补成两行：

```js
import { MID } from "../const.mjs";
import { rollBus } from "./rollbus.mjs";
```

在 `onDelegatedClick` 之后、`export const cards` 之前插入：

```js
/**
 * 唯一的渲染监听器（契约 §0.2 把渲染钩子判给本文件独占）。
 * 按注册顺序把 {message, element, mount, record} 交给每条特性。
 *
 * mount 是惰性 getter：没有任何 handler 读它，就不往卡里插节点——
 * 于是「没有特性关心的消息」不会多出一个空挂点。
 * getter 里走的是 cards.mount(...) 而不是裸函数：这样测试能用探针替掉它，
 * 也方便将来在不改扇出逻辑的前提下换挂载实现。
 * handler 同步执行且不被 await：要等 3D 骰子的特性自己开异步续段
 * （await diceBarrier.awaitDice(message) 后复查 isConnected）。
 */
function onCardRender(message, html) {
  if (renderers.size === 0) return;
  const element = unwrap(html);
  if (!element) return;
  let record = null;
  try {
    record = rollBus.recordOf(message) ?? null;
  } catch (err) {
    console.error(`${MID} | recordOf failed while rendering a card`, err);
  }
  let mountEl;   // undefined = 还没有人要过挂点
  const ctx = {
    message,
    element,
    record,
    get mount() {
      if (mountEl === undefined) mountEl = cards.mount(element, message);
      return mountEl;
    },
  };
  for (const [name, handler] of renderers) {
    try {
      const out = handler(ctx);
      if (out && typeof out.then === "function") {
        out.then(null, (err) => console.error(`${MID} | render handler "${name}" rejected`, err));
      }
    } catch (err) {
      console.error(`${MID} | render handler "${name}" failed`, err);
    }
  }
}

let listening = false;
let hooksBound = false;
/** 实际绑上的钩子名，供 k8.render-fanout 自检条目核对。 */
let boundHookName = null;

/**
 * Foundry 世代号。V13 起渲染钩子改名 renderChatMessageHTML，参数从 jQuery 变成原生元素；
 * 旧名只剩带弃用警告的兼容层（系统在 alienrpg.mjs:464 绑的正是旧名）。
 * 在 V13+ 订阅旧名会刷弃用警告，所以按世代号二选一。
 */
function foundryGeneration() {
  const n = Number.parseInt(globalThis.game?.release?.generation, 10);
  return Number.isFinite(n) ? n : 13;
}

function preloadTemplate() {
  const loader = globalThis.foundry?.applications?.handlebars?.loadTemplates ?? globalThis.loadTemplates;
  if (typeof loader !== "function") return;
  try { loader([TEMPLATE_CARD_ACTIONS]); }
  catch (err) { console.error(`${MID} | template preload failed`, err); }
}
```

并把 `init` 加进 `export const cards`（放在 `mount` 之后）：

```js
  /** 由 main.mjs 在 ready 阶段调用。幂等，缺件时留待下次重试。 */
  init() {
    if (!listening && typeof globalThis.document?.addEventListener === "function") {
      globalThis.document.addEventListener("click", onDelegatedClick);
      listening = true;
    }
    if (!hooksBound && typeof globalThis.Hooks?.on === "function") {
      boundHookName = foundryGeneration() >= 13 ? "renderChatMessageHTML" : "renderChatMessage";
      globalThis.Hooks.on(boundHookName, onCardRender);
      hooksBound = true;
    }
    preloadTemplate();
  },
```

- [ ] **Step 20: Run it and watch it pass**

Run: `npx vitest run test/cards.test.mjs`
Expected: PASS —— 29 passed（15 + 6 + 8）。

- [ ] **Step 21: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/cards.mjs test/cards.test.mjs \
 && git commit -m "feat(kernel): K8 自绑渲染钩子 + 惰性挂点 + 渲染扇出 + 单一委托监听

契约 §0.2/§4 K8：renderChatMessageHTML(V13+)/renderChatMessage(旧版回退) 归 cards 自己挂，
按 game.release.generation 二选一，全模组只此一处、只绑一次。
系统在 alienrpg.mjs:464 绑的是旧名，兼容层没了它的推骰按钮就哑了，本模组不跟着死。

onRender 的回调拿 {message, element, mount, record}：record 来自 rollBus.recordOf，
mount 是惰性 getter——没人读就不插节点，于是无人关心的卡不会多出空挂点。
handler 同步执行、逐条捕获异常，异步拒绝也兜一层日志，一条特性炸不掉整张卡。

mount 幂等且只 append 到 .message-content 末尾：alienrpg.mjs:468 用
previousElementSibling.checked 读多重推骰复选框，插到按钮前面就把它废了。
点击委托挂 document，用 .aea-mount 作用域守卫。

钩子名与只绑一次靠桩的 Hooks.callAll 真派发来行为断言，不去翻桩的内部记录。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 22: Write the failing test（`render()` 的保护分支与作用域）**

契约 §4 K8 把 `render(target, templatePath, data)` 定死成「**只**替换 `target` 自身的 `innerHTML`，绝不触碰它的父节点、兄弟节点或挂点上的其它内容」。「只写 target 自身」这件事在 node 里能测：给一个带 `innerHTML` 属性的普通对象，看它有没有被赋值即可，这不是伪造 DOM 语义。真正需要真 DOM 的是「邻居节点原样保留」，那条放 `k8.render-target` 自检条目。

追加到 `test/cards.test.mjs` 末尾：

```js
describe("cards.render", () => {
  let mod, rendered;

  beforeEach(async () => {
    installFoundryStub();
    rendered = vi.fn(async () => "<p>x</p>");
    globalThis.foundry.applications.handlebars = { renderTemplate: rendered, loadTemplates: vi.fn() };
    vi.resetModules();
    mod = await import("../scripts/kernel/cards.mjs");
  });

  afterEach(() => uninstallFoundryStub());

  it("does nothing when the target is null (the null contract of mount())", async () => {
    await expect(mod.cards.render(null, "modules/x/t.hbs", {})).resolves.toBeUndefined();
    expect(rendered).not.toHaveBeenCalled();
  });

  it("does nothing when the template path is missing or not a string", async () => {
    const target = { innerHTML: "" };
    await mod.cards.render(target, "", {});
    await mod.cards.render(target, undefined, {});
    expect(rendered).not.toHaveBeenCalled();
    expect(target.innerHTML).toBe("");
  });

  it("logs instead of throwing when no renderTemplate is available", async () => {
    delete globalThis.foundry.applications.handlebars.renderTemplate;
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(mod.cards.render({ innerHTML: "" }, "modules/x/t.hbs", {})).resolves.toBeUndefined();
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("replaces only the target's own innerHTML", async () => {
    const target = { innerHTML: "<b>old</b>" };
    await mod.cards.render(target, "modules/x/t.hbs", { a: 1 });
    expect(rendered).toHaveBeenCalledWith("modules/x/t.hbs", { a: 1 });
    expect(target.innerHTML).toBe("<p>x</p>");
    expect(Object.keys(target)).toEqual(["innerHTML"]);   // 没有偷偷挂别的属性
  });

  it("leaves the target untouched when the template fails to render", async () => {
    rendered.mockRejectedValueOnce(new Error("bad template"));
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const target = { innerHTML: "<b>old</b>" };
    await mod.cards.render(target, "modules/x/t.hbs", {});
    expect(target.innerHTML).toBe("<b>old</b>");
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("warns when a caller renders straight into the shared mount", async () => {
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const target = { innerHTML: "", classList: { contains: (c) => c === "aea-mount" } };
    await mod.cards.render(target, "modules/x/t.hbs", {});
    expect(spy).toHaveBeenCalled();
    expect(target.innerHTML).toBe("<p>x</p>");
    spy.mockRestore();
  });
});
```

- [ ] **Step 23: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs -t "cards.render"`
Expected: FAIL —— 六条全红，第一条报 `TypeError: mod.cards.render is not a function`。

- [ ] **Step 24: 实现 `render()`**

把下面这个成员加进 `export const cards`（放在 `init` 之后）：

```js
  /**
   * 特性往挂点里塞内容的唯一入口（契约 §4 K8）。
   *
   * 只替换 target 自身的 innerHTML，绝不碰它的父节点、兄弟节点，
   * 也不替调用方在挂点里创建容器。一期有两条特性共用同一个 .aea-mount，
   * 所以每条特性必须自己先在挂点下建一个 aea-<feature-id> 子元素
   * （mount.querySelector 复用、没有才创建），再把那个子元素传进来。
   * 直接把 mount 传进来会清掉另一条特性的内容——那正是下面这句 warn 要抓的。
   *
   * 调用方约定：任何写判定结果的渲染都要先 await diceBarrier.awaitDice(message)，
   * 并在等待返回后复查 target.isConnected，否则开着 Dice So Nice 时
   * 成功数会先于 3D 骰子出现、或者写进一个已被重渲染丢弃的节点。
   *
   * @param {HTMLElement|null} target 特性自己的 aea-<feature-id> 子元素（可能为 null）
   * @param {string} templatePath 例如 TEMPLATE_CARD_ACTIONS
   * @param {object} data 模板上下文
   * @returns {Promise<void>}
   */
  async render(target, templatePath, data = {}) {
    if (!target || typeof templatePath !== "string" || !templatePath) return;
    if (target.classList?.contains?.(MOUNT_CLASS)) {
      console.warn(
        `${MID} | render() was aimed at the shared .${MOUNT_CLASS} itself; ` +
          "create an aea-<feature-id> child under the mount and render into that, " +
          "otherwise this wipes whatever another feature just wrote on the same card"
      );
    }
    const renderTemplate =
      globalThis.foundry?.applications?.handlebars?.renderTemplate ?? globalThis.renderTemplate;
    if (typeof renderTemplate !== "function") {
      console.error(`${MID} | no renderTemplate available; cannot render ${templatePath}`);
      return;
    }
    let html;
    try { html = await renderTemplate(templatePath, data); }
    catch (err) { console.error(`${MID} | rendering ${templatePath} failed`, err); return; }
    target.innerHTML = html;
  },
```

- [ ] **Step 25: Run it and watch it pass**

Run: `npx vitest run test/cards.test.mjs`
Expected: PASS —— 35 passed。

- [ ] **Step 26: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/cards.mjs test/cards.test.mjs \
 && git commit -m "feat(kernel): K8 render 只替换 target 自身的 innerHTML

契约 §4 K8：render(target, templatePath, data) 不碰父节点与兄弟节点，
分节的责任在特性一侧——各自在挂点下建 aea-<feature-id> 子元素再渲染，
这样一张卡上两条特性同时写才不会互相抹掉，重复渲染也天然幂等。
把 mount 本身当 target 传进来会 warn 一句：静默互相覆盖的症状取决于
onRender 的注册顺序，时灵时不灵，最难查。

target 为 null、模板路径为空、渲染器缺席、模板渲染抛错都只是安静返回或记日志，
且渲染失败时 target 原样保留，不炸卡也不留半截内容。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 27: Write the failing test（模板文件的内容契约）**

追加到 `test/cards.test.mjs` 末尾：

```js
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

describe("templates/aea-card-actions.hbs", () => {
  const path = fileURLToPath(new URL("../templates/aea-card-actions.hbs", import.meta.url));

  it("exists", () => {
    expect(existsSync(path)).toBe(true);
  });

  it("emits data-action buttons inside an .aea-actions wrapper", () => {
    const src = readFileSync(path, "utf8");
    expect(src).toContain("aea-actions");
    expect(src).toContain("{{#each actions}}");
    expect(src).toContain('data-action="{{this.name}}"');
  });

  it("routes every user-facing label through the localize helper", () => {
    const src = readFileSync(path, "utf8");
    expect(src).toContain("{{localize this.label}}");
    expect(src).not.toMatch(/>\s*[A-Z][a-z]{3,}\s+[a-z]{3,}/);   // 模板里不许有硬编码英文散文
  });
});
```

- [ ] **Step 28: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs -t "aea-card-actions"`
Expected: FAIL —— 第一条 `AssertionError: expected false to be true`（`existsSync` 为假），后两条抛 `ENOENT: no such file or directory`。

- [ ] **Step 29: 建模板文件**

新建 `templates/aea-card-actions.hbs`。上下文是 `{cardId, title, actions: [{name, label, icon, value, disabled}]}`，`title` 与 `label` 传的是 i18n **键**，由 Foundry 内建的 `{{localize}}` 在渲染时翻译。

```hbs
<div class="aea-actions" data-aea-card="{{cardId}}">
  {{#if title}}<div class="aea-actions-title">{{localize title}}</div>{{/if}}
  <div class="aea-actions-row">
    {{#each actions}}
    <button type="button"
            class="aea-action"
            data-action="{{this.name}}"
            data-aea-value="{{this.value}}"
            {{#if this.disabled}}disabled{{/if}}>
      {{#if this.icon}}<i class="{{this.icon}}"></i> {{/if}}{{localize this.label}}
    </button>
    {{/each}}
  </div>
</div>
```

- [ ] **Step 30: Run it and watch it pass**

Run: `npx vitest run test/cards.test.mjs`
Expected: PASS —— 38 passed。

- [ ] **Step 31: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add templates/aea-card-actions.hbs test/cards.test.mjs \
 && git commit -m "feat(kernel): K8 模组自有聊天卡模板

aea-card-actions.hbs 按 {cardId, title, actions[]} 渲染一排 data-action 按钮。
label 与 title 传 i18n 键、模板里走 localize，不留硬编码英文；
特性通过 cards.render(自己的 aea-<id> 子元素, TEMPLATE_CARD_ACTIONS, data) 使用它，
不要各自拼 HTML 字符串，否则两条渲染路径的标记与样式迟早分叉。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 32: Write the failing test（`init()` 登记五条自检）**

追加到 `test/cards.test.mjs` 末尾：

```js
describe("cards selftest registration", () => {
  let results;

  beforeEach(async () => {
    installFoundryStub({ generation: 14 });
    globalThis.foundry.applications.handlebars = { loadTemplates: vi.fn() };
    vi.resetModules();
    const mod = await import("../scripts/kernel/cards.mjs");
    const { selftest } = await import("../scripts/kernel/selftest.mjs");
    mod.cards.init();
    mod.cards.init();                       // 幂等：不许登记两遍
    results = (await selftest.runAll()).filter((r) => r.id.startsWith("k8."));
  });

  afterEach(() => uninstallFoundryStub());

  it("registers the five k8 entries exactly once", () => {
    expect(results.map((r) => r.id).sort()).toEqual([
      "k8.delegated-click", "k8.lazy-mount", "k8.mount-idempotent",
      "k8.render-fanout", "k8.render-target",
    ]);
  });

  it("reports honestly instead of pretending to pass outside Foundry", () => {
    expect(results.every((r) => r.ok === false)).toBe(true);
    expect(results.every((r) => r.detail.includes("requires Foundry"))).toBe(true);
  });
});
```

- [ ] **Step 33: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs -t "selftest registration"`
Expected: FAIL —— 第一条报 `AssertionError: expected [] to deeply equal [ 'k8.delegated-click', 'k8.lazy-mount', 'k8.mount-idempotent', 'k8.render-fanout', 'k8.render-target' ]`。

- [ ] **Step 34: 实现五条自检条目（在真 Foundry 里跑的真 DOM 检查）**

在 `scripts/kernel/cards.mjs` 顶部再加一行 import：

```js
import { selftest } from "./selftest.mjs";
```

在 `preloadTemplate` 之后、`export const cards` 之前插入。注意 `label` 传的是 **i18n 键本身**，不在这里 `localize`：自检条目由运行器负责本地化，而登记时刻语言包不一定加载好，就地 localize 只会把键名冻死成字面量。

```js
let selftestsRegistered = false;

/** 真 DOM 判据。node 环境里这五条条目一律如实报 ok:false，不假装通过。 */
function hasLiveDom() {
  return typeof globalThis.document?.createElement === "function"
    && typeof globalThis.document?.body?.appendChild === "function"
    && typeof globalThis.MouseEvent === "function";
}
const NOT_IN_FOUNDRY = { ok: false, detail: "requires Foundry (no live document in this environment)" };

/** 复刻 YZEDiceRoller.mjs:378-392 那个分支产出的真实结构，用的是真 document。 */
function selftestFixture(doc, messageId) {
  const li = doc.createElement("li");
  li.className = "chat-message";
  li.dataset.messageId = messageId;
  const content = doc.createElement("div");
  content.className = "message-content";
  li.appendChild(content);
  content.insertAdjacentHTML(
    "beforeend",
    '<span>MultiPush</span> <input class="multiPush" type="checkbox"/> ' +
      '<button class="alien-Push-button">PUSH</button>' +
      '<span class="dmgBtn-container"></span>'
  );
  return li;
}

function registerSelftests() {
  if (selftestsRegistered) return;
  selftestsRegistered = true;

  selftest.register({
    id: "k8.mount-idempotent",
    label: "AEA.selftest.k8.mountIdempotent",
    run() {
      if (!hasLiveDom()) return NOT_IN_FOUNDRY;
      const doc = globalThis.document;
      const li = selftestFixture(doc, "aea-selftest-mount");
      const a = cards.mount(li, { id: "aea-selftest-mount" });
      const b = cards.mount(li, { id: "aea-selftest-mount" });
      const count = li.querySelectorAll(`.${MOUNT_CLASS}`).length;
      const push = li.querySelector("button.alien-Push-button");
      const sibOk = push?.previousElementSibling?.classList?.contains("multiPush") === true;
      const lastOk = li.querySelector(".message-content").lastElementChild === a;
      return {
        ok: a === b && count === 1 && sibOk && lastOk,
        detail: `same=${a === b} mounts=${count} multiPushStillPrecedesPush=${sibOk} appendedLast=${lastOk}`,
      };
    },
  });

  selftest.register({
    id: "k8.delegated-click",
    label: "AEA.selftest.k8.delegatedClick",
    async run() {
      if (!hasLiveDom()) return NOT_IN_FOUNDRY;
      const doc = globalThis.document;
      const li = selftestFixture(doc, "aea-selftest-click");
      doc.body.appendChild(li);
      let seen = null;
      cards.registerAction("aea-selftest-click", (ev, c) => { seen = c; });
      const mount = cards.mount(li, { id: "aea-selftest-click" });
      mount.innerHTML = '<button type="button" data-action="aea-selftest-click">x</button>';
      mount.querySelector("button").dispatchEvent(new globalThis.MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
      li.remove();
      actions.delete("aea-selftest-click");
      const ok = !!seen && seen.action === "aea-selftest-click"
        && seen.messageId === "aea-selftest-click" && seen.mount === mount;
      return {
        ok,
        detail: seen
          ? `action=${seen.action} messageId=${seen.messageId} mountMatched=${seen.mount === mount}`
          : "handler never fired",
      };
    },
  });

  selftest.register({
    id: "k8.render-target",
    label: "AEA.selftest.k8.renderTarget",
    async run() {
      if (!hasLiveDom()) return NOT_IN_FOUNDRY;
      const doc = globalThis.document;
      const mount = doc.createElement("div");
      mount.className = MOUNT_CLASS;
      const mine = doc.createElement("div");
      mine.className = "aea-selftest-target";
      const neighbour = doc.createElement("div");
      neighbour.className = "aea-selftest-neighbour";
      neighbour.innerHTML = "<em>keep me</em>";
      mount.appendChild(mine);
      mount.appendChild(neighbour);
      const data = { cardId: "aea-selftest", title: "", actions: [
        { name: "aea-selftest-noop", label: "ALIENRPG.Push", icon: "", value: "", disabled: false },
      ] };
      await cards.render(mine, TEMPLATE_CARD_ACTIONS, data);
      await cards.render(mine, TEMPLATE_CARD_ACTIONS, data);
      const buttons = mine.querySelectorAll('button[data-action="aea-selftest-noop"]').length;
      const neighbourKept = neighbour.innerHTML === "<em>keep me</em>";
      const children = mount.children.length;
      return {
        ok: buttons === 1 && neighbourKept && children === 2,
        detail: `buttonsAfterTwoRenders=${buttons} neighbourUntouched=${neighbourKept} mountChildren=${children}`,
      };
    },
  });

  selftest.register({
    id: "k8.render-fanout",
    label: "AEA.selftest.k8.renderFanout",
    run() {
      if (!hasLiveDom()) return NOT_IN_FOUNDRY;
      const expected = foundryGeneration() >= 13 ? "renderChatMessageHTML" : "renderChatMessage";
      const doc = globalThis.document;
      const li = selftestFixture(doc, "aea-selftest-fanout");
      doc.body.appendChild(li);
      const seen = [];
      // 直接调本模块的扇出函数，而不是 Hooks.callAll：后者会把一条伪造的消息
      // 派给系统和 dice-so-nice 的监听器。钩子名另外核对 boundHookName。
      cards.onRender("aea-selftest-fanout-a", (c) => seen.push({ n: "a", id: c.message?.id, el: c.element === li }));
      cards.onRender("aea-selftest-fanout-b", (c) => seen.push({ n: "b", mounted: !!c.mount }));
      onCardRender({ id: "aea-selftest-fanout" }, li);
      renderers.delete("aea-selftest-fanout-a");
      renderers.delete("aea-selftest-fanout-b");
      li.remove();
      const ok = boundHookName === expected && seen.length === 2
        && seen[0].n === "a" && seen[0].id === "aea-selftest-fanout" && seen[0].el === true
        && seen[1].mounted === true;
      return {
        ok,
        detail: `hook=${boundHookName} expected=${expected} handlers=${seen.length}`
          + ` order=${seen.map((s) => s.n).join(",")} elementMatched=${seen[0]?.el} mountCreated=${seen[1]?.mounted}`,
      };
    },
  });

  selftest.register({
    id: "k8.lazy-mount",
    label: "AEA.selftest.k8.lazyMount",
    run() {
      if (!hasLiveDom()) return NOT_IN_FOUNDRY;
      const doc = globalThis.document;
      const li = selftestFixture(doc, "aea-selftest-lazy");
      doc.body.appendChild(li);
      let ran = false;
      cards.onRender("aea-selftest-lazy", () => { ran = true; });   // 故意不碰 ctx.mount
      onCardRender({ id: "aea-selftest-lazy" }, li);
      renderers.delete("aea-selftest-lazy");
      const injected = li.querySelectorAll(`.${MOUNT_CLASS}`).length;
      li.remove();
      // 诊断信息：空挂点在「已注入、判定结果还在等骰子」的窗口里是合法的，
      // 所以它不进 ok，只报给 GM 看趋势。
      const empties = doc.querySelectorAll(`.${MOUNT_CLASS}:empty`).length;
      return {
        ok: ran && injected === 0,
        detail: `handlerRan=${ran} mountsInFixture=${injected} emptyMountsInDocument=${empties}`,
      };
    },
  });
}
```

最后把 `init()` 末尾的一行改成两行：

```js
    preloadTemplate();
    registerSelftests();
```

- [ ] **Step 35: Run it and watch it pass**

Run: `npx vitest run test/cards.test.mjs`
Expected: PASS —— 40 passed。

- [ ] **Step 36: Write the failing test（语言包与样式表的内容契约）**

五条自检的 `label` 传的是 i18n 键，键在两份语言包里都得有，否则 GM 在自检面板里看到的是一串 `AEA.selftest.k8.*`。样式那条则是防「以后有人顺手删掉挂点样式」：空挂点不隐藏的话，每条聊天消息底下会闪一条空白。

追加到 `test/cards.test.mjs` 末尾：

```js
describe("K8 static assets", () => {
  const langPath = (lang) => fileURLToPath(new URL(`../lang/${lang}.json`, import.meta.url));
  const K8_KEYS = ["mountIdempotent", "delegatedClick", "renderTarget", "renderFanout", "lazyMount"];

  it("ships all five k8 selftest labels in en.json", () => {
    const en = JSON.parse(readFileSync(langPath("en"), "utf8"));
    expect(Object.keys(en.AEA.selftest.k8).sort()).toEqual([...K8_KEYS].sort());
    expect(Object.values(en.AEA.selftest.k8).every((v) => typeof v === "string" && v.length > 0)).toBe(true);
  });

  it("mirrors exactly the same key set in cn.json", () => {
    const en = JSON.parse(readFileSync(langPath("en"), "utf8"));
    const cn = JSON.parse(readFileSync(langPath("cn"), "utf8"));
    expect(Object.keys(cn.AEA.selftest.k8).sort()).toEqual(Object.keys(en.AEA.selftest.k8).sort());
    expect(Object.values(cn.AEA.selftest.k8).every((v) => typeof v === "string" && v.length > 0)).toBe(true);
  });

  it("hides an empty mount so a card waiting on 3D dice shows no blank strip", () => {
    const css = readFileSync(fileURLToPath(new URL("../styles/alien-evolved-automation.css", import.meta.url)), "utf8");
    expect(css).toContain(".aea-mount:empty");
    expect(css).toContain("button.aea-action");
  });
});
```

- [ ] **Step 37: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs -t "static assets"`
Expected: FAIL —— 前两条报 `TypeError: Cannot read properties of undefined (reading 'k8')`（`AEA.selftest` 下还没有 `k8`），第三条报 `AssertionError: expected '…' to contain '.aea-mount:empty'`。

- [ ] **Step 38: 补 i18n 键**

把下面的键**合并**进 `lang/en.json` 已有的顶层 `AEA` 对象（语言包是嵌套结构，顶层只有 `AEA` 一个键；别新造第二个顶层键，也别把已有的 `AEA.selftest` 整段覆盖掉）：

```json
{
  "AEA": {
    "selftest": {
      "k8": {
        "mountIdempotent": "K8 card mount is idempotent and does not break the system push button",
        "delegatedClick": "K8 delegated click reaches the registered action handler",
        "renderTarget": "K8 render replaces only its own target and leaves neighbours alone",
        "renderFanout": "K8 hands the live card element and roll record to every render handler",
        "lazyMount": "K8 injects no mount into cards no feature cares about"
      }
    }
  }
}
```

同样的键合并进 `lang/cn.json`：

```json
{
  "AEA": {
    "selftest": {
      "k8": {
        "mountIdempotent": "K8 卡片挂点幂等，且不会打断系统的推骰按钮",
        "delegatedClick": "K8 委托点击能送达已注册的动作处理器",
        "renderTarget": "K8 渲染只替换自己的目标元素，不动兄弟节点",
        "renderFanout": "K8 把真实卡片元素与掷骰记录交给每个渲染回调",
        "lazyMount": "K8 不给任何特性都不关心的卡注入挂点"
      }
    }
  }
}
```

- [ ] **Step 39: 补样式**

把下面这段**追加**到 `styles/alien-evolved-automation.css`（该文件已由清单任务建好并写进 `module.json` 的 `styles`）。挂点在「已注入、判定结果还在等 3D 骰子」的窗口里是空的，让它不占位，免得聊天消息底下闪一条空白；特性自己那层 `aea-<feature-id>` 子元素同理。

```css
/* K8 · 模组自有聊天卡挂点 ------------------------------------------------ */
.aea-mount:empty { display: none; }

/* 每条特性在挂点下有自己的 aea-<feature-id> 子元素；空的同样不占位。 */
.aea-mount > div:empty { display: none; }
.aea-mount > div + div { margin-top: 4px; }

.aea-actions { margin-top: 4px; }

.aea-actions-title {
  font-weight: bold;
  opacity: 0.85;
  margin-bottom: 2px;
}

.aea-actions-row {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

button.aea-action {
  flex: 0 1 auto;
  width: auto;
  line-height: 1.6;
  padding: 0 6px;
}

button.aea-action[disabled] { opacity: 0.5; cursor: not-allowed; }
```

- [ ] **Step 40: Run it and watch it pass**

Run: `npx vitest run test/cards.test.mjs`
Expected: PASS —— 43 passed。

- [ ] **Step 41: Write the failing test（`main.mjs` 的三处接线）**

`main.mjs` 是模组生命周期钩子的唯一挂载处，由骨架任务写好，交付时带十个逐字锚点注释，`export const api` 的八个键值全为 `null`。本任务是 `cards` 那个槽的属主，要做的正好三件事：加 import、填自己的 `api` 槽、在自己的 ready 子锚点后插调用。这三件事各写一条断言。

顺序上：`ready` 段的四个子锚点在骨架里就是 `ready.registry` → `ready.patches` → `ready.rollbus` → `ready.cards` 的固定次序，各自的属主只往自己的锚点后插一行，所以「cards.init() 排在 rollBus.install() 之后、排在特性安装循环之前」是锚点顺序自然给的，不需要谁去猜谁先谁后。断言直接钉在锚点位置上。

追加到 `test/cards.test.mjs` 末尾：

```js
describe("scripts/main.mjs wiring", () => {
  const mainPath = fileURLToPath(new URL("../scripts/main.mjs", import.meta.url));
  const read = () => readFileSync(mainPath, "utf8");

  it("imports cards from the kernel", () => {
    expect(read()).toContain('import { cards } from "./kernel/cards.mjs";');
  });

  it("fills its own api slot without adding or removing keys", () => {
    const src = read();
    const start = src.indexOf("export const api");
    const literal = src.slice(start, src.indexOf("};", start));
    expect(literal).not.toContain("cards: null");
    expect(literal).toMatch(/[\s{,]cards,/);
    for (const key of ["features", "patches", "resolver", "registry", "rollBus", "diceBarrier", "selftest"]) {
      expect(literal).toContain(key);          // 别的槽一个都不许少
    }
  });

  it("calls cards.init() exactly once, right after the ready.cards anchor", () => {
    const src = read();
    const anchor = src.indexOf("/* AEA-ANCHOR: ready.cards */");
    expect(anchor).toBeGreaterThan(-1);
    expect(src.match(/cards\.init\(\)/g)).toHaveLength(1);
    expect(src.indexOf("cards.init();")).toBeGreaterThan(anchor);
    // 骨架的锚点次序保证 rollbus 在前、特性安装循环在后
    expect(src.indexOf("/* AEA-ANCHOR: ready.rollbus */")).toBeLessThan(anchor);
    const loop = [...src.matchAll(/for\s*\(\s*const\s+f\s+of\s+FEATURES\s*\)[^\n]*f\.install\(\)/g)];
    expect(loop).toHaveLength(1);
    expect(src.indexOf("cards.init();")).toBeLessThan(loop[0].index);
  });
});
```

- [ ] **Step 42: Run it and watch it fail**

Run: `npx vitest run test/cards.test.mjs -t "main.mjs wiring"`
Expected: FAIL —— 第一条报 `AssertionError: expected '…' to contain 'import { cards } from "./kernel/cards.mjs";'`；第二条报 `expected '…' not to contain 'cards: null'`；第三条报 `expected null to have length 1`（`src.match(...)` 为 `null`）。

- [ ] **Step 43: 往 `main.mjs` 做那三处接线**

三处编辑，**一处不多**：不动别的锚点、不替换 `api` 这个对象、不增删任何键、不碰另外七个槽（它们各有属主）。全部按锚点文本定位，**不用行号**。

**(1)** 找到这行注释

```
/* AEA-ANCHOR: imports */
```

在它的**下一行**插入：

```js
import { cards } from "./kernel/cards.mjs";
```

**(2)** 在 `export const api = { ... }` 这个对象字面量里，把

```js
  rollBus: null, diceBarrier: null, cards: null, selftest: null,
```

改成

```js
  rollBus: null, diceBarrier: null, cards, selftest: null,
```

（只动 `cards: null` 这四个字符所在的位置；`rollBus`、`diceBarrier`、`selftest` 的 `null` 由它们各自的属主任务替换，本任务碰它们就会把别人的装配覆盖掉。）

**(3)** 找到 `ready` 生命周期钩子体里的这行注释

```
  /* AEA-ANCHOR: ready.cards */
```

在它的**下一行**插入（缩进与锚点注释对齐）：

```js
  cards.init();
```

插入后 ready 段应当读作（省略号处是别的任务往自己锚点后插的行，与本任务无关）：

```js
Hooks.once("ready", async () => {
  await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 });
  /* AEA-ANCHOR: ready.registry */
  ...
  /* AEA-ANCHOR: ready.patches */
  ...
  /* AEA-ANCHOR: ready.rollbus */
  ...
  /* AEA-ANCHOR: ready.cards */
  cards.init();
  for (const f of FEATURES) await safely(`feature ${f.id} install`, () => f.install());
  for (const r of REPAIRS)  await safely(`repair ${r.id} install`,  () => r.install?.());
});
```

这个位置有两个必须成立的理由：`cards.init()` 要排在特性安装循环**之前**（特性的 `install()` 里会调 `cards.onRender()` / `cards.registerAction()`，而渲染钩子与委托监听是在 `init()` 里挂的）；也要排在 `rollBus.install()` **之后**（扇出要交给特性的 `record` 来自 rollBus）。

- [ ] **Step 44: Run the whole suite**

Run: `npm test`
Expected: PASS —— `test/cards.test.mjs` 46 passed；整仓其余测试文件保持全绿。

- [ ] **Step 45: Commit**

```bash
cd "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation" \
 && git add scripts/kernel/cards.mjs scripts/main.mjs test/cards.test.mjs \
       styles/alien-evolved-automation.css lang/en.json lang/cn.json \
 && git commit -m "feat(kernel): K8 五条真 DOM 自检 + 挂点样式 + ready 接线

node 环境没有 document，手搓假 DOM 只能证明作者与自己一致，所以
挂载幂等、事件委托、render 作用域、渲染扇出、惰性挂点这五件事写成 selftest 条目，
在真 Foundry 里跑真 DOM；跑在 node 里时如实报 requires Foundry，不假装通过。
label 存 i18n 键而不是登记时就 localize：登记时刻语言包未必加载好，
就地 localize 只会把键名冻成字面量。
k8.mount-idempotent 顺带断言 alienrpg.mjs:468 依赖的
button.alien-Push-button.previousElementSibling 仍然是 input.multiPush。

main.mjs 三处接线：imports 锚点后加 import、api 的 cards 槽由 null 换成 cards、
ready.cards 子锚点后插 cards.init()。子锚点的固定次序保证它排在
rollBus.install() 之后、特性安装循环之前——特性的 install() 会调
onRender/registerAction，而钩子与委托监听是在 init() 里挂的。

样式让空挂点不占位：判定结果等 3D 骰子的那段窗口里挂点本来就是空的。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 46: MANUAL VERIFICATION（自检覆盖不到的部分：真实卡片种类与人眼判断）**

自检条目验的是合成出来的卡片结构；真实卡片有几种、重渲染会不会翻倍、系统的多重推骰还灵不灵，必须在真世界里看。前置：世界启用 `alienrpg` 系统、本模组已正常启动，场上有一个 `character` actor 和一个 NPC/creature actor。

浏览器控制台（F12）：

```js
const api = game.modules.get("alien-evolved-automation").api;
api.cards;                                                           // 必须是对象，不能是 null
(await api.selftest.runAll()).filter(r => r.id.startsWith("k8."));   // 五条必须全绿

// 先掷骰，此时还没有任何特性注册 onRender
// 1) 用 character 掷一次普通属性骰（走 YZEDiceRoller.mjs:378 分支，有 dmgBtn-container）
// 2) 用 NPC/creature 掷一次骰（走别的分支，没有 dmgBtn-container）
// 3) 在第 1 张卡上点 PUSH，产生一张推骰卡（reRoll === "push"，同样没有）

document.querySelectorAll("#chat-log .aea-mount").length;            // A：现在必须是 0
document.querySelectorAll("#chat-log .dmgBtn-container").length;     // B

// 再注册两个临时回调，然后重渲染聊天日志
api.cards.onRender("aea-demo-render", (ctx) => {
  console.log("RENDER", ctx.message.id, "record=", ctx.record?.kind ?? null);
  const mount = ctx.mount;
  let section = mount.querySelector(".aea-demo-render");
  if (!section) {
    section = document.createElement("div");
    section.className = "aea-demo-render";
    mount.appendChild(section);
  }
  section.innerHTML = '<button type="button" data-action="aea-demo">demo</button>';
});
api.cards.registerAction("aea-demo", (ev, ctx) =>
  console.log("ACTION FIRED", ctx.action, ctx.messageId, ctx.message?.id, ctx.mount));
await ui.chat.render(true);

const C = document.querySelectorAll("#chat-log .aea-mount").length;  // C
await ui.chat.render(true);
console.log(C, document.querySelectorAll("#chat-log .aea-mount").length);   // 两数必须相等

document.querySelector("#chat-log .aea-mount button[data-action='aea-demo']").click();
```

必须逐条确认：

1. `runAll()` 里五条 `k8.*` 都 `ok: true`；若 `k8.render-fanout` 的 `detail` 里 `hook` 与 `expected` 不一致，说明本机 Foundry 世代号判定错了，回去看 `foundryGeneration()`。
2. **注册 `onRender` 之前 A 必须是 0** —— 这是「惰性挂点」的现场证据：没有特性关心的卡不会多出空挂点。
3. 注册之后 C 等于聊天消息条数，且 **B 严格小于 C** —— 这是「系统挂点在推骰卡与怪物卡上不存在」的现场证据；同时控制台对每条消息各打一行 `RENDER`。
4. 第二次 `ui.chat.render(true)` 前后 `.aea-mount` 数量**相等**。若翻倍，回去查 `mount()` 里的 `root.querySelector(".aea-mount")`。
5. 控制台打出 `ACTION FIRED aea-demo <id> <id> <div.aea-mount>`，两个 id 相同且都不是 `null`。
6. **最关键**：在第 1 步那张 character 卡上**勾上「多重推骰」复选框再点 PUSH**，确认系统的多重推骰仍然生效；并跑
   ```js
   const btn = document.querySelector("#chat-log button.alien-Push-button");
   console.log(btn.previousElementSibling.className);   // 必须是 "multiPush"，绝不能是 "aea-mount"
   ```
7. 再验一次「render 只动自己的 target」：
   ```js
   const mount = document.querySelector("#chat-log .aea-mount");
   const other = document.createElement("div");
   other.className = "aea-other-feature";
   other.innerHTML = "<em>keep me</em>";
   mount.appendChild(other);
   await api.cards.render(mount.querySelector(".aea-demo-render"),
     "modules/alien-evolved-automation/templates/aea-card-actions.hbs",
     { cardId: "x", title: "", actions: [{ name: "aea-demo", label: "ALIENRPG.Push", icon: "", value: "", disabled: false }] });
   console.log(other.innerHTML);        // 必须仍然是 "<em>keep me</em>"
   ```
   若它被清空，说明 `render()` 动了 target 以外的节点，回去看 Step 24。
8. 收工后刷新页面，清掉 `aea-demo-render` 与 `aea-demo` 这两个临时注册（它们只活在内存里，刷新即消失）。
