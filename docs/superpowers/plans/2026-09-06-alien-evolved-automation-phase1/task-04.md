> 本文件是实施计划 `2026-09-06-alien-evolved-automation-phase1.md` 的第 4 / 18 个任务。
> 执行前必读：同目录的 `2026-09-06-alien-evolved-automation-phase1.md`（全局约束与合稿裁决）与 `2026-09-06-alien-evolved-automation-phase1-contract.md`（接口契约 v3.2，签名不得改动）。

# Task 4: K4 Resolver —— uuid 优先的 actor/token 解析

仓库根目录：`C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/alien-evolved-automation`。下面所有路径都相对它。被引用的游戏系统在 `C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg`，版本 4.1.13（`system.json:21`，已重新打开核对）。

**Files:**
- Create: `scripts/kernel/resolver.mjs`
- Create: `test/resolver.test.mjs`
- Create: `test/resolver-foundry.test.mjs`
- Modify: `scripts/main.mjs`——**三处，一处不多**：(a) 在 `/* AEA-ANCHOR: imports */` 之后加 import 行（Step 21）；(b) 把 `api` 字面量里 `resolver: null,` 原地换成 `resolver,`（Step 22）；(c) 在 `/* AEA-ANCHOR: init */` 之后插入一个 `selftest.register({...})` 块（Step 23）。**四个 ready 子锚点、`i18nInit`、`diceSoNiceReady`、`FEATURES` / `REPAIRS` 两个数组一概不动。**
- Modify: `lang/en.json`、`lang/cn.json`（各加一个键 `AEA.selftest.resolverTokenIdentity`）

**Interfaces:**

- Consumes:
  - `scripts/const.mjs :: MID`（值 `"alien-evolved-automation"`）
  - `scripts/main.mjs` 的骨架——它由建立 main.mjs 的那个任务一次写出，含**十个**逐字锚点。本任务只碰其中两个（`imports`、`init`）与 `api` 字面量里自己那一个槽：

    ```js
    import { MID } from "./const.mjs";
    /* AEA-ANCHOR: imports */

    export const api = {
      features: null, patches: null, resolver: null, registry: null,
      rollBus: null, diceBarrier: null, cards: null, selftest: null,
    };

    const FEATURES = [
      /* AEA-ANCHOR: features */
    ];
    const REPAIRS = [
      /* AEA-ANCHOR: repairs */
    ];

    Hooks.once("init", () => {
      for (const f of FEATURES) safely(`feature ${f.id} register`, () => f.register());
      for (const r of REPAIRS)  safely(`repair ${r.id} register`,  () => r.register());
      /* AEA-ANCHOR: init */
      publishApi();
    });
    Hooks.once("i18nInit",        () => { /* AEA-ANCHOR: i18nInit */ });
    Hooks.once("diceSoNiceReady", () => { /* AEA-ANCHOR: diceSoNiceReady */ });
    Hooks.once("ready", async () => {
      await waitForWorldSettled({ timeoutMs: 10000, pollMs: 100 });
      /* AEA-ANCHOR: ready.registry */
      /* AEA-ANCHOR: ready.patches */
      /* AEA-ANCHOR: ready.rollbus */
      /* AEA-ANCHOR: ready.cards */
      for (const f of FEATURES) await safely(`feature ${f.id} install`, () => f.install());
      for (const r of REPAIRS)  await safely(`repair ${r.id} install`,  () => r.install?.());
    });
    ```

    定位一律**按锚点文本**，不用行号。`api` 交付时八个槽全是 `null`（骨架写出来的那一刻内核文件还不存在，顶部直接 import 会让整个模组加载失败）；每个内核模块的属主任务负责把自己那一个 `null` 换成自己的 import。**resolver 这个槽的属主就是本任务。** `publishApi()` 把 `api` 挂到 `game.modules.get("alien-evolved-automation").api` 上，手工验收一律写成这个长表达式。

    本任务**不往任何 ready 子锚点插东西**：resolver 是被动查询模块，没有 `install()`、没有 ready 阶段调用，也不挂任何钩子（生命周期钩子只属于 `main.mjs`，领域钩子属于拥有它的内核模块，resolver 两者都不占）。本任务也**不包裹** `yzeRoll` / `abilityRoll` / `itemRoll` / `pushRoll` 中的任何一个——那四个目标由 rollBus 独占，别人要介入只能走 `rollBus.addStage()`；resolver 根本不需要介入它们。
  - `scripts/kernel/selftest.mjs :: selftest.register({id, label, run})`——`label` 收的是 **i18n 键**（形如 `AEA.selftest.<id>`）而不是已本地化文本：登记发生在 `init`，而 Foundry 的 `i18nInit` 晚于 `init`，登记时刻语言包还没加载，`localize()` 只会把键名原样回声。`register` 原样保存整个 def，`runAll()` / `results()` 返回 `[{id, label, ok, detail}]` 时由运行器调 `game.i18n.localize(def.label)` 本地化。`run()` 返回 `{ok: boolean, detail: string}`，可 async。
  - `test/stubs/foundry.mjs :: installFoundryStub(options = {}) -> ctx` / `uninstallFoundryStub()`——**全模组唯一允许的 Foundry 替身**，由建立它的任务独占实现，其余任务**只读不改**。它往 `globalThis` 装 `game` / `ui` / `Hooks` / `ChatMessage` / `Roll` / `CONFIG` / `CONST` / `libWrapper` / `logger` / `foundry.{utils, applications}`。本文件依赖它的三条行为条款（契约 §0.3 逐字规定）：
    - **幂等**：`installFoundryStub()` 可重复调用，`uninstallFoundryStub()` 未安装时是安全空操作。
    - **文档可解析**：`ctx.documents` 是一个 `Map`（uuid → 文档），它就是 `fromUuidSync` / `fromUuid` 背后的后备存储；未命中返回 `null`。测试**往这个 Map 里 `set()` 是正规用法**，不算造全局。
    - **世界视图**：`installFoundryStub({ world })` 接受 `{uuid: doc}` 映射，据此填出 `game.actors` / `game.scenes` 等集合（各带 `get(id)` / `getName(name)` / `contents`）。
    - **禁止**在测试文件里自建 `globalThis.game` / `globalThis.Hooks` / `globalThis.canvas`，也禁止替换桩已定行为的成员（`Hooks`、`game.settings`、`libWrapper`、`ui.notifications`、`ChatMessage`）。本文件用一条 `stub prerequisites` 用例把「桩是否达标」变成一句人话失败，达标不了就报给桩的属主，**不要在本文件里就地补桩**。
  - `npm test` = `vitest run`；单文件 `npx vitest run test/<file>`；单条 `npx vitest run test/<file> -t "<name>"`。

- Produces:
  - `export function purePickRef(speaker, lookups)` → `{actorUuid: string|null, tokenUuid: string|null, source: "token"|"message"|"actor"|"none"}`
  - `export const resolver = { fromSpeaker(speaker), refs(actor, token), actorOf(record), soleToken(actor), actorById(id, {warn = true}) }`——**五个成员，不得增删**。其中 `soleToken(actor)` 返回 `Token|null`：`actor.getActiveTokens()` 恰好一个时返回那一个**已放置的 Token 对象**（带 `.document`），否则 `null`，绝不猜。掷骰总线（K1 rollBus）在包裹 `Item#roll` 时按 `ctx.token = this.actor?.token ?? resolver.soleToken(this.actor)` 使用它，所以返回的必须是 `getActiveTokens()` 原样交出来的那个对象，不要在这里替换成 TokenDocument。
  - `test/resolver.test.mjs`（9 条，纯函数真单测，零桩）与 `test/resolver-foundry.test.mjs`（25 条，全部跑在 `installFoundryStub()` 之上）
  - 自检条目 `resolver-token-identity`（在 `main.mjs` 的 `init` 锚点后登记）
  - i18n 键 `AEA.selftest.resolverTokenIdentity`（en + cn）
  - 往 `scripts/main.mjs` 插入的三处：两行 import、`api.resolver` 槽、`init` 锚点后的 `selftest.register({...})` 块

- vitest 证明不了的断言 → 承接方（这张表让掉了的验收看得见）：

  | vitest 无法证明的东西 | 承接方 |
  |---|---|
  | 真实 uuid 的形状（`Scene.x.Token.y.Actor.z` 由 Foundry 核心生成，桩只能复述作者的假设） | 自检 `resolver-token-identity` + MANUAL VERIFICATION 步骤 3、4 |
  | `canvas.tokens.get()` 这条优先分支（桩不保证提供 `canvas`，测试里走的是 `game.scenes` 回退） | MANUAL VERIFICATION 步骤 5 |
  | 系统真的把 `speaker.token` 丢掉了、事后路径确实退化并打警告 | MANUAL VERIFICATION 步骤 6、7 |
  | 载具乘员槽里真的只存裸 actor id，且收窄逻辑对真文档成立 | MANUAL VERIFICATION 步骤 8、9 |

---

**背景：这个任务在防什么（写给没用过 Foundry 的人）**

Foundry 里有两层文档。**Actor** 是角色卡本身，住在侧边栏的 Actors 目录里。**Token** 是摆在场景地图上的那个小图标。一个 Token 有个布尔开关 `actorLink`：

- `actorLink = true`（**linked / 链接**）：Token 只是基础 Actor 的一个视图。改 Token 的血量就是改那张角色卡。
- `actorLink = false`（**unlinked / 非链接**）：Token 携带自己的一份完整 actor 数据副本，Foundry 管它叫 **synthetic actor（合成 actor）**。从同一张基础卡拖出三个 Token，就得到三个各自独立的合成 actor，它们互不影响。

`alienrpg` 系统对怪物强制走非链接这条路，有两处（都已重新打开核对，系统 4.1.13）：

- `module/documents/actor.mjs:80` 把 `"prototypeToken.actorLink"` 默认设为 `true`，但 `:92` 的 `case "creature":` 分支在 `:93` 把它覆盖成 `false`，同时把 disposition 设为 `HOSTILE`。
- `module/alienrpg.mjs:366-376` —— `preCreateToken` 钩子（"Hook" 是 Foundry 的全局事件总线，`Hooks.on(name, fn)` 注册回调，系统和模组都往同一批事件名上挂函数）：只要 `aTarget.system.header.npc` 为真且类型不是 `spacecraft`，就 `foundry.utils.mergeObject(createChanges, { disposition: HOSTILE, actorLink: false })` 并 `document.updateSource(createChanges)`。

所以战场上三只 Drone = 三个各自独立的合成 actor，共享一张基础卡。

**失效点。** 系统建聊天卡时是这么写发言者的（`module/helpers/YZEDiceRoller.mjs:397-401`，逐行确认过）：

```js
		const chatData = {
			user: game.user.id,
			speaker: ChatMessage.getSpeaker({
				actor: actorid,          // ← 只有 actor id，token 被整个丢掉
			}),
```

`ChatMessage.getSpeaker({actor})` 只填 `speaker.actor`，`speaker.token` 与 `speaker.scene` 留空。于是后续任何一句 `game.actors.get(speaker.actor)` 拿到的都是**基础 Actor**。今天一期不写 actor 数据所以看不出来，但二期一按「扣血」，三只 Drone 会一起掉同样的血、一起上同样的状态、一起加压力，**静默无报错**——GM 只会觉得规则算错了，永远查不到原因。

**uuid 是什么。** uuid 是 Foundry 给每个文档的全局唯一地址字符串，`fromUuidSync(uuid)` 能同步换回文档对象。基础 actor 是 `Actor.abc123`；合成 actor 是 `Scene.s1.Token.t7.Actor.abc123`——**注意后者把 token 编进了地址**，所以三只 Drone 的 uuid 互不相同，而它们的 `actor.id` 完全一样。这就是为什么 RollRecord 里必须存 uuid 而不是 id：存 id 等于把三只 Drone 折叠成一只，schema 定错，后面全部返工。

**五个方法各自的用场（契约 §4 K4 定死的五个签名，不得增删）**

| 方法 | 谁调、什么时候调 |
|---|---|
| `refs(actor, token)` | **主路径，也是 RollRecord 里 `actorUuid`/`tokenUuid` 的唯一产出口。** 掷骰包装器在 `abilityRoll` / `Item#roll` / `pushRoll` 三处入参里当场就拿到了施动的 actor 与 token，直接喂给它。**调用方不得自己拼 uuid 字符串。** |
| `fromSpeaker(speaker)` | **事后路径**。只有一张已经建好的聊天卡时才用（例如二期从旧卡补做结算）。此时 token 已被 `:399-401` 丢掉，只能退化解析，所以它必须记警告。 |
| `actorOf(record)` | 从一条已存的 RollRecord 反查 actor：先 `tokenUuid` 再 `actorUuid`。 |
| `soleToken(actor)` | **收窄用的唯一实现。** 一个 actor 在当前场景恰好只有一个活动 token 时，那个 token 就是无歧义的答案；0 个或 ≥2 个一律返回 `null`（不猜）。掷骰总线包裹 `Item#roll` 时用它补 token，`actorById` 内部也用它——**一份实现两处用，不许各自私有复制一遍。** |
| `actorById(id, {warn})` | **遗留裸 id 专用**。系统有两处只存裸 actor id：`dataset.crewpanic`（`module/documents/actor.mjs:854`、`:1080`，`module/sheets/vehicle-sheet.mjs:440/463/474`，`module/sheets/spacecraft-sheet.mjs:444`，全部写作 `game.actors.get(dataset.crewpanic)`），以及载具乘员槽 `crew.occupants[].id`（`module/data/actor-vehicle.mjs:130-138` 的 `ArrayField(SchemaField{id, position})`，`id` 在 `:133`）。契约 §7 规定这些点也必须经 resolver，不许各自 `game.actors.get()`。 |

**`tokenUuid` 的诚实边界（契约 §4 K1）**：`tokenUuid` 只在三种情况下非空——掷骰源自 token 表单、合成 actor 自带 `Actor#token`、或推骰从父记录继承。**其余情况一律 `null` 并如实记录，绝不允许用 `game.actors.get()` 之类的兜底伪造一个出来。** 本任务的 `refs()` 因此只认两个诚实来源：调用方传进来的 token，以及 `actor.token`（合成 actor 所在的那个 TokenDocument，这是 Foundry 自己的属性，不是猜测）。两个都没有就返回 `tokenUuid: null`。

**解析顺序**（设计文档 §3.2 K4）：`canvas.tokens.get(speaker.token)?.actor` → `ChatMessage.getSpeakerActor(speaker)` → `game.actors.get(speaker.actor)`（仅兜底）。

`purePickRef` 是这条链的可测内核：它不碰任何 Foundry 全局，三个查找函数由调用方注入。契约 §0.1 的分层铁律要求 `pure*` 函数不得引用 `game` / `ui` / `canvas` / `CONFIG` / `Hooks` / `foundry` / `ChatMessage` / `Roll` / `libWrapper`，并且**参数只能是普通对象、数组与原始值**。

注入的 `lookups` 三个函数的返回形状（本任务定义的约定，两个测试与实现都照它写）：

| 函数 | 返回 | 对应的真实 Foundry 对象 |
|---|---|---|
| `lookups.token(tokenId)` | `{uuid: string, actor: {uuid: string}\|null}` 或 `null` | `TokenDocument`，`.uuid` 是 token 地址，`.actor` 是（可能是合成的）actor |
| `lookups.messageActor()` | `{uuid: string}` 或 `null` | `ChatMessage.getSpeakerActor(speaker)` 的返回 |
| `lookups.actor(actorId)` | `{uuid: string}` 或 `null` | `game.actors.get(actorId)` 的返回 |

---

- [ ] **Step 1: 写纯函数的失败测试**

创建 `test/resolver.test.mjs`：

```js
import { describe, expect, it } from "vitest";
import { purePickRef } from "../scripts/kernel/resolver.mjs";

const TOKEN_A_UUID = "Scene.s1.Token.tA";
const SYNTH_A_UUID = "Scene.s1.Token.tA.Actor.drone";
const BASE_UUID = "Actor.drone";

/** Build the injected lookups; every branch defaults to "found nothing". */
function lookups({ token = null, messageActor = null, actor = null } = {}) {
  return {
    token: () => token,
    messageActor: () => messageActor,
    actor: () => actor,
  };
}

describe("purePickRef", () => {
  it("branch 1 — prefers the token's own actor over both fallbacks", () => {
    const ref = purePickRef(
      { scene: "s1", token: "tA", actor: "drone" },
      lookups({
        token: { uuid: TOKEN_A_UUID, actor: { uuid: SYNTH_A_UUID } },
        messageActor: { uuid: BASE_UUID },
        actor: { uuid: BASE_UUID },
      }),
    );
    expect(ref).toEqual({ actorUuid: SYNTH_A_UUID, tokenUuid: TOKEN_A_UUID, source: "token" });
  });

  it("branch 2 — falls back to the message actor when the token document has no actor", () => {
    const ref = purePickRef(
      { scene: "s1", token: "tA", actor: "drone" },
      lookups({
        token: { uuid: TOKEN_A_UUID, actor: null },
        messageActor: { uuid: BASE_UUID },
        actor: { uuid: BASE_UUID },
      }),
    );
    expect(ref).toEqual({ actorUuid: BASE_UUID, tokenUuid: TOKEN_A_UUID, source: "message" });
  });

  it("branch 2 — message actor with no token id at all leaves tokenUuid null", () => {
    const ref = purePickRef({ actor: "drone" }, lookups({ messageActor: { uuid: BASE_UUID } }));
    expect(ref).toEqual({ actorUuid: BASE_UUID, tokenUuid: null, source: "message" });
  });

  it("branch 3 — last-resort base actor lookup is reported as source 'actor'", () => {
    const ref = purePickRef({ actor: "drone" }, lookups({ actor: { uuid: BASE_UUID } }));
    expect(ref).toEqual({ actorUuid: BASE_UUID, tokenUuid: null, source: "actor" });
  });

  it("branch 4 — nothing resolves", () => {
    const ref = purePickRef({ actor: "ghost" }, lookups());
    expect(ref).toEqual({ actorUuid: null, tokenUuid: null, source: "none" });
  });

  it("branch 4 — a deleted token keeps no uuid and still reports 'none'", () => {
    const ref = purePickRef({ scene: "s1", token: "gone", actor: "ghost" }, lookups());
    expect(ref).toEqual({ actorUuid: null, tokenUuid: null, source: "none" });
  });

  it("branch 4 — a token that resolves but nothing else does still records the token uuid", () => {
    const ref = purePickRef(
      { scene: "s1", token: "tA", actor: "ghost" },
      lookups({ token: { uuid: TOKEN_A_UUID, actor: null } }),
    );
    expect(ref).toEqual({ actorUuid: null, tokenUuid: TOKEN_A_UUID, source: "none" });
  });

  it("tolerates a missing speaker and missing lookups", () => {
    const none = { actorUuid: null, tokenUuid: null, source: "none" };
    expect(purePickRef(null, lookups())).toEqual(none);
    expect(purePickRef(undefined, undefined)).toEqual(none);
    expect(purePickRef({ actor: "drone" }, {})).toEqual(none);
  });

  it("touches no Foundry global (the contract's layering rule)", () => {
    expect(globalThis.game).toBeUndefined();
    expect(globalThis.canvas).toBeUndefined();
    expect(globalThis.ChatMessage).toBeUndefined();
    expect(() =>
      purePickRef({ scene: "s1", token: "tA", actor: "drone" }, lookups({ actor: { uuid: BASE_UUID } })),
    ).not.toThrow();
  });
});
```

- [ ] **Step 2: 跑它，看它失败**

Run: `npx vitest run test/resolver.test.mjs`

Expected: FAIL —— 整个文件收集不起来，报 `Error: Failed to load url ../scripts/kernel/resolver.mjs (resolved id: ...). Does the file exist?`（文件还不存在）。

- [ ] **Step 3: 实现 purePickRef（只写纯函数，先不写 resolver 对象）**

创建 `scripts/kernel/resolver.mjs`：

```js
/**
 * K4 · Resolver — uuid-first actor/token resolution.
 *
 * The system builds every chat speaker from an actor id alone
 * (systems/alienrpg/module/helpers/YZEDiceRoller.mjs:397-401), and it forces every
 * NPC token unlinked (module/documents/actor.mjs:93, module/alienrpg.mjs:366-376).
 * Resolving a speaker through game.actors.get() therefore collapses three Drones
 * onto one shared base Actor. Everything here resolves by uuid instead.
 */

/**
 * @typedef {object} RefLookups
 * @property {(tokenId: string) => ({uuid: string, actor: ({uuid: string}|null)}|null)} token
 *   Mirrors a TokenDocument: `.uuid` is the token address, `.actor` its (possibly synthetic) actor.
 * @property {() => ({uuid: string}|null)} messageActor  Mirrors ChatMessage.getSpeakerActor(speaker).
 * @property {(actorId: string) => ({uuid: string}|null)} actor  Mirrors game.actors.get(id).
 */

/**
 * Pick the actor/token reference a chat speaker points at.
 * Pure: never touches game / ui / canvas / CONFIG / Hooks / foundry / ChatMessage / Roll,
 * and every parameter is a plain object or primitive — never a live Foundry Document.
 *
 * @param {{scene?: string, token?: string, actor?: string}|null|undefined} speaker
 * @param {RefLookups|null|undefined} lookups
 * @returns {{actorUuid: string|null, tokenUuid: string|null, source: "token"|"message"|"actor"|"none"}}
 */
export function purePickRef(speaker, lookups) {
  if (!speaker || typeof speaker !== "object") {
    return { actorUuid: null, tokenUuid: null, source: "none" };
  }
  const look = lookups && typeof lookups === "object" ? lookups : {};

  // 1) The token's own actor. For an unlinked token this is the synthetic actor,
  //    whose uuid embeds the token id, so three Drones stay three distinct actors.
  const tokenDoc = speaker.token && typeof look.token === "function" ? look.token(speaker.token) : null;
  const tokenUuid = typeof tokenDoc?.uuid === "string" && tokenDoc.uuid ? tokenDoc.uuid : null;
  const tokenActorUuid =
    typeof tokenDoc?.actor?.uuid === "string" && tokenDoc.actor.uuid ? tokenDoc.actor.uuid : null;
  if (tokenActorUuid) return { actorUuid: tokenActorUuid, tokenUuid, source: "token" };

  // 2) Foundry's own speaker resolution, which understands scene + token + alias.
  if (typeof look.messageActor === "function") {
    const viaMessage = look.messageActor();
    if (typeof viaMessage?.uuid === "string" && viaMessage.uuid) {
      return { actorUuid: viaMessage.uuid, tokenUuid, source: "message" };
    }
  }

  // 3) Last resort: the base Actor by id. Only reached when branch 2 is unavailable
  //    (no ChatMessage class) or returned nothing — core's own getSpeakerActor already
  //    ends with a game.actors.get() fallback of its own.
  if (speaker.actor && typeof look.actor === "function") {
    const base = look.actor(speaker.actor);
    if (typeof base?.uuid === "string" && base.uuid) {
      return { actorUuid: base.uuid, tokenUuid, source: "actor" };
    }
  }

  return { actorUuid: null, tokenUuid, source: "none" };
}
```

- [ ] **Step 4: 跑它，看它通过**

Run: `npx vitest run test/resolver.test.mjs`
Expected: PASS —— `1 passed` 文件，`9 passed` 用例。

- [ ] **Step 5: 提交**

```bash
git add scripts/kernel/resolver.mjs test/resolver.test.mjs && git commit -F - <<'EOF'
feat(kernel): K4 purePickRef —— uuid 优先的引用选取

四条分支依次尝试 token → message → actor → none。三个查找函数由调用方注入，
函数体不触碰任何 Foundry 全局，参数只收普通对象，可被 vitest 直接跑。

非链接 token 是这里要防的失效：系统在 YZEDiceRoller.mjs:397-401 只用 actor id
建 speaker，token 被丢掉；而 actor.mjs:93 与 alienrpg.mjs:366-376 强制怪物走
非链接，于是三只 Drone 共享一张基础卡，按 id 解析会把它们折叠成一只。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 6: 为副作用层写失败测试（只用公共桩的 `world` 选项与 `ctx.documents`，不造任何全局）**

创建 `test/resolver-foundry.test.mjs`。

三条纪律，逐条都有理由：

1. **世界文档只从 `installFoundryStub({ world })` 与 `ctx.documents` 进**。`world` 负责填出 `game.actors` / `game.scenes` 这些集合；`ctx.documents` 是契约 §0.3 写死的 `fromUuidSync` 后备 Map，往里 `set()` 是它的正规用法。
2. **合成 actor 与 TokenDocument 只进 `ctx.documents`，不进 `world`**。真实 Foundry 里 `game.actors` 只装世界 actor，合成 actor 只能按 uuid 拿到；而且合成 actor 的 `id` 与基础卡**完全相同**（这正是本任务要防的病），塞进 `world` 会让 `game.actors.get("drone")` 变成一场赌博。
3. **不给 `globalThis.canvas` 赋值**。桩不保证提供 `canvas`，而实现里 `canvas.tokens.get()` 只是**第一顺位**，后面跟着 `game.scenes.get(speaker.scene).tokens.get()` 这条回退——测试走回退这条，canvas 那条交给 MANUAL VERIFICATION 步骤 5（那里的 canvas 是真的）。

```js
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installFoundryStub, uninstallFoundryStub } from "./stubs/foundry.mjs";
import { resolver } from "../scripts/kernel/resolver.mjs";

let baseActor, synthA, synthB, tokenA, tokenB, placedA, placedB, scene, ctx;
/** What baseActor.getActiveTokens() hands back; individual tests reassign it. */
let activeTokens = [];

/**
 * Build the fixture documents. Plain objects only — the stub stores whatever we hand it.
 * @returns {Record<string, object>} the WORLD documents (base actor + scene), keyed by uuid
 */
function buildFixture() {
  activeTokens = [];

  baseActor = {
    uuid: "Actor.drone",
    documentName: "Actor",
    id: "drone",
    name: "Drone",
    token: null, // Actor#token is null on a base actor…
    prototypeToken: { actorLink: false },
    getActiveTokens: () => activeTokens,
  };
  synthA = { uuid: "Scene.s1.Token.tA.Actor.drone", documentName: "Actor", id: "drone", name: "Drone", token: null };
  synthB = { uuid: "Scene.s1.Token.tB.Actor.drone", documentName: "Actor", id: "drone", name: "Drone", token: null };
  tokenA = { uuid: "Scene.s1.Token.tA", documentName: "Token", id: "tA", actorLink: false, actor: synthA };
  tokenB = { uuid: "Scene.s1.Token.tB", documentName: "Token", id: "tB", actorLink: false, actor: synthB };
  synthA.token = tokenA; // …and IS its TokenDocument on a synthetic one.
  synthB.token = tokenB;
  // A placed Token on the canvas wraps its TokenDocument as `.document`.
  placedA = { id: "tA", document: tokenA, actor: synthA };
  placedB = { id: "tB", document: tokenB, actor: synthB };

  const sceneTokens = new Map([
    ["tA", tokenA],
    ["tB", tokenB],
  ]);
  scene = {
    uuid: "Scene.s1",
    documentName: "Scene",
    id: "s1",
    name: "Deck C",
    tokens: { get: (id) => sceneTokens.get(id) ?? null, contents: [...sceneTokens.values()] },
  };

  return { "Actor.drone": baseActor, "Scene.s1": scene };
}

beforeEach(() => {
  const world = buildFixture();
  ctx = installFoundryStub({ world });
  // Synthetic actors and TokenDocuments are not world documents; they are reachable by
  // uuid only. ctx.documents is the map behind fromUuidSync (contract §0.3).
  for (const doc of [tokenA, tokenB, synthA, synthB]) ctx.documents.set(doc.uuid, doc);
});

afterEach(() => {
  uninstallFoundryStub();
  vi.restoreAllMocks();
});

describe("stub prerequisites", () => {
  it("gives this file a uuid graph and world collections", () => {
    // If any of these is red the shared stub does not meet contract §0.3 — report it to the
    // stub's owner. Do NOT patch globalThis here; a private fake only agrees with itself.
    expect(typeof globalThis.foundry?.utils?.fromUuidSync).toBe("function");
    expect(globalThis.foundry.utils.fromUuidSync("Actor.drone")).toBe(baseActor);
    expect(globalThis.foundry.utils.fromUuidSync("Scene.s1.Token.tA.Actor.drone")).toBe(synthA);
    expect(globalThis.game.actors.get("drone")).toBe(baseActor);
    expect(globalThis.game.scenes.get("s1")?.tokens?.get("tA")).toBe(tokenA);
  });
});

describe("resolver.fromSpeaker", () => {
  it("gives two tokens of one base actor two DIFFERENT actors, silently", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const a = resolver.fromSpeaker({ scene: "s1", token: "tA", actor: "drone" });
    const b = resolver.fromSpeaker({ scene: "s1", token: "tB", actor: "drone" });
    expect(a).toBe(synthA);
    expect(b).toBe(synthB);
    expect(a).not.toBe(b);
    expect(a).not.toBe(baseActor);
    expect(warn).not.toHaveBeenCalled();
  });

  it("degrades to the base actor and warns when the token is gone", () => {
    // Whether the stub's ChatMessage.getSpeakerActor answers or not, the outcome is the same
    // document and the same single warning: the speaker carried no resolvable token.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.fromSpeaker({ scene: "s1", token: "deleted", actor: "drone" })).toBe(baseActor);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("warns with module id and resolved uuid when the speaker has no token at all", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.fromSpeaker({ actor: "drone" })).toBe(baseActor);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("alien-evolved-automation");
    expect(String(warn.mock.calls[0][0])).toContain("Actor.drone");
  });

  it("returns null without warning when nothing resolves", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.fromSpeaker({ actor: "ghost" })).toBeNull();
    expect(resolver.fromSpeaker(null)).toBeNull();
    expect(warn).not.toHaveBeenCalled();
  });
});

describe("resolver.refs", () => {
  it("prefers the token's actor and accepts a placed Token", () => {
    expect(resolver.refs(baseActor, placedA)).toEqual({ actorUuid: synthA.uuid, tokenUuid: tokenA.uuid });
  });

  it("accepts a bare TokenDocument too", () => {
    expect(resolver.refs(baseActor, tokenB)).toEqual({ actorUuid: synthB.uuid, tokenUuid: tokenB.uuid });
  });

  it("recovers the token from a synthetic actor handed over without one", () => {
    // Actor#token is Foundry's own property, not a guess: it IS the TokenDocument
    // the synthetic actor lives inside.
    expect(resolver.refs(synthA, null)).toEqual({ actorUuid: synthA.uuid, tokenUuid: tokenA.uuid });
  });

  it("never invents a token uuid — a base actor with no token stays null", () => {
    expect(resolver.refs(baseActor, null)).toEqual({ actorUuid: "Actor.drone", tokenUuid: null });
  });

  it("returns two nulls when given nothing", () => {
    expect(resolver.refs(null, null)).toEqual({ actorUuid: null, tokenUuid: null });
  });
});

describe("resolver.actorOf", () => {
  it("prefers the record's token uuid over its actor uuid", () => {
    expect(resolver.actorOf({ tokenUuid: tokenB.uuid, actorUuid: baseActor.uuid })).toBe(synthB);
  });

  it("uses the actor uuid when the token was deleted after the roll", () => {
    expect(resolver.actorOf({ tokenUuid: "Scene.s1.Token.gone", actorUuid: baseActor.uuid })).toBe(baseActor);
  });

  it("resolves a record that never had a token", () => {
    expect(resolver.actorOf({ tokenUuid: null, actorUuid: synthA.uuid })).toBe(synthA);
  });

  it("returns null for an empty or unresolvable record", () => {
    expect(resolver.actorOf(null)).toBeNull();
    expect(resolver.actorOf({ tokenUuid: null, actorUuid: null })).toBeNull();
    expect(resolver.actorOf({ tokenUuid: null, actorUuid: "Actor.missing" })).toBeNull();
  });
});
```

- [ ] **Step 7: 跑它，看它失败**

Run: `npx vitest run test/resolver-foundry.test.mjs`

Expected: FAIL —— 14 条用例全红，因为 `resolver.mjs` 目前只导出 `purePickRef`。具体报法取决于你的 Vite 版本是否校验具名导出：要么整文件收集失败并报 `SyntaxError: [vite] The requested module '/scripts/kernel/resolver.mjs' does not provide an export named 'resolver'`，要么每条用例报 `TypeError: Cannot read properties of undefined (reading 'fromSpeaker' / 'refs' / 'actorOf')`。**两种都算见到了预期失败。若报的是别的错（例如找不到 `./stubs/foundry.mjs`、`installFoundryStub is not a function`，或 `stub prerequisites` 那条红了），停下来——那说明桩没装好或不达契约 §0.3，报给桩的属主，不要在本文件里自己补全局。**

- [ ] **Step 8: 实现副作用层（fromSpeaker / refs / actorOf）**

在 `scripts/kernel/resolver.mjs` 的文件注释块之后、`@typedef` 之前插入 import：

```js
import { MID } from "../const.mjs";
```

文件末尾追加：

```js
/**
 * fromUuidSync lives at foundry.utils.fromUuidSync in V13/V14 and is also exposed
 * as a bare global. Resolve it lazily so this module can be imported outside Foundry.
 * @param {string|null|undefined} uuid
 * @returns {object|null}
 */
function syncUuid(uuid) {
  if (typeof uuid !== "string" || !uuid) return null;
  const fn = globalThis.foundry?.utils?.fromUuidSync ?? globalThis.fromUuidSync;
  if (typeof fn !== "function") return null;
  try {
    return fn(uuid) ?? null;
  } catch {
    return null;
  }
}

/**
 * Wire the three lookups to live Foundry state for one specific speaker.
 * @param {{scene?: string, token?: string, actor?: string}} speaker
 * @returns {RefLookups}
 */
function liveLookups(speaker) {
  // The bare global still exists in V13/V14; the namespaced class is the forward-looking home.
  const CM = globalThis.ChatMessage ?? globalThis.foundry?.documents?.ChatMessage;
  return {
    token: (tokenId) => {
      // canvas.tokens holds the ACTIVE scene only; fall back to the speaker's scene.
      const placed = globalThis.canvas?.tokens?.get?.(tokenId);
      if (placed?.document) return placed.document;
      const scene = speaker?.scene ? globalThis.game?.scenes?.get?.(speaker.scene) : null;
      return scene?.tokens?.get?.(tokenId) ?? null;
    },
    messageActor: () => CM?.getSpeakerActor?.(speaker) ?? null,
    actor: (actorId) => globalThis.game?.actors?.get?.(actorId) ?? null,
  };
}

export const resolver = {
  /**
   * Resolve a chat speaker to the actor the roll actually belongs to.
   * This is the AFTER-THE-FACT path: by the time a message exists the system has already
   * discarded the token (YZEDiceRoller.mjs:397-401), so it usually degrades to the base
   * actor and says so. The live path is refs(), fed by the roll wrappers.
   *
   * The warning fires on a missing TOKEN, not on which branch produced the actor: core's own
   * ChatMessage.getSpeakerActor() already ends with a game.actors.get() fallback, so a
   * degraded resolution normally arrives as source "message" and a branch test would stay
   * silent exactly when the caller most needs to be told.
   *
   * @param {{scene?: string, token?: string, actor?: string}|null} speaker
   * @returns {object|null} an Actor document, or null
   */
  fromSpeaker(speaker) {
    if (!speaker || typeof speaker !== "object") return null;
    const ref = purePickRef(speaker, liveLookups(speaker));
    if (ref.actorUuid && !ref.tokenUuid) {
      // One string argument: callers and tests read console.warn's calls[0][0].
      console.warn(
        `${MID} | resolver.fromSpeaker: speaker carried no token ` +
          `(scene="${speaker.scene ?? ""}", token="${speaker.token ?? ""}", actor="${speaker.actor ?? ""}", ` +
          `source="${ref.source}") — resolved to ${ref.actorUuid}. If that is a shared base actor, ` +
          `writes aimed at it hit every token that shares it.`,
      );
    }
    if (!ref.actorUuid) return null;
    return syncUuid(ref.actorUuid);
  },

  /**
   * Build the uuid pair a RollRecord stores. This is the ONLY producer of a record's
   * actorUuid/tokenUuid — callers must never assemble a uuid string themselves.
   *
   * The token's own actor always wins: for an unlinked token that is the synthetic actor,
   * which is what must be written. Exactly two honest token sources are accepted — the token
   * the caller hands over, and Actor#token (the TokenDocument a synthetic actor lives inside).
   * When neither exists tokenUuid stays null; do NOT fabricate one from game.actors.get().
   *
   * @param {object|null} actor an Actor document (base or synthetic)
   * @param {object|null} token a placed Token or a TokenDocument
   * @returns {{actorUuid: string|null, tokenUuid: string|null}}
   */
  refs(actor, token) {
    const tokenDoc = token?.document ?? token ?? actor?.token ?? null;
    const actorDoc = tokenDoc?.actor ?? actor ?? null;
    return {
      actorUuid: typeof actorDoc?.uuid === "string" ? actorDoc.uuid : null,
      tokenUuid: typeof tokenDoc?.uuid === "string" ? tokenDoc.uuid : null,
    };
  },

  /**
   * Resolve the actor a stored RollRecord refers to. Token first, so a record made
   * on an unlinked token still reaches that token's own actor.
   * @param {{actorUuid: string|null, tokenUuid: string|null}|null} record
   * @returns {object|null}
   */
  actorOf(record) {
    if (!record || typeof record !== "object") return null;
    const tokenDoc = syncUuid(record.tokenUuid);
    if (tokenDoc?.actor) return tokenDoc.actor;
    return syncUuid(record.actorUuid);
  },
};
```

- [ ] **Step 9: 跑它，看它通过**

Run: `npx vitest run test/resolver.test.mjs test/resolver-foundry.test.mjs`
Expected: PASS —— `2 passed` 文件，`23 passed` 用例（9 + 14）。

- [ ] **Step 10: 提交**

```bash
git add scripts/kernel/resolver.mjs test/resolver-foundry.test.mjs && git commit -F - <<'EOF'
feat(kernel): K4 resolver 副作用层 —— fromSpeaker/refs/actorOf

liveLookups 把 canvas.tokens / speaker.scene / ChatMessage.getSpeakerActor /
game.actors 喂给 purePickRef；canvas 只覆盖当前活动场景，故对非活动场景补一条
game.scenes.get(speaker.scene).tokens.get() 回退。

refs() 是 RollRecord 里 actorUuid/tokenUuid 的唯一产出口，一律优先 token 自己的
actor：非链接 token 上那就是合成 actor，其 uuid 内嵌 token id，因此同卡的两只
Drone 得到两个不同的 actorUuid。token 只认两个诚实来源——调用方传入的 token 与
Actor#token；两者都没有就如实写 null，不用 game.actors.get() 伪造。

fromSpeaker() 的警告按「有 actor 无 token」触发，而不是按分支：核心自己的
getSpeakerActor() 末尾就带 game.actors.get() 兜底，退化解析通常以 source
"message" 到达，按分支判会在最该说话的时候闭嘴。

测试用公共桩的 installFoundryStub({world}) 与 ctx.documents 建文档图，合成 actor
与 TokenDocument 只进 ctx.documents（它们在真实 Foundry 里也不在 game.actors 里，
且与基础卡同 id），不往 globalThis 上补任何东西。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 11: 为 soleToken 写失败测试**

`soleToken(actor)` 是「把一个 actor 收窄到唯一一个 token」的**唯一实现**：`actor.getActiveTokens()` 恰好返回一个时给出那一个，0 个或 ≥2 个一律 `null`。掷骰总线包裹 `Item#roll` 时用它补 token（`ctx.token = this.actor?.token ?? resolver.soleToken(this.actor)`），本文件的 `actorById` 也用它——所以它必须返回 `getActiveTokens()` 原样交出来的**已放置 Token 对象**（带 `.document`），不要在这里换成 TokenDocument。

在 `test/resolver-foundry.test.mjs` 末尾追加：

```js
describe("resolver.soleToken", () => {
  it("returns the one active token, exactly as getActiveTokens handed it over", () => {
    activeTokens = [placedA];
    const sole = resolver.soleToken(baseActor);
    expect(sole).toBe(placedA);
    expect(sole.document).toBe(tokenA);
  });

  it("refuses to choose between two tokens", () => {
    activeTokens = [placedA, placedB];
    expect(resolver.soleToken(baseActor)).toBeNull();
  });

  it("returns null when the actor has no token on any scene", () => {
    activeTokens = [];
    expect(resolver.soleToken(baseActor)).toBeNull();
  });

  it("tolerates anything that is not a token-bearing actor", () => {
    expect(resolver.soleToken(null)).toBeNull();
    expect(resolver.soleToken(undefined)).toBeNull();
    expect(resolver.soleToken({})).toBeNull(); // no getActiveTokens at all
    expect(resolver.soleToken({ getActiveTokens: () => null })).toBeNull();
  });
});
```

- [ ] **Step 12: 跑它，看它失败**

Run: `npx vitest run test/resolver-foundry.test.mjs -t "resolver.soleToken"`

Expected: FAIL —— 4 条全部报 `TypeError: resolver.soleToken is not a function`。

- [ ] **Step 13: 实现 soleToken**

在 `scripts/kernel/resolver.mjs` 的 `resolver` 对象里、`actorOf` 之后追加：

```js
  /**
   * Narrow an actor to its single active token, or refuse. Foundry's
   * Actor#getActiveTokens() returns the placed Token objects of the actor on the
   * current scene; exactly one of them is the only unambiguous answer, so 0 or 2+
   * yield null rather than a guess.
   *
   * Returned as-is (a placed Token, with `.document` on it) because that is what the
   * roll wrappers feed straight into refs(), which accepts either shape.
   *
   * @param {object|null|undefined} actor an Actor document
   * @returns {object|null} a placed Token, or null
   */
  soleToken(actor) {
    const tokens = typeof actor?.getActiveTokens === "function" ? actor.getActiveTokens() : null;
    if (!Array.isArray(tokens) || tokens.length !== 1) return null;
    return tokens[0] ?? null;
  },
```

- [ ] **Step 14: 跑它，看它通过**

Run: `npx vitest run test/resolver.test.mjs test/resolver-foundry.test.mjs`
Expected: PASS —— `2 passed` 文件，`27 passed` 用例（9 + 18）。

- [ ] **Step 15: 提交**

```bash
git add scripts/kernel/resolver.mjs test/resolver-foundry.test.mjs && git commit -F - <<'EOF'
feat(kernel): K4 soleToken —— 唯一活动 token 的收窄，一份实现

契约 §4 K4 的第五个成员。actor.getActiveTokens() 恰好一个时返回那一个已放置
Token（原样交出，带 .document），0 个或 ≥2 个返回 null，不猜。

掷骰总线包裹 Item#roll 时按 this.actor?.token ?? resolver.soleToken(this.actor)
补 token，本模组的 actorById 收窄裸 id 时也调它——两处共用这一份实现，不再各自
私有复制一遍 getActiveTokens 的判断。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 16: 为 actorById 写失败测试**

`actorById(id, {warn = true})` 专供两条只存裸 actor id 的遗留路径。**裸 id 天生只能指向基础 actor**：非链接原型下它是有歧义的，因为同一张卡的每个 token 都用这个 id。所以真值表是四行——链接原型精确、非链接且场上恰好一个 token 可收窄（走 `soleToken`）、非链接且 0 或 ≥2 个 token 只能退回基础卡并警告、id 根本不存在。

在 `test/resolver-foundry.test.mjs` 末尾追加：

```js
describe("resolver.actorById", () => {
  it("returns null and warns for an id no actor has", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.actorById("nobody")).toBeNull();
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("nobody");
  });

  it("returns the base actor with no warning when the prototype token is LINKED", () => {
    baseActor.prototypeToken.actorLink = true;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.actorById("drone")).toBe(baseActor);
    expect(warn).not.toHaveBeenCalled();
  });

  it("narrows an UNLINKED id to the single active token's synthetic actor, and warns", () => {
    activeTokens = [placedA];
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.actorById("drone")).toBe(synthA);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("Scene.s1.Token.tA.Actor.drone");
  });

  it("refuses to guess between two tokens and returns the shared base actor", () => {
    activeTokens = [placedA, placedB];
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.actorById("drone")).toBe(baseActor);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("2 active token");
  });

  it("returns the base actor when an unlinked actor has no token on any scene", () => {
    activeTokens = [];
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.actorById("drone")).toBe(baseActor);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("0 active token");
  });

  it("stays silent when the caller passes warn:false", () => {
    activeTokens = [placedA];
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.actorById("drone", { warn: false })).toBe(synthA);
    expect(resolver.actorById("nobody", { warn: false })).toBeNull();
    expect(warn).not.toHaveBeenCalled();
  });

  it("returns null for a non-string id without warning", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolver.actorById(null)).toBeNull();
    expect(resolver.actorById("")).toBeNull();
    expect(resolver.actorById(42)).toBeNull();
    expect(warn).not.toHaveBeenCalled();
  });
});
```

注意每条断言读的都是 `warn.mock.calls[0][0]`——**第一个实参**。所以实现必须把整句话拼成一个字符串传给 `console.warn`，不能拆成多个实参（拆了的话 `"2 active token"` 落在第二个实参里，断言会红）。

- [ ] **Step 17: 跑它，看它失败**

Run: `npx vitest run test/resolver-foundry.test.mjs -t "resolver.actorById"`

Expected: FAIL —— 7 条全部报 `TypeError: resolver.actorById is not a function`。

- [ ] **Step 18: 实现 actorById（收窄逻辑调 soleToken，不复制它）**

在 `scripts/kernel/resolver.mjs` 的 `resolver` 对象里、`soleToken` 之后追加：

```js
  /**
   * Resolve a BARE actor id. Contract §7 routes every legacy id through here rather than
   * letting call sites reach for game.actors.get().
   *
   * The two legacy shapes in the system:
   *   - `dataset.crewpanic` — module/documents/actor.mjs:854 and :1080,
   *     module/sheets/vehicle-sheet.mjs:440/463/474, module/sheets/spacecraft-sheet.mjs:444
   *   - a vehicle's crew slots — module/data/actor-vehicle.mjs:130-138, `crew.occupants[].id` (:133)
   *
   * A bare id can only ever name the BASE actor, so on an unlinked prototype it is
   * ambiguous by construction: every token of that actor carries the same id. Narrowing is
   * delegated to soleToken() — one implementation, used here and by the roll wrappers.
   * When it refuses, hand back the base actor and say so, because a write aimed at it
   * lands on every token sharing it.
   *
   * Every warning is ONE string argument: callers and tests read calls[0][0].
   *
   * @param {string} id
   * @param {{warn?: boolean}} [options]
   * @returns {object|null} an Actor document, or null
   */
  actorById(id, { warn = true } = {}) {
    if (typeof id !== "string" || !id) return null;
    const base = globalThis.game?.actors?.get?.(id) ?? null;
    if (!base) {
      if (warn) console.warn(`${MID} | resolver.actorById: no actor has id "${id}"`);
      return null;
    }
    // A linked prototype means token and base actor are the same data — the id is exact.
    if (base.prototypeToken?.actorLink !== false) return base;

    const sole = resolver.soleToken(base);
    const narrowed = sole ? (sole.actor ?? sole.document?.actor ?? null) : null;
    if (narrowed) {
      if (warn) {
        console.warn(
          `${MID} | resolver.actorById: bare id "${id}" narrowed to ${narrowed.uuid} — ` +
            `the id itself carried no token; this is a legacy path pending a uuid migration.`,
        );
      }
      return narrowed;
    }
    if (warn) {
      const count = (typeof base.getActiveTokens === "function" ? base.getActiveTokens() : [])?.length ?? 0;
      console.warn(
        `${MID} | resolver.actorById: bare id "${id}" names an UNLINKED actor with ` +
          `${count} active token(s); returning the shared base actor. ` +
          `Writes aimed at it will hit every token that shares this id.`,
      );
    }
    return base;
  },
```

- [ ] **Step 19: 跑它，看它通过**

Run: `npx vitest run test/resolver.test.mjs test/resolver-foundry.test.mjs`
Expected: PASS —— `2 passed` 文件，`34 passed` 用例（9 + 25）。

- [ ] **Step 20: 提交**

```bash
git add scripts/kernel/resolver.mjs test/resolver-foundry.test.mjs && git commit -F - <<'EOF'
feat(kernel): K4 actorById —— 遗留裸 actor id 的唯一入口

契约 §7 要求所有 actor 解析走 resolver，但系统有两处只存裸 id：
dataset.crewpanic（actor.mjs:854/1080、vehicle-sheet.mjs:440/463/474、
spacecraft-sheet.mjs:444）与载具乘员槽（actor-vehicle.mjs:130-138 的
crew.occupants[].id，字段在 :133）。这两处过去只能 game.actors.get()，
无警告、无收窄。

四行真值表：链接原型精确返回基础卡且不警告；非链接且场上恰好一个 token
经 soleToken 收窄到该 token 的合成 actor 并记一条说明；非链接且 0 或 ≥2 个
token 退回基础卡并明确警告「写入会打到共享同一 id 的每个 token」；id 不存在
返回 null。收窄一律委托 soleToken，不在这里重复一遍 getActiveTokens 的判断。
{warn:false} 供批量调用消音。三条警告各自拼成单个字符串实参，测试逐条断言。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 21: 往 main.mjs 的 imports 锚点加 import 行**

打开 `scripts/main.mjs`，先确认锚点与已有 import：

```bash
grep -n "AEA-ANCHOR: imports" scripts/main.mjs
grep -n "kernel/selftest.mjs" scripts/main.mjs
```

在 `/* AEA-ANCHOR: imports */` **之后**插入这一行（**属于本任务的槽，必插**）：

```js
import { resolver } from "./kernel/resolver.mjs";
```

再看第二条 grep 的结果：

- 若 `kernel/selftest.mjs` **已经有** import 行（自检模块的属主任务已落地），**什么都不做**。
- 若**没有**，紧跟着再插一行——Step 23 要在 `init` 里调 `selftest.register(...)`，缺这一行世界启动时会在 `init` 阶段直接 `ReferenceError: selftest is not defined`，整个模组死在开机：

```js
import { selftest } from "./kernel/selftest.mjs";
```

这一行是幂等补位：谁先落地谁写，后来者看见就跳过，两个任务都不会写出重复 import。

验证（两条都必须恰好是 1，`0` 说明没插上，`2` 说明写重了）：

```bash
grep -c 'from "./kernel/resolver.mjs"' scripts/main.mjs
grep -c 'from "./kernel/selftest.mjs"' scripts/main.mjs
```

- [ ] **Step 22: 把 api 里 resolver 那一个 null 换成 import 进来的对象**

`api` 是八个键的固定字面量，交付时全为 `null`。本任务**只换自己那一个槽**，绝不替换 `api` 这个对象本身，也绝不增删键。先确认目标文本唯一：

```bash
grep -c "resolver: null," scripts/main.mjs   # 必须输出 1
```

把这段：

```js
  features: null, patches: null, resolver: null, registry: null,
```

改成：

```js
  features: null, patches: null, resolver, registry: null,
```

（`resolver,` 是 ES 的属性简写，等价于 `resolver: resolver`。同一行上 `features` / `patches` / `registry` 三个槽是别人的，保持 `null` 原样不动——它们各自的属主任务会来换。）

验证语法与结果：

```bash
node --check scripts/main.mjs && grep -n "patches: null, resolver, registry" scripts/main.mjs
```

Expected: `node --check` 无输出（通过），grep 打出那一行。

- [ ] **Step 23: 把「非链接 token 身份不塌缩」登记为自检条目**

vitest 的桩证明不了真实 Foundry 的 uuid 形状（`Scene.x.Token.y.Actor.z` 由核心生成），所以契约 §0.4 要求这类断言进 `selftest`，并另配人眼步骤（Step 27）。

在 `scripts/main.mjs` 里找到 `init` 钩子体内的这一行**逐字文本**：

```js
  /* AEA-ANCHOR: init */
```

在它**之后**插入下面这一块（若那里已经有别的任务插的代码，接在它们后面即可——`selftest.register` 只是把 def 存进表里，与同段其它调用没有先后依赖）。四个 `ready` 子锚点、`i18nInit`、`diceSoNiceReady` 一概不动：

```js
  selftest.register({
    id: "resolver-token-identity",
    // label is an i18n KEY, not localized text: registration happens in `init`, which fires
    // BEFORE `i18nInit`, so no translation file is loaded yet and localizing here would freeze
    // the raw key into the def. The runner localizes when it reports.
    label: "AEA.selftest.resolverTokenIdentity",
    run: () => {
      const placed = globalThis.canvas?.tokens?.placeables ?? [];
      const unlinked = placed.filter((t) => t.document?.actorLink === false);
      if (unlinked.length === 0) {
        return { ok: true, detail: "no unlinked tokens on the active scene — nothing to check" };
      }
      const refs = unlinked.map((t) => resolver.refs(t.actor, t));
      const actorUuids = new Set(refs.map((r) => r.actorUuid));
      const tokenUuids = new Set(refs.map((r) => r.tokenUuid));
      const ok = actorUuids.size === unlinked.length && tokenUuids.size === unlinked.length;
      return {
        ok,
        detail:
          `${unlinked.length} unlinked token(s) -> ${actorUuids.size} distinct actorUuid, ` +
          `${tokenUuids.size} distinct tokenUuid` +
          (ok ? "" : ` | COLLAPSED: ${[...actorUuids].join(", ")}`),
      };
    },
  });
```

`detail` 是给 GM 看的诊断串（uuid 与计数），刻意不翻译；`label` 走 i18n。

验证：`node --check scripts/main.mjs`（无输出即通过）。

- [ ] **Step 24: 加两个语言键**

语言包是嵌套结构、顶层只有唯一一个 `AEA` 对象（`lang/` 有护栏测试盯着这一点）。在 `lang/en.json` 的 `AEA` 对象里加：

```json
    "selftest": {
      "resolverTokenIdentity": "Resolver: unlinked tokens keep separate identities"
    }
```

在 `lang/cn.json` 的 `AEA` 对象里加：

```json
    "selftest": {
      "resolverTokenIdentity": "解析器：非链接 token 各自保持独立身份"
    }
```

若 `AEA.selftest` 已存在，就只往里加 `resolverTokenIdentity` 这一个键，不要重复创建 `selftest` 对象；两个文件的键必须一一对应。

- [ ] **Step 25: 跑全量测试，确认语言包护栏与其余用例都没被打红**

Run: `npm test`

Expected: PASS —— 本任务的 34 条全绿，语言包护栏（顶层唯一键 `AEA`、en/cn 键集一致）全绿。**若有失败用例既不属于 `test/resolver*.test.mjs` 也不属于语言包护栏，记下来交回去，不要在本任务里顺手改别人的文件。**

- [ ] **Step 26: 提交**

```bash
git add scripts/main.mjs lang/en.json lang/cn.json && git commit -F - <<'EOF'
feat(kernel): resolver 接进 main.mjs，并登记非链接 token 身份自检

三处装配，一处不多：imports 锚点后加 resolver 的 import（selftest 那行若已在
就跳过）、api 八槽里把 resolver: null 换成 resolver（不替换 api 对象、不增删
键）、init 锚点后登记自检条目。ready 的四个子锚点与另外两个生命周期锚点不动
——resolver 是被动查询模块，没有安装步骤，也不挂任何钩子。

真实 uuid 形状（Scene.x.Token.y.Actor.z）由 Foundry 核心生成，vitest 的桩只能
复述作者的假设，证明不了它。按契约 §0.4，这类断言进 selftest：取当前场景全部
actorLink === false 的 token，经 resolver.refs 求 uuid，断言 actorUuid 与
tokenUuid 的去重个数都等于 token 个数。

label 存 i18n 键而不是已本地化文本：init 早于 i18nInit，登记时刻语言包还没
加载，localize 只会回声键名；本地化由 runAll() 在报告时做。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 27: MANUAL VERIFICATION（自检覆盖数字，人眼覆盖形状、canvas 分支与那两条警告）**

在本机 Foundry（V13 或 V14）里打开一个装了 `alienrpg` 4.1.13 的世界，启用本模组，然后：

1. 打开 Actors 侧边栏，找一个 `creature` 类型的怪物（例如 Drone）。把它**拖到场景地图上三次**，得到三个 token。
2. 按 F12 打开控制台，粘贴并回车：

   ```js
   const api = game.modules.get("alien-evolved-automation").api;
   const toks = canvas.tokens.placeables.filter(t => t.document.actorLink === false);
   console.table(toks.map(t => api.resolver.refs(t.actor, t)));
   ```
3. **期望观察**：表格有三行；三行的 `tokenUuid` 互不相同，形如 `Scene.<id>.Token.<id>`；三行的 `actorUuid` 也**互不相同**，形如 `Scene.<id>.Token.<id>.Actor.<id>`。若三行的 `actorUuid` 相同且形如 `Actor.<id>`，说明这三个 token 是链接的——检查该怪物的 prototype token 是否被手动改回了 linked。若 `api.resolver` 打出 `null`，说明 Step 22 的 api 槽没换成功，回去补。
4. 跑一次自检，确认它与人眼看到的一致：

   ```js
   console.table(await game.modules.get("alien-evolved-automation").api.selftest.runAll());
   ```
   **期望观察**：`resolver-token-identity` 那行 `ok` 为 `true`，`detail` 读作 `3 unlinked token(s) -> 3 distinct actorUuid, 3 distinct tokenUuid`；`label` 列显示的是人话（英文世界 `Resolver: unlinked tokens keep separate identities`，中文世界「解析器：非链接 token 各自保持独立身份」）。若 `label` 列显示的是裸键 `AEA.selftest.resolverTokenIdentity`，说明自检运行器没在报告时本地化 label——把这一条报给 selftest 模块的属主，本任务这边照契约存的就是键，不改。
5. 验 canvas 那条优先分支（vitest 里走的是 `game.scenes` 回退，这条只能人眼看）。选中第一个 token 后执行：

   ```js
   const api = game.modules.get("alien-evolved-automation").api;
   const t = canvas.tokens.placeables.find(x => x.document.actorLink === false);
   console.log(api.resolver.fromSpeaker({ scene: canvas.scene.id, token: t.id, actor: t.actor.id })?.uuid);
   ```
   **期望观察**：打出的 uuid 形如 `Scene.<id>.Token.<id>.Actor.<id>`（合成 actor），**且控制台没有黄色警告**——speaker 带着 token，解析没有退化。
6. 在角色卡上点任意一次属性掷骰产生一张聊天卡。控制台执行：

   ```js
   const api = game.modules.get("alien-evolved-automation").api;
   const msg = game.messages.contents.at(-1);
   console.log(msg.speaker, api.resolver.fromSpeaker(msg.speaker)?.uuid);
   ```
7. **期望观察**：`msg.speaker` 里 `token` 与 `scene` 为空（这正是 `YZEDiceRoller.mjs:397-401` 只传 actor id 的后果），`fromSpeaker` 打出的 uuid 形如 `Actor.<id>`，同时控制台出现一条以 `alien-evolved-automation | resolver.fromSpeaker:` 开头的黄色警告，句中写着 `speaker carried no token`。**这条警告出现是正确的**：它证明事后路径确实退化了，也正是掷骰包装器要在 `abilityRoll` / `Item#roll` / `pushRoll` 现场抓 actor 与 token、再喂 `resolver.refs()` 的原因。RollRecord 里的 token 身份来自那条主路径，不来自 speaker。
8. 打开一辆载具（vehicle）的角色卡，往 crew 槽里放一个角色。控制台执行（先把载具取到变量里，例如 `const v = game.actors.getName("你的载具名")`）：

   ```js
   const api = game.modules.get("alien-evolved-automation").api;
   const crewId = v.system.crew.occupants[0].id;
   console.log(api.resolver.actorById(crewId)?.uuid, api.resolver.soleToken(game.actors.get(crewId))?.document?.uuid);
   ```
9. **期望观察**：第一个值是被放进去的那个角色的 uuid。若该角色的 prototype token 是链接的，返回 `Actor.<id>` 且控制台**没有**警告，第二个值随场上有无 token 而定；若它是非链接怪物且场上只有一个它的 token，第一个值形如 `Scene.<id>.Token.<id>.Actor.<id>`（收窄成功）、控制台有一条 `narrowed to Scene...` 的说明、第二个值是那个 token 的 uuid（`soleToken` 与 `actorById` 走的是同一份收窄实现）；若场上有两个它的 token，第一个值退回 `Actor.<id>`、警告里写着 `2 active token(s)`、第二个值是 `undefined`（`soleToken` 拒绝猜）。

---
