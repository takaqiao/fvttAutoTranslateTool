
/**
 * 指示物制作器的**部件显示名**（`<span class="part">` 那一行）。
 *
 * ── 为什么键是 `BeardWizard` 而不是 `Beard Wizard` ─────────────────────────
 * 上屏的串是**运行时拼出来**的：`getLayerChoicesV2()`（ember.mjs:64457）在 `labels:true` 时算
 *   `partId.split("/").at(-1).replace(/(?<!^)([A-Z1-9])/g, " $1")`
 * ⇒ 上游文本里只有 `BeardWizard` 这个 id，**没有** `Beard Wizard` 这个字面量。
 * 所以本表登记的是 **id**（上游真有的字面量），显示名在下面由**同一条正则**现算。
 * 上一轮备好的那份是按显示名建键、按 `kind:"composed"` 登记的 —— 那会把面板 D 档的
 * `uncheckedRaw` 顶过 `max`（186），**主闸当场红**，而那两个数不归本文件的作者。
 * 换成 id 建键之后：671 个键**逐条**在 ember.mjs 里查得到字面量 ⇒ 走 `literal` 正常核，
 * miss 侧一条不添、没核侧一条不添，**主闸不动**（实跑印证见本轮报告）。
 *
 * ── 覆盖账（2026-08-22 第三轮实测，探针在 4-临时脚本/2026-08-22-ui-round3/recon/）──
 * 枚举**以随包图集为准**（`assets/tokens/maker/{Character0,Character1,Monster0,Party0}.json`，
 * 5649 帧，前置自证逐份对上已知真值），不是只扫字面量 —— 上一轮只扫字面量得 1356，
 * 漏掉了 `makeLegPoseParts()` 与 `Marbled/Chiseled` 那两族运行时拼出来的 id。
 *   · 图集里的唯一末段 id **1535** 条；
 *   · 扣掉 23 条**上游未使用的图集残留**（既不在任何 `id: "…"` 声明里、也拼不出来，
 *     如 `Prosthetic` / `Metal4` / `Head1` / `Flag1` —— 结构上进不了 templateLayer.parts，
 *     不会上屏）⇒ **玩家真会看到的唯一显示名 1512 条**；
 *   · 本表直接覆盖 **671** 条 + 下面两族拼串 **107** 条 = **778** 条；
 *   · **仍缺 741 条**，全部在装备族（handItem 144 / helm 86 / chest 83 / symbol 80 /
 *     pants 41 / wrist 36 / backItem 31 / sleeve 22 / eyewear 16 / footwear 10 …
 *     外加 54 条 `…Lower` 手持物下半段）。按「玩家真会看到的」排序，本轮先做身体族。
 *
 * ── 两条既有问题，本轮一并修掉 ─────────────────────────────────────────────
 * ① `Heavy` / `Lithe` **被错译成体格词**（v1.1.27 就有，不是本轮引入）：
 *    TOKEN_MAKER_UI 里 `Heavy`＝壮硕 / `Lithe`＝纤瘦 是 layers.hbs:9 那行**体格**（build）的值，
 *    而 tail/arm/foot/head/torso 里也有叫 `Heavy` 的**部件**（该译「粗壮」）、torso 有 `Lithe`（该译「柔韧」）。
 *    同一张扁平表一个键只能有一个值 ⇒ 本表**不并进 `TOKEN_MAKER_WINDOW_UI`**，
 *    改由 `translateTokenMakerParts()` 只对 `.choice.layer .part` 这个选择器跑一遍，
 *    跑在通用遍历**之前**。体格行是 `.choice.build .part`、姿态行是 `.choice.stance .part`，
 *    选择器天然把它们排除在外 ⇒ 「壮硕 / 纤瘦」原样保留，部件那一侧拿到「粗壮 / 柔韧」。
 * ② 行尾的 `3 of 12`：`of` 写死在 `templates/applications/token-maker/layers.hbs:35`
 *    （`{{layer.chosenLabel}} of {{layer.choices.length}}`），整行是**一个**文本节点，
 *    查表接不住。同样在那个函数里就地改成 `3 / 12`，**只认 `^\d+ of \d+$` 这一种形状**。
 *
 * ⚠ 作用域：这两件事只在 `TOKEN_MAKER_APPS` 那两个窗口的根元素下做，
 *   不进 EXACT / PATTERNS / 任何全局通道 —— 本表里 `Down` / `Up` / `Fine` / `Bob` / `Claw`
 *   这种短词一旦漏到全局，会去吃别的模块的文本。
 */
