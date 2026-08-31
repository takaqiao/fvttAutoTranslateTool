/* ------------------------------------------------------------------ *
 * Runtime converters (translate direction).
 *
 * The EXTRACT direction of these same three converters lives in
 * 3-常用脚本/extract/extract_en.mjs. If you change the shape one side
 * produces, change the other in the same commit.
 *
 * Babele's functional-converter signature is:
 *   fn(value, translation, source, contextCompendium, allTranslations, runtime, params)
 * ------------------------------------------------------------------ */

const isStr = (v) => typeof v === 'string' && v.trim().length > 0;

/**
 * Crucible `system.description` is polymorphic: a plain string on most item
 * types, `{public, private}` on equipment-like ones. Return the translation in
 * whichever shape the SOURCE uses, so nothing downstream sees a type change.
 */
export function crucibleDescription(value, translation) {
  if (translation === undefined || translation === null) return value;

  if (typeof value === 'string' || value === undefined || value === null) {
    if (isStr(translation)) return translation;
    if (isStr(translation?.public)) return translation.public;
    return value;
  }

  // ⚠ dnd5e / 通用形状 `{value, chat}` 必须单独处理，否则会**弄坏别人的翻译**。
  //
  // `babele.registerMapping()` 注册的是**全局**层（`core/babele.js:231` 只是
  // `registeredMappings.push(...)`，没有任何 module/pack 维度的作用域参数；
  // `mapping/document-mappings.js:267-287` 把 built-in→registered→loaded 合成一份
  // effectiveMappings，每个被 Babele 接管的合集都取这一份）。也就是说本项目的
  // `Item.description` 会**按键顶掉** Babele 内建的 `"system.description.value"`，
  // 对同世界里第三方汉化包（例如 dnd-simplified-chinese-babele-patch，它的
  // rules/mapping.json 没有 description 键、完全依赖内建默认）的 Item 一样生效。
  //
  // 少了这一支，`{value, chat}` 会掉进下面的 `{public: …}` 分支：`.value` 仍是英文，
  // 另外塞进一个 dnd5e schema 里根本没有的 `public` 键。这里按内建语义写回 `.value`。
  //
  // crucible 侧不会误入：它唯一的对象形态 description 是
  // `SchemaField{public, private}`（`module/models/item-physical.mjs:26-29`），
  // 没有 `value` 子字段。
  //
  // 只改 translate 方向，**extract 方向刻意保持原样** —— 让 extract 也吐
  // `{value, chat}` 会把 dnd5e 侧约 85 万字符拉进英文基线，那是项目所有者已定
  // 「先不管」的 Z1，属于另一件事。
  if (typeof value.value === 'string' && !('public' in value) && !('private' in value)) {
    const text = isStr(translation) ? translation : (isStr(translation.value) ? translation.value : null);
    return text === null ? value : foundry.utils.mergeObject(value, { value: text }, { inplace: false });
  }

  if (isStr(translation)) {
    return foundry.utils.mergeObject(value, { public: translation }, { inplace: false });
  }

  const patch = {};
  if (isStr(translation.public)) patch.public = translation.public;
  if (isStr(translation.private)) patch.private = translation.private;
  return Object.keys(patch).length
    ? foundry.utils.mergeObject(value, patch, { inplace: false })
    : value;
}

crucibleDescription.extract = (value) => {
  if (isStr(value)) return value;
  if (!value || typeof value !== 'object') return undefined;
  const out = {};
  if (isStr(value.public)) out.public = value.public;
  if (isStr(value.private)) out.private = value.private;
  return Object.keys(out).length ? out : undefined;
};

/**
 * `{name, description}` / `{public, private, appearance}` sub-objects that are
 * translated field-by-field in place.
 */
export function crucibleNested(value, translation) {
  if (!value || typeof value !== 'object' || !translation || typeof translation !== 'object') {
    return value;
  }
  const patch = {};
  for (const [k, v] of Object.entries(translation)) {
    if (isStr(v)) patch[k] = v;
  }
  return Object.keys(patch).length
    ? foundry.utils.mergeObject(value, patch, { inplace: false })
    : value;
}

crucibleNested.extract = (value) => {
  if (!value || typeof value !== 'object') return undefined;
  const out = {};
  for (const k of ['name', 'description', 'public', 'private', 'appearance']) {
    if (isStr(value[k])) out[k] = value[k];
  }
  return Object.keys(out).length ? out : undefined;
};

/**
 * Crucible actions: an array of objects with `id`. Translations are keyed by
 * that id and may carry a positional `effects: [{name}]` array.
 *
 * NOTE: this used to be `actions_converter`, which called
 * `game.babele.converters.actions_converter(...)` from inside
 * `adventure_items_converter`. Under Babele 2.9.1 `.converters` returns
 * FunctionalConverter OBJECTS, not functions, so that call threw a TypeError on
 * every adventure-embedded item that had actions. Nothing calls into the
 * registry any more — embedded items go through Babele's own `document`
 * converter now.
 */
export function crucibleActions(actions, translation) {
  if (!Array.isArray(actions) || !translation || typeof translation !== 'object') {
    return actions;
  }

  return actions.map((action) => {
    const t = translation[action?.id];
    if (!t || typeof t !== 'object') return action;

    const patch = {};
    if (isStr(t.name)) patch.name = t.name;
    if (isStr(t.description)) patch.description = t.description;
    if (isStr(t.condition)) patch.condition = t.condition;

    if (Array.isArray(t.effects) && Array.isArray(action.effects)) {
      patch.effects = action.effects.map((effect, i) => (
        isStr(t.effects[i]?.name)
          ? foundry.utils.mergeObject(effect, { name: t.effects[i].name }, { inplace: false })
          : effect
      ));
    }

    return Object.keys(patch).length
      ? foundry.utils.mergeObject(action, patch, { inplace: false })
      : action;
  });
}

crucibleActions.extract = (actions) => {
  if (!Array.isArray(actions)) return undefined;
  const out = {};
  for (const a of actions) {
    if (!isStr(a?.id)) continue;
    const e = {};
    if (isStr(a.name)) e.name = a.name;
    if (isStr(a.description)) e.description = a.description;
    if (isStr(a.condition)) e.condition = a.condition;
    if (Array.isArray(a.effects) && a.effects.length) {
      const eff = a.effects.map((x) => (isStr(x?.name) ? { name: x.name } : {}));
      if (eff.some((x) => x.name)) e.effects = eff;
    }
    if (Object.keys(e).length && !out[a.id]) out[a.id] = e;
  }
  return Object.keys(out).length ? out : undefined;
};

/**
 * `prototypeToken.name` —— 地图 token 上显示的名字。
 *
 * Babele 内建 Actor 默认把 `tokenName` 写成
 * `{path:'prototypeToken.name', converter:'name'}`，而 `name` ＝
 * `Converters.mappedField("name")`：
 *   `(_value, _translation, data, tc) => tc.translateField("name", data)`
 * 译文参数被丢弃，返回的是 **`name` 字段**的译文 —— `tokenName` 键从来没被读过。
 *
 * 这里做成内建行为的**超集**：有 `tokenName` 译文就用它，没有就原样退回
 * `translateField("name")`。退路必须留着，因为 `registerMapping` 是全局层，
 * 第三方合集的 Actor 也会走这一条，不能让它们的 token 名退回英文。
 *
 * 参数顺序即 Babele 的函数式转换器签名（见文件头）；`contextCompendium` 的取法
 * 与内建 `mappedField` 逐字一致（`tc ?? runtime?.currentCompendium?.()`）。
 */
export function crucibleTokenName(value, translation, source, contextCompendium, _allTranslations, runtime = {}) {
  if (isStr(translation)) return translation;
  const pack = contextCompendium ?? runtime?.currentCompendium?.() ?? null;
  const fallback = pack?.translateField?.('name', source, runtime);
  return isStr(fallback) ? fallback : value;
}

crucibleTokenName.extract = (value) => (isStr(value) ? value : undefined);

/**
 * Ember 遭遇模板里预置 token 的覆盖名：
 * `system.encounter.tokens[].actors[].tokenData.name`（两层嵌套数组）。
 *
 * 译文按**英文名本身**建键（与内建 `nameCollection` 同构），查不到就原样返回。
 * 写回用 `mergeObject(actor, {tokenData: {name}})` —— `tokenData` 两边都是对象，
 * Foundry 的 mergeObject 会**递归**进去只覆盖 `name`，
 * `_id`/`texture`/`x`/`y`/`elevation`/`rotation`/`flags`/`delta`/`disposition`
 * 等 15 个兄弟键一个不动（已用真实 pack 数据逐条回归验证）。
 */
export function emberEncounterTokenNames(tokens, translation) {
  if (!Array.isArray(tokens) || !translation || typeof translation !== 'object') return tokens;

  let changed = false;
  const out = tokens.map((token) => {
    if (!token || typeof token !== 'object' || !Array.isArray(token.actors)) return token;

    let hit = false;
    const actors = token.actors.map((actor) => {
      const name = actor?.tokenData?.name;
      if (!isStr(name)) return actor;
      const cn = translation[name];
      if (!isStr(cn) || cn === name) return actor;
      hit = true;
      return foundry.utils.mergeObject(actor, { tokenData: { name: cn } }, { inplace: false });
    });

    if (!hit) return token;
    changed = true;
    return foundry.utils.mergeObject(token, { actors }, { inplace: false });
  });

  return changed ? out : tokens;
}

emberEncounterTokenNames.extract = (tokens) => {
  if (!Array.isArray(tokens)) return undefined;
  const out = {};
  for (const token of tokens) {
    for (const actor of (token?.actors ?? [])) {
      const name = actor?.tokenData?.name;
      if (isStr(name) && !(name in out)) out[name] = name;
    }
  }
  return Object.keys(out).length ? out : undefined;
};

export const PROJECT_CONVERTERS = {
  crucibleDescription,
  crucibleNested,
  crucibleActions,
  crucibleTokenName,
  emberEncounterTokenNames,
};
