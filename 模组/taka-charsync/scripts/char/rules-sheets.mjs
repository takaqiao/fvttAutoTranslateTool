const installedSheets = new WeakSet();
const installedApplications = new WeakSet();
const boundActors = new WeakMap();
const installedCrafting = new WeakSet();
const latestReads = new WeakMap();
const handledReadErrors = new WeakSet();
// PF2e 8.6.1 calculateDC 的 -1 至 25 级表，越界等级使用 14。
const craftingDCs = [13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 42, 44, 46, 48, 50];

export const isRuleReadError = error => error instanceof Error && handledReadErrors.has(error);

async function readCurrentRules(actor, name, {state, fail, clear}, read, finish, invalidate = () => {}) {
  const reads = latestReads.get(actor) ?? new Map();
  const token = {};
  reads.set(name, token);
  latestReads.set(actor, reads);
  const failCurrent = () => {
    if (reads.get(name) === token) fail(actor, name);
  };
  clear(actor, name);
  for (let attempt = 0; attempt < 2; attempt++) {
    const before = state(actor);
    let data;
    try {
      data = await read();
    } catch (error) {
      if (!isRuleReadError(error)) failCurrent();
      if (error instanceof Error) handledReadErrors.add(error);
      throw error;
    }
    const after = state(actor);
    if (!after.ok) return finish({data, ready: false, fail: failCurrent});
    if (before.ok && before.world === after.world && before.hash === after.hash) {
      return finish({data, ready: true, fail: failCurrent});
    }
    // A pending render may outlive recovery or a profile change. Discard it once and read afresh.
    invalidate();
    if (attempt === 1) {
      failCurrent();
      return finish({data, ready: false, stale: true, fail: failCurrent});
    }
  }
}

export function ruleApplicationActor(app) {
  return boundActors.get(app) ?? app.actor ?? app.document?.actor ?? app.document ?? app.options?.actor ?? null;
}

export function installActorCraftingAdapter(actor, context) {
  const {value, fail} = context;
  const crafting = actor.crafting;
  const name = 'CharacterCrafting.getFormulas';
  if (!crafting || typeof crafting.getFormulas !== 'function' || typeof crafting.initialize !== 'function') {
    fail(actor, name);
    return;
  }
  if (installedCrafting.has(crafting)) return;
  const original = crafting.getFormulas;
  crafting.getFormulas = async function (...args) {
    return readCurrentRules(actor, name, context, () => original.apply(this, args), result => {
      const formulas = result.data;
      if (result.stale) {
        const error = new Error('读取制造配方时团规连续更新，请重新打开角色卡');
        handledReadErrors.add(error);
        throw error;
      }
      if (!result.ready) return formulas;
      try {
        if (!Array.isArray(formulas)) throw new Error('PF2e crafting formulas 数据形状已变化');
        const withoutLevel = value(actor, 'pf2e', 'proficiencyVariant');
        for (const formula of formulas) {
          if (!Number.isFinite(formula.item?.level)) throw new Error('PF2e crafting formula 缺少物品等级');
          const {level, rarity} = formula.item;
          const base = craftingDCs[level + 1] ?? 14;
          const adjustment = {uncommon: 2, rare: 5, unique: 10}[rarity] ?? 0;
          // 原位修改，CharacterCrafting 和卡面的私有缓存也会收到同一 DC。
          formula.dc = base - (withoutLevel ? Math.max(level, 0) : 0) + adjustment;
        }
        return formulas;
      } catch (error) {
        result.fail();
        if (error instanceof Error) handledReadErrors.add(error);
        throw error;
      }
    }, () => this.initialize());
  };
  installedCrafting.add(crafting);
}

/** PF2e 8.6.1 reads these fields after await; no global rule scope may span that wait. */
export function installRuleSheetAdapters(context) {
  const {value, fail, clear} = context;
  function finish(actor, result, name, adapt) {
    const {data} = result;
    if (!result.ready) {
      data.editable = false;
      return data;
    }
    try {
      adapt(data);
    } catch (error) {
      result.fail();
      data.editable = false;
      console.error('taka-charsync | 角色卡团规字段适配失败', name, actor.id, error);
    }
    return data;
  }

  let foundSheet = false;
  for (const entry of Object.values(globalThis.CONFIG?.Actor?.sheetClasses?.character ?? {})) {
    let prototype = entry.cls?.prototype;
    while (prototype && prototype.constructor.name !== 'CharacterSheetPF2e') prototype = Object.getPrototypeOf(prototype);
    if (!prototype || typeof prototype.getData !== 'function') continue;
    foundSheet = true;
    if (installedSheets.has(prototype)) continue;
    const original = prototype.getData;
    prototype.getData = async function (...args) {
      const actor = this.actor;
      const name = 'CharacterSheet.getData';
      return readCurrentRules(actor, name, context, () => original.apply(this, args), result => finish(actor, result, name, data => {
        if (!data.martialProficiencies?.attacks || !data.martialProficiencies?.defenses) {
          throw new Error('PF2e martialProficiencies 数据形状已变化');
        }
        data.hasStamina = value(actor, 'pf2e', 'staminaVariant');
        data.abpEnabled = game.pf2e.variantRules.AutomaticBonusProgression.isEnabled(actor);
        data.apexAttributeOptions = data.abpEnabled ? [] : actor.inventory.contents.flatMap(item =>
          item.system.apex?.selected === false && item.isInvested && item.system.apex.attribute !== actor.system.build.attributes.apex
            ? item.system.apex.attribute : []);
        const withoutLevel = value(actor, 'pf2e', 'proficiencyVariant');
        const modifiers = withoutLevel
          ? ['Untrained', 'Trained', 'Expert', 'Master', 'Legendary'].map(rank => value(actor, 'pf2e', `proficiency${rank}Modifier`))
          : [0, 2, 4, 6, 8];
        // createProficiencyModifier({domains: []}) has no modifier adjustments.
        for (const group of Object.values(data.martialProficiencies)) {
          for (const proficiency of Object.values(group)) {
            const rank = Math.max(0, Math.min(4, proficiency.rank ?? 0));
            proficiency.value = modifiers[rank] + (!withoutLevel && rank > 0 ? actor.level : 0);
          }
        }
      }));
    };
    installedSheets.add(prototype);
  }
  if (!foundSheet) fail(null, 'CharacterSheet.getData');
  else clear(null, 'CharacterSheet.getData');

  for (const entry of Object.values(globalThis.CONFIG?.Actor?.sheetClasses?.familiar ?? {})) {
    let prototype = entry.cls?.prototype;
    while (prototype && prototype.constructor.name !== 'FamiliarSheetPF2e') prototype = Object.getPrototypeOf(prototype);
    if (!prototype || typeof prototype.getData !== 'function' || installedSheets.has(prototype)) continue;
    const original = prototype.getData;
    prototype.getData = async function (...args) {
      const data = await original.apply(this, args);
      data.masters = data.masters.filter(master => context.masterState(this.actor, master.id)?.ok !== false);
      // 没有关联主人时团规尚不可用，但原生选择器仍须可用来修复关联。
      return data;
    };
    const originalSubmit = prototype._getSubmitData;
    prototype._getSubmitData = function (...args) {
      const data = originalSubmit.apply(this, args);
      const masterId = data['system.master.id'];
      const hp = 'system.attributes.hp.value';
      const source = this.actor._source.system;
      // 新魔宠源 HP 为 5，无主人时显示为 0；只换主人时别把未编辑的显示值写回。
      // 在表单端判断，避免 PF2e _preUpdate 钳制 HP 后丢失玩家实际输入的区别。
      if (masterId !== undefined && (masterId || null) !== (source.master.id || null)
        && data[hp] === this.actor.system.attributes.hp.value && data[hp] !== source.attributes.hp.value
        && context.masterState(this.actor, masterId)) delete data[hp];
      return data;
    };
    installedSheets.add(prototype);
  }

  const application = globalThis.foundry?.applications?.api?.ApplicationV2?.prototype;
  if (!application || typeof application._configureRenderOptions !== 'function') {
    fail(null, 'AttributeBuilder._configureRenderOptions');
    return;
  }
  if (installedApplications.has(application)) return;
  clear(null, 'AttributeBuilder._configureRenderOptions');
  const originalConfigure = application._configureRenderOptions;
  const builders = new WeakSet();
  application._configureRenderOptions = function (options) {
    if (this.constructor.name === 'AttributeBuilder') {
      const actor = options.actor ?? boundActors.get(this) ?? this.options?.actor;
      if (actor) boundActors.set(this, actor);
      if (!builders.has(this)) {
        const original = this._prepareContext;
        this._prepareContext = async function (...args) {
          const actor = boundActors.get(this);
          const name = 'AttributeBuilder._prepareContext';
          if (!actor) {
            fail(null, name);
            return original.apply(this, args);
          }
          return readCurrentRules(actor, name, context, () => original.apply(this, args), result => finish(actor, result, name, data => {
            if (!data.state || !Object.hasOwn(data.state, 'gradualBoostsVariant')) {
              throw new Error('PF2e AttributeBuilder state 数据形状已变化');
            }
            data.state.gradualBoostsVariant = value(actor, 'pf2e', 'gradualBoostsVariant');
            data.state.abpEnabled = game.pf2e.variantRules.AutomaticBonusProgression.isEnabled(actor);
          }));
        };
        builders.add(this);
      }
    }
    return originalConfigure.call(this, options);
  };
  installedApplications.add(application);
}
