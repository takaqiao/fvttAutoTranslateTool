export const ruleDefaults = {
  'pf2e.campaignFeats': false,
  'pf2e.campaignFeatSections': [],
  'pf2e.campaignType': 'none',
  'pf2e.mythic': 'disabled',
  'pf2e.automaticBonusVariant': 'noABP',
  'pf2e.freeArchetypeVariant': false,
  'pf2e.gradualBoostsVariant': false,
  'pf2e.proficiencyVariant': false,
  'pf2e.proficiencyUntrainedModifier': 0,
  'pf2e.proficiencyTrainedModifier': 2,
  'pf2e.proficiencyExpertModifier': 4,
  'pf2e.proficiencyMasterModifier': 6,
  'pf2e.proficiencyLegendaryModifier': 8,
  'pf2e.staminaVariant': false,
};

export function ruleSnapshot() {
  return {
    version: 2, generatedAt: 100, enforcedSince: 100,
    rules: {
      a: {...ruleDefaults, 'pf2e.freeArchetypeVariant': true, 'pf2e.gradualBoostsVariant': true,
        'pf2e.staminaVariant': true, 'pf2e.automaticBonusVariant': 'ABPRulesAsWritten'},
      b: {...ruleDefaults},
    },
    hashes: {a: 'a'.repeat(64), b: 'b'.repeat(64)},
    cardWorld: {A: 'a', B: 'b'},
    folderWorld: {},
  };
}

export async function makeCampaignEnvironment({isolated = true, setup = true} = {}) {
  class ClientSettings {
    settings = new Map(Object.entries(ruleDefaults).map(([key, value]) => [key, {default: value}]));
    get(namespace, key, options) {
      if (options?.document) return {key: `${namespace}.${key}`, value: ruleDefaults[`${namespace}.${key}`]};
      return structuredClone(ruleDefaults[`${namespace}.${key}`]);
    }
  }
  class Character {
    constructor(id) {
      this.id = id;
      this.type = 'character';
      this._source = {_id: id, type: this.type, flags: {pf2e: {}}, system: {}};
      this.flags = this._source.flags;
      this.level = 6;
      this.system = {build: {attributes: {apex: null}}};
      this.crafting = new CharacterCrafting(this);
      this.formulaItems = [];
      this.inventory = {contents: []};
      this.apps = {};
      this.resets = 0;
    }
    prepareData() {
      this.crafting.initialize();
      this.prepared = {
        fa: game.settings.get('pf2e', 'freeArchetypeVariant'),
        gradual: game.pf2e.settings.variants.gab,
        stamina: game.pf2e.settings.variants.stamina,
      };
      this.prepareDataFromItems();
      this.duringPrepare?.();
    }
    prepareDataFromItems() {
      this.duringPrepareFromItems?.();
      return game.settings.get('pf2e', 'freeArchetypeVariant');
    }
    reset() { this.resets++; this.prepareData(); }
  }
  class Familiar extends Character {
    constructor(id, master) {
      super(id);
      this.type = this._source.type = 'familiar';
      this._source.system.master = {id: master};
    }
  }
  class CharacterCrafting {
    #formulas = null;
    constructor(actor) { this.actor = actor; }
    initialize() { this.#formulas = null; }
    async getFormulas() {
      if (this.#formulas) return this.#formulas;
      await this.actor.formulaWait;
      this.#formulas = this.actor.formulaItems.map(item => ({
        item, uuid: item.uuid, batchSize: 1,
        dc: 22 + ({common: 0, uncommon: 2, rare: 5, unique: 10}[item.rarity] ?? 0)
          - (game.pf2e.settings.variants.pwol.enabled ? 6 : 0),
      }));
      return this.#formulas;
    }
  }
  class CharacterSheetPF2e {
    constructor(actor) { this.actor = actor; }
    async getData({wait = Promise.resolve()} = {}) {
      await wait;
      return {
        hasStamina: game.pf2e.settings.variants.stamina,
        abpEnabled: game.pf2e.variantRules.AutomaticBonusProgression.isEnabled(this.actor),
        martialProficiencies: {attacks: {simple: {rank: 2, value: 10}}, defenses: {light: {rank: 1, value: 8}}},
        apexAttributeOptions: ['str'],
        crafting: {knownFormulas: await this.actor.crafting.getFormulas()},
      };
    }
  }
  class ApplicationV2 {
    constructor(options) { this.options = options; }
    _configureRenderOptions() {}
    async _prepareContext(options) { await options.wait; return {}; }
    render(options = {}) {
      this._configureRenderOptions(options);
      return this._prepareContext(options);
    }
  }
  class AttributeBuilder extends ApplicationV2 {
    #actor;
    constructor(options) { super(options); this.#actor = options.actor; }
    get abpEnabled() { return game.pf2e.variantRules.AutomaticBonusProgression.isEnabled(this.#actor); }
    _configureRenderOptions(options) {
      super._configureRenderOptions(options);
      this.#actor = options.actor ?? this.#actor;
    }
    async _prepareContext(options) {
      const actor = this.#actor;
      return {...await super._prepareContext(options), state: {
        actorId: actor.id,
        abpEnabled: this.abpEnabled,
        gradualBoostsVariant: game.settings.get('pf2e', 'gradualBoostsVariant'),
      }};
    }
    async handleApexBoost(attribute) {
      this.#actor.system.build.attributes.apex = this.abpEnabled ? attribute : null;
    }
  }
  class DocumentSheetConfig {
    static async initializeSheets() {
      await Promise.resolve();
      CONFIG.Actor.sheetClasses.character['pf2e.CharacterSheetPF2e'] = {cls: CharacterSheetPF2e};
    }
  }
  globalThis.foundry = {
    helpers: {ClientSettings}, utils: {deepClone: structuredClone},
    applications: {api: {ApplicationV2}, apps: {DocumentSheetConfig}},
  };
  const onceHooks = new Map();
  globalThis.Hooks = {
    on() {},
    once(event, callback) {
      const callbacks = onceHooks.get(event) ?? [];
      callbacks.push(callback);
      onceHooks.set(event, callbacks);
    },
  };
  globalThis.CONFIG = {
    PF2E: {Actor: {documentClasses: {character: Character, familiar: Familiar}}},
    Actor: {sheetClasses: {character: {}}},
  };
  globalThis.game = {
    world: {id: 'pf2creation'}, user: {isGM: true}, ready: false,
    actors: new Map(), folders: new Map(), settings: new ClientSettings(),
    pf2e: {
      settings: {
        variants: {fa: false, gab: false, stamina: false, abp: 'noABP',
          pwol: {enabled: false, modifiers: [0, 2, 4, 6, 8]}},
        campaign: {feats: {enabled: false, sections: []}, type: null, mythic: 'disabled'},
      },
      variantRules: {AutomaticBonusProgression: class {
        static isEnabled(actor) {
          return !!actor?.flags?.pf2e && !actor.flags.pf2e.disableABP && game.pf2e.settings.variants.abp !== 'noABP';
        }
      }},
    },
  };
  const api = await import(`../../scripts/char/rules-override.mjs${isolated ? `?context=${Math.random()}` : ''}`);
  api.installOverride(ruleSnapshot());
  api.installActorRuleHooks?.();
  const a = new Character('A'), b = new Character('B');
  game.actors.set(a.id, a);
  game.actors.set(b.id, b);
  a.prepareData();
  b.prepareData();
  const runSetup = async () => {
    for (const callback of onceHooks.get('setup') ?? []) callback();
    onceHooks.delete('setup');
    await DocumentSheetConfig.initializeSheets();
  };
  if (setup) await runSetup();
  return {api, a, b, Character, Familiar, ClientSettings, CharacterSheetPF2e, AttributeBuilder, runSetup};
}

