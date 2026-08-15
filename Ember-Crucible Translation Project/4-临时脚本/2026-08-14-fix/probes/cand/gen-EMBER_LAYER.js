/* generated */
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
 * Ember encounter rosters: `system.encounter.tokens` on `ember.questEvent` /
 * `ember.standaloneEvent` pages is an array of spawn GROUPS, each with an
 * `actors` array whose entries carry a `tokenData` override blob.
 *
 * `tokenData.name` beats the actor's translated `prototypeToken.name`:
 * `EmberScene#_createActorToken` (ember.mjs:29345) passes `tokenData` into
 * `actor.getTokenDocument()` and then writes the result into the token's actor
 * delta, so an untranslated override puts an English name on the canvas the
 * moment a GM runs the encounter.
 *
 * Keyed by the ENGLISH override name, like the built-in `nameCollection`: both
 * nested arrays are unkeyed, so positional keys would re-target on any upstream
 * reorder, and the 382 leaves per pack are only 130 distinct strings.
 */
export function emberEncounterTokens(groups, translation) {
  if (!Array.isArray(groups) || !translation || typeof translation !== 'object') {
    return groups;
  }

  return groups.map((group) => {
    if (!Array.isArray(group?.actors)) return group;

    let changed = false;
    const actors = group.actors.map((entry) => {
      const name = entry?.tokenData?.name;
      if (!isStr(name)) return entry;
      const t = translation[name];
      if (!isStr(t) || t === name) return entry;
      changed = true;
      return foundry.utils.mergeObject(entry, { tokenData: { name: t } }, { inplace: false });
    });

    return changed
      ? foundry.utils.mergeObject(group, { actors }, { inplace: false })
      : group;
  });
}

emberEncounterTokens.extract = (groups) => {
  if (!Array.isArray(groups)) return undefined;
  const out = {};
  for (const group of groups) {
    if (!Array.isArray(group?.actors)) continue;
    for (const entry of group.actors) {
      const name = entry?.tokenData?.name;
      if (isStr(name)) out[name] ??= name;
    }
  }
  return Object.keys(out).length ? out : undefined;
};

export const PROJECT_CONVERTERS = {
  crucibleDescription,
  crucibleNested,
  crucibleActions,
  emberEncounterTokens,
};


export const DOCUMENT_MAPPINGS = {
  "Item": {
    "name": "name",
    "description": {
      "path": "system.description",
      "converter": "crucibleDescription"
    },
    "actions": {
      "path": "system.actions",
      "converter": "crucibleActions"
    },
    "effects": {
      "path": "effects",
      "converter": "document",
      "documentType": "ActiveEffect",
      "cardinality": "many"
    }
  },
  "Actor": {
    "name": "name",
    "tokenName": "prototypeToken.name",
    "biography": {
      "path": "system.details.biography",
      "converter": "crucibleNested"
    },
    "ancestry": {
      "path": "system.details.ancestry",
      "converter": "crucibleNested"
    },
    "background": {
      "path": "system.details.background",
      "converter": "crucibleNested"
    },
    "archetype": {
      "path": "system.details.archetype",
      "converter": "crucibleNested"
    },
    "taxonomy": {
      "path": "system.details.taxonomy",
      "converter": "crucibleNested"
    },
    "items": {
      "path": "items",
      "converter": "document",
      "documentType": "Item",
      "cardinality": "many",
      "fallbackPolicy": "owner-package-before-generic"
    },
    "effects": {
      "path": "effects",
      "converter": "document",
      "documentType": "ActiveEffect",
      "cardinality": "many"
    }
  },
  "ActiveEffect": {
    "adjective": "system.adjective",
    "actions": {
      "path": "system.actions",
      "converter": "crucibleActions"
    }
  },
  "JournalEntryPage.ember.location": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "overview": "system.overview",
    "exposition": "system.exposition"
  },
  "JournalEntryPage.ember.biome": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "overview": "system.overview",
    "exposition": "system.exposition"
  },
  "JournalEntryPage.ember.quest": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "overview": "system.overview"
  },
  "JournalEntryPage.ember.questEvent": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "overview": "system.overview",
    "exposition": "system.exposition",
    "summary": "system.summary",
    "outcomes": {
      "path": "system.outcomes",
      "converter": "structured",
      "cardinality": "many",
      "container": "array",
      "key": "id",
      "mapping": {
        "label": "label",
        "summary": "summary"
      }
    },
    "encounterTokens": {
      "path": "system.encounter.tokens",
      "converter": "emberEncounterTokens"
    }
  },
  "JournalEntryPage.ember.standaloneEvent": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "overview": "system.overview",
    "exposition": "system.exposition",
    "summary": "system.summary",
    "outcomes": {
      "path": "system.outcomes",
      "converter": "structured",
      "cardinality": "many",
      "container": "array",
      "key": "id",
      "mapping": {
        "label": "label",
        "summary": "summary"
      }
    },
    "encounterTokens": {
      "path": "system.encounter.tokens",
      "converter": "emberEncounterTokens"
    }
  },
  "JournalEntryPage.ember.lore": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "contentOverview": "system.content.overview",
    "contentGamemaster": "system.content.gamemaster",
    "pronunciation": "system.pronunciation",
    "bannerCaption": "system.banner.caption"
  },
  "JournalEntryPage.ember.deity": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "contentOverview": "system.content.overview",
    "contentGamemaster": "system.content.gamemaster",
    "subtitle": "system.subtitle",
    "pronunciation": "system.pronunciation",
    "bannerCaption": "system.banner.caption"
  },
  "JournalEntryPage.ember.culture": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "contentOverview": "system.content.overview",
    "contentGamemaster": "system.content.gamemaster",
    "pronunciation": "system.pronunciation",
    "bannerCaption": "system.banner.caption"
  },
  "JournalEntryPage.ember.ancestry": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "contentOverview": "system.content.overview",
    "contentGamemaster": "system.content.gamemaster",
    "pronunciation": "system.pronunciation",
    "height": "system.height",
    "lifespan": "system.lifespan",
    "origin": "system.origin"
  },
  "JournalEntryPage.ember.cosmos": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "contentOverview": "system.content.overview",
    "contentGamemaster": "system.content.gamemaster",
    "subtitle": "system.subtitle",
    "pronunciation": "system.pronunciation",
    "bannerCaption": "system.banner.caption"
  },
  "JournalEntryPage.ember.organization": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "contentOverview": "system.content.overview",
    "contentGamemaster": "system.content.gamemaster",
    "pronunciation": "system.pronunciation"
  },
  "JournalEntryPage.ember.characterClass": {
    "name": "name",
    "text": "text.content",
    "caption": "image.caption",
    "contentOverview": "system.content.overview"
  },
  "JournalEntryPage.ember.questFlowchart": {
    "name": "name"
  },
  "Scene": {
    "levels": {
      "path": "levels",
      "converter": "nameCollection"
    },
    "tokens": {
      "path": "tokens",
      "converter": "nameCollection"
    },
    "navName": "navName",
    "sounds": {
      "path": "sounds",
      "converter": "nameCollection"
    }
  },
  "RegionBehavior.displayScrollingText": {
    "name": "name",
    "text": "system.text"
  },
  "RegionBehavior.teleportToken": {
    "name": "name",
    "revealedDialog": "system.dialog.revealed",
    "unrevealedDialog": "system.dialog.unrevealed"
  },
  "RegionBehavior.ember.trapTrigger": {
    "name": "name",
    "message": "system.message"
  },
  "RegionBehavior.ember.areaEffect": {
    "name": "name",
    "description": "system.description",
    "effects": {
      "path": "system.effects",
      "converter": "nameCollection"
    }
  }
};
