/*
 * Supported Crucible compendium packs (as of system 0.9.1):
 *   Item         : adversary-talents, ancestry, archetype, background,
 *                  crafting, equipment, spell, talent, taxonomy
 *   Actor        : pregens, summons
 *   JournalEntry : rules                 (categories_converter)
 *   Adventure    : playtest
 *   ActiveEffect : affixes               (default string mapping)
 *   Macro        : macros                (name only, default)
 *
 * ActiveEffect and Macro packs do NOT need custom converters — Babele's
 * default string replacement honors the per-pack `mapping` block inside
 * each compendium/cn/*.json. The converters below are only for the
 * non-trivial structures (actions arrays, embedded items, nested objects,
 * journal categories).
 */
Hooks.once('init', async function () {
    if (typeof Babele !== 'undefined') {

        // 1. Register the translation module
        game.babele.register({
            module: 'crucible-cn',
            lang: 'cn',
            dir: 'compendium/cn'
        });

        // 2. Custom converters for complex structures
        game.babele.registerConverters({

            /**
             * Converter pour les actions (arrays avec id)
             * Utilisé par : talents, sorts, consommables, actors
             */
            "actions_converter": (actions, translations) => {
                if (!actions || !translations) return actions;

                return actions.map(action => {
                    const translation = translations[action.id];
                    if (!translation) return action;

                    // Traduction des champs principaux
                    if (translation.name) action.name = translation.name;
                    if (translation.description) action.description = translation.description;
                    if (translation.condition) action.condition = translation.condition;

                    // Traduction des effets (si présents)
                    if (translation.effects && Array.isArray(action.effects)) {
                        action.effects = action.effects.map((effect, index) => {
                            const effectTranslation = translation.effects[index];
                            if (effectTranslation?.name) {
                                effect.name = effectTranslation.name;
                            }
                            return effect;
                        });
                    }

                    return action;
                });
            },

            /**
             * Converter pour les items d'actors (dans Adventures ou Actors directs)
             * Gère : nom, description, et actions des items
             */
            "adventure_items_converter": (items, translations) => {
                if (!items || !translations) return items;

                return items.map(item => {
                    const itemTranslation = translations[item.name];
                    if (!itemTranslation) return item;

                    // Nom de l'item
                    if (itemTranslation.name) {
                        item.name = itemTranslation.name;
                    }

                    // Description (gestion object/string)
                    if (itemTranslation.description) {
                        if (!item.system) item.system = {};

                        if (typeof itemTranslation.description === 'object') {
                            if (!item.system.description) item.system.description = {};
                            if (itemTranslation.description.public) {
                                item.system.description.public = itemTranslation.description.public;
                            }
                            if (itemTranslation.description.private) {
                                item.system.description.private = itemTranslation.description.private;
                            }
                        } else {
                            item.system.description = itemTranslation.description;
                        }
                    }

                    // Actions de l'item (réutilise la logique du actions_converter)
                    if (itemTranslation.actions && Array.isArray(item.system?.actions)) {
                        item.system.actions = game.babele.converters.actions_converter(
                            item.system.actions,
                            itemTranslation.actions
                        );
                    }

                    return item;
                });
            },

            // Converteur pour les catégories des journaux
            "categories_converter": (categories, translations) => {
                if (!categories || !translations) return categories;

                return categories.map(item => {
                    const translation = translations[item._id];

                    if (translation) {
                        if (translation.name) item.name = translation.name;
                    }
                    return item;
                });
            },

            /**
             * Converter pour objets directs (non-arrays)
             * Utilisé par : ancestry, background, biography, details, etc.
             */
            "nested_object_converter": (obj, translations) => {
                if (!obj || !translations || typeof translations !== 'object') return obj;

                // Applique les traductions sur l'objet
                Object.keys(translations).forEach(key => {
                    if (translations[key]) {
                        obj[key] = translations[key];
                    }
                });

                return obj;
            }
        });

        console.log('Crucible cn | 模组已加载');
    }
});

Hooks.once('i18nInit', () => {
    game.i18n.translations.Sort = "Sort";
    game.i18n.translations.sort = "tri"; // garde la minuscule pour le core
});
