/**
 * PF2e Saving Throw Definitions
 * Provides metadata for the three PF2e saves: Fortitude, Reflex, Will
 */

/**
 * Core save definitions with display names, icons, and abbreviations
 */
export const PF2E_SAVES = {
  fortitude: {
    name: '强韧',
    icon: 'fa-heart-pulse',
    abbreviation: '强韧',
  },
  reflex: {
    name: '反射',
    icon: 'fa-wind',
    abbreviation: '反射',
  },
  will: {
    name: '意志',
    icon: 'fa-brain',
    abbreviation: '意志',
  },
};

/**
 * Short display names for saves
 * Used in compact UI displays
 */
export const PF2E_SAVE_SHORT_NAMES = {
  fortitude: '强韧',
  reflex: '反射',
  will: '意志',
};

/**
 * Save name to slug mapping
 * Maps various save name formats to canonical slugs
 */
export const PF2E_SAVE_NAME_MAP = {
  fortitude: 'fortitude',
  reflex: 'reflex',
  will: 'will',
  // Lowercase variations
  fort: 'fortitude',
  ref: 'reflex',
  wil: 'will',
  // Chinese aliases
  强韧: 'fortitude',
  反射: 'reflex',
  意志: 'will',
};
