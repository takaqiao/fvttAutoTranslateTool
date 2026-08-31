/**
 * Journal Integration Handler for GM Sidebar
 * Handles journal check extraction, highlighting, and content monitoring
 */

import { extractParentElement } from '../../../utils/element-utils.mjs';
import { findJournalContent } from '../../../utils/dom-utils.mjs';
import * as SystemAdapter from '../../../system-adapter.mjs';

/**
 * Extract checks from parent journal content
 * Returns checks grouped by skill
 */
export function extractJournalChecks(sidebar) {
  if (!sidebar.parentInterface?.element) {
    return [];
  }

  const content = getJournalContent(sidebar);
  if (!content) {
    return [];
  }

  const checks = sidebar._parseChecksFromContent(content);
  const grouped = groupChecksBySkill(checks);
  return grouped;
}

/**
 * Extract checks from parent journal (ungrouped version for backward compat)
 */
export function extractJournalChecksFlat(sidebar) {
  const grouped = extractJournalChecks(sidebar);
  const flat = [];
  grouped.forEach((group) => flat.push(...group.checks));
  return flat;
}

/**
 * Get journal content element (helper)
 * Returns the container with all journal content for check parsing
 */
export function getJournalContent(sidebar) {
  if (!sidebar.parentInterface) return null;
  const element = extractParentElement(sidebar.parentInterface);
  if (!element || typeof element.querySelector !== 'function') return null;
  return findJournalContent(element);
}

/**
 * Group checks by skill (common logic)
 */
export function groupChecksBySkill(checks) {
  // Remove duplicates
  const unique = [];
  const seen = new Set();
  checks.forEach((check) => {
    const key = `${check.skillName}-${check.dc}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(check);
    }
  });

  // Group by skill type
  const grouped = {};
  unique.forEach((check) => {
    const skill = check.skillName;
    // Get full display name — saves and skills live in different lookup tables
    const isSave = check.checkType === 'save';
    let skillDisplay, skillSlug;
    if (isSave) {
      const normalized = skill.toLowerCase();
      skillSlug = SystemAdapter.getSaveSlugFromName(skill) || normalized;
      const saveData = SystemAdapter.getSaves()[skillSlug] || SystemAdapter.getSaves()[normalized];
      skillDisplay = saveData?.name || (skill.charAt(0).toUpperCase() + skill.slice(1));
    } else {
      const normalized = skill.toLowerCase();
      skillSlug = SystemAdapter.getSkillSlugFromName(skill) || normalized;
      const skillData = SystemAdapter.getSkills()[skillSlug] || SystemAdapter.getSkills()[normalized];
      skillDisplay = skillData?.name || (skill.charAt(0).toUpperCase() + skill.slice(1));
    }
    const groupKey = `${isSave ? 'save' : 'skill'}:${skillSlug}`;
    if (!grouped[groupKey]) {
      grouped[groupKey] = {
        skillName: skillDisplay,
        skillSlug: skillSlug,
        checks: [],
      };
    }
    grouped[groupKey].checks.push(check);
  });

  // Sort groups alphabetically by skill name (A-Z)
  return Object.values(grouped).sort((a, b) =>
    a.skillName.localeCompare(b.skillName)
  );
}

/**
 * Extract the normalised skill name (display name, lowercase) and DC string from an
 * inline-check element, supporting both PF2e and D&D 5e enricher formats.
 * Returns { skillName, dc } or null if the element can't be parsed.
 */
function _getCheckElementInfo(el) {
  const saveTypes = new Set(['fortitude', 'reflex', 'will', 'str', 'dex', 'con', 'int', 'wis', 'cha']);

  // PF2e format: a.inline-check[data-pf2-check][data-pf2-dc]
  if (el.dataset.pf2Check) {
    const rawName = el.dataset.pf2Check.toLowerCase();
    const isSave = saveTypes.has(rawName);
    const skillKey = isSave
      ? (SystemAdapter.getSaveSlugFromName(rawName) || rawName)
      : (SystemAdapter.getSkillSlugFromName(rawName) || rawName);
    return { skillKey, dc: el.dataset.pf2Dc };
  }

  // D&D 5e format: span.roll-link-group[data-type][data-skill/data-ability][data-dc]
  const isSave = el.dataset.type === 'save';
  const rawSlug = isSave
    ? el.dataset.ability
    : (el.dataset.skill || el.dataset.ability);
  const dc = el.dataset.dc;
  if (!rawSlug || !dc) return null;

  // Use the first slug for pipe-separated groups (e.g., "acr|ath")
  const firstSlug = rawSlug.split('|')[0].trim().toLowerCase();

  const skillKey = isSave
    ? (SystemAdapter.getSaveSlugFromName(firstSlug) || firstSlug)
    : (SystemAdapter.getSkillSlugFromName(firstSlug) || firstSlug);

  return { skillKey, dc };
}

function _isScrollableElement(el) {
  if (!el) return false;
  const style = window.getComputedStyle(el);
  const canScroll = /(auto|scroll)/.test(style.overflowY || '');
  return canScroll && el.scrollHeight > el.clientHeight + 2;
}

function _findJournalScrollContainer(parentElement) {
  if (!parentElement) return null;

  const preferred = [
    '.journal-entry-pages',
    '.journal-entry-content',
    '.scrollable',
    '.window-content',
  ];

  for (const selector of preferred) {
    const el = parentElement.querySelector(selector);
    if (el && _isScrollableElement(el)) return el;
  }

  const candidates = Array.from(
    parentElement.querySelectorAll('.scrollable, .window-content, .journal-entry-pages, .journal-entry-content, .journal-page-content')
  );
  const found = candidates.find(_isScrollableElement);
  if (found) return found;

  // Fallback to common containers even if not scrollable at init time.
  for (const selector of preferred) {
    const el = parentElement.querySelector(selector);
    if (el) return el;
  }

  return parentElement;
}

function _collectInlineCheckElements(rootElement) {
  if (!rootElement) return [];
  const pf2eElements = rootElement.querySelectorAll('a.inline-check[data-pf2-check][data-pf2-dc]');
  const dnd5eElements = rootElement.querySelectorAll(
    'span.roll-link-group[data-type="check"], span.roll-link-group[data-type="skill"], span.roll-link-group[data-type="save"]',
  );
  return [...pf2eElements, ...dnd5eElements];
}

function _findScrollAncestorFromNode(node, boundaryRoot) {
  let current = node?.parentElement || null;
  while (current && current !== boundaryRoot) {
    if (_isScrollableElement(current)) return current;
    current = current.parentElement;
  }
  return null;
}

/**
 * Setup IntersectionObserver to highlight journal check buttons when checks are in view
 */
export function setupJournalCheckHighlighting(sidebar) {
  // Clean up existing observer and scroll listener
  if (sidebar._checkObserver) {
    sidebar._checkObserver.disconnect();
    sidebar._checkObserver = null;
  }
  if (sidebar._checkScrollContainer && sidebar._checkScrollHandler) {
    sidebar._checkScrollContainer.removeEventListener('scroll', sidebar._checkScrollHandler);
  }
  if (sidebar._checkResizeHandler) {
    window.removeEventListener('resize', sidebar._checkResizeHandler);
  }
  sidebar._checkScrollContainer = null;
  sidebar._checkScrollHandler = null;
  sidebar._checkResizeHandler = null;
  if (sidebar._scrollCheckTimeout) {
    clearTimeout(sidebar._scrollCheckTimeout);
    sidebar._scrollCheckTimeout = null;
  }

  if (!sidebar.parentInterface?.element) return;

  // Get the journal's scrollable container
  const parentElement = extractParentElement(sidebar.parentInterface);

  if (!parentElement) return;

  const contentRoot = getJournalContent(sidebar) || parentElement;

  // Collect checks from journal content first; fallback to full parent root
  let checkElements = _collectInlineCheckElements(contentRoot);
  if (checkElements.length === 0) {
    checkElements = _collectInlineCheckElements(parentElement);
  }

  // Initialize visible checks map even if no checks, so stale highlights are cleared
  sidebar._visibleChecksMap = new Map();

  // Helper to update button highlights
  const updateButtonHighlights = () => {
    if (!sidebar.element) return;

    // Store for popup highlighting
    sidebar._visibleChecks = new Map(sidebar._visibleChecksMap);

    // Update button highlight state for both skill and save buttons
    const buttons = sidebar.element.querySelectorAll('.journal-skill-btn, .journal-save-btn');
    buttons.forEach((btn) => {
      const checkKey = btn.dataset.skill || btn.dataset.save;
      if (checkKey && sidebar._visibleChecksMap.has(checkKey)) {
        btn.classList.add('in-view');
      } else {
        btn.classList.remove('in-view');
      }
    });
  };

  const scheduleRecheck = () => {
    if (sidebar._scrollCheckTimeout) {
      clearTimeout(sidebar._scrollCheckTimeout);
    }
    sidebar._scrollCheckTimeout = setTimeout(() => {
      forceCheckVisibility(sidebar, scrollContainer, checkElements, updateButtonHighlights);
    }, 40);
  };

  if (checkElements.length === 0) {
    updateButtonHighlights();
    return;
  }

  // Find the actual scroll container used by this journal sheet implementation
  const derivedContainer = _findScrollAncestorFromNode(checkElements[0], parentElement);
  const scrollContainer = derivedContainer || _findJournalScrollContainer(parentElement) || contentRoot;

  // Create observer with a small buffer to reduce edge flickering
  sidebar._checkObserver = new IntersectionObserver(
    (entries) => {
      if (!entries?.length) return;
      scheduleRecheck();
    },
    {
      root: scrollContainer,
      // Small buffer to reduce flickering at viewport edges
      rootMargin: '10px 0px',
      threshold: [0, 0.1, 0.5, 1.0],
    }
  );

  // Observe all check elements
  checkElements.forEach((el) => sidebar._checkObserver.observe(el));

  // Fallback: deterministic recompute on scroll/resize (fixes custom layout observer edge cases)
  sidebar._checkScrollContainer = scrollContainer;
  sidebar._checkScrollHandler = scheduleRecheck;
  sidebar._checkResizeHandler = scheduleRecheck;
  scrollContainer.addEventListener('scroll', scheduleRecheck, { passive: true });
  window.addEventListener('resize', scheduleRecheck, { passive: true });

  // Force an initial highlight update after a short delay to ensure DOM is stable
  // This catches cases where the observer's initial callback fires before layout is complete
  sidebar._scrollCheckTimeout = setTimeout(() => {
    forceCheckVisibility(sidebar, scrollContainer, checkElements, updateButtonHighlights);
  }, 100);
}

/**
 * Force a manual visibility check as a fallback when IntersectionObserver might miss updates
 */
export function forceCheckVisibility(sidebar, scrollContainer, checkElements, updateCallback) {
  if (!scrollContainer || !checkElements || !sidebar._visibleChecksMap) return;

  const containerRect = scrollContainer.getBoundingClientRect();
  sidebar._visibleChecksMap.clear();

  checkElements.forEach((el) => {
    const info = _getCheckElementInfo(el);
    if (!info) return;
    const { skillKey, dc } = info;

    const elRect = el.getBoundingClientRect();

    // Check if element is visible within the scroll container (with small buffer)
    const isVisible = elRect.top < containerRect.bottom + 10 &&
      elRect.bottom > containerRect.top - 10 &&
      elRect.left < containerRect.right &&
      elRect.right > containerRect.left;

    if (isVisible) {
      if (!sidebar._visibleChecksMap.has(skillKey)) {
        sidebar._visibleChecksMap.set(skillKey, new Set());
      }
      sidebar._visibleChecksMap.get(skillKey).add(dc);
    }
  });

  // Always call updateCallback on force check - this ensures buttons are updated
  // even if the IntersectionObserver already populated the map
  updateCallback();
}

/**
 * Setup MutationObserver to detect when new journal pages load (multi-page journals)
 * and trigger a re-render to pick up new images/actors
 */
export function setupJournalContentObserver(sidebar) {
  // Clean up existing observer
  if (sidebar._journalContentObserver) {
    sidebar._journalContentObserver.disconnect();
    sidebar._journalContentObserver = null;
  }
  if (sidebar._journalContentDebounce) {
    clearTimeout(sidebar._journalContentDebounce);
    sidebar._journalContentDebounce = null;
  }

  if (!sidebar.parentInterface?.element) return;

  const parentElement = extractParentElement(sidebar.parentInterface);

  // Watch the journal pages container for new content
  const pagesContainer = parentElement.querySelector('.journal-entry-pages') ||
    parentElement.querySelector('.journal-entry-content') ||
    parentElement;

  if (!pagesContainer) return;

  // Track how many page content elements we've seen
  let lastPageCount = parentElement.querySelectorAll('.journal-page-content').length;

  sidebar._journalContentObserver = new MutationObserver((mutations) => {
    // Check if any new .journal-page-content elements were added
    let hasNewPages = false;
    for (const mutation of mutations) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            if (node.classList?.contains('journal-page-content') ||
              node.querySelector?.('.journal-page-content')) {
              hasNewPages = true;
              break;
            }
          }
        }
      }
      if (hasNewPages) break;
    }

    // Also check if page count increased (covers nested additions)
    const currentPageCount = parentElement.querySelectorAll('.journal-page-content').length;
    if (currentPageCount > lastPageCount) {
      hasNewPages = true;
      lastPageCount = currentPageCount;
    }

    if (hasNewPages) {
      // Debounce re-render to avoid multiple rapid updates
      if (sidebar._journalContentDebounce) {
        clearTimeout(sidebar._journalContentDebounce);
      }
      sidebar._journalContentDebounce = setTimeout(() => {
        sidebar.render();
      }, 150);
    }
  });

  sidebar._journalContentObserver.observe(pagesContainer, {
    childList: true,
    subtree: true,
  });
}

/**
 * Cleanup journal observers
 */
export function cleanupJournalObservers(sidebar) {
  // Clean up IntersectionObserver and timeout
  if (sidebar._checkObserver) {
    sidebar._checkObserver.disconnect();
    sidebar._checkObserver = null;
  }
  if (sidebar._checkScrollContainer && sidebar._checkScrollHandler) {
    sidebar._checkScrollContainer.removeEventListener('scroll', sidebar._checkScrollHandler);
  }
  if (sidebar._checkResizeHandler) {
    window.removeEventListener('resize', sidebar._checkResizeHandler);
  }
  sidebar._checkScrollContainer = null;
  sidebar._checkScrollHandler = null;
  sidebar._checkResizeHandler = null;
  if (sidebar._scrollCheckTimeout) {
    clearTimeout(sidebar._scrollCheckTimeout);
    sidebar._scrollCheckTimeout = null;
  }
  sidebar._visibleChecksMap = null;

  // Clean up journal content observer
  if (sidebar._journalContentObserver) {
    sidebar._journalContentObserver.disconnect();
    sidebar._journalContentObserver = null;
  }
  if (sidebar._journalContentDebounce) {
    clearTimeout(sidebar._journalContentDebounce);
    sidebar._journalContentDebounce = null;
  }
}
