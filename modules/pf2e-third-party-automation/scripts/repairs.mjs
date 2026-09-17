const GREAT_SWORD_SOURCE = "Compendium.pf2e-team-plus-magic.items.Item.xG5u93m95hjqPR4b";
const VOID_WARP_SOURCE = "Compendium.pf2e-team-plus-magic.items.Item.g2xM25qb607keJxw";
const QI_FEAT_SOURCE = "Compendium.pf2e-team-plus-tian-xia.player-options.Item.ny1FcO87ZDyDQYAu";
const QI_SPELL_SOURCE = "Compendium.pf2e-team-plus-tian-xia.player-options.Item.cRnnxbFQLUzvgnPG";
const WRONG_DRAGONBREATH_FORMULA = "((@item.system.runes.striking + 2) * 2)";
const CORRECT_DRAGONBREATH_FORMULA = "((@item.system.runes.striking + 1) * 2)";
const VOID_WARP_PATH = "system.overlays.np6AuFc56zRKjwBM.system.damage.XoyXJJytPkTqPW6c.formula";

/**
 * Inspect serialized actor data without accessing Foundry or changing the input.
 * Each update includes flat document paths and their exact original values so
 * the caller can back up and recheck the original immediately before applying it.
 * Missing spells are recommendations, never fetched or created by this function.
 */
export function buildRepairPlan(actorData) {
  const plan = { updates: [], missingSpells: [], warnings: [] };
  const items = Array.isArray(actorData?.items) ? actorData.items : [];

  for (const item of items) {
    const source = item?._stats?.compendiumSource;
    if (source === GREAT_SWORD_SOURCE && item.type === "weapon") {
      const description = item.system?.description?.value;
      if (typeof description === "string" && description.includes(WRONG_DRAGONBREATH_FORMULA)) {
        plan.updates.push({
          itemId: item._id,
          reason: "dragonbreath-damage-scaling",
          changes: { "system.description.value": description.replaceAll(WRONG_DRAGONBREATH_FORMULA, CORRECT_DRAGONBREATH_FORMULA) },
          originals: { "system.description.value": description },
        });
      } else if (typeof description !== "string" || !description.includes(CORRECT_DRAGONBREATH_FORMULA)) {
        plan.warnings.push({
          code: "dragonbreath-formula-unrecognized", itemId: item._id,
          message: "龙息武器的伤害公式与已核对版本不符，保留原文。",
        });
      }

      const mismatchedRule = (item.system?.rules ?? []).some((rule) =>
        rule.key === "AdjustModifier" && Array.isArray(rule.predicate) &&
        rule.predicate.some((predicate) => Array.isArray(predicate?.eq) &&
          predicate.eq[0] === "{actor|flags.pf2e.dragonblood.dragon}" && predicate.eq[1] === "adamantine"));
      if (mismatchedRule) {
        plan.warnings.push({
          code: "dragonbreath-dragon-mismatch", itemId: item._id,
          message: "圣天龙息巨剑的龙血判定仍引用 adamantine；尚未确认正确龙种标识，保留规则并交由人工核对。",
        });
      }
    }

    if (source === VOID_WARP_SOURCE && item.type === "spell") {
      const formula = item.system?.overlays?.np6AuFc56zRKjwBM?.system?.damage?.XoyXJJytPkTqPW6c?.formula;
      if (formula === "1d4") {
        plan.updates.push({
          itemId: item._id,
          reason: "dynamic-void-warp-two-action-damage",
          changes: { [VOID_WARP_PATH]: "2d4" },
          originals: { [VOID_WARP_PATH]: "1d4" },
        });
      } else if (formula !== "2d4") {
        plan.warnings.push({
          code: "void-warp-overlay-unrecognized", itemId: item._id,
          message: "动态虚能噬的两动作变体与已核对版本不符，保留原有伤害。",
        });
      }
    }
  }

  const qiFeat = items.find((item) => item?.type === "feat" && item._stats?.compendiumSource === QI_FEAT_SOURCE);
  const hasQiSpell = items.some((item) => item?.type === "spell" && item._stats?.compendiumSource === QI_SPELL_SOURCE);
  if (qiFeat && !hasQiSpell) {
    const candidateEntryIds = items.filter((item) => item?.type === "spellcastingEntry" &&
      item.system?.prepared?.value === "focus" && item.system?.tradition?.value === "occult" &&
      item.system?.ability?.value === "wis").map((entry) => entry._id);
    const spellcastingEntryId = candidateEntryIds.length === 1 ? candidateEntryIds[0] : null;
    plan.missingSpells.push({
      featId: qiFeat._id, sourceUuid: QI_SPELL_SOURCE, spellcastingEntryId, candidateEntryIds,
      reason: "sword-qi-wave-spell-missing",
    });
    if (!spellcastingEntryId) {
      plan.warnings.push({
        code: "sword-qi-focus-entry-unresolved", itemId: qiFeat._id,
        message: "缺少剑气波法术，但无法唯一确定使用感知的异能聚能施法条目。",
      });
    }
  }
  return plan;
}
