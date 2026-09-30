/**
 * Supply the former module API location for Team+ consumers that run after setup.
 * Dailies 4.20 exposes game.dailies.api; older Team+ reads the module's .api.
 * This alias does not register dailies or replay an already-failed init hook.
 */
export function installDailiesCompatibility(game) {
  const dailies = game?.modules?.get("pf2e-dailies");
  if (!dailies?.active) return { status: "inactive" };
  if (dailies.api != null) return { status: "existing-api" };

  const api = game.dailies?.api;
  if (api == null) return { status: "unavailable" };

  dailies.api = api;
  return { status: "installed" };
}
