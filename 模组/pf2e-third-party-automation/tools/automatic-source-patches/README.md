# Automatic source patch maintenance

The managed Linux startup entry restores the necessary PF2e, Toolbelt and Patreon source observations before Foundry loads them. Matching original source is patched automatically. Package versions and full-file hashes describe the input; they do not authorize a patch. A changed critical seam is reported for adaptation.

The fixed targets under the configured Foundry data path are:

| Group | File | Observation |
|---|---|---|
| native | `Data/systems/pf2e/pf2e.mjs` | IWR continuation and original shared-pool application |
| native | `Data/modules/pf2e-toolbelt/scripts/main.js` | original owner/GM shared-pool completion |
| patreon | `Data/modules/patreon-v3/src/index.js` | original world-time effects and treatment immunity completion |

Both native candidates must match before either native file changes. Patreon is independent. Existing runtime hooks remain in this module. These file observations retain original input, returned Promises and private handlers that the public hooks do not expose reliably.

Already patched source is skipped. Before replacing source, the entry saves original bytes in `Backups/pf2e-third-party-patches`, checks generated syntax and rechecks the source, package manifests and idle service guard. Replacements use temporary files in the source directory. A pending record contains only changed members of this fixed group. If a write fails or startup was interrupted between native replacements, the next idle start restores the known originals before rebuilding. An entirely installed pending group is finalized. Unknown later edits or corrupt backups stop that group's recovery and are preserved for inspection.

Logs use `pf2e-third-party-source-patches` with `patched`, `unchanged`, `needs-adaptation`, `unavailable` or `failed`, the affected file paths and reason. A patch failure leaves unrelated initialization available. A missing exclusive startup lock or a busy installation prevents startup maintenance and entry into Foundry.

## Server connection

Browser clients cannot write these server files. The administrator connects this entry once to a Linux service that holds `flock -n -E 78 -F <dataPath>/Config/.pf2e-iwr-startup.lock` across maintenance and the original Foundry process. `startup.mjs` uses the existing managed startup config and original PM2 bootstrap. Keep the original Node arguments, Foundry arguments, working directory, PM2 container and five saved environment fields. The legacy `--config-sha` argument is retained as a diagnostic; normal updates do not require a new source hash or an approval prompt.

The small service launcher imports `tools/automatic-source-patches/startup.mjs` from the installed module, then calls `runStartup(parseArguments(process.argv.slice(2)))`. The installed module must retain this whole tools directory, its observer components and their browser-side source qualification helpers. The module update supplies the next startup implementation. Do not execute the maintenance entry from an active Foundry process or replace the idle guard with an operator assertion.

Normal third-party updates are recovered on the next server start. Refresh browser clients afterward. Activity leases, rolls, world time, HP, immunities and settled or unknown results are not replayed by this entry. No paid third-party bundle is included in the release.
