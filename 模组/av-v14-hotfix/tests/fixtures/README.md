# Standalone regression fixtures

The Babele and Chat tests load the module's actual `scripts/patches/babele.mjs` and `scripts/patches/chat.mjs`. Their fixtures contain only the needed original source excerpts, not the full Foundry bundle. No `output` directory, installed Foundry instance, network, or npm dependencies are needed to run them.

From the module directory, using the verified Node 26.7.0 runtime:

```sh
node --test tests/babele-instance.test.mjs tests/babele-boundary.test.mjs tests/chat-coalescing.test.mjs
```

Expected: 27 Babele + 16 Chat = 43 passing tests. Chat's optional `BASELINE` environment variable selects the original behavior for a deliberate regression control; leave it unset for the actual patch tests.

- `babele-core-14.367.json`: StringTree, WordTree, DocumentIndex, and native CompendiumCollection.indexDocument.
- `babele-upstream-3.1.2.json`: the unchanged Chinese translation wrapper and the few helper functions required by the tests.
- `babele-lib-wrapper-1.13.5.1.json`: the actual enum factory and package identity check used by compatibility tests.
- `chat-semaphore.js` / `chat-methods.json`: Semaphore and the three native ChatLog methods used by the tests.
- Each `*-provenance.json` records original source SHA-256, source locations, extraction details, and fragment hashes. Source paths in metadata are historical provenance, not test dependencies.

These Babele/Chat source fixtures occupy 32,401 bytes combined (39,131 bytes including their provenance JSON). Foundry method bodies remain unchanged. Babele extraction normalizes line endings and trims only outer whitespace; Chat uses exact original slices.

Fixtures are committed/generated once. Regeneration alone requires the separately supplied original source files:

```sh
node scripts/build-babele-test-fixtures.mjs /path/to/foundry.mjs /path/to/babele-ondemand-patch.js /path/to/lib-wrapper.js
node scripts/build-chat-test-fixtures.mjs /path/to/foundry.mjs
```

The generators write only the named test fixtures. They do not modify the runtime patches, application data, or original source files.

## PF2e 8.6.0 source capture

`generate-item-name-8.6.0.js.txt` and `tokenizer-chat-native.json` preserve exact official PF2e function excerpts. The latter contains the complete 8.5.1 and 8.6.0 `ChatMessagePF2e.renderHTML` methods. Provenance records release URLs, bundle and fragment SHA-256, and source offsets. Bundle hashes identify the source files; runtime guards hash the target functions.

```sh
node scripts/build-pf2e-test-fixtures.mjs /8.5.1/pf2e.mjs /8.6.0/pf2e.mjs
```

The generator verifies both official bundle hashes before writing. Naming regressions compare native outputs, generated names and map enumeration. Portrait tests execute the complete unmodified render methods with a small DOM fixture and isolated non-portrait listeners; they cover the registered wrapper, hidden/OOC messages, ordinary portraits and mirrored token scale. These checks do not replace browser rendering or third-party wrapper-order QA.

## 0.6.23 source capture

`dsn-queue-6.4.3-native.json` is a separate fixture from the captured 6.4.3 installation. Its provenance records the bundle and manifest bytes and SHA-256, exact queue/Accumulator/DiceBox/ThrowEngine excerpts, and unchanged chat, model and quality fragment contracts. `dsn-queue-native.json` and the other historical fixtures retain their original 6.4.2 source pins.

The 6.4.3 generator verifies `source-capture.json`, `inventory.json` and the manifest before writing the new fixture. It checks every fragment first and writes only the new queue file:

```sh
node scripts/build-dsn-643-test-fixture.mjs /path/to/upstream
```

Both capture files must be alongside `upstream`. Queue completion and chat regressions exercise both source profiles, including asynchronous readiness, failures, late effects and replacement of batch owners during worker cleanup. These fixtures prove isolated behavior and source contracts; they do not measure live-world graphics or frame rate.

`persistent-bridge-0.5.4-native.json` independently preserves the captured PersistentDice adapter, including its actual ready, wrapper and dispose closures. Its generator checks the capture, inventory, manifest and adapter digest before writing this file:

```sh
node scripts/build-persistent-bridge-test-fixture.mjs /path/to/upstream
```

The bridge regressions execute this adapter against both native DsN profiles, in both installation orders, across recovery reinstall and native queue attach after a box rebuild. They preserve the bridge's actual function references and external completion observers. Ticker regressions cover every native animate identity consumer and run the installed Foundry Pixi ticker when `FVTT_NATIVE_APP` is available. The runtime keeps the native prototype function and its closures, with a stable owned instance wrapper for all native ticker registrations.

`dsn-worker-native.json` contains the exact worker RPC `exec` method, independently verified in both captured bundles. `persistent-bridge-settings-0.5.4-native.json` preserves the bridge's actual enable/disable factory, setting registration and constants from the same 0.5.4 capture. Their generators leave the existing queue and adapter fixtures unchanged:

```sh
node scripts/build-dsn-worker-test-fixture.mjs /current/upstream /legacy/upstream
node scripts/build-persistent-bridge-settings-test-fixture.mjs /current/upstream
```

Lifecycle regressions execute actual adapter ready/dispose during simulate, playback, effects, collision and position waits. The setting cases run the captured registered `enabled` onChange callback through the actual bridge factory and adapter, with controlled settings dispatch and small UI resource stubs. They cover successful and rejected batches, both installation orders, coherent off/on transitions and unknown ownership changes. They do not execute the whole module entry's synchronization or a rendered tray; those paths require browser QA.

## 0.6.22 source capture

The DsN chat, model and queue fixtures are exact excerpts from the remote installation's 6.4.2 `main.js`, verified against `source-capture.json`. Their required functions and completion consumers match 6.4.1. The DiceConfig `_prepareContext` excerpt was updated for 6.4.2's medium shadow choice and hidden-die filtering. The associated metadata records original bundle SHA-256, fragment offsets and fragment SHA-256; bundle hashes never authorize a runtime patch.

`bbmm-rules-native.json` includes only BBMM 1.4.11's two setting registrations and the native rule writer needed to prove the reader contract. `turn-lifecycle-native.json` records the current remote Reaction 1.4.3, Sustain 1.1.0 and Summons 2.20.2 callback provenance. Historical core and Toolbelt excerpts keep their original provenance.

The historical generator requires the original capture schema with a `versions` map. Do not use it to rewrite these source pins from a newer capture:

```sh
node scripts/build-current-test-fixtures.mjs /path/to/upstream
```

The capture's `source-capture.json` must be alongside `upstream`. Only the required function excerpts are published with the hotfix; third-party source bundles are not included.

## 0.6.2 additions

`duration-core-14.367.json` is the complete unchanged `CalendarData.formatDuration.toString()` captured from the isolated QA browser, with its SHA-256. It also matches the saved core source at line 82039. The test provides core's lazy `objectEntries`/`iterateEntries` helpers and the original duration unit set; native `Intl.DurationFormat` performs actual formatting.

`patreon-3.2.28-refresh.json` contains only the three relation helpers, listener, outer callback and closed-array declaration needed for regression tests, with original full-bundle and prior source-patch digests. The full paid module bundle is not included. `hook-harness.mjs` shares the existing exact core Hooks fixture between Sundry, Patreon and broker lifecycle tests.
