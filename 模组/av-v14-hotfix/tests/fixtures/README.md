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

## 0.6.22 source capture

The DsN chat, model and queue fixtures are exact excerpts from the remote installation's 6.4.2 `main.js`, verified against `source-capture.json`. Their required functions and completion consumers match 6.4.1. The DiceConfig `_prepareContext` excerpt was updated for 6.4.2's medium shadow choice and hidden-die filtering. The associated metadata records original bundle SHA-256, fragment offsets and fragment SHA-256; bundle hashes never authorize a runtime patch.

`bbmm-rules-native.json` includes only BBMM 1.4.11's two setting registrations and the native rule writer needed to prove the reader contract. `turn-lifecycle-native.json` records the current remote Reaction 1.4.3, Sustain 1.1.0 and Summons 2.20.2 callback provenance. Historical core and Toolbelt excerpts keep their original provenance.

To regenerate these fixtures from a separately captured remote installation:

```sh
node scripts/build-current-test-fixtures.mjs /path/to/upstream
```

The capture's `source-capture.json` must be alongside `upstream`. Only the required function excerpts are published with the hotfix; third-party source bundles are not included.

## 0.6.2 additions

`duration-core-14.367.json` is the complete unchanged `CalendarData.formatDuration.toString()` captured from the isolated QA browser, with its SHA-256. It also matches the saved core source at line 82039. The test provides core's lazy `objectEntries`/`iterateEntries` helpers and the original duration unit set; native `Intl.DurationFormat` performs actual formatting.

`patreon-3.2.28-refresh.json` contains only the three relation helpers, listener, outer callback and closed-array declaration needed for regression tests, with original full-bundle and prior source-patch digests. The full paid module bundle is not included. `hook-harness.mjs` shares the existing exact core Hooks fixture between Sundry, Patreon and broker lifecycle tests.
