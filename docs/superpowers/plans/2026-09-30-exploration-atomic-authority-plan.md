# Exploration unique execution: stage-one implementation plan

This implements the approved exploration design's unique execution requirement. It does not replace the full quality goal: functionality, twelve native GM/player acceptance scenarios, full-module review, measured performance work and coordinated release remain required.

Baseline: `200bc400a439b58dd2d82ddab6a588e7816a1ed1` (.23). The preceding narrow cancellation, receiving-context, eligibility and manual-evidence repairs are verified commit `e2a2c90667677bab2f8653d9ab76387726f789b8` (2338/2338 source-enabled tests, zero skips). Module version is not bumped merely for starting this plan.

## Evidence and boundaries

- Two client-local ledgers can both acquire one completing activity; two owner runtimes can execute it twice. A local promise queue, GM user ID and actor flag do not provide global exclusivity.
- Installed Foundry 14.368 serializes database operations and rejects a fixed-ID embedded JournalEntryPage which already exists in its current parent. Top-level fixed-ID Journal creation can overwrite and must not be used as a lock.
- The actual isolated characterization is in `output/exploration-quality-goal-20260930/qa`. The first run established page uniqueness in the same socket, same-GM sockets and different-GM sockets, but recorded a native client broadcast exception. Its failure is preserved. After source/evidence diagnosis and explicit no-broadcast raw transport, `atomic-primitive-report-2026-09-30T16-16-10-459Z-123d8d76.json` passed all eight checks with no page errors; the matching JSONL records exact requests/acknowledgements/server reads. This establishes the storage primitive, not the product execution protocol or real packet-loss recovery.
- No native API offers a transaction combining journal state, world time, dice and HP. Persist a one-use grant before issuing native work. Unknown outcomes remain unresolved and are never automatically replayed. Already-issued native calls cannot be retroactively revoked by a client-side lock.

## Architecture

1. Pin one explicitly configured, private Journal parent. Execution never silently creates a new root. Migrate a configured legacy ledger using one fixed genesis page. Missing or ambiguous roots stop before native work. New-world setup is a separate visible initialization operation while recovery is idle; it must not claim unattended atomic root election.
2. Commit each revision by creating one immutable embedded page with a deterministic, valid sixteen-character revision ID and `keepId:true`. Metadata binds root, schema, predecessor, writer, client and fresh commit nonce. Genesis contains the legacy seed; later pages contain deterministic changed records and a digest chain, avoiding a complete growing history snapshot in every revision. Validate contiguous revisions, exact predecessor digests and document provenance. Detectable mutation/gaps and regression below this client's known head fail closed. The protocol assumes cooperating GMs do not externally edit/delete committed pages: a cold client cannot distinguish an entirely deleted valid suffix from an earlier legitimate prefix without an independent nonrollbackable head. Do not claim the digest chain protects that case. Never delete pages or reuse IDs in module code; protect ordinary UI deletion and state this limitation explicitly.
3. Use actual authenticated server reads for initial state and after a proven duplicate-ID conflict. A pure mutation can refresh and recompute after that conflict. Empty acknowledgements, disconnects, timeouts and other errors do not grant execution, even if a later read finds the submitted nonce. Never write the legacy full-state flag as authority after migration.
   Raw revision requests explicitly set `broadcast:false`, and validate this in the acknowledgement. Raw socket creation does not populate the sender's native document cache; broadcasting another tab's embedded page into that missing parent can throw in Foundry's native client backend. The protocol projects state from server reads rather than injecting fake client Documents. Peer refresh/stop notification is a separate authenticated, limited-payload signal and cannot grant execution or replace a server read.
4. Move invariants into the atomic mutation: one automatic running session; session status/cursor and unresolved-clock checks; legal expected activity transitions; private driver/executor ownership; unique owner execution. Any retry callback is synchronous and free of native effects. Preserve changes to unrelated sessions and manual evidence.
5. Bind a running coordinator to a randomly generated runtime identity. Only the successful private continuation drives that session. Another tab observing a running session cannot step or stop it merely because its own local maps are empty. Restore is read-only until explicit takeover: that atomic operation pauses and quarantines unresolved grants, without manufacturing a new executable permit. A vanished local context is not proof that another tab is dead. Explicit Stop invalidates local scopes and broadcasts the persisted stopped state.
6. Local GM and remote owner execution use the same ledger one-use grant. Each receiving owner generates its own private attempt nonce and obtains an authenticated GM broker claim. Grant only the exact winning local attempt. Socketlib resolves the first RESULT for a request addressed to a user with multiple tabs: only the persisted session's private driver runtime may respond as broker. Other tabs must not race a rejection against the winner. Implement explicit request routing/final-response semantics, bounded cleanup and unknown-driver timeout. Use the native authenticated module channel with server-side `{recipients:[userId]}` for both request and response; ordinary socketlib message recipients are client filters and its transport broadcasts to all sockets. Send only necessary permit identity fields, never the private ledger/full activity. The actor execution flag is evidence, not a mutex. Bind saved completion to the permit and native activity/source identity.

### Codec and state contract

- Allowed state collections are `sessions`, `activities` and `clocks`; each maps validated nonprototype string keys to complete JSON records. Genesis stores these collections once. A successor stores whole changed records in `changes`, not patches to array positions. Module code never deletes records, pages or predecessor references. Any unsupported collection/deletion fails before submission.
- Canonical JSON sorts object keys, preserves dense array order, drops undefined object properties consistently with native document JSON, and rejects unsupported types/nonfinite numbers. Normalize state and metadata before diff/digest; readers use that same reducer. A retry recomputes from the winning complete state, rather than applying an obsolete delta.
- Schema version, root UUID, genesis epoch, revision, predecessor digest, writer/connection identity, fresh nonce and changes are covered by the page digest. The digest excludes the digest field itself. Genesis additionally covers the legacy migration seed and its fixed source fingerprint. Page ID encodes the safe integer revision without truncation or hashing collisions.
- Every explicit successful acknowledgement must match submitted root, epoch, revision, nonce and digest. No response path may derive an executable grant merely from an existing page or actor flag. Storage and broker calls each latch expired/consumed state; a late acknowledgement remains evidence only.
- Saved native completion must match the exact protocol/root/epoch/activity/operation/owner runtime/attempt/permit plus its actual source receipts. Legacy actor completion can remain historical evidence but cannot confer a new protocol grant.
- Review/close may annotate unresolved records, but cannot release time/HP domains while a previously granted issuer could still act. A timeout, missing local map or absent heartbeat is not quiescence. Define explicit provable settlement/quiescence criteria before permitting another automatic session; otherwise keep the affected domain blocked.
- Provision/migration never starts recovery automatically. It requires recovery disabled, old issuers stopped and all relevant clients reloaded into the protocol. Pin root and epoch for the runtime; configuration changes block the old runtime rather than silently switching roots. Ambiguous candidate roots remain intact for explicit selection. These are controlled administrative prerequisites, not an atomic unattended root election.

## Implementation sequence

### 1. Characterize the server and retain evidence

Resolve the first runner broadcast failure from actual source/events, then repeat independent same-GM and different-GM contexts. Keep requests, responses, persisted server reads, socket identities and runner hashes. Test stale loser, independent pure-field retry, suppressed acknowledgement and empty result. Describe the harness acknowledgement suppression accurately; it is not a production packet-loss test.

### 2. Implement immutable revision storage

Add a small revision codec and append-only store. First add failing behavioral tests using two independent stores and a shared server-semantic fixture. Cover winner/loser, preserved unrelated fields, unknown acknowledgement, empty result, corrupt/gapped chain, private root validation and migration races. The fixture must enforce uniqueness at the server boundary rather than sharing a client-local queue. Then implement the production codec/store and run its focused tests. Do not introduce a fallback mutable writer for automatic recovery.

### 3. Enforce ledger and driver invariants

Add storage transactions to the ledger without side effects in retryable callbacks. Add tests for two concurrent session starts, stale clock claims, stopped sessions and one claimed activity. Connect coordinator ownership and restore behavior. A losing peer must not pause the winner's live session. Preserve manual evidence and current stopped/unknown-state semantics.

### 4. Add authenticated one-use owner grants

Test two independent same-owner runtimes with broadcast delivery. Exactly one handler may perform native work. Add wrong sender, wrong actor/owner, stale activity, mismatched attempt, lost response, reconnect and GM-change cases. Implement the broker and bind reconciliation to the persisted executor identity. Do not weaken ordinary player native dice windows.

### 5. Provide controlled initialization and migration

Add a concrete setup path and diagnostics for missing, nonprivate and conflicting parents. Select/configure the root before allowing recovery. Tests must prove conflicting initialization cannot silently proceed under two roots; retain both artifacts for explicit resolution. Validate existing .23 history remains viewable and readable after genesis migration. No automatic deletion, retention reset or revision ID reuse.

### 6. Run actual multi-client product acceptance

Use the isolated runtime with the exact candidate file inventory. Exercise duplicate same-GM drive and same-owner delivery, concurrent state changes, reload/GM takeover, stopped session and unresolved acknowledgement paths. Count actual native effects and inspect persisted grants, HP/time/resource receipts. Repeat ordinary GM/player manual windows. Do not claim the twelve original scenarios are all covered by these narrower authority tests.

### 7. Integrate and independently review

Run source-enabled full tests, syntax and diff checks once all writers are idle. Obtain independent storage/protocol and owner/native reviews; repair confirmed issues and rerun relevant checks. Save source commit and evidence in goal progress. Keep the full goal active and proceed to stage two; coordinate release only after its remaining requirements are complete.

## Acceptance criteria

- Two independent cooperating clients cannot both obtain executable grants for one clock checkpoint or native activity.
- Pure concurrent mutations preserve both accepted changes without an unprotected full-state overwrite.
- Unknown acknowledgement never becomes permission to replay native work.
- Private player data is not exposed to obtain owner execution.
- Stop, external time, encounter, refresh and GM changes preserve uncertainty and exact source evidence.
- New history does not write complete prior history on every revision. Measure actual reads/writes and long-history growth in stage three before claiming a speed improvement.

Open engineering decisions must be resolved against the installed server and native QA evidence. If a proposed operation cannot enforce these criteria, keep it blocked rather than silently restoring .23's unsafe execution path.
