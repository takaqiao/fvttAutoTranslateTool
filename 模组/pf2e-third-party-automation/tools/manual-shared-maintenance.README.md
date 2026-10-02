# Manual shared-pool maintenance

This command prepares, checks, installs and restores the paired PF2e and Toolbelt source adapters. It reuses the existing generators. It changes only `systems/pf2e/pf2e.mjs` and `modules/pf2e-toolbelt/scripts/main.js` beneath the supplied Foundry Data directory.

The supported pair is PF2e 8.5.1 with the native IWR bridge baseline and Toolbelt 3.56.5. Both package IDs and versions must match. Accepted SHA256 values are:

| File | Original | Installed |
|---|---|---|
| PF2e | `d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157` | `9be357c96dff3d0790edcb0d7889db98cfded0f41f34fd161233ea0bdd0dab11` |
| Toolbelt | `2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f` | `6937743860c8b2fbeef1037bb99792307cb3fbe7a2540ae0b1ac7010a16a5d90` |

Keep the full tool directory and its pinned batch dependencies together. The Toolbelt observer is checked out with LF so the Windows checkout generates the same verified output.

From this module directory:

```powershell
node tools/manual-shared-maintenance.mjs status --data-root 'D:/Foundry/Data'
node tools/manual-shared-maintenance.mjs prepare --data-root 'D:/Foundry/Data' --plan-dir 'D:/PrivateEvidence/shared-pool-2026-10-02'
```

`status` reads the fixed pair and reports `original`, `installed` or `partial`. Unknown versions or source bytes are rejected. `prepare` requires both originals and a new private plan directory outside the checkout and Data directory. Its parent directory must already exist. It saves the original files, generated candidates, target list, source/output hashes and tool identities there. Full third-party source stays in that private directory; do not add it to Git or a release ZIP.

Stop the Foundry service and clients before installing or restoring. The following assertion records the operator's statement; the command does not detect Setup, running processes or connected clients.

```powershell
node tools/manual-shared-maintenance.mjs apply --data-root 'D:/Foundry/Data' --plan-dir 'D:/PrivateEvidence/shared-pool-2026-10-02' --operator-assertion foundry-and-clients-stopped
node tools/manual-shared-maintenance.mjs restore --data-root 'D:/Foundry/Data' --plan-dir 'D:/PrivateEvidence/shared-pool-2026-10-02' --operator-assertion foundry-and-clients-stopped
```

Before either target changes, the command verifies the complete plan, current tool files, originals, regenerated candidates, package versions and both current target hashes. An installed pair is idempotent. Restore accepts a complete or partial installation only while each file is still its saved original or this plan's exact output. Unknown edits are refused before either file changes.

The two writes are sequential. An IO failure can leave a partial installation. The command keeps the originals, candidates and start record, and attempts to save a separate `failed-*.json` with the observed hashes. It does not silently roll back or retry. After reviewing the evidence and current pair, restore the same plan; if either file has unknown bytes, preserve it and resolve that edit before proceeding.

Retain the existing managed IWR startup configuration. Its rebuild path must regenerate this exact PF2e output. Restart through the established service procedure and refresh all clients after maintenance; this tool does not perform or validate those operations. Retire the pair only after upstream interfaces provide and verify the same original source, patient, receiver selection and OWNER/GM completion evidence.

The isolated regression uses fixed authorized originals supplied explicitly:

```powershell
$env:PF2E_MANUAL_POOL_BATCH_SOURCE = '<fixed-P2-pf2e.mjs>'
$env:TOOLBELT_MANUAL_SOURCE = '<fixed-T0-main.js>'
node --test tests/manual-shared-maintenance.test.mjs
```

`FVTT_PF2E_BUNDLE` may supply the PF2e path when the specific variable is absent. Missing inputs fail the test; there are no source skips. Tests generate the actual verified pair, use temporary Data roots, exercise plan refusal/partial restoration and retain a world sentinel. This proves the local maintenance path, not a live deployment or native-world acceptance.
