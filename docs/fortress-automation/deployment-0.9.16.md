# 0.9.16 deployment verification

[0.9.16](https://github.com/takaqiao/fvttAutoTranslateTool/releases/tag/pf2e-third-party-automation-v0.9.16) was published and installed successfully on 2026-09-23 at 12:43:30 UTC.

- Release commit: `3c8fb356dc935d7d7b3ad6ddd6910e209a46b571`.
- The public manifest and ZIP were downloaded and verified against the annotated release tag and local committed package. All 134 runtime files matched the ZIP inventory.
- Deployment changed README, module.json, the two combination providers and added the shared activity sequence helper. No runtime file was removed.
- The transaction verified a complete 0.9.15 backup, atomically replaced the module and refreshed Foundry's native module package cache. Its retained receipt is `installed`.
- Native package version and every one of the 134 served HTTP files matched 0.9.16 afterward.
- Foundry remained in Setup with no world users. Both service identities, startup settings, 130 fortress world files, protected core/system files and other modules matched the fresh deployment baseline. No service restart or world launch was performed.
- Release regression: 1,676 passed, zero failed or skipped. Fourteen native weapon-trait cases passed; the three changed runtime scripts match that QA source after Git line-ending normalization.
- Deployment helper checks: 17 transaction tests and 7 archive-validation tests passed.

Detailed local receipts, checksums and backup information are retained under `C:/Users/Taka/Desktop/fvtt/output/automation-release-0916-20260923`. This completes the user-authorized publication and deployment of the activity-local weapon-trait changes.
