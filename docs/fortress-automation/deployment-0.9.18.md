# 0.9.18 deployment verification

[0.9.18](https://github.com/takaqiao/fvttAutoTranslateTool/releases/tag/pf2e-third-party-automation-v0.9.18) was published and installed successfully at 2026-09-24T19:09:22.728Z (2026-09-25 03:09:22 Asia/Shanghai).

- Release commit: `d26963ac370dca269f32a09f64e9bb3ada0b2332`.
- GitHub latest is 0.9.18. The downloaded public manifest and ZIP match the annotated release tag, committed payload and asset checksums. All 137 runtime files matched the archive inventory and the served production HTTP bytes.
- Runtime delta: nine changed files, three added settlement helpers, no removed files.
- A fresh Setup transaction verified a complete 0.9.17 backup, atomically replaced only this module and refreshed native package metadata. The retained receipt is installed; native package version is 0.9.18.
- Foundry remained in Setup without world users. Both service identities, startup settings, all 176 captured fortress world files and protected core/system/audio/other-module files matched the fresh baseline. No service restart or world launch occurred.
- Full native-fixture regression: 85 files, 1,758 passed, zero failures or skips. Earlier local browser QA passed 27 result assertions; its scope and limits are recorded in the release verification, not claimed as a full production performance test.
- Seventeen deployment-transaction tests and seven archive-validation checks passed.
- World runtime maintenance has not yet run. On the next GM login, successful Patreon rule repair will ask already-online clients to refresh once so their cached rules update.

ZIP SHA-256: `2d85f1c6fe06769557b91b05306d91470af21c59601b607463a6f52003074728`.

The old module backup remains at `/root/fvtt-patch-backups/automation-release-0918-20260925/backup-pf2e-third-party-automation-0.9.17`. Detailed local receipts and checksums are retained under `C:/Users/Taka/Desktop/fvtt/output/automation-release-0918-20260925`.
