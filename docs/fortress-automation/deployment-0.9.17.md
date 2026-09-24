# 0.9.17 deployment verification

[0.9.17](https://github.com/takaqiao/fvttAutoTranslateTool/releases/tag/pf2e-third-party-automation-v0.9.17) was published and installed successfully on 2026-09-24 at 16:58:53 UTC (2026-09-25 00:58:53 Asia/Shanghai).

- Release commit: `93f4f057f1629df4fcc7a3eecc5705f8a8ad9c77`.
- The public manifest and ZIP were downloaded and verified against the release tag and committed source. GitHub's latest release points to 0.9.17. All 134 runtime files matched the archive inventory.
- Runtime changes are exactly README.md, module.json and scripts/spell-combination.mjs. No runtime file was added or removed.
- The transaction verified a complete 0.9.16 backup, atomically replaced the module and refreshed Foundry's native package cache. Its retained receipt is `installed`.
- Native package version and all 134 served HTTP files matched 0.9.17 afterward.
- Foundry remained in Setup with no world users. Both service identities, startup settings, 130 captured fortress world files, protected core/system/audio files and other modules matched the fresh baseline. No service restart or world launch was performed.
- Release regression: 81 files, 1,682 passed, zero failed or skipped. Seventeen native magus cases passed with zero browser errors. The published script matches the native-QA version after Git line-ending normalization.
- Deployment checks: 17 transaction tests and 7 archive-validation tests passed.

The local checkout initially preceded the repository's completed history cleanup. Before deployment, the two new commits were reapplied to the existing cleaned maintenance branch and their complete module trees were verified identical. The newly published release tag was corrected to that history, the accidentally recreated old development branch was removed, and remote references were verified. The release archive bytes were unchanged; existing version tags and unrelated branches were retained.

ZIP SHA-256: `9338329bc41232aaac5e6585dc24f64607046614b88006cb6b7060fac6368869`.

Detailed local receipts, checksums and backup information are retained under `C:/Users/Taka/Desktop/fvtt/output/automation-release-0917-20260925`. This completes the authorized publication and deployment of the magus attack-context and spell-action-variant fixes.
