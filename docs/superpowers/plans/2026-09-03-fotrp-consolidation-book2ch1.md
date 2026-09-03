# FotRP Consolidation and Book2Ch1 Dual-JSON Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate all FotRP GMGuide resources, refresh Book1Ch3 against the Discord export current through 2026-08-31, and deliver one Book2Ch1 GM JournalEntry JSON plus one Book2Ch1 master Playlist JSON.

**Architecture:** A guarded PowerShell migration records file counts and byte totals before moving explicitly enumerated sources into `冒险/GMGUIDE/FotRP`. A separate corpus builder converts the latest raw Discord JSON into searchable monthly text. Editable Markdown remains the source for GM journals, which are converted to Foundry HTML pages; playlist data is authored directly as Foundry JSON and validated against the relocated local music library.

**Tech Stack:** PowerShell 7, Foundry VTT v14.361 JSON, PF2e v8.1.2, Markdown/HTML, local Discord export JSON, `rg`, optional `yt-dlp` only when the local music library has a material gap.

**Spec:** `docs/superpowers/specs/2026-09-03-fotrp-consolidation-book2ch1-design.md`

## Global Constraints

- Preserve all user data; use moves and dated archives, never recursive deletion.
- Before each move, resolve source and destination and verify both remain under `C:\Users\Taka\Desktop\fvtt`.
- Stop on collisions instead of overwriting.
- Preserve every Foundry virtual audio path beginning `assets/FotRP_Music/`.
- Use the localized Book2Ch1 journal for RAW and the raw Discord export through 2026-08-31 for community experience.
- Label non-RAW material as `社区经验` or `本指南建议`.
- All Foundry document and embedded IDs are exactly 16 alphanumeric characters.
- The two Book2Ch1 importables are exactly one `JournalEntry` JSON and one `Playlist` soundboard JSON.
- `冒险/GMGUIDE` is intentionally gitignored; do not force-add its large or generated artifacts.

---

### Task 1: Create the migration contract validator

**Files:**
- Create: `冒险/GMGUIDE/FotRP/工具/脚本/Test-FotRPBundle.ps1`
- Read: `docs/superpowers/specs/2026-09-03-fotrp-consolidation-book2ch1-design.md`

**Interfaces:**
- Consumes: `-WorkspaceRoot <absolute path>` and `-Stage PreMigration|PostMigration|Final`.
- Produces: concise `KEY=VALUE` lines and exit code `0` only when every assertion for the selected stage passes.

- [ ] **Step 1: Write the validator before moving any resource**

Implement these helper functions in `Test-FotRPBundle.ps1`:

```powershell
param(
  [Parameter(Mandatory)][string]$WorkspaceRoot,
  [ValidateSet('PreMigration','PostMigration','Final')][string]$Stage = 'Final'
)
$ErrorActionPreference = 'Stop'

function Resolve-WorkspacePath([string]$RelativePath) {
  [System.IO.Path]::GetFullPath((Join-Path $WorkspaceRoot $RelativePath))
}

function Assert-True([bool]$Condition, [string]$Message) {
  if (-not $Condition) { $script:Failures.Add($Message) }
}

function Get-TreeStats([string]$LiteralPath) {
  $files = @(Get-ChildItem -LiteralPath $LiteralPath -Recurse -File -Force)
  [ordered]@{ fileCount = $files.Count; totalBytes = [long](($files | Measure-Object Length -Sum).Sum) }
}
```

For `PreMigration`, assert that the current sources exist. For `PostMigration`, assert the target layout exists, every manifest entry has matching before/after counts and bytes, and every listed source is absent. For `Final`, add all JSON, journal, playlist, corpus, ID, and audio-path assertions from the spec.

- [ ] **Step 2: Run the pre-migration contract**

Run:

```powershell
pwsh -NoProfile -File .\冒险\GMGUIDE\FotRP\工具\脚本\Test-FotRPBundle.ps1 `
  -WorkspaceRoot C:\Users\Taka\Desktop\fvtt -Stage PreMigration
```

Expected: exit `0`; output includes `LATEST_CHAT_MAX=2026-08-31`, the source-tree counts, and `FAILURES=0`.

- [ ] **Step 3: Prove that the post-migration contract is initially red**

Run the same command with `-Stage PostMigration`.

Expected: nonzero exit because `FotRP/迁移清单_2026-09-03.json` and the target subdirectories do not yet exist.

---

### Task 2: Implement and execute the guarded resource migration

**Files:**
- Create: `冒险/GMGUIDE/FotRP/工具/脚本/Move-FotRPResources.ps1`
- Create mechanically: `冒险/GMGUIDE/FotRP/迁移清单_2026-09-03.json`
- Move: explicitly enumerated sources below

**Interfaces:**
- Consumes: `-WorkspaceRoot <absolute path>` and optional `-DryRun`.
- Produces: target directory tree plus a JSON manifest with `source`, `destination`, `fileCountBefore`, `totalBytesBefore`, `fileCountAfter`, `totalBytesAfter`, and `status`.

- [ ] **Step 1: Create an explicit move table**

The migration script must contain literal mappings, not globs, for directories and important files. Its directory table is:

```powershell
$DirectoryMoves = @(
  @{ source='冒险\GMGUIDE\FotRP_Music'; destination='冒险\GMGUIDE\FotRP\FotRP_Music' },
  @{ source='冒险\GMGUIDE\FotRP_社区资源汇总'; destination='冒险\GMGUIDE\FotRP\社区资源' },
  @{ source='冒险\GMGUIDE\FotRP_Playlists'; destination='冒险\GMGUIDE\FotRP\播放列表\分章' },
  @{ source='冒险\FotRP\fotrpChat'; destination='冒险\GMGUIDE\FotRP\聊天记录\当前导出_2026-08-31' },
  @{ source='冒险\GMGUIDE\fists_of_the_ruby_phoenix-1'; destination='冒险\GMGUIDE\FotRP\聊天记录\旧导出_2026-04\fists_of_the_ruby_phoenix-1' },
  @{ source='冒险\GMGUIDE\fists_of_the_ruby_phoenix-2'; destination='冒险\GMGUIDE\FotRP\聊天记录\旧导出_2026-04\fists_of_the_ruby_phoenix-2' },
  @{ source='冒险\GMGUIDE\_corpus'; destination='冒险\GMGUIDE\FotRP\聊天记录\派生语料_截至2026-04' },
  @{ source='冒险\GMGUIDE\_temp_pdf_pages'; destination='冒险\GMGUIDE\FotRP\工具\临时资料\PDF页' },
  @{ source='冒险\GMGUIDE\_中文素材下载'; destination='冒险\GMGUIDE\FotRP\工具\临时资料\中文素材下载' }
)
```

Implement `Assert-InWorkspace`, `Get-TreeStats`, and `Move-Guarded`. `Move-Guarded` must reject a missing source, an existing destination, a path outside the workspace, or before/after stat mismatch. It uses `Move-Item -LiteralPath` and never calls `Remove-Item`.

- [ ] **Step 2: Add the exact file move groups**

Encode these groups:

- Guides to `指南与日志/总览`: the three `Fists_of_the_Ruby_Phoenix_*.md` guides, `赤凰_引文核实对照.md`, and `FotRP_邦木岛游荡队伍方案.md`.
- Chapter resources to their book folders: `FotRP_B1Ch1_GM_补充_FVTT.json`, `FotRP_B1Ch1_GM须知与推荐.md`, `FotRP_B1Ch1_Macros.js`, `FotRP_B1Ch2_GM_补充_FVTT.json`, `FotRP_B1Ch3_GM须知与推荐.md`, and `fvtt-JournalEntry-GMGuide-FotRP-B1Ch3-7kM2pQ9vR4xN6cTa.json`.
- Master playlists to `播放列表/总表`: `FotRP_通用.json`, `FotRP_章1_绝岛之殇.json`, `FotRP_章2_比赛开始.json`, `FotRP_章3_山中之王.json`, `FotRP_章4_通用.json`, and every `FotRP_B1Ch*.json`, `FotRP_B2Ch*.json`, `FotRP_B3Ch*.json` that is not a `GM_补充` journal, except `FotRP_B2Ch1_寻找赞助人.json`.
- Legacy Book2Ch1 master playlist to `播放列表/分章/第二本_比赛开始/Ch1_寻找赞助人/_backup_2026-09-03/FotRP_B2Ch1_寻找赞助人.json` so no second active Book2Ch1 master remains.
- Reports to `工具/报告`: `_audit_report.txt`, `_comp.log`, `_gen_main.log`, `_gen_report.txt`, `_gen_sub.log`, and `_validation_report.txt`.
- FotRP scripts to `工具/脚本`: every explicitly inventoried `_audit_fotrp*`, `_audit_production_ready.py`, `_comprehensive_check.py`, `_final_preflight.py`, `_generate_fotrp_playlists.py`, `_generate_subplaylists.py`, `_localize.py`, `_proofread_all52.py`, `_stadium_postprocess.py`, `_translate_*.ps1`, `_tts_*.py`, and `_validate_fotrp_playlists.py`.
- Temporary exemplar and extraction files to `工具/临时资料`.
- Four `fvtt-Actor-*-wildblood-lvl-19-*.json` files to `FVTT附加资源/Actors`.
- `FotRP_Music.zip` to `归档/音乐库压缩包` and `Fists_of_the_Ruby_Phoenix_GM指南.md.bak` to `归档/旧版指南`.

- [ ] **Step 3: Dry-run and inspect the manifest preview**

Run:

```powershell
pwsh -NoProfile -File .\冒险\GMGUIDE\FotRP\工具\脚本\Move-FotRPResources.ps1 `
  -WorkspaceRoot C:\Users\Taka\Desktop\fvtt -DryRun
```

Expected: every planned move reports `READY`; unrelated adventures are absent; no collision is reported.

- [ ] **Step 4: Execute the migration**

Run the same command without `-DryRun`.

Expected: every manifest entry reports `MOVED`, and each directory entry has identical before/after file count and byte total.

- [ ] **Step 5: Run the post-migration contract**

Run `Test-FotRPBundle.ps1 -Stage PostMigration`.

Expected: exit `0`, `FAILURES=0`.

---

### Task 3: Rebuild the searchable corpus from the latest Discord export

**Files:**
- Create: `冒险/GMGUIDE/FotRP/工具/脚本/Export-FotRPCorpus.ps1`
- Create mechanically: `冒险/GMGUIDE/FotRP/聊天记录/派生语料_截至2026-08/YYYY-MM.txt`
- Update mechanically: `冒险/GMGUIDE/FotRP/迁移清单_2026-09-03.json`

**Interfaces:**
- Consumes: `-InputRoot` pointing to `聊天记录/当前导出_2026-08-31` and `-OutputRoot` pointing to the new corpus directory.
- Produces: UTF-8 monthly text files sorted chronologically and a summary object with message count, minimum timestamp, maximum timestamp, and channel list.

- [ ] **Step 1: Write a failing corpus assertion**

Extend the validator's `PostMigration` stage to require:

```powershell
$latestCorpus = Join-Path $FotRPRoot '聊天记录\派生语料_截至2026-08\2026-08.txt'
Assert-True (Test-Path -LiteralPath $latestCorpus -PathType Leaf) 'missing 2026-08 corpus'
Assert-True ((Get-Content -LiteralPath $latestCorpus -Raw) -match '2026-08-31') 'latest corpus date missing'
```

Run the validator and expect a nonzero exit before creating the corpus.

- [ ] **Step 2: Implement deterministic extraction**

Read only files matching `fists_of_the_ruby_phoenix-page-*.json`. For each message, emit:

```text
YYYY-MM-DD HH:mm:ss +08:00 | channel-folder | display-name | message text
```

Collapse embedded newlines to ` ⏎ `, preserve URLs, and append `attachments=<relative paths>` when attachments are present. Group by local `yyyy-MM`, sort by timestamp then Discord message ID, and write with UTF-8 no BOM.

- [ ] **Step 3: Generate and verify the corpus**

Run the exporter, then verify the summary maximum timestamp equals the raw export maximum timestamp and falls on `2026-08-31`. Re-run `Test-FotRPBundle.ps1 -Stage PostMigration`; expect `FAILURES=0`.

---

### Task 4: Repair local references and document the consolidated root

**Files:**
- Create: `冒险/GMGUIDE/FotRP/README.md`
- Modify: moved Markdown and FotRP-specific scripts under `指南与日志`, `社区资源`, `FotRP_Music/_脚本`, and `工具/脚本`
- Test: `冒险/GMGUIDE/FotRP/工具/脚本/Test-FotRPBundle.ps1`

**Interfaces:**
- Consumes: the old-to-new mapping recorded in the migration manifest.
- Produces: working local references while leaving Foundry virtual paths unchanged.

- [ ] **Step 1: Enumerate stale filesystem references**

Run `rg -n -uu` for the old strings `GMGUIDE/FotRP_Playlists`, `GMGUIDE/FotRP_Music`, `GMGUIDE/FotRP_社区资源汇总`, `GMGUIDE/_corpus`, and the two old Discord export directory names. Exclude `聊天记录/旧导出_2026-04`, historical corpus text, and JSON manifest `source` fields.

- [ ] **Step 2: Patch only local-path references**

Update local references to:

```text
冒险/GMGUIDE/FotRP/播放列表/分章
冒险/GMGUIDE/FotRP/FotRP_Music
冒险/GMGUIDE/FotRP/社区资源
冒险/GMGUIDE/FotRP/聊天记录/派生语料_截至2026-08
冒险/GMGUIDE/FotRP/聊天记录/当前导出_2026-08-31
```

Do not change any `assets/FotRP_Music/` string.

- [ ] **Step 3: Write the root README**

Document the new directory map, latest chat cutoff, location of both editable guides and importable JSON files, migration manifest, and the difference between the local music path and the Foundry asset prefix.

- [ ] **Step 4: Verify the reference repair**

Run the stale-reference search again. Expected: zero live stale references and unchanged count of `assets/FotRP_Music/` occurrences in playlist JSON.

---

### Task 5: Refresh Book1Ch3 from post-April chat findings

**Files:**
- Modify: `冒险/GMGUIDE/FotRP/指南与日志/第一本/Ch3_山巅女皇/FotRP_B1Ch3_GM须知与推荐.md`
- Regenerate: `冒险/GMGUIDE/FotRP/指南与日志/第一本/Ch3_山巅女皇/fvtt-JournalEntry-GMGuide-FotRP-B1Ch3-7kM2pQ9vR4xN6cTa.json`
- Test: `冒险/GMGUIDE/FotRP/工具/脚本/Test-FotRPBundle.ps1`

**Interfaces:**
- Consumes: the moved Markdown, existing 16-page journal structure, and raw messages dated 2026-04-21 through 2026-08-31.
- Produces: synchronized Markdown and JournalEntry with the same section order and refreshed provenance.

- [ ] **Step 1: Add failing content assertions**

Require the Markdown and rendered journal HTML to contain `超充能法杖`, `宏顺序警告`, `浮空摄影机`, `晋级队伍差异化`, and `聊天截止：2026-08-31`.

Run `Test-FotRPBundle.ps1 -Stage Final`; expect failure on these missing markers.

- [ ] **Step 2: Patch the editable Markdown**

Add the supercharged-wand clarification to the Fallen Moon section; add a boxed macro-order warning to the music/staging section; add differentiated finalist prompts to the palace roleplay section; add the floating-camera/public-recognition bridge to the Book2 transition; update provenance and revision history to the 2026-08-31 cutoff.

- [ ] **Step 3: Regenerate the journal deterministically**

Split the Markdown at level-two headings, convert each section with `ConvertFrom-Markdown`, keep the existing journal ID `7kM2pQ9vR4xN6cTa`, preserve 16 pages and existing page names, and generate unique 16-character alphanumeric page IDs.

- [ ] **Step 4: Revalidate Book1Ch3**

Expected: journal parses, has 16 pages, contains all five new markers, and the active playlist directory still has 10 JSON files and 42 existing audio files with no Book2/Book3 path prefix.

---

### Task 6: Build the Book2Ch1 master Playlist soundboard

**Files:**
- Read: `冒险/GMGUIDE/FotRP/播放列表/总表/FotRP_B2Ch1_寻找赞助人.json`
- Read: `冒险/GMGUIDE/FotRP/播放列表/分章/第二本_比赛开始/Ch1_寻找赞助人/*.json`
- Create: `冒险/GMGUIDE/FotRP/播放列表/分章/第二本_比赛开始/Ch1_寻找赞助人/B2Ch1_音乐总控-B2C1Music7Kp4N8x.json`
- Preserve: old Book2Ch1 playlist JSON files under `_backup_2026-09-03`

**Interfaces:**
- Consumes: the relocated local library `FotRP/FotRP_Music` and the existing curated Book2 tracks.
- Produces: one Foundry Playlist with ID `B2C1Music7Kp4N8x`, `mode: -1`, and scene-first cues grouped A through H.

- [ ] **Step 1: Audit existing tracks before choosing music**

List the current two chapter playlists and the old master playlist, map every `assets/FotRP_Music/` path to the local library, and record missing paths. Search the local library for Goka, city, tea house, patron, bank, opera, market, drake, eclipse, exhibition, bidding, Syndara, and planar-anomaly cues.

- [ ] **Step 2: Select a complete scene chain**

Choose at least one cue for each required group A–H and at least one distinct cue for each of the seven location scenes and four Goka events. Prefer locally existing curated tracks. Use `yt-dlp` only if the audit proves a materially better missing cue and the source is appropriate; record any download in the manifest.

- [ ] **Step 3: Back up old active Book2Ch1 playlists**

Create `_backup_2026-09-03` and confirm it contains the two old per-scene JSON files plus the old master playlist moved during Task 2. Do not delete them and do not leave any legacy Book2Ch1 playlist active elsewhere.

- [ ] **Step 4: Author the soundboard JSON**

Set:

```json
{
  "name": "B2Ch1 ▸ 音乐总控·寻找赞助人",
  "channel": "music",
  "mode": -1,
  "sorting": "m",
  "_stats": {
    "coreVersion": "14.361",
    "systemId": "pf2e",
    "systemVersion": "8.1.2",
    "exportSource": { "uuid": "Playlist.B2C1Music7Kp4N8x" }
  }
}
```

Prefix sound names with `A01·` through `Hxx·`, use unique 16-character IDs, and describe the exact activation condition for each cue.

- [ ] **Step 5: Validate the playlist**

Expected: one active master soundboard, ID and UUID match, all embedded IDs are unique and valid, every path exists, and no sound name starts with a track title before its scene tag.

---

### Task 7: Build the Book2Ch1 GM guide and JournalEntry JSON

**Files:**
- Create: `冒险/GMGUIDE/FotRP/指南与日志/第二本/Ch1_寻找赞助人/FotRP_B2Ch1_GM须知与推荐.md`
- Create: `冒险/GMGUIDE/FotRP/指南与日志/第二本/Ch1_寻找赞助人/fvtt-JournalEntry-GMGuide-FotRP-B2Ch1-b2c1gmJ9Qx4L7sTa.json`
- Read: localized RAW chapter and current Discord export

**Interfaces:**
- Consumes: chapter RAW, latest chat findings, and the finished B2Ch1 soundboard cue names.
- Produces: an editable Markdown source and a 16-page Foundry JournalEntry with ID `b2c1gmJ9Qx4L7sTa`.

- [ ] **Step 1: Write failing journal assertions**

Require a 16-page JournalEntry and these content markers: `聊天截止：2026-08-31`, `Narswani`, `刺骨蔷薇`, `1／4／8`, `每天一次探查`, `2–3 个地点`, `浮空摄影机`, `表演赛可由 NPC 队伍承担`, `Signed Lives`, and `B2Ch1 ▸ 音乐总控·寻找赞助人`.

- [ ] **Step 2: Author the 16-section Markdown source**

Follow the page order in the design spec. Include exact RAW DCs, influence thresholds, rewards, location limits, four event summaries, exhibition rules, signed-lives question, bidding order, 80 XP story award, and level-16 catch-up. Present the five eligible patrons in a decision table. Mark the Narswani contradiction as errata and default to the chapter-ending assignment to the Biting Roses.

Integrate current chat advice: Goka as tournament zeitgeist, 2–3 sites on a productive day, daily Discover usage, avoiding patron-grind padding, NPC-run exhibitions when influence is already capped, and subtle planar clues without prematurely explaining Syndara.

- [ ] **Step 3: Convert Markdown to Foundry JournalEntry JSON**

Use `ConvertFrom-Markdown` for HTML, one page per level-two section, `folder: null`, `type: text`, `text.format: 1`, visible level-one page titles, sequential sort values, and unique 16-character page IDs. Use the exact export UUID `JournalEntry.b2c1gmJ9Qx4L7sTa`.

- [ ] **Step 4: Validate the journal against RAW and chat**

Expected: 16 pages, all required markers present, no unlabeled community recommendation, no claim that all eight patrons are selectable, and no claim that Narswani sponsors the Arms of Balance in the default resolution.

---

### Task 8: Complete the consolidated README, manifest, and final audit

**Files:**
- Modify: `冒险/GMGUIDE/FotRP/README.md`
- Modify mechanically: `冒险/GMGUIDE/FotRP/迁移清单_2026-09-03.json`
- Run: `冒险/GMGUIDE/FotRP/工具/脚本/Test-FotRPBundle.ps1`

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: final evidence that the migration and both Book2Ch1 JSONs are complete and importable.

- [ ] **Step 1: Add final artifact links to the README**

Link the refreshed Book1Ch3 journal, Book2Ch1 journal, Book2Ch1 soundboard, current chat export, fresh corpus, playlist root, music root, community resources, and manifest.

- [ ] **Step 2: Finalize the migration manifest**

Append generated-artifact entries, optional music downloads, verification timestamp, and final status. Keep original source and destination values for auditability.

- [ ] **Step 3: Run the full validator**

Run:

```powershell
pwsh -NoProfile -File .\冒险\GMGUIDE\FotRP\工具\脚本\Test-FotRPBundle.ps1 `
  -WorkspaceRoot C:\Users\Taka\Desktop\fvtt -Stage Final
```

Expected summary:

```text
MIGRATION_MISMATCHES=0
STALE_LIVE_REFERENCES=0
LATEST_CHAT_MAX=2026-08-31
B1CH3_JOURNAL_PAGES=16
B1CH3_PLAYLISTS=10
B1CH3_SOUNDS=42
B2CH1_JOURNAL_PAGES=16
B2CH1_PLAYLISTS=1
MISSING_AUDIO=0
INVALID_IDS=0
FAILURES=0
```

- [ ] **Step 4: Inspect final filesystem scope**

List `冒险/GMGUIDE` top-level items and confirm the moved FotRP-specific objects are absent while unrelated adventure resources remain. List `冒险/GMGUIDE/FotRP` recursively to depth two and compare it with the target layout.

- [ ] **Step 5: Inspect repository status without force-adding ignored artifacts**

Run `git status --short`. Confirm no unrelated tracked file was changed by the migration. Do not use `git add -f` for `冒险/GMGUIDE`.
