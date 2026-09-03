# FotRP GMGuide Consolidation and Book2Ch1 Dual-JSON Design

## Objective

Consolidate every Fists of the Ruby Phoenix resource currently scattered across `冒险/GMGUIDE` into `冒险/GMGUIDE/FotRP`, incorporate the raw FotRP Discord export current through 2026-08-31, refresh the Book1Ch3 GM guide where the newer chat changes the advice, and prepare two Foundry-importable Book2Ch1 artifacts:

1. one paged `JournalEntry` GM guide JSON;
2. one master `Playlist` soundboard JSON.

The work preserves all user data. Files are moved or archived, not discarded.

## Source of Truth

Use sources in this order:

1. The localized Book2Ch1 Foundry journal at `冒险/GMGUIDE/FotRP/汉化/第二本/fvtt-JournalEntry-chapter-1-RzfDjQH8KPxPJ2kK.json` for RAW rules and translated names.
2. The raw Discord exports under `冒险/FotRP/fotrpChat`, whose latest message is dated 2026-08-31.
3. The existing main GM guide, music guides, community-resource indexes, and playlist files for already-curated advice and local asset paths.
4. This guide's recommendations, explicitly labeled as recommendations rather than RAW or community consensus.

The older `_corpus` is not treated as current because it stops at 2026-04.

## Confirmed Finding About Book1Ch3

The Book1Ch3 guide created on 2026-09-02 did not fully incorporate the newest FotRP chat. It used `_corpus` through 2026-04 rather than the raw export through 2026-08-31. The refresh must add or clarify these later findings:

- The Fallen Moon disintegration should visibly depend on an overcharged or exceptional wand so ordinary contestants do not make Hao Jin look incompetent.
- The Hao Jin death and rebirth macros need distinct names, ordering, and a warning because a GM reported accidentally playing the rebirth first.
- The Ruby Palace needs differentiated roleplay beats for the finalist teams instead of interchangeable NPC greetings.
- Clockwork floating cameras can establish that Goka watched qualifier highlights and already recognizes the PCs.
- Finalist introductions, especially Tino's Toughest and the Lightkeepers, should be ensured before the party leaves Bonmu.

The existing Book1Ch3 playlist choices remain valid unless the new audit finds a broken asset path or a scene mismatch.

## Target Directory Layout

```text
冒险/GMGUIDE/FotRP/
├─ README.md
├─ 指南与日志/
│  ├─ 总览/
│  ├─ 第一本/
│  ├─ 第二本/
│  └─ 第三本/
├─ 播放列表/
│  ├─ 总表/
│  └─ 分章/
├─ FotRP_Music/
├─ 社区资源/
├─ 聊天记录/
│  ├─ 当前导出_2026-08-31/
│  ├─ 旧导出_2026-04/
│  ├─ 派生语料_截至2026-04/
│  └─ 派生语料_截至2026-08/
├─ 汉化/
├─ SFX/
├─ FVTT附加资源/
│  └─ Actors/
├─ 工具/
│  ├─ 脚本/
│  ├─ 报告/
│  └─ 临时资料/
└─ 归档/
```

`汉化` and `SFX` already exist and keep their current contents.

## Move Mapping

### Guides and journals

Move into `指南与日志/总览`:

- `Fists_of_the_Ruby_Phoenix_GM指南.md`
- `Fists_of_the_Ruby_Phoenix_音乐_Playlist.md`
- `Fists_of_the_Ruby_Phoenix_音乐_Playlist_补遗.md`
- `赤凰_引文核实对照.md`
- `FotRP_邦木岛游荡队伍方案.md`

Move chapter-specific Markdown, macros, supplemental GM journals, and generated GM journals into the corresponding book directory under `指南与日志`.

Move `Fists_of_the_Ruby_Phoenix_GM指南.md.bak` into `归档/旧版指南`.

### Playlists

Move the top-level master playlist JSON files `FotRP_通用.json`, `FotRP_章*.json`, and `FotRP_B*Ch*.json` into `播放列表/总表`.

Move the contents of `FotRP_Playlists` into `播放列表/分章`, preserving the existing book and chapter subtree, including every dated backup directory.

The Foundry-facing audio prefix remains exactly `assets/FotRP_Music/`. It is a virtual Foundry asset path and must not be rewritten to the new local directory.

### Music and community resources

Move `FotRP_Music` to `FotRP/FotRP_Music` without renaming the leaf directory.

Move `FotRP_社区资源汇总` to `FotRP/社区资源`.

Move `FotRP_Music.zip` to `归档/音乐库压缩包` without modifying the archive.

### Chat records

Move the latest `冒险/FotRP/fotrpChat` export into `聊天记录/当前导出_2026-08-31`.

Move the older `GMGUIDE/fists_of_the_ruby_phoenix-1` and `GMGUIDE/fists_of_the_ruby_phoenix-2` exports into `聊天记录/旧导出_2026-04`.

Move `_corpus` into `聊天记录/派生语料_截至2026-04`.

Generate a fresh monthly plain-text corpus in `聊天记录/派生语料_截至2026-08` from both channels in the current raw export. Each line records timestamp, author, and message content. Attachments are referenced by relative path when present; binary files are not duplicated.

### Tools, reports, temporary files, and actors

Move FotRP-specific `.py`, `.ps1`, and `.js` helpers into `工具/脚本`. Move their `.log` and report `.txt` outputs into `工具/报告`. Move FotRP-specific `_temp_*` files and `_temp_pdf_pages` into `工具/临时资料`.

Move the four level-19 Wildblood actor JSON files into `FVTT附加资源/Actors` because they are FotRP endgame resources.

Do not move unrelated Claws of the Tyrant, Revenge of the Runelords, Temple of the Unlit Star, or general translation-pipeline resources.

## Path Migration Rules

After moving files, update project-local references in Markdown, Python, PowerShell, JavaScript, JSON descriptions, and reports when those references identify a filesystem location.

Do not rewrite:

- Foundry asset URLs beginning with `assets/FotRP_Music/`;
- historical source quotations where the old path is part of the quoted record;
- URLs or external identifiers;
- paths belonging to unrelated adventures.

The new root `README.md` records the target layout, the authoritative latest chat date, the two types of Foundry JSON, and the distinction between local filesystem paths and Foundry virtual asset paths.

## Migration Safety

Before moving any directory:

1. resolve both source and destination to absolute paths;
2. verify both are inside `C:\Users\Taka\Desktop\fvtt`;
3. record source file count and total byte count;
4. verify the destination does not already contain a conflicting object.

After moving, record the same count and byte total. A migration succeeds only when both values match. No recursive deletion is used. If a collision occurs, stop that move and place neither version over the other.

Write the audit to `FotRP/迁移清单_2026-09-03.json`.

## Book1Ch3 Refresh

Update the Markdown source and regenerate its Foundry `JournalEntry` in `指南与日志/第一本/Ch3_山巅女皇`.

The regenerated journal retains the current 16-page structure and adds the post-April findings in the relevant pages:

- Hao Jin assassination and macro safety;
- differentiated finalist roleplay;
- qualifier broadcast and Goka recognition;
- final pre-departure finalist-introduction checklist.

Keep the same playlist collection under `播放列表/分章/第一本_绝岛之殇/Ch3_山巅女皇` and revalidate its 10 active playlists and 42 sounds.

## Book2Ch1 JournalEntry JSON

Create `指南与日志/第二本/Ch1_寻找赞助人/fvtt-JournalEntry-GMGuide-FotRP-B2Ch1-<16-char-id>.json`.

The journal uses Foundry v14/PF2e v8 compatible structure and contains these pages:

1. Cover, scope, source labels, and one-page run summary.
2. Transition from Bonmu, downtime, arrival in Goka, and public recognition.
3. Seven-day pacing model and recommended daily rhythm.
4. Influence subsystem: Influence, Discover, Gather Information, thresholds 1/4/8, and once-per-location limits.
5. Five eligible patrons comparison matrix, including preferences, resistances, rewards, and best-fit party profiles.
6. Locked patrons and the Narswani contradiction erratum. Use the chapter-ending bidding result as the default: Lady Narswani Vangarath sponsors the Biting Roses.
7. Grand Bank opening and initial patron meeting.
8. Goka locations: Ruby Village, Icefang Aerie, Five Pillars Academy, Lantern Lodge Gallery, Empress Yin Opera House, Shelyn's Comb, and Neverending Market.
9. Four events around Goka: Drake Crash, Unexpected Rematch, Golden Opportunity, and Eclipse.
10. Sculptor foreshadowing and planar-anomaly clue ladder, without revealing Syndara prematurely.
11. Exhibition showcase, including the option to let NPC teams take some exhibitions if PC influence is already maximized.
12. Signed Lives and the final all-patrons question.
13. Bidding War, offer comparison, sponsor choice, rewards, and level-16 catch-up.
14. Scene-by-scene music cue sheet.
15. Common problems, rulings, and final GM checklist.
16. Sources and revision history.

Every piece of non-RAW advice is labeled `社区经验` or `本指南建议`.

## Book2Ch1 Playlist JSON

Create exactly one importable master soundboard at `播放列表/分章/第二本_比赛开始/Ch1_寻找赞助人/B2Ch1_音乐总控.json`.

It uses `mode: -1` and scene-first sound names so Foundry truncation preserves the cue. Sound groups are encoded in the names:

- `A` — Goka arrival and tournament zeitgeist;
- `B` — Grand Bank and patron introductions;
- `C` — general city exploration and tea-house downtime;
- `D` — seven patron-location scenes;
- `E` — four Goka events;
- `F` — team and monster exhibitions;
- `G` — Signed Lives and Bidding War;
- `H` — planar anomalies and Sculptor foreshadowing.

Prefer existing files in `FotRP_Music`. Download with `yt-dlp` only if a scene has a clearly superior missing track with an appropriate source. Any download is placed under the relevant `FotRP_Music` category and recorded in the playlist description and migration manifest.

All playlist and sound IDs are exactly 16 alphanumeric characters. The playlist's export UUID is `Playlist.<same-16-character-id>`. All sound paths begin with `assets/FotRP_Music/` and resolve to an existing local file after mapping that prefix to `FotRP/FotRP_Music`.

## Verification

The final verification must prove:

- all planned source objects were moved and no unplanned GMGuide objects were moved;
- file counts and byte totals match for every moved directory;
- the new 2026-08 corpus includes the latest raw message timestamp, 2026-08-31;
- no stale local reference to the old FotRP directories remains outside archived historical records;
- all Foundry JSON files parse;
- both new Book2Ch1 JSON files have correct top-level document types by schema;
- all IDs are valid and unique within each document;
- every playlist audio path exists;
- Book1Ch3 has 16 journal pages, 10 active playlists, and 42 sounds;
- Book2Ch1 has 16 journal pages and one master playlist soundboard;
- the Book2Ch1 journal contains the Narswani erratum and latest-chat provenance;
- no Book2/Book3 music is used in Book1Ch3.

## Deliverables

1. Consolidated `冒险/GMGUIDE/FotRP` directory.
2. `FotRP/README.md`.
3. `FotRP/迁移清单_2026-09-03.json`.
4. Refreshed Book1Ch3 Markdown and JournalEntry JSON.
5. Book2Ch1 JournalEntry GM guide JSON.
6. Book2Ch1 master Playlist soundboard JSON.
7. Updated local references and validation evidence.
