# 0.9.10 verification

This patch addresses two reported 0.9.9 runtime issues. Other automation expansion remains paused at the user's request. Publication and production deployment are separate.

## Changes and boundaries

- HUD 2.55.2 mixes action controllers with stances, strikes, spells and other shortcut kinds. The old render hook submitted every item-bearing controller to an action-only fingerprint check, causing a false compatibility warning. Only the reviewed `ActionsSidebarAction` and `ActionShortcut` controllers now enter that check; persistent shortcuts must also have `type === 'action'`. Exact method hashes and the awaited original Toolbelt helper remain unchanged.
- Toolbelt 3.56.2's `mergeDamage` records multiple source cards. An ordinary card has no such flag. Merging groups primary rolls into one damage application and does not establish a single triggering attack. The source validator continues to reject it for reaction automation.
- Before any reaction scope or payment exists, Glimpse of Redemption and Spiritual Scar now warn that a merged card needs manual reaction handling and return the original damage parameters. Native damage proceeds once without automatic reaction or daily-use expense. Scar receives the same fix so the next damage provider cannot reintroduce the block after Glimpse passes through.
- This does not add automated reactions to combined attacks. Unknown sources, source drift after payment, duplicate use, uncertain native receipts and other existing authorization checks retain their previous behavior.

## Automated and independent checks

The complete suite passed **1443 tests, zero failures and zero skips**, with installed native source fixtures supplied. The new merged-card regressions first reproduced the reported Glimpse exception and corresponding Scar exception, then passed after the fix. Mixed HUD regressions likewise reproduced the reported warning before the render filter changed.

Coverage includes original parameter identity, exactly one native application, no reaction/daily payment or follow-up, unchanged unrelated HUD methods, repeated mixed renders, unchanged fingerprint rejection, original event/item forwarding, awaited begin/start/native/finish, and original exploration/ineligible-actor routes. An independent read-only agent inspected actual Toolbelt and HUD source maps and the patch, then ran 121 affected tests with no failures and reported no actionable findings.

## Native acceptance

Foundry 14.368 / PF2e 8.5.1, Toolbelt 3.56.2 and HUD 2.55.2 were used in the isolated environment. The real mixed HUD sidebar contained one action and two strikes; persistent shortcuts contained an action and a feat. Each reviewed action controller used the original Widen action through the original Toolbelt helper, returned an awaited native card and exactly one committed receipt, and left no pending lease. No false fingerprint warning occurred. Temporary actor/cards were removed, HUD settings restored, and original Claudius remained unchanged.

This is native controller/API acceptance, not a pointer-click claim. The HUD run also recorded one separate parent-render `appendChild` error during its setting/render lifecycle; the run is not claimed to be globally free of upstream HUD errors. This does not affect the specific action-card and receipt assertions.

The real Toolbelt merge API combined two original NPC Strike damage cards (2 + 3) for both slashing and fiend-origin spirit damage. Clicking the actual target damage button applied exactly 5 HP once, emitted the manual-reaction notice, and preserved both reaction ledgers and the Scar daily frequency. Both cases passed with zero page or cleanup errors; original actors and world time were unchanged. These checks do not establish compatibility with every production module.

## Release and deployment

The package contains 129 runtime files. Only three script files change mechanically: the two reaction providers and the metapower HUD render registration. The manifest and README describe 0.9.10. Prior release acceptance is retained in `verification-0.9.9.md`.

The user explicitly authorized applying the fix and reopening the world despite the current online users. Deployment must bind fresh service identities and exact files, preserve the Setup default and character service, and report installation separately from publication.
