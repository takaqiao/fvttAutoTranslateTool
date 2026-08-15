#!/usr/bin/env bash
# §5.4 全套（第 13 项按 PROJECT.md 明文「只在真出问题时跑」故意不跑；第 11 项另起后台）
set +e
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
P="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
Q="$P/3-常用脚本/qa"
R="$P/4-临时脚本/2026-08-15-round18/reports"
B="$P/4-临时脚本/2026-08-15-round18/batches54"
D="C:/Users/Taka/AppData/Local/FoundryVTT/Data"
EB="$P/5-其他内容/english-baseline/ember-cn-v1.0.15-shipped-en"
CB="$P/5-其他内容/english-baseline/crucible-0.9.1-legacy"
mkdir -p "$R" "$B"
cd "$P"

hdr(){ echo; echo "############### $* ###############"; }

hdr "1. lang_gap ember";      python "$Q/lang_gap.py" --repo "1-Ember汉化插件"   --package "$D/modules/ember"    --out "$R/lang_ember"    2>&1 | tail -12
hdr "1. lang_gap crucible";   python "$Q/lang_gap.py" --repo "2-Crucible汉化插件" --package "$D/systems/crucible" --out "$R/lang_crucible" 2>&1 | tail -12
hdr "1b. flatten ember";      python "$Q/flatten_lang.py" --repo "1-Ember汉化插件"   --english "$D/modules/ember/lang/en.json"    2>&1 | tail -8
hdr "1b. flatten crucible";   python "$Q/flatten_lang.py" --repo "2-Crucible汉化插件" --english "$D/systems/crucible/lang/en.json" 2>&1 | tail -8

for RP in "1-Ember汉化插件" "2-Crucible汉化插件"; do
  hdr "2. scan_markup_drift $RP";   python "$Q/scan_markup_drift.py"   --repo "$RP" 2>&1 | tail -12
  hdr "2b. scan_markup_targets $RP";python "$Q/scan_markup_targets.py" --repo "$RP" 2>&1 | tail -12
done
hdr "2c. scan_class_drift ember";    python "$Q/scan_class_drift.py" --repo "1-Ember汉化插件"   --out "$R/class_drift_ember.json"    2>&1 | tail -10
hdr "2c. scan_class_drift crucible"; python "$Q/scan_class_drift.py" --repo "2-Crucible汉化插件" --out "$R/class_drift_crucible.json" 2>&1 | tail -10

for RP in "1-Ember汉化插件" "2-Crucible汉化插件"; do
  hdr "3. scan_content_coverage $RP"; python "$Q/scan_content_coverage.py" --repo "$RP" 2>&1 | tail -12
  hdr "4. scan_foreign_script $RP";   python "$Q/scan_foreign_script.py"   --repo "$RP" 2>&1 | tail -10
  hdr "5. prune_dead $RP";            python "$Q/prune_dead.py"            --repo "$RP" 2>&1 | tail -10
  hdr "6. fill_missing $RP";          python "$P/3-常用脚本/tm/fill_missing.py" --repo "$RP" --out-dir "$B" 2>&1 | tail -10
done

hdr "7. scan_uuid_swap ember";     python "$Q/scan_uuid_swap.py" --repo "1-Ember汉化插件"   --out "$R/uuid_swap_ember.json"    2>&1 | tail -10
hdr "7. scan_uuid_swap crucible";  python "$Q/scan_uuid_swap.py" --repo "2-Crucible汉化插件" --out "$R/uuid_swap_crucible.json" 2>&1 | tail -10

hdr "8. scan_cross_channel ember";    python "$Q/scan_cross_channel.py" --repo "1-Ember汉化插件"   --package "$D/modules/ember"    --mjs "$P/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs" --out "$R/cross_ember.json"    2>&1 | tail -14
hdr "8. scan_cross_channel crucible"; python "$Q/scan_cross_channel.py" --repo "2-Crucible汉化插件" --package "$D/systems/crucible" --mjs "$P/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs" --out "$R/cross_crucible.json" 2>&1 | tail -14

hdr "9. scan_attr_text ember";    python "$Q/scan_attr_text.py" --repo "1-Ember汉化插件"   --out "$R/attr_ember.json"    2>&1 | tail -10
hdr "9. scan_attr_text crucible"; python "$Q/scan_attr_text.py" --repo "2-Crucible汉化插件" --out "$R/attr_crucible.json" 2>&1 | tail -10

hdr "10. dump_bindings (三包合导)"; node "$Q/dump_bindings.mjs" --package "$D/modules/ember" --package "$D/systems/crucible" --package "$D/systems/dnd5e" --out "$R/bindings.json" 2>&1 | tail -10
hdr "10. scan_name_binding ember";    python "$Q/scan_name_binding.py" --repo "1-Ember汉化插件"   --bindings "$R/bindings.json" --out "$R/name_binding_ember.json"    2>&1 | tail -10
hdr "10. scan_name_binding crucible"; python "$Q/scan_name_binding.py" --repo "2-Crucible汉化插件" --bindings "$R/bindings.json" --out "$R/name_binding_crucible.json" 2>&1 | tail -10

hdr "12. dump_ids ember";  node "$P/4-临时脚本/2026-08-12-fix/dump_ids.mjs" --package "$D/modules/ember"    --out "$R/ids_ember.json"    2>&1 | tail -5
hdr "12. dump_ids crucible"; node "$P/4-临时脚本/2026-08-12-fix/dump_ids.mjs" --package "$D/systems/crucible" --out "$R/ids_crucible.json" 2>&1 | tail -5
hdr "12. scan_label_vs_name ember";    python "$Q/scan_label_vs_name.py" --repo "1-Ember汉化插件"   --ids "$R/ids_ember.json"    --out "$R/lvn_ember.json"    2>&1 | tail -10
hdr "12. scan_label_vs_name crucible"; python "$Q/scan_label_vs_name.py" --repo "2-Crucible汉化插件" --ids "$R/ids_crucible.json" --out "$R/lvn_crucible.json" 2>&1 | tail -10

hdr "14. scan_name_splits ember";    python "$Q/scan_name_splits.py" --repo "1-Ember汉化插件"   --out "$R/ns_ember.json"    2>&1 | tail -8
hdr "14. scan_name_splits crucible"; python "$Q/scan_name_splits.py" --repo "2-Crucible汉化插件" --out "$R/ns_crucible.json" 2>&1 | tail -8
hdr "15. scan_token_name ember";     python "$Q/scan_token_name.py" --repo "1-Ember汉化插件"   --out "$R/tn_ember.json"    2>&1 | tail -8
hdr "15. scan_token_name crucible";  python "$Q/scan_token_name.py" --repo "2-Crucible汉化插件" --out "$R/tn_crucible.json" 2>&1 | tail -8
hdr "16. scan_bare_english_names ember";    python "$Q/scan_bare_english_names.py" --repo "1-Ember汉化插件"   --min-words 1 --out "$R/bare_ember.json"    2>&1 | tail -8
hdr "16. scan_bare_english_names crucible"; python "$Q/scan_bare_english_names.py" --repo "2-Crucible汉化插件" --min-words 1 --out "$R/bare_crucible.json" 2>&1 | tail -8

hdr "17. scan_status_name ember";    python "$P/4-临时脚本/2026-08-13-round8/probes/scan_status_name.py" --repo "1-Ember汉化插件"   --lang-en "$D/systems/crucible/lang/en.json" --lang-cn "2-Crucible汉化插件/lang/cn.json" 2>&1 | tail -8
hdr "17. scan_status_name crucible"; python "$P/4-临时脚本/2026-08-13-round8/probes/scan_status_name.py" --repo "2-Crucible汉化插件" --lang-en "$D/systems/crucible/lang/en.json" --lang-cn "2-Crucible汉化插件/lang/cn.json" 2>&1 | tail -8

hdr "18. scan_same_en_split 两仓合跑"; python "$Q/scan_same_en_split.py" --repo "1-Ember汉化插件" --repo "2-Crucible汉化插件" --out "$R/same_en_split.json" 2>&1 | tail -12

hdr "19. assert_resolutions";          python "$Q/assert_resolutions.py" 2>&1 | tail -8
hdr "19. assert_resolutions --selftest"; python "$Q/assert_resolutions.py" --selftest 2>&1 | tail -16

hdr "20. scan_dropped_terms ember (三包 bindings)"; python "$Q/scan_dropped_terms.py" --repo "1-Ember汉化插件" --bindings "$R/bindings.json" --baseline "$EB" --out "$R/dropped_ember.json" 2>&1 | tail -12
hdr "20. scan_dropped_terms crucible (三包 bindings)"; python "$Q/scan_dropped_terms.py" --repo "2-Crucible汉化插件" --bindings "$R/bindings.json" --baseline "$CB" --out "$R/dropped_crucible.json" 2>&1 | tail -12
hdr "21. scan_number_drift ember";    python "$Q/scan_number_drift.py" --repo "1-Ember汉化插件"   --baseline "$EB" --out "$R/nd_ember.json"    2>&1 | tail -10
hdr "21. scan_number_drift crucible"; python "$Q/scan_number_drift.py" --repo "2-Crucible汉化插件" --baseline "$CB" --out "$R/nd_crucible.json" 2>&1 | tail -10
hdr "22. scan_marker_followup ember";    python "$Q/scan_marker_followup.py" --repo "1-Ember汉化插件"   --baseline "$EB" --out "$R/mf_ember.json"    2>&1 | tail -10
hdr "22. scan_marker_followup crucible"; python "$Q/scan_marker_followup.py" --repo "2-Crucible汉化插件" --baseline "$CB" --out "$R/mf_crucible.json" 2>&1 | tail -10

hdr "23. sync_twin_packs（最后一步）"; python "$Q/sync_twin_packs.py" --repo "1-Ember汉化插件" --out-dir "$B" 2>&1 | tail -20

echo; echo "############### DONE ###############"
