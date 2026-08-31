#!/bin/sh
set -e
D="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/cn"
S="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/4-临时脚本/2026-08-18-ember-061/stage1"
cp "$S/BACKUP_cn/"*.json "$D/"
python "$S/a1_apply_gone.py"
python "$S/a2_apply_changed.py"
