#!/usr/bin/env bash
# 把解包后的 taka-charsync 装到 CN。
#
# 两台的 Data/modules/taka-charsync 只放客户端要的 module.json、scripts/、styles/ 和 generated/rules.mjs：
# Data 整个目录都能被静态下载，engine/、test/、tools/ 不进去。generated/rules.mjs 只有建卡服那份由引擎重写，
# 装新版时建卡服沿用现有的（没有才用包里的默认空表），主服每次都换成包里的默认空表。
# 整个模组目录（含引擎和工具）放到 $CHARSYNC_HOME/app/。
# 目标已有的（两台的模组目录和 app/）先整份挪到 $CHARSYNC_HOME/releases/<YYYYmmdd-HHMMSS>/。
# 新内容先拷到同一父目录下的暂存目录，拷完才换上；中途失败时删掉暂存目录，还没换的旧版留在原处。
# 主服装了 pf2e-eidolon-helper 而建卡服没有时，同样经暂存目录整份拷到建卡服；建卡服已有就不动。
# 不碰 pm2，不重启任何进程。
#
# 用法：install.sh [--dry] <解包后的模组目录>
#   --dry  只打印要执行的拷贝、挪动，不改任何文件。
# 环境变量：MAIN_DATA（默认 /root/fvtt14-data/Data）、CHAR_DATA（默认 /root/fvtt14-char-data/Data）、
#           CHARSYNC_HOME（默认 /root/fvtt-charsync）。
set -euo pipefail

MODULE_ID=taka-charsync
EIDOLON_ID=pf2e-eidolon-helper
MAIN_DATA=${MAIN_DATA:-/root/fvtt14-data/Data}
CHAR_DATA=${CHAR_DATA:-/root/fvtt14-char-data/Data}
CHARSYNC_HOME=${CHARSYNC_HOME:-/root/fvtt-charsync}

DRY=0
if [ "${1:-}" = "--dry" ]; then
  DRY=1
  shift
fi
if [ $# -ne 1 ]; then
  echo "用法：$0 [--dry] <解包后的模组目录>" >&2
  exit 2
fi
SRC=${1%/}

fail() {
  echo "$*" >&2
  exit 1
}

run() {
  echo "+ $*"
  if [ "$DRY" = 0 ]; then
    "$@"
  fi
}

[ -f "$SRC/module.json" ] || fail "$SRC 下没有 module.json"
grep -q "\"id\": *\"$MODULE_ID\"" "$SRC/module.json" || fail "$SRC/module.json 的 id 不是 $MODULE_ID"
for part in scripts styles generated/rules.mjs engine/main.mjs; do
  [ -e "$SRC/$part" ] || fail "$SRC 下缺 $part"
done
for data in "$MAIN_DATA" "$CHAR_DATA"; do
  [ -d "$data/modules" ] || fail "$data/modules 不存在"
done

RELEASE=$CHARSYNC_HOME/releases/$(date +%Y%m%d-%H%M%S)
[ ! -e "$RELEASE" ] || fail "$RELEASE 已存在，隔一秒再跑"

STAGING=(
  "$MAIN_DATA/modules/.$MODULE_ID.installing"
  "$CHAR_DATA/modules/.$MODULE_ID.installing"
  "$CHARSYNC_HOME/.app.installing"
  "$CHAR_DATA/modules/.$EIDOLON_ID.installing"
)
cleanup() {
  local status=$?
  if [ "$status" != 0 ] && [ "$DRY" = 0 ]; then
    rm -rf "${STAGING[@]}"
    echo "失败（退出码 $status），已删掉暂存目录" >&2
    if [ -e "$RELEASE" ]; then
      echo "已经挪走的旧版在 $RELEASE" >&2
    fi
  fi
}
trap cleanup EXIT

# $1 换成 $2（先拷好的暂存目录）；$1 已有就先挪到 $RELEASE/$3
swap_in() {
  if [ -e "$1" ]; then
    run mkdir -p "$RELEASE"
    run mv "$1" "$RELEASE/$3"
  fi
  run mv "$2" "$1"
}

# $1 是 Data 目录，$2 是挪进 releases 时用的名字，$3 是要放进去的 generated/rules.mjs
install_module() {
  local dst=$1/modules/$MODULE_ID
  local staging=$1/modules/.$MODULE_ID.installing
  run rm -rf "$staging"
  run mkdir -p "$staging/generated"
  run cp "$SRC/module.json" "$staging/module.json"
  run cp -R "$SRC/scripts" "$staging/scripts"
  run cp -R "$SRC/styles" "$staging/styles"
  run cp "$3" "$staging/generated/rules.mjs"
  swap_in "$dst" "$staging" "$2"
}

char_rules=$SRC/generated/rules.mjs
if [ -f "$CHAR_DATA/modules/$MODULE_ID/generated/rules.mjs" ]; then
  char_rules=$CHAR_DATA/modules/$MODULE_ID/generated/rules.mjs
fi
install_module "$MAIN_DATA" main-module "$SRC/generated/rules.mjs"
install_module "$CHAR_DATA" char-module "$char_rules"

run mkdir -p "$CHARSYNC_HOME"
run rm -rf "$CHARSYNC_HOME/.app.installing"
run cp -R "$SRC" "$CHARSYNC_HOME/.app.installing"
swap_in "$CHARSYNC_HOME/app" "$CHARSYNC_HOME/.app.installing" app

if [ -e "$CHAR_DATA/modules/$EIDOLON_ID" ]; then
  echo "建卡服已有 $EIDOLON_ID，不动"
elif [ -d "$MAIN_DATA/modules/$EIDOLON_ID" ]; then
  run rm -rf "$CHAR_DATA/modules/.$EIDOLON_ID.installing"
  run cp -R "$MAIN_DATA/modules/$EIDOLON_ID" "$CHAR_DATA/modules/.$EIDOLON_ID.installing"
  run mv "$CHAR_DATA/modules/.$EIDOLON_ID.installing" "$CHAR_DATA/modules/$EIDOLON_ID"
else
  echo "主服也没有 $EIDOLON_ID，没拷" >&2
fi

if [ "$DRY" = 1 ]; then
  echo "（--dry：没有改动任何文件）"
elif [ -e "$RELEASE" ]; then
  echo "装好了；旧版在 $RELEASE"
else
  echo "装好了；之前没有装过"
fi
