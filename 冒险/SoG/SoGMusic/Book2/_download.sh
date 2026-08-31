#!/bin/bash
# 下载 Book2 新增曲目并转 ogg (libvorbis -q:a 5，与现有库一致)
# 只处理 _manifest.tsv 里 kind=new 的行；kind=reuse 的是库里早有的文件，不下载。
# 注：本机 yt-dlp 2026.03.17 对默认 web 客户端会 403，必须走 player_client=android
cd "$(dirname "$0")" || exit 1
FF="../ffmpeg.exe"
ok=0; fail=0
while IFS=$'\t' read -r num ch zh en kind src ref note; do
  case "$num" in \#*|"") continue;; esac
  [ "$kind" = "new" ] || continue
  out="${num} ${zh} - ${en}.ogg"
  if [ -f "$out" ]; then echo "SKIP  $out"; ok=$((ok+1)); continue; fi
  echo ">>> [$num] $zh  <-  $en  ($ref)"
  python -m yt_dlp --no-update --no-playlist -q --no-warnings \
    -f bestaudio/best \
    --extractor-args "youtube:player_client=android" \
    --extract-audio --audio-format vorbis --audio-quality 5 \
    --ffmpeg-location "$FF" \
    -o "${num}.%(ext)s" "https://www.youtube.com/watch?v=${ref}" 2>&1 | tail -2
  if [ -f "${num}.ogg" ]; then mv -f "${num}.ogg" "$out"; echo "  OK -> $out"; ok=$((ok+1));
  else echo "  FAIL $num"; fail=$((fail+1)); fi
done < _manifest.tsv
echo "=== done: ok=$ok fail=$fail ==="
