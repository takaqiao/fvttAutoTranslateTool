"""SoGMusic 源文件 -> .ogg 转换器 (libvorbis -q:a 5，与全库一致)

用法:
    python convert_ogg.py            # 按 MAP 转换（已存在的目标跳过）
    python convert_ogg.py --scan     # 只报告：哪些源文件同名 .ogg 不存在且不在 MAP 里
    python convert_ogg.py --dry-run  # 只打印将要执行的转换，不落盘

与旧版的区别（旧版已备份为 convert_ogg.py.bak）：
  1. 旧版第二行 os.chdir(r'C:\\Users\\Taka\\Desktop\\aba') 指向一个已经不存在的目录，
     导致整表 24 条全部 MISSING SRC。现在固定以本脚本所在目录（SoGMusic）为工作目录。
  2. ffmpeg 路径按脚本目录解析，不再依赖当前工作目录。
  3. 目标可以带子目录（UI 音效进 "UI Sound Effects/"），会自动建目录。
  4. 新增 --scan：列出可能漏转的源文件。

为什么不做全自动目录扫描：本库大量源文件在转换时被改过名
（例如 'World of Warcraft Soundtrack - Evil Forest [Night].mp3' -> 'WoW - Duskwood (Night).ogg'），
按文件名扫描会把 100 多个早已转好的文件误判为「未转换」而重复转一遍。
所以保持显式 MAP，用 --scan 辅助人工补录。
"""
import subprocess, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
FFMPEG = os.path.join(BASE, 'ffmpeg.exe')

MAP = {
    # ---- 2026-04-14 批次（历史记录，目标均已存在，运行时会 SKIP）----
    'Gloomhaven Theme 02 Bells.mp3': 'Gloomhaven - Bells.ogg',
    'Gloomhaven Theme 03 Dulcimer.mp3': 'Gloomhaven - Dulcimer.ogg',
    'Gloomhaven Theme 03 Snare Oboe.mp3': 'Gloomhaven - Snare Oboe.ogg',
    'YTDown.com_YouTube_Gloomhaven-Digital-OST-Title-Music_Media_MW1CREiZZH0_007_128k.mp3': 'Gloomhaven - Title Music.ogg',
    'Mists of Pandaria OST - Spirits B.mp3': 'WoW - Mists of Pandaria Spirits B.ogg',
    'World of Warcraft_ Battle for Azeroth OST - Kthir C.mp3': 'WoW - Kthir C.ogg',
    'World of Warcraft Soundtrack - Evil Forest [Day].mp3': 'WoW - Evil Forest (Day).ogg',
    'World of Warcraft Soundtrack - Evil Forest [Night].mp3': 'WoW - Duskwood (Night).ogg',
    'Flooded Suspense.mp3': 'Dishonored 2 - Flooded Suspense.ogg',
    'Divinity Original Sin 2 - The Doctors Basement (+Download Link).mp3': "Divinity Original Sin 2 - The Doctor's Basement.ogg",
    'Divinity Original Sin 2 - Cloisterwood (+Download Link).mp3': 'Divinity Original Sin 2 - Cloisterwood.ogg',
    'Divinity Original Sin 2 - Dungeon Music - Creepier Version (+Download Link).mp3': 'Divinity Original Sin 2 - Dungeon Music Creepier.ogg',
    'Divinity Original Sin 2 -  Blood Moon Island (+Download Link).mp3': 'Divinity Original Sin 2 - Blood Moon Island.ogg',
    'Divinity Original Sin 2 - The Nameless Isle  (+Download Link).mp3': 'Divinity Original Sin 2 - The Nameless Isle.ogg',
    "11 Baldur's Gate 3 Original Soundtrack - The Cult Of The Absolute.mp3": "Baldur's Gate 3 - The Absolute.ogg",
    'FF14 Stormblood - Shell Shocked.mp3': 'FFXIV Stormblood - Shell Shocked.ogg',
    "Assassin's Creed 2 OST _ Jesper Kyd - Approaching Target 1 (Track 06).mp3": "Assassin's Creed 2 - Approaching Target 1.ogg",
    "Assassin's Creed 2 OST _ Jesper Kyd - Approaching Target 2 (Track 07).mp3": "Assassin's Creed 2 - Approaching Target 2.ogg",
    "Dreadlord's Plight - Warcraft III_ The Frozen Throne [music].mp3": "Warcraft 3 - Dreadlord's Plight.ogg",
    'Grim Dawn_ Forgotten Gods Soundtrack - 21 - Korvan Sorrow.mp3': 'Grim Dawn - Korvan Sorrow.ogg',
    "Destroyer's Invocation (Full Version) - Halo 2 Soundtrack.mp3": "Halo 2 - Destroyer's Invocation.ogg",
    'Elden Ring OST 25 Consecrated Snowfield.mp3': 'Elden Ring - Consecrated Snowfield.ogg',
    'TES V Skyrim Soundtrack - Into Darkness.mp3': 'Skyrim - Into Darkness.ogg',
    'Shadows and Echoes.mp3': 'Skyrim - Shadows and Echoes.ogg',
    'TES V Skyrim Soundtrack - Towers and Shadows 4.mp3': 'Skyrim - Towers and Shadows.ogg',

    # ---- 2026-04-18 UI 音效批次：旧脚本从未覆盖，这 7 条一直没转 ----
    'A_magical_critical_s_#2-1776452505011.mp3': 'UI Sound Effects/A_magical_critical_s_#2-1776452505011.ogg',
    'Continuous_supernatu_#3-1776452908358.mp3': 'UI Sound Effects/Continuous_supernatu_#3-1776452908358.ogg',
    'Continuous_supernatu_#4-1776452904241.mp3': 'UI Sound Effects/Continuous_supernatu_#4-1776452904241.ogg',
    'Eerie_but_aesthetic__#3-1776452527466.mp3': 'UI Sound Effects/Eerie_but_aesthetic__#3-1776452527466.ogg',
    'Minimalist_Wuxia_cri_#3-1776452546799.mp3': 'UI Sound Effects/Minimalist_Wuxia_cri_#3-1776452546799.ogg',
    'The_ultimate_critica_#1-1776452475465.mp3': 'UI Sound Effects/The_ultimate_critica_#1-1776452475465.ogg',
    'The_ultimate_critica_#2-1776452485159.mp3': 'UI Sound Effects/The_ultimate_critica_#2-1776452485159.ogg',
}

SRC_EXT = {'.mp3', '.mp4', '.m4a', '.wav', '.flac', '.webm'}


def scan():
    """列出根目录里同名 .ogg 不存在、且不在 MAP 里的源文件。
    注意：这是按文件名判断，改过名的转换会被误报，需人工核对后补进 MAP。"""
    oggs = set()
    for dp, _, fn in os.walk(BASE):
        for f in fn:
            if f.lower().endswith('.ogg'):
                oggs.add(os.path.splitext(f)[0])
    suspects = []
    for f in sorted(os.listdir(BASE)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in SRC_EXT and f not in MAP and stem not in oggs:
            suspects.append(f)
    print(f'扫描到 {len(suspects)} 个可能漏转的源文件（按文件名判断，可能误报）：')
    for s in suspects:
        print('   ', s)
    print('\n核对后请把它们以 "源文件名": "目标.ogg" 的形式补进 MAP。')


def convert(dry=False):
    if not os.path.exists(FFMPEG):
        sys.exit(f'找不到 ffmpeg: {FFMPEG}')
    ok = skip = fail = miss = 0
    for src, dst in MAP.items():
        if not os.path.exists(src):
            print(f'MISSING SRC: {src}')
            miss += 1
            continue
        if os.path.exists(dst):
            print(f'SKIP (exists): {dst}')
            skip += 1
            continue
        d = os.path.dirname(dst)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        cmd = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error',
               '-i', src, '-c:a', 'libvorbis', '-q:a', '5', dst]
        if dry:
            print(f'DRY  {src}  ->  {dst}')
            ok += 1
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dst):
            print(f'OK  {dst}')
            ok += 1
        else:
            print(f'FAIL {dst}: {r.stderr.strip()[:200]}')
            fail += 1
    print(f'\nDone. ok={ok} skip={skip} fail={fail} missing_src={miss}')


if __name__ == '__main__':
    if '--scan' in sys.argv:
        scan()
    else:
        convert(dry='--dry-run' in sys.argv)
