import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

# ===== 配置（无需命令行） =====
# 脚本所在目录作为根目录，递归处理所有子文件夹
ROOT_DIR = Path(__file__).resolve().parent

# 质量设置：默认不保留音频
KEEP_AUDIO = False

# 是否优先使用显卡编码
USE_GPU = True
GPU_ENCODER = "av1_nvenc"
GPU_PRESET = "p4"  # p1(最快) ~ p7(最慢)
GPU_CQ = 30        # 质量系数，数值越小质量越高

# 并行设置（效率拉满）
PARALLEL = True
MAX_WORKERS_CPU = max(1, (os.cpu_count() or 8))
MAX_WORKERS_GPU = 2

# CPU 编码设置（仅在不使用GPU或GPU不可用时）
LOSSLESS = False
CPU_ENCODER = "libvpx-vp9"
CRF = 20
SPEED = 2

# 延长设置：把同一个视频循环播放，直到指定秒数（不足才延长）
ENABLE_EXTEND = False
EXTEND_TO_SECONDS = 120.0


def find_video_files(root: Path, include_webm: bool) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".mp4") or (include_webm and lower.endswith(".webm")):
                files.append(Path(dirpath) / name)
    return files


def print_progress(current: int, total: int, filename: str) -> None:
    bar_len = 30
    filled = int(bar_len * current / max(total, 1))
    bar = "=" * filled + "-" * (bar_len - filled)
    percent = (current / max(total, 1)) * 100
    msg = f"[{bar}] {current}/{total} {percent:6.2f}% | {filename}"
    stream = sys.stderr
    if stream.isatty():
        stream.write("\r" + msg)
        stream.flush()
        if current == total:
            stream.write("\n")
    else:
        stream.write(msg + "\n")
        stream.flush()


def get_ffmpeg_path() -> str:
    local_ffmpeg = ROOT_DIR / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("未找到 ffmpeg。请将 ffmpeg.exe 放在脚本同目录或加入系统 PATH。")
    return ffmpeg


def get_ffprobe_path(ffmpeg: str) -> str:
    local_ffprobe = ROOT_DIR / "ffprobe.exe"
    if local_ffprobe.exists():
        return str(local_ffprobe)

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe

    ffmpeg_path = Path(ffmpeg)
    sibling = ffmpeg_path.with_name("ffprobe.exe")
    if sibling.exists():
        return str(sibling)

    raise FileNotFoundError("启用延长功能时未找到 ffprobe。请将 ffprobe.exe 放在脚本同目录或加入系统 PATH。")


def get_video_duration_seconds(ffprobe: str, src: Path) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(src),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    duration_text = result.stdout.strip()
    duration = float(duration_text)
    if duration <= 0:
        raise ValueError("视频时长无效")
    return duration


def has_encoder(ffmpeg: str, encoder: str) -> bool:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return encoder in result.stdout


def convert_one(
    ffmpeg: str,
    src: Path,
    dst: Path,
    crf: int,
    speed: int,
    lossless: bool,
    keep_audio: bool,
    use_gpu: bool,
    encoder: str,
    stream_loop: int | None,
    trim_seconds: float | None,
) -> None:
    cmd = [ffmpeg, "-y"]

    if stream_loop is not None and stream_loop > 0:
        cmd += ["-stream_loop", str(stream_loop)]

    cmd += [
        "-i",
        str(src),
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
    ]

    if use_gpu:
        cmd += ["-preset", GPU_PRESET, "-rc", "vbr", "-cq", str(GPU_CQ), "-b:v", "0"]
    else:
        if lossless:
            cmd += ["-lossless", "1", "-b:v", "0"]
        else:
            cmd += ["-crf", str(crf), "-b:v", "0"]
        cmd += ["-speed", str(speed), "-deadline", "good"]

    if keep_audio:
        cmd += ["-c:a", "libopus", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    if trim_seconds is not None and trim_seconds > 0:
        cmd += ["-t", f"{trim_seconds:.3f}"]

    cmd.append(str(dst))

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    root = ROOT_DIR
    try:
        ffmpeg = get_ffmpeg_path()
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1

    use_gpu = USE_GPU and has_encoder(ffmpeg, GPU_ENCODER)
    if USE_GPU and not use_gpu:
        print(f"未检测到 {GPU_ENCODER}，将回退到CPU编码。")
    if use_gpu and LOSSLESS:
        print("GPU 编码不支持无损，将改为高质量有损模式。")

    extend_enabled = ENABLE_EXTEND
    if extend_enabled and EXTEND_TO_SECONDS <= 0:
        print("延长功能已开启，但 EXTEND_TO_SECONDS <= 0，已自动关闭延长。")
        extend_enabled = False

    ffprobe: str | None = None
    if extend_enabled:
        try:
            ffprobe = get_ffprobe_path(ffmpeg)
            print(f"延长功能已开启：不足 {EXTEND_TO_SECONDS:.2f} 秒的视频会循环补足。")
        except FileNotFoundError as exc:
            print(f"{exc} 将关闭延长功能。")
            extend_enabled = False

    encoder = GPU_ENCODER if use_gpu else CPU_ENCODER
    lossless = False if use_gpu else LOSSLESS
    video_files = find_video_files(root, include_webm=extend_enabled)

    if not video_files:
        print("未找到可处理的视频文件。")
        return 0

    total = len(video_files)
    if extend_enabled:
        print(f"找到 {total} 个MP4/WebM文件，开始处理...")
    else:
        print(f"找到 {total} 个MP4文件，开始转换...")

    def task(src: Path) -> str | None:
        rel = src.relative_to(root)
        is_webm_input = src.suffix.lower() == ".webm"
        dst = src.with_suffix(".webm")
        temp_dst: Path | None = None

        stream_loop: int | None = None
        trim_seconds: float | None = None
        duration: float | None = None

        if extend_enabled:
            try:
                assert ffprobe is not None
                duration = get_video_duration_seconds(ffprobe, src)
                if duration < EXTEND_TO_SECONDS:
                    repeat_count = math.ceil(EXTEND_TO_SECONDS / duration)
                    stream_loop = max(0, repeat_count - 1)
                    trim_seconds = EXTEND_TO_SECONDS
            except (AssertionError, ValueError, subprocess.CalledProcessError):
                return f"读取时长失败: {rel}"

        if is_webm_input:
            if not extend_enabled:
                return None
            if duration is None or duration >= EXTEND_TO_SECONDS:
                return None
            temp_dst = src.with_name(f"{src.stem}.tmp-extend.webm")
            dst = temp_dst

        try:
            convert_one(
                ffmpeg,
                src,
                dst,
                CRF,
                SPEED,
                lossless,
                KEEP_AUDIO,
                use_gpu,
                encoder,
                stream_loop,
                trim_seconds,
            )

            if is_webm_input:
                assert temp_dst is not None
                temp_dst.replace(src)
            else:
                src.unlink(missing_ok=True)

            return None
        except (AssertionError, OSError, subprocess.CalledProcessError):
            if temp_dst is not None and temp_dst.exists():
                temp_dst.unlink(missing_ok=True)
            return f"转换失败: {rel}"

    completed = 0
    progress_lock = threading.Lock()
    if PARALLEL:
        max_workers = MAX_WORKERS_GPU if use_gpu else MAX_WORKERS_CPU
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(task, src): src for src in video_files}
            for future in as_completed(future_map):
                src = future_map[future]
                rel = src.relative_to(root)
                err = future.result()
                with progress_lock:
                    completed += 1
                    if err:
                        print(f"\n{err}")
                    print_progress(completed, total, rel.as_posix())
    else:
        for src in video_files:
            rel = src.relative_to(root)
            err = task(src)
            with progress_lock:
                completed += 1
                if err:
                    print(f"\n{err}")
                print_progress(completed, total, rel.as_posix())

    print("转换完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
