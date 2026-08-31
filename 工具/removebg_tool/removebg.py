"""
批量去背景脚本 - remove.bg API
用法: 把图片和本脚本放同一文件夹, 运行 `python removebg.py`
或:   python removebg.py <文件夹路径>
输出: 同目录下 `removed_bg/` 子文件夹, 格式 webp (透明), 最高质量
"""
import os
import sys
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

API_KEY = "vMBr2ofntazJAuNjGg6MRJJ2"
API_URL = "https://api.remove.bg/v1.0/removebg"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def process(folder: str):
    out_dir = os.path.join(folder, "removed_bg")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in EXTS
             and os.path.isfile(os.path.join(folder, f))]
    if not files:
        print(f"未找到图片: {folder}")
        return

    print(f"找到 {len(files)} 张图片, 输出到: {out_dir}")
    for i, name in enumerate(files, 1):
        src = os.path.join(folder, name)
        stem = os.path.splitext(name)[0]
        dst = os.path.join(out_dir, stem + ".webp")
        if os.path.exists(dst):
            print(f"[{i}/{len(files)}] 跳过(已存在): {name}")
            continue
        print(f"[{i}/{len(files)}] 处理: {name}", end=" ... ", flush=True)
        try:
            with open(src, "rb") as fh:
                resp = requests.post(
                    API_URL,
                    files={"image_file": fh},
                    data={
                        "size": "auto",      # 最高可用分辨率, 最高 25MP
                        "format": "webp",    # webp 带 alpha 透明
                        "type": "auto",
                        "channels": "rgba",
                    },
                    headers={"X-Api-Key": API_KEY},
                    timeout=120,
                )
            if resp.status_code == 200:
                with open(dst, "wb") as out:
                    out.write(resp.content)
                charged = resp.headers.get("X-Credits-Charged", "?")
                print(f"OK (credits: {charged})")
            else:
                print(f"失败 {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"异常: {e}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    process(os.path.abspath(folder))
