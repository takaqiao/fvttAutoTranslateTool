import os
from pathlib import Path

def list_files_to_txt(root: Path, output: Path) -> None:
    lines = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full_path = Path(dirpath) / name
            rel_path = full_path.relative_to(root)
            lines.append(str(rel_path).replace("\\", "/"))
    lines.sort()
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    root_dir = Path(".").resolve()
    output_file = root_dir / "files_list.txt"
    list_files_to_txt(root_dir, output_file)
    print(f"✅ 已写入: {output_file}")
