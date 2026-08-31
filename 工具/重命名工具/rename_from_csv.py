import argparse
import os
import sys
from dataclasses import dataclass


@dataclass
class RenameResult:
    renamed: int = 0
    skipped_missing: int = 0
    skipped_exists: int = 0
    skipped_invalid: int = 0
    already_done: int = 0


def _normalize_rel_path(path: str) -> str:
    path = path.strip().strip("\ufeff")
    path = path.replace("/", os.sep).replace("\\", os.sep)
    return path.lstrip(os.sep)


def _parse_line(line: str):
    raw = line.strip()
    if not raw:
        return None
    if raw.endswith(","):
        raw = raw[:-1]
    if len(raw) >= 2 and ((raw[0] == raw[-1] == "\"") or (raw[0] == raw[-1] == "'")):
        raw = raw[1:-1]
    if "^" not in raw:
        return None
    left, right = raw.split("^", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None
    return left, right


def rename_from_csv(csv_path: str, base_dir: str, dry_run: bool, overwrite: bool, encoding: str) -> RenameResult:
    result = RenameResult()
    base_dir = os.path.abspath(base_dir)

    with open(csv_path, "r", encoding=encoding) as f:
        for line in f:
            parsed = _parse_line(line)
            if not parsed:
                result.skipped_invalid += 1
                continue

            left_raw, right_raw = parsed
            left_rel = _normalize_rel_path(left_raw)
            right_name = right_raw.strip()

            dir_rel = os.path.dirname(left_rel)
            left_base = os.path.splitext(os.path.basename(left_rel))[0]
            right_stem, right_ext = os.path.splitext(right_name)
            ext = right_ext or os.path.splitext(left_rel)[1]

            bilingual_name = f"{left_base} - {right_stem}{ext}"
            target_rel = os.path.join(dir_rel, bilingual_name)

            source_path = os.path.join(base_dir, dir_rel, right_name)
            alt_source_path = os.path.join(base_dir, dir_rel, f"{left_base}{ext}")
            target_path = os.path.join(base_dir, _normalize_rel_path(target_rel))

            source_to_use = source_path
            if not os.path.exists(source_to_use) and os.path.exists(alt_source_path):
                source_to_use = alt_source_path

            if not os.path.exists(source_to_use):
                if os.path.exists(target_path):
                    result.already_done += 1
                else:
                    result.skipped_missing += 1
                continue

            if os.path.exists(target_path) and not overwrite:
                result.skipped_exists += 1
                continue

            if not dry_run:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                os.replace(source_to_use, target_path)
            result.renamed += 1

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rename files based on the ^ mapping in a CSV.")
    parser.add_argument("--csv", help="Path to CSV mapping file.")
    parser.add_argument("--base", help="Base directory for the audio files.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig).")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without renaming.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite target files if they exist.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    exe_dir = os.path.abspath(os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__))
    default_csv = os.path.join(exe_dir, "索引.csv")

    csv_path = args.csv or default_csv
    base_dir = args.base or exe_dir

    result = rename_from_csv(
        csv_path=csv_path,
        base_dir=base_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        encoding=args.encoding,
    )

    print("完成：")
    print(f"  已重命名: {result.renamed}")
    print(f"  目标已存在跳过: {result.skipped_exists}")
    print(f"  源文件缺失跳过: {result.skipped_missing}")
    print(f"  已是目标名跳过: {result.already_done}")
    print(f"  无效行跳过: {result.skipped_invalid}")


if __name__ == "__main__":
    main()
