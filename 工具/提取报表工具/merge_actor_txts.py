import argparse
import re
from pathlib import Path


KIND_ORDER = ["abc", "actions", "inventory"]


def _detect_actor_files(directory: Path) -> dict[str, dict[str, Path]]:
    """Return mapping: actor_prefix -> {kind -> file_path}."""
    # Example: "艾兹伦 Ezren - actor-abc.txt"
    pattern = re.compile(r"^(?P<prefix>.+?)\s*-\s*actor-(?P<kind>abc|actions|inventory)\.txt$", re.IGNORECASE)

    groups: dict[str, dict[str, Path]] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue

        prefix = match.group("prefix").strip()
        kind = match.group("kind").lower()
        groups.setdefault(prefix, {})[kind] = path

    return groups


def _read_text(path: Path, encoding: str) -> str:
    # utf-8-sig strips BOM if present.
    try_encodings = [encoding]
    if encoding.lower() == "utf-8":
        try_encodings = ["utf-8-sig", "utf-8"]

    last_error: Exception | None = None
    for enc in try_encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise last_error  # type: ignore[misc]


def merge_one_actor(
    prefix: str,
    files_by_kind: dict[str, Path],
    output_dir: Path,
    output_tag: str,
    separator: str,
    with_headers: bool,
    require_all: bool,
    encoding: str,
    overwrite: bool,
) -> Path | None:
    missing = [kind for kind in KIND_ORDER if kind not in files_by_kind]
    if missing:
        msg = f"[{prefix}] missing: {', '.join(missing)}"
        if require_all:
            print(msg + " -> skipped")
            return None
        print(msg + " -> merging what exists")

    output_path = output_dir / f"{prefix} - {output_tag}.txt"
    if output_path.exists() and not overwrite:
        print(f"[{prefix}] output exists -> skipped: {output_path.name}")
        return None

    parts: list[str] = []
    for kind in KIND_ORDER:
        path = files_by_kind.get(kind)
        if not path:
            continue
        content = _read_text(path, encoding=encoding)
        content = content.rstrip("\n")
        if with_headers:
            parts.append(f"=== actor-{kind} ===\n{content}".rstrip("\n"))
        else:
            parts.append(content)

    merged = separator.join([p for p in parts if p != ""]) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(merged, encoding=encoding)
    print(f"[{prefix}] merged -> {output_path.name}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-merge actor txt files like: '<Name> - actor-abc.txt', "
            "'<Name> - actor-actions.txt', '<Name> - actor-inventory.txt'."
        )
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Directory containing the txt files (default: current directory)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: same as --dir)",
    )
    parser.add_argument(
        "--output-tag",
        default="actor-merged",
        help="Output filename tag after prefix (default: actor-merged)",
    )
    parser.add_argument(
        "--no-headers",
        action="store_true",
        help="Do not add section headers like '=== actor-abc ==='",
    )
    parser.add_argument(
        "--separator",
        default="\n\n",
        help="Separator inserted between sections (default: blank line)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for read/write (default: utf-8; reads utf-8-sig too)",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require all 3 files to exist; otherwise skip that actor",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )

    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    output_dir = Path(args.output_dir) if args.output_dir else directory

    groups = _detect_actor_files(directory)
    if not groups:
        print("No matching files found.")
        print("Expected names like: '艾兹伦 Ezren - actor-abc.txt'")
        return

    merged_count = 0
    for prefix, files_by_kind in sorted(groups.items(), key=lambda kv: kv[0]):
        out = merge_one_actor(
            prefix=prefix,
            files_by_kind=files_by_kind,
            output_dir=output_dir,
            output_tag=args.output_tag,
            separator=args.separator,
            with_headers=not args.no_headers,
            require_all=args.require_all,
            encoding=args.encoding,
            overwrite=args.overwrite,
        )
        if out is not None:
            merged_count += 1

    print(f"Done. Merged {merged_count} actor(s).")


if __name__ == "__main__":
    main()
