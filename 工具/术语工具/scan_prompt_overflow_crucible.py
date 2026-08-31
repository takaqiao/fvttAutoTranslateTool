import argparse
import importlib.util
import json
import re
from pathlib import Path


def _load_translator_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("crucible_translator", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载翻译器脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_strings(node, parts=None, out=None):
    if parts is None:
        parts = []
    if out is None:
        out = []

    if isinstance(node, dict):
        for k, v in node.items():
            _iter_strings(v, parts + [str(k)], out)
    elif isinstance(node, list):
        for idx, v in enumerate(node):
            _iter_strings(v, parts + [f"[{idx}]"], out)
    elif isinstance(node, str):
        out.append((".".join(parts), node))

    return out


def _short(text: str, n: int = 120) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t[:n]


def main():
    parser = argparse.ArgumentParser(description="扫描 crucible-cn 下术语候选超阈值的词条")
    parser.add_argument("--root", default="crucible-cn", help="扫描根目录")
    parser.add_argument("--glob", default="**/*.json", help="文件匹配模式")
    parser.add_argument("--threshold", type=int, default=24, help="候选术语阈值")
    parser.add_argument("--translator", default="翻译工具/crucible_translator.py", help="翻译脚本路径")
    parser.add_argument("--output", default="prompt_overflow_report.json", help="输出报告 JSON")
    args = parser.parse_args()

    translator_path = Path(args.translator)
    module = _load_translator_module(translator_path)

    glossary_data, _, _, _, _ = module._simple_build_merged_glossary_data(
        module.SIMPLE_GLOSSARY_PATH,
        module.SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH,
    )
    glossary_buckets, glossary_count = module._simple_load_glossary(
        module.SIMPLE_GLOSSARY_PATH,
        glossary_data=glossary_data,
    )

    if glossary_count <= 0:
        print("⚠️ 当前加载术语数为 0，结果可能为空。请检查 glossary_path / extra_glossary_path")

    root = Path(args.root)
    files = sorted(root.glob(args.glob))

    overflow_rows = []
    by_file = {}

    for file_path in files:
        if not file_path.is_file() or file_path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        local_overflow = 0
        local_max = 0

        for path_str, value in _iter_strings(data):
            if not isinstance(value, str) or not value.strip():
                continue
            if not module._simple_contains_english(value):
                continue

            _, matched = module._simple_apply_glossary(value, glossary_buckets)
            count = len(matched)
            if count > local_max:
                local_max = count

            if count > args.threshold:
                local_overflow += 1
                overflow_rows.append(
                    {
                        "file": file_path.as_posix(),
                        "path": path_str,
                        "matched_terms": count,
                        "text_preview": _short(value),
                    }
                )

        if local_overflow > 0:
            by_file[file_path.as_posix()] = {
                "overflow_entries": local_overflow,
                "max_matched_terms": local_max,
            }

    overflow_rows.sort(key=lambda x: x["matched_terms"], reverse=True)
    by_file_sorted = sorted(
        by_file.items(),
        key=lambda kv: (kv[1]["overflow_entries"], kv[1]["max_matched_terms"]),
        reverse=True,
    )

    report = {
        "meta": {
            "root": root.as_posix(),
            "glob": args.glob,
            "threshold": args.threshold,
            "loaded_glossary_terms": glossary_count,
            "overflow_entry_count": len(overflow_rows),
            "overflow_file_count": len(by_file),
        },
        "files": [
            {
                "file": file_name,
                "overflow_entries": info["overflow_entries"],
                "max_matched_terms": info["max_matched_terms"],
            }
            for file_name, info in by_file_sorted
        ],
        "entries": overflow_rows,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已扫描文件: {len(files)}")
    print(f"术语总数: {glossary_count}")
    print(f"超阈值词条: {len(overflow_rows)}")
    print(f"涉及文件数: {len(by_file)}")
    print(f"报告输出: {out_path}")


if __name__ == "__main__":
    main()
