import argparse
import json
import re
from pathlib import Path


TOKEN_RE = re.compile(
    r"@(?:UUID|Compendium|Localize)\[[^\]]+\](?:\{[^{}]*\})?|\[\[[^\]]+\]\]",
    flags=re.IGNORECASE,
)
UUID_TOKEN_RE = re.compile(r"@UUID\[[^\]]+\](?:\{[^{}]*\})?", flags=re.IGNORECASE)


def extract_tokens(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    return [m.group(0) for m in TOKEN_RE.finditer(text)]


def replace_tokens_by_sequence(target_text: str, source_tokens: list[str]) -> tuple[str, bool, str]:
    if not isinstance(target_text, str):
        return target_text, False, "target-not-string"
    if not source_tokens:
        return target_text, False, "source-no-token"

    target_matches = list(TOKEN_RE.finditer(target_text))
    if not target_matches:
        return target_text, False, "target-no-token"

    source_count = len(source_tokens)
    target_count = len(target_matches)

    if target_count == source_count:
        mode = "one-to-one"
    elif source_count > 0 and target_count % source_count == 0:
        mode = "repeat-source-sequence"
    else:
        return target_text, False, "token-count-mismatch"

    new_parts = []
    cursor = 0
    changed = False

    for i, m in enumerate(target_matches):
        if mode == "one-to-one":
            replacement = source_tokens[i]
        else:
            replacement = source_tokens[i % source_count]
        old = m.group(0)
        new_parts.append(target_text[cursor:m.start()])
        new_parts.append(replacement)
        cursor = m.end()
        if old != replacement:
            changed = True

    new_parts.append(target_text[cursor:])
    return "".join(new_parts), changed, mode


def repair_node(source_node, target_node, path="root", stats=None):
    if stats is None:
        stats = {
            "visited_strings": 0,
            "source_has_tokens": 0,
            "target_has_tokens": 0,
            "changed": 0,
            "changed_one_to_one": 0,
            "changed_repeat_source_sequence": 0,
            "count_mismatch_skipped": 0,
            "missing_path": 0,
            "type_mismatch": 0,
            "skip_details": [],
            "changed_paths": [],
        }

    if isinstance(source_node, dict):
        if not isinstance(target_node, dict):
            stats["type_mismatch"] += 1
            return target_node, stats
        for k, sv in source_node.items():
            if k not in target_node:
                stats["missing_path"] += 1
                continue
            tv = target_node[k]
            target_node[k], stats = repair_node(sv, tv, f"{path}.{k}", stats)
        return target_node, stats

    if isinstance(source_node, list):
        if not isinstance(target_node, list):
            stats["type_mismatch"] += 1
            return target_node, stats
        n = min(len(source_node), len(target_node))
        if len(source_node) != len(target_node):
            stats["type_mismatch"] += 1
        for i in range(n):
            target_node[i], stats = repair_node(source_node[i], target_node[i], f"{path}[{i}]", stats)
        return target_node, stats

    if isinstance(source_node, str) and isinstance(target_node, str):
        stats["visited_strings"] += 1
        source_tokens = extract_tokens(source_node)
        target_tokens = extract_tokens(target_node)

        if source_tokens:
            stats["source_has_tokens"] += 1
        if target_tokens:
            stats["target_has_tokens"] += 1

        if source_tokens and target_tokens:
            replaced, changed, status = replace_tokens_by_sequence(target_node, source_tokens)
            if status == "token-count-mismatch":
                stats["count_mismatch_skipped"] += 1
                stats["skip_details"].append({
                    "path": path,
                    "reason": "token-count-mismatch",
                    "source_token_count": len(source_tokens),
                    "target_token_count": len(target_tokens),
                    "is_multiple": (len(target_tokens) % len(source_tokens) == 0) if len(source_tokens) > 0 else False,
                    "source_uuid_count": len(UUID_TOKEN_RE.findall(source_node)),
                    "target_uuid_count": len(UUID_TOKEN_RE.findall(target_node)),
                    "source_token_sample": source_tokens[:3],
                    "target_token_sample": target_tokens[:3],
                })
                return target_node, stats
            if changed and status in {"one-to-one", "repeat-source-sequence"}:
                stats["changed"] += 1
                if status == "one-to-one":
                    stats["changed_one_to_one"] += 1
                else:
                    stats["changed_repeat_source_sequence"] += 1
                stats["changed_paths"].append({
                    "path": path,
                    "mode": status,
                    "source_token_count": len(source_tokens),
                    "target_token_count": len(target_tokens),
                })
            return replaced, stats

    return target_node, stats


def _collect_uuid_issues(node, path="root", out=None):
    if out is None:
        out = {
            "malformed_uuid_paths": [],
            "uuid_occurrence_paths": [],
        }

    if isinstance(node, dict):
        for k, v in node.items():
            _collect_uuid_issues(v, f"{path}.{k}", out)
        return out

    if isinstance(node, list):
        for i, v in enumerate(node):
            _collect_uuid_issues(v, f"{path}[{i}]", out)
        return out

    if isinstance(node, str):
        uuid_raw_count = len(re.findall(r"@UUID\[", node, flags=re.IGNORECASE))
        uuid_token_count = len(UUID_TOKEN_RE.findall(node))
        if uuid_token_count > 0:
            out["uuid_occurrence_paths"].append({
                "path": path,
                "uuid_token_count": uuid_token_count,
                "uuid_token_sample": UUID_TOKEN_RE.findall(node)[:3],
            })
        if uuid_raw_count != uuid_token_count:
            out["malformed_uuid_paths"].append({
                "path": path,
                "raw_uuid_marker_count": uuid_raw_count,
                "parsed_uuid_token_count": uuid_token_count,
                "text_sample": node[:240],
            })
    return out


def main():
    parser = argparse.ArgumentParser(description="按英文源文件批量修复 merged 文件中的 UUID/Compendium/Localize 链接 token")
    parser.add_argument(
        "--source",
        default="pf2e-abomination-vaults.av.json",
        help="英文源 JSON 路径",
    )
    parser.add_argument(
        "--target",
        default="pf2e-abomination-vaults.av-merged.json",
        help="待修复的 merged JSON 路径",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出路径；留空则覆盖 target",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计，不写文件",
    )
    parser.add_argument(
        "--report",
        default="uuid_repair_report.json",
        help="报告输出路径（JSON）",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=200,
        help="报告中保留的明细条数上限",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)
    output_path = Path(args.output) if args.output else target_path

    if not source_path.exists():
        raise FileNotFoundError(f"source 文件不存在: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"target 文件不存在: {target_path}")

    source_data = json.loads(source_path.read_text(encoding="utf-8-sig"))
    target_data = json.loads(target_path.read_text(encoding="utf-8-sig"))

    repaired, stats = repair_node(source_data, target_data)
    issues_before = _collect_uuid_issues(target_data)
    issues_after = _collect_uuid_issues(repaired)

    report = {
        "source": str(source_path),
        "target": str(target_path),
        "output": str(output_path),
        "dry_run": bool(args.dry_run),
        "stats": {
            k: v for k, v in stats.items() if k not in {"skip_details", "changed_paths"}
        },
        "skip_reason_summary": {
            "token-count-mismatch": stats["count_mismatch_skipped"],
        },
        "skip_details": stats["skip_details"][: max(args.max_details, 0)],
        "changed_paths": stats["changed_paths"][: max(args.max_details, 0)],
        "uuid_issues_before": {
            "malformed_uuid_count": len(issues_before["malformed_uuid_paths"]),
            "uuid_occurrence_path_count": len(issues_before["uuid_occurrence_paths"]),
            "malformed_uuid_paths": issues_before["malformed_uuid_paths"][: max(args.max_details, 0)],
        },
        "uuid_issues_after": {
            "malformed_uuid_count": len(issues_after["malformed_uuid_paths"]),
            "uuid_occurrence_path_count": len(issues_after["uuid_occurrence_paths"]),
            "malformed_uuid_paths": issues_after["malformed_uuid_paths"][: max(args.max_details, 0)],
        },
    }

    print("=== UUID Token Repair Stats ===")
    for k, v in stats.items():
        if k in {"skip_details", "changed_paths"}:
            continue
        print(f"{k}: {v}")
    if stats["count_mismatch_skipped"]:
        print("skip reason: token-count-mismatch (源与目标 token 数量非相等且非整倍数，已跳过避免错位替换)")
    print(f"malformed_uuid_before: {len(issues_before['malformed_uuid_paths'])}")
    print(f"malformed_uuid_after: {len(issues_after['malformed_uuid_paths'])}")

    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入: {report_path}")

    if args.dry_run:
        print("dry-run: 未写入文件")
        return

    output_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {output_path}")


if __name__ == "__main__":
    main()
