import argparse
import copy
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LINK_PATTERN = re.compile(
    r"@(?P<kind>UUID|Compendium|JournalEntry|Item|Actor|Macro|Playlist|RollTable)"
    r"\[(?P<id>[^\]]+)\](?:\{(?P<label>[^}]*)\})?"
)


def contains_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def contains_en(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def split_bilingual(text: str) -> tuple[str, str | None]:
    if not isinstance(text, str) or "\n" not in text:
        return text, None
    first, rest = text.split("\n", 1)
    if contains_zh(first) and contains_en(rest):
        return first, rest
    return text, None


def format_path(parts: list[Any]) -> str:
    out = "root"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return out


def parse_path(path_str: str) -> list[Any]:
    if not path_str.startswith("root"):
        raise ValueError(f"Invalid path: {path_str}")
    tail = path_str[4:]
    parts: list[Any] = []
    index = 0
    while index < len(tail):
        ch = tail[index]
        if ch == ".":
            index += 1
            start = index
            while index < len(tail) and tail[index] not in ".[":
                index += 1
            parts.append(tail[start:index])
            continue
        if ch == "[":
            index += 1
            start = index
            while index < len(tail) and tail[index] != "]":
                index += 1
            parts.append(int(tail[start:index]))
            index += 1
            continue
        index += 1
    return parts


def collect_strings(node: Any, parts: list[Any] | None = None, out: list[tuple[list[Any], str, str]] | None = None):
    if parts is None:
        parts = []
    if out is None:
        out = []
    if isinstance(node, dict):
        for key, value in node.items():
            collect_strings(value, parts + [key], out)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            collect_strings(value, parts + [idx], out)
    elif isinstance(node, str):
        out.append((parts, format_path(parts), node))
    return out


def get_value(data: Any, parts: list[Any]) -> Any:
    cur = data
    for p in parts:
        cur = cur[p]
    return cur


def set_value(data: Any, parts: list[Any], value: Any):
    if not parts:
        return
    cur = data
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


@dataclass
class LinkToken:
    ref: str
    label: str


def extract_link_tokens(text: str) -> list[LinkToken]:
    tokens: list[LinkToken] = []
    for match in LINK_PATTERN.finditer(text or ""):
        ref = f"{match.group('kind')}[{match.group('id')}]"
        label = (match.group("label") or match.group("id") or "").strip()
        if label:
            tokens.append(LinkToken(ref=ref, label=label))
    return tokens


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def command_extract(args: argparse.Namespace):
    target_path = Path(args.target)
    out_json_path = Path(args.out_json)
    out_csv_path = Path(args.out_csv) if args.out_csv else None

    data = load_json(target_path)
    items = collect_strings(data)
    terms: dict[str, dict[str, Any]] = {}

    for _, path_str, value in items:
        zh_part, en_part = split_bilingual(value)
        if not en_part:
            continue
        zh_tokens = extract_link_tokens(zh_part)
        en_tokens = extract_link_tokens(en_part)

        zh_by_ref: dict[str, list[str]] = {}
        en_by_ref: dict[str, list[str]] = {}

        for token in zh_tokens:
            if contains_zh(token.label):
                zh_by_ref.setdefault(token.ref, []).append(token.label)
        for token in en_tokens:
            if contains_en(token.label):
                en_by_ref.setdefault(token.ref, []).append(token.label)

        refs = set(zh_by_ref.keys()) & set(en_by_ref.keys())
        for ref in refs:
            for en_label in en_by_ref[ref]:
                norm_en = re.sub(r"\s+", " ", en_label).strip()
                if not norm_en:
                    continue
                entry = terms.setdefault(
                    norm_en,
                    {
                        "english": norm_en,
                        "preferred_zh": "",
                        "zh_variants": {},
                        "refs": {},
                        "paths": {},
                    },
                )
                entry["refs"][ref] = entry["refs"].get(ref, 0) + 1
                entry["paths"][path_str] = entry["paths"].get(path_str, 0) + 1
                for zh_label in zh_by_ref[ref]:
                    entry["zh_variants"][zh_label] = entry["zh_variants"].get(zh_label, 0) + 1

    rows = []
    for english, entry in terms.items():
        variants_sorted = sorted(entry["zh_variants"].items(), key=lambda item: item[1], reverse=True)
        preferred = variants_sorted[0][0] if variants_sorted else ""
        rows.append(
            {
                "english": english,
                "preferred_zh": preferred,
                "zh_variants": {k: v for k, v in variants_sorted},
                "refs": dict(sorted(entry["refs"].items(), key=lambda item: item[1], reverse=True)),
                "sample_paths": list(sorted(entry["paths"].keys()))[:20],
                "total_occurrences": sum(entry["paths"].values()),
            }
        )

    rows.sort(key=lambda item: item["total_occurrences"], reverse=True)
    output = {
        "meta": {
            "source_file": str(target_path),
            "total_terms": len(rows),
            "note": "你可以直接修改 preferred_zh，然后用 apply 回灌到 JSON。",
        },
        "terms": rows,
    }
    save_json(out_json_path, output)

    if out_csv_path:
        with out_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["english", "preferred_zh", "variants", "occurrences"])
            for row in rows:
                variants = " | ".join([f"{k}:{v}" for k, v in row["zh_variants"].items()])
                writer.writerow([row["english"], row["preferred_zh"], variants, row["total_occurrences"]])

    print(f"✅ 提取完成：{out_json_path} | 术语数: {len(rows)}")
    if out_csv_path:
        print(f"✅ CSV 已导出：{out_csv_path}")


def load_terms_json(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("terms"), list):
        return data["terms"]
    if isinstance(data, list):
        return data
    raise ValueError("术语文件格式错误，应包含 terms 数组")


def build_replace_rules(terms: list[dict[str, Any]], include_english: bool) -> list[tuple[str, str, bool]]:
    rules: list[tuple[str, str, bool]] = []
    for term in terms:
        preferred = str(term.get("preferred_zh", "")).strip()
        english = str(term.get("english", "")).strip()
        if not preferred:
            continue

        aliases = set()
        variants = term.get("zh_variants", {})
        if isinstance(variants, dict):
            for alias in variants.keys():
                alias_s = str(alias).strip()
                if alias_s and alias_s != preferred:
                    aliases.add(alias_s)
        if isinstance(term.get("aliases"), list):
            for alias in term["aliases"]:
                alias_s = str(alias).strip()
                if alias_s and alias_s != preferred:
                    aliases.add(alias_s)

        for alias in sorted(aliases, key=len, reverse=True):
            rules.append((alias, preferred, False))

        if include_english and english and english != preferred:
            rules.append((english, preferred, True))

    rules.sort(key=lambda item: len(item[0]), reverse=True)
    return rules


def replace_by_rules(text: str, rules: list[tuple[str, str, bool]]) -> str:
    new_text = text
    for src, dst, english_mode in rules:
        if src == dst or not src:
            continue
        if english_mode:
            pattern = re.compile(rf"(?<![A-Za-z]){re.escape(src)}(?![A-Za-z])")
            new_text = pattern.sub(dst, new_text)
        else:
            new_text = new_text.replace(src, dst)
    return new_text


def command_apply(args: argparse.Namespace):
    target_path = Path(args.target)
    terms_path = Path(args.terms)
    backup_path = Path(args.backup)

    data = load_json(target_path)
    original = copy.deepcopy(data)
    terms = load_terms_json(terms_path)
    rules = build_replace_rules(terms, include_english=args.include_english)

    changed = []
    for parts, path_str, value in collect_strings(data):
        zh_part, en_part = split_bilingual(value)
        if en_part is not None and not args.apply_english_part:
            new_zh = replace_by_rules(zh_part, rules)
            new_value = new_zh + "\n" + en_part
        else:
            new_value = replace_by_rules(value, rules)

        if new_value != value:
            set_value(data, parts, new_value)
            changed.append(
                {
                    "path": path_str,
                    "before": get_value(original, parts),
                    "after": new_value,
                }
            )

    save_json(target_path, data)
    backup_payload = {
        "meta": {
            "target": str(target_path),
            "changed_count": len(changed),
        },
        "changes": changed,
    }
    save_json(backup_path, backup_payload)
    print(f"✅ 回灌完成：{target_path} | 修改: {len(changed)}")
    print(f"✅ 还原文件已生成：{backup_path}")


def command_restore(args: argparse.Namespace):
    target_path = Path(args.target)
    backup_path = Path(args.backup)

    data = load_json(target_path)
    backup = load_json(backup_path)
    changes = backup.get("changes", []) if isinstance(backup, dict) else []

    restored = 0
    for item in changes:
        path_str = item.get("path", "")
        before = item.get("before")
        if not path_str:
            continue
        parts = parse_path(path_str)
        set_value(data, parts, before)
        restored += 1

    save_json(target_path, data)
    print(f"✅ 已还原：{target_path} | 回退条目: {restored}")


def load_glossary_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    result = {}
    for key, value in data.items():
        if isinstance(value, list):
            if value:
                result[str(key).strip()] = str(value[0]).strip()
        else:
            result[str(key).strip()] = str(value).strip()
    return result


def command_ai_suggest(args: argparse.Namespace):
    terms_path = Path(args.terms)
    glossary_path = Path(args.glossary) if args.glossary else None
    model = args.model
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = args.base_url or "https://api.openai.com/v1"

    if not api_key:
        raise RuntimeError("未提供 API Key，请通过 --api-key 或环境变量 OPENAI_API_KEY 设置")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("未安装 openai 包，请先安装: pip install openai") from exc

    glossary = load_glossary_dict(glossary_path) if glossary_path else {}

    payload = load_json(terms_path)
    terms = payload.get("terms", []) if isinstance(payload, dict) else []
    if not isinstance(terms, list):
        raise ValueError("术语文件格式错误")

    client = OpenAI(api_key=api_key, base_url=base_url)

    remaining = []
    for item in terms:
        english = str(item.get("english", "")).strip()
        if not english:
            continue
        if english in glossary and glossary[english]:
            item["preferred_zh"] = glossary[english]
            continue
        if not str(item.get("preferred_zh", "")).strip():
            remaining.append(item)

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        items = []
        for term in batch:
            variants = term.get("zh_variants", {})
            top_variants = sorted(variants.items(), key=lambda item: item[1], reverse=True)[:6] if isinstance(variants, dict) else []
            items.append(
                {
                    "english": term.get("english", ""),
                    "candidates": [v[0] for v in top_variants],
                }
            )

        instructions = (
            "你是PF2e与奇幻TRPG本地化术语统一助手。"
            "请为每个英文术语给出最稳妥、统一、简体中文译名。"
            "优先延续候选译法风格，不要创造过长译名。"
            "输出严格 JSON 数组，每项格式: {\"english\":\"...\",\"preferred_zh\":\"...\"}。"
        )
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(items, ensure_ascii=False),
        )
        raw = (response.output_text or "").strip()

        try:
            suggested = json.loads(raw)
        except Exception:
            match = re.search(r"\[.*\]", raw, flags=re.S)
            if not match:
                continue
            suggested = json.loads(match.group(0))

        if not isinstance(suggested, list):
            continue

        by_en = {str(item.get("english", "")).strip(): item for item in batch}
        for row in suggested:
            if not isinstance(row, dict):
                continue
            english = str(row.get("english", "")).strip()
            preferred = str(row.get("preferred_zh", "")).strip()
            if english in by_en and preferred:
                by_en[english]["preferred_zh"] = preferred

        print(f"AI 建议进度: {min(start + batch_size, len(remaining))}/{len(remaining)}")

    if isinstance(payload, dict):
        payload["terms"] = terms
    else:
        payload = {"terms": terms}
    save_json(terms_path, payload)
    print(f"✅ AI 建议完成并写回：{terms_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FVTT 术语治理工具：抽取 / 统一 / 回灌 / 回退")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="从双语 JSON 中提炼术语候选")
    p_extract.add_argument("--target", required=True, help="目标 JSON")
    p_extract.add_argument("--out-json", default="terms_candidates.json", help="术语 JSON 输出路径")
    p_extract.add_argument("--out-csv", default="terms_candidates.csv", help="术语 CSV 输出路径")
    p_extract.set_defaults(func=command_extract)

    p_apply = sub.add_parser("apply", help="将术语表回灌到 JSON")
    p_apply.add_argument("--target", required=True, help="目标 JSON")
    p_apply.add_argument("--terms", required=True, help="术语 JSON（可编辑 preferred_zh）")
    p_apply.add_argument("--backup", default="term_apply_backup.json", help="回退文件输出")
    p_apply.add_argument("--include-english", action="store_true", help="同时将英文术语替换为中文")
    p_apply.add_argument("--apply-english-part", action="store_true", help="双语文本中连英文段也一起替换")
    p_apply.set_defaults(func=command_apply)

    p_restore = sub.add_parser("restore", help="按 backup 文件回退术语回灌")
    p_restore.add_argument("--target", required=True, help="目标 JSON")
    p_restore.add_argument("--backup", required=True, help="apply 阶段生成的回退文件")
    p_restore.set_defaults(func=command_restore)

    p_ai = sub.add_parser("ai-suggest", help="使用 AI 为术语表自动补全 preferred_zh")
    p_ai.add_argument("--terms", required=True, help="术语 JSON")
    p_ai.add_argument("--glossary", default="glossary.json", help="已有术语表（优先覆盖）")
    p_ai.add_argument("--model", default="gpt-5.2", help="模型名")
    p_ai.add_argument("--batch-size", type=int, default=40, help="AI 批次大小")
    p_ai.add_argument("--api-key", default="", help="OpenAI API Key，可留空走环境变量")
    p_ai.add_argument("--base-url", default="https://api.openai.com/v1", help="OpenAI Base URL")
    p_ai.set_defaults(func=command_ai_suggest)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
