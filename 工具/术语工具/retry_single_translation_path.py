import argparse
import importlib.util
import json
from pathlib import Path


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("crucible_translator", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_container_and_key(data: dict, full_path: str):
    path = full_path.strip()
    if path.startswith("root."):
        path = path[5:]
    parts = path.split(".")
    node = data
    for key in parts[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"路径不存在: {full_path} (缺少键: {key})")
        node = node[key]
    last_key = parts[-1]
    if not isinstance(node, dict) or last_key not in node:
        raise KeyError(f"路径不存在: {full_path} (缺少键: {last_key})")
    return node, last_key


def main():
    parser = argparse.ArgumentParser(description="使用 crucible_translator 单独补翻一个失败路径")
    parser.add_argument("--translator", default="翻译工具/crucible_translator.py", help="翻译脚本路径")
    parser.add_argument("--file", required=True, help="目标 JSON 文件路径")
    parser.add_argument("--path", required=True, help="词条路径，支持 root.xxx 形式")
    parser.add_argument("--no-keep-original", action="store_true", help="写回时不保留原文")
    args = parser.parse_args()

    mod = _load_module(Path(args.translator))
    json_path = Path(args.file)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    container, key = _resolve_container_and_key(data, args.path)
    old_value = container[key]
    if not isinstance(old_value, str):
        raise TypeError("目标路径值不是字符串，无法补翻")

    source_text = mod._simple_extract_original(old_value)

    glossary_data, _, _, _, _ = mod._simple_build_merged_glossary_data(
        mod.SIMPLE_GLOSSARY_PATH,
        mod.SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH,
    )
    glossary_buckets, glossary_count = mod._simple_load_glossary(
        mod.SIMPLE_GLOSSARY_PATH,
        glossary_data=glossary_data,
    )

    pre_text, matched = mod._simple_apply_glossary(source_text, glossary_buckets)
    print(f"source_len={len(source_text)} glossary_count={glossary_count} matched_terms={len(matched)}")

    translator = mod._SimpleTranslator()
    cn = translator.translate(pre_text, args.path, matched)
    if not cn.strip():
        raise RuntimeError("翻译结果为空，未写回")

    keep_original = not args.no_keep_original
    new_value = mod._simple_merge_translation(args.path, cn, source_text, keep_original=keep_original)
    container[key] = new_value

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK: 单条补翻成功并已写回")


if __name__ == "__main__":
    main()
