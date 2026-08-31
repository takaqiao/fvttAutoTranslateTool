"""Normalize tradition terms appearing in PROSE (publicNotes/descriptions/journal pages).

These are not in name-field exact-match form, so we use targeted prose patterns
identified manually. Run after normalize_tradition_terms.py for the cleanup tail.

Usage: python normalize_prose_traditions.py [target_dir]
"""
import json
import os
import sys

# Specific prose-string fixes (long-enough to be unambiguous)
REPLACEMENTS = {
    # Primal references in prose
    "反制原始法术": "反制原能法术",
    "9 环奥法或原能天生法术": "9 环奥术或原能内在法术",
    "9环奥法或原能天生法术": "9环奥术或原能内在法术",
    # Bard composition reference in prose ("Whitewall House" composition spell)
    "「幻墙之屋」即兴法术": "「幻墙之屋」组曲法术",
    # Witch focus spell in prose
    "巫术焦点法术": "巫术聚能法术",
    "「巫术焦点法术」": "「巫术聚能法术」",
}


def walk_replace(obj, stats):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            if isinstance(v, str):
                new_v = v
                for old, new in REPLACEMENTS.items():
                    if old in new_v:
                        new_v = new_v.replace(old, new)
                        stats["sub_replacements"] = stats.get("sub_replacements", 0) + 1
                if new_v != v:
                    obj[k] = new_v
                    stats["fields_changed"] += 1
            else:
                walk_replace(v, stats)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            v = obj[i]
            if isinstance(v, str):
                new_v = v
                for old, new in REPLACEMENTS.items():
                    if old in new_v:
                        new_v = new_v.replace(old, new)
                        stats["sub_replacements"] = stats.get("sub_replacements", 0) + 1
                if new_v != v:
                    obj[i] = new_v
                    stats["fields_changed"] += 1
            else:
                walk_replace(v, stats)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "FotRP/需要翻译"
    if not os.path.isdir(target):
        print(f"Target dir not found: {target}")
        sys.exit(1)
    total_subs = 0
    total_fields = 0
    files_changed = 0
    for root, dirs, files in os.walk(target):
        if any(s in root for s in ["NEW", "_backup", "_pdf", "_zh_synthetic", "_qa_reports"]):
            continue
        for f in files:
            if not f.endswith(".json"):
                continue
            fp = os.path.join(root, f)
            try:
                data = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            stats = {"fields_changed": 0, "sub_replacements": 0}
            walk_replace(data, stats)
            if stats["fields_changed"] > 0:
                with open(fp, "w", encoding="utf-8") as wf:
                    json.dump(data, wf, ensure_ascii=False, indent=2)
                print(f"  {f}: {stats['fields_changed']} fields, {stats['sub_replacements']} substring replacements")
                total_subs += stats["sub_replacements"]
                total_fields += stats["fields_changed"]
                files_changed += 1
    print(f"\nTotal: {total_subs} substring replacements in {total_fields} fields across {files_changed} files")


if __name__ == "__main__":
    main()
