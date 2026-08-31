import json
from pathlib import Path

# ====== 配置区（按需修改）======
BASE_DIR = Path(r"C:\Users\Taka\Desktop\fvtt")
# 多文件输入：每个元素是 (文件名, pack key)
INPUTS = [
    ("abomination-vaults-addons.abomination-vaults-addons.json", "abomination-vaults-addons.abomination-vaults-addons"),
    ("abomination-vaults-expanded.abomination-vaults-expanded.json", "abomination-vaults-expanded.abomination-vaults-expanded"),
    ("footsteps-of-otari.footsteps-of-otari-Journals.json", "footsteps-of-otari.footsteps-of-otari-Journals"),
    ("pf2e-mercenary-marketplace-vol1.mmv1-journals.json", "pf2e-mercenary-marketplace-vol1.mmv1-journals"),
    ("wise-gaming-promoter-pack-242.2421-pack.json", "wise-gaming-promoter-pack-242.2421-pack"),
]
# 输出文件名（将写入 BASE_DIR）
LABELS_OUTPUT_FILE = "labels.json"
TITLES_OUTPUT_FILE = "titles.json"
# ============================

def extract_titles_from_entries(entries) -> dict:
    titles = {}
    if not isinstance(entries, dict):
        return titles

    for key, entry in entries.items():
        if isinstance(entry, dict):
            name = entry.get("name")
        else:
            name = None
        if isinstance(key, str) and key:
            titles[key] = name if name else key

    return titles


def extract_pack_data(input_path: Path, pack_key: str) -> dict:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    entries = data.get("entries", {})
    folders = data.get("folders", {})
    titles = extract_titles_from_entries(entries)
    label = data.get("label") if isinstance(data, dict) else None
    return {
        "pack_key": pack_key,
        "label": label if isinstance(label, str) and label else None,
        "titles": titles,
        "folders": folders if isinstance(folders, dict) else {},
    }

def main() -> None:
    labels = {}
    titles_index = {}

    for file_name, pack_key in INPUTS:
        input_path = BASE_DIR / file_name
        pack_data = extract_pack_data(input_path, pack_key)

        if pack_data["label"]:
            labels[pack_key] = pack_data["label"]

        titles_index[pack_key] = {
            "titles": pack_data["titles"],
            "folders": pack_data["folders"],
        }

    labels_output_path = BASE_DIR / LABELS_OUTPUT_FILE
    titles_output_path = BASE_DIR / TITLES_OUTPUT_FILE

    labels_output_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    titles_output_path.write_text(
        json.dumps(titles_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
