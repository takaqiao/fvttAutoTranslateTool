import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "工具" / "翻译流程" / "scripts" / "build_3source_tm.py"


def load_script():
    spec = importlib.util.spec_from_file_location("build_3source_tm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TranslationSourceTests(unittest.TestCase):
    def test_compendium_pack_indexes_keep_duplicate_names_separate(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "pf2e.feats-srd.json").write_text(
                json.dumps(
                    {"entries": {"Slither": {"name": "滑溜 Slither"}}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (directory / "pf2e.spells-srd.json").write_text(
                json.dumps(
                    {"entries": {"Slither": {"name": "蛇群术 Slither"}}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            packs = module.load_compendium_packs(directory)

        self.assertEqual(packs["feats-srd"]["Slither"]["name"], "滑溜 Slither")
        self.assertEqual(packs["spells-srd"]["Slither"]["name"], "蛇群术 Slither")

    def test_wiki_wins_and_core_sources_win_over_extra(self):
        module = load_script()
        sources = {
            "other": {"Sure Strike": {"name": "其他译名"}},
            "pf2e_compendium_extra": {"Sure Strike": {"name": "必中术 Sure Strike"}},
            "pf2_cn": {"Sure Strike": {"name": "系统译名"}},
            "pf2e_compendium": {
                "Sure Strike": {
                    "name": "克敌机先 Sure Strike",
                    "description": "核心典籍说明",
                }
            },
            "wiki": {"Sure Strike": {"name": "克敌机先 Sure Strike"}},
        }

        merged = module.merge_source_map(sources)

        self.assertEqual(merged["Sure Strike"]["name"], "克敌机先 Sure Strike")
        self.assertEqual(merged["Sure Strike"]["source"], "wiki")
        self.assertEqual(merged["Sure Strike"]["description"], "核心典籍说明")
        self.assertEqual(
            set(merged["Sure Strike"]["all_sources"]),
            set(sources),
        )

    def test_compendium_exact_entry_wins_same_tier_ui_heuristic(self):
        module = load_script()
        sources = {
            "pf2_cn": {"Critical Success": {"name": "大成功"}},
            "pf2e_compendium": {
                "Critical Success": {"name": "大成功 Critical Success"}
            },
        }

        merged = module.merge_source_map(sources)

        self.assertEqual(
            merged["Critical Success"]["name"],
            "大成功 Critical Success",
        )
        self.assertEqual(
            merged["Critical Success"]["source"],
            "pf2e_compendium",
        )

    def test_default_paths_follow_current_workspace_layout(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = module.default_paths(root)

        self.assertEqual(
            paths["pf2_cn"],
            root / "模组" / "pf2_cn" / "zh_Hans",
        )
        self.assertEqual(
            paths["pf2e_compendium"],
            root / "模组" / "pf2e_compendium_chn" / "compendium",
        )
        self.assertEqual(
            paths["output"],
            root / "工具" / "翻译流程" / "tm_cache" / "tm_3source.json",
        )


if __name__ == "__main__":
    unittest.main()
