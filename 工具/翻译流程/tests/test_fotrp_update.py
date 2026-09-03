import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "工具" / "翻译流程" / "scripts" / "fotrp_update.py"


def load_script(testcase):
    try:
        spec = importlib.util.spec_from_file_location("fotrp_update", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError as exc:
        testcase.fail(f"FotRP updater is missing: {exc}")


def actor(actor_id, name, item_id=None, item_name=None):
    result = {
        "_id": actor_id,
        "name": name,
        "prototypeToken": {"name": name},
        "system": {"details": {"publicNotes": ""}},
        "items": [],
    }
    if item_id:
        result["items"].append(
            {
                "_id": item_id,
                "name": item_name,
                "system": {"description": {"value": "<p>Rules text.</p>"}},
            }
        )
    return result


class FotRPUpdateTests(unittest.TestCase):
    def test_source_loader_hydrates_v14_filename_references(self):
        module = load_script(self)
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp)
            actor_file = source_dir / "Fisher_Actor_ACTOR00000000002.json"
            actor_file.write_text(
                json.dumps(actor("ACTOR00000000002", "Fisher")),
                encoding="utf-8",
            )
            adventure_file = source_dir / "Book_2_ADVENTURE000002.json"
            adventure_file.write_text(
                json.dumps(
                    {
                        "_id": "ADVENTURE000002",
                        "_key": "!adventures!ADVENTURE000002",
                        "name": "Book 2",
                        "actors": [actor_file.name],
                        "items": [],
                        "macros": [],
                        "tables": [],
                        "folders": [],
                    }
                ),
                encoding="utf-8",
            )

            loaded = module.load_adventures(source_dir)

        self.assertEqual(loaded[0]["actors"][0]["_id"], "ACTOR00000000002")
        self.assertEqual(loaded[0]["actors"][0]["name"], "Fisher")

    def test_migration_uses_stable_ids_across_renames(self):
        module = load_script(self)
        old_source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Fist of the Ruby Phoenix: Addons (Book 1)",
                "description": "<p>Addons</p>",
                "caption": "",
                "actors": [
                    actor(
                        "ACTOR00000000001",
                        "Spellcaster",
                        "ITEM000000000001",
                        "True Strike",
                    )
                ],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]
        new_source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 1",
                "description": "<p>Addons</p>",
                "caption": "",
                "actors": [
                    actor(
                        "ACTOR00000000001",
                        "Spellcaster",
                        "ITEM000000000001",
                        "Sure Strike",
                    ),
                    actor("ACTOR00000000002", "Fisher"),
                ],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]
        old_translation = {
            "label": "赤凰斗士：附加内容 FoTRP: Addons",
            "entries": {
                "Fist of the Ruby Phoenix: Addons (Book 1)": {
                    "name": "赤凰斗士：附加内容（第一本） Fist of the Ruby Phoenix: Addons (Book 1)",
                    "description": "<p>赤凰斗士附加内容</p>",
                    "actors": {
                        "Spellcaster": {
                            "name": "施法者 Spellcaster",
                            "items": {
                                "True Strike": {
                                    "name": "克敌机先 True Strike",
                                    "description": "<p>旧版中文规则。</p>",
                                }
                            },
                        }
                    },
                }
            },
        }

        migrated, report = module.migrate_adventure_pack(
            old_translation,
            old_source,
            new_source,
            name_overrides={"Book 1": "赤凰斗士：附加内容（第一本） Book 1"},
        )

        self.assertEqual(list(migrated["entries"]), ["Book 1"])
        entry = migrated["entries"]["Book 1"]
        self.assertEqual(entry["name"], "赤凰斗士：附加内容（第一本） Book 1")
        self.assertIn("Fisher", entry["actors"])
        self.assertIn("Sure Strike", entry["actors"]["Spellcaster"]["items"])
        self.assertEqual(
            entry["actors"]["Spellcaster"]["items"]["Sure Strike"]["name"],
            "克敌机先 Sure Strike",
        )
        self.assertEqual(report["added"]["Actor"], ["Fisher"])
        self.assertEqual(
            report["renamed"]["Item"],
            [{"id": "ITEM000000000001", "old": "True Strike", "new": "Sure Strike"}],
        )

    def test_new_actor_exposes_both_supported_token_mapping_keys(self):
        module = load_script(self)
        source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 2",
                "actors": [actor("ACTOR00000000002", "Fisher")],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]

        migrated, _ = module.migrate_adventure_pack(
            {"label": "test", "entries": {}},
            [],
            source,
        )

        translated = migrated["entries"]["Book 2"]["actors"]["Fisher"]
        self.assertEqual(translated["tokenName"], "Fisher")
        self.assertEqual(translated["prototypeToken"], "Fisher")

    def test_duplicate_old_adventure_key_is_resolved_by_id_and_removed(self):
        module = load_script(self)
        old_source = [
            {"_id": "AAAA00000000hmLe", "name": "Book 2", "actors": [], "items": [], "macros": [], "tables": [], "folders": []},
        ]
        new_source = [
            {"_id": "AAAA00000000hmLe", "name": "Book 3", "actors": [], "items": [], "macros": [], "tables": [], "folders": []},
        ]
        old_translation = {
            "label": "test",
            "entries": {
                "Book 2 (hmLe)": {"name": "第二本 Book 2"},
            },
        }

        migrated, _ = module.migrate_adventure_pack(
            old_translation,
            old_source,
            new_source,
            name_overrides={"Book 3": "第三本 Book 3"},
        )

        self.assertEqual(migrated["entries"], {"Book 3": {"name": "第三本 Book 3"}})

    def test_pack_level_mapping_and_folders_are_preserved(self):
        module = load_script(self)
        old_translation = {
            "label": "test",
            "mapping": {"actors": {"prototypeToken": "prototypeToken.name"}},
            "folders": {"Legacy": "旧目录 Legacy"},
            "entries": {"Book 1": {"name": "第一本 Book 1"}},
        }
        source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 1",
                "actors": [],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]

        migrated, _ = module.migrate_adventure_pack(old_translation, source, source)

        self.assertEqual(migrated["mapping"], old_translation["mapping"])
        self.assertEqual(migrated["folders"], old_translation["folders"])

    def test_i18n_validation_requires_identical_keys_and_placeholders(self):
        module = load_script(self)
        source = {
            "module": {
                "title": "Hex Content ({col}, {row})",
                "plain": "Close",
            }
        }
        good = {
            "module": {
                "title": "六角格内容（{col}, {row}）",
                "plain": "关闭",
            }
        }
        bad = {
            "module": {
                "title": "六角格内容（{col}）",
                "plain": "关闭",
            }
        }

        self.assertEqual(module.validate_i18n(source, good), [])
        self.assertEqual(
            module.validate_i18n(source, bad),
            ["module.title: placeholders ['col', 'row'] != ['col']"],
        )

    def test_folder_name_collection_collapses_same_name_documents(self):
        module = load_script(self)
        old_source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 1",
                "actors": [],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [
                    {"_id": "FOLDER0000000001", "name": "FotRP: Addons"},
                    {"_id": "FOLDER0000000002", "name": "FotRP: Addons"},
                ],
            }
        ]
        new_source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 1",
                "actors": [],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [
                    {"_id": "FOLDER0000000001", "name": "FotRP: Addons"},
                    {"_id": "FOLDER0000000002", "name": "FotRP: Addons"},
                    {"_id": "FOLDER0000000003", "name": "Level 17"},
                ],
            }
        ]
        old_translation = {
            "label": "test",
            "entries": {
                "Book 1": {
                    "name": "第一本 Book 1",
                    "folders": {"FotRP: Addons": "赤凰斗士：附加内容"},
                }
            },
        }

        migrated, report = module.migrate_adventure_pack(
            old_translation,
            old_source,
            new_source,
        )

        self.assertEqual(
            migrated["entries"]["Book 1"]["folders"],
            {
                "FotRP: Addons": "赤凰斗士：附加内容",
                "Level 17": "Level 17",
            },
        )
        self.assertEqual(report["added"]["Folder"], ["Level 17"])

        localized, _ = module.migrate_adventure_pack(
            old_translation,
            old_source,
            new_source,
            folder_name_overrides={"Level 17": "17级 Level 17"},
        )
        self.assertEqual(
            localized["entries"]["Book 1"]["folders"]["Level 17"],
            "17级 Level 17",
        )

    def test_exact_pack_memory_reuses_text_only_when_english_source_matches(self):
        module = load_script(self)
        first = actor("ACTOR00000000001", "Veteran", "ITEM000000000001", "Fist")
        second = actor("ACTOR00000000002", "Commoner", "ITEM000000000002", "Fist")
        third = actor("ACTOR00000000003", "Variant", "ITEM000000000003", "Fist")
        third["items"][0]["system"]["description"]["value"] = "<p>Different rules.</p>"
        source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 2",
                "actors": [first, second, third],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]
        translation = {
            "label": "test",
            "entries": {
                "Book 2": {
                    "name": "第二本 Book 2",
                    "actors": {
                        "Veteran": {
                            "name": "老兵 Veteran",
                            "items": {
                                "Fist": {
                                    "name": "拳击 Fist",
                                    "description": "<p>拳击规则。</p>",
                                }
                            },
                        },
                        "Commoner": {
                            "name": "Commoner",
                            "items": {
                                "Fist": {
                                    "name": "Fist",
                                    "description": "<p>Rules text.</p>",
                                }
                            },
                        },
                        "Variant": {
                            "name": "Variant",
                            "items": {
                                "Fist": {
                                    "name": "Fist",
                                    "description": "<p>Different rules.</p>",
                                }
                            },
                        },
                    },
                }
            },
        }

        memory = module.build_exact_pack_memory(translation, source)
        calibrated = module.apply_exact_pack_memory(translation, source, memory)

        commoner = calibrated["entries"]["Book 2"]["actors"]["Commoner"]["items"]["Fist"]
        variant = calibrated["entries"]["Book 2"]["actors"]["Variant"]["items"]["Fist"]
        self.assertEqual(commoner["name"], "拳击 Fist")
        self.assertEqual(commoner["description"], "<p>拳击规则。</p>")
        self.assertEqual(variant["name"], "拳击 Fist")
        self.assertEqual(variant["description"], "<p>Different rules.</p>")

    def test_core_term_memory_updates_source_backed_item_description_only(self):
        module = load_script(self)
        sourced = actor("ACTOR00000000001", "Mage", "ITEM000000000001", "Sure Strike")
        sourced["items"][0]["_stats"] = {
            "compendiumSource": "Compendium.pf2e.spells-srd.Item.abc"
        }
        custom = actor("ACTOR00000000002", "Custom Mage", "ITEM000000000002", "Sure Strike")
        source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 2",
                "actors": [sourced, custom],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]
        translation = {
            "label": "test",
            "entries": {
                "Book 2": {
                    "name": "第二本 Book 2",
                    "actors": {
                        "Mage": {"name": "Mage", "items": {"Sure Strike": {"name": "Sure Strike", "description": "old"}}},
                        "Custom Mage": {"name": "Custom Mage", "items": {"Sure Strike": {"name": "Sure Strike", "description": "custom"}}},
                    },
                }
            },
        }
        term_memory = {
            "Sure Strike": {
                "name": "克敌机先 Sure Strike",
                "description": "<p>核心中文说明。</p>",
                "source": "pf2e_compendium",
            }
        }

        calibrated = module.apply_term_memory(translation, source, term_memory)

        sourced_tr = calibrated["entries"]["Book 2"]["actors"]["Mage"]["items"]["Sure Strike"]
        custom_tr = calibrated["entries"]["Book 2"]["actors"]["Custom Mage"]["items"]["Sure Strike"]
        self.assertEqual(sourced_tr["name"], "克敌机先 Sure Strike")
        self.assertEqual(sourced_tr["description"], "<p>核心中文说明。</p>")
        self.assertEqual(custom_tr["name"], "克敌机先 Sure Strike")
        self.assertEqual(custom_tr["description"], "custom")

        reviewed = module.apply_term_memory(
            translation,
            source,
            term_memory,
            trusted_description_names={"Sure Strike"},
        )
        self.assertEqual(
            reviewed["entries"]["Book 2"]["actors"]["Custom Mage"]["items"]["Sure Strike"]["description"],
            "<p>核心中文说明。</p>",
        )

        runtime_memory = {
            "Sure Strike": {
                "name": "克敌机先 Sure Strike",
                "description": "<p>@Localize[PF2E.NPC.Abilities.Glossary.Grab]</p>",
                "source": "pf2e_compendium",
            }
        }
        runtime_localized = module.apply_term_memory(
            translation,
            source,
            runtime_memory,
            trusted_description_names={"Sure Strike"},
        )
        self.assertEqual(
            runtime_localized["entries"]["Book 2"]["actors"]["Custom Mage"]["items"]["Sure Strike"]["description"],
            "<p>@Localize[PF2E.NPC.Abilities.Glossary.Grab]</p>",
        )

    def test_term_memory_preserves_bilingual_document_name_style(self):
        module = load_script(self)
        source_actor = actor("ACTOR00000000001", "怪物", "ITEM000000000001", "Claw")
        source_actor["prototypeToken"]["name"] = "怪物"
        source = [{"_id": "ADVENTURE0000001", "name": "第二本", "actors": [source_actor], "items": [], "macros": [], "tables": [], "folders": []}]
        translation = {
            "entries": {
                "第二本": {
                    "name": "第二本",
                    "actors": {
                        "怪物": {
                            "name": "怪物",
                            "tokenName": "怪物",
                            "prototypeToken": "怪物",
                            "items": {"Claw": {"name": "Claw", "description": "<p>规则。</p>"}},
                        }
                    },
                }
            }
        }

        calibrated = module.apply_term_memory(
            translation,
            source,
            {"Claw": {"name": "爪击", "description": None, "source": "pf2_cn"}},
        )

        self.assertEqual(
            calibrated["entries"]["第二本"]["actors"]["怪物"]["items"]["Claw"]["name"],
            "爪击 Claw",
        )

    def test_compendium_source_pack_disambiguates_same_named_items(self):
        module = load_script(self)
        source_actor = actor("ACTOR00000000001", "施法者", "ITEM000000000001", "Slither")
        source_actor["prototypeToken"]["name"] = "施法者"
        source_actor["items"][0]["_stats"] = {
            "compendiumSource": "Compendium.pf2e.spells-srd.Item.spell-id"
        }
        source = [{"_id": "ADVENTURE0000001", "name": "第二本", "actors": [source_actor], "items": [], "macros": [], "tables": [], "folders": []}]
        translation = {
            "entries": {
                "第二本": {
                    "name": "第二本",
                    "actors": {
                        "施法者": {
                            "name": "施法者",
                            "tokenName": "施法者",
                            "prototypeToken": "施法者",
                            "items": {"Slither": {"name": "Slither", "description": "old"}},
                        }
                    },
                }
            }
        }
        global_memory = {
            "Slither": {
                "name": "滑溜 Slither",
                "description": "<p>错误的专长说明。</p>",
                "source": "pf2e_compendium",
            }
        }
        pack_memory = {
            "spells-srd": {
                "Slither": {
                    "name": "蛇群术 Slither",
                    "description": "<p>正确的法术说明。</p>",
                }
            }
        }

        calibrated = module.apply_term_memory(
            translation,
            source,
            global_memory,
            compendium_pack_memory=pack_memory,
        )
        item = calibrated["entries"]["第二本"]["actors"]["施法者"]["items"]["Slither"]
        self.assertEqual(item["name"], "蛇群术 Slither")
        self.assertEqual(item["description"], "<p>正确的法术说明。</p>")

    def test_migration_reports_changed_translatable_source_fields(self):
        module = load_script(self)
        old_actor = actor("ACTOR00000000001", "Mage", "ITEM000000000001", "Custom Power")
        new_actor = actor("ACTOR00000000001", "Mage", "ITEM000000000001", "Custom Power")
        old_actor["items"][0]["system"]["description"]["value"] = "<p>Old rule.</p>"
        new_actor["items"][0]["system"]["description"]["value"] = "<p>New rule.</p>"
        old_source = [{"_id": "ADVENTURE0000001", "name": "Book 2", "actors": [old_actor], "items": [], "macros": [], "tables": [], "folders": []}]
        new_source = [{"_id": "ADVENTURE0000001", "name": "Book 2", "actors": [new_actor], "items": [], "macros": [], "tables": [], "folders": []}]
        translation = {
            "label": "test",
            "entries": {
                "Book 2": {
                    "name": "第二本 Book 2",
                    "actors": {
                        "Mage": {
                            "name": "法师 Mage",
                            "items": {
                                "Custom Power": {
                                    "name": "自定义能力 Custom Power",
                                    "description": "<p>旧规则。</p>",
                                }
                            },
                        }
                    },
                }
            },
        }

        _, report = module.migrate_adventure_pack(translation, old_source, new_source)

        self.assertEqual(
            report["changed"]["Item"],
            [
                {
                    "id": "ITEM000000000001",
                    "name": "Custom Power",
                    "field": "description",
                    "old": "<p>Old rule.</p>",
                    "new": "<p>New rule.</p>",
                }
            ],
        )

    def test_untranslated_audit_reports_document_id_field_and_source_text(self):
        module = load_script(self)
        source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 2",
                "actors": [actor("ACTOR00000000002", "Fisher")],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]
        translation = {
            "entries": {
                "Book 2": {
                    "name": "赤凰斗士：附加内容（第二本） Book 2",
                    "actors": {
                        "Fisher": {
                            "name": "Fisher",
                            "tokenName": "渔民 Fisher",
                            "prototypeToken": "渔民 Fisher",
                        }
                    },
                }
            }
        }

        self.assertEqual(
            module.audit_untranslated(translation, source),
            [
                {
                    "type": "Actor",
                    "id": "ACTOR00000000002",
                    "name": "Fisher",
                    "field": "name",
                    "source": "Fisher",
                    "translation": "Fisher",
                }
            ],
        )

    def test_untranslated_audit_ignores_runtime_localize_only_descriptions(self):
        module = load_script(self)
        localized = actor("ACTOR00000000001", "本地化演员")
        localized["prototypeToken"]["name"] = "本地化演员"
        localized["items"] = [
            {
                "_id": "ITEM000000000001",
                "name": "擒抱 Grab",
                "system": {
                    "description": {
                        "value": "<p>@Localize[PF2E.NPC.Abilities.Glossary.Grab]</p>"
                    }
                },
            }
        ]
        source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "第二本 Book 2",
                "actors": [localized],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]
        translation = {
            "entries": {
                "第二本 Book 2": {
                    "name": "第二本 Book 2",
                    "actors": {
                        "本地化演员": {
                            "name": "本地化演员",
                            "tokenName": "本地化演员",
                            "prototypeToken": "本地化演员",
                            "items": {
                                "擒抱 Grab": {
                                    "name": "擒抱 Grab",
                                    "description": "<p>@Localize[PF2E.NPC.Abilities.Glossary.Grab]</p>",
                                }
                            },
                        }
                    },
                }
            }
        }

        self.assertEqual(module.audit_untranslated(translation, source), [])

    def test_untranslated_audit_accepts_runtime_localize_replacement_for_full_source(self):
        module = load_script(self)
        source_actor = actor("ACTOR00000000001", "本地化演员")
        source_actor["prototypeToken"]["name"] = "本地化演员"
        source_actor["items"] = [
            {
                "_id": "ITEM000000000001",
                "name": "擒获 Grab",
                "system": {"description": {"value": "<p>Full English rules text.</p>"}},
            }
        ]
        source = [{"_id": "ADVENTURE0000001", "name": "第二本", "actors": [source_actor], "items": [], "macros": [], "tables": [], "folders": []}]
        translation = {
            "entries": {
                "第二本": {
                    "name": "第二本",
                    "actors": {
                        "本地化演员": {
                            "name": "本地化演员",
                            "tokenName": "本地化演员",
                            "prototypeToken": "本地化演员",
                            "items": {
                                "擒获 Grab": {
                                    "name": "擒获 Grab",
                                    "description": "<p>@Localize[PF2E.NPC.Abilities.Glossary.Grab]</p>",
                                }
                            },
                        }
                    },
                }
            }
        }

        self.assertEqual(module.audit_untranslated(translation, source), [])

    def test_manual_overrides_are_stable_id_based_and_reject_unused_ids(self):
        module = load_script(self)
        source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 2",
                "actors": [actor("ACTOR00000000002", "Fisher")],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]
        translation = {
            "entries": {
                "Book 2": {
                    "name": "第二本 Book 2",
                    "actors": {
                        "Fisher": {
                            "name": "Fisher",
                            "tokenName": "Fisher",
                            "prototypeToken": "Fisher",
                        }
                    },
                }
            }
        }
        overrides = {
            "Actor": {
                "ACTOR00000000002": {
                    "name": "渔民 Fisher",
                    "tokenName": "渔民 Fisher",
                    "prototypeToken": "渔民 Fisher",
                }
            }
        }

        updated = module.apply_document_overrides(translation, source, overrides)

        self.assertEqual(
            updated["entries"]["Book 2"]["actors"]["Fisher"]["name"],
            "渔民 Fisher",
        )
        with self.assertRaisesRegex(ValueError, "UNKNOWN000000000"):
            module.apply_document_overrides(
                translation,
                source,
                {"Actor": {"UNKNOWN000000000": {"name": "未知"}}},
            )

    def test_update_pack_runs_migration_memory_and_reviewed_overrides_in_order(self):
        module = load_script(self)
        source = [
            {
                "_id": "ADVENTURE0000001",
                "name": "Book 2",
                "actors": [actor("ACTOR00000000002", "Fisher")],
                "items": [],
                "macros": [],
                "tables": [],
                "folders": [],
            }
        ]
        config = {
            "adventure_names": {"Book 2": "第二本 Book 2"},
            "document_overrides": {
                "Actor": {
                    "ACTOR00000000002": {
                        "name": "渔民 Fisher",
                        "tokenName": "渔民 Fisher",
                        "prototypeToken": "渔民",
                    }
                }
            },
        }

        updated, report = module.update_pack(
            {"label": "test", "entries": {}},
            [],
            source,
            {},
            config,
        )

        self.assertEqual(updated["entries"]["Book 2"]["name"], "第二本 Book 2")
        self.assertEqual(report["untranslated"], [])


if __name__ == "__main__":
    unittest.main()
