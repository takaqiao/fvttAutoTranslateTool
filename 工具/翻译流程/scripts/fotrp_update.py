"""Stable-ID migration and validation helpers for the FotRP Addon translation."""

from __future__ import annotations

import copy
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path


COLLECTIONS = (
    ("actors", "actors", "Actor"),
    ("items", "items", "Item"),
    ("macros", "macros", "Macro"),
    ("tables", "tables", "RollTable"),
    ("journal", "journals", "JournalEntry"),
    ("scenes", "scenes", "Scene"),
)


def load_adventures(source_dir):
    """Load Adventure source JSON and hydrate Foundry v14 filename references."""
    source_dir = Path(source_dir)
    cache = {}

    def read(path):
        path = Path(path)
        if path not in cache:
            cache[path] = json.loads(path.read_text(encoding="utf-8"))
        return copy.deepcopy(cache[path])

    adventures = []
    for path in sorted(source_dir.glob("*.json")):
        document = read(path)
        if not str(document.get("_key", "")).startswith("!adventures!"):
            continue
        for source_field, _, _ in COLLECTIONS:
            values = document.get(source_field, [])
            document[source_field] = [
                read(source_dir / value) if isinstance(value, str) else value
                for value in values
            ]
        folders = document.get("folders", [])
        document["folders"] = [
            read(source_dir / value) if isinstance(value, str) else value
            for value in folders
        ]
        adventures.append(document)
    return adventures


def _document_keys(documents):
    """Mirror Babele export keys: first name is plain, duplicates use id suffix."""
    seen = Counter()
    result = {}
    for doc in documents or []:
        name = doc.get("name") or doc.get("_id")
        seen[name] += 1
        key = name if seen[name] == 1 else f"{name} ({doc.get('_id', '')[-4:]})"
        result[doc.get("_id")] = key
    return result


def _refresh_bilingual(value, old_english, new_english):
    if not isinstance(value, str):
        return value
    if value == old_english:
        return new_english
    if old_english and value.endswith(old_english):
        return value[: -len(old_english)] + new_english
    return value


def _string_at(document, *path):
    value = document
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value if isinstance(value, str) and value.strip() else None


def _new_translation(document, document_type):
    result = {}
    name = document.get("name")
    if isinstance(name, str) and name.strip():
        result["name"] = name

    if document_type == "Actor":
        token_name = _string_at(document, "prototypeToken", "name")
        if token_name:
            result["tokenName"] = token_name
            result["prototypeToken"] = token_name
        public_notes = _string_at(document, "system", "details", "publicNotes")
        private_notes = _string_at(document, "system", "details", "privateNotes")
        if public_notes:
            result["publicNotes"] = public_notes
        if private_notes:
            result["privateNotes"] = private_notes
    elif document_type == "Item":
        description = _string_at(document, "system", "description", "value")
        if description:
            result["description"] = description
    elif document_type in {"RollTable", "JournalEntry", "Scene"}:
        description = document.get("description")
        if isinstance(description, str) and description.strip():
            result["description"] = description
    return result


def _record(report, bucket, document_type, value):
    report[bucket][document_type].append(value)


def _record_field_changes(old_document, new_document, document_type, report):
    old_fields = _translatable_fields(old_document, document_type)
    new_fields = _translatable_fields(new_document, document_type)
    for field in sorted((set(old_fields) | set(new_fields)) - {"name"}):
        if old_fields.get(field) == new_fields.get(field):
            continue
        _record(
            report,
            "changed",
            document_type,
            {
                "id": new_document.get("_id"),
                "name": new_document.get("name"),
                "field": field,
                "old": old_fields.get(field),
                "new": new_fields.get(field),
            },
        )


def _migrate_collection(
    old_documents,
    new_documents,
    old_translations,
    document_type,
    report,
):
    old_documents = old_documents or []
    new_documents = new_documents or []
    old_translations = old_translations or {}
    old_keys = _document_keys(old_documents)
    new_keys = _document_keys(new_documents)
    old_by_id = {doc.get("_id"): doc for doc in old_documents}
    translated_by_id = {
        doc_id: old_translations[key]
        for doc_id, key in old_keys.items()
        if key in old_translations
    }
    result = {}

    for new_doc in new_documents:
        doc_id = new_doc.get("_id")
        new_name = new_doc.get("name") or doc_id
        old_doc = old_by_id.get(doc_id)
        translated = copy.deepcopy(translated_by_id.get(doc_id))

        if translated is None:
            translated = _new_translation(new_doc, document_type)
            _record(report, "added", document_type, new_name)
        elif old_doc:
            _record_field_changes(old_doc, new_doc, document_type, report)
            if old_doc.get("name") != new_doc.get("name"):
                translated["name"] = _refresh_bilingual(
                    translated.get("name", old_doc.get("name")),
                    old_doc.get("name"),
                    new_doc.get("name"),
                )
                _record(
                    report,
                    "renamed",
                    document_type,
                    {"id": doc_id, "old": old_doc.get("name"), "new": new_doc.get("name")},
                )

        if document_type == "Actor":
            nested = _migrate_collection(
                old_doc.get("items", []) if old_doc else [],
                new_doc.get("items", []),
                translated.get("items", {}),
                "Item",
                report,
            )
            if nested:
                translated["items"] = nested
            else:
                translated.pop("items", None)

        result[new_keys[doc_id]] = translated
    return result


def _migrate_folders(
    old_folders,
    new_folders,
    old_translations,
    report,
    name_overrides=None,
):
    name_overrides = name_overrides or {}
    old_folders = old_folders or []
    new_folders = new_folders or []
    old_translations = old_translations or {}
    old_by_id = {doc.get("_id"): doc for doc in old_folders}
    result = {}
    for folder in new_folders:
        folder_id = folder.get("_id")
        folder_name = folder.get("name")
        if folder_name in result:
            continue
        old_folder = old_by_id.get(folder_id)
        old_name = old_folder.get("name") if old_folder else None
        if folder_name in old_translations:
            value = old_translations[folder_name]
        elif old_name in old_translations:
            value = _refresh_bilingual(
                old_translations[old_name],
                old_name,
                folder_name,
            )
        else:
            value = folder_name
            _record(report, "added", "Folder", folder_name)
        result[folder_name] = name_overrides.get(folder_name, value)
    return result


def migrate_adventure_pack(
    old_translation,
    old_adventures,
    new_adventures,
    name_overrides=None,
    folder_name_overrides=None,
):
    """Migrate a Babele Adventure pack by stable Foundry document IDs."""
    name_overrides = name_overrides or {}
    report = {
        "added": defaultdict(list),
        "renamed": defaultdict(list),
        "changed": defaultdict(list),
    }
    old_keys = _document_keys(old_adventures)
    old_by_id = {doc.get("_id"): doc for doc in old_adventures}
    old_entries = old_translation.get("entries", {})
    output = {
        key: copy.deepcopy(value)
        for key, value in old_translation.items()
        if key != "entries"
    }
    output["entries"] = {}

    for new_adventure in new_adventures:
        adventure_id = new_adventure.get("_id")
        old_adventure = old_by_id.get(adventure_id)
        old_entry = {}
        if old_adventure:
            old_entry = copy.deepcopy(old_entries.get(old_keys[adventure_id], {}))
            _record_field_changes(old_adventure, new_adventure, "Adventure", report)

        entry = {}
        name = new_adventure.get("name")
        old_name = old_adventure.get("name") if old_adventure else None
        translated_name = old_entry.get("name", name)
        if old_name:
            translated_name = _refresh_bilingual(translated_name, old_name, name)
        entry["name"] = name_overrides.get(name, translated_name)

        for field in ("description", "caption"):
            source_value = new_adventure.get(field)
            if isinstance(source_value, str) and source_value.strip():
                entry[field] = old_entry.get(field, source_value)

        for source_field, translation_field, document_type in COLLECTIONS:
            migrated = _migrate_collection(
                old_adventure.get(source_field, []) if old_adventure else [],
                new_adventure.get(source_field, []),
                old_entry.get(translation_field, {}),
                document_type,
                report,
            )
            if migrated:
                entry[translation_field] = migrated

        folders = _migrate_folders(
            old_adventure.get("folders", []) if old_adventure else [],
            new_adventure.get("folders", []),
            old_entry.get("folders", {}),
            report,
            folder_name_overrides,
        )
        if folders:
            entry["folders"] = folders
        output["entries"][name] = entry

    normalized_report = {
        bucket: {key: value for key, value in groups.items()}
        for bucket, groups in report.items()
    }
    return output, normalized_report


def _flatten_strings(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_strings(child, path))
    elif isinstance(value, str):
        result[prefix] = value
    return result


def _has_cjk(value):
    return isinstance(value, str) and any("\u3400" <= char <= "\u9fff" for char in value)


def _has_literal_english(value):
    if not isinstance(value, str):
        return False
    visible = re.sub(r"@Localize\[[^\]]+\]", "", value)
    visible = re.sub(r"<[^>]+>", "", visible)
    return bool(re.search(r"[A-Za-z]", visible))


def _iter_document_pairs(translation, adventures):
    entries = translation.get("entries", {})
    for adventure in adventures:
        translated_adventure = entries.get(adventure.get("name"))
        if not isinstance(translated_adventure, dict):
            continue
        yield adventure, translated_adventure, "Adventure"
        for source_field, translation_field, document_type in COLLECTIONS:
            documents = adventure.get(source_field, []) or []
            translated_documents = translated_adventure.get(translation_field, {}) or {}
            keys = _document_keys(documents)
            for document in documents:
                translated_document = translated_documents.get(keys.get(document.get("_id")))
                if not isinstance(translated_document, dict):
                    continue
                yield document, translated_document, document_type
                if document_type == "Actor":
                    items = document.get("items", []) or []
                    translated_items = translated_document.get("items", {}) or {}
                    item_keys = _document_keys(items)
                    for item in items:
                        translated_item = translated_items.get(item_keys.get(item.get("_id")))
                        if isinstance(translated_item, dict):
                            yield item, translated_item, "Item"


def _translatable_fields(document, document_type):
    fields = {"name": document.get("name")}
    if document_type == "Adventure":
        fields.update(
            {
                "description": document.get("description"),
                "caption": document.get("caption"),
            }
        )
    elif document_type == "Actor":
        token_name = _string_at(document, "prototypeToken", "name")
        fields.update(
            {
                "tokenName": token_name,
                "prototypeToken": token_name,
                "publicNotes": _string_at(document, "system", "details", "publicNotes"),
                "privateNotes": _string_at(document, "system", "details", "privateNotes"),
                "blurb": _string_at(document, "system", "details", "blurb"),
            }
        )
    elif document_type == "Item":
        fields["description"] = _string_at(document, "system", "description", "value")
    elif document_type in {"RollTable", "JournalEntry", "Scene"}:
        fields["description"] = document.get("description")
    return {
        key: value
        for key, value in fields.items()
        if isinstance(value, str) and value.strip()
    }


def build_exact_pack_memory(translation, adventures):
    """Build exact-English-value memory from already translated pack leaves."""
    candidates = defaultdict(lambda: defaultdict(Counter))
    for document, translated, document_type in _iter_document_pairs(translation, adventures):
        for field, english in _translatable_fields(document, document_type).items():
            chinese = translated.get(field)
            if _has_cjk(chinese) and chinese != english:
                candidates[field][english][chinese] += 1
    return {
        field: {
            english: counts.most_common(1)[0][0]
            for english, counts in values.items()
        }
        for field, values in candidates.items()
    }


def apply_exact_pack_memory(translation, adventures, memory):
    """Fill untranslated leaves only when their exact English source is known."""
    output = copy.deepcopy(translation)
    for document, translated, document_type in _iter_document_pairs(output, adventures):
        for field, english in _translatable_fields(document, document_type).items():
            current = translated.get(field)
            replacement = memory.get(field, {}).get(english)
            if replacement and not _has_cjk(current):
                translated[field] = replacement
    return output


def apply_term_memory(
    translation,
    adventures,
    term_memory,
    trusted_description_names=None,
    compendium_pack_memory=None,
):
    """Apply higher-authority name translations and safe source-backed descriptions."""
    trusted_description_names = set(trusted_description_names or ())
    compendium_pack_memory = compendium_pack_memory or {}
    output = copy.deepcopy(translation)
    for document, translated, document_type in _iter_document_pairs(output, adventures):
        term = term_memory.get(document.get("name"), {})
        pack_term = {}
        compendium_source = None
        if document_type == "Item":
            compendium_source = _string_at(document, "_stats", "compendiumSource")
            match = re.match(r"^Compendium\.pf2e\.([^.]+)\.", compendium_source or "")
            if match:
                pack_term = compendium_pack_memory.get(match.group(1), {}).get(
                    document.get("name"), {}
                )

        # A reviewed Wiki name remains authoritative. Otherwise an exact Foundry
        # pack match beats the lossy flat core index when duplicate English names
        # exist in different packs (for example the Slither feat and spell).
        name_term = term
        if term.get("source") != "wiki" and _has_cjk(pack_term.get("name")):
            name_term = pack_term
        translated_name = name_term.get("name")
        if _has_cjk(translated_name):
            source_name = document.get("name")
            if (
                _has_literal_english(source_name)
                and source_name not in translated_name
            ):
                translated_name = f"{translated_name} {source_name}"
            translated["name"] = translated_name
        if document_type == "Item":
            pack_description = pack_term.get("description")
            translated_description = (
                pack_description
                if _has_cjk(pack_description) or "@Localize[" in str(pack_description)
                else term.get("description")
            )
            trusted_name = document.get("name") in trusted_description_names
            localized_description = _has_cjk(translated_description) or (
                "@Localize[" in str(translated_description)
            )
            if (compendium_source or trusted_name) and localized_description:
                translated["description"] = translated_description
    return output


def apply_document_overrides(translation, adventures, overrides):
    """Apply reviewed translation overrides by Foundry document type and ID."""
    output = copy.deepcopy(translation)
    translated_by_identity = {
        (document_type, document.get("_id")): translated
        for document, translated, document_type in _iter_document_pairs(output, adventures)
    }
    unused = []
    for document_type, documents in overrides.items():
        for document_id, fields in documents.items():
            translated = translated_by_identity.get((document_type, document_id))
            if translated is None:
                unused.append(f"{document_type}:{document_id}")
                continue
            translated.update(copy.deepcopy(fields))
    if unused:
        raise ValueError("unused document override IDs: " + ", ".join(sorted(unused)))
    return output


def audit_untranslated(translation, adventures):
    """List translatable document leaves that still contain no Chinese text."""
    issues = []
    for document, translated, document_type in _iter_document_pairs(translation, adventures):
        for field, source in _translatable_fields(document, document_type).items():
            current = translated.get(field)
            if _has_cjk(current):
                continue
            if field == "description" and "@Localize[" in str(current):
                continue
            if field == "description" and not _has_literal_english(source):
                continue
            issues.append(
                {
                    "type": document_type,
                    "id": document.get("_id"),
                    "name": document.get("name"),
                    "field": field,
                    "source": source,
                    "translation": current,
                }
            )
    return issues


def update_pack(
    old_translation,
    old_adventures,
    new_adventures,
    term_memory,
    config,
    compendium_pack_memory=None,
):
    """Run the reviewed FotRP migration and calibration pipeline."""
    output, report = migrate_adventure_pack(
        old_translation,
        old_adventures,
        new_adventures,
        name_overrides=config.get("adventure_names"),
        folder_name_overrides=config.get("folder_names"),
    )
    exact_memory = build_exact_pack_memory(old_translation, old_adventures)
    output = apply_exact_pack_memory(output, new_adventures, exact_memory)
    output = apply_term_memory(
        output,
        new_adventures,
        term_memory,
        trusted_description_names=config.get("trusted_description_names"),
        compendium_pack_memory=compendium_pack_memory,
    )
    output = apply_document_overrides(
        output,
        new_adventures,
        config.get("document_overrides", {}),
    )
    report["untranslated"] = audit_untranslated(output, new_adventures)
    return output, report


def validate_i18n(source, translation):
    """Return key/placeholder incompatibilities for an external i18n file."""
    source_flat = _flatten_strings(source)
    translated_flat = _flatten_strings(translation)
    errors = []
    missing = sorted(set(source_flat) - set(translated_flat))
    extra = sorted(set(translated_flat) - set(source_flat))
    errors.extend(f"missing key: {path}" for path in missing)
    errors.extend(f"extra key: {path}" for path in extra)
    placeholder = re.compile(r"\{([^{}]+)\}")
    for path in sorted(set(source_flat) & set(translated_flat)):
        expected = sorted(placeholder.findall(source_flat[path]))
        actual = sorted(placeholder.findall(translated_flat[path]))
        if expected != actual:
            errors.append(f"{path}: placeholders {expected} != {actual}")
    return errors


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Migrate the FotRP Addon Babele pack to a new upstream release."
    )
    parser.add_argument("--old-translation", required=True, type=Path)
    parser.add_argument("--old-source", required=True, type=Path)
    parser.add_argument("--new-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "工具" / "翻译流程" / "data" / "fotrp_update_config.json",
    )
    parser.add_argument(
        "--wiki-terms",
        type=Path,
        default=root / "工具" / "翻译流程" / "data" / "fotrp_wiki_terms.json",
    )
    parser.add_argument(
        "--compendium-dir",
        type=Path,
        default=root / "模组" / "pf2e_compendium_chn" / "compendium",
    )
    parser.add_argument(
        "--pf2-cn-dir",
        type=Path,
        default=root / "模组" / "pf2_cn" / "zh_Hans",
    )
    args = parser.parse_args(argv)

    from build_3source_tm import (
        load_compendium,
        load_compendium_packs,
        load_pf2_cn,
        load_wiki,
        merge_source_map,
    )

    old_translation = json.loads(args.old_translation.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    term_memory = merge_source_map(
        {
            "pf2e_compendium": load_compendium(args.compendium_dir),
            "pf2_cn": load_pf2_cn(args.pf2_cn_dir),
            "wiki": load_wiki(args.wiki_terms),
        }
    )
    output, report = update_pack(
        old_translation,
        load_adventures(args.old_source),
        load_adventures(args.new_source),
        term_memory,
        config,
        compendium_pack_memory=load_compendium_packs(args.compendium_dir),
    )
    _write_json(args.output, output)
    if args.report:
        _write_json(args.report, report)
    if report["untranslated"]:
        raise SystemExit(
            f"{len(report['untranslated'])} untranslated fields remain; see the report"
        )


if __name__ == "__main__":
    main()
