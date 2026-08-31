# -*- coding: utf-8 -*-
"""
PF2e Foundry @UUID enricher: add or upgrade {label} on @UUID references.

- No-label @UUID  -> add {bilingual} (or CN-only per pack style)
- EN-only label  -> upgrade to bilingual (or CN-only per pack style)
- Existing CN/bilingual -> leave alone
- pf2cn miss -> skip (preserve existing)
"""
import sys, json, re, os
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'C:/Users/Taka/Desktop/fvtt'
LOOKUP_PATH = os.path.join(ROOT, 'FotRP', '需要翻译', '_uuid_unified_lookup.json')

L = json.load(open(LOOKUP_PATH, encoding='utf-8'))
pack_en_to_cn = L['pack_en_to_cn']
all_en_to_cn = L['all_en_to_cn']
id_to_pack_en = L['id_to_pack_en']  # iid -> [pack, en]
id_to_cn_corpus = L['id_to_cn_corpus']

# Extended corpus from broader project scan (Actor, JournalEntryPage, etc.)
EXT_PATH = os.path.join(ROOT, 'FotRP', '需要翻译', '_uuid_extended_corpus.json')
ID_EXT_BEST = {}
ID_EXT_EN = {}
if os.path.exists(EXT_PATH):
    ext = json.load(open(EXT_PATH, encoding='utf-8'))
    ID_EXT_BEST = ext.get('id_best', {})  # iid -> {label, pack}
    ID_EXT_EN = ext.get('id_en_best', {})  # iid -> en

# Bestiary entry-name lookup
ENTRY_PATH = os.path.join(ROOT, 'FotRP', '需要翻译', '_uuid_entry_name_lookup.json')
ENTRY_LOOKUP = {}
if os.path.exists(ENTRY_PATH):
    ENTRY_LOOKUP = json.load(open(ENTRY_PATH, encoding='utf-8'))

# Per-pack style policy (cn-only vs bilingual)
STYLE = {
    'conditionitems': 'bi',
    'conditions': 'bi',
    'actionspf2e': 'bi',
    'actions': 'bi',
    'spells-srd': 'bi',
    'spells': 'bi',
    'equipment-srd': 'bi',
    'equipment': 'cn',
    'feats': 'bi',
    'feats-srd': 'bi',
    'spell-effects': 'cn',
    'classfeatures': 'bi',
    'feat-effects': 'cn',
    'equipment-effects': 'cn',
    'other-effects': 'cn',
    'bestiary-effects': 'cn',
    'ancestryfeatures': 'bi',
    'heritages': 'bi',
    'journals': 'bi',
    'campaign-effects': 'cn',
    'familiar-abilities': 'bi',
    'backgrounds': 'bi',
    'deities': 'bi',
    'bestiary-ability-glossary-srd': 'bi',
    'bestiary-family-ability-glossary': 'bi',
    'adventure-specific-actions': 'bi',
}

PACK_ALIASES = {
    'conditions': 'conditionitems',
    'actions': 'actionspf2e',
    'feats': 'feats-srd',
    'spells': 'spells-srd',
    'equipment': 'equipment-srd',
}


def lookup_cn_bi(pack, en_name):
    """Look up CN bilingual label 'CN EN' or 'CN'. Returns string or None."""
    if pack and pack in pack_en_to_cn and en_name in pack_en_to_cn[pack]:
        return pack_en_to_cn[pack][en_name]
    alias = PACK_ALIASES.get(pack)
    if alias and alias in pack_en_to_cn and en_name in pack_en_to_cn[alias]:
        return pack_en_to_cn[alias][en_name]
    if en_name in all_en_to_cn:
        return all_en_to_cn[en_name]
    if en_name.lower() in all_en_to_cn:
        return all_en_to_cn[en_name.lower()]
    return None


def split_cn_en(bi):
    """Split 'CN EN' -> ('CN', 'EN'); pure CN -> (CN, ''); pure EN -> ('', EN)."""
    m = re.match(r'^([一-鿿，。：；！？、（）·\d\s]+?)\s+([A-Za-z].*)$', bi)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if not re.search(r'[A-Za-z]', bi):
        return bi, ''
    if not re.search(r'[一-鿿]', bi):
        return '', bi
    return bi, ''


def build_label(pack, en_name, style):
    """Build label per style. Returns (label, source) or (None, reason)."""
    bi = lookup_cn_bi(pack, en_name)
    base_used = False
    base = None
    num = None
    if not bi:
        # Try base (strip trailing number for numbered conditions)
        m_num = re.match(r'^(.+?)\s+(\d+)$', en_name)
        if m_num:
            base = m_num.group(1)
            num = m_num.group(2)
            bi = lookup_cn_bi(pack, base)
            if bi:
                base_used = True
    if not bi:
        return None, 'no_pf2cn'
    cn, en_part = split_cn_en(bi)
    if not cn:
        return None, 'no_cn_part'
    if style == 'cn':
        if base_used:
            return f'{cn} {num}', 'pf2cn'
        return cn, 'pf2cn'
    # bilingual style
    if base_used:
        return f'{cn} {en_name}', 'pf2cn'
    if en_part:
        return f'{cn} {en_part}', 'pf2cn'
    return f'{cn} {en_name}', 'pf2cn'


def build_label_for_id(iid, pack, existing_label):
    """For no-label refs: derive from id + pack via id_to_pack_en, then build label."""
    style = STYLE.get(pack, 'bi')

    if existing_label:
        # EN-only upgrade: existing_label IS the EN name
        label, src = build_label(pack, existing_label, style)
        if label:
            return label, src
        # Try entry-name lookup for the EN-only label
        if existing_label in ENTRY_LOOKUP:
            bi = ENTRY_LOOKUP[existing_label]
            cn_part, en_part = split_cn_en(bi)
            if cn_part:
                if style == 'cn':
                    return cn_part, 'bestiary_entry'
                if en_part:
                    return f'{cn_part} {en_part}', 'bestiary_entry'
                return f'{cn_part} {existing_label}', 'bestiary_entry'
        return None, src

    # No-label: try id -> EN -> pf2cn
    pe = id_to_pack_en.get(iid)
    en_name = pe[1] if pe else (ID_EXT_EN.get(iid) if iid in ID_EXT_EN else None)
    if pe:
        label, src = build_label(pack, pe[1], style)
        if label:
            return label, src
    elif en_name:
        # Try with the EN from extended corpus
        label, src = build_label(pack, en_name, style)
        if label:
            return label, src

    # Fallback: corpus CN (original)
    cn = id_to_cn_corpus.get(iid)
    if cn and not re.search(r'[A-Za-z]', cn):
        if style == 'cn':
            return cn, 'corpus'
        # bilingual but no EN known
        if en_name:
            return f'{cn} {en_name}', 'corpus_built_bi'
        return cn, 'corpus_cn_only'

    # Fallback: extended corpus
    if iid in ID_EXT_BEST:
        cn_lbl = ID_EXT_BEST[iid]['label']
        has_cn_in = bool(re.search(r'[一-鿿]', cn_lbl))
        has_en_in = bool(re.search(r'[A-Za-z]', cn_lbl))
        if has_cn_in and has_en_in:
            return cn_lbl, 'ext_corpus_bi'
        if has_cn_in:
            if style == 'cn':
                return cn_lbl, 'ext_corpus_cn'
            en_alt = ID_EXT_EN.get(iid) or en_name
            if en_alt:
                return f'{cn_lbl} {en_alt}', 'ext_corpus_built_bi'
            return cn_lbl, 'ext_corpus_cn_only'

    # Fallback: bestiary entry-name lookup
    if en_name and en_name in ENTRY_LOOKUP:
        bi = ENTRY_LOOKUP[en_name]
        cn_part, en_part = split_cn_en(bi)
        if cn_part:
            if style == 'cn':
                return cn_part, 'bestiary_entry'
            if en_part:
                return f'{cn_part} {en_part}', 'bestiary_entry'
            return f'{cn_part} {en_name}', 'bestiary_entry'

    return None, 'no_data'


# ========= MAIN =========
UUID_PAT = re.compile(
    r'(@UUID\[Compendium\.(pf2e|sf2e)\.([^.\]]+)\.(?:Item|JournalEntry|Actor)\.([^\]]+)\])(\{([^}]*)\})?'
)


def process_text(text):
    """Process all @UUID refs in given text. Returns (new_text, stats)."""
    stats = Counter()
    added_examples = []  # (iid, new_label)
    upgraded_examples = []  # (iid, old, new)
    skipped_examples = []  # (iid, reason)

    def repl(m):
        prefix = m.group(1)
        ns = m.group(2)
        pack = m.group(3)
        iid_full = m.group(4)
        # For JournalEntryPage, take trailing id segment
        iid = iid_full.split('.')[-1] if '.' in iid_full else iid_full
        existing_label = m.group(6)

        # Style: bilingual unless override
        style = STYLE.get(pack, 'bi')

        if existing_label is None:
            # No label - try to add
            label, src = build_label_for_id(iid, pack, None)
            if label:
                stats[f'added_{src}'] += 1
                if len(added_examples) < 8:
                    added_examples.append((iid, pack, label))
                return f'{prefix}{{{label}}}'
            else:
                stats[f'skip_nolabel_{src}'] += 1
                if len(skipped_examples) < 8:
                    skipped_examples.append((iid, pack, src))
                return m.group(0)

        # Has existing label
        has_cn = bool(re.search(r'[一-鿿]', existing_label))
        has_en = bool(re.search(r'[A-Za-z]', existing_label))

        if has_cn:
            # Already has CN, leave alone
            stats['kept_has_cn'] += 1
            return m.group(0)

        # EN-only label - upgrade
        label, src = build_label(pack, existing_label, style)
        if label:
            stats[f'upgraded_{src}'] += 1
            if len(upgraded_examples) < 8:
                upgraded_examples.append((iid, pack, existing_label, label))
            return f'{prefix}{{{label}}}'
        else:
            stats[f'skip_enonly_{src}'] += 1
            return m.group(0)

    new_text = UUID_PAT.sub(repl, text)
    return new_text, stats, added_examples, upgraded_examples, skipped_examples


def fix_file(path):
    print(f'\n=== Processing {os.path.basename(path)} ===')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    raw = json.dumps(data, ensure_ascii=False)
    new_raw, stats, added, upgraded, skipped = process_text(raw)
    # Validate JSON
    try:
        new_data = json.loads(new_raw)
    except json.JSONDecodeError as e:
        print(f'JSON INVALID after fix: {e}')
        return None, None
    print('Stats:')
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}')
    print('Added examples:')
    for iid, pack, lbl in added[:5]:
        print(f'  {pack}/{iid} -> {{{lbl}}}')
    print('Upgraded examples:')
    for iid, pack, old, new in upgraded[:5]:
        print(f'  {pack}/{iid} {{{old}}} -> {{{new}}}')
    print('Skip examples:')
    for iid, pack, src in skipped[:5]:
        print(f'  {pack}/{iid} (reason={src})')
    return new_data, stats


if __name__ == '__main__':
    targets = [
        os.path.join(ROOT, 'FotRP', '需要翻译',
                     'fist-of-the-ruby-phoenix-addons.fist-of-the-ruby-phoenix-addons.json'),
        os.path.join(ROOT, 'FotRP', '需要翻译',
                     'pf2e.fists-of-the-ruby-phoenix-bestiary.json'),
    ]
    results = []
    for t in targets:
        new_data, stats = fix_file(t)
        if new_data is None:
            print(f'FAILED: {t}')
            continue
        # Backup
        backup = t + '.bak.uuid_enrich'
        if not os.path.exists(backup):
            import shutil
            shutil.copy2(t, backup)
            print(f'Backup: {backup}')
        # Write
        with open(t, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        print(f'Wrote: {t}')
        results.append((t, stats))
    print('\n=== SUMMARY ===')
    for t, st in results:
        print(f'{os.path.basename(t)}:')
        for k, v in sorted(st.items()):
            print(f'  {k}: {v}')
