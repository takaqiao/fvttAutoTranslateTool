# -*- coding: utf-8 -*-
import json, re, sys

KNOWN_TERMS = {
    'Shulzara': '舒尔扎拉',
    'Sangrist': '血奉卫',
    'Herexen': '叛道者',
    'Lukahn': '卢卡恩',
    'Genevye': '吉奈维',
    'Orrery of Light': '光明天球仪',
    'Shadow Walls': '阴影之墙',
    'everlight crystal': '永恒光晶',
    'Critical Success': '大成功',
    'Critical Failure': '严重失败',
    'Success': '成功',
    'Failure': '失败',
}

def has_zh(s):
    return any(ord(c) >= 0x4e00 and ord(c) <= 0x9fff for c in s)

def strip_enrichers(s):
    s = re.sub(r'@\w+\[[^\]]+\](\{[^}]*\})?', '', s)
    s = re.sub(r'\[\[/[^\]]+\]\](\{[^}]*\})?', '', s)
    s = re.sub(r'<[^>]+>', ' ', s)
    return s

files = [
    (r'C:\Users\Taka\Desktop\fvtt\gracye\pf2e-secrets-of-grayce.secrets-of-grayce.json', 'MAIN'),
    (r'C:\Users\Taka\Desktop\fvtt\gracye\pf2e.menace-under-otari-bestiary.json', 'MENACE'),
    (r'C:\Users\Taka\Desktop\fvtt\gracye\pf2e.troubles-in-grayce-bestiary.json', 'TROUBLES'),
]

for fname, label in files:
    try:
        with open(fname, encoding='utf-8') as f:
            data = json.load(f)
        content_str = json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print(f"\nERROR loading {label}: {e}")
        continue
    
    print(f"\n{'='*70}")
    print(f"CONTENT AUDIT: {label}")
    print('='*70)
    
    print("\n--- KNOWN TERM CONSISTENCY ---")
    for en_term, zh_term in KNOWN_TERMS.items():
        zh_count = content_str.count(zh_term)
        en_in_zh = 0
        for m in re.finditer(re.escape(en_term), content_str):
            ctx_start = max(0, m.start() - 100)
            ctx_end = min(len(content_str), m.end() + 100)
            ctx = content_str[ctx_start:ctx_end]
            if has_zh(ctx):
                en_in_zh += 1
        if zh_count > 0 or en_in_zh > 0:
            print(f"  {en_term} -> {zh_term}: zh_occurrences={zh_count}, en_in_zh_context={en_in_zh}")
    
    print("\n--- DESCRIPTION FIELD SAMPLES (Chinese fields, check for MT artifacts) ---")
    samples = []
    
    def collect_samples(obj, path="", depth=0):
        if depth > 8:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ('description', 'publicNotes', 'text') and isinstance(v, str):
                    if has_zh(v) and len(v) > 50:
                        samples.append((path + '/' + k, v))
                else:
                    collect_samples(v, path + '/' + k, depth+1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:3]):
                collect_samples(v, path + f'[{i}]', depth+1)
    
    collect_samples(data)
    print(f"  Total translated description fields: {len(samples)}")
    
    issues = []
    for path, text in samples:
        clean = strip_enrichers(text)
        en_words = re.findall(r'[A-Za-z]{5,}', clean)
        en_long = [w for w in en_words if w not in ('Strong', 'Effect', 'Critical', 'Success', 'Failure', 'Saving', 'Throw', 'Duration', 'Stage', 'Requirements', 'Trigger', 'Activate', 'Frequency', 'Range', 'Onset', 'Maximum', 'Cantrip', 'Description')]
        if len(en_long) > 5:
            issues.append(('EN_WORDS_IN_ZH', path[-60:], ', '.join(en_long[:8])))
        if len(clean.strip()) < 10 and len(text) > 5:
            issues.append(('VERY_SHORT_DESC', path[-60:], repr(text[:60])))
    
    if issues:
        print(f"  Issues found: {len(issues)}")
        for kind, path, info in issues[:30]:
            print(f"  [{kind}] {path}: {info[:100]}")
    else:
        print("  No obvious MT artifacts detected in samples")
    
    print("\n--- @Check DC VALUE AUDIT ---")
    dc_issues = []
    for m in re.finditer(r'@Check\[([^\]]+)\]', content_str):
        inner = m.group(1)
        dc_match = re.search(r'dc:(\d+)', inner)
        if dc_match:
            dc = int(dc_match.group(1))
            if dc < 5 or dc > 60:
                dc_issues.append(f"  Unusual DC={dc}: {repr(inner[:60])}")
        type_match = re.search(r'type:(\w+)', inner)
        if type_match:
            check_type = type_match.group(1)
            valid_types = {'fortitude', 'reflex', 'will', 'perception', 'acrobatics', 'athletics', 'deception', 'diplomacy', 'intimidation', 'medicine', 'nature', 'occultism', 'performance', 'religion', 'society', 'stealth', 'survival', 'thievery', 'arcana', 'crafting'}
            if check_type.lower() not in valid_types:
                dc_issues.append(f"  Unknown check type: {repr(check_type)} in {repr(inner[:60])}")
    
    if dc_issues:
        print(f"  DC issues ({len(dc_issues)}):")
        for issue in dc_issues[:20]:
            print(issue)
    else:
        print("  OK - all @Check DC values look normal")
    
    print("\n--- DOUBLED CHARACTER / STUTTER DETECTION ---")
    
    def walk2(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk2(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from walk2(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            yield (path, obj)
    
    doubled_issues = []
    for path, val in walk2(data):
        if has_zh(val) and len(val) > 5:
            for m in re.finditer(r'([一-鿿])\1{2,}', val):
                doubled_issues.append(f"  Repeated char '{m.group(1)}' in {path[-50:]}: {repr(val[max(0,m.start()-20):m.end()+20])}")
            for m in re.finditer(r'([一-鿿]{2,4})\1', val):
                if len(m.group(1)) >= 2:
                    doubled_issues.append(f"  Doubled phrase '{m.group(1)}' in {path[-50:]}: {repr(val[max(0,m.start()-10):m.end()+10])}")
    
    if doubled_issues:
        print(f"  Doubled/stutter issues ({len(doubled_issues)}):")
        for d in doubled_issues[:20]:
            print(d)
    else:
        print("  OK - no doubled characters/stuttering detected")

print("\n=== CONTENT AUDIT COMPLETE ===")
