#!/usr/bin/env python3
"""Compare a fresh EN extraction against an old EN extraction and the CN translation.

Reports, per pack:
  - entries present in new EN but missing in CN            -> NEW (needs translation)
  - entries present in CN but absent from new EN           -> STALE (removed upstream)
  - entries whose EN source text changed since old EN      -> DRIFT (needs re-translation)
  - leaf fields present in new EN but missing in CN entry  -> PARTIAL (field-level gap)
"""
import json, os, sys, argparse
from collections import OrderedDict

def load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def leaves(obj, prefix=""):
    """Yield (path, str_value) for every string leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield (prefix, obj)

def norm(s):
    return " ".join(s.split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-en', required=True)
    ap.add_argument('--old-en')
    ap.add_argument('--cn', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    report = OrderedDict()
    totals = dict(new=0, stale=0, drift=0, partial=0, en_entries=0, cn_entries=0)

    files = sorted(f for f in os.listdir(a.new_en) if f.endswith('.json'))
    for fn in files:
        new_en = load(os.path.join(a.new_en, fn))
        cn_path = os.path.join(a.cn, fn)
        old_path = os.path.join(a.old_en, fn) if a.old_en else None
        cn = load(cn_path) if os.path.exists(cn_path) else None
        old_en = load(old_path) if old_path and os.path.exists(old_path) else None

        ne = new_en.get('entries', {})
        ce = (cn or {}).get('entries', {})
        oe = (old_en or {}).get('entries', {})

        new_keys = [k for k in ne if k not in ce]
        stale_keys = [k for k in ce if k not in ne]
        drift_keys = []
        partial = {}

        for k in ne:
            if k not in ce:
                continue
            # DRIFT: english text changed vs old extraction
            if oe and k in oe:
                nl = {p: norm(v) for p, v in leaves(ne[k])}
                ol = {p: norm(v) for p, v in leaves(oe[k])}
                changed = [p for p in nl if p in ol and nl[p] != ol[p]]
                added = [p for p in nl if p not in ol]
                if changed or added:
                    drift_keys.append({'key': k, 'changed': changed[:12],
                                       'added': added[:12],
                                       'n_changed': len(changed), 'n_added': len(added)})
            # PARTIAL: leaf path in EN but missing in CN entry
            cl = {p for p, _ in leaves(ce[k])}
            miss = [p for p, _ in leaves(ne[k]) if p not in cl]
            if miss:
                partial[k] = {'missing': miss[:15], 'count': len(miss)}

        report[fn] = {
            'cn_file_exists': cn is not None,
            'old_en_exists': old_en is not None,
            'en_entries': len(ne),
            'cn_entries': len(ce),
            'new_count': len(new_keys),
            'new_keys': new_keys[:400],
            'stale_count': len(stale_keys),
            'stale_keys': stale_keys[:200],
            'drift_count': len(drift_keys),
            'drift': drift_keys[:200],
            'partial_count': len(partial),
            'partial': dict(list(partial.items())[:150]),
            'folders_en': len((new_en.get('folders') or {})),
            'folders_cn': len(((cn or {}).get('folders') or {})),
            'mapping_changed': (new_en.get('mapping') != (cn or {}).get('mapping')),
        }
        totals['new'] += len(new_keys); totals['stale'] += len(stale_keys)
        totals['drift'] += len(drift_keys); totals['partial'] += len(partial)
        totals['en_entries'] += len(ne); totals['cn_entries'] += len(ce)

    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump({'totals': totals, 'packs': report}, f, ensure_ascii=False, indent=2)

    print(f"{'pack':<42}{'EN':>6}{'CN':>6}{'NEW':>6}{'STALE':>7}{'DRIFT':>7}{'PART':>6}  map?")
    for fn, r in report.items():
        flag = '' if r['cn_file_exists'] else '  <-- NO CN FILE'
        print(f"{fn:<42}{r['en_entries']:>6}{r['cn_entries']:>6}{r['new_count']:>6}"
              f"{r['stale_count']:>7}{r['drift_count']:>7}{r['partial_count']:>6}"
              f"  {'CHG' if r['mapping_changed'] else '-'}{flag}")
    print(f"\nTOTAL en={totals['en_entries']} cn={totals['cn_entries']} "
          f"new={totals['new']} stale={totals['stale']} drift={totals['drift']} partial={totals['partial']}")

main()
