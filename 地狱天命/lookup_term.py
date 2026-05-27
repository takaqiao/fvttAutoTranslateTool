"""On-demand lookup against the full SRD TM (tm_3source.json, 38K terms, 20MB).

Usage:
    python lookup_term.py "Halberd"
    python lookup_term.py "Acid Splash" "Magic Missile" "Shield"

Prints JSON-like dict for each found term; "(miss)" for misses.
This is the cheap way to query the giant SRD TM without loading it into context.
"""
import json
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
TM = HERE.parent / "翻译流程" / "tm_cache" / "tm_3source.json"

if not TM.exists():
    print(f"[err] TM not found: {TM}", file=sys.stderr)
    sys.exit(1)

_cache = {"loaded": False, "data": None}


def load():
    if not _cache["loaded"]:
        with TM.open(encoding="utf-8") as f:
            _cache["data"] = json.load(f)
        _cache["loaded"] = True
    return _cache["data"]


def query(en: str) -> dict | None:
    data = load()
    hit = data.get(en)
    if hit:
        return {"en": en, "match": "exact", **hit}
    # Try stripping common suffixes
    for suffix in [" (Greater)", " (Lesser)", " (Major)", " (Minor)", " (Self Only)", " (At Will)", " (Constant)"]:
        if en.endswith(suffix):
            base = en[:-len(suffix)]
            hit = data.get(base)
            if hit:
                return {"en": en, "match": f"stripped:{suffix}", "base": base, **hit}
    # Case-insensitive fallback
    for k in data:
        if k.lower() == en.lower():
            return {"en": en, "match": "case-insensitive", "actual_key": k, **data[k]}
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python lookup_term.py <term1> [<term2> ...]", file=sys.stderr)
        sys.exit(2)
    for term in sys.argv[1:]:
        r = query(term)
        if r is None:
            print(f"--- {term} ---\n(miss)")
        else:
            print(f"--- {term} ---")
            print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
