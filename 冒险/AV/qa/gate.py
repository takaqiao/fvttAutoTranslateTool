"""gate.py - the one command that says PASS or FAIL for the AV family.

Every other script in this directory reports; none of them exit non-zero on their own,
and several write scratch files to the CWD.  That is fine while iterating and useless
as a release gate, so this wraps them: fixed CWD (``qa/reports``), explicit thresholds,
and a non-zero exit the moment any check fails.

Checks, in the order a defect would be introduced:

  targets    every published file name resolves to an installed <module>.<pack>   (0 BROKEN)
  binding    every installed document resolves to a translation key, and every
             key resolves to a document                    (bound >= 98%, orphan 0)
  html       no leaf has unbalanced tags
  markup     enricher/UUID bracket bodies contain no CJK (a translated id = dead link)
  bilingual  prose carries no appended English block; name leaves keep `中文 English`
  names      one English name -> one Chinese rendering
  terms      no known variant survives (the rules in _terms.json are idempotent)
  patches    every entry in _path_patches.json is satisfied

Usage:
  python gate.py                       # workspace, installed modules
  python gate.py --cn-dir <dir>        # e.g. the publish repo's compendium/
  python gate.py --skip targets        # when the modules dir is not available
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
AV = HERE.parent
REPORTS = HERE / "reports"

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN_TAIL = re.compile(r"\s+[\x20-\x7E\u2018\u2019\u2013\u2014]+$")
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
# The machine half of an enricher: everything inside [] must stay byte-identical.
BRACKET = re.compile(r"@[A-Za-z]+\[([^\]]*)\]|\[\[([^\]]*)\]\]")
ENGLISH_RUN = re.compile(r"[A-Za-z][A-Za-z ,.'’\-]{60,}")
TAG = re.compile(r"<[^>]+>")
VOID = {"br", "hr", "img", "input", "meta", "link", "col", "source", "area", "base", "wbr", "embed"}

PROSE_UNDER_NOTES = {"label", "summary", "text", "description", "content",
                     "public", "private", "gamemaster", "caption", "subtitle"}


class Balance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.ok = [], True

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    self.ok = False
            else:
                self.ok = False
        else:
            self.stack.pop()


def balanced(text):
    if "<" not in text:
        return True
    parser = Balance()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return False
    return parser.ok and not parser.stack


# Segments whose value the player reads. Chinese there is correct, not a defect.
LABEL_SEGMENT_KEYS = {"name", "text", "label"}


def machine_cjk(body):
    """True when CJK sits in a MACHINE part of a bracket body.

    `@Check[thievery|dc:22|name:擦去血迹]` is right - `name:` is the label the
    player reads. `traits:机械,陷阱` is wrong - those are matched against English
    keys. A `#flavor` tail is display text too.
    """
    head = body.split("#", 1)[0]
    for segment in head.split("|"):
        key = segment.split(":", 1)[0] if ":" in segment else None
        if key in LABEL_SEGMENT_KEYS:
            continue
        if CJK.search(segment):
            return True
    return False


def homograph_exemptions():
    """English names deliberately rendered two ways, each recorded with its reason."""
    path = HERE / "_path_patches.json"
    if not path.exists():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = set()
    for row in raw.get("_not_patched", []):
        value = row.get("value", "")
        m = re.search(r"\s([ -~]+)$", value)
        if m:
            out.add(m.group(1).strip())
    return out


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def split_bilingual(value):
    if not CJK.search(value):
        return None, None
    m = LATIN_TAIL.search(value.strip())
    if not m:
        return None, None
    head = value.strip()[: m.start()].strip()
    tail = value.strip()[m.start():].strip()
    if not head or not tail or CJK.search(tail):
        return None, None
    return head, tail


def is_prose(path, key):
    if key in NAME_KEYS or "folders" in path:
        return False
    if len(path) >= 2 and path[-2] == "notes" and key not in PROSE_UNDER_NOTES:
        return False
    return True


def run(cmd, cwd=REPORTS):
    proc = subprocess.run([sys.executable, *cmd], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, default=AV / "工作区")
    parser.add_argument("--keys", type=Path, default=REPORTS / "pack-keys.json")
    parser.add_argument("--modules", type=Path,
                        default=Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules"))
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    REPORTS.mkdir(parents=True, exist_ok=True)

    results = []

    def record(name, ok, detail):
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<10} {detail}")

    # ---- targets -----------------------------------------------------------
    if "targets" not in args.skip:
        code, out = run([str(HERE / "check_pack_targets.py"),
                         "--compendium", str(args.cn_dir), "--modules", str(args.modules)])
        broken = re.search(r"BROKEN\s*:\s*(\d+)", out)
        n = int(broken.group(1)) if broken else -1
        record("targets", code == 0 and n == 0, f"BROKEN={n}")

    # ---- binding -----------------------------------------------------------
    if "binding" not in args.skip:
        code, out = run([str(HERE / "scan_pack_binding.py"), "--keys", str(args.keys),
                         "--dir", str(args.cn_dir), "--exclusions", str(HERE / "EXCLUSIONS.binding.json"),
                         "--quiet-unbound", "--report", str(REPORTS / "gate_binding.json")])
        fails = [ln for ln in out.splitlines() if ln.startswith("[FAIL]")]
        passes = [ln for ln in out.splitlines() if ln.startswith("[PASS]")]
        record("binding", code == 0 and not fails, f"{len(passes)} packs pass, {len(fails)} fail")
        if fails and args.verbose:
            for ln in fails:
                print("        " + ln)

    # ---- leaf-level checks (one pass over every file) -----------------------
    html_bad, markup_bad, biling_bad = [], [], []
    renderings = {}
    excused_path = HERE / "EXCLUSIONS.bilingual.json"
    excused_leaves = set()
    if excused_path.exists():
        excused_leaves = {row["path"] for row in
                          json.loads(excused_path.read_text(encoding="utf-8")).get("leaves", [])}
    excused_count = 0
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        if cn_path.name in {"labels.json", "titles.json"}:
            continue
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        for path, value in walk(data.get("entries", {}), ("entries",)):
            key = path[-1]
            if "<" in value and not balanced(value):
                html_bad.append(f"{cn_path.name}:{'.'.join(path[-3:])}")
            for m in BRACKET.finditer(value):
                body = m.group(1) or m.group(2) or ""
                if CJK.search(body) and machine_cjk(body):
                    markup_bad.append(f"{cn_path.name}:{'.'.join(path[-3:])}: {body[:50]}")
            if is_prose(path, key) and CJK.search(value):
                visible = TAG.sub(" ", BRACKET.sub(" ", value))
                if ENGLISH_RUN.search(visible):
                    full = f"{cn_path.name}:{'.'.join(path[1:])}"
                    if full in excused_leaves:
                        excused_count += 1
                    else:
                        biling_bad.append(f"{cn_path.name}:{'.'.join(path[-3:])}")
            if key in NAME_KEYS:
                head, tail = split_bilingual(value)
                if head:
                    renderings.setdefault(tail, {}).setdefault(head, 0)
                    renderings[tail][head] += 1

    record("html", not html_bad, f"{len(html_bad)} unbalanced leaves")
    record("markup", not markup_bad, f"{len(markup_bad)} enricher bodies containing CJK")
    record("bilingual", not biling_bad,
           f"{len(biling_bad)} prose leaves with an English run"
           + (f"  [excused {excused_count}: credits/OGL]" if excused_count else ""))
    conflicts = {en: c for en, c in renderings.items() if len(c) > 1
                 and en not in homograph_exemptions()}
    record("names", not conflicts, f"{len(conflicts)} English names with >1 Chinese rendering")

    for label, rows in (("html", html_bad), ("markup", markup_bad),
                        ("bilingual", biling_bad)):
        if rows and args.verbose:
            for row in rows[:10]:
                print(f"        {label}: {row}")
    if conflicts and args.verbose:
        for en, c in list(conflicts.items())[:10]:
            print(f"        names: {en} -> {c}")

    # ---- terms are idempotent ---------------------------------------------
    if "terms" not in args.skip:
        code, out = run([str(HERE / "normalize_terms.py"), "--terms", str(HERE / "_terms.json"),
                         "--target", str(args.cn_dir)])
        m = re.search(r"total replacements:\s*(\d+)", out)
        n = int(m.group(1)) if m else -1
        record("terms", n == 0, f"{n} variants would still be replaced")

    # ---- path patches all satisfied ---------------------------------------
    patches = HERE / "_path_patches.json"
    if patches.exists() and "patches" not in args.skip:
        code, out = run([str(HERE / "apply_path_patches.py"), "--cn-dir", str(args.cn_dir),
                         "--patches", str(patches)])
        m = re.search(r"(\d+) already satisfied, (\d+) errors", out)
        sat, err = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        would = re.search(r"(\d+) would apply", out)
        pending = int(would.group(1)) if would else 0
        record("patches", err == 0 and pending == 0, f"{sat} satisfied, {pending} pending, {err} errors")

    failed = [name for name, ok, _ in results if not ok]
    print("\n" + ("=" * 62))
    print(f"{'GATE FAILED: ' + ', '.join(failed) if failed else 'GATE PASSED'}"
          f"   ({len(results) - len(failed)}/{len(results)} checks)")
    (REPORTS / "gate.json").write_text(json.dumps(
        {"cn_dir": str(args.cn_dir),
         "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in results],
         "html_bad": html_bad[:200], "markup_bad": markup_bad[:200],
         "bilingual_bad": biling_bad[:200],
         "name_conflicts": {k: v for k, v in list(conflicts.items())[:200]}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
