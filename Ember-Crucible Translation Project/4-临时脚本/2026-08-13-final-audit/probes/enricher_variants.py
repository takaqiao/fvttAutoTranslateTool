# -*- coding: utf-8 -*-
"""
enricher_variants.py —— 探针 B：按「输出面」细分增强器调用形态

对每个增强器，上游源码里往往有 **多条输出分支**（带/不带 award、mood / music /
reset、passive / group …）。ember-hardcoded-cn.mjs 的替换表可能只覆盖了其中一条。
这个探针把调用按形态分桶，让「哪条分支有多少处」可数。

只读，不写库。
"""
import json, os, re, collections

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {
    "ember": os.path.join(ROOT, "1-Ember汉化插件"),
    "crucible": os.path.join(ROOT, "2-Crucible汉化插件"),
}

BUCKETS = [
    # (name, regex)
    ("attunement_plain", re.compile(r"\[\[/attunement (\w+)\]\]")),
    ("attunement_award", re.compile(r"\[\[/attunement (\w+) ([+-]?\d+) (\w+)\]\]")),
    ("soundscape_mood", re.compile(r"\[\[/soundscape\s+mood=(\w+)\s*\]\]")),
    ("soundscape_reset", re.compile(r"\[\[/soundscape\s+(\w+)\s+reset\s*\]\]")),
    ("soundscape_music", re.compile(r"\[\[/soundscape\s+(\w+)\s+(?!reset)(\w+)(?:\s+(\w+))?\s*\]\]")),
    ("talent", re.compile(r"\[\[/talent ([\w\-.]+)\]\]")),
    ("spell_at", re.compile(r"@Spell\[([\w.]+)\]")),
    ("counterspell", re.compile(r"\[\[/counterspell ([^\]]*)\]\]")),
    ("milestone", re.compile(r"\[\[/milestone(?: \d+)?\]\]")),
    ("loot", re.compile(r"@Loot\[([^\]]*)\]")),
    ("scroll", re.compile(r"@Scroll\[([^\]]*)\]")),
    ("skillcheck_group", re.compile(r"\[\[/skillCheck ([\w\s]*\bgroup\b[\w\s]*)\]\]")),
    ("skillcheck_passive", re.compile(r"\[\[/skillCheck ([\w\s]*\bpassive\b[\w\s]*)\]\]")),
    ("hazard_named", re.compile(r"\[\[/hazard ([\w\s]+)\]\]\{([^}]+)\}")),
    ("rule_labeled", re.compile(r"@Rule\[([\w.]+)\]\{([^}]+)\}")),
]


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, str):
        yield path, o


def main():
    counts = collections.Counter()
    per_file = collections.defaultdict(collections.Counter)
    args = collections.defaultdict(collections.Counter)
    samples = collections.defaultdict(list)
    for repo, base in REPOS.items():
        d = os.path.join(base, "compendium", "cn")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                data = json.load(f)
            for p, s in walk(data):
                for name, rx in BUCKETS:
                    for m in rx.finditer(s):
                        counts[name] += 1
                        per_file[name][fn] += 1
                        args[name][m.group(0)] += 1
                        if len(samples[name]) < 3:
                            samples[name].append((fn, p[:100]))
    for name, _ in BUCKETS:
        print(f"\n=== {name}: {counts[name]}  {dict(per_file[name])}")
        for a, c in args[name].most_common(25):
            print(f"      {c:5d}  {a}")


if __name__ == "__main__":
    main()
