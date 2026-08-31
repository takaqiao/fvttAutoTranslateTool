# -*- coding: utf-8 -*-
"""对**真身主闸**逐条复现复核那 12 个「主闸全绿、护栏却真的没了」的单点动作。

⚠ 硬约束 4（前置自证）：先跑一次**原样**副本，必须 66/0/0、rc=0；
  跑不出这个基线就说明我的跑法本身是错的，后面 12 条「变红了」全是假的。
⚠ 变异的是**规则集副本**（--rules），不动仓里那份。
"""
import copy, json, os, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
JUDGE = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")
RULES = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")

base = json.load(open(RULES, encoding="utf-8"))
assert len(base["assertions"]) == 66, "前置自证：条数不是 66"


def idx(A, rid):
    return next(i for i, r in enumerate(A) if r.get("id") == rid)


def run(mut=None, tag="base"):
    d = copy.deepcopy(base)
    if mut:
        mut(d["assertions"])
    p = os.path.join(HERE, f"_rules_{tag}.json")
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, JUDGE, "--rules", p],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, cwd=ROOT)
    out = proc.stdout or ""
    fails = [l.strip() for l in out.splitlines() if l.strip().startswith("FAIL")]
    tally = [l for l in out.splitlines() if l.startswith("通过 ")]
    os.remove(p)
    return proc.returncode, (tally[-1] if tally else "?"), fails


rc, tally, fails = run(None, "base")
print(f"前置自证（原样副本）：rc={rc} {tally}")
assert rc == 0 and "失败 0" in tally, "基线不绿，后面的结论全不作数"

CASES = []


def add(name, fn):
    CASES.append((name, fn))


def _shell(A):
    i = idx(A, "R-hex-tile")
    A[i] = {"id": "R-hex-tile", "kind": "cn_absent", "cn": ["绝不可能XYZ"]}


add("① R-hex-tile 整条换成只剩 id/kind/cn 的壳", _shell)


def _wipe(A):
    for x in A:
        if x.get("kind") == "cn_absent":
            x["cn"] = [] if isinstance(x.get("cn"), list) else ""


add("② 15 条 cn_absent 的禁用词表一次全部清空", _wipe)
add("③ R-bare-cn-fields.fields=[]",
    lambda A: A[idx(A, "R-bare-cn-fields")].__setitem__("fields", []))
add("④ R-lang-parity.packages={}",
    lambda A: A[idx(A, "R-lang-parity")].__setitem__("packages", {}))
for rid in ("R-lantyr", "R-orb-destroyed", "R-obsidian-antiquary", "R-temple-lunarium"):
    add(f"⑤ {rid}.cn_forbidden=[]",
        (lambda rid=rid: lambda A: A[idx(A, rid)].__setitem__("cn_forbidden", []))())
add("⑥ R-anchor-ids.min 900 → 0",
    lambda A: A[idx(A, "R-anchor-ids")].__setitem__("min", 0))
add("⑦ R-readaloud-coverage.min_sentence_frac 0.75 → 0",
    lambda A: A[idx(A, "R-readaloud-coverage")].__setitem__("min_sentence_frac", 0))
add("⑧ R-electricity-lightning.case_sensitive → false",
    lambda A: A[idx(A, "R-electricity-lightning")].__setitem__("case_sensitive", False))
add("⑨ R-glossary-not-laundering.entries={}",
    lambda A: A[idx(A, "R-glossary-not-laundering")].__setitem__("entries", {}))
add("⑩ R-aura-three-way.domains 砍掉一域",
    lambda A: A[idx(A, "R-aura-three-way")].__setitem__(
        "domains", A[idx(A, "R-aura-three-way")]["domains"][:2]))

print(f"\n逐条复现（共 {len(CASES)} 个动作）：")
red = 0
for i, (name, fn) in enumerate(CASES):
    rc, tally, fails = run(fn, f"m{i}")
    ok = rc != 0
    red += ok
    print(f"  {'红 ✅' if ok else '绿 ❌'} {name}")
    print(f"        rc={rc} {tally}；FAIL 行：{fails[:3]}")
print(f"\n合计：{red} / {len(CASES)} 变红")
