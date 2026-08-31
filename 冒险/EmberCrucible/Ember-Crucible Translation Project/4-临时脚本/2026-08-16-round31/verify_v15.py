# -*- coding: utf-8 -*-
"""V15 回测：把执行体掏空之后，**只跑主闸**还能不能全绿。

⚠ 这个探针要动真身判据文件，所以：
  · 先按**字节**备份，跑完立刻还原，并断言 sha256 与备份一致（还原不干净就 abort）；
  · 打补丁走纯 str.replace（不走正则）—— 硬约束 3；
  · 前置自证：还原后先跑一次主闸，必须 rc=0，否则说明我把文件弄坏了。
"""
import hashlib, os, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
JUDGE = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")

orig = open(JUDGE, "rb").read()
h0 = hashlib.sha256(orig).hexdigest()
print(f"备份 ok：{len(orig)} B / {h0[:12]}")


def gate():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, JUDGE], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, cwd=ROOT)
    out = p.stdout or ""
    tally = [l for l in out.splitlines() if l.startswith("通过 ")]
    note = [l for l in out.splitlines() if "补跑一遍自检" in l or "自检没过" in l
            or "不重跑自检" in l or "回测合计" in l]
    return p.returncode, (tally[-1] if tally else "?"), note


try:
    for name, old, new in [
        ("a_cn_absent 整体换成 return [], ''",
         'def a_cn_absent(rule, ctx):\n',
         'def a_cn_absent(rule, ctx):\n    return [], ""\n'),
        ("a_term_gated 整体换成 return [], ''",
         'def a_term_gated(rule, ctx):\n',
         'def a_term_gated(rule, ctx):\n    return [], ""\n'),
        ("_payload_floor_check 整体换成 return [], 0",
         'def _payload_floor_check(A):\n',
         'def _payload_floor_check(A):\n    return [], 0\n'),
    ]:
        src = orig.decode("utf-8")
        assert src.count(old) == 1, f"锚点不唯一：{old!r}"
        open(JUDGE, "w", encoding="utf-8", newline="").write(src.replace(old, new))
        rc, tally, note = gate()
        print(f"\n【{name}】\n  rc={rc}  {tally}")
        for n in note:
            print(f"  {n.strip()[:150]}")
        print(f"  ⇒ {'红 ✅（只跑主闸也被接住）' if rc else '绿 ❌（只跑主闸接不住）'}")
finally:
    open(JUDGE, "wb").write(orig)
    h1 = hashlib.sha256(open(JUDGE, "rb").read()).hexdigest()
    assert h1 == h0, f"还原不干净！{h0[:12]} vs {h1[:12]}"
    print(f"\n还原 ok：{h1[:12]}（与备份逐字节相同）")

rc, tally, note = gate()
print(f"后置自证：还原后主闸 rc={rc} {tally}")
