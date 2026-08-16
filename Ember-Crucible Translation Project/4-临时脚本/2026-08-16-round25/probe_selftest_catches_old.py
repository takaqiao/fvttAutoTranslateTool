# -*- coding: utf-8 -*-
"""落盘探针：证明新加的 twin_files 自检用例**不是空转** —— 把 `.get(k, k)` 那个
静默兜底装回去，用例必须变红。

不改任何在册文件：只在内存里把 `_twin_repo_dir` 换成旧行为（取不到就拿键本身当路径），
然后跑 `run_twin_selftest()`。旧行为下期望：
  · 用例 2（副本树注入漂移，cwd 在项目根）→ FAIL（它比的是真实树）
  · 用例 3（仓名写成目录名）→ FAIL（静默退化成裸路径，不炸）
"""
import importlib.util
import os
import sys

QA = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\3-常用脚本\qa"
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

spec = importlib.util.spec_from_file_location(
    "ar", os.path.join(QA, "assert_resolutions.py"))
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def old_twin_repo_dir(name, ctx):
    """第二十四轮以前的行为：`ctx.repos.get(k, k)`，取不到就拿键本身当路径。"""
    return ctx.repos.get(name, name)


print("### 基线：现行实现")
n_bad_new, n_new = ar.run_twin_selftest()

print("\n### 把静默兜底装回去（模拟旧实现）")
ar._twin_repo_dir = old_twin_repo_dir
n_bad_old, n_old = ar.run_twin_selftest()

print("\n" + "=" * 60)
print(f"现行实现：{n_new - n_bad_new} / {n_new} 通过")
print(f"旧实现  ：{n_old - n_bad_old} / {n_old} 通过"
      f"　←　必须 < {n_old}，否则这组用例是空转")
print("结论：" + ("用例能抓住旧实现 —— 灵敏度回测成立"
                 if n_bad_old > 0 else
                 "⚠ 用例抓不住旧实现 —— 这组用例本身在空转"))
sys.exit(0 if (n_bad_new == 0 and n_bad_old > 0) else 1)
