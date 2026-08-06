import json, os

R = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\5c77a7f2-c3e4-4625-9ecd-a1f06da3f0ba\scratchpad\parallel"
WAVE2 = ["the-winding-trail", "gamemaster-s-guide", "the-expedition-challenge",
         "chapter-2-events", "players-guide", "mythspire-observatory",
         "tables-1", "tables-2", "tables-3"]
for d in WAVE2:
    p = os.path.join(R, d)
    t = os.path.join(p, "todo.json")
    b = os.path.join(p, "batch.json")
    todo = len(json.load(open(t, encoding="utf-8"))["items"]) if os.path.exists(t) else 0
    if os.path.exists(b):
        try:
            n = len(json.load(open(b, encoding="utf-8-sig")))
            print(f"{d:<28} batch {n:>4} / todo {todo:<4}  {'完整' if n >= todo else '不完整'}")
        except Exception as e:
            print(f"{d:<28} batch 解析失败: {e}")
    else:
        print(f"{d:<28} (无 batch) / todo {todo}")
