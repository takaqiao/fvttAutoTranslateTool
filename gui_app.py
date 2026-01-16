import json
import os
import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

CONFIG_PATH = Path("translator_config.json")
GUIDE_PATH = Path("软件指南.md")

DEFAULTS = {
    "GOOGLE_API_KEY": "",
    "OPENAI_API_KEY": "",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "SYNC_MODE": "TARGET_MASTER",
    "SOURCE_EN_JSON_PATH": "pf2e-beginner-box-en.json",
    "TARGET_JSON_PATH": "pf2e-beginner-box.adventures.json",
    "MODEL_PRIORITY_LIST": [["openai", "gpt-5.2"], ["openai", "gpt-5-mini"], ["google", "gemini-3-flash-preview"]],
    "MAX_WORKERS": 16,
    "TARGET_RPM": 450,
    "MAX_RETRIES": 5,
    "MAX_AUDIT_ROUNDS": 1,
    "MAX_AUDIT_PASSES": 2,
    "SAFE_MODE": True,
    "TEST_MODE": False,
    "TEST_MODE_WRITE": False,
    "TEST_OUTPUT_PATH": "pf2e-beginner-box.adventures.test.json",
    "TEST_MODE_SIMULATE_PIPELINE": False,
    "SIMULATE_AI": False,
    "BRUTE_FORCE_MODE": False,
    "FULL_BILINGUAL_MODE": True,
    "BILINGUAL_KEYS": ["name", "label", "navName", "header", "tooltip"],
    "CN_ONLY_KEYS": ["tokenName", "caption"],
    "LONG_TEXT_KEYS": ["description", "text", "content", "gm_notes", "gm_description", "publicnotes", "publicNotes"],
    "TRANSLATE_MACROS": True,
    "SKIP_CONTAINERS": [],
    "GLOBAL_GLOSSARY_PATH": "术语译名对照表.csv",
    "LOCAL_GLOSSARY_EXPORT_PATH": "术语表_本地提取.csv",
    "REPORT_XLSX_PATH": "翻译审查报告.xlsx",
    "PROCESS_LOG_PATH": "运行日志.txt",
    "MISSED_LOG_PATH": "失败漏翻记录.txt",
    "HISTORY_FILE_PATH": "translation_history.json",
    "AUDIT_HISTORY_PATH": "audit_history.json",
    "BACKUP_DIR": "backups",
    "PRINT_LOG_TO_TERMINAL": True,
    "USE_TQDM_WRITE": True,
    "TARGET_KEYS": [
        "name", "description", "text", "label", "caption", "value",
        "unidentifiedName", "tokenName", "publicnotes", "publicNotes",
        "gm_notes", "gm_description", "header", "content", "items",
        "navName", "tooltip", "preAuthored"
    ],
    "SPECIAL_CONTAINERS": [
        "notes", "folders", "journal", "journals", "scenes",
        "actors", "items", "pages", "entries", "flags", "system"
    ],
    "FORCE_SIMPLIFIED_POST": True,
    "DIFF_REPORT_ENABLED": True,
    "DIFF_REPORT_PATH": "translation_diff_report.html",
}

PARAM_SECTIONS = {
    "API": [
        ("GOOGLE_API_KEY", "str", "Google API Key（可留空，使用环境变量）"),
        ("OPENAI_API_KEY", "str", "OpenAI API Key（可留空，使用环境变量）"),
        ("OPENAI_BASE_URL", "str", "OpenAI Base URL（默认 https://api.openai.com/v1）"),
    ],
    "文件路径": [
        ("SOURCE_EN_JSON_PATH", "path", "英文源文件路径"),
        ("TARGET_JSON_PATH", "path", "中文目标文件路径"),
        ("TEST_OUTPUT_PATH", "path", "测试模式输出文件路径"),
        ("GLOBAL_GLOSSARY_PATH", "path", "全局术语表 CSV"),
        ("LOCAL_GLOSSARY_EXPORT_PATH", "path", "本地术语表导出 CSV"),
        ("REPORT_XLSX_PATH", "path", "翻译审查报告 Excel"),
        ("PROCESS_LOG_PATH", "path", "运行日志路径"),
        ("MISSED_LOG_PATH", "path", "漏翻日志路径"),
        ("HISTORY_FILE_PATH", "path", "翻译历史缓存"),
        ("AUDIT_HISTORY_PATH", "path", "校对历史"),
        ("BACKUP_DIR", "path", "备份目录"),
        ("DIFF_REPORT_PATH", "path", "差异报告输出 HTML"),
    ],
    "运行模式": [
        ("SYNC_MODE", "str", "同步模式：TARGET_MASTER 或 SOURCE_MASTER"),
        ("SAFE_MODE", "bool", "HTML 安全分段翻译"),
        ("TEST_MODE", "bool", "测试模式（不调用 AI）"),
        ("TEST_MODE_WRITE", "bool", "测试模式写入文件"),
        ("TEST_MODE_SIMULATE_PIPELINE", "bool", "测试模式模拟流程"),
        ("SIMULATE_AI", "bool", "模拟 AI（直接回传）"),
        ("BRUTE_FORCE_MODE", "bool", "强制翻译（忽略部分过滤）"),
        ("FULL_BILINGUAL_MODE", "bool", "启用全字段双语策略"),
        ("TRANSLATE_MACROS", "bool", "是否翻译宏名称"),
        ("FORCE_SIMPLIFIED_POST", "bool", "简体中文后处理（OpenCC）"),
        ("DIFF_REPORT_ENABLED", "bool", "生成差异报告"),
        ("PRINT_LOG_TO_TERMINAL", "bool", "日志同步输出到终端"),
        ("USE_TQDM_WRITE", "bool", "使用 tqdm.write 输出日志"),
    ],
    "性能": [
        ("MAX_WORKERS", "int", "线程池并发数"),
        ("TARGET_RPM", "int", "目标请求速率（RPM）"),
        ("MAX_RETRIES", "int", "单条重试次数"),
        ("MAX_AUDIT_ROUNDS", "int", "校对轮数"),
        ("MAX_AUDIT_PASSES", "int", "同条目最大校对次数"),
    ],
    "模型优先级": [
        ("MODEL_PRIORITY_LIST", "listtuple", "模型优先级列表。可用 JSON 格式：[[\"openai\",\"gpt-5.2\"],...]，或按行写 provider:model"),
    ],
    "键与容器": [
        ("BILINGUAL_KEYS", "set", "输出双语的短字段键名（逗号或换行分隔）"),
        ("CN_ONLY_KEYS", "set", "仅中文输出字段键名"),
        ("LONG_TEXT_KEYS", "set", "长文本字段键名"),
        ("SKIP_CONTAINERS", "set", "跳过翻译的容器名"),
        ("TARGET_KEYS", "set", "翻译目标字段白名单"),
        ("SPECIAL_CONTAINERS", "set", "特殊容器名"),
    ],
}

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.geometry(f"+{x}+{y}")
        label = tk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, justify="left")
        label.pack(ipadx=6, ipady=3)

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

class ScrollableFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PF2e 翻译配置器")
        self.geometry("980x720")
        self.vars = {}

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self._load_config()
        for section, items in PARAM_SECTIONS.items():
            frame = ScrollableFrame(self.notebook)
            self.notebook.add(frame, text=section)
            self._build_section(frame.scrollable_frame, items)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=8)
        ttk.Button(btn_frame, text="保存配置", command=self.save_config).pack(side="left")
        ttk.Button(btn_frame, text="重新加载", command=self.reload_config).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="运行翻译", command=self.run_translator).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="打开软件指南", command=self.open_guide).pack(side="right")

        self.status = ttk.Label(self, text="就绪")
        self.status.pack(fill="x", padx=10, pady=(0, 10))

    def _load_config(self):
        self.config_data = dict(DEFAULTS)
        if CONFIG_PATH.exists():
            try:
                with CONFIG_PATH.open("r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                if isinstance(user_cfg, dict):
                    self.config_data.update(user_cfg)
            except Exception as e:
                messagebox.showwarning("配置读取失败", str(e))

    def _build_section(self, parent, items):
        for row, (name, kind, desc) in enumerate(items):
            ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            widget = None
            var = None
            value = self.config_data.get(name, DEFAULTS.get(name))

            if kind == "bool":
                var = tk.BooleanVar(value=bool(value))
                widget = ttk.Checkbutton(parent, variable=var)
                widget.grid(row=row, column=1, sticky="w", padx=6, pady=4)
            elif kind in {"int", "str", "path"}:
                var = tk.StringVar(value=str(value) if value is not None else "")
                widget = ttk.Entry(parent, textvariable=var, width=80)
                widget.grid(row=row, column=1, sticky="we", padx=6, pady=4)
            elif kind in {"set", "listtuple"}:
                var = tk.StringVar()
                text = tk.Text(parent, height=4, width=80)
                if kind == "set":
                    if isinstance(value, list):
                        text.insert("1.0", "\n".join(value))
                    elif isinstance(value, str):
                        text.insert("1.0", value)
                    else:
                        text.insert("1.0", "\n".join(list(value or [])))
                else:
                    try:
                        text.insert("1.0", json.dumps(value, ensure_ascii=False, indent=2))
                    except Exception:
                        text.insert("1.0", "\n".join([": ".join(map(str, v)) for v in (value or [])]))
                text.grid(row=row, column=1, sticky="we", padx=6, pady=4)
                widget = text
            else:
                var = tk.StringVar(value=str(value) if value is not None else "")
                widget = ttk.Entry(parent, textvariable=var, width=80)
                widget.grid(row=row, column=1, sticky="we", padx=6, pady=4)

            ToolTip(widget, desc)
            self.vars[name] = (kind, var, widget)

        parent.grid_columnconfigure(1, weight=1)

    def _collect_value(self, name, kind, var, widget):
        if kind == "bool":
            return bool(var.get())
        if kind == "int":
            return int(var.get().strip() or 0)
        if kind in {"str", "path"}:
            return var.get().strip()
        if kind == "set":
            raw = widget.get("1.0", "end").strip()
            items = [x.strip() for x in raw.replace(",", "\n").split("\n") if x.strip()]
            return items
        if kind == "listtuple":
            raw = widget.get("1.0", "end").strip()
            if not raw:
                return []
            try:
                val = json.loads(raw)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
            pairs = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    provider, model = [s.strip() for s in line.split(":", 1)]
                elif "," in line:
                    provider, model = [s.strip() for s in line.split(",", 1)]
                else:
                    continue
                pairs.append([provider, model])
            return pairs
        return var.get().strip()

    def save_config(self):
        out = {}
        for name, (kind, var, widget) in self.vars.items():
            out[name] = self._collect_value(name, kind, var, widget)
        try:
            with CONFIG_PATH.open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            self.status.config(text=f"配置已保存：{CONFIG_PATH}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def reload_config(self):
        self._load_config()
        for name, (kind, var, widget) in self.vars.items():
            value = self.config_data.get(name, DEFAULTS.get(name))
            if kind == "bool":
                var.set(bool(value))
            elif kind in {"int", "str", "path"}:
                var.set(str(value) if value is not None else "")
            elif kind == "set":
                widget.delete("1.0", "end")
                if isinstance(value, list):
                    widget.insert("1.0", "\n".join(value))
                else:
                    widget.insert("1.0", "\n".join(list(value or [])))
            elif kind == "listtuple":
                widget.delete("1.0", "end")
                widget.insert("1.0", json.dumps(value, ensure_ascii=False, indent=2))
        self.status.config(text="配置已重新加载")

    def run_translator(self):
        try:
            subprocess.Popen([sys.executable, "pf2e_translator.py"], cwd=str(Path.cwd()))
            self.status.config(text="已启动翻译脚本")
        except Exception as e:
            messagebox.showerror("启动失败", str(e))

    def open_guide(self):
        if GUIDE_PATH.exists():
            try:
                if os.name == "nt":
                    os.startfile(str(GUIDE_PATH))
                else:
                    subprocess.Popen(["xdg-open", str(GUIDE_PATH)])
            except Exception as e:
                messagebox.showerror("打开失败", str(e))
        else:
            messagebox.showwarning("未找到", f"未找到 {GUIDE_PATH}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
