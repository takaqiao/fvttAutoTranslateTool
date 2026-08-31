import json
import os
import time
import re
import shutil
import hashlib
import concurrent.futures
import pandas as pd
from datetime import datetime
from threading import Lock
from tqdm import tqdm
from google import genai
from google.genai import types

# ================= 配置区域 =================

API_KEY = "AIzaSyA262R6ktQ3dS9Fsmiw4WOXQDcu87670Fk"

# 1. 核心文件配置
SOURCE_EN_JSON_PATH = "pf2e-beginner-box-en.json"
TARGET_JSON_PATH = "pf2e-beginner-box.adventures.json"

# 2. 术语表配置
GLOBAL_GLOSSARY_PATH = "术语译名对照表.csv" 
LOCAL_GLOSSARY_EXPORT_PATH = "术语表_本地提取.csv"

# 3. 性能与重试
TARGET_RPM = 950
MAX_WORKERS = 16   # 稳妥起见，保持 16
MAX_RETRIES = 5

# 4. [核心开关] 暴力防漏模式
BRUTE_FORCE_MODE = True 

# 5. 日志与缓存
REPORT_XLSX_PATH = "翻译审查报告.xlsx"
PROCESS_LOG_PATH = "运行日志.txt"
DROPPED_LOG_PATH = "术语丢弃日志.txt"
MISSED_LOG_PATH = "失败漏翻记录.txt"
HISTORY_FILE_PATH = "translation_history.json"
BACKUP_DIR = "backups"

# 目标字段
TARGET_KEYS = {"name", "description", "text", "label", "caption", "value", "unidentifiedName", "tokenName", "publicnotes", "publicNotes", "gm_notes", "gm_description", "header", "content", "items", "navName"}
SPECIAL_CONTAINERS = {"notes", "folders", "journal", "journals", "scenes", "actors", "items", "pages", "entries", "flags", "system"}

MODEL_ID = 'gemini-3-flash-preview' 
ENABLE_CODE_PROTECTION = True 

# ===========================================

client = genai.Client(api_key=API_KEY)
log_lock = Lock()

report_data = {"New": [], "Fixed": [], "Kept": []}
process_log_buffer = []
missed_log_buffer = [] 
history_cache = set()
new_history_entries = set()

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
]

def write_process_log(msg):
    with log_lock:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        process_log_buffer.append(f"[{timestamp}] {msg}")

def write_missed_log(path, text, reason):
    with log_lock:
        missed_log_buffer.append(f"【{reason}】Path: {path}\nText: {text[:50]}...\n{'-'*30}")

# === V28 新增：结构自愈手术刀 ===
def heal_structure(en_node, cn_node, path="root"):
    """
    递归对比中英结构，如果发现中文结构缺失 'journals' 层级但有内容，
    则自动进行结构迁移。
    """
    if not isinstance(en_node, dict) or not isinstance(cn_node, dict):
        return

    # 1. 针对 entries 下的特殊修复 (Menace Under Otari 等)
    if "journals" in en_node and "journals" not in cn_node:
        # 检查英文版 journals 下面是不是有一个和父节点同名的 Key
        # 例如: entries -> Menace Under Otari -> journals -> Menace Under Otari
        # 而中文版可能是: entries -> Menace Under Otari -> (直接是内容)
        
        # 我们尝试寻找那些“迷路”的数据
        migrated = False
        
        # 建立 journals 容器
        cn_node["journals"] = {}
        
        for journal_key, journal_val in en_node["journals"].items():
            # 如果中文版当前节点下，直接就有 pages, items 等数据
            # 我们假设这些数据其实属于 journals -> journal_key
            if "pages" in cn_node:
                print(f"🔧 [结构修复] 正在迁移 {path} 下的散落数据到 journals/{journal_key}...")
                
                # 创建深层结构
                cn_node["journals"][journal_key] = {
                    "name": cn_node.get("name", journal_key), # 尝试保留名字
                    "pages": cn_node["pages"]
                }
                # 清理旧位置的数据，防止重复
                del cn_node["pages"]
                migrated = True
            
            # 递归修复更深层
            if journal_key in cn_node["journals"]:
                 heal_structure(journal_val, cn_node["journals"][journal_key], f"{path}.journals.{journal_key}")

        if not migrated:
            # 如果没发生迁移，可能只是单纯缺了，不做操作
            pass

    # 2. 常规递归
    for k, v in en_node.items():
        if k in cn_node:
            heal_structure(v, cn_node[k], f"{path}.{k}")

# === 基础系统 ===
def backup_existing_files():
    if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = [TARGET_JSON_PATH, REPORT_XLSX_PATH, LOCAL_GLOSSARY_EXPORT_PATH, HISTORY_FILE_PATH]
    for fp in files:
        if os.path.exists(fp):
            try: shutil.copy2(fp, os.path.join(BACKUP_DIR, f"{timestamp}_{os.path.basename(fp)}.bak"))
            except: pass

def get_content_hash(en, cn):
    return hashlib.md5(f"{en or ''}::{cn or ''}".encode('utf-8')).hexdigest()

def load_history():
    if os.path.exists(HISTORY_FILE_PATH):
        try:
            with open(HISTORY_FILE_PATH, 'r', encoding='utf-8') as f: return set(json.load(f))
        except: return set()
    return set()

def save_history():
    try:
        with open(HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(history_cache.union(new_history_entries)), f)
    except: pass

class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.last_dispatch_time = 0
    def wait_for_slot(self):
        now = time.time()
        wait = self.last_dispatch_time + self.interval - now
        if wait > 0: time.sleep(wait)
        self.last_dispatch_time = time.time()

class CodeProtector:
    def __init__(self):
        self.patterns = [
            re.compile(r'(@[a-zA-Z0-9]+\[[^\]]*\])'),
            re.compile(r'(\[\[.*?\]\])'),
            re.compile(r'(<[^>]+>)'),
            re.compile(r'(&[a-zA-Z0-9#]+;)'),
        ]
    def mask(self, text):
        if not text: return text, {}
        ph, ctr = {}, 0
        def repl(m):
            nonlocal ctr
            k = f"__CODE_{ctr}__"
            ph[k] = m.group(1)
            ctr += 1
            return k
        for p in self.patterns: text = p.sub(repl, text)
        return text, ph
    def unmask(self, text, ph):
        if not text: return text
        for k, v in ph.items():
            text = text.replace(k, v)
            if k not in text: text = re.sub(k.replace('_', r'\s*_\s*'), v, text)
        return text

class GlossaryManager:
    def __init__(self, global_csv, local_csv=None):
        self.term_map = {} 
        self.sorted_keys = []
        self.load_glossary(global_csv)
        if local_csv and os.path.exists(local_csv): self.load_glossary(local_csv)
        self.sorted_keys = sorted(self.term_map.keys(), key=lambda x: len(x), reverse=True)
        print(f"术语库加载: {len(self.sorted_keys)} 条")

    def load_glossary(self, path):
        if not os.path.exists(path): return
        df = None
        for enc in ['utf-8', 'utf-8-sig', 'gbk']:
            try: df = pd.read_csv(path, encoding=enc); break
            except: continue
        if df is None: return
        for r in df.to_dict('records'):
            cn, en = str(r.get('Target','')).strip(), str(r.get('Source','')).strip()
            if cn and en and en.lower() != 'nan':
                flags = 0 if any(c.isupper() for c in en) else re.IGNORECASE
                self.term_map[en] = {"target": cn, "org": en, "re": re.compile(r'\b'+re.escape(en)+r'\b', flags)}

    def pre_inject_text(self, text, path_str):
        if not text: return text, []
        inj, ph, idx = [], {}, 0
        tokens = set(re.findall(r'[a-z]+', text.lower()))
        cands = [k for k in self.sorted_keys if (k.lower() in tokens) or (" " in k and k.lower() in text.lower())]
        
        for k in cands:
            d = self.term_map[k]
            matches = list(d["re"].finditer(text))
            if matches:
                def repl(m):
                    nonlocal idx
                    mt = m.group(0)
                    if mt == d["org"] or (d["org"].islower() and mt.istitle()):
                        inj.append((d["org"], d["target"]))
                        k = f"__Tm_{idx}__"
                        ph[k] = f"⟪{d['target']}|原文:{mt}⟫"
                        idx += 1
                        return k
                    return mt
                text = d["re"].sub(repl, text)
        
        for k, v in ph.items(): text = text.replace(k, v)
        return text, inj

def smart_format_bilingual(cn, en):
    if not cn: return en
    cn = re.sub(r'⟪(.*?)\|原文:.*?⟫', r'\1', cn)
    clean_en = re.sub(r'[\s\W]', '', en).lower()
    clean_cn = re.sub(r'[\s\W]', '', cn).lower()
    if clean_en in clean_cn: return cn
    sep = "<br><br><hr><b>原文:</b><br>" if (len(en) > 80 or "<p>" in en) else " "
    return f"{cn}{sep}{en}"

def extract_local_glossary(en_data, cn_data, output_path):
    print("正在扫描本地术语...")
    # (省略具体实现，保持原样)

def process_single_item(task_type, en_text, cn_draft, glossary_mgr, path_str):
    if not en_text or len(en_text) < 2: return en_text, None
    if not re.search(r'[a-zA-Z]', en_text): return en_text, None
    
    prot = CodeProtector()
    masked, code_ph = prot.mask(en_text)
    injected, terms = glossary_mgr.pre_inject_text(masked, path_str)
    
    clean_draft_txt = cn_draft
    if cn_draft and "<hr>" in cn_draft: clean_draft_txt = cn_draft.split("<hr>")[0].strip()
    
    sys_prompt = "You are a professional Pathfinder 2e translator. Output ONLY Chinese. Keep HTML/Codes."
    if task_type == "AUDIT":
        prompt = f"Original:\n```\n{injected}\n```\nDraft:\n```\n{clean_draft_txt}\n```\nTask: Review draft. If correct, output it. If wrong, correct it."
    else:
        prompt = f"Translate:\n```\n{injected}\n```"

    for attempt in range(MAX_RETRIES):
        try:
            res = client.models.generate_content(
                model=MODEL_ID,
                contents=f"{sys_prompt}\n{prompt}",
                config=types.GenerateContentConfig(temperature=0.1, safety_settings=SAFETY_SETTINGS)
            )
            if not res.text: raise ValueError("Empty")
            
            final = re.sub(r'⟪(.*?)\|原文:.*?⟫', r'\1', prot.unmask(res.text.strip(), code_ph))
            final = re.sub(r'^```.*?(\n|$)', '', final).replace('```', '').strip()

            status = "New"
            if task_type == "AUDIT":
                if re.sub(r'\s','',final) == re.sub(r'\s','',clean_draft_txt): status = "Kept"
                else: status = "Fixed"
            
            with log_lock:
                if status in report_data:
                    report_data[status].append({"Path":path_str, "Original":en_text, "Trans":final})

            return smart_format_bilingual(final, en_text), status

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                write_process_log(f"Fail {path_str}: {e}")
                write_missed_log(path_str, en_text, f"Max Retries: {e}")
                return smart_format_bilingual(cn_draft, en_text) if cn_draft else en_text, None
            time.sleep(2 * (attempt + 1))
            
    return en_text, None

def collect_tasks(en_data, cn_data, path_str="root"):
    tasks = []
    
    def get_cn(d, k):
        if isinstance(d, dict): return d.get(k)
        if isinstance(d, list) and isinstance(k, int) and k < len(d): return d[k]
        return None

    if isinstance(en_data, dict): iter_items = en_data.items()
    elif isinstance(en_data, list): iter_items = enumerate(en_data)
    else: return []

    for k, v in iter_items:
        cur_path = f"{path_str}.{k}" if isinstance(en_data, dict) else f"{path_str}[{k}]"
        cn_v = get_cn(cn_data, k)
        
        should_translate = False
        if isinstance(v, str) and len(v) > 1:
            has_letters = bool(re.search(r'[a-zA-Z]', v))
            if has_letters:
                is_target_key = False
                if isinstance(en_data, dict) and k in TARGET_KEYS: is_target_key = True
                elif any(c in cur_path.split('.') for c in SPECIAL_CONTAINERS): is_target_key = True
                
                is_file = v.lower().endswith(('.png', '.webp', '.jpg', '.mp3', '.ogg', '.m4a', '.webm'))
                has_space = " " in v
                
                if BRUTE_FORCE_MODE:
                    if not is_file and (is_target_key or has_space): should_translate = True
                else:
                    if is_target_key: should_translate = True

        if should_translate:
            if cn_v and get_content_hash(v, cn_v) in history_cache: continue
            tt = 'AUDIT' if (cn_v and isinstance(cn_v, str) and len(cn_v) > 0 and cn_v != v) else 'NEW'
            tasks.append({'type': tt, 'ref': en_data, 'k': k, 'en_v': v, 'cn_v': cn_v if tt=='AUDIT' else None, 'path': cur_path})
            
        elif isinstance(v, (dict, list)):
            new_cn = cn_v if isinstance(cn_v, (dict, list)) else {}
            tasks.extend(collect_tasks(v, new_cn, cur_path))
            
    return tasks

def main():
    print(f"PF2e 汉化脚本 V28 (结构自愈修复版)")
    
    if not os.path.exists(SOURCE_EN_JSON_PATH):
        print("❌ 错误：找不到基准英文文件。")
        return

    backup_existing_files()
    
    global history_cache
    history_cache = load_history()
    print(f"🧠 已加载缓存: {len(history_cache)}")

    with open(SOURCE_EN_JSON_PATH, 'r', encoding='utf-8-sig') as f: en_data = json.load(f)
    cn_data = {}
    if os.path.exists(TARGET_JSON_PATH):
        try:
            with open(TARGET_JSON_PATH, 'r', encoding='utf-8-sig') as f: cn_data = json.load(f)
            
            # --- V28 关键步骤：修复结构 ---
            print("🏥 正在检查并修复中英文件结构差异...")
            heal_structure(en_data, cn_data)
            # ---------------------------
            
            extract_local_glossary(en_data, cn_data, LOCAL_GLOSSARY_EXPORT_PATH)
        except Exception as e: print(f"加载目标文件出错: {e}")
    
    glossary = GlossaryManager(GLOBAL_GLOSSARY_PATH, LOCAL_GLOSSARY_EXPORT_PATH)
    all_tasks = collect_tasks(en_data, cn_data)
    print(f"待处理任务: {len(all_tasks)}")
    
    if not all_tasks: 
        print("🎉 没有需要更新的内容！")
        return

    rl = RateLimiter(TARGET_RPM)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        fut_map = {}
        for t in tqdm(all_tasks, desc="分发"):
            rl.wait_for_slot()
            fut_map[exe.submit(process_single_item, t['type'], t['en_v'], t['cn_v'], glossary, t['path'])] = t
        
        for f in tqdm(concurrent.futures.as_completed(fut_map), total=len(all_tasks), desc="回收"):
            task = fut_map[f]
            try:
                res, st = f.result()
                task['ref'][task['k']] = res
                if st == "Kept":
                    with log_lock: new_history_entries.add(get_content_hash(task['en_v'], res))
            except: pass

    with open(TARGET_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(en_data, f, ensure_ascii=False, indent=2)
    
    if missed_log_buffer:
        with open(MISSED_LOG_PATH, 'w', encoding='utf-8') as f: f.write("\n".join(missed_log_buffer))
        print(f"⚠️ 警告：有 {len(missed_log_buffer)} 条内容漏翻")

    while True:
        try:
            with pd.ExcelWriter(REPORT_XLSX_PATH) as w:
                pd.DataFrame(report_data["New"]).to_excel(w, sheet_name="New", index=False)
                pd.DataFrame(report_data["Fixed"]).to_excel(w, sheet_name="Fixed", index=False)
                pd.DataFrame(report_data["Kept"]).to_excel(w, sheet_name="Kept", index=False)
            break
        except PermissionError: input(f"❌ 请关闭 {REPORT_XLSX_PATH} 后回车...")
        except: break

    save_history()
    print("🎉 完成")

if __name__ == "__main__":
    main()