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

API_KEY = "YOUR_API_KEY_HERE"

# 1. 核心文件配置
SOURCE_EN_JSON_PATH = "pf2e-beginner-box.adventures.json"
TARGET_JSON_PATH = "pf2e-beginner-box_CN.json"

# 2. 术语表配置
GLOBAL_GLOSSARY_PATH = "术语译名对照表.csv" 
LOCAL_GLOSSARY_EXPORT_PATH = "术语表_本地提取.csv"

# 3. 性能与重试
TARGET_RPM = 950
MAX_WORKERS = 64
MAX_RETRIES = 5

# 4. 日志与缓存
REPORT_XLSX_PATH = "翻译审查报告.xlsx"
PROCESS_LOG_PATH = "运行日志.txt"
DROPPED_LOG_PATH = "术语丢弃日志.txt"
HISTORY_FILE_PATH = "translation_history.json" # 缓存文件
BACKUP_DIR = "backups"

# 目标字段
TARGET_KEYS = {"name", "description", "text", "label", "caption", "value", "unidentifiedName", "tokenName", "publicnotes", "publicNotes"}
SPECIAL_CONTAINERS = {"notes", "folders"}

MODEL_ID = 'gemini-3-flash-preview' 
ENABLE_CODE_PROTECTION = True 

# ===========================================

client = genai.Client(api_key=API_KEY)
log_lock = Lock()

report_data = {"New": [], "Fixed": [], "Kept": []}
process_log_buffer = []
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

# === 全量备份系统 (V26 核心) ===

def backup_existing_files():
    """备份所有关键文件，建立版本快照"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 需要备份的文件清单
    files_to_backup = [
        TARGET_JSON_PATH,           # 半成品 JSON
        REPORT_XLSX_PATH,           # 上一次的报告
        LOCAL_GLOSSARY_EXPORT_PATH, # 上一次提取的术语
        HISTORY_FILE_PATH           # 缓存文件
    ]
    
    backup_count = 0
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            # 为了方便排序，统一命名格式: 时间戳_文件名.bak
            backup_name = f"{timestamp}_{filename}.bak"
            backup_path = os.path.join(BACKUP_DIR, backup_name)
            
            try:
                shutil.copy2(file_path, backup_path)
                backup_count += 1
            except Exception as e:
                print(f"❌ 备份失败 {filename}: {e}")
                
    if backup_count > 0:
        print(f"📦 已建立全量快照: {timestamp} (备份了 {backup_count} 个文件)")

# === 缓存系统 ===

def get_content_hash(en_text, cn_text):
    if not en_text: en_text = ""
    if not cn_text: cn_text = ""
    raw = f"{en_text}::{cn_text}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

def load_history():
    if os.path.exists(HISTORY_FILE_PATH):
        try:
            with open(HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data)
        except: return set()
    return set()

def save_history():
    final_history = history_cache.union(new_history_entries)
    try:
        with open(HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(final_history), f)
        print(f"💾 缓存更新: {len(final_history)} 条记录")
    except Exception as e:
        print(f"❌ 缓存保存失败: {e}")

# === 基础工具 ===

class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.last_dispatch_time = 0
    def wait_for_slot(self):
        now = time.time()
        next_slot = self.last_dispatch_time + self.interval
        wait_time = next_slot - now
        if wait_time > 0: time.sleep(wait_time)
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
        placeholders = {}
        counter = 0
        masked_text = text
        for pattern in self.patterns:
            def replace_func(match):
                nonlocal counter
                code_segment = match.group(1)
                key = f"__CODE_{counter}__"
                placeholders[key] = code_segment
                counter += 1
                return key
            masked_text = pattern.sub(replace_func, masked_text)
        return masked_text, placeholders
    def unmask(self, text, placeholders):
        if not text: return text
        result = text
        for key, val in placeholders.items():
            result = result.replace(key, val)
            if key not in result:
                result = re.sub(key.replace('_', r'\s*_\s*'), val, result)
        return result

class GlossaryManager:
    def __init__(self, global_csv, local_csv=None):
        self.term_map = {} 
        self.sorted_keys = []
        self.dropped_logs = []
        count_global = self.load_glossary(global_csv, "全局")
        if local_csv and os.path.exists(local_csv):
            self.load_glossary(local_csv, "本地")
        self.sorted_keys = sorted(self.term_map.keys(), key=lambda x: len(x), reverse=True)
        if self.dropped_logs:
            with open(DROPPED_LOG_PATH, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.dropped_logs))
        print(f"术语库加载完毕: {len(self.sorted_keys)} 条有效术语")

    def load_glossary(self, path, label):
        if not os.path.exists(path): return 0
        encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except: continue
        if df is None: return 0
        records = df.to_dict('records')
        count = 0
        for row in records:
            cn = str(row.get('Target', row.get('target', row.get('0', '')))).strip()
            en = str(row.get('Source', row.get('source', row.get('1', '')))).strip()
            if cn and en and en.lower() != 'nan':
                if en in self.term_map:
                    old_cn = self.term_map[en]['target']
                    if old_cn != cn:
                        self.dropped_logs.append(f"[{label}覆盖] '{en}': '{old_cn}' -> '{cn}'")
                flags = 0 if any(c.isupper() for c in en) else re.IGNORECASE
                self.term_map[en] = {
                    "target": cn, 
                    "source_original": en,
                    "regex": re.compile(r'\b' + re.escape(en) + r'\b', flags)
                }
                count += 1
        return count

    def pre_inject_text(self, text: str, json_path: str):
        if not text: return text, []
        injected_terms = [] 
        temp_text = text
        placeholders = {}
        placeholder_idx = 0
        text_lower = temp_text.lower()
        text_tokens = set(re.findall(r'[a-z]+', text_lower))
        candidates = []
        for k in self.sorted_keys:
            k_lower = k.lower()
            if " " not in k_lower:
                if k_lower in text_tokens: candidates.append(k)
            else:
                if k_lower in text_lower: candidates.append(k)
        for k in candidates:
            data = self.term_map[k]
            pattern = data["regex"]
            glossary_term = data["source_original"]
            matches = list(pattern.finditer(temp_text))
            if matches:
                def replace_func(match):
                    nonlocal placeholder_idx
                    matched_text = match.group(0)
                    should_inject = False
                    if matched_text == glossary_term: should_inject = True
                    elif glossary_term.islower() and matched_text.istitle(): should_inject = True
                    if should_inject:
                        injected_terms.append((glossary_term, data["target"]))
                        key = f"__TERM_{placeholder_idx}__"
                        injection_str = f"⟪{data['target']}|原文:{matched_text}⟫"
                        placeholders[key] = injection_str
                        placeholder_idx += 1
                        return key
                    return matched_text
                temp_text = pattern.sub(replace_func, temp_text)
        final_text = temp_text
        for key, val in placeholders.items():
            final_text = final_text.replace(key, val)
        return final_text, injected_terms

# === 文本处理工具 ===

def cleanup_injection_tags(text):
    if not text: return ""
    return re.sub(r'⟪(.*?)\|原文:.*?⟫', r'\1', text)

def clean_for_ai_audit(cn_text):
    if not cn_text: return ""
    if "<hr>" in cn_text: return cn_text.split("<hr>")[0].strip()
    if "<hr />" in cn_text: return cn_text.split("<hr />")[0].strip()
    if "原文:" in cn_text: return cn_text.split("原文:")[0].strip()
    return cn_text

def smart_format_bilingual(final_cn, original_en):
    if not final_cn: return original_en
    final_cn = cleanup_injection_tags(final_cn)
    cn = final_cn.strip().strip('"').strip("'")
    en = original_en.strip()
    en_clean = re.sub(r'[\s\W]', '', en).lower()
    cn_clean = re.sub(r'[\s\W]', '', cn).lower()
    if en_clean in cn_clean and len(en_clean) > 0: return cn 
    if "<p>" in en or "<br>" in en or len(en) > 80:
        return f"{cn}<br><br><hr><b>原文:</b><br>{en}"
    else:
        return f"{cn} {en}"

def strip_english_part(text, source_en):
    if not text: return ""
    if source_en and source_en in text:
        text = text.replace(source_en, "").strip()
    text = clean_for_ai_audit(text)
    match = re.search(r'[\u4e00-\u9fff]', text)
    if match:
        text = re.sub(r'\s*\(?[a-zA-Z0-9\s\-\']+\)?$', '', text).strip()
    return text

def extract_local_glossary(en_data, cn_data, output_path):
    print("正在扫描本地术语...")
    extracted = []
    def traverse(en_node, cn_node):
        if isinstance(en_node, dict) and isinstance(cn_node, dict):
            for k, v in en_node.items():
                if k in cn_node: traverse(v, cn_node[k])
        elif isinstance(en_node, list) and isinstance(cn_node, list):
            for i in range(min(len(en_node), len(cn_node))):
                traverse(en_node[i], cn_node[i])
        elif isinstance(en_node, str) and isinstance(cn_node, str):
            if len(en_node) < 60 and len(cn_node) > 0 and en_node != cn_node:
                clean_cn = strip_english_part(cn_node, en_node)
                if clean_cn and re.search(r'[\u4e00-\u9fff]', clean_cn):
                    extracted.append({'Source': en_node, 'Target': clean_cn})
    traverse(en_data, cn_data)
    while True:
        try:
            if extracted:
                df = pd.DataFrame(extracted).drop_duplicates(subset=['Source'])
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"已提取本地术语: {len(df)} 条")
            break
        except PermissionError:
            print(f"\n❌ 错误：文件 '{output_path}' 被占用。请关闭Excel后按回车...")
            input()

def clean_response_text(text):
    if not text: return ""
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    text = re.sub(r'^(Here is|Below is|以下是).*?(\n|$)', '', text, flags=re.IGNORECASE).strip()
    return text.strip()

protector = CodeProtector()

def process_single_item(task_type, en_text, cn_draft, glossary_mgr, path_str):
    if not en_text or len(en_text) < 2 or en_text.isdigit(): return en_text, None
    
    masked_text, code_placeholders = protector.mask(en_text)
    injected_text, injected_terms_list = glossary_mgr.pre_inject_text(masked_text, path_str)
    clean_draft = clean_for_ai_audit(cn_draft) if cn_draft else ""

    sys_header = "You are a professional Pathfinder 2e translator."
    sys_rules = "CRITICAL RULES:\n1. Output ONLY the translated Chinese text.\n2. Do NOT append the original English text at the end (I will handle it).\n3. Keep HTML tags/codes unchanged."
    tick_block = "```" 
    
    if task_type == "AUDIT":
        user_prompt = "Original:\n" + tick_block + f"\n{injected_text}\n" + tick_block + "\n\nExisting Draft:\n" + tick_block + f"\n{clean_draft}\n" + tick_block + "\n\nTask: Review the draft. If it is accurate, output it AS IS. If it is wrong, output a corrected Chinese translation."
    else:
        user_prompt = "Translate to Chinese:\n" + tick_block + f"\n{injected_text}\n" + tick_block

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=sys_header + "\n" + sys_rules + "\n" + user_prompt,
                config=types.GenerateContentConfig(temperature=0.1, safety_settings=SAFETY_SETTINGS)
            )
            
            if not response.text: raise ValueError("Empty response")
            trans = clean_response_text(response.text)
            
            final_trans = protector.unmask(trans, code_placeholders)
            final_trans = cleanup_injection_tags(final_trans)

            status = "New"
            if task_type == "AUDIT":
                clean_res = re.sub(r'\s', '', final_trans)
                clean_old = re.sub(r'\s', '', clean_draft)
                if clean_res == clean_old: 
                    status = "Kept"
                else: 
                    status = "Fixed"

            log_report(status, path_str, en_text, final_trans, injected_terms_list)
            
            return smart_format_bilingual(final_trans, en_text), status

        except Exception as e:
            wait_time = 2 * (attempt + 1) 
            if attempt == MAX_RETRIES - 1:
                write_process_log(f"Fail {path_str}: {e}")
                return smart_format_bilingual(cn_draft, en_text) if cn_draft else en_text, None
            else:
                time.sleep(wait_time) 

    return en_text, None

def log_report(status, path, original, translated, injected_terms):
    term_str = " | ".join([f"{e}->{c}" for e, c in injected_terms])
    row = {
        "JSON Path": path,
        "Involved Terms": term_str,
        "Original": original,
        "Translation": translated
    }
    with log_lock:
        if status in report_data:
            report_data[status].append(row)

def collect_tasks(en_data, cn_data, path_str="root"):
    tasks = []
    
    def get_cn_val(data, key):
        if isinstance(data, dict): return data.get(key)
        if isinstance(data, list) and isinstance(key, int) and key < len(data): return data[key]
        return None

    if isinstance(en_data, dict):
        iterator = en_data.items()
    elif isinstance(en_data, list):
        iterator = enumerate(en_data)
    else:
        return []

    for k, v in iterator:
        current_path = f"{path_str}.{k}" if isinstance(en_data, dict) else f"{path_str}[{k}]"
        
        cn_val = get_cn_val(cn_data, k)
        
        is_target_field = False
        if isinstance(en_data, dict) and k in TARGET_KEYS: is_target_field = True
        elif any(c in path_str.split('.') for c in SPECIAL_CONTAINERS): is_target_field = True

        if isinstance(v, str) and len(v) > 1 and is_target_field:
            if cn_val:
                content_hash = get_content_hash(v, cn_val)
                if content_hash in history_cache:
                    continue
            
            task_type = 'AUDIT' if (cn_val and isinstance(cn_val, str) and len(cn_val) > 0 and cn_val != v) else 'NEW'
            tasks.append({
                'type': task_type,
                'ref': en_data,
                'k': k,
                'en_v': v,
                'cn_v': cn_val if task_type == 'AUDIT' else None,
                'path': current_path
            })
        elif isinstance(v, (dict, list)):
            new_cn = cn_val if isinstance(cn_val, (dict, list)) else {}
            tasks.extend(collect_tasks(v, new_cn, current_path))
            
    return tasks

def save_logs():
    print("\n正在生成 Excel 报告...")
    try:
        with pd.ExcelWriter(REPORT_XLSX_PATH) as writer:
            pd.DataFrame(report_data["New"]).to_excel(writer, sheet_name="新译(New)", index=False)
            pd.DataFrame(report_data["Fixed"]).to_excel(writer, sheet_name="修正(Fixed)", index=False)
            pd.DataFrame(report_data["Kept"]).to_excel(writer, sheet_name="保留(Kept)", index=False)
        print(f"✅ 报告已保存: {REPORT_XLSX_PATH}")
    except: pass

    with open(PROCESS_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(process_log_buffer))
        
    save_history()

def main():
    print(f"PF2e 汉化脚本 V26 (全量快照备份版)")
    
    if not os.path.exists(SOURCE_EN_JSON_PATH):
        print("❌ 错误：找不到基准英文文件。")
        return

    # 1. 执行全量备份
    backup_existing_files()
    
    global history_cache
    history_cache = load_history()
    print(f"🧠 已加载历史缓存: {len(history_cache)} 条记录")

    print("读取文件...")
    with open(SOURCE_EN_JSON_PATH, 'r', encoding='utf-8-sig') as f:
        en_data = json.load(f)
    
    cn_data = {}
    if os.path.exists(TARGET_JSON_PATH):
        print(f"🔄 加载上次成果: {TARGET_JSON_PATH}")
        try:
            with open(TARGET_JSON_PATH, 'r', encoding='utf-8-sig') as f:
                cn_data = json.load(f)
            extract_local_glossary(en_data, cn_data, LOCAL_GLOSSARY_EXPORT_PATH)
        except Exception as e:
            print(f"⚠️ 读取目标文件失败 ({e})，将执行全量新译")
    else:
        print("✨ 无历史文件，将执行全量新译")
    
    glossary = GlossaryManager(GLOBAL_GLOSSARY_PATH, LOCAL_GLOSSARY_EXPORT_PATH)
    
    print("构建任务队列 (自动跳过已验证条目)...")
    all_tasks = collect_tasks(en_data, cn_data)
    print(f"当前待处理任务数: {len(all_tasks)}")

    if not all_tasks:
        print("🎉 没有需要更新的内容！所有条目均已通过验证。")
        return

    rate_limiter = RateLimiter(TARGET_RPM)

    print("🚀 引擎启动...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {}
        for t in tqdm(all_tasks, desc="分发任务"):
            rate_limiter.wait_for_slot()
            future = executor.submit(process_single_item, t['type'], t['en_v'], t['cn_v'], glossary, t['path'])
            future_to_task[future] = t
        
        print("\n⏳ 等待回收结果...")
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(all_tasks), desc="回收结果"):
            task = future_to_task[future]
            try:
                result_text, status = future.result()
                task['ref'][task['k']] = result_text
                
                if status == "Kept":
                    h = get_content_hash(task['en_v'], result_text)
                    with log_lock:
                        new_history_entries.add(h)
                        
            except Exception as e: pass

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(en_data, f, ensure_ascii=False, indent=2)
    
    save_logs()
    print("🎉 全部完成。")

if __name__ == "__main__":
    main()