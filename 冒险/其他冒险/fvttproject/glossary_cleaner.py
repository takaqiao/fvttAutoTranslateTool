import pandas as pd
import re
import os

# ================= 配置 =================
RAW_CSV_PATH = "术语译名对照表.csv"       # 原始文件
CLEAN_CSV_PATH = "术语表_已清洗.csv"      # 输出文件
CONFLICT_CSV_PATH = "术语表_冲突项.csv"   # 冲突文件
TRASH_CSV_PATH = "术语表_已剔除.csv"      # 垃圾桶

# === 黑名单 (Stopwords) ===
STOP_WORDS = {
    'a', 'an', 'the', 'of', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 
    'by', 'with', 'as', 'it', 'this', 'that', 'my', 'your', 'his', 'her', 'its', 
    'be', 'do', 'go', 'so', 'or', 'if', 'no', 'up', 'me', 'we', 'us', 'row', 'pin', 
    'can', 'may', 'will', 'but', 'not', 'and', 'all', 'any', 'one', 'two', 'six', 'ten'
}

# === 白名单 (Whitelist) ===
KEEP_WORDS = {
    'hp', 'HP', 'ac', 'AC', 'dc', 'DC', 'xp', 'XP', 'cp', 'sp', 'gp', 'pp',
    'cn', 'CN', 'ln', 'LN', 'ng', 'NG', 'ne', 'NE', 'n', 'N',
    'str', 'dex', 'con', 'int', 'wis', 'cha',
    'eox', 'xin', 'geb', 'nex', 'mut', 'qat', 'ir'
}

EXCEL_ERRORS = {'#NAME?', '#REF!', '#VALUE!', '#DIV/0!', '#NUM!', '#N/A'}
# =======================================

def clean_glossary_v8():
    print(f"=== 术语表清洗工具 V8 (UUID/代码剔除版) ===")
    print(f"正在读取: {RAW_CSV_PATH} ...")
    
    # 1. 读取 CSV
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(RAW_CSV_PATH, encoding=enc, header=None, skip_blank_lines=True)
            print(f"-> 成功识别编码: {enc}")
            break
        except: continue
        
    if df is None:
        print("【错误】无法读取文件。")
        return

    if len(df.columns) < 2:
        print("【错误】CSV 列数不足。")
        return
    df = df.rename(columns={0: 'Source', 1: 'Target'}) 
    df = df[['Source', 'Target']] 

    original_count = len(df)
    
    # 2. 格式清洗
    print("正在执行格式清洗...")
    df['Source'] = df['Source'].astype(str).str.strip()
    df['Target'] = df['Target'].astype(str).str.strip()
    
    # 剔除 Excel 错误
    df = df[~df['Source'].isin(EXCEL_ERRORS)]
    df = df[~df['Target'].isin(EXCEL_ERRORS)]
    
    # 剔除空内容
    df = df[df['Source'].str.len() > 0]
    df = df[df['Target'].str.len() > 0]
    df = df[df['Source'].str.lower() != 'nan']
    df = df[df['Target'].str.lower() != 'nan']
    df = df[df['Source'].str.lower() != 'none']
    
    print(f"-> 格式清理后剩余: {len(df)}")

    # 3. 基础去重
    df = df.drop_duplicates(subset=['Source', 'Target'])

    # 4. 智能过滤
    print("正在执行智能过滤 (含UUID去除)...")
    valid_rows = []
    trash_rows = []

    for index, row in df.iterrows():
        src = row['Source']
        tgt = row['Target']
        src_lower = src.lower()
        tgt_lower = tgt.lower()
        
        is_keep = True
        reason = ""

        # A. 白名单 (最高优先级)
        if src in KEEP_WORDS or src_lower in KEEP_WORDS:
            is_keep = True
            
        # === B. 代码/UUID 标签剔除 (新增逻辑) ===
        # 如果原文或译文里包含 @UUID, @Compendium, @Item... 删掉
        elif '@uuid' in src_lower or '@compendium' in src_lower:
            is_keep = False
            reason = "系统代码残留(@UUID)"
        elif '@uuid' in tgt_lower or '@compendium' in tgt_lower:
            is_keep = False
            reason = "系统代码残留(@UUID)"
            
        # C. 长度检查
        elif len(src) < 2:
            is_keep = False
            reason = "长度不足"
            
        # D. 黑名单检查
        elif src in STOP_WORDS: # 小写
            is_keep = False
            reason = "黑名单(小写)"
        elif src_lower in STOP_WORDS and src[0].isupper(): # 大写放行
            is_keep = True
            
        # E. 纯数字
        elif src.replace(' ', '').isnumeric():
            is_keep = False
            reason = "纯数字"
        
        # F. 页码残留
        elif src_lower.startswith('page ') and src[5:].strip().isnumeric():
             is_keep = False
             reason = "页码残留"

        if is_keep:
            valid_rows.append(row)
        else:
            row['Reason'] = reason
            trash_rows.append(row)

    clean_df = pd.DataFrame(valid_rows)
    trash_df = pd.DataFrame(trash_rows)

    # 5. 冲突检测 (完全严格匹配)
    print("正在检测译名冲突...")
    if not clean_df.empty:
        duplicates = clean_df[clean_df.duplicated('Source', keep=False)]
        
        conflict_groups = duplicates.groupby('Source')['Target'].nunique()
        conflict_keys = conflict_groups[conflict_groups > 1].index
        
        conflicts_df = clean_df[clean_df['Source'].isin(conflict_keys)].sort_values('Source')
        
        if not conflicts_df.empty:
            print(f"⚠️ 发现 {len(conflict_keys)} 组译名冲突！已导出至 {CONFLICT_CSV_PATH}")
            conflicts_df.to_csv(CONFLICT_CSV_PATH, index=False, encoding='utf-8-sig')

        # 生成最终表
        final_df = clean_df.drop_duplicates(subset=['Source'])
        final_df.to_csv(CLEAN_CSV_PATH, index=False, encoding='utf-8-sig')
        
        if not trash_df.empty:
            trash_df.to_csv(TRASH_CSV_PATH, index=False, encoding='utf-8-sig')

        print(f"\n=== V8 清洗完成 ===")
        print(f"原始数据: {original_count}")
        print(f"有效数据: {len(final_df)}")
        print(f"剔除垃圾: {len(trash_df)}")
        print(f"-> 干净表格: {CLEAN_CSV_PATH}")
    else:
        print("【错误】清洗后没有剩余数据。")

if __name__ == "__main__":
    clean_glossary_v8()