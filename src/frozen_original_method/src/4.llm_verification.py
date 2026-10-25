# llm_verification.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os
import re
import time
from tqdm.auto import tqdm
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from config import (
    RULE_ENGINE_OUTPUT_PATH,       
    LLM_VERIFICATION_OUTPUT,       
    LLM_CONFIDENCE_THRESHOLD,      
    BATCH_SAVE_INTERVAL,           
    LLM_TAXONOMY,
    API_KEY,
    BASE_URL,
    CLOUD_MODEL_NAME
)
from utils import safe_str

import httpx 
custom_client = httpx.Client(trust_env=False, verify=False)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=custom_client)

def build_taxonomy_str(taxonomy):
    s = "请严格从以下【主类】和【子类】中选择（不要创造新词）：\n"
    if "__STRONG_KEYWORDS__" in taxonomy:
        strong_kws = taxonomy["__STRONG_KEYWORDS__"]
        examples = [kw for kw in strong_kws if any('\u4e00' <= char <= '\u9fff' for char in kw)][:40] 
        s += f"- **强特征匹配 (高优先级)**: 若名称包含 [{', '.join(examples)}...] 等词，必须优先归类。\n\n"

    for main, subs in taxonomy.items():
        if main == "__STRONG_KEYWORDS__": continue
        s += f"- 主类: {main}\n  可选子类: {', '.join(subs)}\n"
    return s

TAXONOMY_PROMPT_STR = build_taxonomy_str(LLM_TAXONOMY)

def filter_data_for_llm(df):
    
    def is_really_empty(text):
        t = str(text).strip().lower()
        return t in ['none', 'nan', '', '无', 'null', 'undefined']

    name_series = df['OFFICIALBUILDINGNAMETC'].fillna('') + ' ' + df['BUILDINGNAMETC'].fillna('')
    is_meaningful = name_series.apply(lambda x: len(re.sub(r'[0-9\s\W_]+', '', x)) > 1)
    
    has_name = df['OFFICIALBUILDINGNAMETC'].apply(lambda x: not is_really_empty(x))
    has_poi = df['Matched_POIs'].apply(lambda x: not is_really_empty(x))
    
    cond_unknown = df['Final_Main_Class'].isin(['未知类别', 'Unknown', '待评估', np.nan])
    cond_misjudge = (df['Final_Main_Class'] == '非评估类别') & (has_name | has_poi)
    cond_vague = (df['Final_Main_Class'] == '商业类别') & (df['Final_Sub_Class'].str.contains('其他', na=False))
    cond_conflict = df['Is_Conflicted'].isin([True, 'True', 'true'])
    cond_not_processed = df['LLM_Confidence'].isna()

    candidates = df[
        is_meaningful & 
        (has_name | has_poi) & 
        (cond_unknown | cond_misjudge | cond_vague | cond_conflict) &
        cond_not_processed
    ].copy()
    
    return candidates

def generate_prompt(row):
    category_mapping = """
    LandsD Category Code Reference:
    - 1: Legal Private Buildings
    - 2: New Territories Small Houses (Village Houses/Ding Houses)
    - 3: Housing Authority Buildings (Public Housing/HOS)
    - 4: Other Government Buildings (Offices, Schools, Hospitals, etc.)
    - 5: Miscellaneous Structures (Temporary/Open structures)
    - 9: Category is not assigned
    """
    
    name_c = safe_str(row.get('OFFICIALBUILDINGNAMETC') or row.get('BUILDINGNAMETC'))
    addr = safe_str(row.get('BDBIAR_ADDRESS_C') or row.get('ADDRESS_E'))
    
    struct_type = safe_str(row.get('BUILDINGSTRUCTURETYPE', 'Unknown'))
    
    prompt = f"""Task: Assign building category labels based on the provided information.

[Input Data]
- Name: {name_c}
- Address: {addr}
- Structure Type: {struct_type} (T=Tower, P=Podium, U=Underground)
- Spatial Features: {safe_str(row.get('Matched_POIs', 'No record'))}
- Floors: {safe_str(row.get('NUMABOVEGROUNDSTOREYS', 'Unknown'))}
- LandsD Category Code: {safe_str(row.get('CATEGORY'))}

{category_mapping}

[Classification Rules]
1. FUNCTIONAL OVERRIDE (CRITICAL): If the name explicitly contains "Car Park" (停車場), "Market" (街市), "Substation" (變電站), "Pump House" (泵房), or "Public Toilet" (公廁), classify as "商业类别" -> "其他商业（Other Commercial）" even in Residential Estates.
2. MIXED-USE (ESSENTIAL): If Structure Type is "T" (Tower) AND Floors > 1 AND Spatial Features contain commercial labels like [restaurant], [shop], [bank], [clinic], or [school], you MUST label as "混合用途". 
   - Sub-class format: Combine the residential and commercial types (e.g., "私人房屋（Private Housing）-零售（Retail）").
3. RELIGIOUS BUILDINGS: Churches (教堂), Temples (廟宇), Mosques, etc., MUST be classified as "商业类别" -> "其他商业（Other Commercial）".
4. HOUSING NAMES: "邨/公屋" belong to "公共房屋"; "苑/大厦/阁/台" belong to "私人房屋".

[Taxonomy]
{TAXONOMY_PROMPT_STR}

[Output Requirements]
1. Return strictly JSON: reasoning, main_class, sub_class, confidence.
2. CRITICAL: "main_class" MUST be one of: ['住宅类别', '商业类别', '工业类别', '非评估类别', '混合用途'].
3. CRITICAL: "sub_class" MUST strictly follow the names in [Taxonomy]. For "混合用途", combine two valid sub-classes with a hyphen.
4. CRITICAL: "confidence" MUST be a float number between 0.00 and 1.00 (e.g., 0.85, 0.95). DO NOT use strings like "High" or "Low".
5. DO NOT invent new categories.
6. No conversational text.
"""
    return prompt

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return None

def call_llm_api(prompt, retries=3):
    system_msg = (
        "You are a professional building classifier. "
        "Output valid JSON. "
        "Strictly adhere to the hierarchy: main_class must be the parent, and sub_class must be a child from the defined Taxonomy."
    )
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=CLOUD_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, 
                timeout=30 
            )
            if response and response.choices:
                raw_text = response.choices[0].message.content
                parsed_json = extract_json_from_text(raw_text)
                if parsed_json:
                    return parsed_json
        except Exception as e:
            if "429" in str(e):
                time.sleep(10)
            else:
                time.sleep(2)
    return None

def main():

    geojson_path = RULE_ENGINE_OUTPUT_PATH.replace('.csv', '.geojson')
    if not os.path.exists(geojson_path):
        return
    
    df_all = gpd.read_file(geojson_path)
    if 'geometry' in df_all.columns:
        df_all = df_all.drop(columns=['geometry'])
    df_all = pd.DataFrame(df_all)
    
    new_cols = ['LLM_Main_Class', 'LLM_Sub_Class', 'LLM_Reasoning', 'LLM_Confidence']
    for col in new_cols:
        if col not in df_all.columns: df_all[col] = np.nan
    
    candidates = filter_data_for_llm(df_all)
    if candidates.empty:
        return

    if os.path.exists(LLM_VERIFICATION_OUTPUT):
        try:
            df_existing = pd.read_csv(LLM_VERIFICATION_OUTPUT, low_memory=False)
            processed_ids = set(df_existing.loc[df_existing['LLM_Confidence'].notna(), 'BUILDINGSTRUCTUREID'])
            candidates = candidates[~candidates['BUILDINGSTRUCTUREID'].isin(processed_ids)]
            df_all.set_index('BUILDINGSTRUCTUREID', inplace=True)
            df_existing.set_index('BUILDINGSTRUCTUREID', inplace=True)
            df_all.update(df_existing[new_cols])
            df_all.reset_index(inplace=True)
        except: pass

    MAX_THREADS = 15 
    updates_buffer = []
    pbar = tqdm(total=len(candidates), desc="Processing")

    def process_row(idx_and_row):
        idx, row = idx_and_row
        time.sleep(np.random.uniform(0.1, 0.4))
        result = call_llm_api(generate_prompt(row))
        if result:
            return {'idx': idx, **result}
        return None

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(process_row, (idx, row)): idx for idx, row in candidates.iterrows()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                updates_buffer.append({
                    'idx': res['idx'],
                    'LLM_Main_Class': res.get('main_class'),
                    'LLM_Sub_Class': res.get('sub_class'),
                    'LLM_Reasoning': res.get('reasoning'),
                    'LLM_Confidence': res.get('confidence')
                })
            pbar.update(1)
            
            if len(updates_buffer) >= BATCH_SAVE_INTERVAL:
                for u in updates_buffer:
                    t_idx = u.pop('idx')
                    for k, v in u.items(): df_all.at[t_idx, k] = v
                df_all.to_csv(LLM_VERIFICATION_OUTPUT, index=False, encoding='utf-8-sig')
                updates_buffer = []

    pbar.close()
    if updates_buffer:
        for u in updates_buffer:
            t_idx = u.pop('idx')
            for k, v in u.items(): df_all.at[t_idx, k] = v
    df_all.to_csv(LLM_VERIFICATION_OUTPUT, index=False, encoding='utf-8-sig')

import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if '\r' not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_dir = os.path.join(os.path.dirname(base_dir), "log")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, "4.llm_verification.txt")
    
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    
    
    main()
