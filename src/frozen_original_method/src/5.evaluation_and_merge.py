# 5.evaluation_and_merge.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import sys
import re
from tqdm.auto import tqdm
from collections import defaultdict

try:
    from config import (
        RULE_ENGINE_OUTPUT_PATH,       
        LLM_VERIFICATION_OUTPUT,       
        STEP5_MERGED_OUTPUT_PATH,
        LLM_TAXONOMY
    )
    LLM_CONFIDENCE_THRESHOLD = 0.80  
except ImportError:
    RULE_ENGINE_OUTPUT_PATH = "./output/step3_rule_engine_classification.csv"
    LLM_VERIFICATION_OUTPUT = "./output/step4_llm_verified_classification.csv"
    STEP5_MERGED_OUTPUT_PATH = "./output/step5_merged_classification.csv"
    LLM_CONFIDENCE_THRESHOLD = 0.80
    from config import LLM_TAXONOMY

FINAL_CSV_PATH = STEP5_MERGED_OUTPUT_PATH
FINAL_GEOJSON_PATH = STEP5_MERGED_OUTPUT_PATH.replace('.csv', '.geojson')
RULE_GEOJSON_BASE = RULE_ENGINE_OUTPUT_PATH.replace('.csv', '.geojson')

REVERSE_TAXONOMY = {}
for main_cls, subs in LLM_TAXONOMY.items():
    if main_cls == "__STRONG_KEYWORDS__": continue
    for sub in subs:
        REVERSE_TAXONOMY[sub] = main_cls
        chinese_part = sub.split('（')[0]
        REVERSE_TAXONOMY[chinese_part] = main_cls

def get_main_class_of_sub(sub_class):
    clean_sub = str(sub_class).strip()
    return REVERSE_TAXONOMY.get(clean_sub, '未知类别')

def apply_merge_logic(row):
    rule_main = str(row.get('Final_Main_Class', '未知类别'))
    rule_sub = str(row.get('Final_Sub_Class', '未知类别'))
    rule_source = str(row.get('Classification_Source', 'Unknown'))
    rule_main_prop = str(row.get('Main_Class_Proportions', ''))
    rule_sub_prop = str(row.get('Sub_Class_Proportions', ''))
    
    llm_main = str(row.get('LLM_Main_Class', 'nan'))
    llm_sub = str(row.get('LLM_Sub_Class', 'nan'))
    matched_pois = str(row.get('Matched_POIs', '')).lower()
    
    try:
        llm_conf = float(row.get('LLM_Confidence', 0.0))
    except:
        llm_conf = 0.0

    if pd.isna(row.get('LLM_Main_Class')) or llm_main in ['nan', 'None', '', '解析失败', '未知类别'] or llm_conf < LLM_CONFIDENCE_THRESHOLD:
        return pd.Series([rule_main, rule_sub, rule_source, rule_main_prop, rule_sub_prop, False])
    
    valid_mains = [k for k in LLM_TAXONOMY.keys() if k != '__STRONG_KEYWORDS__'] + ['混合用途', '未知类别', '非评估类别']
    
    if llm_main not in valid_mains:
        rescued_main = get_main_class_of_sub(llm_main)
        if rescued_main != '未知类别':
            llm_sub = llm_main
            llm_main = rescued_main
        else:
            reject_source = f"{rule_source} (LLM Rejected - Hallucination)"
            return pd.Series([rule_main, rule_sub, reject_source, rule_main_prop, rule_sub_prop, False])
    
    commercial_tags = [
        '[restaurant]', '[shop]', '[retail]', '[clinic]', '[bank]', 
        '[school]', '[cafe]', '[fast_food]', '[supermarket]', 
        '[pharmacy]', '[hospital]', '[kindergarten]', '[office]'
    ]
    has_real_bottom_shop = any(tag in matched_pois for tag in commercial_tags)
    
    if rule_main == '混合用途' and has_real_bottom_shop:
        if llm_main in ['住宅类别', '商业类别', '工业类别', '非评估类别']:
            protected_source = f"{rule_source} (LLM Override Rejected - Bottom Shop Protected)"
            return pd.Series([rule_main, rule_sub, protected_source, rule_main_prop, rule_sub_prop, False])

    new_source = f"LLM_Verified (Conf: {llm_conf:.2f})"
    is_llm_overridden = True
    
    new_main_prop = ""
    new_sub_prop = ""
    
    if llm_main == '混合用途':
        subs = [s.strip() for s in llm_sub.split('-') if s.strip()]
        if len(subs) > 1:
            weight = 1.0 / len(subs)
            main_dict = defaultdict(float)
            sub_list = []
            
            for s in subs:
                sub_list.append(f"{s}:{weight:.2f}")
                m = get_main_class_of_sub(s)
                if m != '未知类别':
                    main_dict[m] += weight
                    
            new_main_prop = "-".join([f"{k}:{v:.2f}" for k, v in main_dict.items() if v > 0])
            new_sub_prop = "-".join(sub_list)
        else:
            new_main_prop = "混合用途:1.00"
            new_sub_prop = f"{llm_sub}:1.00"
            
    else:
        new_main_prop = f"{llm_main}:1.00"
        new_sub_prop = f"{llm_sub}:1.00"

    return pd.Series([llm_main, llm_sub, new_source, new_main_prop, new_sub_prop, is_llm_overridden])


def main():
    
    if not os.path.exists(LLM_VERIFICATION_OUTPUT):
        return
        
    df_llm = pd.read_csv(LLM_VERIFICATION_OUTPUT, low_memory=False)
    
    tqdm.pandas(desc="Merging")
    merge_results = df_llm.progress_apply(apply_merge_logic, axis=1)
    
    df_llm[['Merged_Main_Class', 'Merged_Sub_Class', 'Merged_Source', 
            'Merged_Main_Proportions', 'Merged_Sub_Proportions', 'Is_LLM_Overridden']] = merge_results
            
    overridden_count = df_llm['Is_LLM_Overridden'].sum()
    
    df_llm['Final_Main_Class'] = df_llm['Merged_Main_Class']
    df_llm['Final_Sub_Class'] = df_llm['Merged_Sub_Class']
    df_llm['Classification_Source'] = df_llm['Merged_Source']
    df_llm['Main_Class_Proportions'] = df_llm['Merged_Main_Proportions']
    df_llm['Sub_Class_Proportions'] = df_llm['Merged_Sub_Proportions']
    
    cols_to_drop = [
        'Merged_Main_Class', 'Merged_Sub_Class', 'Merged_Source', 
        'Merged_Main_Proportions', 'Merged_Sub_Proportions', 'Is_LLM_Overridden',
        'LLM_Main_Class', 'LLM_Sub_Class', 'LLM_Reasoning', 'LLM_Confidence',
        'Pre_Class_Main', 'Pre_Class_Sub', 'Classification_Stage', 'IS_FINAL_RECALC'
    ]
    df_final_csv = df_llm.drop(columns=[c for c in cols_to_drop if c in df_llm.columns], errors='ignore')
    
    if os.path.exists(RULE_GEOJSON_BASE):
        gdf_base = gpd.read_file(RULE_GEOJSON_BASE)
        gdf_base = gdf_base[['BUILDINGSTRUCTUREID', 'geometry']]
        
        gdf_final = gdf_base.merge(df_final_csv, on='BUILDINGSTRUCTUREID', how='inner')
        
        for col in gdf_final.columns:
            if col != 'geometry':
                gdf_final[col] = gdf_final[col].apply(
                    lambda x: "" if pd.isna(x) or str(x).lower() == 'nan' else str(x)
                )
                
        gdf_final.to_file(FINAL_GEOJSON_PATH, driver='GeoJSON')
    else:
        pass

    df_final_csv.to_csv(FINAL_CSV_PATH, index=False, encoding='utf-8-sig')
    
    print(df_final_csv['Final_Main_Class'].value_counts().to_markdown())
    print(df_final_csv['Final_Sub_Class'].value_counts().head(10).to_markdown())
    print(df_final_csv['Classification_Source'].apply(lambda x: 'LLM_Verified' if 'LLM_Verified' in str(x) else x).value_counts().head(5).to_markdown())
    

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
    
    log_file_path = os.path.join(log_dir, "5.evaluation_and_merge.txt")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    
    main()
