# 7.apply_calibrations.py
# -*- coding: utf-8 -*-

"""Module documentation."""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm.auto import tqdm
from collections import defaultdict
import traceback

conda_prefix = sys.prefix
proj_lib_path = os.path.join(conda_prefix, 'Library', 'share', 'proj')
if os.path.exists(proj_lib_path):
    os.environ['PROJ_LIB'] = proj_lib_path
    os.environ['PROJ_DATA'] = proj_lib_path
else:
    fallback_path = os.path.join(conda_prefix, 'share', 'proj')
    os.environ['PROJ_LIB'] = fallback_path
    os.environ['PROJ_DATA'] = fallback_path
# =========================================================

from config import (
    STEP5_MERGED_OUTPUT_PATH, AI_CALIBRATED_OUTPUT_PATH,
    CANDIDATES_RULES_PATH, AI_DECISIONS_LOG_PATH, FINAL_SUGGESTIONS_PATH,
    KEYWORDS_FILE
)

warnings.filterwarnings('ignore')
tqdm.pandas()

ABSOLUTE_FINAL_CSV = STEP5_MERGED_OUTPUT_PATH
ABSOLUTE_FINAL_GEOJSON = STEP5_MERGED_OUTPUT_PATH.replace('.csv', '.geojson')
AI_CALIBRATED_GEOJSON_PATH = AI_CALIBRATED_OUTPUT_PATH.replace('.csv', '.geojson')

def clean_id_global(x):
    s = str(x).strip()
    return s[:-2] if s.endswith('.0') else s

def apply_ai_arbitration(df_base):
    print("[INFO] Status message emitted.")
    try:
        df_rules = pd.read_csv(CANDIDATES_RULES_PATH)
        df_rules['fragment_id'] = df_rules['fragment_id'].apply(clean_id_global)
        df_rules['host_id'] = df_rules['host_id'].apply(clean_id_global)
        
        nan_mask = df_rules['rule_decision'].isna() | (df_rules['rule_decision'].astype(str).str.lower() == 'nan') | (df_rules['rule_decision'] == '')
        
        if nan_mask.any():
            print("[INFO] Status message emitted.")
            if isinstance(df_base, gpd.GeoDataFrame):
                print("[INFO] Status message emitted.")
                df_base_temp = df_base.copy()
                df_base_temp['BUILDINGSTRUCTUREID'] = df_base_temp['BUILDINGSTRUCTUREID'].apply(clean_id_global)
                gdf_all = df_base_temp.set_index('BUILDINGSTRUCTUREID')
                
                def recalculate_rule(row):
                    try:
                        f_id = clean_id_global(row['fragment_id'])
                        h_id = clean_id_global(row['host_id'])
                        if f_id not in gdf_all.index or h_id not in gdf_all.index:
                            return "keep"
                            
                        f_geom = gdf_all.loc[f_id].geometry
                        h_geom = gdf_all.loc[h_id].geometry
                        
                        if isinstance(f_geom, pd.Series): f_geom = f_geom.iloc[0]
                        if isinstance(h_geom, pd.Series): h_geom = h_geom.iloc[0]
                        
                        shared_len = f_geom.buffer(1e-4).intersection(h_geom.buffer(1e-4)).length
                        if shared_len > f_geom.length * 0.3: 
                            return "merge"
                        
                        if f_geom.distance(h_geom) < 1e-4 and f_geom.area < 15: 
                            return "delete"
                            
                        return "keep"
                    except Exception as e:
                        return "keep"
                
                df_rules.loc[nan_mask, 'rule_decision'] = df_rules[nan_mask].apply(recalculate_rule, axis=1)
                
                df_rules.to_csv(CANDIDATES_RULES_PATH, index=False, encoding='utf-8-sig')
                print("[INFO] Status message emitted.")
            else:
                print("[INFO] Status message emitted.")
                df_rules.loc[nan_mask, 'rule_decision'] = 'keep'
        # =========================================================================

        if os.path.exists(AI_DECISIONS_LOG_PATH):
            df_ai = pd.read_csv(AI_DECISIONS_LOG_PATH)
            df_ai['fragment_id'] = df_ai['fragment_id'].apply(clean_id_global)
            df_ai['host_id'] = df_ai['host_id'].apply(clean_id_global)
            df_ai = df_ai.drop_duplicates(subset=['fragment_id', 'host_id'], keep='last')
            df_combined = df_rules.merge(df_ai, on=['fragment_id', 'host_id'], how='left')
        else:
            df_combined = df_rules.copy()
            df_combined['ai_decision'] = None
        
        def final_arbitration_v5(row):
            rule = row.get('rule_decision', 'keep')
            ai = row.get('ai_decision', None)
            if pd.isna(ai) or ai in ['error', 'unknown', None]: return f"Rule-Only_{rule}"
            if ai == rule: return f"Consensus_{ai}"
            if rule == 'merge' and ai == 'keep': return "Final_keep_with_inheritance"
            if rule == 'keep' and ai == 'merge': return "Manual_Review_AI_Wants_Merge" 
            if ai == 'delete':
                if rule == 'merge': return "Final_delete (AI overrides Rule_Merge)"
                else: return "Manual_Review_AI_Wants_Delete"
            if rule == 'delete' and ai != 'delete': return f"Final_{ai}_(AI_overrides_Rule_Delete)"
            return f"Unknown_Conflict_AI({ai})_Rule({rule})"

        df_combined['suggestion'] = df_combined.apply(final_arbitration_v5, axis=1)
        
        def handle_multi_host_conflicts(group):
            suggestions = group['suggestion'].unique()
            has_strong = any(s for s in suggestions if 'merge' in s or 'delete' in s or 'inheritance' in s)
            has_keep = any(s for s in suggestions if 'keep' in s)
            if has_strong and has_keep:
                group['final_suggestion'] = 'Final_keep (Conflict_Override)'
                return group
            merge_suggestions = group[group['suggestion'].str.contains('merge|inheritance', case=False, na=False)]
            if len(merge_suggestions) > 1:
                group.loc[merge_suggestions.index, 'final_suggestion'] = 'Manual_Review_Multi_Merge'
                group['final_suggestion'].fillna('Manual_Review_Multi_Merge', inplace=True)
            else:
                group['final_suggestion'] = group['suggestion']
            return group
            
        tqdm.pandas(desc="处理多主体冲突")
        df_final_suggestions = df_combined.groupby('fragment_id').progress_apply(handle_multi_host_conflicts).reset_index(drop=True)
        df_final_suggestions.to_csv(FINAL_SUGGESTIONS_PATH, index=False, encoding='utf-8-sig')
        return df_final_suggestions
    except Exception as e:
        traceback.print_exc()
        return None

def build_keyword_aligner():
    """Function documentation."""
    alias_map = {}
    valid_mains = set(['混合用途', '未知类别', '非评估类别'])
    valid_subs = set(['未知混合', '未知类别', 'AI/Rule判定为噪声', '绝对噪声相关', '交通基础设施', '临时/杂项设施'])
    
    try:
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            kw_data = json.load(f)
            
        for main_cls, sub_dict in kw_data.items():
            if main_cls == '__STRONG_KEYWORDS__': continue
            valid_mains.add(main_cls)
            
            if isinstance(sub_dict, dict):
                for std_name, aliases in sub_dict.items():
                    valid_subs.add(std_name)
                    alias_map[std_name] = std_name
                    for alias in aliases:
                        alias_map[alias] = std_name
                        
        print("[INFO] Status message emitted.")
    except Exception as e:
        print("[INFO] Status message emitted.")
        
    return alias_map, list(valid_mains), list(valid_subs)

def apply_correction_suggestions(df_rule_classified, df_final_suggestions):
    print("[INFO] Status message emitted.")
    try:
        df_to_apply = df_rule_classified.copy()
        df_to_apply['BUILDINGSTRUCTUREID'] = df_to_apply['BUILDINGSTRUCTUREID'].apply(clean_id_global)
        stats = {'inherited': 0, 'deleted': 0, 'manual': 0}
        
        to_inherit = df_final_suggestions[df_final_suggestions['final_suggestion'].isin(['Consensus_merge', 'Final_keep_with_inheritance', 'Rule-Only_merge'])]
        host_map = df_to_apply.set_index('BUILDINGSTRUCTUREID')[['Final_Main_Class', 'Final_Sub_Class']].to_dict('index')
        
        updates = []
        for _, row in tqdm(to_inherit.iterrows(), total=len(to_inherit), desc="执行合并/继承"):
            frag_id, host_id = clean_id_global(row['fragment_id']), clean_id_global(row['host_id'])
            if host_id in host_map:
                h_info = host_map[host_id]
                updates.append({
                    'BUILDINGSTRUCTUREID': frag_id,
                    'Final_Main_Class': h_info['Final_Main_Class'],
                    'Final_Sub_Class': h_info['Final_Sub_Class'],
                    'Classification_Source': f"Inherited_via_{row['final_suggestion']}"
                })
                stats['inherited'] += 1
        
        if updates:
            update_df = pd.DataFrame(updates).set_index('BUILDINGSTRUCTUREID')
            df_to_apply.set_index('BUILDINGSTRUCTUREID', inplace=True)
            df_to_apply.update(update_df)
            df_to_apply.reset_index(inplace=True)

        to_delete = df_final_suggestions[df_final_suggestions['final_suggestion'].isin(['Consensus_delete', 'Final_delete (AI overrides Rule_Merge)', 'Rule-Only_delete'])]
        delete_ids = to_delete['fragment_id'].apply(clean_id_global).unique()
        mask_delete = df_to_apply['BUILDINGSTRUCTUREID'].isin(delete_ids)
        
        df_to_apply.loc[mask_delete, 'Final_Main_Class'] = '非评估类别'
        df_to_apply.loc[mask_delete, 'Final_Sub_Class'] = 'AI/Rule判定为噪声'
        df_to_apply.loc[mask_delete, 'Classification_Source'] = 'Deleted_AI_Rule_Verified'
        stats['deleted'] = mask_delete.sum()
        stats['manual'] = len(df_final_suggestions[df_final_suggestions['final_suggestion'].str.contains('Manual_Review', na=False)])
        print("[INFO] Status message emitted.")
        
        print("[INFO] Status message emitted.")
        alias_map, valid_mains, valid_subs = build_keyword_aligner()

        def align_main_class(m):
            m_str = str(m).strip()
            if m_str in valid_mains: return m_str
            return '未知类别'

        def align_subclass(sub_name):
            if pd.isna(sub_name): return sub_name
            sub_name_str = str(sub_name)
            parts = sub_name_str.split('-')
            aligned_parts = []
            for p in parts:
                p_clean = p.strip()
                matched = False
                for alias, std_name in sorted(alias_map.items(), key=lambda x: len(x[0]), reverse=True):
                    if alias in p_clean:
                        aligned_parts.append(std_name)
                        matched = True
                        break
                if not matched: 
                    if p_clean in valid_subs: aligned_parts.append(p_clean)
            
            if not aligned_parts: return '未知类别'
            return "-".join(list(dict.fromkeys(aligned_parts))) 

        def align_proportions(prop_str):
            if pd.isna(prop_str) or not prop_str: return prop_str
            parts = str(prop_str).split('-')
            new_props = defaultdict(float)
            for p in parts:
                if ':' in p:
                    k, v = p.split(':', 1)
                    k_aligned = align_subclass(k)
                    if k_aligned != '未知类别':
                        try: new_props[k_aligned] += float(v)
                        except: pass
            if not new_props: return '未知混合:1.00'
            return '-'.join([f"{k}:{v:.2f}" for k, v in new_props.items()])

        df_to_apply['Final_Main_Class'] = df_to_apply['Final_Main_Class'].apply(align_main_class)
        df_to_apply['Final_Sub_Class'] = df_to_apply['Final_Sub_Class'].apply(align_subclass)
        df_to_apply['Sub_Class_Proportions'] = df_to_apply['Sub_Class_Proportions'].apply(align_proportions)
        print("[INFO] Status message emitted.")

        print("[INFO] Status message emitted.")
        def safe_float(val, default=0.0):
            if pd.isna(val): return default
            s = str(val).strip().lower()
            if s in ['', 'none', 'nan', 'null', 'na']: return default
            try: return float(val)
            except: return default

        def estimate_height_and_floors(row):
            floors = safe_float(row.get('NUMABOVEGROUNDSTOREYS'), 0)
            th = safe_float(row.get('TOPHEIGHT'), 0.0)
            bh = safe_float(row.get('BASEHEIGHT'), 0.0)
            b_height = th - bh if (th - bh) > 0 else th
            if floors <= 0 and b_height > 0: floors = max(1, round(b_height / 3.0))
            if floors <= 0: floors = 5 
            est_height = b_height if b_height > 0 else floors * 3.0
            return pd.Series([floors, est_height])

        df_to_apply[['Estimated_Floors', 'Estimated_Height']] = df_to_apply.apply(estimate_height_and_floors, axis=1, result_type='expand')
        df_to_apply['NUMABOVEGROUNDSTOREYS'] = df_to_apply['Estimated_Floors']

        drop_cols = ['Pre_Class_Main', 'Pre_Class_Sub', 'Classification_Stage', 'IS_FINAL_RECALC']
        df_to_apply.drop(columns=[c for c in drop_cols if c in df_to_apply.columns], inplace=True, errors='ignore')

        print("[INFO] Status message emitted.")
        out_cols = [c for c in df_to_apply.columns if c != 'geometry']
        df_to_apply[out_cols].to_csv(AI_CALIBRATED_OUTPUT_PATH, index=False, encoding='utf-8-sig')
        
        if isinstance(df_to_apply, gpd.GeoDataFrame) and 'geometry' in df_to_apply.columns:
            print("[INFO] Status message emitted.")
            gdf_export = df_to_apply.copy()
            for col in gdf_export.columns:
                if col != 'geometry' and gdf_export[col].dtype == object:
                    gdf_export[col] = gdf_export[col].fillna('').astype(str)
            gdf_export.to_file(AI_CALIBRATED_GEOJSON_PATH, driver='GeoJSON')
            
        return df_to_apply
    except Exception as e:
        traceback.print_exc()
        return None

def print_stage_statistics(df):
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    print("[INFO] Status message emitted.")
    main_counts = df['Final_Main_Class'].value_counts(dropna=False)
    for idx, count in main_counts.items():
        print("[INFO] Status message emitted.")

    non_eval = df[df['Final_Main_Class'] == '非评估类别']
    print("[INFO] Status message emitted.")

def main():
    print("="*70)
    print("[INFO] Status message emitted.")
    print("="*70)
    
    if os.path.exists(ABSOLUTE_FINAL_GEOJSON):
        df_base = gpd.read_file(ABSOLUTE_FINAL_GEOJSON)
        if df_base.crs is None: df_base = df_base.set_crs("EPSG:2326", allow_override=True)
    else:
        df_base = pd.read_csv(ABSOLUTE_FINAL_CSV, low_memory=False)
        
    df_suggestions = apply_ai_arbitration(df_base)
    
    if df_suggestions is not None and not df_suggestions.empty:
        final_df = apply_correction_suggestions(df_base, df_suggestions)
        if final_df is not None:
            print_stage_statistics(final_df)
            
    print("[INFO] Status message emitted.")
    return True

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
    def isatty(self): return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(os.path.dirname(base_dir), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "7.apply_calibrations.txt")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout 
    print("[INFO] Status message emitted.")
    try:
        success = main()
        if not success: sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
