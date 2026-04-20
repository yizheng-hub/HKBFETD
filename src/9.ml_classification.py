# 9.ml_classification.py
# -*- coding: utf-8 -*-

import os
import warnings
import json
import re
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm.auto import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, f1_score
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from config import (
    MAIN_CLASSES_ML, PROBABILITY_THRESHOLD, SMOTE_MIN_SAMPLES, SMOTE_K_NEIGHBORS,
    INTERMEDIATE_DIR, OUTPUT_DIR, FEATURE_X_ALL_PATH, FEATURE_X_TRAIN_ML_PATH,
    FEATURE_Y_MULTILABEL_PATH, FEATURE_BASE_INDEXED_PATH, FEATURE_GDF_BASE_CALIBRATED_PATH,
    RULE_ENGINE_OUTPUT_PATH, AI_CALIBRATED_OUTPUT_PATH, AGGREGATED_GDF_PATH, KEYWORDS_FILE, 
    ML_FINAL_OUTPUT_PATH, SUBCLASS_PROPORTION_THRESHOLD
)

from utils import init_keyword_tool, classify_text_by_keywords, safe_str, to_simplified_chinese

warnings.filterwarnings('ignore')
tqdm.pandas()

def load_ml_inputs():
    print("[INFO] Status message emitted.")
    try:
        import pickle
        with open(FEATURE_X_ALL_PATH, 'rb') as f: X_all = pickle.load(f)
        with open(FEATURE_X_TRAIN_ML_PATH, 'rb') as f: X_train_ml = pickle.load(f)
        with open(FEATURE_Y_MULTILABEL_PATH, 'rb') as f: y_multilabel = pickle.load(f)
        with open(FEATURE_BASE_INDEXED_PATH, 'rb') as f: feature_base_indexed = pickle.load(f)
        
        gdf_base_calibrated = gpd.read_file(FEATURE_GDF_BASE_CALIBRATED_PATH)
        df_rule_classified = pd.read_csv(AI_CALIBRATED_OUTPUT_PATH)
        
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            KEYWORDS_CONFIG = json.load(f)
            
        keyword_tool = init_keyword_tool()
        
        return {
            'X_all': X_all, 'X_train_ml': X_train_ml, 'y_multilabel': y_multilabel,
            'feature_base_indexed': feature_base_indexed, 'gdf_base_calibrated': gdf_base_calibrated,
            'df_rule_classified': df_rule_classified, 'keyword_tool': keyword_tool
        }
    except Exception as e:
        print("[INFO] Status message emitted.")
        return None

def train_main_class_models(X_train_ml, y_multilabel, X_all):
    print("[INFO] Status message emitted.")
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_ml, y_multilabel, test_size=0.2, random_state=42)
    models = {}
    predictions_proba = pd.DataFrame(index=X_all.index)

    performance_metrics = {}    
    
    for main_class in MAIN_CLASSES_ML:
        col_name = f'is_{main_class}'
        if col_name not in y_multilabel.columns: y_multilabel[col_name] = 0
        y_train_s[col_name] = y_multilabel.loc[y_train_s.index, col_name]
        y_val_s[col_name] = y_multilabel.loc[y_val_s.index, col_name]

    for main_class in tqdm(MAIN_CLASSES_ML, desc="训练主类分类器", leave=True, position=0):
        print("[INFO] Status message emitted.")
        col_name = f'is_{main_class}'
        y_train_c = y_train_s[col_name]
        pos_count = y_train_c.sum()
        
        X_train_resampled, y_train_resampled = X_train_s, y_train_c
        if pos_count > 0:
            if pos_count < SMOTE_MIN_SAMPLES:
                try:
                    k_neigh = max(1, min(pos_count - 1, 5))
                    smote = SMOTE(random_state=42, sampling_strategy={1: SMOTE_MIN_SAMPLES}, k_neighbors=k_neigh)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_s, y_train_c)
                except: pass
            elif (len(y_train_c) / pos_count) > 2:
                try:
                    smote = SMOTE(random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_s, y_train_c)
                except: pass
        
        try:
            model = lgb.LGBMClassifier(random_state=42, verbose=-1).fit(X_train_resampled, y_train_resampled)
            
            val_preds = model.predict(X_val_s)
            f1 = f1_score(y_val_s[col_name], val_preds, zero_division=0)
            performance_metrics[main_class] = f1
            
            print("[INFO] Status message emitted.")
            print(classification_report(y_val_s[col_name], val_preds, zero_division=0))

            final_y_class = y_multilabel[col_name]
            final_pos_count = final_y_class.sum()
            X_final_resampled, y_final_resampled = X_train_ml, final_y_class
            
            if final_pos_count > 0:
                try:
                    if final_pos_count < SMOTE_MIN_SAMPLES:
                        smote_final = SMOTE(random_state=42, sampling_strategy={1: SMOTE_MIN_SAMPLES}, k_neighbors=max(1, min(final_pos_count - 1, 5)))
                    elif (len(final_y_class) / final_pos_count) > 2:
                        smote_final = SMOTE(random_state=42)
                    if smote_final:
                        X_final_resampled, y_final_resampled = smote_final.fit_resample(X_train_ml, final_y_class)
                except: pass 
            
            final_model = lgb.LGBMClassifier(random_state=42, verbose=-1).fit(X_final_resampled, y_final_resampled)
            models[main_class] = final_model
            proba = final_model.predict_proba(X_all)
            predictions_proba[f'proba_{main_class}'] = proba[:, 1] if len(proba.shape) == 2 else proba.flatten()
            
        except Exception as e:
            print("[INFO] Status message emitted.")
            predictions_proba[f'proba_{main_class}'] = 0.0

    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    if performance_metrics:
        avg_f1 = np.mean(list(performance_metrics.values()))
        print("[INFO] Status message emitted.")
        for m_class, f1_val in performance_metrics.items():
            print(f"   - {m_class}: {f1_val*100:.2f}%")
    print("[INFO] Status message emitted.")
    
    return models, predictions_proba

def train_subclass_models(X_all, X_train_ml, df_rule_classified, keyword_tool):
    print("[INFO] Status message emitted.")
    df_subclass_proportions = pd.DataFrame(index=X_all.index)
    rule_df = df_rule_classified.set_index('BUILDINGSTRUCTUREID')
    train_rule = rule_df.reindex(X_train_ml.index)

    for main_class in MAIN_CLASSES_ML:
        SUBCLASSES = sorted([k for k in keyword_tool['keywords_config'].get(main_class, {}).keys()])
        if not SUBCLASSES: continue

        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        
        mask = train_rule['Final_Main_Class'] == main_class
        X_sub = X_train_ml[mask]
        y_raw = train_rule.loc[mask, 'Final_Sub_Class']
        
        print("[INFO] Status message emitted.")
        if len(X_sub) < 50:
            print("[INFO] Status message emitted.")
            continue

        y_sub = pd.DataFrame(0.0, index=X_sub.index, columns=SUBCLASSES)
        for idx, val in y_raw.items():
            if pd.isna(val): 
                y_sub.loc[idx, SUBCLASSES[0]] = 1.0
                continue
            matched = False
            for sub_col in SUBCLASSES:
                if sub_col.split('（')[0] in str(val) or str(val) in sub_col:
                    y_sub.loc[idx, sub_col] = 1.0
                    matched = True
                    break
            if not matched: y_sub.loc[idx, SUBCLASSES[0]] = 1.0

        X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
        regressor = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1).fit(X_train_r, y_train_r)
        print("[INFO] Status message emitted.")

        props = regressor.predict(X_all)
        df_props = pd.DataFrame(props, index=X_all.index, columns=SUBCLASSES)
        def format_proportions(row):
            return {sub: round(val, 2) for sub, val in row.items() if val > SUBCLASS_PROPORTION_THRESHOLD}
        df_subclass_proportions[f'{main_class}_proportions'] = df_props.apply(format_proportions, axis=1)

    return df_subclass_proportions

def apply_ml_calibration(df_rule_classified, predictions_proba, subclass_props, X_all):
    print("[INFO] Status message emitted.")
    
    PROBABILITY_THRESHOLD = 0.5
    
    df_ml_summary = pd.DataFrame(index=X_all.index)
    df_ml_summary['ML_Confidence'] = predictions_proba.max(axis=1)
    
    def summarize_ml(row):
        active_classes = [c.replace('proba_', '') for c, p in row.items() if p >= PROBABILITY_THRESHOLD]
        if not active_classes: return '未知类别', '未知类别', 'ML Low Confidence'
        elif len(active_classes) == 1: return active_classes[0], active_classes[0], 'ML Single Class'
        else: return '混合用途', '/'.join(sorted(active_classes)), 'ML Multi-Label'
    
    summary_cols = predictions_proba.apply(summarize_ml, axis=1, result_type='expand')
    summary_cols.columns = ['ML_Main_Class', 'ML_Sub_Class', 'ML_Source']
    
    ml_info_table = pd.concat([predictions_proba, subclass_props, df_ml_summary, summary_cols], axis=1)
    ml_info_table.index.name = 'BUILDINGSTRUCTUREID'
    
    df_to_calibrate = df_rule_classified.merge(ml_info_table.reset_index(), on='BUILDINGSTRUCTUREID', how='left')
    df_to_calibrate = df_to_calibrate.merge(X_all[['area']].reset_index(), on='BUILDINGSTRUCTUREID', how='left')

    def final_calibration(row):
        rule_main = str(row.get('Final_Main_Class', '未知类别')).strip()
        rule_sub = str(row.get('Final_Sub_Class', '未知类别')).strip()
        rule_source = str(row.get('Classification_Source', 'None')).strip()
        rule_mix = str(row.get('Sub_Class_Proportions', '')).strip()

        if rule_main.lower() in ['unknown', 'nan', 'none', '']: rule_main = '未知类别'
        if rule_sub.lower() in ['unknown', 'nan', 'none', '']: rule_sub = '未知类别'
        
        rule_sub = rule_sub.replace('餐饮（Catering）', '食肆（Restaurant）').replace('餐饮（Restaurant）', '食肆（Restaurant）')
        rule_mix = rule_mix.replace('餐饮（Catering）', '食肆（Restaurant）').replace('餐饮（Restaurant）', '食肆（Restaurant）')
        
        ml_main = str(row['ML_Main_Class']) if pd.notna(row['ML_Main_Class']) else '未知类别'
        ml_conf = row['ML_Confidence'] if pd.notna(row['ML_Confidence']) else 0

        if rule_main == '非评估类别':
            return ('非评估类别', '非评估类别', '非评估类别:1.00', f"Protected({rule_source})", "保留被判定为噪声/非评估的建筑")

        incomplete_reason = ""
        if rule_main == '未知类别':
            incomplete_reason = "全残(主类未知)"
        elif rule_sub == '未知类别':
            incomplete_reason = "半残(子类未知)"
        elif rule_mix == '' or '未知' in rule_mix or 'unknown' in rule_mix.lower() or 'nan' in rule_mix.lower():
            incomplete_reason = "半残(比例空缺或异常)"
            
        is_rule_incomplete = bool(incomplete_reason)
        weak_sources = ['Fallback_to_Default', 'Fallback_to_Area', 'None', 'Unknown']

        if not is_rule_incomplete and rule_source not in weak_sources:
            return (rule_main, rule_sub, rule_mix, f"Rule_Engine({rule_source})", "保留规则确定性结论")

        
        if ml_conf < 0.5:  
            return ('非评估类别', '置信度过低归为非评估', '置信度过低归为非评估:1.00', 'ML_Low_Confidence_Discard', f"ML推断置信度过低(P={ml_conf:.2f})")

        if str(row.get('ML_Source')) != 'ML Low Confidence' and ml_main != '未知类别':
            note_prefix = f"ML覆盖[{incomplete_reason}]-" if is_rule_incomplete else "ML推断填补-"
            
            if ml_main == '混合用途':
                active = [c.replace('proba_','') for c, p in row.items() if isinstance(c, str) and c.startswith('proba_') and p >= PROBABILITY_THRESHOLD]
                if not active: active = ['住宅类别', '商业类别']
                
                sub_mix_parts = []
                for act_main in active:
                    prop_dict = row.get(f'{act_main}_proportions')
                    if isinstance(prop_dict, dict) and prop_dict:
                        top_sub = max(prop_dict.items(), key=lambda x: x[1])[0]
                    else:
                        top_sub = {'住宅类别': '私人房屋（Private Housing）', '商业类别': '其他商业（Other Commercial）', '工业类别': '其他工业（Other Industrial）'}.get(act_main, '其他商业（Other Commercial）')
                    sub_mix_parts.append(f"{top_sub}:{1.0/len(active):.2f}")
                
                sub_mix = "-".join(sub_mix_parts)
                top_subclass = sub_mix.split(':')[0]
                return (ml_main, top_subclass, sub_mix, 'Machine_Learning_Predicted', f"{note_prefix}ML多标签混合(P={ml_conf:.2f})")
            
            else:
                prop_dict = row.get(f'{ml_main}_proportions')
                if isinstance(prop_dict, dict) and prop_dict:
                    sub_mix = "-".join([f"{k}:{v:.2f}" for k, v in sorted(prop_dict.items(), key=lambda x: x[1], reverse=True) if v > 0])
                    top_subclass = max(prop_dict.items(), key=lambda x: x[1])[0]
                else:
                    top_subclass = {'住宅类别': '私人房屋（Private Housing）', '商业类别': '其他商业（Other Commercial）', '工业类别': '其他工业（Other Industrial）'}.get(ml_main, '其他商业（Other Commercial）')
                    sub_mix = f"{top_subclass}:1.00"
                
                if '未知' in top_subclass or 'Unknown' in top_subclass:
                    top_subclass = '其他商业（Other Commercial）'
                    sub_mix = f"{top_subclass}:1.00"
                    
                return (ml_main, top_subclass, sub_mix, 'Machine_Learning_Predicted', f"{note_prefix}ML回归精准分配(P={ml_conf:.2f})")

        area = row.get('area', 0)
        if area < 100:
            return ('住宅类别', '其他房屋（Other Housing）', '其他房屋（Other Housing）:1.00', 'Fallback_to_Area', '全流程未识别,小面积归住宅')
        return ('商业类别', '其他商业（Other Commercial）', '其他商业（Other Commercial）:1.00', 'Fallback_to_Default', '强制兜底为商业')

    tqdm.pandas(desc="执行终极校准与防篡改验证")
    final_cols = df_to_calibrate.progress_apply(final_calibration, axis=1, result_type='expand')
    final_cols.columns = ['Calibrated_Main_Class', 'Calibrated_Sub_Class', 'Calibrated_Mix_Proportion', 'Calibrated_Source', 'Calibrated_Notes']
    
    print("[INFO] Status message emitted.")
    override_mask = final_cols['Calibrated_Notes'].str.contains('ML覆盖', na=False)
    override_df = final_cols[override_mask]
    print("[INFO] Status message emitted.")
    if len(override_df) > 0:
        print("[INFO] Status message emitted.")
        reasons = override_df['Calibrated_Notes'].str.extract(r'ML覆盖\[(.*?)\]')[0].value_counts()
        for r, c in reasons.items():
            print("[INFO] Status message emitted.")
    print("--------------------------------------------------")

    return pd.concat([df_to_calibrate.drop(columns=['area'], errors='ignore'), final_cols], axis=1)

def generate_final_output(df_calibrated, gdf_base):
    final_df = df_calibrated.copy()
    
    final_df.to_csv(ML_FINAL_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print("[INFO] Status message emitted.")
    
    geojson_path = ML_FINAL_OUTPUT_PATH.replace('.csv', '.geojson')
    print("[INFO] Status message emitted.")
    
    gdf_out = gdf_base[['BUILDINGSTRUCTUREID', 'geometry']].merge(
        final_df, on='BUILDINGSTRUCTUREID', how='inner'
    )
    
    for col in gdf_out.columns:
        if col != 'geometry' and gdf_out[col].dtype == object:
            gdf_out[col] = gdf_out[col].fillna('').astype(str)
            
    gdf_out.to_file(geojson_path, driver='GeoJSON')
    print("[INFO] Status message emitted.")
    
    return final_df

def print_final_statistics(df):
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)

    print("[INFO] Status message emitted.")
    main_counts = df['Calibrated_Main_Class'].value_counts()
    for idx, count in main_counts.items():
        print("[INFO] Status message emitted.")

    print("[INFO] Status message emitted.")
    sub_counts = df['Calibrated_Sub_Class'].value_counts()
    for idx, count in sub_counts.items():
        print("[INFO] Status message emitted.")

    non_eval = df[df['Calibrated_Main_Class'] == '非评估类别']
    print("[INFO] Status message emitted.")

    mixed_df = df[df['Calibrated_Main_Class'] == '混合用途']
    print("[INFO] Status message emitted.")
    mix_combinations = mixed_df['Calibrated_Mix_Proportion'].value_counts()

    print("[INFO] Status message emitted.")
    for idx, count in mix_combinations.head(20).items():
        print("[INFO] Status message emitted.")

    print("\n" + "="*60)

def main():
    print("[INFO] Status message emitted.")
    data = load_ml_inputs()
    if not data: return False
    
    models, proba = train_main_class_models(data['X_train_ml'], data['y_multilabel'], data['X_all'])
    subclass_props = train_subclass_models(data['X_all'], data['X_train_ml'], data['df_rule_classified'], data['keyword_tool'])
    
    df_calibrated = apply_ml_calibration(data['df_rule_classified'], proba, subclass_props, data['X_all'])
    
    final_df = generate_final_output(df_calibrated, data['gdf_base_calibrated'])
    print_final_statistics(final_df)
    
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
        
    def isatty(self):
        return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, "9.ml_classification.txt")
    
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout 
    
    print("[INFO] Status message emitted.")
    
    try:
        success = main()
        if not success: sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)