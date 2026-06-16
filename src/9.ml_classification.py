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
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBClassifier

from config import (
    MAIN_CLASSES_ML, PROBABILITY_THRESHOLD,
    INTERMEDIATE_DIR, OUTPUT_DIR, FEATURE_X_ALL_PATH, FEATURE_X_TRAIN_ML_PATH,
    FEATURE_Y_MULTILABEL_PATH, FEATURE_BASE_INDEXED_PATH, FEATURE_GDF_BASE_CALIBRATED_PATH,
    RULE_ENGINE_OUTPUT_PATH, AI_CALIBRATED_OUTPUT_PATH, AGGREGATED_GDF_PATH, KEYWORDS_FILE, 
    ML_FINAL_OUTPUT_PATH, SUBCLASS_PROPORTION_THRESHOLD
)

from utils import init_keyword_tool, classify_text_by_keywords, safe_str, to_simplified_chinese

warnings.filterwarnings('ignore')
tqdm.pandas()

def normalize_building_id(v):
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s

def canonicalize_main_label(value):
    text = str(value or "").strip()
    low = text.lower()
    if not text:
        return "Unknown_未知类别"
    if ("residential" in low) or ("住宅" in text):
        return "Residential_住宅类别"
    if ("commercial" in low) or ("商业" in text):
        return "Commercial_商业类别"
    if ("industrial" in low) or ("工业" in text):
        return "Industrial_工业类别"
    if ("mixed-use" in low) or ("mixed use" in low) or ("混合" in text):
        return "Mixed-use_混合用途"
    if ("non-assessed" in low) or ("non assessed" in low) or ("非评估" in text):
        return "Non-assessed_非评估类别"
    if ("unknown" in low) or ("未知" in text):
        return "Unknown_未知类别"
    return text

def normalize_bilingual_sub_label(value):
    s = str(value or "").strip()
    if not s:
        return s
    if "_" in s:
        left, right = s.split("_", 1)
        if re.search(r"[A-Za-z]", left) and re.search(r"[\u4e00-\u9fff]", right):
            return f"{left.strip()}_{right.strip()}"

    m = re.match(r"^\s*([^()（）]+?)\s*[（(]\s*([^()（）]+?)\s*[)）]\s*$", s)
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
        a_has_zh = bool(re.search(r"[\u4e00-\u9fff]", a))
        b_has_en = bool(re.search(r"[A-Za-z]", b))
        a_has_en = bool(re.search(r"[A-Za-z]", a))
        b_has_zh = bool(re.search(r"[\u4e00-\u9fff]", b))
        if a_has_zh and b_has_en:
            return f"{b}_{a}"
        if a_has_en and b_has_zh:
            return f"{a}_{b}"
    return s

_SUBCLASS_CANON_MAP = None

def get_subclass_canonical_map():
    global _SUBCLASS_CANON_MAP
    if _SUBCLASS_CANON_MAP is not None:
        return _SUBCLASS_CANON_MAP

    mapping = {
        "未知类别": "Unknown_未知类别",
        "Unknown": "Unknown_未知类别",
        "Unknown_未知类别": "Unknown_未知类别",
        "绝对噪声相关": "Absolute Noise_绝对噪声相关",
        "Absolute Noise_绝对噪声相关": "Absolute Noise_绝对噪声相关",
        "交通基础设施": "Transport Infrastructure_交通基础设施",
        "Transport Infrastructure_交通基础设施": "Transport Infrastructure_交通基础设施",
        "临时/杂项设施": "Temporary/Misc Facilities_临时/杂项设施",
        "Temporary/Misc Facilities_临时/杂项设施": "Temporary/Misc Facilities_临时/杂项设施",
    }
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for main_name, sub_dict in cfg.items():
            if main_name == "__STRONG_KEYWORDS__" or not isinstance(sub_dict, dict):
                continue
            for sub_name in sub_dict.keys():
                canon = normalize_bilingual_sub_label(sub_name)
                mapping[sub_name] = canon
                if "_" in canon:
                    en, zh = canon.split("_", 1)
                    mapping[en.strip()] = canon
                    mapping[zh.strip()] = canon
    except Exception:
        pass
    _SUBCLASS_CANON_MAP = mapping
    return mapping

def canonicalize_subclass_label(value):
    s = str(value or "").strip()
    if not s:
        return "Unknown_未知类别"
    mapping = get_subclass_canonical_map()

    n = normalize_bilingual_sub_label(s)
    if s in mapping:
        return mapping[s]
    if n in mapping:
        return mapping[n]
    if "_" in n:
        en, zh = n.split("_", 1)
        if en in mapping:
            return mapping[en]
        if zh in mapping:
            return mapping[zh]

    if "-" in s:
        parts = []
        for token in s.split("-"):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                k, v = token.split(":", 1)
                parts.append(f"{canonicalize_subclass_label(k)}:{v.strip()}")
            else:
                parts.append(canonicalize_subclass_label(token))
        return "-".join(parts) if parts else "Unknown_未知类别"
    return n

def subclass_label_matches(candidate, observed):
    """Compare subclass labels across legacy Chinese-English and release formats."""
    candidate_canon = canonicalize_subclass_label(candidate)
    observed_text = str(observed or "").strip()
    if not observed_text:
        return False

    if canonicalize_subclass_label(observed_text) == candidate_canon:
        return True

    tokens = []
    for raw_token in observed_text.split("-"):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            token = token.split(":", 1)[0].strip()
        tokens.append(token)

    if not tokens:
        tokens = [observed_text]

    for token in tokens:
        if canonicalize_subclass_label(token) == candidate_canon:
            return True
    return False

def normalize_proportion_string(value, key_func):
    s = str(value or "").strip()
    if not s:
        return s
    if ":" not in s:
        return key_func(s)
    out = []
    for token in re.split(r"(?<=\d)-", s):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            k, v = token.split(":", 1)
            out.append(f"{key_func(k)}:{v.strip()}")
        else:
            out.append(key_func(token))
    return "-".join(out)


def _make_bilingual_note(en_text, zh_text):
    en = str(en_text or "").strip()
    zh = str(zh_text or "").strip()
    if not en:
        return zh
    if not zh:
        return en
    return f"{en}_{zh}"


def normalize_calibrated_note_to_bilingual(note):
    text = str(note or "").strip()
    if not text:
        return text

    # Already bilingual in "EN_ZH" form.
    if "_" in text:
        left, right = text.split("_", 1)
        if re.search(r"[A-Za-z]", left) and re.search(r"[\u4e00-\u9fff]", right):
            return text

    if "保留规则确定性结论" in text:
        return _make_bilingual_note("Rule conclusion retained", text)
    if "保留被判定为噪声/非评估的建筑" in text:
        return _make_bilingual_note("Protected as noise/non-evaluable building", text)
    if "全流程未识别,小面积归住宅" in text:
        return _make_bilingual_note("No rule recognized; small footprint fallback to residential", text)
    if "强制兜底为商业" in text:
        return _make_bilingual_note("Forced fallback to commercial", text)

    if "ML推断置信度过低" in text:
        return _make_bilingual_note("ML confidence too low", text)

    if "ML覆盖[" in text and "ML多标签混合" in text:
        return _make_bilingual_note("ML override with multi-label mixture", text)
    if "ML覆盖[" in text and "ML回归精准分配" in text:
        return _make_bilingual_note("ML override with regression-based subclass allocation", text)
    if "ML推断填补-" in text and "ML多标签混合" in text:
        return _make_bilingual_note("ML completion with multi-label mixture", text)
    if "ML推断填补-" in text and "ML回归精准分配" in text:
        return _make_bilingual_note("ML completion with regression-based subclass allocation", text)

    # Safe fallback for any unmatched legacy note.
    return _make_bilingual_note("Pipeline decision note", text)


def load_ml_inputs():
    try:
        import pickle
        with open(FEATURE_X_ALL_PATH, 'rb') as f: X_all = pickle.load(f)
        with open(FEATURE_X_TRAIN_ML_PATH, 'rb') as f: X_train_ml = pickle.load(f)
        with open(FEATURE_Y_MULTILABEL_PATH, 'rb') as f: y_multilabel = pickle.load(f)
        with open(FEATURE_BASE_INDEXED_PATH, 'rb') as f: feature_base_indexed = pickle.load(f)
        
        gdf_base_calibrated = gpd.read_file(FEATURE_GDF_BASE_CALIBRATED_PATH)
        df_rule_classified = pd.read_csv(AI_CALIBRATED_OUTPUT_PATH)

        # Normalize BUILDINGSTRUCTUREID types across all inputs to avoid merge/index mismatch.
        if hasattr(X_all, "index"):
            X_all.index = X_all.index.map(normalize_building_id)
        if hasattr(X_train_ml, "index"):
            X_train_ml.index = X_train_ml.index.map(normalize_building_id)
        if hasattr(y_multilabel, "index"):
            y_multilabel.index = y_multilabel.index.map(normalize_building_id)
        if hasattr(feature_base_indexed, "index"):
            feature_base_indexed.index = feature_base_indexed.index.map(normalize_building_id)

        if 'BUILDINGSTRUCTUREID' in gdf_base_calibrated.columns:
            gdf_base_calibrated['BUILDINGSTRUCTUREID'] = gdf_base_calibrated['BUILDINGSTRUCTUREID'].apply(normalize_building_id)
        if 'BUILDINGSTRUCTUREID' in df_rule_classified.columns:
            df_rule_classified['BUILDINGSTRUCTUREID'] = df_rule_classified['BUILDINGSTRUCTUREID'].apply(normalize_building_id)
        
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            KEYWORDS_CONFIG = json.load(f)
            
        keyword_tool = init_keyword_tool()
        
        return {
            'X_all': X_all, 'X_train_ml': X_train_ml, 'y_multilabel': y_multilabel,
            'feature_base_indexed': feature_base_indexed, 'gdf_base_calibrated': gdf_base_calibrated,
            'df_rule_classified': df_rule_classified, 'keyword_tool': keyword_tool
        }
    except Exception as e:
        return None

def train_main_class_models(X_train_ml, y_multilabel, X_all):
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_ml, y_multilabel, test_size=0.2, random_state=42)
    models = {}
    predictions_proba = pd.DataFrame(index=X_all.index)

    performance_metrics = {}    
    
    for main_class in MAIN_CLASSES_ML:
        col_name = f'is_{main_class}'
        if col_name not in y_multilabel.columns: y_multilabel[col_name] = 0
        y_train_s[col_name] = y_multilabel.loc[y_train_s.index, col_name]
        y_val_s[col_name] = y_multilabel.loc[y_val_s.index, col_name]

    for main_class in tqdm(MAIN_CLASSES_ML, desc="Training main-class classifier", leave=True, position=0):
        col_name = f'is_{main_class}'
        y_train_c = y_train_s[col_name]
        try:
            model = XGBClassifier(
                n_estimators=100,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ).fit(X_train_s, y_train_c)
            
            val_preds = model.predict(X_val_s)
            f1 = f1_score(y_val_s[col_name], val_preds, zero_division=0)
            performance_metrics[main_class] = f1
            
            print(classification_report(y_val_s[col_name], val_preds, zero_division=0))

            final_y_class = y_multilabel[col_name]
            final_model = XGBClassifier(
                n_estimators=100,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ).fit(X_train_ml, final_y_class)
            models[main_class] = final_model
            proba = final_model.predict_proba(X_all)
            predictions_proba[f'proba_{main_class}'] = proba[:, 1] if len(proba.shape) == 2 else proba.flatten()
            
        except Exception as e:
            predictions_proba[f'proba_{main_class}'] = 0.0

    if performance_metrics:
        avg_f1 = np.mean(list(performance_metrics.values()))
        for m_class, f1_val in performance_metrics.items():
            print(f"   - {canonicalize_main_label(m_class)}: {f1_val*100:.2f}%")
    
    return models, predictions_proba

def train_subclass_models(X_all, X_train_ml, df_rule_classified, keyword_tool):
    df_subclass_proportions = pd.DataFrame(index=X_all.index)
    rule_df = df_rule_classified.set_index('BUILDINGSTRUCTUREID')
    train_rule = rule_df.reindex(X_train_ml.index)

    for main_class in MAIN_CLASSES_ML:
        SUBCLASSES = sorted([k for k in keyword_tool['keywords_config'].get(main_class, {}).keys()])
        if not SUBCLASSES: continue

        
        target_main = canonicalize_main_label(main_class)
        mask = train_rule['Final_Main_Class'].apply(canonicalize_main_label) == target_main
        X_sub = X_train_ml[mask]
        y_raw = train_rule.loc[mask, 'Final_Sub_Class']
        
        if len(X_sub) < 50:
            continue

        y_sub = pd.DataFrame(0.0, index=X_sub.index, columns=SUBCLASSES)
        for idx, val in y_raw.items():
            if pd.isna(val): 
                y_sub.loc[idx, SUBCLASSES[0]] = 1.0
                continue
            matched = False
            for sub_col in SUBCLASSES:
                if subclass_label_matches(sub_col, val):
                    y_sub.loc[idx, sub_col] = 1.0
                    matched = True
                    break
            if not matched: y_sub.loc[idx, SUBCLASSES[0]] = 1.0

        X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
        regressor = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train_r, y_train_r)

        val_preds = regressor.predict(X_val_r)
        mse = mean_squared_error(y_val_r, val_preds)
        print(f"[INFO] Subclass proportion MSE for {target_main}: {mse:.5f}")

        props = regressor.predict(X_all)
        df_props = pd.DataFrame(props, index=X_all.index, columns=SUBCLASSES)
        def format_proportions(row):
            return {sub: round(val, 2) for sub, val in row.items() if val > SUBCLASS_PROPORTION_THRESHOLD}
        df_subclass_proportions[f'{main_class}_proportions'] = df_props.apply(format_proportions, axis=1)

    return df_subclass_proportions

def apply_ml_calibration(df_rule_classified, predictions_proba, subclass_props, X_all):
    
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
    ml_info_reset = ml_info_table.reset_index()
    ml_info_reset['BUILDINGSTRUCTUREID'] = ml_info_reset['BUILDINGSTRUCTUREID'].apply(normalize_building_id)
    x_area_reset = X_all[['area']].reset_index()
    x_area_reset['BUILDINGSTRUCTUREID'] = x_area_reset['BUILDINGSTRUCTUREID'].apply(normalize_building_id)
    df_rule_classified = df_rule_classified.copy()
    df_rule_classified['BUILDINGSTRUCTUREID'] = df_rule_classified['BUILDINGSTRUCTUREID'].apply(normalize_building_id)
    
    df_to_calibrate = df_rule_classified.merge(ml_info_reset, on='BUILDINGSTRUCTUREID', how='left')
    df_to_calibrate = df_to_calibrate.merge(x_area_reset, on='BUILDINGSTRUCTUREID', how='left')

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

    tqdm.pandas(desc="Applying final calibration and integrity guard")
    final_cols = df_to_calibrate.progress_apply(final_calibration, axis=1, result_type='expand')
    final_cols.columns = ['Calibrated_Main_Class', 'Calibrated_Sub_Class', 'Calibrated_Mix_Proportion', 'Calibrated_Source', 'Calibrated_Notes']
    final_cols['Calibrated_Notes'] = final_cols['Calibrated_Notes'].apply(normalize_calibrated_note_to_bilingual)
    final_cols['Calibrated_Main_Class'] = final_cols['Calibrated_Main_Class'].apply(canonicalize_main_label)
    final_cols['Calibrated_Sub_Class'] = final_cols['Calibrated_Sub_Class'].apply(canonicalize_subclass_label)
    final_cols['Calibrated_Mix_Proportion'] = final_cols['Calibrated_Mix_Proportion'].apply(
        lambda x: normalize_proportion_string(x, canonicalize_subclass_label)
    )
    
    override_mask = final_cols['Calibrated_Notes'].str.contains('ML覆盖', na=False)
    override_df = final_cols[override_mask]
    if len(override_df) > 0:
        reasons = override_df['Calibrated_Notes'].str.extract(r'ML覆盖\[(.*?)\]')[0].value_counts()
        for r, c in reasons.items():
            pass

    return pd.concat([df_to_calibrate.drop(columns=['area'], errors='ignore'), final_cols], axis=1)

def generate_final_output(df_calibrated, gdf_base):
    final_df = df_calibrated.copy()
    
    final_df.to_csv(ML_FINAL_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"[INFO] Step 9 CSV exported: {ML_FINAL_OUTPUT_PATH}")
    
    geojson_path = ML_FINAL_OUTPUT_PATH.replace('.csv', '.geojson')
    print(f"[INFO] Step 9 GeoJSON target: {geojson_path}")
    
    gdf_out = gdf_base[['BUILDINGSTRUCTUREID', 'geometry']].merge(
        final_df, on='BUILDINGSTRUCTUREID', how='inner'
    )
    
    for col in gdf_out.columns:
        if col != 'geometry' and gdf_out[col].dtype == object:
            gdf_out[col] = gdf_out[col].fillna('').astype(str)
            
    gdf_out.to_file(geojson_path, driver='GeoJSON')
    print(f"[INFO] Step 9 GeoJSON exported: {geojson_path}")
    
    return final_df

def print_final_statistics(df):
    print("\n" + "=" * 60)
    print("[INFO] Step 9 summary")
    print("=" * 60)

    main_counts = df['Calibrated_Main_Class'].value_counts(dropna=False)
    print("[INFO] Main class distribution:")
    print(main_counts.to_markdown())

    sub_counts = df['Calibrated_Sub_Class'].value_counts(dropna=False).head(15)
    print("[INFO] Top 15 subclass distribution:")
    print(sub_counts.to_markdown())

    non_eval_mask = df['Calibrated_Main_Class'].astype(str).str.contains('non-assessed|non assessed', case=False, regex=True, na=False)
    mixed_mask = df['Calibrated_Main_Class'].astype(str).str.contains('mixed-use|mixed use', case=False, regex=True, na=False)
    print(f"[INFO] Non-assessed records: {int(non_eval_mask.sum())}")
    print(f"[INFO] Mixed-use records: {int(mixed_mask.sum())}")
    print("\n" + "=" * 60)

def main():
    print("[INFO] Step 9 started: ML calibration")
    data = load_ml_inputs()
    if not data:
        print("[ERROR] Failed to load step 8 artifacts.")
        return False

    print(f"[INFO] Loaded features: total={len(data['X_all'])}, train={len(data['X_train_ml'])}")
    print("[INFO] Training main-class models...")
    models, proba = train_main_class_models(data['X_train_ml'], data['y_multilabel'], data['X_all'])
    print("[INFO] Training subclass models...")
    subclass_props = train_subclass_models(data['X_all'], data['X_train_ml'], data['df_rule_classified'], data['keyword_tool'])

    print("[INFO] Applying ML calibration to step 7 records...")
    df_calibrated = apply_ml_calibration(data['df_rule_classified'], proba, subclass_props, data['X_all'])

    final_df = generate_final_output(df_calibrated, data['gdf_base_calibrated'])
    print_final_statistics(final_df)

    print("[INFO] Step 9 completed.")
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
    log_dir = os.path.join(os.path.dirname(base_dir), "log")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, "9.ml_classification.txt")
    
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout 
    
    print(f"[INFO] Logging to: {log_file_path}")

    try:
        success = main()
        if not success: sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
