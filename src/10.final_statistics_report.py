# 10.final_statistics_report.py
# -*- coding: utf-8 -*-

"""Module documentation."""

import os
import sys
import re
import json
import pandas as pd
import geopandas as gpd
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

from config import OUTPUT_DIR, ML_FINAL_OUTPUT_PATH

conda_prefix = sys.prefix
proj_lib_path = os.path.join(conda_prefix, 'Library', 'share', 'proj')
if os.path.exists(proj_lib_path):
    os.environ['PROJ_LIB'] = proj_lib_path
    os.environ['PROJ_DATA'] = proj_lib_path
else:
    fallback_path = os.path.join(conda_prefix, 'share', 'proj')
    os.environ['PROJ_LIB'] = fallback_path
    os.environ['PROJ_DATA'] = fallback_path
# =================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)

INPUT_CSV = ML_FINAL_OUTPUT_PATH
INPUT_GEOJSON = ML_FINAL_OUTPUT_PATH.replace(".csv", ".geojson")

PUBLIC_CSV = os.path.join(OUTPUT_DIR, "HK_UBEM_Buildings_Public_v1.csv")
PUBLIC_GEOJSON = os.path.join(OUTPUT_DIR, "HK_UBEM_Buildings_Public_v1.geojson")

STEP10_MAPPINGS_FILE = os.path.join(os.path.dirname(BASE_DIR), "ctl", "step10_public_mappings.json")
if os.path.exists(STEP10_MAPPINGS_FILE):
    with open(STEP10_MAPPINGS_FILE, "r", encoding="utf-8") as f:
        _step10_mappings = json.load(f)
else:
    _step10_mappings = {}

SUB_TO_MAIN = _step10_mappings.get("SUB_TO_MAIN", {})
MAIN_CLASS_DICT = _step10_mappings.get("MAIN_CLASS_DICT", {})
SUB_CLASS_PATCH_DICT = _step10_mappings.get("SUB_CLASS_PATCH_DICT", {})



def normalize_id(x):
    """Function documentation."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def normalize_label(label: str) -> str:
    if pd.isna(label):
        return ""
    s = str(label).strip()
    s = s.strip(" -")
    s = s.replace("(", "（").replace(")", "）")
    s = re.sub(r"\s*（\s*", "（", s)
    s = re.sub(r"\s*）\s*", "）", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_mixed_main_class(main_value: str) -> bool:
    if pd.isna(main_value):
        return False
    s = str(main_value).strip().lower()
    return ("mixed" in s) or ("混合" in s)

def parse_proportion_string(prop_str: str):
    if pd.isna(prop_str):
        return {}

    s = str(prop_str).strip()
    if not s or ":" not in s:
        return {}

    parsed = {}
    pattern = re.compile(r'([^:]+?)\s*:\s*([0-9]*\.?[0-9]+)')
    matches = pattern.findall(s)

    for raw_label, raw_value in matches:
        label = normalize_label(raw_label)
        try:
            value = float(raw_value)
        except Exception:
            continue

        if label and value > 0:
            parsed[label] = value

    return parsed

def dominant_subclass_from_vector(prop_str: str):
    parsed = parse_proportion_string(prop_str)
    if not parsed:
        return None
    dominant = sorted(parsed.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return dominant

def patch_orphan_subclass_label(label: str) -> str:
    if pd.isna(label):
        return label
    s = str(label)
    return SUB_CLASS_PATCH_DICT.get(s, s)

def translate_mix_prop(prop_str):
    if pd.isna(prop_str):
        return prop_str
    res = str(prop_str)
    for cn_term, en_cn_term in SUB_CLASS_PATCH_DICT.items():
        res = res.replace(cn_term, en_cn_term)
    return res

def repair_subclass_consistency(df: pd.DataFrame):
    df_fixed = df.copy()
    changes = []

    for idx, row in df_fixed.iterrows():
        main_cls = row.get("Calibrated_Main_Class", "")
        old_sub = normalize_label(row.get("Calibrated_Sub_Class", ""))
        mix_prop = row.get("Calibrated_Mix_Proportion", "")
        building_id = row.get("BUILDINGSTRUCTUREID", None)

        if is_mixed_main_class(main_cls):
            continue

        dominant_sub = dominant_subclass_from_vector(mix_prop)
        if dominant_sub is None:
            continue

        dominant_sub = normalize_label(dominant_sub)

        if old_sub != dominant_sub:
            df_fixed.at[idx, "Calibrated_Sub_Class"] = dominant_sub
            changes.append({
                "BUILDINGSTRUCTUREID": building_id,
                "Calibrated_Main_Class": main_cls,
                "Old_Calibrated_Sub_Class": old_sub,
                "New_Calibrated_Sub_Class": dominant_sub,
                "Calibrated_Mix_Proportion": mix_prop
            })

    changed_df = pd.DataFrame(changes)
    print("[INFO] Status message emitted.")
    return df_fixed, changed_df

def qa_subclass_consistency(df: pd.DataFrame):
    issues = []

    for _, row in df.iterrows():
        main_cls = row.get("Calibrated_Main_Class", "")
        if is_mixed_main_class(main_cls):
            continue

        current_sub = normalize_label(row.get("Calibrated_Sub_Class", ""))
        dominant_sub = dominant_subclass_from_vector(row.get("Calibrated_Mix_Proportion", ""))

        if dominant_sub is None:
            continue

        dominant_sub = normalize_label(dominant_sub)

        if current_sub != dominant_sub:
            issues.append({
                "BUILDINGSTRUCTUREID": row.get("BUILDINGSTRUCTUREID", None),
                "Calibrated_Main_Class": main_cls,
                "Calibrated_Sub_Class": current_sub,
                "Dominant_From_Vector": dominant_sub,
                "Calibrated_Mix_Proportion": row.get("Calibrated_Mix_Proportion", "")
            })

    return pd.DataFrame(issues)


def load_and_scan_data():
    print("=" * 70)
    print("[INFO] Status message emitted.")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV) or not os.path.exists(INPUT_GEOJSON):
        print("[INFO] Status message emitted.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    gdf = gpd.read_file(INPUT_GEOJSON)

    if 'BUILDINGSTRUCTUREID' in df.columns:
        df['BUILDINGSTRUCTUREID'] = df['BUILDINGSTRUCTUREID'].apply(normalize_id)
    if 'BUILDINGSTRUCTUREID' in gdf.columns:
        gdf['BUILDINGSTRUCTUREID'] = gdf['BUILDINGSTRUCTUREID'].apply(normalize_id)

    if 'SHAPE_Area' not in df.columns:
        df['SHAPE_Area'] = gdf.area

    print("[INFO] Status message emitted.")
    return df, gdf


def generate_statistics(df):
    print("\n" + "=" * 70)
    print("[INFO] Status message emitted.")
    print("=" * 70)

    total_buildings = len(df)
    main_counts = df['Calibrated_Main_Class'].value_counts()
    mixed_df = df[df['Calibrated_Main_Class'] == '混合用途']
    _ = mixed_df['Calibrated_Mix_Proportion'].value_counts()

    gfa_by_main = defaultdict(float)
    gfa_by_sub = defaultdict(float)

    for _, row in df.iterrows():
        footprint = float(row.get('SHAPE_Area', 0))
        floors = float(row.get('Estimated_Floors', 1))
        if pd.isna(floors) or floors <= 0:
            floors = 1

        total_gfa = footprint * floors
        mix_prop = str(row.get('Calibrated_Mix_Proportion', ''))

        parsed = parse_proportion_string(mix_prop)

        if not parsed:
            sub = normalize_label(row.get('Calibrated_Sub_Class', '未知类别'))
            main = SUB_TO_MAIN.get(sub, '未知类别')
            gfa_by_sub[sub] += total_gfa
            gfa_by_main[main] += total_gfa
        else:
            for sub, ratio in parsed.items():
                main = SUB_TO_MAIN.get(sub, '未知类别')
                split_gfa = total_gfa * ratio
                gfa_by_sub[sub] += split_gfa
                gfa_by_main[main] += split_gfa

    print("[INFO] Status message emitted.")
    for k, v in main_counts.items():
        print("[INFO] Status message emitted.")

    print("\n" + "-" * 50)
    print("[INFO] Status message emitted.")
    print("-" * 50)

    total_gfa_all = sum(gfa_by_main.values()) or 1.0
    for k, v in sorted(gfa_by_main.items(), key=lambda item: item[1], reverse=True):
        print("[INFO] Status message emitted.")

    print("[INFO] Status message emitted.")
    for k, v in sorted(gfa_by_sub.items(), key=lambda item: item[1], reverse=True):
        if v > 0:
            print("[INFO] Status message emitted.")


def generate_public_dataset(df, gdf):
    print("\n" + "=" * 70)
    print("[INFO] Status message emitted.")
    print("=" * 70)

    df_fixed, change_log = repair_subclass_consistency(df)

    qa_issues = qa_subclass_consistency(df_fixed)
    print("[INFO] Status message emitted.")
    if len(qa_issues) > 0:
        print("[INFO] Status message emitted.")
        print(qa_issues.head(10).to_string(index=False))

    change_log_path = os.path.join(OUTPUT_DIR, "HK_UBEM_Buildings_Public_v1_change_log_from_step10.csv")
    change_log.to_csv(change_log_path, index=False, encoding='utf-8-sig')
    print("[INFO] Status message emitted.")

    export_mapping = {
        'BUILDINGSTRUCTUREID': 'Building_ID_建筑ID',
        'SHAPE_Area': 'Footprint_Area_sqm_占地面积',
        'Estimated_Floors': 'Floors_Count_地上层数',
        'Estimated_Height': 'Height_m_建筑高度',

        'Calibrated_Main_Class': 'UBEM_Main_Class_主类别',
        'Calibrated_Sub_Class': 'UBEM_Sub_Class_子类别',
        'Calibrated_Mix_Proportion': 'UBEM_Mixed_Proportions_混合比例',

        'Calibrated_Source': 'Classification_Source_分类来源',
        'Calibrated_Notes': 'Classification_Notes_分类备注',
        'ML_Confidence': 'ML_Prediction_Confidence_机器学习置信度',
        'proba_住宅类别': 'ML_Prob_Residential_住宅概率',
        'proba_商业类别': 'ML_Prob_Commercial_商业概率',
        'proba_工业类别': 'ML_Prob_Industrial_工业概率',

        'BUILDINGSTRUCTURETYPE': 'LandsD_Structure_Type_地政署建筑类型',
        'CATEGORY': 'LandsD_Category_地政署分类',
        'STATUS': 'LandsD_Status_地政署状态',
        'GROSSFLOORAREA': 'LandsD_Registered_GFA_地政署注册面积',
        'BUILDINGNAMEEN': 'LandsD_Building_Name_EN_地政署英文名',
        'BUILDINGNAMETC': 'LandsD_Building_Name_TC_地政署中文名',
        'BDBIAR_OBJECTID': 'BDBIAR_Record_ID_屋宇署记录ID',
        'BDBIAR_CLASS': 'BDBIAR_Classification_屋宇署分类',
        'BDBIAR_AGE': 'BDBIAR_Building_Age_Year_屋宇署楼龄年份',
        'distance_to_bdbiar': 'Dist_to_BDBIAR_m_距屋宇署点位距离',
        'Matched_POIs': 'Crowdsourced_POIs_Matched_众包POI匹配详情',
        'osmid': 'OSM_Record_ID_OSM记录ID',
        'building': 'OSM_Building_Tag_OSM建筑标签',
        'amenity': 'OSM_Amenity_Tag_OSM设施标签',
        'shop': 'OSM_Shop_Tag_OSM商铺标签',
        'OZP_ZONE_LABEL': 'OZP_Zone_Code_法定图则代码',
        'OZP_DESC_ENG': 'OZP_Zone_Desc_EN_法定图则英文描述'
    }

    export_columns_order = [
        'Building_ID_建筑ID',
        'Footprint_Area_sqm_占地面积',
        'Floors_Count_地上层数',
        'Height_m_建筑高度',

        'UBEM_Main_Class_主类别',
        'UBEM_Sub_Class_子类别',
        'UBEM_Mixed_Proportions_混合比例',

        'Classification_Source_分类来源',
        'Classification_Notes_分类备注',

        'ML_Prediction_Confidence_机器学习置信度',
        'ML_Prob_Residential_住宅概率',
        'ML_Prob_Commercial_商业概率',
        'ML_Prob_Industrial_工业概率',

        'LandsD_Structure_Type_地政署建筑类型',
        'LandsD_Category_地政署分类',
        'LandsD_Status_地政署状态',
        'LandsD_Registered_GFA_地政署注册面积',
        'LandsD_Building_Name_EN_地政署英文名',
        'LandsD_Building_Name_TC_地政署中文名',

        'BDBIAR_Record_ID_屋宇署记录ID',
        'BDBIAR_Classification_屋宇署分类',
        'BDBIAR_Building_Age_Year_屋宇署楼龄年份',
        'Dist_to_BDBIAR_m_距屋宇署点位距离',

        'Crowdsourced_POIs_Matched_众包POI匹配详情',
        'OSM_Record_ID_OSM记录ID',
        'OSM_Building_Tag_OSM建筑标签',
        'OSM_Amenity_Tag_OSM设施标签',
        'OSM_Shop_Tag_OSM商铺标签',

        'OZP_Zone_Code_法定图则代码',
        'OZP_Zone_Desc_EN_法定图则英文描述'
    ]

    print("[INFO] Status message emitted.")

    df_renamed = df_fixed.rename(columns=export_mapping)
    final_cols = [c for c in export_columns_order if c in df_renamed.columns]
    public_df = df_renamed[final_cols].copy()

    if 'UBEM_Main_Class_主类别' in public_df.columns:
        public_df['UBEM_Main_Class_主类别'] = (
            public_df['UBEM_Main_Class_主类别']
            .map(MAIN_CLASS_DICT)
            .fillna(public_df['UBEM_Main_Class_主类别'])
        )

    if 'UBEM_Sub_Class_子类别' in public_df.columns:
        public_df['UBEM_Sub_Class_子类别'] = public_df['UBEM_Sub_Class_子类别'].apply(patch_orphan_subclass_label)

    if 'UBEM_Mixed_Proportions_混合比例' in public_df.columns:
        public_df['UBEM_Mixed_Proportions_混合比例'] = public_df['UBEM_Mixed_Proportions_混合比例'].apply(translate_mix_prop)

    public_df.to_csv(PUBLIC_CSV, index=False, encoding='utf-8-sig')
    print("[INFO] Status message emitted.")

    print("[INFO] Status message emitted.")

    gdf_merge = gdf[['BUILDINGSTRUCTUREID', 'geometry']].copy()
    gdf_merge['_merge_id'] = gdf_merge['BUILDINGSTRUCTUREID'].apply(normalize_id)

    public_df_geo = public_df.copy()
    public_df_geo['_merge_id'] = public_df_geo['Building_ID_建筑ID'].apply(normalize_id)

    public_gdf = gdf_merge.merge(
        public_df_geo,
        on='_merge_id',
        how='inner'
    )

    public_gdf.drop(columns=['BUILDINGSTRUCTUREID', '_merge_id'], inplace=True, errors='ignore')

    for col in public_gdf.columns:
        if col != 'geometry' and public_gdf[col].dtype == object:
            public_gdf[col] = public_gdf[col].fillna('').astype(str).replace('nan', '')

    public_gdf.to_file(PUBLIC_GEOJSON, driver='GeoJSON')
    print("[INFO] Status message emitted.")

    print("\n" + "=" * 70)
    print("[INFO] Status message emitted.")
    print("=" * 70)

    readme_text = """
HK_UBEM_Buildings_Public_v1 Dataset
------------------------------------------------------------
This dataset provides a highly granular, 3D-enriched, and semantically calibrated
building stock model for Hong Kong, designed specifically for Urban Building
Energy Modeling (UBEM) and morphological studies. All attributes and records
are fully bilingual (English & Chinese) to ensure international accessibility.

[Data Column Order & Dictionary]
1. Basic & Spatial Info (基础与空间特征)
- Building_ID_建筑ID               : Unique building identifier from LandsD.
- Footprint_Area_sqm_占地面积      : Building footprint area in square meters.
- Floors_Count_地上层数            : Number of above-ground floors.
- Height_m_建筑高度                : 3D building height in meters.

2. UBEM Classification Results (UBEM 分类结论 - 核心输出)
- UBEM_Main_Class_主类别          : Standardized main functional class.
- UBEM_Sub_Class_子类别           : Dominant sub-functional class for non-mixed records.
- UBEM_Mixed_Proportions_混合比例 : Precise volumetric proportion vector.

3. Classification Reasoning (分类依据与算法溯源)
- Classification_Source_分类来源 : Evidence source (Rule Engine / ML Calibrated / LLM Verified).
- Classification_Notes_分类备注  : Detailed reasoning logic behind the algorithm.
- ML_Prediction_Confidence_机器学习置信度 : Confidence score from the LightGBM model.

4. Raw Evidences (五大开源多模态证据库溯源)
- LandsD_xxx        : Base geometric & land use records from the Lands Department.
- BDBIAR_xxx        : Semantic and building age records from the Buildings Department.
- Crowdsourced_POIs_Matched_众包POI匹配详情 :
  Crowdsourced POI trace aligned with the released proportioning basis.
  For rule-engine-derived commercial proportioning cases, this field reflects
  the POIs effectively used to support the released subclass proportion vector.
- OSM_xxx           : Tag details from OpenStreetMap.
- OZP_xxx           : Outline Zoning Plan codes from the Town Planning Board.

[Coordinate System]
EPSG:2326 (Hong Kong 1980 Grid).
"""
    print(readme_text)


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
    log_file_path = os.path.join(LOG_DIR, "10.final_statistics_report.txt")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout

    print("[INFO] Status message emitted.")

    try:
        df, gdf = load_and_scan_data()
        generate_statistics(df)
        generate_public_dataset(df, gdf)
        print("[INFO] Status message emitted.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
