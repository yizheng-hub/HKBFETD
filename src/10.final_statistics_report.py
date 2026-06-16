# 10.final_statistics_report.py
# -*- coding: utf-8 -*-


import os
import sys
import re
import json
import math
import io
import pandas as pd
import geopandas as gpd
from collections import defaultdict
import warnings
from contextlib import redirect_stderr

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
LOG_DIR = os.path.join(PROJECT_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)

INPUT_CSV = ML_FINAL_OUTPUT_PATH
INPUT_GEOJSON = ML_FINAL_OUTPUT_PATH.replace(".csv", ".geojson")

PUBLIC_VERSION_TAG = "v1_1"
PUBLIC_CSV = os.path.join(OUTPUT_DIR, f"HK_UBEM_Buildings_Public_{PUBLIC_VERSION_TAG}.csv")
PUBLIC_GEOJSON = os.path.join(OUTPUT_DIR, f"HK_UBEM_Buildings_Public_{PUBLIC_VERSION_TAG}.geojson")

STEP10_MAPPINGS_FILE = os.path.join(os.path.dirname(BASE_DIR), "ctl", "step10_public_mappings.json")
if os.path.exists(STEP10_MAPPINGS_FILE):
    with open(STEP10_MAPPINGS_FILE, "r", encoding="utf-8") as f:
        _step10_mappings = json.load(f)
else:
    _step10_mappings = {}

SUB_TO_MAIN = _step10_mappings.get("SUB_TO_MAIN", {})
MAIN_CLASS_DICT = _step10_mappings.get("MAIN_CLASS_DICT", {})
SUB_CLASS_PATCH_DICT = _step10_mappings.get("SUB_CLASS_PATCH_DICT", {})

RELEASE_LABEL_PATCHES = {
    "Non-Non-manufacturing_非制造业": "Non-manufacturing_非制造业",
    "Non-Non-manufacturing": "Non-manufacturing",
    "Commercial_商业类别": "Other Commercial_其他商业",
    "Industrial_工业类别": "Other Industrial_其他工业",
    "Residential_住宅类别": "Private Housing_私人房屋",
    "非制造业（Non-manufacturing）": "Non-manufacturing_非制造业",
    "Other Government Buildings (Offices, Schools, Hospitals, etc.)_其他政府建筑": "Other Commercial_其他商业",
    "Transport Infrastructure_Transport Infrastructure (交通基础设施)": "Transport Infrastructure_交通基础设施",
    "Temporary/Misc Facilities_Temporary/Misc Facilities (临时/杂项设施)": "Temporary/Miscellaneous Structures_临时/杂项设施",
    "Temporary/Misc Facilities_临时/杂项设施": "Temporary/Miscellaneous Structures_临时/杂项设施",
    "Absolute Noise_绝对噪声相关": "Noise-related Infrastructure_噪声相关基础设施",
    "Absolute Noise Related (绝对噪声相关)": "Noise-related Infrastructure_噪声相关基础设施",
    "车厂_车厂": "Other Commercial_其他商业",
    "车厂": "Other Commercial_其他商业",
    "庙宇": "Other Commercial_其他商业",
}



def normalize_id(x):
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

def is_non_assessed_main_class(main_value: str) -> bool:
    if pd.isna(main_value):
        return False
    s = str(main_value).strip()
    low = s.lower()
    return ("non-assessed" in low) or ("non assessed" in low) or ("非评估" in s)

def canonical_non_assessed_subclass(row: pd.Series) -> str:
    text = " ".join([
        str(row.get("Calibrated_Sub_Class", "")),
        str(row.get("UBEM_Sub_Class_子类别", "")),
        str(row.get("UBEM_Mixed_Proportions_混合比例", "")),
        str(row.get("Calibrated_Source", "")),
        str(row.get("Calibrated_Notes", "")),
        str(row.get("Classification_Source", "")),
        str(row.get("Classification_Notes", "")),
        str(row.get("BUILDINGNAMEEN", "")),
        str(row.get("BUILDINGNAMETC", "")),
        str(row.get("OZP_DESC_ENG", "")),
    ]).lower()

    if any(token in text for token in ["transport", "road", "rail", "flyover", "viaduct", "bridge", "tunnel", "交通", "道路", "铁路", "天桥", "隧道"]):
        return "Transport Infrastructure_交通基础设施"

    if any(token in text for token in ["noise", "sound", "barrier", "outfall", "lighthouse", "噪声", "噪音", "屏障", "排污口", "灯塔", "燈塔"]):
        return "Noise-related Infrastructure_噪声相关基础设施"

    return "Temporary/Miscellaneous Structures_临时/杂项设施"

def canonicalize_main_label(value: str) -> str:
    if pd.isna(value):
        return "Unknown_未知类别"
    s = str(value).strip()
    low = s.lower()
    if ("residential" in low) or ("住宅" in s):
        return "Residential_住宅类别"
    if ("commercial" in low) or ("商业" in s):
        return "Commercial_商业类别"
    if ("industrial" in low) or ("工业" in s):
        return "Industrial_工业类别"
    if ("mixed-use" in low) or ("mixed use" in low) or ("混合" in s):
        return "Mixed-use_混合用途"
    if ("non-assessed" in low) or ("non assessed" in low) or ("非评估" in s):
        return "Non-assessed_非评估类别"
    if ("unknown" in low) or ("未知" in s):
        return "Unknown_未知类别"
    return s

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

def apply_release_label_patches(value):
    if pd.isna(value):
        return value
    res = str(value)
    for old, new in RELEASE_LABEL_PATCHES.items():
        res = res.replace(old, new)
    return res

def repair_subclass_consistency(df: pd.DataFrame):
    df_fixed = df.copy()
    changes = []

    for idx, row in df_fixed.iterrows():
        main_cls = row.get("Calibrated_Main_Class", "")
        old_sub = normalize_label(row.get("Calibrated_Sub_Class", ""))
        mix_prop = row.get("Calibrated_Mix_Proportion", "")
        building_id = row.get("BUILDINGSTRUCTUREID", None)

        if is_non_assessed_main_class(main_cls):
            fixed_sub = canonical_non_assessed_subclass(row)
            fixed_prop = f"{fixed_sub}:1.00"
            if old_sub != normalize_label(fixed_sub) or normalize_label(mix_prop) != normalize_label(fixed_prop):
                df_fixed.at[idx, "Calibrated_Sub_Class"] = fixed_sub
                df_fixed.at[idx, "Calibrated_Mix_Proportion"] = fixed_prop
                changes.append({
                    "BUILDINGSTRUCTUREID": building_id,
                    "Calibrated_Main_Class": main_cls,
                    "Old_Calibrated_Sub_Class": old_sub,
                    "New_Calibrated_Sub_Class": fixed_sub,
                    "Calibrated_Mix_Proportion": fixed_prop
                })
            continue

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
    print(f"[INFO] Subclass consistency repairs applied: {len(changed_df)}")
    return df_fixed, changed_df

def qa_subclass_consistency(df: pd.DataFrame):
    issues = []

    for _, row in df.iterrows():
        main_cls = row.get("Calibrated_Main_Class", "")
        if is_non_assessed_main_class(main_cls):
            current_sub = normalize_label(row.get("Calibrated_Sub_Class", ""))
            expected_sub = normalize_label(canonical_non_assessed_subclass(row))
            expected_prop = normalize_label(f"{expected_sub}:1.00")
            current_prop = normalize_label(row.get("Calibrated_Mix_Proportion", ""))
            if current_sub != expected_sub or current_prop != expected_prop:
                issues.append({
                    "BUILDINGSTRUCTUREID": row.get("BUILDINGSTRUCTUREID", None),
                    "Calibrated_Main_Class": main_cls,
                    "Calibrated_Sub_Class": current_sub,
                    "Dominant_From_Vector": expected_sub,
                    "Calibrated_Mix_Proportion": row.get("Calibrated_Mix_Proportion", "")
                })
            continue

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

def enforce_public_non_assessed_fields(public_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "UBEM_Main_Class_主类别",
        "UBEM_Sub_Class_子类别",
        "UBEM_Mixed_Proportions_混合比例",
    ]
    if not all(col in public_df.columns for col in required_cols):
        return public_df

    df = public_df.copy()
    non_mask = df["UBEM_Main_Class_主类别"].apply(is_non_assessed_main_class)
    if not non_mask.any():
        return df

    for idx, row in df.loc[non_mask].iterrows():
        fixed_sub = canonical_non_assessed_subclass(row)
        df.at[idx, "UBEM_Sub_Class_子类别"] = fixed_sub
        df.at[idx, "UBEM_Mixed_Proportions_混合比例"] = f"{fixed_sub}:1.00"

    return df


def enforce_public_main_subclass_alignment(public_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "UBEM_Main_Class_主类别",
        "UBEM_Sub_Class_子类别",
    ]
    if not all(col in public_df.columns for col in required_cols):
        return public_df

    df = public_df.copy()
    commercial_subclasses = {
        "Accommodation_住宿",
        "Data Centre_数据中心",
        "Education_教育",
        "Food & Beverage_食品及饮品",
        "Human Health_医疗",
        "Office_办公室",
        "Other Commercial_其他商业",
        "Restaurant_食肆",
        "Retail_零售",
        "Transport Infrastructure_交通基础设施",
    }
    industrial_subclasses = {
        "Electronics_电子产品",
        "Food & Beverage_食品及饮品",
        "Metal & Machinery_金属及机械",
        "Non-manufacturing_非制造业",
        "Other Industrial_其他工业",
        "Textile & Wearing Apparel_纺织及服装制品",
    }
    residential_subclasses = {
        "HA Subsidized Sale Flats_房委会资助出售单位",
        "Other Housing_其他房屋",
        "Private Housing_私人房屋",
        "Public Housing_公共房屋",
    }

    for idx, row in df.iterrows():
        main_cls = row.get("UBEM_Main_Class_主类别", "")
        if is_mixed_main_class(main_cls) or is_non_assessed_main_class(main_cls):
            continue

        sub_cls = row.get("UBEM_Sub_Class_子类别", "")
        if sub_cls in residential_subclasses:
            df.at[idx, "UBEM_Main_Class_主类别"] = "Residential_住宅类别"
        elif sub_cls in commercial_subclasses:
            df.at[idx, "UBEM_Main_Class_主类别"] = "Commercial_商业类别"
        elif sub_cls in industrial_subclasses:
            df.at[idx, "UBEM_Main_Class_主类别"] = "Industrial_工业类别"

    return df


def _read_geojson_quiet(path, **kwargs):
    # Suppress noisy GDAL/OGR stderr warnings for unsupported non-scalar fields.
    err_buffer = io.StringIO()
    with redirect_stderr(err_buffer):
        return gpd.read_file(path, **kwargs)


def _write_geojson_quiet(gdf, path):
    # Suppress noisy GDAL/OGR stderr warnings for unsupported non-scalar fields.
    err_buffer = io.StringIO()
    with redirect_stderr(err_buffer):
        gdf.to_file(path, driver='GeoJSON')


def load_and_scan_data():
    if not os.path.exists(INPUT_CSV) or not os.path.exists(INPUT_GEOJSON):
        print("[ERROR] Step 9 outputs are missing. Run step 9 before step 10.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    gdf = _read_geojson_quiet(INPUT_GEOJSON)
    if 'BUILDINGSTRUCTUREID' in gdf.columns:
        gdf = gdf[['BUILDINGSTRUCTUREID', 'geometry']].copy()
    else:
        gdf = gdf[['geometry']].copy()
        gdf['BUILDINGSTRUCTUREID'] = ""

    if 'BUILDINGSTRUCTUREID' in df.columns:
        df['BUILDINGSTRUCTUREID'] = df['BUILDINGSTRUCTUREID'].apply(normalize_id)
    if 'BUILDINGSTRUCTUREID' in gdf.columns:
        gdf['BUILDINGSTRUCTUREID'] = gdf['BUILDINGSTRUCTUREID'].apply(normalize_id)

    if 'SHAPE_Area' not in df.columns:
        df['SHAPE_Area'] = gdf.area

    if 'Calibrated_Main_Class' in df.columns:
        df['Calibrated_Main_Class'] = df['Calibrated_Main_Class'].apply(canonicalize_main_label)

    return df, gdf


def generate_statistics(df):
    total_buildings = len(df)
    main_counts = df['Calibrated_Main_Class'].apply(canonicalize_main_label).value_counts()
    mixed_df = df[df['Calibrated_Main_Class'].apply(is_mixed_main_class)]
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

    total_gfa_all = sum(gfa_by_main.values()) or 1.0
    _ = (total_buildings, main_counts, total_gfa_all, gfa_by_sub)


def _to_safe_geojson_scalar(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def generate_public_dataset(df, gdf):
    df_fixed, _ = repair_subclass_consistency(df)

    qa_issues = qa_subclass_consistency(df_fixed)
    if len(qa_issues) > 0:
        print(f"[WARN] Subclass consistency issues detected: {len(qa_issues)}")

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
        public_df['UBEM_Sub_Class_子类别'] = public_df['UBEM_Sub_Class_子类别'].apply(apply_release_label_patches)

    if 'UBEM_Mixed_Proportions_混合比例' in public_df.columns:
        public_df['UBEM_Mixed_Proportions_混合比例'] = public_df['UBEM_Mixed_Proportions_混合比例'].apply(translate_mix_prop)
        public_df['UBEM_Mixed_Proportions_混合比例'] = public_df['UBEM_Mixed_Proportions_混合比例'].apply(apply_release_label_patches)

    public_df = enforce_public_non_assessed_fields(public_df)
    public_df = enforce_public_main_subclass_alignment(public_df)

    public_df.to_csv(PUBLIC_CSV, index=False, encoding='utf-8-sig')
    print(f"[INFO] Public CSV exported: {PUBLIC_CSV}")

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
            public_gdf[col] = public_gdf[col].apply(_to_safe_geojson_scalar)
            public_gdf[col] = public_gdf[col].fillna('').astype(str).replace('nan', '')

    _write_geojson_quiet(public_gdf, PUBLIC_GEOJSON)
    print(f"[INFO] Public GeoJSON exported: {PUBLIC_GEOJSON}")

    readme_text = """
HK_UBEM_Buildings_Public_v1_1 Dataset
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
    _ = readme_text


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

    try:
        print(f"[INFO] Logging to: {log_file_path}")
        print("[INFO] Step 10 started: final public dataset export")
        df, gdf = load_and_scan_data()
        generate_public_dataset(df, gdf)
        print("[INFO] Step 10 completed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
