"""Module documentation."""

import os
import sys
import warnings
import json
import re
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm.auto import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import folium
from pyproj import Transformer
import webbrowser

try:
    from config import (
        INTERMEDIATE_DIR, OUTPUT_DIR, AGGREGATED_GDF_PATH, OFFICIAL_LIB_BASE_PATH,
        KEYWORDS_FILE, RULE_ENGINE_OUTPUT_PATH, RULE_ENGINE_INTERMEDIATE,
        UNKNOWN_MAP_PATH, NUM_MAP_SAMPLES, RUN_RULE_ENGINE_EVALUATION,
        HIGH_CONFIDENCE_SOURCES, LANDSD_CATEGORY_MAP_MAIN, LANDSD_CATEGORY_MAP_SUB,
        GFA_DOMINANT_THRESHOLD, GFA_MIXED_THRESHOLD_LOWER, GFA_MIXED_THRESHOLD_UPPER,
        MIXED_USE_THRESHOLD
    )
except ImportError:
    INTERMEDIATE_DIR = "./intermediate"
    OUTPUT_DIR = "./output"
    AGGREGATED_GDF_PATH = os.path.join(INTERMEDIATE_DIR, "step2_osm_aggregated_buildings.geojson")
    OFFICIAL_LIB_BASE_PATH = os.path.join(INTERMEDIATE_DIR, "step1_official_building_library_base.geojson")
    KEYWORDS_FILE = os.path.join(INTERMEDIATE_DIR, "keywords.json")
    RULE_ENGINE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "step3_rule_engine_classification.csv")
    RULE_ENGINE_INTERMEDIATE = os.path.join(INTERMEDIATE_DIR, "step3_rule_engine_intermediate.csv")
    UNKNOWN_MAP_PATH = os.path.join(OUTPUT_DIR, "rule_engine_unknowns_map.html")
    NUM_MAP_SAMPLES = 500
    RUN_RULE_ENGINE_EVALUATION = True
    HIGH_CONFIDENCE_SOURCES = ["BDBIAR", "LandsD"]
    LANDSD_CATEGORY_MAP_MAIN = {}
    LANDSD_CATEGORY_MAP_SUB = {}
    GFA_DOMINANT_THRESHOLD = 0.9
    GFA_MIXED_THRESHOLD_LOWER = 0.1
    GFA_MIXED_THRESHOLD_UPPER = 0.9
    MIXED_USE_THRESHOLD = 0.1

try:
    from utils import (
        init_keyword_tool, classify_text_by_keywords, classify_from_bdbiar_final,
        safe_str, to_simplified_chinese, classify_osm_feature
    )
except ImportError:
    def init_keyword_tool():
        return {'strong_keywords': set(), 'keyword_regex': {}, 'keyword_map': {}, 'noise_regex': None}
    
    def classify_text_by_keywords(text, regex_map, keyword_map):
        return [], []
    
    def classify_from_bdbiar_final(bdbiar_class):
        return None, None
    
    def safe_str(s):
        return str(s) if pd.notna(s) else ""
    
    def to_simplified_chinese(s):
        return s
    
    def classify_osm_feature(row, keyword_tool, use_case='rule_engine'):
        osm_tags = {
            'industrial': ('工业类别', '其他工业（Other Industrial）'),
            'factory': ('工业类别', '其他工业（Other Industrial）'),
            'warehouse': ('工业类别', '非制造业（Non-manufacturing）'),
            'textile': ('工业类别', '纺织及服装制品（Textile & Wearing Apparel）'),
            'food': ('工业类别', '食品及饮品（Food & Beverage）'),
            'metal': ('工业类别', '金属及机械（Metal & Machinery）'),
            
            'restaurant': ('商业类别', '食肆（Restaurant）'),
            'cafe': ('商业类别', '食肆（Restaurant）'),
            'fast_food': ('商业类别', '食肆（Restaurant）'),
            'bar': ('商业类别', '食肆（Restaurant）'),
            
            'retail': ('商业类别', '零售（Retail）'),
            'shop': ('商业类别', '零售（Retail）'),
            'supermarket': ('商业类别', '零售（Retail）'),
            'mall': ('商业类别', '零售（Retail）'),
            'pharmacy': ('商业类别', '零售（Retail）'),
            
            'office': ('商业类别', '办公室（Office）'),
            
            'hotel': ('商业类别', '住宿（Accommodation）'),
            'hostel': ('商业类别', '住宿（Accommodation）'),
            
            'hospital': ('商业类别', '医疗（Human Health）'),
            'clinic': ('商业类别', '医疗（Human Health）'),
            'clinic': ('商业类别', '医疗（Human Health）'),
            
            'school': ('商业类别', '教育（Education）'),
            'university': ('商业类别', '教育（Education）'),
            'college': ('商业类别', '教育（Education）'),
            'kindergarten': ('商业类别', '教育（Education）'),
            'childcare': ('商业类别', '教育（Education）'),
            
            'datacenter': ('商业类别', '数据中心（Data Centre）'),
            
            'library': ('商业类别', '其他商业（Other Commercial）'),
            'community_centre': ('商业类别', '其他商业（Other Commercial）'),
            'place_of_worship': ('商业类别', '其他商业（Other Commercial）'),
            'police': ('商业类别', '其他商业（Other Commercial）'),
            'fire_station': ('商业类别', '其他商业（Other Commercial）'),
            'substation': ('商业类别', '其他商业（Other Commercial）'),
            'post_office': ('商业类别', '其他商业（Other Commercial）'),
            'townhall': ('商业类别', '其他商业（Other Commercial）'),
            
            'apartments': ('住宅类别', '私人房屋（Private Housing）'),
            'residential': ('住宅类别', '私人房屋（Private Housing）'),
            'house': ('住宅类别', '私人房屋（Private Housing）'),
            'public': ('住宅类别', '公共房屋（Public Housing）'),
            'dormitory': ('住宅类别', '其他房屋（Other Housing）'),
        }
        text = f"{row.get('building', '')} {row.get('amenity', '')} {row.get('shop', '')}".lower()
        for tag, (main, sub) in osm_tags.items():
            if tag in text:
                return main, sub
        return None, None

warnings.filterwarnings('ignore')
tqdm.pandas()

VALID_MAIN_CLASSES = {
    '住宅类别', '商业类别', '工业类别', '非评估类别', '混合用途', '未知类别'
}

VALID_SUB_CLASSES = {
    '公共房屋（Public Housing）',
    '房委会资助出售单位（HA Subsidized Sale Flats）',
    '私人房屋（Private Housing）',
    '其他房屋（Other Housing）',
    
    '食肆（Restaurant）',
    '零售（Retail）',
    '办公室（Office）',
    '住宿（Accommodation）',
    '医疗（Human Health）',
    '教育（Education）',
    '数据中心（Data Centre）',
    '其他商业（Other Commercial）',
    
    '纺织及服装制品（Textile & Wearing Apparel）',
    '非制造业（Non-manufacturing）',
    '食品及饮品（Food & Beverage）',
    '金属及机械（Metal & Machinery）',
    '其他工业（Other Industrial）',
    
    '绝对噪声相关',
    '交通基础设施',
    '临时/杂项设施',
    
    '未知类别'
}

MAIN_CLASS_STANDARD_MAP = {
    '混合用途（住宅+商业）': '混合用途',
    '混合用途': '混合用途',
    '商业类别': '商业类别',
    '住宅类别': '住宅类别',
    '未知类别': '未知类别',
    '工业类别': '工业类别',
    '非评估类别': '非评估类别'
}

SUB_CLASS_STANDARD_MAP = {
    '私人房屋': '私人房屋（Private Housing）',
    '私人房屋（Private House）': '私人房屋（Private Housing）',
    '公共房屋': '公共房屋（Public Housing）',
    '其他房屋': '其他房屋（Other Housing）',
    '房委会资助出售单位': '房委会资助出售单位（HA Subsidized Sale Flats）',
    
    '办公室': '办公室（Office）',
    '零售': '零售（Retail）',
    '医疗': '医疗（Human Health）',
    '其他商业': '其他商业（Other Commercial）',
    '其它商业': '其他商业（Other Commercial）',
    '教育': '教育（Education）',
    '食肆': '食肆（Restaurant）',
    '住宿': '住宿（Accommodation）',
    '数据中心': '数据中心（Data Centre）',
    
    '其他工业': '其他工业（Other Industrial）',
    '其它工业': '其他工业（Other Industrial）',
    '非制造业': '非制造业（Non-manufacturing）',
    '纺织及服装制品': '纺织及服装制品（Textile & Wearing Apparel）',
    '食品及饮品': '食品及饮品（Food & Beverage）',
    '金属及机械': '金属及机械（Metal & Machinery）',
    
    '噪声相关': '绝对噪声相关',
    '交通设施': '交通基础设施',
    '临时设施': '临时/杂项设施',
    '杂项设施': '临时/杂项设施',
    
    '公共房屋（Public Housing）': '公共房屋（Public Housing）',
    '房委会资助出售单位（HA Subsidized Sale Flats）': '房委会资助出售单位（HA Subsidized Sale Flats）',
    '私人房屋（Private Housing）': '私人房屋（Private Housing）',
    '其他房屋（Other Housing）': '其他房屋（Other Housing）',
    '食肆（Restaurant）': '食肆（Restaurant）',
    '零售（Retail）': '零售（Retail）',
    '办公室（Office）': '办公室（Office）',
    '住宿（Accommodation）': '住宿（Accommodation）',
    '医疗（Human Health）': '医疗（Human Health）',
    '教育（Education）': '教育（Education）',
    '数据中心（Data Centre）': '数据中心（Data Centre）',
    '其他商业（Other Commercial）': '其他商业（Other Commercial）',
    '纺织及服装制品（Textile & Wearing Apparel）': '纺织及服装制品（Textile & Wearing Apparel）',
    '非制造业（Non-manufacturing）': '非制造业（Non-manufacturing）',
    '食品及饮品（Food & Beverage）': '食品及饮品（Food & Beverage）',
    '金属及机械（Metal & Machinery）': '金属及机械（Metal & Machinery）',
    '其他工业（Other Industrial）': '其他工业（Other Industrial）',
    '绝对噪声相关': '绝对噪声相关',
    '交通基础设施': '交通基础设施',
    '临时/杂项设施': '临时/杂项设施',
    '未知类别': '未知类别'
}

def standardize_main_class(main_class_name):
    """Function documentation."""
    if pd.isna(main_class_name) or main_class_name is None:
        return '未知类别'
    
    clean_name = str(main_class_name).strip()
    
    standardized_name = MAIN_CLASS_STANDARD_MAP.get(clean_name, '未知类别')
    
    if standardized_name not in VALID_MAIN_CLASSES:
        return '未知类别'
    
    return standardized_name

import re

def normalize_sub_class_name(s):
    """Function documentation."""
    if not isinstance(s, str):
        return ''
    s = s.strip()
    s = s.replace('（', '(').replace('）', ')')
    s = re.sub(r'\s+', ' ', s)
    return s

def standardize_sub_class(sub_class_name, is_mixed_use=False):
    if is_mixed_use:
        if pd.isna(sub_class_name) or sub_class_name is None or sub_class_name == '未知类别':
            return '混合用途'
        parts = str(sub_class_name).split('-')
        valid_parts = []
        for part in parts:
            clean_part = part.split(':')[0].strip() if ':' in part else part.strip()
            std_part = SUB_CLASS_STANDARD_MAP.get(normalize_sub_class_name(clean_part), None)
            if std_part and std_part in VALID_SUB_CLASSES:
                valid_parts.append(std_part)
        return '-'.join(valid_parts) if valid_parts else '混合用途'

    if pd.isna(sub_class_name) or sub_class_name is None:
        return '未知类别'
    
    clean_name = normalize_sub_class_name(sub_class_name)
    
    if clean_name in SUB_CLASS_STANDARD_MAP:
        std = SUB_CLASS_STANDARD_MAP[clean_name]
    else:
        match = re.match(r'^([^()]+)', clean_name)
        if match:
            chinese_part = match.group(1).strip()
            if chinese_part in SUB_CLASS_STANDARD_MAP:
                std = SUB_CLASS_STANDARD_MAP[chinese_part]
            else:
                std = '未知类别'
        else:
            std = '未知类别'
    
    if std not in VALID_SUB_CLASSES:
        return '未知类别'
    return std


def classify_podium_commercial_poi(row, keyword_tool):
    """Function documentation."""
    a = str(row.get('amenity', '')).strip().lower()
    s = str(row.get('shop', '')).strip().lower()

    a_clean = '' if a in ['nan', 'none', 'null', ''] else a
    s_clean = '' if s in ['nan', 'none', 'null', ''] else s

    if not a_clean and not s_clean:
        return False, None, "skip:no_amenity_shop"

    direct_commercial_tags = {
        'restaurant': '食肆（Restaurant）',
        'fast_food': '食肆（Restaurant）',
        'cafe': '食肆（Restaurant）',
        'bakery': '食肆（Restaurant）',

        'retail': '零售（Retail）',
        'convenience': '零售（Retail）',
        'supermarket': '零售（Retail）',
        'mall': '零售（Retail）',
        'optician': '零售（Retail）',
        'eyewear_and_optician': '零售（Retail）',
        'fruits_and_vegetables': '零售（Retail）',

        'bank': '其他商业（Other Commercial）',
        'bank_credit_union': '其他商业（Other Commercial）',
        'real_estate': '其他商业（Other Commercial）',
        'hairdresser': '其他商业（Other Commercial）',
        'hair_salon': '其他商业（Other Commercial）',
        'beauty': '其他商业（Other Commercial）',
        'employment_agencies': '其他商业（Other Commercial）',
    }

    if a_clean in direct_commercial_tags:
        return True, direct_commercial_tags[a_clean], f"amenity:{a_clean}"

    if s_clean in direct_commercial_tags:
        return True, direct_commercial_tags[s_clean], f"shop:{s_clean}"

    # fallback
    m, sub = classify_osm_feature(row, keyword_tool, use_case='rule_engine')
    sub_std = standardize_sub_class(sub)
    if m == '商业类别' and sub_std != '未知类别':
        return True, sub_std, f"fallback:{a_clean or s_clean}"

    return False, None, f"unmapped:{a_clean or s_clean}"



def detect_strong_commercial_podium(info, osm_records, keyword_tool):
    """Function documentation."""
    structure_type = str(
        info.get('BUILDINGSTRUCTURETYPE',
        info.get('STRUCTURETYPE',
        info.get('STRUCTURE_TYPE',
        info.get('LandsD_Structure_Type_地政署建筑类型', ''))))
    ).strip().upper()

    if structure_type != 'P' or osm_records is None or osm_records.empty:
        return False, None, {}

    commercial_sub_counts = Counter()
    valid_functional_poi = 0
    commercial_hits = 0

    for _, row in osm_records.iterrows():
        included, detected_sub, _ = classify_podium_commercial_poi(row, keyword_tool)

        a = str(row.get('amenity', '')).strip().lower()
        s = str(row.get('shop', '')).strip().lower()
        a_clean = '' if a in ['nan', 'none', 'null', ''] else a
        s_clean = '' if s in ['nan', 'none', 'null', ''] else s

        if not a_clean and not s_clean:
            continue

        valid_functional_poi += 1

        if included and detected_sub and detected_sub != '未知类别':
            commercial_hits += 1
            commercial_sub_counts[detected_sub] += 1

    if valid_functional_poi == 0:
        return False, None, {}

    commercial_share = commercial_hits / valid_functional_poi

    if commercial_hits >= 2 and commercial_share >= 0.50:
        dominant_sub = (
            max(commercial_sub_counts.items(), key=lambda x: x[1])[0]
            if commercial_sub_counts else '其他商业（Other Commercial）'
        )
        return True, dominant_sub, dict(commercial_sub_counts)

    return False, None, dict(commercial_sub_counts)



def calculate_area_proportions(info, osm_records=None, keyword_tool=None, return_trace=False):
    """Function documentation."""

    recalc_trace = []
    recalc_counts = {}


    main_proportions = defaultdict(float)
    sub_proportions = defaultdict(float)
    
    is_final_recalc = info.get('IS_FINAL_RECALC', False)
    pre_main = str(info.get('Pre_Class_Main', '')).strip()
    pre_sub = str(info.get('Pre_Class_Sub', '')).strip()
    
    if is_final_recalc and pre_main in ['住宅类别', '商业类别', '工业类别', '非评估类别']:
        main_proportions[pre_main] = 1.0
        
        if pre_sub and pre_sub not in ['nan', '未知类别', '']:
            sub_proportions[pre_sub] = 1.0
        else:
            default_subs = {
                '住宅类别': '私人房屋（Private Housing）',
                '商业类别': '其他商业（Other Commercial）',
                '工业类别': '其他工业（Other Industrial）',
                '非评估类别': '临时/杂项设施'
            }
            sub_proportions[default_subs.get(pre_main, '未知类别')] = 1.0
            
        if pre_main == '商业类别' and osm_records is not None and not osm_records.empty:
            osm_sub_counts = Counter()

            for _, row in osm_records.iterrows():
                included, sub_cls, trace_tag = classify_podium_commercial_poi(row, keyword_tool)

                poi_name = str(row.get('name', '')).strip()
                amenity = str(row.get('amenity', '')).strip()
                shop = str(row.get('shop', '')).strip()
                building = str(row.get('building', '')).strip()

                if included and sub_cls and sub_cls != '未知类别':
                    osm_sub_counts[sub_cls] += 1
                    recalc_trace.append({
                        "name": poi_name,
                        "amenity": amenity,
                        "shop": shop,
                        "building": building,
                        "assigned_sub_class": sub_cls,
                        "trace_rule": trace_tag
                    })

            recalc_counts = dict(osm_sub_counts)

            if sum(osm_sub_counts.values()) > 0:
                sub_proportions.clear()
                total_pois = sum(osm_sub_counts.values())
                for sub_cls, count in osm_sub_counts.items():
                    sub_proportions[str(sub_cls)] = round(count / total_pois, 4)
                    
        if return_trace:
            return dict(main_proportions), dict(sub_proportions), recalc_trace, recalc_counts
        return dict(main_proportions), dict(sub_proportions)

    def get_safe_gfa(val):
        if pd.isna(val) or val is None or str(val).strip() == '' or str(val).lower() == 'nan':
            return 0.0
        try:
            v = float(val)
            import math
            return 0.0 if math.isnan(v) else v
        except:
            return 0.0
            
    dom_gfa = get_safe_gfa(info.get('GFA_DOMESTIC_SUM'))
    nondom_gfa = get_safe_gfa(info.get('GFA_NONDOMESTIC_SUM'))
    total_gfa = dom_gfa + nondom_gfa
    
    if total_gfa <= 0:
        num_floors = 0
        csv_floors = info.get('NUMABOVEGROUNDSTOREYS', 0)
        
        if pd.notna(csv_floors) and str(csv_floors).strip() != '':
            try:
                num_floors = float(csv_floors)
            except ValueError:
                pass
                
        if num_floors <= 0:
            top_height = info.get('TOPHEIGHT', 0)
            base_height = info.get('BASEHEIGHT', 0)
            try:
                th = float(top_height) if pd.notna(top_height) else 0.0
                bh = float(base_height) if pd.notna(base_height) else 0.0
                building_height = th - bh if (th - bh) > 0 else th
                if building_height > 0:
                    FLOOR_HEIGHT_ESTIMATE = 3.0
                    num_floors = max(1, round(building_height / FLOOR_HEIGHT_ESTIMATE))
            except ValueError:
                pass
                
        if num_floors <= 0:
            num_floors = 5 
            
        if num_floors > 1:
            commercial_ratio = 1.0 / num_floors
        else:
            commercial_ratio = 1.0
            
        residential_ratio = 1.0 - commercial_ratio
        
        if residential_ratio > 0:
            main_proportions['住宅类别'] = residential_ratio
        if commercial_ratio > 0:
            main_proportions['商业类别'] = commercial_ratio
        
        if residential_ratio > 0:
            res_sub_class = info.get('Pre_Class_Sub')
            if pd.isna(res_sub_class) or str(res_sub_class).strip() == 'nan' or not res_sub_class:
                res_sub_class = '私人房屋（Private Housing）'
            sub_proportions[str(res_sub_class)] = residential_ratio
        
        if commercial_ratio > 0:
            if osm_records is not None and len(osm_records) > 0:
                osm_sub_counts = Counter()
                for _, row in osm_records.iterrows():
                    if str(row.get('amenity', '')).lower() in ['nan', 'none', 'null', ''] and \
                       str(row.get('shop', '')).lower() in ['nan', 'none', 'null', '']:
                        continue
                    m, s = classify_osm_feature(row, keyword_tool)
                    s_std = standardize_sub_class(s)
                    if m == '商业类别' and s_std != '未知类别':
                        osm_sub_counts[s_std] += 1
                
                total_pois = sum(osm_sub_counts.values())
                if total_pois > 0:
                    for sub_cls, count in osm_sub_counts.items():
                        poi_share = (count / total_pois) * commercial_ratio
                        sub_proportions[str(sub_cls)] = round(poi_share, 4)
                else:
                    sub_proportions['其他商业（Other Commercial）'] = commercial_ratio
            else:
                sub_proportions['其他商业（Other Commercial）'] = commercial_ratio

    else:
        res_ratio = dom_gfa / total_gfa
        nondom_ratio = nondom_gfa / total_gfa
        
        if res_ratio > 0:
            main_proportions['住宅类别'] = res_ratio
            
        if nondom_ratio > 0:
            if osm_records is not None and not osm_records.empty:
                osm_main_classes = []
                osm_weights = []
                for _, row in osm_records.iterrows():
                    m, s = classify_osm_feature(row, keyword_tool)
                    if m in ['商业类别', '工业类别']:
                        osm_main_classes.append(m)
                        weight = float(row.get('osm_weight_score', 1.0)) * (row.get('osm_area', 1.0) or 1.0)
                        osm_weights.append(weight)
                
                if osm_main_classes:
                    total_weight = sum(osm_weights) or 1.0
                    commercial_weight = sum(w for m, w in zip(osm_main_classes, osm_weights) if m == '商业类别')
                    industrial_weight = sum(w for m, w in zip(osm_main_classes, osm_weights) if m == '工业类别')
                    
                    commercial_ratio = (commercial_weight / total_weight) * nondom_ratio
                    industrial_ratio = (industrial_weight / total_weight) * nondom_ratio
                    
                    if commercial_ratio > 0:
                        main_proportions['商业类别'] = commercial_ratio
                    if industrial_ratio > 0:
                        main_proportions['工业类别'] = industrial_ratio
                else:
                    main_proportions['商业类别'] = nondom_ratio
            else:
                main_proportions['商业类别'] = nondom_ratio
                
        if osm_records is not None and not osm_records.empty:
            for main_class in ['住宅类别', '商业类别', '工业类别']:
                if main_class not in main_proportions or main_proportions[main_class] == 0:
                    continue
                
                main_osm_records = osm_records[osm_records.apply(
                    lambda x: classify_osm_feature(x, keyword_tool)[0] == main_class, axis=1
                )]
                
                if main_osm_records.empty:
                    default_sub = {'住宅类别': '私人房屋（Private Housing）', '商业类别': '其他商业（Other Commercial）', '工业类别': '其他工业（Other Industrial）'}.get(main_class)
                    if default_sub:
                        sub_proportions[default_sub] = main_proportions[main_class]
                    continue
                
                sub_counts = Counter()
                total_sub_weight = 0.0
                for _, row in main_osm_records.iterrows():
                    _, s = classify_osm_feature(row, keyword_tool)
                    s_std = standardize_sub_class(s)
                    if s_std != '未知类别':
                        weight = float(row.get('osm_weight_score', 1.0)) * (row.get('osm_area', 1.0) or 1.0)
                        sub_counts[str(s_std)] += weight
                        total_sub_weight += weight
                
                if total_sub_weight > 0:
                    main_ratio = main_proportions[main_class]
                    for sub_class, weight in sub_counts.items():
                        sub_proportions[str(sub_class)] = (weight / total_sub_weight) * main_ratio
                else:
                    default_sub = {'住宅类别': '私人房屋（Private Housing）', '商业类别': '其他商业（Other Commercial）', '工业类别': '其他工业（Other Industrial）'}.get(main_class)
                    if default_sub:
                        sub_proportions[default_sub] = main_proportions[main_class]
        else:
            if '住宅类别' in main_proportions:
                sub_proportions['私人房屋（Private Housing）'] = main_proportions['住宅类别']
            if '商业类别' in main_proportions:
                sub_proportions['其他商业（Other Commercial）'] = main_proportions['商业类别']
            if '工业类别' in main_proportions:
                sub_proportions['其他工业（Other Industrial）'] = main_proportions['工业类别']

    if return_trace:
        return dict(main_proportions), dict(sub_proportions), recalc_trace, recalc_counts
    return dict(main_proportions), dict(sub_proportions)



def format_recalc_trace_for_display(recalc_trace):
    """Function documentation."""
    parts = []
    for item in recalc_trace:
        name = str(item.get("name", "")).strip()
        amenity = str(item.get("amenity", "")).strip()
        shop = str(item.get("shop", "")).strip()
        building = str(item.get("building", "")).strip()
        assigned = str(item.get("assigned_sub_class", "")).strip()

        raw_tag = ""
        if amenity and amenity.lower() not in ["nan", "none", "null", ""]:
            raw_tag = amenity
        elif shop and shop.lower() not in ["nan", "none", "null", ""]:
            raw_tag = shop
        elif building and building.lower() not in ["nan", "none", "null", ""]:
            raw_tag = f"building:{building}"

        label = name if name else "Unnamed_POI"
        detail = f"{raw_tag}->{assigned}" if raw_tag else assigned
        parts.append(f"{label}[{detail}]")

    return " | ".join(parts)


def format_proportion_string(proportions):
    """Function documentation."""
    if not proportions:
        return ""
    
    sorted_props = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{str(k)}:{v:.2f}" for k, v in sorted_props if v > 0 and pd.notna(k) and str(k).strip().lower() != 'nan']
    return "-".join(parts)



def classify_building_rule_engine_optimized(group, keyword_tool):
    """Function documentation."""
    info = group.iloc[0]
    result = {
        'Final_Main_Class': '未知类别', 
        'Final_Sub_Class': '未知类别', 
        'Classification_Source': 'Unknown', 
        'Is_Mixed_Use': False, 
        'Confidence_Score': 0.0,
        'Evidence_Details': [], 
        'Is_Conflicted': False, 
        'Main_Class_Proportions': '',  
        'Sub_Class_Proportions': '',
        'Matched_POIs': '',
        'Final_Recalc_Commercial_POI_Trace': '',
        'Final_Recalc_Commercial_POI_Counts': ''
    }
    
    evidences = []
    STRONG_KEYWORDS = keyword_tool.get('strong_keywords', set())

    LANDSD_MAP = {
        '1': {'main': '住宅类别', 'sub': '私人房屋（Private Housing）', 'desc': '合法私人建筑'},
        '2': {'main': '住宅类别', 'sub': '私人房屋（Private Housing）', 'desc': '新界小型屋宇'},
        '3': {'main': '住宅类别', 'sub': '公共房屋（Public Housing）', 'desc': '房委会建筑'},
        '4': {'main': '商业类别', 'sub': '其他商业（Other Commercial）', 'desc': '政府物业'},
        '5': {'main': None, 'sub': None, 'desc': '杂项结构需具体分析'},
        '9': {'main': None, 'sub': None, 'desc': '未分类'}
    }

    osm_records = group.dropna(subset=['osmid'])
    main_props, sub_props = calculate_area_proportions(info, osm_records, keyword_tool)
    result['Main_Class_Proportions'] = format_proportion_string(main_props)
    result['Sub_Class_Proportions'] = format_proportion_string(sub_props)

    def get_safe_gfa(val):
        if pd.isna(val) or val is None or str(val).strip() == '' or str(val).lower() == 'nan':
            return 0.0
        try:
            v = float(val)
            import math
            return 0.0 if math.isnan(v) else v
        except:
            return 0.0
            
    dom_gfa = get_safe_gfa(info.get('GFA_DOMESTIC_SUM'))
    nondom_gfa = get_safe_gfa(info.get('GFA_NONDOMESTIC_SUM'))
    total_gfa = dom_gfa + nondom_gfa

    if total_gfa > 20:
        dom_ratio = dom_gfa / total_gfa
        if dom_ratio > 0.99:
            evidences.append({'src': 'GFA', 'main': '住宅类别', 'sub': '私人房屋（Private Housing）', 'conf': 1.0, 'weight': 100})
            result['Main_Class_Proportions'] = '住宅类别:1.00'
            result['Sub_Class_Proportions'] = '私人房屋（Private Housing）:1.00'
        elif dom_ratio < 0.01:
            evidences.append({'src': 'GFA', 'main': '商业类别', 'sub': None, 'conf': 0.9, 'weight': 80, 'note': '非住宅GFA'})
        else:
            evidences.append({'src': 'GFA', 'main': '混合用途', 'sub': None, 'conf': 1.0, 'weight': 100})
            result['Is_Mixed_Use'] = True

    bdb_main, bdb_sub = classify_from_bdbiar_final(info.get('BDBIAR_CLASS'))
    if bdb_main:
        evidences.append({'src': 'BDBIAR', 'main': standardize_main_class(bdb_main), 'sub': standardize_sub_class(bdb_sub), 'conf': 0.6, 'weight': 60})

    if not osm_records.empty:
        osm_votes = []
        osm_sub_votes = defaultdict(float)
        total_osm_weight = 0.0
        
        poi_details = []
        
        for _, row in osm_records.iterrows():
            poi_name = str(row.get('name', ''))
            
            a = str(row.get('amenity', '')).strip()
            s = str(row.get('shop', '')).strip()
            b = str(row.get('building', '')).strip()
            
            a_clean = '' if a.lower() in ['nan', 'none', 'null', ''] else a
            s_clean = '' if s.lower() in ['nan', 'none', 'null', ''] else s
            b_clean = '' if b.lower() in ['nan', 'none', 'null', ''] else b
            
            if poi_name and str(poi_name).strip().lower() not in ['nan', 'none', 'null', '']:
                poi_type = a_clean or s_clean or b_clean
                if poi_type:
                    poi_details.append(f"{poi_name}[{poi_type}]")
                
            m, s_sub = classify_osm_feature(row, keyword_tool, use_case='rule_engine')
            
            if not a_clean and not s_clean:
                if b_clean in ['apartments', 'residential', 'house', 'yes']:
                    m, s_sub = '住宅类别', '私人房屋（Private Housing）'
                    if '邨' in poi_name or '邨' in str(info.get('BUILDINGNAMETC', '')):
                        s_sub = '公共房屋（Public Housing）'
                elif not b_clean:
                    m = '未知类别'
            m_std = standardize_main_class(m)
            s_std = standardize_sub_class(s_sub)
            w = float(row.get('osm_weight_score', 0.5))
            
            if m_std and m_std != '未知类别':
                osm_votes.append((m_std, w))
                if s_std and s_std != '未知类别':
                    osm_sub_votes[s_std] += w
                total_osm_weight += w
        
        result['Matched_POIs'] = " | ".join(poi_details) if poi_details else "无命名POI"
        
        if osm_votes:
            weighted_counts = {}
            for m, w in osm_votes:
                weighted_counts[m] = weighted_counts.get(m, 0) + w
            
            top_main = max(weighted_counts, key=weighted_counts.get)
            final_weight = 75 + min(total_osm_weight * 20, 20)
            if total_osm_weight < 0.1: final_weight = 40
            
            top_sub = max(osm_sub_votes, key=osm_sub_votes.get) if osm_sub_votes else None
            evidences.append({'src': 'OSM', 'main': top_main, 'sub': top_sub, 'conf': min(total_osm_weight, 0.9), 'weight': final_weight})

    cat_code = str(info.get('CATEGORY', ''))
    landsd_info = LANDSD_MAP.get(cat_code, {'main': None})
    if landsd_info['main'] is not None:
        weight = 50
        if landsd_info['main'] == '非评估类别':
            weight = 30 if len(evidences) > 0 else 60
        elif cat_code == '5':
            building_name = str(info.get('BUILDINGNAMEEN', '')).lower()
            if any(term in building_name for term in ['noise', 'barrier', 'sound']):
                landsd_info['main'], landsd_info['sub'] = '非评估类别', '绝对噪声相关'
            elif any(term in building_name for term in ['road', 'highway', 'rail', 'station', 'transport']):
                landsd_info['main'], landsd_info['sub'] = '非评估类别', '交通基础设施'
            elif any(term in building_name for term in ['temporary', 'temp', 'misc']):
                landsd_info['main'], landsd_info['sub'] = '非评估类别', '临时/杂项设施'
            else:
                landsd_info['main'], landsd_info['sub'] = '商业类别', '其他商业（Other Commercial）'
            weight = 60 if landsd_info['main'] != '非评估类别' else 40
        elif cat_code == '9': weight = 30
        
        evidences.append({'src': 'LandsD_Category', 'main': landsd_info['main'], 'sub': landsd_info['sub'], 'conf': weight/100.0, 'weight': weight})

    all_text_sources = [
        safe_str(info.get('BUILDINGNAMETC', '')), safe_str(info.get('BUILDINGNAMEEN', '')),
        safe_str(info.get('ALL_INFO_DESC', '')), safe_str(info.get('BDBIAR_ADDRESS_C', '')), safe_str(info.get('name', ''))
    ]
    text_for_kw = " ".join([t for t in all_text_sources if t.strip()])
    kws, clfs = classify_text_by_keywords(text_for_kw, keyword_tool['keyword_regex'], keyword_tool['keyword_map'])

    if clfs:
        best_kw_evidence = None
        max_kw_weight = 0
        text_lower = text_for_kw.lower()
        
        for main, sub in clfs:
            main_std, sub_std = standardize_main_class(main), standardize_sub_class(sub)
            current_weight = 50
            
            if main_std == '非评估类别':
                functional_indicators = [r'\bbuilding\b', r'\btower\b', r'\bblock\b', r'\bhall\b', r'\bhouse\b', r'\bcentre\b', r'大厦', r'大楼', r'中心', r'馆']
                has_functional_indicator = any(re.search(p, text_lower, re.IGNORECASE) for p in functional_indicators)
                current_weight = 30 if has_functional_indicator and float(info.get('SHAPE_Area', 0)) > 50 else 65
            elif main_std == '商业类别' and any(term in text_lower for term in ['commercial', '商業', 'office', '寫字樓', 'centre']):
                current_weight = 80
            elif main_std == '工业类别' and any(term in text_lower for term in ['industrial', '工業', 'factory', '工廠', 'warehouse']):
                current_weight = 80
            
            if any(k in STRONG_KEYWORDS for k in kws): current_weight = 90
            
            if main_std == '商业类别' and cat_code in ['1', '2', '3'] and current_weight == 50:
                current_weight = 10
            
            if current_weight > max_kw_weight:
                max_kw_weight = current_weight
                best_kw_evidence = {'src': 'Keyword', 'main': main_std, 'sub': sub_std, 'conf': current_weight/100.0, 'weight': current_weight}
        
        if best_kw_evidence: evidences.append(best_kw_evidence)

    if cat_code == '9' and len(evidences) == 0:
        text_for_name = f"{info.get('BUILDINGNAMETC', '')} {info.get('BUILDINGNAMEEN', '')}".lower()
        name_keywords = {
            'station': ('非评估类别', '交通基础设施', 65), 'restaurant': ('商业类别', '食肆（Restaurant）', 70),
            'school': ('商业类别', '教育（Education）', 70), 'hospital': ('商业类别', '医疗（Human Health）', 70),
            'garden': ('住宅类别', '私人房屋（Private Housing）', 60), 'estate': ('住宅类别', '私人房屋（Private Housing）', 60),
            'factory': ('工业类别', '其他工业（Other Industrial）', 70), 'warehouse': ('工业类别', '非制造业（Non-manufacturing）', 70),
            'noise': ('非评估类别', '绝对噪声相关', 70), 'temporary': ('非评估类别', '临时/杂项设施', 70)
        }
        best_match = max([{'src': 'Name_Keyword', 'main': m, 'sub': s, 'conf': w/100.0, 'weight': w} 
                          for k, (m, s, w) in name_keywords.items() if k in text_for_name], 
                         key=lambda x: x['weight'], default=None)
        if best_match: evidences.append(best_match)

    if any(ev['src'] == 'GFA' and ev['main'] == '混合用途' for ev in evidences):
        result['Final_Main_Class'] = '混合用途'
        result['Is_Mixed_Use'] = True
        significant_subs = {k: v for k, v in sub_props.items() if v > 0.01}
        if significant_subs:
            result['Sub_Class_Proportions'] = format_proportion_string(significant_subs)
            sub_names = [str(k) for k, v in sorted(significant_subs.items(), key=lambda x: x[1], reverse=True) if pd.notna(k) and str(k).strip().lower() != 'nan']
            result['Final_Sub_Class'] = "-".join(sub_names) if sub_names else '混合用途'
        else:
            result['Sub_Class_Proportions'], result['Final_Sub_Class'] = '混合用途:1.00', '混合用途'
        
        result['Classification_Source'] = 'GFA_Mixed'
        result['Confidence_Score'] = 0.95
        result['Evidence_Details'] = str([f"{e['src']}:{e['main']}({e['weight']})" for e in evidences])
        return pd.Series(result)

    if not evidences: return pd.Series(result)

    score_board = defaultdict(float)
    for item in evidences: score_board[item['main']] += item['weight']
    
    if sum(score_board.values()) < 30:
        default_map = {'1': ('住宅类别', '私人房屋（Private Housing）'), '2': ('住宅类别', '私人房屋（Private Housing）'), '3': ('住宅类别', '公共房屋（Public Housing）'), '4': ('商业类别', '其他商业（Other Commercial）')}
        result['Final_Main_Class'], result['Final_Sub_Class'] = default_map.get(cat_code, ('未知类别', '未知类别'))
        result['Classification_Source'], result['Confidence_Score'] = '低证据推断', 0.4
        result['Evidence_Details'] = str([f"{e['src']}:{e['main']}({e['weight']})" for e in evidences])
        if result['Final_Main_Class'] != '未知类别':
            result['Main_Class_Proportions'] = f"{result['Final_Main_Class']}:1.00"
            result['Sub_Class_Proportions'] = f"{result['Final_Sub_Class']}:1.00"
        return pd.Series(result)

    sorted_scores = sorted(score_board.items(), key=lambda x: x[1], reverse=True)
    winner_main, winner_score = sorted_scores[0]
    is_conflicted = False
    forced_sub_class = None

    if len(sorted_scores) > 1:
        runner_up_main, runner_up_score = sorted_scores[1]
        if winner_main != '混合用途' and runner_up_main != '混合用途' and winner_score < runner_up_score * 1.5:
            is_conflicted = True
            winner_main = '混合用途'
            result['Is_Mixed_Use'] = True

    # =========================
    # =========================
    override_to_commercial, podium_sub, podium_counts = detect_strong_commercial_podium(
        info, osm_records, keyword_tool
    )

    if winner_main == '住宅类别' and override_to_commercial:
        winner_main = '商业类别'
        forced_sub_class = podium_sub or '其他商业（Other Commercial）'
        winner_score = max(winner_score, 95)
        is_conflicted = False
        result['Is_Mixed_Use'] = False

        evidences.append({
            'src': 'Podium_Commercial_Override',
            'main': '商业类别',
            'sub': forced_sub_class,
            'conf': 0.95,
            'weight': 95
        })

    if winner_main == '混合用途':
        result['Final_Sub_Class'] = '混合用途'
        source_str = "Fusion_Inferred_Mixed"
    else:
        supporting_evidences = [e for e in evidences if e['main'] == winner_main]

        if forced_sub_class:
            result['Final_Sub_Class'] = standardize_sub_class(forced_sub_class)
        else:
            valid_subs = [e for e in supporting_evidences if e['sub'] and e['sub'] != '未知类别']
            if valid_subs:
                best_sub = max(valid_subs, key=lambda x: x['weight']).get('sub')
                result['Final_Sub_Class'] = standardize_sub_class(best_sub)
            else:
                result['Final_Sub_Class'] = {
                    '住宅类别': '私人房屋（Private Housing）',
                    '商业类别': '其他商业（Other Commercial）',
                    '工业类别': '其他工业（Other Industrial）'
                }.get(winner_main, '未知类别')

        source_str = "+".join(sorted(list(set(e['src'] for e in supporting_evidences)))) if supporting_evidences else "Fusion_Inferred"
    
    info_dict = info.to_dict()
    info_dict['Pre_Class_Main'] = winner_main
    info_dict['Pre_Class_Sub'] = info.get('Pre_Class_Sub', '未知类别') if winner_main == '混合用途' else result['Final_Sub_Class']
    info_dict['IS_FINAL_RECALC'] = True
    
    osm_records_for_calc = group.dropna(subset=['osmid'])
    main_props, sub_props, recalc_trace, recalc_counts = calculate_area_proportions(
        info_dict,
        osm_records=osm_records_for_calc,
        keyword_tool=keyword_tool,
        return_trace=True
    )

    result['Final_Recalc_Commercial_POI_Trace'] = json.dumps(recalc_trace, ensure_ascii=False)
    result['Final_Recalc_Commercial_POI_Counts'] = json.dumps(recalc_counts, ensure_ascii=False)

    result['Matched_POIs'] = format_recalc_trace_for_display(recalc_trace)

    result['Main_Class_Proportions'] = format_proportion_string(main_props)
    result['Sub_Class_Proportions'] = format_proportion_string(sub_props)

    if winner_main == '混合用途':
        significant_subs = {k: v for k, v in sub_props.items() if v > 0}
        sub_names = [
            str(k) for k, v in sorted(significant_subs.items(), key=lambda x: x[1], reverse=True)
            if pd.notna(k) and str(k).strip().lower() != 'nan'
        ]
        result['Final_Sub_Class'] = "-".join(sub_names) if sub_names else '混合用途'
    else:
        significant_subs = {k: v for k, v in sub_props.items() if v > 0}
        if significant_subs:
            dominant_sub = max(significant_subs.items(), key=lambda x: x[1])[0]
            result['Final_Sub_Class'] = standardize_sub_class(dominant_sub)
    
    normalized_conf = min(winner_score / 120.0, 0.99) * (0.7 if is_conflicted else 1.0)
    result.update({
        'Final_Main_Class': standardize_main_class(winner_main),
        'Final_Sub_Class': result['Final_Sub_Class'] if winner_main == '混合用途' else standardize_sub_class(result['Final_Sub_Class']),
        'Classification_Source': source_str,
        'Confidence_Score': round(normalized_conf, 2),
        'Is_Conflicted': is_conflicted,
        'Evidence_Details': str([f"{e['src']}:{e['main']}({e['weight']})" for e in evidences])
    })

    return pd.Series(result)





def load_rule_engine_inputs():
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    try:
        print("[INFO] Status message emitted.")
        gdf_aggregated = gpd.read_file(AGGREGATED_GDF_PATH)
        
        if gdf_aggregated.crs is None:
            gdf_aggregated = gdf_aggregated.set_crs("EPSG:2326", allow_override=True)
        print("[INFO] Status message emitted.")
        
        print("[INFO] Status message emitted.")
        gdf_official_library_full = gpd.read_file(OFFICIAL_LIB_BASE_PATH)
        gdf_official_library_full = gdf_official_library_full.set_crs("EPSG:2326", allow_override=True)
        print("[INFO] Status message emitted.")
        
        print("[INFO] Status message emitted.")
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            KEYWORDS_CONFIG = json.load(f)
        print("[INFO] Status message emitted.")
        
        return {
            'gdf_aggregated': gdf_aggregated,
            'gdf_official_library_full': gdf_official_library_full,
            'keywords_config': KEYWORDS_CONFIG
        }
        
    except FileNotFoundError as e:
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        print(f"  1. {AGGREGATED_GDF_PATH}")
        print(f"  2. {OFFICIAL_LIB_BASE_PATH}")
        print(f"  3. {KEYWORDS_FILE}")
        return None
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()
        return None

def run_rule_engine(gdf_aggregated, gdf_official_library_full, keyword_tool):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    gdf_aggregated['Classification_Stage'] = '待评估'
    
    noise_regex = keyword_tool.get('noise_regex')
    name_str = (gdf_aggregated['BUILDINGNAMETC'].fillna('') + ' ' + gdf_aggregated['BUILDINGNAMEEN'].fillna('')).str.lower().apply(to_simplified_chinese)
    kw_mask = name_str.str.contains(noise_regex, regex=True, na=False) if noise_regex else pd.Series(False, index=gdf_aggregated.index)
    
    is_tiny_area = gdf_aggregated['geometry'].area < 5
    has_bdbiar = gdf_aggregated['BDBIAR_OBJECTID'].notna()
    has_osm = gdf_aggregated['osmid'].notna()
    has_valid_name = (gdf_aggregated['BUILDINGNAMETC'].str.len() > 1) | (gdf_aggregated['BUILDINGNAMEEN'].str.len() > 2)
    
    pre_screen_mask = kw_mask | (is_tiny_area & ~has_bdbiar & ~has_osm & ~has_valid_name)
    
    gdf_aggregated.loc[pre_screen_mask, 'Classification_Stage'] = '已预筛选'
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    
    gdf_aggregated['Pre_Class_Main'] = gdf_aggregated['CATEGORY'].astype(str).map(LANDSD_CATEGORY_MAP_MAIN)
    gdf_aggregated['Pre_Class_Sub'] = gdf_aggregated['CATEGORY'].astype(str).map(LANDSD_CATEGORY_MAP_SUB)
    
    assessable_data = gdf_aggregated[gdf_aggregated['Classification_Stage'] == '待评估'].copy()
    print("[INFO] Status message emitted.")
    
    tqdm.pandas(desc="规则引擎分类")
    rule_engine_results = assessable_data.groupby('BUILDINGSTRUCTUREID').progress_apply(
        lambda group: classify_building_rule_engine_optimized(group, keyword_tool)
    )

    print("[INFO] Status message emitted.")
    df_rule_classified_final = gdf_official_library_full[['BUILDINGSTRUCTUREID', 'geometry']].copy()

    df_rule_classified_final = df_rule_classified_final.merge(
        gdf_aggregated.drop(columns=['geometry']).drop_duplicates('BUILDINGSTRUCTUREID'), 
        on='BUILDINGSTRUCTUREID', how='left'
    )

    df_rule_classified_final = df_rule_classified_final.merge(
        rule_engine_results, on='BUILDINGSTRUCTUREID', how='left', suffixes=('_base', '_result')
    )

    pre_screen_mask_final = df_rule_classified_final['Classification_Stage'] == '已预筛选'
    df_rule_classified_final.loc[pre_screen_mask_final, 'Final_Main_Class'] = '非评估类别'
    df_rule_classified_final.loc[pre_screen_mask_final, 'Final_Sub_Class'] = '绝对噪声相关'
    df_rule_classified_final.loc[pre_screen_mask_final, 'Classification_Source'] = 'Pre-Screening'
    df_rule_classified_final.loc[pre_screen_mask_final, 'Is_Mixed_Use'] = False
    df_rule_classified_final.loc[pre_screen_mask_final, 'Main_Class_Proportions'] = '非评估类别:1.00'
    df_rule_classified_final.loc[pre_screen_mask_final, 'Sub_Class_Proportions'] = '绝对噪声相关:1.00'

    unknown_mask = df_rule_classified_final['Final_Main_Class'].isna() | (df_rule_classified_final['Final_Main_Class'] == '未知类别')
    df_rule_classified_final.loc[unknown_mask, 'Final_Main_Class'] = '未知类别'
    df_rule_classified_final.loc[unknown_mask, 'Final_Sub_Class'] = '未知类别'
    df_rule_classified_final.loc[unknown_mask, 'Classification_Source'] = 'Unknown'

    print("[INFO] Status message emitted.")
    def post_process(row):
        main = row['Final_Main_Class']
        sub = row['Final_Sub_Class']
        area = float(row.get('SHAPE_Area', 0))
        is_mixed = row.get('Is_Mixed_Use', False)
        
        if is_mixed:
            return pd.Series([
                main,
                sub,
                row['Classification_Source'],
                row.get('Is_Conflicted', False)
            ])
        
        if main == '非评估类别':
            name = str(row.get('BUILDINGNAMEEN', '')).lower() + ' ' + str(row.get('BUILDINGNAMETC', '')).lower()
            if any(term in name for term in ['noise', 'sound', 'barrier']):
                sub = '绝对噪声相关'
            elif any(term in name for term in ['road', 'rail', 'station', 'transport']):
                sub = '交通基础设施'
            elif any(term in name for term in ['temporary', 'temp', 'misc']):
                sub = '临时/杂项设施'
        
        return pd.Series([
            standardize_main_class(main),
            standardize_sub_class(sub),
            row['Classification_Source'],
            row.get('Is_Conflicted', False)
        ])
    
    tqdm.pandas(desc="后处理修正")
    post_processed = df_rule_classified_final.progress_apply(post_process, axis=1)
    df_rule_classified_final[['Final_Main_Class', 'Final_Sub_Class', 'Classification_Source', 'Is_Conflicted']] = post_processed

    print("[INFO] Status message emitted.")
    mixed_mask = df_rule_classified_final['Is_Mixed_Use'] == True
    non_mixed_mask = ~mixed_mask
    df_rule_classified_final.loc[non_mixed_mask, 'Final_Main_Class'] = df_rule_classified_final.loc[non_mixed_mask, 'Final_Main_Class'].apply(standardize_main_class)
    df_rule_classified_final.loc[non_mixed_mask, 'Final_Sub_Class'] = df_rule_classified_final.loc[non_mixed_mask, 'Final_Sub_Class'].apply(standardize_sub_class)

    cols_to_drop = [c for c in df_rule_classified_final.columns if c.endswith('_base') or c.endswith('_result')]
    df_rule_classified_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    output_cols = [c for c in df_rule_classified_final.columns if c != 'geometry']
    df_rule_classified_final[output_cols].to_csv(
        RULE_ENGINE_OUTPUT_PATH, index=False, encoding='utf-8-sig'
    )
    
    geojson_output_path = RULE_ENGINE_OUTPUT_PATH.replace('.csv', '.geojson')
    print("[INFO] Status message emitted.")
    
    gdf_export = df_rule_classified_final.copy()
    for col in gdf_export.columns:
        if gdf_export[col].dtype == object and col != 'geometry':
            gdf_export[col] = gdf_export[col].astype(str)
            
    gdf_export.to_file(geojson_output_path, driver='GeoJSON')
    print("[INFO] Status message emitted.")






    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print(df_rule_classified_final['Final_Main_Class'].value_counts(dropna=False).to_markdown())

    print("[INFO] Status message emitted.")
    print(df_rule_classified_final['Final_Sub_Class'].value_counts(dropna=False).head(10).to_markdown())

    print("[INFO] Status message emitted.")

    return df_rule_classified_final

def evaluate_rule_engine_results(df_rule_classified_final, gdf_official_library):
    """Function documentation."""
    if not RUN_RULE_ENGINE_EVALUATION:
        print("[INFO] Status message emitted.")
        return
    
    print("[INFO] Status message emitted.")
    
    try:
        if 'geometry' not in df_rule_classified_final.columns or df_rule_classified_final.geometry.isna().all():
            df_to_evaluate = pd.read_csv(RULE_ENGINE_OUTPUT_PATH)
            if not gdf_official_library.empty:
                df_to_evaluate = df_to_evaluate.merge(
                    gdf_official_library[['BUILDINGSTRUCTUREID', 'geometry']], 
                    on='BUILDINGSTRUCTUREID', how='left'
                )
                df_to_evaluate = gpd.GeoDataFrame(df_to_evaluate, geometry='geometry', crs=gdf_official_library.crs)
        else:
            df_to_evaluate = df_rule_classified_final.copy()
        
        unknown_buildings = df_to_evaluate[df_to_evaluate['Final_Main_Class'] == '未知类别']
        print("[INFO] Status message emitted.")
        
        if len(unknown_buildings) > 0 and 'geometry' in unknown_buildings.columns:
            visualize_unknowns_on_map(unknown_buildings)
        
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()

def visualize_unknowns_on_map(unknown_gdf, num_samples=NUM_MAP_SAMPLES):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    samples = unknown_gdf.sample(min(num_samples, len(unknown_gdf)), random_state=42)
    transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
    
    try:
        center_geom = samples.union_all().centroid
        center_lon, center_lat = transformer.transform(center_geom.x, center_geom.y)
    except Exception:
        center_lat, center_lon = 22.35, 114.1
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    folium.TileLayer(
        'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', 
        attr='Google', name='Google Satellite', overlay=True
    ).add_to(m)
    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
    folium.LayerControl().add_to(m)
    
    print("[INFO] Status message emitted.")
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        if row.geometry is None or row.geometry.is_empty:
            continue
        
        if hasattr(row.geometry, 'exterior'):
            coords = [transformer.transform(x, y)[::-1] for x, y in row.geometry.exterior.coords]
        else:
            coords = [transformer.transform(x, y)[::-1] for x, y in row.geometry.coords]
        
        popup_html = f"""
        <h4>未知类别建筑</h4><hr>
        <b>BSID:</b> {row['BUILDINGSTRUCTUREID']}<br>
        <b>主类别:</b> {row['Final_Main_Class']}<br>
        <b>子类别:</b> {row['Final_Sub_Class']}<br>
        <b>地政署Category:</b> {row.get('CATEGORY', 'N/A')}<br>
        <b>面积 (㎡):</b> {row.geometry.area:.2f}<br>
        <b>比例:</b> {row.get('Sub_Class_Proportions', 'N/A')}
        """
        
        folium.Polygon(
            locations=coords,
            popup=folium.Popup(popup_html, max_width=400),
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=0.7,
            tooltip=f"BSID: {row['BUILDINGSTRUCTUREID']}"
        ).add_to(m)
    
    m.save(UNKNOWN_MAP_PATH)
    print("[INFO] Status message emitted.")
    
    try:
        webbrowser.open(UNKNOWN_MAP_PATH)
    except Exception as e:
        print("[INFO] Status message emitted.")

def main():
    """Function documentation."""
    print("="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    data_dict = load_rule_engine_inputs()
    if data_dict is None:
        print("[INFO] Status message emitted.")
        return False
    
    print("[INFO] Status message emitted.")
    keyword_tool = init_keyword_tool()
    
    df_result = run_rule_engine(
        data_dict['gdf_aggregated'],
        data_dict['gdf_official_library_full'],
        keyword_tool
    )
    
    evaluate_rule_engine_results(df_result, data_dict['gdf_official_library_full'])
    
    try:
        df_result.to_csv(RULE_ENGINE_INTERMEDIATE, index=False, encoding='utf-8-sig')
    except Exception as e:
        print("[INFO] Status message emitted.")
    
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
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
    log_dir = os.path.join(os.path.dirname(base_dir), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "3.rule_engine.txt")
    
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    print("[INFO] Status message emitted.")

    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("[INFO] Status message emitted.")
        sys.exit(0)
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
