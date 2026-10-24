# utils.py

# -*- coding: utf-8 -*-
"""Module documentation."""
import pandas as pd
import numpy as np
from opencc import OpenCC
import re
import json
import os
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

_LEGAL_COURT_PREFIX_RE = re.compile(
    r"\b(?:high|district|supreme|family|juvenile|small claims|magistrate(?:s|'s|s')?|coroner(?:s|'s|s')?)\s+$",
    re.IGNORECASE,
)
_LEGAL_COURT_SUFFIX_RE = re.compile(
    r"^\s+(?:house\b|of\s+(?:final\s+appeal|appeal|first\s+instance)\b)",
    re.IGNORECASE,
)

try:
    cc = OpenCC('t2s')
except Exception as e:
    pass  # placeholder log removed
    class DummyOpenCC:
        def convert(self, text):
            return text
    cc = DummyOpenCC()

def to_simplified_chinese(text):
    """Function documentation."""
    if pd.isna(text) or text is None:
        return ''
    return cc.convert(str(text))

def safe_str(value):
    """Function documentation."""
    if pd.isna(value) or value is None:
        return ''
    if isinstance(value, (list, np.ndarray)):
        return ' '.join(safe_str(item) for item in value if pd.notna(item))
    return str(value)



def init_keyword_tool(keywords_file_path=None):
    """Function documentation."""
    try:
        if keywords_file_path is None:
            from config import KEYWORDS_FILE
            keywords_file_path = KEYWORDS_FILE
        
        with open(keywords_file_path, 'r', encoding='utf-8') as f:
            KEYWORDS_CONFIG = json.load(f)
    except Exception as e:
        raise IOError(f"加载关键词文件失败: {e}")
    
    STRONG_KEYWORDS_SET = set()
    if "__STRONG_KEYWORDS__" in KEYWORDS_CONFIG:
        for kw in KEYWORDS_CONFIG["__STRONG_KEYWORDS__"]:
            if kw_sl := to_simplified_chinese(kw).lower().strip():
                STRONG_KEYWORDS_SET.add(kw_sl)
    
    KEYWORD_MAP = {}
    ABSOLUTE_NOISE_KEYWORDS = []
    uninformative = {'estate','公司','company','集团','group','机构','organization','部门','department',
        '行政','administration','公共','public','政府','government','社区','community','服务','service',
        '设施','facility','基地','base','场地','site','园区','park','区域','area',
        '房屋署','housing authority','房委会','房署','大厦','plaza','广场'}
    commercial_form_only = {to_simplified_chinese(word).lower() for word in [
        '商厦','商廈','商业大厦','商業大廈','Commercial Building','Commercial Centre','商業中心']}
    residential_form_only = {to_simplified_chinese(word).lower() for word in [
        '住宅楼宇','Residential Building','公寓','Apartment','低密度住宅','Low Density Residential',
        '住宅大厦','Residential Tower','屋苑','大廈','住宅樓','别墅','別墅','Villa','Residence',
        'Condominium','Flat','Mansion','独立屋','獨立屋','Detached House','小型屋宇','Small House',
        '洋房','bungalows','平房','独立小屋','类似屋宇','低层住宅','Low-rise Residence','Cottage','Chalet']}
    
    for main, subs in KEYWORDS_CONFIG.items():
        if main == "__STRONG_KEYWORDS__":
            continue
            
        for sub, kws in subs.items():
            for kw in kws:
                if kw_sl := to_simplified_chinese(kw).lower().strip():
                    if kw_sl in uninformative:
                        continue
                    generic_form = (main=='住宅类别' and kw_sl in residential_form_only) or (main=='商业类别' and kw_sl in commercial_form_only)
                    KEYWORD_MAP[kw_sl] = (main, '未知类别' if generic_form else sub)
                    if sub == '绝对噪声':
                        ABSOLUTE_NOISE_KEYWORDS.append(kw_sl)
    
    regex_parts = []
    for k in KEYWORD_MAP.keys():
        if len(k) <= 1: continue
        
        if re.match(r'^[a-zA-Z0-9\s\-\.]+$', k):
            regex_parts.append(r'\b' + re.escape(k) + r'\b')
        else:
            regex_parts.append(re.escape(k))

    regex_parts.sort(key=len, reverse=True)
    KEYWORD_REGEX = re.compile('|'.join(regex_parts), re.IGNORECASE) if regex_parts else None
    NOISE_REGEX = re.compile('|'.join(ABSOLUTE_NOISE_KEYWORDS)) if ABSOLUTE_NOISE_KEYWORDS else None
    
    return {
        'keywords_config': KEYWORDS_CONFIG,
        'keyword_map': KEYWORD_MAP,
        'keyword_regex': KEYWORD_REGEX,
        'noise_regex': NOISE_REGEX,
        'absolute_noise_keywords': ABSOLUTE_NOISE_KEYWORDS,
        'strong_keywords': STRONG_KEYWORDS_SET
    }


def classify_text_by_keywords(text, keyword_regex, keyword_map):
    """Function documentation."""
    if not text or pd.isna(text) or not keyword_regex:
        return [], []
    normalized = to_simplified_chinese(text).lower().replace('’', "'").replace('‘', "'").strip()
    matches = []
    for match in keyword_regex.finditer(normalized):
        keyword = match.group(0)
        if keyword.lower() == 'court':
            prefix = normalized[max(0, match.start() - 40):match.start()]
            suffix = normalized[match.end():]
            if not (_LEGAL_COURT_PREFIX_RE.search(prefix) or _LEGAL_COURT_SUFFIX_RE.search(suffix)):
                continue
        matches.append(keyword)
    if matches:
        ordered = list(dict.fromkeys(matches))
        return ordered, list(dict.fromkeys(keyword_map[kw] for kw in ordered))
    return [], []

def classify_from_bdbiar_final(text):
    """Function documentation."""
    if pd.isna(text): return (None, None)
    text = str(text)
    
    if any(k in text for k in ['綜合用途', 'Composite']):
        return (None, None)
    if any(k in text for k in ['寫字樓', '商業', 'Office', 'Commercial']):
        return ('商业类别', None)
    if any(k in text for k in ['住宅', 'Residential']):
        return ('住宅类别', '私人房屋（Private Housing）')
    if any(k in text for k in ['幼稚園', '學校', '教育', 'School', 'Kindergarten', 'Secondary']):
        return ('商业类别', '教育（Education）')
    if any(k in text for k in ['醫院', '診所', 'Medical', 'Hospital', 'Clinic']):
        return ('商业类别', '医疗（Human Health）')
    if any(k in text for k in ['工業', '工廠', 'Industrial', 'Factory']):
        return ('工业类别', '其他工业（Other Industrial）')
        
    return (None, None)

def estimate_gfa_for_validation(row, config=None):
    """Function documentation."""
    if config is None:
        from config import FLOOR_HEIGHT_ESTIMATE, MIN_FLOORS_FOR_ESTIMATE
    else:
        FLOOR_HEIGHT_ESTIMATE = config.get('FLOOR_HEIGHT_ESTIMATE', 3.0)
        MIN_FLOORS_FOR_ESTIMATE = config.get('MIN_FLOORS_FOR_ESTIMATE', 1)
    
    dom = row.get('GFA_DOMESTIC_SUM', 0)
    nondom = row.get('GFA_NONDOMESTIC_SUM', 0)
    try:
        dom = float(dom) if pd.notna(dom) else 0
        nondom = float(nondom) if pd.notna(nondom) else 0
    except:
        dom, nondom = 0, 0
        
    if (dom + nondom > 10):
        return dom + nondom
    
    area = row.geometry.area
    floors = row.get('NUMABOVEGROUNDSTOREYS')
    
    if pd.isna(floors) or floors == 0:
        floors = MIN_FLOORS_FOR_ESTIMATE
    else:
        try: 
            floors = float(floors)
        except: 
            floors = MIN_FLOORS_FOR_ESTIMATE
        
    return area * floors

def load_and_fuse_ozp(input_gdf, ozp_data_dir):
    """Function documentation."""
    import os
    
    all_features = []
    if not os.path.isdir(ozp_data_dir):
        return input_gdf.assign(ZONE_LABEL=pd.NA)
    
    for f in os.listdir(ozp_data_dir):
        if f.endswith(".json"):
            try:
                all_features.append(gpd.read_file(os.path.join(ozp_data_dir, f)))
            except:
                pass
    
    if not all_features:
        return input_gdf.assign(ZONE_LABEL=pd.NA)
    
    gdf_ozp_full = pd.concat(all_features, ignore_index=True).to_crs(input_gdf.crs)
    
    df_with_centroids = input_gdf.copy()
    if 'index_right' in df_with_centroids.columns:
        df_with_centroids = df_with_centroids.drop(columns=['index_right'])
    
    df_with_centroids['geometry'] = df_with_centroids.geometry.representative_point()
    gdf_with_ozp = gpd.sjoin(df_with_centroids, gdf_ozp_full[['ZONE_LABEL', 'geometry']], how="left", predicate='within')
    ozp_info = gdf_with_ozp.drop_duplicates(subset=['BUILDINGSTRUCTUREID'])[['BUILDINGSTRUCTUREID', 'ZONE_LABEL']]
    return input_gdf.merge(ozp_info, on='BUILDINGSTRUCTUREID', how='left')

def classify_osm_feature(row, keyword_tool=None, use_case='default'):
    """Function documentation."""
    if keyword_tool is None:
        keyword_tool = init_keyword_tool()
    
    text_parts = []
    for col in ['building:use', 'building', 'office', 'shop', 'amenity', 'name']:
        if col in row and pd.notna(row[col]):
            text_parts.append(str(row[col]))
    text = ' '.join(text_parts)
    
    _, classifications = classify_text_by_keywords(
        text, 
        keyword_tool['keyword_regex'], 
        keyword_tool['keyword_map']
    )
    
    if classifications:
        return classifications[0]
    return (None, None)

def calculate_overlap_ratio(osm_geom, official_geom, config=None):
    """Function documentation."""
    if config is None:
        from config import COMPACTNESS_EPSILON
        epsilon = COMPACTNESS_EPSILON
    else:
        epsilon = config.get('COMPACTNESS_EPSILON', 1e-6)
    
    try:
        if osm_geom and official_geom and osm_geom.is_valid and official_geom.is_valid:
            intersection_area = osm_geom.intersection(official_geom).area
            osm_area = osm_geom.area
            if osm_area > epsilon:
                return intersection_area / osm_area
    except Exception:
        pass
    return 0.0

def init_feature_keyword_tool(keywords_config):
    """Function documentation."""
    return init_keyword_tool()


def canonicalize_main_class_label(value):
    text = '' if pd.isna(value) else str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return 'Unknown_未知类别'
    normalized = text.lower().replace('-', ' ')
    for english, chinese in (
        ('Residential', '住宅类别'),
        ('Commercial', '商业类别'),
        ('Industrial', '工业类别'),
        ('Mixed-use', '混合用途'),
        ('Non-assessed', '非评估类别'),
        ('Unknown', '未知类别'),
    ):
        if chinese in text or english.lower().replace('-', ' ') in normalized:
            return f'{english}_{chinese}'
    return text


_SUBCLASS_LABEL_ALIASES = None


def canonicalize_sub_class_label(value):
    global _SUBCLASS_LABEL_ALIASES
    text = '' if pd.isna(value) else str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return 'Unknown_未知类别'
    if '_' in text:
        return text
    if _SUBCLASS_LABEL_ALIASES is None:
        from config import KEYWORDS_FILE
        labels = {
            'Unknown_未知类别',
            'Absolute Noise_绝对噪声相关',
            'Transport Infrastructure_交通基础设施',
            'Temporary/Misc Facilities_临时/杂项设施',
            'Mixed-use_混合用途',
        }
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            keywords = json.load(f)
        aliases = {}
        for subclasses in keywords.values():
            if not isinstance(subclasses, dict):
                continue
            for label in subclasses:
                match = re.fullmatch(r'([^()（）]+)[（(]([^()（）]+)[)）]', label)
                if match:
                    chinese, english = (part.strip() for part in match.groups())
                    canonical = f'{english}_{chinese}'
                    labels.add(canonical)
                    aliases[label.casefold()] = canonical
        for label in labels:
            english, chinese = label.split('_', 1)
            for alias in (label, english, chinese):
                aliases[alias.casefold()] = label
        _SUBCLASS_LABEL_ALIASES = aliases
    if text.casefold() in _SUBCLASS_LABEL_ALIASES:
        return _SUBCLASS_LABEL_ALIASES[text.casefold()]
    return re.sub(
        r'(?<!\w)([\u3400-\u9fff/、 ]+)[（(]([A-Za-z][^()（）]*)[)）]',
        lambda match: f'{match.group(2).strip()}_{match.group(1).strip()}',
        text,
    )


def normalize_classification_columns(df):
    for column, normalize in (
        ('Final_Main_Class', canonicalize_main_class_label),
        ('Final_Sub_Class', canonicalize_sub_class_label),
    ):
        if column in df.columns:
            df[column] = df[column].map(normalize)
    return df


def is_non_assessed_main(value):
    return canonicalize_main_class_label(value) == 'Non-assessed_非评估类别'


def sanitize_dataframe_for_geojson(df):
    def serialize(value):
        if value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return value

    for column in df.select_dtypes(include=['object', 'string']).columns:
        df[column] = df[column].map(serialize)
    return df
