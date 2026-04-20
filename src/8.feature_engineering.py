# feature_engineering.py

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
from collections import Counter
import pickle
import time

from config import (
    MAIN_CLASSES_ML, HIGH_CONFIDENCE_SOURCES,
    INTERMEDIATE_DIR, OUTPUT_DIR,
    AI_CALIBRATED_OUTPUT_PATH, OFFICIAL_LIB_BASE_PATH,
    RULE_ENGINE_OUTPUT_PATH, KEYWORDS_FILE, AGGREGATED_GDF_PATH,
    OZP_DATA_DIR, FORCE_RECOMPUTE_FEATURES,
    FEATURE_X_ALL_PATH, FEATURE_X_TRAIN_ML_PATH,
    FEATURE_Y_MULTILABEL_PATH, FEATURE_TRAINING_DATA_ML_PATH,
    FEATURE_BASE_INDEXED_PATH, FEATURE_GDF_BASE_CALIBRATED_PATH,
    NEIGHBOR_DENSITY_RADIUS, DOMINANT_NEIGHBOR_RADIUS,
    COMPACTNESS_EPSILON
)

from utils import (
    init_keyword_tool, classify_text_by_keywords, safe_str,
    load_and_fuse_ozp, to_simplified_chinese
)

if 'MAIN_CLASSES_ML' not in globals():
    from config import MAIN_CLASSES_ML

warnings.filterwarnings('ignore')
tqdm.pandas()

def load_feature_engineering_inputs():
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    try:
        # df_base = pd.read_csv(AI_CALIBRATED_OUTPUT_PATH)

        print("[INFO] Status message emitted.")
        df_base = pd.read_csv(AI_CALIBRATED_OUTPUT_PATH)
        print("[INFO] Status message emitted.")

        print("[INFO] Status message emitted.")
        gdf_official_library = gpd.read_file(OFFICIAL_LIB_BASE_PATH)
        gdf_official_library = gdf_official_library.set_crs("EPSG:2326", allow_override=True)
        print("[INFO] Status message emitted.")
        
        print("[INFO] Status message emitted.")
        df_rule_classified = pd.read_csv(AI_CALIBRATED_OUTPUT_PATH, low_memory=False)
        print("[INFO] Status message emitted.")
        
        print("[INFO] Status message emitted.")
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            KEYWORDS_CONFIG = json.load(f)
        print("[INFO] Status message emitted.")
        
        return {
            'df_base': df_base,
            'gdf_official_library': gdf_official_library,
            'df_rule_classified': df_rule_classified,
            'keywords_config': KEYWORDS_CONFIG
        }
        
    except FileNotFoundError as e:
        print("[INFO] Status message emitted.")
        print("[INFO] Status message emitted.")
        print(f"  1. {AI_CALIBRATED_OUTPUT_PATH}")
        print(f"  2. {OFFICIAL_LIB_BASE_PATH}")
        print(f"  3. {KEYWORDS_FILE}")
        return None
    
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()
        return None

def prepare_feature_data(df_base, gdf_official_library, keyword_tool):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    gdf_base_calibrated = gdf_official_library.merge(
        df_base.drop(columns=['geometry'], errors='ignore'), 
        on='BUILDINGSTRUCTUREID', 
        how='inner',
        suffixes=('_geo', '_attr')
    )
    
    duplicate_cols = [col for col in gdf_base_calibrated.columns if col.endswith('_geo') or col.endswith('_attr')]
    for col in duplicate_cols:
        if col.endswith('_geo'):
            base_col = col.replace('_geo', '')
            if base_col + '_attr' in gdf_base_calibrated.columns:
                gdf_base_calibrated[base_col] = gdf_base_calibrated[base_col + '_attr']
                gdf_base_calibrated = gdf_base_calibrated.drop(columns=[col, base_col + '_attr'])
    
    print("[INFO] Status message emitted.")
    
    # if 'OZP_ZONE_LABEL' not in gdf_base_calibrated.columns:
    #     gdf_base_calibrated['OZP_ZONE_LABEL'] = 'Unknown'
    
    return gdf_base_calibrated

def prepare_training_data(gdf_base_calibrated):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    # high_confidence_mask = gdf_base_calibrated['Classification_Source'].isin(HIGH_CONFIDENCE_SOURCES)
    
    def is_high_confidence(source):
        if pd.isna(source): return False
        source = str(source)
        if '_LLM_Corrected' in source: return True
        if source in HIGH_CONFIDENCE_SOURCES: return True
        if 'Inherited' in source: return True
        return False

    high_confidence_mask = gdf_base_calibrated['Classification_Source'].apply(is_high_confidence)
    training_candidates = gdf_base_calibrated[high_confidence_mask].copy()
    
    print("[INFO] Status message emitted.")
    
    non_mixed_mask = ~training_candidates['Final_Main_Class'].isin(['混合用途', '未知类别', '非评估类别'])
    training_data = training_candidates[non_mixed_mask].copy()
    
    valid_class_mask = training_data['Final_Main_Class'].isin(MAIN_CLASSES_ML)
    training_data = training_data[valid_class_mask].copy()
    
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    for main_class in MAIN_CLASSES_ML:
        training_data[f'is_{main_class}'] = (training_data['Final_Main_Class'] == main_class).astype(int)
    
    print("[INFO] Status message emitted.")
    for main_class in MAIN_CLASSES_ML:
        count = training_data[f'is_{main_class}'].sum()
        print("[INFO] Status message emitted.")
    
    return training_data


def compute_advanced_features(gdf_base_calibrated, training_data_ml):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    start_time = time.time()
    
    if 'Classification_Stage' in gdf_base_calibrated.columns:
        feature_base = gdf_base_calibrated[gdf_base_calibrated['Classification_Stage'] == '待评估'].copy()
    else:
        feature_base = gdf_base_calibrated.copy()
    
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    feature_base['area'] = feature_base.geometry.area
    feature_base['perimeter'] = feature_base.geometry.length
    
    from config import COMPACTNESS_EPSILON
    feature_base['compactness'] = (4 * np.pi * feature_base.area) / ((feature_base.perimeter ** 2) + COMPACTNESS_EPSILON)
    
    centroids = feature_base.geometry.representative_point()
    feature_base['centroid_x'], feature_base['centroid_y'] = centroids.x, centroids.y
    
    feature_base['OZP_ZONE_LABEL'] = feature_base.get('OZP_ZONE_LABEL', pd.Series(['Unknown']*len(feature_base))).fillna('Unknown')
    ozp_labels, ozp_uniques = pd.factorize(feature_base['OZP_ZONE_LABEL'])
    feature_base['ozp_code'] = ozp_labels
    
    feature_base['Estimated_Height'] = pd.to_numeric(feature_base.get('Estimated_Height', 15.0), errors='coerce').fillna(15.0)
    
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    
    from config import NEIGHBOR_DENSITY_RADIUS, DOMINANT_NEIGHBOR_RADIUS
    
    gdf_all_for_neighbors = gdf_base_calibrated.set_index('BUILDINGSTRUCTUREID')
    feature_base_indexed = feature_base.set_index('BUILDINGSTRUCTUREID')
    
    print("[INFO] Status message emitted.")
    buffered_geoms = feature_base_indexed.geometry.buffer(NEIGHBOR_DENSITY_RADIUS)
    
    neighbor_counts = []
    for geom in tqdm(buffered_geoms, desc="计算邻居密度", total=len(buffered_geoms)):
        possible_matches_index = list(gdf_all_for_neighbors.sindex.query(geom, predicate='intersects'))
        neighbor_counts.append(len(possible_matches_index) - 1)
    
    feature_base_indexed['neighbor_density_50m'] = neighbor_counts
    
    print("[INFO] Status message emitted.")
    known_neighbors = gdf_all_for_neighbors[
        ~gdf_all_for_neighbors['Final_Main_Class'].isin(['未知类别', '非评估类别'])
    ].copy()
    
    class_map = {
        '住宅类别': 0, 
        '商业类别': 1, 
        '工业类别': 2, 
        '社会服务类别': 3,
        '公共事业类别': 4,
        '混合用途': 5,
        '非评估类别': -1,
        '未知类别': -1
    }
    
    def get_dominant_neighbor_class(geom):
        buffer_geom = geom.buffer(100)
        
        possible_matches_index = list(known_neighbors.sindex.query(buffer_geom, predicate='intersects'))
        
        if not possible_matches_index:
            return -1
        
        possible_neighbors = known_neighbors.iloc[possible_matches_index]
        actual_neighbors = possible_neighbors[possible_neighbors.geometry.intersects(buffer_geom)]
        
        if actual_neighbors.empty:
            return -1
        
        dominant_class = actual_neighbors['Final_Main_Class'].mode()
        
        if dominant_class.empty:
            return -1
        
        return class_map.get(dominant_class.iloc[0], -1)
    
    dominant_classes = []
    for geom in tqdm(feature_base_indexed.geometry, desc="计算主导邻居类别", total=len(feature_base_indexed)):
        dominant_classes.append(get_dominant_neighbor_class(geom))
    
    feature_base_indexed['dominant_neighbor_class_100m'] = dominant_classes
    
    feature_columns = [
        'area', 'perimeter', 'compactness', 'centroid_x', 'centroid_y', 'ozp_code',
        'neighbor_density_50m', 'dominant_neighbor_class_100m', 'Estimated_Height'
    ]
    
    missing_cols = [col for col in feature_columns if col not in feature_base_indexed.columns]
    if missing_cols:
        print("[INFO] Status message emitted.")
        for col in missing_cols:
            feature_base_indexed[col] = 0
    
    X_all = feature_base_indexed[feature_columns]
    
    y_multilabel = training_data_ml.set_index('BUILDINGSTRUCTUREID')[[f'is_{c}' for c in MAIN_CLASSES_ML]]
    X_train_ml = X_all.reindex(y_multilabel.index)
    
    common_index = X_train_ml.index.intersection(y_multilabel.index)
    X_train_ml = X_train_ml.loc[common_index]
    y_multilabel = y_multilabel.loc[common_index]
    
    elapsed_time = time.time() - start_time
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    
    return {
        'X_all': X_all,
        'X_train_ml': X_train_ml,
        'y_multilabel': y_multilabel,
        'feature_base_indexed': feature_base_indexed,
        'training_data_ml': training_data_ml,
        'gdf_base_calibrated': gdf_base_calibrated
    }

def save_feature_engineering_results(results_dict):
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    try:
        with open(FEATURE_X_ALL_PATH, 'wb') as f:
            pickle.dump(results_dict['X_all'], f)
        print("[INFO] Status message emitted.")
        
        with open(FEATURE_X_TRAIN_ML_PATH, 'wb') as f:
            pickle.dump(results_dict['X_train_ml'], f)
        print("[INFO] Status message emitted.")
        
        with open(FEATURE_Y_MULTILABEL_PATH, 'wb') as f:
            pickle.dump(results_dict['y_multilabel'], f)
        print("[INFO] Status message emitted.")
        
        with open(FEATURE_TRAINING_DATA_ML_PATH, 'wb') as f:
            pickle.dump(results_dict['training_data_ml'], f)
        print("[INFO] Status message emitted.")
        
        with open(FEATURE_BASE_INDEXED_PATH, 'wb') as f:
            pickle.dump(results_dict['feature_base_indexed'], f)
        print("[INFO] Status message emitted.")
        
        results_dict['gdf_base_calibrated'].to_file(FEATURE_GDF_BASE_CALIBRATED_PATH, driver='GeoJSON')
        print("[INFO] Status message emitted.")
        
        feature_metadata = {
            'feature_columns': list(results_dict['X_all'].columns),
            'main_classes': MAIN_CLASSES_ML,
            'num_samples': len(results_dict['X_all']),
            'num_training_samples': len(results_dict['X_train_ml']),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = os.path.join(INTERMEDIATE_DIR, "feature_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(feature_metadata, f, indent=2, ensure_ascii=False)
        print("[INFO] Status message emitted.")
        
        return True
        
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()
        return False

def load_feature_engineering_results():
    """Function documentation."""
    print("[INFO] Status message emitted.")
    
    try:
        required_files = [
            FEATURE_X_ALL_PATH,
            FEATURE_X_TRAIN_ML_PATH,
            FEATURE_Y_MULTILABEL_PATH,
            FEATURE_TRAINING_DATA_ML_PATH,
            FEATURE_BASE_INDEXED_PATH,
            FEATURE_GDF_BASE_CALIBRATED_PATH
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print("[INFO] Status message emitted.")
                return None
        
        with open(FEATURE_X_ALL_PATH, 'rb') as f:
            X_all = pickle.load(f)
        print("[INFO] Status message emitted.")
        
        with open(FEATURE_X_TRAIN_ML_PATH, 'rb') as f:
            X_train_ml = pickle.load(f)
        print("[INFO] Status message emitted.")
        
        with open(FEATURE_Y_MULTILABEL_PATH, 'rb') as f:
            y_multilabel = pickle.load(f)
        print("[INFO] Status message emitted.")
        
        with open(FEATURE_TRAINING_DATA_ML_PATH, 'rb') as f:
            training_data_ml = pickle.load(f)
        print("[INFO] Status message emitted.")
        
        with open(FEATURE_BASE_INDEXED_PATH, 'rb') as f:
            feature_base_indexed = pickle.load(f)
        print("[INFO] Status message emitted.")
        
        gdf_base_calibrated = gpd.read_file(FEATURE_GDF_BASE_CALIBRATED_PATH)
        print("[INFO] Status message emitted.")
        
        metadata_path = os.path.join(INTERMEDIATE_DIR, "feature_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                feature_metadata = json.load(f)
            print("[INFO] Status message emitted.")
        else:
            feature_metadata = {}
        
        return {
            'X_all': X_all,
            'X_train_ml': X_train_ml,
            'y_multilabel': y_multilabel,
            'feature_base_indexed': feature_base_indexed,
            'training_data_ml': training_data_ml,
            'gdf_base_calibrated': gdf_base_calibrated,
            'feature_metadata': feature_metadata
        }
        
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Function documentation."""
    print("="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    if not FORCE_RECOMPUTE_FEATURES:
        feature_results = load_feature_engineering_results()
        if feature_results is not None:
            print("[INFO] Status message emitted.")
            print("[INFO] Status message emitted.")
            print("[INFO] Status message emitted.")
            return feature_results
        else:
            print("[INFO] Status message emitted.")
    
    data_dict = load_feature_engineering_inputs()
    if data_dict is None:
        print("[INFO] Status message emitted.")
        return None
    
    df_base = data_dict['df_base']
    gdf_official_library = data_dict['gdf_official_library']
    df_rule_classified = data_dict['df_rule_classified']
    keywords_config = data_dict['keywords_config']
    
    keyword_tool = init_keyword_tool()
    
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    gdf_base_calibrated = prepare_feature_data(df_base, gdf_official_library, keyword_tool)
    
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    training_data_ml = prepare_training_data(gdf_base_calibrated)
    
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    feature_results = compute_advanced_features(gdf_base_calibrated, training_data_ml)
    
    feature_results['df_rule_classified'] = df_rule_classified
    feature_results['keyword_tool'] = keyword_tool
    
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    if save_feature_engineering_results(feature_results):
        print("[INFO] Status message emitted.")
    else:
        print("[INFO] Status message emitted.")
    
    print("\n" + "="*60)
    print("[INFO] Status message emitted.")
    print("="*60)
    
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    print("[INFO] Status message emitted.")
    
    return feature_results

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
    
    script_name = os.path.basename(__file__).replace(".py", ".txt")
    log_file_path = os.path.join(log_dir, script_name)
    
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout 
    
    print("[INFO] Status message emitted.")
    
    try:
        success = main()
        if success:
            print("[INFO] Status message emitted.")
        else:
            print("[INFO] Status message emitted.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("[INFO] Status message emitted.")
        sys.exit(0)
    except Exception as e:
        print("[INFO] Status message emitted.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
