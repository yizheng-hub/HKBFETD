# config.py

# -*- coding: utf-8 -*-
"""Module documentation."""
import os
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "intermediate_files")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "log")
CTL_DIR = os.path.join(BASE_DIR, "ctl")
VERIFICATION_DIR = os.path.join(INTERMEDIATE_DIR, "validation_materials")
QUALITY_CHECK_DIR = os.path.join(INTERMEDIATE_DIR, "quality_checks")
QUALITY_CHECK_HTML_DIR = os.path.join(QUALITY_CHECK_DIR, "html_maps")

for dir_path in [DATA_DIR, INTERMEDIATE_DIR, OUTPUT_DIR, LOG_DIR, CTL_DIR, VERIFICATION_DIR, QUALITY_CHECK_DIR, QUALITY_CHECK_HTML_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

LANDSD_DATA_DIR = os.path.join(DATA_DIR, "LandsD")

FILE_PATHS_LANDSD = {
    "BUILDING_STRUCTURE": os.path.join(LANDSD_DATA_DIR, "1.Building_Footprint_Public_20250206.gdb_BUILDING_STRUCTURE_converted.json"),
    "BUILDING_NAME": os.path.join(LANDSD_DATA_DIR, "2.Building_Footprint_Public_20250206.gdb_BUILDING_NAME_converted.json"),
    "BUILDING_INFO": os.path.join(LANDSD_DATA_DIR, "3.Building_Footprint_Public_20250206.gdb_BUILDING_INFO_converted.json"),
    "BUILDING_WORKS_HISTORY": os.path.join(LANDSD_DATA_DIR, "4.Building_Footprint_Public_20250206.gdb_BUILDING_WORKS_HISTORY_converted.json"),
    "BUILDING_LOT_NO_INFO": os.path.join(LANDSD_DATA_DIR, "5.Building_Footprint_Public_20250206.gdb_BUILDING_LOT_NO_INFO_converted.json"),
    "OCCUPATION_PERMIT": os.path.join(LANDSD_DATA_DIR, "6.Building_Footprint_Public_20250206.gdb_OCCUPATION_PERMIT_converted.json"),
    "OP_BUILDING_STRUCTURE": os.path.join(LANDSD_DATA_DIR, "7.Building_Footprint_Public_20250206.gdb_OP_BUILDING_STRUCTURE_converted.json"),
    "CT_BUILDING_CATEGORY": os.path.join(LANDSD_DATA_DIR, "8.Building_Footprint_Public_20250206.gdb_CT_BUILDING_CATEGORY_converted.json"),
    "CT_BUILDING_INFO_TYPE": os.path.join(LANDSD_DATA_DIR, "9.Building_Footprint_Public_20250206.gdb_CT_BUILDING_INFO_TYPE_converted.json"),
    "CT_BUILDING_STATUS": os.path.join(LANDSD_DATA_DIR, "10.Building_Footprint_Public_20250206.gdb_CT_BUILDING_STATUS_converted.json"),
    "CT_BUILDING_STRUCTURE_TYPE": os.path.join(LANDSD_DATA_DIR, "11.Building_Footprint_Public_20250206.gdb_CT_BUILDING_STRUCTURE_TYPE_converted.json"),
    "CT_BUILDING_WORKS_TYPE": os.path.join(LANDSD_DATA_DIR, "12.Building_Footprint_Public_20250206.gdb_CT_BUILDING_WORKS_TYPE_converted.json"),
    "CT_NAME_INFO_SOURCE": os.path.join(LANDSD_DATA_DIR, "13.Building_Footprint_Public_20250206.gdb_CT_NAME_INFO_SOURCE_converted.json"),
    "CT_NAME_STATUS": os.path.join(LANDSD_DATA_DIR, "14.Building_Footprint_Public_20250206.gdb_CT_NAME_STATUS_converted.json"),
}

BDBIAR_FILE_PATH = os.path.join(DATA_DIR, "BDBIAR.gdb_converted.csv")
OSM_PBF_FILE_PATH = os.path.join(DATA_DIR, "hong-kong-latest.osm.pbf")


OZP_DATA_DIR = os.path.join(DATA_DIR, "OZP_Zones")


KEYWORDS_FILE = os.path.join(CTL_DIR, "keywords.json")
LLM_TAXONOMY_FILE = os.path.join(CTL_DIR, "llm_taxonomy.json")
STEP4_LLM_PROMPT_PATH = os.path.join(CTL_DIR, "step4_llm_verification_prompt.txt")

OVERTURE_PARQUET_FILE_PATH = os.path.join(DATA_DIR, "overture", "places.parquet")
OVERTURE_CLEAN_PATH = os.path.join(INTERMEDIATE_DIR, "step1_overture_places_clean.geojson")

HK_BBOX_WGS84 = (113.83, 22.15, 114.45, 22.57) # (minx, miny, maxx, maxy)

OFFICIAL_LIBRARY_PATH = os.path.join(INTERMEDIATE_DIR, "step1_official_building_library.geojson")
OFFICIAL_LIB_BASE_PATH = os.path.join(INTERMEDIATE_DIR, "step1_official_building_library_base.geojson")
AGGREGATED_GDF_PATH = os.path.join(INTERMEDIATE_DIR, "step2_osm_aggregated_buildings.geojson")
OSM_CLEAN_PATH = os.path.join(INTERMEDIATE_DIR, "step2_osm_features_clean.geojson")
OSM_ALL_PATH = os.path.join(INTERMEDIATE_DIR, "step1_osm_features_all.geojson")
BDBIAR_CACHE_PATH = os.path.join(INTERMEDIATE_DIR, "step1_bdbiar_points.geojson")
CANDIDATES_PAIRS_PATH = os.path.join(INTERMEDIATE_DIR, "step6_geometry_candidate_pairs.csv")
CANDIDATES_RULES_PATH = os.path.join(INTERMEDIATE_DIR, "step6_geometry_candidates_with_rules_and_images.csv")
FINAL_SUGGESTIONS_PATH = os.path.join(INTERMEDIATE_DIR, "step6_geometry_correction_suggestions.csv")
IMAGE_OUTPUT_DIR = os.path.join(INTERMEDIATE_DIR, "geometry_correction_candidates/")
OZP_FUSED_PATH = os.path.join(INTERMEDIATE_DIR, "step1_official_library_with_ozp.geojson")

FEATURE_X_ALL_PATH = os.path.join(INTERMEDIATE_DIR, "feature_X_all.pkl")
FEATURE_X_TRAIN_ML_PATH = os.path.join(INTERMEDIATE_DIR, "feature_X_train_ml.pkl")
FEATURE_Y_MULTILABEL_PATH = os.path.join(INTERMEDIATE_DIR, "feature_y_multilabel.pkl")
FEATURE_TRAINING_DATA_ML_PATH = os.path.join(INTERMEDIATE_DIR, "feature_training_data_ml.pkl")
FEATURE_BASE_INDEXED_PATH = os.path.join(INTERMEDIATE_DIR, "feature_base_indexed.pkl")
FEATURE_GDF_BASE_CALIBRATED_PATH = os.path.join(INTERMEDIATE_DIR, "step8_feature_base_calibrated.geojson")
ML_MODELS_PATH = os.path.join(INTERMEDIATE_DIR, "ml_models.pkl")
TRAINING_DATA_PATH = os.path.join(INTERMEDIATE_DIR, "step8_ml_training_data.csv")

RULE_ENGINE_INTERMEDIATE = os.path.join(INTERMEDIATE_DIR, "step3_rule_engine_intermediate.csv")
CONTEXTILY_CACHE_DIR = os.path.join(INTERMEDIATE_DIR, "contextily_cache")

AI_DECISIONS_LOG_PATH = os.path.join(OUTPUT_DIR, "step6_ai_decisions_log.csv")
ANATOMY_MAP_PATH = os.path.join(QUALITY_CHECK_HTML_DIR, "step1_anatomy_map_telford_gardens.html")
UNMATCHED_BD_MAP_PATH = os.path.join(QUALITY_CHECK_HTML_DIR, "step1_investigation_map_unmatched_bd.html")
FOOTPRINT_DISTRIBUTION_PATH = os.path.join(QUALITY_CHECK_DIR, "step1_footprint_distribution.png")
RULE_ENGINE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "step3_rule_engine_classification.csv")
RULE_ENGINE_OUTPUT_GEOJSON_PATH = os.path.join(OUTPUT_DIR, "step3_rule_engine_classification.geojson")
AI_CALIBRATED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "step7_ai_calibrated_classification.csv")
AI_CALIBRATED_OUTPUT_GEOJSON_PATH = os.path.join(OUTPUT_DIR, "step7_ai_calibrated_classification.geojson")
ML_FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "step9_ml_calibrated_classification.csv")
ML_FINAL_OUTPUT_GEOJSON_PATH = os.path.join(OUTPUT_DIR, "step9_ml_calibrated_classification.geojson")
STEP5_MERGED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "step5_merged_classification.csv")
STEP5_MERGED_OUTPUT_GEOJSON_PATH = os.path.join(OUTPUT_DIR, "step5_merged_classification.geojson")
PUBLIC_DATASET_CSV_PATH = os.path.join(OUTPUT_DIR, "HK_UBEM_Buildings_Public_v1.csv")
PUBLIC_DATASET_GEOJSON_PATH = os.path.join(OUTPUT_DIR, "HK_UBEM_Buildings_Public_v1.geojson")
UNKNOWN_MAP_PATH = os.path.join(QUALITY_CHECK_HTML_DIR, "step3_rule_engine_unknowns_map.html")
ISLANDS_MAP_PATH = os.path.join(QUALITY_CHECK_HTML_DIR, "step2_information_islands_map.html")
AGGREGATION_STATISTICS_PATH = os.path.join(QUALITY_CHECK_DIR, "step2_aggregation_statistics.png")
MULTI_OSM_VIS_PATH = os.path.join(QUALITY_CHECK_DIR, "step2_multi_osm_aggregation_bsid_{}.png")
MULTI_OSM_FACETED_PATH = os.path.join(QUALITY_CHECK_DIR, "step2_multi_osm_faceted_bsid_{}.png")

TARGET_LON = 114.2125066
TARGET_LAT = 22.32440032
SEARCH_RADIUS_METERS = 400

MAX_MATCH_DISTANCE = 30
OVERLAP_RATIO_THRESHOLD = 0.2
NEIGHBOR_SEARCH_BUFFER = 3

MAIN_CLASSES_ML = ['住宅类别', '商业类别', '工业类别']
PROBABILITY_THRESHOLD = 0.5
SMOTE_MIN_SAMPLES = 200
SMOTE_K_NEIGHBORS = 5

MULTI_LABEL_THRESHOLD = 0.5
MIXED_USE_THRESHOLD = 0.9

COMPACTNESS_EPSILON = 1e-6
AREA_THRESHOLD_VILLAGE = 65
SMALL_AREA_THRESHOLD = 5

NEIGHBOR_DENSITY_RADIUS = 50
DOMINANT_NEIGHBOR_RADIUS = 100

FLOOR_HEIGHT_ESTIMATE = 3.0
MIN_FLOORS_FOR_ESTIMATE = 1

NUM_MULTIMATCH_SAMPLES = 3
NUM_ISLAND_SAMPLES = 100
NUM_MAP_SAMPLES = 200
VISUALIZATION_DPI = 150

HIGH_CONFIDENCE_SOURCES = [
    'OSM_Aggregation', 'BDBIAR', 'Keyword', 'Keyword_Priority', 
    'LandsD_Category', 'Inherited_via_Consensus_merge',
    'Machine_Learning', 'Corrected_by_Rule'
]

AI_PROMPT_PATH = os.path.join(CTL_DIR, "ai_calibration_prompt.txt")
AI_DECISION_BATCH_SAVE = 10

RUN_AI_GENERATION = True
SKIP_IMAGE_GENERATION_CHECK = True
SKIP_ALL_IMAGES_IF_EXIST = True
FORCE_RECOMPUTE_FEATURES = True
BATCH_IMAGE_CHECK_SIZE = 100
IMAGE_SIZE_PIXELS = 5
IMAGE_ZOOM_LEVEL = 19
TARGET_LANDSD_BSID_TO_INVESTIGATE = 416429

RUN_MICRO_ANATOMY = True
RUN_DEEP_VALIDATION = True
RUN_UNMATCHED_BD_INVESTIGATION = True
RUN_FULL_EXPLORATION = True
RUN_REVERSE_TRACEABILITY = True
RUN_AGGREGATION_EVALUATION = True
RUN_RULE_ENGINE_EVALUATION = True
RUN_CANDIDATE_GENERATION = True

RULE_DECISION_MERGE_THRESHOLD = 0.3
LGBM_RANDOM_STATE = 42
LGBM_VERBOSE = -1
RF_N_ESTIMATORS = 50
TEST_SIZE_RATIO = 0.2
LANDSD_CATEGORY_MAP_MAIN = {'2': '住宅类别', '3': '住宅类别', '4': '商业类别'}
LANDSD_CATEGORY_MAP_SUB = {'2': '私人房屋', '3': '公共房屋'}
GFA_DOMINANT_THRESHOLD = 0.99
GFA_MIXED_THRESHOLD_LOWER = 0.1
GFA_MIXED_THRESHOLD_UPPER = 0.9
SUBCLASS_PROPORTION_THRESHOLD = 0.05

LLM_VERIFICATION_INPUT = RULE_ENGINE_OUTPUT_PATH
LLM_VERIFICATION_OUTPUT = os.path.join(OUTPUT_DIR, "step4_llm_verified_classification.csv")
LLM_MODEL_NAME = "qwen2.5:3b"
LLM_CONFIDENCE_THRESHOLD = 0.7
BATCH_SAVE_INTERVAL = 100
AI_DECISION_SAVE_INTERVAL = 5

# Cloud API settings (centralized)
# Recommended: provide via environment variables instead of hard-coding secrets.
# - HKBFETD_BASE_URL
# - HKBFETD_API_KEY
# - HKBFETD_CLOUD_MODEL_NAME
BASE_URL = os.getenv("HKBFETD_BASE_URL", "https://api.zzz-api.top/v1")
API_KEY = os.getenv("HKBFETD_API_KEY", "sk-zk23b4b8e962c041ca80018b8eb24815bc869b80a1bd6aab")
CLOUD_MODEL_NAME = os.getenv("HKBFETD_CLOUD_MODEL_NAME", "hunyuan-standard-256k")

# Geometry arbitration (vision) API settings
# - HKBFETD_VISION_API_KEY
# - HKBFETD_VISION_CLOUD_MODEL_NAME
VISION_API_KEY = os.getenv("HKBFETD_VISION_API_KEY", "sk-zk23b4b8e962c041ca80018b8eb24815bc869b80a1bd6aab")
VISION_CLOUD_MODEL_NAME = os.getenv("HKBFETD_VISION_CLOUD_MODEL_NAME", "hunyuan-turbos-vision")

def load_llm_taxonomy(file_path=LLM_TAXONOMY_FILE):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

LLM_TAXONOMY = load_llm_taxonomy()


if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"LLM_VERIFICATION_INPUT: {LLM_VERIFICATION_INPUT}")
    print(f"LLM_VERIFICATION_OUTPUT: {LLM_VERIFICATION_OUTPUT}")
