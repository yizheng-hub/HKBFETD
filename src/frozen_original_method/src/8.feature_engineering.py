# 8.feature_engineering.py
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import json
import pickle
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    MAIN_CLASSES_ML,
    HIGH_CONFIDENCE_SOURCES,
    INTERMEDIATE_DIR,
    AI_CALIBRATED_OUTPUT_PATH,
    OFFICIAL_LIB_BASE_PATH,
    KEYWORDS_FILE,
    FORCE_RECOMPUTE_FEATURES,
    FEATURE_X_ALL_PATH,
    FEATURE_X_TRAIN_ML_PATH,
    FEATURE_Y_MULTILABEL_PATH,
    FEATURE_TRAINING_DATA_ML_PATH,
    FEATURE_BASE_INDEXED_PATH,
    FEATURE_GDF_BASE_CALIBRATED_PATH,
    NEIGHBOR_DENSITY_RADIUS,
    COMPACTNESS_EPSILON,
)

from utils import init_keyword_tool

warnings.filterwarnings("ignore")
tqdm.pandas()


def normalize_building_id(v):
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_main_to_ml_target(value):
    text = str(value or "").strip()
    if not text:
        return None
    low = text.lower()

    for target in MAIN_CLASSES_ML:
        if text == str(target):
            return target

    if ("residential" in low) or ("住宅" in text):
        return MAIN_CLASSES_ML[0]
    if ("commercial" in low) or ("商业" in text):
        return MAIN_CLASSES_ML[1]
    if ("industrial" in low) or ("工业" in text):
        return MAIN_CLASSES_ML[2]
    return None


def is_mixed_unknown_or_non_assessed(value):
    text = str(value or "").strip()
    low = text.lower()
    return (
        ("mixed-use" in low)
        or ("mixed use" in low)
        or ("混合" in text)
        or ("unknown" in low)
        or ("未知" in text)
        or ("non-assessed" in low)
        or ("non assessed" in low)
        or ("非评估" in text)
    )


def is_high_confidence_source(source):
    if pd.isna(source):
        return False
    src = str(source)
    return ("_LLM_Corrected" in src) or (src in HIGH_CONFIDENCE_SOURCES) or ("Inherited" in src)


def load_feature_engineering_inputs():
    df_base = pd.read_csv(AI_CALIBRATED_OUTPUT_PATH, low_memory=False)
    gdf_official_library = gpd.read_file(OFFICIAL_LIB_BASE_PATH)
    if gdf_official_library.crs is None:
        gdf_official_library = gdf_official_library.set_crs("EPSG:2326", allow_override=True)

    if "BUILDINGSTRUCTUREID" in df_base.columns:
        df_base["BUILDINGSTRUCTUREID"] = df_base["BUILDINGSTRUCTUREID"].apply(normalize_building_id)
    if "BUILDINGSTRUCTUREID" in gdf_official_library.columns:
        gdf_official_library["BUILDINGSTRUCTUREID"] = gdf_official_library["BUILDINGSTRUCTUREID"].apply(normalize_building_id)

    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        keywords_config = json.load(f)

    return {
        "df_base": df_base,
        "gdf_official_library": gdf_official_library,
        "keywords_config": keywords_config,
    }


def prepare_feature_data(df_base, gdf_official_library):
    gdf = gdf_official_library.merge(
        df_base.drop(columns=["geometry"], errors="ignore"),
        on="BUILDINGSTRUCTUREID",
        how="inner",
        suffixes=("_geo", "_attr"),
    )

    duplicate_cols = [c for c in gdf.columns if c.endswith("_geo") or c.endswith("_attr")]
    for col in duplicate_cols:
        if col.endswith("_geo"):
            base_col = col.replace("_geo", "")
            attr_col = base_col + "_attr"
            if attr_col in gdf.columns:
                gdf[base_col] = gdf[attr_col]
                gdf = gdf.drop(columns=[col, attr_col], errors="ignore")
    return gdf


def prepare_training_data(gdf_base_calibrated):
    training_candidates = gdf_base_calibrated[gdf_base_calibrated["Classification_Source"].apply(is_high_confidence_source)].copy()
    training_candidates = training_candidates[~training_candidates["Final_Main_Class"].apply(is_mixed_unknown_or_non_assessed)].copy()
    training_candidates["_ml_main"] = training_candidates["Final_Main_Class"].apply(normalize_main_to_ml_target)
    training_data = training_candidates[training_candidates["_ml_main"].notna()].copy()

    for main_class in MAIN_CLASSES_ML:
        training_data[f"is_{main_class}"] = (training_data["_ml_main"] == main_class).astype(int)

    training_data.drop(columns=["_ml_main"], inplace=True, errors="ignore")
    return training_data


def dominant_neighbor_code(main_class_value):
    mc = normalize_main_to_ml_target(main_class_value)
    if mc is None:
        if is_mixed_unknown_or_non_assessed(main_class_value):
            text = str(main_class_value or "").lower()
            if ("mixed-use" in text) or ("mixed use" in text) or ("混合" in str(main_class_value)):
                return 5
        return -1
    if mc == MAIN_CLASSES_ML[0]:
        return 0
    if mc == MAIN_CLASSES_ML[1]:
        return 1
    if mc == MAIN_CLASSES_ML[2]:
        return 2
    return -1


def compute_advanced_features(gdf_base_calibrated, training_data_ml):
    feature_base = gdf_base_calibrated.copy()
    if "Classification_Stage" in feature_base.columns:
        stage = feature_base["Classification_Stage"].astype(str)
        mask = stage.str.contains("待评估|to evaluate|unknown", case=False, regex=True, na=False)
        if mask.any():
            feature_base = feature_base[mask].copy()

    feature_base["area"] = feature_base.geometry.area
    feature_base["perimeter"] = feature_base.geometry.length
    feature_base["compactness"] = (4 * np.pi * feature_base["area"]) / ((feature_base["perimeter"] ** 2) + COMPACTNESS_EPSILON)

    centroids = feature_base.geometry.representative_point()
    feature_base["centroid_x"] = centroids.x
    feature_base["centroid_y"] = centroids.y

    feature_base["OZP_ZONE_LABEL"] = feature_base.get("OZP_ZONE_LABEL", pd.Series(["Unknown"] * len(feature_base))).fillna("Unknown")
    feature_base["ozp_code"] = pd.factorize(feature_base["OZP_ZONE_LABEL"])[0]
    feature_base["Estimated_Height"] = pd.to_numeric(feature_base.get("Estimated_Height", 15.0), errors="coerce").fillna(15.0)

    gdf_all = gdf_base_calibrated.set_index("BUILDINGSTRUCTUREID")
    feature_base_indexed = feature_base.set_index("BUILDINGSTRUCTUREID")

    buffered_geoms = feature_base_indexed.geometry.buffer(NEIGHBOR_DENSITY_RADIUS)
    neighbor_counts = []
    for geom in tqdm(buffered_geoms, desc="Computing neighbor density", total=len(buffered_geoms)):
        idx = list(gdf_all.sindex.query(geom, predicate="intersects"))
        neighbor_counts.append(max(0, len(idx) - 1))
    feature_base_indexed["neighbor_density_50m"] = neighbor_counts

    known_neighbors = gdf_all[~gdf_all["Final_Main_Class"].apply(is_mixed_unknown_or_non_assessed)].copy()

    def get_dominant_neighbor_class(geom):
        buf = geom.buffer(100)
        idx = list(known_neighbors.sindex.query(buf, predicate="intersects"))
        if not idx:
            return -1
        cand = known_neighbors.iloc[idx]
        cand = cand[cand.geometry.intersects(buf)]
        if cand.empty:
            return -1
        mode_val = cand["Final_Main_Class"].mode()
        if mode_val.empty:
            return -1
        return dominant_neighbor_code(mode_val.iloc[0])

    dominant = []
    for geom in tqdm(feature_base_indexed.geometry, desc="Computing dominant neighbor class", total=len(feature_base_indexed)):
        dominant.append(get_dominant_neighbor_class(geom))
    feature_base_indexed["dominant_neighbor_class_100m"] = dominant

    feature_columns = [
        "area",
        "perimeter",
        "compactness",
        "centroid_x",
        "centroid_y",
        "ozp_code",
        "neighbor_density_50m",
        "dominant_neighbor_class_100m",
        "Estimated_Height",
    ]
    for col in feature_columns:
        if col not in feature_base_indexed.columns:
            feature_base_indexed[col] = 0

    X_all = feature_base_indexed[feature_columns]
    target_cols = [f"is_{c}" for c in MAIN_CLASSES_ML]
    for col in target_cols:
        if col not in training_data_ml.columns:
            training_data_ml[col] = 0
    y_multilabel = training_data_ml.set_index("BUILDINGSTRUCTUREID")[target_cols]
    X_train_ml = X_all.reindex(y_multilabel.index)

    common_idx = X_train_ml.index.intersection(y_multilabel.index)
    X_train_ml = X_train_ml.loc[common_idx]
    y_multilabel = y_multilabel.loc[common_idx]

    return {
        "X_all": X_all,
        "X_train_ml": X_train_ml,
        "y_multilabel": y_multilabel,
        "feature_base_indexed": feature_base_indexed,
        "training_data_ml": training_data_ml,
        "gdf_base_calibrated": gdf_base_calibrated,
    }


def save_feature_engineering_results(results):
    with open(FEATURE_X_ALL_PATH, "wb") as f:
        pickle.dump(results["X_all"], f)
    with open(FEATURE_X_TRAIN_ML_PATH, "wb") as f:
        pickle.dump(results["X_train_ml"], f)
    with open(FEATURE_Y_MULTILABEL_PATH, "wb") as f:
        pickle.dump(results["y_multilabel"], f)
    with open(FEATURE_TRAINING_DATA_ML_PATH, "wb") as f:
        pickle.dump(results["training_data_ml"], f)
    with open(FEATURE_BASE_INDEXED_PATH, "wb") as f:
        pickle.dump(results["feature_base_indexed"], f)

    results["gdf_base_calibrated"].to_file(FEATURE_GDF_BASE_CALIBRATED_PATH, driver="GeoJSON")

    metadata = {
        "feature_columns": list(results["X_all"].columns),
        "main_classes": MAIN_CLASSES_ML,
        "num_samples": len(results["X_all"]),
        "num_training_samples": len(results["X_train_ml"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = os.path.join(INTERMEDIATE_DIR, "feature_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return True


def load_feature_engineering_results():
    required = [
        FEATURE_X_ALL_PATH,
        FEATURE_X_TRAIN_ML_PATH,
        FEATURE_Y_MULTILABEL_PATH,
        FEATURE_TRAINING_DATA_ML_PATH,
        FEATURE_BASE_INDEXED_PATH,
        FEATURE_GDF_BASE_CALIBRATED_PATH,
    ]
    if any(not os.path.exists(p) for p in required):
        return None
    with open(FEATURE_X_ALL_PATH, "rb") as f:
        X_all = pickle.load(f)
    with open(FEATURE_X_TRAIN_ML_PATH, "rb") as f:
        X_train_ml = pickle.load(f)
    with open(FEATURE_Y_MULTILABEL_PATH, "rb") as f:
        y_multilabel = pickle.load(f)
    with open(FEATURE_TRAINING_DATA_ML_PATH, "rb") as f:
        training_data_ml = pickle.load(f)
    with open(FEATURE_BASE_INDEXED_PATH, "rb") as f:
        feature_base_indexed = pickle.load(f)
    gdf_base_calibrated = gpd.read_file(FEATURE_GDF_BASE_CALIBRATED_PATH)
    return {
        "X_all": X_all,
        "X_train_ml": X_train_ml,
        "y_multilabel": y_multilabel,
        "feature_base_indexed": feature_base_indexed,
        "training_data_ml": training_data_ml,
        "gdf_base_calibrated": gdf_base_calibrated,
    }


def main():
    if not FORCE_RECOMPUTE_FEATURES:
        cached = load_feature_engineering_results()
        if cached is not None:
            return cached

    data = load_feature_engineering_inputs()
    gdf_base_calibrated = prepare_feature_data(data["df_base"], data["gdf_official_library"])
    training_data_ml = prepare_training_data(gdf_base_calibrated)
    feature_results = compute_advanced_features(gdf_base_calibrated, training_data_ml)
    feature_results["keyword_tool"] = init_keyword_tool()

    ok = save_feature_engineering_results(feature_results)
    return feature_results if ok else None


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if "\r" not in message:
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
    log_file_path = os.path.join(log_dir, "8.feature_engineering.txt")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout

    try:
        result = main()
        if result is None:
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
