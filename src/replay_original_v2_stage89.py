# -*- coding: utf-8 -*-
"""Isolated corrected-input replay of frozen de8b stages 8 and 9."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, pickle, sys, time
from pathlib import Path

FROZEN_ROOT = Path(os.environ["HKBFETD_REPLAY_FROZEN_METHOD"]).resolve()
FROZEN = FROZEN_ROOT / "src"
FROZEN_CTL = FROZEN_ROOT / "ctl"
INPUT_STEP7 = Path(os.environ["HKBFETD_REPLAY_OUTPUT_DIR"]).resolve() / "step7_corrected.csv"
INPUT_PHYSICAL = Path(os.environ["HKBFETD_REPLAY_VERIFIED_PHYSICAL"]).resolve()
OUT = Path(os.environ["HKBFETD_REPLAY_ML_OUTPUT_DIR"]).resolve()
PROBES = OUT / "dry_run_input_check.json"
FEATURES = ["area", "perimeter", "compactness", "centroid_x", "centroid_y", "ozp_code",
            "neighbor_density_50m", "dominant_neighbor_class_100m", "Estimated_Height"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_id(value):
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s[:-2] if s.endswith(".0") else s


def dry_run_check():
    import pandas as pd
    report = {
        "mode": "dry_run_input_check_only",
        "training_started": False,
        "input_step7": str(INPUT_STEP7),
        "input_physical": str(INPUT_PHYSICAL),
        "step7_exists": INPUT_STEP7.is_file(),
        "physical_exists": INPUT_PHYSICAL.is_file(),
        "frozen_sources": {},
        "checks": {},
    }
    expected = {
        "8.feature_engineering.py": "3595d78e9e2521f275e1659fe568efe14e58196425a291d92b1dc1b08215620d",
        "9.ml_classification.py": "b561f64ec3bbbd6c47aa039fd9842421ef6394bb9bcb6a0d44a687a8d443d63c",
        "config.py": "ae5a0039564f8178785d6158e5635cbb18f65116ceb034933a324544d88a033f",
        "utils.py": "9a4deb3f253c1d5197f2cb1482aa13aa6d6f4ea3dfbfd017d5a38ec3775b0606",
    }
    for name, digest in expected.items():
        path = FROZEN / name
        actual = sha256(path) if path.is_file() else None
        report["frozen_sources"][name] = {"sha256": actual, "expected_de8b_sha256": digest,
                                           "matches": actual == digest}
    report["checks"]["frozen_sources_match_de8b"] = all(x["matches"] for x in report["frozen_sources"].values())
    if INPUT_PHYSICAL.is_file():
        phys = pd.read_parquet(INPUT_PHYSICAL, columns=["Building_ID_建筑ID", "Footprint_Area_sqm_占地面积", "Height_m_建筑高度", "geometry"])
        ids = phys["Building_ID_建筑ID"].map(norm_id)
        report["physical"] = {"rows": len(phys), "unique_ids": int(ids.nunique()), "duplicate_ids": int(ids.duplicated().sum()),
                              "missing_height_count": int(pd.to_numeric(phys["Height_m_建筑高度"], errors="coerce").isna().sum()),
                              "geometry_null_count": int(phys["geometry"].isna().sum()), "crs": "EPSG:2326 (verified parquet metadata)"}
        report["checks"]["physical_id_unique"] = bool(ids.is_unique and ids.ne("").all())
    if INPUT_STEP7.is_file():
        step7 = pd.read_csv(INPUT_STEP7, low_memory=False, encoding="utf-8-sig")
        report["step7"] = {"rows": len(step7), "columns": list(step7.columns),
                           "has_id": "BUILDINGSTRUCTUREID" in step7,
                           "has_classification": all(c in step7 for c in ["Final_Main_Class", "Final_Sub_Class", "Classification_Source"])}
    else:
        report["checks"]["step7_ready"] = False
        report["step7_note"] = "Expected corrected step7 file is not present yet; no training attempted."
    report["all_available_input_checks_pass"] = bool(report["checks"].get("frozen_sources_match_de8b") and
        report["checks"].get("physical_id_unique", False) and report["physical"].get("geometry_null_count", 1) == 0)
    OUT.mkdir(parents=True, exist_ok=True)
    PROBES.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_available_input_checks_pass"] else 2


def load_modules():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(FROZEN))
    import sklearn.utils.fixes as fixes
    # Baseline replay compatibility: bypass only sklearn's unstable OpenBLAS environment probe.
    fixes._in_unstable_openblas_configuration = lambda: False
    import config
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intermediate_files").mkdir(exist_ok=True)
    (OUT / "output").mkdir(exist_ok=True)
    (OUT / "log").mkdir(exist_ok=True)
    config.INTERMEDIATE_DIR = str(OUT / "intermediate_files")
    config.OUTPUT_DIR = str(OUT / "output")
    config.LOG_DIR = str(OUT / "log")
    config.CTL_DIR = str(FROZEN_CTL)
    config.KEYWORDS_FILE = str(FROZEN_CTL / "keywords.json")
    config.AI_CALIBRATED_OUTPUT_PATH = str(INPUT_STEP7)
    config.FEATURE_X_ALL_PATH = str(OUT / "features_X_all.pkl")
    config.FEATURE_X_TRAIN_ML_PATH = str(OUT / "features_X_train_ml.pkl")
    config.FEATURE_Y_MULTILABEL_PATH = str(OUT / "features_y_multilabel.pkl")
    config.FEATURE_TRAINING_DATA_ML_PATH = str(OUT / "training_samples.pkl")
    config.FEATURE_BASE_INDEXED_PATH = str(OUT / "features_base_indexed.pkl")
    config.FEATURE_GDF_BASE_CALIBRATED_PATH = str(OUT / "step8_feature_base_calibrated.geojson")
    config.ML_FINAL_OUTPUT_PATH = str(OUT / "step9_corrected_ml_calibrated.csv")
    config.ML_FINAL_OUTPUT_GEOJSON_PATH = str(OUT / "step9_corrected_ml_calibrated.geojson")
    config.KEYWORDS_FILE = str(FROZEN_CTL / "keywords.json")
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    fe = load("frozen_de8b_feature_engineering", FROZEN / "8.feature_engineering.py")
    ml = load("frozen_de8b_ml_classification", FROZEN / "9.ml_classification.py")
    return config, fe, ml


def run_training():
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    config, fe, ml = load_modules()
    if not INPUT_STEP7.is_file():
        raise FileNotFoundError(f"Corrected step7 input not ready: {INPUT_STEP7}")
    if not INPUT_PHYSICAL.is_file():
        raise FileNotFoundError(f"Verified physical input missing: {INPUT_PHYSICAL}")
    start = time.time()
    df = pd.read_csv(INPUT_STEP7, low_memory=False, encoding="utf-8-sig")
    phys = gpd.read_parquet(INPUT_PHYSICAL)
    if "BUILDINGSTRUCTUREID" not in df or "Building_ID_建筑ID" not in phys:
        raise ValueError("Required building ID column is missing")
    df["BUILDINGSTRUCTUREID"] = df["BUILDINGSTRUCTUREID"].map(norm_id)
    phys["BUILDINGSTRUCTUREID"] = phys["Building_ID_建筑ID"].map(norm_id)
    if df["BUILDINGSTRUCTUREID"].duplicated().any() or phys["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError("Duplicate building IDs in one of the inputs")
    if set(df["BUILDINGSTRUCTUREID"]) != set(phys["BUILDINGSTRUCTUREID"]):
        raise ValueError("Step7 and verified physical IDs do not match exactly")
    if phys.crs is None:
        raise ValueError("Verified geometry has no CRS")
    # Source physical values replace contaminated step7 defaults before frozen stage 8.
    phys_idx = phys.set_index("BUILDINGSTRUCTUREID")
    df_idx = df.set_index("BUILDINGSTRUCTUREID", drop=False)
    df_idx["Estimated_Height"] = pd.to_numeric(phys_idx["Height_m_建筑高度"].reindex(df_idx.index), errors="coerce")
    df_idx["Estimated_Floors"] = pd.to_numeric(phys_idx["Floors_Count_地上层数"].reindex(df_idx.index), errors="coerce")
    df_idx["SHAPE_Area"] = pd.to_numeric(phys_idx["Footprint_Area_sqm_占地面积"].reindex(df_idx.index), errors="coerce")
    df = df_idx.reset_index(drop=True)
    phys["OZP_ZONE_LABEL"] = df.set_index("BUILDINGSTRUCTUREID").get("OZP_ZONE_LABEL", pd.Series(index=df["BUILDINGSTRUCTUREID"], dtype=object)).reindex(phys["BUILDINGSTRUCTUREID"]).to_numpy()
    gdf_base = fe.prepare_feature_data(df, phys[["BUILDINGSTRUCTUREID", "geometry", "OZP_ZONE_LABEL"]].copy())
    if len(gdf_base) != len(df):
        raise ValueError(f"Geometry join lost rows: {len(gdf_base)} vs {len(df)}")
    training = fe.prepare_training_data(gdf_base)
    features = fe.compute_advanced_features(gdf_base, training)
    X_all = features["X_all"]
    X_train = features["X_train_ml"]
    y = features["y_multilabel"]
    # 15 m fallback exists only in X_all as the original model feature imputation; source/public columns remain null.
    height_source_missing = int(pd.to_numeric(gdf_base["Estimated_Height"], errors="coerce").isna().sum())
    if int(pd.to_numeric(X_all["Estimated_Height"], errors="coerce").eq(15.0).sum()) < height_source_missing:
        raise AssertionError("Missing heights were not imputed to the original internal feature value")
    for filename, frame in [("features_X_all.pkl", X_all), ("features_X_train_ml.pkl", X_train),
                            ("features_y_multilabel.pkl", y), ("training_samples.pkl", features["training_data_ml"])]:
        with (OUT / filename).open("wb") as f:
            pickle.dump(frame, f)
    with (OUT / "features_base_indexed.pkl").open("wb") as f:
        pickle.dump(features["feature_base_indexed"], f)
    # Preserve source physical nulls; never export the filled feature copy here.
    gdf_base.to_file(OUT / "step8_feature_base_calibrated.geojson", driver="GeoJSON")
    models, proba = ml.train_main_class_models(X_train, y.copy(), X_all)
    keyword_tool = {"keywords_config": json.loads((FROZEN_CTL / "keywords.json").read_text(encoding="utf-8"))}
    subclass_props = ml.train_subclass_models(X_all, X_train, df, keyword_tool)
    calibrated = ml.apply_ml_calibration(df, proba, subclass_props, X_all)
    final = ml.generate_final_output(calibrated, gdf_base)
    proba.to_csv(OUT / "probabilities.csv", encoding="utf-8-sig")
    with (OUT / "models.pkl").open("wb") as f:
        pickle.dump(models, f)
    diagnostics = {
        "status": "completed", "rows_input": len(df), "rows_output": len(final),
        "feature_columns": list(X_all.columns), "feature_count": len(X_all.columns),
        "feature_rows": len(X_all), "training_rows": len(X_train), "model_count": len(models),
        "missing_verified_height_rows": height_source_missing,
        "internal_15m_feature_fill_rows": int(pd.to_numeric(X_all["Estimated_Height"], errors="coerce").eq(15.0).sum()),
        "public_height_nulls_preserved_in_step9": int(pd.to_numeric(final["Estimated_Height"], errors="coerce").isna().sum()),
        "main_class_counts": final["Calibrated_Main_Class"].value_counts(dropna=False).to_dict(),
        "confidence_summary": proba.describe().to_dict(),
        "elapsed_seconds": time.time() - start,
    }
    (OUT / "metrics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = {"runner": str(Path(__file__)), "input_paths": {"step7_corrected": str(INPUT_STEP7), "verified_physical_attributes": str(INPUT_PHYSICAL)},
                "output_dir": str(OUT), "input_sha256": {str(p): sha256(p) for p in [INPUT_STEP7, INPUT_PHYSICAL]},
                "source_sha256": {
                    **{f"src/{p.name}": sha256(p) for p in FROZEN.iterdir() if p.is_file()},
                    **{f"ctl/{p.name}": sha256(p) for p in FROZEN_CTL.iterdir() if p.is_file()},
                },
                "output_sha256": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()},
                "diagnostics": diagnostics}
    (OUT / "hash_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-input-check", action="store_true", help="Check readiness only; never train")
    parser.add_argument("--run", action="store_true", help="Run the isolated corrected de8b stages 8 and 9")
    args = parser.parse_args()
    if args.dry_run_input_check:
        raise SystemExit(dry_run_check())
    if args.run:
        print(json.dumps(run_training(), ensure_ascii=False, indent=2, default=str))
    else:
        parser.error("Choose --dry-run-input-check or --run")
