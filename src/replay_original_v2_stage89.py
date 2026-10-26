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
HEIGHT_SENSITIVITY_SCENARIOS_M = (3.0, 6.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 150.0)


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


def protect_upstream_classifications(calibrated, ml):
    """Keep rule/LLM/VLM decisions ahead of ML in the frozen method order."""
    out = calibrated.copy()
    main = out["Final_Main_Class"].fillna("").astype(str).str.strip()
    sub = out["Final_Sub_Class"].fillna("").astype(str).str.strip()
    source = out["Classification_Source"].fillna("").astype(str).str.strip()
    mix = out.get("Sub_Class_Proportions", "")
    if not hasattr(mix, "fillna"):
        mix = main.map(lambda _: "")
    mix = mix.fillna("").astype(str).str.strip()

    unknown_main = main.str.lower().isin({"", "nan", "none", "null", "unknown"}) | main.str.contains(
        "未知", regex=False
    ) | main.str.contains("unknown", case=False, regex=False)
    weak_source = source.str.lower().isin({"", "none", "unknown", "fallback_to_default", "fallback_to_area"})
    protect_main = ~unknown_main & ~weak_source
    known_sub = ~(sub.str.lower().isin({"", "nan", "none", "null", "unknown"}) |
                  sub.str.contains("未知", regex=False) |
                  sub.str.contains("unknown", case=False, regex=False))
    known_mix = ~(mix.str.lower().isin({"", "nan", "none", "null", "unknown"}) |
                  mix.str.contains("未知", regex=False) |
                  mix.str.contains("unknown", case=False, regex=False))

    canonical_main = main.map(ml.canonicalize_main_label)
    canonical_sub = sub.map(ml.canonicalize_subclass_label)
    out.loc[protect_main, "Calibrated_Main_Class"] = canonical_main[protect_main]
    out.loc[protect_main & known_sub, "Calibrated_Sub_Class"] = canonical_sub[protect_main & known_sub]
    if hasattr(ml, "normalize_proportion_string"):
        canonical_mix = mix.map(lambda value: ml.normalize_proportion_string(value, ml.canonicalize_subclass_label))
        out.loc[protect_main & known_mix, "Calibrated_Mix_Proportion"] = canonical_mix[protect_main & known_mix]

    changed_by_frozen_ml = protect_main & out["Calibrated_Source"].fillna("").astype(str).str.contains(
        "Machine_Learning|ML_", case=False, regex=True
    )
    out.loc[changed_by_frozen_ml, "Calibrated_Source"] = (
        "Protected_Upstream_Main(" + source[changed_by_frozen_ml] + ")"
    )
    out.loc[changed_by_frozen_ml, "Calibrated_Notes"] = (
        out.loc[changed_by_frozen_ml, "Calibrated_Notes"].fillna("").astype(str)
        + "; 前序主类别受保护，机器学习仅可补充缺失成分"
    )
    if not out.loc[protect_main, "Calibrated_Main_Class"].equals(canonical_main[protect_main]):
        raise AssertionError("ML changed an upstream-determined main class")
    return out, {
        "upstream_known_main_rows": int(protect_main.sum()),
        "upstream_known_sub_rows": int((protect_main & known_sub).sum()),
        "upstream_known_mix_rows": int((protect_main & known_mix).sum()),
        "frozen_ml_main_overrides_rejected": int(changed_by_frozen_ml.sum()),
        "ml_eligible_unknown_or_weak_main_rows": int((~protect_main).sum()),
        "upstream_main_changed_after_gate": 0,
    }


def apply_missing_height_sensitivity_gate(calibrated, models, x_all, verified_height, ml):
    """Withhold ML labels that depend on the legacy internal 15 m height fill."""
    import numpy as np
    import pandas as pd

    out = calibrated.copy()
    out_ids = out["BUILDINGSTRUCTUREID"].map(norm_id)
    heights = pd.to_numeric(verified_height, errors="coerce").copy()
    heights.index = heights.index.map(norm_id)
    missing_height = heights.reindex(out_ids).isna().to_numpy()
    source = out["Calibrated_Source"].fillna("").astype(str)
    ml_path = source.isin({"Machine_Learning_Predicted", "ML_Low_Confidence_Discard"}).to_numpy()
    target = missing_height & ml_path

    audit_columns = [
        "BUILDINGSTRUCTUREID",
        "Candidate_Main_Class_at_15m",
        "Candidate_Sub_Class_at_15m",
        "Candidate_Mix_at_15m",
        "Sensitivity_Status",
        "Unique_Main_Classes",
        *[f"Main_at_{height:g}m" for height in HEIGHT_SENSITIVITY_SCENARIOS_M],
    ]
    if not target.any():
        return out, pd.DataFrame(columns=audit_columns), {
            "scenario_heights_m": list(HEIGHT_SENSITIVITY_SCENARIOS_M),
            "missing_height_ml_path_rows": 0,
            "stable_main_rows": 0,
            "withheld_unstable_main_rows": 0,
            "subclass_and_mixture_withheld_rows": 0,
        }

    expected_models = ["住宅类别", "商业类别", "工业类别"]
    if list(models) != expected_models:
        raise AssertionError(f"Unexpected frozen ML model order: {list(models)}")
    x_indexed = x_all.copy()
    x_indexed.index = x_indexed.index.map(norm_id)
    target_ids = out_ids[target]
    x_target = x_indexed.reindex(target_ids).copy()
    if x_target.isna().any().any():
        raise AssertionError("Missing-height sensitivity matrix contains unexpected nulls")

    scenario_labels = {}
    frozen_scenario_labels = {}
    for height in HEIGHT_SENSITIVITY_SCENARIOS_M:
        scenario = x_target.copy()
        scenario["Estimated_Height"] = height
        probabilities = np.column_stack(
            [models[name].predict_proba(scenario)[:, 1] for name in expected_models]
        )
        active = probabilities >= 0.5
        codes = (active * np.array([1, 2, 4])).sum(axis=1)
        frozen_code_to_main = {
            0: "Non-assessed_非评估类别",
            1: "Residential_住宅类别",
            2: "Commercial_商业类别",
            3: "Mixed-use_混合用途",
            4: "Industrial_工业类别",
            5: "Mixed-use_混合用途",
            6: "Mixed-use_混合用途",
            7: "Mixed-use_混合用途",
        }
        sensitivity_code_to_main = {**frozen_code_to_main, 0: "Unknown_未知类别"}
        scenario_labels[height] = pd.Series(
            [sensitivity_code_to_main[int(code)] for code in codes], index=target_ids.index
        )
        frozen_scenario_labels[height] = pd.Series(
            [frozen_code_to_main[int(code)] for code in codes], index=target_ids.index
        )

    label_frame = pd.DataFrame(scenario_labels)
    candidate_main = out.loc[out.index[target], "Calibrated_Main_Class"].map(
        ml.canonicalize_main_label
    )
    if not candidate_main.reset_index(drop=True).equals(
        frozen_scenario_labels[15.0].reset_index(drop=True)
    ):
        raise AssertionError("15 m sensitivity baseline does not reproduce frozen ML main classes")
    unique_count = label_frame.nunique(axis=1)
    stable = unique_count.eq(1)
    unstable = ~stable
    target_positions = out.index[target]
    stable_positions = target_positions[stable.to_numpy()]
    unstable_positions = target_positions[unstable.to_numpy()]

    out["ML_Height_Sensitivity_Status"] = ""
    out["ML_Height_Sensitivity_Candidate_Main"] = ""
    out.loc[target_positions, "ML_Height_Sensitivity_Candidate_Main"] = out.loc[
        target_positions, "Calibrated_Main_Class"
    ].to_numpy()
    out.loc[stable_positions, "ML_Height_Sensitivity_Status"] = "stable_main_3_to_150m"
    out.loc[unstable_positions, "ML_Height_Sensitivity_Status"] = "withheld_unstable_main_3_to_150m"

    unknown_main = ml.canonicalize_main_label("未知类别")
    unknown_sub = ml.canonicalize_subclass_label("未知类别")
    out.loc[unstable_positions, "Calibrated_Main_Class"] = unknown_main
    out.loc[unstable_positions, "Calibrated_Sub_Class"] = unknown_sub
    out.loc[unstable_positions, "Calibrated_Mix_Proportion"] = ""
    out.loc[unstable_positions, "Calibrated_Source"] = "ML_Withheld_Missing_Height_Sensitive"
    out.loc[unstable_positions, "Calibrated_Notes"] = (
        "ML candidate withheld: main class changes across 3-150 m height scenarios_"
        "机器学习候选因主类别随3至150米高度情景变化而撤回"
    )

    # The main class is robust for these rows, but height-sensitive subclass
    # regressions and mixture proportions are not published without a height.
    out.loc[stable_positions, "Calibrated_Main_Class"] = label_frame.loc[
        stable, HEIGHT_SENSITIVITY_SCENARIOS_M[0]
    ].to_numpy()
    out.loc[stable_positions, "Calibrated_Sub_Class"] = unknown_sub
    out.loc[stable_positions, "Calibrated_Mix_Proportion"] = ""
    out.loc[stable_positions, "Calibrated_Source"] = "ML_Height_Robust_Main_Only"
    out.loc[stable_positions, "Calibrated_Notes"] = (
        "ML main class stable across 3-150 m; subclass and mixture withheld_"
        "机器学习主类别在3至150米情景下稳定，子类别和混合比例因高度缺失而撤回"
    )
    stable_unknown = stable_positions[
        out.loc[stable_positions, "Calibrated_Main_Class"].eq(unknown_main).to_numpy()
    ]
    out.loc[stable_unknown, "Calibrated_Source"] = "ML_Withheld_Missing_Height_No_Active_Class"
    out.loc[stable_unknown, "Calibrated_Notes"] = (
        "No active ML class across 3-150 m height scenarios_"
        "在3至150米高度情景下均无达到阈值的机器学习主类别"
    )

    audit = pd.DataFrame({
        "BUILDINGSTRUCTUREID": target_ids.to_numpy(),
        "Candidate_Main_Class_at_15m": calibrated.loc[target_positions, "Calibrated_Main_Class"].to_numpy(),
        "Candidate_Sub_Class_at_15m": calibrated.loc[target_positions, "Calibrated_Sub_Class"].to_numpy(),
        "Candidate_Mix_at_15m": calibrated.loc[target_positions, "Calibrated_Mix_Proportion"].to_numpy(),
        "Sensitivity_Status": np.where(stable, "stable_main_3_to_150m", "withheld_unstable_main_3_to_150m"),
        "Unique_Main_Classes": unique_count.to_numpy(),
    })
    for height in HEIGHT_SENSITIVITY_SCENARIOS_M:
        audit[f"Main_at_{height:g}m"] = label_frame[height].to_numpy()

    summary = {
        "scenario_heights_m": list(HEIGHT_SENSITIVITY_SCENARIOS_M),
        "missing_height_ml_path_rows": int(target.sum()),
        "stable_main_rows": int(stable.sum()),
        "stable_known_main_rows": int((stable & label_frame.iloc[:, 0].ne(unknown_main)).sum()),
        "stable_unknown_main_rows": int((stable & label_frame.iloc[:, 0].eq(unknown_main)).sum()),
        "withheld_unstable_main_rows": int(unstable.sum()),
        "subclass_and_mixture_withheld_rows": int(target.sum()),
    }
    return out, audit[audit_columns], summary


def apply_missing_floor_mixture_gate(calibrated, verified_floors, ml):
    """Withhold floor-derived frozen-rule mixtures when no floor count is verified."""
    import pandas as pd

    out = calibrated.copy()
    out_ids = out["BUILDINGSTRUCTUREID"].map(norm_id)
    floors = pd.to_numeric(verified_floors, errors="coerce").copy()
    floors.index = floors.index.map(norm_id)
    missing_floor = floors.reindex(out_ids).isna().to_numpy()
    source = out["Calibrated_Source"].fillna("").astype(str)
    affected = missing_floor & source.eq("Rule_Engine(Fusion_Inferred_Mixed)").to_numpy()
    positions = out.index[affected]
    audit = out.loc[
        positions,
        [
            "BUILDINGSTRUCTUREID",
            "Calibrated_Main_Class",
            "Calibrated_Sub_Class",
            "Calibrated_Mix_Proportion",
            "Calibrated_Source",
        ],
    ].copy()
    audit = audit.rename(
        columns={
            "Calibrated_Sub_Class": "Candidate_Sub_Class",
            "Calibrated_Mix_Proportion": "Candidate_Mix_Proportion",
        }
    )
    out["Floor_Missing_Classification_Gate"] = ""
    if len(positions):
        out.loc[positions, "Calibrated_Sub_Class"] = ml.canonicalize_subclass_label("未知类别")
        out.loc[positions, "Calibrated_Mix_Proportion"] = ""
        out.loc[positions, "Floor_Missing_Classification_Gate"] = (
            "mixed_main_retained_subclass_and_mixture_withheld"
        )
        out.loc[positions, "Calibrated_Notes"] = (
            out.loc[positions, "Calibrated_Notes"].fillna("").astype(str)
            + "; mixed main retained; floor-derived subclass and mixture withheld_"
              "保留混合用途主类别，因层数缺失撤回楼层派生的子类别和比例"
        )
    return out, audit, {
        "missing_floor_rule_mixed_rows": int(affected.sum()),
        "main_class_changed_rows": 0,
        "subclass_and_mixture_withheld_rows": int(affected.sum()),
    }


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
    calibrated, upstream_gate = protect_upstream_classifications(calibrated, ml)
    calibrated, height_sensitivity_audit, height_sensitivity_gate = (
        apply_missing_height_sensitivity_gate(
            calibrated,
            models,
            X_all,
            phys_idx["Height_m_建筑高度"],
            ml,
        )
    )
    calibrated, floor_mixture_audit, floor_mixture_gate = apply_missing_floor_mixture_gate(
        calibrated,
        phys_idx["Floors_Count_地上层数"],
        ml,
    )
    height_sensitivity_audit.to_csv(
        OUT / "height_missing_ml_sensitivity_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "height_missing_ml_sensitivity_summary.json").write_text(
        json.dumps(height_sensitivity_gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    floor_mixture_audit.to_csv(
        OUT / "floor_missing_rule_mixture_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "floor_missing_rule_mixture_summary.json").write_text(
        json.dumps(floor_mixture_gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
        "classification_order_gate": upstream_gate,
        "missing_height_sensitivity_gate": height_sensitivity_gate,
        "missing_floor_mixture_gate": floor_mixture_gate,
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
