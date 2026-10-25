"""Apply the frozen original Step 10 classification standardization.

This adapter keeps all Step 9 columns and changes only the three public
classification fields using the functions and mapping frozen at ``de8b46b``.
It is intentionally separate from package formatting and physical attributes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pandas as pd


FROZEN_ROOT = Path(os.environ["HKBFETD_REPLAY_FROZEN_METHOD"]).resolve()
FROZEN_SRC = FROZEN_ROOT / "src"
ML_OUTPUT = Path(os.environ["HKBFETD_REPLAY_ML_OUTPUT_DIR"]).resolve()
RAW_STEP9 = ML_OUTPUT / "step9_corrected_ml_calibrated.csv"
OUT = ML_OUTPUT / "step10_original_method_standardized.csv"
CHANGES = ML_OUTPUT / "step10_original_method_standardization_changes.csv"
MANIFEST = ML_OUTPUT / "step10_original_method_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_step10():
    sys.path.insert(0, str(FROZEN_SRC))
    spec = importlib.util.spec_from_file_location(
        "hkbfetd_frozen_original_step10",
        FROZEN_SRC / "10.final_statistics_report.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def norm_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def main() -> None:
    if not RAW_STEP9.is_file():
        raise FileNotFoundError(f"Frozen Step 9 output is missing: {RAW_STEP9}")

    step10 = load_step10()
    raw = pd.read_csv(
        RAW_STEP9,
        low_memory=False,
        encoding="utf-8-sig",
        dtype={"BUILDINGSTRUCTUREID": str},
    )
    raw["BUILDINGSTRUCTUREID"] = raw["BUILDINGSTRUCTUREID"].map(norm_id)
    if len(raw) != 341153 or not raw["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Step 9 must contain 341,153 unique building IDs")

    standardized, repair_changes = step10.repair_subclass_consistency(raw)
    standardized["Calibrated_Main_Class"] = (
        standardized["Calibrated_Main_Class"]
        .map(step10.MAIN_CLASS_DICT)
        .fillna(standardized["Calibrated_Main_Class"])
    )
    standardized["Calibrated_Sub_Class"] = (
        standardized["Calibrated_Sub_Class"]
        .apply(step10.patch_orphan_subclass_label)
        .apply(step10.apply_release_label_patches)
    )
    standardized["Calibrated_Mix_Proportion"] = (
        standardized["Calibrated_Mix_Proportion"]
        .apply(step10.translate_mix_prop)
        .apply(step10.apply_release_label_patches)
    )

    public_fields = standardized[
        [
            "BUILDINGSTRUCTUREID",
            "Calibrated_Main_Class",
            "Calibrated_Sub_Class",
            "Calibrated_Mix_Proportion",
        ]
    ].rename(
        columns={
            "BUILDINGSTRUCTUREID": "Building_ID_建筑ID",
            "Calibrated_Main_Class": "UBEM_Main_Class_主类别",
            "Calibrated_Sub_Class": "UBEM_Sub_Class_子类别",
            "Calibrated_Mix_Proportion": "UBEM_Mixed_Proportions_混合比例",
        }
    )
    public_fields = step10.enforce_public_non_assessed_fields(public_fields)
    standardized["Calibrated_Main_Class"] = public_fields["UBEM_Main_Class_主类别"].to_numpy()
    standardized["Calibrated_Sub_Class"] = public_fields["UBEM_Sub_Class_子类别"].to_numpy()
    standardized["Calibrated_Mix_Proportion"] = public_fields[
        "UBEM_Mixed_Proportions_混合比例"
    ].to_numpy()

    comparison = raw[
        [
            "BUILDINGSTRUCTUREID",
            "Calibrated_Main_Class",
            "Calibrated_Sub_Class",
            "Calibrated_Mix_Proportion",
        ]
    ].merge(
        standardized[
            [
                "BUILDINGSTRUCTUREID",
                "Calibrated_Main_Class",
                "Calibrated_Sub_Class",
                "Calibrated_Mix_Proportion",
            ]
        ],
        on="BUILDINGSTRUCTUREID",
        suffixes=("_raw_step9", "_step10"),
        validate="one_to_one",
    )
    changed = pd.Series(False, index=comparison.index)
    for field in (
        "Calibrated_Main_Class",
        "Calibrated_Sub_Class",
        "Calibrated_Mix_Proportion",
    ):
        changed |= (
            comparison[f"{field}_raw_step9"].fillna("").astype(str)
            != comparison[f"{field}_step10"].fillna("").astype(str)
        )
    comparison = comparison.loc[changed].copy()
    comparison.to_csv(CHANGES, index=False, encoding="utf-8-sig")
    standardized.to_csv(OUT, index=False, encoding="utf-8-sig")

    qa = step10.qa_subclass_consistency(standardized)
    if len(qa):
        raise AssertionError(f"Frozen Step 10 left {len(qa)} subclass consistency issues")

    report = {
        "status": "completed",
        "classification_method": "frozen_original_step10",
        "input_rows": len(raw),
        "output_rows": len(standardized),
        "step10_repair_records": len(repair_changes),
        "public_standardization_changed_records": len(comparison),
        "remaining_step10_consistency_issues": len(qa),
        "input_sha256": sha256(RAW_STEP9),
        "output_sha256": sha256(OUT),
        "changes_sha256": sha256(CHANGES),
        "frozen_step10_sha256": sha256(FROZEN_SRC / "10.final_statistics_report.py"),
        "frozen_public_mapping_sha256": sha256(
            FROZEN_ROOT / "ctl" / "step10_public_mappings.json"
        ),
    }
    MANIFEST.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
