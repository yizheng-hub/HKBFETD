"""Fail closed on unbound VLM decisions and apply verified physical scope."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(os.environ["HKBFETD_REPLAY_OUTPUT_DIR"]).resolve()
STEP5 = RUN / "step5_merged_classification.csv"
PHYSICAL = Path(os.environ["HKBFETD_REPLAY_VERIFIED_PHYSICAL"]).resolve()
NONBUILDING_SCOPE = Path(os.environ["HKBFETD_REPLAY_NONBUILDING_SCOPE"]).resolve()
SUGGESTIONS = Path(os.environ["HKBFETD_REPLAY_STAGE6_SUGGESTIONS"]).resolve()
VLM_AUDIT = Path(os.environ["HKBFETD_REPLAY_VLM_AUDIT"]).resolve()
OUT = RUN / "step7_corrected.csv"
CHANGES = RUN / "stage7_classification_changes.csv"
WITHHELD = RUN / "stage7_withheld_vlm_decisions.csv"
MANIFEST = RUN / "stage7_build_manifest.json"


INHERIT = {
    "Consensus_merge",
    "Final_keep_with_inheritance",
    "Rule-Only_merge",
    "Final_merge_(AI_overrides_Rule_Delete)",
}
DELETE = {
    "Consensus_delete",
    "Final_delete (AI overrides Rule_Merge)",
    "Rule-Only_delete",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def prefixed(names, prefix):
    return next(name for name in names if name == prefix or name.startswith(prefix + "_"))


def main() -> None:
    print("[1/6] Read stage 5", flush=True)
    base = pd.read_csv(STEP5, dtype={"BUILDINGSTRUCTUREID": "string"}, low_memory=False)
    base["BUILDINGSTRUCTUREID"] = base["BUILDINGSTRUCTUREID"].map(norm_id)
    if len(base) != 341153 or not base["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Stage 5 must contain 341,153 unique IDs")
    result = base.set_index("BUILDINGSTRUCTUREID", drop=False).copy()
    result["Stage7_Repair_Action"] = "unchanged"

    print("[2/6] Withhold VLM decisions without a current input contract", flush=True)
    suggestions = pd.read_csv(SUGGESTIONS, dtype={"fragment_id": str, "host_id": str}, low_memory=False)
    suggestions["fragment_id"] = suggestions["fragment_id"].map(norm_id)
    suggestions["host_id"] = suggestions["host_id"].map(norm_id)
    decisions = suggestions[suggestions["final_suggestion"].isin(INHERIT | DELETE)].copy()
    decisions["VLM_Reuse_Status"] = "withheld_missing_current_pair_prompt_image_contract"
    decisions.to_csv(WITHHELD, index=False, encoding="utf-8-sig")
    affected = set(decisions["fragment_id"]) & set(result.index)
    result.loc[list(affected), "Stage7_Repair_Action"] = (
        "vlm_checkpoint_withheld_missing_current_input_contract"
    )

    print("[3/6] Overlay verified physical values", flush=True)
    physical = pd.read_parquet(PHYSICAL)
    id_col = prefixed(physical.columns, "Building_ID")
    floor_col = prefixed(physical.columns, "Floors_Count")
    height_col = prefixed(physical.columns, "Height_m")
    area_col = prefixed(physical.columns, "Footprint_Area_sqm")
    physical["BUILDINGSTRUCTUREID"] = physical[id_col].map(norm_id)
    physical = physical.set_index("BUILDINGSTRUCTUREID", verify_integrity=True).reindex(result.index)
    if physical.index.hasnans or physical["BUILDINGCSUID"].astype(str).ne(
        result["BUILDINGCSUID"].astype(str)
    ).any():
        raise AssertionError("Verified physical table does not align with stage 5")

    floors = pd.to_numeric(physical[floor_col], errors="coerce")
    height = pd.to_numeric(physical[height_col], errors="coerce")
    area = pd.to_numeric(physical[area_col], errors="coerce")
    result["NUMABOVEGROUNDSTOREYS"] = floors
    result["Estimated_Floors"] = floors
    result["Estimated_Height"] = height
    result["Corrected_Structure_Height_m"] = height
    result["SHAPE_Area"] = area
    result["GFA_DOMESTIC_SUM"] = np.nan
    result["GFA_NONDOMESTIC_SUM"] = np.nan

    print("[4/6] Apply verified nonbuilding scope", flush=True)
    scope = pd.read_csv(
        NONBUILDING_SCOPE,
        dtype={"BUILDINGSTRUCTUREID": str, "BUILDINGCSUID": str},
        encoding="utf-8-sig",
    )
    scope["BUILDINGSTRUCTUREID"] = scope["BUILDINGSTRUCTUREID"].map(norm_id)
    if len(scope) != 19 or not scope["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Verified nonbuilding scope must contain 19 unique IDs")
    scope = scope.set_index("BUILDINGSTRUCTUREID", verify_integrity=True)
    if not set(scope.index).issubset(result.index):
        raise AssertionError("Verified nonbuilding scope contains an unknown building")
    if scope["BUILDINGCSUID"].astype(str).ne(result.loc[scope.index, "BUILDINGCSUID"].astype(str)).any():
        raise AssertionError("Verified nonbuilding scope does not align by ID and CSUID")

    sub_by_type = {
        "bridge_structure": "Transport Infrastructure_交通基础设施",
        "water_storage_structure": "Temporary/Misc Facilities_临时/杂项设施",
        "equipment_structure": "Temporary/Misc Facilities_临时/杂项设施",
    }
    if not scope["Structure_Use_Type"].isin(sub_by_type).all():
        raise AssertionError("Unsupported verified nonbuilding structure type")
    for building_id, row in scope.iterrows():
        sub = sub_by_type[row["Structure_Use_Type"]]
        result.at[building_id, "Final_Main_Class"] = "Non-assessed_非评估类别"
        result.at[building_id, "Final_Sub_Class"] = sub
        result.at[building_id, "Classification_Source"] = "Verified_nonbuilding_physical_scope"
        result.at[building_id, "Is_Mixed_Use"] = False
        result.at[building_id, "Main_Class_Proportions"] = "Non-assessed_非评估类别:1.00"
        result.at[building_id, "Sub_Class_Proportions"] = f"{sub}:1.00"
        result.at[building_id, "Confidence_Score"] = 1.0
        result.at[building_id, "Evidence_Details"] = str([
            f"Physical_scope:{row['Structure_Use_Type']}",
            f"Source:{row['Structure_Use_Evidence']}",
        ])
        result.at[building_id, "Stage7_Repair_Action"] = "verified_nonbuilding_structure_excluded_before_ML"
        result.at[building_id, "Structure_Use_Type"] = row["Structure_Use_Type"]
        result.at[building_id, "Structure_Use_Name"] = row["Structure_Use_Name"]
        result.at[building_id, "Structure_Use_Source"] = row["Structure_Use_Source"]
        result.at[building_id, "Structure_Use_Evidence"] = row["Structure_Use_Evidence"]
        result.at[building_id, "Structure_Use_Quality"] = row["Structure_Use_Quality"]
        result.at[building_id, "Energy_Assessment_Scope"] = "excluded_nonbuilding_structure"

    print("[5/6] Enforce evidence-scope gates and write output", flush=True)
    unsafe_bdbiar = result["BDBIAR_Match_Quality"].ne("inside_unique") | ~result[
        "BDBIAR_Source_Attributes_Usable"
    ].fillna(False).astype(bool)
    cites_bdbiar = result["Classification_Source"].astype(str).str.contains("BDBIAR", regex=False)
    cites_gfa = result["Classification_Source"].astype(str).str.contains("GFA", regex=False)
    if (unsafe_bdbiar & cites_bdbiar).any():
        raise AssertionError("Stage 7 contains unsafe BDBIAR classification evidence")
    if cites_gfa.any():
        raise AssertionError("Stage 7 contains unusable permit GFA classification evidence")
    if result["Classification_Source"].astype(str).str.startswith("Inherited_via_").any():
        raise AssertionError("An uncontracted VLM inheritance survived fail-closed handling")

    result = result.loc[base["BUILDINGSTRUCTUREID"]]
    result.reset_index(drop=True).to_csv(OUT, index=False, encoding="utf-8-sig")

    compare = pd.DataFrame(index=result.index)
    changed = pd.Series(False, index=result.index)
    base_i = base.set_index("BUILDINGSTRUCTUREID", verify_integrity=True)
    for field in ["Final_Main_Class", "Final_Sub_Class", "Classification_Source"]:
        compare["Stage5_" + field] = base_i.loc[result.index, field]
        compare["Stage7_" + field] = result[field]
        field_changed = compare["Stage5_" + field].fillna("").astype(str).ne(
            compare["Stage7_" + field].fillna("").astype(str)
        )
        compare[field + "_Changed"] = field_changed
        changed |= field_changed
    compare["Stage7_Repair_Action"] = result["Stage7_Repair_Action"]
    compare.loc[changed].reset_index().to_csv(CHANGES, index=False, encoding="utf-8-sig")

    manifest = {
        "rows": len(result),
        "vlm_candidate_decisions": len(decisions),
        "vlm_decisions_applied": 0,
        "vlm_decisions_withheld_missing_current_contract": len(decisions),
        "verified_nonbuilding_records": len(scope),
        "classification_rows_changed_from_stage5": int(changed.sum()),
        "verified_height_known": int(height.notna().sum()),
        "verified_height_unknown": int(height.isna().sum()),
        "verified_floors_known": int(floors.notna().sum()),
        "verified_floors_unknown": int(floors.isna().sum()),
        "unsafe_bdbiar_sources": 0,
        "unusable_gfa_sources": 0,
        "input_sha256": {
            "stage5": sha256(STEP5),
            "verified_physical": sha256(PHYSICAL),
            "nonbuilding_scope": sha256(NONBUILDING_SCOPE),
            "stage6_suggestions_comparison_only": sha256(SUGGESTIONS),
            "vlm_audit_comparison_only": sha256(VLM_AUDIT),
        },
        "output_sha256": {OUT.name: sha256(OUT)},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[6/6] " + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
