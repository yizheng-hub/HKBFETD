"""Apply the frozen stage-6 suggestion checkpoint to corrected stage 5.

Only the original inheritance/delete decisions and original field-update scope
are applied.  The frozen method inherits main class, subclass and source; it
does not rewrite the fragment's proportion fields or mixed-use flag.  Verified
physical values then replace the legacy stage-7 five-floor/15 m defaults.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


RUN = Path(os.environ["HKBFETD_REPLAY_OUTPUT_DIR"]).resolve()
STEP5 = RUN / "step5_merged_classification.csv"
PHYSICAL = Path(os.environ["HKBFETD_REPLAY_VERIFIED_PHYSICAL"]).resolve()
NONBUILDING_SCOPE = Path(os.environ["HKBFETD_REPLAY_NONBUILDING_SCOPE"]).resolve()
SUGGESTIONS = Path(os.environ["HKBFETD_REPLAY_STAGE6_SUGGESTIONS"]).resolve()
VLM_AUDIT = Path(os.environ["HKBFETD_REPLAY_VLM_AUDIT"]).resolve()
OUT = RUN / "step7_corrected.csv"
CHANGES = RUN / "stage7_classification_changes.csv"
MANIFEST = RUN / "stage7_build_manifest.json"

EXPECTED_SUGGESTIONS_SHA256 = "09898410ac58f67f9932055ad0339cbda4078d5c5e301de02dcc4d9c5f947dba"
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
EXPECTED_VLM_TARGET_IDS = ["365549", "429041", "259298", "403078", "429673"]
EXPECTED_VLM_SPECIAL_PAIR = {"fragment_id": "428882", "host_id": "429041"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def validate_vlm_audit(vlm_audit: dict) -> None:
    """Require the exact structured audit conclusion used by this replay."""
    if vlm_audit.get("prior_vlm_merge_still_supported_by_new_geometry") is not True:
        raise AssertionError("VLM audit does not explicitly approve reuse")

    scope = vlm_audit.get("scope")
    if not isinstance(scope, dict):
        raise AssertionError("VLM audit scope is missing")
    target_ids = [norm_id(value) for value in scope.get("target_ids", [])]
    if target_ids != EXPECTED_VLM_TARGET_IDS:
        raise AssertionError("VLM audit target IDs differ from the frozen review scope")
    special_pair = scope.get("special_pair")
    if not isinstance(special_pair, dict):
        raise AssertionError("VLM audit special pair is missing")
    normalized_pair = {
        "fragment_id": norm_id(special_pair.get("fragment_id")),
        "host_id": norm_id(special_pair.get("host_id")),
    }
    if normalized_pair != EXPECTED_VLM_SPECIAL_PAIR:
        raise AssertionError("VLM audit special pair differs from the frozen review scope")

    comparisons = vlm_audit.get("geometry_change_comparison")
    if not isinstance(comparisons, list):
        raise AssertionError("VLM audit geometry comparison is missing")
    comparison_ids = [norm_id(row.get("BUILDINGSTRUCTUREID")) for row in comparisons]
    if comparison_ids != EXPECTED_VLM_TARGET_IDS:
        raise AssertionError("VLM audit geometry rows do not cover the exact target IDs")
    for row in comparisons:
        if row.get("old_present") is not True or row.get("new_present") is not True:
            raise AssertionError("VLM audit target geometry is missing")
        if row.get("old_geometry", {}).get("valid") is not True:
            raise AssertionError("VLM audit old geometry is not explicitly valid")
        if row.get("new_geometry", {}).get("valid") is not True:
            raise AssertionError("VLM audit new geometry is not explicitly valid")

    prior = vlm_audit.get("prior_vlm_decision")
    if not isinstance(prior, dict):
        raise AssertionError("VLM audit prior decision is missing")
    prior_pair = {
        "fragment_id": norm_id(prior.get("fragment_id")),
        "host_id": norm_id(prior.get("host_id")),
    }
    if prior_pair != EXPECTED_VLM_SPECIAL_PAIR or prior.get("final_suggestion") != "Consensus_merge":
        raise AssertionError("VLM audit prior merge decision differs from the frozen decision")


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    if sha256(SUGGESTIONS) != EXPECTED_SUGGESTIONS_SHA256:
        raise AssertionError("Stage-6 suggestion checkpoint changed")
    if not VLM_AUDIT.is_file():
        raise FileNotFoundError("Independent geometry/VLM cache audit is not complete")
    vlm_audit = json.loads(VLM_AUDIT.read_text(encoding="utf-8-sig"))
    validate_vlm_audit(vlm_audit)

    base = pd.read_csv(STEP5, dtype={"BUILDINGSTRUCTUREID": str}, low_memory=False, encoding="utf-8-sig")
    base["BUILDINGSTRUCTUREID"] = base["BUILDINGSTRUCTUREID"].map(norm_id)
    if len(base) != 341153 or not base["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Stage 5 must contain 341,153 unique IDs")
    result = base.set_index("BUILDINGSTRUCTUREID", drop=False).copy()
    result["Stage7_Repair_Action"] = "unchanged"
    guarded_fields = [
        "Final_Main_Class",
        "Final_Sub_Class",
        "Classification_Source",
        "Main_Class_Proportions",
        "Sub_Class_Proportions",
        "Is_Mixed_Use",
    ]
    classification_guard = result[guarded_fields].copy()

    suggestions = pd.read_csv(SUGGESTIONS, dtype={"fragment_id": str, "host_id": str}, low_memory=False)
    suggestions["fragment_id"] = suggestions["fragment_id"].map(norm_id)
    suggestions["host_id"] = suggestions["host_id"].map(norm_id)
    inherited = suggestions[suggestions["final_suggestion"].isin(INHERIT)].copy()
    if inherited["fragment_id"].duplicated().any():
        raise AssertionError("Frozen inheritance decisions contain multiple hosts for one fragment")

    host_fields = ["Final_Main_Class", "Final_Sub_Class"]
    inherited_count = 0
    for row in inherited.itertuples(index=False):
        fragment_id = norm_id(row.fragment_id)
        host_id = norm_id(row.host_id)
        if fragment_id not in result.index or host_id not in result.index:
            raise AssertionError(f"Missing inheritance ID: {fragment_id} <- {host_id}")
        for field in host_fields:
            if field in result.columns:
                result.at[fragment_id, field] = result.at[host_id, field]
        result.at[fragment_id, "Classification_Source"] = f"Inherited_via_{row.final_suggestion}"
        result.at[fragment_id, "Stage7_Repair_Action"] = f"inherited_from_{host_id}"
        inherited_count += 1

    delete_ids = set(suggestions.loc[suggestions["final_suggestion"].isin(DELETE), "fragment_id"])
    delete_ids &= set(result.index)
    for building_id in delete_ids:
        result.at[building_id, "Final_Main_Class"] = "Non-assessed_非评估类别"
        result.at[building_id, "Final_Sub_Class"] = "Absolute Noise_绝对噪声相关"
        result.at[building_id, "Classification_Source"] = "Deleted_AI_Rule_Verified"
        result.at[building_id, "Stage7_Repair_Action"] = "deleted_by_frozen_vlm_decision"

    decision_ids = set(inherited["fragment_id"]) | delete_ids
    outside_decisions = ~result.index.isin(decision_ids)
    for field in ("Final_Main_Class", "Final_Sub_Class", "Classification_Source"):
        if not result.loc[outside_decisions, field].equals(
            classification_guard.loc[outside_decisions, field]
        ):
            raise AssertionError(f"Frozen Stage 7 changed {field} outside its decision IDs")
    for field in ("Main_Class_Proportions", "Sub_Class_Proportions", "Is_Mixed_Use"):
        if not result[field].equals(classification_guard[field]):
            raise AssertionError(f"Frozen Stage 7 is not allowed to change {field}")

    physical_columns = [
        "Building_ID_建筑ID",
        "BUILDINGCSUID",
        "Floors_Count_地上层数",
        "Height_m_建筑高度",
        "Footprint_Area_sqm_占地面积",
        "GFA_DOMESTIC_SUM",
        "GFA_NONDOMESTIC_SUM",
        "Floors_Source",
        "Floors_Quality",
        "GFA_Proxy_Method",
        "GFA_Quality",
    ]
    physical = pd.read_parquet(PHYSICAL, columns=physical_columns)
    physical["BUILDINGSTRUCTUREID"] = physical["Building_ID_建筑ID"].map(norm_id)
    physical = physical.set_index("BUILDINGSTRUCTUREID", verify_integrity=True).reindex(result.index)
    if physical.index.hasnans or physical["BUILDINGCSUID"].astype(str).ne(result["BUILDINGCSUID"].astype(str)).any():
        raise AssertionError("Verified physical table does not align with stage 5 by ID and CSUID")

    floors = pd.to_numeric(physical["Floors_Count_地上层数"], errors="coerce")
    height = pd.to_numeric(physical["Height_m_建筑高度"], errors="coerce")
    area = pd.to_numeric(physical["Footprint_Area_sqm_占地面积"], errors="coerce")
    result["NUMABOVEGROUNDSTOREYS"] = floors
    result["Estimated_Floors"] = floors
    result["Estimated_Height"] = height
    result["Corrected_Structure_Height_m"] = height
    result["SHAPE_Area"] = area
    result["GFA_DOMESTIC_SUM"] = pd.to_numeric(physical["GFA_DOMESTIC_SUM"], errors="coerce")
    result["GFA_NONDOMESTIC_SUM"] = pd.to_numeric(physical["GFA_NONDOMESTIC_SUM"], errors="coerce")

    scope = pd.read_csv(NONBUILDING_SCOPE, dtype={"BUILDINGSTRUCTUREID": str, "BUILDINGCSUID": str}, encoding="utf-8-sig")
    scope["BUILDINGSTRUCTUREID"] = scope["BUILDINGSTRUCTUREID"].map(norm_id)
    if len(scope) != 19 or not scope["BUILDINGSTRUCTUREID"].is_unique or not scope["BUILDINGCSUID"].is_unique:
        raise AssertionError("Verified nonbuilding scope must contain 19 unique IDs and CSUIDs")
    scope = scope.set_index("BUILDINGSTRUCTUREID", verify_integrity=True)
    if not set(scope.index).issubset(result.index):
        raise AssertionError("Verified nonbuilding scope contains an unknown building ID")
    if scope["BUILDINGCSUID"].astype(str).ne(result.loc[scope.index, "BUILDINGCSUID"].astype(str)).any():
        raise AssertionError("Verified nonbuilding scope does not align by ID and CSUID")
    for field in ("Floors_Source", "Floors_Quality", "GFA_Proxy_Method", "GFA_Quality"):
        if not physical.loc[scope.index, field].astype(str).eq("not_applicable_nonbuilding_structure").all():
            raise AssertionError(f"Nonbuilding scope conflicts with verified physical field {field}")
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

    result = result.loc[base["BUILDINGSTRUCTUREID"]].reset_index(drop=True)
    if len(result) != 341153 or not result["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Corrected stage 7 lost or duplicated records")
    if int(pd.to_numeric(result["Estimated_Height"], errors="coerce").isna().sum()) != 112897:
        raise AssertionError("Corrected stage 7 did not preserve verified unknown heights")
    result.to_csv(OUT, index=False, encoding="utf-8-sig")

    compare = pd.DataFrame({"BUILDINGSTRUCTUREID": base["BUILDINGSTRUCTUREID"]})
    changed_mask = pd.Series(False, index=base.index)
    for field in ["Final_Main_Class", "Final_Sub_Class", "Classification_Source", "Main_Class_Proportions", "Sub_Class_Proportions"]:
        compare[f"stage5_{field}"] = base[field]
        compare[f"stage7_{field}"] = result[field]
        changed_mask |= base[field].fillna("<NA>").astype(str).ne(result[field].fillna("<NA>").astype(str))
    compare["Stage7_Repair_Action"] = result["Stage7_Repair_Action"]
    compare = compare[changed_mask].copy()
    compare.to_csv(CHANGES, index=False, encoding="utf-8-sig")

    manifest = {
        "rows": len(result),
        "inherited_records": inherited_count,
        "deleted_records": len(delete_ids),
        "verified_nonbuilding_records": len(scope),
        "classification_or_proportion_rows_changed_from_stage5": len(compare),
        "verified_height_known": int(height.notna().sum()),
        "verified_height_unknown": int(height.isna().sum()),
        "verified_floors_known": int(floors.notna().sum()),
        "verified_floors_unknown": int(floors.isna().sum()),
        "legacy_five_floor_or_15m_defaults_written_to_physical_columns": False,
        "input_sha256": {
            "corrected_stage5": sha256(STEP5),
            "verified_physical": sha256(PHYSICAL),
            "nonbuilding_scope": sha256(NONBUILDING_SCOPE),
            "stage6_suggestions": sha256(SUGGESTIONS),
            "vlm_audit": sha256(VLM_AUDIT),
        },
        "output_sha256": {path.name: sha256(path) for path in [OUT, CHANGES]},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
