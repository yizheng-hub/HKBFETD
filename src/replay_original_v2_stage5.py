"""Rebuild stage 5 without changing established functional classifications.

The legacy stage-5 checkpoint remains authoritative for every building that
already had an original LLM cache entry.  A height or floor correction may
change the historical prompt text, but it is not evidence that the building's
use changed.  Native re-reviews of those prompt deltas are retained only as
diagnostic evidence.  Exact-original-prompt native decisions are applied only
to newly selected candidates that had no legacy cache entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


RUN = Path(os.environ["HKBFETD_REPLAY_OUTPUT_DIR"]).resolve()
PREPARER_SUMMARY = Path(os.environ["HKBFETD_REPLAY_LLM_REVIEW_SUMMARY"]).resolve()
PROMPT_CONTRACT = Path(os.environ["HKBFETD_REPLAY_LLM_PROMPT_CONTRACT"]).resolve()
NATIVE = Path(os.environ["HKBFETD_REPLAY_NATIVE_REVIEWS"]).resolve()
OLD_STEP4 = Path(os.environ["HKBFETD_REPLAY_LEGACY_STAGE4"]).resolve()
OLD_STEP5 = Path(os.environ["HKBFETD_REPLAY_LEGACY_STAGE5"]).resolve()
NEW_STEP3 = RUN / "step3_rule_engine_classification.csv"
OUT_STEP4 = RUN / "step4_combined_llm_evidence.csv"
OUT_STEP5 = RUN / "step5_merged_classification.csv"
OUT_AUDIT = RUN / "stage5_build_manifest.json"
OUT_CHANGES = RUN / "stage5_classification_changes.csv"
OUT_PHYSICAL_RECHECKS = RUN / "stage4_physical_prompt_rechecks_not_applied.csv"


MAIN_MAP = {
    "住宅类别": "Residential_住宅类别",
    "商业类别": "Commercial_商业类别",
    "工业类别": "Industrial_工业类别",
    "非评估类别": "Non-assessed_非评估类别",
    "混合用途": "Mixed-use_混合用途",
}
SUB_PARENT = {
    "公共房屋（Public Housing）": "住宅类别",
    "房委会资助出售单位（HA Subsidized Sale Flats）": "住宅类别",
    "私人房屋（Private Housing）": "住宅类别",
    "其他房屋（Other Housing）": "住宅类别",
    "食肆（Restaurant）": "商业类别",
    "零售（Retail）": "商业类别",
    "办公室（Office）": "商业类别",
    "住宿（Accommodation）": "商业类别",
    "医疗（Human Health）": "商业类别",
    "教育（Education）": "商业类别",
    "数据中心（Data Centre）": "商业类别",
    "其他商业（Other Commercial）": "商业类别",
    "纺织及服装制品（Textile & Wearing Apparel）": "工业类别",
    "非制造业（Non-manufacturing）": "工业类别",
    "食品及饮品（Food & Beverage）": "工业类别",
    "金属及机械（Metal & Machinery）": "工业类别",
    "其他工业（Other Industrial）": "工业类别",
    "绝对噪声相关": "非评估类别",
    "交通基础设施": "非评估类别",
    "临时/杂项设施": "非评估类别",
}
NON_ASSESSED_SUB_MAP = {
    "绝对噪声相关": "Absolute Noise_绝对噪声相关",
    "交通基础设施": "Transport Infrastructure_交通基础设施",
    "临时/杂项设施": "Temporary/Misc Facilities_临时/杂项设施",
}
CLASS_FIELDS = [
    "Final_Main_Class",
    "Final_Sub_Class",
    "Classification_Source",
    "Main_Class_Proportions",
    "Sub_Class_Proportions",
    "Is_Mixed_Use",
]
COMMERCIAL_POI_TAGS = [
    "[restaurant]", "[shop]", "[retail]", "[clinic]", "[bank]",
    "[school]", "[cafe]", "[fast_food]", "[supermarket]",
    "[pharmacy]", "[hospital]", "[kindergarten]", "[office]",
]
EXPECTED_BOTTOM_SHOP_PROTECTED_IDS = {
    "103579", "145296", "155864", "179273", "201976",
    "203663", "209627", "309843", "343319", "370628",
}
EXPECTED_NATIVE_REVIEW_COUNT = 506
PROMPT_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def validate_prompt_contract(
    contract: dict,
    reviews: list[dict],
    expected_status_by_id: dict[str, str],
) -> dict[str, dict]:
    """Bind every native result to the exact prepared prompt and review status."""
    if contract.get("schema_version") != 1:
        raise AssertionError("LLM prompt contract must have schema_version=1")
    records = contract.get("records")
    if not isinstance(records, list):
        raise AssertionError("LLM prompt contract records must be a list")
    if contract.get("record_count") != len(records) or len(records) != EXPECTED_NATIVE_REVIEW_COUNT:
        raise AssertionError(
            f"LLM prompt contract must contain exactly {EXPECTED_NATIVE_REVIEW_COUNT} records"
        )

    contract_by_id: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "BUILDINGSTRUCTUREID",
            "status",
            "new_prompt_sha256",
        }:
            raise AssertionError("Each LLM prompt contract record must use the exact contract fields")
        building_id = norm_id(record["BUILDINGSTRUCTUREID"])
        if building_id in contract_by_id:
            raise AssertionError(f"Duplicate building ID in LLM prompt contract: {building_id}")
        digest = str(record["new_prompt_sha256"])
        if PROMPT_HASH_PATTERN.fullmatch(digest) is None:
            raise AssertionError(f"Invalid prompt SHA-256 for building {building_id}")
        contract_by_id[building_id] = {
            "status": str(record["status"]),
            "new_prompt_sha256": digest,
        }

    if set(contract_by_id) != set(expected_status_by_id):
        raise AssertionError("LLM prompt contract IDs do not match the current review set")
    observed_status_by_id = {
        building_id: record["status"] for building_id, record in contract_by_id.items()
    }
    if observed_status_by_id != expected_status_by_id:
        raise AssertionError("LLM prompt contract statuses do not match the review summary")
    observed_status_counts = dict(sorted(
        pd.Series(list(observed_status_by_id.values())).value_counts().to_dict().items()
    ))
    if contract.get("status_counts") != observed_status_counts:
        raise AssertionError("LLM prompt contract status counts do not match its records")

    if not isinstance(reviews, list) or len(reviews) != EXPECTED_NATIVE_REVIEW_COUNT:
        raise AssertionError(
            f"Native reviews must contain exactly {EXPECTED_NATIVE_REVIEW_COUNT} records"
        )
    review_by_id: dict[str, dict] = {}
    for review in reviews:
        building_id = norm_id(review["BUILDINGSTRUCTUREID"])
        if building_id in review_by_id:
            raise AssertionError(f"Duplicate building ID in native reviews: {building_id}")
        review_by_id[building_id] = review
    if set(review_by_id) != set(contract_by_id):
        raise AssertionError("Native review IDs do not exactly match the LLM prompt contract")

    for building_id, review in review_by_id.items():
        expected = contract_by_id[building_id]
        if str(review.get("status")) != expected["status"]:
            raise AssertionError(f"Native review status differs from contract for {building_id}")
        if str(review.get("input_prompt_sha256")) != expected["new_prompt_sha256"]:
            raise AssertionError(f"Native review input prompt SHA-256 differs for {building_id}")
        if str(review.get("new_prompt_sha256")) != expected["new_prompt_sha256"]:
            raise AssertionError(f"Native review current prompt SHA-256 differs for {building_id}")
    return review_by_id


def canonical_sub(label: str) -> str:
    label = str(label).strip()
    if label in NON_ASSESSED_SUB_MAP:
        return NON_ASSESSED_SUB_MAP[label]
    match = re.fullmatch(r"\s*(.*?)\s*[（(]\s*(.*?)\s*[）)]\s*", label)
    if not match:
        raise ValueError(f"Unexpected native subclass label: {label!r}")
    chinese, english = match.group(1).strip(), match.group(2).strip()
    return f"{english}_{chinese}"


def split_native_subclasses(main_zh: str, value: str) -> list[str]:
    text = str(value).strip()
    if main_zh != "混合用途":
        return [text]
    return [part.strip() for part in re.split(r"(?<=[）)])-(?=\S)", text) if part.strip()]


def native_merge_values(review: dict) -> dict | None:
    confidence = float(review["confidence"])
    if confidence < 0.80:
        return None
    main_zh = str(review["main_class"]).strip()
    main = MAIN_MAP[main_zh]
    sub_labels = split_native_subclasses(main_zh, review["sub_class"])
    if main_zh != "混合用途":
        if len(sub_labels) != 1 or SUB_PARENT.get(sub_labels[0]) != main_zh:
            raise ValueError(f"Invalid native main/sub pair for {review['BUILDINGSTRUCTUREID']}")
        sub = canonical_sub(sub_labels[0])
        return {
            "Final_Main_Class": main,
            "Final_Sub_Class": sub,
            "Classification_Source": f"LLM_Verified (Conf: {confidence:.2f})",
            "Main_Class_Proportions": f"{main}:1.00",
            "Sub_Class_Proportions": f"{sub}:1.00",
            "Is_Mixed_Use": False,
        }

    if len(sub_labels) < 2 or any(label not in SUB_PARENT for label in sub_labels):
        raise ValueError(f"Invalid native mixed subclass for {review['BUILDINGSTRUCTUREID']}")
    weight = 1.0 / len(sub_labels)
    main_weights: dict[str, float] = defaultdict(float)
    sub_parts = []
    for label in sub_labels:
        canonical = canonical_sub(label)
        sub_parts.append(f"{canonical}:{weight:.2f}")
        main_weights[MAIN_MAP[SUB_PARENT[label]]] += weight
    main_parts = [f"{label}:{value:.2f}" for label, value in main_weights.items()]
    return {
        "Final_Main_Class": main,
        "Final_Sub_Class": "-".join(canonical_sub(label) for label in sub_labels),
        "Classification_Source": f"LLM_Verified (Conf: {confidence:.2f})",
        "Main_Class_Proportions": "-".join(main_parts),
        "Sub_Class_Proportions": "-".join(sub_parts),
        "Is_Mixed_Use": True,
    }


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    summary = json.loads(PREPARER_SUMMARY.read_text(encoding="utf-8-sig"))
    unchanged_ids = set(map(str, summary["ids"]["cache_prompt_unchanged"]))
    changed_ids = set(map(str, summary["ids"]["cache_prompt_changed"]))
    new_ids = set(map(str, summary["ids"]["new_candidates_without_cache"]))
    if changed_ids & new_ids:
        raise AssertionError("Changed-prompt and new-candidate review sets overlap")
    cache_ids = unchanged_ids | changed_ids
    review_ids = changed_ids | new_ids
    expected_status_by_id = {
        **{building_id: "cache_prompt_changed" for building_id in changed_ids},
        **{building_id: "new_candidate_without_cache" for building_id in new_ids},
    }

    reviews = json.loads(NATIVE.read_text(encoding="utf-8-sig"))
    prompt_contract = json.loads(PROMPT_CONTRACT.read_text(encoding="utf-8-sig"))
    review_by_id = validate_prompt_contract(
        prompt_contract,
        reviews,
        expected_status_by_id,
    )

    stage3 = pd.read_csv(NEW_STEP3, dtype={"BUILDINGSTRUCTUREID": str}, low_memory=False, encoding="utf-8-sig")
    old5 = pd.read_csv(OLD_STEP5, dtype={"BUILDINGSTRUCTUREID": str}, low_memory=False, encoding="utf-8-sig")
    old4 = pd.read_csv(OLD_STEP4, dtype={"BUILDINGSTRUCTUREID": str}, low_memory=False, encoding="utf-8-sig")
    for frame in (stage3, old5, old4):
        frame["BUILDINGSTRUCTUREID"] = frame["BUILDINGSTRUCTUREID"].map(norm_id)
    if len(stage3) != 341153 or not stage3["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Corrected stage 3 must contain 341,153 unique IDs")
    if set(stage3["BUILDINGSTRUCTUREID"]) != set(old5["BUILDINGSTRUCTUREID"]):
        raise AssertionError("Corrected stage 3 and legacy stage 5 IDs differ")

    result = stage3.copy().set_index("BUILDINGSTRUCTUREID", drop=False)
    old5i = old5.set_index("BUILDINGSTRUCTUREID", verify_integrity=True)
    legacy_llm_mask = old5i["Classification_Source"].astype(str).str.startswith("LLM_Verified")
    preserve_ids = set(old5i.index[legacy_llm_mask]) & cache_ids
    for field in CLASS_FIELDS:
        if field in old5i.columns:
            result.loc[list(preserve_ids), field] = old5i.loc[list(preserve_ids), field]
    result["Stage5_Repair_Action"] = "corrected_original_rule_result_retained"
    preserve_unchanged_ids = preserve_ids & unchanged_ids
    preserve_physical_delta_ids = preserve_ids & changed_ids
    changed_rule_ids = changed_ids - preserve_ids
    result.loc[list(preserve_unchanged_ids), "Stage5_Repair_Action"] = (
        "legacy_llm_checkpoint_preserved_prompt_unchanged"
    )
    result.loc[list(preserve_physical_delta_ids), "Stage5_Repair_Action"] = (
        "legacy_llm_checkpoint_preserved_physical_prompt_delta"
    )
    result.loc[list(changed_rule_ids), "Stage5_Repair_Action"] = (
        "corrected_original_rule_result_retained_physical_prompt_delta"
    )

    native_override_ids = []
    native_fallback_ids = []
    bottom_shop_protected_ids = []
    for building_id in sorted(new_ids, key=lambda value: int(value)):
        review = review_by_id[building_id]
        merged = native_merge_values(review)
        if merged is None:
            native_fallback_ids.append(building_id)
            result.at[building_id, "Stage5_Repair_Action"] = "native_llm_below_threshold_rule_retained"
            continue
        rule_main = str(result.at[building_id, "Final_Main_Class"]).strip()
        matched_pois = str(result.at[building_id, "Matched_POIs"]).lower()
        native_main = str(review["main_class"]).strip()
        if (
            rule_main in {"混合用途", "Mixed-use_混合用途"}
            and any(tag in matched_pois for tag in COMMERCIAL_POI_TAGS)
            and native_main in {"住宅类别", "商业类别", "工业类别", "非评估类别"}
        ):
            bottom_shop_protected_ids.append(building_id)
            source = str(result.at[building_id, "Classification_Source"])
            result.at[building_id, "Classification_Source"] = (
                f"{source} (LLM Override Rejected - Bottom Shop Protected)"
            )
            result.at[building_id, "Stage5_Repair_Action"] = "original_bottom_shop_protection_rule_retained"
            continue
        native_override_ids.append(building_id)
        for field, value in merged.items():
            result.at[building_id, field] = value
        result.at[building_id, "Stage5_Repair_Action"] = "native_llm_override_new_candidate_without_legacy_cache"

    if set(bottom_shop_protected_ids) != EXPECTED_BOTTOM_SHOP_PROTECTED_IDS:
        raise AssertionError(
            "Frozen bottom-shop protection regression: "
            f"expected {sorted(EXPECTED_BOTTOM_SHOP_PROTECTED_IDS)}, got {sorted(bottom_shop_protected_ids)}"
        )

    # Match the frozen stage-5 output contract.  Keeping Classification_Stage
    # would make frozen stage 8 filter the full table to only pending records.
    result = result.drop(
        columns=[
            "Pre_Class_Main",
            "Pre_Class_Sub",
            "Classification_Stage",
            "IS_FINAL_RECALC",
            "LLM_Main_Class",
            "LLM_Sub_Class",
            "LLM_Reasoning",
            "LLM_Confidence",
        ],
        errors="ignore",
    )
    result = result.loc[stage3["BUILDINGSTRUCTUREID"]].reset_index(drop=True)
    if len(result) != 341153 or not result["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Corrected stage 5 lost or duplicated records")
    result.to_csv(OUT_STEP5, index=False, encoding="utf-8-sig")

    legacy_cache = old4[old4["BUILDINGSTRUCTUREID"].isin(cache_ids)].copy()
    legacy_cache["LLM_Result_Origin"] = "legacy_cache_prompt_unchanged"
    legacy_cache.loc[
        legacy_cache["BUILDINGSTRUCTUREID"].isin(changed_ids), "LLM_Result_Origin"
    ] = "legacy_cache_preserved_physical_prompt_delta"
    native_rows = []
    for review in reviews:
        native_rows.append({
            "BUILDINGSTRUCTUREID": norm_id(review["BUILDINGSTRUCTUREID"]),
            "LLM_Main_Class": MAIN_MAP[str(review["main_class"]).strip()],
            "LLM_Sub_Class": "-".join(
                canonical_sub(x) for x in split_native_subclasses(str(review["main_class"]).strip(), review["sub_class"])
            ),
            "LLM_Reasoning": review["reasoning"],
            "LLM_Confidence": float(review["confidence"]),
            "LLM_Prompt_SHA256": review["input_prompt_sha256"],
            "LLM_Result_Origin": "native_original_prompt_review",
        })
    native_frame = pd.DataFrame(native_rows)
    physical_rechecks = native_frame[native_frame["BUILDINGSTRUCTUREID"].isin(changed_ids)].copy()
    physical_rechecks["Publishing_Decision"] = "diagnostic_only_not_applied_to_function_classification"
    if len(physical_rechecks) != len(changed_ids) or not physical_rechecks["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError(
            "Physical-only prompt recheck evidence must contain every changed-prompt ID exactly once"
        )
    physical_rechecks.to_csv(OUT_PHYSICAL_RECHECKS, index=False, encoding="utf-8-sig")

    native_new_rows = native_frame[native_frame["BUILDINGSTRUCTUREID"].isin(new_ids)].copy()
    combined4 = pd.concat([legacy_cache, native_new_rows], ignore_index=True, sort=False)
    expected_combined_rows = len(cache_ids) + len(new_ids)
    if len(combined4) != expected_combined_rows or not combined4["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError(
            "Combined stage-4 evidence must contain every legacy cache ID and current new candidate exactly once"
        )
    combined4.to_csv(OUT_STEP4, index=False, encoding="utf-8-sig")

    old5_aligned = old5i.reindex(result["BUILDINGSTRUCTUREID"])
    changes = pd.DataFrame({"BUILDINGSTRUCTUREID": result["BUILDINGSTRUCTUREID"]})
    for field in CLASS_FIELDS[:5]:
        changes[f"old_{field}"] = old5_aligned[field].to_numpy()
        changes[f"new_{field}"] = result[field].to_numpy()
    changes["Stage5_Repair_Action"] = result["Stage5_Repair_Action"]
    changed_mask = False
    for field in CLASS_FIELDS[:5]:
        changed_mask = changed_mask | changes[f"old_{field}"].fillna("<NA>").astype(str).ne(
            changes[f"new_{field}"].fillna("<NA>").astype(str)
        )
    changes = changes[changed_mask].copy()
    changes.to_csv(OUT_CHANGES, index=False, encoding="utf-8-sig")

    manifest = {
        "method": "legacy functional checkpoint preservation plus exact-original-prompt review for new candidates only",
        "rows": len(result),
        "legacy_llm_checkpoint_preserved": len(preserve_ids),
        "legacy_llm_checkpoint_preserved_prompt_unchanged": len(preserve_unchanged_ids),
        "legacy_llm_checkpoint_preserved_physical_prompt_delta": len(preserve_physical_delta_ids),
        "physical_prompt_delta_rule_rows_kept_from_corrected_stage3": len(changed_rule_ids),
        "native_reviews_total": len(review_ids),
        "native_physical_prompt_rechecks_diagnostic_only_not_applied": len(changed_ids),
        "native_new_candidates_without_legacy_cache": len(new_ids),
        "native_llm_overrides_confidence_ge_0_80": len(native_override_ids),
        "native_llm_below_threshold_rule_retained": len(native_fallback_ids),
        "original_bottom_shop_protection_rule_retained": len(bottom_shop_protected_ids),
        "original_bottom_shop_protected_ids": bottom_shop_protected_ids,
        "classification_or_proportion_rows_changed_vs_legacy_step5": len(changes),
        "input_sha256": {
            "corrected_stage3": sha256(NEW_STEP3),
            "legacy_stage4": sha256(OLD_STEP4),
            "legacy_stage5": sha256(OLD_STEP5),
            "native_reviews": sha256(NATIVE),
            "review_summary": sha256(PREPARER_SUMMARY),
            "prompt_contract": sha256(PROMPT_CONTRACT),
        },
        "output_sha256": {
            path.name: sha256(path)
            for path in [OUT_STEP4, OUT_STEP5, OUT_CHANGES, OUT_PHYSICAL_RECHECKS]
        },
        "no_new_rules_or_web_function_facts": True,
        "physical_attribute_changes_do_not_rewrite_legacy_llm_function_labels": True,
    }
    OUT_AUDIT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
