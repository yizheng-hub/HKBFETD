"""Reuse only exact-input LLM checkpoints, then run the frozen merge step."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


METHOD = Path(os.environ["HKBFETD_REPLAY_FROZEN_METHOD"]).resolve()
RUN = Path(os.environ["HKBFETD_REPLAY_OUTPUT_DIR"]).resolve()
OLD_STEP3 = Path(os.environ["HKBFETD_REPLAY_LEGACY_STAGE3"]).resolve()
OLD_STEP4 = Path(os.environ["HKBFETD_REPLAY_LEGACY_STAGE4"]).resolve()
NATIVE = Path(os.environ["HKBFETD_REPLAY_NATIVE_REVIEWS"]).resolve()
PROMPT_CONTRACT = Path(os.environ["HKBFETD_REPLAY_LLM_PROMPT_CONTRACT"]).resolve()
NEW_STEP3 = RUN / "step3_rule_engine_classification.csv"
OUT_STEP4 = RUN / "step4_exact_input_llm_evidence.csv"
OUT_STEP5 = RUN / "step5_merged_classification.csv"
OUT_INVALID = RUN / "stage4_invalidated_llm_cache.csv"
OUT_CHANGES = RUN / "stage5_classification_changes.csv"
OUT_AUDIT = RUN / "stage5_build_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def norm_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_prompt_functions(config, utils):
    """Compile the prompt functions verbatim from the frozen source snapshot."""
    source_path = METHOD / "src/4.llm_verification.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"), filename=str(source_path))
    wanted = {"build_taxonomy_str", "filter_data_for_llm", "generate_prompt"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in selected} != wanted:
        raise AssertionError("Frozen LLM prompt functions are incomplete")
    namespace = {
        "pd": pd,
        "np": np,
        "re": re,
        "LLM_TAXONOMY": config.LLM_TAXONOMY,
        "safe_str": utils.safe_str,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    namespace["TAXONOMY_PROMPT_STR"] = namespace["build_taxonomy_str"](config.LLM_TAXONOMY)
    return namespace["filter_data_for_llm"], namespace["generate_prompt"]


def canonical_proportion(value, canonicalizer):
    if pd.isna(value):
        return value
    parts = []
    for token in str(value).split("-"):
        if ":" not in token:
            parts.append(token)
            continue
        label, weight = token.rsplit(":", 1)
        parts.append(f"{canonicalizer(label)}:{weight}")
    return "-".join(parts)


def original_main(value):
    reverse = {
        "Residential_住宅类别": "住宅类别",
        "Commercial_商业类别": "商业类别",
        "Industrial_工业类别": "工业类别",
        "Non-assessed_非评估类别": "非评估类别",
        "Mixed-use_混合用途": "混合用途",
        "Unknown_未知类别": "未知类别",
    }
    return reverse.get(str(value).strip(), value)


def original_sub(value):
    text = str(value)
    replacements = {}
    for main, subclasses in config.LLM_TAXONOMY.items():
        if main == "__STRONG_KEYWORDS__":
            continue
        for subclass in subclasses:
            replacements[utils.canonicalize_sub_class_label(subclass)] = subclass
    for canonical, source in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(canonical, source)
    return text


if os.environ.get("PYTHONHASHSEED") != "0":
    raise RuntimeError("Run in a fresh process with PYTHONHASHSEED=0")

sys.path.insert(0, str(METHOD / "src"))
import config  # noqa: E402
import utils  # noqa: E402

merge = load_module("frozen_original_merge", METHOD / "src/5.evaluation_and_merge.py")
filter_candidates, generate_prompt = frozen_prompt_functions(config, utils)


def main() -> None:
    print("[1/7] Read current and archived rule outputs", flush=True)
    current = pd.read_csv(NEW_STEP3, dtype={"BUILDINGSTRUCTUREID": "string"}, low_memory=False)
    archived = pd.read_csv(OLD_STEP3, dtype={"BUILDINGSTRUCTUREID": "string"}, low_memory=False)
    old4 = pd.read_csv(OLD_STEP4, dtype={"BUILDINGSTRUCTUREID": "string"}, low_memory=False)
    for frame in (current, archived, old4):
        frame["BUILDINGSTRUCTUREID"] = frame["BUILDINGSTRUCTUREID"].map(norm_id)
    if len(current) != 341153 or not current["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Current stage 3 must contain 341,153 unique IDs")
    if len(archived) != 341153 or not archived["BUILDINGSTRUCTUREID"].is_unique:
        raise AssertionError("Archived stage 3 must contain 341,153 unique IDs")
    if old4["BUILDINGSTRUCTUREID"].duplicated().any():
        raise AssertionError("Archived LLM cache contains duplicate IDs")

    current_i = current.set_index("BUILDINGSTRUCTUREID", verify_integrity=True)
    archived_i = archived.set_index("BUILDINGSTRUCTUREID", verify_integrity=True)
    old4_i = old4.set_index("BUILDINGSTRUCTUREID", verify_integrity=True)

    print("[2/7] Recompute the frozen LLM candidate set", flush=True)
    candidate_frame = current.copy()
    # The frozen selector predates canonical bilingual labels and compares the
    # original Chinese main/subclass strings literally.  Restore that input
    # representation only for candidate selection; the frozen logic itself is
    # unchanged and published labels remain canonical.
    candidate_frame["Final_Main_Class"] = candidate_frame["Final_Main_Class"].map(
        original_main
    )
    candidate_frame["Final_Sub_Class"] = candidate_frame["Final_Sub_Class"].map(
        original_sub
    )
    candidate_frame["LLM_Confidence"] = np.nan
    candidates = filter_candidates(candidate_frame)
    candidate_ids = set(candidates["BUILDINGSTRUCTUREID"].map(norm_id))

    print("[3/7] Bind old and native results to exact current prompt hashes", flush=True)
    current_prompt_hash = {}
    for building_id in sorted(candidate_ids, key=int):
        current_prompt_hash[building_id] = prompt_sha(generate_prompt(current_i.loc[building_id]))

    valid = {}
    audit_rows = []
    for building_id, row in old4_i.iterrows():
        if building_id not in current_i.index:
            status = "missing_current_building"
            old_hash = None
            new_hash = None
        elif building_id not in candidate_ids:
            status = "no_longer_current_candidate"
            old_hash = prompt_sha(generate_prompt(archived_i.loc[building_id]))
            new_hash = None
        else:
            old_hash = prompt_sha(generate_prompt(archived_i.loc[building_id]))
            new_hash = current_prompt_hash[building_id]
            status = "legacy_cache_exact_prompt" if old_hash == new_hash else "legacy_cache_prompt_changed"
            if status == "legacy_cache_exact_prompt":
                valid[building_id] = {
                    "LLM_Main_Class": row["LLM_Main_Class"],
                    "LLM_Sub_Class": row["LLM_Sub_Class"],
                    "LLM_Reasoning": row["LLM_Reasoning"],
                    "LLM_Confidence": row["LLM_Confidence"],
                    "LLM_Cache_Status": status,
                    "LLM_Prompt_SHA256": new_hash,
                }
        audit_rows.append({
            "BUILDINGSTRUCTUREID": building_id,
            "Cache_Origin": "legacy_stage4",
            "Cache_Status": status,
            "Old_Prompt_SHA256": old_hash,
            "Current_Prompt_SHA256": new_hash,
        })

    contract = json.loads(PROMPT_CONTRACT.read_text(encoding="utf-8-sig"))
    contract_by_id = {norm_id(row["BUILDINGSTRUCTUREID"]): row for row in contract["records"]}
    reviews = json.loads(NATIVE.read_text(encoding="utf-8-sig"))
    native_exact = 0
    for review in reviews:
        building_id = norm_id(review["BUILDINGSTRUCTUREID"])
        new_hash = current_prompt_hash.get(building_id)
        contract_row = contract_by_id.get(building_id)
        exact = (
            building_id in candidate_ids
            and new_hash is not None
            and contract_row is not None
            and str(contract_row.get("new_prompt_sha256")) == new_hash
            and str(review.get("input_prompt_sha256")) == new_hash
            and str(review.get("new_prompt_sha256")) == new_hash
        )
        status = "native_cache_exact_prompt" if exact else "native_cache_not_current_exact_prompt"
        if exact and building_id not in valid:
            valid[building_id] = {
                "LLM_Main_Class": utils.canonicalize_main_class_label(review.get("main_class")),
                "LLM_Sub_Class": utils.canonicalize_sub_class_label(review.get("sub_class")),
                "LLM_Reasoning": review.get("reasoning"),
                "LLM_Confidence": review.get("confidence"),
                "LLM_Cache_Status": status,
                "LLM_Prompt_SHA256": new_hash,
            }
            native_exact += 1
        audit_rows.append({
            "BUILDINGSTRUCTUREID": building_id,
            "Cache_Origin": "native_review",
            "Cache_Status": status,
            "Old_Prompt_SHA256": review.get("old_prompt_sha256"),
            "Current_Prompt_SHA256": new_hash,
        })

    print("[4/7] Run frozen stage-5 merge for exact-input cache rows", flush=True)
    result = current_i.copy()
    result["LLM_Cache_Status"] = "not_current_candidate"
    result.loc[list(candidate_ids), "LLM_Cache_Status"] = "current_candidate_without_exact_cache"
    evidence_rows = []
    for building_id in sorted(valid, key=int):
        record = valid[building_id]
        temp = result.loc[building_id].copy()
        for field in ("LLM_Main_Class", "LLM_Sub_Class", "LLM_Reasoning", "LLM_Confidence"):
            temp[field] = record[field]
        temp["LLM_Main_Class"] = original_main(temp["LLM_Main_Class"])
        temp["LLM_Sub_Class"] = original_sub(temp["LLM_Sub_Class"])
        merged = merge.apply_merge_logic(temp)
        main, sub, source, main_prop, sub_prop, overridden = merged.tolist()
        result.at[building_id, "Final_Main_Class"] = utils.canonicalize_main_class_label(main)
        result.at[building_id, "Final_Sub_Class"] = utils.canonicalize_sub_class_label(sub)
        result.at[building_id, "Classification_Source"] = source
        result.at[building_id, "Main_Class_Proportions"] = canonical_proportion(
            main_prop, utils.canonicalize_main_class_label
        )
        result.at[building_id, "Sub_Class_Proportions"] = canonical_proportion(
            sub_prop, utils.canonicalize_sub_class_label
        )
        result.at[building_id, "LLM_Cache_Status"] = record["LLM_Cache_Status"]
        evidence_rows.append({
            "BUILDINGSTRUCTUREID": building_id,
            **record,
            "Is_LLM_Overridden": bool(overridden),
            "Merged_Main_Class": result.at[building_id, "Final_Main_Class"],
            "Merged_Sub_Class": result.at[building_id, "Final_Sub_Class"],
        })

    llm_source = result["Classification_Source"].astype(str).str.contains("LLM_Verified", regex=False)
    exact_status = result["LLM_Cache_Status"].isin(
        ["legacy_cache_exact_prompt", "native_cache_exact_prompt"]
    )
    if (llm_source & ~exact_status).any():
        raise AssertionError("An LLM classification lacks an exact current prompt contract")
    if result["Classification_Source"].astype(str).str.contains("GFA", regex=False).any():
        raise AssertionError("Stage 5 restored unusable GFA evidence")
    unsafe_bdbiar = result["BDBIAR_Match_Quality"].ne("inside_unique") | ~result[
        "BDBIAR_Source_Attributes_Usable"
    ].fillna(False).astype(bool)
    cites_bdbiar = result["Classification_Source"].astype(str).str.contains("BDBIAR", regex=False)
    if (unsafe_bdbiar & cites_bdbiar).any():
        raise AssertionError("Stage 5 restored unsafe BDBIAR evidence")

    print("[5/7] Write stage-4 evidence, invalidations and stage-5 output", flush=True)
    pd.DataFrame(evidence_rows).to_csv(OUT_STEP4, index=False, encoding="utf-8-sig")
    invalid = pd.DataFrame(audit_rows)
    invalid = invalid[~invalid["Cache_Status"].isin(
        ["legacy_cache_exact_prompt", "native_cache_exact_prompt"]
    )]
    invalid.to_csv(OUT_INVALID, index=False, encoding="utf-8-sig")

    drop_columns = [
        "Pre_Class_Main",
        "Pre_Class_Sub",
        "Classification_Stage",
        "IS_FINAL_RECALC",
        "LLM_Main_Class",
        "LLM_Sub_Class",
        "LLM_Reasoning",
        "LLM_Confidence",
    ]
    result = result.drop(columns=drop_columns, errors="ignore")
    result = result.loc[current["BUILDINGSTRUCTUREID"]]
    result.reset_index().to_csv(OUT_STEP5, index=False, encoding="utf-8-sig")

    compare = pd.DataFrame(index=result.index)
    changed = pd.Series(False, index=result.index)
    for field in ["Final_Main_Class", "Final_Sub_Class", "Classification_Source"]:
        compare["Stage3_" + field] = current_i.loc[result.index, field]
        compare["Stage5_" + field] = result[field]
        field_changed = compare["Stage3_" + field].fillna("").astype(str).ne(
            compare["Stage5_" + field].fillna("").astype(str)
        )
        compare[field + "_Changed"] = field_changed
        changed |= field_changed
    compare["LLM_Cache_Status"] = result["LLM_Cache_Status"]
    compare.loc[changed].reset_index().to_csv(OUT_CHANGES, index=False, encoding="utf-8-sig")

    print("[6/7] Record exact-cache coverage", flush=True)
    status_counts = result["LLM_Cache_Status"].value_counts().to_dict()
    manifest = {
        "rows": len(result),
        "current_candidates": len(candidate_ids),
        "legacy_cache_rows": len(old4_i),
        "legacy_exact_prompt_cache_rows": int(status_counts.get("legacy_cache_exact_prompt", 0)),
        "native_exact_prompt_cache_rows": int(status_counts.get("native_cache_exact_prompt", 0)),
        "native_exact_added_beyond_legacy": native_exact,
        "current_candidates_without_exact_cache": int(
            status_counts.get("current_candidate_without_exact_cache", 0)
        ),
        "published_llm_source_rows": int(llm_source.sum()),
        "llm_sources_without_exact_prompt": int((llm_source & ~exact_status).sum()),
        "gfa_sources": 0,
        "unsafe_bdbiar_sources": 0,
        "stage5_rows_changed_from_stage3": int(changed.sum()),
        "frozen_prompt_source_sha256": sha256(METHOD / "src/4.llm_verification.py"),
        "frozen_merge_source_sha256": sha256(METHOD / "src/5.evaluation_and_merge.py"),
        "input_sha256": {
            "current_stage3": sha256(NEW_STEP3),
            "archived_stage3_comparison_only": sha256(OLD_STEP3),
            "archived_stage4_cache": sha256(OLD_STEP4),
            "native_reviews": sha256(NATIVE),
            "prompt_contract": sha256(PROMPT_CONTRACT),
        },
        "output_sha256": {OUT_STEP5.name: sha256(OUT_STEP5)},
    }
    OUT_AUDIT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[7/7] " + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
