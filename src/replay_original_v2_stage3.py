"""Run the frozen rule engine for every building on corrected V2 inputs."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


METHOD = Path(os.environ["HKBFETD_REPLAY_FROZEN_METHOD"]).resolve()
OUT = Path(os.environ["HKBFETD_REPLAY_OUTPUT_DIR"]).resolve()
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prefixed(names, prefix):
    return next(name for name in names if name == prefix or name.startswith(prefix + "_"))


def canonical_main_prop(value):
    text = "" if pd.isna(value) else str(value)
    for source, target in [
        ("住宅类别", "Residential_住宅类别"),
        ("商业类别", "Commercial_商业类别"),
        ("工业类别", "Industrial_工业类别"),
        ("混合用途", "Mixed-use_混合用途"),
        ("非评估类别", "Non-assessed_非评估类别"),
        ("未知类别", "Unknown_未知类别"),
    ]:
        text = text.replace(source + ":", target + ":")
    return text


def canonical_sub_prop(value):
    if pd.isna(value):
        return value
    import re

    return re.sub(
        r"([\u3400-\u9fff/、 ]+)[（(]([^()（）]+)[)）](?=:)",
        lambda match: f"{match.group(2).strip()}_{match.group(1).strip()}",
        str(value),
    )


if os.environ.get("PYTHONHASHSEED") != "0":
    raise RuntimeError("Run in a fresh process with PYTHONHASHSEED=0")

sys.path.insert(0, str(METHOD / "src"))
import utils  # noqa: E402

rule = load_module("frozen_original_rule_engine", METHOD / "src/3.rule_engine.py")

old_path = Path(os.environ["HKBFETD_REPLAY_LEGACY_STAGE3"]).resolve()
step2_path = Path(os.environ["HKBFETD_REPLAY_LEGACY_STAGE2"]).resolve()
physical_path = Path(os.environ["HKBFETD_REPLAY_VERIFIED_PHYSICAL"]).resolve()
bdbiar_path = Path(os.environ["HKBFETD_REPLAY_BDBIAR_SCOPE"]).resolve()

classification_columns = [
    "Final_Main_Class",
    "Final_Sub_Class",
    "Classification_Source",
    "Is_Mixed_Use",
    "Confidence_Score",
    "Evidence_Details",
    "Is_Conflicted",
    "Main_Class_Proportions",
    "Sub_Class_Proportions",
    "Matched_POIs",
    "Classification_Stage",
    "Pre_Class_Main",
    "Pre_Class_Sub",
]

print("[1/7] Read the legacy stage-3 carrier", flush=True)
old = pd.read_csv(
    old_path,
    low_memory=False,
    dtype={"BUILDINGSTRUCTUREID": "string", "BUILDINGCSUID": "string"},
)
old["BUILDINGSTRUCTUREID"] = ids(old["BUILDINGSTRUCTUREID"])
old = old.set_index("BUILDINGSTRUCTUREID", verify_integrity=True)
if len(old) != 341153:
    raise AssertionError("Stage 3 carrier must contain 341,153 unique buildings")
legacy_rule = old[
    classification_columns
    + [
        "NUMABOVEGROUNDSTOREYS",
        "TOPHEIGHT",
        "BASEHEIGHT",
        "SHAPE_Area",
        "BDBIAR_OBJECTID",
        "BDBIAR_CLASS",
        "BDBIAR_ADDRESS_C",
        "GFA_DOMESTIC_SUM",
        "GFA_NONDOMESTIC_SUM",
    ]
].copy()

print("[2/7] Read corrected physical and BDBIAR projections", flush=True)
schema = pq.read_schema(physical_path)
id_col = prefixed(schema.names, "Building_ID")
height_col = prefixed(schema.names, "Height_m")
floor_col = prefixed(schema.names, "Floors_Count")
area_col = prefixed(schema.names, "Footprint_Area_sqm")
physical_columns = [
    id_col,
    height_col,
    floor_col,
    area_col,
    "BUILDINGCSUID",
    "Permit_GFA_Usable",
    "GFA_DOMESTIC_SUM",
    "GFA_NONDOMESTIC_SUM",
]
physical = pq.read_table(physical_path, columns=physical_columns).to_pandas()
physical["BUILDINGSTRUCTUREID"] = ids(physical[id_col])
physical = physical.set_index("BUILDINGSTRUCTUREID", verify_integrity=True).reindex(old.index)
if physical.index.hasnans or not physical.index.equals(old.index):
    raise AssertionError("Verified physical input does not align with the stage-3 carrier")
permit_usable = physical["Permit_GFA_Usable"].fillna(False).astype(bool)
if permit_usable.any():
    raise AssertionError("This release expects no permit GFA with verified component scope")

bdbiar = pd.read_csv(bdbiar_path, dtype={"BUILDINGSTRUCTUREID": "string"}, low_memory=False)
bdbiar["BUILDINGSTRUCTUREID"] = ids(bdbiar["BUILDINGSTRUCTUREID"])
bdbiar = bdbiar.set_index("BUILDINGSTRUCTUREID", verify_integrity=True).reindex(old.index)
if bdbiar.index.hasnans or not bdbiar.index.equals(old.index):
    raise AssertionError("BDBIAR scope input does not align with the stage-3 carrier")
non_unique_bdbiar = bdbiar["BDBIAR_Match_Quality"].ne("inside_unique")
unsafe = non_unique_bdbiar | ~bdbiar[
    "BDBIAR_Source_Attributes_Usable"
].fillna(False).astype(bool)
adopted_bdbiar_attributes = [
    "BDBIAR_CLASS",
    "BDBIAR_AGE",
    "BDBIAR_ADDRESS_C",
    "BDBIAR_NSEARCH3_C",
    "BDBIAR_NSEARCH4_C",
]
adopted_bdbiar = ["BDBIAR_OBJECTID"] + adopted_bdbiar_attributes
if bdbiar.loc[unsafe, adopted_bdbiar_attributes].notna().any().any():
    raise AssertionError("Unsafe BDBIAR rows still contain adopted attributes")
if bdbiar.loc[unsafe, "BDBIAR_OBJECTID"].notna().any():
    raise AssertionError("An unsafe BDBIAR point is still presented as the building record")

print("[3/7] Read only the fields used by the frozen rule engine", flush=True)
rule_columns = [
    "BUILDINGSTRUCTUREID",
    "BUILDINGCSUID",
    "BUILDINGSTRUCTURETYPE",
    "CATEGORY",
    "OFFICIALBUILDINGNAMEEN",
    "OFFICIALBUILDINGNAMETC",
    "NUMABOVEGROUNDSTOREYS",
    "TOPHEIGHT",
    "BASEHEIGHT",
    "SHAPE_Area",
    "BUILDINGNAMEEN",
    "BUILDINGNAMETC",
    "GFA_DOMESTIC_SUM",
    "GFA_NONDOMESTIC_SUM",
    "ALL_INFO_DESC",
    "BDBIAR_OBJECTID",
    "BDBIAR_ADDRESS_C",
    "BDBIAR_CLASS",
    "osmid",
    "name",
    "building",
    "amenity",
    "shop",
    "osm_area",
    "overlap_ratio",
    "osm_weight_score",
]
step2 = gpd.read_file(step2_path, columns=rule_columns)
step2["BUILDINGSTRUCTUREID"] = ids(step2["BUILDINGSTRUCTUREID"])
if set(step2["BUILDINGSTRUCTUREID"].dropna()) != set(old.index):
    raise AssertionError("Stage 2 and stage 3 building IDs differ")

height = pd.to_numeric(physical[height_col], errors="coerce")
for column, values in {
    "NUMABOVEGROUNDSTOREYS": pd.to_numeric(physical[floor_col], errors="coerce"),
    "TOPHEIGHT": height,
    "BASEHEIGHT": pd.Series(np.where(height.notna(), 0.0, np.nan), index=physical.index),
    "SHAPE_Area": pd.to_numeric(physical[area_col], errors="coerce"),
    "GFA_DOMESTIC_SUM": pd.Series(np.nan, index=physical.index),
    "GFA_NONDOMESTIC_SUM": pd.Series(np.nan, index=physical.index),
}.items():
    step2[column] = step2["BUILDINGSTRUCTUREID"].map(values)

for column in adopted_bdbiar + [
    "BDBIAR_OBJECTID_CANDIDATE",
    "BDBIAR_CLASS_CANDIDATE",
    "BDBIAR_AGE_CANDIDATE",
    "BDBIAR_ADDRESS_C_CANDIDATE",
    "BDBIAR_NSEARCH3_C_CANDIDATE",
    "BDBIAR_NSEARCH4_C_CANDIDATE",
    "BDBIAR_Source_Block_ID",
    "distance_to_bdbiar",
    "BDBIAR_Match_Quality",
    "BDBIAR_Containing_Footprint_Count",
    "BDBIAR_Nearest_Tie_Count",
    "BDBIAR_Source_Attributes_Usable",
    "BDBIAR_Source_Integrity_Status",
]:
    if column in bdbiar:
        step2[column] = step2["BUILDINGSTRUCTUREID"].map(bdbiar[column])

non_unique_step2 = step2["BDBIAR_Match_Quality"].ne("inside_unique")
unsafe_step2 = non_unique_step2 | ~step2[
    "BDBIAR_Source_Attributes_Usable"
].fillna(False).astype(bool)
if step2.loc[unsafe_step2, adopted_bdbiar_attributes].notna().any().any():
    raise AssertionError("Unsafe BDBIAR values leaked into frozen-rule inputs")
if step2.loc[unsafe_step2, "BDBIAR_OBJECTID"].notna().any():
    raise AssertionError("Unsafe BDBIAR identity leaked into frozen-rule inputs")
if step2[["GFA_DOMESTIC_SUM", "GFA_NONDOMESTIC_SUM"]].notna().any().any():
    raise AssertionError("Unusable permit GFA leaked into frozen-rule inputs")

print("[4/7] Run the frozen rule engine for all 341,153 buildings", flush=True)
keyword_tool = utils.init_keyword_tool(str(METHOD / "ctl/keywords.json"))
official = step2.drop_duplicates("BUILDINGSTRUCTUREID")[["BUILDINGSTRUCTUREID", "geometry"]].copy()
fresh = rule.run_rule_engine(step2, official, keyword_tool)
fresh["BUILDINGSTRUCTUREID"] = ids(fresh["BUILDINGSTRUCTUREID"])
fresh = fresh.set_index("BUILDINGSTRUCTUREID", verify_integrity=True).reindex(old.index)
if len(fresh) != 341153 or fresh.index.hasnans:
    raise AssertionError("Frozen rule engine did not return every building exactly once")

fresh["Final_Main_Class"] = fresh["Final_Main_Class"].map(utils.canonicalize_main_class_label)
fresh["Final_Sub_Class"] = fresh["Final_Sub_Class"].map(utils.canonicalize_sub_class_label)
fresh["Main_Class_Proportions"] = fresh["Main_Class_Proportions"].map(canonical_main_prop)
fresh["Sub_Class_Proportions"] = fresh["Sub_Class_Proportions"].map(canonical_sub_prop)

print("[5/7] Replace every rule result and restore exact unaffected rule checkpoints", flush=True)
for column in classification_columns:
    if column in fresh:
        old[column] = fresh[column]

# Physical and BDBIAR/GFA fields are the only rule inputs changed in this
# repair.  Reuse a checkpoint only when the declared evidence excludes repaired
# sources and every changed input is semantically identical.  The frozen rule
# engine can use height/floors for proportions and BDBIAR address text for
# keyword matching even when those inputs are absent from its final source
# label, so the label alone is not an exact-input contract.
legacy_source = legacy_rule["Classification_Source"].fillna("").astype(str)
unchanged_rule_family = (
    legacy_source.str.contains("Keyword|Name_Keyword|OSM|LandsD_Category", regex=True)
    & ~legacy_source.str.contains(
        "BDBIAR|GFA|LLM|Inherited|Podium|Pre-Screening|Fusion|VLM",
        case=False,
        regex=True,
    )
)


def numeric_equal(left, right, *, atol=1e-8, rtol=1e-10):
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    scale = pd.concat([left.abs(), right.abs()], axis=1).max(axis=1)
    return (left.isna() & right.isna()) | (
        left.notna()
        & right.notna()
        & (left - right).abs().le(atol + rtol * scale)
    )


def text_equal(left, right):
    return left.fillna("").astype(str).str.strip().eq(
        right.fillna("").astype(str).str.strip()
    )


legacy_top = pd.to_numeric(legacy_rule["TOPHEIGHT"], errors="coerce")
legacy_base = pd.to_numeric(legacy_rule["BASEHEIGHT"], errors="coerce").fillna(0.0)
legacy_delta = legacy_top - legacy_base
legacy_effective_height = legacy_delta.where(legacy_delta.gt(0), legacy_top)
legacy_domestic_gfa = pd.to_numeric(
    legacy_rule["GFA_DOMESTIC_SUM"], errors="coerce"
).fillna(0.0)
legacy_nondomestic_gfa = pd.to_numeric(
    legacy_rule["GFA_NONDOMESTIC_SUM"], errors="coerce"
).fillna(0.0)

rule_input_semantic_match = (
    numeric_equal(
        legacy_rule["NUMABOVEGROUNDSTOREYS"], physical[floor_col]
    )
    & numeric_equal(legacy_effective_height, physical[height_col])
    & numeric_equal(legacy_rule["SHAPE_Area"], physical[area_col])
    & legacy_domestic_gfa.eq(0.0)
    & legacy_nondomestic_gfa.eq(0.0)
    & text_equal(legacy_rule["BDBIAR_OBJECTID"], bdbiar["BDBIAR_OBJECTID"])
    & text_equal(legacy_rule["BDBIAR_CLASS"], bdbiar["BDBIAR_CLASS"])
    & text_equal(
        legacy_rule["BDBIAR_ADDRESS_C"], bdbiar["BDBIAR_ADDRESS_C"]
    )
)
exact_unaffected_rule = unchanged_rule_family & rule_input_semantic_match
for column in classification_columns:
    if column in old and column in legacy_rule:
        old.loc[exact_unaffected_rule, column] = legacy_rule.loc[
            exact_unaffected_rule, column
        ]
old["Legacy_Rule_Checkpoint_Status"] = "rerun_after_foundation_scope_repair"
old.loc[
    exact_unaffected_rule, "Legacy_Rule_Checkpoint_Status"
] = "reused_exact_unaffected_rule_inputs"

old["NUMABOVEGROUNDSTOREYS"] = pd.to_numeric(physical[floor_col], errors="coerce")
old["GFA_DOMESTIC_SUM"] = np.nan
old["GFA_NONDOMESTIC_SUM"] = np.nan
old["Corrected_Structure_Height_m"] = height
old["TOPHEIGHT"] = height
old["BASEHEIGHT"] = np.where(height.notna(), 0.0, np.nan)
old["SHAPE_Area"] = pd.to_numeric(physical[area_col], errors="coerce")
old["Physical_Attribute_Overlay"] = "verified_v2_physical_repair"
excluded_scope_columns = {
    "Classification_Source",
    "Final_Main_Class",
    "Final_Sub_Class",
    "Published_BDBIAR_Unsafe",
    "Classification_Uses_BDBIAR",
    "Unsafe_BDBIAR_Classification",
    "Published_ID_Matches_Recomputed",
}
for column in bdbiar.columns:
    if column.startswith("Published_") or column in excluded_scope_columns:
        continue
    old[column] = bdbiar[column]
old["Classification_Repair_Reason"] = "frozen_rule_full_rerun_after_foundation_scope_repair"
old.loc[
    exact_unaffected_rule, "Classification_Repair_Reason"
] = "fixed_rule_checkpoint_reused_exact_unaffected_inputs"

non_unique_final = old["BDBIAR_Match_Quality"].ne("inside_unique")
unsafe_final = non_unique_final | ~old[
    "BDBIAR_Source_Attributes_Usable"
].fillna(False).astype(bool)
if old.loc[unsafe_final, adopted_bdbiar_attributes].notna().any().any():
    raise AssertionError("Unsafe BDBIAR values survived the full rule rerun")
if old.loc[unsafe_final, "BDBIAR_OBJECTID"].notna().any():
    raise AssertionError("Unsafe BDBIAR identity survived the full rule rerun")
uses_bdbiar = old["Classification_Source"].astype(str).str.contains("BDBIAR", regex=False)
if (uses_bdbiar & unsafe_final).any():
    raise AssertionError("Frozen rule output cites unsafe BDBIAR evidence")
uses_gfa = old["Classification_Source"].astype(str).str.contains("GFA", regex=False)
if uses_gfa.any():
    raise AssertionError("Frozen rule output cites unusable permit GFA")

print("[6/7] Write full-rerun output and row-level comparison", flush=True)
comparison = pd.DataFrame(index=old.index)
legacy = pd.read_csv(
    old_path,
    usecols=["BUILDINGSTRUCTUREID", "Final_Main_Class", "Final_Sub_Class", "Classification_Source"],
    dtype={"BUILDINGSTRUCTUREID": "string"},
    low_memory=False,
).set_index("BUILDINGSTRUCTUREID")
legacy.index = ids(legacy.index.to_series()).array
legacy = legacy.reindex(old.index)
for column in ["Final_Main_Class", "Final_Sub_Class", "Classification_Source"]:
    comparison["Legacy_" + column] = legacy[column]
    comparison["Corrected_" + column] = old[column]
    comparison[column + "_Changed"] = legacy[column].fillna("").astype(str).ne(
        old[column].fillna("").astype(str)
    )
comparison.index.name = "BUILDINGSTRUCTUREID"
comparison.reset_index().to_csv(
    OUT / "stage3_foundation_scope_comparison.csv", index=False, encoding="utf-8-sig"
)

corrected_path = OUT / "step3_rule_engine_classification.csv"
old.reset_index().to_csv(corrected_path, index=False, encoding="utf-8-sig")

summary = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "STAGE3_FULL_FROZEN_RULE_RERUN",
    "records": len(old),
    "full_rule_rerun_records": len(old),
    "legacy_checkpoint_classifications_preserved": int(exact_unaffected_rule.sum()),
    "legacy_checkpoint_preservation_contract": (
        "same archived text/OSM/LandsD carrier; declared source has no BDBIAR, "
        "GFA, LLM, VLM, inheritance, podium, prescreen or fusion dependency; "
        "floors, effective height, footprint area, GFA and BDBIAR rule inputs "
        "are semantically identical"
    ),
    "unsafe_bdbiar_rows": int(unsafe_final.sum()),
    "safe_bdbiar_rows": int((~unsafe_final).sum()),
    "unsafe_bdbiar_classification_sources": int((uses_bdbiar & unsafe_final).sum()),
    "unusable_gfa_classification_sources": int(uses_gfa.sum()),
    "main_changed": int(comparison["Final_Main_Class_Changed"].sum()),
    "sub_changed": int(comparison["Final_Sub_Class_Changed"].sum()),
    "source_changed": int(comparison["Classification_Source_Changed"].sum()),
    "input_sha256": {
        "legacy_stage2": sha(step2_path),
        "legacy_stage3_comparison_only": sha(old_path),
        "verified_physical": sha(physical_path),
        "bdbiar_scope": sha(bdbiar_path),
        "frozen_rule_source": sha(METHOD / "src/3.rule_engine.py"),
        "frozen_keywords": sha(METHOD / "ctl/keywords.json"),
    },
    "output_sha256": {corrected_path.name: sha(corrected_path)},
    "classification_logic_changed": False,
    "keyword_file_changed": False,
}
(OUT / "stage3_build_manifest.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("[7/7] " + json.dumps(summary, ensure_ascii=False), flush=True)
