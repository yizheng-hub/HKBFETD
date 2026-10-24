"""Step 4: resume text verification only for matching input/prompt/model hashes.

Default mode writes a review queue without API calls. Use --refresh when an
API key is configured to request unresolved cases, checkpointing each success.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ai_cache import fingerprint, read_cache, save_cache
from classification import UNKNOWN, main_label, sub_label, taxonomy, MIXED
from config import (RULE_ENGINE_OUTPUT_PATH, LLM_VERIFICATION_OUTPUT, LEGACY_OUTPUT_DIR,
                    API_KEY, BASE_URL, CLOUD_MODEL_NAME, LLM_TAXONOMY,
                    NATIVE_FUNCTION_CHECKPOINT, NATIVE_FUNCTION_POLICY_CLARIFICATION, KEYWORDS_FILE)
from ai_cache import file_hash
from native_function_checkpoint import load_checkpoint
from data_quality import ID, ids, require_unique

PROMPT_VERSION = "v2-component-function-evidence-20261023"
API_BACKEND = "api"
RESULT_FIELDS = [ID, "Input_SHA256", "LLM_Main_Class", "LLM_Sub_Class", "LLM_Confidence", "LLM_Reasoning", "Checkpoint_Backend"]


def generate_prompt(row):
    fields = ["BUILDINGNAMEEN", "BUILDINGNAMETC", "BDBIAR_CLASS", "BDBIAR_ADDRESS_C", "BDBIAR_Match_Quality", "CATEGORY",
              "BUILDINGSTRUCTURETYPE", "Primary_Building_Main", "Primary_Building_Sub", "Matched_POIs",
              "Estimated_Floors", "Floors_Quality", "Estimated_Height", "Height_Quality", "Height_Reference"]
    evidence = {field: None if pd.isna(row.get(field)) else row.get(field) for field in fields}
    if evidence["CATEGORY"] is not None:
        evidence["CATEGORY"] = str(evidence["CATEGORY"])
    if evidence['BDBIAR_Match_Quality'] != 'inside_unique':
        evidence['BDBIAR_CLASS'] = evidence['BDBIAR_ADDRESS_C'] = None
    return ("Classify the building's function using the supplied evidence and taxonomy. "
            "A POI is evidence of a tenant, not its floor area or the building's dominant function. "
            "Do not force mixed use from tower height, storey count or the mere presence of a shop. "
            "Height is structural component top minus base, which may start on a podium roof. "
            "Legal-private/government ownership categories and building=yes do not establish function. "
            "A residential/composite source label does not distinguish those two possibilities. "
            "Retain unknown when evidence is insufficient. Never estimate use-area fractions. "
            "Return JSON main_class, sub_class, confidence (0..1), reasoning.\n"
            + json.dumps({"taxonomy": LLM_TAXONOMY, "evidence": evidence}, ensure_ascii=False, default=str, sort_keys=True))


def input_hash(row):
    return fingerprint({"prompt_version": PROMPT_VERSION, "model": CLOUD_MODEL_NAME, "prompt": generate_prompt(row)})


def call_llm_api(prompt):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    result = client.chat.completions.create(model=CLOUD_MODEL_NAME, temperature=0,
        messages=[{"role": "user", "content": prompt}], timeout=60)
    text = result.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)



def native_results_for_base(base):
    checkpoint = Path(NATIVE_FUNCTION_CHECKPOINT)
    if not checkpoint.is_file():
        return {}
    clarification = Path(NATIVE_FUNCTION_POLICY_CLARIFICATION)
    if not clarification.is_file():
        raise ValueError("native checkpoint exists but its frozen root policy clarification is missing")
    import geopandas as gpd
    artifact = json.loads(checkpoint.read_text(encoding='utf-8'))
    native_ids = set(str(bid) for bid in artifact.get('row_guards', {}))
    for batch in artifact.get('batches', []):
        results = json.loads(batch['results_utf8'])
        result_rows = results.get('results', results) if isinstance(results, dict) else results
        native_ids.update(str(row['building_id']) for row in result_rows)
    if not native_ids:
        return {}
    subset = base[base[ID].astype(str).isin(native_ids)].copy()
    subset[ID] = ids(subset[ID])
    geometry_path = Path(RULE_ENGINE_OUTPUT_PATH).with_suffix('.geojson')
    quoted_ids = ",".join("'" + bid.replace("'", "''") + "'" for bid in sorted(native_ids))
    geo = gpd.read_file(geometry_path, engine='pyogrio', columns=[ID], where=f'"{ID}" IN ({quoted_ids})')
    geo[ID] = ids(geo[ID])
    geo = geo[geo[ID].astype(str).isin(native_ids)].set_index(ID)
    if geo.index.astype(str).duplicated().any():
        raise ValueError("Step 3 geometry IDs must be unique for native replay")
    current_geometry = {}
    base_by_id = subset.set_index(ID, drop=False)
    for bid, geo_row in geo.iterrows():
        geom = geo_row.geometry
        source_row = base_by_id.loc[bid]
        scope = {key: (None if pd.isna(source_row.get(key)) else source_row.get(key).item() if hasattr(source_row.get(key), 'item') else source_row.get(key))
                 for key in ("CSDI_Native_Building_ID", "CSDI_Component_Use_Type")}
        current_geometry[str(bid)] = {
            "wkb_sha256": __import__('hashlib').sha256(geom.wkb).hexdigest(),
            "crs": geo.crs.to_string(),
            "component_scope": {"BUILDINGSTRUCTUREID": str(bid), **scope}}
    prompt_by_id = {str(row[ID]): generate_prompt(row) for _, row in subset.iterrows()}
    current_rows = {}
    for _, row in subset.iterrows():
        bid = str(row[ID])
        current_rows[bid] = {str(k): (None if pd.isna(v) else v.item() if hasattr(v, 'item') else v)
                             for k, v in row.items() if k != 'geometry'}
    return load_checkpoint(
        checkpoint, current_rows, current_geometry,
        file_hash(Path(RULE_ENGINE_OUTPUT_PATH)), file_hash(geometry_path),
        file_hash(Path(KEYWORDS_FILE)), file_hash(clarification),
        main_label, sub_label, taxonomy(),
        prompt_evidence=lambda row: json.loads(prompt_by_id[str(row[ID])].split("\n", 1)[1])["evidence"],
        prompt_builder=lambda row: prompt_by_id[str(row[ID])], prompt_version=PROMPT_VERSION)


def main(refresh=False, limit=100):
    base = pd.read_csv(RULE_ENGINE_OUTPUT_PATH, low_memory=False)
    base[ID] = ids(base[ID])
    require_unique(base)
    needs = base.Final_Main_Class.eq(UNKNOWN) | base.Final_Sub_Class.eq(UNKNOWN)
    names = base.BUILDINGNAMEEN.fillna('') + base.BUILDINGNAMETC.fillna('')
    meaningful = names.str.replace(r'[0-9\W_]+','',regex=True).str.len().gt(2)
    poi = base.Matched_POIs.fillna('').astype(str)
    meaningful |= poi.str.len().gt(10) & ~poi.str.lower().isin(['nan','none','no record'])
    candidates = base[needs & meaningful].copy()
    candidates["Input_SHA256"] = [input_hash(row) for _,row in candidates.iterrows()]
    cache = read_cache(LLM_VERIFICATION_OUTPUT, RESULT_FIELDS)
    for col in RESULT_FIELDS:
        if col not in cache:
            cache[col] = pd.NA
    cache['Checkpoint_Backend'] = cache['Checkpoint_Backend'].fillna(API_BACKEND).astype(str)
    # Native responses are loaded only from the provenance-checked frozen source, never from this API cache.
    cache = cache[cache.Checkpoint_Backend.eq(API_BACKEND)].copy()
    completed = set(zip(cache[ID].astype(str), cache.Input_SHA256.astype(str)))
    missing = [((str(row[ID]), row.Input_SHA256) not in completed) for _, row in candidates.iterrows()]
    pending = candidates.loc[missing]
    if refresh and not API_KEY:
        raise ValueError("HKBFETD_API_KEY is required for --refresh; checkpoints were preserved")
    if refresh:
        for _, row in (pending.head(limit) if limit else pending).iterrows():
            try:
                result = call_llm_api(generate_prompt(row))
                main, sub = main_label(result.get("main_class")), sub_label(result.get("sub_class"))
                confidence = float(result.get("confidence", np.nan))
                if not np.isfinite(confidence) or not 0 <= confidence <= 1 or main == UNKNOWN:
                    continue
                if main != MIXED and taxonomy().get(sub) != main:
                    continue
                entry = {ID: row[ID], "Input_SHA256": row.Input_SHA256, "LLM_Main_Class": main,
                         "LLM_Sub_Class": sub, "LLM_Confidence": confidence,
                         "LLM_Reasoning": result.get("reasoning", ""), "Checkpoint_Backend": API_BACKEND}
                cache = pd.concat([cache, pd.DataFrame([entry])], ignore_index=True)
                save_cache(cache, LLM_VERIFICATION_OUTPUT)
            except Exception as error:
                print(f"[WARN] LLM request not checkpointed: {type(error).__name__}")
    cache = cache.drop_duplicates([ID, "Input_SHA256"], keep="last")
    save_cache(cache, LLM_VERIFICATION_OUTPUT)
    current = candidates[[ID, "Input_SHA256"]].merge(cache, on=[ID, "Input_SHA256"], how="left", validate="one_to_one")
    current["LLM_Status"] = np.where(current.LLM_Confidence.notna(), "matching_checkpoint", "pending_no_matching_checkpoint")
    native = native_results_for_base(base)
    if native:
        native_status = {bid: item for bid, item in native.items() if bid in set(candidates[ID].astype(str))}
        for idx, row in current.iterrows():
            item = native_status.get(str(row[ID]))
            if item:
                current.at[idx, "Native_Status"] = item["status"]
                current.at[idx, "Native_Context_SHA256"] = item.get("context_sha256")
                current.at[idx, "Native_Backend"] = item.get("backend")
                current.at[idx, "Native_Model"] = item.get("model")
                current.at[idx, "Native_Batch_ID"] = item.get("batch_id")
                current.at[idx, "Native_Policy_SHA256"] = item.get("policy_sha256")
                current.at[idx, "Native_Clarification_SHA256"] = item.get("clarification_sha256")
                current.at[idx, "Native_Root_Acceptance_SHA256"] = item.get("root_acceptance_sha256")
                current.at[idx, "Native_Source_CSV_SHA256"] = item.get("source_csv_sha256")
                current.at[idx, "Native_Stage3_GeoJSON_SHA256"] = item.get("stage3_geojson_sha256")
                current.at[idx, "Native_Response_JSON"] = json.dumps(item.get("response", {}), ensure_ascii=False)
                current.at[idx, "LLM_Status"] = item["status"]
    legacy_path = Path(LEGACY_OUTPUT_DIR) / "step4_llm_verified_classification.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path, low_memory=False)
        legacy[ID] = ids(legacy[ID])
        legacy = legacy.drop_duplicates(ID).rename(columns={c: "Legacy_" + c for c in legacy if c != ID})
        current = current.merge(legacy, on=ID, how="left", validate="one_to_one")
    current.to_csv(Path(LLM_VERIFICATION_OUTPUT).with_name("step4_current_status.csv"), index=False, encoding="utf-8-sig")
    completed_native = {"reviewed_unknown", "reviewed_partial", "reviewed_low_confidence", "reviewed_not_adopted_root_rejected"}
    pending_mask = ~current.LLM_Status.isin(["matching_checkpoint", "matching_native_checkpoint", *completed_native])
    if "Native_Status" in current:
        native_status_present = current.Native_Status.notna()
        pending_mask |= native_status_present & ~current.Native_Status.isin(["matching_native_checkpoint", *completed_native])
    current[pending_mask].to_csv(
        Path(LLM_VERIFICATION_OUTPUT).with_name("step4_pending_review.csv"), index=False, encoding="utf-8-sig")
    matching_count = current.LLM_Status.isin(["matching_checkpoint", "matching_native_checkpoint"]).sum()
    print(f"[INFO] Step 4: {len(candidates)} candidates; {matching_count} matching API/native checkpoints; legacy results retained")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    main(args.refresh, args.limit)
