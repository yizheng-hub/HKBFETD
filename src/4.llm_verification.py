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
                    API_KEY, BASE_URL, CLOUD_MODEL_NAME, LLM_TAXONOMY)
from data_quality import ID, ids, require_unique

PROMPT_VERSION = "v2-component-function-evidence-20261023"
RESULT_FIELDS = [ID, "Input_SHA256", "LLM_Main_Class", "LLM_Sub_Class", "LLM_Confidence", "LLM_Reasoning"]


def generate_prompt(row):
    fields = ["BUILDINGNAMEEN", "BUILDINGNAMETC", "BDBIAR_CLASS", "BDBIAR_ADDRESS_C", "BDBIAR_Match_Quality", "CATEGORY",
              "BUILDINGSTRUCTURETYPE", "Primary_Building_Main", "Primary_Building_Sub", "Matched_POIs",
              "Estimated_Floors", "Floors_Quality", "Estimated_Height", "Height_Quality", "Height_Reference"]
    evidence = {field: None if pd.isna(row.get(field)) else row.get(field) for field in fields}
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
                         "LLM_Reasoning": result.get("reasoning", "")}
                cache = pd.concat([cache, pd.DataFrame([entry])], ignore_index=True)
                save_cache(cache, LLM_VERIFICATION_OUTPUT)
            except Exception as error:
                print(f"[WARN] LLM request not checkpointed: {type(error).__name__}")
    cache = cache.drop_duplicates([ID, "Input_SHA256"], keep="last")
    save_cache(cache, LLM_VERIFICATION_OUTPUT)
    current = candidates[[ID, "Input_SHA256"]].merge(cache, on=[ID, "Input_SHA256"], how="left", validate="one_to_one")
    current["LLM_Status"] = np.where(current.LLM_Confidence.notna(), "matching_checkpoint", "pending_no_matching_checkpoint")
    legacy_path = Path(LEGACY_OUTPUT_DIR) / "step4_llm_verified_classification.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path, low_memory=False)
        legacy[ID] = ids(legacy[ID])
        legacy = legacy.drop_duplicates(ID).rename(columns={c: "Legacy_" + c for c in legacy if c != ID})
        current = current.merge(legacy, on=ID, how="left", validate="one_to_one")
    current.to_csv(Path(LLM_VERIFICATION_OUTPUT).with_name("step4_current_status.csv"), index=False, encoding="utf-8-sig")
    current[current.LLM_Status.ne("matching_checkpoint")].to_csv(
        Path(LLM_VERIFICATION_OUTPUT).with_name("step4_pending_review.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] Step 4: {len(candidates)} candidates; {(current.LLM_Status == 'matching_checkpoint').sum()} matching checkpoints; legacy results retained")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    main(args.refresh, args.limit)
