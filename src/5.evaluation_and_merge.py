"""Step 5: left-join verified AI corrections to the entire rule-engine stock."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from classification import finalize_function_fields, main_label, sub_label, taxonomy, UNKNOWN, MIXED
from config import RULE_ENGINE_OUTPUT_PATH, LLM_VERIFICATION_OUTPUT, STEP5_MERGED_OUTPUT_PATH
from data_quality import ID, ids, require_unique
from stage_io import read_table, attach_geometry, save_buildings


def merge_verified_results(base, cache, hash_function, native_results=None):
    result = base.copy()
    require_unique(result)
    result['LLM_Review_Status'] = 'not_verified_by_v2_LLM'
    by_id = result.set_index(ID)
    applied = set()
    if not cache.empty:
        cache = cache.copy()
        cache[ID] = ids(cache[ID])
        cache = cache.drop_duplicates([ID, 'Input_SHA256'], keep='last')
        if 'Checkpoint_Backend' in cache:
            cache = cache[cache.Checkpoint_Backend.fillna('api').astype(str).ne('interactive_native_session')]
        for _, decision in cache.iterrows():
            bid = decision[ID]
            if bid not in by_id.index or decision.Input_SHA256 != hash_function(by_id.loc[bid]):
                continue
            main, sub = main_label(decision.LLM_Main_Class), sub_label(decision.LLM_Sub_Class)
            confidence = float(decision.LLM_Confidence)
            if not np.isfinite(confidence) or confidence < .8 or main == UNKNOWN:
                continue
            if main != MIXED and taxonomy().get(sub) != main:
                continue
            by_id.loc[bid, ['Final_Main_Class','Final_Sub_Class','Classification_Source','LLM_Review_Status']] = [main, sub, 'LLM_matching_checkpoint', 'context_matched_self_reported_confidence']
            applied.add(bid)
    if native_results:
        for bid, decision in native_results.items():
            if bid not in by_id.index:
                continue
            status = decision.get('status', 'pending_no_matching_checkpoint')
            by_id.at[bid, 'LLM_Review_Status'] = status
            if not decision.get('adoptable') or status != 'matching_native_checkpoint':
                continue
            main, sub = main_label(decision.get('main')), sub_label(decision.get('sub'))
            confidence = float(decision.get('confidence', np.nan))
            if not np.isfinite(confidence) or confidence < .8 or main == UNKNOWN:
                continue
            if main != MIXED and taxonomy().get(sub) != main:
                continue
            by_id.loc[bid, ['Final_Main_Class','Final_Sub_Class','Classification_Source']] = [main, sub, 'LLM_native_interactive']
            applied.add(bid)
    result = by_id.reset_index()
    assert len(result) == len(base) and set(result[ID]) == set(base[ID])
    print(f'[INFO] Step 5: retained {len(base)} buildings; applied {len(applied)} matching corrections')
    finalized = finalize_function_fields(result)
    return finalized


def main():
    spec = importlib.util.spec_from_file_location('llm_stage', Path(__file__).with_name('4.llm_verification.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = read_table(RULE_ENGINE_OUTPUT_PATH)
    cache = pd.read_csv(LLM_VERIFICATION_OUTPUT, low_memory=False) if Path(LLM_VERIFICATION_OUTPUT).exists() else pd.DataFrame()
    native_results = module.native_results_for_base(base)
    result = merge_verified_results(base, cache, module.input_hash, native_results=native_results)
    geometry = str(Path(RULE_ENGINE_OUTPUT_PATH).with_suffix('.geojson'))
    save_buildings(attach_geometry(result, geometry), STEP5_MERGED_OUTPUT_PATH)
    return True


if __name__ == '__main__':
    main()
