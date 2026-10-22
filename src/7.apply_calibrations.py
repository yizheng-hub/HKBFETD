"""Step 7: apply only context-matched visual decisions, preserving physical fields."""
from pathlib import Path
import pandas as pd
from classification import finalize_function_fields, NONASSESSED, UNKNOWN
from config import STEP5_MERGED_OUTPUT_PATH, AI_CALIBRATED_OUTPUT_PATH, FINAL_SUGGESTIONS_PATH
from data_quality import ID, ids, require_unique
from stage_io import read_table, attach_geometry, save_buildings


def apply_correction_suggestions(base, suggestions):
    result = base.copy().set_index(ID)
    result['Geometry_Review_Status'] = 'no_verified_v2_geometry_change'
    if not suggestions.empty:
        usable = suggestions[suggestions.get('Cache_Status', pd.Series('',index=suggestions.index)).eq('matching_checkpoint')].copy()
        for fragment, group in usable.groupby('fragment_id', sort=True):
            if str(fragment) not in result.index:
                continue
            merges = group[group.ai_decision.eq('merge')]
            if len(merges) == 1:
                host = str(merges.host_id.iloc[0])
                host_qa = result.loc[host,'Classification_QA'] if host in result.index and 'Classification_QA' in result else ''
                host_conflict = pd.notna(host_qa) and any(flag in str(host_qa) for flag in ['function_evidence_conflict','registered_area_function_conflict'])
                if host_conflict:
                    result.loc[str(fragment),'Geometry_Review_Status'] = 'host_function_conflict_no_inheritance'
                elif host in result.index and result.loc[host, 'Final_Main_Class'] not in [UNKNOWN,NONASSESSED]:
                    for col in ['Final_Main_Class','Final_Sub_Class','Evidence_Subclass_Weights']:
                        if col in result:
                            result.loc[str(fragment),col] = result.loc[host,col]
                    result.loc[str(fragment),'Classification_Source'] = 'Inherited_context_matched_VLM'
                    result.loc[str(fragment),'Geometry_Review_Status'] = 'function_inherited_geometry_not_dissolved'
            elif len(merges) > 1:
                result.loc[str(fragment),'Geometry_Review_Status'] = 'conflicting_hosts_unresolved'
            if group.ai_decision.eq('delete').any():
                # A visual noise decision never physically removes a footprint.
                result.loc[str(fragment),'Geometry_Review_Status'] = 'visual_delete_suggestion_requires_manual_review'
        pending = suggestions[suggestions.Cache_Status.ne('matching_checkpoint')].fragment_id.astype(str).unique()
        missing = result.index.intersection(pending)
        result.loc[missing,'Geometry_Review_Status'] = 'pending_visual_evidence_or_context_review'
    result = finalize_function_fields(result.reset_index())
    require_unique(result)
    assert set(result[ID]) == set(base[ID])
    for raw in ['NUMABOVEGROUNDSTOREYS','TOPHEIGHT','BASEHEIGHT','Estimated_Height','Estimated_Floors']:
        left, right = base.set_index(ID)[raw], result.set_index(ID)[raw].reindex(base[ID])
        assert (left.eq(right) | (left.isna() & right.isna())).all(), raw
    return result


def main():
    base = read_table(STEP5_MERGED_OUTPUT_PATH)
    suggestions = pd.read_csv(FINAL_SUGGESTIONS_PATH, dtype={'fragment_id':'string','host_id':'string'}, low_memory=False) if Path(FINAL_SUGGESTIONS_PATH).exists() else pd.DataFrame()
    result = apply_correction_suggestions(base,suggestions)
    save_buildings(attach_geometry(result, str(Path(STEP5_MERGED_OUTPUT_PATH).with_suffix('.geojson'))),AI_CALIBRATED_OUTPUT_PATH)
    print(f'[INFO] Step 7: {len(result)} buildings; raw dimensions and observed/estimated distinctions preserved')
    return True


if __name__ == '__main__':
    main()
