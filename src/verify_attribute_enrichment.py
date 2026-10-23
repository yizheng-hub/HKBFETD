"""Compare the preserved pre-enrichment v2 with the current release and its facts."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ai_cache import file_hash
from config import BASE_DIR, OUTPUT_DIR, DOCUMENTED_ATTRIBUTES_PATH


def main():
    root, out = Path(BASE_DIR), Path(OUTPUT_DIR)
    baseline = root/'output'/'v2_before_enrichment_20261023'
    index = json.loads((baseline/'BASELINE_INDEX.json').read_text(encoding='utf-8'))
    for name, expected in index['files'].items():
        assert file_hash(baseline/name) == expected['sha256'], 'Preserved baseline changed: '+name
    key = 'Building_ID_建筑ID'
    old = gpd.read_parquet(baseline/'HKBFETD_v2.parquet').set_index(key)
    new = gpd.read_parquet(out/'HKBFETD_v2.parquet').set_index(key)
    assert new.index.is_unique and set(old.index) == set(new.index)
    old = old.reindex(new.index)
    assert old.crs == new.crs and shapely.equals(old.geometry.to_numpy(),new.geometry.to_numpy()).all()
    summaries, changes = [], []
    for col in sorted(set(old.columns) & set(new.columns) - {'geometry'}):
        a, b = old[col], new[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            equal = pd.Series(np.isclose(a.to_numpy(dtype=float),b.to_numpy(dtype=float),rtol=1e-10,atol=1e-10,equal_nan=True),index=new.index)
        else:
            equal = a.astype('string').fillna('<NULL>').eq(b.astype('string').fillna('<NULL>'))
        summaries.append({'field':col,'before_nonmissing':int(a.notna().sum()),
                          'after_nonmissing':int(b.notna().sum()),'changed':int((~equal).sum())})
        if (~equal).any():
            changes.append(pd.DataFrame({key:new.index[~equal],'field':col,
                'before':a[~equal].astype('string').to_numpy(),'after':b[~equal].astype('string').to_numpy()}))
        if col in ['NUMABOVEGROUNDSTOREYS','TOPHEIGHT','BASEHEIGHT','Footprint_Area_sqm_占地面积']:
            assert equal.all(), 'Original footprint/physical source field changed: '+col
    pd.DataFrame(summaries).to_csv(out/'attribute_enrichment_field_changes.csv',index=False,encoding='utf-8-sig')
    detail = pd.concat(changes,ignore_index=True) if changes else pd.DataFrame(columns=[key,'field','before','after'])
    detail.to_parquet(out/'before_enrichment_to_v2_attribute_changes.parquet',index=False)
    facts = pd.read_csv(DOCUMENTED_ATTRIBUTES_PATH,dtype=str,keep_default_na=False)
    facts.to_csv(out/'documented_attribute_evidence.csv',index=False,encoding='utf-8-sig')
    for _, fact in facts.iterrows():
        row = new.loc[fact.building_id]
        assert row.BUILDINGCSUID == fact.csuid
        stored = json.loads(row.Reported_Attribute_Evidence)
        assert fact.evidence_id in {r['evidence_id'] for r in stored}
        if fact.adoption == 'component_storeys_supported':
            assert row['Floors_Count_地上层数'] == float(fact.value)
            assert row.Floors_Source == 'Documented_component_storeys'
            assert fact.evidence_id in json.loads(row.Floors_Document_Evidence_IDs)
        elif fact.adoption == 'component_gfa_supported':
            assert np.isclose(row.Documented_GFA_sqm,float(fact.value))
            assert fact.evidence_id in json.loads(row.Documented_GFA_Evidence_IDs)
    counts = {}
    for col in ['Height_m_建筑高度','Floors_Count_地上层数','Registered_GFA_sqm','GFA_Best_Available_sqm']:
        counts[col] = {'before_known':int(old[col].notna().sum()),'after_known':int(new[col].notna().sum()),
            'newly_filled':int((old[col].isna() & new[col].notna()).sum()),
            'newly_withheld':int((old[col].notna() & new[col].isna()).sum())}
    summary = {'buildings':len(new),'baseline_commit':index['code_commit'],
        'baseline_files_preserved':len(index['files']),'original_ids_and_geometry_unchanged':True,
        'evidence_facts':len(facts),'web_researched_buildings':int(facts.building_id.nunique()),
        'component_storey_documents':int(facts.adoption.eq('component_storeys_supported').sum()),
        'component_gfa_documents':int(facts.adoption.eq('component_gfa_supported').sum()),
        'coverage':counts,'fields_compared':len(summaries),'changed_cells':len(detail),
        'changed_main_function_records':int((old['UBEM_Main_Class_主类别'] != new['UBEM_Main_Class_主类别']).sum()),
        'changed_subclass_records':int((old['UBEM_Sub_Class_子类别'] != new['UBEM_Sub_Class_子类别']).sum()),
        'known_limits':'Every row is checked by code; online research covers only the explicitly counted named buildings. Source reports, harmonization and model outputs are not independent whole-stock ground truth.'}
    (out/'attribute_enrichment_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
