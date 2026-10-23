"""Supplement the numbered release with preservation and baseline comparisons."""
import json
from pathlib import Path
import sys
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ai_cache import file_hash
from config import CSDI_BUILDINGS_PATH, CSDI_METADATA_PATH

out = ROOT/'output/v2'
baseline = ROOT/'output/v2_terrain_baseline_20261023'
key = 'Building_ID_建筑ID'
new = pd.read_parquet(out/'HKBFETD_v2.parquet').set_index(key)
old = pd.read_parquet(baseline/'HKBFETD_v2.parquet').set_index(key)
assert new.index.is_unique and set(new.index) == set(old.index)
old = old.reindex(new.index)
columns = ['Height_m_建筑高度','Floors_Count_地上层数','Height_Source','Height_Quality',
           'Floors_Source','UBEM_Main_Class_主类别','UBEM_Sub_Class_子类别','GFA_Proxy_sqm']
comparison = pd.DataFrame(index=new.index)
for col in columns:
    comparison['Terrain_baseline_'+col] = old[col]
    comparison['Component_revision_'+col] = new[col]
comparison['New_Height_Reference'] = new.Height_Reference
comparison['New_Ground_To_Top_Height_m'] = new.Ground_To_Top_Height_m
comparison['New_CSDI_Match_Quality'] = new.CSDI_Match_Quality
comparison['New_Physical_Review_Flags'] = new.Physical_Review_Flags
comparison.to_csv(out/'terrain_baseline_to_component_revision.csv',encoding='utf-8-sig')

protected = json.loads((baseline/'legacy_preservation_validation.json').read_text(encoding='utf-8'))
archive_manifest = ROOT/'output'/'history_archive_manifest.json'
archive = json.loads(archive_manifest.read_text(encoding='utf-8-sig')) if archive_manifest.exists() else None
archived = {r['original_path']:r for r in archive['files']} if archive else {}
for record in protected['checks']:
    source = ROOT/record['path']
    record['preserved_in_history_archive'] = False
    if not source.is_file() and record['path'] in archived:
        source = Path(archived[record['path']]['archived_path'])
        assert source.resolve().is_relative_to(Path(archive['archive_root']).resolve()), 'Archive path escapes its declared root'
        record['preserved_in_history_archive'] = True
    record['sha256'] = file_hash(source)
    record['unchanged'] = record['sha256'] == record['baseline_sha256']
protected['unchanged'] = sum(r['unchanged'] for r in protected['checks'])
protected['changed'] = [r['path'] for r in protected['checks'] if not r['unchanged']]
assert not protected['changed'], 'Original protected artifact changed'
(out/'legacy_preservation_validation.json').write_text(json.dumps(protected,ensure_ascii=False,indent=2),encoding='utf-8')
for record in json.loads((baseline/'BASELINE_INDEX.json').read_text(encoding='utf-8'))['files']:
    assert file_hash(baseline/record['file']) == record['sha256']

with zipfile.ZipFile(out/'reproduction_inputs.zip') as archive:
    input_names = archive.namelist()
    for name in [Path(CSDI_BUILDINGS_PATH).relative_to(ROOT).as_posix(), Path(CSDI_METADATA_PATH).relative_to(ROOT).as_posix()]:
        assert name in input_names
with zipfile.ZipFile(out/'v2_source_snapshot.zip') as archive:
    for path in list((ROOT/'src').glob('*.py')) + list((ROOT/'tests').glob('*.py')):
        assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes()

model = json.loads((out/'model_validation.json').read_text(encoding='utf-8'))
previous_model = json.loads((baseline/'model_validation.json').read_text(encoding='utf-8'))
qa = json.loads((out/'dimension_qa_summary.json').read_text(encoding='utf-8'))
published_v1 = pd.read_csv(ROOT/'output/HK_UBEM_Buildings_Public_v1_1.csv',
    usecols=['Height_m_建筑高度','Footprint_Area_sqm_占地面积'])
summary = {'buildings':len(new),'original_artifacts_preserved':protected['unchanged'],
    'terrain_baseline_preserved':True,'source_zip_matches_current_code':True,'frozen_input_files':len(input_names),
    'primary_height_definition':'structural component top minus base; not architectural/ground-to-top height',
    'source_match_counts':new.CSDI_Match_Quality.value_counts().to_dict(),
    'component_height_known':int(new['Height_m_建筑高度'].notna().sum()),
    'component_height_unknown':int(new['Height_m_建筑高度'].isna().sum()),
    'ground_to_top_known':int(new.Ground_To_Top_Height_m.notna().sum()),
    'reported_storeys_known':int(new['Floors_Count_地上层数'].notna().sum()),
    'function_unknown':int(new['UBEM_Main_Class_主类别'].eq('Unknown_未知类别').sum()),
    'small_tall_threshold_records':qa['flag_counts']['small_footprint_tall_building'],
    'height_rounded_to_15m':qa['height_rounded_to_15m_records'],
    'v1_1_height_exactly_15m':int(published_v1['Height_m_建筑高度'].eq(15).sum()),
    'v1_1_height_rounded_to_15m':int(published_v1['Height_m_建筑高度'].round(1).eq(15).sum()),
    'v1_1_small_tall_threshold_records':int((published_v1['Footprint_Area_sqm_占地面积'].lt(150)&published_v1['Height_m_建筑高度'].gt(150)).sum()),
    'main_classes_changed_from_terrain_baseline':int(new['UBEM_Main_Class_主类别'].ne(old['UBEM_Main_Class_主类别']).sum()),
    'terrain_baseline_spatial_macro_f1':previous_model['macro_f1'],'component_revision_spatial_macro_f1':model['macro_f1'],
    'model_metric_interpretation':model['label_basis'],
    'comparison_scope':'CSDI augmentation plus subsequent function/structural-use fixes; not a height-only controlled experiment',
    'small_tall_identified_bridge_structures':qa['small_tall_identified_bridge_structures'],
    'small_tall_unresolved_records':qa['small_tall_unresolved_records'],
    'caution':'Height definitions changed between terrain baseline and component revision; numeric differences are not independently measured errors.'}
(out/'component_revision_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
fusion={'buildings':len(new),'height_quality':new.Height_Quality.value_counts().to_dict(),
    'floor_quality':new.Floors_Quality.value_counts().to_dict(),'height_sources':new.Height_Source.value_counts().to_dict(),
    'csdi_match_quality':new.CSDI_Match_Quality.value_counts().to_dict(),
    'structural_use_types':new.Structure_Use_Type.value_counts().to_dict(),
    'energy_assessment_scope':new.Energy_Assessment_Scope.value_counts().to_dict(),
    'reader_cases':json.loads(new.loc[new.index.isin(['181189','170369','192326','87318']),
        ['Height_m_建筑高度','Height_Source','Floors_Count_地上层数','Floors_Source','Ground_To_Top_Height_m']].reset_index().to_json(orient='records'))}
(out/'fusion_quality_summary.json').write_text(json.dumps(fusion,ensure_ascii=False,indent=2),encoding='utf-8')
coverage=pd.DataFrame({'source_structure_type':new.CSDI_Structure_Type.fillna('source_id_unmatched'),
    'height_known':new['Height_m_建筑高度'].notna(),'storeys_known':new['Floors_Count_地上层数'].notna(),
    'function_known':~new['UBEM_Main_Class_主类别'].eq('Unknown_未知类别')})
coverage=coverage.groupby('source_structure_type').agg(records=('height_known','size'),
    known_component_heights=('height_known','sum'),known_component_storeys=('storeys_known','sum'),
    known_or_inferred_functions=('function_known','sum'))
coverage['height_coverage_fraction']=coverage.known_component_heights/coverage.records
coverage['function_coverage_fraction']=coverage.known_or_inferred_functions/coverage.records
coverage.to_csv(out/'height_coverage_by_structure_type.csv',encoding='utf-8-sig')
geometry_qa=new[['CSDI_Geometry_IoU','CSDI_Match_Quality','CSDI_Structure_Type','LandsD_Structure_Type_地政署建筑类型']].copy()
geometry_qa['CSDI_Geometry_Valid']=~new.CSDI_Match_Quality.isin(['id_unmatched','invalid_source_geometry'])
geometry_qa['CSDI_Type_Match']=new['LandsD_Structure_Type_地政署建筑类型'].eq(new.CSDI_Structure_Type.map({
    'Tower':'T','Podium':'P','Open-sided Structure':'T','Temporary Structure':'T'}))
geometry_qa.to_csv(out/'csdi_geometry_quality.csv',encoding='utf-8-sig')
new[new.Physical_Review_Flags.str.contains('small_footprint_tall_building',na=False)].drop(columns='geometry').to_csv(out/'small_tall_resolved_structures.csv',encoding='utf-8-sig')
new[new.Energy_Assessment_Scope.eq('excluded_nonbuilding_structure')].drop(columns='geometry').to_csv(out/'verified_nonbuilding_structures.csv',encoding='utf-8-sig')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
