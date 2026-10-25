"""Apply verified physical inputs to the original stage-3 classification checkpoint.

The original classification checkpoint remains the authority for unchanged inputs.
Only records whose GFA rule evidence was withdrawn are re-evaluated with the frozen
original rule function. No keyword, taxonomy, weight, or decision rule is changed.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import sys
import ast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

METHOD = Path(os.environ['HKBFETD_REPLAY_FROZEN_METHOD']).resolve()
OUT = Path(os.environ['HKBFETD_REPLAY_OUTPUT_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def ids(series):
    return series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if os.environ.get('PYTHONHASHSEED') != '0':
    raise RuntimeError('Run in a fresh process with PYTHONHASHSEED=0 for reproducible set iteration')

sys.path.insert(0, str(METHOD / 'src'))
import config  # noqa: E402
import utils  # noqa: E402
rule = load_module('frozen_original_rule_engine', METHOD / 'src/3.rule_engine.py')

old_path = Path(os.environ['HKBFETD_REPLAY_LEGACY_STAGE3']).resolve()
step2_path = Path(os.environ['HKBFETD_REPLAY_LEGACY_STAGE2']).resolve()
physical_path = Path(os.environ['HKBFETD_REPLAY_VERIFIED_PHYSICAL']).resolve()
delta_path = Path(os.environ['HKBFETD_REPLAY_GFA_DELTA']).resolve()

assert sha(old_path) == 'c8b36fa65e5cf07fca77ce4cddf4a46a1e7902fdb82b369bc8e960fa777ee010'
assert sha(step2_path) == 'f37a03a0d076e97d77d2c700b0153fdb0be3415d70a81e100145761853b2885a'
assert sha(physical_path) == '783e30db5240eb02cf6217e03707f98395a129d942968ff6115cc1b23b418c88'

print('[1/6] Reading original stage-3 checkpoint', flush=True)
old = pd.read_csv(old_path, low_memory=False, dtype={'BUILDINGSTRUCTUREID': 'string', 'BUILDINGCSUID': 'string'})
old['BUILDINGSTRUCTUREID'] = ids(old['BUILDINGSTRUCTUREID'])
old = old.set_index('BUILDINGSTRUCTUREID', verify_integrity=True)
assert len(old) == 341153

schema = pq.read_schema(physical_path)
id_col = next(c for c in schema.names if c.startswith('Building_ID_'))
height_col = next(c for c in schema.names if c.startswith('Height_m_'))
floor_col = next(c for c in schema.names if c.startswith('Floors_Count_'))
physical_cols = [id_col, height_col, floor_col, 'GFA_DOMESTIC_SUM', 'GFA_NONDOMESTIC_SUM', 'BUILDINGCSUID']
physical = pq.read_table(physical_path, columns=physical_cols).to_pandas()
physical['BUILDINGSTRUCTUREID'] = ids(physical[id_col])
physical = physical.set_index('BUILDINGSTRUCTUREID', verify_integrity=True).reindex(old.index)
assert physical.index.equals(old.index)

old['NUMABOVEGROUNDSTOREYS'] = pd.to_numeric(physical[floor_col], errors='coerce')
old['GFA_DOMESTIC_SUM'] = pd.to_numeric(physical['GFA_DOMESTIC_SUM'], errors='coerce')
old['GFA_NONDOMESTIC_SUM'] = pd.to_numeric(physical['GFA_NONDOMESTIC_SUM'], errors='coerce')
old['Corrected_Structure_Height_m'] = pd.to_numeric(physical[height_col], errors='coerce')
old['Physical_Attribute_Overlay'] = 'verified_v2_physical_repair'

delta = pd.read_csv(delta_path, dtype={'BUILDINGSTRUCTUREID': 'string'}, low_memory=False)
delta_ids = set(ids(delta['BUILDINGSTRUCTUREID']))
pre_screened = set(old.index[(old['Classification_Source'].astype(str) == 'Pre-Screening') & old.index.isin(delta_ids)])
gfa_only = {'GFA', 'GFA_Mixed'}
rerun_ids = set(old.index[old.index.isin(delta_ids) & old['Classification_Source'].astype(str).isin(gfa_only)])
independent_ids = delta_ids - pre_screened - rerun_ids
assert len(delta_ids) == 8985 and len(pre_screened) == 18 and len(rerun_ids) == 4159 and len(independent_ids) == 4808

def without_gfa_source(value):
    tokens = [token for token in str(value).split('+') if token and token != 'GFA']
    return '+'.join(tokens)

def without_gfa_evidence(value):
    if pd.isna(value):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return value
    if not isinstance(parsed, list):
        return value
    return str([item for item in parsed if not str(item).startswith('GFA:')])

old.loc[list(independent_ids), 'Classification_Source'] = old.loc[list(independent_ids), 'Classification_Source'].map(without_gfa_source)
old.loc[list(independent_ids), 'Evidence_Details'] = old.loc[list(independent_ids), 'Evidence_Details'].map(without_gfa_evidence)
old.loc[list(independent_ids), 'Classification_Repair_Reason'] = 'unverified_GFA_withdrawn_original_non_GFA_classification_preserved'

print('[2/6] Reading the archived original stage-2 groups', flush=True)
step2 = gpd.read_file(step2_path)
step2['BUILDINGSTRUCTUREID'] = ids(step2['BUILDINGSTRUCTUREID'])
subset = step2[step2['BUILDINGSTRUCTUREID'].isin(rerun_ids)].copy()
assert set(subset['BUILDINGSTRUCTUREID']) == rerun_ids
del step2

overlay = physical.loc[list(rerun_ids), [height_col, floor_col, 'GFA_DOMESTIC_SUM', 'GFA_NONDOMESTIC_SUM']].copy()
overlay[height_col] = pd.to_numeric(overlay[height_col], errors='coerce')
overlay[floor_col] = pd.to_numeric(overlay[floor_col], errors='coerce')
for col in [height_col, floor_col, 'GFA_DOMESTIC_SUM', 'GFA_NONDOMESTIC_SUM']:
    values = overlay[col]
    subset[col + '__overlay'] = subset['BUILDINGSTRUCTUREID'].map(values)
subset['NUMABOVEGROUNDSTOREYS'] = subset[floor_col + '__overlay']
subset['GFA_DOMESTIC_SUM'] = subset['GFA_DOMESTIC_SUM__overlay']
subset['GFA_NONDOMESTIC_SUM'] = subset['GFA_NONDOMESTIC_SUM__overlay']
# The frozen rule reads TOPHEIGHT/BASEHEIGHT only as a fallback height. Feed the
# verified relative structural height to that interface without publishing it as mPD.
subset['TOPHEIGHT'] = subset[height_col + '__overlay']
subset['BASEHEIGHT'] = np.where(subset[height_col + '__overlay'].notna(), 0.0, np.nan)
subset['Pre_Class_Main'] = subset['CATEGORY'].astype(str).map(config.LANDSD_CATEGORY_MAP_MAIN)
subset['Pre_Class_Sub'] = subset['CATEGORY'].astype(str).map(config.LANDSD_CATEGORY_MAP_SUB)
subset.drop(columns=[c for c in subset.columns if c.endswith('__overlay')], inplace=True)

print(f'[3/6] Running frozen original rule function for {len(rerun_ids)} GFA-only decisions', flush=True)
keyword_tool = utils.init_keyword_tool(str(METHOD / 'ctl/keywords.json'))
results = {}
for count, (building_id, group) in enumerate(subset.groupby('BUILDINGSTRUCTUREID', sort=True), start=1):
    results[building_id] = rule.classify_building_rule_engine_optimized(group, keyword_tool).to_dict()
    if count % 500 == 0:
        print(f'  classified {count}/{len(rerun_ids)}', flush=True)
result = pd.DataFrame.from_dict(results, orient='index')
assert set(result.index) == rerun_ids

def canonical_main(value):
    return utils.canonicalize_main_class_label(value)

def canonical_sub(value):
    return utils.canonicalize_sub_class_label(value)

def canonical_main_prop(value):
    text = '' if pd.isna(value) else str(value)
    for source, target in [('住宅类别', 'Residential_住宅类别'), ('商业类别', 'Commercial_商业类别'),
                           ('工业类别', 'Industrial_工业类别'), ('混合用途', 'Mixed-use_混合用途'),
                           ('非评估类别', 'Non-assessed_非评估类别'), ('未知类别', 'Unknown_未知类别')]:
        text = text.replace(source + ':', target + ':')
    return text

def canonical_sub_prop(value):
    text = '' if pd.isna(value) else str(value)
    import re
    return re.sub(r'([\u3400-\u9fff/、 ]+)[（(]([^()（）]+)[)）](?=:)',
                  lambda m: f'{m.group(2).strip()}_{m.group(1).strip()}', text)

result['Final_Main_Class'] = result['Final_Main_Class'].map(canonical_main)
result['Final_Sub_Class'] = result['Final_Sub_Class'].map(canonical_sub)
result['Main_Class_Proportions'] = result['Main_Class_Proportions'].map(canonical_main_prop)
result['Sub_Class_Proportions'] = result['Sub_Class_Proportions'].map(canonical_sub_prop)

weak_fallback = result['Classification_Source'].astype(str).eq('LandsD_Category')
weak_fallback_audit = result.loc[weak_fallback].copy()
weak_fallback_audit.index.name = 'BUILDINGSTRUCTUREID'
weak_fallback_audit.reset_index().to_csv(
    OUT / 'stage3_gfa_weak_fallbacks_withheld.csv', index=False, encoding='utf-8-sig'
)
if int(weak_fallback.sum()) != 117:
    raise AssertionError(f'Expected 117 GFA-withdrawal LandsD-only fallbacks, got {int(weak_fallback.sum())}')
result.loc[weak_fallback, 'Final_Main_Class'] = 'Unknown_未知类别'
result.loc[weak_fallback, 'Final_Sub_Class'] = 'Unknown_未知类别'
result.loc[weak_fallback, 'Classification_Source'] = 'Unknown_after_unverified_GFA_withdrawal'
result.loc[weak_fallback, 'Is_Mixed_Use'] = False
result.loc[weak_fallback, 'Confidence_Score'] = 0.0
result.loc[weak_fallback, 'Is_Conflicted'] = True
result.loc[weak_fallback, 'Main_Class_Proportions'] = np.nan
result.loc[weak_fallback, 'Sub_Class_Proportions'] = np.nan
result.loc[weak_fallback, 'Evidence_Details'] = 'Withheld: GFA withdrawn; only generic LandsD category remained'

print('[4/6] Replacing only original classifier result columns', flush=True)
result_columns = ['Final_Main_Class', 'Final_Sub_Class', 'Classification_Source', 'Is_Mixed_Use',
                  'Confidence_Score', 'Evidence_Details', 'Is_Conflicted', 'Main_Class_Proportions',
                  'Sub_Class_Proportions', 'Final_Recalc_Commercial_POI_Trace',
                  'Final_Recalc_Commercial_POI_Counts']
before = old.loc[list(rerun_ids), result_columns].copy()
for col in result_columns:
    old.loc[result.index, col] = result[col]
old.loc[result.index, 'Classification_Repair_Reason'] = 'original_rule_rerun_after_unverified_GFA_withdrawal'
old.loc[list(pre_screened), 'Classification_Repair_Reason'] = 'pre_screen_preserved_GFA_not_classification_evidence'
old['Classification_Repair_Reason'] = old['Classification_Repair_Reason'].fillna('original_checkpoint_preserved')

comparison = pd.DataFrame(index=result.index)
for col in ['Final_Main_Class', 'Final_Sub_Class', 'Classification_Source', 'Main_Class_Proportions', 'Sub_Class_Proportions']:
    comparison['Old_' + col] = before[col]
    comparison['Corrected_' + col] = old.loc[result.index, col]
comparison['Main_Changed'] = comparison['Old_Final_Main_Class'].fillna('') != comparison['Corrected_Final_Main_Class'].fillna('')
comparison['Sub_Changed'] = comparison['Old_Final_Sub_Class'].fillna('') != comparison['Corrected_Final_Sub_Class'].fillna('')
comparison['Source_Changed'] = comparison['Old_Classification_Source'].fillna('') != comparison['Corrected_Classification_Source'].fillna('')
comparison.index.name = 'BUILDINGSTRUCTUREID'
comparison.reset_index().to_csv(OUT / 'stage3_gfa_withdrawal_comparison.csv', index=False, encoding='utf-8-sig')

print('[5/6] Writing corrected original-method stage-3 CSV', flush=True)
corrected_path = OUT / 'step3_rule_engine_classification.csv'
old.reset_index().to_csv(corrected_path, index=False, encoding='utf-8-sig')

summary = {
    'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'STAGE3_CORRECTED_ORIGINAL_METHOD',
    'pythonhashseed': os.environ['PYTHONHASHSEED'], 'original_checkpoint_sha256': sha(old_path),
    'original_step2_sha256': sha(step2_path), 'verified_physical_sha256': sha(physical_path),
    'frozen_rule_source_sha256': sha(METHOD / 'src/3.rule_engine.py'),
    'frozen_keywords_sha256': sha(METHOD / 'ctl/keywords.json'),
    'records': len(old), 'original_checkpoint_records_preserved': len(old) - len(rerun_ids),
    'gfa_rule_delta_records': len(delta_ids), 'pre_screened_preserved': len(pre_screened),
    'independent_non_gfa_classification_preserved': len(independent_ids),
    'original_rule_rerun_records': len(rerun_ids),
    'gfa_withdrawal_landsd_only_fallbacks_withheld': int(weak_fallback.sum()),
    'main_changed': int(comparison['Main_Changed'].sum()), 'sub_changed': int(comparison['Sub_Changed'].sum()),
    'source_changed': int(comparison['Source_Changed'].sum()),
    'corrected_stage3_path': corrected_path.name, 'corrected_stage3_sha256': sha(corrected_path),
    'classification_logic_changed': False, 'keyword_file_changed': False,
    'unchanged_osm_matched_poi_prompt_context_preserved': True,
    'adapter': 'Verified floors/GFA and relative structural height were supplied to the original function interfaces; raw mPD was not interpreted as height. GFA-only decisions were rerun; unchanged keyword/BDBIAR/OSM/LandsD/podium decisions were preserved with the withdrawn GFA token removed from provenance.',
    'limitations': ['The historical run did not record PYTHONHASHSEED; unchanged classifications are therefore preserved from its exact checkpoint.',
                    'Only classification records whose original GFA evidence was withdrawn are rerun.']
}
(OUT / 'stage3_build_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('[6/6] ' + json.dumps(summary, ensure_ascii=False), flush=True)
