"""Fixed-model sensitivity to legacy heights; not independent accuracy testing."""
from pathlib import Path
import json
import pickle
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classification import UNKNOWN
from config import OUTPUT_DIR, INTERMEDIATE_DIR

out=Path(OUTPUT_DIR)
with open(Path(INTERMEDIATE_DIR)/'feature_X_all.pkl','rb') as stream:X=pickle.load(stream)
with open(Path(INTERMEDIATE_DIR)/'feature_base_indexed.pkl','rb') as stream:base=pickle.load(stream)
with open(Path(INTERMEDIATE_DIR)/'ml_models.pkl','rb') as stream:package=pickle.load(stream)
old=pd.read_csv(ROOT/'output/HK_UBEM_Buildings_Public_v1_1.csv',
    usecols=['Building_ID_建筑ID','Height_m_建筑高度'],dtype={'Building_ID_建筑ID':'string'}).set_index('Building_ID_建筑ID')
assert set(old.index)==set(X.index)
perturbed=X.copy()
perturbed['Estimated_Height']=pd.to_numeric(old.reindex(X.index)['Height_m_建筑高度'],errors='coerce').to_numpy()
perturbed['height_missing']=perturbed.Estimated_Height.isna().astype('int8')
model=package['main']
actual=model.predict_proba(X)
legacy=model.predict_proba(perturbed)
classes=model.named_steps['classifier'].classes_
probability_fields={'Residential_住宅类别':'ML_Prob_Residential_住宅概率',
    'Commercial_商业类别':'ML_Prob_Commercial_商业概率','Industrial_工业类别':'ML_Prob_Industrial_工业概率'}
published=pd.read_parquet(out/'HKBFETD_v2.parquet',columns=['Building_ID_建筑ID',
    'Classification_Source_分类来源',*probability_fields.values()]).set_index('Building_ID_建筑ID').reindex(X.index)
assert np.allclose(actual,published[[probability_fields[c] for c in classes]].to_numpy(),atol=1e-10)
changed=actual.argmax(axis=1)!=legacy.argmax(axis=1)
conflict=base.Classification_QA.fillna('').str.contains('function_evidence_conflict|subclass_parent_conflict|registered_area_function_conflict',regex=True)
component_scope_unresolved=base.Dimension_Ratio_Scope_QA.fillna('').eq('low_height_to_count_scope_review')
eligible=base.Final_Main_Class.eq(UNKNOWN)&~conflict&~component_scope_unresolved
actual_fill=actual.max(axis=1)>=np.array([package['main_thresholds'][label] for label in classes[actual.argmax(axis=1)]])
legacy_fill=legacy.max(axis=1)>=np.array([package['main_thresholds'][label] for label in classes[legacy.argmax(axis=1)]])
actual_fill &= np.isin(classes[actual.argmax(axis=1)],package['validated_main_classes'])
legacy_fill &= np.isin(classes[legacy.argmax(axis=1)],package['validated_main_classes'])
unverified_structure=base.Structure_Use_Type.fillna('').eq('')
actual_fill &= ~(X.Estimated_Height.gt(100) & X.area.lt(150) & unverified_structure)
legacy_fill &= ~(perturbed.Estimated_Height.gt(100) & X.area.lt(150) & unverified_structure)
assert np.array_equal(eligible.to_numpy() & actual_fill,
    published['Classification_Source_分类来源'].eq('Machine_Learning_v2').to_numpy()), 'Sensitivity acceptance differs from published decisions'
decision_changed=eligible.to_numpy()&((actual_fill!=legacy_fill)|(changed&actual_fill&legacy_fill))
table=pd.DataFrame({'BUILDINGSTRUCTUREID':X.index,'new_component_height':X.Estimated_Height,
    'legacy_height':perturbed.Estimated_Height,'new_model_argmax':classes[actual.argmax(axis=1)],
    'legacy_height_model_argmax':classes[legacy.argmax(axis=1)],'new_probability_max':actual.max(axis=1),
    'legacy_height_probability_max':legacy.max(axis=1),'main_argmax_changed':changed,
    'eligible_function_decision_changed':decision_changed})
table.loc[changed|decision_changed].to_csv(out/'height_only_fixed_model_sensitivity.csv',index=False,encoding='utf-8-sig')
summary={'analysis':'Same final trained main classifier and all other final features; replace only height and its missingness flag with v1.1 values',
    'not_a_retraining_ablation':True,'not_independent_validation_accuracy':True,
    'original_height_reference_differs':'v1.1 includes default heights/unbased datum elevations; v2 is component height',
    'records':len(X),'argmax_class_changed_records':int(changed.sum()),
    'eligible_final_function_decision_changed_records':int(decision_changed.sum()),
    'mean_probability_vector_L1_difference':float(np.abs(actual-legacy).sum(axis=1).mean()),
    'scope':'Main-function sensitivity only. Full release changes also include storeys, fusion, semantic rules and retraining; these counts are not independently measured corrected errors.'}
(out/'height_only_fixed_model_sensitivity.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
