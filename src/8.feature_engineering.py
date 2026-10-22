"""Step 8: label-free spatial/physical features with explicit missingness.

No neighbor class labels, default physical heights or pre-split imputations.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from classification import MAIN_CLASSES, UNKNOWN, sub_label, taxonomy
from config import (AI_CALIBRATED_OUTPUT_PATH, OFFICIAL_LIB_BASE_PATH, INTERMEDIATE_DIR,
    FEATURE_X_ALL_PATH, FEATURE_X_TRAIN_ML_PATH, FEATURE_Y_MULTILABEL_PATH,
    FEATURE_TRAINING_DATA_ML_PATH, FEATURE_BASE_INDEXED_PATH)
from data_quality import ID, numbers
from stage_io import read_table, attach_geometry


def prepare_training_data(frame):
    source = frame.Classification_Source.fillna('').astype(str)
    category = frame.CATEGORY.astype(str).str.replace(r'\.0$','',regex=True)
    official = category.isin(['2','3']) & source.str.contains('LandsD_Category',regex=False)
    bdbiar = source.str.contains('BDBIAR',regex=False) & frame.BDBIAR_Match_Quality.eq('inside_unique')
    keyword = source.str.contains('Keyword',regex=False)
    building = source.eq('OSM_Building_Function')
    eligible = frame.Final_Main_Class.isin(MAIN_CLASSES) & (official | bdbiar | keyword | building)
    eligible &= ~source.str.contains('Machine_Learning|Inherited|LLM',regex=True)
    if 'Classification_QA' in frame:
        eligible &= frame.Classification_QA.fillna('').eq('')
    if 'Is_Conflicted' in frame:
        eligible &= ~frame.Is_Conflicted.astype(str).str.lower().eq('true')
    training = frame[eligible].copy()
    training['Training_Label_Basis'] = 'weak_supervision_source_inferred_not_independent_ground_truth'
    return training


def compute_advanced_features(frame):
    indexed = frame.set_index(ID)
    X = pd.DataFrame(index=indexed.index)
    X['area'] = indexed.geometry.area
    X['perimeter'] = indexed.geometry.length
    X['compactness'] = 4*np.pi*X.area / X.perimeter.pow(2).clip(lower=1e-6)
    points = indexed.geometry.representative_point()
    X['centroid_x'],X['centroid_y'] = points.x,points.y
    X['Estimated_Height'] = numbers(indexed,'Estimated_Height')
    X['Reported_Floors'] = numbers(indexed,'Estimated_Floors')
    X['height_missing'] = X.Estimated_Height.isna().astype('int8')
    X['floors_missing'] = X.Reported_Floors.isna().astype('int8')
    point_quality = indexed.get('BDBIAR_Match_Quality',pd.Series('',index=indexed.index))
    year = numbers(indexed,'BDBIAR_AGE')
    X['Reported_Completion_Year'] = year.where(point_quality.eq('inside_unique') & year.between(1600,2026))
    X['completion_year_missing'] = X.Reported_Completion_Year.isna().astype('int8')
    kind = indexed.get('CSDI_Structure_Type',pd.Series('Unknown',index=indexed.index)).fillna('Unknown')
    kind = kind.where(indexed.get('CSDI_Match_Quality',pd.Series('',index=indexed.index)).eq('same_component_id_type_footprint'),'Unknown')
    for number,value in enumerate(['Tower','Podium','Open-sided Structure','Temporary Structure','Unknown']):
        X[f'csdi_structure_type_{number}'] = kind.eq(value).astype('int8')
    counts = np.zeros(len(indexed),dtype='int32')
    for start in range(0,len(indexed),10000):
        batch = indexed.iloc[start:start+10000]
        left,right = indexed.sindex.query(batch.geometry.buffer(50),predicate='intersects')
        counts[start:start+len(batch)] = np.maximum(0,np.bincount(left,minlength=len(batch))-1)
    X['neighbor_density_50m'] = counts
    zones = indexed.OZP_ZONE_LABEL.fillna('Unknown').astype(str)
    mapping = {zone:f'ozp_{i:03d}' for i,zone in enumerate(sorted(zones.unique()))}
    for zone,col in mapping.items():
        X[col] = zones.eq(zone).astype('int8')
    if any('neighbor_class' in col for col in X):
        raise AssertionError('A label-derived feature entered the design matrix')
    return X,mapping


def main():
    base = attach_geometry(read_table(AI_CALIBRATED_OUTPUT_PATH),OFFICIAL_LIB_BASE_PATH)
    X,mapping = compute_advanced_features(base)
    training = prepare_training_data(base)
    training = training.set_index(ID)
    y = pd.DataFrame({f'is_{main}':training.Final_Main_Class.eq(main).astype('int8') for main in MAIN_CLASSES},index=training.index)
    payloads = {FEATURE_X_ALL_PATH:X, FEATURE_X_TRAIN_ML_PATH:X.reindex(training.index),
        FEATURE_Y_MULTILABEL_PATH:y, FEATURE_TRAINING_DATA_ML_PATH:training,
        FEATURE_BASE_INDEXED_PATH:base.set_index(ID)}
    for path,value in payloads.items():
        with open(path,'wb') as stream:
            pickle.dump(value,stream,protocol=5)
    training.drop(columns='geometry').to_csv(Path(INTERMEDIATE_DIR)/'step8_ml_training_data.csv',encoding='utf-8-sig')
    training.groupby(['Final_Main_Class','Classification_Source']).size().rename('records').to_csv(Path(INTERMEDIATE_DIR)/'training_source_counts.csv',encoding='utf-8-sig')
    metadata = {'feature_columns':X.columns.tolist(),'ozp_mapping':mapping,'num_samples':len(X),
        'num_training_samples':len(training),'main_classes':MAIN_CLASSES,
        'training_labels':'weakly supervised; metrics measure agreement with these labels',
        'missing_values':'preserved; imputer is fitted inside each training partition in Step 9',
        'neighbor_features':'unlabelled geometry density only; no own/validation labels'}
    (Path(INTERMEDIATE_DIR)/'feature_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[INFO] Step 8: {len(X)} rows, {len(X.columns)} features, {len(training)} source-inferred training labels')
    return True


if __name__ == '__main__':
    main()
