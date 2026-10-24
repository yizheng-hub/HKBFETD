"""Step 9: retrain dominant-function classifiers using a spatial holdout.

Classifier probabilities are not use-area fractions. Ambiguous predictions
remain unknown; known mixed use and non-assessed classes are preserved.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from classification import (MAIN_CLASSES, RESIDENTIAL, UNKNOWN, MIXED, NONASSESSED, taxonomy,
                            sub_label, finalize_function_fields)
from config import (FEATURE_X_ALL_PATH, FEATURE_TRAINING_DATA_ML_PATH, FEATURE_BASE_INDEXED_PATH,
    ML_MODELS_PATH, ML_FINAL_OUTPUT_PATH, OUTPUT_DIR)
from data_quality import ID, require_unique
from stage_io import save_buildings

MAIN_CONFIDENCE = .70
SUB_CONFIDENCE = .60
MIN_VALIDATION_PREDICTIONS = 30
MIN_SOURCE_LABEL_PRECISION = .80
CALIBRATION_PRECISION = .90


def validation_acceptance(probability, labels, classes, threshold):
    """Screen deployment classes on held-out source labels, not measured truth."""
    labels = np.asarray(labels)
    predicted = classes[probability.argmax(axis=1)]
    cutoffs = np.array([threshold[label] for label in predicted]) if isinstance(threshold,dict) else threshold
    accepted = probability.max(axis=1) >= cutoffs
    details, allowed = {}, []
    for parent in classes:
        mask = accepted & (predicted == parent)
        support = int(mask.sum())
        precision = float(np.mean(labels[mask] == parent)) if support else None
        usable = support >= MIN_VALIDATION_PREDICTIONS and precision >= MIN_SOURCE_LABEL_PRECISION
        details[parent] = {'accepted_predictions':support,'precision_against_source_labels':precision,
                           'eligible_for_model_inference':usable}
        if usable:
            allowed.append(parent)
    return allowed, details


def calibration_thresholds(probability, labels, classes):
    predicted = classes[probability.argmax(axis=1)]
    labels = np.asarray(labels)
    thresholds = {}
    for parent in classes:
        thresholds[parent] = 1.01  # No automatic inference if calibration support is inadequate.
        for cutoff in [.70,.80,.90,.95,.99]:
            selected = (predicted == parent) & (probability.max(axis=1) >= cutoff)
            if selected.sum() >= MIN_VALIDATION_PREDICTIONS and np.mean(labels[selected] == parent) >= CALIBRATION_PRECISION:
                thresholds[parent] = cutoff
                break
    return thresholds


def spatial_split(X,y):
    groups = (np.floor(X.centroid_x/2000).astype(int).astype(str) + '_' +
              np.floor(X.centroid_y/2000).astype(int).astype(str))
    for seed in range(42,92):
        train,val = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(X,y,groups))
        if set(y.iloc[train]) == set(MAIN_CLASSES) and set(y.iloc[val]) == set(MAIN_CLASSES):
            assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[val]))
            return train,val,groups,seed
    raise ValueError('Unable to construct a spatial holdout containing every main class')


def main_estimator():
    return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('classifier',LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=31,class_weight='balanced',
            random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True))])


def subclass_estimator():
    return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('classifier',RandomForestClassifier(n_estimators=100,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=4))])


def train_models(X,training):
    training = training.reindex(training.index.intersection(X.index))
    y = training.Final_Main_Class
    Xt = X.reindex(training.index)
    counts = y.value_counts()
    if not set(MAIN_CLASSES).issubset(counts.index) or counts.min() < 20:
        raise ValueError('Insufficient source labels to train all three main classifiers')
    train,val,groups,seed = spatial_split(Xt,y)
    inner_train,inner_val,_,inner_seed = spatial_split(Xt.iloc[train],y.iloc[train])
    calibration_model = main_estimator().fit(Xt.iloc[train].iloc[inner_train],y.iloc[train].iloc[inner_train])
    main_thresholds = calibration_thresholds(calibration_model.predict_proba(Xt.iloc[train].iloc[inner_val]),
        y.iloc[train].iloc[inner_val],calibration_model.named_steps['classifier'].classes_)
    validation_model = main_estimator().fit(Xt.iloc[train],y.iloc[train])
    predicted = validation_model.predict(Xt.iloc[val])
    validation_probability = validation_model.predict_proba(Xt.iloc[val])
    validation_classes = validation_model.named_steps['classifier'].classes_
    accepted = validation_probability.max(axis=1) >= np.array([main_thresholds[label] for label in predicted])
    validated_main, main_acceptance = validation_acceptance(validation_probability,y.iloc[val],validation_classes,main_thresholds)
    report = {'validation':'2 km spatial groups; disjoint train/holdout blocks', 'split_seed':seed,
        'label_basis':'weakly supervised source labels, not a new independent manual survey',
        'train_records':len(train),'validation_records':len(val),
        'accuracy':float(accuracy_score(y.iloc[val],predicted)),
        'balanced_accuracy':float(balanced_accuracy_score(y.iloc[val],predicted)),
        'macro_f1':float(f1_score(y.iloc[val],predicted,average='macro',zero_division=0)),
        'classification_report':classification_report(y.iloc[val],predicted,labels=MAIN_CLASSES,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y.iloc[val],predicted,labels=MAIN_CLASSES).tolist(),
        'class_order':MAIN_CLASSES,'subclass_models':{}}
    report['threshold_calibration'] = {'partition':'Spatial groups inside outer training partition only; outer holdout labels not used',
        'inner_split_seed':inner_seed,'inner_fit_records':len(inner_train),'inner_calibration_records':len(inner_val),
        'precision_target_against_source_labels':CALIBRATION_PRECISION,'minimum_selected_records':MIN_VALIDATION_PREDICTIONS,
        'thresholds_by_predicted_class':main_thresholds,'disabled_threshold':1.01}
    report['calibrated_threshold_holdout'] = {'thresholds_by_predicted_class':main_thresholds,'accepted_records':int(accepted.sum()),
        'coverage':float(accepted.mean()),'source_label_agreement':float(np.mean(predicted[accepted]==y.iloc[val].to_numpy()[accepted])) if accepted.any() else None,
        'per_predicted_class':{parent:{'accepted_predictions':int((accepted&(predicted==parent)).sum()),
            'correct_source_labels':int((accepted&(predicted==parent)&y.iloc[val].eq(parent).to_numpy()).sum()),
            'precision_against_source_labels':float(np.mean(y.iloc[val].to_numpy()[accepted&(predicted==parent)]==parent)) if (accepted&(predicted==parent)).any() else None}
            for parent in MAIN_CLASSES},'interpretation':'Thresholds selected inside training only; outer weak-label agreement, not independent field accuracy'}
    report['deployment_class_gate'] = {'minimum_accepted_holdout_predictions':MIN_VALIDATION_PREDICTIONS,
        'minimum_precision_against_source_labels':MIN_SOURCE_LABEL_PRECISION,'classes':main_acceptance,
        'interpretation':'Holdout used to screen deployable classes; these same scores are not independent post-selection accuracy'}
    split = pd.DataFrame({ID:training.index,'spatial_group':groups.to_numpy(),'partition':'train'})
    split.loc[val,'partition'] = 'holdout'
    split['calibration_partition'] = 'outer_holdout_not_used_for_calibration'
    split.loc[train,'calibration_partition'] = 'inner_fit'
    split.loc[np.asarray(train)[inner_val],'calibration_partition'] = 'inner_calibration'
    split.to_csv(Path(OUTPUT_DIR)/'model_spatial_split.csv',index=False,encoding='utf-8-sig')
    validation = pd.DataFrame({ID:training.index[val],'source_label':y.iloc[val].to_numpy(),'prediction':predicted})
    validation['maximum_probability'] = validation_probability.max(axis=1)
    validation['accepted_at_calibrated_threshold'] = accepted
    for index,parent in enumerate(validation_classes):
        validation['probability_'+parent] = validation_probability[:,index]
    validation.to_csv(Path(OUTPUT_DIR)/'model_holdout_predictions.csv',index=False,encoding='utf-8-sig')
    main_model = main_estimator().fit(Xt,y)
    subclass_models = {}
    validated_subclasses = {}
    train_ids, val_ids = set(training.index[train]), set(training.index[val])
    for parent in MAIN_CLASSES:
        targets = training[training.Final_Main_Class.eq(parent)].Final_Sub_Class.map(sub_label)
        targets = targets[targets.map(taxonomy()).eq(parent)]
        available = targets.value_counts()
        targets = targets[targets.isin(available[available >= 20].index)]
        fitting = targets.index.intersection(list(train_ids))
        holding = targets.index.intersection(list(val_ids))
        details = {'eligible_full_records':len(targets),'subclasses':targets.value_counts().to_dict(),
                   'holdout_records':len(holding),'validation_macro_f1':None}
        validated_subclasses[parent] = []
        if len(targets) < 20 or targets.nunique() < 2:
            details['status'] = 'insufficient_subclass_evidence_predictions_remain_unknown'
        else:
            if len(fitting) >= 20 and len(holding):
                check = subclass_estimator().fit(X.reindex(fitting),targets.reindex(fitting))
                guess = check.predict(X.reindex(holding))
                details['validation_macro_f1'] = float(f1_score(targets.reindex(holding),guess,average='macro',zero_division=0))
                details['validation_report'] = classification_report(targets.reindex(holding),guess,output_dict=True,zero_division=0)
                valid_children, child_acceptance = validation_acceptance(check.predict_proba(X.reindex(holding)),
                    targets.reindex(holding),check.named_steps['classifier'].classes_,SUB_CONFIDENCE)
                validated_subclasses[parent] = valid_children
                if parent == RESIDENTIAL:
                    validated_subclasses[parent] = []
                    details['publication_policy'] = 'Housing subtype requires documentary evidence; morphology does not establish tenure'
                    for decision in child_acceptance.values():
                        decision['passed_source_label_gate'] = decision['eligible_for_model_inference']
                        decision['eligible_for_model_inference'] = False
                details['deployment_subclass_gate'] = child_acceptance
            subclass_models[parent] = subclass_estimator().fit(X.reindex(targets.index),targets)
            details['status'] = 'trained_function_classifier_not_area_regressor'
        report['subclass_models'][parent] = details
    package = {'main':main_model,'subclasses':subclass_models,'features':X.columns.tolist(),
               'main_classes':MAIN_CLASSES,'main_threshold':MAIN_CONFIDENCE,'subclass_threshold':SUB_CONFIDENCE,
               'main_thresholds':main_thresholds,
               'validated_main_classes':validated_main,'validated_subclasses':validated_subclasses,
               'neighbor_class_feature':False,'label_basis':report['label_basis']}
    with open(ML_MODELS_PATH,'wb') as stream:
        pickle.dump(package,stream,protocol=5)
    (Path(OUTPUT_DIR)/'model_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return package,report


def apply_ml_calibration(base,X,package):
    result = base.copy().set_index(ID)
    result = result.reindex(X.index)
    probability = package['main'].predict_proba(X)
    classes = package['main'].named_steps['classifier'].classes_
    prediction = classes[probability.argmax(axis=1)]
    confidence = probability.max(axis=1)
    result['ML_Main_Class_Candidate'] = prediction
    for i,parent in enumerate(classes):
        result['proba_'+parent] = probability[:,i]
    result['ML_Confidence'] = confidence
    result['Calibrated_Main_Class'] = result.Final_Main_Class
    result['Calibrated_Sub_Class'] = result.Final_Sub_Class
    result['Calibrated_Source'] = result.Classification_Source
    result['Subclass_Source'] = np.where(result.Final_Sub_Class.eq(UNKNOWN),'missing',result.Classification_Source)
    result['Calibrated_Notes'] = 'Source-inferred function retained; quantitative shares require separate area evidence'
    conflict = result.get('Classification_QA',pd.Series('',index=result.index)).fillna('').str.contains(
        'function_evidence_conflict|subclass_parent_conflict|registered_area_function_conflict',regex=True)
    main_conflict = result.get('Classification_QA',pd.Series('',index=result.index)).fillna('').str.contains(
        'function_evidence_conflict|registered_area_function_conflict',regex=True)
    withhold = main_conflict & result.Calibrated_Main_Class.isin(MAIN_CLASSES)
    result.loc[withhold,['Calibrated_Main_Class','Calibrated_Sub_Class']] = UNKNOWN
    result.loc[withhold,'Subclass_Source'] = 'source_conflict_withheld'
    result.loc[withhold,'Calibrated_Notes'] = 'Function withheld because source evidence conflicts; see Classification_QA and retained source fields'
    unknown = result.Calibrated_Main_Class.eq(UNKNOWN)
    # Small, very tall records can be pylons/masts or building components.
    # A geometry-only function model cannot establish that they are occupied buildings.
    slender_unverified = (pd.to_numeric(result.get('Estimated_Height',pd.Series(np.nan,index=result.index)),errors='coerce').gt(100) &
        pd.to_numeric(result.get('SHAPE_Area',pd.Series(np.nan,index=result.index)),errors='coerce').lt(150) &
        result.get('Structure_Use_Type',pd.Series('',index=result.index)).fillna('').eq(''))
    cutoffs = np.array([package.get('main_thresholds',{}).get(label,MAIN_CONFIDENCE) for label in prediction])
    fill = unknown & (confidence >= cutoffs) & ~conflict & ~slender_unverified
    deployment_ok = np.isin(prediction,package.get('validated_main_classes',MAIN_CLASSES))
    fill &= deployment_ok
    result.loc[fill,'Calibrated_Main_Class'] = prediction[fill]
    result.loc[fill,'Calibrated_Source'] = 'Machine_Learning_v2'
    result.loc[fill,'Calibrated_Notes'] = 'Dominant function inferred by spatially validated model; not a measured use-area fraction'
    result['ML_Decision_Status'] = np.where(fill,'main_function_inferred',np.where(unknown,'low_confidence_unknown','source_function_retained'))
    result.loc[unknown & conflict,'ML_Decision_Status'] = 'unresolved_source_conflict_unknown'
    result.loc[unknown & conflict,'Calibrated_Notes'] = 'Function withheld because source evidence conflicts; see Classification_QA and retained source fields'
    result.loc[unknown & ~conflict & ~fill,'Calibrated_Notes'] = 'No supported source function and model confidence below threshold; function remains unknown'
    gated = unknown & ~conflict & (confidence >= MAIN_CONFIDENCE) & ~deployment_ok
    result.loc[gated,'ML_Decision_Status'] = 'model_class_validation_insufficient_unknown'
    result.loc[gated,'Calibrated_Notes'] = 'Predicted class lacks sufficient held-out source-label support or precision; function remains unknown'
    result.loc[unknown & slender_unverified & ~conflict,'ML_Decision_Status'] = 'slender_structure_function_unverified_unknown'
    result.loc[unknown & slender_unverified & ~conflict,'Calibrated_Notes'] = 'Small tall structure lacks occupied-building function evidence; model-only function withheld, source dimensions retained'
    result['ML_Subclass_Confidence'] = np.nan
    result['ML_Subclass_Candidate'] = UNKNOWN
    for parent,model in package['subclasses'].items():
        needs_sub = result.Calibrated_Main_Class.eq(parent) & result.Calibrated_Sub_Class.eq(UNKNOWN) & ~conflict
        needs_sub &= ~result.Calibrated_Source.eq('Documented_component_function')
        if not needs_sub.any():
            continue
        probs = model.predict_proba(X.loc[needs_sub])
        child_classes = model.named_steps['classifier'].classes_
        child_conf = probs.max(axis=1)
        result.loc[needs_sub,'ML_Subclass_Confidence'] = child_conf
        child_prediction = child_classes[probs.argmax(axis=1)]
        result.loc[needs_sub,'ML_Subclass_Candidate'] = child_prediction
        eligible = (child_conf >= SUB_CONFIDENCE) & np.isin(child_prediction,
            package.get('validated_subclasses',{}).get(parent,child_classes))
        if parent == RESIDENTIAL:
            eligible[:] = False
            result.loc[needs_sub,'Subclass_Source'] = 'housing_subtype_requires_documentary_evidence'
            result.loc[needs_sub,'Calibrated_Notes'] += '; housing subtype candidate retained separately; tenure requires a documentary source'
        selected = result.index[needs_sub][eligible]
        result.loc[selected,'Calibrated_Sub_Class'] = child_prediction[eligible]
        result.loc[selected,'Subclass_Source'] = 'Machine_Learning_v2'
        result.loc[selected,'Calibrated_Notes'] += '; subclass inferred by v2 model at recorded confidence'
    # Non-assessed and mixed-use classes are never filled from classifier ambiguity.
    protected = base.set_index(ID).Final_Main_Class.isin([NONASSESSED,MIXED])
    assert (result.loc[protected[protected].index,'Calibrated_Main_Class'] == base.set_index(ID).loc[protected[protected].index,'Final_Main_Class']).all()
    result = finalize_function_fields(result.reset_index(),'Calibrated_Main_Class','Calibrated_Sub_Class')
    # A new model fill can conflict with scoped registration even when the
    # previously unknown source class did not. Recheck after applying predictions.
    new_conflict = result.Calibrated_Main_Class.isin(MAIN_CLASSES) & result.Classification_QA.fillna('').str.contains(
        'function_evidence_conflict|registered_area_function_conflict',regex=True)
    if new_conflict.any():
        result.loc[new_conflict,['Calibrated_Main_Class','Calibrated_Sub_Class']] = UNKNOWN
        result.loc[new_conflict,'Calibrated_Source'] = 'Model_prediction_withheld_registration_conflict'
        result.loc[new_conflict,'Subclass_Source'] = 'source_conflict_withheld'
        result.loc[new_conflict,'ML_Decision_Status'] = 'unresolved_source_conflict_unknown'
        result.loc[new_conflict,'Calibrated_Notes'] = 'Model function conflicts with scoped registered use area; function and subclass withheld'
        corrected = finalize_function_fields(result.loc[new_conflict].copy(),'Calibrated_Main_Class','Calibrated_Sub_Class')
        result.loc[new_conflict,corrected.columns] = corrected
    result['Function_Quality'] = np.where(result.Calibrated_Source.str.startswith(
        ('Machine_Learning', 'LLM_', 'Inherited_context_matched_VLM'), na=False),'model_inferred',
        np.where(result.Calibrated_Main_Class.eq(UNKNOWN),'unknown','source_inferred'))
    disagreement = (result.Calibrated_Main_Class.isin(MAIN_CLASSES) &
        result.Calibrated_Main_Class.ne(result.ML_Main_Class_Candidate) & result.ML_Confidence.ge(MAIN_CONFIDENCE))
    for index in result.index[disagreement]:
        prior = result.at[index,'Classification_QA']
        flags = str(prior).split(';') if pd.notna(prior) and prior else []
        result.at[index,'Classification_QA'] = ';'.join(sorted(set(flags+['source_model_main_disagreement_review_only'])))
    result['Calibrated_Mix_Proportion'] = result.Sub_Class_Proportions
    require_unique(result)
    assert set(result[ID]) == set(base[ID])
    return result


def main():
    with open(FEATURE_X_ALL_PATH,'rb') as stream:
        X = pickle.load(stream)
    with open(FEATURE_TRAINING_DATA_ML_PATH,'rb') as stream:
        training = pickle.load(stream)
    with open(FEATURE_BASE_INDEXED_PATH,'rb') as stream:
        indexed = pickle.load(stream)
    package,report = train_models(X,training)
    result = apply_ml_calibration(indexed.drop(columns='geometry').reset_index(),X,package)
    result = indexed[['geometry']].reset_index().merge(result,on=ID,validate='one_to_one')
    import geopandas as gpd
    result = gpd.GeoDataFrame(result,geometry='geometry',crs=indexed.crs)
    save_buildings(result,ML_FINAL_OUTPUT_PATH)
    print(f"[INFO] Step 9: trained main + {len(package['subclasses'])} subclass models; spatial holdout macro-F1={report['macro_f1']:.4f}")
    return True


if __name__ == '__main__':
    main()
