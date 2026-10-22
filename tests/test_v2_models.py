import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from classification import UNKNOWN,MIXED,NONASSESSED,RESIDENTIAL,COMMERCIAL,INDUSTRIAL
spec=importlib.util.spec_from_file_location('v2_models',SRC/'9.ml_classification.py')
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class FixedMain:
    def __init__(self,probs):
        self.probs=probs
        self.classes_=np.array([RESIDENTIAL,COMMERCIAL,INDUSTRIAL])
        self.named_steps={'classifier':self}
    def predict_proba(self,X):
        return np.tile(self.probs,(len(X),1))


class ModelBehaviorTests(unittest.TestCase):
    def test_class_thresholds_require_calibration_precision_and_sample_support(self):
        classes=np.array([RESIDENTIAL,COMMERCIAL,INDUSTRIAL])
        probs=np.repeat([[.85,.1,.05],[.02,.95,.03],[.1,.8,.1],[.005,.005,.99]],
                        [40,36,20,5],axis=0)
        labels=[RESIDENTIAL]*40+[COMMERCIAL]*36+[RESIDENTIAL]*20+[INDUSTRIAL]*5
        thresholds=stage.calibration_thresholds(probs,labels,classes)
        self.assertEqual(thresholds[RESIDENTIAL],.7)
        self.assertEqual(thresholds[COMMERCIAL],.9)
        self.assertEqual(thresholds[INDUSTRIAL],1.01)
        allowed,_=stage.validation_acceptance(probs,labels,classes,thresholds)
        self.assertEqual(allowed,[RESIDENTIAL,COMMERCIAL])

    def test_calibrated_class_threshold_is_used_for_inference(self):
        index=pd.Index(['1'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'Final_Main_Class':[UNKNOWN],
            'Final_Sub_Class':[UNKNOWN],'Classification_Source':['Unknown']})
        result=stage.apply_ml_calibration(base,pd.DataFrame({'area':[50.]},index=index),
            {'main':FixedMain([.1,.85,.05]),'subclasses':{},'validated_main_classes':[COMMERCIAL],
             'main_thresholds':{COMMERCIAL:.9}})
        self.assertEqual(result.Calibrated_Main_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.ML_Decision_Status.iloc[0],'low_confidence_unknown')

    def test_new_prediction_cannot_introduce_conflict_with_registered_use_area(self):
        index=pd.Index(['1'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'Final_Main_Class':[UNKNOWN],
            'Final_Sub_Class':[UNKNOWN],'Classification_Source':['Unknown'],
            'GFA_DOMESTIC_SUM':[1000.],'GFA_NONDOMESTIC_SUM':[0.]})
        result=stage.apply_ml_calibration(base,pd.DataFrame({'area':[50.]},index=index),
            {'main':FixedMain([.005,.99,.005]),'subclasses':{}})
        self.assertEqual(result.Calibrated_Main_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.Calibrated_Sub_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.ML_Decision_Status.iloc[0],'unresolved_source_conflict_unknown')
        self.assertNotEqual(result.Calibrated_Source.iloc[0],'Machine_Learning_v2')
        self.assertEqual(result.Calibrated_Mix_Proportion.iloc[0],'')
        self.assertEqual(result.Observed_Residential_Area_Share.iloc[0],1)

    def test_slender_unverified_structure_cannot_receive_model_only_function(self):
        index=pd.Index(['1','2'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':index,'Final_Main_Class':[UNKNOWN,RESIDENTIAL],
            'Final_Sub_Class':[UNKNOWN,UNKNOWN],'Classification_Source':['Unknown','OSM_Building_Function'],
            'Estimated_Height':[120.,120.],'SHAPE_Area':[40.,40.],'Structure_Use_Type':['','']})
        result=stage.apply_ml_calibration(base,pd.DataFrame({'area':[40.,40.]},index=index),
            {'main':FixedMain([.99,.005,.005]),'subclasses':{},'validated_main_classes':[RESIDENTIAL]})
        self.assertEqual(result.Calibrated_Main_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.ML_Decision_Status.iloc[0],'slender_structure_function_unverified_unknown')
        self.assertEqual(result.Calibrated_Main_Class.iloc[1],RESIDENTIAL)

    def test_validation_gate_requires_support_and_source_label_precision(self):
        classes=np.array([RESIDENTIAL,COMMERCIAL,INDUSTRIAL])
        probs=np.repeat([[.95,.03,.02],[.02,.95,.03],[.02,.03,.95]], [40,40,5],axis=0)
        labels=[RESIDENTIAL]*40+[RESIDENTIAL]*20+[COMMERCIAL]*20+[INDUSTRIAL]*5
        allowed,details=stage.validation_acceptance(probs,labels,classes,.7)
        self.assertEqual(allowed,[RESIDENTIAL])
        self.assertFalse(details[COMMERCIAL]['eligible_for_model_inference'])
        self.assertFalse(details[INDUSTRIAL]['eligible_for_model_inference'])

    def test_high_probability_class_blocked_when_validation_insufficient(self):
        index=pd.Index(['1'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'Final_Main_Class':[UNKNOWN],
            'Final_Sub_Class':[UNKNOWN],'Classification_Source':['source']})
        result=stage.apply_ml_calibration(base,pd.DataFrame({'area':[50.]},index=index),
            {'main':FixedMain([.005,.005,.99]),'subclasses':{},'validated_main_classes':[RESIDENTIAL],
             'main_thresholds':{INDUSTRIAL:1.01}})
        self.assertEqual(result.Calibrated_Main_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.ML_Decision_Status.iloc[0],'model_class_validation_insufficient_unknown')

    def test_spatial_holdout_uses_disjoint_blocks(self):
        index=pd.Index([str(i) for i in range(90)],name='BUILDINGSTRUCTUREID')
        X=pd.DataFrame({'centroid_x':np.repeat(np.arange(30)*3000,3),
                        'centroid_y':np.tile([0,20,40],30)},index=index)
        y=pd.Series([RESIDENTIAL,COMMERCIAL,INDUSTRIAL]*30,index=index)
        train,val,groups,seed=stage.spatial_split(X,y)
        self.assertTrue(set(groups.iloc[train]).isdisjoint(set(groups.iloc[val])))
        self.assertEqual(set(y.iloc[val]),{RESIDENTIAL,COMMERCIAL,INDUSTRIAL})

    def test_imputer_uses_training_values_only(self):
        X=pd.DataFrame({'height':[10.,np.nan,30.,np.nan],'area':[1.,2.,3.,4.]})
        y=pd.Series([RESIDENTIAL,COMMERCIAL,RESIDENTIAL,COMMERCIAL])
        model=stage.main_estimator().fit(X,y)
        self.assertEqual(model.named_steps['imputer'].statistics_[0],20.)
        self.assertEqual(model.named_steps['imputer'].transform(pd.DataFrame({'height':[1e9],'area':[1.]}))[0,0],1e9)
        self.assertTrue(pd.isna(X.height.iloc[1]))

    def calibrate(self,probs,classes):
        index=pd.Index([str(i) for i in range(len(classes))],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':index,'Final_Main_Class':classes,
            'Final_Sub_Class':[UNKNOWN]*len(classes),'Classification_Source':['source']*len(classes)})
        return stage.apply_ml_calibration(base,pd.DataFrame({'area':[50.]*len(classes)},index=index),
            {'main':FixedMain(probs),'subclasses':{}})

    def test_uncertainty_remains_unknown_instead_of_noise_or_mixed(self):
        result=self.calibrate([.5,.49,.01],[UNKNOWN])
        self.assertEqual(result.Calibrated_Main_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.ML_Decision_Status.iloc[0],'low_confidence_unknown')
        self.assertEqual(result.Calibrated_Mix_Proportion.iloc[0],'')

    def test_known_mixed_and_nonassessed_are_protected(self):
        result=self.calibrate([.99,.005,.005],[MIXED,NONASSESSED])
        self.assertEqual(result.Calibrated_Main_Class.tolist(),[MIXED,NONASSESSED])

    def test_main_prediction_does_not_invent_a_subclass_or_area_fraction(self):
        result=self.calibrate([.99,.005,.005],[UNKNOWN])
        self.assertEqual(result.Calibrated_Main_Class.iloc[0],RESIDENTIAL)
        self.assertEqual(result.Calibrated_Sub_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.Calibrated_Mix_Proportion.iloc[0],'')

    def test_model_cannot_fill_unresolved_source_conflict(self):
        index=pd.Index(['1'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'Final_Main_Class':[UNKNOWN],
            'Final_Sub_Class':[UNKNOWN],'Classification_Source':['Conflicting_sources'],
            'Classification_QA':['function_evidence_conflict']})
        result=stage.apply_ml_calibration(base,pd.DataFrame({'area':[50.]},index=index),
            {'main':FixedMain([.99,.005,.005]),'subclasses':{}})
        self.assertEqual(result.Calibrated_Main_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.ML_Decision_Status.iloc[0],'unresolved_source_conflict_unknown')

    def test_conflicting_source_class_is_withheld_even_before_model_filling(self):
        index=pd.Index(['1'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'Final_Main_Class':[COMMERCIAL],
            'Final_Sub_Class':['Office_办公室'],'Classification_Source':['source'],
            'Classification_QA':['registered_area_function_conflict'],
            'GFA_DOMESTIC_SUM':[1000.],'GFA_NONDOMESTIC_SUM':[0.]})
        result=stage.apply_ml_calibration(base,pd.DataFrame({'area':[50.]},index=index),
            {'main':FixedMain([.005,.99,.005]),'subclasses':{}})
        self.assertEqual(result.Calibrated_Main_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.Calibrated_Sub_Class.iloc[0],UNKNOWN)
        self.assertEqual(result.ML_Decision_Status.iloc[0],'unresolved_source_conflict_unknown')


if __name__=='__main__':unittest.main()
