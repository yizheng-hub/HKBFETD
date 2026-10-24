import importlib.util
from pathlib import Path
import sys
import unittest
import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
spec=importlib.util.spec_from_file_location('training_function_area_scope',SRC/'8.feature_engineering.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

AREA_FLAGS=['permit_accounting_not_observed_use','permit_physical_scope_unverified',
            'permit_component_scope_unresolved','registered_gfa_component_scope_unresolved']

def records(qa,source='Documented_component_function'):
    return pd.DataFrame({'BUILDINGSTRUCTUREID':[str(i) for i in range(len(qa))],
        'CATEGORY':['9']*len(qa),'Classification_Source':[source]*len(qa),
        'Final_Main_Class':['Commercial_商业类别']*len(qa),
        'BDBIAR_Match_Quality':['unmatched']*len(qa),
        'Classification_QA':qa,'Is_Conflicted':[False]*len(qa)})

class FunctionLabelAreaScopeTests(unittest.TestCase):
    def test_area_uncertainty_keeps_independently_supported_function_labels(self):
        flags=AREA_FLAGS+[';'.join(AREA_FLAGS)]
        for source in ['Documented_component_function','OSM_Building_Function']:
            with self.subTest(source=source):
                frame=records(flags,source)
                trained=module.prepare_training_data(frame)
                self.assertEqual(trained.BUILDINGSTRUCTUREID.tolist(),frame.BUILDINGSTRUCTUREID.tolist())
                self.assertEqual(trained.Classification_QA.tolist(),flags)
                self.assertTrue(trained.Training_Label_Basis.eq('weak_supervision_source_inferred_not_independent_ground_truth').all())

    def test_function_conflicts_and_absent_sources_still_block_training(self):
        for flag in ['function_scope_ambiguous','function_evidence_conflict',
                     'subclass_parent_conflict','registered_area_function_conflict']:
            with self.subTest(flag=flag):
                self.assertTrue(module.prepare_training_data(records([AREA_FLAGS[0]+';'+flag])).empty)
        conflicted=records([AREA_FLAGS[0]])
        conflicted.loc[0,'Is_Conflicted']=True
        self.assertTrue(module.prepare_training_data(conflicted).empty)
        self.assertTrue(module.prepare_training_data(records([AREA_FLAGS[0]],'Unknown')).empty)
        unknown=records([AREA_FLAGS[0]])
        unknown.loc[0,'Final_Main_Class']='Unknown_未知类别'
        self.assertTrue(module.prepare_training_data(unknown).empty)

if __name__=='__main__':unittest.main()
