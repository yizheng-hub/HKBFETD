from pathlib import Path
import os
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativeRealFinalizerTests(unittest.TestCase):
    def test_real_finalizer_preserves_numeric_areas_and_recomputed_vectors_are_idempotent(self):
        program = r"""
import pandas as pd
from classification import COMMERCIAL, UNKNOWN, finalize_function_fields, taxonomy
sub = next(label for label, parent in taxonomy().items() if parent == COMMERCIAL and 'Office' in label)
frame = pd.DataFrame([{
    'BUILDINGSTRUCTUREID':'synthetic-test-only', 'Final_Main_Class':COMMERCIAL, 'Final_Sub_Class':sub,
    'Classification_Source':'LLM_native_interactive', 'GFA_DOMESTIC_SUM':100.0, 'GFA_NONDOMESTIC_SUM':50.0,
    'GFA_Best_Available_sqm':150.0, 'HEIGHT':16.7, 'FOOTPRINT':525.0,
    'Evidence_Subclass_Weights':'Education_教育:1', 'Sub_Class_Proportions':'stale_old_vector',
    'Observed_Main_Area_Proportions':'stale-observation', 'Observed_Residential_Area_Share':0.4,
    'Observed_Nondomestic_Area_Share':0.6, 'Main_Class_Proportions':'stale-main-vector',
    'Classification_QA':'', 'Is_Conflicted':False,
}])
first = finalize_function_fields(frame)
second = finalize_function_fields(first)
assert first.loc[0,'Sub_Class_Proportions'] == f'{sub}:1.00000000'
assert first.loc[0,'Proportion_Method'] == 'single_function_assumption'
assert first.loc[0,'Proportion_Quality'] == 'assumed_not_measured'
assert pd.isna(first.loc[0,'Observed_Residential_Area_Share'])
assert pd.isna(first.loc[0,'Observed_Nondomestic_Area_Share'])
assert first.loc[0,'GFA_DOMESTIC_SUM'] == 100.0
assert first.loc[0,'GFA_NONDOMESTIC_SUM'] == 50.0
assert first.loc[0,'GFA_Best_Available_sqm'] == 150.0
pd.testing.assert_frame_equal(first, second)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run([sys.executable, "-B", "-c", program], cwd=ROOT,
                                   env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_real_step5_merge_then_finalize_is_idempotent_with_physical_fields(self):
        program = r"""
import importlib.util
from pathlib import Path
import pandas as pd
from classification import COMMERCIAL, UNKNOWN, taxonomy
spec=importlib.util.spec_from_file_location('native_step5', Path('src/5.evaluation_and_merge.py'))
step5=importlib.util.module_from_spec(spec); spec.loader.exec_module(step5)
sub=next(label for label,parent in taxonomy().items() if parent==COMMERCIAL and 'Office' in label)
base=pd.DataFrame([{'BUILDINGSTRUCTUREID':'synthetic-merge-finalizer','Final_Main_Class':UNKNOWN,'Final_Sub_Class':UNKNOWN,
 'Classification_Source':'Rules','HEIGHT':18.25,'GFA_DOMESTIC_SUM':100.0,'GFA_NONDOMESTIC_SUM':50.0,
 'GFA_Best_Available_sqm':150.0,'FOOTPRINT':525.0,'Evidence_Subclass_Weights':'Education_教育:1',
 'Sub_Class_Proportions':'stale-vector','Observed_Main_Area_Proportions':'stale-observation',
 'Observed_Residential_Area_Share':0.4,'Observed_Nondomestic_Area_Share':0.6}])
native={'synthetic-merge-finalizer':{'status':'matching_native_checkpoint','adoptable':True,'main':COMMERCIAL,'sub':sub,'confidence':.91}}
first=step5.merge_verified_results(base,pd.DataFrame(),lambda row:'unused',native_results=native)
second=step5.finalize_function_fields(first)
assert first.loc[0,'Final_Main_Class']==COMMERCIAL and first.loc[0,'Final_Sub_Class']==sub
assert first.loc[0,'HEIGHT']==18.25 and first.loc[0,'GFA_DOMESTIC_SUM']==100.0
assert first.loc[0,'GFA_NONDOMESTIC_SUM']==50.0 and first.loc[0,'GFA_Best_Available_sqm']==150.0
assert first.loc[0,'Proportion_Method']=='single_function_assumption'
assert first.loc[0,'Proportion_Quality']=='assumed_not_measured'
assert pd.isna(first.loc[0,'Observed_Residential_Area_Share']) and pd.isna(first.loc[0,'Observed_Nondomestic_Area_Share'])
pd.testing.assert_frame_equal(first,second)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run([sys.executable, "-B", "-c", program], cwd=ROOT,
                                   env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

if __name__ == "__main__":
    unittest.main()
