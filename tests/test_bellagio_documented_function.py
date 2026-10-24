import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from classification import RESIDENTIAL, UNKNOWN

spec = importlib.util.spec_from_file_location('bellagio_rule_engine', ROOT / 'src' / '3.rule_engine.py')
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)


class BellagioDocumentedFunctionTests(unittest.TestCase):
    def test_named_towers_keep_broad_reported_use_without_tenure_or_measured_area(self):
        bindings = {
            '128806': '2434425301T20050430', '128809': '2438225301T20050430',
            '128868': '2430725293T20050430', '132343': '2427225281T20050430',
            '132382': '2441825274T20050608', '135985': '2442725235T20050608',
            '136868': '2443025203T20050608', '140487': '2442925162T20050608',
        }
        facts = pd.read_csv(ROOT / 'ctl' / 'building_function_evidence.csv', dtype=str, keep_default_na=False)
        facts = facts[facts.building_id.isin(bindings)].copy()
        self.assertEqual(set(facts.building_id), set(bindings))
        self.assertTrue(facts.csuid.eq(facts.building_id.map(bindings)).all())
        self.assertTrue(facts.main_class.eq(RESIDENTIAL).all())
        self.assertTrue(facts.sub_class.eq(UNKNOWN).all())
        self.assertTrue(facts.scope_limit.str.contains('not current occupancy', regex=False).all())
        frame = pd.DataFrame([{
            'BUILDINGSTRUCTUREID': bid, 'BUILDINGCSUID': csuid,
            'Final_Main_Class': UNKNOWN, 'Final_Sub_Class': UNKNOWN,
            'Classification_Source': 'Rules', 'Classification_QA': 'function_scope_ambiguous',
            'Is_Conflicted': True, 'CSDI_Match_Quality': 'same_component_id_type_footprint',
            'CSDI_Structure_Type': 'Tower', 'Estimated_Height': 160., 'Estimated_Floors': 55.,
            'GFA_Best_Available_sqm': 30000., 'Evidence_Subclass_Weights': '',
        } for bid, csuid in bindings.items()])
        physical = ['Estimated_Height', 'Estimated_Floors', 'GFA_Best_Available_sqm']
        result = rules.apply_documented_functions(frame, facts)
        self.assertTrue(result.Final_Main_Class.eq(RESIDENTIAL).all())
        self.assertTrue(result.Final_Sub_Class.eq(UNKNOWN).all())
        self.assertTrue(result.Classification_Source.eq('Documented_component_function').all())
        self.assertTrue(result.Sub_Class_Proportions.eq('').all())
        self.assertTrue(result.Observed_Residential_Area_Share.isna().all())
        self.assertTrue(result.Observed_Nondomestic_Area_Share.isna().all())
        pd.testing.assert_frame_equal(frame[physical], result[physical])


if __name__ == '__main__':
    unittest.main()
