import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from classification import COMMERCIAL, MIXED, RESIDENTIAL, UNKNOWN, enrich_function_evidence, finalize_function_fields, format_weights


class UnknownFunctionComponentsTests(unittest.TestCase):
    def test_unknown_does_not_publish_old_weights_as_current_components(self):
        frame = pd.DataFrame({'Final_Main_Class': [UNKNOWN], 'Final_Sub_Class': [UNKNOWN],
            'Evidence_Subclass_Weights': [format_weights({COMMERCIAL: 1., 'Other Commercial_其他商业': 1.})],
            'Classification_QA': ['function_evidence_conflict']})
        row = finalize_function_fields(frame).iloc[0]
        self.assertEqual(json.loads(row.Functional_Main_Classes), [])
        self.assertEqual(json.loads(row.Functional_Subclasses), [])
        self.assertEqual(row.Sub_Class_Proportions, '')

    def test_unresolved_component_preserves_history_and_clears_current_weights(self):
        old_weights = 'Education_教育:1.00000000'
        old_mains = json.dumps([COMMERCIAL], ensure_ascii=False)
        old_subs = json.dumps(['Education_教育'], ensure_ascii=False)
        base = pd.DataFrame({'BUILDINGSTRUCTUREID': ['137654'],
            'Final_Main_Class': [COMMERCIAL], 'Final_Sub_Class': ['Education_教育'],
            'Classification_Source': ['Keyword'], 'Classification_QA': [''], 'Is_Conflicted': [False],
            'Sub_Class_Proportions': [old_weights], 'Evidence_Subclass_Weights': [old_weights],
            'Functional_Main_Classes': [old_mains], 'Functional_Subclasses': [old_subs],
            'CSDI_Building_Name_EN': ['Student Residence City University of Hong Kong Moon Chun Multi-function Hall'],
            'CSDI_Building_Name_TC': ['城市大學學生宿舍滿珍綜合禮堂'],
            'CSDI_Match_Quality': ['same_component_id_type_footprint'], 'CSDI_Status': ['Active'],
            'CSDI_Geometry_IoU': [1.]})
        osm = pd.DataFrame(columns=['BUILDINGSTRUCTUREID', 'Source_Object_ID', 'osm_area', 'SHAPE_Area', 'overlap_ratio', 'building'])
        row = enrich_function_evidence(base, osm).iloc[0]
        self.assertEqual(row.Final_Main_Class, UNKNOWN)
        self.assertEqual(row.Evidence_Subclass_Weights, '')
        self.assertEqual(json.loads(row.Functional_Main_Classes), [])
        history = json.loads(row.CSDI_Name_Function_Evidence)['before']
        self.assertEqual(history['evidence_subclass_weights'], old_weights)
        self.assertEqual(history['functional_main_classes'], old_mains)
        self.assertEqual(history['functional_subclasses'], old_subs)

    def test_supported_mixed_components_are_retained_without_area_claim(self):
        frame = pd.DataFrame({'Final_Main_Class': [MIXED], 'Final_Sub_Class': [UNKNOWN],
            'Evidence_Subclass_Weights': [format_weights({'Other Housing_其他房屋': .5, 'Office_办公室': .5})]})
        row = finalize_function_fields(frame).iloc[0]
        self.assertEqual(set(json.loads(row.Functional_Main_Classes)), {RESIDENTIAL, COMMERCIAL})
        self.assertEqual(set(json.loads(row.Functional_Subclasses)), {'Other Housing_其他房屋', 'Office_办公室'})
        self.assertEqual(row.Observed_Main_Area_Proportions, '')

    def test_adopted_component_name_replaces_current_weights_and_preserves_prior(self):
        for name, use, old_sub, expected in [
            ('Example School', '', 'Other Commercial_其他商业', 'Education_教育'),
            ('Example School Substation', 'substation', 'Education_教育', 'Other Commercial_其他商业'),
        ]:
            with self.subTest(name=name):
                old_weights = format_weights({old_sub: 1.})
                base = pd.DataFrame({'BUILDINGSTRUCTUREID': ['one'], 'Final_Main_Class': [COMMERCIAL],
                    'Final_Sub_Class': [old_sub], 'Classification_Source': ['Keyword'],
                    'Sub_Class_Proportions': [old_weights], 'Evidence_Subclass_Weights': [old_weights],
                    'CSDI_Building_Name_EN': [name], 'CSDI_Building_Name_TC': [''],
                    'CSDI_Component_Use_Type': [use], 'CSDI_Match_Quality': ['same_component_id_type_footprint'],
                    'CSDI_Status': ['Active'], 'CSDI_Geometry_IoU': [1.]})
                osm = pd.DataFrame(columns=['BUILDINGSTRUCTUREID', 'Source_Object_ID', 'osm_area', 'SHAPE_Area', 'overlap_ratio', 'building'])
                row = enrich_function_evidence(base, osm).iloc[0]
                self.assertEqual(row.Evidence_Subclass_Weights, format_weights({expected: 1.}))
                self.assertEqual(json.loads(row.CSDI_Name_Function_Evidence)['before']['evidence_subclass_weights'], old_weights)


if __name__ == '__main__':
    unittest.main()
