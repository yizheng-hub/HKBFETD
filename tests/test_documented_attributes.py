import json
from pathlib import Path
import sys
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_quality import ID, apply_documented_attributes, attach_csdi_dimensions


def component():
    return gpd.GeoDataFrame({ID:['1'], 'BUILDINGCSUID':['tower-1'],
        'CSDI_Match_Quality':['same_component_id_type_footprint'], 'CSDI_Structure_Type':['Tower'],
        'Estimated_Height':[128.4], 'Estimated_Floors':[np.nan], 'SHAPE_Area':[800.],
        'Floors_Source':['conflicting_reported_storeys'], 'Floors_Quality':['conflict_unresolved'],
        'Floors_Height_Proxy':[np.nan], 'Floor_Height_Proxy_Assumption_m':[np.nan]},
        geometry=[box(835000,815000,835010,815080)], crs=2326)


def facts(**changes):
    row={'evidence_id':'owner-1','building_id':'1','csuid':'tower-1',
        'attribute':'aboveground_storeys','value':'32','unit':'storeys',
        'scope':'tower_component_aboveground','source_url':'https://example.org/report.pdf',
        'source_date':'2022','adoption':'component_storeys_supported','notes':'Includes mechanical/refuge floors; basement separate'}
    return pd.DataFrame([{**row,**changes}])


class DocumentedAttributesTests(unittest.TestCase):
    def test_storey_reference_conflict_withholds_count_and_proxy_but_retains_raw_evidence(self):
        frame=component()
        frame['Estimated_Floors']=25.
        frame['CSDI_Reported_Storeys']=25.
        frame['Floors_Source']='CSDI_reported_storeys'
        frame['Floors_Height_Proxy']=43.
        frame['Floor_Height_Proxy_Assumption_m']=3.
        row=apply_documented_attributes(frame,facts(value='35',scope='tower count reference unresolved',
            adoption='unresolved_component_storey_reference_conflict')).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertTrue(pd.isna(row.GFA_Best_Available_sqm))
        self.assertTrue(pd.isna(row.Floors_Height_Proxy))
        self.assertEqual(row.CSDI_Reported_Storeys,25.)
        self.assertEqual(row.Floors_Before_Document_Review,25.)
        self.assertEqual(row.Floors_Source,'documented_storey_reference_conflict_withheld')
        self.assertEqual(row.Floors_Quality,'conflict_unresolved')
        self.assertIn('documented_storey_reference_conflict',row.Floors_Source_Comparison_QA)
        self.assertEqual(json.loads(row.Reported_Attribute_Evidence)[0]['value'],'35')

    def test_unresolved_count_cannot_silently_override_selected_document_count(self):
        rows=pd.concat([facts(),facts(evidence_id='other-reference',value='35',
            adoption='unresolved_component_storey_reference_conflict')],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'source reconciliation'):
            apply_documented_attributes(component(),rows)

    def test_withholding_count_preserves_independent_registered_area(self):
        frame=component()
        frame['Estimated_Floors']=25.
        frame['GROSSFLOORAREA']=55000.
        row=apply_documented_attributes(frame,facts(value='35',
            adoption='unresolved_component_storey_reference_conflict')).iloc[0]
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertEqual(row.GFA_Best_Available_sqm,55000.)
        self.assertEqual(row.GFA_Quality,'reported_structure_gfa_not_independently_validated')

    def test_unresolved_documented_height_reference_withholds_primary_and_retains_source_value(self):
        frame=component()
        frame['Estimated_Floors']=32.
        frame['Floors_Height_Proxy']=43.
        frame['Floor_Height_Proxy_Assumption_m']=3.
        frame['Ground_To_Top_Height_m']=132.6
        frame['Ground_Height_Quality']='terrain_point_estimate'
        row=apply_documented_attributes(frame,facts(attribute='height',value='374',unit='m',
            scope='main building including mast',adoption='unresolved_height_reference_conflict')).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertEqual(row.Height_Before_Document_Review,128.4)
        self.assertEqual(row.Height_Quality,'conflict_unresolved')
        self.assertIn('documented_height_reference_conflict',row.Height_Source_Comparison_QA)
        self.assertEqual(row.Height_Reference,'unresolved_documented_reference')
        self.assertTrue(pd.isna(row.Floors_Height_Proxy))
        self.assertTrue(pd.isna(row.Floor_Height_Proxy_Assumption_m))
        self.assertEqual(row.Ground_To_Top_Height_m,132.6)
        self.assertEqual(row.Ground_Height_Quality,'documented_reference_unresolved_estimate')
        self.assertEqual(row.Estimated_Floors,32.)
        self.assertEqual(row.GFA_Proxy_sqm,25600.)

    def test_explicit_count_resolves_legacy_conflict_and_recomputes_area(self):
        row=apply_documented_attributes(component(),facts()).iloc[0]
        self.assertEqual(row.Estimated_Floors,32)
        self.assertEqual(row.GFA_Proxy_sqm,25600)
        self.assertEqual(row.Floors_Source,'Documented_component_storeys')
        self.assertEqual(row.Floors_Source_Before_Document_Review,'conflicting_reported_storeys')
        self.assertEqual(json.loads(row.Floors_Document_Evidence_IDs),['owner-1'])
        self.assertTrue(pd.isna(row.Floors_Before_Document_Review))

    def test_source_scoped_levels_or_areas_do_not_donate_storeys_or_gfa(self):
        for attribute,unit,value in [('floor_labels','label','34/F'),('office_floors','floors','34'),
                                     ('rentable_area','sqft','217955'),('height','m','374')]:
            with self.subTest(attribute=attribute):
                row=apply_documented_attributes(component(),facts(attribute=attribute,unit=unit,value=value,
                    scope='whole_building',adoption='source_scope_only')).iloc[0]
                self.assertTrue(pd.isna(row.Estimated_Floors))
                self.assertTrue(pd.isna(row.GFA_Best_Available_sqm))
                self.assertEqual(json.loads(row.Reported_Attribute_Evidence)[0]['value'],value)

    def test_component_id_binding_cannot_be_transferred_to_another_asset(self):
        for changes in [{'building_id':'2'},{'csuid':'tower-2'}]:
            with self.assertRaises(ValueError):
                apply_documented_attributes(component(),facts(**changes))

    def test_aboveground_count_cannot_include_podium_or_basement_without_scope(self):
        for scope in ['whole_building','development','including_basement','podium']:
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError,'physical-count scope'):
                apply_documented_attributes(component(),facts(scope=scope))

    def test_count_labels_and_noninteger_counts_are_rejected(self):
        for value in ['32/F','32.5','0','unknown']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'Invalid documented'):
                apply_documented_attributes(component(),facts(value=value))

    def test_a_tower_document_cannot_donate_to_changed_geometry_or_podium(self):
        for column,value in [('CSDI_Match_Quality','footprint_changed'),('CSDI_Structure_Type','Podium')]:
            frame=component()
            frame[column]=value
            with self.assertRaisesRegex(ValueError,'matched tower component'):
                apply_documented_attributes(frame,facts())

    def test_conflicting_documents_require_reconciliation_instead_of_max_or_first(self):
        rows=pd.concat([facts(),facts(evidence_id='owner-2',value='40')],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'Conflicting documented'):
            apply_documented_attributes(component(),rows)

    def test_documented_storeys_cannot_bypass_physical_ratio_guard(self):
        with self.assertRaisesRegex(ValueError,'contradict the component height'):
            apply_documented_attributes(component(),facts(value='1'))

    def test_selected_reported_count_removes_height_proxy(self):
        frame=component()
        frame['Floors_Height_Proxy']=43.
        frame['Floor_Height_Proxy_Assumption_m']=3.
        row=apply_documented_attributes(frame,facts()).iloc[0]
        self.assertTrue(pd.isna(row.Floors_Height_Proxy))
        self.assertTrue(pd.isna(row.Floor_Height_Proxy_Assumption_m))

    def test_documented_gfa_is_distinct_from_registered_area_and_floor_proxy(self):
        row=apply_documented_attributes(component(),facts(attribute='gross_floor_area',value='57000',
            unit='m2',scope='tower_component_gfa',adoption='component_gfa_supported')).iloc[0]
        self.assertEqual(row.Documented_GFA_sqm,57000)
        self.assertEqual(row.GFA_Best_Available_sqm,57000)
        self.assertEqual(row.GFA_Quality,'documented_component_gfa')
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))

    def test_rentable_or_internal_area_cannot_be_adopted_as_component_gfa(self):
        for attribute,scope in [('rentable_area','tower_component_gfa'),('gross_floor_area','development'),
                                ('internal_floor_area','tower_component_gfa')]:
            with self.subTest(attribute=attribute,scope=scope), self.assertRaisesRegex(ValueError,'gross-floor-area scope'):
                apply_documented_attributes(component(),facts(attribute=attribute,value='57000',unit='m2',
                    scope=scope,adoption='component_gfa_supported'))

    def test_component_documented_gfa_supersedes_unresolved_raw_registration_scope(self):
        frame=component()
        frame['Floors_Source']='component_storeys_scope_unresolved'
        frame['GROSSFLOORAREA']=58000.
        row=apply_documented_attributes(frame,facts(attribute='gross_floor_area',value='57000',
            unit='m2',scope='tower_component_gfa',adoption='component_gfa_supported')).iloc[0]
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertEqual(row.Registered_GFA_Source,'LandsD_structure_GFA_component_scope_unresolved')
        self.assertEqual(row.Documented_GFA_sqm,57000)
        self.assertEqual(row.GFA_Best_Available_sqm,57000)
        self.assertEqual(row.GFA_Quality,'documented_component_gfa')
        self.assertEqual(row.GROSSFLOORAREA,58000)

    def test_record_dates_are_preserved_without_becoming_completion_years(self):
        frame=component()
        frame['BUILDINGSTRUCTURETYPE']='T'
        source=gpd.GeoDataFrame({'BuildingCSUID':['tower-1'],'BaseHeight':[4.2],
            'TopHeight':[132.6],'Storeys':[32.], 'StoreysInBasement':[2.],
            'BuildingBlockType':['Tower'],'Status':['Active'],'BuildingID':[1234567890],
            'DateCreate':['20050430'],'DateStamp':['20261020']},geometry=frame.geometry,crs=2326)
        row=attach_csdi_dimensions(frame,source).iloc[0]
        self.assertEqual(row.CSDI_Record_Created_Date,'20050430')
        self.assertEqual(row.CSDI_Record_Updated_Date,'20261020')
        self.assertNotIn('Year_Built',row.index)


if __name__=='__main__':
    unittest.main()
