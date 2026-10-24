import sys
from pathlib import Path
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from data_quality import ID, apply_documented_attributes, assign_floor_area, fuse_dimensions, permit_attributes


def fixture(top, floors, explicit_scope=''):
    frame = gpd.GeoDataFrame({ID: ['1'], 'BUILDINGCSUID': ['component-1'],
        'BUILDINGSTRUCTURETYPE': ['T'], 'NUMABOVEGROUNDSTOREYS': [floors],
        'TOPHEIGHT': [top], 'BASEHEIGHT': [0.], 'SHAPE_Area': [100.],
        'SHAPE_Length': [40.], 'GROSSFLOORAREA': [np.nan],
        'Component_Vertical_Scope_QA': [explicit_scope]},
        geometry=[box(835000, 815000, 835010, 815010)], crs=2326)
    return frame


def empty_matches():
    return pd.DataFrame({ID: ['1'], 'Source_Feature_ID': ['osm:none'], 'Source_Object_ID': ['osm:none'],
        'osm_area': [np.nan], 'overlap_ratio': [np.nan], 'SHAPE_Area': [100.],
        'osm_height': [np.nan], 'osm_storeys': [np.nan]})


class ReviewedDimensionAreaScopeTests(unittest.TestCase):
    def test_high_ratio_is_review_signal_and_does_not_by_itself_overwrite_reports(self):
        for top, floors, flag in [(9., 1., 'high_height_to_count_scope_review'),
                                  (14.9, 1., 'high_height_to_count_scope_review')]:
            with self.subTest(top=top, floors=floors):
                row = fuse_dimensions(fixture(top, floors), empty_matches()).iloc[0]
                self.assertEqual(row.Estimated_Height, top)
                self.assertEqual(row.Estimated_Floors, floors)
                self.assertEqual(row.Dimension_Ratio_Scope_QA, flag)
                self.assertEqual(row.GFA_Proxy_sqm, floors * 100.)

    def test_unreconciled_low_multistorey_pair_is_retained_raw_but_not_adopted(self):
        for top, floors in [(7., 5.), (15.4, 15.), (21.6, 21.), (5.99, 3.)]:
            with self.subTest(top=top, floors=floors):
                row = fuse_dimensions(fixture(top, floors), empty_matches()).iloc[0]
                self.assertTrue(pd.isna(row.Estimated_Height))
                self.assertTrue(pd.isna(row.Estimated_Floors))
                self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
                self.assertTrue(pd.isna(row.Floors_Height_Proxy))
                self.assertEqual(row.Reported_Component_Height_m, top)
                self.assertEqual(row.NUMABOVEGROUNDSTOREYS, floors)
                self.assertEqual(row.Dimension_Ratio_Scope_QA, 'low_height_to_count_scope_review')
                self.assertEqual(row.Height_Quality, 'conflict_unresolved')
                self.assertEqual(row.Floors_Source, 'component_storeys_scope_unresolved')

    def test_low_pair_screen_does_not_reject_single_levels_boundary_or_fifteen_metres(self):
        for top, floors in [(1.9, 1.), (6., 3.), (15., 5.)]:
            with self.subTest(top=top, floors=floors):
                row = fuse_dimensions(fixture(top, floors), empty_matches()).iloc[0]
                self.assertEqual(row.Estimated_Height, top)
                self.assertEqual(row.Estimated_Floors, floors)
                self.assertEqual(row.GFA_Proxy_sqm, floors * 100.)

    def test_low_pair_qa_uses_selected_count_when_comparative_count_differs_by_one(self):
        frame = fixture(5.9, 3.)
        frame['CSDI_Match_Quality'] = 'same_component_id_type_footprint'
        frame['CSDI_Reported_Storeys'] = 2.
        row = fuse_dimensions(frame, empty_matches()).iloc[0]
        self.assertEqual(row.Dimension_Ratio_Scope_QA, 'low_height_to_count_scope_review')
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.CSDI_Reported_Storeys, 2.)
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS, 3.)

    def test_explicit_vertical_scope_conflict_withholds_primary_and_dependent_area(self):
        row = fuse_dimensions(fixture(7., 5., 'unresolved'), empty_matches()).iloc[0]
        self.assertEqual(row.Estimated_Height, 7.)
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertTrue(pd.isna(row.GFA_Best_Available_sqm))
        self.assertEqual(row.Floors_Source, 'component_storeys_scope_unresolved')
        self.assertEqual(row.TOPHEIGHT, 7.)
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS, 5.)

    def test_reported_storeys_conflict_control_withholds_primary_count(self):
        frame = fuse_dimensions(fixture(122.1, 7.), empty_matches())
        frame['CSDI_Match_Quality'] = 'same_component_id_type_footprint'
        frame['CSDI_Structure_Type'] = 'Tower'
        frame['CSDI_Reported_Storeys'] = 7.
        evidence = pd.DataFrame([{'evidence_id': 'ref-1', 'building_id': '1', 'csuid': 'component-1',
            'attribute': 'reported_storeys', 'value': '52', 'unit': 'storeys',
            'scope': 'whole building reference; component scope unresolved',
            'source_url': 'https://example.org/source', 'source_date': '2026',
            'adoption': 'unresolved_component_storey_reference_conflict', 'notes': 'reference only'}])
        row = apply_documented_attributes(frame, evidence).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS, 7.)
        self.assertEqual(row.CSDI_Reported_Storeys, 7.)
        self.assertEqual(row.Floors_Source, 'documented_storey_reference_conflict_withheld')

    def test_conflicted_storeys_withhold_registered_area_and_proxy_but_keep_raw_amount(self):
        base = pd.DataFrame({'GROSSFLOORAREA': [900.], 'SHAPE_Area': [100.],
            'Estimated_Floors': [np.nan], 'Floors_Source': ['component_storeys_scope_unresolved']})
        row = assign_floor_area(base).iloc[0]
        self.assertEqual(row.GROSSFLOORAREA, 900.)
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertTrue(pd.isna(row.GFA_Best_Available_sqm))
        self.assertEqual(row.Registered_GFA_Source, 'LandsD_structure_GFA_component_scope_unresolved')

    def test_one_admin_permit_link_preserves_raw_amount_but_is_not_physical_gfa(self):
        permits = pd.DataFrame({'OPNO': ['KN1/2020'], 'DOMESTICGFA': [80.], 'NONDOMESTICGFA': [20.]})
        links = pd.DataFrame({ID: ['1'], 'OPNO': ['KN1/2020']})
        row = permit_attributes(permits, links).iloc[0]
        self.assertEqual(row.Permit_GFA_Scope, 'single_structure_component_scope_unresolved')
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertEqual(row.Permit_IDs, '["KN1/2020"]')
        self.assertEqual(row.Permit_Raw_DOMESTICGFA, 80.)
        self.assertEqual(row.GFA_DOMESTIC_SUM, 80.)
        with_area = assign_floor_area(pd.DataFrame([{**row.to_dict(), 'SHAPE_Area': 100.,
            'Estimated_Floors': 2., 'Floors_Source': 'LandsD_storeys'}])).iloc[0]
        self.assertFalse(with_area.Permit_GFA_Usable)
        self.assertTrue(pd.isna(with_area.Registered_GFA_sqm))
        self.assertEqual(with_area.GFA_Proxy_sqm, 200.)

    def test_explicit_component_scope_can_release_permit_but_admin_link_alone_cannot(self):
        base = pd.DataFrame({'Permit_GFA_Scope': ['single_structure_physical_scope_unverified'],
            'Permit_GFA_Physical_Scope_Verified': [True], 'Permit_GFA_Usable': [False],
            'GFA_DOMESTIC_SUM': [80.], 'GFA_NONDOMESTIC_SUM': [20.],
            'SHAPE_Area': [100.], 'Estimated_Floors': [2.], 'Floors_Source': ['LandsD_storeys']})
        row = assign_floor_area(base).iloc[0]
        self.assertTrue(row.Permit_GFA_Usable)
        self.assertEqual(row.Registered_GFA_sqm, 100.)

    def test_independent_registered_landsd_area_survives_unverified_permit(self):
        base = pd.DataFrame({'Permit_GFA_Scope': ['single_structure_physical_scope_unverified'],
            'Permit_GFA_Physical_Scope_Verified': [False], 'Permit_GFA_Usable': [False],
            'GFA_DOMESTIC_SUM': [80.], 'GFA_NONDOMESTIC_SUM': [20.], 'GROSSFLOORAREA': [900.],
            'SHAPE_Area': [100.], 'Estimated_Floors': [2.], 'Floors_Source': ['LandsD_storeys']})
        row = assign_floor_area(base).iloc[0]
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertEqual(row.Registered_GFA_sqm, 900.)
        self.assertEqual(row.Registered_GFA_Source, 'LandsD_structure_GFA')


if __name__ == '__main__':
    unittest.main()




