import sys
from pathlib import Path
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from data_quality import ID, dimension_matches, fuse_dimensions


def components(ids=('1', '2'), floors=None):
    floors = list(floors) if floors is not None else [np.nan] * len(ids)
    if len(floors) != len(ids):
        raise ValueError('floor fixture length must match ids')
    return gpd.GeoDataFrame({ID: list(ids), 'NUMABOVEGROUNDSTOREYS': list(floors),
        'TOPHEIGHT': [np.nan] * len(ids), 'BASEHEIGHT': [np.nan] * len(ids),
        'SHAPE_Area': [100.] * len(ids)},
        geometry=[box(i * 20, 0, i * 20 + 10, 10) for i in range(len(ids))], crs=2326)


def osm_rows(ids=('1', '2'), overlaps=(.8, .02), *, object_id='osm:way:shared', height=20., storeys=3.):
    return pd.DataFrame({ID: list(ids), 'Source_Object_ID': [object_id] * len(ids),
        'Source_Feature_ID': [object_id + ':geometry'] * len(ids),
        'osm_area': [120.] * len(ids), 'overlap_ratio': list(overlaps),
        'SHAPE_Area': [100.] * len(ids), 'osm_height': [height] * len(ids),
        'osm_storeys': [storeys] * len(ids)})


class OSMDimensionScopeTests(unittest.TestCase):
    def test_source_object_shared_before_dimension_threshold_is_only_a_candidate(self):
        # The second component's intersection is 2.4 sqm: it is raw evidence of
        # shared scope, even though it fails the donation overlap thresholds.
        aggregated = osm_rows()
        matches = dimension_matches(aggregated).set_index(ID)
        self.assertEqual(matches.loc['1', 'OSM_Dimension_Scope_QA'], 'shared_component_scope')
        self.assertEqual(matches.loc['1', 'OSM_Dimension_Raw_Matched_Component_Count'], 2)
        self.assertEqual(matches.loc['1', 'OSM_Storeys'], 3.)
        row = fuse_dimensions(components(), aggregated).set_index(ID).loc['1']
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.Ground_To_Top_Height_m))
        self.assertEqual(row.OSM_Storeys_Scope_QA, 'shared_component_scope_not_used_for_component')

    def test_missing_source_object_id_is_unknown_not_single_component(self):
        aggregated = osm_rows(ids=('1',), overlaps=(1.,))
        aggregated['Source_Object_ID'] = np.nan
        row = fuse_dimensions(components(ids=('1',)), aggregated).iloc[0]
        self.assertEqual(row.OSM_Dimension_Scope_QA, 'source_object_scope_unknown')
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.OSM_Storeys_Scope_QA, 'source_object_scope_unknown_not_used_for_component')
        self.assertEqual(row.OSM_Storeys, 3.)

    def test_single_component_osm_dimension_remains_eligible(self):
        aggregated = osm_rows(ids=('1',), overlaps=(1.,), object_id='osm:way:unique')
        frame = components(ids=('1',))
        frame['BASEHEIGHT'] = 0.
        point = frame.to_crs(4326).geometry.representative_point().iloc[0]
        terrain = pd.DataFrame({ID: ['1'], 'longitude': [point.x], 'latitude': [point.y],
            'dtm_elevation_mpd': [0.]})
        row = fuse_dimensions(frame, aggregated, terrain).iloc[0]
        self.assertEqual(row.OSM_Dimension_Scope_QA, 'single_component_scope')
        self.assertEqual(row.Estimated_Floors, 3.)
        self.assertEqual(row.Ground_To_Top_Height_m, 20.)

    def test_shared_osm_does_not_withhold_independent_native_storeys(self):
        aggregated = osm_rows()
        row = fuse_dimensions(components(ids=('1',), floors=(4.,)), aggregated).iloc[0]
        self.assertEqual(row.Estimated_Floors, 4.)
        self.assertEqual(row.Floors_Source, 'LandsD_storeys')

    def test_shared_osm_height_cannot_conflict_with_or_replace_terrain_height(self):
        aggregated = osm_rows(height=200., storeys=3.)
        frame = components(ids=('1',))
        frame['TOPHEIGHT'] = 100.
        frame['BASEHEIGHT'] = 0.
        point = frame.to_crs(4326).geometry.representative_point().iloc[0]
        terrain = pd.DataFrame({ID: ['1'], 'longitude': [point.x], 'latitude': [point.y],
            'dtm_elevation_mpd': [0.]})
        row = fuse_dimensions(frame, aggregated, terrain).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.OSM_Storeys_Scope_QA, 'shared_component_scope_not_used_for_component')
        self.assertEqual(row.Ground_To_Top_Height_m, 100.)
        self.assertEqual(row.Ground_Height_Quality, 'terrain_point_estimate')
        self.assertNotIn('OSM_differs', row.Height_Source_Comparison_QA)

    def test_candidate_evidence_keeps_metrics_and_raw_values(self):
        row = dimension_matches(osm_rows()).iloc[0]
        evidence = __import__('json').loads(row.OSM_Dimension_Candidate_Evidence)
        self.assertEqual(evidence[0]['source_object_id'], 'osm:way:shared')
        self.assertEqual(evidence[0]['raw_matched_component_count'], 2)
        self.assertAlmostEqual(evidence[0]['intersection_area_sqm'], 96.)
        self.assertEqual(evidence[0]['reported_storeys'], 3.)

    def test_raw_count_from_another_component_does_not_claim_candidate_is_single(self):
        # The candidate itself is below the 1 sqm raw-intersection gate; a
        # different component has the one qualifying intersection for this ID.
        aggregated = pd.DataFrame({ID: ['1', '2'],
            'Source_Object_ID': ['osm:way:mixed', 'osm:way:mixed'],
            'Source_Feature_ID': ['osm:way:mixed:geometry', 'osm:way:mixed:geometry'],
            'osm_area': [1.2, 1.2], 'overlap_ratio': [.6, 1.],
            'SHAPE_Area': [.8, 10.], 'osm_height': [np.nan, np.nan],
            'osm_storeys': [3., 3.]})
        row = dimension_matches(aggregated).set_index(ID).loc['1']
        self.assertTrue(pd.isna(row.OSM_Dimension_Raw_Matched_Component_Count))
        self.assertEqual(row.OSM_Dimension_Scope_QA, 'source_object_scope_unknown')
        evidence = __import__('json').loads(row.OSM_Dimension_Candidate_Evidence)
        self.assertFalse(evidence[0]['candidate_component_in_raw_intersection_set'])


if __name__ == '__main__':
    unittest.main()
