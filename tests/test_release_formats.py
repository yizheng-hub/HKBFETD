import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box, Point
import pyogrio

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('release_stage', SRC/'10.final_statistics_report.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class ReleaseGeoJSONTests(unittest.TestCase):
    def test_near_integer_coordinates_keep_requested_decimal_precision(self):
        longitude, latitude = 113.99999990077416, 22.434135147231423
        frame = gpd.GeoDataFrame({'ID': ['precision_case']},
            geometry=[Point(longitude, latitude)], crs=4326)
        rounding = pyogrio.get_gdal_config_option('OGR_WKT_ROUND')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'public.geojson'
            stage.write_public_geojson(frame, path)
            x, y = json.loads(path.read_text(encoding='utf-8'))['features'][0]['geometry']['coordinates']
            self.assertLessEqual(abs(x-longitude), 5e-10)
            self.assertLessEqual(abs(y-latitude), 5e-10)
        self.assertEqual(pyogrio.get_gdal_config_option('OGR_WKT_ROUND'), rounding)

    def test_metric_source_exports_standard_geojson_without_changing_attributes(self):
        frame = gpd.GeoDataFrame({
            'Building_ID_建筑ID': ['001'], 'Height_m': [np.nan],
            'evidence': ['[{"source":"地政署"}]'], 'eligible': [False],
            'area_sqm': [100.0],
        }, geometry=[box(835000, 815000, 835010, 815010)], crs=2326)
        original = frame.copy()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'public.geojson'
            stage.write_public_geojson(frame, path)
            payload = json.loads(path.read_text(encoding='utf-8'),
                parse_constant=lambda value: self.fail('Non-JSON number: '+value))
            self.assertNotIn('crs', payload)
            feature = payload['features'][0]
            self.assertEqual(feature['properties'], {
                'Building_ID_建筑ID': '001', 'Height_m': None,
                'evidence': '[{"source":"地政署"}]', 'eligible': False, 'area_sqm': 100.0})
            ring = feature['geometry']['coordinates'][0]
            self.assertTrue(all(113 < x < 115 and 22 < y < 23 for x, y in ring))
            self.assertTrue(shapely.is_ccw(shapely.LinearRing(ring)))
            loaded = gpd.read_file(path)
            self.assertEqual(loaded.crs.to_epsg(), 4326)
            expected = original.to_crs(4326).geometry.iloc[0]
            self.assertTrue(shapely.equals_exact(shapely.normalize(expected),
                shapely.normalize(loaded.geometry.iloc[0]), tolerance=1e-9))
        self.assertEqual(frame.crs, original.crs)
        self.assertTrue(frame.geometry.equals(original.geometry))
        self.assertEqual(frame.area_sqm.tolist(), [100.0])


if __name__ == '__main__':
    unittest.main()
