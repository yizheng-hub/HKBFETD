import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import geopandas as gpd
from shapely.geometry import box

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('osm_keyword_stage', SRC / '2.osm_aggregation.py')
osm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osm)


class OSMKeywordBoundaryTests(unittest.TestCase):
    def test_osm_keyword_classification_obeys_token_and_court_context(self):
        with tempfile.TemporaryDirectory() as directory:
            osm.OSM_CLEAN_PATH = str(Path(directory) / 'clean.geojson')
            names = ['SATISFACTORY COURT', 'Factory', 'Food Factory', 'Scenery Court',
                     'High Court', 'Food Court', 'Public Housing', 'Public HousingX',
                     'FooData CentreX', '工廠', '示例法院']
            features = gpd.GeoDataFrame({
                'id': range(1, len(names) + 1), 'osm_type': ['way'] * len(names), 'name': names,
                'building': [None] * len(names), 'building:use': [None] * len(names),
                'amenity': [None] * len(names), 'shop': [None] * len(names), 'office': [None] * len(names),
            }, geometry=[box(i * 10, 0, i * 10 + 5, 5) for i in range(len(names))], crs=2326)
            empty = gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs=2326)
            result = osm.preprocess_osm(features, empty, empty).set_index('name')
            self.assertTrue(result.loc['SATISFACTORY COURT', 'initial_main_class'] is None)
            self.assertEqual(result.loc['Factory', 'initial_main_class'], '工业类别')
            self.assertIn('食品及饮品', result.loc['Food Factory', 'initial_sub_class'])
            self.assertTrue(result.loc['Scenery Court', 'initial_main_class'] is None)
            self.assertEqual(result.loc['High Court', 'initial_main_class'], '商业类别')
            self.assertEqual(result.loc['Food Court', 'initial_main_class'], '商业类别')
            self.assertIn('食肆', result.loc['Food Court', 'initial_sub_class'])
            self.assertEqual(result.loc['Public Housing', 'initial_main_class'], '住宅类别')
            self.assertTrue(result.loc['Public HousingX', 'initial_main_class'] is None)
            self.assertTrue(result.loc['FooData CentreX', 'initial_main_class'] is None)
            self.assertEqual(result.loc['工廠', 'initial_main_class'], '工业类别')
            self.assertEqual(result.loc['示例法院', 'initial_main_class'], '商业类别')


if __name__ == '__main__':
    unittest.main()
