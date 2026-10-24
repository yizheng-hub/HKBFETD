import sys
import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdbiar_integrity import apply_bdbiar_integrity_guard, load_bdbiar_integrity_registry


class BdbiarSourceIntegrityTests(unittest.TestCase):
    def test_quarantined_nulls_survive_actual_geojson_write_and_read(self):
        frame=gpd.GeoDataFrame({'BDBIAR_Source_Block_ID':['5145170'],
            'BDBIAR_AGE':[1972.], 'BDBIAR_CLASS':pd.Series(['Residential'],dtype='string'),
            'BDBIAR_Match_Quality':['shared_point']},geometry=[box(0,0,1,1)],crs=2326)
        guarded=apply_bdbiar_integrity_guard(frame,self.registry)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'guarded.geojson'
            guarded.to_file(path,driver='GeoJSON',AUTODETECT_JSON_STRINGS='NO')
            properties=json.loads(path.read_text(encoding='utf-8'))['features'][0]['properties']
            self.assertIsNone(properties['BDBIAR_CLASS'])
            self.assertIsNone(properties['BDBIAR_AGE'])
            read=gpd.read_file(path)
        self.assertTrue(pd.isna(read.BDBIAR_CLASS.iloc[0]))
        self.assertEqual(read.BDBIAR_CLASS_CANDIDATE.iloc[0],'Residential')
        self.assertEqual(float(read.BDBIAR_AGE_CANDIDATE.iloc[0]),1972.)

    @classmethod
    def setUpClass(cls):
        cls.registry = load_bdbiar_integrity_registry(
            Path(__file__).resolve().parents[1] / "ctl" / "bdbiar_source_integrity.csv"
        )

    def test_stable_block_id_controls_across_objectid_changes(self):
        frame = pd.DataFrame({
            "BDBIAR_Source_Block_ID": ["5145170", "5145170"],
            "BDBIAR_OBJECTID": [93144, 359836],
            "BDBIAR_AGE": [1972, 1972],
            "BDBIAR_CLASS": ["Residential", "Residential"],
            "BDBIAR_Match_Quality": ["inside_unique", "inside_unique"],
        })
        result = apply_bdbiar_integrity_guard(frame, self.registry)
        self.assertTrue(result["BDBIAR_AGE"].isna().all())
        self.assertTrue(result["BDBIAR_CLASS"].isna().all())
        self.assertEqual(result["BDBIAR_AGE_CANDIDATE"].tolist(), [1972, 1972])
        self.assertEqual(result["BDBIAR_CLASS_CANDIDATE"].tolist(), ["Residential", "Residential"])
        self.assertEqual(result["BDBIAR_OBJECTID"].tolist(), [93144, 359836])
        self.assertTrue(result["BDBIAR_Match_Quality"].eq("inside_unique").all())
        self.assertFalse(result["BDBIAR_Source_Attributes_Usable"].any())

    def test_nearby_match_is_not_promoted_and_qa_row_does_not_suppress(self):
        frame = pd.DataFrame({
            "BDBIAR_Source_Block_ID": ["6079064", "5953959"],
            "BDBIAR_AGE": [2001, 1980],
            "BDBIAR_CLASS": ["Residential", "Commercial"],
            "BDBIAR_Match_Quality": ["nearby_only", "nearby_only"],
        })
        result = apply_bdbiar_integrity_guard(frame, self.registry)
        self.assertEqual(result.loc[0, "BDBIAR_Source_Integrity_Status"], "snapshot_coordinate_change_uncertainty")
        self.assertTrue(pd.isna(result.loc[0, "BDBIAR_AGE"]))
        self.assertEqual(result.loc[0, "BDBIAR_AGE_CANDIDATE"], 2001)
        self.assertEqual(result.loc[1, "BDBIAR_Source_Integrity_Status"], "identity_scope_candidate")
        self.assertTrue(pd.isna(result.loc[1, "BDBIAR_CLASS"]))
        self.assertEqual(result.loc[1, "BDBIAR_CLASS_CANDIDATE"], "Commercial")
        self.assertFalse(result["BDBIAR_Source_Attributes_Usable"].any())

    def test_independent_landsd_permit_and_gfa_are_unchanged(self):
        frame = pd.DataFrame({
            "BDBIAR_Source_Block_ID": ["6060764"],
            "BDBIAR_AGE": [2010],
            "BDBIAR_CLASS": ["Residential"],
            "Permit_IDs": ["HK58/2016(OP)"],
            "GFA_DOMESTIC_SUM": [1234.5],
            "GFA_NONDOMESTIC_SUM": [67.8],
            "GROSSFLOORAREA": [1302.3],
        })
        result = apply_bdbiar_integrity_guard(frame, self.registry)
        for column in ("Permit_IDs", "GFA_DOMESTIC_SUM", "GFA_NONDOMESTIC_SUM", "GROSSFLOORAREA"):
            self.assertEqual(result.loc[0, column], frame.loc[0, column])

    def test_non_unique_and_registry_quarantine_preserve_candidates_idempotently(self):
        frame = pd.DataFrame({
            "BDBIAR_Source_Block_ID": ["5145170"],
            "BDBIAR_OBJECTID": [93144],
            "BDBIAR_AGE": [1972],
            "BDBIAR_CLASS": ["Residential"],
            "BDBIAR_Match_Quality": ["nearby_only"],
        })
        first = apply_bdbiar_integrity_guard(frame, self.registry)
        twice = apply_bdbiar_integrity_guard(first, self.registry)
        self.assertEqual(first.loc[0, "BDBIAR_AGE_CANDIDATE"], 1972)
        self.assertEqual(first.loc[0, "BDBIAR_CLASS_CANDIDATE"], "Residential")
        self.assertEqual(twice.loc[0, "BDBIAR_AGE_CANDIDATE"], 1972)
        self.assertEqual(twice.loc[0, "BDBIAR_CLASS_CANDIDATE"], "Residential")
        self.assertTrue(pd.isna(twice.loc[0, "BDBIAR_AGE"]))
        self.assertTrue(pd.isna(twice.loc[0, "BDBIAR_CLASS"]))
        self.assertTrue(twice.loc[0, "BDBIAR_Source_Integrity_Evidence"])

    def test_new_nonempty_fallback_updates_quarantined_candidate(self):
        frame = pd.DataFrame({
            "BDBIAR_Source_Block_ID": ["5145170"],
            "BDBIAR_AGE": [1972],
            "BDBIAR_CLASS": ["Residential"],
            "BDBIAR_Match_Quality": ["inside_unique"],
        })
        guarded = apply_bdbiar_integrity_guard(frame, self.registry)
        fallback = guarded.copy()
        fallback.loc[0, "BDBIAR_AGE"] = 1973
        refreshed = apply_bdbiar_integrity_guard(fallback, self.registry)
        self.assertEqual(refreshed.loc[0, "BDBIAR_AGE_CANDIDATE"], 1973)
        self.assertEqual(refreshed.loc[0, "BDBIAR_CLASS_CANDIDATE"], "Residential")
        self.assertTrue(pd.isna(refreshed.loc[0, "BDBIAR_AGE"]))


if __name__ == "__main__":
    unittest.main()
