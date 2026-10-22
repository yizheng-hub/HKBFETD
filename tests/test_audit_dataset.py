import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location("audit_dataset", Path(__file__).resolve().parents[1] / "src/audit_dataset.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ProportionParsingTests(unittest.TestCase):
    def test_hyphenated_label_is_not_split(self):
        pairs, valid = audit.parse_proportions("Non-manufacturing_非制造业:0.60-Office_办公室:0.40")
        self.assertTrue(valid)
        self.assertEqual(pairs, [("Non-manufacturing_非制造业", .6), ("Office_办公室", .4)])

    def test_duplicate_labels_remain_visible(self):
        pairs, valid = audit.parse_proportions("Office:0.4-Office:0.6")
        self.assertTrue(valid)
        self.assertEqual(len(pairs), 2)

    def test_negative_and_scientific_values_remain_visible(self):
        self.assertEqual(audit.parse_proportions("Office:-0.1-Retail:1.1"), ([("Office", -.1), ("Retail", 1.1)], True))
        self.assertEqual(audit.parse_proportions("Office:1e-2-Retail:0.99"), ([("Office", .01), ("Retail", .99)], True))

    def test_does_not_accept_partial_vector(self):
        for value in [None, np.nan, "", "Office", "Office:1-trailing", "Office:1-", "Office:nan", ":1", "Office:1garbage"]:
            with self.subTest(value=value):
                self.assertFalse(audit.parse_proportions(value)[1])

    def test_any_tied_maximum_is_a_valid_dominant_subclass(self):
        pairs = [("Retail", .33), ("Restaurant", .33), ("Office", .33)]
        self.assertTrue(audit.is_dominant_subclass("Office", pairs))
        self.assertFalse(audit.is_dominant_subclass("Housing", pairs))
        self.assertFalse(audit.is_dominant_subclass("Office", [("Retail", .8), ("Office", .2)]))


class AuditIntegrityTests(unittest.TestCase):
    def test_preserves_missing_ids(self):
        result = audit.normalized_ids(pd.Series([None, "181189.0", " 170369 "]))
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertEqual(result.iloc[1:].tolist(), ["181189", "170369"])

    def test_refuses_overwriting_existing_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                audit.Audit(Path(temp), Path(temp))

    def test_array_flags_keep_building_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = audit.Audit(Path(temp), Path(temp) / "audit")
            runner.flags = pd.DataFrame(index=["2", "1"])
            runner.flag("example", np.array([True, False]), "test")
            self.assertEqual(runner.flags.example.to_dict(), {"2": True, "1": False})

    def test_duplicate_stage_ids_fail_instead_of_silent_join(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = audit.Audit(Path(temp), Path(temp) / "audit")
            with self.assertRaises(ValueError):
                runner.stage("invalid", pd.DataFrame({audit.ID: ["1", "1"]}), ["1", "2"])

    def test_distinguishes_no_dimensions_from_real_five_storeys(self):
        ids = pd.Index(["default", "five", "top_only", "top_and_floors", "negative_top"], name=audit.ID)
        raw = pd.DataFrame({"TOPHEIGHT": [np.nan, np.nan, 957.4, 80., -1.],
                            "BASEHEIGHT": np.nan, "NUMABOVEGROUNDSTOREYS": [np.nan, 5., np.nan, 20., np.nan]}, index=ids)
        public = pd.DataFrame({"Height_m_建筑高度": [15., 15., 957.4, 80., 15.],
                               "Floors_Count_地上层数": [5., 5., 319., 20., 5.],
                               "Footprint_Area_sqm_占地面积": 50., "UBEM_Main_Class_主类别": "Commercial_商业类别",
                               "Classification_Source_分类来源": "test"}, index=ids)
        with tempfile.TemporaryDirectory() as temp:
            runner = audit.Audit(Path(temp), Path(temp) / "audit")
            runner.flags = pd.DataFrame(index=ids)
            runner.morphology(raw, public)
            self.assertEqual(runner.flags.default_15m_no_raw_dimensions.tolist(), [True, False, False, False, True])
            self.assertEqual(runner.flags.height_15m_from_raw_5_floors.tolist(), [False, True, False, False, False])
            self.assertEqual(runner.flags.top_elevation_exported_without_base.tolist(), [False, False, True, True, False])
            self.assertEqual(runner.flags.storeys_inferred_from_unbased_top.tolist(), [False, False, True, False, False])


if __name__ == "__main__":
    unittest.main()
