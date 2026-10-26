from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "release_classification_integrity.py"
SPEC = importlib.util.spec_from_file_location("release_integrity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class ReleaseClassificationIntegrityTests(unittest.TestCase):
    def test_aliases_and_parent_conflicts_do_not_change_main_classes(self) -> None:
        source = pd.DataFrame(
            {
                "BUILDINGSTRUCTUREID": ["26509.0", "alias", "conflict", "mixed"],
                "Calibrated_Main_Class": [
                    "Residential_住宅类别",
                    "Industrial_工业类别",
                    "Residential_住宅类别",
                    "Mixed-use_混合用途",
                ],
                "Calibrated_Sub_Class": [
                    "Private Housing_私人房屋",
                    "非制造业（Non-manufacturing）",
                    "Office_办公室",
                    "Private Housing_私人房屋-Office_办公室",
                ],
                "Calibrated_Mix_Proportion": [
                    "Private Housing_私人房屋:1.00",
                    "Non-Non-manufacturing_非制造业:1.00",
                    "Office_办公室:1.00",
                    "Private Housing_私人房屋:0.50-Office_办公室:0.50",
                ],
                "Unrelated_Value": [1, 2, 3, 4],
            }
        )

        result, audit, summary = gate.enforce_integrity(source)

        pd.testing.assert_series_equal(
            result["Calibrated_Main_Class"], source["Calibrated_Main_Class"]
        )
        pd.testing.assert_series_equal(
            result["BUILDINGSTRUCTUREID"], source["BUILDINGSTRUCTUREID"]
        )
        pd.testing.assert_series_equal(result["Unrelated_Value"], source["Unrelated_Value"])
        # The gate has no building-ID scope overrides.  Stage 7 owns the known
        # nonbuilding list, so even that list's IDs are handled only by values.
        self.assertEqual(
            result.loc[0, "Calibrated_Sub_Class"], "Private Housing_私人房屋"
        )
        self.assertEqual(
            result.loc[1, "Calibrated_Sub_Class"], "Non-manufacturing_非制造业"
        )
        self.assertEqual(
            result.loc[1, "Calibrated_Mix_Proportion"],
            "Non-manufacturing_非制造业:1.00",
        )
        self.assertEqual(result.loc[2, "Calibrated_Sub_Class"], gate.UNKNOWN_SUBCLASS)
        self.assertTrue(pd.isna(result.loc[2, "Calibrated_Mix_Proportion"]))
        self.assertEqual(
            result.loc[3, "Calibrated_Sub_Class"],
            "Private Housing_私人房屋-Office_办公室",
        )
        self.assertEqual(summary["main_classes_changed"], 0)
        self.assertEqual(set(audit["BUILDINGSTRUCTUREID"]), {"alias", "conflict"})

    def test_non_assessed_aliases_are_value_driven(self) -> None:
        source = pd.DataFrame(
            {
                "BUILDINGSTRUCTUREID": ["arbitrary-id"],
                "Calibrated_Main_Class": ["Non-assessed_非评估类别"],
                "Calibrated_Sub_Class": [
                    "Temporary/Misc Facilities_临时/杂项设施"
                ],
                "Calibrated_Mix_Proportion": [
                    "Temporary/Misc Facilities_临时/杂项设施:1.00"
                ],
            }
        )
        result, _, _ = gate.enforce_integrity(source)
        expected = "Temporary/Miscellaneous Structures_临时/杂项设施"
        self.assertEqual(
            result.loc[0, "Calibrated_Main_Class"],
            source.loc[0, "Calibrated_Main_Class"],
        )
        self.assertEqual(result.loc[0, "Calibrated_Sub_Class"], expected)
        self.assertEqual(result.loc[0, "Calibrated_Mix_Proportion"], f"{expected}:1.00")

    def test_mixed_single_parent_dependents_fail_closed(self) -> None:
        source = pd.DataFrame(
            {
                "BUILDINGSTRUCTUREID": ["one"],
                "Calibrated_Main_Class": ["Mixed-use_混合用途"],
                "Calibrated_Sub_Class": ["Private Housing_私人房屋"],
                "Calibrated_Mix_Proportion": ["Private Housing_私人房屋:1.00"],
            }
        )
        result, audit, _ = gate.enforce_integrity(source)
        self.assertEqual(result.loc[0, "Calibrated_Main_Class"], "Mixed-use_混合用途")
        self.assertEqual(result.loc[0, "Calibrated_Sub_Class"], gate.UNKNOWN_SUBCLASS)
        self.assertTrue(pd.isna(result.loc[0, "Calibrated_Mix_Proportion"]))
        self.assertIn(
            "mixture_withheld_mixed_single_parent",
            audit.loc[0, "Release_Integrity_Actions"],
        )

    def test_unsupported_main_class_is_rejected_not_rewritten(self) -> None:
        source = pd.DataFrame(
            {
                "BUILDINGSTRUCTUREID": ["one"],
                "Calibrated_Main_Class": ["住宅类别"],
                "Calibrated_Sub_Class": ["Private Housing_私人房屋"],
                "Calibrated_Mix_Proportion": ["Private Housing_私人房屋:1.00"],
            }
        )
        with self.assertRaisesRegex(ValueError, "Unsupported main-class"):
            gate.enforce_integrity(source)

    def test_explicit_unknown_main_class_is_preserved(self) -> None:
        source = pd.DataFrame(
            {
                "BUILDINGSTRUCTUREID": ["one"],
                "Calibrated_Main_Class": ["Unknown_未知类别"],
                "Calibrated_Sub_Class": ["Unknown_未知类别"],
                "Calibrated_Mix_Proportion": [pd.NA],
            }
        )
        result, _, summary = gate.enforce_integrity(source)
        self.assertEqual(result.loc[0, "Calibrated_Main_Class"], "Unknown_未知类别")
        self.assertEqual(result.loc[0, "Calibrated_Sub_Class"], gate.UNKNOWN_SUBCLASS)
        self.assertTrue(pd.isna(result.loc[0, "Calibrated_Mix_Proportion"]))
        self.assertEqual(summary["main_classes_changed"], 0)

    def test_cli_uses_requested_output_and_audit_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = base / "input.csv"
            output_path = base / "nested" / "output.csv"
            audit_path = base / "ledger" / "audit.csv"
            manifest_path = base / "manifest" / "manifest.json"
            pd.DataFrame(
                {
                    "BUILDINGSTRUCTUREID": ["one"],
                    "Calibrated_Main_Class": ["Residential_住宅类别"],
                    "Calibrated_Sub_Class": ["Office_办公室"],
                    "Calibrated_Mix_Proportion": ["Office_办公室:1.00"],
                }
            ).to_csv(input_path, index=False, encoding="utf-8-sig")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--audit",
                    str(audit_path),
                    "--manifest",
                    str(manifest_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.is_file())
            self.assertTrue(audit_path.is_file())
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["main_classes_changed"], 0)
            self.assertEqual(manifest["fail_closed_rows"], 1)


if __name__ == "__main__":
    unittest.main()
