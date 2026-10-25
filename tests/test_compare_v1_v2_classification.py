from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "compare_v1_v2_classification.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_v1_v2_classification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompareV1V2ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        ids = ["1", "2", "3", "4", "5", "6", "7"]
        self.v1 = self.root / "v1.csv"
        self.v2 = self.root / "v2.csv"
        self.stage3 = self.root / "stage3.csv"
        self.stage5 = self.root / "stage5.csv"
        self.stage7 = self.root / "stage7.csv"
        pd.DataFrame(
            {
                "Building_ID_建筑ID": ids,
                "UBEM_Main_Class_主类别": [
                    "Residential_住宅类别",
                    "Commercial_商业类别",
                    "Unknown_未知类别",
                    "Unknown_未知类别",
                    "Unknown_未知类别",
                    "Residential_住宅类别",
                    "Industrial_工业类别",
                ],
                "UBEM_Sub_Class_子类别": [
                    "Private Housing_私人房屋",
                    "Office_办公室",
                    "Unknown_未知类别",
                    "Unknown_未知类别",
                    "Unknown_未知类别",
                    "Unknown_未知类别",
                    "Non-manufacturing_非制造业",
                ],
                "Classification_Source_分类来源": [
                    "Machine_Learning_v1",
                    "GFA",
                    "Unknown",
                    "Unknown",
                    "Unknown",
                    "OSM+Keyword",
                    "Keyword",
                ],
                "LandsD_Building_Name_EN_地政署英文名": [f"Building {value}" for value in ids],
            }
        ).to_csv(self.v1, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "Building_ID_建筑ID": ids,
                "UBEM_Main_Class_主类别": [
                    "Non-assessed_非评估类别",
                    "Unknown_未知类别",
                    "Commercial_商业类别",
                    "Residential_住宅类别",
                    "Industrial_工业类别",
                    "Residential_住宅类别",
                    "Industrial_工业类别",
                ],
                "UBEM_Sub_Class_子类别": [
                    "Transport Infrastructure_交通基础设施",
                    "Unknown_未知类别",
                    "Office_办公室",
                    "Private Housing_私人房屋",
                    "Non-manufacturing_非制造业",
                    "Private Housing_私人房屋",
                    "Non-manufacturing_非制造业",
                ],
                "Classification_Source_分类来源": [
                    "Verified_nonbuilding_physical_scope",
                    "Unknown_after_unverified_GFA_withdrawal",
                    "LLM_Verified (Conf: 0.90)",
                    "Inherited_context_matched_VLM",
                    "Machine_Learning_v2",
                    "Keyword",
                    "Keyword",
                ],
                "LandsD_Building_Name_EN_地政署英文名": [f"Building {value}" for value in ids],
            }
        ).to_csv(self.v2, index=False, encoding="utf-8-sig")
        pd.DataFrame({"BUILDINGSTRUCTUREID": ["1", "2"]}).to_csv(
            self.stage3, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "BUILDINGSTRUCTUREID": ["2", "3"],
                "Stage5_Repair_Action": [
                    "native_llm_override_new_candidate_without_legacy_cache",
                    "native_llm_override_new_candidate_without_legacy_cache",
                ],
            }
        ).to_csv(self.stage5, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "BUILDINGSTRUCTUREID": ["1", "4"],
                "Stage7_Repair_Action": [
                    "verified_nonbuilding_structure_excluded_before_ML",
                    "inherited_from_99",
                ],
            }
        ).to_csv(self.stage7, index=False, encoding="utf-8-sig")

    def tearDown(self):
        self.temporary.cleanup()

    def test_counts_exclusive_priority_and_keyword_gate(self):
        out = self.root / "out"
        summary = self.module.compare_and_write(
            self.v1, self.v2, self.stage3, self.stage5, self.stage7, out,
            examples_per_group=1,
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["counts"]["main_changed"], 5)
        self.assertEqual(summary["counts"]["sub_changed"], 6)
        self.assertEqual(summary["counts"]["either_changed"], 6)
        self.assertEqual(
            summary["attribution_counts"],
            {
                "nonbuilding_scope": 1,
                "gfa_withdrawal": 1,
                "new_llm": 1,
                "vlm": 1,
                "ml_retraining_or_step10": 2,
            },
        )
        gate = summary["gates"]["old_keyword_main_class_stability"]
        self.assertEqual(gate["rows"], 2)
        self.assertEqual(gate["violations"], 0)
        self.assertEqual(summary["unknown_known"]["main_unknown_to_known"], 3)
        self.assertEqual(summary["unknown_known"]["main_known_to_unknown"], 1)

        changes = pd.read_csv(out / "classification_changes.csv", dtype=str)
        attribution = changes.set_index("Building_ID_建筑ID")["Attribution"].to_dict()
        self.assertEqual(attribution["1"], "nonbuilding_scope")
        self.assertEqual(attribution["2"], "gfa_withdrawal")
        self.assertEqual(attribution["3"], "new_llm")
        self.assertEqual(attribution["4"], "vlm")
        self.assertEqual(attribution["5"], "ml_retraining_or_step10")
        self.assertEqual(attribution["6"], "ml_retraining_or_step10")
        self.assertEqual(len(changes), sum(summary["attribution_counts"].values()))
        self.assertTrue((out / "classification_typical_examples.csv").is_file())
        self.assertIn("硬门通过", (out / "classification_comparison_zh.md").read_text(encoding="utf-8"))

    def test_keyword_main_change_writes_fail_report_and_cli_exits_nonzero(self):
        v2 = pd.read_csv(self.v2, dtype="string", encoding="utf-8-sig")
        v2.loc[v2["Building_ID_建筑ID"].eq("6"), "UBEM_Main_Class_主类别"] = "Commercial_商业类别"
        v2.to_csv(self.v2, index=False, encoding="utf-8-sig")
        out = self.root / "failed"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--v1", str(self.v1),
                "--v2", str(self.v2),
                "--stage3-gfa", str(self.stage3),
                "--stage5", str(self.stage5),
                "--stage7", str(self.stage7),
                "--out-dir", str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        report = json.loads((out / "classification_comparison.json").read_text(encoding="utf-8"))
        gate = report["gates"]["old_keyword_main_class_stability"]
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(gate["violations"], 1)
        self.assertEqual(gate["violation_ids"], ["6"])
        self.assertIn("不得发布", (out / "classification_comparison_zh.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
