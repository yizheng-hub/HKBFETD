import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "src" / "replay_original_v2.py"
STAGE5_PATH = ROOT / "src" / "replay_original_v2_stage5.py"
STAGE7_PATH = ROOT / "src" / "replay_original_v2_stage7.py"
FROZEN = ROOT / "src" / "frozen_original_method"


def load_runner():
    spec = importlib.util.spec_from_file_location("hkbfetd_original_replay", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_stage(path, environment):
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.dict(os.environ, environment):
        spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OriginalMethodReplayTests(unittest.TestCase):
    def test_frozen_sources_match_manifest(self):
        runner = load_runner()
        verified = runner.verify_frozen_method()
        self.assertEqual(
            verified["source_git_commit"],
            "de8b46b32fb5853a0777cacd09cbe099b4141a08",
        )
        self.assertEqual(
            verified["classification_order"],
            [
                "rule_engine",
                "llm_checkpoint",
                "vlm_checkpoint",
                "nine_feature_machine_learning",
                "step10_public_classification_standardization",
            ],
        )

    def test_original_nine_features_are_explicit(self):
        tree = ast.parse((ROOT / "src" / "replay_original_v2_stage89.py").read_text(encoding="utf-8"))
        values = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "FEATURES"
                for target in node.targets
            ):
                values = ast.literal_eval(node.value)
                break
        self.assertEqual(
            values,
            [
                "area",
                "perimeter",
                "compactness",
                "centroid_x",
                "centroid_y",
                "ozp_code",
                "neighbor_density_50m",
                "dominant_neighbor_class_100m",
                "Estimated_Height",
            ],
        )

    def test_replay_scripts_have_no_local_work_paths(self):
        scripts = [RUNNER_PATH, *sorted((ROOT / "src").glob("replay_original_v2_stage*.py"))]
        for path in scripts:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("D:\\HKBFETD", text, path.name)
            self.assertNotIn("work/original_method_recovery", text, path.name)
            self.assertNotIn("work\\original_method_recovery", text, path.name)

    def test_web_function_evidence_is_rejected_not_loaded(self):
        runner_text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("Web-derived building function evidence is not a replay input", runner_text)
        for path in sorted((ROOT / "src").glob("replay_original_v2_stage*.py")):
            self.assertNotIn("building_function_evidence.csv", path.read_text(encoding="utf-8"))

    def test_step10_and_public_mapping_are_frozen(self):
        manifest = json.loads((FROZEN / "source_manifest.json").read_text(encoding="utf-8"))
        for relative in (
            "src/10.final_statistics_report.py",
            "ctl/step10_public_mappings.json",
        ):
            self.assertEqual(sha256(FROZEN / relative), manifest["files"][relative])
        runner = load_runner()
        self.assertIn(10, runner.STAGES)
        self.assertIn(
            "ml/step10_original_method_standardized.csv",
            runner.STAGES[10]["outputs"],
        )

    def test_stage7_keeps_original_inheritance_field_scope(self):
        source = (ROOT / "src" / "replay_original_v2_stage7.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        host_fields = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "host_fields"
                for target in node.targets
            ):
                host_fields = ast.literal_eval(node.value)
                break
        self.assertEqual(host_fields, ["Final_Main_Class", "Final_Sub_Class"])
        self.assertIn("verified_nonbuilding_structure_excluded_before_ML", source)
        self.assertEqual(
            source.count('result.at[building_id, "Main_Class_Proportions"]'),
            1,
        )

    def test_stage7_requires_structured_vlm_audit_approval_and_exact_scope(self):
        dummy = str(ROOT)
        stage7 = load_stage(
            STAGE7_PATH,
            {
                "HKBFETD_REPLAY_OUTPUT_DIR": dummy,
                "HKBFETD_REPLAY_VERIFIED_PHYSICAL": dummy,
                "HKBFETD_REPLAY_NONBUILDING_SCOPE": dummy,
                "HKBFETD_REPLAY_STAGE6_SUGGESTIONS": dummy,
                "HKBFETD_REPLAY_VLM_AUDIT": dummy,
            },
        )
        audit = {
            "scope": {
                "target_ids": list(stage7.EXPECTED_VLM_TARGET_IDS),
                "special_pair": dict(stage7.EXPECTED_VLM_SPECIAL_PAIR),
            },
            "geometry_change_comparison": [
                {
                    "BUILDINGSTRUCTUREID": building_id,
                    "old_present": True,
                    "new_present": True,
                    "old_geometry": {"valid": True},
                    "new_geometry": {"valid": True},
                }
                for building_id in stage7.EXPECTED_VLM_TARGET_IDS
            ],
            "prior_vlm_decision": {
                **stage7.EXPECTED_VLM_SPECIAL_PAIR,
                "final_suggestion": "Consensus_merge",
            },
            "prior_vlm_merge_still_supported_by_new_geometry": True,
        }
        stage7.validate_vlm_audit(audit)

        rejected = copy.deepcopy(audit)
        rejected["prior_vlm_merge_still_supported_by_new_geometry"] = False
        rejected["irrelevant_text"] = "valid reuse"
        with self.assertRaisesRegex(AssertionError, "explicitly approve reuse"):
            stage7.validate_vlm_audit(rejected)

        wrong_scope = copy.deepcopy(audit)
        wrong_scope["scope"]["target_ids"][0] = 999999
        with self.assertRaisesRegex(AssertionError, "target IDs"):
            stage7.validate_vlm_audit(wrong_scope)

    def test_stage5_prompt_contract_rejects_tampered_review_hashes(self):
        dummy = str(ROOT)
        stage5 = load_stage(
            STAGE5_PATH,
            {
                "HKBFETD_REPLAY_OUTPUT_DIR": dummy,
                "HKBFETD_REPLAY_LLM_REVIEW_SUMMARY": dummy,
                "HKBFETD_REPLAY_LLM_PROMPT_CONTRACT": dummy,
                "HKBFETD_REPLAY_NATIVE_REVIEWS": dummy,
                "HKBFETD_REPLAY_LEGACY_STAGE4": dummy,
                "HKBFETD_REPLAY_LEGACY_STAGE5": dummy,
            },
        )
        expected_status_by_id = {
            str(value): (
                "cache_prompt_changed" if value <= 427 else "new_candidate_without_cache"
            )
            for value in range(1, 507)
        }
        records = []
        reviews = []
        for building_id, status in expected_status_by_id.items():
            digest = hashlib.sha256(f"prompt-{building_id}".encode()).hexdigest()
            records.append({
                "BUILDINGSTRUCTUREID": building_id,
                "status": status,
                "new_prompt_sha256": digest,
            })
            reviews.append({
                "BUILDINGSTRUCTUREID": building_id,
                "status": status,
                "input_prompt_sha256": digest,
                "new_prompt_sha256": digest,
            })
        contract = {
            "schema_version": 1,
            "record_count": 506,
            "status_counts": {
                "cache_prompt_changed": 427,
                "new_candidate_without_cache": 79,
            },
            "records": records,
        }
        verified = stage5.validate_prompt_contract(
            contract,
            reviews,
            expected_status_by_id,
        )
        self.assertEqual(len(verified), 506)

        for field in ("input_prompt_sha256", "new_prompt_sha256"):
            tampered = copy.deepcopy(reviews)
            tampered[0][field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(
                AssertionError,
                "prompt SHA-256 differs",
            ):
                stage5.validate_prompt_contract(
                    contract,
                    tampered,
                    expected_status_by_id,
                )

    def test_stage9_resume_requires_every_reproducibility_artifact(self):
        expected = {
            "ml/features_base_indexed.pkl",
            "ml/features_X_all.pkl",
            "ml/features_X_train_ml.pkl",
            "ml/features_y_multilabel.pkl",
            "ml/models.pkl",
            "ml/probabilities.csv",
            "ml/step8_feature_base_calibrated.geojson",
            "ml/step9_corrected_ml_calibrated.csv",
            "ml/step9_corrected_ml_calibrated.geojson",
            "ml/training_samples.pkl",
            "ml/metrics.json",
            "ml/hash_manifest.json",
        }
        self.assertEqual(set(load_runner().STAGES[9]["outputs"]), expected)

    def test_stage3_preserves_independent_non_gfa_decisions(self):
        source = (ROOT / "src" / "replay_original_v2_stage3.py").read_text(encoding="utf-8")
        self.assertIn("gfa_only = {'GFA', 'GFA_Mixed'}", source)
        self.assertIn("independent_non_gfa_classification_preserved", source)
        self.assertIn("without_gfa_evidence", source)
        self.assertIn("stage3_gfa_weak_fallbacks_withheld.csv", source)
        self.assertIn("Expected 117 GFA-withdrawal LandsD-only fallbacks", source)

    def test_input_schema_has_exact_roles(self):
        schema = json.loads(
            (ROOT / "ctl" / "original_method_replay_inputs.schema.json").read_text(
                encoding="utf-8"
            )
        )
        schema_roles = set(schema["properties"]["files"]["required"])
        self.assertEqual(schema_roles, load_runner().REQUIRED_INPUT_ROLES)


if __name__ == "__main__":
    unittest.main()
