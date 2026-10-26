import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "src" / "replay_original_v2.py"
STAGE5_PATH = ROOT / "src" / "replay_original_v2_stage5.py"
FROZEN = ROOT / "src" / "frozen_original_method"


def load_runner():
    spec = importlib.util.spec_from_file_location("hkbfetd_original_replay", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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

    def test_stage7_fails_closed_without_current_vlm_contract(self):
        source = (ROOT / "src" / "replay_original_v2_stage7.py").read_text(encoding="utf-8")
        self.assertIn("withheld_missing_current_pair_prompt_image_contract", source)
        self.assertIn("stage7_withheld_vlm_decisions.csv", source)
        self.assertIn("verified_nonbuilding_structure_excluded_before_ML", source)
        self.assertNotIn("prior_vlm_merge_still_supported_by_new_geometry", source)

    def test_stage5_reuses_only_exact_prompt_hashes(self):
        source = STAGE5_PATH.read_text(encoding="utf-8")
        self.assertIn("Current_Prompt_SHA256", source)
        self.assertIn("Old_Prompt_SHA256", source)
        self.assertIn("legacy_cache_exact_prompt", source)
        self.assertIn("legacy_cache_prompt_changed", source)
        self.assertIn("An LLM classification lacks an exact current prompt contract", source)
        self.assertIn(
            'candidate_frame["Final_Main_Class"] = candidate_frame["Final_Main_Class"].map(',
            source,
        )
        self.assertIn(
            'candidate_frame["Final_Sub_Class"] = candidate_frame["Final_Sub_Class"].map(',
            source,
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

    def test_stage9_ml_cannot_replace_an_upstream_known_main_class(self):
        source = (ROOT / "src" / "replay_original_v2_stage89.py").read_text(encoding="utf-8")
        self.assertIn("def protect_upstream_classifications", source)
        self.assertIn("ML changed an upstream-determined main class", source)
        self.assertIn("ml_eligible_unknown_or_weak_main_rows", source)
        self.assertIn("前序主类别受保护，机器学习仅可补充缺失成分", source)

    def test_stage3_fully_reruns_rules_on_safe_foundation_inputs(self):
        source = (ROOT / "src" / "replay_original_v2_stage3.py").read_text(encoding="utf-8")
        self.assertIn("Run the frozen rule engine for all 341,153 buildings", source)
        self.assertIn("reused_exact_unaffected_rule_inputs", source)
        self.assertIn("legacy_checkpoint_preservation_contract", source)
        self.assertIn("BDBIAR|GFA|LLM|Inherited|Podium|Pre-Screening|Fusion|VLM", source)
        self.assertIn("rule_input_semantic_match", source)
        self.assertIn("legacy_effective_height", source)
        self.assertIn('legacy_rule["SHAPE_Area"]', source)
        self.assertIn('legacy_rule["BDBIAR_ADDRESS_C"]', source)
        self.assertIn("Unsafe BDBIAR values leaked into frozen-rule inputs", source)
        self.assertIn("Unusable permit GFA leaked into frozen-rule inputs", source)
        self.assertIn("stage3_foundation_scope_comparison.csv", source)

    def test_replay_requires_scoped_bdbiar_input(self):
        runner = load_runner()
        self.assertIn("bdbiar_scope", runner.REQUIRED_INPUT_ROLES)
        self.assertEqual(
            runner.ENV_BY_ROLE["bdbiar_scope"],
            "HKBFETD_REPLAY_BDBIAR_SCOPE",
        )
        self.assertIn(
            "checkpoints/stage7_withheld_vlm_decisions.csv",
            runner.STAGES[7]["outputs"],
        )

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
