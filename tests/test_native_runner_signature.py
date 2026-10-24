from pathlib import Path
import importlib.util
import shutil
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("candidate_runner", ROOT / "src" / "run_pipeline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class NativeSourceSignatureTests(unittest.TestCase):
    def test_native_input_changes_leave_steps_1_to_3_signatures_unchanged_and_invalidate_step4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "data" / "sources" / "native_function_classifier_checkpoint.json"
            policy = root / "data" / "sources" / "native_function_policy_clarification.txt"
            checkpoint.parent.mkdir(parents=True)
            base = root / "data" / "sources" / "source_manifest.json"
            base.write_text("{}", encoding="utf-8")
            before = runner.hash_paths([base], root)
            native = [checkpoint, policy]
            with checkpoint.open("w", encoding="utf-8") as stream:
                stream.write('{"schema_version":1}')
            with policy.open("w", encoding="utf-8") as stream:
                stream.write("policy A")
            added = runner.hash_paths([base, checkpoint, policy], root)
            for step in (1, 2, 3):
                self.assertEqual(runner.stage_input_signature(before, step, native, root),
                                 runner.stage_input_signature(added, step, native, root))
            self.assertNotEqual(runner.stage_input_signature(before, 4, native, root),
                                runner.stage_input_signature(added, 4, native, root))
            step4_before = runner.stage_input_signature(added, 4, native, root)
            checkpoint.write_text('{"schema_version":1,"edit":true}', encoding="utf-8")
            edited = runner.hash_paths([base, checkpoint, policy], root)
            self.assertEqual(runner.stage_input_signature(added, 3, native, root),
                             runner.stage_input_signature(edited, 3, native, root))
            self.assertNotEqual(step4_before, runner.stage_input_signature(edited, 4, native, root))

    def test_reproduction_bundle_freezes_native_inputs_for_a_fresh_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def setup_root(base):
                paths = {}
                data = base / "data"
                sources = data / "sources"
                sources.mkdir(parents=True, exist_ok=True)
                for name, contents in (("native_function_classifier_checkpoint.json", "{\"schema_version\":1}"),
                                       ("native_function_policy_clarification.txt", "policy"),
                                       ("source_manifest.json", "manifest")):
                    path = sources / name
                    path.write_text(contents, encoding="utf-8")
                    paths[name] = path
                for folder in ("src", "ctl", "tests", "ozp", "legacy", "legacy_intermediate", "output"):
                    (base / folder).mkdir(exist_ok=True)
                (base / "src" / "placeholder.py").write_text("code", encoding="utf-8")
                (base / "ctl" / "placeholder.json").write_text("{}", encoding="utf-8")
                (base / "tests" / "placeholder.py").write_text("test", encoding="utf-8")
                required = {}
                names = ("bdbiar", "overture", "osm_snapshot", "terrain_samples", "terrain_metadata",
                         "csdi_buildings", "csdi_metadata", "osm_structures", "osm_structures_metadata")
                for name in names:
                    path = data / (name + ".dat")
                    path.write_text(name, encoding="utf-8")
                    required[name] = path
                public = base / "legacy" / "HK_UBEM_Buildings_Public_v1_1.csv"
                public.write_text("id\n", encoding="utf-8")
                cfg = SimpleNamespace(BASE_DIR=str(base), OUTPUT_DIR=str(base / "output"),
                    FILE_PATHS_LANDSD={}, OZP_DATA_DIR=str(base / "ozp"),
                    BDBIAR_FILE_PATH=str(required["bdbiar"]), OVERTURE_PARQUET_FILE_PATH=str(required["overture"]),
                    OSM_SOURCE_SNAPSHOT=str(required["osm_snapshot"]), TERRAIN_SAMPLES_PATH=str(required["terrain_samples"]),
                    TERRAIN_METADATA_PATH=str(required["terrain_metadata"]), CSDI_BUILDINGS_PATH=str(required["csdi_buildings"]),
                    CSDI_METADATA_PATH=str(required["csdi_metadata"]), OSM_STRUCTURES_PATH=str(required["osm_structures"]),
                    OSM_STRUCTURES_METADATA_PATH=str(required["osm_structures_metadata"]),
                    NATIVE_FUNCTION_CHECKPOINT=str(paths["native_function_classifier_checkpoint.json"]),
                    NATIVE_FUNCTION_POLICY_CLARIFICATION=str(paths["native_function_policy_clarification.txt"]),
                    LEGACY_OUTPUT_DIR=str(base / "legacy"), LEGACY_INTERMEDIATE_DIR=str(base / "legacy_intermediate"))
                return cfg
            cfg = setup_root(root)
            files = runner.input_files(cfg)
            runner.make_bundles(cfg, files, include_inputs=True)
            extracted = root / "fresh"
            with zipfile.ZipFile(root / "output" / "reproduction_inputs.zip") as archive:
                archive.extractall(extracted)
            original_paths = runner.input_files(cfg)
            fresh_paths = [extracted / path.relative_to(root) for path in original_paths]
            original_inputs = runner.hash_paths(original_paths, root)
            fresh_inputs = runner.hash_paths(fresh_paths, extracted)
            self.assertEqual(original_inputs, fresh_inputs)
            self.assertEqual(runner.stage_input_signature(original_inputs, 4,
                             [cfg.NATIVE_FUNCTION_CHECKPOINT, cfg.NATIVE_FUNCTION_POLICY_CLARIFICATION], root),
                             runner.stage_input_signature(fresh_inputs, 4,
                             [extracted / Path(cfg.NATIVE_FUNCTION_CHECKPOINT).relative_to(root),
                              extracted / Path(cfg.NATIVE_FUNCTION_POLICY_CLARIFICATION).relative_to(root)], extracted))

    def test_native_loader_source_hash_invalidates_step4_and_step5(self):
        with tempfile.TemporaryDirectory() as directory:
            copy_root = Path(directory)
            for step in (4, 5):
                for source in runner.stage_source_paths(ROOT, step):
                    target = copy_root / source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            for step in (4, 5):
                paths = runner.stage_source_paths(copy_root, step)
                before = runner.hash_paths(paths, copy_root)
                self.assertIn("src/native_function_checkpoint.py", before)
                helper = copy_root / "src" / "native_function_checkpoint.py"
                helper.write_text(helper.read_text(encoding="utf-8") + "\\n# signature invalidation probe\\n", encoding="utf-8")
                after = runner.hash_paths(runner.stage_source_paths(copy_root, step), copy_root)
                self.assertNotEqual(before, after)
                helper.write_text(helper.read_text(encoding="utf-8").replace("\\n# signature invalidation probe\\n", ""), encoding="utf-8")

if __name__ == "__main__":
    unittest.main()

