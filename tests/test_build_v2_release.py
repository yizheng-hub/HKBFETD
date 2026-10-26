import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ModuleNotFoundError:
    gpd = None
    Point = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "build_v2_release.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_v2_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildV2ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        frame = pd.DataFrame(
            {
                "Building_ID_建筑ID": ["001", "002"],
                "Height_m_建筑高度": [10.0, float("nan")],
                "Physical_Review_Flags": ["", "verified"],
            }
        )
        self.parquet = self.root / "final.parquet"
        frame.to_parquet(self.parquet, index=False)
        self.csv = self.root / "final.csv"
        frame.to_csv(self.csv, index=False, encoding="utf-8-sig", na_rep="")
        self.geojson = self.root / "final.geojson"
        self._write_geojson(["001", "002"])
        self.schema = self._file("schema.json", "{}")
        self.readme = self._file("release_readme.md", "# release\n")
        self.model = self._file("model.joblib", "model")
        self.metrics = self._file("metrics.json", "{}")
        self.validation = self._file("independent_validation.json", '{"status":"passed"}')
        self.reproduction = self._file("reproduction_inputs.zip", "inputs")
        self.snapshot = self._file("source_snapshot.zip", "source")

    def tearDown(self):
        self.temporary.cleanup()

    def _file(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def _write_geojson(self, ids, crs="EPSG:4326"):
        features = []
        values = [(10.0, None), (None, "verified")]
        for index, building_id in enumerate(ids):
            height, review = values[index]
            features.append({
                "type": "Feature",
                "properties": {
                    "Building_ID_建筑ID": building_id,
                    "Height_m_建筑高度": height,
                    "Physical_Review_Flags": review,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [114.1 + index * 0.01, 22.3 + index * 0.01],
                },
            })
        document = {"type": "FeatureCollection", "features": features}
        if crs != "EPSG:4326":
            document["crs"] = {
                "type": "name",
                "properties": {"name": crs},
            }
        self.geojson.write_text(
            json.dumps(document, ensure_ascii=False),
            encoding="utf-8",
        )

    def _inputs(self):
        return self.module.ReleaseInputs(
            csv_path=self.csv,
            parquet_path=self.parquet,
            geojson_path=self.geojson,
            schema_path=self.schema,
            readme_path=self.readme,
            model_paths=(self.model,),
            metrics_path=self.metrics,
            validation_report_paths=(self.validation,),
            reproduction_inputs_path=self.reproduction,
            source_snapshot_path=self.snapshot,
        )

    def test_build_normalises_blank_strings_and_creates_complete_archive(self):
        out = self.root / "release"
        result = self.module.build_release(out, self._inputs())
        self.assertEqual(result["status"], "passed")
        parquet = pd.read_parquet(out / "HKBFETD_v2.parquet")
        csv_frame = pd.read_csv(
            out / "HKBFETD_v2.csv",
            dtype={"Building_ID_建筑ID": "string"},
            encoding="utf-8-sig",
        )
        self.assertTrue(pd.isna(parquet.loc[0, "Physical_Review_Flags"]))
        self.assertTrue(pd.isna(csv_frame.loc[0, "Physical_Review_Flags"]))
        self.assertEqual(csv_frame["Building_ID_建筑ID"].tolist(), ["001", "002"])
        manifest = json.loads((out / "release_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["table_validation"]["csv_parquet_semantic_match"])
        self.assertEqual(
            manifest["table_validation"]["geojson"],
            {
                "feature_collection": True,
                "features": 2,
                "property_fields_match": True,
                "unique_id_set_matches": True,
                "crs": "EPSG:4326",
            },
        )
        self.assertEqual(
            manifest["table_validation"]["blank_strings_normalised_to_null"],
            {"Physical_Review_Flags": 1},
        )
        checksum_lines = (out / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(line.endswith("  models/model.joblib") for line in checksum_lines))
        with zipfile.ZipFile(out / "HKBFETD_v2_release.zip") as archive:
            names = set(archive.namelist())
        self.assertIn("SHA256SUMS.txt", names)
        self.assertIn("reproduction/reproduction_inputs.zip", names)
        self.assertNotIn("HKBFETD_v2_release.zip", names)
        archive_digest = hashlib.sha256(
            (out / "HKBFETD_v2_release.zip").read_bytes()
        ).hexdigest()
        self.assertTrue(
            (out / "HKBFETD_v2_release.zip.sha256")
            .read_text(encoding="utf-8")
            .startswith(archive_digest)
        )

    def test_dry_run_validates_without_creating_output(self):
        out = self.root / "dry-release"
        result = self.module.build_release(out, self._inputs(), dry_run=True)
        self.assertEqual(result["status"], "dry_run_passed")
        self.assertFalse(out.exists())

    def test_transient_windows_directory_lock_is_retried(self):
        out = self.root / "retry-release"
        real_replace = Path.replace
        attempts = []

        def replace_once_locked(path, target):
            if Path(target) == out and not attempts:
                attempts.append(path)
                raise PermissionError("transient directory lock")
            return real_replace(path, target)

        with mock.patch.object(Path, "replace", replace_once_locked), mock.patch.object(
            self.module.time, "sleep"
        ) as sleep:
            result = self.module.build_release(out, self._inputs())

        self.assertEqual(result["status"], "passed")
        self.assertTrue(out.is_dir())
        sleep.assert_called_once_with(1)

    @unittest.skipIf(gpd is None, "GeoPandas is unavailable in this Python environment")
    def test_geoparquet_geometry_and_crs_are_preserved(self):
        attributes = pd.read_parquet(self.parquet)
        geospatial = gpd.GeoDataFrame(
            attributes,
            geometry=[Point(835000, 815000), Point(835010, 815010)],
            crs="EPSG:2326",
        )
        geospatial.to_parquet(self.parquet, index=False)
        out = self.root / "geospatial-release"
        self.module.build_release(out, self._inputs())
        released = gpd.read_parquet(out / "HKBFETD_v2.parquet")
        self.assertEqual(released.crs.to_epsg(), 2326)
        self.assertTrue(released.geometry.equals(geospatial.geometry))

    def test_missing_required_input_fails_before_output_is_created(self):
        self.model.unlink()
        out = self.root / "missing-release"
        with self.assertRaises(FileNotFoundError):
            self.module.build_release(out, self._inputs())
        self.assertFalse(out.exists())

    def test_csv_parquet_value_mismatch_fails(self):
        changed = pd.read_csv(self.csv, dtype="string", encoding="utf-8-sig")
        changed.loc[0, "Height_m_建筑高度"] = "11"
        changed.to_csv(self.csv, index=False, encoding="utf-8-sig", na_rep="")
        out = self.root / "mismatch-release"
        with self.assertRaisesRegex(ValueError, "values differ"):
            self.module.build_release(out, self._inputs())
        self.assertFalse(out.exists())

    def test_validation_report_must_explicitly_pass(self):
        rejected = [
            {"status": "FAIL"},
            {"verdict": "BLOCKED_FOR_RELEASE"},
            {"result": "HOLD"},
            {"status": "error"},
            {"status": "not passed"},
            {"checks": {"rows_match": True}},
        ]
        for index, report in enumerate(rejected):
            with self.subTest(report=report):
                self.validation.write_text(json.dumps(report), encoding="utf-8")
                out = self.root / f"rejected-report-{index}"
                with self.assertRaisesRegex(ValueError, "report"):
                    self.module.build_release(out, self._inputs())
                self.assertFalse(out.exists())

    def test_geojson_empty_or_wrong_ids_fail(self):
        cases = [
            (["", "002"], "empty"),
            (["001", "001"], "duplicate"),
            (["001", "999"], "different"),
        ]
        for index, (ids, message) in enumerate(cases):
            with self.subTest(ids=ids):
                self._write_geojson(ids)
                out = self.root / f"bad-geojson-id-{index}"
                with self.assertRaisesRegex(ValueError, message):
                    self.module.build_release(out, self._inputs())
                self.assertFalse(out.exists())

    def test_geojson_non_wgs84_crs_fails(self):
        self._write_geojson(["001", "002"], crs="EPSG:2326")
        out = self.root / "bad-geojson-crs"
        with self.assertRaisesRegex(ValueError, "EPSG:4326"):
            self.module.build_release(out, self._inputs())
        self.assertFalse(out.exists())

    def test_missing_pyogrio_fails_explicitly(self):
        self.module.pyogrio = None
        out = self.root / "missing-pyogrio"
        with self.assertRaisesRegex(RuntimeError, "pyogrio is required"):
            self.module.build_release(out, self._inputs())
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
