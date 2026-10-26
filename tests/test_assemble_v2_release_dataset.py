from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "assemble_v2_release_dataset.py"
sys.path.insert(0, str(ROOT / "src"))

EXPECTED_PUBLIC_FIELDS = (
    ("Building_ID_建筑ID", "BUILDINGSTRUCTUREID"),
    ("Footprint_Area_sqm_占地面积", "Footprint_Area_sqm_占地面积"),
    ("Floors_Count_地上层数", "Floors_Count_地上层数"),
    ("Height_m_建筑高度", "Height_m_建筑高度"),
    ("UBEM_Main_Class_主类别", "Calibrated_Main_Class"),
    ("UBEM_Sub_Class_子类别", "Calibrated_Sub_Class"),
    ("UBEM_Mixed_Proportions_混合比例", "Calibrated_Mix_Proportion"),
    ("Classification_Source_分类来源", "Calibrated_Source"),
    ("Classification_Notes_分类备注", "Calibrated_Notes"),
    ("ML_Prediction_Confidence_机器学习置信度", "ML_Confidence"),
    ("ML_Prob_Residential_住宅概率", "proba_住宅类别"),
    ("ML_Prob_Commercial_商业概率", "proba_商业类别"),
    ("ML_Prob_Industrial_工业概率", "proba_工业类别"),
    ("ML_Height_Sensitivity_Status_高度缺失敏感性状态", "ML_Height_Sensitivity_Status"),
    (
        "Floor_Missing_Classification_Gate_层数缺失分类门",
        "Floor_Missing_Classification_Gate",
    ),
    ("LandsD_Structure_Type_地政署建筑类型", "BUILDINGSTRUCTURETYPE"),
    ("LandsD_Category_地政署分类", "CATEGORY"),
    ("LandsD_Status_地政署状态", "STATUS"),
    ("LandsD_Registered_GFA_地政署注册面积", "GROSSFLOORAREA"),
    ("LandsD_Building_Name_EN_地政署英文名", "BUILDINGNAMEEN"),
    ("LandsD_Building_Name_TC_地政署中文名", "BUILDINGNAMETC"),
    ("BDBIAR_Record_ID_屋宇署记录ID", "BDBIAR_OBJECTID"),
    ("BDBIAR_Classification_屋宇署分类", "BDBIAR_CLASS"),
    ("BDBIAR_Building_Age_Year_屋宇署楼龄年份", "BDBIAR_AGE"),
    ("BDBIAR_Address_TC_屋宇署中文地址", "BDBIAR_ADDRESS_C"),
    ("Dist_to_BDBIAR_m_距屋宇署点位距离", "distance_to_bdbiar"),
    ("BDBIAR_Match_Quality_屋宇署匹配质量", "BDBIAR_Match_Quality"),
    ("BDBIAR_Source_Attributes_Usable_屋宇署属性可采用", "BDBIAR_Source_Attributes_Usable"),
    ("BDBIAR_Source_Integrity_Status_屋宇署来源完整性", "BDBIAR_Source_Integrity_Status"),
    ("BDBIAR_Candidate_Record_ID_屋宇署候选记录ID", "BDBIAR_OBJECTID_CANDIDATE"),
    ("BDBIAR_Candidate_Classification_屋宇署候选分类", "BDBIAR_CLASS_CANDIDATE"),
    ("BDBIAR_Candidate_Building_Age_Year_屋宇署候选楼龄年份", "BDBIAR_AGE_CANDIDATE"),
    ("BDBIAR_Candidate_Address_TC_屋宇署候选中文地址", "BDBIAR_ADDRESS_C_CANDIDATE"),
    ("Crowdsourced_POIs_Matched_众包POI匹配详情", "Matched_POIs"),
    ("OSM_Record_ID_OSM记录ID", "osmid"),
    ("OSM_Building_Tag_OSM建筑标签", "building"),
    ("OSM_Amenity_Tag_OSM设施标签", "amenity"),
    ("OSM_Shop_Tag_OSM商铺标签", "shop"),
    ("OZP_Zone_Code_法定图则代码", "OZP_ZONE_LABEL"),
    ("OZP_Zone_Desc_EN_法定图则英文描述", "OZP_DESC_ENG"),
)


def load_module():
    spec = importlib.util.spec_from_file_location("assemble_v2_release_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AssembleV2ReleaseDatasetTests(unittest.TestCase):
    def test_field_type_recognises_nullable_object_boolean(self):
        self.assertEqual(
            self.module._field_type(pd.Series([True, pd.NA], dtype="object")),
            "boolean",
        )

    def setUp(self):
        self.module = load_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.classification = self.root / "classification.csv"
        self.physical = self.root / "physical.parquet"
        self.output = self.root / "assembled"
        self._write_inputs()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_inputs(self):
        m = self.module
        rows = []
        for index, building_id in enumerate(("001", "002"), start=1):
            row = {
                source: f"value-{index}-{source}"
                for _, source in m.PUBLIC_FIELDS
                if source not in m.PHYSICAL_CORE
            }
            row.update(
                {
                    m.RAW_ID: building_id,
                    m.CSUID: f"component-{index}",
                    "Calibrated_Main_Class": (
                        "Commercial_商业类别" if index == 1 else "Non-assessed_非评估类别"
                    ),
                    "Calibrated_Sub_Class": (
                        "Retail_零售"
                        if index == 1
                        else "Transport Infrastructure_交通基础设施"
                    ),
                    "Calibrated_Mix_Proportion": (
                        "Retail_零售:1.00"
                        if index == 1
                        else "Transport Infrastructure_交通基础设施:1.00"
                    ),
                    "Calibrated_Notes": "" if index == 1 else "verified scope",
                    "ML_Confidence": 0.91 if index == 1 else "",
                    "proba_住宅类别": 0.8 if index == 1 else "",
                    "proba_商业类别": 0.1 if index == 1 else "",
                    "proba_工业类别": 0.1 if index == 1 else "",
                    "CATEGORY": index,
                    "GROSSFLOORAREA": 900 + index,
                    "BDBIAR_AGE": 2000 + index,
                    "BDBIAR_Match_Quality": "inside_unique",
                    "BDBIAR_Source_Attributes_Usable": True,
                    "BDBIAR_Source_Integrity_Status": "",
                    "BDBIAR_OBJECTID_CANDIDATE": "",
                    "BDBIAR_CLASS_CANDIDATE": "",
                    "BDBIAR_AGE_CANDIDATE": "",
                    "BDBIAR_ADDRESS_C_CANDIDATE": "",
                    "distance_to_bdbiar": float(index),
                    "Classification_Source": "Keyword",
                    "LLM_Cache_Status": "",
                    "Structure_Use_Type": "" if index == 1 else "bridge_structure",
                    "Structure_Use_Name": "" if index == 1 else "Test Bridge",
                    "Structure_Use_Source": "" if index == 1 else "official",
                    "Structure_Use_Evidence": "" if index == 1 else "record-2",
                    "Structure_Use_Quality": "" if index == 1 else "verified",
                    "Energy_Assessment_Scope": "" if index == 1 else "excluded_nonbuilding",
                    # Intermediate fields must never be copied.
                    "Stage5_Repair_Action": "internal",
                    "Stage7_Repair_Action": "internal",
                    "ML_Main_Class": "candidate-only",
                }
            )
            rows.append(row)
        # Reverse the classification order to exercise key-based alignment.
        pd.DataFrame(rows[::-1]).to_csv(
            self.classification, index=False, encoding="utf-8-sig"
        )

        physical = gpd.GeoDataFrame(
            {
                m.ID: ["001", "002"],
                "Footprint_Area_sqm_占地面积": [101.0, 202.0],
                "Floors_Count_地上层数": [10.0, pd.NA],
                "Height_m_建筑高度": [31.5, 12.0],
                "LandsD_Structure_Type_地政署建筑类型": ["physical-a", "physical-b"],
                "LandsD_Category_地政署分类": [90, 91],
                "LandsD_Status_地政署状态": ["physical", "physical"],
                "LandsD_Registered_GFA_地政署注册面积": [9999.0, 8888.0],
                m.CSUID: ["component-1", "component-2"],
                "Height_Source": ["survey", "survey"],
                "Height_Quality": ["verified", "verified"],
                "Physical_Review_Flags": ["", "nonbuilding"],
                "Permit_GFA_Usable": [True, False],
            },
            geometry=[
                # Clockwise on purpose: RFC 7946 output must reverse this outer ring.
                Polygon([(835000, 815000), (835000, 815010), (835010, 815010), (835010, 815000)]),
                Polygon([(835020, 815020), (835030, 815020), (835030, 815030), (835020, 815030)]),
            ],
            crs="EPSG:2326",
        )
        physical.to_parquet(self.physical, index=False)

    def test_assembles_all_formats_with_authoritative_physical_values(self):
        m = self.module
        manifest = m.assemble_release_dataset(
            self.classification, self.physical, self.output
        )
        self.assertEqual(
            sorted(path.name for path in self.output.iterdir()),
            [
                "HKBFETD_v2.csv",
                "HKBFETD_v2.geojson",
                "HKBFETD_v2.parquet",
                "assembly_manifest.json",
                "classification_integrity_audit.csv",
                "classification_integrity_manifest.json",
                "schema.json",
            ],
        )
        parquet = gpd.read_parquet(self.output / "HKBFETD_v2.parquet")
        csv_frame = pd.read_csv(
            self.output / "HKBFETD_v2.csv",
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        csv_frame = csv_frame.mask(csv_frame.eq(""), pd.NA)
        self.assertEqual(m.PUBLIC_FIELDS, EXPECTED_PUBLIC_FIELDS)
        public_names = [output for output, _ in m.PUBLIC_FIELDS]
        self.assertEqual(parquet.columns[: len(public_names)].tolist(), public_names)
        self.assertEqual(parquet.crs.to_epsg(), 2326)
        self.assertEqual(parquet[m.ID].tolist(), ["001", "002"])
        self.assertEqual(parquet["Height_m_建筑高度"].tolist(), [31.5, 12.0])
        self.assertEqual(
            parquet.loc[0, "LandsD_Structure_Type_地政署建筑类型"],
            "value-1-BUILDINGSTRUCTURETYPE",
        )
        self.assertEqual(
            parquet["UBEM_Main_Class_主类别"].tolist(),
            ["Commercial_商业类别", "Non-assessed_非评估类别"],
        )
        integrity = json.loads(
            (self.output / "classification_integrity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(integrity["changed_rows"], 0)
        self.assertEqual(integrity["remaining_changed_rows_after_gate"], 0)
        self.assertNotIn("Stage5_Repair_Action", parquet.columns)
        self.assertNotIn("Stage7_Repair_Action", parquet.columns)
        self.assertNotIn("ML_Main_Class", parquet.columns)
        self.assertIn("Height_Source", parquet.columns)
        self.assertIn("Permit_GFA_Usable", parquet.columns)
        physical_columns = set(gpd.read_parquet(self.physical).columns)
        expected_physical_extras = physical_columns - m.PHYSICAL_PUBLIC_FIELDS - {m.GEOMETRY}
        self.assertTrue(expected_physical_extras.issubset(parquet.columns))
        self.assertTrue(set(m.STRUCTURE_SCOPE_FIELDS).issubset(parquet.columns))
        self.assertEqual(parquet.loc[1, "Structure_Use_Type"], "bridge_structure")
        self.assertTrue(pd.isna(parquet.loc[0, "Physical_Review_Flags"]))
        self.assertTrue(pd.isna(csv_frame.loc[0, "Physical_Review_Flags"]))
        self.assertTrue(pd.isna(parquet.loc[0, "Classification_Notes_分类备注"]))
        self.assertTrue(pd.isna(csv_frame.loc[0, "Classification_Notes_分类备注"]))
        for column in csv_frame.columns:
            self.assertTrue(
                csv_frame[column].isna().equals(parquet[column].isna()),
                f"CSV/Parquet null masks differ for {column}",
            )
            present = parquet[column].notna()
            if pd.api.types.is_bool_dtype(parquet[column].dtype):
                self.assertEqual(
                    csv_frame.loc[present, column].str.casefold().tolist(),
                    parquet.loc[present, column].astype(str).str.casefold().tolist(),
                    f"CSV/Parquet boolean values differ for {column}",
                )
            elif pd.api.types.is_numeric_dtype(parquet[column].dtype):
                self.assertTrue(
                    np.allclose(
                        pd.to_numeric(csv_frame.loc[present, column]).to_numpy(float),
                        pd.to_numeric(parquet.loc[present, column]).to_numpy(float),
                    ),
                    f"CSV/Parquet numeric values differ for {column}",
                )
            else:
                self.assertEqual(
                    csv_frame.loc[present, column].astype(str).tolist(),
                    parquet.loc[present, column].astype(str).tolist(),
                    f"CSV/Parquet values differ for {column}",
                )

        geojson_text = (self.output / "HKBFETD_v2.geojson").read_text(encoding="utf-8")
        self.assertNotIn('"crs"', geojson_text)
        self.assertNotIn("NaN", geojson_text)
        self.assertNotIn("Infinity", geojson_text)
        def reject_constant(value):
            raise ValueError(f"Non-standard JSON constant: {value}")

        geojson = json.loads(geojson_text, parse_constant=reject_constant)
        self.assertEqual(len(geojson["features"]), 2)
        for feature in geojson["features"]:
            self.assertEqual(set(feature["properties"]), set(csv_frame.columns))
        self.assertIsNone(
            geojson["features"][0]["properties"]["Physical_Review_Flags"]
        )
        longitude = geojson["features"][0]["geometry"]["coordinates"][0][0][0]
        self.assertGreater(longitude, 113)
        self.assertLess(longitude, 115)
        ring = geojson["features"][0]["geometry"]["coordinates"][0]
        signed_area = sum(
            ring[index][0] * ring[index + 1][1]
            - ring[index + 1][0] * ring[index][1]
            for index in range(len(ring) - 1)
        ) / 2
        self.assertGreater(signed_area, 0, "RFC 7946 exterior ring must be counterclockwise")

        saved_manifest = json.loads(
            (self.output / "assembly_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_manifest["join_validation"]["rows"], 2)
        self.assertEqual(saved_manifest["join_validation"]["structure_scope_rows_retained"], 1)
        self.assertEqual(saved_manifest["inputs"]["classification_csv"]["rows"], 2)
        self.assertGreater(
            saved_manifest["inputs"]["classification_csv"]["source_columns"],
            saved_manifest["inputs"]["classification_csv"]["consumed_columns"],
        )
        self.assertEqual(
            saved_manifest["inputs"]["verified_physical_geoparquet"][
                "null_counts"
            ]["Physical_Review_Flags"],
            1,
        )
        self.assertEqual(manifest["outputs"], saved_manifest["outputs"])
        for filename, record in saved_manifest["outputs"].items():
            digest = hashlib.sha256((self.output / filename).read_bytes()).hexdigest()
            self.assertEqual(record["sha256"], digest)
        for filename in ("HKBFETD_v2.csv", "HKBFETD_v2.parquet", "HKBFETD_v2.geojson"):
            self.assertEqual(saved_manifest["outputs"][filename]["rows"], 2)
            self.assertEqual(
                saved_manifest["outputs"][filename]["attribute_columns"],
                len(csv_frame.columns),
            )
            self.assertEqual(
                saved_manifest["outputs"][filename]["null_counts"][
                    "Physical_Review_Flags"
                ],
                1,
            )

    def test_rejects_component_mismatch_without_partial_output(self):
        physical = gpd.read_parquet(self.physical)
        physical.loc[1, self.module.CSUID] = "wrong-component"
        physical.to_parquet(self.physical, index=False)
        with self.assertRaisesRegex(ValueError, "BUILDINGCSUID differs"):
            self.module.assemble_release_dataset(
                self.classification, self.physical, self.output
            )
        self.assertFalse(self.output.exists())

    def test_rejects_non_metric_physical_geometry(self):
        physical = gpd.read_parquet(self.physical).to_crs("EPSG:4326")
        physical.to_parquet(self.physical, index=False)
        with self.assertRaisesRegex(ValueError, "EPSG:2326"):
            self.module.assemble_release_dataset(
                self.classification, self.physical, self.output
            )
        self.assertFalse(self.output.exists())

    def test_rejects_adopted_attributes_from_unsafe_bdbiar_match(self):
        classification = pd.read_csv(
            self.classification,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        classification.loc[0, "BDBIAR_Match_Quality"] = "nearby_only"
        classification.loc[0, "BDBIAR_Source_Attributes_Usable"] = "False"
        classification.to_csv(self.classification, index=False, encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "Unsafe BDBIAR"):
            self.module.assemble_release_dataset(
                self.classification, self.physical, self.output
            )
        self.assertFalse(self.output.exists())

    def test_fail_closes_dependent_classification_without_changing_main(self):
        m = self.module
        classification = pd.read_csv(
            self.classification,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        row = classification[m.RAW_ID].eq("001")
        classification.loc[row, "Calibrated_Sub_Class"] = "Private Housing_私人房屋"
        classification.loc[row, "Calibrated_Mix_Proportion"] = (
            "Private Housing_私人房屋:1.00"
        )
        classification.to_csv(self.classification, index=False, encoding="utf-8-sig")

        manifest = m.assemble_release_dataset(
            self.classification, self.physical, self.output
        )
        result = pd.read_csv(
            self.output / "HKBFETD_v2.csv",
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        published = result.loc[result[m.ID].eq("001")].iloc[0]
        self.assertEqual(published["UBEM_Main_Class_主类别"], "Commercial_商业类别")
        self.assertEqual(published["UBEM_Sub_Class_子类别"], "Unknown_未知类别")
        self.assertEqual(published["UBEM_Mixed_Proportions_混合比例"], "")
        self.assertEqual(manifest["classification_integrity"]["fail_closed_rows"], 1)


if __name__ == "__main__":
    unittest.main()
