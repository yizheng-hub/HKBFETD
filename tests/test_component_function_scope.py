import json
import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from classification import COMMERCIAL, RESIDENTIAL, UNKNOWN, enrich_function_evidence


class ComponentFunctionScopeTests(unittest.TestCase):
    def base(self, ids):
        return pd.DataFrame({
            "BUILDINGSTRUCTUREID": ids,
            "Final_Main_Class": [UNKNOWN] * len(ids),
            "Final_Sub_Class": [UNKNOWN] * len(ids),
            "Classification_Source": ["Unknown"] * len(ids),
            "Classification_QA": [""] * len(ids),
            "Is_Conflicted": [False] * len(ids),
            "Sub_Class_Proportions": [""] * len(ids),
        })

    def rows(self, targets, tags, ratios, areas=None):
        areas = areas or [100.] * len(targets)
        return pd.DataFrame({
            "BUILDINGSTRUCTUREID": targets,
            "Source_Object_ID": ["osm:way:shared"] * len(targets),
            "osm_area": [100.] * len(targets),
            "SHAPE_Area": areas,
            "overlap_ratio": ratios,
            "building": tags,
        })

    def station(self, base, name="Oi Shun House 'A' Substation", csuid="csdi:one", iou=.999):
        base = base.copy()
        base["CSDI_Component_Use_Type"] = "substation"
        base["CSDI_Building_Name_EN"] = name
        base["CSDI_Building_Name_TC"] = "愛信樓Ａ變電站"
        base["BUILDINGCSUID"] = csuid
        base["CSDI_Geometry_IoU"] = iou
        base["CSDI_Match_Quality"] = "same_component_id_type_footprint"
        base["CSDI_Status"] = "Active"
        return base

    def empty_osm(self):
        return pd.DataFrame(columns=["BUILDINGSTRUCTUREID", "Source_Object_ID", "osm_area", "SHAPE_Area", "overlap_ratio", "building"])

    def test_exact_component_club_and_hall_names_do_not_become_housing(self):
        base = self.base(["club", "hall", "open_club"])
        base["CSDI_Building_Name_EN"] = ["Residents Club", "Assembly Hall", "RESIDENTS CLUB"]
        base["CSDI_Building_Name_TC"] = ["", "禮堂", "住戶俱樂部"]
        base["CSDI_Match_Quality"] = "same_component_id_type_footprint"
        base["CSDI_Status"] = "Active"
        base["CSDI_Geometry_IoU"] = .999
        result = enrich_function_evidence(base, self.empty_osm())
        self.assertTrue(result.Final_Main_Class.eq(COMMERCIAL).all())
        self.assertTrue(result.Final_Sub_Class.eq("Other Commercial_其他商业").all())
        self.assertTrue(result.Function_Quality.eq("source_inferred").all())
        self.assertTrue(result.Observed_Residential_Area_Share.isna().all())
        for value in result.CSDI_Name_Function_Evidence:
            self.assertEqual(json.loads(value)["component_name_rule"], "community_hall_or_club")

    def test_club_hall_rule_requires_whole_name_and_active_component_match(self):
        base = self.base(["compound", "school", "changed", "inactive", "distant"])
        base["CSDI_Building_Name_EN"] = ["Residents Club Tower", "School and Residents Club",
                                          "Assembly Hall", "Residents Club", "Residents Club"]
        base["CSDI_Building_Name_TC"] = ""
        base["CSDI_Match_Quality"] = ["same_component_id_type_footprint"] * 4 + ["footprint_changed"]
        base["CSDI_Status"] = ["Active", "Active", "Active", "Demolished", "Active"]
        base["CSDI_Geometry_IoU"] = [.999, .999, .90, .999, .999]
        result = enrich_function_evidence(base, self.empty_osm())
        self.assertFalse(result.Final_Sub_Class.eq("Other Commercial_其他商业").any())

    def test_same_retail_object_with_strong_and_weak_component_matches_is_excluded(self):
        osm = self.rows(["tower", "podium"], ["retail", "retail"], [1., .1])
        result = enrich_function_evidence(self.base(["tower", "podium"]), osm)
        self.assertEqual(result.Final_Main_Class.tolist(), [UNKNOWN, UNKNOWN])
        self.assertEqual(result.OSM_Function_Candidate_Main.tolist(), [COMMERCIAL, UNKNOWN])
        self.assertFalse(result.Is_Conflicted.any())
        self.assertTrue(result.Classification_QA.str.contains("function_scope_ambiguous").all())
        tower_evidence = json.loads(result.loc[result.BUILDINGSTRUCTUREID.eq("tower"), "OSM_Function_Evidence"].iloc[0])
        retail = next(item for item in tower_evidence if item["source_object_id"] == "osm:way:shared")
        self.assertEqual(retail["raw_matched_component_count"], 2)
        self.assertFalse(retail["adoption_eligible"])
        self.assertEqual(retail["scope_status"], "shared_component_scope")

    def test_residential_composite_tower_keeps_retail_tag_as_review_candidate(self):
        base = self.base(["tower"])
        base["BUILDINGSTRUCTURETYPE"] = "T"
        base["BDBIAR_CLASS"] = "住宅/綜合用途"
        base["BDBIAR_Match_Quality"] = "inside_unique"
        osm = self.rows(["tower"], ["retail"], [0.7108], [570.05])
        osm["osm_area"] = 546.1
        result = enrich_function_evidence(base, osm).iloc[0]
        self.assertEqual(result.Final_Main_Class, UNKNOWN)
        self.assertEqual(result.OSM_Function_Candidate_Main, COMMERCIAL)
        self.assertTrue(result.Is_Conflicted)
        self.assertIn("function_evidence_conflict", result.Classification_QA)

    def test_ambiguous_object_does_not_block_separate_valid_unique_object(self):
        osm = self.rows(["tower", "podium", "tower"], ["retail", "retail", "apartments"], [1., .1, 1.])
        osm["Source_Object_ID"] = ["osm:way:shared", "osm:way:shared", "osm:way:unique"]
        result = enrich_function_evidence(self.base(["tower", "podium"]), osm)
        self.assertEqual(result.loc[result.BUILDINGSTRUCTUREID.eq("tower"), "Final_Main_Class"].iloc[0], RESIDENTIAL)
        self.assertEqual(result.loc[result.BUILDINGSTRUCTUREID.eq("podium"), "Final_Main_Class"].iloc[0], UNKNOWN)
        self.assertTrue(result.Classification_QA.str.contains("function_scope_ambiguous").all())

    def test_ambiguous_object_does_not_clear_known_independent_function(self):
        base = self.base(["tower", "podium"])
        base.loc[0, "Final_Main_Class"] = RESIDENTIAL
        base.loc[0, "Classification_Source"] = "Documented_component_function"
        osm = self.rows(["tower", "podium"], ["retail", "retail"], [1., .1])
        result = enrich_function_evidence(base, osm).iloc[0]
        self.assertEqual(result.Final_Main_Class, RESIDENTIAL)
        self.assertEqual(result.Classification_Source, "Documented_component_function")
        self.assertFalse(result.Is_Conflicted)
        self.assertIn("function_scope_ambiguous", result.Classification_QA)

    def test_unique_building_and_part_function_still_donates(self):
        osm = self.rows(["one", "one"], ["apartments", "apartments"], [1., 1.])
        osm["Source_Object_ID"] = ["osm:way:whole", "osm:way:part"]
        result = enrich_function_evidence(self.base(["one"]), osm).iloc[0]
        self.assertEqual(result.Final_Main_Class, RESIDENTIAL)
        self.assertFalse(result.Is_Conflicted)
        evidence = json.loads(result.OSM_Function_Evidence)
        self.assertTrue(all(item["adoption_eligible"] for item in evidence))
        self.assertEqual({item["source_object_id"] for item in evidence}, {"osm:way:whole", "osm:way:part"})

    def test_nonresidential_bdbiar_label_is_not_treated_as_residential(self):
        base = self.base(["one"])
        base["BDBIAR_CLASS"] = "Non-residential"
        base["BDBIAR_Match_Quality"] = "inside_unique"
        osm = self.rows(["one"], ["retail"], [1.])
        result = enrich_function_evidence(base, osm).iloc[0]
        self.assertEqual(result.Final_Main_Class, COMMERCIAL)
        self.assertFalse(result.Is_Conflicted)

    def test_explicit_station_name_overrides_host_residential_tag_and_records_before_state(self):
        base = self.base(["station"])
        base["Final_Main_Class"] = RESIDENTIAL
        base["Final_Sub_Class"] = "Private Housing_私人房屋"
        base["Classification_Source"] = "Keyword"
        base["Classification_QA"] = "function_evidence_conflict;subclass_parent_conflict;keep_this_flag"
        base["Is_Conflicted"] = True
        result = enrich_function_evidence(self.station(base), self.rows(["station"], ["apartments"], [1.])).iloc[0]

        self.assertEqual(result.Final_Main_Class, COMMERCIAL)
        self.assertEqual(result.Final_Sub_Class, "Other Commercial_其他商业")
        self.assertEqual(result.Classification_Source, "CSDI_Component_Name_Keyword")
        self.assertEqual(result.OSM_Function_Candidate_Main, RESIDENTIAL)
        self.assertTrue(result.Is_Conflicted)
        self.assertEqual(result.Classification_QA, "keep_this_flag")
        evidence = json.loads(result.CSDI_Name_Function_Evidence)
        self.assertEqual(evidence["csuid"], "csdi:one")
        self.assertEqual(evidence["geometry_iou"], .999)
        self.assertEqual(evidence["before"]["main"], RESIDENTIAL)
        self.assertEqual(evidence["before"]["sub"], "Private Housing_私人房屋")
        self.assertEqual(evidence["before"]["classification_source"], "OSM_Building_Function+Keyword")
        self.assertEqual(evidence["before"]["classification_qa"], "function_evidence_conflict;subclass_parent_conflict;keep_this_flag")
        self.assertEqual(evidence["frozen_source"]["path"], "data/sources/csdi_official_20261020.parquet")

    def test_explicit_station_name_overrides_school_and_medical_osm_subclasses(self):
        base = self.base(["school", "clinic"])
        base["Final_Main_Class"] = [RESIDENTIAL, RESIDENTIAL]
        base["Final_Sub_Class"] = ["Unknown_未知类别", "Unknown_未知类别"]
        base = self.station(base)
        osm = self.rows(["school", "clinic"], ["school", "hospital"], [1., 1.])
        result = enrich_function_evidence(base, osm)
        self.assertEqual(result.Final_Main_Class.tolist(), [COMMERCIAL, COMMERCIAL])
        self.assertEqual(result.Final_Sub_Class.tolist(), ["Other Commercial_其他商业"] * 2)
        self.assertEqual(result.Classification_Source.tolist(), ["CSDI_Component_Name_Keyword"] * 2)

    def test_station_evidence_handles_native_integer_id_and_absent_prior_qa(self):
        base=self.station(self.base(['station'])).drop(columns=['Classification_QA','Is_Conflicted'])
        base['CSDI_Native_Building_ID']=123
        result=enrich_function_evidence(base,self.empty_osm()).iloc[0]
        self.assertEqual(json.loads(result.CSDI_Name_Function_Evidence)['native_building_id'],123)
        self.assertEqual(result.Final_Sub_Class,'Other Commercial_其他商业')

    def test_station_rule_clears_only_resolved_function_conflict(self):
        base = self.base(["station"])
        base["Final_Main_Class"] = RESIDENTIAL
        base["Classification_QA"] = "function_scope_ambiguous;function_evidence_conflict;subclass_parent_conflict;registered_area_function_conflict;permit_component_scope_unresolved;keep_this_flag"
        base["Is_Conflicted"] = True
        result = enrich_function_evidence(self.station(base), self.empty_osm()).iloc[0]
        self.assertTrue(result.Is_Conflicted)
        self.assertNotIn("function_scope_ambiguous", result.Classification_QA)
        self.assertNotIn("function_evidence_conflict", result.Classification_QA)
        self.assertNotIn("subclass_parent_conflict", result.Classification_QA)
        self.assertIn("registered_area_function_conflict", result.Classification_QA)
        self.assertIn("permit_component_scope_unresolved", result.Classification_QA)
        self.assertIn("keep_this_flag", result.Classification_QA)
        evidence=json.loads(result.CSDI_Name_Function_Evidence)
        self.assertEqual(evidence["before"]["classification_qa"],base.iloc[0].Classification_QA)

    def test_station_with_only_resolved_scope_flags_is_no_longer_conflicted(self):
        base = self.base(["station"])
        base["Classification_QA"] = "function_scope_ambiguous;function_evidence_conflict;subclass_parent_conflict"
        base["Is_Conflicted"] = True
        result = enrich_function_evidence(self.station(base), self.empty_osm()).iloc[0]
        self.assertEqual(result.Classification_QA, "")
        self.assertFalse(result.Is_Conflicted)
        self.assertEqual(json.loads(result.CSDI_Name_Function_Evidence)["before"]["classification_qa"], base.iloc[0].Classification_QA)

    def test_station_rule_requires_explicit_marker_and_valid_component_match(self):
        base = self.base(["no_marker", "low_iou", "bad_match"])
        base["Final_Main_Class"] = [RESIDENTIAL] * 3
        base["Final_Sub_Class"] = ["Unknown_未知类别"] * 3
        base["Classification_QA"] = "function_scope_ambiguous;keep_this_flag"
        base["Classification_Source"] = ["Keyword"] * 3
        base = self.station(base)
        base.loc[0, "CSDI_Component_Use_Type"] = ""
        base.loc[1, "CSDI_Geometry_IoU"] = .94
        base.loc[2, "CSDI_Match_Quality"] = "footprint_changed"
        result = enrich_function_evidence(base, self.empty_osm())
        self.assertEqual(result.Final_Main_Class.tolist(), [RESIDENTIAL] * 3)
        self.assertEqual(result.Classification_Source.tolist(), ["Keyword"] * 3)
        self.assertEqual(result.CSDI_Name_Function_Evidence.tolist(), ["[]"] * 3)
        self.assertTrue(result.Classification_QA.str.contains("function_scope_ambiguous").all())

    def test_disabled_component_scoped_residential_permit_does_not_force_mixed(self):
        base = self.base(["station"])
        base["Final_Main_Class"] = RESIDENTIAL
        base["Final_Sub_Class"] = "Unknown_未知类别"
        base["GFA_DOMESTIC_SUM"] = 120.
        base["GFA_NONDOMESTIC_SUM"] = 0.
        base["Permit_GFA_Usable"] = False
        base["Permit_GFA_Scope"] = "single_structure_component_use_scope_unresolved"
        result = enrich_function_evidence(self.station(base), self.empty_osm()).iloc[0]
        self.assertEqual(result.Final_Main_Class, COMMERCIAL)
        self.assertEqual(result.Final_Sub_Class, "Other Commercial_其他商业")
        self.assertEqual(result.GFA_DOMESTIC_SUM, 120.)
        self.assertFalse(result.Permit_GFA_Usable)
        self.assertTrue(pd.isna(result.Observed_Residential_Area_Share))
        self.assertIn("permit_component_scope_unresolved", result.Classification_QA)


if __name__ == "__main__":
    unittest.main()
