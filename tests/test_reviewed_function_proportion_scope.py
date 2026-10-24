import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from classification import COMMERCIAL, RESIDENTIAL, MIXED, UNKNOWN, enrich_function_evidence, finalize_function_fields
from utils import init_keyword_tool


def empty_osm():
    return pd.DataFrame(columns=["BUILDINGSTRUCTUREID", "Source_Object_ID", "osm_area",
                                 "SHAPE_Area", "overlap_ratio", "building"])


def matched_component(name, main=UNKNOWN, sub=UNKNOWN):
    return pd.DataFrame({
        "BUILDINGSTRUCTUREID": ["component-1"], "BUILDINGCSUID": ["csdi:1"],
        "Final_Main_Class": [main], "Final_Sub_Class": [sub],
        "Classification_Source": ["Parent_Name"], "Classification_QA": [""],
        "Is_Conflicted": [False], "Sub_Class_Proportions": [""],
        "CSDI_Building_Name_EN": [name], "CSDI_Building_Name_TC": [""],
        "CSDI_Match_Quality": ["same_component_id_type_footprint"],
        "CSDI_Status": ["Active"], "CSDI_Geometry_IoU": [.99],
        "CSDI_Component_Use_Type": [""],
    })


class ReviewedFunctionProportionScopeTests(unittest.TestCase):
    def test_permit_accounting_does_not_create_use_class_or_measured_share(self):
        frame = pd.DataFrame({
            "Final_Main_Class": [UNKNOWN], "Final_Sub_Class": [UNKNOWN],
            "GFA_DOMESTIC_SUM": [1200.], "GFA_NONDOMESTIC_SUM": [800.],
            "Permit_GFA_Usable": [True], "Permit_GFA_Scope": ["single_structure"],
        })
        row = finalize_function_fields(frame).iloc[0]
        self.assertEqual(row.Final_Main_Class, UNKNOWN)
        self.assertEqual(row.GFA_DOMESTIC_SUM, 1200.)
        self.assertEqual(row.GFA_NONDOMESTIC_SUM, 800.)
        self.assertTrue(pd.isna(row.Observed_Residential_Area_Share))
        self.assertTrue(pd.isna(row.Observed_Nondomestic_Area_Share))
        self.assertEqual(row.Observed_Main_Area_Proportions, "")
        self.assertIn("permit_accounting_not_observed_use", row.Classification_QA)
        self.assertIn("permit_physical_scope_unverified", row.Classification_QA)

    def test_permit_accounting_does_not_overwrite_existing_component_function(self):
        frame = pd.DataFrame({
            "Final_Main_Class": [RESIDENTIAL], "Final_Sub_Class": ["Private Housing_私人房屋"],
            "GFA_DOMESTIC_SUM": [600.], "GFA_NONDOMESTIC_SUM": [400.],
            "Permit_GFA_Usable": [True],
        })
        row = finalize_function_fields(frame).iloc[0]
        self.assertEqual(row.Final_Main_Class, RESIDENTIAL)
        self.assertEqual(row.Proportion_Method, "single_function_assumption")
        self.assertEqual(row.Proportion_Quality, "assumed_not_measured")
        self.assertTrue(pd.isna(row.Observed_Residential_Area_Share))
        self.assertEqual(row.Sub_Class_Proportions, "Private Housing_私人房屋:1.00000000")

    def test_exact_component_carpark_name_overrides_parent_but_combined_name_does_not(self):
        pure = enrich_function_evidence(matched_component("North Car Park", RESIDENTIAL), empty_osm()).iloc[0]
        self.assertEqual(pure.Final_Main_Class, COMMERCIAL)
        self.assertEqual(pure.Final_Sub_Class, "Other Commercial_其他商业")
        self.assertEqual(pure.Classification_Source, "CSDI_Component_Name_Keyword")
        self.assertEqual(pure.Function_Quality, "source_inferred")

        combined = enrich_function_evidence(matched_component("Shopping Centre and Car Park", RESIDENTIAL,
                                                               "Private Housing_私人房屋"), empty_osm()).iloc[0]
        self.assertEqual(combined.Final_Main_Class, RESIDENTIAL)
        self.assertEqual(combined.Classification_Source, "Parent_Name")
        self.assertEqual(combined.CSDI_Name_Function_Evidence, "[]")

    def test_component_name_requires_active_exact_geometry_match(self):
        frame = matched_component("Student Hostel", COMMERCIAL)
        frame.loc[0, "CSDI_Match_Quality"] = "inside_unique"
        row = enrich_function_evidence(frame, empty_osm()).iloc[0]
        self.assertEqual(row.Final_Main_Class, COMMERCIAL)
        self.assertEqual(row.CSDI_Name_Function_Evidence, "[]")

    def test_school_token_does_not_bypass_combined_carpark_exclusion(self):
        row=enrich_function_evidence(matched_component('College Shopping Centre and Car Park'),empty_osm()).iloc[0]
        self.assertNotEqual(row.Final_Sub_Class,'Other Commercial_其他商业')
        self.assertNotIn('car_park',row.CSDI_Name_Function_Evidence)

    def test_residence_name_requires_student_semantics_but_explicit_hall_is_adopted(self):
        postgraduate = matched_component("Postgraduate Hall II")
        postgraduate.loc[0, "CSDI_Building_Name_TC"] = "研究生宿舍"
        row = enrich_function_evidence(postgraduate, empty_osm()).iloc[0]
        self.assertEqual(row.Final_Main_Class, RESIDENTIAL)
        self.assertEqual(row.Final_Sub_Class, "Other Housing_其他房屋")
        self.assertEqual(row.Function_Quality, "source_inferred")

        student_hostel = enrich_function_evidence(matched_component("University Student Hostel"), empty_osm()).iloc[0]
        self.assertEqual(student_hostel.Final_Main_Class, RESIDENTIAL)

        generic_hostel = enrich_function_evidence(matched_component("Harbour Hostel"), empty_osm()).iloc[0]
        self.assertEqual(generic_hostel.Final_Main_Class, COMMERCIAL)
        self.assertEqual(generic_hostel.Final_Sub_Class, "Accommodation_住宿")

    def test_university_parent_token_does_not_override_named_child_facility(self):
        for name in ["University Student Centre", "University Clinic", "University Hospital",
                     "University Care Home", "University Nursing Home", "University Church"]:
            with self.subTest(name=name):
                row = enrich_function_evidence(matched_component(name), empty_osm()).iloc[0]
                self.assertEqual(row.Final_Main_Class, UNKNOWN)
                self.assertEqual(row.CSDI_Name_Function_Evidence, "[]")

        school = enrich_function_evidence(matched_component("University Secondary School"), empty_osm()).iloc[0]
        self.assertEqual(school.Final_Main_Class, COMMERCIAL)
        self.assertEqual(school.Final_Sub_Class, "Education_教育")

    def test_rule_engine_does_not_use_permit_mix_or_aggregate_resort_as_function(self):
        path = SRC / "3.rule_engine.py"
        spec = importlib.util.spec_from_file_location("reviewed_function_rule_engine", path)
        rule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rule)
        frame = pd.DataFrame({
            "BUILDINGSTRUCTUREID": ["1"], "osmid": [np.nan], "CATEGORY": ["9"],
            "BUILDINGSTRUCTURETYPE": ["T"], "BUILDINGNAMEEN": ["Island Resort"],
            "BUILDINGNAMETC": [""], "SHAPE_Area": [100.],
            "GFA_DOMESTIC_SUM": [1000.], "GFA_NONDOMESTIC_SUM": [1000.],
            "Permit_GFA_Usable": [True],
        })
        row = rule.classify_building_rule_engine_optimized(frame, init_keyword_tool())
        self.assertEqual(row.Final_Main_Class, "未知类别")
        self.assertNotEqual(row.Classification_Source, "GFA_Mixed")

    def test_religious_school_name_exception_is_case_insensitive(self):
        name = 'THE CHURCH OF CHRIST IN CHINA HOH FUK TONG COLLEGE'
        for text in [name, name.lower(), name.title(), 'CHURCH SECONDARY SCHOOL']:
            with self.subTest(name=text):
                row = enrich_function_evidence(matched_component(text), empty_osm()).iloc[0]
                self.assertEqual(row.Final_Main_Class, COMMERCIAL)
                self.assertEqual(row.Final_Sub_Class, 'Education_教育')
                self.assertEqual(row.Classification_Source, 'CSDI_Component_Name_Keyword')

    def test_school_dental_clinic_is_not_overridden_by_school_exception(self):
        for name in ['Tuen Mun School Dental Clinic',
                     'PAMELA YOUDE CHILD ASSESSMENT CENTRE AND SCHOOL DENTAL CLINIC',
                     'Argyle Street Jockey Club School Dental Clinic',
                     'MacLehose Dental Centre Tang Shiu Kin School Dental Clinic']:
            with self.subTest(name=name):
                row=enrich_function_evidence(matched_component(name,COMMERCIAL,'Human Health_医疗'),empty_osm()).iloc[0]
                self.assertEqual(row.Final_Sub_Class,'Human Health_医疗')
                self.assertEqual(row.Classification_Source,'Parent_Name')
        sponsor=enrich_function_evidence(matched_component('YAN CHAI HOSPITAL CHAN IU SENG PRIMARY SCHOOL'),empty_osm()).iloc[0]
        self.assertEqual(sponsor.Final_Sub_Class,'Education_教育')

    def test_residence_child_facilities_do_not_become_housing(self):
        for name in ['Student Residence City University Student Residence Office',
                     'Student Residence City University Multi-function Hall B',
                     'The Education University Senior Staff Quarters Refuse Store Room']:
            with self.subTest(name=name):
                row=enrich_function_evidence(matched_component(name,COMMERCIAL,'Education_教育'),empty_osm()).iloc[0]
                self.assertEqual(row.Final_Main_Class,UNKNOWN)
                self.assertEqual(row.Classification_Source,'CSDI_Component_Name_Scope_Unresolved')
                self.assertIn('function_evidence_conflict',row.Classification_QA)

    def test_explicit_combined_names_do_not_force_one_function(self):
        for name in ['Princess Margaret Hospital School of General Nursing & Quarters',
                     'QUARTERS CUM SCHOOL BLDG','Sheltered Workshop and Hostel']:
            with self.subTest(name=name):
                frame=matched_component(name,COMMERCIAL,'Education_教育')
                row=enrich_function_evidence(frame,empty_osm()).iloc[0]
                self.assertEqual(row.Final_Main_Class,UNKNOWN)
                self.assertEqual(row.Proportion_Quality,'unknown')
                self.assertIn('function_scope_ambiguous',row.Classification_QA)
        residence=matched_component('College Staff Quarters')
        residence.loc[0,'CSDI_Building_Name_TC']='學院教職員宿舍'
        self.assertEqual(enrich_function_evidence(residence,empty_osm()).iloc[0].Final_Main_Class,RESIDENTIAL)

    def test_existing_mixed_component_is_not_collapsed_by_name(self):
        row=enrich_function_evidence(matched_component('University Student Residential & Amenities Complex',MIXED),empty_osm()).iloc[0]
        self.assertEqual(row.Final_Main_Class,MIXED)

    def test_sponsor_hospital_name_does_not_determine_student_union_room_use(self):
        frame=matched_component('Yan Chai Hospital No.2 Secondary School Student Association Room',COMMERCIAL,'Human Health_医疗')
        row=enrich_function_evidence(frame,empty_osm()).iloc[0]
        self.assertEqual(row.Final_Main_Class,UNKNOWN)
        self.assertIn('function_scope_ambiguous',row.Classification_QA)


if __name__ == "__main__":
    unittest.main()
