import importlib.util
import sys
import types
from pathlib import Path
import unittest
import pandas as pd
_missing = object()
_previous_project_modules = {name: sys.modules.get(name, _missing)
                             for name in ("classification", "config", "data_quality", "stage_io")}

ROOT = Path(__file__).resolve().parents[1]
# Stub only project imports so this focused Step 5 test does not depend on the full GIS stack.
classification = types.ModuleType("classification")
UNKNOWN = "Unknown_未知类别"
COMMERCIAL = "Commercial_商业类别"
OFFICE = "Office_办公室"
classification.UNKNOWN = UNKNOWN
classification.MAIN_CLASSES = ["Residential_住宅类别", COMMERCIAL, "Industrial_工业类别"]
classification.MIXED = "Mixed-use_混合用途"
classification.main_label = lambda value: value
classification.sub_label = lambda value: value
classification.taxonomy = lambda: {OFFICE: COMMERCIAL}
def finalize(frame):
    frame = frame.copy()
    for column in ("Observed_Main_Area_Proportions", "Observed_Residential_Area_Share",
                   "Observed_Nondomestic_Area_Share"):
        if column not in frame:
            frame[column] = None
    for index, row in frame.iterrows():
        main, sub = row["Final_Main_Class"], row["Final_Sub_Class"]
        if main in classification.MAIN_CLASSES and sub != UNKNOWN:
            frame.at[index, "Sub_Class_Proportions"] = f"assumed:{sub}=1"
            frame.at[index, "Proportion_Method"] = "single_function_assumption"
            frame.at[index, "Proportion_Quality"] = "assumed_not_measured"
        else:
            frame.at[index, "Sub_Class_Proportions"] = ""
            frame.at[index, "Proportion_Method"] = "unknown_no_use_area_evidence"
            frame.at[index, "Proportion_Quality"] = "unknown"
        for column in ("Observed_Main_Area_Proportions", "Observed_Residential_Area_Share",
                       "Observed_Nondomestic_Area_Share"):
            if column in frame:
                frame.at[index, column] = None
        frame.at[index, "Main_Class_Proportions"] = frame.at[index, "Observed_Main_Area_Proportions"]
    return frame
classification.finalize_function_fields = finalize
sys.modules["classification"] = classification
config = types.ModuleType("config")
config.RULE_ENGINE_OUTPUT_PATH = "unused.csv"
config.LLM_VERIFICATION_OUTPUT = "unused.csv"
config.STEP5_MERGED_OUTPUT_PATH = "unused.csv"
config.AI_CALIBRATED_OUTPUT_PATH = "unused.csv"
config.OFFICIAL_LIB_BASE_PATH = "unused.csv"
config.INTERMEDIATE_DIR = "unused"
config.FEATURE_X_ALL_PATH = "unused"
config.FEATURE_X_TRAIN_ML_PATH = "unused"
config.FEATURE_Y_MULTILABEL_PATH = "unused"
config.FEATURE_TRAINING_DATA_ML_PATH = "unused"
config.FEATURE_BASE_INDEXED_PATH = "unused"
sys.modules["config"] = config
data_quality = types.ModuleType("data_quality")
data_quality.ID = "BUILDINGSTRUCTUREID"
data_quality.ids = lambda s: s.astype(str)
data_quality.require_unique = lambda frame: (_ for _ in ()).throw(AssertionError("duplicate ids")) if frame[data_quality.ID].duplicated().any() else None
data_quality.numbers = lambda *args, **kwargs: None
sys.modules["data_quality"] = data_quality
stage_io = types.ModuleType("stage_io")
for name in ("read_table", "attach_geometry", "save_buildings"):
    setattr(stage_io, name, lambda *args, **kwargs: None)
sys.modules["stage_io"] = stage_io
spec = importlib.util.spec_from_file_location("candidate_step5", ROOT / "src" / "5.evaluation_and_merge.py")
step5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step5)
spec8 = importlib.util.spec_from_file_location("candidate_step8", ROOT / "src" / "8.feature_engineering.py")
step8 = importlib.util.module_from_spec(spec8)
spec8.loader.exec_module(step8)
for _name, _module in _previous_project_modules.items():
    if _module is _missing:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module

class Step5NativeMergeTests(unittest.TestCase):
    def test_step8_keeps_native_classifier_out_of_training(self):
        frame = pd.DataFrame([
            {"BUILDINGSTRUCTUREID": "42", "Classification_Source": "LLM_native_interactive",
             "CATEGORY": "1", "Final_Main_Class": COMMERCIAL, "BDBIAR_Match_Quality": ""},
            {"BUILDINGSTRUCTUREID": "43", "Classification_Source": "Keyword_exact_name",
             "CATEGORY": "1", "Final_Main_Class": COMMERCIAL, "BDBIAR_Match_Quality": ""},
        ])
        training = step8.prepare_training_data(frame)
        self.assertEqual(training.BUILDINGSTRUCTUREID.astype(str).tolist(), ["43"])

    def test_native_backend_row_in_api_cache_is_never_promoted(self):
        frame = pd.DataFrame([{"BUILDINGSTRUCTUREID": "42", "Final_Main_Class": UNKNOWN,
                               "Final_Sub_Class": UNKNOWN, "Classification_Source": "Rules"}])
        cache = pd.DataFrame([{"BUILDINGSTRUCTUREID": "42", "Input_SHA256": "same",
                               "LLM_Main_Class": COMMERCIAL, "LLM_Sub_Class": OFFICE,
                               "LLM_Confidence": .99, "Checkpoint_Backend": "interactive_native_session"}])
        result = step5.merge_verified_results(frame, cache, lambda row: "same")
        self.assertEqual(result.iloc[0].Final_Main_Class, UNKNOWN)

    def test_only_approved_complete_response_changes_function_labels_and_physical_fields_survive(self):
        frame = pd.DataFrame([
            {"BUILDINGSTRUCTUREID": "42", "Final_Main_Class": UNKNOWN, "Final_Sub_Class": UNKNOWN,
             "Classification_Source": "Rules", "HEIGHT": 12.2, "GFA": 404.5, "FOOTPRINT": 80.0,
             "Residential_Area_Proportion": 0.0, "Sub_Class_Proportions": "old-vector",
             "Observed_Main_Area_Proportions": "", "Proportion_Method": "unknown_no_use_area_evidence",
             "Proportion_Quality": "unknown", "Observed_Residential_Area_Share": None,
             "Observed_Nondomestic_Area_Share": None, "Main_Class_Proportions": "old-main-vector"},
            {"BUILDINGSTRUCTUREID": "43", "Final_Main_Class": UNKNOWN, "Final_Sub_Class": UNKNOWN,
             "Classification_Source": "Rules", "HEIGHT": 18.0, "GFA": 900.0, "FOOTPRINT": 100.0,
             "Residential_Area_Proportion": 0.0, "Sub_Class_Proportions": "old-vector-2",
             "Observed_Main_Area_Proportions": "", "Proportion_Method": "unknown_no_use_area_evidence",
             "Proportion_Quality": "unknown", "Observed_Residential_Area_Share": None,
             "Observed_Nondomestic_Area_Share": None, "Main_Class_Proportions": "old-main-vector-2"},
        ])
        physical_before = frame[["HEIGHT", "GFA", "FOOTPRINT", "Residential_Area_Proportion"]].copy()
        native = {
            "42": {"status": "matching_native_checkpoint", "adoptable": True,
                   "main": COMMERCIAL, "sub": OFFICE, "confidence": .91},
            "43": {"status": "reviewed_partial", "adoptable": False,
                   "main": COMMERCIAL, "sub": UNKNOWN, "confidence": .92},
        }
        result = step5.merge_verified_results(frame, pd.DataFrame(), lambda row: "unused", native_results=native)
        by_id = result.set_index("BUILDINGSTRUCTUREID")
        self.assertEqual(by_id.loc["42", "Final_Main_Class"], COMMERCIAL)
        self.assertEqual(by_id.loc["42", "Final_Sub_Class"], OFFICE)
        self.assertEqual(by_id.loc["42", "Classification_Source"], "LLM_native_interactive")
        self.assertEqual(by_id.loc["43", "Final_Main_Class"], UNKNOWN)
        self.assertEqual(by_id.loc["43", "LLM_Review_Status"], "reviewed_partial")
        pd.testing.assert_frame_equal(result[["HEIGHT", "GFA", "FOOTPRINT", "Residential_Area_Proportion"]], physical_before)
        self.assertEqual(by_id.loc["42", "Sub_Class_Proportions"], f"assumed:{OFFICE}=1")
        self.assertEqual(by_id.loc["42", "Proportion_Method"], "single_function_assumption")
        self.assertEqual(by_id.loc["42", "Proportion_Quality"], "assumed_not_measured")
        self.assertTrue(pd.isna(by_id.loc["42", "Observed_Residential_Area_Share"]))
        self.assertTrue(pd.isna(by_id.loc["42", "Observed_Nondomestic_Area_Share"]))
        self.assertEqual(by_id.loc["42", "GFA"], 404.5)
        second = finalize(result)
        columns = ["Sub_Class_Proportions", "Observed_Main_Area_Proportions", "Proportion_Method",
                   "Proportion_Quality", "Observed_Residential_Area_Share", "Observed_Nondomestic_Area_Share",
                   "Main_Class_Proportions"]
        pd.testing.assert_frame_equal(finalize(result)[columns], second[columns])

if __name__ == "__main__":
    unittest.main()
