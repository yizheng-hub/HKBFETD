# 7.apply_calibrations.py
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import traceback
from collections import defaultdict

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

from config import (
    LOG_DIR,
    STEP5_MERGED_OUTPUT_PATH,
    AI_CALIBRATED_OUTPUT_PATH,
    CANDIDATES_RULES_PATH,
    AI_DECISIONS_LOG_PATH,
    FINAL_SUGGESTIONS_PATH,
)
from utils import (
    normalize_classification_columns,
    sanitize_dataframe_for_geojson,
    is_non_assessed_main,
    canonicalize_main_class_label,
    canonicalize_sub_class_label,
)

warnings.filterwarnings("ignore")
tqdm.pandas()

ABSOLUTE_FINAL_CSV = STEP5_MERGED_OUTPUT_PATH
ABSOLUTE_FINAL_GEOJSON = STEP5_MERGED_OUTPUT_PATH.replace(".csv", ".geojson")
AI_CALIBRATED_GEOJSON_PATH = AI_CALIBRATED_OUTPUT_PATH.replace(".csv", ".geojson")


def clean_id_global(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _safe_float(value, default=0.0):
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "null", "na"}:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _final_arbitration_rule(row):
    rule = str(row.get("rule_decision", "keep")).strip().lower()
    ai = row.get("ai_decision", None)
    ai_str = "" if pd.isna(ai) else str(ai).strip().lower()

    if ai_str in {"", "none", "nan", "error", "unknown"}:
        return f"Rule-Only_{rule}"
    if ai_str == rule:
        return f"Consensus_{ai_str}"

    if rule == "merge" and ai_str == "keep":
        return "Final_keep_with_inheritance"
    if rule == "keep" and ai_str == "merge":
        return "Manual_Review_AI_Wants_Merge"

    if ai_str == "delete":
        if rule == "merge":
            return "Final_delete (AI overrides Rule_Merge)"
        return "Manual_Review_AI_Wants_Delete"

    if rule == "delete" and ai_str in {"keep", "merge"}:
        return f"Final_{ai_str}_(AI_overrides_Rule_Delete)"

    return f"Unknown_Conflict_AI({ai_str})_Rule({rule})"


def _resolve_multi_host_conflicts(group):
    suggestions = group["suggestion"].astype(str).tolist()
    has_strong = any(("merge" in s.lower()) or ("delete" in s.lower()) or ("inheritance" in s.lower()) for s in suggestions)
    has_keep = any("keep" in s.lower() for s in suggestions)

    if has_strong and has_keep:
        group["final_suggestion"] = "Final_keep (Conflict_Override)"
        return group

    merge_like = group[group["suggestion"].astype(str).str.contains("merge|inheritance", case=False, na=False)]
    if len(merge_like) > 1:
        group["final_suggestion"] = "Manual_Review_Multi_Merge"
    else:
        group["final_suggestion"] = group["suggestion"]
    return group


def apply_ai_arbitration(df_base):
    if not os.path.exists(CANDIDATES_RULES_PATH):
        raise FileNotFoundError(f"Missing file: {CANDIDATES_RULES_PATH}")

    df_rules = pd.read_csv(CANDIDATES_RULES_PATH, low_memory=False)
    df_rules["fragment_id"] = df_rules["fragment_id"].apply(clean_id_global)
    df_rules["host_id"] = df_rules["host_id"].apply(clean_id_global)

    nan_mask = (
        df_rules["rule_decision"].isna()
        | (df_rules["rule_decision"].astype(str).str.strip() == "")
        | (df_rules["rule_decision"].astype(str).str.lower() == "nan")
    )
    if nan_mask.any():
        df_rules.loc[nan_mask, "rule_decision"] = "keep"
        df_rules.to_csv(CANDIDATES_RULES_PATH, index=False, encoding="utf-8-sig")

    if os.path.exists(AI_DECISIONS_LOG_PATH):
        df_ai = pd.read_csv(AI_DECISIONS_LOG_PATH, low_memory=False)
        if "fragment_id" in df_ai.columns and "host_id" in df_ai.columns:
            df_ai["fragment_id"] = df_ai["fragment_id"].apply(clean_id_global)
            df_ai["host_id"] = df_ai["host_id"].apply(clean_id_global)
            df_ai = df_ai.drop_duplicates(subset=["fragment_id", "host_id"], keep="last")
            df_combined = df_rules.merge(df_ai, on=["fragment_id", "host_id"], how="left")
        else:
            df_combined = df_rules.copy()
            df_combined["ai_decision"] = None
    else:
        df_combined = df_rules.copy()
        df_combined["ai_decision"] = None

    df_combined["suggestion"] = df_combined.apply(_final_arbitration_rule, axis=1)
    df_final_suggestions = (
        df_combined.groupby("fragment_id", group_keys=False)
        .progress_apply(_resolve_multi_host_conflicts)
        .reset_index(drop=True)
    )
    df_final_suggestions.to_csv(FINAL_SUGGESTIONS_PATH, index=False, encoding="utf-8-sig")
    return df_final_suggestions


def apply_correction_suggestions(df_base, df_final_suggestions):
    df_to_apply = normalize_classification_columns(df_base.copy())
    df_to_apply["BUILDINGSTRUCTUREID"] = df_to_apply["BUILDINGSTRUCTUREID"].apply(clean_id_global)

    stats = {"inherited": 0, "deleted": 0, "manual_review_pairs": 0}

    host_map = (
        df_to_apply.set_index("BUILDINGSTRUCTUREID")[["Final_Main_Class", "Final_Sub_Class"]]
        .to_dict("index")
    )

    inherit_set = {
        "Consensus_merge",
        "Final_keep_with_inheritance",
        "Rule-Only_merge",
        "Final_merge_(AI_overrides_Rule_Delete)",
    }
    to_inherit = df_final_suggestions[df_final_suggestions["final_suggestion"].isin(inherit_set)]

    updates = []
    for _, row in tqdm(to_inherit.iterrows(), total=len(to_inherit), desc="Applying merge/inheritance"):
        frag_id = clean_id_global(row.get("fragment_id"))
        host_id = clean_id_global(row.get("host_id"))
        if host_id in host_map and frag_id:
            host_cls = host_map[host_id]
            updates.append(
                {
                    "BUILDINGSTRUCTUREID": frag_id,
                    "Final_Main_Class": host_cls.get("Final_Main_Class"),
                    "Final_Sub_Class": host_cls.get("Final_Sub_Class"),
                    "Classification_Source": f"Inherited_via_{row.get('final_suggestion', 'Unknown')}",
                }
            )
            stats["inherited"] += 1

    if updates:
        update_df = pd.DataFrame(updates).set_index("BUILDINGSTRUCTUREID")
        df_to_apply = df_to_apply.set_index("BUILDINGSTRUCTUREID")
        df_to_apply.update(update_df)
        df_to_apply = df_to_apply.reset_index()

    delete_set = {
        "Consensus_delete",
        "Final_delete (AI overrides Rule_Merge)",
        "Rule-Only_delete",
    }
    delete_ids = (
        df_final_suggestions[df_final_suggestions["final_suggestion"].isin(delete_set)]["fragment_id"]
        .apply(clean_id_global)
        .unique()
    )
    mask_delete = df_to_apply["BUILDINGSTRUCTUREID"].isin(delete_ids)
    if mask_delete.any():
        df_to_apply.loc[mask_delete, "Final_Main_Class"] = canonicalize_main_class_label("Non-assessed_非评估类别")
        df_to_apply.loc[mask_delete, "Final_Sub_Class"] = canonicalize_sub_class_label("Absolute Noise_绝对噪声相关")
        df_to_apply.loc[mask_delete, "Classification_Source"] = "Deleted_AI_Rule_Verified"
        stats["deleted"] = int(mask_delete.sum())

    stats["manual_review_pairs"] = int(
        df_final_suggestions["final_suggestion"].astype(str).str.contains("Manual_Review", na=False).sum()
    )

    df_to_apply = normalize_classification_columns(df_to_apply)

    def estimate_height_and_floors(row):
        floors = _safe_float(row.get("NUMABOVEGROUNDSTOREYS"), 0)
        top_h = _safe_float(row.get("TOPHEIGHT"), 0.0)
        base_h = _safe_float(row.get("BASEHEIGHT"), 0.0)
        building_h = top_h - base_h if (top_h - base_h) > 0 else top_h
        if floors <= 0 and building_h > 0:
            floors = max(1, round(building_h / 3.0))
        if floors <= 0:
            floors = 5
        est_h = building_h if building_h > 0 else floors * 3.0
        return pd.Series([floors, est_h])

    df_to_apply[["Estimated_Floors", "Estimated_Height"]] = df_to_apply.apply(
        estimate_height_and_floors, axis=1, result_type="expand"
    )
    df_to_apply["NUMABOVEGROUNDSTOREYS"] = df_to_apply["Estimated_Floors"]

    drop_cols = ["Pre_Class_Main", "Pre_Class_Sub", "Classification_Stage", "IS_FINAL_RECALC"]
    df_to_apply.drop(columns=[c for c in drop_cols if c in df_to_apply.columns], inplace=True, errors="ignore")

    out_cols = [c for c in df_to_apply.columns if c != "geometry"]
    df_to_apply[out_cols].to_csv(AI_CALIBRATED_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    if isinstance(df_to_apply, gpd.GeoDataFrame) and "geometry" in df_to_apply.columns:
        gdf_export = sanitize_dataframe_for_geojson(df_to_apply.copy())
        gdf_export.to_file(AI_CALIBRATED_GEOJSON_PATH, driver="GeoJSON")

    return df_to_apply, stats


def print_stage_statistics(df, stats):
    print("[INFO] Step 7 summary")
    main_counts = df["Final_Main_Class"].value_counts(dropna=False)
    print("[INFO] Main class distribution")
    print(main_counts.to_markdown())

    non_eval_count = int(df["Final_Main_Class"].apply(is_non_assessed_main).sum())
    print(f"[INFO] Non-assessed records: {non_eval_count}")
    print(f"[INFO] Inherited records: {stats.get('inherited', 0)}")
    print(f"[INFO] Deleted records: {stats.get('deleted', 0)}")
    print(f"[INFO] Manual-review pairs: {stats.get('manual_review_pairs', 0)}")


def load_step5_base():
    if os.path.exists(ABSOLUTE_FINAL_GEOJSON):
        gdf = gpd.read_file(ABSOLUTE_FINAL_GEOJSON)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:2326", allow_override=True)
        return normalize_classification_columns(gdf)
    df = pd.read_csv(ABSOLUTE_FINAL_CSV, low_memory=False)
    return normalize_classification_columns(df)


def main():
    print("[INFO] Running Step 7 apply calibrations")
    df_base = load_step5_base()
    df_suggestions = apply_ai_arbitration(df_base)
    if df_suggestions is None or df_suggestions.empty:
        print("[WARN] No suggestions generated")
        return False
    final_df, stats = apply_correction_suggestions(df_base, df_suggestions)
    print_stage_statistics(final_df, stats)
    print("[INFO] Step 7 completed")
    return True


class Logger:
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if "\r" not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return True


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, "7.apply_calibrations.txt")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    try:
        ok = main()
        if not ok:
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
