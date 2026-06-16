# -*- coding: utf-8 -*-
"""
Diagnostic model-selection checks for the Step 9 machine-learning module.

This script reproduces the no-API comparisons reported in the Supplementary
Information. It does not modify the released dataset. It uses the saved Step 8
feature matrices and Step 7 classification output to compare:

1. Main-class classifier and sampling-strategy combinations under the same
   20% holdout split used by Step 9.
2. Subclass proportion regressors under the same 20% holdout split.
"""

import json
import os
import re
import sys
import warnings
import importlib.util

import numpy as np
import pandas as pd

try:
    import sklearn.utils.fixes as _sklearn_fixes
    _sklearn_fixes._in_unstable_openblas_configuration = lambda: False
except Exception:
    pass

from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor

from config import (
    AI_CALIBRATED_OUTPUT_PATH,
    FEATURE_X_TRAIN_ML_PATH,
    FEATURE_Y_MULTILABEL_PATH,
    INTERMEDIATE_DIR,
    KEYWORDS_FILE,
    SMOTE_MIN_SAMPLES,
)


warnings.filterwarnings("ignore")


def normalize_building_id(value):
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


_ML9_PATH = os.path.join(os.path.dirname(__file__), "9.ml_classification.py")
_ML9_SPEC = importlib.util.spec_from_file_location("ml9_pipeline", _ML9_PATH)
ml9 = importlib.util.module_from_spec(_ML9_SPEC)
_ML9_SPEC.loader.exec_module(ml9)


def load_inputs():
    X = pd.read_pickle(FEATURE_X_TRAIN_ML_PATH)
    y = pd.read_pickle(FEATURE_Y_MULTILABEL_PATH)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    rule = pd.read_csv(AI_CALIBRATED_OUTPUT_PATH, low_memory=False)
    rule["BUILDINGSTRUCTUREID"] = rule["BUILDINGSTRUCTUREID"].apply(normalize_building_id)
    rule = rule.set_index("BUILDINGSTRUCTUREID").reindex(X.index)

    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        keywords = json.load(f)

    return X, y, rule, keywords


def smote_like_step9(X_train, y_train):
    pos_count = int(y_train.sum())
    if pos_count <= 0:
        return X_train, y_train, "none"

    if pos_count < SMOTE_MIN_SAMPLES:
        k_neighbors = max(1, min(pos_count - 1, 5))
        sampler = SMOTE(
            random_state=42,
            sampling_strategy={1: SMOTE_MIN_SAMPLES},
            k_neighbors=k_neighbors,
        )
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res, f"pos {pos_count}->{SMOTE_MIN_SAMPLES}"

    if (len(y_train) / pos_count) > 2:
        sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res, f"balanced pos {pos_count}->{int(y_res.sum())}"

    return X_train, y_train, "none"


def compare_main_classifier_sampler_grid(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "LightGBM": lambda: lgb.LGBMClassifier(random_state=42, verbose=-1),
        "XGBoost": lambda: XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_estimators=100,
            n_jobs=-1,
            verbosity=0,
        ),
        "RF-100": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "RF-200": lambda: RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "ExtraTrees-100": lambda: ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "ExtraTrees-200": lambda: ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }
    samplers = {
        "No resampling": None,
        "ROS": RandomOverSampler(random_state=42),
        "RUS": RandomUnderSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42, k_neighbors=5),
        "ADASYN": ADASYN(random_state=42, n_neighbors=5),
    }

    rows = []
    for model_name, factory in models.items():
        for sampler_name, sampler in samplers.items():
            class_rows = []
            f1_values = []
            failed = None
            for col in y.columns:
                y_train_c = y_train[col].astype(int)
                y_val_c = y_val[col].astype(int)
                X_fit, y_fit = X_train, y_train_c
                if sampler is not None:
                    try:
                        X_fit, y_fit = sampler.fit_resample(X_train, y_train_c)
                    except Exception as exc:
                        failed = f"{col}: {exc}"
                        break

                clf = factory()
                clf.fit(X_fit, y_fit)
                pred = clf.predict(X_val)
                f1 = f1_score(y_val_c, pred, zero_division=0)
                f1_values.append(f1)
                class_rows.append(
                    {
                        "model": model_name,
                        "sampler": sampler_name,
                        "class": col.replace("is_", ""),
                        "precision": precision_score(y_val_c, pred, zero_division=0),
                        "recall": recall_score(y_val_c, pred, zero_division=0),
                        "f1": f1,
                        "status": "ok",
                    }
                )

            if failed is not None:
                rows.append(
                    {
                        "model": model_name,
                        "sampler": sampler_name,
                        "class": "",
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "macro_f1": np.nan,
                        "status": failed,
                    }
                )
                continue

            macro = float(np.mean(f1_values))
            for row in class_rows:
                row["macro_f1"] = macro
            rows.extend(class_rows)

    return pd.DataFrame(rows)


def compare_main_classifiers(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "LightGBM": lambda: lgb.LGBMClassifier(random_state=42, verbose=-1),
        "XGBoost": lambda: XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            n_jobs=1,
            verbosity=0,
        ),
        "RF-50": lambda: RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=-1
        ),
        "RF-100": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "RF-200": lambda: RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "ExtraTrees-100": lambda: ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "ExtraTrees-200": lambda: ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    rows = []
    for model_name, factory in models.items():
        f1_values = []
        for col in y.columns:
            y_train_c = y_train[col].astype(int)
            y_val_c = y_val[col].astype(int)
            X_res, y_res, note = smote_like_step9(X_train, y_train_c)

            clf = factory()
            clf.fit(X_res, y_res)
            pred = clf.predict(X_val)

            f1 = f1_score(y_val_c, pred, zero_division=0)
            f1_values.append(f1)
            rows.append(
                {
                    "model": model_name,
                    "class": col.replace("is_", ""),
                    "precision": precision_score(y_val_c, pred, zero_division=0),
                    "recall": recall_score(y_val_c, pred, zero_division=0),
                    "f1": f1,
                    "sampler_note": note,
                }
            )

        macro = float(np.mean(f1_values))
        for row in rows:
            if row["model"] == model_name:
                row["macro_f1"] = macro

    return pd.DataFrame(rows)


def compare_samplers_for_extratrees(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def make_sampler(name, pos_count, total):
        trigger = (pos_count < SMOTE_MIN_SAMPLES) or ((total / pos_count) > 2)
        if name == "No resampling" or not trigger:
            return None, "none"
        if name == "SMOTE":
            return SMOTE(random_state=42), "SMOTE"
        if name == "ADASYN":
            return ADASYN(random_state=42), "ADASYN"
        if name == "ROS":
            return RandomOverSampler(random_state=42), "ROS"
        if name == "RUS":
            return RandomUnderSampler(random_state=42), "RUS"
        raise ValueError(name)

    rows = []
    for sampler_name in ["No resampling", "ROS", "RUS", "SMOTE", "ADASYN"]:
        f1_values = []
        for col in y.columns:
            y_train_c = y_train[col].astype(int)
            y_val_c = y_val[col].astype(int)
            sampler, note = make_sampler(sampler_name, int(y_train_c.sum()), len(y_train_c))
            if sampler is None:
                X_res, y_res = X_train, y_train_c
            else:
                X_res, y_res = sampler.fit_resample(X_train, y_train_c)

            clf = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_res, y_res)
            pred = clf.predict(X_val)

            f1 = f1_score(y_val_c, pred, zero_division=0)
            f1_values.append(f1)
            rows.append(
                {
                    "sampler": sampler_name,
                    "class": col.replace("is_", ""),
                    "precision": precision_score(y_val_c, pred, zero_division=0),
                    "recall": recall_score(y_val_c, pred, zero_division=0),
                    "f1": f1,
                    "sampler_note": note,
                }
            )

        macro = float(np.mean(f1_values))
        for row in rows:
            if row["sampler"] == sampler_name:
                row["macro_f1"] = macro

    return pd.DataFrame(rows)


def build_subclass_target(y_raw, subclasses):
    y_sub = pd.DataFrame(0.0, index=y_raw.index, columns=subclasses)
    for idx, val in y_raw.items():
        matched = False
        for subclass in subclasses:
            if ml9.subclass_label_matches(subclass, val):
                y_sub.loc[idx, subclass] = 1.0
                matched = True
                break
        if not matched and subclasses:
            y_sub.loc[idx, subclasses[0]] = 1.0
    return y_sub


def compare_subclass_regressors(X, rule, keywords):
    main_classes = [k for k in keywords.keys() if k != "__STRONG_KEYWORDS__"][:3]
    models = {
        "RF-50": lambda: RandomForestRegressor(
            n_estimators=50, random_state=42, n_jobs=-1
        ),
        "RF-100": lambda: RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "RF-200": lambda: RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "ET-50": lambda: ExtraTreesRegressor(
            n_estimators=50, random_state=42, n_jobs=-1
        ),
        "ET-100": lambda: ExtraTreesRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "ET-200": lambda: ExtraTreesRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "LightGBM": lambda: MultiOutputRegressor(
            lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100), n_jobs=1
        ),
        "XGBoost": lambda: MultiOutputRegressor(
            XGBRegressor(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.3,
                n_jobs=1,
                verbosity=0,
            ),
            n_jobs=1,
        ),
    }

    rows = []
    for main_class in main_classes:
        target_main = ml9.canonicalize_main_label(main_class)
        subclasses = sorted(keywords[main_class].keys())
        mask = rule["Final_Main_Class"].apply(ml9.canonicalize_main_label) == target_main
        X_sub = X.loc[mask]
        y_raw = rule.loc[mask, "Final_Sub_Class"]
        y_sub = build_subclass_target(y_raw, subclasses)

        X_train, X_val, y_train, y_val = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )
        for model_name, factory in models.items():
            regressor = factory()
            regressor.fit(X_train, y_train)
            pred = np.clip(regressor.predict(X_val), 0, 1)
            rows.append(
                {
                    "model": model_name,
                    "main_class": target_main,
                    "mse": mean_squared_error(y_val, pred),
                    "n_samples": len(X_sub),
                    "n_targets": len(subclasses),
                }
            )

    return pd.DataFrame(rows)


def save_summary(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {path}")


def main():
    X, y, rule, keywords = load_inputs()

    main_df = compare_main_classifier_sampler_grid(X, y)
    save_summary(
        main_df,
        os.path.join(INTERMEDIATE_DIR, "model_sampler_grid_main_classifier_step9_split.csv"),
    )

    reg_df = compare_subclass_regressors(X, rule, keywords)
    save_summary(
        reg_df,
        os.path.join(INTERMEDIATE_DIR, "model_choice_subclass_regressor_corrected.csv"),
    )

    print("[INFO] Diagnostic model-selection checks completed.")


if __name__ == "__main__":
    main()
