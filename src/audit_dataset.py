"""Read-only, full-stock audit of the HKBFETD inputs and published outputs.

Run with the HKbuilding environment. No API calls, repairs or model training are
performed. Threshold flags are review candidates, not verified factual errors.
"""

import argparse
import hashlib
import json
import pickle
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import shapely


ID = "BUILDINGSTRUCTUREID"
PUBLIC_ID = "Building_ID_建筑ID"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PROPORTION = re.compile(rf"\s*(.*?)\s*:\s*({NUMBER})\s*(?=-|$)")


def parse_proportions(value):
    """Parse the entire vector, retaining hyphens inside labels and duplicates."""
    if pd.isna(value) or not str(value).strip():
        return [], False
    text = str(value).strip()
    pairs, pos = [], 0
    while pos < len(text):
        match = PROPORTION.match(text, pos)
        if match is None or not match[1].strip():
            return pairs, False
        pairs.append((match[1].strip(), float(match[2])))
        pos = match.end()
        if pos < len(text):
            pos += 1
            if pos == len(text):
                return pairs, False
    return pairs, bool(pairs)


def normalized_ids(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def equal_values(left, right):
    return (left.eq(right) | (left.isna() & right.isna())).fillna(False)


def is_dominant_subclass(label, pairs):
    if not pairs:
        return False
    maximum = max(value for _, value in pairs)
    return any(key == label and np.isclose(value, maximum, rtol=0, atol=1e-12) for key, value in pairs)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Audit:
    def __init__(self, root, output):
        self.root, self.output = root, output
        output.mkdir(parents=True, exist_ok=False)
        self.checks, self.profiles, self.lineage, self.manifest = [], [], [], []
        self.flags = None
        self.details = {}

    def record(self, name, value, meaning, level="information"):
        self.checks.append(dict(check=name, value=value, level=level, meaning=meaning))

    def flag(self, name, mask, meaning, level="review"):
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=self.flags.index)
        mask = mask.fillna(False).astype(bool)
        self.flags[name] = mask.reindex(self.flags.index, fill_value=False)
        self.record(name, int(mask.sum()), meaning, level)

    def csv(self, name, frame, index=False):
        frame.to_csv(self.output / name, index=index, encoding="utf-8-sig")

    def profile(self, name, frame):
        self.record(name + ".rows", len(frame), "Records, including repeated IDs where applicable")
        if ID in frame:
            self.record(name + ".duplicate_id_rows", int(frame[ID].duplicated(keep=False).sum()),
                        "Repeated IDs; aggregation tables can legitimately have multiple rows")
        for col in frame:
            if col == "geometry":
                continue
            s = frame[col]
            missing = s.isna() | s.astype(str).str.strip().str.lower().isin(["", "none", "null", "nan"])
            nums = pd.to_numeric(s, errors="coerce")
            finite = nums[np.isfinite(nums)]
            self.profiles.append(dict(dataset=name, field=col, rows=len(s), missing=int(missing.sum()),
                                      distinct=int(s.nunique()), numeric_values=int(nums.notna().sum()),
                                      nonfinite=int((nums.notna() & ~np.isfinite(nums)).sum()),
                                      minimum=finite.min(), median=finite.median(), maximum=finite.max()))

    def read_csv(self, path):
        df = pd.read_csv(path, low_memory=False)
        if PUBLIC_ID in df:
            df = df.rename(columns={PUBLIC_ID: ID})
        if ID in df:
            df[ID] = normalized_ids(df[ID])
        self.profile(path.stem, df)
        return df

    def table(self, number):
        path = next((self.root / "data/LandsD").glob(f"{number}.*"))
        df = pd.DataFrame([f["properties"] for f in json.loads(path.read_text(encoding="utf-8"))["features"]])
        if ID in df:
            df[ID] = normalized_ids(df[ID])
        self.profile(path.stem, df)
        return df

    def geometry(self, raw):
        self.record("geometry.crs", str(raw.crs), "Area/distance checks require projected Hong Kong metres")
        if raw.crs.to_epsg() != 2326:
            raise ValueError("Raw footprint CRS must be EPSG:2326 for these checks")
        geom = raw.geometry
        self.flag("geometry_missing", geom.isna() | geom.is_empty, "Missing or empty footprint", "error")
        self.flag("geometry_invalid", ~geom.is_valid, "Invalid source geometry; never silently repaired", "error")
        self.flag("geometry_nonpolygon", ~geom.geom_type.isin(["Polygon", "MultiPolygon"]), "Unexpected geometry type", "error")
        area = geom.area
        self.flags["geometry_area_sqm"] = area
        self.flag("area_attribute_differs", (area - numeric(raw, "SHAPE_Area")).abs() > np.maximum(1, area * .01),
                  "Stored area differs from geometry by over max(1 sqm, 1%); check footprint changes")
        self.flag("geometry_duplicate", geom.to_wkb().duplicated(keep=False), "Exact duplicate geometry across records")
        coords = raw.to_crs(4326).geometry.representative_point()
        self.flag("coordinate_outside_hk_bbox", ~coords.x.between(113.83, 114.45) | ~coords.y.between(22.15, 22.57),
                  "Outside configured Hong Kong bounding box; offshore locations require review")
        self.flags["longitude"], self.flags["latitude"] = coords.x, coords.y
        valid = raw[geom.is_valid & ~geom.is_empty & geom.notna()]
        pairs = []
        for start in range(0, len(valid), 10000):
            query = valid.iloc[start:start + 10000]
            left, right = valid.sindex.query(query.geometry, predicate="intersects")
            absolute_left = start + left
            keep = absolute_left < right
            absolute_left, right = absolute_left[keep], right[keep]
            if not len(right):
                continue
            a, b = valid.geometry.array[absolute_left], valid.geometry.array[right]
            overlap = shapely.area(shapely.intersection(a, b))
            keep = overlap > 1.0
            for i, j, overlap_area in zip(absolute_left[keep], right[keep], overlap[keep]):
                first, second = valid.iloc[i], valid.iloc[j]
                pairs.append((first.name, second.name, first.BUILDINGSTRUCTURETYPE, second.BUILDINGSTRUCTURETYPE,
                              overlap_area, overlap_area / min(first.geometry.area, second.geometry.area)))
        pair_df = pd.DataFrame(pairs, columns=["id_a", "id_b", "type_a", "type_b", "overlap_sqm", "smaller_footprint_fraction"])
        self.csv("geometry_overlap_pairs.csv", pair_df)
        self.record("geometry.overlap_pairs_gt_1sqm", len(pair_df),
                    "Review candidates: tower/podium and vertical building parts may legitimately overlap")
        self.flag("footprint_overlaps_another", self.flags.index.to_series().isin(set(pair_df.id_a) | set(pair_df.id_b)),
                  "Footprint overlap; volume/GFA aggregation needs building-part semantics")

    def morphology(self, raw, public):
        top, base, floors = (numeric(raw, c) for c in ["TOPHEIGHT", "BASEHEIGHT", "NUMABOVEGROUNDSTOREYS"])
        h, f, area = (numeric(public, c) for c in ["Height_m_建筑高度", "Floors_Count_地上层数", "Footprint_Area_sqm_占地面积"])
        self.flags["raw_top_elevation"] = top
        self.flags["raw_base_elevation"] = base
        self.flags["raw_storeys"] = floors
        self.flags["published_height_m"], self.flags["published_storeys"] = h, f
        no_dimension = ~(floors > 0) & ~(top > 0)
        self.flag("all_dimensions_missing_or_nonpositive", no_dimension,
                  "No positive raw storeys or top elevation; published dimensions have no raw dimensional evidence")
        self.flag("height_15m", h.eq(15), "All exact 15 m values, including legitimate measurements/estimates")
        self.flag("default_15m_no_raw_dimensions", h.eq(15) & no_dimension,
                  "15 m despite no positive source height/floors: legacy 5-floor default", "error")
        self.flag("height_15m_from_raw_5_floors", h.eq(15) & floors.eq(5) & ~(top > 0),
                  "15 m inferred from raw 5 floors; distinct from default and requires source plausibility review")
        self.flag("top_elevation_exported_without_base", (top > 0) & base.isna() & np.isclose(h, top, atol=1e-8),
                  "Published height equals raw top elevation with no base; physical height is unverified", "error")
        self.flag("storeys_inferred_from_unbased_top", ~(floors > 0) & (top > 0) & base.isna(),
                  "Storeys derived from unbased absolute elevation", "error")
        self.flag("height_from_raw_storeys_only", (floors > 0) & ~(top > 0), "Height is raw storeys times assumed 3 m")
        self.flag("small_footprint_tall", area.lt(150) & h.gt(150), "Footprint <150 sqm and height >150 m; screen, not automatic rejection")
        self.flag("height_gt_600m", h.gt(600), "Hong Kong height >600 m requires verification")
        self.flag("storeys_gt_150", f.gt(150), "Above-ground floor count >150 requires verification")
        self.flag("nonpositive_or_missing_dimension", h.isna() | f.isna() | h.le(0) | f.le(0), "Missing/invalid published dimensions")
        self.flag("fractional_storeys", f.notna() & ~np.isclose(f, f.round()), "Fractional floor count")
        self.flag("floor_height_outside_2_to_6m", floors.gt(0) & (top > 0) & ~((h / floors).between(2, 6)),
                  "Height/raw-storeys outside [2,6] m; industrial and complex buildings may be valid")
        self.record("dimensions.floor_count_equals_round_height_div3", int(np.isclose(f, np.maximum(1, (h / 3).round())).sum()),
                    "Agreement does not constitute independent height validation")
        self.flags["legacy_footprint_times_floors_sqm"] = area * f
        for group in ["default_15m_no_raw_dimensions", "top_elevation_exported_without_base"]:
            self.record(group + ".legacy_floor_area_proxy_sqm", float((area * f)[self.flags[group]].sum()),
                        "Legacy proxy contribution, NOT a corrected or observed GFA")
        self.csv("dimension_source_by_class.csv", self.flags.join(public[["UBEM_Main_Class_主类别", "Classification_Source_分类来源"]])
                 .groupby(["UBEM_Main_Class_主类别", "Classification_Source_分类来源"], dropna=False)
                 .agg(records=("published_height_m", "size"), default_15m=("default_15m_no_raw_dimensions", "sum"),
                      unbased_top=("top_elevation_exported_without_base", "sum"),
                      floor_area_proxy_sqm=("legacy_footprint_times_floors_sqm", "sum")).reset_index())

    def permits(self, raw, step3):
        permits, links = self.table(6), self.table(7)
        self.record("permit.duplicate_numbers", int(permits.OPNO.duplicated(keep=False).sum()), "Duplicate occupation-permit keys")
        self.record("permit.duplicate_building_links", int(links.duplicated([ID, "OPNO"], keep=False).sum()), "Duplicate building/permit pairs")
        scope = links.groupby("OPNO")[ID].nunique().rename("linked_buildings")
        expanded = links.merge(permits, on="OPNO", how="left", indicator=True)
        self.record("permit.orphan_links", int(expanded._merge.eq("left_only").sum()), "Building links without permit record")
        expanded["permit_gfa"] = numeric(expanded, "DOMESTICGFA").fillna(0) + numeric(expanded, "NONDOMESTICGFA").fillna(0)
        expanded["linked_buildings"] = expanded.OPNO.map(scope)
        self.csv("permit_scope.csv", permits.merge(scope, on="OPNO", how="left"))
        risky = expanded[(expanded.linked_buildings > 1) & (expanded.permit_gfa > 0)]
        self.flag("shared_permit_gfa_scope", self.flags.index.to_series().isin(risky[ID]),
                  "A positive permit-level GFA is copied to multiple footprints; per-building allocation is unverified")
        self.record("permit.shared_positive_gfa_permits", int(risky.OPNO.nunique()), "Positive GFA permits linked to multiple buildings")
        unique_positive = expanded.drop_duplicates("OPNO")
        self.record("permit.unique_positive_gfa_sqm", float(unique_positive.permit_gfa.clip(lower=0).sum()), "Sum once per permit")
        self.record("permit.expanded_positive_gfa_sqm", float(expanded.permit_gfa.clip(lower=0).sum()), "Sum after building links; differs with shared/duplicate links")
        expected = expanded.groupby(ID)[["DOMESTICGFA", "NONDOMESTICGFA"]].sum().rename(
            columns={"DOMESTICGFA": "GFA_DOMESTIC_SUM", "NONDOMESTICGFA": "GFA_NONDOMESTIC_SUM"})
        for col in expected:
            left, right = numeric(step3, col), numeric(expected.reindex(step3.index), col)
            self.flag("permit_aggregation_mismatch_" + col, ~equal_values(left, right), "Existing Step 3 does not match the current raw permit aggregation")
        registered = numeric(raw, "GROSSFLOORAREA")
        self.record("gfa.raw_registered_positive", int(registered.gt(0).sum()), "Sparse raw registered-GFA coverage; not the same field as permit GFA")

    def vectors(self, public):
        cfg = json.loads((self.root / "ctl/keywords.json").read_text(encoding="utf-8"))
        parents, main_alias = {}, {}
        english = {"住宅类别": "Residential", "商业类别": "Commercial", "工业类别": "Industrial", "非评估类别": "Non-assessed"}
        for main, subs in cfg.items():
            if main not in english:
                continue
            canonical_main = english[main] + "_" + main
            for alias in [main, english[main], canonical_main]:
                main_alias[alias] = canonical_main
            for sub in subs:
                aliases = {sub}
                match = re.fullmatch(r"(.+?)（(.+)）", sub)
                if match:
                    aliases.update([match[1], match[2], match[2] + "_" + match[1]])
                else:
                    aliases.update({"绝对噪声相关": ["Absolute Noise_绝对噪声相关"],
                                    "交通基础设施": ["Transport Infrastructure_交通基础设施"],
                                    "临时/杂项设施": ["Temporary/Misc Facilities_临时/杂项设施"]}.get(sub, []))
                for alias in aliases:
                    parents[alias] = canonical_main
        rows = []
        for bid, row in public.iterrows():
            pairs, valid = parse_proportions(row.get("UBEM_Mixed_Proportions_混合比例"))
            labels = [label for label, value in pairs]
            values = [value for label, value in pairs]
            main = row.get("UBEM_Main_Class_主类别", "")
            mixed = "Mixed" in str(main) or "混合" in str(main)
            nonassessed = "Non-assessed" in str(main) or "非评估" in str(main)
            unknown = [k for k in labels if k not in parents and k not in main_alias]
            coarse = [k for k in labels if k in main_alias]
            parent_set = {parents.get(k, main_alias.get(k)) for k in labels} - {None}
            mass = sum(values)
            # Existing vectors are rounded to 2 decimal places; do not flag normal rounding.
            tolerance = max(1e-8, len(pairs) * .005 + 1e-8)
            parent_mismatch = bool(parent_set - {main_alias.get(main, main)}) if not mixed and not nonassessed else False
            subclass = row.get("UBEM_Sub_Class_子类别", "")
            rows.append((bid, not valid, mass, abs(mass - 1) > tolerance,
                         any(v < 0 or v > 1 or not np.isfinite(v) for v in values), len(labels) != len(set(labels)),
                         bool(unknown) and not nonassessed, bool(coarse), parent_mismatch,
                         mixed and len(parent_set) < 2, not mixed and not nonassessed and not is_dominant_subclass(subclass, pairs)))
        result = pd.DataFrame(rows, columns=[ID, "vector_parse_failed", "vector_sum", "vector_mass_outside_rounding_tolerance",
                                            "vector_invalid_value", "vector_duplicate_label", "vector_unknown_taxonomy",
                                            "vector_contains_main_as_subclass", "vector_parent_main_mismatch",
                                            "mixed_without_multiple_main_classes", "subclass_not_largest_vector_entry"]).set_index(ID)
        self.flags["vector_sum"] = result.vector_sum
        for col in result.drop(columns="vector_sum"):
            self.flag(col, result[col], "Public classification/proportion consistency check; see taxonomy and vector semantics")
        self.csv("public_class_source_counts.csv", public.groupby(["UBEM_Main_Class_主类别", "Classification_Source_分类来源"], dropna=False).size().rename("records").reset_index())

    def stage(self, name, df, reference):
        ids = df[ID]
        self.lineage.append(dict(stage=name, rows=len(df), unique_ids=ids.nunique(), missing_id=int(ids.isna().sum()),
                                 duplicate_id_rows=int(ids.duplicated(keep=False).sum()),
                                 missing_raw_ids=len(set(reference) - set(ids)), extra_ids=len(set(ids) - set(reference))))
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{name}: ID integrity failed; cannot compare one record per building")
        return df.set_index(ID).reindex(reference)

    def ml(self, stages, public):
        frames = {}
        for stem in ["feature_X_all", "feature_X_train_ml", "feature_y_multilabel", "feature_training_data_ml"]:
            path = self.root / "intermediate_files" / (stem + ".pkl")
            # Pickles are existing, trusted local pipeline artifacts, never fetched externally.
            with path.open("rb") as stream:
                frame = pickle.load(stream)
            frame.index = normalized_ids(pd.Series(frame.index)).array
            frames[stem] = frame
            self.profile(stem, frame)
        X, train, targets = (frames[n] for n in ["feature_X_all", "feature_X_train_ml", "feature_y_multilabel"])
        self.record("ml.train_target_same_order", train.index.equals(targets.index), "Feature/target alignment")
        self.record("ml.all_ids_equal_published", set(X.index) == set(public.index), "All-feature coverage")
        self.record("ml.target_counts", json.dumps(targets.sum().to_dict(), ensure_ascii=False), "One-vs-rest class counts")
        height = numeric(X, "Estimated_Height").reindex(public.index)
        self.flag("cached_model_height_differs_from_public", ~np.isclose(height, numeric(public, "Height_m_建筑高度"), equal_nan=True),
                  "Feature cache differs from exported height; current cache is not interchangeable with release")
        self.flag("ml_training_record", self.flags.index.to_series().isin(train.index), "Records used to train the cached classifiers", "information")
        stage7 = stages[7]
        for col in ["default_15m_no_raw_dimensions", "top_elevation_exported_without_base"]:
            self.record("ml.training_" + col, int(self.flags.loc[self.flags.ml_training_record, col].sum()), "Dimension issues inside training data")
        target_to_code = {"is_住宅类别": 0, "is_商业类别": 1, "is_工业类别": 2}
        code = sum(targets[col] * value for col, value in target_to_code.items())
        self.record("ml.training_neighbor_feature_equals_target", int(train.dominant_neighbor_class_100m.eq(code).sum()),
                    "Current neighborhood construction includes each training building and is computed before splitting; label leakage risk")
        notes = public.get("Classification_Notes_分类备注", pd.Series("", index=public.index)).fillna("").astype(str)
        self.flag("ml_mixture_equal_weight_assumption", notes.str.contains("多标签混合|multi-label mixture", regex=True),
                  "Class presence probabilities are converted to equal shares; not measured use-area fractions")
        main7 = stage7.Final_Main_Class.astype(str)
        main9 = public["UBEM_Main_Class_主类别"].astype(str)
        self.flag("nonassessed_became_assessed", main7.str.contains("非评估|Non-assessed") & ~main9.str.contains("非评估|Non-assessed"),
                  "Check language normalization and non-assessed protection")
        self.flag("rule_unknown_retained_as_known", main7.str.contains("未知|Unknown") & ~main9.str.contains("未知|Unknown") &
                  public["Classification_Source_分类来源"].astype(str).str.startswith("Rule_Engine"),
                  "Unknown alias bypassed incomplete-rule handling")
        self.record("ml.subclass_training_rows_matching_current_code", int(stage7.Final_Main_Class.isin(["住宅类别", "商业类别", "工业类别"]).sum()),
                    "Step 9 subclass trainer compares exact Chinese labels; bilingual labels match no class")
        self.record("ml.nonassessed_rows_missing_exact_chinese_guard", int((main7.str.contains("非评估|Non-assessed") & main7.ne("非评估类别")).sum()),
                    "Rows missed by Step 9's exact Chinese non-assessed guard, regardless of later output repair")
        probabilities = [c for c in public if c.startswith("ML_Prob_")]
        for col in probabilities + ["ML_Prediction_Confidence_机器学习置信度"]:
            values = numeric(public, col)
            self.record(col + ".nonmissing", int(values.notna().sum()), "Published probability coverage")
            self.flag(col + "_out_of_bounds", values.notna() & ~values.between(0, 1), "Probability outside [0,1]", "error")

    def pipeline_links(self, raw, stages):
        print("Checking Steps 1/2/4/6/8 and source linkage contracts", flush=True)
        llm = self.read_csv(self.root / "output/step4_llm_verified_classification.csv")
        self.record("step4.missing_raw_ids", len(set(raw.index) - set(llm[ID])), "AI cache is a result subset, not necessarily a full building table")
        required = ["Final_Main_Class", "Final_Sub_Class", "Classification_Source", "Sub_Class_Proportions"]
        self.record("step5.missing_input_columns", json.dumps([c for c in required if c not in llm]),
                    "Current Step 5 reads Step 4 directly without joining the full Step 3 table", "error")
        for number in [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]:
            table = self.table(number)
            if ID in table:
                self.record(f"landsd_table_{number}.orphan_building_links", int((~table[ID].isin(raw.index)).sum()), "Foreign-key references outside raw building universe")
        for stem in ["step1_official_building_library", "step1_official_building_library_base", "step8_feature_base_calibrated"]:
            path = self.root / f"intermediate_files/{stem}.geojson"
            fields = pyogrio.read_info(path)["fields"]
            cols = [c for c in [ID, "TOPHEIGHT", "BASEHEIGHT", "NUMABOVEGROUNDSTOREYS", "GROSSFLOORAREA", "SHAPE_Area", "Estimated_Height"] if c in fields]
            frame = pyogrio.read_dataframe(path, columns=cols, read_geometry=False)
            frame[ID] = normalized_ids(frame[ID])
            aligned = self.stage(stem, frame, raw.index)
            for col in ["TOPHEIGHT", "BASEHEIGHT", "NUMABOVEGROUNDSTOREYS", "GROSSFLOORAREA", "SHAPE_Area"]:
                self.record(stem + ".raw_drift." + col, int((~equal_values(numeric(raw, col), numeric(aligned, col))).sum()), "Raw attribute lineage")
        path = self.root / "intermediate_files/step2_osm_aggregated_buildings.geojson"
        aggregation = pyogrio.read_dataframe(path, columns=[ID, "osmid", "index_right", "overlap_ratio", "osm_height"], read_geometry=False)
        aggregation[ID] = normalized_ids(aggregation[ID])
        self.profile("step2_osm_aggregated_buildings", aggregation)
        self.record("step2.missing_raw_ids", len(set(raw.index) - set(aggregation[ID])), "All source buildings must survive the POI expansion")
        self.record("step2.extra_ids", len(set(aggregation[ID]) - set(raw.index)), "Unexpected buildings introduced by spatial join")
        matched = aggregation.dropna(subset=["index_right"])
        self.record("step2.duplicate_building_feature_rows", int(matched.duplicated([ID, "index_right"], keep=False).sum()), "Repeated same building/clean-feature links")
        reuse = matched.groupby("index_right")[ID].nunique()
        self.record("step2.features_linked_to_multiple_buildings", int((reuse > 1).sum()), "Shared features may reflect overlapping parts; double-counted POI evidence needs review")
        counts = matched.groupby(ID).size()
        self.flags["matched_feature_count"] = counts.reindex(raw.index, fill_value=0)
        suggestions = pd.read_csv(self.root / "intermediate_files/step6_geometry_correction_suggestions.csv", low_memory=False)
        for col in ["fragment_id", "host_id"]:
            suggestions[col] = normalized_ids(suggestions[col])
            self.record("step6.orphan_" + col, int((~suggestions[col].isin(raw.index)).sum()), "Geometry suggestion IDs outside building stock")
        self.record("step6.duplicate_pairs", int(suggestions.duplicated(["fragment_id", "host_id"], keep=False).sum()), "Duplicate arbitration pairs")
        self.csv("step6_decision_counts.csv", suggestions.groupby("final_suggestion", dropna=False).size().rename("pairs").reset_index())
        inherited = stages[7].Classification_Source.astype(str).str.startswith("Inherited")
        self.record("step7.inherited_records", int(inherited.sum()), "Records with inherited function; proportions must be inherited/recomputed consistently")
        bdb = pd.read_csv(self.root / "data/BDBIAR.gdb_converted.csv", low_memory=False)
        self.profile("raw_BDBIAR", bdb)
        self.record("bdbiar.raw_duplicate_objectids", int(bdb.OBJECTID.duplicated(keep=False).sum()), "BDBIAR point identity")
        if not bdb.OBJECTID.duplicated().any():
            import geopandas as gpd
            bdb = bdb.dropna(subset=["LONGITUDE", "LATITUDE"])
            points = gpd.GeoDataFrame(bdb, geometry=gpd.points_from_xy(bdb.LONGITUDE, bdb.LATITUDE), crs=4326).to_crs(2326).set_index("OBJECTID")
            match_ids = numeric(stages[3], "BDBIAR_OBJECTID")
            associated = points.reindex(match_ids.to_numpy())
            actual = pd.Series(shapely.distance(raw.geometry.array, associated.geometry.array), index=raw.index)
            reported = numeric(stages[3], "distance_to_bdbiar")
            self.flags["recomputed_bdbiar_distance_m"] = actual
            self.flag("bdbiar_distance_value_mismatch", reported.notna() & ~np.isclose(actual, reported, atol=.05, rtol=0),
                      "Recorded nearest-point distance differs from raw building geometry by >5 cm")
            self.flag("bdbiar_actual_distance_gt_30m", actual.gt(30.05), "Recomputed raw-geometry match exceeds configured radius")

    def run(self, terrain=None):
        print("Reading raw LandsD footprint and public release", flush=True)
        path = next((self.root / "data/LandsD").glob("1.*"))
        raw = pyogrio.read_dataframe(path)
        raw[ID] = normalized_ids(raw[ID])
        self.profile("raw_BUILDING_STRUCTURE", raw)
        if raw[ID].isna().any() or raw[ID].duplicated().any():
            raise ValueError("Raw building IDs must be unique and nonmissing")
        raw = raw.set_index(ID)
        self.flags = pd.DataFrame(index=raw.index)
        self.flags.index.name = ID
        stages = {}
        for step, stem in [(3, "rule_engine"), (5, "merged"), (7, "ai_calibrated"), (9, "ml_calibrated")]:
            print(f"Reading Step {step}", flush=True)
            df = self.read_csv(self.root / f"output/step{step}_{stem}_classification.csv")
            stages[step] = self.stage("step" + str(step), df, raw.index)
        public = self.stage("public_v1_1", self.read_csv(self.root / "output/HK_UBEM_Buildings_Public_v1_1.csv"), raw.index)
        v1 = self.stage("public_v1", self.read_csv(self.root / "output/HK_UBEM_Buildings_Public_v1.csv"), raw.index)
        for col in public.columns.intersection(v1.columns):
            self.record("v1_to_v1_1.changed." + col, int((~equal_values(v1[col], public[col])).sum()), "Values changed between existing public releases")
        for step, df in stages.items():
            for col in ["TOPHEIGHT", "BASEHEIGHT", "NUMABOVEGROUNDSTOREYS", "SHAPE_Area", "GROSSFLOORAREA"]:
                mismatch = ~equal_values(numeric(raw, col), numeric(df, col))
                self.record(f"step{step}.raw_drift.{col}", int(mismatch.sum()), "Raw field modified or lost downstream; estimates should have distinct fields")
            self.flag(f"step{step}_mixed_missing_positive_permit_gfa", df.Final_Main_Class.astype(str).str.contains("Mixed|混合") &
                      (numeric(df, "GFA_DOMESTIC_SUM").fillna(0) + numeric(df, "GFA_NONDOMESTIC_SUM").fillna(0)).le(0),
                      "Mixed classification without positive permit GFA; inspect heuristic or POI-based proportions")
        labels_changed = ~equal_values(stages[5].Final_Main_Class, stages[7].Final_Main_Class)
        same_vector = equal_values(stages[5].Sub_Class_Proportions, stages[7].Sub_Class_Proportions)
        self.flag("stage7_main_changed_vector_unchanged", labels_changed & same_vector, "Geometry arbitration changed main class without updating proportions")
        self.geometry(raw)
        self.morphology(raw, public)
        print("Checking permits, taxonomy and model artifacts", flush=True)
        self.permits(raw, stages[3])
        self.vectors(public)
        self.ml(stages, public)
        self.pipeline_links(raw, stages)
        bdb = stages[3]
        self.flag("bdbiar_match_missing_reported_distance", bdb.BDBIAR_OBJECTID.notna() & numeric(bdb, "distance_to_bdbiar").isna(),
                  "Matched BDBIAR with missing distance; inspect podium d_p concatenation")
        self.flag("bdbiar_match_gt_30m", numeric(bdb, "distance_to_bdbiar").gt(30), "BDBIAR match exceeds configured radius")
        reused = bdb.groupby("BDBIAR_OBJECTID").size()
        self.flag("bdbiar_point_used_for_multiple_footprints", bdb.BDBIAR_OBJECTID.isin(reused[reused > 1].index),
                  "Many footprints share a BDBIAR record; building-part relation must be verified")
        ages = numeric(public, "BDBIAR_Building_Age_Year_屋宇署楼龄年份")
        self.flag("building_year_outside_1800_to_2025", ages.notna() & ~ages.between(1800, 2025), "Year outside source snapshot range")
        if terrain:
            dtm = pd.read_csv(terrain)
            dtm[ID] = normalized_ids(dtm[ID])
            if dtm[ID].duplicated().any():
                raise ValueError("Terrain evidence IDs are duplicated")
            dtm = dtm.set_index(ID).reindex(raw.index)
            coordinate_error = np.hypot(dtm.longitude - self.flags.longitude, dtm.latitude - self.flags.latitude)
            self.record("terrain.coordinate_disagreement_gt_1e_6deg", int(coordinate_error.gt(1e-6).sum()), "Evidence sampling location vs raw representative point")
            self.flags["diagnostic_ground_elevation_mpd"] = numeric(dtm, "dtm_elevation_mpd")
            self.flags["diagnostic_top_minus_ground_m"] = numeric(raw, "TOPHEIGHT") - numeric(dtm, "dtm_elevation_mpd")
            self.flag("terrain_adjusted_height_nonpositive", self.flags.diagnostic_top_minus_ground_m.le(0),
                      "Point terrain subtraction is nonpositive; cannot blindly accept terrain-based replacement")
            self.details["terrain_evidence"] = {"path": str(terrain), "sha256": sha256(terrain),
                                                 "role": "diagnostic only; point DTM is not surveyed building base height"}
        print("Checking GeoJSON/CSV agreement and OSM inputs", flush=True)
        pub_geo = pyogrio.read_dataframe(self.root / "output/HK_UBEM_Buildings_Public_v1_1.geojson")
        pub_geo = pub_geo.rename(columns={PUBLIC_ID: ID})
        pub_geo[ID] = normalized_ids(pub_geo[ID])
        pub_geo = self.stage("public_geojson_v1_1", pub_geo, raw.index)
        self.record("public_geojson.crs", str(pub_geo.crs), "GeoJSON is in EPSG:2326; consumer CRS expectations must be documented")
        self.record("public_geometry.invalid", int((~pub_geo.geometry.is_valid).sum()), "Check exported geometry independently of source repairs")
        self.record("public_geometry.empty_or_missing", int((pub_geo.geometry.isna() | pub_geo.geometry.is_empty).sum()), "Exported geometry completeness")
        geometry_area = pub_geo.geometry.area
        self.record("public_geometry.area_mismatch_gt_1pct_or_1sqm", int(((numeric(public, "Footprint_Area_sqm_占地面积") - geometry_area).abs() > np.maximum(1, geometry_area * .01)).sum()),
                    "Published footprint area vs exported geometry; tolerates serialization precision")
        for raw_col, pub_col in [("Calibrated_Main_Class", "UBEM_Main_Class_主类别"), ("Calibrated_Source", "Classification_Source_分类来源")]:
            self.record("step9_to_public.changed." + raw_col, int((~equal_values(stages[9][raw_col], public[pub_col])).sum()), "Final export main-class and source lineage")
        for col in public.columns.intersection(pub_geo.columns):
            a, b = public[col].fillna("").astype(str), pub_geo[col].fillna("").astype(str)
            nums_a, nums_b = pd.to_numeric(public[col], errors="coerce"), pd.to_numeric(pub_geo[col], errors="coerce")
            numeric_equal = nums_a.notna() & nums_b.notna() & np.isclose(nums_a, nums_b, equal_nan=False)
            self.record("csv_geojson.diff." + col, int((~(a.eq(b) | numeric_equal)).sum()), "ID-aligned attribute comparison")
        changed_geom = ~shapely.equals_exact(raw.geometry.array, pub_geo.geometry.array, tolerance=1e-7)
        self.flag("export_geometry_changed_from_raw", pd.Series(changed_geom, index=raw.index), "Geometry differs from raw source, including legitimate repairs")
        for stem in ["step1_osm_features_all", "step2_osm_features_clean"]:
            opath = self.root / f"intermediate_files/{stem}.geojson"
            info = pyogrio.read_info(opath)
            columns = [c for c in ["osmid", "osm_type", "id", "height", "building:height", "building:levels", "levels", "osm_height"] if c in info["fields"]]
            osm = pyogrio.read_dataframe(opath, columns=columns)
            self.profile(stem, osm)
            keys = [c for c in ["osm_type", "osmid"] if c in osm]
            identity = osm[keys].copy()
            identity["wkb"] = osm.geometry.to_wkb()
            self.record(stem + ".extra_duplicate_identity_geometry_rows", int(identity.duplicated().sum()),
                        "Repeated source identity and exact geometry; attributes may differ between building/POI extractions")
            self.details[stem + ".available_fields"] = info["fields"].tolist()
        sample = ["181189", "170369", "192326", "87318"]
        self.csv("reader_cases_trace.csv", self.flags.join(raw[["OFFICIALBUILDINGNAMEEN", "BUILDINGSTRUCTURETYPE"]])
                 .join(public[["UBEM_Main_Class_主类别", "UBEM_Sub_Class_子类别", "UBEM_Mixed_Proportions_混合比例"]]).reindex(sample), index=True)
        self.csv("record_flags.csv", self.flags, index=True)
        self.csv("checks.csv", pd.DataFrame(self.checks))
        self.csv("field_profiles.csv", pd.DataFrame(self.profiles))
        self.csv("stage_lineage.csv", pd.DataFrame(self.lineage))
        print("Hashing existing inputs, source and release artifacts", flush=True)
        paths = set(self.root.glob("data/**/*")) | set(self.root.glob("src/*")) | set(self.root.glob("ctl/*"))
        paths |= set(self.root.glob("output/*.*")) | set(self.root.glob("intermediate_files/*.*"))
        paths |= {self.root / n for n in ["README.md", "environment.yml", "requirements.txt"]}
        paths |= set(self.root.glob("tests/*.py"))
        for path in sorted(p for p in paths if p.is_file()):
            stat = path.stat()
            self.manifest.append(dict(path=path.relative_to(self.root).as_posix(), bytes=stat.st_size,
                                      sha256=sha256(path), modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()))
        self.csv("input_manifest.csv", pd.DataFrame(self.manifest))
        self.details.update({"generated_utc": datetime.now(timezone.utc).isoformat(), "root": str(self.root),
                             "git_head": subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip(),
                             "scope": "full existing building stock, raw LandsD/permit relations, Steps 3/5/7/9, public v1/v1.1, geometry, taxonomy, model caches, OSM height fields",
                             "limits": "No new survey validation; rule/ML labels are not independent ground truth. No repair, model refit or remote push.",
                             "checks": self.checks})
        (self.output / "summary.json").write_text(json.dumps(self.details, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"Audit complete: {self.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrites a previous audit")
    parser.add_argument("--terrain-samples", type=Path, help="Optional previously cached, ID-keyed point DTM evidence, used only diagnostically")
    args = parser.parse_args()
    Audit(args.root.resolve(), args.output.resolve()).run(args.terrain_samples)


if __name__ == "__main__":
    main()
