"""Canonical function labels, evidence weights and explicitly scoped area shares."""

import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from config import KEYWORDS_FILE
from utils import canonicalize_main_class_label, canonicalize_sub_class_label

UNKNOWN = "Unknown_未知类别"
MIXED = "Mixed-use_混合用途"
RESIDENTIAL = "Residential_住宅类别"
COMMERCIAL = "Commercial_商业类别"
INDUSTRIAL = "Industrial_工业类别"
NONASSESSED = "Non-assessed_非评估类别"
MAIN_CLASSES = [RESIDENTIAL, COMMERCIAL, INDUSTRIAL]
VECTOR_TOKEN = re.compile(r"\s*(.*?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?=-|$)")


@lru_cache(maxsize=1)
def taxonomy():
    with open(KEYWORDS_FILE, encoding="utf-8") as stream:
        raw = json.load(stream)
    parents = {}
    for main, subs in raw.items():
        if main == "__STRONG_KEYWORDS__":
            continue
        for sub in subs:
            parents[canonicalize_sub_class_label(sub)] = canonicalize_main_class_label(main)
    return parents


def main_label(value):
    result = canonicalize_main_class_label(value)
    return result if result in MAIN_CLASSES + [UNKNOWN, MIXED, NONASSESSED] else UNKNOWN


def sub_label(value):
    if pd.isna(value):
        return UNKNOWN
    result = canonicalize_sub_class_label(value)
    return result if result in taxonomy() else UNKNOWN


def parse_weights(value):
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if np.isfinite(v) and v > 0}
    if pd.isna(value):
        return {}
    text, pos, pairs = str(value).strip(), 0, {}
    while pos < len(text):
        match = VECTOR_TOKEN.match(text, pos)
        if match is None:
            return {}
        pairs[match[1].strip()] = pairs.get(match[1].strip(), 0.) + float(match[2])
        pos = match.end() + (match.end() < len(text))
    return pairs


def format_weights(weights):
    clean = {str(k): float(v) for k, v in weights.items() if np.isfinite(v) and v > 0}
    total = sum(clean.values())
    if total <= 0:
        return ""
    # Retain small shares and enough precision to conserve mass after serialization.
    return "-".join(f"{key}:{value / total:.8f}" for key, value in sorted(clean.items()))


def building_tag_function(row):
    mapping = {
        "apartments": (RESIDENTIAL, UNKNOWN),
        "residential": (RESIDENTIAL, UNKNOWN),
        "house": (RESIDENTIAL, UNKNOWN),
        "dormitory": (RESIDENTIAL, "Other Housing_其他房屋"),
        "office": (COMMERCIAL, "Office_办公室"),
        "school": (COMMERCIAL, "Education_教育"),
        "university": (COMMERCIAL, "Education_教育"),
        "hospital": (COMMERCIAL, "Human Health_医疗"),
        "hotel": (COMMERCIAL, "Accommodation_住宿"),
        "retail": (COMMERCIAL, "Retail_零售"),
        "commercial": (COMMERCIAL, UNKNOWN),
        "industrial": (INDUSTRIAL, UNKNOWN),
        "warehouse": (INDUSTRIAL, "Non-manufacturing_非制造业"),
    }
    for col in ["building:use", "building"]:
        value = str(row.get(col, "")).lower().strip()
        if value in mapping:
            return mapping[value]
    office = str(row.get("office", "")).strip().lower()
    if office not in {"", "none", "nan", "null", "no"}:
        return COMMERCIAL, "Office_办公室"
    return UNKNOWN, UNKNOWN


def enrich_function_evidence(frame, aggregated):
    frame = frame.copy()
    records = aggregated.copy()
    intersection = pd.to_numeric(records.overlap_ratio, errors="coerce") * pd.to_numeric(records.osm_area, errors="coerce")
    union = pd.to_numeric(records.osm_area, errors="coerce") + pd.to_numeric(records.SHAPE_Area, errors="coerce") - intersection
    polygons = records[(pd.to_numeric(records.osm_area, errors="coerce") > 1) &
                       (pd.to_numeric(records.overlap_ratio, errors="coerce") >= .6) &
                       ((intersection / pd.to_numeric(records.SHAPE_Area, errors="coerce")) >= .6) &
                       ((intersection / union) >= .5)].copy()
    counts = polygons.groupby('Source_Object_ID').BUILDINGSTRUCTUREID.nunique()
    polygons = polygons[~polygons.Source_Object_ID.isin(counts[counts.gt(1)].index)]
    primary = {}
    for bid, group in polygons.groupby("BUILDINGSTRUCTUREID", sort=True):
        functions = [building_tag_function(row) for _, row in group.drop_duplicates("Source_Object_ID").iterrows()]
        known = [(m, s) for m, s in functions if m != UNKNOWN]
        mains = sorted({m for m, s in known})
        if len(mains) == 1:
            children = sorted({s for m, s in known if s != UNKNOWN})
            primary[str(bid)] = (mains[0], children[0] if len(children) == 1 else UNKNOWN)
    frame["Primary_Building_Main"] = frame.BUILDINGSTRUCTUREID.astype(str).map(lambda bid: primary.get(bid, (UNKNOWN, UNKNOWN))[0])
    frame["Primary_Building_Sub"] = frame.BUILDINGSTRUCTUREID.astype(str).map(lambda bid: primary.get(bid, (UNKNOWN, UNKNOWN))[1])
    frame["Evidence_Subclass_Weights"] = frame.get('Evidence_Subclass_Weights',frame.Sub_Class_Proportions).fillna("")
    frame["Final_Main_Class"] = frame.Final_Main_Class.map(main_label)
    frame["Final_Sub_Class"] = frame.Final_Sub_Class.map(sub_label)
    # A whole-building office tag is stronger evidence of dominant function than
    # the number of mapped restaurants inside that office building.
    residential_mixed_evidence = frame.Final_Main_Class.eq(MIXED) & frame.Primary_Building_Main.eq(RESIDENTIAL)
    use_primary = frame.Primary_Building_Main.ne(UNKNOWN) & frame.Final_Main_Class.ne(NONASSESSED) & ~residential_mixed_evidence
    existing_parent = frame.Final_Sub_Class.map(taxonomy())
    independent_sub = frame.Classification_Source.fillna('').str.contains('LandsD_Category|Keyword|BDBIAR')
    retain_sub = use_primary & frame.Primary_Building_Sub.eq(UNKNOWN) & existing_parent.eq(frame.Primary_Building_Main) & independent_sub
    retained_sub_sources = frame.loc[retain_sub,'Classification_Source'].copy()
    frame.loc[retain_sub,'Primary_Building_Sub'] = frame.loc[retain_sub,'Final_Sub_Class']
    frame.loc[use_primary, "Final_Main_Class"] = frame.loc[use_primary, "Primary_Building_Main"]
    frame.loc[use_primary, "Final_Sub_Class"] = frame.loc[use_primary, "Primary_Building_Sub"]
    frame.loc[use_primary, "Classification_Source"] = "OSM_Building_Function"
    frame.loc[retain_sub,'Classification_Source'] += '+' + retained_sub_sources
    frame["Function_Quality"] = np.where(frame.Final_Main_Class.eq(UNKNOWN), "unknown", "source_inferred")
    return finalize_function_fields(frame)


def finalize_function_fields(frame, main_col="Final_Main_Class", sub_col="Final_Sub_Class"):
    """Never present POI counts or classifier probabilities as observed area shares."""
    frame = frame.copy()
    frame[main_col] = frame[main_col].map(main_label)
    frame[sub_col] = frame[sub_col].map(sub_label)
    output = []
    for _, row in frame.iterrows():
        main, sub = row[main_col], row[sub_col]
        sub_parent = taxonomy().get(sub)
        previous = row.get('Classification_QA', '')
        flags = str(previous).split(';') if pd.notna(previous) and str(previous) else []
        flags = [flag for flag in flags if flag != 'registered_nonresidential_function_unspecified']
        if str(row.get('Is_Conflicted','')).lower() == 'true':
            flags.append('function_evidence_conflict')
        if main not in [MIXED, UNKNOWN] and sub_parent is not None and sub_parent != main:
            sub = UNKNOWN
            flags.append("subclass_parent_conflict")
        evidence = parse_weights(row.get("Evidence_Subclass_Weights", row.get("Sub_Class_Proportions", "")))
        components = {sub} if sub != UNKNOWN else set()
        if main == MIXED:
            components |= {sub_label(k) for k in evidence if sub_label(k) != UNKNOWN}
        main_components = {taxonomy().get(k) for k in components} - {None, NONASSESSED}
        main_components |= {main_label(k) for k in evidence if main_label(k) in MAIN_CLASSES}
        if main in MAIN_CLASSES:
            main_components = {main}
        dom, nondom = row.get("GFA_DOMESTIC_SUM", np.nan), row.get("GFA_NONDOMESTIC_SUM", np.nan)
        area_shares = {}
        residential_share, nondomestic_share = np.nan, np.nan
        area_conflict = False
        if pd.notna(dom) and pd.notna(nondom) and dom >= 0 and nondom >= 0 and dom + nondom > 0:
            residential_share, nondomestic_share = float(dom/(dom+nondom)),float(nondom/(dom+nondom))
            parents = {taxonomy().get(k) for k in components} - {None, RESIDENTIAL, NONASSESSED}
            nonres = main if main in [COMMERCIAL, INDUSTRIAL] else next(iter(parents)) if len(parents) == 1 else None
            if dom > 0:
                area_shares[RESIDENTIAL] = float(dom / (dom + nondom))
            if nondom > 0 and nonres is not None:
                area_shares[nonres] = float(nondom / (dom + nondom))
            if nondom > 0 and nonres is None:
                area_shares = {}
            if dom > 0 and nondom > 0:
                main = MIXED
                main_components.add(RESIDENTIAL)
                if nonres is not None:
                    main_components.add(nonres)
                else:
                    flags.append('registered_nonresidential_function_unspecified')
            elif (dom > 0 and main in [COMMERCIAL,INDUSTRIAL]) or (nondom > 0 and main == RESIDENTIAL):
                flags.append('registered_area_function_conflict')
                area_conflict = True
        if main in MAIN_CLASSES and sub != UNKNOWN and not area_conflict:
            vector = format_weights({sub: 1})
            method, quality = "single_function_assumption", "assumed_not_measured"
        elif main == NONASSESSED and sub != UNKNOWN:
            vector, method, quality = format_weights({sub: 1}), "nonassessed_label", "not_an_area_estimate"
        else:
            vector = ""
            method = "registered_main_area_only" if area_shares else "unknown_no_use_area_evidence"
            quality = "main_area_observed_subarea_unknown" if area_shares else "unknown"
        output.append((main, sub, vector, format_weights(area_shares), method, quality,
                       json.dumps(sorted(components), ensure_ascii=False),
                       json.dumps(sorted(main_components), ensure_ascii=False), residential_share,nondomestic_share,";".join(sorted(set(flags)))))
    cols = [main_col, sub_col, "Sub_Class_Proportions", "Observed_Main_Area_Proportions", "Proportion_Method",
            "Proportion_Quality", "Functional_Subclasses", "Functional_Main_Classes",
            "Observed_Residential_Area_Share", "Observed_Nondomestic_Area_Share", "Classification_QA"]
    frame[cols] = pd.DataFrame(output, index=frame.index)
    frame["Main_Class_Proportions"] = frame.Observed_Main_Area_Proportions
    frame["Is_Mixed_Use"] = frame[main_col].eq(MIXED)
    return frame
