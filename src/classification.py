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
    # Assess the source object's scope before applying donation thresholds. A
    # polygon can strongly match one component while weakly matching another.
    raw_polygons = records[(pd.to_numeric(records.osm_area, errors="coerce") > 1) &
                           (intersection > 1) & records.Source_Object_ID.notna()].copy()
    raw_counts = raw_polygons.groupby("Source_Object_ID").BUILDINGSTRUCTUREID.nunique()
    ambiguous_objects = set(raw_counts[raw_counts.gt(1)].index)
    raw_polygons["_function_main"] = raw_polygons.apply(lambda row: building_tag_function(row)[0], axis=1)
    function_rows = raw_polygons[raw_polygons._function_main.ne(UNKNOWN)].copy()
    function_rows["_intersection_area"] = intersection.loc[function_rows.index]
    function_rows["_official_overlap"] = function_rows._intersection_area / pd.to_numeric(function_rows.SHAPE_Area, errors="coerce")
    function_rows["_union_area"] = pd.to_numeric(function_rows.osm_area, errors="coerce") + pd.to_numeric(function_rows.SHAPE_Area, errors="coerce") - function_rows._intersection_area
    function_rows["_iou"] = function_rows._intersection_area / function_rows._union_area
    function_rows["_raw_component_count"] = function_rows.Source_Object_ID.map(raw_counts)
    function_rows["_adoption_eligible"] = (
        function_rows._raw_component_count.eq(1) &
        pd.to_numeric(function_rows.overlap_ratio, errors="coerce").ge(.6) &
        function_rows._official_overlap.ge(.6) & function_rows._iou.ge(.5)
    )
    function_rows["_function_sub"] = function_rows.apply(lambda row: building_tag_function(row)[1], axis=1)
    evidence_by_building = {}
    for bid, group in function_rows.groupby("BUILDINGSTRUCTUREID", sort=True):
        evidence = []
        for _, row in group.sort_values("Source_Object_ID", kind="stable").drop_duplicates("Source_Object_ID").iterrows():
            evidence.append({
                "source_object_id": str(row.Source_Object_ID),
                "building": None if pd.isna(row.get("building")) else str(row.get("building")),
                "building_use": None if pd.isna(row.get("building:use")) else str(row.get("building:use")),
                "inferred_main": row._function_main,
                "inferred_sub": row._function_sub,
                "raw_matched_component_count": int(row._raw_component_count),
                "overlap_ratio": float(row.overlap_ratio),
                "intersection_area_sqm": float(row._intersection_area),
                "official_overlap_ratio": float(row._official_overlap),
                "iou": float(row._iou),
                "adoption_eligible": bool(row._adoption_eligible),
                "eligibility_scope": "component_overlap_only; source-use conflicts checked separately",
                "scope_status": "shared_component_scope" if row._raw_component_count > 1 else "single_component_scope",
            })
        evidence_by_building[str(bid)] = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    ambiguous_scope_ids = set(raw_polygons.loc[
        raw_polygons.Source_Object_ID.isin(ambiguous_objects), "BUILDINGSTRUCTUREID"
    ].astype(str))
    polygons = records[(pd.to_numeric(records.osm_area, errors="coerce") > 1) &
                       (pd.to_numeric(records.overlap_ratio, errors="coerce") >= .6) &
                       ((intersection / pd.to_numeric(records.SHAPE_Area, errors="coerce")) >= .6) &
                       ((intersection / union) >= .5)].copy()
    candidates = {}
    for bid, group in polygons.groupby("BUILDINGSTRUCTUREID", sort=True):
        functions = [building_tag_function(row) for _, row in group.drop_duplicates("Source_Object_ID").iterrows()]
        known = [(m, s) for m, s in functions if m != UNKNOWN]
        mains = sorted({m for m, s in known})
        if len(mains) == 1:
            children = sorted({s for m, s in known if s != UNKNOWN})
            candidates[str(bid)] = (mains[0], children[0] if len(children) == 1 else UNKNOWN)
    polygons = polygons[~polygons.Source_Object_ID.isin(ambiguous_objects)]
    primary = {}
    for bid, group in polygons.groupby("BUILDINGSTRUCTUREID", sort=True):
        functions = [building_tag_function(row) for _, row in group.drop_duplicates("Source_Object_ID").iterrows()]
        known = [(m, s) for m, s in functions if m != UNKNOWN]
        mains = sorted({m for m, s in known})
        if len(mains) == 1:
            children = sorted({s for m, s in known if s != UNKNOWN})
            primary[str(bid)] = (mains[0], children[0] if len(children) == 1 else UNKNOWN)
    # Reliable residential/composite records cannot be overwritten by a
    # valid exclusive non-residential polygon tag.
    bdbiar_class = frame.get("BDBIAR_CLASS", pd.Series("", index=frame.index)).fillna("").astype(str)
    has_residential = bdbiar_class.str.contains(r"residential|住宅", case=False, regex=True) & ~bdbiar_class.str.contains(r"non-residential|非住宅", case=False, regex=True)
    primary_main = frame.BUILDINGSTRUCTUREID.astype(str).map(lambda bid: primary.get(bid, (UNKNOWN, UNKNOWN))[0])
    function_conflicts = set(frame.loc[
        has_residential &
        frame.get("BDBIAR_Match_Quality", pd.Series("", index=frame.index)).astype(str).eq("inside_unique") &
        primary_main.isin([COMMERCIAL, INDUSTRIAL]),
        "BUILDINGSTRUCTUREID"
    ].astype(str))
    frame["Primary_Building_Main"] = frame.BUILDINGSTRUCTUREID.astype(str).map(lambda bid: primary.get(bid, (UNKNOWN, UNKNOWN))[0])
    frame["Primary_Building_Sub"] = frame.BUILDINGSTRUCTUREID.astype(str).map(lambda bid: primary.get(bid, (UNKNOWN, UNKNOWN))[1])
    frame["OSM_Function_Candidate_Main"] = frame.BUILDINGSTRUCTUREID.astype(str).map(lambda bid: candidates.get(bid, (UNKNOWN, UNKNOWN))[0])
    frame["OSM_Function_Candidate_Sub"] = frame.BUILDINGSTRUCTUREID.astype(str).map(lambda bid: candidates.get(bid, (UNKNOWN, UNKNOWN))[1])
    frame["OSM_Function_Evidence"] = frame.BUILDINGSTRUCTUREID.astype(str).map(lambda bid: evidence_by_building.get(bid, "[]"))
    scope_ambiguous = frame.BUILDINGSTRUCTUREID.astype(str).isin(ambiguous_scope_ids)
    scope_conflict = frame.BUILDINGSTRUCTUREID.astype(str).isin(function_conflicts)
    frame["Evidence_Subclass_Weights"] = frame.get('Evidence_Subclass_Weights',frame.Sub_Class_Proportions).fillna("")
    frame["Final_Main_Class"] = frame.Final_Main_Class.map(main_label)
    frame["Final_Sub_Class"] = frame.Final_Sub_Class.map(sub_label)
    # A whole-building office tag is stronger evidence of dominant function than
    # the number of mapped restaurants inside that office building.
    residential_mixed_evidence = frame.Final_Main_Class.eq(MIXED) & frame.Primary_Building_Main.eq(RESIDENTIAL)
    use_primary = frame.Primary_Building_Main.ne(UNKNOWN) & frame.Final_Main_Class.ne(NONASSESSED) & ~residential_mixed_evidence & ~scope_conflict
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
    frame.loc[scope_conflict, "Function_Quality"] = "ambiguous_component_scope"
    if scope_ambiguous.any() or scope_conflict.any():
        if "Classification_QA" not in frame:
            frame["Classification_QA"] = ""
        if scope_ambiguous.any():
            frame.loc[scope_ambiguous, "Classification_QA"] = frame.loc[scope_ambiguous, "Classification_QA"].fillna("").map(
                lambda value: ";".join(sorted(set(filter(None, str(value).split(";"))) | {"function_scope_ambiguous"}))
            )
        if scope_conflict.any():
            frame.loc[scope_conflict, "Classification_QA"] = frame.loc[scope_conflict, "Classification_QA"].fillna("").map(
                lambda value: ";".join(sorted(set(filter(None, str(value).split(";"))) | {"function_evidence_conflict"}))
            )
    frame.loc[scope_conflict, "Is_Conflicted"] = True

    # A strict active CSDI component name can identify the component's own
    # function, independently of labels inherited from its host building.
    frame["CSDI_Name_Function_Evidence"] = "[]"
    exact_component_rules = [
        (re.compile(r"\b(?:multi[ -]?storey\s+)?car\s*park\b|\bcarpark\b|停車場|停车场", re.I),
         re.compile(r"\b(?:shopping\s+)?(?:centre|center|mall|plaza)\b|商場|商场|購物中心|购物中心|商業中心|商业中心", re.I),
         COMMERCIAL, "Other Commercial_其他商业", "car_park"),
        (re.compile(r"\b(?:student|students|university|postgraduate|staff|trainee)\s+(?:hostels?|dormitor(?:y|ies)|dorm|residences?|quarters?|halls?(?:\s+of\s+residence)?)\b|\bhalls?\s+of\s+residence\b|學生宿舍|学生宿舍|研究生宿舍|宿舍", re.I),
         re.compile(r"\boffice\b|\bmulti[ -]?function(?:al)?\s+hall\b|\brefuse\s+store\s*room\b|\bworkshop\b|\bamenities\s+complex\b|宿舍處|宿舍处|辦事處|办公室|辦公室|綜合禮堂|综合礼堂|多用途禮堂|多用途礼堂|垃圾房|工場|工场", re.I),
         RESIDENTIAL, "Other Housing_其他房屋", "student_residence"),
        (re.compile(r"\b(?:primary|secondary|international|special|vocational)?\s*school\b|\bcollege\b|\buniversity\b|學校|学校|中學|中学|小學|小学|大學|大学|學院|学院", re.I),
         re.compile(r"\bstudent\s+cent(?:re|er)\b|\bclinic\b|\bhospital\b|\bmedical\s+cent(?:re|er)\b|\bhealth\s+cent(?:re|er)\b|\bcare\s+home\b|\bnursing\s+home\b|\bchurch\b|\bchapel\b|學生中心|學生會|诊所|診所|醫院|医院|醫療中心|医疗中心|安老院|護老院|老人院|教堂|禮拜堂|礼拜堂", re.I),
         COMMERCIAL, "Education_教育", "school"),
        (re.compile(r"\bhotel\b|\bguest\s+house\b|\bhostel\b|酒店|賓館|宾馆|旅館|旅馆", re.I),
         re.compile(r"\boffice\b|\bmulti[ -]?function(?:al)?\s+hall\b|\bworkshop\b|辦事處|辦公室|办公室|工場|工场", re.I),
         COMMERCIAL, "Accommodation_住宿", "hotel_or_hostel"),
        (re.compile(r"^\s*(?:residents(?:'|’)?\s+club(?:\s+(?:住戶俱樂部|住客會所))?|assembly\s+hall(?:\s+禮堂)?|住戶俱樂部|住客會所|禮堂)\s*$", re.I),
         None, COMMERCIAL, "Other Commercial_其他商业", "community_hall_or_club"),
    ]
    component_name_main = pd.Series(UNKNOWN, index=frame.index, dtype=object)
    component_name_sub = pd.Series(UNKNOWN, index=frame.index, dtype=object)
    component_name_kind = pd.Series("", index=frame.index, dtype=object)
    name_en = frame.get("CSDI_Building_Name_EN", pd.Series("", index=frame.index)).fillna("").astype(str)
    name_tc = frame.get("CSDI_Building_Name_TC", pd.Series("", index=frame.index)).fillna("").astype(str)
    component_names = name_en + " " + name_tc
    for pattern, excluded, rule_main, rule_sub, kind in exact_component_rules:
        matched = component_names.str.contains(pattern)
        if excluded is not None:
            blocked_child = component_names.str.contains(excluded)
            if kind == 'school':
                explicit_education = component_names.str.contains(
                    r"\b(?:primary|secondary|international|special|vocational)?\s*school\b|\bcollege\b|學校|学校|中學|中学|小學|小学|學院|学院", flags=re.I)
                matched &= ~blocked_child | explicit_education
                matched &= ~component_names.str.contains(
                    r"\bclinic\b|\bstudent\s+(?:cent(?:re|er)|association|union)\b|\b(?:medical|health)\s+cent(?:re|er)\b|\b(?:care|nursing)\s+home\b|\boffice\b|\bmulti[ -]?function(?:al)?\s+hall\b|\brefuse\s+store\s*room\b|診所|诊所|學生會|学生会|學生中心|学生中心|護老院|安老院|老人院|宿舍處|宿舍处|綜合禮堂|综合礼堂|多用途禮堂|多用途礼堂|垃圾房", flags=re.I)
            else:
                matched &= ~blocked_child
        matched &= component_name_main.eq(UNKNOWN)
        component_name_main.loc[matched] = rule_main
        component_name_sub.loc[matched] = rule_sub
        component_name_kind.loc[matched] = kind
    component_iou = pd.to_numeric(frame.get("CSDI_Geometry_IoU", pd.Series(np.nan, index=frame.index)), errors="coerce")
    exact_component_match = (
        frame.get("CSDI_Match_Quality", pd.Series("", index=frame.index)).eq("same_component_id_type_footprint") &
        frame.get("CSDI_Status", pd.Series("", index=frame.index)).eq("Active") & component_iou.ge(.95)
    )
    compound_school_housing = component_names.str.contains(
        r"\b(?:school(?:\s+of\s+(?:general\s+)?nursing)?|college)\s*(?:&|and|cum)\s*(?:quarters?|dormitor(?:y|ies)|hostels?)\b|\bquarters?\s+cum\s+school\b|(?:學校|学校|中學|中学)(?:[（(][^）)]*[）)])?\s*(?:及|兼|與|与)\s*宿舍", flags=re.I)
    residence_child = component_names.str.contains(exact_component_rules[1][0]) & component_names.str.contains(exact_component_rules[1][1])
    hostel_workshop = component_names.str.contains(r"\bhostels?\b|宿舍", flags=re.I) & component_names.str.contains(r"\bworkshop\b|工場|工场", flags=re.I)
    school_association = component_names.str.contains(r"\bschool\b|\bcollege\b|\buniversity\b|學校|学校|中學|中学|學院|学院|大學|大学", flags=re.I) & component_names.str.contains(r"\bstudent\s+(?:association|union)\b|學生會|学生会", flags=re.I)
    unresolved_name = (compound_school_housing | residence_child | hostel_workshop | school_association) & ~frame.Final_Main_Class.eq(MIXED)
    component_name_main.loc[unresolved_name] = UNKNOWN
    component_name_sub.loc[unresolved_name] = UNKNOWN
    component_name_kind.loc[unresolved_name] = 'single_function_scope_unresolved'
    valid_named_component = exact_component_match & (component_name_main.ne(UNKNOWN) | unresolved_name) & ~frame.Final_Main_Class.eq(MIXED)
    for index in frame.index[valid_named_component]:
        qa_before = frame.at[index, "Classification_QA"] if "Classification_QA" in frame else ""
        conflict_before = frame.at[index, "Is_Conflicted"] if "Is_Conflicted" in frame else False
        evidence = {
            "matched_by": {"match_quality": frame.at[index, "CSDI_Match_Quality"],
                           "status": frame.at[index, "CSDI_Status"], "geometry_iou_minimum": 0.95},
            "component_name_en": name_en.at[index] or None,
            "component_name_tc": name_tc.at[index] or None,
            "component_name_rule": component_name_kind.at[index],
            "geometry_iou": float(component_iou.at[index]),
            "before": {"main": frame.at[index, "Final_Main_Class"],
                       "sub": frame.at[index, "Final_Sub_Class"],
                       "classification_source": frame.at[index, "Classification_Source"],
                       "classification_qa": None if pd.isna(qa_before) else qa_before,
                       "is_conflicted": None if pd.isna(conflict_before) else bool(conflict_before),
                       **{field.lower(): None if field not in frame or pd.isna(frame.at[index, field]) else frame.at[index, field]
                          for field in ['Evidence_Subclass_Weights', 'Functional_Main_Classes', 'Functional_Subclasses']}},
        }
        frame.at[index, "CSDI_Name_Function_Evidence"] = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        frame.at[index, "Final_Main_Class"] = component_name_main.at[index]
        frame.at[index, "Final_Sub_Class"] = component_name_sub.at[index]
        unresolved = bool(unresolved_name.at[index])
        frame.at[index, "Classification_Source"] = "CSDI_Component_Name_Scope_Unresolved" if unresolved else "CSDI_Component_Name_Keyword"
        frame.at[index, "Function_Quality"] = "unknown" if unresolved else "source_inferred"
        frame.at[index, "Is_Conflicted"] = unresolved
        frame.at[index, "Evidence_Subclass_Weights"] = "" if unresolved else format_weights({component_name_sub.at[index]: 1.})
        if "Classification_QA" in frame:
            qa_text = str(qa_before) if pd.notna(qa_before) else ""
            if unresolved:
                frame.at[index, "Classification_QA"] = ';'.join(sorted(set(filter(None,qa_text.split(';'))) | {'function_evidence_conflict','function_scope_ambiguous'}))
                continue
            remaining = [flag for flag in qa_text.split(";") if flag and flag not in {
                "function_scope_ambiguous", "function_evidence_conflict", "subclass_parent_conflict"}]
            frame.at[index, "Classification_QA"] = ";".join(remaining)
            if remaining:
                frame.at[index, "Is_Conflicted"] = bool(conflict_before)

    # Generic station names remain unclassified; only explicit substation names
    # with the existing component-use and geometry gate are adopted below.
    station_name = re.compile(r"\bsub[\s-]?station\b|\btransformer\s+station\b|變電站|变电站|變壓站|变压站", re.I)
    component_use = frame.get("CSDI_Component_Use_Type", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower().eq("substation")
    name_en = frame.get("CSDI_Building_Name_EN", pd.Series("", index=frame.index)).fillna("").astype(str)
    name_tc = frame.get("CSDI_Building_Name_TC", pd.Series("", index=frame.index)).fillna("").astype(str)
    named_station = name_en.str.contains(station_name) | name_tc.str.contains(station_name)
    iou = pd.to_numeric(frame.get("CSDI_Geometry_IoU", pd.Series(np.nan, index=frame.index)), errors="coerce")
    valid_station = (
        component_use & named_station &
        frame.get("CSDI_Match_Quality", pd.Series("", index=frame.index)).eq("same_component_id_type_footprint") &
        frame.get("CSDI_Status", pd.Series("", index=frame.index)).eq("Active") & iou.ge(.95)
    )
    for index in frame.index[valid_station]:
        qa_before = frame.at[index, "Classification_QA"] if "Classification_QA" in frame else ""
        conflict_before = frame.at[index, "Is_Conflicted"] if "Is_Conflicted" in frame else False
        csuid = frame.at[index, "BUILDINGCSUID"] if "BUILDINGCSUID" in frame else None
        native_id = frame.at[index, "CSDI_Native_Building_ID"] if "CSDI_Native_Building_ID" in frame else None
        native_id = None if pd.isna(native_id) else native_id.item() if hasattr(native_id, 'item') else native_id
        evidence = {
            "matched_by": {
                "component_use_type": "substation",
                "match_quality": frame.at[index, "CSDI_Match_Quality"],
                "status": frame.at[index, "CSDI_Status"],
                "geometry_iou_minimum": 0.95,
            },
            "source_name_en": name_en.at[index] or None,
            "source_name_tc": name_tc.at[index] or None,
            "csuid": None if pd.isna(csuid) else str(csuid),
            "native_building_id": native_id,
            "geometry_iou": float(iou.at[index]),
            "official_source_url": "https://portal.csdi.gov.hk/server/rest/services/common/landsd_rcd_1637211194312_35158/MapServer/0",
            "frozen_source": {
                "path": "data/sources/csdi_official_20261020.parquet",
                "metadata_path": "data/sources/csdi_official_20261020.metadata.json",
                "checksum_reference": "The frozen source metadata and pipeline input manifest retain the exact SHA256 and snapshot/retrieval dates.",
                "note": "Frozen official Lands Department/CSDI component record; no claim of a new live retrieval.",
            },
            "adopted_rule": "An exact, active CSDI component match with an explicit substation/transformer-station component name maps to Commercial_商业类别 / Other Commercial_其他商业; host-building tags and classifications do not transfer to this component.",
            "before": {
                "main": frame.at[index, "Final_Main_Class"],
                "sub": frame.at[index, "Final_Sub_Class"],
                "classification_source": frame.at[index, "Classification_Source"],
                "classification_qa": None if pd.isna(qa_before) else qa_before,
                "is_conflicted": None if pd.isna(conflict_before) else bool(conflict_before),
                **{field.lower(): None if field not in frame or pd.isna(frame.at[index, field]) else frame.at[index, field]
                   for field in ['Evidence_Subclass_Weights', 'Functional_Main_Classes', 'Functional_Subclasses']},
                "component_name_evidence": json.loads(frame.at[index, "CSDI_Name_Function_Evidence"]),
            },
        }
        frame.at[index, "CSDI_Name_Function_Evidence"] = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        frame.at[index, "Final_Main_Class"] = COMMERCIAL
        frame.at[index, "Final_Sub_Class"] = "Other Commercial_其他商业"
        frame.at[index, "Evidence_Subclass_Weights"] = format_weights({"Other Commercial_其他商业": 1.})
        frame.at[index, "Classification_Source"] = "CSDI_Component_Name_Keyword"
        frame.at[index, "Function_Quality"] = "source_inferred"
        qa = "" if pd.isna(qa_before) else str(qa_before)
        removable = {"function_scope_ambiguous", "function_evidence_conflict", "subclass_parent_conflict"}
        remaining = [flag for flag in qa.split(";") if flag and flag not in removable]
        frame.at[index, "Classification_QA"] = ";".join(remaining)
        if scope_conflict.at[index] or (not remaining and bool(set(qa.split(";")) & removable)):
            frame.at[index, "Is_Conflicted"] = False
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
        has_component_name_function = row.get('CSDI_Name_Function_Evidence') not in (None, '', '[]')
        if str(row.get('Is_Conflicted','')).lower() == 'true' and not has_component_name_function:
            flags.append('function_evidence_conflict')
        if main not in [MIXED, UNKNOWN] and sub_parent is not None and sub_parent != main:
            sub = UNKNOWN
            flags.append("subclass_parent_conflict")
        evidence = parse_weights(row.get("Evidence_Subclass_Weights", row.get("Sub_Class_Proportions", "")))
        components = {sub} if sub != UNKNOWN and main != UNKNOWN else set()
        if main == MIXED:
            components |= {sub_label(k) for k in evidence if sub_label(k) != UNKNOWN}
        main_components = {taxonomy().get(k) for k in components} - {None, NONASSESSED}
        main_components |= {main_label(k) for k in evidence if main_label(k) in MAIN_CLASSES}
        if main in MAIN_CLASSES:
            main_components = {main}
        elif main == UNKNOWN:
            main_components = set()
        area_shares = {}
        residential_share, nondomestic_share = np.nan, np.nan
        permit_values = [pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors='coerce').iloc[0]
                         for name in ('GFA_DOMESTIC_SUM', 'GFA_NONDOMESTIC_SUM')]
        has_permit_account = any(pd.notna(value) and value >= 0 for value in permit_values)
        if has_permit_account:
            flags.append('permit_accounting_not_observed_use')
            physical_scope_verified = str(row.get('Permit_GFA_Physical_Scope_Verified', False)).lower() == 'true'
            if not physical_scope_verified:
                flags.append('permit_physical_scope_unverified')
        if row.get('Permit_GFA_Scope') in {
            'single_structure_component_scope_unresolved',
            'single_structure_component_use_scope_unresolved',
        }:
            flags.append('permit_component_scope_unresolved')
        if row.get('Registered_GFA_Source') == 'LandsD_structure_GFA_component_scope_unresolved':
            flags.append('registered_gfa_component_scope_unresolved')
        if main in MAIN_CLASSES and sub != UNKNOWN:
            vector = format_weights({sub: 1})
            method, quality = "single_function_assumption", "assumed_not_measured"
        elif main == NONASSESSED and sub != UNKNOWN:
            vector, method, quality = format_weights({sub: 1}), "nonassessed_label", "not_an_area_estimate"
        else:
            vector = ""
            method = "unknown_no_use_area_evidence"
            quality = "unknown"
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
