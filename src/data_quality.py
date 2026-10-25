"""Physical attributes and source-scope checks shared by the numbered pipeline."""

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ID = "BUILDINGSTRUCTUREID"


def ids(values):
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def numbers(frame, column):
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def require_unique(frame, column=ID):
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"Missing/duplicate {column}; refusing an ambiguous building merge")


def match_bdbiar_points(footprints, points, max_distance=30):
    """Containment wins; only a unique point in a unique footprint is evidence."""
    require_unique(footprints)
    require_unique(points, 'BDBIAR_OBJECTID')
    points = points.to_crs(footprints.crs).reset_index(drop=True)
    pairs = points.sindex.query(footprints.geometry, predicate='intersects')
    owners = np.bincount(pairs[1], minlength=len(points))
    points['BDBIAR_Containing_Footprint_Count'] = owners
    joined = gpd.sjoin_nearest(footprints, points, how='left',
                              max_distance=max_distance, distance_col='distance_to_bdbiar')
    ties = joined.groupby(ID).BDBIAR_OBJECTID.nunique()
    result = joined.sort_values([ID, 'distance_to_bdbiar', 'BDBIAR_OBJECTID'],
                                kind='stable').drop_duplicates(ID).copy()
    result['BDBIAR_Nearest_Tie_Count'] = result[ID].map(ties)
    result['BDBIAR_Match_Quality'] = np.select(
        [result.BDBIAR_OBJECTID.isna(), result.BDBIAR_Nearest_Tie_Count.gt(1),
         result.distance_to_bdbiar.gt(0), result.BDBIAR_Containing_Footprint_Count.gt(1)],
        ['unmatched', 'ambiguous_nearest_points', 'nearby_only', 'shared_point'],
        default='inside_unique')
    return result.drop(columns='index_right')


def repair_footprints(frame):
    frame = frame.copy()
    frame[ID] = ids(frame[ID])
    require_unique(frame)
    if frame.crs is None or frame.crs.to_epsg() != 2326:
        raise ValueError("LandsD footprint CRS must explicitly be EPSG:2326")
    frame["Source_Footprint_Area_sqm"] = numbers(frame, "SHAPE_Area")
    frame["Source_Footprint_Length_m"] = numbers(frame, "SHAPE_Length")
    invalid = ~frame.geometry.is_valid
    for index in frame.index[invalid]:
        repaired = shapely.make_valid(frame.at[index, "geometry"])
        if repaired.geom_type == "GeometryCollection":
            polygons = [part for part in shapely.get_parts(repaired) if part.geom_type in {"Polygon", "MultiPolygon"}]
            repaired = shapely.union_all(polygons)
        if repaired.is_empty or repaired.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"Unrecoverable footprint: {frame.at[index, ID]}")
        frame.at[index, "geometry"] = repaired
    frame["Geometry_Quality"] = np.where(invalid, "repaired_make_valid", "source_valid")
    frame["SHAPE_Area"] = frame.geometry.area
    frame["SHAPE_Length"] = frame.geometry.length
    delta = (frame.SHAPE_Area - frame.Source_Footprint_Area_sqm).abs()
    frame["Footprint_Area_QA"] = np.where(delta > np.maximum(1, frame.SHAPE_Area * .01),
                                           "source_attribute_replaced_by_geometry", "geometry_area")
    frame["Duplicate_Footprint_Geometry"] = frame.geometry.to_wkb().duplicated(keep=False)
    return frame


def permit_attributes(permits, links):
    """Never allocate a permit covering several structures to every structure."""
    permits, links = permits.copy(), links.copy()
    require_unique(permits, "OPNO")
    links[ID] = ids(links[ID])
    if links.duplicated([ID, "OPNO"]).any():
        raise ValueError("Duplicate permit/building links")
    for col in ["DOMESTICGFA", "NONDOMESTICGFA"]:
        permits[col] = numbers(permits, col)
    merged = links.merge(permits, on="OPNO", how="left", validate="many_to_one", indicator=True)
    if merged._merge.eq("left_only").any():
        raise ValueError("Occupation-permit foreign key is missing")
    scope = links.groupby("OPNO")[ID].nunique()
    merged["structures_per_permit"] = merged.OPNO.map(scope)
    rows = []
    for bid, group in merged.groupby(ID, sort=True):
        single = len(group) == 1 and group.structures_per_permit.iloc[0] == 1
        domestic = group.DOMESTICGFA.iloc[0] if single else np.nan
        nondomestic = group.NONDOMESTICGFA.iloc[0] if single else np.nan
        rows.append({ID: bid, "Permit_IDs": json.dumps(sorted(group.OPNO.astype(str).unique().tolist())),
                     "Permit_GFA_Scope": "single_structure_component_scope_unresolved" if single else "shared_or_multiple_permits",
                     "Permit_GFA_Link_QA": "single_administrative_link_physical_scope_unverified" if single else "shared_or_multiple_links",
                     "Permit_GFA_Usable": False,
                     "Permit_GFA_Physical_Scope_Verified": False,
                     "GFA_DOMESTIC_SUM": domestic if single else np.nan,
                     "GFA_NONDOMESTIC_SUM": nondomestic if single else np.nan,
                     "Permit_Raw_DOMESTICGFA": domestic if single else np.nan,
                     "Permit_Raw_NONDOMESTICGFA": nondomestic if single else np.nan,
                     "Permit_Group_Domestic_GFA": group.DOMESTICGFA.sum(min_count=1),
                     "Permit_Group_Nondomestic_GFA": group.NONDOMESTICGFA.sum(min_count=1)})
    return pd.DataFrame(rows)


def deduplicate_osm(frame):
    """Merge attributes of the same source object/geometry, retaining conflicts."""
    frame = frame.copy()
    source = frame.get("data_source", pd.Series("osm", index=frame.index)).fillna("osm")
    kind = frame.get("osm_type", frame.geometry.geom_type.map({"Point": "node", "Polygon": "way", "MultiPolygon": "relation"})).fillna("place")
    frame["Source_Object_ID"] = source + ":" + kind + ":" + ids(frame["osmid"])
    frame["_wkb"] = frame.geometry.to_wkb(hex=True)
    duplicate = frame.duplicated(["Source_Object_ID", "_wkb"], keep=False)
    unique = frame[~duplicate].copy()
    merged = []
    for _, group in frame[duplicate].groupby(["Source_Object_ID", "_wkb"], sort=True):
        row = group.iloc[0].copy()
        conflicts = []
        for col in group.columns.difference(["geometry", "_wkb"]):
            available = group[col].dropna()
            available = available[~available.astype(str).str.lower().isin(["", "none", "nan", "null"])]
            if available.empty:
                continue
            distinct = available.astype(str).unique()
            row[col] = available.iloc[0]
            if len(distinct) > 1 and col in ["height", "building:height", "building:levels", "building:min_level"]:
                row[col] = ";".join(sorted(distinct))
                conflicts.append(col)
        row["OSM_Attribute_Conflicts"] = ";".join(conflicts)
        merged.append(row)
    if merged:
        unique = pd.concat([unique, pd.DataFrame(merged)], ignore_index=True)
    unique = unique.sort_values(["Source_Object_ID", "_wkb"], kind="stable").reset_index(drop=True)
    unique["Source_Feature_ID"] = unique.Source_Object_ID + ":" + unique._wkb.map(lambda text: __import__("hashlib").sha256(text.encode()).hexdigest()[:16])
    return gpd.GeoDataFrame(unique.drop(columns="_wkb"), geometry="geometry", crs=frame.crs)


def parse_height(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(m|metres?|meters?|ft|feet|')?(?:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:in|\"))?", text)
    if not match:
        return np.nan
    value = float(match[1])
    if match[2] in {"ft", "feet", "'"}:
        value = value * .3048 + float(match[3] or 0) * .0254
    elif match[3] is not None:
        return np.nan
    return value if 0 < value <= 600 else np.nan


def parse_storeys(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) and result.is_integer() and 0 < result <= 150 else np.nan


def dimension_matches(aggregated):
    """Match polygon dimensions and keep raw source-object component scope as QA."""
    attrs = aggregated.copy().reset_index(drop=True)
    overlap = numbers(attrs, "overlap_ratio")
    osm_area, area = numbers(attrs, "osm_area"), numbers(attrs, "SHAPE_Area")
    intersection = overlap * osm_area
    # Establish source-object scope before applying the stricter dimension
    # donation thresholds. A small but real intersection still shows that the
    # same OSM geometry may describe more than one official component.
    raw_scope_rows = attrs.loc[(osm_area > 1) & (intersection > 1) & attrs[ID].notna()].copy()
    raw_scope_rows["_source_object_id"] = raw_scope_rows.get(
        "Source_Object_ID", pd.Series(pd.NA, index=raw_scope_rows.index)).astype("string").str.strip()
    raw_scope_rows = raw_scope_rows.loc[raw_scope_rows["_source_object_id"].notna() &
                                       raw_scope_rows["_source_object_id"].ne("")]
    raw_scope_counts = raw_scope_rows.groupby("_source_object_id")[ID].nunique()
    raw_scope_pairs = set(zip(raw_scope_rows["_source_object_id"].astype(str), raw_scope_rows[ID].astype(str)))
    component_overlap = intersection / area
    attrs["dimension_iou"] = intersection / (osm_area + area - intersection)
    candidates = attrs[(osm_area > 1) & (overlap >= .6) & (component_overlap >= .6) & (attrs.dimension_iou >= .5)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=[ID, "OSM_Height_m", "OSM_Storeys", "OSM_Dimension_Match_QA",
            "OSM_Dimension_Object_ID", "OSM_Dimension_Scope_QA",
            "OSM_Dimension_Raw_Matched_Component_Count", "OSM_Dimension_Candidate_Evidence"])
    counts = candidates.groupby("Source_Feature_ID")[ID].nunique()
    candidates["_shared"] = candidates.Source_Feature_ID.map(counts).gt(1)
    rows = []
    for bid, group in candidates.groupby(ID, sort=True):
        # Equally good overlapping polygons remain a conflict instead of donating arbitrary dimensions.
        best = group[group.dimension_iou >= group.dimension_iou.max() - .05]
        heights = numbers(best, "osm_height").dropna().unique()
        floors = numbers(best, "osm_storeys").dropna().unique()
        shared = best._shared.any()
        conflict = len(heights) > 1 or len(floors) > 1
        reported_conflict = best.get('OSM_Attribute_Conflicts',pd.Series('',index=best.index)).fillna('').str.len().gt(0).any()
        object_ids = best.get("Source_Object_ID", pd.Series(pd.NA, index=best.index)).astype("string").str.strip()
        object_counts = object_ids.map(raw_scope_counts)
        raw_component_match = pd.Series(
            [(pd.notna(source_id) and str(source_id) != "" and
              (str(source_id), str(bid)) in raw_scope_pairs) for source_id in object_ids],
            index=best.index)
        missing_scope = object_ids.isna() | object_ids.eq("") | object_counts.isna() | ~raw_component_match
        shared_scope = (object_counts.gt(1) & raw_component_match).any()
        scope_qa = ("shared_component_scope" if shared_scope else
                    "source_object_scope_unknown" if missing_scope.any() else "single_component_scope")
        known_counts = object_counts.dropna()
        raw_count = int(known_counts.max()) if not known_counts.empty and not missing_scope.any() else np.nan
        def json_value(value):
            if pd.isna(value):
                return None
            return value.item() if isinstance(value, np.generic) else value

        evidence = []
        for index, candidate in best.iterrows():
            source_id = object_ids.loc[index]
            source_count = raw_scope_counts.get(source_id, np.nan) if pd.notna(source_id) and source_id else np.nan
            evidence.append({
                "source_object_id": json_value(source_id),
                "source_feature_id": json_value(candidate.get("Source_Feature_ID")),
                "raw_matched_component_count": int(source_count) if pd.notna(source_count) else None,
                "candidate_component_in_raw_intersection_set": bool(raw_component_match.loc[index]),
                "intersection_area_sqm": float(intersection.loc[index]) if pd.notna(intersection.loc[index]) else None,
                "osm_area_sqm": float(osm_area.loc[index]) if pd.notna(osm_area.loc[index]) else None,
                "component_overlap_ratio": float(component_overlap.loc[index]) if pd.notna(component_overlap.loc[index]) else None,
                "osm_overlap_ratio": float(overlap.loc[index]) if pd.notna(overlap.loc[index]) else None,
                "iou": float(candidate.dimension_iou) if pd.notna(candidate.dimension_iou) else None,
                "reported_height": json_value(candidate.get("osm_height")),
                "reported_storeys": json_value(candidate.get("osm_storeys")),
            })
        rows.append({ID: bid, "OSM_Height_m": heights[0] if len(heights) == 1 else np.nan,
                     "OSM_Storeys": floors[0] if len(floors) == 1 else np.nan,
                     "OSM_Dimension_Match_QA": "shared_object" if shared else "reported_attribute_conflict" if reported_conflict else "multiple_values" if conflict else "comparable_footprint",
                     "OSM_Dimension_Object_ID": ";".join(sorted(best.Source_Feature_ID.dropna().astype(str).unique())),
                     "OSM_Dimension_Scope_QA": scope_qa,
                     "OSM_Dimension_Raw_Matched_Component_Count": raw_count,
                     "OSM_Dimension_Candidate_Evidence": json.dumps(evidence, ensure_ascii=False, allow_nan=False)})
    return pd.DataFrame(rows)


def attach_csdi_dimensions(frame, source):
    """Match frozen CSDI components by ID, type and >=95% footprint IoU.

    Keep the original footprint and all raw source attributes. The distributor's
    clipped/rounded height_m is deliberately not used.
    """
    require_unique(frame, "BUILDINGCSUID")
    require_unique(source, "BuildingCSUID")
    if source.crs is None:
        raise ValueError("CSDI snapshot must declare its geometry CRS")
    source = source.to_crs(frame.crs).set_index("BuildingCSUID").reindex(frame.BUILDINGCSUID)
    native = source.geometry.to_numpy()
    local = frame.geometry.to_numpy()
    valid = source.geometry.notna().to_numpy() & source.geometry.is_valid.to_numpy()
    intersection = np.zeros(len(frame))
    intersection[valid] = shapely.area(shapely.intersection(local[valid], native[valid]))
    iou = intersection / (frame.geometry.area.to_numpy() + source.geometry.area.to_numpy() - intersection)
    # The old footprint table uses T for non-podium structures, including open-sided/temporary structures.
    kind = source.BuildingBlockType.map({"Tower":"T", "Podium":"P", "Open-sided Structure":"T", "Temporary Structure":"T"})
    same_type = frame.BUILDINGSTRUCTURETYPE.astype(str).eq(kind.to_numpy()).to_numpy()
    matched = source.BuildingBlockType.notna().to_numpy()
    frame = frame.copy()
    columns = {"BaseHeight":"CSDI_Base_Elevation_mPD", "TopHeight":"CSDI_Top_Elevation_mPD",
        "Storeys":"CSDI_Reported_Storeys", "StoreysInBasement":"CSDI_Basement_Storeys",
        "BuildingBlockType":"CSDI_Structure_Type", "Status":"CSDI_Status"}
    for original, name in columns.items():
        frame[name] = source[original].to_numpy()
    for original, name in {"BuildingID":"CSDI_Native_Building_ID", "DateCreate":"CSDI_Record_Created_Date",
                           "DateStamp":"CSDI_Record_Updated_Date", "BuildingNameEN":"CSDI_Building_Name_EN",
                           "BuildingNameTC":"CSDI_Building_Name_TC"}.items():
        frame[name] = source[original].to_numpy() if original in source else np.nan
    frame["CSDI_Geometry_IoU"] = iou
    frame["CSDI_Match_Quality"] = np.select(
        [~matched, ~valid, ~same_type, ~(iou >= .95)],
        ["id_unmatched", "invalid_source_geometry", "structure_type_mismatch", "footprint_changed"],
        default="same_component_id_type_footprint")
    component_names=frame.CSDI_Building_Name_EN.fillna('').astype(str)+' '+frame.CSDI_Building_Name_TC.fillna('').astype(str)
    substation=component_names.str.contains(r'(?i)\bsub[\s-]?station\b|\btransformer\s+station\b|[變变][電电壓压]站',regex=True)
    source_usable=frame.CSDI_Match_Quality.eq('same_component_id_type_footprint')&frame.CSDI_Status.eq('Active')
    frame['CSDI_Component_Use_Type']=np.where(source_usable&substation,'substation','')
    return frame


def fuse_dimensions(frame, aggregated, terrain=None, csdi=None):
    frame = frame.copy()
    frame[ID] = ids(frame[ID])
    require_unique(frame)
    matches = dimension_matches(aggregated)
    frame = frame.merge(matches, on=ID, how="left", validate="one_to_one")
    if csdi is not None:
        frame = attach_csdi_dimensions(frame, csdi)
    frame["Terrain_Elevation_mPD"] = np.nan
    if terrain is not None:
        terrain = terrain.copy()
        terrain[ID] = ids(terrain[ID])
        require_unique(terrain)
        terrain = terrain.set_index(ID).reindex(frame[ID])
        points = frame.to_crs(4326).geometry.representative_point()
        near = np.hypot(terrain.longitude.to_numpy() - points.x.to_numpy(), terrain.latitude.to_numpy() - points.y.to_numpy()) <= 1e-6
        frame["Terrain_Elevation_mPD"] = np.where(near, terrain.dtm_elevation_mpd.to_numpy(), np.nan)
    top, base = numbers(frame, "TOPHEIGHT"), numbers(frame, "BASEHEIGHT")
    native_height = (top - base).where((top - base).between(0, 600, inclusive="right"))
    csdi_ok = frame.get("CSDI_Match_Quality", pd.Series("not_supplied", index=frame.index)).eq("same_component_id_type_footprint")
    csdi_top = numbers(frame, "CSDI_Top_Elevation_mPD").where(csdi_ok)
    csdi_base = numbers(frame, "CSDI_Base_Elevation_mPD").where(csdi_ok)
    csdi_height = (csdi_top - csdi_base).where((csdi_top - csdi_base).between(0, 600, inclusive="right"))
    h = csdi_height.combine_first(native_height)
    conflict = csdi_height.notna() & native_height.notna() & ((csdi_height-native_height).abs() > np.maximum(10, np.maximum(csdi_height,native_height)*.2))
    h_source = pd.Series(np.where(csdi_height.notna(), "CSDI_top_minus_component_base",
        np.where(native_height.notna(), "LandsD_top_minus_base", "missing_component_base_or_top")), index=frame.index)
    h = h.mask(conflict)
    h_source.loc[conflict] = "conflicting_component_height_sources"
    frame["Height_Reference"] = np.where(h.notna(), "structure_base_to_top", "unknown")
    # Ground-to-top and a component resting on a podium have different reference planes.
    selected_top = csdi_top.combine_first(top)
    terrain_delta = selected_top - frame.Terrain_Elevation_mPD
    terrain_ok = terrain_delta.between(2, 600)
    terrain_height = terrain_delta.where(terrain_ok)
    osm_scope = frame.get("OSM_Dimension_Scope_QA", pd.Series("source_object_scope_unknown", index=frame.index)).fillna("source_object_scope_unknown")
    osm_scope_single = osm_scope.eq("single_component_scope")
    osm_h_reported = numbers(frame, "OSM_Height_m")
    osm_h = osm_h_reported.where(osm_scope_single)
    ground_conflict = osm_h.notna() & terrain_height.notna() & ((osm_h-terrain_height).abs() > np.maximum(10, np.maximum(osm_h,terrain_height)*.2))
    frame["Ground_To_Top_Height_m"] = terrain_height.combine_first(osm_h).mask(ground_conflict)
    frame["Ground_Height_Source"] = np.select([ground_conflict, terrain_height.notna() & csdi_top.notna(), terrain_height.notna(), osm_h.notna(), osm_h_reported.notna()],
        ["conflicting_ground_height_sources", "CSDI_top_minus_point_DTM", "LandsD_top_minus_point_DTM", "OSM_height_reference_unverified", "OSM_shared_or_unknown_scope_candidate_not_used"], default="missing")
    frame["Ground_Height_Quality"] = np.select([ground_conflict, terrain_height.notna(), osm_h.notna(), osm_h_reported.notna()],
        ["conflict_unresolved", "terrain_point_estimate", "reported_reference_unverified", "candidate_scope_unresolved_not_used"], default="unknown")
    reference_difference = h.notna() & osm_h.notna() & ((h-osm_h).abs() > np.maximum(10, np.maximum(h,osm_h)*.2))
    frame["Height_Source_Comparison_QA"] = np.where(reference_difference, "OSM_differs_component_reference_may_differ", "")
    frame["Component_Base_Above_Terrain_m"] = csdi_base.combine_first(base) - frame.Terrain_Elevation_mPD
    raw_f = numbers(frame, "NUMABOVEGROUNDSTOREYS").map(parse_storeys)
    osm_reported_f = numbers(frame, "OSM_Storeys").map(parse_storeys)
    # OSM whole-building levels may include a podium. Only a compatible ground
    # base supports donating them to this component; retain other reports separately.
    osm_scope_supported = osm_scope_single & frame.Component_Base_Above_Terrain_m.abs().le(3)
    osm_f = osm_reported_f.where(osm_scope_supported)
    frame["OSM_Storeys_Scope_QA"] = np.select(
        [osm_reported_f.isna(), osm_scope.eq("shared_component_scope"), ~osm_scope_single, osm_scope_supported],
        ["no_reported_storeys", "shared_component_scope_not_used_for_component",
         "source_object_scope_unknown_not_used_for_component", "component_ground_base_compatible_by_point_DTM"],
        default="whole_building_scope_unverified_not_used_for_component")
    csdi_f = numbers(frame, "CSDI_Reported_Storeys").map(parse_storeys).where(csdi_ok)
    floor_values = pd.concat([raw_f, osm_f, csdi_f], axis=1)
    lowest, highest = floor_values.min(axis=1), floor_values.max(axis=1)
    floor_conflict = (highest-lowest) > np.maximum(1, highest*.1)
    reported_f = csdi_f.combine_first(raw_f).combine_first(osm_f)
    # The retrieved CSDI schema does not define Storeys as aboveground-only.
    # Retain its report for comparison; require an explicit aboveground source.
    f = raw_f.combine_first(osm_f)
    unconfirmed_count_scope = csdi_f.notna() & f.isna()
    frame["CSDI_Storeys_Scope_QA"] = np.select([csdi_f.isna(), csdi_f.eq(f)],
        ["no_eligible_reported_count", "same_count_as_explicit_aboveground_source"],
        default="reported_count_vertical_scope_unverified")
    floor_source = pd.Series(np.select([raw_f.notna(),osm_f.notna()],
        ["LandsD_storeys","OSM_building_levels"],default="missing"), index=frame.index)
    floor_source.loc[unconfirmed_count_scope] = "CSDI_storeys_vertical_scope_unverified"
    # These ratios screen source compatibility, not legal floor heights or
    # measured errors. A short multi-storey pair without reconciled vertical
    # evidence is retained raw, but is unsuitable for adopted dimensions/GFA.
    ratio_count = reported_f.combine_first(f)
    ratio = h / ratio_count
    ratio_review = h.notna() & ratio_count.notna() & ((ratio < 2) | (ratio > 8))
    frame["Dimension_Ratio_Scope_QA"] = np.select(
        [ratio_review & ratio.lt(2), ratio_review & ratio.gt(8)],
        ["low_height_to_count_scope_review", "high_height_to_count_scope_review"], default="")
    explicit_scope_unresolved = frame.get(
        "Component_Vertical_Scope_QA", pd.Series("", index=frame.index)
    ).fillna("").astype(str).str.lower().isin(
        ["unresolved", "scope_unresolved", "component_vertical_scope_unresolved"])
    scope_ambiguous = h.notna() & ((reported_f.notna() & ~((h/reported_f).between(1,20))) |
                                   (f.notna() & ~((h/f).between(1,20))))
    unreconciled_low_pair = h.notna() & f.ge(2) & ~floor_conflict & ~explicit_scope_unresolved & (h/f).lt(2-1e-9)
    frame.loc[unreconciled_low_pair, "Dimension_Ratio_Scope_QA"] = "low_height_to_count_scope_review"
    scope_ambiguous |= unreconciled_low_pair
    frame["Reported_Component_Height_m"] = h
    low_height_scope = h.notna() & (((h/reported_f) < 1-1e-9) | ((h/f) < 1-1e-9) | unreconciled_low_pair)
    h = h.mask(low_height_scope)
    h_source.loc[low_height_scope] = "component_height_storeys_scope_unresolved"
    frame.loc[low_height_scope, "Height_Reference"] = "unresolved_component_scope"
    frame.loc[low_height_scope, "Height_Source_Comparison_QA"] = frame.loc[
        low_height_scope, "Height_Source_Comparison_QA"].map(
            lambda value: ";".join(filter(None, [value, "height_storeys_component_scope_unresolved"])))
    frame.loc[low_height_scope & frame.Ground_To_Top_Height_m.notna(), "Ground_Height_Quality"] = "component_scope_unresolved_estimate"
    floor_conflict |= scope_ambiguous | explicit_scope_unresolved
    f = f.mask(floor_conflict)
    floor_source.loc[floor_conflict] = "conflicting_reported_storeys"
    floor_source.loc[scope_ambiguous | explicit_scope_unresolved] = "component_storeys_scope_unresolved"
    frame["Estimated_Height"], frame["Estimated_Floors"] = h, f
    frame["Height_Source"], frame["Floors_Source"] = h_source, floor_source
    frame["Height_Quality"] = np.select([conflict | low_height_scope, h.isna()],
        ["conflict_unresolved", "unknown"], default="reported_component_not_independently_validated")
    frame["Floors_Quality"] = np.select([floor_conflict,unconfirmed_count_scope,f.isna()],
        ["conflict_unresolved","reported_vertical_scope_unverified","unknown"], default="reported_not_independently_validated")
    frame["Height_Terrain_QA"] = np.where(selected_top.notna() & frame.Terrain_Elevation_mPD.notna() & ~terrain_ok,
                                          "nonpositive_small_or_excessive_top_minus_terrain", "")
    # Normalize source subtraction noise; nearest whole storey, with half ties upward.
    frame["Floors_Height_Proxy"] = np.maximum(1, np.floor((h / 3).round(9) + .5)).where(f.isna() & ~floor_conflict & ~unconfirmed_count_scope & ~ratio_review & h.notna())
    frame["Floor_Height_Proxy_Assumption_m"] = np.where(frame.Floors_Height_Proxy.notna(), 3., np.nan)
    return assign_floor_area(frame)


def assign_floor_area(frame):
    frame = frame.copy()
    raw_registered = numbers(frame, "GROSSFLOORAREA").where(lambda s: s > 0)
    prior_source = frame.get('Registered_GFA_Source', pd.Series('', index=frame.index)).fillna('').astype(str)
    prior_quality = frame.get('GFA_Quality', pd.Series('', index=frame.index)).fillna('').astype(str)
    unresolved_registered_source = 'LandsD_structure_GFA_component_scope_unresolved'
    previously_unresolved = (prior_source.str.contains(unresolved_registered_source, regex=False) |
                             prior_quality.eq('registered_component_scope_unresolved'))
    floor_source = frame.get('Floors_Source', pd.Series('', index=frame.index))
    unresolved_floor_scope = floor_source.isin(
        ['component_storeys_scope_unresolved', 'documented_storey_reference_conflict_withheld']) | (
            floor_source.eq('conflicting_reported_storeys') & numbers(frame, 'Estimated_Floors').isna())
    component_scope_unresolved = raw_registered.notna() & (
        floor_source.isin(['component_storeys_scope_unresolved', 'CSDI_storeys_vertical_scope_unverified']) |
        previously_unresolved)
    registered = raw_registered.where(~component_scope_unresolved)
    dom, nondom = numbers(frame, "GFA_DOMESTIC_SUM"), numbers(frame, "GFA_NONDOMESTIC_SUM")
    permit_total = (dom + nondom).where((dom >= 0) & (nondom >= 0) & ((dom + nondom) > 0))
    unresolved = pd.Series(False, index=frame.index)
    if 'Permit_GFA_Scope' in frame:
        single = frame.Permit_GFA_Scope.isin(['single_structure', 'single_structure_physical_scope_unverified',
                                            'single_structure_component_scope_unresolved',
                                            'single_structure_component_use_scope_unresolved'])
        verified = frame.get('Permit_GFA_Physical_Scope_Verified', pd.Series(False, index=frame.index)).astype('string').str.lower().eq('true').fillna(False)
        vertical_unresolved = (frame.Permit_GFA_Scope.eq('single_structure_component_scope_unresolved') & ~verified) | (single &
            (unresolved_floor_scope |
             component_scope_unresolved))
        use_unresolved = frame.Permit_GFA_Scope.eq('single_structure_component_use_scope_unresolved') | (single & dom.gt(0) &
            frame.get('CSDI_Component_Use_Type',pd.Series('',index=frame.index)).eq('substation'))
        physical_scope_unverified = single & ~verified
        unresolved = vertical_unresolved | use_unresolved | physical_scope_unverified
        frame.loc[single, 'Permit_GFA_Scope'] = np.select([vertical_unresolved.loc[single],use_unresolved.loc[single]],
            ['single_structure_component_scope_unresolved','single_structure_component_use_scope_unresolved'],
            default=np.where(verified.loc[single], 'single_structure',
                             'single_structure_component_scope_unresolved'))
        frame['Permit_GFA_Usable'] = single & verified & ~unresolved & permit_total.notna()
    if 'Permit_GFA_Usable' in frame:
        permit_total = permit_total.where(frame.Permit_GFA_Usable.fillna(False))
    area = registered.combine_first(permit_total)
    frame["Registered_GFA_sqm"] = area
    frame["Registered_GFA_Source"] = np.select(
        [registered.notna(), permit_total.notna(), component_scope_unresolved],
        ["LandsD_structure_GFA", "single_structure_single_permit", unresolved_registered_source],
        default="missing")
    documented = numbers(frame, 'Documented_GFA_sqm').where(lambda s: s > 0)
    dimension_scope_unresolved = unresolved_floor_scope
    frame["GFA_Proxy_sqm"] = (numbers(frame, "SHAPE_Area") * numbers(frame, "Estimated_Floors")).where(~dimension_scope_unresolved)
    frame["GFA_Proxy_Method"] = np.where(frame.GFA_Proxy_sqm.notna(), "footprint_times_reported_storeys_uniform_floorplate", "unknown")
    frame["GFA_Best_Available_sqm"] = area.combine_first(documented).combine_first(frame.GFA_Proxy_sqm)
    permit_selected = permit_total.notna() & registered.isna()
    frame["GFA_Quality"] = np.select(
        [registered.notna(), permit_selected, documented.notna(), component_scope_unresolved,
         frame.GFA_Proxy_sqm.notna()],
        ["reported_structure_gfa_not_independently_validated", "registered_scope_checked",
         "documented_component_gfa", "registered_component_scope_unresolved",
         "uniform_floorplate_proxy"],
        default=np.where(unresolved, np.where(frame.get('Permit_GFA_Scope',pd.Series('',index=frame.index)).eq(
            'single_structure_component_use_scope_unresolved'),'permit_component_use_scope_unresolved','permit_component_scope_unresolved'), 'unknown'))
    return frame


def apply_documented_attributes(frame, evidence):
    """Keep source-scoped facts; only explicit same-component counts donate floors."""
    frame, evidence = frame.copy(), evidence.copy()
    require_unique(evidence, 'evidence_id')
    evidence['building_id'] = ids(evidence['building_id'])
    lookup = frame.set_index(ID)['BUILDINGCSUID']
    if not evidence['building_id'].isin(lookup.index).all():
        raise ValueError('Documented attribute has an unknown local building ID')
    if not evidence.csuid.eq(evidence.building_id.map(lookup)).all():
        raise ValueError('Documented attribute local ID/BuildingCSUID mismatch')
    if not evidence.source_url.str.startswith(('https://', 'http://')).all():
        raise ValueError('Documented attributes require source URLs')
    frame['Reported_Attribute_Evidence'] = '[]'
    frame['Floors_Document_Evidence_IDs'] = '[]'
    frame['Floors_Before_Document_Review'] = numbers(frame, 'Estimated_Floors')
    frame['Floors_Source_Before_Document_Review'] = frame.Floors_Source
    frame['Floors_Source_Comparison_QA'] = ''
    frame['Height_Before_Document_Review'] = numbers(frame, 'Estimated_Height').combine_first(
        numbers(frame, 'Reported_Component_Height_m'))
    frame['Documented_GFA_sqm'] = np.nan
    frame['Documented_GFA_Evidence_IDs'] = '[]'
    by_id = {bid: index for index, bid in frame[ID].items()}
    for bid, group in evidence.groupby('building_id', sort=True):
        index = by_id[bid]
        frame.at[index, 'Reported_Attribute_Evidence'] = json.dumps(
            group.fillna('').to_dict('records'), ensure_ascii=False, separators=(',', ':'))
        references = group[group.adoption.eq('unresolved_height_reference_conflict')]
        if not references.empty:
            if not (references.attribute.eq('height') & references.unit.eq('m')).all():
                raise ValueError('Height reference review requires explicit metre-valued height facts')
            previous = frame.at[index,'Height_Source_Comparison_QA'] if 'Height_Source_Comparison_QA' in frame else ''
            flags = str(previous).split(';') if pd.notna(previous) and previous else []
            frame.at[index,'Height_Source_Comparison_QA'] = ';'.join(sorted(set(flags+['documented_height_reference_conflict'])))
            frame.at[index,'Estimated_Height'] = np.nan
            frame.at[index,'Height_Source'] = 'documented_height_reference_conflict_withheld'
            frame.at[index,'Height_Quality'] = 'conflict_unresolved'
            frame.at[index,'Height_Reference'] = 'unresolved_documented_reference'
            frame.at[index,'Floors_Height_Proxy'] = np.nan
            frame.at[index,'Floor_Height_Proxy_Assumption_m'] = np.nan
            if 'Ground_Height_Quality' in frame:
                frame.at[index,'Ground_Height_Quality'] = 'documented_reference_unresolved_estimate'
        areas = group[group.adoption.eq('component_gfa_supported')]
        if not areas.empty:
            if not (areas.attribute.eq('gross_floor_area') & areas.unit.eq('m2') &
                    areas.scope.eq('tower_component_gfa')).all():
                raise ValueError('Documented GFA requires same-component gross-floor-area scope')
            values = pd.to_numeric(areas.value, errors='coerce')
            if not (np.isfinite(values) & values.gt(0)).all() or len(values.unique()) != 1:
                raise ValueError('Invalid/conflicting documented component GFA')
            row = frame.loc[index]
            if row.get('CSDI_Match_Quality') != 'same_component_id_type_footprint' or row.get('CSDI_Structure_Type') != 'Tower':
                raise ValueError('Documented GFA lacks a matched tower component')
            frame.at[index, 'Documented_GFA_sqm'] = values.iloc[0]
            frame.at[index, 'Documented_GFA_Evidence_IDs'] = json.dumps(areas.evidence_id.tolist())
        counts = group[group.adoption.eq('component_storeys_supported')]
        count_conflicts = group[group.adoption.eq('unresolved_component_storey_reference_conflict')]
        if not count_conflicts.empty:
            if not counts.empty:
                raise ValueError('Conflicting documented storey decisions require source reconciliation')
            values = pd.to_numeric(count_conflicts.value, errors='coerce').map(parse_storeys)
            row = frame.loc[index]
            if not (count_conflicts.attribute.isin(['aboveground_storeys', 'reported_storeys']) &
                    count_conflicts.unit.eq('storeys')).all() or values.isna().any():
                raise ValueError('Storey reference conflict requires valid reported storey counts')
            if row.get('CSDI_Match_Quality') != 'same_component_id_type_footprint' or row.get('CSDI_Structure_Type') not in {'Tower', 'Podium', 'Temporary Structure'}:
                raise ValueError('Documented storey conflict lacks a matched tower, podium or temporary component')
            frame.at[index,'Estimated_Floors'] = np.nan
            frame.at[index,'Floors_Source'] = 'documented_storey_reference_conflict_withheld'
            frame.at[index,'Floors_Quality'] = 'conflict_unresolved'
            frame.at[index,'Floors_Source_Comparison_QA'] = 'documented_storey_reference_conflict'
            frame.at[index,'Floors_Height_Proxy'] = np.nan
            frame.at[index,'Floor_Height_Proxy_Assumption_m'] = np.nan
        if counts.empty:
            continue
        if not (counts.attribute.eq('aboveground_storeys') & counts.unit.eq('storeys') &
                counts.scope.eq('tower_component_aboveground')).all():
            raise ValueError('Component storeys require explicit aboveground physical-count scope')
        values = pd.to_numeric(counts.value, errors='coerce').map(parse_storeys)
        if values.isna().any():
            raise ValueError('Invalid documented component storey count')
        row = frame.loc[index]
        if row.get('CSDI_Match_Quality') != 'same_component_id_type_footprint' or row.get('CSDI_Structure_Type') != 'Tower':
            raise ValueError('Documented tower count lacks a matched tower component')
        if len(values.unique()) != 1:
            raise ValueError('Conflicting documented component counts require source reconciliation')
        value = values.iloc[0]
        height = row.Estimated_Height
        if pd.notna(height) and not 1 <= height / value <= 20:
            raise ValueError('Documented storeys contradict the component height')
        frame.at[index, 'Estimated_Floors'] = value
        frame.at[index, 'Floors_Source'] = 'Documented_component_storeys'
        frame.at[index, 'Floors_Quality'] = 'documented_component_count_legacy_conflict_resolved'
        frame.at[index, 'Floors_Document_Evidence_IDs'] = json.dumps(counts.evidence_id.tolist())
        frame.at[index, 'Floors_Height_Proxy'] = np.nan
        frame.at[index, 'Floor_Height_Proxy_Assumption_m'] = np.nan
    return assign_floor_area(frame)
