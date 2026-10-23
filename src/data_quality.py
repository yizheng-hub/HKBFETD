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
        valid = single and pd.notna(domestic) and pd.notna(nondomestic) and domestic >= 0 and nondomestic >= 0 and domestic + nondomestic > 0
        rows.append({ID: bid, "Permit_IDs": json.dumps(sorted(group.OPNO.astype(str).unique().tolist())),
                     "Permit_GFA_Scope": "single_structure" if single else "shared_or_multiple_permits",
                     "Permit_GFA_Usable": valid,
                     "GFA_DOMESTIC_SUM": domestic if valid else np.nan,
                     "GFA_NONDOMESTIC_SUM": nondomestic if valid else np.nan,
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
    """Only comparable polygon footprints can donate height/storey attributes."""
    attrs = aggregated.copy()
    overlap = numbers(attrs, "overlap_ratio")
    osm_area, area = numbers(attrs, "osm_area"), numbers(attrs, "SHAPE_Area")
    intersection = overlap * osm_area
    attrs["dimension_iou"] = intersection / (osm_area + area - intersection)
    candidates = attrs[(osm_area > 1) & (overlap >= .6) & ((intersection / area) >= .6) & (attrs.dimension_iou >= .5)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=[ID, "OSM_Height_m", "OSM_Storeys", "OSM_Dimension_Match_QA", "OSM_Dimension_Object_ID"])
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
        rows.append({ID: bid, "OSM_Height_m": heights[0] if len(heights) == 1 and not shared else np.nan,
                     "OSM_Storeys": floors[0] if len(floors) == 1 and not shared else np.nan,
                     "OSM_Dimension_Match_QA": "shared_object" if shared else "reported_attribute_conflict" if reported_conflict else "multiple_values" if conflict else "comparable_footprint",
                     "OSM_Dimension_Object_ID": ";".join(sorted(best.Source_Feature_ID.unique()))})
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
                           "DateStamp":"CSDI_Record_Updated_Date"}.items():
        frame[name] = source[original].to_numpy() if original in source else np.nan
    frame["CSDI_Geometry_IoU"] = iou
    frame["CSDI_Match_Quality"] = np.select(
        [~matched, ~valid, ~same_type, ~(iou >= .95)],
        ["id_unmatched", "invalid_source_geometry", "structure_type_mismatch", "footprint_changed"],
        default="same_component_id_type_footprint")
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
    osm_h = numbers(frame, "OSM_Height_m")
    ground_conflict = osm_h.notna() & terrain_height.notna() & ((osm_h-terrain_height).abs() > np.maximum(10, np.maximum(osm_h,terrain_height)*.2))
    frame["Ground_To_Top_Height_m"] = terrain_height.combine_first(osm_h).mask(ground_conflict)
    frame["Ground_Height_Source"] = np.select([ground_conflict, terrain_height.notna() & csdi_top.notna(), terrain_height.notna(), osm_h.notna()],
        ["conflicting_ground_height_sources", "CSDI_top_minus_point_DTM", "LandsD_top_minus_point_DTM", "OSM_height_reference_unverified"], default="missing")
    frame["Ground_Height_Quality"] = np.select([ground_conflict, terrain_height.notna(), osm_h.notna()],
        ["conflict_unresolved", "terrain_point_estimate", "reported_reference_unverified"], default="unknown")
    reference_difference = h.notna() & osm_h.notna() & ((h-osm_h).abs() > np.maximum(10, np.maximum(h,osm_h)*.2))
    frame["Height_Source_Comparison_QA"] = np.where(reference_difference, "OSM_differs_component_reference_may_differ", "")
    frame["Component_Base_Above_Terrain_m"] = csdi_base.combine_first(base) - frame.Terrain_Elevation_mPD
    raw_f = numbers(frame, "NUMABOVEGROUNDSTOREYS").map(parse_storeys)
    osm_reported_f = numbers(frame, "OSM_Storeys").map(parse_storeys)
    # OSM whole-building levels may include a podium. Only a compatible ground
    # base supports donating them to this component; retain other reports separately.
    osm_scope_supported = frame.Component_Base_Above_Terrain_m.abs().le(3)
    osm_f = osm_reported_f.where(osm_scope_supported)
    frame["OSM_Storeys_Scope_QA"] = np.select([osm_reported_f.isna(),osm_scope_supported],
        ["no_reported_storeys","component_ground_base_compatible_by_point_DTM"],
        default="whole_building_scope_unverified_not_used_for_component")
    csdi_f = numbers(frame, "CSDI_Reported_Storeys").map(parse_storeys).where(csdi_ok)
    floor_values = pd.concat([raw_f, osm_f, csdi_f], axis=1)
    lowest, highest = floor_values.min(axis=1), floor_values.max(axis=1)
    floor_conflict = (highest-lowest) > np.maximum(1, highest*.1)
    f = csdi_f.combine_first(raw_f).combine_first(osm_f)
    floor_source = pd.Series(np.select([csdi_f.notna(),raw_f.notna(),osm_f.notna()],
        ["CSDI_reported_storeys","LandsD_storeys","OSM_building_levels"],default="missing"), index=frame.index)
    # Extreme ratios indicate incompatible component/storey scope. Never infer a replacement count.
    scope_ambiguous = h.notna() & f.notna() & ~((h/f).between(1,20))
    floor_conflict |= scope_ambiguous
    f = f.mask(floor_conflict)
    floor_source.loc[floor_conflict] = "conflicting_reported_storeys"
    floor_source.loc[scope_ambiguous] = "component_storeys_scope_unresolved"
    frame["Estimated_Height"], frame["Estimated_Floors"] = h, f
    frame["Height_Source"], frame["Floors_Source"] = h_source, floor_source
    frame["Height_Quality"] = np.select([conflict, h.isna()],
        ["conflict_unresolved", "unknown"], default="reported_component_not_independently_validated")
    frame["Floors_Quality"] = np.select([floor_conflict, f.isna()], ["conflict_unresolved", "unknown"], default="reported_not_independently_validated")
    frame["Height_Terrain_QA"] = np.where(selected_top.notna() & frame.Terrain_Elevation_mPD.notna() & ~terrain_ok,
                                          "nonpositive_small_or_excessive_top_minus_terrain", "")
    # Normalize source subtraction noise; nearest whole storey, with half ties upward.
    frame["Floors_Height_Proxy"] = np.maximum(1, np.floor((h / 3).round(9) + .5)).where(f.isna() & ~floor_conflict & h.notna())
    frame["Floor_Height_Proxy_Assumption_m"] = np.where(frame.Floors_Height_Proxy.notna(), 3., np.nan)
    return assign_floor_area(frame)


def assign_floor_area(frame):
    frame = frame.copy()
    registered = numbers(frame, "GROSSFLOORAREA").where(lambda s: s > 0)
    dom, nondom = numbers(frame, "GFA_DOMESTIC_SUM"), numbers(frame, "GFA_NONDOMESTIC_SUM")
    permit_total = (dom + nondom).where((dom >= 0) & (nondom >= 0) & ((dom + nondom) > 0))
    area = registered.combine_first(permit_total)
    frame["Registered_GFA_sqm"] = area
    frame["Registered_GFA_Source"] = np.where(registered.notna(), "LandsD_structure_GFA", np.where(permit_total.notna(), "single_structure_single_permit", "missing"))
    documented = numbers(frame, 'Documented_GFA_sqm').where(lambda s: s > 0)
    frame["GFA_Proxy_sqm"] = numbers(frame, "SHAPE_Area") * numbers(frame, "Estimated_Floors")
    frame["GFA_Proxy_Method"] = np.where(frame.GFA_Proxy_sqm.notna(), "footprint_times_reported_storeys_uniform_floorplate", "unknown")
    frame["GFA_Best_Available_sqm"] = area.combine_first(documented).combine_first(frame.GFA_Proxy_sqm)
    frame["GFA_Quality"] = np.select([area.notna(),documented.notna(),frame.GFA_Proxy_sqm.notna()],
        ["registered_scope_checked","documented_component_gfa","uniform_floorplate_proxy"],default="unknown")
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
    frame['Documented_GFA_sqm'] = np.nan
    frame['Documented_GFA_Evidence_IDs'] = '[]'
    by_id = {bid: index for index, bid in frame[ID].items()}
    for bid, group in evidence.groupby('building_id', sort=True):
        index = by_id[bid]
        frame.at[index, 'Reported_Attribute_Evidence'] = json.dumps(
            group.fillna('').to_dict('records'), ensure_ascii=False, separators=(',', ':'))
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
