"""Identify non-building structures without treating every tower as equipment."""
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from classification import NONASSESSED, sub_label, finalize_function_fields
from data_quality import ID, require_unique


def infrastructure_kind(row):
    def value(key):
        item = row.get(key)
        return '' if pd.isna(item) else str(item).strip().lower()
    building, made, tower = value('building'), value('man_made'), value('tower:type')
    support = value('bridge:support')
    if building in {'bridge','bridge_support'} or support not in {'','no','false','0'} or tower == 'bridge':
        return 'bridge_structure'
    if made in {'mast','chimney','crane'} or tower == 'communication':
        return 'equipment_structure'
    # Pagodas, observation/clock towers and pumping-station buildings can be occupied.
    if made in {'water_tower','storage_tank'} and not any(word in value('name') for word in ['pump','station','plant','抽水','泵','機房']):
        return 'water_storage_structure'
    return None


def apply_structural_use(frame, osm, verified):
    frame = frame.copy()
    frame['Structure_Use_Type'] = ''
    frame['Structure_Use_Name'] = ''
    frame['Structure_Use_Source'] = ''
    frame['Structure_Use_Evidence'] = ''
    frame['Structure_Use_Quality'] = 'not_independently_reviewed'
    osm = osm.to_crs(frame.crs).copy()
    osm['_kind'] = [infrastructure_kind(row) for _,row in osm.iterrows()]
    osm = osm[osm._kind.notna() & osm.geometry.is_valid].reset_index(drop=True)
    left,right = osm.sindex.query(frame.geometry,predicate='intersects')
    area = shapely.area(shapely.intersection(frame.geometry.to_numpy()[left],osm.geometry.to_numpy()[right]))
    iou = area/(frame.geometry.area.to_numpy()[left]+osm.geometry.area.to_numpy()[right]-area)
    pairs = pd.DataFrame({'local':left,'donor':right,'iou':iou})
    pairs = pairs[pairs.iou.ge(.8)]
    shared = pairs.groupby('donor').local.nunique()
    pairs = pairs[~pairs.donor.isin(shared[shared.gt(1)].index)]
    for position,group in pairs.groupby('local'):
        donors = osm.iloc[group.donor]
        if donors._kind.nunique() != 1:
            raise ValueError('Competing structural-use polygon donors; resolve source evidence before publishing')
        index = frame.index[position]
        frame.loc[index,['Structure_Use_Type','Structure_Use_Source','Structure_Use_Evidence','Structure_Use_Quality']] = [
            donors._kind.iloc[0],'OSM_explicit_structure_polygon',
            ';'.join('https://www.openstreetmap.org/'+str(row.osm_type)+'/'+str(int(row.id)) for _,row in donors.iterrows()),
            'source_tag_and_comparable_unique_footprint']
        frame.loc[index,'Structure_Use_Name'] = '; '.join(sorted(donors['name'].dropna().astype(str).unique()))
    require_unique(verified,'BUILDINGCSUID')
    for _,record in verified.iterrows():
        mask = frame.BUILDINGCSUID.eq(record.BUILDINGCSUID)
        if not mask.any():
            raise ValueError('Verified structure ID no longer present: '+record.BUILDINGCSUID)
        point = gpd.GeoSeries(gpd.points_from_xy([record.longitude],[record.latitude]),crs=4326).to_crs(frame.crs).iloc[0]
        if frame.loc[mask].geometry.centroid.distance(point).gt(5).any():
            raise ValueError('Verified structure geometry moved; re-identification required')
        frame.loc[mask,['Structure_Use_Type','Structure_Use_Name','Structure_Use_Source','Structure_Use_Evidence','Structure_Use_Quality']] = [
            record.structure_use,record.landmark_name,'official_document_and_spatial_identification',record.evidence_url,
            'documented_landmark_spatially_identified_not_height_survey']
    excluded = frame.Structure_Use_Type.ne('')
    frame['Energy_Assessment_Scope'] = np.where(excluded,'excluded_nonbuilding_structure','building_or_structural_use_unverified')
    frame['Pre_Structure_Review_Main_Class'] = frame.Final_Main_Class
    frame['Pre_Structure_Review_Sub_Class'] = frame.Final_Sub_Class
    frame.loc[excluded,'Final_Main_Class'] = NONASSESSED
    frame.loc[excluded,'Final_Sub_Class'] = frame.loc[excluded,'Structure_Use_Type'].map(
        lambda kind:sub_label('交通基础设施' if kind=='bridge_structure' else '临时/杂项设施'))
    frame.loc[excluded,'Classification_Source'] = frame.loc[excluded,'Structure_Use_Source']
    frame.loc[excluded,'Estimated_Floors'] = np.nan
    frame.loc[excluded,'Floors_Source'] = 'not_applicable_nonbuilding_structure'
    frame.loc[excluded,'Floors_Quality'] = 'not_applicable_nonbuilding_structure'
    frame.loc[excluded,['GFA_Proxy_sqm','GFA_Best_Available_sqm','Floors_Height_Proxy','Floor_Height_Proxy_Assumption_m']] = np.nan
    frame.loc[excluded,'GFA_Proxy_Method'] = 'not_applicable_nonbuilding_structure'
    frame.loc[excluded,'GFA_Quality'] = 'not_applicable_nonbuilding_structure'
    return finalize_function_fields(frame)
