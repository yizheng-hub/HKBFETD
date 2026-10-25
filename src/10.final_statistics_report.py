"""Step 10: versioned release, physical-value gates and complete v1.1 comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyogrio
import geopandas as gpd
from ai_cache import file_hash
from classification import parse_weights, MAIN_CLASSES, MIXED, NONASSESSED, UNKNOWN, taxonomy, sub_label
from config import (ML_FINAL_OUTPUT_PATH, PUBLIC_DATASET_CSV_PATH, PUBLIC_DATASET_GEOJSON_PATH,
    OUTPUT_DIR, LEGACY_OUTPUT_DIR, FILE_PATHS_LANDSD, INTERMEDIATE_DIR)
from data_quality import ID, ids, numbers, require_unique
from stage_io import read_table, attach_geometry
from utils import sanitize_dataframe_for_geojson
import pyogrio

EXPORT_MAPPING = {
    ID:'Building_ID_建筑ID','SHAPE_Area':'Footprint_Area_sqm_占地面积',
    'Estimated_Floors':'Floors_Count_地上层数','Estimated_Height':'Height_m_建筑高度',
    'Calibrated_Main_Class':'UBEM_Main_Class_主类别','Calibrated_Sub_Class':'UBEM_Sub_Class_子类别',
    'Calibrated_Mix_Proportion':'UBEM_Mixed_Proportions_混合比例',
    'Calibrated_Source':'Classification_Source_分类来源','Calibrated_Notes':'Classification_Notes_分类备注',
    'ML_Confidence':'ML_Prediction_Confidence_机器学习置信度',
    'proba_Residential_住宅类别':'ML_Prob_Residential_住宅概率',
    'proba_Commercial_商业类别':'ML_Prob_Commercial_商业概率','proba_Industrial_工业类别':'ML_Prob_Industrial_工业概率',
    'BUILDINGSTRUCTURETYPE':'LandsD_Structure_Type_地政署建筑类型','CATEGORY':'LandsD_Category_地政署分类',
    'STATUS':'LandsD_Status_地政署状态','GROSSFLOORAREA':'LandsD_Registered_GFA_地政署注册面积',
    'BUILDINGNAMEEN':'LandsD_Building_Name_EN_地政署英文名','BUILDINGNAMETC':'LandsD_Building_Name_TC_地政署中文名',
    'BDBIAR_OBJECTID':'BDBIAR_Record_ID_屋宇署记录ID','BDBIAR_CLASS':'BDBIAR_Classification_屋宇署分类',
    'BDBIAR_AGE':'BDBIAR_Building_Age_Year_屋宇署楼龄年份','distance_to_bdbiar':'Dist_to_BDBIAR_m_距屋宇署点位距离',
    'Matched_POIs':'Crowdsourced_POIs_Matched_众包POI匹配详情','osmid':'OSM_Record_ID_OSM记录ID',
    'building':'OSM_Building_Tag_OSM建筑标签','amenity':'OSM_Amenity_Tag_OSM设施标签','shop':'OSM_Shop_Tag_OSM商铺标签',
    'OZP_ZONE_LABEL':'OZP_Zone_Code_法定图则代码','OZP_DESC_ENG':'OZP_Zone_Desc_EN_法定图则英文描述'}
PROVENANCE = ['BUILDINGCSUID','NUMABOVEGROUNDSTOREYS','TOPHEIGHT','BASEHEIGHT','Source_Footprint_Area_sqm','Geometry_Quality',
    'Footprint_Area_QA','Duplicate_Footprint_Geometry','Height_Source','Height_Quality','Height_Terrain_QA',
    'Floors_Source','Floors_Quality','Floors_Height_Proxy','Floor_Height_Proxy_Assumption_m',
    'OSM_Height_m','OSM_Storeys','OSM_Storeys_Scope_QA','OSM_Dimension_Match_QA','OSM_Dimension_Object_ID',
    'OSM_Dimension_Scope_QA','OSM_Dimension_Raw_Matched_Component_Count','OSM_Dimension_Candidate_Evidence','Terrain_Elevation_mPD',
    'Height_Reference','Ground_To_Top_Height_m','Ground_Height_Source','Ground_Height_Quality',
    'Height_Source_Comparison_QA','Component_Base_Above_Terrain_m',
    'Dimension_Ratio_Scope_QA','Component_Vertical_Scope_QA',
    'CSDI_Base_Elevation_mPD','CSDI_Top_Elevation_mPD','CSDI_Reported_Storeys','CSDI_Basement_Storeys',
    'CSDI_Structure_Type','CSDI_Status','CSDI_Geometry_IoU','CSDI_Match_Quality',
    'CSDI_Native_Building_ID','CSDI_Record_Created_Date','CSDI_Record_Updated_Date',
    'CSDI_Building_Name_EN','CSDI_Building_Name_TC','CSDI_Storeys_Scope_QA',
    'CSDI_Component_Use_Type','CSDI_Name_Function_Evidence',
    'Reported_Attribute_Evidence','Floors_Document_Evidence_IDs',
    'Floors_Before_Document_Review','Floors_Source_Before_Document_Review',
    'Floors_Source_Comparison_QA',
    'Documented_GFA_sqm','Documented_GFA_Evidence_IDs',
    'Documented_Function_Evidence','Function_Before_Document_Review','ML_Subclass_Candidate',
    'Height_Before_Document_Review',
    'ML_Main_Class_Candidate',
    'OSM_Function_Candidate_Main','OSM_Function_Candidate_Sub','OSM_Function_Evidence',
    'Structure_Use_Type','Structure_Use_Name','Structure_Use_Source','Structure_Use_Evidence','Structure_Use_Quality',
    'Energy_Assessment_Scope','Pre_Structure_Review_Main_Class','Pre_Structure_Review_Sub_Class',
    'Registered_GFA_sqm','Registered_GFA_Source','GFA_Proxy_sqm','GFA_Proxy_Method','GFA_Best_Available_sqm','GFA_Quality',
    'Permit_IDs','Permit_GFA_Scope','Permit_GFA_Usable','GFA_DOMESTIC_SUM','GFA_NONDOMESTIC_SUM',
    'Permit_GFA_Link_QA','Permit_GFA_Physical_Scope_Verified','Permit_Raw_DOMESTICGFA','Permit_Raw_NONDOMESTICGFA',
    'Observed_Main_Area_Proportions','Observed_Residential_Area_Share','Observed_Nondomestic_Area_Share',
    'Proportion_Method','Proportion_Quality','Functional_Subclasses','Functional_Main_Classes',
    'Evidence_Subclass_Weights','Classification_QA','Function_Quality','ML_Decision_Status','ML_Subclass_Confidence','Subclass_Source',
    'BDBIAR_Match_Quality','BDBIAR_Source_Block_ID','BDBIAR_Source_Integrity_Status',
    'BDBIAR_Source_Integrity_QA','BDBIAR_Source_Integrity_Evidence','BDBIAR_Source_Attributes_Usable',
    'BDBIAR_AGE_CANDIDATE','BDBIAR_CLASS_CANDIDATE','Geometry_Review_Status','LLM_Review_Status']


def validate_release(frame):
    require_unique(frame)
    raw = pyogrio.read_dataframe(FILE_PATHS_LANDSD['BUILDING_STRUCTURE'],columns=[ID,'NUMABOVEGROUNDSTOREYS','TOPHEIGHT','BASEHEIGHT'],read_geometry=False)
    raw[ID] = ids(raw[ID])
    require_unique(raw)
    assert set(frame[ID]) == set(raw[ID]), 'Building IDs lost/added'
    source = raw.set_index(ID).reindex(frame[ID])
    for col in ['NUMABOVEGROUNDSTOREYS','TOPHEIGHT','BASEHEIGHT']:
        left,right = numbers(frame,col).to_numpy(),numbers(source,col).to_numpy()
        assert np.isclose(left,right,equal_nan=True).all(), 'Raw field changed: '+col
    assert frame.geometry.is_valid.all() and not frame.geometry.is_empty.any()
    assert np.isclose(numbers(frame,'SHAPE_Area'),frame.geometry.area,atol=1e-5,rtol=1e-8).all()
    height,floors = numbers(frame,'Estimated_Height'),numbers(frame,'Estimated_Floors')
    assert height.dropna().between(0,600,inclusive='right').all()
    assert floors.dropna().between(1,150).all() and np.isclose(floors.dropna(),floors.dropna().round()).all()
    assert height[frame.Height_Quality.isin(['unknown','conflict_unresolved'])].isna().all()
    low_height_scope = frame.Height_Source_Comparison_QA.fillna('').str.contains('height_storeys_component_scope_unresolved',regex=False)
    assert height[low_height_scope].isna().all()
    assert numbers(frame.loc[low_height_scope], 'Height_Before_Document_Review').gt(0).all()
    assert floors[frame.Floors_Quality.isin(['unknown','conflict_unresolved'])].isna().all()
    unconfirmed_count = frame.Floors_Source.eq('CSDI_storeys_vertical_scope_unverified')
    assert floors[unconfirmed_count].isna().all()
    assert numbers(frame.loc[unconfirmed_count], 'GFA_Proxy_sqm').isna().all()
    assert not frame.Floors_Source.eq('CSDI_reported_storeys').any(), 'Undefined CSDI count scope cannot supply primary aboveground storeys'
    proxy = numbers(frame,'Floors_Height_Proxy')
    assert np.isclose(proxy.dropna(),np.maximum(1,np.floor((height[proxy.notna()]/3).round(9)+.5))).all(), 'Optional height proxy disagrees with published height'
    assert frame.loc[height.notna(),'Height_Source'].isin(['CSDI_top_minus_component_base','LandsD_top_minus_base']).all()
    assert frame.loc[height.notna(),'Height_Reference'].eq('structure_base_to_top').all()
    csdi = frame.Height_Source.eq('CSDI_top_minus_component_base')
    assert frame.loc[csdi,'CSDI_Match_Quality'].eq('same_component_id_type_footprint').all()
    assert np.isclose(height[csdi],numbers(frame.loc[csdi],'CSDI_Top_Elevation_mPD')-numbers(frame.loc[csdi],'CSDI_Base_Elevation_mPD')).all()
    assert np.isclose(numbers(frame,'GFA_Proxy_sqm'),numbers(frame,'SHAPE_Area')*floors,equal_nan=True).all()
    excluded = frame.Energy_Assessment_Scope.eq('excluded_nonbuilding_structure')
    assert frame.loc[excluded,'Calibrated_Main_Class'].eq(NONASSESSED).all()
    assert frame.loc[excluded,'GFA_Proxy_sqm'].isna().all() and floors[excluded].isna().all()
    tall_small = numbers(frame,'SHAPE_Area').lt(150) & height.gt(150)
    assert frame.loc[tall_small,'Structure_Use_Type'].eq('bridge_structure').all(), 'Unexplained small/tall component requires resolution'
    ratio = (height/floors).dropna()
    assert ratio.between(1,20).all(), 'Reported storeys have unresolved component scope'
    assert frame.loc[floors.notna(),'Floors_Source'].isin(['CSDI_reported_storeys','LandsD_storeys','OSM_building_levels','Documented_component_storeys']).all()
    documented = frame.Floors_Source.eq('Documented_component_storeys')
    assert frame.loc[documented,'Floors_Document_Evidence_IDs'].ne('[]').all()
    assert frame.loc[documented,'CSDI_Match_Quality'].eq('same_component_id_type_footprint').all()
    for col in ['Registered_GFA_sqm','GFA_Proxy_sqm','GFA_Best_Available_sqm']:
        values = numbers(frame,col).dropna()
        assert (np.isfinite(values) & values.ge(0)).all(), 'Invalid floor area: '+col
    probabilities = frame[['proba_'+main for main in MAIN_CLASSES]].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
    assert np.isclose(probabilities.sum(axis=1),1,atol=1e-6).all(), 'Classifier probability mass is not conserved'
    source_conflict = frame.ML_Decision_Status.eq('unresolved_source_conflict_unknown')
    assert frame.loc[source_conflict,'Calibrated_Main_Class'].eq(UNKNOWN).all()
    active_conflict = frame.Classification_QA.fillna('').str.contains('function_evidence_conflict|registered_area_function_conflict',regex=True)
    assert not frame.loc[active_conflict,'Calibrated_Main_Class'].isin(MAIN_CLASSES).any(), 'Conflicting function published'
    slender_unknown = frame.ML_Decision_Status.eq('slender_structure_function_unverified_unknown')
    assert frame.loc[slender_unknown,'Calibrated_Main_Class'].eq(UNKNOWN).all()
    validation_report = json.loads((Path(OUTPUT_DIR)/'model_validation.json').read_text(encoding='utf-8'))
    permitted_main = [label for label,details in validation_report['deployment_class_gate']['classes'].items()
                      if details['eligible_for_model_inference']]
    inferred = frame.Calibrated_Source.eq('Machine_Learning_v2')
    assert frame.loc[inferred,'Calibrated_Main_Class'].isin(permitted_main).all(), 'Unvalidated model main class published'
    for parent,details in validation_report['subclass_models'].items():
        permitted_children = [label for label,values in details.get('deployment_subclass_gate',{}).items()
                              if values['eligible_for_model_inference']]
        child_inferred = frame.Calibrated_Main_Class.eq(parent) & frame.Subclass_Source.eq('Machine_Learning_v2')
        assert frame.loc[child_inferred,'Calibrated_Sub_Class'].isin(permitted_children).all(), 'Unvalidated model subclass published'
    shared = frame.Permit_GFA_Scope.eq('shared_or_multiple_permits')
    assert numbers(frame.loc[shared],'GFA_DOMESTIC_SUM').isna().all()
    assert numbers(frame.loc[shared],'GFA_NONDOMESTIC_SUM').isna().all()
    incompatible_permit = frame.Floors_Source.eq('component_storeys_scope_unresolved') & frame.Permit_GFA_Scope.isin(
        ['single_structure', 'single_structure_component_scope_unresolved'])
    assert frame.loc[incompatible_permit, 'Permit_GFA_Scope'].eq('single_structure_component_scope_unresolved').all()
    permit_scope_unresolved = frame.Permit_GFA_Scope.isin(['single_structure_component_scope_unresolved',
                                                       'single_structure_component_use_scope_unresolved'])
    assert not frame.loc[permit_scope_unresolved,'Permit_GFA_Usable'].fillna(False).any()
    assert not frame.loc[permit_scope_unresolved,'Registered_GFA_Source'].eq('single_structure_single_permit').any()
    assert numbers(frame.loc[permit_scope_unresolved],'Observed_Residential_Area_Share').isna().all()
    assert numbers(frame.loc[permit_scope_unresolved],'Observed_Nondomestic_Area_Share').isna().all()
    assert not frame.loc[permit_scope_unresolved,'Calibrated_Source'].isin(['GFA_Mixed','GFA_Residential']).any()
    # Administrative permit accounts are never measurements of current use.
    assert numbers(frame,'Observed_Residential_Area_Share').isna().all()
    assert numbers(frame,'Observed_Nondomestic_Area_Share').isna().all()
    permit_adopted = frame.Permit_GFA_Usable.fillna(False).astype('string').str.lower().eq('true')
    physically_verified = frame.Permit_GFA_Physical_Scope_Verified.astype('string').str.lower().eq('true').fillna(False)
    assert not (permit_adopted & ~physically_verified).any()
    bd_usable = frame.BDBIAR_Source_Attributes_Usable.astype('string').str.lower().eq('true').fillna(False)
    assert frame.loc[~bd_usable,['BDBIAR_AGE','BDBIAR_CLASS']].isna().all().all()
    registration_scope = numbers(frame,'GROSSFLOORAREA').gt(0) & (
        frame.Floors_Source.eq('component_storeys_scope_unresolved') |
        frame.Registered_GFA_Source.eq('LandsD_structure_GFA_component_scope_unresolved'))
    assert numbers(frame.loc[registration_scope],'Registered_GFA_sqm').isna().all()
    assert frame.loc[registration_scope,'Registered_GFA_Source'].eq('LandsD_structure_GFA_component_scope_unresolved').all()
    registration_only = registration_scope & numbers(frame,'Documented_GFA_sqm').isna() & numbers(frame,'GFA_Proxy_sqm').isna()
    assert numbers(frame.loc[registration_only],'GFA_Best_Available_sqm').isna().all()
    assert frame.loc[registration_only,'GFA_Quality'].eq('registered_component_scope_unresolved').all()
    named_station = frame.CSDI_Component_Use_Type.eq('substation') & ~frame.Energy_Assessment_Scope.eq('excluded_nonbuilding_structure')
    assert frame.loc[named_station,'Calibrated_Main_Class'].eq('Commercial_商业类别').all()
    assert frame.loc[named_station,'Calibrated_Sub_Class'].eq('Other Commercial_其他商业').all()
    assert frame.loc[named_station,'CSDI_Name_Function_Evidence'].fillna('[]').ne('[]').all()
    assert frame.Calibrated_Main_Class.isin(MAIN_CLASSES+[MIXED,NONASSESSED,UNKNOWN]).all()
    invalid = []
    for _,row in frame.iterrows():
        vector = parse_weights(row.Calibrated_Mix_Proportion)
        observed = parse_weights(row.Observed_Main_Area_Proportions)
        if pd.notna(row.Calibrated_Mix_Proportion) and str(row.Calibrated_Mix_Proportion).strip() and not vector:
            invalid.append(row[ID])
        if pd.notna(row.Observed_Main_Area_Proportions) and str(row.Observed_Main_Area_Proportions).strip() and not observed:
            invalid.append(row[ID])
        if row.Calibrated_Main_Class in MAIN_CLASSES+[NONASSESSED] and row.Calibrated_Sub_Class != UNKNOWN:
            assert taxonomy().get(row.Calibrated_Sub_Class)==row.Calibrated_Main_Class, 'Subclass parent mismatch: '+row[ID]
        if vector and (abs(sum(vector.values())-1)>1e-6 or any(k not in taxonomy() for k in vector)):
            invalid.append(row[ID])
        if observed and (abs(sum(observed.values())-1)>1e-6 or any(k not in MAIN_CLASSES for k in observed)):
            invalid.append(row[ID])
    assert not invalid, 'Invalid quantitative vectors: '+str(invalid[:10])
    return {'buildings':len(frame),'unique_ids':frame[ID].nunique(),'raw_dimensions_preserved':True,
        'geometries_valid':True,'area_matches_geometry':True,'unsupported_height_source':0,
        'unknown_height_records':int(height.isna().sum()),'unknown_floor_records':int(floors.isna().sum()),
        'height_conflict_records':int(frame.Height_Quality.eq('conflict_unresolved').sum()),
        'floor_conflict_records':int(frame.Floors_Quality.eq('conflict_unresolved').sum()),
        'nonbuilding_structures_excluded_from_energy_area':int(excluded.sum()),
        'shared_permit_not_allocated_records':int(shared.sum()),'invalid_proportion_vectors':0,
        'model_inferences_respect_validation_gate':True,
        'main_class_counts':frame.Calibrated_Main_Class.value_counts().to_dict(),
        'height_quality_counts':frame.Height_Quality.value_counts().to_dict(),
        'floor_quality_counts':frame.Floors_Quality.value_counts().to_dict()}


def write_public_geojson(geometry, path):
    """Publish RFC 7946 longitude/latitude; retain metric geometry in Parquet."""
    rounding = pyogrio.get_gdal_config_option('OGR_WKT_ROUND')
    # GDAL's smart truncation can snap 113.999999901 to 114.0 despite precision=9.
    pyogrio.set_gdal_config_options({'OGR_WKT_ROUND': 'NO'})
    try:
        sanitize_dataframe_for_geojson(geometry.to_crs(4326)).to_file(
            path, driver='GeoJSON', RFC7946='YES', COORDINATE_PRECISION=9,
            AUTODETECT_JSON_STRINGS='NO')
    finally:
        pyogrio.set_gdal_config_options({'OGR_WKT_ROUND': rounding})


def generate_public_dataset(frame):
    columns = [c for c in list(EXPORT_MAPPING)+PROVENANCE if c in frame]
    public = frame[columns].rename(columns=EXPORT_MAPPING).copy()
    for column in ['CSDI_Record_Created_Date','CSDI_Record_Updated_Date']:
        raw_date = public[column].astype('string').str.replace(r'\.0$', '', regex=True)
        public[column] = pd.to_datetime(raw_date,format='%Y%m%d',errors='coerce').dt.strftime('%Y-%m-%d')
    public['Dataset_Version'] = '2.0'
    public['Unknown_Value_Policy'] = 'null_with_quality_and_source_flags'
    # A nearest point is useful linkage evidence, not a verified building attribute.
    point_usable = public.BDBIAR_Match_Quality.eq('inside_unique') & public.BDBIAR_Source_Attributes_Usable.astype('string').str.lower().eq('true').fillna(False)
    bdb_fields = [EXPORT_MAPPING['BDBIAR_CLASS'],EXPORT_MAPPING['BDBIAR_AGE']]
    withheld_point_attributes = public.loc[~point_usable,bdb_fields].notna().sum().to_dict()
    public.loc[~point_usable,bdb_fields] = np.nan
    assert public.loc[~point_usable,bdb_fields].isna().all().all()
    area,height,floors = (public['Footprint_Area_sqm_占地面积'],public['Height_m_建筑高度'],public['Floors_Count_地上层数'])
    flags = {
        'small_footprint_tall_building':(area<150)&(height>150),
        'component_height_below_2m':height.notna()&(height<2),
        'component_base_differs_point_terrain_over_3m':public.Component_Base_Above_Terrain_m.abs().gt(3),
        'height_storeys_ratio_outside_2_to_8_m':height.notna()&floors.notna()&~(height/floors).between(2,8),
        'height_sources_conflict':public.Height_Quality.eq('conflict_unresolved'),
        'storeys_sources_conflict':public.Floors_Quality.eq('conflict_unresolved'),
        'component_storeys_scope_unresolved':public.Floors_Source.eq('component_storeys_scope_unresolved'),
        'CSDI_reported_storeys_vertical_scope_unverified':public.Floors_Source.eq('CSDI_storeys_vertical_scope_unverified'),
        'height_storeys_component_scope_unresolved':public.Height_Source_Comparison_QA.fillna('').str.contains('height_storeys_component_scope_unresolved',regex=False),
        'registered_gfa_component_scope_unresolved':public.Registered_GFA_Source.eq('LandsD_structure_GFA_component_scope_unresolved'),
        'BDBIAR_source_integrity_or_identity_unresolved':public.BDBIAR_Source_Integrity_Status.isin(
            ['confirmed_source_internal_conflict','component_identity_unresolved']),
        'csdi_id_matched_but_component_geometry_changed':public.CSDI_Match_Quality.isin(['invalid_source_geometry','structure_type_mismatch','footprint_changed']),
        'OSM_component_height_reference_difference':public.Height_Source_Comparison_QA.fillna('').str.contains('OSM_differs',regex=False),
        'OSM_dimension_source_scope_unresolved':public.OSM_Dimension_Scope_QA.isin(['shared_component_scope','source_object_scope_unknown']),
        'documented_height_reference_conflict':public.Height_Source_Comparison_QA.fillna('').str.contains('documented_height_reference_conflict',regex=False),
        'ground_height_sources_conflict':public.Ground_Height_Quality.eq('conflict_unresolved'),
        'top_minus_terrain_rejected':public.Height_Terrain_QA.fillna('').ne(''),
        'registered_vs_uniform_floorplate_factor_over_3':public.Registered_GFA_sqm.notna()&public.GFA_Proxy_sqm.notna()&
            ((public.Registered_GFA_sqm/public.GFA_Proxy_sqm>3)|(public.GFA_Proxy_sqm/public.Registered_GFA_sqm>3)),
        'duplicate_footprint_geometry':public.Duplicate_Footprint_Geometry.astype(str).str.lower().eq('true'),
        'slender_structure_function_unverified':public.ML_Decision_Status.eq('slender_structure_function_unverified_unknown'),
    }
    public['Physical_Review_Flags'] = [';'.join(key for key,mask in flags.items() if bool(mask.iloc[i])) for i in range(len(public))]
    resolved_bridge = public.Structure_Use_Type.eq('bridge_structure') & flags['small_footprint_tall_building']
    public['Small_Tall_Resolution'] = np.where(resolved_bridge,'identified_bridge_pylon_height_retained_energy_area_excluded',
        np.where(flags['small_footprint_tall_building'],'unresolved','not_triggered'))
    summary={'thresholds_are_review_triggers_not_error_proofs':True,'records':len(public),
        'flag_counts':{key:int(mask.sum()) for key,mask in flags.items()},
        'distinct_flagged_records':int(public.Physical_Review_Flags.ne('').sum()),
        'height_rounded_to_15m_records':int(height.round(1).eq(15).sum()),
        'primary_height_definition':'structure_base_to_top; ground height is separate',
        'csdi_match_quality_counts':public.CSDI_Match_Quality.value_counts().to_dict(),
        'osm_dimension_scope_counts':public.OSM_Dimension_Scope_QA.value_counts(dropna=False).to_dict(),
        'ground_height_known_records':int(public.Ground_To_Top_Height_m.notna().sum()),
        'small_tall_identified_bridge_structures':int(resolved_bridge.sum()),
        'small_tall_unresolved_records':int((flags['small_footprint_tall_building'] & ~resolved_bridge).sum()),
        'nonunique_or_outside_point_attributes_withheld':withheld_point_attributes,
        'height_rounded_to_15m_by_source':public.loc[height.round(1).eq(15),'Height_Source'].value_counts().to_dict()}
    (Path(OUTPUT_DIR)/'dimension_qa_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    public.drop(columns='geometry',errors='ignore').to_csv(PUBLIC_DATASET_CSV_PATH,index=False,encoding='utf-8-sig')
    geometry = gpd.GeoDataFrame(public,geometry=frame.geometry,crs=frame.crs)
    write_public_geojson(geometry, PUBLIC_DATASET_GEOJSON_PATH)
    geometry.to_parquet(Path(PUBLIC_DATASET_CSV_PATH).with_suffix('.parquet'),index=False)
    return public


def proportion_values_equal(a,b):
    first,second = parse_weights(a),parse_weights(b)
    if any(pd.notna(value) and str(value).strip() and not parsed for value,parsed in [(a,first),(b,second)]):
        return False
    def canonical(vector):
        result={}
        for key,value in vector.items():
            label=sub_label(key)
            label=key if label==UNKNOWN else label
            result[label]=result.get(label,0.)+value
        return result
    first,second=canonical(first),canonical(second)
    return first.keys()==second.keys() and all(np.isclose(first[key],second[key],atol=1e-6,rtol=1e-6) for key in first)


def compare_releases(public):
    path = Path(LEGACY_OUTPUT_DIR)/'HK_UBEM_Buildings_Public_v1_1.csv'
    old = pd.read_csv(path,low_memory=False,dtype={'Building_ID_建筑ID':'string'}).set_index('Building_ID_建筑ID')
    new = public.set_index('Building_ID_建筑ID').reindex(old.index)
    assert set(public['Building_ID_建筑ID']) == set(old.index)
    comparison = pd.DataFrame(index=old.index)
    flags = {}
    for col,alias in [('Footprint_Area_sqm_占地面积','Footprint_Area'),('Height_m_建筑高度','Height_m'),
                      ('Floors_Count_地上层数','Floors'),('UBEM_Main_Class_主类别','Main_Class'),
                      ('UBEM_Sub_Class_子类别','Sub_Class'),('UBEM_Mixed_Proportions_混合比例','Use_Proportions')]:
        a,b = old[col],new[col]
        comparison['Old_'+alias],comparison['New_'+alias] = a,b
        if alias in ['Footprint_Area','Height_m','Floors']:
            changed = ~np.isclose(pd.to_numeric(a,errors='coerce'),pd.to_numeric(b,errors='coerce'),equal_nan=True,rtol=1e-8,atol=1e-6)
        elif alias=='Use_Proportions':
            comparison['Use_Proportions_Text_Changed']=a.fillna('').ne(b.fillna('')).to_numpy()
            changed=np.array([not proportion_values_equal(x,y) for x,y in zip(a,b)])
        else:
            changed = a.fillna('').ne(b.fillna('')).to_numpy()
        comparison[alias+'_Changed'] = changed
        flags[alias] = int(np.sum(changed))
    old_proxy = pd.to_numeric(old['Footprint_Area_sqm_占地面积'],errors='coerce')*pd.to_numeric(old['Floors_Count_地上层数'],errors='coerce')
    comparison['Old_GFA_Proxy_sqm'] = old_proxy
    for col in ['Height_Source','Height_Quality','Floors_Source','Floors_Quality','Proportion_Method','Proportion_Quality',
                'GFA_Best_Available_sqm','GFA_Quality','Registered_GFA_sqm','GFA_Proxy_sqm','Classification_QA','Geometry_Review_Status']:
        comparison['New_'+col] = new[col]
    comparison['Height_Withheld_Unknown'] = new['Height_m_建筑高度'].isna()
    comparison['Floors_Withheld_Unknown'] = new['Floors_Count_地上层数'].isna()
    comparison.to_csv(Path(OUTPUT_DIR)/'v1_1_to_v2_building_changes.csv',encoding='utf-8-sig')
    matrix = pd.crosstab(old['UBEM_Main_Class_主类别'],new['UBEM_Main_Class_主类别'])
    matrix.to_csv(Path(OUTPUT_DIR)/'v1_1_to_v2_class_transition.csv',encoding='utf-8-sig')
    known_gfa = new.GFA_Best_Available_sqm.notna()
    height_known = new['Height_m_建筑高度'].notna()
    summary = {'buildings':len(public),'id_set_unchanged':True,'changed':flags,
        'old_height_known':int(old['Height_m_建筑高度'].notna().sum()),'new_height_known':int(height_known.sum()),
        'old_floor_known':int(old['Floors_Count_地上层数'].notna().sum()),'new_floor_known':int(new['Floors_Count_地上层数'].notna().sum()),
        'old_default_height_not_replaced_by_another_constant':True,
        'new_main_class_counts':public['UBEM_Main_Class_主类别'].value_counts().to_dict(),
        'new_height_source_counts':public.Height_Source.value_counts().to_dict(),
        'old_gfa_proxy_all_records_sqm':float(old_proxy.sum()),'new_gfa_known_records':int(known_gfa.sum()),
        'new_gfa_best_available_covered_records_sqm':float(new.GFA_Best_Available_sqm.sum()),
        'old_gfa_proxy_on_same_new_covered_ids_sqm':float(old_proxy[known_gfa].sum()),
        'gfa_comparison_caution':'Coverage changed. These sums are not estimates of a physical change in Hong Kong total GFA.',
        'height_delta_median_on_common_known_ids_m':float((new.loc[height_known,'Height_m_建筑高度']-old.loc[height_known,'Height_m_建筑高度']).median()),
        'old_release_sha256':file_hash(path)}
    (Path(OUTPUT_DIR)/'comparison_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    compare_all_common_fields(old,new)
    return old,new,summary


def compare_all_common_fields(old,new):
    """Cover every original CSV field, including point attributes and probabilities."""
    summaries,changes = [],[]
    for col in old.columns.intersection(new.columns):
        a,b = old[col],new[col]
        if col == EXPORT_MAPPING['Calibrated_Mix_Proportion']:
            changed = np.array([not proportion_values_equal(x,y) for x,y in zip(a,b)])
            policy = 'canonical_labels_and_numeric_shares'
        elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            changed = ~np.isclose(a.to_numpy(dtype=float,na_value=np.nan),b.to_numpy(dtype=float,na_value=np.nan),equal_nan=True,atol=1e-6,rtol=1e-8)
            policy = 'numeric_tolerance_1e-6_absolute_1e-8_relative'
        else:
            changed = a.fillna('').astype(str).ne(b.fillna('').astype(str)).to_numpy()
            policy = 'literal_text_or_categorical'
        summaries.append({'field':col,'old_nonmissing':int(a.notna().sum()),'new_nonmissing':int(b.notna().sum()),
            'changed_records':int(changed.sum()),'newly_withheld_nulls':int((a.notna()&b.isna()).sum()),
            'previously_null_filled':int((a.isna()&b.notna()).sum()),'comparison_policy':policy})
        if changed.any():
            changes.append(pd.DataFrame({'Building_ID_建筑ID':old.index[changed],'field':col,
                'old_value':a.loc[changed].astype('string').to_numpy(),
                'new_value':b.loc[changed].astype('string').to_numpy()}).astype('string'))
    pd.DataFrame(summaries).to_csv(Path(OUTPUT_DIR)/'all_common_field_change_summary.csv',index=False,encoding='utf-8-sig')
    table = pd.concat(changes,ignore_index=True) if changes else pd.DataFrame(columns=['Building_ID_建筑ID','field','old_value','new_value']).astype('string')
    table.to_parquet(Path(OUTPUT_DIR)/'all_common_field_changes.parquet',index=False)


def create_review_tables(public):
    frame = public.copy()
    unresolved = frame.Height_Quality.isin(['unknown','conflict_unresolved']) | frame.Floors_Quality.isin(['unknown','conflict_unresolved'])
    unresolved |= frame['UBEM_Main_Class_主类别'].eq(UNKNOWN) | frame.Proportion_Quality.eq('unknown')
    unresolved |= frame.Physical_Review_Flags.ne('') | frame.Classification_QA.fillna('').ne('')
    cols = ['Building_ID_建筑ID','Height_m_建筑高度','Floors_Count_地上层数','Height_Source','Height_Quality',
        'Height_Reference','Ground_To_Top_Height_m','CSDI_Match_Quality',
        'Floors_Source','Floors_Quality','UBEM_Main_Class_主类别','UBEM_Sub_Class_子类别','Classification_QA','Geometry_Review_Status','Physical_Review_Flags']
    frame.loc[unresolved,cols].to_csv(Path(OUTPUT_DIR)/'unresolved_building_review_queue.csv',index=False,encoding='utf-8-sig')
    frame['_terrain_band'] = pd.cut(frame.Terrain_Elevation_mPD,[-np.inf,10,100,300,np.inf],labels=['low','medium','high','mountain']).astype('string').fillna('unknown')
    frame['_area_band'] = pd.cut(frame['Footprint_Area_sqm_占地面积'],[0,50,150,1000,np.inf],labels=['tiny','small','medium','large']).astype('string')
    frame['_flagged'] = unresolved
    samples = []
    for _,group in frame.groupby(['Height_Source','_terrain_band','_area_band','UBEM_Main_Class_主类别','_flagged'],observed=True,dropna=False):
        samples.append(group.sample(min(2,len(group)),random_state=42))
    sample = pd.concat(samples).drop_duplicates('Building_ID_建筑ID')
    for field in ['Independently_Verified_Height_m','Independently_Verified_Storeys','Independently_Verified_Function','Evidence_URL','Reviewer_Notes']:
        sample[field] = ''
    sample.to_csv(Path(OUTPUT_DIR)/'stratified_manual_validation_template.csv',index=False,encoding='utf-8-sig')
    cases = public[public['Building_ID_建筑ID'].isin(['181189','170369','192326','87318'])]
    cases.to_csv(Path(OUTPUT_DIR)/'reader_cases_v2.csv',index=False,encoding='utf-8-sig')


def make_figures(old,new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(OUTPUT_DIR)/'figures'
    directory.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    height_bins = np.arange(0,max(old['Height_m_建筑高度'].max(),new['Height_m_建筑高度'].max())+20,10)
    floor_bins = np.arange(0,max(old['Floors_Count_地上层数'].max(),new['Floors_Count_地上层数'].max())+6,3)
    for data,label in [(old,'v1.1'),(new,'v2 known values')]:
        axes[0].hist(data['Height_m_建筑高度'].dropna(),bins=height_bins,histtype='step',label=label)
        axes[1].hist(data['Floors_Count_地上层数'].dropna(),bins=floor_bins,histtype='step',label=label)
    for ax,title in zip(axes,['Height (m); missing values excluded','Reported storeys; v2 unknown values excluded']):
        ax.set_title(title);ax.set_yscale('log');ax.set_ylabel('Buildings');ax.legend()
    fig.tight_layout();fig.savefig(directory/'dimensions_before_after.png',dpi=180);plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,4.5))
    labels = ['Height','Storeys','GFA best available']
    counts = [new['Height_m_建筑高度'].notna().sum(),new['Floors_Count_地上层数'].notna().sum(),new.GFA_Best_Available_sqm.notna().sum()]
    ax.bar(labels,counts,label='Known/reported/explicit estimate')
    ax.bar(labels,[len(new)-n for n in counts],bottom=counts,label='Unknown',color='#b8bec9')
    ax.set_ylabel('Buildings');ax.set_title('v2 coverage; estimates retain method flags');ax.legend()
    fig.tight_layout();fig.savefig(directory/'coverage_and_unknowns.png',dpi=180);plt.close(fig)
    model=json.loads((Path(OUTPUT_DIR)/'model_validation.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(6,5))
    matrix=np.array(model['confusion_matrix'])
    image=ax.imshow(matrix,cmap='Blues');fig.colorbar(image,ax=ax)
    names=[name.split('_')[0] for name in model['class_order']]
    ax.set_xticks(range(3),names,rotation=20);ax.set_yticks(range(3),names)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    ax.set_xlabel('Prediction');ax.set_ylabel('Source-inferred label');ax.set_title('Spatial holdout before deployment gates\nWeak source labels')
    fig.tight_layout();fig.savefig(directory/'model_spatial_validation.png',dpi=180);plt.close(fig)


def write_documentation(public,validation,comparison):
    meanings = {
        'Height_m_建筑高度':'Structural component top minus base elevation; towers and podiums remain separate; not architectural spire height',
        'Height_Source':'CSDI matched component top-minus-base, native top-minus-base, missing or conflict',
        'Height_Reference':'structure_base_to_top for known primary heights; never a terrain/OSM substitute',
        'Height_Quality':'Source-reported component elevations, with documentary or height/storeys component-scope conflicts withheld; not independent height measurements',
        'Ground_To_Top_Height_m':'Separate datum top minus point terrain estimate, otherwise comparable OSM reported ground height with unverified reference',
        'Ground_Height_Source':'CSDI/LandsD top minus point DTM, OSM reported height with reference unverified, missing or conflict',
        'Ground_Height_Quality':'Point terrain estimate, reported reference unverified, or component/documentary scope unresolved estimate; never silently a component height',
        'Component_Base_Above_Terrain_m':'Component base elevation minus sampled terrain; raised podium bases and sampling errors both possible',
        'CSDI_Geometry_IoU':'Footprint intersection/union after reprojection; >=0.95 required for donation, local footprint retained',
        'CSDI_Match_Quality':'Exact unique BuildingCSUID, compatible structure type and valid footprint IoU >=0.95; otherwise attributes not used',
        'CSDI_Base_Elevation_mPD':'CSDI raw component base elevation in Hong Kong Principal Datum; may be a podium roof',
        'CSDI_Top_Elevation_mPD':'CSDI raw component top elevation in Hong Kong Principal Datum; spire/antenna conventions may differ',
        'CSDI_Reported_Storeys':'Raw CSDI reported count; retrieved official schema does not define aboveground/basement/mechanical-floor scope; not used alone as primary aboveground storeys',
        'CSDI_Storeys_Scope_QA':'Whether an eligible CSDI count exists and agrees with an explicit aboveground source; agreement does not establish special-floor conventions',
        'CSDI_Building_Name_EN':'Raw matched-ID CSDI English component name; check CSDI_Match_Quality and distinguish named ancillary facilities from parent buildings',
        'CSDI_Building_Name_TC':'Raw matched-ID CSDI Chinese component name; original LandsD name is retained separately',
        'CSDI_Component_Use_Type':'Explicit substation/transformer-station source-name clue from an Active component with matching ID/type/footprint; empty is not source-certified use',
        'CSDI_Name_Function_Evidence':'JSON exact active component-name rule, geometry match and previous function decision; inferred function under the project taxonomy, never an official measured use field',
        'CSDI_Native_Building_ID':'BuildingID in the official CSDI snapshot; distinct from the local Building_ID_建筑ID',
        'CSDI_Record_Created_Date':'Official record creation date as YYYY-MM-DD; not construction completion date; raw dates remain in frozen source',
        'CSDI_Record_Updated_Date':'Official record timestamp as YYYY-MM-DD; not measurement or renovation date; raw dates remain in frozen source',
        'Floors_Source':'Reported CSDI/LandsD/OSM or explicitly documented same-component storeys; height/3 is never used here',
        'Reported_Attribute_Evidence':'JSON source facts with attribute-specific scope, unit, document date, source URL and adoption decision; unavailable/unharmonized values are not silently used as main fields',
        'Floors_Document_Evidence_IDs':'JSON IDs of explicitly scoped aboveground component storey documents selected for the primary count',
        'Floors_Before_Document_Review':'Count after ordinary source fusion and before documented component-count reconciliation',
        'Floors_Source_Before_Document_Review':'Floor source/conflict state before documented component-count reconciliation',
        'Floors_Source_Comparison_QA':'Explicit unresolved same-component documentary storey-reference conflicts; primary storeys and their geometric GFA proxy withheld, raw source counts retained',
        'Dimension_Ratio_Scope_QA':'Height/count outside2–8m is a review signal, not proof of either value being wrong. Existing extreme1–20 scope gate and explicit documentary scope withdrawals remain separate',
        'Component_Vertical_Scope_QA':'Explicit unresolved component vertical-scope control where present; source-reported numeric candidates remain separately preserved',
        'Documented_GFA_sqm':'Positive owner/source-reported gross floor area with explicit same-component scope; distinct from a registered area',
        'Documented_GFA_Evidence_IDs':'JSON IDs of documents supporting the same-component gross floor area; lettable/saleable/office/internal areas do not qualify',
        'Documented_Function_Evidence':'JSON reviewed primary-source component function facts; prior decisions and scope limitations retained separately',
        'Function_Before_Document_Review':'JSON main/subclass, source and QA before source-documented function reconciliation',
        'ML_Subclass_Candidate':'Unverified model suggestion, including withheld housing subtypes; not an adopted class or floor-area fraction',
        'Height_Before_Document_Review':'Source-fused component height candidate before documentary review; retained when documentary or height/storeys component-scope conflict withholds the primary height; not an accepted fallback height',
        'ML_Main_Class_Candidate':'Main-model argmax proposal; retained separately from the adopted source/documented class. High-confidence disagreement is a review trigger, not proof that the source is wrong',
        'OSM_Function_Candidate_Main':'Function-bearing polygon candidate before excluding shared component scope; not necessarily adopted',
        'OSM_Function_Candidate_Sub':'Polygon subclass candidate before component and source-conflict checks; not an observed use-area fraction',
        'OSM_Function_Evidence':'JSON polygon object IDs, tags, inferred use, raw matched-component counts and overlap metrics. adoption_eligible refers only to component/geometry thresholds; source-use conflicts and documentary reconciliation apply separately',
        'Structure_Use_Type':'Explicitly matched structural use: bridge, equipment or water-storage structure; empty is not reviewed/identified',
        'Structure_Use_Name':'Documented landmark/component name; original LandsD name is not overwritten',
        'Structure_Use_Source':'Official document plus spatial identification, or comparable unique OSM polygon with explicit structural tags',
        'Structure_Use_Evidence':'Official factsheet or exact source OSM object URL supporting structural-use identification, not a new height survey',
        'Structure_Use_Quality':'Scope/strength of structural-use evidence; OSM tags alone are not independently verified measurements',
        'Energy_Assessment_Scope':'Identified nonbuilding structures are excluded from floors/GFA energy allocation; other rows may still require use evidence',
        'Small_Tall_Resolution':'Eight tall small-footprint cases identified as bridge pylons; physical heights retained, not modeled as occupied buildings',
        'Pre_Structure_Review_Main_Class':'Main function before explicit structural-use review, for traceability',
        'Pre_Structure_Review_Sub_Class':'Subclass before explicit structural-use review, for traceability',
        'Floors_Height_Proxy':'Optional height/3 m estimate, nearest integer with half ties upward (ratio normalized to 9 decimals); excluded from reported storeys and GFA',
        'GFA_Proxy_sqm':'Geometric footprint times reported storeys; uniform floorplate assumption',
        'Registered_GFA_sqm':'Positive reported structure registration without unresolved component scope, otherwise uniquely scoped single-structure permit total; raw registration remains separately available',
        'GFA_Best_Available_sqm':'Registered GFA preferred, then documented same-component GFA, then uniform-floorplate proxy; compare with coverage',
        'Observed_Main_Area_Proportions':'Empty without independently scoped current-use area evidence; permit domestic/nondomestic accounts are not observations of current function',
        'Observed_Residential_Area_Share':'Null in this revision: no independent component current-residential-area measurement is adopted',
        'Observed_Nondomestic_Area_Share':'Null in this revision: permit nondomestic accounts are not observed current-use shares',
        'Evidence_Subclass_Weights':'Classification evidence weights; never measured floor-area fractions',
        'Proportion_Method':'single_function_assumption, registered_main_area_only, unknown or nonassessed label',
        'Functional_Subclasses':'JSON list of evidenced functional components, with no quantitative allocation',
        'Physical_Review_Flags':'Semicolon-separated review triggers; flags do not by themselves prove an error',
        'BDBIAR_Match_Quality':'Only inside_unique donates direct function evidence; nearest/shared/tied matches require review',
        'BUILDINGCSUID':'Stable CSDI component identifier used for exact source matching; original local value retained',
        'NUMABOVEGROUNDSTOREYS':'Original footprint-table storey count, including null/zero/conflicting reports; not overwritten',
        'TOPHEIGHT':'Original footprint-table top elevation in mPD; never directly interpreted as relative height',
        'BASEHEIGHT':'Original footprint-table base elevation in mPD; missing remains null, not zero',
        'Source_Footprint_Area_sqm':'Original source area attribute before geometry-derived replacement',
        'Geometry_Quality':'source_valid or repaired_make_valid; local footprint retained after valid polygon repair',
        'Footprint_Area_QA':'Whether original stored area differs from recomputed projected polygon area',
        'Duplicate_Footprint_Geometry':'Equal planar footprint geometry; distinct tower/podium components may legitimately coincide',
        'Height_Terrain_QA':'Rejected point-terrain difference; applies to the separate ground-height estimate',
        'Height_Source_Comparison_QA':'OSM/documentary reference differences or height_storeys_component_scope_unresolved; incompatible height/count scope withholds primary height and retains the candidate and raw elevations',
        'Floors_Quality':'Reported, missing, conflicting/scope unresolved, or not applicable to nonbuilding structure',
        'Floor_Height_Proxy_Assumption_m':'3 m assumed storey height, only where optional height-derived proxy exists',
        'OSM_Height_m':'Source-reported height parsed into metres from a comparable polygon; component reference unverified',
        'OSM_Storeys':'OSM building:levels report from a comparable polygon, including conflicting withheld values',
        'OSM_Storeys_Scope_QA':'OSM levels donate only when raw source-object geometry intersects one official component by more than 1 sqm and component base is within 3 m of point terrain; shared or unknown scope is withheld',
        'OSM_Dimension_Match_QA':'Polygon coverage/identity/conflicting-attribute checks for OSM dimension donation',
        'OSM_Dimension_Object_ID':'Source object and geometry identity used for the dimension match',
        'OSM_Dimension_Scope_QA':'Raw source-object geometry scope across official components before dimension-match thresholds; shared or unknown scope remains candidate evidence only',
        'OSM_Dimension_Raw_Matched_Component_Count':'Distinct official component count intersected by the OSM source object by more than 1 sqm, before dimension-match thresholds',
        'OSM_Dimension_Candidate_Evidence':'JSON candidate records with source IDs, raw component count, source geometry overlap metrics and reported dimensions',
        'Terrain_Elevation_mPD':'Coordinate-checked single-point DTM elevation, not a surveyed building base',
        'Registered_GFA_Source':'Structure registration or uniquely scoped single-structure permit; LandsD_structure_GFA_component_scope_unresolved marks a withheld raw registration, not an adopted area',
        'GFA_Proxy_Method':'Uniform floorplate approximation or unknown/not applicable; no inferred height/3 storeys used',
        'GFA_Quality':'reported_structure_gfa_not_independently_validated, registered_scope_checked for eligible permit, documented_component_gfa, uniform_floorplate_proxy, registered_component_scope_unresolved, permit scope unresolved, unknown or not applicable; no label certifies physical area',
        'Permit_IDs':'JSON list of linked occupation permits; link multiplicity is retained',
        'Permit_GFA_Scope':'single_structure, single_structure_component_scope_unresolved, single_structure_component_use_scope_unresolved or shared_or_multiple_permits; one link alone does not settle contradictory vertical/use scope',
        'Permit_GFA_Usable':'Requires explicit same-physical-component scope verification as well as eligible administrative link and complete area; a single link alone is insufficient',
        'Permit_GFA_Physical_Scope_Verified':'False unless independent physical/vertical scope evidence explicitly establishes the component allocation; does not certify current measured use shares',
        'Permit_GFA_Link_QA':'Administrative single versus shared/multiple permit linkage; uniqueness is not physical/phase/use correspondence',
        'Permit_Raw_DOMESTICGFA':'Unaltered domestic permit account amount for an administratively single link; not measured current residential area',
        'Permit_Raw_NONDOMESTICGFA':'Unaltered nondomestic permit account amount for an administratively single link; not measured current commercial area',
        'GFA_DOMESTIC_SUM':'Reported domestic permit-account area for a single link; retained for audit, never used to infer current observed residential share',
        'GFA_NONDOMESTIC_SUM':'Reported nondomestic permit-account area for a single link; retained for audit, never used as current observed use share',
        'BDBIAR_Source_Block_ID':'Stable NSEARCH1_E Block ID from the frozen BDBIAR source; OBJECTID is snapshot-specific',
        'BDBIAR_Source_Integrity_Status':'Explicit internal source conflict, component-identity uncertainty, snapshot-coordinate uncertainty or linkage-only state; these are distinct evidence roles',
        'BDBIAR_Source_Integrity_QA':'Reason for withholding BD donations or recording a source uncertainty; no replacement coordinates or identities are guessed',
        'BDBIAR_Source_Integrity_Evidence':'Audit locator in the bundled integrity registry; evidence is retained separately with the native review archive',
        'BDBIAR_Source_Attributes_Usable':'Only inside_unique geometry plus absence of explicit source/identity quarantine permits BD age/class donation; true is not factual certification',
        'BDBIAR_AGE_CANDIDATE':'Original BD-derived year candidate retained when age donation is withheld; not an adopted completion year',
        'BDBIAR_CLASS_CANDIDATE':'Original BD classification candidate retained when donation is withheld; not current component use',
        'Proportion_Quality':'Observed main area, assumed single function, unknown, or nonassessed label; never model probability',
        'Functional_Main_Classes':'JSON list of evidenced main functional components; no quantitative area allocation',
        'Classification_QA':'Preserved function-source/registered-area/parent-subclass conflicts; empty means no configured flag',
        'Function_Quality':'source_inferred, model_inferred or unknown; none denotes independent whole-stock validation',
        'ML_Decision_Status':'Source retained, inferred main function, low confidence, insufficient class validation, unverified slender structure, or source-conflict unknown',
        'ML_Subclass_Confidence':'Maximum subclass model probability only where subclass inference was considered; not area share',
        'Subclass_Source':'Retained source evidence, v2 model inference, source-conflict withholding or missing; may differ from main-class source',
        'Geometry_Review_Status':'Geometry-review/cache state; function inheritance does not dissolve or delete original footprints',
        'LLM_Review_Status':'Step 4 API/native checkpoint replay status; matching_native_checkpoint requires exact current input/geometry/policy guards and separate root adoption. Reviewed unknown, partial, low-confidence, rejected and stale responses are not adopted labels. Frozen native source packets and the separate review ledger retain evidence; default replay makes no model requests',
        'CSDI_Basement_Storeys':'Original CSDI basement count, separate from above-ground storeys and not added to proxy GFA',
        'CSDI_Structure_Type':'CSDI Tower, Podium, Open-sided Structure or Temporary Structure; not itself a functional class',
        'CSDI_Status':'Original status in the frozen public distribution; snapshot acquisition is not survey date',
        'Dataset_Version':'Version of the revised dataset and field semantics',
        'Unknown_Value_Policy':'Null with quality/source fields; class unknown has its explicit bilingual taxonomy label',
    }
    meanings.update({EXPORT_MAPPING[key]:text for key,text in {
        ID:'Original LandsD structure identifier; unique string and complete original inventory retained',
        'SHAPE_Area':'Area of the valid projected structure footprint; overlapping tower/podium records are not disjoint land area',
        'Estimated_Floors':'Conflict-checked reported above-ground storeys; null includes nonbuilding not applicable',
        'Calibrated_Main_Class':'Source- or model-inferred dominant function; unknown/mixed/nonassessed are explicit classes',
        'Calibrated_Sub_Class':'Canonical functional subclass consistent with parent; unknown if unsupported',
        'Calibrated_Mix_Proportion':'Subclass:share vector; single-function 1 is an assumption, mixed shares need scoped area evidence',
        'Calibrated_Source':'Main-function source or model-inference method; separate from dimension/subclass provenance',
        'Calibrated_Notes':'Interpretation of main/subclass inference and proportion limitations',
        'ML_Confidence':'Maximum main-model probability for every record, including source-retained classes; not source correctness confidence',
        'BUILDINGSTRUCTURETYPE':'Original T non-podium structure or P podium; T also contains structural infrastructure',
        'CATEGORY':'Original legal/administrative building category; not automatically building function',
        'STATUS':'Original local footprint-table lifecycle status code',
        'GROSSFLOORAREA':'Original structure-level registered GFA attribute before validity/scope selection',
        'BUILDINGNAMEEN':'Original selected official English building name',
        'BUILDINGNAMETC':'Original selected official Traditional Chinese building name',
        'BDBIAR_OBJECTID':'Nearest selected BDBIAR point ID; only inside_unique donates direct use evidence',
        'BDBIAR_CLASS':'BDBIAR classification only for a point uniquely inside the footprint; composite labels are not pure functions, other point attributes are withheld',
        'BDBIAR_AGE':'Source completion/age-year only for a point uniquely inside the footprint; not elapsed age; outside/shared/nearest-only point years are withheld',
        'distance_to_bdbiar':'Distance to the selected BDBIAR point; proximity alone does not prove attribute correspondence',
        'Matched_POIs':'Serialized matched POI evidence; tenant counts do not determine use-area shares',
        'osmid':'Matched OSM/Overture record identity inherited from spatial aggregation; not a local building ID',
        'building':'Matched OSM building tag; generic yes does not establish function',
        'amenity':'Matched OSM amenity evidence; may describe a tenant',
        'shop':'Matched OSM shop evidence; may describe a tenant',
        'OZP_ZONE_LABEL':'Matched planning-zone code; source literal NULL is normalized to null (original in frozen OZP inputs), while Nullah description is retained; planning intent does not prove current use',
        'OZP_DESC_ENG':'English statutory planning-zone description',
    }.items()})
    for main in MAIN_CLASSES:
        meanings[EXPORT_MAPPING['proba_'+main]] = 'Uncalibrated main-model probability for '+main+'; three probabilities sum to 1, never floor-area shares'
    area_fields={'Source_Footprint_Area_sqm','Registered_GFA_sqm','Documented_GFA_sqm','GFA_Proxy_sqm','GFA_Best_Available_sqm',
        'GFA_DOMESTIC_SUM','GFA_NONDOMESTIC_SUM',EXPORT_MAPPING['SHAPE_Area'],EXPORT_MAPPING['GROSSFLOORAREA']}
    elevation_fields={'TOPHEIGHT','BASEHEIGHT','Terrain_Elevation_mPD','CSDI_Base_Elevation_mPD','CSDI_Top_Elevation_mPD'}
    metre_fields={EXPORT_MAPPING['Estimated_Height'],EXPORT_MAPPING['distance_to_bdbiar'],'Ground_To_Top_Height_m',
        'Component_Base_Above_Terrain_m','OSM_Height_m','Floor_Height_Proxy_Assumption_m'}
    dictionary=[]
    for col in public:
        dictionary.append({'field':col,'nonmissing':int(public[col].notna().sum()),'dtype':str(public[col].dtype),
            'unit_or_format':'m2' if col in area_fields else 'm above Hong Kong Principal Datum' if col in elevation_fields else 'm' if col in metre_fields else 'see meaning; categorical/count/serialized evidence',
            'null_meaning':'No configured flag' if col.endswith('_QA') else 'Unknown, unavailable or not applicable; consult source/quality fields',
            'meaning':meanings[col]})
    pd.DataFrame(dictionary).to_csv(Path(OUTPUT_DIR)/'field_dictionary.csv',index=False,encoding='utf-8-sig')
    text = f'''HKBFETD v2.0 — 修订启动于 2026-10-22；实际生成时间见 pipeline_run_manifest.json
建筑/结构记录数：{len(public):,}；完整保留 v1/v1.1，结构 ID 集合不变。
有效高度：{comparison['new_height_known']:,}；有效报告层数：{comparison['new_floor_known']:,}。
主高度空值（缺失/冲突）：{validation['unknown_height_records']:,}；主层数空值（缺失/冲突/不适用）：{validation['unknown_floor_records']:,}。

Height_m 主列采用结构本体的源报顶底高程差，塔楼与裙楼分别计算。不是统一口径的地面至顶部或建筑学高度；原始顶部是否包含桅杆等构件不保证一致。
CSDI Building 原始高程由地政总署 2026-10-20 官方批量快照补充，按唯一 BuildingCSUID、结构类型及有效足迹交并比 >=0.95 匹配；原有足迹不替换。
本版保持原 2025-02-06 建筑清单以便逐栋对照，不声称覆盖快照中所有新增或变更的建筑；未匹配及几何变化另作标记。
公开分发者生成的 height_m 不使用；所有主高度直接由原始 BaseHeight/TopHeight 重算。快照版本、下载链接、SHA256 见 data/sources 元数据。
原表 T 泛指非裙楼结构，CSDI 进一步区分 Tower、Open-sided Structure、Temporary Structure；原表 P 仅匹配 Podium。
Ground_To_Top_Height_m 单列保存地面至顶部估计，优先 CSDI 顶高减点 DTM，否则原始 LandsD 顶高减 DTM 或通过原始来源对象单组件范围门控的 OSM 报告高度。
点地形证据采于 2026-08-30；单点采样、时间变化、屋顶/建筑学高度及不同来源的参考面限制仍存在。它不作为结构本体高度的缺失替代。
ICC 塔楼底部在裙楼上，因此结构本体高度与地面至顶部高度不同；中银大厦结构顶高也不能代表包含尖顶的建筑学总高。
OSM_Height_m 原标签值保留，单位按 OSM 规则解析；与本体高度不同可能源自参考面不同，仅标记复核，不混算或平均。
Floors_Count 使用通过源间冲突检查或明确组件文档调和后的报告层数；Floors_Height_Proxy 是单独的 3 m 层高假设，未当作报告层数或面积计算依据。
逐栋文档中的塔楼、裙楼、整栋、附楼、楼层标签及不同面积口径分别记录在 Reported_Attribute_Evidence。中环广场75层保留对应组件文档；泓富产业千禧广场32层与原始25层尚无可靠实体层对应，主层数及依赖面积代理留空，办公用途保留。朗豪坊59个level含屋面层，撤回旧规范地上层数采纳及其面积代理，不猜57或46层。帝豪阁C座与Goodman Interlink的层数口径冲突同样保留原报告并撤不可靠主层数，不向邻近组件捐值。
中环广场CSDI顶底差376.8m与业主374m（含桅杆）的参考范围未能统一，主高度留空；保留原高程、复核前高度及来源差异标记，不强行替换为374m。
主层数与当前结构对应；OSM 整栋建筑层数可能包含裙楼，只有原始来源对象几何在阈值前仅与一个官方组件有超过 1 平方米交集，且结构底部与地形点相差不超过 3 m 时才作为可比较候选；共享或范围未知时仅保留 OSM 报值和候选证据。地形范围判断同样受采样误差限制。
原始 NUMABOVEGROUNDSTOREYS、TOPHEIGHT、BASEHEIGHT 保留原值。
空值表示无法可靠补齐；原因见 Height_Quality、Floors_Quality、Height_Source、Floors_Source。

Registered_GFA_sqm 仅使用没有已知组件范围冲突的来源报告登记面积，或有明确物理组件范围证据的许可面积；行政上的单结构单许可证不构成物理范围认证，报告值不表示已独立实测认证。
组件层数范围未解时，原始 LandsD_Registered_GFA 保留，但撤回派生登记面积采纳并标 registered_component_scope_unresolved；单独补层数不能重新认证面积。独立文档组件GFA仍按自身来源保留。
共享/多份许可证的面积不擅自分摊；Permit_Group 等组级证据保存在阶段数据。
GFA_Proxy_sqm 使用几何面积×报告层数，假设每层同样大小；GFA_Best_Available_sqm 优先登记面积，其次明确组件范围的文档GFA，再次代理面积。文档GFA不冒充注册面积；可出租、可售、单位内部面积不当作GFA。
比较总量必须同时报告覆盖率；未知值增加不是香港实体建筑面积减少。

UBEM_Mixed_Proportions 不再混入 POI 数量或模型概率；单功能的 1.0 是明确标记的 single_function_assumption。
混合建筑无可靠子类面积时比例为空；本轮没有采纳独立组件当前用途面积实测证据，Observed_Main_Area_Proportions 及两个 Observed_*_Area_Share 留空。
Functional_Subclasses 与 Evidence_Subclass_Weights 只表示功能及分类证据，不是实测面积份额。
许可DOMESTIC/NONDOMESTIC原账目保留作溯源，不再映射为当前用途实测份额、不据此强制Mixed-use；用途推断和单功能1.0假定有独立的来源及质量字段。
法定图则中源标签字面值 NULL 规范化为真正空值，其 Nullah 描述仍保留；原始字面值保存在冻结区划输入中，不生成名为 NULL 的用途类别。
法律管辖类别“合法私人建筑”、政府产权、OSM building=yes 不再直接决定功能。
小于 5 m² 且缺少其他资料，不再足以把足迹判为噪声。
已核实的青马、汀九、深圳湾、昂船洲和汲水门桥塔，以及严格匹配的显式基础设施多边形，归为非评估结构；保留结构高度，楼宇层数及能源面积代理留空，原报告值保留。
高度超过 100 m、占地小于 150 m² 而又无占用楼宇功能证据的未知结构，不允许仅靠模型补成楼宇用途；原始高程及结构高度仍保留，不凭细长形状判成确定的非建筑。
源基础设施筛查不会把所有 tower 标签当作非建筑，宗教/观景塔及泵站不因此自动排除。结构用途来源与空间识别证据逐条记录。
屋宇署“住宅/综合用途”组合标签不再被当成确定的纯住宅；政府建筑与房委会裙楼需具体功能证据。
屋宇署点位只有唯一落在当前足迹内且未命中明确源内冲突或组件身份隔离时，其分类及楼龄年份才可能进入公开属性；inside_unique本身不是事实认证。稳定Block ID、原匹配质量、隔离候选值与证据路径保留，快照坐标变化单独标记，不自动采新点或猜坐标；独立地政署许可外键链不被另一坏屋宇署点自动否定。
功能来源冲突不再自动变成混合用途；未能判定时保留未知并标记。

分类模型使用几何、分区和来源明确的物理特征；缺失值只在训练分区拟合插补器，不回写公开物理字段。
移除带标签的邻居主类特征，使用 2 km 空间分组验证。主类概率表示互斥的主导功能类别，不能识别任意混合比例。
训练标签为来源推断的弱监督标签，model_validation.json 的分数是对这些标签的空间留出一致性，不是新完成的独立实测精度。
旧验证表只有旧预测“正确/错误”标记，没有独立正确类别及实测高度，不能据此自动计算新版独立准确率。
低于阈值的主类/子类保留未知；原有明确的混合及非评估主类受到保护。
主类/子类还需在空间留出集中有至少 30 条超过置信阈值的预测，且对来源标签的精确率至少 0.80，才允许模型补齐该类别。
主类置信门槛先在外层训练分区内另切空间组校准：在 0.70/0.80/0.90/0.95/0.99 中选择达到至少 30 条、来源标签精确率至少 0.90 的最低门槛；无法达标的主类禁用自动补齐。校准不使用外层留出标签，具体分区和每类门槛随结果保存。
此门槛使用留出结果筛选可发布类别，不把同一结果冒充筛选后独立测试精度；不达标的预测不回退到另一类别，直接保留未知。
用途来源存在尚未解决的冲突时，模型置信度再高也不补成确定主类/子类；模型概率仍保留供检查。
组件高度与整楼报数已标记低高度/高层数范围未决时，不仅凭模型补齐该组件主用途或细子类；候选和概率保留，已有可靠来源用途保留。同名主塔用途不向低矮附属组件自动继承。

多模态与 LLM 原有结果完整保留为历史证据。旧缓存缺少输入/提示词/图像指纹，未冒充 v2 新验证结果。
默认重现不调用 API，不下载新影像；待核验任务见中间目录和 unresolved_building_review_queue.csv。
Validation_Materials.zip 中的 200 张旧图可恢复；其余原图无法保证与未来下载的影像时间版本相同。
新请求逐条保存输入、提示词/模型及图像哈希；成功结果与短暂文件路径分离，可按内容断点续跑。

物理数值冲突筛查：同口径高度跨来源差异 > max(10 m, 20%)；报告层数差异 > max(1 层, 10%)。
结构本体高度/报告层数小于 1 m 或大于 20 m 时，层数范围可能不对应当前结构，融合层数及其代理面积留空；原报告值保留。这是保守复核约定，不是自动纠错真值。
地形估计只接受 2–600 m 正差；不可接受差值明确标记。以上为质量控制约定，不是实测误差界限。
足迹 <150 m² 且高度 >150 m 等阈值用于人工复核，不自动判为错误或删除。
结构本体高度 <2 m、底部高程与地形点相差 >3 m 也单列复核；薄结构、裙楼托起的塔楼和地形采样偏差均可能触发，不据此自动删除高度。
高度/报告层数不在 2–8 m、登记面积与均匀楼板代理相差超过 3 倍，也只作为复核线索。
OSM 尺寸匹配要求双方覆盖至少 60%、交并比至少 50%；共享对象和近似等优冲突另作标记。
dimension_qa_summary.json 汇总这些复核条件及新版 15 m 数量；Physical_Review_Flags 保留逐栋原因。
独立验证模板包括已标记和未标记建筑，尚待人工与外部资料核实；本轮没有伪造独立真值。

GeoParquet 坐标：EPSG:2326，平面单位米；几何面积及全部高度、面积属性仍分别使用米和平方米。
GeoJSON 按 RFC 7946 导出为 WGS84 经度、纬度（EPSG:4326），坐标保留9位小数；不包含旧式 crs 成员。重投影只改变几何坐标表达，不改变建筑属性。面积计算请使用 GeoParquet 的米制几何，不要直接以经纬度计算平方米。
数据源参考：https://data.gov.hk/en-data/dataset/hk-landsd-openmap-landsd-building
高程基准：https://www.geodetic.gov.hk/common/data/pdf/explanatorynotes_c.pdf
OSM 标签定义：https://wiki.openstreetmap.org/wiki/Key:height
数据归属与许可继续遵循各提供方及 OpenStreetMap 的条款，源版本与校验见复现清单。
'''
    (Path(OUTPUT_DIR)/'RELEASE_NOTES_zh.txt').write_text(text,encoding='utf-8')


def main():
    frame = attach_geometry(read_table(ML_FINAL_OUTPUT_PATH),str(Path(ML_FINAL_OUTPUT_PATH).with_suffix('.geojson')))
    validation = validate_release(frame)
    public = generate_public_dataset(frame)
    old,new,comparison = compare_releases(public)
    create_review_tables(public)
    make_figures(old,new)
    write_documentation(public,validation,comparison)
    (Path(OUTPUT_DIR)/'release_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    files = [PUBLIC_DATASET_CSV_PATH,PUBLIC_DATASET_GEOJSON_PATH,str(Path(PUBLIC_DATASET_CSV_PATH).with_suffix('.parquet'))]
    manifest = {Path(path).name:{'bytes':Path(path).stat().st_size,'sha256':file_hash(path)} for path in files}
    (Path(OUTPUT_DIR)/'release_checksums.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'[INFO] Step 10: v2 release and complete old/new comparison generated for {len(public)} buildings')
    return True


if __name__ == '__main__':
    main()
