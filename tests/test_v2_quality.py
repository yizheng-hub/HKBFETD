import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box, Point

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0,str(SRC))
from data_quality import ID, repair_footprints, permit_attributes, deduplicate_osm, parse_height, parse_storeys, fuse_dimensions, match_bdbiar_points, apply_documented_attributes
from classification import (UNKNOWN,MIXED,RESIDENTIAL,COMMERCIAL,NONASSESSED,INDUSTRIAL,
                            sub_label,parse_weights,format_weights,finalize_function_fields,building_tag_function,enrich_function_evidence)
from ai_cache import fingerprint,vision_checkpoint_matches,save_cache,read_cache


def load_stage(number):
    path = next(SRC.glob(str(number)+'.*.py'))
    spec = importlib.util.spec_from_file_location('v2_stage_'+str(number),path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def building(top=np.nan,base=np.nan,floors=np.nan):
    return gpd.GeoDataFrame({ID:['1'],'NUMABOVEGROUNDSTOREYS':[floors],'TOPHEIGHT':[top],
        'BASEHEIGHT':[base],'SHAPE_Area':[100.],'SHAPE_Length':[40.]},geometry=[box(835000,815000,835010,815010)],crs=2326)


def matches(height=np.nan,storeys=np.nan,osm_area=100.,overlap=1.):
    return pd.DataFrame({ID:['1'],'Source_Feature_ID':['osm:way:1:x'],'Source_Object_ID':['osm:way:1'],
        'osm_area':[osm_area],'overlap_ratio':[overlap],'SHAPE_Area':[100.],
        'osm_height':[height],'osm_storeys':[storeys]})


def csdi_building(top=30.,base=0.,storeys=np.nan,kind='Tower',geometry=None):
    return gpd.GeoDataFrame({'BuildingCSUID':['component-1'],'BaseHeight':[base],'TopHeight':[top],
        'Storeys':[storeys],'StoreysInBasement':[np.nan],'BuildingBlockType':[kind],
        'Status':['Active'],'height_m':[15.]},
        geometry=[geometry if geometry is not None else box(835000,815000,835010,815010)],crs=2326)


def component(top=np.nan,base=np.nan,floors=np.nan):
    frame=building(top,base,floors)
    frame['BUILDINGCSUID']='component-1'
    frame['BUILDINGSTRUCTURETYPE']='T'
    return frame


class PointOwnershipTests(unittest.TestCase):
    def test_nearby_reuse_does_not_disqualify_true_owner(self):
        buildings = gpd.GeoDataFrame({ID:['1','2']}, geometry=[box(0,0,10,10),box(12,0,22,10)], crs=2326)
        points = gpd.GeoDataFrame({'BDBIAR_OBJECTID':[5], 'year':[1995]}, geometry=[Point(5,5)], crs=2326)
        result = match_bdbiar_points(buildings,points).set_index(ID)
        self.assertEqual(result.loc['1','BDBIAR_Match_Quality'],'inside_unique')
        self.assertEqual(result.loc['2','BDBIAR_Match_Quality'],'nearby_only')
        self.assertEqual(result.loc['1','BDBIAR_Containing_Footprint_Count'],1)

    def test_all_contained_points_beat_name_based_nearby_candidate(self):
        buildings = gpd.GeoDataFrame({ID:['1']},geometry=[box(0,0,10,10)],crs=2326)
        points = gpd.GeoDataFrame({'BDBIAR_OBJECTID':[9,5], 'NSEARCH4_C':['平台','甲座'],
            'year':[1981,2020]},geometry=[Point(5,5),Point(11,5)],crs=2326)
        row=match_bdbiar_points(buildings,points).iloc[0]
        self.assertEqual(row.BDBIAR_OBJECTID,9)
        self.assertEqual(row.year,1981)
        self.assertEqual(row.BDBIAR_Match_Quality,'inside_unique')

    def test_multiple_contained_points_are_ambiguous_and_row_is_intact(self):
        buildings = gpd.GeoDataFrame({ID:['1']},geometry=[box(0,0,10,10)],crs=2326)
        points = gpd.GeoDataFrame({'BDBIAR_OBJECTID':[9,5], 'year':[1981,2020]},
            geometry=[Point(5,5),Point(6,5)],crs=2326)
        row=match_bdbiar_points(buildings,points).iloc[0]
        self.assertEqual(row.BDBIAR_Match_Quality,'ambiguous_nearest_points')
        self.assertEqual(row.BDBIAR_OBJECTID,5)
        self.assertEqual(row.year,2020)

    def test_overlap_point_is_shared_and_distant_point_is_unmatched(self):
        buildings = gpd.GeoDataFrame({ID:['1','2','3']},
            geometry=[box(0,0,10,10),box(2,0,12,10),box(100,0,110,10)],crs=2326)
        points = gpd.GeoDataFrame({'BDBIAR_OBJECTID':[5]},geometry=[Point(5,5)],crs=2326)
        result=match_bdbiar_points(buildings,points).set_index(ID)
        self.assertEqual(result.loc['1','BDBIAR_Match_Quality'],'shared_point')
        self.assertEqual(result.loc['2','BDBIAR_Match_Quality'],'shared_point')
        self.assertEqual(result.loc['3','BDBIAR_Match_Quality'],'unmatched')


class PhysicalValuesTests(unittest.TestCase):
    def test_optional_height_proxy_half_ties_are_stable_and_do_not_supply_gfa(self):
        for height,expected in [(4.5-1e-12,2),(4.5+1e-12,2),(7.5,3),(6.,2),(1.5,1)]:
            with self.subTest(height=height):
                row=fuse_dimensions(building(top=height,base=0.),matches()).iloc[0]
                self.assertEqual(row.Floors_Height_Proxy,expected)
                self.assertTrue(pd.isna(row.Estimated_Floors))
                self.assertTrue(pd.isna(row.GFA_Proxy_sqm))

    def test_high_floor_warehouse_is_flagged_downstream_not_rejected_as_conflicting(self):
        row=fuse_dimensions(building(floors=1),matches(height=12)).iloc[0]
        self.assertEqual(row.Estimated_Floors,1)
        self.assertNotEqual(row.Floors_Quality,'conflict_unresolved')

    def test_missing_dimensions_remain_missing(self):
        row=fuse_dimensions(building(),matches()).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))

    def test_unbased_top_is_never_a_relative_height(self):
        row=fuse_dimensions(building(top=957.4),matches()).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertEqual(row.TOPHEIGHT,957.4)

    def test_point_ground_is_separate_from_component_height(self):
        raw=building(top=957.4)
        p=raw.to_crs(4326).geometry.representative_point().iloc[0]
        terrain=pd.DataFrame({ID:['1'],'longitude':[p.x],'latitude':[p.y],'dtm_elevation_mpd':[950.]})
        row=fuse_dimensions(raw,matches(),terrain).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertAlmostEqual(row.Ground_To_Top_Height_m,7.4)
        self.assertEqual(row.Ground_Height_Quality,'terrain_point_estimate')
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.Floors_Height_Proxy))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))

    def test_terrain_at_wrong_location_is_rejected(self):
        terrain=pd.DataFrame({ID:['1'],'longitude':[0.],'latitude':[0.],'dtm_elevation_mpd':[950.]})
        self.assertTrue(pd.isna(fuse_dimensions(building(top=957.4),matches(),terrain).Estimated_Height.iloc[0]))

    def test_explicit_zero_base_is_valid_and_distinct_from_missing(self):
        self.assertEqual(fuse_dimensions(building(top=15.,base=0.),matches()).Estimated_Height.iloc[0],15.)
        self.assertTrue(pd.isna(fuse_dimensions(building(top=15.),matches()).Estimated_Height.iloc[0]))

    def test_no_floor_to_height_default_is_exported(self):
        row=fuse_dimensions(building(floors=5.),matches()).iloc[0]
        self.assertEqual(row.Estimated_Floors,5.)
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS,5.)

    def test_conflicting_reported_storeys_preserve_raw_and_withhold_fused(self):
        raw=building(top=20,base=0,floors=5.)
        p=raw.to_crs(4326).geometry.representative_point().iloc[0]
        terrain=pd.DataFrame({ID:['1'],'longitude':[p.x],'latitude':[p.y],'dtm_elevation_mpd':[0.]})
        row=fuse_dimensions(raw,matches(height=20.,storeys=78.),terrain).iloc[0]
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS,5.)
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.Floors_Quality,'conflict_unresolved')
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))

    def test_unknown_osm_reference_does_not_replace_component_height(self):
        row=fuse_dimensions(building(top=20.,base=5.),matches(height=100.)).iloc[0]
        self.assertEqual(row.Estimated_Height,15.)
        self.assertEqual(row.Ground_To_Top_Height_m,100.)
        self.assertIn('reference_may_differ',row.Height_Source_Comparison_QA)

    def test_small_inside_polygon_cannot_donate_whole_building_height(self):
        row=fuse_dimensions(building(),matches(height=200.,storeys=50.,osm_area=10.)).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))

    def test_units_and_uncertain_lists(self):
        self.assertAlmostEqual(parse_height("100 ft"),30.48)
        self.assertAlmostEqual(parse_height("10'6\""),3.2004)
        self.assertEqual(parse_height('15 m'),15.)
        for value in ['15;30','unknown','-9.3','957.4',float('inf')]:
            self.assertTrue(pd.isna(parse_height(value)),value)
        for value in ['2.5',0,-1,151]:
            self.assertTrue(pd.isna(parse_storeys(value)),value)

    def test_stored_area_is_replaced_by_projected_geometry(self):
        raw=building()
        raw['SHAPE_Area']=999.
        fixed=repair_footprints(raw)
        self.assertEqual(fixed.SHAPE_Area.iloc[0],100.)
        self.assertEqual(fixed.Source_Footprint_Area_sqm.iloc[0],999.)
        self.assertEqual(fixed.Footprint_Area_QA.iloc[0],'source_attribute_replaced_by_geometry')


class CSDIComponentTests(unittest.TestCase):
    def test_low_height_storeys_scope_withholds_height_and_keeps_source_candidate(self):
        raw=component(floors=17)
        p=raw.to_crs(4326).geometry.representative_point().iloc[0]
        terrain=pd.DataFrame({ID:['1'],'longitude':[p.x],'latitude':[p.y],'dtm_elevation_mpd':[5.]})
        frame=fuse_dimensions(raw,matches(),terrain,csdi_building(top=24.,base=19.9,storeys=17))
        evidence=pd.DataFrame(columns=pd.read_csv(SRC.parent/'ctl/building_attribute_evidence.csv',nrows=0).columns)
        row=apply_documented_attributes(frame,evidence).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertAlmostEqual(row.Height_Before_Document_Review,4.1)
        self.assertEqual(row.CSDI_Top_Elevation_mPD,24.)
        self.assertEqual(row.CSDI_Base_Elevation_mPD,19.9)
        self.assertEqual(row.CSDI_Reported_Storeys,17.)
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS,17.)
        self.assertEqual(row.Height_Quality,'conflict_unresolved')
        self.assertEqual(row.Height_Reference,'unresolved_component_scope')
        self.assertIn('height_storeys_component_scope_unresolved',row.Height_Source_Comparison_QA)
        self.assertEqual(row.Ground_To_Top_Height_m,19.)
        self.assertEqual(row.Ground_Height_Quality,'component_scope_unresolved_estimate')

    def test_bridge_pylon_height_is_not_rejected_due_to_one_reported_storey(self):
        row=fuse_dimensions(component(),matches(),csdi=csdi_building(top=200.,base=0.,storeys=1)).iloc[0]
        self.assertEqual(row.Estimated_Height,200.)
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.Height_Source,'CSDI_top_minus_component_base')

    def test_compatible_height_and_storeys_remain_available(self):
        row=fuse_dimensions(component(floors=16),matches(),csdi=csdi_building(top=60.,base=0.,storeys=16)).iloc[0]
        self.assertEqual(row.Estimated_Height,60.)
        self.assertEqual(row.Estimated_Floors,16.)

    def test_subtraction_noise_does_not_withhold_one_metre_per_reported_storey_height(self):
        row=fuse_dimensions(component(),matches(),csdi=csdi_building(top=31.,base=21.+1e-12,storeys=10)).iloc[0]
        self.assertAlmostEqual(row.Estimated_Height,10.)

    def test_original_elevations_used_instead_of_distributor_height(self):
        row=fuse_dimensions(component(),matches(),csdi=csdi_building(top=30,base=0,storeys=10)).iloc[0]
        self.assertEqual(row.Estimated_Height,30.)
        self.assertEqual(row.Height_Source,'CSDI_top_minus_component_base')
        self.assertEqual(row.Height_Reference,'structure_base_to_top')
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.CSDI_Reported_Storeys,10.)
        self.assertEqual(row.Floors_Source,'CSDI_storeys_vertical_scope_unverified')
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertTrue(pd.isna(row.Floors_Height_Proxy))
        self.assertTrue(pd.isna(row.BASEHEIGHT))

    def test_csdi_total_and_basement_do_not_imply_an_aboveground_count(self):
        source=csdi_building(top=40,base=0,storeys=12)
        source['StoreysInBasement']=2
        row=fuse_dimensions(component(),matches(),csdi=source).iloc[0]
        self.assertEqual(row.CSDI_Reported_Storeys,12.)
        self.assertEqual(row.CSDI_Basement_Storeys,2.)
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.Floors_Quality,'reported_vertical_scope_unverified')

    def test_primary_aboveground_count_also_controls_height_scope_check(self):
        row=fuse_dimensions(component(floors=10),matches(),csdi=csdi_building(top=9,base=0,storeys=9)).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.Floors_Source,'component_storeys_scope_unresolved')

    def test_component_source_names_remain_distinct_from_parent_names(self):
        source=csdi_building()
        source['BuildingNameEN']="Oi Shun House 'A' Substation"
        source['BuildingNameTC']='愛信樓Ａ變電站'
        raw=component(); raw['BUILDINGNAMEEN']='OI SHUN HSE'
        row=fuse_dimensions(raw,matches(),csdi=source).iloc[0]
        self.assertEqual(row.CSDI_Building_Name_EN,"Oi Shun House 'A' Substation")
        self.assertEqual(row.CSDI_Building_Name_TC,'愛信樓Ａ變電站')
        self.assertEqual(row.BUILDINGNAMEEN,'OI SHUN HSE')
        self.assertEqual(row.CSDI_Component_Use_Type,'substation')

    def test_raised_tower_base_is_distinct_from_ground(self):
        raw=component(top=479.73,floors=103)
        p=raw.to_crs(4326).geometry.representative_point().iloc[0]
        terrain=pd.DataFrame({ID:['1'],'longitude':[p.x],'latitude':[p.y],'dtm_elevation_mpd':[5.]})
        row=fuse_dimensions(raw,matches(height=484,storeys=118),terrain,csdi_building(top=490,base=29.4,storeys=103)).iloc[0]
        self.assertAlmostEqual(row.Estimated_Height,460.6)
        self.assertAlmostEqual(row.Ground_To_Top_Height_m,485.)
        self.assertAlmostEqual(row.Component_Base_Above_Terrain_m,24.4)
        self.assertEqual(row.Estimated_Floors,103.)
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS,103)
        self.assertEqual(row.OSM_Storeys,118.)
        self.assertEqual(row.OSM_Storeys_Scope_QA,'whole_building_scope_unverified_not_used_for_component')

    def test_unverified_whole_building_levels_cannot_supply_component_storeys(self):
        row=fuse_dimensions(component(),matches(storeys=12),csdi=csdi_building()).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.OSM_Storeys,12)
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))

    def test_missing_base_is_not_assumed_zero(self):
        row=fuse_dimensions(component(),matches(),csdi=csdi_building(base=np.nan)).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))

    def test_nonpositive_component_is_not_clipped_or_defaulted(self):
        for base in [30.,40.]:
            row=fuse_dimensions(component(),matches(),csdi=csdi_building(top=30.,base=base)).iloc[0]
            self.assertTrue(pd.isna(row.Estimated_Height))

    def test_changed_footprint_cannot_donate_dimensions(self):
        row=fuse_dimensions(component(),matches(),csdi=csdi_building(storeys=10,geometry=box(835020,815000,835030,815010))).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertEqual(row.CSDI_Match_Quality,'footprint_changed')
        self.assertEqual(row.CSDI_Reported_Storeys,10)

    def test_podium_tower_mismatch_is_rejected(self):
        row=fuse_dimensions(component(),matches(),csdi=csdi_building(kind='Podium')).iloc[0]
        self.assertEqual(row.CSDI_Match_Quality,'structure_type_mismatch')
        self.assertTrue(pd.isna(row.Estimated_Height))

    def test_open_sided_structure_is_compatible_with_generic_old_T(self):
        row=fuse_dimensions(component(),matches(),csdi=csdi_building(kind='Open-sided Structure')).iloc[0]
        self.assertEqual(row.Estimated_Height,30.)

    def test_unmatched_component_id_is_not_nearest_matched(self):
        native=csdi_building()
        native.BuildingCSUID='different-component'
        row=fuse_dimensions(component(),matches(),csdi=native).iloc[0]
        self.assertEqual(row.CSDI_Match_Quality,'id_unmatched')
        self.assertTrue(pd.isna(row.Estimated_Height))

    def test_duplicate_component_ids_fail(self):
        native=csdi_building()
        with self.assertRaises(ValueError):
            fuse_dimensions(component(),matches(),csdi=pd.concat([native,native]))

    def test_extreme_height_storeys_scope_is_withheld_without_guessing(self):
        row=fuse_dimensions(component(floors=5),matches(),csdi=csdi_building(top=380.7,base=3.9)).iloc[0]
        self.assertAlmostEqual(row.Estimated_Height,376.8)
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertEqual(row.Floors_Source,'component_storeys_scope_unresolved')
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS,5)

    def test_same_reference_height_conflict_is_withheld(self):
        row=fuse_dimensions(component(top=100,base=0),matches(),csdi=csdi_building(top=30,base=0)).iloc[0]
        self.assertTrue(pd.isna(row.Estimated_Height))
        self.assertEqual(row.Height_Quality,'conflict_unresolved')


class PermitAndFusionTests(unittest.TestCase):
    def test_shared_permit_is_not_assigned_to_each_building(self):
        permits=pd.DataFrame({'OPNO':['shared','single'],'DOMESTICGFA':[100.,20.],'NONDOMESTICGFA':[50.,0.]})
        links=pd.DataFrame({ID:['1','2','3'],'OPNO':['shared','shared','single']})
        result=permit_attributes(permits,links).set_index(ID)
        self.assertTrue(result.loc[['1','2'],'GFA_DOMESTIC_SUM'].isna().all())
        self.assertEqual(result.loc['3','GFA_DOMESTIC_SUM'],20.)
        self.assertEqual(result.loc['1','Permit_Group_Domestic_GFA'],100.)

    def test_missing_permit_component_is_not_assumed_zero(self):
        p=pd.DataFrame({'OPNO':['x'],'DOMESTICGFA':[100.],'NONDOMESTICGFA':[np.nan]})
        result=permit_attributes(p,pd.DataFrame({ID:['1'],'OPNO':['x']}))
        self.assertFalse(result.Permit_GFA_Usable.iloc[0])

    def test_duplicate_permit_links_fail(self):
        p=pd.DataFrame({'OPNO':['x'],'DOMESTICGFA':[100.],'NONDOMESTICGFA':[0.]})
        with self.assertRaises(ValueError):
            permit_attributes(p,pd.DataFrame({ID:['1','1'],'OPNO':['x','x']}))

    def test_duplicate_osm_objects_merge_missing_attributes(self):
        frame=gpd.GeoDataFrame({'osmid':['123','123'],'osm_type':['way','way'],'height':[np.nan,'30'],
            'building:levels':['10',np.nan]},geometry=[box(0,0,10,10),box(0,0,10,10)],crs=2326)
        result=deduplicate_osm(frame)
        self.assertEqual(len(result),1)
        self.assertEqual(parse_height(result.height.iloc[0]),30.)
        self.assertEqual(parse_storeys(result['building:levels'].iloc[0]),10.)

    def test_conflicting_duplicate_heights_are_retained_as_conflict(self):
        f=gpd.GeoDataFrame({'osmid':['1','1'],'osm_type':['way','way'],'height':['10','50']},geometry=[box(0,0,1,1)]*2,crs=2326)
        r=deduplicate_osm(f)
        self.assertTrue(pd.isna(parse_height(r.height.iloc[0])))
        self.assertIn('height',r.OSM_Attribute_Conflicts.iloc[0])

    def test_osm_object_types_are_part_of_identity(self):
        f=gpd.GeoDataFrame({'osmid':['1','1'],'osm_type':['way','node']},geometry=[Point(0,0)]*2,crs=2326)
        self.assertEqual(len(deduplicate_osm(f)),2)


class ClassificationAndResumeTests(unittest.TestCase):
    def test_every_release_field_has_an_explicit_dictionary_entry(self):
        stage=load_stage(10)
        columns=list(stage.EXPORT_MAPPING.values())+stage.PROVENANCE+[
            'Dataset_Version','Unknown_Value_Policy','Physical_Review_Flags','Small_Tall_Resolution']
        public=pd.DataFrame({column:pd.Series(dtype='object') for column in columns})
        with tempfile.TemporaryDirectory() as temp:
            stage.OUTPUT_DIR=temp
            stage.write_documentation(public,{'unknown_height_records':0,'unknown_floor_records':0},
                {'new_height_known':0,'new_floor_known':0})
            dictionary=pd.read_csv(Path(temp)/'field_dictionary.csv')
            self.assertEqual(set(dictionary.field),set(public.columns))
            self.assertTrue(dictionary.meaning.notna().all())

    def test_proportion_comparison_ignores_decimal_formatting(self):
        stage=load_stage(10)
        self.assertTrue(stage.proportion_values_equal('Office_办公室:1.00','Office_办公室:1.00000000'))
        self.assertFalse(stage.proportion_values_equal('Office_办公室:0.8-Retail_零售:0.2','Office_办公室:1.00000000'))
        self.assertTrue(stage.proportion_values_equal(np.nan,''))
        self.assertFalse(stage.proportion_values_equal('unparseable',''))

    def test_all_field_comparison_includes_withheld_point_years(self):
        stage=load_stage(10)
        field='UBEM_Mixed_Proportions_混合比例'
        old=pd.DataFrame({'age':[2000.,1999.],field:['Office_办公室:1.0','']},index=['1','2'])
        new=pd.DataFrame({'age':[np.nan,1999.],field:['Office_办公室:1.00000000','']},index=['1','2'])
        with tempfile.TemporaryDirectory() as temp:
            stage.OUTPUT_DIR=temp
            stage.compare_all_common_fields(old,new)
            summary=pd.read_csv(Path(temp)/'all_common_field_change_summary.csv').set_index('field')
            self.assertEqual(summary.loc['age','changed_records'],1)
            self.assertEqual(summary.loc[field,'changed_records'],0)
            rows=pd.read_parquet(Path(temp)/'all_common_field_changes.parquet')
            self.assertEqual(rows['Building_ID_建筑ID'].tolist(),['1'])
            self.assertTrue(rows.new_value.isna().all())

    def test_missing_poi_names_do_not_match_as_literal_nan(self):
        stage=load_stage(2)
        osm=gpd.GeoDataFrame({'osmid':['1'],'name':[np.nan],'amenity':['school'],'shop':[np.nan]},geometry=[Point(0,0)],crs=2326)
        overture=gpd.GeoDataFrame({'osmid':['a'],'name':[np.nan],'amenity':['restaurant'],'shop':[np.nan]},geometry=[Point(0,0)],crs=2326)
        result=stage.fuse_poi_datasets(osm,overture)
        self.assertEqual(set(result.osmid),{'1','a'})

    def test_generic_residential_tag_does_not_determine_housing_tenure(self):
        self.assertEqual(building_tag_function({'building':'apartments'}),(RESIDENTIAL,UNKNOWN))
        base=pd.DataFrame({ID:['1'],'Final_Main_Class':[RESIDENTIAL],'Final_Sub_Class':['Public Housing_公共房屋'],
            'Classification_Source':['LandsD_Category'],'Sub_Class_Proportions':['']})
        joined=pd.DataFrame({ID:['1'],'Source_Object_ID':['osm:way:1'],'osm_area':[100.],
            'SHAPE_Area':[100.],'overlap_ratio':[1.],'building':['apartments']})
        result=enrich_function_evidence(base,joined).iloc[0]
        self.assertEqual(result.Final_Sub_Class,'Public Housing_公共房屋')

    def test_generic_names_and_residential_forms_do_not_invent_function_or_tenure(self):
        from utils import init_keyword_tool,classify_text_by_keywords
        tool=init_keyword_tool()
        kws,classes=classify_text_by_keywords('Example Company Garden Estate',tool['keyword_regex'],tool['keyword_map'])
        self.assertEqual(classes,[])
        kws,classes=classify_text_by_keywords('示例大廈 Example Plaza',tool['keyword_regex'],tool['keyword_map'])
        self.assertEqual(classes,[])
        kws,classes=classify_text_by_keywords('Example Commercial Building',tool['keyword_regex'],tool['keyword_map'])
        self.assertEqual(classes,[('商业类别','未知类别')])
        kws,classes=classify_text_by_keywords('Example Apartment',tool['keyword_regex'],tool['keyword_map'])
        self.assertEqual(classes,[('住宅类别','未知类别')])
        stage=load_stage(3)
        base=pd.DataFrame({ID:['1'],'osmid':['1'],'building':['apartments'],'CATEGORY':['9'],
            'BUILDINGSTRUCTURETYPE':['T'],'SHAPE_Area':[100.]})
        self.assertEqual(stage.classify_building_rule_engine_optimized(base,tool).Final_Sub_Class,'未知类别')

    def test_large_site_polygon_does_not_donate_whole_building_function(self):
        base=pd.DataFrame({ID:['1'],'Final_Main_Class':[UNKNOWN],'Final_Sub_Class':[UNKNOWN],
            'Classification_Source':['Unknown'],'Sub_Class_Proportions':['']})
        campus=pd.DataFrame({ID:['1'],'Source_Object_ID':['osm:way:1'],'osm_area':[10000.],
            'SHAPE_Area':[100.],'overlap_ratio':[.01],'building':['office']})
        result=enrich_function_evidence(base,campus).iloc[0]
        self.assertEqual(result.Final_Main_Class,UNKNOWN)
    def test_generic_source_categories_and_building_yes_do_not_imply_housing(self):
        stage=load_stage(3)
        from utils import init_keyword_tool, classify_from_bdbiar_final
        tool=init_keyword_tool()
        for category,structure,osm in [('1','T',np.nan),('4','T',np.nan),('9','T','1'),('3','P',np.nan)]:
            row=pd.DataFrame({ID:['1'],'osmid':[osm],'building':['yes'],'CATEGORY':[category],
                'BUILDINGSTRUCTURETYPE':[structure],'SHAPE_Area':[100.],
                'BDBIAR_Match_Quality':['inside_unique'],'BDBIAR_CLASS':['住宅/綜合用途']})
            with self.subTest(category=category,structure=structure):
                self.assertEqual(stage.classify_building_rule_engine_optimized(row,tool).Final_Main_Class,'未知类别')
        self.assertEqual(classify_from_bdbiar_final('住宅/綜合用途'),(None,None))

    def test_nonresidential_registered_area_does_not_assume_commercial(self):
        stage=load_stage(3)
        from utils import init_keyword_tool
        row=pd.DataFrame({ID:['1'],'osmid':[np.nan],'CATEGORY':['1'],'BUILDINGSTRUCTURETYPE':['T'],
            'SHAPE_Area':[100.],'GFA_DOMESTIC_SUM':[0.],'GFA_NONDOMESTIC_SUM':[1000.]})
        self.assertEqual(stage.classify_building_rule_engine_optimized(row,init_keyword_tool()).Final_Main_Class,'未知类别')

    def test_conflicting_function_sources_do_not_create_mixed_use(self):
        stage=load_stage(3)
        from utils import init_keyword_tool
        row=pd.DataFrame({ID:['1'],'osmid':[np.nan],'CATEGORY':['2'],'BUILDINGSTRUCTURETYPE':['T'],
            'SHAPE_Area':[100.],'BDBIAR_Match_Quality':['inside_unique'],'BDBIAR_CLASS':['工業']})
        result=stage.classify_building_rule_engine_optimized(row,init_keyword_tool())
        self.assertEqual(result.Final_Main_Class,'未知类别')
        self.assertTrue(result.Is_Conflicted)
        self.assertFalse(result.Is_Mixed_Use)

    def test_nonmanufacturing_hyphen_is_preserved(self):
        self.assertEqual(sub_label('Non-manufacturing_非制造业'),'Non-manufacturing_非制造业')
        self.assertEqual(parse_weights('Non-manufacturing_非制造业:0.6-Office_办公室:0.4'),
                         {'Non-manufacturing_非制造业':.6,'Office_办公室':.4})

    def test_filtered_evidence_weights_are_normalized(self):
        vector=parse_weights(format_weights({'a':.7,'b':.1}))
        self.assertAlmostEqual(sum(vector.values()),1.)

    def test_mixed_poi_counts_are_not_area_shares(self):
        f=pd.DataFrame({'Final_Main_Class':[MIXED],'Final_Sub_Class':[UNKNOWN],
            'Sub_Class_Proportions':['Private Housing_私人房屋:0.8-Retail_零售:0.2']})
        result=finalize_function_fields(f).iloc[0]
        self.assertEqual(result.Sub_Class_Proportions,'')
        self.assertEqual(result.Proportion_Quality,'unknown')
        self.assertEqual(set(json.loads(result.Functional_Main_Classes)),{RESIDENTIAL,COMMERCIAL})

    def test_parent_conflict_does_not_leak_wrong_subclass(self):
        f=pd.DataFrame({'Final_Main_Class':[COMMERCIAL],'Final_Sub_Class':['Private Housing_私人房屋']})
        result=finalize_function_fields(f).iloc[0]
        self.assertEqual(result.Final_Sub_Class,UNKNOWN)
        self.assertIn('subclass_parent_conflict',result.Classification_QA)

    def test_classification_conflict_survives_later_stages(self):
        f=pd.DataFrame({'Final_Main_Class':[COMMERCIAL],'Final_Sub_Class':['Private Housing_私人房屋']})
        result=finalize_function_fields(finalize_function_fields(f)).iloc[0]
        self.assertIn('subclass_parent_conflict',result.Classification_QA)

    def test_permit_subclasses_are_not_measured_current_use_shares(self):
        f=pd.DataFrame({'Final_Main_Class':[RESIDENTIAL],'Final_Sub_Class':['Private Housing_私人房屋'],
            'GFA_DOMESTIC_SUM':[80.],'GFA_NONDOMESTIC_SUM':[20.]})
        result=finalize_function_fields(f).iloc[0]
        self.assertEqual(result.Final_Main_Class,RESIDENTIAL)
        self.assertEqual(result.GFA_DOMESTIC_SUM,80.)
        self.assertEqual(result.GFA_NONDOMESTIC_SUM,20.)
        self.assertTrue(pd.isna(result.Observed_Residential_Area_Share))
        self.assertTrue(pd.isna(result.Observed_Nondomestic_Area_Share))
        self.assertEqual(result.Observed_Main_Area_Proportions,'')
        self.assertEqual(result.Sub_Class_Proportions,'Private Housing_私人房屋:1.00000000')
        self.assertEqual(result.Proportion_Method,'single_function_assumption')
        self.assertEqual(result.Proportion_Quality,'assumed_not_measured')
        self.assertIn('permit_accounting_not_observed_use',result.Classification_QA)
        self.assertIn('permit_physical_scope_unverified',result.Classification_QA)
        self.assertNotIn('registered_nonresidential_function_unspecified',result.Classification_QA)
        self.assertNotIn('registered_area_function_conflict',result.Classification_QA)

    def test_permit_accounting_does_not_clear_independent_function_conflict(self):
        f=pd.DataFrame({'Final_Main_Class':[MIXED],'Final_Sub_Class':[UNKNOWN],
            'GFA_DOMESTIC_SUM':[80.],'GFA_NONDOMESTIC_SUM':[20.],
            'Evidence_Subclass_Weights':[''],'Classification_QA':['function_evidence_conflict']})
        first=finalize_function_fields(f)
        self.assertEqual(first.Final_Main_Class.iloc[0],MIXED)
        self.assertEqual(first.GFA_DOMESTIC_SUM.iloc[0],80.)
        self.assertEqual(first.GFA_NONDOMESTIC_SUM.iloc[0],20.)
        self.assertTrue(pd.isna(first.Observed_Residential_Area_Share.iloc[0]))
        self.assertTrue(pd.isna(first.Observed_Nondomestic_Area_Share.iloc[0]))
        self.assertIn('function_evidence_conflict',first.Classification_QA.iloc[0])
        self.assertIn('permit_accounting_not_observed_use',first.Classification_QA.iloc[0])
        self.assertNotIn('registered_nonresidential_function_unspecified',first.Classification_QA.iloc[0])
        first['Evidence_Subclass_Weights']='Office_办公室:1'
        row=finalize_function_fields(first).iloc[0]
        self.assertEqual(row.Final_Main_Class,MIXED)
        self.assertNotIn('registered_nonresidential_function_unspecified',row.Classification_QA)
        self.assertIn('function_evidence_conflict',row.Classification_QA)
        self.assertEqual(row.GFA_DOMESTIC_SUM,80.)
        self.assertEqual(row.GFA_NONDOMESTIC_SUM,20.)
        self.assertTrue(pd.isna(row.Observed_Residential_Area_Share))
        self.assertTrue(pd.isna(row.Observed_Nondomestic_Area_Share))
        self.assertEqual(parse_weights(row.Observed_Main_Area_Proportions),{})

    def test_successful_visual_checkpoint_survives_image_loss(self):
        context='context'
        hashes={'plain_sha256':'a'*64,'overlay_sha256':'b'*64}
        checkpoint={'Context_SHA256':context,'ai_decision':'keep',**hashes,
                    'Input_SHA256':fingerprint({'context':context,**hashes})}
        self.assertTrue(vision_checkpoint_matches(checkpoint,context,{}))
        self.assertTrue(vision_checkpoint_matches(checkpoint,context,hashes))
        self.assertFalse(vision_checkpoint_matches(checkpoint,'new_context',hashes))
        self.assertFalse(vision_checkpoint_matches(checkpoint,context,{'plain_sha256':'c'*64,'overlay_sha256':'b'*64}))
        self.assertFalse(vision_checkpoint_matches({'ai_decision':'keep'},context,{}))

    def test_visual_merge_does_not_inherit_a_conflicted_host_function(self):
        stage=load_stage(7)
        base=pd.DataFrame({ID:['1','2'],'Final_Main_Class':[UNKNOWN,COMMERCIAL],
            'Final_Sub_Class':[UNKNOWN,'Office_办公室'],'Classification_Source':['Unknown','source'],
            'Classification_QA':['','registered_area_function_conflict']})
        for col in ['NUMABOVEGROUNDSTOREYS','TOPHEIGHT','BASEHEIGHT','Estimated_Height','Estimated_Floors']:
            base[col]=np.nan
        decisions=pd.DataFrame({'fragment_id':['1'],'host_id':['2'],'ai_decision':['merge'],'Cache_Status':['matching_checkpoint']})
        row=stage.apply_correction_suggestions(base,decisions).set_index(ID).loc['1']
        self.assertEqual(row.Final_Main_Class,UNKNOWN)
        self.assertEqual(row.Geometry_Review_Status,'host_function_conflict_no_inheritance')

    def test_visual_merge_cannot_replace_substation_use_with_parent_housing(self):
        stage=load_stage(7)
        base=pd.DataFrame({ID:['1','2'],'Final_Main_Class':[COMMERCIAL,RESIDENTIAL],
            'Final_Sub_Class':['Other Commercial_其他商业','Private Housing_私人房屋'],
            'Classification_Source':['CSDI_Component_Name_Keyword','source'],
            'CSDI_Component_Use_Type':['substation','']})
        for col in ['NUMABOVEGROUNDSTOREYS','TOPHEIGHT','BASEHEIGHT','Estimated_Height','Estimated_Floors']:
            base[col]=np.nan
        decisions=pd.DataFrame({'fragment_id':['1'],'host_id':['2'],'ai_decision':['merge'],'Cache_Status':['matching_checkpoint']})
        row=stage.apply_correction_suggestions(base,decisions).set_index(ID).loc['1']
        self.assertEqual(row.Final_Main_Class,COMMERCIAL)
        self.assertEqual(row.Final_Sub_Class,'Other Commercial_其他商业')
        self.assertEqual(row.Geometry_Review_Status,'component_use_scope_conflict_no_inheritance')

    def test_atomic_checkpoint_and_empty_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache.csv'
            self.assertTrue(read_cache(path,['fragment_id']).empty)
            save_cache(pd.DataFrame({'fragment_id':['12'],'value':[1]}),path)
            self.assertEqual(read_cache(path,['fragment_id']).fragment_id.iloc[0],'12')
            self.assertFalse(list(Path(temp).glob('*.pending')))

    def test_ai_subset_merge_preserves_whole_stock_and_rejects_stale(self):
        stage=load_stage(5)
        base=pd.DataFrame({ID:['1','2'],'Final_Main_Class':[UNKNOWN,UNKNOWN],
            'Final_Sub_Class':[UNKNOWN,UNKNOWN],'Classification_Source':['Unknown','Unknown']})
        cache=pd.DataFrame({ID:['1'],'Input_SHA256':['stale'],'LLM_Main_Class':[COMMERCIAL],
            'LLM_Sub_Class':['Office_办公室'],'LLM_Confidence':[.99]})
        result=stage.merge_verified_results(base,cache,lambda row:'current')
        self.assertEqual(set(result[ID]),{'1','2'})
        self.assertTrue(result.Final_Main_Class.eq(UNKNOWN).all())
        cache.Input_SHA256='current'
        result=stage.merge_verified_results(base,cache,lambda row:'current')
        self.assertEqual(result.set_index(ID).loc['1','Final_Main_Class'],COMMERCIAL)
        self.assertEqual(result.set_index(ID).loc['2','Final_Main_Class'],UNKNOWN)

    def test_feature_matrix_does_not_use_class_labels(self):
        stage=load_stage(8)
        f=gpd.GeoDataFrame({ID:['1','2'],'OZP_ZONE_LABEL':['R','C'],'Estimated_Height':[np.nan,20.],
            'Estimated_Floors':[np.nan,5.],'Height_Quality':['unknown','reported'],
            'Final_Main_Class':[RESIDENTIAL,COMMERCIAL]},geometry=[box(0,0,10,10),box(20,0,30,10)],crs=2326)
        a,_=stage.compute_advanced_features(f)
        f.Final_Main_Class=[INDUSTRIAL,RESIDENTIAL]
        b,_=stage.compute_advanced_features(f)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(pd.isna(a.loc['1','Estimated_Height']))

    def test_model_physical_attributes_require_valid_point_and_component_matches(self):
        stage=load_stage(8)
        f=gpd.GeoDataFrame({ID:['1','2'],'OZP_ZONE_LABEL':['R','C'],'Estimated_Height':[10.,20.],
            'Estimated_Floors':[3.,5.],'BDBIAR_AGE':[1990.,2000.],
            'BDBIAR_Match_Quality':['nearest_outside','inside_unique'],
            'CSDI_Structure_Type':['Tower','Podium'],
            'CSDI_Match_Quality':['footprint_changed','same_component_id_type_footprint']},
            geometry=[box(0,0,10,10),box(20,0,30,10)],crs=2326)
        X,_=stage.compute_advanced_features(f)
        self.assertTrue(pd.isna(X.loc['1','Reported_Completion_Year']))
        self.assertEqual(X.loc['2','Reported_Completion_Year'],2000)
        self.assertEqual(X.loc['1','csdi_structure_type_4'],1)
        self.assertEqual(X.loc['2','csdi_structure_type_1'],1)


if __name__ == '__main__':
    unittest.main()
