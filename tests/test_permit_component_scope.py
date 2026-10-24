import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from classification import UNKNOWN, finalize_function_fields
from data_quality import assign_floor_area


def component():
    return pd.DataFrame({'SHAPE_Area':[5.587183314535984], 'Estimated_Floors':[np.nan],
        'Floors_Source':['component_storeys_scope_unresolved'], 'Permit_GFA_Scope':['single_structure'],
        'Permit_GFA_Usable':[True], 'GFA_DOMESTIC_SUM':[1960.5], 'GFA_NONDOMESTIC_SUM':[393.4]})


class PermitComponentScopeTests(unittest.TestCase):
    def test_residential_permit_area_does_not_belong_to_named_substation_by_one_link(self):
        frame=component()
        frame['Estimated_Floors']=1.;frame['Floors_Source']='LandsD_storeys'
        frame['CSDI_Component_Use_Type']='substation'
        row=assign_floor_area(frame).iloc[0]
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertEqual(row.Permit_GFA_Scope,'single_structure_component_use_scope_unresolved')
        self.assertEqual(row.GFA_DOMESTIC_SUM,1960.5)
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertEqual(row.GFA_Best_Available_sqm,frame.SHAPE_Area.iloc[0])
        self.assertEqual(row.GFA_Quality,'uniform_floorplate_proxy')

    def test_pure_nondomestic_account_requires_physical_scope_evidence(self):
        frame=component();frame['Estimated_Floors']=1.;frame['Floors_Source']='LandsD_storeys'
        frame['CSDI_Component_Use_Type']='substation';frame['GFA_DOMESTIC_SUM']=0.
        row=assign_floor_area(frame).iloc[0]
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertEqual(row.GFA_NONDOMESTIC_SUM,393.4)
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertEqual(row.GFA_Quality,'uniform_floorplate_proxy')
        frame['Permit_GFA_Physical_Scope_Verified']=True
        verified=assign_floor_area(frame).iloc[0]
        self.assertTrue(verified.Permit_GFA_Usable)
        self.assertEqual(verified.Registered_GFA_sqm,393.4)
        self.assertEqual(verified.GFA_Quality,'registered_scope_checked')

    def test_serialized_false_cannot_certify_permit_scope(self):
        frame=component();frame['Estimated_Floors']=1.;frame['Floors_Source']='LandsD_storeys'
        frame['Permit_GFA_Physical_Scope_Verified']='False'
        row=assign_floor_area(frame).iloc[0]
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))

    def test_aggregation_preserves_repaired_permit_flags_through_classification(self):
        src=Path(__file__).resolve().parents[1]/'src'
        spec=importlib.util.spec_from_file_location('permit_scope_aggregation',src/'2.osm_aggregation.py')
        stage=importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)
        official=gpd.GeoDataFrame(component().assign(BUILDINGSTRUCTUREID='1',BUILDINGCSUID='tower-1',
            BUILDINGSTRUCTURETYPE='T',TOPHEIGHT=np.nan,BASEHEIGHT=np.nan,NUMABOVEGROUNDSTOREYS=17),
            geometry=[box(835000,815000,835010,815010)],crs=2326).drop(columns=['Estimated_Floors','Floors_Source'])
        official['GROSSFLOORAREA']=43408.7
        csdi=gpd.GeoDataFrame({'BuildingCSUID':['tower-1'],'BaseHeight':[19.9],'TopHeight':[24.],
            'Storeys':[17.],'StoreysInBasement':[0.],'BuildingBlockType':['Tower'],'Status':['Active']},
            geometry=official.geometry,crs=2326)
        osm=gpd.GeoDataFrame({'osmid':[1]},geometry=[box(836000,816000,836010,816010)],crs=2326)
        with tempfile.TemporaryDirectory() as folder:
            metadata=Path(folder)/'metadata.json'; metadata.write_text(json.dumps({'sha256':'fixture'}))
            facts=Path(folder)/'facts.csv'
            pd.read_csv(src.parent/'ctl/building_attribute_evidence.csv',nrows=0).to_csv(facts,index=False)
            with patch('config.CSDI_METADATA_PATH',str(metadata)), patch('config.CSDI_BUILDINGS_PATH','fixture'), \
                 patch('config.TERRAIN_SAMPLES_PATH',str(Path(folder)/'missing.csv')), \
                 patch('config.DOCUMENTED_ATTRIBUTES_PATH',str(facts)), patch('ai_cache.file_hash',return_value='fixture'), \
                 patch('geopandas.read_parquet',return_value=csdi), patch.object(gpd.GeoDataFrame,'to_file'):
                frame=stage.spatial_aggregation(official,osm)
        self.assertFalse(frame.Permit_GFA_Usable.iloc[0])
        self.assertEqual(frame.Permit_GFA_Scope.iloc[0],'single_structure_component_scope_unresolved')
        self.assertEqual(frame.GROSSFLOORAREA.iloc[0],43408.7)
        self.assertTrue(pd.isna(frame.Registered_GFA_sqm.iloc[0]))
        self.assertTrue(pd.isna(frame.GFA_Best_Available_sqm.iloc[0]))
        self.assertEqual(frame.Registered_GFA_Source.iloc[0],
            'LandsD_structure_GFA_component_scope_unresolved')
        self.assertEqual(frame.GFA_Quality.iloc[0],'registered_component_scope_unresolved')
        self.assertFalse(any(c.endswith(('_x','_y')) for c in frame))
        frame['Final_Main_Class']=UNKNOWN; frame['Final_Sub_Class']=UNKNOWN
        row=finalize_function_fields(frame).iloc[0]
        self.assertEqual(row.Final_Main_Class,UNKNOWN)
        self.assertTrue(pd.isna(row.Observed_Residential_Area_Share))
        self.assertTrue(pd.isna(row.Observed_Nondomestic_Area_Share))

    def test_single_permit_link_does_not_resolve_incompatible_component_scope(self):
        row=assign_floor_area(component()).iloc[0]
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertEqual(row.Permit_GFA_Scope,'single_structure_component_scope_unresolved')
        self.assertEqual(row.GFA_DOMESTIC_SUM,1960.5)
        self.assertEqual(row.GFA_NONDOMESTIC_SUM,393.4)
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertTrue(pd.isna(row.GFA_Best_Available_sqm))
        self.assertEqual(row.GFA_Quality,'permit_component_scope_unresolved')

    def test_machine_layer_scope_precedes_height_count_reconciliation(self):
        src=Path(__file__).resolve().parents[1]/'src'
        spec=importlib.util.spec_from_file_location('machine_scope_aggregation',src/'2.osm_aggregation.py')
        stage=importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)
        official=gpd.GeoDataFrame({'BUILDINGSTRUCTUREID':['1'],'BUILDINGCSUID':['machine-1'],
            'BUILDINGSTRUCTURETYPE':['T'],'TOPHEIGHT':[5.6],'BASEHEIGHT':[0.],
            'NUMABOVEGROUNDSTOREYS':[3.],'SHAPE_Area':[100.],'GROSSFLOORAREA':[np.nan]},
            geometry=[box(835000,815000,835010,815010)],crs=2326)
        csdi=gpd.GeoDataFrame({'BuildingCSUID':['machine-1'],'BaseHeight':[0.],'TopHeight':[5.6],
            'Storeys':[3.],'StoreysInBasement':[0.],'BuildingBlockType':['Tower'],'Status':['Active']},
            geometry=official.geometry,crs=2326)
        osm=gpd.GeoDataFrame({'osmid':[1]},geometry=[box(836000,816000,836010,816010)],crs=2326)
        with tempfile.TemporaryDirectory() as folder:
            metadata=Path(folder)/'metadata.json'; metadata.write_text(json.dumps({'sha256':'fixture'}))
            facts=Path(folder)/'facts.csv'
            evidence=pd.read_csv(src.parent/'ctl/building_attribute_evidence.csv',nrows=0)
            evidence.loc[0]=dict(evidence_id='machine-reference',building_id='1',csuid='machine-1',
                attribute='reported_storeys',value='3',unit='storeys',
                scope='Reported parking machine layers are not established occupied building storeys',
                source_url='https://example.org/parking',source_date='2026',
                adoption='unresolved_component_storey_reference_conflict',notes='Machine tiers only')
            evidence.to_csv(facts,index=False)
            with patch('config.CSDI_METADATA_PATH',str(metadata)), patch('config.CSDI_BUILDINGS_PATH','fixture'), \
                 patch('config.TERRAIN_SAMPLES_PATH',str(Path(folder)/'missing.csv')), \
                 patch('config.DOCUMENTED_ATTRIBUTES_PATH',str(facts)), patch('ai_cache.file_hash',return_value='fixture'), \
                 patch('geopandas.read_parquet',return_value=csdi), patch.object(gpd.GeoDataFrame,'to_file'):
                frame=stage.spatial_aggregation(official,osm)
        row=frame.iloc[0]
        self.assertAlmostEqual(row.Estimated_Height,5.6)
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.GFA_Best_Available_sqm))
        self.assertEqual(row.Floors_Source,'documented_storey_reference_conflict_withheld')
        self.assertNotIn('Component_Vertical_Scope_QA',frame)

    def test_independent_structure_area_survives_unusable_permit(self):
        frame=component(); frame['Estimated_Floors']=1.; frame['Floors_Source']='LandsD_storeys'
        frame['CSDI_Component_Use_Type']='substation'
        frame['GROSSFLOORAREA']=1200.
        row=assign_floor_area(frame).iloc[0]
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertEqual(row.Permit_GFA_Scope,'single_structure_component_use_scope_unresolved')
        self.assertEqual(row.GFA_Best_Available_sqm,1200.)
        self.assertEqual(row.Registered_GFA_Source,'LandsD_structure_GFA')
        self.assertEqual(row.GFA_Quality,'reported_structure_gfa_not_independently_validated')

    def test_raw_registered_area_is_withheld_when_component_storey_scope_is_unresolved(self):
        frame=component(); frame['GROSSFLOORAREA']=43408.7
        row=assign_floor_area(frame).iloc[0]
        self.assertEqual(row.GROSSFLOORAREA,43408.7)
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertTrue(pd.isna(row.GFA_Best_Available_sqm))
        self.assertEqual(row.Registered_GFA_Source,'LandsD_structure_GFA_component_scope_unresolved')
        self.assertEqual(row.GFA_Quality,'registered_component_scope_unresolved')

    def test_count_repair_alone_cannot_release_unreconciled_permit_area(self):
        initial=component(); initial['GROSSFLOORAREA']=1200.
        frame=assign_floor_area(initial)
        frame['Estimated_Floors']=10.
        frame['Floors_Source']='Documented_component_storeys'
        row=assign_floor_area(frame).iloc[0]
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertEqual(row.GROSSFLOORAREA,1200.)
        self.assertEqual(row.Registered_GFA_Source,'LandsD_structure_GFA_component_scope_unresolved')
        self.assertEqual(row.GFA_Quality,'registered_component_scope_unresolved')
        self.assertEqual(row.GFA_Best_Available_sqm,row.GFA_Proxy_sqm)

    def test_reset_single_permit_scope_cannot_clear_prior_registered_area_warning(self):
        initial=component(); initial['GROSSFLOORAREA']=1200.
        frame=assign_floor_area(initial)
        frame['Estimated_Floors']=10.
        frame['Floors_Source']='Documented_component_storeys'
        frame['Permit_GFA_Scope']='single_structure'
        row=assign_floor_area(frame).iloc[0]
        self.assertEqual(row.GROSSFLOORAREA,1200.)
        self.assertFalse(row.Permit_GFA_Usable)
        self.assertEqual(row.Permit_GFA_Scope,'single_structure_component_scope_unresolved')
        self.assertTrue(pd.isna(row.Registered_GFA_sqm))
        self.assertEqual(row.Registered_GFA_Source,'LandsD_structure_GFA_component_scope_unresolved')
        self.assertEqual(row.GFA_Quality,'registered_component_scope_unresolved')

    def test_unusable_raw_permit_does_not_force_mixed_use_or_measured_shares(self):
        frame=assign_floor_area(component())
        frame['Final_Main_Class']=UNKNOWN
        frame['Final_Sub_Class']=UNKNOWN
        row=finalize_function_fields(frame).iloc[0]
        self.assertEqual(row.Final_Main_Class,UNKNOWN)
        self.assertTrue(pd.isna(row.Observed_Residential_Area_Share))
        self.assertTrue(pd.isna(row.Observed_Nondomestic_Area_Share))
        self.assertIn('permit_component_scope_unresolved',row.Classification_QA)

    def test_rule_engine_cannot_donate_unusable_permit_mixed_function(self):
        path=Path(__file__).resolve().parents[1]/'src/3.rule_engine.py'
        spec=importlib.util.spec_from_file_location('permit_scope_rule',path)
        rule=importlib.util.module_from_spec(spec); spec.loader.exec_module(rule)
        from utils import init_keyword_tool
        frame=assign_floor_area(component())
        for key,value in {'BUILDINGSTRUCTUREID':'1','osmid':np.nan,'CATEGORY':'9',
            'BUILDINGSTRUCTURETYPE':'T','BUILDINGNAMEEN':'','BUILDINGNAMETC':''}.items():
            frame[key]=value
        row=rule.classify_building_rule_engine_optimized(frame,init_keyword_tool())
        self.assertEqual(row.Final_Main_Class,'未知类别')


if __name__ == '__main__':
    unittest.main()
