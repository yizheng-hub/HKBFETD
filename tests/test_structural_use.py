import sys
from pathlib import Path
import unittest
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from structural_use import infrastructure_kind, apply_structural_use
from classification import RESIDENTIAL,NONASSESSED,UNKNOWN
from data_quality import fuse_dimensions
from test_v2_quality import component,matches


class StructuralUseTests(unittest.TestCase):
    def frame(self):
        f=fuse_dimensions(component(top=30,base=0,floors=10),matches())
        f['Final_Main_Class']=RESIDENTIAL
        f['Final_Sub_Class']='Private Housing_私人房屋'
        f['Classification_Source']='source'
        return f

    def osm(self,kind='bridge',geometry=None):
        return gpd.GeoDataFrame({'id':[1],'osm_type':['way'],'building':[kind],'man_made':[None],
            'tower:type':[None],'bridge:support':[None],'name':[None]},
            geometry=[geometry if geometry is not None else box(835000,815000,835010,815010)],crs=2326)

    def verified(self):
        return pd.DataFrame(columns=['BUILDINGCSUID','longitude','latitude','structure_use','landmark_name','evidence_url'])

    def test_bridge_is_nonassessed_and_never_contributes_proxy_floor_area(self):
        row=apply_structural_use(self.frame(),self.osm(),self.verified()).iloc[0]
        self.assertEqual(row.Final_Main_Class,NONASSESSED)
        self.assertEqual(row.Final_Sub_Class,'Transport Infrastructure_交通基础设施')
        self.assertEqual(row.Estimated_Height,30)
        self.assertTrue(pd.isna(row.Estimated_Floors))
        self.assertTrue(pd.isna(row.GFA_Proxy_sqm))
        self.assertEqual(row.NUMABOVEGROUNDSTOREYS,10)
        self.assertEqual(row.Pre_Structure_Review_Main_Class,RESIDENTIAL)

    def test_small_rooftop_equipment_does_not_reclassify_host_building(self):
        row=apply_structural_use(self.frame(),self.osm(geometry=box(835000,815000,835005,815005)),self.verified()).iloc[0]
        self.assertEqual(row.Final_Main_Class,RESIDENTIAL)
        self.assertEqual(row.Estimated_Floors,10)

    def test_generic_or_occupied_towers_are_not_automatically_equipment(self):
        for tower in [None,'pagoda','religious','observation','bell_tower']:
            self.assertIsNone(infrastructure_kind({'man_made':'tower','tower:type':tower}))
        self.assertIsNone(infrastructure_kind({'man_made':'storage_tank','name':'Fresh Water Pumping Station'}))
        self.assertEqual(infrastructure_kind({'man_made':'tower','tower:type':'communication'}),'equipment_structure')

    def test_verified_structure_requires_same_id_and_location(self):
        f=self.frame();p=f.geometry.centroid.to_crs(4326).iloc[0]
        verified=pd.DataFrame({'BUILDINGCSUID':['component-1'],'longitude':[p.x],'latitude':[p.y],
            'structure_use':['bridge_structure'],'landmark_name':['Test bridge'],'evidence_url':['https://example.org/factsheet']})
        row=apply_structural_use(f,self.osm().iloc[0:0],verified).iloc[0]
        self.assertEqual(row.Structure_Use_Name,'Test bridge')
        verified.longitude=0
        with self.assertRaises(ValueError):
            apply_structural_use(f,self.osm().iloc[0:0],verified)

    def test_shared_donor_is_not_assigned_to_two_structures(self):
        f=pd.concat([self.frame(),self.frame()],ignore_index=True)
        f.BUILDINGSTRUCTUREID=['1','2'];f.BUILDINGCSUID=['first','second']
        result=apply_structural_use(f,self.osm(),self.verified())
        self.assertTrue(result.Final_Main_Class.eq(RESIDENTIAL).all())


if __name__=='__main__':unittest.main()
