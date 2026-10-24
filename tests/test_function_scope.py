import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from classification import UNKNOWN,RESIDENTIAL,COMMERCIAL
from utils import init_keyword_tool

def load(number):
    spec=importlib.util.spec_from_file_location('scope_stage_'+str(number),next(SRC.glob(str(number)+'.*.py')))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

rule,model=load(3),load(9)

class Fixed:
    def __init__(self,classes,probabilities):
        self.classes_=np.array(classes)
        self.probabilities=probabilities
        self.named_steps={'classifier':self}
    def predict_proba(self,X):
        return np.tile(self.probabilities,(len(X),1))

class FunctionScopeTests(unittest.TestCase):
    def classify(self,**changes):
        row={'BUILDINGSTRUCTUREID':'1','osmid':'1','CATEGORY':'9','BUILDINGSTRUCTURETYPE':'T',
             'SHAPE_Area':500.,'BUILDINGNAMEEN':'','BUILDINGNAMETC':'','name':'Example tenant',
             'amenity':'restaurant','osm_area':0.,'overlap_ratio':1.}
        return rule.classify_building_rule_engine_optimized(pd.DataFrame([{**row,**changes}]),init_keyword_tool())

    def test_restaurant_tenant_cannot_label_entire_tower(self):
        result=self.classify()
        self.assertEqual(result.Final_Main_Class,'未知类别')
        self.assertIn('Example tenant[restaurant]',result.Matched_POIs)

    def test_construction_tag_is_not_nonmanufacturing_function(self):
        result=self.classify(amenity=np.nan,building='construction',osm_area=500.)
        self.assertEqual(result.Final_Main_Class,'未知类别')

    def test_arbitrary_poi_name_is_not_native_building_name(self):
        result=self.classify(name='Example Factory',amenity=np.nan)
        self.assertEqual(result.Final_Main_Class,'未知类别')

    def test_linked_access_description_cannot_turn_building_into_bridge(self):
        result=self.classify(BUILDINGNAMEEN='AIA Central',ALL_INFO_DESC='Commercial, Footbridge')
        self.assertEqual(result.Final_Main_Class,'未知类别')

    def test_factory_name_keyword_requires_an_ascii_token_boundary(self):
        for name in ['SATISFACTORY', 'SATISFACTORY COURT']:
            with self.subTest(name=name):
                result=self.classify(BUILDINGNAMEEN=name,BUILDINGNAMETC='',amenity=np.nan,building=np.nan,name='')
                self.assertEqual(result.Final_Main_Class,'未知类别')
        valid=self.classify(BUILDINGNAMEEN='Example Factory',BUILDINGNAMETC='',amenity=np.nan,building=np.nan,name='')
        self.assertEqual(valid.Final_Main_Class,'工业类别')

    def test_partial_office_substring_does_not_boost_an_independent_retail_keyword(self):
        result=self.classify(BUILDINGNAMEEN='Example Retail Officers',BUILDINGNAMETC='',
            amenity=np.nan,building=np.nan,name='')
        self.assertEqual(result.Final_Main_Class,'商业类别')
        self.assertIn('商业类别(50)',result.Evidence_Details)
        valid=self.classify(BUILDINGNAMEEN='Example Office',BUILDINGNAMETC='',
            amenity=np.nan,building=np.nan,name='')
        self.assertIn('商业类别(80)',valid.Evidence_Details)

    def test_native_school_name_still_supplies_building_evidence(self):
        result=self.classify(BUILDINGNAMEEN='Example School')
        self.assertEqual(result.Final_Main_Class,'商业类别')
        self.assertIn('教育',result.Final_Sub_Class)

    def test_generic_court_name_does_not_invent_judicial_or_commercial_use(self):
        for name in ['Scenery Court','Golden Court','Cheron Court','PO CHI COURT',
                     'Example Tennis Court','Example Sports Court']:
            with self.subTest(name=name):
                result=self.classify(BUILDINGNAMEEN=name,BUILDINGNAMETC='',amenity=np.nan,building=np.nan,name='')
                self.assertEqual(result.Final_Main_Class,'未知类别')
                self.assertEqual(result.Final_Sub_Class,'未知类别')

    def test_explicit_court_house_and_legal_court_terms_remain_recognized(self):
        for name in ['Example Court House','Example High Court','Example District Court',
                     'Example Court of Final Appeal','示例法院','示例法庭']:
            with self.subTest(name=name):
                result=self.classify(BUILDINGNAMEEN=name,BUILDINGNAMETC='',amenity=np.nan,building=np.nan,name='')
                self.assertEqual(result.Final_Main_Class,'商业类别')
                self.assertEqual(result.Final_Sub_Class,'其他商业（Other Commercial）')

    def test_coroner_and_magistrate_court_possessives_are_recognized(self):
        for name in ["Example Coroner's Court", "Example Magistrate's Court",
                     "Example Coroners' Court", "Example Magistrates' Court",
                     "Example Coroner’s Court", "Example Magistrate’s Court",
                     "Example Coroners’ Court", "Example Magistrates’ Court"]:
            with self.subTest(name=name):
                result=self.classify(BUILDINGNAMEEN=name,BUILDINGNAMETC='',amenity=np.nan,building=np.nan,name='')
                self.assertEqual(result.Final_Main_Class,'商业类别')
                self.assertEqual(result.Final_Sub_Class,'其他商业（Other Commercial）')

    def test_food_court_and_independent_function_evidence_are_preserved(self):
        food=self.classify(BUILDINGNAMEEN='Example Food Court',BUILDINGNAMETC='',amenity=np.nan,building=np.nan,name='')
        self.assertEqual(food.Final_Main_Class,'商业类别')
        self.assertEqual(food.Final_Sub_Class,'食肆（Restaurant）')
        office=self.classify(BUILDINGNAMEEN='Example Office',BUILDINGNAMETC='',amenity=np.nan,building=np.nan,name='')
        self.assertEqual(office.Final_Main_Class,'商业类别')
        self.assertIn('办公室',office.Final_Sub_Class)
        residential=self.classify(BUILDINGNAMEEN='Golden Court',BUILDINGNAMETC='',amenity=np.nan,building=np.nan,name='',
                                  BDBIAR_Match_Quality='inside_unique',BDBIAR_CLASS='住宅')
        self.assertEqual(residential.Final_Main_Class,'住宅类别')

    def test_housing_authority_category_alone_does_not_establish_use_or_tenure(self):
        result=self.classify(CATEGORY='3',amenity=np.nan,name='')
        self.assertEqual(result.Final_Main_Class,'未知类别')
        self.assertEqual(result.Final_Sub_Class,'未知类别')

    def test_generic_residential_point_class_does_not_establish_private_tenure(self):
        result=self.classify(amenity=np.nan,name='',BDBIAR_Match_Quality='inside_unique',BDBIAR_CLASS='住宅')
        self.assertEqual(result.Final_Main_Class,'住宅类别')
        self.assertEqual(result.Final_Sub_Class,'未知类别')

    def component(self):
        return pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'BUILDINGCSUID':['csuid-1'],
            'CSDI_Match_Quality':['same_component_id_type_footprint'],'CSDI_Structure_Type':['Tower'],
            'Final_Main_Class':[UNKNOWN],'Final_Sub_Class':[UNKNOWN],
            'Classification_Source':['Unknown'],'Classification_QA':['function_evidence_conflict'],
            'Is_Conflicted':[True],'Estimated_Height':[197.4],'Estimated_Floors':[38.],
            'GFA_Proxy_sqm':[58034.]})

    def facts(self,**changes):
        row={'evidence_id':'doc-1','building_id':'1','csuid':'csuid-1','component_scope':'tower_component',
             'source_url':'https://example.org/owner','main_class':COMMERCIAL,'sub_class':'Office_办公室'}
        return pd.DataFrame([{**row,**changes}])

    def test_documented_function_keeps_prior_conflict_and_dimensions(self):
        before=self.component()
        result=rule.apply_documented_functions(before,self.facts()).iloc[0]
        self.assertEqual(result.Final_Sub_Class,'Office_办公室')
        self.assertEqual(result.Classification_Source,'Documented_component_function')
        self.assertNotIn('function_evidence_conflict',result.Classification_QA)
        self.assertIn('function_evidence_conflict',json.loads(result.Function_Before_Document_Review)['Classification_QA'])
        self.assertEqual(result.Estimated_Height,197.4)
        self.assertEqual(result.Estimated_Floors,38.)
        self.assertEqual(result.GFA_Proxy_sqm,58034.)

    def test_documented_function_resolves_stale_scope_flag_and_enters_training(self):
        from importlib.util import spec_from_file_location, module_from_spec
        before=self.component()
        before['CATEGORY']='9'
        before['BDBIAR_Match_Quality']='unmatched'
        before['Classification_QA']='function_scope_ambiguous;function_evidence_conflict;subclass_parent_conflict'
        result=rule.apply_documented_functions(before,self.facts())
        row=result.iloc[0]
        self.assertNotIn('function_scope_ambiguous',row.Classification_QA)
        self.assertNotIn('function_evidence_conflict',row.Classification_QA)
        self.assertNotIn('subclass_parent_conflict',row.Classification_QA)
        prior=json.loads(row.Function_Before_Document_Review)
        self.assertEqual(prior['Classification_QA'],before.iloc[0].Classification_QA)

        training_spec=spec_from_file_location('scope_training_stage_8',SRC/'8.feature_engineering.py')
        training_module=module_from_spec(training_spec)
        training_spec.loader.exec_module(training_module)
        unresolved=result.copy()
        unresolved.loc[unresolved.index[0],'BUILDINGSTRUCTUREID']='2'
        unresolved.loc[unresolved.index[0],'Classification_Source']='OSM_Building_Function'
        unresolved.loc[unresolved.index[0],'Classification_QA']='registered_area_function_conflict'
        unresolved.loc[unresolved.index[0],'Final_Main_Class']=RESIDENTIAL
        permit_conflict=unresolved.copy()
        permit_conflict.loc[permit_conflict.index[0],'BUILDINGSTRUCTUREID']='3'
        permit_conflict.loc[permit_conflict.index[0],'Classification_QA']='permit_component_scope_unresolved'
        combined=pd.concat([result,unresolved,permit_conflict],ignore_index=True)
        training=training_module.prepare_training_data(combined)
        self.assertEqual(training.BUILDINGSTRUCTUREID.tolist(),['1','3'])
        self.assertEqual(training.loc[training.BUILDINGSTRUCTUREID.eq('3'),'Classification_QA'].iloc[0],
                         'permit_component_scope_unresolved')

    def test_documented_function_cannot_cross_ids_or_component_scope(self):
        for changes in [{'building_id':'2'},{'csuid':'other'},{'component_scope':'whole_development'},
                        {'sub_class':'Public Housing_公共房屋'}]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                rule.apply_documented_functions(self.component(),self.facts(**changes))

    def test_weak_geometry_match_cannot_accept_document(self):
        frame=self.component()
        frame['CSDI_Match_Quality']='footprint_changed'
        with self.assertRaisesRegex(ValueError,'matched tower'):
            rule.apply_documented_functions(frame,self.facts())

    def test_duplicate_component_documents_require_reconciliation(self):
        facts=pd.concat([self.facts(),self.facts(evidence_id='doc-2')],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'unique'):
            rule.apply_documented_functions(self.component(),facts)

    def test_documented_function_keeps_permit_accounting_separate_from_observed_use(self):
        frame=self.component()
        frame['GFA_DOMESTIC_SUM']=1000.
        frame['GFA_NONDOMESTIC_SUM']=0.
        result=rule.apply_documented_functions(frame,self.facts()).iloc[0]
        self.assertEqual(result.Final_Main_Class,COMMERCIAL)
        self.assertEqual(result.Final_Sub_Class,'Office_办公室')
        self.assertEqual(result.Classification_Source,'Documented_component_function')
        self.assertEqual(result.Proportion_Method,'single_function_assumption')
        self.assertEqual(result.Proportion_Quality,'assumed_not_measured')
        self.assertEqual(result.Sub_Class_Proportions,'Office_办公室:1.00000000')
        self.assertEqual(result.GFA_DOMESTIC_SUM,1000.)
        self.assertEqual(result.GFA_NONDOMESTIC_SUM,0.)
        self.assertTrue(pd.isna(result.Observed_Residential_Area_Share))
        self.assertTrue(pd.isna(result.Observed_Nondomestic_Area_Share))
        self.assertIn('permit_accounting_not_observed_use',result.Classification_QA)
        self.assertIn('permit_physical_scope_unverified',result.Classification_QA)
        self.assertNotIn('registered_area_function_conflict',result.Classification_QA)
        before=json.loads(result.Function_Before_Document_Review)
        self.assertIn('function_evidence_conflict',before['Classification_QA'])

    def test_high_probability_cannot_establish_housing_tenure(self):
        index=pd.Index(['1'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'Final_Main_Class':[RESIDENTIAL],
            'Final_Sub_Class':[UNKNOWN],'Classification_Source':['OSM_Building_Function']})
        package={'main':Fixed([RESIDENTIAL,COMMERCIAL,'Industrial_工业类别'],[.99,.005,.005]),
            'subclasses':{RESIDENTIAL:Fixed(['Public Housing_公共房屋','Private Housing_私人房屋'],[.99,.01])},
            'validated_subclasses':{RESIDENTIAL:['Public Housing_公共房屋']}}
        result=model.apply_ml_calibration(base,pd.DataFrame({'area':[500.]},index=index),package).iloc[0]
        self.assertEqual(result.Calibrated_Sub_Class,UNKNOWN)
        self.assertEqual(result.ML_Subclass_Candidate,'Public Housing_公共房屋')
        self.assertEqual(result.Subclass_Source,'housing_subtype_requires_documentary_evidence')
        self.assertEqual(result.Calibrated_Mix_Proportion,'')

    def test_source_model_disagreement_is_visible_without_overriding_source(self):
        index=pd.Index(['1'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'Final_Main_Class':[RESIDENTIAL],
            'Final_Sub_Class':[UNKNOWN],'Classification_Source':['Documented_component_function']})
        package={'main':Fixed([RESIDENTIAL,COMMERCIAL,'Industrial_工业类别'],[.005,.99,.005]),'subclasses':{}}
        result=model.apply_ml_calibration(base,pd.DataFrame({'area':[500.]},index=index),package).iloc[0]
        self.assertEqual(result.Calibrated_Main_Class,RESIDENTIAL)
        self.assertEqual(result.ML_Main_Class_Candidate,COMMERCIAL)
        self.assertIn('source_model_main_disagreement_review_only',result.Classification_QA)

    def test_documented_parent_with_multiple_unresolved_children_stays_unknown(self):
        index=pd.Index(['1'],name='BUILDINGSTRUCTUREID')
        base=pd.DataFrame({'BUILDINGSTRUCTUREID':['1'],'Final_Main_Class':[COMMERCIAL],
            'Final_Sub_Class':[UNKNOWN],'Classification_Source':['Documented_component_function']})
        package={'main':Fixed([RESIDENTIAL,COMMERCIAL,'Industrial_工业类别'],[.005,.99,.005]),
            'subclasses':{COMMERCIAL:Fixed(['Office_办公室','Retail_零售'],[.99,.01])},
            'validated_subclasses':{COMMERCIAL:['Office_办公室']}}
        result=model.apply_ml_calibration(base,pd.DataFrame({'area':[500.]},index=index),package).iloc[0]
        self.assertEqual(result.Calibrated_Sub_Class,UNKNOWN)
        self.assertEqual(result.ML_Subclass_Candidate,UNKNOWN)

if __name__=='__main__':unittest.main()
