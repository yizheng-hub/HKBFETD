import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
import config
import run_pipeline
from ai_cache import file_hash,fingerprint
from classification import UNKNOWN,COMMERCIAL
from data_quality import ID


class ReproductionTests(unittest.TestCase):
    def test_source_null_zone_placeholder_keeps_nullah_description(self):
        from stage_io import read_table
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'zones.csv'
            path.write_text('BUILDINGSTRUCTUREID,OZP_ZONE_LABEL,OZP_DESC_ENG\n116642,NULL,Nullah\n1,R(A),Residential (Group A)\n',encoding='utf-8')
            table=read_table(path).set_index(ID)
            self.assertTrue(pd.isna(table.loc['116642','OZP_ZONE_LABEL']))
            self.assertEqual(table.loc['116642','OZP_DESC_ENG'],'Nullah')
            self.assertEqual(table.loc['1','OZP_ZONE_LABEL'],'R(A)')

    def test_environment_change_rejects_resume_without_overwriting_record(self):
        old={'python':'3.11','platform':'Windows','packages':{'numpy':'1'}}
        changed={**old,'packages':{'numpy':'2'}}
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            path=output/'environment_versions.json'
            path.write_text(json.dumps(old),encoding='utf-8')
            before=path.read_bytes()
            previous={'stages':{'1':{'signature':'old'}}}
            run_pipeline.validate_resume_environment(previous,output,old,True)
            with self.assertRaisesRegex(RuntimeError,'without --resume'):
                run_pipeline.validate_resume_environment(previous,output,changed,True)
            self.assertEqual(path.read_bytes(),before)
            # A complete rebuild may use another environment.
            run_pipeline.validate_resume_environment(previous,output,changed,False)
            # Manifest provenance takes precedence over a subsequently altered sidecar.
            previous['environment']=old
            path.write_text(json.dumps(changed),encoding='utf-8')
            with self.assertRaises(RuntimeError):
                run_pipeline.validate_resume_environment(previous,output,changed,True)

    def test_unknown_checkpoint_environment_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                run_pipeline.validate_resume_environment({'stages':{'1':{}}},Path(temporary),{},True)

    def test_three_public_figures_are_required_checkpoint_outputs(self):
        paths=run_pipeline.expected_outputs(config)[10]
        figures=[path for path in paths if path.parent.name=='figures']
        self.assertEqual(len(figures),3)
        self.assertEqual({path.name for path in figures},set(run_pipeline.RELEASE_FIGURES))

    def test_visual_stage_checks_every_surviving_image_before_reusing_decision(self):
        spec=importlib.util.spec_from_file_location('visual_stage',SRC/'6.geometry_ai_arbitration.py')
        stage=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            plain,overlay=directory/'plain.png',directory/'overlay.png'
            plain.write_bytes(b'original image')
            original={'plain_sha256':file_hash(plain),'overlay_sha256':'b'*64}
            checkpoint={**original,'fragment_id':'1','host_id':'2','Context_SHA256':'context',
                'Input_SHA256':fingerprint({'context':'context',**original}),
                'ai_decision':'keep','ai_reasoning':'original decision'}
            cache=pd.DataFrame([checkpoint])
            geometry=gpd.GeoDataFrame({ID:['1','2']},geometry=[box(0,0,1,1),box(2,0,3,1)],crs=2326)
            table=pd.DataFrame({ID:['1','2'],'Final_Main_Class':[UNKNOWN,COMMERCIAL],
                'Final_Sub_Class':[UNKNOWN,UNKNOWN]})
            candidates=pd.DataFrame([{'fragment_id':'1','host_id':'2','Context_SHA256':'context',
                'plain_image_path':str(plain),'overlay_image_path':str(overlay)}])
            paths={'CANDIDATES_RULES_PATH':str(directory/'candidates.csv'),
                'FINAL_SUGGESTIONS_PATH':str(directory/'suggestions.csv'),
                'INTERMEDIATE_DIR':str(directory),'LEGACY_OUTPUT_DIR':str(directory/'legacy'),
                'LEGACY_INTERMEDIATE_DIR':str(directory/'legacy')}
            with patch.multiple(stage,**paths),patch.object(stage,'read_table',return_value=table), \
                    patch.object(stage.pyogrio,'read_dataframe',return_value=geometry), \
                    patch.object(stage,'generate_candidates',return_value=candidates), \
                    patch.object(stage,'read_cache',return_value=cache),patch.object(stage,'save_cache'), \
                    patch.object(stage,'request_decision') as request,patch.object(stage,'render_pair_images') as images:
                stage.main()
                self.assertEqual(pd.read_csv(paths['FINAL_SUGGESTIONS_PATH']).Cache_Status.iloc[0],'matching_checkpoint')
                plain.write_bytes(b'changed surviving image')
                stage.main()
                self.assertEqual(pd.read_csv(paths['FINAL_SUGGESTIONS_PATH']).Cache_Status.iloc[0],'pending_no_matching_checkpoint')
                plain.unlink()
                stage.main()
                self.assertEqual(pd.read_csv(paths['FINAL_SUGGESTIONS_PATH']).Cache_Status.iloc[0],'matching_checkpoint')
                request.assert_not_called()
                images.assert_not_called()


if __name__=='__main__':
    unittest.main()
