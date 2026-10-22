"""Rebuild v2 offline with input/code/output hashes and resumable stages.

Run inside HKbuilding. --resume skips only content-matched, completed stages.
--input-bundle saves the exact, minimal local input snapshot for another machine.
External AI requests are explicit opt-in commands in Steps 4 and 6.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

from ai_cache import file_hash, fingerprint

SCRIPTS = ['1.data_preprocessing.py','2.osm_aggregation.py','3.rule_engine.py','4.llm_verification.py',
    '5.evaluation_and_merge.py','6.geometry_ai_arbitration.py','7.apply_calibrations.py',
    '8.feature_engineering.py','9.ml_classification.py','10.final_statistics_report.py']
RELEASE_FIGURES = ['dimensions_before_after.png','coverage_and_unknowns.png','model_spatial_validation.png']


def input_files(cfg):
    root = Path(cfg.BASE_DIR)
    files = [Path(p) for p in cfg.FILE_PATHS_LANDSD.values()]
    files += sorted(Path(cfg.OZP_DATA_DIR).glob('*.json'))
    files += [Path(p) for p in [cfg.BDBIAR_FILE_PATH,cfg.OVERTURE_PARQUET_FILE_PATH,
        cfg.OSM_SOURCE_SNAPSHOT,cfg.TERRAIN_SAMPLES_PATH,cfg.TERRAIN_METADATA_PATH,
        cfg.CSDI_BUILDINGS_PATH,cfg.CSDI_METADATA_PATH,cfg.OSM_STRUCTURES_PATH,cfg.OSM_STRUCTURES_METADATA_PATH]]
    files.append(root/'data'/'sources'/'source_manifest.json')
    # Historical evidence is optional for inference, but frozen when available.
    for path in [Path(cfg.LEGACY_OUTPUT_DIR)/'step4_llm_verified_classification.csv',
                 Path(cfg.LEGACY_OUTPUT_DIR)/'step6_ai_decisions_log.csv',
                 Path(cfg.LEGACY_OUTPUT_DIR)/'Validation_Materials.zip',
                 Path(cfg.LEGACY_INTERMEDIATE_DIR)/'step6_geometry_candidates_with_rules_and_images.csv']:
        if path.exists():
            files.append(path)
    files.append(Path(cfg.LEGACY_OUTPUT_DIR)/'HK_UBEM_Buildings_Public_v1_1.csv')
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'Required frozen input missing: {path}')
    return files


def expected_outputs(cfg):
    inter,out=Path(cfg.INTERMEDIATE_DIR),Path(cfg.OUTPUT_DIR)
    return {
        1:[Path(cfg.OFFICIAL_LIBRARY_PATH),Path(cfg.OFFICIAL_LIB_BASE_PATH),Path(cfg.OSM_ALL_PATH),
           Path(cfg.BDBIAR_CACHE_PATH),Path(cfg.OVERTURE_CLEAN_PATH)],
        2:[Path(cfg.AGGREGATED_GDF_PATH),Path(cfg.OSM_CLEAN_PATH)],
        3:[Path(cfg.RULE_ENGINE_OUTPUT_PATH),Path(cfg.RULE_ENGINE_OUTPUT_PATH).with_suffix('.geojson')],
        4:[Path(cfg.LLM_VERIFICATION_OUTPUT),out/'step4_current_status.csv',out/'step4_pending_review.csv'],
        5:[Path(cfg.STEP5_MERGED_OUTPUT_PATH),Path(cfg.STEP5_MERGED_OUTPUT_PATH).with_suffix('.geojson')],
        6:[Path(cfg.CANDIDATES_RULES_PATH),Path(cfg.FINAL_SUGGESTIONS_PATH),Path(cfg.AI_DECISIONS_LOG_PATH),inter/'legacy_vision_resume_inventory.csv'],
        7:[Path(cfg.AI_CALIBRATED_OUTPUT_PATH),Path(cfg.AI_CALIBRATED_OUTPUT_PATH).with_suffix('.geojson')],
        8:[Path(cfg.FEATURE_X_ALL_PATH),Path(cfg.FEATURE_X_TRAIN_ML_PATH),Path(cfg.FEATURE_Y_MULTILABEL_PATH),
           Path(cfg.FEATURE_TRAINING_DATA_ML_PATH),Path(cfg.FEATURE_BASE_INDEXED_PATH),inter/'feature_metadata.json'],
        9:[Path(cfg.ML_FINAL_OUTPUT_PATH),Path(cfg.ML_FINAL_OUTPUT_PATH).with_suffix('.geojson'),
           Path(cfg.ML_MODELS_PATH),out/'model_validation.json',out/'model_spatial_split.csv',out/'model_holdout_predictions.csv'],
        10:[Path(cfg.PUBLIC_DATASET_CSV_PATH),Path(cfg.PUBLIC_DATASET_GEOJSON_PATH),
            Path(cfg.PUBLIC_DATASET_CSV_PATH).with_suffix('.parquet'),out/'comparison_summary.json',
            out/'v1_1_to_v2_building_changes.csv',out/'release_validation.json',out/'RELEASE_NOTES_zh.txt',
            out/'stratified_manual_validation_template.csv',out/'release_checksums.json',
            out/'field_dictionary.csv',out/'dimension_qa_summary.json',out/'all_common_field_change_summary.csv',
            out/'all_common_field_changes.parquet',out/'v1_1_to_v2_class_transition.csv',out/'reader_cases_v2.csv']
            + [out/'figures'/name for name in RELEASE_FIGURES]}


def hash_paths(paths,root):
    return {str(path.relative_to(root)).replace('\\','/'):{'sha256':file_hash(path),'bytes':path.stat().st_size} for path in paths}


def current_environment():
    versions={distribution.metadata['Name']:distribution.version for distribution in importlib.metadata.distributions() if distribution.metadata['Name']}
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':dict(sorted(versions.items()))}


def validate_resume_environment(previous, output, environment, resume):
    if not resume or not previous.get('stages'):
        return
    recorded=previous.get('environment')
    if recorded is None and (output/'environment_versions.json').is_file():
        recorded=json.loads((output/'environment_versions.json').read_text(encoding='utf-8'))
    if recorded != environment:
        raise RuntimeError('Resume environment differs from the recorded run or is unknown; restore its environment or rebuild without --resume. Existing records were not overwritten.')


def freeze_environment(output, environment):
    versions=environment['packages']
    (output/'environment_versions.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    required=['numpy','pandas','geopandas','shapely','pyproj','pyarrow','scikit-learn','lightgbm','imbalanced-learn',
        'matplotlib','seaborn','folium','tqdm','pyogrio','openai','httpx','opencc-python-reimplemented','pyrosm',
        'fuzzywuzzy','python-Levenshtein','scipy','contextily','Pillow','psutil','tabulate']
    lock=[]
    for name in required:
        if name=='opencc-python-reimplemented' and name not in versions:
            name='OpenCC'
        lock.append(name+'=='+importlib.metadata.version(name))
    (output/'requirements.lock.txt').write_text('\n'.join(lock)+'\n',encoding='utf-8')
    conda=os.environ.get('CONDA_EXE')
    if conda and Path(conda).is_file():
        result=subprocess.run([conda,'list','--prefix',sys.prefix,'--explicit'],capture_output=True,text=True)
        if result.returncode==0:
            # Do not export authenticated channel URLs or local credential settings.
            lines=[line for line in result.stdout.splitlines() if not ('@' in line and '://' in line) and '/t/' not in line and 'token=' not in line.lower()]
            (output/'conda-win64-explicit.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def make_bundles(cfg,files,include_inputs):
    root,out=Path(cfg.BASE_DIR),Path(cfg.OUTPUT_DIR)
    code=list((root/'src').glob('*.py'))+list((root/'ctl').glob('*'))+list((root/'tests').glob('*.py'))
    code += [root/name for name in ['README.md','LICENSE','environment.yml','requirements.txt']]
    with zipfile.ZipFile(out/'v2_source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(p for p in code if p.is_file()):
            archive.write(path,path.relative_to(root).as_posix())
        for name in ['requirements.lock.txt','environment_versions.json','conda-win64-explicit.txt']:
            if (out/name).exists():
                archive.write(out/name,name)
    if include_inputs:
        with zipfile.ZipFile(out/'reproduction_inputs.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for path in files:
                archive.write(path,path.relative_to(root).as_posix())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--through',type=int,default=10,choices=range(1,11))
    parser.add_argument('--input-bundle',action='store_true')
    args=parser.parse_args()
    if args.run_root:
        os.environ['HKBFETD_RUN_ROOT']=str(args.run_root.resolve())
    import config as cfg
    root,out=Path(cfg.BASE_DIR),Path(cfg.OUTPUT_DIR)
    files=input_files(cfg)
    inputs=hash_paths(files,root)
    input_signature=fingerprint(inputs)
    manifest_path=out/'pipeline_run_manifest.json'
    previous=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    runtime=current_environment()
    validate_resume_environment(previous,out,runtime,args.resume)
    manifest={'dataset_version':'2.0','input_signature':input_signature,'inputs':inputs,'stages':{},
        'environment':runtime,'environment_signature':fingerprint(runtime),
        'api_requests':'disabled in default replay','started_utc':datetime.now(timezone.utc).isoformat()}
    freeze_environment(out,runtime)
    outputs=expected_outputs(cfg)
    shared=[root/'src'/name for name in ['config.py','data_quality.py','classification.py','utils.py','stage_io.py','ai_cache.py','structural_use.py']]
    upstream='raw_inputs'
    environment=os.environ.copy()
    environment.update({'PYTHONIOENCODING':'utf-8','TQDM_DISABLE':'1','PYTHONHASHSEED':'0'})
    for step,script in enumerate(SCRIPTS[:args.through],start=1):
        source=hash_paths([root/'src'/script]+shared,root)
        signature_fields={'source':source,'inputs':input_signature,'upstream':upstream}
        if step in [2,3]:
            signature_fields['controls']=hash_paths(sorted(p for p in (root/'ctl').glob('*') if p.is_file()),root)
        if step==4:
            signature_fields['model']=cfg.CLOUD_MODEL_NAME
        if step==6:
            signature_fields['model']=cfg.VISION_CLOUD_MODEL_NAME
        signature=fingerprint(signature_fields)
        old=previous.get('stages',{}).get(str(step),{})
        if args.resume and old.get('signature')==signature and all(path.is_file() and old.get('outputs',{}).get(str(path),{}).get('sha256')==file_hash(path) for path in outputs[step]):
            record=old
            print(f'[INFO] Step {step}: matching completed checkpoint reused',flush=True)
        else:
            print(f'[INFO] Step {step}: rebuilding {script}',flush=True)
            logfile=Path(cfg.LOG_DIR)/f'runner_step_{step}.txt'
            with logfile.open('w',encoding='utf-8') as stream:
                process=subprocess.run([sys.executable,'-u',str(root/'src'/script)],cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            if process.returncode:
                print(f'[ERROR] Step {step} failed; see {logfile}',flush=True)
                raise SystemExit(process.returncode)
            missing=[str(path) for path in outputs[step] if not path.is_file() or path.stat().st_size==0]
            if missing:
                raise RuntimeError('Stage did not produce required outputs: '+str(missing))
            record={'signature':signature,'source':source,'outputs':{str(path):{'sha256':file_hash(path),'bytes':path.stat().st_size} for path in outputs[step]},
                'completed_utc':datetime.now(timezone.utc).isoformat()}
            print(f'[INFO] Step {step}: complete, output hashes verified',flush=True)
        manifest['stages'][str(step)]=record
        upstream=fingerprint(record)
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.through==10:
        make_bundles(cfg,files,args.input_bundle)
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[INFO] Reproducible v2 run complete through Step {args.through}: {out}',flush=True)


if __name__=='__main__':
    main()
