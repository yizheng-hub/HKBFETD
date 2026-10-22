"""Step 6: rebuild review candidates and resume visual jobs by content hashes.

No API or imagery requests by default. --images regenerates missing comparison
images; --refresh requests unresolved visual decisions. Existing results are
never identified solely by a transient filename.
"""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from ai_cache import fingerprint, file_hash, read_cache, save_cache, vision_checkpoint_matches
from classification import UNKNOWN, NONASSESSED
from config import (STEP5_MERGED_OUTPUT_PATH, OFFICIAL_LIB_BASE_PATH, CANDIDATES_RULES_PATH,
    FINAL_SUGGESTIONS_PATH, AI_DECISIONS_LOG_PATH, IMAGE_OUTPUT_DIR, LEGACY_OUTPUT_DIR,
    LEGACY_INTERMEDIATE_DIR, INTERMEDIATE_DIR, DATA_DIR, VISION_API_KEY, VISION_CLOUD_MODEL_NAME, BASE_URL)
from data_quality import ID, ids, require_unique
from stage_io import read_table

PROMPT_VERSION = 'v2-geometry-review-preserve-records-20261022'
PROMPT = ('Compare the aerial image and footprint overlay. Red is a candidate fragment, green its proposed host. '
          'Decide keep, merge, or delete (noise only), using visible geometry and the supplied function context. '
          'An adjacent or small building is not sufficient evidence for merge/delete. Return JSON decision and reasoning. '
          'Do not infer building height, storey counts, floor area or use percentages from these images.')
CACHE_FIELDS = ['fragment_id','host_id','Context_SHA256','Input_SHA256','plain_sha256','overlay_sha256',
                'ai_decision','ai_reasoning','model','prompt_version','timestamp']


def recover_validation_images():
    directory = Path(DATA_DIR) / 'evidence' / 'legacy_vision_images'
    directory.mkdir(parents=True,exist_ok=True)
    archive = Path(LEGACY_OUTPUT_DIR) / 'Validation_Materials.zip'
    if not archive.exists():
        return directory
    rows = []
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.namelist():
            name = Path(entry).name
            if not re.fullmatch(r'\d+_vs_\d+_(?:plain|overlay)\.png',name):
                continue
            path = directory / name
            payload = zipped.read(entry)
            if not path.exists():
                path.write_bytes(payload)
            rows.append({'name':name,'sha256':file_hash(path),'source':'Validation_Materials.zip/' + entry})
    pd.DataFrame(rows).to_csv(directory / 'recovery_manifest.csv',index=False,encoding='utf-8-sig')
    return directory


def context_hash(fragment, host):
    return fingerprint({'fragment_id':fragment.name,'host_id':host.name,
        'fragment_geometry':fragment.geometry.wkb_hex,'host_geometry':host.geometry.wkb_hex,
        'host_main':host.Final_Main_Class,'host_sub':host.Final_Sub_Class,
        'model':VISION_CLOUD_MODEL_NAME,'prompt_version':PROMPT_VERSION,'prompt':PROMPT})


def generate_candidates(frame):
    indexed = frame.set_index(ID)
    fragments = indexed[indexed.Final_Main_Class.eq(UNKNOWN)]
    hosts = indexed[~indexed.Final_Main_Class.isin([UNKNOWN,NONASSESSED])]
    rows = []
    recovered = recover_validation_images()
    image_root = Path(IMAGE_OUTPUT_DIR)
    image_root.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(fragments),10000):
        batch = fragments.iloc[start:start+10000]
        left,right = hosts.sindex.query(batch.geometry.buffer(3),predicate='intersects')
        for i,j in zip(left,right):
            fragment,host = batch.iloc[i],hosts.iloc[j]
            names = {kind:f'{fragment.name}_vs_{host.name}_{kind}.png' for kind in ['plain','overlay']}
            paths = {kind:recovered / name if (recovered / name).exists() else image_root / name for kind,name in names.items()}
            rows.append({'fragment_id':str(fragment.name),'host_id':str(host.name),
                'rule_decision':'review', 'distance_m':fragment.geometry.distance(host.geometry),
                'Context_SHA256':context_hash(fragment,host),
                'plain_image_path':str(paths['plain']),'overlay_image_path':str(paths['overlay'])})
    return pd.DataFrame(rows,columns=['fragment_id','host_id','rule_decision','distance_m','Context_SHA256','plain_image_path','overlay_image_path'])


def generate_images(pair, indexed):
    import contextily as cx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fragment = indexed.loc[[pair.fragment_id]]
    host = indexed.loc[[pair.host_id]]
    bounds = gpd.GeoSeries(pd.concat([fragment.geometry,host.geometry]),crs=indexed.crs).union_all().buffer(25).bounds
    for kind in ['plain','overlay']:
        path = Path(pair[kind+'_image_path'])
        if path.exists() and path.stat().st_size:
            continue
        fig,ax = plt.subplots(figsize=(5,5))
        ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[1],bounds[3]); ax.axis('off')
        cx.add_basemap(ax,crs=indexed.crs,source=cx.providers.Esri.WorldImagery,zoom=19)
        if kind == 'overlay':
            host.plot(ax=ax,color='lime',edgecolor='white',alpha=.6)
            fragment.plot(ax=ax,color='red',edgecolor='white',alpha=.6)
        fig.savefig(path,dpi=150,bbox_inches='tight',pad_inches=0)
        plt.close(fig)


def request_decision(pair, host):
    from openai import OpenAI
    images = []
    for kind in ['plain','overlay']:
        payload = base64.b64encode(Path(pair[kind+'_image_path']).read_bytes()).decode('ascii')
        images.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+payload}})
    text = PROMPT + '\nHost function: ' + host.Final_Main_Class + ' / ' + host.Final_Sub_Class
    client = OpenAI(api_key=VISION_API_KEY,base_url=BASE_URL)
    response = client.chat.completions.create(model=VISION_CLOUD_MODEL_NAME,temperature=0,
        messages=[{'role':'user','content':[{'type':'text','text':text}]+images}],timeout=90)
    text = response.choices[0].message.content.strip()
    if text.startswith('```'):
        text = text.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(text)


def main(refresh=False, images=False, limit=100):
    table = read_table(STEP5_MERGED_OUTPUT_PATH)
    geometry = pyogrio.read_dataframe(OFFICIAL_LIB_BASE_PATH,columns=[ID])
    geometry[ID] = ids(geometry[ID])
    require_unique(geometry)
    frame = geometry.merge(table[[ID,'Final_Main_Class','Final_Sub_Class']],on=ID,validate='one_to_one')
    require_unique(frame)
    candidates = generate_candidates(frame)
    cache = read_cache(AI_DECISIONS_LOG_PATH,CACHE_FIELDS)
    for col in CACHE_FIELDS:
        if col not in cache:
            cache[col] = pd.NA
    cache = cache.drop_duplicates(['fragment_id','host_id','Context_SHA256'],keep='last')
    done = {(str(row.fragment_id),str(row.host_id),str(row.Context_SHA256)):row for _,row in cache.iterrows()}
    indexed = frame.set_index(ID)
    if refresh and not VISION_API_KEY:
        raise ValueError('HKBFETD_VISION_API_KEY is required for --refresh; checkpoints remain intact')
    attempted, statuses = 0, []
    for _,pair in candidates.iterrows():
        key = (pair.fragment_id,pair.host_id,pair.Context_SHA256)
        previous = done.get(key)
        available = all(Path(pair[kind+'_image_path']).is_file() for kind in ['plain','overlay'])
        hashes = {kind+'_sha256':file_hash(pair[kind+'_image_path']) for kind in ['plain','overlay']
                  if Path(pair[kind+'_image_path']).is_file()}
        same_images = vision_checkpoint_matches(previous,pair.Context_SHA256,hashes)
        if previous is not None and same_images:
            statuses.append({**pair.to_dict(),'Cache_Status':'matching_checkpoint','Image_Status':'present' if available else 'missing_but_success_checkpoint_retained',
                             'ai_decision':previous.ai_decision,'ai_reasoning':previous.ai_reasoning})
            continue
        may_attempt = limit is None or attempted < limit
        if may_attempt and (images or refresh):
            attempted += 1
            try:
                if images and not available:
                    generate_images(pair,indexed)
                available = all(Path(pair[kind+'_image_path']).is_file() for kind in ['plain','overlay'])
                if refresh and available:
                    hashes = {kind+'_sha256':file_hash(pair[kind+'_image_path']) for kind in ['plain','overlay']}
                    response = request_decision(pair,indexed.loc[pair.host_id])
                    decision = str(response.get('decision','')).lower()
                    if decision not in ['keep','merge','delete']:
                        raise ValueError('Unrecognized visual decision')
                    entry = {**{k:pair[k] for k in ['fragment_id','host_id','Context_SHA256']},**hashes,
                        'Input_SHA256':fingerprint({'context':pair.Context_SHA256,**hashes}),
                        'ai_decision':decision,'ai_reasoning':response.get('reasoning',''),
                        'model':VISION_CLOUD_MODEL_NAME,'prompt_version':PROMPT_VERSION,
                        'timestamp':datetime.now(timezone.utc).isoformat()}
                    cache = pd.concat([cache,pd.DataFrame([entry])],ignore_index=True)
                    save_cache(cache,AI_DECISIONS_LOG_PATH)
                    statuses.append({**pair.to_dict(),'Cache_Status':'matching_checkpoint','Image_Status':'present',
                                     'ai_decision':decision,'ai_reasoning':entry['ai_reasoning']})
                    continue
            except Exception as error:
                print(f'[WARN] Visual job not checkpointed: {type(error).__name__}')
        statuses.append({**pair.to_dict(),'Cache_Status':'pending_no_matching_checkpoint',
                         'Image_Status':'present' if available else 'missing','ai_decision':None,'ai_reasoning':None})
    save_cache(cache,AI_DECISIONS_LOG_PATH)
    result = pd.DataFrame(statuses,columns=list(candidates.columns)+['Cache_Status','Image_Status','ai_decision','ai_reasoning'])
    legacy_path = Path(LEGACY_OUTPUT_DIR)/'step6_ai_decisions_log.csv'
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path,dtype={'fragment_id':'string','host_id':'string'},low_memory=False)
        legacy = legacy.drop_duplicates(['fragment_id','host_id'],keep='last')
        legacy = legacy.rename(columns={c:'Legacy_'+c for c in legacy if c not in ['fragment_id','host_id']})
        result = result.merge(legacy,on=['fragment_id','host_id'],how='left',validate='one_to_one')
    result.to_csv(CANDIDATES_RULES_PATH,index=False,encoding='utf-8-sig')
    result.to_csv(FINAL_SUGGESTIONS_PATH,index=False,encoding='utf-8-sig')
    result[result.Cache_Status.ne('matching_checkpoint')].to_csv(Path(INTERMEDIATE_DIR)/'step6_pending_review.csv',index=False,encoding='utf-8-sig')
    candidate_path = Path(LEGACY_INTERMEDIATE_DIR)/'step6_geometry_candidates_with_rules_and_images.csv'
    if candidate_path.exists() and legacy_path.exists():
        old_candidates = pd.read_csv(candidate_path,dtype={'fragment_id':'string','host_id':'string'})
        old_log = pd.read_csv(legacy_path,dtype={'fragment_id':'string','host_id':'string'})
        inventory = old_candidates[['fragment_id','host_id']].merge(old_log[['fragment_id','host_id','ai_decision']],on=['fragment_id','host_id'],how='left',validate='one_to_one')
    else:
        inventory = pd.DataFrame(columns=['fragment_id','host_id','ai_decision'])
    inventory['Status'] = np.where(inventory.ai_decision.notna(),'completed_legacy_unverifiable_context','legacy_request_not_completed')
    inventory.to_csv(Path(INTERMEDIATE_DIR)/'legacy_vision_resume_inventory.csv',index=False,encoding='utf-8-sig')
    print(f'[INFO] Step 6: {len(result)} current pairs; legacy decisions retained; {(inventory.ai_decision.isna()).sum()} old pairs lacked a decision')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--images',action='store_true')
    parser.add_argument('--limit',type=int,default=100)
    args = parser.parse_args()
    main(args.refresh,args.images,args.limit)
