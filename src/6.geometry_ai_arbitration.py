"""Step 6: rebuild review candidates and resume visual jobs by content hashes.

No API or imagery requests by default. --images regenerates missing comparison
images; --refresh requests unresolved visual decisions. Existing results are
never identified solely by a transient filename.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
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
    LEGACY_INTERMEDIATE_DIR, INTERMEDIATE_DIR, DATA_DIR, BASE_DIR, RUN_ROOT,
    VISION_API_KEY, VISION_CLOUD_MODEL_NAME, BASE_URL)
from data_quality import ID, ids, require_unique
from stage_io import read_table

PROMPT_VERSION = 'v2-geometry-review-preserve-records-20261022'
PROMPT = ('Compare the aerial image and footprint overlay. Red is a candidate fragment, green its proposed host. '
          'Decide keep, merge, or delete (noise only), using visible geometry and the supplied function context. '
          'An adjacent or small building is not sufficient evidence for merge/delete. Return JSON decision and reasoning. '
          'Do not infer building height, storey counts, floor area or use percentages from these images.')
CACHE_FIELDS = ['fragment_id','host_id','Context_SHA256','Input_SHA256','plain_sha256','overlay_sha256',
                'ai_decision','ai_reasoning','model','prompt_version','timestamp']
IMAGE_PROVIDER = 'Esri.WorldImagery'
IMAGE_ZOOM = 19
IMAGE_RENDER_STYLE = 'step6-shared-basemap-plain-then-red-fragment-green-host-v1'
IMAGE_BUFFER_M = 25


def context_hash(fragment, host):
    """Classifier/checkpoint context: includes both geometries and host function."""
    return fingerprint({'fragment_id':str(fragment.name),'host_id':str(host.name),
        'fragment_geometry':fragment.geometry.wkb_hex,'host_geometry':host.geometry.wkb_hex,
        'host_main':host.Final_Main_Class,'host_sub':host.Final_Sub_Class,
        'model':VISION_CLOUD_MODEL_NAME,'prompt_version':PROMPT_VERSION,'prompt':PROMPT})


def geometry_render_context(fragment, host, crs):
    """Render context is geometry/provider-specific and independent of text class."""
    bounds = gpd.GeoSeries([fragment.geometry, host.geometry], crs=crs).union_all().buffer(IMAGE_BUFFER_M).bounds
    bounds = [float(value) for value in bounds]
    fragment_sha = hashlib.sha256(fragment.geometry.wkb).hexdigest()
    host_sha = hashlib.sha256(host.geometry.wkb).hexdigest()
    payload = {'fragment_id':str(fragment.name),'host_id':str(host.name),
        'fragment_geometry_sha256':fragment_sha,'host_geometry_sha256':host_sha,
        'crs':str(crs),'tile_bounds':bounds,
        'tile_bounds_semantics':'viewport bounds passed to contextily in the recorded CRS; not provider tile indices',
        'provider':IMAGE_PROVIDER,'zoom':IMAGE_ZOOM,
        'render_style':IMAGE_RENDER_STYLE,'buffer_m':IMAGE_BUFFER_M,'canvas_inches':[5,5],
        'dpi':150,'bbox_inches':'tight','pad_inches':0}
    return payload, fingerprint(payload)


def isolated_image_paths(pair, image_root, suffix=None):
    stem = f"{pair.fragment_id}_vs_{pair.host_id}_{pair.Context_SHA256[:12]}_{pair.Render_Context_SHA256[:12]}"
    if suffix:
        stem += '_' + suffix
    root = Path(image_root)
    return {'plain_image_path':str(root/(stem+'_plain.png')),
            'overlay_image_path':str(root/(stem+'_overlay.png')),
            'image_context_path':str(root/(stem+'.image_context.json'))}


def recover_validation_images(archive_path=None, destination=None):
    """Recover frozen validation pixels into the run root without touching inputs."""
    archive = Path(archive_path or Path(LEGACY_OUTPUT_DIR) / 'Validation_Materials.zip')
    directory = Path(destination or Path(INTERMEDIATE_DIR) / 'legacy_vision_images')
    run_root = Path(RUN_ROOT).resolve()
    data_root = Path(DATA_DIR).resolve()
    resolved = directory.resolve()
    if not resolved.is_relative_to(run_root) or resolved.is_relative_to(data_root):
        raise ValueError('Historical image recovery must stay inside the run root and outside DATA_DIR')
    if not archive.is_file():
        return directory
    archive_sha = file_hash(archive)
    manifest_path = directory / f'recovery_manifest_{archive_sha[:12]}.json'
    rows = []
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            name = Path(entry.filename).name
            if entry.is_dir() or not re.fullmatch(r'\d+_vs_\d+_(?:plain|overlay)\.png', name):
                continue
            payload = zipped.read(entry)
            payload_sha = hashlib.sha256(payload).hexdigest()
            target = directory / name
            if target.exists() and file_hash(target) != payload_sha:
                target = directory / f'{Path(name).stem}__zip-{archive_sha[:12]}{Path(name).suffix}'
                if target.exists() and file_hash(target) != payload_sha:
                    raise ValueError(f'Conflicting recovered historical image: {target}')
            if not target.exists():
                directory.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                status = 'recovered'
            else:
                status = 'already_present_matching_hash'
            rows.append({'archive_path':str(archive),'archive_sha256':archive_sha,
                'archive_entry':entry.filename,'payload_sha256':payload_sha,
                'recovered_path':str(target),'status':status,
                'context_status':'context_unverified_historical_only'})
    if rows:
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({'archive_path':str(archive),'archive_sha256':archive_sha,
            'entries':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    return directory


def generate_candidates(frame, image_root=None, isolated=False):
    """Build candidate inventory; historical images remain outside scoped paths."""
    image_root = Path(image_root or IMAGE_OUTPUT_DIR)
    indexed = frame.set_index(ID)
    fragments = indexed[indexed.Final_Main_Class.eq(UNKNOWN)]
    hosts = indexed[~indexed.Final_Main_Class.isin([UNKNOWN,NONASSESSED])]
    rows = []
    # Frozen historical pixels are replay inputs only, not eligible for scoped VLM use.
    legacy_root = recover_validation_images()
    for start in range(0,len(fragments),10000):
        batch = fragments.iloc[start:start+10000]
        left,right = hosts.sindex.query(batch.geometry.buffer(3),predicate='intersects')
        for i,j in zip(left,right):
            fragment,host = batch.iloc[i],hosts.iloc[j]
            pair_context = context_hash(fragment,host)
            render_payload,render_sha = geometry_render_context(fragment,host,indexed.crs)
            base = {'fragment_id':str(fragment.name),'host_id':str(host.name),
                'rule_decision':'review','distance_m':fragment.geometry.distance(host.geometry),
                'Context_SHA256':pair_context,'Render_Context_SHA256':render_sha}
            if isolated:
                paths = isolated_image_paths(pd.Series(base),image_root)
                stem = Path(paths['image_context_path']).name[:-len('.image_context.json')]
                for sidecar in sorted(Path(image_root).glob(stem+'*.image_context.json')):
                    candidate_paths = {'plain_image_path':str(sidecar.with_name(sidecar.name[:-len('.image_context.json')]+'_plain.png')),
                        'overlay_image_path':str(sidecar.with_name(sidecar.name[:-len('.image_context.json')]+'_overlay.png')),
                        'image_context_path':str(sidecar)}
                    if sidecar_is_valid(pd.Series({**base,**candidate_paths}),indexed):
                        paths = candidate_paths
                        break
            else:
                names = {kind:f'{fragment.name}_vs_{host.name}_{kind}.png' for kind in ['plain','overlay']}
                paths = {kind+'_image_path':str(legacy_root/name if (legacy_root/name).is_file() else image_root/name)
                         for kind,name in names.items()}
                paths['image_context_path']=''
            rows.append({**base,**paths})
    return pd.DataFrame(rows,columns=['fragment_id','host_id','rule_decision','distance_m','Context_SHA256',
        'Render_Context_SHA256','plain_image_path','overlay_image_path','image_context_path'])


def read_scope_ids(path):
    path=Path(path)
    if path.suffix.lower()=='.csv':
        values=pd.read_csv(path,dtype='string')
        for name in ('fragment_id','building_id','Building_ID_建筑ID',ID):
            if name in values.columns:
                return set(values[name].dropna().astype(str).str.strip())
        if not len(values.columns):
            raise ValueError('Scope-ID CSV has no columns')
        return set(values.iloc[:,0].dropna().astype(str).str.strip())
    return {line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()}


def filter_candidates(candidates, scope_ids=None, pair_whitelist=None):
    """Filter before any image generation. Pair whitelist also binds current context."""
    if scope_ids is not None and pair_whitelist is not None:
        raise ValueError('Use one scope mechanism: --scope-id-file or --pair-whitelist')
    if scope_ids is not None:
        return candidates[candidates.fragment_id.astype(str).isin(set(map(str,scope_ids)))].copy()
    if pair_whitelist is None:
        return candidates.copy()
    allowed={}
    for row in pair_whitelist:
        key=(str(row['fragment_id']),str(row['host_id']))
        if key in allowed:
            raise ValueError(f'Duplicate pair in whitelist: {key}')
        allowed[key]=str(row['Context_SHA256'])
    selected=candidates[candidates.apply(lambda r:(str(r.fragment_id),str(r.host_id)) in allowed,axis=1)].copy()
    found=set()
    for _,row in selected.iterrows():
        key=(str(row.fragment_id),str(row.host_id)); found.add(key)
        if str(row.Context_SHA256)!=allowed[key]:
            raise ValueError(f'Whitelist Context_SHA256 mismatch for {key}')
    missing=set(allowed)-found
    if missing:
        raise ValueError(f'Whitelist pairs missing from current candidates: {sorted(missing)[:5]}')
    return selected


def image_context_payload(pair,indexed):
    fragment=indexed.loc[pair.fragment_id]
    host=indexed.loc[pair.host_id]
    render_payload,render_sha=geometry_render_context(fragment,host,indexed.crs)
    if str(pair.Render_Context_SHA256)!=render_sha:
        raise ValueError('Candidate render context no longer matches current geometries')
    return {'fragment_id':str(pair.fragment_id),'host_id':str(pair.host_id),
        'Context_SHA256':str(pair.Context_SHA256),'Render_Context_SHA256':render_sha,**render_payload}


def sidecar_is_valid(pair,indexed):
    sidecar=Path(pair.image_context_path)
    paths=[Path(pair.plain_image_path),Path(pair.overlay_image_path)]
    if not sidecar.is_file() or not all(p.is_file() and p.stat().st_size for p in paths):
        return False
    try:
        saved=json.loads(sidecar.read_text(encoding='utf-8'))
        expected=image_context_payload(pair,indexed)
        if any(saved.get(key)!=value for key,value in expected.items()):
            return False
        return saved.get('plain_sha256')==file_hash(paths[0]) and saved.get('overlay_sha256')==file_hash(paths[1])
    except (OSError,ValueError,KeyError,TypeError):
        return False


def _add_basemap(ax, crs, bounds):
    import contextily as cx
    ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[1],bounds[3]); ax.axis('off')
    cx.add_basemap(ax,crs=crs,source=cx.providers.Esri.WorldImagery,zoom=IMAGE_ZOOM)


def render_pair_images(pair,indexed):
    """Fetch one basemap capture, save plain, then overlay from that same capture."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fragment=indexed.loc[[pair.fragment_id]]
    host=indexed.loc[[pair.host_id]]
    context=image_context_payload(pair,indexed)
    bounds=context['tile_bounds']
    plain=Path(pair.plain_image_path); overlay=Path(pair.overlay_image_path)
    plain.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(5,5))
    try:
        _add_basemap(ax,indexed.crs,bounds)
        fig.savefig(plain,dpi=150,bbox_inches='tight',pad_inches=0)
        host.plot(ax=ax,color='lime',edgecolor='white',alpha=.6)
        fragment.plot(ax=ax,color='red',edgecolor='white',alpha=.6)
        fig.savefig(overlay,dpi=150,bbox_inches='tight',pad_inches=0)
    finally:
        plt.close(fig)
    return {'acquired_at_utc':datetime.now(timezone.utc).isoformat(),
        'provider':IMAGE_PROVIDER,'zoom':IMAGE_ZOOM,'crs':str(indexed.crs),'tile_bounds':bounds,
        'fragment_geometry_sha256':context['fragment_geometry_sha256'],
        'host_geometry_sha256':context['host_geometry_sha256'],'render_context_sha256':context['Render_Context_SHA256']}


def _new_isolated_paths(pair,image_root):
    paths=isolated_image_paths(pair,image_root)
    existing=[Path(paths[k]) for k in ('plain_image_path','overlay_image_path','image_context_path')]
    if not any(p.exists() for p in existing):
        return paths
    seed='|'.join(file_hash(p) if p.is_file() else p.read_text(encoding='utf-8',errors='replace') for p in existing if p.exists())
    suffix='rebuild-'+hashlib.sha256(seed.encode('utf-8')).hexdigest()[:10]
    paths=isolated_image_paths(pair,image_root,suffix)
    n=1
    while any(Path(paths[k]).exists() for k in ('plain_image_path','overlay_image_path','image_context_path')):
        paths=isolated_image_paths(pair,image_root,f'{suffix}-{n}'); n+=1
    return paths


def ensure_isolated_images(pair,indexed,image_root,renderer=render_pair_images):
    """Reuse only sidecar-verified image pairs; mismatches move to new paths."""
    pair=pair.copy()
    if sidecar_is_valid(pair,indexed):
        return pair,False
    paths=_new_isolated_paths(pair,image_root)
    for key,value in paths.items(): pair[key]=value
    Path(pair.plain_image_path).parent.mkdir(parents=True,exist_ok=True)
    render_meta=renderer(pair,indexed)
    context=image_context_payload(pair,indexed)
    sidecar={**context,**render_meta,
        'plain_sha256':file_hash(pair.plain_image_path),'overlay_sha256':file_hash(pair.overlay_image_path),
        'plain_image_path':str(pair.plain_image_path),'overlay_image_path':str(pair.overlay_image_path),
        'render_style':IMAGE_RENDER_STYLE}
    Path(pair.image_context_path).write_text(json.dumps(sidecar,ensure_ascii=False,indent=2),encoding='utf-8')
    return pair,True


def run_scoped_image_jobs(candidates,indexed,image_root,image_limit,renderer=render_pair_images):
    """Attempt at most image_limit scoped pairs; failed renders consume a slot."""
    rows=[]; attempts=0
    for _,source in candidates.iterrows():
        pair=source.copy()
        if sidecar_is_valid(pair,indexed):
            pair['Image_Generation_Status']='sidecar_reused'
            rows.append(pair); continue
        if image_limit is not None and attempts>=image_limit:
            pair['Image_Generation_Status']='limit_not_attempted'
            rows.append(pair); continue
        attempts+=1
        try:
            pair,_=ensure_isolated_images(pair,indexed,image_root,renderer=renderer)
            pair['Image_Generation_Status']='generated' if sidecar_is_valid(pair,indexed) else 'failed_validation'
        except Exception as error:
            pair['Image_Generation_Status']='failed_'+type(error).__name__
        rows.append(pair)
    return pd.DataFrame(rows).reset_index(drop=True),attempts


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


def main(refresh=False, images=False, limit=100, image_limit=100, scope_id_file=None,
         pair_whitelist_path=None, image_output_dir=None):
    if images and scope_id_file is None and pair_whitelist_path is None:
        raise ValueError('--images requires --scope-id-file or --pair-whitelist; full-scope image generation is disabled')
    image_root = Path(image_output_dir or IMAGE_OUTPUT_DIR)
    if images:
        run_root = Path(RUN_ROOT).resolve()
        base_root = Path(BASE_DIR).resolve()
        data_root = Path(DATA_DIR).resolve()
        resolved = image_root.resolve()
        if run_root == base_root:
            raise ValueError('Scoped image generation requires HKBFETD_RUN_ROOT to name an isolated run root')
        if not resolved.is_relative_to(run_root) or resolved.is_relative_to(data_root):
            raise ValueError('Scoped image output must be inside the isolated run root and outside DATA_DIR')
    table = read_table(STEP5_MERGED_OUTPUT_PATH)
    geometry = pyogrio.read_dataframe(OFFICIAL_LIB_BASE_PATH,columns=[ID])
    geometry[ID] = ids(geometry[ID])
    require_unique(geometry)
    frame = geometry.merge(table[[ID,'Final_Main_Class','Final_Sub_Class']],on=ID,validate='one_to_one')
    require_unique(frame)
    scoped = scope_id_file is not None or pair_whitelist_path is not None
    candidates = generate_candidates(frame,image_root=image_root,isolated=scoped)
    scope_ids = read_scope_ids(scope_id_file) if scope_id_file is not None else None
    pair_rows = None
    if pair_whitelist_path is not None:
        pair_frame = pd.read_csv(pair_whitelist_path,dtype='string',low_memory=False)
        required = {'fragment_id','host_id','Context_SHA256'}
        if not required.issubset(pair_frame.columns):
            raise ValueError(f'Pair whitelist requires columns: {sorted(required)}')
        pair_rows = pair_frame.to_dict('records')
    candidates = filter_candidates(candidates,scope_ids=scope_ids,pair_whitelist=pair_rows)
    cache = read_cache(AI_DECISIONS_LOG_PATH,CACHE_FIELDS)
    for col in CACHE_FIELDS:
        if col not in cache:
            cache[col] = pd.NA
    cache = cache.drop_duplicates(['fragment_id','host_id','Context_SHA256'],keep='last')
    done = {(str(row.fragment_id),str(row.host_id),str(row.Context_SHA256)):row for _,row in cache.iterrows()}
    indexed = frame.set_index(ID)
    if refresh and not VISION_API_KEY:
        raise ValueError('HKBFETD_VISION_API_KEY is required for --refresh; checkpoints remain intact')
    image_attempts = 0
    if images:
        work = []
        for _,pair in candidates.iterrows():
            previous = done.get((pair.fragment_id,pair.host_id,pair.Context_SHA256))
            if previous is None or not vision_checkpoint_matches(previous,pair.Context_SHA256,{}):
                work.append(pair)
        if work:
            generated,image_attempts = run_scoped_image_jobs(pd.DataFrame(work),indexed,image_root,image_limit)
            generated = generated.set_index(['fragment_id','host_id','Context_SHA256'])
            for col in ['plain_image_path','overlay_image_path','image_context_path','Image_Generation_Status']:
                candidates[col] = [generated.loc[(r.fragment_id,r.host_id,r.Context_SHA256),col]
                                   if (r.fragment_id,r.host_id,r.Context_SHA256) in generated.index
                                   else (r[col] if col in r.index else
                                         ('sidecar_reused' if sidecar_is_valid(r,indexed) else 'skipped_matching_checkpoint'))
                                   for _,r in candidates.iterrows()]
    request_attempts, statuses = 0, []
    for _,pair in candidates.iterrows():
        key = (pair.fragment_id,pair.host_id,pair.Context_SHA256)
        previous = done.get(key)
        isolated = scoped
        if isolated:
            available = sidecar_is_valid(pair,indexed)
        else:
            available = all(Path(pair[k+'_image_path']).is_file() for k in ['plain','overlay'])
        if isolated and images:
            available = sidecar_is_valid(pair,indexed)
        hashes = {k+'_sha256':file_hash(pair[k+'_image_path']) for k in ['plain','overlay']
                  if Path(pair[k+'_image_path']).is_file()}
        same_images = vision_checkpoint_matches(previous,pair.Context_SHA256,hashes)
        if previous is not None and same_images:
            statuses.append({**pair.to_dict(),'Cache_Status':'matching_checkpoint','Image_Status':'present' if available else 'missing_but_success_checkpoint_retained',
                             'ai_decision':previous.ai_decision,'ai_reasoning':previous.ai_reasoning})
            continue
        if refresh and available and (limit is None or request_attempts<limit):
            request_attempts += 1
            try:
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
                statuses.append({**pair.to_dict(),'Cache_Status':'matching_checkpoint','Image_Status':'present','ai_decision':decision,'ai_reasoning':entry['ai_reasoning']})
                continue
            except Exception as error:
                print(f'[WARN] Visual job not checkpointed: {type(error).__name__}')
        statuses.append({**pair.to_dict(),'Cache_Status':'pending_no_matching_checkpoint','Image_Status':'present' if available else 'missing',
                         'ai_decision':None,'ai_reasoning':None})
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
        inventory=pd.DataFrame(columns=['fragment_id','host_id','ai_decision'])
    inventory['Status'] = np.where(inventory.ai_decision.notna(),'completed_legacy_unverifiable_context','legacy_request_not_completed')
    inventory.to_csv(Path(INTERMEDIATE_DIR)/'legacy_vision_resume_inventory.csv',index=False,encoding='utf-8-sig')
    print(f'[INFO] Step 6: {len(result)} current pairs; image pair attempts={image_attempts}; API request attempts={request_attempts}')
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true',help='Request decisions (requires API key; never implicit).')
    parser.add_argument('--images',action='store_true',help='Generate scoped images; requires an explicit scope file.')
    parser.add_argument('--scope-id-file',help='CSV/text file listing fragment IDs; applied before image generation.')
    parser.add_argument('--pair-whitelist',help='CSV with fragment_id, host_id, Context_SHA256; context must match.')
    parser.add_argument('--limit',type=int,default=100,help='Maximum API request attempts when --refresh is set.')
    parser.add_argument('--image-limit',type=int,default=100,help='Maximum scoped pair image-generation attempts per run, including failures.')
    parser.add_argument('--image-output-dir',help='Isolated image directory below HKBFETD_RUN_ROOT; required run root is validated.')
    args=parser.parse_args()
    main(args.refresh,args.images,args.limit,args.image_limit,args.scope_id_file,args.pair_whitelist,args.image_output_dir)
