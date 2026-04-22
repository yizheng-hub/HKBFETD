# 6.geometry_ai_arbitration.py

import os
import sys

conda_prefix = sys.prefix
proj_lib_path = os.path.join(conda_prefix, 'Library', 'share', 'proj')

if os.path.exists(proj_lib_path):
    os.environ['PROJ_LIB'] = proj_lib_path
    os.environ['PROJ_DATA'] = proj_lib_path
else:
    fallback_path = os.path.join(conda_prefix, 'share', 'proj')
    os.environ['PROJ_LIB'] = fallback_path
    os.environ['PROJ_DATA'] = fallback_path

import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import traceback
import time
import json
import base64
import re
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI
import httpx

from config import (
    INTERMEDIATE_DIR, OUTPUT_DIR, AI_CALIBRATED_OUTPUT_PATH, OFFICIAL_LIB_BASE_PATH,
    STEP5_MERGED_OUTPUT_PATH, AGGREGATED_GDF_PATH, CANDIDATES_PAIRS_PATH,
    CANDIDATES_RULES_PATH, FINAL_SUGGESTIONS_PATH, IMAGE_OUTPUT_DIR,
    CONTEXTILY_CACHE_DIR, NEIGHBOR_SEARCH_BUFFER, RUN_CANDIDATE_GENERATION,
    SKIP_IMAGE_GENERATION_CHECK, SKIP_ALL_IMAGES_IF_EXIST, BATCH_IMAGE_CHECK_SIZE,
    AI_DECISIONS_LOG_PATH, VISUALIZATION_DPI, LLM_VERIFICATION_OUTPUT,
    BASE_URL, VISION_API_KEY, VISION_CLOUD_MODEL_NAME
)

ABSOLUTE_FINAL_CSV = STEP5_MERGED_OUTPUT_PATH
ABSOLUTE_FINAL_GEOJSON = STEP5_MERGED_OUTPUT_PATH.replace('.csv', '.geojson')
AI_CALIBRATED_GEOJSON_PATH = AI_CALIBRATED_OUTPUT_PATH.replace('.csv', '.geojson')

try:
    from config import AI_PROMPT_PATH
except ImportError:
    AI_PROMPT_PATH = "../ctl/ai_calibration_prompt.txt"

MAX_THREADS = 5
print_lock = Lock()

custom_client = httpx.Client(
    trust_env=False, 
    verify=False, 
    limits=httpx.Limits(max_connections=50, max_keepalive_connections=50)
)
client = OpenAI(api_key=VISION_API_KEY, base_url=BASE_URL, http_client=custom_client)

warnings.filterwarnings('ignore')
tqdm.pandas()


def create_side_by_side_b64(plain_path, overlay_path):
    try:
        img_plain = Image.open(plain_path)
        img_overlay = Image.open(overlay_path)
        dst = Image.new('RGB', (img_plain.width + img_overlay.width, img_plain.height))
        dst.paste(img_plain, (0, 0))               
        dst.paste(img_overlay, (img_plain.width, 0)) 
        buffered = io.BytesIO()
        dst.save(buffered, format="JPEG", quality=85) 
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        try:
            if os.path.exists(plain_path): os.remove(plain_path)
            if os.path.exists(overlay_path): os.remove(overlay_path)
        except: pass
        return None

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return None

def test_api_model(model_name, prompt, b64_stitched):
    messages = [
        {"role": "system", "content": "You are a precise GIS AI. Follow rules strictly."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_stitched}"}}
        ]}
    ]
    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.1, max_tokens=300, timeout=60
        )
        reply = response.choices[0].message.content
        parsed = extract_json_from_text(reply)
        if parsed:
            return parsed.get("decision", "unknown"), parsed.get("reasoning", "")
        return "error", reply[:100]
    except Exception as e:
        return "error", str(e)

def load_prompt_template():
    if not os.path.exists(AI_PROMPT_PATH):
        default_dir = os.path.dirname(AI_PROMPT_PATH)
        if not os.path.exists(default_dir): os.makedirs(default_dir, exist_ok=True)
        default_prompt = """
Task: Arbitrate geometric fragmentation in Hong Kong. 
The image is a side-by-side view: Left is the plain satellite map, Right is the overlay where Red = unknown fragment, Green = host building.

CRITICAL RULES:
1. Podium-Tower Rule ("merge"): If the left plain image shows the red polygon is a vertical tower sitting on top of the green podium's roof, you MUST output "merge".
2. Separation Rule ("keep"): If the left plain image shows they are clearly separated independent structures with a **horizontal physical gap on the 2D plane** (e.g. shadow between them), you MUST output "keep". 
   -> FATAL WARNING: Do NOT confuse a 3D height difference (a tower sitting atop a podium) with a 2D horizontal gap! If it sits ON TOP of the podium, it's NOT a gap, it must be "merge".
3. Noise Rule ("delete"): ONLY output "delete" if the red polygon is clearly NOT a building (e.g., it is a dark ground shadow, a tiny rooftop vent, or empty road).

Context: Host Building Class is {host_class}.

Respond in strict JSON: 
{{"reasoning": "briefly analyze the left plain image vs right overlay", "decision": "keep/merge/delete", "confidence": 0.0-1.0}}
"""
        with open(AI_PROMPT_PATH, 'w', encoding='utf-8') as f:
            f.write(default_prompt)
        return default_prompt
    with open(AI_PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()

def clean_id_global(x):
    s = str(x).strip()
    return s[:-2] if s.endswith('.0') else s

def load_ai_arbitration_inputs():
    
    try:
        if os.path.exists(ABSOLUTE_FINAL_GEOJSON):
            df_rule_classified = gpd.read_file(ABSOLUTE_FINAL_GEOJSON)
            if df_rule_classified.crs is None:
                df_rule_classified = df_rule_classified.set_crs("EPSG:2326", allow_override=True)
        elif os.path.exists(ABSOLUTE_FINAL_CSV):
            df_rule_classified = pd.read_csv(ABSOLUTE_FINAL_CSV, low_memory=False)
        else:
            return None
        
        
        gdf_official_library = gpd.read_file(OFFICIAL_LIB_BASE_PATH)
        gdf_official_library = gdf_official_library.set_crs("EPSG:2326", allow_override=True)
        
        gdf_aggregated = None
        if os.path.exists(AGGREGATED_GDF_PATH):
            gdf_aggregated = gpd.read_file(AGGREGATED_GDF_PATH)
            if gdf_aggregated.crs is None:
                gdf_aggregated = gdf_aggregated.set_crs("EPSG:2326", allow_override=True)
        
        if 'BUILDINGSTRUCTUREID' in df_rule_classified.columns:
            df_rule_classified['BUILDINGSTRUCTUREID'] = df_rule_classified['BUILDINGSTRUCTUREID'].apply(clean_id_global)
        if 'BUILDINGSTRUCTUREID' in gdf_official_library.columns:
            gdf_official_library['BUILDINGSTRUCTUREID'] = gdf_official_library['BUILDINGSTRUCTUREID'].apply(clean_id_global)
        if gdf_aggregated is not None and 'BUILDINGSTRUCTUREID' in gdf_aggregated.columns:
            gdf_aggregated['BUILDINGSTRUCTUREID'] = gdf_aggregated['BUILDINGSTRUCTUREID'].apply(clean_id_global)

        return {
            'df_rule_classified': df_rule_classified,
            'gdf_official_library': gdf_official_library,
            'gdf_aggregated': gdf_aggregated
        }
        
    except FileNotFoundError as e:
        return None
    except Exception as e:
        traceback.print_exc()
        return None

def generate_candidates_and_images_final_v15(rule_classified_df, official_gdf, num_to_process=None):
    try:
        import contextily as cx
        CACHE_DIR = CONTEXTILY_CACHE_DIR
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
        cx.set_cache_dir(CACHE_DIR)
    except ImportError:
        cx = None

    
    official_gdf_processed = official_gdf.copy()
    invalid_mask = ~official_gdf_processed.geometry.is_valid
    if invalid_mask.any():
        official_gdf_processed.loc[invalid_mask, 'geometry'] = official_gdf_processed.loc[invalid_mask, 'geometry'].buffer(0)
        
    fragments = rule_classified_df[rule_classified_df['Final_Main_Class'] == '未知类别'].copy()
    hosts = rule_classified_df[~rule_classified_df['Final_Main_Class'].isin(['未知类别', '非评估类别'])].copy()
    
    if fragments.empty or hosts.empty:
        return pd.DataFrame()
        

    
    if os.path.exists(CANDIDATES_PAIRS_PATH):
        df_all_candidates = pd.read_csv(CANDIDATES_PAIRS_PATH)
    else:
        gdf_all = official_gdf_processed.set_index('BUILDINGSTRUCTUREID')
        gdf_hosts = gdf_all.loc[gdf_all.index.isin(hosts['BUILDINGSTRUCTUREID'].values)]
        all_candidate_pairs = []
        
        gdf_hosts_sindex = gdf_hosts.sindex
        
    for frag_id in tqdm(fragments['BUILDINGSTRUCTUREID'], desc="Finding all candidate pairs"):
        if frag_id not in gdf_all.index: continue
        fragment_geom = gdf_all.loc[frag_id].geometry
        if fragment_geom is None or fragment_geom.is_empty: continue
        
        search_area = fragment_geom.buffer(NEIGHBOR_SEARCH_BUFFER)
        possible_host_ilocs = gdf_hosts_sindex.query(search_area, predicate='intersects')
        
        if len(possible_host_ilocs) > 0:
            candidate_hosts = gdf_hosts.iloc[possible_host_ilocs]
            actual_intersecting_hosts = candidate_hosts[candidate_hosts.geometry.intersects(search_area)]
            for host_id in actual_intersecting_hosts.index:
                all_candidate_pairs.append({'fragment_id': frag_id, 'host_id': host_id})
        
        if not all_candidate_pairs:
            return pd.DataFrame()
            
        df_all_candidates = pd.DataFrame(all_candidate_pairs).drop_duplicates()
        df_all_candidates.to_csv(CANDIDATES_PAIRS_PATH, index=False)
    
    
    
    gdf_all = official_gdf_processed.set_index('BUILDINGSTRUCTUREID')
    
    def heuristic_rule_decision(row):
        try:
            frag_id = clean_id_global(row['fragment_id'])
            host_id = clean_id_global(row['host_id'])
            frag = gdf_all.loc[frag_id]
            host = gdf_all.loc[host_id]
            shared_len = frag.geometry.buffer(1e-4).intersection(host.geometry.buffer(1e-4)).length
            if shared_len > frag.geometry.length * 0.3: 
                return "merge"
            
            is_touching_or_very_close = frag.geometry.distance(host.geometry) < 1e-4
            if frag.geometry.area < 15 and is_touching_or_very_close: 
                return "delete"
        except Exception:
            return "keep"

    if 'rule_decision' not in df_all_candidates.columns:
        tqdm.pandas(desc="Applying heuristic rules")
        df_all_candidates['rule_decision'] = df_all_candidates.progress_apply(heuristic_rule_decision, axis=1)
    
    df_all_candidates['plain_image_path'] = df_all_candidates.apply(
        lambda r: f"{IMAGE_OUTPUT_DIR}{r['fragment_id']}_vs_{r['host_id']}_plain.png", axis=1
    )
    df_all_candidates['overlay_image_path'] = df_all_candidates.apply(
        lambda r: f"{IMAGE_OUTPUT_DIR}{r['fragment_id']}_vs_{r['host_id']}_overlay.png", axis=1
    )
    
    df_all_candidates.to_csv(CANDIDATES_RULES_PATH, index=False)

    df_candidates_to_process = df_all_candidates if num_to_process is None else df_all_candidates.head(num_to_process)

    if not os.path.exists(IMAGE_OUTPUT_DIR):
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    needs_generation = []
    
    if False: 
        sample_size = min(BATCH_IMAGE_CHECK_SIZE, len(df_candidates_to_process))
        sample = df_candidates_to_process.sample(sample_size, random_state=42)
        all_exist = all(os.path.exists(r['plain_image_path']) and os.path.exists(r['overlay_image_path']) for _, r in sample.iterrows())
        if all_exist:
            return df_candidates_to_process

    for _, pair in tqdm(df_candidates_to_process.iterrows(), total=len(df_candidates_to_process), desc="Checking image status"):
        p_path = pair['plain_image_path']
        o_path = pair['overlay_image_path']
        if not (os.path.exists(p_path) and os.path.getsize(p_path) > 0 and 
                os.path.exists(o_path) and os.path.getsize(o_path) > 0):
            needs_generation.append(pair)
            
    
    if len(needs_generation) == 0:
        return df_candidates_to_process
    
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    generated_count = 0
    crs_str = gdf_all.crs.to_string()

    def process_image(pair):
        try:
            frag_id = clean_id_global(pair['fragment_id'])
            host_id = clean_id_global(pair['host_id'])
            frag_s = gdf_all.loc[[frag_id]]
            host_s = gdf_all.loc[[host_id]]
            bounds = pd.concat([frag_s, host_s]).union_all().buffer(25).bounds
            
            if not os.path.exists(pair['plain_image_path']):
                fig = Figure(figsize=(5, 5), frameon=False)
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.axis('off')
                if cx: cx.add_basemap(ax, crs=crs_str, source=cx.providers.Esri.WorldImagery, zoom=19)
                fig.savefig(pair['plain_image_path'], bbox_inches='tight', pad_inches=0, dpi=150)
                fig.clf() 
                
            if not os.path.exists(pair['overlay_image_path']):
                fig = Figure(figsize=(5, 5), frameon=False)
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.axis('off')
                
                host_s.plot(ax=ax, color='lime', edgecolor='white', linewidth=1, alpha=0.6)
                frag_s.plot(ax=ax, color='red', edgecolor='white', linewidth=1, alpha=0.6)
                
                if cx: cx.add_basemap(ax, crs=crs_str, source=cx.providers.Esri.WorldImagery, zoom=19)
                fig.savefig(pair['overlay_image_path'], bbox_inches='tight', pad_inches=0, dpi=150)
                fig.clf()
                
            return True
        except Exception as e:
            return False

    max_threads = 12 
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_image, pair): pair for pair in needs_generation}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating images in parallel"):
            if future.result():
                generated_count += 1

    return df_candidates_to_process


def process_single_ai_task(row, host_class_map, prompt_template):
    frag_id = clean_id_global(row['fragment_id'])
    host_id = clean_id_global(row['host_id'])
    plain_img_path = row['plain_image_path']
    overlay_img_path = row['overlay_image_path']
    
    if not (os.path.exists(plain_img_path) and os.path.exists(overlay_img_path)):
        return None
        
    b64_stitched = create_side_by_side_b64(plain_img_path, overlay_img_path)
    if not b64_stitched: return None

    host_info = host_class_map.get(host_id, {})
    host_class_str = f"{host_info.get('Final_Main_Class', 'Unknown')} - {host_info.get('Final_Sub_Class', 'Unknown')}"
        
    filled_prompt = prompt_template.format(host_class=host_class_str)
    decision, reasoning = test_api_model(VISION_CLOUD_MODEL_NAME, filled_prompt, b64_stitched)
    
    decision = decision.lower()
    if 'keep' in decision: decision = 'keep'
    elif 'merge' in decision: decision = 'merge'
    elif 'delete' in decision: decision = 'delete'
    else: decision = 'error'

    with print_lock:
        pass

    if decision == 'error':
        if 'quota' in reasoning.lower() or 'balance' in reasoning.lower():
                        raise ValueError(f"API balance is insufficient. Forced stop. Details: {reasoning}")
        return None

    return {
        'fragment_id': frag_id,
        'host_id': host_id,
        'ai_decision': decision,
        'ai_reasoning': reasoning,
        'timestamp': pd.Timestamp.now().isoformat()
    }

def generate_ai_decisions_with_api(df_candidates, gdf_all, rule_classified_df):

    rule_classified_df['BUILDINGSTRUCTUREID'] = rule_classified_df['BUILDINGSTRUCTUREID'].astype(str)
    host_class_map = rule_classified_df.set_index('BUILDINGSTRUCTUREID')[['Final_Main_Class', 'Final_Sub_Class']].to_dict('index')

    if os.path.exists(AI_DECISIONS_LOG_PATH):
        df_existing_log = pd.read_csv(AI_DECISIONS_LOG_PATH)
        processed_pairs = set(zip(df_existing_log['fragment_id'].astype(str), df_existing_log['host_id'].astype(str)))
    else:
        df_existing_log = pd.DataFrame(columns=['fragment_id', 'host_id', 'ai_decision', 'ai_reasoning', 'timestamp'])
        df_existing_log.to_csv(AI_DECISIONS_LOG_PATH, index=False)
        processed_pairs = set()

    tasks = []
    for _, row in df_candidates.iterrows():
        pair_key = (str(row['fragment_id']), str(row['host_id']))
        if pair_key not in processed_pairs:
            tasks.append(row)
    

    if not tasks:
        return pd.read_csv(AI_DECISIONS_LOG_PATH)

    prompt_template = load_prompt_template()
    results_buffer = []
    save_interval = 5 

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        for row in tasks:
            futures.append(executor.submit(process_single_ai_task, row, host_class_map, prompt_template))
            
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running API inference in parallel"):
                res = future.result()
                if res:
                    results_buffer.append(res)
                    
                if len(results_buffer) >= save_interval:
                    pd.DataFrame(results_buffer).to_csv(AI_DECISIONS_LOG_PATH, mode='a', header=False, index=False)
                    results_buffer = []
        except ValueError as ve:
            if '余额不足' in str(ve):
                print(f"\n{ve}")
                if results_buffer:
                    pd.DataFrame(results_buffer).to_csv(AI_DECISIONS_LOG_PATH, mode='a', header=False, index=False)
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except:
                    pass
                sys.exit(1)
            else:
                raise ve
                
    if results_buffer:
        pd.DataFrame(results_buffer).to_csv(AI_DECISIONS_LOG_PATH, mode='a', header=False, index=False)

    return pd.read_csv(AI_DECISIONS_LOG_PATH)

def apply_ai_arbitration():
    
    try:
        df_rules = pd.read_csv(CANDIDATES_RULES_PATH)
        if os.path.exists(AI_DECISIONS_LOG_PATH):
            df_ai = pd.read_csv(AI_DECISIONS_LOG_PATH)
            df_rules['fragment_id'] = df_rules['fragment_id'].astype(str)
            df_rules['host_id'] = df_rules['host_id'].astype(str)
            df_ai['fragment_id'] = df_ai['fragment_id'].astype(str)
            df_ai['host_id'] = df_ai['host_id'].astype(str)
            
            df_ai = df_ai.drop_duplicates(subset=['fragment_id', 'host_id'], keep='last')
            df_combined = df_rules.merge(df_ai, on=['fragment_id', 'host_id'], how='left')
        else:
            df_combined = df_rules.copy()
            df_combined['ai_decision'] = None
        
        def final_arbitration_v5(row):
            rule = row.get('rule_decision', 'keep')
            ai = row.get('ai_decision', None)
            if pd.isna(ai) or ai in ['error', 'unknown', None]: return f"Rule-Only_{rule}"
            if ai == rule: return f"Consensus_{ai}"
            if rule == 'merge' and ai == 'keep': return "Final_keep_with_inheritance"
            if rule == 'keep' and ai == 'merge': return "Manual_Review_AI_Wants_Delete" 
            if ai == 'delete':
                if rule == 'merge': return "Final_delete (AI overrides Rule_Merge)"
                else: return "Manual_Review_AI_Wants_Delete"
            if rule == 'delete' and ai != 'delete': return f"Final_{ai}_(AI_overrides_Rule_Delete)"
            return f"Unknown_Conflict_AI({ai})_Rule({rule})"

        df_combined['suggestion'] = df_combined.apply(final_arbitration_v5, axis=1)
        
        def handle_multi_host_conflicts(group):
            suggestions = group['suggestion'].unique()
            has_strong = any(s for s in suggestions if 'merge' in s or 'delete' in s or 'inheritance' in s)
            has_keep = any(s for s in suggestions if 'keep' in s)
            if has_strong and has_keep:
                group['final_suggestion'] = 'Final_keep (Conflict_Override)'
                return group
            merge_suggestions = group[group['suggestion'].str.contains('merge|inheritance', case=False, na=False)]
            if len(merge_suggestions) > 1:
                group.loc[merge_suggestions.index, 'final_suggestion'] = 'Manual_Review_Multi_Merge'
                group['final_suggestion'].fillna('Manual_Review_Multi_Merge', inplace=True)
            else:
                group['final_suggestion'] = group['suggestion']
            return group
            
        tqdm.pandas(desc="Resolving multi-host conflicts")
        df_final_suggestions = df_combined.groupby('fragment_id').progress_apply(handle_multi_host_conflicts).reset_index(drop=True)

        df_final_suggestions.to_csv(FINAL_SUGGESTIONS_PATH, index=False)
        print(df_final_suggestions['final_suggestion'].value_counts().to_markdown())
        return df_final_suggestions
    except Exception as e:
        return None

def main():
    
    data = load_ai_arbitration_inputs()
    if not data: return False
    df_rules = data['df_rule_classified']
    gdf_official = data['gdf_official_library']
    
    if 'geometry' not in df_rules.columns:
        df_rules_geo = df_rules.merge(gdf_official[['BUILDINGSTRUCTUREID', 'geometry']], on='BUILDINGSTRUCTUREID', how='left')
        df_rules_geo = gpd.GeoDataFrame(df_rules_geo, geometry='geometry', crs=gdf_official.crs)
    else:
        df_rules_geo = df_rules

    candidates_df = generate_candidates_and_images_final_v15(df_rules_geo, gdf_official)
    if candidates_df is None or candidates_df.empty: return True

    gdf_all = gdf_official.set_index('BUILDINGSTRUCTUREID')
    generate_ai_decisions_with_api(candidates_df, gdf_all, df_rules)

    apply_ai_arbitration()
    
    return True

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message) 
        if '\r' not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(os.path.dirname(base_dir), "log")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, "6.geometry_ai_arbitration.txt")
    
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout 
    
    
    try:
        success = main()
        if not success: sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
